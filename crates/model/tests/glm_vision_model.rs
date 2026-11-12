//! The vision tower and the splice on the whole GLM-5.3-Flash model: ONE session (one 87 GiB load per process, under the GPU lock, refusing under 100 GiB
//! `MemAvailable` and aborting under 5), in this order:
//!
//! 1. the five text prompts through the model with NO tower in memory, dumped (forced routing) to `<out>/text_a`;
//! 2. the tower is loaded beside the model;
//! 3. the oracle's two image prompts (`img_small`: 391 rows, 368 of them image; `img_cap`: 7,985 rows, 7,957 of them image) dumped with the image rows
//!    spliced: forced routing with the tower's features (`img_forced`), forced routing with the ORACLE's features (`img_oracle`, the language model's splice
//!    alone), natural routing with the tower's (`img_natural`) and with the oracle's (`img_oracle_natural`, natural routing's own noise without the tower's),
//!    for `tools/compare_glm_dump.py ORACLE CANDIDATE --skip mtp,kda_conv`;
//! 4. the five text prompts again, dumped to `<out>/text_b`, and every file of `text_a` and `text_b` compared by SHA-256: a text request is bit-identical with
//!    the tower loaded and with it not (and, with `REXL3_GLM_BASE_DIR`, with the dumps of the text-only build this branch started from);
//! 5. the splice itself: a prompt with TWO images and text between them, the embedding rows captured: image rows are exactly the features, every other row is
//!    exactly the token's embedding;
//! 6. the backend as the engine drives it: an image prompt prefilled slice by slice between the decode steps of another slot (that slot's bits do not move; the
//!    image prompt's candidates equal the dump path's and an undisturbed run's), a bad image failing only its own sequence, text after images in a slot.
//!
//! `REXL3_GLM_DIR=... REXL3_GLM_ORACLE_DIR=~/rexl3-ab/glm53-oracle/final4-A REXL3_GLM_VISION_DIR=~/rexl3-ab/glm53-vision/ref REXL3_GLM_DUMP_OUT=... \
//!  cargo test --release -p gb-model --test glm_vision_model -- --ignored --nocapture --test-threads=1`

mod glm_support;

use gb_cuda::{DevBuf, Stream};
use gb_engine::{Backend, ImageSpan, RowOutput, RowRequest};
use gb_model::families::glm::backend::GlmBackend;
use gb_model::families::glm::dump::{dump_prompt, dump_prompt_mm, DumpImage};
use gb_model::families::glm::model::{ForcedRouting, GlmModel, Head, Mode};
use gb_model::families::glm::npy;
use gb_model::families::glm::vision::{GlmVisionTower, TowerOptions};
use gb_model::families::glm::vision_serve::{GlmImageFeatures, SpliceOp};
use gb_model::families::glm::GlmFamily;
use gb_model::rows::RowBatch;
use glm_support::*;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const SLOTS: usize = 4;
const KV_BLOCKS: usize = 80; // 20,480 tokens
const MAX_CONTEXT: usize = 8192;
const TEXT: &str = "one_token,short_20,chat_300,code_700,long_2300";

fn env_dir(name: &str) -> Option<PathBuf> {
    match std::env::var(name) {
        Ok(d) if Path::new(&d).is_dir() => Some(PathBuf::from(d)),
        Ok(d) => panic!("{name}={d} is not a directory"),
        Err(_) => {
            eprintln!("{name} is not set: not run");
            None
        }
    }
}

fn ids_of(oracle: &Path, prompt: &str) -> Vec<u32> {
    npy::read(&oracle.join(prompt).join("ids.npy")).unwrap().to_i32().unwrap().iter().map(|&x| x as u32).collect()
}

fn forced_of(oracle: &Path, prompt: &str) -> ForcedRouting {
    let d = oracle.join(prompt);
    let ids = npy::read(&d.join("route_ids.npy")).unwrap();
    let w = npy::read(&d.join("route_w.npy")).unwrap();
    let lay = npy::read(&d.join("route_layers.npy")).unwrap().to_i32().unwrap();
    ForcedRouting {
        layers: lay.iter().map(|&l| l as usize).collect(),
        t: ids.shape[1],
        ids: ids.to_i32().unwrap(),
        w: w.to_f32().unwrap().iter().map(|&x| gb_exl3::f16::f64_to_f16(x as f64)).collect(),
    }
}

/// SHA-256 of every file under `dir` except `meta.json` (its `seconds` differ run to run), by relative path.
fn hashes(dir: &Path) -> std::collections::BTreeMap<String, String> {
    let mut out = std::collections::BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().unwrap() != "meta.json" {
                let h: String = Sha256::digest(std::fs::read(&p).unwrap()).iter().map(|b| format!("{b:02x}")).collect();
                out.insert(p.strip_prefix(dir).unwrap().display().to_string(), h);
            }
        }
    }
    out
}

fn dump_text(m: &mut GlmModel, oracle: &Path, out: &Path) {
    for p in TEXT.split(',') {
        m.forced = Some(forced_of(oracle, p));
        let d = dump_prompt(m, p, "text", &ids_of(oracle, p), out).unwrap();
        eprintln!("  {p}: T={} greedy {} ({:.1} s)", d.tokens, d.greedy_next_token, d.seconds);
    }
    m.forced = None;
}

struct Img {
    start: usize,
    rows: usize,
    grid: (usize, usize),
    still: Vec<u16>,
}

fn image_of(oracle: &Path, prompt: &str) -> Img {
    let vd = oracle.join(prompt).join("vision");
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(vd.join("meta.json")).unwrap()).unwrap();
    let g = meta["grid_thw"].as_array().unwrap();
    let (gh, gw) = (g[1].as_u64().unwrap() as usize, g[2].as_u64().unwrap() as usize);
    let patches = npy::read(&vd.join("patches.npy")).unwrap().to_u16().unwrap();
    let pre = gb_vision::Preprocessed { size: (gw * 14, gh * 14), grid: (1, gh, gw), patches, patch_dim: 1176, pixel_hash: 0 };
    Img {
        start: meta["image_start_position"].as_u64().unwrap() as usize,
        rows: meta["image_rows"].as_u64().unwrap() as usize,
        grid: (gh, gw),
        still: gb_vision::glm::still_patches(&pre, 2),
    }
}

fn encode(t: &mut GlmVisionTower, im: &Img) -> Arc<GlmImageFeatures> {
    let st = Stream::new().unwrap();
    let mut buf = DevBuf::zeroed(im.rows * 4096 * 4).unwrap();
    t.encode(&im.still, im.grid, &mut buf, &st).unwrap();
    st.sync().unwrap();
    Arc::new(GlmImageFeatures { buf, tokens: im.rows, merged: (im.grid.0 / 2, im.grid.1 / 2), key: 0, permit: None })
}

fn oracle_features(oracle: &Path, prompt: &str, im: &Img) -> Arc<GlmImageFeatures> {
    let f = npy::read(&oracle.join(prompt).join("vision").join("features.npy")).unwrap().to_f32().unwrap();
    let mut buf = DevBuf::zeroed(im.rows * 4096 * 4).unwrap();
    buf.upload(&f).unwrap();
    Arc::new(GlmImageFeatures { buf, tokens: im.rows, merged: (im.grid.0 / 2, im.grid.1 / 2), key: 0, permit: None })
}

fn span(start: usize, f: &Arc<GlmImageFeatures>) -> ImageSpan {
    ImageSpan { start, tokens: f.tokens, merged: f.merged, features: f.clone(), key: 0 }
}

/// Prefill `prompt` into `slot` slice by slice and return the candidates after it.
fn prefill(be: &mut GlmBackend, slot: usize, prompt: &[u32], images: &[ImageSpan]) -> RowOutput {
    let mut done = 0;
    loop {
        let step = be.prefill_slice(slot, prompt, images, done, 8);
        assert!(step.done > done, "a slice must advance");
        done = step.done;
        if let Some(row) = step.row {
            assert_eq!(done, prompt.len());
            return row;
        }
    }
}

fn same(a: &RowOutput, b: &RowOutput, what: &str) {
    assert_eq!(a.ids, b.ids, "{what}: ids");
    assert!(a.values.iter().zip(&b.values).all(|(x, y)| x.to_bits() == y.to_bits()), "{what}: values");
}

#[test]
#[ignore]
fn the_vision_session() {
    let (Some(oracle), Some(vdump), Some(out)) =
        (env_dir("REXL3_GLM_ORACLE_DIR"), env_dir("REXL3_GLM_VISION_DIR"), env_dir("REXL3_GLM_DUMP_OUT"))
    else {
        return;
    };
    let Some(model) = load_model(SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let mut be = GlmBackend::new(model);

    eprintln!("== 1. text prompts, no tower in memory");
    dump_text(&mut be.m, &oracle, &out.join("text_a"));

    eprintln!("== 2. the tower beside the model");
    let dir = pack_dir().unwrap();
    let pack = GlmFamily::map_pack(&dir).unwrap();
    let before = mem_available();
    let mut tower = GlmVisionTower::load(&pack, &dir, &TowerOptions::new(32_000, 8 * GIB)).unwrap();
    eprintln!(
        "  tower: {:.3} GiB weights + {:.3} GiB scratch, boot alloc {:.2}s read {:.2}s prepare {:.2}s scratch {:.2}s; MemAvailable {:.1} -> {:.1} GiB",
        gib(tower.weight_bytes()),
        gib(tower.scratch_bytes()),
        tower.boot.alloc_s,
        tower.boot.load_s,
        tower.boot.prepare_s,
        tower.boot.scratch_s,
        gib(before),
        gib(mem_available())
    );

    eprintln!("== 3. the oracle's image prompts, image rows spliced");
    let prompts = ["img_small", "img_cap"];
    for p in prompts {
        let im = image_of(&oracle, p);
        let ids = ids_of(&oracle, p);
        let t0 = std::time::Instant::now();
        let f_tower = encode(&mut tower, &im);
        let enc_s = t0.elapsed().as_secs_f64();
        let f_oracle = oracle_features(&oracle, p, &im);
        for (label, feats, forced) in [
            ("img_forced", &f_tower, true),
            ("img_oracle", &f_oracle, true),
            ("img_natural", &f_tower, false),
            ("img_oracle_natural", &f_oracle, false),
        ] {
            be.m.forced = forced.then(|| forced_of(&oracle, p));
            let d = dump_prompt_mm(
                &mut be.m,
                p,
                "image",
                &ids,
                &[DumpImage { start: im.start, feats: feats.clone() }],
                &out.join(label),
            )
            .unwrap();
            eprintln!(
                "  {label}/{p}: T={} greedy {} ({:.1} s; tower encode {:.3} s)",
                d.tokens, d.greedy_next_token, d.seconds, enc_s
            );
        }
        be.m.forced = None;
    }

    eprintln!("== 4. the text prompts again, with the tower loaded");
    dump_text(&mut be.m, &oracle, &out.join("text_b"));
    let (a, b) = (hashes(&out.join("text_a")), hashes(&out.join("text_b")));
    assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>(), "the two text runs wrote different files");
    let differing: Vec<&String> = a.keys().filter(|k| a[*k] != b[*k]).collect();
    assert!(differing.is_empty(), "text dumps differ with the tower loaded: {differing:?}");
    eprintln!("  {} files of the five text prompts: SHA-256 identical with the tower loaded and with it not", a.len());
    if let Some(base) = std::env::var("REXL3_GLM_BASE_DIR").ok().map(PathBuf::from) {
        let c = hashes(&base);
        let common: Vec<&String> = a.keys().filter(|k| c.contains_key(*k)).collect();
        let bad: Vec<&&String> = common.iter().filter(|k| a[**k] != c[**k]).collect();
        eprintln!("  against {}: {} common files, {} differ", base.display(), common.len(), bad.len());
        assert!(bad.is_empty(), "the text path moved against the base build's dumps: {:?}", &bad[..bad.len().min(8)]);
    }

    eprintln!("== 5. the splice: two images and text between, the embedding rows captured");
    let small = image_of(&oracle, "img_small");
    let f_small = encode(&mut tower, &small);
    let (pl_ids, pl_tok) = (ids_of(&oracle, "img_small"), ids_of(&oracle, "img_small")[small.start]);
    // a second image: the 640x448 demo scene's tower features again would hide a wrong row order, so use the photo-like fixture
    let photo = {
        let g = (28usize, 38usize);
        let patches = npy::read(&vdump.join("photo_like").join("patches.npy")).unwrap().to_u16().unwrap();
        let pre =
            gb_vision::Preprocessed { size: (g.1 * 14, g.0 * 14), grid: (1, g.0, g.1), patches, patch_dim: 1176, pixel_hash: 0 };
        Img { start: 0, rows: g.0 * g.1 / 4, grid: g, still: gb_vision::glm::still_patches(&pre, 2) }
    };
    let f_photo = encode(&mut tower, &photo);
    let tail = &pl_ids[small.start + small.rows..];
    let mut two: Vec<u32> = pl_ids[..small.start].to_vec();
    let s1 = two.len();
    two.extend(std::iter::repeat_n(pl_tok, small.rows));
    two.extend_from_slice(&tail[..9]);
    let s2 = two.len();
    two.extend(std::iter::repeat_n(pl_tok, photo.rows));
    two.extend_from_slice(&tail[..5]);
    be.m.capture = Some(gb_model::families::glm::model::Capture::default());
    be.m.begin_slot(0).unwrap();
    let mut at = 0;
    while at < two.len() {
        let end = (at + 128).min(two.len());
        let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: two[at..end].to_vec(), first_position: at as u64 }]);
        let splice: Vec<SpliceOp> = [(s1, &f_small), (s2, &f_photo)]
            .iter()
            .filter_map(|(start, f)| {
                let (lo, hi) = ((*start).max(at), (*start + f.tokens).min(end));
                (lo < hi).then(|| SpliceOp { feats: (*f).clone(), dst_row: lo - at, src_row: lo - *start, n: hi - lo })
            })
            .collect();
        be.m.forward_mm(&b, Mode::Prefill, Head::None, splice).unwrap();
        at = end;
    }
    be.m.end_slot(0);
    let cap = be.m.capture.take().unwrap();
    let embed = &cap.embed;
    let feats_small: Vec<f32> = f_small.buf.to_vec().unwrap();
    let feats_photo: Vec<f32> = f_photo.buf.to_vec().unwrap();
    // the text rows' embeddings: the same prompt as a text-only pass (the placeholders embed as the placeholder token)
    let text_embed: Vec<f32> = {
        be.m.capture = Some(gb_model::families::glm::model::Capture::default());
        be.m.begin_slot(1).unwrap();
        let mut at = 0;
        while at < two.len() {
            let end = (at + 128).min(two.len());
            be.m.prefill_chunk(1, at, &two[at..end], None).unwrap();
            at = end;
        }
        be.m.end_slot(1);
        be.m.capture.take().unwrap().embed
    };
    let h = 4096;
    let mut checked = (0, 0);
    for r in 0..two.len() {
        let row = &embed[r * h..(r + 1) * h];
        let want_img = if (s1..s1 + small.rows).contains(&r) {
            Some(&feats_small[(r - s1) * h..(r - s1 + 1) * h])
        } else if (s2..s2 + photo.rows).contains(&r) {
            Some(&feats_photo[(r - s2) * h..(r - s2 + 1) * h])
        } else {
            None
        };
        match want_img {
            Some(w) => {
                assert!(row.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()), "image row {r} is not its feature row");
                checked.0 += 1;
            }
            None => {
                // a text row: the token's embedding, as a text-only pass holds it
                let w = &text_embed[r * h..(r + 1) * h];
                assert!(row.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()), "text row {r} changed");
                checked.1 += 1;
            }
        }
    }
    eprintln!("  prompt of {} tokens, images at {s1} ({} rows, straddling chunks) and {s2} ({} rows): {} image rows are exactly their features, {} text rows are exactly their embeddings", two.len(), small.rows, photo.rows, checked.0, checked.1);

    eprintln!("== 6. the backend as the engine drives it");
    backend_checks(&mut be, &oracle, &small, &f_small, &f_photo, &photo, &two, (s1, s2), &out);
}

#[allow(clippy::too_many_arguments)]
fn backend_checks(
    be: &mut GlmBackend,
    oracle: &Path,
    small: &Img,
    f_small: &Arc<GlmImageFeatures>,
    f_photo: &Arc<GlmImageFeatures>,
    photo: &Img,
    two: &[u32],
    (s1, s2): (usize, usize),
    out: &Path,
) {
    let _ = photo;
    let ids = ids_of(oracle, "img_small");
    let text = ids_of(oracle, "code_700");
    // (a) the image prompt through the backend (prefill_slice, free routing) gives the candidates of the dump path's free-routing run (last row, top 8)
    let first = prefill(be, 1, &ids, &[span(small.start, f_small)]);
    assert!(be.take_error().is_none());
    let tid = npy::read(&out.join("img_natural").join("img_small").join("top64_ids.npy")).unwrap().to_i32().unwrap();
    let tvl = npy::read(&out.join("img_natural").join("img_small").join("top64_vals.npy")).unwrap().to_f32().unwrap();
    let last = ids.len() - 1;
    assert_eq!(
        first.ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
        tid[last * 64..last * 64 + 8].to_vec(),
        "backend and dump path: top-8 ids"
    );
    let dv: Vec<f32> = tvl[last * 64..last * 64 + 8].to_vec();
    assert!(first.values.iter().zip(&dv).all(|(a, b)| a.to_bits() == b.to_bits()), "backend and dump path: top-8 values");
    be.release(1);
    eprintln!("  (a) img_small through prefill_slice: top-8 equals the dump path's");
    // (b) an image prompt in slot 0 prefilled slice by slice BETWEEN the decode steps of text in slot 1: slot 1's bits do not move, slot 0's candidates
    // equal an undisturbed image prefill's, and a two-image prompt works through the same path
    let tp = &text[..90];
    let r1 = prefill(be, 1, tp, &[]);
    let mut ref_rows = vec![r1];
    for i in 0..8 {
        let tok = ref_rows.last().unwrap().ids[0];
        let o = be.step(&[RowRequest { slot: 1, tokens: vec![tok], first_position: (tp.len() + i) as u64 }], 8);
        be.commit(1, 1);
        ref_rows.push(o[0][0].clone());
    }
    be.release(1);
    let undisturbed = prefill(be, 0, &ids, &[span(small.start, f_small)]);
    be.release(0);
    let r1 = prefill(be, 1, tp, &[]);
    let mut rows = vec![r1];
    let images = [span(small.start, f_small)];
    let (mut done, mut slices, mut final_row) = (0usize, 0usize, None);
    for i in 0..8 {
        // one slice of the image prompt, then a decode step of slot 1
        if final_row.is_none() {
            let s = be.prefill_slice(0, &ids, &images, done, 8);
            done = s.done;
            slices += 1;
            final_row = s.row;
        }
        let tok = rows.last().unwrap().ids[0];
        let o = be.step(&[RowRequest { slot: 1, tokens: vec![tok], first_position: (tp.len() + i) as u64 }], 8);
        be.commit(1, 1);
        rows.push(o[0][0].clone());
    }
    while final_row.is_none() {
        let s = be.prefill_slice(0, &ids, &images, done, 8);
        done = s.done;
        slices += 1;
        final_row = s.row;
    }
    assert!(be.take_error().is_none());
    for (i, (a, b)) in ref_rows.iter().zip(&rows).enumerate() {
        same(a, b, &format!("slot 1 step {i} while an image prefills in slot 0"));
    }
    same(&undisturbed, final_row.as_ref().unwrap(), "the image prompt prefilled between decode steps");
    eprintln!("  (b) an image prompt in {slices} slices between 8 decode steps of text in another slot: both bit-identical to their undisturbed runs");
    // slot 0 decodes on after the image prompt (positions continue at the prompt's length, no rope delta to carry)
    let tok = final_row.as_ref().unwrap().ids[0];
    let o0 = be.step(&[RowRequest { slot: 0, tokens: vec![tok], first_position: ids.len() as u64 }], 8);
    be.commit(0, 1);
    assert!(be.take_error().is_none() && o0[0][0].values.iter().all(|v| v.is_finite()));
    be.release(0);
    be.release(1);
    // (c) a bad image fails only its own sequence: wrong rows, features of another backend, a span that does not hold placeholders
    let r1 = prefill(be, 1, tp, &[]);
    let mut rows2 = vec![r1];
    for i in 0..8 {
        let bad = match i % 4 {
            0 => ImageSpan { start: small.start, tokens: small.rows, merged: f_small.merged, features: Arc::new(()), key: 0 },
            1 => ImageSpan {
                start: small.start,
                tokens: small.rows + 1,
                merged: (f_small.merged.0, f_small.merged.1 + 1),
                features: f_small.clone(),
                key: 0,
            },
            2 => ImageSpan { start: 5, tokens: small.rows, merged: f_small.merged, features: f_small.clone(), key: 0 }, // 5.. holds text, not placeholders
            _ => {
                ImageSpan { start: ids.len() - 10, tokens: small.rows, merged: f_small.merged, features: f_small.clone(), key: 0 }
            } // past the prompt
        };
        let s = be.prefill_slice(0, &ids, &[bad], 0, 8);
        let row = s.row.expect("a failed prefill answers with a row of NaN");
        assert!(row.values.iter().all(|v| v.is_nan()));
        let e = be.take_error().expect("the failure is recorded");
        assert!(!e.fatal, "{e}");
        if i < 4 {
            eprintln!("  (c) bad image {i}: {e}");
        }
        let tok = rows2.last().unwrap().ids[0];
        let o = be.step(&[RowRequest { slot: 1, tokens: vec![tok], first_position: (tp.len() + i) as u64 }], 8);
        be.commit(1, 1);
        rows2.push(o[0][0].clone());
    }
    for (i, (a, b)) in ref_rows.iter().zip(&rows2).enumerate() {
        same(a, b, &format!("slot 1 step {i} while bad images fail in slot 0"));
    }
    be.release(1);
    // the freed slot serves the next (good) image prompt with the same bits
    let again = prefill(be, 0, &ids, &[span(small.start, f_small)]);
    same(&undisturbed, &again, "slot 0 after bad images");
    be.release(0);
    eprintln!("  (c) 8 bad images (wrong features, wrong rows, no placeholders, past the prompt) failed only their sequences; slot 1 and the freed slot 0 kept their bits");
    // (d) two images in one prompt through the backend equal the model-level splice, bit for bit, and the spec hook sees the spans
    let both = [span(s1, f_small), span(s2, f_photo)];
    let w1 = prefill(be, 2, two, &both);
    be.release(2);
    let w2 = prefill(be, 3, two, &both);
    same(&w1, &w2, "the two-image prompt in two slots");
    assert!(!be.spec_allowed_for_window(3, s1 + 3, 4), "speculation must not run over an image row");
    assert!(!be.spec_allowed_for_window(3, s1 - 2, 4), "a window that starts before an image and ends inside it touches it");
    assert!(!be.spec_allowed_for_window(3, s2 + 1, 1));
    assert!(be.spec_allowed_for_window(3, 0, s1), "the text before the first image is free");
    assert!(be.spec_allowed_for_window(3, s1 + small.rows, 5), "the text between the images is free");
    assert!(be.spec_allowed_for_window(3, two.len(), 8), "decode positions after the prompt are free");
    be.release(3);
    assert!(be.spec_allowed_for_window(3, s1 + 3, 4), "a released slot has no images");
    eprintln!("  (d) a prompt with two images: two slots, same bits; the speculation predicate sees both spans");
}
