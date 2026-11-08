//! GPU tests of the image path of the Nemotron 3.5 Super VL engine (milestone M3, task VS-b): the RADIO tower and projector as the image
//! encoder, the feature splice in the prefill slices, chunked prefill over an image, the MTP head's rows for image positions, drafted ==
//! serial on an image stream, the backend's refusals, and the engine's logits over image prompts for the KL against the bf16 original.
//!
//! All `#[ignore]` and env-gated: without the variables a test prints "not run" and passes. One model per process (about 70 GiB resident
//! with the head), so run with `--test-threads=1 --ignored` under the GPU lock:
//!
//!     REXL3_NEMO_OMNI_DIR=<pack> REXL3_NEMO_ORACLE_DIR=<ref-text-1k> REXL3_NEMOTRON_VIT_REF=<dump dir> \
//!       cargo test --release -p gb-model --test nemotron_omni_vision -- --ignored --test-threads=1 --nocapture
//!
//! * `REXL3_NEMO_OMNI_DIR`: the published pack. `REXL3_NEMO_ORACLE_DIR`: the fork's dumps, whose `ids_0.u32` is the text around the images.
//! * `REXL3_NEMO_VIT_REF` (or `REXL3_NEMOTRON_VIT_REF`): the torch dumps of the tower (`ref_512x512.safetensors`, `ref_320x448.safetensors`).
//! * `REXL3_NEMO_VIS_CASES` + `REXL3_NEMO_VIS_OUT`: `image_prompts_logits_for_the_bf16_kl` (see the test).
//!
//! The images are the repository's own public test images (`testdata/vision`); nothing derived from the weights is printed but scalars.
#![allow(clippy::type_complexity, clippy::manual_repeat_n)]

mod nemotron_omni_mtp_support;

use gb_cuda::{DevBuf, Stream};
use gb_engine::{Backend, BackendError, ImageSpan, RowOutput, RowRequest};
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::{ModelOptions, NemotronBackend, NemotronRuntime};
use gb_model::families::nemotron_h_omni::model::{NemotronModel, Point, Tap};
use gb_model::families::nemotron_h_omni::mtp::{DraftAsk, Drafts, MtpMutant};
use gb_model::families::nemotron_h_omni::vision::{NemotronEncoder, NemotronImageFeatures, NemotronTower, SpliceOp};
use gb_model::rows::RowBatch;
use gb_model::serve::{FeatureBudget, ImageEncoder};
use gb_sampling::Sampling;
use gb_vision::nemotron::NemotronPreprocessing;
use gb_vision::preprocessor::ImagePreprocessor;
use nemotron_omni_mtp_support as sup;
use nemotron_omni_mtp_support::{RowCands, CMP_CANDS};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

/// `<image>`, `<img>` and `</img>` of the pack's tokenizer.
const IMG: u32 = 18;
const IMG_START: u32 = 19;
const IMG_END: u32 = 20;

fn image_dir() -> PathBuf {
    repo_root().join("testdata/vision")
}

fn env_dir(name: &str) -> Option<PathBuf> {
    let v = std::env::var_os(name).map(PathBuf::from);
    if v.is_none() {
        eprintln!("not run: {name} is not set");
    }
    v
}

// ------------------------------------------------------------------------------------------------------------- the tower, once

struct TowerShared {
    enc: NemotronEncoder,
    pre: NemotronPreprocessing,
}

// SAFETY: used only under the mutex, one thread at a time.
unsafe impl Send for TowerShared {}

/// The encoder for images of up to 1,024 tokens (4,096 patches), loaded once per process with an unbounded feature budget.
fn tower() -> Option<MutexGuard<'static, TowerShared>> {
    static T: OnceLock<Mutex<TowerShared>> = OnceLock::new();
    let dir = sup::pack_dir()?;
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    if T.get().is_none() {
        gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
        let t = std::time::Instant::now();
        let tower = NemotronTower::load(&dir, 1024).unwrap_or_else(|e| panic!("the tower: {e}"));
        eprintln!(
            "tower loaded in {:.1} s: weights {} B, scratch {} B",
            t.elapsed().as_secs_f64(),
            tower.weight_bytes(),
            tower.scratch_bytes()
        );
        let pre = NemotronPreprocessing::from_dir(&dir).unwrap_or_else(|e| panic!("preprocessor_config.json: {e}"));
        let _ = T.set(Mutex::new(TowerShared { enc: NemotronEncoder::new(tower, FeatureBudget::new(u64::MAX / 2)), pre }));
    }
    Some(T.get().unwrap().lock().unwrap_or_else(|p| p.into_inner()))
}

/// One image through the engine's own preprocessing and the tower: its features, tokens and merged grid.
struct Encoded {
    feats: Arc<NemotronImageFeatures>,
    merged: (usize, usize),
    /// The tile as the preprocessing made it: bf16 bits `[3][H][W]`, its canvas `(w, h)` and patch grid.
    pixels: Vec<u16>,
    size: (usize, usize),
    grid: (usize, usize, usize),
}

fn encode(t: &mut TowerShared, file: &str) -> Encoded {
    let bytes = std::fs::read(image_dir().join(file)).unwrap_or_else(|e| panic!("{file}: {e}"));
    encode_bytes(t, &bytes)
}

fn encode_bytes(t: &mut TowerShared, bytes: &[u8]) -> Encoded {
    let rgb = t.pre.decode(bytes).unwrap_or_else(|e| panic!("decode: {e}"));
    let p = ImagePreprocessor::preprocess(&t.pre, &rgb).unwrap_or_else(|e| panic!("preprocess: {e}"));
    encode_pre(t, p)
}

fn encode_pre(t: &mut TowerShared, p: gb_vision::Preprocessed) -> Encoded {
    let e = t.enc.encode(&p).unwrap_or_else(|e| panic!("encode: {e}"));
    let feats = e.features.downcast::<NemotronImageFeatures>().unwrap_or_else(|_| panic!("not this encoder's features"));
    assert_eq!(feats.tokens, e.tokens);
    Encoded { feats, merged: e.merged, pixels: p.patches, size: p.size, grid: p.grid }
}

/// A prompt: `lead` text tokens, an image (`<img>`, its placeholders, `</img>`), `mid` text, optionally another image, `tail` text.
struct Prompt {
    ids: Vec<u32>,
    /// `(start of the placeholder run, its features)` per image, in order.
    spans: Vec<(usize, Arc<NemotronImageFeatures>)>,
}

fn build_prompt(text: &[u32], parts: &[(usize, &Encoded)], tail: usize) -> Prompt {
    // `text` is cut into `parts.len() + 1` pieces: `lead` before the first image, then between and after
    let mut ids = Vec::new();
    let mut spans = Vec::new();
    let mut at = 0;
    for (lead, e) in parts {
        ids.extend_from_slice(&text[at..at + lead]);
        at += lead;
        ids.push(IMG_START);
        spans.push((ids.len(), e.feats.clone()));
        ids.extend(std::iter::repeat(IMG).take(e.feats.tokens));
        ids.push(IMG_END);
    }
    ids.extend_from_slice(&text[at..at + tail]);
    Prompt { ids, spans }
}

fn image_spans(p: &Prompt) -> Vec<ImageSpan> {
    p.spans
        .iter()
        .map(|(start, f)| ImageSpan {
            start: *start,
            tokens: f.tokens,
            merged: f.merged,
            features: f.clone() as Arc<dyn std::any::Any + Send + Sync>,
            key: 0,
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------------------- model-level driving

/// The splice of the slice `at..end` of a prompt (the backend's `splice_for`).
fn splice_for(spans: &[(usize, Arc<NemotronImageFeatures>)], at: usize, end: usize) -> Vec<SpliceOp> {
    spans
        .iter()
        .filter_map(|(start, f)| {
            let (lo, hi) = ((*start).max(at), (*start + f.tokens).min(end));
            (lo < hi).then(|| SpliceOp { feats: f.clone(), dst_row: lo - at, src_row: lo - start, n: hi - lo })
        })
        .collect()
}

/// `prompt` into `slot` in slices of at most `chunk` rows on the absolute grid, each with its share of the splice. Returns the greedy token
/// and the candidates after the prompt; `taps` collects every position's post-`norm_f` row; `drafting` false switches the head off for the slot.
fn prefill_img(
    m: &mut NemotronModel,
    slot: usize,
    p: &Prompt,
    chunk: usize,
    drafting: bool,
    mut taps: Option<&mut Vec<f32>>,
) -> (u32, RowCands) {
    m.end_slot(slot);
    m.begin_slot(slot).unwrap();
    if !drafting {
        m.mtp_disable_slot(slot);
    }
    let n = p.ids.len();
    let (mut at, mut first) = (0, None);
    while at < n {
        let end = (at + chunk - at % chunk).min(n);
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: at as u64, tokens: p.ids[at..end].to_vec() }]);
        let want = (end == n).then_some(b.rows - 1);
        let splice = splice_for(&p.spans, at, end);
        if let Some(c) = m.prefill_chunk_mm(&b, want, &splice).unwrap() {
            first = Some((
                c.ids[0] as u32,
                RowCands {
                    ids: c.ids[..CMP_CANDS].to_vec(),
                    val_bits: c.vals[..CMP_CANDS].iter().map(|v| v.to_bits()).collect(),
                },
            ));
        }
        if let Some(t) = taps.as_deref_mut() {
            let mut v = vec![0f32; (end - at) * 4096];
            m.core.bufs.xn.download_range(0, &mut v).unwrap();
            t.extend(v);
        }
        at = end;
    }
    first.unwrap()
}

/// `prompt` into `slot` in pieces cut at exactly `cuts` (ascending positions inside the prompt, a piece longer than 64 rows is cut again
/// at 64), each with its share of the splice: the harness's `chunked_prefill_equals_whole` cuts, over an image prompt.
fn prefill_img_cuts(m: &mut NemotronModel, slot: usize, p: &Prompt, cuts: &[usize]) -> (u32, RowCands) {
    m.end_slot(slot);
    m.begin_slot(slot).unwrap();
    let n = p.ids.len();
    let mut bounds: Vec<usize> = Vec::new();
    let mut at = 0;
    for &c in cuts.iter().chain(std::iter::once(&n)) {
        assert!(c > at && c <= n, "cuts must ascend inside the prompt");
        while c - at > 64 {
            at += 64;
            bounds.push(at);
        }
        bounds.push(c);
        at = c;
    }
    let (mut at, mut first) = (0, None);
    for end in bounds {
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: at as u64, tokens: p.ids[at..end].to_vec() }]);
        let want = (end == n).then_some(b.rows - 1);
        let splice = splice_for(&p.spans, at, end);
        if let Some(c) = m.prefill_chunk_mm(&b, want, &splice).unwrap() {
            first = Some((
                c.ids[0] as u32,
                RowCands {
                    ids: c.ids[..CMP_CANDS].to_vec(),
                    val_bits: c.vals[..CMP_CANDS].iter().map(|v| v.to_bits()).collect(),
                },
            ));
        }
        at = end;
    }
    first.unwrap()
}

/// Plain decoding after an image prompt (the head's bookkeeping off): one row per step, the candidates that produced each token.
fn serial_img(m: &mut NemotronModel, slot: usize, p: &Prompt, n_new: usize) -> sup::Gen {
    let mut taps = Vec::new();
    let (first, c0) = prefill_img(m, slot, p, 64, false, Some(&mut taps));
    let mut g = sup::Gen { tokens: p.ids.clone(), cands: vec![c0], taps: Vec::new() };
    g.tokens.push(first);
    while g.tokens.len() - p.ids.len() < n_new {
        let last = *g.tokens.last().unwrap();
        let mut rc = sup::window_cands(m, slot, &[last], g.tokens.len() - 1);
        taps.extend(m.mtp_tap_rows(1).unwrap());
        m.commit(slot, 1);
        let c = rc.remove(0);
        g.tokens.push(c.ids[0] as u32);
        g.cands.push(c);
    }
    m.flush().unwrap();
    g.taps = taps;
    g
}

/// Speculative decoding with the head at `depth` after an image prompt (`sup::spec_ex`, over `prefill_img`).
fn spec_img(m: &mut NemotronModel, slot: usize, p: &Prompt, n_new: usize, depth: usize) -> (sup::Gen, sup::Run) {
    let mut run = sup::Run::default();
    let (first, c0) = prefill_img(m, slot, p, 64, true, None);
    let mut g = sup::Gen { tokens: p.ids.clone(), cands: vec![c0], taps: Vec::new() };
    g.tokens.push(first);
    while g.tokens.len() - p.ids.len() < n_new {
        let left = n_new - (g.tokens.len() - p.ids.len());
        let max = depth.min(left - 1);
        let drafts: Vec<u32> = if max > 0 {
            match m.mtp_draft(&[DraftAsk { slot, tokens: &g.tokens, depth: max }], &[]).unwrap().remove(0) {
                Some(d) => d.ids,
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };
        if drafts.is_empty() {
            run.undrafted += 1;
        }
        let mut window = vec![*g.tokens.last().unwrap()];
        window.extend_from_slice(&drafts);
        let rows = sup::window_cands(m, slot, &window, g.tokens.len() - 1);
        let samples: Vec<u32> = rows.iter().map(|c| c.ids[0] as u32).collect();
        let acc = drafts.iter().zip(&samples).take_while(|(d, s)| d == s).count();
        m.commit(slot, acc + 1);
        for c in rows.into_iter().take(acc + 1) {
            g.tokens.push(c.ids[0] as u32);
            g.cands.push(c);
        }
        run.rounds += 1;
        run.drafted += drafts.len();
        run.accepted += acc;
    }
    m.flush().unwrap();
    g.tokens.truncate(p.ids.len() + n_new);
    g.cands.truncate(n_new);
    (g, run)
}

/// A tap that keeps the rows of the points it is asked for (f32).
struct Rows {
    want: Vec<Point>,
    got: Arc<Mutex<Vec<(Point, Vec<f32>)>>>,
}

impl Tap for Rows {
    fn tap(&mut self, point: Point, buf: &DevBuf, rows: usize, width: usize, st: &Stream) -> Result<(), ModelError> {
        if self.want.contains(&point) {
            st.sync()?;
            let mut v = vec![0f32; rows * width];
            buf.download_range(0, &mut v)?;
            self.got.lock().unwrap().push((point, v));
        }
        Ok(())
    }
}

fn bf16_rows(f: &NemotronImageFeatures) -> Vec<f32> {
    let mut bits = vec![0u16; f.tokens * 4096];
    f.buf.download_range(0, &mut bits).unwrap();
    bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
}

fn oracle_text(n: usize) -> Option<Vec<u32>> {
    let ids = sup::oracle_ids()?;
    assert!(ids.len() >= n, "the oracle's text has {} tokens", ids.len());
    Some(ids[..n].to_vec())
}

// ------------------------------------------------------------------------------------------------------------- the tower and mlp1

fn st_f32(path: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    use std::io::{Read, Seek, SeekFrom};
    let (hs, _len, header) = gb_exl3::checkpoint::read_header_checked(path).unwrap();
    let v = header.get(name).unwrap_or_else(|| panic!("{name} is not in {}", path.display()));
    let dtype = v.get("dtype").and_then(|d| d.as_str()).unwrap().to_string();
    let shape: Vec<usize> =
        v.get("shape").and_then(|s| s.as_array()).unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let off = v.get("data_offsets").and_then(|o| o.as_array()).unwrap();
    let (a, b) = (off[0].as_u64().unwrap(), off[1].as_u64().unwrap());
    let mut f = std::fs::File::open(path).unwrap();
    f.seek(SeekFrom::Start(8 + hs + a)).unwrap();
    let mut bytes = vec![0u8; (b - a) as usize];
    f.read_exact(&mut bytes).unwrap();
    let vals = match dtype.as_str() {
        "BF16" => bytes.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        "F32" => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        other => panic!("{name}: unexpected dtype {other}"),
    };
    (vals, shape)
}

fn rel(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut dmax, mut bmax, mut d2, mut b2) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dmax = dmax.max((x - y).abs());
        bmax = bmax.max(y.abs());
        d2 += (x - y) * (x - y);
        b2 += y * y;
    }
    (dmax / bmax.max(1e-30), d2.sqrt() / b2.sqrt().max(1e-30))
}

/// The encoder (upload, bf16 to fp16 widen, tower, vision_final_layernorm, pixel shuffle, mlp1, round to bf16) against the torch dumps of
/// the pack's own code on its own bf16 weights: the projector's output rel_l2 against the owner's PROPOSED gate 8e-2 (rel_max 1e-1), and
/// the whole path's allocations against the memory model.
#[test]
#[ignore]
fn the_encoder_matches_the_torch_dumps_of_the_projector() {
    // `not run: REXL3_NEMO_...` is the form scripts/gpu_gate.sh does not count as a skip
    let Some(refd) =
        std::env::var_os("REXL3_NEMO_VIT_REF").or_else(|| std::env::var_os("REXL3_NEMOTRON_VIT_REF")).map(PathBuf::from)
    else {
        eprintln!("not run: REXL3_NEMO_VIT_REF is not set (the torch dumps of tools/dump_radio_reference.py)");
        return;
    };
    let Some(mut t) = tower() else { return };
    let tw = t.enc.tower();
    assert_eq!(tw.weight_bytes(), gb_model::families::nemotron_h_omni::vision::weight_bytes());
    assert_eq!(tw.scratch_bytes(), gb_model::families::nemotron_h_omni::vision::scratch_bytes(1024));
    for (tag, h, w) in [("512x512", 512usize, 512usize), ("320x448", 320, 448)] {
        let dump = refd.join(format!("ref_{tag}.safetensors"));
        let (px, shape) = st_f32(&dump, "pixels_bf16");
        assert_eq!(shape, [1, 3, h, w]);
        // bf16-exact values: the top half of the f32
        let patches: Vec<u16> = px.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let pre = gb_vision::Preprocessed { size: (w, h), grid: (1, h / 16, w / 16), patches, patch_dim: 768, pixel_hash: 0 };
        let e = t.enc.encode(&pre).unwrap();
        let f = e.features.downcast::<NemotronImageFeatures>().unwrap();
        assert_eq!((f.tokens, e.merged), (h * w / 1024, (h / 32, w / 32)));
        let got = bf16_rows(&f);
        let (want, mshape) = st_f32(&dump, "mlp1");
        assert_eq!(mshape, [1, f.tokens, 4096]);
        let (rel_max, rel_l2) = rel(&got, &want);
        println!("{tag}: encoder features (bf16) against the dump's mlp1: rel_max {rel_max:.3e} rel_l2 {rel_l2:.3e} (proposed gates 1e-1 / 8e-2)");
        assert!(rel_max < 1.0e-1 && rel_l2 < 8.0e-2, "{tag}: rel_max {rel_max:e} rel_l2 {rel_l2:e}");
        // the same image again: the encoder is a function of its input
        let again = t.enc.encode(&pre).unwrap().features.downcast::<NemotronImageFeatures>().unwrap();
        assert_eq!(
            bf16_rows(&again).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    // an image the tower cannot run is a request error and launches nothing
    let bad = gb_vision::Preprocessed {
        size: (528, 512),
        grid: (1, 32, 33),
        patches: vec![0; 3 * 528 * 512],
        patch_dim: 768,
        pixel_hash: 0,
    };
    let e = t.enc.encode(&bad).err().expect("33 patches wide is not a runnable grid");
    assert!(matches!(&e, ModelError::Request(m) if m.contains("multiples of 32")), "{e}");
    let big = gb_vision::Preprocessed {
        size: (64 * 16, 80 * 16),
        grid: (1, 80, 64),
        patches: vec![0; 3 * 1024 * 1280],
        patch_dim: 768,
        pixel_hash: 0,
    };
    let e = t.enc.encode(&big).err().expect("5,120 patches is above this engine's 4,096");
    assert!(matches!(&e, ModelError::Request(m) if m.contains("--max-image-tokens")), "{e}");
}

/// The feature budget bounds the encodes and gives every permit back: an encode that finds it exhausted is a `Busy` (503 with Retry-After at
/// the server) and holds nothing; dropping an image's features (the sequence ended, or the request failed before it was submitted) frees it.
#[test]
#[ignore]
fn the_feature_budget_bounds_the_encodes_and_gives_permits_back() {
    let Some(dir) = sup::pack_dir() else { return };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap();
    let pre = NemotronPreprocessing::from_dir(&dir).unwrap();
    let rgb = pre.decode(&std::fs::read(image_dir().join("check_512x512.png")).unwrap()).unwrap();
    let p = ImagePreprocessor::preprocess(&pre, &rgb).unwrap();
    let tokens = p.patches.len() / 768 / 4;
    assert_eq!(tokens, 256);
    let one = gb_model::families::nemotron_h_omni::vision::feature_bytes(tokens);
    // room for exactly one image of this size
    let budget = FeatureBudget::new(one);
    let mut enc = NemotronEncoder::new(NemotronTower::load(&dir, 256).unwrap(), budget.clone());
    let first = enc.encode(&p).unwrap();
    assert_eq!((budget.free(), budget.total()), (0, one), "the permit of the first image is held by its features");
    let busy = enc.encode(&p).err().expect("a second image does not fit the budget");
    assert!(matches!(&busy, ModelError::Busy(m) if m.contains("retry")), "{busy}");
    assert_eq!(budget.free(), 0, "a refused encode holds nothing");
    drop(first);
    assert_eq!(budget.free(), one, "the features' permit came back with them");
    let again = enc.encode(&p).unwrap();
    assert_eq!(budget.free(), 0);
    drop(again);
    assert_eq!(budget.free(), one);
    println!("feature budget of {one} B (one 256-token image): the second encode is Busy, a drop gives the permit back");
}

/// The encoder's latency by image size, against the device's roofline: the median of 5 encodes (after 2 warm-ups) of a synthetic image of
/// 256, 1,024, 2,048 and 3,328 tokens, with the FLOPs of the tower (the GEMMs and the attention) and the rate they make. The fp16
/// `mma.sync` peak of the GB10 is 123-124 TFLOP/s (`tools`: `mma_peak` of `crates/cuda/tests/glm_vision.rs`).
#[test]
#[ignore]
fn the_encoders_latency_by_image_size() {
    let Some(dir) = sup::pack_dir() else { return };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap();
    let mut enc = NemotronEncoder::new(NemotronTower::load(&dir, 3328).unwrap(), FeatureBudget::new(u64::MAX / 2));
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    for (h, w) in [(512usize, 512usize), (1024, 1024), (1024, 2048), (1664, 2048)] {
        let patches: Vec<u16> = (0..3 * h * w)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // bf16 bits of a value in about [-2, 2]
                ((((s >> 40) as u32 % 4000) as f32 / 1000.0 - 2.0).to_bits() >> 16) as u16
            })
            .collect();
        let pre = gb_vision::Preprocessed { size: (w, h), grid: (1, h / 16, w / 16), patches, patch_dim: 768, pixel_hash: 0 };
        let mut ms = Vec::new();
        for i in 0..7 {
            let t0 = std::time::Instant::now();
            let e = enc.encode(&pre).unwrap();
            let d = t0.elapsed().as_secs_f64() * 1e3;
            drop(e);
            if i >= 2 {
                ms.push(d);
            }
        }
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ms[ms.len() / 2];
        let n = (h / 16 * (w / 16)) as f64;
        let tk = n + 10.0;
        let gemm = 32.0 * 2.0 * tk * 1280.0 * (3840.0 + 1280.0 + 5120.0 + 5120.0)
            + 2.0 * n * 768.0 * 1280.0
            + (n / 4.0) * (2.0 * 5120.0 * 20480.0 + 2.0 * 20480.0 * 4096.0);
        let attn = 32.0 * 4.0 * tk * tk * 1280.0;
        println!(
            "{} tokens ({} patches): {med:.1} ms (min {:.1}, max {:.1}); {:.2} TFLOP of GEMMs + {:.2} TFLOP of attention = {:.1} TFLOP/s ({:.1}% of the 124 TFLOP/s mma peak)",
            n / 4.0,
            n,
            ms[0],
            ms[ms.len() - 1],
            gemm / 1e12,
            attn / 1e12,
            (gemm + attn) / 1e12 / (med / 1e3),
            100.0 * (gemm + attn) / 1e12 / (med / 1e3) / 124.0
        );
    }
}

// ------------------------------------------------------------------------------------------------------------- the splice

/// The embedding rows of an image prompt are the features' rows exactly (bf16 widened) over the image's positions and the token embeddings
/// everywhere else, and a prompt cut at any row, inside an image included, leaves the same state, head cache and next-token candidates as
/// the 64-row slicing.
#[test]
#[ignore]
fn the_splice_is_exact_and_chunked_prefill_equals_whole_over_an_image() {
    let (Some(mut t), Some(text)) = (tower(), oracle_text(80)) else { return };
    let Some(mut g) = sup::model() else { return };
    let m: &mut NemotronModel = &mut g;
    let a = encode(&mut t, "demo_scene.png");
    let b = encode(&mut t, "check_512x512.png");
    println!(
        "images: demo_scene {} tokens ({}x{}), check_512x512 {} tokens",
        a.feats.tokens, a.merged.0, a.merged.1, b.feats.tokens
    );
    // text, an image, text, a second image, text: slices of 64 rows cut both
    let p = build_prompt(&text, &[(21, &a), (13, &b)], 17);
    println!("prompt of {} tokens, images at {:?}", p.ids.len(), p.spans.iter().map(|s| s.0).collect::<Vec<_>>());

    // -- the embedding rows, tapped
    let got = Arc::new(Mutex::new(Vec::new()));
    m.set_tap(Some(Box::new(Rows { want: vec![Point::Embed], got: got.clone() })));
    prefill_img(m, 0, &p, 64, true, None);
    let with: Vec<f32> = got.lock().unwrap().drain(..).flat_map(|(_, v)| v).collect();
    // the same prompt as plain text (the placeholders' own embeddings)
    let text_only = Prompt { ids: p.ids.clone(), spans: vec![] };
    prefill_img(m, 1, &text_only, 64, true, None);
    let without: Vec<f32> = got.lock().unwrap().drain(..).flat_map(|(_, v)| v).collect();
    m.set_tap(None);
    assert_eq!((with.len(), without.len()), (p.ids.len() * 4096, p.ids.len() * 4096));
    let (mut image_rows, mut text_rows) = (0, 0);
    for (start, f) in &p.spans {
        let rows = bf16_rows(f);
        for r in 0..f.tokens {
            let row = &with[(start + r) * 4096..(start + r + 1) * 4096];
            assert!(
                row.iter().zip(&rows[r * 4096..(r + 1) * 4096]).all(|(x, y)| x.to_bits() == y.to_bits()),
                "image row {r} at {} is not its feature row",
                start + r
            );
            assert!(
                row.iter().zip(&without[(start + r) * 4096..(start + r + 1) * 4096]).any(|(x, y)| x.to_bits() != y.to_bits()),
                "a feature row equals the placeholder's embedding"
            );
            image_rows += 1;
        }
    }
    let is_image = |i: usize| p.spans.iter().any(|(s, f)| (*s..s + f.tokens).contains(&i));
    for i in (0..p.ids.len()).filter(|&i| !is_image(i)) {
        assert!(
            with[i * 4096..(i + 1) * 4096]
                .iter()
                .zip(&without[i * 4096..(i + 1) * 4096])
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "text row {i} changed"
        );
        text_rows += 1;
    }
    println!(
        "embedding rows: {image_rows} image rows are exactly their features, {text_rows} text rows exactly their embeddings"
    );
    m.end_slot(1);

    // -- chunked equals whole, cuts inside an image included: slot 0 holds the 64-row slicing, each other slicing goes to slot 2
    let (tok0, c0) = prefill_img(m, 0, &p, 64, true, None);
    let head0 = m.mtp_kv_rows(0, 0, p.ids.len() - 1).unwrap();
    let mut checked = Vec::new();
    for chunk in [48usize, 37, 16, 7] {
        let (tok, c) = prefill_img(m, 2, &p, chunk, true, None);
        assert_eq!((tok, &c), (tok0, &c0), "chunk {chunk}: the candidates after the prompt differ");
        sup::same_trunk_state(m, 0, 2, p.ids.len()).unwrap_or_else(|e| panic!("chunk {chunk}: {e}"));
        let head = m.mtp_kv_rows(2, 0, p.ids.len() - 1).unwrap();
        assert!(head == head0, "chunk {chunk}: the head's cache differs from the 64-row slicing's");
        checked.push(chunk);
    }
    println!("chunked prefill equals whole over the image prompt: {checked:?}-row slicings against 64: next-token candidates, all 40 Mamba layers, 8 KV layers and the head's cache bit for bit");

    // -- arbitrary cuts (the harness's `chunked_prefill_equals_whole` cuts): one row before an image, on its first row, one in, its middle,
    // its last row, one past it, and a few odd ones, for both images
    let mut cuts = vec![1usize, 5, 111];
    for (start, f) in &p.spans {
        for c in
            [start - 1, *start, start + 1, start + f.tokens / 2, start + f.tokens - 1, start + f.tokens, start + f.tokens + 1]
        {
            cuts.push(c);
        }
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts.retain(|&c| c > 0 && c < p.ids.len());
    let (tok, c) = prefill_img_cuts(m, 2, &p, &cuts);
    assert_eq!((tok, &c), (tok0, &c0), "cuts {cuts:?}: the candidates after the prompt differ");
    sup::same_trunk_state(m, 0, 2, p.ids.len()).unwrap_or_else(|e| panic!("cuts {cuts:?}: {e}"));
    assert!(m.mtp_kv_rows(2, 0, p.ids.len() - 1).unwrap() == head0, "cuts {cuts:?}: the head's cache differs");
    println!(
        "cut at {} positions around and inside both images ({cuts:?}): same candidates, state and head cache bit for bit",
        cuts.len()
    );
    m.end_slot(0);
    m.end_slot(2);
}

// ------------------------------------------------------------------------------------------------------------- the head's rows

/// The head's cache over an image prompt, built chunk by chunk with the prompt, equals teacher forcing from the same taps and the same
/// features bit for bit; the control (the placeholder's embedding in the image rows) differs in exactly the image rows.
#[test]
#[ignore]
fn the_drafters_cache_over_an_image_prompt_equals_teacher_forcing() {
    let (Some(mut t), Some(text)) = (tower(), oracle_text(80)) else { return };
    let Some(mut g) = sup::model() else { return };
    let m: &mut NemotronModel = &mut g;
    assert!(m.has_mtp());
    let a = encode(&mut t, "demo_scene.png");
    let p = build_prompt(&text, &[(30, &a)], 20);
    let n = p.ids.len();
    let mut taps = Vec::new();
    prefill_img(m, 0, &p, 64, true, Some(&mut taps));
    assert_eq!(taps.len(), n * 4096);
    let built = m.mtp_kv_rows(0, 0, n - 1).unwrap();

    // teacher forcing: entry i pairs the tap of position i with the token at i + 1; the image's rows come in as features (token index - 1)
    let tf = |m: &mut NemotronModel, slot: usize, img: bool| {
        m.end_slot(slot);
        m.begin_slot(slot).unwrap();
        let ops: Vec<SpliceOp> = if img {
            p.spans.iter().map(|(s, f)| SpliceOp { feats: f.clone(), dst_row: s - 1, src_row: 0, n: f.tokens }).collect()
        } else {
            Vec::new()
        };
        m.mtp_debug_forward_mm(slot, 0, &p.ids[1..], &taps[..(n - 1) * 4096], 1, &ops).unwrap();
        m.mtp_kv_rows(slot, 0, n - 1).unwrap()
    };
    let forced = tf(m, 1, true);
    assert!(forced == built, "the head's cache over the image prompt differs from the teacher-forced one");
    let differing = |x: &(Vec<u16>, Vec<u16>), y: &(Vec<u16>, Vec<u16>)| {
        (0..n - 1)
            .filter(|&r| {
                x.0[r * 256..(r + 1) * 256] != y.0[r * 256..(r + 1) * 256]
                    || x.1[r * 256..(r + 1) * 256] != y.1[r * 256..(r + 1) * 256]
            })
            .count()
    };
    // the control: the same taps with the placeholder's embedding in the image rows
    let placeholder = tf(m, 2, false);
    let diff = differing(&placeholder, &built);
    println!(
        "head cache over an image prompt ({n} tokens, {} image rows): equals teacher forcing from the same taps and features over {} entries bit for bit; \
         the placeholder rebuild differs in {diff} entries",
        a.feats.tokens,
        n - 1
    );
    assert_eq!(diff, a.feats.tokens, "the placeholder rebuild must differ in exactly the image entries");
    // and the mutant (the production path with the features ignored) is the control through the real hook
    m.set_mtp_mutant(Some(MtpMutant::PlaceholderImages));
    prefill_img(m, 3, &p, 64, true, None);
    m.set_mtp_mutant(None);
    let mutated = m.mtp_kv_rows(3, 0, n - 1).unwrap();
    assert!(mutated == placeholder, "the PlaceholderImages mutant is the placeholder rebuild");
    assert_eq!(differing(&mutated, &built), a.feats.tokens);
    for s in 0..4 {
        m.end_slot(s);
    }
}

// ------------------------------------------------------------------------------------------------------------- drafted == serial

#[test]
#[ignore]
fn an_image_prompt_is_drafted_for_and_equals_serial() {
    let (Some(mut t), Some(text)) = (tower(), oracle_text(60)) else { return };
    let Some(mut g) = sup::model() else { return };
    let m: &mut NemotronModel = &mut g;
    m.set_mtp_mutant(None);
    let a = encode(&mut t, "check_512x512.png");
    let p = build_prompt(&text, &[(25, &a)], 20);
    let n_new = 40;
    let serial = serial_img(m, 0, &p, n_new);
    let fed = p.ids.len() + n_new - 1;
    let mut lines = Vec::new();
    for depth in [1usize, 2, 3] {
        let (spec, run) = spec_img(m, 1, &p, n_new, depth);
        assert_eq!(spec.tokens, serial.tokens, "K={depth}: the drafted token stream differs from the serial one");
        for j in 0..n_new {
            assert_eq!(spec.cands[j], serial.cands[j], "K={depth}: the candidates of generated token {j} differ");
        }
        sup::same_trunk_state(m, 0, 1, fed).unwrap_or_else(|e| panic!("K={depth}: {e}"));
        assert!(run.drafted > 0, "K={depth}: the head never drafted on the image stream");
        let tpr = (n_new - 1) as f64 / run.rounds as f64;
        lines.push(format!(
            "K={depth}: {} rounds, {} drafted, {} accepted ({tpr:.2} tokens per round); tokens, candidates and state equal serial",
            run.rounds, run.drafted, run.accepted
        ));
        m.end_slot(1);
    }
    for l in &lines {
        println!("image stream, {} prompt tokens ({} image), {l}", p.ids.len(), a.feats.tokens);
    }
    m.end_slot(0);
}

// ------------------------------------------------------------------------------------------------------------- the backend

/// A runtime over a borrowed model, so a backend can be built over the process's one loaded model.
struct Borrowed<'a>(&'a mut NemotronModel);

impl NemotronRuntime for Borrowed<'_> {
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn has_mtp(&self) -> bool {
        self.0.has_mtp()
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.0.begin_slot(slot)
    }
    fn end_slot(&mut self, slot: usize) {
        self.0.end_slot(slot)
    }
    fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<gb_model::forward::Candidates>, ModelError> {
        self.0.prefill_chunk(b, want)
    }
    fn prefill_chunk_mm(
        &mut self,
        b: &RowBatch,
        want: Option<usize>,
        splice: &[SpliceOp],
    ) -> Result<Option<gb_model::forward::Candidates>, ModelError> {
        self.0.prefill_chunk_mm(b, want, splice)
    }
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<gb_model::forward::Candidates, ModelError> {
        self.0.window_candidates(b, count)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.0.commit(slot, keep)
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        self.0.flush()
    }
    fn mtp_draft(&mut self, asks: &[DraftAsk<'_>], keyed: &[Option<Sampling>]) -> Result<Vec<Option<Drafts>>, ModelError> {
        self.0.mtp_draft(asks, keyed)
    }
    fn mtp_disable_slot(&mut self, slot: usize) {
        self.0.mtp_disable_slot(slot)
    }
    fn mtp_pending_rows(&self, slot: usize) -> Option<usize> {
        self.0.mtp_pending_rows(slot)
    }
}

fn backend<'a>(m: &'a mut NemotronModel) -> NemotronBackend<Borrowed<'a>> {
    let opts = ModelOptions {
        slots: sup::SLOTS,
        kv_pages: sup::KV_PAGES,
        max_context: sup::MAX_CONTEXT,
        prefill_rows: 64,
        layers: None,
        mtp: true,
    };
    NemotronBackend::with_runtime(Borrowed(m), 64, &opts).unwrap().with_image_token(IMG)
}

/// Prefills with the backend's slices (`prefill_slice`) and decodes `n` greedy tokens one row at a time: the last prefill row and the tokens.
fn drive(
    b: &mut NemotronBackend<Borrowed<'_>>,
    slot: usize,
    prompt: &[u32],
    images: &[ImageSpan],
    n: usize,
) -> (RowOutput, Vec<u32>) {
    let mut done = 0;
    let row = loop {
        let s = b.prefill_slice(slot, prompt, images, done, CMP_CANDS);
        done = s.done;
        if done >= prompt.len() {
            break s.row.expect("the last slice returns the row");
        }
    };
    let mut toks = vec![row.ids[0]];
    let mut pos = prompt.len();
    for _ in 0..n {
        let o = b.step(&[RowRequest { slot, tokens: vec![*toks.last().unwrap()], first_position: pos as u64 }], CMP_CANDS);
        b.commit(slot, 1);
        pos += 1;
        toks.push(o[0][0].ids[0]);
    }
    (row, toks)
}

fn err_of(b: &mut NemotronBackend<Borrowed<'_>>) -> String {
    let BackendError { message, .. } = b.take_error().expect("an error was expected");
    message
}

/// Through the backend: chunked prefill (the row limits `set_prefill_row_limit` gives) equals whole over an image prompt, a window over an
/// image span is refused by name, bad images fail only their own sequence with a named error, a text prompt before and after an image
/// prompt on one slot gives the same text as without images, and the head's absorb predicate sees the span.
#[test]
#[ignore]
fn the_backend_serves_image_prompts_and_refuses_what_it_must() {
    let (Some(mut t), Some(text)) = (tower(), oracle_text(70)) else { return };
    let Some(mut g) = sup::model() else { return };
    let a = encode(&mut t, "rings_333x517.png");
    let p = build_prompt(&text, &[(18, &a)], 22);
    let images = image_spans(&p);
    let n = p.ids.len();
    let m: &mut NemotronModel = &mut g;
    let mut b = backend(m);

    // -- chunked equals whole through the backend's own slicing
    b.set_prefill_row_limit(None);
    let (row0, toks0) = drive(&mut b, 0, &p.ids, &images, 8);
    let mut lines = Vec::new();
    for limit in [48usize, 37, 16, 5] {
        b.set_prefill_row_limit(Some(limit));
        let (row, toks) = drive(&mut b, 1, &p.ids, &images, 8);
        assert!(row.bits_eq(&row0), "row limit {limit}: the last prefill row differs");
        assert_eq!(toks, toks0, "row limit {limit}: the decoded tokens differ");
        let fed = n + 8;
        {
            let m = &mut *b.m.0;
            sup::same_trunk_state(m, 0, 1, fed).unwrap_or_else(|e| panic!("row limit {limit}: {e}"));
        }
        lines.push(limit);
        b.release(1);
    }
    b.set_prefill_row_limit(None);
    println!("backend: image prompt of {n} tokens prefilled in slices of {lines:?} rows equals the 64-row slicing: last row, 8 decoded tokens and trunk state bit for bit");
    assert!(b.take_error().is_none());

    // -- the head's absorb predicate sees the span; once the prompt is whole the generated position is text
    let (s, e) = (images[0].start, images[0].start + images[0].tokens);
    assert!(
        !b.spec_allowed_for_window(0, s + 3, 1)
            && !b.spec_allowed_for_window(0, s - 2, 4)
            && !b.spec_allowed_for_window(0, e - 1, 3)
    );
    assert!(b.spec_allowed_for_window(0, s - 4, 4) && b.spec_allowed_for_window(0, e, 3) && b.spec_allowed_for_window(0, n, 2));

    // -- a window over image rows is refused by name; the slot lives on and its next window is its own
    let o = b.step(&[RowRequest { slot: 0, tokens: vec![IMG], first_position: (s + 5) as u64 }], CMP_CANDS);
    assert!(o[0][0].values[0].is_nan());
    let msg = err_of(&mut b);
    assert!(msg.contains("image span") && msg.contains("slot 0"), "{msg}");
    println!("window over an image span: {msg}");

    // -- text before and after an image prompt on one slot: the same text as without images (state isolation)
    let plain = Prompt { ids: text[..50].to_vec(), spans: vec![] };
    let (trow, ttoks) = drive(&mut b, 2, &plain.ids, &[], 12);
    b.release(2);
    let (_, _) = drive(&mut b, 2, &p.ids, &images, 6); // the image prompt on the same slot
    let (trow2, ttoks2) = drive(&mut b, 2, &plain.ids, &[], 12); // begin resets the slot
    assert!(trow.bits_eq(&trow2) && ttoks == ttoks2, "text before and after an image prompt on one slot differ");
    b.release(2);
    println!(
        "state isolation: a 50-token text prompt gives the same row and 12 tokens before and after an image prompt on its slot"
    );

    // -- bad images fail alone: slot 3 decodes beside them and equals its undisturbed run
    let (_, undisturbed) = drive(&mut b, 4, &text[..40], &[], 10);
    b.release(4);
    let foreign: Arc<dyn std::any::Any + Send + Sync> = Arc::new(());
    let cases: Vec<(&str, Vec<u32>, Vec<ImageSpan>, &str)> = vec![
        (
            "features of another backend",
            p.ids.clone(),
            vec![ImageSpan {
                start: images[0].start,
                tokens: images[0].tokens,
                merged: images[0].merged,
                features: foreign,
                key: 1,
            }],
            "not produced by this backend",
        ),
        (
            "a row count that is not the placeholder count",
            p.ids.clone(),
            vec![ImageSpan { tokens: images[0].tokens - 1, ..images[0].clone() }],
            "encoder rows",
        ),
        (
            "placeholders that are not placeholders",
            {
                let mut q = p.ids.clone();
                q[images[0].start + 3] = 1234;
                q
            },
            images.clone(),
            "not all image placeholders",
        ),
        ("a span past the prompt", p.ids[..images[0].start + 10].to_vec(), images.clone(), "does not fit the prompt"),
    ];
    // slot 3 starts a text sequence and decodes while each bad image is submitted to slot 5
    let (row3, _) = drive(&mut b, 3, &text[..40], &[], 0);
    let mut pos3 = 40;
    let mut tok3 = vec![row3.ids[0]];
    for (what, ids, spans, needle) in cases {
        let s = b.prefill_slice(5, &ids, &spans, 0, CMP_CANDS);
        assert!(s.row.unwrap().values[0].is_nan(), "{what}");
        let msg = err_of(&mut b);
        assert!(msg.contains(needle), "{what}: {msg}");
        assert!(b.live_slots().iter().all(|&x| x != 5), "{what}: the failed slot is given up");
        // the neighbour decodes on, undisturbed
        let o = b.step(&[RowRequest { slot: 3, tokens: vec![*tok3.last().unwrap()], first_position: pos3 as u64 }], CMP_CANDS);
        b.commit(3, 1);
        pos3 += 1;
        tok3.push(o[0][0].ids[0]);
        println!("bad image ({what}): refused alone: {msg}");
    }
    assert_eq!(tok3[..], undisturbed[..tok3.len()], "slot 3 decoded differently beside the failing images");
    // and the slot that failed is clean for the next request
    let (_, again) = drive(&mut b, 5, &text[..40], &[], 4);
    assert_eq!(again[..], undisturbed[..again.len()]);
    for s in 0..6 {
        b.release(s);
    }
    assert!(b.take_error().is_none());
}

// ------------------------------------------------------------------------------------------------------------- the KL's inputs

/// The engine's side of the KL leg against the bf16 original on image prompts. `REXL3_NEMO_VIS_CASES` is a directory with `cases.json`:
/// `[{"name", "image" (a file of testdata/vision), "ids" (the processor's expanded prompt ids, `<name>.ids.u32`), "pixels" (optional: the
/// processor's pixel_values as bf16 bits, `<name>.pixels.bf16`), "n_new"}]`. For each case the image goes through the engine's own
/// preprocessing and the tower, the prompt through the 64-row slices with the splice, and the greedy answer is generated one row at a time
/// with the logits of every position that predicts an answer token tapped. Written to `REXL3_NEMO_VIS_OUT`: `<name>.answer.u32` (the
/// generated ids) and `<name>.engine_logits.f32` (`[n_new][131072]` f32, the logits that produced them). Scalars are printed.
#[test]
#[ignore]
fn image_prompts_logits_for_the_bf16_kl() {
    let (Some(cases_dir), Some(out)) = (env_dir("REXL3_NEMO_VIS_CASES"), env_dir("REXL3_NEMO_VIS_OUT")) else { return };
    let Some(mut t) = tower() else { return };
    let Some(mut g) = sup::model() else { return };
    let m: &mut NemotronModel = &mut g;
    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(cases_dir.join("cases.json")).unwrap()).unwrap();
    std::fs::create_dir_all(&out).unwrap();
    let read_u32 = |p: &Path| -> Vec<u32> {
        std::fs::read(p).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    for case in cases.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let n_new = case["n_new"].as_u64().unwrap_or(24) as usize;
        let image = case["image"].as_str().unwrap();
        let mut e = encode(&mut t, image);
        let ids = read_u32(&cases_dir.join(case["ids"].as_str().unwrap()));
        // the processor's pixel_values against the engine's own preprocessing, bit for bit. A PNG must be exact (PIL's decode is lossless);
        // a JPEG is decoded by another IDCT (libjpeg's in the processor, zune-jpeg's here) and is not: the model's numerics are then
        // judged on the PROCESSOR's pixels (the same input as the bf16 reference's), and the decoder's difference is reported on its own.
        let mut px_diff = None;
        let mut px_max = 0f32;
        if let Some(f) = case.get("pixels").and_then(|v| v.as_str()) {
            let want: Vec<u16> =
                std::fs::read(cases_dir.join(f)).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            assert_eq!(want.len(), e.pixels.len(), "{name}: the processor's tile and the engine's differ in size");
            let d = want.iter().zip(&e.pixels).filter(|(a, b)| a != b).count();
            px_max = want
                .iter()
                .zip(&e.pixels)
                .map(|(&a, &b)| (f32::from_bits((a as u32) << 16) - f32::from_bits((b as u32) << 16)).abs())
                .fold(0f32, f32::max);
            if d > 0 {
                assert!(!image.ends_with(".png"), "{name}: a PNG's pixels differ from the processor's in {d} values");
                let pre = gb_vision::Preprocessed { size: e.size, grid: e.grid, patches: want, patch_dim: 768, pixel_hash: 0 };
                e = encode_pre(&mut t, pre);
            }
            px_diff = Some(d);
        }
        // the placeholders' run in the processor's ids is the image's tokens
        let start = ids.iter().position(|&x| x == IMG).expect("an image placeholder");
        assert!(ids[start..start + e.feats.tokens].iter().all(|&x| x == IMG), "{name}: the run of placeholders");
        assert_eq!(
            ids.iter().filter(|&&x| x == IMG).count(),
            e.feats.tokens,
            "{name}: the processor's token count is the engine's"
        );
        let p = Prompt { ids, spans: vec![(start, e.feats.clone())] };

        let got = Arc::new(Mutex::new(Vec::new()));
        m.set_tap(Some(Box::new(Rows { want: vec![Point::Logits], got: got.clone() })));
        let t0 = std::time::Instant::now();
        let (first, _) = prefill_img(m, 0, &p, 64, false, None);
        let t_prefill = t0.elapsed().as_secs_f64();
        let mut answer = vec![first];
        let mut pos = p.ids.len();
        while answer.len() < n_new {
            let c = sup::window_cands(m, 0, &[*answer.last().unwrap()], pos);
            m.commit(0, 1);
            pos += 1;
            answer.push(c[0].ids[0] as u32);
        }
        m.flush().unwrap();
        m.set_tap(None);
        m.end_slot(0);
        let rows: Vec<f32> = got.lock().unwrap().drain(..).flat_map(|(_, v)| v).collect();
        assert_eq!(rows.len(), n_new * 131072, "{name}: one logits row per answer token");
        let bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{name}.engine_logits.f32")), bytes).unwrap();
        let a: Vec<u8> = answer.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(out.join(format!("{name}.answer.u32")), a).unwrap();
        // the image's features as the engine made them (bf16 bits), for the comparison with the reference's own
        let mut fbits = vec![0u16; e.feats.tokens * 4096];
        e.feats.buf.download_range(0, &mut fbits).unwrap();
        std::fs::write(
            out.join(format!("{name}.engine_features.bf16")),
            fbits.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
        )
        .unwrap();
        println!(
            "case {name}: image {} tokens ({}x{} merged), prompt {} tokens, prefill {t_prefill:.2} s, {n_new} answer tokens; engine's pixels differ from the processor's in {} values (max |d| {px_max:.3e})",
            e.feats.tokens,
            e.merged.0,
            e.merged.1,
            p.ids.len(),
            px_diff.map(|d| d.to_string()).unwrap_or_else(|| "n/a (no pixels file)".into())
        );
    }
}

/// The repository root: `REXL3_REPO_ROOT` when set (a prebuilt test binary run from an installed bin directory that carries the
/// tree's `testdata/`, e.g. on a node without this checkout), else the checkout this binary was built from.
fn repo_root() -> std::path::PathBuf {
    std::env::var_os("REXL3_REPO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}
