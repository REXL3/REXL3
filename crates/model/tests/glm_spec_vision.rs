//! Speculation over images on the whole GLM-5.3-Flash model with the NextN layer (90.5 GiB resident, one model per process): the drafter's cache is built
//! over an image prompt chunk by chunk from the image's FEATURES (the oracle's drafter does the same: `indexed_embeddings` of its prefill), the verify
//! window decides every token, so a request with an image drafts and gives exactly its serial tokens. The oracle's image dumps (`img_small`: 391 rows of
//! which 368 are image, `img_cap`: 7,985 rows of which 7,957 are image) supply the prompts and the features, so no tower is loaded.
//!
//! * `an_image_prompt_is_drafted_for_and_equals_serial`: the speculation predicate over an image's rows (no draft is proposed while the rows pending an
//!   absorb are an image's), then `img_small` at greedy and with seeded sampling, beside text streams, serially and with the NextN drafter at K = 2, 3
//!   and with the arbiter: the committed tokens equal; the acceptance of the image requests; a request with features of the wrong kind fails alone.
//! * `image_acceptance_by_depth`: `img_small` greedy to the end of the answer at K = 1..4: tokens per round against the fork's on the same prompt.
//! * `the_drafters_cache_over_an_image_prompt_equals_teacher_forcing`: the production path (`prefill_chunk_mm`, the splice of each chunk) leaves the
//!   drafter's cache, over an image prompt, bit for bit what the oracle-style rebuild from the same taps and features gives, and not what the placeholder
//!   rows would give. Exactness cannot see a drafter that ignores the features (drafts only propose); this can. `REXL3_MTP_MUTANT=placeholder_images`
//!   runs it with that bug.
//! * `an_image_prompt_past_the_dense_window_drafts_and_equals_serial`: `img_cap`, 7,985 rows (the DSA indexer selects: past its 2,048-token window),
//!   serially and with K = 3 and the arbiter.
//!
//! `#[ignore]`, env-gated (REXL3_GLM_DIR, REXL3_GLM_ORACLE_DIR), under the GPU lock.

mod glm_support;

use gb_engine::{Backend, Engine, ImageSpan, Output, ProposeRequest, Request};
use gb_model::families::glm::backend::GlmBackend;
use gb_model::families::glm::engine::engine_over;
use gb_model::families::glm::npy;
use gb_model::families::glm::spec::{Source, SpecConfig};
use gb_model::families::glm::vision_serve::GlmImageFeatures;
use gb_sampling::Sampling;
use gb_sched::{Admission, ModelMemory};
use gb_spec::SpecMode;
use glm_support::*;
use std::path::PathBuf;
use std::sync::Arc;

const SLOTS: usize = 4;
const KV_BLOCKS: usize = 64; // 16,384 tokens
const MAX_CONTEXT: usize = 8192;

fn oracle() -> Option<PathBuf> {
    match std::env::var("REXL3_GLM_ORACLE_DIR") {
        Ok(d) => Some(PathBuf::from(d)),
        Err(_) => {
            eprintln!("REXL3_GLM_ORACLE_DIR is not set: not run");
            None
        }
    }
}

fn ids_of(oracle: &std::path::Path, name: &str) -> Vec<u32> {
    npy::read(&oracle.join(name).join("ids.npy")).unwrap().to_i32().unwrap().iter().map(|&x| x as u32).collect()
}

/// An oracle image prompt: its token ids, where the image rows sit, and the oracle's features on the device.
struct Img {
    ids: Vec<u32>,
    start: usize,
    feats: Arc<GlmImageFeatures>,
}

impl Img {
    fn load(oracle: &std::path::Path, name: &str) -> Self {
        let d = oracle.join(name);
        let vm: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("vision/meta.json")).unwrap()).unwrap();
        let rows = vm["image_rows"].as_u64().unwrap() as usize;
        let g = vm["grid_thw"].as_array().unwrap();
        let merged = (g[1].as_u64().unwrap() as usize / 2, g[2].as_u64().unwrap() as usize / 2);
        let f = npy::read(&d.join("vision/features.npy")).unwrap().to_f32().unwrap();
        assert_eq!(f.len(), rows * 4096);
        let mut buf = gb_cuda::DevBuf::zeroed(rows * 4096 * 4).unwrap();
        buf.upload(&f).unwrap();
        Self {
            ids: ids_of(oracle, name),
            start: vm["image_start_position"].as_u64().unwrap() as usize,
            feats: Arc::new(GlmImageFeatures { buf, tokens: rows, merged, key: 0, permit: None }),
        }
    }

    fn span(&self) -> ImageSpan {
        ImageSpan {
            start: self.start,
            tokens: self.feats.tokens,
            merged: self.feats.merged,
            features: self.feats.clone(),
            key: 0,
        }
    }

    fn request(&self, sampling: Sampling, max_new: usize, draft: bool) -> Request {
        Request { prompt: self.ids.clone(), sampling, max_new, stop: vec![], draft, images: vec![self.span()], cache: true }
    }
}

type Eng = Engine<GlmBackend>;

fn admission() -> Admission {
    let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 1, activation_reserve: 0 };
    Admission::new((KV_BLOCKS * 256 + SLOTS) as u64, 0, mm, SLOTS).unwrap().with_max_context(MAX_CONTEXT as u64)
}

fn text(prompt: &[u32], sampling: Sampling, max_new: usize, draft: bool) -> Request {
    Request { prompt: prompt.to_vec(), sampling, max_new, stop: vec![], draft, images: vec![], cache: true }
}

fn run(e: &mut Eng, reqs: Vec<Request>) -> Vec<Output> {
    let n = reqs.len();
    for r in reqs {
        e.submit_deferred(r).unwrap();
    }
    while e.busy() {
        e.step().unwrap();
    }
    assert!(e.take_failed().is_empty(), "no request may fail");
    let outs = e.take_outputs();
    assert_eq!(outs.len(), n);
    outs
}

fn set_mode(e: &mut Eng, mode: SpecMode, depth: usize) {
    let s = e.backend.spec.as_mut().unwrap();
    s.cfg.mode = mode;
    s.cfg.mtp_depth = depth;
}

fn greedy() -> Sampling {
    Sampling { temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0, seed: 0 }
}

/// No draft is proposed while the rows the drafter would absorb are an image's, and the same slot drafts once the prompt is whole.
///
/// The first slice is the model's chunk (`chunk_rows()`: 128 rows by default, `REXL3_GLM_CHUNK_ROWS` as scripts/gate_inputs.sh sets it, 2,048).
/// At 128 it ends inside `img_small`'s image (rows 11..379): the next row to absorb, 128, is an image row, and nothing may be drafted there. A chunk
/// that holds the whole prompt (391 rows at 2,048) leaves no slice boundary inside the image, so that check has nothing to stand on and is skipped
/// by name; the predicate over the image's rows and the drafting at the first generated position are checked at every chunk.
fn the_predicate_gates_the_drafter(be: &mut GlmBackend, img: &Img) {
    let sp = greedy();
    let spans = [img.span()];
    let (t, chunk) = (img.ids.len(), chunk_rows());
    let first_slice = chunk.min(t);
    let in_image = |r: usize| r >= img.start && r < img.start + img.feats.tokens;
    let s = be.prefill_slice(0, &img.ids, &spans, 0, 8);
    assert_eq!(s.done, first_slice, "the first slice is the chunk's {chunk} rows, or the whole {t}-row prompt");
    assert_eq!(s.row.is_some(), first_slice == t, "a row is sampled exactly when the first slice ends the prompt");
    let mid_image = s.row.is_none() && in_image(s.done);
    if mid_image {
        // e.g. at 128 rows: 117 image rows absorbed (the image starts at 11); the next row to absorb, 128, is an image row
        assert!(!be.spec_allowed_for_window(0, s.done, 1), "row {} is an image row", s.done);
        let out = be.propose_batch(&[ProposeRequest { slot: 0, tokens: &img.ids[..s.done], sampling: &sp, max: 3 }]);
        assert!(out[0].as_ref().is_none_or(|d| d.is_empty()), "nothing is drafted over an image row: {out:?}");
    } else {
        eprintln!(
            "the predicate: chunk {chunk}: the first slice ({} of {t} rows) does not end inside the image (rows {}..{}): the \
             no-draft-mid-image check has no slice boundary to stand on at this chunk (skipped by name)",
            s.done,
            img.start,
            img.start + img.feats.tokens
        );
    }
    // the rest of the prompt: the slot is intact, the drafter has built its cache over the image rows, and the first generated position is text
    let mut done = s.done;
    let first = match s.row {
        Some(row) => row.ids[0],
        None => loop {
            let s = be.prefill_slice(0, &img.ids, &spans, done, 8);
            done = s.done;
            if let Some(row) = s.row {
                break row.ids[0];
            }
        },
    };
    assert!(be.take_error().is_none());
    assert!(!be.spec_allowed_for_window(0, img.start + 5, 3), "inside the image");
    assert!(be.spec_allowed_for_window(0, img.ids.len(), 4), "generated positions are text");
    let mut toks = img.ids.clone();
    toks.push(first);
    let out = be.propose_batch(&[ProposeRequest { slot: 0, tokens: &toks, sampling: &sp, max: 3 }]);
    let drafts = out[0].as_ref().expect("a slot whose prompt is whole drafts");
    assert_eq!(drafts.len(), be.spec.as_ref().unwrap().cfg.mtp_depth.min(3), "the head drafts after an image prompt: {drafts:?}");
    be.release(0);
    eprintln!(
        "the predicate (chunk {chunk}): {}, {} drafts at the first generated position",
        if mid_image {
            format!("no draft over an image row (row {first_slice} of {t})")
        } else {
            format!("the first slice is {first_slice} of {t} rows (no mid-image boundary)")
        },
        drafts.len()
    );
}

#[test]
#[ignore]
fn an_image_prompt_is_drafted_for_and_equals_serial() {
    let Some(oracle) = oracle() else { return };
    let (small, chat, code) = (Img::load(&oracle, "img_small"), ids_of(&oracle, "chat_300"), ids_of(&oracle, "code_700"));
    let Some(model) = load_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let mut be = GlmBackend::with_spec(model, SpecConfig::new(SpecMode::Mtp));
    the_predicate_gates_the_drafter(&mut be, &small);
    let mut e = engine_over(be, admission(), true);
    let samp = [
        greedy(),
        Sampling { temperature: 0.7, top_k: 40, top_p: 0.95, min_p: 0.0, seed: 1 },
        Sampling { temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0, seed: 2 },
    ];
    let group = |draft: bool| -> Vec<Request> {
        vec![
            small.request(samp[0], 64, draft),
            small.request(samp[1], 64, draft),
            text(&chat[..160], samp[0], 48, draft),
            small.request(samp[2], 48, draft),
        ]
    };
    set_mode(&mut e, SpecMode::None, 2);
    let serial = run(&mut e, group(false));
    assert!(serial.iter().all(|o| o.drafted == 0));
    let mut compared = 0;
    for (label, mode, depth) in [("MTP K=2", SpecMode::Mtp, 2usize), ("MTP K=3", SpecMode::Mtp, 3), ("auto", SpecMode::Auto, 2)] {
        set_mode(&mut e, mode, depth);
        let before = e.backend.spec.as_ref().unwrap().stats.clone();
        let outs = run(&mut e, group(true));
        for (i, (a, b)) in serial.iter().zip(&outs).enumerate() {
            assert_eq!(a.tokens, b.tokens, "stream {i}: {label} differs from the serial stream");
            compared += 1;
            if mode == SpecMode::Mtp && i != 2 {
                // an image request must really be drafted for (a drafter that quietly turned off after the image still gives the serial tokens)
                assert!(
                    b.drafted + 8 >= b.rounds * depth as u64,
                    "image stream {i}: {label}: {} drafts verified in {} rounds",
                    b.drafted,
                    b.rounds
                );
            }
        }
        let st = &e.backend.spec.as_ref().unwrap().stats;
        let img: Vec<&Output> = [&outs[0], &outs[1], &outs[3]].into_iter().collect();
        eprintln!(
            "img_small {label}: image streams {} tokens in {} rounds ({:.2} tokens a round, {} drafted, {} accepted); MTP rounds this run {}",
            img.iter().map(|o| o.tokens.len()).sum::<usize>(),
            img.iter().map(|o| o.rounds).sum::<u64>(),
            img.iter().map(|o| o.tokens.len()).sum::<usize>() as f64 / img.iter().map(|o| o.rounds).sum::<u64>() as f64,
            img.iter().map(|o| o.drafted).sum::<u64>(),
            img.iter().map(|o| o.accepted).sum::<u64>(),
            st.rounds[Source::Mtp as usize] - before.rounds[Source::Mtp as usize]
        );
    }
    // a request whose features are not this backend's fails alone, beside an image stream that drafts
    set_mode(&mut e, SpecMode::Mtp, 2);
    let mut bad = small.request(samp[0], 8, true);
    bad.images = vec![ImageSpan {
        start: small.start,
        tokens: small.feats.tokens,
        merged: small.feats.merged,
        features: Arc::new(()),
        key: 0,
    }];
    e.submit_deferred(small.request(samp[0], 64, true)).unwrap();
    e.submit_deferred(bad).unwrap();
    e.submit_deferred(text(&code[..200], samp[0], 32, true)).unwrap();
    while e.busy() {
        e.step().unwrap();
    }
    let failed = e.take_failed();
    assert_eq!(failed.len(), 1, "exactly the request with bad features fails: {failed:?}");
    assert!(failed[0].1.contains("not produced by this backend"), "{}", failed[0].1);
    let outs = e.take_outputs();
    assert_eq!(outs.len(), 2);
    assert!(
        outs.iter().any(|o| o.tokens == serial[0].tokens),
        "the image stream beside the failed request equals its serial run"
    );
    assert!(outs.iter().all(|o| o.drafted > 0), "both survivors drafted");
    // and the slots serve the next requests, image and text, with the same tokens and drafts
    set_mode(&mut e, SpecMode::Mtp, 2);
    let again = run(&mut e, vec![small.request(samp[0], 64, true), text(&chat[..160], samp[0], 48, true)]);
    assert_eq!(again[0].tokens, serial[0].tokens);
    assert_eq!(again[1].tokens, serial[2].tokens);
    assert!(again.iter().all(|o| o.drafted > 0));
    eprintln!("{compared} (stream, drafter) comparisons over image and text streams: all equal serial; a bad image failed alone");
}

/// The acceptance of the head on an image prompt by depth, greedy, to the end of the answer: `img_small` is the demo scene with "Describe this image in
/// one or two sentences", the prompt the ExLlamaV3 fork's `mtp_accept2.py` ran (`img:demo_scene.png:0`: 1.93, 2.70, 3.26 tokens a round at K = 1, 2, 3).
#[test]
#[ignore]
fn image_acceptance_by_depth() {
    let Some(oracle) = oracle() else { return };
    let small = Img::load(&oracle, "img_small");
    let Some(model) = load_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let eos = model.cfg.eos.clone();
    let mut e = engine_over(GlmBackend::with_spec(model, SpecConfig::new(SpecMode::Mtp)), admission(), true);
    let req = |draft: bool| Request { stop: eos.clone(), ..small.request(greedy(), 384, draft) };
    set_mode(&mut e, SpecMode::None, 2);
    let serial = run(&mut e, vec![req(false)]);
    let n = serial[0].tokens.len();
    for k in 1..=4usize {
        set_mode(&mut e, SpecMode::Mtp, k);
        let o = run(&mut e, vec![req(true)]);
        assert_eq!(o[0].tokens, serial[0].tokens, "K = {k}: the image answer differs from serial");
        eprintln!(
            "img_small greedy K={k}: {n} tokens in {} rounds = {:.2} tokens a round ({} drafted, {} accepted)",
            o[0].rounds,
            o[0].tokens_per_round(),
            o[0].drafted,
            o[0].accepted
        );
    }
}

#[test]
#[ignore]
fn the_drafters_cache_over_an_image_prompt_equals_teacher_forcing() {
    use gb_model::families::glm::model::MtpMutant;
    use gb_model::families::glm::vision_serve::SpliceOp;
    let Some(oracle) = oracle() else { return };
    let small = Img::load(&oracle, "img_small");
    let Some(mut m) = load_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    if std::env::var("REXL3_MTP_MUTANT").as_deref() == Ok("placeholder_images") {
        m.set_mtp_mutant(Some(MtpMutant::PlaceholderImages));
    }
    let h = m.cfg.hidden;
    let t = small.ids.len();
    let rows = small.feats.tokens;
    // the production path: the prompt in 128-row chunks, each with its share of the image's rows spliced, in slot 0; the taps of every position kept
    m.begin_slot(0).unwrap();
    let mut taps: Vec<u16> = Vec::new();
    let mut at = 0;
    while at < t {
        let end = (at + m.chunk_rows()).min(t);
        let (lo, hi) = (small.start.max(at), (small.start + rows).min(end));
        let splice = if lo < hi {
            vec![SpliceOp { feats: small.feats.clone(), dst_row: lo - at, src_row: lo - small.start, n: hi - lo }]
        } else {
            Vec::new()
        };
        let want = (end == t).then_some(1);
        let _ = m.prefill_chunk_mm(0, at, &small.ids[at..end], want, splice).unwrap();
        taps.extend(m.mtp_tap_rows(end - at).unwrap());
        at = end;
    }
    assert_eq!(taps.len(), t * h);
    // the rebuild in slot 1: entry m takes token m and the tap of position m - 1 (zeros for m = 0), the image rows from the features
    let mut hidden = vec![0u16; t * h];
    for mm in 1..t {
        hidden[mm * h..(mm + 1) * h].copy_from_slice(&taps[(mm - 1) * h..mm * h]);
    }
    m.begin_slot(1).unwrap();
    m.mtp_debug_forward_mm(1, 0, &small.ids, &hidden, 1, &[(small.feats.clone(), small.start, rows)]).unwrap();
    let (lat_prod, lat_ref) = (m.mtp_latent_rows(0, t).unwrap(), m.mtp_latent_rows(1, t).unwrap());
    let first_diff = (0..t).find(|&r| lat_prod[r * 512..(r + 1) * 512] != lat_ref[r * 512..(r + 1) * 512]);
    assert!(
        first_diff.is_none(),
        "the drafter's latent cache over the image prompt differs from the feature-based rebuild from position {first_diff:?}"
    );
    let (pool_prod, pool_ref) = (m.mtp_pool_rows(0, t / 4).unwrap(), m.mtp_pool_rows(1, t / 4).unwrap());
    assert!(pool_prod == pool_ref, "the drafter's DSA pooled keys over the image prompt differ from the rebuild's");
    // the control: the same rebuild with the placeholder in the image rows gives a different cache (the check has teeth)
    m.begin_slot(2).unwrap();
    m.mtp_debug_forward(2, 0, &small.ids, &hidden, 1).unwrap();
    let lat_ph = m.mtp_latent_rows(2, t).unwrap();
    let differing = (0..t).filter(|&r| lat_ph[r * 512..(r + 1) * 512] != lat_ref[r * 512..(r + 1) * 512]).count();
    assert!(
        differing >= rows / 2,
        "the placeholder rebuild must differ from the feature one in the image rows (differs in {differing} of {t})"
    );
    eprintln!("the drafter's cache over img_small ({t} rows, {rows} image rows) equals the feature-based rebuild bit for bit (latent rows, pooled keys); the placeholder rebuild differs in {differing} rows");
}

#[test]
#[ignore]
fn an_image_prompt_past_the_dense_window_drafts_and_equals_serial() {
    let Some(oracle) = oracle() else { return };
    let cap = Img::load(&oracle, "img_cap");
    assert!(cap.ids.len() > 7000, "img_cap is the large prompt");
    let Some(model) = load_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let mut e = engine_over(GlmBackend::with_spec(model, SpecConfig::new(SpecMode::Mtp)), admission(), true);
    set_mode(&mut e, SpecMode::None, 2);
    let serial = run(&mut e, vec![cap.request(greedy(), 40, false)]);
    for (label, mode, depth) in [("MTP K=3", SpecMode::Mtp, 3usize), ("auto", SpecMode::Auto, 2)] {
        set_mode(&mut e, mode, depth);
        let o = run(&mut e, vec![cap.request(greedy(), 40, true)]);
        assert_eq!(
            o[0].tokens,
            serial[0].tokens,
            "{label}: img_cap ({} rows, past the DSA window) differs from serial",
            cap.ids.len()
        );
        assert!(o[0].drafted > 0, "{label}: the head must draft after a long image prompt");
        eprintln!(
            "img_cap ({} rows) {label}: {} tokens in {} rounds ({:.2} tokens a round, {} drafted, {} accepted): equals serial",
            cap.ids.len(),
            o[0].tokens.len(),
            o[0].rounds,
            o[0].tokens_per_round(),
            o[0].drafted,
            o[0].accepted
        );
    }
}
