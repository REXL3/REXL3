//! GPU tests of the video path of the Nemotron 3.5 Super VL engine (milestone M4, task M4G): the tower as the video encoder (tubelets batched
//! through the blocks, EVS rows gathered on the host), a video's rows spliced into the prefill slices (chunked == whole with cuts inside a
//! tubelet's run, a pruned prompt's kept rows at their positions), the encoder's budget and its latency, and the engine's logits over video
//! prompts for the KL against the bf16 original.
//!
//! All `#[ignore]` and env-gated: without the variables a test prints "not run" and passes. One model per process (about 70 GiB resident
//! with the head), so run with `--test-threads=1 --ignored` under the GPU lock:
//!
//!     REXL3_NEMO_OMNI_DIR=<pack> REXL3_NEMO_ORACLE_DIR=<ref-text-1k> \
//!       cargo test --release -p gb-model --test nemotron_omni_video_gpu -- --ignored --test-threads=1 --nocapture
//!
//! * `REXL3_NEMO_OMNI_DIR`: the published pack. `REXL3_NEMO_ORACLE_DIR`: the fork's dumps, whose `ids_0.u32` is the text around the video.
//! * `REXL3_NEMO_VID_CASES` + `REXL3_NEMO_VID_OUT`: `video_prompts_logits_for_the_bf16_kl` (see the test).
//!
//! The frames are crops of the repository's own public test images (`testdata/vision`); nothing derived from the weights is printed but scalars.
#![allow(clippy::type_complexity, dead_code)] // `wide_clip` is a clip kept for a manual run
mod nemotron_omni_mtp_support;

use gb_cuda::{DevBuf, Stream};
use gb_engine::RowRequest;
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::model::{NemotronModel, Point, Tap};
use gb_model::families::nemotron_h_omni::vision::{
    feature_bytes, NemotronEncoder, NemotronImageFeatures, NemotronTower, SpliceOp,
};
use gb_model::rows::RowBatch;
use gb_model::serve::{EncodedImage, FeatureBudget, ImageEncoder, VideoInput};
use gb_vision::nemotron::NemotronPreprocessing;
use gb_vision::nemotron_video::{retention_mask_bits, PreprocessedVideo, SumOrder, VideoBuilder, FEATURE_WIDTH};
use gb_vision::preprocessor::ImagePreprocessor;
use gb_vision::Rgb;
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

/// The encoder at the default cap (3,328 tokens: 13,312 patches, 12 tubelets of 1,024 patches at once), loaded once per process with an
/// unbounded feature budget.
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
        let tower = NemotronTower::load(&dir, 3328).unwrap_or_else(|e| panic!("the tower: {e}"));
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

// ------------------------------------------------------------------------------------------------------------- frames and videos

/// A video of the tests: a public image and the crops that are its frames `(x, y, w, h)`.
struct Clip {
    file: &'static str,
    crops: Vec<(usize, usize, usize, usize)>,
}

fn crop(img: &Rgb, x: usize, y: usize, w: usize, h: usize) -> Rgb {
    let mut data = Vec::with_capacity(w * h * 3);
    for row in y..y + h {
        data.extend_from_slice(&img.data[(row * img.width + x) * 3..(row * img.width + x + w) * 3]);
    }
    Rgb { width: w, height: h, data }
}

/// The clip's frames through the engine's own preprocessing (the processor's video rule, bf16).
fn preprocess(t: &TowerShared, clip: &Clip) -> PreprocessedVideo {
    let bytes = std::fs::read(image_dir().join(clip.file)).unwrap_or_else(|e| panic!("{}: {e}", clip.file));
    let rgb = t.pre.decode(&bytes).unwrap_or_else(|e| panic!("decode: {e}"));
    let mut b = VideoBuilder::new(&t.pre);
    for &(x, y, w, h) in &clip.crops {
        b.push(&crop(&rgb, x, y, w, h)).unwrap_or_else(|e| panic!("frame: {e}"));
    }
    b.finish().unwrap()
}

fn square_clip(frames: usize) -> Clip {
    // 400 x 400 crops of the 640 x 448 scene, drifting 16 px a frame
    Clip { file: "demo_scene.png", crops: (0..frames).map(|i| (16 * i, 8 * i, 400, 400)).collect() }
}

fn wide_clip(frames: usize) -> Clip {
    // 1600 x 900 crops of the 1920 x 1080 gradient: a 16:9 tile of 42 x 24 patches (252 tokens a tubelet)
    Clip { file: "grad_1080x1920.png", crops: (0..frames).map(|i| (40 * i, 22 * i, 1600, 900)).collect() }
}

/// The features of an encoded tubelet (this encoder's).
fn feats_of(e: EncodedImage) -> (Arc<NemotronImageFeatures>, (usize, usize)) {
    let merged = e.merged;
    let f = e.features.downcast::<NemotronImageFeatures>().unwrap_or_else(|_| panic!("not this encoder's features"));
    assert_eq!(f.tokens, e.tokens);
    (f, merged)
}

fn encode_video(
    t: &mut TowerShared,
    v: &PreprocessedVideo,
    rate: Option<f64>,
) -> (Vec<(Arc<NemotronImageFeatures>, (usize, usize))>, Option<Vec<bool>>) {
    let input = VideoInput { frames: &v.frames, canvas: v.canvas() };
    let (g, (rows, cols)) = (v.tubelets(), v.tile.merged());
    let seen: Mutex<Option<Vec<bool>>> = Mutex::new(None);
    let keep = |f: &[u16]| -> Result<Vec<bool>, String> {
        let m =
            retention_mask_bits(f, g, rows, cols, FEATURE_WIDTH, rate.unwrap(), SumOrder::TorchCpu).map_err(|e| e.to_string())?;
        *seen.lock().unwrap() = Some(m.clone());
        Ok(m)
    };
    let keep_ref: Option<&dyn Fn(&[u16]) -> Result<Vec<bool>, String>> = if rate.is_some() { Some(&keep) } else { None };
    let enc = t.enc.encode_video(&input, keep_ref).unwrap_or_else(|e| panic!("encode_video: {e}"));
    let mask = seen.lock().unwrap().clone();
    (enc.into_iter().map(feats_of).collect(), mask)
}

fn bf16_rows(f: &NemotronImageFeatures) -> Vec<u16> {
    let mut bits = vec![0u16; f.tokens * 4096];
    if f.tokens > 0 {
        f.buf.download_range(0, &mut bits).unwrap();
    }
    bits
}

fn as_f32(b: &[u16]) -> Vec<f32> {
    b.iter().map(|&x| f32::from_bits((x as u32) << 16)).collect()
}

// ------------------------------------------------------------------------------------------------------------- the encoder

/// The batched encoder equals the tubelets encoded one at a time (bit for bit, through the whole encoder: upload, widen, six-plane embedding,
/// blocks, mlp1, bf16), an odd count's padded tubelet is `[last, last]`, the host mask picks exactly the rows the kept buffers hold, and the
/// budget balances: permits per tubelet, the unpruned set's permit given back once the kept rows are resident.
#[test]
#[ignore]
fn the_batched_encoder_equals_its_tubelets_alone_and_the_budget_balances() {
    let Some(dir) = sup::pack_dir() else { return };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap();
    let pre = NemotronPreprocessing::from_dir(&dir).unwrap();
    let t = TowerShared {
        enc: NemotronEncoder::new(NemotronTower::load(&dir, 3328).unwrap(), FeatureBudget::new(u64::MAX / 2)),
        pre,
    };
    let mut t = t;
    // 5 frames: three tubelets, the last [frame 4, frame 4]
    let v = preprocess(&t, &square_clip(5));
    assert_eq!((v.tubelets(), v.tile.tokens), (3, 256));
    let (all, _) = encode_video(&mut t, &v, None);
    assert_eq!(all.len(), 3);
    for (g, (f, merged)) in all.iter().enumerate() {
        assert_eq!((f.tokens, *merged), (256, (16, 16)), "tubelet {g}");
        // the same tubelet alone: frames (2g, 2g + 1), or (4, 4) for the padded one
        let [a, b] = gb_vision::nemotron_video::tubelet_frame_indices(5, g);
        let alone = PreprocessedVideo {
            tile: v.tile,
            frames: vec![v.frames[a].clone(), v.frames[b].clone()],
            frame_hashes: vec![0, 0],
            pixel_hash: 0,
        };
        let (one, _) = encode_video(&mut t, &alone, None);
        assert_eq!(one.len(), 1);
        assert!(bf16_rows(&one[0].0) == bf16_rows(f), "tubelet {g} of the batch differs from the same frames alone");
    }
    // the padded tubelet is `[last, last]`: a video of the last frame twice gives its rows
    let twice = PreprocessedVideo { tile: v.tile, frames: vec![v.frames[4].clone(); 2], frame_hashes: vec![0, 0], pixel_hash: 0 };
    let single = PreprocessedVideo { tile: v.tile, frames: vec![v.frames[4].clone()], frame_hashes: vec![0], pixel_hash: 0 };
    let ((a, _), (b, _)) = (encode_video(&mut t, &twice, None), encode_video(&mut t, &single, None));
    assert!(bf16_rows(&a[0].0) == bf16_rows(&b[0].0), "one frame is [f, f]");
    println!("encoder: 3 tubelets of 256 rows (the last [f4, f4]) are bit-identical to their own runs; one frame equals [f, f]");

    // pruning: the kept rows of every tubelet are the unpruned rows the host mask names, in order
    let (kept, mask) = encode_video(&mut t, &v, Some(0.5));
    let mask = mask.expect("the mask was asked for");
    assert_eq!(mask.len(), 3 * 256);
    let want_total = (3.0f64 * 256.0 * 0.5) as usize;
    assert_eq!(kept.iter().map(|(f, _)| f.tokens).sum::<usize>(), want_total.max(256));
    for g in 0..3 {
        let rows = bf16_rows(&kept[g].0);
        let full = bf16_rows(&all[g].0);
        let mut at = 0;
        for r in 0..256 {
            if mask[g * 256 + r] {
                assert!(
                    rows[at * 4096..(at + 1) * 4096] == full[r * 4096..(r + 1) * 4096],
                    "tubelet {g}: kept row {at} is not row {r}"
                );
                at += 1;
            }
        }
        assert_eq!(at, kept[g].0.tokens);
        assert_eq!(kept[g].1, if at == 256 { (16, 16) } else { (1, at) }, "tubelet {g}: the merged grid of a pruned tubelet");
    }
    assert_eq!(kept[0].0.tokens, 256, "the first tubelet is kept whole");
    println!(
        "pruning 0.5 of 3 x 256 rows: kept {:?}; each kept row is the unpruned row the host mask names",
        kept.iter().map(|(f, _)| f.tokens).collect::<Vec<_>>()
    );

    // the budget: room for the unpruned set of 3 tubelets and the same again while pruning
    let full = feature_bytes(256);
    let budget = FeatureBudget::new(6 * full);
    let mut enc = NemotronEncoder::new(NemotronTower::load(&dir, 3328).unwrap(), budget.clone());
    let input = VideoInput { frames: &v.frames, canvas: v.canvas() };
    let e = enc.encode_video(&input, None).unwrap();
    assert_eq!((budget.free(), budget.total()), (3 * full, 6 * full), "a permit per tubelet");
    // a second video takes the other three permits; a third finds none and holds nothing
    let x = enc.encode_video(&input, None).unwrap();
    assert_eq!(budget.free(), 0);
    let busy = enc.encode_video(&input, None).err().expect("the budget is exhausted");
    assert!(matches!(&busy, ModelError::Busy(m) if m.contains("retry")), "{busy}");
    assert_eq!(budget.free(), 0, "a refused encode holds nothing");
    drop(x);
    assert_eq!(budget.free(), 3 * full, "a video's permits came back with its features");
    drop(e);
    assert_eq!(budget.free(), 6 * full, "permits came back with the features");
    // all or nothing: with room for four tubelets a second video takes the one that is left and gives it back when the second is refused
    let tight = FeatureBudget::new(4 * full);
    let mut enc3 = NemotronEncoder::new(NemotronTower::load(&dir, 3328).unwrap(), tight.clone());
    let held = enc3.encode_video(&input, None).unwrap();
    assert_eq!(tight.free(), full);
    let busy = enc3.encode_video(&input, None).err().expect("one tubelet's room is not three");
    assert!(matches!(&busy, ModelError::Busy(_)), "{busy}");
    assert_eq!(tight.free(), full, "the permit taken before the refusal was given back");
    drop(held);
    assert_eq!(tight.free(), 4 * full);
    // pruning needs the unpruned set and the kept rows together: 6 x 256 rows of budget is exactly enough, 5 is a refusal by name
    let keep = |f: &[u16]| retention_mask_bits(f, 3, 16, 16, 4096, 0.5, SumOrder::TorchCpu).map_err(|e| e.to_string());
    let pruned = enc.encode_video(&input, Some(&keep)).unwrap();
    let kept_bytes: u64 = pruned.iter().map(|e| feature_bytes(e.tokens)).sum();
    assert_eq!(budget.free(), 6 * full - kept_bytes, "only the kept rows' permits remain");
    drop(pruned);
    assert_eq!(budget.free(), 6 * full);
    let small = FeatureBudget::new(5 * full);
    let mut enc2 = NemotronEncoder::new(NemotronTower::load(&dir, 3328).unwrap(), small.clone());
    let e = enc2.encode_video(&input, Some(&keep)).err().expect("pruning needs room for both");
    assert!(matches!(&e, ModelError::Request(m) if m.contains("twice while pruning")), "{e}");
    assert_eq!(small.free(), 5 * full, "a refused video holds nothing");
    println!("budget: a permit per tubelet, all or nothing, the unpruned set's share given back after the gather; pruning refused by name when the budget cannot hold both");
}

/// A tile the workspace cannot stage twice is a request error naming the flag, before any device work.
#[test]
#[ignore]
fn a_tile_over_half_the_workspace_is_refused_by_name() {
    let Some(dir) = sup::pack_dir() else { return };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap();
    // 512 tokens (2,048 patches): a 256-token tile (1,024 patches) is exactly half
    let mut enc = NemotronEncoder::new(NemotronTower::load(&dir, 512).unwrap(), FeatureBudget::new(u64::MAX / 2));
    let frames = vec![vec![0u16; 3 * 512 * 512]; 2];
    let ok = enc.encode_video(&VideoInput { frames: &frames, canvas: (512, 512) }, None).unwrap();
    assert_eq!(ok.len(), 1);
    // a 640 x 480 canvas is 1,200 patches (300 tokens): over half of 2,048
    let big = vec![vec![0u16; 3 * 640 * 480]; 2];
    let e = enc.encode_video(&VideoInput { frames: &big, canvas: (640, 480) }, None).err().unwrap();
    assert!(matches!(&e, ModelError::Request(m) if m.contains("--max-image-tokens must be at least 600")), "{e}");
    let mut enc = NemotronEncoder::new(NemotronTower::load(&dir, 256).unwrap(), FeatureBudget::new(u64::MAX / 2));
    let e = enc.encode_video(&VideoInput { frames: &frames, canvas: (512, 512) }, None).err().unwrap();
    assert!(matches!(&e, ModelError::Request(m) if m.contains("--max-image-tokens must be at least 512")), "{e}");
    println!("a video tile needs --max-image-tokens of twice its tokens: refused by name at 256, served at 512");
}

// ------------------------------------------------------------------------------------------------------------- prompts

/// A video prompt: text, then for each tubelet `<img>`, its kept placeholders, `</img>`, with `sep` text tokens between, then text.
struct Prompt {
    ids: Vec<u32>,
    /// `(start of the placeholder run, its features)` per tubelet that kept rows, in order.
    spans: Vec<(usize, Arc<NemotronImageFeatures>)>,
}

fn build_prompt(text: &[u32], lead: usize, sep: usize, tubelets: &[Arc<NemotronImageFeatures>], tail: usize) -> Prompt {
    let mut ids = Vec::new();
    let mut spans = Vec::new();
    let mut at = 0;
    ids.extend_from_slice(&text[at..at + lead]);
    at += lead;
    for (g, f) in tubelets.iter().enumerate() {
        ids.push(IMG_START);
        if f.tokens > 0 {
            spans.push((ids.len(), f.clone()));
            ids.extend(std::iter::repeat_n(IMG, f.tokens));
        }
        ids.push(IMG_END);
        if g + 1 < tubelets.len() {
            ids.extend_from_slice(&text[at..at + sep]);
            at += sep;
        }
    }
    ids.extend_from_slice(&text[at..at + tail]);
    Prompt { ids, spans }
}

fn splice_for(spans: &[(usize, Arc<NemotronImageFeatures>)], at: usize, end: usize) -> Vec<SpliceOp> {
    spans
        .iter()
        .filter_map(|(start, f)| {
            let (lo, hi) = ((*start).max(at), (*start + f.tokens).min(end));
            (lo < hi).then(|| SpliceOp { feats: f.clone(), dst_row: lo - at, src_row: lo - start, n: hi - lo })
        })
        .collect()
}

/// `prompt` into `slot` in slices of at most `chunk` rows on the absolute grid, each with its share of the splice.
fn prefill(m: &mut NemotronModel, slot: usize, p: &Prompt, chunk: usize, drafting: bool) -> (u32, RowCands) {
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
        at = end;
    }
    first.unwrap()
}

/// `prompt` into `slot` in pieces cut at exactly `cuts` (a piece longer than 64 rows is cut again at 64).
fn prefill_cuts(m: &mut NemotronModel, slot: usize, p: &Prompt, cuts: &[usize]) -> (u32, RowCands) {
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

fn oracle_text(n: usize) -> Option<Vec<u32>> {
    let ids = sup::oracle_ids()?;
    assert!(ids.len() >= n, "the oracle's text has {} tokens", ids.len());
    Some(ids[..n].to_vec())
}

// ------------------------------------------------------------------------------------------------------------- the splice

/// A video prompt's embedding rows are the kept feature rows exactly (bf16 widened) over each tubelet's run and the token embeddings
/// everywhere else, and a prompt cut at any row, inside a tubelet's run included, leaves the same state, head cache and next-token
/// candidates as the 64-row slicing; for the full video and for a pruned one (kept runs of any length, a tubelet with none).
#[test]
#[ignore]
fn a_videos_rows_are_spliced_exactly_and_chunked_prefill_equals_whole() {
    let (Some(mut t), Some(text)) = (tower(), oracle_text(160)) else { return };
    let Some(mut g) = sup::model() else { return };
    let m: &mut NemotronModel = &mut g;
    let v = preprocess(&t, &square_clip(6));
    assert_eq!((v.tubelets(), v.tile.tokens), (3, 256));
    for (label, rate) in [("full", None), ("pruned 0.5", Some(0.5)), ("pruned 0.9", Some(0.9))] {
        let (tubs, mask) = encode_video(&mut t, &v, rate);
        let feats: Vec<Arc<NemotronImageFeatures>> = tubs.iter().map(|(f, _)| f.clone()).collect();
        let kept: Vec<usize> = feats.iter().map(|f| f.tokens).collect();
        // text, three tubelets with a label between them, text: slices of 64 rows cut every run
        let p = build_prompt(&text, 21, 13, &feats, 17);
        println!(
            "{label}: kept {kept:?}, prompt of {} tokens, runs at {:?}",
            p.ids.len(),
            p.spans.iter().map(|s| s.0).collect::<Vec<_>>()
        );
        if let Some(mask) = &mask {
            assert_eq!(mask.iter().filter(|&&k| k).count(), kept.iter().sum::<usize>());
        }

        // -- the embedding rows, tapped, against the same prompt as plain text
        let got = Arc::new(Mutex::new(Vec::new()));
        m.set_tap(Some(Box::new(Rows { want: vec![Point::Embed], got: got.clone() })));
        prefill(m, 0, &p, 64, true);
        let with: Vec<f32> = got.lock().unwrap().drain(..).flat_map(|(_, v)| v).collect();
        let text_only = Prompt { ids: p.ids.clone(), spans: vec![] };
        prefill(m, 1, &text_only, 64, true);
        let without: Vec<f32> = got.lock().unwrap().drain(..).flat_map(|(_, v)| v).collect();
        m.set_tap(None);
        m.end_slot(1);
        assert_eq!((with.len(), without.len()), (p.ids.len() * 4096, p.ids.len() * 4096));
        let mut feature_rows = 0;
        for (start, f) in &p.spans {
            let rows = as_f32(&bf16_rows(f));
            for r in 0..f.tokens {
                let row = &with[(start + r) * 4096..(start + r + 1) * 4096];
                assert!(
                    row.iter().zip(&rows[r * 4096..(r + 1) * 4096]).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{label}: row {r} of the run at {start} is not its feature row"
                );
                feature_rows += 1;
            }
        }
        let is_video = |i: usize| p.spans.iter().any(|(s, f)| (*s..s + f.tokens).contains(&i));
        let mut text_rows = 0;
        for i in (0..p.ids.len()).filter(|&i| !is_video(i)) {
            assert!(
                with[i * 4096..(i + 1) * 4096]
                    .iter()
                    .zip(&without[i * 4096..(i + 1) * 4096])
                    .all(|(x, y)| x.to_bits() == y.to_bits()),
                "{label}: text row {i} changed"
            );
            text_rows += 1;
        }
        println!("{label}: {feature_rows} video rows are exactly their features, {text_rows} text rows exactly their embeddings");

        // -- chunked equals whole, cuts inside every tubelet's run included
        let (tok0, c0) = prefill(m, 0, &p, 64, true);
        let n = p.ids.len();
        let head0 = m.mtp_kv_rows(0, 0, n - 1).unwrap();
        for chunk in [48usize, 37, 16, 7] {
            let (tok, c) = prefill(m, 2, &p, chunk, true);
            assert_eq!((tok, &c), (tok0, &c0), "{label}: chunk {chunk}: the candidates after the prompt differ");
            sup::same_trunk_state(m, 0, 2, n).unwrap_or_else(|e| panic!("{label}: chunk {chunk}: {e}"));
            assert!(m.mtp_kv_rows(2, 0, n - 1).unwrap() == head0, "{label}: chunk {chunk}: the head's cache differs");
        }
        let mut cuts = vec![1usize, 5];
        for (start, f) in &p.spans {
            for c in [
                start - 2,
                start - 1,
                *start,
                start + 1,
                start + f.tokens / 2,
                start + f.tokens - 1,
                start + f.tokens,
                start + f.tokens + 1,
            ] {
                cuts.push(c);
            }
        }
        cuts.sort_unstable();
        cuts.dedup();
        cuts.retain(|&c| c > 0 && c < n);
        let (tok, c) = prefill_cuts(m, 2, &p, &cuts);
        assert_eq!((tok, &c), (tok0, &c0), "{label}: cuts {cuts:?}: the candidates differ");
        sup::same_trunk_state(m, 0, 2, n).unwrap_or_else(|e| panic!("{label}: cuts {cuts:?}: {e}"));
        assert!(m.mtp_kv_rows(2, 0, n - 1).unwrap() == head0, "{label}: cuts: the head's cache differs");
        println!(
            "{label}: chunked == whole for 48/37/16/7-row slicings and {} cuts around and inside the runs: candidates, trunk state and head cache bit for bit",
            cuts.len()
        );
        m.end_slot(0);
        m.end_slot(2);
    }
}

// ------------------------------------------------------------------------------------------------------------- latency

/// The encoder's latency for 8 and 32 frames of a square and a 16:9 clip (synthetic pixels): the batched call against the tubelets one at a
/// time, per tubelet, with the tower's TFLOP/s. The median of 5 calls after 2 warm-ups.
#[test]
#[ignore]
fn the_video_encoders_latency() {
    let Some(mut t) = tower() else { return };
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut frame = |n: usize| -> Vec<u16> {
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((((s >> 40) as u32 % 4000) as f32 / 1000.0 - 2.0).to_bits() >> 16) as u16
            })
            .collect()
    };
    for (name, (w, h)) in [("square 512x512", (512usize, 512usize)), ("16:9 672x384", (672, 384))] {
        for frames_n in [8usize, 32] {
            let frames: Vec<Vec<u16>> = (0..frames_n).map(|_| frame(3 * w * h)).collect();
            let tubelets = frames_n / 2;
            let med = |mut v: Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            };
            let mut batched = Vec::new();
            let mut alone = Vec::new();
            for i in 0..7 {
                let t0 = std::time::Instant::now();
                let e = t.enc.encode_video(&VideoInput { frames: &frames, canvas: (w, h) }, None).unwrap();
                let d = t0.elapsed().as_secs_f64() * 1e3;
                drop(e);
                let t1 = std::time::Instant::now();
                for g in 0..tubelets {
                    let pair = [frames[2 * g].clone(), frames[2 * g + 1].clone()];
                    let e = t.enc.encode_video(&VideoInput { frames: &pair, canvas: (w, h) }, None).unwrap();
                    drop(e);
                }
                let d1 = t1.elapsed().as_secs_f64() * 1e3;
                if i >= 2 {
                    batched.push(d);
                    alone.push(d1);
                }
            }
            let (mb, ma) = (med(batched), med(alone));
            let n = ((w / 16) * (h / 16)) as f64;
            let tk = n + 10.0;
            let gemm = 32.0 * 2.0 * tk * 1280.0 * (3840.0 + 1280.0 + 5120.0 + 5120.0)
                + 2.0 * n * 1536.0 * 1280.0
                + (n / 4.0) * (2.0 * 5120.0 * 20480.0 + 2.0 * 20480.0 * 4096.0);
            let attn = 32.0 * 4.0 * tk * tk * 1280.0;
            let flops = tubelets as f64 * (gemm + attn);
            println!(
                "{name}, {frames_n} frames ({tubelets} tubelets of {} rows): batched {mb:.1} ms ({:.1} ms a tubelet, {:.1} TFLOP/s); one tubelet a call {ma:.1} ms ({:.1} ms a tubelet)",
                (n / 4.0) as usize,
                mb / tubelets as f64,
                flops / 1e12 / (mb / 1e3),
                ma / tubelets as f64
            );
        }
    }
}

// ------------------------------------------------------------------------------------------------------------- the KL leg

/// The engine's logits over video prompts for the KL against the bf16 original (`bf16_reference_layerstream.py --videos`).
///
/// `REXL3_NEMO_VID_CASES` is the directory of `cases.json` (`[{"name", "file", "crops": [[x, y, w, h]], "question", "n_new", "rate"}]`, made
/// and completed by `--videos-prepare`: `ids` is the processor's expanded prompt, `pixels` its `pixel_values_videos`, `pixels_shape`);
/// `REXL3_NEMO_VID_OUT` receives, per case, `<name>.answer.u32` (the engine's greedy answer), `<name>.engine_logits.f32` (the rows that
/// predict it), `<name>.engine_features.bf16` (the unpruned features), `<name>.engine_mask.u8` (the host EVS mask, when `rate` > 0) and
/// the scalars printed. The prompt is the processor's ids with the dropped rows removed, as the model's `forward` does after the scatter.
#[test]
#[ignore]
fn video_prompts_logits_for_the_bf16_kl() {
    let (Some(cases_dir), Some(out)) = (env_dir("REXL3_NEMO_VID_CASES"), env_dir("REXL3_NEMO_VID_OUT")) else { return };
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
        let rate = case["rate"].as_f64().filter(|&r| r > 0.0);
        let clip = Clip {
            file: Box::leak(case["file"].as_str().unwrap().to_string().into_boxed_str()),
            crops: case["crops"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    let c: Vec<usize> = c.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
                    (c[0], c[1], c[2], c[3])
                })
                .collect(),
        };
        let mut v = preprocess(&t, &clip);
        // the processor's pixel_values_videos against the engine's own preprocessing: a PNG source must be exact; otherwise the model's
        // numerics are judged on the PROCESSOR's pixels (the reference's own input)
        let mut px_diff = None;
        if let Some(f) = case.get("pixels").and_then(|x| x.as_str()) {
            let want: Vec<u16> =
                std::fs::read(cases_dir.join(f)).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let have: Vec<u16> = v.frames.iter().flatten().copied().collect();
            assert_eq!(want.len(), have.len(), "{name}: the processor's frames and the engine's differ in size");
            let d = want.iter().zip(&have).filter(|(a, b)| a != b).count();
            if d > 0 {
                assert!(!clip.file.ends_with(".png"), "{name}: a PNG's pixels differ from the processor's in {d} values");
                let per = want.len() / v.frames.len();
                v.frames = want.chunks(per).map(|c| c.to_vec()).collect();
            }
            px_diff = Some(d);
        }
        let ids_full = read_u32(&cases_dir.join(case["ids"].as_str().unwrap()));
        let g = v.tubelets();
        let n = v.tile.tokens;
        // the processor's placeholders: g runs of n, in order
        let placeholders: Vec<usize> = ids_full.iter().enumerate().filter(|(_, &x)| x == IMG).map(|(i, _)| i).collect();
        assert_eq!(placeholders.len(), g * n, "{name}: the processor's placeholder count is the video's tokens");

        let t0 = std::time::Instant::now();
        let (tubs, mask) = encode_video(&mut t, &v, rate);
        let t_encode = t0.elapsed().as_secs_f64();
        // the pruned prompt: the processor's ids without the placeholders the mask drops
        let mut keep = vec![true; ids_full.len()];
        if let Some(mask) = &mask {
            for (k, &pos) in placeholders.iter().enumerate() {
                keep[pos] = mask[k];
            }
        }
        let ids: Vec<u32> = ids_full.iter().zip(&keep).filter(|(_, &k)| k).map(|(&x, _)| x).collect();
        // spans: each tubelet's kept run, found in the pruned ids (runs of the placeholder token, in order, one per tubelet that kept any)
        let mut spans = Vec::new();
        let mut run_of = tubs.iter().filter(|(f, _)| f.tokens > 0).map(|(f, _)| f.clone());
        let mut i = 0;
        while i < ids.len() {
            if ids[i] == IMG {
                let j = (i..ids.len()).take_while(|&k| ids[k] == IMG).count() + i;
                let f = run_of.next().unwrap_or_else(|| panic!("{name}: more runs of placeholders than tubelets that kept rows"));
                assert_eq!(f.tokens, j - i, "{name}: the run at {i} is not the tubelet's kept count");
                spans.push((i, f));
                i = j;
            } else {
                i += 1;
            }
        }
        assert!(run_of.next().is_none(), "{name}: a tubelet that kept rows has no run");
        let p = Prompt { ids, spans };

        let got = Arc::new(Mutex::new(Vec::new()));
        m.set_tap(Some(Box::new(Rows { want: vec![Point::Logits], got: got.clone() })));
        let t1 = std::time::Instant::now();
        let (first, _) = prefill(m, 0, &p, 64, false);
        let t_prefill = t1.elapsed().as_secs_f64();
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
        std::fs::write(
            out.join(format!("{name}.engine_logits.f32")),
            rows.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
        )
        .unwrap();
        std::fs::write(out.join(format!("{name}.answer.u32")), answer.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
            .unwrap();
        // the unpruned features as the engine made them (for the comparison with the reference's own): the encode keeps only the kept
        // rows when pruning, so a second, unpruned encode of the same frames
        let (full, _) = if rate.is_some() { encode_video(&mut t, &v, None) } else { (tubs.clone(), None) };
        let mut fbits = Vec::new();
        for (f, _) in &full {
            fbits.extend(bf16_rows(f));
        }
        std::fs::write(
            out.join(format!("{name}.engine_features.bf16")),
            fbits.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
        )
        .unwrap();
        if let Some(mask) = &mask {
            std::fs::write(out.join(format!("{name}.engine_mask.u8")), mask.iter().map(|&k| k as u8).collect::<Vec<u8>>())
                .unwrap();
        }
        println!(
            "case {name}: {} frames, {g} tubelets of {n} rows{}, prompt {} tokens (from {}), encode {t_encode:.2} s, prefill {t_prefill:.2} s, {n_new} answer tokens; \
             engine's pixels differ from the processor's in {} values",
            v.frames.len(),
            rate.map(|r| format!(", pruning {r}: kept {:?}", tubs.iter().map(|(f, _)| f.tokens).collect::<Vec<_>>())).unwrap_or_default(),
            p.ids.len(),
            ids_full.len(),
            px_diff.map(|d| d.to_string()).unwrap_or_else(|| "n/a".into())
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
