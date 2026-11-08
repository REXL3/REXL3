//! The Flash-Next forward: host tests of the rules a prefix cache depends on, and the real-pack tests that prove them
//! on the device.
//!
//! The host tests need no pack and no CUDA context. The real-pack tests load the whole pack and run the GPU, so they
//! are gated on `REXL3_FLASH_NEXT_DIR` **and** `REXL3_FN_REAL=1`; without both they report what they did not run and
//! pass. Never set them on a machine that is serving a model: the pack is tens of gigabytes.
//!
//! ```text
//! REXL3_FN_REAL=1 REXL3_FLASH_NEXT_DIR=~/models/flashnext-exl3-3.05bpw \
//!   REXL3_FN_ORACLE_DIR=~/rexl3-oracle/ref305 \
//!   cargo test --release -p gb-model --test flash_next_forward -- --nocapture --test-threads=1
//! ```
//!
//! Sizes: `REXL3_FN_SLOTS` (1), `REXL3_FN_PAGES` (64 pages of 256 tokens), `REXL3_FN_CTX` (16384),
//! `REXL3_FN_PREFILL_ROWS` (1024, the prefill chunk grid), `REXL3_FN_PROMPT` (1031, synthetic prompt length).
//!
//! The oracle test (`the_logits_agree_with_the_reference_oracle_at_every_position`) reads a reference dump directory
//! (`REXL3_FN_ORACLE_DIR`, e.g. `~/rexl3-oracle/ref305` or `~/rexl3-oracle/ref305_8k`): `meta.json`, and per sample
//! `ids_<s>.u32`, `top1_<s>.i32` (every position) and `logits_<s>.f16` (`[keep, vocab]` of the LAST `keep` positions,
//! `keep` = `meta.keep_last`, or every position without it). Optional: `REXL3_FN_ORACLE_SAMPLES` (all of meta),
//! `REXL3_FN_ORACLE_PREFILL_UPTO` (positions from here on run as decode windows; default: all prefill),
//! `REXL3_FN_ORACLE_WIN` (8 rows a window), `REXL3_FN_ORACLE_MIN_TOP1` (0.97), `REXL3_FN_ORACLE_MIN_COS` (0.995, the
//! mean cosine of the kept rows' logits).
//!
//! The MTP test drafts on real text when `REXL3_FN_ORACLE_DIR` is set (its `ids_0.u32`), else on a synthetic prompt:
//! `REXL3_FN_DRAFTS` (4), `REXL3_FN_MTP_STEPS` (64 tokens), `REXL3_FN_MTP_MIN_ACCEPT` (0.2 on real text, 0 synthetic).
//!
//! The upload-order test (`the_per_step_uploads_are_ordered_on_the_models_stream`): `REXL3_FN_UPLOAD_PROMPT` (300 tokens),
//! `REXL3_FN_UPLOAD_WINDOWS` (300 windows back to back), `REXL3_FN_UPLOAD_SPEC_RUNS` (2 speculative decodes a mode),
//! `REXL3_FN_UPLOAD_STEPS` (96 tokens each), `REXL3_FN_UPLOAD_LOAD_MIB` (0: off; N: a thread makes pageable copies of N MiB
//! on the legacy default stream while the compared runs go, which also stalls the model's own read-backs, so it is slow).
use gb_engine::RowRequest;
use gb_model::families::qwen_flash_next::config::FlashNextConfig;
use gb_model::families::qwen_flash_next::model::{
    set_upload_mode, FlashNextModel, ModelOptions, UploadMode, MTP_HEAD_N, PLE_RING, QSA_RING, WINDOW_ROWS,
};
use gb_model::families::qwen_flash_next::ngram::{row_ids, TokContext};
use gb_model::forward::{Candidates, PrefillPath};
use gb_model::rows::RowBatch;
use serde_json::Value;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const CONFIG: &str = include_str!("../../../testdata/qwen_flash_next/config.json");

// ------------------------------------------------------------------ host tests

/// A directory holding only the given `config.json` text.
fn dir_with(name: &str, cfg: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rexl3-fn-fwd-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), cfg).unwrap();
    dir
}

fn opts() -> ModelOptions {
    ModelOptions { slots: 2, kv_pages: 8, max_context: 2048, prefill_rows: 256, warm: false }
}

/// Every chunk of one prompt takes the arithmetic of the prompt's TOTAL length, whatever grid it was cut into: that is
/// the whole of the path rule a snapshot relies on.
#[test]
fn the_prefill_path_is_the_prompts_total_and_no_chunk_grid_changes_it() {
    assert_eq!(PrefillPath::GEMV_MAX_PROMPT, 128);
    for total in [1usize, 2, 64, 127, 128] {
        assert_eq!(PrefillPath::for_prompt(total), PrefillPath::Gemv, "total {total}");
    }
    for total in [129usize, 130, 256, 1000, 4096] {
        assert_eq!(PrefillPath::for_prompt(total), PrefillPath::Tiled, "total {total}");
    }
    // the grid a prompt is cut into never enters the choice: a 1-row tail of a long prompt is Tiled like the rest
    for total in [129usize, 257, 1000, 4096] {
        let path = PrefillPath::for_prompt(total);
        for grid in [1usize, 3, 97, 128, 256, 1024] {
            let mut done = 0;
            while done < total {
                let rows = grid.min(total - done);
                assert_eq!(PrefillPath::for_prompt(total), path, "total {total} grid {grid} chunk of {rows}");
                done += rows;
            }
        }
    }
}

/// The hashed n-gram context of a position is a function of the prompt alone: cutting the prompt into chunks and
/// rebuilding the two context tokens from the per-slot ring gives the same 3-grams at every position.
#[test]
fn the_hashed_ngram_context_of_a_position_does_not_depend_on_the_chunk_grid() {
    const EOS: u32 = 248_044;
    let prompt: Vec<i32> = (0..300).map(|i| (i * 7919 + 13) % 100_000).collect();
    // the hash parameters of a 3-gram table with 8 heads a layer
    let multipliers = vec![1_000_003i64, 998_244_353, 2_147_483_647];
    let heads_per_ngram = 8usize;
    let heads = 2 * heads_per_ngram;
    let offsets: Vec<i64> = (0..heads).map(|j| (j as i64) * 1_024).collect();
    let sizes: Vec<i64> = vec![1_024i64; heads];

    let ids_for = |grid: usize| -> Vec<i64> {
        let mut tok = TokContext::new(1);
        let mut out = Vec::new();
        let mut first = 0usize;
        while first < prompt.len() {
            let n = grid.min(prompt.len() - first);
            let chunk = &prompt[first..first + n];
            let hist = tok.history(0, first, chunk, EOS);
            assert_eq!(hist.len(), n + 2, "two context tokens before the chunk's rows");
            for w in hist.windows(3) {
                let mut row = vec![0i64; heads];
                row_ids([w[2], w[1], w[0]], EOS, &multipliers, &offsets, &sizes, heads_per_ngram, &mut row).unwrap();
                out.extend(row);
            }
            tok.store(0, first, chunk);
            first += n;
        }
        out
    };

    let whole = ids_for(prompt.len());
    assert_eq!(whole.len(), prompt.len() * heads);
    for grid in [1usize, 2, 3, 97, 128, 256] {
        assert_eq!(ids_for(grid), whole, "chunk grid {grid} moved an n-gram row");
    }
}

/// The per-slot rings must be long enough that a window never overwrites a position the next window still reads.
#[test]
fn the_per_slot_rings_outlast_a_window_and_the_reach_of_what_reads_them() {
    // at compile time, so the sizes cannot drift apart without this file failing to build
    const _: () = assert!(PLE_RING >= WINDOW_ROWS + 9, "the PLE conv reaches 9 positions back at dilation 3");
    const _: () = assert!(QSA_RING >= WINDOW_ROWS + 3, "a QSA block pools the 3 keys before the window");
    const _: () = assert!(WINDOW_ROWS <= PrefillPath::GEMV_MAX_PROMPT);
    assert_eq!((PLE_RING, QSA_RING, WINDOW_ROWS), (128, 128, 64), "the sizes this file reasons about");
}

/// A pack is refused by naming what is wrong with it, before a byte is read or a buffer is allocated (so these run with
/// no CUDA device).
#[test]
fn a_pack_the_kernels_cannot_run_is_refused_by_name_before_anything_is_allocated() {
    let dir = dir_with("nocfg", "{}");
    let e = FlashNextModel::load(&dir, opts()).err().expect("an empty config is not a Flash-Next pack").to_string();
    assert!(e.contains("hidden_size") || e.contains("text_config"), "{e}");

    let mut v: Value = serde_json::from_str(CONFIG).unwrap();
    v["text_config"]["num_key_value_heads"] = serde_json::json!(4);
    let dir = dir_with("kv4", &v.to_string());
    let e = FlashNextModel::load(&dir, opts()).err().expect("4 KV heads is not the cache the kernels have").to_string();
    assert!(e.contains("num_key_value_heads 4"), "{e}");

    // the epsilon every norm of this model runs with is a shape, not a tuning knob
    let mut v: Value = serde_json::from_str(CONFIG).unwrap();
    v["text_config"]["rms_norm_eps"] = serde_json::json!(1e-5);
    let dir = dir_with("eps", &v.to_string());
    let e = FlashNextModel::load(&dir, opts()).err().expect("another epsilon is another function").to_string();
    assert!(e.contains("rms_norm_eps"), "{e}");

    // a size the model cannot be built for is refused before the weights are read
    let dir = dir_with("sizes", CONFIG);
    for bad in [
        ModelOptions { slots: 0, ..opts() },
        ModelOptions { kv_pages: 0, ..opts() },
        ModelOptions { max_context: 0, ..opts() },
        ModelOptions { prefill_rows: 0, ..opts() },
    ] {
        let e = FlashNextModel::load(&dir, bad).err().expect("a zero size is refused").to_string();
        assert!(e.contains("must be positive"), "{e}");
    }
    let e = FlashNextModel::load(&dir, ModelOptions { prefill_rows: 1 << 20, ..opts() })
        .err()
        .expect("a chunk past the kernels is refused")
        .to_string();
    assert!(e.contains("past what the kernels take"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The committed config is the one these tests reason about.
#[test]
fn the_committed_config_is_the_shape_the_forward_is_written_for() {
    let c = FlashNextConfig::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap();
    c.check_shapes().unwrap();
    assert_eq!((c.full_layers(), c.linear_layers()), (12, 36));
    assert_eq!(c.rms_norm_eps, 1e-6, "the forward's EPS");
    assert_eq!(c.rope_theta, 1e7);
    assert_eq!(c.mtp_layers, 1);
}

// ------------------------------------------------------- the real pack on the GPU

/// The pack and the explicit opt-in, or the reason the device tests did not run.
fn real() -> Option<PathBuf> {
    if std::env::var("REXL3_FN_REAL").ok().as_deref() != Some("1") {
        eprintln!("not run: REXL3_FN_REAL=1 is the opt-in for the real-pack device tests");
        return None;
    }
    let Some(dir) = std::env::var("REXL3_FLASH_NEXT_DIR").ok().map(PathBuf::from) else {
        eprintln!("not run: REXL3_FLASH_NEXT_DIR is not set");
        return None;
    };
    if !dir.join("config.json").exists() {
        eprintln!("not run: {} holds no config.json", dir.display());
        return None;
    }
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    Some(dir)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn real_opts() -> ModelOptions {
    ModelOptions {
        slots: env_usize("REXL3_FN_SLOTS", 1),
        kv_pages: env_usize("REXL3_FN_PAGES", 64),
        max_context: env_usize("REXL3_FN_CTX", 16_384),
        prefill_rows: env_usize("REXL3_FN_PREFILL_ROWS", 1024),
        warm: false,
    }
}

/// A deterministic prompt of `n` ids inside the vocabulary (the oracle test uses real ids instead).
fn synthetic_prompt(n: usize, vocab: usize) -> Vec<u32> {
    (0..n).map(|i| ((i as u64 * 2_654_435_761) % (vocab as u64 - 16) + 8) as u32).collect()
}

fn chain(slot: usize, first: usize, tokens: &[u32]) -> RowBatch {
    RowBatch::from_requests(&[RowRequest { slot, first_position: first as u64, tokens: tokens.to_vec() }])
}

/// Prefill `prompt` into `slot` in chunks of `grid`, on the path its TOTAL length picks, running the MTP head after each
/// chunk when `mtp` is set, while the trunk's stack still holds that chunk (which is how the engine drives it; the head
/// silently does nothing for a chunk the stack no longer holds, so its progress is asserted chunk by chunk). Returns the
/// candidates of the last row of the last chunk.
fn prefill_grid_mtp(m: &mut FlashNextModel, slot: usize, prompt: &[u32], grid: usize, count: usize, mtp: bool) -> Candidates {
    let path = PrefillPath::for_prompt(prompt.len());
    let mut done = 0usize;
    let mut last = None;
    while done < prompt.len() {
        let n = grid.min(prompt.len() - done);
        let chunk = &prompt[done..done + n];
        let want = if done + n == prompt.len() { Some(count) } else { None };
        last = m.prefill_chunk(&chain(slot, done, chunk), path, want).unwrap();
        if mtp {
            let next = prompt.get(done + n).copied();
            m.mtp_after_prefill(slot, done, chunk, next, path).unwrap();
            // through the chunk's last row when the token after it is known, one row short at the prompt's end
            let through = done + n - usize::from(next.is_none());
            if through > 0 {
                assert_eq!(m.mtp_done.get(&slot).copied(), Some(through), "the head after the chunk at {done} of {n} rows");
            }
        }
        done += n;
    }
    last.expect("the last chunk was asked for candidates")
}

fn prefill_in_grid(m: &mut FlashNextModel, slot: usize, prompt: &[u32], grid: usize, count: usize) -> Candidates {
    prefill_grid_mtp(m, slot, prompt, grid, count, false)
}

fn top1(c: &Candidates, row: usize) -> u32 {
    c.ids[row * c.count] as u32
}

/// A prompt of more than 128 tokens leaves the same bits in its last row's logits whatever grid it was cut into: the
/// claim every in-chunk snapshot rests on.
#[test]
fn a_tiled_prefill_leaves_the_same_bits_in_any_chunk_grid() {
    let Some(dir) = real() else { return };
    let mut m = FlashNextModel::load(&dir, real_opts()).unwrap();
    let total = env_usize("REXL3_FN_PROMPT", 1031);
    assert!(total > PrefillPath::GEMV_MAX_PROMPT, "this test is about the Tiled path");
    assert_eq!(PrefillPath::for_prompt(total), PrefillPath::Tiled);
    let prompt = synthetic_prompt(total, m.vocab_size());

    m.begin_slot(0).unwrap();
    let grid = m.opts.prefill_rows.min(total);
    let whole = prefill_in_grid(&mut m, 0, &prompt, grid, 8);
    // and one decode window after it, so the recurrent state and the pages are compared too
    let after_whole = m.window_candidates(&chain(0, total, &[top1(&whole, 0)]), 8).unwrap();
    m.commit(0, 1);
    m.flush().unwrap();

    for grid in [1usize, 3, 97, 128, 257] {
        m.begin_slot(0).unwrap();
        let cut = prefill_in_grid(&mut m, 0, &prompt, grid, 8);
        assert_eq!(cut.ids, whole.ids, "grid {grid}: the last row's candidate ids moved");
        assert_eq!(cut.vals, whole.vals, "grid {grid}: the last row's logit bits moved");
        let after = m.window_candidates(&chain(0, total, &[top1(&cut, 0)]), 8).unwrap();
        assert_eq!(after.ids, after_whole.ids, "grid {grid}: the state the prefill left behind differs");
        assert_eq!(after.vals, after_whole.vals, "grid {grid}: the state the prefill left behind differs");
        m.commit(0, 1);
        m.flush().unwrap();
        eprintln!("grid {grid}: identical bits in the last row and in the window after it");
    }
}

/// One window of `n` rows is the same function as `n` one-row windows: the decode regime is row invariant, which is what
/// lets a verify window stand in for serial decoding.
#[test]
fn a_window_of_many_rows_equals_the_same_rows_decoded_one_at_a_time() {
    let Some(dir) = real() else { return };
    let mut m = FlashNextModel::load(&dir, real_opts()).unwrap();
    let total = env_usize("REXL3_FN_PROMPT", 1031);
    let rows = env_usize("REXL3_FN_WINDOW", 8).min(WINDOW_ROWS);
    let prompt = synthetic_prompt(total, m.vocab_size());
    let grid = m.opts.prefill_rows.min(total);

    // serial: one row a window, committing each accepted row
    m.begin_slot(0).unwrap();
    let first = prefill_in_grid(&mut m, 0, &prompt, grid, 4);
    let mut fed: Vec<u32> = Vec::new();
    let mut serial: Vec<(Vec<i32>, Vec<f32>)> = Vec::new();
    let mut next = top1(&first, 0);
    for j in 0..rows {
        fed.push(next);
        let c = m.window_candidates(&chain(0, total + j, &[next]), 4).unwrap();
        serial.push((c.ids.clone(), c.vals.clone()));
        m.commit(0, 1);
        next = top1(&c, 0);
    }
    m.flush().unwrap();

    // the same rows as one window
    m.begin_slot(0).unwrap();
    let again = prefill_in_grid(&mut m, 0, &prompt, grid, 4);
    assert_eq!(again.ids, first.ids, "the re-prefill is not the same prefill");
    let w = m.window_candidates(&chain(0, total, &fed), 4).unwrap();
    m.commit(0, rows);
    m.flush().unwrap();
    assert_eq!(w.rows, rows);
    for (j, (ids, vals)) in serial.iter().enumerate() {
        assert_eq!(&w.ids[j * w.count..(j + 1) * w.count], &ids[..], "row {j}: the window's candidates differ");
        assert_eq!(&w.vals[j * w.count..(j + 1) * w.count], &vals[..], "row {j}: the window's logit bits differ");
    }
    eprintln!("a window of {rows} rows matched {rows} serial decodes bit for bit");
}

// ------------------------------------------------------------ the MTP head, verified by the trunk

/// Greedy speculative decoding from a fresh slot: prefill with the head run after each chunk, then `steps` tokens of
/// propose `k` drafts, verify them in one trunk window over the last token and the drafts, keep the agreeing prefix
/// plus the trunk's own next token.
struct SpecRun {
    tokens: Vec<u32>,
    drafts: Vec<Vec<u32>>,
    proposed: usize,
    accepted: usize,
}

fn spec_decode(m: &mut FlashNextModel, prompt: &[u32], grid: usize, k: usize, steps: usize) -> SpecRun {
    m.begin_slot(0).unwrap();
    let first = prefill_grid_mtp(m, 0, prompt, grid, 1, true);
    let mut history = prompt.to_vec();
    history.push(top1(&first, 0));
    let mut run = SpecRun { tokens: vec![top1(&first, 0)], drafts: Vec::new(), proposed: 0, accepted: 0 };
    let limit = MTP_HEAD_N.min(m.vocab_size()) as u32;
    while run.tokens.len() < steps {
        // the position of the last token, which the trunk has not run yet
        let pos = history.len() - 1;
        let d = m.mtp_propose(0, &history, k).unwrap();
        assert!(!d.is_empty() && d.len() <= k, "{} drafts for k = {k} at position {pos}", d.len());
        assert_eq!(m.mtp_done.get(&0).copied(), Some(pos), "the head caught up to the last token");
        assert!(d.iter().all(|&t| t < limit), "a draft outside the head's drafting range: {d:?}");
        let mut rows = vec![history[pos]];
        rows.extend_from_slice(&d);
        let c = m.window_candidates(&chain(0, pos, &rows), 1).unwrap();
        let a = (0..d.len()).take_while(|&j| top1(&c, j) == d[j]).count();
        m.commit(0, a + 1);
        history.extend_from_slice(&d[..a]);
        history.push(top1(&c, a));
        run.tokens.extend_from_slice(&d[..a]);
        run.tokens.push(top1(&c, a));
        run.proposed += d.len();
        run.accepted += a;
        run.drafts.push(d);
    }
    m.flush().unwrap();
    run
}

/// The MTP head's drafts, checked by the trunk: after a chunked prefill (the head's progress asserted chunk by chunk),
/// speculative greedy decoding must produce exactly the tokens serial greedy decoding produces, a second run from a fresh
/// slot must draft the same chains, and enough drafts must be accepted to be worth verifying.
#[test]
fn the_mtp_drafts_are_verified_by_the_trunk_and_reproduce_from_a_fresh_slot() {
    let Some(dir) = real() else { return };
    let mut m = FlashNextModel::load(&dir, real_opts()).unwrap();
    if !m.has_mtp() {
        eprintln!("not run: this pack has no MTP head");
        return;
    }
    let total = env_usize("REXL3_FN_PROMPT", 1031);
    let k = env_usize("REXL3_FN_DRAFTS", 4).clamp(1, 16);
    let steps = env_usize("REXL3_FN_MTP_STEPS", 64).max(2);
    let (prompt, real_text) = match std::env::var("REXL3_FN_ORACLE_DIR").ok() {
        Some(o) => {
            let ids = read_u32s(&Path::new(&o).join("ids_0.u32"));
            (ids[..total.min(ids.len())].to_vec(), true)
        }
        None => (synthetic_prompt(total, m.vocab_size()), false),
    };
    let total = prompt.len();
    assert!(total + steps + k < m.opts.max_context, "the prompt and the decode do not fit the context");
    let grid = m.opts.prefill_rows.min(total);

    let a = spec_decode(&mut m, &prompt, grid, k, steps);

    // serial greedy decoding from a fresh slot, one row a window, no head at all
    m.begin_slot(0).unwrap();
    let first = prefill_in_grid(&mut m, 0, &prompt, grid, 1);
    let mut serial = vec![top1(&first, 0)];
    while serial.len() < a.tokens.len() {
        let j = serial.len() - 1;
        let c = m.window_candidates(&chain(0, total + j, &[serial[j]]), 1).unwrap();
        m.commit(0, 1);
        serial.push(top1(&c, 0));
    }
    m.flush().unwrap();
    assert_eq!(a.tokens, serial, "speculative decoding left the trunk's greedy path");

    // the same run from a fresh slot drafts the same chains (nothing replayed on state the first run mutated)
    let b = spec_decode(&mut m, &prompt, grid, k, steps);
    assert_eq!(a.drafts, b.drafts, "the draft chains differ between two runs from a fresh slot");
    assert_eq!(a.tokens, b.tokens);

    let rate = a.accepted as f64 / a.proposed as f64;
    let min =
        std::env::var("REXL3_FN_MTP_MIN_ACCEPT").ok().and_then(|v| v.parse().ok()).unwrap_or(if real_text { 0.2 } else { 0.0 });
    eprintln!(
        "MTP: {} tokens in {} verify windows, {}/{} drafts accepted ({rate:.3}), k = {k}, {} prompt",
        a.tokens.len(),
        a.drafts.len(),
        a.accepted,
        a.proposed,
        if real_text { "real-text" } else { "synthetic" }
    );
    assert!(rate >= min, "draft acceptance {rate:.3} below {min}");
}

// ------------------------------------------------------------ the per-step uploads, in the stream's order

/// Puts the engine's upload mode back when the test that changed it ends, passing or not.
struct ModeGuard;

impl Drop for ModeGuard {
    fn drop(&mut self) {
        set_upload_mode(UploadMode::Ordered);
    }
}

/// A thread making pageable copies of `mib` MiB both ways on the legacy default stream (the stream the old per-step uploads
/// were queued on) until it is dropped.
struct CopyLoad {
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<u64>>,
}

impl CopyLoad {
    fn start(mib: usize) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let quit = Arc::clone(&stop);
        let worker = std::thread::spawn(move || {
            let host = vec![0x5au8; mib << 20];
            let mut back = vec![0u8; mib << 20];
            let mut dev = gb_cuda::DevBuf::alloc(mib << 20).expect("load buffer");
            let mut trips = 0u64;
            while !quit.load(Ordering::Relaxed) {
                dev.upload(&host).expect("load copy to the device");
                dev.download(&mut back).expect("load copy to the host");
                trips += 1;
            }
            trips
        });
        Self { stop, worker: Some(worker) }
    }

    /// Stops the thread; the round trips it made.
    fn finish(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        self.worker.take().map_or(0, |w| w.join().expect("load thread"))
    }
}

impl Drop for CopyLoad {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

type Quiet<T> = Result<T, String>;

fn quiet<T>(r: Result<T, gb_model::error::ModelError>) -> Quiet<T> {
    r.map_err(|e| e.to_string())
}

/// `prefill_grid_mtp` without assertions (a pass that races must report, not panic): the last row's candidates.
fn prefill_quiet(m: &mut FlashNextModel, prompt: &[u32], grid: usize, count: usize, mtp: bool) -> Quiet<Candidates> {
    quiet(m.begin_slot(0))?;
    let path = PrefillPath::for_prompt(prompt.len());
    let (mut done, mut last) = (0usize, None);
    while done < prompt.len() {
        let n = grid.min(prompt.len() - done);
        let chunk = &prompt[done..done + n];
        let want = (done + n == prompt.len()).then_some(count);
        last = quiet(m.prefill_chunk(&chain(0, done, chunk), path, want))?;
        if mtp {
            quiet(m.mtp_after_prefill(0, done, chunk, prompt.get(done + n).copied(), path))?;
        }
        done += n;
    }
    last.ok_or_else(|| "the last chunk gave no candidates".to_string())
}

/// `spec_decode` without assertions: the tokens and the draft chains of a greedy speculative decode from a fresh slot.
type Decode = (Vec<u32>, Vec<Vec<u32>>);

fn spec_quiet(m: &mut FlashNextModel, prompt: &[u32], grid: usize, k: usize, steps: usize) -> Quiet<Decode> {
    let first = prefill_quiet(m, prompt, grid, 1, true)?;
    let mut history = prompt.to_vec();
    history.push(top1(&first, 0));
    let (mut tokens, mut drafts) = (vec![top1(&first, 0)], Vec::new());
    while tokens.len() < steps {
        let pos = history.len() - 1;
        let d = quiet(m.mtp_propose(0, &history, k))?;
        let mut rows = vec![history[pos]];
        rows.extend_from_slice(&d);
        let c = quiet(m.window_candidates(&chain(0, pos, &rows), 1))?;
        let a = (0..d.len()).take_while(|&j| top1(&c, j) == d[j]).count();
        m.commit(0, a + 1);
        history.extend_from_slice(&d[..a]);
        history.push(top1(&c, a));
        tokens.extend_from_slice(&d[..a]);
        tokens.push(top1(&c, a));
        drafts.push(d);
    }
    quiet(m.flush())?;
    Ok((tokens, drafts))
}

fn same_candidates(a: &Candidates, b: &Candidates) -> bool {
    a.rows == b.rows
        && a.count == b.count
        && a.ids == b.ids
        && a.vals.iter().zip(&b.vals).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The model's stream is non-blocking, and before the fix every per-step input (token ids, positions, sequence starts and
/// slots, the row index, the commit's keeps, the draft chain's positions, the PLE rows, the page table, the MTP row-0 index)
/// was written with `DevBuf::upload`, a pageable `cudaMemcpy` on the legacy default stream that can return before its DMA
/// lands: a kernel launched right after it could read the previous step's bytes. The same work runs in three upload modes
/// and is compared bit for bit:
///
/// * the reference: `UploadMode::LegacySynced` (the old writes, each followed by a device sync), no load;
/// * `UploadMode::Ordered` (the engine: queued on the model's stream), with the optional copy load: every answer must equal
///   the reference (asserted, so the fix keeps every bit of the old arithmetic);
/// * `UploadMode::Legacy` (the old code) under the same load: reported, not asserted, since a race may not fire in a run.
///
/// The work: windows of fresh ids back to back with no commit, so each answer is a function of its ids alone (a window
/// that read its predecessor's ids answers differently), then greedy speculative decodes (MTP draft chain, verify window,
/// commit) whose tokens and drafts must repeat. Every pass starts from a fresh prefill of the same prompt.
#[test]
fn the_per_step_uploads_are_ordered_on_the_models_stream() {
    let Some(dir) = real() else { return };
    let _restore = ModeGuard;
    let mut m = FlashNextModel::load(&dir, real_opts()).unwrap();
    let total = env_usize("REXL3_FN_UPLOAD_PROMPT", 300);
    let n = env_usize("REXL3_FN_UPLOAD_WINDOWS", 300);
    let runs = env_usize("REXL3_FN_UPLOAD_SPEC_RUNS", 2);
    let steps = env_usize("REXL3_FN_UPLOAD_STEPS", 96).max(2);
    let k = env_usize("REXL3_FN_DRAFTS", 4).clamp(1, 16);
    let load_mib = env_usize("REXL3_FN_UPLOAD_LOAD_MIB", 0);
    let vocab = m.vocab_size();
    let prompt = synthetic_prompt(total, vocab);
    let grid = m.opts.prefill_rows.min(total);
    assert!(total + steps + k + 8 < m.opts.max_context, "the prompt and the decode do not fit the context");
    let mtp = m.has_mtp();

    // 16 windows of 1..=8 rows of fresh ids at the end of the prompt, in an order with no window twice in a row
    let batches: Vec<RowBatch> = (0..16usize)
        .map(|i| {
            let toks: Vec<u32> = (0..1 + i % 8).map(|j| ((i * 7919 + j * 104_729 + 17) % (vocab - 16) + 8) as u32).collect();
            chain(0, total, &toks)
        })
        .collect();
    let mut order = Vec::with_capacity(n);
    let (mut s, mut last) = (0x9E37_79B9_7F4A_7C15u64, usize::MAX);
    while order.len() < n {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        let i = (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize % batches.len();
        if i != last {
            order.push(i);
            last = i;
        }
    }

    let pass = |m: &mut FlashNextModel, mode: UploadMode, load: bool| -> (Quiet<Vec<Candidates>>, Vec<Quiet<Decode>>, u64) {
        set_upload_mode(mode);
        let load = (load && load_mib > 0).then(|| CopyLoad::start(load_mib));
        let windows = prefill_quiet(m, &prompt, grid, 4, false)
            .and_then(|_| order.iter().map(|&i| quiet(m.window_candidates(&batches[i], 4))).collect());
        let decodes = if mtp { (0..runs).map(|_| spec_quiet(m, &prompt, grid, k, steps)).collect() } else { Vec::new() };
        let trips = load.map_or(0, CopyLoad::finish);
        set_upload_mode(UploadMode::Ordered);
        (windows, decodes, trips)
    };
    let tally = |got: &(Quiet<Vec<Candidates>>, Vec<Quiet<Decode>>, u64), want_w: &[Candidates], want_d: &Decode| {
        let bad_w = match &got.0 {
            Ok(w) => w.iter().zip(want_w).filter(|(a, b)| !same_candidates(a, b)).count(),
            Err(_) => n,
        };
        let bad_d = got.1.iter().filter(|d| d.as_ref().map_or(true, |d| d != want_d)).count();
        (bad_w, bad_d)
    };

    let (ref_w, ref_d, _) = pass(&mut m, UploadMode::LegacySynced, false);
    let ref_w = ref_w.expect("the reference windows");
    let ref_d: Vec<Decode> = ref_d.into_iter().map(|d| d.expect("a reference decode")).collect();
    assert!(ref_d.windows(2).all(|p| p[0] == p[1]), "the reference decode does not repeat from a fresh slot");
    let want_d = ref_d.first().cloned().unwrap_or_default();
    if let Some((tokens, drafts)) = ref_d.first() {
        let proposed: usize = drafts.iter().map(Vec::len).sum();
        eprintln!("UPLOADS reference: {} tokens in {} verify windows, {proposed} drafts proposed", tokens.len(), drafts.len());
    }

    let ordered = pass(&mut m, UploadMode::Ordered, true);
    if let Err(e) = &ordered.0 {
        panic!("the ordered windows failed: {e}");
    }
    let (ow, od) = tally(&ordered, &ref_w, &want_d);
    eprintln!(
        "UPLOADS ordered: {ow}/{n} windows and {od}/{} decodes differ from the reference ({} load round trips of {load_mib} MiB)",
        ordered.1.len(),
        ordered.2
    );
    // the old code last: a race in it may leave the context unusable
    let legacy = pass(&mut m, UploadMode::Legacy, true);
    let (lw, ld) = tally(&legacy, &ref_w, &want_d);
    let errors: Vec<&String> = legacy.1.iter().filter_map(|d| d.as_ref().err()).chain(legacy.0.as_ref().err()).collect();
    eprintln!(
        "UPLOADS legacy: {lw}/{n} windows and {ld}/{} decodes differ from the reference ({} load round trips; errors: {errors:?})",
        legacy.1.len(),
        legacy.2
    );
    assert_eq!((ow, od), (0, 0), "the ordered uploads changed an answer: they must equal the synced reference bit for bit");
}

// ------------------------------------------------------- all-position parity with the reference dump

/// Every position of every sample of a reference dump: the prefill (or decode windows past
/// `REXL3_FN_ORACLE_PREFILL_UPTO`) leaves its collapsed rows, the LM head runs over them in [`WINDOW_ROWS`]-row slices,
/// and each row's argmax is compared with the reference's top-1 and, for the positions the dump kept logits of, the
/// logits with the reference's (cosine). Host memory stays at two slices of logits whatever the prompt length.
#[test]
fn the_logits_agree_with_the_reference_oracle_at_every_position() {
    let Some(dir) = real() else { return };
    let Some(odir) = std::env::var("REXL3_FN_ORACLE_DIR").ok().map(PathBuf::from) else {
        eprintln!("not run: REXL3_FN_ORACLE_DIR is not set (e.g. ~/rexl3-oracle/ref305)");
        return;
    };
    let meta = OracleMeta::read(&odir);
    let min_top1: f64 = std::env::var("REXL3_FN_ORACLE_MIN_TOP1").ok().and_then(|v| v.parse().ok()).unwrap_or(0.97);
    let min_cos: f64 = std::env::var("REXL3_FN_ORACLE_MIN_COS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.995);
    let n_samples = env_usize("REXL3_FN_ORACLE_SAMPLES", meta.samples.len()).min(meta.samples.len());
    let win = env_usize("REXL3_FN_ORACLE_WIN", 8).clamp(1, WINDOW_ROWS);

    let mut m = FlashNextModel::load(&dir, real_opts()).unwrap();
    let vocab = m.vocab_size();
    let mut ours = vec![0u16; WINDOW_ROWS * vocab];
    let mut theirs = vec![0u8; WINDOW_ROWS * vocab * 2];
    let (mut agree, mut total, mut self_agree, mut kept) = (0usize, 0usize, 0usize, 0usize);
    let (mut cos_sum, mut cos_min) = (0f64, f64::INFINITY);

    for &s in &meta.samples[..n_samples] {
        let ids = read_u32s(&odir.join(format!("ids_{s}.u32")));
        let want = read_i32s(&odir.join(format!("top1_{s}.i32")));
        let n = ids.len();
        assert_eq!(want.len(), n, "sample {s}: {} top-1 for {n} ids", want.len());
        assert!(n <= m.opts.max_context, "sample {s}: {n} tokens past REXL3_FN_CTX {}", m.opts.max_context);
        let keep = meta.kept(n);
        let lpath = odir.join(format!("logits_{s}.f16"));
        let mut lf = std::fs::File::open(&lpath).unwrap_or_else(|e| panic!("{}: {e}", lpath.display()));
        let len = lf.metadata().unwrap().len();
        assert_eq!(len, (keep * vocab * 2) as u64, "{}: not [{keep}, {vocab}] f16", lpath.display());
        let first_kept = n - keep;
        let upto = env_usize("REXL3_FN_ORACLE_PREFILL_UPTO", n).min(n);
        let path = PrefillPath::for_prompt(upto.max(1));
        let grid = m.opts.prefill_rows;
        let (mut s_agree, mut s_cos_sum, mut s_kept) = (0usize, 0f64, 0usize);

        m.begin_slot(0).unwrap();
        let mut at = 0usize;
        while at < n {
            let window = at >= upto;
            let end = if window { (at + win).min(n) } else { (at + grid).min(upto) };
            let b = chain(0, at, &ids[at..end]);
            if window {
                m.window_candidates(&b, 1).unwrap();
            } else {
                m.prefill_chunk(&b, path, None).unwrap();
            }
            let mut r = 0usize;
            while r < end - at {
                let k = WINDOW_ROWS.min(end - at - r);
                m.last_rows_logits(r, k, &mut ours).unwrap();
                let p0 = at + r;
                // the dump's rows of this slice, if it kept any
                let (k0, k1) = (p0.max(first_kept), (p0 + k).max(first_kept));
                if k1 > k0 {
                    lf.seek(SeekFrom::Start(((k0 - first_kept) * vocab * 2) as u64)).unwrap();
                    lf.read_exact(&mut theirs[..(k1 - k0) * vocab * 2]).unwrap();
                }
                for j in 0..k {
                    let p = p0 + j;
                    let row: Vec<f32> = ours[j * vocab..(j + 1) * vocab].iter().map(|&b| bf16(b)).collect();
                    s_agree += (argmax(&row) as i64 == want[p] as i64) as usize;
                    if p >= k0 && p < k1 {
                        let at_ref = (p - k0) * vocab * 2;
                        let refrow: Vec<f32> = theirs[at_ref..at_ref + vocab * 2]
                            .chunks_exact(2)
                            .map(|c| f16(u16::from_le_bytes([c[0], c[1]])))
                            .collect();
                        self_agree += (argmax(&refrow) as i64 == want[p] as i64) as usize;
                        let c = cosine(&row, &refrow);
                        s_cos_sum += c;
                        cos_min = cos_min.min(c);
                        s_kept += 1;
                    }
                }
                r += k;
            }
            if window {
                m.commit(0, end - at);
            }
            at = end;
        }
        m.flush().unwrap();
        eprintln!(
            "sample {s}: top-1 {s_agree}/{n} = {:.4}; logits of the last {s_kept} positions: mean cosine {:.6}",
            s_agree as f64 / n as f64,
            s_cos_sum / s_kept.max(1) as f64
        );
        agree += s_agree;
        total += n;
        cos_sum += s_cos_sum;
        kept += s_kept;
    }
    let (top1_rate, cos_mean) = (agree as f64 / total as f64, cos_sum / kept.max(1) as f64);
    eprintln!(
        "ALL: top-1 {agree}/{total} = {top1_rate:.4}; mean cosine {cos_mean:.6} (min {cos_min:.6}) over {kept} positions; \
         the dump's own argmax agrees with its top-1 at {self_agree}/{kept}"
    );
    // a misread layout (prefix for suffix, another vocab) shows up here before it is blamed on the forward
    assert!(self_agree as f64 >= 0.99 * kept as f64, "the dump's logits do not match its own top-1: layout misread");
    assert!(top1_rate >= min_top1, "top-1 agreement {top1_rate:.4} below {min_top1}");
    assert!(cos_mean >= min_cos, "mean logit cosine {cos_mean:.6} below {min_cos}");
}

// ------------------------------------------------------- reference dump readers (host)

fn read_u32s(path: &Path) -> Vec<u32> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn read_i32s(path: &Path) -> Vec<i32> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    raw.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// What `meta.json` of a dump states: the samples, and how many trailing positions the logits files hold.
struct OracleMeta {
    samples: Vec<usize>,
    keep_last: Option<usize>,
}

impl OracleMeta {
    fn read(dir: &Path) -> Self {
        let path = dir.join("meta.json");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        Self::parse(&serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
    }

    fn parse(v: &Value) -> Self {
        let samples: Vec<usize> = v["samples"]
            .as_array()
            .expect("meta.json: samples")
            .iter()
            .map(|s| s["sample"].as_u64().expect("meta.json: samples[].sample") as usize)
            .collect();
        assert!(!samples.is_empty(), "meta.json: no samples");
        // absent or 0: every position
        let keep_last = v["keep_last"].as_u64().map(|k| k as usize).filter(|&k| k > 0);
        Self { samples, keep_last }
    }

    /// Positions of an `n`-token sample whose logits the dump holds (the last ones).
    fn kept(&self, n: usize) -> usize {
        self.keep_last.map_or(n, |k| k.min(n))
    }
}

fn bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// IEEE half to f32 (subnormals, infinities and NaN included).
fn f16(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 if frac == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        e => sign * (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The first index of the largest value (the reference's `argmax`).
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE)
}

/// The dump readers the oracle test rests on, on known values: the meta of both committed dump shapes, half and bf16
/// decoding, and the reference's first-maximum argmax.
#[test]
fn the_reference_dump_readers_decode_known_values() {
    let full = OracleMeta::parse(&serde_json::json!({"tokens": 1024, "samples": [{"sample": 0, "n": 1024}, {"sample": 3}]}));
    assert_eq!((full.samples.clone(), full.kept(1024)), (vec![0, 3], 1024));
    let tail = OracleMeta::parse(&serde_json::json!({"tokens": 8192, "keep_last": 2048, "samples": [{"sample": 1}]}));
    assert_eq!((tail.kept(8192), tail.kept(100)), (2048, 100));
    for (h, x) in [(0x3c00u16, 1.0f32), (0xc000, -2.0), (0x3555, 0.333_251_95), (0x0001, 5.960_464_5e-8), (0x7bff, 65504.0)] {
        assert_eq!(f16(h), x, "{h:#06x}");
    }
    assert!(f16(0x7c00).is_infinite() && f16(0x7e00).is_nan());
    assert_eq!((bf16(0x3f80), bf16(0xc040)), (1.0, -3.0));
    assert_eq!(argmax(&[1.0, 3.0, 3.0, -1.0]), 1);
    assert!((cosine(&[1.0, 2.0], &[2.0, 4.0]) - 1.0).abs() < 1e-12);
}

// ------------------------------------------------------- exact MTP draft-head prefix

/// Check every retained BF16 logit of the model's exact draft-head copy against the original
/// full-vocabulary EXL3 head, not only the winning token. No re-encoding or re-quantization
/// is performed: the strips are a device copy of the same packed weights, so the retained
/// logits must be bit-identical at any row count.
#[test]
#[ignore]
fn lithos_exact_draft_head_prefix() {
    use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
    use gb_cuda::{DevBuf, Stream};
    use std::time::Instant;
    let Some(dir) = real() else { return };
    let ck = gb_exl3::checkpoint::scan(&dir, false).unwrap();
    let name = ck.groups.keys().find(|p| p.ends_with("lm_head")).expect("the pack's LM head group");
    let full = Exl3Weights::from_loaded(&ck.load_group(name).unwrap()).unwrap();
    let sub = full.select_strips(&[(0, MTP_HEAD_N / 128)]).unwrap();
    assert_eq!(sub.n, MTP_HEAD_N.min(full.n));
    let st = Stream::new_nonblocking().unwrap();
    let mut measurements = Vec::new();
    for rows in [1usize, 4, 8] {
        let host: Vec<u16> = (0..rows * full.k)
            .map(|i| {
                let x = (((i * 7919 + 11) % 10007) as f32 / 5003.5) - 1.;
                (x.to_bits() >> 16) as u16
            })
            .collect();
        let x = DevBuf::from_slice(&host).unwrap();
        let mut yf = DevBuf::zeroed(rows * full.n * 2).unwrap();
        let mut ys = DevBuf::zeroed(rows * sub.n * 2).unwrap();
        let mut ws = DevBuf::zeroed(full.workspace_bytes(rows).max(sub.workspace_bytes(rows))).unwrap();
        // DevBuf::zeroed uses default-stream cudaMemset; a nonblocking stream is not
        // ordered behind it, so complete setup before any timed/model operation.
        gb_cuda::rt::device_sync().unwrap();
        full.forward(&x, &mut yf, rows, Dtype::BF16, Dtype::BF16, &mut ws, &st).unwrap();
        sub.forward(&x, &mut ys, rows, Dtype::BF16, Dtype::BF16, &mut ws, &st).unwrap();
        st.sync().unwrap();
        let a: Vec<u16> = yf.to_vec().unwrap();
        let b: Vec<u16> = ys.to_vec().unwrap();
        for r in 0..rows {
            let differences: Vec<_> = a[r * full.n..r * full.n + sub.n]
                .iter()
                .zip(&b[r * sub.n..(r + 1) * sub.n])
                .enumerate()
                .filter(|(_, (x, y))| x != y)
                .take(8)
                .map(|(i, (x, y))| (i, *x, *y))
                .collect();
            assert!(differences.is_empty(), "retained logits differ, row={r}, first mismatches={differences:?}");
        }
        let mut times = [Vec::new(), Vec::new()];
        for round in 0..4 {
            for mode in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                let begin = Instant::now();
                for _ in 0..24 {
                    if mode == 0 {
                        full.forward(&x, &mut yf, rows, Dtype::BF16, Dtype::BF16, &mut ws, &st).unwrap();
                    } else {
                        sub.forward(&x, &mut ys, rows, Dtype::BF16, Dtype::BF16, &mut ws, &st).unwrap();
                    }
                }
                st.sync().unwrap();
                times[mode].push(begin.elapsed().as_secs_f64() * 1000. / 24.);
            }
        }
        for v in times.iter_mut() {
            v.sort_by(f64::total_cmp);
        }
        let f = (times[0][1] + times[0][2]) / 2.;
        let s = (times[1][1] + times[1][2]) / 2.;
        println!("EXACT_HEAD_PREFIX rows={rows} logits_checked={} full_ms={f:.6} prefix_ms={s:.6} speedup={:.4} full_bytes={} prefix_bytes={} exact=true", rows * sub.n, f / s, full.weight_bytes(), sub.weight_bytes());
        measurements.push(serde_json::json!({"rows": rows, "logits_checked": rows * sub.n, "full_ms": f, "prefix_ms": s, "speedup": f / s, "exact": true}));
    }
    let out = std::env::var("REXL3_FN_RESULT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
        .join("draft_head_prefix_exactness.json");
    std::fs::write(
        out,
        serde_json::to_vec_pretty(&serde_json::json!({
            "full_n": full.n,
            "prefix_n": sub.n,
            "full_bytes": full.weight_bytes(),
            "extra_prefix_bytes": sub.weight_bytes(),
            "measurements": measurements,
        }))
        .unwrap(),
    )
    .unwrap();
}

/// The whole-model MTP path with the exact draft-head prefix always on: greedy runs at
/// k=4/7 reproduce byte-identical tokens, draft chains and counters, and the speculative
/// trace stays on the trunk's serial-greedy path (the head's retained logits are separately
/// proven bit-identical to the full head by `lithos_exact_draft_head_prefix`).
#[test]
#[ignore]
fn lithos_mtp_prefix_complete_run() {
    use std::time::Instant;
    let Some(dir) = real() else { return };
    let mut m = FlashNextModel::load(&dir, real_opts()).unwrap();
    if !m.has_mtp() {
        eprintln!("not run: this pack has no MTP head");
        return;
    }
    let total = env_usize("REXL3_FN_PROMPT", 1031);
    let prompt = match std::env::var("REXL3_FN_ORACLE_DIR").ok() {
        Some(o) => {
            let ids = read_u32s(&Path::new(&o).join("ids_0.u32"));
            ids[..total.min(ids.len())].to_vec()
        }
        None => synthetic_prompt(total, m.vocab_size()),
    };
    let grid = m.opts.prefill_rows.min(prompt.len());
    let mut runs = Vec::new();
    for k in [4usize, 7] {
        for rep in 0..3 {
            let start = Instant::now();
            let r = spec_decode(&mut m, &prompt, grid, k, 128);
            let ms = start.elapsed().as_secs_f64() * 1000.;
            println!(
                "MTP_EXACT_AB mode=prefix k={k} rep={rep} total_ms={ms:.3} output={} accepted={} proposed={} rounds={}",
                r.tokens.len(),
                r.accepted,
                r.proposed,
                r.drafts.len()
            );
            runs.push(serde_json::json!({"k": k, "rep": rep, "total_ms": ms, "tokens": r.tokens, "draft_chains": r.drafts, "accepted": r.accepted, "proposed": r.proposed}));
        }
    }
    // byte-identity across reps and k, always-on (no second mode to compare against)
    for pair in runs.windows(2) {
        assert_eq!(pair[0]["tokens"], pair[1]["tokens"], "tokens differ between runs");
        assert_eq!(pair[0]["draft_chains"], pair[1]["draft_chains"], "draft chains differ between runs");
    }
    // the whole speculative trace stays on the trunk's greedy path
    m.begin_slot(0).unwrap();
    let first = prefill_in_grid(&mut m, 0, &prompt, grid, 1);
    let mut serial = vec![top1(&first, 0)];
    let want = runs[0]["tokens"].as_array().unwrap().len();
    while serial.len() < want {
        let j = serial.len() - 1;
        let c = m.window_candidates(&chain(0, prompt.len() + j, &[serial[j]]), 1).unwrap();
        m.commit(0, 1);
        serial.push(top1(&c, 0));
    }
    m.flush().unwrap();
    let got: Vec<u32> = runs[0]["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    assert_eq!(got, serial, "speculative decoding left the trunk's greedy path");
    let p = std::env::var("REXL3_FN_RESULT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
        .join("mtp_head_ab_prefix.json");
    std::fs::write(
        p,
        serde_json::to_vec_pretty(&serde_json::json!({
            "mode": "prefix",
            "prompt": prompt,
            "runs": runs,
            "allocated_bytes": m.allocated_bytes(),
        }))
        .unwrap(),
    )
    .unwrap();
}
