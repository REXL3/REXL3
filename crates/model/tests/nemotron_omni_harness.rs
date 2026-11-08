//! Task I3: the exactness-harness checks (`gb_engine::harness`) through `NemotronBackend` over the REAL `NemotronModel`, and the
//! verify-window step times of the loaded model.
//!
//! Needs `REXL3_NEMO_OMNI_DIR` (the published pack) and a CUDA device; without either it reports what it did not run and passes.
//!
//! * `REXL3_NEMO_HARNESS_LAYERS`: trunk layers to load. Default 8 (the `MEMEMEM*` prefix); `88` or `full` loads the whole model.
//! * `REXL3_NEMO_HARNESS_PARTS`: comma list of `checks` (default), `times` (the step-time table) and `probe` (which prompts make the
//!   copy drafter fire on the loaded model).
//! * `REXL3_NEMO_HARNESS_ONLY`: comma list of check keys (`row_invariance`, `serial_batched`, `chunked_prefill`, `rollback`,
//!   `drafted_copy`, `drafted_replay`; a substring selects every key it is part of, so `drafted` is both); default all.
//! * `REXL3_NEMO_HARNESS_COST`: `measured` runs the engines on the measured step curve instead of a flat one.
//!
//!     REXL3_NEMO_OMNI_DIR=<pack> REXL3_NEMO_HARNESS_LAYERS=8 cargo test --release -p gb-model --test nemotron_omni_harness -- --nocapture
//!
//! The model is loaded once and shared by every check (`Shared`, as `prefill_route.rs` does for the Qwen backend). The harness builds a
//! fresh backend per run; here each of those is the same backend with every slot released, so a run starts from nothing (the real
//! model refuses a slot that has begun until it is ended, which the harness's re-prefill at position 0 does not do). Nothing in a check
//! is relaxed: a failing check is reported with the harness's own message and the backend's latched error, if any, and a failing
//! row-invariance check is followed by a layer-by-layer diagnosis through the model's tap.
//!
//! The wrapper counts what reaches the model (windows, wide windows, kept rows against window rows), so a drafted run can be shown to
//! have verified drafts and committed part of a window, not merely to have matched.

use gb_cuda::{DevBuf, Stream};
use gb_engine::harness::{
    chunked_prefill_equals_whole, drafted_equals_serial, rollback_equals_replay, row_invariance, serial_batched_equals_solo,
    Report,
};
use gb_engine::{Backend, BackendError, Drafter, Engine, EngineConfig, ImageSpan, PrefillStep, Request, RowOutput, RowRequest};
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::{ModelOptions, NemotronBackend};
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::model::{NemotronModel, Point, Tap};
use gb_model::families::nemotron_h_omni::weights::LayerPlan;
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};
use gb_spec::CopyDrafter;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Sequences the engine may hold at once (the model's slots).
const SLOTS: usize = 8;
/// Candidates per row the sampler asks for: top-k 20 plus the sampler's margin.
const COUNT: usize = 28;
/// Positions of one sequence (prompt plus generation) any check reaches.
const MAX_CONTEXT: usize = 2048;
/// KV pages of 256 tokens: a few pages a slot, with room for every slot's partial last page.
const KV_PAGES: usize = 64;
const KV_BYTES_PER_TOKEN: u64 = 8192;
const STATE_BYTES_PER_SLOT: u64 = 170_229_760;

/// Verify-window step time of the full model at `ac7f4fc` (88 layers, one sequence at a 256-token context, median of 5 after 2
/// warmups, node 4, release build): rows against milliseconds. The curve `REXL3_NEMO_HARNESS_COST=measured` hands the engine.
const MEASURED_MS: &[(usize, f64)] =
    &[(1, 269.3), (2, 374.1), (3, 474.0), (4, 574.8), (6, 702.7), (8, 829.0), (16, 1243.0), (32, 1870.9), (64, 2618.2)];

fn pack_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set (the published pack directory)");
    }
    dir
}

fn config(dir: &Path) -> OmniConfig {
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| panic!("config.json: {e}"));
    OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"))
}

/// A deterministic token stream inside the vocabulary and above the special tokens (xorshift64*).
fn tokens(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as u32 % 100_000 + 1000
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------------- the wrapper

/// What reached the model through the wrapper.
#[derive(Clone, Debug, Default)]
struct Counters {
    prefill_calls: u64,
    windows: u64,
    /// Windows with a sequence of two or more rows: a base row and at least one draft.
    wide_windows: u64,
    window_rows: u64,
    max_window_rows: usize,
    commits: u64,
    /// Commits that kept fewer rows than the window had: a draft was rejected and its rows dropped.
    partial_commits: u64,
    rejected_rows: u64,
}

/// One real backend shared by every run of the harness (reloading the pack for each is not an option). It forwards everything the
/// engine and the harness call, so an `Engine<Shared>` is the engine over the real backend, and it counts the windows and commits.
#[derive(Clone)]
struct Shared {
    inner: Rc<RefCell<NemotronBackend<NemotronModel>>>,
    counters: Rc<RefCell<Counters>>,
    pending: Rc<RefCell<BTreeMap<usize, usize>>>,
}

impl Shared {
    fn new(b: NemotronBackend<NemotronModel>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(b)),
            counters: Rc::new(RefCell::new(Counters::default())),
            pending: Rc::new(RefCell::new(BTreeMap::new())),
        }
    }

    /// The backend a harness run starts from: every slot released (the real model refuses a slot that has begun until it is ended, and
    /// the harness's reference runs leave windows and sequences behind), and nothing pending.
    fn fresh(&self) -> Self {
        {
            let mut b = self.inner.borrow_mut();
            for slot in 0..SLOTS {
                b.release(slot);
            }
        }
        self.pending.borrow_mut().clear();
        self.clone()
    }

    /// An error the backend latched or the device reports now, if any (a refused commit is returned by `flush`).
    fn drain(&self) -> Option<String> {
        let mut b = self.inner.borrow_mut();
        let flushed = b.quiesce().err();
        let latched =
            b.take_error().map(|BackendError { fatal, message }| format!("{}{message}", if fatal { "FATAL: " } else { "" }));
        match (latched, flushed) {
            (None, None) => None,
            (a, b) => Some([a, b].into_iter().flatten().collect::<Vec<_>>().join("; ")),
        }
    }

    fn counters(&self) -> Counters {
        self.counters.borrow().clone()
    }
}

impl Backend for Shared {
    fn set_prefill_row_limit(&mut self, rows: Option<usize>) -> bool {
        self.inner.borrow_mut().set_prefill_row_limit(rows)
    }
    fn quiesce(&mut self) -> Result<(), String> {
        self.inner.borrow_mut().quiesce()
    }
    fn vocab_size(&self) -> usize {
        self.inner.borrow().vocab_size()
    }
    fn take_error(&mut self) -> Option<BackendError> {
        self.inner.borrow_mut().take_error()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.counters.borrow_mut().prefill_calls += 1;
        self.inner.borrow_mut().prefill(slot, start, tokens, candidates)
    }
    fn prefill_piece(&mut self, slot: usize, start: u64, tokens: &[u32], total: usize, candidates: usize) -> RowOutput {
        self.counters.borrow_mut().prefill_calls += 1;
        self.inner.borrow_mut().prefill_piece(slot, start, tokens, total, candidates)
    }
    fn prefill_mm(&mut self, slot: usize, start: u64, tokens: &[u32], images: &[ImageSpan], candidates: usize) -> RowOutput {
        self.inner.borrow_mut().prefill_mm(slot, start, tokens, images, candidates)
    }
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> PrefillStep {
        self.counters.borrow_mut().prefill_calls += 1;
        self.inner.borrow_mut().prefill_slice(slot, prompt, images, done, candidates)
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        {
            let mut c = self.counters.borrow_mut();
            c.windows += 1;
            let rows: usize = reqs.iter().map(|r| r.tokens.len()).sum();
            c.window_rows += rows as u64;
            c.max_window_rows = c.max_window_rows.max(rows);
            if reqs.iter().any(|r| r.tokens.len() > 1) {
                c.wide_windows += 1;
            }
            let mut p = self.pending.borrow_mut();
            for r in reqs {
                p.insert(r.slot, r.tokens.len());
            }
        }
        self.inner.borrow_mut().step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        if let Some(rows) = self.pending.borrow_mut().remove(&slot) {
            let mut c = self.counters.borrow_mut();
            c.commits += 1;
            if keep < rows {
                c.partial_commits += 1;
                c.rejected_rows += (rows - keep) as u64;
            }
        }
        self.inner.borrow_mut().commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.pending.borrow_mut().remove(&slot);
        self.inner.borrow_mut().release(slot)
    }
}

// ----------------------------------------------------------------------------------------------------------------------- the engine

/// The engine over the shared backend, with the copy drafter the production engine uses.
fn engine(shared: &Shared, drafting: bool) -> Engine<Shared> {
    engine_with(shared, drafting, Box::new(|| Box::new(CopyDrafter::new(8, 7))))
}

fn engine_with(shared: &Shared, drafting: bool, make: Box<dyn Fn() -> Box<dyn Drafter>>) -> Engine<Shared> {
    let mm = ModelMemory {
        weights_bytes: 0,
        kv_bytes_per_token: KV_BYTES_PER_TOKEN,
        state_bytes_per_slot: STATE_BYTES_PER_SLOT,
        activation_reserve: 0,
    };
    let pool = KV_PAGES as u64 * 256 * KV_BYTES_PER_TOKEN + SLOTS as u64 * STATE_BYTES_PER_SLOT;
    let admission = Admission::new(pool, 0, mm, SLOTS).unwrap().with_max_context(MAX_CONTEXT as u64);
    // A flat step curve, as the engine's own exactness tests use: it makes the budget ask for drafts, which is what a drafted-equals-
    // serial check needs to be about. The first-principles `STEP_MS` of the production engine does not draft at the starting
    // acceptance. `REXL3_NEMO_HARNESS_COST=measured` runs the engine on this model's own measured curve instead (see `MEASURED_MS`).
    let table = match std::env::var("REXL3_NEMO_HARNESS_COST").as_deref() {
        Ok("measured") => MEASURED_MS.to_vec(),
        _ => vec![(1, 70.0), (16, 74.0), (32, 110.0), (64, 190.0)],
    };
    let cost = CostModel::from_table(table);
    Engine::new(shared.fresh(), admission, EngineConfig { max_rows: 64, max_drafts: 7, cost, drafting }, make)
}

/// A drafter that replays what the serial run produced, with a deliberate miss now and then: the verify window gets drafts that are
/// right for a while and then wrong, whatever the model's own tendency to copy. It knows a sequence by the prefix it is asked about.
struct Replay {
    refs: Rc<Vec<Vec<u32>>>,
    miss_every: u64,
}

impl Drafter for Replay {
    fn propose(&mut self, tokens: &[u32], _: &Sampling, max: usize) -> Vec<u32> {
        let n = tokens.len();
        let Some(r) = self.refs.iter().find(|r| r.len() > n && r[..n] == *tokens) else { return Vec::new() };
        (0..max.min(r.len() - n))
            .map(|i| {
                let mix = (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
                let t = r[n + i];
                if (mix ^ (mix >> 29)) % self.miss_every == 0 {
                    t ^ 1
                } else {
                    t
                }
            })
            .collect()
    }
}

fn sampling(seed: u64, temperature: f64) -> Sampling {
    Sampling { seed, temperature, top_k: 20, top_p: 0.95, min_p: 0.0 }
}

fn request(prompt: Vec<u32>, seed: u64, temperature: f64, max_new: usize) -> Request {
    Request { sampling: sampling(seed, temperature), prompt, max_new, stop: vec![], draft: true, images: vec![], cache: true }
}

/// Prompts on both sides of the 64-row slice grid, sampled and greedy, a few tokens each.
fn mixed_requests() -> Vec<Request> {
    [3usize, 40, 63, 70, 129, 190]
        .iter()
        .enumerate()
        .map(|(i, &n)| request(tokens(100 + i as u64, n), 4000 + i as u64, [0.0, 0.7, 1.0][i % 3], 20 + (i % 3) * 6))
        .collect()
}

/// Prompts that repeat: short units cycled and one random block five times. A model that copies a repeat it has seen several times
/// continues it, which is what the copy drafter proposes, so drafts are verified and accepted.
fn repeated_requests() -> Vec<Request> {
    let mut reqs = Vec::new();
    for (i, unit) in [3usize, 6, 12].iter().enumerate() {
        let u = tokens(300 + *unit as u64, *unit);
        reqs.push(request(u.iter().cycle().take(160).copied().collect(), 7000 + i as u64, 0.0, 64));
    }
    for (i, block) in [24usize, 48].iter().enumerate() {
        let u = tokens(900 + *block as u64, *block);
        reqs.push(request(u.iter().cycle().take(block * 5).copied().collect(), 7100 + i as u64, 0.0, 64));
    }
    reqs
}

// ------------------------------------------------------------------------------------------------------------------------ the checks

/// A check's verdict. `Vacuous` is a run that matched but never exercised what it is for.
enum Verdict {
    Pass(String),
    Vacuous(String),
    Fail(String),
}

struct Outcome {
    name: &'static str,
    verdict: Verdict,
}

fn finish(name: &'static str, shared: &Shared, r: Result<String, String>) -> Outcome {
    classify(name, shared, r.map(Verdict::Pass))
}

fn classify(name: &'static str, shared: &Shared, r: Result<Verdict, String>) -> Outcome {
    // a backend that swallowed an error into NaN rows must not read as a pass
    let verdict = match (r, shared.drain()) {
        (Ok(v), None) => v,
        (Ok(Verdict::Pass(m) | Verdict::Vacuous(m)), Some(e)) => {
            Verdict::Fail(format!("the check matched but the backend latched an error: {e} ({m})"))
        }
        (Ok(Verdict::Fail(m)), Some(e)) | (Err(m), Some(e)) => Verdict::Fail(format!("{m} [backend: {e}]")),
        (Err(m), None) => Verdict::Fail(m),
    };
    Outcome { name, verdict }
}

fn report_line(r: &Report) -> String {
    format!("{} sequences, {} tokens, {} rounds, {} drafted, {} accepted", r.sequences, r.tokens, r.rounds, r.drafted, r.accepted)
}

fn wanted(only: &[String], name: &str) -> bool {
    only.is_empty() || only.iter().any(|o| name.contains(o.as_str()))
}

fn run_checks(shared: &Shared, only: &[String]) -> Vec<Outcome> {
    let mut out = Vec::new();
    let make = || shared.fresh();

    // 1. a row's bits in a multi-sequence, multi-row window equal a one-row step's. 1 + 3 + 7 + 16 + 37 = 64 rows in the one window
    // the harness builds, on contexts that end before, on and past a 64-row slice boundary
    if wanted(only, "row_invariance") {
        let t = Instant::now();
        let cases: Vec<(Vec<u32>, Vec<u32>)> = [(5usize, 1usize), (63, 3), (64, 7), (65, 16), (130, 37)]
            .iter()
            .enumerate()
            .map(|(i, &(ctx, win))| (tokens(10 + i as u64, ctx), tokens(50 + i as u64, win)))
            .collect();
        let r = row_invariance(&make, &cases, COUNT)
            .map(|()| format!("{} cases, 64 window rows, {:.1} s", cases.len(), t.elapsed().as_secs_f64()))
            .map_err(|m| {
                // "case {i} row {j} ... differs": say where in the stack, and whether the computation repeats itself
                let at = m.strip_prefix("case ").and_then(|r| {
                    let (i, r) = r.split_once(" row ")?;
                    let (j, _) = r.split_once(' ')?;
                    Some((i.parse::<usize>().ok()?, j.parse::<usize>().ok()?))
                });
                match at {
                    Some((i, j)) if i < cases.len() => format!("{m}\n{}", diagnose(shared, &cases[i].0, &cases[i].1, j)),
                    _ => m,
                }
            });
        out.push(finish("row_invariance", shared, r));
    }

    // 2. N streams decoding together equal each stream alone
    if wanted(only, "serial_batched") {
        let t = Instant::now();
        let reqs = mixed_requests();
        let r = serial_batched_equals_solo(&|d| engine(shared, d), &reqs)
            .map(|rep| format!("{}, {:.1} s", report_line(&rep), t.elapsed().as_secs_f64()));
        out.push(finish("serial_batched_equals_solo", shared, r));
    }

    // 3. a prompt fed in pieces equals one piece: cuts on, off and either side of the slice grid, a piece longer than a slice
    if wanted(only, "chunked_prefill") {
        let t = Instant::now();
        let prompt = tokens(77, 200);
        let s = sampling(5, 0.0);
        let mut r = Ok(String::new());
        let cut_sets: [&[usize]; 7] = [&[1], &[63, 64], &[64], &[65], &[100], &[16, 64, 130], &[199]];
        for cuts in cut_sets {
            if let Err(e) = chunked_prefill_equals_whole(&make, &prompt, cuts, &s, 8) {
                r = Err(format!("cuts {cuts:?}: {e}"));
                break;
            }
        }
        let r =
            r.map(|_| format!("{} cut sets on 200 tokens, 8 decoded each, {:.1} s", cut_sets.len(), t.elapsed().as_secs_f64()));
        out.push(finish("chunked_prefill_equals_whole", shared, r));
    }

    // 3b. slices of 64, 128 and 256 rows (and odd widths) over the same prompt: the last row's candidates and the next rows decoded after
    // it are the same bits. The 256-row slices run the MoE layers through the large-row prefill path, the 64-row ones through the decode
    // path (the cuts of a prompt sit on the slice grid, so no two of these cut it alike); pieces of a prompt add their own cuts.
    if wanted(only, "slice_sizes") {
        let t = Instant::now();
        let cand = sampling(9, 0.0).candidate_count(shared.vocab_size());
        let mut r: Result<String, String> = Ok(String::new());
        let mut cases = 0usize;
        let wide = gb_model::families::nemotron_h_omni::model::prefill_rows_cap() > 256;
        'sizes: for (n, seed) in [(300usize, 101u64), (600, 102), (257, 103), (1100, 104)] {
            let prompt = tokens(seed, n);
            let forced = tokens(seed + 50, 4);
            // run one configuration: (row limit, pieces); the last prefill row and four teacher-forced decode rows
            let run = |limit: Option<usize>, cuts: &[usize]| -> Vec<RowOutput> {
                let mut b = shared.fresh();
                b.set_prefill_row_limit(limit);
                let (mut rows, mut at, mut last) = (Vec::new(), 0usize, None);
                for &c in cuts.iter().chain(std::iter::once(&n)) {
                    last = Some(b.prefill_piece(0, at as u64, &prompt[at..c], n, cand));
                    at = c;
                }
                rows.push(last.unwrap());
                let mut pos = n as u64;
                for &tok in &forced {
                    let o = b.step(&[RowRequest { slot: 0, tokens: vec![tok], first_position: pos }], cand);
                    b.commit(0, 1);
                    pos += 1;
                    rows.push(o[0][0].clone());
                }
                b.release(0);
                rows
            };
            let reference = run(Some(64), &[]);
            let configs: Vec<(Option<usize>, Vec<usize>)> = vec![
                (Some(128), vec![]),
                (None, vec![]),
                (Some(100), vec![]),
                (None, vec![1, 64, 65]),
                (None, vec![255]),
                (Some(256), vec![100, 257.min(n - 1)]),
                (Some(77), vec![n / 3, 2 * n / 3]),
            ];
            // at the 512-row default (not under the opt-out REXL3_NEMO_PREFILL_ROWS_MAX=256) `None` is a 512-row slice: keep 256 explicit and add cuts
            // on either side of the 512 grid
            let mut configs = configs;
            // contract P2: slices of 1,024 and the whole prompt in one slice (cap 4,096), and cuts either side of 1,024
            if gb_model::families::nemotron_h_omni::contract::current()
                == gb_model::families::nemotron_h_omni::contract::PrefillContract::P2
            {
                configs.push((Some(1024), vec![]));
                configs.push((Some(1024), vec![1023.min(n - 1)]));
                configs.push((None, vec![1025.min(n - 1)]));
                configs.push((Some(4096), vec![]));
            }
            if wide {
                configs.push((Some(256), vec![]));
                configs.push((None, vec![511.min(n - 1)]));
                configs.push((Some(512), vec![100, 513.min(n - 1)]));
                configs.push((Some(300), vec![]));
            }
            for (limit, cuts) in configs {
                cases += 1;
                let got = run(limit, &cuts);
                for (i, (a, b)) in reference.iter().zip(&got).enumerate() {
                    if !a.bits_eq(b) {
                        r = Err(format!(
                            "{n} tokens, row limit {limit:?}, cuts {cuts:?}: {} differs from 64-row slices",
                            if i == 0 { "the last prefill row".to_string() } else { format!("decoded row {i}") }
                        ));
                        break 'sizes;
                    }
                }
            }
        }
        shared.inner.borrow_mut().set_prefill_row_limit(None);
        let widest = gb_model::families::nemotron_h_omni::contract::current()
            .max_prefill_rows(gb_model::families::nemotron_h_omni::model::prefill_rows_cap());
        let r = r.map(|_| {
            format!(
                "{cases} configurations on 4 prompts (257 to 1100 tokens, slice cap {widest}), {:.1} s",
                t.elapsed().as_secs_f64()
            )
        });
        out.push(finish("slice_sizes_agree", shared, r));
    }

    // 4. after a partly rejected window the next rows equal a clean replay
    // P2 probe (lane S; only when asked by name: REXL3_NEMO_HARNESS_ONLY=p2probe): the last prefill row's 64 candidates (ids and
    // logits) of every prefix of the oracle's real texts at a stride, each prefix prefilled fresh, written to REXL3_P2PROBE_OUT; run
    // once per contract (the contract is process-wide) and compare the files (tools/p2_probe_compare.py: top-1, top-k overlap, logit
    // deltas, KL on the candidates' support against the fork's dumped logits).
    if only.iter().any(|s| s == "p2probe") {
        let t = Instant::now();
        let path = std::env::var("REXL3_P2PROBE_OUT").unwrap_or_else(|_| "p2probe.txt".into());
        let stride = std::env::var("REXL3_P2PROBE_STRIDE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(48);
        let dir = std::env::var("REXL3_NEMO_ORACLE_DIR").expect("p2probe needs REXL3_NEMO_ORACLE_DIR");
        let mut lines = Vec::new();
        for text in 0..2 {
            let raw = std::fs::read(format!("{dir}/ids_{text}.u32")).expect("ids file");
            let ids: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let mut p = stride;
            while p <= ids.len() {
                let mut b = shared.fresh();
                let o = b.prefill(0, 0, &ids[..p], 64);
                b.release(0);
                let idt: Vec<String> = o.ids.iter().map(|v| v.to_string()).collect();
                let vt: Vec<String> = o.values.iter().map(|v| format!("{:08x}", v.to_bits())).collect();
                lines.push(format!("{text} {} {} | {}", p - 1, idt.join(","), vt.join(",")));
                p += stride;
            }
        }
        std::fs::write(&path, lines.join("\n") + "\n").expect("p2probe out");
        let r: Result<String, String> = Ok(format!("{} rows to {path}, {:.1} s", lines.len(), t.elapsed().as_secs_f64()));
        out.push(finish("p2probe", shared, r));
    }

    // P2 routing probe (lane S; only by name: REXL3_NEMO_HARNESS_ONLY=p2route): the 22 picks of every row of every MoE layer while the
    // oracle's two real texts are prefilled whole (fresh slot), written to REXL3_P2ROUTE_OUT as `text layer row e0,e1,..` (in pick order);
    // run once per contract and compare (tools/p2_route_compare.py: per layer, the rows whose pick SET differs).
    if only.iter().any(|s| s == "p2route") {
        use gb_model::families::nemotron_h_omni::tp_balance;
        let t = Instant::now();
        let path = std::env::var("REXL3_P2ROUTE_OUT").unwrap_or_else(|_| "p2route.txt".into());
        let dir = std::env::var("REXL3_NEMO_ORACLE_DIR").expect("p2route needs REXL3_NEMO_ORACLE_DIR");
        let mut lines = Vec::new();
        for text in 0..2 {
            let raw = std::fs::read(format!("{dir}/ids_{text}.u32")).expect("ids file");
            let ids: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let mut b = shared.fresh();
            tp_balance::tap_take();
            tp_balance::tap_enable(true);
            let _ = b.prefill(0, 0, &ids, 8);
            // take before switching the tap off (switching it off clears what it holds)
            let taken = tp_balance::tap_take();
            tp_balance::tap_enable(false);
            b.release(0);
            let mut next_row: std::collections::BTreeMap<usize, usize> = Default::default();
            for (layer, rows, picks) in taken {
                let at = next_row.entry(layer).or_insert(0);
                for r in 0..rows {
                    let p: Vec<String> = picks[r * 22..(r + 1) * 22].iter().map(|v| v.to_string()).collect();
                    lines.push(format!("{text} {layer} {} {}", *at + r, p.join(",")));
                }
                *at += rows;
            }
        }
        std::fs::write(&path, lines.join("\n") + "\n").expect("p2route out");
        let r: Result<String, String> = Ok(format!("{} rows to {path}, {:.1} s", lines.len(), t.elapsed().as_secs_f64()));
        out.push(finish("p2route", shared, r));
    }

    if wanted(only, "rollback") {
        let t = Instant::now();
        let ctx = tokens(31, 70);
        let mut r = Ok(String::new());
        let mut n = 0;
        'outer: for rows in [2usize, 8, 33, 64] {
            let window = tokens(40 + rows as u64, rows);
            let keeps: Vec<usize> = if rows <= 8 { (1..=rows).collect() } else { vec![1, 2, rows / 2, rows - 1, rows] };
            for keep in keeps {
                n += 1;
                if let Err(e) = rollback_equals_replay(&make, &ctx, &window, keep, &tokens(60, 3), COUNT) {
                    r = Err(format!("window {rows} keep {keep}: {e}"));
                    break 'outer;
                }
            }
        }
        let r = r.map(|_| format!("{n} (window, keep) cases up to 64 rows, {:.1} s", t.elapsed().as_secs_f64()));
        out.push(finish("rollback_equals_replay", shared, r));
    }

    // 5. drafting on, streams together, equals drafting off, each alone, with the copy drafter. Prompts that repeat. The check is
    // vacuous unless drafts were verified and some accepted, so the counters are held to it
    if wanted(only, "drafted_copy") {
        let t = Instant::now();
        let before = shared.counters();
        let reqs = repeated_requests();
        let r = drafted_equals_serial(&|d| engine(shared, d), &reqs);
        let after = shared.counters();
        let r = r.map(|rep| {
            let (wide, partial) = (after.wide_windows - before.wide_windows, after.partial_commits - before.partial_commits);
            let line = format!(
                "{}, {wide} wide windows, {partial} partial commits ({} rows dropped), {:.1} s",
                report_line(&rep),
                after.rejected_rows - before.rejected_rows,
                t.elapsed().as_secs_f64()
            );
            if rep.drafted == 0 || rep.accepted == 0 || wide == 0 {
                Verdict::Vacuous(line)
            } else {
                Verdict::Pass(line)
            }
        });
        out.push(classify("drafted_equals_serial (copy drafter)", shared, r));
    }

    // 6. the same check with a drafter that is right for a while and then wrong, whatever the model's tendency to copy: verify windows
    // that are committed in part, on the real model. Held to partial commits.
    if wanted(only, "drafted_replay") {
        let t = Instant::now();
        // the serial outputs the drafter replays: prompt plus what each sequence produced alone
        let reqs = mixed_requests();
        let mut refs = Vec::new();
        let mut serial_error = None;
        for r in &reqs {
            let mut e = engine(shared, false);
            if let Err(m) = e.submit(r.clone()).map_err(|er| er.to_string()).and_then(|_| e.run().map_err(|er| er.to_string())) {
                serial_error = Some(m);
                break;
            }
            let mut full = r.prompt.clone();
            full.extend(e.take_outputs().remove(0).tokens);
            refs.push(full);
        }
        let r = match serial_error {
            Some(m) => Err(format!("the serial reference runs failed: {m}")),
            None => {
                let refs = Rc::new(refs);
                let before = shared.counters();
                let factory = move || -> Box<dyn Fn() -> Box<dyn Drafter>> {
                    let refs = refs.clone();
                    Box::new(move || Box::new(Replay { refs: refs.clone(), miss_every: 6 }))
                };
                let r = drafted_equals_serial(&|d| engine_with(shared, d, factory()), &reqs);
                let after = shared.counters();
                r.map(|rep| {
                    let (wide, partial) =
                        (after.wide_windows - before.wide_windows, after.partial_commits - before.partial_commits);
                    let line = format!(
                        "{}, {wide} wide windows, {partial} partial commits ({} rows dropped), {:.1} s",
                        report_line(&rep),
                        after.rejected_rows - before.rejected_rows,
                        t.elapsed().as_secs_f64()
                    );
                    if rep.drafted == 0 || rep.accepted == 0 || partial == 0 {
                        Verdict::Fail(format!("vacuous: the replay drafter drafted, accepted or was rejected nothing ({line})"))
                    } else {
                        Verdict::Pass(line)
                    }
                })
            }
        };
        out.push(classify("drafted_equals_serial (replay drafter)", shared, r));
    }
    out
}

// ------------------------------------------------------------------------------------------------------------------- the diagnosis

/// One tapped buffer.
struct Cap {
    point: Point,
    rows: usize,
    width: usize,
    data: Vec<f32>,
}

#[derive(Clone)]
struct CaptureTap(Arc<Mutex<Vec<Cap>>>);

impl Tap for CaptureTap {
    fn tap(&mut self, point: Point, buf: &DevBuf, rows: usize, width: usize, st: &Stream) -> Result<(), ModelError> {
        st.sync()?;
        let mut v = vec![0f32; rows * width];
        buf.download(&mut v)?;
        self.0.lock().unwrap().push(Cap { point, rows, width, data: v });
        Ok(())
    }
}

/// Every tapped buffer of one window: a fresh slot 0, `ctx` prefilled, `prior` single-row steps each committed, then `rows` as one
/// window with the tap on. The tap sees only that last window.
fn capture_window(shared: &Shared, ctx: &[u32], prior: &[u32], rows: &[u32]) -> Vec<Cap> {
    let mut b = shared.fresh();
    b.prefill(0, 0, ctx, COUNT);
    let mut pos = ctx.len() as u64;
    for &t in prior {
        b.step(&[RowRequest { slot: 0, tokens: vec![t], first_position: pos }], COUNT);
        b.commit(0, 1);
        pos += 1;
    }
    let seen = Arc::new(Mutex::new(Vec::new()));
    shared.inner.borrow_mut().m.set_tap(Some(Box::new(CaptureTap(seen.clone()))));
    b.step(&[RowRequest { slot: 0, tokens: rows.to_vec(), first_position: pos }], COUNT);
    let _ = b.quiesce();
    shared.inner.borrow_mut().m.set_tap(None);
    b.release(0);
    let v = std::mem::take(&mut *seen.lock().unwrap());
    v
}

fn bits_of(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Where row `row` of the window `win` (after `ctx`) first differs between the window run whole and the same row run alone, layer by
/// layer; and whether either computation differs from itself on a repeat, which is what a nondeterminism would be.
fn diagnose(shared: &Shared, ctx: &[u32], win: &[u32], row: usize) -> String {
    let whole = capture_window(shared, ctx, &[], win);
    let whole2 = capture_window(shared, ctx, &[], win);
    let alone = capture_window(shared, ctx, &win[..row], &win[row..=row]);
    let alone2 = capture_window(shared, ctx, &win[..row], &win[row..=row]);
    let mut out = Vec::new();
    let repeat = |a: &[Cap], b: &[Cap], what: &str, out: &mut Vec<String>| {
        let first = a.iter().zip(b).position(|(x, y)| x.point != y.point || bits_of(&x.data) != bits_of(&y.data));
        match first {
            None => out.push(format!("{what}: identical on a repeat ({} taps)", a.len())),
            Some(k) => out.push(format!("{what}: DIFFERS ON A REPEAT, first at {:?}: nondeterminism", a[k].point)),
        }
    };
    repeat(&whole, &whole2, &format!("the {}-row window", win.len()), &mut out);
    repeat(&alone, &alone2, &format!("row {row} alone"), &mut out);
    if whole.len() != alone.len() {
        out.push(format!("the runs tapped {} and {} points", whole.len(), alone.len()));
        return out.join("\n");
    }
    for (w, a) in whole.iter().zip(&alone) {
        if w.point != a.point {
            out.push(format!("taps out of step: {:?} against {:?}", w.point, a.point));
            break;
        }
        let width = w.width;
        let wrow = &w.data[row * width..(row + 1) * width];
        let arow = &a.data[..width];
        let differ: Vec<usize> = (0..width).filter(|&i| wrow[i].to_bits() != arow[i].to_bits()).collect();
        if !differ.is_empty() {
            let max = differ.iter().map(|&i| (wrow[i] - arow[i]).abs()).fold(0f32, f32::max);
            let scale = arow.iter().fold(0f32, |m, x| m.max(x.abs()));
            out.push(format!(
                "first difference at {:?} (rows {} against {}): {} of {width} elements, max |diff| {max:e} (row max |x| {scale:e}), \
                 first at {} ({:e} against {:e})",
                w.point,
                w.rows,
                a.rows,
                differ.len(),
                differ[0],
                wrow[differ[0]],
                arow[differ[0]]
            ));
            return out.join("\n");
        }
    }
    out.push(format!("row {row}: every tapped point is bit-identical whole and alone; the difference is after the head"));
    out.join("\n")
}

// ------------------------------------------------------------------------------------------------------------- the probe and times

/// Which prompt shapes make the copy drafter fire on the loaded model: one drafting engine over candidate prompts, the counters of each
/// sequence. A probe for choosing the prompts of the drafted check, not a check.
fn probe_drafts(shared: &Shared) {
    let mut reqs = Vec::new();
    let mut names = Vec::new();
    for unit in [1usize, 2, 3, 4, 6, 8, 12] {
        let u = tokens(300 + unit as u64, unit);
        reqs.push(request(u.iter().cycle().take(160).copied().collect(), 1, 0.0, 96));
        names.push(format!("unit {unit} x 160"));
    }
    let block = 48usize;
    let u = tokens(900 + block as u64, block);
    reqs.push(request(u.iter().cycle().take(block * 5).copied().collect(), 1, 0.0, 96));
    names.push(format!("block {block} x 5"));
    let mut e = engine(shared, true);
    for (i, r) in reqs.iter().enumerate() {
        e.submit(r.clone()).unwrap_or_else(|er| panic!("{}: {er}", names[i]));
    }
    e.run().unwrap_or_else(|er| panic!("probe run: {er}"));
    for (o, name) in e.take_outputs().iter().zip(&names) {
        eprintln!(
            "probe {name:<18} rounds {:>3} drafted {:>3} accepted {:>3} first tokens {:?}",
            o.rounds,
            o.drafted,
            o.accepted,
            &o.tokens[..o.tokens.len().min(10)]
        );
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// The step time of a verify window of `rows` rows of one sequence at a 256-token context: median of 5 after 2 warmups, each on fresh
/// token ids, host call to candidates on the host (the step ends with the stream synchronised).
fn step_times(shared: &Shared) -> Vec<(usize, f64, f64)> {
    let mut b = shared.fresh();
    let ctx = tokens(5, 256);
    let row = b.prefill(0, 0, &ctx, COUNT);
    assert!(row.values.iter().all(|v| v.is_finite()), "the context's prefill gave a non-finite row");
    let mut table = Vec::new();
    for rows in [1usize, 2, 3, 4, 6, 8, 16, 32, 64] {
        let mut ms = Vec::new();
        for it in 0..7 {
            let toks = tokens(1000 + rows as u64 * 16 + it, rows);
            let req = [RowRequest { slot: 0, tokens: toks, first_position: 256 }];
            let t = Instant::now();
            let out = b.step(&req, COUNT);
            let dt = t.elapsed().as_secs_f64() * 1e3;
            assert_eq!(out[0].len(), rows);
            assert!(out[0].iter().all(|r| r.values.iter().all(|v| v.is_finite())), "{rows} rows: a non-finite candidate");
            if it >= 2 {
                ms.push(dt);
            }
        }
        let (lo, hi) = (ms.iter().cloned().fold(f64::MAX, f64::min), ms.iter().cloned().fold(0.0, f64::max));
        let med = median(&mut ms);
        table.push((rows, med, hi - lo));
    }
    b.release(0);
    table
}

// -------------------------------------------------------------------------------------------------------------------------- the test

#[test]
fn the_harness_checks_run_through_the_backend_over_the_real_model() {
    let Some(dir) = pack_dir() else { return };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let layers = match std::env::var("REXL3_NEMO_HARNESS_LAYERS").as_deref() {
        Ok("88") | Ok("full") => None,
        Ok(n) => Some(n.parse::<usize>().unwrap_or_else(|_| panic!("REXL3_NEMO_HARNESS_LAYERS: `{n}`"))),
        Err(_) => Some(8),
    };
    let parts = std::env::var("REXL3_NEMO_HARNESS_PARTS").unwrap_or_else(|_| "checks".into());
    let only: Vec<String> = std::env::var("REXL3_NEMO_HARNESS_ONLY")
        .map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default();

    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    // 256 rows a prefill slice: slices above a window's 64 rows run the MoE layers through the large-row prefill path
    // (512 by default, 256 under the opt-out REXL3_NEMO_PREFILL_ROWS_MAX=256: slice_sizes_agree then also runs 512-row slices against 64)
    // under contract P2 the slice cap is P2's (4,096): the model is loaded for it, so the P2 invariance gates reach the widths P2 serves
    let prefill_rows = gb_model::families::nemotron_h_omni::contract::current()
        .max_prefill_rows(gb_model::families::nemotron_h_omni::model::prefill_rows_cap());
    let opts = ModelOptions { slots: SLOTS, kv_pages: KV_PAGES, max_context: MAX_CONTEXT, prefill_rows, layers, mtp: false };
    let t = Instant::now();
    let model = match NemotronModel::load(&dir, &cfg, &plan, opts.clone()) {
        Ok(m) => m,
        Err(ModelError::NotBuilt(what)) => {
            eprintln!("not run: {what} is not built yet");
            return;
        }
        Err(ModelError::Cuda(e)) if e.0 == 801 => {
            eprintln!("not run: a kernel the model needs is still a stub");
            return;
        }
        Err(e) => panic!("load: {e}"),
    };
    let n_layers = layers.unwrap_or(cfg.pattern.chars().count());
    let (free, total) = gb_cuda::rt::mem_info().unwrap_or((0, 0));
    eprintln!(
        "loaded {n_layers} layers in {:.1} s; device free {:.1} of {:.1} GiB",
        t.elapsed().as_secs_f64(),
        free as f64 / (1u64 << 30) as f64,
        total as f64 / (1u64 << 30) as f64
    );
    let backend = NemotronBackend::with_runtime(model, 256, &opts).unwrap_or_else(|e| panic!("backend: {e}"));
    let shared = Shared::new(backend);

    if parts.contains("probe") {
        probe_drafts(&shared);
        if let Some(e) = shared.drain() {
            panic!("the probe latched an error: {e}");
        }
    }
    let mut failed = Vec::new();
    if parts.contains("checks") {
        for o in run_checks(&shared, &only) {
            let label = format!("{} ({n_layers} layers)", o.name);
            match o.verdict {
                Verdict::Pass(m) => eprintln!("PASS    {label:<54} {m}"),
                Verdict::Vacuous(m) => {
                    eprintln!("VACUOUS {label:<54} {m}");
                    // a prefix of the trunk is not a language model: nothing it writes repeats. The full model must not be vacuous.
                    if layers.is_none() {
                        failed.push(o.name);
                    }
                }
                Verdict::Fail(m) => {
                    eprintln!("FAIL    {label:<54} {m}");
                    failed.push(o.name);
                }
            }
        }
        eprintln!("model calls: {:?}", shared.counters());
    }
    if parts.contains("times") {
        let table = step_times(&shared);
        eprintln!("verify-window step time, {n_layers} layers, 256-token context, one sequence (median of 5 after 2 warmups):");
        eprintln!("  rows       ms  spread_ms   ms/row");
        for (rows, ms, spread) in &table {
            eprintln!("  {rows:>4} {ms:>8.1} {spread:>10.1} {:>8.2}", ms / *rows as f64);
        }
        if let Some(e) = shared.drain() {
            panic!("the timing runs latched an error: {e}");
        }
    }
    assert!(failed.is_empty(), "harness checks failed or were vacuous on {n_layers} layers: {failed:?}");
}
