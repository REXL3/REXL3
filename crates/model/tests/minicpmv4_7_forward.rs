//! The MiniCPM-V-4.7 text forward, backend and engine on a real pack (E3 of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`).
//!
//! `REXL3_MINICPM_DIR` names the EXL3 pack: the converter's 8-layer smoke pack (its config says 8 layers, so the boot is asked for the
//! truncated layer count) or the full 40-layer pack. Every test is `#[ignore]` (GPU) and prints `not run:` without it. One model is
//! loaded per test: run with `--test-threads=1` under `flock -s /tmp/rexl3-gpu.lock`.
//!
//! ```text
//! flock -s -w 1800 /tmp/rexl3-gpu.lock env REXL3_MINICPM_DIR=~/models/minicpm-v-4.7-35b-a3b/work/smokeB/o \
//!     cargo test --release -p gb-model --test minicpmv4_7_forward -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! What runs (names are the plan's, the invariants are the repo's): `row_invariance`, `serial_batched_equals_solo`,
//! `drafted_equals_serial` (the copy drafter of the engine core), `chunked_prefill_equals_whole` (chunk grids 1 to 4096, odd grids,
//! the `PrefillPath` rule keyed on the prompt total), `a_window_of_n_rows_equals_n_serial_rows`, `rollback_equals_replay` (up to 16
//! rows a stream, four streams, every keep), `graph_replay_equals_eager`, `failure_isolation`, `the_pad_never_reaches_a_consumer`
//! (the poison seam through top-k, the model's logits export and the engine's sampler), the seams of the later slices (explicit
//! M-RoPE triples, a rope delta, a feature splice), the plan == allocations of the working set, and the oracle against the exllamav3
//! fork's forward of the same pack (`REXL3_MINICPM_ORACLE_DIR`, `REXL3_MINICPM_IDS`).

use gb_cuda::DevBuf;
use gb_engine::harness::{
    chunked_prefill_equals_whole, drafted_equals_serial, rollback_equals_replay, row_invariance, serial_batched_equals_solo,
};
use gb_engine::{Backend, Drafter, Engine, EngineConfig, NoDrafts, Request, RowOutput, RowRequest};
use gb_model::families::minicpmv4_7::backend::MiniCpmBackend;
use gb_model::families::minicpmv4_7::boot::{boot_profile, boot_with, BootExtras, BootPlan, LoadedModel};
use gb_model::families::minicpmv4_7::config::MiniCpmConfig;
use gb_model::families::minicpmv4_7::engine::STEP_MS;
use gb_model::families::minicpmv4_7::model::{MiniCpmModel, ModelOptions, WINDOW_ROWS};
use gb_model::forward::{Candidates, PrefillPath, SpliceOp};
use gb_model::rows::RowBatch;
use gb_model::serve::ServeOptions;
use gb_sampling::Sampling;
use gb_sched::{CostModel, MemoryProfile, ModelMemory};
use gb_spec::CopyDrafter;
use std::cell::RefCell;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::rc::Rc;

// ------------------------------------------------------------------------------------------------------------------ the rig

fn pack_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_MINICPM_DIR") {
        Ok(d) if Path::new(&d).join("config.json").exists() => {
            if !gb_cuda::device_available() {
                eprintln!("not run: no CUDA device");
                return None;
            }
            Some(PathBuf::from(d))
        }
        _ => {
            eprintln!("not run: REXL3_MINICPM_DIR is not set to a pack");
            None
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// One backend, shared by the engines and harness calls of a test (loading the pack is the cost; every call starts from released slots).
#[derive(Clone)]
struct Shared(Rc<RefCell<MiniCpmBackend>>);

impl Backend for Shared {
    fn vocab_size(&self) -> usize {
        self.0.borrow().vocab_size()
    }
    fn take_error(&mut self) -> Option<gb_engine::BackendError> {
        self.0.borrow_mut().take_error()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.0.borrow_mut().prefill(slot, start, tokens, candidates)
    }
    fn prefill_piece(&mut self, slot: usize, start: u64, tokens: &[u32], total: usize, candidates: usize) -> RowOutput {
        self.0.borrow_mut().prefill_piece(slot, start, tokens, total, candidates)
    }
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[gb_engine::ImageSpan],
        done: usize,
        candidates: usize,
    ) -> gb_engine::PrefillStep {
        self.0.borrow_mut().prefill_slice(slot, prompt, images, done, candidates)
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.0.borrow_mut().step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.0.borrow_mut().commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.0.borrow_mut().release(slot)
    }
    fn set_prefill_row_limit(&mut self, rows: Option<usize>) -> bool {
        self.0.borrow_mut().set_prefill_row_limit(rows)
    }
}

struct Rig {
    backend: Rc<RefCell<MiniCpmBackend>>,
    plan: BootPlan,
    profile: MemoryProfile,
    cfg: MiniCpmConfig,
    slots: usize,
    max_ctx: usize,
    /// Embedding rows of the tokens a test asks for before the weights moved into the model: `token -> bf16 row`.
    embed_rows: Vec<(u32, Vec<u16>)>,
}

const SPLICE_TOKENS: [u32; 6] = [1234, 5678, 9012, 3456, 7890, 2345];

impl Rig {
    fn load(slots: usize, max_ctx: usize, prefill_rows: usize) -> Option<Rig> {
        let dir = pack_dir()?;
        let cfg = MiniCpmConfig::from_dir(&dir).unwrap();
        let mut o = ServeOptions::new(dir);
        o.slots = slots;
        o.slots_auto = false;
        o.allow_unvalidated_pack = true;
        // the prefix cache has its own rig (`tests/minicpmv4_7_prefix.rs`); `auto` would take up to 8 GiB
        o.prefix_cache = gb_model::serve::PrefixCacheOption::Off;
        o.max_model_len = max_ctx;
        o.prefill_chunk = prefill_rows;
        let t = std::time::Instant::now();
        let loaded =
            boot_with(&o, BootExtras { allow_truncated_layers: cfg.layers < 40 }).unwrap_or_else(|e| panic!("boot: {e}"));
        let (profile, _) = boot_profile(&loaded.cfg, &o).unwrap();
        let LoadedModel { cfg, weights, state, plan, .. } = loaded;
        let h = cfg.hidden;
        let embed_rows = SPLICE_TOKENS
            .iter()
            .map(|&t| {
                let mut row = vec![0u16; h];
                weights.embed.download_range(t as usize * h * 2, &mut row).unwrap();
                (t, row)
            })
            .collect();
        let opts = ModelOptions { slots: plan.slots, kv_pages: plan.kv_pages(), max_context: plan.max_context, prefill_rows };
        let model = MiniCpmModel::new(cfg.clone(), weights, state, opts).unwrap_or_else(|e| panic!("model: {e}"));
        eprintln!(
            "loaded {} layers, {} slots, {} pages, context {}, prefill rows {} in {:.1} s; working set {:?}",
            cfg.layers,
            plan.slots,
            plan.kv_pages(),
            plan.max_context,
            prefill_rows,
            t.elapsed().as_secs_f64(),
            model.bytes()
        );
        let backend = MiniCpmBackend::new(model, prefill_rows);
        Some(Rig {
            backend: Rc::new(RefCell::new(backend)),
            max_ctx: plan.max_context,
            slots: plan.slots,
            plan,
            profile,
            cfg,
            embed_rows,
        })
    }

    fn vocab(&self) -> usize {
        self.cfg.vocab
    }

    /// A handle on the one backend with every slot released and no error pending.
    fn shared(&self) -> Shared {
        let mut b = self.backend.borrow_mut();
        for s in 0..self.slots {
            b.release(s);
        }
        let _ = b.take_error();
        drop(b);
        Shared(self.backend.clone())
    }

    fn engine(&self, drafting: bool) -> Engine<Shared> {
        let mm = ModelMemory::from_profile(&self.profile, 0, 0);
        let admission = self.plan.memory.admission(mm, self.slots, self.max_ctx).unwrap();
        let make: Box<dyn Fn() -> Box<dyn Drafter>> =
            if drafting { Box::new(|| Box::new(CopyDrafter::new(8, 7))) } else { Box::new(|| Box::new(NoDrafts)) };
        Engine::new(
            self.shared(),
            admission,
            EngineConfig { max_rows: WINDOW_ROWS, max_drafts: 7, cost: CostModel::from_table(STEP_MS.to_vec()), drafting },
            make,
        )
    }

    fn with_model<R>(&self, f: impl FnOnce(&mut MiniCpmModel) -> R) -> R {
        f(&mut self.backend.borrow_mut().m)
    }
}

/// Real token ids when an ids set is given (`REXL3_MINICPM_IDS`, a `make_kl_ids.py` file: int32 `[rows, T]`), else a deterministic
/// synthetic stream inside the vocabulary.
fn real_ids() -> Option<(usize, usize, Vec<i32>)> {
    let path = std::env::var("REXL3_MINICPM_IDS").ok().map(PathBuf::from).or_else(|| {
        let p = PathBuf::from(std::env::var("HOME").ok()?).join("projects/minicpm-kl/ids/legA.safetensors");
        p.exists().then_some(p)
    })?;
    Some(read_ids(&path))
}

/// `input_ids` of a safetensors file: (rows, tokens, ids).
fn read_ids(path: &Path) -> (usize, usize, Vec<i32>) {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let head: serde_json::Value = serde_json::from_slice(&raw[8..8 + n]).unwrap();
    let t = &head["input_ids"];
    assert_eq!(t["dtype"], "I32", "{}: input_ids is not int32", path.display());
    let shape: Vec<usize> = t["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let (a, b) = (t["data_offsets"][0].as_u64().unwrap() as usize, t["data_offsets"][1].as_u64().unwrap() as usize);
    let ids = raw[8 + n + a..8 + n + b].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    (shape[0], shape[1], ids)
}

/// `n` tokens: a window of the real ids set (row `seed`, from `offset`) or a synthetic stream. Never past the vocabulary.
fn prompt(n: usize, seed: usize, vocab: usize) -> Vec<u32> {
    if let Some((rows, t, ids)) = real_ids() {
        if n <= t {
            let r = seed % rows;
            let off = (seed * 131) % (t - n + 1);
            return ids[r * t + off..r * t + off + n].iter().map(|&x| x as u32).collect();
        }
    }
    (0..n).map(|i| ((i as u64 * 2_654_435_761 + seed as u64 * 40_503) % (vocab as u64 - 16) + 8) as u32).collect()
}

fn window_tokens(n: usize, seed: usize, vocab: usize) -> Vec<u32> {
    (0..n).map(|j| ((j as u64 * 7 + seed as u64 * 977 + 13) % (vocab as u64 - 16) + 8) as u32).collect()
}

fn greedy() -> Sampling {
    Sampling { seed: 0, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 }
}

fn sampled(seed: u64) -> Sampling {
    Sampling { seed, temperature: 0.8, top_k: 20, top_p: 0.95, min_p: 0.0 }
}

fn request(prompt: Vec<u32>, sampling: Sampling, max_new: usize, draft: bool) -> Request {
    Request { prompt, sampling, max_new, stop: vec![], draft, images: vec![], cache: false }
}

fn chain(slot: usize, first: usize, tokens: &[u32]) -> RowBatch {
    RowBatch::from_requests(&[RowRequest { slot, first_position: first as u64, tokens: tokens.to_vec() }])
}

fn top1(c: &Candidates, row: usize) -> u32 {
    c.ids[row * c.count] as u32
}

// -------------------------------------------------------------------------------------------------------- the invariants

#[test]
#[ignore]
fn row_invariance_of_the_window() {
    let Some(rig) = Rig::load(6, 4096, 1024) else { return };
    let v = rig.vocab();
    // contexts on both arms of the prefill rule (up to 128 tokens the GEMV, above it the pinned GEMM) and windows from 1 to 16 rows
    let cases: Vec<(Vec<u32>, Vec<u32>)> = [(2, 1), (50, 7), (97, 16), (130, 16), (300, 5), (700, 16)]
        .iter()
        .enumerate()
        .map(|(i, &(c, w))| (prompt(c, i, v), window_tokens(w, i, v)))
        .collect();
    let make = || rig.shared();
    row_invariance(&make, &cases, 8).unwrap_or_else(|e| panic!("row invariance: {e}"));
    eprintln!(
        "row_invariance: {} cases, windows of 1..16 rows, all in one step (64 rows), in two orders: bit-equal to one-row steps",
        cases.len()
    );
    assert!(rig.backend.borrow_mut().take_error().is_none());
}

#[test]
#[ignore]
fn a_window_of_n_rows_equals_n_serial_rows() {
    // the context the windows run at (default 200; a long one exercises the split-K attention over many 512-key chunks)
    let ctx_len = env_usize("REXL3_MINICPM_WINDOW_CTX", 200);
    let Some(rig) = Rig::load(2, ctx_len + 256, 4096.min(ctx_len.max(1024))) else { return };
    let v = rig.vocab();
    let mut ctx = prompt(ctx_len.min(1024), 3, v);
    while ctx.len() < ctx_len {
        let more = prompt(1024, 3 + ctx.len() / 1024, v);
        ctx.extend_from_slice(&more[..(ctx_len - ctx.len()).min(1024)]);
    }
    let path = PrefillPath::for_prompt(ctx.len());
    let grid = rig.with_model(|m| m.opts.prefill_rows);
    let prefill = |m: &mut MiniCpmModel| {
        m.begin_slot(0).unwrap();
        let mut at = 0;
        while at < ctx.len() {
            let end = (at + grid).min(ctx.len());
            m.prefill_chunk(&chain(0, at, &ctx[at..end]), path, (end == ctx.len()).then_some(4)).unwrap();
            at = end;
        }
    };
    for rows in [1usize, 2, 3, 8, 15, 16, 33, 64] {
        let toks = window_tokens(rows, rows, v);
        // serial: one row a window, committing each
        let serial: Vec<(Vec<i32>, Vec<f32>)> = rig.with_model(|m| {
            prefill(m);
            let mut out = Vec::new();
            for (j, &t) in toks.iter().enumerate() {
                let c = m.window_candidates(&chain(0, ctx.len() + j, &[t]), 4).unwrap();
                m.commit(0, 1);
                out.push((c.ids, c.vals));
            }
            m.flush().unwrap();
            out
        });
        let (w, next) = rig.with_model(|m| {
            prefill(m);
            let w = m.window_candidates(&chain(0, ctx.len(), &toks), 4).unwrap();
            m.commit(0, rows);
            // and the state the commit left: one more row after it
            let next = m.window_candidates(&chain(0, ctx.len() + rows, &[top1(&w, rows - 1)]), 4).unwrap();
            m.commit(0, 1);
            m.flush().unwrap();
            (w, next)
        });
        for (j, (ids, vals)) in serial.iter().enumerate() {
            assert_eq!(&w.ids[j * 4..(j + 1) * 4], &ids[..], "window of {rows}: row {j} candidates differ from the serial row");
            let (a, b): (Vec<u32>, Vec<u32>) =
                (w.vals[j * 4..(j + 1) * 4].iter().map(|x| x.to_bits()).collect(), vals.iter().map(|x| x.to_bits()).collect());
            assert_eq!(a, b, "window of {rows}: row {j} logit bits differ from the serial row");
        }
        // the serial path's state after `rows` committed rows gives the same next row
        let after_serial = rig.with_model(|m| {
            prefill(m);
            for (j, &t) in toks.iter().enumerate() {
                m.window_candidates(&chain(0, ctx.len() + j, &[t]), 4).unwrap();
                m.commit(0, 1);
            }
            let c = m.window_candidates(&chain(0, ctx.len() + rows, &[top1(&w, rows - 1)]), 4).unwrap();
            m.commit(0, 1);
            m.flush().unwrap();
            c
        });
        assert_eq!(next.ids, after_serial.ids, "window of {rows}: the state after the commit differs");
        assert_eq!(
            next.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            after_serial.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "window of {rows}: the state after the commit differs"
        );
    }
    eprintln!("a window of 1, 2, 3, 8, 15, 16, 33, 64 rows matched the same rows decoded serially, and the state it committed");
}

#[test]
#[ignore]
fn serial_batched_equals_solo_through_the_engine() {
    let Some(rig) = Rig::load(4, 4096, 512) else { return };
    let v = rig.vocab();
    // four streams on both arms of the prefill rule, greedy and seeded-sampled, finishing at different times
    let reqs = vec![
        request(prompt(20, 0, v), greedy(), 24, false),
        request(prompt(90, 1, v), sampled(11), 30, false),
        request(prompt(160, 2, v), greedy(), 18, false),
        request(prompt(300, 3, v), sampled(23), 26, false),
    ];
    let make_engine = |drafting: bool| rig.engine(drafting);
    let r = serial_batched_equals_solo(&make_engine, &reqs).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("serial_batched_equals_solo: {} sequences, {} tokens, {} rounds", r.sequences, r.tokens, r.rounds);
    assert!(rig.backend.borrow_mut().take_error().is_none());
}

/// A drafter that knows each request's serial continuation (the tokens a solo run without drafts generated) and proposes it with a
/// deterministic pattern of wrong tokens in it, whatever the model is: verify windows of every width and partial acceptances at every
/// offset happen, so the rollback of the GDN records and the KV rows runs for real on every layer. (A natural drafter proposes only
/// where the text repeats, which a truncated smoke model does not.)
struct Scripted {
    scripts: Rc<Vec<(Vec<u32>, Vec<u32>)>>,
    vocab: u32,
}

impl Drafter for Scripted {
    fn propose(&mut self, tokens: &[u32], _: &Sampling, max: usize) -> Vec<u32> {
        let Some((prompt, cont)) = self.scripts.iter().find(|(p, _)| tokens.len() >= p.len() && tokens[..p.len()] == p[..])
        else {
            return Vec::new();
        };
        let g = tokens.len() - prompt.len();
        let n = max.min(15).min(cont.len().saturating_sub(g));
        let mut d: Vec<u32> = cont[g..g + n].to_vec();
        // every second round one proposal in the window is wrong, at an offset that walks through the window
        if (g / 3) % 2 == 1 && n > 0 {
            let at = g % n;
            d[at] = (d[at] + 1) % self.vocab;
        }
        d
    }
}

#[test]
#[ignore]
fn staggered_arrivals_equal_solo() {
    let Some(rig) = Rig::load(3, 4096, 128) else { return };
    let v = rig.vocab();
    // a small prefill chunk, so that the later prompts are prefilled in slices BETWEEN the decode rounds of the earlier streams
    let reqs = vec![
        request(prompt(40, 24, v), greedy(), 30, false),
        request(prompt(500, 25, v), sampled(8), 20, false),
        request(prompt(260, 26, v), greedy(), 20, false),
    ];
    let solo: Vec<Vec<u32>> = reqs
        .iter()
        .map(|r| {
            let mut e = rig.engine(false);
            e.submit(r.clone()).unwrap();
            e.run().unwrap();
            e.take_outputs().remove(0).tokens
        })
        .collect();
    let mut e = rig.engine(false);
    let mut ids = Vec::new();
    ids.push(e.submit_deferred(reqs[0].clone()).unwrap());
    for _ in 0..4 {
        e.step().unwrap();
    }
    ids.push(e.submit_deferred(reqs[1].clone()).unwrap());
    for _ in 0..3 {
        e.step().unwrap();
    }
    ids.push(e.submit_deferred(reqs[2].clone()).unwrap());
    e.run().unwrap();
    let mut out = e.take_outputs();
    out.sort_by_key(|o| ids.iter().position(|&i| i == o.id).unwrap());
    assert_eq!(out.len(), 3);
    for (k, o) in out.iter().enumerate() {
        assert_eq!(o.tokens, solo[k], "stream {k}: arriving while others decode changed its tokens");
    }
    eprintln!("staggered arrivals: three streams, prefill slices of 128 rows between decode rounds, equal to solo");
}

#[test]
#[ignore]
fn drafted_equals_serial_with_scripted_and_copy_drafts() {
    let Some(rig) = Rig::load(4, 4096, 512) else { return };
    let v = rig.vocab();
    let reqs = vec![
        request(prompt(60, 4, v), greedy(), 40, true),
        request(prompt(150, 5, v), sampled(31), 40, true),
        request(prompt(220, 6, v), greedy(), 40, true),
        request(prompt(40, 7, v), sampled(47), 40, true),
    ];
    // the serial continuations, solo and without drafts
    let scripts: Vec<(Vec<u32>, Vec<u32>)> = reqs
        .iter()
        .map(|r| {
            let mut e = rig.engine(false);
            e.submit(r.clone()).unwrap();
            e.run().unwrap();
            (r.prompt.clone(), e.take_outputs().remove(0).tokens)
        })
        .collect();
    let scripts = Rc::new(scripts);
    let make_engine = |drafting: bool| {
        let mm = ModelMemory::from_profile(&rig.profile, 0, 0);
        let admission = rig.plan.memory.admission(mm, rig.slots, rig.max_ctx).unwrap();
        let sc = scripts.clone();
        let make: Box<dyn Fn() -> Box<dyn Drafter>> = if drafting {
            Box::new(move || Box::new(Scripted { scripts: sc.clone(), vocab: v as u32 }))
        } else {
            Box::new(|| Box::new(NoDrafts))
        };
        Engine::new(
            rig.shared(),
            admission,
            EngineConfig { max_rows: WINDOW_ROWS, max_drafts: 15, cost: CostModel::from_table(STEP_MS.to_vec()), drafting },
            make,
        )
    };
    let r = drafted_equals_serial(&make_engine, &reqs).unwrap_or_else(|e| panic!("{e}"));
    eprintln!(
        "drafted_equals_serial (scripted drafts, up to 15 a round): {} sequences, {} tokens, {} drafted, {} accepted, {:.2} tokens a round",
        r.sequences,
        r.tokens,
        r.drafted,
        r.accepted,
        r.tokens_per_round()
    );
    assert!(r.accepted > 0 && r.drafted > r.accepted, "the scripted drafts were not both accepted and rejected");
    assert!(r.tokens_per_round() > 1.5, "verify windows were not exercised: {:.2} tokens a round", r.tokens_per_round());
    // the engine core's own copy drafter on prompts that repeat themselves (proposes only where the text does)
    let repeat = |n: usize, seed: usize| {
        let a = prompt(n, seed, v);
        let mut p = a.clone();
        p.extend_from_slice(&a[..n.min(40)]);
        p
    };
    let reqs = vec![request(repeat(60, 14), greedy(), 30, true), request(repeat(150, 15), sampled(33), 30, true)];
    let make_copy = |drafting: bool| rig.engine(drafting);
    let r = drafted_equals_serial(&make_copy, &reqs).unwrap_or_else(|e| panic!("copy drafter: {e}"));
    eprintln!("drafted_equals_serial (copy drafter): {} drafted, {} accepted", r.drafted, r.accepted);
}

#[test]
#[ignore]
fn chunked_prefill_equals_whole_through_the_backend() {
    let Some(rig) = Rig::load(1, 4096, 1024) else { return };
    let v = rig.vocab();
    let make = || rig.shared();
    // a prompt of 300 on the pinned GEMM (the rule is keyed on the total) and one of 100 on the GEMV, cut at a one-row head and tail,
    // either side of the 128-row GEMV limit, off every grid
    let long = prompt(300, 8, v);
    for cuts in [vec![1], vec![127, 128], vec![129], vec![200], vec![64, 128, 256], vec![299], vec![3, 97, 131, 299]] {
        chunked_prefill_equals_whole(&make, &long, &cuts, &greedy(), 10)
            .unwrap_or_else(|e| panic!("300 tokens, cuts {cuts:?}: {e}"));
    }
    let short = prompt(100, 9, v);
    for cuts in [vec![1], vec![50], vec![99]] {
        chunked_prefill_equals_whole(&make, &short, &cuts, &sampled(5), 10)
            .unwrap_or_else(|e| panic!("100 tokens, cuts {cuts:?}: {e}"));
    }
    // the same on odd chunk grids of the backend (a resumed slice lands on the absolute grid)
    for chunk in [97usize, 64, 1000] {
        rig.backend.borrow_mut().chunk = chunk;
        for cuts in [vec![1], vec![129], vec![150, 255]] {
            chunked_prefill_equals_whole(&make, &long, &cuts, &greedy(), 6)
                .unwrap_or_else(|e| panic!("chunk {chunk}, cuts {cuts:?}: {e}"));
        }
    }
    eprintln!("chunked_prefill_equals_whole: 300 and 100 tokens, 13 cut sets, backend grids 97, 64, 1000 and 1024");
}

/// The engine's row limit (`set_prefill_row_limit(Some(128))`, what `Engine::step` asks for near a drain or request deadline) changes
/// no bit: a prompt prefilled through the backend's own slicing with the limit on throughout, or switched on and off mid-prompt, leaves
/// the last row and the two decode windows after it bit-identical to the unlimited 1024-row slicing; and an engine whose slice deadline
/// is near (so every step runs under the limit) commits the same tokens as one without a deadline.
#[test]
#[ignore]
fn the_engines_row_limit_leaves_the_same_bits() {
    let total = env_usize("REXL3_MINICPM_PROMPT", 1031);
    let Some(rig) = Rig::load(2, total + 256, 1024) else { return };
    let v = rig.vocab();
    let p = prompt(total, 11, v);
    // the whole prompt through `prefill_slice` with the limit `limit_at(done)` set around each slice, then two decode windows
    let run = |slot: usize, limit_at: &dyn Fn(usize) -> Option<usize>| -> (Vec<usize>, Vec<RowOutput>) {
        let mut b = rig.backend.borrow_mut();
        b.release(slot);
        let (mut ends, mut done) = (Vec::new(), 0);
        let last = loop {
            assert!(b.set_prefill_row_limit(limit_at(done)));
            let step = b.prefill_slice(slot, &p, &[], done, 8);
            b.set_prefill_row_limit(None);
            done = step.done;
            ends.push(done);
            if let Some(row) = step.row {
                break row;
            }
        };
        assert!(b.take_error().is_none());
        let mut rows = vec![last];
        for k in 0..2 {
            let t = rows.last().unwrap().ids[0];
            let out = b.step(&[RowRequest { slot, first_position: (total + k) as u64, tokens: vec![t] }], 8);
            b.commit(slot, 1);
            rows.push(out[0][0].clone());
        }
        assert!(b.take_error().is_none());
        b.release(slot);
        (ends, rows)
    };
    let (ends0, ref_rows) = run(0, &|_| None);
    type Limit<'a> = &'a dyn Fn(usize) -> Option<usize>;
    let cases: [(&str, Limit); 3] = [
        ("on throughout", &|_| Some(128)),
        ("on for two slices, then off", &|d| (d < 256).then_some(128)),
        ("37 rows", &|_| Some(37)),
    ];
    for (what, limit_at) in cases {
        let (ends, rows) = run(1, limit_at);
        assert!(ends.len() > ends0.len(), "{what}: the limit shrank the slices ({ends:?} against {ends0:?})");
        for (i, (a, b)) in rows.iter().zip(&ref_rows).enumerate() {
            assert!(a.bits_eq(b), "{what}: row {i} (0 = last prompt row, then the decode windows) differs");
        }
        eprintln!("row limit {what}: slices end at {ends:?}; last row and two decode windows are the 1024-row slicing's bits");
    }

    // the engine: a near slice deadline (inside the 10 s urgency window) makes every step run under the limit, and the tokens are the same
    let reqs = || vec![request(p.clone(), greedy(), 24, false), request(prompt(300, 12, v), sampled(3), 24, false)];
    let mut plain = rig.engine(false);
    for r in reqs() {
        plain.submit_deferred(r).unwrap();
    }
    while plain.busy() {
        plain.step().unwrap();
    }
    let mut want: Vec<_> = plain.take_outputs().into_iter().map(|o| (o.id, o.tokens)).collect();
    want.sort();
    drop(plain);
    use gb_engine::ServeEngine as _;
    let mut near = rig.engine(false);
    for r in reqs() {
        near.submit_deferred(r).unwrap();
    }
    near.step().unwrap(); // a measured step first: the deadline guard never starts work it has no measurement for
    near.set_slice_deadline(Some(std::time::Instant::now() + std::time::Duration::from_secs(9)));
    let t = std::time::Instant::now();
    while near.busy() {
        assert!(near.step().unwrap() || !near.busy(), "the engine refused work under the deadline");
    }
    let mut got: Vec<_> = near.take_outputs().into_iter().map(|o| (o.id, o.tokens)).collect();
    got.sort();
    assert_eq!(got, want, "a near deadline changed the committed tokens");
    eprintln!(
        "engine under a 9 s slice deadline: both replies finished in {:.2} s with the same tokens",
        t.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore]
fn a_tiled_prefill_leaves_the_same_bits_in_any_chunk_grid() {
    let total = env_usize("REXL3_MINICPM_PROMPT", 1031);
    let Some(rig) = Rig::load(1, total + 256, 4096.min(env_usize("REXL3_MINICPM_PREFILL_ROWS", 4096))) else { return };
    let v = rig.vocab();
    let ids = prompt(total.min(1024), 1, v);
    let mut p = ids.clone();
    while p.len() < total {
        let more = prompt(1024, 2 + p.len() / 1024, v);
        p.extend_from_slice(&more[..(total - p.len()).min(1024)]);
    }
    assert_eq!(PrefillPath::for_prompt(total), PrefillPath::Tiled);
    let path = PrefillPath::for_prompt(total);
    let rows = rig.with_model(|m| m.opts.prefill_rows);
    let run = |m: &mut MiniCpmModel, grid: usize| -> (Candidates, Candidates) {
        m.begin_slot(0).unwrap();
        let mut done = 0;
        let mut last = None;
        while done < total {
            let n = grid.min(total - done);
            let want = (done + n == total).then_some(8);
            last = m.prefill_chunk(&chain(0, done, &p[done..done + n]), path, want).unwrap();
            done += n;
        }
        let last = last.unwrap();
        // and one decode window after it, so the recurrent state and the pages are compared too
        let after = m.window_candidates(&chain(0, total, &[top1(&last, 0)]), 8).unwrap();
        m.commit(0, 1);
        m.flush().unwrap();
        (last, after)
    };
    let (whole, after_whole) = rig.with_model(|m| run(m, rows.min(total)));
    // grids: single rows, odd, off the GEMV limit, a power of two and the chunk (4096 when the prompt is long enough)
    let mut grids = vec![1usize.min(total), 3, 97, 128, 257, 512, 1000, 4096];
    grids.retain(|&g| g <= rows);
    grids.dedup();
    for grid in grids {
        if grid == 1 && total > 2048 {
            continue; // one row a chunk through every layer is a latency test, not a bits test, at this length
        }
        let (cut, after) = rig.with_model(|m| run(m, grid));
        assert_eq!(cut.ids, whole.ids, "grid {grid}: the last row's candidate ids moved");
        assert_eq!(cut.vals, whole.vals, "grid {grid}: the last row's logit bits moved");
        assert_eq!(after.ids, after_whole.ids, "grid {grid}: the state the prefill left behind differs");
        assert_eq!(after.vals, after_whole.vals, "grid {grid}: the state the prefill left behind differs");
        eprintln!("grid {grid}: identical bits in the last row and in the window after it ({total} tokens)");
    }
}

/// The strongest form of chunk invariance the model offers: the residual stream after EVERY layer of EVERY row (the debug capture of
/// `set_capture`) is the same bits whatever grid the prompt is cut into, on the tiled path (the grouped prefill MoE included). The
/// last-row test above reaches the same rows through the KV and the delta-net state; this one compares them where they are made.
#[test]
#[ignore]
fn every_layer_of_every_row_is_the_same_in_any_chunk_grid() {
    let total = env_usize("REXL3_MINICPM_PROMPT", 1031);
    let Some(rig) = Rig::load(1, total + 256, 4096.min(env_usize("REXL3_MINICPM_PREFILL_ROWS", 4096))) else { return };
    let v = rig.vocab();
    let mut p = prompt(total.min(1024), 3, v);
    while p.len() < total {
        let more = prompt(1024, 7 + p.len() / 1024, v);
        p.extend_from_slice(&more[..(total - p.len()).min(1024)]);
    }
    // a prompt past 128 tokens runs the tiled path, one of up to 128 the GEMV path (`REXL3_MINICPM_PROMPT=100`): the MoE is the same
    // kernels on both, the grids are cut to the prompt
    let path = PrefillPath::for_prompt(total);
    let hidden = rig.cfg.hidden;
    let layers = rig.cfg.layers;
    let rows = rig.with_model(|m| m.opts.prefill_rows);
    // per layer, the bf16 bits of all `total` rows, for a grid
    let run = |m: &mut MiniCpmModel, grid: usize| -> Vec<Vec<u16>> {
        m.begin_slot(0).unwrap();
        let mut per_layer: Vec<Vec<u16>> = vec![Vec::with_capacity(total * hidden); layers];
        let mut done = 0;
        while done < total {
            let n = grid.min(total - done);
            m.set_capture(true);
            m.prefill_chunk(&chain(0, done, &p[done..done + n]), path, None).unwrap();
            m.set_capture(false);
            let got = m.take_captured();
            assert_eq!(got.len(), layers, "one capture a layer a forward");
            for (l, rows_bits) in got.into_iter().enumerate() {
                assert_eq!(rows_bits.len(), n * hidden);
                per_layer[l].extend_from_slice(&rows_bits);
            }
            done += n;
        }
        per_layer
    };
    let whole = rig.with_model(|m| run(m, rows.min(total)));
    for grid in [1usize, 3, 7, 33, 97, 128, 257, 1000] {
        if grid > rows || grid >= total || (grid < 16 && total > 2048) || (grid == 1 && total > 128) {
            continue;
        }
        let cut = rig.with_model(|m| run(m, grid));
        for l in 0..layers {
            if cut[l] != whole[l] {
                let at = cut[l].iter().zip(&whole[l]).position(|(a, b)| a != b).unwrap();
                panic!("grid {grid}: layer {l} differs from the whole prompt at row {} column {}", at / hidden, at % hidden);
            }
        }
        eprintln!("grid {grid}: the residual after all {layers} layers of all {total} rows is identical");
    }
}

#[test]
#[ignore]
fn rollback_equals_replay_for_every_keep() {
    let Some(rig) = Rig::load(4, 4096, 1024) else { return };
    let v = rig.vocab();
    let make = || rig.shared();
    // one stream: windows of 2 to 16 rows (the verify window of a 15-draft round), every keep that matters, and a follow-up window
    // of another width, on both prefill arms
    for ctx_len in [60usize, 200] {
        let ctx = prompt(ctx_len, ctx_len, v);
        for rows in [2usize, 5, 16] {
            let window = window_tokens(rows, rows + 1, v);
            let keeps: Vec<usize> = if rows <= 5 { (1..=rows).collect() } else { vec![1, 2, 7, 15, 16] };
            for keep in keeps {
                for next in [vec![17u32], window_tokens(4, 3, v), window_tokens(16, 4, v)] {
                    rollback_equals_replay(&make, &ctx, &window, keep, &next, 8)
                        .unwrap_or_else(|e| panic!("context {ctx_len}, window {rows}, keep {keep}, next {}: {e}", next.len()));
                }
            }
        }
    }
    // four streams of 16 rows in one window, each committing a different prefix, then the next window of all four: each stream's rows
    // must equal a backend that decoded only its accepted rows one at a time
    let ctxs = [prompt(40, 11, v), prompt(129, 12, v), prompt(200, 13, v), prompt(77, 14, v)];
    let w1: Vec<Vec<u32>> = (0..4).map(|s| window_tokens(16, 20 + s, v)).collect();
    let w2: Vec<Vec<u32>> = (0..4).map(|s| window_tokens(16, 30 + s, v)).collect();
    let keeps = [16usize, 1, 9, 5];
    let mut b = make();
    for (s, c) in ctxs.iter().enumerate() {
        b.prefill(s, 0, c, 8);
    }
    let first: Vec<RowRequest> =
        (0..4).map(|s| RowRequest { slot: s, tokens: w1[s].clone(), first_position: ctxs[s].len() as u64 }).collect();
    assert_eq!(b.step(&first, 8).iter().map(Vec::len).sum::<usize>(), 64);
    for (s, &k) in keeps.iter().enumerate() {
        b.commit(s, k);
    }
    let second: Vec<RowRequest> = (0..4)
        .map(|s| RowRequest { slot: s, tokens: w2[s].clone(), first_position: (ctxs[s].len() + keeps[s]) as u64 })
        .collect();
    let got = b.step(&second, 8);
    for s in 0..4 {
        let mut r = make();
        r.prefill(0, 0, &ctxs[s], 8);
        for (j, &t) in w1[s][..keeps[s]].iter().enumerate() {
            r.step(&[RowRequest { slot: 0, tokens: vec![t], first_position: (ctxs[s].len() + j) as u64 }], 8);
            r.commit(0, 1);
        }
        let want = r.step(&[RowRequest { slot: 0, tokens: w2[s].clone(), first_position: (ctxs[s].len() + keeps[s]) as u64 }], 8);
        for (j, (g, w)) in got[s].iter().zip(&want[0]).enumerate() {
            assert!(
                g.bits_eq(w),
                "stream {s} (keep {}) row {j} of the window after the rollback differs from a clean replay",
                keeps[s]
            );
        }
    }
    eprintln!("rollback_equals_replay: windows of 2/5/16 rows, every keep, and four streams x 16 rows with keeps {keeps:?}");
}

#[test]
#[ignore]
fn graph_replay_equals_eager() {
    let Some(rig) = Rig::load(4, 4096, 512) else { return };
    let v = rig.vocab();
    let ctxs = [prompt(33, 1, v), prompt(150, 2, v), prompt(500, 3, v), prompt(2, 4, v)];
    // window shapes: one stream of 1, 4 and 16 rows; four streams of 1; two streams of 8 and 3; a fixed shape run twice more
    let shapes: Vec<Vec<(usize, usize)>> = vec![
        vec![(0, 1)],
        vec![(1, 4)],
        vec![(2, 16)],
        vec![(0, 1), (1, 1), (2, 1), (3, 1)],
        vec![(1, 8), (3, 3)],
        vec![(2, 16)],
    ];
    let run_all = |m: &mut MiniCpmModel| -> Vec<Vec<u32>> {
        let mut pos = [0usize; 4];
        for (s, c) in ctxs.iter().enumerate() {
            m.begin_slot(s).unwrap();
            m.prefill_chunk(&chain(s, 0, c), PrefillPath::for_prompt(c.len()), None).unwrap();
            pos[s] = c.len();
        }
        let mut bits = Vec::new();
        for (k, shape) in shapes.iter().enumerate() {
            let reqs: Vec<RowRequest> = shape
                .iter()
                .map(|&(s, n)| RowRequest { slot: s, tokens: window_tokens(n, 40 + k, v), first_position: pos[s] as u64 })
                .collect();
            let b = RowBatch::from_requests(&reqs);
            let c = m.window_candidates(&b, 8).unwrap();
            bits.push(c.ids.iter().map(|&i| i as u32).chain(c.vals.iter().map(|x| x.to_bits())).collect());
            for &(s, n) in shape {
                // keep a prefix: the replay and the next window run on the committed state either way
                let keep = n.div_ceil(2);
                m.commit(s, keep);
                pos[s] += keep;
            }
        }
        m.flush().unwrap();
        bits
    };
    let eager = rig.with_model(|m| {
        m.set_graphs(false);
        run_all(m)
    });
    let (captured, n_graphs) = rig.with_model(|m| {
        m.set_graphs(true);
        m.clear_graphs().unwrap();
        let c = run_all(m);
        (c, m.graph_count())
    });
    assert!(n_graphs > 0, "the capture was refused or never happened: the replay check would be vacuous");
    assert_eq!(captured, eager, "a captured window differs bit for bit from eager");
    let (replayed, n_after) = rig.with_model(|m| {
        let r = run_all(m);
        (r, m.graph_count())
    });
    assert_eq!(n_after, n_graphs, "expected replay, not new captures");
    assert_eq!(replayed, eager, "a replayed window differs bit for bit from eager");
    assert!(rig.with_model(|m| m.graphs_on()), "the capture path fell back to eager");
    eprintln!("graph replay == eager over {} window shapes ({} graphs)", shapes.len(), n_graphs);
}

#[test]
#[ignore]
fn failure_isolation_a_bad_request_does_not_poison_the_others() {
    let Some(rig) = Rig::load(4, 2048, 512) else { return };
    let v = rig.vocab();
    let make = || rig.shared();
    let (ca, cb) = (prompt(50, 21, v), prompt(180, 22, v));
    let (wa, wb) = (window_tokens(7, 1, v), window_tokens(3, 2, v));
    let good_a = |pos: usize| RowRequest { slot: 0, tokens: wa.clone(), first_position: pos as u64 };
    let good_b = |pos: usize| RowRequest { slot: 2, tokens: wb.clone(), first_position: pos as u64 };
    // the reference: each alone on a clean backend
    let solo = |slot: usize, ctx: &[u32], req: RowRequest| {
        let mut b = make();
        b.prefill(slot, 0, ctx, 8);
        b.step(&[req], 8).remove(0)
    };
    let want_a = solo(0, &ca, good_a(ca.len()));
    let want_b = solo(2, &cb, good_b(cb.len()));
    let bad: Vec<(&str, RowRequest)> = vec![
        ("a slot that does not exist", RowRequest { slot: 17, tokens: vec![5, 6], first_position: 0 }),
        ("rows past the context", RowRequest { slot: 1, tokens: vec![5, 6], first_position: 2047 }),
        ("an id past the vocabulary", RowRequest { slot: 1, tokens: vec![5, v as u32], first_position: 0 }),
        ("an id inside the pad", RowRequest { slot: 1, tokens: vec![5, v as u32 + 7], first_position: 0 }),
        ("a window of 65 rows", RowRequest { slot: 1, tokens: vec![5; 65], first_position: 0 }),
    ];
    for (what, bad_req) in &bad {
        let mut b = make();
        b.prefill(0, 0, &ca, 8);
        b.prefill(2, 0, &cb, 8);
        let reqs = vec![good_a(ca.len()), bad_req.clone(), good_b(cb.len())];
        let out = b.step(&reqs, 8);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1].len(), bad_req.tokens.len(), "{what}: one output row per input row");
        assert!(out[1].iter().all(|r| r.values.iter().all(|x| x.is_nan())), "{what}: the bad request is answered with NaN rows");
        for (g, w) in out[0].iter().zip(&want_a) {
            assert!(g.bits_eq(w), "{what}: the request before it changed");
        }
        for (g, w) in out[2].iter().zip(&want_b) {
            assert!(g.bits_eq(w), "{what}: the request after it changed");
        }
        // and the state they left is clean: commit both and the next windows equal a backend that never saw the bad one
        b.commit(0, 7);
        b.commit(2, 3);
        let next_a = RowRequest { slot: 0, tokens: window_tokens(2, 9, v), first_position: (ca.len() + 7) as u64 };
        let next_b = RowRequest { slot: 2, tokens: window_tokens(2, 8, v), first_position: (cb.len() + 3) as u64 };
        let got = b.step(&[next_a.clone(), bad_req.clone(), next_b.clone()], 8);
        let mut clean = make();
        clean.prefill(0, 0, &ca, 8);
        clean.prefill(2, 0, &cb, 8);
        clean.step(&[good_a(ca.len()), good_b(cb.len())], 8);
        clean.commit(0, 7);
        clean.commit(2, 3);
        let want = clean.step(&[next_a, next_b], 8);
        for (g, w) in got[0].iter().zip(&want[0]) {
            assert!(g.bits_eq(w), "{what}: the next window of the first stream differs after a bad request");
        }
        for (g, w) in got[2].iter().zip(&want[1]) {
            assert!(g.bits_eq(w), "{what}: the next window of the second stream differs after a bad request");
        }
        assert!(b.take_error().is_none(), "{what}: a rejected request must not be an error that fails the other streams");
    }
    // a prompt that cannot be served fails alone through the engine: its neighbours finish with their serial tokens
    let mut e = rig.engine(false);
    let good = request(prompt(70, 5, v), greedy(), 8, false);
    let want = {
        let mut f = rig.engine(false);
        f.submit(good.clone()).unwrap();
        f.run().unwrap();
        f.take_outputs().remove(0).tokens
    };
    e.submit(good.clone()).unwrap();
    assert!(e.submit(request(vec![1; 5000], greedy(), 4, false)).is_err(), "a prompt past the context is refused at admission");
    e.run().unwrap();
    assert_eq!(e.take_outputs().remove(0).tokens, want);
    eprintln!("failure isolation: {} kinds of bad request beside two good streams, state clean afterwards", bad.len());
}

#[test]
#[ignore]
fn the_pad_never_reaches_a_consumer() {
    let Some(rig) = Rig::load(3, 4096, 512) else { return };
    let v = rig.vocab();
    let stride = rig.with_model(|m| m.logits_stride());
    assert_eq!((v, stride), (248_144, 248_192));
    let ctx = prompt(150, 31, v);
    let path = PrefillPath::for_prompt(ctx.len());
    let toks = window_tokens(16, 5, v);
    let run = |m: &mut MiniCpmModel| {
        m.begin_slot(0).unwrap();
        let last = m.prefill_chunk(&chain(0, 0, &ctx), path, Some(64)).unwrap().unwrap();
        let mut logits = vec![0u16; 16 * v];
        let w = m.window_candidates(&chain(0, ctx.len(), &toks), 64).unwrap();
        m.last_rows_logits(0, 16, &mut logits).unwrap();
        m.commit(0, 16);
        m.flush().unwrap();
        (last, w, logits)
    };
    let clean = rig.with_model(|m| {
        m.set_graphs(false);
        run(m)
    });
    for graphs in [false, true] {
        let poisoned = rig.with_model(|m| {
            m.set_graphs(graphs);
            m.set_pad_poison(true);
            let r = run(m);
            m.set_pad_poison(false);
            r
        });
        for (name, a, b) in [("the prefill's last row", &clean.0, &poisoned.0), ("the window", &clean.1, &poisoned.1)] {
            assert!(b.ids.iter().all(|&i| (i as usize) < v), "graphs {graphs}: {name}: an id at or past the vocabulary");
            assert_eq!(a.ids, b.ids, "graphs {graphs}: {name}: the pad changed the candidates");
            assert_eq!(
                a.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "graphs {graphs}: {name}: the pad changed the values"
            );
        }
        // the logits export strips the pad: v columns a row, equal to the clean ones, and no value is the poison
        assert_eq!(clean.2, poisoned.2, "graphs {graphs}: the logits export read the pad");
    }
    // the sampler: engine requests with the poison on emit the tokens the clean engine does, none outside the vocabulary
    let reqs = vec![
        request(prompt(60, 41, v), greedy(), 24, false),
        request(prompt(60, 42, v), sampled(99), 24, false),
        request(prompt(200, 43, v), Sampling { seed: 5, temperature: 1.5, top_k: 56, top_p: 1.0, min_p: 0.0 }, 24, true),
    ];
    let outs = |poison: bool| -> Vec<Vec<u32>> {
        rig.with_model(|m| m.set_pad_poison(poison));
        let mut e = rig.engine(true);
        for r in &reqs {
            e.submit(r.clone()).unwrap();
        }
        e.run().unwrap();
        let mut o = e.take_outputs();
        o.sort_by_key(|x| x.id);
        rig.with_model(|m| m.set_pad_poison(false));
        o.into_iter().map(|x| x.tokens).collect()
    };
    let (clean_tokens, poisoned_tokens) = (outs(false), outs(true));
    assert_eq!(clean_tokens, poisoned_tokens, "the pad changed what the engine emitted");
    assert!(poisoned_tokens.iter().flatten().all(|&t| (t as usize) < v), "an id past the vocabulary was emitted");
    eprintln!("pad: 48 columns poisoned with the largest finite value of the head's dtype; top-k, logits export and engine tokens unchanged, all ids < {v}");
}

// ------------------------------------------------------------------------------------------------------ the seams

#[test]
#[ignore]
fn the_mrope_seam_text_rows_are_the_one_dimensional_rotation_and_a_delta_moves_them() {
    let Some(rig) = Rig::load(2, 2048, 512) else { return };
    let v = rig.vocab();
    let ctx = prompt(180, 51, v);
    let path = PrefillPath::for_prompt(ctx.len());
    let last = |m: &mut MiniCpmModel, b: &RowBatch| {
        m.begin_slot(0).unwrap();
        let c = m.prefill_chunk(b, path, Some(8)).unwrap().unwrap();
        (c.ids, c.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
    };
    let plain = chain(0, 0, &ctx);
    let base = rig.with_model(|m| last(m, &plain));
    // explicit triples equal to the position on all three streams are the default
    let mut same = plain.clone();
    same.rope3 = Some((0..ctx.len() as i32).flat_map(|p| [p, p, p]).collect());
    assert_eq!(rig.with_model(|m| last(m, &same)), base, "(p, p, p) is not the 1-D rotation");
    // a different H or W stream changes the rows (the sections are read), and a text-only delta shifts them like positions do
    let mut moved = plain.clone();
    moved.rope3 = Some((0..ctx.len() as i32).flat_map(|p| [p, p + 3, p + 5]).collect());
    assert_ne!(rig.with_model(|m| last(m, &moved)), base, "the H and W streams of the triples are ignored");
    let delta = rig.with_model(|m| {
        m.begin_slot(0).unwrap();
        m.set_rope_delta(0, 11).unwrap();
        let c = m.prefill_chunk(&plain, path, Some(8)).unwrap().unwrap();
        (c.ids, c.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
    });
    let mut shifted = plain.clone();
    shifted.rope3 = Some((0..ctx.len() as i32).flat_map(|p| [p + 11, p + 11, p + 11]).collect());
    assert_eq!(rig.with_model(|m| last(m, &shifted)), delta, "a rope delta is not the same as shifted triples");
    assert_ne!(delta, base, "the rope delta changed nothing");
    // a decode window after the prompt uses the slot's delta (index + delta on every stream)
    let w = rig.with_model(|m| {
        m.begin_slot(0).unwrap();
        m.set_rope_delta(0, 11).unwrap();
        m.prefill_chunk(&shifted, path, None).unwrap();
        let c = m.window_candidates(&chain(0, ctx.len(), &[9]), 4).unwrap();
        m.commit(0, 1);
        m.flush().unwrap();
        (c.ids, c.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
    });
    let w2 = rig.with_model(|m| {
        m.begin_slot(0).unwrap();
        m.set_rope_delta(0, 11).unwrap();
        m.prefill_chunk(&plain, path, None).unwrap();
        let c = m.window_candidates(&chain(0, ctx.len(), &[9]), 4).unwrap();
        m.commit(0, 1);
        m.flush().unwrap();
        (c.ids, c.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
    });
    assert_eq!(w, w2, "explicit triples and the slot delta disagree in the window after the prompt");
    // a slot's delta is dropped when the slot begins again
    rig.with_model(|m| {
        m.begin_slot(0).unwrap();
        assert_eq!(m.rope_delta[0], 0);
    });
}

#[test]
#[ignore]
fn the_feature_splice_seam_replaces_embedding_rows() {
    let Some(rig) = Rig::load(1, 2048, 512) else { return };
    let v = rig.vocab();
    let a = prompt(40, 61, v);
    let h = rig.cfg.hidden;
    let feats: Vec<u16> = rig.embed_rows.iter().flat_map(|(_, r)| r.iter().copied()).collect();
    let feats = DevBuf::from_slice(&feats).unwrap();
    let toks: Vec<u32> = rig.embed_rows.iter().map(|(t, _)| *t).collect();
    assert_eq!(feats.bytes(), toks.len() * h * 2);
    let mut b = a.clone();
    b[10..16].copy_from_slice(&toks);
    for path in [PrefillPath::Gemv, PrefillPath::Tiled] {
        let (want, got) = rig.with_model(|m| {
            m.begin_slot(0).unwrap();
            let want = m.prefill_chunk(&chain(0, 0, &b), path, Some(8)).unwrap().unwrap();
            m.begin_slot(0).unwrap();
            // the same prompt with six placeholder ids over those rows and the real rows spliced in
            let ops = [SpliceOp { feats: &feats, dst_row: 10, src_row: 0, n: 6 }];
            let got = m.prefill_chunk_spliced(&chain(0, 0, &a), path, Some(8), &ops).unwrap().unwrap();
            (want, got)
        });
        assert_eq!(got.ids, want.ids, "{path:?}: the splice changed the candidates");
        assert_eq!(
            got.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "{path:?}: the splice changed the logits"
        );
    }
    // a splice outside the chunk is refused by name
    let e = rig
        .with_model(|m| {
            m.begin_slot(0).unwrap();
            let ops = [SpliceOp { feats: &feats, dst_row: 38, src_row: 0, n: 6 }];
            m.prefill_chunk_spliced(&chain(0, 0, &a), PrefillPath::Gemv, None, &ops)
        })
        .unwrap_err();
    assert!(e.to_string().contains("splice"), "{e}");
}

#[test]
#[ignore]
fn the_working_set_is_inside_the_plans_engine_reserve() {
    // the attention partials grow with the context: REXL3_MINICPM_RESERVE_CTX=262144 prices the whole of it
    let ctx = env_usize("REXL3_MINICPM_RESERVE_CTX", 16384);
    let Some(rig) = Rig::load(4, ctx, 4096.min(env_usize("REXL3_MINICPM_PREFILL_ROWS", 4096))) else { return };
    let b = rig.with_model(|m| m.bytes());
    let reserve = rig.plan.memory.engine_reserve_bytes;
    eprintln!(
        "working set: scratch {:.1} MiB, workspaces {:.1} MiB, window records {:.1} MiB = {:.1} MiB of the {:.1} MiB engine reserve",
        b.scratch as f64 / 1048576.0,
        b.workspaces as f64 / 1048576.0,
        b.window_records as f64 / 1048576.0,
        b.engine_reserve() as f64 / 1048576.0,
        reserve as f64 / 1048576.0
    );
    assert!(b.engine_reserve() <= reserve, "the forward allocated more than the plan held back");
    // the plan's own per-row scratch figure bounds the activation buffers at the chunk the boot priced (the model allocates a
    // handful of wider buffers than the allowance names; both are inside the reserve)
    assert_eq!(b.window_records, WINDOW_ROWS as u64 * rig.cfg.linear_layers() as u64 * 33_024);
}

// ------------------------------------------------------------------------------------------------------------ the oracle

/// What the fork's `fork_logits.py --store full` wrote (`common.LogitSet`): row-centred fp16 logits `[rows, T, V]`, the row max and the
/// fork's argmax per position.
struct ForkSet {
    dir: PathBuf,
    rows: usize,
    tokens: usize,
    vocab: usize,
    rows_selected: Vec<usize>,
}

impl ForkSet {
    fn read(dir: &Path) -> Self {
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("set.json")).unwrap()).unwrap();
        let rows = meta["rows"].as_u64().unwrap() as usize;
        let rows_selected = meta["rows_selected"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap() as usize).collect())
            .unwrap_or_else(|| (0..rows).collect());
        assert!(
            meta["files"].as_array().unwrap().iter().any(|f| f == "logits"),
            "{}: the set holds no logits.f16 (--store full)",
            dir.display()
        );
        Self {
            dir: dir.to_path_buf(),
            rows,
            tokens: meta["tokens"].as_u64().unwrap() as usize,
            vocab: meta["vocab"].as_u64().unwrap() as usize,
            rows_selected,
        }
    }

    fn floats(&self, name: &str) -> Vec<f32> {
        let raw = std::fs::read(self.dir.join(name)).unwrap();
        raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn ints(&self, name: &str) -> Vec<i32> {
        let raw = std::fs::read(self.dir.join(name)).unwrap();
        raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
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

/// One position of the oracle comparison.
struct PosStats {
    /// The engine's argmax (the first maximum over the real vocabulary).
    arg: usize,
    cos: f64,
    /// KL(fork || engine) in nats.
    kl: f64,
    /// The fork's top logit minus the fork's logit at the engine's argmax (0 when they agree).
    flip_gap: f64,
    /// The fork's top-1 minus top-2 logit: how decided the position is.
    margin: f64,
}

/// One position: the engine's bf16 logits against the fork's raw logits (`f16` row-centred + `max`), over the real vocabulary, the
/// sums in f64.
fn compare_position(engine: &[u16], fork_centered: &[u8], fork_max: f32, engine_f16: bool) -> PosStats {
    let n = engine.len();
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    let (mut emax, mut fmax) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut arg = 0usize;
    let mut best = f32::NEG_INFINITY;
    let (mut f1, mut f2) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    let mut e = Vec::with_capacity(n);
    let mut f = Vec::with_capacity(n);
    for i in 0..n {
        let x = if engine_f16 { f16(engine[i]) } else { bf16(engine[i]) };
        let y = f16(u16::from_le_bytes([fork_centered[2 * i], fork_centered[2 * i + 1]])) + fork_max;
        if x > best {
            best = x;
            arg = i;
        }
        let (xd, yd) = (x as f64, y as f64);
        ab += xd * yd;
        aa += xd * xd;
        bb += yd * yd;
        emax = emax.max(xd);
        fmax = fmax.max(yd);
        if yd > f1 {
            f2 = f1;
            f1 = yd;
        } else if yd > f2 {
            f2 = yd;
        }
        e.push(xd);
        f.push(yd);
    }
    let (mut ze, mut zf) = (0f64, 0f64);
    for i in 0..n {
        ze += (e[i] - emax).exp();
        zf += (f[i] - fmax).exp();
    }
    let (lze, lzf) = (ze.ln() + emax, zf.ln() + fmax);
    let mut kl = 0f64;
    for i in 0..n {
        let lp = f[i] - lzf;
        let lq = e[i] - lze;
        kl += lp.exp() * (lp - lq);
    }
    PosStats { arg, cos: ab / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE), kl, flip_gap: f1 - f[arg], margin: f1 - f2 }
}

#[derive(Default)]
struct OracleStats {
    positions: usize,
    agree: usize,
    fork_self_agree: usize,
    cos_sum: f64,
    cos_min: f64,
    kl_sum: f64,
    kl_max: f64,
    kls: Vec<f64>,
    /// Disagreements by the fork's logit gap between its top-1 and the engine's pick: < 1/16, < 1/8, < 1/4, < 1/2, the rest.
    flips: [usize; 5],
    /// Positions whose fork top-1 / top-2 margin is at least 1/4, and how many of them agree.
    decided: (usize, usize),
}

/// Every position of every row of the fork's dump through the engine: prefill in chunks of the model's prefill rows, the LM head over
/// 64-row slices, and per position the argmax, the cosine and the KL against the fork's logits.
#[test]
#[ignore]
fn the_logits_agree_with_the_fork_oracle_at_every_position() {
    let Some(odir) = std::env::var("REXL3_MINICPM_ORACLE_DIR").ok().map(PathBuf::from) else {
        eprintln!("not run: REXL3_MINICPM_ORACLE_DIR is not set (the output of tools/minicpm_v47/fork_logits.py --store full)");
        return;
    };
    let set = ForkSet::read(&odir);
    let ids_path = std::env::var("REXL3_MINICPM_IDS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join("projects/minicpm-kl/ids/legA.safetensors"));
    let (id_rows, id_t, ids) = read_ids(&ids_path);
    assert_eq!(id_t, set.tokens, "the ids file has {id_t} tokens a row, the fork's set {}", set.tokens);
    let min_top1: f64 = std::env::var("REXL3_MINICPM_ORACLE_MIN_TOP1").ok().and_then(|v| v.parse().ok()).unwrap_or(0.97);
    let min_cos: f64 = std::env::var("REXL3_MINICPM_ORACLE_MIN_COS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.995);
    let max_rows = env_usize("REXL3_MINICPM_ORACLE_ROWS", set.rows).min(set.rows);
    let prefill_rows = env_usize("REXL3_MINICPM_PREFILL_ROWS", 1024).clamp(1, 4096);
    let Some(rig) = Rig::load(1, set.tokens + 64, prefill_rows) else { return };
    let v = rig.vocab();
    assert_eq!(set.vocab, v, "the fork's vocabulary");
    let head_f16 = rig.with_model(|m| m.logits_are_f16());
    eprintln!("the head writes {} logits", if head_f16 { "fp16" } else { "bf16" });
    let rowmax = set.floats("rowmax.f32");
    let argmax = set.ints("argmax.i32");
    let mut lf = std::fs::File::open(odir.join("logits.f16")).unwrap();
    assert_eq!(lf.metadata().unwrap().len(), (set.rows * set.tokens * v * 2) as u64, "logits.f16 is not [rows, T, V] f16");
    let mut ours = vec![0u16; WINDOW_ROWS * v];
    let mut theirs = vec![0u8; WINDOW_ROWS * v * 2];
    let mut st = OracleStats { cos_min: f64::INFINITY, ..Default::default() };
    let t0 = std::time::Instant::now();
    for r in 0..max_rows {
        let src = set.rows_selected[r];
        assert!(src < id_rows, "row {src} of the ids file");
        let toks: Vec<u32> = ids[src * id_t..(src + 1) * id_t].iter().map(|&x| x as u32).collect();
        let n = toks.len();
        let path = PrefillPath::for_prompt(n);
        let (mut r_agree, mut r_cos, mut r_kl) = (0usize, 0f64, 0f64);
        rig.with_model(|m| {
            m.begin_slot(0).unwrap();
            let mut at = 0;
            while at < n {
                let end = (at + prefill_rows).min(n);
                m.prefill_chunk(&chain(0, at, &toks[at..end]), path, None).unwrap();
                let mut q = 0;
                while q < end - at {
                    let k = WINDOW_ROWS.min(end - at - q);
                    m.last_rows_logits(q, k, &mut ours).unwrap();
                    let p0 = at + q;
                    lf.seek(SeekFrom::Start(((r * set.tokens + p0) * v * 2) as u64)).unwrap();
                    lf.read_exact(&mut theirs[..k * v * 2]).unwrap();
                    // positions of this slice in parallel: the comparison is the cost, not the forward
                    let res: Vec<PosStats> = std::thread::scope(|s| {
                        let hs: Vec<_> = (0..k)
                            .map(|j| {
                                let (e, f) = (&ours[j * v..(j + 1) * v], &theirs[j * v * 2..(j + 1) * v * 2]);
                                let mx = rowmax[r * set.tokens + p0 + j];
                                s.spawn(move || compare_position(e, f, mx, head_f16))
                            })
                            .collect();
                        hs.into_iter().map(|h| h.join().unwrap()).collect()
                    });
                    for (j, ps) in res.into_iter().enumerate() {
                        let p = p0 + j;
                        let fork_top = argmax[r * set.tokens + p];
                        let agrees = ps.arg as i32 == fork_top;
                        r_agree += agrees as usize;
                        r_cos += ps.cos;
                        r_kl += ps.kl;
                        st.cos_min = st.cos_min.min(ps.cos);
                        st.kl_max = st.kl_max.max(ps.kl);
                        st.kls.push(ps.kl);
                        if ps.margin >= 0.25 {
                            st.decided.0 += 1;
                            st.decided.1 += agrees as usize;
                        }
                        if !agrees {
                            let b = [0.0625, 0.125, 0.25, 0.5].iter().position(|&t| ps.flip_gap < t).unwrap_or(4);
                            st.flips[b] += 1;
                        }
                        // the fork's own stored logits must reproduce its argmax: a misread layout shows here, not as a forward bug
                        let row = &theirs[j * v * 2..(j + 1) * v * 2];
                        let mut best = (f32::NEG_INFINITY, 0usize);
                        for i in 0..v {
                            let x = f16(u16::from_le_bytes([row[2 * i], row[2 * i + 1]]));
                            if x > best.0 {
                                best = (x, i);
                            }
                        }
                        st.fork_self_agree += (best.1 as i32 == fork_top) as usize;
                    }
                    q += k;
                }
                at = end;
            }
        });
        st.positions += n;
        st.agree += r_agree;
        st.cos_sum += r_cos;
        st.kl_sum += r_kl;
        eprintln!(
            "row {r} (ids row {src}, {n} tokens): top-1 {:.4}, mean cosine {:.6}, mean KL(fork||engine) {:.5} [{:.0} s]",
            r_agree as f64 / n as f64,
            r_cos / n as f64,
            r_kl / n as f64,
            t0.elapsed().as_secs_f64()
        );
    }
    let (top1_rate, cos_mean, kl_mean) =
        (st.agree as f64 / st.positions as f64, st.cos_sum / st.positions as f64, st.kl_sum / st.positions as f64);
    let mut kls = st.kls.clone();
    kls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| kls[((kls.len() - 1) as f64 * p) as usize];
    eprintln!(
        "ORACLE {} layers, {} rows, {} positions ({} tokens a row): top-1 {}/{} = {:.4}; mean cosine {:.6} (min {:.5}); \
         KL(fork||engine) mean {:.5} median {:.5} p95 {:.4} p99 {:.4} max {:.3}; the dump's own argmax agrees with its stored logits at {}/{}",
        rig.cfg.layers,
        max_rows,
        st.positions,
        set.tokens,
        st.agree,
        st.positions,
        top1_rate,
        cos_mean,
        st.cos_min,
        kl_mean,
        q(0.5),
        q(0.95),
        q(0.99),
        st.kl_max,
        st.fork_self_agree,
        st.positions
    );
    eprintln!(
        "ORACLE disagreements by the fork's gap to the engine's pick: <1/16 {}, <1/8 {}, <1/4 {}, <1/2 {}, >=1/2 {}; positions the \
         fork decides by 1/4 or more: {}, of which top-1 agrees at {:.4}",
        st.flips[0],
        st.flips[1],
        st.flips[2],
        st.flips[3],
        st.flips[4],
        st.decided.0,
        st.decided.1 as f64 / st.decided.0.max(1) as f64
    );
    assert!(
        st.fork_self_agree as f64 >= 0.99 * st.positions as f64,
        "the dump's logits do not match its own argmax: layout misread"
    );
    assert!(top1_rate >= min_top1, "top-1 agreement {top1_rate:.4} below {min_top1}");
    assert!(cos_mean >= min_cos, "mean logit cosine {cos_mean:.6} below {min_cos}");
}

/// The residual stream after each of the fork's dumped layers (`fork_logits.py --dump-layers ... --dump-rows 1 --store none`, fp32
/// `[1, T, 2048]` a layer) against the engine's, layer by layer: the curve that tells a bug (a step at one layer kind) from rounding
/// (bf16 activations and a bf16 residual against the fork's fp16 activations and fp32 residual: a slow rise).
#[test]
#[ignore]
fn the_layer_outputs_follow_the_forks_dump() {
    let Some(ldir) = std::env::var("REXL3_MINICPM_LAYERS_DIR").ok().map(PathBuf::from) else {
        eprintln!(
            "not run: REXL3_MINICPM_LAYERS_DIR is not set (fork_logits.py --dump-layers 0,1,.. --dump-rows 1 --store none)"
        );
        return;
    };
    let dump: serde_json::Value = serde_json::from_slice(&std::fs::read(ldir.join("dump.json")).unwrap()).unwrap();
    let t = dump["tokens"].as_u64().unwrap() as usize;
    let src = dump["rows_selected"][0].as_u64().unwrap() as usize;
    let ids_path = std::env::var("REXL3_MINICPM_IDS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap()).join("projects/minicpm-kl/ids/legA.safetensors"));
    let (_, id_t, ids) = read_ids(&ids_path);
    assert_eq!(id_t, t);
    let toks: Vec<u32> = ids[src * t..(src + 1) * t].iter().map(|&x| x as u32).collect();
    let Some(rig) = Rig::load(1, t + 64, t.min(4096)) else { return };
    let h = rig.cfg.hidden;
    let got = rig.with_model(|m| {
        m.begin_slot(0).unwrap();
        m.set_capture(true);
        m.prefill_chunk(&chain(0, 0, &toks), PrefillPath::for_prompt(t), None).unwrap();
        m.set_capture(false);
        m.take_captured()
    });
    let layers: Vec<usize> = {
        let mut v: Vec<usize> = dump["layers"].as_object().unwrap().keys().map(|k| k.parse().unwrap()).collect();
        v.sort_unstable();
        v
    };
    assert!(got.len() >= layers.len(), "{} layers captured", got.len());
    let kinds: Vec<&str> = rig.cfg.layer_types.iter().map(|k| if k == "full_attention" { "attn" } else { "gdn " }).collect();
    let mut prev_rel = 0f64;
    for &l in &layers {
        let raw = std::fs::read(ldir.join(dump["layers"][l.to_string()]["out"].as_str().unwrap())).unwrap();
        let want: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(want.len(), t * h);
        let (mut num, mut den, mut dot, mut aa) = (0f64, 0f64, 0f64, 0f64);
        let mut per_pos = Vec::with_capacity(t);
        for p in 0..t {
            let (mut n, mut d) = (0f64, 0f64);
            for i in 0..h {
                let a = bf16(got[l][p * h + i]) as f64;
                let b = want[p * h + i] as f64;
                n += (a - b) * (a - b);
                d += b * b;
                dot += a * b;
                aa += a * a;
            }
            num += n;
            den += d;
            per_pos.push((n / d.max(1e-30)).sqrt());
        }
        per_pos.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rel = (num / den).sqrt();
        eprintln!(
            "layer {l:2} {}: rel_l2 {:.5} (position median {:.5}, p99 {:.5}, max {:.4}), cosine {:.6}, step {:+.5}",
            kinds[l],
            rel,
            per_pos[t / 2],
            per_pos[t * 99 / 100],
            per_pos[t - 1],
            dot / (aa.sqrt() * den.sqrt()),
            rel - prev_rel
        );
        prev_rel = rel;
        // a few positions can sit far from the fork's (a near-tie routing flip in a token whose residual is large moves the whole
        // layer's sum of squares): the gate is the typical position and the cosine of the layer, the table above is the evidence
        assert!(per_pos[t / 2] < 0.1, "layer {l}: the typical position is {:.4} (relative L2) from the fork's", per_pos[t / 2]);
        assert!(dot / (aa.sqrt() * den.sqrt()) > 0.99, "layer {l}: the layer's cosine against the fork's is below 0.99");
    }
}

// ------------------------------------------------------------------------------------------------------------ timing lane

/// Decode and verify-window cost, prefill throughput and the per-stage breakdown on the pack. A timing lane: run it alone
/// (`flock -x`), never in the gate (the name carries `bench`).
///
/// `REXL3_MINICPM_BENCH_CTX` is the context the windows run at (default 1,024).
#[test]
#[ignore]
fn bench_decode_windows_prefill_and_stages() {
    let ctx_len = env_usize("REXL3_MINICPM_BENCH_CTX", 1024);
    // TTFT lengths (a whole prompt in chunks of 4,096): `REXL3_MINICPM_BENCH_TTFT=1024,8192,32768,...` (empty: none)
    let ttft_lens: Vec<usize> = std::env::var("REXL3_MINICPM_BENCH_TTFT")
        .unwrap_or_else(|_| "1024,8192,32768".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let longest = ttft_lens.iter().copied().max().unwrap_or(0);
    let Some(rig) = Rig::load(4, ctx_len.max(8192).max(longest) + 512, 4096) else { return };
    let v = rig.vocab();
    let layers = rig.cfg.layers;
    let mean_ms = |f: &mut dyn FnMut() -> f64, warm: usize, n: usize| -> f64 {
        for _ in 0..warm {
            f();
        }
        (0..n).map(|_| f()).sum::<f64>() / n as f64
    };
    // one stream at `rows` rows, or `streams` streams of `rows` rows each, at the context `ctx_len`; steps commit everything
    let window_ms = |rig: &Rig, streams: usize, rows: usize, graphs: bool| -> f64 {
        rig.with_model(|m| {
            m.set_graphs(graphs);
            let mut pos = vec![0usize; streams];
            for (s, at) in pos.iter_mut().enumerate() {
                m.begin_slot(s).unwrap();
                let c = prompt(ctx_len, 70 + s, v);
                let mut done = 0;
                while done < c.len() {
                    let n = 4096.min(c.len() - done);
                    m.prefill_chunk(&chain(s, done, &c[done..done + n]), PrefillPath::for_prompt(c.len()), None).unwrap();
                    done += n;
                }
                *at = ctx_len;
            }
            let mut step = |m: &mut MiniCpmModel| -> f64 {
                let reqs: Vec<RowRequest> = (0..streams)
                    .map(|s| RowRequest {
                        slot: s,
                        tokens: window_tokens(rows, 80 + s + pos[s] % 7, v),
                        first_position: pos[s] as u64,
                    })
                    .collect();
                let t = std::time::Instant::now();
                m.window_candidates(&RowBatch::from_requests(&reqs), 8).unwrap();
                for (s, at) in pos.iter_mut().enumerate() {
                    m.commit(s, 1);
                    *at += 1;
                }
                m.flush().unwrap();
                t.elapsed().as_secs_f64() * 1e3
            };
            mean_ms(&mut || step(m), 3, 20)
        })
    };
    // REXL3_MINICPM_BENCH_PREFILL_ONLY=1 skips the decode and window rows (a prefill timing pass)
    let prefill_only = std::env::var("REXL3_MINICPM_BENCH_PREFILL_ONLY").is_ok_and(|v| v == "1");
    eprintln!("BENCH {layers} layers, context {ctx_len}, bf16 residual, graphs on unless noted; wall ms a step (window + candidates + commit)");
    if !prefill_only {
        for rows in [1usize, 2, 4, 8, 16] {
            eprintln!("BENCH window 1 stream x {rows:2} rows: {:7.2} ms", window_ms(&rig, 1, rows, true));
        }
        eprintln!("BENCH window 1 stream x  1 rows, eager: {:7.2} ms", window_ms(&rig, 1, 1, false));
        for (streams, rows) in [(2usize, 1usize), (4, 1), (4, 4), (4, 8), (4, 16)] {
            eprintln!(
                "BENCH window {streams} streams x {rows:2} rows = {:2}: {:7.2} ms",
                streams * rows,
                window_ms(&rig, streams, rows, true)
            );
        }
    }
    // prefill throughput at chunk 512 and 4096 (the pinned GEMM path: prompts past 128 tokens)
    let total = 4096;
    let p = prompt(total, 90, v);
    for chunk in [512usize, 4096] {
        let t = rig.with_model(|m| {
            // one warm-up pass, then the timed pass
            let run = |m: &mut MiniCpmModel| -> f64 {
                m.begin_slot(0).unwrap();
                let t = std::time::Instant::now();
                let mut done = 0;
                while done < total {
                    let n = chunk.min(total - done);
                    m.prefill_chunk(&chain(0, done, &p[done..done + n]), PrefillPath::Tiled, None).unwrap();
                    done += n;
                }
                t.elapsed().as_secs_f64()
            };
            run(m);
            run(m)
        });
        eprintln!("BENCH prefill {total} tokens in chunks of {chunk}: {:.0} ms = {:.0} tok/s", t * 1e3, total as f64 / t);
    }
    // the per-stage breakdown of one-row decode and of a 16-row window (eager launches: gaps included), and of a 512-row prefill chunk
    for (what, rows) in [("decode 1 row", 1usize), ("window 16 rows", 16)].into_iter().filter(|_| !prefill_only) {
        rig.with_model(|m| {
            m.set_graphs(false);
            m.begin_slot(0).unwrap();
            let c = prompt(ctx_len, 99, v);
            m.prefill_chunk(&chain(0, 0, &c[..c.len().min(4096)]), PrefillPath::for_prompt(c.len()), None).unwrap();
            let n = 12;
            let t = std::time::Instant::now();
            m.set_profile(true);
            for j in 0..n {
                m.window_candidates(&chain(0, ctx_len + j, &window_tokens(rows, j, v)), 8).unwrap();
                m.commit(0, 1);
                m.flush().unwrap();
            }
            let prof = m.take_profile();
            m.set_profile(false);
            let total: f64 = prof.iter().map(|p| p.1).sum();
            eprintln!(
                "BENCH stages, {what}, ctx {ctx_len}, {n} steps, eager, GPU time {:.2} ms a step ({:.2} ms wall a step incl. commit):",
                total / n as f64,
                t.elapsed().as_secs_f64() * 1e3 / n as f64
            );
            for (name, ms, calls) in prof.iter().take(14) {
                eprintln!("BENCH   {name:14} {:7.3} ms a step  {:5.1}%  ({} calls, {:.1} us a call)", ms / n as f64, 100.0 * ms / total, calls, ms * 1e3 / *calls as f64);
            }
        });
    }
    // the stage table of a chunk of 512 and of 4,096 rows (one chunk behind it, so the attention and the delta net have a context)
    for rows in [512usize, 4096] {
        rig.with_model(|m| {
            m.begin_slot(0).unwrap();
            let c = prompt(rows, 91, v);
            let more = prompt(rows, 92, v);
            m.prefill_chunk(&chain(0, 0, &c), PrefillPath::Tiled, None).unwrap();
            m.set_profile(true);
            m.prefill_chunk(&chain(0, rows, &more), PrefillPath::Tiled, None).unwrap();
            let prof = m.take_profile();
            m.set_profile(false);
            let total: f64 = prof.iter().map(|p| p.1).sum();
            eprintln!(
                "BENCH stages, prefill chunk of {rows} rows at position {rows}, GPU time {total:.1} ms = {:.0} tok/s:",
                rows as f64 / total * 1e3
            );
            for (name, ms, calls) in prof.iter().take(14) {
                eprintln!("BENCH   {name:14} {ms:8.2} ms  {:5.1}%  ({calls} calls)", 100.0 * ms / total);
            }
        });
    }
    // time to the first token's logits: a whole prompt in chunks of 4,096 (the pinned GEMM path), the head on the last row
    for &len in &ttft_lens {
        let p = prompt(len.min(1024), 93, v);
        let mut toks = p.clone();
        while toks.len() < len {
            let more = prompt(1024, 94 + toks.len() / 1024, v);
            toks.extend_from_slice(&more[..(len - toks.len()).min(1024)]);
        }
        let t = rig.with_model(|m| {
            m.begin_slot(0).unwrap();
            let path = PrefillPath::for_prompt(len);
            let t = std::time::Instant::now();
            let mut done = 0;
            while done < len {
                let n = 4096.min(len - done);
                let want = (done + n == len).then_some(8);
                m.prefill_chunk(&chain(0, done, &toks[done..done + n]), path, want).unwrap();
                done += n;
            }
            let t = t.elapsed().as_secs_f64();
            m.end_slot(0);
            t
        });
        eprintln!("BENCH TTFT {len:>6} tokens, chunks of 4096: {:.0} ms = {:.0} tok/s", t * 1e3, len as f64 / t);
    }
}

/// Decode and verify-window cost AT DEPTH, with the attention core separated (E-perf, docs/MINICPMV4_7_INTEGRATION_PLAN.md section 17).
///
/// One slot is prefilled in chunks of 4,096 tokens through the cuts of `REXL3_MINICPM_DEPTH_CUTS` (default `4096,32000,128000,261000`;
/// the pool is the 262,144-token one); at each cut the slot is stepped like the scorecard of section 14.4 (a window of `rows` rows
/// through the graph path, candidates read back, one row committed, the next forward replays it; the step tokens stay in the context
/// and the prefill resumes after them), for rows in `REXL3_MINICPM_DEPTH_ROWS` (default `1,4,8,16`), and then an eager, profiled pass
/// of one row and of 16 rows reports the stages with `attn.core` apart and the bytes a second the core achieves over the K/V it must
/// read (`20,480` B a token over the ten attention layers). Real prefill, so the K/V the kernels read is data, not untouched pages.
/// A timing lane: exclusive lock, nothing else running.
#[test]
#[ignore]
fn bench_decode_attention_at_depth() {
    let cuts: Vec<usize> = std::env::var("REXL3_MINICPM_DEPTH_CUTS")
        .unwrap_or_else(|_| "4096,32000,128000,261000".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let rows_list: Vec<usize> = std::env::var("REXL3_MINICPM_DEPTH_ROWS")
        .unwrap_or_else(|_| "1,4,8,16".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let steps = env_usize("REXL3_MINICPM_DEPTH_STEPS", 24);
    let top = cuts.iter().copied().max().unwrap_or(4096);
    let Some(rig) = Rig::load(1, (top + 4096).min(262_144).max(top + 64), 4096) else { return };
    let v = rig.vocab();
    let full_layers = rig.cfg.full_layers();
    let ids = prompt(top + 4096, 71, v);
    let mut at = 0usize; // the slot's length: the prompt so far and the step tokens committed after it
    rig.with_model(|m| {
        m.begin_slot(0).unwrap();
        eprintln!("BENCH DEPTH {} layers ({} attention), pool {} tokens, cuts {cuts:?}, rows {rows_list:?}, {steps} steps a point", rig.cfg.layers, full_layers, rig.max_ctx);
        let t_all = std::time::Instant::now();
        for &cut in &cuts {
            let t = std::time::Instant::now();
            while at < cut {
                let n = 4096.min(cut - at);
                m.prefill_chunk(&chain(0, at, &ids[at..at + n]), PrefillPath::Tiled, None).unwrap();
                at += n;
            }
            eprintln!("BENCH depth {cut}: prefilled in {:.1} s (total {:.1} s)", t.elapsed().as_secs_f64(), t_all.elapsed().as_secs_f64());
            // wall: window + candidates + commit + the next window's replay, median over steps after 6 warm-ups
            m.set_graphs(true);
            for &rows in &rows_list {
                let mut samples = Vec::with_capacity(steps);
                for j in 0..steps + 6 {
                    let b = chain(0, at, &window_tokens(rows, 5 + j, v));
                    let t = std::time::Instant::now();
                    m.window_candidates(&b, 8).unwrap();
                    m.commit(0, 1);
                    m.flush().unwrap();
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    at += 1;
                    if j >= 6 {
                        samples.push(ms);
                    }
                }
                samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = samples[samples.len() / 2];
                eprintln!(
                    "BENCH depth {cut:>6} window {rows:2} rows (graph): median {med:7.2} ms  min {:7.2}  max {:7.2}  ({:.1} tok/s one row)",
                    samples[0],
                    samples[samples.len() - 1],
                    1e3 / med
                );
            }
            // eager, profiled: the stage table of a one-row step and of a 16-row window (the attention core apart)
            m.set_graphs(false);
            for &rows in &[1usize, 16] {
                m.set_profile(true);
                let n = 8usize;
                for j in 0..n {
                    let b = chain(0, at, &window_tokens(rows, 40 + j, v));
                    m.window_candidates(&b, 8).unwrap();
                    m.commit(0, 1);
                    m.flush().unwrap();
                    at += 1;
                }
                let prof = m.take_profile();
                m.set_profile(false);
                let total: f64 = prof.iter().map(|p| p.1).sum::<f64>() / n as f64;
                let core = prof.iter().find(|p| p.0 == "attn.core").map(|p| p.1).unwrap_or(0.0) / n as f64;
                let kv_bytes = 20_480.0 * (at + rows) as f64 * full_layers as f64 / 10.0; // 20,480 B a token at 10 attention layers
                eprintln!(
                    "BENCH depth {cut:>6} stages, {rows:2} rows, eager: GPU {total:6.2} ms a step; attn.core {core:6.3} ms ({:5.1} us a layer) = {:6.1} GB/s over {:.2} GB of K/V; the rest {:6.2} ms",
                    core * 1e3 / full_layers as f64,
                    kv_bytes / (core * 1e-3) / 1e9,
                    kv_bytes / 1e9,
                    total - core
                );
                for (name, ms, calls) in prof.iter().take(8) {
                    eprintln!("BENCH     {name:14} {:7.3} ms a step  ({} calls)", ms / n as f64, calls);
                }
            }
        }
    });
}

#[test]
fn the_reference_dump_readers_decode_known_values() {
    for (h, x) in [(0x3c00u16, 1.0f32), (0xc000, -2.0), (0x3555, 0.333_251_95), (0x0001, 5.960_464_5e-8), (0x7bff, 65504.0)] {
        assert_eq!(f16(h), x, "{h:#06x}");
    }
    assert!(f16(0x7c00).is_infinite() && f16(0x7e00).is_nan());
    assert_eq!((bf16(0x3f80), bf16(0xc040)), (1.0, -3.0));
    // the position comparison: identical logits give cosine 1, KL 0 and the first maximum; a shifted copy keeps the argmax and the KL 0
    let raw: Vec<f32> = (0..64).map(|i| ((i * 37 % 64) as f32) * 0.25 - 4.0).collect();
    let engine: Vec<u16> = raw.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
    let max = raw.iter().cloned().fold(f32::MIN, f32::max);
    let half = |x: f32| -> u16 {
        // an exactly representable quarter-step value in IEEE half
        let a = x.abs();
        let e = a.log2().floor() as i32;
        let m = ((a / 2f32.powi(e) - 1.0) * 1024.0) as u16;
        let s = if x < 0.0 { 0x8000 } else { 0 };
        if x == 0.0 {
            0
        } else {
            s | (((e + 15) as u16) << 10) | m
        }
    };
    let centered: Vec<u8> = raw.iter().flat_map(|&x| half(x - max).to_le_bytes()).collect();
    let ps = compare_position(&engine, &centered, max, false);
    assert_eq!(ps.arg, raw.iter().position(|&x| x == max).unwrap());
    assert!((ps.cos - 1.0).abs() < 1e-9 && ps.kl.abs() < 1e-9, "cos {} kl {}", ps.cos, ps.kl);
    assert_eq!(ps.flip_gap, 0.0);
    assert!(ps.margin >= 0.25, "{}", ps.margin);
    let shifted = compare_position(&engine, &centered, max + 1.0, false);
    assert!(shifted.kl < 1e-9, "a shift of the whole row is not a change of the distribution");
}
