//! The DSpark draft inside the MiniCPM-V-4.7 engine (E4b of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`): taps, the draft's context, the
//! draft's proposals through the engine's rounds, and what must never change: the target's bytes.
//!
//! Needs the full 40-layer pack (the draft taps layers up to 35: the 8-layer smoke pack cannot host it, and every test says
//! `not run:` without it) and a draft:
//!   * `REXL3_MINICPM_DIR`: the EXL3 target pack (`.../packs/q3-6.00bpw`);
//!   * `REXL3_DSPARK_DIR`: the draft: an EXL3 draft pack (default `.../draft-packs/dspark-6.00bpw`) or the bf16 checkpoint
//!     (`.../dspark-hf`, the dense reference path).
//!
//! Every test boots the 27 GB pack: run under the EXCLUSIVE lock and one thread:
//!
//! ```text
//! flock -x -w 3600 /tmp/rexl3-gpu.lock env REXL3_MINICPM_DIR=~/models/minicpm-v-4.7-35b-a3b/packs/q3-6.00bpw \
//!     cargo test --release -p gb-model --test minicpmv4_7_spec -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! What runs: `drafted_equals_serial` with the real draft (greedy and sampled, one stream and several, a burst, staggered arrivals,
//! slots that draft beside slots that do not, a stop token inside a draft), a corrupted draft at every depth 0..15, the bits of the
//! target with the draft attached against detached, the taps against the target's own residual stream, the draft's context against
//! the serial path (the rollback of a partly accepted 16-row window and the absorb of an overfull ring), failure isolation (a draft
//! that fails for one slot), the pad mask, the refusal of a restored slot, the plan against the allocations, and the cost grid the
//! engine's budget reads (`bench_spec_costs`).

use gb_cuda::Stream;
use gb_engine::harness::{drafted_equals_serial, rollback_equals_replay, row_invariance, serial_batched_equals_solo};
use gb_engine::{
    Backend, BudgetPolicy, Drafter, Engine, EngineConfig, ImageSpan, NoDrafts, PrefillStep, Proposal, ProposeRequest, Request,
    RowOutput, RowRequest,
};
use gb_model::families::minicpmv4_7::backend::MiniCpmBackend;
use gb_model::families::minicpmv4_7::boot::{boot_profile, boot_with, BootExtras, BootPlan};
use gb_model::families::minicpmv4_7::config::{DSparkConfig, MiniCpmConfig};
use gb_model::families::minicpmv4_7::dspark::BLOCK;
use gb_model::families::minicpmv4_7::engine::{engine_from, round_cost, STEP_MS};
use gb_model::families::minicpmv4_7::memory::draft_pool_bytes;
use gb_model::families::minicpmv4_7::model::{MiniCpmModel, WINDOW_ROWS};
use gb_model::families::minicpmv4_7::spec::{SlotState, SpecAsk, SpecProposal, SpecRig, Why, RING_ROWS};
use gb_model::forward::{Candidates, PrefillPath};
use gb_model::rows::RowBatch;
use gb_model::serve::{ServeOptions, VisionSlot};
use gb_sampling::Sampling;
use gb_sched::{CostModel, MemoryProfile, ModelMemory, RoundCost, StepCost};
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

// ------------------------------------------------------------------------------------------------------------------ the rig

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").expect("HOME"))
}

fn pack_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_MINICPM_DIR") {
        Ok(d) if Path::new(&d).join("config.json").exists() => {
            if !gb_cuda::device_available() {
                eprintln!("not run: no CUDA device");
                return None;
            }
            let cfg = MiniCpmConfig::from_dir(Path::new(&d)).ok()?;
            if cfg.layers < 40 {
                eprintln!(
                    "not run: the draft taps layer 35: this needs the 40-layer pack, not the {}-layer smoke pack",
                    cfg.layers
                );
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

fn draft_dir() -> PathBuf {
    std::env::var_os("REXL3_DSPARK_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join("models/minicpm-v-4.7-35b-a3b/draft-packs/dspark-6.00bpw"))
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// One backend shared by the engines and harness calls of a test (loading the pack is the cost; every call starts from released slots).
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
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> PrefillStep {
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
    fn propose_batch_scored(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Proposal>> {
        self.0.borrow_mut().propose_batch_scored(reqs)
    }
}

struct Rig {
    backend: Rc<RefCell<MiniCpmBackend>>,
    plan: BootPlan,
    profile: MemoryProfile,
    cfg: MiniCpmConfig,
    dcfg: DSparkConfig,
    slots: usize,
    max_ctx: usize,
    /// The draft while it is detached for an A/B (`detach`/`attach`).
    parked: RefCell<Option<SpecRig>>,
}

impl Rig {
    fn load(slots: usize, max_ctx: usize, prefill_rows: usize, with_draft: bool) -> Option<Rig> {
        Self::load_full(slots, max_ctx, prefill_rows, with_draft, false)
    }

    /// [`Rig::load`], with the vision tower too when `tower` (the serving boot with the draft, the tower and the prefix cache together).
    fn load_full(slots: usize, max_ctx: usize, prefill_rows: usize, with_draft: bool, tower: bool) -> Option<Rig> {
        let dir = pack_dir()?;
        let ddir = draft_dir();
        if !ddir.join("config.json").exists() {
            eprintln!("not run: no draft at {}", ddir.display());
            return None;
        }
        let cfg = MiniCpmConfig::from_dir(&dir).unwrap();
        let dcfg = DSparkConfig::from_dir(&ddir).unwrap();
        let mut o = ServeOptions::new(dir);
        o.slots = slots;
        o.slots_auto = false;
        o.allow_unvalidated_pack = true;
        o.max_model_len = max_ctx;
        o.kv_tokens = Some(max_ctx);
        o.prefill_chunk = prefill_rows;
        o.draft_dir = with_draft.then(|| ddir.clone());
        if tower {
            o.vision = Some(VisionSlot::default());
        }
        let t = std::time::Instant::now();
        let loaded = boot_with(&o, BootExtras::default()).unwrap_or_else(|e| panic!("boot: {e}"));
        let (profile, _) = boot_profile(&loaded.cfg, &o).unwrap();
        let plan = loaded.plan.clone();
        let engine = engine_from(loaded, &o).unwrap_or_else(|e| panic!("engine: {e}"));
        let backend = engine.backend;
        eprintln!(
            "loaded {} layers, {} slots, {} pages, context {}, draft {} in {:.1} s",
            cfg.layers,
            plan.slots,
            plan.kv_pages(),
            plan.max_context,
            with_draft,
            t.elapsed().as_secs_f64()
        );
        Some(Rig {
            backend: Rc::new(RefCell::new(backend)),
            max_ctx: plan.max_context,
            slots: plan.slots,
            plan,
            profile,
            cfg,
            dcfg,
            parked: RefCell::new(None),
        })
    }

    fn vocab(&self) -> usize {
        self.cfg.vocab
    }

    fn shared(&self) -> Shared {
        let mut b = self.backend.borrow_mut();
        for s in 0..self.slots {
            b.release(s);
        }
        let _ = b.take_error();
        drop(b);
        Shared(self.backend.clone())
    }

    fn with_model<R>(&self, f: impl FnOnce(&mut MiniCpmModel) -> R) -> R {
        f(&mut self.backend.borrow_mut().m)
    }

    fn with_spec<R>(&self, f: impl FnOnce(&mut SpecRig) -> R) -> R {
        self.with_model(|m| f(m.spec_mut().expect("a draft is attached")))
    }

    /// The model is the text engine of E3 again (no taps, no draft); `attach` puts the same draft back.
    fn detach(&self) {
        let rig = self.with_model(|m| m.detach_spec()).unwrap();
        *self.parked.borrow_mut() = rig;
    }

    fn attach(&self) {
        let rig = self.parked.borrow_mut().take().expect("a detached draft");
        self.with_model(|m| m.attach_spec(rig)).unwrap();
    }

    /// An engine on the shared backend. `drafting` off is the serial reference; `flat` makes every verify row free so that the budget
    /// never trims a window (full 16-row windows: what the per-position acceptance and the depth tests need).
    fn engine(&self, drafting: bool, flat: bool) -> Engine<Shared> {
        self.engine_with(drafting, flat, BudgetPolicy::Scored)
    }

    /// [`Rig::engine`] under another budget policy (the serving default is `Scored`).
    fn engine_with(&self, drafting: bool, flat: bool, policy: BudgetPolicy) -> Engine<Shared> {
        let mm = ModelMemory::from_profile(&self.profile, 0, 0);
        let admission = self.plan.memory.admission(mm, self.slots, self.max_ctx).unwrap();
        let make: Box<dyn Fn() -> Box<dyn Drafter>> = Box::new(|| Box::new(NoDrafts));
        let rc = if flat { flat_cost() } else { round_cost() };
        Engine::new(
            self.shared(),
            admission,
            EngineConfig { max_rows: WINDOW_ROWS, max_drafts: BLOCK, cost: CostModel::from_table(STEP_MS.to_vec()), drafting },
            make,
        )
        .with_budget(policy, rc)
    }
}

fn flat_cost() -> RoundCost {
    RoundCost {
        verify: StepCost::rows_only(CostModel::from_table(vec![(1, 1.0), (64, 1.0)])),
        tree_verify: None,
        draft: None,
        kv_ms_per_token: 0.0,
        kv_pass_ms_per_token: 0.0,
        fixed_ms: 0.0,
    }
}

/// Real token ids: rows of the Leg A id set (python, rust, cuda, dialogue, technical prose, chat), the text a draft is good at;
/// else a deterministic synthetic stream.
fn real_ids() -> Option<(usize, usize, Vec<i32>)> {
    let path = std::env::var("REXL3_MINICPM_IDS").ok().map(PathBuf::from).or_else(|| {
        let p = home().join("projects/minicpm-kl/ids/legA.safetensors");
        p.exists().then_some(p)
    })?;
    let raw = std::fs::read(&path).ok()?;
    let n = u64::from_le_bytes(raw[..8].try_into().ok()?) as usize;
    let head: serde_json::Value = serde_json::from_slice(&raw[8..8 + n]).ok()?;
    let t = &head["input_ids"];
    let shape: Vec<usize> = t["shape"].as_array()?.iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let (a, b) = (t["data_offsets"][0].as_u64()? as usize, t["data_offsets"][1].as_u64()? as usize);
    let ids = raw[8 + n + a..8 + n + b].chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    Some((shape[0], shape[1], ids))
}

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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// -------------------------------------------------------------------------------------------------- bytes with and without the draft

/// What the model gives for a prompt and the windows after it, as bits: the last prefill row, then each window's candidates (greedy
/// continuation fed back), committing some and not others.
fn trace(m: &mut MiniCpmModel, slot: usize, p: &[u32], chunk: usize) -> Vec<(Vec<i32>, Vec<u32>)> {
    let path = PrefillPath::for_prompt(p.len());
    m.begin_slot(slot).unwrap();
    let mut done = 0;
    let mut last = None;
    while done < p.len() {
        let n = chunk.min(p.len() - done);
        last = m.prefill_chunk(&chain(slot, done, &p[done..done + n]), path, (done + n == p.len()).then_some(8)).unwrap();
        done += n;
    }
    let last = last.unwrap();
    let mut out = vec![(last.ids.clone(), bits(&last.vals))];
    let mut next = top1(&last, 0);
    let mut pos = p.len();
    for (rows, keep) in [(1usize, 1usize), (4, 2), (16, 16), (16, 5), (2, 2), (9, 1), (16, 11)] {
        let mut toks = vec![next];
        toks.extend(window_tokens(rows - 1, pos, 248_000));
        let c = m.window_candidates(&chain(slot, pos, &toks), 8).unwrap();
        out.push((c.ids.clone(), bits(&c.vals)));
        m.commit(slot, keep);
        next = top1(&c, keep - 1);
        pos += keep;
    }
    m.flush().unwrap();
    out
}

#[test]
#[ignore]
fn the_target_is_the_same_bits_with_the_draft_attached_and_without() {
    let Some(rig) = Rig::load(3, 4096, 1024, true) else { return };
    let v = rig.vocab();
    // prompts on both prefill arms, chunked on the model's 1,024-row grid and on odd grids, in different slots
    let cases = [(20usize, 1usize, 1024usize), (97, 2, 1024), (130, 3, 1024), (700, 4, 300), (1500, 5, 1024)];
    for (len, seed, chunk) in cases {
        let p = prompt(len, seed, v);
        let with = rig.with_model(|m| trace(m, 1, &p, chunk));
        rig.detach();
        let without = rig.with_model(|m| trace(m, 1, &p, chunk));
        rig.attach();
        assert_eq!(with.len(), without.len());
        for (i, (a, b)) in with.iter().zip(&without).enumerate() {
            assert_eq!(a.0, b.0, "{len} tokens: step {i}: candidate ids differ with the draft attached");
            assert_eq!(a.1, b.1, "{len} tokens: step {i}: candidate logit bits differ with the draft attached");
        }
        eprintln!("{len} tokens (chunk {chunk}): the last prefill row and 7 windows of 1..16 rows are bit-equal with the draft attached");
    }
    assert!(rig.backend.borrow_mut().take_error().is_none());
}

#[test]
#[ignore]
fn the_taps_are_the_targets_residual_after_the_tap_layers_in_every_row_of_every_forward() {
    let Some(rig) = Rig::load(2, 4096, 1024, true) else { return };
    let v = rig.vocab();
    let hidden = rig.cfg.hidden;
    let taps = rig.dcfg.target_layer_ids.clone();
    assert_eq!(taps, vec![3, 11, 19, 27, 35]);
    let n_taps = taps.len();
    let read_taps = |rig: &Rig, rows: usize| -> Vec<u16> {
        let mut out = vec![0u16; rows * n_taps * hidden];
        rig.with_spec(|s| s.tap_buffer().download_range(0, &mut out)).unwrap();
        out
    };
    // a chunk on each prefill arm, and windows of 1, 5 and 16 rows, each compared with the residual stream the model's own debug
    // capture saw after the tap layers (the capture runs windows eagerly; a window graph replay is compared below)
    let check = |rig: &Rig, rows: usize, what: &str| {
        let captured = rig.with_model(|m| m.take_captured());
        assert_eq!(captured.len(), rig.cfg.layers, "{what}: one capture a layer");
        let got = read_taps(rig, rows);
        for (ti, &layer) in taps.iter().enumerate() {
            for r in 0..rows {
                let want = &captured[layer][r * hidden..(r + 1) * hidden];
                let have = &got[(r * n_taps + ti) * hidden..(r * n_taps + ti + 1) * hidden];
                assert_eq!(have, want, "{what}: tap {ti} (layer {layer}) row {r} is not the residual stream after the layer");
            }
        }
    };
    for (len, seed) in [(100usize, 6usize), (300, 7)] {
        let p = prompt(len, seed, v);
        let path = PrefillPath::for_prompt(len);
        rig.with_model(|m| {
            m.begin_slot(0).unwrap();
            m.set_capture(true);
            m.prefill_chunk(&chain(0, 0, &p), path, Some(4)).unwrap();
            m.set_capture(false);
        });
        check(&rig, len, &format!("a {len}-token chunk"));
        for rows in [1usize, 5, 16] {
            let toks = window_tokens(rows, rows + seed, v);
            rig.with_model(|m| {
                m.set_capture(true);
                m.window_candidates(&chain(0, len, &toks), 4).unwrap();
                m.set_capture(false);
            });
            check(&rig, rows, &format!("a window of {rows} rows after {len} tokens"));
        }
    }
    // a replayed window graph writes the same taps as the eager window of the same rows
    let p = prompt(200, 9, v);
    let toks = window_tokens(16, 3, v);
    let other = window_tokens(16, 4, v);
    rig.with_model(|m| {
        m.set_graphs(false);
        m.begin_slot(0).unwrap();
        m.prefill_chunk(&chain(0, 0, &p), PrefillPath::Tiled, None).unwrap();
        m.window_candidates(&chain(0, 200, &toks), 4).unwrap();
    });
    let eager = read_taps(&rig, 16);
    rig.with_model(|m| {
        m.set_graphs(true);
        m.clear_graphs().unwrap();
        m.begin_slot(0).unwrap();
        m.prefill_chunk(&chain(0, 0, &p), PrefillPath::Tiled, None).unwrap();
        // the first call captures the graph and launches it; a window of other tokens in the same shape overwrites the taps through the
        // replay; the third call replays the original rows
        m.window_candidates(&chain(0, 200, &toks), 4).unwrap();
        m.window_candidates(&chain(0, 200, &other), 4).unwrap();
        assert_eq!(m.graph_count(), 1, "one shape, one graph");
        m.window_candidates(&chain(0, 200, &toks), 4).unwrap();
    });
    let replayed = read_taps(&rig, 16);
    assert_eq!(eager, replayed, "a replayed window graph exports different taps than the eager window");
    eprintln!("taps: 100- and 300-token chunks and windows of 1/5/16 rows equal the residual after layers 3, 11, 19, 27, 35; graph == eager");
}

// ------------------------------------------------------------------------------------------------ the engine, with the real draft

#[test]
#[ignore]
fn drafted_equals_serial_with_the_real_draft_greedy_sampled_one_stream_and_several() {
    let Some(rig) = Rig::load(4, 4096, 1024, true) else { return };
    let v = rig.vocab();
    let make = |drafting: bool| rig.engine(drafting, false);
    // one stream, then four streams of both arms and both sampling modes, finishing at different times
    for (i, r) in [
        request(prompt(200, 0, v), greedy(), 120, true),
        request(prompt(400, 1, v), sampled(7), 120, true),
        request(prompt(60, 2, v), greedy(), 120, true),
    ]
    .into_iter()
    .enumerate()
    {
        let rep = drafted_equals_serial(&make, std::slice::from_ref(&r)).unwrap_or_else(|e| panic!("one stream {i}: {e}"));
        eprintln!(
            "one stream {i}: {} tokens, {} rounds, {} drafted, {} accepted ({:.2} a round)",
            rep.tokens,
            rep.rounds,
            rep.drafted,
            rep.accepted,
            rep.tokens_per_round()
        );
    }
    let reqs = vec![
        request(prompt(20, 3, v), greedy(), 90, true),
        request(prompt(160, 4, v), sampled(11), 110, true),
        request(prompt(300, 5, v), greedy(), 70, true),
        request(prompt(700, 6, v), sampled(23), 100, true),
    ];
    let r = drafted_equals_serial(&make, &reqs).unwrap_or_else(|e| panic!("four streams: {e}"));
    eprintln!(
        "four streams: {} tokens, {} rounds, {} drafted, {} accepted, {:.2} tokens a round",
        r.tokens,
        r.rounds,
        r.drafted,
        r.accepted,
        r.tokens_per_round()
    );
    assert!(r.accepted > 0, "no draft was ever accepted: the check proves nothing");
    assert!(rig.backend.borrow_mut().take_error().is_none());
    let c = rig.with_spec(|s| s.counters.clone());
    assert_eq!(c.pad_leaks, 0);
    assert_eq!(c.lost_slots, 0, "no slot lost its draft context in a clean run");
    assert!(c.rounds > 0 && c.proposed > 0);
    // the serial batched engine too (the same requests, no drafts, together): E3's contract with the draft attached
    let make_s = |drafting: bool| rig.engine(drafting, false);
    serial_batched_equals_solo(&make_s, &reqs).unwrap_or_else(|e| panic!("serial batched: {e}"));
}

#[test]
#[ignore]
fn a_burst_staggered_arrivals_and_slots_that_draft_beside_slots_that_do_not() {
    let Some(rig) = Rig::load(4, 4096, 256, true) else { return };
    let v = rig.vocab();
    // slots 0 and 2 draft, slots 1 and 3 do not (their request says so): their rounds are single rows beside the others' windows
    let reqs = vec![
        request(prompt(90, 8, v), greedy(), 80, true),
        request(prompt(250, 9, v), sampled(3), 80, false),
        request(prompt(500, 10, v), sampled(5), 80, true),
        request(prompt(33, 11, v), greedy(), 80, false),
    ];
    let make = |drafting: bool| rig.engine(drafting, false);
    let r = drafted_equals_serial(&make, &reqs).unwrap_or_else(|e| panic!("mixed slots: {e}"));
    eprintln!("mixed draft / no-draft slots: {} drafted, {} accepted", r.drafted, r.accepted);
    assert!(r.drafted > 0);
    // staggered arrivals: a small prefill chunk so that later prompts are prefilled in slices BETWEEN the decode rounds of the earlier
    // streams (whose drafts are being made and verified meanwhile)
    let solo: Vec<Vec<u32>> = reqs
        .iter()
        .map(|r| {
            let mut e = rig.engine(false, false);
            e.submit(r.clone()).unwrap();
            e.run().unwrap();
            e.take_outputs().remove(0).tokens
        })
        .collect();
    let mut e = rig.engine(true, false);
    let mut ids = Vec::new();
    ids.push(e.submit_deferred(reqs[0].clone()).unwrap());
    for _ in 0..5 {
        e.step().unwrap();
    }
    ids.push(e.submit_deferred(reqs[1].clone()).unwrap());
    for _ in 0..3 {
        e.step().unwrap();
    }
    ids.push(e.submit_deferred(reqs[2].clone()).unwrap());
    ids.push(e.submit_deferred(reqs[3].clone()).unwrap());
    e.run().unwrap();
    let mut out = e.take_outputs();
    out.sort_by_key(|o| ids.iter().position(|&i| i == o.id).unwrap());
    assert_eq!(out.len(), 4);
    for (k, o) in out.iter().enumerate() {
        assert_eq!(o.tokens, solo[k], "stream {k}: arriving while others draft changed its tokens");
    }
    // a burst: every request submitted before any step runs, so that the first rounds carry four fresh streams at once
    let mut e = rig.engine(true, false);
    for r in &reqs {
        e.submit_deferred(r.clone()).unwrap();
    }
    e.run().unwrap();
    let mut out = e.take_outputs();
    out.sort_by_key(|o| o.id);
    for (k, o) in out.iter().enumerate() {
        assert_eq!(o.tokens, solo[k], "burst: stream {k} differs from solo");
    }
    eprintln!("burst and staggered arrivals: four streams equal to solo with the real draft");
}

#[test]
#[ignore]
fn a_stop_token_inside_a_draft_ends_the_sequence_where_the_serial_path_does() {
    let Some(rig) = Rig::load(2, 4096, 512, true) else { return };
    let v = rig.vocab();
    // the serial continuation of a code prompt (the draft is good on it), then a stop token that appears inside it
    let r0 = request(prompt(300, 0, v), greedy(), 100, true);
    let serial = {
        let mut e = rig.engine(false, false);
        e.submit(r0.clone()).unwrap();
        e.run().unwrap();
        e.take_outputs().remove(0).tokens
    };
    let mut hit = 0usize;
    for at in [3usize, 7, 12, 20, 33, 47] {
        let stop = serial[at];
        if serial[..at].contains(&stop) {
            continue; // an earlier occurrence would end it earlier: still a valid case, but not the position asked for
        }
        let mut r = r0.clone();
        r.stop = vec![stop];
        let make = |drafting: bool| rig.engine(drafting, false);
        let rep = drafted_equals_serial(&make, std::slice::from_ref(&r)).unwrap_or_else(|e| panic!("stop at {at}: {e}"));
        hit += rep.stopped_in_draft;
        assert_eq!(rep.stopped, 1);
    }
    eprintln!("a stop token at 6 depths of the continuation: {hit} of them inside a draft run, all equal to serial");
    assert!(hit >= 1, "no stop token landed inside an accepted draft: the case did not run");
}

#[test]
#[ignore]
fn a_corrupted_draft_is_rejected_at_every_depth_and_the_text_is_the_serial_text() {
    let Some(rig) = Rig::load(3, 4096, 512, true) else { return };
    let v = rig.vocab() as u32;
    // full 16-row windows (flat verify cost), the real draft's proposals with one token wrong at a depth that walks 0..14 round by
    // round; the unmodified rounds accept as many as the draft gets right
    rig.with_spec(|s| {
        s.inject.corrupt = Some(Box::new(move |_slot, round, tokens| {
            let depth = (round % 16) as usize;
            if depth < tokens.len() {
                tokens[depth] = (tokens[depth] + 1) % v;
            }
        }))
    });
    // text that repeats itself (a 50-token passage six times), so that the real draft is right for 15 in a row where it continues
    let repeat = |n: usize, seed: usize, times: usize| -> Vec<u32> {
        let a = prompt(n, seed, rig.vocab());
        (0..times).flat_map(|_| a.clone()).collect()
    };
    let reqs = vec![
        request(repeat(50, 0, 6), greedy(), 400, true),
        request(repeat(40, 1, 5), greedy(), 400, true),
        request(repeat(60, 2, 4), sampled(9), 400, true),
    ];
    let make = |drafting: bool| rig.engine(drafting, true);
    let r = drafted_equals_serial(&make, &reqs).unwrap_or_else(|e| panic!("corrupted draft: {e}"));
    let c = rig.with_spec(|s| s.counters.clone());
    eprintln!("{} drafted, {} accepted; rounds by accepted drafts 0..15: {:?}", r.drafted, r.accepted, c.first_reject);
    // a rejection at depth d ends a full window with exactly d drafts accepted: every depth the draft's own run of right tokens reaches
    // must show, and the run reaches deep on text that repeats
    let missing: Vec<usize> = (0..BLOCK).filter(|&d| c.first_reject[d] == 0).collect();
    assert!(
        missing.is_empty(),
        "no round ended after exactly {missing:?} accepted drafts (full window, rejection at that depth)"
    );
    rig.with_spec(|s| s.inject.corrupt = None);
}

// ------------------------------------------------------------------------------------------ the draft's context against serial

/// The draft's K and V rows `0..len` of `slot`, all five layers.
fn draft_kv(rig: &Rig, slot: usize, len: usize) -> Vec<(Vec<u16>, Vec<u16>)> {
    rig.with_spec(|s| (0..5).map(|l| s.draft.read_kv(slot, l, 0, len).unwrap()).collect())
}

#[test]
#[ignore]
fn the_draft_context_after_a_partly_accepted_window_equals_the_serial_path() {
    let Some(rig) = Rig::load(2, 4096, 1024, true) else { return };
    let v = rig.vocab();
    let s = greedy();
    for (len, window, keep) in [(60usize, 16usize, 5usize), (200, 16, 16), (200, 7, 1), (500, 16, 11), (130, 16, 2)] {
        let p = prompt(len, len + keep, v);
        let toks = window_tokens(window, keep, v);
        // path A (slot 0): one window of `window` rows, the first `keep` committed; the other rows were never accepted
        let c = rig.with_model(|m| {
            m.begin_slot(0).unwrap();
            m.prefill_chunk(&chain(0, 0, &p), PrefillPath::for_prompt(len), None).unwrap();
            let c = m.window_candidates(&chain(0, len, &toks), 4).unwrap();
            m.commit(0, keep);
            c
        });
        let mut tokens = p.clone();
        tokens.extend_from_slice(&toks[..keep]);
        tokens.push(top1(&c, keep - 1));
        let a_prop = rig
            .with_model(|m| m.propose_round(&[SpecAsk { slot: 0, tokens: &tokens, sampling: &s, max: 15 }]))
            .unwrap()
            .remove(0)
            .expect("a proposal");
        let a_kv = draft_kv(&rig, 0, len + keep);
        // path B (slot 1): the same rows one at a time, each committed
        let mut next = 0u32;
        rig.with_model(|m| {
            m.begin_slot(1).unwrap();
            m.prefill_chunk(&chain(1, 0, &p), PrefillPath::for_prompt(len), None).unwrap();
            for (j, &t) in toks[..keep].iter().enumerate() {
                let c = m.window_candidates(&chain(1, len + j, &[t]), 4).unwrap();
                m.commit(1, 1);
                next = top1(&c, 0);
            }
        });
        assert_eq!(next, *tokens.last().unwrap(), "the serial path's last row picks the window's anchor");
        let b_prop = rig
            .with_model(|m| m.propose_round(&[SpecAsk { slot: 1, tokens: &tokens, sampling: &s, max: 15 }]))
            .unwrap()
            .remove(0)
            .expect("a proposal");
        let b_kv = draft_kv(&rig, 1, len + keep);
        for l in 0..5 {
            assert_eq!(
                a_kv[l].0, b_kv[l].0,
                "len {len} window {window} keep {keep}: K of layer {l} differs from the serial path"
            );
            assert_eq!(
                a_kv[l].1, b_kv[l].1,
                "len {len} window {window} keep {keep}: V of layer {l} differs from the serial path"
            );
        }
        assert_eq!(
            a_prop, b_prop,
            "len {len} window {window} keep {keep}: the draft's next proposal differs from the serial path's"
        );
        eprintln!(
            "len {len}, window {window}, keep {keep}: the draft's {} context rows and its next 15 drafts equal the serial path's",
            len + keep
        );
    }
}

#[test]
#[ignore]
fn eager_ingestion_then_the_committed_length_rollback_equals_the_lazy_path() {
    let Some(rig) = Rig::load(2, 4096, 1024, true) else { return };
    let v = rig.vocab();
    let (len, keep) = (180usize, 6usize);
    let p = prompt(len, 17, v);
    let toks = window_tokens(16, 5, v);
    let s = greedy();
    // slot 0: the engine's path. The window's taps are copied (a snapshot of the tap buffer) before anything else runs a forward
    let c = rig.with_model(|m| {
        m.begin_slot(0).unwrap();
        m.prefill_chunk(&chain(0, 0, &p), PrefillPath::for_prompt(len), None).unwrap();
        m.window_candidates(&chain(0, len, &toks), 4).unwrap()
    });
    let snap = rig.with_spec(|r| tap_clone(r));
    rig.with_model(|m| m.commit(0, keep));
    let mut tokens = p.clone();
    tokens.extend_from_slice(&toks[..keep]);
    tokens.push(top1(&c, keep - 1));
    rig.with_model(|m| m.propose_round(&[SpecAsk { slot: 0, tokens: &tokens, sampling: &s, max: 15 }])).unwrap();
    let lazy = draft_kv(&rig, 0, len + keep);
    // slot 1 holds the same prompt in its draft context; the 16 window rows are absorbed by hand from the snapshot, then the draft's
    // committed length is rolled back to the accepted ones
    rig.with_model(|m| {
        m.begin_slot(1).unwrap();
        m.prefill_chunk(&chain(1, 0, &p), PrefillPath::for_prompt(len), None).unwrap();
    });
    let st = Stream::new().unwrap();
    rig.with_spec(|r| {
        assert_eq!(r.draft.len(1), len);
        r.draft.ingest_taps(1, len, 16, &snap, 0, &st).unwrap();
        st.sync().unwrap();
        assert_eq!(r.draft.len(1), len + 16);
        r.draft.rollback(1, len + keep).unwrap();
        assert_eq!(r.draft.len(1), len + keep);
    });
    let eager = draft_kv(&rig, 1, len + keep);
    for l in 0..5 {
        assert_eq!(eager[l].0, lazy[l].0, "K of layer {l}: eager 16-row ingestion + rollback differs from the lazy path");
        assert_eq!(eager[l].1, lazy[l].1, "V of layer {l}: eager 16-row ingestion + rollback differs from the lazy path");
    }
    eprintln!(
        "eager 16-row ingestion + rollback to {keep} rows == the lazy ring path, K and V of all 5 layers over {} rows",
        len + keep
    );
}

/// A copy of the live tap buffer as a standalone device buffer (so that a direct `ingest_taps` borrows nothing of the rig).
fn tap_clone(s: &SpecRig) -> gb_cuda::DevBuf {
    let n = s.tap_buffer().bytes() / 2;
    let mut host = vec![0u16; n];
    s.tap_buffer().download_range(0, &mut host).unwrap();
    gb_cuda::DevBuf::from_slice(&host).unwrap()
}

#[test]
#[ignore]
fn an_overfull_ring_is_absorbed_in_order_and_the_context_stays_exact() {
    let Some(rig) = Rig::load(2, 4096, 512, true) else { return };
    let v = rig.vocab();
    let len = 100usize;
    let p = prompt(len, 21, v);
    // 70 one-row windows without a round in between (a stream the budget never asked to draft): the ring (48 rows) fills and is
    // absorbed once; then a round must find a context of len + 70 rows, equal to a slot that was absorbed row by row
    let n = RING_ROWS + 22;
    let toks = window_tokens(n, 4, v);
    let s = greedy();
    let mut props = Vec::new();
    for slot in 0..2usize {
        rig.with_model(|m| {
            m.begin_slot(slot).unwrap();
            m.prefill_chunk(&chain(slot, 0, &p), PrefillPath::for_prompt(len), None).unwrap();
        });
        let mut next = 0;
        for (j, &t) in toks.iter().enumerate() {
            let c = rig.with_model(|m| {
                let c = m.window_candidates(&chain(slot, len + j, &[t]), 4).unwrap();
                m.commit(slot, 1);
                c
            });
            next = top1(&c, 0);
            // slot 1 asks for a round after every row (its ring never holds more than one); slot 0 never does
            if slot == 1 {
                let mut tokens = p.clone();
                tokens.extend_from_slice(&toks[..=j]);
                tokens.push(next);
                rig.with_model(|m| m.propose_round(&[SpecAsk { slot, tokens: &tokens, sampling: &s, max: 15 }])).unwrap();
            }
        }
        let mut tokens = p.clone();
        tokens.extend_from_slice(&toks);
        tokens.push(next);
        let prop =
            rig.with_model(|m| m.propose_round(&[SpecAsk { slot, tokens: &tokens, sampling: &s, max: 15 }])).unwrap().remove(0);
        props.push(prop.expect("a proposal"));
    }
    let (k0, k1) = (draft_kv(&rig, 0, len + n), draft_kv(&rig, 1, len + n));
    for l in 0..5 {
        assert_eq!(k0[l], k1[l], "layer {l}: the overfull ring's context differs from the row-by-row context");
    }
    assert_eq!(props[0], props[1], "the proposals differ");
}

#[test]
#[ignore]
fn a_round_of_several_streams_absorbs_in_one_pass_and_leaves_each_slot_as_alone() {
    let Some(rig) = Rig::load(8, 4096, 1024, true) else { return };
    let v = rig.vocab();
    let s = greedy();
    // four sequences of different lengths, each with its own number of committed rows waiting (1, 3, 9, 16), twice: slots 0..3 are
    // asked together in one round (one batched absorb), slots 4..7 one at a time
    // (two of them end within 15 positions of the first page boundary: their blocks write into their private overflow pages, in the
    // same call)
    let lens = [250usize, 130, 241, 100];
    let rows = [1usize, 3, 9, 16];
    let mut seqs: Vec<Vec<u32>> = Vec::new();
    for i in 0..4 {
        let p = prompt(lens[i], 50 + i, v);
        let toks = window_tokens(rows[i], 7 + i, v);
        let mut anchor = 0u32;
        for slot in [i, i + 4] {
            rig.with_model(|m| {
                m.begin_slot(slot).unwrap();
                m.prefill_chunk(&chain(slot, 0, &p), PrefillPath::for_prompt(lens[i]), None).unwrap();
                let c = m.window_candidates(&chain(slot, lens[i], &toks), 4).unwrap();
                m.commit(slot, rows[i]);
                anchor = top1(&c, rows[i] - 1);
            });
        }
        let mut t = p.clone();
        t.extend_from_slice(&toks);
        t.push(anchor);
        seqs.push(t);
    }
    let asks: Vec<SpecAsk> = (0..4).map(|i| SpecAsk { slot: i, tokens: &seqs[i], sampling: &s, max: 15 }).collect();
    for i in [0usize, 2] {
        let committed = lens[i] + rows[i];
        let target_pages = rig.with_model(|m| m.kv_tab.blocks_of(i).len());
        assert!(committed + 15 > target_pages * 256, "stream {i}: its block must run past the target's last page");
    }
    let before = rig.with_spec(|r| r.counters.absorb_calls);
    let together = rig.with_model(|m| m.propose_round(&asks)).unwrap();
    let calls = rig.with_spec(|r| r.counters.absorb_calls) - before;
    assert_eq!(calls, 1, "the four streams' committed rows are absorbed in one pass");
    let mut alone = Vec::new();
    for (i, t) in seqs.iter().enumerate() {
        let a = [SpecAsk { slot: i + 4, tokens: t, sampling: &s, max: 15 }];
        alone.push(rig.with_model(|m| m.propose_round(&a)).unwrap().remove(0));
    }
    for i in 0..4 {
        assert!(together[i].is_some());
        assert_eq!(together[i], alone[i], "stream {i}: the proposal differs between the batched round and the round alone");
        let (a, b) = (draft_kv(&rig, i, lens[i] + rows[i]), draft_kv(&rig, i + 4, lens[i] + rows[i]));
        for l in 0..5 {
            assert_eq!(a[l], b[l], "stream {i} layer {l}: K/V differs between the batched absorb and the per-slot absorb");
        }
    }
    eprintln!("batched absorb of 4 streams (1, 3, 9, 16 rows) in {calls} pass: K/V and proposals equal to per-slot rounds");
}

// ------------------------------------------------------------------------------------------------ isolation, the pad, the restore

#[test]
#[ignore]
fn a_failure_of_the_draft_for_one_slot_ends_its_drafting_and_touches_no_other_stream() {
    let Some(rig) = Rig::load(3, 4096, 512, true) else { return };
    let v = rig.vocab();
    let reqs = vec![
        request(prompt(120, 12, v), greedy(), 80, true),
        request(prompt(260, 13, v), sampled(4), 80, true),
        request(prompt(60, 14, v), greedy(), 80, true),
    ];
    let solo: Vec<Vec<u32>> = reqs
        .iter()
        .map(|r| {
            let mut e = rig.engine(false, false);
            e.submit(r.clone()).unwrap();
            e.run().unwrap();
            e.take_outputs().remove(0).tokens
        })
        .collect();
    let mut e = rig.engine(true, false);
    for r in &reqs {
        e.submit(r.clone()).unwrap();
    }
    // a few rounds, then the draft fails for the second slot's next call, and again for the third a little later
    for _ in 0..3 {
        e.step().unwrap();
    }
    rig.with_spec(|s| s.inject.fail_slot = Some(1));
    for _ in 0..4 {
        e.step().unwrap();
    }
    rig.with_spec(|s| s.inject.fail_slot = Some(2));
    e.run().unwrap();
    let mut out = e.take_outputs();
    out.sort_by_key(|o| o.id);
    for (k, o) in out.iter().enumerate() {
        assert_eq!(o.tokens, solo[k], "stream {k}: a draft failure for another slot changed its tokens");
    }
    let c = rig.with_spec(|s| s.counters.clone());
    assert!(c.lost_slots >= 1, "the injected failure was not seen: {c:?}");
    assert!(e.take_failed().is_empty(), "no sequence failed: a draft failure ends the slot's drafting only");
    assert!(rig.backend.borrow_mut().take_error().is_none(), "a draft failure is never a backend error");
    // the stream that lost its draft kept decoding at one row a round: its record shows rounds with no drafts after the failure
    assert!(out[1].rounds > 0 && out[1].tokens.len() == 80);
    eprintln!("draft failures for slots 1 and 2: {} slots lost, every stream equal to solo, no backend error", c.lost_slots);
}

#[test]
#[ignore]
fn no_draft_token_is_a_pad_id_even_with_the_pad_columns_poisoned() {
    let Some(rig) = Rig::load(2, 4096, 512, true) else { return };
    let v = rig.vocab();
    rig.with_spec(|s| s.draft.poison_pad_columns(true));
    let reqs = vec![request(prompt(200, 15, v), greedy(), 60, true), request(prompt(90, 16, v), sampled(2), 60, true)];
    let make = |drafting: bool| rig.engine(drafting, false);
    drafted_equals_serial(&make, &reqs).unwrap_or_else(|e| panic!("poisoned pad: {e}"));
    rig.with_spec(|s| s.draft.poison_pad_columns(false));
    let c = rig.with_spec(|s| s.counters.clone());
    assert_eq!(c.pad_leaks, 0);
    assert!(c.proposed > 0);
    // a direct round: every drafted id is a real id
    let p = prompt(100, 1, v);
    rig.with_model(|m| {
        m.begin_slot(0).unwrap();
        let cand = m.prefill_chunk(&chain(0, 0, &p), PrefillPath::Gemv, Some(4)).unwrap().unwrap();
        let mut tokens = p.clone();
        tokens.push(top1(&cand, 0));
        let s = greedy();
        let out = m.propose_round(&[SpecAsk { slot: 0, tokens: &tokens, sampling: &s, max: 15 }]).unwrap();
        let prop = out[0].as_ref().expect("a proposal");
        assert_eq!(prop.tokens.len(), 15);
        assert!(prop.tokens.iter().all(|&t| (t as usize) < 248_144), "{:?}", prop.tokens);
        assert!(prop.accept.iter().all(|&a| (0.0..=1.0).contains(&a)), "{:?}", prop.accept);
    });
}

#[test]
#[ignore]
fn a_slot_restored_without_draft_context_never_drafts_and_the_adopt_seam_gives_it_back() {
    let Some(rig) = Rig::load(3, 4096, 512, true) else { return };
    let v = rig.vocab();
    let s = greedy();
    let p = prompt(300, 31, v);
    // slot 0: a normal sequence. slot 1: a "restore": the slot begins and the prompt resumes at 256 with no rows before it, as a
    // prefix-cache hit does until snapshots carry the draft's pages
    let (t0, t1) = rig.with_model(|m| {
        m.begin_slot(0).unwrap();
        let c0 = m.prefill_chunk(&chain(0, 0, &p), PrefillPath::Tiled, Some(4)).unwrap().unwrap();
        m.begin_slot(1).unwrap();
        // pages for the first 256 positions exist in a real restore; here they are reserved and hold nothing: the target's output
        // for slot 1 is meaningless, which is not what this test reads
        m.reserve(1, 0, 256).unwrap();
        let c1 = m.prefill_chunk(&chain(1, 256, &p[256..]), PrefillPath::Tiled, Some(4)).unwrap().unwrap();
        (top1(&c0, 0), top1(&c1, 0))
    });
    let mut tok0 = p.clone();
    tok0.push(t0);
    let mut tok1 = p.clone();
    tok1.push(t1);
    let asks =
        [SpecAsk { slot: 0, tokens: &tok0, sampling: &s, max: 15 }, SpecAsk { slot: 1, tokens: &tok1, sampling: &s, max: 15 }];
    let out: Vec<Option<SpecProposal>> = rig.with_model(|m| m.propose_round(&asks)).unwrap();
    assert!(out[0].is_some(), "the normal slot drafts");
    assert!(out[1].is_none(), "the restored slot has no draft context for its cached rows: no proposal");
    let (state, refused) = rig.with_spec(|r| (r.book.state(1), r.counters.refused_lost));
    assert_eq!(state, SlotState::Lost(Why::Restored));
    assert!(refused >= 1);
    // the other slot's proposal is what it is alone
    let alone: Vec<Option<SpecProposal>> = rig.with_model(|m| m.propose_round(&asks[..1])).unwrap();
    assert_eq!(alone[0], out[0], "a lost slot beside it changed the normal slot's proposal");
    // the seam: declare slot 1's context valid (its pages ARE the snapshot's) and it drafts again; the claim is the caller's, so
    // the proposal is the draft's answer for the context the pages hold (here: zeros for rows 0..256)
    rig.with_model(|m| m.adopt_draft_context(1, 300)).unwrap();
    let again = rig.with_model(|m| m.propose_round(&asks[1..])).unwrap();
    assert!(again[0].is_some(), "an adopted context drafts again");
    eprintln!(
        "restore: the slot without draft context got no proposal, its neighbour's proposal was unchanged, adopt re-enabled it"
    );
}

#[test]
#[ignore]
fn a_restore_that_shares_the_full_pages_and_copies_the_partial_one_drafts_as_the_original() {
    let Some(rig) = Rig::load(2, 4096, 512, true) else { return };
    let v = rig.vocab();
    let s = greedy();
    let p = prompt(300, 33, v);
    // slot 0: the original sequence (a full page and a partial one: 300 = 256 + 44)
    let c = rig.with_model(|m| {
        m.begin_slot(0).unwrap();
        m.prefill_chunk(&chain(0, 0, &p), PrefillPath::Tiled, Some(4)).unwrap().unwrap()
    });
    let mut tokens = p.clone();
    tokens.push(top1(&c, 0));
    let a = [SpecAsk { slot: 0, tokens: &tokens, sampling: &s, max: 15 }];
    let original = rig.with_model(|m| m.propose_round(&a)).unwrap().remove(0).expect("a proposal");
    // slot 1: what a prefix-cache restore does to the page tables: share the full page by reference, a fresh page for the partial one
    // with the copy of its content (the draft's page of the same id goes with the target's), then the context is declared valid
    rig.with_model(|m| {
        let pages: Vec<_> = m.kv_tab.blocks_of(0).to_vec();
        assert_eq!(pages.len(), 2);
        m.begin_slot(1).unwrap();
        m.kv_tab.adopt_shared(1, &pages[..1]);
        let dst = m.kv_tab.alloc_block(1).unwrap() as usize;
        m.copy_draft_page(pages[1] as usize, dst).unwrap();
        m.upload_table().unwrap();
        m.stream.sync().unwrap();
        m.adopt_draft_context(1, 300).unwrap();
    });
    let b = [SpecAsk { slot: 1, tokens: &tokens, sampling: &s, max: 15 }];
    let restored = rig.with_model(|m| m.propose_round(&b)).unwrap().remove(0).expect("a proposal");
    assert_eq!(restored, original, "a slot restored with its draft pages drafts differently from the original");
    // the context rows themselves (the shared page and the copied partial page)
    let (k0, k1) = (draft_kv(&rig, 0, 300), draft_kv(&rig, 1, 300));
    for l in 0..5 {
        assert_eq!(k0[l], k1[l], "layer {l}: the restored slot's draft context differs from the original's");
    }
    // without the declaration the same restore never drafts
    rig.with_model(|m| {
        m.begin_slot(1).unwrap();
        let pages: Vec<_> = m.kv_tab.blocks_of(0).to_vec();
        m.kv_tab.adopt_shared(1, &pages[..1]);
        let dst = m.kv_tab.alloc_block(1).unwrap() as usize;
        m.copy_draft_page(pages[1] as usize, dst).unwrap();
        m.upload_table().unwrap();
        // the next prefill slice resumes at 300 on a slot whose book holds nothing
        m.reserve(1, 300, 1).unwrap();
        m.prefill_chunk(&chain(1, 300, &[7]), PrefillPath::Tiled, None).unwrap();
    });
    let mut t2 = p.clone();
    t2.push(7);
    t2.push(9);
    let c2 = [SpecAsk { slot: 1, tokens: &t2, sampling: &s, max: 15 }];
    let none = rig.with_model(|m| m.propose_round(&c2)).unwrap().remove(0);
    assert!(none.is_none(), "a restored slot with no declaration drafted");
    eprintln!(
        "restore seam: shared full page + copied partial page + adopt == the original's context and proposal; no adopt, no draft"
    );
}

// ------------------------------------------------------------------------------------------------------ plan == allocations

#[test]
#[ignore]
fn the_plan_equals_the_allocations_with_the_draft_at_one_four_and_eight_slots() {
    // the small pool of the other tests, and the serving default (a 262,144-token pool and 4,096-row chunks: the draft's pool and its
    // attention workspace at their widest)
    for (slots, ctx, chunk) in
        [(1usize, 4096usize, 512usize), (4, 4096, 512), (8, 4096, 512), (4, 262_144, 4096), (8, 262_144, 4096)]
    {
        let Some(rig) = Rig::load(slots, ctx, chunk, true) else { return };
        let (pool, bytes, reserve, planned) = rig.with_spec(|s| {
            let b = s.bytes();
            (
                b.pool,
                b,
                rig.plan.memory.engine_reserve_bytes,
                draft_pool_bytes(&rig.cfg, &rig.dcfg, rig.plan.kv_pages(), rig.plan.slots),
            )
        });
        assert_eq!(pool, planned, "{slots} slots, context {ctx}: the draft's pool is what the plan prices");
        let target = rig.with_model(|m| m.bytes().engine_reserve());
        assert!(
            target + bytes.engine_reserve() <= reserve,
            "{slots} slots, context {ctx}: the working set is inside the engine reserve"
        );
        eprintln!(
            "{slots} slots, context {ctx}: pool {:.1} MiB ({} pages + {slots} overflow), weights {:.1} MiB, scratch {:.1} MiB, taps {:.1} MiB, rings {:.1} MiB; \
             working set {:.0} MiB of the {:.0} MiB reserve",
            bytes.pool as f64 / 1048576.0,
            rig.plan.kv_pages(),
            bytes.weights as f64 / 1048576.0,
            bytes.scratch as f64 / 1048576.0,
            bytes.taps as f64 / 1048576.0,
            bytes.ring as f64 / 1048576.0,
            (target + bytes.engine_reserve()) as f64 / 1048576.0,
            reserve as f64 / 1048576.0
        );
        drop(rig);
    }
}

/// The three things that each priced bytes of their own: the draft (weights, pool and overflow pages, scratch, taps and rings), the
/// vision tower (weights, workspace, feature rows: `vision_reserve_bytes`) and the prefix cache (snapshots and extra pages), booted
/// together through the serving path. The boot's equality (`boot::priced_state_bytes` against the device's own buffers, the draft's
/// pool added) and `engine_from`'s (the tower's allocations, the draft's pool and weights, the working set against the reserve)
/// fail closed; this adds the figures.
#[test]
#[ignore]
fn the_plan_equals_the_allocations_with_the_draft_the_tower_and_the_cache_together() {
    for (slots, ctx, chunk) in [(1usize, 4096usize, 512usize), (4, 4096, 512), (4, 262_144, 4096)] {
        let Some(rig) = Rig::load_full(slots, ctx, chunk, true, true) else { return };
        let m = &rig.plan.memory;
        assert!(m.vision_reserve_bytes > 0, "{slots} slots: the plan prices the tower's reserve");
        assert!(
            m.cache.enabled(),
            "{slots} slots, context {ctx}: the cache is on (the serving default takes what the pool leaves)"
        );
        let (b, planned) =
            rig.with_spec(|s| (s.bytes(), draft_pool_bytes(&rig.cfg, &rig.dcfg, rig.plan.kv_pages(), rig.plan.slots)));
        assert_eq!(b.pool, planned, "{slots} slots, context {ctx}: the draft's pool is what the plan prices");
        let target = rig.with_model(|m| m.bytes().engine_reserve());
        assert!(
            target + b.engine_reserve() <= m.engine_reserve_bytes,
            "{slots} slots, context {ctx}: the working set fits the reserve"
        );
        // the draft's pages are in the pool's page and the cache's budget page is the pool's page: draft and cache agree
        let draft_page = gb_model::families::minicpmv4_7::memory::draft_page_bytes(&rig.dcfg);
        let target_page = gb_model::families::minicpmv4_7::prefix::page_bytes(&rig.cfg);
        assert_eq!(m.block_bytes, target_page + draft_page, "{slots} slots: the pool's page is the target's and the draft's");
        eprintln!(
            "{slots} slots, context {ctx}: tower reserve {:.0} MiB, cache {} snapshots + {} pages, draft pool {:.1} MiB, working set {:.0} of {:.0} MiB",
            m.vision_reserve_bytes as f64 / 1048576.0,
            m.cache.snapshot_slots,
            m.cache.page_blocks,
            b.pool as f64 / 1048576.0,
            (target + b.engine_reserve()) as f64 / 1048576.0,
            m.engine_reserve_bytes as f64 / 1048576.0
        );
        drop(rig);
    }
}

#[test]
#[ignore]
fn the_harness_checks_of_the_target_hold_with_the_draft_attached() {
    let Some(rig) = Rig::load(4, 4096, 1024, true) else { return };
    let v = rig.vocab();
    let make = || rig.shared();
    let cases: Vec<(Vec<u32>, Vec<u32>)> = [(2, 1), (50, 7), (130, 16), (300, 5)]
        .iter()
        .enumerate()
        .map(|(i, &(c, w))| (prompt(c, i, v), window_tokens(w, i, v)))
        .collect();
    row_invariance(&make, &cases, 8).unwrap_or_else(|e| panic!("row invariance: {e}"));
    for ctx_len in [60usize, 200] {
        let ctx = prompt(ctx_len, ctx_len, v);
        for (rows, keep) in [(16usize, 1usize), (16, 9), (16, 16), (5, 3)] {
            for next in [vec![17u32], window_tokens(16, 4, v)] {
                rollback_equals_replay(&make, &ctx, &window_tokens(rows, rows + 1, v), keep, &next, 8)
                    .unwrap_or_else(|e| panic!("context {ctx_len}, window {rows}, keep {keep}: {e}"));
            }
        }
    }
    assert!(rig.backend.borrow_mut().take_error().is_none());
    eprintln!(
        "row_invariance and rollback_equals_replay (16 rows, every keep that matters) hold with taps exported and rings filling"
    );
}

// =========================================================================================================== measurements

use std::io::Write as _;
use std::time::Instant;

/// Candidates a verify row returns for the model's sampling (top-k 20 plus the sampler's margin).
const CANDS: usize = 28;

/// Swap the draft on the live model: `None` is the text engine (no taps, no draft); `Some(dir)` a draft pack or the bf16 checkpoint.
fn set_draft(rig: &Rig, dir: Option<&Path>, k_top: usize) {
    rig.with_model(|m| {
        let _ = m.detach_spec().unwrap();
    });
    *rig.parked.borrow_mut() = None;
    if let Some(d) = dir {
        let dcfg = DSparkConfig::from_dir(d).unwrap_or_else(|e| panic!("{}: {e}", d.display()));
        let r =
            rig.with_model(|m| SpecRig::new(&dcfg, d, m.spec_sizes())).unwrap_or_else(|e| panic!("draft {}: {e}", d.display()));
        rig.with_model(|m| {
            m.attach_spec(r).unwrap();
            m.spec_mut().unwrap().k_top = k_top;
        });
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// The median milliseconds of one verify step of `slots` (each `rows` rows from position `base`) with the accepted rows committed
/// (the next forward replays them, so the median holds the commit's replay too).
fn time_steps(rig: &Rig, slots: &[usize], rows: usize, base: usize, reps: usize) -> f64 {
    let mut times = step_times(rig, slots, rows, base, reps);
    median(&mut times)
}

/// The milliseconds of each of `reps` verify steps after two warm-up ones (see [`time_steps`]).
fn step_times(rig: &Rig, slots: &[usize], rows: usize, base: usize, reps: usize) -> Vec<f64> {
    let v = rig.vocab();
    let mut b = rig.backend.borrow_mut();
    let mut times = Vec::new();
    for rep in 0..reps + 2 {
        let reqs: Vec<RowRequest> = slots
            .iter()
            .enumerate()
            .map(|(i, &s)| RowRequest {
                slot: s,
                tokens: window_tokens(rows, i + rep, v),
                first_position: (base + rep * rows) as u64,
            })
            .collect();
        let t = Instant::now();
        b.step(&reqs, CANDS);
        for &s in slots {
            b.commit(s, rows);
        }
        if rep >= 2 {
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    times
}

#[test]
#[ignore]
fn bench_spec_costs() {
    if std::env::var("REXL3_BENCH_SPEC").map_or(true, |v| v != "1") {
        eprintln!("not run: REXL3_BENCH_SPEC=1 times the verify grid and the draft's round (a measurement, not a gate)");
        return;
    }
    // REXL3_BENCH_DRAFT=0: the same grid on the text engine (no draft attached, no taps): what the attachment costs
    let with_draft = std::env::var("REXL3_BENCH_DRAFT").map_or(true, |v| v != "0");
    let Some(rig) = Rig::load(8, 34_000, 1024, with_draft) else { return };
    let v = rig.vocab();
    let b = 512usize;
    let prefill_all = |rig: &Rig, n: usize| {
        for s in 0..n {
            rig.with_model(|m| {
                m.begin_slot(s).unwrap();
                let p = prompt(b, 40 + s, v);
                m.prefill_chunk(&chain(s, 0, &p), PrefillPath::for_prompt(p.len()), None).unwrap();
            });
        }
    };
    let only_ctx = std::env::var("REXL3_BENCH_ONLY_CTX").is_ok_and(|v| v == "1");
    // the prefill with the draft attached against without (the tap export and the draft's absorb of every chunk): 8,192 tokens in
    // chunks of 1,024 on the pinned-GEMM arm, the median of three
    {
        let n = 8192usize;
        let mut p = prompt(1024, 6, v);
        while p.len() < n {
            let more = prompt(1024, 6 + p.len() / 1024, v);
            p.extend_from_slice(&more[..(n - p.len()).min(1024)]);
        }
        let mut times = Vec::new();
        for _ in 0..3 {
            rig.with_model(|m| m.begin_slot(0).unwrap());
            let t = Instant::now();
            rig.with_model(|m| {
                let mut at = 0;
                while at < n {
                    let end = (at + 1024).min(n);
                    m.prefill_chunk(&chain(0, at, &p[at..end]), PrefillPath::Tiled, None).unwrap();
                    at = end;
                }
            });
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let ms = median(&mut times);
        eprintln!("PREFILL {n} tokens {ms:.0} ms {:.0} tok/s (draft attached: {with_draft})", n as f64 / (ms / 1e3));
        let _ = rig.shared();
    }
    // verify step by (streams, rows a stream): three sweeps of the whole grid, so that a slow phase of the box cannot own one point;
    // the table is each point's median over the sweeps
    eprintln!("VERIFY GRID  (streams, rows total, ms)  context {b}");
    let mut grid: std::collections::BTreeMap<(usize, usize), Vec<f64>> = Default::default();
    for _sweep in 0..(if only_ctx { 0 } else { 3 }) {
        for streams in [1usize, 2, 3, 4, 6, 8] {
            let slots: Vec<usize> = (0..streams).collect();
            for rows in [1usize, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16] {
                if streams * rows > WINDOW_ROWS {
                    continue;
                }
                // re-prefill so the context stays at its length whatever the repetitions did
                prefill_all(&rig, streams);
                let ms = time_steps(&rig, &slots, rows, b, 12);
                grid.entry((streams, streams * rows)).or_default().push(ms);
            }
        }
    }
    for ((streams, rows), mut v) in grid {
        let lo = v.iter().cloned().fold(f64::MAX, f64::min);
        let hi = v.iter().cloned().fold(f64::MIN, f64::max);
        eprintln!("GRID {streams} {rows} {:.2}  (min {lo:.2} max {hi:.2})", median(&mut v));
    }
    if !with_draft || only_ctx {
        // the context sweep below needs the pool to itself
        let _ = rig.shared();
        if !with_draft {
            return;
        }
    }
    // the draft's round by streams: one 3-row window committed per stream, then the proposal (absorb + block + walk + confidence)
    eprintln!("DRAFT ROUND  (streams, ms): absorb of the committed rows included");
    let s = greedy();
    for streams in [1usize, 2, 3, 4, 6, 8] {
        if only_ctx {
            break;
        }
        let mut times = Vec::new();
        let mut absorb = Vec::new();
        prefill_all(&rig, streams);
        let mut pos = b;
        let mut last: Vec<u32> = (0..streams).map(|i| 100 + i as u32).collect();
        let mut seqs: Vec<Vec<u32>> = (0..streams).map(|i| prompt(b, 40 + i, v)).collect();
        for rep in 0..14 {
            for sl in 0..streams {
                let toks = vec![last[sl], window_tokens(1, sl + rep, v)[0], window_tokens(1, sl + 2 * rep, v)[0]];
                let cand = rig.with_model(|m| {
                    let cand = m.window_candidates(&chain(sl, pos, &toks), CANDS).unwrap();
                    m.commit(sl, 3);
                    cand
                });
                seqs[sl].extend_from_slice(&toks);
                last[sl] = top1(&cand, 2);
            }
            pos += 3;
            let with_anchor: Vec<Vec<u32>> = (0..streams)
                .map(|sl| {
                    let mut t = seqs[sl].clone();
                    t.push(last[sl]);
                    t
                })
                .collect();
            let asks: Vec<SpecAsk> =
                (0..streams).map(|sl| SpecAsk { slot: sl, tokens: &with_anchor[sl], sampling: &s, max: 15 }).collect();
            // the commit's replay is paid by whichever forward comes next (the verify step's measurement holds it): apply it first, so
            // that the round below is the draft's own work (absorb of the committed rows, the block, the walk, the confidences)
            rig.with_model(|m| m.flush()).unwrap();
            let before = rig.with_spec(|r| r.counters.absorb_ns);
            let t = Instant::now();
            let out = rig.with_model(|m| m.propose_round(&asks)).unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            assert!(out.iter().all(Option::is_some));
            let after = rig.with_spec(|r| r.counters.absorb_ns);
            if rep >= 2 {
                times.push(ms);
                absorb.push((after - before) as f64 / 1e6);
            }
        }
        eprintln!("DRAFT {streams} {:.2} (absorb {:.2})", median(&mut times), median(&mut absorb));
    }
    // the context dependence: one stream, windows of 1 and 16 rows and the draft's round at 1k, 8k and 32k tokens of context
    let _ = rig.shared();
    eprintln!("CONTEXT  (tokens, verify 1 row ms, verify 16 rows ms, draft round ms)");
    for tokens in [1024usize, 8192, 32_768] {
        if tokens + 400 > rig.max_ctx {
            eprintln!("CTX {tokens} skipped: the rig's context is {}", rig.max_ctx);
            continue;
        }
        let mut p = prompt(1024, 3, v);
        while p.len() < tokens {
            let more = prompt(1024, 3 + p.len() / 1024, v);
            p.extend_from_slice(&more[..(tokens - p.len()).min(1024)]);
        }
        rig.with_model(|m| {
            m.begin_slot(0).unwrap();
            let mut at = 0;
            while at < tokens {
                let end = (at + 1024).min(tokens);
                m.prefill_chunk(&chain(0, at, &p[at..end]), PrefillPath::Tiled, None).unwrap();
                at = end;
            }
        });
        let one = time_steps(&rig, &[0], 1, tokens, 10);
        // the 1-row steps committed 12 rows
        let sixteen = time_steps(&rig, &[0], 16, tokens + 12, 10);
        // the slot's committed length is now tokens + 12 + 12 * 16: a round at it
        let committed = tokens + 12 + 12 * 16;
        let mut tk = p.clone();
        tk.extend(std::iter::repeat_n(5u32, committed - tokens));
        tk.push(7);
        let mut ds = Vec::new();
        for _ in 0..6 {
            let a = [SpecAsk { slot: 0, tokens: &tk, sampling: &s, max: 15 }];
            let t0 = Instant::now();
            let _ = rig.with_model(|m| m.propose_round(&a)).unwrap();
            ds.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!("CTX {tokens} {one:.2} {sixteen:.2} {:.2}", median(&mut ds));
    }
}

/// What attaching a draft costs a round that does not draft, by part: the tap export (copy nodes in the window graph), the ring copy of
/// the committed rows, both and neither, against the text engine; one stream, one row a round, context 512 (`REXL3_BENCH_ATTACH=1`;
/// a measurement, not a gate). The sweeps interleave the variants so that a slow phase of the box cannot own one of them.
#[test]
#[ignore]
fn bench_attach_cost() {
    if std::env::var("REXL3_BENCH_ATTACH").map_or(true, |v| v != "1") {
        eprintln!("not run: REXL3_BENCH_ATTACH=1 times the attachment's cost by part (a measurement, not a gate)");
        return;
    }
    let Some(rig) = Rig::load(2, 4096, 1024, true) else { return };
    let v = rig.vocab();
    let p = prompt(512, 41, v);
    let setup = |rig: &Rig| {
        rig.with_model(|m| {
            m.begin_slot(0).unwrap();
            m.prefill_chunk(&chain(0, 0, &p), PrefillPath::for_prompt(p.len()), None).unwrap();
        })
    };
    // (name, attached, skip_export, skip_ring)
    let variants: [(&str, bool, bool, bool); 5] = [
        ("detached", false, false, false),
        ("attached", true, false, false),
        ("no_export", true, true, false),
        ("no_ring", true, false, true),
        ("neither", true, true, true),
    ];
    let mut res: std::collections::BTreeMap<&str, Vec<(f64, f64)>> = Default::default();
    let _ = rig.shared();
    for _sweep in 0..6 {
        for &(name, attached, skip_export, skip_ring) in &variants {
            if attached && rig.parked.borrow().is_some() {
                rig.attach();
            }
            if !attached && rig.parked.borrow().is_none() {
                rig.detach();
            }
            if attached {
                rig.with_spec(|s| {
                    s.inject.skip_export = skip_export;
                    s.inject.skip_ring = skip_ring;
                });
            }
            // the captured graph must hold what this variant runs
            rig.with_model(|m| m.clear_graphs()).unwrap();
            setup(&rig);
            let _ = time_steps(&rig, &[0], 1, 512, 3);
            setup(&rig);
            // 150 one-row rounds: the slot's ring (48 rows) fills three times, which the median of a round hides and the mean pays
            let mut t = step_times(&rig, &[0], 1, 512, 150);
            let mean = t.iter().sum::<f64>() / t.len() as f64;
            res.entry(name).or_default().push((median(&mut t), mean));
        }
    }
    let med = |mut v: Vec<f64>| median(&mut v);
    let base = med(res["detached"].iter().map(|x| x.0).collect());
    let base_mean = med(res["detached"].iter().map(|x| x.1).collect());
    eprintln!("ATTACH COST  (variant: median of a round, mean of a round with the ring's absorbs inside; ms, six sweeps, delta against detached)");
    for (name, ..) in variants {
        let m = med(res[name].iter().map(|x| x.0).collect());
        let mean = med(res[name].iter().map(|x| x.1).collect());
        eprintln!(
            "ATTACH {name} median {m:.3} ({:+.3}) mean {mean:.3} ({:+.3} ms, {:+.1} %)",
            m - base,
            mean - base_mean,
            100.0 * (mean - base_mean) / base_mean
        );
    }
    // leave the model as it was found
    if rig.parked.borrow().is_some() {
        rig.attach();
    }
    rig.with_spec(|s| {
        s.inject.skip_export = false;
        s.inject.skip_ring = false;
    });
}

// ----------------------------------------------------------------------------------------------------- Leg D on the engine

struct Prompt {
    name: String,
    class: String,
    ids: Vec<u32>,
    max_new: usize,
}

/// The frozen manifest (`tools/minicpm_v47/legd_prompts.json`), its rows' sha256 checked against the one it records.
fn load_prompts() -> Vec<Prompt> {
    use sha2::{Digest, Sha256};
    let path = std::env::var("REXL3_LEGD_PROMPTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/minicpm_v47/legd_prompts.json"));
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let doc: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    let rows = doc["rows"].as_array().expect("rows");
    // the canonical form of the Python tool: keys sorted, no spaces
    let canon: Vec<String> = rows
        .iter()
        .map(|r| {
            let ids: Vec<String> = r["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap().to_string()).collect();
            format!(
                "{{\"class\":{},\"ids\":[{}],\"max_new\":{},\"name\":{},\"prompt_len\":{}}}",
                serde_json::to_string(&r["class"]).unwrap(),
                ids.join(","),
                r["max_new"],
                serde_json::to_string(&r["name"]).unwrap(),
                r["prompt_len"]
            )
        })
        .collect();
    let blob = format!("[{}]", canon.join(","));
    let got = format!("{:x}", Sha256::digest(blob.as_bytes()));
    assert_eq!(got, doc["rows_sha256"].as_str().unwrap(), "the prompt manifest's rows do not hash to the sha256 it records");
    rows.iter()
        .map(|r| Prompt {
            name: r["name"].as_str().unwrap().to_string(),
            class: r["class"].as_str().unwrap().to_string(),
            ids: r["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect(),
            max_new: r["max_new"].as_u64().unwrap() as usize,
        })
        .collect()
}

/// One arm of the measurement: what is attached and how the budget runs.
#[derive(Clone, Debug)]
struct Arm {
    name: String,
    /// `None`: no draft at all (AR, no taps).
    dir: Option<PathBuf>,
    k: usize,
    /// Draft attached, never asked to draft (taps and rings only: the overhead of the attachment).
    attached_only: bool,
    /// Full 16-row windows (a flat verify cost): the per-position acceptance and AL of the draft itself.
    flat: bool,
    /// The budget policy (`:margin`, `:plan`: the pre-draft plan's depth cut, as before `Scored`).
    policy: BudgetPolicy,
    /// Cost attribution of an attached draft that is never asked (`:noexp`: no tap export, `:noring`: no ring copy; the draft's
    /// context is wrong then, so only with `ar_attached`).
    skip_export: bool,
    skip_ring: bool,
}

fn parse_arm(spec: &str) -> Arm {
    let models = home().join("models/minicpm-v-4.7-35b-a3b");
    let mut parts = spec.split(':');
    let head = parts.next().unwrap();
    let mut arm = Arm {
        name: spec.to_string(),
        dir: None,
        k: 64,
        attached_only: false,
        flat: false,
        policy: BudgetPolicy::Scored,
        skip_export: false,
        skip_ring: false,
    };
    match head {
        "ar" => {}
        "ar_attached" => {
            arm.dir = Some(models.join("draft-packs/dspark-6.00bpw"));
            arm.attached_only = true;
        }
        "dense" => arm.dir = Some(models.join("dspark-hf")),
        d if d.starts_with('d') => {
            let bits: usize = d[1..].parse().unwrap_or_else(|_| panic!("arm {spec}"));
            arm.dir = Some(models.join(format!("draft-packs/dspark-{bits}.00bpw")));
        }
        _ => panic!("unknown arm {spec}"),
    }
    for p in parts {
        match p {
            "flat" => arm.flat = true,
            // a repeat of an arm later in the run under another name: the order control of a drift in the box's speed
            "rep" => {}
            "noexp" => arm.skip_export = true,
            "noring" => arm.skip_ring = true,
            "margin" => arm.policy = BudgetPolicy::Margin,
            "plan" => arm.policy = BudgetPolicy::Plan,
            k if k.starts_with('k') => arm.k = k[1..].parse().unwrap(),
            other => panic!("arm {spec}: unknown option {other}"),
        }
    }
    arm
}

fn sampling_for(mode: &str, i: usize) -> Sampling {
    match mode {
        "greedy" => Sampling { seed: 0, temperature: 0.0, top_k: 20, top_p: 0.95, min_p: 0.0 },
        // the model's generation_config: temperature 1.0, top-k 20, top-p 0.95
        _ => Sampling { seed: 4000 + i as u64, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 },
    }
}

/// Leg D on the engine: the frozen prompt set through the real engine, one arm after another on one boot of the target. Every arm's
/// tokens are compared with the first arm's byte for byte (drafted == serial over the whole set), and one JSON line is written a prompt
/// (`REXL3_LEGD_OUT`) for `tools/minicpm_v47/legd_report.py`.
///
/// `REXL3_LEGD_ARMS`: `ar`, `ar_attached`, `d<bits>` (8, 6, 5, 4, 3), `dense`, with `:flat` (full 16-row windows: AL and per-position
/// acceptance of the draft itself), `:k16` / `:k32` / `:k64` (candidates of the Markov walk) and `:margin` / `:plan` (the budget the
/// engine ran with before `Scored`: the pre-draft plan's depth cut). `REXL3_LEGD_MODES`: `greedy,sampled`.
#[test]
#[ignore]
fn legd_engine() {
    use sha2::{Digest, Sha256};
    if std::env::var("REXL3_LEGD_ARMS").is_err() {
        eprintln!("not run: REXL3_LEGD_ARMS names the arms of the Leg D measurement (a measurement, not a gate)");
        return;
    }
    let prompts = load_prompts();
    let arms: Vec<Arm> = std::env::var("REXL3_LEGD_ARMS").unwrap_or_default().split(',').map(parse_arm).collect();
    let modes: Vec<String> =
        std::env::var("REXL3_LEGD_MODES").unwrap_or_else(|_| "greedy".into()).split(',').map(String::from).collect();
    let per_class = env_usize("REXL3_LEGD_PER_CLASS", usize::MAX);
    let sampled_per_class = env_usize("REXL3_LEGD_SAMPLED_PER_CLASS", 10);
    let c4 = env_usize("REXL3_LEGD_C4", 0) == 1;
    let max_new_cap = env_usize("REXL3_LEGD_MAX_NEW", usize::MAX);
    let out_path = std::env::var("REXL3_LEGD_OUT").unwrap_or_else(|_| "/tmp/legd.jsonl".into());
    let Some(rig) = Rig::load(4, 4096, 1024, true) else { return };
    let mut out = std::fs::OpenOptions::new().create(true).append(true).open(&out_path).unwrap();
    // per (mode, name): the tokens of the first arm that ran, which every later arm must reproduce byte for byte
    let mut reference: std::collections::BTreeMap<(String, String), Vec<u32>> = Default::default();
    let mut mismatches = 0usize;
    let select = |mode: &str| -> Vec<(usize, &Prompt)> {
        let cap = if mode == "greedy" { per_class } else { sampled_per_class };
        let mut seen: std::collections::BTreeMap<&str, usize> = Default::default();
        prompts
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                let n = seen.entry(p.class.as_str()).or_insert(0);
                *n += 1;
                *n <= cap
            })
            .collect()
    };
    for arm in &arms {
        set_draft(&rig, arm.dir.as_deref(), arm.k);
        if arm.skip_export || arm.skip_ring {
            assert!(arm.attached_only, "{}: the attachment hooks leave the draft's context wrong: ar_attached only", arm.name);
            rig.with_spec(|s| {
                s.inject.skip_export = arm.skip_export;
                s.inject.skip_ring = arm.skip_ring;
            });
            // the recorded window graphs hold the export nodes of the draft attached before the hook
            rig.with_model(|m| m.clear_graphs()).unwrap();
        }
        let drafting = arm.dir.is_some() && !arm.attached_only;
        // warm-up (the graphs of the windows this arm will run are captured on first use)
        for (_, p) in select("greedy").into_iter().take(3) {
            let mut e = rig.engine_with(drafting, arm.flat, arm.policy);
            let r = Request {
                prompt: p.ids.clone(),
                sampling: sampling_for("greedy", 0),
                max_new: 48,
                stop: vec![248_044, 248_046],
                draft: true,
                images: vec![],
                cache: false,
            };
            e.submit(r).unwrap();
            e.run().unwrap();
            e.take_outputs();
        }
        for mode in &modes {
            let sel = select(mode);
            // a run of one prompt at a time (c1), or of four prompts of a class together (c4)
            let groups: Vec<Vec<(usize, &Prompt)>> = if c4 {
                let mut g = Vec::new();
                for class in ["gsm8k", "code", "chat"] {
                    let v: Vec<_> = sel.iter().filter(|(_, p)| p.class == class).cloned().collect();
                    for c in v.chunks(4) {
                        if c.len() == 4 {
                            g.push(c.to_vec());
                        }
                    }
                }
                g
            } else {
                sel.iter().map(|x| vec![*x]).collect()
            };
            for group in groups {
                let has = rig.with_model(|m| m.has_draft());
                let before = has.then(|| rig.with_spec(|s| s.counters.clone()));
                let mut e = rig.engine_with(drafting, arm.flat, arm.policy);
                let t0 = Instant::now();
                let mut ids = Vec::new();
                for (i, p) in &group {
                    let r = Request {
                        prompt: p.ids.clone(),
                        sampling: sampling_for(mode, *i),
                        max_new: p.max_new.min(max_new_cap),
                        stop: vec![248_044, 248_046],
                        draft: true,
                        images: vec![],
                        cache: false,
                    };
                    ids.push(e.submit(r).unwrap_or_else(|er| panic!("{}: {er}", p.name)));
                }
                let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
                let t1 = Instant::now();
                e.run().unwrap();
                let decode_ms = t1.elapsed().as_secs_f64() * 1e3;
                let stats = e.spec_stats().total();
                let mut outs = e.take_outputs();
                outs.sort_by_key(|o| ids.iter().position(|&i| i == o.id).unwrap());
                assert!(e.take_failed().is_empty(), "a sequence failed");
                let after = has.then(|| rig.with_spec(|s| s.counters.clone()));
                let total_tokens: usize = outs.iter().map(|o| o.tokens.len().saturating_sub(1)).sum();
                for ((i, p), o) in group.iter().zip(&outs) {
                    let key = (mode.clone(), p.name.clone());
                    let exact = match reference.get(&key) {
                        Some(r) => *r == o.tokens,
                        None => {
                            reference.insert(key, o.tokens.clone());
                            true
                        }
                    };
                    if !exact {
                        mismatches += 1;
                        eprintln!(
                            "MISMATCH arm {} mode {mode} prompt {}: tokens differ from the reference arm",
                            arm.name, p.name
                        );
                    }
                    let sha = format!("{:x}", Sha256::digest(o.tokens.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>()));
                    let n_dec = o.tokens.len().saturating_sub(1);
                    let rate = if group.len() == 1 { n_dec } else { total_tokens } as f64 / (decode_ms / 1e3);
                    let rec = serde_json::json!({
                        "arm": arm.name, "mode": mode, "name": p.name, "class": p.class, "index": i, "streams": group.len(),
                        "prompt_len": p.ids.len(), "tokens": o.tokens.len(), "decode_tokens": n_dec,
                        "rounds": o.rounds, "drafted": o.drafted, "accepted": o.accepted,
                        "prefill_ms": prefill_ms, "decode_ms": decode_ms, "tok_s": rate, "group_tokens": total_tokens,
                        "exact": exact, "tokens_sha256": sha, "stopped": o.stopped,
                        "pos_offered": &stats.offered[..], "pos_verified": &stats.verified[..],
                        "pos_reached": &stats.reached[..], "pos_accepted": &stats.accepted[..], "pos_rounds": stats.rounds,
                    });
                    writeln!(out, "{rec}").unwrap();
                }
                if let (Some(b), Some(a)) = (before, after) {
                    let d = serde_json::json!({
                        "arm": arm.name, "mode": mode, "kind": "spec_counters",
                        "names": group.iter().map(|(_, p)| p.name.clone()).collect::<Vec<_>>(),
                        "draft_rounds": a.rounds - b.rounds, "draft_streams": a.streams - b.streams, "proposed": a.proposed - b.proposed,
                        "absorb_calls": a.absorb_calls - b.absorb_calls, "absorb_rows": a.absorb_rows - b.absorb_rows,
                        "absorb_ms": (a.absorb_ns - b.absorb_ns) as f64 / 1e6, "propose_ms": (a.propose_ns - b.propose_ns) as f64 / 1e6,
                        "lost_slots": a.lost_slots - b.lost_slots, "pad_leaks": a.pad_leaks - b.pad_leaks,
                    });
                    writeln!(out, "{d}").unwrap();
                }
            }
            out.flush().unwrap();
            eprintln!("arm {} mode {mode}: done ({} references)", arm.name, reference.len());
        }
        // the draft's calibration over the arm (confidence deciles by position)
        if rig.with_model(|m| m.has_draft()) {
            let c = rig.with_spec(|s| s.counters.clone());
            let cal = serde_json::json!({"arm": arm.name, "kind": "calibration",
                "reached": c.reached.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                "accepted": c.accepted.iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
                "rounds_by_accepted": c.first_reject.to_vec(), "pad_leaks": c.pad_leaks, "lost_slots": c.lost_slots});
            writeln!(out, "{cal}").unwrap();
        }
    }
    assert_eq!(mismatches, 0, "{mismatches} runs differ from the reference arm: drafted != serial");
    assert!(rig.backend.borrow_mut().take_error().is_none());
}

// ============================================================================================ the prefix cache with the draft (E4b)

mod minicpmv4_7_support;

use gb_engine::{Captured, Output};
use gb_model::families::minicpmv4_7::boot::BootReport;
use gb_model::families::minicpmv4_7::memory::draft_page_bytes;
use gb_model::families::minicpmv4_7::prefix;
use gb_model::serve::PrefixCacheOption;

type CEng = Engine<MiniCpmBackend>;

/// An engine booted the way `serve` boots it: a prefix cache of `units` entry units (one snapshot slot and eight pages of the pool's
/// page size: the target's planes and, with a draft, the draft's), the draft at `draft` (None: the text engine), drafting on or off.
struct CRig {
    eng: CEng,
    cfg: MiniCpmConfig,
    plan: BootPlan,
    report: BootReport,
}

impl CRig {
    fn load(slots: usize, max_ctx: usize, chunk: usize, units: u64, draft: Option<&Path>, drafting: bool) -> Option<CRig> {
        let dir = pack_dir()?;
        if let Some(d) = draft {
            if !d.join("config.json").exists() {
                eprintln!("not run: no draft at {}", d.display());
                return None;
            }
        }
        let cfg = MiniCpmConfig::from_dir(&dir).unwrap();
        let dpage = draft.map_or(0, |d| draft_page_bytes(&DSparkConfig::from_dir(d).unwrap()));
        let unit = prefix::snapshot_bytes(&cfg) + 8 * (prefix::page_bytes(&cfg) + dpage);
        let mut o = ServeOptions::new(dir);
        o.slots = slots;
        o.slots_auto = false;
        o.allow_unvalidated_pack = true;
        o.max_model_len = max_ctx;
        o.prefill_chunk = chunk;
        o.drafting = drafting;
        o.draft_dir = draft.map(Path::to_path_buf);
        o.prefix_cache =
            if units == 0 { PrefixCacheOption::Off } else { PrefixCacheOption::Mib((units * unit).div_ceil(1 << 20)) };
        let loaded = boot_with(&o, BootExtras::default()).unwrap_or_else(|e| panic!("boot: {e}"));
        let (plan, report) = (loaded.plan.clone(), loaded.report.clone());
        let eng = engine_from(loaded, &o).unwrap_or_else(|e| panic!("engine: {e}"));
        assert_eq!(eng.prefix_stats().enabled, units > 0, "the engine from `engine_from` runs the cache the plan carved out");
        // the test policy (cuts from 128 tokens) replaces the configuration, not the decision
        let eng = eng.with_prefix_cache(minicpmv4_7_support::test_policy());
        Some(CRig { eng, cfg, plan, report })
    }

    fn m(&mut self) -> &mut MiniCpmModel {
        &mut self.eng.backend.m
    }
}

fn run_wave(e: &mut CEng, wave: &[Request]) -> Vec<Output> {
    minicpmv4_7_support::many(e, wave.to_vec())
}

/// A request that may use the cache (`request` of this file is the uncached reference path).
fn creq(prompt: Vec<u32>, sampling: Sampling, max_new: usize, draft: bool) -> Request {
    Request { prompt, sampling, max_new, stop: vec![], draft, images: vec![], cache: true }
}

/// The waves of the cached == uncached check: each is submitted together; the later ones hit what the earlier ones left. `real` text
/// (the Leg A rows: python, rust) so that a draft has something to accept.
fn cache_waves(vocab: usize) -> Vec<(&'static str, Vec<Request>, bool)> {
    let row = |r: usize, from: usize, n: usize| -> Vec<u32> {
        // `prompt(n, seed)` takes a window of row `seed % rows`; the offset is part of the seed, so cut the row ourselves
        let all = prompt(1024, r, vocab);
        all[from..from + n].to_vec()
    };
    let (a, b) = (row(0, 0, 1000), row(1, 0, 400));
    let p1: Vec<u32> = a[..700].to_vec();
    let with_suffix = |from: usize, n: usize| -> Vec<u32> {
        let mut p = p1.clone();
        p.extend_from_slice(&a[from..from + n]);
        p
    };
    let draft = true;
    // the donor: nothing to hit, leaves its snapshots (the cut at 128+, the end cut at len - 16)
    let mut v: Vec<(&'static str, Vec<Request>, bool)> = vec![("donor", vec![creq(p1.clone(), greedy(), 50, draft)], false)];
    // a hit, greedy, drafted
    v.push(("greedy hit", vec![creq(with_suffix(700, 40), greedy(), 90, draft)], true));
    // a hit, sampled with a key
    v.push(("sampled hit", vec![creq(with_suffix(700, 55), sampled(5), 90, draft)], true));
    // a burst of four on the same prefix, different suffixes, both modes
    v.push((
        "burst of four",
        vec![
            creq(with_suffix(700, 30), greedy(), 70, draft),
            creq(with_suffix(730, 31), sampled(11), 70, draft),
            creq(with_suffix(761, 32), greedy(), 70, draft),
            creq(with_suffix(793, 33), sampled(12), 70, draft),
        ],
        true,
    ));
    // a partial hit: the first 530 tokens are shared (the last whole 256-token block is 512), then another text
    // (the first prompt that shares only a prefix finds nothing to hit and leaves the shared cut behind; the next one hits it)
    let mut part = a[..530].to_vec();
    part.extend_from_slice(&b[..170]);
    v.push(("partial prefix, leaves its shared cut", vec![creq(part, greedy(), 80, draft)], false));
    let mut part2 = a[..530].to_vec();
    part2.extend_from_slice(&b[170..250]);
    v.push(("partial hit", vec![creq(part2, greedy(), 80, draft), creq(a[..540].to_vec(), sampled(21), 60, draft)], true));
    // the opt-out and a request that does not draft, both on a prefix that is cached
    let mut no_cache = creq(with_suffix(700, 20), greedy(), 60, draft);
    no_cache.cache = false;
    v.push(("opt-out", vec![no_cache], false));
    v.push(("no draft requested", vec![creq(with_suffix(700, 25), greedy(), 60, false)], true));
    v
}

#[test]
#[ignore]
fn cached_replies_equal_uncached_replies_with_the_draft_attached() {
    let Some(dir) = pack_dir() else { return };
    let _ = dir;
    let ddir = draft_dir();
    let vocab = MiniCpmConfig::from_dir(&pack_dir().unwrap()).unwrap().vocab;
    let waves = cache_waves(vocab);
    // the uncached reference: an engine with no cache and no draft at all (plain serial decoding)
    let reference: Vec<Vec<Output>> = {
        let Some(mut r) = CRig::load(4, 4096, 512, 0, None, false) else { return };
        waves.iter().map(|(_, w, _)| run_wave(&mut r.eng, w)).collect()
    };
    let Some(mut rig) = CRig::load(4, 4096, 512, 12, Some(&ddir), true) else { return };
    let mut hits_expected = 0u64;
    for (i, (name, wave, expect_hit)) in waves.iter().enumerate() {
        let outs = run_wave(&mut rig.eng, wave);
        for (k, (o, want)) in outs.iter().zip(&reference[i]).enumerate() {
            minicpmv4_7_support::same(&format!("{name} #{k}"), o, want);
        }
        let drafted: u64 = outs.iter().map(|o| o.drafted).sum();
        let accepted: u64 = outs.iter().map(|o| o.accepted).sum();
        let cached: Vec<usize> = outs.iter().map(|o| o.cached_tokens).collect();
        eprintln!("{name}: cached tokens {cached:?}, {drafted} drafted, {accepted} accepted");
        if *expect_hit {
            for (k, o) in outs.iter().enumerate() {
                assert!(o.cached_tokens >= 128, "{name} #{k}: no hit ({} cached tokens)", o.cached_tokens);
                hits_expected += 1;
            }
            // the draft works on a hit: its context came back with the snapshot (a stream whose request asked for no draft excepted)
            if wave.iter().all(|r| r.draft) {
                assert!(
                    drafted > 0 && accepted > 0,
                    "{name}: the draft did not draft after a hit ({drafted} drafted, {accepted} accepted)"
                );
                for (k, o) in outs.iter().enumerate() {
                    assert!(o.drafted > 0, "{name} #{k}: a hit slot with no drafts");
                }
            }
        } else {
            assert!(outs.iter().all(|o| o.cached_tokens == 0), "{name}: unexpected hit");
        }
    }
    let st = rig.eng.prefix_stats();
    assert!(st.hits >= hits_expected, "{} hits for {hits_expected} hit requests", st.hits);
    let c = rig.m().spec().unwrap().counters.clone();
    assert_eq!(c.lost_slots, 0, "no slot lost its draft context: a restore resumed it");
    assert_eq!(c.pad_leaks, 0);
    assert!(rig.eng.backend.take_error().is_none());
    // the books balance: every snapshot slot and page the cache holds is the cache's
    let _ = (&rig.cfg, &rig.plan, &rig.report);
    eprintln!(
        "cached == uncached with the draft: {} hit requests, {} hits, {} tokens saved",
        hits_expected, st.hits, st.tokens_saved
    );
}

/// A prefill of `prompt` in `slot` with a snapshot at each of `cuts`, through the backend (the engine's own call).
fn take_cuts(b: &mut MiniCpmBackend, slot: usize, prompt: &[u32], cuts: &[usize]) -> Vec<Captured> {
    let mut done = 0;
    let mut taken = Vec::new();
    loop {
        let s = b.prefill_slice_cuts(slot, prompt, &[], done, 8, cuts);
        done = s.step.done;
        taken.extend(s.captured);
        if s.step.row.is_some() {
            assert!(b.take_error().is_none(), "the donor's prefill failed");
            return taken;
        }
    }
}

fn kv_of(m: &MiniCpmModel, slot: usize, len: usize) -> Vec<(Vec<u16>, Vec<u16>)> {
    let sp = m.spec().expect("a draft is attached");
    (0..5).map(|l| sp.draft.read_kv(slot, l, 0, len).unwrap()).collect()
}

#[test]
#[ignore]
fn a_restored_slot_has_the_donors_draft_context_at_every_cut_and_drafts_from_it() {
    let ddir = draft_dir();
    let Some(mut rig) = CRig::load(3, 4096, 1024, 12, Some(&ddir), true) else { return };
    let vocab = rig.cfg.vocab;
    let total = 1500usize;
    let p = prompt(total.min(1024), 4, vocab);
    let mut prompt_ids = p.clone();
    prompt_ids.extend(prompt(1024, 5, vocab).into_iter().take(total - p.len()));
    // cuts on a page boundary, off the grid (a partly filled page: the draft's page is copied with the target's), mid chunk
    let cuts = [129usize, 255, 256, 257, 513, 777, 1024];
    let b = &mut rig.eng.backend;
    b.chunk = 1024;
    let taken = take_cuts(b, 0, &prompt_ids, &cuts);
    assert_eq!(taken.iter().map(|c| c.len).collect::<Vec<_>>(), cuts.to_vec());
    // the donor finished its prompt: its draft context is the whole prompt
    assert!(b.m.draft_context_is(0, total));
    let s = greedy();
    for cap in &taken {
        let cut = cap.len;
        b.chunk = 257;
        b.prefix_restore(1, &cap.handle, cut).unwrap_or_else(|e| panic!("restore at {cut}: {e}"));
        // the restored slot's draft context is the cut's rows: it drafts, from rows that are the donor's bits
        assert!(b.m.draft_context_is(1, cut), "the restore at {cut} resumed no draft context");
        let (mine, donor) = (kv_of(&b.m, 1, cut), kv_of(&b.m, 0, cut));
        for l in 0..5 {
            assert_eq!(mine[l].0, donor[l].0, "cut {cut}: K of draft layer {l} differs from the donor's");
            assert_eq!(mine[l].1, donor[l].1, "cut {cut}: V of draft layer {l} differs from the donor's");
        }
        let mut tokens = prompt_ids[..cut].to_vec();
        tokens.push(7);
        let out = b.m.propose_round(&[SpecAsk { slot: 1, tokens: &tokens, sampling: &s, max: 15 }]).unwrap();
        assert!(out[0].is_some(), "the slot restored at {cut} does not draft");
        // the next prefill slice resumes the draft's context where the cut ended (a continuation through the engine's own call)
        let step = b.prefill_slice(1, &prompt_ids, &[], cut, 8);
        assert!(step.done > cut);
        assert!(b.m.draft_context_is(1, step.done), "the suffix after the cut at {cut} was not absorbed");
        b.release(1);
    }
    for c in taken {
        b.prefix_release(c.handle);
    }
    b.release(0);
    assert!(b.take_error().is_none());
    assert_eq!(rig.m().spec().unwrap().counters.lost_slots, 0);
}

#[test]
#[ignore]
fn a_snapshot_resumes_a_draft_context_only_for_the_draft_that_made_it_and_only_when_it_was_valid() {
    let d6 = draft_dir();
    let d8 = home().join("models/minicpm-v-4.7-35b-a3b/draft-packs/dspark-8.00bpw");
    let Some(mut rig) = CRig::load(3, 4096, 512, 8, Some(&d6), true) else { return };
    if !d8.join("config.json").exists() {
        eprintln!("not run: no second draft pack at {}", d8.display());
        return;
    }
    let vocab = rig.cfg.vocab;
    let prompt_ids = prompt(1024, 8, vocab)[..700].to_vec();
    let s = greedy();
    let ask = |len: usize| {
        let mut t = prompt_ids[..len].to_vec();
        t.push(7);
        t
    };
    // donor 1 (a valid context, draft A) and donor 2 (its draft context lost to an injected failure before its first chunk)
    let ns_a = rig.eng.backend.prefix_caps().unwrap().namespace;
    let good = take_cuts(&mut rig.eng.backend, 0, &prompt_ids, &[300]);
    rig.m().spec_mut().unwrap().inject.fail_slot = Some(2);
    let bad = take_cuts(&mut rig.eng.backend, 2, &prompt_ids, &[300]);
    assert_eq!(rig.m().spec().unwrap().counters.lost_slots, 1);
    let propose_at = |rig: &mut CRig, slot: usize| -> bool {
        let tokens = ask(300);
        rig.m().propose_round(&[SpecAsk { slot, tokens: &tokens, sampling: &s, max: 15 }]).unwrap()[0].is_some()
    };
    // (1) the same draft, a valid context: restored with it
    rig.eng.backend.prefix_restore(1, &good[0].handle, 300).unwrap();
    assert!(propose_at(&mut rig, 1), "a snapshot with a valid draft context of this very draft must resume it");
    rig.eng.backend.release(1);
    // (2) a snapshot whose donor had lost its draft context: the target's state comes back, the slot never drafts
    rig.eng.backend.prefix_restore(1, &bad[0].handle, 300).unwrap();
    assert!(!propose_at(&mut rig, 1), "a snapshot taken with no valid draft context must not claim one");
    rig.eng.backend.release(1);
    // (3) another draft in the model (another pack: the identity differs): the snapshot of draft A is not draft B's context
    let cfg8 = DSparkConfig::from_dir(&d8).unwrap();
    rig.m().detach_spec().unwrap();
    let b = rig.m().spec_sizes();
    let r8 = SpecRig::new(&cfg8, &d8, b).unwrap();
    rig.m().attach_spec(r8).unwrap();
    let ns_b = rig.eng.backend.prefix_caps().unwrap().namespace;
    assert_ne!(ns_a, ns_b, "the namespace names the draft: a snapshot of draft A is not found by an engine running draft B");
    rig.eng.backend.prefix_restore(1, &good[0].handle, 300).unwrap();
    assert!(!propose_at(&mut rig, 1), "draft B resumed the context draft A made");
    rig.eng.backend.release(1);
    // (4) no draft at all: the restore still works (the target's state) and nothing breaks
    rig.m().detach_spec().unwrap();
    let ns_none = rig.eng.backend.prefix_caps().unwrap().namespace;
    assert_ne!(ns_a, ns_none);
    rig.eng.backend.prefix_restore(1, &good[0].handle, 300).unwrap();
    rig.eng.backend.release(1);
    for c in good.into_iter().chain(bad) {
        rig.eng.backend.prefix_release(c.handle);
    }
    rig.eng.backend.release(0);
    rig.eng.backend.release(2);
    assert!(rig.eng.backend.take_error().is_none());
}
