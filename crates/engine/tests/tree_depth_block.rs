//! The 16-row draft block (`REXL3_DRAFT_BLOCK`, `docs/TREE_DRAFTING.md` section 13): the depth-adaptive ask of a tree round,
//! the exactness of a 15-level lattice (greedy, sampled, multi-stream, burst prefill, a prefix-cache hit), the depth cap
//! `limit_depth_with` at 15, tree rows inside `--window-rows`, the plan's shape against the allocations, and the by-depth
//! counters to 15. The knob itself is `Draft`'s (`crates/model/src/draft.rs`): these tests cover the ENGINE side (the ask,
//! the window, the counters), which is what the CPU mock can see; the draft-kernel side (levels 1..7 of a 16-block equal
//! the 8-block's, bit for bit) is the GPU test `crates/model/tests/draft_batch.rs`
//! `a_16_row_draft_block_gives_levels_1_to_7_the_same_bits_as_the_trained_8_row_block` and the GPU mutation entries of
//! `evals/tree_gpu_mutations.sh`.

use gb_engine::{Engine, Request, TreeConfig};
use gb_sampling::Sampling;
use gb_spec::stats::MAX_POSITIONS;
use std::cell::RefCell;
use std::rc::Rc;

use gb_engine::mock::{Fault, MockBackend};
use gb_engine::{Drafter, NoDrafts};
use gb_spec::tree::Lattice;

const VOCAB: usize = 389;
const SLOTS: usize = 16;
const K: usize = 16;

fn admission() -> gb_sched::Admission {
    let mm = gb_sched::ModelMemory {
        weights_bytes: 0,
        kv_bytes_per_token: 65536,
        state_bytes_per_slot: 150 << 20,
        activation_reserve: 0,
    };
    gb_sched::Admission::new(64 << 30, 0, mm, SLOTS).unwrap()
}

fn hash(a: u64, b: u64) -> u64 {
    let mut x = a ^ (b.wrapping_mul(0x9E3779B97F4A7C15));
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// The mock's own logits (`gb_engine::mock::mock_logits`): the target's argmax the engine's rows will sample.
fn mock_logits(ctx: &[u32], vocab: usize) -> Vec<f32> {
    gb_engine::mock::mock_logits(ctx, vocab)
}

fn chain_ids(ids: &[u32], d: usize) -> &[u32] {
    &ids[d * K..(d + 1) * K]
}

/// A lattice of `rows` levels built exactly as `tree_rounds.rs`'s (candidates from the mock's own logits under the chain
/// context and under each previous candidate), so the best-first search over it ranks the mock's own greedy text first.
fn lattice(tokens: &[u32], rows: usize, noise: f32) -> (Vec<u32>, Lattice) {
    let mut ids = vec![0u32; rows * K];
    let mut unary = vec![0f32; rows * K];
    let mut edge = vec![0f32; rows * K * K];
    let mut ctx = tokens.to_vec();
    let mut chain = Vec::new();
    let mut parent = 0usize;
    for d in 0..rows {
        let parents: Vec<Vec<f32>> = if d == 0 {
            vec![mock_logits(&ctx, VOCAB)]
        } else {
            let mut base = ctx.clone();
            base.pop();
            (0..K)
                .map(|p| {
                    let mut c = base.clone();
                    c.push(chain_ids(&ids, d - 1)[p]);
                    mock_logits(&c, VOCAB)
                })
                .collect()
        };
        let own = mock_logits(&ctx, VOCAB);
        let mut score: Vec<(f32, u32)> =
            (0..VOCAB).map(|t| (parents.iter().map(|l| l[t]).fold(f32::NEG_INFINITY, f32::max), t as u32)).collect();
        score.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut cands: Vec<u32> = score.iter().take(K).map(|s| s.1).collect();
        let noisy = |t: u32| {
            own[t as usize] + ((hash(d as u64 * 7919 + ctx.len() as u64, t as u64) >> 40) as f32 / 16777216.0 - 0.5) * noise
        };
        cands.sort_by(|&a, &b| noisy(b).total_cmp(&noisy(a)).then(a.cmp(&b)));
        for (c, &t) in cands.iter().enumerate() {
            ids[d * K + c] = t;
            unary[d * K + c] = noisy(t);
            for p in 0..K {
                edge[(d * K + p) * K + c] = if d == 0 { 0.0 } else { parents[p][t as usize] };
            }
        }
        let s = |c: usize| unary[d * K + c] + edge[(d * K + parent) * K + c];
        let best = (1..K).fold(0, |b, c| if s(c) > s(b) { c } else { b });
        chain.push(ids[d * K + best]);
        ctx.push(ids[d * K + best]);
        parent = best;
    }
    (chain, Lattice { rows, k: K, ids, unary, edge, restricted_head: false })
}

#[derive(Default)]
struct Seen {
    windows: usize,
    branched: usize,
    max_rows_in_a_step: usize,
    max_depth: usize,
    /// The deepest ASK the engine made of the drafter (`ProposeRequest::max`).
    max_ask: usize,
    /// The deepest ASK made while more than `DEEP_BLOCK_STREAMS` streams were decoding (the group limit's regime).
    max_ask_wide: usize,
}

/// The tree-rounds mock backend wrapped to serve 15-deep lattices and record the asks.
struct Lat15 {
    inner: MockBackend,
    noise: f32,
    seen: Rc<RefCell<Seen>>,
}

impl Lat15 {
    fn new(inner: MockBackend) -> (Self, Rc<RefCell<Seen>>) {
        let seen = Rc::new(RefCell::new(Seen::default()));
        (Self { inner, noise: 3.0, seen: seen.clone() }, seen)
    }
    /// A draft without error (the mock's own greedy chain first): the stream keeps levels, so the adaptive block grows.
    fn quiet(inner: MockBackend) -> (Self, Rc<RefCell<Seen>>) {
        let seen = Rc::new(RefCell::new(Seen::default()));
        (Self { inner, noise: 0.0, seen: seen.clone() }, seen)
    }
}

impl gb_engine::Backend for Lat15 {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn supports_trees(&self) -> bool {
        self.inner.supports_trees()
    }
    fn step_tree(&mut self, reqs: &[gb_engine::TreeRequest], candidates: usize) -> Vec<Vec<gb_engine::RowOutput>> {
        let mut seen = self.seen.borrow_mut();
        let mut rows = 0;
        for r in reqs {
            seen.windows += 1;
            seen.branched += usize::from(!r.is_chain());
            seen.max_depth = seen.max_depth.max(r.depths().into_iter().max().unwrap());
            rows += r.tokens.len();
        }
        seen.max_rows_in_a_step = seen.max_rows_in_a_step.max(rows);
        drop(seen);
        self.inner.step_tree(reqs, candidates)
    }
    fn commit_path(&mut self, slot: usize, path: &[usize]) {
        self.inner.commit_path(slot, path)
    }
    fn take_error(&mut self) -> Option<gb_engine::BackendError> {
        self.inner.take_error()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> gb_engine::RowOutput {
        self.inner.prefill(slot, start, tokens, candidates)
    }
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[gb_engine::ImageSpan],
        done: usize,
        candidates: usize,
    ) -> gb_engine::PrefillStep {
        self.inner.prefill_slice(slot, prompt, images, done, candidates)
    }
    fn step(&mut self, reqs: &[gb_engine::RowRequest], candidates: usize) -> Vec<Vec<gb_engine::RowOutput>> {
        self.inner.step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.inner.commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.inner.release(slot)
    }
    fn propose_batch_scored(&mut self, reqs: &[gb_engine::ProposeRequest]) -> Vec<Option<gb_engine::Proposal>> {
        let mut seen = self.seen.borrow_mut();
        let wide = reqs.len() > gb_sched::DEEP_BLOCK_STREAMS;
        reqs.iter()
            .map(|r| {
                if wide {
                    seen.max_ask_wide = seen.max_ask_wide.max(r.max);
                }
                seen.max_ask = seen.max_ask.max(r.max);
                let (chain, l) = lattice(r.tokens, r.max.max(1), self.noise);
                Some(gb_engine::Proposal { tokens: chain, lattice: Some(l), ..Default::default() })
            })
            .collect()
    }
}

fn no_drafts() -> Box<dyn Fn() -> Box<dyn Drafter>> {
    Box::new(|| Box::new(NoDrafts) as Box<dyn Drafter>)
}

fn config(drafting: bool, max_rows: usize) -> gb_engine::EngineConfig {
    gb_engine::EngineConfig {
        max_rows,
        max_drafts: 7,
        cost: gb_sched::CostModel::from_table(vec![(1, 70.0), (256, 71.0)]),
        drafting,
    }
}

/// The engine over the 15-level mock: the depth-adaptive block grows the ask as the stream keeps levels, up to the kernels' 16
/// (`gb_sched::adaptive_block`); the mock's drafter serves whatever ask it gets.
fn deep_engine(drafting: bool, max_rows: usize) -> (Engine<Lat15>, Rc<RefCell<Seen>>) {
    let (b, seen) = Lat15::new(MockBackend::new(VOCAB, SLOTS, Fault::None).with_trees());
    let e = Engine::new(b, admission(), config(drafting, max_rows), no_drafts());
    (if drafting { e.with_trees(TreeConfig::default()) } else { e }, seen)
}

/// The same with a noiseless draft: every level the mock offers is the target's own pick, so a round keeps its whole depth and
/// the adaptive block grows the ask (the growth tests need the acceptance; the exactness tests above do not).
fn quiet_deep_engine(drafting: bool, max_rows: usize) -> (Engine<Lat15>, Rc<RefCell<Seen>>) {
    let (b, seen) = Lat15::quiet(MockBackend::new(VOCAB, SLOTS, Fault::None).with_trees());
    let e = Engine::new(b, admission(), config(drafting, max_rows), no_drafts());
    (if drafting { e.with_trees(TreeConfig::default()) } else { e }, seen)
}

fn requests(n: usize) -> Vec<Request> {
    let temps = [0.0, 0.6, 1.0];
    (0..n)
        .map(|i| {
            let prompt: Vec<u32> = (0..(5 + i * 3)).map(|j| ((i * 131 + j * 17 + 11) % VOCAB) as u32).collect();
            Request {
                sampling: Sampling { seed: 4321 + i as u64, temperature: temps[i % 3], top_k: 20, top_p: 0.95, min_p: 0.0 },
                prompt,
                max_new: 96 + (i % 5) * 17,
                stop: vec![],
                draft: true,
                images: vec![],
                cache: true,
            }
        })
        .collect()
}

/// A 15-LEVEL lattice under the depth-adaptive ask: every reply equals its serial solo reply, greedy and sampled, at one and
/// several streams. The adaptive block grows the ask only when levels are KEPT, so this also covers the asks between 7 and 15.
#[test]
fn a_15_level_lattice_keeps_every_reply_serial_greedy_and_sampled() {
    let make = |drafting: bool| deep_engine(drafting, 64).0;
    let r = gb_engine::harness::drafted_equals_serial(&make, &requests(9)).unwrap();
    assert!(r.accepted > 0 && r.drafted > r.accepted, "{r:?}");
}

/// The depth-adaptive block reaches the full 16 when the stream keeps its levels, and never asks deeper than the window.
#[test]
fn the_adaptive_block_grows_the_ask_to_fifteen_on_a_stream_that_keeps_levels() {
    let (mut e, seen) = quiet_deep_engine(true, 64);
    for q in requests(1) {
        e.submit(q).unwrap();
    }
    e.run().unwrap();
    let seen = seen.borrow();
    assert!(seen.windows > 0, "no tree round ran");
    assert!(seen.max_ask > 7, "the ask never grew past the trained 7 ({}): the adaptive block is inert", seen.max_ask);
    assert!(seen.max_ask <= 15, "the ask passed the kernels' 15 levels: {}", seen.max_ask);
}

/// Past `DEEP_BLOCK_STREAMS` streams the block drops back to the trained 8: the ask never grows past 7 levels.
#[test]
fn the_adaptive_block_stays_at_the_trained_eight_past_four_streams() {
    for streams in [5usize, 8] {
        let (mut e, seen) = quiet_deep_engine(true, 64);
        for q in requests(streams) {
            e.submit(q).unwrap();
        }
        e.run().unwrap();
        let seen = seen.borrow();
        assert!(seen.windows > 0);
        assert!(
            seen.max_ask_wide <= 7,
            "{streams} streams: the ask reached {} levels while the round drafted more than four streams; the block is then the trained 8",
            seen.max_ask_wide
        );
    }
}

/// A tree window never holds a row deeper than `--window-rows` (the depth cap `limit_depth_with` at 15 inside a narrow window).
#[test]
fn a_15_level_tree_stays_inside_the_window_rows() {
    for (streams, max_rows) in [(1usize, 12usize), (1, 16), (3, 24)] {
        // the QUIET draft (its chain is the mock's own greedy text, so levels are kept): the ask grows and the
        // window is what must stop the tree, not the draft's own 7 levels
        let (mut e, seen) = quiet_deep_engine(true, max_rows);
        for q in requests(streams) {
            e.submit(q).unwrap();
        }
        e.run().unwrap();
        let seen = seen.borrow();
        assert!(seen.windows > 0);
        assert!(
            seen.max_rows_in_a_step <= max_rows.max(streams),
            "{streams} streams, window {max_rows}: a step ran {} rows",
            seen.max_rows_in_a_step
        );
        // a chain-shaped tree (one row a level) cannot be deeper than the window
        assert!(seen.max_depth < max_rows, "{streams} streams, window {max_rows}: depth {} needs a row a level", seen.max_depth);
    }
}

/// The by-depth counters reach 15 (`gb_spec::stats`), and the depth-adaptive rounds feed them: depth 8..=14 verified is
/// nonzero once the ask grows past 7. `MAX_POSITIONS` is 16, so depth 15 lands in the last slot without wrap.
#[test]
fn the_by_depth_counters_reach_fifteen() {
    let (mut e, seen) = quiet_deep_engine(true, 64);
    for q in requests(4) {
        e.submit(q).unwrap();
    }
    e.run().unwrap();
    let st = e.spec_stats();
    let deep_verified: u64 = st.trees.iter().map(|t| t.depth_verified[7..].iter().sum::<u64>()).sum();
    assert!(seen.borrow().max_ask > 7, "the ask never grew ({}): the deep counters are untested", seen.borrow().max_ask);
    assert!(deep_verified > 0, "no node deeper than level 7 was verified although the ask grew to {}", seen.borrow().max_ask);
    assert!(st.trees.iter().all(|t| t.depth_verified.len() == MAX_POSITIONS));
}

/// `limit_depth_with` at 15 keeps the tree, its probabilities and its copied flags in step (the depth cap the engine applies
/// when the window allows the full block).
#[test]
fn limit_depth_with_at_fifteen_keeps_the_three_vectors_in_step() {
    use gb_spec::tree::{limit_depth_with, DraftTree};
    let mut t = DraftTree::chain(&(100..115).collect::<Vec<u32>>());
    t.depth = vec![15; t.len()]; // a degenerate all-at-depth-15 tree: the cap keeps everything or nothing
    let probs: Vec<f64> = (0..t.len()).map(|i| 0.9 - i as f64 * 0.01).collect();
    let copied = vec![false; t.len()];
    let (t2, p2, _c2) = limit_depth_with(t.clone(), probs.clone(), copied.clone(), 15);
    assert_eq!(t2.len(), t.len());
    assert_eq!(p2, probs);
    let (t3, p3, c3) = limit_depth_with(t, probs.clone(), copied, 14);
    assert!(t3.is_empty() && p3.is_empty() && c3.is_empty(), "cap 14 drops every node of a depth-15 tree");
    assert_eq!(p3.len(), c3.len());
}

/// `gb_sched::adaptive_block` is the re-expressed rule: next block = deepest kept + 2, or + 5 when the last level was kept;
/// floor the trained 8, cap 16, and back at 8 past four streams (the group limit of a 16-row block).
#[test]
fn adaptive_block_is_kept_plus_two_or_five_with_floors_caps_and_the_group_limit() {
    use gb_sched::{adaptive_block, MAX_DRAFT_BLOCK, MIN_DRAFT_BLOCK};
    // a stream that kept nothing (or has no history) starts at the trained block
    assert_eq!(adaptive_block(0, false, 1), MIN_DRAFT_BLOCK);
    assert_eq!(adaptive_block(3, false, 1), MIN_DRAFT_BLOCK);
    // kept + 2
    assert_eq!(adaptive_block(6, false, 1), 8);
    assert_eq!(adaptive_block(9, false, 2), 11);
    assert_eq!(adaptive_block(13, false, 3), 15);
    // kept + 5 when the block's last level was itself kept
    assert_eq!(adaptive_block(7, true, 1), 12);
    assert_eq!(adaptive_block(10, true, 4), 15);
    // the kernels' cap
    assert_eq!(adaptive_block(14, false, 1), MAX_DRAFT_BLOCK);
    assert_eq!(adaptive_block(15, true, 1), MAX_DRAFT_BLOCK);
    // past four streams the block is the trained 8 whatever the stream kept
    for kept in [0usize, 7, 12, 15] {
        assert_eq!(adaptive_block(kept, true, 5), MIN_DRAFT_BLOCK, "kept {kept} at 5 streams");
        assert_eq!(adaptive_block(kept, false, 8), MIN_DRAFT_BLOCK, "kept {kept} at 8 streams");
    }
}

/// The plan's shape is what the allocations are: the window's rows are the streams' roots plus the nodes the allocation kept,
/// the streams that draft are those with an ask, and the head rows are the asks' sums. (`plan == allocations` on the mock:
/// the engine's own `Seen::max_rows_in_a_step` above already bounds the window; this pins the plan arithmetic.)
#[test]
fn the_plan_matches_the_allocations_of_a_deep_round() {
    use gb_sched::{plan_trees, DraftCharge, RoundCost, TreeDemand};
    // a flat tree table (rows nearly free) with a spent draft: the allocation buys every node offered
    let cost = RoundCost {
        verify: gb_sched::StepCost::rows_only(gb_sched::CostModel::from_table(vec![(1usize, 70.0f64), (256, 71.0)])),
        tree_verify: Some(gb_sched::StepCost::rows_only(gb_sched::CostModel::from_table(vec![(1usize, 70.0f64), (256, 71.0)]))),
        draft: None,
        kv_ms_per_token: 0.0,
        kv_pass_ms_per_token: 0.0,
        fixed_ms: 0.0,
    };
    let probs: Vec<f64> = (0..15).map(|i| 0.8f64.mul_add(-i as f64, 0.0).max(0.01)).collect();
    let demands: Vec<TreeDemand> = (0..3)
        .map(|s| TreeDemand { probs: probs.iter().map(|p| p * (1.0 - 0.1 * s as f64)).collect(), head_rows: 15, kv_tokens: 100 })
        .collect();
    let plan = plan_trees(&cost, &demands, 64, DraftCharge::Spent(9.5));
    assert_eq!(plan.alloc.len(), demands.len());
    assert_eq!(plan.rows, demands.len() + plan.alloc.iter().sum::<usize>(), "rows = roots + kept nodes");
    assert_eq!(plan.drafting, demands.len());
    assert!(plan.alloc.iter().all(|&a| a <= 15), "a stream never keeps more nodes than its ask");
    // expected tokens = one a stream (the sample past the root) plus the kept nodes' probabilities
    let want: f64 =
        demands.len() as f64 + plan.alloc.iter().enumerate().map(|(i, &a)| demands[i].expected_tokens(a) - 1.0).sum::<f64>();
    assert!((plan.tokens - want).abs() < 1e-9, "tokens {} against the allocations' sum {}", plan.tokens, want);
}
