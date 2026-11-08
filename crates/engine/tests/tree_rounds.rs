//! Tree rounds through the engine on the CPU mock (docs/TREE_DRAFTING.md, S2): a backend that drafts a candidate lattice and verifies trees
//! must leave every reply byte for byte what serial decoding gives, whatever tree the search builds, commit the accepted PATH (a sibling's
//! recurrent state, never the last row's), keep the allocation inside the row budget, fail a poisoned stream alone, and fall back to chains
//! where the backend or the drafter has no trees. Each test here was run against the code it guards with the guarded line broken.

use gb_engine::harness::drafted_equals_serial;
use gb_engine::mock::{mock_logits, Fault, Faults, MockBackend};
use gb_engine::{
    Backend, BackendError, Drafter, Engine, EngineConfig, FinishReason, NoDrafts, PrefillStep, Proposal, ProposeRequest, Request,
    RowOutput, RowRequest, TreeConfig, TreeRequest,
};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};
use gb_spec::tree::Lattice;
use std::cell::RefCell;
use std::rc::Rc;

const VOCAB: usize = 512;
const SLOTS: usize = 16;
const K: usize = 8;

fn admission() -> Admission {
    let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 65536, state_bytes_per_slot: 150 << 20, activation_reserve: 0 };
    Admission::new(64 << 30, 0, mm, SLOTS).unwrap()
}

fn cost() -> CostModel {
    CostModel::from_table(vec![(1, 70.0), (16, 74.0), (32, 110.0), (64, 190.0)])
}

fn hash(a: u64, b: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9E3779B97F4A7C15) ^ b.wrapping_add(0xD1B54A32D192ED03);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^ (x >> 29)
}

/// A drafter's lattice over the mock's own next-token function: `rows` levels of the `K` best tokens under the chain's context or under
/// any of the previous level's top candidates as the parent. The unary score carries a hashed error, so the best-first walk sometimes
/// picks a wrong token with the right one a sibling or a cousin. Returns the greedy chain too.
fn lattice(tokens: &[u32], rows: usize, noise: f32) -> (Vec<u32>, Lattice) {
    let mut ids = vec![0u32; rows * K];
    let mut unary = vec![0f32; rows * K];
    let mut edge = vec![0f32; rows * K * K];
    let mut ctx = tokens.to_vec();
    let mut chain = Vec::new();
    let mut parent = 0usize;
    for d in 0..rows {
        // candidates: the best of the logits under the chain context and under each previous candidate as the last token
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
        // the selector's own walk: first maximum of unary + edge
        let s = |c: usize| unary[d * K + c] + edge[(d * K + parent) * K + c];
        let best = (1..K).fold(0, |b, c| if s(c) > s(b) { c } else { b });
        chain.push(ids[d * K + best]);
        ctx.push(ids[d * K + best]);
        parent = best;
    }
    (chain, Lattice { rows, k: K, ids, unary, edge, restricted_head: false })
}

fn chain_ids(ids: &[u32], d: usize) -> &[u32] {
    &ids[d * K..(d + 1) * K]
}

/// What the wrapper saw of every tree window: checked for shape at the call, kept for the tests.
#[derive(Default)]
struct Seen {
    windows: usize,
    /// Windows with a row that has a parent other than the row before it.
    branched: usize,
    max_rows_in_a_step: usize,
    max_depth: usize,
}

/// The mock behind a lattice-drafting backend. `trees` says whether `supports_trees`; `flat`: the drafts carry no lattice.
struct Lat {
    inner: MockBackend,
    flat: bool,
    /// Size of the hashed error on the lattice's unary scores (0: the draft's picks are the mock's own greedy chain).
    noise: f32,
    seen: Rc<RefCell<Seen>>,
}

impl Lat {
    fn new(inner: MockBackend, flat: bool) -> (Self, Rc<RefCell<Seen>>) {
        let seen = Rc::new(RefCell::new(Seen::default()));
        (Self { inner, flat, noise: 3.0, seen: seen.clone() }, seen)
    }
}

impl Backend for Lat {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn supports_trees(&self) -> bool {
        self.inner.supports_trees()
    }
    fn step_tree(&mut self, reqs: &[TreeRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        let mut seen = self.seen.borrow_mut();
        let mut rows = 0;
        for r in reqs {
            assert_eq!(r.tokens.len(), r.parents.len());
            assert_eq!(r.parents[0], -1, "row 0 is the root");
            assert!(
                r.parents.iter().enumerate().skip(1).all(|(i, &p)| 0 <= p && (p as usize) < i),
                "parents come first: {:?}",
                r.parents
            );
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
    fn take_error(&mut self) -> Option<BackendError> {
        self.inner.take_error()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.inner.prefill(slot, start, tokens, candidates)
    }
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[gb_engine::ImageSpan],
        done: usize,
        candidates: usize,
    ) -> PrefillStep {
        self.inner.prefill_slice(slot, prompt, images, done, candidates)
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.inner.step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.inner.commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.inner.release(slot)
    }
    fn propose_batch_scored(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Proposal>> {
        reqs.iter()
            .map(|r| {
                let (chain, l) = lattice(r.tokens, r.max, self.noise);
                Some(Proposal { tokens: chain, lattice: (!self.flat).then_some(l), ..Default::default() })
            })
            .collect()
    }
}

fn no_drafts() -> Box<dyn Fn() -> Box<dyn Drafter>> {
    Box::new(|| Box::new(NoDrafts) as Box<dyn Drafter>)
}

fn config(drafting: bool, max_rows: usize) -> EngineConfig {
    EngineConfig { max_rows, max_drafts: 7, cost: cost(), drafting }
}

fn mock(trees: bool, faults: Faults) -> MockBackend {
    let m = MockBackend::new(VOCAB, SLOTS, Fault::None).with_deep_state().with_faults(faults);
    if trees {
        m.with_trees()
    } else {
        m
    }
}

fn requests(n: usize) -> Vec<Request> {
    let temps = [0.0, 0.6, 1.0];
    (0..n)
        .map(|i| {
            let prompt: Vec<u32> = (0..(5 + i * 3)).map(|j| ((i * 131 + j * 17 + 11) % VOCAB) as u32).collect();
            Request {
                sampling: Sampling { seed: 4321 + i as u64, temperature: temps[i % 3], top_k: 20, top_p: 0.95, min_p: 0.0 },
                prompt,
                max_new: 64 + (i % 5) * 17,
                stop: vec![],
                draft: true,
                images: vec![],
                cache: true,
            }
        })
        .collect()
}

/// An engine that verifies trees when `drafting`, and the serial engine (the reference) otherwise.
fn tree_engine(drafting: bool, max_rows: usize, flat: bool, trees_backend: bool) -> (Engine<Lat>, Rc<RefCell<Seen>>) {
    tree_engine_with(drafting, max_rows, flat, trees_backend, TreeConfig::default())
}

fn tree_engine_with(
    drafting: bool,
    max_rows: usize,
    flat: bool,
    trees_backend: bool,
    cfg: TreeConfig,
) -> (Engine<Lat>, Rc<RefCell<Seen>>) {
    let (b, seen) = Lat::new(mock(trees_backend, Faults::default()), flat);
    tree_engine_on(b, seen, drafting, max_rows, cfg)
}

fn tree_engine_on(
    b: Lat,
    seen: Rc<RefCell<Seen>>,
    drafting: bool,
    max_rows: usize,
    cfg: TreeConfig,
) -> (Engine<Lat>, Rc<RefCell<Seen>>) {
    let e = Engine::new(b, admission(), config(drafting, max_rows), no_drafts());
    (if drafting { e.with_trees(cfg) } else { e }, seen)
}

/// Prompts that repeat themselves (a block of 12 tokens four times, the last block cut short), so that a copy of the stream's own text
/// is what the mock decodes next.
fn repetitive_requests(n: usize) -> Vec<Request> {
    requests(n)
        .into_iter()
        .enumerate()
        .map(|(i, mut r)| {
            let block: Vec<u32> = (0..12).map(|j| ((i * 37 + j * 11 + 5) % VOCAB) as u32).collect();
            r.prompt = block.iter().cycle().take(46).copied().collect();
            r.max_new = 90;
            r
        })
        .collect()
}

#[test]
fn tree_rounds_equal_serial_decoding_and_commit_branches() {
    let seen = Rc::new(RefCell::new(None));
    let keep = seen.clone();
    let make = move |drafting: bool| {
        let (e, s) = tree_engine(drafting, 64, false, true);
        if drafting {
            *keep.borrow_mut() = Some(s);
        }
        e
    };
    let r = drafted_equals_serial(&make, &requests(9)).unwrap();
    assert!(r.accepted > 0 && r.drafted > r.accepted, "{r:?}");
    let seen = seen.borrow();
    let seen = seen.as_ref().unwrap().borrow();
    assert!(
        seen.windows > 0 && seen.branched > 0,
        "the trees must have branched: {} windows, {} branched",
        seen.windows,
        seen.branched
    );
    // the depth-adaptive block (`gb_sched::adaptive_block`, docs/TREE_DRAFTING.md section 13) grows the draft's ask past the
    // trained 7 once a stream keeps its levels, so the bound is the kernels' 15 here, not 7
    assert!(seen.max_depth <= 15, "a tree never goes deeper than the block's 15 levels: {}", seen.max_depth);
    // the stats saw accepted branches past the first child (an accepted sibling is committed through commit_path); the
    // draft must be wrong often enough for that to happen whatever the shipped calibration's knots say (noise 6.0: the
    // matched-dump table of docs/TREE_DRAFTING.md section 13 rates a mid-range raw number higher than the interim one
    // did, so the allocation buys depth first and the default 3.0 stops producing accepted siblings)
    let (mut b, _) = Lat::new(mock(true, Faults::default()), false);
    b.noise = 6.0;
    let (mut e, _) = tree_engine_on(b, Rc::new(RefCell::new(Seen::default())), true, 64, TreeConfig::default());
    for q in requests(9) {
        e.submit(q).unwrap();
    }
    e.run().unwrap();
    let stats = e.spec_stats();
    let rounds: u64 = stats.trees.iter().map(|t| t.rounds).sum();
    let siblings: u64 = stats.trees.iter().map(|t| t.rank_accepted[1..].iter().sum::<u64>()).sum();
    assert!(rounds > 0, "no tree round ran");
    assert!(siblings > 0, "no sibling branch was ever accepted: the path commit was never exercised");
    for t in &stats.trees {
        assert!(t.accepted_total <= t.verified, "{t:?}");
        assert!(t.verified <= t.offered, "the allocation never verifies more than the search ranked: {t:?}");
    }
}

#[test]
fn committing_the_last_row_instead_of_the_path_breaks_exactness() {
    // the mock under Fault::RollbackLeak keeps the state of the window's LAST row after commit: with trees that is not the accepted path's
    let make = |drafting: bool| {
        let (b, _) = Lat::new(MockBackend::new(VOCAB, SLOTS, Fault::RollbackLeak).with_deep_state().with_trees(), false);
        let e = Engine::new(b, admission(), config(drafting, 64), no_drafts());
        if drafting {
            e.with_trees(TreeConfig::default())
        } else {
            e
        }
    };
    assert!(drafted_equals_serial(&make, &requests(9)).is_err(), "a leaked state must be caught");
}

#[test]
fn the_allocation_respects_the_row_budget_at_every_stream_count() {
    for (streams, max_rows) in [(1usize, 12usize), (3, 20), (6, 24), (9, 18)] {
        let (mut e, seen) = tree_engine(true, max_rows, false, true);
        for q in requests(streams) {
            e.submit(q).unwrap();
        }
        e.run().unwrap();
        let seen = seen.borrow();
        assert!(seen.windows > 0);
        // every stream keeps its root row; past that the planner stays inside the budget
        assert!(
            seen.max_rows_in_a_step <= max_rows.max(streams),
            "{streams} streams, budget {max_rows}: a step ran {} rows",
            seen.max_rows_in_a_step
        );
    }
}

/// `--window-rows 128` (c32) widens the window CHAINS may use, not the trees' without a tree cost table: a tree window is then never planned past `TREE_MAX_ROWS` rows (the tree
/// table's reach and the tree kernels' tested width), whatever `EngineConfig::max_rows` says. Rows are nearly free in this cost model and 16
/// streams offer far more nodes than 64 rows hold, so the cap is what binds; below it the engine's window still binds first.
#[test]
fn a_wide_engine_window_never_gives_a_tree_window_more_than_the_tree_rows() {
    let flat = |max_rows: usize| EngineConfig {
        max_rows,
        max_drafts: 7,
        cost: CostModel::from_table(vec![(1, 70.0), (256, 71.0)]),
        drafting: true,
    };
    let widest = |max_rows: usize, streams: usize| {
        let (b, seen) = Lat::new(mock(true, Faults::default()), false);
        let mut e = Engine::new(b, admission(), flat(max_rows), no_drafts()).with_trees(TreeConfig::default());
        for q in requests(streams) {
            e.submit(q).unwrap();
        }
        e.run().unwrap();
        let rows = seen.borrow().max_rows_in_a_step;
        rows
    };
    let wide = widest(128, 16);
    assert_eq!(wide, gb_sched::TREE_MAX_ROWS, "the 128-row window gave a tree window of {wide} rows");
    assert_eq!(widest(64, 16), wide, "a 64-row window plans the same trees");
    assert!(widest(40, 16) <= 40, "a window under the cap binds first");
}

#[test]
fn a_drafter_without_a_lattice_or_a_backend_without_trees_keeps_chains() {
    // no lattice: chains through the tree round, same replies, never a branch
    let make = |drafting: bool| tree_engine(drafting, 64, true, true).0;
    let r = drafted_equals_serial(&make, &requests(6)).unwrap();
    assert!(r.accepted > 0, "{r:?}");
    let (mut e, seen) = tree_engine(true, 64, true, true);
    for q in requests(6) {
        e.submit(q).unwrap();
    }
    e.run().unwrap();
    assert_eq!(seen.borrow().branched, 0);
    // a backend that does not verify trees: the engine runs chain rounds, never calls step_tree
    let make = |drafting: bool| tree_engine(drafting, 64, false, false).0;
    let r = drafted_equals_serial(&make, &requests(6)).unwrap();
    assert!(r.accepted > 0, "{r:?}");
    let (mut e, seen) = tree_engine(true, 64, false, false);
    assert!(!e.verifies_trees());
    for q in requests(6) {
        e.submit(q).unwrap();
    }
    e.run().unwrap();
    assert_eq!(seen.borrow().windows, 0);
    assert_eq!(e.spec_stats().trees.iter().map(|t| t.rounds).sum::<u64>(), 0);
}

#[test]
fn a_poisoned_stream_fails_alone_in_a_tree_round_and_the_others_are_exact() {
    const POISON: u32 = 7;
    let mut reqs = requests(4);
    for (i, r) in reqs.iter_mut().enumerate() {
        r.prompt[0] = if i == 2 { POISON } else { 100 + i as u32 };
    }
    let faults = Faults { poison_first_token: Some(POISON), ..Default::default() };
    let (b, _) = Lat::new(mock(true, faults), false);
    let mut e = Engine::new(b, admission(), config(true, 64), no_drafts()).with_trees(TreeConfig::default());
    let free = e.admission().available();
    let ids: Vec<u64> = reqs.iter().map(|r| e.submit_deferred(r.clone()).unwrap()).collect();
    let mut steps = 0;
    while e.busy() {
        e.step().expect("a poisoned row is not a step error");
        steps += 1;
        assert!(steps < 10_000);
    }
    let failed = e.take_failed();
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert_eq!(failed[0].0, ids[2]);
    let outs = e.take_outputs();
    assert_eq!(outs.len(), 3);
    for (i, id) in ids.iter().enumerate().filter(|(i, _)| *i != 2) {
        let o = outs.iter().find(|o| o.id == *id).unwrap();
        assert_eq!(o.finish, Some(FinishReason::Length));
        let (b, _) = Lat::new(mock(false, Faults::default()), false);
        let mut solo = Engine::new(b, admission(), config(false, 64), no_drafts());
        solo.submit(reqs[i].clone()).unwrap();
        solo.run().unwrap();
        assert_eq!(
            o.tokens,
            solo.take_outputs().remove(0).tokens,
            "sequence {i} must equal its serial reply despite the neighbour's failure"
        );
    }
    assert_eq!(e.admission().available(), free, "the failed sequence's slot and memory came back");
}

#[test]
fn counters_define_drafted_and_accepted_for_trees() {
    // drafted = nodes verified (rows past the root); accepted = accepted path length past the root; both per request and in the tree stats
    let (mut e, _) = tree_engine(true, 64, false, true);
    for q in requests(5) {
        e.submit(q).unwrap();
    }
    e.run().unwrap();
    let outs = e.take_outputs();
    let drafted: u64 = outs.iter().map(|o| o.drafted).sum();
    let accepted: u64 = outs.iter().map(|o| o.accepted).sum();
    let stats = e.spec_stats();
    assert_eq!(stats.trees.iter().map(|t| t.verified).sum::<u64>(), drafted);
    assert_eq!(stats.trees.iter().map(|t| t.accepted_total).sum::<u64>(), accepted);
    assert!(accepted <= drafted && accepted > 0);
    // the per-position counters (spec_hist, the B6 tables) see the same rounds and the same accepted drafts (every path here is under 16 deep)
    let total = stats.total();
    assert_eq!(total.rounds, stats.trees.iter().map(|t| t.rounds).sum::<u64>());
    assert_eq!(total.accepted.iter().sum::<u64>(), accepted);
    // a committed token is an accepted draft or the sample past it (the last round may be cut at max_new), after the prefill's own token
    for o in &outs {
        let rounds_tokens = o.tokens.len() as u64 - 1;
        assert!(rounds_tokens <= o.accepted + o.rounds && rounds_tokens + 8 > o.accepted + o.rounds, "{o:?}");
    }
}

#[test]
fn a_tree_never_goes_deeper_than_the_tokens_the_request_has_left() {
    // max_new 4: the prefill gave the first token, so the first round has 3 left and may verify at most 2 drafts past the root
    let (mut e, seen) = tree_engine(true, 64, false, true);
    let r = Request { max_new: 4, ..requests(1).remove(0) };
    e.submit(r).unwrap();
    e.run().unwrap();
    let seen = seen.borrow();
    assert!(seen.windows > 0);
    assert!(seen.max_depth <= 2, "a tree reached depth {} with 3 tokens left", seen.max_depth);
}

#[test]
fn a_copy_branch_keeps_replies_exact_goes_deeper_than_the_draft_and_is_accepted() {
    let cfg = TreeConfig { copy_min_match: 4, ..TreeConfig::default() };
    let make = |drafting: bool| quiet_engine(drafting, cfg).0;
    let r = drafted_equals_serial(&make, &repetitive_requests(7)).unwrap();
    assert!(r.accepted > 0, "{r:?}");
    let (mut e, seen) = quiet_engine(true, cfg);
    for q in repetitive_requests(7) {
        e.submit(q).unwrap();
    }
    e.run().unwrap();
    let st = e.spec_stats();
    let (rounds, verified, accepted): (u64, u64, u64) =
        st.trees.iter().fold((0, 0, 0), |a, t| (a.0 + t.copy_rounds, a.1 + t.copy_verified, a.2 + t.copy_accepted));
    assert!(rounds > 0 && verified > 0, "no window carried a copy branch");
    assert!(accepted > 0 && accepted <= verified, "{accepted} of {verified} copied tokens accepted");
    assert!(seen.borrow().max_depth > 7, "a copy branch goes past the draft's 7 levels: depth {}", seen.borrow().max_depth);
    // without the knob nothing is copied and no tree is deeper than the draft
    let (mut off, seen_off) = quiet_engine(true, TreeConfig::default());
    for q in repetitive_requests(7) {
        off.submit(q).unwrap();
    }
    off.run().unwrap();
    assert_eq!(off.spec_stats().trees.iter().map(|t| t.copy_rounds).sum::<u64>(), 0);
    // without the copy knob the depth is the draft's own (the adaptive block's), never a copied token's 24
    assert!(seen_off.borrow().max_depth <= 15, "no copy branch: depth {}", seen_off.borrow().max_depth);
}

#[test]
fn a_copy_branch_never_outruns_the_tokens_the_request_has_left() {
    let cfg = TreeConfig { copy_min_match: 4, ..TreeConfig::default() };
    let (mut e, seen) = quiet_engine(true, cfg);
    // 40 tokens to go with a long repeated context: the deepest tree of a round never has more depth than the tokens left
    let r = Request { max_new: 14, ..repetitive_requests(1).remove(0) };
    e.submit(r).unwrap();
    e.run().unwrap();
    assert!(seen.borrow().max_depth <= 12, "depth {} with 14 new tokens in all", seen.borrow().max_depth);
}

/// The copy branch goes through the same row budget as every other node (the c32 join: `--window-rows` is the engine's `max_rows`, and
/// `RoundCost::tree_window_rows` caps a tree window further): copied tokens are `TreeDemand` probabilities like the lattice's, so the allocation
/// that stops a lattice tree at the window stops a copy chain. Rows are nearly free in this cost model, the draft is right and the text repeats,
/// so the allocation WANTS every row it may have (a copy chain of 24 and a 7-deep picks path want 31 rows on top of the lattice's): what binds is
/// the budget. Each configuration must also have run copy rounds and accepted copied tokens, or the bound it checks is vacuous.
#[test]
fn a_copy_branch_is_planned_inside_the_window_rows_at_every_stream_count() {
    let flat = |max_rows: usize| EngineConfig {
        max_rows,
        max_drafts: 7,
        cost: CostModel::from_table(vec![(1, 70.0), (256, 71.0)]),
        drafting: true,
    };
    let cfg = TreeConfig { copy_min_match: 4, ..TreeConfig::default() };
    let run = |max_rows: usize, streams: usize| {
        let (mut b, seen) = Lat::new(MockBackend::new(VOCAB, SLOTS, Fault::None).with_trees(), false);
        b.noise = 0.0;
        let mut e = Engine::new(b, admission(), flat(max_rows), no_drafts()).with_trees(cfg);
        for q in repetitive_requests(streams) {
            e.submit(q).unwrap();
        }
        e.run().unwrap();
        let st = e.spec_stats();
        let copy_rounds: u64 = st.trees.iter().map(|t| t.copy_rounds).sum();
        let copy_accepted: u64 = st.trees.iter().map(|t| t.copy_accepted).sum();
        let rows = seen.borrow().max_rows_in_a_step;
        (rows, copy_rounds, copy_accepted)
    };
    for (streams, max_rows) in [(1usize, 12usize), (1, 24), (2, 20), (4, 30), (6, 48)] {
        let (rows, copy_rounds, copy_accepted) = run(max_rows, streams);
        assert!(
            copy_rounds > 0 && copy_accepted > 0,
            "{streams} streams, window {max_rows}: no copy branch ran ({copy_rounds} rounds, {copy_accepted} accepted)"
        );
        assert!(
            rows <= max_rows.max(streams),
            "{streams} streams, window {max_rows}: a step ran {rows} rows with the copy branch on"
        );
        assert!(
            rows >= max_rows.min(24),
            "{streams} streams, window {max_rows}: the budget never bound ({rows} rows), the bound is untested"
        );
    }
    // a wide engine window (`--window-rows 128`) without a tree cost table never gives a tree window more than the tree rows, copy or not
    let (rows, copy_rounds, _) = run(128, 2);
    assert!(copy_rounds > 0);
    assert!(rows <= gb_sched::TREE_MAX_ROWS, "the wide window gave a copy tree window of {rows} rows");
}

/// A mock without the recurrent-state quirk and a draft without error: its picks are the mock's own greedy chain, so a long chain is likely
/// right and the copy branch off its end is worth rows.
fn quiet_engine(drafting: bool, cfg: TreeConfig) -> (Engine<Lat>, Rc<RefCell<Seen>>) {
    let (mut b, seen) = Lat::new(MockBackend::new(VOCAB, SLOTS, Fault::None).with_trees(), false);
    b.noise = 0.0;
    tree_engine_on(b, seen, drafting, 64, cfg)
}

// ---------------------------------------------------------------- reasoning budget on tree rounds (gb_engine::ReasoningBudget)

fn run_budgeted<B: Backend>(e: &mut Engine<B>, reqs: &[(Request, Option<gb_engine::ReasoningBudget>)]) -> Vec<gb_engine::Output> {
    let ids: Vec<u64> = reqs.iter().map(|(r, b)| e.submit_with_budget(r.clone(), *b).unwrap()).collect();
    e.run().unwrap();
    let mut outs = e.take_outputs();
    assert_eq!(outs.len(), reqs.len());
    outs.sort_by_key(|o| ids.iter().position(|&i| i == o.id).unwrap());
    outs
}

/// Drafted (trees) against serial with budgets that bite at several indices; returns the drafted engine's counters, the forced cases and
/// what the tree windows looked like.
fn tree_budget_run(flat: bool, clip: bool) -> (gb_engine::BudgetStats, u64, u64) {
    let serial = || tree_engine(false, 64, flat, true).0;
    let mut cases = Vec::new();
    for (i, r) in repetitive_requests(4).into_iter().chain(requests(8)).enumerate() {
        let plain = run_budgeted(&mut serial(), &[(r.clone(), None)]).remove(0).tokens;
        let absent = (0..VOCAB as u32).find(|t| !plain.contains(t)).unwrap();
        let after = [0usize, 2, 7, 13, 21, 34][i % 6];
        cases.push((r, Some(gb_engine::ReasoningBudget { after, close: absent }), plain));
    }
    let pairs: Vec<_> = cases.iter().map(|(r, b, _)| (r.clone(), *b)).collect();
    let want: Vec<Vec<u32>> =
        pairs.iter().map(|p| run_budgeted(&mut serial(), std::slice::from_ref(p)).remove(0).tokens).collect();
    let (e, seen) = tree_engine(true, 64, flat, true);
    let mut e = if clip { e } else { e.without_budget_draft_clip() };
    let outs = run_budgeted(&mut e, &pairs);
    let mut accepted = 0;
    for (i, ((o, w), (_, b, plain))) in outs.iter().zip(&want).zip(&cases).enumerate() {
        assert_eq!(&o.tokens, w, "case {i}: drafted (trees) != serial");
        let n = b.unwrap().after;
        assert_eq!(o.tokens[n], b.unwrap().close, "case {i}");
        assert_eq!(o.tokens[..n], plain[..n], "case {i}");
        accepted += o.accepted;
    }
    assert!(accepted > 0 && seen.borrow().windows > 0, "no tree window landed a draft");
    let branched = seen.borrow().branched as u64;
    (e.budget_stats(), cases.len() as u64, branched)
}

#[test]
fn a_reasoning_budget_keeps_tree_rounds_equal_to_serial() {
    for flat in [false, true] {
        let (stats, n, branched) = tree_budget_run(flat, true);
        assert_eq!(stats.forced, n, "flat {flat}");
        assert_eq!(stats.cut_runs, 0, "flat {flat}: the clip keeps every window short of the index");
        if !flat {
            assert!(branched > 0, "the lattice drafter built no branching tree");
        }
    }
}

#[test]
fn the_commit_time_rule_alone_keeps_tree_rounds_equal_to_serial() {
    for flat in [false, true] {
        let (stats, n, _) = tree_budget_run(flat, false);
        assert_eq!(stats.forced, n, "flat {flat}");
        assert!(stats.cut_runs > 0, "flat {flat}: no accepted path crossed a budget's index; the cut was not exercised");
    }
}
