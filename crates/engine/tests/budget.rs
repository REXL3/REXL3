//! Budget policies change how many drafts are verified and how much draft work is done, never what is committed.

use gb_engine::harness::drafted_equals_serial;
use gb_engine::mock::{mock_logits, ChainDrafter, Fault, MockBackend};
use gb_engine::{
    Backend, BudgetPolicy, Drafter, Engine, EngineConfig, ImageSpan, PrefillStep, Proposal, ProposeRequest, Request, RowOutput,
    RowRequest,
};
use gb_sampling::{top_candidates, Sampling};
use gb_sched::{Admission, CostModel, DraftCost, ModelMemory, RoundCost, StepCost};
use std::cell::RefCell;
use std::rc::Rc;

const VOCAB: usize = 512;
const SLOTS: usize = 16;

fn admission() -> Admission {
    let mm = ModelMemory {
        weights_bytes: 1 << 30,
        kv_bytes_per_token: 65536,
        state_bytes_per_slot: 150 << 20,
        activation_reserve: 1 << 30,
    };
    Admission::new(64 << 30, 6 << 30, mm, SLOTS).unwrap()
}

fn verify_table() -> CostModel {
    CostModel::from_table(vec![(1, 66.0), (8, 69.1), (16, 72.3), (32, 80.7), (48, 97.7), (64, 127.5)])
}

fn round_cost() -> RoundCost {
    RoundCost {
        verify: StepCost::rows_only(verify_table()),
        tree_verify: None,
        draft: Some(DraftCost::new(
            vec![(1, 7.5), (2, 8.0), (4, 9.5), (6, 11.0)],
            vec![(1, 3.0), (7, 4.0), (16, 4.5), (32, 8.0), (48, 12.0), (64, 16.0)],
            0.5,
        )),
        kv_ms_per_token: 0.0003,
        kv_pass_ms_per_token: 0.0,
        fixed_ms: 0.0,
    }
}

fn config() -> EngineConfig {
    EngineConfig { max_rows: 64, max_drafts: 7, cost: verify_table(), drafting: true }
}

fn requests(n: usize) -> Vec<Request> {
    let temps = [0.0, 0.6, 1.0];
    (0..n)
        .map(|i| Request {
            prompt: (0..(5 + i * 3)).map(|j| ((i * 131 + j * 17) % VOCAB) as u32).collect(),
            sampling: Sampling { seed: 4321 + i as u64, temperature: temps[i % 3], top_k: 20, top_p: 0.95, min_p: 0.0 },
            max_new: 80 + (i % 4) * 11,
            stop: vec![],
            draft: true,
            images: vec![],
            cache: true,
        })
        .collect()
}

fn engine(policy: BudgetPolicy, drafting: bool, learn: bool) -> Engine<MockBackend> {
    Engine::new(
        MockBackend::new(VOCAB, SLOTS, Fault::None),
        admission(),
        EngineConfig { drafting, ..config() },
        Box::new(|| Box::new(ChainDrafter { vocab: VOCAB, corrupt_every: 4 }) as Box<dyn Drafter>),
    )
    .with_budget(policy, round_cost())
    .with_learning(learn)
}

#[test]
fn every_policy_keeps_drafted_equal_to_serial() {
    for policy in [BudgetPolicy::Legacy, BudgetPolicy::Curve, BudgetPolicy::Plan, BudgetPolicy::Margin] {
        for learn in [false, true] {
            let r = drafted_equals_serial(&|drafting| engine(policy, drafting, learn), &requests(12))
                .unwrap_or_else(|e| panic!("{policy:?} learn={learn}: {e}"));
            assert!(r.accepted > 0, "{policy:?}: no draft was ever accepted, so the check proves nothing");
        }
    }
}

/// The mock with a native drafter whose quality depends on the slot: even slots draft the model's own continuation, odd slots
/// draft tokens it will never choose. Every ask (slot, max) is recorded, so a test can see what the policy asked the drafter to do.
struct Split {
    inner: MockBackend,
    asks: Rc<RefCell<Vec<(usize, usize)>>>,
    /// Quality varies pick by pick instead of by stream: a pick is wrong with probability 0.35, scattered, and says so with a
    /// margin of 0.2 (right picks lead by 9).
    picky: bool,
    /// `Some(n)`: odd slots draft wrongly until their context has `n` tokens and rightly after (a stream that changes its mind);
    /// even slots are right 95% of the time, scattered.
    flip: Option<usize>,
    /// What a pick is scored with: 0 its margin, 1 its own calibrated chance of being accepted (`Proposal::accept`), 2 nothing.
    scored: u8,
}

impl Backend for Split {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.inner.prefill(slot, start, tokens, candidates)
    }
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
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
                self.asks.borrow_mut().push((r.slot, r.max));
                let mut ctx = r.tokens.to_vec();
                let (mut tokens, mut margin, mut accept) = (Vec::new(), Vec::new(), Vec::new());
                for _ in 0..r.max {
                    let truth = top_candidates(&mock_logits(&ctx, VOCAB), 1).0[0];
                    // picky: about a third of the picks are wrong, scattered the way a stream's history cannot predict
                    let h = (ctx.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ (*ctx.last().unwrap() as u64).wrapping_mul(0xD1B5_4A32);
                    let wrong = match self.flip {
                        Some(at) if r.slot % 2 == 1 => ctx.len() < at,
                        Some(_) => (h >> 33) % 100 < 5,
                        None if self.picky => (h >> 33) % 100 < 35,
                        None => r.slot % 2 == 1,
                    };
                    let t = if wrong { (truth + 1) % VOCAB as u32 } else { truth };
                    tokens.push(t);
                    // a good pick leads by a lot, a bad one is a coin flip: a margin that tells the two apart
                    margin.push(if self.picky {
                        if wrong {
                            0.2
                        } else {
                            9.0
                        }
                    } else if wrong {
                        0.0
                    } else {
                        12.0
                    });
                    // a head that knows: 0.92 for a pick that will land, 0.08 for one that will not
                    accept.push(if wrong { 0.08 } else { 0.92 });
                    ctx.push(t);
                }
                match self.scored {
                    1 => Some(Proposal { tokens, accept, ..Default::default() }),
                    2 => Some(Proposal { tokens, ..Default::default() }),
                    _ => Some(Proposal { tokens, margin, ..Default::default() }),
                }
            })
            .collect()
    }
}

fn split_engine(policy: BudgetPolicy, asks: Rc<RefCell<Vec<(usize, usize)>>>, picky: bool, learn: bool) -> Engine<Split> {
    Engine::new(
        Split { inner: MockBackend::new(VOCAB, SLOTS, Fault::None), asks, picky, flip: None, scored: 0 },
        admission(),
        config(),
        Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn Drafter>),
    )
    .with_budget(policy, round_cost())
    .with_learning(learn)
}

type Run = (Vec<Vec<u32>>, Vec<(usize, usize)>, Vec<(u64, u64)>);

fn run_split(policy: BudgetPolicy) -> Run {
    run_split_with(policy, false)
}

fn run_split_with(policy: BudgetPolicy, learn: bool) -> Run {
    let asks = Rc::new(RefCell::new(Vec::new()));
    let mut e = split_engine(policy, asks.clone(), false, learn);
    for mut r in requests(6) {
        r.sampling.temperature = 0.0;
        r.max_new = 160;
        e.submit(r).unwrap();
    }
    e.run().unwrap();
    let outs = e.take_outputs();
    let a = asks.borrow().clone();
    (outs.iter().map(|o| o.tokens.clone()).collect(), a, outs.iter().map(|o| (o.drafted, o.accepted)).collect())
}

#[test]
fn deciding_before_the_draft_stops_drafting_for_streams_that_never_land() {
    let (legacy_text, legacy_asks, _) = run_split(BudgetPolicy::Legacy);
    // the frozen class prior (0.79 at the first position, measured on text that lands) takes a few rounds of a stream's own history to
    // overcome; a prior learned from half-dead traffic has already seen that most streams do not land, so it stops sooner
    for (learn, factor) in [(false, 2), (true, 4)] {
        let (plan_text, plan_asks, plan_counts) = run_split_with(BudgetPolicy::Plan, learn);
        assert_eq!(plan_text, legacy_text, "a budget policy must never change the text");
        let positions =
            |asks: &[(usize, usize)], odd: bool| -> usize { asks.iter().filter(|a| (a.0 % 2 == 1) == odd).map(|a| a.1).sum() };
        // legacy asks 7 positions of every stream every round; the plan stops asking for the streams whose drafts never match
        assert!(
            positions(&plan_asks, true) * factor < positions(&legacy_asks, true),
            "learn={learn}, odd (dead) streams: plan {} legacy {}",
            positions(&plan_asks, true),
            positions(&legacy_asks, true)
        );
        // and keeps asking for the streams that do land
        let good = positions(&plan_asks, false) as f64 / positions(&legacy_asks, false) as f64;
        assert!(good > 0.5, "learn={learn}: the plan should keep drafting for the good streams: {good}");
        assert!(plan_counts.iter().enumerate().any(|(i, c)| i % 2 == 0 && c.1 > 0), "good streams still get drafts accepted");
    }
}

#[test]
fn every_measured_policy_gives_the_same_text_as_the_legacy_one_with_a_scored_split_drafter() {
    let (legacy, _, _) = run_split(BudgetPolicy::Legacy);
    for policy in [BudgetPolicy::Curve, BudgetPolicy::Plan, BudgetPolicy::Margin] {
        assert_eq!(run_split(policy).0, legacy, "{policy:?} changed the text");
    }
}

/// Run six greedy requests whose drafts are right or wrong pick by pick, and report the text, verify rows computed for nothing
/// (verified drafts the chain never reached) and tokens per round.
fn run_picky(policy: BudgetPolicy, learn: bool) -> (Vec<Vec<u32>>, u64, f64) {
    let mut e = split_engine(policy, Rc::new(RefCell::new(Vec::new())), true, learn);
    for mut r in requests(6) {
        r.sampling.temperature = 0.0;
        r.max_new = 240;
        e.submit(r).unwrap();
    }
    e.run().unwrap();
    let t = e.spec_stats().total();
    let wasted = t.verified.iter().sum::<u64>() - t.reached.iter().sum::<u64>();
    let tokens = e.take_outputs().iter().map(|o| o.tokens.clone()).collect();
    (tokens, wasted, t.tokens_per_round())
}

#[test]
fn the_margin_policy_stops_verifying_picks_behind_a_coin_flip_and_loses_nothing_that_would_have_landed() {
    // with the measured tables frozen in (the default) and with the tables learned from this traffic
    for learn in [false, true] {
        let (plan_text, plan_waste, plan_tpr) = run_picky(BudgetPolicy::Plan, learn);
        let (margin_text, margin_waste, margin_tpr) = run_picky(BudgetPolicy::Margin, learn);
        assert_eq!(margin_text, plan_text, "a budget policy must never change the text (learn={learn})");
        // every third pick is wrong and says so; the picks behind it are never reached, so they are verify rows for nothing
        assert!(
            margin_waste * 10 < plan_waste * 7,
            "learn={learn}: wasted verify rows: margin {margin_waste}, plan {plan_waste}"
        );
        // and the picks it still verifies are the ones in front of the first coin flip, which land
        assert!(margin_tpr > plan_tpr * 0.9, "learn={learn}: tokens per round: margin {margin_tpr:.2}, plan {plan_tpr:.2}");
    }
}

/// What one probe request was offered and accepted, alone on an engine that has (or has not) already served six other requests.
fn probe(policy: BudgetPolicy, learn: bool, warm: bool) -> (u64, u64) {
    let mut e = split_engine(policy, Rc::new(RefCell::new(Vec::new())), true, learn);
    if warm {
        for mut r in requests(6) {
            r.sampling.temperature = 0.0;
            r.max_new = 240;
            e.submit(r).unwrap();
        }
        e.run().unwrap();
        e.take_outputs();
    }
    let mut r = requests(1).remove(0);
    r.sampling.temperature = 0.0;
    r.max_new = 200;
    e.submit(r).unwrap();
    e.run().unwrap();
    let o = e.take_outputs().remove(0);
    (o.drafted, o.accepted)
}

#[test]
fn a_requests_speculation_counters_do_not_depend_on_what_the_engine_served_before_unless_it_learns() {
    // a new engine does not learn: the default is the deterministic one
    assert!(!Engine::new(
        MockBackend::new(VOCAB, SLOTS, Fault::None),
        admission(),
        config(),
        Box::new(|| Box::new(gb_engine::NoDrafts))
    )
    .learns());
    assert!(engine(BudgetPolicy::Margin, true, true).learns());
    for policy in [BudgetPolicy::Plan, BudgetPolicy::Margin] {
        assert_eq!(
            probe(policy, false, false),
            probe(policy, false, true),
            "{policy:?}: the frozen budget made a different decision for the same request after other traffic"
        );
    }
    // the check can fail: a budget that learns does react to the traffic before it
    assert_ne!(
        probe(BudgetPolicy::Margin, true, false),
        probe(BudgetPolicy::Margin, true, true),
        "a learning budget should have made a different decision after other traffic, or the check above proves nothing"
    );
}

/// Four streams and room for eight draft rows in a round: the budget has to choose whose drafts ride.
fn scarce_config() -> EngineConfig {
    EngineConfig { max_rows: 12, ..config() }
}

#[test]
fn a_stream_the_budget_stopped_drafting_is_probed_again_when_it_starts_to_land() {
    // slots 0 and 2 are right 95% of the time; slots 1 and 3 are dead until their context reaches 48 tokens and right every time
    // after. With eight draft rows to give, the dead streams' estimates fall below the value of the good streams' fourth draft and
    // they are given none. A round that drafts nothing measures nothing, so unless the estimates drift back to their prior in such
    // rounds a stream stays at zero for the rest of the request, however well it would have done.
    for policy in [BudgetPolicy::Plan, BudgetPolicy::Margin] {
        let mut e = Engine::new(
            Split {
                inner: MockBackend::new(VOCAB, SLOTS, Fault::None),
                asks: Rc::default(),
                picky: false,
                flip: Some(48),
                scored: 0,
            },
            admission(),
            scarce_config(),
            Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn Drafter>),
        )
        .with_budget(policy, round_cost());
        let mut reqs = requests(4);
        for (i, r) in reqs.iter_mut().enumerate() {
            r.sampling.temperature = 0.0;
            r.max_new = if i % 2 == 0 { 900 } else { 260 };
        }
        for r in reqs {
            e.submit(r).unwrap();
        }
        e.run().unwrap();
        let outs = e.take_outputs();
        assert!(
            outs.iter().skip(1).step_by(2).all(|o| o.tokens.len() == 260),
            "the dead streams finish while the good ones still run"
        );
        for good in outs.iter().step_by(2) {
            assert!(good.accepted > 100, "{policy:?}: a good stream drafts and lands: {} of {}", good.accepted, good.drafted);
        }
        // nothing a dead stream was offered before it changed could match, so what it accepted it accepted after being probed again;
        // a stream whose estimate drifts back finishes in about 113 rounds (146 drafts accepted), one that never does needs 205 (54)
        for dead in outs.iter().skip(1).step_by(2) {
            assert!(
                dead.accepted >= 100,
                "{policy:?}: the stream that started to land was not offered drafts again: {} drafted, {} accepted",
                dead.drafted,
                dead.accepted
            );
        }
    }
}

/// Six greedy requests whose picks are wrong scattered (a third of them) and whose drafter reports only `Proposal::accept`: the text,
/// the verify rows computed for nothing and the tokens per round.
fn run_accept(policy: BudgetPolicy, scored: u8) -> (Vec<Vec<u32>>, u64, f64) {
    let mut e = Engine::new(
        Split { inner: MockBackend::new(VOCAB, SLOTS, Fault::None), asks: Rc::default(), picky: true, flip: None, scored },
        admission(),
        config(),
        Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn Drafter>),
    )
    .with_budget(policy, round_cost());
    for mut r in requests(6) {
        r.sampling.temperature = 0.0;
        r.max_new = 240;
        e.submit(r).unwrap();
    }
    e.run().unwrap();
    let t = e.spec_stats().total();
    let wasted = t.verified.iter().sum::<u64>() - t.reached.iter().sum::<u64>();
    let tokens = e.take_outputs().iter().map(|o| o.tokens.clone()).collect();
    (tokens, wasted, t.tokens_per_round())
}

#[test]
fn the_margin_policy_spends_verify_rows_by_the_drafters_own_acceptance_and_never_changes_the_text() {
    // `accept` is the only score the drafter gives here (no margin, no conf): the policy must read it
    let (plan_text, plan_waste, plan_tpr) = run_accept(BudgetPolicy::Plan, 2);
    let (scored_text, scored_waste, scored_tpr) = run_accept(BudgetPolicy::Margin, 1);
    let (blind_text, blind_waste, _) = run_accept(BudgetPolicy::Margin, 2);
    assert_eq!(scored_text, plan_text, "a budget policy must never change the text");
    assert_eq!(blind_text, plan_text);
    // without a score the margin policy has nothing to trim by and behaves as the plan does: the check below can fail
    assert_eq!(blind_waste, plan_waste, "a drafter with no score leaves the margin policy at the plan's rows");
    assert!(scored_waste * 10 < plan_waste * 7, "wasted verify rows: scored {scored_waste}, plan {plan_waste}");
    assert!(scored_tpr > plan_tpr * 0.9, "tokens per round: scored {scored_tpr:.2}, plan {plan_tpr:.2}");
}

#[test]
fn an_accept_vector_shorter_than_the_picks_is_ignored_and_one_above_one_is_clamped() {
    // a malformed score never panics and never changes the text
    struct Odd(MockBackend, usize);
    impl Backend for Odd {
        fn vocab_size(&self) -> usize {
            self.0.vocab_size()
        }
        fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
            self.0.prefill(slot, start, tokens, candidates)
        }
        fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
            self.0.step(reqs, candidates)
        }
        fn commit(&mut self, slot: usize, keep: usize) {
            self.0.commit(slot, keep)
        }
        fn release(&mut self, slot: usize) {
            self.0.release(slot)
        }
        fn propose_batch_scored(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Proposal>> {
            reqs.iter()
                .map(|r| {
                    let mut ctx = r.tokens.to_vec();
                    let mut tokens = Vec::new();
                    for _ in 0..r.max {
                        let t = top_candidates(&mock_logits(&ctx, VOCAB), 1).0[0];
                        tokens.push(t);
                        ctx.push(t);
                    }
                    let accept = match self.1 {
                        0 => vec![0.9; tokens.len().saturating_sub(1)],
                        1 => vec![7.5; tokens.len()],
                        _ => vec![f32::NAN; tokens.len()],
                    };
                    Some(Proposal { tokens, accept, ..Default::default() })
                })
                .collect()
        }
    }
    let run = |kind: usize| {
        let mut e = Engine::new(
            Odd(MockBackend::new(VOCAB, SLOTS, Fault::None), kind),
            admission(),
            config(),
            Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn Drafter>),
        )
        .with_budget(BudgetPolicy::Margin, round_cost());
        for mut r in requests(3) {
            r.sampling.temperature = 0.0;
            r.max_new = 60;
            e.submit(r).unwrap();
        }
        e.run().unwrap();
        e.take_outputs().iter().map(|o| o.tokens.clone()).collect::<Vec<_>>()
    };
    let (short, big, nan) = (run(0), run(1), run(2));
    assert_eq!(short, big);
    assert_eq!(short, nan);
}

/// A draft block that runs whole (the cost of a round does not depend on how many picks are asked of it: a flat head table) and a
/// verify step that grows by about a millisecond a row, as a deep-block drafter on a mixture-of-experts target prices it.
fn block_cost() -> RoundCost {
    RoundCost {
        verify: StepCost::rows_only(CostModel::from_table(vec![(1, 15.0), (16, 31.0), (32, 47.0), (64, 79.0)])),
        tree_verify: None,
        draft: Some(DraftCost::new(vec![(1, 4.0), (2, 4.5), (4, 5.5), (8, 8.0)], vec![(1, 0.001), (128, 0.001)], 0.0)),
        kv_ms_per_token: 0.0,
        kv_pass_ms_per_token: 0.0,
        fixed_ms: 0.0,
    }
}

/// Two greedy streams under `policy` on a drafter that scores its picks (`Proposal::accept`): slot 0's picks always land, slot 1's never
/// do. Returns each stream's text, rounds, drafted and accepted, and what the drafter was asked for slot 0 each round.
#[allow(clippy::type_complexity)]
fn run_block(policy: BudgetPolicy) -> (Vec<Vec<u32>>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<usize>) {
    let asks = Rc::new(RefCell::new(Vec::new()));
    let mut e = Engine::new(
        Split { inner: MockBackend::new(VOCAB, SLOTS, Fault::None), asks: asks.clone(), picky: false, flip: None, scored: 1 },
        admission(),
        EngineConfig { max_drafts: 15, ..config() },
        Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn Drafter>),
    )
    .with_budget(policy, block_cost());
    for mut r in requests(2) {
        r.sampling.temperature = 0.0;
        r.max_new = 300;
        e.submit(r).unwrap();
    }
    e.run().unwrap();
    let outs = e.take_outputs();
    let first: Vec<usize> = asks.borrow().iter().filter(|a| a.0 == 0).map(|a| a.1).collect();
    (
        outs.iter().map(|o| o.tokens.clone()).collect(),
        outs.iter().map(|o| o.rounds).collect(),
        outs.iter().map(|o| o.drafted).collect(),
        outs.iter().map(|o| o.accepted).collect(),
        first,
    )
}

#[test]
fn the_scored_policy_asks_a_scoring_drafter_for_its_whole_block_and_trims_by_its_confidences() {
    let (plan_text, ..) = run_block(BudgetPolicy::Plan);
    let (margin_text, margin_rounds, _, margin_acc, margin_asks) = run_block(BudgetPolicy::Margin);
    let (scored_text, scored_rounds, scored_drafted, scored_acc, scored_asks) = run_block(BudgetPolicy::Scored);
    // a budget policy never changes what is committed
    assert_eq!(margin_text, plan_text);
    assert_eq!(scored_text, plan_text);
    // the pre-draft plan's depth cut: the frozen class prior fades with depth, so a stream whose picks all land is still not asked for
    // the deep ones (the cut saves no draft work here: the block runs whole); `Scored` asks for everything the stream can use
    let deepest = |asks: &[usize]| asks.iter().copied().max().unwrap_or(0);
    assert!(deepest(&margin_asks) < 15, "the margin policy's ask is cut by the plan: {}", deepest(&margin_asks));
    assert_eq!(deepest(&scored_asks), 15, "the scored policy asks for the whole block");
    assert!(
        scored_asks.iter().filter(|&&a| a == 15).count() * 2 > scored_asks.len(),
        "most rounds of a stream that lands ask for the whole block: {scored_asks:?}"
    );
    // and the confidences keep what lands: fewer rounds for the same text on the stream whose picks land
    assert!(
        scored_rounds[0] * 10 < margin_rounds[0] * 9,
        "rounds of the stream that lands: scored {} margin {}",
        scored_rounds[0],
        margin_rounds[0]
    );
    assert!(scored_acc[0] > margin_acc[0] / 2, "accepted {} vs {}", scored_acc[0], margin_acc[0]);
    // the stream whose picks never land is not given verify rows for them: its drafted rows stay a small part of its rounds
    assert!(
        scored_drafted[1] < scored_rounds[1] * 3,
        "dead stream: {} drafted in {} rounds",
        scored_drafted[1],
        scored_rounds[1]
    );
    assert_eq!(scored_acc[1], 0);
}
