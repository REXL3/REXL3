//! The verify-row cap (`EngineConfig::max_rows`) against many streams.
//!
//! `max_rows` is the most rows one forward carries (every stream's base row plus its drafts). Every admitted sequence needs its base row
//! in the next round, so the same number bounds how many sequences the engine admits (`NoSlot`), and drafts are what the budget trims:
//! 32 streams that each offer a whole block of 8 rows (256 rows) must run at the cap, with every stream in the round and none refused.

use gb_engine::harness::drafted_equals_serial;
use gb_engine::mock::{ChainDrafter, Fault, MockBackend};
use gb_engine::{
    Backend, BudgetPolicy, Drafter, Engine, EngineConfig, ImageSpan, PrefillStep, Request, RowOutput, RowRequest, SubmitError,
};
use gb_sampling::Sampling;
use gb_sched::{Admission, AdmitError, CostModel, DraftCost, ModelMemory, RoundCost, StepCost};
use std::cell::RefCell;
use std::rc::Rc;

const VOCAB: usize = 512;

/// The mock backend, recording the shape of every verify step as (streams, rows).
struct Logged {
    inner: MockBackend,
    steps: Rc<RefCell<Vec<(usize, usize)>>>,
}

impl Backend for Logged {
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
        self.steps.borrow_mut().push((reqs.len(), reqs.iter().map(|r| r.tokens.len()).sum()));
        self.inner.step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.inner.commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.inner.release(slot)
    }
}

fn admission(slots: usize) -> Admission {
    let mm = ModelMemory {
        weights_bytes: 1 << 30,
        kv_bytes_per_token: 65536,
        state_bytes_per_slot: 150 << 20,
        activation_reserve: 1 << 30,
    };
    Admission::new(100 << 30, 6 << 30, mm, slots).unwrap()
}

/// Rows are nearly free (70 ms for 1 row, 71 for 256), so the budget spends every row it is allowed: the cap is what binds.
fn flat_cost() -> CostModel {
    CostModel::from_table(vec![(1, 70.0), (256, 71.0)])
}

fn round_cost() -> RoundCost {
    RoundCost {
        verify: StepCost::rows_only(flat_cost()),
        tree_verify: None,
        draft: Some(DraftCost::new(vec![(1, 5.0), (32, 6.0)], vec![(1, 4.0), (256, 4.5)], 0.0)),
        kv_ms_per_token: 0.0,
        kv_pass_ms_per_token: 0.0,
        fixed_ms: 0.0,
    }
}

fn requests(n: usize) -> Vec<Request> {
    (0..n)
        .map(|i| Request {
            prompt: (0..(6 + i)).map(|j| ((i * 131 + j * 17) % VOCAB) as u32).collect(),
            sampling: Sampling {
                seed: 99 + i as u64,
                temperature: if i % 2 == 0 { 0.0 } else { 0.8 },
                top_k: 20,
                top_p: 0.95,
                min_p: 0.0,
            },
            max_new: 48,
            stop: vec![],
            draft: true,
            images: vec![],
            cache: true,
        })
        .collect()
}

fn engine(
    max_rows: usize,
    slots: usize,
    policy: BudgetPolicy,
    drafting: bool,
    log: &Rc<RefCell<Vec<(usize, usize)>>>,
) -> Engine<Logged> {
    Engine::new(
        Logged { inner: MockBackend::new(VOCAB, slots, Fault::None), steps: log.clone() },
        admission(slots),
        EngineConfig { max_rows, max_drafts: 7, cost: flat_cost(), drafting },
        // a perfect drafter: the whole block of 7 is always on offer and always right
        Box::new(|| Box::new(ChainDrafter { vocab: VOCAB, corrupt_every: u64::MAX }) as Box<dyn Drafter>),
    )
    .with_budget(policy, round_cost())
}

#[test]
fn thirty_two_streams_offering_whole_blocks_run_at_the_cap_and_none_is_refused() {
    for max_rows in [64usize, 96, 128] {
        for policy in [BudgetPolicy::Legacy, BudgetPolicy::Margin] {
            let log = Rc::new(RefCell::new(Vec::new()));
            let mut e = engine(max_rows, 40, policy, true, &log);
            for (i, r) in requests(32).into_iter().enumerate() {
                e.submit(r).unwrap_or_else(|err| panic!("max_rows {max_rows} {policy:?}: stream {i} refused: {err}"));
            }
            e.run().unwrap();
            assert_eq!(e.take_outputs().len(), 32, "every stream finished");
            let steps = log.borrow();
            let full: Vec<_> = steps.iter().filter(|s| s.0 == 32).collect();
            assert!(!full.is_empty(), "{policy:?}: no round had all 32 streams");
            for &&(streams, rows) in &full {
                assert!(rows >= streams, "every stream has its base row: {streams} streams, {rows} rows");
                assert!(rows <= max_rows, "max_rows {max_rows}: a round carried {rows} rows");
            }
            // 256 rows are on offer and rows are nearly free: the budget trims to the cap, not below it, in the rounds where every stream drafts
            let widest = full.iter().map(|s| s.1).max().unwrap();
            assert_eq!(
                widest, max_rows,
                "max_rows {max_rows} {policy:?}: the cap should bind, the widest round had {widest} rows"
            );
            assert!(steps.iter().all(|s| s.1 <= max_rows), "no round of any size is past the cap");
        }
    }
}

#[test]
fn trimming_to_the_cap_never_changes_a_reply() {
    // drafted at 32 streams, whatever the cap, equals each stream decoded alone without drafts
    let log = Rc::new(RefCell::new(Vec::new()));
    for max_rows in [48usize, 64, 128] {
        let r = drafted_equals_serial(&|drafting| engine(max_rows, 40, BudgetPolicy::Margin, drafting, &log), &requests(32))
            .unwrap_or_else(|e| panic!("max_rows {max_rows}: {e}"));
        assert!(r.accepted > 0, "max_rows {max_rows}: no draft was accepted, so the check proves nothing");
    }
}

#[test]
fn the_cap_is_also_the_most_sequences_the_engine_admits() {
    // 70 slots but a 64-row cap: a 65th sequence would have no base row in the next round, so it is not admitted (it waits for a slot)
    let log = Rc::new(RefCell::new(Vec::new()));
    let mut e = engine(64, 70, BudgetPolicy::Margin, true, &log);
    for (i, r) in requests(70).into_iter().enumerate() {
        match e.submit_deferred(r.clone()) {
            Ok(_) => assert!(i < 64, "stream {i} was admitted past the cap"),
            Err(SubmitError::Admit(AdmitError::NoSlot)) => assert!(i >= 64, "stream {i} was refused below the cap"),
            Err(other) => panic!("stream {i}: {other}"),
        }
    }
    assert_eq!(e.prefilling(), 64);
    // everything admitted runs to its end, and the waiting ones are admitted once slots are back
    e.run().unwrap();
    assert_eq!(e.take_outputs().len(), 64);
    for r in requests(70).into_iter().skip(64) {
        e.submit(r).unwrap();
    }
    e.run().unwrap();
    assert_eq!(e.take_outputs().len(), 6);
    // the cap of 128 admits all 70 at once, and its rounds stay inside it
    let log = Rc::new(RefCell::new(Vec::new()));
    let mut e = engine(128, 70, BudgetPolicy::Margin, true, &log);
    for r in requests(70) {
        e.submit_deferred(r).expect("70 streams under a 128-row cap");
    }
    e.run().unwrap();
    assert_eq!(e.take_outputs().len(), 70);
    assert!(log.borrow().iter().all(|s| s.1 <= 128 && s.1 >= s.0), "{:?}", log.borrow().iter().max());
}
