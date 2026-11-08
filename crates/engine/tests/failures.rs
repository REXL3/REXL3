//! Failure handling in the engine (release plan, Phase 1 "never a zombie"): a failed row fails only its own sequence, backend errors
//! are classified (recoverable or fatal) and acted on, prefill slices are scheduled round-robin, and a stream that was budgeted no
//! drafts earns them back. Every test here was run against the code it guards with the guarded line broken, and failed.

use gb_engine::mock::{ChainDrafter, Fault, Faults, MockBackend};
use gb_engine::{
    Backend, BackendError, Drafter, Engine, EngineConfig, FinishReason, PrefillStep, ProposeRequest, Request, RowOutput,
    RowRequest, ServeEngine, StepError,
};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};

const VOCAB: usize = 512;
const SLOTS: usize = 8;

fn admission() -> Admission {
    let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 65536, state_bytes_per_slot: 150 << 20, activation_reserve: 0 };
    Admission::new(64 << 30, 0, mm, SLOTS).unwrap()
}

fn cost() -> CostModel {
    CostModel::from_table(vec![(1, 70.0), (16, 74.0), (32, 110.0), (64, 190.0)])
}

fn chain() -> Box<dyn Fn() -> Box<dyn Drafter>> {
    Box::new(|| Box::new(ChainDrafter { vocab: VOCAB, corrupt_every: 5 }) as Box<dyn Drafter>)
}

fn engine_on<B: Backend>(backend: B, drafting: bool) -> Engine<B> {
    Engine::new(backend, admission(), EngineConfig { max_rows: 64, max_drafts: 7, cost: cost(), drafting }, chain())
}

fn request(i: usize, first: u32, max_new: usize) -> Request {
    let mut prompt: Vec<u32> = (0..(6 + i * 3)).map(|j| ((i * 131 + j * 17 + 3) % VOCAB) as u32).collect();
    prompt[0] = first;
    Request {
        prompt,
        sampling: Sampling { seed: 700 + i as u64, temperature: [0.0, 0.7, 1.0][i % 3], top_k: 20, top_p: 0.95, min_p: 0.0 },
        max_new,
        stop: vec![],
        draft: true,
        images: vec![],
        cache: true,
    }
}

/// The tokens `r` produces alone on a clean engine (the reference every concurrent run must equal).
fn solo(r: &Request) -> Vec<u32> {
    solo_with(r, false)
}

fn solo_deep(r: &Request) -> Vec<u32> {
    solo_with(r, true)
}

fn solo_with(r: &Request, deep: bool) -> Vec<u32> {
    let backend = MockBackend::new(VOCAB, SLOTS, Fault::None);
    let mut e = engine_on(if deep { backend.with_deep_state() } else { backend }, false);
    e.submit(r.clone()).unwrap();
    e.run().unwrap();
    e.take_outputs().remove(0).tokens
}

// ------------------------------------------------------------------------------------------------ (2) a failed row fails alone

#[test]
fn one_poisoned_sequence_among_four_fails_alone_and_the_other_three_are_exact() {
    const POISON: u32 = 7;
    // request 2 starts with the poison token: its rows come back NaN from every step; the others never start with it
    let reqs: Vec<Request> = (0..4).map(|i| request(i, if i == 2 { POISON } else { 100 + i as u32 }, 60 + i * 11)).collect();
    let backend = MockBackend::new(VOCAB, SLOTS, Fault::None)
        .with_deep_state()
        .with_faults(Faults { poison_first_token: Some(POISON), ..Default::default() });
    let mut e = engine_on(backend, true);
    let free = e.admission().available();
    let ids: Vec<u64> = reqs.iter().map(|r| e.submit_deferred(r.clone()).unwrap()).collect();
    let mut steps = 0;
    while e.busy() {
        e.step().expect("a poisoned row is not a step error: the engine must keep stepping");
        steps += 1;
        assert!(steps < 10_000);
    }
    let failed = e.take_failed();
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert_eq!(failed[0].0, ids[2]);
    assert!(failed[0].1.contains("NaN"), "{failed:?}");
    let outs = e.take_outputs();
    assert_eq!(outs.len(), 3, "the other three finished");
    for (i, id) in ids.iter().enumerate().filter(|(i, _)| *i != 2) {
        let o = outs.iter().find(|o| o.id == *id).unwrap_or_else(|| panic!("sequence {i} has no output"));
        assert_eq!(o.finish, Some(FinishReason::Length));
        assert_eq!(
            o.tokens,
            solo_deep(&reqs[i]),
            "sequence {i} must equal its solo reply, bit for bit, despite the neighbour's failure"
        );
    }
    assert_eq!(e.admission().available(), free, "the failed sequence's slot and memory came back with the others'");
    assert_eq!(e.active(), 0);
}

#[test]
fn a_poisoned_prompt_among_queued_ones_does_not_stop_the_queue() {
    // the same through the sliced prefill path: the poisoned sequence fails when it first decodes, the queued ones behind it go on
    const POISON: u32 = 9;
    let reqs: Vec<Request> = (0..3).map(|i| request(i, if i == 0 { POISON } else { 50 + i as u32 }, 40)).collect();
    let backend = MockBackend::new(VOCAB, SLOTS, Fault::None)
        .with_prefill_slice(2)
        .with_faults(Faults { poison_first_token: Some(POISON), ..Default::default() });
    let mut e = engine_on(backend, false);
    let ids: Vec<u64> = reqs.iter().map(|r| e.submit_deferred(r.clone()).unwrap()).collect();
    while e.busy() {
        e.step().unwrap();
    }
    assert_eq!(e.take_failed().iter().map(|f| f.0).collect::<Vec<_>>(), vec![ids[0]]);
    let outs = e.take_outputs();
    assert_eq!(outs.iter().map(|o| o.id).collect::<Vec<_>>(), vec![ids[1], ids[2]]);
}

// ------------------------------------------------------------------------------------------------ (1) classified backend errors

#[test]
fn a_fatal_backend_error_is_a_fatal_step_error_that_sticks() {
    let backend =
        MockBackend::new(VOCAB, SLOTS, Fault::None).with_faults(Faults { fatal_at_step: Some(2), ..Default::default() });
    let mut e = engine_on(backend, false);
    let free = e.admission().available();
    e.submit(request(0, 11, 50)).unwrap();
    e.submit(request(1, 12, 50)).unwrap();
    assert_eq!(e.step(), Ok(true)); // step 1 is fine
    let err = e.step().unwrap_err();
    assert!(err.is_fatal() && matches!(&err, StepError::Fatal(m) if m.contains("injected fatal")), "{err}");
    assert_eq!(e.active(), 2, "bookkeeping is untouched: the caller decides what to do with the sequences");
    assert!(e.step().unwrap_err().is_fatal(), "a fatal error is sticky: no later step may pass");
    assert_eq!(e.abort_all(), 2);
    assert_eq!(e.admission().available(), free);
}

#[test]
fn a_recoverable_step_error_fails_the_running_sequences_and_the_next_step_can_succeed() {
    let backend = MockBackend::new(VOCAB, SLOTS, Fault::None).with_faults(Faults {
        retryable_from_step: Some(2),
        retryable_steps: 1,
        ..Default::default()
    });
    let mut e = engine_on(backend, false);
    let free = e.admission().available();
    e.submit(request(0, 11, 50)).unwrap();
    e.submit(request(1, 12, 50)).unwrap();
    e.step().unwrap();
    let err = e.step().unwrap_err();
    assert!(!err.is_fatal() && matches!(&err, StepError::Backend(m) if m.contains("injected recoverable")), "{err}");
    assert_eq!(e.active(), 2, "an error leaves the engine's sequences as they were");
    assert_eq!(e.fail_active("step failed"), 2);
    assert_eq!(e.take_failed().len(), 2);
    assert_eq!(e.admission().available(), free, "both sequences gave their slots back");
    // the device is sound: a new request is served normally and exactly
    let r = request(2, 13, 40);
    e.submit(r.clone()).unwrap();
    e.run().unwrap();
    assert_eq!(e.take_outputs().remove(0).tokens, solo(&r));
}

/// A backend that reports an error from one chosen kind of call, to check what the engine does with each.
struct Flaky {
    inner: MockBackend,
    /// Report `error` after the Nth call of this kind (1-based).
    on: &'static str,
    nth: u32,
    calls: u32,
    error: Option<BackendError>,
    pending: Option<BackendError>,
}

impl Flaky {
    fn new(on: &'static str, nth: u32, error: BackendError) -> Self {
        Self { inner: MockBackend::new(VOCAB, SLOTS, Fault::None), on, nth, calls: 0, error: Some(error), pending: None }
    }
    fn tick(&mut self, kind: &str) {
        if kind == self.on {
            self.calls += 1;
            if self.calls == self.nth {
                self.pending = self.error.take();
            }
        }
    }
}

impl Backend for Flaky {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn take_error(&mut self) -> Option<BackendError> {
        self.pending.take()
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
        self.tick("prefill");
        self.inner.prefill_slice(slot, prompt, images, done, candidates)
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.inner.step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.tick("commit");
        self.inner.commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.inner.release(slot)
    }
    fn propose_batch(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Vec<u32>>> {
        self.tick("propose");
        // like the CUDA backend after a failed draft: nothing proposed
        if self.pending.is_some() {
            return vec![None; reqs.len()];
        }
        self.inner.propose_batch(reqs)
    }
}

#[test]
fn a_recoverable_prefill_error_fails_only_that_prompt() {
    let mut e = engine_on(Flaky::new("prefill", 1, BackendError::retryable("prefill ran out of workspace")), false);
    let free = e.admission().available();
    let (a, b) = (request(0, 21, 30), request(1, 22, 30));
    let ia = e.submit_deferred(a.clone()).unwrap();
    let ib = e.submit_deferred(b.clone()).unwrap();
    while e.busy() {
        e.step().unwrap();
    }
    let failed = e.take_failed();
    assert_eq!(failed, vec![(ia, "prefill ran out of workspace".to_string())], "the first prompt's slice hit the error");
    let outs = e.take_outputs();
    assert_eq!(outs.len(), 1);
    assert_eq!((outs[0].id, outs[0].tokens.clone()), (ib, solo(&b)));
    assert_eq!(e.admission().available(), free);
}

#[test]
fn a_fatal_prefill_error_stops_the_engine_and_keeps_the_prompt_queued() {
    let mut e = engine_on(Flaky::new("prefill", 1, BackendError::fatal("illegal address")), false);
    e.submit_deferred(request(0, 21, 30)).unwrap();
    let err = e.step().unwrap_err();
    assert_eq!(err, StepError::Fatal("illegal address".into()));
    assert_eq!(e.prefilling(), 1, "nothing was silently dropped");
}

#[test]
fn a_recoverable_commit_error_fails_only_the_sequence_whose_state_it_spoiled() {
    // the first commit of the first step belongs to the first active sequence (request order)
    let mut e = engine_on(Flaky::new("commit", 1, BackendError::retryable("commit replay failed")), false);
    let free = e.admission().available();
    let (a, b) = (request(0, 31, 30), request(1, 32, 30));
    let ia = e.submit(a).unwrap();
    e.submit(b.clone()).unwrap();
    while e.busy() {
        e.step().unwrap();
    }
    assert_eq!(e.take_failed(), vec![(ia, "commit replay failed".to_string())]);
    let outs = e.take_outputs();
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].tokens, solo(&b), "the neighbour's reply is untouched");
    assert_eq!(e.admission().available(), free);
}

#[test]
fn a_fatal_commit_error_is_fatal_and_loses_no_sequence() {
    let mut e = engine_on(Flaky::new("commit", 1, BackendError::fatal("state commit failed")), false);
    e.submit(request(0, 31, 30)).unwrap();
    e.submit(request(1, 32, 30)).unwrap();
    let err = e.step().unwrap_err();
    assert_eq!(err, StepError::Fatal("state commit failed".into()));
    assert_eq!(e.active(), 2, "every sequence is still held, for the caller to report");
}

#[test]
fn a_recoverable_drafting_error_only_costs_the_round_its_native_drafts() {
    // drafts never change what is emitted: the tokens equal a clean run's, with the sequence's own drafter filling in
    let r = request(1, 41, 60);
    let mut e = engine_on(Flaky::new("propose", 2, BackendError::retryable("draft kernel failed")), true);
    e.submit(r.clone()).unwrap();
    e.run().unwrap();
    assert!(e.take_failed().is_empty());
    assert_eq!(e.take_outputs().remove(0).tokens, solo(&r));
}

#[test]
fn a_fatal_drafting_error_stops_the_engine() {
    let mut e = engine_on(Flaky::new("propose", 1, BackendError::fatal("draft context corrupted")), true);
    e.submit(request(1, 41, 60)).unwrap();
    assert_eq!(e.step().unwrap_err(), StepError::Fatal("draft context corrupted".into()));
}

// ------------------------------------------------------------------------------------------------ (6) round-robin prefill

fn sliced(slice: usize) -> Engine<MockBackend> {
    engine_on(MockBackend::new(VOCAB, SLOTS, Fault::None).with_deep_state().with_prefill_slice(slice), false)
}

#[test]
fn a_short_prompt_behind_a_long_one_gets_its_first_token_within_a_few_slices() {
    let long = Request { prompt: (0..40).map(|i| (i * 7 + 1) as u32 % VOCAB as u32).collect(), max_new: 6, ..request(0, 5, 6) };
    let short = Request { prompt: vec![3, 4, 5], max_new: 6, ..request(1, 3, 6) };
    let mut e = sliced(4);
    let il = e.submit_deferred(long.clone()).unwrap();
    let is = e.submit_deferred(short.clone()).unwrap();
    let (mut first_short, mut first_long) = (None, None);
    for step in 1..=30 {
        e.step().unwrap();
        for ev in e.take_events() {
            if ev.id == is && first_short.is_none() {
                first_short = Some(step);
            }
            if ev.id == il && first_long.is_none() {
                first_long = Some(step);
            }
        }
    }
    // the long prompt is 10 slices of 4: it used to hold the queue for all of them (first token for the short one at step 11)
    assert!(first_short.unwrap() <= 3, "the short prompt's first token came at step {first_short:?}");
    assert!(first_long.unwrap() >= 10, "the long prompt still takes its 10 slices (step {first_long:?})");
    while e.busy() {
        e.step().unwrap();
    }
    // replies are what whole-prompt prefill gives
    let mut once = sliced(1 << 20);
    once.submit(long).unwrap();
    once.submit(short).unwrap();
    once.run().unwrap();
    let want: Vec<Vec<u32>> = once.take_outputs().into_iter().map(|o| o.tokens).collect();
    let got: Vec<Vec<u32>> = e.take_outputs().into_iter().map(|o| o.tokens).collect();
    assert_eq!(got, want);
}

#[test]
fn round_robin_prefill_is_bit_preserving_for_many_interleavings() {
    let reqs: Vec<Request> = (0..5)
        .map(|i| Request {
            prompt: (0..(7 + i * 11)).map(|j| ((i * 29 + j * 3 + 1) % VOCAB) as u32).collect(),
            ..request(i, 60, 45)
        })
        .collect();
    let mut once = sliced(1 << 20);
    for r in &reqs {
        once.submit(r.clone()).unwrap();
    }
    once.run().unwrap();
    let want: Vec<Vec<u32>> = once.take_outputs().into_iter().map(|o| o.tokens).collect();
    for slice in [1usize, 2, 5, 13] {
        let mut e = sliced(slice);
        for r in &reqs {
            e.submit_deferred(r.clone()).unwrap();
        }
        while e.busy() {
            e.step().unwrap();
        }
        let got: Vec<Vec<u32>> = e.take_outputs().into_iter().map(|o| o.tokens).collect();
        assert_eq!(got, want, "slice {slice}");
    }
}

// ------------------------------------------------------------------------------------------------ (8) the draft estimate recovers

/// Proposes wrong tokens for its first `bad` calls, then follows the mock's own argmax chain (every draft right).
struct Staged {
    calls: usize,
    bad: usize,
}

impl Drafter for Staged {
    fn propose(&mut self, tokens: &[u32], s: &Sampling, max: usize) -> Vec<u32> {
        self.calls += 1;
        let mut good = ChainDrafter { vocab: VOCAB, corrupt_every: u64::MAX };
        let mut out = good.propose(tokens, s, max);
        if self.calls <= self.bad {
            for t in out.iter_mut() {
                *t = (*t + 1) % VOCAB as u32;
            }
        }
        out
    }
}

#[test]
fn a_stream_budgeted_no_drafts_earns_them_back() {
    // one extra row costs 30% of a step, so a draft only pays when the stream's acceptance estimate is above about 0.3
    let cost = CostModel::from_table(vec![(1, 10.0), (2, 13.0), (8, 40.0), (64, 200.0)]);
    let make: Box<dyn Fn() -> Box<dyn Drafter>> = Box::new(|| Box::new(Staged { calls: 0, bad: 6 }) as Box<dyn Drafter>);
    let mut e = Engine::new(
        MockBackend::new(VOCAB, SLOTS, Fault::None),
        admission(),
        EngineConfig { max_rows: 64, max_drafts: 7, cost, drafting: true },
        make,
    );
    // greedy: the mock's chain is deterministic, so the good drafter is right every time
    let r = Request { sampling: Sampling { temperature: 0.0, ..request(0, 8, 200).sampling }, ..request(0, 8, 200) };
    e.submit(r).unwrap();
    e.run().unwrap();
    let o = e.take_outputs().remove(0);
    // rounds 1-4: drafts allocated, all rejected, the estimate falls under 0.3; rounds 5-9: budgeted nothing (the drafter is still
    // asked and still wrong for two of those); then the estimate has drifted back over 0.3 and the now-good drafts are allocated.
    // Before the fix the estimate stayed under 0.3 for good: drafted stopped at the 4 early drafts and accepted stayed 0.
    assert!(o.accepted > 100, "drafts were never earned back: drafted {} accepted {} rounds {}", o.drafted, o.accepted, o.rounds);
    assert!(o.tokens_per_round() > 3.0, "tokens per round {}", o.tokens_per_round());
}

// ------------------------------------------------------------------------------------------------ the erased engine fails like the generic one

fn deep(faults: Faults) -> MockBackend {
    MockBackend::new(VOCAB, SLOTS, Fault::None).with_deep_state().with_faults(faults)
}

/// Everything the serving loop can observe of an engine over four failure scenarios, as text: a poisoned row among healthy ones, a
/// fatal backend error, a recoverable step error with the engine going on, and what `validate` and `would_admit` say. Expanded once
/// on an `Engine<MockBackend>` (its own methods) and once on a `Box<dyn ServeEngine>` (the trait's, the shape every family's `build`
/// hands the runtime), so a method the trait forwards wrongly, or leaves out, shows as a difference.
macro_rules! failure_transcript {
    ($build:expr) => {{
        let build = $build;
        let mut log: Vec<String> = Vec::new();

        // 1. a row that cannot be sampled fails only its own sequence (request 2 starts with the poison token)
        {
            const POISON: u32 = 7;
            let reqs: Vec<Request> =
                (0..4).map(|i| request(i, if i == 2 { POISON } else { 100 + i as u32 }, 60 + i * 11)).collect();
            let mut e = build(Faults { poison_first_token: Some(POISON), ..Default::default() }, true);
            let ids: Vec<u64> = reqs.iter().map(|r| e.submit_deferred(r.clone()).unwrap()).collect();
            let mut events: std::collections::BTreeMap<u64, Vec<u32>> = ids.iter().map(|&i| (i, vec![])).collect();
            while e.busy() {
                let r = e.step();
                log.push(format!("poison step {r:?} active={} prefilling={}", e.active(), e.prefilling()));
                for ev in e.take_events() {
                    events.get_mut(&ev.id).unwrap().extend(ev.tokens);
                }
            }
            log.push(format!("poison failed {:?}", e.take_failed()));
            for o in e.take_outputs() {
                log.push(format!(
                    "poison output {} {:?} {:?} events_equal={}",
                    o.id,
                    o.finish,
                    o.tokens,
                    events[&o.id] == o.tokens
                ));
            }
            log.push(format!("poison end active={} busy={}", e.active(), e.busy()));
        }

        // 2. a fatal backend error: the step says Fatal, and keeps saying it; the sequences are still the caller's to answer
        {
            let mut e = build(Faults { fatal_at_step: Some(2), ..Default::default() }, false);
            for i in 0..2 {
                e.submit_deferred(request(i, 11 + i as u32, 50)).unwrap();
            }
            // (a deferred prefill advances in slices inside the first steps: run until the error shows, then once more)
            let mut errors = 0;
            for _ in 0..16 {
                let r = e.step();
                log.push(format!("fatal step {r:?} active={}", e.active()));
                errors += r.is_err() as u32;
                if errors == 2 {
                    break;
                }
            }
            log.push(format!("fatal abort_all={} busy={}", e.abort_all(), e.busy()));
        }

        // 3. a recoverable step error: fail_active answers the running sequences, then the engine serves a new request normally
        {
            let mut e = build(Faults { retryable_from_step: Some(2), retryable_steps: 1, ..Default::default() }, false);
            let ids: Vec<u64> = (0..2).map(|i| e.submit_deferred(request(i, 11 + i as u32, 50)).unwrap()).collect();
            for _ in 0..16 {
                let r = e.step();
                log.push(format!("retry step {r:?} active={}", e.active()));
                if r.is_err() {
                    break;
                }
            }
            log.push(format!("retry fail_active={}", e.fail_active("step failed")));
            log.push(format!("retry failed {:?} (ids {ids:?})", e.take_failed()));
            let id = e.submit_deferred(request(2, 13, 40)).unwrap();
            while e.busy() {
                log.push(format!("retry later step {:?}", e.step()));
            }
            for o in e.take_outputs() {
                log.push(format!("retry output {} {:?} {:?}", o.id == id, o.finish, o.tokens));
            }
        }

        // 4. what an idle engine, a full one and a hopeless request answer without reserving anything
        {
            let mut e = build(Faults::default(), false);
            let normal = request(0, 5, 20);
            let hopeless = request(1, 5, 2_000_000);
            let empty = Request { prompt: vec![], ..request(2, 5, 20) };
            for (name, r) in [("normal", &normal), ("hopeless", &hopeless), ("empty", &empty)] {
                log.push(format!("admission idle {name}: validate={:?} would_admit={:?}", e.validate(r), e.would_admit(r)));
            }
            for i in 0..SLOTS {
                e.submit_deferred(request(i, 20 + i as u32, 30)).unwrap();
            }
            log.push(format!("admission full: would_admit={:?} busy={}", e.would_admit(&normal), e.busy()));
        }
        log
    }};
}

#[test]
fn the_erased_engine_fails_exactly_like_the_generic_one() {
    let generic = failure_transcript!(|f: Faults, drafting: bool| engine_on(deep(f), drafting));
    let erased = failure_transcript!(|f: Faults, drafting: bool| Box::new(engine_on(deep(f), drafting)) as Box<dyn ServeEngine>);
    if let Some(i) = (0..generic.len().max(erased.len())).find(|&i| generic.get(i) != erased.get(i)) {
        panic!(
            "Engine<B> and Box<dyn ServeEngine> must be the same engine under failure; the first difference is line {i}:\n  generic: {:?}\n  erased:  {:?}",
            generic.get(i),
            erased.get(i)
        );
    }
    // the agreement is about the failure behaviour, not two empty logs
    let text = generic.join("\n");
    assert_eq!(text.matches("poison output").count(), 3, "the three healthy sequences finished:\n{text}");
    assert!(text.contains("poison failed [(") && text.contains("NaN"), "the poisoned one failed with its reason:\n{text}");
    assert!(!text.contains("poison step Err"), "a poisoned row is not a step error:\n{text}");
    assert_eq!(text.matches("fatal step Err(Fatal(").count(), 2, "the fatal error sticks:\n{text}");
    assert!(text.contains("fatal abort_all=2"), "{text}");
    assert!(text.contains("retry step Err(Backend(") && text.contains("retry fail_active=2"), "{text}");
    assert!(text.contains("NeverFits") && text.contains("NoSlot"), "{text}");
}
