//! The harness against the CPU mock: the contract holds when the backend honours it, and each check
//! fails when a fault is injected. On the GB10 the same checks run against the CUDA backend.

use gb_engine::harness::{
    chunked_prefill_equals_whole, drafted_equals_serial, rollback_equals_replay, row_invariance, serial_batched_equals_solo,
};
use gb_engine::mock::{ChainDrafter, Fault, MockBackend};
use gb_engine::{Backend, Drafter, Engine, EngineConfig, FinishReason, Request, RowOutput, RowRequest, StepError, SubmitError};
use gb_sampling::{Sampling, SamplingError};
use gb_sched::{Admission, AdmitError, CostModel, ModelMemory};
use gb_spec::CopyDrafter;

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

fn cost() -> CostModel {
    // a synthetic curve with a knee at 16 rows; replace with measured points on the GB10
    CostModel::from_table(vec![(1, 70.0), (16, 74.0), (32, 110.0), (64, 190.0)])
}

fn make_engine(fault: Fault, drafter: &'static str) -> impl Fn(bool) -> Engine<MockBackend> {
    make_engine_with(fault, drafter, false)
}

fn make_engine_with(fault: Fault, drafter: &'static str, deep: bool) -> impl Fn(bool) -> Engine<MockBackend> {
    move |drafting| {
        let make: Box<dyn Fn() -> Box<dyn Drafter>> = match drafter {
            "copy" => Box::new(|| Box::new(CopyDrafter::new(8, 7)) as Box<dyn Drafter>),
            _ => Box::new(|| Box::new(ChainDrafter { vocab: VOCAB, corrupt_every: 5 }) as Box<dyn Drafter>),
        };
        let backend = MockBackend::new(VOCAB, SLOTS, fault);
        let backend = if deep { backend.with_deep_state() } else { backend };
        Engine::new(backend, admission(), EngineConfig { max_rows: 64, max_drafts: 7, cost: cost(), drafting }, make)
    }
}

fn requests(n: usize) -> Vec<Request> {
    let temps = [0.0, 0.6, 1.0];
    (0..n)
        .map(|i| {
            let prompt: Vec<u32> = (0..(5 + i * 3)).map(|j| ((i * 131 + j * 17) % VOCAB) as u32).collect();
            Request {
                sampling: Sampling { seed: 1234 + i as u64, temperature: temps[i % 3], top_k: 20, top_p: 0.95, min_p: 0.0 },
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

#[test]
fn rows_are_invariant() {
    let cases: Vec<(Vec<u32>, Vec<u32>)> = (0..6)
        .map(|i| ((0..(4 + i)).map(|j| (j * 7 + i) as u32).collect(), (0..(1 + i)).map(|j| (j * 3 + 1) as u32).collect()))
        .collect();
    row_invariance(&|| MockBackend::new(VOCAB, SLOTS, Fault::None), &cases, 28).unwrap();
}

#[test]
fn row_invariance_catches_window_dependent_kernel() {
    let cases = vec![(vec![1, 2, 3], vec![4, 5, 6])];
    let err = row_invariance(&|| MockBackend::new(VOCAB, SLOTS, Fault::WindowDependent), &cases, 28).unwrap_err();
    assert!(err.contains("differs from its one-row step"), "wrong failure: {err}");
}

#[test]
fn concurrent_serial_equals_solo() {
    let r = serial_batched_equals_solo(&make_engine(Fault::None, "chain"), &requests(12)).unwrap();
    assert_eq!(r.sequences, 12);
}

#[test]
fn drafted_equals_serial_with_model_like_drafter() {
    let r = drafted_equals_serial(&make_engine(Fault::None, "chain"), &requests(12)).unwrap();
    assert!(r.accepted > 0, "the drafter never got a token accepted; the test would be vacuous");
    assert!(r.tokens_per_round() > 1.5, "tokens/round {}", r.tokens_per_round());
    eprintln!("chain drafter: {:.2} tokens/round over {} rounds", r.tokens_per_round(), r.rounds);
}

#[test]
fn drafted_equals_serial_with_copy_drafter() {
    // repetitive prompts so copy windows fire
    let reqs: Vec<Request> = (0..6)
        .map(|i| {
            let unit: Vec<u32> = (0..12).map(|j| ((i * 13 + j * 5) % VOCAB) as u32).collect();
            Request {
                prompt: unit.repeat(4),
                sampling: Sampling { seed: 7 + i as u64, temperature: 0.0, ..Default::default() },
                max_new: 80,
                stop: vec![],
                draft: true,
                images: vec![],
                cache: true,
            }
        })
        .collect();
    let r = drafted_equals_serial(&make_engine(Fault::None, "copy"), &reqs).unwrap();
    assert!(r.accepted > 0, "no copy draft was accepted; the test would be vacuous");
    eprintln!("copy drafter: {:.2} tokens/round, accepted {} of {} drafted", r.tokens_per_round(), r.accepted, r.drafted);
}

#[test]
fn drafted_mismatch_is_caught() {
    let err = drafted_equals_serial(&make_engine(Fault::WindowDependent, "chain"), &requests(6)).unwrap_err();
    assert!(err.contains("diverges at generated token"), "failed for the wrong reason: {err}");
}

#[test]
fn stop_tokens_truncate_identically() {
    let mut reqs = requests(8);
    for r in reqs.iter_mut() {
        r.stop = vec![3, 17, 101];
        r.max_new = 200;
    }
    let r = drafted_equals_serial(&make_engine(Fault::None, "chain"), &reqs).unwrap();
    assert!(r.stopped > 0, "no sequence ended on a stop token, so truncation was never exercised");
    assert!(r.stopped_in_draft > 0, "no stop token landed inside a draft run; accepted={} drafted={}", r.accepted, r.drafted);
    eprintln!("{} of {} stopped, {} inside a draft run", r.stopped, r.sequences, r.stopped_in_draft);
}

#[test]
fn chunked_prefill_matches_whole() {
    let prompt: Vec<u32> = (0..100).map(|j| (j * 37 % VOCAB) as u32).collect();
    let s = Sampling { seed: 5, ..Default::default() };
    let ok = || MockBackend::new(VOCAB, SLOTS, Fault::None);
    chunked_prefill_equals_whole(&ok, &prompt, &[16, 64], &s, 20).unwrap();
    let bad = || MockBackend::new(VOCAB, SLOTS, Fault::ChunkDependent);
    let err = chunked_prefill_equals_whole(&bad, &prompt, &[16, 64], &s, 20).unwrap_err();
    assert!(err.contains("last prefill row differs"), "failed for the wrong reason: {err}");
}

/// Logs the `total` of every prefill piece it is given.
struct TotalLog {
    inner: MockBackend,
    totals: std::rc::Rc<std::cell::RefCell<Vec<usize>>>,
}

impl Backend for TotalLog {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.inner.prefill(slot, start, tokens, candidates)
    }
    fn prefill_piece(&mut self, slot: usize, start: u64, tokens: &[u32], total: usize, candidates: usize) -> RowOutput {
        self.totals.borrow_mut().push(total);
        self.inner.prefill(slot, start, tokens, candidates)
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
}

/// A backend whose prefill arithmetic depends on the length of the prompt (the Qwen backend picks its path by it) can only be
/// checked if every piece the harness feeds it carries the whole prompt's length, not the piece's.
#[test]
fn chunked_prefill_hands_every_piece_the_prompt_total() {
    let prompt: Vec<u32> = (0..100).map(|j| (j * 37 % VOCAB) as u32).collect();
    let s = Sampling { seed: 5, ..Default::default() };
    let totals = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let make = || TotalLog { inner: MockBackend::new(VOCAB, SLOTS, Fault::None), totals: totals.clone() };
    chunked_prefill_equals_whole(&make, &prompt, &[16, 64], &s, 4).unwrap();
    // the whole run is one piece, the cut run three
    assert_eq!(*totals.borrow(), vec![100; 4]);
}

#[test]
fn admission_refuses_when_full() {
    let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1 << 20, state_bytes_per_slot: 0, activation_reserve: 0 };
    let mut e = Engine::new(
        MockBackend::new(VOCAB, SLOTS, Fault::None),
        Admission::new(100 << 20, 0, mm, SLOTS).unwrap(),
        EngineConfig { max_rows: 64, max_drafts: 7, cost: cost(), drafting: false },
        Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn Drafter>),
    );
    let r = |n| Request {
        prompt: vec![1, 2, 3],
        sampling: Sampling::default(),
        max_new: n,
        stop: vec![],
        draft: true,
        images: vec![],
        cache: true,
    };
    assert!(e.submit(r(60)).is_ok()); // 63 MiB
                                      // would need 126 MiB of the 37 left: refused for memory, not for some other reason
    assert!(matches!(e.submit(r(60)), Err(SubmitError::Admit(AdmitError::NoMemory { .. }))));
}

// ---------------------------------------------------------------- deep recurrent state and rollback

#[test]
fn drafted_equals_serial_when_logits_depend_on_recurrent_state() {
    // a quarter of positions are decided by the persisted state, so drafts get rejected and rollbacks happen
    let r = drafted_equals_serial(&make_engine_with(Fault::None, "chain", true), &requests(12)).unwrap();
    assert!(r.accepted > 0 && r.accepted < r.drafted, "need both accepted and rejected drafts: {r:?}");
}

#[test]
fn rollback_leak_is_caught() {
    let err = drafted_equals_serial(&make_engine_with(Fault::RollbackLeak, "chain", true), &requests(12)).unwrap_err();
    assert!(err.contains("diverges at generated token"), "failed for the wrong reason: {err}");
}

#[test]
fn rollback_matches_a_clean_replay_at_every_keep() {
    let ctx: Vec<u32> = (0..9).map(|j| (j * 11 + 2) as u32).collect();
    let window: Vec<u32> = vec![40, 41, 42, 43, 44, 45];
    let next = vec![77u32, 78];
    let ok = || MockBackend::new(VOCAB, SLOTS, Fault::None).with_deep_state();
    for keep in 1..=window.len() {
        rollback_equals_replay(&ok, &ctx, &window, keep, &next, 28).unwrap_or_else(|e| panic!("keep {keep}: {e}"));
    }
    // a leak only shows when rows were rejected, and the window must be wide enough for its state to matter
    let leaky = || MockBackend::new(VOCAB, SLOTS, Fault::RollbackLeak).with_deep_state();
    let mut caught = 0;
    for keep in 1..window.len() {
        if let Err(e) = rollback_equals_replay(&leaky, &ctx, &window, keep, &next, 28) {
            assert!(e.contains("differs from a clean replay"), "{e}");
            caught += 1;
        }
    }
    assert!(caught > 0, "the leaky backend was never caught: the check is vacuous");
    // keeping the whole window is not a rollback, so even the leaky backend passes it
    rollback_equals_replay(&leaky, &ctx, &window, window.len(), &next, 28).unwrap();
}

// ---------------------------------------------------------------- request validation and failure paths

fn plain_engine() -> Engine<MockBackend> {
    make_engine(Fault::None, "chain")(false)
}

fn ok_request() -> Request {
    Request {
        prompt: vec![1, 2, 3],
        sampling: Sampling::default(),
        max_new: 8,
        stop: vec![],
        draft: true,
        images: vec![],
        cache: true,
    }
}

#[test]
fn malformed_requests_are_errors_and_reserve_nothing() {
    let mut e = plain_engine();
    let free = e.admission().available();
    let bad = [
        Request { prompt: vec![], ..ok_request() },
        Request { max_new: 0, ..ok_request() },
        Request { sampling: Sampling { top_k: 0, ..Default::default() }, ..ok_request() },
        Request { sampling: Sampling { min_p: 1.5, ..Default::default() }, ..ok_request() },
        Request { sampling: Sampling { temperature: f64::NAN, ..Default::default() }, ..ok_request() },
        Request { prompt: vec![1, VOCAB as u32], ..ok_request() },
    ];
    for r in bad {
        assert!(matches!(e.submit(r.clone()), Err(SubmitError::Invalid(_))), "{r:?}");
    }
    // a size that overflows is refused, not wrapped into a small cost
    let huge = Request { max_new: usize::MAX, ..ok_request() };
    assert!(matches!(e.submit(huge), Err(SubmitError::Admit(AdmitError::TooLarge))));
    assert_eq!(e.admission().available(), free);
    assert_eq!(e.active(), 0);
    assert!(e.submit(ok_request()).is_ok(), "the engine still serves after refusing bad requests");
}

#[test]
fn context_limit_is_enforced() {
    let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 0, activation_reserve: 0 };
    let mut e = Engine::new(
        MockBackend::new(VOCAB, SLOTS, Fault::None),
        Admission::new(1 << 30, 0, mm, SLOTS).unwrap().with_max_context(100),
        EngineConfig { max_rows: 64, max_drafts: 7, cost: cost(), drafting: false },
        Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn Drafter>),
    );
    assert!(matches!(
        e.submit(Request { max_new: 98, ..ok_request() }),
        Err(SubmitError::Admit(AdmitError::ContextTooLong { tokens: 101, max: 100 }))
    ));
    assert!(e.submit(Request { max_new: 97, ..ok_request() }).is_ok());
}

/// A backend that misbehaves on demand: NaN logits, or fewer outputs than requests.
struct Broken {
    inner: MockBackend,
    nan_prefill: bool,
    drop_last_output: bool,
    nan_step: bool,
    released: Vec<usize>,
}

impl Broken {
    fn new() -> Self {
        Self {
            inner: MockBackend::new(VOCAB, SLOTS, Fault::None),
            nan_prefill: false,
            drop_last_output: false,
            nan_step: false,
            released: vec![],
        }
    }
}

impl Backend for Broken {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        let mut r = self.inner.prefill(slot, start, tokens, candidates);
        if self.nan_prefill {
            r.values[0] = f32::NAN;
        }
        r
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        let mut o = self.inner.step(reqs, candidates);
        if self.drop_last_output {
            o.pop();
        }
        if self.nan_step {
            o[0][0].values[0] = f32::NAN;
        }
        o
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.inner.commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.released.push(slot);
        self.inner.release(slot)
    }
}

fn broken_engine(b: Broken) -> Engine<Broken> {
    Engine::new(
        b,
        admission(),
        EngineConfig { max_rows: 64, max_drafts: 7, cost: cost(), drafting: false },
        Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn Drafter>),
    )
}

#[test]
fn a_nan_from_prefill_fails_the_request_and_frees_its_slot() {
    let mut e = broken_engine(Broken { nan_prefill: true, ..Broken::new() });
    let free = e.admission().available();
    assert_eq!(e.submit(ok_request()), Err(SubmitError::Sampling(SamplingError::NanLogit)));
    assert_eq!(e.admission().available(), free, "admission must be given back");
    assert_eq!(e.backend.released, vec![0], "the backend must release the slot it prefilled");
    assert_eq!(e.active(), 0);
}

#[test]
fn a_short_backend_reply_is_an_error_not_a_lost_sequence() {
    let mut e = broken_engine(Broken::new());
    let free = e.admission().available();
    e.submit(ok_request()).unwrap();
    e.submit(Request { prompt: vec![9, 8, 7], ..ok_request() }).unwrap();
    e.backend.drop_last_output = true;
    let err = e.step().unwrap_err();
    assert!(matches!(err, StepError::BackendShape(ref m) if m.contains("1 outputs for 2 requests")), "{err}");
    assert_eq!(e.active(), 2, "an error must not silently drop sequences");
    // after the error the caller can give everything back
    assert_eq!(e.abort_all(), 2);
    assert_eq!(e.admission().available(), free);
    assert_eq!(e.active(), 0);
}

#[test]
fn a_nan_in_a_step_row_fails_only_that_sequence() {
    // (was: the whole step errored and every running sequence had to be given up; one bad row now costs only its own sequence)
    let mut e = broken_engine(Broken::new());
    let free = e.admission().available();
    let first = e.submit(ok_request()).unwrap(); // the one whose row 0 turns NaN (Broken poisons request 0 of the step)
    let second_req = Request { prompt: vec![9, 8, 7], ..ok_request() };
    let second = e.submit(second_req.clone()).unwrap();
    e.backend.nan_step = true;
    assert_eq!(e.step(), Ok(true), "a bad row is not a step error");
    e.backend.nan_step = false;
    let failed = e.take_failed();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0].0, first);
    assert!(failed[0].1.contains("NaN"), "{failed:?}");
    assert_eq!(e.active(), 1, "the other sequence is still running");
    e.run().unwrap();
    let outs = e.take_outputs();
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].id, second);
    assert_eq!(outs[0].finish, Some(FinishReason::Length));
    assert_eq!(outs[0].tokens, drafted_solo(&second_req), "the survivor's reply is untouched by its neighbour's failure");
    assert_eq!(e.admission().available(), free, "the failed sequence's slot and memory came back");
    assert_eq!(e.backend.released.len(), 2, "both slots were released in the backend");
}

// ---------------------------------------------------------------- events, cancellation, backend-native drafting

#[test]
fn events_reassemble_every_output_exactly() {
    let mut e = make_engine(Fault::None, "chain")(true);
    for r in requests(6) {
        e.submit(r).unwrap();
    }
    let mut events = e.take_events(); // first tokens
    while e.step().unwrap() {
        events.extend(e.take_events());
    }
    events.extend(e.take_events());
    let outs = e.take_outputs();
    assert_eq!(outs.len(), 6);
    for o in &outs {
        let mine: Vec<&gb_engine::Event> = events.iter().filter(|ev| ev.id == o.id).collect();
        let joined: Vec<u32> = mine.iter().flat_map(|ev| ev.tokens.iter().copied()).collect();
        assert_eq!(joined, o.tokens, "events of sequence {} must concatenate to its output", o.id);
        assert!(mine.last().unwrap().finish.is_some(), "the last event carries the finish reason");
        assert!(mine[..mine.len() - 1].iter().all(|ev| ev.finish.is_none()), "only the last event finishes");
        assert!(mine.iter().all(|ev| !ev.tokens.is_empty() || ev.finish.is_some()), "no empty progress events");
        assert_eq!(mine.last().unwrap().finish, o.finish);
        assert_eq!(o.finish, Some(gb_engine::FinishReason::Length));
    }
    // a round can commit several tokens at once, so some event must carry more than one
    assert!(events.iter().any(|ev| ev.tokens.len() > 1), "drafting should commit multi-token rounds");
}

#[test]
fn stop_tokens_report_the_stop_reason() {
    let mut e = make_engine(Fault::None, "chain")(true);
    let mut r = requests(1).remove(0);
    r.stop = vec![3, 17, 101];
    r.max_new = 300;
    e.submit(r).unwrap();
    e.run().unwrap();
    let o = e.take_outputs().remove(0);
    assert_eq!(o.finish, Some(gb_engine::FinishReason::Stop));
    assert!(o.stopped && [3, 17, 101].contains(o.tokens.last().unwrap()));
}

#[test]
fn cancel_frees_memory_and_the_rest_keep_running() {
    let mut e = make_engine(Fault::None, "chain")(true);
    let free = e.admission().available();
    let a = e.submit(requests(2).remove(0)).unwrap();
    let b = e.submit(requests(2).remove(1)).unwrap();
    e.step().unwrap();
    let _ = e.take_events();
    assert!(e.cancel(a));
    assert!(!e.cancel(a), "a sequence can be cancelled once");
    assert!(!e.cancel(999));
    let ev = e.take_events();
    assert!(ev.iter().any(|x| x.id == a && x.finish == Some(gb_engine::FinishReason::Cancelled)), "{ev:?}");
    assert_eq!(e.active(), 1);
    e.run().unwrap();
    let outs = e.take_outputs();
    assert_eq!(outs.len(), 2);
    assert_eq!(outs.iter().find(|o| o.id == a).unwrap().finish, Some(gb_engine::FinishReason::Cancelled));
    assert_eq!(outs.iter().find(|o| o.id == b).unwrap().finish, Some(gb_engine::FinishReason::Length));
    assert_eq!(e.admission().available(), free, "everything must be given back");
    // the survivor's tokens are untouched by the neighbour's cancellation
    let solo = drafted_solo(&requests(2)[1]);
    assert_eq!(outs.iter().find(|o| o.id == b).unwrap().tokens, solo);
}

fn drafted_solo(r: &Request) -> Vec<u32> {
    let mut e = make_engine(Fault::None, "chain")(false);
    e.submit(r.clone()).unwrap();
    e.run().unwrap();
    e.take_outputs().remove(0).tokens
}

/// A backend that drafts for itself, as the CUDA backend will (DFlash2 needs device state).
struct Native {
    inner: MockBackend,
    chain: ChainDrafter,
    calls: usize,
}

impl Backend for Native {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.inner.prefill(slot, start, tokens, candidates)
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
    fn propose(&mut self, _slot: usize, tokens: &[u32], s: &Sampling, max: usize) -> Option<Vec<u32>> {
        self.calls += 1;
        Some(self.chain.propose(tokens, s, max))
    }
}

#[test]
fn backend_native_drafts_keep_drafted_equal_to_serial() {
    let make = |drafting: bool| {
        Engine::new(
            Native {
                inner: MockBackend::new(VOCAB, SLOTS, Fault::None).with_deep_state(),
                chain: ChainDrafter { vocab: VOCAB, corrupt_every: 5 },
                calls: 0,
            },
            admission(),
            EngineConfig { max_rows: 64, max_drafts: 7, cost: cost(), drafting },
            // the per-sequence drafter must NOT be consulted when the backend answers
            Box::new(|| Box::new(Panics) as Box<dyn Drafter>),
        )
    };
    let r = drafted_equals_serial(&make, &requests(8)).unwrap();
    assert!(r.accepted > 0 && r.drafted > r.accepted, "{r:?}");
    let mut e = make(true);
    e.submit(requests(1).remove(0)).unwrap();
    e.run().unwrap();
    assert!(e.backend.calls > 0, "the backend's propose was never asked");
    // with drafting off the backend is never asked either
    let mut off = make(false);
    off.submit(requests(1).remove(0)).unwrap();
    off.run().unwrap();
    assert_eq!(off.backend.calls, 0);
}

struct Panics;
impl Drafter for Panics {
    fn propose(&mut self, _: &[u32], _: &Sampling, _: usize) -> Vec<u32> {
        panic!("the sequence's drafter must not be asked when the backend drafts");
    }
}

#[test]
fn image_spans_must_fit_the_prompt_and_reserve_nothing_when_refused() {
    use gb_engine::ImageSpan;
    let span = |start, tokens, merged| ImageSpan { start, tokens, merged, features: std::sync::Arc::new(()), key: 1 };
    let mut e = plain_engine();
    let free = e.admission().available();
    for bad in [
        vec![span(0, 0, (0, 0))],
        vec![span(1, 4, (1, 3))],                     // tokens != rows * columns
        vec![span(2, 4, (2, 2))],                     // runs past the prompt of 3 tokens
        vec![span(0, 2, (1, 2)), span(1, 1, (1, 1))], // overlap
    ] {
        let r = Request { images: bad, ..ok_request() };
        assert!(matches!(e.submit(r), Err(SubmitError::Invalid(_))));
        assert_eq!(e.admission().available(), free, "a refused request must reserve nothing");
    }
    // a well-formed span is accepted; the default backend treats the prompt as text
    let ok = Request { prompt: vec![1, 2, 3, 4, 5], images: vec![span(1, 4, (2, 2))], ..ok_request() };
    assert!(e.submit(ok).is_ok());
}

// ---------------------------------------------------------------------------------------------------------------
// Sliced prefill: a long prompt is prefilled one slice a step, so running sequences keep decoding meanwhile.
// ---------------------------------------------------------------------------------------------------------------

fn sliced_engine(slice: usize) -> Engine<MockBackend> {
    Engine::new(
        MockBackend::new(VOCAB, SLOTS, Fault::None).with_deep_state().with_prefill_slice(slice),
        admission(),
        EngineConfig { max_rows: 64, max_drafts: 7, cost: cost(), drafting: false },
        Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn Drafter>),
    )
}

fn run_out(e: &mut Engine<MockBackend>) -> Vec<Vec<u32>> {
    while e.step().unwrap() {}
    let mut outs = e.take_outputs();
    outs.sort_by_key(|o| o.id);
    outs.into_iter().map(|o| o.tokens).collect()
}

#[test]
fn a_long_prefill_does_not_stall_running_sequences() {
    let mut e = sliced_engine(4);
    let a = Request { prompt: vec![5, 6, 7], max_new: 200, ..ok_request() };
    let ida = e.submit(a).unwrap();
    let long: Vec<u32> = (0..40).map(|i| (i * 7 + 1) as u32 % VOCAB as u32).collect();
    let idb = e.submit_deferred(Request { prompt: long, max_new: 5, ..ok_request() }).unwrap();
    assert_eq!(e.prefilling(), 1, "the long prompt is queued, not prefilled");
    assert_eq!(e.active(), 1);
    let mut a_progress = Vec::new();
    let mut steps_until_b = None;
    for step in 1..=20 {
        e.step().unwrap();
        let evs = e.take_events();
        a_progress.push(evs.iter().filter(|ev| ev.id == ida).map(|ev| ev.tokens.len()).sum::<usize>());
        if steps_until_b.is_none() && evs.iter().any(|ev| ev.id == idb) {
            steps_until_b = Some(step);
        }
    }
    assert_eq!(steps_until_b, Some(10), "40 tokens in slices of 4 is 10 slices, one per step");
    assert!(
        a_progress[..10].iter().all(|&n| n >= 1),
        "the running sequence decoded in every step of B's prefill: {a_progress:?}"
    );
}

#[test]
fn sliced_and_deferred_prefill_give_the_same_replies_as_prefilling_at_once() {
    let reqs: Vec<Request> = (0..4)
        .map(|i| Request {
            prompt: (0..(9 + i * 13)).map(|j| ((i * 31 + j * 5) % VOCAB) as u32).collect(),
            sampling: Sampling { seed: 40 + i as u64, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 },
            max_new: 30 + i * 7,
            ..ok_request()
        })
        .collect();
    let mut once = sliced_engine(1 << 20); // one slice covers any prompt: the old behaviour
    for r in &reqs {
        once.submit(r.clone()).unwrap();
    }
    let want = run_out(&mut once);
    for slice in [1usize, 3, 8] {
        let mut e = sliced_engine(slice);
        for r in &reqs {
            e.submit_deferred(r.clone()).unwrap(); // all queued first: prefills interleave with each other's decoding
        }
        assert_eq!(run_out(&mut e), want, "slice {slice}");
    }
}

#[test]
fn cancelling_a_queued_prefill_gives_everything_back() {
    let mut e = sliced_engine(2);
    let free = e.admission().available();
    let id = e.submit_deferred(Request { prompt: (0..20).collect(), ..ok_request() }).unwrap();
    e.step().unwrap(); // one slice in
    assert!(e.admission().available() < free);
    assert!(e.cancel(id));
    assert_eq!(e.admission().available(), free);
    assert_eq!(e.prefilling(), 0);
    assert!(!e.busy());
    let out = e.take_outputs();
    assert_eq!((out.len(), out[0].finish), (1, Some(FinishReason::Cancelled)));
}

#[test]
fn a_failed_deferred_prefill_is_reported_and_released() {
    let mut e = broken_engine(Broken { nan_prefill: true, ..Broken::new() });
    let free = e.admission().available();
    let id = e.submit_deferred(ok_request()).unwrap();
    assert!(e.busy());
    while e.step().unwrap() {}
    assert_eq!(e.take_failed().iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![id]);
    assert_eq!(e.admission().available(), free);
    assert_eq!(e.active(), 0);
}

/// Run `reqs` to completion through the serving loop's view of an engine; the tokens of each sequence, by id.
fn drive(e: &mut dyn gb_engine::ServeEngine, reqs: &[Request]) -> Vec<(u64, Vec<u32>)> {
    let ids: Vec<u64> = reqs.iter().map(|r| e.submit_deferred(r.clone()).unwrap()).collect();
    let mut tokens: std::collections::BTreeMap<u64, Vec<u32>> = ids.iter().map(|&i| (i, vec![])).collect();
    while e.busy() {
        e.step().unwrap();
        for ev in e.take_events() {
            tokens.get_mut(&ev.id).unwrap().extend(ev.tokens);
        }
    }
    assert_eq!(e.active(), 0);
    assert!(e.take_failed().is_empty());
    let outs = e.take_outputs();
    assert_eq!(outs.len(), reqs.len());
    tokens.into_iter().collect()
}

#[test]
fn an_erased_engine_serves_the_same_tokens_as_the_generic_one() {
    // the type is erased once, at startup: Engine<B> behind `dyn ServeEngine` must be the same engine, token for token
    let reqs = requests(6);
    let make = make_engine(Fault::None, "copy");
    let mut generic = make(true);
    let mut erased: Box<dyn gb_engine::ServeEngine> = Box::new(make(true));
    let a = drive(&mut generic, &reqs);
    let b = drive(erased.as_mut(), &reqs);
    assert_eq!(a, b);
    assert!(
        a.iter().all(|(_, t)| t.len() >= 64),
        "every sequence ran to its length: {:?}",
        a.iter().map(|(_, t)| t.len()).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------- reasoning budget (gb_engine::ReasoningBudget)

/// Run `reqs` (each with its budget) through `e` to the end; the generated tokens in submission order, and the engine's budget counters.
fn run_budgeted(
    e: &mut Engine<MockBackend>,
    reqs: &[(Request, Option<gb_engine::ReasoningBudget>)],
) -> (Vec<gb_engine::Output>, gb_engine::BudgetStats) {
    let ids: Vec<u64> = reqs.iter().map(|(r, b)| e.submit_with_budget(r.clone(), *b).unwrap()).collect();
    e.run().unwrap();
    let mut outs = e.take_outputs();
    assert_eq!(outs.len(), reqs.len());
    outs.sort_by_key(|o| ids.iter().position(|&i| i == o.id).unwrap());
    (outs, e.budget_stats())
}

/// Requests with budgets of every kind, built from the serial unbudgeted replies so each case is what its name says: a budget the reply
/// reaches with the thought still open (forced), one where the reply emits the close itself before the index (nothing forced), one past
/// `max_new` (never reached), a budget of 0 (the first token) and no budget at all.
fn budget_cases(make: &dyn Fn(bool) -> Engine<MockBackend>) -> Vec<(Request, Option<gb_engine::ReasoningBudget>, Option<usize>)> {
    let reqs = requests(12);
    let mut cases = Vec::new();
    for (i, r) in reqs.into_iter().enumerate() {
        let plain = run_budgeted(&mut make(false), &[(r.clone(), None)]).0.remove(0).tokens;
        // an id the reply never produces: every budget below `max_new` with it is forced
        let absent = (0..VOCAB as u32).find(|t| !plain.contains(t)).unwrap();
        let (budget, forced_at) = match i % 6 {
            0 => (Some(gb_engine::ReasoningBudget { after: 0, close: absent }), Some(0)),
            1 => (Some(gb_engine::ReasoningBudget { after: 1, close: absent }), Some(1)),
            2 => (Some(gb_engine::ReasoningBudget { after: 9, close: absent }), Some(9)),
            3 => (Some(gb_engine::ReasoningBudget { after: 23, close: absent }), Some(23)),
            4 => {
                // the reply's own token at index 5 is the close (its first occurrence): the model ends the thought before a budget of 30
                let k = (5..plain.len()).find(|&k| !plain[..k].contains(&plain[k])).unwrap();
                (Some(gb_engine::ReasoningBudget { after: 30.max(k + 1), close: plain[k] }), None)
            }
            _ => (if i == 5 { None } else { Some(gb_engine::ReasoningBudget { after: r.max_new + 5, close: absent }) }, None),
        };
        cases.push((r, budget, forced_at));
    }
    cases
}

/// Returns the drafted engine's budget counters and the number of cases whose budget forces a token.
fn check_budgeted(
    make: &dyn Fn(bool) -> Engine<MockBackend>,
    drafted: &mut Engine<MockBackend>,
) -> (gb_engine::BudgetStats, u64) {
    let cases = budget_cases(make);
    let n_forced = cases.iter().filter(|c| c.2.is_some()).count() as u64;
    let pairs: Vec<(Request, Option<gb_engine::ReasoningBudget>)> = cases.iter().map(|(r, b, _)| (r.clone(), *b)).collect();
    // the serial reference: each request alone on a fresh serial engine
    let serial: Vec<Vec<u32>> =
        pairs.iter().map(|p| run_budgeted(&mut make(false), std::slice::from_ref(p)).0.remove(0).tokens).collect();
    let (outs, stats) = run_budgeted(drafted, &pairs);
    let mut accepted = 0;
    for (i, ((o, want), (r, b, forced_at))) in outs.iter().zip(&serial).zip(&cases).enumerate() {
        assert_eq!(&o.tokens, want, "case {i}: drafted != serial (budget {b:?})");
        accepted += o.accepted;
        let plain = run_budgeted(&mut make(false), &[(r.clone(), None)]).0.remove(0).tokens;
        match (*b, *forced_at) {
            (Some(b), Some(n)) => {
                assert_eq!(o.tokens[n], b.close, "case {i}: the close is at the budget's index");
                assert_eq!(o.tokens[..n], plain[..n], "case {i}: nothing before the index changes");
                assert_ne!(o.tokens[n], plain[n], "case {i}: the forced token is not what the model sampled");
            }
            _ => assert_eq!(o.tokens, plain, "case {i}: a budget that does not act changes nothing"),
        }
    }
    assert!(accepted > 0, "no draft was accepted: the drafted run would prove nothing");
    (stats, n_forced)
}

#[test]
fn a_reasoning_budget_keeps_drafted_equal_to_serial_with_the_draft_clip() {
    let make = make_engine_with(Fault::None, "chain", true);
    let (stats, n) = check_budgeted(&make, &mut make(true));
    assert_eq!(stats.forced, n, "every forced case, each once");
    assert_eq!(stats.cut_runs, 0, "with the clip no drafted run reaches the index");
}

#[test]
fn the_commit_time_rule_alone_keeps_drafted_equal_to_serial() {
    // the clip off: drafted windows run past the budget's index, and only the commit loop's rule keeps the reply exact
    let make = make_engine_with(Fault::None, "chain", true);
    let (stats, n) = check_budgeted(&make, &mut make(true).without_budget_draft_clip());
    assert_eq!(stats.forced, n);
    assert!(stats.cut_runs > 0, "no drafted run crossed the budget's index: the test would not exercise the cut");
}
