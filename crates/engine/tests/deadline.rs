//! Deadline slicing changes scheduling only; the prompt identity/arithmetic never changes.
use gb_engine::mock::{Fault, MockBackend};
use gb_engine::{Engine, EngineConfig, NoDrafts, Request, ServeEngine};
use gb_sched::{Admission, CostModel, ModelMemory};
use std::time::{Duration, Instant};
fn engine() -> Engine<MockBackend> {
    Engine::new(
        MockBackend::new(256, 1, Fault::None).with_path_dependent_state().with_prefill_slice(512),
        Admission::new(
            1 << 24,
            0,
            ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 1, activation_reserve: 0 },
            1,
        )
        .unwrap(),
        EngineConfig { max_rows: 16, max_drafts: 0, drafting: false, cost: CostModel::from_table(vec![(1, 1.0), (16, 2.0)]) },
        Box::new(|| Box::new(NoDrafts)),
    )
}
fn request() -> Request {
    Request {
        prompt: (0..2048).map(|i| (i % 255) as u32).collect(),
        sampling: Default::default(),
        max_new: 32,
        stop: vec![],
        draft: false,
        images: vec![],
        cache: false,
    }
}
#[test]
fn smaller_deadline_chunks_keep_committed_tokens_bit_identical() {
    let mut plain = engine();
    let mut limited = engine();
    plain.submit_deferred(request()).unwrap();
    limited.submit_deferred(request()).unwrap();
    plain.step().unwrap();
    limited.step().unwrap(); // actual prior measurement, not a guessed cold budget
    limited.set_slice_deadline(Some(Instant::now() + Duration::from_secs(5)));
    let mut plain_steps = 1;
    let mut limited_steps = 1;
    while plain.busy() {
        plain.step().unwrap();
        plain_steps += 1;
    }
    while limited.busy() {
        limited.step().unwrap();
        limited_steps += 1;
    }
    assert!(limited_steps > plain_steps, "deadline actually shrank the prefill, not merely a gauge");
    assert_eq!(plain.take_outputs()[0].tokens, limited.take_outputs()[0].tokens);
    assert!(limited.would_admit(&request()).is_ok(), "all slot/KV ownership released");
}
#[test]
fn an_unknown_or_overbudget_slice_is_not_started_and_cancel_releases_it() {
    let mut e = engine();
    let id = e.submit_deferred(request()).unwrap();
    e.set_slice_deadline(Some(Instant::now() + Duration::from_millis(10)));
    assert!(!e.step().unwrap(), "no measured safe cold budget: refuse work");
    assert!(e.take_events().is_empty());
    assert_eq!(e.prefilling(), 1);
    assert!(e.cancel(id));
    assert!(!e.busy());
    assert!(e.would_admit(&request()).is_ok());
}

/// The drain after SIGTERM: the server sets a slice deadline of now + the grace (8 s by default), which is inside the engine's 10 s
/// urgency window at once. A short reply that is decoding while a long prompt prefills next to it must still finish inside the grace,
/// with the tokens it gets undisturbed, when the backend limits its prefill rows; a backend that cannot limit them is refused every
/// step (fail-closed) and the reply is cut.
fn drain_engine(backend: MockBackend) -> Engine<MockBackend> {
    Engine::new(
        backend.with_prefill_slice(512).with_latency(Duration::from_millis(2), Duration::from_millis(20)),
        Admission::new(
            1 << 24,
            0,
            ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 1, activation_reserve: 0 },
            2,
        )
        .unwrap(),
        EngineConfig { max_rows: 16, max_drafts: 0, drafting: false, cost: CostModel::from_table(vec![(1, 1.0), (16, 2.0)]) },
        Box::new(|| Box::new(NoDrafts)),
    )
}
fn short_request() -> Request {
    Request { prompt: (1..9).collect(), max_new: 16, ..request() }
}
/// Starts the short reply (prefilled, decoding), then the long prompt, then "SIGTERM": the deadline is set. The engine and both ids.
fn start_and_drain(backend: MockBackend) -> (Engine<MockBackend>, u64, u64) {
    let mut e = drain_engine(backend);
    let short = e.submit_deferred(short_request()).unwrap();
    for _ in 0..3 {
        e.step().unwrap();
    }
    let long = e.submit_deferred(request()).unwrap();
    e.step().unwrap();
    assert!(e.prefilling() == 1, "the long prompt is prefilling next to the decoding reply");
    e.set_slice_deadline(Some(Instant::now() + Duration::from_secs(8)));
    (e, short, long)
}
#[test]
fn an_in_flight_short_reply_finishes_inside_the_drain_grace_when_the_backend_limits_prefill_rows() {
    let mut alone = drain_engine(MockBackend::new(256, 2, Fault::None).with_path_dependent_state());
    let id = alone.submit_deferred(short_request()).unwrap();
    while alone.busy() {
        alone.step().unwrap();
    }
    let want = alone.take_outputs().into_iter().find(|o| o.id == id).unwrap().tokens;
    assert_eq!(want.len(), 16);

    let (mut e, short, _long) = start_and_drain(MockBackend::new(256, 2, Fault::None).with_path_dependent_state());
    let t = Instant::now();
    let mut finished = None;
    while finished.is_none() && t.elapsed() < Duration::from_secs(8) {
        assert!(e.step().unwrap(), "a backend that limits its rows is not refused under the drain deadline");
        finished = e.take_outputs().into_iter().find(|o| o.id == short);
    }
    let out = finished.expect("the short reply finished inside the 8 s grace");
    assert_eq!(out.tokens, want, "the reply's tokens are the ones it gets undisturbed");
    assert!(t.elapsed() < Duration::from_secs(8));

    // the control: the same backend without the row limit is refused at once and the reply never completes
    let (mut e, short, _long) =
        start_and_drain(MockBackend::new(256, 2, Fault::None).with_path_dependent_state().without_row_limit());
    for _ in 0..4 {
        assert!(!e.step().unwrap(), "an urgent step on a backend that cannot limit its rows is refused");
    }
    assert!(e.take_outputs().iter().all(|o| o.id != short), "the reply was cut");
    assert!(e.cancel(short));
}
