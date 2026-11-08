//! Admission and the KV pool (serve hardening, finding 1): a request that cannot fit the pool the plan allocated is refused at submission,
//! before any prefill, and requests that fit alone but not together wait for each other instead of running the pool dry. The byte-level
//! arithmetic is in `gb-sched` (`Admission::with_kv_tokens`) and the plans of both families are driven against the real page tables in
//! `gb-model` (`plan::tests`); this file is the engine's side: what `submit_deferred` and `step` do with it.

use gb_engine::mock::{Fault, MockBackend};
use gb_engine::{Drafter, Engine, EngineConfig, FinishReason, NoDrafts, Request, SubmitError};
use gb_sampling::Sampling;
use gb_sched::{Admission, AdmitError, CostModel, ModelMemory};

const VOCAB: usize = 512;
const SLOTS: usize = 8;
/// The shape of a real plan: a slot's state costs 1,000 tokens of KV, and the pool holds 2,000 tokens.
const POOL: u64 = 2_000;

fn memory() -> ModelMemory {
    ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1000, state_bytes_per_slot: 1_000_000, activation_reserve: 0 }
}

/// The byte capacity of a plan: the pool's KV bytes and the state of every slot.
fn capacity() -> u64 {
    POOL * 1000 + SLOTS as u64 * 1_000_000
}

fn engine(admission: Admission) -> Engine<MockBackend> {
    let cost = CostModel::from_table(vec![(1, 70.0), (16, 74.0), (64, 190.0)]);
    let make: Box<dyn Fn() -> Box<dyn Drafter>> = Box::new(|| Box::new(NoDrafts) as Box<dyn Drafter>);
    Engine::new(
        MockBackend::new(VOCAB, SLOTS, Fault::None),
        admission,
        EngineConfig { max_rows: 64, max_drafts: 7, cost, drafting: false },
        make,
    )
}

/// What a plan builds: the byte capacity, the window, and the pool in tokens.
fn with_pool() -> Engine<MockBackend> {
    engine(Admission::new(capacity(), 0, memory(), SLOTS).unwrap().with_max_context(262_144).with_kv_tokens(POOL))
}

/// What the engines built before: the byte capacity and the window.
fn without_pool() -> Engine<MockBackend> {
    engine(Admission::new(capacity(), 0, memory(), SLOTS).unwrap().with_max_context(262_144))
}

fn request(prompt_len: usize, max_new: usize, seed: u64) -> Request {
    Request {
        prompt: (0..prompt_len).map(|j| ((seed as usize * 131 + j * 17 + 3) % VOCAB) as u32).collect(),
        sampling: Sampling { seed, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 },
        max_new,
        stop: vec![],
        draft: false,
        images: vec![],
        cache: false,
    }
}

fn drain(e: &mut Engine<MockBackend>) {
    let mut steps = 0;
    while e.busy() {
        e.step().unwrap();
        steps += 1;
        assert!(steps < 100_000, "the engine did not finish: a deadlock");
    }
}

/// The tokens `r` produces alone on an engine with the pool (the reference a concurrent run must equal).
fn solo(r: &Request) -> Vec<u32> {
    let mut e = with_pool();
    e.submit(r.clone()).unwrap();
    e.run().unwrap();
    e.take_outputs().remove(0).tokens
}

#[test]
fn a_prompt_past_the_pool_is_refused_at_submission_before_any_prefill() {
    // 2,500 prompt tokens and 4 new ones: 2,504 against a pool of 2,000. The byte capacity is 2,000,000 + 8,000,000 of slot state and one
    // sequence costs 2,504,000 + 1,000,000, so the old admission takes it.
    let r = request(2_500, 4, 1);
    let mut old = without_pool();
    assert!(old.validate(&r).is_ok(), "the old rule admits it: this is the bug");
    assert!(old.submit_deferred(r.clone()).is_ok());

    let mut e = with_pool();
    let why = SubmitError::Admit(AdmitError::ContextTooLong { tokens: 2_504, max: POOL });
    assert_eq!(e.validate(&r), Err(why.clone()), "refused where the server validates, before the request is queued");
    assert_eq!(e.submit_deferred(r.clone()), Err(why.clone()));
    assert_eq!(e.submit(r.clone()), Err(why));
    assert_eq!((e.prefilling(), e.active(), e.busy()), (0, 0, false), "nothing was started");
    assert_eq!(e.admission().available(), e.admission().capacity(), "nothing stays reserved");
    assert_eq!(e.admission().kv_tokens_in_use(), 0);
    // the pool's own size is served, and the engine is untouched by the refusal
    let ok = request(1_990, 10, 2);
    assert_eq!(e.submit(ok.clone()), Ok(0));
    e.run().unwrap();
    assert_eq!(e.take_outputs().remove(0).tokens, solo(&ok));
    // and the limit the server reports is the pool, below the window
    assert_eq!(e.admission().request_limit(), Some(POOL));
}

#[test]
fn requests_that_fit_alone_but_not_together_queue_and_none_is_failed() {
    // 1,300 tokens each: 2,600 against the pool of 2,000. The idle slots' state would have let both in (2 x 2,300,000 of 10,000,000 bytes).
    let (a, b) = (request(1_200, 100, 3), request(1_250, 50, 4));
    let mut old = without_pool();
    old.submit_deferred(a.clone()).unwrap();
    assert!(old.would_admit(&b).is_ok(), "the old rule admits the second beside the first (the pool then runs dry)");

    let mut e = with_pool();
    e.submit_deferred(a.clone()).unwrap();
    assert!(e.validate(&b).is_ok(), "alone it fits, so the server queues it rather than refusing it");
    assert!(matches!(e.would_admit(&b), Err(AdmitError::NoMemory { .. })), "{:?}", e.would_admit(&b));
    assert!(matches!(e.submit_deferred(b.clone()), Err(SubmitError::Admit(AdmitError::NoMemory { .. }))));
    // the first runs to its end undisturbed, the second is not in the engine
    let mut steps = 0;
    while e.busy() {
        e.step().unwrap();
        steps += 1;
        assert!(steps < 100_000);
        assert!(e.take_failed().is_empty(), "a running request must never fail for a queued one");
    }
    let first = e.take_outputs().remove(0);
    assert_eq!((first.finish, first.tokens.clone()), (Some(FinishReason::Length), solo(&a)));
    // the pool is free again, the second is admitted and finishes with the reply it has alone
    assert_eq!(e.admission().kv_tokens_in_use(), 0);
    e.submit_deferred(b.clone()).unwrap();
    drain(&mut e);
    assert!(e.take_failed().is_empty());
    assert_eq!(e.take_outputs().remove(0).tokens, solo(&b));
}

#[test]
fn a_pool_full_of_small_requests_and_a_big_one_behind_them_never_deadlocks() {
    // five requests of 400 tokens fill the pool; a request of 1,500 waits for the whole pool and gets it when they are done
    let small: Vec<Request> = (0..5).map(|i| request(350, 50, 10 + i)).collect();
    let big = request(1_450, 50, 20);
    let mut e = with_pool();
    for r in &small {
        e.submit_deferred(r.clone()).unwrap();
    }
    assert!(matches!(e.would_admit(&big), Err(AdmitError::NoMemory { .. })));
    // the loop the server runs: admit from the front while it fits, step otherwise
    let mut waiting = vec![big.clone()];
    let mut done = 0;
    let mut steps = 0;
    while e.busy() || !waiting.is_empty() {
        if let Some(front) = waiting.first() {
            if e.would_admit(front).is_ok() {
                e.submit_deferred(waiting.remove(0)).unwrap();
            }
        }
        e.step().unwrap();
        assert!(e.take_failed().is_empty());
        done += e.take_outputs().len();
        steps += 1;
        assert!(steps < 100_000, "deadlock: {} waiting, {} active", waiting.len(), e.active());
    }
    assert_eq!(done, 6, "all five small requests and the big one finished");
    assert_eq!(e.admission().kv_tokens_in_use(), 0);
}
