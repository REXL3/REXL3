//! The engine thread over every way the server can be handed an engine (release plan Phase 1 "never a zombie", seen from the way a
//! model family's engine really reaches the runtime).
//!
//! After the family registry a family's `build` returns `Box<dyn ServeEngine>` and the server starts it with `spawn_with`
//! (`spawn_dyn` with a configuration); a concrete `Engine<B>` is erased by `spawn`/`spawn_cfg`. All four end in the one loop
//! (`runtime::run`, over `&mut dyn ServeEngine`). These tests drive a poisoned sequence and a fatal backend error, injected through
//! the mock backend's `Faults`, through every entry point and require the same outcome from each: the same tokens for the sequences
//! that were not poisoned, the same error for the one that was, the same last words for everything in flight when the engine dies.
//!
//! `spawn` and `spawn_dyn` carry the default `RuntimeConfig`, whose `on_fatal` ends the process, so only the two entry points that
//! take a configuration can run the fatal scenario in this process (the real binary's exit status 70 is `tests/process.rs`).

use gb_engine::mock::{Faults, MockBackend, MockOptions};
use gb_engine::{Drafter, Engine, EngineConfig, NoDrafts, Request, ServeEngine};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};
use gb_server::flow::{channel as unbounded_channel, Receiver as UnboundedReceiver};
use gb_server::runtime::{
    spawn, spawn_cfg, spawn_dyn, spawn_with, EngineHandle, FatalInfo, GenRequest, RuntimeConfig, SeqMsg, State,
};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

const VOCAB: usize = 512;

fn engine(mock: &MockOptions, slots: usize) -> Engine<MockBackend> {
    let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1024, state_bytes_per_slot: 1 << 20, activation_reserve: 0 };
    Engine::new(
        mock.build(VOCAB, slots),
        Admission::new((1 << 20) * 1024 + slots as u64 * (1 << 20), 0, mm, slots).unwrap(),
        EngineConfig { max_rows: 64, max_drafts: 7, cost: CostModel::from_table(vec![(1, 1.0), (64, 2.0)]), drafting: false },
        Box::new(|| Box::new(NoDrafts) as Box<dyn Drafter>),
    )
}

/// The ways to start an engine thread.
#[derive(Clone, Copy, Debug)]
enum Entry {
    /// `spawn`: a concrete `Engine<B>`, default configuration.
    Spawn,
    /// `spawn_dyn`: an erased engine, default configuration.
    SpawnDyn,
    /// `spawn_cfg`: a concrete `Engine<B>`, with a configuration.
    SpawnCfg,
    /// `spawn_with`: an erased engine, with a configuration (what `main` calls for the mock and for every CUDA family).
    SpawnWith,
}

const ALL: [Entry; 4] = [Entry::Spawn, Entry::SpawnDyn, Entry::SpawnCfg, Entry::SpawnWith];
const CONFIGURABLE: [Entry; 2] = [Entry::SpawnCfg, Entry::SpawnWith];

impl Entry {
    fn start(self, mock: &MockOptions, slots: usize, cfg: RuntimeConfig) -> (EngineHandle, JoinHandle<()>) {
        let mock = mock.clone();
        match self {
            Entry::Spawn => spawn(move || engine(&mock, slots)),
            Entry::SpawnDyn => spawn_dyn(Box::new(move || Box::new(engine(&mock, slots)) as Box<dyn ServeEngine>)),
            Entry::SpawnCfg => spawn_cfg(move || engine(&mock, slots), cfg),
            Entry::SpawnWith => spawn_with(Box::new(move || Box::new(engine(&mock, slots)) as Box<dyn ServeEngine>), cfg),
        }
    }
}

fn gen_request(first: u32, max_new: usize) -> GenRequest {
    GenRequest {
        prompt: vec![first, 5, 6, 7],
        sampling: Sampling { temperature: 0.0, ..Sampling::default() },
        max_new,
        stop: vec![],
        draft: false,
        images: vec![],
        cache: true,
        reasoning_budget: None,
    }
}

/// The tokens `first` produces alone on a clean engine, driven without the runtime: the reference every run must equal.
fn solo(first: u32, max_new: usize) -> Vec<u32> {
    let g = gen_request(first, max_new);
    let mut e = engine(&MockOptions { vocab: VOCAB, ..Default::default() }, 8);
    e.submit(Request {
        prompt: g.prompt,
        sampling: g.sampling,
        max_new: g.max_new,
        stop: g.stop,
        draft: g.draft,
        images: g.images,
        cache: g.cache,
    })
    .unwrap();
    e.run().unwrap();
    e.take_outputs().remove(0).tokens
}

/// What a request got back: its tokens, and how it ended.
#[derive(Debug, Clone, PartialEq)]
struct Reply {
    tokens: Vec<u32>,
    end: String,
}

async fn collect(mut rx: UnboundedReceiver<SeqMsg>) -> Reply {
    let mut tokens = Vec::new();
    loop {
        match tokio::time::timeout(Duration::from_secs(30), rx.recv()).await {
            Ok(Some(SeqMsg::Tokens(t))) => tokens.extend(t),
            Ok(Some(SeqMsg::Done(d))) => {
                let end = match (d.finish, d.error) {
                    (Some(f), None) => format!("{f:?}"),
                    (None, Some(e)) => format!("{:?}: {}", e.kind, e.message),
                    other => format!("inconsistent end {other:?}"),
                };
                return Reply { tokens, end };
            }
            Ok(None) => return Reply { tokens, end: "the channel closed with no end".into() },
            Err(_) => panic!("no reply within 30 s"),
        }
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap()
}

async fn ready(handle: &EngineHandle) {
    for _ in 0..2000 {
        if handle.metrics.state() == State::Ready {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("the engine never became ready");
}

/// Four requests share an engine; the one that starts with `POISON` has every row it produces come back NaN.
fn poison_scenario(entry: Entry) -> Vec<Reply> {
    const POISON: u32 = 11;
    let mock = MockOptions {
        step_ms: 5,
        vocab: VOCAB,
        faults: Faults { poison_first_token: Some(POISON), ..Default::default() },
        ..Default::default()
    };
    let (handle, thread) = entry.start(&mock, 8, RuntimeConfig::default());
    let replies = runtime().block_on(async {
        ready(&handle).await;
        let mut rxs = Vec::new();
        for first in [100, 101, POISON, 103] {
            let (tx, rx) = unbounded_channel();
            handle.submit(gen_request(first, 60), tx).await.expect("queued");
            rxs.push(rx);
        }
        let mut replies = Vec::new();
        for rx in rxs {
            replies.push(collect(rx).await);
        }
        replies
    });
    let m = &handle.metrics;
    assert_eq!(m.failed_sequences.load(Relaxed), 1, "{entry:?}: one sequence failed");
    assert_eq!(m.step_errors.load(Relaxed), 0, "{entry:?}: a poisoned row is not a step error");
    assert_eq!(m.state(), State::Ready, "{entry:?}: one bad sequence is not an unhealthy engine");
    handle.shutdown();
    thread.join().unwrap();
    replies
}

#[test]
fn a_poisoned_sequence_fails_alone_through_every_way_of_starting_an_engine() {
    // mutation: make the engine fail the whole step on one bad row (engine/src/lib.rs decode_round) and the healthy three end in an error
    let by_entry: Vec<(Entry, Vec<Reply>)> = ALL.iter().map(|&e| (e, poison_scenario(e))).collect();
    let (first, expected) = &by_entry[0];
    for (entry, replies) in &by_entry {
        assert_eq!(replies, expected, "{entry:?} and {first:?} must give the same replies");
    }
    for (i, r) in expected.iter().enumerate() {
        if i == 2 {
            assert!(r.end.starts_with("Engine: ") && r.end.contains("NaN"), "the poisoned sequence: {r:?}");
        } else {
            assert_eq!(r.end, "Length", "sequence {i}: {r:?}");
            assert_eq!(
                r.tokens,
                solo([100, 101, 0, 103][i], 60),
                "sequence {i} must equal its solo reply despite the neighbour's failure"
            );
        }
    }
}

/// What the engine's last words were: how every request in flight or queued ended, who was told the engine died, and what a request
/// made afterwards got.
#[derive(Debug, PartialEq)]
struct Death {
    ends: Vec<String>,
    fatal: Vec<(String, bool, i32)>,
    state: State,
    ready: bool,
    later: String,
}

/// Three requests on two slots (two running, one queued), then a fatal backend error in the sixth step.
fn fatal_scenario(entry: Entry) -> Death {
    let mock = MockOptions {
        step_ms: 10,
        vocab: VOCAB,
        faults: Faults { fatal_at_step: Some(6), ..Default::default() },
        ..Default::default()
    };
    let seen: Arc<Mutex<Vec<FatalInfo>>> = Default::default();
    let hook = seen.clone();
    let cfg = RuntimeConfig { on_fatal: Arc::new(move |info| hook.lock().unwrap().push(info.clone())), ..Default::default() };
    let (handle, thread) = entry.start(&mock, 2, cfg);
    let (ends, later) = runtime().block_on(async {
        ready(&handle).await;
        let mut rxs = Vec::new();
        for first in [100, 101, 102] {
            let (tx, rx) = unbounded_channel();
            handle.submit(gen_request(first, 500), tx).await.expect("queued");
            rxs.push(rx);
        }
        let mut ends = Vec::new();
        for rx in rxs {
            ends.push(collect(rx).await.end);
        }
        // the engine thread tells `on_fatal` last, after it has answered everyone
        fatal_reported(&seen).await;
        let (tx, _rx) = unbounded_channel();
        (ends, format!("{:?}", handle.submit(gen_request(103, 5), tx).await))
    });
    thread.join().unwrap();
    let fatal = seen.lock().unwrap().iter().map(|i| (i.reason.clone(), i.panicked, i.exit_code())).collect();
    Death { ends, fatal, state: handle.metrics.state(), ready: handle.metrics.ready(), later }
}

async fn fatal_reported(seen: &Arc<Mutex<Vec<FatalInfo>>>) {
    for _ in 0..2000 {
        if !seen.lock().unwrap().is_empty() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("on_fatal was never called");
}

#[test]
fn a_fatal_backend_error_ends_everything_the_same_way_through_the_erased_and_the_generic_engine() {
    // mutation: treat `StepError::Fatal` like a recoverable one in runtime::run (drop the `is_fatal` arm) and the engine goes on to
    // `fail_active` the sequences and only gives up after `exit_after` steps, with another reason
    let deaths: Vec<(Entry, Death)> = CONFIGURABLE.iter().map(|&e| (e, fatal_scenario(e))).collect();
    let (first, expected) = &deaths[0];
    for (entry, d) in &deaths {
        assert_eq!(d, expected, "{entry:?} and {first:?} must die the same way");
    }
    let why = "fatal backend error: injected fatal error at step 6";
    assert_eq!(expected.fatal, vec![(why.to_string(), false, 70)], "one report, a fatal error (not a panic), exit status 70");
    assert_eq!(
        expected.ends,
        vec![format!("EngineDown: the engine failed: {why}"); 3],
        "the two running requests and the queued one are each answered with the error, never a normal finish"
    );
    assert_eq!((expected.state, expected.ready), (State::Failed, false));
    // a request made afterwards is refused (`http` answers it 503), not queued for an engine that is gone
    assert_eq!(expected.later, "Err(EngineDown(\"the engine has stopped\"))");
}

/// An engine whose `quiesce` fails, as a tensor-parallel leader's does when its group was lost while it idled (the device state is
/// lost), and which says when it is dropped: the engine's destructor is what frees the device buffers, the graphs and the communicator.
struct QuiesceFails {
    inner: Engine<MockBackend>,
    dropped: Arc<std::sync::atomic::AtomicBool>,
}

impl Drop for QuiesceFails {
    fn drop(&mut self) {
        self.dropped.store(true, Relaxed);
    }
}

impl ServeEngine for QuiesceFails {
    fn quiesce(&mut self) -> Result<(), String> {
        Err("device state lost: tensor-parallel group lost: the connection to rank 1 failed".into())
    }
    fn validate(&self, req: &Request) -> Result<(), gb_engine::SubmitError> {
        ServeEngine::validate(&self.inner, req)
    }
    fn would_admit(&self, req: &Request) -> Result<(), gb_sched::AdmitError> {
        ServeEngine::would_admit(&self.inner, req)
    }
    fn submit_deferred(&mut self, req: Request) -> Result<u64, gb_engine::SubmitError> {
        ServeEngine::submit_deferred(&mut self.inner, req)
    }
    fn busy(&self) -> bool {
        ServeEngine::busy(&self.inner)
    }
    fn step(&mut self) -> Result<bool, gb_engine::StepError> {
        ServeEngine::step(&mut self.inner)
    }
    fn cancel(&mut self, id: u64) -> bool {
        ServeEngine::cancel(&mut self.inner, id)
    }
    fn abort_all(&mut self) -> usize {
        ServeEngine::abort_all(&mut self.inner)
    }
    fn fail_active(&mut self, why: &str) -> usize {
        ServeEngine::fail_active(&mut self.inner, why)
    }
    fn active(&self) -> usize {
        ServeEngine::active(&self.inner)
    }
    fn prefilling(&self) -> usize {
        ServeEngine::prefilling(&self.inner)
    }
    fn take_events(&mut self) -> Vec<gb_engine::Event> {
        ServeEngine::take_events(&mut self.inner)
    }
    fn take_failed(&mut self) -> Vec<(u64, String)> {
        ServeEngine::take_failed(&mut self.inner)
    }
    fn take_outputs(&mut self) -> Vec<gb_engine::Output> {
        ServeEngine::take_outputs(&mut self.inner)
    }
    fn spec_stats(&self) -> gb_spec::stats::SpecStats {
        ServeEngine::spec_stats(&self.inner)
    }
}

/// Stop an idle runtime whose engine's `quiesce` fails; returns (what `on_fatal` was told, whether the engine was dropped).
fn drained_with_failing_quiesce(leak: bool) -> (Vec<FatalInfo>, bool) {
    let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let seen: Arc<Mutex<Vec<FatalInfo>>> = Default::default();
    let hook = seen.clone();
    let cfg = RuntimeConfig {
        on_fatal: Arc::new(move |info| hook.lock().unwrap().push(info.clone())),
        leak_engine_on_fatal: leak,
        ..Default::default()
    };
    let d = dropped.clone();
    let mock = MockOptions { vocab: VOCAB, ..Default::default() };
    let (handle, thread) =
        spawn_with(Box::new(move || Box::new(QuiesceFails { inner: engine(&mock, 2), dropped: d }) as Box<dyn ServeEngine>), cfg);
    runtime().block_on(async {
        ready(&handle).await;
        handle.drain(Duration::from_millis(500));
        fatal_reported(&seen).await;
    });
    thread.join().unwrap();
    let told = seen.lock().unwrap().clone();
    (told, dropped.load(Relaxed))
}

#[test]
fn a_failed_quiescence_leaks_the_engine_of_a_group_like_every_other_fatal_exit() {
    // The one fatal path that dropped the engine: `quiesce` fails after a drain (a leader SIGTERMed after its group was lost), and the
    // old code `return`ed from inside the closure, ahead of the `forget`. The destructor then freed the device state while the abort
    // thread was inside the communicator, and the process died of SIGSEGV in its exit-time teardown (139 where 70 was logged).
    // mutation: put the early `return Exit::Fatal(..)` back and the engine is dropped here.
    let (told, dropped) = drained_with_failing_quiesce(true);
    assert_eq!(told.len(), 1, "{told:?}");
    assert!(told[0].reason.starts_with("engine quiescence failed: device state lost"), "{}", told[0].reason);
    assert_eq!((told[0].panicked, told[0].exit_code()), (false, 70));
    assert!(!dropped, "a group's engine is leaked on every fatal exit, not dropped");
    // and without the flag (every single-process family) nothing changes: the engine is dropped on the way out, as before
    let (told, dropped) = drained_with_failing_quiesce(false);
    assert_eq!(told.len(), 1);
    assert!(dropped, "a single-process engine is dropped");
}

#[test]
fn the_exit_mode_rides_on_the_failure_report() {
    // the default `on_fatal` ends the process the way the report says: a group's `_exit`, everything else `exit`
    let seen: Arc<Mutex<Vec<FatalInfo>>> = Default::default();
    let hook = seen.clone();
    let cfg = RuntimeConfig {
        on_fatal: Arc::new(move |info| hook.lock().unwrap().push(info.clone())),
        fatal_exit: gb_model::exit::fatal_exit_mode(true),
        ..Default::default()
    };
    let mock = MockOptions {
        step_ms: 10,
        vocab: VOCAB,
        faults: Faults { fatal_at_step: Some(2), ..Default::default() },
        ..Default::default()
    };
    let (handle, thread) = spawn_with(Box::new(move || Box::new(engine(&mock, 2)) as Box<dyn ServeEngine>), cfg);
    runtime().block_on(async {
        ready(&handle).await;
        let (tx, rx) = unbounded_channel();
        handle.submit(gen_request(100, 500), tx).await.expect("queued");
        collect(rx).await;
        fatal_reported(&seen).await;
    });
    thread.join().unwrap();
    let told = seen.lock().unwrap().clone();
    assert_eq!(told.len(), 1);
    assert_eq!(told[0].exit, gb_model::exit::FatalExit::Immediate);
    assert_eq!(told[0].exit_code(), 70, "the code does not change with the way the process ends");
    assert_eq!(gb_model::exit::fatal_exit_mode(false), gb_model::exit::FatalExit::Exit);
}
