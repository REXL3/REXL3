//! Weight-loader phase timers: where the boot's seconds go.
//!
//! Every blocking call of the load path adds its wall time, bytes and call count to a (component, stage) cell: the header
//! scan and the target, draft and vision weights each get their own row, and the stages are the read from disk (including
//! the widening that is fused into the read), CPU conversion, `cudaMalloc`, host-to-device copies, `cudaMemset`, and
//! (once the loader is pipelined) waiting on the device and on a free staging buffer. While loading is serial the stage
//! times add up to the load's wall time; with several loader threads a cell is the sum over those threads (thread
//! seconds), which can exceed the wall time of the phase and says so in the metric's help text.
//!
//! The cells are process-wide atomics so the crates that load (exl3 reads, cuda copies, model assembly) can record without
//! passing a handle around, and `/metrics` can read them without a lock. Recording costs two clock reads per call.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// What is being loaded; set with [`set_component`] by whoever drives the load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Component {
    /// Header scan, marker reads, fingerprints (before any weight is read).
    Scan = 0,
    Target = 1,
    Draft = 2,
    Vision = 3,
    /// Engine scratch and the KV pool (`Model::new`) after the weights.
    Engine = 4,
    /// Rows that belong to no single component: the pipelined load of all of them at once.
    All = 5,
}

pub const COMPONENTS: [Component; 6] =
    [Component::Scan, Component::Target, Component::Draft, Component::Vision, Component::Engine, Component::All];

impl Component {
    pub fn name(self) -> &'static str {
        match self {
            Component::Scan => "scan",
            Component::Target => "target",
            Component::Draft => "draft",
            Component::Vision => "vision",
            Component::Engine => "engine",
            Component::All => "all",
        }
    }
}

/// One kind of work in the load path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Reading file bytes (`pread`, or `read_exact`), including the widening fused into the read of a 16-bit tensor.
    Read = 0,
    /// CPU work on bytes already in memory: f16/bf16 to f32 or f64, sign unpacking, f64 to f16, transposes.
    Convert = 1,
    Malloc = 2,
    /// Host to device copies (and the time to issue them when they are asynchronous).
    Memcpy = 3,
    Memset = 4,
    /// Waiting for the device: a staging buffer's previous copy, or the end-of-load synchronisation.
    Wait = 5,
    /// Allocating and freeing the pinned staging ring.
    Pin = 6,
    /// Returning host load buffers to the OS (a multi-hundred-MiB `Vec` is an `mmap`: dropping it frees every page).
    Free = 7,
    /// The rest of the phase's wall time: whatever no timer above covers (see [`phase_end`]).
    Other = 8,
    /// Wall time of a whole pipelined run (not additive with the thread-seconds of the stages above).
    Wall = 9,
}

pub const STAGES: [Stage; 10] = [
    Stage::Read,
    Stage::Convert,
    Stage::Malloc,
    Stage::Memcpy,
    Stage::Memset,
    Stage::Wait,
    Stage::Pin,
    Stage::Free,
    Stage::Other,
    Stage::Wall,
];

impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Stage::Read => "read",
            Stage::Convert => "convert",
            Stage::Malloc => "malloc",
            Stage::Memcpy => "memcpy",
            Stage::Memset => "memset",
            Stage::Wait => "wait",
            Stage::Pin => "pin",
            Stage::Free => "free",
            Stage::Other => "other",
            Stage::Wall => "wall",
        }
    }
}

const NC: usize = COMPONENTS.len();
const NS: usize = STAGES.len();

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static NANOS: [[AtomicU64; NS]; NC] = [const { [const { AtomicU64::new(0) }; NS] }; NC];
static BYTES: [[AtomicU64; NS]; NC] = [const { [const { AtomicU64::new(0) }; NS] }; NC];
static CALLS: [[AtomicU64; NS]; NC] = [const { [const { AtomicU64::new(0) }; NS] }; NC];

/// Attribute everything recorded from now on to `c`.
pub fn set_component(c: Component) {
    CURRENT.store(c as usize, Ordering::Relaxed);
}

thread_local! {
    static OVERRIDE: std::cell::Cell<Option<Component>> = const { std::cell::Cell::new(None) };
}

/// The component timers on this thread are attributed to: the scoped override if one is set, else the global one.
pub fn component() -> Component {
    OVERRIDE.with(|o| o.get()).unwrap_or_else(|| COMPONENTS[CURRENT.load(Ordering::Relaxed).min(NC - 1)])
}

/// Restores the previous per-thread component when dropped.
#[must_use = "the override ends when the guard drops"]
pub struct ScopedComponent(Option<Component>);

/// Attribute everything this thread records to `c` until the guard drops, whatever the driving thread has set as current.
/// A loader thread uses it per job, because it runs while another thread has already moved on to the next component.
pub fn scoped_component(c: Component) -> ScopedComponent {
    ScopedComponent(OVERRIDE.with(|o| o.replace(Some(c))))
}

impl Drop for ScopedComponent {
    fn drop(&mut self) {
        OVERRIDE.with(|o| o.set(self.0));
    }
}

/// Add one call of `stage` that took `d` and moved `bytes` bytes.
pub fn add(stage: Stage, d: Duration, bytes: u64) {
    add_for(component(), stage, d, bytes);
}

/// [`add`] for an explicit component (loader threads record for the component whose data they are moving, whatever the
/// driving thread has set as current).
pub fn add_for(component: Component, stage: Stage, d: Duration, bytes: u64) {
    let c = component as usize;
    let s = stage as usize;
    NANOS[c][s].fetch_add(d.as_nanos().min(u64::MAX as u128) as u64, Ordering::Relaxed);
    BYTES[c][s].fetch_add(bytes, Ordering::Relaxed);
    CALLS[c][s].fetch_add(1, Ordering::Relaxed);
}

/// Times the enclosing scope as one call of `stage`.
#[must_use = "the timer records when it is dropped"]
pub struct Timer {
    stage: Stage,
    bytes: u64,
    t0: Instant,
}

/// Start timing one call of `stage` over `bytes` bytes; it is recorded when the returned guard drops.
pub fn timer(stage: Stage, bytes: u64) -> Timer {
    Timer { stage, bytes, t0: Instant::now() }
}

impl Drop for Timer {
    fn drop(&mut self) {
        add(self.stage, self.t0.elapsed(), self.bytes);
    }
}

/// Wall clock and recorded stage seconds at the start of a phase.
pub struct Phase {
    t0: Instant,
    recorded0: f64,
    component: Component,
}

/// Begin a phase of `component`'s load; [`phase_end`] turns the wall time no timer covered into [`Stage::Other`], so a
/// serial phase's stages add up to its wall time.
pub fn phase_start(component: Component) -> Phase {
    Phase { t0: Instant::now(), recorded0: component_seconds(component), component }
}

/// End a phase: returns its wall seconds. With several loader threads the recorded stage seconds can exceed the wall time;
/// then nothing is added.
pub fn phase_end(p: Phase) -> f64 {
    let wall = p.t0.elapsed();
    let covered = component_seconds(p.component) - p.recorded0;
    let rest = wall.as_secs_f64() - covered;
    if rest > 0.0 {
        let c = p.component as usize;
        let s = Stage::Other as usize;
        NANOS[c][s].fetch_add((rest * 1e9) as u64, Ordering::Relaxed);
        CALLS[c][s].fetch_add(1, Ordering::Relaxed);
    }
    wall.as_secs_f64()
}

/// One non-empty cell.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub component: Component,
    pub stage: Stage,
    pub seconds: f64,
    pub bytes: u64,
    pub calls: u64,
}

/// Every cell that has recorded at least one call, in (component, stage) order.
pub fn snapshot() -> Vec<Row> {
    let mut rows = Vec::new();
    for (ci, &component) in COMPONENTS.iter().enumerate() {
        for (si, &stage) in STAGES.iter().enumerate() {
            let calls = CALLS[ci][si].load(Ordering::Relaxed);
            if calls > 0 {
                rows.push(Row {
                    component,
                    stage,
                    seconds: NANOS[ci][si].load(Ordering::Relaxed) as f64 * 1e-9,
                    bytes: BYTES[ci][si].load(Ordering::Relaxed),
                    calls,
                });
            }
        }
    }
    rows
}

/// Sum of the seconds recorded for `component` over every stage.
pub fn component_seconds(component: Component) -> f64 {
    snapshot().iter().filter(|r| r.component == component).map(|r| r.seconds).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tests below set the process-wide current component, so they must not overlap.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn a_scoped_component_wins_on_its_thread_and_only_there() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        set_component(Component::Scan);
        let before =
            |c: Component| snapshot().into_iter().find(|r| r.component == c && r.stage == Stage::Free).map_or(0, |r| r.calls);
        let (v0, t0) = (before(Component::Vision), before(Component::Target));
        {
            let _g = scoped_component(Component::Vision);
            assert_eq!(component(), Component::Vision);
            let _t = timer(Stage::Free, 1);
            // another thread still sees the global component
            std::thread::spawn(|| assert_eq!(component(), Component::Scan)).join().unwrap();
            {
                let _inner = scoped_component(Component::Target);
                assert_eq!(component(), Component::Target);
                drop(timer(Stage::Free, 1));
            }
            assert_eq!(component(), Component::Vision, "the outer override comes back");
        }
        assert_eq!(component(), Component::Scan);
        assert_eq!((before(Component::Vision) - v0, before(Component::Target) - t0), (1, 1));
    }

    #[test]
    fn a_timer_lands_in_the_current_component_and_stage() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        set_component(Component::Engine);
        let before = snapshot().into_iter().find(|r| r.component == Component::Engine && r.stage == Stage::Pin);
        {
            let _t = timer(Stage::Pin, 4096);
            std::thread::sleep(Duration::from_millis(5));
        }
        let after = snapshot().into_iter().find(|r| r.component == Component::Engine && r.stage == Stage::Pin).unwrap();
        let (b_s, b_b, b_c) = before.map_or((0.0, 0, 0), |r| (r.seconds, r.bytes, r.calls));
        assert!(after.seconds - b_s >= 0.005, "slept 5 ms, recorded {}", after.seconds - b_s);
        assert_eq!(after.bytes - b_b, 4096);
        assert_eq!(after.calls - b_c, 1);
        set_component(Component::Scan);
    }
}
