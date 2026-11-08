//! Boot phases and identity facts, recorded once per process.
//!
//! Each phase is logged as one `key=value` line (`rexl3 event=boot phase=target_loaded t=14.376 gib=14.65`) and kept here, so
//! `/metrics` can export `rexl3_boot_phase_seconds{phase}` and tools never have to parse the wording of a log line. Facts such
//! as the pack fingerprints are kept as name/value pairs for the same reason.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
thread_local! { static CANCEL: std::cell::RefCell<Option<Arc<AtomicBool>>> = const { std::cell::RefCell::new(None) }; }
/// Typed cooperative stop, NOT an engine panic. Only raised at quiescent safe boundaries.
#[derive(Debug)]
pub struct Cancelled;
pub fn install_cancel(flag: Arc<AtomicBool>) {
    CANCEL.with(|c| *c.borrow_mut() = Some(flag));
}
pub fn cancel_requested() -> bool {
    CANCEL.with(|c| c.borrow().as_ref().is_some_and(|f| f.load(Ordering::Acquire)))
}
pub fn checkpoint() {
    if cancel_requested() {
        std::panic::resume_unwind(Box::new(Cancelled));
    }
}
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub struct Phase {
    pub name: &'static str,
    /// Seconds since [`start`].
    pub secs: f64,
    /// Memory the phase made resident, when it has a size.
    pub gib: Option<f64>,
}

static START: OnceLock<Instant> = OnceLock::new();
static PHASES: Mutex<Vec<Phase>> = Mutex::new(Vec::new());
static FACTS: Mutex<Vec<(&'static str, String)>> = Mutex::new(Vec::new());

/// Fix time zero. Call first thing in `main`; later calls do nothing.
pub fn start() {
    START.get_or_init(Instant::now);
}

/// Seconds since [`start`] (which runs now if nobody called it).
pub fn elapsed() -> f64 {
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// Record and log a phase. A phase recorded twice keeps its first time.
pub fn mark(name: &'static str, gib: Option<f64>) {
    let secs = elapsed();
    let mut p = PHASES.lock().unwrap_or_else(|e| e.into_inner());
    if p.iter().any(|x| x.name == name) {
        return;
    }
    match gib {
        Some(g) => eprintln!("rexl3 event=boot phase={name} t={secs:.3} gib={g:.2}"),
        None => eprintln!("rexl3 event=boot phase={name} t={secs:.3}"),
    }
    p.push(Phase { name, secs, gib });
}

/// Every phase recorded so far, in order.
pub fn phases() -> Vec<Phase> {
    PHASES.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Record an identity fact (replacing an earlier value under the same name).
pub fn fact(name: &'static str, value: impl Into<String>) {
    let value = value.into();
    let mut f = FACTS.lock().unwrap_or_else(|e| e.into_inner());
    match f.iter_mut().find(|(n, _)| *n == name) {
        Some(slot) => slot.1 = value,
        None => f.push((name, value)),
    }
}

/// Every fact recorded so far, in order.
pub fn facts() -> Vec<(&'static str, String)> {
    FACTS.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases_keep_their_first_time_and_facts_their_last_value() {
        start();
        mark("test_phase", Some(1.5));
        let first = phases().into_iter().find(|p| p.name == "test_phase").unwrap();
        mark("test_phase", None);
        let again = phases().into_iter().find(|p| p.name == "test_phase").unwrap();
        assert_eq!(first, again);
        assert_eq!(first.gib, Some(1.5));
        fact("test_fact", "a");
        fact("test_fact", "b");
        assert_eq!(facts().iter().filter(|(n, _)| *n == "test_fact").map(|(_, v)| v.as_str()).collect::<Vec<_>>(), ["b"]);
    }
}
