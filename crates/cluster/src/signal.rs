//! SIGINT, SIGTERM and SIGHUP set one flag: the launcher and the supervisor poll it between steps and stop the group by pid.
//! std has no signal API and the libc crate is not a dependency, so the one function is declared here (it links from the libc that
//! std already links).

use std::sync::atomic::{AtomicBool, Ordering};

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn signal(signum: i32, handler: usize) -> usize;
}

extern "C" fn on_signal(_sig: i32) {
    STOP.store(true, Ordering::SeqCst);
}

/// Install the handlers (idempotent).
pub fn install() {
    const SIGHUP: i32 = 1;
    const SIGINT: i32 = 2;
    const SIGTERM: i32 = 15;
    // SAFETY: `signal` is given a valid `extern "C" fn(i32)`; the handler only stores to an atomic (async-signal-safe).
    unsafe {
        signal(SIGHUP, on_signal as usize);
        signal(SIGINT, on_signal as usize);
        signal(SIGTERM, on_signal as usize);
    }
}

/// A stop was requested by the operator (a signal arrived).
pub fn requested() -> bool {
    STOP.load(Ordering::SeqCst)
}
