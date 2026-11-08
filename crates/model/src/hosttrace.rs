//! Host-side timeline marks of the lockstep command stream (`REXL3_TP_TRACE=1`), for attributing the follower's entry skew
//! (`docs/NEMOTRON_TP.md` 9.2 and 17). Off (the default): one relaxed load per mark, nothing is stored, no thread exists.
//!
//! A mark is `(CLOCK_MONOTONIC ns, role, tag, arg)`; the role is the marking thread's (`eng`: the leader's engine thread, `lrx`: a
//! leader reader, `frx`: the follower's reader, `fapply`: the thread that applies commands). A flusher thread drains the buffer to stderr
//! every 250 ms as `TPTRACE <L|F> <ns> <role> <tag> <arg>`, so a run's stderr is the timeline of that rank; the two ranks' clocks differ
//! by an offset that `tools/nemotron_tp_skew.py` estimates from the command / ack round trips. Nothing here reads or changes what the
//! model computes.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once};
use std::time::Duration;

static ON: AtomicBool = AtomicBool::new(false);
static START: Once = Once::new();
static BUF: Mutex<Vec<Mark>> = Mutex::new(Vec::new());

#[derive(Clone, Copy)]
struct Mark {
    ns: u64,
    role: &'static str,
    tag: &'static str,
    arg: u64,
}

thread_local! {
    static ROLE: Cell<&'static str> = const { Cell::new("eng") };
}

/// Names the calling thread's role in the marks it makes.
pub fn set_role(role: &'static str) {
    ROLE.with(|c| c.set(role));
}

/// Whether marks are being recorded.
#[inline]
pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// `CLOCK_MONOTONIC` in nanoseconds.
pub fn now_ns() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `ts` is a valid out-pointer for the duration of the call.
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// Reads `REXL3_TP_TRACE` once and, when it is `1`, starts recording (and the flusher). `who` is `"L"` or `"F"`.
pub fn start_from_env(who: &'static str) {
    START.call_once(|| {
        if std::env::var("REXL3_TP_TRACE").map(|v| v == "1").unwrap_or(false) {
            BUF.lock().unwrap_or_else(|p| p.into_inner()).reserve(1 << 16);
            ON.store(true, Ordering::Release);
            let _ = std::thread::Builder::new().name("tp-trace".into()).spawn(move || loop {
                std::thread::sleep(Duration::from_millis(250));
                flush(who);
            });
        }
    });
}

fn flush(who: &str) {
    let marks: Vec<Mark> = std::mem::take(&mut *BUF.lock().unwrap_or_else(|p| p.into_inner()));
    if marks.is_empty() {
        return;
    }
    let mut out = String::with_capacity(marks.len() * 40);
    for m in marks {
        use std::fmt::Write;
        let _ = writeln!(out, "TPTRACE {who} {} {} {} {}", m.ns, m.role, m.tag, m.arg);
    }
    use std::io::Write;
    let _ = std::io::stderr().lock().write_all(out.as_bytes());
}

/// Records a mark (a no-op unless tracing is on).
#[inline]
pub fn mark(tag: &'static str, arg: u64) {
    if !ON.load(Ordering::Relaxed) {
        return;
    }
    let m = Mark { ns: now_ns(), role: ROLE.with(|c| c.get()), tag, arg };
    BUF.lock().unwrap_or_else(|p| p.into_inner()).push(m);
}
