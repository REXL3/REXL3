//! How a process ends on a FATAL error, and the one place that decides it.
//!
//! A single-process server ends with `std::process::exit`: the atexit handlers run, among them the CUDA runtime's own teardown, and
//! the device memory is returned in an orderly way. A member of a tensor-parallel group must not: by the time it is fatal, its
//! communicator was aborted (`ncclCommAbort` runs on a thread of its own when the group is lost, and it has been seen to block,
//! `docs/NEMOTRON_TP.md` 12.3 B2), its engine's device state was leaked or dropped behind kernels a lost peer will never join, and the
//! runtime's exit-time teardown of that context can fault. Observed once on the real pair (2026-10-11, the end-to-end smoke): a leader
//! SIGTERMed after its group was lost logged `engine_failed` and `exit code=70` and then died of SIGSEGV (status 139) in the
//! teardown, so the supervisor read "a signal" where the documented answer is 70 (a lost group).
//!
//! A group's fatal path therefore ends with `_exit(code)`: stderr flushed first, no atexit handler and no destructor. The operating
//! system reclaims the memory (device memory is host memory on a GB10). `lockstep::exit_backstop` has always done exactly this.

use std::io::Write;

/// How the process ends on a fatal error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FatalExit {
    /// `std::process::exit`: atexit handlers and the CUDA runtime's teardown run. Every single-process family, as before.
    #[default]
    Exit,
    /// stdout and stderr flushed, then `_exit`: no atexit handler, no static destructor. Every rank of a tensor-parallel group.
    Immediate,
}

/// The decision, as a pure function: a tensor-parallel group ends without the exit-time teardown, everything else as before.
pub fn fatal_exit_mode(tp_group: bool) -> FatalExit {
    if tp_group {
        FatalExit::Immediate
    } else {
        FatalExit::Exit
    }
}

/// End the process with `code` the way `mode` says. Never returns.
pub fn end_process(code: i32, mode: FatalExit) -> ! {
    match mode {
        FatalExit::Exit => std::process::exit(code),
        FatalExit::Immediate => {
            // Rust's stdout is line-buffered and only the runtime's own exit cleanup flushes it, which `_exit` skips
            let _ = std::io::stdout().flush();
            let _ = std::io::stderr().flush();
            // SAFETY: `_exit` takes no pointers and does not return.
            unsafe { libc::_exit(code) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ends a child of this very test binary in each mode (the test below runs it): an atexit handler (what the CUDA runtime's exit-time
    /// teardown is) says whether it ran, and text without a newline says whether stdout was flushed.
    #[test]
    fn the_two_ways_of_ending_a_process_differ_in_the_atexit_handlers_and_both_flush() {
        if let Ok(mode) = std::env::var("REXL3_EXIT_TEST_CHILD") {
            extern "C" fn at_exit() {
                println!("ATEXIT-RAN");
            }
            // SAFETY: registers a plain function with the C runtime; it runs once, at `exit`.
            unsafe { libc::atexit(at_exit) };
            print!("flushed-before-exit");
            end_process(70, if mode == "immediate" { FatalExit::Immediate } else { FatalExit::Exit });
        }
        let me = std::env::current_exe().expect("the test binary");
        for (mode, atexit_runs) in [("immediate", false), ("exit", true)] {
            let out = std::process::Command::new(&me)
                .args([
                    "--exact",
                    "exit::tests::the_two_ways_of_ending_a_process_differ_in_the_atexit_handlers_and_both_flush",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env("REXL3_EXIT_TEST_CHILD", mode)
                .output()
                .expect("spawn the child");
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert_eq!(out.status.code(), Some(70), "{mode}: {stdout}");
            assert!(stdout.contains("flushed-before-exit"), "{mode}: stdout was not flushed: {stdout}");
            assert_eq!(stdout.contains("ATEXIT-RAN"), atexit_runs, "{mode}: atexit handlers: {stdout}");
        }
    }

    #[test]
    fn only_a_tensor_parallel_group_skips_the_exit_time_teardown() {
        assert_eq!(fatal_exit_mode(true), FatalExit::Immediate);
        assert_eq!(fatal_exit_mode(false), FatalExit::Exit, "every single-process family keeps `exit`");
        assert_eq!(FatalExit::default(), FatalExit::Exit);
    }
}
