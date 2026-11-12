//! Process-wide counters of the Bailing-3.1 prefix cache that the engine's own `PrefixStats` cannot see: a snapshot the backend declined
//! to take (the host's `MemAvailable` was below the headroom a snapshot needs: `BailingModel::snapshot_headroom`), and a snapshot it took
//! and lost to an error. Both are exactness-safe (a skipped capture is a cold prefill) but they make the hit rate depend on the host's
//! state, so they are counted and logged (`event=ling31_snapshot_skipped`, `event=ling31_snapshot_failed`, each with its running total)
//! instead of passing silently.
//!
//! The counters are plain process-wide atomics: the backend increments them from its engine thread; [`skipped_low_memory`] and
//! [`capture_failed`] read them. They are not on `/metrics`: that is a change in the server crate, which every family shares.

use std::sync::atomic::{AtomicU64, Ordering};

static SKIPPED_LOW_MEMORY: AtomicU64 = AtomicU64::new(0);
static CAPTURE_FAILED: AtomicU64 = AtomicU64::new(0);

/// A skip is logged for the first one and then for every `LOG_EVERY`-th, so a node that sits under the guard does not fill its log.
pub const LOG_EVERY: u64 = 16;

/// Cuts declined because `MemAvailable` was under the headroom a snapshot needs.
pub fn skipped_low_memory() -> u64 {
    SKIPPED_LOW_MEMORY.load(Ordering::Relaxed)
}

/// Snapshots that were started and failed (the cut is lost, the prompt continued cold).
pub fn capture_failed() -> u64 {
    CAPTURE_FAILED.load(Ordering::Relaxed)
}

/// Whether the `n`-th event of a counter is logged.
fn logged(n: u64) -> bool {
    n == 1 || n % LOG_EVERY == 0
}

/// `cuts` snapshot cuts of one prefill were declined for lack of memory: `avail` is what `/proc/meminfo` said then, `needed` what the
/// snapshot of the first declined cut needs (its state, its MLA copies and 1 GiB of headroom). Returns the line it logged, if any.
pub fn note_skipped_low_memory(cuts: u64, avail: u64, needed: u64) -> Option<String> {
    let total = SKIPPED_LOW_MEMORY.fetch_add(cuts, Ordering::Relaxed) + cuts;
    // log when this call crossed a logging point (it may have added several at once)
    let before = total - cuts;
    let crossed = (before == 0 && total > 0) || before / LOG_EVERY != total / LOG_EVERY;
    crossed.then(|| {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let line = format!(
            "rexl3 level=warn event=ling31_snapshot_skipped reason=mem_available_below_guard cuts={cuts} avail_gib={:.2} needed_gib={:.2} total={total}",
            gib(avail),
            gib(needed)
        );
        eprintln!("{line}");
        line
    })
}

/// A snapshot failed after it was started (a CUDA or host error): the cut is released. Returns the line it logged, if any.
pub fn note_capture_failed(error: &str) -> Option<String> {
    let total = CAPTURE_FAILED.fetch_add(1, Ordering::Relaxed) + 1;
    logged(total).then(|| {
        let line = format!("rexl3 level=warn event=ling31_snapshot_failed total={total} error=\"{}\"", error.replace('"', "'"));
        eprintln!("{line}");
        line
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // The counters are process-wide: one test owns them, in sequence, and reads them as deltas.
    #[test]
    fn a_skip_is_counted_and_the_first_and_every_sixteenth_are_logged() {
        let base = skipped_low_memory();
        let gib = 1u64 << 30;
        let first = note_skipped_low_memory(1, 3 * gib / 2, 17 * gib / 10);
        assert_eq!(skipped_low_memory(), base + 1);
        if base == 0 {
            let l = first.expect("the first skip is logged");
            assert!(
                l.contains("event=ling31_snapshot_skipped") && l.contains("avail_gib=1.50") && l.contains("needed_gib=1.70"),
                "{l}"
            );
        }
        // a run of single skips logs exactly at multiples of LOG_EVERY (and at the first)
        let mut logged_at = Vec::new();
        for _ in 0..(3 * LOG_EVERY) {
            let before = skipped_low_memory();
            if note_skipped_low_memory(1, gib, 2 * gib).is_some() {
                logged_at.push(before + 1);
            }
        }
        assert_eq!(logged_at.len(), 3, "{logged_at:?}");
        assert!(logged_at.iter().all(|n| n % LOG_EVERY == 0), "{logged_at:?}");
        // a call that adds several at once and crosses a logging point logs once
        let b = skipped_low_memory();
        let next_point = (b / LOG_EVERY + 1) * LOG_EVERY;
        let step = next_point - b + 1;
        assert!(note_skipped_low_memory(step, gib, 2 * gib).is_some());
        let f = capture_failed();
        note_capture_failed("cuda \"oops\"");
        assert_eq!(capture_failed(), f + 1);
    }

    #[test]
    fn a_failure_is_logged_the_first_time_and_every_sixteenth() {
        assert!(logged(1) && logged(16) && logged(32));
        assert!(!logged(2) && !logged(15) && !logged(17) && !logged(31));
    }
}
