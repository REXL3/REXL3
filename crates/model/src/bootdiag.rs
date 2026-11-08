//! Kernel-side counters around one boot step, for finding out why a step was slow (`REXL3_BOOT_DIAG=1`; off by default, and then
//! [`Window::open`] is one environment read).
//!
//! A window records the wall time, the process's page faults (`getrusage`), the system's memory-pressure stall time
//! (`/proc/pressure/memory`, microseconds some or all tasks waited on reclaim, compaction or refaults) and the counters of
//! `/proc/vmstat` that say what the allocator did meanwhile (direct reclaim, compaction, transparent huge pages), and prints the
//! deltas as one `rexl3 event=boot_window step=...` line when it closes. Reading those files costs well under a millisecond.
//! The vmstat and pressure figures are system-wide: they are only a diagnosis on a box that is otherwise quiet.

use std::time::Instant;

/// `/proc/vmstat` counters worth reading around an allocation-heavy step (a name ending in `_` stands for every counter with that prefix,
/// summed).
const KEYS: [&str; 14] = [
    "allocstall_",
    "compact_stall",
    "compact_fail",
    "compact_success",
    "pgscan_direct",
    "pgsteal_direct",
    "pgscan_kswapd",
    "pgmigrate_success",
    "thp_fault_alloc",
    "thp_fault_fallback",
    "workingset_refault_",
    "numa_hint_faults",
    "pgfault",
    "pgmajfault",
];

fn vmstat() -> Vec<(String, u64)> {
    let text = std::fs::read_to_string("/proc/vmstat").unwrap_or_default();
    let mut out: Vec<(String, u64)> = Vec::new();
    for line in text.lines() {
        let Some((k, v)) = line.split_once(' ') else { continue };
        let Some(key) = KEYS.iter().find(|p| if p.ends_with('_') { k.starts_with(**p) } else { k == **p }) else { continue };
        let v: u64 = v.trim().parse().unwrap_or(0);
        match out.iter_mut().find(|(n, _)| n == key) {
            Some(slot) => slot.1 += v,
            None => out.push((key.to_string(), v)),
        }
    }
    out
}

/// Microseconds stalled so far: (some, full).
fn pressure() -> (u64, u64) {
    let text = std::fs::read_to_string("/proc/pressure/memory").unwrap_or_default();
    let total = |kind: &str| {
        text.lines()
            .find(|l| l.starts_with(kind))
            .and_then(|l| l.split_whitespace().find_map(|w| w.strip_prefix("total=")))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    (total("some"), total("full"))
}

/// (minor faults, major faults) of this process so far.
fn faults() -> (i64, i64) {
    // SAFETY: getrusage fills the struct it is given; all-zero is a valid value for it
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        (ru.ru_minflt, ru.ru_majflt)
    }
}

/// One measured step.
pub struct Window {
    start: Instant,
    faults: (i64, i64),
    psi: (u64, u64),
    vm: Vec<(String, u64)>,
}

impl Window {
    /// Start a window when `REXL3_BOOT_DIAG` is set to something other than `0`.
    pub fn open() -> Option<Self> {
        match std::env::var("REXL3_BOOT_DIAG") {
            Ok(v) if !v.is_empty() && v != "0" => {}
            _ => return None,
        }
        Some(Self { start: Instant::now(), faults: faults(), psi: pressure(), vm: vmstat() })
    }

    /// Print the deltas since [`Self::open`] as one line named `step`.
    pub fn close(self, step: &str) {
        let ms = self.start.elapsed().as_secs_f64() * 1e3;
        let (f, p, vm) = (faults(), pressure(), vmstat());
        let mut line = format!(
            "rexl3 event=boot_window step={step} ms={ms:.1} minflt={} majflt={} psi_some_us={} psi_full_us={}",
            f.0 - self.faults.0,
            f.1 - self.faults.1,
            p.0.saturating_sub(self.psi.0),
            p.1.saturating_sub(self.psi.1)
        );
        for (k, v) in &vm {
            let before = self.vm.iter().find(|(n, _)| n == k).map_or(0, |x| x.1);
            if *v != before {
                line.push_str(&format!(" {}={}", k.trim_end_matches('_'), v.saturating_sub(before)));
            }
        }
        eprintln!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_is_off_unless_asked_for() {
        // the test harness does not set the variable
        if std::env::var("REXL3_BOOT_DIAG").is_err() {
            assert!(Window::open().is_none());
        }
    }

    #[test]
    fn the_counters_parse_on_this_kernel() {
        // on Linux /proc/vmstat has pgfault; elsewhere the readers return empty values and nothing panics
        let vm = vmstat();
        if std::path::Path::new("/proc/vmstat").exists() {
            assert!(vm.iter().any(|(k, _)| k == "pgfault"));
        }
        let (a, b) = faults();
        assert!(a >= 0 && b >= 0);
        let _ = pressure();
    }
}
