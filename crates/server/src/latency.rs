//! Apache-2.0. Fixed, unlabelled histograms: 1ms * 2^k (k=0..23), +Inf.
//! Ceiling buckets (not floor-log2); seconds exposed, saturating nanosecond sum.
use std::fmt::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::Duration;

pub struct Hist {
    bins: [AtomicU64; 25],
    sum_ns: AtomicU64,
    // Test/performance gate only: recording is enabled by default, never environment-controlled.
    disabled: AtomicBool,
}
impl Default for Hist {
    fn default() -> Self {
        Self { bins: std::array::from_fn(|_| AtomicU64::new(0)), sum_ns: AtomicU64::new(0), disabled: AtomicBool::new(false) }
    }
}
impl Hist {
    pub fn record(&self, d: Duration) {
        self.record_ns(d.as_nanos().min(u64::MAX as u128) as u64);
    }
    pub fn record_ns(&self, ns: u64) {
        if self.disabled.load(Relaxed) {
            return;
        }
        let ms = ns / 1_000_000 + u64::from(ns % 1_000_000 != 0);
        let index = if ms <= 1 { 0 } else { (64 - (ms - 1).leading_zeros() as usize).min(24) };
        self.bins[index].fetch_add(1, Relaxed);
        let _ = self.sum_ns.fetch_update(Relaxed, Relaxed, |n| Some(n.saturating_add(ns)));
    }
    /// For the interleaved CPU instrumentation gate; production always records.
    pub fn set_recording(&self, enabled: bool) {
        self.disabled.store(!enabled, Relaxed);
    }
    pub fn render(&self, out: &mut String, name: &str, help: &str) {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        // Count is derived from this same snapshot, so +Inf always equals count even during concurrent writes.
        let mut total = 0u64;
        for (i, bin) in self.bins.iter().enumerate() {
            total = total.saturating_add(bin.load(Relaxed));
            let le = if i == 24 { "+Inf".into() } else { format!("{:.3}", (1u64 << i) as f64 / 1000.0) };
            let _ = writeln!(out, "{name}_bucket{{le=\"{le}\"}} {total}");
        }
        let _ = writeln!(out, "{name}_sum {:.9}\n{name}_count {total}", self.sum_ns.load(Relaxed) as f64 / 1e9);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_ceiling_boundaries_seconds_minutes_and_infinity() {
        let h = Hist::default();
        for ns in [0, 1_000_000, 1_000_001, 2_000_000, 2_000_001, 300_000_000_000, u64::MAX] {
            h.record_ns(ns);
        }
        let mut t = String::new();
        h.render(&mut t, "test_seconds", "test");
        for s in [
            "le=\"0.001\"} 2",
            "le=\"0.002\"} 4",
            "le=\"0.004\"} 5",
            "le=\"524.288\"} 6",
            "le=\"+Inf\"} 7",
            "test_seconds_count 7",
        ] {
            assert!(t.contains(s), "{t}");
        }
        assert_eq!(h.sum_ns.load(Relaxed), u64::MAX);
    }
    #[test]
    fn empty_and_recording_off_are_zero() {
        let h = Hist::default();
        h.set_recording(false);
        h.record_ns(10);
        let mut t = String::new();
        h.render(&mut t, "test_seconds", "test");
        assert!(t.contains("test_seconds_count 0"));
        assert!(t.contains("le=\"+Inf\"} 0"));
    }
}
