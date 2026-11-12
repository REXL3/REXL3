//! The helpers of the crate's build script (`build.rs`), which `cargo test` does not otherwise compile: how many nvcc
//! processes run at once, the bounded pool that runs them, and the host-only flag filter.

#[allow(dead_code)]
#[path = "../build.rs"]
mod build_script;

use build_script::{host_flags, nvcc_jobs, run_bounded};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

#[test]
fn nvcc_jobs_follows_cargo_jobs() {
    assert_eq!(nvcc_jobs(Some("6")), 6);
    assert_eq!(nvcc_jobs(Some(" 1 ")), 1);
}

#[test]
fn nvcc_jobs_falls_back_to_the_cpu_count() {
    let cpus = std::thread::available_parallelism().map_or(4, |n| n.get());
    for v in [None, Some("0"), Some("lots"), Some("")] {
        assert_eq!(nvcc_jobs(v), cpus, "{v:?}");
    }
}

/// Runs `n` tasks of 20 ms on `jobs` slots; returns the peak number running at once and how often each index ran.
fn peak(jobs: usize, n: usize) -> (usize, Vec<usize>) {
    let running = AtomicUsize::new(0);
    let peak = AtomicUsize::new(0);
    let ran = Mutex::new(vec![0; n]);
    let failed = run_bounded(jobs, n, |i| {
        let now = running.fetch_add(1, Ordering::SeqCst) + 1;
        peak.fetch_max(now, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(20));
        ran.lock().unwrap()[i] += 1;
        running.fetch_sub(1, Ordering::SeqCst);
        true
    });
    assert!(failed.is_empty());
    (peak.into_inner(), ran.into_inner().unwrap())
}

#[test]
fn the_pool_never_runs_more_than_jobs_at_once_and_runs_each_task_once() {
    for (jobs, n) in [(1, 5), (3, 12), (4, 73), (16, 5)] {
        let (p, ran) = peak(jobs, n);
        assert!(p <= jobs, "jobs {jobs}: {p} ran at once");
        assert_eq!(ran, vec![1; n], "jobs {jobs}");
    }
    // the slots are actually used: 12 tasks of 20 ms on 3 slots overlap
    assert!(peak(3, 12).0 >= 2);
}

#[test]
fn a_failure_is_reported_and_stops_new_tasks() {
    let started = AtomicUsize::new(0);
    let failed = run_bounded(1, 10, |i| {
        started.fetch_add(1, Ordering::SeqCst);
        i != 3
    });
    assert_eq!(failed, vec![3]);
    assert_eq!(started.into_inner(), 4, "nothing starts after the failing task on one slot");
    assert!(run_bounded(4, 0, |_| false).is_empty());
}

fn v(s: &str) -> Vec<String> {
    s.split_whitespace().map(str::to_string).collect()
}

#[test]
fn aarch64_keeps_every_kernel_flag() {
    let f = v("-Xcompiler -mno-outline-atomics --fmad=false");
    assert_eq!(host_flags(f.clone(), "aarch64"), f);
}

#[test]
fn another_arch_drops_only_the_aarch64_only_pair() {
    let f = v("--fmad=false -Xcompiler -mno-outline-atomics -Xcompiler -fPIC --expt-relaxed-constexpr");
    assert_eq!(host_flags(f, "x86_64"), v("--fmad=false -Xcompiler -fPIC --expt-relaxed-constexpr"));
    // a bare option (not a host-compiler option) and a trailing -Xcompiler are left alone
    let odd = v("-mno-outline-atomics -Xcompiler");
    assert_eq!(host_flags(odd.clone(), "x86_64"), odd);
}

#[test]
fn the_roce_shim_keeps_its_flag_on_aarch64_only() {
    // the kernel file that asks for the AArch64-only host option
    let first = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/roce_shim.cu")).unwrap();
    let flags = v(first.lines().next().unwrap().strip_prefix("// gb-nvcc:").unwrap());
    let has = |f: &[String]| f.iter().any(|x| x == "-mno-outline-atomics");
    assert!(has(&flags));
    assert_eq!(host_flags(flags.clone(), "aarch64"), flags);
    assert!(!has(&host_flags(flags, "x86_64")));
}
