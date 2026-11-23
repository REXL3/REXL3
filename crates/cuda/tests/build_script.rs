//! The helpers of the crate's build script (`build.rs`), which `cargo test` does not otherwise compile: how many nvcc
//! processes run at once, the bounded pool that runs them, and the host-only flag filter.

#[allow(dead_code)]
#[path = "../build.rs"]
mod build_script;

use build_script::{
    host_flags, initial_kernel_hash, invalidate_stamp_before_compile, kernel_build_stamp, kernel_object_is_fresh, nvcc_jobs,
    run_bounded, NVCC_FLAGS,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

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

#[test]
fn kernel_hash_excludes_nvcc_version_but_the_build_stamp_includes_it() {
    let base = initial_kernel_hash("sm_121", NVCC_FLAGS);
    assert_eq!(base, initial_kernel_hash("sm_121", NVCC_FLAGS));
    assert_ne!(base, initial_kernel_hash("sm_120", NVCC_FLAGS));
    assert_ne!(base, initial_kernel_hash("sm_121", &["-c", "-O2"]));

    let stamp = kernel_build_stamp("sm_121", NVCC_FLAGS, "13.0.88", base);
    assert!(stamp.contains("arch=sm_121\n"));
    assert!(stamp.contains(&format!("nvcc_flags={}\n", NVCC_FLAGS.join(" "))));
    assert!(stamp.contains("nvcc_version=13.0.88\n"));
    assert!(stamp.contains(&format!("kernel_hash={base:016x}\n")));
    assert_ne!(stamp, kernel_build_stamp("sm_120", NVCC_FLAGS, "13.0.88", base));
    assert_ne!(stamp, kernel_build_stamp("sm_121", NVCC_FLAGS, "13.0.89", base));
}

#[test]
fn changed_identity_is_stale_and_an_aborted_arch_switch_leaves_no_stamp() {
    let d = std::env::temp_dir().join(format!("rexl3-kernel-stamp-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let src = d.join("kernel.cu");
    let obj = d.join("kernel.o");
    let stamp_path = d.join("rexl3-kernel-build.stamp");
    std::fs::write(&src, "kernel").unwrap();
    std::fs::write(&obj, "sm_121").unwrap();
    let old = SystemTime::now() - Duration::from_secs(60);
    let times = std::fs::FileTimes::new().set_modified(old);
    std::fs::File::options().write(true).open(&src).unwrap().set_times(times).unwrap();

    let hash = initial_kernel_hash("sm_121", NVCC_FLAGS);
    let stamp_121 = kernel_build_stamp("sm_121", NVCC_FLAGS, "13.0.88", hash);
    let stamp_120 = kernel_build_stamp("sm_120", NVCC_FLAGS, "13.0.88", hash);
    std::fs::write(&stamp_path, &stamp_121).unwrap();
    assert!(kernel_object_is_fresh(&src, &obj, None, false));

    // The object is newer than its source, but an arch change still forces compilation.
    let identity_changed = std::fs::read_to_string(&stamp_path).unwrap() != stamp_120;
    assert!(identity_changed);
    assert!(!kernel_object_is_fresh(&src, &obj, None, identity_changed));

    // Simulate compiling one sm_120 object, then aborting before the other objects/archive complete.
    invalidate_stamp_before_compile(&stamp_path, true);
    assert!(!stamp_path.exists(), "the old identity must be gone before the first object is written");
    std::fs::write(&obj, "sm_120").unwrap();
    assert!(!stamp_path.exists(), "a failed build must not commit a new identity");

    // Switching back cannot trust the mixed object directory: the missing stamp makes everything stale.
    let identity_changed = std::fs::read_to_string(&stamp_path).ok().as_deref() != Some(stamp_121.as_str());
    assert!(identity_changed);
    assert!(!kernel_object_is_fresh(&src, &obj, None, identity_changed));
    let _ = std::fs::remove_dir_all(&d);
}
