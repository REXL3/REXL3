//! Compiles every `csrc/*.cu` with nvcc into one static library and links it with cudart.
//!
//! No nvcc (CI on x86, a laptop) means a stub build: the crate still compiles, `cfg(gb_no_cuda)` is set, and GPU
//! tests skip. `REXL3_CUDA_ARCH` overrides the target (default `sm_121`, the GB10); `REXL3_NO_CUDA` forces the stub build.
//! The old names `GB_CUDA_ARCH` and `GB_NO_CUDA` still work and print a warning. A source file can request extra nvcc
//! flags on its first line: `// gb-nvcc: --fmad=false`. At most Cargo's job count (`-j`, passed as `NUM_JOBS`) nvcc
//! processes run at once.
//!
//! The build identity is exported to the crate as `REXL3_NVCC_VERSION`, `REXL3_CUDA_ARCH` and `REXL3_KERNEL_HASH` (FNV-1a
//! over every file in `csrc/`, the arch and each file's flags), so a running server can say exactly which kernels it runs.
//!
//! nvcc compiles a staged copy of `csrc/`, not the checkout's files (`kernel_build.rs`): the copy's path and modification
//! times are the same in every checkout, which is what makes one commit build to one file on every node.

#[path = "kernel_build.rs"]
pub mod kernel_build;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::{env, fs, thread};

/// Flags every kernel file is compiled with (part of the kernel hash).
pub const NVCC_FLAGS: &[&str] = &["-c", "-O3", "--std=c++17", "-Xcompiler", "-fPIC", "-lineinfo", "-I", "csrc"];

/// `REXL3_<name>`, else the old `GB_<name>` with a warning.
fn setting(name: &str) -> Option<String> {
    if let Ok(v) = env::var(format!("REXL3_{name}")) {
        return Some(v);
    }
    let old = env::var(format!("GB_{name}")).ok();
    if old.is_some() {
        println!("cargo:warning=GB_{name} is deprecated: set REXL3_{name}");
    }
    old
}

fn fnv1a(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

/// The arch-and-flags prefix of the kernel hash. Source names and bytes are folded in after this value.
/// Keep this exactly in sync with `tools/release_scope.py::kernel_hash`; the compiler identity is stamp-only.
pub fn initial_kernel_hash(arch: &str, flags: &[&str]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    fnv1a(&mut hash, arch.as_bytes());
    fnv1a(&mut hash, flags.join(" ").as_bytes());
    hash
}

/// The persisted identity that decides whether every object must be recompiled.
pub fn kernel_build_stamp(arch: &str, flags: &[&str], nvcc_version: &str, kernel_hash: u64) -> String {
    format!("arch={arch}\nnvcc_flags={}\nnvcc_version={nvcc_version}\nkernel_hash={kernel_hash:016x}\n", flags.join(" "))
}

/// Whether an object can be reused. A changed identity makes every object stale, even with newer mtimes.
pub fn kernel_object_is_fresh(
    source: &Path,
    object: &Path,
    header_mtime: Option<std::time::SystemTime>,
    identity_changed: bool,
) -> bool {
    !identity_changed
        && match (fs::metadata(object).and_then(|m| m.modified()).ok(), fs::metadata(source).and_then(|m| m.modified()).ok()) {
            (Some(om), Some(sm)) => om >= sm && header_mtime.is_none_or(|hm| om >= hm),
            _ => false,
        }
}

/// Invalidate the old identity before touching any object. A failed build must not leave a valid stamp behind.
pub fn invalidate_stamp_before_compile(stamp: &Path, compiling: bool) {
    if compiling {
        let _ = fs::remove_file(stamp);
    }
}

/// `13.0.88` from `nvcc --version` ("Cuda compilation tools, release 13.0, V13.0.88").
fn nvcc_version(nvcc: &Path) -> String {
    let out =
        Command::new(nvcc).arg("--version").output().map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
    out.split_whitespace().rev().find_map(|w| w.strip_prefix('V')).unwrap_or("unknown").to_string()
}

pub fn find_cuda() -> Option<PathBuf> {
    if setting("NO_CUDA").is_some() {
        return None; // force the stub build (used to test machines without a GPU toolchain)
    }
    if let Ok(h) = env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")) {
        let p = PathBuf::from(h);
        if p.join("bin/nvcc").exists() {
            return Some(p);
        }
    }
    if let Some(paths) = env::var_os("PATH") {
        for dir in env::split_paths(&paths) {
            let candidate = dir.join("nvcc");
            if candidate.exists() {
                if let Some(root) = dir.parent() {
                    return Some(root.to_path_buf());
                }
            }
        }
    }
    for p in ["/usr/local/cuda", "/usr/local/cuda-13.0", "/usr/local/cuda-13"] {
        let p = PathBuf::from(p);
        if p.join("bin/nvcc").exists() {
            return Some(p);
        }
    }
    None
}

/// How many nvcc processes may run at once: Cargo's job count (`NUM_JOBS`, which Cargo sets from `-j`), else the CPU
/// count. One nvcc of a large kernel file can take more than 1 GB, so a clean build that started every file at once could
/// exhaust the memory of a node that is also serving a model. The count changes scheduling only, never the objects.
pub fn nvcc_jobs(cargo_jobs: Option<&str>) -> usize {
    match cargo_jobs.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(n) if n > 0 => n,
        _ => thread::available_parallelism().map_or(4, |n| n.get()),
    }
}

/// Runs `run(0..n)` on at most `jobs` threads at once; `run` returns false on failure. After the first failure no new
/// index starts (the running ones finish). Returns the failed indices, sorted.
pub fn run_bounded<F: Fn(usize) -> bool + Sync>(jobs: usize, n: usize, run: F) -> Vec<usize> {
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let failed = Mutex::new(Vec::new());
    thread::scope(|s| {
        for _ in 0..jobs.clamp(1, n.max(1)) {
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    if !run(i) {
                        stop.store(true, Ordering::Relaxed);
                        failed.lock().unwrap().push(i);
                    }
                }
            });
        }
    });
    let mut failed = failed.into_inner().unwrap();
    failed.sort_unstable();
    failed
}

/// Host-compiler options a kernel file may request that GCC accepts on one target architecture only, with that
/// architecture (`-mno-outline-atomics` exists on AArch64 only; `roce_shim.cu` asks for it).
const HOST_ONLY: &[(&str, &str)] = &[("-mno-outline-atomics", "aarch64")];

/// `flags` without the `-Xcompiler <option>` pairs whose option is in [`HOST_ONLY`] for another architecture than
/// `target_arch`, so a compile-only build on another host does not stop at them. On the GB10 (aarch64) nothing is
/// dropped. The kernel hash covers each file's bytes, first line included, so it does not depend on this.
pub fn host_flags(flags: Vec<String>, target_arch: &str) -> Vec<String> {
    let foreign = |opt: &String| HOST_ONLY.iter().any(|&(o, arch)| opt == o && arch != target_arch);
    let mut out = Vec::with_capacity(flags.len());
    let mut it = flags.into_iter().peekable();
    while let Some(f) = it.next() {
        if f == "-Xcompiler" && it.peek().is_some_and(foreign) {
            it.next();
            continue;
        }
        out.push(f);
    }
    out
}

fn extra_flags(src: &Path) -> Vec<String> {
    let text = fs::read_to_string(src).unwrap_or_default();
    let flags = match text.lines().next().and_then(|l| l.strip_prefix("// gb-nvcc:")) {
        Some(f) => f.split_whitespace().map(str::to_string).collect(),
        None => vec![],
    };
    host_flags(flags, &env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default())
}

fn main() {
    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernel_build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    for v in ["CUDA_ARCH", "NO_CUDA"] {
        println!("cargo:rerun-if-env-changed=REXL3_{v}");
        println!("cargo:rerun-if-env-changed=GB_{v}");
    }
    println!("cargo::rustc-check-cfg=cfg(gb_no_cuda)");
    let Some(cuda) = find_cuda() else {
        println!("cargo:rustc-cfg=gb_no_cuda");
        println!("cargo:rustc-env=REXL3_NVCC_VERSION=none");
        println!("cargo:rustc-env=REXL3_CUDA_ARCH=none");
        println!("cargo:rustc-env=REXL3_KERNEL_HASH=none");
        println!("cargo:warning=nvcc not found: building gb-cuda without GPU support");
        return;
    };
    let nvcc = cuda.join("bin/nvcc");
    let arch = setting("CUDA_ARCH").unwrap_or_else(|| "sm_121".into());
    let nvcc_version = nvcc_version(&nvcc);
    let mut hash = initial_kernel_hash(&arch, NVCC_FLAGS);
    let mut all: Vec<PathBuf> = fs::read_dir("csrc").expect("csrc").filter_map(|e| e.ok().map(|e| e.path())).collect();
    all.sort();
    for p in &all {
        fnv1a(&mut hash, p.file_name().unwrap().to_string_lossy().as_bytes());
        fnv1a(&mut hash, &fs::read(p).expect("read a kernel source"));
    }
    println!("cargo:rustc-env=REXL3_NVCC_VERSION={nvcc_version}");
    println!("cargo:rustc-env=REXL3_CUDA_ARCH={arch}");
    println!("cargo:rustc-env=REXL3_KERNEL_HASH={hash:016x}");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let stamp_path = out.join("rexl3-kernel-build.stamp");
    let stage = kernel_build::stage_or_warn("cuda", Path::new("csrc"));
    // Where the sources were compiled from is part of the identity: objects made in place, by an older build script or
    // after a failed staging, carry this checkout's path and are not reused once staging works.
    let sources = stage.as_deref().map_or("in place".to_string(), |d| d.display().to_string());
    let stamp = format!("{}sources={sources}\n", kernel_build_stamp(&arch, NVCC_FLAGS, &nvcc_version, hash));
    let identity_changed = fs::read_to_string(&stamp_path).ok().as_deref() != Some(stamp.as_str());
    let mut srcs: Vec<PathBuf> = fs::read_dir("csrc")
        .expect("csrc")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "cu").unwrap_or(false))
        .collect();
    srcs.sort();
    let hdr_mtime = all
        .iter()
        .filter(|p| p.extension().map(|x| x == "cuh").unwrap_or(false))
        .filter_map(|p| fs::metadata(p).and_then(|m| m.modified()).ok())
        .max();
    let objs: Vec<PathBuf> = srcs.iter().map(|s| out.join(s.file_stem().unwrap()).with_extension("o")).collect();
    let mut stale = Vec::new();
    for (s, obj) in srcs.iter().zip(objs.iter()) {
        if !kernel_object_is_fresh(s, obj, hdr_mtime, identity_changed) {
            stale.push((s, obj));
        }
    }
    invalidate_stamp_before_compile(&stamp_path, !stale.is_empty());
    let cuda_bin = cuda.join("bin");
    let compile = |s: &Path, obj: &Path| {
        let extra = extra_flags(s);
        let file = s.file_name().expect("a kernel source has a file name").to_string_lossy();
        kernel_build::Compile {
            nvcc: &nvcc,
            cuda_bin: &cuda_bin,
            stage: stage.as_deref(),
            flags: NVCC_FLAGS,
            arch: &arch,
            extra: &extra,
            file: &file,
            object: obj,
        }
        .run()
    };
    let jobs = nvcc_jobs(env::var("NUM_JOBS").ok().as_deref());
    let failed = run_bounded(jobs, stale.len(), |i| compile(stale[i].0, stale[i].1));
    let names: Vec<String> = failed.iter().map(|&i| stale[i].0.display().to_string()).collect();
    assert!(names.is_empty(), "nvcc failed on {}", names.join(", "));
    let lib = out.join("libgbkernels.a");
    let _ = fs::remove_file(&lib);
    let st = Command::new("ar").arg("crs").arg(&lib).args(&objs).status().expect("run ar");
    assert!(st.success(), "ar failed");
    // Commit the identity only after every object compiled and the archive was rebuilt successfully.
    if !stale.is_empty() {
        fs::write(&stamp_path, &stamp).expect("write kernel build stamp");
    }
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=gbkernels");
    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
