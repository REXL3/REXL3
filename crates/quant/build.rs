//! Compiles `csrc/*.cu` (the quantizer's own kernels) with nvcc into a static library and links it with cudart,
//! cuBLAS and cuSOLVER. cuSOLVER is linked here only: the serving binary does not depend on this crate.
//!
//! The CUDA rules follow gb-cuda's build script: `REXL3_NO_CUDA` (or no nvcc) gives a stub build with
//! `cfg(gb_quant_no_cuda)`, `REXL3_CUDA_ARCH` overrides the target (default `sm_121`), every file gets gb-cuda's base
//! nvcc flags, and a file can add its own on its first line: `// gb-nvcc: --fmad=false` (the core's kernels).
//! `exl3_trellis.cu` keeps the flags it had in gb-cuda, so it compiles to the same SASS.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

/// gb-cuda's base flags (crates/cuda/build.rs); per-file additions come from the file's first line.
const NVCC_FLAGS: &[&str] = &["-c", "-O3", "--std=c++17", "-Xcompiler", "-fPIC", "-lineinfo", "-I", "csrc"];

/// The flags a source asks for on its first line (`// gb-nvcc: ...`).
fn extra_flags(src: &Path) -> Vec<String> {
    let text = fs::read_to_string(src).unwrap_or_default();
    match text.lines().next().and_then(|l| l.strip_prefix("// gb-nvcc:")) {
        Some(f) => f.split_whitespace().map(str::to_string).collect(),
        None => vec![],
    }
}

fn setting(name: &str) -> Option<String> {
    env::var(format!("REXL3_{name}")).ok().or_else(|| env::var(format!("GB_{name}")).ok())
}

fn find_cuda() -> Option<PathBuf> {
    if setting("NO_CUDA").is_some() {
        return None;
    }
    if let Ok(h) = env::var("CUDA_HOME").or_else(|_| env::var("CUDA_PATH")) {
        let p = PathBuf::from(h);
        if p.join("bin/nvcc").exists() {
            return Some(p);
        }
    }
    if let Some(paths) = env::var_os("PATH") {
        for dir in env::split_paths(&paths) {
            if dir.join("nvcc").exists() {
                if let Some(root) = dir.parent() {
                    return Some(root.to_path_buf());
                }
            }
        }
    }
    ["/usr/local/cuda", "/usr/local/cuda-13.0", "/usr/local/cuda-13"]
        .iter()
        .map(PathBuf::from)
        .find(|p| p.join("bin/nvcc").exists())
}

fn main() {
    println!("cargo::rustc-check-cfg=cfg(gb_quant_no_cuda)");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=csrc");
    for v in ["REXL3_NO_CUDA", "GB_NO_CUDA", "REXL3_CUDA_ARCH", "CUDA_HOME", "CUDA_PATH"] {
        println!("cargo:rerun-if-env-changed={v}");
    }
    let Some(cuda) = find_cuda() else {
        println!("cargo:rustc-cfg=gb_quant_no_cuda");
        return;
    };
    let arch = setting("CUDA_ARCH").unwrap_or_else(|| "sm_121".to_string());
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let nvcc = cuda.join("bin/nvcc");
    let mut sources: Vec<PathBuf> =
        fs::read_dir("csrc").map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default();
    sources.retain(|p| p.extension().is_some_and(|e| e == "cu"));
    sources.sort();
    let mut objects = Vec::new();
    for src in &sources {
        println!("cargo:rerun-if-changed={}", src.display());
        let obj = out.join(src.file_stem().unwrap()).with_extension("o");
        let status = Command::new(&nvcc)
            .args(NVCC_FLAGS)
            .arg(format!("-arch={arch}"))
            .args(extra_flags(src))
            .arg("-o")
            .arg(&obj)
            .arg(src)
            .status()
            .expect("run nvcc");
        assert!(status.success(), "nvcc failed on {}", src.display());
        objects.push(obj);
    }
    if !objects.is_empty() {
        let lib = out.join("libgbquantkernels.a");
        let _ = fs::remove_file(&lib);
        let status = Command::new("ar").arg("crs").arg(&lib).args(&objects).status().expect("run ar");
        assert!(status.success(), "ar failed");
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static=gbquantkernels");
    }
    link_cuda_libs(&cuda);
}

fn link_cuda_libs(cuda: &Path) {
    println!("cargo:rustc-link-search=native={}", cuda.join("lib64").display());
    for lib in ["cusolver", "cublas", "cublasLt", "cudart", "stdc++"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
