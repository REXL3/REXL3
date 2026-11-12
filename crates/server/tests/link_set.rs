//! The serving binary's dynamic link set: `rexl3-serve` needs exactly the libraries pinned here, and never cuSOLVER
//! (which only the quantizer, `rexl3-quant` in crates/quant, links). A stub build (no CUDA) links no CUDA library at
//! all; it is checked for cuSOLVER only.

use std::process::Command;

/// `readelf -d` NEEDED entries of a GPU build, in link order.
const GPU_NEEDED: &[&str] = &[
    "libcudart.so.13",
    "libcublas.so.13",
    "libcublasLt.so.13",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "libm.so.6",
    "libc.so.6",
    "ld-linux-aarch64.so.1",
];

fn needed(path: &str) -> Option<Vec<String>> {
    let out = Command::new("readelf").arg("-d").arg(path).output().ok()?;
    assert!(out.status.success(), "readelf -d {path}: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).expect("readelf output");
    Some(
        text.lines()
            .filter(|l| l.contains("(NEEDED)"))
            .filter_map(|l| l.split('[').nth(1).and_then(|s| s.split(']').next()).map(str::to_string))
            .collect(),
    )
}

#[test]
fn rexl3_serve_links_the_pinned_libraries_and_never_cusolver() {
    let Some(libs) = needed(env!("CARGO_BIN_EXE_rexl3-serve")) else {
        eprintln!("readelf not found: link set not checked");
        return;
    };
    assert!(!libs.iter().any(|l| l.contains("cusolver")), "rexl3-serve links cuSOLVER: {libs:?}");
    if libs.iter().any(|l| l.starts_with("libcudart")) && cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        assert_eq!(libs, GPU_NEEDED, "rexl3-serve's link set changed");
    } else {
        assert!(!libs.iter().any(|l| l.contains("cublas") || l.contains("cudart")), "stub build links CUDA: {libs:?}");
    }
}
