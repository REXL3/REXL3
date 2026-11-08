//! Only the explicit CPU-test profile excludes measurement tests that declare raw CUDA FFI.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(rexl3_no_cuda)");
    println!("cargo::rerun-if-env-changed=REXL3_NO_CUDA");
    if std::env::var_os("REXL3_NO_CUDA").is_some() {
        println!("cargo::rustc-cfg=rexl3_no_cuda");
    }
}
