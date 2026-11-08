//! The public embedding seam must remain usable in CPU-only builds.
#![cfg(gb_no_cuda)]

#[test]
fn embedding_launcher_rejects_cpu_only_execution() {
    // This branch must not dereference buffers or create GPU resources.
    let result = unsafe {
        gb_cuda::dsv4_mtp_gpu::mtp_embed(
            core::ptr::null_mut(),
            core::ptr::null(),
            core::ptr::null(),
            0,
            0,
            0,
            0,
            1,
            core::ptr::null_mut(),
        )
    };
    assert!(result.is_err());
}
