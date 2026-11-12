//! The plain bf16-linear `gb_bf16_linear` (`csrc/dsv4_bf16.cu`): `x` fp16 `[m, k]`, `w` bf16 `[n, k]` (row-major),
//! `y` fp16 (`y_f32 = 0`) or f32 (`y_f32 = 1`) `[m, n]`, fp32 accumulation, row-invariant. Used by the Bailing Moe V3
//! family for the plain bf16 `eh_proj` and MoE router projections.
#![allow(clippy::too_many_arguments)]

use crate::rt::{CudaError, DevBuf, Stream};

fn need(b: &DevBuf, bytes: usize, what: &str) -> Result<(), CudaError> {
    if b.bytes() < bytes {
        eprintln!("{what}: buffer of {} bytes, the launch touches {bytes}", b.bytes());
        return Err(CudaError(1));
    }
    Ok(())
}

#[cfg(not(gb_no_cuda))]
extern "C" {
    fn gb_bf16_linear(
        x: *const core::ffi::c_void,
        w: *const core::ffi::c_void,
        y: *mut core::ffi::c_void,
        m: i32,
        k: i32,
        n: i32,
        y_f32: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// `y [m, n] = x [m, k] @ w^T [n, k]` (`x` fp16, `w` bf16, `y` fp16 or f32). Asynchronous on `st`.
pub fn forward(
    x: &DevBuf,
    w: &DevBuf,
    y: &mut DevBuf,
    m: usize,
    k: usize,
    n: usize,
    y_f32: bool,
    st: &Stream,
) -> Result<(), CudaError> {
    if m > 65535 || k == 0 || n == 0 || k % 8 != 0 || n % 4 != 0 {
        return Err(CudaError(1));
    }
    let yb = m * n * if y_f32 { 4 } else { 2 };
    need(x, m * k * 2, "bf16_linear x")?;
    need(w, n * k * 2, "bf16_linear w")?;
    need(y, yb, "bf16_linear y")?;
    #[cfg(not(gb_no_cuda))]
    unsafe {
        CudaError::check(gb_bf16_linear(x.ptr(), w.ptr(), y.ptr(), m as i32, k as i32, n as i32, y_f32 as i32, st.raw()))
    }
    #[cfg(gb_no_cuda)]
    {
        let _ = (x, w, y, m, k, n, y_f32, st);
        Err(CudaError(-1))
    }
}
