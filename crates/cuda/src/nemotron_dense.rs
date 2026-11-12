//! Dense bf16 GEMV of the Nemotron 3.5 Super VL engine (`csrc/nemotron_dense.cu`): `y[m, n] = x[m, k] . W[n, k]^T` with `W` bf16
//! row-major (the HF `[out, in]` layout), `x` and `y` fp32, fp32 accumulation, no cuBLAS. Row invariant: a row of `y` is bitwise
//! the same whatever `m` is, and the same on every launch path of the kernel (a register path for 1 to 4 rows and above 64, a
//! shared-memory pipeline for 5 to 64: see the header of `csrc/nemotron_dense.cu`; the order of f32 operations of one output is
//! written out in `tests/nemotron_dense.rs`). Every function launches on `stream` and returns without synchronising; buffer sizes
//! and alignment are checked (a short or misaligned buffer panics rather than faulting on the device).

pub use gpu::*;

/// The `k` widths the kernel is built for (the Nemotron 3.5 Super hidden, router, expert and eh_proj input sizes).
pub const SUPPORTED_K: [usize; 5] = [1024, 2688, 4096, 5376, 8192];

mod gpu {
    use crate::{CudaError, DevBuf, Stream};

    mod ffi {
        use core::ffi::c_void;
        crate::launchers! {
            pub fn gb_nemo_dense_gemv(
                w: *const c_void,
                x: *const c_void,
                y: *mut c_void,
                m: i32,
                n: i32,
                k: i32,
                s: *mut c_void,
            ) -> i32;
        }
    }

    fn need(b: &DevBuf, bytes: usize, what: &str) {
        assert!(b.bytes() >= bytes, "nemotron_dense {what}: buffer of {} bytes, need {bytes}", b.bytes());
        assert!(b.ptr() as usize % 16 == 0, "nemotron_dense {what}: buffer is not 16-byte aligned");
    }

    fn i(v: usize) -> Result<i32, CudaError> {
        i32::try_from(v).map_err(|_| CudaError(1))
    }

    /// `y[m, n] = x[m, k] . W[n, k]^T`: `w` bf16 `[n, k]`, `x` f32 `[m, k]`, `y` f32 `[m, n]` (row-major). `k` must be a multiple
    /// of 8 and one of [`SUPPORTED_K`]; any other `k` returns `CudaError` (cudaErrorInvalidValue) without launching.
    pub fn gemv_bf16(w: &DevBuf, x: &DevBuf, y: &mut DevBuf, m: usize, n: usize, k: usize, st: &Stream) -> Result<(), CudaError> {
        assert!(k % 8 == 0, "nemotron_dense: k = {k} is not a multiple of 8");
        need(w, n * k * 2, "w");
        need(x, m * k * 4, "x");
        need(y, m * n * 4, "y");
        CudaError::check(unsafe { ffi::gb_nemo_dense_gemv(w.ptr(), x.ptr(), y.ptr(), i(m)?, i(n)?, i(k)?, st.raw()) })
    }
}
