//! cuBLAS: a handle bound to a stream with an explicit math mode, and the fp32 GEMMs the core uses. Column-major, as
//! cuBLAS itself; a row-major `C = A B` is `sgemm(N, N, n, m, k, B, A, C)`.

use super::{foreign, GpuError};
use core::ffi::c_void;

mod ffi {
    use super::*;
    foreign! { crate::gpu::STUB_STATUS;
        pub fn cublasCreate_v2(handle: *mut *mut c_void) -> i32;
        pub fn cublasDestroy_v2(handle: *mut c_void) -> i32;
        pub fn cublasSetStream_v2(handle: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cublasSetMathMode(handle: *mut c_void, mode: i32) -> i32;
        pub fn cublasGetMathMode(handle: *mut c_void, mode: *mut i32) -> i32;
        pub fn cublasGetProperty(kind: i32, value: *mut i32) -> i32;
        pub fn cublasSetWorkspace_v2(handle: *mut c_void, workspace: *mut c_void, bytes: usize) -> i32;
        pub fn cublasSetAtomicsMode(handle: *mut c_void, mode: i32) -> i32;
        pub fn cublasSgetrfBatched(
            handle: *mut c_void, n: i32, a: *const *mut f32, lda: i32, pivots: *mut i32, info: *mut i32, batch: i32,
        ) -> i32;
        pub fn cublasSgetrsBatched(
            handle: *mut c_void, trans: i32, n: i32, nrhs: i32, a: *const *const f32, lda: i32, pivots: *const i32,
            b: *const *mut f32, ldb: i32, info: *mut i32, batch: i32,
        ) -> i32;
        pub fn cublasStrsm_v2(
            handle: *mut c_void, side: i32, uplo: i32, trans: i32, diag: i32, m: i32, n: i32, alpha: *const f32,
            a: *const f32, lda: i32, b: *mut f32, ldb: i32,
        ) -> i32;
        pub fn cublasSgemm_v2(
            handle: *mut c_void, transa: i32, transb: i32, m: i32, n: i32, k: i32, alpha: *const f32,
            a: *const f32, lda: i32, b: *const f32, ldb: i32, beta: *const f32, c: *mut f32, ldc: i32,
        ) -> i32;
        pub fn cublasSgemmStridedBatched(
            handle: *mut c_void, transa: i32, transb: i32, m: i32, n: i32, k: i32, alpha: *const f32,
            a: *const f32, lda: i32, stride_a: i64, b: *const f32, ldb: i32, stride_b: i64, beta: *const f32,
            c: *mut f32, ldc: i32, stride_c: i64, batch: i32,
        ) -> i32;
    }
}

fn check(s: i32) -> Result<(), GpuError> {
    if s == 0 {
        Ok(())
    } else if cfg!(gb_quant_no_cuda) {
        Err(GpuError::NoCuda)
    } else {
        Err(GpuError::Cublas(s))
    }
}

/// `cublasMath_t` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathMode {
    /// CUBLAS_DEFAULT_MATH: fp32 GEMMs in fp32 (no TF32).
    Default = 0,
    /// CUBLAS_PEDANTIC_MATH: the prescribed precision in every phase.
    Pedantic = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    N = 0,
    T = 1,
}

/// A cuBLAS handle. Not `Sync`: one handle per stream and thread.
pub struct Handle(*mut c_void);

unsafe impl Send for Handle {}

impl Handle {
    pub fn new(stream: &gb_cuda::rt::Stream, mode: MathMode) -> Result<Self, GpuError> {
        let mut h = core::ptr::null_mut();
        check(unsafe { ffi::cublasCreate_v2(&mut h) })?;
        let handle = Handle(h);
        check(unsafe { ffi::cublasSetStream_v2(h, stream.raw()) })?;
        check(unsafe { ffi::cublasSetMathMode(h, mode as i32) })?;
        Ok(handle)
    }

    /// Hands cuBLAS a caller-owned workspace (`bytes` at `ws`); kernel selection depends on its size.
    ///
    /// # Safety
    /// `ws` is a device buffer of `bytes` that outlives every call on this handle.
    pub unsafe fn set_workspace(&self, ws: *mut c_void, bytes: usize) -> Result<(), GpuError> {
        check(unsafe { ffi::cublasSetWorkspace_v2(self.0, ws, bytes) })?;
        check(unsafe { ffi::cublasSetAtomicsMode(self.0, 0) })
    }

    /// LU with partial pivoting of `batch` n x n column-major matrices (pointer array `a` on the device).
    ///
    /// # Safety
    /// `a` is a device array of `batch` device pointers; `pivots` holds n * batch ints, `info` batch ints, on the device.
    pub unsafe fn sgetrf_batched(
        &self,
        n: i32,
        a: *const *mut f32,
        lda: i32,
        pivots: *mut i32,
        info: *mut i32,
        batch: i32,
    ) -> Result<(), GpuError> {
        check(unsafe { ffi::cublasSgetrfBatched(self.0, n, a, lda, pivots, info, batch) })
    }

    /// Solves with the factors of [`Handle::sgetrf_batched`]; `info` is a host int.
    ///
    /// # Safety
    /// As [`Handle::sgetrf_batched`]; `b` is a device array of `batch` device pointers.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgetrs_batched(
        &self,
        n: i32,
        nrhs: i32,
        a: *const *const f32,
        lda: i32,
        pivots: *const i32,
        b: *const *mut f32,
        ldb: i32,
        batch: i32,
    ) -> Result<i32, GpuError> {
        let mut info = 0i32;
        check(unsafe { ffi::cublasSgetrsBatched(self.0, 0, n, nrhs, a, lda, pivots, b, ldb, &mut info, batch) })?;
        Ok(info)
    }

    /// Left-side triangular solve op(A) X = alpha B in place in `b` (column-major).
    ///
    /// # Safety
    /// Device pointers sized by the leading dimensions.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn strsm_left(
        &self,
        upper: bool,
        unit: bool,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *mut f32,
        ldb: i32,
    ) -> Result<(), GpuError> {
        check(unsafe { ffi::cublasStrsm_v2(self.0, 0, upper as i32, 0, unit as i32, m, n, &alpha, a, lda, b, ldb) })
    }

    pub fn math_mode(&self) -> Result<i32, GpuError> {
        let mut m = -1;
        check(unsafe { ffi::cublasGetMathMode(self.0, &mut m) })?;
        Ok(m)
    }

    /// C = alpha op(A) op(B) + beta C, column-major (m x n result, inner dimension k).
    ///
    /// # Safety
    /// The pointers are device pointers to buffers of the sizes the leading dimensions imply.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemm(
        &self,
        ta: Op,
        tb: Op,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    ) -> Result<(), GpuError> {
        check(unsafe { ffi::cublasSgemm_v2(self.0, ta as i32, tb as i32, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc) })
    }

    /// The strided-batched form of [`Handle::sgemm`].
    ///
    /// # Safety
    /// As [`Handle::sgemm`], for every batch entry.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemm_strided_batched(
        &self,
        ta: Op,
        tb: Op,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        sa: i64,
        b: *const f32,
        ldb: i32,
        sb: i64,
        beta: f32,
        c: *mut f32,
        ldc: i32,
        sc: i64,
        batch: i32,
    ) -> Result<(), GpuError> {
        check(unsafe {
            ffi::cublasSgemmStridedBatched(
                self.0, ta as i32, tb as i32, m, n, k, &alpha, a, lda, sa, b, ldb, sb, &beta, c, ldc, sc, batch,
            )
        })
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { ffi::cublasDestroy_v2(self.0) };
        }
    }
}

/// The loaded cuBLAS version: major * 10000 + minor * 100 + patch.
pub fn version() -> Result<i32, GpuError> {
    let (mut a, mut b, mut c) = (0, 0, 0);
    check(unsafe { ffi::cublasGetProperty(0, &mut a) })?;
    check(unsafe { ffi::cublasGetProperty(1, &mut b) })?;
    check(unsafe { ffi::cublasGetProperty(2, &mut c) })?;
    Ok(a * 10000 + b * 100 + c)
}
