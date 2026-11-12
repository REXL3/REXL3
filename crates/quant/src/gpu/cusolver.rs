//! cuSOLVER dense: a handle bound to a stream, and the fp32 Cholesky factorization (64-bit `Xpotrf`).

use super::{foreign, GpuError};
use core::ffi::c_void;

mod ffi {
    use super::*;
    foreign! { crate::gpu::STUB_STATUS;
        pub fn cusolverDnCreate(handle: *mut *mut c_void) -> i32;
        pub fn cusolverDnDestroy(handle: *mut c_void) -> i32;
        pub fn cusolverDnSetStream(handle: *mut c_void, stream: *mut c_void) -> i32;
        pub fn cusolverDnCreateParams(params: *mut *mut c_void) -> i32;
        pub fn cusolverDnDestroyParams(params: *mut c_void) -> i32;
        pub fn cusolverGetProperty(kind: i32, value: *mut i32) -> i32;
        pub fn cusolverDnSgetrf_bufferSize(handle: *mut c_void, m: i32, n: i32, a: *mut f32, lda: i32, lwork: *mut i32) -> i32;
        pub fn cusolverDnSgetrf(
            handle: *mut c_void, m: i32, n: i32, a: *mut f32, lda: i32, work: *mut f32, ipiv: *mut i32, info: *mut i32,
        ) -> i32;
        pub fn cusolverDnXpotrf_bufferSize(
            handle: *mut c_void, params: *mut c_void, uplo: i32, n: i64, data_type: i32, a: *const c_void, lda: i64,
            compute_type: i32, device_bytes: *mut usize, host_bytes: *mut usize,
        ) -> i32;
        pub fn cusolverDnXpotrf(
            handle: *mut c_void, params: *mut c_void, uplo: i32, n: i64, data_type: i32, a: *mut c_void, lda: i64,
            compute_type: i32, device_buf: *mut c_void, device_bytes: usize, host_buf: *mut c_void, host_bytes: usize,
            info: *mut i32,
        ) -> i32;
    }
}

const CUDA_R_32F: i32 = 0;

fn check(s: i32) -> Result<(), GpuError> {
    if s == 0 {
        Ok(())
    } else if cfg!(gb_quant_no_cuda) {
        Err(GpuError::NoCuda)
    } else {
        Err(GpuError::Cusolver(s))
    }
}

/// `cublasFillMode_t`: which triangle of a column-major matrix holds the factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fill {
    Lower = 0,
    Upper = 1,
}

/// A cuSOLVER dense handle with its 64-bit API parameter object.
pub struct Handle {
    h: *mut c_void,
    params: *mut c_void,
}

unsafe impl Send for Handle {}

impl Handle {
    pub fn new(stream: &gb_cuda::rt::Stream) -> Result<Self, GpuError> {
        let mut h = core::ptr::null_mut();
        check(unsafe { ffi::cusolverDnCreate(&mut h) })?;
        let mut handle = Handle { h, params: core::ptr::null_mut() };
        check(unsafe { ffi::cusolverDnSetStream(h, stream.raw()) })?;
        check(unsafe { ffi::cusolverDnCreateParams(&mut handle.params) })?;
        Ok(handle)
    }

    /// Device and host workspace bytes for [`Handle::potrf`] of an n x n fp32 matrix.
    pub fn potrf_workspace(&self, fill: Fill, n: i64, lda: i64) -> Result<(usize, usize), GpuError> {
        let (mut d, mut h) = (0usize, 0usize);
        check(unsafe {
            ffi::cusolverDnXpotrf_bufferSize(
                self.h,
                self.params,
                fill as i32,
                n,
                CUDA_R_32F,
                core::ptr::null(),
                lda,
                CUDA_R_32F,
                &mut d,
                &mut h,
            )
        })?;
        Ok((d, h))
    }

    /// In-place fp32 Cholesky of the column-major n x n matrix `a`. `info` (a device int) receives the
    /// factorization status; the call is asynchronous on the handle's stream.
    ///
    /// # Safety
    /// `a` holds n x n floats with leading dimension `lda`, `work` at least the device workspace
    /// [`Handle::potrf_workspace`] returned, `info` one device int.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn potrf(
        &self,
        fill: Fill,
        n: i64,
        a: *mut f32,
        lda: i64,
        work: *mut c_void,
        work_bytes: usize,
        host_work: &mut [u8],
        info: *mut i32,
    ) -> Result<(), GpuError> {
        check(unsafe {
            ffi::cusolverDnXpotrf(
                self.h,
                self.params,
                fill as i32,
                n,
                CUDA_R_32F,
                a as *mut c_void,
                lda,
                CUDA_R_32F,
                work,
                work_bytes,
                if host_work.is_empty() { core::ptr::null_mut() } else { host_work.as_mut_ptr() as *mut c_void },
                host_work.len(),
                info,
            )
        })
    }
}

impl Handle {
    /// Workspace floats for [`Handle::sgetrf`] of an m x n column-major matrix.
    ///
    /// # Safety
    /// `a` is the device matrix the factorization will run on.
    pub unsafe fn sgetrf_workspace(&self, m: i32, n: i32, a: *mut f32, lda: i32) -> Result<i32, GpuError> {
        let mut w = 0i32;
        check(unsafe { ffi::cusolverDnSgetrf_bufferSize(self.h, m, n, a, lda, &mut w) })?;
        Ok(w)
    }

    /// In-place LU with partial pivoting (1-based `ipiv`, device `info`), asynchronous on the handle's stream.
    ///
    /// # Safety
    /// Device pointers: `a` m x n with `lda`, `work` the [`Handle::sgetrf_workspace`] floats, `ipiv` min(m, n) ints, `info`
    /// one int.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgetrf(
        &self,
        m: i32,
        n: i32,
        a: *mut f32,
        lda: i32,
        work: *mut f32,
        ipiv: *mut i32,
        info: *mut i32,
    ) -> Result<(), GpuError> {
        check(unsafe { ffi::cusolverDnSgetrf(self.h, m, n, a, lda, work, ipiv, info) })
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            if !self.params.is_null() {
                ffi::cusolverDnDestroyParams(self.params);
            }
            if !self.h.is_null() {
                ffi::cusolverDnDestroy(self.h);
            }
        }
    }
}

/// The loaded cuSOLVER version: major * 10000 + minor * 100 + patch.
pub fn version() -> Result<i32, GpuError> {
    let (mut a, mut b, mut c) = (0, 0, 0);
    check(unsafe { ffi::cusolverGetProperty(0, &mut a) })?;
    check(unsafe { ffi::cusolverGetProperty(1, &mut b) })?;
    check(unsafe { ffi::cusolverGetProperty(2, &mut c) })?;
    Ok(a * 10000 + b * 100 + c)
}
