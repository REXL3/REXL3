//! cuBLAS and cuSOLVER bindings for the quantizer (strict fp32: cuBLAS pedantic math, no TF32).
//!
//! A stub build (`cfg(gb_quant_no_cuda)`) compiles the same functions; every call fails with [`GpuError::NoCuda`].

pub mod cublas;
pub mod cusolver;

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuError {
    /// A cuBLAS status other than CUBLAS_STATUS_SUCCESS.
    Cublas(i32),
    /// A cuSOLVER status other than CUSOLVER_STATUS_SUCCESS.
    Cusolver(i32),
    /// A cuSOLVER factorization reported `info != 0` (for potrf: the leading minor of that order is not positive).
    Info(i64),
    /// A CUDA runtime error.
    Cuda(i32),
    /// Built without CUDA.
    NoCuda,
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::Cublas(s) => write!(f, "cuBLAS status {s}"),
            GpuError::Cusolver(s) => write!(f, "cuSOLVER status {s}"),
            GpuError::Info(i) => write!(f, "factorization info {i}"),
            GpuError::Cuda(e) => write!(f, "CUDA error {e}"),
            GpuError::NoCuda => write!(f, "built without CUDA"),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<gb_cuda::rt::CudaError> for GpuError {
    fn from(e: gb_cuda::rt::CudaError) -> Self {
        GpuError::Cuda(e.0)
    }
}

/// Declares foreign functions; a stub build gets functions with the same signatures that return `$fail`.
macro_rules! foreign {
    ($fail:expr; $( $(#[$m:meta])* pub fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) -> i32 ; )*) => {
        #[cfg(not(gb_quant_no_cuda))]
        extern "C" { $( $(#[$m])* pub fn $name($($arg: $ty),*) -> i32; )* }
        $(
            $(#[$m])*
            #[cfg(gb_quant_no_cuda)]
            #[allow(dead_code, non_snake_case, unused_variables, clippy::too_many_arguments, clippy::missing_safety_doc)]
            pub unsafe fn $name($($arg: $ty),*) -> i32 { $fail }
        )*
    };
}
pub(crate) use foreign;

/// The status every stub returns (no CUDA): CUBLAS_STATUS_NOT_INITIALIZED / CUSOLVER_STATUS_NOT_INITIALIZED.
#[cfg(gb_quant_no_cuda)]
pub(crate) const STUB_STATUS: i32 = 1;

/// What a stub CUDA runtime or kernel launcher returns: cudaErrorNoDevice.
#[cfg(gb_quant_no_cuda)]
pub(crate) const CUDA_ERROR_NO_DEVICE: i32 = 100;

/// Library versions (major * 10000 + minor * 100 + patch), as the loaded libraries report them.
#[derive(Debug, Clone, Copy)]
pub struct Versions {
    pub cublas: i32,
    pub cusolver: i32,
}

pub fn versions() -> Result<Versions, GpuError> {
    Ok(Versions { cublas: cublas::version()?, cusolver: cusolver::version()? })
}
