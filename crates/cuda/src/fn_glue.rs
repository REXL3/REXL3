//! Flash-Next gated norm and shared-expert addition.
#![allow(clippy::too_many_arguments)]
use crate::rt::{CudaError, DevBuf, Stream};

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_gated_rmsnorm_sigmoid(o: *const c_void, z: *const c_void, w: *const f32,
            out: *mut c_void, rows: i32, heads: i32, dim: i32, eps: f32, stream: *mut c_void) -> i32;
        pub fn gb_add_gated_shared(shared_down: *const c_void, gate_scalar: *const c_void,
            moe_out: *mut c_void, hidden: u32, num_tokens: u32, stream: *mut c_void) -> i32;
    }
}
fn need(b: &DevBuf, size: usize) -> Result<(), CudaError> {
    if b.bytes() < size {
        return Err(CudaError(1));
    }
    Ok(())
}
/// BF16 input/output `[rows, heads, dim]`, FP32 gamma `[dim]`; optional in-place input via `None`.
pub fn gated_rmsnorm_sigmoid(
    o: Option<&DevBuf>,
    z: &DevBuf,
    w: &DevBuf,
    out: &mut DevBuf,
    rows: usize,
    heads: usize,
    dim: usize,
    eps: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    if !matches!(dim, 128 | 256) || heads == 0 {
        return Err(CudaError(1));
    }
    let n = rows.checked_mul(heads).and_then(|n| n.checked_mul(dim)).and_then(|n| n.checked_mul(2)).ok_or(CudaError(1))?;
    need(z, n)?;
    need(w, dim * 4)?;
    need(out, n)?;
    if let Some(o) = o {
        need(o, n)?;
    }
    let op = o.map_or(out.ptr(), DevBuf::ptr);
    CudaError::check(unsafe {
        ffi::gb_gated_rmsnorm_sigmoid(
            op,
            z.ptr(),
            w.ptr() as *const f32,
            out.ptr(),
            i32::try_from(rows).map_err(|_| CudaError(1))?,
            i32::try_from(heads).map_err(|_| CudaError(1))?,
            dim as i32,
            eps,
            st.raw(),
        )
    })
}
/// In-place BF16 `moe_out[t,d] += sigmoid(gate_scalar[t]) * shared_down[t,d]`.
pub fn add_gated_shared(
    shared_down: &DevBuf,
    gate_scalar: &DevBuf,
    moe_out: &mut DevBuf,
    hidden: usize,
    rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    let n = rows.checked_mul(hidden).and_then(|n| n.checked_mul(2)).ok_or(CudaError(1))?;
    need(shared_down, n)?;
    need(gate_scalar, rows.checked_mul(2).ok_or(CudaError(1))?)?;
    need(moe_out, n)?;
    CudaError::check(unsafe {
        ffi::gb_add_gated_shared(
            shared_down.ptr(),
            gate_scalar.ptr(),
            moe_out.ptr(),
            u32::try_from(hidden).map_err(|_| CudaError(1))?,
            u32::try_from(rows).map_err(|_| CudaError(1))?,
            st.raw(),
        )
    })
}
