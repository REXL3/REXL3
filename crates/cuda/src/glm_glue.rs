//! The two glue kernels GLM-5.3-Flash needs that the bf16 glue ([`crate::glue`]) does not have (`csrc/glm_glue.cu`): an fp16
//! RMSNorm (the MLA chain keeps fp16 activations end to end) and the exact top-k over fp16 logits with a valid-vocabulary mask.
//!
//! Both are row-invariant by construction (a row's output bits depend on that row alone: fixed reduction order set by the row
//! length, no atomics on data) and asynchronous on the stream they are given.

use crate::rt::{CudaError, DevBuf, Stream};

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_glm_rmsnorm_f16(x: *const c_void, w: *const c_void, y: *mut c_void, rows: i32, dim: i32, eps: f32, stream: *mut c_void) -> i32;
        pub fn gb_glm_topk_f16_workspace_bytes(rows: i32, valid: i32, count: i32) -> usize;
        pub fn gb_glm_topk_f16(
            logits: *const c_void,
            rows: i32,
            stride: i32,
            valid: i32,
            count: i32,
            ids: *mut c_void,
            vals: *mut c_void,
            workspace: *mut c_void,
            ws_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
    }
}

const INVALID: CudaError = CudaError(1);

fn i32_of(v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| INVALID)
}

/// `y = f16((x * inv) * w)` per row with `inv = 1 / sqrtf(mean(x^2) + eps)`: `x` and `y` fp16 `[rows, dim]`, `w` fp32
/// `[dim]` as given (GLM's RMSNorm gamma multiplies as it is stored: there is no `+ 1`). `dim % 8 == 0`.
pub fn rmsnorm_f16(
    x: &DevBuf,
    w: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    dim: usize,
    eps: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    if x.bytes() < rows * dim * 2 || y.bytes() < rows * dim * 2 || w.bytes() < dim * 4 {
        return Err(INVALID);
    }
    // SAFETY: the buffers hold the extents checked above and the launch is asynchronous on `st`
    let rc = unsafe { ffi::gb_glm_rmsnorm_f16(x.ptr(), w.ptr(), y.ptr(), i32_of(rows)?, i32_of(dim)?, eps, st.raw()) };
    CudaError::check(rc)
}

/// Workspace bytes [`topk_f16`] needs (0 for an unsupported shape).
pub fn topk_workspace_bytes(rows: usize, valid: usize, count: usize) -> usize {
    match (i32::try_from(rows), i32::try_from(valid), i32::try_from(count)) {
        (Ok(r), Ok(v), Ok(c)) => unsafe { ffi::gb_glm_topk_f16_workspace_bytes(r, v, c) },
        _ => 0,
    }
}

/// Exact per-row top-`count` (value descending, ties to the lower id) of fp16 logits `[rows, stride]` over columns `0..valid`
/// (the columns past `valid` are never read: the vocabulary mask). `ids` i32 `[rows, count]`, `vals` f32 `[rows, count]` (an empty
/// slot, only when `count > valid`, which is refused, would be id -1 and -inf). `1 <= count <= min(64, valid)`, `valid <= stride`.
#[allow(clippy::too_many_arguments)]
pub fn topk_f16(
    logits: &DevBuf,
    rows: usize,
    stride: usize,
    valid: usize,
    count: usize,
    ids: &mut DevBuf,
    vals: &mut DevBuf,
    workspace: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    let need = topk_workspace_bytes(rows, valid, count);
    if rows == 0
        || need == 0
        || valid > stride
        || logits.bytes() < rows * stride * 2
        || ids.bytes() < rows * count * 4
        || vals.bytes() < rows * count * 4
        || workspace.bytes() < need
    {
        return Err(INVALID);
    }
    // SAFETY: extents checked above; asynchronous on `st`
    let rc = unsafe {
        ffi::gb_glm_topk_f16(
            logits.ptr(),
            i32_of(rows)?,
            i32_of(stride)?,
            i32_of(valid)?,
            i32_of(count)?,
            ids.ptr(),
            vals.ptr(),
            workspace.ptr(),
            workspace.bytes(),
            st.raw(),
        )
    };
    CudaError::check(rc)
}
