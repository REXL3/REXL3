//! Per-row log-probability readout of bf16 logits (kernel in `csrc/logprob.cu`, top-k by `glue::topk_valid`).
//!
//! For logits `[rows, stride]` and one chosen column per row, [`rows`] leaves on the device what a log-probability of
//! the raw distribution needs, so a caller reads back `rows x (2 + 2k)` numbers instead of whole rows:
//!
//! - `norm[r] = [lse, chosen_logit]`: the fp32 log-sum-exp over the valid columns and the chosen column's logit. The
//!   chosen token's log-probability is `chosen_logit - lse`.
//! - the top `k` `(id, logit)` pairs by (logit descending, id ascending), as `glue::topk` orders its candidates.
//!
//! Row invariance: both launches work one row at a time in an order fixed by the row length, so a row's output bits do
//! not depend on its index, on the number of rows in the call, or on how a prompt's rows were split over calls.
//!
//! # Safety
//! As in `glue`: the pointers must be valid device addresses for the sizes implied by the arguments, and the buffers must
//! stay alive until the stream has run the launches. Nothing is synchronised.
#![allow(clippy::missing_safety_doc)]

use core::ffi::c_void;

use crate::glm_glue;
use crate::glue;
use crate::rt::{CudaError, Stream};

/// Most top entries a readout row carries (the HTTP `top_logprobs` ceiling).
pub const MAX_TOP: usize = 20;

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_logprob_norm(
            logits: *const c_void,
            rows: i32,
            stride: i32,
            valid: i32,
            chosen: *const c_void,
            out: *mut c_void,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_logprob_norm_f16(
            logits: *const c_void,
            rows: i32,
            stride: i32,
            valid: i32,
            chosen: *const c_void,
            out: *mut c_void,
            s: *mut c_void,
        ) -> i32;
    }
}

fn i(v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| CudaError(1))
}

/// Workspace bytes [`rows`] needs (16-byte aligned): the top-k's.
pub fn workspace_bytes(rows: usize, valid: usize, k: usize) -> usize {
    glue::topk_valid_workspace_bytes(rows, valid, k)
}

/// `norm` f32 `[rows, 2]`: per row the log-sum-exp of columns `[0, valid)` of bf16 `logits` `[rows, stride]` (max-shifted,
/// fp32, fixed order; NaN for a row holding a NaN, +inf for one holding +inf, -inf when every column is -inf) and the logit
/// of column `chosen[r]` (i32 `[rows]`; -inf when it is outside `[0, valid)`: a column that is not in the row has
/// probability zero). `1 <= valid <= stride`.
pub unsafe fn norm(
    logits: *const c_void,
    rows: usize,
    stride: usize,
    valid: usize,
    chosen: *const c_void,
    norm: *mut c_void,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_logprob_norm(logits, i(rows)?, i(stride)?, i(valid)?, chosen, norm, stream.raw()))
}

/// The whole readout of `rows` rows: [`norm`], and the top `k` of every row into `ids` i32 / `vals` f32 `[rows, k]`
/// (`glue::topk_valid`: bf16 widened, a NaN counted and reported as -inf). `1 <= k <= min(MAX_TOP, valid)`; `workspace`
/// holds [`workspace_bytes`] bytes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn rows(
    logits: *const c_void,
    rows: usize,
    stride: usize,
    valid: usize,
    k: usize,
    chosen: *const c_void,
    norm_out: *mut c_void,
    ids: *mut c_void,
    vals: *mut c_void,
    workspace: *mut c_void,
    ws_bytes: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    if k == 0 || k > MAX_TOP || k > valid {
        return Err(CudaError(1));
    }
    if rows == 0 {
        return Ok(());
    }
    norm(logits, rows, stride, valid, chosen, norm_out, stream)?;
    glue::topk_valid(logits, rows, stride, valid, k, ids, vals, workspace, ws_bytes, stream)
}

/// Workspace bytes [`rows_f16`] needs (16-byte aligned): the fp16 top-k's.
pub fn workspace_bytes_f16(rows: usize, valid: usize, k: usize) -> usize {
    glm_glue::topk_workspace_bytes(rows, valid, k)
}

/// [`norm`] for fp16 logits. The normaliser's fixed fp32 reduction is identical; only the input conversion differs.
pub unsafe fn norm_f16(
    logits: *const c_void,
    rows: usize,
    stride: usize,
    valid: usize,
    chosen: *const c_void,
    norm: *mut c_void,
    stream: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_logprob_norm_f16(logits, i(rows)?, i(stride)?, i(valid)?, chosen, norm, stream.raw()))
}

/// The whole readout of fp16 logits: [`norm_f16`] and `glm_glue::topk_f16` over the same valid columns.
#[allow(clippy::too_many_arguments)]
pub unsafe fn rows_f16(
    logits: *const c_void,
    rows: usize,
    stride: usize,
    valid: usize,
    k: usize,
    chosen: *const c_void,
    norm_out: *mut c_void,
    ids: *mut c_void,
    vals: *mut c_void,
    workspace: *mut c_void,
    ws_bytes: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    if k == 0 || k > MAX_TOP || k > valid {
        return Err(CudaError(1));
    }
    if rows == 0 {
        return Ok(());
    }
    norm_f16(logits, rows, stride, valid, chosen, norm_out, stream)?;
    glm_glue::topk_f16_raw(logits, rows, stride, valid, k, ids, vals, workspace, ws_bytes, stream)
}
