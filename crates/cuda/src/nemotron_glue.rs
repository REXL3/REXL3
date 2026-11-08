//! Glue kernels the Nemotron layer loop needs between the mixers (`csrc/nemotron_glue.cu`, written by wave 1 task K1).
//!
//! The signatures here were frozen at wave 0. Every function launches on `st` and returns without synchronising. Activations between
//! kernels are f32. Buffers are whole `DevBuf`s whose sizes the wrappers check (a short or misaligned buffer panics rather than
//! faulting on the device). Every kernel is row-invariant: the bits of a row do not depend on how many rows share the call.
//!
//! Numerics are explicit IEEE binary32 operations with no fused multiply-add, no `rsqrt` and no fast-math, and the order of every sum
//! is fixed in the `.cu` file's header, so a CPU model that does the same operations in the same order matches bit for bit.

use crate::{CudaError, DevBuf, Stream};

/// `cudaErrorNotSupported`. The wave-0 stub bodies returned it; no function here does any more. Kept because callers and
/// `graph::is_capture_code` match the code.
pub const NOT_IMPLEMENTED: CudaError = CudaError(801);

/// What a call outside its contract returns, without launching: `cudaErrorInvalidValue`.
const INVALID: CudaError = CudaError(1);

/// The hidden width of the trunk.
pub const HIDDEN: usize = 4096;
/// The Mamba2 `in_proj` output width: z 8192 | xBC 10240 | dt 128.
pub const INPROJ_COLS: usize = 18560;
pub const Z_COLS: usize = 8192;
pub const XBC_COLS: usize = 10240;
pub const DT_COLS: usize = 128;

/// The most candidates `topk_f32` returns per row.
pub const TOPK_MAX_COUNT: usize = 64;
/// Candidate lists a stage-1 block of `topk_f32` writes (one per warp; `TK_WARPS` in the `.cu` file).
pub const TOPK_WARPS: usize = 8;
/// Stage-1 blocks the plan spreads a call over: one per SM of the GB10 (48), which measured fastest at 1 to 8 rows.
const TOPK_TARGET_BLOCKS: usize = 48;
/// The most stage-1 blocks one row is split into.
const TOPK_MAX_BLOCKS: usize = 16;
/// No stage-1 block gets less than this many elements of a row (256 per warp).
const TOPK_MIN_SLICE: usize = 2048;

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_nemo_glue_embed_gather(table: *const c_void, ids: *const c_void, vocab: i32, rows: i32, out: *mut c_void, s: *mut c_void) -> i32;
        pub fn gb_nemo_glue_rmsnorm(
            x: *const c_void,
            w: *const c_void,
            rows: i32,
            d: i32,
            x_stride: i64,
            eps: f32,
            out: *mut c_void,
            out_stride: i64,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_nemo_glue_split_inproj(y: *const c_void, rows: i32, z: *mut c_void, xbc: *mut c_void, dt: *mut c_void, s: *mut c_void) -> i32;
        pub fn gb_nemo_glue_widen_bf16(src: *const c_void, n: i64, dst: *mut c_void, s: *mut c_void) -> i32;
        pub fn gb_nemo_glue_widen_f16(src: *const c_void, n: i64, dst: *mut c_void, s: *mut c_void) -> i32;
        pub fn gb_nemo_glue_topk(
            logits: *const c_void,
            rows: i32,
            vocab: i32,
            count: i32,
            blocks: i32,
            ws: *mut c_void,
            ws_bytes: usize,
            ids: *mut c_void,
            vals: *mut c_void,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_nemo_glue_splice_rows(feats: *const c_void, src_row: i64, dst: *mut c_void, dst_row: i64, n: i64, s: *mut c_void) -> i32;
    }
}

/// Panics when `b` holds fewer than `bytes` bytes or its base address is not a multiple of `align`.
fn need(b: &DevBuf, bytes: usize, align: usize, what: &str) {
    assert!(b.bytes() >= bytes, "nemotron_glue {what}: buffer of {} bytes, need {bytes}", b.bytes());
    assert!(b.ptr() as usize % align == 0, "nemotron_glue {what}: buffer is not {align}-byte aligned");
}

fn i(v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| INVALID)
}

fn l(v: usize) -> Result<i64, CudaError> {
    i64::try_from(v).map_err(|_| INVALID)
}

/// `out[r] = table[ids[r]]` widened to f32. `table` is bf16 `[vocab, 4096]` (the vocabulary is `table.bytes() / 8192`), `ids` is i32
/// `[rows]`, `out` is f32 `[rows, 4096]`. An id outside the vocabulary (negative, or at or past it) yields a zero row (the engine
/// rejects such ids earlier). `table` and `out` are 16-byte aligned.
pub fn embed_gather_f32(table: &DevBuf, ids: &DevBuf, rows: usize, out: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    need(table, 0, 16, "table");
    need(ids, rows * 4, 4, "ids");
    need(out, rows * HIDDEN * 4, 16, "out");
    let vocab = table.bytes() / (HIDDEN * 2);
    CudaError::check(unsafe { ffi::gb_nemo_glue_embed_gather(table.ptr(), ids.ptr(), i(vocab)?, i(rows)?, out.ptr(), st.raw()) })
}

/// RMSNorm with a bf16 weight and fp32 math: `out[r][c] = (x[r][c] * inv) * w[c]` with `inv = 1.0f / sqrtf(mean(x[r]^2) + eps)`, `x` f32
/// with row stride `x_stride` elements, `out` f32 with row stride `out_stride` (so the MTP head can write into one half of a
/// concatenated row). `w` is bf16 `[d]`. The sum of squares is taken in the fixed tree documented in `csrc/nemotron_glue.cu` (256
/// strided partial sums, a warp xor tree, then a balanced tree over the 8 warps), whatever `rows` and `d` are. For more than one
/// row the strides must be at least `d` (rows of `out` may not overlap); `d == 0` is refused.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_bf16w_f32(
    x: &DevBuf,
    w: &DevBuf,
    rows: usize,
    d: usize,
    x_stride: usize,
    eps: f32,
    out: &mut DevBuf,
    out_stride: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    if d == 0 || (rows > 1 && (x_stride < d || out_stride < d)) {
        return Err(INVALID);
    }
    need(x, ((rows - 1) * x_stride + d) * 4, 4, "x");
    need(w, d * 2, 2, "w");
    need(out, ((rows - 1) * out_stride + d) * 4, 4, "out");
    CudaError::check(unsafe {
        ffi::gb_nemo_glue_rmsnorm(x.ptr(), w.ptr(), i(rows)?, i(d)?, l(x_stride)?, eps, out.ptr(), l(out_stride)?, st.raw())
    })
}

/// Splits the `in_proj` output `y` (f32 `[rows, 18560]`) into `z` (f32 `[rows, 8192]`, rounded to the bf16 grid, round to nearest even,
/// as the bf16 model does), `xbc` (f32 `[rows, 10240]`) and `dt` (f32 `[rows, 128]`), the last two exact copies. All buffers are
/// 16-byte aligned.
pub fn split_inproj(
    y: &DevBuf,
    rows: usize,
    z: &mut DevBuf,
    xbc: &mut DevBuf,
    dt: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    need(y, rows * INPROJ_COLS * 4, 16, "y");
    need(z, rows * Z_COLS * 4, 16, "z");
    need(xbc, rows * XBC_COLS * 4, 16, "xbc");
    need(dt, rows * DT_COLS * 4, 16, "dt");
    CudaError::check(unsafe { ffi::gb_nemo_glue_split_inproj(y.ptr(), i(rows)?, z.ptr(), xbc.ptr(), dt.ptr(), st.raw()) })
}

/// Load-time widen: `dst[i] = f32(src[i])` for `n` bf16 elements. `src` and `dst` are 16-byte aligned.
pub fn widen_bf16_f32(src: &DevBuf, n: usize, dst: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
    if n == 0 {
        return Ok(());
    }
    need(src, n * 2, 16, "src");
    need(dst, n * 4, 16, "dst");
    CudaError::check(unsafe { ffi::gb_nemo_glue_widen_bf16(src.ptr(), l(n)?, dst.ptr(), st.raw()) })
}

/// Load-time widen: `dst[i] = f32(src[i])` for `n` fp16 elements. `src` and `dst` are 16-byte aligned.
pub fn widen_f16_f32(src: &DevBuf, n: usize, dst: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
    if n == 0 {
        return Ok(());
    }
    need(src, n * 2, 16, "src");
    need(dst, n * 4, 16, "dst");
    CudaError::check(unsafe { ffi::gb_nemo_glue_widen_f16(src.ptr(), l(n)?, dst.ptr(), st.raw()) })
}

/// Stage-1 blocks per row of `topk_f32` for a call of `rows` rows over `vocab` logits: 48 blocks over the call (one per SM), at most 16
/// per row (stage 2 merges every block's 8 lists, one after another), none thinner than 2048 elements, and at least one. The result of
/// `topk_f32` does not depend on it; the workspace size does.
pub fn topk_blocks_per_row(rows: usize, vocab: usize) -> usize {
    (TOPK_TARGET_BLOCKS / rows.max(1)).min(TOPK_MAX_BLOCKS).min(vocab.div_ceil(TOPK_MIN_SLICE)).max(1)
}

/// The workspace bytes `topk_f32` needs for `rows` rows over `vocab` logits and `count` candidates: one sorted list of `count` u64
/// keys per stage-1 warp, `rows * blocks * 8 * count * 8` (163,840 bytes at 64 rows, vocab 131072, count 40). Exact: the launcher
/// recomputes it from its own arguments and refuses a workspace of fewer bytes. 8-byte aligned. It is not monotone in `rows` (the block
/// plan changes with it: 3 rows need more than 7), but for `rows <= 64` the largest is at 64, so a buffer sized for the largest row
/// count a caller will use serves every smaller one.
pub fn topk_workspace_bytes(rows: usize, vocab: usize, count: usize) -> usize {
    rows * topk_blocks_per_row(rows, vocab) * TOPK_WARPS * count * 8
}

/// The `count` largest of each row's f32 logits `[rows, vocab]`: values descending, ties to the lowest id. `ids` is i32
/// `[rows, count]` and `vals` is f32 `[rows, count]` (the logits as stored, so a NaN payload or the sign of a zero is kept).
/// The order is the stable sort on (value desc, id asc) with -0.0 equal to +0.0 and NaN below every number (including -inf), NaNs
/// among themselves by lowest id; the result is that sort's, bit for bit. `1 <= count <= min(64, vocab)` and `rows <= 65535`
/// (otherwise `CudaError(1)`, nothing launched). `ws` needs [`topk_workspace_bytes`] bytes, 8-byte aligned; its contents on entry do not matter.
#[allow(clippy::too_many_arguments)]
pub fn topk_f32(
    logits: &DevBuf,
    rows: usize,
    vocab: usize,
    count: usize,
    ws: &mut DevBuf,
    ids: &mut DevBuf,
    vals: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    if count == 0 || count > TOPK_MAX_COUNT || count > vocab {
        return Err(INVALID);
    }
    need(logits, rows * vocab * 4, 4, "logits");
    need(ws, topk_workspace_bytes(rows, vocab, count), 8, "ws");
    need(ids, rows * count * 4, 4, "ids");
    need(vals, rows * count * 4, 4, "vals");
    let blocks = topk_blocks_per_row(rows, vocab);
    CudaError::check(unsafe {
        ffi::gb_nemo_glue_topk(
            logits.ptr(),
            i(rows)?,
            i(vocab)?,
            i(count)?,
            i(blocks)?,
            ws.ptr(),
            ws.bytes(),
            ids.ptr(),
            vals.ptr(),
            st.raw(),
        )
    })
}

/// Copies `n` rows of bf16 image features `[*, 4096]` starting at `src_row` over the f32 rows of `dst` starting at `dst_row`
/// (the image-token splice), widening each exactly. Built in milestone M3. Both buffers are 16-byte aligned.
pub fn splice_rows(
    feats: &DevBuf,
    src_row: usize,
    dst: &mut DevBuf,
    dst_row: usize,
    n: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if n == 0 {
        return Ok(());
    }
    need(feats, (src_row + n) * HIDDEN * 2, 16, "feats");
    need(dst, (dst_row + n) * HIDDEN * 4, 16, "dst");
    CudaError::check(unsafe { ffi::gb_nemo_glue_splice_rows(feats.ptr(), l(src_row)?, dst.ptr(), l(dst_row)?, l(n)?, st.raw()) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_inproj_split_adds_up() {
        assert_eq!(Z_COLS + XBC_COLS + DT_COLS, INPROJ_COLS);
    }

    #[test]
    fn the_not_implemented_code_is_kept_for_the_capture_classifier() {
        assert_eq!(NOT_IMPLEMENTED.0, 801);
        assert!(crate::graph::is_capture_code(NOT_IMPLEMENTED.0));
    }

    #[test]
    fn the_topk_plan_spreads_a_call_over_the_gpu_and_the_workspace_follows_it() {
        // the decode call: 16 blocks of one row; a verify window of 8 rows: 6 each (one per SM); 64 rows: one block per row
        assert_eq!(topk_blocks_per_row(1, 131072), 16);
        assert_eq!(topk_blocks_per_row(8, 131072), 6);
        assert_eq!(topk_blocks_per_row(7, 131072), 6);
        assert_eq!(topk_blocks_per_row(64, 131072), 1);
        // a thin row is not cut below 2048 elements per block, and there is always one block
        assert_eq!(topk_blocks_per_row(1, 1003), 1);
        assert_eq!(topk_blocks_per_row(1, 5000), 3);
        assert_eq!(topk_blocks_per_row(1, 1), 1);
        assert_eq!(topk_blocks_per_row(0, 131072), 16);
        for rows in [1usize, 2, 3, 7, 8, 63, 64] {
            for vocab in [1usize, 40, 1003, 4096, 5000, 131072] {
                let b = topk_blocks_per_row(rows, vocab);
                assert!((1..=16).contains(&b) && b <= vocab.div_ceil(2048).max(1));
                for count in [1usize, 40, 64] {
                    assert_eq!(topk_workspace_bytes(rows, vocab, count), rows * b * 8 * count * 8);
                }
            }
        }
        assert_eq!(topk_workspace_bytes(64, 131072, 40), 64 * 8 * 40 * 8);
        assert_eq!(topk_workspace_bytes(0, 131072, 40), 0);
        // not monotone, but a workspace sized at 64 rows serves every row count up to 64
        assert!(topk_workspace_bytes(3, 131072, 40) > topk_workspace_bytes(7, 131072, 40));
        for vocab in [1usize, 1003, 5000, 131072] {
            for count in [1usize, 40, 64] {
                let top = topk_workspace_bytes(64, vocab, count);
                assert!((1..=64).all(|r| topk_workspace_bytes(r, vocab, count) <= top), "vocab {vocab} count {count}");
            }
        }
    }
}
