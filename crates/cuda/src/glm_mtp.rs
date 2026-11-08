//! The glue kernels of GLM-5.3-Flash's NextN (MTP) layer (`csrc/glm_mtp.cu`): the input stage (embedding gather, `enorm`, `hnorm`,
//! concat), the fp32-input RMSNorms and the fp32 residual adds of a plain-residual block. Row-invariant by construction (a row's bits
//! depend on that row alone; the sums of squares are in the one fixed order of `glm_glue::rmsnorm_f16`), asynchronous on the stream.

use crate::rt::{CudaError, DevBuf, Stream};

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_glm_mtp_input(
            table: *const c_void,
            ids: *const i32,
            hid: *const c_void,
            we: *const f32,
            wh: *const f32,
            out: *mut c_void,
            rows: i32,
            dim: i32,
            vocab: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_glm_mtp_embed_f32(feat: *const c_void, we: *const f32, out: *mut c_void, rows: i32, dim: i32, eps: f32, stream: *mut c_void) -> i32;
        pub fn gb_glm_rmsnorm_f32_f16(x: *const c_void, w: *const c_void, y: *mut c_void, rows: i32, dim: i32, eps: f32, stream: *mut c_void) -> i32;
        pub fn gb_glm_add_rmsnorm_f32(
            x: *mut c_void,
            y: *const c_void,
            w: *const c_void,
            y_out: *mut c_void,
            rows: i32,
            dim: i32,
            eps: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_glm_add_f32(x: *mut c_void, y: *const c_void, n: i64, stream: *mut c_void) -> i32;
        pub fn gb_glm_keyed_pick(
            ids: *const i32,
            vals: *const f32,
            count: i32,
            rows: i32,
            params: *const c_void,
            depth: i32,
            out_ids: *mut i32,
            out_vals: *mut f32,
            stream: *mut c_void,
        ) -> i32;
    }
}

const INVALID: CudaError = CudaError(1);

fn i32_of(v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| INVALID)
}

/// `out[r] = [ f16((e * inv_e) * we) | f16((h * inv_h) * wh) ]` (fp16 `[rows, 2 * dim]`) with `e` the bf16 embedding row of `ids[r]` rounded
/// to fp16 (an id outside `0..vocab` is a zero row), `h` the fp16 hidden row `hid[r]` (`[rows, dim]`) and `inv_x = 1 / sqrtf(mean(x^2) + eps)`.
/// `table` is bf16 `[vocab, dim]`, `ids` i32 `[rows]`, `we` and `wh` fp32 `[dim]` (RMSNorm gammas as stored). `dim % 8 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn mtp_input(
    table: &DevBuf,
    ids: &DevBuf,
    hid: &DevBuf,
    we: &DevBuf,
    wh: &DevBuf,
    out: &mut DevBuf,
    rows: usize,
    dim: usize,
    vocab: usize,
    eps: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    if table.bytes() < vocab * dim * 2
        || ids.bytes() < rows * 4
        || hid.bytes() < rows * dim * 2
        || we.bytes() < dim * 4
        || wh.bytes() < dim * 4
        || out.bytes() < rows * 2 * dim * 2
    {
        return Err(INVALID);
    }
    // SAFETY: the buffers hold the extents checked above and the launch is asynchronous on `st`
    let rc = unsafe {
        ffi::gb_glm_mtp_input(
            table.ptr(),
            ids.ptr() as *const i32,
            hid.ptr(),
            we.ptr() as *const f32,
            wh.ptr() as *const f32,
            out.ptr(),
            i32_of(rows)?,
            i32_of(dim)?,
            i32_of(vocab)?,
            eps,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// The embedding half of `mtp_input` for rows whose embedding is a vision feature: `cat[dst_row + i][..dim] = f16((e * inv_e) * we)` with `e` the fp32
/// feature row `src_row + i` of `feat` (`[_, dim]`) rounded to fp16, for `i` in `0..n`; the rest of `cat` is untouched. The sum of squares is the one of
/// `mtp_input`, so the bits of a row depend on that row alone.
#[allow(clippy::too_many_arguments)]
pub fn mtp_embed_f32(
    feat: &DevBuf,
    src_row: usize,
    we: &DevBuf,
    cat: &mut DevBuf,
    dst_row: usize,
    n: usize,
    dim: usize,
    eps: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    if feat.bytes() < (src_row + n) * dim * 4 || we.bytes() < dim * 4 || cat.bytes() < (dst_row + n) * 2 * dim * 2 {
        return Err(INVALID);
    }
    // SAFETY: the buffers hold the extents checked above (the offsets keep the 16-byte alignment: whole rows) and the launch is asynchronous on `st`
    let rc = unsafe {
        ffi::gb_glm_mtp_embed_f32(
            (feat.ptr() as *const u8).add(src_row * dim * 4) as *const core::ffi::c_void,
            we.ptr() as *const f32,
            (cat.ptr() as *mut u8).add(dst_row * 2 * dim * 2) as *mut core::ffi::c_void,
            i32_of(n)?,
            i32_of(dim)?,
            eps,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// `y = f16((x * inv) * w)` per row: `x` fp32 `[rows, dim]`, `y` fp16, `w` fp32 `[dim]`. `dim % 8 == 0`.
pub fn rmsnorm_f32_f16(
    x: &DevBuf,
    w: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    dim: usize,
    eps: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    if x.bytes() < rows * dim * 4 || y.bytes() < rows * dim * 2 || w.bytes() < dim * 4 {
        return Err(INVALID);
    }
    // SAFETY: as above
    let rc = unsafe { ffi::gb_glm_rmsnorm_f32_f16(x.ptr(), w.ptr(), y.ptr(), i32_of(rows)?, i32_of(dim)?, eps, st.raw()) };
    CudaError::check(rc)
}

/// `x += y` (fp32, in place), then `y_out = rmsnorm(x)` in fp16 with the new `x`. `x` and `y` are distinct buffers.
#[allow(clippy::too_many_arguments)]
pub fn add_rmsnorm_f32(
    x: &mut DevBuf,
    y: &DevBuf,
    w: &DevBuf,
    y_out: &mut DevBuf,
    rows: usize,
    dim: usize,
    eps: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    if x.bytes() < rows * dim * 4 || y.bytes() < rows * dim * 4 || y_out.bytes() < rows * dim * 2 || w.bytes() < dim * 4 {
        return Err(INVALID);
    }
    // SAFETY: as above
    let rc = unsafe {
        ffi::gb_glm_add_rmsnorm_f32(x.ptr(), y.ptr(), w.ptr(), y_out.ptr(), i32_of(rows)?, i32_of(dim)?, eps, st.raw())
    };
    CudaError::check(rc)
}

/// `x += y` over `n` fp32 elements (`n % 4 == 0`), in place; distinct buffers.
pub fn add_f32(x: &mut DevBuf, y: &DevBuf, n: usize, st: &Stream) -> Result<(), CudaError> {
    if x.bytes() < n * 4 || y.bytes() < n * 4 {
        return Err(INVALID);
    }
    let n = i64::try_from(n).map_err(|_| INVALID)?;
    // SAFETY: as above
    let rc = unsafe { ffi::gb_glm_add_f32(x.ptr(), y.ptr(), n, st.raw()) };
    CudaError::check(rc)
}

/// One row of the keyed pick (`csrc/glm_mtp.cu::KeyedRow`, 48 bytes): the sampling of the sequence the row belongs to and where its depth-0 draft is
/// compared with the target's sample (the sequence's length, `gb_spec::draft_position(len, 0)`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KeyedRow {
    pub seed: u64,
    pub first_position: u64,
    pub temperature: f64,
    pub top_p: f64,
    pub min_p: f64,
    pub top_k: i32,
    /// 0: the row takes its first candidate (the argmax); 1: the target's keyed draw over its candidates.
    pub keyed: i32,
}

const _: () = assert!(core::mem::size_of::<KeyedRow>() == 48);

/// The draft head's proposals as the target's keyed draw: for each of `rows` rows, its `count` best candidates (`ids` i32 and `vals` f32 `[rows, count]`, by
/// value descending and id ascending, ids already TOKEN ids) and `params[r]`; writes the pick of every row to `out_ids` / `out_vals` (i32 / f32 `[rows]`).
/// A keyed row draws Gumbel noise keyed on `(seed, first_position + depth, id)` over the candidates the target's sampler keeps (the first `top_k`, `top_p`,
/// `min_p`), exactly `gb_sampling::try_choose` on that list; any other row takes candidate 0. Row-invariant: a row reads only its own list and record.
/// `1 <= count <= 64`; `params` is a device array of [`KeyedRow`].
#[allow(clippy::too_many_arguments)]
pub fn keyed_pick(
    ids: &DevBuf,
    vals: &DevBuf,
    count: usize,
    rows: usize,
    params: &DevBuf,
    depth: usize,
    out_ids: &mut DevBuf,
    out_vals: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    if count == 0
        || count > 64
        || ids.bytes() < rows * count * 4
        || vals.bytes() < rows * count * 4
        || params.bytes() < rows * core::mem::size_of::<KeyedRow>()
        || out_ids.bytes() < rows * 4
        || out_vals.bytes() < rows * 4
    {
        return Err(INVALID);
    }
    // SAFETY: the buffers hold the extents checked above and the launch is asynchronous on `st`
    let rc = unsafe {
        ffi::gb_glm_keyed_pick(
            ids.ptr() as *const i32,
            vals.ptr() as *const f32,
            i32_of(count)?,
            i32_of(rows)?,
            params.ptr(),
            i32_of(depth)?,
            out_ids.ptr() as *mut i32,
            out_vals.ptr() as *mut f32,
            st.raw(),
        )
    };
    CudaError::check(rc)
}
