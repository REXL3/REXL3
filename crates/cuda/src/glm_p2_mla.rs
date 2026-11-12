//! GLM-5.3-Flash prefill under contract P2 (Lane P step (d); `docs/GLM_PREFILL_P2.md` section 12): the MLA absorb and expand of a prompt
//! chunk's rows on tensor cores (`csrc/glm_p2_mla.cu`), in the dense P2 GEMM's fold128 K order: per 128-k block the chain of its 8 k16
//! MMAs from +0.0 (fp16 operands, fp32 accumulate), the blocks' partials added in order, one round to fp16. Layouts, scratch and dtype
//! are P1's ([`crate::mla::absorb`] / [`crate::mla::expand`], one fp32 FMA chain per output on CUDA cores); the bits are P2's
//! ([`ARITH_ID`]), a row's a function of that row and the weights alone. [`mla_rows_auto_p2`] and [`mla_rows_pf_p2`] are P1's
//! [`crate::mla::mla_rows_auto`] and [`crate::mla_prefill::mla_rows_pf`] with these two in place of P1's (the latent attention between them
//! is P1's kernel under both contracts).

use crate::attention::{BlockTable, RowIndex};
use crate::exl3_linear::Dtype;
use crate::mla::{attend_auto, LatentCache, MlaScratch, MlaWeights, SparseSel, LATENT_DIM, NOPE_DIM, N_HEADS};
use crate::mla_prefill::attend_pf;
use crate::rt::{CudaError, DevBuf, Stream};
use std::sync::atomic::{AtomicU64, Ordering};

/// The arithmetic id of the P2 absorb / expand (the P2 tag's and fingerprint's `mla-ae=`): a change to this file's bits needs a new one.
pub const ARITH_ID: &str = "glm-mla-ae-fold128-v1";

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_glm_p2_mla_absorb(
            q: *const c_void,
            q_row_stride: i64,
            w_uk: *const c_void,
            q_lat: *mut c_void,
            rows: i32,
            nh: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_glm_p2_mla_expand(
            o_lat: *const c_void,
            w_uvt: *const c_void,
            out: *mut c_void,
            rows: i32,
            nh: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

static RUNS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static RUNS_HERE: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

/// Absorb and expand launches of this module, process-wide (the launch audit of a P2 prompt chunk).
pub fn runs() -> u64 {
    RUNS.load(Ordering::Relaxed)
}

/// [`runs`] made by this thread (a loopback TP rank is a thread).
pub fn runs_here() -> u64 {
    RUNS_HERE.with(|c| c.get())
}

fn count() {
    RUNS.fetch_add(1, Ordering::Relaxed);
    RUNS_HERE.with(|c| c.set(c.get() + 1));
}

fn check_f16(w: &MlaWeights) {
    assert!(
        w.dtype == Dtype::F16,
        "the P2 MLA absorb / expand ({ARITH_ID}) is fp16 only: kv_b is bound as {:?} (GLM binds it as fp16)",
        w.dtype
    );
}

/// P2 absorb: `q_lat[r][h][c] = sum_d q[r][h * 256 + d] * W_UK[h][d][c]` for `n_rows` rows of all 64 heads, as [`crate::mla::absorb`]'s
/// layouts. fp16 only.
pub fn absorb(
    q: &DevBuf,
    q_row_stride: usize,
    w: &MlaWeights,
    q_lat: &mut DevBuf,
    n_rows: usize,
    stream: &Stream,
) -> Result<(), CudaError> {
    check_f16(w);
    assert!(q_row_stride >= N_HEADS * NOPE_DIM && q_row_stride % 8 == 0, "q rows must be 16-byte aligned and hold 64 x 256");
    assert!(n_rows == 0 || q.bytes() >= ((n_rows - 1) * q_row_stride + N_HEADS * NOPE_DIM) * 2);
    assert!(q_lat.bytes() >= n_rows * N_HEADS * LATENT_DIM * 2);
    count();
    let rc = unsafe {
        ffi::gb_glm_p2_mla_absorb(
            q.ptr(),
            q_row_stride as i64,
            w.uk.ptr(),
            q_lat.ptr(),
            n_rows as i32,
            N_HEADS as i32,
            stream.raw(),
        )
    };
    CudaError::check(rc)
}

/// P2 expand: `out[r][h * 256 + n] = sum_c o_lat[r][h][c] * W_UV[h][c][n]`, as [`crate::mla::expand`]'s layouts. fp16 only.
pub fn expand(o_lat: &DevBuf, w: &MlaWeights, out: &mut DevBuf, n_rows: usize, stream: &Stream) -> Result<(), CudaError> {
    check_f16(w);
    assert!(o_lat.bytes() >= n_rows * N_HEADS * LATENT_DIM * 2);
    assert!(out.bytes() >= n_rows * N_HEADS * NOPE_DIM * 2);
    count();
    let rc =
        unsafe { ffi::gb_glm_p2_mla_expand(o_lat.ptr(), w.uvt.ptr(), out.ptr(), n_rows as i32, N_HEADS as i32, stream.raw()) };
    CudaError::check(rc)
}

/// [`crate::mla::mla_rows_auto`] under P2: [`absorb`], P1's [`attend_auto`], [`expand`].
#[allow(clippy::too_many_arguments)]
pub fn mla_rows_auto_p2(
    q: &DevBuf,
    q_row_stride: usize,
    w: &MlaWeights,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    sel: &SparseSel,
    out: &mut DevBuf,
    max_context: usize,
    scratch: &mut MlaScratch,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert_eq!(w.dtype, cache.dtype, "weights and cache must share one dtype");
    assert!(n_rows <= scratch.max_rows && max_context <= scratch.max_context, "scratch too small");
    absorb(q, q_row_stride, w, &mut scratch.q_lat, n_rows, stream)?;
    attend_auto(&scratch.q_lat, cache, table, rows, n_rows, sel, &mut scratch.o_lat, max_context, &mut scratch.ws, stream)?;
    expand(&scratch.o_lat, w, out, n_rows, stream)
}

/// [`crate::mla_prefill::mla_rows_pf`] under P2 (a large chunk's slab): [`absorb`], P1's fused [`attend_pf`], [`expand`].
#[allow(clippy::too_many_arguments)]
pub fn mla_rows_pf_p2(
    q: &DevBuf,
    q_row_stride: usize,
    w: &MlaWeights,
    cache: &LatentCache,
    table: &BlockTable,
    rows: &RowIndex,
    n_rows: usize,
    sel: &SparseSel,
    out: &mut DevBuf,
    scratch: &mut MlaScratch,
    stream: &Stream,
) -> Result<(), CudaError> {
    assert_eq!(w.dtype, cache.dtype, "weights and cache must share one dtype");
    assert!(n_rows <= scratch.max_rows, "scratch too small");
    absorb(q, q_row_stride, w, &mut scratch.q_lat, n_rows, stream)?;
    attend_pf(&scratch.q_lat, cache, table, rows, n_rows, sel, &mut scratch.o_lat, stream)?;
    expand(&scratch.o_lat, w, out, n_rows, stream)
}
