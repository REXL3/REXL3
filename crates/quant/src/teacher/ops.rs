//! Launch wrappers of `csrc/teacher.cu` and the GEMM call forms the teacher issues to cuBLAS.
//!
//! The GEMM forms are those PyTorch issues for the same operations (row-major tensors on column-major cuBLAS), so a
//! parity run gets the same kernels: `linear` is F.linear without bias, `gram` is torch.mm(x.T, x), and
//! `linear_batched_shared_x` is torch.matmul((1, rows, in) x, (n, in, out) w^T) with the batch stride of x zero.

use crate::core::dev::{Gpu, QuantError, Result};
use crate::gpu::cublas::Op;
use crate::gpu::GpuError;
use core::ffi::c_void;

mod ffi {
    use core::ffi::c_void;
    crate::gpu::foreign! { 100;
        pub fn tq_bf16_to_f32(out: *mut f32, inp: *const u16, n: i64, s: *mut c_void) -> i32;
        pub fn tq_embed_bf16(out: *mut f32, table: *const u16, ids: *const i32, n: i64, h: i64, s: *mut c_void) -> i32;
        pub fn tq_rmsnorm(
            out: *mut f32, ldo: i64, x: *const f32, ldx: i64, w: *const f32, gate: *const f32, ldg: i64, rows: i64, n: i32,
            eps: f32, mode: i32, s: *mut c_void,
        ) -> i32;
        pub fn tq_conv_silu(out: *mut f32, x: *const f32, w: *const f32, b: i32, t: i32, c: i32, k: i32, s: *mut c_void) -> i32;
        pub fn tq_gdn_gates(
            g: *mut f32, beta: *mut f32, a: *const f32, b: *const f32, a_log: *const f32, dt_bias: *const f32, rows: i64,
            heads: i32, s: *mut c_void,
        ) -> i32;
        pub fn tq_l2norm_f64(
            out: *mut f64, x: *const f32, ld: i64, off: i64, bt: i64, heads: i32, n: i32, eps: f64, scale: f64, s: *mut c_void,
        ) -> i32;
        pub fn tq_gdn_f64(
            o: *mut f32, q: *const f64, k: *const f64, x: *const f32, ldx: i64, v_off: i64, g: *const f32, beta: *const f32,
            b: i32, t: i32, hk: i32, hv: i32, dk: i32, dv: i32, s: *mut c_void,
        ) -> i32;
        pub fn tq_silu_mul(
            out: *mut f32, ldo: i64, a: *const f32, lda: i64, b: *const f32, ldb: i64, rows: i64, n: i32, s: *mut c_void,
        ) -> i32;
        pub fn tq_mul_sigmoid(out: *mut f32, a: *const f32, g: *const f32, n: i64, s: *mut c_void) -> i32;
        pub fn tq_shared_combine(
            out: *mut f32, moe: *const f32, shv: *const f32, gl: *const f32, rows: i64, h: i32, s: *mut c_void,
        ) -> i32;
        pub fn tq_route(
            top_i: *mut i32, top_w: *mut f32, counts: *mut u32, logits: *const f32, rows: i64, e: i32, k: i32, s: *mut c_void,
        ) -> i32;
        pub fn tq_scale_add_rows(
            out: *mut f32, y: *const f32, tok: *const i32, w: *const f32, rows: i64, h: i32, s: *mut c_void,
        ) -> i32;
        pub fn tq_gather_rows(dst: *mut f32, src: *const f32, idx: *const i32, rows: i64, cols: i32, s: *mut c_void) -> i32;
        pub fn tq_rope(x: *mut f32, rows: i64, t: i32, h: i32, hd: i32, rd: i32, inv_freq: *const f32, s: *mut c_void) -> i32;
        pub fn tq_attn_f64(
            o: *mut f32, q: *const f32, k: *const f32, v: *const f32, b: i32, t: i32, hq: i32, hkv: i32, d: i32, scale: f64,
            s: *mut c_void,
        ) -> i32;
        pub fn tq_nonfinite_rows(bad: *mut u32, x: *const f32, rows: i64, cols: i32, s: *mut c_void) -> i32;
    }
}

fn ck(rc: i32) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else if cfg!(gb_quant_no_cuda) {
        Err(GpuError::NoCuda.into())
    } else {
        Err(GpuError::Cuda(rc).into())
    }
}

fn s(g: &Gpu) -> *mut c_void {
    g.s()
}

fn int(v: usize, what: &str) -> Result<i32> {
    i32::try_from(v).map_err(|_| QuantError::Shape(format!("{what} = {v} exceeds i32")))
}

/// y (rows x out) = x (rows x in) w^T, w (out x in) row-major: `sgemm(T, N, out, rows, in, w, in, x, in, y, out)`.
///
/// # Safety
/// Device pointers of the implied sizes.
pub unsafe fn linear(g: &Gpu, x: *const f32, rows: usize, inp: usize, w: *const f32, out: usize, y: *mut f32) -> Result<()> {
    if rows == 0 {
        return Ok(());
    }
    unsafe { g.sgemm(Op::T, Op::N, out, rows, inp, 1.0, w, inp, x, inp, 0.0, y, out) }
}

/// c (k x k) = x^T x for x (rows x k) row-major (torch.mm(x.T, x)): `sgemm(N, T, k, k, rows, x, k, x, k, c, k)`.
///
/// # Safety
/// Device pointers of the implied sizes.
pub unsafe fn gram(g: &Gpu, x: *const f32, rows: usize, k: usize, c: *mut f32) -> Result<()> {
    unsafe { g.sgemm(Op::N, Op::T, k, k, rows, 1.0, x, k, x, k, 0.0, c, k) }
}

/// y_b (rows x out) = x w_b^T for b < batch, w_b at w + b * out * inp, y_b at y + b * rows * out.
///
/// # Safety
/// Device pointers of the implied sizes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn linear_batched_shared_x(
    g: &Gpu,
    x: *const f32,
    rows: usize,
    inp: usize,
    w: *const f32,
    out: usize,
    batch: usize,
    y: *mut f32,
) -> Result<()> {
    unsafe {
        g.blas.sgemm_strided_batched(
            Op::T,
            Op::N,
            int(out, "out")?,
            int(rows, "rows")?,
            int(inp, "in")?,
            1.0,
            w,
            int(inp, "in")?,
            (out * inp) as i64,
            x,
            int(inp, "in")?,
            0,
            0.0,
            y,
            int(out, "out")?,
            (rows * out) as i64,
            int(batch, "batch")?,
        )?
    };
    Ok(())
}

macro_rules! launch {
    ($e:expr) => {
        ck(unsafe { $e })
    };
}

pub(crate) fn bf16_to_f32(g: &Gpu, out: *mut f32, inp: *const u16, n: usize) -> Result<()> {
    launch!(ffi::tq_bf16_to_f32(out, inp, n as i64, s(g)))
}
pub(crate) fn embed_bf16(g: &Gpu, out: *mut f32, table: *const u16, ids: *const i32, n: usize, h: usize) -> Result<()> {
    launch!(ffi::tq_embed_bf16(out, table, ids, n as i64, h as i64, s(g)))
}

/// mode 0: (x rsqrt(mean x^2 + eps)) (1 + w); mode 1: (w (x rsqrt(mean x^2 + eps))) silu(gate). Rows of n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rmsnorm(
    g: &Gpu,
    out: *mut f32,
    ldo: usize,
    x: *const f32,
    ldx: usize,
    w: *const f32,
    gate: Option<(*const f32, usize)>,
    rows: usize,
    n: usize,
    eps: f32,
) -> Result<()> {
    let (gp, ldg, mode) = match gate {
        Some((p, ld)) => (p, ld, 1),
        None => (core::ptr::null(), 0, 0),
    };
    launch!(ffi::tq_rmsnorm(out, ldo as i64, x, ldx as i64, w, gp, ldg as i64, rows as i64, int(n, "n")?, eps, mode, s(g)))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_silu(
    g: &Gpu,
    out: *mut f32,
    x: *const f32,
    w: *const f32,
    b: usize,
    t: usize,
    c: usize,
    k: usize,
) -> Result<()> {
    launch!(ffi::tq_conv_silu(out, x, w, int(b, "B")?, int(t, "T")?, int(c, "C")?, int(k, "K")?, s(g)))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_gates(
    g: &Gpu,
    gout: *mut f32,
    beta: *mut f32,
    a: *const f32,
    b: *const f32,
    a_log: *const f32,
    dt_bias: *const f32,
    rows: usize,
    heads: usize,
) -> Result<()> {
    launch!(ffi::tq_gdn_gates(gout, beta, a, b, a_log, dt_bias, rows as i64, int(heads, "heads")?, s(g)))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn l2norm_f64(
    g: &Gpu,
    out: *mut f64,
    x: *const f32,
    ld: usize,
    off: usize,
    bt: usize,
    heads: usize,
    n: usize,
    eps: f64,
    scale: f64,
) -> Result<()> {
    launch!(ffi::tq_l2norm_f64(out, x, ld as i64, off as i64, bt as i64, int(heads, "heads")?, int(n, "n")?, eps, scale, s(g)))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_f64(
    g: &Gpu,
    o: *mut f32,
    q: *const f64,
    k: *const f64,
    x: *const f32,
    ldx: usize,
    v_off: usize,
    gg: *const f32,
    beta: *const f32,
    dims: [usize; 6],
) -> Result<()> {
    let [b, t, hk, hv, dk, dv] = dims;
    launch!(ffi::tq_gdn_f64(
        o,
        q,
        k,
        x,
        ldx as i64,
        v_off as i64,
        gg,
        beta,
        int(b, "B")?,
        int(t, "T")?,
        int(hk, "Hk")?,
        int(hv, "Hv")?,
        int(dk, "dk")?,
        int(dv, "dv")?,
        s(g)
    ))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn silu_mul(
    g: &Gpu,
    out: *mut f32,
    ldo: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    rows: usize,
    n: usize,
) -> Result<()> {
    launch!(ffi::tq_silu_mul(out, ldo as i64, a, lda as i64, b, ldb as i64, rows as i64, int(n, "n")?, s(g)))
}
pub(crate) fn mul_sigmoid(g: &Gpu, out: *mut f32, a: *const f32, gate: *const f32, n: usize) -> Result<()> {
    launch!(ffi::tq_mul_sigmoid(out, a, gate, n as i64, s(g)))
}
pub(crate) fn shared_combine(
    g: &Gpu,
    out: *mut f32,
    moe: *const f32,
    shv: *const f32,
    gl: *const f32,
    rows: usize,
    h: usize,
) -> Result<()> {
    launch!(ffi::tq_shared_combine(out, moe, shv, gl, rows as i64, int(h, "H")?, s(g)))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn route(
    g: &Gpu,
    top_i: *mut i32,
    top_w: *mut f32,
    counts: *mut u32,
    logits: *const f32,
    rows: usize,
    e: usize,
    k: usize,
) -> Result<()> {
    launch!(ffi::tq_route(top_i, top_w, counts, logits, rows as i64, int(e, "E")?, int(k, "K")?, s(g)))
}
pub(crate) fn scale_add_rows(
    g: &Gpu,
    out: *mut f32,
    y: *const f32,
    tok: *const i32,
    w: *const f32,
    rows: usize,
    h: usize,
) -> Result<()> {
    launch!(ffi::tq_scale_add_rows(out, y, tok, w, rows as i64, int(h, "H")?, s(g)))
}
pub(crate) fn gather_rows(g: &Gpu, dst: *mut f32, src: *const f32, idx: *const i32, rows: usize, cols: usize) -> Result<()> {
    launch!(ffi::tq_gather_rows(dst, src, idx, rows as i64, int(cols, "cols")?, s(g)))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope(
    g: &Gpu,
    x: *mut f32,
    rows: usize,
    t: usize,
    h: usize,
    hd: usize,
    rd: usize,
    inv_freq: *const f32,
) -> Result<()> {
    launch!(ffi::tq_rope(x, rows as i64, int(t, "T")?, int(h, "H")?, int(hd, "hd")?, int(rd, "rd")?, inv_freq, s(g)))
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_f64(
    g: &Gpu,
    o: *mut f32,
    q: *const f32,
    k: *const f32,
    v: *const f32,
    dims: [usize; 5],
    scale: f64,
) -> Result<()> {
    let [b, t, hq, hkv, d] = dims;
    launch!(ffi::tq_attn_f64(o, q, k, v, int(b, "B")?, int(t, "T")?, int(hq, "Hq")?, int(hkv, "Hkv")?, int(d, "d")?, scale, s(g)))
}
pub(crate) fn nonfinite_rows(g: &Gpu, bad: *mut u32, x: *const f32, rows: usize, cols: usize) -> Result<()> {
    launch!(ffi::tq_nonfinite_rows(bad, x, rows as i64, int(cols, "cols")?, s(g)))
}
