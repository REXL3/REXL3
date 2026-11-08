//! Kernels of the Qwen3.8 vision tower (`csrc/vision.cu`, spec in `docs/VISION.md`): position-embedding add, LayerNorm with
//! bias, axial 2D RoPE, residual add, tanh GELU, fp32 -> bf16 export, and bidirectional flash attention for head_dim 72.
//!
//! The residual stream is fp32, GEMM operands and activations fp16. Every function launches on `stream` and returns without
//! synchronising; buffer sizes are checked (a short buffer panics rather than faulting on the device).

pub use gpu::*;

/// Head dimension of the vision attention.
pub const HEAD_DIM: usize = 72;

mod gpu {
    use crate::{CudaError, DevBuf, Stream};
    use core::ffi::c_void;

    mod ffi {
        use core::ffi::c_void;
        crate::launchers! {
            pub fn gb_vis_pos_add(
                x: *mut c_void,
                bias: *const c_void,
                table: *const c_void,
                idx: *const c_void,
                w: *const c_void,
                p: i32,
                h: i32,
                s: *mut c_void,
            ) -> i32;
            pub fn gb_vis_layernorm(
                x: *const c_void,
                x_is_f32: i32,
                w: *const c_void,
                b: *const c_void,
                y: *mut c_void,
                rows: i32,
                dim: i32,
                eps: f32,
                s: *mut c_void,
            ) -> i32;
            pub fn gb_vis_rope2d(q: *mut c_void, k: *mut c_void, cs: *const c_void, p: i32, heads: i32, s: *mut c_void) -> i32;
            pub fn gb_vis_add_f16(x: *mut c_void, y: *const c_void, n: usize, s: *mut c_void) -> i32;
            pub fn gb_vis_gelu_tanh(x: *mut c_void, n: usize, s: *mut c_void) -> i32;
            pub fn gb_vis_f32_to_bf16(x: *const c_void, y: *mut c_void, n: usize, s: *mut c_void) -> i32;
            pub fn gb_exl3_prefill_gemm(
                xh: *const c_void,
                w: *const c_void,
                c: *mut c_void,
                m: i32,
                k: i32,
                n: i32,
                s: *mut c_void,
            ) -> i32;
            pub fn gb_vis_attention(
                q: *const c_void,
                k: *const c_void,
                v: *const c_void,
                out: *mut c_void,
                p: i32,
                heads: i32,
                s: *mut c_void,
            ) -> i32;
        }
    }

    fn need(b: &DevBuf, bytes: usize, what: &str) {
        assert!(b.bytes() >= bytes, "vision {what}: buffer of {} bytes, need {bytes}", b.bytes());
    }

    fn i(v: usize) -> Result<i32, CudaError> {
        i32::try_from(v).map_err(|_| CudaError(1))
    }

    /// `x[p, :] = (x[p, :] + bias) + bf16 bilinear position embedding of patch p`. `x` f32 `[p, h]`; `bias` f16 `[h]` (optional);
    /// `table` f32 `[n, h]` holding bf16 values; `idx` i32 `[p, 4]`; `w` f32 `[p, 4]` (bf16-rounded weights).
    #[allow(clippy::too_many_arguments)]
    pub fn pos_add(
        x: &mut DevBuf,
        bias: Option<&DevBuf>,
        table: &DevBuf,
        idx: &DevBuf,
        w: &DevBuf,
        p: usize,
        h: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(x, p * h * 4, "pos_add x");
        need(idx, p * 16, "pos_add idx");
        need(w, p * 16, "pos_add w");
        let bp = bias.map_or(core::ptr::null(), |b| {
            need(b, h * 2, "pos_add bias");
            b.ptr() as *const c_void
        });
        CudaError::check(unsafe { ffi::gb_vis_pos_add(x.ptr(), bp, table.ptr(), idx.ptr(), w.ptr(), i(p)?, i(h)?, st.raw()) })
    }

    /// `y = LayerNorm(x) * w + b` over rows of `dim`, fp32 statistics; `x` f32 or f16, `w` and `b` f16 `[dim]`, `y` f16.
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm(
        x: &DevBuf,
        x_is_f32: bool,
        w: &DevBuf,
        b: &DevBuf,
        y: &mut DevBuf,
        rows: usize,
        dim: usize,
        eps: f32,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(x, rows * dim * if x_is_f32 { 4 } else { 2 }, "layernorm x");
        need(y, rows * dim * 2, "layernorm y");
        CudaError::check(unsafe {
            ffi::gb_vis_layernorm(x.ptr(), x_is_f32 as i32, w.ptr(), b.ptr(), y.ptr(), i(rows)?, i(dim)?, eps, st.raw())
        })
    }

    /// NEOX 2D RoPE in place on `q` and `k` f16 `[p, heads * 72]`; `cs` f32 `[p, 2, 36]` (cosines then sines).
    pub fn rope2d(q: &mut DevBuf, k: &mut DevBuf, cs: &DevBuf, p: usize, heads: usize, st: &Stream) -> Result<(), CudaError> {
        need(q, p * heads * 144, "rope2d q");
        need(k, p * heads * 144, "rope2d k");
        need(cs, p * 72 * 4, "rope2d cs");
        CudaError::check(unsafe { ffi::gb_vis_rope2d(q.ptr(), k.ptr(), cs.ptr(), i(p)?, i(heads)?, st.raw()) })
    }

    /// `x += y` over `n` elements, `x` f32 and `y` f16.
    pub fn add_f16(x: &mut DevBuf, y: &DevBuf, n: usize, st: &Stream) -> Result<(), CudaError> {
        need(x, n * 4, "add_f16 x");
        need(y, n * 2, "add_f16 y");
        CudaError::check(unsafe { ffi::gb_vis_add_f16(x.ptr(), y.ptr(), n, st.raw()) })
    }

    /// tanh-approximate GELU in place on `n` f16 elements.
    pub fn gelu_tanh(x: &mut DevBuf, n: usize, st: &Stream) -> Result<(), CudaError> {
        need(x, n * 2, "gelu x");
        CudaError::check(unsafe { ffi::gb_vis_gelu_tanh(x.ptr(), n, st.raw()) })
    }

    /// `y = bf16(x)` over `n` elements.
    pub fn f32_to_bf16(x: &DevBuf, y: &mut DevBuf, n: usize, st: &Stream) -> Result<(), CudaError> {
        need(x, n * 4, "f32_to_bf16 x");
        need(y, n * 2, "f32_to_bf16 y");
        CudaError::check(unsafe { ffi::gb_vis_f32_to_bf16(x.ptr(), y.ptr(), n, st.raw()) })
    }

    /// `c` f32 `[m, n] = a` f16 `[m, k]` `@ w` f16 `[k, n]` with fp32 accumulation (cuBLAS; the patch embedding).
    pub fn gemm_f16(a: &DevBuf, w: &DevBuf, c: &mut DevBuf, m: usize, k: usize, n: usize, st: &Stream) -> Result<(), CudaError> {
        need(a, m * k * 2, "gemm a");
        need(w, k * n * 2, "gemm w");
        need(c, m * n * 4, "gemm c");
        CudaError::check(unsafe { ffi::gb_exl3_prefill_gemm(a.ptr(), w.ptr(), c.ptr(), i(m)?, i(k)?, i(n)?, st.raw()) })
    }

    /// Full attention among the `p` patches of one image: `q`, `k`, `v`, `out` f16 `[p, heads * 72]` (4-byte aligned).
    pub fn attention(
        q: &DevBuf,
        k: &DevBuf,
        v: &DevBuf,
        out: &mut DevBuf,
        p: usize,
        heads: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let bytes = p * heads * 144;
        need(q, bytes, "attention q");
        need(k, bytes, "attention k");
        need(v, bytes, "attention v");
        need(out, bytes, "attention out");
        CudaError::check(unsafe { ffi::gb_vis_attention(q.ptr(), k.ptr(), v.ptr(), out.ptr(), i(p)?, i(heads)?, st.raw()) })
    }
}
