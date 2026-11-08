//! Kernels of the GLM-5.3-Flash vision tower (`csrc/glm_vision.cu`): weight preparation, the fused residual RMSNorm, qkv preparation (bias, head norm,
//! 2-D rope), bidirectional flash attention for head dimension 64, the two gated activations, the merger's LayerNorm and GELU, the embedding splice,
//! and the cuBLAS GEMM over weights as the pack stores them (`[out, in]`).
//!
//! Every function launches on `stream` and returns without synchronising; buffer sizes are checked (a short buffer panics rather than faulting on the
//! device). The numerics are documented in the `.cu` file: they follow the ExLlamaV3 fork, stage for stage.

pub use gpu::*;

/// Head dimension of the tower's attention.
pub const HEAD_DIM: usize = 64;
/// Hidden width of the tower.
pub const HIDDEN: usize = 1024;
/// Heads of the tower's attention.
pub const HEADS: usize = 16;

/// `flags` of [`qkv_prep`].
pub mod qkv {
    /// Swap the row and column axes of the rope angles (a mutation).
    pub const SWAP_AXES: i32 = 1;
    /// Skip the bias add (a mutation).
    pub const NO_BIAS: i32 = 2;
    /// Skip the per-head RMSNorm of q and k (a mutation).
    pub const NO_HEAD_NORM: i32 = 4;
    /// Skip the rope (a mutation).
    pub const NO_ROPE: i32 = 8;
}

/// `mask` of [`attention`].
pub mod mask {
    pub const NONE: i32 = 0;
    /// Causal (a mutation: the tower is bidirectional).
    pub const CAUSAL: i32 = 1;
}

mod gpu {
    use crate::{CudaError, DevBuf, Stream};
    use core::ffi::c_void;

    mod ffi {
        use core::ffi::c_void;
        crate::launchers! {
            pub fn gb_glmvis_bf16_to_f16(p: *mut c_void, n: usize, bad: *mut i32, s: *mut c_void) -> i32;
            pub fn gb_glmvis_fold_patch(src: *const c_void, dst: *mut c_void, o: i32, kp: i32, slots: i32, s: *mut c_void) -> i32;
            pub fn gb_glmvis_permute_downsample(src: *const c_void, dst: *mut c_void, o: i32, c: i32, mode: i32, bad: *mut i32, s: *mut c_void) -> i32;
            pub fn gb_glmvis_upload_rows(dst: *mut c_void, dpitch: usize, src: *const c_void, spitch: usize, width: usize, rows: usize, s: *mut c_void) -> i32;
            pub fn gb_glmvis_rmsnorm(
                x: *mut c_void,
                raw: *const c_void,
                bias: *const c_void,
                w: *const c_void,
                y: *mut c_void,
                rows: i32,
                eps: f32,
                s: *mut c_void,
            ) -> i32;
            pub fn gb_glmvis_qkv_prep(
                qkv: *mut c_void,
                bias: *const c_void,
                qn: *const c_void,
                kn: *const c_void,
                inv_freq: *const c_void,
                p: i32,
                gw: i32,
                eps: f32,
                flags: i32,
                s: *mut c_void,
            ) -> i32;
            pub fn gb_glmvis_mma_peak(sink: *mut c_void, blocks: i32, iters: i32, s: *mut c_void) -> i32;
            pub fn gb_glmvis_attention(qkv: *const c_void, out: *mut c_void, p: i32, mask: i32, variant: i32, s: *mut c_void) -> i32;
            pub fn gb_glmvis_swiglu_block(
                gu: *mut c_void,
                bg: *const c_void,
                bu: *const c_void,
                i: i32,
                ld: i32,
                limit: f32,
                flags: i32,
                rows: usize,
                s: *mut c_void,
            ) -> i32;
            pub fn gb_glmvis_layernorm(x: *const c_void, w: *const c_void, b: *const c_void, y: *mut c_void, rows: i32, dim: i32, eps: f32, s: *mut c_void) -> i32;
            pub fn gb_glmvis_gelu(x: *mut c_void, n: usize, flags: i32, s: *mut c_void) -> i32;
            pub fn gb_glmvis_swiglu_merger(gu: *mut c_void, m: i32, ld: i32, limit: f32, rows: usize, flags: i32, s: *mut c_void) -> i32;
            pub fn gb_glmvis_bias_add_f16(x: *mut c_void, bias: *const c_void, n: i32, rows: usize, s: *mut c_void) -> i32;
            pub fn gb_glmvis_bias_add_f32(x: *mut c_void, bias: *const c_void, n: i32, rows: usize, s: *mut c_void) -> i32;
            pub fn gb_glmvis_splice(
                streams: *mut c_void,
                feats: *const c_void,
                dst_row: i32,
                src_row: i32,
                n: i32,
                hidden: i32,
                nstreams: i32,
                s: *mut c_void,
            ) -> i32;
            pub fn gb_glmvis_gemm(
                a: *const c_void,
                lda: i32,
                w: *const c_void,
                c: *mut c_void,
                ldc: i32,
                m: i32,
                k: i32,
                n: i32,
                c_is_f32: i32,
                ws: *mut c_void,
                ws_bytes: usize,
                s: *mut c_void,
            ) -> i32;
        }
    }

    fn need(b: &DevBuf, bytes: usize, what: &str) {
        assert!(b.bytes() >= bytes, "glm vision {what}: buffer of {} bytes, need {bytes}", b.bytes());
    }

    fn i(v: usize) -> Result<i32, CudaError> {
        i32::try_from(v).map_err(|_| CudaError(1))
    }

    /// bf16 -> fp16 in place over `n` elements; `bad` (one device i32, zeroed by the caller) is incremented per result that is not finite.
    pub fn bf16_to_f16(p: &mut DevBuf, n: usize, bad: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
        need(p, n * 2, "bf16_to_f16 p");
        need(bad, 4, "bf16_to_f16 bad");
        CudaError::check(unsafe { ffi::gb_glmvis_bf16_to_f16(p.ptr(), n, bad.ptr() as *mut i32, st.raw()) })
    }

    /// The patch-embedding conv3d weight `[o, 3, 2, 14, 14]` bf16 folded over the two temporal slots into `[o, kp]` fp16 (`slots` 2; 1 keeps slot 0 only).
    pub fn fold_patch(src: &DevBuf, dst: &mut DevBuf, o: usize, kp: usize, slots: usize, st: &Stream) -> Result<(), CudaError> {
        need(src, o * 3 * 2 * 196 * 2, "fold_patch src");
        need(dst, o * kp * 2, "fold_patch dst");
        CudaError::check(unsafe { ffi::gb_glmvis_fold_patch(src.ptr(), dst.ptr(), i(o)?, i(kp)?, i(slots)?, st.raw()) })
    }

    /// The downsample conv2d weight `[o, c, 2, 2]` bf16 as the GEMM weight `[o, 4c]` fp16 (see the kernel for `mode`; 0 is the right one).
    pub fn permute_downsample(
        src: &DevBuf,
        dst: &mut DevBuf,
        o: usize,
        c: usize,
        mode: i32,
        bad: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(src, o * c * 4 * 2, "permute_downsample src");
        need(dst, o * c * 4 * 2, "permute_downsample dst");
        need(bad, 4, "permute_downsample bad");
        CudaError::check(unsafe {
            ffi::gb_glmvis_permute_downsample(src.ptr(), dst.ptr(), i(o)?, i(c)?, mode, bad.ptr() as *mut i32, st.raw())
        })
    }

    /// Copy `rows` rows of `width` bytes from host memory (row pitch `spitch`) into `dst` (row pitch `dpitch`).
    pub fn upload_rows(
        dst: &mut DevBuf,
        dpitch: usize,
        src: &[u16],
        spitch: usize,
        width: usize,
        rows: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(dst, dpitch * rows, "upload_rows dst");
        assert!(
            src.len() * 2 >= spitch * rows,
            "glm vision upload_rows: host source of {} bytes, need {}",
            src.len() * 2,
            spitch * rows
        );
        CudaError::check(unsafe {
            ffi::gb_glmvis_upload_rows(dst.ptr(), dpitch, src.as_ptr() as *const c_void, spitch, width, rows, st.raw())
        })
    }

    /// `y = RMSNorm(x) * w` (`x` f32 `[rows, 1024]`, `w` bf16 `[1024]`, `y` f16). With `raw` (f16 `[rows, 1024]`, the output of a GEMM) the residual add comes
    /// first: `x += f16(raw + bias)`, written back.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm(
        x: &mut DevBuf,
        raw: Option<&DevBuf>,
        bias: Option<&DevBuf>,
        w: &DevBuf,
        y: &mut DevBuf,
        rows: usize,
        eps: f32,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(x, rows * super::HIDDEN * 4, "rmsnorm x");
        need(w, super::HIDDEN * 2, "rmsnorm w");
        need(y, rows * super::HIDDEN * 2, "rmsnorm y");
        let rp = raw.map_or(core::ptr::null(), |r| {
            need(r, rows * super::HIDDEN * 2, "rmsnorm raw");
            r.ptr() as *const c_void
        });
        let bp = bias.map_or(core::ptr::null(), |b| {
            need(b, super::HIDDEN * 2, "rmsnorm bias");
            b.ptr() as *const c_void
        });
        CudaError::check(unsafe { ffi::gb_glmvis_rmsnorm(x.ptr(), rp, bp, w.ptr(), y.ptr(), i(rows)?, eps, st.raw()) })
    }

    /// The qkv GEMM's output `[p, 3072]` f16 -> q, k, v in place: bias, then q and k's per-head RMSNorm (bf16 gammas `qn`, `kn` of 64) and the 2-D rope
    /// (`inv_freq` f32 `[16]`, patches block-major over a grid `gw` wide).
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_prep(
        qkv: &mut DevBuf,
        bias: &DevBuf,
        qn: &DevBuf,
        kn: &DevBuf,
        inv_freq: &DevBuf,
        p: usize,
        gw: usize,
        eps: f32,
        flags: i32,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(qkv, p * 3 * super::HIDDEN * 2, "qkv_prep qkv");
        need(bias, 3 * super::HIDDEN * 2, "qkv_prep bias");
        need(qn, super::HEAD_DIM * 2, "qkv_prep qn");
        need(kn, super::HEAD_DIM * 2, "qkv_prep kn");
        need(inv_freq, 16 * 4, "qkv_prep inv_freq");
        CudaError::check(unsafe {
            ffi::gb_glmvis_qkv_prep(
                qkv.ptr(),
                bias.ptr(),
                qn.ptr(),
                kn.ptr(),
                inv_freq.ptr(),
                i(p)?,
                i(gw)?,
                eps,
                flags,
                st.raw(),
            )
        })
    }

    /// A benchmark: `blocks` CTAs of 8 warps each run `iters` rounds of 8 independent `mma.m16n8k16` (fp16 in, fp32 accumulate); `sink` holds 256 f32.
    /// FLOP = blocks * 8 * iters * 8 * 4096. The device's tensor-core peak for the tower's roofline.
    pub fn mma_peak(sink: &mut DevBuf, blocks: usize, iters: usize, st: &Stream) -> Result<(), CudaError> {
        need(sink, 256 * 4, "mma_peak sink");
        CudaError::check(unsafe { ffi::gb_glmvis_mma_peak(sink.ptr(), i(blocks)?, i(iters)?, st.raw()) })
    }

    /// Full attention among the `p` patches of one image: `qkv` f16 `[p, 3072]` (q | k | v, each `[p, 16, 64]`) -> `out` f16 `[p, 1024]`.
    pub fn attention(qkv: &DevBuf, out: &mut DevBuf, p: usize, mask: i32, st: &Stream) -> Result<(), CudaError> {
        attention_variant(qkv, out, p, mask, 2, st)
    }

    /// [`attention`] on the first version of the kernel (`variant` 1), which the default (2) equals bit for bit: the reference of the tests.
    pub fn attention_variant(
        qkv: &DevBuf,
        out: &mut DevBuf,
        p: usize,
        mask: i32,
        variant: i32,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(qkv, p * 3 * super::HIDDEN * 2, "attention qkv");
        need(out, p * super::HIDDEN * 2, "attention out");
        CudaError::check(unsafe { ffi::gb_glmvis_attention(qkv.ptr(), out.ptr(), i(p)?, mask, variant, st.raw()) })
    }

    /// The block MLP's activation in place on `gu` f16 `[rows, ld]` (gate in columns `[0, inter)`, up in `[inter, 2 inter)`), biases added in fp16.
    #[allow(clippy::too_many_arguments)]
    pub fn swiglu_block(
        gu: &mut DevBuf,
        bg: &DevBuf,
        bu: &DevBuf,
        inter: usize,
        ld: usize,
        limit: f32,
        flags: i32,
        rows: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(gu, rows * ld * 2, "swiglu_block gu");
        need(bg, inter * 2, "swiglu_block bg");
        need(bu, inter * 2, "swiglu_block bu");
        CudaError::check(unsafe {
            ffi::gb_glmvis_swiglu_block(gu.ptr(), bg.ptr(), bu.ptr(), i(inter)?, i(ld)?, limit, flags, rows, st.raw())
        })
    }

    /// LayerNorm over rows of `dim` (f16 in and out, f16 weight and bias).
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm(
        x: &DevBuf,
        w: &DevBuf,
        b: &DevBuf,
        y: &mut DevBuf,
        rows: usize,
        dim: usize,
        eps: f32,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(x, rows * dim * 2, "layernorm x");
        need(w, dim * 2, "layernorm w");
        need(b, dim * 2, "layernorm b");
        need(y, rows * dim * 2, "layernorm y");
        CudaError::check(unsafe {
            ffi::gb_glmvis_layernorm(x.ptr(), w.ptr(), b.ptr(), y.ptr(), i(rows)?, i(dim)?, eps, st.raw())
        })
    }

    /// Exact GELU in place on `n` f16 elements (`flags` 1: the tanh formula, a mutation).
    pub fn gelu(x: &mut DevBuf, n: usize, flags: i32, st: &Stream) -> Result<(), CudaError> {
        need(x, n * 2, "gelu x");
        CudaError::check(unsafe { ffi::gb_glmvis_gelu(x.ptr(), n, flags, st.raw()) })
    }

    /// The merger's activation in place on `gu` f16 `[rows, ld]` (gate in `[0, m)`, up in `[m, 2m)`): clamp, then silu (flags 1 skips the clamps).
    pub fn swiglu_merger(
        gu: &mut DevBuf,
        m: usize,
        ld: usize,
        limit: f32,
        rows: usize,
        flags: i32,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(gu, rows * ld * 2, "swiglu_merger gu");
        CudaError::check(unsafe { ffi::gb_glmvis_swiglu_merger(gu.ptr(), i(m)?, i(ld)?, limit, rows, flags, st.raw()) })
    }

    /// `x += bias` in place, f16 (`rows` rows of `n`).
    pub fn bias_add_f16(x: &mut DevBuf, bias: &DevBuf, n: usize, rows: usize, st: &Stream) -> Result<(), CudaError> {
        need(x, rows * n * 2, "bias_add_f16 x");
        need(bias, n * 2, "bias_add_f16 bias");
        CudaError::check(unsafe { ffi::gb_glmvis_bias_add_f16(x.ptr(), bias.ptr(), i(n)?, rows, st.raw()) })
    }

    /// `x += bias` in place, `x` f32, `bias` f16.
    pub fn bias_add_f32(x: &mut DevBuf, bias: &DevBuf, n: usize, rows: usize, st: &Stream) -> Result<(), CudaError> {
        need(x, rows * n * 4, "bias_add_f32 x");
        need(bias, n * 2, "bias_add_f32 bias");
        CudaError::check(unsafe { ffi::gb_glmvis_bias_add_f32(x.ptr(), bias.ptr(), i(n)?, rows, st.raw()) })
    }

    /// Write `n` rows of an image's f32 features (from row `src_row`) over the embedding rows `dst_row..` of the mHC streams `[rows][nstreams][hidden]` f32
    /// at `streams` (a device pointer of `stream_rows` rows). The embedding is the same row in every stream.
    ///
    /// # Safety
    /// `streams` must point to a live device buffer of at least `(dst_row + n) * nstreams * hidden` f32, written only by work on `st`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn splice(
        streams: *mut c_void,
        feats: &DevBuf,
        dst_row: usize,
        src_row: usize,
        n: usize,
        hidden: usize,
        nstreams: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(feats, (src_row + n) * hidden * 4, "splice feats");
        CudaError::check(unsafe {
            ffi::gb_glmvis_splice(streams, feats.ptr(), i(dst_row)?, i(src_row)?, i(n)?, i(hidden)?, i(nstreams)?, st.raw())
        })
    }

    /// `c [m, n] = a [m, k] (row stride lda) @ w [n, k]^T`: fp16 inputs, fp32 accumulate, `c` f16 or f32 (row stride `ldc`). `ws` is the cuBLAS workspace.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        a: &DevBuf,
        lda: usize,
        w: &DevBuf,
        c: &mut DevBuf,
        ldc: usize,
        m: usize,
        k: usize,
        n: usize,
        c_is_f32: bool,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(a, m.saturating_sub(1).saturating_mul(lda).saturating_add(k) * 2, "gemm a");
        need(w, n * k * 2, "gemm w");
        let esz = if c_is_f32 { 4 } else { 2 };
        need(c, m.saturating_sub(1).saturating_mul(ldc).saturating_add(n) * esz, "gemm c");
        CudaError::check(unsafe {
            ffi::gb_glmvis_gemm(
                a.ptr(),
                i(lda)?,
                w.ptr(),
                c.ptr(),
                i(ldc)?,
                i(m)?,
                i(k)?,
                i(n)?,
                c_is_f32 as i32,
                ws.ptr(),
                ws.bytes(),
                st.raw(),
            )
        })
    }
}
