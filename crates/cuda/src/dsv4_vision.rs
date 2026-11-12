//! The DeepSeek-V4.1-Flash vision tower (`csrc/dsv4_vision.cu`): 32 pre-norm blocks of hidden 1024 (16 heads of 64 with a
//! 2-D RoPE on q and k, an RMSNorm, and a SwiGLU MLP of 2816), a final RMSNorm, and the 3 x 3 aligner from 9216 to 5120.
//!
//! The reference is `/tmp/ds41ref/vision.py` (`ViT.forward`, `Block.forward`, `Aligner.forward`), which is 119 lines and
//! settles every ordering question this file has to answer:
//!
//! * [`patchify`] — `x.reshape(3, n_h, p, n_w, p).permute(1, 3, 0, 2, 4).reshape(P, 3, p, p)` then `flatten(1)`, i.e. a
//!   patch row is **channel-major**, then `i` within the patch, then `j`: `index = c * p * p + i * p + j`. That is the 588.
//! * [`rope_angles`] — `stack([hpos, wpos], -1).reshape(-1, 2, 1) * inv_freq` flattened, so a head's first
//!   [`ROPE_DIM`] channels turn by the row angle and the second half by the column angle.
//! * [`aligner_index`] — `F.unfold` over `[C, n_h, n_w]` emits its window `(C, kH, kW)` with `C` outermost, so the 9216
//!   input of `aligner.w1` is `c * 9 + kh * 3 + kw` — **not** the patch-major order a 2 x 2 merge would give.
//!
//! # What runs without a GPU
//!
//! Everything in this module except the `gpu` submodule is host code. `tests/dsv4_vision.rs` runs the whole tower on a toy
//! config in f32 and compares it with a direct f64 transcription of `vision.py`; the device tests print `not run: no device`
//! and return on a box with no nvcc, so this binary is green on CPU CI.

pub use gpu::*;

// ------------------------------------------------------------------------------------------------------------------ shapes

/// Side of one patch, in pixels (`vision_patch_size`).
pub const PATCH: usize = 14;
/// Channels of one patch.
pub const IN_CH: usize = 3;
/// Values of one patch: `3 * 14 * 14 = 588`. `PatchEmbed.proj` is `Linear(588, 1024)`.
pub const PATCH_K: usize = IN_CH * PATCH * PATCH;
/// Width of the tower and of every RMSNorm in it (`vision_dim`).
pub const HIDDEN: usize = 1024;
/// Attention heads (`vision_n_heads`).
pub const HEADS: usize = 16;
/// `HIDDEN / HEADS`.
pub const HEAD_DIM: usize = 64;
/// `HIDDEN / HEADS / 2`: the angle vector `get_vision_cos_sin` is asked for.
pub const ROPE_DIM: usize = HEAD_DIM / 2;
/// `vision_inter_dim`: each half of the `w1` output.
pub const INTER: usize = 2816;
/// Width of `mlp.w1`'s output (`2 * INTER`, gate then up).
pub const MLP: usize = 2 * INTER;
/// Blocks (`vision_n_layers`).
pub const LAYERS: usize = 32;
/// Side of the aligner's window (`vision_downsample_ratio`).
pub const MERGE: usize = 3;
/// Width of one aligner input row: `HIDDEN * MERGE * MERGE`.
pub const ALIGN_IN: usize = HIDDEN * MERGE * MERGE;
/// Width of the aligner's output: the language model's `dim`.
pub const OUT_HIDDEN: usize = 5120;
/// `RMSNorm(dim, eps=1e-6)` — the module default, and the pack's `rms_norm_eps` for the tower.
pub const RMS_EPS: f32 = 1e-6;
/// `vision_rope_theta`.
pub const ROPE_THETA: f32 = 10000.0;
/// The image-token budget preprocessing enforces ([`num_image_tokens`] per image, delimiters included) — 1,000 for this
/// deployment. The pack's `vision_config.max_image_tokens` (1,024) is the ViT grid's own ceiling; the smaller of the two
/// is what an image may cost, and only this one is a served constant rather than a shape.
pub const MAX_IMAGE_TOKENS: usize = 1000;
/// cuBLAS workspace, held for the life of the process and counted by the memory model.
pub const GEMM_WORKSPACE: usize = 32 << 20;

/// The per-image token span delimiters (`image_processor.py`: `IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)`).
/// Every one of these positions carries `image_token_id` in `input_ids`; only the type tells them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i8)]
pub enum TokenType {
    /// Not part of an image (`TEXT = -1`).
    Text = -1,
    ImageStart = 0,
    /// An aligner row.
    Image = 1,
    ImageNewLine = 2,
    ImageEnd = 3,
}

impl TokenType {
    pub fn from_i8(v: i8) -> Option<Self> {
        Some(match v {
            -1 => TokenType::Text,
            0 => TokenType::ImageStart,
            1 => TokenType::Image,
            2 => TokenType::ImageNewLine,
            3 => TokenType::ImageEnd,
            _ => return None,
        })
    }
}

// ------------------------------------------------------------------------------------------------------------------ host layout

/// The RoPE frequencies `get_vision_cos_sin` builds: `theta^(-2j / dim)` for `j in 0..dim/2`.
pub fn inv_freq(dim: usize, theta: f32) -> Vec<f32> {
    (0..dim / 2).map(|j| 1.0 / theta.powf((2 * j) as f32 / dim as f32)).collect()
}

/// The `[P, ROPE_DIM]` angle table for a `n_h x n_w` patch grid, in the reference's order: axis-major, so column
/// `axis * (ROPE_DIM / 2) + j` is `pos_axis * inv_freq[j]` and `ROPE_DIM / 2` frequencies per axis.
pub fn rope_angles(n_h: usize, n_w: usize, dim: usize, theta: f32) -> Vec<f32> {
    let f = inv_freq(dim, theta);
    let half = f.len();
    let mut out = vec![0f32; n_h * n_w * dim];
    for h in 0..n_h {
        for w in 0..n_w {
            let row = &mut out[(h * n_w + w) * dim..(h * n_w + w + 1) * dim];
            for j in 0..half {
                row[j] = h as f32 * f[j];
                row[half + j] = w as f32 * f[j];
            }
        }
    }
    out
}

/// Offset of `(c, i, j)` within one 588-value patch: `PatchEmbed.forward`'s `x.flatten(1)` on `[3, p, p]`.
#[inline]
pub fn flatten_index(c: usize, i: usize, j: usize) -> usize {
    c * PATCH * PATCH + i * PATCH + j
}

/// Offset of `(c, kh, kw)` within one 9216-value aligner row: `F.unfold`'s `(C, kH, kW)` window order.
#[inline]
pub fn aligner_index(c: usize, kh: usize, kw: usize) -> usize {
    c * MERGE * MERGE + kh * MERGE + kw
}

/// Turn an RGB image `[3, n_h * PATCH, n_w * PATCH]` (row-major, already normalised to `[-1, 1]`) into `[P, 588]`
/// patches in the reference's order: patch row-major over `(n_h, n_w)`, and within a patch `c`, then `i`, then `j`.
pub fn patchify(pix: &[f32], n_h: usize, n_w: usize, out: &mut [f32]) {
    let (h, w) = (n_h * PATCH, n_w * PATCH);
    assert_eq!(pix.len(), IN_CH * h * w, "pixel tensor is [3, {h}, {w}]");
    assert_eq!(out.len(), n_h * n_w * PATCH_K, "patch tensor is [{} , {PATCH_K}]", n_h * n_w);
    for ph in 0..n_h {
        for pw in 0..n_w {
            let dst = &mut out[(ph * n_w + pw) * PATCH_K..(ph * n_w + pw + 1) * PATCH_K];
            for c in 0..IN_CH {
                for i in 0..PATCH {
                    for j in 0..PATCH {
                        dst[flatten_index(c, i, j)] = pix[c * h * w + (ph * PATCH + i) * w + (pw * PATCH + j)];
                    }
                }
            }
        }
    }
}

/// `ceil(n / MERGE)`, the token grid an aligner produces from a patch grid (`image_processor.llm_grid`).
pub fn llm_grid(n: usize) -> usize {
    n.div_ceil(MERGE)
}

/// `image_processor.num_image_tokens`.
pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

/// `image_processor.image_token_types`: one `ImageStart`, then `n_llm_w` `Image` slots and one `ImageNewLine` per row of
/// the aligner grid, then one `ImageEnd`.
pub fn token_types(n_llm_h: usize, n_llm_w: usize) -> Vec<i8> {
    let mut t = Vec::with_capacity(num_image_tokens(n_llm_h, n_llm_w));
    t.push(TokenType::ImageStart as i8);
    for _ in 0..n_llm_h {
        t.extend(std::iter::repeat_n(TokenType::Image as i8, n_llm_w));
        t.push(TokenType::ImageNewLine as i8);
    }
    t.push(TokenType::ImageEnd as i8);
    t
}

/// The engram layer's mask: `None` when the prompt has no image at all (the fast path: no `where`, no `masked_fill`), and
/// otherwise one `bool` per token, `false` across every image span.
///
/// This is `model.py`'s `engram_mask = None if image_mask is None else ~image_mask` over `image_mask = token_types >= 0`.
/// A `false` position must be replaced by the engram layer's `DEAD` in the compressed token stream, so no n-gram crosses an
/// image and the image's positions take no engram contribution; `None` means the caller can skip both steps.
pub fn engram_mask(token_types: Option<&[i8]>) -> Option<Vec<bool>> {
    token_types.map(|t| t.iter().map(|&v| v == TokenType::Text as i8).collect())
}

/// The complement: `token_types >= 0`, the mask the text attention reads (`model.py`'s `image_mask`).
pub fn image_mask(token_types: &[i8]) -> Vec<bool> {
    token_types.iter().map(|&v| v >= 0).collect()
}

// ------------------------------------------------------------------------------------------------------------------ host numerics

/// `0.5 * (1 + erf(x / sqrt(2)))`, the `F.gelu` default (exact erf, not the tanh form).
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Abramowitz & Stegun 7.1.26: `|error| <= 1.5e-7`, well inside f32 for a gate that only has to match to 1e-5.
fn erf(x: f32) -> f32 {
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.0614054 * t - 1.4531521) * t) + 1.4214138) * t - 0.28449672) * t + 0.2548296) * t * (-x * x).exp();
    s * y
}

/// `silu(x) = x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `x * rsqrt(mean(x^2) + eps) * weight`, the reference's `RMSNorm` (the mean and the rsqrt are f32, the weight is applied
/// after).
pub fn rmsnorm_host(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len() % w.len(), 0, "rmsnorm rows");
    let d = w.len();
    let mut out = vec![0f32; x.len()];
    for (r, (src, dst)) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)).enumerate() {
        let _ = r;
        let ms = src.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let s = 1.0 / (ms + eps).sqrt();
        for i in 0..d {
            dst[i] = w[i] * (src[i] * s);
        }
    }
    out
}

/// `y[r, :] = x[r, :] @ w^T + b`, `w` row-major `[n, k]`, fp32 accumulate.
pub fn linear_host(x: &[f32], rows: usize, k: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let n = b.len();
    assert_eq!(w.len(), n * k, "linear weight is [{n}, {k}]");
    assert_eq!(x.len(), rows * k, "linear input is [{rows}, {k}]");
    let mut out = vec![0f32; rows * n];
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        for o in 0..n {
            let wr = &w[o * k..(o + 1) * k];
            let mut acc = 0f32;
            for i in 0..k {
                acc += xr[i] * wr[i];
            }
            out[r * n + o] = acc + b[o];
        }
    }
    out
}

/// `[x1 * cos - x2 * sin, x2 * cos + x1 * sin]` on the last axis, `cos`/`sin` per row of `ROPE_DIM` shared by every head.
fn apply_rope(x: &mut [f32], rows: usize, heads: usize, angles: &[f32]) {
    let dim = HEAD_DIM;
    let half = ROPE_DIM;
    for r in 0..rows {
        let ang = &angles[r * ROPE_DIM..(r + 1) * ROPE_DIM];
        for h in 0..heads {
            let base = (r * heads + h) * dim;
            for j in 0..half {
                let (c, s) = (ang[j].cos(), ang[j].sin());
                let x1 = x[base + j];
                let x2 = x[base + half + j];
                x[base + j] = x1 * c - x2 * s;
                x[base + half + j] = x2 * c + x1 * s;
            }
        }
    }
}

/// One block's weights, row-major `[out, in]`, as the host twin reads them.
pub struct BlockHostW<'a> {
    pub norm1: &'a [f32],
    pub norm2: &'a [f32],
    pub qkv_w: &'a [f32],
    pub qkv_b: &'a [f32],
    pub wo_w: &'a [f32],
    pub wo_b: &'a [f32],
    /// `[2 * INTER, HIDDEN]`, gate in the first half of the output and up in the second (`chunk(2, -1)`).
    pub w1: &'a [f32],
    /// `[HIDDEN, INTER]`.
    pub w2: &'a [f32],
}

/// `Block.forward` on the host: `x + attn(norm1(x))`, then `x + mlp(norm2(x))`. Full bidirectional attention over the
/// `p` patches of one image, fp32 throughout — the same arithmetic `csrc/dsv4_vision.cu` runs with fp16 operands.
pub fn block_host(x: &[f32], p: usize, n_w: usize, w: &BlockHostW, eps: f32) -> Vec<f32> {
    let (d, heads) = (HIDDEN, HEADS);
    let angles = rope_angles(p / n_w, n_w, ROPE_DIM, ROPE_THETA);

    // attention
    let h = rmsnorm_host(x, w.norm1, eps);
    let mut qkv = linear_host(&h, p, d, w.qkv_w, w.qkv_b);
    let (mut q, mut k, v): (Vec<f32>, Vec<f32>, Vec<f32>) = {
        let mut q = vec![0f32; p * d];
        let mut k = vec![0f32; p * d];
        let mut v = vec![0f32; p * d];
        for r in 0..p {
            q[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d..r * 3 * d + d]);
            k[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d + d..r * 3 * d + 2 * d]);
            v[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d + 2 * d..(r + 1) * 3 * d]);
        }
        (std::mem::take(&mut q), std::mem::take(&mut k), v)
    };
    qkv.clear();
    apply_rope(&mut q, p, heads, &angles);
    apply_rope(&mut k, p, heads, &angles);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut att = vec![0f32; p * d];
    for hd in 0..heads {
        for i in 0..p {
            let mut scores = vec![0f32; p];
            for j in 0..p {
                let mut acc = 0f32;
                for c in 0..HEAD_DIM {
                    acc += q[(i * heads + hd) * HEAD_DIM + c] * k[(j * heads + hd) * HEAD_DIM + c];
                }
                scores[j] = acc * scale;
            }
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in &mut scores {
                *s = (*s - m).exp();
                sum += *s;
            }
            for j in 0..p {
                let wgt = scores[j] / sum;
                for c in 0..HEAD_DIM {
                    att[(i * heads + hd) * HEAD_DIM + c] += wgt * v[(j * heads + hd) * HEAD_DIM + c];
                }
            }
        }
    }
    let o = linear_host(&att, p, d, w.wo_w, w.wo_b);
    let mut x1: Vec<f32> = x.iter().zip(&o).map(|(a, b)| a + b).collect();

    // mlp
    let h2 = rmsnorm_host(&x1, w.norm2, eps);
    let gu = linear_host(&h2, p, d, w.w1, &vec![0f32; MLP]);
    let mut act = vec![0f32; p * INTER];
    for r in 0..p {
        for i in 0..INTER {
            act[r * INTER + i] = silu(gu[r * MLP + i]) * gu[r * MLP + INTER + i];
        }
    }
    let down = linear_host(&act, p, INTER, w.w2, &vec![0f32; d]);
    for i in 0..x1.len() {
        x1[i] += down[i];
    }
    x1
}

/// `Aligner.forward` on the host: `F.pad`, `F.unfold(r, stride=r)`, `w2(gelu(w1(x)))`. `x` is the tower's `[P, HIDDEN]`
/// output; the result is `[ceil(n_h/r) * ceil(n_w/r), OUT_HIDDEN]`.
pub fn aligner_host(x: &[f32], n_h: usize, n_w: usize, w1: &[f32], b1: &[f32], w2: &[f32], b2: &[f32]) -> Vec<f32> {
    let r = MERGE;
    // pad on the right and the bottom (`F.pad(x, (0, -n_w % r, 0, -n_h % r))`), zeros
    let (ph, pw) = (n_h + (r - n_h % r) % r, n_w + (r - n_w % r) % r);
    let (oh, ow) = (ph / r, pw / r);
    let mut rows = vec![0f32; oh * ow * ALIGN_IN];
    for oy in 0..oh {
        for ox in 0..ow {
            let dst = &mut rows[(oy * ow + ox) * ALIGN_IN..(oy * ow + ox + 1) * ALIGN_IN];
            for c in 0..HIDDEN {
                for kh in 0..r {
                    for kw in 0..r {
                        let (y, xx) = (oy * r + kh, ox * r + kw);
                        if y < n_h && xx < n_w {
                            dst[aligner_index(c, kh, kw)] = x[(y * n_w + xx) * HIDDEN + c];
                        }
                    }
                }
            }
        }
    }
    let h = linear_host(&rows, oh * ow, ALIGN_IN, w1, b1);
    let g: Vec<f32> = h.iter().map(|&v| gelu(v)).collect();
    linear_host(&g, oh * ow, OUT_HIDDEN, w2, b2)
}

// ------------------------------------------------------------------------------------------------------------------ device

mod gpu {
    use super::*;
    use crate::{CudaError, DevBuf, Stream};
    use core::ffi::c_void;

    mod ffi {
        use core::ffi::c_void;
        crate::launchers! {
            pub fn gb_dsv4vis_bf16_to_f16(p: *mut c_void, n: usize, bad: *mut i32, s: *mut c_void) -> i32;
            pub fn gb_dsv4vis_gemm(epi: i32, a: *const c_void, b: *const c_void, bias: *const c_void, c: *mut c_void, m: i32, n: i32, k: i32, lda: i32, ldc: i32, s: *mut c_void) -> i32;
            pub fn gb_dsv4vis_rmsnorm(x: *const c_void, w: *const c_void, y: *mut c_void, rows: i32, dim: i32, eps: f32, s: *mut c_void) -> i32;
            pub fn gb_dsv4vis_bias_add_f32(x: *mut c_void, bias: *const c_void, n: i32, total: usize, s: *mut c_void) -> i32;
            pub fn gb_dsv4vis_bias_add_f16(x: *mut c_void, bias: *const c_void, n: i32, total: usize, s: *mut c_void) -> i32;
            pub fn gb_dsv4vis_rope(qkv: *mut c_void, ang: *const c_void, rows: i32, n_w: i32, heads: i32, s: *mut c_void) -> i32;
            pub fn gb_dsv4vis_attention(qkv: *const c_void, out: *mut c_void, p: i32, s: *mut c_void) -> i32;
            pub fn gb_dsv4vis_silu_mul(gu: *mut c_void, inter: i32, ld: i32, rows: i32, s: *mut c_void) -> i32;
            pub fn gb_dsv4vis_gelu_erf(x: *mut c_void, n: usize, s: *mut c_void) -> i32;
            pub fn gb_dsv4vis_aligner_gather(x: *const c_void, out: *mut c_void, n_h: i32, n_w: i32, dim: i32, s: *mut c_void) -> i32;
            pub fn gb_dsv4vis_splice(
                dst: *mut c_void, feats: *const c_void, start: i32, n_llm_h: i32, n_llm_w: i32, dim: i32,
                start_emb: *const c_void, end_emb: *const c_void, nl_emb: *const c_void, s: *mut c_void,
            ) -> i32;
            pub fn gb_dsv4vis_rope_table(ang: *mut c_void, n_h: i32, n_w: i32, dim: i32, theta: f32, s: *mut c_void) -> i32;
        }
    }

    fn i(v: usize) -> Result<i32, CudaError> {
        i32::try_from(v).map_err(|_| CudaError(1))
    }

    fn need(b: &DevBuf, off: usize, bytes: usize, what: &str) {
        assert!(off + bytes <= b.bytes(), "dsv4_vision {what}: buffer of {} bytes, need {bytes} at {off}", b.bytes());
    }

    /// Epilogue of [`gemm`].
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Epi {
        /// `C = fp16(acc + bias)`.
        BiasF16 = 0,
        /// `C = acc + bias`, stored fp32.
        StoreF32 = 1,
        /// `C += acc + bias` in fp32, in place.
        BiasAddF32 = 2,
        /// `C = fp16(gelu_erf(acc + bias))`.
        BiasGeluF16 = 3,
    }

    impl Epi {
        fn out_bytes(self) -> usize {
            match self {
                Epi::BiasF16 | Epi::BiasGeluF16 => 2,
                Epi::StoreF32 | Epi::BiasAddF32 => 4,
            }
        }
    }

    /// bf16 -> f16 in place; `bad` (one `i32`) counts values that are not finite in f16.
    pub fn bf16_to_f16(p: &mut DevBuf, n: usize, bad: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
        need(p, 0, n * 2, "bf16_to_f16 p");
        need(bad, 0, 4, "bf16_to_f16 bad");
        CudaError::check(unsafe { ffi::gb_dsv4vis_bf16_to_f16(p.ptr(), n, bad.ptr() as *mut i32, st.raw()) })
    }

    /// `C[m, n] = A[m, k] * B[n, k]^T` with the epilogue, fp16 operands, fp32 accumulation; `A` at row stride `lda`, `C`
    /// at row stride `ldc`.
    ///
    /// `lda` exists because the MLP keeps its activation in the first `INTER` columns of a `2 * INTER`-strided row
    /// (`silu_mul` writes in place); every other caller passes `lda == k`. `k` is arbitrary: the patch embedding's 588
    /// is not a multiple of 8, and padding it would mean rewriting the pack's `[1024, 588]` weight, so the kernel
    /// predicates its last K tile instead.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        epi: Epi,
        a: &DevBuf,
        b: &DevBuf,
        bias: Option<&DevBuf>,
        c: &mut DevBuf,
        c_off: usize,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldc: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        assert!(lda >= k, "dsv4_vision gemm: lda {lda} < k {k}");
        need(a, 0, ((m.max(1) - 1) * lda + k) * 2, "gemm a");
        need(b, 0, n * k * 2, "gemm b");
        if let Some(bi) = bias {
            need(bi, 0, n * 2, "gemm bias");
        }
        if m > 0 {
            need(c, c_off, ((m - 1) * ldc + n) * epi.out_bytes(), "gemm c");
        }
        let bp = bias.map_or(core::ptr::null(), |x| x.ptr() as *const c_void);
        let cp = if m > 0 { c.ptr_at(c_off) } else { core::ptr::null_mut() };
        CudaError::check(unsafe {
            ffi::gb_dsv4vis_gemm(epi as i32, a.ptr(), b.ptr(), bp, cp, i(m)?, i(n)?, i(k)?, i(lda)?, i(ldc)?, st.raw())
        })
    }

    /// `y = fp16(RMSNorm(x) * w)`: `x` and `w` fp32, `y` fp16 — the narrow to the GEMM operand dtype happens in the
    /// kernel, where the reference's `RMSNorm.forward` does `.to(dtype)`.
    pub fn rmsnorm(
        x: &DevBuf,
        w: &DevBuf,
        y: &mut DevBuf,
        rows: usize,
        dim: usize,
        eps: f32,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(x, 0, rows * dim * 4, "rmsnorm x");
        need(w, 0, dim * 4, "rmsnorm w");
        need(y, 0, rows * dim * 2, "rmsnorm y");
        CudaError::check(unsafe { ffi::gb_dsv4vis_rmsnorm(x.ptr(), w.ptr(), y.ptr(), i(rows)?, i(dim)?, eps, st.raw()) })
    }

    /// `x += bias` over `total` elements, fp32 stream and fp16 bias.
    pub fn bias_add_f32(x: &mut DevBuf, bias: &DevBuf, n: usize, total: usize, st: &Stream) -> Result<(), CudaError> {
        need(x, 0, total * 4, "bias_add_f32 x");
        need(bias, 0, n * 2, "bias_add_f32 bias");
        CudaError::check(unsafe { ffi::gb_dsv4vis_bias_add_f32(x.ptr(), bias.ptr(), i(n)?, total, st.raw()) })
    }

    /// `x += bias` over `total` elements, fp16 stream and fp16 bias.
    pub fn bias_add_f16(x: &mut DevBuf, bias: &DevBuf, n: usize, total: usize, st: &Stream) -> Result<(), CudaError> {
        need(x, 0, total * 2, "bias_add_f16 x");
        need(bias, 0, n * 2, "bias_add_f16 bias");
        CudaError::check(unsafe { ffi::gb_dsv4vis_bias_add_f16(x.ptr(), bias.ptr(), i(n)?, total, st.raw()) })
    }

    /// The `[P, ROPE_DIM]` angle table, on the device (`get_vision_cos_sin` without its cache).
    pub fn rope_table(ang: &mut DevBuf, n_h: usize, n_w: usize, dim: usize, theta: f32, st: &Stream) -> Result<(), CudaError> {
        need(ang, 0, n_h * n_w * dim * 4, "rope_table");
        CudaError::check(unsafe { ffi::gb_dsv4vis_rope_table(ang.ptr(), i(n_h)?, i(n_w)?, i(dim)?, theta, st.raw()) })
    }

    /// The 2-D RoPE in place on q and k of `qkv` `[P, 3 * HIDDEN]` (`q | k | v`, each `[P][heads][64]`).
    pub fn rope(qkv: &mut DevBuf, ang: &DevBuf, rows: usize, n_w: usize, heads: usize, st: &Stream) -> Result<(), CudaError> {
        need(qkv, 0, rows * 3 * heads * HEAD_DIM * 2, "rope qkv");
        need(ang, 0, rows * ROPE_DIM * 4, "rope angles");
        CudaError::check(unsafe { ffi::gb_dsv4vis_rope(qkv.ptr(), ang.ptr(), i(rows)?, i(n_w)?, i(heads)?, st.raw()) })
    }

    /// Bidirectional attention over `p` patches; `qkv` is `[P, 3 * HIDDEN]`, `out` `[P, HIDDEN]` fp16.
    pub fn attention(qkv: &DevBuf, out: &mut DevBuf, p: usize, st: &Stream) -> Result<(), CudaError> {
        need(qkv, 0, p * 3 * HIDDEN * 2, "attention qkv");
        need(out, 0, p * HIDDEN * 2, "attention out");
        CudaError::check(unsafe { ffi::gb_dsv4vis_attention(qkv.ptr(), out.ptr(), i(p)?, st.raw()) })
    }

    /// `gu[.., :INTER] = silu(gu[.., :INTER]) * gu[.., INTER:2*INTER]` in place, row stride `ld`.
    pub fn silu_mul(gu: &mut DevBuf, inter: usize, ld: usize, rows: usize, st: &Stream) -> Result<(), CudaError> {
        need(gu, 0, rows * ld * 2, "silu_mul");
        CudaError::check(unsafe { ffi::gb_dsv4vis_silu_mul(gu.ptr(), i(inter)?, i(ld)?, i(rows)?, st.raw()) })
    }

    /// `x = gelu_erf(x)` in place, fp16.
    pub fn gelu_erf(x: &mut DevBuf, n: usize, st: &Stream) -> Result<(), CudaError> {
        need(x, 0, n * 2, "gelu_erf");
        CudaError::check(unsafe { ffi::gb_dsv4vis_gelu_erf(x.ptr(), n, st.raw()) })
    }

    /// `[n_h * n_w, HIDDEN]` -> `[oh * ow, 9216]`, the `F.pad` + `F.unfold(3, stride=3)` gather with `aligner_index`'s
    /// `(C, kH, kW)` order; the padded border is zero.
    pub fn aligner_gather(
        x: &DevBuf,
        out: &mut DevBuf,
        n_h: usize,
        n_w: usize,
        dim: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let (oh, ow) = (llm_grid(n_h), llm_grid(n_w));
        need(x, 0, n_h * n_w * dim * 2, "aligner_gather x");
        need(out, 0, oh * ow * dim * MERGE * MERGE * 2, "aligner_gather out");
        CudaError::check(unsafe { ffi::gb_dsv4vis_aligner_gather(x.ptr(), out.ptr(), i(n_h)?, i(n_w)?, i(dim)?, st.raw()) })
    }

    /// Write one image's span of the text stream: the `ImageStart` / `ImageNewLine` / `ImageEnd` rows take their learned
    /// embeddings, the `Image` rows take the aligner's rows in reading order (see [`crate::dsv4_vision::splice_order`]).
    #[allow(clippy::too_many_arguments)]
    pub fn splice(
        dst: &mut DevBuf,
        feats: &DevBuf,
        start: usize,
        n_llm_h: usize,
        n_llm_w: usize,
        dim: usize,
        start_emb: &DevBuf,
        end_emb: &DevBuf,
        nl_emb: &DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let n = num_image_tokens(n_llm_h, n_llm_w);
        need(dst, start * dim * 2, n * dim * 2, "splice dst");
        need(feats, 0, n_llm_h * n_llm_w * dim * 2, "splice feats");
        for e in [start_emb, end_emb, nl_emb] {
            need(e, 0, dim * 2, "splice embedding");
        }
        CudaError::check(unsafe {
            ffi::gb_dsv4vis_splice(
                dst.ptr(),
                feats.ptr(),
                i(start)?,
                i(n_llm_h)?,
                i(n_llm_w)?,
                i(dim)?,
                start_emb.ptr(),
                end_emb.ptr(),
                nl_emb.ptr(),
                st.raw(),
            )
        })
    }
}

/// Where every position of one image's span gets its value, as `(span_offset, source)` — the host mirror of the splice
/// kernel, and the thing the model's `splice_into` and `tests/dsv4_vision.rs` both walk.
///
/// The order is the reference's: the span opens with `ImageStart`, then for each of the `n_llm_h` rows `n_llm_w` aligner rows
/// in reading order followed by one `ImageNewLine`, and closes with `ImageEnd`.
pub fn splice_order(n_llm_h: usize, n_llm_w: usize) -> Vec<(usize, SpliceSrc)> {
    let mut v = Vec::with_capacity(num_image_tokens(n_llm_h, n_llm_w));
    v.push((0, SpliceSrc::Start));
    let mut at = 1;
    let mut row = 0;
    for _ in 0..n_llm_h {
        for _ in 0..n_llm_w {
            v.push((at, SpliceSrc::Aligner(row)));
            at += 1;
            row += 1;
        }
        v.push((at, SpliceSrc::NewLine));
        at += 1;
    }
    v.push((at, SpliceSrc::End));
    v
}

/// What fills one position of an image span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpliceSrc {
    /// The learned `image_start` embedding.
    Start,
    /// The learned `image_end` embedding.
    End,
    /// The learned `image_newline` embedding (one per aligner row).
    NewLine,
    /// Row `n` of the aligner's output.
    Aligner(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_patch_is_three_channels_of_14_squared() {
        assert_eq!(PATCH_K, 588);
        assert_eq!(PATCH_K, 3 * 196);
        assert_eq!(ALIGN_IN, HIDDEN * 9);
        assert_eq!(MLP, 2 * INTER);
        assert_eq!(ROPE_DIM, 32);
    }

    /// `x.reshape(3, n_h, p, n_w, p).permute(1, 3, 0, 2, 4).reshape(-1, 3, p, p)` then `flatten(1)`, over a grid whose
    /// every value is its own coordinate, so a wrong axis shows up as a wrong number rather than as noise.
    #[test]
    fn patchify_is_channel_then_row_then_column() {
        let (n_h, n_w) = (2, 3);
        let (h, w) = (n_h * PATCH, n_w * PATCH);
        let pix: Vec<f32> = (0..IN_CH * h * w).map(|i| i as f32).collect();
        let mut out = vec![0f32; n_h * n_w * PATCH_K];
        patchify(&pix, n_h, n_w, &mut out);
        for ph in 0..n_h {
            for pw in 0..n_w {
                for c in 0..IN_CH {
                    for i in 0..PATCH {
                        for j in 0..PATCH {
                            let want = (c * h * w + (ph * PATCH + i) * w + (pw * PATCH + j)) as f32;
                            assert_eq!(out[(ph * n_w + pw) * PATCH_K + flatten_index(c, i, j)], want);
                        }
                    }
                }
            }
        }
        // the first patch is the top-left 14 x 14 of every channel, channel-major
        assert_eq!(out[0], 0.0);
        assert_eq!(out[flatten_index(0, 0, 1)], 1.0);
        assert_eq!(out[flatten_index(0, 1, 0)], w as f32);
        assert_eq!(out[flatten_index(1, 0, 0)], (h * w) as f32);
    }

    #[test]
    fn the_aligner_row_is_channel_major_over_the_3x3_window() {
        assert_eq!(aligner_index(0, 0, 0), 0);
        assert_eq!(aligner_index(0, 0, 2), 2);
        assert_eq!(aligner_index(0, 2, 0), 6);
        assert_eq!(aligner_index(1, 0, 0), 9);
        assert_eq!(aligner_index(HIDDEN - 1, 2, 2), ALIGN_IN - 1);
    }

    #[test]
    fn a_token_grid_is_the_patch_grid_over_three_rounded_up() {
        for (h, w, eh, ew) in [(3, 3, 1, 1), (14, 28, 5, 10), (1, 1, 1, 1), (4, 5, 2, 2)] {
            assert_eq!((llm_grid(h), llm_grid(w)), (eh, ew));
            assert_eq!(num_image_tokens(eh, ew), eh * (ew + 1) + 2);
        }
        // a 3 x 3 grid costs 6 tokens: START, three IMAGE, NEW_LINE, END
        assert_eq!(token_types(1, 3), vec![0, 1, 1, 1, 2, 3]);
    }

    #[test]
    fn a_text_only_prompt_has_no_mask_and_an_image_span_is_false_throughout() {
        // no image: the engram layer's fast path
        assert_eq!(engram_mask(None), None);
        // an image span: every position of it is false, the text around it true
        let mut types = vec![TokenType::Text as i8; 4];
        types.extend(token_types(2, 3));
        types.extend([TokenType::Text as i8; 2]);
        let m = engram_mask(Some(&types)).unwrap();
        assert_eq!(m.len(), types.len());
        assert!(m[..4].iter().all(|&b| b), "text before the image");
        assert!(m[4 + num_image_tokens(2, 3)..].iter().all(|&b| b), "text after the image");
        assert!(m[4..4 + num_image_tokens(2, 3)].iter().all(|&b| !b), "the span takes no part in an n-gram");
        // and the complement is what the text attention sees
        let im = image_mask(&types);
        assert_eq!(im, m.iter().map(|b| !b).collect::<Vec<_>>());
        assert_eq!(im.iter().filter(|&&b| b).count(), num_image_tokens(2, 3));
    }

    #[test]
    fn the_splice_fills_the_span_in_reading_order() {
        let order = splice_order(2, 3);
        assert_eq!(order.len(), num_image_tokens(2, 3));
        assert_eq!(order[0], (0, SpliceSrc::Start));
        assert_eq!(order[1], (1, SpliceSrc::Aligner(0)));
        assert_eq!(order[3], (3, SpliceSrc::Aligner(2)));
        assert_eq!(order[4], (4, SpliceSrc::NewLine));
        assert_eq!(order[5], (5, SpliceSrc::Aligner(3)));
        assert_eq!(order[7], (7, SpliceSrc::Aligner(5)));
        assert_eq!(order[8], (8, SpliceSrc::NewLine));
        assert_eq!(order[9], (9, SpliceSrc::End));
        // the aligner rows appear once each, in order
        let rows: Vec<usize> =
            order.iter().filter_map(|&(_, s)| if let SpliceSrc::Aligner(r) = s { Some(r) } else { None }).collect();
        assert_eq!(rows, (0..6).collect::<Vec<_>>());
    }
}
