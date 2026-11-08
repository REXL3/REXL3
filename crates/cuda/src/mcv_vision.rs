//! Kernels and the forward pass of the MiniCPM-V-4.7 vision path (`csrc/mcv_vision.cu`): the NaViT-style SigLIP tower (27 layers,
//! hidden 1152, 16 heads of 72, gelu_tanh MLP 4304, LayerNorm with bias and eps 1e-6, no RoPE, a learned 70 x 70 position table
//! read by nearest bucket), the 2 x 2 window merger after layer 6 (`16x` mode only) and the projector to the language model's 2048.
//!
//! The reference is the Hugging Face `modeling_minicpmv4_7.py` (transformers 5.18, Apache-2.0) run in bf16. [`forward`] starts from
//! the processor's packed pixel tensor (bf16 bits `[3, 14, 14 * patches]`, all crops of an image back to back) and the crops'
//! patch grids ([`Plan`]) and reproduces it: fp32 residual stream, fp16 GEMM operands and activations with fp32 accumulation,
//! fp32 LayerNorm and softmax statistics. Every crop is its own attention block, and every row's arithmetic is independent of the
//! other rows, so a crop's output does not depend on which other crops share the launch (`tests/mcv_vision.rs`).
//!
//! Every function launches on `stream` and returns without synchronising (except where a function says it reads back); buffer
//! sizes are checked (a short buffer panics rather than faulting on the device).

pub use gpu::*;

use std::fmt;

/// Hidden size of the tower, of its LayerNorms and of the features it hands the projector.
pub const HIDDEN: usize = 1152;
/// Attention heads of the tower (and of the merger's window attention).
pub const HEADS: usize = 16;
/// Head dimension (`HIDDEN / HEADS`).
pub const HEAD_DIM: usize = 72;
/// Width of the tower's MLP (gelu_tanh).
pub const MLP_HIDDEN: usize = 4304;
/// The MLP width padded to a multiple of 32 so that the fast GEMM path applies (zero weights; gelu_tanh(0) is 0).
pub const MLP_PADDED: usize = 4320;
/// Layers of the tower (all 27 are used).
pub const LAYERS: usize = 27;
/// The encoder layer after which the window merger runs in `16x` mode (`insert_layer_id`).
pub const MERGER_AFTER_LAYER: usize = 6;
/// Patch side in pixels.
pub const PATCH: usize = 14;
/// Values of one patch vector (3 x 14 x 14); the patch GEMM pads it to [`PATCH_PADDED`].
pub const PATCH_DIM: usize = 3 * PATCH * PATCH;
/// [`PATCH_DIM`] padded to a multiple of 32.
pub const PATCH_PADDED: usize = 608;
/// Side of the learned position table (70 x 70).
pub const POS_SIDE: usize = 70;
/// Width of a 2 x 2 window's concat (the merger's and the projector's LayerNorm).
pub const WINDOW_HIDDEN: usize = 4 * HIDDEN;
/// The window merger's hidden width (4 x 4304).
pub const MERGER_MLP: usize = 4 * MLP_HIDDEN;
/// The language model's hidden size (the projector's output).
pub const LLM_HIDDEN: usize = 2048;
/// Epsilon of every LayerNorm of the path.
pub const LN_EPS: f32 = 1e-6;

/// How many positions the tower's plan accepts at most (the workspace is sized by the caller within it): 37 crops of 1024 patches
/// (the processor's thumbnail and 36 slices) plus a margin.
pub const MAX_PATCHES: usize = 40 * 1024;

/// `downsample_mode`: `X16` runs the window merger after layer 6 (64 tokens a 448-pixel crop), `X4` skips it (256 tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    X16,
    X4,
}

impl Mode {
    /// The mode named by `config.downsample_mode` (`"16x"` or `"4x"`).
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "16x" => Some(Mode::X16),
            "4x" => Some(Mode::X4),
            _ => None,
        }
    }
    /// Patches per output token.
    pub fn patches_per_token(self) -> usize {
        match self {
            Mode::X16 => 16,
            Mode::X4 => 4,
        }
    }
    /// What the patch grid's sides must be a multiple of (4 for 16x: the window merger and the projector each merge 2 x 2; 2 for 4x).
    pub fn side_multiple(self) -> usize {
        match self {
            Mode::X16 => 4,
            Mode::X4 => 2,
        }
    }
}

/// Why a [`Plan`] was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    NoCrops,
    /// A crop whose patch grid is empty or whose sides are not multiples of the mode's merge factor.
    BadGrid {
        crop: usize,
        gh: usize,
        gw: usize,
        multiple: usize,
    },
    /// More patches than the caller allowed.
    TooManyPatches {
        patches: usize,
        max: usize,
    },
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanError::NoCrops => write!(f, "no crops"),
            PlanError::BadGrid { crop, gh, gw, multiple } => {
                write!(f, "crop {crop}: patch grid {gh} x {gw} is not a positive multiple of {multiple} on both sides")
            }
            PlanError::TooManyPatches { patches, max } => write!(f, "{patches} patches exceed the limit of {max}"),
        }
    }
}

impl std::error::Error for PlanError {}

/// The torch `bucketize` nearest-position rule of `get_vision_nearest_position_ids`, for one crop of `gh x gw` patches: the flat index
/// `bucket(i / gh) * 70 + bucket(j / gw)` into the 70 x 70 table, row-major over the crop's patches. float32 throughout, as torch:
/// the coordinates are `float32(i) / float32(n)` and the boundaries `float32(1/70 + k * (1/70))` for `k` in `arange(1/70, 1.0, 1/70)`
/// (computed in double, cast to float32), and `bucketize(.., right=True)` counts the boundaries that are `<= x`.
pub fn nearest_position_ids(gh: usize, gw: usize) -> Vec<i32> {
    let bounds = position_boundaries();
    let bucket = |n: usize| -> Vec<i32> {
        (0..n)
            .map(|i| {
                let x = i as f32 / n as f32;
                bounds.partition_point(|&b| b <= x) as i32
            })
            .collect()
    };
    let (bh, bw) = (bucket(gh), bucket(gw));
    let mut out = Vec::with_capacity(gh * gw);
    for &r in &bh {
        for &c in &bw {
            out.push(r * POS_SIDE as i32 + c);
        }
    }
    out
}

/// `torch.arange(1/70, 1.0, 1/70)` in float32: the size is `ceil((end - start) / step)` and element `k` is `start + k * step` in
/// double, rounded to float32.
fn position_boundaries() -> Vec<f32> {
    let step = 1.0f64 / POS_SIDE as f64;
    let start = step;
    let n = ((1.0f64 - start) / step).ceil() as usize;
    (0..n).map(|k| (start + k as f64 * step) as f32).collect()
}

/// The source patch (flat index inside the crop) of every token of the window merger's window-major order for a crop of `gh x gw`
/// patches: windows of 2 x 2 in raster order of windows, each window's four patches in raster order
/// (`get_vision_window_index` with `window_size=2, merge=1`). `gh` and `gw` must be even.
pub fn window_source(gh: usize, gw: usize) -> Vec<u32> {
    assert!(gh % 2 == 0 && gw % 2 == 0, "window grid {gh} x {gw} must be even");
    let mut out = Vec::with_capacity(gh * gw);
    for wr in 0..gh / 2 {
        for wc in 0..gw / 2 {
            for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                out.push(((2 * wr + dr) * gw + 2 * wc + dc) as u32);
            }
        }
    }
    out
}

/// The four source patches of every output token of a 2 x 2 merge of a `gh x gw` grid, in the projector's concat order
/// `(row in block, column in block)`: output token `(R, C)` takes patches `(2R, 2C), (2R, 2C + 1), (2R + 1, 2C), (2R + 1, 2C + 1)`.
pub fn merge_source(gh: usize, gw: usize) -> Vec<u32> {
    assert!(gh % 2 == 0 && gw % 2 == 0, "merge grid {gh} x {gw} must be even");
    let mut out = Vec::with_capacity(gh * gw);
    for r in 0..gh / 2 {
        for c in 0..gw / 2 {
            for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                out.push(((2 * r + dr) * gw + 2 * c + dc) as u32);
            }
        }
    }
    out
}

/// The shape of one forward: the crops' patch grids (thumbnail first, then the slices, as the processor's `target_sizes`), the
/// mode, and every index table the kernels read, built on the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub mode: Mode,
    /// `(gh, gw)` of every crop in patches.
    pub crops: Vec<(usize, usize)>,
    /// Patches in all crops (the rows of the tower's first layers).
    pub patches: usize,
    /// Rows the layers after the window merger run on (`patches / 4` in 16x mode, `patches` in 4x).
    pub late_rows: usize,
    /// Output tokens (`patches / 16` or `patches / 4`).
    pub tokens: usize,
    /// First row of every crop in the first layers' rows (cumulative `gh * gw`).
    pub early_row0: Vec<usize>,
    /// First row of every crop after the merger.
    pub late_row0: Vec<usize>,
    /// First output token of every crop.
    pub token0: Vec<usize>,
    /// Output grid `(rows, columns)` of every crop (`gh / 4` x `gw / 4` in 16x mode, halves in 4x).
    pub out_grid: Vec<(usize, usize)>,
    /// Position-table index of every patch (concatenated over the crops).
    pub pos_idx: Vec<i32>,
    /// Window-major source row of every merger row (16x only; empty in 4x), crop-major with the crops' row offsets applied.
    pub window_src: Vec<i32>,
    /// The four source rows (of the late rows) of every output token, crop-major.
    pub merge_src: Vec<i32>,
}

impl Plan {
    /// `max_patches` bounds the total (the caller's workspace); `usize::MAX` for none.
    pub fn new(crops: &[(usize, usize)], mode: Mode, max_patches: usize) -> Result<Plan, PlanError> {
        if crops.is_empty() {
            return Err(PlanError::NoCrops);
        }
        let m = mode.side_multiple();
        let mut patches = 0usize;
        for (i, &(gh, gw)) in crops.iter().enumerate() {
            if gh == 0 || gw == 0 || gh % m != 0 || gw % m != 0 {
                return Err(PlanError::BadGrid { crop: i, gh, gw, multiple: m });
            }
            patches = patches.saturating_add(gh.saturating_mul(gw));
        }
        if patches > max_patches {
            return Err(PlanError::TooManyPatches { patches, max: max_patches });
        }
        let merged = mode == Mode::X16;
        let mut early_row0 = Vec::with_capacity(crops.len());
        let mut late_row0 = Vec::with_capacity(crops.len());
        let mut token0 = Vec::with_capacity(crops.len());
        let mut out_grid = Vec::with_capacity(crops.len());
        let (mut pos_idx, mut window_src, mut merge_src) = (Vec::with_capacity(patches), Vec::new(), Vec::new());
        let (mut e, mut l, mut t) = (0usize, 0usize, 0usize);
        for &(gh, gw) in crops {
            early_row0.push(e);
            late_row0.push(l);
            token0.push(t);
            pos_idx.extend(nearest_position_ids(gh, gw));
            let (lh, lw) = if merged { (gh / 2, gw / 2) } else { (gh, gw) };
            if merged {
                window_src.extend(window_source(gh, gw).into_iter().map(|s| (e + s as usize) as i32));
            }
            merge_src.extend(merge_source(lh, lw).into_iter().map(|s| (l + s as usize) as i32));
            out_grid.push((lh / 2, lw / 2));
            e += gh * gw;
            l += lh * lw;
            t += lh / 2 * (lw / 2);
        }
        Ok(Plan {
            mode,
            crops: crops.to_vec(),
            patches,
            late_rows: l,
            tokens: t,
            early_row0,
            late_row0,
            token0,
            out_grid,
            pos_idx,
            window_src,
            merge_src,
        })
    }

    /// `(first row, rows)` of every crop among the first layers' rows.
    pub fn early_crops(&self) -> Vec<[i32; 2]> {
        self.crops.iter().zip(&self.early_row0).map(|(&(h, w), &r)| [r as i32, (h * w) as i32]).collect()
    }

    /// `(first row, rows)` of every crop among the rows after the merger.
    pub fn late_crops(&self) -> Vec<[i32; 2]> {
        let q = if self.mode == Mode::X16 { 4 } else { 1 };
        self.crops.iter().zip(&self.late_row0).map(|(&(h, w), &r)| [r as i32, (h * w / q) as i32]).collect()
    }

    /// Largest crop's rows among the first layers.
    pub fn max_early_rows(&self) -> usize {
        self.crops.iter().map(|&(h, w)| h * w).max().unwrap_or(0)
    }

    /// Largest crop's rows after the merger.
    pub fn max_late_rows(&self) -> usize {
        let q = if self.mode == Mode::X16 { 4 } else { 1 };
        self.max_early_rows() / q
    }

    /// Multiply-accumulate FLOPs (2 per MAC) of GEMMs and attention for this plan: the numerator of the achieved-TFLOP/s figures.
    pub fn flops(&self) -> f64 {
        let dense = |rows: f64, k: f64, n: f64| 2.0 * rows * k * n;
        let (p, q) = (self.patches as f64, self.late_rows as f64);
        let attn = |rows: &[f64]| rows.iter().map(|r| 4.0 * r * r * HIDDEN as f64).sum::<f64>();
        let layer = |rows: f64| {
            dense(rows, HIDDEN as f64, 3.0 * HIDDEN as f64)
                + dense(rows, HIDDEN as f64, HIDDEN as f64)
                + dense(rows, HIDDEN as f64, MLP_HIDDEN as f64)
                + dense(rows, MLP_HIDDEN as f64, HIDDEN as f64)
        };
        let early_n = if self.mode == Mode::X16 { MERGER_AFTER_LAYER + 1 } else { LAYERS };
        let early: Vec<f64> = self.crops.iter().map(|&(h, w)| (h * w) as f64).collect();
        let mut f = dense(p, PATCH_DIM as f64, HIDDEN as f64) + early_n as f64 * (layer(p) + attn(&early));
        if self.mode == Mode::X16 {
            let late: Vec<f64> = early.iter().map(|r| r / 4.0).collect();
            // merger: qkv + out projection on all rows, then Linear 4608 -> 17216 -> 1152 on the windows
            f += dense(p, HIDDEN as f64, 4.0 * HIDDEN as f64);
            f += dense(q, WINDOW_HIDDEN as f64, MERGER_MLP as f64) + dense(q, MERGER_MLP as f64, HIDDEN as f64);
            f += (LAYERS - early_n) as f64 * (layer(q) + attn(&late));
        }
        // projector on the tokens
        let t = self.tokens as f64;
        f + dense(t, WINDOW_HIDDEN as f64, WINDOW_HIDDEN as f64) + dense(t, WINDOW_HIDDEN as f64, LLM_HIDDEN as f64)
    }
}

mod gpu {
    use super::*;
    use crate::{CudaError, DevBuf, Stream};
    use core::ffi::c_void;
    use std::error::Error;
    use std::path::Path;

    mod ffi {
        use core::ffi::c_void;
        crate::launchers! {
            pub fn gb_mcv_layernorm(x: *const c_void, w: *const c_void, b: *const c_void, y: *mut c_void, rows: i32, dim: i32, eps: f32, s: *mut c_void) -> i32;
            pub fn gb_mcv_layernorm_f16(x: *const c_void, w: *const c_void, b: *const c_void, y: *mut c_void, rows: i32, dim: i32, eps: f32, s: *mut c_void) -> i32;
            pub fn gb_mcv_patchify(pix: *const c_void, out: *mut c_void, patches: i32, s: *mut c_void) -> i32;
            pub fn gb_mcv_pos_add(h: *mut c_void, pos: *const c_void, idx: *const c_void, rows: i32, s: *mut c_void) -> i32;
            pub fn gb_mcv_window_gather(h: *const c_void, xn: *const c_void, src: *const c_void, hw: *mut c_void, xw: *mut c_void, rows: i32, s: *mut c_void) -> i32;
            pub fn gb_mcv_mean4(hw: *const c_void, out: *mut c_void, nwin: i32, s: *mut c_void) -> i32;
            pub fn gb_mcv_merge4(x: *const c_void, src: *const c_void, out: *mut c_void, n: i32, s: *mut c_void) -> i32;
            pub fn gb_mcv_window_attention(qkv: *const c_void, out: *mut c_void, nwin: i32, s: *mut c_void) -> i32;
            pub fn gb_mcv_gemm(epi: i32, a: *const c_void, b: *const c_void, bias: *const c_void, c: *mut c_void, m: i32, n: i32, k: i32, ldc: i32, s: *mut c_void) -> i32;
            pub fn gb_mcv_attention(qkv: *const c_void, out: *mut c_void, crops: *const c_void, ncrops: i32, max_rows: i32, s: *mut c_void) -> i32;
            pub fn gb_mcv_absmax(x: *const c_void, is_f32: i32, n: usize, out: *mut c_void, s: *mut c_void) -> i32;
            pub fn gb_vis_f32_to_bf16(x: *const c_void, y: *mut c_void, n: usize, s: *mut c_void) -> i32;
        }
    }

    fn i(v: usize) -> Result<i32, CudaError> {
        i32::try_from(v).map_err(|_| CudaError(1))
    }

    fn need(b: &DevBuf, off: usize, bytes: usize, what: &str) {
        assert!(off + bytes <= b.bytes(), "mcv_vision {what}: buffer of {} bytes, need {} at {off}", b.bytes(), bytes);
    }

    /// Epilogue of [`gemm`]: what happens to each output element.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Epi {
        /// `C = fp16(acc + bias)`.
        BiasF16 = 0,
        /// `C = fp16(gelu_erf(acc + bias))` (the projector).
        BiasGeluF16 = 1,
        /// `C += acc + bias` in fp32, in place (the residual adds).
        BiasAddF32 = 2,
        /// `C = acc + bias` in fp32.
        StoreF32 = 3,
        /// `C = fp16(gelu_tanh(acc + bias))` (the tower MLP and the window merger).
        BiasGeluTanhF16 = 4,
    }

    impl Epi {
        fn out_bytes(self) -> usize {
            match self {
                Epi::BiasF16 | Epi::BiasGeluF16 | Epi::BiasGeluTanhF16 => 2,
                Epi::BiasAddF32 | Epi::StoreF32 => 4,
            }
        }
    }

    /// `C[m, n] = A[m, k] * B[n, k]^T` plus the epilogue, fp16 `A` and `B`, fp32 accumulation. `C` is written at byte offset `c_off`
    /// with row stride `ldc` elements; `bias` (if any) has `n` fp32 values. Fast path for `n % 8 == 0` and `k % 32 == 0`.
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
        ldc: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(a, 0, m * k * 2, "gemm a");
        need(b, 0, n * k * 2, "gemm b");
        if let Some(bi) = bias {
            need(bi, 0, n * 4, "gemm bias");
        }
        if m > 0 {
            need(c, c_off, ((m - 1) * ldc + n) * epi.out_bytes(), "gemm c");
        }
        let bp = bias.map_or(core::ptr::null(), |x| x.ptr() as *const c_void);
        let cp = if m > 0 { c.ptr_at(c_off) } else { core::ptr::null_mut() };
        CudaError::check(unsafe {
            ffi::gb_mcv_gemm(epi as i32, a.ptr(), b.ptr(), bp, cp, i(m)?, i(n)?, i(k)?, i(ldc)?, st.raw())
        })
    }

    /// `y = LayerNorm(x) * w + b` over rows of `dim`, eps [`LN_EPS`] by default: fp32 `x` `[rows, dim]`, fp32 `w`/`b` `[dim]`, fp16 `y`.
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
        need(x, 0, rows * dim * 4, "layernorm x");
        need(w, 0, dim * 4, "layernorm w");
        need(b, 0, dim * 4, "layernorm b");
        need(y, 0, rows * dim * 2, "layernorm y");
        CudaError::check(unsafe { ffi::gb_mcv_layernorm(x.ptr(), w.ptr(), b.ptr(), y.ptr(), i(rows)?, i(dim)?, eps, st.raw()) })
    }

    /// [`layernorm`] over fp16 `x` (the projector's `pre_norm`).
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm_f16(
        x: &DevBuf,
        w: &DevBuf,
        b: &DevBuf,
        y: &mut DevBuf,
        rows: usize,
        dim: usize,
        eps: f32,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(x, 0, rows * dim * 2, "layernorm_f16 x");
        need(w, 0, dim * 4, "layernorm_f16 w");
        need(b, 0, dim * 4, "layernorm_f16 b");
        need(y, 0, rows * dim * 2, "layernorm_f16 y");
        CudaError::check(unsafe {
            ffi::gb_mcv_layernorm_f16(x.ptr(), w.ptr(), b.ptr(), y.ptr(), i(rows)?, i(dim)?, eps, st.raw())
        })
    }

    /// Patch matrix fp16 `[patches, 608]` from the packed bf16 pixels `[3, 14, 14 * patches]`.
    pub fn patchify(pix: &DevBuf, out: &mut DevBuf, patches: usize, st: &Stream) -> Result<(), CudaError> {
        need(pix, 0, 3 * PATCH * PATCH * patches * 2, "patchify pixels");
        need(out, 0, patches * PATCH_PADDED * 2, "patchify out");
        CudaError::check(unsafe { ffi::gb_mcv_patchify(pix.ptr(), out.ptr(), i(patches)?, st.raw()) })
    }

    /// Full bidirectional attention inside every crop. `qkv` fp16 `[rows, 3 * 1152]`; `out` fp16 `[rows, 1152]`; `crops` i32
    /// `[ncrops, 2]` = (first row, rows). Softmax in fp32.
    pub fn attention(
        qkv: &DevBuf,
        out: &mut DevBuf,
        crops: &DevBuf,
        ncrops: usize,
        max_rows: usize,
        rows: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(qkv, 0, rows * 3 * HIDDEN * 2, "attention qkv");
        need(out, 0, rows * HIDDEN * 2, "attention out");
        need(crops, 0, ncrops * 8, "attention crops");
        CudaError::check(unsafe { ffi::gb_mcv_attention(qkv.ptr(), out.ptr(), crops.ptr(), i(ncrops)?, i(max_rows)?, st.raw()) })
    }

    /// One block of layer weights on the device. Matrices are fp16 in the checkpoint's `[out, in]` layout (q, k, v stacked; the MLP
    /// padded to 4320), vectors fp32.
    pub struct Layer {
        ln1_w: DevBuf,
        ln1_b: DevBuf,
        qkv_w: DevBuf,
        qkv_b: DevBuf,
        out_w: DevBuf,
        out_b: DevBuf,
        ln2_w: DevBuf,
        ln2_b: DevBuf,
        fc1_w: DevBuf,
        fc1_b: DevBuf,
        fc2_w: DevBuf,
        fc2_b: DevBuf,
    }

    /// `vit_merger`.
    pub struct Merger {
        ln1_w: DevBuf,
        ln1_b: DevBuf,
        qkv_w: DevBuf,
        qkv_b: DevBuf,
        out_w: DevBuf,
        out_b: DevBuf,
        pre_w: DevBuf,
        pre_b: DevBuf,
        lin1_w: DevBuf,
        lin1_b: DevBuf,
        lin2_w: DevBuf,
        lin2_b: DevBuf,
    }

    /// `merger.mlp.0`.
    pub struct Projector {
        pre_w: DevBuf,
        pre_b: DevBuf,
        lin1_w: DevBuf,
        lin1_b: DevBuf,
        lin2_w: DevBuf,
        lin2_b: DevBuf,
    }

    /// What the bf16 -> fp16 conversion of the weights did (the overflow check of the plan, risk "bf16 tower to fp16 range").
    #[derive(Clone, Debug, Default, PartialEq)]
    pub struct ConvertReport {
        pub tensors: usize,
        pub elements: u64,
        /// Elements whose magnitude exceeds fp16's 65504 (they would be infinite).
        pub overflow: u64,
        /// Non-zero bf16 elements that became fp16 zero (below half of the smallest subnormal).
        pub flushed_to_zero: u64,
        /// Elements that became fp16 subnormals (less than 6.1e-5 in magnitude: reduced precision).
        pub subnormal: u64,
        /// Largest magnitude seen.
        pub max_abs: f32,
    }

    impl ConvertReport {
        fn absorb(&mut self, bf16: &[u16]) {
            self.tensors += 1;
            self.elements += bf16.len() as u64;
            for &b in bf16 {
                let v = crate::num::bf16_to_f32(b);
                let a = v.abs();
                if a > self.max_abs {
                    self.max_abs = a;
                }
                if a > 65504.0 {
                    self.overflow += 1;
                } else if a != 0.0 && a < 5.960_464_5e-8 * 0.5 {
                    self.flushed_to_zero += 1;
                } else if a != 0.0 && a < 6.103_515_6e-5 {
                    self.subnormal += 1;
                }
            }
        }
    }

    fn f16_of(b: u16) -> u16 {
        gb_exl3::f16::f64_to_f16(crate::num::bf16_to_f32(b) as f64)
    }

    /// The vision tensors of a MiniCPM-V-4.7 checkpoint (or an EXL3 pack: the tower is copied through unchanged) on the device.
    pub struct Weights {
        patch_w: DevBuf,
        patch_b: DevBuf,
        pos: DevBuf,
        layers: Vec<Layer>,
        post_w: DevBuf,
        post_b: DevBuf,
        merger: Merger,
        projector: Projector,
        report: ConvertReport,
    }

    const VT: &str = "model.vision_tower.";
    const MG: &str = "model.merger.mlp.0.";

    /// Reads and converts the checkpoint's tensors (one pass, counting what the fp16 conversion did).
    struct Loader<'a> {
        ck: &'a gb_exl3::checkpoint::Checkpoint,
        dir: &'a Path,
        report: ConvertReport,
    }

    impl Loader<'_> {
        /// The raw bf16 bits of `name`, checked for dtype and shape.
        fn raw(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>, Box<dyn Error>> {
            let e = self.ck.plain.get(name).ok_or_else(|| format!("{name}: not a plain tensor of {}", self.dir.display()))?;
            if e.dtype != "BF16" || e.shape != shape {
                return Err(format!("{name}: {} {:?}, want BF16 {shape:?}", e.dtype, e.shape).into());
            }
            Ok(self.ck.read_u16(e)?)
        }

        /// A matrix to fp16, zero padded to `rows_to` x `cols_to` (the checkpoint's `[out, in]` layout).
        fn mat16(&mut self, name: &str, shape: &[usize], rows_to: usize, cols_to: usize) -> Result<DevBuf, Box<dyn Error>> {
            let raw = self.raw(name, shape)?;
            self.report.absorb(&raw);
            let (r, c) = (shape[0], shape[1]);
            let mut out = vec![0u16; rows_to * cols_to];
            for row in 0..r {
                for col in 0..c {
                    out[row * cols_to + col] = f16_of(raw[row * c + col]);
                }
            }
            Ok(DevBuf::from_slice(&out)?)
        }

        /// A vector widened to fp32 (exact), zero padded to `to` entries.
        fn vec32(&self, name: &str, n: usize, to: usize) -> Result<DevBuf, Box<dyn Error>> {
            let raw = self.raw(name, &[n])?;
            let mut v = vec![0f32; to];
            for (d, &s) in v.iter_mut().zip(&raw) {
                *d = crate::num::bf16_to_f32(s);
            }
            Ok(DevBuf::from_slice(&v)?)
        }

        /// q, k, v projections stacked into one `[3456, 1152]` fp16 matrix and one `[3456]` fp32 bias.
        fn qkv(&mut self, prefix: &str) -> Result<(DevBuf, DevBuf), Box<dyn Error>> {
            let mut w = vec![0u16; 3 * HIDDEN * HIDDEN];
            let mut b = vec![0f32; 3 * HIDDEN];
            for (k, nm) in ["q_proj", "k_proj", "v_proj"].iter().enumerate() {
                let raw = self.raw(&format!("{prefix}self_attn.{nm}.weight"), &[HIDDEN, HIDDEN])?;
                self.report.absorb(&raw);
                for (d, &s) in w[k * HIDDEN * HIDDEN..(k + 1) * HIDDEN * HIDDEN].iter_mut().zip(&raw) {
                    *d = f16_of(s);
                }
                let braw = self.raw(&format!("{prefix}self_attn.{nm}.bias"), &[HIDDEN])?;
                for (d, &s) in b[k * HIDDEN..(k + 1) * HIDDEN].iter_mut().zip(&braw) {
                    *d = crate::num::bf16_to_f32(s);
                }
            }
            Ok((DevBuf::from_slice(&w)?, DevBuf::from_slice(&b)?))
        }
    }

    impl Weights {
        /// Reads the 459 tower, merger and projector tensors from the checkpoint folder's safetensors (through its index when it has
        /// one); every tensor is checked for dtype (BF16) and shape.
        pub fn load(dir: &Path) -> Result<Weights, Box<dyn Error>> {
            let ck = gb_exl3::checkpoint::scan(dir, false)?;
            let mut ld = Loader { ck: &ck, dir, report: ConvertReport::default() };
            let patch_w = {
                let raw = ld.raw(&format!("{VT}embeddings.patch_embedding.weight"), &[HIDDEN, 3, PATCH, PATCH])?;
                ld.report.absorb(&raw);
                let mut out = vec![0u16; HIDDEN * PATCH_PADDED];
                for row in 0..HIDDEN {
                    for col in 0..PATCH_DIM {
                        out[row * PATCH_PADDED + col] = f16_of(raw[row * PATCH_DIM + col]);
                    }
                }
                DevBuf::from_slice(&out)?
            };
            let patch_b = ld.vec32(&format!("{VT}embeddings.patch_embedding.bias"), HIDDEN, HIDDEN)?;
            let pos = {
                let raw = ld.raw(&format!("{VT}embeddings.position_embedding.weight"), &[POS_SIDE * POS_SIDE, HIDDEN])?;
                DevBuf::from_slice(&raw.iter().map(|&b| crate::num::bf16_to_f32(b)).collect::<Vec<f32>>())?
            };
            let mut layers = Vec::with_capacity(LAYERS);
            for l in 0..LAYERS {
                let p = format!("{VT}encoder.layers.{l}.");
                let (qkv_w, qkv_b) = ld.qkv(&p)?;
                layers.push(Layer {
                    ln1_w: ld.vec32(&format!("{p}layer_norm1.weight"), HIDDEN, HIDDEN)?,
                    ln1_b: ld.vec32(&format!("{p}layer_norm1.bias"), HIDDEN, HIDDEN)?,
                    qkv_w,
                    qkv_b,
                    out_w: ld.mat16(&format!("{p}self_attn.out_proj.weight"), &[HIDDEN, HIDDEN], HIDDEN, HIDDEN)?,
                    out_b: ld.vec32(&format!("{p}self_attn.out_proj.bias"), HIDDEN, HIDDEN)?,
                    ln2_w: ld.vec32(&format!("{p}layer_norm2.weight"), HIDDEN, HIDDEN)?,
                    ln2_b: ld.vec32(&format!("{p}layer_norm2.bias"), HIDDEN, HIDDEN)?,
                    fc1_w: ld.mat16(&format!("{p}mlp.fc1.weight"), &[MLP_HIDDEN, HIDDEN], MLP_PADDED, HIDDEN)?,
                    fc1_b: ld.vec32(&format!("{p}mlp.fc1.bias"), MLP_HIDDEN, MLP_PADDED)?,
                    fc2_w: ld.mat16(&format!("{p}mlp.fc2.weight"), &[HIDDEN, MLP_HIDDEN], HIDDEN, MLP_PADDED)?,
                    fc2_b: ld.vec32(&format!("{p}mlp.fc2.bias"), HIDDEN, HIDDEN)?,
                });
            }
            let post_w = ld.vec32(&format!("{VT}post_layernorm.weight"), HIDDEN, HIDDEN)?;
            let post_b = ld.vec32(&format!("{VT}post_layernorm.bias"), HIDDEN, HIDDEN)?;
            let m = format!("{VT}vit_merger.");
            let (mqkv_w, mqkv_b) = ld.qkv(&m)?;
            let merger = Merger {
                ln1_w: ld.vec32(&format!("{m}layer_norm1.weight"), HIDDEN, HIDDEN)?,
                ln1_b: ld.vec32(&format!("{m}layer_norm1.bias"), HIDDEN, HIDDEN)?,
                qkv_w: mqkv_w,
                qkv_b: mqkv_b,
                out_w: ld.mat16(&format!("{m}self_attn.out_proj.weight"), &[HIDDEN, HIDDEN], HIDDEN, HIDDEN)?,
                out_b: ld.vec32(&format!("{m}self_attn.out_proj.bias"), HIDDEN, HIDDEN)?,
                pre_w: ld.vec32(&format!("{m}pre_norm.weight"), WINDOW_HIDDEN, WINDOW_HIDDEN)?,
                pre_b: ld.vec32(&format!("{m}pre_norm.bias"), WINDOW_HIDDEN, WINDOW_HIDDEN)?,
                lin1_w: ld.mat16(&format!("{m}linear_1.weight"), &[MERGER_MLP, WINDOW_HIDDEN], MERGER_MLP, WINDOW_HIDDEN)?,
                lin1_b: ld.vec32(&format!("{m}linear_1.bias"), MERGER_MLP, MERGER_MLP)?,
                lin2_w: ld.mat16(&format!("{m}linear_2.weight"), &[HIDDEN, MERGER_MLP], HIDDEN, MERGER_MLP)?,
                lin2_b: ld.vec32(&format!("{m}linear_2.bias"), HIDDEN, HIDDEN)?,
            };
            let projector = Projector {
                pre_w: ld.vec32(&format!("{MG}pre_norm.weight"), WINDOW_HIDDEN, WINDOW_HIDDEN)?,
                pre_b: ld.vec32(&format!("{MG}pre_norm.bias"), WINDOW_HIDDEN, WINDOW_HIDDEN)?,
                lin1_w: ld.mat16(
                    &format!("{MG}linear_1.weight"),
                    &[WINDOW_HIDDEN, WINDOW_HIDDEN],
                    WINDOW_HIDDEN,
                    WINDOW_HIDDEN,
                )?,
                lin1_b: ld.vec32(&format!("{MG}linear_1.bias"), WINDOW_HIDDEN, WINDOW_HIDDEN)?,
                lin2_w: ld.mat16(&format!("{MG}linear_2.weight"), &[LLM_HIDDEN, WINDOW_HIDDEN], LLM_HIDDEN, WINDOW_HIDDEN)?,
                lin2_b: ld.vec32(&format!("{MG}linear_2.bias"), LLM_HIDDEN, LLM_HIDDEN)?,
            };
            let report = ld.report;
            Ok(Weights { patch_w, patch_b, pos, layers, post_w, post_b, merger, projector, report })
        }

        /// The conversion report: elements converted, overflow, underflow and subnormal counts, the largest magnitude.
        pub fn report(&self) -> &ConvertReport {
            &self.report
        }

        /// Device bytes of every tensor (matrices fp16, vectors and the position table fp32).
        pub fn bytes(&self) -> u64 {
            let mut n = [&self.patch_w, &self.patch_b, &self.pos, &self.post_w, &self.post_b]
                .iter()
                .map(|b| b.bytes() as u64)
                .sum::<u64>();
            for l in &self.layers {
                n += [
                    &l.ln1_w, &l.ln1_b, &l.qkv_w, &l.qkv_b, &l.out_w, &l.out_b, &l.ln2_w, &l.ln2_b, &l.fc1_w, &l.fc1_b, &l.fc2_w,
                    &l.fc2_b,
                ]
                .iter()
                .map(|b| b.bytes() as u64)
                .sum::<u64>();
            }
            let m = &self.merger;
            n += [
                &m.ln1_w, &m.ln1_b, &m.qkv_w, &m.qkv_b, &m.out_w, &m.out_b, &m.pre_w, &m.pre_b, &m.lin1_w, &m.lin1_b, &m.lin2_w,
                &m.lin2_b,
            ]
            .iter()
            .map(|b| b.bytes() as u64)
            .sum::<u64>();
            let p = &self.projector;
            n + [&p.pre_w, &p.pre_b, &p.lin1_w, &p.lin1_b, &p.lin2_w, &p.lin2_b].iter().map(|b| b.bytes() as u64).sum::<u64>()
        }
    }

    /// Activation buffers and index tables, sized for up to `max_patches` patches. Reuse one across images.
    pub struct Workspace {
        max_patches: usize,
        patches: DevBuf,
        /// fp32 `[patches, 1152]`: the residual stream.
        h: DevBuf,
        xn: DevBuf,
        qkv: DevBuf,
        att: DevBuf,
        mid: DevBuf,
        hw: DevBuf,
        xw: DevBuf,
        /// Output tokens, fp32 `[tokens, 2048]`.
        out: DevBuf,
        pos_idx: DevBuf,
        window_src: DevBuf,
        merge_src: DevBuf,
        crops_early: DevBuf,
        crops_late: DevBuf,
        absmax: DevBuf,
    }

    /// Most crops one forward takes (the crop tables are this long): the thumbnail and 36 slices, with room.
    pub const MAX_CROPS: usize = 64;

    impl Workspace {
        /// Buffers for plans of up to `max_patches` patches (a multiple of 16, at most [`MAX_PATCHES`]).
        pub fn new(max_patches: usize) -> Result<Workspace, CudaError> {
            assert!(max_patches > 0 && max_patches % 16 == 0 && max_patches <= MAX_PATCHES, "max_patches {max_patches}");
            let p = max_patches;
            Ok(Workspace {
                max_patches,
                patches: DevBuf::zeroed(p * PATCH_PADDED * 2)?,
                h: DevBuf::zeroed(p * HIDDEN * 4)?,
                xn: DevBuf::zeroed(p * HIDDEN * 2)?,
                qkv: DevBuf::zeroed(p * 3 * HIDDEN * 2)?,
                att: DevBuf::zeroed(p * HIDDEN * 2)?,
                mid: DevBuf::zeroed(p * MLP_PADDED * 2)?,
                hw: DevBuf::zeroed(p * HIDDEN * 4)?,
                xw: DevBuf::zeroed(p * HIDDEN * 2)?,
                out: DevBuf::zeroed(p / 4 * LLM_HIDDEN * 4)?,
                pos_idx: DevBuf::zeroed(p * 4)?,
                window_src: DevBuf::zeroed(p * 4)?,
                merge_src: DevBuf::zeroed(p * 4)?,
                crops_early: DevBuf::zeroed(MAX_CROPS * 8)?,
                crops_late: DevBuf::zeroed(MAX_CROPS * 8)?,
                absmax: DevBuf::zeroed(64)?,
            })
        }

        /// Patch capacity this workspace was sized for.
        pub fn max_patches(&self) -> usize {
            self.max_patches
        }

        /// The projector's output of the last [`forward`]: fp32 `[plan.tokens, 2048]`.
        pub fn out(&self) -> &DevBuf {
            &self.out
        }

        /// Device bytes of every buffer.
        pub fn bytes(&self) -> u64 {
            [
                &self.patches,
                &self.h,
                &self.xn,
                &self.qkv,
                &self.att,
                &self.mid,
                &self.hw,
                &self.xw,
                &self.out,
                &self.pos_idx,
                &self.window_src,
                &self.merge_src,
                &self.crops_early,
                &self.crops_late,
                &self.absmax,
            ]
            .iter()
            .map(|b| b.bytes() as u64)
            .sum()
        }

        /// Largest magnitude of the projector's last output over `tokens` rows, `inf` when any value is not finite (one small kernel and a
        /// sync): the guard against an fp16 overflow somewhere in the tower, which would otherwise be served as garbage embeddings.
        pub fn output_absmax(&mut self, tokens: usize, st: &Stream) -> Result<f32, CudaError> {
            assert!(tokens * LLM_HIDDEN * 4 <= self.out.bytes());
            absmax(&self.out, true, tokens * LLM_HIDDEN, &mut self.absmax, st)
        }

        /// The residual stream's device buffer (fp32 `[rows, 1152]`).
        pub fn h(&self) -> &DevBuf {
            &self.h
        }
    }

    /// Per-stage readout of [`forward`] for comparing against the reference (every readout syncs and downloads: tests only).
    #[derive(Debug, Default, Clone)]
    pub struct Trace {
        /// `embed` (patch embedding plus position), `layer0` .. `layer26` (the residual stream after each, in that layer's row
        /// order), `merger` (the window merger's output, 16x only), `post_ln` (after `post_layernorm`, fp16 widened), `projector`.
        /// Each is the whole tensor, row-major.
        pub stages: Vec<(String, Vec<f32>)>,
        /// Largest magnitude of the fp16 intermediates and of the fp32 stream, over the whole forward: `xn`, `qkv`, `att`, `mid`
        /// (fp16: overflow at 65504) and `h` (fp32).
        pub absmax: Vec<(String, f32)>,
    }

    impl Trace {
        /// The stage `name`'s values (None when it did not run).
        pub fn stage(&self, name: &str) -> Option<&[f32]> {
            self.stages.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_slice())
        }
    }

    fn read_f32(b: &DevBuf, rows: usize, dim: usize, st: &Stream) -> Result<Vec<f32>, CudaError> {
        st.sync()?;
        let mut v = vec![0f32; rows * dim];
        b.download_range(0, &mut v)?;
        Ok(v)
    }

    fn read_f16(b: &DevBuf, rows: usize, dim: usize, st: &Stream) -> Result<Vec<f32>, CudaError> {
        st.sync()?;
        let mut v = vec![0u16; rows * dim];
        b.download_range(0, &mut v)?;
        Ok(v.iter().map(|&x| gb_exl3::f16::f16_to_f64(x) as f32).collect())
    }

    fn absmax(b: &DevBuf, is_f32: bool, n: usize, scratch: &mut DevBuf, st: &Stream) -> Result<f32, CudaError> {
        scratch.upload(&[0u32])?;
        CudaError::check(unsafe { ffi::gb_mcv_absmax(b.ptr(), is_f32 as i32, n, scratch.ptr(), st.raw()) })?;
        st.sync()?;
        let mut v = [0u32];
        scratch.download_range(0, &mut v)?;
        Ok(f32::from_bits(v[0]))
    }

    /// GPU time of one forward, from events between its groups.
    #[derive(Clone, Debug, Default)]
    pub struct Timing {
        /// `(group, milliseconds)`: `embed`, `early` (layers up to the merger), `merger`, `late`, `projector`.
        pub groups: Vec<(&'static str, f64)>,
        pub total_ms: f64,
    }

    static EVENT_POOL: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

    struct Rec {
        events: Vec<usize>,
        names: Vec<&'static str>,
    }

    impl Rec {
        fn event() -> Result<usize, CudaError> {
            if let Some(e) = EVENT_POOL.lock().unwrap().pop() {
                return Ok(e);
            }
            let mut e: *mut c_void = core::ptr::null_mut();
            CudaError::check(unsafe { crate::rt::timed_events::create(&mut e) })?;
            Ok(e as usize)
        }
        fn new(st: &Stream) -> Result<Rec, CudaError> {
            let mut r = Rec { events: Vec::new(), names: Vec::new() };
            let e = Rec::event()?;
            r.events.push(e);
            CudaError::check(unsafe { crate::rt::timed_events::record(e as *mut c_void, st.raw()) })?;
            Ok(r)
        }
        fn mark(&mut self, name: &'static str, st: &Stream) -> Result<(), CudaError> {
            let e = Rec::event()?;
            self.events.push(e);
            self.names.push(name);
            CudaError::check(unsafe { crate::rt::timed_events::record(e as *mut c_void, st.raw()) })
        }
        fn finish(self) -> Result<Timing, CudaError> {
            let last = *self.events.last().unwrap();
            let mut res = CudaError::check(unsafe { crate::rt::timed_events::synchronize(last as *mut c_void) });
            let mut t = Timing::default();
            if res.is_ok() {
                for (k, &name) in self.names.iter().enumerate() {
                    let mut ms = 0f32;
                    let rc = unsafe {
                        crate::rt::timed_events::elapsed_ms(
                            &mut ms,
                            self.events[k] as *mut c_void,
                            self.events[k + 1] as *mut c_void,
                        )
                    };
                    if rc != 0 {
                        res = Err(CudaError(rc));
                        break;
                    }
                    t.total_ms += ms as f64;
                    t.groups.push((name, ms as f64));
                }
            }
            EVENT_POOL.lock().unwrap().extend(self.events);
            res.map(|_| t)
        }
    }

    /// The vision path from the packed pixels: bf16 bits `[3, 14, 14 * plan.patches]` (device), the crops of `plan`. The projector's
    /// output (fp32 `[plan.tokens, 2048]`, crop-major tokens, row-major inside a crop's output grid) is [`Workspace::out`].
    pub fn forward(
        w: &Weights,
        ws: &mut Workspace,
        pixels: &DevBuf,
        plan: &Plan,
        trace: Option<&mut Trace>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        run(w, ws, pixels, plan, trace, None, st)
    }

    /// [`forward`] with the GPU time of its groups read back from events (one sync, at the end; the launches are the same).
    pub fn forward_timed(
        w: &Weights,
        ws: &mut Workspace,
        pixels: &DevBuf,
        plan: &Plan,
        st: &Stream,
    ) -> Result<Timing, CudaError> {
        let mut rec = Rec::new(st)?;
        run(w, ws, pixels, plan, None, Some(&mut rec), st)?;
        rec.finish()
    }

    /// Stream-ordered host-to-device copy of `v` into the start of `dst` (pageable memory is staged before the call returns).
    pub fn h2d<T: Copy>(dst: &mut DevBuf, v: &[T], st: &Stream) -> Result<(), CudaError> {
        let n = std::mem::size_of_val(v);
        assert!(n <= dst.bytes(), "h2d of {n} bytes into a {}-byte buffer", dst.bytes());
        if n == 0 {
            return Ok(());
        }
        unsafe { st.memcpy_h2d_async(dst.ptr(), v.as_ptr() as *const u8, n) }
    }

    /// `y[y_off..] = bf16(x[x_off..])` over `n` elements (fp32 -> bf16, round to nearest even), both offsets in bytes.
    pub fn f32_to_bf16_at(
        x: &DevBuf,
        x_off: usize,
        y: &mut DevBuf,
        y_off: usize,
        n: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        need(x, x_off, n * 4, "f32_to_bf16 x");
        need(y, y_off, n * 2, "f32_to_bf16 y");
        CudaError::check(unsafe { ffi::gb_vis_f32_to_bf16(x.ptr_at(x_off), y.ptr_at(y_off), n, st.raw()) })
    }

    /// `y = bf16(x)` over `n` elements (fp32 -> bf16, round to nearest even): the export the encoder hands the language model.
    pub fn f32_to_bf16(x: &DevBuf, y: &mut DevBuf, n: usize, st: &Stream) -> Result<(), CudaError> {
        need(x, 0, n * 4, "f32_to_bf16 x");
        need(y, 0, n * 2, "f32_to_bf16 y");
        CudaError::check(unsafe { ffi::gb_vis_f32_to_bf16(x.ptr(), y.ptr(), n, st.raw()) })
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        w: &Weights,
        ws: &mut Workspace,
        pixels: &DevBuf,
        plan: &Plan,
        mut trace: Option<&mut Trace>,
        mut rec: Option<&mut Rec>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let p = plan.patches;
        assert!(p <= ws.max_patches, "forward: {p} patches, the workspace holds {}", ws.max_patches);
        assert!(plan.crops.len() <= MAX_CROPS, "forward: {} crops, the tables hold {MAX_CROPS}", plan.crops.len());
        let merged = plan.mode == Mode::X16;
        macro_rules! mark {
            ($name:expr) => {
                if let Some(r) = rec.as_deref_mut() {
                    r.mark($name, st)?;
                }
            };
        }
        // the host-built tables
        h2d(&mut ws.pos_idx, &plan.pos_idx, st)?;
        if merged {
            h2d(&mut ws.window_src, &plan.window_src, st)?;
        }
        h2d(&mut ws.merge_src, &plan.merge_src, st)?;
        h2d(&mut ws.crops_early, &plan.early_crops(), st)?;
        h2d(&mut ws.crops_late, &plan.late_crops(), st)?;
        if let Some(tr) = trace.as_deref_mut() {
            *tr = Trace::default();
        }
        let mut seen = std::collections::BTreeMap::<&'static str, f32>::new();
        macro_rules! watch {
            ($name:expr, $buf:expr, $f32:expr, $n:expr) => {
                if trace.is_some() {
                    let a = absmax(&$buf, $f32, $n, &mut ws.absmax, st)?;
                    let e = seen.entry($name).or_insert(0.0);
                    if a > *e {
                        *e = a;
                    }
                }
            };
        }

        // patch embedding: gather, GEMM (bias), position add
        patchify(pixels, &mut ws.patches, p, st)?;
        gemm(Epi::StoreF32, &ws.patches, &w.patch_w, Some(&w.patch_b), &mut ws.h, 0, p, HIDDEN, PATCH_PADDED, HIDDEN, st)?;
        CudaError::check(unsafe { ffi::gb_mcv_pos_add(ws.h.ptr(), w.pos.ptr(), ws.pos_idx.ptr(), i(p)?, st.raw()) })?;
        mark!("embed");
        if let Some(tr) = trace.as_deref_mut() {
            tr.stages.push(("embed".into(), read_f32(&ws.h, p, HIDDEN, st)?));
        }

        let mut rows = p;
        let mut max_rows = plan.max_early_rows();
        let mut crops_late = false;
        for (li, l) in w.layers.iter().enumerate() {
            layernorm(&ws.h, &l.ln1_w, &l.ln1_b, &mut ws.xn, rows, HIDDEN, LN_EPS, st)?;
            gemm(Epi::BiasF16, &ws.xn, &l.qkv_w, Some(&l.qkv_b), &mut ws.qkv, 0, rows, 3 * HIDDEN, HIDDEN, 3 * HIDDEN, st)?;
            {
                let crops = if crops_late { &ws.crops_late } else { &ws.crops_early };
                attention(&ws.qkv, &mut ws.att, crops, plan.crops.len(), max_rows, rows, st)?;
            }
            gemm(Epi::BiasAddF32, &ws.att, &l.out_w, Some(&l.out_b), &mut ws.h, 0, rows, HIDDEN, HIDDEN, HIDDEN, st)?;
            layernorm(&ws.h, &l.ln2_w, &l.ln2_b, &mut ws.xn, rows, HIDDEN, LN_EPS, st)?;
            gemm(
                Epi::BiasGeluTanhF16,
                &ws.xn,
                &l.fc1_w,
                Some(&l.fc1_b),
                &mut ws.mid,
                0,
                rows,
                MLP_PADDED,
                HIDDEN,
                MLP_PADDED,
                st,
            )?;
            gemm(Epi::BiasAddF32, &ws.mid, &l.fc2_w, Some(&l.fc2_b), &mut ws.h, 0, rows, HIDDEN, MLP_PADDED, HIDDEN, st)?;
            if trace.is_some() {
                watch!("xn", ws.xn, false, rows * HIDDEN);
                watch!("qkv", ws.qkv, false, rows * 3 * HIDDEN);
                watch!("att", ws.att, false, rows * HIDDEN);
                watch!("mid", ws.mid, false, rows * MLP_PADDED);
                watch!("h", ws.h, true, rows * HIDDEN);
                if let Some(tr) = trace.as_deref_mut() {
                    tr.stages.push((format!("layer{li}"), read_f32(&ws.h, rows, HIDDEN, st)?));
                }
            }
            if merged && li == MERGER_AFTER_LAYER {
                mark!("early");
                let m = &w.merger;
                let nwin = rows / 4;
                layernorm(&ws.h, &m.ln1_w, &m.ln1_b, &mut ws.xn, rows, HIDDEN, LN_EPS, st)?;
                CudaError::check(unsafe {
                    ffi::gb_mcv_window_gather(
                        ws.h.ptr(),
                        ws.xn.ptr(),
                        ws.window_src.ptr(),
                        ws.hw.ptr(),
                        ws.xw.ptr(),
                        i(rows)?,
                        st.raw(),
                    )
                })?;
                gemm(Epi::BiasF16, &ws.xw, &m.qkv_w, Some(&m.qkv_b), &mut ws.qkv, 0, rows, 3 * HIDDEN, HIDDEN, 3 * HIDDEN, st)?;
                CudaError::check(unsafe { ffi::gb_mcv_window_attention(ws.qkv.ptr(), ws.att.ptr(), i(nwin)?, st.raw()) })?;
                gemm(Epi::BiasAddF32, &ws.att, &m.out_w, Some(&m.out_b), &mut ws.hw, 0, rows, HIDDEN, HIDDEN, HIDDEN, st)?;
                // hw viewed as [nwin, 4608] is the concat of each window's four tokens
                layernorm(&ws.hw, &m.pre_w, &m.pre_b, &mut ws.xw, nwin, WINDOW_HIDDEN, LN_EPS, st)?;
                CudaError::check(unsafe { ffi::gb_mcv_mean4(ws.hw.ptr(), ws.h.ptr(), i(nwin)?, st.raw()) })?;
                gemm(
                    Epi::BiasGeluTanhF16,
                    &ws.xw,
                    &m.lin1_w,
                    Some(&m.lin1_b),
                    &mut ws.mid,
                    0,
                    nwin,
                    MERGER_MLP,
                    WINDOW_HIDDEN,
                    MERGER_MLP,
                    st,
                )?;
                gemm(Epi::BiasAddF32, &ws.mid, &m.lin2_w, Some(&m.lin2_b), &mut ws.h, 0, nwin, HIDDEN, MERGER_MLP, HIDDEN, st)?;
                rows = nwin;
                max_rows = plan.max_late_rows();
                crops_late = true;
                mark!("merger");
                if let Some(tr) = trace.as_deref_mut() {
                    tr.stages.push(("merger".into(), read_f32(&ws.h, rows, HIDDEN, st)?));
                }
            }
        }
        if !merged {
            mark!("early");
        }
        mark!("late");
        debug_assert_eq!(rows, plan.late_rows);
        // post_layernorm, the 2 x 2 concat, the projector
        layernorm(&ws.h, &w.post_w, &w.post_b, &mut ws.xn, rows, HIDDEN, LN_EPS, st)?;
        if let Some(tr) = trace.as_deref_mut() {
            tr.stages.push(("post_ln".into(), read_f16(&ws.xn, rows, HIDDEN, st)?));
        }
        let nt = plan.tokens;
        CudaError::check(unsafe { ffi::gb_mcv_merge4(ws.xn.ptr(), ws.merge_src.ptr(), ws.att.ptr(), i(nt)?, st.raw()) })?;
        let pr = &w.projector;
        layernorm_f16(&ws.att, &pr.pre_w, &pr.pre_b, &mut ws.qkv, nt, WINDOW_HIDDEN, LN_EPS, st)?;
        gemm(
            Epi::BiasGeluF16,
            &ws.qkv,
            &pr.lin1_w,
            Some(&pr.lin1_b),
            &mut ws.mid,
            0,
            nt,
            WINDOW_HIDDEN,
            WINDOW_HIDDEN,
            WINDOW_HIDDEN,
            st,
        )?;
        gemm(
            Epi::StoreF32,
            &ws.mid,
            &pr.lin2_w,
            Some(&pr.lin2_b),
            &mut ws.out,
            0,
            nt,
            LLM_HIDDEN,
            WINDOW_HIDDEN,
            LLM_HIDDEN,
            st,
        )?;
        mark!("projector");
        if let Some(tr) = trace {
            tr.stages.push(("projector".into(), read_f32(&ws.out, nt, LLM_HIDDEN, st)?));
            tr.absmax = seen.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_the_checkpoint() {
        assert_eq!(HIDDEN, HEADS * HEAD_DIM);
        assert_eq!(MLP_PADDED % 32, 0);
        const { assert!(MLP_PADDED >= MLP_HIDDEN && MLP_PADDED - MLP_HIDDEN < 32) };
        assert_eq!(PATCH_PADDED % 32, 0);
        const { assert!(PATCH_PADDED >= PATCH_DIM && PATCH_PADDED - PATCH_DIM < 32) };
        assert_eq!(MERGER_MLP, 17216);
        assert_eq!(WINDOW_HIDDEN, 4608);
        assert_eq!(MERGER_MLP % 32, 0);
    }

    #[test]
    fn a_448_pixel_crop_is_64_tokens_at_16x_and_256_at_4x() {
        let p16 = Plan::new(&[(32, 32)], Mode::X16, usize::MAX).unwrap();
        assert_eq!((p16.patches, p16.late_rows, p16.tokens), (1024, 256, 64));
        assert_eq!(p16.out_grid, vec![(8, 8)]);
        let p4 = Plan::new(&[(32, 32)], Mode::X4, usize::MAX).unwrap();
        assert_eq!((p4.patches, p4.late_rows, p4.tokens), (1024, 1024, 256));
        assert_eq!(p4.out_grid, vec![(16, 16)]);
    }

    #[test]
    fn plans_refuse_grids_the_merges_cannot_take() {
        assert_eq!(Plan::new(&[], Mode::X16, 100), Err(PlanError::NoCrops));
        assert!(matches!(Plan::new(&[(30, 32)], Mode::X16, usize::MAX), Err(PlanError::BadGrid { crop: 0, .. })));
        assert!(Plan::new(&[(30, 32)], Mode::X4, usize::MAX).is_ok(), "4x needs even sides only");
        assert!(matches!(Plan::new(&[(31, 32)], Mode::X4, usize::MAX), Err(PlanError::BadGrid { .. })));
        assert!(matches!(Plan::new(&[(0, 4)], Mode::X16, usize::MAX), Err(PlanError::BadGrid { .. })));
        assert!(matches!(
            Plan::new(&[(32, 32), (32, 32)], Mode::X16, 2000),
            Err(PlanError::TooManyPatches { patches: 2048, max: 2000 })
        ));
    }

    #[test]
    fn window_and_merge_orders_are_what_the_reference_builds() {
        // 4 x 4 grid, patch index = row * 4 + col
        assert_eq!(window_source(4, 4), vec![0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]);
        assert_eq!(merge_source(4, 4), window_source(4, 4), "the merge is the same 2 x 2 blocking");
        // crops are concatenated with their row offsets
        let p = Plan::new(&[(4, 4), (4, 8)], Mode::X16, usize::MAX).unwrap();
        assert_eq!(&p.window_src[..4], &[0, 1, 4, 5]);
        assert_eq!(&p.window_src[16..20], &[16, 17, 24, 25]);
        // late rows: 4 + 8; merge source indexes the late rows, offsets 0 and 4
        assert_eq!(p.merge_src.len(), 12);
        assert_eq!(&p.merge_src[..4], &[0, 1, 2, 3]);
        assert_eq!(
            &p.merge_src[4..8],
            &[4, 5, 8, 9],
            "crop 2 late grid is 2 x 4: token (0, 0) takes rows 0, 1 of its first two columns"
        );
        assert_eq!(&p.merge_src[8..12], &[6, 7, 10, 11]);
    }

    #[test]
    fn nearest_position_ids_cover_the_table_corners() {
        // a 70 x 70 grid maps every patch to itself
        let ids = nearest_position_ids(70, 70);
        assert_eq!(ids.len(), 4900);
        assert!(ids.iter().enumerate().all(|(k, &v)| v == k as i32), "70 x 70 is the identity");
        // a single patch sits at bucket 0
        assert_eq!(nearest_position_ids(1, 1), vec![0]);
        // the last patch of any grid stays inside the table
        for n in 1..=200usize {
            let v = nearest_position_ids(n, n);
            assert!(*v.iter().max().unwrap() < 4900);
            assert_eq!(v[0], 0);
        }
    }

    #[test]
    fn flops_of_one_crop_at_16x_are_about_480_gflop() {
        let f = Plan::new(&[(32, 32)], Mode::X16, usize::MAX).unwrap().flops();
        assert!((4.5e11..5.2e11).contains(&f), "{f:e}");
        let f4 = Plan::new(&[(32, 32)], Mode::X4, usize::MAX).unwrap().flops();
        assert!((9.0e11..1.1e12).contains(&f4), "{f4:e}");
    }
}
