//! The GLM-5.3-Flash vision tower on the GPU: 24 blocks of hidden 1024 (16 heads of 64, a SwiGLU MLP of 4096, RMSNorm), a 2x2 merge to 4096 and a
//! merger with a gated MLP of 10,240. The tower is plain bf16 in the pack (347 tensors, no EXL3), loaded through the pipelined loader as the plan's
//! one `Vision` region and prepared in place on the device: bf16 -> fp16 for every Linear/Conv/LayerNorm tensor (the fork casts at load), the
//! temporal conv folded over its two slots, the downsample weight put in the order of the merged patches. RMSNorm gammas stay bf16.
//!
//! The numerics are the ExLlamaV3 fork's `glm5_next_vision` (the oracle, `tools/dump_glm_vision_reference.py`), stage by stage; `csrc/glm_vision.cu`
//! documents them. There is no learned position embedding and no M-RoPE: positions enter only through the 2-D rope of q and k, and GLM's text attention
//! is NoPE, so an image's features replace the placeholder embeddings in place (`GlmModel::splice`).
//!
//! One image per call. An image's features are a function of that image alone (no batching, fixed kernels, a fixed cuBLAS call per shape), so the output
//! is deterministic for an image whatever else the process does; it is not part of the serial-equals-drafted contract (`docs/CUDA_BACKEND.md`).

use super::config::{GlmConfig, GlmVisionConfig, VisionTowerFormat};
use super::load::{allocate, LoadOptions, Regions};
use super::plan::{LoadPlan, RegionKind};
use super::weights::{align_up, GlmVisionWeights, PlainLinear, PlainTensor, VisionLinear, VisionQkv};
use super::GlmPack;
use crate::error::ModelError;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::glm_vision as gv;
use gb_cuda::loader::Options as LoaderOptions;
use gb_cuda::{DevBuf, Stream};
use gb_exl3::loadstats::Component;
use std::path::Path;

pub const DEPTH: usize = 24;
pub const HIDDEN: usize = 1024;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 64;
/// The block MLP's width.
pub const MLP: usize = 4096;
/// Width of the embeddings handed to the language model.
pub const OUT_HIDDEN: usize = 4096;
/// The merger's gated MLP.
pub const MERGER_MLP: usize = 10240;
pub const MERGE: usize = 2;
/// Values of one still-image patch without the temporal repeat: 3 channels x 14 x 14.
pub const PATCH_K: usize = 3 * 14 * 14;
/// `PATCH_K` padded to a multiple of 8 so the patch GEMM's leading dimension is aligned for the tensor cores (the pad columns are zero in input and weight).
pub const PATCH_KP: usize = 592;
pub const RMS_EPS: f32 = 1e-5;
pub const LN_EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 10000.0;
pub const SWIGLU_LIMIT: f32 = 10.0;
/// cuBLAS workspace the tower hands the library (its own, so it is counted).
pub const GEMM_WORKSPACE: usize = 32 << 20;
/// Merged patches per token.
pub const PATCHES_PER_TOKEN: usize = MERGE * MERGE;

/// The numbers the kernels are written for, checked against the pack's `vision_config`: a different tower fails loudly.
pub fn check_config(v: &GlmVisionConfig) -> Result<(), ModelError> {
    let checks = [
        ("depth", v.depth, DEPTH),
        ("hidden_size", v.hidden, HIDDEN),
        ("num_heads", v.heads, HEADS),
        ("intermediate_size", v.intermediate, MLP),
        ("out_hidden_size", v.out_hidden, OUT_HIDDEN),
        ("projection_intermediate_size", v.projection_intermediate, MERGER_MLP),
        ("spatial_merge_size", v.spatial_merge_size, MERGE),
        ("patch_size", v.patch_size, 14),
        ("temporal_patch_size", v.temporal_patch_size, 2),
        ("in_channels", v.in_channels, 3),
    ];
    for (k, got, want) in checks {
        if got != want {
            return Err(ModelError::Config(format!("vision_config.{k} is {got}, the GLM-5.3 tower is written for {want}")));
        }
    }
    if (v.rms_eps - 1e-5).abs() > 1e-12 || v.swiglu_limit != SWIGLU_LIMIT as f64 || !v.attention_bias {
        return Err(ModelError::Config(format!(
            "vision_config: rms_norm_eps {}, swiglu_limit {}, attention_bias {} (the tower is written for {RMS_EPS}, {SWIGLU_LIMIT}, true)",
            v.rms_eps, v.swiglu_limit, v.attention_bias
        )));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------------ memory arithmetic

/// The most tokens one image of this server may take: the operator's `--max-image-tokens`, capped by the processor's own `max_image_tokens` (8,000 in this
/// pack's `processor_config.json`; the processor never produces more, so a tower sized for more would only waste memory). The tower's scratch, the memory
/// model's reserve and the preprocessing's canvas budget all use this one number.
pub fn image_token_cap(cfg: &GlmConfig, requested: usize) -> usize {
    match &cfg.processor {
        Some(p) => requested.min(p.image.max_image_tokens),
        None => requested,
    }
}

/// What the tower allocates for its scratch for images of up to `max_patches` patches (the encoder's working buffers, one set, held for the life
/// of the process): `(name, bytes)`. The memory model (`GlmVisionConfig::encoder_scratch_bytes`) and the allocations are both this function, so the
/// plan equals what is allocated by construction, and a GPU test sums the real allocations against it.
///
/// The EXL3 tower adds the staging of its linears (`Exl3Stage`): one reconstructed `W_q` of the widest linear (`[4096, 10240]` fp16), and for a chunk of
/// [`EXL3_ROWS`] rows the contiguous input (a strided GEMM input: the down projections read the activation's columns), the rotated input, the fp32
/// GEMM output and the fp16 output before it is placed at its columns (q | k | v, gate | up).
pub fn scratch_items(max_patches: usize, tower: VisionTowerFormat) -> Vec<(&'static str, u64)> {
    let p = max_patches as u64;
    let mut v = vec![
        // the still patches as the patch GEMM reads them: [P, 592] fp16 (the 4 pad columns stay zero)
        ("patches", p * PATCH_KP as u64 * 2),
        // the fp32 residual stream [P, 1024]
        ("residual", p * HIDDEN as u64 * 4),
        // the normed input of the next GEMM [P, 1024] fp16 (after the blocks: the post norm, read by the downsample as [P/4, 4096])
        ("normed", p * HIDDEN as u64 * 2),
        // the qkv GEMM's output [P, 3072] fp16, turned into q | k | v in place
        ("qkv", p * 3 * HIDDEN as u64 * 2),
        // the attention output [P, 1024]; later the downsample output [P/4, 4096] (the same bytes) and the merger LayerNorm output
        ("attention", p * HIDDEN as u64 * 2),
        // the output projection's and down projection's raw GEMM output [P, 1024]; later the merger proj output [P/4, 4096] (the same bytes)
        ("sublayer", p * HIDDEN as u64 * 2),
        // gate | up [P, 8192] fp16, the activation overwrites the gate; later the merger's [P/4, 20480] (fewer bytes)
        ("gate_up", p * 2 * MLP as u64 * 2),
        // the rope's inverse frequencies, 16 f32, and the non-finite counter of the weight conversion (one i32)
        ("inv_freq", 16 * 4),
        ("counter", 4),
        ("gemm_workspace", GEMM_WORKSPACE as u64),
    ];
    if let VisionTowerFormat::Exl3 { .. } = tower {
        let rows = exl3_rows(max_patches) as u64;
        // the widest dimension of any linear (the merger's MLP) bounds every staged row
        let wide = MERGER_MLP as u64;
        v.extend([
            ("exl3_w", (OUT_HIDDEN * MERGER_MLP) as u64 * 2),
            ("exl3_x", rows * wide * 2),
            ("exl3_xh", rows * wide * 2),
            ("exl3_c", rows * wide * 4),
            ("exl3_y", rows * wide * 2),
        ]);
    }
    v
}

pub fn scratch_bytes(max_patches: usize, tower: VisionTowerFormat) -> u64 {
    scratch_items(max_patches, tower).iter().map(|i| i.1).sum()
}

/// Rows of one staged EXL3 GEMM: the tower's linears run over the image in chunks of this many rows, which bounds the staging (and the EXL3
/// prefill's fp32 output) whatever the image's size.
pub const EXL3_ROWS: usize = 1024;

fn exl3_rows(max_patches: usize) -> usize {
    EXL3_ROWS.min(max_patches.max(1))
}

/// Bytes of the `Vision` region the plan allocates (the tensors' bytes, each placed at a 256-byte boundary), from the config alone: it equals the
/// plan's region on the real pack (a test), which is what the loader allocates and the tower keeps as its weights.
pub fn weight_region_bytes(v: &GlmVisionConfig) -> u64 {
    let (h, i, o, p) = (v.hidden as u64, v.intermediate as u64, v.out_hidden as u64, v.projection_intermediate as u64);
    let hd = h / v.heads as u64;
    let m = v.spatial_merge_size as u64;
    let kk = v.in_channels as u64 * v.temporal_patch_size as u64 * (v.patch_size * v.patch_size) as u64;
    let mut sizes: Vec<u64> = vec![h * kk * 2, h * 2];
    match v.tower {
        VisionTowerFormat::Bf16 => {
            for _ in 0..v.depth {
                sizes.extend([h, h, hd, hd].map(|n| n * 2));
                sizes.extend([3 * h * h, 3 * h, h * h, h, i * h, i, i * h, i, h * i, h].map(|n| n * 2));
            }
            sizes.push(h * 2);
            sizes.extend([o * h * m * m, o].map(|n| n * 2));
            sizes.extend([o * o, p * o, p * o, o * p, o, o].map(|n| n * 2));
        }
        VisionTowerFormat::Exl3 { bits } => {
            // the binder's order: a group's trellis, suh, svh, then its bias (q, k and v: the three groups, then the three biases)
            let g = |k: u64, n: u64| [k * n * bits as u64 / 8, k * 2, n * 2];
            for _ in 0..v.depth {
                sizes.extend([h, h, hd, hd].map(|n| n * 2));
                (0..3).for_each(|_| sizes.extend(g(h, h)));
                sizes.extend([h, h, h].map(|n| n * 2));
                sizes.extend(g(h, h));
                sizes.push(h * 2);
                sizes.extend(g(h, i));
                sizes.push(i * 2);
                sizes.extend(g(h, i));
                sizes.push(i * 2);
                sizes.extend(g(i, h));
                sizes.push(h * 2);
            }
            sizes.push(h * 2);
            sizes.extend([o * h * m * m, o].map(|n| n * 2));
            sizes.extend(g(o, o));
            sizes.extend(g(o, p));
            sizes.extend(g(o, p));
            sizes.extend(g(p, o));
            sizes.extend([o, o].map(|n| n * 2));
        }
    }
    sizes.iter().fold(0u64, |end, &b| align_up(end) + b)
}

// ------------------------------------------------------------------------------------------------------------------ mutations

/// A deliberate bug of the tower, to show that a test notices it (`tests/glm_vision.rs`). Never set in serving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutant {
    /// The rope's row and column angles exchanged.
    RopeAxisSwap,
    /// No rope on q and k.
    NoRope,
    /// No per-head RMSNorm of q and k.
    NoHeadNorm,
    /// The qkv bias dropped.
    NoQkvBias,
    /// The attention output projection's bias dropped.
    NoProjBias,
    /// The MLP's gate and up biases dropped.
    NoMlpBias,
    /// The MLP's down projection bias dropped.
    NoDownBias,
    /// The patch embedding's bias dropped.
    NoPatchBias,
    /// The downsample's bias dropped.
    NoDownsampleBias,
    /// The blocks' RMSNorm eps 1e-6 instead of 1e-5.
    RmsEps1e6,
    /// The merger's LayerNorm eps 1e-5 instead of 1e-6.
    LnEps1e5,
    /// The temporal conv folded from slot 0 alone.
    FoldSlot0Only,
    /// The downsample weight left in the conv's own order (channel-major) instead of the merged patches' order.
    MergeOrderConv,
    /// The two patches of the merge block's diagonal exchanged (a transposed block).
    MergeOrderTransposed,
    /// Causal attention instead of bidirectional.
    AttnCausal,
    /// The merger's GELU by the tanh formula.
    GeluTanh,
    /// The block MLP without its clamps.
    BlockNoClamp,
    /// The block MLP clamping the gate before the silu (the merger's order).
    BlockClampBeforeSilu,
    /// The merger without its clamps.
    MergerNoClamp,
}

impl Mutant {
    pub const ALL: [Mutant; 19] = [
        Mutant::RopeAxisSwap,
        Mutant::NoRope,
        Mutant::NoHeadNorm,
        Mutant::NoQkvBias,
        Mutant::NoProjBias,
        Mutant::NoMlpBias,
        Mutant::NoDownBias,
        Mutant::NoPatchBias,
        Mutant::NoDownsampleBias,
        Mutant::RmsEps1e6,
        Mutant::LnEps1e5,
        Mutant::FoldSlot0Only,
        Mutant::MergeOrderConv,
        Mutant::MergeOrderTransposed,
        Mutant::AttnCausal,
        Mutant::GeluTanh,
        Mutant::BlockNoClamp,
        Mutant::BlockClampBeforeSilu,
        Mutant::MergerNoClamp,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Mutant::RopeAxisSwap => "rope row/column axes exchanged",
            Mutant::NoRope => "no rope",
            Mutant::NoHeadNorm => "no q/k head RMSNorm",
            Mutant::NoQkvBias => "qkv bias dropped",
            Mutant::NoProjBias => "attention output bias dropped",
            Mutant::NoMlpBias => "MLP gate/up bias dropped",
            Mutant::NoDownBias => "MLP down bias dropped",
            Mutant::NoPatchBias => "patch embedding bias dropped",
            Mutant::NoDownsampleBias => "downsample bias dropped",
            Mutant::RmsEps1e6 => "block RMSNorm eps 1e-6",
            Mutant::LnEps1e5 => "merger LayerNorm eps 1e-5",
            Mutant::FoldSlot0Only => "temporal fold from slot 0 only",
            Mutant::MergeOrderConv => "downsample weight in conv order",
            Mutant::MergeOrderTransposed => "merge block transposed",
            Mutant::AttnCausal => "causal attention",
            Mutant::GeluTanh => "merger GELU tanh",
            Mutant::BlockNoClamp => "block MLP clamps dropped",
            Mutant::BlockClampBeforeSilu => "block MLP clamps g before silu",
            Mutant::MergerNoClamp => "merger clamps dropped",
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------ weights

/// A linear of the tower on the device: an fp16 weight `[n, k]` for the dense GEMM (the bf16 tower, converted in place), or an EXL3 group run by
/// the EXL3 prefill GEMM in staged chunks (the EXL3 tower). Biases are separate buffers either way.
enum Lin {
    Dense(DevBuf),
    Exl3(Exl3Weights),
}

struct BlockW {
    norm1: DevBuf,
    norm2: DevBuf,
    q_norm: DevBuf,
    k_norm: DevBuf,
    /// The fused `[3h, h]` weight (bf16 tower) or q, k and v (EXL3 tower).
    qkv: Vec<Lin>,
    /// `[3h]`: the fused bias, or the q, k and v biases the plan placed back to back.
    qkv_b: DevBuf,
    proj: Lin,
    proj_b: DevBuf,
    gate: Lin,
    gate_b: DevBuf,
    up: Lin,
    up_b: DevBuf,
    down: Lin,
    down_b: DevBuf,
}

struct Weights {
    format: VisionTowerFormat,
    /// `[1024, 592]` fp16 (the folded conv weight) in the front of its slot in the region.
    patch_w: DevBuf,
    patch_b: DevBuf,
    blocks: Vec<BlockW>,
    post_norm: DevBuf,
    /// `[4096, 4096]` fp16, K in the order of the merged patches (patch within the block, then channel).
    ds_w: DevBuf,
    ds_b: DevBuf,
    m_proj: Lin,
    m_gate: Lin,
    m_up: Lin,
    m_down: Lin,
    m_norm_w: DevBuf,
    m_norm_b: DevBuf,
    /// The tensors the bf16 tower converts to fp16 in place (`GlmVisionWeights::bf16_conversions`; none in the EXL3 tower).
    conv: Vec<DevBuf>,
}

fn view(r: &Regions, t: &PlainTensor) -> Result<DevBuf, ModelError> {
    r.view(t.dst.region, t.dst.offset, t.dst_bytes)
}

fn linear(r: &Regions, l: &PlainLinear) -> Result<(DevBuf, DevBuf), ModelError> {
    let b =
        l.bias.as_ref().ok_or_else(|| ModelError::Config(format!("`{}` has no bias in this pack's layout", l.weight.name)))?;
    Ok((view(r, &l.weight)?, view(r, b)?))
}

fn lin(r: &Regions, l: &VisionLinear) -> Result<Lin, ModelError> {
    Ok(match l {
        VisionLinear::Plain(p) => Lin::Dense(view(r, &p.weight)?),
        VisionLinear::Exl3 { group, .. } => Lin::Exl3(super::device::exl3_weights(r, group)?),
    })
}

fn lin_bias(r: &Regions, l: &VisionLinear) -> Result<(Lin, DevBuf), ModelError> {
    let b = l.bias().ok_or_else(|| ModelError::Config(format!("`{}` has no bias in this pack's layout", l.name())))?;
    Ok((lin(r, l)?, view(r, b)?))
}

/// The fused `[3h]` qkv bias as one view: the bf16 tower's own tensor, or the EXL3 tower's q, k and v biases, which the plan placed back to back
/// (checked here: a plan that did not would make the qkv kernel read the wrong bytes, so it is refused by name).
fn qkv_bias(r: &Regions, q: &VisionQkv) -> Result<DevBuf, ModelError> {
    match q {
        VisionQkv::Fused(l) => linear(r, l).map(|(_, b)| b),
        VisionQkv::Split(qkv) => {
            let bs: Vec<&PlainTensor> = qkv
                .iter()
                .map(|l| l.bias().ok_or_else(|| ModelError::Config(format!("`{}` has no bias in this pack's layout", l.name()))))
                .collect::<Result<_, _>>()?;
            let adjacent =
                bs.windows(2).all(|w| w[1].dst.region == w[0].dst.region && w[1].dst.offset == w[0].dst.offset + w[0].dst_bytes);
            if !adjacent || bs.iter().any(|b| b.dst_bytes != (HIDDEN * 2) as u64) {
                return Err(ModelError::Config(format!(
                    "the q, k and v biases of `{}` are not one [3 x {HIDDEN}] fp16 span in the vision region",
                    qkv[0].name()
                )));
            }
            r.view(bs[0].dst.region, bs[0].dst.offset, 3 * bs[0].dst_bytes)
        }
    }
}

impl Weights {
    fn bind(r: &Regions, w: &GlmVisionWeights) -> Result<Self, ModelError> {
        if w.blocks.len() != DEPTH {
            return Err(ModelError::Config(format!(
                "the vision weights have {} blocks, the tower is written for {DEPTH}",
                w.blocks.len()
            )));
        }
        let (patch_w, patch_b) = linear(r, &w.patch_embed)?;
        let mut blocks = Vec::with_capacity(DEPTH);
        for b in &w.blocks {
            let qkv = match &b.qkv {
                VisionQkv::Fused(l) => vec![Lin::Dense(view(r, &l.weight)?)],
                VisionQkv::Split(qkv) => qkv.iter().map(|l| lin(r, l)).collect::<Result<_, _>>()?,
            };
            let (proj, proj_b) = lin_bias(r, &b.proj)?;
            let (gate, gate_b) = lin_bias(r, &b.gate_proj)?;
            let (up, up_b) = lin_bias(r, &b.up_proj)?;
            let (down, down_b) = lin_bias(r, &b.down_proj)?;
            blocks.push(BlockW {
                norm1: view(r, &b.norm1)?,
                norm2: view(r, &b.norm2)?,
                q_norm: view(r, &b.q_norm)?,
                k_norm: view(r, &b.k_norm)?,
                qkv,
                qkv_b: qkv_bias(r, &b.qkv)?,
                proj,
                proj_b,
                gate,
                gate_b,
                up,
                up_b,
                down,
                down_b,
            });
        }
        let (ds_w, ds_b) = linear(r, &w.downsample)?;
        let m = &w.merger;
        let conv = w.bf16_conversions().into_iter().map(|t| view(r, t)).collect::<Result<_, _>>()?;
        Ok(Self {
            format: w.format,
            patch_w,
            patch_b,
            blocks,
            post_norm: view(r, &w.post_layernorm)?,
            ds_w,
            ds_b,
            m_proj: lin(r, &m.proj)?,
            m_gate: lin(r, &m.gate_proj)?,
            m_up: lin(r, &m.up_proj)?,
            m_down: lin(r, &m.down_proj)?,
            m_norm_w: view(r, &m.post_projection_norm_weight)?,
            m_norm_b: view(r, &m.post_projection_norm_bias)?,
            conv,
        })
    }

    /// The bf16 tower: bf16 -> fp16 in place for every tensor the fork loads through `Linear`, `Conv` or `LayerNorm` (its `get_tensor` casts bf16
    /// to fp16 unless the module asks for bf16, which only RMSNorm does), the temporal fold of the patch embedding and the merge-order permutation
    /// of the downsample. Fails if any converted value is not finite (a bf16 beyond fp16's range: the fork would hold an inf there too, but a pack
    /// that has one is not a tower this build was validated on).
    ///
    /// The EXL3 tower: nothing. Its plain tensors are fp16 as stored or bf16 RMSNorm gammas, its patch and downsample weights were folded and
    /// permuted by the plan on the host (`Xform::FoldPatchF16`, `Xform::PermuteDownsampleF16`), and its linears are EXL3 groups. A bf16 -> fp16
    /// conversion of fp16 bytes would decode garbage, so none is run (the list of conversions is empty by construction, and a CPU test holds every
    /// listed tensor to bf16). The fold and merge-order mutants act on the bf16 preparation and are refused here.
    fn prepare(&mut self, mutant: Option<Mutant>, bad: &mut DevBuf, st: &Stream) -> Result<(), ModelError> {
        if self.format != VisionTowerFormat::Bf16 {
            if !self.conv.is_empty() {
                return Err(ModelError::Config(
                    "an EXL3 vision tower lists bf16 conversions; its tensors are fp16 as stored".into(),
                ));
            }
            if matches!(mutant, Some(Mutant::FoldSlot0Only | Mutant::MergeOrderConv | Mutant::MergeOrderTransposed)) {
                return Err(ModelError::Config(format!(
                    "the mutant `{}` acts on the bf16 tower's device preparation; the EXL3 tower's fold and permutation are the plan's",
                    mutant.map_or("", |m| m.name())
                )));
            }
            return Ok(());
        }
        for b in &mut self.conv {
            let n = b.bytes() / 2;
            gv::bf16_to_f16(b, n, bad, st)?;
        }
        // the folded patch weight: the bf16 source is copied aside, the fold writes over the front of its own slot
        let src_bytes = self.patch_w.bytes();
        let mut tmp = DevBuf::alloc(src_bytes)?;
        tmp.copy_from(&self.patch_w, src_bytes, st)?;
        let slots = if mutant == Some(Mutant::FoldSlot0Only) { 1 } else { 2 };
        gv::fold_patch(&tmp, &mut self.patch_w, HIDDEN, PATCH_KP, slots, st)?;
        // the downsample weight likewise (32 MiB of scratch, freed when the function returns)
        let ds_bytes = self.ds_w.bytes();
        let mut tmp2 = DevBuf::alloc(ds_bytes)?;
        tmp2.copy_from(&self.ds_w, ds_bytes, st)?;
        let mode = match mutant {
            Some(Mutant::MergeOrderConv) => 1,
            Some(Mutant::MergeOrderTransposed) => 2,
            _ => 0,
        };
        gv::permute_downsample(&tmp2, &mut self.ds_w, OUT_HIDDEN, HIDDEN, mode, bad, st)?;
        st.sync()?;
        let n: Vec<i32> = bad.to_vec()?;
        if n.first().copied().unwrap_or(1) != 0 {
            return Err(ModelError::Config(format!("{} vision weights are not finite in fp16", n[0])));
        }
        Ok(())
    }

    /// Every weight buffer, named, in a fixed order (what the tests compare or hash). The bf16 tower's names are those of the earlier releases;
    /// an EXL3 linear contributes its trellis, `suh` and `svh` (`blocks.N.q.trellis`, ...).
    fn named(&self) -> Vec<(String, &DevBuf)> {
        fn push<'a>(v: &mut Vec<(String, &'a DevBuf)>, name: &str, l: &'a Lin) {
            match l {
                Lin::Dense(b) => v.push((name.to_string(), b)),
                Lin::Exl3(w) => {
                    v.extend(w.buffers().into_iter().map(|(part, b)| (format!("{}.{part}", name.trim_end_matches("_w")), b)))
                }
            }
        }
        let mut v: Vec<(String, &DevBuf)> = vec![("patch_w".into(), &self.patch_w), ("patch_b".into(), &self.patch_b)];
        for (i, b) in self.blocks.iter().enumerate() {
            let p = format!("blocks.{i}");
            for (n, t) in [("norm1", &b.norm1), ("norm2", &b.norm2), ("q_norm", &b.q_norm), ("k_norm", &b.k_norm)] {
                v.push((format!("{p}.{n}"), t));
            }
            match b.qkv.as_slice() {
                [fused] => push(&mut v, &format!("{p}.qkv_w"), fused),
                split => {
                    for (n, l) in ["q", "k", "v"].iter().zip(split) {
                        push(&mut v, &format!("{p}.{n}"), l);
                    }
                }
            }
            v.push((format!("{p}.qkv_b"), &b.qkv_b));
            for (n, l, bias) in
                [("proj", &b.proj, &b.proj_b), ("gate", &b.gate, &b.gate_b), ("up", &b.up, &b.up_b), ("down", &b.down, &b.down_b)]
            {
                push(&mut v, &format!("{p}.{n}_w"), l);
                v.push((format!("{p}.{n}_b"), bias));
            }
        }
        v.push(("post_norm".into(), &self.post_norm));
        v.push(("ds_w".into(), &self.ds_w));
        v.push(("ds_b".into(), &self.ds_b));
        for (n, l) in [("m_proj", &self.m_proj), ("m_gate", &self.m_gate), ("m_up", &self.m_up), ("m_down", &self.m_down)] {
            push(&mut v, n, l);
        }
        v.push(("m_norm_w".into(), &self.m_norm_w));
        v.push(("m_norm_b".into(), &self.m_norm_b));
        v
    }
}

/// The staging of the EXL3 tower's linears (`scratch_items`' `exl3_*`): a linear's `W_q` is reconstructed once per call into `w`, then the rows
/// go through the EXL3 prefill pieces (`Exl3Weights::prefill_with_w`: rotate in, fp16 GEMM with fp32 accumulation, rotate out) in chunks of
/// `rows`, read from `x` when the input is strided and written through `y` when the output is (q | k | v and gate | up are interleaved per row).
struct Exl3Stage {
    w: DevBuf,
    x: DevBuf,
    xh: DevBuf,
    c: DevBuf,
    y: DevBuf,
    rows: usize,
}

impl Exl3Stage {
    /// `y[m, n]` at row pitch `ldc` = layer(`x[m, k]` at row pitch `lda`), fp16 in, fp16 or (with `f32_out`) fp32 out, no bias.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        w: &Exl3Weights,
        x: &DevBuf,
        lda: usize,
        y: &mut DevBuf,
        ldc: usize,
        m: usize,
        f32_out: bool,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let (k, n) = (w.k, w.n);
        let (ydt, ysz) = if f32_out { (Dtype::F32, 4) } else { (Dtype::F16, 2) };
        w.reconstruct(&mut self.w, st)?;
        // a contiguous block-backed input or output is used in place (a view of its rows); anything else goes through the staging
        let (x_direct, y_direct) = (lda == k && x.block().is_some(), ldc == n && y.block().is_some());
        let mut r0 = 0;
        while r0 < m {
            let rows = self.rows.min(m - r0);
            let staged;
            let xin = if x_direct {
                staged = x.subview(r0 * k * 2, rows * k * 2);
                &staged
            } else {
                self.x.copy_2d_from(0, k * 2, x, r0 * lda * 2, lda * 2, k * 2, rows, st)?;
                &self.x
            };
            if y_direct {
                let mut yv = y.subview(r0 * n * ysz, rows * n * ysz);
                w.prefill_with_w(xin, Dtype::F16, &self.w, &mut self.xh, &mut self.c, &mut yv, ydt, rows, st)?;
            } else {
                w.prefill_with_w(xin, Dtype::F16, &self.w, &mut self.xh, &mut self.c, &mut self.y, ydt, rows, st)?;
                y.copy_2d_from(r0 * ldc * ysz, ldc * ysz, &self.y, 0, n * ysz, n * ysz, rows, st)?;
            }
            r0 += rows;
        }
        Ok(())
    }
}

/// `bytes` of uninitialised device memory as a view of a block of its own (`DevBuf::subview` works on it).
fn block_buf(bytes: usize) -> Result<DevBuf, ModelError> {
    let b = std::sync::Arc::new(gb_cuda::rt::Block::alloc(bytes)?);
    Ok(DevBuf::view(&b, 0, bytes))
}

/// One linear of the tower: the dense fp16 GEMM (`gv::gemm`) or the staged EXL3 GEMM, `y[m, n]` (pitch `ldc`) from `x[m, k]` (pitch `lda`), no bias.
#[allow(clippy::too_many_arguments)]
fn forward(
    l: &Lin,
    x: &DevBuf,
    lda: usize,
    y: &mut DevBuf,
    ldc: usize,
    m: usize,
    k: usize,
    n: usize,
    f32_out: bool,
    ws: &mut DevBuf,
    ex: &mut Option<Exl3Stage>,
    st: &Stream,
) -> Result<(), ModelError> {
    match l {
        Lin::Dense(wt) => Ok(gv::gemm(x, lda, wt, y, ldc, m, k, n, f32_out, ws, st)?),
        Lin::Exl3(w) => {
            if (w.k, w.n) != (k, n) {
                return Err(ModelError::Config(format!(
                    "a vision EXL3 linear is {}x{}, the tower calls it as {k}x{n}",
                    w.k, w.n
                )));
            }
            let ex = ex.as_mut().ok_or_else(|| ModelError::Config("an EXL3 vision linear without its staging".into()))?;
            ex.run(w, x, lda, y, ldc, m, f32_out, st)
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------ the tower

/// What a tower is built with.
#[derive(Clone, Debug)]
pub struct TowerOptions {
    /// Most patches (before the 2x2 merge) one image may have: the scratch is sized for it (`4 x max_image_tokens`).
    pub max_patches: usize,
    /// Refuse to allocate the weights' region when it would leave less than this much `MemAvailable`.
    pub floor_bytes: u64,
    pub loader: LoaderOptions,
    /// Debug (the tests): a deliberate bug. Never set in serving.
    pub mutant: Option<Mutant>,
}

impl TowerOptions {
    pub fn new(max_patches: usize, floor_bytes: u64) -> Self {
        Self { max_patches, floor_bytes, loader: LoadOptions::from_env(floor_bytes).loader, mutant: None }
    }
}

struct Scratch {
    xin: DevBuf,
    x: DevBuf,
    h: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    tmp: DevBuf,
    gu: DevBuf,
    inv_freq: DevBuf,
    counter: DevBuf,
    ws: DevBuf,
    /// The EXL3 tower's staging (`None` for the bf16 tower).
    ex: Option<Exl3Stage>,
}

/// The debug hook of [`GlmVisionTower::teacher`].
pub type Teacher = Box<dyn FnMut(&str) -> Option<Vec<f32>> + Send>;

pub struct GlmVisionTower {
    regions: Regions,
    w: Weights,
    s: Scratch,
    pub max_patches: usize,
    /// The pack's storage of the tower (the scratch and the GEMMs follow it).
    pub format: VisionTowerFormat,
    mutant: Option<Mutant>,
    /// Debug (the tests): after the stage `name` (`block_BB`: the fp32 residual stream, `post_layernorm`: the fp16 normed stream) the stream is replaced by
    /// what `teacher(name)` returns (f32 `[P, 1024]`), so the next stage's error is that stage's own and not the accumulated one. Never set in serving.
    pub teacher: Option<Teacher>,
    /// Seconds of the last `load` (allocate, read and convert, prepare, scratch).
    pub boot: BootTimes,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BootTimes {
    pub alloc_s: f64,
    pub load_s: f64,
    pub prepare_s: f64,
    pub scratch_s: f64,
}

impl GlmVisionTower {
    /// Map nothing and read nothing but the tower: allocate the plan's `Vision` region, fill it through the pipelined loader, prepare the weights in
    /// place and allocate the scratch. `pack` is the mapped pack (`GlmFamily::map_pack`), `dir` its folder.
    pub fn load(pack: &GlmPack, dir: &Path, o: &TowerOptions) -> Result<Self, ModelError> {
        let cfg = pack.config.vision.as_ref().ok_or_else(|| ModelError::Config("the pack has no vision tower".into()))?;
        check_config(cfg)?;
        let weights =
            pack.weights.vision.as_ref().ok_or_else(|| ModelError::Config("the pack's weight map has no vision tower".into()))?;
        let plan = pack.plan.select_regions(|k| matches!(k, RegionKind::Vision));
        if weights.format != cfg.tower {
            return Err(ModelError::Config("the vision weights were mapped for another tower format than the config's".into()));
        }
        let region_bytes: u64 = plan.regions.iter().map(|r| r.bytes).sum();
        if region_bytes != weight_region_bytes(cfg) {
            return Err(ModelError::Config(format!(
                "the vision region is {region_bytes} bytes, the memory model says {}",
                weight_region_bytes(cfg)
            )));
        }
        gb_cuda::rt::init_context()?;
        let t = std::time::Instant::now();
        let regions = allocate(&plan, o.floor_bytes)?;
        let alloc_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        run_vision(dir, &plan, &regions, &o.loader)?;
        let load_s = t.elapsed().as_secs_f64();
        let mut tower = Self::from_regions(weights, regions, o)?;
        tower.boot.alloc_s = alloc_s;
        tower.boot.load_s = load_s;
        Ok(tower)
    }

    /// Bind the tower to regions the loader has filled, prepare the weights and allocate the scratch.
    pub fn from_regions(weights: &GlmVisionWeights, regions: Regions, o: &TowerOptions) -> Result<Self, ModelError> {
        if o.max_patches == 0 || o.max_patches % PATCHES_PER_TOKEN != 0 {
            return Err(ModelError::Config("max_patches must be a positive multiple of 4".into()));
        }
        let st = Stream::new()?;
        let mut w = Weights::bind(&regions, weights)?;
        let t = std::time::Instant::now();
        let mut counter = DevBuf::zeroed(4)?;
        w.prepare(o.mutant, &mut counter, &st)?;
        let prepare_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let format = weights.format;
        let items = scratch_items(o.max_patches, format);
        let sz = |name: &str| items.iter().find(|i| i.0 == name).map_or(0, |i| i.1 as usize);
        let mut s = Scratch {
            xin: DevBuf::zeroed(sz("patches"))?,
            x: DevBuf::alloc(sz("residual"))?,
            // block-backed (one cudaMalloc each, as before), so the EXL3 tower's staged GEMMs can take views of their rows and columns
            h: block_buf(sz("normed"))?,
            qkv: block_buf(sz("qkv"))?,
            att: block_buf(sz("attention"))?,
            tmp: block_buf(sz("sublayer"))?,
            // a view of a block of its own: the second GEMM of gate | up and of the merger write at an offset into it (`subview`)
            gu: {
                let b = std::sync::Arc::new(gb_cuda::rt::Block::alloc(sz("gate_up"))?);
                DevBuf::view(&b, 0, sz("gate_up"))
            },
            inv_freq: DevBuf::alloc(sz("inv_freq"))?,
            counter,
            ws: DevBuf::alloc(sz("gemm_workspace"))?,
            ex: match format {
                VisionTowerFormat::Bf16 => None,
                VisionTowerFormat::Exl3 { .. } => Some(Exl3Stage {
                    w: DevBuf::alloc(sz("exl3_w"))?,
                    x: DevBuf::alloc(sz("exl3_x"))?,
                    xh: DevBuf::alloc(sz("exl3_xh"))?,
                    c: DevBuf::alloc(sz("exl3_c"))?,
                    y: DevBuf::alloc(sz("exl3_y"))?,
                    rows: exl3_rows(o.max_patches),
                }),
            },
        };
        s.inv_freq.upload(&gb_vision::rope2d::inv_freq(HEAD_DIM, ROPE_THETA))?;
        let scratch_s = t.elapsed().as_secs_f64();
        Ok(Self {
            regions,
            w,
            s,
            max_patches: o.max_patches,
            format,
            mutant: o.mutant,
            teacher: None,
            boot: BootTimes { prepare_s, scratch_s, ..Default::default() },
        })
    }

    /// Bytes of the weights on the device (the region, padding included).
    pub fn weight_bytes(&self) -> u64 {
        self.regions.bytes()
    }

    /// Bytes of the scratch (`scratch_items`), sized for `max_patches`.
    pub fn scratch_bytes(&self) -> u64 {
        self.scratch_allocations().iter().map(|i| i.1).sum()
    }

    /// The scratch buffers as allocated, named like `scratch_items` (a test compares the two).
    pub fn scratch_allocations(&self) -> Vec<(&'static str, u64)> {
        let s = &self.s;
        vec![
            ("patches", s.xin.bytes() as u64),
            ("residual", s.x.bytes() as u64),
            ("normed", s.h.bytes() as u64),
            ("qkv", s.qkv.bytes() as u64),
            ("attention", s.att.bytes() as u64),
            ("sublayer", s.tmp.bytes() as u64),
            ("gate_up", s.gu.bytes() as u64),
            ("inv_freq", s.inv_freq.bytes() as u64),
            ("counter", s.counter.bytes() as u64),
            ("gemm_workspace", s.ws.bytes() as u64),
        ]
        .into_iter()
        .chain(s.ex.iter().flat_map(|e| {
            [
                ("exl3_w", e.w.bytes() as u64),
                ("exl3_x", e.x.bytes() as u64),
                ("exl3_xh", e.xh.bytes() as u64),
                ("exl3_c", e.c.bytes() as u64),
                ("exl3_y", e.y.bytes() as u64),
            ]
        }))
        .collect()
    }

    /// Every weight buffer as prepared, named in a fixed order.
    pub fn weight_buffers(&self) -> Vec<(String, &DevBuf)> {
        self.w.named()
    }

    /// Encode one image: `patches` are its still-image patches `[gh * gw, 588]` fp16 (`gb_vision::glm::still_patches`: the first temporal slot, the same
    /// values the fork's conv reads twice), `grid` its patch grid `(gh, gw)`, both even. `out` receives f32 `[gh * gw / 4, 4096]`. Runs on `st` without
    /// synchronising (the host-side upload of the patches reads from `patches` before this returns).
    pub fn encode(&mut self, patches: &[u16], grid: (usize, usize), out: &mut DevBuf, st: &Stream) -> Result<(), ModelError> {
        self.encode_observed(patches, grid, out, st, &mut |_, _, _| {})
    }

    /// [`GlmVisionTower::encode`] that shows intermediate buffers to `obs(name, buffer, stream)` as they are produced (the observer must synchronise
    /// `stream` before reading). Names: `patch_embed` (f32 `[P, 1024]`), `block_BB` (the fp32 residual stream after block BB), `post_layernorm`
    /// (f16 `[P, 1024]`), `downsample` (f16 `[N, 4096]`), `m_proj`, `m_norm`, `m_gelu` (f16 `[N, 4096]`), `m_act` (f16 `[N, 20480]`, the product in the
    /// first 10,240 columns), `features` (f32 `[N, 4096]`), and for every block BB: `dBB_norm1_out`, `dBB_qkv_raw` (the GEMM output, no bias, f16
    /// `[P, 3072]`), `dBB_qkv` (q and k after norm and rope, v, f16 `[P, 3072]`), `dBB_attn_core` (f16 `[P, 1024]`), `dBB_attn_out_raw` (the output
    /// projection's GEMM output, no bias), `dBB_norm2_out`, `dBB_gu_raw` (f16 `[P, 8192]`), `dBB_mlp_act` (the gate columns of `[P, 8192]` after the
    /// activation) and `dBB_mlp_out_raw`.
    pub fn encode_observed(
        &mut self,
        patches: &[u16],
        grid: (usize, usize),
        out: &mut DevBuf,
        st: &Stream,
        obs: &mut dyn FnMut(&str, &DevBuf, &Stream),
    ) -> Result<(), ModelError> {
        let (gh, gw) = grid;
        let p = gh.checked_mul(gw).unwrap_or(0);
        if p == 0 || p > self.max_patches || gh % MERGE != 0 || gw % MERGE != 0 {
            return Err(ModelError::Request(format!(
                "image of {gh}x{gw} patches; the tower takes even grids of up to {} patches",
                self.max_patches
            )));
        }
        if patches.len() != p * PATCH_K {
            return Err(ModelError::Request(format!("{} patch values for a {gh}x{gw} grid of {PATCH_K}", patches.len())));
        }
        let tokens = p / PATCHES_PER_TOKEN;
        if out.bytes() < tokens * OUT_HIDDEN * 4 {
            return Err(ModelError::Request("output buffer too small for the image's features".into()));
        }
        let m = self.mutant;
        let qkv_flags = match m {
            Some(Mutant::RopeAxisSwap) => gv::qkv::SWAP_AXES,
            Some(Mutant::NoRope) => gv::qkv::NO_ROPE,
            Some(Mutant::NoHeadNorm) => gv::qkv::NO_HEAD_NORM,
            Some(Mutant::NoQkvBias) => gv::qkv::NO_BIAS,
            _ => 0,
        };
        let mask = if m == Some(Mutant::AttnCausal) { gv::mask::CAUSAL } else { gv::mask::NONE };
        let rms_eps = if m == Some(Mutant::RmsEps1e6) { 1e-6 } else { RMS_EPS };
        let ln_eps = if m == Some(Mutant::LnEps1e5) { 1e-5 } else { LN_EPS };
        let mlp_flags = match m {
            Some(Mutant::BlockNoClamp) => 1,
            Some(Mutant::BlockClampBeforeSilu) => 2,
            Some(Mutant::NoMlpBias) => 4,
            _ => 0,
        };
        let merger_flags = if m == Some(Mutant::MergerNoClamp) { 1 } else { 0 };
        let gelu_flags = if m == Some(Mutant::GeluTanh) { 1 } else { 0 };

        let Self { w, s, teacher, .. } = self;
        let Scratch { xin, x, h, qkv, att, tmp, gu, inv_freq, ws, ex, .. } = s;
        // the patches into the GEMM's padded rows
        gv::upload_rows(xin, PATCH_KP * 2, patches, PATCH_K * 2, PATCH_K * 2, p, st)?;
        // patch embedding: fp16 GEMM into fp32, the fp16 bias added in fp32 (the fork's `hgemm` into a float tensor, then `y += bias`)
        gv::gemm(xin, PATCH_KP, &w.patch_w, x, HIDDEN, p, PATCH_KP, HIDDEN, true, ws, st)?;
        if m != Some(Mutant::NoPatchBias) {
            gv::bias_add_f32(x, &w.patch_b, HIDDEN, p, st)?;
        }
        obs("patch_embed", x, st);
        gv::rmsnorm(x, None, None, &w.blocks[0].norm1, h, p, rms_eps, st)?;
        // the second GEMM of gate | up writes at an offset into the same buffer
        let gu_total = gu.bytes();
        let mut gu_up = gu.subview(MLP * 2, gu_total - MLP * 2);
        for bi in 0..DEPTH {
            let b = &w.blocks[bi];
            let tag = |n: &str| format!("d{bi:02}_{n}");
            obs(&tag("norm1_out"), h, st);
            match b.qkv.as_slice() {
                [fused] => forward(fused, h, HIDDEN, qkv, 3 * HIDDEN, p, HIDDEN, 3 * HIDDEN, false, ws, ex, st)?,
                // q, k and v into their columns of the fused row, as the fused GEMM writes them
                split => {
                    for (j, l) in split.iter().enumerate() {
                        let mut col = qkv.subview(j * HIDDEN * 2, qkv.bytes() - j * HIDDEN * 2);
                        forward(l, h, HIDDEN, &mut col, 3 * HIDDEN, p, HIDDEN, HIDDEN, false, ws, ex, st)?;
                    }
                }
            }
            obs(&tag("qkv_raw"), qkv, st);
            gv::qkv_prep(qkv, &b.qkv_b, &b.q_norm, &b.k_norm, inv_freq, p, gw, rms_eps, qkv_flags, st)?;
            obs(&tag("qkv"), qkv, st);
            gv::attention(qkv, att, p, mask, st)?;
            obs(&tag("attn_core"), att, st);
            forward(&b.proj, att, HIDDEN, tmp, HIDDEN, p, HIDDEN, HIDDEN, false, ws, ex, st)?;
            obs(&tag("attn_out_raw"), tmp, st);
            // x += f16(tmp + bias); h = RMSNorm(x) * norm2
            let proj_bias = if m == Some(Mutant::NoProjBias) { None } else { Some(&b.proj_b) };
            gv::rmsnorm(x, Some(tmp), proj_bias, &b.norm2, h, p, rms_eps, st)?;
            obs(&tag("norm2_out"), h, st);
            forward(&b.gate, h, HIDDEN, gu, 2 * MLP, p, HIDDEN, MLP, false, ws, ex, st)?;
            forward(&b.up, h, HIDDEN, &mut gu_up, 2 * MLP, p, HIDDEN, MLP, false, ws, ex, st)?;
            obs(&tag("gu_raw"), gu, st);
            gv::swiglu_block(gu, &b.gate_b, &b.up_b, MLP, 2 * MLP, SWIGLU_LIMIT, mlp_flags, p, st)?;
            obs(&tag("mlp_act"), gu, st);
            forward(&b.down, gu, 2 * MLP, tmp, HIDDEN, p, MLP, HIDDEN, false, ws, ex, st)?;
            obs(&tag("mlp_out_raw"), tmp, st);
            // x += f16(tmp + bias); h = the next block's norm1 of the new stream (or the post norm after the last block)
            let down_bias = if m == Some(Mutant::NoDownBias) { None } else { Some(&b.down_b) };
            let next_norm = if bi + 1 < DEPTH { &w.blocks[bi + 1].norm1 } else { &w.post_norm };
            gv::rmsnorm(x, Some(tmp), down_bias, next_norm, h, p, rms_eps, st)?;
            obs(&format!("block_{bi:02}"), x, st);
            if let Some(force) = teacher.as_mut() {
                if let Some(xr) = force(&format!("block_{bi:02}")) {
                    st.sync()?;
                    x.upload(&xr)?;
                    gv::rmsnorm(x, None, None, next_norm, h, p, rms_eps, st)?;
                }
            }
        }
        obs("post_layernorm", h, st);
        if let Some(force) = teacher.as_mut() {
            if let Some(hr) = force("post_layernorm") {
                st.sync()?;
                let bits: Vec<u16> = hr.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
                h.upload(&bits)?;
            }
        }
        // downsample: the four patches of a merge block are consecutive rows of `h`, so [P, 1024] is [N, 4096] in place
        gv::gemm(h, 4 * HIDDEN, &w.ds_w, att, OUT_HIDDEN, tokens, 4 * HIDDEN, OUT_HIDDEN, false, ws, st)?;
        if m != Some(Mutant::NoDownsampleBias) {
            gv::bias_add_f16(att, &w.ds_b, OUT_HIDDEN, tokens, st)?;
        }
        obs("downsample", att, st);
        // merger: proj (no bias) -> LayerNorm -> GELU -> gate, up -> clamp, silu, product -> down (fp32 out)
        forward(&w.m_proj, att, OUT_HIDDEN, tmp, OUT_HIDDEN, tokens, OUT_HIDDEN, OUT_HIDDEN, false, ws, ex, st)?;
        obs("m_proj", tmp, st);
        gv::layernorm(tmp, &w.m_norm_w, &w.m_norm_b, att, tokens, OUT_HIDDEN, ln_eps, st)?;
        obs("m_norm", att, st);
        gv::gelu(att, tokens * OUT_HIDDEN, gelu_flags, st)?;
        obs("m_gelu", att, st);
        let mut gu_up_m = gu.subview(MERGER_MLP * 2, gu_total - MERGER_MLP * 2);
        forward(&w.m_gate, att, OUT_HIDDEN, gu, 2 * MERGER_MLP, tokens, OUT_HIDDEN, MERGER_MLP, false, ws, ex, st)?;
        forward(&w.m_up, att, OUT_HIDDEN, &mut gu_up_m, 2 * MERGER_MLP, tokens, OUT_HIDDEN, MERGER_MLP, false, ws, ex, st)?;
        gv::swiglu_merger(gu, MERGER_MLP, 2 * MERGER_MLP, SWIGLU_LIMIT, tokens, merger_flags, st)?;
        obs("m_act", gu, st);
        forward(&w.m_down, gu, 2 * MERGER_MLP, out, OUT_HIDDEN, tokens, MERGER_MLP, OUT_HIDDEN, true, ws, ex, st)?;
        obs("features", out, st);
        Ok(())
    }
}

/// The plan's items through the pipelined loader, charged to the vision component in the boot statistics (`load::run_as`): the bf16 tower's
/// tensors as stored (its conversions are made on the device afterwards), the EXL3 tower's groups as stored with their codebook markers checked,
/// and its fp16 patch and downsample weights through the plan's host fold and permutation.
fn run_vision(dir: &Path, plan: &LoadPlan, regions: &Regions, o: &LoaderOptions) -> Result<(), ModelError> {
    super::load::run_as(dir, plan, regions, o, Component::Vision)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GLM_CONFIG: &str = include_str!("../../../../../testdata/glm53/config.json");

    fn real() -> super::super::GlmConfig {
        super::super::GlmConfig::from_json(&serde_json::from_str(GLM_CONFIG).unwrap()).unwrap()
    }

    #[test]
    fn the_real_config_is_the_tower_the_kernels_are_written_for() {
        let v = real().vision.unwrap();
        check_config(&v).unwrap();
        assert_eq!(v.patch_dim(), 2 * PATCH_K);
        assert_eq!(v.hidden / v.heads, HEAD_DIM);
        for bad in [
            GlmVisionConfig { depth: 23, ..v.clone() },
            GlmVisionConfig { hidden: 1032, ..v.clone() },
            GlmVisionConfig { rms_eps: 1e-6, ..v.clone() },
            GlmVisionConfig { swiglu_limit: 7.0, ..v.clone() },
            GlmVisionConfig { attention_bias: false, ..v.clone() },
            GlmVisionConfig { projection_intermediate: 8192, ..v.clone() },
        ] {
            assert!(check_config(&bad).is_err());
        }
    }

    #[test]
    fn the_scratch_is_34_kib_a_patch_plus_the_workspace() {
        let per_patch: u64 = scratch_items(4, VisionTowerFormat::Bf16)
            .iter()
            .filter(|i| i.0 != "inv_freq" && i.0 != "counter" && i.0 != "gemm_workspace")
            .map(|i| i.1)
            .sum::<u64>()
            / 4;
        // patches 1,184 + residual 4,096 + normed 2,048 + qkv 6,144 + attention 2,048 + sublayer 2,048 + gate|up 16,384
        assert_eq!(per_patch, 33_952);
        assert_eq!(scratch_bytes(32_000, VisionTowerFormat::Bf16), 32_000 * 33_952 + 64 + 4 + GEMM_WORKSPACE as u64);
        // every merger buffer fits the bytes of the block-stage buffer it reuses
        let tokens = 8_000u64;
        let get = |n: &str| scratch_items(32_000, VisionTowerFormat::Bf16).iter().find(|i| i.0 == n).unwrap().1;
        assert_eq!(tokens * OUT_HIDDEN as u64 * 2, get("attention"), "downsample output is the attention buffer's bytes");
        assert!(tokens * 2 * MERGER_MLP as u64 * 2 <= get("gate_up"), "merger gate|up fits the block gate|up buffer");
        assert_eq!(tokens * OUT_HIDDEN as u64 * 2, get("sublayer"), "merger proj output is the sublayer buffer's bytes");
        assert_eq!(
            tokens * OUT_HIDDEN as u64 * 4,
            get("residual"),
            "the features are the residual stream's bytes (they are written to the caller's buffer)"
        );
    }

    #[test]
    fn the_weight_region_is_the_plan_s_arithmetic() {
        let v = real().vision.unwrap();
        // 347 tensors, 1,127,254,016 bytes of data; the 256-byte alignment of each adds a little
        let region = weight_region_bytes(&v);
        assert!(region >= v.weight_bytes() && region < v.weight_bytes() + 347 * 256, "{region} vs {}", v.weight_bytes());
    }

    /// The constants the mutation table varies (`RmsEps1e6`, `LnEps1e5`, ...), pinned to what the fork uses: the blocks' RMSNorm eps is the config's
    /// `rms_norm_eps` (checked against the pack by `check_config`), the merger's LayerNorm eps is the literal 1e-6 of `Glm4VVisionPatchMerger`, the SwiGLU
    /// limit the config's 10, the rope's theta the fork's `v.rope_theta = 10000`, and the patch GEMM's padded width a multiple of 8 that holds the 588.
    #[test]
    fn the_tower_constants_are_the_forks() {
        assert_eq!(RMS_EPS, 1e-5f32);
        assert_eq!(LN_EPS, 1e-6f32);
        assert_eq!(SWIGLU_LIMIT, 10.0f32);
        assert_eq!(ROPE_THETA, 10000.0f32);
        const { assert!(PATCH_KP % 8 == 0 && PATCH_KP >= PATCH_K && PATCH_KP - PATCH_K < 8) };
        assert_eq!(PATCH_K, 588);
        assert_eq!(MERGE * MERGE, PATCHES_PER_TOKEN);
        let v = real().vision.unwrap();
        assert_eq!(v.rms_eps as f32, RMS_EPS);
    }

    #[test]
    fn mutants_have_distinct_names_and_all_are_listed() {
        let mut names: Vec<&str> = Mutant::ALL.iter().map(|m| m.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), Mutant::ALL.len());
    }
}
