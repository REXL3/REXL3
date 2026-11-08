//! The typed description of every tensor `BailingMoeV3ForCausalLM`'s pack holds, and where each one goes.
//!
//! [`BailingWeights`] is a tree with one leaf per tensor or EXL3 group. Every leaf records what the pack says (dtype, shape,
//! bits, the shard and offset the bytes are read from) and where the load puts them (a [`Dst`]: a region of device memory and an
//! offset in it). It is built by one pure function over a [`PackInventory`]: [`BailingWeights::map`], in `bind.rs`.
//!
//! Bits are read from the trellis shapes (`Group::bits`), never from `quantization_config.json`; half-integer widths (1.5, 2.5,
//! 3.5) are ordinary here, the pack is mixed-K. Bytes are uploaded verbatim (the pack's fp16 scales, bf16/fp16/fp32 tensors and
//! int16 trellises are what the kernels read), so a group's device footprint is its trellis, `suh` and `svh`; the codebook marker
//! (a 4-byte int32 the scan has already verified) is recorded but not uploaded.

use super::plan::Region;
use gb_exl3::budget::Category;
use gb_exl3::{Bits, Codebook};

/// A region of device memory the plan fills: an index into [`BailingWeights::regions`] (and `LoadPlan::regions`).
pub type RegionId = u32;

/// Every destination offset is a multiple of this.
pub const ALIGN: u64 = 256;

pub fn align_up(x: u64) -> u64 {
    x.div_ceil(ALIGN) * ALIGN
}

/// Dtypes of the plain tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlainDtype {
    F16,
    Bf16,
    F32,
}

impl PlainDtype {
    pub fn name(self) -> &'static str {
        match self {
            PlainDtype::F16 => "F16",
            PlainDtype::Bf16 => "BF16",
            PlainDtype::F32 => "F32",
        }
    }
    pub fn size(self) -> u64 {
        match self {
            PlainDtype::F16 | PlainDtype::Bf16 => 2,
            PlainDtype::F32 => 4,
        }
    }
}

/// Where a tensor's bytes are read from: a shard (an index into [`BailingWeights::files`]) and an absolute offset in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Src {
    pub file: u16,
    pub offset: u64,
}

/// Where a tensor's bytes go: an offset into a region of device memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dst {
    pub region: RegionId,
    pub offset: u64,
}

impl Dst {
    pub const NONE: Dst = Dst { region: RegionId::MAX, offset: 0 };
}

/// What the loader does to a plain tensor's bytes between the shard and the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Xform {
    /// The file's bytes, unchanged.
    None,
    /// bf16 widened to fp32 (an RMSNorm gamma).
    Bf16ToF32,
    /// `kv_b_proj` fp16 `[heads * (nope + v), kv_lora_rank]` split per head and packed into `W_UK` and transposed `W_UV` for the MLA
    /// kernels (`gb_cuda::mla::pack_kv_b`): the same number of bytes, `uk` first, then `uvt`.
    PackKvB,
    /// The three KDA conv1d weights `q_conv1d.weight`, `k_conv1d.weight`, `v_conv1d.weight` (`[8192, 1, 4]` bf16 each) claimed
    /// contiguously into a single `[24576, 1, 4]` device slot as the KDA prep reads them (`[3 * channels, 1, conv]`).
    CatQkvConv,
}

impl Xform {
    pub fn dst_bytes(self, src_bytes: u64) -> u64 {
        match self {
            Xform::Bf16ToF32 => src_bytes * 2,
            Xform::None | Xform::PackKvB | Xform::CatQkvConv => src_bytes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlainTensor {
    pub name: String,
    pub dtype: PlainDtype,
    pub shape: Vec<usize>,
    /// Bytes in the shard.
    pub bytes: u64,
    pub xform: Xform,
    /// Bytes on the device (`xform.dst_bytes(bytes)`).
    pub dst_bytes: u64,
    pub src: Src,
    pub dst: Dst,
    pub category: Category,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exl3Part {
    pub bytes: u64,
    pub src: Src,
    pub dst: Dst,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Exl3Linear {
    /// The tensor name without `.trellis`.
    pub prefix: String,
    pub k: usize,
    pub n: usize,
    pub bits: Bits,
    pub codebook: Codebook,
    pub trellis: Exl3Part,
    pub suh: Exl3Part,
    pub svh: Exl3Part,
    /// The codebook marker: read by the scan, not uploaded.
    pub marker: Option<(Src, u64)>,
    pub category: Category,
}

impl Exl3Linear {
    pub fn weight_bytes(&self) -> u64 {
        self.trellis.bytes + self.suh.bytes + self.svh.bytes
    }
}

/// KDA linear attention (63 trunk layers).
#[derive(Clone, Debug, PartialEq)]
pub struct KdaWeights {
    /// `[hidden -> 3 x heads x head_dim]`: q, k and v fused.
    pub qkv_proj: Exl3Linear,
    /// `[hidden -> heads x head_dim]`.
    pub f_proj: Exl3Linear,
    /// `[hidden -> heads x head_dim]`.
    pub g_proj: Exl3Linear,
    /// `[heads x head_dim -> hidden]`.
    pub o_proj: Exl3Linear,
    /// bf16 `[heads, hidden]`.
    pub b_proj: PlainTensor,
    /// fp32 `[heads]`.
    pub a_log: PlainTensor,
    /// fp32 `[heads x head_dim]`.
    pub dt_bias: PlainTensor,
    /// The three conv1d weights fused into `[3 x channels, 1, conv]` (`CatQkvConv`).
    pub conv1d_weight: CatQkvConv,
    /// bf16 `[head_dim]`.
    pub o_norm: PlainTensor,
}

/// Gated MLA (9 trunk layers + the MTP layer).
#[derive(Clone, Debug, PartialEq)]
pub struct MlaWeights {
    /// `[hidden -> q_lora_rank]`.
    pub q_a_proj: Exl3Linear,
    /// bf16 `[q_lora_rank]`.
    pub q_a_layernorm: PlainTensor,
    /// `[q_lora_rank -> heads x qk_head_dim]`.
    pub q_b_proj: Exl3Linear,
    /// `[hidden -> kv_lora_rank + qk_rope_head_dim]` padded to a multiple of 128 (`kv_a_group_n`).
    pub kv_a_proj_with_mqa: Exl3Linear,
    /// bf16 `[kv_lora_rank]`.
    pub kv_a_layernorm: PlainTensor,
    /// fp16 `[heads x (qk_nope_head_dim + v_head_dim), kv_lora_rank]` (packed into `W_UK` / `W_UV`).
    pub kv_b_proj: PlainTensor,
    /// bf16 `[heads, hidden]` head-wise sigmoid gate.
    pub g_proj: PlainTensor,
    /// `[heads x v_head_dim -> hidden]`.
    pub dense: Exl3Linear,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MixerWeights {
    Kda(Box<KdaWeights>),
    Mla(Box<MlaWeights>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct GatedMlp {
    pub gate_proj: Exl3Linear,
    pub up_proj: Exl3Linear,
    pub down_proj: Exl3Linear,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertWeights {
    pub gate_proj: Exl3Linear,
    pub up_proj: Exl3Linear,
    pub down_proj: Exl3Linear,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProjLayout {
    pub trellis: u64,
    pub trellis_stride: u64,
    pub suh: u64,
    pub suh_stride: u64,
    pub svh: u64,
    pub svh_stride: u64,
}

impl ProjLayout {
    pub fn trellis_at(&self, expert: usize) -> u64 {
        self.trellis + expert as u64 * self.trellis_stride
    }
    pub fn suh_at(&self, expert: usize) -> u64 {
        self.suh + expert as u64 * self.suh_stride
    }
    pub fn svh_at(&self, expert: usize) -> u64 {
        self.svh + expert as u64 * self.svh_stride
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertLayout {
    pub gate: ProjLayout,
    pub up: ProjLayout,
    pub down: ProjLayout,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertSubbank {
    pub key: gb_cuda::moe::SubbankKey,
    pub ids: Vec<u32>,
    pub layout: ExpertLayout,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertBank {
    pub layer: usize,
    pub region: RegionId,
    pub hidden: usize,
    pub inter: usize,
    pub experts: Vec<ExpertWeights>,
    pub layout: ExpertLayout,
    pub subbanks: Vec<ExpertSubbank>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeWeights {
    /// `mlp.gate.weight` (bf16 `[experts, hidden]`).
    pub router: PlainTensor,
    /// `mlp.gate.expert_bias` (fp32 `[experts]`).
    pub expert_bias: PlainTensor,
    pub experts: ExpertBank,
    /// `mlp.shared_experts.*`; `None` when the layer has none (never here: every MoE layer has one).
    pub shared: Option<GatedMlp>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FfnWeights {
    Dense(Box<GatedMlp>),
    Moe(Box<MoeWeights>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct BailingLayerWeights {
    pub index: usize,
    pub input_layernorm: PlainTensor,
    pub post_attention_layernorm: PlainTensor,
    pub mixer: MixerWeights,
    pub ffn: FfnWeights,
}

/// The NextN/MTP layer (72): `eh_proj` fuses the embedded token and the trunk's hidden, then a Gated MLA + MoE layer on a plain
/// residual, then `final_layernorm` and the trunk's `lm_head`.
#[derive(Clone, Debug, PartialEq)]
pub struct BailingMtpWeights {
    /// Plain bf16 `[hidden, 2 x hidden]`.
    pub eh_proj: PlainTensor,
    pub enorm: PlainTensor,
    pub hnorm: PlainTensor,
    pub final_layernorm: PlainTensor,
    pub layer: BailingLayerWeights,
}

/// Every tensor of the pack, typed.
#[derive(Clone, Debug, PartialEq)]
pub struct BailingWeights {
    /// The shard files, as in the inventory; a [`Src::file`] indexes this.
    pub files: Vec<String>,
    /// The device regions the leaves are placed in (sizes include alignment padding).
    pub regions: Vec<Region>,
    /// bf16 `[vocab, hidden]`, mmap'd from shard 1 (`Placement::Sidecar`), never resident.
    pub embed_tokens: PlainTensor,
    /// bf16 `[hidden]`.
    pub final_norm: PlainTensor,
    /// `[hidden -> vocab]`.
    pub lm_head: Exl3Linear,
    /// The trunk.
    pub layers: Vec<BailingLayerWeights>,
    /// The NextN layer (none or one).
    pub mtp: Vec<BailingMtpWeights>,
}

/// The fused KDA conv1d: three source tensors `q/k/v_conv1d.weight` (`[channels, 1, conv]` bf16) claimed from the pack and
/// placed contiguously into one `[3 * channels, 1, conv]` device slot as the KDA prep reads them. The `parts` are claimed and
/// checked but never placed; `fused` is the single device allocation.
#[derive(Clone, Debug, PartialEq)]
pub struct CatQkvConv {
    pub parts: [PlainTensor; 3],
    pub fused: PlainTensor,
}

/// One leaf of the tree, as the plan builder and the totals see it.
#[derive(Clone, Copy, Debug)]
pub enum Leaf<'a> {
    Plain(&'a PlainTensor),
    Exl3(&'a Exl3Linear),
    CatQkvConv(&'a CatQkvConv),
}

impl BailingWeights {
    /// Visit every leaf (a plain tensor, an EXL3 group, or a fused conv1d) exactly once.
    pub fn visit(&self, f: &mut impl FnMut(Leaf<'_>)) {
        f(Leaf::Plain(&self.embed_tokens));
        f(Leaf::Plain(&self.final_norm));
        f(Leaf::Exl3(&self.lm_head));
        for l in &self.layers {
            let _ = l.visit(f);
        }
        for m in &self.mtp {
            f(Leaf::Plain(&m.eh_proj));
            f(Leaf::Plain(&m.enorm));
            f(Leaf::Plain(&m.hnorm));
            f(Leaf::Plain(&m.final_layernorm));
            let _ = m.layer.visit(f);
        }
    }
}

impl BailingLayerWeights {
    fn visit<'a>(&'a self, f: &mut impl FnMut(Leaf<'a>)) {
        f(Leaf::Plain(&self.input_layernorm));
        f(Leaf::Plain(&self.post_attention_layernorm));
        match &self.mixer {
            MixerWeights::Kda(k) => {
                f(Leaf::Exl3(&k.qkv_proj));
                f(Leaf::Exl3(&k.f_proj));
                f(Leaf::Exl3(&k.g_proj));
                f(Leaf::Exl3(&k.o_proj));
                f(Leaf::Plain(&k.b_proj));
                f(Leaf::Plain(&k.a_log));
                f(Leaf::Plain(&k.dt_bias));
                f(Leaf::CatQkvConv(&k.conv1d_weight));
                f(Leaf::Plain(&k.o_norm));
            }
            MixerWeights::Mla(m) => {
                f(Leaf::Exl3(&m.q_a_proj));
                f(Leaf::Plain(&m.q_a_layernorm));
                f(Leaf::Exl3(&m.q_b_proj));
                f(Leaf::Exl3(&m.kv_a_proj_with_mqa));
                f(Leaf::Plain(&m.kv_a_layernorm));
                f(Leaf::Plain(&m.kv_b_proj));
                f(Leaf::Plain(&m.g_proj));
                f(Leaf::Exl3(&m.dense));
            }
        }
        match &self.ffn {
            FfnWeights::Dense(g) => {
                f(Leaf::Exl3(&g.gate_proj));
                f(Leaf::Exl3(&g.up_proj));
                f(Leaf::Exl3(&g.down_proj));
            }
            FfnWeights::Moe(m) => {
                f(Leaf::Plain(&m.router));
                f(Leaf::Plain(&m.expert_bias));
                for e in &m.experts.experts {
                    f(Leaf::Exl3(&e.gate_proj));
                    f(Leaf::Exl3(&e.up_proj));
                    f(Leaf::Exl3(&e.down_proj));
                }
                if let Some(s) = &m.shared {
                    f(Leaf::Exl3(&s.gate_proj));
                    f(Leaf::Exl3(&s.up_proj));
                    f(Leaf::Exl3(&s.down_proj));
                }
            }
        }
    }

    /// Routed expert groups of the layer's MoE (for accounting).
    pub fn ffn_moe_groups(&self) -> usize {
        match &self.ffn {
            FfnWeights::Moe(m) => m.experts.experts.len() * 3 + if m.shared.is_some() { 3 } else { 0 },
            _ => 0,
        }
    }
}
