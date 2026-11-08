//! The typed description of every tensor DeepSeek-V4.1-Flash's pack holds: what each one is, and the dtype and shape it must
//! have.
//!
//! [`DeepseekV41Weights`] is a tree with one leaf per tensor: the embedding, the head and the final norm; the 40 trunk layers
//! (two norms, an MLA mixer with its five FP8 projections and their block scales, the router and its two biases, the shared
//! expert, 384 routed EXL3 experts, the mHC hyper-connections, and — on the layers that own one — a compressor, a DSA indexer
//! and an engram table); the 3 NextN layers with their int8 experts, their `main_proj` tap and the DSpark Markov and
//! confidence heads; and the vision tower with the aligner that splices its features into the text stream.
//!
//! Three things this file is deliberately *not*:
//!
//! * not a loader: there is no shard, offset or device address here, only the dtype and shape a tensor must have. Placement is
//!   the load plan's job, and it starts from this tree;
//! * not a policy: every shape is derived from [`super::config::DeepseekV41Config`] by [`super::bind::bind`], never stated here
//!   as a literal. The only two numbers that are not config fields are named and justified where they live
//!   ([`ENGRAM_WKV_ROW_FACTOR`], [`HC_COEFFS_PER_STREAM`], [`HC_SCALE`], [`MAIN_PROJ_TAPS`]);
//! * not a materialisation: the two engram embedding tables are ~101.5 GB each. They are described by [`MmapTable`], a pair of
//!   tensors the forward addresses through an `mmap` of the shard, and no code path in this family ever builds a `Vec` of their
//!   contents. The type says so, so the plan that consumes it can too.
//!
//! Bit widths are a property of the *expert*, not of the layer: the pack's 46,080 routed experts carry trellises from 1 to 6
//! bits, and 16 of the 32 possible (w1, w2, w3) width triples occur. [`Exl3Linear::bits`] is therefore per projection and
//! [`ExpertBank::widths`] reports the histogram rather than a single number.

/// Coefficients the mHC hyper-connections keep per residual stream (`hc_*_base` is `[6 x mult]`, `hc_*_fn` is
/// `[6 x mult, mult x hidden]`).
pub const HC_COEFFS_PER_STREAM: usize = 6;
/// Coefficients of the mHC scale vector (`hc_*_scale` is `[3]`).
pub const HC_SCALE: usize = 3;
/// Hidden rows the first NextN layer's `main_proj` fuses: three trunk-layer taps, not `hc_mult`'s four.
pub const MAIN_PROJ_TAPS: usize = 3;
/// The engram's fused key/value projection is `[5 x hidden, 3 x n_heads x head_dim]`. Neither the five nor the three is a
/// `config.json` key — `[25600, 6144]` is simply what both engram layers hold — so the two factors are named here.
pub const ENGRAM_WKV_ROW_FACTOR: usize = 5;
pub const ENGRAM_WKV_COL_FACTOR: usize = 3;

/// A safetensors dtype, spelled as the pack's headers spell it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
    I16,
    I32,
    I8,
    /// `F8_E4M3`: the FP8 weight of every dense projection, and the engram tables.
    Fp8,
    /// `F8_E8M0`: the 32x32 block scale that goes with an [`Dtype::Fp8`] weight, and the per-32-input scale of an int8 expert.
    E8m0,
}

impl Dtype {
    /// The header's spelling.
    pub fn name(self) -> &'static str {
        match self {
            Dtype::Bf16 => "BF16",
            Dtype::F16 => "F16",
            Dtype::F32 => "F32",
            Dtype::I16 => "I16",
            Dtype::I32 => "I32",
            Dtype::I8 => "I8",
            Dtype::Fp8 => "F8_E4M3",
            Dtype::E8m0 => "F8_E8M0",
        }
    }

    /// Bytes one element occupies.
    pub fn bytes(self) -> u64 {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::Bf16 | Dtype::F16 | Dtype::I16 => 2,
            Dtype::Fp8 | Dtype::E8m0 | Dtype::I8 => 1,
        }
    }

    /// The dtype a header names, when this family knows it.
    pub fn parse(s: &str) -> Option<Dtype> {
        Some(match s {
            "BF16" => Dtype::Bf16,
            "F16" => Dtype::F16,
            "F32" => Dtype::F32,
            "I16" => Dtype::I16,
            "I32" => Dtype::I32,
            "I8" => Dtype::I8,
            "F8_E4M3" => Dtype::Fp8,
            "F8_E8M0" => Dtype::E8m0,
            _ => return None,
        })
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// One tensor of the pack: its name, and the dtype and shape it must have.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(name: impl Into<String>, dtype: Dtype, shape: impl Into<Vec<usize>>) -> Self {
        Self { name: name.into(), dtype, shape: shape.into() }
    }

    /// Elements, as the header states them.
    pub fn numel(&self) -> u64 {
        self.shape.iter().product::<usize>() as u64
    }

    /// Bytes in the shard.
    pub fn bytes(&self) -> u64 {
        self.numel() * self.dtype.bytes()
    }
}

/// An FP8 projection: an `F8_E4M3` weight `[n, k]` and its `F8_E8M0` 32x32 block scales `[n / 32, k / 32]`
/// (or an EXL3 group `k x n` when quantized in place).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fp8Linear {
    pub weight: Tensor,
    pub scale: Tensor,
    pub exl3: Option<Exl3Linear>,
}

impl Fp8Linear {
    /// The reduction width.
    pub fn k(&self) -> usize {
        self.weight.shape[1]
    }
    /// The output width.
    pub fn n(&self) -> usize {
        self.weight.shape[0]
    }
    pub fn bytes(&self) -> u64 {
        if let Some(g) = &self.exl3 {
            g.bytes()
        } else {
            self.weight.bytes() + self.scale.bytes()
        }
    }
    pub(crate) fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        if let Some(g) = &self.exl3 {
            g.visit(f);
        } else {
            f(&self.weight);
            f(&self.scale);
        }
    }
}

/// An EXL3 routed expert projection: an `I16` trellis `[k / 16, n / 16, 16 x bits]`, a scalar `I32` `mul1` codebook marker,
/// and the `F16` scales `suh` `[k]` and `svh` `[n]`.
///
/// `k` and `n` are carried per projection rather than derived from a layer: the pack stores them per expert, and a future pack
/// whose experts differ in width is described here without a change to the type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exl3Linear {
    /// The tensor name without `.trellis`.
    pub prefix: String,
    pub k: usize,
    pub n: usize,
    /// `trellis.shape[2] / 16`, in `1..=8`.
    pub bits: u8,
    pub trellis: Tensor,
    pub mul1: Tensor,
    pub suh: Tensor,
    pub svh: Tensor,
}

impl Exl3Linear {
    /// Bytes the kernels read: the trellis, the two scale vectors and the marker.
    pub fn bytes(&self) -> u64 {
        self.trellis.bytes() + self.mul1.bytes() + self.suh.bytes() + self.svh.bytes()
    }
}

/// One routed expert: gate, down and up, each an EXL3 group of its own bit width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expert {
    pub w1: Exl3Linear,
    pub w2: Exl3Linear,
    pub w3: Exl3Linear,
}

impl Expert {
    pub fn projections(&self) -> [&Exl3Linear; 3] {
        [&self.w1, &self.w2, &self.w3]
    }

    /// Where the twelve parts of this expert sit in its record, and the record's length — the one walk every placement of an
    /// expert uses (the resident slab, the streamed cache record, the streamed slot's views). Per projection (`w1`, `w2`, `w3`):
    /// `trellis`, `mul1`, `suh`, `svh`, each starting on a [`gb_cuda::expert_cache::EXPERT_PART_ALIGN`] boundary; the record is
    /// rounded up to the same boundary, so records placed back to back stay aligned too. `offsets[p][q]` is projection `p`'s part
    /// `q` in that order.
    pub fn record_layout(&self) -> ([[u64; 4]; 3], u64) {
        let a = gb_cuda::expert_cache::EXPERT_PART_ALIGN;
        let mut offsets = [[0u64; 4]; 3];
        let mut off = 0u64;
        for (p, g) in self.projections().into_iter().enumerate() {
            for (q, t) in [&g.trellis, &g.mul1, &g.suh, &g.svh].into_iter().enumerate() {
                off = off.div_ceil(a) * a;
                offsets[p][q] = off;
                off += t.bytes();
            }
        }
        (offsets, off.div_ceil(a) * a)
    }

    /// The record's length: the parts plus their alignment padding ([`Expert::record_layout`]).
    pub fn record_bytes(&self) -> u64 {
        self.record_layout().1
    }
}

/// A layer's routed experts, in expert-id order (`experts[e]` is global id `e`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertBank {
    pub layer: usize,
    pub experts: Vec<Expert>,
}

impl ExpertBank {
    pub fn len(&self) -> usize {
        self.experts.len()
    }
    pub fn is_empty(&self) -> bool {
        self.experts.is_empty()
    }

    /// How many experts each bit width has, over all three projections, ascending by width.
    pub fn widths(&self) -> Vec<(u8, usize)> {
        let mut h = std::collections::BTreeMap::new();
        for e in &self.experts {
            for g in e.projections() {
                *h.entry(g.bits).or_insert(0usize) += 1;
            }
        }
        h.into_iter().collect()
    }

    /// The bit widths of one projection (`1` for `w1`, `2` for `w2`, `3` for `w3`), ascending.
    pub fn widths_of(&self, projection: usize) -> Vec<u8> {
        self.experts.iter().map(|e| e.projections()[projection].bits).collect()
    }

    pub fn bytes(&self) -> u64 {
        self.experts.iter().flat_map(|e| e.projections()).map(|g| g.bytes()).sum()
    }
}

/// An int8 4-bit expert projection (the NextN layers'): two values per byte, one `F8_E8M0` scale per 32 inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Int4Linear {
    pub prefix: String,
    pub k: usize,
    pub n: usize,
    pub weight: Tensor,
    pub scale: Tensor,
}

impl Int4Linear {
    pub fn bytes(&self) -> u64 {
        self.weight.bytes() + self.scale.bytes()
    }
}

/// One NextN routed expert (int8 4-bit, not EXL3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Int4Expert {
    pub w1: Int4Linear,
    pub w2: Int4Linear,
    pub w3: Int4Linear,
}

impl Int4Expert {
    pub fn projections(&self) -> [&Int4Linear; 3] {
        [&self.w1, &self.w2, &self.w3]
    }
}

/// One NextN routed expert, in whichever format the pack stores it: int8 4-bit (the first upload) or EXL3 `K2` (the
/// re-upload). The two are the same three projections of the same shape; only their bytes differ, so every reader of a NextN
/// expert branches here rather than in each of them.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)] // the weights tree is cold; the two formats differ in size by construction
pub enum MtpExpert {
    /// int8 4-bit: `weight [n, k / 2]` + `scale [n / 32, k / 32]`.
    Int4(Int4Expert),
    /// EXL3: the trunk's own expert type — `trellis`/`mul1`/`suh`/`svh` a projection, at the pack's bit width.
    Exl3(Expert),
}

impl MtpExpert {
    /// Bytes the pack holds for this expert: its parts, without any record padding.
    pub fn bytes(&self) -> u64 {
        match self {
            MtpExpert::Int4(e) => e.projections().iter().map(|g| g.bytes()).sum(),
            MtpExpert::Exl3(e) => e.projections().iter().map(|g| g.bytes()).sum(),
        }
    }

    /// Where each part of this expert sits in its record, and the record's length — the same walk [`Expert::record_layout`] makes
    /// for the trunk's experts, with two parts a projection instead of four for the int8 format. `offsets[p][q]` is projection
    /// `p`'s part `q` in the order `weight`, `scale` (int8) or `trellis`, `mul1`, `suh`, `svh` (EXL3).
    pub fn record_layout(&self) -> ([[u64; 4]; 3], u64) {
        match self {
            MtpExpert::Exl3(e) => e.record_layout(),
            MtpExpert::Int4(e) => {
                let a = gb_cuda::expert_cache::EXPERT_PART_ALIGN;
                let mut offsets = [[0u64; 4]; 3];
                let mut off = 0u64;
                for (p, g) in e.projections().into_iter().enumerate() {
                    for (q, t) in [&g.weight, &g.scale].into_iter().enumerate() {
                        off = off.div_ceil(a) * a;
                        offsets[p][q] = off;
                        off += t.bytes();
                    }
                }
                (offsets, off.div_ceil(a) * a)
            }
        }
    }

    /// The record's length: the parts plus their alignment padding ([`MtpExpert::record_layout`]).
    pub fn record_bytes(&self) -> u64 {
        self.record_layout().1
    }

    /// The EXL3 expert, when this is one: what the drafter's grouped GEMM path takes.
    pub fn as_exl3(&self) -> Option<&Expert> {
        match self {
            MtpExpert::Exl3(e) => Some(e),
            MtpExpert::Int4(_) => None,
        }
    }

    /// The format's name, for the boot report and the errors.
    pub fn format(&self) -> &'static str {
        match self {
            MtpExpert::Int4(_) => "int4",
            MtpExpert::Exl3(_) => "exl3",
        }
    }
}

/// Multi-head latent attention: low-rank query, one shared KV head, grouped low-rank output, and the per-head sink logits that
/// let a head attend to nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mla {
    /// `[hidden -> q_lora_rank]`.
    pub wq_a: Fp8Linear,
    /// `[q_lora_rank -> heads x head_dim]`.
    pub wq_b: Fp8Linear,
    /// `[hidden -> kv_heads x head_dim]`: the single shared latent.
    pub wkv: Fp8Linear,
    /// `[o_groups x head_dim -> o_groups x o_lora_rank]`.
    pub wo_a: Fp8Linear,
    /// `[o_groups x o_lora_rank -> hidden]`.
    pub wo_b: Fp8Linear,
    /// bf16 `[q_lora_rank]`, applied to the query before `wq_b`.
    pub q_norm: Tensor,
    /// bf16 `[kv_heads x head_dim]`, applied to the latent.
    pub kv_norm: Tensor,
    /// fp32 `[heads]`.
    pub attn_sink: Tensor,
}

impl Mla {
    pub fn tensors(&self) -> [&Tensor; 13] {
        [
            &self.wq_a.weight,
            &self.wq_a.scale,
            &self.wq_b.weight,
            &self.wq_b.scale,
            &self.wkv.weight,
            &self.wkv.scale,
            &self.wo_a.weight,
            &self.wo_a.scale,
            &self.wo_b.weight,
            &self.wo_b.scale,
            &self.q_norm,
            &self.kv_norm,
            &self.attn_sink,
        ]
    }
}

/// The router: bf16 `[experts, hidden]` logits, and the two fp32 `[experts]` correction biases (`bias` for the text path,
/// `bias_vl` for the vision one).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Router {
    pub weight: Tensor,
    pub bias: Tensor,
    pub bias_vl: Tensor,
}

/// The shared expert of a MoE layer: `down(act(gate(x)) * up(x))`, FP8 throughout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedExperts {
    pub w1: Fp8Linear,
    pub w2: Fp8Linear,
    pub w3: Fp8Linear,
}

impl SharedExperts {
    pub fn bytes(&self) -> u64 {
        self.w1.bytes() + self.w2.bytes() + self.w3.bytes()
    }
}

/// mHC hyper-connections around one sublayer: the fp32 mix matrix (`fn`), its bias (`base`) and the fp32 `[3]` scale.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hc {
    pub fn_: Tensor,
    pub base: Tensor,
    pub scale: Tensor,
}

impl Hc {
    pub fn tensors(&self) -> [&Tensor; 3] {
        [&self.fn_, &self.base, &self.scale]
    }
}

/// The compressed KV a layer with a non-zero `compress_ratio` caches: bf16, with no block scales, and a gate only where one
/// entry spans several tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compressor {
    /// bf16 `[head_dim, hidden]`.
    pub wkv: Tensor,
    /// bf16 `[head_dim]`.
    pub norm: Tensor,
    /// bf16 `[head_dim, hidden]`, absent on the layer whose ratio is 1.
    pub gate: Option<Tensor>,
}

/// The DeepSeek sparse-attention indexer: a query projection on every selector, a key projection and its norm only on the
/// selectors that own a pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Indexer {
    /// FP8 `[n_heads x head_dim, q_lora_rank]` plus its block scales.
    pub wq_b: Fp8Linear,
    /// bf16 `[n_heads, hidden]`.
    pub weights_proj: Tensor,
    /// bf16 `[indexer_head_dim, head_dim]`; `None` where the layer shares the preceding selector's key.
    pub wk: Option<Tensor>,
    /// bf16 `[indexer_head_dim]`; `None` with [`Indexer::wk`].
    pub k_norm: Option<Tensor>,
}

/// One of the two engram embedding tables: an FP8 `[table, head_dim]` matrix and its `F8_E8M0` `[table, head_dim / 32]` block
/// scales.
///
/// ~101.5 GB each. This is an `mmap` sidecar: the forward addresses rows through a mapping of the shard and never materialises
/// the table, so nothing here may be turned into a `Vec` of the tensor's contents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MmapTable {
    /// The layer this table belongs to (`1` or `14`), and its row count from `engram_num_embeddings`.
    pub layer: usize,
    pub rows: usize,
    pub embed: Tensor,
    pub scale: Tensor,
}

impl MmapTable {
    pub fn bytes(&self) -> u64 {
        self.embed.bytes() + self.scale.bytes()
    }
}

/// The engram's fused key/value projection in whichever form the pack stores it: plain FP8
/// (`wkv.weight` + `wkv.scale`, the base layout) or an EXL3 group (`wkv.{trellis,suh,svh,mul1}`,
/// the quantized packs' form) that the engine decodes and tile-quantises once at boot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WkvProj {
    Plain(Fp8Linear),
    Group(Exl3Linear),
}

/// The engram token-lookup branch of one layer: the table, the two n-gram projections and the fused key/value projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Engram {
    pub table: MmapTable,
    /// `[max_ngram_size, hidden]` — bf16 in the base layout, f32 in the quantized packs.
    pub q_weight: Tensor,
    /// `[max_ngram_size, hidden]` — bf16 in the base layout, f32 in the quantized packs.
    pub k_weight: Tensor,
    /// `[5 x hidden, 3 x n_heads x head_dim]`, in the pack's [`WkvProj`] form.
    pub wkv: WkvProj,
}

/// One trunk layer: its two norms, its mixer, its MoE block, its hyper-connections, and the three branches that exist only on
/// some layers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrunkLayer {
    pub index: usize,
    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    pub mla: Mla,
    pub router: Router,
    pub shared: SharedExperts,
    pub experts: ExpertBank,
    pub hc_attn: Hc,
    pub hc_ffn: Hc,
    /// `Some` exactly on the layers in `kv_source_layer_ids`.
    pub compressor: Option<Compressor>,
    /// `Some` exactly on the layers in `index_source_layer_ids`.
    pub indexer: Option<Indexer>,
    /// `Some` exactly on the layers in `engram_layer_ids`.
    pub engram: Option<Engram>,
}

/// One NextN layer: the trunk's MLA + MoE block on int8 experts, the tap the first one fuses, and the DSpark heads the last one
/// carries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MtpLayer {
    pub index: usize,
    pub attn_norm: Tensor,
    pub ffn_norm: Tensor,
    pub mla: Mla,
    pub router: Router,
    pub shared: SharedExperts,
    /// The drafter's routed experts, in the pack's own format (see [`MtpExpert`]).
    pub experts: Vec<MtpExpert>,
    pub hc_attn: Hc,
    pub hc_ffn: Hc,
    /// `Some` on layer 0 only: FP8 `[hidden, 3 x hidden]`, the three trunk taps it fuses.
    pub main_proj: Option<Fp8Linear>,
    /// `Some` on layer 0 only: bf16 `[hidden]`.
    pub main_norm: Option<Tensor>,
    /// `Some` on the last layer only: bf16 `[vocab, markov_rank]`.
    pub markov_embed: Option<Tensor>,
    /// `Some` on the last layer only: bf16 `[vocab, markov_rank]`.
    pub markov_head: Option<Tensor>,
    /// `Some` on the last layer only: bf16 `[1, hidden + markov_rank]`.
    pub confidence: Option<Tensor>,
    /// `Some` on the last layer only: bf16 `[hidden]`.
    pub norm: Option<Tensor>,
}

impl MtpLayer {
    pub fn experts_bytes(&self) -> u64 {
        self.experts.iter().map(|e| e.bytes()).sum()
    }

    /// The format these experts are stored in, or `None` for a layer with no routed expert.
    pub fn expert_format(&self) -> Option<&'static str> {
        self.experts.first().map(|e| e.format())
    }
}

/// A bf16 weight with its bias (the vision tower's blocks are all of this shape).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionLinear {
    pub weight: Tensor,
    pub bias: Tensor,
}

/// One vision block. The MLP runs `w2(act(w1(x))[:, :inter] * x[:, inter:])`, which is why `w1` is `[2 x inter, hidden]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionBlock {
    pub norm1: Tensor,
    pub norm2: Tensor,
    /// `[3 x hidden, hidden]`, fused.
    pub qkv: VisionLinear,
    /// `[hidden, hidden]`.
    pub wo: VisionLinear,
    /// `[2 x intermediate, hidden]`.
    pub mlp_w1: Tensor,
    /// `[hidden, intermediate]`.
    pub mlp_w2: Tensor,
}

/// The projector that splices the tower's features into the text stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Aligner {
    /// `[hidden, tower_hidden x downsample_ratio^2]`.
    pub w1: VisionLinear,
    /// `[hidden, hidden]`.
    pub w2: VisionLinear,
}

/// The vision tower: a component of its own (266 bf16 tensors, none of them FP8 or EXL3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vision {
    /// `[tower_hidden, 3 x patch x patch]`.
    pub patch_embed: VisionLinear,
    pub blocks: Vec<VisionBlock>,
    /// `[tower_hidden]`.
    pub norm: Tensor,
    /// Three learned rows the prompt splices around an image's features: `[hidden]` each.
    pub image_start: Tensor,
    pub image_end: Tensor,
    pub image_newline: Tensor,
}

/// The LM head (`head.weight` in the first upload, an EXL3 group `head.{trellis,mul1,suh,svh}` in the v7 re-upload).
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)] // Bf16 vs Exl3 carriers differ; boxed indirection would only complicate the tree walk
pub enum LmHead {
    Bf16(Tensor),
    Exl3(Exl3Linear),
}

impl LmHead {
    pub fn bytes(&self) -> u64 {
        match self {
            LmHead::Bf16(t) => t.bytes(),
            LmHead::Exl3(g) => g.bytes(),
        }
    }

    pub(crate) fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        match self {
            LmHead::Bf16(t) => f(t),
            LmHead::Exl3(g) => g.visit(f),
        }
    }
}

/// Every tensor of the pack, typed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeepseekV41Weights {
    /// bf16 `[vocab, hidden]`.
    pub embed: Tensor,
    /// bf16 `[vocab, hidden]` or EXL3 group `k = hidden, n = vocab`.
    pub head: LmHead,
    /// bf16 `[hidden]`.
    pub norm: Tensor,
    /// The trunk, `layers[i].index == i`.
    pub layers: Vec<TrunkLayer>,
    /// The NextN layers after the trunk, `mtp[i].index == i`.
    pub mtp: Vec<MtpLayer>,
    pub vision: Option<Vision>,
    /// `Some` exactly when there is a vision tower: three `[hidden, ..]` bf16 matrices.
    pub aligner: Option<Aligner>,
}

/// A tensor of the tree, as a traversal sees it.
pub type Leaf<'a> = &'a Tensor;

macro_rules! visit_each {
    ($f:expr, $($t:expr),* $(,)?) => { $( $f($t); )* };
}

impl Mla {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        self.wq_a.visit(f);
        self.wq_b.visit(f);
        self.wkv.visit(f);
        self.wo_a.visit(f);
        self.wo_b.visit(f);
        visit_each!(f, &self.q_norm, &self.kv_norm, &self.attn_sink);
    }
}

impl Hc {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        for t in self.tensors() {
            f(t);
        }
    }
}

impl SharedExperts {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        for g in [&self.w1, &self.w2, &self.w3] {
            g.visit(f);
        }
    }
}

impl Router {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        visit_each!(f, &self.weight, &self.bias, &self.bias_vl);
    }
}

impl Exl3Linear {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        visit_each!(f, &self.trellis, &self.mul1, &self.suh, &self.svh);
    }
}

impl Int4Linear {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        visit_each!(f, &self.weight, &self.scale);
    }
}

impl ExpertBank {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        for e in &self.experts {
            for g in e.projections() {
                g.visit(f);
            }
        }
    }
}

impl Compressor {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        visit_each!(f, &self.wkv, &self.norm);
        if let Some(g) = &self.gate {
            f(g);
        }
    }
}

impl Indexer {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        visit_each!(f, &self.wq_b.weight, &self.wq_b.scale, &self.weights_proj);
        if let Some(t) = &self.wk {
            f(t);
        }
        if let Some(t) = &self.k_norm {
            f(t);
        }
    }
}

impl Engram {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        visit_each!(f, &self.table.embed, &self.table.scale, &self.q_weight, &self.k_weight);
        match &self.wkv {
            WkvProj::Plain(w) => {
                visit_each!(f, &w.weight, &w.scale);
            }
            WkvProj::Group(g) => {
                visit_each!(f, &g.trellis, &g.suh, &g.svh, &g.mul1);
            }
        }
    }
}

impl TrunkLayer {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        visit_each!(f, &self.attn_norm, &self.ffn_norm);
        self.hc_attn.visit(f);
        self.hc_ffn.visit(f);
        self.mla.visit(f);
        self.router.visit(f);
        self.shared.visit(f);
        self.experts.visit(f);
        if let Some(c) = &self.compressor {
            c.visit(f);
        }
        if let Some(i) = &self.indexer {
            i.visit(f);
        }
        if let Some(e) = &self.engram {
            e.visit(f);
        }
    }
}

impl MtpLayer {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        visit_each!(f, &self.attn_norm, &self.ffn_norm);
        self.hc_attn.visit(f);
        self.hc_ffn.visit(f);
        self.mla.visit(f);
        self.router.visit(f);
        self.shared.visit(f);
        for e in &self.experts {
            match e {
                MtpExpert::Int4(e) => {
                    for g in e.projections() {
                        g.visit(f);
                    }
                }
                MtpExpert::Exl3(e) => {
                    for g in e.projections() {
                        g.visit(f);
                    }
                }
            }
        }
        if let Some(g) = &self.main_proj {
            visit_each!(f, &g.weight, &g.scale);
        }
        for t in [&self.main_norm, &self.markov_embed, &self.markov_head, &self.confidence, &self.norm].into_iter().flatten() {
            f(t);
        }
    }
}

impl VisionLinear {
    fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        visit_each!(f, &self.weight, &self.bias);
    }
}

impl Vision {
    /// Every tensor of the tower, in block order.
    pub fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        self.patch_embed.visit(f);
        for b in &self.blocks {
            visit_each!(f, &b.norm1, &b.norm2);
            b.qkv.visit(f);
            b.wo.visit(f);
            visit_each!(f, &b.mlp_w1, &b.mlp_w2);
        }
        visit_each!(f, &self.norm, &self.image_start, &self.image_end, &self.image_newline);
    }
}

impl Aligner {
    /// Every tensor of the projector.
    pub fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        self.w1.visit(f);
        self.w2.visit(f);
    }
}

impl DeepseekV41Weights {
    /// Every tensor of the tree (text, NextN, vision and aligner), in tree order.
    pub fn visit<'a>(&'a self, f: &mut dyn FnMut(Leaf<'a>)) {
        f(&self.embed);
        self.head.visit(f);
        f(&self.norm);
        for l in &self.layers {
            l.visit(f);
        }
        for m in &self.mtp {
            m.visit(f);
        }
        if let Some(v) = &self.vision {
            v.visit(f);
        }
        if let Some(a) = &self.aligner {
            a.visit(f);
        }
    }

    /// How many tensors the tree describes.
    pub fn len(&self) -> usize {
        let mut n = 0;
        self.visit(&mut |_| n += 1);
        n
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total bytes on disk.
    pub fn bytes(&self) -> u64 {
        let mut b = 0;
        self.visit(&mut |t| b += t.bytes());
        b
    }

    /// How many tensors each dtype has, ascending by dtype.
    pub fn dtypes(&self) -> Vec<(Dtype, usize)> {
        let mut h = std::collections::BTreeMap::new();
        self.visit(&mut |t| *h.entry(t.dtype).or_insert(0usize) += 1);
        h.into_iter().collect()
    }

    /// The two engram tables, addressed through an `mmap` of their shard rather than read into memory.
    pub fn engram_tables(&self) -> impl Iterator<Item = &MmapTable> {
        self.layers.iter().filter_map(|l| l.engram.as_ref().map(|e| &e.table))
    }

    /// The routed experts of one layer, when it has a bank.
    pub fn experts(&self, layer: usize) -> Option<&ExpertBank> {
        self.layers.get(layer).map(|l| &l.experts)
    }
}
