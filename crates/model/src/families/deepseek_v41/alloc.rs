//! What a DeepSeek-V4.1-Flash pack costs, as arithmetic: one function per allocation class, each an exact byte count computed from
//! [`DeepseekV41Config`] (and, for the routed experts, from the width of each expert's own trellis), with the formula in a comment
//! and a test that pins the value for the real pack. Nothing here touches a file or a GPU, so every figure in this module can be
//! checked on a host with no device — which is the point: the load plan ([`super::plan`]) and admission both price themselves from
//! this module, and the real allocations are checked against it when the backend is written.
//!
//! The pack this family targets is 332,513,253,720 B of tensors (309.7 GiB) — 2.75x the 121 GiB of unified memory the device has —
//! so the arithmetic is not a formality. It is the reason the plan is partial: the routed experts alone are 111.0 GB and the two
//! engram tables 202.8 GB, and each is a different kind of memory (see [`super::plan::Placement`]).
//!
//! ## The classes, and where the bytes go
//!
//! | class | bytes | share |
//! |---|---:|---:|
//! | routed experts (EXL3, per-expert widths) | 111,005,061,120 | 33.4% |
//! | engram tables (`mmap` sidecar) | 202,758,032,400 | 61.0% |
//! | MLA projections | 5,069,721,600 | 1.52% |
//! | MTP (3 NextN layers, int8 experts included) | 7,932,874,632 | 2.39% |
//! | embed + head | 2,647,654,400 | 0.80% |
//! | shared experts | 1,416,960,000 | 0.43% |
//! | vision tower | 823,715,840 | 0.25% |
//! | router + mHC + norms | 315,533,760 | 0.09% |
//! | engram branch (the table's projections) | 315,043,840 | 0.09% |
//! | aligner | 146,821,120 | 0.04% |
//! | DSA indexer | 45,130,752 | 0.01% |
//! | DSA compressor | 36,704,256 | 0.01% |
//!
//! ## What is *not* weight
//!
//! One forward's nominal scratch is priced here too ([`scratch_bytes`], the activations of one call item by item), and
//! [`compress_slots`] is the per-layer compressed-entry count every plane of the cache is sized from.
//!
//! **What admission charges is the constructors' own formulas, not this module's nominal ones.** This pack allocates no paged
//! latent — its KV is the DSA planes, five buffers a layer ([`super::plan::attn_planes_bytes`] over
//! `gb_cuda::dsv4_attn::Dsv4Planes::bytes`) — and its scratch is [`super::model::Scratch`], so [`super::plan`] charges those
//! two instead. The old `kv_latent_bytes` / `kv_compressed_bytes` / `indexer_key_bytes` triple is gone: it priced a paged
//! latent that is never allocated, at a ratio taken from the table's maximum instead of from each layer.
//!
//! [`super::memory`]'s `SCRATCH_BYTES_PER_ROW` (a flat 1 MiB a row) is a placeholder that predates this module: [`scratch_bytes`]
//! is the same quantity computed from the buffers' shapes, and a test here pins it below that placeholder, so admission that prices
//! scratch with this module never admits *more* than the placeholder would have.

use super::config::DeepseekV41Config;
use super::inventory::MtpExpertFormat;
use super::weights::{Exl3Linear, ExpertBank};

/// An FP8 block scale is 32x32: one `F8_E8M0` byte per 32x32 tile of the weight it scales.
pub const FP8_TILE: usize = 32;
/// fp16/bf16 element.
pub const ELEM_2: u64 = 2;
/// fp32 element.
pub const ELEM_4: u64 = 4;
/// Tokens per KV page (`crate::kvtables::BLOCK_TOKENS`): the paged cache allocates whole pages, so a context is rounded up to one.
pub const PAGE_TOKENS: usize = 256;

/// The dtype the cache's planes hold a value in.
///
/// The device cache is three planes ([`gb_cuda::dsv4_attn::Dsv4Planes`]): a 128-row sliding-window ring, one compressed
/// entry per `compress_ratio` tokens, and the DSA indexer's keys. This enum is their element type, and the three pricing
/// functions below are the device's own byte counts for them — [`gb_cuda::dsv4_kvq4::KvFmt`] is the layout, so a change
/// to the row there moves the price here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvQuant {
    /// 4 bits over the 448 nope dims (one E8M0 scale a group of 32) and 64 RoPE dims fp16: a 368-byte compressed
    /// entry, an 80-byte index key. 2.83x smaller than [`Self::F16`] at 32K.
    Q4,
    /// fp16 everywhere: a 1,024-byte compressed entry, a 256-byte index key — the arithmetic every golden was made with.
    F16,
}

impl KvQuant {
    /// The device layout this names.
    pub const fn fmt(self) -> gb_cuda::dsv4_kvq4::KvFmt {
        match self {
            KvQuant::Q4 => gb_cuda::dsv4_kvq4::KvFmt::Q4,
            KvQuant::F16 => gb_cuda::dsv4_kvq4::KvFmt::F16,
        }
    }

    /// [`Self::fmt`]'s inverse: the price's spelling of a device layout.
    pub const fn from_fmt(f: gb_cuda::dsv4_kvq4::KvFmt) -> Self {
        match f {
            gb_cuda::dsv4_kvq4::KvFmt::Q4 => KvQuant::Q4,
            gb_cuda::dsv4_kvq4::KvFmt::F16 => KvQuant::F16,
        }
    }
}

/// Compressed entries a layer's plane holds at `tokens`: one per `ratio` tokens plus the partial one.
///
/// Per layer, because a ratio-1 layer's plane holds twice what a ratio-2 layer's does
/// ([`gb_cuda::dsv4_attn::compress_slots`]).
pub fn compress_slots(cfg: &DeepseekV41Config, layer: usize, tokens: u64) -> u64 {
    let r = (cfg.compress.ratio(layer) as u64).max(1);
    tokens.div_ceil(r) + 1
}

/// One named allocation, or group of equal ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Item {
    pub name: &'static str,
    pub bytes: u64,
}

/// Bytes of an FP8 block projection `[n, k]`: the `F8_E4M3` weight plus its `F8_E8M0` `[n / 32, k / 32]` scales.
///
/// `n * k * 1 + (n / 32) * (k / 32) * 1`
pub fn fp8_linear_bytes(n: usize, k: usize) -> u64 {
    (n as u64) * (k as u64) + (n as u64 / FP8_TILE as u64) * (k as u64 / FP8_TILE as u64)
}

/// The embedding and the (untied) head: two bf16 `[vocab, hidden]` matrices.
///
/// `2 * vocab * hidden * 2`  ->  2,647,654,400 B for the real config (2.648 GB).
pub fn embed_head_bytes(cfg: &DeepseekV41Config) -> u64 {
    2 * (cfg.vocab as u64) * (cfg.hidden as u64) * ELEM_2
}

/// One trunk layer's MLA: the five FP8 projections with their block scales, the two norms and the per-head sink logits.
///
/// `fp8(q_rank, hidden) + fp8(heads * head_dim, q_rank) + fp8(kv_width, hidden)
///  + fp8(o_rank, o_inner) + fp8(hidden, o_rank) + q_rank * 2 + kv_width * 2 + heads * 4`
pub fn mla_layer_bytes(cfg: &DeepseekV41Config) -> u64 {
    let q_rank = cfg.q_lora_rank;
    let kv = cfg.kv_heads * cfg.head_dim;
    let o_inner = cfg.o_groups * cfg.head_dim;
    let o_rank = cfg.o_groups * cfg.o_lora_rank;
    fp8_linear_bytes(q_rank, cfg.hidden)
        + fp8_linear_bytes(cfg.heads * cfg.head_dim, q_rank)
        + fp8_linear_bytes(kv, cfg.hidden)
        + fp8_linear_bytes(o_rank, o_inner)
        + fp8_linear_bytes(cfg.hidden, o_rank)
        + (q_rank as u64) * ELEM_2
        + (kv as u64) * ELEM_2
        + (cfg.heads as u64) * ELEM_4
}

/// Every trunk layer's MLA: `layers * mla_layer_bytes`  ->  5,069,721,600 B (5.070 GB), 126,743,040 B a layer.
pub fn mla_bytes(cfg: &DeepseekV41Config) -> u64 {
    (cfg.layers as u64) * mla_layer_bytes(cfg)
}

/// The DSA compressors: on each of the four source layers a bf16 `[head_dim, hidden]` `wkv` and a bf16 `[head_dim]` norm, and on
/// the three whose entry spans more than one token a bf16 `[head_dim, hidden]` gate.
///
/// `sources * (head_dim * hidden * 2 + head_dim * 2) + gates * head_dim * hidden * 2`  ->  36,704,256 B (four compressors, three gated).
pub fn compressor_bytes(cfg: &DeepseekV41Config) -> u64 {
    let wkv = (cfg.head_dim as u64) * (cfg.hidden as u64) * ELEM_2;
    let norm = (cfg.head_dim as u64) * ELEM_2;
    let sources = cfg.compress.source_layer_ids.len() as u64;
    let gates = (0..cfg.layers).filter(|&l| cfg.compress.has_gate(l)).count() as u64;
    sources * (wkv + norm) + gates * wkv
}

/// The DSA indexers: on each of the eight selectors an FP8 `[n_heads * head_dim, q_rank]` query projection and a bf16
/// `[n_heads, hidden]` score projection; on the four that own a pool also a bf16 `[index_dim, head_dim]` key and its norm.
///
/// `selectors * (fp8(n_heads * index_dim, q_rank) + n_heads * hidden * 2) + owners * (index_dim * head_dim * 2 + index_dim * 2)`
/// -> 45,130,752 B.
pub fn indexer_bytes(cfg: &DeepseekV41Config) -> u64 {
    let ix = &cfg.indexer;
    let q = fp8_linear_bytes(ix.n_heads * ix.head_dim, cfg.q_lora_rank) + (ix.n_heads as u64) * (cfg.hidden as u64) * ELEM_2;
    let k = (ix.head_dim as u64) * (cfg.head_dim as u64) * ELEM_2 + (ix.head_dim as u64) * ELEM_2;
    let selectors = ix.source_layer_ids.len() as u64;
    let owners = cfg.compress.source_layer_ids.len() as u64;
    selectors * q + owners * k
}

/// One layer's shared expert: three FP8 block projections `[inter, hidden]`, `[hidden, inter]`, `[inter, hidden]`.
///
/// `2 * fp8(inter, hidden) + fp8(hidden, inter)`  ->  35,424,000 B a layer, 1,416,960,000 B over the 40 layers (1.417 GB).
pub fn shared_expert_bytes(cfg: &DeepseekV41Config) -> u64 {
    let inter = cfg.moe.intermediate;
    (cfg.layers as u64) * (2 * fp8_linear_bytes(inter, cfg.hidden) + fp8_linear_bytes(cfg.hidden, inter))
}

/// The routers: a bf16 `[experts, hidden]` logit matrix and two fp32 `[experts]` correction biases a layer.
///
/// `layers * (experts * hidden * 2 + 2 * experts * 4)`  ->  157,409,280 B (3,935,232 B a layer).
pub fn router_bytes(cfg: &DeepseekV41Config) -> u64 {
    let e = cfg.moe.routed_experts as u64;
    (cfg.layers as u64) * (e * (cfg.hidden as u64) * ELEM_2 + 2 * e * ELEM_4)
}

/// The mHC hyper-connections: two sets a layer (one around the mixer, one around the FFN) of an fp32 `[6 * mult, mult * hidden]`
/// mix matrix, an fp32 `[6 * mult]` bias and an fp32 `[3]` scale.
///
/// `layers * 2 * (6 * mult * mult * hidden * 4 + 6 * mult * 4 + 3 * 4)`  ->  157,295,040 B (3,932,376 B a layer).
pub fn hc_bytes(cfg: &DeepseekV41Config) -> u64 {
    let m = cfg.hc.mult as u64;
    let q = 6; // HC_COEFFS_PER_STREAM
    (cfg.layers as u64) * 2 * (q * m * m * (cfg.hidden as u64) * ELEM_4 + q * m * ELEM_4 + 3 * ELEM_4)
}

/// Every norm of the text model: two bf16 `[hidden]` a trunk layer and the final one.
///
/// `(2 * layers + 1) * hidden * 2`  ->  829,440 B.
pub fn norm_bytes(cfg: &DeepseekV41Config) -> u64 {
    (2 * cfg.layers as u64 + 1) * (cfg.hidden as u64) * ELEM_2
}

/// The three NextN layers: the same MLA + router + shared expert + mHC block as a trunk layer (on 128 experts, not 384), their
/// routed experts, the first layer's `main_proj` tap and the last layer's DSpark Markov and confidence heads.
///
/// The experts are `bits / 8` bytes a weight plus the two fp16 scale vectors and the `I32` marker (EXL3), or a half-byte a weight
/// plus one `E8M0` a 32 inputs (the first upload's int8).
///
/// `nextn * (mla_layer + 2 * hidden * 2 + router(128) + shared + hc)
///  + nextn * 128 * 3 * projection
///  + [fp8(hidden, 3 * hidden) + hidden * 2] + [2 * vocab * markov_rank * 2 + (hidden + markov_rank) * 2 + hidden * 2]`
///
/// -> 7,932,874,632 B (7.933 GB) for the int8 pack — 7.22 GB of it the experts — and 4,127,924,616 B (4.128 GB) for the
/// re-upload, whose 384 `K2` experts are 3,414,495,744 B.
pub fn mtp_bytes(cfg: &DeepseekV41Config, fmt: MtpExpertFormat) -> u64 {
    let h = cfg.hidden as u64;
    let inter = cfg.moe.intermediate as u64;
    let n = cfg.nextn as u64;
    let es = cfg.dspark.n_routed_experts as u64;
    let m = cfg.hc.mult as u64;
    let proj = |k: u64, nn: u64| match fmt {
        // an int8 expert: the weight packs two values a byte and the scale is one byte per 32 inputs
        MtpExpertFormat::Int4 => nn * (k / 2) + nn * (k / 32),
        // an EXL3 K2 projection: k * n * 2 bits, one fp16 a row and a column, and the I32 codebook marker
        MtpExpertFormat::Exl3 => nn * k / 4 + nn * ELEM_2 + k * ELEM_2 + ELEM_4,
    };
    let expert = 2 * proj(h, inter) + proj(inter, h); // w1, w3: [inter, hidden]; w2: [hidden, inter]
    let block = mla_layer_bytes(cfg)
        + 2 * h * ELEM_2
        + (es * h * ELEM_2 + 2 * es * ELEM_4)
        + (2 * fp8_linear_bytes(inter as usize, cfg.hidden) + fp8_linear_bytes(cfg.hidden, inter as usize))
        + 2 * (6 * m * m * h * ELEM_4 + 6 * m * ELEM_4 + 3 * ELEM_4);
    let tap = if n > 0 { fp8_linear_bytes(cfg.hidden, 3 * cfg.hidden) + h * ELEM_2 } else { 0 };
    let head = if n > 0 {
        2 * (cfg.vocab as u64) * (cfg.dspark.markov_rank as u64) * ELEM_2
            + ((cfg.hidden + cfg.dspark.markov_rank) as u64) * ELEM_2
            + h * ELEM_2
    } else {
        0
    };
    n * (block + es * expert) + tap + head
}

/// The vision tower: a bf16 patch embedding, 32 bf16 blocks (fused `[3h, h]` qkv, `[h, h]` output, `[2i, h]` and `[h, i]` MLP,
/// two norms) and three learned `[hidden]` rows the prompt splices around an image.
///
/// `h * patch_dim * 2 + h * 2 + layers * (12h^2 + 6h + 4ih + 2h) + h * 2 + 3 * hidden * 2`  ->  823,715,840 B (0.824 GB).
pub fn vision_bytes(cfg: &DeepseekV41Config) -> u64 {
    let Some(v) = &cfg.vision else { return 0 };
    let h = v.hidden as u64;
    let i = v.intermediate as u64;
    let block = 3 * h * h * ELEM_2   // qkv weight [3h, h]
        + 3 * h * ELEM_2             // qkv bias
        + h * h * ELEM_2             // wo weight
        + h * ELEM_2                 // wo bias
        + 2 * i * h * ELEM_2         // mlp w1 [2i, h]
        + h * i * ELEM_2             // mlp w2 [h, i]
        + 2 * h * ELEM_2; // norm1, norm2 (one [h] each)
    h * (v.patch_dim() as u64) * ELEM_2 + h * ELEM_2 + (v.layers as u64) * block + h * ELEM_2 + 3 * (cfg.hidden as u64) * ELEM_2
}

/// The projector that splices the tower's features into the text stream: `[hidden, tower_hidden * downsample^2]` and
/// `[hidden, hidden]`, each bf16 with a bf16 bias.
///
/// `hidden * (tower_hidden * m2) * 2 + hidden * 2 + hidden * hidden * 2 + hidden * 2`  ->  146,821,120 B (0.147 GB).
pub fn aligner_bytes(cfg: &DeepseekV41Config) -> u64 {
    let Some(v) = &cfg.vision else { return 0 };
    let h = cfg.hidden as u64;
    let m2 = (v.downsample_ratio * v.downsample_ratio) as u64;
    h * (v.hidden as u64) * m2 * ELEM_2 + h * ELEM_2 + h * h * ELEM_2 + h * ELEM_2
}

/// The two engram embedding tables: an FP8 `[rows, head_dim]` matrix and its `F8_E8M0` `[rows, head_dim / 32]` scales each.
///
/// `sum over the two tables of rows * (head_dim + head_dim / 32)`  ->  202,758,032,400 B (202.758 GB). This is the `mmap`
/// sidecar: it is read through a mapping of the shard, never loaded, so it is the one class that does not compete for device memory.
pub fn engram_table_bytes(cfg: &DeepseekV41Config) -> u64 {
    let hd = cfg.engram.head_dim as u64;
    cfg.engram.num_embeddings.iter().map(|&r| (r as u64) * (hd + hd / 32)).sum()
}

/// The engram branch around each table: a bf16 `[max_ngram, hidden]` query and key projection and an FP8
/// `[5 * hidden, 3 * n_heads * head_dim]` fused key/value projection with its block scales.
///
/// `tables * (2 * max_ngram * hidden * 2 + fp8(5 * hidden, 3 * n_heads * head_dim))`  ->  315,043,840 B (two tables).
pub fn engram_branch_bytes(cfg: &DeepseekV41Config) -> u64 {
    let e = &cfg.engram;
    let tables = e.layer_ids.len() as u64;
    let ng = (e.max_ngram_size as u64) * (cfg.hidden as u64) * ELEM_2;
    tables * (2 * ng + fp8_linear_bytes(5 * cfg.hidden, 3 * e.n_heads * e.head_dim))
}

/// Bytes one EXL3 projection of a `k x n` expert takes: an `I16` trellis `[k / 16, n / 16, 16 * bits]`, an `I32` codebook
/// marker, and the `F16` scales `suh` `[k]` and `svh` `[n]`.
///
/// `(k / 16) * (n / 16) * (16 * bits) * 2 + 4 + k * 2 + n * 2`  =  `k * n * bits / 8 + 4 + 2k + 2n`
pub fn expert_projection_bytes(k: usize, n: usize, bits: u8) -> u64 {
    let trellis = (k as u64 / 16) * (n as u64 / 16) * (16 * bits as u64) * ELEM_2;
    trellis + 4 + (k as u64) * ELEM_2 + (n as u64) * ELEM_2
}

/// The bytes of one projection group of a bound expert (its four tensors, as the pack states them).
pub fn exl3_linear_bytes(g: &Exl3Linear) -> u64 {
    g.bytes()
}

/// One layer's routed experts: `sum over experts and projections of expert_projection_bytes`.
pub fn expert_bank_bytes(bank: &ExpertBank) -> u64 {
    bank.bytes()
}

/// The routed experts of every layer, from the bound tree: 111,005,061,120 B (111.005 GB) for the real pack.
///
/// The bit width is a property of the *expert*, not of the layer (the pack's 46,080 experts carry trellises from 1 to 6 bits, in
/// 16 distinct `(w1, w2, w3)` triples), so this figure cannot be derived from the config alone: it is the one class whose bytes
/// come from the pack's headers, through [`super::weights`]. A uniform-width pack would compute it as
/// `layers * experts * expert_projection_bytes(..)`.
pub fn experts_bytes(w: &super::weights::DeepseekV41Weights) -> u64 {
    w.layers.iter().map(|l| l.experts.bytes()).sum()
}

/// The slab every routed expert would occupy: each expert's aligned record ([`super::weights::Expert::record_layout`]), so the
/// parts plus their alignment padding. The largest expert budget that means anything, and what "every expert resident" costs.
pub fn expert_records_bytes(w: &super::weights::DeepseekV41Weights) -> u64 {
    w.layers.iter().flat_map(|l| l.experts.experts.iter()).map(|e| e.record_bytes()).sum()
}

/// The activation and workspace buffers of one forward call of `rows` rows at `context` tokens of history, item by item.
///
/// Every item is a buffer whose shape the config fixes; none of them is a guess at what a kernel will want, and the list is
/// exhaustive over the shapes the family's recipes name. The two that grow with `context` are the indexer's block scores
/// (`rows x ceil(context / block_size)` fp32) and the sparse walk's top-k picks (`rows x topk`).
pub fn scratch_items(cfg: &DeepseekV41Config, rows: usize, context: usize) -> Vec<Item> {
    let r = rows as u64;
    let h = cfg.hidden as u64;
    let inter = cfg.moe.intermediate as u64;
    let kv = (cfg.kv_heads * cfg.head_dim) as u64;
    let o_rank = (cfg.o_groups * cfg.o_lora_rank) as u64;
    let ix = &cfg.indexer;
    let blocks = (context.div_ceil(ix.candidate_block_size.max(1))) as u64;
    vec![
        Item { name: "hidden", bytes: r * h * ELEM_2 },
        // mHC carries `hc_mult` fp32 residual streams
        Item { name: "residual_streams", bytes: r * (cfg.hc.mult as u64) * h * ELEM_4 },
        Item { name: "logits", bytes: r * (cfg.vocab as u64) * ELEM_2 },
        Item { name: "mla_q", bytes: r * (cfg.heads * cfg.head_dim) as u64 * ELEM_2 },
        Item { name: "mla_kv", bytes: r * kv * ELEM_2 },
        Item { name: "mla_o", bytes: r * o_rank * ELEM_2 },
        Item { name: "indexer_q", bytes: r * (ix.n_heads * ix.head_dim) as u64 * ELEM_2 },
        Item { name: "indexer_block_scores", bytes: r * blocks * ELEM_4 },
        Item { name: "indexer_topk", bytes: r * (ix.topk as u64) * ELEM_4 },
        Item { name: "router_logits", bytes: r * (cfg.moe.routed_experts as u64) * ELEM_4 },
        Item { name: "expert_gate_up", bytes: r * 2 * inter * ELEM_2 },
        Item { name: "expert_act", bytes: r * inter * ELEM_2 },
        Item { name: "expert_out", bytes: r * h * ELEM_4 },
        Item { name: "shared_gate_up", bytes: r * 2 * inter * ELEM_2 },
        Item { name: "shared_out", bytes: r * h * ELEM_4 },
        Item { name: "engram_rows", bytes: r * (cfg.engram.max_ngram_size as u64) * (cfg.engram.head_dim as u64) * ELEM_2 },
        Item { name: "sampling", bytes: r * (cfg.moe.experts_per_token as u64) * 8 },
    ]
}

/// The scratch of one forward call: the sum of [`scratch_items`].
///
/// 511,536 B a row plus the indexer's block scores -> 270,295,040 B at 512 rows and 32,768 tokens (0.270 GB), under the 1 MiB a
/// row that [`super::memory`] reserves as a placeholder.
pub fn scratch_bytes(cfg: &DeepseekV41Config, rows: usize, context: usize) -> u64 {
    scratch_items(cfg, rows, context).iter().map(|i| i.bytes).sum()
}

/// Every byte the pack holds, class by class, without needing a bound tree: the twelve config-derived classes.
///
/// The routed experts are not here (their widths are the pack's, not the config's); add [`experts_bytes`] to reach the pack's
/// 332,513,253,720 B.
pub fn config_classes(cfg: &DeepseekV41Config, mtp: MtpExpertFormat) -> Vec<Item> {
    vec![
        Item { name: "embed_head", bytes: embed_head_bytes(cfg) },
        Item { name: "mla", bytes: mla_bytes(cfg) },
        Item { name: "compressor", bytes: compressor_bytes(cfg) },
        Item { name: "indexer", bytes: indexer_bytes(cfg) },
        Item { name: "shared_experts", bytes: shared_expert_bytes(cfg) },
        Item { name: "router", bytes: router_bytes(cfg) },
        Item { name: "hc", bytes: hc_bytes(cfg) },
        Item { name: "norms", bytes: norm_bytes(cfg) },
        Item { name: "mtp", bytes: mtp_bytes(cfg, mtp) },
        Item { name: "vision", bytes: vision_bytes(cfg) },
        Item { name: "aligner", bytes: aligner_bytes(cfg) },
        Item { name: "engram_table", bytes: engram_table_bytes(cfg) },
        Item { name: "engram_branch", bytes: engram_branch_bytes(cfg) },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const CONFIG: &str = include_str!("../../../../../testdata/deepseek_v41/config.json");

    fn real() -> DeepseekV41Config {
        DeepseekV41Config::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    const GIB: u64 = 1 << 30;

    /// Every class, by hand, from the config the pack ships: the numbers docs place at 332.513 GB.
    #[test]
    fn every_weight_class_by_hand() {
        let c = real();
        // 2 x 129,280 x 5,120 x 2
        assert_eq!(embed_head_bytes(&c), 2_647_654_400);
        // 6,560,000 (wq_a) + 41,984,000 (wq_b) + 2,624,000 (wkv) + 33,587,200 (wo_a) + 41,984,000 (wo_b) + 3,840 (norms, sink)
        assert_eq!(mla_layer_bytes(&c), 126_743_040);
        assert_eq!(mla_bytes(&c), 5_069_721_600);
        // four compressors, three of them gated: 4 x (5,242,880 + 1,024) + 3 x 5,242,880
        assert_eq!(compressor_bytes(&c), 36_704_256);
        // eight selectors x 5,575,680 + four keys x 131,328
        assert_eq!(indexer_bytes(&c), 45_130_752);
        // 40 x 3 x 11,808,000
        assert_eq!(shared_expert_bytes(&c), 1_416_960_000);
        // 40 x (384 x 5,120 x 2 + 2 x 384 x 4)
        assert_eq!(router_bytes(&c), 157_409_280);
        // 40 x 2 x (24 x 20,480 x 4 + 24 x 4 + 12)
        assert_eq!(hc_bytes(&c), 157_295_040);
        // 81 norms of 5,120 bf16
        assert_eq!(norm_bytes(&c), 829_440);
        let int4 = mtp_bytes(&c, MtpExpertFormat::Int4);
        assert_eq!(int4, 7_932_874_632);
        // the same three layers on EXL3 K2 experts: 7.219 GB becomes 3.414 GB
        assert_eq!(mtp_bytes(&c, MtpExpertFormat::Exl3), int4 - 7_219_445_760 + 3_414_495_744);
        assert_eq!(mtp_bytes(&c, MtpExpertFormat::Exl3), 4_127_924_616);
        assert_eq!(vision_bytes(&c), 823_715_840);
        assert_eq!(aligner_bytes(&c), 146_821_120);
        // (384,006,168 + 384,016,682) x (256 + 8)
        assert_eq!(engram_table_bytes(&c), 202_758_032_400);
        // two tables x (2 x 4 x 5,120 x 2 + fp8(25,600, 6,144))
        assert_eq!(engram_branch_bytes(&c), 315_043_840);
        // and the twelve config classes total 221,508,192,600 B: the pack minus its experts
        let summed: u64 = config_classes(&c, MtpExpertFormat::Int4).iter().map(|i| i.bytes).sum();
        assert_eq!(summed, 221_508_192_600);
        assert!((summed as f64 / 1e9 - 221.508).abs() < 0.001);
    }

    /// The one class the config cannot state: the experts' own widths. 46,080 projections over 1..=6 bits.
    #[test]
    fn the_expert_width_arithmetic_is_k_n_bits_over_eight() {
        // a 5,120 x 2,304 projection: trellis 320 x 144 x (16 x bits) int16, a 4 B marker and 7,424 B of scales
        assert_eq!(expert_projection_bytes(5_120, 2_304, 1), 5_120 * 2_304 / 8 + 4 + 2 * 5_120 + 2 * 2_304);
        assert_eq!(expert_projection_bytes(5_120, 2_304, 4), 4 * (5_120 * 2_304 / 8) + 4 + 2 * 5_120 + 2 * 2_304);
        // 2,304 x 5,120 is the same rectangle
        assert_eq!(expert_projection_bytes(2_304, 5_120, 3), expert_projection_bytes(5_120, 2_304, 3));
        // the smallest and largest expert the pack holds (1-bit w1/w3 and 6-bit w2, and the reverse)
        assert_eq!(expert_projection_bytes(5_120, 2_304, 1) * 2 + expert_projection_bytes(2_304, 5_120, 1), 4_468_236);
    }

    /// The compressed-entry count is **per layer**: a ratio-1 layer's plane holds twice a ratio-2 layer's, and sizing from the
    /// table's maximum (`ratio_hint`) under-allocates it by 16.8 MB a layer at 32K — the overflow this replaces.
    #[test]
    fn the_compressed_entry_count_is_per_layer() {
        let c = real();
        let dcfg = gb_cuda::dsv4_attn::Dsv4Config::v41();
        assert_eq!((c.layers, c.compress.ratio(0), c.compress.ratio(2), c.compress.ratio(20)), (40, 0, 2, 1));
        // 18 layers at ratio 2 (16,385 entries), 20 at ratio 1 and 2 that pool nothing (32,769: one a token, plus the partial)
        for (layer, want) in [(0usize, 32_769u64), (2, 16_385), (20, 32_769)] {
            assert_eq!(compress_slots(&c, layer, 32_768), want, "layer {layer}");
            // the model's count is the kernel's, layer for layer
            assert_eq!(compress_slots(&c, layer, 32_768) as usize, gb_cuda::dsv4_attn::compress_slots(&dcfg, layer, 32_768));
        }
        let pooled: u64 = (0..c.layers).map(|l| compress_slots(&c, l, 32_768)).sum();
        assert_eq!(pooled, 1_015_848);
        // and the old figure, over the table's maximum ratio, was short on every ratio-1 layer
        let hint = dcfg.ratio_hint() as u64;
        assert_eq!(hint, 2, "the table's largest ratio");
        assert_eq!((0..c.layers).map(|_| 32_768u64.div_ceil(hint) + 1).sum::<u64>(), 655_400);
        assert!(pooled > 655_400);
    }

    /// The price is the device's own bytes, exactly: `layers x slots x Dsv4Planes::bytes` — the five planes of every layer,
    /// at the cache's format, with nothing subtracted and nothing charged twice.
    #[test]
    fn the_price_is_the_device_allocation() {
        use gb_cuda::dsv4_attn::Dsv4Planes;
        let c = real();
        let dcfg = gb_cuda::dsv4_attn::Dsv4Config::v41();
        for fmt in [KvQuant::Q4, KvQuant::F16] {
            for ctx in [4_096usize, 32_768, 131_072] {
                for slots in [1usize, 8] {
                    let device: u64 =
                        (0..c.layers).map(|l| Dsv4Planes::bytes(&dcfg, ctx, l, fmt.fmt()) as u64).sum::<u64>() * slots as u64;
                    let planes = crate::families::deepseek_v41::plan::attn_planes_bytes(&c, ctx, slots, fmt);
                    assert_eq!(device, planes, "{fmt:?} at {ctx} x {slots}: the price is not the allocation");
                }
            }
        }
    }

    /// The scratch of a 512-row chunk: 0.270 GB, and never more than the 1 MiB a row the memory model reserves.
    #[test]
    fn the_scratch_is_the_sum_of_shaped_buffers_and_stays_under_the_placeholder() {
        let c = real();
        let items = scratch_items(&c, 512, 32_768);
        let total: u64 = items.iter().map(|i| i.bytes).sum();
        assert_eq!(scratch_bytes(&c, 512, 32_768), total);
        // 511,536 B a row, plus 512 rows x 4,096 blocks x 4 B of indexer scores
        assert_eq!(total, 512 * 511_536 + 512 * 4_096 * 4);
        assert_eq!(total, 270_295_040);
        assert!((total as f64 / 1e9 - 0.270).abs() < 0.001);
        // what memory.rs holds back as a placeholder is more, at every chunk size
        for rows in [1usize, 128, 512, 2_048] {
            assert!(
                scratch_bytes(&c, rows, 1_048_576) <= rows as u64 * crate::families::deepseek_v41::memory::SCRATCH_BYTES_PER_ROW,
                "{rows} rows"
            );
        }
        // and it grows with rows and with context, monotonically
        assert!(scratch_bytes(&c, 128, 32_768) < scratch_bytes(&c, 512, 32_768));
        assert!(scratch_bytes(&c, 512, 32_768) < scratch_bytes(&c, 512, 1_048_576));
        assert!(scratch_bytes(&c, 512, 1_048_576) < 2 * GIB);
        // no item is empty, and names are unique
        let mut names: Vec<&str> = items.iter().map(|i| i.name).collect();
        assert!(items.iter().all(|i| i.bytes > 0), "an item of zero bytes is not an allocation");
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), items.len());
    }

    /// A text-only pack prices no tower and no projector, rather than a zero-length one.
    #[test]
    fn a_text_only_config_has_no_vision_classes() {
        let mut j: Value = serde_json::from_str(CONFIG).unwrap();
        j.as_object_mut().unwrap().remove("vision_config");
        let c = DeepseekV41Config::from_json(&j).unwrap();
        assert_eq!((vision_bytes(&c), aligner_bytes(&c)), (0, 0));
        // everything else is unchanged
        assert_eq!(config_classes(&c, MtpExpertFormat::Int4).iter().filter(|i| i.bytes == 0).count(), 2);
    }
}
