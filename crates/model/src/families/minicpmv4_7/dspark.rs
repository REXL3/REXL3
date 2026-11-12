//! The DSpark draft of MiniCPM-V-4.7 (`openbmb/MiniCPM-V-4.7-35B-A3B-DSpark`): the block forward, stand-alone (E4a of
//! `docs/MINICPMV4_7_INTEGRATION_PLAN.md`). Nothing here knows the engine, the target's forward or the `Backend`: a target forward
//! hands it tap rows ([`DraftModel::ingest_taps`]) and asks for 15 drafted tokens with their confidences
//! ([`DraftModel::propose`]); E4b wires those two calls into the engine's rounds.
//!
//! # Semantics (pinned by the torch oracle `tools/minicpm_v47/dspark_ref.py` and its acceptance gate, section 11 of the plan)
//!
//! * **Context rows.** For every committed target position `p` the draft keeps one K/V row per layer, computed from the target's
//!   residual stream *after* layers 3, 11, 19, 27, 35 at `p` (the ids are the draft config's `target_layer_ids`, a runtime `Vec`):
//!   the rows are concatenated in ascending layer order, `fc` ([2048, 10240]) and `hidden_norm` give one 2,048-wide row, and every
//!   draft layer projects it with its own `k_proj` / `v_proj` (the layer's `input_layernorm` is NOT applied to a context row),
//!   norms K per head with `k_norm`, rotates it at position `p` and stores K and V in its page.
//! * **Block.** 15 rows `[anchor, mask x 14]` at positions `P .. P + 14`, `P` = the committed length (the number of positions whose
//!   taps exist: the anchor is the last sampled token, which the target has not processed). Rows are the TARGET's embedding rows
//!   (the mask id 248,076 is a row of its table). Each row writes its K/V at its true position and sees every key in `[0, P + 15)`:
//!   the block is NON-causal and every row's visibility bound is `P + 14`.
//! * **Layer.** pre-norm Qwen3 layer with plain-gamma norms (`y = x / rms * w`), q/k per-head norms, rotary over the whole 256-wide
//!   head (theta 1e7, rotate-half pairs `(d, d + 128)`, one-dimensional positions: `glue::attn_prep` at rotary 256), 8 query /
//!   2 KV heads, silu MLP 6,144.
//! * **Heads.** The final norm's rows go through the TARGET's `lm_head` (row `k` predicts the token at `P + k + 1`: the anchor row
//!   predicts the first draft). The Markov head is the keyed selector walk over the top-K candidates of those base logits: score
//!   `= base[c] + markov_w1[x_{k-1}] . markov_w2[c]` (`x_{-1}` = the anchor), the pick of row `k` is the predecessor of row `k + 1`
//!   (`gate` = ones). The confidence of row `k` is `sigmoid(proj . [h_k ; markov_w1[x_{k-1}]] + b)` in fp32.
//!
//! # Memory and rollback
//!
//! The draft's pages are 5 layers x 2 x 2 KV heads x 256 x 2 B = 10,240 B a token (`memory::allocations` prices them as
//! `kv_draft`; [`DraftKv::bytes`] and [`compare_with_plan`] keep plan == allocation). A slot's page list is the engine's: draft page
//! `i` of a slot is the page with the same id as the target's, so prefix snapshots share draft pages by reference exactly like the
//! target's. Context rows are written once per committed position; block rows are overwritten by the next round's context or block
//! before anything reads them, so **rolling back the committed length is a counter**: [`DraftModel::rollback`] never touches a
//! page and the replay of the same taps rewrites the same bits (every linear runs in chunks of at most 128 rows of the row-invariant
//! decode kernel, so even the chunking of a replay cannot move a bit).

use super::config::DSparkConfig;
use super::memory::{AllocItem, AllocKind, PAGE_TOKENS};
use crate::error::ModelError;
use crate::loadctx::LoadCtx;
use crate::pack::Pack;
use core::ffi::c_void;
use gb_cuda::attention::{self, AttnShape, BlockTable, PagedKv, RowIndex};
use gb_cuda::dflash2::{self, GateDtype, WalkSampling};
use gb_cuda::exl3_linear::{linear_workspace_bytes, Dtype, Exl3Weights};
use gb_cuda::glue::{self, AttnPrepCfg};
use gb_cuda::{dspark as kern, glm_vision, DevBuf, Stream};
use std::path::Path;

pub const HIDDEN: usize = 2048;
pub const HEADS: usize = 8;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 256;
pub const INTERMEDIATE: usize = 6144;
pub const LAYERS: usize = 5;
/// The Markov embedding width.
pub const RANK: usize = 256;
/// Draft rows (and drafted tokens) of one round: the anchor and 14 mask rows, each a next-token prediction.
pub const BLOCK: usize = 15;
/// Target rows the verify window needs per stream: the anchor row and the 15 drafts.
pub const VERIFY_ROWS: usize = BLOCK + 1;
/// Candidates per row the selector walk takes at most.
pub const MAX_TOP_K: usize = 64;
pub const EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 1e7;
/// The attention shape of the draft's layers.
pub const ATTN: AttnShape = AttnShape::Q8Kv2;
/// Bytes of one cached token over the draft's layers (K and V, bf16).
pub const KV_BYTES_PER_TOKEN: usize = LAYERS * 2 * KV_HEADS * HEAD_DIM * 2;
/// Rows of one decode-kernel call: draft linears run in chunks of this many rows, so a row's bits never depend on the chunking.
pub const CHUNK_ROWS: usize = 128;
/// Rows of one dense (cuBLAS) call of the reference path.
const DENSE_ROWS: usize = 256;

/// The attention prep of the block rows: 8 query and 2 KV heads of 256, no gate, rotary over the whole head.
pub const BLOCK_PREP: AttnPrepCfg = AttnPrepCfg {
    heads: HEADS,
    kv_heads: KV_HEADS,
    head_dim: HEAD_DIM,
    rotary_dim: HEAD_DIM,
    has_gate: false,
    eps: EPS,
    theta: ROPE_THETA,
};
/// The attention prep of a context row: only K exists, so the kernel's "q" path runs on the same rows with the same weights (its
/// output is discarded); two heads in, two heads out.
const CTX_PREP: AttnPrepCfg = AttnPrepCfg {
    heads: KV_HEADS,
    kv_heads: KV_HEADS,
    head_dim: HEAD_DIM,
    rotary_dim: HEAD_DIM,
    has_gate: false,
    eps: EPS,
    theta: ROPE_THETA,
};

/// Why a draft config cannot run on these kernels (every disagreement, by name; empty when it can).
pub fn config_problems(c: &DSparkConfig, target_vocab: usize, target_layers: usize) -> Vec<String> {
    let mut p = Vec::new();
    let mut eq = |what: &str, got: usize, want: usize| {
        if got != want {
            p.push(format!("{what} is {got}, the kernels implement {want}"));
        }
    };
    eq("hidden_size", c.hidden, HIDDEN);
    eq("num_attention_heads", c.heads, HEADS);
    eq("num_key_value_heads", c.kv_heads, KV_HEADS);
    eq("head_dim", c.head_dim, HEAD_DIM);
    eq("intermediate_size", c.intermediate, INTERMEDIATE);
    eq("num_hidden_layers", c.layers, LAYERS);
    eq("rope_dim (the whole head rotates)", c.rope_dim, HEAD_DIM);
    eq("block_size", c.block_size, BLOCK);
    eq("markov_rank", c.markov_rank, RANK);
    eq("vocab_size", c.vocab, target_vocab);
    eq("draft_vocab_size", c.draft_vocab, target_vocab);
    if c.rope_theta != ROPE_THETA as f64 {
        p.push(format!("rope_theta is {}, the kernels implement {}", c.rope_theta, ROPE_THETA));
    }
    if c.rope_type != "default" {
        p.push(format!("rope_type is {:?}, only \"default\" is implemented", c.rope_type));
    }
    if c.sliding_window.is_some() || c.use_sliding_window || c.layer_types.iter().any(|t| t != "full_attention") {
        p.push("a sliding-window layer: the draft is five full-attention layers".into());
    }
    if c.target_layer_ids.is_empty() || c.target_layer_ids.len() > 8 {
        p.push(format!("target_layer_ids has {} entries (1 to 8 taps)", c.target_layer_ids.len()));
    }
    if !c.target_layer_ids.windows(2).all(|w| w[0] < w[1]) {
        p.push(format!("target_layer_ids {:?} is not strictly ascending (fc concatenates in that order)", c.target_layer_ids));
    }
    if let Some(&m) = c.target_layer_ids.iter().max() {
        if m >= target_layers {
            p.push(format!("target_layer_ids names layer {m}, the target has {target_layers}"));
        }
    }
    if c.num_target_layers != target_layers {
        p.push(format!("num_target_layers is {}, the target has {target_layers}", c.num_target_layers));
    }
    if (c.mask_token_id as usize) >= target_vocab {
        p.push(format!("mask_token_id {} is outside the {target_vocab}-id vocabulary", c.mask_token_id));
    }
    if c.markov_head_type != "vanilla" || c.projector_type != "dspark" {
        p.push(format!(
            "markov_head_type {:?} / projector_type {:?}: only vanilla / dspark are implemented",
            c.markov_head_type, c.projector_type
        ));
    }
    if !c.enable_confidence_head || !c.confidence_head_with_markov {
        p.push("the confidence head is off or without the Markov embedding (the kernel reads [h ; markov_w1[x]])".into());
    }
    p
}

// ------------------------------------------------------------------------------------------------------------ weights

/// A draft linear: an EXL3 group, or the dense reference form (fp16 `[n, k]` converted on the device from the checkpoint's bf16).
enum Lin {
    Exl3(Exl3Weights),
    Dense { w: DevBuf, k: usize, n: usize },
}

impl Lin {
    fn k(&self) -> usize {
        match self {
            Lin::Exl3(w) => w.k,
            Lin::Dense { k, .. } => *k,
        }
    }
    fn n(&self) -> usize {
        match self {
            Lin::Exl3(w) => w.n,
            Lin::Dense { n, .. } => *n,
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Lin::Exl3(w) => w.weight_bytes(),
            Lin::Dense { w, .. } => w.bytes(),
        }
    }
}

struct Layer {
    input_norm: DevBuf,
    post_norm: DevBuf,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    q_norm: DevBuf,
    k_norm: DevBuf,
    gate: Lin,
    up: Lin,
    down: Lin,
}

/// How the draft's linears are stored: EXL3 groups (a draft pack), or dense fp16 converted from the bf16 checkpoint (the
/// reference path, through cuBLAS; `bf16 -> fp16` overflow is counted and refused).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightKind {
    Exl3,
    Dense,
}

struct Weights {
    kind: WeightKind,
    fc: Lin,
    hidden_norm: DevBuf,
    norm: DevBuf,
    layers: Vec<Layer>,
    /// fp16 `[vocab, 256]` each (the checkpoint's bf16 converted in place; every value is far inside fp16's range).
    markov_w1: DevBuf,
    markov_w2: DevBuf,
    /// fp32 `[hidden + rank]` and fp32 `[1]`.
    conf_w: DevBuf,
    conf_b: DevBuf,
}

fn lin_decl<'p>(ctx: &mut LoadCtx<'p>, pack: &'p Pack, name: &str, k: usize, n: usize) -> Result<Lin, ModelError> {
    if pack.has_group(name) {
        let g = pack.group_meta(name)?;
        if (g.k, g.n) != (k, n) {
            return Err(ModelError::Config(format!("draft {name}: expected K={k} N={n}, found K={} N={}", g.k, g.n)));
        }
        return Ok(Lin::Exl3(ctx.group(pack, name)?));
    }
    let w = ctx.raw16(pack, &format!("{name}.weight"), "BF16", &[n, k])?;
    Ok(Lin::Dense { w, k, n })
}

impl Weights {
    fn declare<'p>(ctx: &mut LoadCtx<'p>, pack: &'p Pack, c: &DSparkConfig) -> Result<Self, ModelError> {
        let nt = c.target_layer_ids.len();
        let fc = lin_decl(ctx, pack, "fc", nt * HIDDEN, HIDDEN)?;
        let hidden_norm = ctx.f32_from_bf16(pack, "hidden_norm.weight", &[HIDDEN], false)?;
        let norm = ctx.f32_from_bf16(pack, "norm.weight", &[HIDDEN], false)?;
        let mut layers = Vec::new();
        for l in 0..LAYERS {
            let p = format!("layers.{l}.");
            layers.push(Layer {
                input_norm: ctx.f32_from_bf16(pack, &format!("{p}input_layernorm.weight"), &[HIDDEN], false)?,
                post_norm: ctx.f32_from_bf16(pack, &format!("{p}post_attention_layernorm.weight"), &[HIDDEN], false)?,
                q: lin_decl(ctx, pack, &format!("{p}self_attn.q_proj"), HIDDEN, HEADS * HEAD_DIM)?,
                k: lin_decl(ctx, pack, &format!("{p}self_attn.k_proj"), HIDDEN, KV_HEADS * HEAD_DIM)?,
                v: lin_decl(ctx, pack, &format!("{p}self_attn.v_proj"), HIDDEN, KV_HEADS * HEAD_DIM)?,
                o: lin_decl(ctx, pack, &format!("{p}self_attn.o_proj"), HEADS * HEAD_DIM, HIDDEN)?,
                q_norm: ctx.f32_from_bf16(pack, &format!("{p}self_attn.q_norm.weight"), &[HEAD_DIM], false)?,
                k_norm: ctx.f32_from_bf16(pack, &format!("{p}self_attn.k_norm.weight"), &[HEAD_DIM], false)?,
                gate: lin_decl(ctx, pack, &format!("{p}mlp.gate_proj"), HIDDEN, INTERMEDIATE)?,
                up: lin_decl(ctx, pack, &format!("{p}mlp.up_proj"), HIDDEN, INTERMEDIATE)?,
                down: lin_decl(ctx, pack, &format!("{p}mlp.down_proj"), INTERMEDIATE, HIDDEN)?,
            });
        }
        let markov_w1 = ctx.raw16(pack, "markov_head.markov_w1.weight", "BF16", &[c.vocab, RANK])?;
        let markov_w2 = ctx.raw16(pack, "markov_head.markov_w2.weight", "BF16", &[c.draft_vocab, RANK])?;
        let conf_w = ctx.f32_from_bf16(pack, "confidence_head.proj.weight", &[1, HIDDEN + RANK], false)?;
        let conf_b = ctx.f32_from_bf16(pack, "confidence_head.proj.bias", &[1], false)?;
        let kind = if matches!(fc, Lin::Exl3(_)) { WeightKind::Exl3 } else { WeightKind::Dense };
        Ok(Self { kind, fc, hidden_norm, norm, layers, markov_w1, markov_w2, conf_w, conf_b })
    }

    /// After `ctx.finish()`: every dense linear and the Markov tables become fp16 in place; any value past fp16's range refuses the
    /// draft by name (nothing is clamped silently).
    fn convert(&mut self, st: &Stream) -> Result<(), ModelError> {
        let mut bad = DevBuf::zeroed(4)?;
        let mut jobs: Vec<(String, &mut DevBuf, usize)> = Vec::new();
        if let Lin::Dense { w, k, n } = &mut self.fc {
            jobs.push(("fc".into(), w, *k * *n));
        }
        for (l, ly) in self.layers.iter_mut().enumerate() {
            let Layer { q, k, v, o, gate, up, down, .. } = ly;
            for (name, lin) in [("q", q), ("k", k), ("v", v), ("o", o), ("gate", gate), ("up", up), ("down", down)] {
                if let Lin::Dense { w, k, n } = lin {
                    jobs.push((format!("layers.{l}.{name}"), w, *k * *n));
                }
            }
        }
        let (n1, n2) = (self.markov_w1.bytes() / 2, self.markov_w2.bytes() / 2);
        jobs.push(("markov_w1".into(), &mut self.markov_w1, n1));
        jobs.push(("markov_w2".into(), &mut self.markov_w2, n2));
        for (name, buf, n) in jobs {
            kern::bf16_to_f16_in_place(buf, n, &mut bad, st)?;
            st.sync()?;
            let mut h = [0i32; 1];
            bad.download(&mut h)?;
            if h[0] != 0 {
                return Err(ModelError::Config(format!("draft {name}: {} values do not fit fp16", h[0])));
            }
        }
        Ok(())
    }

    fn bytes(&self) -> usize {
        let mut b = self.fc.bytes() + self.markov_w1.bytes() + self.markov_w2.bytes() + self.conf_w.bytes() + self.conf_b.bytes();
        b += self.hidden_norm.bytes() + self.norm.bytes();
        for l in &self.layers {
            b += l.input_norm.bytes() + l.post_norm.bytes() + l.q_norm.bytes() + l.k_norm.bytes();
            b += [&l.q, &l.k, &l.v, &l.o, &l.gate, &l.up, &l.down].iter().map(|x| x.bytes()).sum::<usize>();
        }
        b
    }
}

// ------------------------------------------------------------------------------------------------------------ the KV pool

/// The draft's paged K/V: five layers' pages (`AttnShape::Q8Kv2`) addressed by one block table with a row per slot. Page ids are the
/// engine's (the target's page ids), so the pool holds `pages` pages exactly like the target's.
pub struct DraftKv {
    pub layers: Vec<PagedKv>,
    pub pages: usize,
    table: BlockTable,
    slot_blocks: Vec<Vec<u32>>,
}

impl DraftKv {
    pub fn new(slots: usize, pages: usize, max_blocks: usize) -> Result<Self, ModelError> {
        if slots == 0 || pages == 0 || max_blocks == 0 {
            return Err(ModelError::Config(format!("draft KV of {slots} slots, {pages} pages, {max_blocks} blocks")));
        }
        let layers = (0..LAYERS).map(|_| PagedKv::with_shape(pages, ATTN)).collect::<Result<Vec<_>, _>>()?;
        let slot_blocks = vec![Vec::new(); slots];
        let table = BlockTable::from_host(&slot_blocks, max_blocks)?;
        Ok(Self { layers, pages, table, slot_blocks })
    }

    /// Bytes of every K and V buffer.
    pub fn bytes(&self) -> u64 {
        self.layers.iter().map(|c| (c.k.bytes() + c.v.bytes()) as u64).sum()
    }

    pub fn max_blocks(&self) -> usize {
        self.table.max_blocks
    }
}

/// The draft's pool against the memory model's `kv_draft` item (`memory::allocations`): every difference named, empty when plan and
/// device agree.
pub fn compare_with_plan(items: &[AllocItem], kv: &DraftKv) -> Vec<String> {
    match items.iter().find(|i| i.name == "kv_draft") {
        None => vec!["the plan prices no kv_draft item".to_string()],
        Some(it) => {
            debug_assert_eq!(it.kind, AllocKind::PerPage);
            let (want, held) = (it.bytes * kv.pages as u64, kv.bytes());
            if held == want {
                Vec::new()
            } else {
                vec![format!("kv_draft: the device holds {held} B, the plan prices {} x {} = {want} B", it.bytes, kv.pages)]
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------ the model

/// The target's vocabulary head as the draft needs it.
pub enum Head<'a> {
    /// The target's `lm_head` as one EXL3 group (N = the padded width, `valid` real ids): the serving path.
    Exl3 { w: &'a Exl3Weights, valid: usize },
    /// A dense fp16 head `[valid, 2048]` (stand-alone tests against the bf16 checkpoint; see [`DenseTarget`]).
    Dense { w: &'a DevBuf, valid: usize },
}

impl Head<'_> {
    pub fn valid(&self) -> usize {
        match self {
            Head::Exl3 { valid, .. } | Head::Dense { valid, .. } => *valid,
        }
    }
    /// Columns of one logits row (the padded width for the EXL3 head).
    pub fn stride(&self) -> usize {
        match self {
            Head::Exl3 { w, .. } => w.n,
            Head::Dense { valid, .. } => *valid,
        }
    }
}

/// The target's bf16 embedding table and an fp16 copy of its head, loaded from the bf16 checkpoint: what the stand-alone tests (and
/// the oracle comparison, whose head is the bf16 one) use in place of the serving engine's loaded weights.
pub struct DenseTarget {
    /// bf16 `[vocab, 2048]`.
    pub embed: DevBuf,
    /// fp16 `[vocab, 2048]`.
    pub head: DevBuf,
    pub vocab: usize,
}

impl DenseTarget {
    pub fn load(dir: &Path, vocab: usize) -> Result<Self, ModelError> {
        let pack = Pack::open(dir)?;
        let mut ctx = LoadCtx::from_env();
        let embed = ctx.raw16(&pack, "model.language_model.embed_tokens.weight", "BF16", &[vocab, HIDDEN])?;
        let mut head = ctx.raw16(&pack, "lm_head.weight", "BF16", &[vocab, HIDDEN])?;
        ctx.finish()?;
        let st = Stream::new()?;
        let mut bad = DevBuf::zeroed(4)?;
        kern::bf16_to_f16_in_place(&mut head, vocab * HIDDEN, &mut bad, &st)?;
        st.sync()?;
        let mut h = [0i32; 1];
        bad.download(&mut h)?;
        if h[0] != 0 {
            return Err(ModelError::Config(format!("lm_head: {} values do not fit fp16", h[0])));
        }
        Ok(Self { embed, head, vocab })
    }

    pub fn head(&self) -> Head<'_> {
        Head::Dense { w: &self.head, valid: self.vocab }
    }
}

/// One slot's share of a batched ingestion ([`DraftModel::ingest_taps_batch`]): `rows` tap rows starting at row `taps_row0` of the tap
/// buffer, for positions `pos0 ..` of `slot`.
#[derive(Clone, Copy, Debug)]
pub struct IngestSeg {
    pub slot: usize,
    pub pos0: usize,
    pub rows: usize,
    pub taps_row0: usize,
}

/// One stream of a draft round.
#[derive(Clone, Copy, Debug)]
pub struct ProposeReq<'a> {
    pub slot: usize,
    /// The anchor token: the last sampled one, not yet through the target.
    pub anchor: u32,
    /// The anchor's position = the slot's committed length `P` (context rows `0..P` exist).
    pub p: usize,
    /// The request's sampling: the walk draws with the target's keyed sampler on sampled requests, by argmax on greedy ones.
    pub sampling: &'a gb_sampling::Sampling,
}

/// One stream's proposal: the 15 drafted tokens (positions `P + 1 .. P + 15`) and the draft's confidence in each.
#[derive(Clone, Debug, PartialEq)]
pub struct Proposal {
    pub tokens: Vec<u32>,
    pub confidence: Vec<f32>,
}

/// What a stage test reads back (bf16 bits of the engine's activations): see [`DraftModel::propose_debug`].
#[derive(Clone, Debug, Default)]
pub struct StageDump {
    /// `[rows, 2048]` the block's embedding rows.
    pub block_embed: Vec<u16>,
    /// per layer `[rows, 2048]`: the residual stream after the layer.
    pub layer_out: Vec<Vec<u16>>,
    /// `[rows, 2048]` the final norm's rows.
    pub final_norm: Vec<u16>,
    /// `[rows, stride]` bf16 base logits.
    pub logits: Vec<u16>,
    pub stride: usize,
    /// `[rows, k]` the candidates the walk saw (the top-K of the base logits, value descending).
    pub cand_ids: Vec<i32>,
    pub cand_vals: Vec<f32>,
}

/// What a context test reads back: the rows after `fc` and after `hidden_norm` (bf16 bits, `[rows, 2048]`).
#[derive(Clone, Debug, Default)]
pub struct CtxDump {
    pub fc: Vec<u16>,
    pub hidden_norm: Vec<u16>,
}

/// Construction options.
#[derive(Clone, Debug)]
pub struct DraftOptions {
    /// Sequence slots (context lengths tracked, page lists held).
    pub slots: usize,
    /// KV pages of 256 tokens (the engine's pool size).
    pub pages: usize,
    /// Streams per round (at most 8: their rows go through the 128-row decode kernels).
    pub max_streams: usize,
    /// Most tap rows one chunk of `ingest_taps` takes (longer inputs are cut into chunks of this size).
    pub max_ctx_rows: usize,
    /// Positions the attention workspace is sized for.
    pub max_context: usize,
    /// Blocks per slot of the table (the engine's `max_blocks`); `max_context + 15` must fit.
    pub max_blocks: usize,
    /// The target's vocabulary (248,144) and layer count (40), for the config check.
    pub vocab: usize,
    pub target_layers: usize,
    /// Columns of a logits row of the head the draft will be run with (`Head::stride`).
    pub head_stride: usize,
    /// The kind of head the draft will be run with: the head's decode workspace (or the dense path's fp16 staging) is allocated and
    /// counted at construction, so the bytes `DraftModel::new` returns are every byte the draft will ever hold.
    pub head: HeadKind,
}

/// The vocabulary head a draft is built for (see [`Head`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadKind {
    /// The target pack's EXL3 `lm_head` at `bits` bits.
    Exl3 { bits: usize },
    /// The dense fp16 stand-in of the stand-alone tests.
    Dense,
}

impl DraftOptions {
    pub fn check(&self) -> Result<(), ModelError> {
        let bad = |m: String| Err(ModelError::Config(format!("draft options: {m}")));
        if self.max_streams == 0 || self.max_streams * BLOCK > CHUNK_ROWS {
            return bad(format!("{} streams x {BLOCK} rows exceeds the decode kernels' {CHUNK_ROWS} rows", self.max_streams));
        }
        if self.max_ctx_rows == 0 || self.slots == 0 || self.pages == 0 || self.head_stride < self.vocab {
            return bad(format!(
                "{} context rows, {} slots, {} pages, head stride {} for a {}-id vocabulary",
                self.max_ctx_rows, self.slots, self.pages, self.head_stride, self.vocab
            ));
        }
        if self.max_context + BLOCK > self.max_blocks * PAGE_TOKENS as usize {
            return bad(format!(
                "max_context {} + a block exceeds {} blocks of {PAGE_TOKENS}",
                self.max_context, self.max_blocks
            ));
        }
        Ok(())
    }
}

/// Scratch the linears share.
struct LinScratch {
    /// EXL3 decode workspace (every shape at up to 128 rows).
    ws: DevBuf,
    /// dense path: fp16 activations in and out, the cuBLAS workspace, and the overflow counter.
    dx: DevBuf,
    dy: DevBuf,
    cublas: DevBuf,
    bad: DevBuf,
}

/// Context-row scratch (one chunk of at most `max_ctx_rows` rows).
struct CtxBufs {
    fc_in: DevBuf,
    ctx: DevBuf,
    kproj: DevBuf,
    vproj: DevBuf,
    kn: DevBuf,
    qd: DevBuf,
    pos: DevBuf,
    rows: RowIndex,
}

/// Block-row scratch (`max_streams x 15` rows).
struct BlockBufs {
    ids: DevBuf,
    pos: DevBuf,
    anchors: DevBuf,
    kv_rows: RowIndex,
    att_rows: RowIndex,
    x: DevBuf,
    h: DevBuf,
    o: DevBuf,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    q2: DevBuf,
    k2: DevBuf,
    att: DevBuf,
    g: DevBuf,
    u: DevBuf,
    sw: DevBuf,
    attn_ws: DevBuf,
    logits: DevBuf,
    logits16: DevBuf,
    head_xh: DevBuf,
    head_ws: DevBuf,
    top_ids: DevBuf,
    top_vals: DevBuf,
    top_ws: DevBuf,
    ones: DevBuf,
    out_tokens: DevBuf,
    conf: DevBuf,
    walk: DevBuf,
}

fn z(n: usize) -> Result<DevBuf, ModelError> {
    Ok(DevBuf::zeroed_view(n)?)
}

fn cp(b: &DevBuf) -> *const c_void {
    b.ptr() as *const c_void
}

/// `y [rows, n] = lin(x [rows, k])`, bf16 in and out, in chunks of at most `CHUNK_ROWS` rows (a row's bits do not depend on the
/// chunking: the decode kernel is row invariant at 1..=128 rows). `x` and `y` must be views (`DevBuf::zeroed_view`) when `rows`
/// exceeds a chunk.
fn lin_rows(l: &Lin, x: &DevBuf, y: &mut DevBuf, rows: usize, sc: &mut LinScratch, st: &Stream) -> Result<(), ModelError> {
    let (k, n) = (l.k(), l.n());
    let step = match l {
        Lin::Exl3(_) => CHUNK_ROWS,
        Lin::Dense { .. } => DENSE_ROWS,
    };
    let mut r0 = 0;
    while r0 < rows {
        let m = (rows - r0).min(step);
        if r0 == 0 && m == rows {
            lin_chunk(l, x, y, m, k, n, sc, st)?;
        } else {
            let xs = x.subview(r0 * k * 2, m * k * 2);
            let mut ys = y.subview(r0 * n * 2, m * n * 2);
            lin_chunk(l, &xs, &mut ys, m, k, n, sc, st)?;
        }
        r0 += m;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lin_chunk(
    l: &Lin,
    x: &DevBuf,
    y: &mut DevBuf,
    m: usize,
    k: usize,
    n: usize,
    sc: &mut LinScratch,
    st: &Stream,
) -> Result<(), ModelError> {
    match l {
        Lin::Exl3(w) => w.forward(x, y, m, Dtype::BF16, Dtype::BF16, &mut sc.ws, st)?,
        Lin::Dense { w, .. } => {
            kern::bf16_to_f16_copy(x, &mut sc.dx, m * k, &mut sc.bad, st)?;
            glm_vision::gemm(&sc.dx, k, w, &mut sc.dy, n, m, k, n, false, &mut sc.cublas, st)?;
            kern::f16_to_bf16_copy(&sc.dy, y, m * n, st)?;
        }
    }
    Ok(())
}

impl DraftModel {
    /// Load the draft from `dir` (an EXL3 draft pack, or the bf16 checkpoint for the dense reference path) and allocate its pool and
    /// scratch. Returns the model and the bytes it holds (weights, pool and scratch: what the memory plan must account for).
    pub fn new(cfg: &DSparkConfig, dir: &Path, opts: DraftOptions) -> Result<(Self, usize), ModelError> {
        opts.check()?;
        let problems = config_problems(cfg, opts.vocab, opts.target_layers);
        if !problems.is_empty() {
            return Err(ModelError::Config(format!("draft config: {}", problems.join("; "))));
        }
        let pack = Pack::open(dir)?;
        let mut ctx = LoadCtx::from_env();
        let mut w = Weights::declare(&mut ctx, &pack, cfg)?;
        ctx.finish()?;
        let st = Stream::new()?;
        w.convert(&st)?;
        let dense = w.kind == WeightKind::Dense;
        let rows = opts.max_streams * BLOCK;
        let cr = opts.max_ctx_rows;
        let tap_w = cfg.target_layer_ids.len() * HIDDEN;
        let kv = DraftKv::new(opts.slots, opts.pages, opts.max_blocks)?;
        let shapes = [
            (tap_w, HIDDEN),
            (HIDDEN, HEADS * HEAD_DIM),
            (HIDDEN, KV_HEADS * HEAD_DIM),
            (HEADS * HEAD_DIM, HIDDEN),
            (HIDDEN, INTERMEDIATE),
            (INTERMEDIATE, HIDDEN),
        ];
        let ws_bytes = shapes.iter().map(|&(k, n)| linear_workspace_bytes(CHUNK_ROWS, k, n, 8)).max().unwrap_or(0);
        let max_k = shapes.iter().map(|s| s.0).max().unwrap_or(0);
        let max_n = shapes.iter().map(|s| s.1).max().unwrap_or(0);
        let (dxb, dyb) = if dense { (DENSE_ROWS * max_k * 2, DENSE_ROWS * max_n * 2) } else { (16, 16) };
        let hs = opts.head_stride;
        let dense_head = opts.head == HeadKind::Dense;
        let head_ws = match opts.head {
            HeadKind::Exl3 { bits } => linear_workspace_bytes(rows, HIDDEN, hs, bits),
            HeadKind::Dense => 16,
        };
        let sc = LinScratch {
            ws: z(ws_bytes)?,
            dx: z(dxb)?,
            dy: z(dyb)?,
            cublas: z(if dense || dense_head { 32 << 20 } else { 16 })?,
            bad: z(4)?,
        };
        let cx = CtxBufs {
            fc_in: z(cr * tap_w * 2)?,
            ctx: z(cr * HIDDEN * 2)?,
            kproj: z(cr * KV_HEADS * HEAD_DIM * 2)?,
            vproj: z(cr * KV_HEADS * HEAD_DIM * 2)?,
            kn: z(cr * KV_HEADS * HEAD_DIM * 2)?,
            qd: z(cr * KV_HEADS * HEAD_DIM * 2)?,
            pos: z(cr * 4)?,
            rows: RowIndex::from_host(&vec![0; cr], &vec![0; cr])?,
        };
        let bk = BlockBufs {
            ids: z(rows * 4)?,
            pos: z(rows * 4)?,
            anchors: z(opts.max_streams * 4)?,
            kv_rows: RowIndex::from_host(&vec![0; rows], &vec![0; rows])?,
            att_rows: RowIndex::from_host(&vec![0; rows], &vec![0; rows])?,
            x: z(rows * HIDDEN * 2)?,
            h: z(rows * HIDDEN * 2)?,
            o: z(rows * HIDDEN * 2)?,
            q: z(rows * HEADS * HEAD_DIM * 2)?,
            k: z(rows * KV_HEADS * HEAD_DIM * 2)?,
            v: z(rows * KV_HEADS * HEAD_DIM * 2)?,
            q2: z(rows * HEADS * HEAD_DIM * 2)?,
            k2: z(rows * KV_HEADS * HEAD_DIM * 2)?,
            att: z(rows * HEADS * HEAD_DIM * 2)?,
            g: z(rows * INTERMEDIATE * 2)?,
            u: z(rows * INTERMEDIATE * 2)?,
            sw: z(rows * INTERMEDIATE * 2)?,
            attn_ws: z(attention::decode_workspace_bytes_for(ATTN, rows, opts.max_context + BLOCK))?,
            logits: z(rows * hs * 2)?,
            logits16: z(if dense_head { rows * hs * 2 } else { 16 })?,
            head_xh: z(if dense_head { rows * HIDDEN * 2 } else { 16 })?,
            head_ws: z(head_ws)?,
            top_ids: z(rows * MAX_TOP_K * 4)?,
            top_vals: z(rows * MAX_TOP_K * 4)?,
            top_ws: z(glue::topk_valid_workspace_bytes(rows, opts.vocab, MAX_TOP_K).max(16))?,
            ones: {
                let mut b = z(rows * RANK * 4)?;
                b.upload(&vec![1.0f32; rows * RANK])?;
                b
            },
            out_tokens: z(rows * 4)?,
            conf: z(rows * 4)?,
            walk: z(opts.max_streams * std::mem::size_of::<WalkSampling>())?,
        };
        let model = Self { cfg: cfg.clone(), w, kv, len: vec![0; opts.slots], opts, sc, cx, bk, poison_pad: false };
        let bytes = model.weight_bytes() + model.kv.bytes() as usize + model.scratch_bytes();
        Ok((model, bytes))
    }

    /// Bytes of the scratch buffers (not weights, not the pool).
    pub fn scratch_bytes(&self) -> usize {
        let (s, c, b) = (&self.sc, &self.cx, &self.bk);
        let bufs = [
            &s.ws,
            &s.dx,
            &s.dy,
            &s.cublas,
            &s.bad,
            &c.fc_in,
            &c.ctx,
            &c.kproj,
            &c.vproj,
            &c.kn,
            &c.qd,
            &c.pos,
            &b.ids,
            &b.pos,
            &b.anchors,
            &b.x,
            &b.h,
            &b.o,
            &b.q,
            &b.k,
            &b.v,
            &b.q2,
            &b.k2,
            &b.att,
            &b.g,
            &b.u,
            &b.sw,
            &b.attn_ws,
            &b.logits,
            &b.logits16,
            &b.head_xh,
            &b.head_ws,
            &b.top_ids,
            &b.top_vals,
            &b.top_ws,
            &b.ones,
            &b.out_tokens,
            &b.conf,
            &b.walk,
        ];
        bufs.iter().map(|x| x.bytes()).sum::<usize>() + 4 * 2 * (c.rows.rows + b.kv_rows.rows + b.att_rows.rows)
    }

    /// Fill the head's pad columns (those past `valid`) with the largest finite bf16 before every top-K: a test of the pad mask. Costs a
    /// synchronisation and a few small copies per round; never on in serving.
    pub fn poison_pad_columns(&mut self, on: bool) {
        self.poison_pad = on;
    }

    pub fn weight_kind(&self) -> WeightKind {
        self.w.kind
    }

    /// Bytes of the weights (EXL3 trellis and scales or dense fp16, norms, Markov tables, confidence head).
    pub fn weight_bytes(&self) -> usize {
        self.w.bytes()
    }

    pub fn kv(&self) -> &DraftKv {
        &self.kv
    }

    pub fn options(&self) -> &DraftOptions {
        &self.opts
    }

    /// Committed context rows of a slot.
    pub fn len(&self, slot: usize) -> usize {
        self.len[slot]
    }

    pub fn is_empty(&self, slot: usize) -> bool {
        self.len[slot] == 0
    }

    // -------------------------------------------------------------------------------------------- slots and rollback

    /// Start a sequence in `slot` with the page list `blocks` (the engine's block ids for positions `256 i ..= 256 i + 255`): the
    /// committed length is 0. Pages for positions up to `P + 14` of every round must be in the list before `propose`.
    pub fn begin_slot(&mut self, slot: usize, blocks: &[u32]) -> Result<(), ModelError> {
        self.set_blocks(slot, blocks)?;
        self.len[slot] = 0;
        Ok(())
    }

    /// Replace a slot's page list (the engine grew the sequence); the committed length is unchanged.
    pub fn set_blocks(&mut self, slot: usize, blocks: &[u32]) -> Result<(), ModelError> {
        if slot >= self.opts.slots {
            return Err(ModelError::Request(format!("draft slot {slot} of {}", self.opts.slots)));
        }
        if blocks.len() > self.kv.max_blocks() {
            return Err(ModelError::Request(format!("{} blocks, the table holds {}", blocks.len(), self.kv.max_blocks())));
        }
        if let Some(&b) = blocks.iter().find(|&&b| b as usize >= self.kv.pages) {
            return Err(ModelError::Request(format!("block {b} of a {}-page draft pool", self.kv.pages)));
        }
        self.kv.slot_blocks[slot] = blocks.to_vec();
        let tables = self.kv.slot_blocks.clone();
        self.kv.table.upload(&tables)?;
        Ok(())
    }

    /// Roll the committed length of `slot` back to `len` rows. Exact and free: rows past `len` are never read (every block row's
    /// bound is `P + 14` with `P` the new committed length, and the block rewrites `P ..= P + 14` before its attention), and the
    /// replay of the same taps rewrites the same bits.
    pub fn rollback(&mut self, slot: usize, len: usize) -> Result<(), ModelError> {
        match self.len.get(slot) {
            Some(&cur) if len <= cur => {
                self.len[slot] = len;
                Ok(())
            }
            Some(&cur) => Err(ModelError::Request(format!("rollback of slot {slot} to {len} rows, committed {cur}"))),
            None => Err(ModelError::Request(format!("draft slot {slot} of {}", self.opts.slots))),
        }
    }

    /// Copy page `src` of every layer's K and V into page `dst` (the copy-on-write of the partial last page of a prefix-cache restore:
    /// the draft's page ids are the target's, so the target's copy of a page and the draft's go together). `st` orders it with the
    /// caller's other work.
    pub fn copy_page(&mut self, src: usize, dst: usize, st: &Stream) -> Result<(), ModelError> {
        if src >= self.kv.pages || dst >= self.kv.pages {
            return Err(ModelError::Request(format!("draft page copy {src} -> {dst} in a pool of {}", self.kv.pages)));
        }
        for layer in self.kv.layers.iter_mut() {
            let bytes = layer.block_bytes();
            layer.k.copy_within(dst * bytes, src * bytes, bytes, st)?;
            layer.v.copy_within(dst * bytes, src * bytes, bytes, st)?;
        }
        Ok(())
    }

    /// Declare that rows `0..len` of `slot` are already in its pages: a prefix-cache restore that shared a snapshot's draft pages by
    /// reference (E6b's seam; the draft's page ids are the target's). The committed length becomes `len`; the caller owns the claim and
    /// must have installed the slot's page list ([`DraftModel::set_blocks`]) first.
    pub fn adopt_len(&mut self, slot: usize, len: usize) -> Result<(), ModelError> {
        if slot >= self.opts.slots {
            return Err(ModelError::Request(format!("draft slot {slot} of {}", self.opts.slots)));
        }
        let cap = self.kv.slot_blocks[slot].len() * PAGE_TOKENS as usize;
        if len > cap {
            return Err(ModelError::Request(format!(
                "adopting {len} context rows into slot {slot}, which holds pages for {cap}"
            )));
        }
        self.len[slot] = len;
        Ok(())
    }

    // -------------------------------------------------------------------------------------------- context rows

    /// Absorb tap rows: `rows` rows of `taps` (bf16 `[*, taps x 2048]`, the target's residual stream after each tap layer in ascending
    /// layer order, row-major, starting at row `taps_row0`) for positions `pos0 .. pos0 + rows` of `slot`. Writes each layer's K/V
    /// rows into the slot's pages and sets the committed length to `pos0 + rows`. `pos0` may not leave a hole past the committed
    /// length; re-ingesting already committed rows rewrites the same bits.
    pub fn ingest_taps(
        &mut self,
        slot: usize,
        pos0: usize,
        rows: usize,
        taps: &DevBuf,
        taps_row0: usize,
        st: &Stream,
    ) -> Result<(), ModelError> {
        self.ingest_taps_debug(slot, pos0, rows, taps, taps_row0, st, None)
    }

    /// [`DraftModel::ingest_taps`] that also reads back the rows after `fc` and after `hidden_norm` (the first chunk only: a test
    /// ingests at most `max_ctx_rows` rows).
    #[allow(clippy::too_many_arguments)]
    pub fn ingest_taps_debug(
        &mut self,
        slot: usize,
        pos0: usize,
        rows: usize,
        taps: &DevBuf,
        taps_row0: usize,
        st: &Stream,
        mut dump: Option<&mut CtxDump>,
    ) -> Result<(), ModelError> {
        if slot >= self.opts.slots {
            return Err(ModelError::Request(format!("draft slot {slot} of {}", self.opts.slots)));
        }
        if pos0 > self.len[slot] {
            return Err(ModelError::Request(format!(
                "ingest at position {pos0} leaves a hole: slot {slot} holds {} context rows",
                self.len[slot]
            )));
        }
        let tap_w = self.cfg.target_layer_ids.len() * HIDDEN;
        if taps.bytes() < (taps_row0 + rows) * tap_w * 2 {
            return Err(ModelError::Request(format!(
                "tap buffer of {} bytes holds fewer than {} rows of {tap_w}",
                taps.bytes(),
                taps_row0 + rows
            )));
        }
        let cap = self.kv.slot_blocks[slot].len() * PAGE_TOKENS as usize;
        if pos0 + rows > cap {
            return Err(ModelError::Request(format!("positions up to {} but slot {slot} holds pages for {cap}", pos0 + rows)));
        }
        let Self { w, kv, sc, cx, opts, .. } = self;
        let mut c0 = 0;
        while c0 < rows {
            let c = (rows - c0).min(opts.max_ctx_rows);
            // the chunk's tap rows into the fc input (a view the chunked linear can cut)
            unsafe {
                glue::copy_rows(taps.ptr_at((taps_row0 + c0) * tap_w * 2), tap_w, cx.fc_in.ptr(), tap_w, c, tap_w, st)?;
            }
            lin_rows(&w.fc, &cx.fc_in, &mut cx.ctx, c, sc, st)?;
            if let Some(d) = dump.as_deref_mut().filter(|_| c0 == 0) {
                st.sync()?;
                d.fc = vec![0u16; c * HIDDEN];
                cx.ctx.download_range(0, &mut d.fc)?;
            }
            unsafe {
                glue::rmsnorm(cp(&cx.ctx), cp(&w.hidden_norm), cx.ctx.ptr(), c, HIDDEN, EPS, st)?;
            }
            if let Some(d) = dump.as_deref_mut().filter(|_| c0 == 0) {
                st.sync()?;
                d.hidden_norm = vec![0u16; c * HIDDEN];
                cx.ctx.download_range(0, &mut d.hidden_norm)?;
            }
            let pos: Vec<i32> = (0..c).map(|i| (pos0 + c0 + i) as i32).collect();
            cx.pos.upload(&pos)?;
            cx.rows.upload(&vec![slot as i32; c], &pos)?;
            for (l, layer) in w.layers.iter().enumerate() {
                lin_rows(&layer.k, &cx.ctx, &mut cx.kproj, c, sc, st)?;
                lin_rows(&layer.v, &cx.ctx, &mut cx.vproj, c, sc, st)?;
                unsafe {
                    glue::attn_prep(
                        &CTX_PREP,
                        cp(&cx.kproj),
                        cp(&cx.kproj),
                        cp(&cx.pos),
                        cp(&layer.k_norm),
                        cp(&layer.k_norm),
                        cx.qd.ptr(),
                        std::ptr::null_mut(),
                        cx.kn.ptr(),
                        c,
                        st,
                    )?;
                }
                attention::kv_write(&cx.kn, &cx.vproj, &cx.rows, &kv.table, &mut kv.layers[l], c, st)?;
            }
            c0 += c;
        }
        self.len[slot] = self.len[slot].max(pos0 + rows);
        Ok(())
    }

    /// Absorb the tap rows of SEVERAL slots in one pass (the engine's round: every drafting stream committed a few rows since its last
    /// round): the segments' rows go through `fc`, `hidden_norm` and each layer's K and V projections together, and each row is written
    /// to its own slot's pages at its own position. A row's bits do not depend on the other rows of the pass (every linear is the
    /// row-invariant decode kernel in chunks of at most 128 rows; the prep and the page write are per row), so the pages hold exactly what
    /// one [`DraftModel::ingest_taps`] call per slot would have written. At most `max_ctx_rows` rows in all, no slot twice, no hole.
    pub fn ingest_taps_batch(&mut self, segs: &[IngestSeg], taps: &DevBuf, st: &Stream) -> Result<(), ModelError> {
        let total: usize = segs.iter().map(|g| g.rows).sum();
        if segs.is_empty() || total == 0 {
            return Ok(());
        }
        if total > self.opts.max_ctx_rows {
            return Err(ModelError::Request(format!(
                "a batched ingestion of {total} rows; the draft takes {}",
                self.opts.max_ctx_rows
            )));
        }
        let tap_w = self.cfg.target_layer_ids.len() * HIDDEN;
        for (i, g) in segs.iter().enumerate() {
            if g.slot >= self.opts.slots || segs[..i].iter().any(|o| o.slot == g.slot) {
                return Err(ModelError::Request(format!(
                    "draft slot {} is out of range or repeated in a batched ingestion",
                    g.slot
                )));
            }
            if g.pos0 > self.len[g.slot] {
                return Err(ModelError::Request(format!(
                    "ingest at position {} leaves a hole: slot {} holds {} context rows",
                    g.pos0, g.slot, self.len[g.slot]
                )));
            }
            let cap = self.kv.slot_blocks[g.slot].len() * PAGE_TOKENS as usize;
            if g.pos0 + g.rows > cap {
                return Err(ModelError::Request(format!(
                    "positions up to {} but slot {} holds pages for {cap}",
                    g.pos0 + g.rows,
                    g.slot
                )));
            }
            if taps.bytes() < (g.taps_row0 + g.rows) * tap_w * 2 {
                return Err(ModelError::Request(format!(
                    "tap buffer of {} bytes holds fewer than {} rows of {tap_w}",
                    taps.bytes(),
                    g.taps_row0 + g.rows
                )));
            }
        }
        let Self { w, kv, sc, cx, .. } = self;
        let c = total;
        let mut off = 0usize;
        for g in segs {
            // SAFETY: `g.rows` rows of `tap_w` bf16 inside the tap buffer (checked above) into the fc input, which holds `max_ctx_rows`.
            unsafe {
                glue::copy_rows(
                    taps.ptr_at(g.taps_row0 * tap_w * 2),
                    tap_w,
                    cx.fc_in.ptr_at(off * tap_w * 2),
                    tap_w,
                    g.rows,
                    tap_w,
                    st,
                )?;
            }
            off += g.rows;
        }
        lin_rows(&w.fc, &cx.fc_in, &mut cx.ctx, c, sc, st)?;
        unsafe {
            glue::rmsnorm(cp(&cx.ctx), cp(&w.hidden_norm), cx.ctx.ptr(), c, HIDDEN, EPS, st)?;
        }
        let (mut slots, mut pos) = (Vec::with_capacity(c), Vec::with_capacity(c));
        for g in segs {
            for i in 0..g.rows {
                slots.push(g.slot as i32);
                pos.push((g.pos0 + i) as i32);
            }
        }
        cx.pos.upload(&pos)?;
        cx.rows.upload(&slots, &pos)?;
        for (l, layer) in w.layers.iter().enumerate() {
            lin_rows(&layer.k, &cx.ctx, &mut cx.kproj, c, sc, st)?;
            lin_rows(&layer.v, &cx.ctx, &mut cx.vproj, c, sc, st)?;
            unsafe {
                glue::attn_prep(
                    &CTX_PREP,
                    cp(&cx.kproj),
                    cp(&cx.kproj),
                    cp(&cx.pos),
                    cp(&layer.k_norm),
                    cp(&layer.k_norm),
                    cx.qd.ptr(),
                    std::ptr::null_mut(),
                    cx.kn.ptr(),
                    c,
                    st,
                )?;
            }
            attention::kv_write(&cx.kn, &cx.vproj, &cx.rows, &kv.table, &mut kv.layers[l], c, st)?;
        }
        for g in segs {
            self.len[g.slot] = self.len[g.slot].max(g.pos0 + g.rows);
        }
        Ok(())
    }

    /// The K and V rows (`[rows, 2, 256]` bf16 bits each) of `layer` at positions `pos0 ..` of `slot`, read back from its pages.
    pub fn read_kv(&self, slot: usize, layer: usize, pos0: usize, rows: usize) -> Result<(Vec<u16>, Vec<u16>), ModelError> {
        let blocks = &self.kv.slot_blocks[slot];
        let te = KV_HEADS * HEAD_DIM;
        let (mut k, mut v) = (vec![0u16; rows * te], vec![0u16; rows * te]);
        for r in 0..rows {
            let p = pos0 + r;
            let blk = *blocks
                .get(p / PAGE_TOKENS as usize)
                .ok_or_else(|| ModelError::Request(format!("position {p} has no page")))? as usize;
            let off = (blk * PAGE_TOKENS as usize + p % PAGE_TOKENS as usize) * te * 2;
            self.kv.layers[layer].k.download_range(off, &mut k[r * te..(r + 1) * te])?;
            self.kv.layers[layer].v.download_range(off, &mut v[r * te..(r + 1) * te])?;
        }
        Ok((k, v))
    }

    // -------------------------------------------------------------------------------------------- the block

    /// One draft round for up to `max_streams` streams: 15 drafted tokens and their confidences per stream. `embed` is the target's
    /// bf16 embedding table `[vocab, 2048]`, `head` its vocabulary head, `k_top` (1..=64) the candidates per row the Markov walk
    /// takes. Every stream's context (`0..P`) must have been ingested (`len(slot) == p`). Synchronises `stream` and returns.
    pub fn propose(
        &mut self,
        embed: &DevBuf,
        head: &Head<'_>,
        reqs: &[ProposeReq<'_>],
        k_top: usize,
        stream: &Stream,
    ) -> Result<Vec<Proposal>, ModelError> {
        self.propose_debug(embed, head, reqs, k_top, stream, None)
    }

    /// [`DraftModel::propose`] that also reads back the block's activations (the stage tests).
    #[allow(clippy::too_many_arguments)]
    pub fn propose_debug(
        &mut self,
        embed: &DevBuf,
        head: &Head<'_>,
        reqs: &[ProposeReq<'_>],
        k_top: usize,
        st: &Stream,
        mut dump: Option<&mut StageDump>,
    ) -> Result<Vec<Proposal>, ModelError> {
        let n = reqs.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > self.opts.max_streams {
            return Err(ModelError::Request(format!("a draft round of {n} streams; the draft takes {}", self.opts.max_streams)));
        }
        if !(1..=MAX_TOP_K).contains(&k_top) || k_top > head.valid() {
            return Err(ModelError::Request(format!("k_top {k_top}: 1..={MAX_TOP_K} candidates per row")));
        }
        let kind = match head {
            Head::Exl3 { w, .. } => HeadKind::Exl3 { bits: w.bits },
            Head::Dense { .. } => HeadKind::Dense,
        };
        if kind != self.opts.head || head.stride() != self.opts.head_stride {
            return Err(ModelError::Request(format!(
                "a {kind:?} head of {} columns for a draft built for {:?} of {} columns",
                head.stride(),
                self.opts.head,
                self.opts.head_stride
            )));
        }
        if head.valid() != self.opts.vocab {
            return Err(ModelError::Request(format!(
                "head of {} columns ({} valid) for a draft sized for {} ({} ids)",
                head.stride(),
                head.valid(),
                self.opts.head_stride,
                self.opts.vocab
            )));
        }
        for (i, r) in reqs.iter().enumerate() {
            if r.slot >= self.opts.slots || reqs[..i].iter().any(|o| o.slot == r.slot) {
                return Err(ModelError::Request(format!("stream {i}: slot {} is out of range or repeated in the round", r.slot)));
            }
            if self.len[r.slot] != r.p {
                return Err(ModelError::Request(format!(
                    "stream {i}: anchor position {} but slot {} holds {} context rows",
                    r.p, r.slot, self.len[r.slot]
                )));
            }
            let cap = self.kv.slot_blocks[r.slot].len() * PAGE_TOKENS as usize;
            if r.p + BLOCK > cap || r.p + BLOCK > self.opts.max_context + BLOCK {
                return Err(ModelError::Request(format!(
                    "stream {i}: position {} + a {BLOCK}-row block past {cap} cached positions / max_context {}",
                    r.p, self.opts.max_context
                )));
            }
        }
        let rows = n * BLOCK;
        let mask = self.cfg.mask_token_id as i32;
        let (mut ids, mut pos, mut kv_slot, mut att_pos) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for r in reqs {
            ids.push(r.anchor as i32);
            ids.extend(std::iter::repeat_n(mask, BLOCK - 1));
            for j in 0..BLOCK {
                pos.push((r.p + j) as i32);
                kv_slot.push(r.slot as i32);
                att_pos.push((r.p + BLOCK - 1) as i32); // every row sees the whole block: the bound is the last row's position
            }
        }
        let anchors: Vec<i32> = reqs.iter().map(|r| r.anchor as i32).collect();
        let walks: Vec<WalkSampling> = reqs
            .iter()
            .map(|r| WalkSampling {
                seed: r.sampling.seed,
                first_position: r.p as u64 + 1,
                temperature: r.sampling.temperature,
                top_p: r.sampling.top_p,
                min_p: r.sampling.min_p,
                keyed: (r.sampling.temperature > 0.0) as u32,
                pad: 0,
            })
            .collect();
        let max_ctx = reqs.iter().map(|r| r.p).max().unwrap_or(0) + BLOCK;

        let Self { cfg, w, kv, sc, bk, poison_pad, .. } = self;
        bk.ids.upload(&ids)?;
        bk.pos.upload(&pos)?;
        bk.anchors.upload(&anchors)?;
        bk.kv_rows.upload(&kv_slot, &pos)?;
        bk.att_rows.upload(&kv_slot, &att_pos)?;
        bk.walk.upload(&walks)?;
        let vocab = cfg.vocab;
        let rd = |b: &DevBuf, n: usize| -> Result<Vec<u16>, ModelError> {
            let mut v = vec![0u16; n];
            b.download_range(0, &mut v)?;
            Ok(v)
        };
        unsafe {
            glue::embed(cp(embed), cp(&bk.ids), bk.x.ptr(), rows, HIDDEN, vocab, st)?;
        }
        if let Some(d) = dump.as_deref_mut() {
            st.sync()?;
            d.block_embed = rd(&bk.x, rows * HIDDEN)?;
        }
        unsafe {
            glue::rmsnorm(cp(&bk.x), cp(&w.layers[0].input_norm), bk.h.ptr(), rows, HIDDEN, EPS, st)?;
        }
        for (l, layer) in w.layers.iter().enumerate() {
            lin_rows(&layer.q, &bk.h, &mut bk.q, rows, sc, st)?;
            lin_rows(&layer.k, &bk.h, &mut bk.k, rows, sc, st)?;
            lin_rows(&layer.v, &bk.h, &mut bk.v, rows, sc, st)?;
            unsafe {
                glue::attn_prep(
                    &BLOCK_PREP,
                    cp(&bk.q),
                    cp(&bk.k),
                    cp(&bk.pos),
                    cp(&layer.q_norm),
                    cp(&layer.k_norm),
                    bk.q2.ptr(),
                    std::ptr::null_mut(),
                    bk.k2.ptr(),
                    rows,
                    st,
                )?;
            }
            attention::kv_write(&bk.k2, &bk.v, &bk.kv_rows, &kv.table, &mut kv.layers[l], rows, st)?;
            attention::attn_decode(
                &bk.q2,
                &kv.layers[l],
                &kv.table,
                &bk.att_rows,
                rows,
                &mut bk.att,
                max_ctx,
                &mut bk.attn_ws,
                st,
            )?;
            lin_rows(&layer.o, &bk.att, &mut bk.o, rows, sc, st)?;
            // x = bf16(o + x) is the residual after attention; h = post_attention_layernorm(x)
            unsafe {
                glue::add_rmsnorm(cp(&bk.o), cp(&bk.x), cp(&layer.post_norm), bk.x.ptr(), bk.h.ptr(), rows, HIDDEN, EPS, st)?;
            }
            lin_rows(&layer.gate, &bk.h, &mut bk.g, rows, sc, st)?;
            lin_rows(&layer.up, &bk.h, &mut bk.u, rows, sc, st)?;
            unsafe {
                glue::swiglu(cp(&bk.g), cp(&bk.u), bk.sw.ptr(), rows * INTERMEDIATE, st)?;
            }
            lin_rows(&layer.down, &bk.sw, &mut bk.o, rows, sc, st)?;
            // x = bf16(mlp + x) is the layer's output; h = the next layer's input_layernorm (or the final norm) of it
            let next = if l + 1 < LAYERS { &w.layers[l + 1].input_norm } else { &w.norm };
            unsafe {
                glue::add_rmsnorm(cp(&bk.o), cp(&bk.x), cp(next), bk.x.ptr(), bk.h.ptr(), rows, HIDDEN, EPS, st)?;
            }
            if let Some(d) = dump.as_deref_mut() {
                st.sync()?;
                d.layer_out.push(rd(&bk.x, rows * HIDDEN)?);
            }
        }
        if let Some(d) = dump.as_deref_mut() {
            st.sync()?;
            d.final_norm = rd(&bk.h, rows * HIDDEN)?;
        }
        // the block's rows through the target's head; the candidates are the top-K of the BASE logits
        let stride = head.stride();
        match head {
            Head::Exl3 { w: hw, .. } => {
                // the workspace was sized at construction for `HeadKind::Exl3 { bits }` (checked at the top of this call)
                hw.forward(&bk.h, &mut bk.logits, rows, Dtype::BF16, Dtype::BF16, &mut bk.head_ws, st)?;
            }
            Head::Dense { w: hw, .. } => {
                kern::bf16_to_f16_copy(&bk.h, &mut bk.head_xh, rows * HIDDEN, &mut sc.bad, st)?;
                glm_vision::gemm(
                    &bk.head_xh,
                    HIDDEN,
                    hw,
                    &mut bk.logits16,
                    stride,
                    rows,
                    HIDDEN,
                    stride,
                    false,
                    &mut sc.cublas,
                    st,
                )?;
                kern::f16_to_bf16_copy(&bk.logits16, &mut bk.logits, rows * stride, st)?;
            }
        }
        if *poison_pad && stride > head.valid() {
            // the test knob: the head's pad columns hold the largest finite bf16, so a top-K that read them would return a pad id
            st.sync()?;
            let pad = stride - head.valid();
            let junk = vec![0x7f7fu16; pad];
            for r in 0..rows {
                bk.logits.subview((r * stride + head.valid()) * 2, pad * 2).upload(&junk)?;
            }
        }
        unsafe {
            glue::topk_valid(
                cp(&bk.logits),
                rows,
                stride,
                head.valid(),
                k_top,
                bk.top_ids.ptr(),
                bk.top_vals.ptr(),
                bk.top_ws.ptr(),
                bk.top_ws.bytes(),
                st,
            )?;
        }
        dflash2::selector_walk_keyed(
            &bk.anchors,
            &bk.top_ids,
            &bk.top_vals,
            &bk.ones,
            GateDtype::F32,
            &w.markov_w1,
            &w.markov_w2,
            n,
            BLOCK,
            k_top,
            RANK,
            &mut bk.out_tokens,
            Some(&bk.walk),
            None,
            st,
        )?;
        kern::confidence(
            &bk.h,
            &bk.out_tokens,
            &bk.anchors,
            &w.markov_w1,
            &w.conf_w,
            &w.conf_b,
            &mut bk.conf,
            n,
            BLOCK,
            HIDDEN,
            RANK,
            vocab,
            st,
        )?;
        st.sync()?;
        if let Some(d) = dump {
            d.stride = stride;
            d.logits = rd(&bk.logits, rows * stride)?;
            d.cand_ids = vec![0i32; rows * k_top];
            bk.top_ids.download_range(0, &mut d.cand_ids)?;
            d.cand_vals = vec![0f32; rows * k_top];
            bk.top_vals.download_range(0, &mut d.cand_vals)?;
        }
        let mut toks = vec![0i32; rows];
        bk.out_tokens.download_range(0, &mut toks)?;
        let mut conf = vec![0f32; rows];
        bk.conf.download_range(0, &mut conf)?;
        Ok((0..n)
            .map(|s| Proposal {
                tokens: toks[s * BLOCK..(s + 1) * BLOCK].iter().map(|&t| t as u32).collect(),
                confidence: conf[s * BLOCK..(s + 1) * BLOCK].to_vec(),
            })
            .collect())
    }
}

/// The draft on a device: weights, pool and scratch.
pub struct DraftModel {
    pub cfg: DSparkConfig,
    w: Weights,
    kv: DraftKv,
    opts: DraftOptions,
    /// Committed context rows per slot (the draft's view of the committed length).
    len: Vec<usize>,
    sc: LinScratch,
    cx: CtxBufs,
    bk: BlockBufs,
    /// Test knob ([`DraftModel::poison_pad_columns`]).
    poison_pad: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::minicpmv4_7::config::tests::{draft, real};
    use crate::families::minicpmv4_7::memory::allocations;

    #[test]
    fn the_real_draft_config_has_no_problems() {
        let c = draft();
        assert!(config_problems(&c, 248_144, 40).is_empty(), "{:?}", config_problems(&c, 248_144, 40));
        assert_eq!(KV_BYTES_PER_TOKEN, 10_240);
        assert_eq!(VERIFY_ROWS, 16);
        assert_eq!(c.block_size, BLOCK);
        assert_eq!(c.target_layer_ids, vec![3, 11, 19, 27, 35]);
    }

    #[test]
    fn every_misreading_of_the_config_is_named() {
        let base = draft();
        type Edit = Box<dyn Fn(&mut DSparkConfig)>;
        let cases: Vec<(&str, Edit, &str)> = vec![
            ("a partial rotary", Box::new(|c| c.rope_dim = 64), "rope_dim"),
            ("a sliding window", Box::new(|c| c.sliding_window = Some(4096)), "sliding-window"),
            ("a 14-row block", Box::new(|c| c.block_size = 14), "block_size"),
            ("another kv head count", Box::new(|c| c.kv_heads = 4), "num_key_value_heads"),
            ("taps past the target", Box::new(|c| c.target_layer_ids = vec![3, 11, 19, 27, 40]), "names layer 40"),
            ("unsorted taps", Box::new(|c| c.target_layer_ids = vec![11, 3, 19, 27, 35]), "ascending"),
            ("a mask id outside the vocabulary", Box::new(|c| c.mask_token_id = 248_144), "mask_token_id"),
            ("no confidence head", Box::new(|c| c.enable_confidence_head = false), "confidence head"),
            ("a reduced draft vocabulary", Box::new(|c| c.draft_vocab = 100_000), "draft_vocab_size"),
            ("rope theta 1e4", Box::new(|c| c.rope_theta = 1e4), "rope_theta"),
        ];
        for (what, f, needle) in cases {
            let mut c = base.clone();
            f(&mut c);
            let p = config_problems(&c, 248_144, 40);
            assert!(p.iter().any(|m| m.contains(needle)), "{what}: {p:?}");
        }
    }

    #[test]
    fn the_priced_page_is_the_pool_arithmetic() {
        let items = allocations(&real(), Some(&draft()));
        let it = items.iter().find(|i| i.name == "kv_draft").unwrap();
        assert_eq!(it.kind, AllocKind::PerPage);
        assert_eq!(it.bytes, PAGE_TOKENS * KV_BYTES_PER_TOKEN as u64);
        assert_eq!(it.bytes, 2_621_440); // one 256-token page of five layers' K and V
    }

    #[test]
    fn options_are_checked_by_name() {
        let ok = DraftOptions {
            slots: 4,
            pages: 64,
            max_streams: 4,
            max_ctx_rows: 512,
            max_context: 4096,
            max_blocks: 32,
            vocab: 248_144,
            target_layers: 40,
            head_stride: 248_192,
            head: HeadKind::Exl3 { bits: 6 },
        };
        assert!(ok.check().is_ok());
        let mut o = ok.clone();
        o.max_streams = 9;
        assert!(o.check().unwrap_err().to_string().contains("128 rows"));
        let mut o = ok.clone();
        o.max_context = 8192;
        assert!(o.check().unwrap_err().to_string().contains("max_context"));
        let mut o = ok;
        o.head_stride = 1000;
        assert!(o.check().unwrap_err().to_string().contains("head stride"));
    }
}
