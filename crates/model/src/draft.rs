//! The DFlash2 draft model (spec section 8): weights, the per-slot context cache, and one draft round.
//!
//! A round, for one sequence at position `P` (the pending anchor token is at `P`; the draft context covers `0..P`):
//!
//! 1. Absorb the target's tap rows for the tokens committed since the last round: `ctx = hidden_norm(fc(taps))`, then
//!    per draft layer the context K/V (`k = rope(norm(k_proj(ctx)))`, `v = v_proj(ctx)`) go into the slot's ring.
//! 2. Run the 5-layer block on `[anchor, MASK x 7]` at positions `P..P+7` (dynamic convs, causal in-block attention
//!    over the ring plus the block's own K/V), then the target's own head on rows 1..7, top-16 per row, and the
//!    selector's chain walk, which picks the 7 draft tokens.

use crate::error::ModelError;
use crate::loadctx::{push_linear, LoadCtx};
use crate::pack::Pack;
use core::ffi::c_void;
use gb_cuda::dflash2::{self, DraftCache, DraftCtxLayout, GateDtype, WalkSampling};
use gb_cuda::exl3_linear::{linear_workspace_bytes, prefill_scratch_bytes, Dtype, Exl3Weights};
use gb_cuda::{glue, DevBuf, Stream};

pub const BLOCK: usize = 8;
/// Most sequences one batched draft round carries (its rows go through the decode kernels, which take at most 128).
pub const MAX_DRAFT_SEQS: usize = 16;
pub const DRAFTS: usize = BLOCK - 1;
/// The least and most rows a draft block may run (`REXL3_DRAFT_BLOCK`, for the depth study; the pack was trained with 8,
/// which stays the default and what `BLOCK`/`DRAFTS` say). A 16-row block halves the batched round to 8 sequences
/// (`MAX_DRAFT_SEQS` blocks of rows still fit the decode kernels' 128), so it is opt-in.
pub const MIN_DRAFT_BLOCK: usize = 8;
pub const MAX_DRAFT_BLOCK: usize = 16;

/// `REXL3_DRAFT_BLOCK`: the draft block's rows, 8..=16 (the trained value when unset). Only the depth study sets it;
/// the engine's chains stay at `DRAFTS` deep whatever it says (`arbiter::MAX_DEPTH` does not move).
pub fn draft_block_from_env() -> Result<usize, ModelError> {
    match std::env::var("REXL3_DRAFT_BLOCK").ok().as_deref() {
        None | Some("") => Ok(BLOCK),
        Some(v) => {
            let b = v.trim().parse::<usize>().map_err(|_| {
                ModelError::Config(format!(
                    "REXL3_DRAFT_BLOCK={v:?}: expected a number from {MIN_DRAFT_BLOCK} to {MAX_DRAFT_BLOCK}"
                ))
            })?;
            if (MIN_DRAFT_BLOCK..=MAX_DRAFT_BLOCK).contains(&b) {
                Ok(b)
            } else {
                Err(ModelError::Config(format!(
                    "REXL3_DRAFT_BLOCK={v:?}: expected a number from {MIN_DRAFT_BLOCK} to {MAX_DRAFT_BLOCK}"
                )))
            }
        }
    }
}
/// One sequence's drafts with the draft's own confidence in each: `score` is the winning selector score (on the target head's
/// logit scale), `margin` the winner's lead over the runner-up in what the pick maximised (the noised score in temperature units,
/// or the raw gap when greedy; at most 30). Each is as long as the number of drafts the sequence asked for.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Picked {
    pub tokens: Vec<u32>,
    pub score: Vec<f32>,
    pub margin: Vec<f32>,
    /// The selector's candidate lattice behind these picks, when the draft was asked to capture it (`Draft::capture_lattice`).
    pub lattice: Option<Lattice>,
}

/// One sequence's selector lattice for a draft round (see `gb_spec::tree::Lattice`): per draft row the head's top-16 candidates with their
/// unary scores and the dot terms between a node's token and each candidate of the next row. The walk's score of candidate `c` of row `i`
/// after pick `parent` is `unary[i][c] + edge[i][parent][c]`, bit for bit (`gb_dflash_selector_edges`). The tree policy and the S0 study
/// read it.
pub use gb_spec::tree::Lattice;
pub const HIDDEN: usize = 5120;
pub const LAYERS: usize = 5;
pub const Q_HEADS: usize = 32;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const INTERMEDIATE: usize = 17408;
pub const CONV_GROUP: usize = 16;
pub const SELECTOR_RANK: usize = 256;
pub const SELECTOR_TOP_K: usize = 16;
pub const MASK_TOKEN: i32 = 248070;
pub const EPS: f32 = 1e-6;
pub const ROPE_THETA: f64 = 1e7;
/// Tap rows per fc input row: 5 taps x 5120.
pub const FC_IN: usize = 5 * HIDDEN;
/// Context positions the draft can ever look back over (its window plus a block of slack).
pub const CTX_KEEP: usize = DraftCtxLayout::WINDOW + 16;

struct ConvPair {
    /// f16 `[2, 2, 5120]`
    base: DevBuf,
    /// f16 `[1280, 5120]`
    proj: DevBuf,
}

struct DraftLayer {
    input_norm: DevBuf,
    post_norm: DevBuf,
    q: Exl3Weights,
    k: Exl3Weights,
    v: Exl3Weights,
    o: Exl3Weights,
    q_norm: DevBuf,
    k_norm: DevBuf,
    gate: Exl3Weights,
    up: Exl3Weights,
    down: Exl3Weights,
    attn_conv: ConvPair,
    mlp_conv: ConvPair,
}

/// One sequence's context rows to absorb: `n` tap rows from `row0` of the target's tap buffer, at positions `pos0..` of
/// `slot`'s ring.
#[derive(Clone, Copy, Debug)]
pub struct Absorb {
    pub row0: usize,
    pub n: usize,
    pub slot: usize,
    pub pos0: usize,
}

/// One sequence in a draft round: the anchor token (the last one, not yet fed) at position `p` of `slot`, and the
/// request's sampling (the walk draws with the target's keyed sampler on sampled requests).
#[derive(Clone, Copy, Debug)]
pub struct DraftReq<'a> {
    pub anchor: u32,
    pub slot: usize,
    pub p: usize,
    pub sampling: &'a gb_sampling::Sampling,
    /// How many drafts to return (1..=7). The block always runs its 8 rows through the layers; only the first `want` picks are
    /// computed (head rows, top-k and walk steps), and they are the same picks a full round would make first.
    pub want: usize,
    /// Share of the sequence's recent tokens that lie inside the draft's restricted vocabulary (see [`Draft::coverage`]); only an
    /// adaptive draft head reads it. 1.0 when nothing is restricted.
    pub coverage: f32,
}

/// Recent tokens an adaptive draft head looks at to tell whether a sequence is writing text its restricted vocabulary can reach.
pub const COVERAGE_WINDOW: usize = 64;

/// Whether a token id lies in `ranges`, `(first id, length)` pairs.
fn in_ranges(ranges: &[(usize, usize)], id: u32) -> bool {
    ranges.iter().any(|&(first, len)| (first..first + len).contains(&(id as usize)))
}

/// Share of `ids` inside `ranges` (1.0 for an empty list: nothing the set cannot reach).
pub fn coverage_of(ranges: &[(usize, usize)], ids: &[u32]) -> f32 {
    if ids.is_empty() {
        return 1.0;
    }
    ids.iter().filter(|&&t| in_ranges(ranges, t)).count() as f32 / ids.len() as f32
}

/// A restricted candidate set for the draft's vocabulary head: the target's `lm_head` columns of a few ranges of token ids, so the
/// head reads a fraction of its 0.95 GB each round. The verify path still uses the full head; this only decides which tokens the
/// draft can propose, so a token outside the set is a draft that cannot match, never a different output.
struct SubHead {
    /// The head over the kept columns (a copy of those strips of the target's `lm_head`).
    w: Exl3Weights,
    /// Token ids covered, as ascending `(first id, length)` ranges laid end to end in the compact numbering.
    ranges: Vec<(usize, usize)>,
    /// bf16 `[max rows, columns]`
    logits: DevBuf,
    ws: DevBuf,
}

/// Wall time of the steps of [`Draft::restrict_vocab`], in seconds: the strip copy ([`gb_cuda::exl3_linear::StripTimes`]), the logits and
/// workspace buffers, and the whole call.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RestrictTimes {
    pub strips: gb_cuda::exl3_linear::StripTimes,
    pub buffers: f64,
    pub total: f64,
}

fn cp(b: &DevBuf) -> *const c_void {
    b.ptr() as *const c_void
}
fn mp(b: &DevBuf) -> *mut c_void {
    b.ptr()
}

pub struct Draft {
    fc: Exl3Weights,
    hidden_norm: DevBuf,
    norm: DevBuf,
    layers: Vec<DraftLayer>,
    sel_proj: DevBuf,
    pred: DevBuf,
    succ: DevBuf,
    cache: DraftCache,
    pub slots: usize,
    /// Sequences per batched round: `slots` up to `MAX_DRAFT_SEQS`, further when a 16-row block halves it (8 sequences
    /// keep the round's rows within the decode kernels' 128).
    pub max_seqs: usize,
    /// Rows of the draft's block (`REXL3_DRAFT_BLOCK`, the trained 8 when unset). The layers run this many rows per
    /// sequence; `DRAFTS`-deep chains and the engine's `MAX_DEPTH` stay at 7/8 whatever it says.
    pub block: usize,
    max_ctx_rows: usize,
    // scratch (block rows = 8; context rows up to max_ctx_rows)
    fc_in: DevBuf,
    ctx: DevBuf,
    kproj: DevBuf,
    vproj: DevBuf,
    ctx_pos: DevBuf,
    ctx_slot: DevBuf,
    ws: DevBuf,
    pws: DevBuf,
    attn_ws: DevBuf,
    x: DevBuf,
    x2: DevBuf,
    h: DevBuf,
    y: DevBuf,
    o: DevBuf,
    dynv: DevBuf,
    qb: DevBuf,
    kb: DevBuf,
    vb: DevBuf,
    q2: DevBuf,
    k2: DevBuf,
    att: DevBuf,
    g: DevBuf,
    u: DevBuf,
    sw: DevBuf,
    sel: DevBuf,
    logits: DevBuf,
    top_ids: DevBuf,
    top_vals: DevBuf,
    top_ws: DevBuf,
    gate: DevBuf,
    ids: DevBuf,
    pos: DevBuf,
    anchor: DevBuf,
    slot_a: DevBuf,
    p_a: DevBuf,
    out_tokens: DevBuf,
    /// f32 `[nd]`: the selector score of each pick (the draft's own confidence in it).
    conf: DevBuf,
    /// f32 `[nd]`: the winner's lead over the runner-up of each pick.
    margin: DevBuf,
    walk: DevBuf,
    /// i32 `[max_seqs + 1]`: where each sequence's picks start in the packed head rows
    row_off: DevBuf,
    sub: Option<SubHead>,
    /// With a restricted vocabulary: `Some(min)` runs the restricted head only in rounds where every sequence has at least `min` of
    /// its recent tokens inside the set (text the set cannot reach gets the full head), `None` runs it every round.
    adaptive_min: Option<f32>,
    /// Where the last [`Draft::restrict_vocab`] spent its time (boot-time diagnostics).
    restrict_times: Option<RestrictTimes>,
    /// Rounds the draft head ran over every column (`[0]`) and over the restricted set (`[1]`).
    pub head_rounds: [u64; 2],
    /// Download every round's selector lattice into [`Picked::lattice`] (tree drafting measurements; one extra kernel and three small
    /// downloads a round). Off for serving.
    pub capture_lattice: bool,
    /// f32 `[max_seqs * DRAFTS, k, k]`: the lattice's edge terms, allocated on first capture.
    edge: Option<DevBuf>,
}

fn load_exl3<'p>(
    ctx: &mut LoadCtx<'p>,
    pack: &'p Pack,
    name: &str,
    k: usize,
    n: usize,
    bytes: &mut usize,
) -> Result<Exl3Weights, ModelError> {
    let g = pack.group_meta(name)?;
    if (g.k, g.n) != (k, n) {
        return Err(ModelError::Config(format!("draft {name}: expected K={k} N={n}, found K={} N={}", g.k, g.n)));
    }
    let w = ctx.group(pack, name)?;
    *bytes += w.weight_bytes();
    Ok(w)
}

fn f32_dev<'p>(ctx: &mut LoadCtx<'p>, pack: &'p Pack, name: &str, shape: &[usize]) -> Result<DevBuf, ModelError> {
    ctx.f32_from_bf16(pack, name, shape, false)
}

fn f16_dev<'p>(
    ctx: &mut LoadCtx<'p>,
    pack: &'p Pack,
    name: &str,
    shape: &[usize],
    bytes: &mut usize,
) -> Result<DevBuf, ModelError> {
    let b = ctx.raw16(pack, name, "F16", shape)?;
    *bytes += b.bytes();
    Ok(b)
}

fn conv_pair<'p>(ctx: &mut LoadCtx<'p>, pack: &'p Pack, p: &str, which: &str, bytes: &mut usize) -> Result<ConvPair, ModelError> {
    Ok(ConvPair {
        base: f16_dev(ctx, pack, &format!("{p}{which}.base_kernel"), &[2, 2, HIDDEN], bytes)?,
        proj: f16_dev(ctx, pack, &format!("{p}{which}.kernel_projection.weight"), &[4 * (HIDDEN / CONV_GROUP), HIDDEN], bytes)?,
    })
}

impl Draft {
    /// Load the draft pack in a pipeline of its own. `slots` sequence slots get a context ring each; `max_ctx_rows` is the
    /// most context rows absorbed in one call (a prefill chunk). The block is the trained 8; `Draft::load_block` for the
    /// depth study's.
    pub fn load(pack: &Pack, vocab: usize, slots: usize, max_ctx_rows: usize) -> Result<(Self, usize), ModelError> {
        Self::load_block(pack, vocab, slots, max_ctx_rows, BLOCK)
    }

    /// [`Draft::load`] with an explicit draft block (8..=16, `draft_block_from_env`): a 16-row block halves the batched
    /// round to 8 sequences, so it is opt-in.
    pub fn load_block(
        pack: &Pack,
        vocab: usize,
        slots: usize,
        max_ctx_rows: usize,
        block: usize,
    ) -> Result<(Self, usize), ModelError> {
        let mut ctx = LoadCtx::from_env();
        let r = Self::declare(&mut ctx, pack, vocab, slots, max_ctx_rows, block)?;
        ctx.finish()?;
        Ok(r)
    }

    /// Every weight buffer of the draft (not its scratch), named and in a fixed order, for comparing two loads of one pack.
    pub fn weight_buffers(&self) -> Vec<(String, &DevBuf)> {
        let mut v: Vec<(String, &DevBuf)> = Vec::new();
        push_linear(&mut v, "fc".into(), &self.fc);
        v.push(("hidden_norm".into(), &self.hidden_norm));
        v.push(("norm".into(), &self.norm));
        for (i, l) in self.layers.iter().enumerate() {
            let p = format!("layers.{i}");
            v.push((format!("{p}.input_norm"), &l.input_norm));
            v.push((format!("{p}.post_norm"), &l.post_norm));
            for (n, w) in
                [("q", &l.q), ("k", &l.k), ("v", &l.v), ("o", &l.o), ("gate", &l.gate), ("up", &l.up), ("down", &l.down)]
            {
                push_linear(&mut v, format!("{p}.{n}"), w);
            }
            v.push((format!("{p}.q_norm"), &l.q_norm));
            v.push((format!("{p}.k_norm"), &l.k_norm));
            for (n, c) in [("attn_conv", &l.attn_conv), ("mlp_conv", &l.mlp_conv)] {
                v.push((format!("{p}.{n}.base"), &c.base));
                v.push((format!("{p}.{n}.proj"), &c.proj));
            }
        }
        v.push(("sel_proj".into(), &self.sel_proj));
        v.push(("pred".into(), &self.pred));
        v.push(("succ".into(), &self.succ));
        v
    }

    /// Declare the draft's weights against `ctx` and allocate its scratch. The weight buffers hold nothing until
    /// `ctx.finish()` has run.
    pub fn declare<'p>(
        ctx: &mut LoadCtx<'p>,
        pack: &'p Pack,
        vocab: usize,
        slots: usize,
        max_ctx_rows: usize,
        block: usize,
    ) -> Result<(Self, usize), ModelError> {
        if !(MIN_DRAFT_BLOCK..=MAX_DRAFT_BLOCK).contains(&block) {
            return Err(ModelError::Config(format!(
                "the draft block is {block} rows, the kernels are built for {MIN_DRAFT_BLOCK} to {MAX_DRAFT_BLOCK}"
            )));
        }
        let mut bytes = 0usize;
        let fc = load_exl3(ctx, pack, "fc", FC_IN, HIDDEN, &mut bytes)?;
        let hidden_norm = f32_dev(ctx, pack, "hidden_norm.weight", &[HIDDEN])?;
        let norm = f32_dev(ctx, pack, "norm.weight", &[HIDDEN])?;
        let mut layers = Vec::new();
        for l in 0..LAYERS {
            let p = format!("layers.{l}.");
            layers.push(DraftLayer {
                input_norm: f32_dev(ctx, pack, &format!("{p}input_layernorm.weight"), &[HIDDEN])?,
                post_norm: f32_dev(ctx, pack, &format!("{p}post_attention_layernorm.weight"), &[HIDDEN])?,
                q: load_exl3(ctx, pack, &format!("{p}self_attn.q_proj"), HIDDEN, Q_HEADS * HEAD_DIM, &mut bytes)?,
                k: load_exl3(ctx, pack, &format!("{p}self_attn.k_proj"), HIDDEN, KV_HEADS * HEAD_DIM, &mut bytes)?,
                v: load_exl3(ctx, pack, &format!("{p}self_attn.v_proj"), HIDDEN, KV_HEADS * HEAD_DIM, &mut bytes)?,
                o: load_exl3(ctx, pack, &format!("{p}self_attn.o_proj"), Q_HEADS * HEAD_DIM, HIDDEN, &mut bytes)?,
                q_norm: f32_dev(ctx, pack, &format!("{p}self_attn.q_norm.weight"), &[HEAD_DIM])?,
                k_norm: f32_dev(ctx, pack, &format!("{p}self_attn.k_norm.weight"), &[HEAD_DIM])?,
                gate: load_exl3(ctx, pack, &format!("{p}mlp.gate_proj"), HIDDEN, INTERMEDIATE, &mut bytes)?,
                up: load_exl3(ctx, pack, &format!("{p}mlp.up_proj"), HIDDEN, INTERMEDIATE, &mut bytes)?,
                down: load_exl3(ctx, pack, &format!("{p}mlp.down_proj"), INTERMEDIATE, HIDDEN, &mut bytes)?,
                attn_conv: conv_pair(ctx, pack, &p, "attention_conv", &mut bytes)?,
                mlp_conv: conv_pair(ctx, pack, &p, "mlp_conv", &mut bytes)?,
            });
        }
        let sel_proj = f16_dev(ctx, pack, "candidate_selector.hidden_projection.weight", &[SELECTOR_RANK, HIDDEN], &mut bytes)?;
        let pred = f16_dev(ctx, pack, "candidate_selector.predecessor_codebook.weight", &[vocab, SELECTOR_RANK], &mut bytes)?;
        let succ = f16_dev(ctx, pack, "candidate_selector.successor_codebook.weight", &[vocab, SELECTOR_RANK], &mut bytes)?;
        let layout = DraftCtxLayout::qwen38(slots);
        let cache = DraftCache::new(layout)?;
        bytes += layout.total_bytes();
        // decode workspace for every draft shape at up to 128 rows; prefill scratch for the fc over a whole chunk
        let shapes = [
            (HIDDEN, Q_HEADS * HEAD_DIM),
            (HIDDEN, KV_HEADS * HEAD_DIM),
            (Q_HEADS * HEAD_DIM, HIDDEN),
            (HIDDEN, INTERMEDIATE),
            (INTERMEDIATE, HIDDEN),
            (FC_IN, HIDDEN),
        ];
        let max_seqs = slots.clamp(1, MAX_DRAFT_SEQS.min(128 / block));
        let (nb, nd) = (max_seqs * block, max_seqs * (block - 1));
        let ws_bytes = shapes
            .iter()
            .map(|&(k, n)| linear_workspace_bytes(max_ctx_rows.min(128).max(nb), k, n, 4))
            .max()
            .unwrap_or(0)
            .max(linear_workspace_bytes(nd, HIDDEN, vocab, 6));
        let pws_bytes = shapes.iter().map(|&(k, n)| prefill_scratch_bytes(max_ctx_rows, k, n)).max().unwrap_or(0);
        let z = |n: usize| DevBuf::zeroed(n);
        let d = Self {
            fc,
            hidden_norm,
            norm,
            layers,
            sel_proj,
            pred,
            succ,
            cache,
            slots,
            max_seqs,
            block,
            max_ctx_rows,
            fc_in: z(max_ctx_rows * FC_IN * 2)?,
            ctx: z(max_ctx_rows * HIDDEN * 2)?,
            kproj: z(max_ctx_rows * KV_HEADS * HEAD_DIM * 2)?,
            vproj: z(max_ctx_rows * KV_HEADS * HEAD_DIM * 2)?,
            ctx_pos: z(max_ctx_rows * 4)?,
            ctx_slot: z(max_ctx_rows * 4)?,
            ws: z(ws_bytes)?,
            pws: z(pws_bytes)?,
            attn_ws: z(dflash2::attn_workspace_bytes(max_seqs, DraftCtxLayout::WINDOW, Q_HEADS, block))?,
            x: z(nb * HIDDEN * 2)?,
            x2: z(nb * HIDDEN * 2)?,
            h: z(nb * HIDDEN * 2)?,
            y: z(nb * HIDDEN * 2)?,
            o: z(nb * HIDDEN * 2)?,
            dynv: z(nb * 4 * (HIDDEN / CONV_GROUP) * 2)?,
            qb: z(nb * Q_HEADS * HEAD_DIM * 2)?,
            kb: z(nb * KV_HEADS * HEAD_DIM * 2)?,
            vb: z(nb * KV_HEADS * HEAD_DIM * 2)?,
            q2: z(nb * Q_HEADS * HEAD_DIM * 2)?,
            k2: z(nb * KV_HEADS * HEAD_DIM * 2)?,
            att: z(nb * Q_HEADS * HEAD_DIM * 2)?,
            g: z(nb * INTERMEDIATE * 2)?,
            u: z(nb * INTERMEDIATE * 2)?,
            sw: z(nb * INTERMEDIATE * 2)?,
            sel: z(nd * HIDDEN * 2)?,
            logits: z(nd * vocab * 2)?,
            top_ids: z(nd * SELECTOR_TOP_K * 4)?,
            top_vals: z(nd * SELECTOR_TOP_K * 4)?,
            top_ws: z(glue::topk_workspace_bytes(nd, vocab, SELECTOR_TOP_K))?,
            gate: z(nd * SELECTOR_RANK * 2)?,
            ids: z(nb * 4)?,
            pos: z(nb * 4)?,
            anchor: z(max_seqs * 4)?,
            slot_a: z(max_seqs * 4)?,
            p_a: z(max_seqs * 4)?,
            out_tokens: z(nd * 4)?,
            conf: z(nd * 4)?,
            margin: z(nd * 4)?,
            walk: z(max_seqs * std::mem::size_of::<WalkSampling>())?,
            row_off: z((max_seqs + 1) * 4)?,
            sub: None,
            adaptive_min: None,
            restrict_times: None,
            head_rounds: [0; 2],
            capture_lattice: false,
            edge: None,
        };
        Ok((d, bytes))
    }

    /// Bytes of one ring row (one position's K or V, all kv heads) of one draft layer.
    pub const RING_ROW_BYTES: usize = KV_HEADS * HEAD_DIM * 2;

    /// Bytes of the context window a prefix-cache snapshot keeps: K and V of every draft layer for the last [`DraftCtxLayout::WINDOW`] positions.
    pub const fn window_snapshot_bytes() -> u64 {
        (LAYERS * 2 * DraftCtxLayout::WINDOW * Self::RING_ROW_BYTES) as u64
    }

    /// The ring segments (`(ring byte offset, bytes, offset in the window copy)`) of positions `lo..lo + n` of `slot`: one, or two when the
    /// positions wrap the ring.
    fn ring_segments(&self, slot: usize, lo: usize, n: usize) -> Vec<(usize, usize, usize)> {
        let l = &self.cache.layout;
        let first = n.min(l.cap - lo % l.cap);
        let mut v = vec![((slot * l.cap + lo % l.cap) * Self::RING_ROW_BYTES, first * Self::RING_ROW_BYTES, 0)];
        if first < n {
            v.push((slot * l.cap * Self::RING_ROW_BYTES, (n - first) * Self::RING_ROW_BYTES, first * Self::RING_ROW_BYTES));
        }
        v
    }

    /// FNV-1a over the ring rows of positions `lo..lo + n` of `slot` (every layer, K then V, in position order), for tests that compare a
    /// restored window with the saved one ([`crate::snapshot::SnapPool::window_hash`] hashes a snapshot's the same way).
    pub fn ring_window_hash(&self, slot: usize, lo: usize, n: usize) -> Result<u64, ModelError> {
        self.cache_sync_hash(slot, lo, n)
    }

    fn cache_sync_hash(&self, slot: usize, lo: usize, n: usize) -> Result<u64, ModelError> {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for layer in 0..LAYERS {
            for buf in [self.cache.k(layer), self.cache.v(layer)] {
                for (off, bytes, _) in self.ring_segments(slot, lo, n) {
                    let mut v = vec![0u8; bytes];
                    buf.download_range(off, &mut v)?;
                    h = v.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
                }
            }
        }
        Ok(h)
    }

    /// Copy the context ring's rows of positions `lo..lo + n` of `slot` into snapshot `snap` of `dk` / `dv` (one buffer per draft layer, each
    /// `[snapshots][WINDOW][kv_heads x head_dim]` bf16): row `i` of the snapshot is position `lo + i`. `n` is at most the window.
    #[allow(clippy::too_many_arguments)] // slot, the positions, both snapshot buffers and which snapshot, the stream
    pub fn save_window(
        &self,
        slot: usize,
        lo: usize,
        n: usize,
        dk: &mut [DevBuf],
        dv: &mut [DevBuf],
        snap: usize,
        stream: &Stream,
    ) -> Result<(), ModelError> {
        assert!(n <= DraftCtxLayout::WINDOW && dk.len() == LAYERS && dv.len() == LAYERS);
        for (layer, (k, v)) in dk.iter_mut().zip(dv.iter_mut()).enumerate() {
            for (off, bytes, at) in self.ring_segments(slot, lo, n) {
                let dst = snap * DraftCtxLayout::WINDOW * Self::RING_ROW_BYTES + at;
                k.copy_range_from(dst, self.cache.k(layer), off, bytes, stream)?;
                v.copy_range_from(dst, self.cache.v(layer), off, bytes, stream)?;
            }
        }
        Ok(())
    }

    /// The reverse of [`Draft::save_window`]: put snapshot `snap`'s rows back at positions `lo..lo + n` of `slot`'s ring.
    #[allow(clippy::too_many_arguments)]
    pub fn load_window(
        &mut self,
        slot: usize,
        lo: usize,
        n: usize,
        dk: &[DevBuf],
        dv: &[DevBuf],
        snap: usize,
        stream: &Stream,
    ) -> Result<(), ModelError> {
        assert!(n <= DraftCtxLayout::WINDOW && dk.len() == LAYERS && dv.len() == LAYERS);
        let segs = self.ring_segments(slot, lo, n);
        for (layer, (k, v)) in dk.iter().zip(dv.iter()).enumerate() {
            let (rk, rv) = self.cache.kv_mut(layer);
            for &(off, bytes, at) in &segs {
                let src = snap * DraftCtxLayout::WINDOW * Self::RING_ROW_BYTES + at;
                rk.copy_range_from(off, k, src, bytes, stream)?;
                rv.copy_range_from(off, v, src, bytes, stream)?;
            }
        }
        Ok(())
    }

    /// Absorb `n` contiguous tap rows (`taps` is the target's `[rows, 5, 5120]` tap buffer; rows `row0..row0+n`) of
    /// `slot` at absolute positions `pos0..pos0+n` into the slot's context ring.
    #[allow(clippy::too_many_arguments)]
    pub fn absorb(
        &mut self,
        taps: &DevBuf,
        row0: usize,
        n: usize,
        slot: usize,
        pos0: usize,
        stream: &Stream,
    ) -> Result<(), ModelError> {
        self.absorb_batch(taps, &[Absorb { row0, n, slot, pos0 }], stream)
    }

    /// Absorb several sequences' committed tap rows in one pass: each item's rows `row0..row0+n` of `taps` land at
    /// positions `pos0..pos0+n` of its slot's ring. The rows share one fc, one K and one V projection per layer and one
    /// ring write per layer, instead of one set per sequence. Every row goes through row-invariant kernels, so the ring
    /// holds the same bits as absorbing the items one at a time.
    pub fn absorb_batch(&mut self, taps: &DevBuf, items: &[Absorb], stream: &Stream) -> Result<(), ModelError> {
        let total: usize = items.iter().map(|a| a.n).sum();
        if total == 0 {
            return Ok(());
        }
        if total > self.max_ctx_rows {
            return Err(ModelError::Request(format!(
                "absorbing {total} context rows at once; the draft takes {}",
                self.max_ctx_rows
            )));
        }
        if items.iter().any(|a| a.n >= DraftCtxLayout::DEFAULT_CAP) {
            return Err(ModelError::Request("one context write must span fewer positions than the ring holds".into()));
        }
        let mut pos: Vec<i32> = Vec::with_capacity(total);
        let mut slots: Vec<i32> = Vec::with_capacity(total);
        for a in items {
            pos.extend((0..a.n).map(|i| (a.pos0 + i) as i32));
            slots.extend(std::iter::repeat_n(a.slot as i32, a.n));
        }
        let n = total;
        self.ctx_pos.upload(&pos)?;
        self.ctx_slot.upload(&slots)?;
        let Draft { fc, hidden_norm, layers, cache, fc_in, ctx, kproj, vproj, ctx_pos, ctx_slot, ws, pws, .. } = self;
        let mut at = 0usize;
        for a in items.iter().filter(|a| a.n > 0) {
            unsafe {
                // fc input rows are the tap rows themselves: 5 consecutive 5120-wide taps
                glue::copy_rows(
                    (taps.ptr() as *const u16).add(a.row0 * FC_IN) as *const c_void,
                    FC_IN,
                    (fc_in.ptr() as *mut u16).add(at * FC_IN) as *mut c_void,
                    FC_IN,
                    a.n,
                    FC_IN,
                    stream,
                )?;
            }
            at += a.n;
        }
        if n > 128 {
            fc.forward_prefill(fc_in, ctx, n, Dtype::BF16, pws, stream)?;
        } else {
            fc.forward(fc_in, ctx, n, Dtype::BF16, Dtype::BF16, ws, stream)?;
        }
        unsafe {
            glue::rmsnorm(cp(ctx), cp(hidden_norm), mp(ctx), n, HIDDEN, EPS, stream)?;
        }
        for (l, layer) in layers.iter().enumerate() {
            if n > 128 {
                layer.k.forward_prefill(ctx, kproj, n, Dtype::BF16, pws, stream)?;
                layer.v.forward_prefill(ctx, vproj, n, Dtype::BF16, pws, stream)?;
            } else {
                layer.k.forward(ctx, kproj, n, Dtype::BF16, Dtype::BF16, ws, stream)?;
                layer.v.forward(ctx, vproj, n, Dtype::BF16, Dtype::BF16, ws, stream)?;
            }
            cache.write_ctx(l, kproj, vproj, &layer.k_norm, ctx_pos, ctx_slot, n, EPS, ROPE_THETA, stream)?;
        }
        Ok(())
    }

    /// One draft round for `slot` at position `p` (the anchor token's position; the ring must hold the context
    /// through `p - 1`). Returns the 7 proposed tokens. `embed` and `head` are the TARGET's embedding table and
    /// lm_head (the draft has none of its own).
    #[allow(clippy::too_many_arguments)] // mirrors the kernel inputs one for one
    pub fn propose(
        &mut self,
        embed: &DevBuf,
        head: &Exl3Weights,
        vocab: usize,
        anchor: u32,
        slot: usize,
        p: usize,
        sampling: &gb_sampling::Sampling,
        stream: &Stream,
    ) -> Result<Vec<u32>, ModelError> {
        let want = self.block - 1;
        Ok(self
            .propose_batch(embed, head, vocab, &[DraftReq { anchor, slot, p, sampling, want, coverage: 1.0 }], stream)?
            .swap_remove(0))
    }

    /// One draft round for up to `max_seqs` sequences at once. Their blocks share every weight read (the draft's layers
    /// and the target's lm_head are read once for all of them) and one sync. The kernels are row-invariant and each
    /// block is independent, so a sequence's drafts are the same bits whether it runs alone or in any batch.
    pub fn propose_batch(
        &mut self,
        embed: &DevBuf,
        head: &Exl3Weights,
        vocab: usize,
        reqs: &[DraftReq],
        stream: &Stream,
    ) -> Result<Vec<Vec<u32>>, ModelError> {
        Ok(self.propose_batch_scored(embed, head, vocab, reqs, stream)?.into_iter().map(|p| p.tokens).collect())
    }

    /// Restrict the draft's vocabulary head to the target `head`'s columns of these token-id ranges (each `(first id, end id)`,
    /// widened to whole 128-column strips; overlapping or touching ranges merge; at most 4 remain and at least 64 strips must).
    /// The kept columns are bit-identical to the full head's, so the drafts are the full draft's whenever its choices lie inside
    /// the set. Costs one device-to-device copy of those strips of the head (3.6 ms plus 12 ms of `cudaMalloc` on a GB10; no host
    /// memory is involved) and their memory; returns the bytes added.
    pub fn restrict_vocab(&mut self, head: &Exl3Weights, token_ranges: &[(usize, usize)]) -> Result<usize, ModelError> {
        let t0 = std::time::Instant::now();
        let max_rows = self.max_seqs * (self.block - 1);
        let strips = vocab_strips(token_ranges, head.n)?;
        let (w, strip_times) = head.select_strips_timed(&strips)?;
        let ranges: Vec<(usize, usize)> = strips.iter().map(|&(lo, hi)| (lo * 128, (hi - lo) * 128)).collect();
        let t1 = std::time::Instant::now();
        let logits = DevBuf::zeroed(max_rows * w.n * 2)?;
        let ws = DevBuf::zeroed(linear_workspace_bytes(max_rows, HIDDEN, w.n, w.bits))?;
        let bytes = w.weight_bytes() + logits.bytes() + ws.bytes();
        self.sub = Some(SubHead { w, ranges, logits, ws });
        self.restrict_times =
            Some(RestrictTimes { strips: strip_times, buffers: t1.elapsed().as_secs_f64(), total: t0.elapsed().as_secs_f64() });
        Ok(bytes)
    }

    /// Capture every round's selector lattice into [`Picked::lattice`] (what draft trees are searched over). Allocates the edge buffer now,
    /// so a server that calls it before its memory plan has the 114 KB counted (the plan measures what is free after it).
    pub fn enable_lattice(&mut self) -> Result<(), ModelError> {
        if self.edge.is_none() {
            self.edge = Some(DevBuf::zeroed(self.max_seqs * (self.block - 1) * SELECTOR_TOP_K * SELECTOR_TOP_K * 4)?);
        }
        self.capture_lattice = true;
        Ok(())
    }

    /// Where the last [`Self::restrict_vocab`] spent its time, or None when the draft has not been restricted.
    pub fn restrict_times(&self) -> Option<&RestrictTimes> {
        self.restrict_times.as_ref()
    }

    /// Back to the full vocabulary.
    pub fn use_full_vocab(&mut self) {
        self.sub = None;
        self.adaptive_min = None;
    }

    /// With a restricted vocabulary, run the restricted head only in rounds where every sequence has at least `min_in_set` of its
    /// last [`COVERAGE_WINDOW`] tokens inside the set, and the full head otherwise: the restricted head saves 2.5-4 ms a round on
    /// English and code (which live inside the set) and loses a quarter of the acceptance on text that does not (Chinese, Russian,
    /// Arabic), which the full head keeps. `None` runs the restricted head every round.
    pub fn set_adaptive(&mut self, min_in_set: Option<f32>) {
        self.adaptive_min = min_in_set;
    }

    /// Share of `ids` (a sequence's recent tokens) inside the restricted vocabulary; 1.0 when nothing is restricted.
    pub fn coverage(&self, ids: &[u32]) -> f32 {
        self.sub.as_ref().map_or(1.0, |s| coverage_of(&s.ranges, ids))
    }

    /// Token-id ranges the draft head covers, `(first id, length)`, or None for the full vocabulary.
    pub fn vocab_ranges(&self) -> Option<&[(usize, usize)]> {
        self.sub.as_ref().map(|s| s.ranges.as_slice())
    }

    /// [`Draft::propose_batch`] with the draft's own confidence in each pick (see [`Picked`]).
    pub fn propose_batch_scored(
        &mut self,
        embed: &DevBuf,
        head: &Exl3Weights,
        vocab: usize,
        reqs: &[DraftReq],
        stream: &Stream,
    ) -> Result<Vec<Picked>, ModelError> {
        let n = reqs.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > self.max_seqs {
            return Err(ModelError::Request(format!("a draft round of {n} sequences; the draft takes {}", self.max_seqs)));
        }
        let block = self.block;
        let mut ids: Vec<i32> = Vec::with_capacity(n * block);
        let mut pos: Vec<i32> = Vec::with_capacity(n * block);
        for r in reqs {
            ids.push(r.anchor as i32);
            ids.extend(std::iter::repeat_n(MASK_TOKEN, block - 1));
            pos.extend((0..block).map(|i| (r.p + i) as i32));
        }
        let anchors: Vec<i32> = reqs.iter().map(|r| r.anchor as i32).collect();
        let slots: Vec<i32> = reqs.iter().map(|r| r.slot as i32).collect();
        let ps: Vec<i32> = reqs.iter().map(|r| r.p as i32).collect();
        // the target samples draft i (absolute position p + 1 + i) with its keyed sampler; the walk uses the same draw
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
        // the first `want` picks of each sequence: its rows 1..=want go through the head and the walk, packed one sequence after
        // another (the block's own 8 rows are all needed by the layers, but row 0 and the rows past `want` are not by the head)
        let wants: Vec<usize> = reqs.iter().map(|r| r.want.clamp(1, block - 1)).collect();
        let mut row_off = Vec::with_capacity(n + 1);
        row_off.push(0i32);
        for &w in &wants {
            row_off.push(row_off.last().unwrap() + w as i32);
        }
        self.ids.upload(&ids)?;
        self.pos.upload(&pos)?;
        self.anchor.upload(&anchors)?;
        self.slot_a.upload(&slots)?;
        self.p_a.upload(&ps)?;
        self.walk.upload(&walks)?;
        self.row_off.upload(&row_off)?;
        // the restricted head runs when there is one and, if it is adaptive, when every sequence is writing text it can reach
        let use_sub = self.sub.is_some() && self.adaptive_min.is_none_or(|min| reqs.iter().all(|r| r.coverage >= min));
        let capture = self.capture_lattice;
        if capture && self.edge.is_none() {
            self.edge = Some(DevBuf::zeroed(self.max_seqs * (self.block - 1) * SELECTOR_TOP_K * SELECTOR_TOP_K * 4)?);
        }
        let Draft {
            layers,
            norm,
            sel_proj,
            pred,
            succ,
            cache,
            ws,
            attn_ws,
            x,
            x2,
            h,
            y,
            o,
            dynv,
            qb,
            kb,
            vb,
            q2,
            k2,
            att,
            g,
            u,
            sw,
            sel,
            logits,
            top_ids,
            top_vals,
            top_ws,
            gate,
            ids: ids_d,
            pos: pos_d,
            anchor: anchor_d,
            slot_a,
            p_a,
            out_tokens,
            conf,
            margin,
            walk,
            row_off: row_off_d,
            sub,
            head_rounds,
            edge,
            ..
        } = self;
        let rows = n * block;
        let ff = |w: &Exl3Weights, a: &DevBuf, b: &mut DevBuf, m: usize, ws: &mut DevBuf| {
            w.forward(a, b, m, Dtype::BF16, Dtype::BF16, ws, stream)
        };
        unsafe {
            glue::embed(cp(embed), cp(ids_d), mp(x), rows, HIDDEN, vocab, stream)?;
        }
        for (l, layer) in layers.iter().enumerate() {
            unsafe {
                glue::rmsnorm(cp(x), cp(&layer.input_norm), mp(h), rows, HIDDEN, EPS, stream)?;
                glue::small_linear_f16w(
                    cp(h),
                    cp(&layer.attn_conv.proj),
                    mp(dynv),
                    rows,
                    HIDDEN,
                    4 * (HIDDEN / CONV_GROUP),
                    stream,
                )?;
            }
            dflash2::dynconv(h, dynv, &layer.attn_conv.base, 0, n, block, HIDDEN, CONV_GROUP, None, y, stream)?;
            ff(&layer.q, y, qb, rows, ws)?;
            ff(&layer.k, y, kb, rows, ws)?;
            ff(&layer.v, y, vb, rows, ws)?;
            unsafe {
                glue::attn_prep(
                    &glue::AttnPrepCfg::DFLASH2_DRAFT,
                    cp(qb),
                    cp(kb),
                    cp(pos_d),
                    cp(&layer.q_norm),
                    cp(&layer.k_norm),
                    mp(q2),
                    std::ptr::null_mut(),
                    mp(k2),
                    rows,
                    stream,
                )?;
            }
            dflash2::attn(
                q2,
                k2,
                vb,
                cache.k(l),
                cache.v(l),
                slot_a,
                p_a,
                n,
                Q_HEADS,
                KV_HEADS,
                DraftCtxLayout::DEFAULT_CAP,
                DraftCtxLayout::WINDOW,
                block,
                attn_ws,
                att,
                stream,
            )?;
            ff(&layer.o, att, o, rows, ws)?;
            // x' = x + conv_finish(o), written to the other buffer, then swapped in
            dflash2::dynconv(o, dynv, &layer.attn_conv.base, 1, n, block, HIDDEN, CONV_GROUP, Some(x), x2, stream)?;
            std::mem::swap(x, x2);
            unsafe {
                glue::rmsnorm(cp(x), cp(&layer.post_norm), mp(h), rows, HIDDEN, EPS, stream)?;
                glue::small_linear_f16w(
                    cp(h),
                    cp(&layer.mlp_conv.proj),
                    mp(dynv),
                    rows,
                    HIDDEN,
                    4 * (HIDDEN / CONV_GROUP),
                    stream,
                )?;
            }
            dflash2::dynconv(h, dynv, &layer.mlp_conv.base, 0, n, block, HIDDEN, CONV_GROUP, None, y, stream)?;
            ff(&layer.gate, y, g, rows, ws)?;
            ff(&layer.up, y, u, rows, ws)?;
            unsafe {
                glue::swiglu(cp(g), cp(u), mp(sw), rows * INTERMEDIATE, stream)?;
            }
            ff(&layer.down, sw, o, rows, ws)?;
            dflash2::dynconv(o, dynv, &layer.mlp_conv.base, 1, n, block, HIDDEN, CONV_GROUP, Some(x), x2, stream)?;
            std::mem::swap(x, x2);
        }
        unsafe {
            glue::rmsnorm(cp(x), cp(norm), mp(h), rows, HIDDEN, EPS, stream)?;
            // rows 1..=want of each sequence's block feed the target's head and the selector, packed
            let mut at = 0usize;
            for (s, &want) in wants.iter().enumerate() {
                glue::copy_rows(
                    (h.ptr() as *const u16).add((s * block + 1) * HIDDEN) as *const c_void,
                    HIDDEN,
                    (sel.ptr() as *mut u16).add(at * HIDDEN) as *mut c_void,
                    HIDDEN,
                    want,
                    HIDDEN,
                    stream,
                )?;
                at += want;
            }
        }
        let nd = row_off[n] as usize;
        // the head over every column, or over the restricted set (whose top-k ids are then mapped back to token ids)
        head_rounds[use_sub as usize] += 1;
        match sub.as_mut().filter(|_| use_sub) {
            None => {
                head.forward(sel, logits, nd, Dtype::BF16, Dtype::BF16, ws, stream)?;
                unsafe {
                    glue::topk(
                        cp(logits),
                        nd,
                        vocab,
                        SELECTOR_TOP_K,
                        mp(top_ids),
                        mp(top_vals),
                        mp(top_ws),
                        top_ws.bytes(),
                        stream,
                    )?;
                }
            }
            Some(sub) => {
                sub.w.forward(sel, &mut sub.logits, nd, Dtype::BF16, Dtype::BF16, &mut sub.ws, stream)?;
                unsafe {
                    glue::topk(
                        cp(&sub.logits),
                        nd,
                        sub.w.n,
                        SELECTOR_TOP_K,
                        mp(top_ids),
                        mp(top_vals),
                        mp(top_ws),
                        top_ws.bytes(),
                        stream,
                    )?;
                }
                dflash2::remap_ids(top_ids, nd * SELECTOR_TOP_K, &sub.ranges, stream)?;
            }
        }
        unsafe {
            glue::small_linear_f16w(cp(sel), cp(sel_proj), mp(gate), nd, HIDDEN, SELECTOR_RANK, stream)?;
        }
        dflash2::selector_walk_ragged(
            anchor_d,
            top_ids,
            top_vals,
            gate,
            GateDtype::Bf16,
            pred,
            succ,
            n,
            nd,
            SELECTOR_TOP_K,
            SELECTOR_RANK,
            out_tokens,
            Some(walk),
            row_off_d,
            Some(conf),
            Some(margin),
            stream,
        )?;
        if capture {
            let edge = edge.as_mut().expect("allocated above");
            dflash2::selector_edges(
                anchor_d,
                top_ids,
                gate,
                GateDtype::Bf16,
                pred,
                succ,
                n,
                nd,
                SELECTOR_TOP_K,
                SELECTOR_RANK,
                row_off_d,
                edge,
                stream,
            )?;
        }
        stream.sync()?;
        let mut t = vec![0i32; nd];
        out_tokens.download(&mut t)?;
        let mut c = vec![0f32; nd];
        conf.download(&mut c)?;
        let mut m = vec![0f32; nd];
        margin.download(&mut m)?;
        let lat = if capture {
            let mut ids = vec![0i32; nd * SELECTOR_TOP_K];
            top_ids.download(&mut ids)?;
            let mut vals = vec![0f32; nd * SELECTOR_TOP_K];
            top_vals.download(&mut vals)?;
            let mut ed = vec![0f32; nd * SELECTOR_TOP_K * SELECTOR_TOP_K];
            edge.as_ref().expect("allocated above").download(&mut ed)?;
            Some((ids, vals, ed))
        } else {
            None
        };
        Ok((0..n)
            .map(|s| {
                let (a, b) = (row_off[s] as usize, row_off[s + 1] as usize);
                let k = SELECTOR_TOP_K;
                let lattice = lat.as_ref().map(|(ids, vals, ed)| Lattice {
                    rows: b - a,
                    k,
                    ids: ids[a * k..b * k].iter().map(|&x| x as u32).collect(),
                    unary: vals[a * k..b * k].to_vec(),
                    edge: ed[a * k * k..b * k * k].to_vec(),
                    restricted_head: use_sub,
                });
                Picked {
                    tokens: t[a..b].iter().map(|&x| x as u32).collect(),
                    score: c[a..b].to_vec(),
                    margin: m[a..b].to_vec(),
                    lattice,
                }
            })
            .collect())
    }
}

/// The 128-column strips (half-open ranges, ascending, merged) that cover these token-id ranges (`(first id, end id)`) of a head with
/// `n` columns, widened outward to whole strips. Fails on an empty or out-of-range request, more than 4 resulting ranges, or fewer
/// than 64 strips (the decode kernel's K split depends on the strip count below that, so the kept columns would not be bit-identical to
/// the full head's).
pub fn vocab_strips(token_ranges: &[(usize, usize)], n: usize) -> Result<Vec<(usize, usize)>, ModelError> {
    let mut v: Vec<(usize, usize)> = Vec::new();
    for &(lo, hi) in token_ranges {
        if lo >= hi || hi > n {
            return Err(ModelError::Config(format!(
                "draft vocabulary range {lo}..{hi} is empty or past the {n} columns of the head"
            )));
        }
        v.push((lo / 128, hi.div_ceil(128)));
    }
    v.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (lo, hi) in v {
        match merged.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    let strips: usize = merged.iter().map(|&(lo, hi)| hi - lo).sum();
    if merged.is_empty() || merged.len() > 4 || strips < 64 {
        return Err(ModelError::Config(format!(
            "draft vocabulary needs 1 to 4 ranges covering at least 64 strips of 128 columns, got {} ranges and {strips} strips",
            merged.len()
        )));
    }
    Ok(merged)
}

/// The draft vocabulary TensorFold uses for this model family: token ids `[0, 98304)` and the special tokens at the top
/// (`[248032, 248320)`, the strips from 247936).
pub const TF_DRAFT_VOCAB: [(usize, usize); 2] = [(0, 98304), (248032, 248320)];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_ranges_become_merged_strips() {
        // TensorFold's set: 768 strips from 0, and the strip holding 248032 up to the end (1937..1940)
        assert_eq!(vocab_strips(&TF_DRAFT_VOCAB, 248320).unwrap(), vec![(0, 768), (1937, 1940)]);
        // overlapping and touching ranges merge; a range is widened outward to whole strips
        assert_eq!(vocab_strips(&[(0, 100 * 128 + 1), (101 * 128, 200 * 128)], 248320).unwrap(), vec![(0, 200)]);
        assert_eq!(vocab_strips(&[(100, 9000), (50, 70)], 248320).unwrap(), vec![(0, 71)]);
    }

    #[test]
    fn coverage_is_the_share_of_tokens_the_set_can_reach() {
        let set = [(0usize, 100usize), (1000, 10)];
        assert_eq!(coverage_of(&set, &[]), 1.0, "nothing written, nothing out of reach");
        assert_eq!(coverage_of(&set, &[5, 99, 1000, 1009]), 1.0);
        // the first id past each range is outside it
        assert_eq!(coverage_of(&set, &[100, 1010]), 0.0);
        assert!((coverage_of(&set, &[5, 150, 1005]) - 2.0 / 3.0).abs() < 1e-6);
        // 63 of 64 inside is above the adaptive threshold, 62 of 64 is below it
        let mut ids = vec![5u32; 64];
        ids[0] = 5000;
        assert!(coverage_of(&set, &ids) >= 0.98);
        ids[1] = 5000;
        assert!(coverage_of(&set, &ids) < 0.98);
    }

    #[test]
    fn a_vocabulary_the_kernel_could_not_keep_exact_is_refused() {
        assert!(vocab_strips(&[], 248320).is_err());
        assert!(vocab_strips(&[(0, 128 * 63)], 248320).is_err(), "63 strips is below the 64 that keep the K split");
        assert!(vocab_strips(&[(0, 128 * 64)], 248320).is_ok());
        assert!(vocab_strips(&[(10, 10)], 248320).is_err(), "empty");
        assert!(vocab_strips(&[(0, 248321)], 248320).is_err(), "past the head");
        let five: Vec<(usize, usize)> = (0..5).map(|i| (i * 20_000, i * 20_000 + 12_800)).collect();
        assert!(vocab_strips(&five, 248320).is_err(), "more than four ranges");
    }
}
