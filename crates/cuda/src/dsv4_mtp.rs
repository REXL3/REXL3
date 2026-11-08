//! DeepSeek-V4.1-Flash NextN / DSpark draft chain (`/tmp/ds41ref/model.py`:1034-1185), and the arbiter that decides how
//! much of a draft the trunk may keep.
//!
//! # The chain
//!
//! The pack carries **three full MTP layers**, not depth-`K` repeats of one: `mtp.0`, `mtp.1`, `mtp.2`, each a complete MLA
//! layer plus a 128-expert MoE (`dspark_n_routed_experts`, top-3), each with its own window KV cache. They are chained, not
//! parallel:
//!
//! ```text
//!   main_hidden = concat(h[layer 37], h[layer 38], h[layer 39])       # the *attention input* of each tap
//!   main_x      = main_norm(main_proj(main_hidden))                   # [b, 1, 5120]   (mtp.0 only)
//!   h0 = embed([t, NOISE, NOISE, NOISE, NOISE])                       # [b, 5, 5120]   (mtp.0 only)
//!   h1 = mtp.0(h0, main_x)                                            # full MLA + MoE
//!   h2 = mtp.1(h1, main_x)                                            # same main_x, not h1
//!   h3 = mtp.2(h2, main_x)
//!   out = mtp.2.forward_head(h3)                                      # markov walk + confidence
//! ```
//!
//! Two things about that picture are easy to get wrong and both are load-bearing:
//!
//! * **`main_x` is shared.** Only stage 0 owns `main_proj`/`main_norm`; every stage consumes the same projected taps as the
//!   key/value source of its DSpark attention (`DSparkAttention.forward(x, start_pos, main_x)`). What chains is `h`.
//! * **The block is bidirectional.** `get_dspark_topk_idxs` returns *one row* broadcast over all `block_size` queries, so
//!   draft position 0 attends to draft positions 1..4 as well as to the window. This is not a causal draft; it is five
//!   queries over a five-slot block, which is why a single forward pass produces the whole block.
//!
//! # The noise token
//!
//! `forward_embed` builds `draft_input_ids = [t, NOISE, NOISE, NOISE, NOISE]` — the trunk's own sampled token in slot 0
//! and `dspark_noise_token_id` = **128799** (`config.json`; every other value in this file is derived from it, and the
//! one place a stale 124799 was written down is this comment) in the other `block_size - 1` slots. The noise token is a
//! real vocabulary row: its embedding is the "I do not know yet" input the block attends from, not a mask or a pad.
//!
//! # The acceptance rule
//!
//! See [`accept_prefix`]. `model.py` contains no rejection loop — `forward_spec` is a pure proposer that returns
//! `output_ids [b, block_size + 1]`, `logits [b, block_size, vocab]` and `confidence [b, block_size]`. The only acceptance
//! signal the pack trains is the confidence head, `sigmoid(proj . [h_k ; markov_embed(x_{k-1})])` over the 5120 + 256 = 5376
//! concatenation, and the draft is committed as a **prefix**: `d_{k}` was sampled from `logits_k + head(embed(d_{k-1}))`, so
//! keeping `d_3` while dropping `d_1` would keep a token conditioned on a context that no longer exists.
//!
//! # What is verified here and what is not
//!
//! Everything in this module is host code driven by [`MtpConfig`], so the toy config runs on a CPU-only box. The kernels in
//! `csrc/dsv4_mtp.cu` (index build, markov logits bias, confidence dot, prefix accept) have launchers here but **no device
//! test has been run**: this box has no GPU, and `gpu_*` in `tests/dsv4_mtp.rs` prints `not run: no device`.

use crate::dsv4_attn::{quant_dequant, rms_norm, Rope};
use crate::dsv4_moe::{hc_mixes_host, hc_post_host, hc_pre_host, select_host, sqrtsoftplus, Dsv4Dims, HcCfg};
use crate::rt::{CudaError, DevBuf, Stream};

// ------------------------------------------------------------------------------------------------ the pack's constants

/// `dspark_block_size`: draft tokens per step.
pub const BLOCK_SIZE: usize = 5;
/// `dspark_noise_token_id`: the embedding every non-anchor slot of the draft block starts from.
pub const NOISE_TOKEN_ID: i32 = 128_799;
/// `dspark_target_layer_ids`: the trunk layers whose **attention input** is tapped (`i in target_layer_ids` in
/// `Transformer.forward`, before `layer(h, ...)` runs).
pub const TARGET_LAYER_IDS: [usize; 3] = [37, 38, 39];
/// `dspark_markov_rank`.
pub const MARKOV_RANK: usize = 256;
/// `dspark_n_routed_experts` / `dspark_num_experts_per_tok`.
pub const N_ROUTED_EXPERTS: usize = 128;
pub const TOPK: usize = 3;
/// `num_nextn_predict_layers`: three chained stages.
pub const N_MTP_LAYERS: usize = 3;
/// `candidate_topk_blocks` x `candidate_block_size`, anchored at `candidate_source_layer_id`.
pub const CAND_TOPK_BLOCKS: usize = 2048;
pub const CAND_BLOCK: usize = 8;
pub const CANDIDATE_SOURCE_LAYER: usize = 20;
/// `dim + dspark_markov_rank`: the confidence head's input width.
pub const CONF_IN: usize = 5120 + MARKOV_RANK;

// ------------------------------------------------------------------------------------------------ config

/// Shapes of the MTP stack, so the same code drives the pack and a toy.
///
/// The layer recipe is `DSparkBlock`'s: an MLA layer identical in shape to the trunk's (`compress_ratio == 0`, so plain
/// RoPE at `rope_theta` and no compressed plane, no indexer, no candidate masks), then a 128-expert MoE, both inside
/// hyper-connections with `hc_mult` residual streams.
#[derive(Clone, Debug, PartialEq)]
pub struct MtpConfig {
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_lora: usize,
    pub o_lora: usize,
    pub o_groups: usize,
    pub window: usize,
    pub block_size: usize,
    pub noise_token_id: i32,
    pub n_stages: usize,
    pub target_layer_ids: Vec<usize>,
    pub vocab: usize,
    pub markov_rank: usize,
    pub n_routed_experts: usize,
    pub topk: usize,
    pub inter_dim: usize,
    pub swiglu_limit: f32,
    pub route_scale: f32,
    pub gate_temp: f32,
    pub hc: HcCfg,
    pub eps: f32,
    pub rope_theta: f64,
    pub temperature: f32,
}

impl MtpConfig {
    /// This pack.
    pub fn v41() -> Self {
        Self {
            dim: 5120,
            n_heads: 64,
            head_dim: 512,
            rope_dim: 64,
            q_lora: 1280,
            o_lora: 1024,
            o_groups: 8,
            window: 128,
            block_size: BLOCK_SIZE,
            noise_token_id: NOISE_TOKEN_ID,
            n_stages: N_MTP_LAYERS,
            target_layer_ids: TARGET_LAYER_IDS.to_vec(),
            vocab: 129_280,
            markov_rank: MARKOV_RANK,
            n_routed_experts: N_ROUTED_EXPERTS,
            topk: TOPK,
            inter_dim: 2304,
            swiglu_limit: 10.0,
            route_scale: 1.5,
            gate_temp: 1.0,
            hc: HcCfg::V41,
            eps: 1e-20,
            rope_theta: 10_000.0,
            temperature: 1.0,
        }
    }

    /// A toy of the same structure: `head_dim == 32` so one FP8 activation block is exactly one head, four heads, two
    /// output groups, a four-slot window, eight experts.
    ///
    /// `dim` is a multiple of 128 because the hyper-connection mixer's dot product walks `ATOM_D`-wide atoms
    /// (`dsv4_moe::hc_dots_host`), which is a property of the kernel's layout, not of this layer.
    pub fn toy() -> Self {
        Self {
            dim: 128,
            n_heads: 4,
            head_dim: 32,
            rope_dim: 8,
            q_lora: 128,
            o_lora: 64,
            o_groups: 2,
            window: 4,
            block_size: 5,
            noise_token_id: 7,
            n_stages: 3,
            target_layer_ids: vec![1, 2, 3],
            vocab: 16,
            markov_rank: 8,
            n_routed_experts: 8,
            topk: 3,
            inter_dim: 16,
            swiglu_limit: 10.0,
            route_scale: 1.5,
            gate_temp: 1.0,
            hc: HcCfg::V41,
            eps: 1e-20,
            rope_theta: 10_000.0,
            temperature: 0.0, // greedy, so a test can name the token it expects
        }
    }

    pub fn heads_per_group(&self) -> usize {
        self.n_heads / self.o_groups
    }

    pub fn o_in_per_group(&self) -> usize {
        self.heads_per_group() * self.head_dim
    }

    /// `main_proj`'s input width: one `dim` per tapped trunk layer.
    pub fn tap_width(&self) -> usize {
        self.dim * self.target_layer_ids.len()
    }

    /// The confidence head's input width.
    pub fn conf_in(&self) -> usize {
        self.dim + self.markov_rank
    }

    /// KV slots a draft block attends to: the reachable part of the window, then the block itself.
    pub fn n_kv(&self, start_pos: usize) -> usize {
        self.window.min(start_pos + 1) + self.block_size
    }
}

// ------------------------------------------------------------------------------------------------ topk_idxs

/// `get_dspark_topk_idxs` (model.py:1023) — the slot list every draft query attends to, `[bsz][block_size][n_kv]`.
///
/// The reference builds
///
/// ```text
///   cat([arange(min(window, start_pos + 1)), window + arange(block_size)])   # .view(1, 1, -1).expand(bsz, block, -1)
/// ```
///
/// so **all `block_size` rows are identical** and the list is not `arange`: it is the reachable window prefix
/// `0..min(window, start_pos+1)` followed by the `block_size` block slots `window..window+block_size`. A window slot that
/// the ring has not filled yet (`start_pos + 1 < window`) is simply not listed — that is the difference between this and
/// `arange`, and it is why the list is one element shorter on every step before the window is full.
///
/// Returned flat, `bsz * block_size * n_kv` entries.
pub fn get_dspark_topk_idxs(window_size: usize, bsz: usize, block_size: usize, start_pos: usize) -> Vec<i32> {
    assert!(start_pos > 0, "a start_pos of 0 only seeds the window cache; it has no draft block");
    assert!(block_size >= 1);
    let reach = window_size.min(start_pos + 1);
    let n_kv = reach + block_size;
    let mut row = Vec::with_capacity(n_kv);
    row.extend((0..reach).map(|i| i as i32));
    row.extend((0..block_size).map(|i| (window_size + i) as i32));
    let mut out = Vec::with_capacity(bsz * block_size * n_kv);
    for _ in 0..bsz * block_size {
        out.extend_from_slice(&row);
    }
    out
}

/// The single distinct row of [`get_dspark_topk_idxs`], without the `bsz * block_size` repeat.
pub fn get_dspark_topk_row(window_size: usize, block_size: usize, start_pos: usize) -> Vec<i32> {
    let reach = window_size.min(start_pos + 1);
    (0..reach).map(|i| i as i32).chain((0..block_size).map(|i| (window_size + i) as i32)).collect()
}

// ------------------------------------------------------------------------------------------------ small host ops

/// `y [m, n] = x [m, k] @ W^T`, `W` stored `[n, k]`, f64 accumulate.
fn mm(w: &[f32], n: usize, k: usize, x: &[f32], m: usize) -> Vec<f32> {
    assert_eq!(w.len(), n * k, "weight is [n, k]");
    assert_eq!(x.len(), m * k, "x is [m, k]");
    let mut y = vec![0f32; m * n];
    for r in 0..m {
        let xr = &x[r * k..][..k];
        for c in 0..n {
            let wr = &w[c * k..][..k];
            let mut acc = 0f64;
            for i in 0..k {
                acc += xr[i] as f64 * wr[i] as f64;
            }
            y[r * n + c] = acc as f32;
        }
    }
    y
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ------------------------------------------------------------------------------------------------ markov + confidence

/// `DSparkMarkovHead` (model.py:1080): the rank-256 token embedding, and the rank->vocab head that biases the block's
/// logits.
///
/// `embed` is `[vocab, rank]`, `head` is `[vocab, rank]`. `logits` comes back `[rows, vocab]` — the reference calls
/// `self.head(embed, full_logits=True)`, so the bias is a full distribution over the vocabulary, not one scalar.
#[derive(Clone, Debug)]
pub struct MarkovHeadHost {
    pub embed: Vec<f32>,
    pub head: Vec<f32>,
    pub vocab: usize,
    pub rank: usize,
}

impl MarkovHeadHost {
    pub fn new(cfg: &MtpConfig, embed: Vec<f32>, head: Vec<f32>) -> Self {
        assert_eq!(embed.len(), cfg.vocab * cfg.markov_rank);
        assert_eq!(head.len(), cfg.vocab * cfg.markov_rank);
        Self { embed, head, vocab: cfg.vocab, rank: cfg.markov_rank }
    }

    /// `embed(token_ids)` for `rows` ids: `[rows, rank]`. An id outside `0..vocab` is rejected — the reference indexes a
    /// table and would fault, so a caller that guesses is better off than one that silently reads row 0.
    pub fn embed_tokens(&self, ids: &[i32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(ids.len() * self.rank);
        for &t in ids {
            let t = usize::try_from(t).ok().filter(|&t| t < self.vocab).unwrap_or_else(|| {
                panic!("markov token {t} is outside the vocabulary");
            });
            out.extend_from_slice(&self.embed[t * self.rank..][..self.rank]);
        }
        out
    }

    /// `head(embed)` (full logits): `[rows, vocab]`.
    pub fn logits(&self, embed: &[f32], rows: usize) -> Vec<f32> {
        mm(&self.head, self.vocab, self.rank, embed, rows)
    }
}

/// `DSparkConfidenceHead` (model.py:1092): `proj . [h ; markov_embed]`, one scalar per block row, fp32.
///
/// The score is **not** a probability: it is the pre-sigmoid logit, exactly as `proj(hidden.float()).squeeze(-1)` returns
/// it. [`Self::confidence`] applies the sigmoid; the arbiter compares the sigmoid against its threshold.
#[derive(Clone, Debug)]
pub struct ConfidenceHeadHost {
    pub proj: Vec<f32>,
    pub bias: f32,
    pub dim: usize,
    pub rank: usize,
}

impl ConfidenceHeadHost {
    pub fn new(cfg: &MtpConfig, proj: Vec<f32>, bias: f32) -> Self {
        assert_eq!(proj.len(), cfg.conf_in(), "proj is [1, dim + markov_rank]");
        Self { proj, bias, dim: cfg.dim, rank: cfg.markov_rank }
    }

    /// The logit of one row: `sum_d h[d] * proj[d] + sum_r m[r] * proj[dim + r] + bias`.
    pub fn score(&self, h: &[f32], m: &[f32]) -> f32 {
        assert_eq!(h.len(), self.dim);
        assert_eq!(m.len(), self.rank);
        let mut acc = 0f64;
        for (i, &hv) in h.iter().enumerate() {
            acc += hv as f64 * self.proj[i] as f64;
        }
        for (i, &mv) in m.iter().enumerate() {
            acc += mv as f64 * self.proj[self.dim + i] as f64;
        }
        (acc as f32) + self.bias
    }

    /// `sigmoid(score)`.
    pub fn confidence(&self, h: &[f32], m: &[f32]) -> f32 {
        1.0 / (1.0 + (-self.score(h, m)).exp())
    }

    /// Every block row of one sequence: `h` is `[rows, dim]`, `m` is `[rows, rank]`.
    pub fn confidence_rows(&self, h: &[f32], m: &[f32], rows: usize) -> Vec<f32> {
        (0..rows).map(|r| self.confidence(&h[r * self.dim..][..self.dim], &m[r * self.rank..][..self.rank])).collect()
    }
}

// ------------------------------------------------------------------------------------------------ the acceptance rule

/// How much of a draft block the trunk keeps: the longest prefix whose confidence clears `threshold`.
///
/// `confidence[k]` is the head's score for the token at block position `k` (`output_ids[:, k + 1]`), so the return value is
/// the number of **drafted** tokens accepted — `0..=block_size`. The anchor (`output_ids[:, 0]`, the trunk's own sample) is
/// not part of the draft and is always kept, so a step that accepts `n` drafts emits `n + 1` tokens.
///
/// Why a prefix and not an arbitrary subset: row `k`'s logits were biased by `markov_head(output_ids[:, k])`, i.e. by the
/// token at position `k - 1`. Keeping `d_3` after rejecting `d_1` would keep a token sampled from a context the trunk never
/// agreed to, and the block attention is bidirectional besides (`get_dspark_topk_idxs` gives every row the whole block), so
/// every row after the first rejected one is conditioned on it. A rejected suffix is discarded and re-drafted next step.
///
/// This is the one part of DSpark that `model.py` does not contain: the reference stops at `confidence` and leaves the
/// comparison to the serving loop. The rule is pinned in `tests/dsv4_mtp.rs` so a change to it is a visible change.
pub fn accept_prefix(confidence: &[f32], threshold: f32) -> usize {
    confidence.iter().take_while(|&&c| c >= threshold).count()
}

// ------------------------------------------------------------------------------------------------ one MTP layer, host

/// One `DSparkBlock` on the host: DSpark attention + 128-expert MoE, both between hyper-connections.
///
/// `stage == 0` owns `main_proj`/`main_norm`; `stage == n_stages - 1` owns `norm`, `markov_head` and `confidence_head`.
/// The intermediate stages own neither, which is exactly the reference's `if stage_id == 0` / `if stage_id == n-1`.
#[derive(Clone, Debug)]
pub struct MtpLayerHost {
    pub cfg: MtpConfig,
    pub stage: usize,
    // --- attention (DSparkAttention)
    pub wq_a: Vec<f32>,
    pub wq_b: Vec<f32>,
    pub wkv: Vec<f32>,
    pub wo_a: Vec<f32>,
    pub wo_b: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub kv_norm: Vec<f32>,
    pub sink: Vec<f32>,
    pub rope: Rope,
    /// `[window][head_dim]` ring of raw KV latents — this layer's own cache, distinct per stage.
    pub window_kv: Vec<f32>,
    // --- feed-forward
    pub gate: Vec<f32>,
    pub gate_bias: Vec<f32>,
    pub w1: Vec<f32>,
    pub w3: Vec<f32>,
    pub w2: Vec<f32>,
    pub shared_w1: Vec<f32>,
    pub shared_w3: Vec<f32>,
    pub shared_w2: Vec<f32>,
    // --- hyper-connections
    pub hc_attn_fn: Vec<f32>,
    pub hc_attn_base: Vec<f32>,
    pub hc_attn_scale: [f32; 3],
    pub hc_ffn_fn: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub hc_ffn_scale: [f32; 3],
    // --- stage 0 only
    pub main_proj: Option<Vec<f32>>,
    pub main_norm: Option<Vec<f32>>,
    // --- last stage only
    pub norm: Option<Vec<f32>>,
    pub markov: Option<MarkovHeadHost>,
    pub confidence_head: Option<ConfidenceHeadHost>,
    // --- borrowed from the trunk (`self.mtp[-1].embed = self.embed` / `self.head`), plus the two sublayer norms
    pub trunk_embed: Option<Vec<f32>>,
    pub lm_head: Option<Vec<f32>>,
    pub attn_norm_w: Option<Vec<f32>>,
    pub ffn_norm_w: Option<Vec<f32>>,
}

impl MtpLayerHost {
    /// Allocate one stage with zeroed weights (a fixture fills them).
    pub fn zeros(cfg: &MtpConfig, stage: usize, seq: usize) -> Self {
        let (d, hd, ql, ol) = (cfg.dim, cfg.head_dim, cfg.q_lora, cfg.o_lora);
        let (nh, g, oin) = (cfg.n_heads, cfg.o_groups, cfg.o_in_per_group());
        let (e, i) = (cfg.n_routed_experts, cfg.inter_dim);
        let nfn = 24;
        let row = 4 * d;
        let mut s = Self {
            cfg: cfg.clone(),
            stage,
            wq_a: vec![0.0; ql * d],
            wq_b: vec![0.0; nh * hd * ql],
            wkv: vec![0.0; hd * d],
            wo_a: vec![0.0; g * ol * oin],
            wo_b: vec![0.0; d * g * ol],
            q_norm: vec![1.0; ql],
            kv_norm: vec![1.0; hd],
            sink: vec![0.0; nh],
            rope: Rope::new(cfg.rope_dim, seq, 0, cfg.rope_theta, 1.0, 32.0, 1.0),
            window_kv: vec![0.0; cfg.window * hd],
            gate: vec![0.0; e * d],
            gate_bias: vec![0.0; e],
            w1: vec![0.0; e * i * d],
            w3: vec![0.0; e * i * d],
            w2: vec![0.0; e * d * i],
            shared_w1: vec![0.0; i * d],
            shared_w3: vec![0.0; i * d],
            shared_w2: vec![0.0; d * i],
            hc_attn_fn: vec![0.0; nfn * row],
            hc_attn_base: vec![0.0; nfn],
            hc_attn_scale: [0.0; 3],
            hc_ffn_fn: vec![0.0; nfn * row],
            hc_ffn_base: vec![0.0; nfn],
            hc_ffn_scale: [0.0; 3],
            main_proj: (stage == 0).then(|| vec![0.0; d * cfg.tap_width()]),
            main_norm: (stage == 0).then(|| vec![1.0; d]),
            norm: (stage + 1 == cfg.n_stages).then(|| vec![1.0; d]),
            markov: None,
            confidence_head: None,
            trunk_embed: None,
            lm_head: None,
            attn_norm_w: None,
            ffn_norm_w: None,
        };
        if stage + 1 == cfg.n_stages {
            s.markov =
                Some(MarkovHeadHost::new(cfg, vec![0.0; cfg.vocab * cfg.markov_rank], vec![0.0; cfg.vocab * cfg.markov_rank]));
            s.confidence_head = Some(ConfidenceHeadHost::new(cfg, vec![0.0; cfg.conf_in()], 0.0));
        }
        s
    }

    /// `DSparkBlock.forward_embed` (model.py:1131): `main_x = main_norm(main_proj(main_hidden))`, and the draft block
    /// `[t, NOISE x block_size]` expanded to `hc_mult` residual copies. Stage 0 only.
    pub fn forward_embed(&self, main_hidden: &[f32], rows: usize, input_ids: &[i32]) -> (Vec<f32>, Vec<f32>) {
        let cfg = &self.cfg;
        let (proj, norm) = (self.main_proj.as_ref().expect("forward_embed is stage 0's"), self.main_norm.as_ref().unwrap());
        assert_eq!(main_hidden.len(), rows * cfg.tap_width(), "main_hidden is [rows, dim * taps]");
        let mut main_x = mm(proj, cfg.dim, cfg.tap_width(), main_hidden, rows);
        for r in 0..rows {
            rms_norm(&mut main_x[r * cfg.dim..][..cfg.dim], norm, cfg.eps);
        }
        // the anchor is the trunk's own pick, the rest of the block is the noise token
        let mut h = vec![0f32; rows * cfg.block_size * 4 * cfg.dim];
        for r in 0..rows {
            for k in 0..cfg.block_size {
                let t = if k == 0 { input_ids[r] } else { cfg.noise_token_id };
                assert!(t >= 0 && (t as usize) < cfg.vocab, "draft token {t} outside the vocabulary");
                let e = &self.embed_row(t as usize);
                for s in 0..4 {
                    h[((r * cfg.block_size + k) * 4 + s) * cfg.dim..][..cfg.dim].copy_from_slice(e);
                }
            }
        }
        (h, main_x)
    }

    /// The anchor/noise embedding. `mtp[i].embed` is the trunk's `Transformer.embed` (the reference assigns
    /// `self.mtp[-1].embed = self.embed`), so this reads [`Self::trunk_embed`] — installed by
    /// [`Self::with_trunk_tables`], and an all-zero table stands in when a fixture does not exercise the head.
    fn embed_row(&self, t: usize) -> Vec<f32> {
        match &self.trunk_embed {
            Some(e) => e[t * self.cfg.dim..][..self.cfg.dim].to_vec(),
            None => vec![0.0; self.cfg.dim],
        }
    }

    /// `DSparkBlock.forward` (model.py:1125) — `Block.forward` with `main_x` as the attention's KV source.
    ///
    /// `start_pos == 0` is prefill: it seeds the window cache from `main_x` and returns `x` and `pre_mix` untouched
    /// (`if start_pos == 0 { self.attn(x, start_pos, main_x); return x, pre_mix }`). Otherwise `x` is the block of
    /// `block_size` draft rows.
    pub fn forward(&mut self, x: &[f32], start_pos: usize, pre_mix: &[f32], main_x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let cfg = self.cfg.clone();
        let rows = if start_pos == 0 { main_x.len() / cfg.dim } else { cfg.block_size };
        if start_pos == 0 {
            self.seed_cache(main_x, rows, 0);
            return (x.to_vec(), pre_mix.to_vec());
        }
        let (row, d) = (4 * cfg.dim, cfg.dim);

        // --- attention sub-block
        let (attn_pre, attn_post, attn_comb) =
            hc_mixes_host(x, rows, d, &self.hc_attn_fn, &self.hc_attn_scale, &self.hc_attn_base, row, cfg.hc);
        let an = self.attn_norm_w.clone().unwrap_or_else(|| vec![1.0; d]);
        let fnw = self.ffn_norm_w.clone().unwrap_or_else(|| vec![1.0; d]);
        let mut xn = hc_pre_host(x, rows, d, pre_mix);
        for r in 0..rows {
            rms_norm(&mut xn[r * d..][..d], &an, cfg.eps);
        }
        let y = self.attn(&xn, start_pos, main_x);
        let x1 = hc_post_host(x, &y, rows, d, &attn_post, &attn_comb);

        // --- feed-forward sub-block
        let (ffn_pre, ffn_post, ffn_comb) =
            hc_mixes_host(&x1, rows, d, &self.hc_ffn_fn, &self.hc_ffn_scale, &self.hc_ffn_base, row, cfg.hc);
        let mut xf = hc_pre_host(&x1, rows, d, &attn_pre);
        for r in 0..rows {
            rms_norm(&mut xf[r * d..][..d], &fnw, cfg.eps);
        }
        let y = self.moe(&xf, rows);
        let x2 = hc_post_host(&x1, &y, rows, d, &ffn_post, &ffn_comb);
        (x2, ffn_pre)
    }

    /// `DSparkAttention.forward` (model.py:1034) for `start_pos > 0`.
    ///
    /// `x` is `[block_size, dim]` (already normed), `main_x` is `[seqlen, dim]` -- `[1, dim]` at decode. Its KV row is
    /// appended to this layer's ring at `start_pos % window`, then every draft query attends over `window ++ block` at
    /// the slots [`get_dspark_topk_idxs`] names -- the whole block, not a causal prefix.
    ///
    /// **Positions.** `freqs_cis` is sliced twice: the tap rotates at `start_pos .. start_pos + seqlen`, and the block
    /// at `start_pos + seqlen ..` (`model.py:1041` and `:1057`). The block's first row is therefore the token *after* the
    /// tap, not the tap itself, which is one position of difference on every draft row -- see the audit note in the
    /// module doc.
    pub fn attn(&mut self, x: &[f32], start_pos: usize, main_x: &[f32]) -> Vec<f32> {
        let cfg = self.cfg.clone();
        let (d, hd, nh, n) = (cfg.dim, cfg.head_dim, cfg.n_heads, cfg.block_size);
        let seqlen = main_x.len() / d;
        assert!(seqlen >= 1, "main_x is [seqlen, dim]");
        let q_pos = start_pos + seqlen;

        // --- the trunk tap becomes the newest window row
        let mut mkv = mm(&self.wkv, hd, d, main_x, seqlen);
        for r in 0..seqlen {
            rms_norm(&mut mkv[r * hd..][..hd], &self.kv_norm, cfg.eps);
        }
        self.rope.apply(&mut mkv, seqlen, hd, hd, 1, start_pos, false);
        quant_dequant(&mut mkv, 32, 8, true);
        let slot = start_pos % cfg.window;
        // at decode `seqlen == 1`; a multi-row tap is the prefill branch (`seed_cache`), which never reaches here
        self.window_kv[slot * hd..][..hd].copy_from_slice(&mkv[..hd]);

        // --- the draft block's own Q and KV
        let mut qr = mm(&self.wq_a, cfg.q_lora, d, x, n);
        for r in 0..n {
            rms_norm(&mut qr[r * cfg.q_lora..][..cfg.q_lora], &self.q_norm, cfg.eps);
        }
        let mut q = mm(&self.wq_b, nh * hd, cfg.q_lora, &qr, n);
        self.rope.apply(&mut q, n, nh * hd, hd, nh, q_pos, false);
        let mut kv = mm(&self.wkv, hd, d, x, n);
        for r in 0..n {
            rms_norm(&mut kv[r * hd..][..hd], &self.kv_norm, cfg.eps);
        }
        self.rope.apply(&mut kv, n, hd, hd, 1, q_pos, false);
        quant_dequant(&mut kv, 32, 8, true);

        // --- combined plane: the window ring, then the block
        let win = cfg.window;
        let mut combined = self.window_kv.clone();
        combined.extend_from_slice(&kv);
        let idxs = get_dspark_topk_row(win, n, start_pos);

        // --- sink attention
        let scale = 1.0 / (hd as f32).sqrt();
        let mut o = vec![0f32; n * nh * hd];
        for r in 0..n {
            for hh in 0..nh {
                let qh = &q[(r * nh + hh) * hd..][..hd];
                let mut mx = f32::NEG_INFINITY;
                let mut s = vec![0f32; idxs.len()];
                for (j, &i) in idxs.iter().enumerate() {
                    let kk = &combined[i as usize * hd..][..hd];
                    let mut acc = 0f64;
                    for e in 0..hd {
                        acc += qh[e] as f64 * kk[e] as f64;
                    }
                    s[j] = acc as f32 * scale;
                    mx = mx.max(s[j]);
                }
                let mut sum = 0f64;
                let mut acc = vec![0f64; hd];
                for (j, &i) in idxs.iter().enumerate() {
                    let e = (s[j] - mx).exp() as f64;
                    sum += e;
                    let kk = &combined[i as usize * hd..][..hd];
                    for c in 0..hd {
                        acc[c] += e * kk[c] as f64;
                    }
                }
                // the sink is one extra logit with no value: it only renormalises
                sum += ((self.sink[hh] - mx).exp()) as f64;
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for c in 0..hd {
                    o[(r * nh + hh) * hd + c] = (acc[c] * inv) as f32;
                }
            }
        }

        // --- undo the query rotation, then the grouped output projection
        self.rope.apply(&mut o, n, nh * hd, hd, nh, q_pos, true);
        quant_dequant(&mut o, 32, 8, true);
        let (g, ol, oin) = (cfg.o_groups, cfg.o_lora, cfg.o_in_per_group());
        let mut olv = vec![0f32; n * g * ol];
        for r in 0..n {
            for gi in 0..g {
                let xg = &o[r * nh * hd + gi * oin..][..oin];
                let wg = &self.wo_a[gi * ol * oin..][..ol * oin];
                for j in 0..ol {
                    let mut acc = 0f64;
                    for c in 0..oin {
                        acc += xg[c] as f64 * wg[j * oin + c] as f64;
                    }
                    olv[(r * g + gi) * ol + j] = acc as f32;
                }
            }
        }
        mm(&self.wo_b, d, g * ol, &olv, n)
    }

    /// `MoE.forward` (model.py:891): `noaux_tc` router, top-3 routed SwiGLU experts, plus one shared expert.
    ///
    /// Routed experts are the pack's int8-4bit blocks; this is the f32 host form of the same arithmetic. The router's
    /// correction bias picks experts but does not scale them ([`select_host`]).
    pub fn moe(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let (d, i, e) = (cfg.dim, cfg.inter_dim, cfg.n_routed_experts);
        let mut logits = mm(&self.gate, e, d, x, rows);
        for v in logits.iter_mut() {
            *v /= cfg.gate_temp;
        }
        let dims = Dsv4Dims { hidden: d, experts: e, topk: cfg.topk, scale: cfg.route_scale, gate_temp: 1.0 };
        let (pick, wts) = select_host(&logits, rows, dims, &self.gate_bias, None, None);

        let mut y = vec![0f32; rows * d];
        for r in 0..rows {
            let xr = &x[r * d..][..d];
            for k in 0..cfg.topk {
                let ex = pick[r * cfg.topk + k] as usize;
                let wgt = wts[r * cfg.topk + k];
                let g = mm(&self.w1[ex * i * d..][..i * d], i, d, xr, 1);
                let mut u = mm(&self.w3[ex * i * d..][..i * d], i, d, xr, 1);
                let mut g = g;
                for c in 0..i {
                    u[c] = u[c].clamp(-cfg.swiglu_limit, cfg.swiglu_limit);
                    g[c] = g[c].min(cfg.swiglu_limit);
                    u[c] = silu(g[c]) * u[c] * wgt;
                }
                let down = mm(&self.w2[ex * d * i..][..d * i], d, i, &u, 1);
                for c in 0..d {
                    y[r * d + c] += down[c];
                }
            }
            // the shared expert is not routed and not scaled
            let g = mm(&self.shared_w1, i, d, xr, 1);
            let mut u = mm(&self.shared_w3, i, d, xr, 1);
            let mut g = g;
            for c in 0..i {
                u[c] = u[c].clamp(-cfg.swiglu_limit, cfg.swiglu_limit);
                g[c] = g[c].min(cfg.swiglu_limit);
                u[c] *= silu(g[c]);
            }
            let down = mm(&self.shared_w2, d, i, &u, 1);
            for c in 0..d {
                y[r * d + c] += down[c];
            }
        }
        y
    }

    /// `if start_pos == 0: self.attn(x, 0, main_x)` — fill the ring from the prefill taps, wrapping at `n % window`.
    ///
    /// `pos` is the absolute position of row 0, so a prompt can be seeded in chunks: the reference's rotated split
    /// leaves `slot = pos % window` for each of the last `window` positions, which is what [`crate::dsv4_attn`]'s
    /// `ring_write_dev` produces one row at a time. A single un-chunked call (`pos == 0`) is the reference's own form.
    pub fn seed_cache(&mut self, main_x: &[f32], rows: usize, pos: usize) {
        let cfg = self.cfg.clone();
        let (hd, win) = (cfg.head_dim, cfg.window);
        let mut kv = mm(&self.wkv, hd, cfg.dim, main_x, rows);
        for r in 0..rows {
            rms_norm(&mut kv[r * hd..][..hd], &self.kv_norm, cfg.eps);
        }
        for r in 0..rows {
            self.rope.apply(&mut kv[r * hd..][..hd], 1, hd, hd, 1, pos + r, false);
        }
        quant_dequant(&mut kv, 32, 8, true);
        if rows <= win {
            self.window_kv[..rows * hd].copy_from_slice(&kv);
        } else {
            let cutoff = rows % win;
            let last = &kv[(rows - win) * hd..];
            self.window_kv[cutoff * hd..][..(win - cutoff) * hd].copy_from_slice(&last[..(win - cutoff) * hd]);
            self.window_kv[..cutoff * hd].copy_from_slice(&last[(win - cutoff) * hd..]);
        }
    }

    /// `DSparkBlock.forward_head` (model.py:1140): collapse, final norm, then the markov walk over the block, then the
    /// confidence of every row.
    ///
    /// The walk is `block_size` steps of `logits_k + head(embed(x_{k-1}))`, sampling `x_k` from the sum — so the draft is
    /// autoregressive *through the markov head only*; the MLA/MoE ran once for all five rows.
    pub fn forward_head(&self, x: &[f32], pre_mix: &[f32], input_ids: &[i32]) -> DsparkDraft {
        let cfg = &self.cfg;
        let rows = cfg.block_size;
        let d = cfg.dim;
        let norm = self.norm.as_ref().expect("forward_head is the last stage's");
        let markov = self.markov.as_ref().unwrap();
        let conf = self.confidence_head.as_ref().unwrap();

        // `x = self.hc_pre(x, pre_mix)` — and the reference then rebinds only the *logits* through `self.norm(x)`, so the
        // confidence head reads the collapsed stream before the final norm, not after it. Two tensors, one line apart.
        let h = hc_pre_host(x, rows, d, pre_mix);
        let mut hn = h.clone();
        for r in 0..rows {
            rms_norm(&mut hn[r * d..][..d], norm, cfg.eps);
        }
        let logits = mm(&self.head_w(), cfg.vocab, d, &hn, rows);

        let mut ids = vec![0i32; rows + 1];
        ids[0] = input_ids[0];
        let mut full = logits.clone();
        let mut embeds = Vec::with_capacity(rows * cfg.markov_rank);
        for k in 0..rows {
            let prev = ids[k];
            let e = markov.embed_tokens(&[prev]);
            let bias = markov.logits(&e, 1);
            for v in 0..cfg.vocab {
                full[k * cfg.vocab + v] += bias[v];
            }
            embeds.extend_from_slice(&e);
            ids[k + 1] = sample_row(&full[k * cfg.vocab..][..cfg.vocab], cfg.temperature);
        }
        let confidence = conf.confidence_rows(&h, &embeds, rows);
        DsparkDraft { ids, logits: full, confidence }
    }

    /// `mtp[i].head` is the trunk's `ParallelHead` (`[vocab, dim]`), installed by [`Self::with_trunk_tables`].
    fn head_w(&self) -> Vec<f32> {
        self.lm_head.clone().unwrap_or_else(|| vec![0.0; self.cfg.vocab * self.cfg.dim])
    }
}

// The two trunk tables a stage borrows rather than owns (`self.mtp[-1].embed = self.embed`), kept out of `zeros` so a
// fixture can leave them out when it does not exercise the head.
impl MtpLayerHost {
    /// Replaces the borrowed trunk tables: `embed` `[vocab, dim]`, `lm_head` `[vocab, dim]`, and the two sublayer norms.
    pub fn with_trunk_tables(mut self, embed: Vec<f32>, lm_head: Vec<f32>, attn_norm: Vec<f32>, ffn_norm: Vec<f32>) -> Self {
        let d = self.cfg.dim;
        assert_eq!(embed.len(), self.cfg.vocab * d);
        assert_eq!(lm_head.len(), self.cfg.vocab * d);
        self.trunk_embed = Some(embed);
        self.lm_head = Some(lm_head);
        self.attn_norm_w = Some(attn_norm);
        self.ffn_norm_w = Some(ffn_norm);
        self
    }
}

// ------------------------------------------------------------------------------------------------ the draft

/// What one `forward_spec` call returns (model.py:1140-1159).
#[derive(Clone, Debug, PartialEq)]
pub struct DsparkDraft {
    /// `[block_size + 1]` token ids: slot 0 is the anchor (the trunk's own sample), slots 1.. are the drafted block.
    pub ids: Vec<i32>,
    /// `[block_size, vocab]`, the biased logits each draft was sampled from.
    pub logits: Vec<f32>,
    /// `[block_size]` confidences, one per drafted token.
    pub confidence: Vec<f32>,
}

impl DsparkDraft {
    /// The drafted tokens (`ids[1..]`).
    pub fn drafted(&self) -> &[i32] {
        &self.ids[1..]
    }

    /// How many drafts the arbiter keeps at `threshold`.
    pub fn accepted(&self, threshold: f32) -> usize {
        accept_prefix(&self.confidence, threshold)
    }
}

/// `sample(logits, temperature)` (model.py:1288) with the Gumbel-max replaced by its deterministic special case: at
/// `temperature == 0` the reference is `argmax`. Any other temperature needs a random source the host reference does not
/// carry, so it is refused rather than silently made greedy.
fn sample_row(logits: &[f32], temperature: f32) -> i32 {
    assert!(temperature == 0.0, "the host reference samples greedily; temperature {temperature} is not modelled");
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as i32
}

// ------------------------------------------------------------------------------------------------ the chain

/// The three chained MTP stages plus the arbiter, i.e. `Transformer.forward_spec` (model.py:1278).
///
/// Each stage keeps its **own** window cache; `main_x` is computed once by stage 0 and handed to all three; `pre_mix` is
/// threaded through, because `Block.forward` returns the mix its FFN produced for the next block's attention.
#[derive(Clone, Debug)]
pub struct DsparkChain {
    pub cfg: MtpConfig,
    pub stages: Vec<MtpLayerHost>,
    pub pos: usize,
}

impl DsparkChain {
    pub fn new(cfg: MtpConfig, stages: Vec<MtpLayerHost>) -> Self {
        assert_eq!(stages.len(), cfg.n_stages, "the chain has one host per stage");
        assert!(stages[0].main_proj.is_some(), "stage 0 owns main_proj");
        assert!(stages[cfg.n_stages - 1].markov.is_some(), "the last stage owns the markov head");
        Self { cfg, stages, pos: 1 }
    }

    /// `make_identity_pre_mix` (model.py:1162): `[rows, hc_mult]` with stream 0 hot.
    pub fn identity_pre_mix(_cfg: &MtpConfig, rows: usize) -> Vec<f32> {
        let mut p = vec![0f32; rows * 4];
        for r in 0..rows {
            p[r * 4] = 1.0;
        }
        p
    }

    /// `Transformer.forward_spec` for `start_pos > 0`: seed nothing, run the three stages in order, then the head.
    ///
    /// `main_hidden` is `[1, dim * taps]` (the trunk's tapped attention inputs, concatenated in `target_layer_ids` order);
    /// `input_ids` is `[1]`, the trunk's own sampled token.
    pub fn step(&mut self, main_hidden: &[f32], input_ids: &[i32]) -> DsparkDraft {
        let start_pos = self.pos.max(1);
        let (h0, main_x) = self.stages[0].forward_embed(main_hidden, 1, input_ids);
        let mut pre_mix = Self::identity_pre_mix(&self.cfg, self.cfg.block_size);
        let mut h = h0;
        for stage in self.stages.iter_mut() {
            let (nh, npm) = stage.forward(&h, start_pos, &pre_mix, &main_x);
            h = nh;
            pre_mix = npm;
        }
        self.pos = start_pos + 1;
        self.stages[self.cfg.n_stages - 1].forward_head(&h, &pre_mix, input_ids)
    }

    /// `Transformer.forward_spec` at `start_pos == 0`: every stage seeds its window cache and returns `None` — prefill
    /// produces no draft. `main_hidden` is `[rows, dim * taps]`.
    pub fn prefill(&mut self, main_hidden: &[f32], rows: usize) {
        self.prefill_at(main_hidden, 0, rows);
    }

    /// [`Self::prefill`] for a chunk that does not start at position 0. `main_hidden` is `[rows, dim * taps]`; the
    /// anchor ids are irrelevant to the seed (the block is not run), so zeros stand in.
    pub fn prefill_at(&mut self, main_hidden: &[f32], pos0: usize, rows: usize) {
        // `if start_pos == 0: self.attn(x, 0, main_x)` and nothing else: the block is not run, so `h` and `pre_mix`
        // are untouched and only the taps reach the ring.
        let (_, main_x) = self.stages[0].forward_embed(main_hidden, rows, &vec![0i32; rows]);
        for stage in self.stages.iter_mut() {
            stage.seed_cache(&main_x, rows, pos0);
        }
        self.pos = pos0 + rows;
    }

    /// One step's token count in the best case: the anchor plus every drafted token. See [`DsparkChain::best_case_tokens`].
    pub fn best_case_tokens(&self) -> usize {
        self.cfg.block_size + 1
    }
}

// ------------------------------------------------------------------------------------------------ device

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        /// Fill `[bsz][block_size][n_kv]` with the DSpark slot list.
        pub fn gb_dsv4_mtp_topk_idxs(out: *mut c_void, bsz: i32, block_size: i32, reach: i32, window: i32, stream: *mut c_void) -> i32;
        /// `logits[r, :] += head @ embed[token[r]]` for `rows` rows.
        pub fn gb_dsv4_mtp_markov_bias(
            logits: *mut c_void,
            tokens: *const c_void,
            embed: *const c_void,
            head: *const c_void,
            rows: i32,
            vocab: i32,
            rank: i32,
            stream: *mut c_void,
        ) -> i32;
        /// `out[r] = sigmoid(dot([h[r]; m[r]], w) + b)`.
        pub fn gb_dsv4_mtp_confidence(
            h: *const c_void,
            m: *const c_void,
            w: *const c_void,
            b: *const c_void,
            out: *mut c_void,
            rows: i32,
            dim: i32,
            rank: i32,
            stream: *mut c_void,
        ) -> i32;
        /// `out[s]` = the accepted prefix length of sequence `s` at `threshold`.
        pub fn gb_dsv4_mtp_accept(
            conf: *const c_void,
            out: *mut c_void,
            n_seq: i32,
            block_size: i32,
            threshold: f32,
            stream: *mut c_void,
        ) -> i32;
    }
}

fn need(b: &DevBuf, bytes: usize, what: &str) -> Result<(), CudaError> {
    if b.bytes() < bytes {
        eprintln!("dsv4_mtp {what}: buffer of {} bytes, the launch touches {bytes}", b.bytes());
        return Err(CudaError(1));
    }
    Ok(())
}

fn i(v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| CudaError(1))
}

/// Device form of [`get_dspark_topk_idxs`]: `out` i32 `[bsz * block_size * n_kv]`, `n_kv = reach + block_size`.
pub fn topk_idxs_dev(
    out: &mut DevBuf,
    bsz: usize,
    block_size: usize,
    reach: usize,
    window: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    let n_kv = reach + block_size;
    need(out, bsz * block_size * n_kv * 4, "topk_idxs out")?;
    CudaError::check(unsafe { ffi::gb_dsv4_mtp_topk_idxs(out.ptr(), i(bsz)?, i(block_size)?, i(reach)?, i(window)?, st.raw()) })
}

/// Device form of [`MarkovHeadHost::logits`] added in place onto `[rows, vocab]` logits.
#[allow(clippy::too_many_arguments)]
pub fn markov_bias_dev(
    logits: &mut DevBuf,
    tokens: &DevBuf,
    embed: &DevBuf,
    head: &DevBuf,
    rows: usize,
    vocab: usize,
    rank: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    need(logits, rows * vocab * 4, "markov logits")?;
    need(tokens, rows * 4, "markov tokens")?;
    need(embed, vocab * rank * 4, "markov embed")?;
    need(head, vocab * rank * 4, "markov head")?;
    CudaError::check(unsafe {
        ffi::gb_dsv4_mtp_markov_bias(logits.ptr(), tokens.ptr(), embed.ptr(), head.ptr(), i(rows)?, i(vocab)?, i(rank)?, st.raw())
    })
}

/// Device form of [`ConfidenceHeadHost::confidence_rows`].
#[allow(clippy::too_many_arguments)]
pub fn confidence_dev(
    h: &DevBuf,
    m: &DevBuf,
    w: &DevBuf,
    b: &DevBuf,
    out: &mut DevBuf,
    rows: usize,
    dim: usize,
    rank: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    need(h, rows * dim * 4, "confidence h")?;
    need(m, rows * rank * 4, "confidence m")?;
    need(w, (dim + rank) * 4, "confidence w")?;
    need(b, 4, "confidence b")?;
    need(out, rows * 4, "confidence out")?;
    CudaError::check(unsafe {
        ffi::gb_dsv4_mtp_confidence(h.ptr(), m.ptr(), w.ptr(), b.ptr(), out.ptr(), i(rows)?, i(dim)?, i(rank)?, st.raw())
    })
}

/// Device form of [`accept_prefix`], one count per sequence.
pub fn accept_dev(
    conf: &DevBuf,
    out: &mut DevBuf,
    n_seq: usize,
    block_size: usize,
    threshold: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    need(conf, n_seq * block_size * 4, "accept conf")?;
    need(out, n_seq * 4, "accept out")?;
    CudaError::check(unsafe { ffi::gb_dsv4_mtp_accept(conf.ptr(), out.ptr(), i(n_seq)?, i(block_size)?, threshold, st.raw()) })
}

/// Unused re-export guard: `sqrtsoftplus` is the router's activation and the host reference must use the same one the
/// kernel does, so it is named here rather than reimplemented.
pub fn router_score(x: f32) -> f32 {
    sqrtsoftplus(x)
}
