//! DeepSeek V4.1 attention: MLA with a grouped output projection, the CSA2 compressor, and the DSA indexer
//! retargeted to this pack. Specification: `/tmp/ds41ref/model.py` (`precompute_freqs_cis`, `Compressor`,
//! `Indexer`, `select_candidate_blocks`, `Attention`) and `/tmp/ds41ref/kernel.py` (`sparse_attn_kernel`).
//!
//! # What the block is
//!
//! One layer attends over **two** KV sources at once, concatenated into one index list:
//!
//! * a sliding window of `window` raw KV latents, and
//! * when `compress_ratio > 0`, up to `index_topk` positions of a **compressed** plane that reaches further back:
//!   every `compress_ratio` consecutive tokens pooled into one 512-wide latent by a learned softmax gate.
//!
//! Q and the output projection are both low rank; the output projection is additionally **block diagonal over
//! `o_groups` groups** (group `g` projects only the `n_heads / o_groups` heads it owns), so `wo_a` is an einsum, not
//! a linear. Both halves of the output projection and every other attention matrix are FP8 E4M3 with 32x32 E8M0
//! scales, i.e. the [`crate::fp8_linear`] path.
//!
//! # Layer roles (the three-level sparse scheme)
//!
//! | role | layers (this pack) | what it does |
//! | --- | --- | --- |
//! | KV source | 2, 8, 14, 20 | runs its own [`Compressor`] and owns the compressed plane everything else reads |
//! | index source | 2, 8, 14, 20, 24, 28, 32, 36 | runs its own [`DsaIndexer`] and publishes the top-k selection |
//! | candidate source | 20 | publishes the level-1 block mask the later index sources mask against |
//!
//! A layer that is not a KV source reads the most recent source layer's plane ([`Dsv4Config::kv_source_of`]); a layer
//! that is not an index source reuses the selection its source published. Layers with `compress_ratio == 0` (0, 1 and
//! the three MTP layers) are pure sliding-window attention: no compressor, no indexer, no compressed read.
//!
//! # RoPE, and why there are two tables
//!
//! There are two tables because the reference builds two, and which one a layer uses is decided by one bit:
//!
//! * a layer with `compress_ratio > 0` rotates **everything** — queries, the window KV, the compressed latent, the
//!   indexer's q and k — at `compress_rope_theta` (160000) with YaRN on (factor 16 over `original_seq_len` 65536,
//!   beta 32/1). model.py:681-697 picks the base, and both `_compress_kv` and `Indexer.forward` then read that same
//!   `self.freqs_cis`, so a layer never mixes bases;
//! * a layer with `compress_ratio == 0` rotates at plain `rope_theta` (10000) with YaRN **disabled**.
//!
//! [`Rope::for_layer`] applies that rule; [`Rope::compress`] is the compressed plane's table, which is therefore equal
//! to [`Rope::for_layer`] for every layer that has a compressed plane. Getting this wrong (rotating a compressing
//! layer's queries at 10000 while its plane is at 160000) is silent — every tensor is self-consistent, only the
//! product is not — which is what `block_matches_the_reference_transcription` pins.
//!
//! # Host reference and device path
//!
//! Every piece of math in this module exists twice: once as a host implementation over `f32` slices (the `Host*`
//! types and the free functions, which are what the tests compare against a direct transcription of the reference),
//! and once as a CUDA kernel in `csrc/dsv4_attn.cu` behind the `ffi` launchers, wired up by [`MlaDsv4`]. The host
//! side is the specification the kernels must satisfy; it is not a fallback device path.
//!
//! # The top-k trap this module exists to avoid
//!
//! `csrc/dsa.cuh` hardcodes `DSA_TOPK 2048` / `DSA_SEL_POOLS 512` as compile-time constants and derives the whole
//! selection geometry from them, so a pack that selects 512 positions cannot be served by that kernel without a
//! recompile. Here the selection length is a **launch parameter** (`sel_topk`), the number of scored positions is a
//! launch parameter, the row stride of the index list is a launch parameter, and the selection itself is a radix
//! select over the `f32` score bits whose `k` is read from the argument (see [`gb_dsv4_topk_select`] and
//! [`topk_indices`]). Nothing in the selection path is a function of `index_topk`.

use crate::exl3_linear::Dtype;
use crate::fp8_linear::{e4m3_to_f32, e8m0_to_f32, Fp8Error, Fp8Weights, TILE};
use crate::rt::{CudaError, DevBuf, Stream};

// ---------------------------------------------------------------------------------------------- pack geometry

/// `dim`.
pub const DIM: usize = 5120;
/// Attention heads.
pub const N_HEADS: usize = 64;
/// `head_dim` = `qk_nope_head_dim` + `qk_rope_head_dim` = 448 + 64. Also the compressed/window latent width.
pub const HEAD_DIM: usize = 512;
/// `qk_rope_head_dim`: the rotated tail of every head (and of the KV latent).
pub const ROPE_DIM: usize = 64;
/// `qk_nope_head_dim`: the unrotated part, carried as the latent.
pub const NOPE_DIM: usize = HEAD_DIM - ROPE_DIM;
/// `q_lora_rank`.
pub const Q_LORA: usize = 1280;
/// `kv_lora_rank` (equals [`HEAD_DIM`] here: the latent *is* K and V).
pub const KV_LORA: usize = 512;
/// `o_lora_rank`.
pub const O_LORA: usize = 1024;
/// `o_groups`: the block-diagonal groups of `wo_a`.
pub const O_GROUPS: usize = 8;
/// Heads per `wo_a` group.
pub const HEADS_PER_GROUP: usize = N_HEADS / O_GROUPS;
/// `wo_a` columns per group: `HEADS_PER_GROUP * head_dim`.
pub const O_IN_PER_GROUP: usize = HEADS_PER_GROUP * HEAD_DIM;
/// `wo_a` rows.
pub const WOA_ROWS: usize = O_GROUPS * O_LORA;
/// `wo_b` input width.
pub const WOB_IN: usize = O_GROUPS * O_LORA;
/// Sliding window, in tokens.
pub const WINDOW: usize = 128;
/// Backbone layers.
pub const N_LAYERS: usize = 40;
/// Total layers including the three MTP layers.
pub const N_LAYERS_TOTAL: usize = 43;
/// `index_n_heads`.
pub const INDEX_H: usize = 32;
/// `index_head_dim`.
pub const INDEX_D: usize = 128;
/// `index_topk`: **tokens** attended per sparse row. 512 in this pack, half of GLM-5.3's 2048.
pub const INDEX_TOPK: usize = 512;
/// Level-1 candidate blocks.
pub const CAND_TOPK_BLOCKS: usize = 2048;
/// Tokens per candidate block.
pub const CAND_BLOCK: usize = 8;
/// `norm_eps`.
pub const NORM_EPS: f32 = 1e-20;
/// `rope_theta` of a compressing layer (YaRN base).
pub const ROPE_THETA: f64 = 10_000.0;
/// `compress_rope_theta`: the compressed plane's base, deliberately not the sliding-window base.
pub const COMPRESS_ROPE_THETA: f64 = 160_000.0;
/// YaRN factor.
pub const ROPE_FACTOR: f64 = 16.0;
/// `original_seq_len`: the context YaRN was fitted to.
pub const ORIGINAL_SEQ_LEN: usize = 65_536;
/// YaRN fast boundary (rotations).
pub const BETA_FAST: f64 = 32.0;
/// YaRN slow boundary (rotations).
pub const BETA_SLOW: f64 = 1.0;
/// Rows per call.
pub const MAX_ROWS: usize = 2048;
pub const MAX_SLOTS: usize = 32;
/// Rows of selection/mask held inside each per-layer `Dsv4Planes`; wider calls (`n > PLANE_ROWS`) use `Dsv4Scratch::oq`.
pub const PLANE_ROWS: usize = 128;
/// `kv_source_layer_ids`.
pub const KV_SOURCE_LAYERS: [usize; 4] = [2, 8, 14, 20];
/// `index_source_layer_ids`.
pub const INDEX_SOURCE_LAYERS: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];
/// `candidate_source_layer_id`.
pub const CANDIDATE_SOURCE_LAYER: usize = 20;

/// `compress_ratios[43]`: `[0, 0, 2 x 18, 1 x 20, 0, 0, 0]`. Layers 0-1 are sliding-window only, 2-19 pool pairs,
/// 20-39 take one token per compressed position, and the three MTP layers 40-42 are sliding-window only again.
pub const COMPRESS_RATIOS: [u8; N_LAYERS_TOTAL] = build_compress_ratios();

const fn build_compress_ratios() -> [u8; N_LAYERS_TOTAL] {
    let mut r = [0u8; N_LAYERS_TOTAL];
    let mut i = 2;
    while i < 20 {
        r[i] = 2;
        i += 1;
    }
    while i < 40 {
        r[i] = 1;
        i += 1;
    }
    r
}

const _: () = assert!(HEAD_DIM == NOPE_DIM + ROPE_DIM);
const _: () = assert!(HEADS_PER_GROUP * O_GROUPS == N_HEADS);
const _: () = assert!(O_IN_PER_GROUP == N_HEADS * HEAD_DIM / O_GROUPS);
const _: () = assert!(WOB_IN == O_GROUPS * O_LORA);
const _: () = assert!(INDEX_D % TILE == 0 && Q_LORA % TILE == 0 && DIM % TILE == 0);

// ---------------------------------------------------------------------------------------------- config

/// The shapes and the layer-role tables of one model, so the same code drives the pack and a toy config.
///
/// Every `usize` here is a launch parameter somewhere; nothing in this module is a function of the pack's magnitudes
/// except the `const`s above, which are this pack's instance of these fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Dsv4Config {
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_lora: usize,
    pub o_lora: usize,
    pub o_groups: usize,
    pub window: usize,
    pub n_layers: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_topk: usize,
    pub compress_ratios: Vec<u8>,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub candidate_source_layer: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub rope_theta: f64,
    pub compress_rope_theta: f64,
    pub rope_factor: f64,
    pub original_seq_len: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub eps: f32,
}

impl Dsv4Config {
    /// This pack: DeepSeek V4.1, 40 layers + 3 MTP, 64 heads x 512, 8 output groups.
    pub fn v41() -> Self {
        Self {
            dim: DIM,
            n_heads: N_HEADS,
            head_dim: HEAD_DIM,
            rope_dim: ROPE_DIM,
            q_lora: Q_LORA,
            o_lora: O_LORA,
            o_groups: O_GROUPS,
            window: WINDOW,
            n_layers: N_LAYERS_TOTAL,
            index_heads: INDEX_H,
            index_dim: INDEX_D,
            index_topk: INDEX_TOPK,
            compress_ratios: COMPRESS_RATIOS.to_vec(),
            kv_source_layers: KV_SOURCE_LAYERS.to_vec(),
            index_source_layers: INDEX_SOURCE_LAYERS.to_vec(),
            candidate_source_layer: CANDIDATE_SOURCE_LAYER,
            candidate_topk_blocks: CAND_TOPK_BLOCKS,
            candidate_block_size: CAND_BLOCK,
            rope_theta: ROPE_THETA,
            compress_rope_theta: COMPRESS_ROPE_THETA,
            rope_factor: ROPE_FACTOR,
            original_seq_len: ORIGINAL_SEQ_LEN,
            beta_fast: BETA_FAST,
            beta_slow: BETA_SLOW,
            eps: NORM_EPS,
        }
    }

    /// A toy of the same structure, small enough for a host reference: 4 tokens of context, 2 layers.
    ///
    /// Layer 0 is a KV source, an index source and the candidate source, and pools `ratio` tokens; layer 1 reads
    /// layer 0's plane and its selection. `ratio == 0` makes both layers pure sliding window.
    pub fn toy(ratio: u8) -> Self {
        Self {
            dim: 32,
            n_heads: 2,
            head_dim: 32,
            rope_dim: 8,
            q_lora: 32,
            o_lora: 32,
            o_groups: 2,
            window: 4,
            n_layers: 2,
            // the FP8 scale tile is 32x32, so every weight's leading dim is a multiple of 32
            index_heads: 32,
            index_dim: 32,
            index_topk: 2,
            compress_ratios: vec![ratio, ratio],
            kv_source_layers: vec![0],
            index_source_layers: vec![0],
            candidate_source_layer: 0,
            candidate_topk_blocks: 2,
            candidate_block_size: 2,
            rope_theta: ROPE_THETA,
            compress_rope_theta: COMPRESS_ROPE_THETA,
            rope_factor: ROPE_FACTOR,
            original_seq_len: ORIGINAL_SEQ_LEN,
            beta_fast: BETA_FAST,
            beta_slow: BETA_SLOW,
            eps: NORM_EPS,
        }
    }

    /// `compress_ratios[layer]`, 0 for a layer past the table (never happens in this pack).
    pub fn ratio(&self, layer: usize) -> u8 {
        self.compress_ratios.get(layer).copied().unwrap_or(0)
    }

    /// `nope_head_dim`.
    pub fn nope_dim(&self) -> usize {
        self.head_dim - self.rope_dim
    }

    /// Heads per output group.
    pub fn heads_per_group(&self) -> usize {
        self.n_heads / self.o_groups
    }

    /// `wo_a` columns per group.
    pub fn o_in_per_group(&self) -> usize {
        self.heads_per_group() * self.head_dim
    }

    /// Whether `layer` pools its own KV and owns the compressed plane.
    pub fn is_kv_source(&self, layer: usize) -> bool {
        self.kv_source_layers.contains(&layer)
    }

    /// Whether `layer` runs its own indexer.
    pub fn is_index_source(&self, layer: usize) -> bool {
        self.index_source_layers.contains(&layer)
    }

    /// The source layer whose compressed plane `layer` reads: the largest KV source `<= layer`, or `None` when
    /// `layer` does not compress at all (`ratio == 0`). A layer that is its own source returns itself.
    pub fn kv_source_of(&self, layer: usize) -> Option<usize> {
        if self.ratio(layer) == 0 {
            return None;
        }
        self.kv_source_layers.iter().copied().filter(|&s| s <= layer).max()
    }

    /// The index source whose selection `layer` reuses (itself when it is one).
    pub fn index_source_of(&self, layer: usize) -> Option<usize> {
        if self.ratio(layer) == 0 {
            return None;
        }
        self.index_source_layers.iter().copied().filter(|&s| s <= layer).max()
    }

    /// Level 2 of the two-level top-k: `layer` is masked by the candidate blocks the candidate source published.
    pub fn uses_candidates(&self, layer: usize) -> bool {
        self.candidate_source_layer < layer && self.is_index_source(layer)
    }

    /// Rejects a config whose layers do not share their source's ratio (the compressed plane is read at the reader's
    /// ratio, so a mismatch would silently read the wrong number of positions) or whose shapes do not tile.
    pub fn validate(&self) -> Result<(), String> {
        for d in [self.dim, self.q_lora, self.head_dim, self.index_dim, self.o_lora] {
            if d == 0 || d % TILE != 0 {
                return Err(format!("dimension {d} is not a positive multiple of {TILE}"));
            }
        }
        if self.n_heads % self.o_groups != 0 || self.o_groups == 0 {
            return Err("o_groups must divide n_heads".into());
        }
        if self.rope_dim == 0 || self.rope_dim % 2 != 0 || self.rope_dim > self.head_dim {
            return Err("rope_dim must be even and at most head_dim".into());
        }
        if self.compress_ratios.len() < self.n_layers {
            return Err("compress_ratios is shorter than n_layers".into());
        }
        for layer in 0..self.n_layers {
            if let Some(src) = self.kv_source_of(layer) {
                if self.ratio(src) != self.ratio(layer) {
                    return Err(format!(
                        "layer {layer} reads source {src}'s plane at ratio {} but the source pools at {}",
                        self.ratio(layer),
                        self.ratio(src)
                    ));
                }
            }
            if let Some(src) = self.index_source_of(layer) {
                if self.ratio(src) != self.ratio(layer) {
                    return Err(format!("layer {layer} reads source {src}'s selection at a different ratio"));
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------- RoPE

/// Rotary frequencies, one row per position: `[seq][dim / 2]` in radians, adjacent pairs `(2j, 2j + 1)`.
///
/// This is the reference's `precompute_freqs_cis` (model.py:369) without the complex packing: `inv_freq[j]` is the
/// **angle per position** of pair `j`, and the table is `pos * inv_freq[j]`. With `original_seq_len > 0` the
/// frequencies go through YaRN: a dimension whose wavelength already completes `beta_fast` rotations inside the
/// training context keeps its frequency, one below `beta_slow` rotations is divided by `factor`, and the band between
/// them is faded linearly.
///
/// The ramp is on the *pair index* against `low`/`high` computed from `corrected_dim`, exactly as the reference does
/// it, because that is what the checkpoint was trained with — a "cleaner" YaRN (ramp on log-frequency) gives
/// different bits.
pub fn precompute_freqs_cis(
    dim: usize,
    seqlen: usize,
    original_seq_len: usize,
    base: f64,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> Vec<f32> {
    let inv = yarn_inv_freq(dim, original_seq_len, base, factor, beta_fast, beta_slow);
    let half = dim / 2;
    let mut out = vec![0f32; seqlen * half];
    for pos in 0..seqlen {
        for (j, &f) in inv.iter().enumerate() {
            out[pos * half + j] = pos as f32 * f;
        }
    }
    out
}

/// The per-position angle of each adjacent pair — the `torch.outer` operand of [`precompute_freqs_cis`], exposed
/// because it is the part worth pinning: `inv_freq[j] = base^(-2j/dim)`, YaRN-blended over the `beta_fast`..
/// `beta_slow` band when `original_seq_len > 0`.
pub fn yarn_inv_freq(dim: usize, original_seq_len: usize, base: f64, factor: f64, beta_fast: f64, beta_slow: f64) -> Vec<f32> {
    assert!(dim >= 2 && dim % 2 == 0, "rope dim must be even, got {dim}");
    let half = dim / 2;
    let mut inv = vec![0f64; half];
    for (j, f) in inv.iter_mut().enumerate() {
        // pair j is `torch.arange(0, dim, 2)[j] = 2j`, hence the 2j/dim exponent
        *f = 1.0 / base.powf(2.0 * j as f64 / dim as f64);
    }
    if original_seq_len > 0 {
        // The pair whose wavelength completes `rotations` turns over exactly the training context.
        let corrected_dim = |rotations: f64| -> f64 {
            dim as f64 * (original_seq_len as f64 / (rotations * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
        };
        let low = corrected_dim(beta_fast).floor().max(0.0);
        let high = corrected_dim(beta_slow).ceil().min((dim - 1) as f64);
        let span = (high - low).max(1e-3);
        for (j, f) in inv.iter_mut().enumerate() {
            let ramp = ((j as f64 - low) / span).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / factor * (1.0 - smooth) + *f * smooth;
        }
    }
    inv.into_iter().map(|f| f as f32).collect()
}

/// One RoPE table: the per-position angles and their `cos`/`sin`, laid out `[seq][dim / 2]`.
#[derive(Clone, Debug)]
pub struct Rope {
    pub dim: usize,
    pub seq: usize,
    pub inv_freq: Vec<f32>,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl Rope {
    /// Builds a table of `seq` positions. `original_seq_len == 0` disables YaRN (plain `1 / base^(2j/dim)`).
    pub fn new(dim: usize, seq: usize, original_seq_len: usize, base: f64, factor: f64, beta_fast: f64, beta_slow: f64) -> Self {
        let half = dim / 2;
        let inv_freq = yarn_inv_freq(dim, original_seq_len, base, factor, beta_fast, beta_slow);
        let mut cos = vec![0f32; seq * half];
        let mut sin = vec![0f32; seq * half];
        for pos in 0..seq {
            let p = pos as f32;
            let row = pos * half;
            for (j, &f) in inv_freq.iter().enumerate() {
                let a = p * f;
                cos[row + j] = a.cos();
                sin[row + j] = a.sin();
            }
        }
        Self { dim, seq, inv_freq, cos, sin }
    }

    /// The table a layer rotates with (model.py:681-697): a layer that compresses rotates **everything** -- queries, the
    /// window KV, the compressed latent, the indexer -- at `compress_rope_theta` (160000) with YaRN on; a layer that
    /// does not compress rotates at `rope_theta` (10000) with YaRN off. There is one table per layer, not one per
    /// tensor, which is why [`Self::for_layer`] and [`Self::compress`] agree for every layer that has a compressed
    /// plane.
    pub fn for_layer(cfg: &Dsv4Config, layer: usize, seq: usize) -> Self {
        RopeSpec::for_layer(cfg, layer, seq).build()
    }

    /// The compressed plane's table (`compress_rope_theta`, YaRN on).
    pub fn compress(cfg: &Dsv4Config, seq: usize) -> Self {
        RopeSpec::compress(cfg, seq).build()
    }

    /// Rotates the last `dim` of every `dim`-wide block of `x` in place, taking adjacent elements as complex pairs:
    /// `(a, b) -> (a cos - b sin, a sin + b cos)`, or the conjugate rotation when `inverse` (how the output
    /// projection's rotation is undone so the cache can stay in one shared rotated form).
    ///
    /// `x` is `[rows][stride]` with `sub` blocks of `block` elements each (`block == head_dim`) — `n_heads` blocks
    /// for a `[rows][64 * 512]` query, whose **last `rope_dim`** are rotated (the reference rotates `q[..., -rd:]`
    /// under a `[1, s, 1, rd]` table, i.e. per head), and 1 block for a `[rows][512]` latent. Row `r` rotates at
    /// position `pos0 + r`.
    // the block geometry arrives as five small integers; bundling them into a struct here would fork the
    // one signature the whole file (host twin included) already speaks
    #[allow(clippy::too_many_arguments)]
    pub fn apply(&self, x: &mut [f32], rows: usize, stride: usize, block: usize, sub: usize, pos0: usize, inverse: bool) {
        let half = self.dim / 2;
        assert!(sub >= 1 && block >= self.dim && stride >= sub * block, "a row must hold `sub` blocks of `block`");
        for r in 0..rows {
            let pos = pos0 + r;
            assert!(pos < self.seq, "rope position {pos} past the table ({})", self.seq);
            for blk in 0..sub {
                let base = r * stride + blk * block + (block - self.dim);
                for j in 0..half {
                    let (c, s) = (self.cos[pos * half + j], self.sin[pos * half + j]);
                    let s = if inverse { -s } else { s };
                    let a = x[base + 2 * j];
                    let b = x[base + 2 * j + 1];
                    x[base + 2 * j] = a * c - b * s;
                    x[base + 2 * j + 1] = a * s + b * c;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------- shared RoPE tables

/// The parameters a RoPE table is built from.
///
/// Two layers with equal specs rotate identically, so they can share one table: this pack has exactly two — the compressed plane's
/// (`compress_rope_theta` 160000 with YaRN, used by every layer with a non-zero ratio) and the plain one (`rope_theta` 10000, no
/// YaRN). Building one pair of tables a layer cost 0.67 GiB of device memory at 32K for 43 layers, of which 2 x 2 tables is all that
/// is distinct.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeSpec {
    pub dim: usize,
    pub seq: usize,
    pub original_seq_len: usize,
    pub base: f64,
    pub factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

impl RopeSpec {
    /// The table a layer rotates with: the compressed plane's when it compresses, the plain one otherwise ([`Rope::for_layer`]).
    pub fn for_layer(cfg: &Dsv4Config, layer: usize, seq: usize) -> Self {
        if cfg.ratio(layer) > 0 {
            Self::compress(cfg, seq)
        } else {
            Self {
                dim: cfg.rope_dim,
                seq,
                original_seq_len: 0,
                base: cfg.rope_theta,
                factor: cfg.rope_factor,
                beta_fast: cfg.beta_fast,
                beta_slow: cfg.beta_slow,
            }
        }
    }

    /// The compressed plane's table (`compress_rope_theta`, YaRN on).
    pub fn compress(cfg: &Dsv4Config, seq: usize) -> Self {
        Self {
            dim: cfg.rope_dim,
            seq,
            original_seq_len: cfg.original_seq_len,
            base: cfg.compress_rope_theta,
            factor: cfg.rope_factor,
            beta_fast: cfg.beta_fast,
            beta_slow: cfg.beta_slow,
        }
    }

    /// Bytes of one `cos` (or `sin`) table: `seq x dim / 2` f32.
    pub fn table_bytes(&self) -> u64 {
        self.seq as u64 * (self.dim / 2) as u64 * 4
    }

    pub fn build(&self) -> Rope {
        Rope::new(self.dim, self.seq, self.original_seq_len, self.base, self.factor, self.beta_fast, self.beta_slow)
    }
}

/// One rotation on the device: the host table it was built from and its `cos`/`sin` buffers, `[seq][rope_dim / 2]` f32.
#[derive(Debug)]
pub struct RopeTables {
    pub spec: RopeSpec,
    pub rope: Rope,
    pub cos: DevBuf,
    pub sin: DevBuf,
}

impl RopeTables {
    pub fn new(spec: RopeSpec) -> Result<Self, CudaError> {
        let pad_seq =
            if spec.original_seq_len == 0 && spec.seq >= 65_536 && spec.seq < 1_048_576 { spec.seq + 32 } else { spec.seq };
        let mut build_spec = spec;
        build_spec.seq = pad_seq;
        let mut rope = build_spec.build();
        rope.seq = spec.seq;
        let view_bytes = spec.seq * (spec.dim / 2) * 4;
        let alloc_buf = |bytes: usize| -> Result<DevBuf, CudaError> { DevBuf::zeroed_view(bytes) };
        let mut cos_full = alloc_buf(rope.cos.len() * 4)?;
        cos_full.upload(&rope.cos)?;
        rope.cos = Vec::new();
        let mut sin_full = alloc_buf(rope.sin.len() * 4)?;
        sin_full.upload(&rope.sin)?;
        rope.sin = Vec::new();
        #[cfg(target_os = "linux")]
        unsafe {
            libc::malloc_trim(0);
        }
        let cos = cos_full.subview(0, view_bytes);
        let sin = sin_full.subview(0, view_bytes);
        Ok(Self { spec, rope, cos, sin })
    }

    pub fn clone_views(&self) -> Self {
        Self {
            spec: self.spec,
            rope: Rope { dim: self.rope.dim, seq: self.rope.seq, inv_freq: Vec::new(), cos: Vec::new(), sin: Vec::new() },
            cos: self.cos.subview(0, self.cos.bytes()),
            sin: self.sin.subview(0, self.sin.bytes()),
        }
    }

    pub fn clone_views_padded(&self) -> Self {
        let (cos, sin, seq) = match (self.cos.block(), self.sin.block()) {
            (Some(cb), Some(sb)) => {
                let half_b = (self.spec.dim / 2).max(1) * 4;
                let s = cb.bytes().min(sb.bytes()) / half_b;
                (DevBuf::view(cb, 0, s * half_b), DevBuf::view(sb, 0, s * half_b), s)
            }
            _ => (self.cos.subview(0, self.cos.bytes()), self.sin.subview(0, self.sin.bytes()), self.spec.seq),
        };
        let mut spec = self.spec;
        spec.seq = seq;
        Self { spec, rope: Rope { dim: self.rope.dim, seq, inv_freq: Vec::new(), cos: Vec::new(), sin: Vec::new() }, cos, sin }
    }

    /// Device bytes of the pair.
    pub fn bytes(&self) -> u64 {
        (self.cos.bytes() + self.sin.bytes()) as u64
    }

    /// The `(cos, sin)` pair the `rope_rows_dev` kernels take.
    pub fn pair(&self) -> (&DevBuf, &DevBuf) {
        (&self.cos, &self.sin)
    }
}

/// Every distinct rotation a model needs, built once and shared by reference: the layer's `Arc` is what makes the sharing real, so a
/// table is uploaded once however many layers rotate with it.
#[derive(Debug, Default)]
pub struct SharedRope {
    tables: Vec<std::sync::Arc<RopeTables>>,
}

impl SharedRope {
    /// The distinct specs, in first-use order: one a layer's own rotation uses, and the compressed plane's when any layer has a
    /// compressed plane. Pure, so the sharing decision is testable without a device.
    pub fn specs(cfg: &Dsv4Config, max_seq: usize) -> Vec<RopeSpec> {
        let mut out: Vec<RopeSpec> = Vec::new();
        let mut push = |s: RopeSpec| {
            if !out.contains(&s) {
                out.push(s);
            }
        };
        for layer in 0..cfg.n_layers {
            push(RopeSpec::for_layer(cfg, layer, max_seq));
        }
        if (0..cfg.n_layers).any(|l| cfg.ratio(l) > 0) {
            push(RopeSpec::compress(cfg, max_seq));
        }
        out
    }

    pub fn new(cfg: &Dsv4Config, max_seq: usize) -> Result<Self, CudaError> {
        let tables = Self::specs(cfg, max_seq)
            .into_iter()
            .map(|s| RopeTables::new(s).map(std::sync::Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { tables })
    }

    /// This layer's table. Panics only if a layer outside `0..n_layers` is asked for, which no caller can do.
    pub fn get(&self, cfg: &Dsv4Config, layer: usize) -> std::sync::Arc<RopeTables> {
        let want = RopeSpec::for_layer(cfg, layer, self.seq());
        self.tables.iter().find(|t| t.spec == want).cloned().expect("every layer's rotation is in the shared set")
    }

    /// The compressed plane's table.
    pub fn compress(&self, cfg: &Dsv4Config) -> std::sync::Arc<RopeTables> {
        let want = RopeSpec::compress(cfg, self.seq());
        self.tables.iter().find(|t| t.spec == want).cloned().or_else(|| self.tables.first().cloned()).expect("at least one table")
    }

    fn seq(&self) -> usize {
        self.tables.first().map_or(0, |t| t.spec.seq)
    }

    /// Device bytes of every distinct table: what sharing saves is the layers that are not counted here.
    pub fn bytes(&self) -> u64 {
        self.tables.iter().map(|t| t.bytes()).sum()
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    /// The device bytes this model's tables would cost if every one of its `layers` layers built its own pair — the figure admission
    /// used to miss.
    pub fn unshared_bytes(cfg: &Dsv4Config, max_seq: usize) -> u64 {
        let table = RopeSpec::compress(cfg, max_seq);
        (0..cfg.n_layers)
            .map(|l| {
                let own = RopeSpec::for_layer(cfg, l, max_seq);
                let crope = if own == table { 0 } else { table.table_bytes() * 2 };
                own.table_bytes() * 2 + crope
            })
            .sum()
    }
}

/// The bytes of the shared tables a model of `cfg` at `max_seq` needs: what admission charges for RoPE.
pub fn shared_rope_bytes(cfg: &Dsv4Config, max_seq: usize) -> u64 {
    SharedRope::specs(cfg, max_seq).iter().map(|s| s.table_bytes() * 2).sum()
}

// ---------------------------------------------------------------------------------------------- host quantisation

/// The reference's `act_quant(..., inplace=True)` / `fp4_act_quant(..., inplace=True)`: quantise a `block`-wide
/// group to `bits` bits and dequantise it back, so the value downstream is the one the hardware would have produced.
///
/// `pow2_scale` selects the E8M0 (power-of-two) scale of the FP8 path; otherwise the scale is the exact `amax / max`
/// of an E4M3 scale, which is what the E4M3-scale FP4 path uses.
pub fn quant_dequant(x: &mut [f32], block: usize, bits: u32, pow2_scale: bool) {
    let max = if bits == 8 { 448.0f32 } else { 6.0f32 }; // E4M3 max, E2M1 max
    for chunk in x.chunks_mut(block) {
        let amax = chunk.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-4);
        let s = if pow2_scale { (amax / max).log2().ceil().exp2() } else { amax / max };
        let inv = 1.0 / s;
        for v in chunk.iter_mut() {
            let q = *v * inv;
            *v = if bits == 8 { round_e4m3(q) * s } else { round_e2m1(q) * s };
        }
    }
}

/// Round to the nearest E4M3 value (3-bit mantissa, 4-bit exponent, no subnormal games: `|q| <= 448`).
fn round_e4m3(q: f32) -> f32 {
    let a = q.abs();
    if a < 1e-30 {
        return 0.0;
    }
    if a >= 448.0 {
        return q.signum() * 448.0;
    }
    // E4M3 has 3 mantissa bits; its smallest normal is 2^-6 and its subnormals are 2^-9 steps.
    let e = a.log2().floor();
    let step = if e < -6.0 { 2f32.powi(-9) } else { 2f32.powi(e as i32 - 3) };
    (a / step).round() * step * q.signum()
}

/// Round to the nearest E2M1 value: `{0, .5, 1, 1.5, 2, 3, 4, 6}` and their negatives.
fn round_e2m1(q: f32) -> f32 {
    const LUT: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let a = q.abs().min(6.0);
    let mut best = 0.0f32;
    let mut best_d = f32::INFINITY;
    for &v in LUT.iter() {
        let d = (a - v).abs();
        if d < best_d {
            best_d = d;
            best = v;
        }
    }
    best * q.signum()
}

/// `RMSNorm(x) * w` in place, in f32, with the reference's epsilon (`1e-20`, i.e. a guard against zero, not a
/// regulariser).
pub fn rms_norm(x: &mut [f32], w: &[f32], eps: f32) {
    assert_eq!(x.len(), w.len());
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ss as f32 + eps).sqrt();
    for (v, &wi) in x.iter_mut().zip(w) {
        *v = *v * r * wi;
    }
}

// ---------------------------------------------------------------------------------------------- host FP8 matrix

/// A `[n, k]` E4M3 + E8M0-32x32 weight, dequantised to `f32` on the host. The device path is [`Fp8Weights`]; this
/// is the same bytes read the same way, which is what makes the two comparable.
#[derive(Clone, Debug)]
pub struct Fp8Host {
    pub n: usize,
    pub k: usize,
    pub w: Vec<f32>,
}

impl Fp8Host {
    /// Dequantise `w` (`[n, k]` E4M3 bytes) with `s` (`[n/32, k/32]` E8M0 bytes).
    pub fn new(n: usize, k: usize, w: &[u8], s: &[u8]) -> Self {
        assert_eq!(w.len(), n * k);
        let kt = k / TILE;
        assert_eq!(s.len(), (n / TILE) * kt);
        let mut d = vec![0f32; n * k];
        for row in 0..n {
            for col in 0..k {
                let sc = e8m0_to_f32(s[(row / TILE) * kt + col / TILE]);
                d[row * k + col] = e4m3_to_f32(w[row * k + col]) * sc;
            }
        }
        Self { n, k, w: d }
    }

    /// A dense `f32` weight, for the toy configs (the device path stores it as E4M3 + E8M0, so a test that wants
    /// to compare against the packed path goes through [`Fp8Host::new`]).
    pub fn from_f32(n: usize, k: usize, w: Vec<f32>) -> Self {
        assert_eq!(w.len(), n * k);
        Self { n, k, w }
    }

    /// `y [m, n] = x [m, k] @ W^T`, f64 accumulate, f32 out.
    pub fn matmul(&self, x: &[f32], m: usize) -> Vec<f32> {
        assert_eq!(x.len(), m * self.k, "x is [m, k]");
        let mut y = vec![0f32; m * self.n];
        for r in 0..m {
            let xr = &x[r * self.k..][..self.k];
            for c in 0..self.n {
                let wr = &self.w[c * self.k..][..self.k];
                let mut acc = 0f64;
                for i in 0..self.k {
                    acc += xr[i] as f64 * wr[i] as f64;
                }
                y[r * self.n + c] = acc as f32;
            }
        }
        y
    }

    /// `y [m, n] = x [m, k] @ W^T` with `W` transposed: `W` is stored `[k, n]` (the `wo_a` group layout, where the
    /// kernel reads `W[g][r][d]`).
    pub fn matmul_t(&self, x: &[f32], m: usize) -> Vec<f32> {
        assert_eq!(x.len(), m * self.n, "x is [m, n] for a [k, n]-stored weight");
        let mut y = vec![0f32; m * self.k];
        for r in 0..m {
            for j in 0..self.k {
                let mut acc = 0f64;
                for c in 0..self.n {
                    acc += x[r * self.n + c] as f64 * self.w[j * self.n + c] as f64;
                }
                y[r * self.k + j] = acc as f32;
            }
        }
        y
    }
}

// ---------------------------------------------------------------------------------------------- host / device BF16 matrix

/// A `[n, k]` BF16 weight, dequantised to `f32` on the host. The device path is [`Bf16W`]; this is the same bytes read
/// the same way, which is what makes the two comparable.
///
/// The four matrices the pack keeps in BF16 (`compressor.wkv`, `compressor.wgate`, `indexer.weights_proj`,
/// `indexer.wk`) are read through this and [`Bf16W`] — **not** re-quantised to FP8. See `csrc/dsv4_bf16.cu`.
#[derive(Clone, Debug)]
pub struct Bf16Host {
    pub n: usize,
    pub k: usize,
    pub w: Vec<f32>,
}

impl Bf16Host {
    /// Read `raw` (`[n, k]` BF16 little-endian) as it is: no scale, no rounding.
    pub fn new(n: usize, k: usize, raw: &[u8]) -> Self {
        assert_eq!(raw.len(), n * k * 2, "a [n, k] bf16 weight is n*k*2 bytes");
        let w = raw.chunks_exact(2).map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        Self { n, k, w }
    }

    /// A dense `f32` weight, for the toy configs.
    pub fn from_f32(n: usize, k: usize, w: Vec<f32>) -> Self {
        assert_eq!(w.len(), n * k);
        Self { n, k, w }
    }

    /// `y [m, n] = x [m, k] @ W^T`, f64 accumulate, f32 out.
    pub fn matmul(&self, x: &[f32], m: usize) -> Vec<f32> {
        assert_eq!(x.len(), m * self.k, "x is [m, k]");
        let mut y = vec![0f32; m * self.n];
        for r in 0..m {
            let xr = &x[r * self.k..][..self.k];
            for c in 0..self.n {
                let wr = &self.w[c * self.k..][..self.k];
                let mut acc = 0f64;
                for i in 0..self.k {
                    acc += xr[i] as f64 * wr[i] as f64;
                }
                y[r * self.n + c] = acc as f32;
            }
        }
        y
    }
}

/// `f32` -> `bf16` bits, round to nearest even (the pack's own rounding, for a test that starts from f32).
pub fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let r = (b >> 16) + (((b >> 15) & 1) & (((b & 0x7fff) != 0) as u32 | ((b >> 16) & 1)));
    (r & 0xffff) as u16
}

/// `bf16` bits -> `f32` (exact: bf16 is the top 16 bits of an f32).
pub fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// A `[n, k]` BF16 weight on the device, with fp32 accumulation: `y = x @ w^T`.
///
/// The pack keeps these four matrices in BF16 and this linear reads them unrounded — the FP8 packer that used to sit
/// here (`fp8_linear::pack_bf16`) cost ~3% RMS on the weights and bought nothing.
#[derive(Debug)]
pub struct Bf16W {
    w: DevBuf,
    pub n: usize,
    pub k: usize,
}

impl Bf16W {
    /// `raw` is the checkpoint's `[n, k]` BF16 little-endian bytes, uploaded as they are.
    pub fn new(n: usize, k: usize, raw: &[u8]) -> Result<Self, Fp8Error> {
        if raw.len() != n * k * 2 {
            return Err(Fp8Error::ShapeMismatch { what: "bf16 [n, k]", expected: n * k * 2, got: raw.len() });
        }
        let bits: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let w = DevBuf::from_slice(&bits).map_err(|_| Fp8Error::ShapeMismatch {
            what: "bf16 [n, k] (device)",
            expected: n * k * 2,
            got: 0,
        })?;
        Ok(Self { w, n, k })
    }

    /// The bytes this weight holds: the checkpoint's own, with no per-tile scale beside them.
    pub fn weight_bytes(&self) -> usize {
        self.w.bytes()
    }

    /// `y [m, n] = x [m, k] @ W^T`, fp32 accumulate. `dtype` is the output's (`F16` or `F32`); the input is fp16.
    pub fn forward(&self, x: &DevBuf, y: &mut DevBuf, m: usize, dtype: Dtype, st: &Stream) -> Result<(), CudaError> {
        if m == 0 {
            return Ok(());
        }
        if x.bytes() < m * self.k * 2 || y.bytes() < m * self.n * dtype.size() {
            return Err(CudaError(1));
        }
        let y_f32 = match dtype {
            Dtype::F32 => 1,
            Dtype::F16 => 0,
            _ => return Err(CudaError(1)),
        };
        unsafe {
            CudaError::check(ffi::gb_bf16_linear(
                x.ptr(),
                self.w.ptr(),
                y.ptr(),
                m as i32,
                self.k as i32,
                self.n as i32,
                y_f32,
                st.raw(),
            ))
        }
    }
}

// ---------------------------------------------------------------------------------------------- compressor

/// CSA2: pools `compress_ratio` consecutive tokens into one KV latent with a learned softmax gate (model.py:429).
///
/// Returns the latent **before** RoPE — the indexer consumes the unrotated form, so the caller rotates afterwards.
/// `None` means the current group is still filling: during decode the partial group lives in [`Self::kv_state`] /
/// [`Self::score_state`] and only every `compress_ratio`-th step yields a latent.
///
/// `ratio == 1` is a plain projection: no gate, no state, no fp32 promotion (the reference keeps those weights in the
/// checkpoint's bf16). `ratio > 1` pools in fp32 with `softmax(score)` over the group, per dimension.
#[derive(Clone, Debug)]
pub struct CompressorHost {
    pub ratio: usize,
    pub wkv: Bf16Host,
    pub wgate: Option<Bf16Host>,
    pub norm: Vec<f32>,
    pub kv_state: Vec<f32>,
    pub score_state: Vec<f32>,
}

impl CompressorHost {
    /// `wkv` is `[head_dim, dim]`; `wgate` is `[head_dim, dim]` and required for `ratio > 1`.
    pub fn new(ratio: usize, wkv: Bf16Host, wgate: Option<Bf16Host>, norm: Vec<f32>) -> Self {
        let hd = wkv.n;
        assert!(ratio >= 1, "compress ratio 0 is not a compressor");
        assert_eq!(norm.len(), hd);
        if ratio > 1 {
            let g = wgate.as_ref().expect("ratio > 1 needs the gate");
            assert_eq!((g.n, g.k), (hd, wkv.k));
        }
        Self { ratio, wkv, wgate, norm, kv_state: vec![0f32; ratio * hd], score_state: vec![f32::NEG_INFINITY; ratio * hd] }
    }

    /// `x [n][dim]` starting at absolute position `start_pos` -> `Some([n_groups][head_dim])` pre-RoPE, or `None`
    /// while the group is incomplete. `start_pos == 0` is the prefill form (several groups per call); otherwise the
    /// call is one decode token that fills a slot of the state.
    pub fn forward(&mut self, x: &[f32], n: usize, start_pos: usize) -> Option<Vec<f32>> {
        let hd = self.wkv.n;
        let ratio = self.ratio;
        if ratio == 1 {
            // one token per group: nothing to pool, so no gate and no fp32 -- but the norm still runs
            let mut y = self.wkv.matmul(x, n);
            for r in 0..n {
                rms_norm(&mut y[r * hd..][..hd], &self.norm, NORM_EPS);
            }
            return Some(y);
        }
        let kv = self.wkv.matmul(x, n);
        let score = self.wgate.as_ref().unwrap().matmul(x, n);

        let mut pooled: Vec<f32> = Vec::new();
        if start_pos == 0 {
            let groups = n / ratio;
            let cutoff = groups * ratio;
            let remainder = n - cutoff;
            if remainder > 0 {
                // the trailing partial group waits in the state, exactly as the reference splits it
                self.kv_state[..remainder * hd].copy_from_slice(&kv[cutoff * hd..]);
                self.score_state[..remainder * hd].copy_from_slice(&score[cutoff * hd..]);
            }
            if groups == 0 {
                return None;
            }
            pooled.resize(groups * hd, 0.0);
            for g in 0..groups {
                self.pool_group(&kv, &score, g * ratio, hd, &mut pooled[g * hd..][..hd]);
            }
        } else {
            // one token per step: fill the slot, and pool only when the group just completed
            let slot = start_pos % ratio;
            self.kv_state[slot * hd..][..hd].copy_from_slice(&kv[..hd]);
            self.score_state[slot * hd..][..hd].copy_from_slice(&score[..hd]);
            if (start_pos + 1) % ratio != 0 {
                return None;
            }
            pooled = vec![0f32; hd];
            self.pool_group(&self.kv_state.clone(), &self.score_state.clone(), 0, hd, &mut pooled[..hd]);
        }
        for g in 0..pooled.len() / hd {
            rms_norm(&mut pooled[g * hd..][..hd], &self.norm, NORM_EPS);
        }
        Some(pooled)
    }

    /// Sum over the group of `softmax_j(score[j][d]) * kv[j][d]`, per dimension `d` — the softmax runs along the
    /// group axis, so every dimension has its own gate.
    fn pool_group(&self, kv: &[f32], score: &[f32], first: usize, hd: usize, out: &mut [f32]) {
        let ratio = self.ratio;
        for d in 0..hd {
            let mut mx = f32::NEG_INFINITY;
            for j in 0..ratio {
                mx = mx.max(score[(first + j) * hd + d]);
            }
            let mut sum = 0f64;
            let mut acc = 0f64;
            for j in 0..ratio {
                let e = ((score[(first + j) * hd + d]) - mx).exp() as f64;
                sum += e;
                acc += e * kv[(first + j) * hd + d] as f64;
            }
            out[d] = (acc / sum) as f32;
        }
    }
}

// ---------------------------------------------------------------------------------------------- indexer

/// Keeps the `index_topk` best compressed positions per query (model.py:488).
///
/// A small side attention: `index_heads` x `index_dim` queries against one shared key per compressed position,
/// scores rectified (`relu`) then combined by `weights_proj`. When the layer is the candidate source this also
/// publishes the level-1 block mask; when a candidate source precedes it, its scores are masked to those blocks.
///
/// The index keys are derived from the compressor's latent, so only a KV source can produce them — every other
/// indexer reads them from that layer's cache ([`Dsv4Config::index_source_of`]).
#[derive(Clone, Debug)]
pub struct IndexerHost {
    pub heads: usize,
    pub dim: usize,
    pub topk: usize,
    pub wq_b: Fp8Host,
    pub weights_proj: Bf16Host,
    pub owns_k: bool,
    pub wk: Option<Bf16Host>,
    pub k_norm: Vec<f32>,
    /// This layer publishes the level-1 block mask.
    pub candidate_source: bool,
    /// A candidate source precedes this layer, so its scores are masked to the published blocks.
    pub uses_candidates: bool,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
}

impl IndexerHost {
    /// `wq_b` is `[heads * dim, q_lora]`; `weights_proj` is `[heads, x_dim]`; `wk` (`[dim, head_dim]`) is present
    /// iff the layer owns its index keys.
    pub fn new(
        heads: usize,
        dim: usize,
        topk: usize,
        wq_b: Fp8Host,
        weights_proj: Bf16Host,
        wk: Option<Bf16Host>,
        k_norm: Vec<f32>,
    ) -> Self {
        assert_eq!(wq_b.n, heads * dim);
        assert_eq!(weights_proj.n, heads);
        if let Some(wk) = &wk {
            // `wk` maps a head_dim-wide latent to an index key; the input width is checked against the
            // latent at matmul time (it is head_dim, not k_norm.len() -- equal only on toy configs)
            assert_eq!(wk.n, dim);
        }
        Self {
            heads,
            dim,
            topk,
            wq_b,
            weights_proj,
            owns_k: wk.is_some(),
            wk,
            k_norm,
            candidate_source: false,
            uses_candidates: false,
            candidate_topk_blocks: 0,
            candidate_block_size: 1,
        }
    }

    /// Marks this indexer as the level-1 candidate source (publishes the block mask).
    pub fn with_candidate_source(mut self, topk_blocks: usize, block_size: usize) -> Self {
        self.candidate_source = true;
        self.candidate_topk_blocks = topk_blocks;
        self.candidate_block_size = block_size;
        self
    }

    /// Marks this indexer as a level-2 consumer: its scores are masked to the blocks the source published.
    pub fn with_candidates(mut self) -> Self {
        self.uses_candidates = true;
        self
    }

    /// Scale of the index score: `index_head_dim^-0.5`, with the extra `n_heads^-0.5` folded into the weights.
    pub fn softmax_scale(&self) -> f32 {
        1.0 / (self.dim as f32).sqrt()
    }

    /// Appends index keys (when this layer owns them) and returns the selected compressed positions for `n` rows.
    ///
    /// `qr` is the layer's normalised q-lora activation `[n][q_lora]`, `latent` this layer's pre-RoPE compressed
    /// latent (`None` when the group is incomplete). Returns, per row, up to `topk` compressed positions **sorted
    /// ascending** and shifted by `offset`, with `-1` for slots that hold nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &[f32],
        qr: &[f32],
        latent: Option<&[f32]>,
        n: usize,
        start_pos: usize,
        offset: usize,
        ratio: usize,
        rope: &Rope,
        planes: &mut HostPlanes,
    ) -> Vec<i32> {
        // `planes` is the group's: its `index_k` holds the keys the source layer produced and its `topk_idxs` is the
        // selection every layer of the group reads (model.py:549 `shared_attn.index_k`, :737 `shared_attn.topk_idxs`).
        let end_pos = start_pos + n;
        // --- index keys, before the attention overwrites that storage with the rotated, quantised values
        if self.owns_k {
            if let Some(lat) = latent {
                let wk = self.wk.as_ref().unwrap();
                let hd = wk.k;
                let groups = lat.len() / hd;
                let mut k = wk.matmul(lat, groups);
                for g in 0..groups {
                    rms_norm(&mut k[g * self.dim..][..self.dim], &self.k_norm, NORM_EPS);
                    // a latent stands for the first token of its group: group j takes position j * ratio
                    let pos0 = if start_pos == 0 { g * ratio } else { start_pos + 1 - ratio };
                    rope.apply(&mut k[g * self.dim..][..self.dim], 1, self.dim, self.dim, 1, pos0, false);
                    quant_dequant(&mut k[g * self.dim..][..self.dim], 32, 4, true);
                }
                let at = start_pos / ratio;
                planes.index_k[at * self.dim..][..groups * self.dim].copy_from_slice(&k);
            }
        }

        // --- queries: rotate, quantise, score
        let mut q = self.wq_b.matmul(qr, n);
        for r in 0..n {
            let row = &mut q[r * self.heads * self.dim..][..self.heads * self.dim];
            for h in 0..self.heads {
                rope.apply(&mut row[h * self.dim..][..self.dim], 1, self.dim, self.dim, 1, start_pos + r, false);
                quant_dequant(&mut row[h * self.dim..][..self.dim], 32, 4, true);
            }
        }
        let n_pos = end_pos / ratio;
        let mut w = self.weights_proj.matmul(x, n);
        let ws = self.softmax_scale() * (self.heads as f32).powf(-0.5);
        for v in w.iter_mut() {
            *v *= ws;
        }
        let mut score = vec![0f32; n * n_pos];
        for r in 0..n {
            for p in 0..n_pos {
                let mut acc = 0f64;
                for h in 0..self.heads {
                    let qh = &q[(r * self.heads + h) * self.dim..][..self.dim];
                    let kh = &planes.index_k[p * self.dim..][..self.dim];
                    let mut d = 0f64;
                    for i in 0..self.dim {
                        d += qh[i] as f64 * kh[i] as f64;
                    }
                    acc += d.max(0.0) * w[r * self.heads + h] as f64;
                }
                score[r * n_pos + p] = acc as f32;
            }
        }

        // --- how many compressed positions each query can see: a block becomes visible once the query has passed
        // its last token
        let mut out = vec![-1i32; n * self.topk];
        let reach = |r: usize| -> usize {
            if start_pos == 0 {
                (r + 1) / ratio
            } else {
                end_pos / ratio
            }
        };
        // --- level 1 of the two-level top-k: publish the block mask, or mask against the published one
        for r in 0..n {
            let len = reach(r);
            if self.candidate_source {
                let mask = select_candidate_blocks(
                    &score[r * n_pos..][..n_pos],
                    len.max(1),
                    self.candidate_topk_blocks,
                    self.candidate_block_size,
                );
                planes.candidates[r * planes.cand_stride..][..n_pos].copy_from_slice(&mask);
            } else if self.uses_candidates {
                for p in 0..len {
                    if !planes.candidates[r * planes.cand_stride + p] {
                        score[r * n_pos + p] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        for r in 0..n {
            let len = reach(r);
            for p in len..n_pos {
                score[r * n_pos + p] = f32::NEG_INFINITY;
            }
            let k = self.topk.min(n_pos);
            let sel = topk_indices(&score[r * n_pos..][..n_pos], k);
            // the reference emits exactly `k` entries and maps the unreachable ones to -1
            for (i, &p) in sel.iter().enumerate() {
                out[r * self.topk + i] = if p < len { p as i32 + offset as i32 } else { -1 };
            }
        }
        out
    }
}

/// Level 1 of the two-level top-k: keep the `topk_blocks` highest-scoring blocks per query (model.py:584).
///
/// A block scores as its best position (with `-inf` padding the last block out to `block_size`), and the block that
/// holds the query's newest reachable position is pinned to `+inf`, because it is only partly filled and could
/// otherwise be outscored by an older, full block. Returns a bool mask shaped like the logits.
pub fn select_candidate_blocks(logits: &[f32], compress_len: usize, topk_blocks: usize, block_size: usize) -> Vec<bool> {
    assert!(compress_len >= 1, "a query always reaches at least one compressed position");
    assert!(block_size >= 1 && topk_blocks >= 1);
    let width = logits.len();
    let n_blocks = width.div_ceil(block_size);
    let mut scores = vec![f32::NEG_INFINITY; n_blocks];
    for b in 0..n_blocks {
        let hi = ((b + 1) * block_size).min(width);
        for &v in &logits[b * block_size..hi] {
            scores[b] = scores[b].max(v);
        }
    }
    let last = (compress_len - 1) / block_size;
    if last < n_blocks {
        scores[last] = f32::INFINITY;
    }
    let mut order: Vec<usize> = (0..n_blocks).collect();
    order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let keep = topk_blocks.min(n_blocks);
    let mut mask = vec![false; width];
    for &b in &order[..keep] {
        if scores[b] > f32::NEG_INFINITY {
            for slot in mask.iter_mut().take(((b + 1) * block_size).min(width)).skip(b * block_size) {
                *slot = true;
            }
        }
    }
    mask
}

/// The `k` highest-scoring positions, ascending by position (ties to the **lowest** index), which is the order the
/// reference's `topk(...).indices.sort()` produces for distinct scores and a deterministic choice for ties.
///
/// Exactly `k = min(k, n)` entries come back, `-inf` scores included: the reference selects `min(index_topk,
/// n_pos)` positions and only then maps the unreachable ones (`pos >= compress_len`) to `-1`, so an under-full
/// selection has holes rather than a short list. This is the host twin of the device radix select: `k` is a value,
/// not a compile-time constant, and the cost is `O(n log n)` here versus `O(n * bits / 8)` on the device.
pub fn topk_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(scores.len());
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    order.truncate(k);
    order.sort_unstable();
    order
}

// ---------------------------------------------------------------------------------------------- the block, on the host

/// The sliding-window slot list of every row of a call (model.py:410).
///
/// The cache is a ring of `window` slots. Prefill needs one row per query, each seeing its own causal window; a decode
/// step has a single query that sees the whole ring, oldest first. `-1` marks a slot holding nothing.
pub fn window_idxs(window: usize, n: usize, start_pos: usize) -> Vec<Vec<i32>> {
    if start_pos == 0 {
        let width = n.min(window);
        (0..n)
            .map(|r| {
                let start = (r + 1).saturating_sub(window);
                (0..width)
                    .map(|j| {
                        let idx = start + j;
                        if idx > r {
                            -1
                        } else {
                            idx as i32
                        }
                    })
                    .collect()
            })
            .collect()
    } else {
        assert_eq!(n, 1, "a decode step is one row");
        let oldest = start_pos % window + 1;
        let mut idxs: Vec<i32> = (oldest..window).chain(0..oldest).map(|v| v as i32).collect();
        for v in idxs.iter_mut() {
            if *v as usize > start_pos {
                *v = -1;
            }
        }
        vec![idxs]
    }
}

/// The planes one layer reads and writes, and the selection it shares with the layers that reuse it (the reference's
/// `shared_attn`). One sequence; `rows` are tokens of it.
#[derive(Clone, Debug)]
pub struct HostPlanes {
    /// `[window][head_dim]` ring of raw KV latents.
    pub window: Vec<f32>,
    /// `[max_compress][head_dim]` compressed plane (shared by every layer that reads the same source).
    pub compress: Vec<f32>,
    /// `[max_compress][index_dim]` index keys.
    pub index_k: Vec<f32>,
    /// `[MAX_ROWS][max_compress]` level-1 block mask the candidate source published (`true` = allowed), one row per
    /// query (a query's reachable set depends on its own position, so the mask is per row).
    pub candidates: Vec<bool>,
    /// Elements between rows of [`Self::candidates`].
    pub cand_stride: usize,
    /// `[MAX_ROWS][index_topk]` the selection the last index source published, into the combined plane.
    pub topk_idxs: Vec<i32>,
    /// Positions of the combined plane: the window rows plus the compressed rows.
    pub n_compress: usize,
}

impl HostPlanes {
    pub fn new(cfg: &Dsv4Config, max_seq: usize, layer: usize) -> Self {
        let nc = compress_slots(cfg, layer, max_seq);
        Self {
            window: vec![0f32; cfg.window * cfg.head_dim],
            compress: vec![0f32; nc * cfg.head_dim],
            index_k: vec![0f32; nc * cfg.index_dim],
            candidates: vec![false; MAX_ROWS * nc],
            cand_stride: nc,
            topk_idxs: vec![-1; MAX_ROWS * cfg.index_topk],
            n_compress: 0,
        }
    }
}

/// Compressed entries a layer's plane holds at `max_seq`: one per `ratio` tokens plus the partial one.
///
/// This is a **per-layer** number, and it has to be: layers 20-39 pool one token per compressed position, so their
/// plane holds `max_seq + 1` entries where a ratio-2 layer's holds half that. Sizing every plane from the largest
/// ratio in the table ([`Dsv4Config::ratio_hint`]) under-allocates the ratio-1 layers' plane, the indexer's keys and
/// the level-1 mask by 2x and the indexer's score row by 2x, and a long prefill walks off the end of all four.
pub fn compress_slots(cfg: &Dsv4Config, layer: usize, max_seq: usize) -> usize {
    let r = (cfg.ratio(layer) as usize).max(1);
    max_seq.div_ceil(r) + 1
}

/// The widest [`compress_slots`] over the table: what a buffer shared by every layer must hold.
pub fn compress_slots_max(cfg: &Dsv4Config, max_seq: usize) -> usize {
    (0..cfg.compress_ratios.len()).map(|l| compress_slots(cfg, l, max_seq)).max().unwrap_or(1)
}

impl Dsv4Config {
    /// The largest ratio in the table. A *hint*, not a plane size: [`compress_slots`] is the size.
    pub fn ratio_hint(&self) -> usize {
        self.compress_ratios.iter().copied().max().unwrap_or(1).max(1) as usize
    }
}

/// The whole attention block of one layer on the host: `q_a -> q_norm -> q_b -> rope`, the window KV, the compressed
/// plane and its indexer, sink attention over the concatenation, the inverse rotation, and the grouped output
/// projection (model.py:766).
///
/// This is the executable form of the specification, and the thing the kernels are checked against. `x` is
/// `[n][dim]`, the output is `[n][dim]`.
#[derive(Clone, Debug)]
pub struct MlaDsv4Host {
    pub cfg: Dsv4Config,
    pub layer_id: usize,
    pub wq_a: Fp8Host,
    pub wq_b: Fp8Host,
    pub wkv: Fp8Host,
    pub wo_a: Fp8Host,
    pub wo_b: Fp8Host,
    pub q_norm: Vec<f32>,
    pub kv_norm: Vec<f32>,
    pub sink: Vec<f32>,
    /// This layer's rotation ([`Rope::for_layer`]): `compress_rope_theta` + YaRN when the layer compresses.
    pub rope: Rope,
    /// The compressed plane's rotation ([`Rope::compress`]). Equal to [`Self::rope`] for every layer that has a
    /// compressed plane, and unused by a layer that does not.
    pub crope: Rope,
    pub compressor: Option<CompressorHost>,
    pub indexer: Option<IndexerHost>,
}

impl MlaDsv4Host {
    /// Rotates the last `rope_dim` of each of the `sub` `block`-wide blocks of each of `rows` rows of `x` at
    /// `pos0 + r`.
    #[allow(clippy::too_many_arguments)]
    fn rope_rows(&self, x: &mut [f32], rows: usize, stride: usize, block: usize, sub: usize, pos0: usize, inverse: bool) {
        self.rope.apply(x, rows, stride, block, sub, pos0, inverse);
    }

    /// `planes` holds this layer's window ring; `shared` is the group's planes ([`Dsv4Config::kv_source_of`]) — the
    /// compressed plane, its index keys, the level-1 mask and the published selection — and a source layer passes `None`
    /// for its own. (The device twin takes the same two, but only by shared reference: there is no `&mut` on a device
    /// buffer, so [`MlaDsv4::forward`] cannot alias one the way this one does.)
    pub fn forward(
        &mut self,
        x: &[f32],
        n: usize,
        start_pos: usize,
        planes: &mut HostPlanes,
        shared: Option<&mut HostPlanes>,
    ) -> Vec<f32> {
        let cfg = self.cfg.clone();
        let (h, d, eps) = (cfg.n_heads, cfg.head_dim, cfg.eps);
        assert!((1..=MAX_ROWS).contains(&n));
        assert_eq!(x.len(), n * cfg.dim);

        // --- Q: q_a -> q_norm -> q_b -> rope on each head's tail
        let mut qr = self.wq_a.matmul(x, n);
        for r in 0..n {
            rms_norm(&mut qr[r * cfg.q_lora..][..cfg.q_lora], &self.q_norm, eps);
        }
        let mut q = self.wq_b.matmul(&qr, n);
        self.rope_rows(&mut q, n, h * d, d, h, start_pos, false);

        // --- window KV: wkv -> kv_norm -> rope -> fp8 (block 32, E8M0 scales)
        let mut kv = self.wkv.matmul(x, n);
        for r in 0..n {
            rms_norm(&mut kv[r * d..][..d], &self.kv_norm, eps);
        }
        self.rope_rows(&mut kv, n, d, d, 1, start_pos, false);
        quant_dequant(&mut kv, 32, 8, true);

        let win = cfg.window;
        let mut combined: Vec<f32>;
        if start_pos == 0 {
            if n <= win {
                planes.window[..n * d].copy_from_slice(&kv);
            } else {
                // ring: the last `win` tokens, with the wrap at `n % win`
                let cutoff = n % win;
                let last = &kv[(n - win) * d..];
                planes.window[cutoff * d..][..(win - cutoff) * d].copy_from_slice(&last[..(win - cutoff) * d]);
                planes.window[..cutoff * d].copy_from_slice(&last[(win - cutoff) * d..]);
            }
            combined = kv.clone();
        } else {
            let slot = start_pos % win;
            planes.window[slot * d..][..d].copy_from_slice(&kv[..d]);
            combined = planes.window[..win * d].to_vec();
        }
        let win_rows = combined.len() / d;
        debug_assert_eq!(win_rows, if start_pos == 0 { n } else { win });

        // --- compressed plane, and the selection over it
        let ratio = cfg.ratio(self.layer_id) as usize;
        let mut lists: Vec<Vec<i32>> = window_idxs(win, n, start_pos);
        let sel: &mut HostPlanes = match shared {
            Some(s) => s,
            None => planes,
        };
        if ratio > 0 {
            let src = cfg.kv_source_of(self.layer_id).expect("a compressing layer has a source");
            let compress_len = (start_pos + n) / ratio;
            let mut latent = if src == self.layer_id {
                self.compressor.as_mut().expect("a KV source has a compressor").forward(x, n, start_pos)
            } else {
                None
            };
            // the indexer reads the latent before RoPE, so it runs before the cache is written
            let picked = match &self.indexer {
                Some(ix) => ix.forward(x, &qr, latent.as_deref(), n, start_pos, win_rows, ratio, &self.crope, sel),
                None => sel.topk_idxs[..n * cfg.index_topk].to_vec(),
            };
            sel.topk_idxs[..n * cfg.index_topk].copy_from_slice(&picked);

            if let Some(l) = latent.as_mut() {
                // a latent stands for the first token of its group: group j takes position j * ratio
                let groups = l.len() / d;
                for g in 0..groups {
                    let pos = if start_pos == 0 { g * ratio } else { start_pos + 1 - ratio };
                    self.rope.apply(&mut l[g * d..][..d], 1, d, d, 1, pos, false);
                }
                // the compressed plane is FP4 with E4M3 scales, block 16
                quant_dequant(l, 16, 4, false);
                let at = start_pos / ratio;
                sel.compress[at * d..][..groups * d].copy_from_slice(l);
            }
            sel.n_compress = compress_len;
            combined.extend_from_slice(&sel.compress[..compress_len * d]);
            for r in 0..n {
                for k in 0..cfg.index_topk {
                    let v = picked[r * cfg.index_topk + k];
                    if v >= 0 {
                        lists[r].push(v);
                    }
                }
            }
        } else {
            sel.n_compress = 0;
        }

        // --- sink attention over the concatenated planes
        let scale = 1.0 / (d as f32).sqrt();
        let mut o = vec![0f32; n * h * d];
        for r in 0..n {
            for hh in 0..h {
                let qh = &q[(r * h + hh) * d..][..d];
                let mut mx = -1e30f32;
                for &idx in &lists[r] {
                    if idx < 0 {
                        continue;
                    }
                    let kk = &combined[idx as usize * d..][..d];
                    let mut acc = 0f64;
                    for i in 0..d {
                        acc += qh[i] as f64 * kk[i] as f64;
                    }
                    mx = mx.max(acc as f32 * scale);
                }
                let mut sum = 0f64;
                let mut acc = vec![0f64; d];
                for &idx in &lists[r] {
                    if idx < 0 {
                        continue;
                    }
                    let kk = &combined[idx as usize * d..][..d];
                    let mut s = 0f64;
                    for i in 0..d {
                        s += qh[i] as f64 * kk[i] as f64;
                    }
                    let e = ((s as f32) * scale - mx).exp() as f64;
                    sum += e;
                    for i in 0..d {
                        acc[i] += e * kk[i] as f64;
                    }
                }
                // the sink is one extra logit with no value: it only renormalises
                sum += ((self.sink[hh] - mx).exp()) as f64;
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for i in 0..d {
                    o[(r * h + hh) * d + i] = (acc[i] * inv) as f32;
                }
            }
        }

        // --- undo the query rotation, then the grouped output projection
        self.rope_rows(&mut o, n, h * d, d, h, start_pos, true);
        let mut oq = o;
        quant_dequant(&mut oq, 32, 8, true);
        let (g, olora, oin) = (cfg.o_groups, cfg.o_lora, cfg.o_in_per_group());
        let mut ol = vec![0f32; n * g * olora];
        for r in 0..n {
            for gi in 0..g {
                let xg = &oq[r * h * d + gi * oin..][..oin];
                let wg = &self.wo_a.w[gi * olora * oin..][..olora * oin];
                for j in 0..olora {
                    let mut acc = 0f64;
                    for c in 0..oin {
                        acc += xg[c] as f64 * wg[j * oin + c] as f64;
                    }
                    ol[(r * g + gi) * olora + j] = acc as f32;
                }
            }
        }
        self.wo_b.matmul(&ol, n)
    }
}

// ---------------------------------------------------------------------------------------------- the block, on the device
//
// The pipeline below mirrors `MlaDsv4Host::forward` stage for stage; `csrc/dsv4_attn.cu` is the other half of each
// call. Every geometry that differs between this pack and GLM-5.3 (top-k length, index head count, group count, ratio,
// row stride of a selection) is passed at launch: the kernels in `csrc/dsa*.cu` cannot serve a 512-token selection
// because `DSA_TOPK`/`DSA_SEL_POOLS` are compile-time constants there.
/// Drive the level-1 candidate kernels directly over host-supplied scores (`rows x width` f32,
/// already on the device) into `mask` (`rows x mask_stride` u8): the diagnostic/test surface for
/// block counts past the shared-memory cliff, where a serving-shaped call cannot be replayed
/// bit-comparably because later stages of the same forward reuse the score buffer.
#[allow(clippy::too_many_arguments)]
pub fn candidate_blocks_dev(
    score: &DevBuf,
    bscore: &mut DevBuf,
    mask: &mut DevBuf,
    mask_stride: usize,
    rows: usize,
    width: usize,
    start_pos: usize,
    ratio: usize,
    topk_blocks: usize,
    block_size: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    assert!(score.bytes() >= rows * width * 4, "score staging too small");
    assert!(bscore.bytes() >= rows * width.div_ceil(block_size) * 4, "bscore staging too small");
    assert!(mask.bytes() >= rows * mask_stride, "mask staging too small");
    unsafe {
        CudaError::check(ffi::gb_dsv4_candidate_blocks(
            score.ptr() as *const f32,
            bscore.ptr() as *mut f32,
            mask.ptr() as *mut u8,
            mask_stride as i32,
            rows as i32,
            width as i32,
            start_pos as i32,
            ratio as i32,
            topk_blocks as i32,
            block_size as i32,
            st.raw(),
        ))?;
    }
    Ok(())
}

/// Device ABI codes for the element type of an activation plane (the [`Dtype`] discriminants).
fn dt(d: Dtype) -> i32 {
    d as i32
}

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
            /// `y [m, n] = x [m, k] @ w [n, k]^T`, fp16 in, bf16 weights, fp32 accumulate, `y` fp16 or fp32 by `y_f32`.
            pub fn gb_bf16_linear(
                x: *const c_void,
                w: *const c_void,
                y: *mut c_void,
                m: i32,
                k: i32,
                n: i32,
                y_f32: i32,
                stream: *mut c_void,
            ) -> i32;
            /// Rotate the last `dim` of every `stride`-wide row in place, `cos`/`sin` looked up by absolute position.
            pub fn gb_dsv4_q_rope_rot_f16(
                x: *mut c_void,
                rows: i32,
                stride: i32,
                sub: i32,
                cos: *const f32,
                sin: *const f32,
                table_stride: i32,
                pos0: i32,
                pos_step: i32,
                do_rot: i32,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dsv4_o_rot_rope_quant_f16(
                x: *mut c_void,
                rows: i32,
                stride: i32,
                sub: i32,
                cos: *const f32,
                sin: *const f32,
                table_stride: i32,
                pos0: i32,
                pos_step: i32,
                do_rot: i32,
                stream: *mut c_void,
            ) -> i32;
            /// Fused `rmsnorm -> rope -> quant(block 32, 8-bit, pow2) -> hadamard` over one row block, in place.
            pub fn gb_dsv4_kv_prepare(
                x: *mut c_void,
                w: *const f32,
                rows: i32,
                width: i32,
                rope_dim: i32,
                rope_block: i32,
                cos: *const f32,
                sin: *const f32,
                table_stride: i32,
                pos0: i32,
                pos_step: i32,
                do_rot: i32,
                eps: f32,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dsv4_rope(
                x: *mut c_void,
                dtype: i32,
                rows: i32,
                stride: i32,
                dim: i32,
                block: i32,
                sub: i32,
                cos: *const f32,
                sin: *const f32,
                table_stride: i32,
                pos0: i32,
                pos_step: i32,
                inverse: i32,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dsv4_rope_params(
                x: *mut c_void,
                dtype: i32,
                rows: i32,
                stride: i32,
                dim: i32,
                block: i32,
                sub: i32,
                cos: *const f32,
                sin: *const f32,
                table_stride: i32,
                pos_off: i32,
                pos_step: i32,
                inverse: i32,
                params: *const c_void,
                stream: *mut c_void,
            ) -> i32;
            /// Quantise-dequantise in place over `block`-wide groups: `bits` 8 (E4M3) or 4 (E2M1), `pow2_scale` picks E8M0.
            pub fn gb_dsv4_quant(
                x: *mut c_void,
                dtype: i32,
                rows: i32,
                width: i32,
                block: i32,
                bits: i32,
                pow2_scale: i32,
                stream: *mut c_void,
            ) -> i32;
            /// `x[r] = rms_norm(x[r]) * w` in place, f32 math, one rounding back to the element type.
            pub fn gb_dsv4_rmsnorm_f32_to_f16(
                x: *const f32,
                w: *const f32,
                y: *mut c_void,
                rows: i32,
                cols: i32,
                eps: f32,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dsv4_rmsnorm(
                x: *mut c_void,
                w: *const f32,
                dtype: i32,
                rows: i32,
                width: i32,
                eps: f32,
                stream: *mut c_void,
            ) -> i32;
            /// Sink attention: `out[r][h] = sum_j softmax_j(q.k) v_j` over the `idx_len` gathered rows, the sink one extra logit.
            /// Index `< split` addresses `kv_win`, the rest `kv_cmp` — the two planes are read where they live, never concatenated.
            #[allow(clippy::too_many_arguments)]
            pub fn gb_dsv4_attn(
                q: *const c_void,
                kv_win: *const c_void,
                kv_cmp: *const c_void,
                split: i32,
                sink: *const f32,
                idx: *const i32,
                idx_stride: i32,
                idx_len: i32,
                out: *mut c_void,
                rows: i32,
                n_heads: i32,
                head_dim: i32,
                scale: f32,
                dtype: i32,
                stream: *mut c_void,
            ) -> i32;
            /// Block-diagonal `wo_a`: group `g` projects its own `o_in` columns to `o_lora`, FP8 weights, runtime group count.
            /// `use_classic` forces the classic (non-pipelined) kernel and is only used by the A/B tests.
            pub fn gb_dsv4_woa_grouped(
                o: *const c_void,
                w: *const u8,
                s: *const u8,
                out: *mut c_void,
                rows: i32,
                groups: i32,
                o_lora: i32,
                o_in: i32,
                dtype: i32,
                use_classic: i32,
                force_mma: i32,
                stream: *mut c_void,
            ) -> i32;
            /// Softmax-gate pooling of `ratio` tokens into one latent (`ratio == 1` is a plain copy).
            pub fn gb_dsv4_pool(
                kv: *const c_void,
                score: *const c_void,
                out: *mut c_void,
                rows: i32,
                head_dim: i32,
                ratio: i32,
                n_out: i32,
                dtype: i32,
                stream: *mut c_void,
            ) -> i32;
            /// The decode form: pool the completed group out of the carry-over state.
            pub fn gb_dsv4_pool_state(
                kv_state: *const f32,
                score_state: *const f32,
                out: *mut c_void,
                head_dim: i32,
                ratio: i32,
                dtype: i32,
                stream: *mut c_void,
            ) -> i32;
            /// One token into slot `slot` of the carry-over state.
            pub fn gb_dsv4_state_write(
                kv: *const c_void,
                score: *const c_void,
                kv_state: *mut f32,
                score_state: *mut f32,
                head_dim: i32,
                slot: i32,
                dtype: i32,
                stream: *mut c_void,
            ) -> i32;
            /// Index scores `[rows][n_pos]`: `sum_h relu(q_h . k_p) * w_h`, masked to the reachable positions and, when `mask`
    /// is not null, to the level-1 candidate blocks the source published.
            #[allow(clippy::too_many_arguments)]
            pub fn gb_dsv4_index_score(
                q: *const c_void,
                index_k: *const c_void,
                w: *const f32,
                mask: *const u8,
                mask_stride: i32,
                score: *mut f32,
                rows: i32,
                n_pos: i32,
                n_heads: i32,
                dim: i32,
                scale: f32,
                start_pos: i32,
                ratio: i32,
                dtype: i32,
                stream: *mut c_void,
            ) -> i32;
            /// `topk` best positions per row, ascending, `+ val_offset`, `-1` past the count, written at `col_offset`.
            pub fn gb_dsv4_topk_select(
                score: *const f32,
                out: *mut i32,
                rows: i32,
                n_pos: i32,
                topk: i32,
                col_offset: i32,
                val_offset: i32,
                row_stride: i32,
                stream: *mut c_void,
            ) -> i32;
            #[allow(dead_code)]
            pub fn gb_dsv4_window_idxs(
                out: *mut i32,
                rows: i32,
                split: i32,
                row_stride: i32,
                win: i32,
                start_pos: i32,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dsv4_window_and_cmp_idxs(
                out: *mut i32,
                rows: i32,
                split: i32,
                row_stride: i32,
                win: i32,
                start_pos: i32,
                cmp_topk: i32,
                ratio: i32,
                stream: *mut c_void,
            ) -> i32;
            /// Level 1: the `topk_blocks` best blocks per row as a bool mask over positions.
            /// `bscore` is caller scratch for the per-row block-max scores (`rows x ceil(width/block_size)`
            /// f32) -- the selection reads them back from gmem because the array outgrows shared memory
            /// at the long-context block counts (48 KB default; 512K-1M contexts need 256-512 KB).
            pub fn gb_dsv4_candidate_blocks(
                logits: *const f32,
                bscore: *mut f32,
                mask: *mut u8,
                mask_stride: i32,
                rows: i32,
                width: i32,
                start_pos: i32,
                ratio: i32,
                topk_blocks: i32,
                block_size: i32,
                stream: *mut c_void,
            ) -> i32;
            /// Copy `rows` rows of `width` into a plane at `dst_start + r`.
            pub fn gb_dsv4_plane_write(
                src: *const c_void,
                plane: *mut c_void,
                rows: i32,
                width: i32,
                dst_start: i32,
                dtype: i32,
                stream: *mut c_void,
            ) -> i32;
            /// Copy `rows` rows of `width` into a ring of `n_slots` at `(slot0 + r) % n_slots`.
            pub fn gb_dsv4_ring_write(
                src: *const c_void,
                ring: *mut c_void,
                rows: i32,
                width: i32,
                slot0: i32,
                n_slots: i32,
                dtype: i32,
                stream: *mut c_void,
            ) -> i32;
            pub fn gb_dsv4_ring_write_params(
                src: *const c_void,
                ring: *mut c_void,
                rows: i32,
                width: i32,
                params: *const c_void,
                dtype: i32,
                stream: *mut c_void,
            ) -> i32;
        }
}

/// Every plane a DSV4.1 layer reads and writes, plus the selection it shares with the layers that reuse it: the device
/// twin of [`HostPlanes`], with the same addressing.
#[derive(Debug)]
pub struct Dsv4Planes {
    /// `[window][head_dim]` ring of raw KV latents, fp16 (`win_fmt`).
    pub window: DevBuf,
    pub window_bak: DevBuf,
    /// `[n_cmp][head_dim]` compressed plane (one per KV source group; a reader shares its source's), `fmt`.
    pub compress: DevBuf,
    /// `[n_cmp][index_dim]` index keys, `fmt`.
    pub index_k: DevBuf,
    /// `u8 [MAX_ROWS][n_cmp]` level-1 block mask (`1` = allowed), one row per query.
    pub candidates: DevBuf,
    /// `i32 [MAX_ROWS][window + index_topk]` the published selection, into the combined plane: the `split` window slots
    /// first, then the compressed positions (model.py:779 `cat([topk_idxs, compress_idxs])`). A source layer's indexer
    /// writes the second part, [`window_idxs_dev`] the first.
    pub topk: DevBuf,
    /// Elements between rows of [`Self::topk`]: `window + index_topk`.
    pub sel_stride: usize,
    pub max_seq: usize,
    pub window_slots: usize,
    /// Bytes between rows of [`Self::candidates`].
    pub cand_stride: usize,
    /// Compressed entries this layer's plane holds ([`compress_slots`]).
    pub n_cmp: usize,
    /// The compressed plane's and the indexer keys' format ([`crate::dsv4_kvq4::KvFmt`]).
    pub fmt: crate::dsv4_kvq4::KvFmt,
    /// The sliding-window ring's format: fp16 unless the cache was built q4-window, which the microbenchmark decides.
    pub win_fmt: crate::dsv4_kvq4::KvFmt,
}

impl Dsv4Planes {
    /// Bytes per compressed position across all four planes: one `head_dim` entry, one `index_dim` key and one mask column
    /// ([`MAX_ROWS`] bytes, a row a query) — the derivative of [`Self::bytes`] in `max_seq`, which is what the scheduler
    /// prices a token with.
    pub fn bytes_per_position(cfg: &Dsv4Config, fmt: crate::dsv4_kvq4::KvFmt) -> usize {
        use crate::dsv4_kvq4::ROPE_DIMS;
        fmt.row_bytes(cfg.head_dim, ROPE_DIMS) + fmt.row_bytes(cfg.index_dim, 0) + PLANE_ROWS
    }
    /// The five buffers [`Self::new`] allocates, by the same formulas: the window ring, the compressed plane, the indexer's keys, the
    /// level-1 candidate mask and the published selection. Admission charges `layers x slots` of this, so it is stated once here and
    /// `new` allocates from it — the price and the allocation cannot drift.
    pub fn bytes(cfg: &Dsv4Config, max_seq: usize, layer: usize, fmt: crate::dsv4_kvq4::KvFmt) -> usize {
        // the window ring is fp16 in both layouts ([`Self::new`]); only the compressed plane and the indexer's keys move
        Self::bytes_of(cfg, max_seq, layer, fmt, crate::dsv4_kvq4::KvFmt::F16)
    }
    /// [`Self::bytes`] with the window ring's format stated separately (it is fp16 by default).
    pub fn bytes_of(
        cfg: &Dsv4Config,
        max_seq: usize,
        layer: usize,
        fmt: crate::dsv4_kvq4::KvFmt,
        win_fmt: crate::dsv4_kvq4::KvFmt,
    ) -> usize {
        use crate::dsv4_kvq4::ROPE_DIMS;
        let nc = compress_slots(cfg, layer, max_seq);
        cfg.window * win_fmt.row_bytes(cfg.head_dim, ROPE_DIMS)
            + nc * fmt.row_bytes(cfg.head_dim, ROPE_DIMS)
            + nc * fmt.row_bytes(cfg.index_dim, 0)
            + PLANE_ROWS * nc
            + PLANE_ROWS * (cfg.window + cfg.index_topk) * 4
    }
    pub fn new(cfg: &Dsv4Config, max_seq: usize, layer: usize, fmt: crate::dsv4_kvq4::KvFmt) -> Result<Self, CudaError> {
        Self::with_window_fmt(cfg, max_seq, layer, fmt, crate::dsv4_kvq4::KvFmt::F16)
    }
    /// [`Self::new`] with the window ring's format stated separately.
    pub fn with_window_fmt(
        cfg: &Dsv4Config,
        max_seq: usize,
        layer: usize,
        fmt: crate::dsv4_kvq4::KvFmt,
        win_fmt: crate::dsv4_kvq4::KvFmt,
    ) -> Result<Self, CudaError> {
        use crate::dsv4_kvq4::ROPE_DIMS;
        let nc = compress_slots(cfg, layer, max_seq);
        let align256 = |n: usize| (n + 255) & !255;
        let b_win = cfg.window * win_fmt.row_bytes(cfg.head_dim, ROPE_DIMS);
        let b_wbak = (cfg.window + 64) * win_fmt.row_bytes(cfg.head_dim, ROPE_DIMS);
        let b_cmp = nc * fmt.row_bytes(cfg.head_dim, ROPE_DIMS);
        let b_ik = nc * fmt.row_bytes(cfg.index_dim, 0);
        let b_cand = if max_seq >= 4096 && layer != cfg.candidate_source_layer { PLANE_ROWS } else { PLANE_ROWS * nc };
        let b_topk = if max_seq == 1 && layer >= 2 && !cfg.is_kv_source(layer) {
            64
        } else {
            PLANE_ROWS * (cfg.window + cfg.index_topk) * 4
        };
        let o0 = 0usize;
        let o1 = o0 + align256(b_win);
        let o2 = o1 + align256(b_wbak);
        let o3 = o2 + align256(b_cmp);
        let o4 = o3 + align256(b_ik);
        let o5 = o4 + align256(b_cand);
        let total = o5 + align256(b_topk);
        let arena =
            if max_seq > 786_432 { DevBuf::zeroed_pageable_view(total.max(256))? } else { DevBuf::zeroed_view(total.max(256))? };
        let blk = arena.block().expect("zeroed_view has a block").clone();
        Ok(Self {
            window: DevBuf::view(&blk, o0, b_win),
            window_bak: DevBuf::view(&blk, o1, b_wbak),
            compress: DevBuf::view(&blk, o2, b_cmp),
            index_k: DevBuf::view(&blk, o3, b_ik),
            candidates: DevBuf::view(&blk, o4, b_cand),
            topk: DevBuf::view(&blk, o5, b_topk),
            sel_stride: cfg.window + cfg.index_topk,
            max_seq,
            window_slots: cfg.window,
            cand_stride: nc,
            n_cmp: nc,
            fmt,
            win_fmt,
        })
    }
    /// The bytes the five buffers actually hold — what [`Self::bytes`] predicts.
    pub fn held_bytes(&self) -> usize {
        self.window.bytes() + self.compress.bytes() + self.index_k.bytes() + self.candidates.bytes() + self.topk.bytes()
    }
    /// Compressed positions the planes hold.
    pub fn capacity(&self, cfg: &Dsv4Config) -> usize {
        use crate::dsv4_kvq4::ROPE_DIMS;
        self.compress.bytes() / self.fmt.row_bytes(cfg.head_dim, ROPE_DIMS)
    }
}

/// Activations and scores of one block call: `f16` everywhere the kernels read or write an activation, `f32` for scores.
#[derive(Debug)]
pub struct Dsv4Scratch {
    pub qr: DevBuf,
    pub q: DevBuf,
    pub kv: DevBuf,
    /// The compressor's own `wkv` output (a different weight from [`Self::kv`]).
    pub ckv: DevBuf,
    pub gate: DevBuf,
    pub o: DevBuf,
    pub oq: DevBuf,
    pub olora: DevBuf,
    pub out: DevBuf,
    pub score: DevBuf,
    pub latent: DevBuf,
    pub index_q: DevBuf,
    pub index_kt: DevBuf,
    pub index_w: DevBuf,
    pub rows: usize,
    pub max_seq: usize,
}

impl Dsv4Scratch {
    fn layout(cfg: &Dsv4Config, rows: usize, max_seq: usize) -> (usize, usize, usize, usize, [usize; 7], usize, usize, usize) {
        let align4k = |n: usize| (n + 4095) & !4095;
        let hd = |n: usize| n * 2;
        let nc = compress_slots_max(cfg, max_seq);
        let score_rows = if rows * nc <= 2048 * 8193 { rows } else { (2048 * 8193 / nc.max(1)).max(1).min(rows) };
        let b_ckv = align4k(hd(rows * cfg.head_dim));
        let b_gate = align4k(hd(rows * cfg.head_dim));
        let b_lat = align4k(hd(rows * cfg.head_dim));
        let b_ikt = align4k(hd(rows * cfg.index_dim));
        let b_iq = align4k(hd(rows * cfg.index_heads * cfg.index_dim));
        let b_iw = align4k(rows * cfg.index_heads * 4);
        let b_score = align4k(score_rows * nc * 4);
        let off_ckv = 0usize;
        let off_gate = off_ckv + b_ckv;
        let off_lat = off_gate + b_gate;
        let off_ikt = off_lat + b_lat;
        let off_iq = off_ikt + b_ikt;
        let off_iw = off_iq + b_iq;
        let off_score = off_iw + b_iw;
        let aux_end = off_score + b_score;
        let qo_bytes = hd(rows * cfg.n_heads * cfg.head_dim).max(aux_end);

        let win_stage = align4k((cfg.window + rows) * cfg.head_dim * 2);
        let topk_stage = align4k(rows * (cfg.window + cfg.index_topk) * 4);
        let cand_off = win_stage + topk_stage;
        // Both sizes are padded to 16 so the f32 `bscore` staging that follows the larger of the two
        // starts 16-aligned (its base is `cand_off + max(olora_bytes, cand_bytes)`; an unpadded size
        // puts the `float* bscore` array at a byte offset that is not even a 4-byte multiple, which
        // CUDA rejects with error 716 once the candidate-block path engages).
        let align16 = |n: usize| (n + 15) & !15;
        let olora_bytes = align16(hd(rows * cfg.o_groups * cfg.o_lora));
        let cand_bytes = align16(score_rows * nc);
        // the candidate selection's per-row block scores: one f32 per 8-position block, every score row,
        // stacked after the candidate-mask staging so the two never alias
        let bscore_bytes = align4k(score_rows * nc.div_ceil(8) * 4);
        let bscore_off = cand_off + olora_bytes.max(cand_bytes);
        let oq_olora_bytes = bscore_off + bscore_bytes;
        (
            hd(rows * cfg.q_lora),
            hd(rows * cfg.head_dim),
            qo_bytes,
            oq_olora_bytes,
            [off_ckv, off_gate, off_lat, off_ikt, off_iq, off_iw, off_score],
            score_rows * nc * 4,
            cand_off,
            hd(rows * cfg.dim),
        )
    }

    /// Byte offset of the candidate-selection block-score staging inside [`Dsv4Scratch::oq`]: rows of
    /// `ceil(compress_slots_max / candidate_block_size)` f32, one per score row.
    pub fn bscore_off(cfg: &Dsv4Config, rows: usize, max_seq: usize) -> usize {
        let align4k = |n: usize| (n + 4095) & !4095;
        let align16 = |n: usize| (n + 15) & !15;
        let nc = compress_slots_max(cfg, max_seq);
        let score_rows = if rows * nc <= 2048 * 8193 { rows } else { (2048 * 8193 / nc.max(1)).max(1).min(rows) };
        let win_stage = align4k((cfg.window + rows) * cfg.head_dim * 2);
        let topk_stage = align4k(rows * (cfg.window + cfg.index_topk) * 4);
        let cand_off = win_stage + topk_stage;
        let olora_bytes = align16(rows * cfg.o_groups * cfg.o_lora * 2);
        let cand_bytes = align16(score_rows * nc);
        let off = cand_off + olora_bytes.max(cand_bytes);
        debug_assert_eq!(off % 16, 0, "bscore staging must stay 16-aligned");
        off
    }

    /// Bytes of the candidate-selection block-score staging (what [`Self::bscore_off`] sits on).
    pub fn bscore_bytes(cfg: &Dsv4Config, rows: usize, max_seq: usize) -> usize {
        let nc = compress_slots_max(cfg, max_seq);
        let score_rows = if rows * nc <= 2048 * 8193 { rows } else { (2048 * 8193 / nc.max(1)).max(1).min(rows) };
        score_rows * nc.div_ceil(8) * 4
    }

    pub fn bytes(cfg: &Dsv4Config, rows: usize, max_seq: usize) -> usize {
        let (qr_b, kv_b, qo_b, oq_olora_b, _, _, _, out_b) = Self::layout(cfg, rows, max_seq);
        qr_b + kv_b + qo_b + oq_olora_b + out_b
    }

    pub fn new(cfg: &Dsv4Config, rows: usize, max_seq: usize) -> Result<Self, CudaError> {
        Self::new_with_shared_block(cfg, rows, max_seq, None)
    }

    pub fn new_with_shared_block(
        cfg: &Dsv4Config,
        rows: usize,
        max_seq: usize,
        shared_blk: Option<&std::sync::Arc<crate::rt::Block>>,
    ) -> Result<Self, CudaError> {
        let hd = |n: usize| n * 2;
        let align4k = |n: usize| (n + 4095) & !4095;
        let (qr_b, kv_b, qo_b, oq_olora_b, offs, score_b, cand_off, out_b) = Self::layout(cfg, rows, max_seq);
        let qo_aligned = align4k(qo_b);
        let oq_aligned = align4k(oq_olora_b);
        let qr_aligned = align4k(qr_b);
        let kv_aligned = align4k(kv_b);
        let _ = oq_aligned;
        let total_shared = qo_aligned + qr_aligned + kv_aligned;
        let oq_view = DevBuf::zeroed_view(oq_olora_b)?;
        let oq_blk = oq_view.block().expect("zeroed_view has a block").clone();
        let oq_base = 0usize;
        let (qo_blk, qr_view, kv_view) = match shared_blk {
            Some(blk) if blk.bytes() >= total_shared => {
                let b = (*blk).clone();
                let qr_v = DevBuf::view(&b, qo_aligned, qr_b);
                let kv_v = DevBuf::view(&b, qo_aligned + qr_aligned, kv_b);
                (b, qr_v, kv_v)
            }
            Some(blk) if blk.bytes() >= qo_b => ((*blk).clone(), DevBuf::zeroed_view(qr_b)?, DevBuf::zeroed_view(kv_b)?),
            _ => {
                let b = DevBuf::zeroed_view(qo_b)?.block().expect("zeroed_view has a block").clone();
                (b, DevBuf::zeroed_view(qr_b)?, DevBuf::zeroed_view(kv_b)?)
            }
        };
        let q_bytes = hd(rows * cfg.n_heads * cfg.head_dim);
        let h_bytes = hd(rows * cfg.head_dim);
        let olora_bytes = ((rows * cfg.o_groups * cfg.o_lora * 2) + 15) & !15;
        Ok(Self {
            qr: qr_view,
            q: DevBuf::view(&qo_blk, 0, q_bytes),
            kv: kv_view,
            ckv: DevBuf::view(&qo_blk, offs[0], h_bytes),
            gate: DevBuf::view(&qo_blk, offs[1], h_bytes),
            o: DevBuf::view(&qo_blk, 0, q_bytes),
            oq: oq_view,
            olora: DevBuf::view(&oq_blk, oq_base + cand_off, olora_bytes),
            out: DevBuf::zeroed_view(out_b)?,
            score: DevBuf::view(&qo_blk, offs[6], score_b),
            latent: DevBuf::view(&qo_blk, offs[2], h_bytes),
            index_q: DevBuf::view(&qo_blk, offs[4], hd(rows * cfg.index_heads * cfg.index_dim)),
            index_kt: DevBuf::view(&qo_blk, offs[3], hd(rows * cfg.index_dim)),
            index_w: DevBuf::view(&qo_blk, offs[5], rows * cfg.index_heads * 4),
            rows,
            max_seq,
        })
    }
}

/// One sequence slice inside a batched multi-slot [`MlaDsv4::forward_batch`] call.
pub struct MlaSeq<'a> {
    pub slot: usize,
    pub row0: usize,
    pub rows: usize,
    pub start_pos: usize,
    pub planes: &'a Dsv4Planes,
    pub shared: Option<&'a Dsv4Planes>,
}

/// Host snapshot of a slot's compressor carry-over state (`kv_state` and `score_state`).
#[derive(Clone, Debug, PartialEq)]
pub struct CompressorStateSnap {
    pub kv_state: Vec<f32>,
    pub score_state: Vec<f32>,
    pub has_carry: bool,
}

/// The compressor's device weights and its carry-over state (`ratio > 1` only).
#[derive(Debug)]
pub struct DevCompressor {
    pub ratio: usize,
    pub wkv: Bf16W,
    pub wgate: Option<Bf16W>,
    pub norm: DevBuf,
    pub kv_state: DevBuf,
    pub score_state: DevBuf,
    pub ckv_stage: DevBuf,
    pub gate_stage: DevBuf,
}

/// The indexer's device weights.
#[derive(Debug)]
pub struct DevIndexer {
    pub heads: usize,
    pub dim: usize,
    pub topk: usize,
    pub wq_b: Fp8Weights,
    pub weights_proj: Bf16W,
    pub wk: Option<Bf16W>,
    pub k_norm: DevBuf,
}

/// The pack's host bytes for one layer (`layers.N.attn.*`), as they come out of the safetensors file.
///
/// Every FP8 pair is `(E4M3 elements, one E8M0 byte per 32x32 tile)`. The compressor's and the indexer's small matrices
/// are BF16 in this pack (`compressor.wkv`, `compressor.wgate`, `indexer.wk`, `indexer.weights_proj` — no `.scale` beside
/// any of them), so they arrive as raw BF16 bytes and [`MlaDsv4::with_rope`] runs them through
/// [`crate::fp8_linear::pack_bf16`] on the way in; `indexer.wq_b` is the one that is already FP8.
#[derive(Clone, Copy)]
pub struct MlaDsv4Pack<'a> {
    pub wq_a: (&'a [u8], &'a [u8]),
    pub wq_b: (&'a [u8], &'a [u8]),
    pub wkv: (&'a [u8], &'a [u8]),
    pub wo_a: (&'a [u8], &'a [u8]),
    pub wo_b: (&'a [u8], &'a [u8]),
    pub q_norm: &'a [f32],
    pub kv_norm: &'a [f32],
    pub sink: &'a [f32],
    /// `Some` exactly on the layers in `kv_source_layer_ids`.
    pub compressor: Option<CompressorPack<'a>>,
    /// `Some` exactly on the layers in `index_source_layer_ids`.
    pub indexer: Option<IndexerPack<'a>>,
}

/// One layer's compressor, as the checkpoint holds it: `wkv` `[head_dim, dim]`, the gate `[head_dim, dim]` where an entry
/// spans more than one token, and the post-pooling norm's gamma `[head_dim]` — all BF16 without scales.
#[derive(Clone, Copy)]
pub struct CompressorPack<'a> {
    pub wkv: &'a [u8],
    pub wgate: Option<&'a [u8]>,
    pub norm: &'a [f32],
}

/// One selector's indexer: the query projection `[index_n_heads * index_head_dim, q_lora_rank]` (FP8, with scales), the
/// per-head weight `[index_n_heads, dim]` (BF16), and — only where the layer also owns a pool — the key projection
/// `[index_head_dim, head_dim]` and its norm gamma (BF16).
#[derive(Clone, Copy)]
pub struct IndexerPack<'a> {
    pub wq_b: (&'a [u8], &'a [u8]),
    pub weights_proj: &'a [u8],
    pub wk: Option<&'a [u8]>,
    pub k_norm: &'a [f32],
}

/// One DSV4.1 attention layer on the device: the five FP8 matrices through [`Fp8Weights`], the two norms, the sink
/// bias, both RoPE tables, and — when this layer is a source — its compressor and indexer.
#[derive(Debug)]
pub struct MlaDsv4 {
    pub cfg: Dsv4Config,
    pub layer_id: usize,
    /// The query heads THIS device computes: `wq_b`'s output rows over `head_dim`. The single-rank engine holds every head's row
    /// of `wq_b` (the pack's own matrix), so this is `cfg.n_heads` there and every launch below is the single-node one; a rank of
    /// a tensor-parallel group holds its heads' row slice of `wq_b` (`tp::ShardTable::o_row_range`, an FP8 row cut at a multiple
    /// of 32), so its `q`, its attention output `o` and the o-projection rows it owns are `q_heads` wide — the shape the weights
    /// hold decides, never a flag (`docs/NEMOTRON_TP.md`'s "the arms follow the weights' shapes").
    pub q_heads: usize,
    pub wq_a: Fp8Weights,
    pub wq_b: Fp8Weights,
    pub wkv: Fp8Weights,
    pub wo_a: Fp8Packed,
    pub wo_b: Fp8Weights,
    pub q_norm: DevBuf,
    pub kv_norm: DevBuf,
    pub sink: DevBuf,
    /// This layer's rotation (`rope_theta` + YaRN off when it does not compress, `compress_rope_theta` + YaRN on when it does).
    /// Shared: every layer with the same [`RopeSpec`] holds the same `Arc`, so the table is uploaded once for the model.
    pub rope: std::sync::Arc<RopeTables>,
    /// The compressed plane's rotation. The same `Arc` as [`Self::rope`] on a layer that compresses: one table, two names.
    pub crope: std::sync::Arc<RopeTables>,
    pub compressor: Option<DevCompressor>,
    pub indexer: Option<DevIndexer>,
}

/// An `[n, k]` E4M3 weight and its `[n/32, k/32]` E8M0 scales, kept as raw bytes: `wo_a` is consumed by a
/// block-diagonal kernel rather than a linear, but the format is exactly [`Fp8Weights`]'s (E4M3 elements, one E8M0
/// scale per 32x32 tile), which is why the host reference dequantises it with [`e4m3_to_f32`]/[`e8m0_to_f32`].
#[derive(Debug)]
pub struct Fp8Packed {
    pub n: usize,
    pub k: usize,
    pub w: DevBuf,
    pub s: DevBuf,
}

impl Fp8Packed {
    pub fn new(n: usize, k: usize, w: &[u8], s: &[u8]) -> Result<Self, Fp8Error> {
        if n == 0 || k == 0 || n % TILE != 0 || k % TILE != 0 {
            return Err(Fp8Error::NotTiled { what: "wo_a", value: n.min(k), tile: TILE });
        }
        if w.len() != n * k {
            return Err(Fp8Error::ShapeMismatch { what: "w [n, k]", expected: n * k, got: w.len() });
        }
        if s.len() != (n / TILE) * (k / TILE) {
            return Err(Fp8Error::ShapeMismatch { what: "scale [n/32, k/32]", expected: (n / TILE) * (k / TILE), got: s.len() });
        }
        if let Some(i) = s.iter().position(|&b| b == 0xFF) {
            return Err(Fp8Error::ScaleNotFinite { n_tile: i / (k / TILE), k_tile: i % (k / TILE), bits: 0xFF });
        }
        Ok(Self { n, k, w: DevBuf::from_slice(w)?, s: DevBuf::from_slice(s)? })
    }
}

/// `Fp8Error` as the CUDA error the block's signature speaks (a shape error is `cudaErrorInvalidValue`).
fn fp8_err(e: Fp8Error) -> CudaError {
    match e {
        Fp8Error::Cuda(c) => c,
        _ => CudaError(1),
    }
}

impl MlaDsv4 {
    /// Uploads one layer. `max_seq` sizes both RoPE tables.
    ///
    /// This builds the layer's own tables; a model builds them once and shares them ([`Self::new_shared`]), which is what makes 43
    /// layers cost two tables instead of 86.
    // the Arc is the layer-sharing handle by design; `RopeTables` holds device buffers the runtime guards to one
    // thread at a time, so the shared handle itself is what keeps the tables from being `Sync`
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(cfg: &Dsv4Config, layer_id: usize, pack: &MlaDsv4Pack<'_>, max_seq: usize) -> Result<Self, Fp8Error> {
        let own = RopeTables::new(RopeSpec::for_layer(cfg, layer_id, max_seq))?;
        let comp = RopeSpec::compress(cfg, max_seq);
        let (rope, crope) = if comp == own.spec {
            let a = std::sync::Arc::new(own);
            (std::sync::Arc::clone(&a), a)
        } else {
            (std::sync::Arc::new(own), std::sync::Arc::new(RopeTables::new(comp)?))
        };
        Self::with_rope(cfg, layer_id, pack, rope, crope)
    }

    /// Uploads one layer over tables the model already built ([`SharedRope`]): the same buffers every layer with the same spec
    /// points at.
    pub fn new_shared(cfg: &Dsv4Config, layer_id: usize, pack: &MlaDsv4Pack<'_>, tables: &SharedRope) -> Result<Self, Fp8Error> {
        Self::with_rope(cfg, layer_id, pack, tables.get(cfg, layer_id), tables.compress(cfg))
    }

    fn with_rope(
        cfg: &Dsv4Config,
        layer_id: usize,
        pack: &MlaDsv4Pack<'_>,
        rope: std::sync::Arc<RopeTables>,
        crope: std::sync::Arc<RopeTables>,
    ) -> Result<Self, Fp8Error> {
        cfg.validate().map_err(|_| Fp8Error::ShapeMismatch { what: "config", expected: 0, got: 1 })?;
        let (h, d, ql) = (cfg.n_heads, cfg.head_dim, cfg.q_lora);
        let wq_a = Fp8Weights::new(ql, cfg.dim, pack.wq_a.0, pack.wq_a.1)?;
        // the query heads this device holds come from the weights: a whole `wq_b` is every head's rows, a rank of a
        // tensor-parallel group holds its heads' row slice (`Fp8Weights::new` validates the slice's bytes and scales).
        // When `pack.wq_b.0` is empty, the caller supplies an EXL3 `wq_b` right after construction.
        let (q_heads, wq_b) = if pack.wq_b.0.is_empty() {
            (h, Fp8Weights::new(32, 32, &[0u8; 1024], &[127u8; 1])?)
        } else {
            let qh = pack.wq_b.0.len() / ql / d;
            if qh == 0 || qh > h || (h % qh) != 0 {
                return Err(Fp8Error::ShapeMismatch { what: "wq_b (query heads)", expected: h * d * ql, got: pack.wq_b.0.len() });
            }
            (qh, Fp8Weights::new(qh * d, ql, pack.wq_b.0, pack.wq_b.1)?)
        };
        let wkv = Fp8Weights::new(d, cfg.dim, pack.wkv.0, pack.wkv.1)?;
        let wo_a = Fp8Packed::new(cfg.o_groups * cfg.o_lora, cfg.o_in_per_group(), pack.wo_a.0, pack.wo_a.1)?;
        let wo_b = if pack.wo_b.0.is_empty() {
            Fp8Weights::new(32, 32, &[0u8; 1024], &[127u8; 1])?
        } else {
            Fp8Weights::new(cfg.dim, cfg.o_groups * cfg.o_lora, pack.wo_b.0, pack.wo_b.1)?
        };

        // --- the compressor: this layer pools its own KV exactly on the four `kv_source_layer_ids`
        let is_kv_source = cfg.kv_source_of(layer_id) == Some(layer_id);
        let compressor = match (is_kv_source, pack.compressor) {
            (true, Some(c)) => {
                let ratio = (cfg.ratio(layer_id) as usize).max(1);
                // read the pack's BF16 bytes as they are: no FP8 re-quantisation (see `csrc/dsv4_bf16.cu`)
                let bf16 = |n: usize, k: usize, raw: &[u8]| -> Result<Bf16W, Fp8Error> { Bf16W::new(n, k, raw) };
                Some(DevCompressor {
                    ratio,
                    wkv: bf16(d, cfg.dim, c.wkv)?,
                    // model.py:447: a gate exists only where an entry spans more than one token
                    wgate: match (ratio > 1, c.wgate) {
                        (true, Some(g)) => Some(bf16(d, cfg.dim, g)?),
                        (true, None) => {
                            return Err(Fp8Error::ShapeMismatch { what: "compressor.wgate", expected: d * cfg.dim, got: 0 })
                        }
                        (false, _) => None,
                    },
                    norm: DevBuf::from_slice(c.norm)?,
                    // model.py:452: `[ratio][head_dim]` of the group still filling up, carried across decode steps. Every
                    // slot is written before the pool that reads it, so zeroes are the right initial value (and, unlike
                    // the reference's `-inf`, they stay finite through a `ratio` of 2 with a single token seen).
                    kv_state: DevBuf::zeroed(MAX_SLOTS * ratio * d * 4)?,
                    score_state: DevBuf::zeroed(MAX_SLOTS * ratio * d * 4)?,
                    ckv_stage: DevBuf::zeroed(MAX_SLOTS * 64 * d * 2)?,
                    gate_stage: DevBuf::zeroed(MAX_SLOTS * 64 * d * 2)?,
                })
            }
            (true, None) => {
                return Err(Fp8Error::ShapeMismatch { what: "compressor (layer is a KV source)", expected: 1, got: 0 })
            }
            (false, Some(_)) => {
                return Err(Fp8Error::ShapeMismatch { what: "compressor (layer is not a KV source)", expected: 0, got: 1 })
            }
            (false, None) => None,
        };

        // --- the indexer: queries on every `index_source_layer_ids` layer, keys only where the layer also pools
        let is_index_source = cfg.index_source_of(layer_id) == Some(layer_id);
        let indexer = match (is_index_source, pack.indexer) {
            (true, Some(i)) => {
                // read the pack's BF16 bytes as they are: no FP8 re-quantisation (see `csrc/dsv4_bf16.cu`)
                let bf16 = |n: usize, k: usize, raw: &[u8]| -> Result<Bf16W, Fp8Error> { Bf16W::new(n, k, raw) };
                // model.py:500: the index keys come out of the compressor's latent, so only a pool owner can build them;
                // every other selector reads them from that layer's cache (one shared `index_k` plane).
                let wk = match (is_kv_source, i.wk) {
                    (true, Some(w)) => Some(bf16(cfg.index_dim, d, w)?),
                    (true, None) => {
                        return Err(Fp8Error::ShapeMismatch { what: "indexer.wk", expected: cfg.index_dim * d, got: 0 })
                    }
                    (false, _) => None,
                };
                Some(DevIndexer {
                    heads: cfg.index_heads,
                    dim: cfg.index_dim,
                    topk: cfg.index_topk,
                    wq_b: Fp8Weights::new(cfg.index_heads * cfg.index_dim, ql, i.wq_b.0, i.wq_b.1)?,
                    weights_proj: bf16(cfg.index_heads, cfg.dim, i.weights_proj)?,
                    wk,
                    k_norm: DevBuf::from_slice(i.k_norm)?,
                })
            }
            (true, None) => {
                return Err(Fp8Error::ShapeMismatch { what: "indexer (layer is an index source)", expected: 1, got: 0 })
            }
            (false, Some(_)) => {
                return Err(Fp8Error::ShapeMismatch { what: "indexer (layer is not an index source)", expected: 0, got: 1 })
            }
            (false, None) => None,
        };

        Ok(Self {
            cfg: cfg.clone(),
            layer_id,
            q_heads,
            wq_a,
            wq_b,
            wkv,
            wo_a,
            wo_b,
            q_norm: DevBuf::from_slice(pack.q_norm)?,
            kv_norm: DevBuf::from_slice(pack.kv_norm)?,
            sink: DevBuf::from_slice(pack.sink)?,
            rope,
            crope,
            compressor,
            indexer,
        })
    }

    /// Bytes of every device tensor this layer holds (weights, norms, both tables, compressor and indexer state).
    ///
    /// A shared table is counted once here: on a compressing layer `rope` and `crope` are the same buffer. Summing this over the
    /// layers would count a shared table once a layer, which is why [`super::dsv4_attn`]'s model-level total dedupes by pointer.
    pub fn bytes(&self) -> usize {
        let tables = if std::sync::Arc::ptr_eq(&self.rope, &self.crope) {
            self.rope.bytes() as usize
        } else {
            (self.rope.bytes() + self.crope.bytes()) as usize
        };
        self.weight_bytes() + tables
    }

    /// The bytes of this layer's weights, norms and compressor/indexer state — everything except its RoPE tables.
    pub fn weight_bytes(&self) -> usize {
        self.wq_a.weight_bytes()
            + self.wq_b.weight_bytes()
            + self.wkv.weight_bytes()
            + self.wo_a.w.bytes()
            + self.wo_a.s.bytes()
            + self.wo_b.weight_bytes()
            + self.q_norm.bytes()
            + self.kv_norm.bytes()
            + self.sink.bytes()
            + self.compressor.as_ref().map_or(0, |c| {
                c.wkv.weight_bytes()
                    + c.wgate.as_ref().map_or(0, |g| g.weight_bytes())
                    + c.norm.bytes()
                    + (c.kv_state.bytes() / MAX_SLOTS)
                    + (c.score_state.bytes() / MAX_SLOTS)
            })
            + self.indexer.as_ref().map_or(0, |i| {
                i.wq_b.weight_bytes()
                    + i.weights_proj.weight_bytes()
                    + i.wk.as_ref().map_or(0, |w| w.weight_bytes())
                    + i.k_norm.bytes()
            })
    }

    /// Reconcile the compressor carry-over state after a speculative window of `n` rows at `start_pos` commits `keep < n` rows.
    pub fn commit_window(&mut self, start_pos: usize, n: usize, keep: usize, st: &Stream) -> Result<(), CudaError> {
        self.commit_window_slot(0, start_pos, n, keep, st)
    }

    pub fn commit_window_slot(
        &mut self,
        seq_slot: usize,
        start_pos: usize,
        n: usize,
        keep: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if start_pos == 0 || n <= 1 || keep >= n || keep == 0 {
            return Ok(());
        }
        if self.cfg.kv_source_of(self.layer_id) == Some(self.layer_id) {
            if let Some(comp) = self.compressor.as_ref() {
                if comp.ratio == 2 && (start_pos + keep) % 2 == 1 {
                    let s0 = (seq_slot % MAX_SLOTS) * 64;
                    let st_slot0 = (seq_slot % MAX_SLOTS) * comp.ratio;
                    state_write_dev(
                        &comp.ckv_stage,
                        s0 + keep - 1,
                        &comp.gate_stage,
                        s0 + keep - 1,
                        comp,
                        st_slot0,
                        self.cfg.head_dim,
                        st,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn commit_window_planes(
        &mut self,
        planes: &Dsv4Planes,
        seq_slot: usize,
        start_pos: usize,
        n: usize,
        keep: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if start_pos == 0 || n <= 1 || keep >= n || keep == 0 {
            return Ok(());
        }
        let win = self.cfg.window;
        let d = self.cfg.head_dim;
        if planes.win_fmt == crate::dsv4_kvq4::KvFmt::F16 && n <= 64 {
            plane_write_dev(&planes.window_bak, 0, &planes.window, win, d, 0, st)?;
            ring_write_dev(&planes.window_bak, win, &planes.window, keep, d, start_pos % win, win, st)?;
        }
        self.commit_window_slot(seq_slot, start_pos, n, keep, st)
    }

    /// Snapshot `seq_slot`'s compressor carry-over state (`kv_state` and `score_state`) at token length `len`.
    pub fn comp_state_snapshot(&self, seq_slot: usize, len: usize) -> Result<Option<CompressorStateSnap>, CudaError> {
        let Some(comp) = self.compressor.as_ref() else {
            return Ok(None);
        };
        let d = self.cfg.head_dim;
        let st_slot0 = (seq_slot % MAX_SLOTS) * comp.ratio;
        let elems = comp.ratio * d;
        let byte_off = st_slot0 * d * 4;
        let mut kv_state = vec![0.0f32; elems];
        let mut score_state = vec![0.0f32; elems];
        comp.kv_state.download_range(byte_off, &mut kv_state)?;
        comp.score_state.download_range(byte_off, &mut score_state)?;
        let has_carry = comp.ratio > 1 && (len % comp.ratio) != 0;
        Ok(Some(CompressorStateSnap { kv_state, score_state, has_carry }))
    }

    /// The device ranges [`Self::comp_state_snapshot`] copies for `seq_slot` (and [`Self::comp_state_restore`] writes):
    /// `(kv_state, score_state, bytes of each)`; `None` without a compressor. What a prefix state digest reads.
    pub fn comp_state_range(&self, seq_slot: usize) -> Option<(*const std::ffi::c_void, *const std::ffi::c_void, usize)> {
        let comp = self.compressor.as_ref()?;
        let d = self.cfg.head_dim;
        let byte_off = (seq_slot % MAX_SLOTS) * comp.ratio * d * 4;
        let bytes = comp.ratio * d * 4;
        Some((comp.kv_state.ptr_at(byte_off) as *const _, comp.score_state.ptr_at(byte_off) as *const _, bytes))
    }

    /// Restore `seq_slot`'s compressor carry-over state (`kv_state` and `score_state`).
    pub fn comp_state_restore(
        &mut self,
        seq_slot: usize,
        snap: Option<&CompressorStateSnap>,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let (Some(comp), Some(s)) = (self.compressor.as_mut(), snap) else {
            return Ok(());
        };
        let d = self.cfg.head_dim;
        let st_slot0 = (seq_slot % MAX_SLOTS) * comp.ratio;
        let byte_off = st_slot0 * d * 4;
        let bytes = comp.ratio * d * 4;
        assert_eq!(s.kv_state.len() * 4, bytes);
        assert_eq!(s.score_state.len() * 4, bytes);
        unsafe {
            st.memcpy_h2d_async(comp.kv_state.ptr_at(byte_off), s.kv_state.as_ptr() as *const u8, bytes)?;
            st.memcpy_h2d_async(comp.score_state.ptr_at(byte_off), s.score_state.as_ptr() as *const u8, bytes)?;
        }
        Ok(())
    }

    /// One block call: the device twin of [`MlaDsv4Host::forward`].
    ///
    /// `x` is `f16 [n][dim]` and the result lands in `scratch.out` (`f16 [n][dim]`). `start_pos == 0` is prefill of
    /// `n` tokens (up to [`MAX_ROWS`]); otherwise `n == 1` is one decode step. The planes must hold every position
    /// below `start_pos + n` and `scratch` must be sized for `n` rows and the call's context.
    ///
    /// `shared` is the planes of this layer's KV source when the layer is not itself a source (`Dsv4Config::kv_source_of`):
    /// the compressed plane, its index keys, the level-1 candidate mask and the published selection live there, and the
    /// layers between two sources reuse them (model.py:749). A source passes `None` — it *is* the group's plane.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        x: &DevBuf,
        n: usize,
        start_pos: usize,
        planes: &Dsv4Planes,
        shared: Option<&Dsv4Planes>,
        scratch: &mut Dsv4Scratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.forward_slot(x, n, start_pos, 0, planes, shared, scratch, st)
    }

    /// Run only this KV-source layer's compressor and indexer-key projection over `n` rows at `start_pos`,
    /// populating `planes.compress` and `planes.index_k` without running query projections or attention.
    #[allow(clippy::too_many_arguments)]
    pub fn compress_only_slot(
        &mut self,
        x: &DevBuf,
        n: usize,
        start_pos: usize,
        seq_slot: usize,
        planes: &Dsv4Planes,
        scratch: &mut Dsv4Scratch,
        prefill: bool,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let cfg = self.cfg.clone();
        let (d, rd, f16) = (cfg.head_dim, cfg.rope_dim, dt(Dtype::F16));
        assert!((1..=MAX_ROWS).contains(&n) && n <= scratch.rows, "row count {n} out of range");
        assert!(x.bytes() >= n * cfg.dim * 2, "x must hold n rows of f16 dim");
        use crate::dsv4_kvq4::{self as kvq4, KvFmt, ROPE_DIMS};
        let q4_cache = planes.fmt == KvFmt::Q4;
        // `nope` is the q4 rotation's un-rotated width; fp16 never reads it, and a fixture head_dim below
        // ROPE_DIMS must not underflow a debug build (the real pack is 512 + 64).
        let nope = if q4_cache { d.checked_sub(ROPE_DIMS).ok_or(CudaError(-1))? } else { 0 };
        let ratio = cfg.ratio(self.layer_id) as usize;
        let no_dsa = std::env::var_os("REXL3_DSV41_NO_DSA").is_some();
        if ratio == 0 || no_dsa || cfg.kv_source_of(self.layer_id) != Some(self.layer_id) {
            return Ok(());
        }
        let cached_len = if start_pos == 0 { 0 } else { start_pos % ratio };
        let total = cached_len + n;
        let groups = total / ratio;
        let rem = total % ratio;
        let dst_group0 = start_pos / ratio;
        let pos0 = dst_group0 * ratio;
        let comp = self.compressor.as_ref().expect("a KV source has a compressor");
        comp.wkv.forward(x, &mut scratch.ckv, n, Dtype::F16, st)?;
        if let Some(g) = &comp.wgate {
            g.forward(x, &mut scratch.gate, n, Dtype::F16, st)?;
        }
        let st_slot0 = (seq_slot % MAX_SLOTS) * ratio;
        let stg_row0 = (seq_slot % MAX_SLOTS) * 64;
        if start_pos > 0 && n > 1 && ratio > 1 {
            let nw = n.min(64);
            plane_write_dev(&scratch.ckv, 0, &comp.ckv_stage, nw, d, stg_row0 * d, st)?;
            plane_write_dev(&scratch.gate, 0, &comp.gate_stage, nw, d, stg_row0 * d, st)?;
        }
        if cached_len == 0 {
            if groups > 0 {
                let rc = unsafe {
                    ffi::gb_dsv4_pool(
                        scratch.ckv.ptr(),
                        scratch.gate.ptr(),
                        scratch.latent.ptr(),
                        (groups * ratio) as i32,
                        d as i32,
                        ratio as i32,
                        groups as i32,
                        f16,
                        st.raw(),
                    )
                };
                CudaError::check(rc)?;
            }
            for t in 0..rem {
                let r = n - rem + t;
                state_write_dev(&scratch.ckv, r, &scratch.gate, r, comp, st_slot0 + t, d, st)?;
            }
        } else if total < ratio {
            for t in 0..n {
                state_write_dev(&scratch.ckv, t, &scratch.gate, t, comp, st_slot0 + cached_len + t, d, st)?;
            }
        } else {
            let need = ratio - cached_len;
            for t in 0..need {
                state_write_dev(&scratch.ckv, t, &scratch.gate, t, comp, st_slot0 + cached_len + t, d, st)?;
            }
            let rc = unsafe {
                ffi::gb_dsv4_pool_state(
                    comp.kv_state.ptr_at(st_slot0 * d * 4) as *const f32,
                    comp.score_state.ptr_at(st_slot0 * d * 4) as *const f32,
                    scratch.latent.ptr(),
                    d as i32,
                    ratio as i32,
                    f16,
                    st.raw(),
                )
            };
            CudaError::check(rc)?;
            let rest_groups = groups - 1;
            if rest_groups > 0 {
                let rc = unsafe {
                    ffi::gb_dsv4_pool(
                        scratch.ckv.ptr_at(need * d * 2),
                        scratch.gate.ptr_at(need * d * 2),
                        scratch.latent.ptr_at(d * 2),
                        (rest_groups * ratio) as i32,
                        d as i32,
                        ratio as i32,
                        rest_groups as i32,
                        f16,
                        st.raw(),
                    )
                };
                CudaError::check(rc)?;
            }
            for t in 0..rem {
                let r = n - rem + t;
                state_write_dev(&scratch.ckv, r, &scratch.gate, r, comp, st_slot0 + t, d, st)?;
            }
        }
        if groups > 0 {
            rmsnorm_dev(&scratch.latent, &comp.norm, groups, d, cfg.eps, st)?;

            if let Some(ix) = self.indexer.as_ref() {
                if let Some(wk) = &ix.wk {
                    wk.forward(&scratch.latent, &mut scratch.index_kt, groups, Dtype::F16, st)?;
                    rmsnorm_dev(&scratch.index_kt, &ix.k_norm, groups, ix.dim, cfg.eps, st)?;
                    rope_rows_dev(&scratch.index_kt, groups, ix.dim, rd, ix.dim, 1, self.crope.pair(), pos0, ratio, false, st)?;
                    if q4_cache {
                        kvq4::store_dev(&scratch.index_kt, 0, &planes.index_k, groups, dst_group0, ix.dim, 0, 0, st)?;
                    } else {
                        quant_dev(&scratch.index_kt, groups, ix.dim, 32, 4, true, st)?;
                        plane_write_dev(&scratch.index_kt, 0, &planes.index_k, groups, ix.dim, dst_group0 * ix.dim, st)?;
                    }
                }
            }
            rope_rows_dev(&scratch.latent, groups, d, rd, d, 1, self.crope.pair(), pos0, ratio, false, st)?;
            if q4_cache {
                kvq4::rot_dev(&scratch.latent, groups, d, nope, d, 1, st)?;
                kvq4::store_dev(&scratch.latent, 0, &planes.compress, groups, dst_group0, d, ROPE_DIMS, 0, st)?;
            } else {
                quant_dev(&scratch.latent, groups, d, 16, 4, false, st)?;
                plane_write_dev(&scratch.latent, 0, &planes.compress, groups, d, dst_group0 * d, st)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_slot(
        &mut self,
        x: &DevBuf,
        n: usize,
        start_pos: usize,
        seq_slot: usize,
        planes: &Dsv4Planes,
        shared: Option<&Dsv4Planes>,
        scratch: &mut Dsv4Scratch,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.forward_slot_ex(x, n, start_pos, seq_slot, planes, shared, false, false, scratch, false, st)
    }

    /// Batched multi-sequence MLA forward across `seqs` (`total_rows == sum(seq.rows)`).
    ///
    /// Dense projections (`wq_a + wkv`, `q_norm`, `kv_norm`, `comp.wkv`, `comp.wgate`, `ix.wk`, `ix.wq_b`,
    /// `ix.weights_proj`, `wq_b`, `wo_a`, `wo_b`) run once per layer over all `total_rows` rows, reading
    /// each attention weight matrix once instead of `seqs.len()` times. Per-sequence RoPE, window/compressed
    /// plane updates, and sparse sink attention loop over `seqs`.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn forward_batch(
        &mut self,
        x: &DevBuf,
        total_rows: usize,
        seqs: &[MlaSeq<'_>],
        scratch: &mut Dsv4Scratch,
        out: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if seqs.is_empty() || total_rows == 0 {
            return Ok(());
        }
        let cfg = self.cfg.clone();
        let (h, d, rd, f16) = (cfg.n_heads, cfg.head_dim, cfg.rope_dim, dt(Dtype::F16));
        assert!((1..=MAX_ROWS).contains(&total_rows) && total_rows <= scratch.rows, "row count {total_rows} out of range");
        assert!(x.bytes() >= total_rows * cfg.dim * 2, "x must hold total_rows of f16 dim");
        assert!(out.bytes() >= total_rows * cfg.dim * 2, "out must hold total_rows of f16 dim");
        use crate::dsv4_kvq4::{self as kvq4, KvFmt, ROPE_DIMS};
        let q4_cache = seqs[0].planes.fmt == KvFmt::Q4;
        let nope = if q4_cache { d.checked_sub(ROPE_DIMS).ok_or(CudaError(-1))? } else { 0 };
        let win = cfg.window;
        let ratio = cfg.ratio(self.layer_id) as usize;
        let no_dsa = std::env::var_os("REXL3_DSV41_NO_DSA").is_some();

        // 1. Batched Q-LoRA + window KV projection and RMSNorms across all total_rows
        Fp8Weights::forward_pair(&self.wq_a, &mut scratch.qr, &self.wkv, &mut scratch.kv, x, total_rows, Dtype::F16, st)
            .map_err(fp8_err)?;
        rmsnorm_dev(&scratch.qr, &self.q_norm, total_rows, cfg.q_lora, cfg.eps, st)?;
        rmsnorm_dev(&scratch.kv, &self.kv_norm, total_rows, d, cfg.eps, st)?;
        for seq in seqs {
            let kv_i = scratch.kv.subview(seq.row0 * d * 2, seq.rows * d * 2);
            rope_rows_dev(&kv_i, seq.rows, d, rd, d, 1, self.rope.pair(), seq.start_pos, 1, false, st)?;
        }
        quant_dev(&scratch.kv, total_rows, d, 32, 8, true, st)?;
        if q4_cache {
            kvq4::rot_dev(&scratch.kv, total_rows, d, nope, d, 1, st)?;
        }

        // 2. Batched Compressor + Indexer-K on KV-source layers
        let mut fill_cmp_topk = vec![0usize; seqs.len()];
        let mut n_cmp_vec = vec![0usize; seqs.len()];
        if ratio > 0 && !no_dsa {
            let is_src = cfg.kv_source_of(self.layer_id) == Some(self.layer_id);
            if is_src {
                let comp = self.compressor.as_ref().expect("a KV source has a compressor");
                comp.wkv.forward(x, &mut scratch.ckv, total_rows, Dtype::F16, st)?;
                if let Some(g) = &comp.wgate {
                    g.forward(x, &mut scratch.gate, total_rows, Dtype::F16, st)?;
                }
                let mut lat_off = 0usize;
                let mut grp_meta: Vec<(usize, usize, usize, usize, usize)> = Vec::with_capacity(seqs.len());
                if ratio == 1 {
                    plane_write_dev(&scratch.ckv, 0, &scratch.latent, total_rows, d, 0, st)?;
                    for (si, seq) in seqs.iter().enumerate() {
                        grp_meta.push((si, seq.row0, seq.rows, seq.start_pos, seq.start_pos));
                    }
                    lat_off = total_rows;
                } else {
                    for (si, seq) in seqs.iter().enumerate() {
                        let n = seq.rows;
                        let start_pos = seq.start_pos;
                        let seq_slot = seq.slot;
                        let cached_len = if start_pos == 0 { 0 } else { start_pos % ratio };
                        let total = cached_len + n;
                        let groups = total / ratio;
                        let rem = total % ratio;
                        let dst_group0 = start_pos / ratio;
                        let pos0 = dst_group0 * ratio;
                        let st_slot0 = (seq_slot % MAX_SLOTS) * ratio;
                        let stg_row0 = (seq_slot % MAX_SLOTS) * 64;
                        if start_pos > 0 && n > 1 {
                            let nw = n.min(64);
                            plane_write_dev(&scratch.ckv, seq.row0, &comp.ckv_stage, nw, d, stg_row0 * d, st)?;
                            plane_write_dev(&scratch.gate, seq.row0, &comp.gate_stage, nw, d, stg_row0 * d, st)?;
                        }
                        if cached_len == 0 {
                            if groups > 0 {
                                let rc = unsafe {
                                    ffi::gb_dsv4_pool(
                                        scratch.ckv.ptr_at(seq.row0 * d * 2),
                                        scratch.gate.ptr_at(seq.row0 * d * 2),
                                        scratch.latent.ptr_at(lat_off * d * 2),
                                        (groups * ratio) as i32,
                                        d as i32,
                                        ratio as i32,
                                        groups as i32,
                                        f16,
                                        st.raw(),
                                    )
                                };
                                CudaError::check(rc)?;
                            }
                            for t in 0..rem {
                                let r = seq.row0 + n - rem + t;
                                state_write_dev(&scratch.ckv, r, &scratch.gate, r, comp, st_slot0 + t, d, st)?;
                            }
                        } else if total < ratio {
                            for t in 0..n {
                                state_write_dev(
                                    &scratch.ckv,
                                    seq.row0 + t,
                                    &scratch.gate,
                                    seq.row0 + t,
                                    comp,
                                    st_slot0 + cached_len + t,
                                    d,
                                    st,
                                )?;
                            }
                        } else {
                            let need = ratio - cached_len;
                            for t in 0..need {
                                state_write_dev(
                                    &scratch.ckv,
                                    seq.row0 + t,
                                    &scratch.gate,
                                    seq.row0 + t,
                                    comp,
                                    st_slot0 + cached_len + t,
                                    d,
                                    st,
                                )?;
                            }
                            let rc = unsafe {
                                ffi::gb_dsv4_pool_state(
                                    comp.kv_state.ptr_at(st_slot0 * d * 4) as *const f32,
                                    comp.score_state.ptr_at(st_slot0 * d * 4) as *const f32,
                                    scratch.latent.ptr_at(lat_off * d * 2),
                                    d as i32,
                                    ratio as i32,
                                    f16,
                                    st.raw(),
                                )
                            };
                            CudaError::check(rc)?;
                            let rest_groups = groups - 1;
                            if rest_groups > 0 {
                                let rc = unsafe {
                                    ffi::gb_dsv4_pool(
                                        scratch.ckv.ptr_at((seq.row0 + need) * d * 2),
                                        scratch.gate.ptr_at((seq.row0 + need) * d * 2),
                                        scratch.latent.ptr_at((lat_off + 1) * d * 2),
                                        (rest_groups * ratio) as i32,
                                        d as i32,
                                        ratio as i32,
                                        rest_groups as i32,
                                        f16,
                                        st.raw(),
                                    )
                                };
                                CudaError::check(rc)?;
                            }
                            for t in 0..rem {
                                let r = seq.row0 + n - rem + t;
                                state_write_dev(&scratch.ckv, r, &scratch.gate, r, comp, st_slot0 + t, d, st)?;
                            }
                        }
                        if groups > 0 {
                            grp_meta.push((si, lat_off, groups, dst_group0, pos0));
                            lat_off += groups;
                        }
                    }
                }
                let total_groups = lat_off;
                if total_groups > 0 {
                    rmsnorm_dev(&scratch.latent, &comp.norm, total_groups, d, cfg.eps, st)?;
                    if let Some(ix) = self.indexer.as_ref() {
                        if let Some(wk) = &ix.wk {
                            wk.forward(&scratch.latent, &mut scratch.index_kt, total_groups, Dtype::F16, st)?;
                            rmsnorm_dev(&scratch.index_kt, &ix.k_norm, total_groups, ix.dim, cfg.eps, st)?;
                            for &(si, g_off, groups, _dst_group0, pos0) in &grp_meta {
                                let _ = si;
                                let ikt_i = scratch.index_kt.subview(g_off * ix.dim * 2, groups * ix.dim * 2);
                                rope_rows_dev(&ikt_i, groups, ix.dim, rd, ix.dim, 1, self.crope.pair(), pos0, ratio, false, st)?;
                            }
                            if !q4_cache {
                                quant_dev(&scratch.index_kt, total_groups, ix.dim, 32, 4, true, st)?;
                            }
                            for &(si, g_off, groups, dst_group0, _pos0) in &grp_meta {
                                let sel = seqs[si].shared.unwrap_or(seqs[si].planes);
                                if q4_cache {
                                    kvq4::store_dev(
                                        &scratch.index_kt,
                                        g_off,
                                        &sel.index_k,
                                        groups,
                                        dst_group0,
                                        ix.dim,
                                        0,
                                        0,
                                        st,
                                    )?;
                                } else {
                                    plane_write_dev(
                                        &scratch.index_kt,
                                        g_off,
                                        &sel.index_k,
                                        groups,
                                        ix.dim,
                                        dst_group0 * ix.dim,
                                        st,
                                    )?;
                                }
                            }
                        }
                    }
                    for &(_si, g_off, groups, _dst_group0, pos0) in &grp_meta {
                        let lat_i = scratch.latent.subview(g_off * d * 2, groups * d * 2);
                        rope_rows_dev(&lat_i, groups, d, rd, d, 1, self.crope.pair(), pos0, ratio, false, st)?;
                    }
                    if q4_cache {
                        kvq4::rot_dev(&scratch.latent, total_groups, d, nope, d, 1, st)?;
                        for &(si, g_off, groups, dst_group0, _pos0) in &grp_meta {
                            let sel = seqs[si].shared.unwrap_or(seqs[si].planes);
                            kvq4::store_dev(&scratch.latent, g_off, &sel.compress, groups, dst_group0, d, ROPE_DIMS, 0, st)?;
                        }
                    } else {
                        quant_dev(&scratch.latent, total_groups, d, 16, 4, false, st)?;
                        for &(si, g_off, groups, dst_group0, _pos0) in &grp_meta {
                            let sel = seqs[si].shared.unwrap_or(seqs[si].planes);
                            plane_write_dev(&scratch.latent, g_off, &sel.compress, groups, d, dst_group0 * d, st)?;
                        }
                    }
                }
            }

            // 3. Batched Indexer Q + Weights Projection (when any sequence exceeds short-circuit threshold)
            if let Some(ix) = self.indexer.as_ref() {
                let force_ix = std::env::var_os("REXL3_DSV41_FORCE_INDEXER").is_some();
                let mut any_active_ix = false;
                for seq in seqs {
                    let n_pos = (seq.start_pos + seq.rows) / ratio;
                    let n_blocks = n_pos.div_ceil(cfg.candidate_block_size.max(1));
                    if force_ix || n_pos > ix.topk || n_blocks > cfg.candidate_topk_blocks {
                        any_active_ix = true;
                        break;
                    }
                }
                if any_active_ix {
                    ix.wq_b.forward(&scratch.qr, &mut scratch.index_q, total_rows, Dtype::F16, st).map_err(fp8_err)?;
                    ix.weights_proj.forward(x, &mut scratch.index_w, total_rows, Dtype::F32, st)?;
                }
                let iq = ix.heads * ix.dim;
                let ix_scale = 1.0 / ((ix.dim * ix.heads) as f32).sqrt();
                for (si, seq) in seqs.iter().enumerate() {
                    let n = seq.rows;
                    let start_pos = seq.start_pos;
                    let n_pos = (start_pos + n) / ratio;
                    let n_blocks = n_pos.div_ceil(cfg.candidate_block_size.max(1));
                    if !force_ix && n_pos <= ix.topk && n_blocks <= cfg.candidate_topk_blocks {
                        fill_cmp_topk[si] = ix.topk;
                    } else {
                        let sel = seq.shared.unwrap_or(seq.planes);
                        let use_prev_win = start_pos > 0;
                        let win_slots = if !use_prev_win { n.min(win) } else { win };
                        let split = if !use_prev_win {
                            n
                        } else if n == 1 {
                            win
                        } else {
                            win + n
                        };
                        let topk_ptr: *mut i32 = sel.topk.ptr() as *mut i32;
                        let cand_ptr: *mut u8 = sel.candidates.ptr() as *mut u8;
                        let iq_seq = scratch.index_q.subview(seq.row0 * iq * 2, n * iq * 2);
                        rope_rows_dev(&iq_seq, n, iq, rd, ix.dim, ix.heads, self.crope.pair(), start_pos, 1, false, st)?;
                        quant_dev(&iq_seq, n, iq, 32, 4, true, st)?;
                        let cand_rows = sel.candidates.bytes() / sel.cand_stride.max(1);
                        // V57: candidate masking must depend only on the global compressed-block count, never on how
                        // many rows this call carries. `n <= PLANE_ROWS` made a 128-row chunk apply the level-1
                        // mask while 256/512 and the whole prompt did not, so the same KV produced different
                        // attention under different chunk grids (chunked != whole).
                        let need_cand = force_ix || n_blocks > cfg.candidate_topk_blocks;
                        let use_mask =
                            self.layer_id != cfg.candidate_source_layer && cfg.uses_candidates(self.layer_id) && need_cand;
                        // REXL3 field patch P3: batched sequences always use the plane candidate buffer.
                        if need_cand && n > cand_rows && (use_mask || self.layer_id == cfg.candidate_source_layer) {
                            eprintln!(
                                "rexl3 level=error event=dsv41_candidate_guard layer={} rows={n} start={start_pos} plane_rows={cand_rows} batched (field patch P3)",
                                self.layer_id
                            );
                            return Err(CudaError(9));
                        }
                        let score_cap = (scratch.score.bytes() / (n_pos.max(1) * 4)).max(1).min(n);
                        let mut r0 = 0usize;
                        while r0 < n {
                            let chunk = (n - r0).min(score_cap);
                            let q_slice = scratch.index_q.subview((seq.row0 + r0) * iq * 2, chunk * iq * 2);
                            let w_slice = scratch.index_w.subview((seq.row0 + r0) * ix.heads * 4, chunk * ix.heads * 4);
                            let mask = if use_mask {
                                unsafe { (cand_ptr as *const u8).add(r0 * sel.cand_stride) }
                            } else {
                                std::ptr::null()
                            };
                            if q4_cache {
                                kvq4::index_score_raw_dev(
                                    &q_slice,
                                    &sel.index_k,
                                    &w_slice,
                                    mask,
                                    sel.cand_stride,
                                    &scratch.score,
                                    chunk,
                                    n_pos,
                                    ix.heads,
                                    ix.dim,
                                    ix_scale,
                                    start_pos + r0,
                                    ratio,
                                    sel.fmt,
                                    st,
                                )?;
                            } else {
                                let rc = unsafe {
                                    ffi::gb_dsv4_index_score(
                                        q_slice.ptr(),
                                        sel.index_k.ptr(),
                                        w_slice.ptr() as *const f32,
                                        mask,
                                        sel.cand_stride as i32,
                                        scratch.score.ptr() as *mut f32,
                                        chunk as i32,
                                        n_pos as i32,
                                        ix.heads as i32,
                                        ix.dim as i32,
                                        ix_scale,
                                        (start_pos + r0) as i32,
                                        ratio as i32,
                                        f16,
                                        st.raw(),
                                    )
                                };
                                CudaError::check(rc)?;
                            }
                            if self.layer_id == cfg.candidate_source_layer && n_pos > 0 && need_cand {
                                let rc = unsafe {
                                    ffi::gb_dsv4_candidate_blocks(
                                        scratch.score.ptr() as *const f32,
                                        scratch.oq.ptr_at(Dsv4Scratch::bscore_off(&cfg, scratch.rows, scratch.max_seq))
                                            as *mut f32,
                                        cand_ptr.add(r0 * sel.cand_stride),
                                        sel.cand_stride as i32,
                                        chunk as i32,
                                        n_pos as i32,
                                        (start_pos + r0) as i32,
                                        ratio as i32,
                                        cfg.candidate_topk_blocks as i32,
                                        cfg.candidate_block_size as i32,
                                        st.raw(),
                                    )
                                };
                                CudaError::check(rc)?;
                            }
                            if n_pos > 0 {
                                let rc = unsafe {
                                    ffi::gb_dsv4_topk_select(
                                        scratch.score.ptr() as *const f32,
                                        topk_ptr.add(r0 * sel.sel_stride),
                                        chunk as i32,
                                        n_pos as i32,
                                        ix.topk as i32,
                                        win_slots as i32,
                                        split as i32,
                                        sel.sel_stride as i32,
                                        st.raw(),
                                    )
                                };
                                CudaError::check(rc)?;
                            }
                            r0 += chunk;
                        }
                    }
                }
            }
            for (si, seq) in seqs.iter().enumerate() {
                let compress_len = (seq.start_pos + seq.rows) / ratio;
                n_cmp_vec[si] = compress_len.min(cfg.index_topk);
            }
        }

        // 4. Batched wq_b across all total_rows (now that compressor/indexer are done with qo_blk)
        self.wq_b.forward(&scratch.qr, &mut scratch.q, total_rows, Dtype::F16, st).map_err(fp8_err)?;

        // 5. Per-sequence window ring update, Q RoPE, Sink Attention, O inverse-RoPE + quant
        let scale = 1.0 / (d as f32).sqrt();
        for (si, seq) in seqs.iter().enumerate() {
            let n = seq.rows;
            let start_pos = seq.start_pos;
            let planes = seq.planes;
            let sel = seq.shared.unwrap_or(planes);
            let topk_ptr: *mut i32 = sel.topk.ptr() as *mut i32;
            let kv_i = scratch.kv.subview(seq.row0 * d * 2, n * d * 2);
            let (src_off, win_rows) = if n <= win { (0, n) } else { (n - win, win) };
            let slot0 = (start_pos + src_off) % win;
            let use_prev_win = start_pos > 0;
            let win_slots = if !use_prev_win { n.min(win) } else { win };
            let split = if !use_prev_win {
                n
            } else if n == 1 {
                win
            } else {
                win + n
            };
            if use_prev_win && n > 1 {
                if planes.win_fmt == KvFmt::Q4 {
                    let rb = KvFmt::Q4.row_bytes(d, ROPE_DIMS);
                    let rc = unsafe {
                        ffi::gb_dsv4_plane_write(
                            planes.window.ptr(),
                            scratch.oq.ptr(),
                            win as i32,
                            (rb / 2) as i32,
                            0,
                            f16,
                            st.raw(),
                        )
                    };
                    CudaError::check(rc)?;
                    kvq4::store_dev(&kv_i, 0, &scratch.oq, n, win, d, ROPE_DIMS, win + n, st)?;
                } else {
                    plane_write_dev(&planes.window, 0, &scratch.oq, win, d, 0, st)?;
                    plane_write_dev(&kv_i, 0, &scratch.oq, n, d, win * d, st)?;
                    if n <= 64 && planes.window_bak.bytes() >= (win + n) * d * 2 {
                        plane_write_dev(&scratch.oq, 0, &planes.window_bak, win + n, d, 0, st)?;
                    }
                }
            }
            if planes.win_fmt == KvFmt::Q4 {
                kvq4::store_dev(&kv_i, src_off, &planes.window, win_rows, slot0, d, ROPE_DIMS, win, st)?;
            } else {
                ring_write_dev(&kv_i, src_off, &planes.window, win_rows, d, slot0, win, st)?;
            }

            if seq.shared.is_none() || fill_cmp_topk[si] > 0 {
                let rc = unsafe {
                    ffi::gb_dsv4_window_and_cmp_idxs(
                        topk_ptr,
                        n as i32,
                        split as i32,
                        sel.sel_stride as i32,
                        win as i32,
                        start_pos as i32,
                        fill_cmp_topk[si] as i32,
                        ratio as i32,
                        st.raw(),
                    )
                };
                CudaError::check(rc)?;
            }

            let q_i = scratch.q.subview(seq.row0 * h * d * 2, n * h * d * 2);
            let o_i = scratch.o.subview(seq.row0 * h * d * 2, n * h * d * 2);
            if d == 512 && rd == 64 && nope == 448 {
                let (cos, sin) = self.rope.pair();
                let rc = unsafe {
                    ffi::gb_dsv4_q_rope_rot_f16(
                        q_i.ptr(),
                        n as i32,
                        (h * d) as i32,
                        h as i32,
                        cos.ptr() as *const f32,
                        sin.ptr() as *const f32,
                        (rd / 2) as i32,
                        start_pos as i32,
                        1,
                        i32::from(q4_cache),
                        st.raw(),
                    )
                };
                CudaError::check(rc)?;
            } else {
                rope_rows_dev(&q_i, n, h * d, rd, d, h, self.rope.pair(), start_pos, 1, false, st)?;
                if q4_cache {
                    kvq4::rot_dev(&q_i, n, h * d, nope, d, h, st)?;
                }
            }

            // Row-invariance of the sliding-window format: when the window ring is Q4, the
            // `start_pos > 0` path reads Q4 keys (ring + freshly quantized chunk in `scratch.oq`),
            // but the `start_pos == 0` path used to read raw fp16 `scratch.kv`. The SAME key must be
            // attended at the SAME precision regardless of chunking or a prefix-cache cut, so here we
            // quantize this chunk's keys through the identical `kvq4::store_dev` used at start_pos > 0
            // into `scratch.oq` and attend that, making both paths bit-identical.
            if !use_prev_win && planes.win_fmt == KvFmt::Q4 {
                kvq4::store_dev(&kv_i, 0, &scratch.oq, n, 0, d, ROPE_DIMS, 0, st)?;
            }
            let win_plane: &DevBuf = if !use_prev_win {
                if planes.win_fmt == KvFmt::Q4 {
                    &scratch.oq
                } else {
                    &kv_i
                }
            } else if n == 1 {
                &planes.window
            } else {
                &scratch.oq
            };
            let idx_len = win_slots + n_cmp_vec[si];
            if d == 512 || q4_cache || planes.win_fmt != KvFmt::F16 {
                let wf = if !use_prev_win && planes.win_fmt != KvFmt::Q4 { KvFmt::F16 } else { planes.win_fmt };
                kvq4::attn_raw_dev(
                    &q_i,
                    win_plane,
                    &sel.compress,
                    split,
                    &self.sink,
                    topk_ptr as *const i32,
                    sel.sel_stride,
                    idx_len,
                    &o_i,
                    n,
                    h,
                    d,
                    wf,
                    sel.fmt,
                    scale,
                    st,
                )?;
            } else {
                let rc = unsafe {
                    ffi::gb_dsv4_attn(
                        q_i.ptr(),
                        win_plane.ptr(),
                        sel.compress.ptr(),
                        split as i32,
                        self.sink.ptr() as *const f32,
                        topk_ptr as *const i32,
                        sel.sel_stride as i32,
                        idx_len as i32,
                        o_i.ptr(),
                        n as i32,
                        h as i32,
                        d as i32,
                        scale,
                        f16,
                        st.raw(),
                    )
                };
                CudaError::check(rc)?;
            }

            if d == 512 && rd == 64 && nope == 448 {
                let (cos, sin) = self.rope.pair();
                let rc = unsafe {
                    ffi::gb_dsv4_o_rot_rope_quant_f16(
                        o_i.ptr(),
                        n as i32,
                        (h * d) as i32,
                        h as i32,
                        cos.ptr() as *const f32,
                        sin.ptr() as *const f32,
                        (rd / 2) as i32,
                        start_pos as i32,
                        1,
                        i32::from(q4_cache),
                        st.raw(),
                    )
                };
                CudaError::check(rc)?;
            } else {
                if q4_cache {
                    kvq4::rot_dev(&o_i, n, h * d, nope, d, h, st)?;
                }
                rope_rows_dev(&o_i, n, h * d, rd, d, h, self.rope.pair(), start_pos, 1, true, st)?;
                quant_dev(&o_i, n, h * d, 32, 8, true, st)?;
            }
        }

        // 6. Batched wo_a (grouped) and wo_b across all total_rows directly into `out`
        let rc = unsafe {
            ffi::gb_dsv4_woa_grouped(
                scratch.o.ptr(),
                self.wo_a.w.ptr() as *const u8,
                self.wo_a.s.ptr() as *const u8,
                scratch.olora.ptr(),
                total_rows as i32,
                cfg.o_groups as i32,
                cfg.o_lora as i32,
                cfg.o_in_per_group() as i32,
                f16,
                0,
                0,
                st.raw(),
            )
        };
        CudaError::check(rc)?;
        self.wo_b.forward(&scratch.olora, out, total_rows, Dtype::F16, st).map_err(fp8_err)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_slot_ex(
        &mut self,
        x: &DevBuf,
        n: usize,
        start_pos: usize,
        seq_slot: usize,
        planes: &Dsv4Planes,
        shared: Option<&Dsv4Planes>,
        skip_compress: bool,
        ced_replay_tail: bool,
        scratch: &mut Dsv4Scratch,
        prefill: bool,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.forward_slot_to_o(x, n, start_pos, seq_slot, planes, shared, skip_compress, ced_replay_tail, scratch, prefill, st)?;
        self.o_proj(&scratch.o, &mut scratch.olora, &mut scratch.out, n, prefill, st)
    }

    /// [`Self::forward_slot_ex`] up to and including the attention output's rotation-undo and quantisation. On return
    /// `scratch.o` holds THIS device's heads' attention output — `[n, q_heads * head_dim]` f16, compact at the buffer's start
    /// (the buffer is full-width; a single rank fills all of it, a rank of a tensor-parallel group fills its half and the
    /// combine at the o-projection boundary places the peer's beside it). The latent path, the compressor, the indexer and the
    /// selection run replicated and identically on every rank, so nothing here is a collective.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_slot_to_o(
        &mut self,
        x: &DevBuf,
        n: usize,
        start_pos: usize,
        seq_slot: usize,
        planes: &Dsv4Planes,
        shared: Option<&Dsv4Planes>,
        skip_compress: bool,
        ced_replay_tail: bool,
        scratch: &mut Dsv4Scratch,
        prefill: bool,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let cfg = self.cfg.clone();
        // the heads this device computes come from the weights' shape: every head on a single rank (`h == cfg.n_heads`, the
        // launches the single-node engine's), this rank's slice of `wq_b` on a group — the q and o widths below follow it
        let (h, d, rd, f16) = (self.q_heads, cfg.head_dim, cfg.rope_dim, dt(Dtype::F16));
        assert!((1..=MAX_ROWS).contains(&n) && n <= scratch.rows, "row count {n} out of range");
        assert!(x.bytes() >= n * cfg.dim * 2, "x must hold n rows of f16 dim");
        // The q4 cache rotates the 448 nope dims of every latent by an orthonormal 64-point Hadamard and keeps the
        // query in the same basis (see `dsv4_kvq4`): `q . k` is invariant and the value sum comes back rotated, so
        // one inverse rotation of the output undoes it. fp16, the default, leaves every one of these calls out.
        use crate::dsv4_kvq4::{self as kvq4, KvFmt, ROPE_DIMS};
        let q4_cache = planes.fmt == KvFmt::Q4;
        // `nope` is the q4 rotation's un-rotated width; fp16 never reads it, and a fixture head_dim below
        // ROPE_DIMS must not underflow a debug build (the real pack is 512 + 64).
        let nope = if q4_cache { d.checked_sub(ROPE_DIMS).ok_or(CudaError(-1))? } else { 0 };
        // The compressed plane, its index keys, the level-1 block mask and the published selection belong to the group's
        // source layer, not to this one (model.py:749 `shared_attn.compress_kv`): a reader is handed its source's planes
        // and only its own window ring. `shared` is `None` on a source, which is its own plane.
        let sel = shared.unwrap_or(planes);
        let win_stage_bytes = (cfg.window + n) * d * 2;
        let topk_stage_off = (win_stage_bytes + 4095) & !4095;
        let cand_stage_off = topk_stage_off + ((n * sel.sel_stride * 4 + 4095) & !4095);
        let topk_ptr: *mut i32 =
            if n <= PLANE_ROWS { sel.topk.ptr() as *mut i32 } else { scratch.oq.ptr_at(topk_stage_off) as *mut i32 };
        let cand_rows = sel.candidates.bytes() / sel.cand_stride.max(1);
        let cand_ptr: *mut u8 = if n <= cand_rows || cand_stage_off + n * sel.cand_stride > scratch.oq.bytes() {
            sel.candidates.ptr() as *mut u8
        } else {
            scratch.oq.ptr_at(cand_stage_off) as *mut u8
        };

        // --- Q + window KV: fuse wq_a (-> scratch.qr) and wkv (-> scratch.kv) in one launch
        if prefill {
            Fp8Weights::forward_pair_force(&self.wq_a, &mut scratch.qr, &self.wkv, &mut scratch.kv, x, n, Dtype::F16, st)
        } else {
            Fp8Weights::forward_pair(&self.wq_a, &mut scratch.qr, &self.wkv, &mut scratch.kv, x, n, Dtype::F16, st)
        }
        .map_err(fp8_err)?;
        rmsnorm_dev(&scratch.qr, &self.q_norm, n, cfg.q_lora, cfg.eps, st)?;
        // The KV prep chain (norm + rope + quant + q4 rotation) runs fused when the shapes match the
        // single-kernel template; otherwise the original four launches.
        let can_fuse_kv = nope % 64 == 0 && d % 32 == 0 && rd == 64 && rd % 2 == 0;
        if can_fuse_kv {
            kv_prepare_dev(&scratch.kv, &self.kv_norm, n, d, rd, d, self.rope.pair(), start_pos, 1, q4_cache, cfg.eps, st)?;
        } else {
            rmsnorm_dev(&scratch.kv, &self.kv_norm, n, d, cfg.eps, st)?;
            rope_rows_dev(&scratch.kv, n, d, rd, d, 1, self.rope.pair(), start_pos, 1, false, st)?;
            quant_dev(&scratch.kv, n, d, 32, 8, true, st)?;
            if q4_cache {
                kvq4::rot_dev(&scratch.kv, n, d, nope, d, 1, st)?;
            }
        }
        let win = cfg.window;
        let (src_off, rows) = if n <= win { (0, n) } else { (n - win, win) };
        let slot0 = (start_pos + src_off) % win;
        let use_prev_win = start_pos > 0 && !ced_replay_tail;
        let win_slots = if !use_prev_win { n.min(win) } else { win };
        let split = if !use_prev_win {
            n
        } else if n == 1 {
            win
        } else {
            win + n
        };
        if use_prev_win && n > 1 {
            // Assemble [planes.window (win rows) | scratch.kv (n rows)] in scratch.oq before updating the ring so
            // multi-row chunks/windows at start_pos > 0 never overwrite past window slots or leak future rows.
            if planes.win_fmt == KvFmt::Q4 {
                let rb = KvFmt::Q4.row_bytes(d, ROPE_DIMS);
                let rc = unsafe {
                    ffi::gb_dsv4_plane_write(planes.window.ptr(), scratch.oq.ptr(), win as i32, (rb / 2) as i32, 0, f16, st.raw())
                };
                CudaError::check(rc)?;
                kvq4::store_dev(&scratch.kv, 0, &scratch.oq, n, win, d, ROPE_DIMS, win + n, st)?;
            } else {
                plane_write_dev(&planes.window, 0, &scratch.oq, win, d, 0, st)?;
                plane_write_dev(&scratch.kv, 0, &scratch.oq, n, d, win * d, st)?;
                if n <= 64 {
                    plane_write_dev(&scratch.oq, 0, &planes.window_bak, win + n, d, 0, st)?;
                }
            }
        }
        if planes.win_fmt == KvFmt::Q4 {
            kvq4::store_dev(&scratch.kv, src_off, &planes.window, rows, slot0, d, ROPE_DIMS, win, st)?;
        } else {
            ring_write_dev(&scratch.kv, src_off, &planes.window, rows, d, slot0, win, st)?;
        }
        // --- compressed plane, and the selection over it
        let ratio = cfg.ratio(self.layer_id) as usize;
        let mut n_cmp = 0usize;
        let mut fill_cmp_topk = 0usize;
        // TEMP-BISECT: REXL3_DSV41_NO_DSA skips the compressed half entirely (window attention only)
        let no_dsa = std::env::var_os("REXL3_DSV41_NO_DSA").is_some();
        if ratio > 0 && !no_dsa {
            let compress_len = (start_pos + n) / ratio;
            let cached_len = if start_pos == 0 { 0 } else { start_pos % ratio };
            let total = cached_len + n;
            let groups = total / ratio;
            let rem = total % ratio;
            let dst_group0 = start_pos / ratio;
            let pos0 = dst_group0 * ratio;
            let _ = (cached_len, total, groups, rem, dst_group0, pos0, nope);
            let is_src = cfg.kv_source_of(self.layer_id) == Some(self.layer_id);
            if is_src && !skip_compress {
                self.compress_only_slot(x, n, start_pos, seq_slot, sel, scratch, prefill, st)?;
            }

            // --- indexer: q -> rope -> fp4 -> scores -> top-k
            if let Some(ix) = self.indexer.as_ref() {
                let n_pos = (start_pos + n) / ratio;
                let n_blocks = n_pos.div_ceil(cfg.candidate_block_size.max(1));
                let force_ix = std::env::var_os("REXL3_DSV41_FORCE_INDEXER").is_some();
                if !force_ix && n_pos <= ix.topk && n_blocks <= cfg.candidate_topk_blocks {
                    fill_cmp_topk = ix.topk;
                } else {
                    let iq = ix.heads * ix.dim;
                    let ix_scale = 1.0 / ((ix.dim * ix.heads) as f32).sqrt();
                    if prefill {
                        ix.wq_b.forward_force(&scratch.qr, &mut scratch.index_q, n, Dtype::F16, st)
                    } else {
                        ix.wq_b.forward(&scratch.qr, &mut scratch.index_q, n, Dtype::F16, st)
                    }
                    .map_err(fp8_err)?;
                    rope_rows_dev(&scratch.index_q, n, iq, rd, ix.dim, ix.heads, self.crope.pair(), start_pos, 1, false, st)?;
                    quant_dev(&scratch.index_q, n, iq, 32, 4, true, st)?;
                    ix.weights_proj.forward(x, &mut scratch.index_w, n, Dtype::F32, st)?;
                    let need_cand = force_ix || n_blocks > cfg.candidate_topk_blocks;
                    let use_mask = self.layer_id != cfg.candidate_source_layer && cfg.uses_candidates(self.layer_id) && need_cand;
                    // REXL3 field patch P3: the plane buffer holds cand_rows mask rows; refuse rather than run past it.
                    if need_cand
                        && n > cand_rows
                        && (use_mask || self.layer_id == cfg.candidate_source_layer)
                        && cand_ptr == sel.candidates.ptr() as *mut u8
                    {
                        eprintln!(
                            "rexl3 level=error event=dsv41_candidate_guard layer={} rows={n} start={start_pos} plane_rows={cand_rows} (field patch P3)",
                            self.layer_id
                        );
                        return Err(CudaError(9));
                    }
                    let score_cap = (scratch.score.bytes() / (n_pos.max(1) * 4)).max(1).min(n);
                    let mut r0 = 0usize;
                    while r0 < n {
                        let chunk = (n - r0).min(score_cap);
                        let q_slice = scratch.index_q.subview(r0 * iq * 2, chunk * iq * 2);
                        let w_slice = scratch.index_w.subview(r0 * ix.heads * 4, chunk * ix.heads * 4);
                        let mask = if use_mask {
                            unsafe { (cand_ptr as *const u8).add(r0 * sel.cand_stride) }
                        } else {
                            std::ptr::null()
                        };
                        if q4_cache {
                            kvq4::index_score_raw_dev(
                                &q_slice,
                                &sel.index_k,
                                &w_slice,
                                mask,
                                sel.cand_stride,
                                &scratch.score,
                                chunk,
                                n_pos,
                                ix.heads,
                                ix.dim,
                                ix_scale,
                                start_pos + r0,
                                ratio,
                                sel.fmt,
                                st,
                            )?;
                        } else {
                            let rc = unsafe {
                                ffi::gb_dsv4_index_score(
                                    q_slice.ptr(),
                                    sel.index_k.ptr(),
                                    w_slice.ptr() as *const f32,
                                    mask,
                                    sel.cand_stride as i32,
                                    scratch.score.ptr() as *mut f32,
                                    chunk as i32,
                                    n_pos as i32,
                                    ix.heads as i32,
                                    ix.dim as i32,
                                    ix_scale,
                                    (start_pos + r0) as i32,
                                    ratio as i32,
                                    f16,
                                    st.raw(),
                                )
                            };
                            CudaError::check(rc)?;
                        }
                        if self.layer_id == cfg.candidate_source_layer && n_pos > 0 && need_cand {
                            let bscore_cap =
                                scratch.oq.bytes().saturating_sub(Dsv4Scratch::bscore_off(&cfg, scratch.rows, scratch.max_seq))
                                    / (n_pos.div_ceil(cfg.candidate_block_size.max(1)).max(1) * 4);
                            assert!(
                                chunk <= bscore_cap,
                                "candidate block-score staging holds {bscore_cap} rows of {} blocks, need {chunk}",
                                n_pos.div_ceil(cfg.candidate_block_size.max(1))
                            );
                            let rc = unsafe {
                                ffi::gb_dsv4_candidate_blocks(
                                    scratch.score.ptr() as *const f32,
                                    scratch.oq.ptr_at(Dsv4Scratch::bscore_off(&cfg, scratch.rows, scratch.max_seq)) as *mut f32,
                                    cand_ptr.add(r0 * sel.cand_stride),
                                    sel.cand_stride as i32,
                                    chunk as i32,
                                    n_pos as i32,
                                    (start_pos + r0) as i32,
                                    ratio as i32,
                                    cfg.candidate_topk_blocks as i32,
                                    cfg.candidate_block_size as i32,
                                    st.raw(),
                                )
                            };
                            CudaError::check(rc)?;
                        }
                        if n_pos > 0 {
                            // REXL3_DSV41_DUMP_IX=<prefix>: write this selection round's scores and the
                            // inputs beside them (little-endian f32/i32/i16) -- the long-context selection
                            // diagnostic. Fires on the small-row calls (decode/verify windows), where the
                            // round's inputs are still intact; later stages of the same forward reuse them.
                            if n <= 8 {
                                if let Some(prefix) = std::env::var_os("REXL3_DSV41_DUMP_IX") {
                                    let tag = format!("{}-l{}-s{}", prefix.to_string_lossy(), self.layer_id, start_pos);
                                    let wr = |name: &str, bytes: Vec<u8>| {
                                        std::fs::write(format!("{tag}.{name}"), bytes).unwrap();
                                    };
                                    let score_v: Vec<f32> = scratch.score.to_vec().unwrap();
                                    wr(
                                        "score",
                                        score_v[..n_pos.min(score_v.len())].iter().flat_map(|v| v.to_le_bytes()).collect(),
                                    );
                                    // never a raw CPU read of device memory: the small-row path always
                                    // targets the persistent plane
                                    let sel_v: Vec<i32> = sel.topk.to_vec::<i32>().unwrap();
                                    wr(
                                        "sel",
                                        sel_v[(r0 * sel.sel_stride)..]
                                            [..sel.sel_stride.min(sel_v.len().saturating_sub(r0 * sel.sel_stride))]
                                            .iter()
                                            .flat_map(|v| v.to_le_bytes())
                                            .collect(),
                                    );
                                    let heads = self.indexer.as_ref().map(|i| i.heads).unwrap_or(0);
                                    let w_v: Vec<f32> = scratch.index_w.to_vec().unwrap();
                                    wr(
                                        "iw",
                                        w_v[(r0 * heads)..][..heads.min(w_v.len().saturating_sub(r0 * heads))]
                                            .iter()
                                            .flat_map(|v| v.to_le_bytes())
                                            .collect(),
                                    );
                                    let qdim = self.indexer.as_ref().map(|i| i.heads * i.dim).unwrap_or(0);
                                    let q_v: Vec<u16> = scratch.index_q.to_vec::<u16>().unwrap();
                                    wr("iq", q_v[(r0 * qdim)..].iter().flat_map(|v| v.to_le_bytes()).collect());
                                    eprintln!(
                                        "rexl3 event=dsv41_dump_ix layer={} start_pos={} n_pos={}",
                                        self.layer_id, start_pos, n_pos
                                    );
                                }
                            }
                            let rc = unsafe {
                                ffi::gb_dsv4_topk_select(
                                    scratch.score.ptr() as *const f32,
                                    topk_ptr.add(r0 * sel.sel_stride),
                                    chunk as i32,
                                    n_pos as i32,
                                    ix.topk as i32,
                                    win_slots as i32,
                                    split as i32,
                                    sel.sel_stride as i32,
                                    st.raw(),
                                )
                            };
                            CudaError::check(rc)?;
                        }
                        r0 += chunk;
                    }
                }
            }
            // the published compressed half is `index_topk` wide whatever the sequence length (model.py:579)
            n_cmp = compress_len.min(cfg.index_topk);
        }

        // --- the selection's window half: `get_window_topk_idxs` for every query of this layer, into the first `win_slots`
        // slots of the row (the compressed half, when there is one, sits after them — model.py:779). Reader layers reuse
        // the selection their source already published into `shared`.
        if shared.is_none() || fill_cmp_topk > 0 {
            let rc = unsafe {
                ffi::gb_dsv4_window_and_cmp_idxs(
                    topk_ptr,
                    n as i32,
                    split as i32,
                    sel.sel_stride as i32,
                    win as i32,
                    start_pos as i32,
                    fill_cmp_topk as i32,
                    ratio as i32,
                    st.raw(),
                )
            };
            CudaError::check(rc)?;
        }

        // --- Q: q_b -> rope (now that compressor and indexer are done with qo_block)
        if prefill {
            self.wq_b.forward_force(&scratch.qr, &mut scratch.q, n, Dtype::F16, st)
        } else {
            self.wq_b.forward(&scratch.qr, &mut scratch.q, n, Dtype::F16, st)
        }
        .map_err(fp8_err)?;
        if d == 512 && rd == 64 && nope == 448 {
            let (cos, sin) = self.rope.pair();
            let rc = unsafe {
                ffi::gb_dsv4_q_rope_rot_f16(
                    scratch.q.ptr(),
                    n as i32,
                    (h * d) as i32,
                    h as i32,
                    cos.ptr() as *const f32,
                    sin.ptr() as *const f32,
                    (rd / 2) as i32,
                    start_pos as i32,
                    1,
                    i32::from(q4_cache),
                    st.raw(),
                )
            };
            CudaError::check(rc)?;
        } else {
            rope_rows_dev(&scratch.q, n, h * d, rd, d, h, self.rope.pair(), start_pos, 1, false, st)?;
            if q4_cache {
                kvq4::rot_dev(&scratch.q, n, h * d, nope, d, h, st)?;
            }
        }

        // --- sink attention over [window | compressed]
        // Row-invariance of the sliding-window format (see the batched path above): at start_pos == 0
        // with a Q4 ring, quantize this chunk's keys into `scratch.oq` and attend that so the keys are
        // read at the exact same Q4 precision as every start_pos > 0 / prefix-cache-cut chunk.
        if !use_prev_win && planes.win_fmt == KvFmt::Q4 {
            kvq4::store_dev(&scratch.kv, 0, &scratch.oq, n, 0, d, ROPE_DIMS, 0, st)?;
        }
        let win_plane: &DevBuf = if !use_prev_win {
            if planes.win_fmt == KvFmt::Q4 {
                &scratch.oq
            } else {
                &scratch.kv
            }
        } else if n == 1 {
            &planes.window
        } else {
            &scratch.oq
        };
        let scale = 1.0 / (d as f32).sqrt();
        let idx_len = win_slots + n_cmp;
        if d == 512 || q4_cache || planes.win_fmt != KvFmt::F16 {
            // prefill gathers the un-compressed chunk out of `scratch.kv` (fp16); a decode step gathers the ring
            let wf = if !use_prev_win && planes.win_fmt != KvFmt::Q4 { KvFmt::F16 } else { planes.win_fmt };
            kvq4::attn_raw_dev(
                &scratch.q,
                win_plane,
                &sel.compress,
                split,
                &self.sink,
                topk_ptr as *const i32,
                sel.sel_stride,
                idx_len,
                &scratch.o,
                n,
                h,
                d,
                wf,
                sel.fmt,
                scale,
                st,
            )?;
        } else {
            let rc = unsafe {
                ffi::gb_dsv4_attn(
                    scratch.q.ptr(),
                    win_plane.ptr(),
                    sel.compress.ptr(),
                    split as i32,
                    self.sink.ptr() as *const f32,
                    topk_ptr as *const i32,
                    sel.sel_stride as i32,
                    idx_len as i32,
                    scratch.o.ptr(),
                    n as i32,
                    h as i32,
                    d as i32,
                    scale,
                    f16,
                    st.raw(),
                )
            };
            CudaError::check(rc)?;
        }

        // --- undo the value rotation and the query rotation, then the grouped output projection
        if d == 512 && rd == 64 && nope == 448 {
            let (cos, sin) = self.rope.pair();
            let rc = unsafe {
                ffi::gb_dsv4_o_rot_rope_quant_f16(
                    scratch.o.ptr(),
                    n as i32,
                    (h * d) as i32,
                    h as i32,
                    cos.ptr() as *const f32,
                    sin.ptr() as *const f32,
                    (rd / 2) as i32,
                    start_pos as i32,
                    1,
                    i32::from(q4_cache),
                    st.raw(),
                )
            };
            CudaError::check(rc)?;
        } else {
            if q4_cache {
                kvq4::rot_dev(&scratch.o, n, h * d, nope, d, h, st)?;
            }
            rope_rows_dev(&scratch.o, n, h * d, rd, d, h, self.rope.pair(), start_pos, 1, true, st)?;
            quant_dev(&scratch.o, n, h * d, 32, 8, true, st)?;
        }
        Ok(())
    }

    /// The grouped output projection over a FULL-width attention output: `o` is `[n, n_heads * head_dim]` f16 — every head's
    /// columns (on a rank of a group that is the combined buffer, not the compact half [`Self::forward_slot_to_o`] leaves) —
    /// and the block-diagonal `wo_a` and `wo_b` are replicated whole on every rank, so these two launches are the single-node
    /// projection's, and a group runs them identically over the combined `o` on every rank.
    pub fn o_proj(
        &self,
        o: &DevBuf,
        olora: &mut DevBuf,
        out: &mut DevBuf,
        n: usize,
        prefill: bool,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let cfg = &self.cfg;
        let f16 = dt(Dtype::F16);
        assert!(o.bytes() >= n * cfg.n_heads * cfg.head_dim * 2, "o must hold {n} rows of every head's columns");
        let rc = unsafe {
            ffi::gb_dsv4_woa_grouped(
                o.ptr(),
                self.wo_a.w.ptr() as *const u8,
                self.wo_a.s.ptr() as *const u8,
                olora.ptr(),
                n as i32,
                cfg.o_groups as i32,
                cfg.o_lora as i32,
                cfg.o_in_per_group() as i32,
                f16,
                0,
                i32::from(prefill),
                st.raw(),
            )
        };
        CudaError::check(rc)?;
        if prefill {
            self.wo_b.forward_force(olora, out, n, Dtype::F16, st)
        } else {
            self.wo_b.forward(olora, out, n, Dtype::F16, st)
        }
        .map_err(fp8_err)
    }

    /// Block-diagonal `wo_a` over THIS rank's `groups_local` output groups `[g0, g0 + groups_local)`: reads `o_compact`
    /// (`[n, groups_local * o_in_per_group]`, produced by [`Self::forward_slot_to_o`]) and writes `[n, groups_local * o_lora]`
    /// into `olora_compact`. Because `wo_a` is independent per group, running each rank's half before the TP exchange cuts
    /// both the `wo_a` weight traffic per rank in half and the exchanged byte payload by `o_in_per_group / o_lora` (16x),
    /// while remaining bit-identical to running all `o_groups` on the gathered `o`.
    pub fn woa_local(
        &self,
        o_compact: &DevBuf,
        olora_compact: &mut DevBuf,
        n: usize,
        g0: usize,
        groups_local: usize,
        prefill: bool,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let cfg = &self.cfg;
        let f16 = dt(Dtype::F16);
        let o_in = cfg.o_in_per_group();
        let o_lora = cfg.o_lora;
        let ktiles = o_in / 32;
        assert!(g0 + groups_local <= cfg.o_groups, "woa_local group range {g0}..{} exceeds {}", g0 + groups_local, cfg.o_groups);
        assert!(o_compact.bytes() >= n * groups_local * o_in * 2, "o_compact must hold {n} rows of {groups_local} groups");
        assert!(
            olora_compact.bytes() >= n * groups_local * o_lora * 2,
            "olora_compact must hold {n} rows of {groups_local} groups"
        );
        let w_off = g0 * o_lora * o_in;
        let s_off = g0 * (o_lora / 32) * ktiles;
        let rc = unsafe {
            ffi::gb_dsv4_woa_grouped(
                o_compact.ptr(),
                self.wo_a.w.ptr_at(w_off) as *const u8,
                self.wo_a.s.ptr_at(s_off) as *const u8,
                olora_compact.ptr(),
                n as i32,
                groups_local as i32,
                o_lora as i32,
                o_in as i32,
                f16,
                0,
                i32::from(prefill),
                st.raw(),
            )
        };
        CudaError::check(rc)
    }

    /// Test-only A/B entry point for the grouped output projection: runs the classic (`use_classic == true`)
    /// or the pipelined (false) kernel. Production calls always pass `false`; the GPU tests compare both
    /// bit-for-bit.
    pub fn woa_grouped_ab(
        o: &DevBuf,
        w: &DevBuf,
        s: &DevBuf,
        out: &mut DevBuf,
        rows: usize,
        groups: usize,
        o_lora: usize,
        o_in: usize,
        dtype: Dtype,
        use_classic: bool,
        st: &Stream,
    ) -> Result<(), CudaError> {
        unsafe {
            CudaError::check(ffi::gb_dsv4_woa_grouped(
                o.ptr(),
                w.ptr() as *const u8,
                s.ptr() as *const u8,
                out.ptr(),
                rows as i32,
                groups as i32,
                o_lora as i32,
                o_in as i32,
                dtype as i32,
                i32::from(use_classic),
                0,
                st.raw(),
            ))
        }
    }

    /// The outer output projection `wo_b` over a full-width `olora_full` (`[n, o_groups * o_lora]`).
    pub fn wob_proj(&self, olora_full: &DevBuf, out: &mut DevBuf, n: usize, prefill: bool, st: &Stream) -> Result<(), CudaError> {
        if prefill {
            self.wo_b.forward_force(olora_full, out, n, Dtype::F16, st)
        } else {
            self.wo_b.forward(olora_full, out, n, Dtype::F16, st)
        }
        .map_err(fp8_err)
    }
}

// --- the kernels the device forward needs, as small helpers over the launchers

/// Rotate `rows` rows of `x` (`[rows][stride]`, `f16`) at positions `pos0 + r`, in place.
#[allow(clippy::too_many_arguments)]
pub fn rope_rows_dev(
    x: &DevBuf,
    rows: usize,
    stride: usize,
    dim: usize,
    block: usize,
    sub: usize,
    table: (&DevBuf, &DevBuf),
    pos0: usize,
    pos_step: usize,
    inverse: bool,
    st: &Stream,
) -> Result<(), CudaError> {
    let rc = unsafe {
        ffi::gb_dsv4_rope(
            x.ptr(),
            dt(Dtype::F16),
            rows as i32,
            stride as i32,
            dim as i32,
            block as i32,
            sub as i32,
            table.0.ptr() as *const f32,
            table.1.ptr() as *const f32,
            (dim / 2) as i32,
            pos0 as i32,
            pos_step as i32,
            inverse as i32,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// [`Self::rope_rows_dev`] with `pos0` coming from a device round-params block (`Dsv4MtpRoundParams`): the row
/// `r` rotates at `params.start_pos + pos_off + r * pos_step`. Used by the DSV4.1 draft-round graphs so a captured
/// launch sequence replays as `start_pos` advances.
#[allow(clippy::too_many_arguments)]
pub fn rope_rows_dev_params(
    x: &DevBuf,
    rows: usize,
    stride: usize,
    dim: usize,
    block: usize,
    sub: usize,
    table: (&DevBuf, &DevBuf),
    pos_off: usize,
    pos_step: usize,
    inverse: bool,
    params: &DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    let rc = unsafe {
        ffi::gb_dsv4_rope_params(
            x.ptr(),
            dt(Dtype::F16),
            rows as i32,
            stride as i32,
            dim as i32,
            block as i32,
            sub as i32,
            table.0.ptr() as *const f32,
            table.1.ptr() as *const f32,
            (dim / 2) as i32,
            pos_off as i32,
            pos_step as i32,
            inverse as i32,
            params.ptr(),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// Normalise `x` (`f32 [rows, cols]`) by `w` (`f32 [cols]`) into `y` (`f16 [rows, cols]`).
pub fn rmsnorm_f32_to_f16_dev(
    x: &DevBuf,
    w: &DevBuf,
    y: &mut DevBuf,
    rows: usize,
    cols: usize,
    eps: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    assert!(x.bytes() >= rows * cols * 4 && w.bytes() >= cols * 4 && y.bytes() >= rows * cols * 2);
    let rc = unsafe {
        ffi::gb_dsv4_rmsnorm_f32_to_f16(
            x.ptr() as *const f32,
            w.ptr() as *const f32,
            y.ptr(),
            rows as i32,
            cols as i32,
            eps,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// `x[r] = rms_norm(x[r]) * w` in place over `rows` rows of `width`.
pub fn rmsnorm_dev(x: &DevBuf, w: &DevBuf, rows: usize, width: usize, eps: f32, st: &Stream) -> Result<(), CudaError> {
    let rc =
        unsafe { ffi::gb_dsv4_rmsnorm(x.ptr(), w.ptr() as *const f32, dt(Dtype::F16), rows as i32, width as i32, eps, st.raw()) };
    CudaError::check(rc)
}

/// Quantise-dequantise `rows` rows of `width` in place (`block`-wide groups, `bits` = 8 or 4).
#[allow(clippy::too_many_arguments)]
pub fn quant_dev(
    x: &DevBuf,
    rows: usize,
    width: usize,
    block: usize,
    bits: u32,
    pow2_scale: bool,
    st: &Stream,
) -> Result<(), CudaError> {
    let rc = unsafe {
        ffi::gb_dsv4_quant(
            x.ptr(),
            dt(Dtype::F16),
            rows as i32,
            width as i32,
            block as i32,
            bits as i32,
            pow2_scale as i32,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// The fused KV prep: `rmsnorm(x)`, `rope`, 8-bit pow2 quant over 32-wide groups and the q4 hadamard rotation in a
/// single launch. Bit-identical to running [`rmsnorm_dev`] + [`rope_rows_dev`] + [`quant_dev`] + [`rot_dev`] in
/// sequence over the same buffer (each phase replicates the standalone kernel's arithmetic and reduction order).
#[allow(clippy::too_many_arguments)]
pub fn kv_prepare_dev(
    x: &DevBuf,
    w: &DevBuf,
    rows: usize,
    width: usize,
    rope_dim: usize,
    rope_block: usize,
    table: (&DevBuf, &DevBuf),
    pos0: usize,
    pos_step: usize,
    do_rot: bool,
    eps: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    let rc = unsafe {
        ffi::gb_dsv4_kv_prepare(
            x.ptr(),
            w.ptr() as *const f32,
            rows as i32,
            width as i32,
            rope_dim as i32,
            rope_block as i32,
            table.0.ptr() as *const f32,
            table.1.ptr() as *const f32,
            (rope_dim / 2) as i32,
            pos0 as i32,
            pos_step as i32,
            i32::from(do_rot),
            eps,
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// Copy `rows` rows of `width` from `src[src_row0 + r]` into a plane at `dst_row0 + r`.
#[allow(clippy::too_many_arguments)]
pub fn plane_write_dev(
    src: &DevBuf,
    src_row0: usize,
    plane: &DevBuf,
    rows: usize,
    width: usize,
    dst_off: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    assert!(src.bytes() >= (src_row0 + rows) * width * 2, "src too small for the copy");
    assert!(plane.bytes() >= dst_off * 2 + rows * width * 2, "plane too small for the copy");
    let src_ptr = src.ptr_at(src_row0 * width * 2);
    let rc = unsafe {
        ffi::gb_dsv4_plane_write(src_ptr, plane.ptr_at(dst_off * 2), rows as i32, width as i32, 0, dt(Dtype::F16), st.raw())
    };
    CudaError::check(rc)
}

/// Copy `rows` rows of `width` from `src[src_row0 + r]` into a ring of `n_slots` at `(slot0 + r) % n_slots`.
#[allow(clippy::too_many_arguments)]
pub fn ring_write_dev(
    src: &DevBuf,
    src_row0: usize,
    ring: &DevBuf,
    rows: usize,
    width: usize,
    slot0: usize,
    n_slots: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    assert!(src.bytes() >= (src_row0 + rows) * width * 2, "src too small for the copy");
    assert!(ring.bytes() >= n_slots * width * 2, "ring too small");
    let src_ptr = src.ptr_at(src_row0 * width * 2);
    let rc = unsafe {
        ffi::gb_dsv4_ring_write(
            src_ptr,
            ring.ptr(),
            rows as i32,
            width as i32,
            slot0 as i32,
            n_slots as i32,
            dt(Dtype::F16),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

/// [`Self::ring_write_dev`] with the slot list coming from a device round-params block (`Dsv4MtpRoundParams`):
/// the slot starts at `start_pos % window` and the ring is `window` slots. Used by the DSV4.1 draft-round graphs
/// so a captured launch sequence replays as `start_pos` advances.
#[allow(clippy::too_many_arguments)]
pub fn ring_write_dev_params(
    src: &DevBuf,
    src_row0: usize,
    ring: &DevBuf,
    rows: usize,
    width: usize,
    params: &DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    assert!(src.bytes() >= (src_row0 + rows) * width * 2, "src too small for the copy");
    let src_ptr = src.ptr_at(src_row0 * width * 2);
    let rc = unsafe {
        ffi::gb_dsv4_ring_write_params(src_ptr, ring.ptr(), rows as i32, width as i32, params.ptr(), dt(Dtype::F16), st.raw())
    };
    CudaError::check(rc)
}

/// One token (`src` row `src_row`) into slot `slot` of the compressor's carry-over state.
#[allow(clippy::too_many_arguments)]
pub fn state_write_dev(
    kv: &DevBuf,
    src_row: usize,
    score: &DevBuf,
    score_row: usize,
    comp: &DevCompressor,
    slot: usize,
    head_dim: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    assert_eq!(comp.kv_state.bytes(), comp.score_state.bytes(), "the two halves of the state must match");
    assert!(comp.kv_state.bytes() >= (slot + 1) * head_dim * 4, "state slot {slot} out of range");
    let rc = unsafe {
        ffi::gb_dsv4_state_write(
            kv.ptr_at(src_row * head_dim * 2),
            score.ptr_at(score_row * head_dim * 2),
            comp.kv_state.ptr() as *mut f32,
            comp.score_state.ptr() as *mut f32,
            head_dim as i32,
            slot as i32,
            dt(Dtype::F16),
            st.raw(),
        )
    };
    CudaError::check(rc)
}

// --------------------------------------------------------------------------------- the tree round's surface moves (T2.5a)
// A draft-tree round verifies one CHAIN window per root-to-leaf path; the moves below are the byte
// surgery its commit runs over the persistent surfaces those windows mutate. They are plain byte movers,
// stated once here so the model's driver and the device-gated tests exercise the same code.

/// Copy the window ring's slots of positions `p0..p0+rows` into `dst`, in POSITION order (dst row `j` is
/// position `p0 + j`'s slot) — the round's base snapshot, taken before the first path window runs. The ring
/// is fp16 (`win_fmt == F16`; a q4-window cache is outside this surgery, as it is outside
/// [`commit_window_planes`]).
pub fn window_ring_save(planes: &Dsv4Planes, p0: usize, rows: usize, dst: &mut DevBuf, st: &Stream) -> Result<(), CudaError> {
    assert!(planes.win_fmt == crate::dsv4_kvq4::KvFmt::F16, "the tree surgery runs on the fp16 window ring");
    let win = planes.window_slots;
    let rb = planes.window.bytes() / win.max(1);
    let region = rows.min(win);
    let (s0, first) = (p0 % win, region.min(win - p0 % win));
    dst.copy_from_at(0, &planes.window, s0 * rb, first * rb, st)?;
    if region > first {
        dst.copy_from_at(first * rb, &planes.window, 0, (region - first) * rb, st)?;
    }
    Ok(())
}

/// [`window_ring_save`]'s inverse: the base bytes back into the ring's slots of `p0..p0+rows`.
pub fn window_ring_restore(planes: &mut Dsv4Planes, p0: usize, rows: usize, src: &DevBuf, st: &Stream) -> Result<(), CudaError> {
    assert!(planes.win_fmt == crate::dsv4_kvq4::KvFmt::F16, "the tree surgery runs on the fp16 window ring");
    let win = planes.window_slots;
    let rb = planes.window.bytes() / win.max(1);
    let region = rows.min(win);
    let (s0, first) = (p0 % win, region.min(win - p0 % win));
    planes.window.copy_from_at(s0 * rb, src, 0, first * rb, st)?;
    if region > first {
        planes.window.copy_from_at(0, src, first * rb, (region - first) * rb, st)?;
    }
    Ok(())
}

/// Write `rows` linear KV rows into the window ring at positions `p0..` (the commit's rewrite of the
/// accepted path's own rows over the restored base — the same move `commit_window_planes` makes from
/// `window_bak`'s tail for a chain's kept prefix).
pub fn window_ring_write(
    planes: &mut Dsv4Planes,
    d: usize,
    p0: usize,
    rows: usize,
    src: &DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    assert!(planes.win_fmt == crate::dsv4_kvq4::KvFmt::F16, "the tree surgery runs on the fp16 window ring");
    ring_write_dev(src, 0, &planes.window, rows, d, p0 % planes.window_slots, planes.window_slots, st)
}

/// Save `seq_slot`'s compressor carry (both halves, `ratio` rows of f32 each) out of the state buffers.
#[allow(clippy::too_many_arguments)]
pub fn compressor_carry_save(
    kv_state: &DevBuf,
    score_state: &DevBuf,
    ratio: usize,
    d: usize,
    seq_slot: usize,
    kv_dst: &mut DevBuf,
    score_dst: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    let (off, bytes) = ((seq_slot % MAX_SLOTS) * ratio * d * 4, ratio * d * 4);
    kv_dst.copy_from_at(0, kv_state, off, bytes, st)?;
    score_dst.copy_from_at(0, score_state, off, bytes, st)
}

/// [`compressor_carry_save`]'s inverse — the base carry back into the state buffers (the per-path
/// isolation's restore, and an abandoned round's).
#[allow(clippy::too_many_arguments)]
pub fn compressor_carry_restore(
    kv_state: &mut DevBuf,
    score_state: &mut DevBuf,
    ratio: usize,
    d: usize,
    seq_slot: usize,
    kv_src: &DevBuf,
    score_src: &DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    let (off, bytes) = ((seq_slot % MAX_SLOTS) * ratio * d * 4, ratio * d * 4);
    kv_state.copy_from_at(off, kv_src, 0, bytes, st)?;
    score_state.copy_from_at(off, score_src, 0, bytes, st)
}

#[cfg(test)]
mod rope_share_tests {
    use super::*;

    /// **Two layers, one table.** Every layer of this pack rotates with one of exactly two specs — the compressed plane's (160000,
    /// YaRN on) and the plain one (10000, YaRN off) — so the shared set is two tables, not one a layer. Building the pair a layer
    /// cost 0.67 GiB at 32K for 43 layers, which admission did not price.
    #[test]
    fn two_layers_with_the_same_rope_config_share_one_table() {
        let cfg = Dsv4Config::v41();
        let seq = 32_768;
        // layer 0 does not compress, layer 1 does (the pack's ratios start 0, 0, 1, ...); both are in the set, once each
        let specs = SharedRope::specs(&cfg, seq);
        assert_eq!(specs.len(), 2, "{specs:?}");
        assert_eq!(specs[0], RopeSpec::for_layer(&cfg, 0, seq));
        assert_eq!(specs[1], RopeSpec::compress(&cfg, seq));
        assert_ne!(specs[0], specs[1], "the two rotations differ in base and in YaRN");
        assert_eq!(specs[0].base, cfg.rope_theta);
        assert_eq!(specs[0].original_seq_len, 0, "the plain table disables YaRN");
        assert_eq!(specs[1].base, cfg.compress_rope_theta);
        assert!(specs[1].original_seq_len > 0);
        // the two layers that share a spec are the same spec object
        assert_eq!(RopeSpec::for_layer(&cfg, 0, seq), RopeSpec::for_layer(&cfg, 1, seq));
        // and the price is two tables of `seq x rope_dim / 2` f32, cos and sin, whatever the depth
        assert_eq!(shared_rope_bytes(&cfg, seq), 2 * 2 * seq as u64 * (cfg.rope_dim / 2) as u64 * 4);
        assert_eq!(shared_rope_bytes(&cfg, seq), 16_777_216);
        // the unshared figure is what the boot used to spend: a pair a layer, plus the compressed pair on each layer that does not
        // compress (its `crope` is still built, and never read)
        let table = 2 * seq as u64 * (cfg.rope_dim / 2) as u64 * 4;
        let zero = (0..cfg.n_layers).filter(|&l| cfg.ratio(l) == 0).count() as u64;
        let unshared = SharedRope::unshared_bytes(&cfg, seq);
        assert_eq!(unshared, (cfg.n_layers as u64 + zero) * table, "a pair a layer, plus the unused compressed pair on five");
        assert_eq!(zero, 5, "five trunk layers cache no compressed entry and rotate at the plain base");
        assert_eq!(unshared / shared_rope_bytes(&cfg, seq), 24, "48 pairs against 2");
        // a table's own bytes are the formula's
        assert_eq!(RopeSpec::compress(&cfg, seq).table_bytes() * 2, shared_rope_bytes(&cfg, seq) / 2);
    }

    /// `Rope::for_layer` and `RopeSpec::build` are the same table: the sharing cannot change what a layer rotates with.
    #[test]
    fn the_shared_spec_builds_the_table_the_layer_used_to_build() {
        let cfg = Dsv4Config::v41();
        for layer in [0usize, 1, 2, 41, 42] {
            let a = Rope::for_layer(&cfg, layer, 1_024);
            let b = RopeSpec::for_layer(&cfg, layer, 1_024).build();
            assert_eq!((a.dim, a.seq), (b.dim, b.seq));
            assert_eq!(a.cos, b.cos, "layer {layer}");
            assert_eq!(a.sin, b.sin, "layer {layer}");
        }
    }

    /// The planes' price is the five buffers `new` allocates, buffer for buffer, **per layer**: a ratio-1 layer's compressed
    /// plane, index keys and mask are twice a ratio-2 layer's, and sizing from `ratio_hint` (the table's maximum) would
    /// under-allocate every ratio-1 layer by 16.8 MB at 32K.
    #[test]
    fn the_planes_price_is_the_sum_of_what_new_allocates() {
        let cfg = Dsv4Config::v41();
        assert_eq!(cfg.ratio_hint(), 2);
        let mut total = 0usize;
        for layer in 0..cfg.n_layers {
            let nc = compress_slots(&cfg, layer, 32_768);
            let expected = cfg.window * cfg.head_dim * 2
                + nc * cfg.head_dim * 2
                + nc * cfg.index_dim * 2
                + PLANE_ROWS * nc
                + PLANE_ROWS * (cfg.window + cfg.index_topk) * 4;
            assert_eq!(Dsv4Planes::bytes(&cfg, 32_768, layer, crate::dsv4_kvq4::KvFmt::F16), expected, "layer {layer}");
            total += expected;
        }
        // 23.5 MB a layer at ratio 2, 34 MB at ratio 1: 1.48 GiB over the table's 43 layers (40 trunk + 3 NextN)
        assert!((1_500_000_000..1_650_000_000).contains(&total), "{total}");
    }
}

#[cfg(test)]
mod scratch_layout_tests {
    use super::*;

    /// Every sub-buffer offset of [`Dsv4Scratch`] must be 16-byte aligned — the f32 `bscore`
    /// staging used to start at an arbitrary byte (CUDA error 716 once the candidate-block path
    /// engages) — and no two regions may overlap, across a sweep of context windows, row counts,
    /// ratio tables and dims.
    #[test]
    fn every_scratch_offset_is_aligned_and_no_region_overlaps() {
        let max_seqs = [4_096usize, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576];
        let mut cfgs: Vec<Dsv4Config> = vec![Dsv4Config::v41()];
        // a ratio-2-only table (smaller compress_slots_max) and different o/head dims
        let mut cfg2 = Dsv4Config::v41();
        cfg2.compress_ratios = vec![2u8; cfg2.n_layers];
        cfgs.push(cfg2);
        let mut cfg3 = Dsv4Config::v41();
        cfg3.o_lora = 512;
        cfg3.o_groups = 4;
        cfg3.head_dim = 256;
        cfg3.index_dim = 64;
        cfgs.push(cfg3);

        for cfg in &cfgs {
            for &max_seq in &max_seqs {
                for rows in 1..=8 {
                    let (_, _, qo_b, oq_olora_b, offs, score_b, cand_off, _) = Dsv4Scratch::layout(cfg, rows, max_seq);
                    let hd = |n: usize| n * 2;
                    let nc = compress_slots_max(cfg, max_seq);
                    let score_rows = if rows * nc <= 2048 * 8193 { rows } else { (2048 * 8193 / nc.max(1)).max(1).min(rows) };
                    let q_bytes = hd(rows * cfg.n_heads * cfg.head_dim);
                    let h_bytes = hd(rows * cfg.head_dim);
                    let olora_bytes = ((rows * cfg.o_groups * cfg.o_lora * 2) + 15) & !15;
                    let cand_bytes = ((score_rows * nc) + 15) & !15;
                    let bscore_bytes = Dsv4Scratch::bscore_bytes(cfg, rows, max_seq);
                    let bscore_off = Dsv4Scratch::bscore_off(cfg, rows, max_seq);
                    let win_stage = ((cfg.window + rows) * cfg.head_dim * 2 + 4095) & !4095;

                    // every offset is 16-aligned
                    assert_eq!(cand_off % 16, 0);
                    assert_eq!(bscore_off % 16, 0, "bscore_off at max_seq={max_seq} rows={rows}");
                    for (i, &o) in offs.iter().enumerate() {
                        assert_eq!(o % 16, 0, "aux offset {i} at max_seq={max_seq} rows={rows}");
                    }
                    // the bscore staging sits at the end of the oq arena (the arena pads it to 4K)
                    assert!(bscore_off + bscore_bytes <= oq_olora_b, "bscore past oq arena");

                    // spans inside the shared qo block (q reuses the aux staging region by design:
                    // the compressor/index stages finish before the output is written, so only the
                    // aux spans themselves must be mutually disjoint)
                    let mut spans: Vec<(usize, usize)> = vec![];
                    spans.push((offs[0], h_bytes)); // ckv
                    spans.push((offs[1], h_bytes)); // gate
                    spans.push((offs[2], h_bytes)); // latent
                    spans.push((offs[3], hd(rows * cfg.index_dim))); // index_kt
                    spans.push((offs[4], hd(rows * cfg.index_heads * cfg.index_dim))); // index_q
                    spans.push((offs[5], rows * cfg.index_heads * 4)); // index_w
                    spans.push((offs[6], score_b)); // score
                    assert!(q_bytes <= qo_b, "q {q_bytes} past qo {qo_b}");
                    for &(start, len) in &spans {
                        assert!(start + len <= qo_b, "qo span {start}+{len} past {qo_b}");
                    }
                    for i in 0..spans.len() {
                        for j in i + 1..spans.len() {
                            let (a0, a1) = (spans[i].0, spans[i].0 + spans[i].1);
                            let (b0, b1) = (spans[j].0, spans[j].0 + spans[j].1);
                            assert!(
                                a1 <= b0 || b1 <= a0,
                                "qo spans overlap: ({a0},{a1}) vs ({b0},{b1}) at max_seq={max_seq} rows={rows}"
                            );
                        }
                    }

                    // spans inside the oq arena
                    let oq = [
                        (0, win_stage), // window staging
                        (win_stage, cand_off),
                        (cand_off, cand_off + olora_bytes),      // olora
                        (bscore_off, bscore_off + bscore_bytes), // bscore
                    ];
                    for &(start, end) in &oq {
                        assert!(end <= oq_olora_b, "oq span {start}..{end} past {oq_olora_b}");
                    }
                    for i in 0..oq.len() {
                        for j in i + 1..oq.len() {
                            assert!(
                                oq[i].1 <= oq[j].0 || oq[j].1 <= oq[i].0,
                                "oq spans overlap at max_seq={max_seq} rows={rows}: {oq:?}"
                            );
                        }
                    }
                    // olora and cand staging occupy the same position choice; the max of the padded sizes
                    // is exactly the delta from cand_off to bscore_off
                    assert_eq!(bscore_off - cand_off, olora_bytes.max(cand_bytes));
                }
            }
        }
    }

    /// The exact shapes of the 716 reproductions are now 16-aligned: the longctx prefill chunk
    /// (rows 449, 49152 window) and the FORCE_INDEXER 35-token run (rows 5).
    #[test]
    fn the_report_repro_shapes_are_aligned() {
        let cfg = Dsv4Config::v41();
        for (rows, max_seq) in [(449usize, 49_152usize), (5, 49_152), (63, 262_144), (3, 262_144)] {
            let off = Dsv4Scratch::bscore_off(&cfg, rows, max_seq);
            assert_eq!(off % 16, 0, "rows={rows} max_seq={max_seq} bscore_off={off}");
        }
    }
}
