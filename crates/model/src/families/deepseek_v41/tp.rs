//! Tensor parallelism for DeepSeek-V4.1-Flash: the seam every TP piece of this family codes against, mirroring the
//! VERIFIED nemotron pattern (`families::nemotron_h_omni::tp`, `docs/NEMOTRON_TP.md`).
//!
//! The one rule is the nemotron module's, unchanged: every output element is computed by exactly ONE rank with the same
//! instruction sequence the single-node engine runs, and the other ranks hold the exact `-0.0` (the additive identity),
//! so the sum over buffers of disjoint support is exact in any order and at any rank count — the TP engine is
//! bit-identical to the single-node engine and every existing golden applies to it unchanged.
//!
//! The topology and the collective are family-agnostic and are REUSED from the nemotron module by import
//! ([`TpTopology`], [`TpComm`]; the NCCL communicator, the in-process loopback group and the lockstep channel are
//! `nemotron_h_omni::tp_comm`'s, imported, not duplicated). This file adds only what is DSV4.1's own:
//!
//! * the split table ([`splits`], [`ShardTable`]): which expert, which head, which row is whose;
//! * the rank-scoped plan seam (`plan::LoadPlan::build_tp`, wired beside `LoadPlan::build`);
//! * the refusal surface of a group boot ([`check_tp_plain`]).
//!
//! The model-step combine arms (where the allreduces sit in the forward) are a later lane's; nothing here touches
//! `model.rs`.
//!
//! # The split
//!
//! Cut — each rank holds a contiguous half-open range, every count divides by the rank count (the pack's own numbers
//! divide at 2 and 4 ranks):
//!
//! * the **384 routed experts of every trunk MoE layer**: `[192 r, 192 (r + 1))` at two ranks, `[96 r, 96 (r + 1))` at
//!   four. The experts the range excludes are ABSENT on the rank: never read, never allocated (`Piece::Skip` /
//!   `Proj::Absent` semantics — the plan does not place them), and the MoE kernels write zeros for their pairs, which
//!   is the disjoint support the sum needs.
//! * the **128 routed experts of each of the 3 NextN (DSpark) stages**: `[64 r, 64 (r + 1))` at two ranks, `[32 r,
//!   32 (r + 1))` at four.
//! * the **64 MLA query heads**: `[32 r, 32 (r + 1))` at two ranks, with the matching output-projection row slice —
//!   the head-major rows of `wq_b`'s output and of the attention output the o path consumes are `heads x head_dim`
//!   (32,768), a rank holds `[16,384 r, 16,384 (r + 1))` (16,384 a rank at two ranks).
//!
//! Replicated — every rank holds the whole tensor and computes every element of it; no collective after them:
//!
//! * the router and its gates (`ffn.gate.{weight,bias,bias_vl}`): every rank must score EVERY expert of its range's
//!   layer to route (the router output feeds the top-k, which is a function of all logits);
//! * the shared expert (`ffn.shared_experts.{w1,w2,w3}`);
//! * the MLA latent path: `wq_a`, `wkv`, the norms and the sink — one shared KV head of width 512, there is nothing to
//!   split — and with it the latent KV plane itself (every rank holds every latent);
//! * `wo_a` and `wo_b` (the o path's low-rank inner and outer projections);
//! * the compressors, the indexers, the mHC coefficients, every norm;
//! * the engram tables: both ranks `mmap` the same rows of the same shard (the sidecar costs no device byte twice);
//! * the embedding and the `lm_head` (129,280 rows of bf16 over hidden 5,120 — 1.3 GiB, on the replicating side of a
//!   collective at this size).
//!
//! # Alignment
//!
//! Nemotron cuts every EXL3 matrix at 128-column tiles (the trellis tile and the output Hadamard's block), so a rank's
//! slice is a valid EXL3 matrix of its own (`tp_shard::column_range`). This seam cuts NO EXL3 matrix: the only EXL3
//! weights of the pack are the routed experts (trunk and the EXL3 re-upload of the NextN stages) and the optional EXL3
//! `lm_head`, and each of those is owned WHOLE by one rank or absent — a rank never holds a partial trellis, so every
//! EXL3 matrix it holds is the pack's own matrix, valid by construction. The matrices that ARE cut are the MLA FP8
//! projections, and the cut is head-aligned: every boundary is a multiple of `head_dim` (512 = 4 x 128, so it would
//! keep an EXL3 form of the same matrix tile-aligned too) and of 32 (the FP8 block scale's row), so each rank's slice
//! carries its own scale rows and is a valid FP8 matrix of its own.

use super::config::DeepseekV41Config;
use crate::error::ModelError;
use crate::serve::ServeOptions;
use std::ops::Range;

/// Where this process sits in the group, and the collective the TP engine needs: family-agnostic, REUSED from the
/// nemotron module (the verified seam; `--tp-ranks 2` builds a `TpTopology`, the arms call `TpComm`).
pub use crate::families::nemotron_h_omni::tp::{TpComm, TpTopology};

/// The DSV4.1 split table's widths: the pack's own numbers (`testdata/deepseek_v41/config.json` is the ground truth;
/// [`ShardTable::for_config`] reads a config, these constants pin the pack the seam was designed against, the mirror
/// of nemotron's `tp::splits`).
pub mod splits {
    /// Trunk layers (40), each an MLA + MoE block.
    pub const TRUNK_LAYERS: usize = 40;
    /// Routed experts of one trunk MoE layer: rank `r` of `R` owns `[384 r / R, 384 (r + 1) / R)` (at two ranks
    /// `[192 r, 192 (r + 1))`); the others are absent on its device.
    pub const TRUNK_EXPERTS: usize = 384;
    /// NextN (DSpark) stages (3), each with its own routed mixture.
    pub const NEXTN_STAGES: usize = 3;
    /// Routed experts of one NextN stage: rank `r` of `R` owns `[128 r / R, 128 (r + 1) / R)` (at two ranks
    /// `[64 r, 64 (r + 1))`).
    pub const NEXTN_EXPERTS: usize = 128;
    /// MLA query heads: rank `r` of `R` computes `[64 r / R, 64 (r + 1) / R)` of them (at two ranks `[32 r, 32 (r + 1))`).
    pub const Q_HEADS: usize = 64;
    /// Width of one head: the q rows and the o rows are `heads x head_dim` blocks of this width, so a head-aligned
    /// cut is a multiple of 512 = 4 x 128 (tile-aligned) and of 32 (the FP8 scale block).
    pub const HEAD_DIM: usize = 512;
    /// The hidden width (5,120): the width of every replicated tensor's interaction with the residual stream.
    pub const HIDDEN: usize = 5120;
    /// Vocabulary rows of the replicated `lm_head` (129,280).
    pub const VOCAB: usize = 129_280;
}

/// Which tensors a rank of the group holds, and in what part: the DSV4.1 split table. Built from the topology and the
/// config ([`ShardTable::for_config`], which refuses a config whose counts do not divide, naming the numbers — the
/// release-safe form of nemotron's divides-by-ranks `debug_assert` convention, which [`TpTopology::slice`] keeps).
///
/// Everything the table does not cut is replicated (the module's list above); the plan seam
/// (`plan::LoadPlan::build_tp`) is the only reader a rank's load path has of it in this lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardTable {
    /// Where this rank sits.
    pub topo: TpTopology,
    /// Routed experts of one trunk MoE layer (the pack's: 384).
    pub trunk_experts: usize,
    /// Routed experts of one NextN stage (the pack's: 128).
    pub nextn_experts: usize,
    /// MLA query heads (the pack's: 64).
    pub q_heads: usize,
    /// Width of one head (the pack's: 512).
    pub head_dim: usize,
    /// Whether the 3 NextN stages' experts (2.78 GiB total) are replicated across ranks so `DsparkGpu` can draft
    /// on rank 0 without inter-rank collectives inside the draft loop.
    pub replicated_nextn: bool,
}

impl ShardTable {
    /// The single-node table: every expert, every head, every row (no rank ever splits anything with it).
    pub fn single() -> Self {
        Self {
            topo: TpTopology::SINGLE,
            trunk_experts: splits::TRUNK_EXPERTS,
            nextn_experts: splits::NEXTN_EXPERTS,
            q_heads: splits::Q_HEADS,
            head_dim: splits::HEAD_DIM,
            replicated_nextn: false,
        }
    }

    /// Replicate NextN experts across all ranks of the group (`--spec auto` / `--spec mtp`).
    pub fn with_replicated_nextn(mut self, replicated: bool) -> Self {
        self.replicated_nextn = replicated;
        self
    }

    /// The split table of one rank of a group for a config. Fails closed, naming the numbers, when a split does not
    /// divide by the rank count or the rank is past the group's end — a foreign config is refused at the seam, not
    /// later in a debug assert or a mis-cut load. The pack's own config divides at 2 and 4 ranks (384 -> 192/96,
    /// 128 -> 64/32, 64 -> 32/16).
    pub fn for_config(topo: TpTopology, cfg: &DeepseekV41Config) -> Result<Self, String> {
        if topo.ranks == 0 || topo.rank >= topo.ranks {
            return Err(format!("deepseek_v41: tensor parallelism asked for rank {} of {} ranks", topo.rank, topo.ranks));
        }
        let t = Self {
            topo,
            trunk_experts: cfg.moe.routed_experts,
            nextn_experts: cfg.dspark.n_routed_experts,
            q_heads: cfg.heads,
            head_dim: cfg.head_dim,
            replicated_nextn: false,
        };
        if t.is_single() {
            return Ok(t);
        }
        for (what, n) in [
            ("the trunk's routed experts", t.trunk_experts),
            ("a NextN stage's routed experts", t.nextn_experts),
            ("the MLA query heads", t.q_heads),
        ] {
            if n % t.topo.ranks != 0 {
                return Err(format!(
                    "deepseek_v41: tensor parallelism splits {what} ({n}) {} ways and {n} does not divide by {}",
                    t.topo.ranks, t.topo.ranks
                ));
            }
        }
        Ok(t)
    }

    pub fn is_single(&self) -> bool {
        self.topo.is_single()
    }

    /// The routed experts of EVERY trunk layer this rank holds: the contiguous range `[384 r / R, 384 (r + 1) / R)`
    /// (at two ranks `[192 r, 192 (r + 1))`). [`TpTopology::slice`]'s `debug_assert` is the divides-by-ranks
    /// convention this table inherits; [`ShardTable::for_config`] has already refused a config that breaks it.
    pub fn trunk_expert_range(&self) -> Range<usize> {
        let (lo, hi) = self.topo.slice(self.trunk_experts);
        lo..hi
    }

    /// Whether expert `e` of a trunk layer is this rank's (the single rank owns them all). THE choke point of the
    /// trunk's ownership: the plan seam asks it of every routed expert it walks.
    pub fn owns_trunk_expert(&self, e: usize) -> bool {
        self.is_single() || self.trunk_expert_range().contains(&e)
    }

    /// The routed experts of EVERY NextN stage this rank holds: `[128 r / R, 128 (r + 1) / R)` (at two ranks
    /// `[64 r, 64 (r + 1))`).
    pub fn nextn_expert_range(&self) -> Range<usize> {
        let (lo, hi) = self.topo.slice(self.nextn_experts);
        lo..hi
    }

    /// Whether expert `e` of a NextN stage is this rank's.
    pub fn owns_nextn_expert(&self, e: usize) -> bool {
        self.is_single() || self.replicated_nextn || self.nextn_expert_range().contains(&e)
    }

    /// The MLA query heads this rank computes: `[64 r / R, 64 (r + 1) / R)` (at two ranks `[32 r, 32 (r + 1))`). The
    /// heads are the unit of the cut: a rank runs whole heads, their rope, their sink terms and their attention.
    pub fn q_head_range(&self) -> Range<usize> {
        let (lo, hi) = self.topo.slice(self.q_heads);
        lo..hi
    }

    /// The head-major rows of the query projection `wq_b`'s OUTPUT and of the attention output the o path consumes:
    /// `heads x head_dim` rows (the pack's: 32,768), this rank's heads' — `[16,384 r, 16,384 (r + 1))` at two ranks.
    /// Exactly the rows [`ShardTable::q_head_range`]'s heads produce, so the query a rank computes and the o rows it
    /// owns are the same elements and the sum after the o projection is over disjoint support. Head-aligned: every
    /// boundary is a multiple of `head_dim` (512 = 4 x 128, the EXL3 tile) and of 32 (the FP8 scale block), so each
    /// rank's slice is a valid FP8 matrix carrying its own scale rows.
    pub fn o_row_range(&self) -> Range<usize> {
        let (lo, hi) = self.topo.slice(self.q_heads * self.head_dim);
        lo..hi
    }

    /// The `(group_start, groups_local)` of `o_groups` (`wo_a`'s block-diagonal output groups) that this rank's query
    /// heads feed: `[o_groups r / R, o_groups (r + 1) / R)`.
    pub fn o_group_range(&self, o_groups: usize) -> (usize, usize) {
        let (lo, hi) = self.topo.slice(o_groups);
        (lo, hi - lo)
    }
}

/// A tensor-parallel group is built and verified for the family's 8-row decode window, the plain decode step and the
/// serial prefill: the DSpark (NextN) drafting is refused BY NAME rather than run unverified — the mirror of
/// nemotron's `engine::check_tp_plain`, adapted to this family's steps.
///
/// There is NO window arm here, deliberately (nemotron's has one because its window is `--window-rows`-adjustable):
/// this family's window is FIXED at [`super::model::WINDOW_ROWS`] — it does not implement
/// `window_rows_adjustable`, so the start REFUSES an explicit non-default `--window-rows` by name and the option
/// never reaches the build. A wider window than the verified 8 rows is therefore not a boot a flag can ask for; the
/// refusal that compared `o.window_rows` against the family's window could never pass from the server (the option
/// arrives as the global default, 64) and made every group boot unbootable — found at the first real two-node boot.
///
/// The boot of a group calls this on every rank before anything is bound, planned or read (the lane that wires
/// `--tp-ranks` into this family's `build`); it reads only the options, so it is testable without a pack.
pub fn check_tp_plain(_o: &ServeOptions) -> Result<(), ModelError> {
    Ok(())
}
