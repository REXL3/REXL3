//! TP-W: the per-rank device weights of TP-exact (`docs/NEMOTRON_TP.md`, sections 2, 3 and 13), for any rank count `R` (verified and
//! measured at two ranks; R > 2 is proved per shape on one device and by the CPU shard-property tests, never measured on a fabric).
//!
//! A rank `r` of `R` holds, of the pack:
//!
//! - every dense EXL3 linear of the trunk **sliced along its output columns** at 128 alignment (Mamba `in_proj`, attention q, k, v, o),
//!   rank `r` the tiles `[floor(T r / R), floor(T (r + 1) / R))` of the `T` tiles ([`column_range`], the one cut rule), except the
//!   matrices on the plan's `replicate` list (a rank then holds the whole matrix and computes every output column itself).
//!   The slice is a valid EXL3 layer of `n_local` columns: the trellis `[K/16, n_local/16, 16 * bits]`, the output scale `svh` `[n_local]`,
//!   the input scale `suh` whole. The model's [`MambaW`](super::device::MambaW) and [`AttnW`](super::device::AttnW) keep their types:
//!   the `Exl3Weights` reports `n = n_local` and the arms read that;
//! - the routed experts `[E r / R, E (r + 1) / R)` of every MoE layer of the trunk (at two ranks `[256 r, 256 (r + 1))`; at most one
//!   expert of skew); the others are not read from disk and not on the device (`Proj::Absent`: the MoE kernels write zeros for those
//!   pairs, the disjoint support the allreduce sums). The router, fc1, fc2 and the shared expert stay whole;
//! - the `lm_head` rows [`head_row_cut`]`(vocab, r, R)` (two ranks: `[vocab r / 2, vocab (r + 1) / 2)`);
//! - the MTP head's routed experts, by the same expert range (the head is a latent-MoE block of 512 bf16 experts:
//!   [`ShardPlan::head_experts`], on by default for a group, the draft call then allreduces the block's `y` like the trunk's layers
//!   do); the head's dense bf16 projections (`eh_proj`, q, k, v, o, the shared expert, fc1, fc2, the router) stay whole;
//! - everything else (embedding, norms, conv, A_log, D, dt_bias, gated norm) whole.
//!
//! **What R > 2 changes in the plan.** [`ShardPlan::for_config`] replicates a matrix whose slice cannot keep the whole matrix's kernel
//! plan (the `in_proj` below 64 tiles a slice, which is every `R >= 3`; the `out_proj` at every `R`) and one with fewer tiles than
//! ranks (k and v: 2 tiles, so none can be cut at `R >= 3`). Replicating never changes a bit; it costs the matrix's bytes on every
//! rank and no collective after it.
//!
//! **Slicing happens as the loader reads** ([`upload_tp`]): each slice's bytes go straight from the shard into a buffer of the slice's own
//! size (one file range per trellis K-row run, one for `svh`, one for the `lm_head` rows), and an expert of the other rank is never
//! requested. No whole matrix is ever on the device, so the peak is the rank's share. The loader merges file ranges that start within
//! 256 KiB of each other into one read, so the dense linears' runs (16 to 75 KiB, a rank-width apart) are read from disk whole and
//! copied in part; the experts, 97 percent of the bytes, are skipped outright.
//!
//! **Why this is exact** (and where it is not). An output column of an EXL3 layer is `(x H_K . W_q[:, col] . H_N) * svh[col]`:
//! the Hadamard over the output is per 128-block, each column accumulates over K alone, and the K reduction order is the decode
//! kernel's `plan_for(K, N)`: (K splits, warps a block), a function of the shape. A slice keeps every column's bits exactly when
//! `plan_for(K, n_local) == plan_for(K, N)`. [`decode_plan`] mirrors that function so a plan can decide on the host, and the GPU test
//! (`tests/nemotron_omni_tp_weights.rs`) holds the mirror against the kernel and the slices against the whole matrices byte for byte.
//! A matrix whose plan moves (Mamba `out_proj`: K = 8192, N 4096 to 2048 goes from 8 K-splits to 16) is replicated by
//! [`ShardPlan::for_config`]: it costs its bytes on both ranks and never different bits.

use super::config::OmniConfig;

// the kernel-plan mirror and the one cut rule are family-neutral (`crate::shard`); this family's plan is built on them
use super::device::NemotronWeights;
use super::inventory::Inventory;
use super::tp::{splits, TpTopology};
use super::tp_balance::{contiguous_range, ExpertMap};
use super::weights::{self, LayerPlan, Linear, Unit, UnitId, Uploaded, Want, Weights};
use crate::error::ModelError;
pub use crate::shard::{column_range, decode_plan, replication_reason, slice_keeps_plan, sliceable, TILE};
use gb_cuda::arena::Arena;
use gb_cuda::loader::{Loader, Options, Xform};
use gb_cuda::Stream;
use gb_exl3::checkpoint::dtype_size;
use std::collections::BTreeSet;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

/// Arena block size of a rank's upload (the same as [`weights::upload`]'s).
const ARENA_BLOCK_BYTES: usize = 1 << 30;

/// The dense EXL3 linears of the trunk the plan can slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dense {
    MambaIn,
    MambaOut,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
}

impl Dense {
    pub const ALL: [Dense; 6] = [Dense::MambaIn, Dense::MambaOut, Dense::AttnQ, Dense::AttnK, Dense::AttnV, Dense::AttnO];

    pub fn name(self) -> &'static str {
        match self {
            Dense::MambaIn => "mamba in_proj",
            Dense::MambaOut => "mamba out_proj",
            Dense::AttnQ => "attn q_proj",
            Dense::AttnK => "attn k_proj",
            Dense::AttnV => "attn v_proj",
            Dense::AttnO => "attn o_proj",
        }
    }

    /// The role of a trunk layer's EXL3 linear, from the last component of its prefix. The MoE layers' linears (`fc1_latent_proj`,
    /// `shared_experts.up_proj`, `experts.{e}.down_proj`, ...) have other names and are never sliced.
    pub fn of_prefix(prefix: &str) -> Option<Dense> {
        match prefix.rsplit('.').next()? {
            "in_proj" => Some(Dense::MambaIn),
            "out_proj" => Some(Dense::MambaOut),
            "q_proj" => Some(Dense::AttnQ),
            "k_proj" => Some(Dense::AttnK),
            "v_proj" => Some(Dense::AttnV),
            "o_proj" => Some(Dense::AttnO),
            _ => None,
        }
    }
}

/// `(K, N)` of each dense linear of the config, as `weights::units` plans them.
pub fn dense_shapes(cfg: &OmniConfig) -> [(Dense, usize, usize); 6] {
    let inner = cfg.mamba_heads * cfg.mamba_dim;
    let in_proj_n = 2 * inner + 2 * cfg.groups * cfg.state + cfg.mamba_heads;
    let attn_n = cfg.heads * cfg.head_dim;
    let kv_n = cfg.kv_heads * cfg.head_dim;
    [
        (Dense::MambaIn, cfg.hidden, in_proj_n),
        (Dense::MambaOut, inner, cfg.hidden),
        (Dense::AttnQ, cfg.hidden, attn_n),
        (Dense::AttnK, cfg.hidden, kv_n),
        (Dense::AttnV, cfg.hidden, kv_n),
        (Dense::AttnO, attn_n, cfg.hidden),
    ]
}

/// Bytes the decode kernel reads for a `k x n` 4-bit EXL3 layer: the trellis (`k * n / 2`) and the two scale vectors (fp16).
pub fn dense_bytes(k: usize, n: usize) -> u64 {
    (k * n / 2 + 2 * (k + n)) as u64
}

/// Rows of the `lm_head` (or of a restricted draft table over its rows) are cut in units of this many rows: every boundary but the
/// last is a multiple of it, so a rank's column offset into the `[rows, total]` logits is 16-byte aligned (`scatter_cols_f32`'s float4
/// path) and a row of 4,096 bf16 (8 KiB) starts 16-byte aligned in the table. The last rank absorbs the remainder of a row count that
/// is not a multiple of it.
pub const HEAD_ROW_UNIT: usize = 4;

/// The unit an EXL3 `lm_head`'s rows (its output columns) are cut in: the 128-column block of the EXL3 output Hadamard ([`TILE`]), so a
/// rank's slice is a valid EXL3 layer whose every column is the whole layer's (its trellis `[K/16, n_local/16, 16 * bits]`, its `svh`
/// `[n_local]`), and its decode plan is the whole layer's while the slice keeps at least 64 blocks (`slice_keeps_plan`).
pub const EXL3_HEAD_ROW_UNIT: usize = TILE;

/// A table of fewer rows than this per rank is not worth a collective (1 MiB of bf16 rows a rank, about 4 us of read against a
/// collective of 15 us and more) and every rank scores it whole (`tp_arm::lm_head_logits`).
pub const HEAD_MIN_ROWS_PER_RANK: usize = 128;

/// The rows `[lo, hi)` rank `rank` of `ranks` scores of a `total`-row table (the trunk's `lm_head`, or the restricted draft head's
/// sub-table): the interior boundaries are `total * r / ranks` rounded UP to a multiple of [`HEAD_ROW_UNIT`], so every rank's width is
/// at most `ceil4(ceil(total / ranks))` (a smaller table never needs a wider temp than the vocabulary does: [`head_temp_rows`]) and a
/// vocabulary that divides evenly is cut exactly in equal parts (two ranks: 65,536 | 65,536 of 131,072, 24,576 | 24,576 of 49,152).
/// Any row cut is exact: a logit depends on its own weight row only (`nemotron_dense.cu`: the order of every operation depends on k
/// and the stripe, never on m or the column set; the device test holds it at the R = 3 to 8 cuts).
pub fn head_row_cut(total: usize, rank: usize, ranks: usize) -> (usize, usize) {
    head_row_cut_in(total, rank, ranks, HEAD_ROW_UNIT)
}

/// [`head_row_cut`] in units of `unit` rows ([`HEAD_ROW_UNIT`] for a bf16 table, [`EXL3_HEAD_ROW_UNIT`] for an EXL3 `lm_head`).
pub fn head_row_cut_in(total: usize, rank: usize, ranks: usize, unit: usize) -> (usize, usize) {
    assert!(ranks >= 1 && rank < ranks, "rank {rank} of {ranks}");
    let edge = |r: usize| -> usize {
        if r == 0 {
            0
        } else if r >= ranks {
            total
        } else {
            (total * r / ranks).div_ceil(unit).saturating_mul(unit).min(total)
        }
    };
    (edge(rank), edge(rank + 1))
}

/// The widest rank's row count of [`head_row_cut`] over `ranks`: what a temp that holds any rank's logits must hold per row.
pub fn head_row_cut_max(total: usize, ranks: usize) -> usize {
    (0..ranks.max(1)).map(|r| head_row_cut(total, r, ranks.max(1))).map(|(lo, hi)| hi - lo).max().unwrap_or(0)
}

/// Rows of the logits temp (`Bufs::tp_logits`) of a group over a vocabulary of `vocab` rows: `ceil4(ceil(vocab / ranks))`, the same
/// on every rank (not the rank's own width, so a call's capacity never depends on which rank runs it), and an upper bound of every
/// rank's width of any table of at most `vocab` rows. Two ranks: 65,536.
pub fn head_temp_rows(vocab: usize, ranks: usize) -> usize {
    head_temp_rows_in(vocab, ranks, HEAD_ROW_UNIT)
}

/// [`head_temp_rows`] for a head cut in units of `unit` rows (an EXL3 `lm_head`: [`EXL3_HEAD_ROW_UNIT`]; two ranks: 65,536 too).
pub fn head_temp_rows_in(vocab: usize, ranks: usize, unit: usize) -> usize {
    vocab.div_ceil(ranks.max(1)).div_ceil(unit) * unit
}

/// Whether a group of `ranks` splits a `total`-row table across its ranks (each scores its rows, one collective combines them) and
/// does not score it whole on every rank: a function of the rank count and the row count only, so every rank decides alike.
pub fn head_splits(total: usize, ranks: usize) -> bool {
    ranks >= 2 && total >= HEAD_MIN_ROWS_PER_RANK * ranks
}

/// What a rank's upload does with one planned tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Piece {
    /// The tensor as stored.
    Whole,
    /// Not on this rank: a routed expert of another rank. Never read, never allocated.
    Skip,
    /// The `.trellis` of a `k x n` linear, output columns `[lo, hi)`: `[k/16, (hi - lo)/16, 16 * bits]`.
    Trellis { k: usize, n: usize, lo: usize, hi: usize },
    /// The `.svh` of a sliced linear, elements `[lo, hi)`.
    Svh { lo: usize, hi: usize },
    /// Rows `[lo, hi)` of a two-dimensional tensor (`lm_head`).
    Rows { lo: usize, hi: usize },
}

/// One dense matrix's decision in a plan ([`ShardPlan::decisions`]): `(role, K, N, held)`, `held` being the columns of each rank
/// (`Ok`, rank 0 first: the matrix is sliced) or the reason every rank holds it whole (`Err`).
pub type Decision = (Dense, usize, usize, Result<Vec<usize>, String>);

/// Which tensors a rank of the group holds, and in what part. Built from the topology and the dense matrices that stay whole.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardPlan {
    pub topo: TpTopology,
    /// Dense linears every rank holds whole. Their output is computed by every rank (no collective is needed after them).
    pub replicate: BTreeSet<Dense>,
    /// Keep the whole `lm_head` on every rank (the arms then read the rank's rows as a sub-range): costs `vocab * hidden / 2` bf16 bytes
    /// of residency and no read. The MTP head's restricted sub-table copies rows of the trunk's `lm_head` by token id, and the full-vocab
    /// draft head scores every row, so a rank that builds either from the device needs this; the restricted table can instead be read
    /// from the pack with [`read_lm_head_rows`].
    pub lm_head_full: bool,
    /// The MTP head's routed experts are split like the trunk's: rank `r` holds experts `[256 r, 256 (r + 1))` and the other half is
    /// `Proj::Absent` (never read, never on the device); the draft call's MoE then runs `forward_in_tp` and allreduces its `y`
    /// (`mtp::head_block`). Off: the head is whole on every rank and its call has no collective. Both ranks must agree
    /// (`tp_arm::plan_flags`). Meaningless for a single rank.
    pub head_experts: bool,
    /// The routed experts' ownership table (`tp_balance::ExpertMap`, B7): which rank holds each expert of each MoE layer. `None`:
    /// the contiguous ranges [`ShardPlan::experts`] (what every verified run used). Ownership is a lookup (a rank that does not hold
    /// an expert has `Proj::Absent` for it), so any table is bit-exact; a table must be the same on every rank
    /// (`tp_balance::expert_map_fingerprint`, compared at the rendezvous).
    pub expert_map: Option<Arc<ExpertMap>>,
}

impl ShardPlan {
    /// One rank: every tensor whole, no collective. [`upload_tp`] and `into_device_tp` with it are the single-node load.
    pub fn single() -> Self {
        Self { topo: TpTopology::SINGLE, replicate: BTreeSet::new(), lm_head_full: false, head_experts: false, expert_map: None }
    }

    /// A rank of a group with every dense linear sliced (the test's plan: it measures every shape, and the serving plan
    /// [`ShardPlan::for_config`] replicates only the ones whose kernel plan moves). The head's routed experts are split.
    pub fn slice_everything(topo: TpTopology) -> Self {
        Self { topo, replicate: BTreeSet::new(), lm_head_full: false, head_experts: !topo.is_single(), expert_map: None }
    }

    /// This plan with the MTP head's routed experts whole on every rank (the draft call then has no collective in its MoE).
    pub fn replicating_head(mut self) -> Self {
        self.head_experts = false;
        self
    }

    /// This plan with the routed experts owned by `map` instead of the contiguous ranges (B7). A single rank ignores it (it holds
    /// every expert). The caller checks the map against the model and the group (`ExpertMap::check_for`).
    pub fn with_expert_map(mut self, map: Arc<ExpertMap>) -> Self {
        if !self.topo.is_single() {
            self.expert_map = Some(map);
        }
        self
    }

    /// This plan with `roles` also held whole by every rank. Replicating never changes a bit; it trades the matrix's bytes on the
    /// rank (half of them are read again every step) for the collective that sums the slices: a sliced matrix saves `bytes / 2` of
    /// read, `bytes / 2 / 273 GB/s`, and pays one allreduce after it (30 us or more on the fabric). Only `in_proj` (38 MB a layer) is
    /// clearly on the slicing side of that; `out_proj` (17 MB) is at break-even, and the attention matrices (8 MB q and o, 0.5 MB k and
    /// v) are on the replicating side unless their allreduces are fused (q, k and v can share one).
    pub fn replicating(mut self, roles: &[Dense]) -> Self {
        self.replicate.extend(roles.iter().copied());
        self
    }

    /// The serving plan of a rank: dense linears sliced unless `plan_for` of the slice differs from the whole matrix's for some rank
    /// (then the K reduction would be regrouped, the bits would differ, and the matrix is replicated), or a rank would hold no tile
    /// ([`slice_keeps_plan`] says both). See [`ShardPlan::replicating`] for the matrices it is cheaper not to slice. At two ranks only
    /// the Mamba `out_proj` is replicated; at three and more also the `in_proj` (a slice below 64 tiles) and k and v (fewer tiles than
    /// ranks): [`ShardPlan::decisions`] lists each with its reason.
    pub fn for_config(topo: TpTopology, cfg: &OmniConfig) -> Self {
        let mut p = Self::slice_everything(topo);
        if !topo.is_single() {
            for (role, k, n) in dense_shapes(cfg) {
                if !slice_keeps_plan(k, n, topo.ranks) {
                    p.replicate.insert(role);
                }
            }
        }
        p
    }

    /// The per-matrix decision of this plan for `cfg`: `(role, k, n, held)` where `held` is `Ok(widths)` (each rank's column count,
    /// rank 0 first: the matrix is sliced) or `Err(reason)` (every rank holds it whole: the plan's `replicate` list, with the reason
    /// the plan keeps for it when it is the plan's own: [`replication_reason`]; "plan choice" when the caller replicated an exact
    /// matrix for cost). What the boot logs (`tp_arm::shard_plan_line`).
    pub fn decisions(&self, cfg: &OmniConfig) -> Vec<Decision> {
        let ranks = self.topo.ranks;
        dense_shapes(cfg)
            .into_iter()
            .map(|(role, k, n)| {
                let held = if ranks <= 1 {
                    Err("single rank".to_string())
                } else if self.replicate.contains(&role) || !sliceable(n, ranks) {
                    Err(replication_reason(k, n, ranks).unwrap_or_else(|| "plan choice (exact, replicated for cost)".to_string()))
                } else {
                    Ok((0..ranks)
                        .map(|r| {
                            let (lo, hi) = column_range(n, r, ranks);
                            hi - lo
                        })
                        .collect())
                };
                (role, k, n, held)
            })
            .collect()
    }

    /// The widest slice of any matrix `for_config` slices at `ranks` ranks, in columns (0 when none is sliced): what a sliced linear's
    /// local output temp must hold per row, whatever the rank. At two ranks the second half of the `in_proj`, 9,344; at three and
    /// more the `in_proj` is replicated and the widest slice is a q or o share. The memory plan prices this (`memory::tp_scratch_bytes`);
    /// the serving plan may replicate more (`REXL3_TP_ATTN`, `REXL3_TP_INPROJ`), which only makes the price conservative.
    pub fn widest_slice(cfg: &OmniConfig, ranks: usize) -> usize {
        if ranks <= 1 {
            return 0;
        }
        dense_shapes(cfg)
            .into_iter()
            .filter(|&(_, k, n)| slice_keeps_plan(k, n, ranks))
            .map(|(_, _, n)| (0..ranks).map(|r| column_range(n, r, ranks)).map(|(lo, hi)| hi - lo).max().unwrap_or(0))
            .max()
            .unwrap_or(0)
    }

    /// [`ShardPlan::for_config`], and also replicated: every group of dense linears whose slicing saves less read time than the
    /// allreduce after it costs. A group is a set of matrices that share one collective (q, k and v; the others are alone); slicing it
    /// saves `bytes * (ranks - 1) / ranks` of read at `bandwidth_gb_s`, and costs `collective_us`. The numbers are the caller's
    /// (TP-N's fabric measurement for the collective at the step's row count; 273 GB/s is the GB10's); the exactness is the same
    /// either way.
    pub fn for_cost(topo: TpTopology, cfg: &OmniConfig, collective_us: f64, bandwidth_gb_s: f64) -> Self {
        let mut p = Self::for_config(topo, cfg);
        if topo.is_single() {
            return p;
        }
        let shapes = dense_shapes(cfg);
        let bytes = |roles: &[Dense]| -> u64 {
            shapes.iter().filter(|(r, _, _)| roles.contains(r)).map(|&(_, k, n)| dense_bytes(k, n)).sum()
        };
        for group in [&[Dense::MambaIn][..], &[Dense::MambaOut], &[Dense::AttnQ, Dense::AttnK, Dense::AttnV], &[Dense::AttnO]] {
            let saved_us = bytes(group) as f64 * (topo.ranks - 1) as f64 / topo.ranks as f64 / (bandwidth_gb_s * 1e3);
            if saved_us <= collective_us {
                p.replicate.extend(group.iter().copied());
            }
        }
        p
    }

    pub fn is_single(&self) -> bool {
        self.topo.is_single()
    }

    /// Whether `role` is cut across the ranks.
    pub fn slices(&self, role: Dense) -> bool {
        !self.topo.is_single() && !self.replicate.contains(&role)
    }

    /// Output columns of the `n`-wide matrix `role` this rank holds (and computes): the whole width unless the role is sliced AND
    /// `n` has a whole tile for every rank ([`sliceable`]: a matrix of fewer tiles than ranks stays whole on every rank, in every plan,
    /// including [`ShardPlan::slice_everything`]).
    pub fn columns(&self, role: Dense, n: usize) -> (usize, usize) {
        if self.slices(role) && sliceable(n, self.topo.ranks) {
            column_range(n, self.topo.rank, self.topo.ranks)
        } else {
            (0, n)
        }
    }

    /// The routed experts (of `experts`) this rank holds under the CONTIGUOUS rule ([`tp_balance::contiguous_range`]). With an
    /// ownership table the experts of a layer are not a range: use [`ShardPlan::layer_experts`]. Panics when a table is present (a
    /// range read of a table would silently load the wrong experts).
    pub fn experts(&self, experts: usize) -> Range<usize> {
        assert!(
            self.expert_map.is_none(),
            "ShardPlan::experts is the contiguous rule: with an expert map ownership is per layer (layer_experts, owns_layer_expert)"
        );
        contiguous_range(experts, self.topo.rank, self.topo.ranks)
    }

    /// Whether this rank holds expert `e` under the contiguous rule. With an ownership table use [`ShardPlan::owns_layer_expert`] or
    /// [`ShardPlan::owns_head_expert`] (panics here, as [`ShardPlan::experts`]).
    pub fn owns_expert(&self, e: usize, experts: usize) -> bool {
        self.experts(experts).contains(&e)
    }

    /// Whether this rank holds expert `e` of trunk layer `layer`: the table's row when the plan has one, the contiguous range
    /// otherwise. THE choke point of the trunk's ownership (`piece`, `local_linears`, the bridge's MoE tables).
    pub fn owns_layer_expert(&self, layer: usize, e: usize, experts: usize) -> bool {
        match &self.expert_map {
            Some(m) => m
                .owner(layer, e)
                .map_or_else(|| contiguous_range(experts, self.topo.rank, self.topo.ranks).contains(&e), |r| r == self.topo.rank),
            None => contiguous_range(experts, self.topo.rank, self.topo.ranks).contains(&e),
        }
    }

    /// Whether this rank holds expert `e` of the MTP head's block: the table's head row when it has one, the contiguous range
    /// otherwise (the head's experts are contiguous unless the table carries its own row for the block).
    pub fn owns_head_expert(&self, e: usize, experts: usize) -> bool {
        match self.expert_map.as_ref().and_then(|m| m.head_owner(e)) {
            Some(r) => r == self.topo.rank,
            None => contiguous_range(experts, self.topo.rank, self.topo.ranks).contains(&e),
        }
    }

    /// The experts of trunk layer `layer` this rank holds, ascending.
    pub fn layer_experts(&self, layer: usize, experts: usize) -> Vec<usize> {
        (0..experts).filter(|&e| self.owns_layer_expert(layer, e, experts)).collect()
    }

    /// The `lm_head` rows (of `vocab`) this rank's GEMV computes ([`head_row_cut`]: boundaries in units of [`HEAD_ROW_UNIT`] rows, any
    /// rank count). With [`ShardPlan::lm_head_full`] the buffer holds every row, and these are the rows of it the rank reads.
    pub fn lm_head_rows(&self, vocab: usize) -> (usize, usize) {
        head_row_cut(vocab, self.topo.rank, self.topo.ranks)
    }

    /// The output columns (of `vocab`) of an EXL3 `lm_head` this rank holds and computes: [`head_row_cut_in`] at
    /// [`EXL3_HEAD_ROW_UNIT`] (two ranks: the bf16 cut, 65,536 | 65,536).
    pub fn lm_head_cols_exl3(&self, vocab: usize) -> (usize, usize) {
        head_row_cut_in(vocab, self.topo.rank, self.topo.ranks, EXL3_HEAD_ROW_UNIT)
    }

    /// What the upload does with `want` of `unit`. `experts` is the routed-expert count of an MoE layer (`OmniConfig::experts`).
    pub fn piece(&self, unit: &Unit, want: &Want, experts: usize) -> Piece {
        if self.is_single() {
            return Piece::Whole;
        }
        match unit.id {
            UnitId::Trunk(layer) => {
                if let Some(e) = routed_expert_of(&want.name) {
                    return if self.owns_layer_expert(layer, e, experts) { Piece::Whole } else { Piece::Skip };
                }
                for l in &unit.linears {
                    let Some(part) = want.name.strip_prefix(l.prefix.as_str()).and_then(|r| r.strip_prefix('.')) else {
                        continue;
                    };
                    let Some(role) = Dense::of_prefix(&l.prefix) else { break };
                    if !self.slices(role) || !sliceable(l.n, self.topo.ranks) {
                        break;
                    }
                    let (lo, hi) = column_range(l.n, self.topo.rank, self.topo.ranks);
                    return match part {
                        "trellis" => Piece::Trellis { k: l.k, n: l.n, lo, hi },
                        "svh" => Piece::Svh { lo, hi },
                        _ => Piece::Whole,
                    };
                }
                Piece::Whole
            }
            // an EXL3 `lm_head` (the MixedK pack's): its trellis and `svh` cut at 128-column blocks, `suh` and the marker whole
            UnitId::LmHead if !self.lm_head_full => match unit.linears.first() {
                Some(l) => {
                    let (lo, hi) = self.lm_head_cols_exl3(l.n);
                    match want.name.strip_prefix(l.prefix.as_str()) {
                        Some(".trellis") => Piece::Trellis { k: l.k, n: l.n, lo, hi },
                        Some(".svh") => Piece::Svh { lo, hi },
                        _ => Piece::Whole,
                    }
                }
                None => {
                    let (lo, hi) = self.lm_head_rows(want.shape[0]);
                    Piece::Rows { lo, hi }
                }
            },
            // the MTP head's bf16 experts: a rank holds its half, the other half is never read
            UnitId::Mtp(_) if self.head_experts => match head_expert_of(&want.name) {
                Some(e) if !self.owns_head_expert(e, experts) => Piece::Skip,
                _ => Piece::Whole,
            },
            _ => Piece::Whole,
        }
    }

    /// The shape `want` has on this rank's device, `None` when the rank does not hold it.
    pub fn local_shape(&self, want: &Want, piece: &Piece) -> Option<Vec<usize>> {
        Some(match *piece {
            Piece::Skip => return None,
            Piece::Whole => want.shape.clone(),
            Piece::Trellis { k, lo, hi, .. } => vec![k / 16, (hi - lo) / 16, want.shape[2]],
            Piece::Svh { lo, hi } => vec![hi - lo],
            Piece::Rows { lo, hi } => vec![hi - lo, want.shape[1]],
        })
    }

    /// The EXL3 linears of `unit` this rank holds, with the width each has here (`n_local`, the plan's `n` for a whole one).
    pub fn local_linears(&self, unit: &Unit, experts: usize) -> Vec<Linear> {
        let mut out = Vec::with_capacity(unit.linears.len());
        for l in &unit.linears {
            if self.is_single() {
                out.push(l.clone());
                continue;
            }
            if let UnitId::Trunk(layer) = unit.id {
                if let Some(e) = routed_expert_of(&l.prefix) {
                    if !self.owns_layer_expert(layer, e, experts) {
                        continue;
                    }
                }
            }
            // a 4-bit head's routed experts: a rank holds the ones it owns, as for the bf16 head (`piece`)
            if let (UnitId::Mtp(_), true) = (unit.id, self.head_experts) {
                if let Some(e) = head_expert_of(&l.prefix) {
                    if !self.owns_head_expert(e, experts) {
                        continue;
                    }
                }
            }
            let n = match (unit.id, Dense::of_prefix(&l.prefix)) {
                (UnitId::Trunk(_), Some(role)) => {
                    let (lo, hi) = self.columns(role, l.n);
                    hi - lo
                }
                (UnitId::LmHead, _) if !self.lm_head_full => {
                    let (lo, hi) = self.lm_head_cols_exl3(l.n);
                    hi - lo
                }
                _ => l.n,
            };
            out.push(Linear { prefix: l.prefix.clone(), k: l.k, n, bits: l.bits });
        }
        out
    }

    /// Bytes `want` takes on this rank's device (`None`: not held), from the plan's shape.
    pub fn local_bytes(&self, want: &Want, piece: &Piece) -> Option<u64> {
        let shape = self.local_shape(want, piece)?;
        let numel: u64 = shape.iter().map(|&d| d as u64).product();
        Some(numel * dtype_size(want.dtype).unwrap_or(0))
    }

    /// Bytes this rank holds of the units `pick` selects (the plan's tensors; `weights::LayerPlan::bytes` is the same sum for one rank,
    /// except a 4-bit head's EXL3 projections, counted as the bridge holds them: [`head_exl3_held_delta`]).
    pub fn rank_bytes(&self, plan: &LayerPlan, experts: usize, pick: impl Fn(UnitId) -> bool) -> u64 {
        self.rank_bytes_with(plan, experts, pick, hg_wanted(self))
    }

    /// [`ShardPlan::rank_bytes`] with the head-group in_proj (`hg_wanted`) counted or not, as given (pure, for the tests).
    pub fn rank_bytes_with(&self, plan: &LayerPlan, experts: usize, pick: impl Fn(UnitId) -> bool, hg: bool) -> u64 {
        let mut total = 0u64;
        for u in plan.units.iter().filter(|u| pick(u.id)) {
            for w in &u.tensors {
                let p = self.piece(u, w, experts);
                let b = self.local_bytes(w, &p).unwrap_or(0);
                // a form our packs do not have (an F16 gate) is held as the bridge holds it (`weights::held_extra_bytes`)
                total += b + weights::held_extra_bytes(w, b);
            }
            // a 4-bit head's EXL3 projections, the ones this rank holds, as the bridge holds them (nothing for a bf16 head: no linears)
            if matches!(u.id, UnitId::Mtp(_)) {
                for l in self.local_linears(u, experts) {
                    total = total.saturating_add_signed(head_exl3_held_delta(&l));
                }
            }
            // contract P2 at two ranks: the head-group in_proj this rank loads beside its slice (`upload_tp`, lane S)
            if hg {
                for l in u.linears.iter().filter(|l| l.prefix.ends_with(".mixer.in_proj") && l.n == splits::IN_PROJ_COLS) {
                    let n_hg: usize = hg_runs(self.topo.rank).iter().map(|(_, n)| n).sum();
                    for w in u.tensors.iter().filter(|w| w.name.starts_with(&format!("{}.", l.prefix))) {
                        let part = &w.name[l.prefix.len() + 1..];
                        let whole = self.local_bytes(w, &Piece::Whole).unwrap_or(0);
                        total += match part {
                            "trellis" => (l.k / 16 * n_hg / 16 * w.shape[2] * 2) as u64,
                            "svh" => (n_hg * 2) as u64,
                            _ => whole,
                        };
                    }
                }
            }
        }
        total
    }

    /// The tensors of the units `pick` selects that this rank loads, in plan order, each with its piece.
    pub fn wants<'a>(
        &'a self,
        plan: &'a LayerPlan,
        experts: usize,
        pick: impl Fn(UnitId) -> bool + 'a,
    ) -> impl Iterator<Item = (&'a Unit, &'a Want, Piece)> + 'a {
        plan.units
            .iter()
            .filter(move |u| pick(u.id))
            .flat_map(move |u| u.tensors.iter().map(move |w| (u, w, self.piece(u, w, experts))))
            .filter(|(_, _, p)| *p != Piece::Skip)
    }
}

/// The routed expert a tensor or linear name belongs to: `...mixer.experts.{e}.{up,down}_proj...` of a trunk layer. The shared expert
/// (`mixer.shared_experts`) and the MTP head's experts (`mtp.layers`) are not routed experts of this split.
fn routed_expert_of(name: &str) -> Option<usize> {
    let (head, tail) = name.split_once(".mixer.experts.")?;
    if !head.contains(".backbone.layers.") {
        return None;
    }
    tail.split('.').next()?.parse().ok()
}

/// The routed expert a tensor of the MTP head belongs to: `...mtp.layers.1.mixer.experts.{e}.{up,down}_proj.weight` (bf16, whole
/// tensors: the head's experts are not EXL3). The head's shared expert (`mixer.shared_experts`) and latent projections are not routed
/// experts of this split.
pub fn head_expert_of(name: &str) -> Option<usize> {
    let (head, tail) = name.split_once(".mixer.experts.")?;
    if !head.contains(".mtp.layers.") {
        return None;
    }
    tail.split('.').next()?.parse().ok()
}

/// What every routed expert of the plan's MoE layers weighs (`tp_balance::ExpertBytes`): per trunk layer, each expert's planned bytes
/// (every tensor of its up and down projections, the shape's product times the dtype's size: the numbers [`ShardPlan::rank_bytes`]
/// counts), and the distinct trellis widths (`shape[2] / 16`) of the layer's EXL3 experts. The input of the byte-weighted balance
/// (`tp_balance::byte_balanced_map`, `solve_weighted`); a plan whose experts are all one width (today's packs) has no mixed layer.
pub fn expert_bytes(plan: &LayerPlan, experts: usize) -> super::tp_balance::ExpertBytes {
    let mut out = super::tp_balance::ExpertBytes { experts, ..Default::default() };
    for u in &plan.units {
        let UnitId::Trunk(layer) = u.id else { continue };
        for w in &u.tensors {
            let Some(e) = routed_expert_of(&w.name) else { continue };
            if e >= experts {
                continue;
            }
            let numel: u64 = w.shape.iter().map(|&d| d as u64).product();
            out.bytes.entry(layer).or_insert_with(|| vec![0; experts])[e] += numel * dtype_size(w.dtype).unwrap_or(0);
            if w.name.ends_with(".trellis") && w.shape.len() == 3 {
                let widths = out.widths.entry(layer).or_default();
                let k = w.shape[2] / 16;
                if let Err(at) = widths.binary_search(&k) {
                    widths.insert(at, k);
                }
            }
        }
    }
    out
}

/// [`weights::upload`] for one rank: the same tensors, each read by its index entry, but a routed expert of another rank is not read
/// and a sliced linear or the `lm_head` is copied in part, straight into a buffer of its own (smaller) size. With
/// [`ShardPlan::single`] it is [`weights::upload`].
///
/// The result is a [`Weights`] whose tensors have their LOCAL shapes (`{prefix}.trellis` `[K/16, n_local/16, 16 * bits]`, `{prefix}.svh`
/// `[n_local]`, `lm_head` `[rows_local, hidden]`) and whose EXL3 linears have `n = n_local`; `bridge::into_device_tp` takes it from there.
pub fn upload_tp(
    dir: &Path,
    inv: &Inventory,
    plan: &LayerPlan,
    experts: usize,
    pick: impl Fn(UnitId) -> bool,
    o: &Options,
    shard: &ShardPlan,
) -> Result<Weights, String> {
    if shard.is_single() {
        return weights::upload(dir, inv, plan, pick, o);
    }
    let arena = Arena::new(ARENA_BLOCK_BYTES);
    let mut loader = Loader::new();
    let mut tensors: Vec<Uploaded> = Vec::new();
    let mut bytes = 0u64;
    for (_, w, piece) in shard.wants(plan, experts, &pick) {
        let t = inv.get(&w.name).ok_or_else(|| format!("`{}` is not in the inventory", w.name))?;
        let file = loader.file(&dir.join(&t.shard));
        let local = shard.local_bytes(w, &piece).expect("a loaded piece has bytes");
        let len = usize::try_from(local).map_err(|_| format!("`{}`: {local} bytes", w.name))?;
        let buf = arena.alloc(len).map_err(|e| format!("`{}`: {e}", w.name))?;
        let err = |e: gb_cuda::loader::LoadError| format!("`{}`: {e}", w.name);
        match piece {
            Piece::Skip => unreachable!("skipped pieces are filtered"),
            Piece::Whole => {
                if local != t.nbytes {
                    return Err(format!("`{}`: the plan holds {local} bytes, the index {}", w.name, t.nbytes));
                }
                loader.copy_file(&buf, file, t.offset, len, Xform::None).map_err(err)?;
            }
            Piece::Trellis { k, n, lo, hi } => {
                // [K/16, N/16, 16 * bits] int16: a K-row is N/16 tiles of `tile` bytes in column order, and the columns [lo, hi) of it are
                // one contiguous run, a row pitch apart
                let tile = w.shape[2] * 2;
                let (pitch, run, at) = (n / 16 * tile, (hi - lo) / 16 * tile, lo / 16 * tile);
                if k / 16 * pitch != t.nbytes as usize || k / 16 * run != len {
                    return Err(format!("`{}`: trellis geometry [{k}, {n}] does not match its {} bytes", w.name, t.nbytes));
                }
                for kk in 0..k / 16 {
                    let dst = buf.subview(kk * run, run);
                    loader.copy_file(&dst, file, t.offset + (kk * pitch + at) as u64, run, Xform::None).map_err(err)?;
                }
            }
            Piece::Svh { lo, hi } => {
                debug_assert_eq!(len, (hi - lo) * 2);
                loader.copy_file(&buf, file, t.offset + (lo * 2) as u64, len, Xform::None).map_err(err)?;
            }
            Piece::Rows { lo, hi } => {
                let row = w.shape[1] * dtype_size(w.dtype).unwrap_or(0) as usize;
                debug_assert_eq!(len, (hi - lo) * row);
                loader.copy_file(&buf, file, t.offset + (lo * row) as u64, len, Xform::None).map_err(err)?;
            }
        }
        bytes += local;
        tensors.push(Uploaded {
            name: w.name.clone(),
            dtype: w.dtype,
            shape: shard.local_shape(w, &piece).expect("a loaded piece has a shape"),
            shard: t.shard.clone(),
            offset: t.offset,
            nbytes: local,
            buf,
        });
    }
    let mut linears = Vec::new();
    for u in plan.units.iter().filter(|u| pick(u.id)) {
        linears.extend(shard.local_linears(u, experts));
    }
    // contract P2 at two ranks (lane S): each Mamba layer's in_proj also as this rank's HEAD-GROUP weight (`{prefix}_hg`), the columns of
    // its 4 B/C groups: z, x of its 64 heads, B and C of its groups, and dt of every head (the dt tile is 128 columns: both ranks keep it).
    // The column runs are cut at multiples of 128 (the EXL3 output Hadamard's block), so every column is the whole matrix's column.
    if hg_wanted(shard) {
        for u in plan.units.iter().filter(|u| pick(u.id)) {
            for l in u.linears.iter().filter(|l| l.prefix.ends_with(".mixer.in_proj") && l.n == splits::IN_PROJ_COLS) {
                let runs = hg_runs(shard.topo.rank);
                let n_hg: usize = runs.iter().map(|(_, n)| n).sum();
                let hg = format!("{}_hg", l.prefix);
                for part in ["trellis", "suh", "svh", "mul1"] {
                    let src = format!("{}.{part}", l.prefix);
                    let w = u.tensors.iter().find(|w| w.name == src).ok_or_else(|| format!("`{src}` is not planned"))?;
                    let t = inv.get(&src).ok_or_else(|| format!("`{src}` is not in the inventory"))?;
                    let file = loader.file(&dir.join(&t.shard));
                    let err = |e: gb_cuda::loader::LoadError| format!("`{src}` (head groups): {e}");
                    let (len, shape) = match part {
                        "trellis" => {
                            let tile = w.shape[2] * 2;
                            (l.k / 16 * n_hg / 16 * tile, vec![l.k / 16, n_hg / 16, w.shape[2]])
                        }
                        "svh" => (n_hg * 2, vec![n_hg]),
                        _ => (t.nbytes as usize, w.shape.clone()),
                    };
                    let buf = arena.alloc(len).map_err(|e| format!("`{hg}.{part}`: {e}"))?;
                    match part {
                        "trellis" => {
                            let tile = w.shape[2] * 2;
                            let pitch = l.n / 16 * tile;
                            let mut at = 0;
                            for kk in 0..l.k / 16 {
                                for &(lo, n) in &runs {
                                    let run = n / 16 * tile;
                                    let dst = buf.subview(at, run);
                                    loader
                                        .copy_file(&dst, file, t.offset + (kk * pitch + lo / 16 * tile) as u64, run, Xform::None)
                                        .map_err(err)?;
                                    at += run;
                                }
                            }
                        }
                        "svh" => {
                            let mut at = 0;
                            for &(lo, n) in &runs {
                                loader
                                    .copy_file(&buf.subview(at, n * 2), file, t.offset + (lo * 2) as u64, n * 2, Xform::None)
                                    .map_err(err)?;
                                at += n * 2;
                            }
                        }
                        _ => loader.copy_file(&buf, file, t.offset, len, Xform::None).map_err(err)?,
                    }
                    bytes += len as u64;
                    tensors.push(Uploaded {
                        name: format!("{hg}.{part}"),
                        dtype: w.dtype,
                        shape,
                        shard: t.shard.clone(),
                        offset: t.offset,
                        nbytes: len as u64,
                        buf,
                    });
                }
                linears.push(Linear { prefix: hg, k: l.k, n: n_hg, bits: l.bits });
            }
        }
    }
    let report = loader.run(o).map_err(|e| format!("loader: {e}"))?;
    if report.bytes != bytes {
        return Err(format!("the loader copied {} bytes, the rank's plan holds {bytes}", report.bytes));
    }
    let mut up = Weights::from_parts(tensors, linears);
    up.report = report;
    up.bytes = bytes;
    Ok(up)
}

/// Whether this rank loads the head-group in_proj (`{prefix}_hg`, lane S): contract P2, two ranks, the in_proj sliced by the plan, and
/// the head split on (`REXL3_NEMO_P2TP_MAMBA_HG`, default `contract::MAMBA_HG_DEFAULT`).
pub fn hg_wanted(shard: &ShardPlan) -> bool {
    shard.topo.ranks == 2
        && super::contract::current() == super::contract::PrefillContract::P2
        && shard.slices(Dense::MambaIn)
        && super::contract::p2tp_on(super::contract::MAMBA_HG_VAR)
}

/// The in_proj column runs (start, count) of rank `rank`'s head groups, in the merged weight's order: z of its heads, x of its heads, B
/// of its groups, C of its groups, dt of every head. (8,192 z | 8,192 x | 1,024 B | 1,024 C | 128 dt; rank r: groups [4 r, 4 r + 4).)
pub fn hg_runs(rank: usize) -> [(usize, usize); 5] {
    let (h, g) = (4096, 512);
    [(h * rank, h), (8192 + h * rank, h), (16384 + g * rank, g), (17408 + g * rank, g), (18432, 128)]
}

/// The pack's `lm_head` rows of the token ids `ids` (ascending, no duplicates), read straight from the shard into a bf16 `[ids.len(),
/// hidden]` device table: the rows a restricted MTP draft head scores (`mtp::SubHead`) without the rank holding the other half of the
/// `lm_head`. Runs of consecutive ids are one file range each; the rows are the file's bytes, so the table is bit-identical to the
/// single node's `copy_range_from` gather.
pub fn read_lm_head_rows(
    dir: &Path,
    inv: &Inventory,
    plan: &LayerPlan,
    ids: &[u32],
    o: &Options,
) -> Result<gb_cuda::DevBuf, String> {
    let unit = plan.unit(UnitId::LmHead).ok_or("the plan has no lm_head unit")?;
    if plan.lm_head_exl3().is_some() {
        return Err(
            "the lm_head is EXL3: its rows are not stored, a restricted table decodes them at load (`mtp_restrict_vocab`)".into(),
        );
    }
    let want = unit.tensors.first().ok_or("the lm_head unit lists no tensor")?;
    let t = inv.get(&want.name).ok_or_else(|| format!("`{}` is not in the inventory", want.name))?;
    let row = want.shape[1] * dtype_size(want.dtype).unwrap_or(0) as usize;
    if ids.is_empty() {
        return Err("no token ids".into());
    }
    if let Some(w) = ids.windows(2).find(|w| w[0] >= w[1]) {
        return Err(format!("the ids must ascend without repeats ({} then {})", w[0], w[1]));
    }
    if let Some(&bad) = ids.iter().find(|&&i| i as usize >= want.shape[0]) {
        return Err(format!("token {bad} is past the {} rows of `{}`", want.shape[0], want.name));
    }
    let arena = Arena::new(ids.len() * row);
    let table = arena.alloc(ids.len() * row).map_err(|e| format!("lm_head rows: {e}"))?;
    let mut loader = Loader::new();
    let file = loader.file(&dir.join(&t.shard));
    let mut at = 0;
    while at < ids.len() {
        let mut end = at + 1;
        while end < ids.len() && ids[end] == ids[end - 1] + 1 {
            end += 1;
        }
        let dst = table.subview(at * row, (end - at) * row);
        loader
            .copy_file(&dst, file, t.offset + ids[at] as u64 * row as u64, (end - at) * row, Xform::None)
            .map_err(|e| format!("lm_head rows: {e}"))?;
        at = end;
    }
    let report = loader.run(o).map_err(|e| format!("loader: {e}"))?;
    if report.bytes != (ids.len() * row) as u64 {
        return Err(format!("the loader copied {} bytes of {}", report.bytes, ids.len() * row));
    }
    Ok(table)
}

/// Loads trunk layers `0..layers` and the globals of the pack onto this rank (and the MTP head when `head`): what
/// `NemotronModel::load` does between `weights::upload` and the model's constructor, for one rank. The caller constructs the model.
/// With [`ShardPlan::single`] it is that code path exactly.
#[allow(clippy::too_many_arguments)]
pub fn load_weights(
    dir: &Path,
    cfg: &OmniConfig,
    inv: &Inventory,
    plan: &LayerPlan,
    layers: usize,
    head: bool,
    shard: &ShardPlan,
    o: &Options,
) -> Result<NemotronWeights, ModelError> {
    let pick = |u: UnitId| match u {
        UnitId::Trunk(i) => i < layers,
        UnitId::Embed | UnitId::NormF | UnitId::LmHead => true,
        UnitId::Mtp(_) => head,
    };
    let mut up = upload_tp(dir, inv, plan, cfg.experts, pick, o, shard)
        .map_err(|e| ModelError::Config(format!("nemotron upload (rank {} of {}): {e}", shard.topo.rank, shard.topo.ranks)))?;
    let st = Stream::new_nonblocking()?;
    let mut w = super::bridge::into_device_tp(&mut up, cfg, plan, 0..layers, &st, shard)?;
    if head {
        w.mtp = super::bridge::take_mtp_tp(&mut up, cfg, plan, &st, shard)?;
    }
    st.sync()?;
    Ok(w)
}

/// What the device holds of a 4-bit head's EXL3 projection `l` beyond (or short of) its four tensors in the index: the bridge
/// (`bridge::take_mtp_tp`) drops the 4-byte `mul1` codebook marker after checking it, and the MoE block's projections (block 1) take `suh`
/// and `svh` widened from fp16 to f32 (`Proj::Exl3`, 2 more bytes per entry: `2 (k + n)`), while block 0's (`eh_proj` and an EXL3
/// attention's q, k, v, o) keep them fp16 (`Exl3Weights`). `mtp::weight_bytes`, the head's `mtp_weights` allocation, counts exactly that.
/// The single-node load test applies the same rule.
pub fn head_exl3_held_delta(l: &Linear) -> i64 {
    let widened = if l.prefix.contains(".mtp.layers.0.") { 0 } else { 2 * (l.k + l.n) as i64 };
    widened - 4
}

/// Inventory-derived resident bytes of a rank, for the memory plan: `(everything this rank holds, of which the MTP head)`. The same
/// pair `engine::resident_weight_bytes` takes for one node.
pub fn rank_listed_bytes(shard: &ShardPlan, plan: &LayerPlan, cfg: &OmniConfig) -> (u64, u64) {
    let all = shard.rank_bytes(plan, cfg.experts, |_| true);
    let head = shard.rank_bytes(plan, cfg.experts, |u| matches!(u, UnitId::Mtp(_)));
    (all, head)
}

/// The Nemotron plan as the family-neutral description of a shard plan (`crate::shard`): the role of a matrix is [`Dense::name`], and
/// the answer is [`ShardPlan::columns`]. The cut rule, the proof ([`slice_keeps_plan`]) and the plan's decisions are unchanged.
impl crate::shard::ShardPlan for ShardPlan {
    fn topology(&self) -> (usize, usize) {
        (self.topo.rank, self.topo.ranks)
    }

    fn cut(&self, m: &crate::shard::MatrixSpec) -> crate::shard::Cut {
        use crate::shard::Cut;
        let Some(role) = Dense::ALL.into_iter().find(|d| d.name() == m.role) else { return Cut::Whole };
        let (lo, hi) = self.columns(role, m.n);
        if (lo, hi) == (0, m.n) {
            Cut::Whole
        } else {
            Cut::Cols { lo, n_local: hi - lo }
        }
    }
}

/// The dense matrices of the trunk as the neutral description's specs (name and role are [`Dense::name`]; one spec stands for every
/// layer of its role, and for every bit width: the kernel's plan depends on the shape alone).
pub fn matrix_specs(cfg: &OmniConfig) -> Vec<crate::shard::MatrixSpec> {
    dense_shapes(cfg).into_iter().map(|(role, k, n)| crate::shard::MatrixSpec::new(role.name(), role.name(), k, n, 0)).collect()
}

#[cfg(test)]
mod tests {
    use super::super::tp::splits;
    use super::super::weights::{units, Storage, Storages};
    use super::*;

    const CONFIG: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");
    const BF16_PAIRS: [(usize, usize); 3] = [(26, 111), (39, 304), (81, 461)];

    #[test]
    fn the_plan_is_a_neutral_shard_plan_whose_proof_table_and_audit_agree_with_it_at_every_rank_count() {
        use crate::shard::{audit, prove_shape, Cut, ShardPlan as _};
        let c = cfg();
        let specs = matrix_specs(&c);
        assert_eq!(specs.len(), 6);
        assert_eq!((specs[0].role.as_str(), specs[0].k, specs[0].n), ("mamba in_proj", 4096, 18560));
        for ranks in 1..=8usize {
            for rank in 0..ranks {
                let topo = TpTopology { rank, ranks };
                let serving = ShardPlan::for_config(topo, &c);
                // the serving plan passes the neutral audit, and its cuts are exactly the proof's exact shapes (replication for cost is
                // the plan's choice, and the serving plan makes none)
                assert_eq!(audit(&serving, &specs), Vec::<String>::new(), "R = {ranks} rank {rank}");
                for m in &specs {
                    let proof = prove_shape(m, ranks);
                    let cut = serving.cut(m);
                    assert_eq!(matches!(cut, Cut::Cols { .. }), ranks > 1 && proof.is_exact(), "{} at {ranks}", m.name);
                    if let Cut::Cols { lo, n_local } = cut {
                        assert_eq!((lo, lo + n_local), column_range(m.n, rank, ranks));
                        assert_eq!(proof.slices[rank].0, n_local);
                    }
                }
            }
        }
        // the test plan that slices every matrix is caught where the proof says the bits would differ
        let all = ShardPlan::slice_everything(TpTopology { rank: 0, ranks: 2 });
        let v = audit(&all, &specs);
        assert_eq!(v.len(), 1, "{v:?}");
        assert!(v[0].contains("mamba out_proj"), "{v:?}");
        let v = audit(&ShardPlan::slice_everything(TpTopology { rank: 2, ranks: 4 }), &specs);
        assert!(v.iter().any(|s| s.contains("mamba in_proj")) && v.iter().any(|s| s.contains("mamba out_proj")), "{v:?}");
        // one rank: nothing is cut
        assert!(specs.iter().all(|m| ShardPlan::single().cut(m) == Cut::Whole));
    }

    fn cfg() -> OmniConfig {
        OmniConfig::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    fn plan(cfg: &OmniConfig) -> LayerPlan {
        let mut s = Storages::new();
        for (l, e) in BF16_PAIRS {
            for p in ["up_proj", "down_proj"] {
                s.insert((l, e, p), Storage::Bf16);
            }
        }
        LayerPlan { units: units(cfg, &s), outside_plan: 0 }
    }

    fn ranks() -> [ShardPlan; 2] {
        let c = cfg();
        [0, 1].map(|rank| ShardPlan::for_config(TpTopology { rank, ranks: 2 }, &c))
    }

    #[test]
    fn the_column_cuts_are_the_seams_splits() {
        assert_eq!(column_range(splits::IN_PROJ_COLS, 0, 2), (0, 9216));
        assert_eq!(column_range(splits::IN_PROJ_COLS, 1, 2), (9216, splits::IN_PROJ_COLS));
        assert_eq!(splits::in_proj_cut(2), 9216, "derived from the one cut rule: the constant it replaces");
        for (n, half) in [(splits::HIDDEN, 2048), (splits::Q_COLS, 2048), (splits::O_COLS, 2048), (splits::KV_COLS, 128)] {
            assert_eq!(column_range(n, 0, 2), (0, half));
            assert_eq!(column_range(n, 1, 2), (half, n));
        }
        // the single rank holds every column, and the slices of any width tile it exactly
        assert_eq!(column_range(4096, 0, 1), (0, 4096));
        for ranks in [2usize, 3, 5] {
            let mut at = 0;
            for r in 0..ranks {
                let (lo, hi) = column_range(2560, r, ranks);
                assert_eq!(lo, at);
                assert_eq!((lo % 128, hi % 128), (0, 0));
                at = hi;
            }
            assert_eq!(at, 2560);
        }
    }

    #[test]
    fn decode_plan_is_the_kernels_for_every_shape_of_the_pack() {
        // the values the kernel's plan_for gives (exl3_linear.cu): the dense linears, whole
        assert_eq!(decode_plan(4096, 18560), (1, 8), "in_proj");
        assert_eq!(decode_plan(8192, 4096), (8, 4), "out_proj");
        assert_eq!(decode_plan(4096, 4096), (8, 4), "q, o");
        assert_eq!(decode_plan(4096, 256), (8, 4), "k, v");
    }

    #[test]
    fn only_the_mamba_out_proj_changes_its_plan_when_cut_and_so_only_it_is_replicated() {
        let c = cfg();
        let shapes = dense_shapes(&c);
        assert_eq!(shapes[0], (Dense::MambaIn, 4096, 18560));
        assert_eq!(shapes[1], (Dense::MambaOut, 8192, 4096));
        assert_eq!(shapes[2], (Dense::AttnQ, 4096, 4096));
        assert_eq!(shapes[3], (Dense::AttnK, 4096, 256));
        assert_eq!(shapes[5], (Dense::AttnO, 4096, 4096));
        // in_proj: 72 and 73 tiles are both at least 64, the plan stays (1, 8)
        assert_eq!((decode_plan(4096, 9216), decode_plan(4096, 9344)), ((1, 8), (1, 8)));
        // out_proj: 2048 columns runs 16 K splits, the whole matrix 8
        assert_eq!(decode_plan(8192, 2048), (16, 4));
        for (role, k, n) in shapes {
            assert_eq!(slice_keeps_plan(k, n, 2), role != Dense::MambaOut, "{}", role.name());
        }
        for p in ranks() {
            assert_eq!(p.replicate, BTreeSet::from([Dense::MambaOut]));
            assert!(p.slices(Dense::MambaIn) && p.slices(Dense::AttnO) && !p.slices(Dense::MambaOut));
        }
        assert!(ShardPlan::for_config(TpTopology::SINGLE, &c).replicate.is_empty());
        assert!(ShardPlan::single().replicate.is_empty() && !ShardPlan::single().slices(Dense::MambaIn));
    }

    #[test]
    fn the_cost_model_slices_what_pays_for_its_collective_and_replicates_the_rest() {
        let c = cfg();
        let topo = TpTopology { rank: 0, ranks: 2 };
        // 38 MB of in_proj: half of it is 70 us of read at 273 GB/s
        assert_eq!(dense_bytes(4096, 18560), 38_010_880 + 45_312);
        let free = ShardPlan::for_cost(topo, &c, 0.0, 273.0);
        assert_eq!(free.replicate, BTreeSet::from([Dense::MambaOut]), "a free collective slices everything exact");
        let at30 = ShardPlan::for_cost(topo, &c, 30.0, 273.0);
        assert!(at30.slices(Dense::MambaIn), "in_proj saves 70 us of read for a 30 us collective");
        assert!(!at30.slices(Dense::MambaOut));
        assert!(
            !at30.slices(Dense::AttnQ) && !at30.slices(Dense::AttnO),
            "q+k+v save about 17 us, o about 15 us: not worth a collective"
        );
        let at100 = ShardPlan::for_cost(topo, &c, 100.0, 273.0);
        assert!(Dense::ALL.iter().all(|&d| !at100.slices(d)), "above 70 us nothing is worth slicing");
        assert!(ShardPlan::for_cost(TpTopology::SINGLE, &c, 30.0, 273.0).replicate.is_empty());
    }

    #[test]
    fn replicating_a_matrix_holds_it_whole_on_both_ranks_and_changes_nothing_else() {
        let c = cfg();
        let p = plan(&c);
        let [r0, r1] = ranks();
        let all = ShardPlan::for_config(TpTopology { rank: 1, ranks: 2 }, &c).replicating(&Dense::ALL);
        assert!(Dense::ALL.iter().all(|&d| !all.slices(d) && all.columns(d, 4096) == (0, 4096)));
        let l7 = p.unit(UnitId::Trunk(7)).unwrap();
        for w in &l7.tensors {
            assert_eq!(all.piece(l7, w, c.experts), Piece::Whole, "{}", w.name);
        }
        // the experts and the lm_head are still cut, and the bytes only grow
        assert!(all.owns_expert(300, 512) && !all.owns_expert(5, 512));
        let (a, b) = (all.rank_bytes(&p, c.experts, |_| true), r1.rank_bytes(&p, c.experts, |_| true));
        assert!(a > b);
        let attn_and_in_proj_halves = a - b;
        assert!(attn_and_in_proj_halves < 1 << 30, "{attn_and_in_proj_halves}");
        let _ = r0;
    }

    #[test]
    fn expert_and_row_ranges_halve_the_layer_and_the_vocabulary() {
        let [a, b] = ranks();
        assert_eq!(a.experts(512), 0..256);
        assert_eq!(b.experts(512), 256..512);
        assert!(a.owns_expert(255, 512) && !a.owns_expert(256, 512) && b.owns_expert(256, 512));
        assert_eq!(a.lm_head_rows(splits::VOCAB), (0, 65536));
        assert_eq!(b.lm_head_rows(splits::VOCAB), (65536, splits::VOCAB));
        assert_eq!(ShardPlan::single().experts(512), 0..512);
        assert_eq!(a.columns(Dense::MambaIn, 18560), (0, 9216));
        assert_eq!(b.columns(Dense::MambaIn, 18560), (9216, 18560));
        assert_eq!(b.columns(Dense::MambaOut, 4096), (0, 4096), "replicated: the whole width");
    }

    #[test]
    fn the_heads_experts_are_named_by_their_own_rule() {
        let h = "language_model.mtp.layers.1.mixer";
        assert_eq!(head_expert_of(&format!("{h}.experts.300.up_proj.weight")), Some(300));
        assert_eq!(head_expert_of(&format!("{h}.experts.0.down_proj.weight")), Some(0));
        assert_eq!(head_expert_of(&format!("{h}.shared_experts.up_proj.weight")), None);
        assert_eq!(head_expert_of(&format!("{h}.fc1_latent_proj.weight")), None);
        assert_eq!(head_expert_of(&format!("{h}.gate.weight")), None);
        // a trunk layer's expert is not the head's
        assert_eq!(head_expert_of("language_model.backbone.layers.1.mixer.experts.300.up_proj.trellis"), None);
    }

    #[test]
    fn routed_experts_are_told_apart_from_the_shared_expert_and_the_head() {
        let b = "language_model.backbone.layers.1.mixer";
        assert_eq!(routed_expert_of(&format!("{b}.experts.300.up_proj.trellis")), Some(300));
        assert_eq!(routed_expert_of(&format!("{b}.experts.0.down_proj")), Some(0));
        assert_eq!(routed_expert_of(&format!("{b}.shared_experts.up_proj.trellis")), None);
        assert_eq!(routed_expert_of(&format!("{b}.fc1_latent_proj.trellis")), None);
        assert_eq!(routed_expert_of("language_model.mtp.layers.1.mixer.experts.300.up_proj.weight"), None);
    }

    #[test]
    fn a_rank_loads_its_experts_its_columns_and_its_rows_and_nothing_else_changes() {
        let c = cfg();
        let p = plan(&c);
        let [r0, r1] = ranks();
        let single = ShardPlan::single();
        let all = |s: &ShardPlan| s.rank_bytes(&p, c.experts, |_| true);
        let whole: u64 = p
            .units
            .iter()
            .flat_map(|u| u.tensors.iter())
            .map(|w| w.shape.iter().product::<usize>() as u64 * dtype_size(w.dtype).unwrap())
            .sum();
        assert_eq!(all(&single), whole, "the single rank holds every planned byte");
        // per-tensor dispositions on layer 0 (Mamba) and layer 7 (attention) and layer 1 (MoE)
        let l0 = p.unit(UnitId::Trunk(0)).unwrap();
        let tr = l0.tensors.iter().find(|w| w.name.ends_with("mixer.in_proj.trellis")).unwrap();
        assert_eq!(r0.piece(l0, tr, c.experts), Piece::Trellis { k: 4096, n: 18560, lo: 0, hi: 9216 });
        assert_eq!(r1.piece(l0, tr, c.experts), Piece::Trellis { k: 4096, n: 18560, lo: 9216, hi: 18560 });
        assert_eq!(r1.local_shape(tr, &r1.piece(l0, tr, c.experts)), Some(vec![256, 9344 / 16, 64]));
        let svh = l0.tensors.iter().find(|w| w.name.ends_with("mixer.in_proj.svh")).unwrap();
        assert_eq!(r1.piece(l0, svh, c.experts), Piece::Svh { lo: 9216, hi: 18560 });
        let suh = l0.tensors.iter().find(|w| w.name.ends_with("mixer.in_proj.suh")).unwrap();
        assert_eq!(r1.piece(l0, suh, c.experts), Piece::Whole, "the input scale stays whole");
        let outp = l0.tensors.iter().find(|w| w.name.ends_with("mixer.out_proj.trellis")).unwrap();
        assert_eq!(r1.piece(l0, outp, c.experts), Piece::Whole, "out_proj is replicated");
        assert_eq!(r1.piece(l0, outp, c.experts), single.piece(l0, outp, c.experts));
        let conv = l0.tensors.iter().find(|w| w.name.ends_with("conv1d.weight")).unwrap();
        assert_eq!(r0.piece(l0, conv, c.experts), Piece::Whole);
        let l1 = p.unit(UnitId::Trunk(1)).unwrap();
        let e300 = l1.tensors.iter().find(|w| w.name.ends_with("experts.300.up_proj.trellis")).unwrap();
        assert_eq!((r0.piece(l1, e300, c.experts), r1.piece(l1, e300, c.experts)), (Piece::Skip, Piece::Whole));
        let e5 = l1.tensors.iter().find(|w| w.name.ends_with("experts.5.down_proj.svh")).unwrap();
        assert_eq!((r0.piece(l1, e5, c.experts), r1.piece(l1, e5, c.experts)), (Piece::Whole, Piece::Skip));
        for keep in ["mixer.gate.weight", "fc1_latent_proj.trellis", "fc2_latent_proj.suh", "shared_experts.up_proj.trellis"] {
            let w = l1.tensors.iter().find(|w| w.name.ends_with(keep)).unwrap();
            assert_eq!((r0.piece(l1, w, c.experts), r1.piece(l1, w, c.experts)), (Piece::Whole, Piece::Whole), "{keep}");
        }
        let head = p.unit(UnitId::LmHead).unwrap();
        let hw = &head.tensors[0];
        assert_eq!(r0.piece(head, hw, c.experts), Piece::Rows { lo: 0, hi: 65536 });
        assert_eq!(r1.piece(head, hw, c.experts), Piece::Rows { lo: 65536, hi: 131072 });
        assert_eq!(r1.local_bytes(hw, &r1.piece(head, hw, c.experts)), Some(65536 * 4096 * 2));
        let embed = p.unit(UnitId::Embed).unwrap();
        assert_eq!(r0.piece(embed, &embed.tensors[0], c.experts), Piece::Whole);
        // the MTP head: the routed experts are split (rank 0 holds [0, 256), rank 1 [256, 512)), everything else of it is whole
        let mtp = p.unit(UnitId::Mtp(1)).unwrap();
        for w in &mtp.tensors {
            let want = match head_expert_of(&w.name) {
                Some(e) => (if e < 256 { Piece::Whole } else { Piece::Skip }, if e < 256 { Piece::Skip } else { Piece::Whole }),
                None => (Piece::Whole, Piece::Whole),
            };
            assert_eq!((r0.piece(mtp, w, c.experts), r1.piece(mtp, w, c.experts)), want, "{}", w.name);
        }
        // replicated head (`REXL3_TP_HEAD=replicate`): every tensor whole on both ranks
        let rep = ShardPlan::for_config(TpTopology { rank: 1, ranks: 2 }, &c).replicating_head();
        assert!(mtp.tensors.iter().all(|w| rep.piece(mtp, w, c.experts) == Piece::Whole));
        // linears: absent experts are not linears of the rank, a sliced one reports its local width
        let l0_lin = r1.local_linears(l0, c.experts);
        assert_eq!(l0_lin.iter().map(|l| l.n).collect::<Vec<_>>(), vec![9344, 4096]);
        assert_eq!(r0.local_linears(l1, c.experts).len() + r1.local_linears(l1, c.experts).len(), l1.linears.len() + 4);
        let l7 = p.unit(UnitId::Trunk(7)).unwrap();
        assert_eq!(r0.local_linears(l7, c.experts).iter().map(|l| l.n).collect::<Vec<_>>(), vec![2048, 128, 128, 2048]);
        // bytes: the two ranks together hold the whole pack plus the replicated part once more
        let (b0, b1) = (all(&r0), all(&r1));
        assert!(b0 < whole && b1 < whole);
        let replicated = r0.rank_bytes(&p, c.experts, |_| true) + r1.rank_bytes(&p, c.experts, |_| true) - whole;
        // the replicated bytes are the embedding, the MTP head, the norms and small tensors, the out_proj, the MoE dense parts and the
        // input scales: well under the whole pack, and the experts and the sliced matrices are not in it
        assert!(replicated < whole / 2, "{replicated} of {whole}");
        let experts_bytes: u64 = p
            .units
            .iter()
            .filter(|u| matches!(u.id, UnitId::Trunk(_)))
            .flat_map(|u| u.tensors.iter())
            .filter(|w| routed_expert_of(&w.name).is_some())
            .map(|w| w.shape.iter().product::<usize>() as u64 * dtype_size(w.dtype).unwrap())
            .sum();
        // each rank holds exactly half of the routed experts (the bf16 pairs fall in one half or the other), so a rank is
        // at most `whole - experts/2 + a margin` and at least that minus the sliced matrices
        assert!(b0.max(b1) <= whole - experts_bytes / 2 + (64 << 20), "{b0} {b1} whole {whole} experts {experts_bytes}");
    }

    /// What the head-group in_proj adds to a rank of two (`rank_bytes_with(.., true)` minus `(.., false)`), measured on this config.
    const HG_EXTRA_RANK0: u64 = 766_535_840;
    const HG_EXTRA_RANK1: u64 = 766_535_840;

    /// What the two-rank plan held BEFORE the sharding was made rank-generic (`921d8d5`, captured with the committed config and the
    /// three bf16 expert pairs of `plan`): every number of the verified two-node runs follows from these, so none may move.
    #[test]
    fn the_two_rank_plan_holds_exactly_the_bytes_it_held_before_the_generalisation() {
        let c = cfg();
        let p = plan(&c);
        let pins: [(u64, u64, u64); 2] =
            [(35_746_806_328, 3_066_079_232, 38_565_378_616), (35_765_545_008, 3_066_079_232, 38_584_117_296)];
        // the pins are the plan without the head-group in_proj (contract P1, or the head split off); with it (P2 at two ranks, the
        // default since rc2d) a rank also holds its 9,344 in_proj columns of every Mamba layer beside its slice
        let hg_extra: [u64; 2] = [HG_EXTRA_RANK0, HG_EXTRA_RANK1];
        for (r, (all, head, all_rep)) in pins.into_iter().enumerate() {
            let sp = ShardPlan::for_config(TpTopology { rank: r, ranks: 2 }, &c);
            assert_eq!(sp.rank_bytes_with(&p, c.experts, |_| true, false), all, "rank {r}");
            assert_eq!(
                sp.rank_bytes_with(&p, c.experts, |_| true, true),
                all + hg_extra[r],
                "rank {r}: with the head-group in_proj"
            );
            let now = if hg_wanted(&sp) { all + hg_extra[r] } else { all };
            assert_eq!(sp.rank_bytes(&p, c.experts, |_| true), now, "rank {r}");
            assert_eq!(sp.rank_bytes(&p, c.experts, |u| matches!(u, UnitId::Mtp(_))), head, "rank {r}: the head");
            assert_eq!(
                sp.clone().replicating_head().rank_bytes_with(&p, c.experts, |_| true, false),
                all_rep,
                "rank {r}: replicated head"
            );
            assert_eq!(sp.lm_head_rows(c.vocab), if r == 0 { (0, 65_536) } else { (65_536, 131_072) });
            assert_eq!(sp.experts(c.experts), if r == 0 { 0..256 } else { 256..512 });
            assert_eq!(sp.columns(Dense::MambaIn, 18560), if r == 0 { (0, 9216) } else { (9216, 18560) });
        }
        assert_eq!(ShardPlan::single().rank_bytes(&p, c.experts, |_| true), 68_294_164_008);
    }

    #[test]
    fn every_column_expert_and_head_row_is_held_by_exactly_one_rank_at_any_rank_count() {
        let c = cfg();
        for ranks in [2usize, 3, 4, 5, 8] {
            let plans: Vec<ShardPlan> = (0..ranks).map(|rank| ShardPlan::slice_everything(TpTopology { rank, ranks })).collect();
            // dense matrices, every one sliced (the exactness test's plan): contiguous, tile aligned, skew at most a tile; a matrix with
            // fewer tiles than ranks is whole on every rank
            for (role, _k, n) in dense_shapes(&c) {
                let spans: Vec<(usize, usize)> = plans.iter().map(|p| p.columns(role, n)).collect();
                if sliceable(n, ranks) {
                    let mut at = 0;
                    for (r, &(lo, hi)) in spans.iter().enumerate() {
                        assert_eq!(lo, at, "{} at {ranks} ranks, rank {r}: contiguous", role.name());
                        assert!(hi > lo, "{} at {ranks} ranks, rank {r}: a slice has a tile", role.name());
                        assert_eq!((lo % TILE, hi % TILE), (0, 0), "{}: 128-column aligned", role.name());
                        at = hi;
                    }
                    assert_eq!(at, n, "{} at {ranks} ranks: every column once", role.name());
                    let widths: Vec<usize> = spans.iter().map(|&(lo, hi)| hi - lo).collect();
                    let skew = widths.iter().max().unwrap() - widths.iter().min().unwrap();
                    assert!(skew <= TILE, "{} at {ranks} ranks: skew {skew}", role.name());
                } else {
                    assert!(spans.iter().all(|&s| s == (0, n)), "{} at {ranks} ranks is held whole", role.name());
                }
            }
            // routed experts: every expert on exactly one rank, skew at most one expert
            for experts in [c.experts, 256, 512, 7] {
                let mut count = vec![0usize; experts];
                let mut sizes = Vec::new();
                for p in &plans {
                    let range = p.experts(experts);
                    sizes.push(range.len());
                    for e in range.clone() {
                        count[e] += 1;
                        assert!(p.owns_expert(e, experts));
                    }
                }
                assert!(count.iter().all(|&n| n == 1), "{experts} experts at {ranks} ranks: each once");
                assert!(
                    sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1,
                    "{experts} experts at {ranks}: {sizes:?}"
                );
            }
            // the head's rows (the trunk's lm_head and restricted tables of other sizes): every row once, boundaries in units of 4,
            // skew within two units, and every width within the logits temp the plan allocates
            for total in [c.vocab, 49_152, 62_017, 5_000, 131_071] {
                let mut at = 0;
                let mut widths = Vec::new();
                for p in &plans {
                    let (lo, hi) = head_row_cut(total, p.topo.rank, ranks);
                    assert_eq!(lo, at, "{total} rows at {ranks} ranks: contiguous");
                    if hi < total {
                        assert_eq!(
                            hi % HEAD_ROW_UNIT,
                            0,
                            "{total} rows at {ranks} ranks: interior boundary {hi} is a multiple of 4"
                        );
                    }
                    widths.push(hi - lo);
                    at = hi;
                }
                assert_eq!(at, total, "{total} rows at {ranks} ranks: every row once");
                let skew = widths.iter().max().unwrap() - widths.iter().min().unwrap();
                assert!(skew <= 2 * HEAD_ROW_UNIT, "{total} rows at {ranks} ranks: skew {skew}: {widths:?}");
                assert_eq!(*widths.iter().max().unwrap(), head_row_cut_max(total, ranks));
                assert!(
                    head_row_cut_max(total, ranks) <= head_temp_rows(c.vocab, ranks),
                    "{total} rows at {ranks} ranks: a rank's share fits the logits temp of the vocabulary"
                );
            }
            // ShardPlan::lm_head_rows is the same function
            for p in &plans {
                assert_eq!(p.lm_head_rows(c.vocab), head_row_cut(c.vocab, p.topo.rank, ranks));
            }
        }
    }

    #[test]
    fn the_head_row_cut_is_the_two_halves_at_two_ranks_and_whole_units_of_four_at_three() {
        assert_eq!(head_row_cut(131_072, 0, 2), (0, 65_536));
        assert_eq!(head_row_cut(131_072, 1, 2), (65_536, 131_072));
        assert_eq!(head_row_cut(49_152, 0, 2), (0, 24_576));
        assert_eq!(head_row_cut(49_152, 1, 2), (24_576, 49_152));
        assert_eq!(
            (0..3).map(|r| head_row_cut(131_072, r, 3)).collect::<Vec<_>>(),
            vec![(0, 43_692), (43_692, 87_384), (87_384, 131_072)]
        );
        assert_eq!(
            (0..3).map(|r| head_row_cut(49_152, r, 3)).collect::<Vec<_>>(),
            vec![(0, 16_384), (16_384, 32_768), (32_768, 49_152)]
        );
        // the logits temp: 65,536 rows at two ranks (the pinned figure), 43,692 at three
        assert_eq!(
            (head_temp_rows(131_072, 2), head_temp_rows(131_072, 3), head_temp_rows(131_072, 8)),
            (65_536, 43_692, 16_384)
        );
        // a table too small to be worth a collective per rank is scored whole
        assert!(head_splits(131_072, 8) && head_splits(49_152, 2) && head_splits(256, 2));
        assert!(!head_splits(255, 2) && !head_splits(1_000, 8) && !head_splits(131_072, 1));
    }

    #[test]
    fn the_cut_rule_is_the_engines_floor_rule_of_the_exl3_crate_and_not_its_uneven_one() {
        use gb_exl3::tp::{split_width, SplitMode};
        for ranks in [2usize, 3, 4, 5, 8] {
            for n in [128usize * 2, 128 * 3, 128 * 9, 2048, 4096, 18560, 10240, 131_072] {
                if n / TILE < ranks {
                    continue;
                }
                for rank in 0..ranks {
                    let floor = split_width(n, rank, ranks, SplitMode::Floor);
                    assert_eq!(floor.cols(), column_range(n, rank, ranks), "{n} at {rank}/{ranks}");
                    assert_eq!(floor.width, column_range(n, rank, ranks).1 - column_range(n, rank, ranks).0);
                }
            }
        }
        // the two exl3 rules are mirror images at two ranks and an odd tile count: `Uneven` gives the odd tile to rank 0, the engine to
        // rank 1. A loader built on `Uneven` would misplace every slice of the verified two-rank engine.
        assert_eq!(column_range(18560, 0, 2), (0, 9216));
        assert_eq!(split_width(18560, 0, 2, SplitMode::Uneven).cols(), (0, 9344));
        assert_ne!(split_width(18560, 0, 2, SplitMode::Uneven).cols(), column_range(18560, 0, 2));
        // and they agree whenever the tile count divides
        for rank in 0..2 {
            assert_eq!(split_width(4096, rank, 2, SplitMode::Uneven).cols(), column_range(4096, rank, 2));
        }
        // the tile of the two crates is the same number
        assert_eq!(TILE, gb_exl3::tp::TILE);
    }

    #[test]
    fn in_proj_is_sliced_at_two_ranks_and_replicated_at_three_and_more() {
        let c = cfg();
        // two ranks: only out_proj is replicated (the verified plan)
        for rank in 0..2 {
            let p = ShardPlan::for_config(TpTopology { rank, ranks: 2 }, &c);
            assert_eq!(p.replicate, BTreeSet::from([Dense::MambaOut]));
            assert!(p.slices(Dense::MambaIn));
            assert_eq!(p.columns(Dense::MambaIn, 18560).1 - p.columns(Dense::MambaIn, 18560).0, [9216, 9344][rank]);
        }
        // three and more: a slice of the 145 tiles is below 64 tiles, the stock kernel's K reduction (1 split, 8 warps) becomes
        // (4, 4) or (8, 4), so the slice is not bit-exact and the plan replicates the matrix; k and v (2 tiles) have no tile for
        // every rank; q and o keep their plan to eight ranks
        for ranks in [3usize, 4, 5, 8] {
            for rank in 0..ranks {
                let p = ShardPlan::for_config(TpTopology { rank, ranks }, &c);
                assert_eq!(
                    p.replicate,
                    BTreeSet::from([Dense::MambaIn, Dense::MambaOut, Dense::AttnK, Dense::AttnV]),
                    "{ranks} ranks"
                );
                assert!(!p.slices(Dense::MambaIn) && p.slices(Dense::AttnQ) && p.slices(Dense::AttnO));
                assert_eq!(p.columns(Dense::MambaIn, 18560), (0, 18560), "the whole matrix on every rank");
                let pl = plan(&c);
                let l0 = pl.unit(UnitId::Trunk(0)).unwrap();
                let tr = l0.tensors.iter().find(|w| w.name.ends_with("mixer.in_proj.trellis")).unwrap();
                assert_eq!(p.piece(l0, tr, c.experts), Piece::Whole, "{ranks} ranks, rank {rank}: in_proj is read whole");
            }
            let reasons: Vec<(Dense, Option<String>)> =
                dense_shapes(&c).into_iter().map(|(role, k, n)| (role, replication_reason(k, n, ranks))).collect();
            let why = |d: Dense| reasons.iter().find(|r| r.0 == d).unwrap().1.clone();
            assert!(why(Dense::MambaIn).unwrap().contains("K reduction moves"), "{ranks}: {:?}", why(Dense::MambaIn));
            assert!(why(Dense::MambaOut).unwrap().contains("K reduction moves"));
            assert!(why(Dense::AttnK).unwrap().contains("no column"), "{ranks}: {:?}", why(Dense::AttnK));
            assert_eq!((why(Dense::AttnQ), why(Dense::AttnO)), (None, None));
        }
        // the decisions the boot logs: widths of the sliced, reasons of the replicated
        let d = ShardPlan::for_config(TpTopology { rank: 0, ranks: 3 }, &c).decisions(&c);
        assert_eq!(d[0].0, Dense::MambaIn);
        assert!(d[0].3.is_err());
        assert_eq!(d[2].3, Ok(vec![1280, 1408, 1408]), "q at three ranks: 10 | 11 | 11 tiles");
        let d2 = ShardPlan::for_config(TpTopology { rank: 0, ranks: 2 }, &c).decisions(&c);
        assert_eq!(d2[0].3, Ok(vec![9216, 9344]));
    }

    /// The slice-keeps-plan mirror for every dense shape of the pack at each rank count's widths. The device test
    /// (`nemotron_omni_tp_weights.rs`) holds the mirror against the kernel's own plan (`big_plan`) and the slices against the whole
    /// matrices byte for byte; this is the table it is expected to find.
    #[test]
    fn the_exactness_table_by_shape_and_rank_count() {
        let c = cfg();
        let table: Vec<(Dense, Vec<(usize, bool)>)> = dense_shapes(&c)
            .into_iter()
            .map(|(role, k, n)| (role, [2usize, 3, 4, 5, 8].into_iter().map(|r| (r, slice_keeps_plan(k, n, r))).collect()))
            .collect();
        for (role, row) in &table {
            let line: Vec<String> =
                row.iter().map(|(r, ok)| format!("R={r}: {}", if *ok { "exact" } else { "NOT exact / not sliced" })).collect();
            eprintln!("{:<15} {}", role.name(), line.join("  "));
            let want: Vec<bool> = match role {
                Dense::MambaIn => vec![true, false, false, false, false],
                Dense::MambaOut => vec![false; 5],
                Dense::AttnQ | Dense::AttnO => vec![true; 5],
                Dense::AttnK | Dense::AttnV => vec![true, false, false, false, false],
            };
            assert_eq!(row.iter().map(|(_, ok)| *ok).collect::<Vec<_>>(), want, "{}", role.name());
        }
        // the mirror's plans at the widths that move: in_proj 48 | 48 | 49 tiles at three ranks
        assert_eq!(decode_plan(4096, 48 * 128), (4, 4));
        assert_eq!(decode_plan(4096, 49 * 128), (4, 4));
        assert_eq!(decode_plan(4096, 18560), (1, 8));
        assert_eq!(decode_plan(4096, 29 * 128), (8, 4), "five ranks");
        // a zero-width slice has the plan of the 256-column matrix: the mirror alone would call it exact
        assert_eq!(decode_plan(4096, 0), decode_plan(4096, 256));
        assert!(!slice_keeps_plan(4096, 256, 3), "...so the guard is explicit");
    }

    #[test]
    fn a_matrix_with_fewer_tiles_than_ranks_stays_whole_in_every_plan() {
        let c = cfg();
        let p = plan(&c);
        let l7 = p.unit(UnitId::Trunk(7)).unwrap();
        for ranks in [3usize, 4, 8] {
            for rank in 0..ranks {
                let all = ShardPlan::slice_everything(TpTopology { rank, ranks });
                // k and v: 256 columns, 2 tiles: whole even when the plan says slice everything; q and o are cut
                assert_eq!(all.columns(Dense::AttnK, 256), (0, 256));
                assert_eq!(all.columns(Dense::AttnV, 256), (0, 256));
                let widths: Vec<usize> = all.local_linears(l7, c.experts).iter().map(|l| l.n).collect();
                let (q, o) = (column_range(4096, rank, ranks), column_range(4096, rank, ranks));
                assert_eq!(widths, vec![q.1 - q.0, 256, 256, o.1 - o.0], "{ranks} ranks, rank {rank}");
                for w in l7.tensors.iter().filter(|w| w.name.contains("k_proj") || w.name.contains("v_proj")) {
                    assert_eq!(all.piece(l7, w, c.experts), Piece::Whole, "{}", w.name);
                }
            }
        }
    }

    #[test]
    fn the_cut_widths_the_kernel_plan_depends_on_follow_the_rank_count() {
        // the widths of the 145-tile in_proj, and the 32-tile q and o, at each rank count (what the table above is made of)
        let widths = |n: usize, ranks: usize| -> Vec<usize> {
            (0..ranks).map(|r| column_range(n, r, ranks)).map(|(lo, hi)| (hi - lo) / TILE).collect()
        };
        assert_eq!(widths(18560, 2), vec![72, 73]);
        assert_eq!(widths(18560, 3), vec![48, 48, 49]);
        assert_eq!(widths(18560, 4), vec![36, 36, 36, 37]);
        assert_eq!(widths(18560, 5), vec![29; 5]);
        assert_eq!(widths(18560, 8), vec![18, 18, 18, 18, 18, 18, 18, 19]);
        assert_eq!(widths(4096, 3), vec![10, 11, 11]);
        assert_eq!(widths(4096, 5), vec![6, 6, 7, 6, 7]);
    }

    /// The head attention's prefixes (q, k, v, o of block 0).
    fn head_attention() -> Vec<String> {
        ["q_proj", "k_proj", "v_proj", "o_proj"].map(|p| format!("language_model.mtp.layers.0.mixer.{p}")).to_vec()
    }

    /// A 4-bit head's plan: every head projection EXL3 except `keep_bf16` (the published 4-bit pack keeps `eh_proj` bf16, and our 4-bit
    /// packs keep the attention bf16: pass it to keep it).
    fn four_bit_head_plan(cfg: &OmniConfig, keep_bf16: &[&str]) -> LayerPlan {
        let head: weights::HeadExl3 =
            weights::head_projections(cfg).into_iter().map(|(p, _, _)| p).filter(|p| !keep_bf16.contains(&p.as_str())).collect();
        let mut s = Storages::new();
        for (l, e) in BF16_PAIRS {
            for p in ["up_proj", "down_proj"] {
                s.insert((l, e, p), Storage::Bf16);
            }
        }
        LayerPlan { units: weights::units_with_head(cfg, &s, &head), outside_plan: 0 }
    }

    /// The loader's formula, independent of the plan: what `bridge::take_mtp_tp` leaves on a rank and `mtp::weight_bytes` counts. The bf16
    /// tensors in place; an EXL3 projection's trellis (K N / 2 bytes at 4 bits) with its scales, f32 in the MoE block's tables and fp16 for
    /// `eh_proj`, and no codebook marker; the routed experts the rank holds (`owns`).
    fn head_held_formula(cfg: &OmniConfig, bf16: &[&str], owns: impl Fn(usize) -> bool) -> u64 {
        let (h, lat, inter, sh) = (cfg.hidden, cfg.latent, cfg.expert_inter, cfg.shared_inter);
        let (q, kv) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
        let m = "language_model.mtp.layers.1.mixer";
        let proj = |prefix: &str, k: usize, n: usize, f32_scales: bool| -> usize {
            if bf16.contains(&prefix) {
                n * k * 2
            } else {
                k * n / 2 + (k + n) * if f32_scales { 4 } else { 2 }
            }
        };
        let mut b = (5 * h) * 2; // the five norms
        b += proj("language_model.mtp.layers.0.eh_proj", 2 * h, h, false);
        // the attention: bf16, or EXL3 with fp16 scales (`Exl3Weights`, as `eh_proj`)
        let a = |p: &str| format!("language_model.mtp.layers.0.mixer.{p}");
        b += proj(&a("q_proj"), h, q, false) + proj(&a("k_proj"), h, kv, false);
        b += proj(&a("v_proj"), h, kv, false) + proj(&a("o_proj"), q, h, false);
        b += cfg.experts * h * 2 + cfg.experts * 4; // the router and its f32 bias
        b += proj(&format!("{m}.fc1_latent_proj"), h, lat, true) + proj(&format!("{m}.fc2_latent_proj"), lat, h, true);
        b += proj(&format!("{m}.shared_experts.up_proj"), h, sh, true)
            + proj(&format!("{m}.shared_experts.down_proj"), sh, h, true);
        for e in (0..cfg.experts).filter(|&e| owns(e)) {
            b += proj(&format!("{m}.experts.{e}.up_proj"), lat, inter, true)
                + proj(&format!("{m}.experts.{e}.down_proj"), inter, lat, true);
        }
        b as u64
    }

    /// The shard plan counts a 4-bit head as the loader holds it, on one node and on each rank of a two-rank group (the head's experts
    /// split, and replicated), and a bf16 head exactly as before (its tensors' bytes). On the published 4-bit pack's form the difference
    /// to the index is 3,857,392 bytes a rank at two ranks: what the loopback test measured.
    #[test]
    fn a_four_bit_heads_plan_bytes_are_what_the_loader_holds_on_every_rank() {
        let c = cfg();
        let mtp = |u: UnitId| matches!(u, UnitId::Mtp(_));
        let index_sum = |sp: &ShardPlan, p: &LayerPlan| -> u64 {
            sp.wants(p, c.experts, mtp).map(|(_, w, piece)| sp.local_bytes(w, &piece).unwrap()).sum()
        };
        let group = |rank: usize| ShardPlan::for_config(TpTopology { rank, ranks: 2 }, &c);
        let shards: Vec<(String, ShardPlan)> = vec![
            ("single".into(), ShardPlan::single()),
            ("split rank 0".into(), group(0)),
            ("split rank 1".into(), group(1)),
            ("replicated rank 0".into(), group(0).replicating_head()),
            ("replicated rank 1".into(), group(1).replicating_head()),
        ];
        // a bf16 head: its tensors' bytes, as before
        let bf16 = plan(&c);
        for (what, sp) in &shards {
            assert_eq!(sp.rank_bytes(&bf16, c.experts, mtp), index_sum(sp, &bf16), "bf16 head, {what}");
        }
        let owner = |sp: &ShardPlan| {
            let sp = sp.clone();
            move |e: usize| sp.is_single() || !sp.head_experts || sp.owns_head_expert(e, c.experts)
        };
        // the published 4-bit pack's form (eh_proj bf16) and a mixed one (eh_proj EXL3, expert 3's down and the shared up bf16), the
        // attention bf16 as our packs hold it; and the MixedK pack's head, EXL3 throughout
        let m = "language_model.mtp.layers.1.mixer";
        let (e3, su) = (format!("{m}.experts.3.down_proj"), format!("{m}.shared_experts.up_proj"));
        let attn = head_attention();
        let with_attn = |v: &[&str]| -> Vec<String> { v.iter().map(|s| s.to_string()).chain(attn.iter().cloned()).collect() };
        let keeps: Vec<Vec<String>> =
            vec![with_attn(&["language_model.mtp.layers.0.eh_proj"]), with_attn(&[e3.as_str(), su.as_str()]), Vec::new()];
        for keep in &keeps {
            let keep: Vec<&str> = keep.iter().map(String::as_str).collect();
            let p = four_bit_head_plan(&c, &keep);
            for (what, sp) in &shards {
                let want = head_held_formula(&c, &keep, owner(sp));
                assert_eq!(sp.rank_bytes(&p, c.experts, mtp), want, "4-bit head (bf16 {keep:?}), {what}");
            }
        }
        let published_keep: Vec<String> = with_attn(&["language_model.mtp.layers.0.eh_proj"]);
        let published = four_bit_head_plan(&c, &published_keep.iter().map(String::as_str).collect::<Vec<_>>());
        for rank in 0..2 {
            let sp = group(rank);
            assert_eq!(sp.rank_bytes(&published, c.experts, mtp) - index_sum(&sp, &published), 3_857_392, "rank {rank}");
        }
        // the trunk is counted as before: only the head's units move
        let all = |u: UnitId| !matches!(u, UnitId::Mtp(_));
        for (what, sp) in &shards {
            assert_eq!(sp.rank_bytes(&published, c.experts, all), sp.rank_bytes(&bf16, c.experts, all), "trunk, {what}");
        }
    }

    /// P0-4: an EXL3 `lm_head` is cut along its output columns at 128-column blocks (its trellis and `svh`; `suh` and the marker whole),
    /// the rank's linear has the slice's width, and at two ranks the cut is the bf16 head's (65,536 | 65,536). Every slice keeps the decode
    /// plan of the whole head (bit-exact columns).
    #[test]
    fn an_exl3_lm_head_is_cut_at_128_column_blocks_on_every_rank() {
        let c = cfg();
        let widths = weights::Widths::from([(weights::lm_head_prefix(), 8)]);
        let mut s = Storages::new();
        for (l, e) in BF16_PAIRS {
            for p in ["up_proj", "down_proj"] {
                s.insert((l, e, p), Storage::Bf16);
            }
        }
        let plan = LayerPlan { units: weights::units_with(&c, &s, &weights::HeadExl3::new(), &widths), outside_plan: 0 };
        let unit = plan.unit(UnitId::LmHead).unwrap();
        let want = |part: &str| unit.tensors.iter().find(|t| t.name == format!("language_model.lm_head.{part}")).unwrap();
        for ranks in [2usize, 3, 4, 8] {
            let mut covered = 0;
            for rank in 0..ranks {
                let sp = ShardPlan::for_config(TpTopology { rank, ranks }, &c);
                let (lo, hi) = sp.lm_head_cols_exl3(c.vocab);
                assert_eq!((lo % 128, hi % 128, lo), (0, 0, covered), "{ranks} ranks, rank {rank}");
                covered = hi;
                assert_eq!(sp.piece(unit, want("trellis"), c.experts), Piece::Trellis { k: c.hidden, n: c.vocab, lo, hi });
                assert_eq!(sp.piece(unit, want("svh"), c.experts), Piece::Svh { lo, hi });
                assert_eq!(sp.piece(unit, want("suh"), c.experts), Piece::Whole);
                assert_eq!(sp.piece(unit, want("mul1"), c.experts), Piece::Whole);
                assert_eq!(
                    sp.local_shape(want("trellis"), &sp.piece(unit, want("trellis"), c.experts)),
                    Some(vec![256, (hi - lo) / 16, 128])
                );
                let l = sp.local_linears(unit, c.experts);
                assert_eq!(l, vec![Linear { prefix: "language_model.lm_head".into(), k: c.hidden, n: hi - lo, bits: 8 }]);
                assert!(slice_keeps_plan(c.hidden, c.vocab, ranks), "the decode plan of every slice is the whole head's");
                if ranks == 2 {
                    assert_eq!((lo, hi), sp.lm_head_rows(c.vocab), "two ranks: the bf16 head's cut");
                }
                // the rank holds its slice's bytes: the trellis at 8 bits a weight, its svh, the whole suh and the marker
                let bytes = sp.rank_bytes(&plan, c.experts, |u| u == UnitId::LmHead);
                assert_eq!(bytes, (c.hidden * (hi - lo) + 2 * (hi - lo) + 2 * c.hidden + 4) as u64, "{ranks} ranks, rank {rank}");
            }
            assert_eq!(covered, c.vocab, "{ranks} ranks cover the vocabulary");
        }
        assert_eq!(ShardPlan::single().piece(unit, want("trellis"), c.experts), Piece::Whole);
        assert_eq!(head_temp_rows_in(c.vocab, 3, EXL3_HEAD_ROW_UNIT), 43_776, "an upper bound of the EXL3 cut's widest rank");
        assert_eq!(head_row_cut_in(c.vocab, 0, 3, EXL3_HEAD_ROW_UNIT), (0, 43_776));
    }

    #[test]
    fn the_head_is_part_of_the_listed_bytes_and_the_rank_pair_is_what_engine_takes() {
        let c = cfg();
        let p = plan(&c);
        let [r0, _] = ranks();
        let (all, head) = rank_listed_bytes(&r0, &p, &c);
        let (all1, head1) = rank_listed_bytes(&ShardPlan::single(), &p, &c);
        assert!(all < all1 && head > 0);
        assert!(head < head1, "the MTP head's experts are split: a rank holds {head} of {head1}");
        // the experts are 5.6 of the head's 5.9 GB: a rank holds the dense part whole and about half of the experts
        assert!(head > head1 / 2 && head < head1 * 6 / 10, "{head} of {head1}");
        let rep = ShardPlan::for_config(TpTopology { rank: 0, ranks: 2 }, &c).replicating_head();
        let (_, head_rep) = rank_listed_bytes(&rep, &p, &c);
        assert_eq!(head_rep, head1, "REXL3_TP_HEAD=replicate: a rank holds all of it");
        let (_, head_b) = rank_listed_bytes(&ShardPlan::for_config(TpTopology { rank: 1, ranks: 2 }, &c), &p, &c);
        // the two ranks together hold the head's experts exactly once, and its dense part twice
        let dense: u64 = [UnitId::Mtp(0), UnitId::Mtp(1)]
            .into_iter()
            .flat_map(|id| p.unit(id).unwrap().tensors.iter())
            .filter(|w| head_expert_of(&w.name).is_none())
            .map(|w| w.shape.iter().product::<usize>() as u64 * dtype_size(w.dtype).unwrap())
            .sum();
        assert_eq!(head + head_b, head1 + dense, "experts once, the rest on both ranks");
    }

    // ---- B7: an ownership table instead of the contiguous ranges

    use super::super::tp_balance::{moe_layers, shuffled_map, ExpertMap};

    /// The serving plan of every rank of `ranks`, with `map` when given.
    fn with_map(ranks: usize, c: &OmniConfig, map: Option<&Arc<ExpertMap>>) -> Vec<ShardPlan> {
        (0..ranks)
            .map(|rank| {
                let p = ShardPlan::for_config(TpTopology { rank, ranks }, c);
                match map {
                    Some(m) => p.with_expert_map(m.clone()),
                    None => p,
                }
            })
            .collect()
    }

    #[test]
    fn a_contiguous_table_is_the_default_rule_bit_for_bit() {
        let c = cfg();
        let p = plan(&c);
        let layers = moe_layers(&c);
        for ranks in [2usize, 3, 4, 8] {
            let table = Arc::new(ExpertMap::contiguous(ranks, c.experts, layers.iter().copied()).unwrap());
            let (plain, tabled) = (with_map(ranks, &c, None), with_map(ranks, &c, Some(&table)));
            for r in 0..ranks {
                // every tensor of every unit: the same piece, the same bytes, the same linears
                for u in &p.units {
                    assert_eq!(plain[r].local_linears(u, c.experts), tabled[r].local_linears(u, c.experts), "{:?}", u.id);
                    for w in &u.tensors {
                        assert_eq!(plain[r].piece(u, w, c.experts), tabled[r].piece(u, w, c.experts), "{}", w.name);
                    }
                }
                assert_eq!(
                    plain[r].rank_bytes(&p, c.experts, |_| true),
                    tabled[r].rank_bytes(&p, c.experts, |_| true),
                    "R={ranks} rank {r}"
                );
                for &l in &layers {
                    assert_eq!(plain[r].layer_experts(l, c.experts), tabled[r].layer_experts(l, c.experts));
                    assert_eq!(plain[r].layer_experts(l, c.experts), plain[r].experts(c.experts).collect::<Vec<_>>());
                }
            }
        }
    }

    #[test]
    fn a_shuffled_table_holds_every_expert_on_one_rank_with_the_contiguous_counts_and_loads_exactly_those_weights() {
        let c = cfg();
        let p = plan(&c);
        let layers = moe_layers(&c);
        for ranks in [2usize, 3, 5, 8] {
            let map = Arc::new(shuffled_map(ranks, c.experts, &layers, 42).unwrap());
            map.check_for(&c, ranks).unwrap();
            let plans = with_map(ranks, &c, Some(&map));
            let plain = with_map(ranks, &c, None);
            for &l in &layers {
                let unit = p.unit(UnitId::Trunk(l)).unwrap();
                let mut holders = vec![0usize; c.experts];
                let mut lin_total = 0usize;
                for (r, sp) in plans.iter().enumerate() {
                    let own = sp.layer_experts(l, c.experts);
                    // the same number of experts a rank as the contiguous plan: the memory plan's slots do not move
                    assert_eq!(own.len(), plain[r].experts(c.experts).len(), "layer {l} rank {r} of {ranks}");
                    for &e in &own {
                        holders[e] += 1;
                        assert_eq!(map.owner(l, e), Some(r));
                    }
                    // the loader's view: exactly the owned experts' tensors are wanted, in the experts' own names
                    let wanted: usize =
                        unit.tensors.iter().filter(|w| routed_expert_of(&w.name).is_some_and(|e| own.contains(&e))).count();
                    let loaded: usize = unit
                        .tensors
                        .iter()
                        .filter(|w| routed_expert_of(&w.name).is_some() && sp.piece(unit, w, c.experts) != Piece::Skip)
                        .count();
                    assert_eq!(wanted, loaded);
                    lin_total +=
                        sp.local_linears(unit, c.experts).iter().filter(|l| routed_expert_of(&l.prefix).is_some()).count();
                }
                let all_lin = unit.linears.iter().filter(|l| routed_expert_of(&l.prefix).is_some()).count();
                assert_eq!(lin_total, all_lin, "layer {l}: every expert's EXL3 linears on exactly one rank");
                assert!(holders.iter().all(|&n| n == 1), "layer {l}: every expert on exactly one rank");
            }
            // across the group every tensor of the pack is held exactly once (the experts) or by every rank (the rest): the sum of
            // the ranks' bytes is the same as the contiguous plan's up to the three bf16 pairs falling on other ranks
            let (a, b): (u64, u64) = (
                plans.iter().map(|sp| sp.rank_bytes(&p, c.experts, |_| true)).sum(),
                plain.iter().map(|sp| sp.rank_bytes(&p, c.experts, |_| true)).sum(),
            );
            assert_eq!(a, b, "R={ranks}: every expert exactly once, whatever the table");
            // and the fullest rank differs from the contiguous plan's by at most the bf16 pairs' excess (3 pairs, ~16 MB each rank)
            let max_a = plans.iter().map(|sp| sp.rank_bytes(&p, c.experts, |_| true)).max().unwrap();
            let max_b = plain.iter().map(|sp| sp.rank_bytes(&p, c.experts, |_| true)).max().unwrap();
            assert!(max_a.abs_diff(max_b) < 64 << 20, "R={ranks}: {max_a} vs {max_b}");
        }
    }

    #[test]
    fn the_head_follows_the_tables_head_row_or_stays_contiguous() {
        let c = cfg();
        let layers = moe_layers(&c);
        let m = shuffled_map(2, c.experts, &layers, 3).unwrap();
        let no_head = Arc::new(m.clone());
        let mut rows: std::collections::BTreeMap<usize, Vec<u8>> =
            layers.iter().map(|&l| (l, m.row(l).unwrap().to_vec())).collect();
        let head_row: Vec<u8> = (0..c.experts).map(|e| (1 - (e / 256)) as u8).collect();
        rows.insert(layers[0], m.row(layers[0]).unwrap().to_vec());
        let with_head = Arc::new(ExpertMap::new(2, c.experts, rows, Some(head_row)).unwrap());
        let p = |map: &Arc<ExpertMap>, rank: usize| {
            ShardPlan::for_config(TpTopology { rank, ranks: 2 }, &c).with_expert_map(map.clone())
        };
        // no head row: the head's experts are the contiguous ranges, whatever the trunk's table
        assert!(p(&no_head, 0).owns_head_expert(0, 512) && !p(&no_head, 0).owns_head_expert(300, 512));
        // a head row: the table's (here the halves swapped)
        assert!(!p(&with_head, 0).owns_head_expert(0, 512) && p(&with_head, 0).owns_head_expert(300, 512));
        assert!(p(&with_head, 1).owns_head_expert(0, 512));
    }

    // ---- v0.1.2 P2-8: the byte-weighted balance

    use super::super::tp_balance::byte_balanced_map;

    /// Every unit's pieces and linears, the rank's bytes (all units, the head) and every MoE layer's experts: what "the same plan" means.
    fn assert_same_plan(a: &ShardPlan, b: &ShardPlan, p: &LayerPlan, c: &OmniConfig, what: &str) {
        assert_eq!(a, b, "{what}: the plan value");
        for u in &p.units {
            assert_eq!(a.local_linears(u, c.experts), b.local_linears(u, c.experts), "{what}: {:?}", u.id);
            for w in &u.tensors {
                assert_eq!(a.piece(u, w, c.experts), b.piece(u, w, c.experts), "{what}: {}", w.name);
            }
        }
        for hg in [false, true] {
            assert_eq!(
                a.rank_bytes_with(p, c.experts, |_| true, hg),
                b.rank_bytes_with(p, c.experts, |_| true, hg),
                "{what}: bytes (hg {hg})"
            );
        }
        assert_eq!(rank_listed_bytes(a, p, c), rank_listed_bytes(b, p, c), "{what}: listed bytes");
        for l in moe_layers(c) {
            assert_eq!(a.layer_experts(l, c.experts), b.layer_experts(l, c.experts), "{what}: layer {l}");
        }
    }

    /// R-MK-noreg (c) on the committed config's plan (one width, the three bf16 expert pairs, a bf16 or a 4-bit head): the byte balance
    /// gives no table at any rank count, so the plan is today's, with the bytes the two-rank pins above hold.
    #[test]
    fn the_byte_balance_leaves_a_plan_of_one_width_on_todays_plan_bit_for_bit() {
        let c = cfg();
        let moe = moe_layers(&c);
        for p in [plan(&c), four_bit_head_plan(&c, &["language_model.mtp.layers.0.eh_proj"])] {
            let t = expert_bytes(&p, c.experts);
            assert_eq!(t.bytes.len(), 40, "every MoE layer");
            assert!(t.widths.values().all(|w| w == &vec![4]), "one width: {:?}", t.widths);
            assert!(t.mixed_layers().is_empty());
            // the bf16 pairs are heavier (the table sees them) and still move nothing
            for (l, e) in BF16_PAIRS {
                let row = &t.bytes[&l];
                assert_eq!(row[e], 2 * 1024 * 2688 * 2, "layer {l} expert {e}: a bf16 pair");
                assert!(row.iter().enumerate().all(|(x, &b)| x == e || b == row[(e + 1) % 512]));
            }
            for ranks in [2usize, 3, 4, 5, 8] {
                let map = byte_balanced_map(&t, ranks, &moe).unwrap();
                assert_eq!(map, None, "R={ranks}");
                for rank in 0..ranks {
                    let topo = TpTopology { rank, ranks };
                    let today = ShardPlan::for_config(topo, &c);
                    let balanced = match &map {
                        Some(m) => ShardPlan::for_config(topo, &c).with_expert_map(Arc::new(m.clone())),
                        None => ShardPlan::for_config(topo, &c),
                    };
                    assert_same_plan(&today, &balanced, &p, &c, &format!("R={ranks} rank {rank}"));
                }
            }
        }
        // and the pins of the verified two-rank plan
        let p = plan(&c);
        let t = expert_bytes(&p, c.experts);
        for (r, want) in [35_746_806_328u64, 35_765_545_008].into_iter().enumerate() {
            let sp = ShardPlan::for_config(TpTopology { rank: r, ranks: 2 }, &c);
            let sp = match byte_balanced_map(&t, 2, &moe).unwrap() {
                Some(m) => sp.with_expert_map(Arc::new(m)),
                None => sp,
            };
            assert_eq!(sp.rank_bytes_with(&p, c.experts, |_| true, false), want, "rank {r}");
        }
    }

    /// A plan whose layer 1 mixes 2, 4 and 6-bit experts (the first 96 of them, all in rank 0's contiguous half): the table moves only
    /// that layer's experts, every expert is on one rank, the counts and every other tensor's piece stay, and the fuller rank sheds bytes.
    #[test]
    fn a_mixed_width_plan_gets_a_byte_balanced_table_that_moves_only_its_mixed_layer() {
        let c = cfg();
        let moe = moe_layers(&c);
        let mut p = plan(&c);
        let unit = p.units.iter_mut().find(|u| u.id == UnitId::Trunk(1)).unwrap();
        for w in unit.tensors.iter_mut().filter(|w| w.name.ends_with(".trellis")) {
            if let Some(e) = routed_expert_of(&w.name) {
                if e < 64 {
                    w.shape[2] = 16 * 6;
                } else if e < 96 {
                    w.shape[2] = 16 * 2;
                }
            }
        }
        let t = expert_bytes(&p, c.experts);
        assert_eq!(t.mixed_layers(), vec![1]);
        assert_eq!(t.widths[&1], vec![2, 4, 6]);
        let m = Arc::new(byte_balanced_map(&t, 2, &moe).unwrap().expect("layer 1 mixes widths"));
        m.check_for(&c, 2).unwrap();
        let today: Vec<ShardPlan> = (0..2).map(|r| ShardPlan::for_config(TpTopology { rank: r, ranks: 2 }, &c)).collect();
        let balanced: Vec<ShardPlan> = today.iter().map(|s| s.clone().with_expert_map(m.clone())).collect();
        for r in 0..2 {
            for u in &p.units {
                for w in &u.tensors {
                    if routed_expert_of(&w.name).is_none() || u.id != UnitId::Trunk(1) {
                        assert_eq!(today[r].piece(u, w, c.experts), balanced[r].piece(u, w, c.experts), "{}", w.name);
                    }
                }
            }
            assert_eq!(balanced[r].layer_experts(1, c.experts).len(), 256);
            for &l in moe.iter().filter(|&&l| l != 1) {
                assert_eq!(balanced[r].layer_experts(l, c.experts), today[r].layer_experts(l, c.experts), "layer {l}");
            }
        }
        let bytes =
            |s: &[ShardPlan]| -> Vec<u64> { s.iter().map(|sp| sp.rank_bytes_with(&p, c.experts, |_| true, false)).collect() };
        let (b0, b1) = (bytes(&today), bytes(&balanced));
        assert_eq!(b0.iter().sum::<u64>(), b1.iter().sum::<u64>(), "every expert's bytes on exactly one rank");
        assert!(b1.iter().max() < b0.iter().max(), "{b0:?} -> {b1:?}");
        // layer 1 alone: within one bit of one projection pair's step of even
        let (l0, l1) = (
            t.rank_bytes(1, m.row(1).unwrap(), 2),
            t.rank_bytes(1, &[0u8; 256].iter().chain([1u8; 256].iter()).copied().collect::<Vec<_>>(), 2),
        );
        assert!(l0[0].abs_diff(l0[1]) < l1[0].abs_diff(l1[1]));
        assert!(l0[0].abs_diff(l0[1]) <= 2 * 1024 * 2688 * 2 / 8, "{l0:?}");
    }

    #[test]
    #[should_panic(expected = "contiguous rule")]
    fn the_range_form_refuses_a_plan_with_a_table() {
        let c = cfg();
        let m = Arc::new(ExpertMap::contiguous(2, c.experts, moe_layers(&c)).unwrap());
        let _ = ShardPlan::for_config(TpTopology { rank: 0, ranks: 2 }, &c).with_expert_map(m).owns_expert(0, 512);
    }

    #[test]
    fn a_single_rank_ignores_a_table() {
        let c = cfg();
        let m = Arc::new(ExpertMap::contiguous(2, c.experts, moe_layers(&c)).unwrap());
        let s = ShardPlan::single().with_expert_map(m);
        assert!(s.expert_map.is_none() && s.owns_layer_expert(1, 300, 512) && s.owns_head_expert(7, 512));
    }
}
