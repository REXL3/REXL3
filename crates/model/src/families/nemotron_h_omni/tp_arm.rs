//! The TP-exact pieces the three arms and the head share (`docs/NEMOTRON_TP.md` section 3). One rank of a group of any size runs the
//! layer loop; every output element of a sharded matrix is computed by exactly one rank with the single-node kernel, written into a
//! full-width buffer whose other columns are `-0.0`, the additive identity (`x + (-0.0) == x` for every `x`, the sign of a zero
//! included: a `+0.0` filler would turn a `-0.0` of the row into `+0.0`, which the raw-bit digests see), and the sum of the ranks'
//! buffers is the gathered row, bit for bit (at most one rank is non-zero per element, so the sum is exact in any order and at any
//! rank count).
//!
//! For one N-split linear, in stream order: `Exl3Weights::forward` on the rank's slice into a local `[rows, n_local]` temp, a scatter
//! of that temp into the full-width destination (every other column exact `-0.0`), then one `allreduce_sum_f32` over the destination.
//! A weight that is whole on this rank (a single-rank model, or a shape the plan replicated: its slice failed the slice-vs-whole
//! proof, had no tile for some rank, or the plan chose so) runs the single-node call alone: no scatter, no temp, no collective.
//!
//! **Who decides whether a collective runs.** Only plan-level facts, the same on every rank by construction: the rank count, the
//! shapes the shard plan gives the weights (`shard`, held at boot by `check_weights`), the plan flags exchanged at the rendezvous
//! ([`plan_flags`]) and the call's own arguments, which the lockstep protocol makes identical (the rows `m`). A rank-local buffer
//! property (a table that is a view of a block, a temp's size) never decides it: a buffer that does not fit the plan is a loud error
//! (`head_path`, `qkv_tp`, [`check_buffers`] at boot), never a quiet fallback that would leave the peers in a collective.
//!
//! Everything here is a function of the host values the graph key already holds (`rows`) and of the weights' shapes, so a captured
//! window replays the same launches on every rank, and a single rank's launch sequence is the one it always was.

use super::backend::WINDOW_ROWS;
use super::config::OmniConfig;
use super::device::{AttnW, HeadLinear, Mixer, NemotronWeights};
use super::model::linear_rows;
use super::tp::{splits, TpComm, TpTopology};
use super::tp_balance::{self, contiguous_range};
use super::tp_shard::{
    column_range, dense_shapes, head_row_cut, head_row_cut_in, head_splits, head_temp_rows_in, sliceable, Dense, ShardPlan,
    EXL3_HEAD_ROW_UNIT, HEAD_ROW_UNIT, TILE,
};
use crate::error::ModelError;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::nemotron_dense::gemv_bf16;
use gb_cuda::nemotron_glue::{widen_bf16_f32, HIDDEN};
use gb_cuda::nemotron_moe::MoeScratch;
use gb_cuda::nemotron_moe_prefill::{PrefillMoe, PrefillScratch};
use gb_cuda::vision::f32_to_bf16;
use gb_cuda::{CudaError, DevBuf, Stream};
use std::sync::{Arc, Mutex};

/// Whether the attention matrices (q, k, v, o) are sliced across the ranks. Off by default: slicing them saves at most 15 to 17 us of
/// weight read a layer and pays one collective of 30 to 60 us each (q, k, v share one), so only the Mamba `in_proj` and the experts
/// clearly pay (`docs/NEMOTRON_TP.md` section 8). Exactness is the same either way (every slice is bit-exact); the arms follow the
/// weights' shapes, so this is the one switch. `REXL3_TP_ATTN=slice` or `=replicate` overrides it for a run; BOTH ranks must run with
/// the same setting (a rank that slices a matrix its peer holds whole would wait for a collective that never comes).
pub const SLICE_ATTENTION: bool = false;

/// The environment variable that overrides [`SLICE_ATTENTION`].
pub const ATTN_ENV: &str = "REXL3_TP_ATTN";

/// Whether this process slices the attention matrices: [`ATTN_ENV`], else [`SLICE_ATTENTION`].
pub fn slice_attention() -> bool {
    match std::env::var(ATTN_ENV).as_deref() {
        Ok("slice") => true,
        Ok("replicate") => false,
        _ => SLICE_ATTENTION,
    }
}

/// The environment variable that replicates the Mamba `in_proj` too (`replicate`): every rank holds the whole matrix and computes the whole
/// output (no scatter and no collective, 40 fewer a step) at the price of reading the other rank's half again (38 MB a layer, 19 MB
/// saved by slicing, about 70 us at 273 GB/s) against a collective of 45 to 80 us and its launch gaps: a bring-up knob to measure which
/// side wins on this fabric. Exactness is the same (the slices are bit-exact; the whole matrix is the single-node call). BOTH ranks must
/// run with the same setting (`make_comm` checks it, [`plan_flags`]).
pub const INPROJ_ENV: &str = "REXL3_TP_INPROJ";

/// Whether this process replicates the Mamba `in_proj` ([`INPROJ_ENV`]).
pub fn replicate_in_proj() -> bool {
    matches!(std::env::var(INPROJ_ENV).as_deref(), Ok("replicate"))
}

/// The environment variable that says how the MTP draft head is split across the ranks (`docs/NEMOTRON_TP.md` section 9.2): unset or
/// `split` both of the head's parts, `experts` only its routed experts (the restricted `lm_head` table is read whole on every rank),
/// `lm_head` only the restricted table (the experts are whole on every rank), `replicate` neither (the head is the identical call on
/// both ranks with no collective: the layout of the first two-node bring-up). Exactness is the same in every mode; BOTH ranks must run
/// with the same value ([`plan_flags`]).
pub const HEAD_ENV: &str = "REXL3_TP_HEAD";

/// What [`HEAD_ENV`] splits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadSplit {
    /// The head's routed experts: a rank holds half and the block allreduces its `y` (`mtp::head_block`).
    pub experts: bool,
    /// The head's restricted `lm_head` table: a rank scores half of its rows and the logits are combined by one allreduce
    /// (`mtp::head_pick`, `lm_head_logits(split_whole = true)`).
    pub lm_head: bool,
}

impl HeadSplit {
    /// The setting of a variable's text (unset, empty and anything unrecognised are the default: both).
    pub fn parse(v: Option<&str>) -> Self {
        match v.map(str::trim) {
            Some("replicate") => HeadSplit { experts: false, lm_head: false },
            Some("experts") => HeadSplit { experts: true, lm_head: false },
            Some("lm_head") => HeadSplit { experts: false, lm_head: true },
            _ => HeadSplit { experts: true, lm_head: true },
        }
    }
}

/// How this process splits the MTP head ([`HEAD_ENV`]).
pub fn head_split() -> HeadSplit {
    HeadSplit::parse(std::env::var(HEAD_ENV).ok().as_deref())
}

/// The serving plan's switches as one byte, which the ranks compare at the rendezvous: bit 0 the attention matrices sliced, bit 1 the
/// `in_proj` replicated, bit 2 the MTP head's experts whole, bit 3 the head's restricted `lm_head` whole, bit 4 an expert ownership
/// table configured (`REXL3_TP_EXPERT_MAP`, B7; the table's 64-bit content hash is `tp_balance::expert_map_fingerprint`), bit 5 RESERVED
/// (it meant an fp8 KV cache, which was evaluated and rejected: always zero, never reuse it), bit 6 RESERVED for the attention split by
/// kv head (branch nemotron/kv-split; zero here), bit 7 an int8 KV cache (`--kv-dtype int8`) (the defaults are all zero).
pub fn plan_flags() -> u8 {
    let h = head_split();
    u8::from(slice_attention())
        | (u8::from(replicate_in_proj()) << 1)
        | (u8::from(!h.experts) << 2)
        | (u8::from(!h.lm_head) << 3)
        | (u8::from(tp_balance::expert_map_fingerprint() != 0) << 4)
        | (u8::from(gb_cuda::nemotron_attn::configured_kv_dtype().unwrap_or_default() == gb_cuda::nemotron_attn::KvDtype::Int8)
            << 7)
}

/// The serving shard plan of a rank (`tp_shard::ShardPlan`): the Mamba `in_proj` sliced (9,216 | 9,344), the Mamba `out_proj` whole
/// (its sliced kernel plan is not bit-exact), the attention matrices whole unless [`slice_attention`], the routed experts split, the
/// `lm_head` rows split, and the MTP head's routed experts split unless [`HEAD_ENV`] says otherwise. The memory plan prices the same
/// plan (`engine::resident_weight_bytes_tp`).
pub fn serving_shard_plan(topo: TpTopology, cfg: &OmniConfig) -> ShardPlan {
    // a configured expert map that cannot be used is refused by name where the weights load (`serving_shard_plan_checked`); this
    // infallible form (the memory plan's) prices the contiguous ranges then
    serving_shard_plan_checked(topo, cfg).unwrap_or_else(|_| serving_shard_plan_with(topo, cfg, None))
}

/// [`serving_shard_plan`] that refuses a configured expert map (`REXL3_TP_EXPERT_MAP`) which is unreadable, damaged, or made for
/// another group size or model, by name. With no map configured this is the contiguous plan, bit for bit.
pub fn serving_shard_plan_checked(topo: TpTopology, cfg: &OmniConfig) -> Result<ShardPlan, String> {
    let map = if topo.is_single() { None } else { tp_balance::configured_map()? };
    if let Some(m) = &map {
        m.check_for(cfg, topo.ranks).map_err(|e| format!("{}: {e}", tp_balance::MAP_ENV))?;
    }
    Ok(serving_shard_plan_with(topo, cfg, map))
}

/// The serving plan of a rank with the ownership table `map` (`None`: the contiguous ranges).
pub fn serving_shard_plan_with(topo: TpTopology, cfg: &OmniConfig, map: Option<Arc<tp_balance::ExpertMap>>) -> ShardPlan {
    let mut plan = ShardPlan::for_config(topo, cfg);
    if let Some(m) = map {
        plan = plan.with_expert_map(m);
    }
    if !head_split().experts {
        plan = plan.replicating_head();
    }
    if replicate_in_proj() {
        plan = plan.replicating(&[Dense::MambaIn]);
    }
    if slice_attention() {
        plan
    } else {
        plan.replicating(&[Dense::AttnQ, Dense::AttnK, Dense::AttnV, Dense::AttnO])
    }
}

/// One line for the boot log with the serving plan's decision for every dense matrix and the cuts of the other tensors, for a group of
/// `ranks` (identical on every rank): which matrices are sliced and into what widths, which are replicated and why (the K reduction
/// moves, a rank would hold no tile, or the plan's switches), the routed-expert ranges and the `lm_head` row cut. The record of
/// "in_proj is replicated at three ranks and more".
pub fn shard_plan_line(ranks: usize, cfg: &OmniConfig) -> String {
    let plan = serving_shard_plan(TpTopology { rank: 0, ranks }, cfg);
    let mut parts: Vec<String> = Vec::new();
    for (role, _k, n, held) in plan.decisions(cfg) {
        let name = role.name().replace(' ', "_");
        let attn = matches!(role, Dense::AttnQ | Dense::AttnK | Dense::AttnV | Dense::AttnO);
        parts.push(match held {
            Ok(widths) => format!(
                "{name}=sliced[{}]",
                widths.iter().map(|w| w.to_string()).collect::<Vec<_>>().join("|")
            ),
            // the serving plan's own switches are the reason when the matrix is exact and the plan still holds it whole
            Err(reason) if reason.starts_with("plan choice") && attn && !slice_attention() => {
                format!("{name}=replicated(reason=\"{ATTN_ENV} default: slicing {n} columns saves less read than its collective costs\")")
            }
            Err(reason) if reason.starts_with("plan choice") && role == Dense::MambaIn && replicate_in_proj() => {
                format!("{name}=replicated(reason=\"{INPROJ_ENV}=replicate\")")
            }
            Err(reason) => format!("{name}=replicated(reason=\"{reason}\")"),
        });
    }
    let experts: Vec<String> = match plan.expert_map.as_deref() {
        // a table is not a range: the counts a rank holds (equal to the contiguous plan's) and the table's content hash
        Some(m) => {
            (0..ranks).map(|r| format!("{}", m.experts_on(r))).chain(std::iter::once(format!("map={:016x}", m.hash()))).collect()
        }
        None => (0..ranks)
            .map(|r| {
                let e = contiguous_range(cfg.experts, r, ranks);
                format!("{}..{}", e.start, e.end)
            })
            .collect(),
    };
    let rows: Vec<String> = (0..ranks)
        .map(|r| {
            let (lo, hi) = head_row_cut(cfg.vocab, r, ranks);
            format!("{lo}..{hi}")
        })
        .collect();
    format!(
        "rexl3 event=tp_shard_plan ranks={ranks} {} experts=[{}] lm_head_rows=[{}] plan_flags={:#04x}",
        parts.join(" "),
        experts.join("|"),
        rows.join("|"),
        plan_flags()
    )
}

/// Held at construction of a TP model (`NemotronModel::new_tp_lag`, after the step buffers exist): every buffer the collective paths
/// need is there and big enough for the plan, so a call never discovers a shape problem on one rank while its peers wait in a
/// collective. `tmp`, `qkv`, `logits` are the byte sizes of `Bufs::{tp_tmp, tp_qkv, tp_logits}` (`None`: not allocated), `rows_cap`
/// the model's widest step. Checks: the sliced linears' temp, the merged q, k, v buffer when attention layers exist (and always when
/// all three are sliced), the logits temp at a window of rows of [`head_temp_rows`], and that a trunk head held WHOLE which the plan
/// cuts is a view of a block.
pub fn check_buffers(
    w: &NemotronWeights,
    topo: TpTopology,
    vocab: usize,
    rows_cap: usize,
    tmp: Option<usize>,
    qkv: Option<usize>,
    logits: Option<usize>,
) -> Result<(), String> {
    if topo.is_single() {
        return if tmp.is_none() && qkv.is_none() && logits.is_none() {
            Ok(())
        } else {
            Err("a single rank allocates no tensor-parallel buffer".into())
        };
    }
    let need_tmp = rows_cap * max_local_cols(w) * 4;
    match tmp {
        Some(b) if b >= need_tmp => {}
        other => {
            return Err(format!("the sliced linears' temp holds {other:?} bytes, the plan needs {need_tmp} ({rows_cap} rows)"))
        }
    }
    let has_attn = w.layers.iter().any(|l| matches!(l.mixer, Mixer::Attn(_)));
    let need_qkv = rows_cap * QKV_COLS * 4;
    match (has_attn, qkv) {
        (true, Some(b)) if b >= need_qkv => {}
        (false, None) => {}
        (has, other) => {
            return Err(format!(
                "the merged q, k, v buffer holds {other:?} bytes, attention layers on the device: {has}, needs {need_qkv}"
            ))
        }
    }
    if lm_head_rows(&w.lm_head) > 0 {
        let need = rows_cap.min(WINDOW_ROWS) * head_temp_rows_in(vocab, topo.ranks, HeadTable::from(&w.lm_head).unit()) * 4;
        match logits {
            Some(b) if b >= need => {}
            other => return Err(format!("the logits temp holds {other:?} bytes, the plan needs {need}")),
        }
        // (an EXL3 head held whole is scored whole on every rank: it is never cut as a view, `head_path_in`)
        let whole_bf16 = w.lm_head.bf16().filter(|_| lm_head_rows(&w.lm_head) == vocab);
        if whole_bf16.is_some_and(|b| b.block().is_none()) && head_splits(vocab, topo.ranks) {
            return Err(format!(
                "the trunk's {vocab}-row head is held whole but is not a view of a block: the plan cuts it ({} ranks) and a rank \
                 reads its rows as a sub-range",
                topo.ranks
            ));
        }
    }
    Ok(())
}

/// Routed experts per token and the latent width of the MoE block's `y` buffer (`[rows, 22, 1024]` f32).
const MOE_TOPK: usize = gb_cuda::nemotron_moe::TOPK;
const MOE_LATENT: usize = gb_cuda::nemotron_moe::LATENT;

/// Columns of one attention layer's q, k and v outputs, which one merged buffer holds block by block (`[rows, 4096]`, `[rows, 256]`,
/// `[rows, 256]`) so that one collective serves the three.
pub const QKV_COLS: usize = splits::Q_COLS + 2 * splits::KV_COLS;

/// How a rank holds one weight's output columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shard {
    /// Every column: the single-node call, no collective.
    Whole,
    /// Columns `[lo, lo + n_local)` of the full width.
    Cols { lo: usize, n_local: usize },
}

/// Where a weight of `n_have` output columns sits in an `n_total`-wide output on this rank of any group: the whole output (the
/// plan replicated it: no collective), or this rank's tiles of it (`tp_shard::column_range`, the one cut rule: 9,216 | 9,344 of the
/// `in_proj` at two ranks, `floor(tiles r / R)` tiles at R). A width that is neither the whole output nor this rank's share is refused
/// by name, and so is a slice of a matrix that has fewer tiles than ranks (such a matrix is never sliced): the engine never computes
/// with a slice it did not expect. `n_have == n_total` is unambiguous: a sliced rank always holds fewer columns than the whole.
pub fn shard(topo: TpTopology, n_total: usize, n_have: usize) -> Result<Shard, String> {
    if n_have == n_total {
        return Ok(Shard::Whole);
    }
    if topo.is_single() {
        return Err(format!("a single-rank model holds whole matrices: {n_have} columns of {n_total}"));
    }
    if topo.rank >= topo.ranks {
        return Err(format!("rank {} is not in a group of {}", topo.rank, topo.ranks));
    }
    if !sliceable(n_total, topo.ranks) {
        return Err(format!(
            "{n_total} columns ({} tiles of {TILE}) cannot be cut into {} slices of a whole tile: the matrix is held whole on every \
             rank, a slice of {n_have} columns is not one",
            n_total / TILE,
            topo.ranks
        ));
    }
    let (lo, hi) = column_range(n_total, topo.rank, topo.ranks);
    if n_have != hi - lo {
        return Err(format!(
            "rank {} of {} holds {n_have} columns of {n_total}, its share is [{lo}, {hi})",
            topo.rank, topo.ranks
        ));
    }
    Ok(Shard::Cols { lo, n_local: n_have })
}

/// Refuses weights that are not this rank's share, by layer and tensor, before anything is allocated. A single rank holds whole
/// matrices by construction (the loader's own checks), so nothing is checked for it: the single-node engine is untouched.
pub fn check_weights(w: &NemotronWeights, vocab: usize, topo: TpTopology) -> Result<(), String> {
    if topo.is_single() {
        return Ok(());
    }
    let bad = |what: String, e: String| format!("{what}: {e}");
    for (i, l) in w.layers.iter().enumerate() {
        let li = w.first + i;
        match &l.mixer {
            Mixer::Mamba(m) => {
                shard(topo, splits::IN_PROJ_COLS, m.in_proj.n).map_err(|e| bad(format!("layer {li} in_proj"), e))?;
                shard(topo, splits::HIDDEN, m.out_proj.n).map_err(|e| bad(format!("layer {li} out_proj"), e))?;
            }
            Mixer::Attn(a) => {
                shard(topo, splits::Q_COLS, a.q.n).map_err(|e| bad(format!("layer {li} q"), e))?;
                shard(topo, splits::KV_COLS, a.k.n).map_err(|e| bad(format!("layer {li} k"), e))?;
                shard(topo, splits::KV_COLS, a.v.n).map_err(|e| bad(format!("layer {li} v"), e))?;
                shard(topo, splits::O_COLS, a.o.n).map_err(|e| bad(format!("layer {li} o"), e))?;
            }
            Mixer::Moe(_) => {}
        }
    }
    let unit = HeadTable::from(&w.lm_head).unit();
    lm_head_shard_in(topo, vocab, lm_head_rows(&w.lm_head), unit).map_err(|e| bad("lm_head".into(), e))?;
    Ok(())
}

/// The widest local output of any SLICED linear of the layer loop (columns, at least 1): what the temp of a sharded linear must hold
/// per row. A matrix that is whole on this rank never goes through the temp (at three ranks and more the whole `in_proj`, 18,560
/// columns, would otherwise size it), so it does not count. Two ranks: 9,344 on rank 1, 9,216 on rank 0 (`tp_shard::ShardPlan::widest_slice`
/// is the group-wide figure the memory plan prices).
pub fn max_local_cols(w: &NemotronWeights) -> usize {
    let mut widest = 1usize;
    let mut sliced = |n: usize, total: usize| {
        if n < total {
            widest = widest.max(n);
        }
    };
    for l in &w.layers {
        match &l.mixer {
            Mixer::Mamba(m) => {
                sliced(m.in_proj.n, splits::IN_PROJ_COLS);
                sliced(m.out_proj.n, splits::HIDDEN);
                // the head-group in_proj (P2 at two ranks, lane S) writes its 9,344 columns into the same temp: on rank 0 that is wider
                // than its 9,216-column slice
                if let Some(hg) = &m.in_proj_hg {
                    sliced(hg.n, splits::IN_PROJ_COLS);
                }
            }
            Mixer::Attn(a) => {
                sliced(a.q.n, splits::Q_COLS);
                sliced(a.k.n, splits::KV_COLS);
                sliced(a.v.n, splits::KV_COLS);
                sliced(a.o.n, splits::O_COLS);
            }
            Mixer::Moe(_) => {}
        }
    }
    widest
}

/// The rows of the logits temp (`Bufs::tp_logits`, per window row) a rank of a group needs, `None` on a single rank (which computes
/// them all in the logits buffer): [`head_temp_rows`], the same on every rank and an upper bound of every rank's share of any table of
/// at most `vocab` rows, whether the device holds just the rank's rows (a sliced upload) or the whole head (the rank then reads its
/// rows as a sub-range of the buffer, which keeps the restricted draft head's gather a plain copy).
pub fn lm_head_rows_local(w: &NemotronWeights, vocab: usize, topo: TpTopology) -> Option<usize> {
    if topo.is_single() {
        return None;
    }
    (lm_head_rows(&w.lm_head) > 0).then(|| head_temp_rows_in(vocab, topo.ranks, HeadTable::from(&w.lm_head).unit()))
}

/// The rows of the trunk's `lm_head` a device holds: `vocab` when it is whole, the rank's `head_row_cut` width for a slice (an EXL3
/// head's: its output columns, `Exl3Weights::n`).
pub fn lm_head_rows(lm_head: &HeadLinear) -> usize {
    lm_head.rows(HIDDEN)
}

/// A table the head GEMV scores rows of: the trunk's `lm_head` (bf16 `[rows, 4096]`, or an EXL3 linear of `n` output columns: the MixedK
/// pack's 8-bit head) or the restricted draft head's bf16 sub-table.
#[derive(Clone, Copy)]
pub enum HeadTable<'a> {
    Bf16(&'a DevBuf),
    Exl3(&'a Exl3Weights),
}

impl<'a> From<&'a HeadLinear> for HeadTable<'a> {
    fn from(l: &'a HeadLinear) -> Self {
        match l {
            HeadLinear::Bf16(b) => HeadTable::Bf16(b),
            HeadLinear::Exl3(e) => HeadTable::Exl3(e),
        }
    }
}

impl<'a> From<&'a DevBuf> for HeadTable<'a> {
    fn from(b: &'a DevBuf) -> Self {
        HeadTable::Bf16(b)
    }
}

impl HeadTable<'_> {
    /// The rows (vocabulary entries) the table holds.
    pub fn rows(&self) -> usize {
        match self {
            HeadTable::Bf16(b) => b.bytes() / (HIDDEN * 2),
            HeadTable::Exl3(l) => l.n,
        }
    }

    /// The unit the plan cuts the table's rows in ([`HEAD_ROW_UNIT`], or [`EXL3_HEAD_ROW_UNIT`]: an EXL3 head's 128-column blocks).
    pub fn unit(&self) -> usize {
        match self {
            HeadTable::Bf16(_) => HEAD_ROW_UNIT,
            HeadTable::Exl3(_) => EXL3_HEAD_ROW_UNIT,
        }
    }

    /// `y[m, n] = x[m, 4096] . table^T` over the table's first `n` rows: the bf16 GEMV, or the EXL3 linear (which is exactly `n` wide)
    /// through `linear_rows` (calls of at most 128 rows, row invariant; `ws` holds its workspace).
    fn score(&self, x: &DevBuf, y: &mut DevBuf, m: usize, n: usize, ws: &mut DevBuf, st: &Stream) -> Result<(), ModelError> {
        match self {
            HeadTable::Bf16(b) => gemv_bf16(b, x, y, m, n, HIDDEN, st)?,
            HeadTable::Exl3(l) if l.n == n => linear_rows(l, x, y, m, Dtype::F32, Dtype::F32, ws, st)?,
            HeadTable::Exl3(l) => {
                return Err(ModelError::Config(format!("the EXL3 lm_head has {} output columns, the call scores {n}", l.n)))
            }
        }
        Ok(())
    }
}

/// Where the rows of a `lm_head` (or a restricted table) of `rows_have` rows sit in the `vocab` logits of this rank: whole, or this
/// rank's `head_row_cut` share; any other count is refused by name.
pub fn lm_head_shard(topo: TpTopology, vocab: usize, rows_have: usize) -> Result<Shard, String> {
    lm_head_shard_in(topo, vocab, rows_have, HEAD_ROW_UNIT)
}

/// [`lm_head_shard`] for a table cut in units of `unit` rows (an EXL3 `lm_head`: [`EXL3_HEAD_ROW_UNIT`]).
pub fn lm_head_shard_in(topo: TpTopology, vocab: usize, rows_have: usize, unit: usize) -> Result<Shard, String> {
    if rows_have == vocab {
        return Ok(Shard::Whole);
    }
    if topo.is_single() {
        return Err(format!("a single-rank model holds whole matrices: {rows_have} rows of {vocab}"));
    }
    if topo.rank >= topo.ranks {
        return Err(format!("rank {} is not in a group of {}", topo.rank, topo.ranks));
    }
    let (lo, hi) = head_row_cut_in(vocab, topo.rank, topo.ranks, unit);
    if rows_have != hi - lo {
        return Err(format!(
            "rank {} of {} holds {rows_have} rows of the {vocab}-row head, its share is [{lo}, {hi})",
            topo.rank, topo.ranks
        ));
    }
    Ok(Shard::Cols { lo, n_local: rows_have })
}

/// How a call of `m` rows over a `total`-row head table runs on a group of `topo.ranks`: scored whole on every rank (the single-node
/// GEMV, no collective) or cut (this rank's rows, the scatter and one allreduce). A function of the rank count, the table's row count,
/// whether the rank holds every row (`have == total`, which the shard plan fixes: the trunk's head is a slice, the restricted
/// sub-table whole), the plan's switch (`split_whole`, [`HeadSplit::lm_head`], exchanged at the rendezvous) and the call's rows `m`
/// (identical on every rank under the lockstep protocol): no buffer's size or kind enters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadPath {
    /// Every rank scores every row: no collective.
    Whole,
    /// This rank scores `[lo, lo + n_local)` and the group combines the rows with one allreduce. `view`: the table holds every row
    /// and the rank reads its rows as a sub-range of it (a held-whole table); else the rank's buffer is just its rows.
    Split { lo: usize, n_local: usize, view: bool },
}

/// Rows a call may have for a held-whole table to be cut: the window (the logits temp holds a window of rows at the rank's share). A
/// wider call, a draft call of many streams, is scored whole on every rank, the same way on every rank because `m` is the same.
pub const HEAD_SPLIT_MAX_ROWS: usize = WINDOW_ROWS;

/// The [`HeadPath`] of a call: see the type. Errors by name when the rank's rows are not its share of the table.
pub fn head_path(topo: TpTopology, total: usize, have: usize, m: usize, split_whole: bool) -> Result<HeadPath, String> {
    head_path_in(topo, total, have, m, split_whole, HEAD_ROW_UNIT)
}

/// [`head_path`] for a table cut in units of `unit` rows. An EXL3 table ([`EXL3_HEAD_ROW_UNIT`]) held whole is scored whole on every rank:
/// its columns are not a byte range of its trellis, so it is never cut as a view (a function of the table's form, the same on every rank).
pub fn head_path_in(
    topo: TpTopology,
    total: usize,
    have: usize,
    m: usize,
    split_whole: bool,
    unit: usize,
) -> Result<HeadPath, String> {
    if topo.is_single() {
        return Ok(HeadPath::Whole);
    }
    let split_whole = split_whole && unit == HEAD_ROW_UNIT;
    if have == total {
        // the rank holds every row: cut it only when the plan splits tables of this size, the switch is on, and the window holds the call
        if split_whole && head_splits(total, topo.ranks) && m <= HEAD_SPLIT_MAX_ROWS {
            let (lo, hi) = head_row_cut(total, topo.rank, topo.ranks);
            return Ok(HeadPath::Split { lo, n_local: hi - lo, view: true });
        }
        return Ok(HeadPath::Whole);
    }
    // a slice: the rank cannot score the rows it does not hold, so the group combines whatever the switch says
    match lm_head_shard_in(topo, total, have, unit)? {
        Shard::Cols { lo, n_local } => Ok(HeadPath::Split { lo, n_local, view: false }),
        Shard::Whole => Ok(HeadPath::Whole),
    }
}

/// `gb_cuda::nemotron_tp::scatter_cols_f32(src, dst, rows, n_local, n_total, col_off, st)`: writes `src` (f32 `[rows, n_local]`) into
/// the columns `[col_off, col_off + n_local)` of `dst` (f32 `[rows, n_total]`) and exact `-0.0` into every other column, in one fused
/// launch: the disjoint-support buffer the allreduce sums.
pub use gb_cuda::nemotron_tp::scatter_cols_f32;

fn temp<'a>(tmp: Option<&'a mut DevBuf>, rows: usize, n_local: usize, what: &str) -> Result<&'a mut DevBuf, ModelError> {
    match tmp {
        Some(t) if t.bytes() >= rows * n_local * 4 => Ok(t),
        Some(t) => Err(ModelError::Request(format!(
            "tensor parallelism: the {what} of {rows} rows x {n_local} columns needs {} bytes of temp, the model holds {}",
            rows * n_local * 4,
            t.bytes()
        ))),
        None => Err(ModelError::Config(format!("tensor parallelism: the {what} has no temp (the model was built for one rank)"))),
    }
}

/// `dst[rows, n_total] = layer(x[rows])` through the TP-exact path: the single-node call when the weight is whole on this rank, else
/// the local linear into `tmp`, the scatter and one allreduce. `dst` is `f32 [rows, n_total]`. `x_dtype` is the input's (the linear's
/// output is always f32).
#[allow(clippy::too_many_arguments)]
pub fn linear_tp(
    comm: &dyn TpComm,
    l: &Exl3Weights,
    x: &DevBuf,
    x_dtype: Dtype,
    dst: &mut DevBuf,
    tmp: Option<&mut DevBuf>,
    rows: usize,
    n_total: usize,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), ModelError> {
    // a single rank: the call the engine always made, whatever the shapes
    if comm.topology().is_single() {
        linear_rows(l, x, dst, rows, x_dtype, Dtype::F32, ws, st)?;
        return Ok(());
    }
    match shard(comm.topology(), n_total, l.n).map_err(ModelError::Config)? {
        Shard::Whole => linear_rows(l, x, dst, rows, x_dtype, Dtype::F32, ws, st)?,
        Shard::Cols { lo, n_local } => {
            let tmp = temp(tmp, rows, n_local, "linear")?;
            linear_rows(l, x, tmp, rows, x_dtype, Dtype::F32, ws, st)?;
            scatter_cols_f32(tmp, dst, rows, n_local, n_total, lo, st)?;
            comm.allreduce_sum_f32(dst, 0, rows * n_total, st)?;
        }
    }
    Ok(())
}

/// A lane-S P2 TP exchange lever's switch (A/B and fallback; the bits do not depend on it): `contract::p2tp_on`, the one place that
/// knows each lever's default, so the boot gate's lever label always says what runs.
fn p2tp_lever(var: &str) -> bool {
    super::contract::p2tp_on(var)
}

/// Lane S: MoE blocks run as two row halves (the y-overlap) in this process; the gates' proof that the halves ran.
static YOVERLAP_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// See `YOVERLAP_CALLS`.
pub fn yoverlap_calls() -> u64 {
    YOVERLAP_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn count_yoverlap() {
    YOVERLAP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Lane S: Mamba layers of a P2 prefill slice run as the head-group split (`arm_mamba::run_hg`) in this process; the gates' proof that
/// the split ran (a split that fell back to the whole in_proj would give the same bits). The first call also logs
/// `rexl3 event=p2tp_split_first kind=head rows=..`, as the router and shared splits do.
static HEAD_SPLIT_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// See `HEAD_SPLIT_CALLS`.
pub fn head_split_calls() -> u64 {
    HEAD_SPLIT_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn count_head_split(rows: usize) {
    if HEAD_SPLIT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
        eprintln!("rexl3 event=p2tp_split_first kind=head rows={rows}");
    }
}

/// One line of the lane-S path counters (router splits, shared splits, head splits, y-overlap half pairs) for the test logs.
pub fn p2tp_counters_line() -> String {
    let (r, sh) = gb_cuda::nemotron_moe_prefill::split_counters();
    counters_line(r, sh, head_split_calls(), yoverlap_calls())
}

/// [`p2tp_counters_line`]'s format (pure, for the tests).
pub fn counters_line(router: u64, shared: u64, head: u64, yoverlap: u64) -> String {
    format!("P2TP COUNTERS router_split={router} shared_split={shared} head_split={head} yoverlap={yoverlap}")
}

/// Lane S: the shared-expert split's switch (`docs/NEMOTRON_PREFILL_SPEED.md` 6.13; default `contract::SHARED_SPLIT_DEFAULT`).
pub use super::contract::SHARED_SPLIT_VAR;

/// Whether a rank of a group of `ranks` runs the shared-expert split, and so joins a SECOND communicator at boot: contract P2, two
/// ranks, a sharded group, not opted out ([`SHARED_SPLIT_VAR`]). Both ranks compare it at the Hello (the lever label's `S`, in the
/// prefill fact) and again before NCCL init (the bootstrap's length, `tp_boot::check_aux_offer`); it is NOT a plan flag.
pub fn shared_split_wanted(ranks: usize) -> bool {
    ranks == 2
        && super::tp_boot::sharded()
        && super::contract::current() == super::contract::PrefillContract::P2
        && p2tp_lever(SHARED_SPLIT_VAR)
}

/// What the memory plan charges a rank for the second communicator's NCCL buffers when [`shared_split_wanted`] (the boot measures the
/// real figure, `event=tp_comm2 mem_dev_gib=`, and flags `OVER_RESERVE` above this). PROVISIONAL until measured on the nodes.
pub const AUX_COMM_RESERVE_BYTES: u64 = 512 << 20;

/// The `in_proj` columns whose every consumer rounds to bf16 itself: z (`split_inproj` rounds it, `nemotron_glue.cu`) and xBC (`pre_conv`
/// reads `rbf(xBC)`, `commit_conv` stores `__float2bfloat16_rn(xBC)`, `nemotron_ssm.cu`); dt (the last 128 columns) feeds a softplus in
/// f32 and is exchanged as f32. (z 8,192 + xBC 10,240 = 18,432.)
pub const INPROJ_BF16_COLS: usize = 18_432;

/// Contract P2 under two-rank TP (lane S): `linear_tp` for the `in_proj` of a prefill slice, the exchange cut to what the consumers read:
/// the local columns as before, then ONE byte exchange of the rank's columns packed (z and xBC as bf16, rounded to nearest even exactly
/// as their consumers round; dt as f32) instead of a -0.0-filled f32 allreduce of the whole row, then the peer's columns widened into
/// place. This rank's own columns stay f32 bit-exact. Every value a consumer computes from `dst` is the same bits as with the
/// allreduce: a consumer of z or xBC rounds to bf16 first, and rn(widen(rn(x))) == rn(x). Bytes a row and rank: 18,432 / 18,944 instead
/// of 74,240 (the ring allreduce sends the whole row). Falls back to [`linear_tp`] when it does not apply.
#[allow(clippy::too_many_arguments)]
pub fn linear_tp_p2(
    comm: &dyn TpComm,
    l: &Exl3Weights,
    x: &DevBuf,
    x_dtype: Dtype,
    dst: &mut DevBuf,
    tmp: Option<&mut DevBuf>,
    pack: Option<&mut (DevBuf, DevBuf)>,
    split: Option<(&mut DevBuf, &mut DevBuf, &mut DevBuf)>,
    aux: Option<&(Stream, [gb_cuda::rt::Event; 3])>,
    rows: usize,
    n_total: usize,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<bool, ModelError> {
    let topo = comm.topology();
    let applies = topo.ranks == 2
        && comm.supports_swap()
        && n_total == splits::IN_PROJ_COLS
        && gb_cuda::nemotron_dense_mma::active(rows)
        && p2tp_lever("REXL3_NEMO_P2TP_INPROJ");
    let (Some(pack), true) = (pack, applies) else {
        return linear_tp(comm, l, x, x_dtype, dst, tmp, rows, n_total, ws, st).map(|_| false);
    };
    let Shard::Cols { lo, n_local } = shard(topo, n_total, l.n).map_err(ModelError::Config)? else {
        return linear_tp(comm, l, x, x_dtype, dst, tmp, rows, n_total, ws, st).map(|_| false);
    };
    let (plo, pn) = if lo == 0 { (n_local, n_total - n_local) } else { (0, lo) };
    let tmp = temp(tmp, rows, n_local, "linear")?;
    let (send, recv) = (&mut pack.0, &mut pack.1);
    let srow = gb_cuda::nemotron_tp::packed_row_bytes(n_local, lo, INPROJ_BF16_COLS);
    let rrow = gb_cuda::nemotron_tp::packed_row_bytes(pn, plo, INPROJ_BF16_COLS);
    let xrow = l.k * x_dtype.size();
    // overlap (REXL3_NEMO_P2TP_OVERLAP, default on): the first row half's exchange on the exchange stream while the second half
    // computes. Rows are independent under P2 (a), so the halves give the same bits as the whole slice in one call.
    let half =
        if rows >= 128 && p2tp_lever("REXL3_NEMO_P2TP_OVERLAP") { (rows / 2).next_multiple_of(64).min(rows) } else { rows };
    match aux {
        Some((ax, ev)) if half < rows => {
            let parts = [(0usize, half), (half, rows - half)];
            for (i, &(r0, n)) in parts.iter().enumerate() {
                let xs = x.subview(r0 * xrow, n * xrow);
                let mut ts = tmp.subview(r0 * n_local * 4, n * n_local * 4);
                linear_rows(l, &xs, &mut ts, n, x_dtype, Dtype::F32, ws, st)?;
                let mut ss = send.subview(r0 * srow, n * srow);
                gb_cuda::nemotron_tp::pack_cols(&ts, n, n_local, lo, INPROJ_BF16_COLS, &mut ss, st)?;
                // the exchange stream waits for this half's pack, then exchanges it while the main stream goes on
                gb_cuda::graph::fork(st, ax, &ev[i])?;
                comm.swap_bytes(send, r0 * srow, n * srow, recv, r0 * rrow, n * rrow, ax)?;
            }
            gb_cuda::graph::join(st, ax, &ev[2])?;
        }
        _ => {
            linear_rows(l, x, tmp, rows, x_dtype, Dtype::F32, ws, st)?;
            gb_cuda::nemotron_tp::pack_cols(tmp, rows, n_local, lo, INPROJ_BF16_COLS, send, st)?;
            comm.swap_bytes(send, 0, rows * srow, recv, 0, rows * rrow, st)?;
        }
    }
    // with the split's destinations: z, xBC and dt written straight (the fused kernel: `place_cols` + `split_inproj`, the same bits)
    if let (Some((z, xbc, dt)), true) = (split, p2tp_lever("REXL3_NEMO_P2TP_FUSESPLIT")) {
        gb_cuda::nemotron_tp::place_split(tmp, n_local, lo, recv, pn, plo, INPROJ_BF16_COLS, z, xbc, dt, rows, st)?;
        return Ok(true);
    }
    gb_cuda::nemotron_tp::place_cols(tmp, n_local, lo, recv, pn, plo, INPROJ_BF16_COLS, dst, rows, n_total, st)?;
    Ok(false)
}

/// The three views of one merged q, k, v buffer for `rows` rows, block by block: q `[rows, 4096]`, k `[rows, 256]`, v `[rows, 256]`.
fn qkv_views(qkv: &DevBuf, rows: usize) -> (DevBuf, DevBuf, DevBuf) {
    let (qb, kb) = (rows * splits::Q_COLS * 4, rows * splits::KV_COLS * 4);
    (qkv.subview(0, qb), qkv.subview(qb, kb), qkv.subview(qb + kb, kb))
}

/// An attention layer's q, k and v for `rows` rows of `x`. On a single rank (or with a shape kept whole) each is the single-node call
/// into `q`, `k`, `v` and `None` is returned; when all three are sliced they land block by block in the merged `qkv` buffer under ONE
/// allreduce and the views `(q, k, v)` into it are returned (the attention kernels read them in place of `bufs.q`, `bufs.k`, `bufs.v`).
///
/// Whether the three share one collective is a function of the weights' shapes (which the shard plan fixes: all three sliced), not of
/// the buffer: all three sliced and no merged buffer is a configuration error by name, never a quiet switch to three collectives on
/// one rank. At three ranks and more k and v have fewer tiles than ranks and are held whole (`tp_shard::sliceable`), so a sliced q
/// then runs on its own, with its own collective; whole k and v are the single-node call.
#[allow(clippy::too_many_arguments)]
pub fn qkv_tp(
    comm: &dyn TpComm,
    a: &AttnW,
    x: &DevBuf,
    bufs: (&mut DevBuf, &mut DevBuf, &mut DevBuf),
    qkv: Option<&mut DevBuf>,
    mut tmp: Option<&mut DevBuf>,
    rows: usize,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<Option<(DevBuf, DevBuf, DevBuf)>, ModelError> {
    let topo = comm.topology();
    let (q, k, v) = bufs;
    if topo.is_single() {
        linear_rows(&a.q, x, q, rows, Dtype::F32, Dtype::F32, ws, st)?;
        linear_rows(&a.k, x, k, rows, Dtype::F32, Dtype::F32, ws, st)?;
        linear_rows(&a.v, x, v, rows, Dtype::F32, Dtype::F32, ws, st)?;
        return Ok(None);
    }
    let sq = shard(topo, splits::Q_COLS, a.q.n).map_err(|e| ModelError::Config(format!("attention q: {e}")))?;
    let sk = shard(topo, splits::KV_COLS, a.k.n).map_err(|e| ModelError::Config(format!("attention k: {e}")))?;
    let sv = shard(topo, splits::KV_COLS, a.v.n).map_err(|e| ModelError::Config(format!("attention v: {e}")))?;
    let (Shard::Cols { lo: lq, .. }, Shard::Cols { lo: lk, .. }, Shard::Cols { lo: lv, .. }) = (sq, sk, sv) else {
        // a shape kept whole (or three of them): each linear on its own, which for a whole one is exactly the old call and for the
        // one sliced matrix its own scatter and collective
        linear_tp(comm, &a.q, x, Dtype::F32, q, tmp.as_deref_mut(), rows, splits::Q_COLS, ws, st)?;
        linear_tp(comm, &a.k, x, Dtype::F32, k, tmp.as_deref_mut(), rows, splits::KV_COLS, ws, st)?;
        linear_tp(comm, &a.v, x, Dtype::F32, v, tmp.as_deref_mut(), rows, splits::KV_COLS, ws, st)?;
        return Ok(None);
    };
    let los = [lq, lk, lv];
    let Some(qkv) = qkv else {
        return Err(ModelError::Config(
            "tensor parallelism: attention q, k and v are all sliced, which takes one merged buffer, and the model has none (the plan \
             and the allocation disagree; every rank must run the same plan)"
                .into(),
        ));
    };
    if qkv.bytes() < rows * QKV_COLS * 4 {
        return Err(ModelError::Request(format!(
            "tensor parallelism: q, k and v of {rows} rows need {} bytes, the merged buffer holds {}",
            rows * QKV_COLS * 4,
            qkv.bytes()
        )));
    }
    let (mut qv, mut kv, mut vv) = qkv_views(qkv, rows);
    for (l, view, n_total, lo) in [
        (&a.q, &mut qv, splits::Q_COLS, los[0]),
        (&a.k, &mut kv, splits::KV_COLS, los[1]),
        (&a.v, &mut vv, splits::KV_COLS, los[2]),
    ] {
        let tmp = temp(tmp.as_deref_mut(), rows, l.n, "attention projection")?;
        linear_rows(l, x, tmp, rows, Dtype::F32, Dtype::F32, ws, st)?;
        scatter_cols_f32(tmp, view, rows, l.n, n_total, lo, st)?;
    }
    comm.allreduce_sum_f32(qkv, 0, rows * QKV_COLS, st)?;
    Ok(Some((qv, kv, vv)))
}

/// The lm_head of `m` rows of `x` (f32 `[m, 4096]`) into the full logits `[m, vocab]`: the single-node GEMV when the table is scored
/// whole on this rank, else the GEMV on this rank's rows into `local` (`[m, n_local]`), the scatter into the zeroed logits and one
/// allreduce. A logit depends on its own weight row only (`nemotron_dense.cu`: the order of every operation depends on `k` and the
/// stripe, never on `m` or the column set), so a rank's logits are bit-identical to the same columns of the whole head's, at ANY row
/// cut.
///
/// Whole or cut is [`head_path`], a plan-level decision (the rank count, the rows of the table, the plan's `split_whole` and the call's
/// `m`). `split_whole`: a head held whole on a TP rank is cut into the rank's rows as a view of the buffer (the trunk's step: cuts its
/// read, one collective). The draft head passes it too (`REXL3_TP_HEAD`, default on): its restricted sub-table is whole on every rank,
/// each rank scores its rows and the logits are combined exactly (`-0.0` scatter and one allreduce of `[m, cols]`), so the top-k, the
/// id map and the keyed pick run on the identical full row on every rank. With it `false` the call stays the replicated,
/// collective-free one of the first bring-up. A head that is itself a rank's slice needs the collective whatever the flag says. A
/// table that the plan cuts but that is not a view of a block (`DevBuf::block()` is `None`), or a temp that is too short for it, is an
/// ERROR here, not a quiet whole GEMV (a rank that skipped the collective its peers run would hold them in it until the watchdog).
///
/// `table` is bf16 rows (the bf16 `lm_head`, the restricted sub-table) or an EXL3 `lm_head` (the MixedK pack's): the EXL3 head runs
/// through `linear_rows` (calls of at most 128 rows, row invariant, `ws` its workspace) and is cut at 128-column blocks (a rank's slice
/// is a valid EXL3 layer whose columns are the whole layer's, bit for bit: the decode plan of K = 4096 is the same at N = 131,072 and
/// at any slice of at least 64 blocks).
#[allow(clippy::too_many_arguments)]
pub fn lm_head_logits(
    comm: &dyn TpComm,
    table: HeadTable<'_>,
    x: &DevBuf,
    logits: &mut DevBuf,
    local: Option<&mut DevBuf>,
    m: usize,
    vocab: usize,
    split_whole: bool,
    ws: &mut DevBuf,
    st: &Stream,
) -> Result<(), ModelError> {
    let topo = comm.topology();
    if topo.is_single() {
        table.score(x, logits, m, vocab, ws, st)?;
        return Ok(());
    }
    let have = table.rows();
    let path = head_path_in(topo, vocab, have, m, split_whole, table.unit()).map_err(ModelError::Config)?;
    let (lo, n_local, held_whole) = match path {
        HeadPath::Whole => {
            table.score(x, logits, m, vocab, ws, st)?;
            return Ok(());
        }
        HeadPath::Split { lo, n_local, view } => (lo, n_local, view),
    };
    let lm_head = match table {
        HeadTable::Bf16(b) => b,
        // an EXL3 table is its rank's slice here (`head_path_in` never cuts one held whole)
        HeadTable::Exl3(_) => {
            let local = temp(local, m, n_local, "lm_head")?;
            table.score(x, local, m, n_local, ws, st)?;
            scatter_cols_f32(local, logits, m, n_local, vocab, lo, st)?;
            comm.allreduce_sum_f32(logits, 0, m * vocab, st)?;
            return Ok(());
        }
    };
    // the rank's rows: its own buffer, or (a table held whole) its rows as a view into the buffer, which needs a view of a block
    let view: Option<DevBuf> = if held_whole {
        if lm_head.block().is_none() {
            return Err(ModelError::Config(format!(
                "tensor parallelism: the {vocab}-row table is cut across {} ranks, but it is an allocation of its own, not a view of a \
                 block, so a rank cannot read its rows as a sub-range (allocate it with DevBuf::zeroed_view or the arena)",
                topo.ranks
            )));
        }
        Some(lm_head.subview(lo * HIDDEN * 2, n_local * HIDDEN * 2))
    } else {
        None
    };
    let w = view.as_ref().unwrap_or(lm_head);
    let local = temp(local, m, n_local, "lm_head")?;
    gemv_bf16(w, x, local, m, n_local, HIDDEN, st)?;
    scatter_cols_f32(local, logits, m, n_local, vocab, lo, st)?;
    comm.allreduce_sum_f32(logits, 0, m * vocab, st)?;
    Ok(())
}

/// The decode-path MoE block of `rows` rows (at most the scratch's): `out = moe(h)`. A single rank runs `forward_in` as it always did;
/// a TP rank runs `forward_in_tp`, whose hook allreduces the routed down projection's `y` (`[rows, 22, 1024]` f32: the other rank's
/// experts' pairs are exact zeros here, ours are exact zeros there) between the projection and the combine.
pub fn moe_decode(
    comm: &dyn TpComm,
    moe: &gb_cuda::nemotron_moe::NemotronMoe,
    s: &mut MoeScratch,
    h: &DevBuf,
    rows: usize,
    out: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    if comm.topology().is_single() {
        return moe.forward_in(s, h, rows, out, st);
    }
    moe.forward_in_tp(s, h, rows, out, st, &mut |y, elems, st| comm.allreduce_sum_f32(y, 0, elems, st))
}

/// Whether the large-row prefill MoE block runs under tensor parallelism (`PrefillMoe::forward_prefill_in_tp`). With it off a TP rank
/// would cut a wide slice into decode-path calls of its own width (the same bits, one read of the experts' weights per call).
pub const PREFILL_MOE_TP: bool = true;

/// Whether this group runs the large-row prefill MoE path ([`PREFILL_MOE_TP`] for a group of ranks).
pub fn prefill_moe_runs(comm: &dyn TpComm) -> bool {
    comm.topology().is_single() || PREFILL_MOE_TP
}

/// The large-row prefill MoE block, the same way (`PrefillMoe::forward_prefill_in_tp`).
pub fn moe_prefill(
    comm: &dyn TpComm,
    moe: &PrefillMoe,
    s: &mut PrefillScratch,
    h: &DevBuf,
    rows: usize,
    out: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    moe_prefill_with(comm, moe, s, None, None, None, h, rows, out, st, false)
}

/// [`moe_prefill`], on the tensor-core block (contract P2, lane S2) when `mma` is given: the same single-rank / expert-parallel split,
/// the same one reduce of `y` between the routed down projection and the combine (exact: disjoint support, as P1's).
///
/// `resid` (P2 only, needs `mma` and `moe.supports_prefill_mma_resid()`): `out` is the residual stream `x` and the block adds its
/// output into it in fc2's epilogue (`forward_prefill_mma_resid[_tp]`, `docs/NEMOTRON_PREFILL_SPEED.md` 6.11). Under TP the `+ x`
/// comes AFTER the cross-rank exchange of `y` (the reduce hook runs before the combine) and fc2 is replicated: every rank adds `x` to
/// the complete `mix`, never to a partial.
#[allow(clippy::too_many_arguments)]
pub fn moe_prefill_with(
    comm: &dyn TpComm,
    moe: &PrefillMoe,
    s: &mut PrefillScratch,
    mma: Option<&mut gb_cuda::nemotron_moe_prefill::mma::MmaScratch>,
    ypack: Option<&mut (DevBuf, DevBuf, DevBuf)>,
    bucket_ev: Option<&gb_cuda::rt::Event>,
    h: &DevBuf,
    rows: usize,
    out: &mut DevBuf,
    st: &Stream,
    resid: bool,
) -> Result<(), CudaError> {
    assert!(!resid || mma.is_some(), "the fused residual add needs the tensor-core block (contract P2)");
    // contract P2, two ranks, contiguous expert ownership: the MoE `y` exchange as a pair GATHER (`moe_pair_gather`)
    let topo = comm.topology();
    let gather_ok = topo.ranks == 2
        && comm.supports_swap()
        && tp_balance::expert_map_fingerprint() == 0
        && p2tp_lever("REXL3_NEMO_P2TP_YGATHER");
    let mma = match (mma, ypack) {
        (Some(m), Some(yp)) if gather_ok => {
            let (cnt, lists) = s.route_ptrs();
            let mut gather = |y: &mut DevBuf, _elems: usize, st: &Stream| -> Result<(), CudaError> {
                moe_pair_gather(comm, cnt, lists, rows, y, yp, bucket_ev, st)
            };
            // every absent expert's pair is the peer's, and the gather writes those rows: the block's zero fill for them is skipped
            gb_cuda::nemotron_moe_prefill::set_gather_fills_absent(moe.slow_experts() > 0);
            gb_cuda::nemotron_moe_prefill::set_after_bucket_event(bucket_ev);
            // the router row split (6.12): each rank computes its half of the rows' logits, one byte exchange makes them whole;
            // off by default since v0.1.1 (rule R-ROUTER), REXL3_NEMO_P2TP_ROUTER=1 opts in (part of the boot gate's lever label)
            let row_bytes = gb_cuda::nemotron_moe::EXPERTS * 4;
            let mut rx = |send: &DevBuf,
                          logits: &mut DevBuf,
                          own: std::ops::Range<usize>,
                          peer: std::ops::Range<usize>,
                          st: &Stream|
             -> Result<(), CudaError> {
                comm.swap_bytes(
                    send,
                    own.start * row_bytes,
                    own.len() * row_bytes,
                    logits,
                    peer.start * row_bytes,
                    peer.len() * row_bytes,
                    st,
                )
            };
            // the shared-expert split (6.13): its exchanges on the group's second communicator, which exists only when both ranks
            // agreed on it at boot (REXL3_NEMO_P2TP_SHARED=0 opts out)
            let aux = comm.aux();
            let mut sx = |send: &DevBuf, sb: usize, recv: &mut DevBuf, rb: usize, st: &Stream| -> Result<(), CudaError> {
                aux.expect("checked").swap_bytes(send, 0, sb, recv, 0, rb, st)
            };
            let router: Option<&mut gb_cuda::nemotron_moe_prefill::RouterExchange<'_>> =
                if p2tp_lever(super::contract::ROUTER_SPLIT_VAR) { Some(&mut rx) } else { None };
            let shared: Option<&mut gb_cuda::nemotron_moe_prefill::SharedExchange<'_>> =
                if aux.is_some() { Some(&mut sx) } else { None };
            let r = gb_cuda::nemotron_moe_prefill::with_router_split(topo.rank, router, || {
                gb_cuda::nemotron_moe_prefill::with_shared_split(topo.rank, shared, || {
                    if resid {
                        moe.forward_prefill_mma_resid_tp(s, m, h, rows, out, st, &mut gather)
                    } else {
                        moe.forward_prefill_mma_tp(s, m, h, rows, out, st, &mut gather)
                    }
                })
            });
            gb_cuda::nemotron_moe_prefill::set_after_bucket_event(None);
            gb_cuda::nemotron_moe_prefill::set_gather_fills_absent(false);
            // the gather's hook may have pointed the block's combine at the receive buffer: never past this call
            gb_cuda::nemotron_moe_prefill::set_combine_from(None);
            return r;
        }
        (m, _) => m,
    };
    match (mma, comm.topology().is_single()) {
        (None, true) => moe.forward_prefill_in(s, h, rows, out, st),
        (None, false) => {
            moe.forward_prefill_in_tp(s, h, rows, out, st, &mut |y, elems, st| comm.allreduce_sum_f32(y, 0, elems, st))
        }
        (Some(m), true) if resid => moe.forward_prefill_mma_resid(s, m, h, rows, out, st),
        (Some(m), true) => moe.forward_prefill_mma(s, m, h, rows, out, st),
        (Some(m), false) if resid => {
            moe.forward_prefill_mma_resid_tp(s, m, h, rows, out, st, &mut |y, elems, st| comm.allreduce_sum_f32(y, 0, elems, st))
        }
        (Some(m), false) => {
            moe.forward_prefill_mma_tp(s, m, h, rows, out, st, &mut |y, elems, st| comm.allreduce_sum_f32(y, 0, elems, st))
        }
    }
}

/// Contract P2 at two ranks (lane S): the MoE `y` [pairs, 1024] made whole on both ranks by a GATHER of the pairs each rank's experts own,
/// instead of a +0.0-filled allreduce: this rank packs the `y` rows of its own experts' pairs (expert order, list order: the bucket
/// outputs, identical on both ranks), one byte exchange, the peer's rows copied into their places. No arithmetic. The bits the combine
/// computes are the allreduce's: the gathered `y` equals the summed one except that an exact -0.0 stays -0.0 where `x + (+0.0)` made it
/// +0.0, and the combine (`acc` starts at +0.0, `acc + w*y`) maps both to the same result. About half the bytes of the ring allreduce.
#[allow(clippy::too_many_arguments)]
fn moe_pair_gather(
    comm: &dyn TpComm,
    cnt: usize,
    lists: usize,
    rows: usize,
    y: &mut DevBuf,
    yp: &mut (DevBuf, DevBuf, DevBuf),
    bucket_ev: Option<&gb_cuda::rt::Event>,
    st: &Stream,
) -> Result<(), CudaError> {
    let topo = comm.topology();
    let own = contiguous_range(gb_cuda::nemotron_moe::EXPERTS, topo.rank, 2);
    let peer = contiguous_range(gb_cuda::nemotron_moe::EXPERTS, 1 - topo.rank, 2);
    // the counts decide the message sizes: the host waits for the block's after-bucket event when it recorded one (the GEMMs stay
    // queued), else for the whole stream
    match bucket_ev {
        Some(ev) => ev.sync()?,
        None => st.sync()?,
    }
    let mut c = vec![0i32; gb_cuda::nemotron_moe::EXPERTS];
    // SAFETY: `cnt` is the prefill scratch's [512] i32 counts, alive for the call
    unsafe { gb_cuda::nemotron_tp::download_i32_raw(cnt, &mut c)? };
    let n = gb_cuda::nemotron_moe::LATENT;
    let own_n: usize = c[own.clone()].iter().map(|v| *v as usize).sum();
    let peer_n: usize = c[peer.clone()].iter().map(|v| *v as usize).sum();
    let (send, recv, map) = (&mut yp.0, &mut yp.1, &mut yp.2);
    // SAFETY: the bucket outputs of this call (list stride = rows) and y [rows * 22, n]; the packed buffers hold rows * 22 rows
    unsafe { gb_cuda::nemotron_tp::route_pairs(cnt, lists, rows, own.start, own.end, n, y, send, true, st)? };
    comm.swap_bytes(send, 0, own_n * n * 4, recv, 0, peer_n * n * 4, st)?;
    if p2tp_lever("REXL3_NEMO_P2TP_COMBINE") {
        // the block's combine reads the peer's rows from `recv` through the map (no unpack copy; the same arithmetic on the same
        // values: `nemotron_tp::combine_gather`); the caller clears the hook after the block call
        unsafe { gb_cuda::nemotron_tp::pair_map(cnt, lists, rows, peer.start, peer.end, map, st)? };
        gb_cuda::nemotron_moe_prefill::set_combine_from(Some((recv.ptr() as usize, map.ptr() as usize)));
    } else {
        unsafe { gb_cuda::nemotron_tp::route_pairs(cnt, lists, rows, peer.start, peer.end, n, y, recv, false, st)? };
    }
    Ok(())
}

/// One kind of collective a TP step issues: how many calls, and the f32 elements each one allreduces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Collective {
    pub what: &'static str,
    pub calls: usize,
    pub elements: usize,
}

impl Collective {
    /// Payload bytes of all the calls of this kind (f32).
    pub fn bytes(&self) -> usize {
        self.calls * self.elements * 4
    }
}

/// The collectives one step of `rows` rows issues over the layers of `pattern` (`M`, `*`, `E` per layer), as [`linear_tp`], [`qkv_tp`],
/// [`moe_decode`] and [`lm_head_logits`] issue them with every sharded shape sliced: `head_rows` is the rows the lm_head runs on (the
/// window's rows, one for a prefill slice that wants candidates, 0 for one that does not). The point of the list is the latency budget:
/// calls times the fabric's small-message latency, and the bytes at its bandwidth.
pub fn step_collectives(pattern: &str, rows: usize, head_rows: usize, vocab: usize) -> Vec<Collective> {
    let count = |c: char| pattern.chars().filter(|&x| x == c).count();
    let (m, a, e) = (count('M'), count('*'), count('E'));
    let moe = MOE_TOPK * MOE_LATENT;
    let mut v = vec![
        Collective { what: "mamba in_proj", calls: m, elements: rows * splits::IN_PROJ_COLS },
        Collective { what: "mamba out_proj", calls: m, elements: rows * splits::HIDDEN },
        Collective { what: "attention q+k+v", calls: a, elements: rows * QKV_COLS },
        Collective { what: "attention o", calls: a, elements: rows * splits::O_COLS },
        Collective { what: "moe routed y", calls: e, elements: rows * moe },
    ];
    if head_rows > 0 {
        v.push(Collective { what: "lm_head logits", calls: 1, elements: head_rows * vocab });
    }
    v.retain(|c| c.calls > 0);
    v
}

/// [`step_collectives`] for the weights a rank actually holds: a matrix that is whole on this rank (a shape TP-W kept replicated) has no
/// collective, the attention's q, k and v share one only when all three are sliced. This is what [`linear_tp`], [`qkv_tp`] and
/// [`lm_head_logits`] issue.
pub fn model_collectives(w: &NemotronWeights, topo: TpTopology, rows: usize, head_rows: usize, vocab: usize) -> Vec<Collective> {
    let mut tally: Vec<Collective> = Vec::new();
    let mut add = |what: &'static str, elements: usize| match tally.iter_mut().find(|c| c.what == what) {
        Some(c) => c.calls += 1,
        None => tally.push(Collective { what, calls: 1, elements }),
    };
    if topo.is_single() {
        return tally;
    }
    let cols = |total: usize, n: usize| matches!(shard(topo, total, n), Ok(Shard::Cols { .. }));
    for l in &w.layers {
        match &l.mixer {
            Mixer::Mamba(m) => {
                if cols(splits::IN_PROJ_COLS, m.in_proj.n) {
                    add("mamba in_proj", rows * splits::IN_PROJ_COLS);
                }
                if cols(splits::HIDDEN, m.out_proj.n) {
                    add("mamba out_proj", rows * splits::HIDDEN);
                }
            }
            Mixer::Attn(a) => {
                let (q, k, v) = (cols(splits::Q_COLS, a.q.n), cols(splits::KV_COLS, a.k.n), cols(splits::KV_COLS, a.v.n));
                if q && k && v {
                    add("attention q+k+v", rows * QKV_COLS);
                } else {
                    if q {
                        add("attention q", rows * splits::Q_COLS);
                    }
                    if k {
                        add("attention k", rows * splits::KV_COLS);
                    }
                    if v {
                        add("attention v", rows * splits::KV_COLS);
                    }
                }
                if cols(splits::O_COLS, a.o.n) {
                    add("attention o", rows * splits::O_COLS);
                }
            }
            Mixer::Moe(_) => add("moe routed y", rows * MOE_TOPK * MOE_LATENT),
        }
    }
    if head_rows > 0 && lm_head_rows_local(w, vocab, topo).is_some() {
        add("lm_head logits", head_rows * vocab);
    }
    tally
}

/// The collectives one step of `rows` rows issues under `plan` for the model `cfg` (`cfg.pattern`, `cfg.vocab`), straight from the
/// shard plan and the config: no weights, no device. This is what [`model_collectives`] reads off the weights a rank holds and what
/// the rank then issues (the three agree when the weights were loaded under the plan; a test asserts it). Unlike
/// [`step_collectives`], which prices every shape sliced, this follows the plan: a replicated matrix has no collective, q, k and v
/// share one only when all three are sliced, a matrix with fewer tiles than ranks is whole. Under a ring or a one-shot exchange the
/// call COUNT does not depend on the rank count; what changes with it is which matrices are sliced:
///
/// | ranks | default serving plan | calls at one row |
/// | --- | --- | ---: |
/// | 2 | `in_proj` and MoE `y` and the head | 40 + 40 + 1 = 81 |
/// | 3, 4, 8 | `in_proj` replicated (a slice below 64 tiles moves the K reduction) | 40 + 1 = 41 |
pub fn planned_collectives(plan: &ShardPlan, cfg: &OmniConfig, rows: usize, head_rows: usize) -> Vec<Collective> {
    let mut v: Vec<Collective> = Vec::new();
    if plan.is_single() {
        return v;
    }
    let count = |c: char| cfg.pattern.chars().filter(|&x| x == c).count();
    let (m, a, e) = (count('M'), count('*'), count('E'));
    let sliced = |role: Dense| -> bool {
        dense_shapes(cfg).iter().find(|(r, _, _)| *r == role).is_some_and(|&(_, _, n)| {
            let (lo, hi) = plan.columns(role, n);
            hi - lo < n
        })
    };
    let mut add = |what: &'static str, calls: usize, elements: usize| {
        if calls > 0 {
            v.push(Collective { what, calls, elements });
        }
    };
    if sliced(Dense::MambaIn) {
        add("mamba in_proj", m, rows * splits::IN_PROJ_COLS);
    }
    if sliced(Dense::MambaOut) {
        add("mamba out_proj", m, rows * splits::HIDDEN);
    }
    let (q, k, kv) = (sliced(Dense::AttnQ), sliced(Dense::AttnK), sliced(Dense::AttnV));
    if q && k && kv {
        add("attention q+k+v", a, rows * QKV_COLS);
    } else {
        if q {
            add("attention q", a, rows * splits::Q_COLS);
        }
        if k {
            add("attention k", a, rows * splits::KV_COLS);
        }
        if kv {
            add("attention v", a, rows * splits::KV_COLS);
        }
    }
    if sliced(Dense::AttnO) {
        add("attention o", a, rows * splits::O_COLS);
    }
    add("moe routed y", e, rows * MOE_TOPK * MOE_LATENT);
    if head_rows > 0 {
        add("lm_head logits", 1, head_rows * cfg.vocab);
    }
    v
}

/// The cost shape of one allreduce of `bytes` over `ranks` ranks, per rank, under a ring and under the one-shot exchange of the
/// cluster plan (`docs/CLUSTER_N_NODE_PLAN_2026-10-11.md` A8: every rank sends its buffer to every peer in one group and sums the
/// slots in rank order). Hops are latency terms (about 15 us each on the measured fabric), wire bytes the bandwidth term. A model:
/// nothing at R > 2 is measured on a fabric.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllreduceShape {
    pub ring_hops: usize,
    pub ring_wire_bytes: u64,
    pub oneshot_hops: usize,
    pub oneshot_wire_bytes: u64,
}

/// [`AllreduceShape`] of one allreduce of `bytes` over `ranks` (zeros for one rank).
pub fn allreduce_shape(ranks: usize, bytes: u64) -> AllreduceShape {
    if ranks <= 1 {
        return AllreduceShape { ring_hops: 0, ring_wire_bytes: 0, oneshot_hops: 0, oneshot_wire_bytes: 0 };
    }
    let r = ranks as u64;
    AllreduceShape {
        // reduce-scatter and all-gather: 2 (R - 1) hops, each rank sending 2 (R - 1) / R of the buffer
        ring_hops: 2 * (ranks - 1),
        ring_wire_bytes: (2 * (r - 1) * bytes).div_ceil(r),
        oneshot_hops: 1,
        oneshot_wire_bytes: (r - 1) * bytes,
    }
}

/// What a step's collectives add up to on a group of `ranks`: the number of calls, the payload (the bytes of the buffers reduced),
/// and the ring's and the one-shot's hops and per-rank wire bytes summed over the calls ([`allreduce_shape`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepTally {
    pub calls: usize,
    pub payload_bytes: u64,
    pub ring_hops: usize,
    pub ring_wire_bytes: u64,
    pub oneshot_hops: usize,
    pub oneshot_wire_bytes: u64,
}

impl StepTally {
    fn add(&mut self, calls: usize, bytes: u64, ranks: usize) {
        let s = allreduce_shape(ranks, bytes);
        self.calls += calls;
        self.payload_bytes += calls as u64 * bytes;
        self.ring_hops += calls * s.ring_hops;
        self.ring_wire_bytes += calls as u64 * s.ring_wire_bytes;
        self.oneshot_hops += calls * s.oneshot_hops;
        self.oneshot_wire_bytes += calls as u64 * s.oneshot_wire_bytes;
    }
}

/// The tally of a planned or modelled step ([`planned_collectives`], [`model_collectives`], [`step_collectives`]) on `ranks` ranks.
pub fn tally(list: &[Collective], ranks: usize) -> StepTally {
    let mut t = StepTally::default();
    for c in list {
        t.add(c.calls, (c.elements * 4) as u64, ranks);
    }
    t
}

/// The tally of calls a [`CountingComm`] recorded (`(f32 elements, byte offset)` each, [`CountingComm::take`]) on `ranks` ranks.
pub fn tally_of(calls: &[(usize, usize)], ranks: usize) -> StepTally {
    let mut t = StepTally::default();
    for &(elements, _) in calls {
        t.add(1, (elements * 4) as u64, ranks);
    }
    t
}

/// A [`TpComm`] that forwards to `inner` and records every allreduce as `(elements, byte offset)`: the collectives a step issues, as
/// the model issues them (for the latency budget and for the tests that pin the count).
pub struct CountingComm {
    inner: Arc<dyn TpComm>,
    log: Mutex<Vec<(usize, usize)>>,
}

impl CountingComm {
    pub fn new(inner: Arc<dyn TpComm>) -> Self {
        Self { inner, log: Mutex::new(Vec::new()) }
    }

    /// The calls since the last [`CountingComm::take`], as `(f32 elements, byte offset)`.
    pub fn take(&self) -> Vec<(usize, usize)> {
        std::mem::take(&mut *self.log.lock().unwrap())
    }

    /// The calls since the last take, tallied for the group's rank count ([`tally_of`]): calls, payload and the ring's and one-shot's
    /// hops and wire bytes, to set beside [`tally`] of the plan.
    pub fn take_tally(&self) -> StepTally {
        tally_of(&self.take(), self.inner.topology().ranks)
    }
}

impl TpComm for CountingComm {
    fn topology(&self) -> TpTopology {
        self.inner.topology()
    }

    fn allreduce_sum_f32(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
        self.log.lock().unwrap().push((count, off_bytes));
        self.inner.allreduce_sum_f32(buf, off_bytes, count, st)
    }
}

/// Rows of the staging buffers [`gather_head_rows`] sums a rank's rows in (bf16 `[2048, 4096]` and f32 `[2048, 4096]`: 50 MB).
const GATHER_ROWS: usize = 2048;

/// The rows `ids` (token ids, ascending, distinct) of the trunk's `lm_head`, as a bf16 `[ids.len(), 4096]` table that every rank holds
/// whole (the restricted draft head's sub-table, which both ranks run replicated). A bf16 head's rows are copied bit for bit: a whole
/// head is the plain row copies; a rank's half is gathered through the group: a rank copies the ids it holds into a bf16 chunk whose other
/// rows are `-0.0`, widens it (exact), one allreduce per chunk makes it whole (`x + (-0.0) == x`, signed zeros too), and the narrowing back
/// to bf16 is exact (every value is a bf16). An EXL3 head (the MixedK pack's; decision D2 of v0.1.2) is DECODED: each rank decodes the
/// rows of the ids it holds on the host (`gb_exl3::decode_strip`, float64, rounded once to bf16 by `f64_to_bf16`) and the rows are
/// gathered the same way, so the table is the decoded weights, the same bits on every rank. (A draft only proposes: what the target
/// verifies, and so every reply, cannot change.)
/// Load time only: both ranks make the same calls in the same order.
pub fn gather_head_rows(
    comm: &dyn TpComm,
    lm_head: &HeadLinear,
    vocab: usize,
    ids: &[u32],
    st: &Stream,
) -> Result<DevBuf, ModelError> {
    let row = HIDDEN * 2;
    let topo = comm.topology();
    match lm_head {
        HeadLinear::Bf16(lm_head) => match lm_head_shard(topo, vocab, lm_head.bytes() / row).map_err(ModelError::Config)? {
            Shard::Whole => {
                // a view of a block of its own: `lm_head_logits` cuts a whole table into a rank's half with `subview`, which an allocation
                // of its own cannot do (it would silently compute the whole table on every rank: no split, no error)
                let mut table = DevBuf::zeroed_view(ids.len().max(1) * row)?;
                // runs of consecutive ids are one copy each
                let mut at = 0;
                while at < ids.len() {
                    let mut end = at + 1;
                    while end < ids.len() && ids[end] == ids[end - 1] + 1 {
                        end += 1;
                    }
                    table.copy_range_from(at * row, lm_head, ids[at] as usize * row, (end - at) * row, st)?;
                    at = end;
                }
                Ok(table)
            }
            Shard::Cols { lo, n_local } => {
                let src_row = |id: u32| (id as usize).checked_sub(lo).filter(|&r| r < n_local);
                gather_through_group(comm, lm_head, &src_row, ids, st)
            }
        },
        HeadLinear::Exl3(l) => {
            let (lo, n_local) = match lm_head_shard_in(topo, vocab, l.n, EXL3_HEAD_ROW_UNIT).map_err(ModelError::Config)? {
                Shard::Whole => (0, vocab),
                Shard::Cols { lo, n_local } => (lo, n_local),
            };
            // the ids this rank holds, decoded into a bf16 table of their own (in ascending id order: a run of ids is a run of rows)
            let own: Vec<u32> = ids.iter().copied().filter(|&id| (lo..lo + n_local).contains(&(id as usize))).collect();
            let host = decode_head_rows(l, lo, &own)?;
            let mut dec = DevBuf::zeroed_view(own.len().max(1) * row)?;
            if !host.is_empty() {
                dec.upload(&host)?;
            }
            if own.len() == ids.len() && n_local == vocab {
                // the whole head on this device: the decoded table is the sub-table (a view of a block, as `lm_head_logits` needs)
                return Ok(dec);
            }
            let src_row = |id: u32| own.binary_search(&id).ok();
            let table = gather_through_group(comm, &dec, &src_row, ids, st)?;
            Ok(table)
        }
    }
}

/// The bf16 rows of the token ids `own` (ascending, all inside `[lo, lo + l.n)`) of an EXL3 `lm_head` slice `l` whose first column is
/// token `lo`: column `v - lo` of the layer's effective weight (`gb_exl3::decode_strip`, the host decode the references use), each value
/// rounded once to bf16. `[own.len() * 4096]` bf16 bits. The strips are decoded on up to 16 threads; the result does not depend on how
/// many.
fn decode_head_rows(l: &Exl3Weights, lo: usize, own: &[u32]) -> Result<Vec<u16>, ModelError> {
    if own.is_empty() {
        return Ok(Vec::new());
    }
    let bad = |what: String| ModelError::Config(format!("the EXL3 lm_head's restricted rows: {what}"));
    if l.k != HIDDEN || l.codebook != gb_exl3::Codebook::Mul1 {
        return Err(bad(format!("a [{}, {}] {:?} layer, not a [{HIDDEN}, n] mul1 one", l.k, l.n, l.codebook)));
    }
    let bits = gb_exl3::Bits::from_twice((2 * l.bits) as u8).ok_or_else(|| bad(format!("{} bits", l.bits)))?;
    let bufs = l.buffers();
    let get = |name: &str| bufs.iter().find(|(n, _)| *n == name).map(|(_, b)| *b).ok_or_else(|| bad(format!("no `{name}`")));
    let (tb, sb, vb) = (get("trellis")?, get("suh")?, get("svh")?);
    let mut trellis = vec![0i16; tb.bytes() / 2];
    tb.download_range(0, &mut trellis)?;
    let f16s = |b: &DevBuf| -> Result<Vec<f64>, ModelError> {
        let mut v = vec![0u16; b.bytes() / 2];
        b.download_range(0, &mut v)?;
        Ok(v.into_iter().map(gb_exl3::f16::f16_to_f64).collect())
    };
    let (suh, svh) = (f16s(sb)?, f16s(vb)?);
    let (kt, nt) = (l.k / 16, l.n / 16);
    if trellis.len() != kt * nt * bits.tile_words() || svh.len() != l.n || suh.len() != l.k {
        return Err(bad("the buffers do not match the layer's shape".into()));
    }
    Ok(decode_rows_host(&trellis, kt, nt, bits, &suh, &svh, lo, own))
}

/// The pure host part of [`decode_head_rows`]: the bf16 rows of the token ids `own` (ascending, inside `[lo, lo + 16 nt)`) of the mul1
/// EXL3 layer `trellis` `[kt, nt, 16 * bits]` with its scales, a row being the layer's column `id - lo`, each value rounded once.
#[allow(clippy::too_many_arguments)]
pub fn decode_rows_host(
    trellis: &[i16],
    kt: usize,
    nt: usize,
    bits: gb_exl3::Bits,
    suh: &[f64],
    svh: &[f64],
    lo: usize,
    own: &[u32],
) -> Vec<u16> {
    let k = kt * 16;
    let table = gb_exl3::codebook(gb_exl3::Codebook::Mul1);
    // the strips the ids fall in, as (strip, first id index, ids): `own` ascends, so a strip's ids are consecutive rows of the table
    let mut strips: Vec<(usize, usize, usize)> = Vec::new();
    for (i, &id) in own.iter().enumerate() {
        let s = (id as usize - lo) / 128;
        match strips.last_mut() {
            Some((last, _, count)) if *last == s => *count += 1,
            _ => strips.push((s, i, 1)),
        }
    }
    let mut out = vec![0u16; own.len() * k];
    if own.is_empty() {
        return out;
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 16).min(strips.len());
    let per = strips.len().div_ceil(threads);
    // each job: its strips and their (disjoint, consecutive) rows of the table
    let mut jobs: Vec<Vec<(usize, usize, &mut [u16])>> = (0..threads).map(|_| Vec::new()).collect();
    let mut rest: &mut [u16] = &mut out;
    for (j, &(strip, first, count)) in strips.iter().enumerate() {
        let (mine, tail) = std::mem::take(&mut rest).split_at_mut(count * k);
        rest = tail;
        jobs[j / per].push((strip, first, mine));
    }
    std::thread::scope(|sc| {
        for job in jobs {
            let (trellis, suh, svh, table) = (&trellis, &suh, &svh, &table);
            sc.spawn(move || {
                for (strip, first, rows) in job {
                    let d = gb_exl3::decode_strip(trellis, kt, nt, bits, table, suh, svh, strip);
                    for (r, dst) in rows.chunks_mut(k).enumerate() {
                        let col = (own[first + r] as usize - lo) % 128;
                        for (i, v) in dst.iter_mut().enumerate() {
                            *v = gb_exl3::f16::f64_to_bf16(d[i * 128 + col]);
                        }
                    }
                }
            });
        }
    });
    out
}

/// The rows `ids` of a head table gathered through the group: `src` holds this rank's rows (`src_row(id)`: the row of `id` in `src`, or
/// `None` when another rank holds it). Per chunk of [`GATHER_ROWS`] ids: this rank's rows at their places in a bf16 chunk whose other rows
/// are `-0.0`, widened (exact), one allreduce makes it whole (`x + (-0.0) == x`, signed zeros too), and the narrowing back to bf16 is exact.
fn gather_through_group(
    comm: &dyn TpComm,
    src: &DevBuf,
    src_row: &dyn Fn(u32) -> Option<usize>,
    ids: &[u32],
    st: &Stream,
) -> Result<DevBuf, ModelError> {
    let row = HIDDEN * 2;
    let cap = GATHER_ROWS.min(ids.len().max(1));
    // a view of a block of its own, like the whole head's table (the draft head's call then cuts its half of it)
    let mut table = DevBuf::zeroed_view(ids.len().max(1) * row)?;
    let mut bf = DevBuf::zeroed_view(cap * row)?;
    let mut stage = DevBuf::zeroed_view(cap * HIDDEN * 4)?;
    // the rows another rank holds are -0.0 (bf16 0x8000), the additive identity: x + (-0.0) == x for every x, a weight of -0.0
    // included (a +0.0 filler would turn it into +0.0)
    let neg_zero = DevBuf::from_slice(&vec![0x8000u16; cap * HIDDEN])?;
    let mut base = 0;
    while base < ids.len() {
        let n = GATHER_ROWS.min(ids.len() - base);
        let chunk = &ids[base..base + n];
        // this rank's rows of the chunk at their place, the identity elsewhere
        bf.copy_range_from(0, &neg_zero, 0, n * row, st)?;
        let mut at = 0;
        while at < n {
            let Some(first) = src_row(chunk[at]) else {
                at += 1;
                continue;
            };
            let mut end = at + 1;
            while end < n && chunk[end] == chunk[end - 1] + 1 && src_row(chunk[end]) == Some(first + (end - at)) {
                end += 1;
            }
            bf.copy_range_from(at * row, src, first * row, (end - at) * row, st)?;
            at = end;
        }
        widen_bf16_f32(&bf, n * HIDDEN, &mut stage, st)?;
        comm.allreduce_sum_f32(&mut stage, 0, n * HIDDEN, st)?;
        // the widened rows summed whole: narrowing is exact, and the chunk joins the table
        f32_to_bf16(&stage, &mut bf, n * HIDDEN, st)?;
        table.copy_range_from(base * row, &bf, 0, n * row, st)?;
        base += n;
    }
    st.sync()?;
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An EXL3 table is cut at 128-column blocks and never as a view: held whole on a rank it is scored whole (no collective), a rank's
    /// slice is combined through the group, and a slice that is not the rank's 128-block share is refused by name.
    #[test]
    fn an_exl3_head_is_cut_in_128_column_blocks_and_never_as_a_view() {
        let u = EXL3_HEAD_ROW_UNIT;
        let v = splits::VOCAB;
        for split_whole in [false, true] {
            assert_eq!(head_path_in(r(0), v, v, 4, split_whole, u), Ok(HeadPath::Whole), "held whole: whole on every rank");
        }
        assert_eq!(head_path_in(r(1), v, 65_536, 64, true, u), Ok(HeadPath::Split { lo: 65_536, n_local: 65_536, view: false }));
        let three = TpTopology { rank: 1, ranks: 3 };
        assert_eq!(head_path_in(three, v, 43_648, 1, true, u), Ok(HeadPath::Split { lo: 43_776, n_local: 43_648, view: false }));
        // the bf16 cut's share (43,692 rows) is not an EXL3 rank's
        assert!(head_path_in(TpTopology { rank: 0, ranks: 3 }, v, 43_692, 1, true, u)
            .unwrap_err()
            .contains("its share is [0, 43776)"));
        assert_eq!(
            lm_head_shard_in(TpTopology { rank: 2, ranks: 3 }, v, 43_648, u),
            Ok(Shard::Cols { lo: 87_424, n_local: 43_648 })
        );
        // the bf16 rule is unchanged
        assert_eq!(head_path_in(r(1), 49_152, 49_152, 4, true, HEAD_ROW_UNIT), head_path(r(1), 49_152, 49_152, 4, true));
    }

    /// D2: the restricted rows of an EXL3 head are its decoded columns (`gb_exl3::decode_weight`), each rounded once to bf16, for any id
    /// set and slice offset, whatever the threads.
    #[test]
    fn the_restricted_rows_of_an_exl3_head_are_its_decoded_columns_in_bf16() {
        let (kt, nt) = (8, 48); // K 128, N 768: six strips
        let bits = gb_exl3::Bits::from_twice(16).unwrap();
        let mut s = 0x0123_4567_89AB_CDEFu64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let trellis: Vec<i16> = (0..kt * nt * bits.tile_words()).map(|_| next() as i16).collect();
        let suh: Vec<f64> = (0..kt * 16).map(|_| ((next() % 2000) as f64 - 1000.0) / 3000.0).collect();
        let svh: Vec<f64> = (0..nt * 16).map(|_| ((next() % 2000) as f64 - 1000.0) / 7000.0).collect();
        let (k, n) = (kt * 16, nt * 16);
        let w = gb_exl3::decode_weight(&trellis, kt, nt, bits, &gb_exl3::codebook(gb_exl3::Codebook::Mul1), &suh, &svh);
        let lo = 1000; // a rank's slice starting at token 1000
        let own: Vec<u32> = [0usize, 1, 2, 127, 128, 300, 301, 302, 640, 767].iter().map(|&c| (lo + c) as u32).collect();
        let rows = decode_rows_host(&trellis, kt, nt, bits, &suh, &svh, lo, &own);
        assert_eq!(rows.len(), own.len() * k);
        for (r, &id) in own.iter().enumerate() {
            let col = id as usize - lo;
            for i in 0..k {
                assert_eq!(rows[r * k + i], gb_exl3::f16::f64_to_bf16(w[i * n + col]), "id {id} [{i}]");
            }
        }
        assert!(decode_rows_host(&trellis, kt, nt, bits, &suh, &svh, lo, &[]).is_empty());
    }

    fn r(rank: usize) -> TpTopology {
        TpTopology { rank, ranks: 2 }
    }

    #[test]
    fn the_head_split_has_a_path_counter_in_the_counters_line() {
        // the gates count a lever's leg only when its counter is above zero; the head split (6.10) had none before rc2d
        assert_eq!(counters_line(48, 36, 40, 2), "P2TP COUNTERS router_split=48 shared_split=36 head_split=40 yoverlap=2");
        let before = head_split_calls();
        count_head_split(4096);
        count_head_split(100);
        // other tests of this process may count too (the counter is process-wide): at least our two
        assert!(head_split_calls() >= before + 2);
        assert!(p2tp_counters_line().contains(" head_split="), "{}", p2tp_counters_line());
    }

    #[test]
    fn the_head_split_is_both_parts_unless_the_variable_says_otherwise() {
        let both = HeadSplit { experts: true, lm_head: true };
        for v in [None, Some(""), Some("split"), Some("both"), Some(" split ")] {
            assert_eq!(HeadSplit::parse(v), both, "{v:?}");
        }
        assert_eq!(HeadSplit::parse(Some("replicate")), HeadSplit { experts: false, lm_head: false });
        assert_eq!(HeadSplit::parse(Some("experts")), HeadSplit { experts: true, lm_head: false });
        assert_eq!(HeadSplit::parse(Some("lm_head")), HeadSplit { experts: false, lm_head: true });
    }

    fn rr(rank: usize, ranks: usize) -> TpTopology {
        TpTopology { rank, ranks }
    }

    const PACK: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");

    fn config() -> OmniConfig {
        OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap()
    }

    #[test]
    fn a_whole_weight_is_whole_on_every_rank_and_a_single_rank_only_holds_whole_ones() {
        assert_eq!(shard(TpTopology::SINGLE, 4096, 4096), Ok(Shard::Whole));
        assert_eq!(shard(r(0), 4096, 4096), Ok(Shard::Whole));
        assert_eq!(shard(r(1), 18560, 18560), Ok(Shard::Whole));
        assert!(shard(TpTopology::SINGLE, 4096, 2048).is_err());
    }

    #[test]
    fn the_even_splits_are_the_two_halves() {
        for (n_total, half) in [(splits::HIDDEN, 2048), (splits::Q_COLS, 2048), (splits::KV_COLS, 128), (splits::VOCAB, 65536)] {
            assert_eq!(shard(r(0), n_total, half), Ok(Shard::Cols { lo: 0, n_local: half }));
            assert_eq!(shard(r(1), n_total, half), Ok(Shard::Cols { lo: half, n_local: half }));
        }
    }

    #[test]
    fn the_in_proj_cut_is_uneven_at_a_tile_boundary_and_derived() {
        // the constant that was `IN_PROJ_CUT` is the end of rank 0's range under the one cut rule
        assert_eq!(splits::in_proj_cut(2), 9216);
        assert_eq!(shard(r(0), splits::IN_PROJ_COLS, 9216), Ok(Shard::Cols { lo: 0, n_local: 9216 }));
        assert_eq!(shard(r(1), splits::IN_PROJ_COLS, 9344), Ok(Shard::Cols { lo: 9216, n_local: 9344 }));
        // each side's slice is a whole number of 128-column tiles, and together they are the whole output
        assert_eq!(9216 % 128, 0);
        assert_eq!(9344 % 128, 0);
        assert_eq!(9216 + 9344, splits::IN_PROJ_COLS);
        // the other rank's share, a half that is not a share, and a share of another rank count are refused by name
        assert!(shard(r(0), splits::IN_PROJ_COLS, 9344).is_err());
        assert!(shard(r(1), splits::IN_PROJ_COLS, 9216).is_err());
        assert!(shard(r(0), splits::IN_PROJ_COLS, 9280).is_err());
        assert!(shard(rr(0, 3), 4096, 2048).is_err());
        // three ranks: the 32 tiles of a hidden-wide matrix are 10 | 11 | 11
        let w = |rank| match shard(rr(rank, 3), 4096, [1280, 1408, 1408][rank]) {
            Ok(Shard::Cols { lo, n_local }) => (lo, n_local),
            other => panic!("{other:?}"),
        };
        assert_eq!((w(0), w(1), w(2)), ((0, 1280), (1280, 1408), (2688, 1408)));
    }

    #[test]
    fn shards_of_any_rank_count_tile_each_output_once_and_a_matrix_of_too_few_tiles_is_never_cut() {
        for ranks in [2usize, 3, 4, 5, 8] {
            for total in [splits::IN_PROJ_COLS, splits::HIDDEN, splits::Q_COLS, splits::O_COLS] {
                let mut at = 0;
                for rank in 0..ranks {
                    let (lo, hi) = column_range(total, rank, ranks);
                    let got = shard(rr(rank, ranks), total, hi - lo);
                    assert_eq!(got, Ok(Shard::Cols { lo, n_local: hi - lo }), "{total} at {rank}/{ranks}");
                    assert_eq!(lo, at, "{total} at {rank}/{ranks}: contiguous");
                    assert_eq!((lo % 128, hi % 128), (0, 0));
                    at = hi;
                    // whole is still whole, and another rank's width is not this rank's
                    assert_eq!(shard(rr(rank, ranks), total, total), Ok(Shard::Whole));
                }
                assert_eq!(at, total, "{total} at {ranks} ranks: every column once");
            }
            // k and v: 256 columns are 2 tiles: cut at two ranks, never at three or more (no rank may hold zero columns)
            for rank in 0..ranks {
                let r_ = rr(rank, ranks);
                if ranks == 2 {
                    assert_eq!(shard(r_, splits::KV_COLS, 128), Ok(Shard::Cols { lo: 128 * rank, n_local: 128 }));
                } else {
                    for have in [0usize, 128] {
                        assert!(shard(r_, splits::KV_COLS, have).is_err(), "{ranks} ranks, rank {rank}: {have} columns of 256");
                    }
                    assert_eq!(shard(r_, splits::KV_COLS, 256), Ok(Shard::Whole));
                }
            }
        }
    }

    #[test]
    fn whole_or_cut_of_a_head_table_is_the_same_on_every_rank_and_never_reads_a_buffer() {
        use super::super::tp_shard::{head_row_cut, head_splits};
        for ranks in [2usize, 3, 4, 5, 8] {
            for total in [splits::VOCAB, 49_152, 62_017, 1000, 128 * ranks - 1, 128 * ranks] {
                for m in [1usize, 5, 64, 65, 170] {
                    for split_whole in [true, false] {
                        // a table held whole on every rank (the restricted sub-table, a whole trunk head)
                        let paths: Vec<HeadPath> =
                            (0..ranks).map(|rank| head_path(rr(rank, ranks), total, total, m, split_whole).unwrap()).collect();
                        let cut = paths.iter().filter(|p| matches!(p, HeadPath::Split { .. })).count();
                        assert!(cut == 0 || cut == ranks, "{ranks} ranks, {total} rows, m {m}: ranks disagree on the collective");
                        let want_cut = split_whole && head_splits(total, ranks) && m <= HEAD_SPLIT_MAX_ROWS;
                        assert_eq!(cut == ranks, want_cut, "{ranks} ranks, {total} rows, m {m}, switch {split_whole}");
                        // the rows the ranks score tile the table once
                        if want_cut {
                            let mut at = 0;
                            for (rank, p) in paths.iter().enumerate() {
                                let HeadPath::Split { lo, n_local, view } = *p else { unreachable!() };
                                assert!(view);
                                assert_eq!((lo, lo + n_local), head_row_cut(total, rank, ranks));
                                assert_eq!(lo, at);
                                at = lo + n_local;
                            }
                            assert_eq!(at, total);
                        }
                        // a slice (what the trunk's head is under the serving plan) is always combined, whatever the switch and m
                        for rank in 0..ranks {
                            let (lo, hi) = head_row_cut(total, rank, ranks);
                            let p = head_path(rr(rank, ranks), total, hi - lo, m, split_whole);
                            if hi - lo == total {
                                continue;
                            }
                            assert_eq!(p, Ok(HeadPath::Split { lo, n_local: hi - lo, view: false }));
                        }
                    }
                }
            }
        }
        // a count that is neither whole nor the rank's share is refused by name; one rank is whole
        assert!(head_path(rr(0, 3), splits::VOCAB, 65_536, 1, true).is_err());
        assert_eq!(head_path(TpTopology::SINGLE, splits::VOCAB, splits::VOCAB, 1, true), Ok(HeadPath::Whole));
        // two ranks: the restricted table's halves and the trunk's, exactly what the verified engine ran
        assert_eq!(head_path(r(1), 49_152, 49_152, 4, true), Ok(HeadPath::Split { lo: 24_576, n_local: 24_576, view: true }));
        assert_eq!(head_path(r(0), splits::VOCAB, 65_536, 64, true), Ok(HeadPath::Split { lo: 0, n_local: 65_536, view: false }));
    }

    #[test]
    fn the_planned_collectives_follow_the_plan_at_every_rank_count() {
        let cfg = config();
        let calls = |plan: &ShardPlan, rows, head_rows| -> Vec<(&'static str, usize)> {
            let mut v: Vec<_> = planned_collectives(plan, &cfg, rows, head_rows).iter().map(|c| (c.what, c.calls)).collect();
            v.sort();
            v
        };
        let total = |v: &[(&'static str, usize)]| v.iter().map(|c| c.1).sum::<usize>();
        let serving = |ranks| {
            // the default serving plan: attention replicated (`REXL3_TP_ATTN` unset), the head's experts split
            ShardPlan::for_config(rr(0, ranks), &cfg).replicating(&[Dense::AttnQ, Dense::AttnK, Dense::AttnV, Dense::AttnO])
        };
        // two ranks: 40 in_proj + 40 MoE + 1 head, the 81 of the measured two-node runs
        let two = calls(&serving(2), 1, 1);
        assert_eq!(two, vec![("lm_head logits", 1), ("mamba in_proj", 40), ("moe routed y", 40)]);
        assert_eq!(total(&two), 81);
        // three ranks and more: the in_proj is replicated (a slice below 64 tiles moves the K reduction), so only the MoE y and the head
        for ranks in [3usize, 4, 5, 8] {
            let v = calls(&serving(ranks), 1, 1);
            assert_eq!(v, vec![("lm_head logits", 1), ("moe routed y", 40)], "{ranks} ranks");
            assert_eq!(total(&v), 41);
            // a prefill slice that wants no candidates runs no head
            assert_eq!(total(&calls(&serving(ranks), 64, 0)), 40);
        }
        // attention sliced (`REXL3_TP_ATTN=slice`): two ranks 8 + 8 more, merged q+k+v; three ranks and more: q and o only (k and v are
        // 2 tiles: not cut), so no merged buffer and q has its own collective
        let attn2 = ShardPlan::for_config(rr(0, 2), &cfg);
        let v = calls(&attn2, 1, 1);
        assert!(v.contains(&("attention q+k+v", 8)) && v.contains(&("attention o", 8)));
        assert_eq!(total(&v), 97);
        for ranks in [3usize, 4, 8] {
            let v = calls(&ShardPlan::for_config(rr(0, ranks), &cfg), 1, 1);
            assert!(v.contains(&("attention q", 8)) && v.contains(&("attention o", 8)), "{ranks}: {v:?}");
            assert!(
                !v.iter().any(|c| c.0 == "attention k" || c.0 == "attention v" || c.0 == "attention q+k+v"),
                "{ranks}: {v:?}"
            );
            assert_eq!(total(&v), 41 + 16);
        }
        // everything sliced (the exactness test's plan): in_proj at three ranks is sliced here (the plan says exact only above 64
        // tiles, which a test of the other slices may want anyway), k and v stay whole
        let all3 = ShardPlan::slice_everything(rr(0, 3));
        let v = calls(&all3, 1, 1);
        assert!(v.contains(&("mamba in_proj", 40)) && v.contains(&("mamba out_proj", 40)) && v.contains(&("attention q", 8)));
        assert!(!v.iter().any(|c| c.0 == "attention k" || c.0 == "attention v"));
        // a single rank issues none
        assert!(planned_collectives(&ShardPlan::single(), &cfg, 1, 1).is_empty());
    }

    #[test]
    fn the_planned_collectives_are_what_the_weights_widths_imply() {
        // the CPU image of `model_collectives`: derive each matrix's local width from the plan, run it through `shard` as the arms do
        let cfg = config();
        for ranks in [2usize, 3, 4, 5, 8] {
            for attn in [false, true] {
                let mut plan = ShardPlan::for_config(rr(2.min(ranks - 1), ranks), &cfg);
                if !attn {
                    plan = plan.replicating(&[Dense::AttnQ, Dense::AttnK, Dense::AttnV, Dense::AttnO]);
                }
                let topo = plan.topo;
                let held = |role: Dense, total: usize| {
                    let (lo, hi) = plan.columns(role, total);
                    matches!(shard(topo, total, hi - lo), Ok(Shard::Cols { .. }))
                };
                let (m, a, e) = (40usize, 8usize, 40usize);
                let mut want = 0;
                want += m * usize::from(held(Dense::MambaIn, splits::IN_PROJ_COLS));
                want += m * usize::from(held(Dense::MambaOut, splits::HIDDEN));
                let (q, k, v) = (
                    held(Dense::AttnQ, splits::Q_COLS),
                    held(Dense::AttnK, splits::KV_COLS),
                    held(Dense::AttnV, splits::KV_COLS),
                );
                want += a * if q && k && v { 1 } else { usize::from(q) + usize::from(k) + usize::from(v) };
                want += a * usize::from(held(Dense::AttnO, splits::O_COLS));
                want += e + 1;
                let got: usize = planned_collectives(&plan, &cfg, 1, 1).iter().map(|c| c.calls).sum();
                assert_eq!(got, want, "{ranks} ranks, attention sliced {attn}");
            }
        }
    }

    /// Prints the per-step collective table of `docs/NEMOTRON_TP.md` ("R > 2 sharding (as built)"): calls, payload and the ring's and
    /// one-shot's hops and per-rank wire bytes for a 1-row and a 5-row step, in the three plan modes, at 2, 3, 4, 5 and 8 ranks.
    #[test]
    fn the_collectives_per_step_by_rank_count_and_mode() {
        let cfg = config();
        let modes: [(&str, Vec<Dense>, bool); 3] = [
            ("serving (attention replicated)", Dense::ALL[2..].to_vec(), false),
            ("REXL3_TP_ATTN=slice", vec![], false),
            ("REXL3_TP_INPROJ=replicate", Dense::ALL[2..].to_vec(), true),
        ];
        eprintln!("| ranks | mode | calls | payload 1 row (MB) | payload 5 rows (MB) | ring hops | one-shot hops | ring wire/rank 1 row (MB) | one-shot wire/rank 1 row (MB) |");
        eprintln!("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        let mut by = std::collections::BTreeMap::new();
        for ranks in [2usize, 3, 4, 5, 8] {
            for (name, replicated, in_proj_rep) in &modes {
                let mut plan = ShardPlan::for_config(rr(0, ranks), &cfg).replicating(replicated);
                if *in_proj_rep {
                    plan = plan.replicating(&[Dense::MambaIn]);
                }
                let one = planned_collectives(&plan, &cfg, 1, 1);
                let five = planned_collectives(&plan, &cfg, 5, 5);
                let (t1, t5) = (tally(&one, ranks), tally(&five, ranks));
                eprintln!(
                    "| {ranks} | {name} | {} | {:.2} | {:.2} | {} | {} | {:.2} | {:.2} |",
                    t1.calls,
                    t1.payload_bytes as f64 / 1e6,
                    t5.payload_bytes as f64 / 1e6,
                    t1.ring_hops,
                    t1.oneshot_hops,
                    t1.ring_wire_bytes as f64 / 1e6,
                    t1.oneshot_wire_bytes as f64 / 1e6
                );
                assert_eq!(t5.payload_bytes, 5 * t1.payload_bytes, "payload is linear in the rows");
                assert_eq!(t5.calls, t1.calls);
                by.insert((ranks, *name), (t1.calls, t1.payload_bytes));
            }
        }
        // the verified plan and its rank-count variants
        assert_eq!(by[&(2, "serving (attention replicated)")], (81, 7_098_368));
        for ranks in [3usize, 4, 5, 8] {
            assert_eq!(by[&(ranks, "serving (attention replicated)")].0, 41, "{ranks} ranks");
            assert_eq!(by[&(ranks, "REXL3_TP_INPROJ=replicate")].0, 41);
        }
        assert_eq!(by[&(2, "REXL3_TP_INPROJ=replicate")].0, 41, "two ranks with in_proj replicated by the switch: the same 41");
        assert_eq!(by[&(2, "REXL3_TP_ATTN=slice")].0, 97);
    }

    #[test]
    fn the_boot_line_names_every_dense_decision_and_the_cuts() {
        if [ATTN_ENV, INPROJ_ENV, HEAD_ENV].iter().any(|k| std::env::var_os(k).is_some()) {
            eprintln!("not run: the environment names a TP layout switch");
            return;
        }
        let cfg = config();
        let two = shard_plan_line(2, &cfg);
        for want in [
            "event=tp_shard_plan ranks=2",
            "mamba_in_proj=sliced[9216|9344]",
            "mamba_out_proj=replicated(reason=\"the K reduction moves: rank 0's 16 tiles run 16 K splits x 4 warps, the whole 32 tiles 8 x 4\")",
            "attn_q_proj=replicated(reason=\"REXL3_TP_ATTN default",
            "experts=[0..256|256..512]",
            "lm_head_rows=[0..65536|65536..131072]",
        ] {
            assert!(two.contains(want), "{want}\nin {two}");
        }
        let three = shard_plan_line(3, &cfg);
        for want in [
            "ranks=3",
            "mamba_in_proj=replicated(reason=\"the K reduction moves: rank 0's 48 tiles run 4 K splits x 4 warps, the whole 145 tiles 1 x 8\")",
            "attn_k_proj=replicated(reason=\"2 tiles for 3 ranks: a rank would hold no column\")",
            "experts=[0..170|170..341|341..512]",
            "lm_head_rows=[0..43692|43692..87384|87384..131072]",
        ] {
            assert!(three.contains(want), "{want}\nin {three}");
        }
    }

    #[test]
    fn the_allreduce_shape_and_a_steps_tally_per_rank_count() {
        assert_eq!(
            allreduce_shape(1, 1 << 20),
            AllreduceShape { ring_hops: 0, ring_wire_bytes: 0, oneshot_hops: 0, oneshot_wire_bytes: 0 }
        );
        // MoE y at one row: 22 x 1024 f32 = 90,112 bytes
        let y = (22 * 1024 * 4) as u64;
        let two = allreduce_shape(2, y);
        assert_eq!((two.ring_hops, two.ring_wire_bytes, two.oneshot_hops, two.oneshot_wire_bytes), (2, y, 1, y));
        let four = allreduce_shape(4, y);
        assert_eq!((four.ring_hops, four.oneshot_hops, four.oneshot_wire_bytes), (6, 1, 3 * y));
        assert_eq!(four.ring_wire_bytes, (6 * y).div_ceil(4));
        let eight = allreduce_shape(8, y);
        assert_eq!((eight.ring_hops, eight.oneshot_hops), (14, 1));
        // the 81 collectives of the serving plan at two ranks: 8,032,256 - ... bytes of payload, the same at any rank count it is issued
        let cfg = config();
        let plan = ShardPlan::for_config(r(0), &cfg).replicating(&[Dense::AttnQ, Dense::AttnK, Dense::AttnV, Dense::AttnO]);
        let t = tally(&planned_collectives(&plan, &cfg, 1, 1), 2);
        assert_eq!(t.calls, 81);
        assert_eq!(t.payload_bytes, 40 * 18560 * 4 + 40 * 22 * 1024 * 4 + 131072 * 4);
        assert_eq!(t.ring_hops, 81 * 2, "one ring hop pair a collective at two ranks");
        assert_eq!(t.oneshot_hops, 81);
        let t8 = tally(&planned_collectives(&ShardPlan::for_config(rr(0, 8), &cfg).replicating(&Dense::ALL), &cfg, 1, 1), 8);
        assert_eq!(t8.calls, 41);
        assert_eq!(t8.ring_hops, 41 * 14, "a ring of eight is 14 hops a collective");
        assert_eq!(t8.oneshot_hops, 41, "the one-shot exchange is one hop whatever the rank count");
    }

    #[test]
    fn a_step_of_the_88_layer_trunk_issues_137_collectives_of_8_mb_at_one_row() {
        const PACK: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");
        let cfg = super::super::config::OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap();
        let one = step_collectives(&cfg.pattern, 1, 1, cfg.vocab);
        assert_eq!(one.iter().map(|c| c.calls).sum::<usize>(), 40 * 2 + 8 * 2 + 40 + 1);
        assert_eq!(one.iter().map(|c| c.bytes()).sum::<usize>(), 8_032_256);
        // the bytes are linear in the rows: five rows (a verify window) are five times as many
        let five = step_collectives(&cfg.pattern, 5, 5, cfg.vocab);
        assert_eq!(five.iter().map(|c| c.calls).sum::<usize>(), 137);
        assert_eq!(five.iter().map(|c| c.bytes()).sum::<usize>(), 5 * 8_032_256);
        // a prefill slice that wants no candidates runs no lm_head
        assert_eq!(step_collectives(&cfg.pattern, 64, 0, cfg.vocab).iter().map(|c| c.calls).sum::<usize>(), 136);
        assert_eq!(step_collectives("", 4, 4, 131072).iter().map(|c| c.calls).sum::<usize>(), 1);
    }

    #[test]
    fn the_merged_qkv_buffer_is_block_by_block() {
        assert_eq!(QKV_COLS, 4608);
        // 16-byte aligned block starts at any row count (the kernels read float4)
        for rows in 1..=64usize {
            let (qb, kb) = (rows * splits::Q_COLS * 4, rows * splits::KV_COLS * 4);
            assert_eq!(qb % 16, 0);
            assert_eq!((qb + kb) % 16, 0);
            assert_eq!(qb + 2 * kb, rows * QKV_COLS * 4);
        }
    }
}
