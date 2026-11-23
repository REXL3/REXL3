//! A GLM-5.3-Flash model as ONE RANK of a tensor-parallel group (`docs/GLM_TP.md`): whole-expert ownership of the routed experts
//! (`docs/CLUSTER_FAMILY_PORTING.md` 3.5.1), everything else replicated.
//!
//! **What changes on a rank.** Only the routed experts of a MoE layer. The router runs on every rank (replicated weights, identical picks
//! and fp16 gate weights); each rank runs the pairs stage of the experts it owns (`MoeExperts::run_pairs` / `run_prefill_pairs`, `-0.0` in
//! every pair row it does not compute), the ranks sum the pair buffer `y_pairs` `[rows * topk, hidden]` f32 in place over the group's
//! collective (equal count on every rank, no flush to zero), and EVERY rank runs the unchanged slot combine over the sum with the shared
//! expert added last (`moe::combine_pairs`). With disjoint support the sum is the single-node pair buffer bit for bit, and the combine is
//! the single-node combine (the batched path's kernel; the decode path fuses the same `fmaf` chain into `down_combine`), so every rank
//! leaves the single-node `mixer_out` (proved per layer by `gb-cuda`'s `tests/moe_owned.rs`).
//!
//! **What does not change.** The dense MLPs, the KDA and MLA/DSA mixers, the shared expert, the router and the head run as on one node. The
//! NextN layer's routed experts (lane L12) are owned and exchanged like a trunk layer's, in every forward of that layer (`mtp::mtp_block`,
//! `docs/GLM_TP.md` 9). A model without [`GlmModel::enable_tp`] never reads any of this.
//!
//! **Graphs and branches (lane E, Phase 1).** With a capturable communicator (`TpComm::capturable`: NCCL) a rank's decode windows go
//! through the graph cache like one node's, the exchange captured inside the window's graph, and the shared-expert and MLA side branches
//! run as on one node (`REXL3_GLM_OVERLAP`). Each collective is still issued on the model's stream in plan order, the same order on every
//! rank, whether eagerly or from a replay: the side streams carry no collective, and the shared expert's output is joined into the main
//! stream right before the combine that reads it (after the exchange, which it overlaps). The rules of `docs/NEMOTRON_TP.md` 4: a rank
//! captures only after it has run a collective eagerly (NCCL connects its peers on the first: [`GlmTp::graphs_ready`]); the graph cache
//! is a field before this one in `GlmModel`, so its graphs drop before the communicator (which the boot's `GROUP_COMM` also holds); the
//! graph key is a function of the call's batch, so both ranks capture and replay the same shapes in the same order. A capture that fails
//! on a rank is fatal there (`graphs.rs`): the ranks never issue one window's collectives by different paths. `REXL3_GLM_TP_GRAPHS=0`
//! (or `REXL3_GLM_GRAPHS=0`) keeps a rank eager on one stream, as before; the setting is the Hello fact `graphs` (with `overlap` and
//! `graph_at_first`), so ranks that disagree refuse to start. The loopback communicator synchronises inside every call: graphs off.
//!
//! **Bank composition does not matter here.** A layer's bank may hold every expert (the whole pack loaded on each rank: the table
//! [`OwnScratch::masked`] gives it the rank's share) or only the owned ones (`from_region_owned`, the reduced loader: the bank's own
//! `local_of`); the exchange point is the summed `y_pairs` either way.

use super::GlmModel;
use crate::error::ModelError;
use crate::families::glm::alloc;
use crate::families::glm::device::{FfnDevice, MixerDevice, MtpMoe};
use crate::families::nemotron_h_omni::tp::TpComm;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::exl3_split::PlannedExl3;
use gb_cuda::kda;
use gb_cuda::moe::{
    self, ExpertOwnership, MoeDims, MoeLayer, MoeScratch, MoeSubbankLayer, OwnScratch, SharedExpert, SharedScratch,
};
use gb_cuda::moe_prefill::PrefillScratch;
use gb_cuda::{DevBuf, Stream};
use std::sync::Arc;

/// `REXL3_GLM_TP_GRAPHS`: `0` / `off` keeps a rank's windows eager on one stream with no layer branches (the behaviour before lane E's
/// Phase 1); anything else (and unset) lets a rank with a capturable communicator run them as one node does (`REXL3_GLM_GRAPHS` too).
pub const TP_GRAPHS_ENV: &str = "REXL3_GLM_TP_GRAPHS";

/// Whether a rank with a communicator that is `capturable` runs its windows through graphs: both switches on (the Hello fact `graphs`).
pub fn rank_graphs(capturable: bool) -> bool {
    let off = |v: &str| matches!(std::env::var(v).as_deref(), Ok("0" | "off" | "false" | "no"));
    capturable && !off(TP_GRAPHS_ENV) && !off(super::graphs::ENV)
}

/// Lane P step (c): the routed experts of a P2 prompt chunk on a rank (the parts' swap).
mod p2;
pub(in crate::families::glm::model) use p2::P2Experts;
mod p2_split;

/// The rank's collective and its share of the routed experts.
pub struct GlmTp {
    comm: Arc<dyn TpComm>,
    own: ExpertOwnership,
    os: OwnScratch,
    /// The windows of this rank may be captured ([`rank_graphs`] of the communicator, at `enable_tp`).
    graphs: bool,
    /// Bytes this rank put through the collective (the pair buffers), and how many exchanges (issued by the host: eagerly, or into a
    /// graph being captured; a replay adds none).
    pub exchanged_bytes: u64,
    pub exchanges: u64,
    /// Under the P2 prefill contract: the parts' exchange of a P2 prompt chunk's MoE layers (`p2.rs`).
    p2: Option<p2::P2Exchange>,
    /// Of `exchanged_bytes`, those the shared-expert split's partials added (decode windows only: a prefill chunk's exchange keeps its bytes).
    pub sh_exchanged_bytes: u64,
    /// The replicated weights this rank splits with its peer (lane E Phase 2, [`GlmModel::enable_head_split`]); `None`: nothing split.
    pub(super) split: Option<HeadSplit>,
}

impl GlmTp {
    /// May this rank capture a window now: graphs are on for it and it has issued a collective eagerly (the first window of every
    /// shape runs eagerly too). The same answer on every rank: the same calls make the same exchanges.
    pub(super) fn graphs_ready(&self) -> bool {
        self.graphs && self.exchanges > 0
    }

    /// The decode-window path (1..=`MAX_ROWS` rows; picks in `s`): pairs, exchange, `join` (the main stream waits for a shared expert
    /// that ran on a side stream: the combine is its one reader), combine into `out`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn experts(
        &mut self,
        m: &MoeLayer,
        x: &DevBuf,
        rows: usize,
        s: &mut MoeScratch,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        st: &Stream,
        join: &mut dyn FnMut() -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        let d = m.dims;
        m.experts.run_pairs(x, Dtype::F16, rows, s, &mut self.os, st)?;
        self.exchange(&mut s.y_pairs, rows * d.topk * d.hidden, st)?;
        join()?;
        moe::combine_pairs(&s.y_pairs, &s.pick, &s.wts, shared, out, rows, &d, st)?;
        Ok(())
    }

    /// The batched path (a prefill chunk above `MAX_ROWS` rows, or one that takes snapshots inside itself; picks in `s`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn experts_big(
        &mut self,
        m: &MoeLayer,
        x: &DevBuf,
        rows: usize,
        s: &mut PrefillScratch,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let d = m.dims;
        m.experts.run_prefill_pairs(x, Dtype::F16, rows, s, &mut self.os, st)?;
        self.exchange(&mut s.y_pairs, rows * d.topk * d.hidden, st)?;
        moe::combine_pairs(&s.y_pairs, &s.pick, &s.wts, shared, out, rows, &d, st)?;
        Ok(())
    }

    /// A layer whose routed experts are subbanks (a mixed-width pack, or this rank's share loaded alone): every subbank's pairs into
    /// one pair buffer (`-0.0` in every pair row no subbank of this rank holds), the exchange, `join` (as [`GlmTp::experts`]), the
    /// combine. Decode window.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn subbanks(
        &mut self,
        m: &MoeSubbankLayer,
        x: &DevBuf,
        rows: usize,
        s: &mut MoeScratch,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        st: &Stream,
        join: &mut dyn FnMut() -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        let d = m.dims;
        m.experts.run_pairs(x, Dtype::F16, rows, s, &mut self.os, st)?;
        self.exchange(&mut s.y_pairs, rows * d.topk * d.hidden, st)?;
        join()?;
        moe::combine_pairs(&s.y_pairs, &s.pick, &s.wts, shared, out, rows, &d, st)?;
        Ok(())
    }

    /// [`GlmTp::subbanks`] on the batched path (`sd`: the decode scratch, which a split-width subbank's windows run through).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn subbanks_big(
        &mut self,
        m: &MoeSubbankLayer,
        x: &DevBuf,
        rows: usize,
        s: &mut PrefillScratch,
        sd: &mut MoeScratch,
        shared: Option<&DevBuf>,
        out: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let d = m.dims;
        m.experts.run_prefill_pairs_windowed(x, Dtype::F16, rows, s, sd, &mut self.os, st)?;
        self.exchange(&mut s.y_pairs, rows * d.topk * d.hidden, st)?;
        moe::combine_pairs(&s.y_pairs, &s.pick, &s.wts, shared, out, rows, &d, st)?;
        Ok(())
    }

    /// Whether the shared-expert split ([`ShSplit`], `sh`) runs a call of `rows` rows: a decode window (`window`) no wider than the split's
    /// widest. Both ranks give the same answer for the same call (it is a function of the call), so they exchange the same counts.
    pub(super) fn shared_split_on(&self, window: bool, rows: usize) -> bool {
        window && self.split.as_ref().and_then(|h| h.sh.as_ref()).is_some_and(|s| rows <= s.max_rows)
    }

    /// The shared-expert split, a window's first half (on the stream the shared expert runs on: the side stream of a graphed window): this
    /// rank's down partials into the tail of the pair buffer `y_pairs` (`[sk][rows][hidden]` f32 after the `rows x topk` pair rows), `-0.0`
    /// in the peer's splits. The routed experts write the head of the same buffer meanwhile; the exchange ([`GlmTp::experts_sh`],
    /// [`GlmTp::subbanks_sh`]) waits for both.
    pub(super) fn shared_partials(
        &mut self,
        shared: &SharedExpert,
        x: &DevBuf,
        rows: usize,
        y_pairs: &DevBuf,
        topk: usize,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let sp =
            self.split.as_mut().and_then(|h| h.sh.as_mut()).ok_or_else(|| ModelError::Config("no shared-expert split".into()))?;
        let (h, sk) = (shared.hidden, sp.sk);
        let (base, slot) = (rows * topk * h * 4, rows * h * 4);
        let (s0, s1) = sp.splits;
        let (p0, p1) = if s0 == 0 { (s1, sk) } else { (0, s0) };
        // the peer's splits: the additive identity, so the all-reduce returns the peer's partials there bit for bit
        let mut peer = y_pairs.subview(base + p0 * slot, (p1 - p0) * slot);
        gb_cuda::nemotron_tp::scatter_cols_f32(&sp.ws, &mut peer, (p1 - p0) * rows, 0, h, 0, st)?;
        let mut z = y_pairs.subview(base, sk * slot);
        shared.forward_split_partials(x, Dtype::F16, rows, sp.splits, &mut z, s0 * slot, &mut sp.scratch, &mut sp.ws, st)?;
        Ok(())
    }

    /// [`GlmTp::experts`] with the shared-expert split: the pairs, `join` (the shared expert's partials, on a side stream, are in the pair
    /// buffer's tail: the exchange reads them), ONE exchange of the pairs and the partials, the finish of the partials into `shared_out`,
    /// the combine with it added last when `add_shared`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn experts_sh(
        &mut self,
        m: &MoeLayer,
        x: &DevBuf,
        rows: usize,
        s: &mut MoeScratch,
        shared: &SharedExpert,
        shared_out: &mut DevBuf,
        add_shared: bool,
        out: &mut DevBuf,
        st: &Stream,
        join: &mut dyn FnMut() -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        let d = m.dims;
        m.experts.run_pairs(x, Dtype::F16, rows, s, &mut self.os, st)?;
        self.exchange_sh(&d, shared, rows, &mut s.y_pairs, shared_out, st, join)?;
        moe::combine_pairs(&s.y_pairs, &s.pick, &s.wts, add_shared.then_some(&*shared_out), out, rows, &d, st)?;
        Ok(())
    }

    /// [`GlmTp::subbanks`] with the shared-expert split (as [`GlmTp::experts_sh`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn subbanks_sh(
        &mut self,
        m: &MoeSubbankLayer,
        x: &DevBuf,
        rows: usize,
        s: &mut MoeScratch,
        shared: &SharedExpert,
        shared_out: &mut DevBuf,
        add_shared: bool,
        out: &mut DevBuf,
        st: &Stream,
        join: &mut dyn FnMut() -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        let d = m.dims;
        m.experts.run_pairs(x, Dtype::F16, rows, s, &mut self.os, st)?;
        self.exchange_sh(&d, shared, rows, &mut s.y_pairs, shared_out, st, join)?;
        moe::combine_pairs(&s.y_pairs, &s.pick, &s.wts, add_shared.then_some(&*shared_out), out, rows, &d, st)?;
        Ok(())
    }

    /// The exchange of a window under the shared-expert split: `join`, then the pairs and the partials in one all-reduce, then the finish.
    #[allow(clippy::too_many_arguments)]
    fn exchange_sh(
        &mut self,
        d: &MoeDims,
        shared: &SharedExpert,
        rows: usize,
        y_pairs: &mut DevBuf,
        shared_out: &mut DevBuf,
        st: &Stream,
        join: &mut dyn FnMut() -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        let sk = self
            .split
            .as_ref()
            .and_then(|h| h.sh.as_ref())
            .map(|s| s.sk)
            .ok_or_else(|| ModelError::Config("no shared-expert split".into()))?;
        join()?;
        self.exchange(y_pairs, rows * (d.topk + sk) * d.hidden, st)?;
        self.sh_exchanged_bytes += (rows * sk * d.hidden * 4) as u64;
        let z = y_pairs.subview(rows * d.topk * d.hidden * 4, sk * rows * d.hidden * 4);
        shared.finish_split(&z, rows, shared_out, st)?;
        Ok(())
    }

    fn exchange(&mut self, y: &mut DevBuf, count: usize, st: &Stream) -> Result<(), ModelError> {
        self.comm.allreduce_sum_f32(y, 0, count, st)?;
        self.exchanged_bytes += count as u64 * 4;
        self.exchanges += 1;
        Ok(())
    }

    pub fn ownership(&self) -> &ExpertOwnership {
        &self.own
    }

    /// `(rank, ranks)` of the collective.
    pub fn place(&self) -> (usize, usize) {
        let t = self.comm.topology();
        (t.rank, t.ranks)
    }
}

impl GlmModel {
    /// Make this model rank `comm.topology().rank` of a tensor-parallel group that owns the routed experts `own` of every MoE layer of the
    /// trunk. Every layer's bank must hold every expert (it is masked to `own`) or exactly `own` (a reduced loader's rank bank); the NextN
    /// layer's too, when loaded (its draft calls are forwarded: `tp_mtp`). Graphs stay on only when [`rank_graphs`] says so for this
    /// communicator (the exchange is then captured inside the window's graph; otherwise they are switched off for good), and the prefix
    /// cache must be absent (its calls are not forwarded). Call it before the rank serves (a slot's state is not touched: the next `begin_slot` starts it as on one node).
    pub fn enable_tp(&mut self, comm: Arc<dyn TpComm>, own: ExpertOwnership) -> Result<(), ModelError> {
        if self.tp.is_some() {
            return Err(ModelError::Config("this model is already a rank of a tensor-parallel group".into()));
        }
        let p2 = self.prefill == crate::families::glm::contract::PrefillContract::P2;
        if p2 {
            self.check_p2_tp(comm.as_ref(), &own)?;
        }
        // a rank may hold a snapshot pool: each rank snapshots and restores its own state between chunks (`docs/GLM_TP.md` 4.2); the pool's layout
        // is checked against the rank's KDA pools at every snapshot and restore (`SnapPool::check_dims`), and a split rank takes no in-chunk cut
        let dims = crate::families::glm::alloc::moe_dims(&self.cfg);
        if own.experts != dims.experts || own.local() == 0 {
            return Err(ModelError::Config(format!(
                "an expert ownership over {} experts with {} owned, for a model of {} experts",
                own.experts,
                own.local(),
                dims.experts
            )));
        }
        for (l, layer) in self.dev.layers.iter().enumerate() {
            // subbanks hold exactly the experts they were cut for: those must be this rank's share (a mixed layer loaded whole at one
            // rank is refused here: it would compute every expert on every rank)
            if let FfnDevice::MoeSubbanks { layer: m, .. } = &layer.ffn {
                let o = m.experts.ownership();
                if *o != own {
                    return Err(ModelError::Config(format!(
                        "layer {l}'s routed experts are subbanks of {} experts, not this rank's {} (a mixed-width pack's rank loads its \
                         share alone: map it with the rank's ownership)",
                        o.local(),
                        own.local()
                    )));
                }
            }
            if let FfnDevice::Moe { layer: m, .. } = &layer.ffn {
                match m.experts.ownership() {
                    None => {}
                    Some(o) if *o == own => {}
                    Some(o) => {
                        return Err(ModelError::Config(format!(
                            "layer {l} holds {} experts of another ownership than this rank's {}",
                            o.local(),
                            own.local()
                        )))
                    }
                }
            }
        }
        // the NextN layer (`--spec mtp|auto`): its experts are owned like a trunk layer's (subbanks of exactly this rank's share, or the whole
        // bank, masked), and its pairs are exchanged in every forward of the layer (`mtp::mtp_block`, `docs/GLM_TP.md` 9)
        if let Some(w) = self.dev.mtp.as_ref() {
            match &w.moe {
                MtpMoe::Subbanks(m) if *m.experts.ownership() != own => {
                    return Err(ModelError::Config(format!(
                        "the NextN layer's routed experts are subbanks of {} experts, not this rank's {} (map the pack with the rank's \
                         ownership)",
                        m.experts.ownership().local(),
                        own.local()
                    )))
                }
                MtpMoe::Single(m) => match m.experts.ownership() {
                    None => {}
                    Some(o) if *o == own => {}
                    Some(o) => {
                        return Err(ModelError::Config(format!(
                            "the NextN layer holds {} experts of another ownership than this rank's {}",
                            o.local(),
                            own.local()
                        )))
                    }
                },
                _ => {}
            }
        }
        let os = OwnScratch::masked(&dims, self.s.rows, &own)?;
        let graphs = rank_graphs(comm.capturable()) && self.graphs_on();
        if !graphs {
            self.set_graphs(false);
        }
        self.clear_graphs()?;
        let p2 = if p2 { Some(p2::P2Exchange::new(self.s.rows, dims.hidden)?) } else { None };
        self.tp = Some(GlmTp { comm, own, os, graphs, exchanged_bytes: 0, exchanges: 0, sh_exchanged_bytes: 0, p2, split: None });
        Ok(())
    }

    /// The rank's state, when this model is one.
    pub fn tp(&self) -> Option<&GlmTp> {
        self.tp.as_ref()
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The head split (lane E, GLM Phase 2 item 5): a rank computes half the heads of the replicated mixers and half the vocabulary

/// `REXL3_GLM_TP_SPLIT`: what two ranks split besides the routed experts, a comma list of `kda` (the KDA layers' 64 heads, 32 a rank,
/// at decode and at prefill: the layer holds state), `lm` (the lm_head's vocabulary halves, the fp16 logits swapped so both ranks hold
/// them) and `mla` (the MLA heads at decode, 32 a rank: q_b strips, the absorb / attention / expand of the rank's heads, the o_proj by
/// K-split partials; MLA prefill stays replicated, it holds no state) and `sh` (the shared expert of every MoE layer in decode windows:
/// gate / up strips of the rank's intermediate half, the down projection's K-split partials of that half riding in the routed experts'
/// exchange; prefill chunks keep it replicated). **Unset: all four, `kda,lm,mla,sh` ([`DEFAULT_SPLIT`], the default since w10); `none`
/// (or a value that is set but empty): nothing, the layout of the builds before it, byte for byte.** Every cut keeps the single-node bits
/// (`gb_cuda::exl3_split`, `crates/cuda/tests/kda_split.rs`, `shared_split.rs`). A fact of the TP Hello (`GlmFacts::split`): two ranks
/// that disagree are refused by name before the load.
pub const TP_SPLIT_ENV: &str = "REXL3_GLM_TP_SPLIT";
/// [`TP_SPLIT_ENV`] bits.
pub const SPLIT_KDA: u8 = 1;
pub const SPLIT_LM: u8 = 2;
pub const SPLIT_MLA: u8 = 4;
pub const SPLIT_SHARED: u8 = 8;
/// `lm-held` (opt-in, never in [`DEFAULT_SPLIT`]): `lm`, and the rank HOLDS only its vocabulary half of the lm_head (`GlmConfig::lm_strips`:
/// the mapper places only those strips, the memory plan is that much smaller). The half runs with the whole head's plan
/// (`PlannedExl3::of_loaded_cut`), so its logits are the whole head's columns bit for bit. Every reader of the head then takes its half:
/// the trunk's head (`split_lm_head`), the NextN drafter's full-vocabulary head calls (the same split, on both ranks in the same calls),
/// and the drafter's restricted head is built from the pack's bytes (`GlmModel::mtp_restrict_vocab_from_pack`, the same bytes as the copy
/// of a whole head). The all-row debug logits and the DFlash2 draft, which read the whole head, are refused by name.
pub const SPLIT_LM_HELD: u8 = 16;
/// What two ranks split when [`TP_SPLIT_ENV`] is not set (w10): the KDA heads, the vocabulary halves, the MLA heads and the shared expert.
/// `REXL3_GLM_TP_SPLIT=none` is the opt-out.
pub const DEFAULT_SPLIT: u8 = SPLIT_KDA | SPLIT_LM | SPLIT_MLA | SPLIT_SHARED;

/// The split set of [`TP_SPLIT_ENV`] from a lookup. A name it does not know is an error naming the variable (a typo must not leave one rank
/// unsplit while the other splits: the Hello would refuse, but by a less obvious name).
pub fn split_from_vars(get: impl Fn(&str) -> Option<String>) -> Result<u8, String> {
    // not set: the default; a value that is set (even an empty one) is read as before
    let Some(value) = get(TP_SPLIT_ENV) else { return Ok(DEFAULT_SPLIT) };
    let mut bits = 0u8;
    for part in value.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        bits |= match part {
            "none" => 0,
            "kda" => SPLIT_KDA,
            "lm" => SPLIT_LM,
            "lm-held" => SPLIT_LM | SPLIT_LM_HELD,
            "mla" => SPLIT_MLA,
            "sh" => SPLIT_SHARED,
            other => return Err(format!("{TP_SPLIT_ENV}: {other:?} is not one of kda, lm, lm-held, mla, sh, none")),
        };
    }
    Ok(bits)
}

/// Why the head split `bits` cannot run under the prefill contract `contract` (`REXL3_GLM_PREFILL_CONTRACT`), or `None`. Lane P's P5b
/// (2026-10-19) LIFTED the refusals of lane E's KDA and MLA splits under P2: a P2 prompt chunk on a KDA-split rank runs its qkv strips
/// through the P2 GEMM (whose columns are the whole layer's, `glm_p2_dense.rs` `p2_on_a_strip_cut_...`), the chunked recurrence on its
/// heads and the full P2 o_proj over the gathered heads (`model/tp/p2_split.rs`); a P2 prompt chunk never takes the MLA split's
/// decode-path arm (MLA prefill stays replicated). Kept, always `None` now, for the call sites (the rank planner and `enable_head_split`).
pub fn split_refused_under_contract(bits: u8, contract: Option<&str>) -> Option<String> {
    let _ = (bits, contract);
    None
}

/// [`split_from_vars`] of the process environment.
pub fn split_from_env() -> Result<u8, String> {
    split_from_vars(|k| std::env::var(k).ok())
}

/// The split set as text (`kda,lm`, `none`): logs and the Hello's refusal.
pub fn split_names(bits: u8) -> String {
    let lm = if bits & SPLIT_LM_HELD != 0 { "lm-held" } else { "lm" };
    let v: Vec<&str> = [(SPLIT_KDA, "kda"), (SPLIT_LM, lm), (SPLIT_MLA, "mla"), (SPLIT_SHARED, "sh")]
        .iter()
        .filter(|(b, _)| bits & b != 0)
        .map(|(_, n)| *n)
        .collect();
    if v.is_empty() {
        "none".into()
    } else {
        v.join(",")
    }
}

/// One KDA layer of a rank that runs half its heads: the rank's columns of `qkv_proj` (its heads' q, k and v strips, the parent's plan
/// kept) and the KDA weights of its heads.
pub(super) struct KdaRank {
    pub qkv: PlannedExl3,
    pub w: kda::LayerWeights,
}

/// Bytes of the device buffers a rank's head split ([`GlmModel::enable_head_split`]) and its P2 exchange add beside the memory profile's
/// items: priced before the load ([`split_planned_bytes`], `rank_extras`) and read from the rank after it ([`GlmModel::split_allocated_bytes`]),
/// so the plan and the allocations can be compared item by item.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SplitBytes {
    /// The KDA layers' per-head weight copies (`kda::LayerWeights::heads_of`).
    pub kda_copies: u64,
    /// The fp32 K-split partials of a split `o_proj` (`z`).
    pub partials: u64,
    /// This rank's half and the peer's of a gathered activation (`own`, `peer`).
    pub halves: u64,
    /// The shared-expert split's scratch and workspace (`sh`).
    pub shared: u64,
    /// The P2 exchange's parts (`p2::P2Exchange`).
    pub p2_parts: u64,
}

impl SplitBytes {
    pub fn total(&self) -> u64 {
        self.kda_copies + self.partials + self.halves + self.shared + self.p2_parts
    }
}

/// The P2 exchange's bytes at `rows` prompt rows a chunk (`p2::parts_bytes`).
pub fn p2_parts_bytes(cfg: &crate::families::glm::GlmConfig, rows: usize) -> u64 {
    p2::parts_bytes(cfg, rows)
}

/// What [`GlmModel::enable_head_split`] of `bits` allocates on rank `rank` of two for a model of `w` (the pack's weights, mapped for the
/// rank) at `rows` scratch rows (the prefill chunk), and the P2 exchange when `p2`: the same arithmetic over the pack's shapes, before
/// anything is loaded (the kernels' plans and workspace sizes are host functions). `kda_copies` are `kda::LayerWeights::heads_of`'s.
pub fn split_planned_bytes(
    w: &crate::families::glm::weights::GlmWeights,
    cfg: &crate::families::glm::GlmConfig,
    bits: u8,
    rank: usize,
    rows: usize,
    p2: bool,
) -> Result<SplitBytes, String> {
    use crate::families::glm::weights::{FfnWeights, MixerWeights};
    let p2_parts = if p2 { p2::parts_bytes(cfg, rows) } else { 0 };
    if bits == 0 {
        return Ok(SplitBytes { p2_parts, ..SplitBytes::default() });
    }
    let sk_of = |k: usize, n: usize| {
        gb_cuda::exl3_linear::decode_plan(k, n).map(|p| p.0).map_err(|e| format!("the plan of a {k} x {n} layer: {e}"))
    };
    let heads = cfg.kda.heads;
    let (h0, h1) = (rank * heads / 2, (rank + 1) * heads / 2);
    let (mut z, mut half, mut kda_copies) = (0usize, 0usize, 0u64);
    if bits & SPLIT_KDA != 0 {
        for l in &w.layers {
            if let MixerWeights::Kda(k) = &l.mixer {
                z = z.max(sk_of(k.o_proj.k, k.o_proj.n)? * alloc::MAX_ROWS * k.o_proj.n * 4);
                kda_copies += kda::LayerWeights::heads_of_copied_bytes(h0, h1) as u64;
            }
        }
        half = half.max(rows * (h1 - h0) * kda::HEAD_DIM * 2);
    }
    if bits & SPLIT_MLA != 0 {
        for l in &w.layers {
            if let MixerWeights::Mla(m) = &l.mixer {
                z = z.max(sk_of(m.o_proj.k, m.o_proj.n)? * alloc::MAX_ROWS * m.o_proj.n * 4);
            }
        }
    }
    if bits & SPLIT_LM != 0 {
        half = half.max(alloc::MAX_ROWS * w.lm_head.n / 2 * 2);
    }
    let mut shared = 0u64;
    if bits & SPLIT_SHARED != 0 {
        let first = w.layers.iter().find_map(|l| match &l.ffn {
            FfnWeights::Moe(m) => m.shared.as_ref(),
            FfnWeights::Dense(_) => None,
        });
        if let Some(sh) = first {
            let (hidden, inter, rows) = (sh.gate_proj.k, sh.gate_proj.n, alloc::WINDOW_ROWS);
            let sk = sk_of(sh.down_proj.k, sh.down_proj.n)?;
            let (s0, s1) = (rank * sk / 2, (rank + 1) * sk / 2);
            let strips = inter / 2 / 128;
            let strip_ws = |k: usize, n: usize| -> Result<usize, String> {
                let sk = sk_of(k, n).unwrap_or(1);
                Ok(gb_cuda::exl3_split::ex_workspace_bytes(rows, k, strips * 128, sk, false))
            };
            let ws = strip_ws(hidden, inter)?
                .max(strip_ws(sh.up_proj.k, sh.up_proj.n)?)
                .max(gb_cuda::exl3_split::ex_workspace_bytes(rows, inter / 2, sh.down_proj.n, s1 - s0, true))
                .max(16);
            shared = (gb_cuda::moe::SharedScratch::bytes_for(hidden, inter, rows) + ws) as u64;
        }
    }
    Ok(SplitBytes { kda_copies, partials: z.max(256) as u64, halves: 2 * half.max(256) as u64, shared, p2_parts })
}

impl GlmModel {
    /// What this rank's head split and P2 exchange hold (`None` before `enable_tp`): [`split_planned_bytes`]'s items, read from the buffers.
    pub fn split_allocated_bytes(&self) -> Option<SplitBytes> {
        let t = self.tp.as_ref()?;
        let p2_parts = t.p2_exchange_bytes() as u64;
        let Some(sp) = t.split.as_ref() else { return Some(SplitBytes { p2_parts, ..SplitBytes::default() }) };
        Some(SplitBytes {
            kda_copies: sp.kda.iter().map(|k| k.w.bytes()).sum(),
            partials: sp.z.bytes() as u64,
            halves: (sp.own.bytes() + sp.peer.bytes()) as u64,
            shared: sp.sh.as_ref().map_or(0, |s| (s.scratch.bytes() + s.ws.bytes()) as u64),
            p2_parts,
        })
    }
}

/// A rank's half of the replicated weights and the buffers of the exchanges they add.
pub(super) struct HeadSplit {
    pub bits: u8,
    /// This rank's place (0 or 1) and the KDA heads it runs.
    pub rank: usize,
    pub kda_heads: (usize, usize),
    /// The MLA heads this rank attends with at decode (`mla`; MLA prefill stays replicated: it holds no state).
    pub mla_heads: (usize, usize),
    /// Per KDA layer (in KDA order), when `kda` is split.
    pub kda: Vec<KdaRank>,
    /// The fp32 K-split partials of a split `o_proj`, `[sk][rows][hidden]` (windows: up to `MAX_ROWS` rows).
    z: DevBuf,
    /// This rank's half of a gathered activation and the peer's, `[rows][w]` fp16 (a prefill chunk's KDA output, a window's logits).
    own: DevBuf,
    peer: DevBuf,
    /// The shared-expert split (`sh`), when set.
    sh: Option<ShSplit>,
    /// `lm-held`: the rank's vocabulary half of the lm_head as loaded (the model's `lm_head` IS that half), run with the whole head's plan.
    lm: Option<PlannedExl3>,
    /// The whole head's columns (the vocabulary, padded to whole strips).
    vocab: usize,
}

/// A rank's half of every MoE layer's shared expert in decode windows (lane E, GLM Phase 2 E2.6: `sh`). Each trunk MoE layer's shared
/// expert runs [`SharedExpert::forward_split_partials`] over the down projection's K splits `splits` (of `sk`) into the TAIL of the
/// window's pair buffer, `[sk][rows][hidden]` f32 after the `rows x topk` pair rows, the peer's splits `-0.0`; the routed experts'
/// all-reduce then carries both (one exchange a layer: `x + -0.0 == x`, so the sum IS the gather of the partials), and every rank finishes
/// the partials into the shared expert's output ([`SharedExpert::finish_split`]: `rot_out` in split order), the single node's bit for bit.
/// Prefill chunks and the NextN layer keep the shared expert replicated.
pub(super) struct ShSplit {
    splits: (usize, usize),
    sk: usize,
    /// The widest window it runs (its scratch and workspace, and the pair buffer's tail, are sized for it); a wider call runs replicated.
    max_rows: usize,
    scratch: SharedScratch,
    ws: DevBuf,
}

/// The activations a split KDA layer reads and writes (fields of the model's scratch).
pub(super) struct KdaIo<'a> {
    pub hidden: &'a DevBuf,
    pub qkv: &'a mut DevBuf,
    /// The core's output: this rank's heads `[rows, 32 * 128]` fp16 on the window path; the GATHERED `[rows, 64 * 128]` on the big path.
    pub out: &'a mut DevBuf,
    pub mixer_out: &'a mut DevBuf,
    pub ws: &'a mut DevBuf,
    pub kws: &'a mut kda::Workspace,
}

/// Which KDA path a split layer runs.
pub(super) enum KdaPath<'a, 'b> {
    /// A decode / verify window, or a prompt chunk of at most `MAX_ROWS` rows (`commit`: every row accepted, the prefill chunk's way).
    Window { batch: &'a kda::Batch<'b>, rec: &'a mut kda::Record, commit: bool },
    /// A large prompt chunk of one sequence.
    Prefill { slot: usize, rec: &'a mut kda::Record, desc: &'a DevBuf },
}

impl GlmTp {
    /// The split set ([`SPLIT_KDA`] ...), 0 without a split.
    pub fn split_bits(&self) -> u8 {
        self.split.as_ref().map_or(0, |s| s.bits)
    }

    /// `[first, end)` of this rank's KDA heads, when KDA is split.
    pub fn kda_heads(&self) -> Option<(usize, usize)> {
        self.split.as_ref().filter(|s| s.bits & SPLIT_KDA != 0).map(|s| s.kda_heads)
    }

    /// `(first head, heads)` of this rank's MLA heads at decode, when MLA is split.
    pub(super) fn mla_heads(&self) -> Option<(usize, usize)> {
        self.split.as_ref().filter(|s| s.bits & SPLIT_MLA != 0).map(|s| (s.mla_heads.0, s.mla_heads.1 - s.mla_heads.0))
    }

    /// The KDA weights of this rank's heads of KDA layer `ord`, when KDA is split.
    pub(super) fn kda_rank(&self, ord: usize) -> Option<&KdaRank> {
        self.split.as_ref().filter(|s| s.bits & SPLIT_KDA != 0).and_then(|s| s.kda.get(ord))
    }

    fn swap(
        &mut self,
        send: &DevBuf,
        send_off: usize,
        recv: &mut DevBuf,
        recv_off: usize,
        bytes: usize,
        st: &Stream,
    ) -> Result<(), ModelError> {
        self.comm.swap_bytes(send, send_off, bytes, recv, recv_off, bytes, st)?;
        self.exchanged_bytes += bytes as u64;
        self.exchanges += 1;
        Ok(())
    }

    /// S2 `o_proj` of a split mixer: this rank's partials of its K splits (its heads' columns `x` `[rows, K / 2]` fp16, the splits
    /// `[r sk / 2, (r + 1) sk / 2)` of the layer's plan), the peer's swapped in, then the finish over all `sk` splits in split order into `out`
    /// `[rows, N]` f32: bit for bit the full `o_proj` of the gathered heads (`exl3_split`).
    pub(super) fn split_o(
        &mut self,
        o: &Exl3Weights,
        x: &DevBuf,
        rows: usize,
        out: &mut DevBuf,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let sp = self.split.as_mut().ok_or_else(|| ModelError::Config("no head split".into()))?;
        let r = sp.rank;
        let (sk, _) = o.plan()?;
        let per = sk / 2;
        let slab = rows * o.n * 4;
        o.forward_split_partials(r * per, (r + 1) * per, x, Dtype::F16, &mut sp.z, r * per * slab, rows, ws, st)?;
        let own = sp.z.subview(r * per * slab, per * slab);
        let mut z = sp.z.subview(0, sk * slab);
        self.swap(&own, 0, &mut z, (1 - r) * per * slab, per * slab, st)?;
        o.finish_partials(&z, rows, out, Dtype::F32, st)?;
        Ok(())
    }

    /// `full` `[rows, 2 w]` fp16 from this rank's half `own` `[rows, w]` (its columns `[r w, (r + 1) w)`) and the peer's, swapped in: byte copies.
    fn gather_cols_f16(&mut self, own: &DevBuf, full: &mut DevBuf, rows: usize, w: usize, st: &Stream) -> Result<(), ModelError> {
        let r = self.split.as_ref().ok_or_else(|| ModelError::Config("no head split".into()))?.rank;
        let bytes = rows * w * 2;
        let mut peer = self.split.as_ref().unwrap().peer.subview(0, bytes);
        self.swap(own, 0, &mut peer, 0, bytes, st)?;
        full.copy_2d_from(r * w * 2, 2 * w * 2, own, 0, w * 2, w * 2, rows, st)?;
        full.copy_2d_from((1 - r) * w * 2, 2 * w * 2, &peer, 0, w * 2, w * 2, rows, st)?;
        Ok(())
    }

    /// The lm_head's vocabulary halves: this rank's strips of `lm` into its half of `logits` `[n, vocab]` fp16, the peer's half swapped in
    /// (both ranks end with every logit). The columns are the full head's bit for bit (`Exl3Weights::forward_strip`, the parent's plan; under
    /// `lm-held` `lm` is the rank's half itself, run with the whole head's plan: `PlannedExl3::of_loaded_cut`).
    pub(super) fn split_lm_head(
        &mut self,
        lm: &Exl3Weights,
        x: &DevBuf,
        logits: &mut DevBuf,
        n: usize,
        ws: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let sp = self.split.as_ref().ok_or_else(|| ModelError::Config("no head split".into()))?;
        let r = sp.rank;
        let half = sp.vocab / 128 / 2;
        let w = half * 128;
        let run = |out: &mut DevBuf, ws: &mut DevBuf| -> Result<(), ModelError> {
            match &sp.lm {
                Some(held) => held.forward(x, out, n, Dtype::F16, Dtype::F16, ws, st)?,
                None => lm.forward_strip(r * half, (r + 1) * half, x, out, n, Dtype::F16, Dtype::F16, ws, st)?,
            }
            Ok(())
        };
        if n == 1 {
            let mut mine = logits.subview(r * w * 2, w * 2);
            run(&mut mine, ws)?;
            let mine = logits.subview(r * w * 2, w * 2);
            self.swap(&mine, 0, logits, (1 - r) * w * 2, w * 2, st)?;
            return Ok(());
        }
        let mut own = sp.own.subview(0, n * w * 2);
        run(&mut own, ws)?;
        self.gather_cols_f16(&own, logits, n, w, st)
    }

    /// One KDA layer on this rank's heads (32 of 64): the rank's qkv strips, the KDA core at 32 heads on the rank's pool, then the
    /// replicated `o_proj` of the 64 heads: on the window path by S2 partials (the rank's 4 of the 8 K splits swapped, 64 KiB a row), on the
    /// big path by the gathered core output (`out` `[rows, 8192]`, the peer's 4,096 columns swapped in) through the full layer. Every output
    /// is the single node's bit for bit.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn kda_split_layer(
        &mut self,
        ord: usize,
        o: &Exl3Weights,
        dims: kda::Dims,
        params: &kda::Params,
        pool: &mut kda::Pool,
        io: KdaIo<'_>,
        path: KdaPath<'_, '_>,
        rows: usize,
        st: &Stream,
    ) -> Result<(), ModelError> {
        let KdaIo { hidden, qkv, out, mixer_out, ws, kws } = io;
        let c = dims.v_dim();
        match path {
            KdaPath::Window { batch, rec, commit } => {
                let kr =
                    self.kda_rank(ord).ok_or_else(|| ModelError::Config(format!("no KDA head split for KDA layer {ord}")))?;
                kr.qkv.forward(hidden, qkv, rows, Dtype::F16, Dtype::F32, ws, st)?;
                kda::forward_window(
                    st,
                    dims,
                    &kr.w.view(),
                    params,
                    hidden,
                    Dtype::F16,
                    qkv,
                    Dtype::F32,
                    pool,
                    batch,
                    rec,
                    kws,
                    out,
                    true,
                )?;
                if commit {
                    kda::commit(st, dims, &kr.w.view(), params, pool, batch, rec, kws, None, true)?;
                }
                let x = out.subview(0, rows * c * 2);
                self.split_o(o, &x, rows, mixer_out, ws, st)
            }
            KdaPath::Prefill { slot, rec, desc } => {
                let mut own = self.split.as_ref().unwrap().own.subview(0, rows * c * 2);
                {
                    let kr =
                        self.kda_rank(ord).ok_or_else(|| ModelError::Config(format!("no KDA head split for KDA layer {ord}")))?;
                    kr.qkv.forward_big(hidden, qkv, rows, Dtype::F16, Dtype::F32, ws, st)?;
                    kda::prefill_with(
                        st,
                        dims,
                        &kr.w.view(),
                        params,
                        hidden,
                        Dtype::F16,
                        qkv,
                        Dtype::F32,
                        pool,
                        slot,
                        rows,
                        rec,
                        kws,
                        &mut own,
                        desc,
                    )?;
                }
                self.gather_cols_f16(&own, out, rows, c, st)?;
                o.forward_big(out, mixer_out, rows, Dtype::F16, Dtype::F32, ws, st)?;
                Ok(())
            }
        }
    }
}

impl GlmModel {
    /// Split the replicated weights `bits` ([`SPLIT_KDA`], [`SPLIT_LM`]) with the peer of this rank's two-rank group (after
    /// [`GlmModel::enable_tp`], before the rank serves): KDA runs heads `[32 r, 32 r + 32)` of every KDA layer of the trunk (its qkv strips
    /// copied with the parent's plan, its heads' KDA weights, pools and window records of 32 heads in place of the 64-head ones: the state of
    /// a slot halves), the `o_proj` and the lm_head stay whole (their cuts run on them in place). Refused by name: without a group of two
    /// ranks that can swap bytes (under P2 the split runs too since Lane P's P5b: `tp/p2_split.rs`).
    /// The NextN layer is never split. `mla`: the trunk's MLA layers at decode (their prefill stays replicated).
    pub fn enable_head_split(&mut self, bits: u8) -> Result<(), ModelError> {
        let cfg_err = |m: String| Err(ModelError::Config(m));
        // a model built for a rank's KDA heads alone (`KdaConfig::rank`: the rank strips) runs only with the KDA split
        if let Some((a, b)) = self.cfg.kda.rank {
            if bits & SPLIT_KDA == 0 {
                return cfg_err(format!(
                    "this rank loaded only the KDA heads {a}..{b} (its qkv strips and their state) but {TP_SPLIT_ENV} does not split kda \
                     ({}): the rank planner and the split must agree",
                    split_names(bits)
                ));
            }
        }
        // a model built for a rank's vocabulary half (`GlmConfig::lm_strips`) runs only with `lm-held`, and `lm-held` only on such a model
        if self.cfg.lm_strips.is_some() != (bits & SPLIT_LM_HELD != 0) {
            return cfg_err(format!(
                "this rank {} but {TP_SPLIT_ENV} {} ({}): the rank planner and the split must agree",
                match self.cfg.lm_strips {
                    Some((a, b)) => format!("loaded only the lm_head strips {a}..{b}"),
                    None => "loaded the whole lm_head".into(),
                },
                if bits & SPLIT_LM_HELD != 0 { "holds a vocabulary half" } else { "does not" },
                split_names(bits)
            ));
        }
        if bits == 0 {
            return Ok(());
        }
        let Some(t) = self.tp.as_ref() else {
            return cfg_err(format!("{TP_SPLIT_ENV}: a head split needs a tensor-parallel group (enable_tp first)"));
        };
        if t.split.is_some() {
            return cfg_err("this rank's heads are already split".into());
        }
        let topo = t.comm.topology();
        if topo.ranks != 2 || !t.comm.supports_swap() {
            return cfg_err(format!(
                "{TP_SPLIT_ENV}: the head split takes a group of two ranks whose communicator swaps bytes (ranks {}, swap {})",
                topo.ranks,
                t.comm.supports_swap()
            ));
        }
        if let Some(why) = split_refused_under_contract(bits, std::env::var("REXL3_GLM_PREFILL_CONTRACT").ok().as_deref()) {
            return cfg_err(why);
        }
        let r = topo.rank;
        let st = Stream::new()?;
        let heads = self.cfg.kda.heads;
        let (h0, h1) = (r * heads / 2, (r + 1) * heads / 2);
        let mut kda_ranks = Vec::new();
        let mut z_bytes = 0usize;
        let mut half_bytes = 0usize;
        if bits & SPLIT_KDA != 0 {
            let full = kda::Dims { heads, hidden: self.kda_dims.hidden };
            if heads % 8 != 0 || kda::Dims::glm() != full || self.cfg.kda.head_dim != kda::HEAD_DIM {
                return cfg_err(format!(
                    "the KDA head split takes 64-wide heads in a multiple of 8, not {heads} x {}",
                    self.cfg.kda.head_dim
                ));
            }
            let dims = kda::Dims { heads: h1 - h0, hidden: full.hidden };
            let ranges = crate::families::glm::weights::kda_qkv_strips(heads, kda::HEAD_DIM, h0, h1);
            // the rank strips: a rank that loaded only its heads' qkv strips (`KdaConfig::rank`, the mapper's `StripCut`) runs them in place
            // with the parent's plan; one that loaded the whole qkv copies them (`select_strips_planned`): the same bytes either way
            let loaded = self.cfg.kda.rank;
            if loaded.is_some_and(|r| r != (h0, h1)) {
                return cfg_err(format!(
                    "this rank loaded the KDA heads {:?} but rank {r} of the head split holds heads {h0}..{h1}",
                    loaded.unwrap()
                ));
            }
            let parent_n = 3 * full.v_dim();
            for layer in &self.dev.layers {
                let MixerDevice::Kda(k) = &layer.mixer else { continue };
                let (sk, _) = k.o.plan()?;
                if sk % 2 != 0 || k.o.split_k_range(r * sk / 2, (r + 1) * sk / 2)? != (h0 * kda::HEAD_DIM, h1 * kda::HEAD_DIM) {
                    return cfg_err(format!("the KDA o_proj's K splits ({sk}) do not cut at this rank's heads {h0}..{h1}"));
                }
                let qkv = match (&k.qkv_cut, loaded) {
                    (None, None) => k.qkv.select_strips_planned(&ranges)?,
                    (Some(c), Some(_)) if c.ranges == ranges && k.qkv.n == c.kept_columns() => {
                        PlannedExl3::of_loaded_cut(&k.qkv, parent_n)?
                    }
                    (c, _) => {
                        return cfg_err(format!(
                            "a KDA layer's loaded qkv strips {:?} are not this rank's {ranges:?} (heads {h0}..{h1}, the model's {loaded:?})",
                            c.as_ref().map(|c| &c.ranges)
                        ))
                    }
                };
                kda_ranks.push(KdaRank { qkv, w: kda::LayerWeights::heads_of(&k.view(), full, h0, h1, &st)? });
                z_bytes = z_bytes.max(sk * alloc::MAX_ROWS * k.o.n * 4);
            }
            // the rank's pools and window records of 32 heads (the state of a slot halves): a model built for the rank's heads has them
            // from the start (`alloc::dims_of`); one built for every head frees its 64-head ones here
            if self.kda_dims != dims {
                let n = self.pools.len();
                let slots = self.pools.first().map_or(self.slots, |p| p.n_slots);
                self.pools.clear();
                self.recs.clear();
                for _ in 0..n {
                    self.pools.push(kda::Pool::new(dims, slots)?);
                    self.recs.push(kda::Record::alloc(dims, alloc::WINDOW_ROWS)?);
                }
                self.kda_dims = dims;
            }
            half_bytes = half_bytes.max(self.s.rows * dims.v_dim() * 2);
        }
        let mh = self.cfg.mla.heads;
        let (m0, m1) = (r * mh / 2, (r + 1) * mh / 2);
        if bits & SPLIT_MLA != 0 {
            let (qk, vd) = (self.cfg.mla.qk_head_dim(), self.cfg.mla.v_head_dim);
            if mh % 32 != 0 || qk != gb_cuda::mla::NOPE_DIM || vd != gb_cuda::mla::NOPE_DIM {
                return cfg_err(format!(
                    "the MLA head split takes 16-head tiles of 256-wide heads, not {mh} heads x {qk} / {vd}"
                ));
            }
            for layer in &self.dev.layers {
                let MixerDevice::Mla(m) = &layer.mixer else { continue };
                let (sk, _) = m.o.plan()?;
                if m.q_b.n != mh * qk || sk % 2 != 0 || m.o.split_k_range(r * sk / 2, (r + 1) * sk / 2)? != (m0 * vd, m1 * vd) {
                    return cfg_err(format!(
                        "an MLA layer's q_b ({} columns) or o_proj K splits ({sk}) do not cut at heads {m0}..{m1}",
                        m.q_b.n
                    ));
                }
                z_bytes = z_bytes.max(sk * alloc::MAX_ROWS * m.o.n * 4);
            }
        }
        let vocab = self.cfg.vocab;
        let mut lm_held = None;
        if bits & SPLIT_LM != 0 {
            let lm = &self.dev.lm_head;
            if vocab % 256 != 0 {
                return cfg_err(format!("the lm_head's {vocab} columns do not halve into 128-column strips"));
            }
            let half = vocab / 256;
            match self.cfg.lm_strips {
                None if lm.n != vocab => {
                    return cfg_err(format!("the lm_head holds {} of its {vocab} columns without lm-held", lm.n));
                }
                None => {}
                Some(s) => {
                    if s != (r * half, (r + 1) * half) || lm.n != half * 128 || lm.k != self.cfg.hidden {
                        return cfg_err(format!(
                            "this rank holds the lm_head strips {s:?} ({} columns) but rank {r}'s vocabulary half is strips {}..{}",
                            lm.n,
                            r * half,
                            (r + 1) * half
                        ));
                    }
                    lm_held = Some(PlannedExl3::of_loaded_cut(lm, vocab)?);
                }
            }
            half_bytes = half_bytes.max(alloc::MAX_ROWS * vocab / 2 * 2);
        }
        let mut sh = None;
        if bits & SPLIT_SHARED != 0 {
            // every trunk MoE layer's shared expert: the down projection's K splits halve at the rank's intermediate half, alike in every layer
            let max_rows = alloc::WINDOW_ROWS;
            let mut cut: Option<(usize, usize, usize)> = None; // (sk, inter, hidden)
            let mut ws_bytes = 16usize;
            let mut first: Option<&SharedExpert> = None;
            for layer in &self.dev.layers {
                let (FfnDevice::Moe { shared, .. } | FfnDevice::MoeSubbanks { shared, .. }) = &layer.ffn else { continue };
                let (sk, _) = shared.down_plan()?;
                let (inter, hidden) = (shared.inter, shared.hidden);
                let halves = sk % 2 == 0
                    && inter % 256 == 0
                    && shared.down_split_k_range(r * sk / 2, (r + 1) * sk / 2)? == (r * inter / 2, (r + 1) * inter / 2);
                if !halves || cut.is_some_and(|c| c != (sk, inter, hidden)) {
                    return cfg_err(format!(
                        "layer {}'s shared expert ({hidden} -> {inter}, down plan {sk} K splits) does not halve at the intermediate columns as \
                         the other layers' do",
                        layer.index
                    ));
                }
                cut = Some((sk, inter, hidden));
                ws_bytes = ws_bytes.max(shared.split_workspace_bytes(r * sk / 2, (r + 1) * sk / 2, max_rows)?);
                first.get_or_insert(shared);
            }
            let (Some((sk, _, hidden)), Some(first)) = (cut, first) else {
                return cfg_err(format!("{TP_SPLIT_ENV}=sh: this model has no MoE layer with a shared expert"));
            };
            // the partials ride in the window's pair buffer, after its `rows x topk` pair rows: it must be a view (cut into ranges) and hold
            // both at the widest window
            let (y, topk) = (&self.s.moe.y_pairs, alloc::moe_dims(&self.cfg).topk);
            if y.block().is_none() || y.bytes() < max_rows * (topk + sk) * hidden * 4 {
                return cfg_err(format!(
                    "{TP_SPLIT_ENV}=sh: the pair buffer ({} bytes, a view: {}) cannot hold {max_rows} rows of pairs and {sk} partial splits",
                    y.bytes(),
                    y.block().is_some()
                ));
            }
            sh = Some(ShSplit {
                splits: (r * sk / 2, (r + 1) * sk / 2),
                sk,
                max_rows,
                scratch: SharedScratch::new(first, max_rows)?,
                ws: DevBuf::zeroed(ws_bytes)?,
            });
        }
        // views of blocks of their own: the exchanges cut them into ranges (`DevBuf::subview`)
        let z = DevBuf::zeroed_view(z_bytes.max(256))?;
        let own = DevBuf::zeroed_view(half_bytes.max(256))?;
        let peer = DevBuf::zeroed_view(half_bytes.max(256))?;
        let t = self.tp.as_mut().unwrap();
        t.split = Some(HeadSplit {
            bits,
            rank: r,
            kda_heads: (h0, h1),
            mla_heads: (m0, m1),
            kda: kda_ranks,
            z,
            own,
            peer,
            sh,
            lm: lm_held,
            vocab,
        });
        // the swap's point-to-point transport connects on its first use: once, eagerly, before any window is captured. Only where windows
        // are captured (a capturable communicator, NCCL): the peer runs the same call at the same point of its boot (the Hello has already
        // refused ranks that disagree on the split); the in-process loopback group waits for its peer inside every call and never captures.
        // Two sizes: a small one and one above every site of a transport that routes swaps by size (LANE-T's RoCE: up to 1 MiB on
        // its sites, NCCL above), so each path a window's swap can take is connected.
        if t.comm.capturable() {
            for bytes in [16usize, crate::families::glm::tp_boot::STEP_PROBE_BYTES] {
                let probe = DevBuf::zeroed(bytes)?;
                let mut back = DevBuf::zeroed(bytes)?;
                t.comm.swap_bytes(&probe, 0, bytes, &mut back, 0, bytes, &self.stream)?;
                self.stream.sync()?;
            }
        }
        self.clear_graphs()?;
        eprintln!(
            "rexl3 event=tp_glm_split rank={r} split={} kda_heads={h0}..{h1} kda_slot_mib={:.1} qkv_strips={}",
            split_names(bits),
            self.kda_dims.slot_bytes() as f64 * self.pools.len() as f64 / 1048576.0,
            match (bits & SPLIT_KDA != 0, self.cfg.kda.rank.is_some()) {
                (false, _) => "none",
                (true, true) => "loaded",
                (true, false) => "copied",
            }
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_set_parses_by_name_and_refuses_a_typo() {
        let of = |v: &str| split_from_vars(|_| Some(v.to_string()));
        assert_eq!(of(""), Ok(0));
        assert_eq!(of("kda, lm"), Ok(SPLIT_KDA | SPLIT_LM));
        assert_eq!(of("kda,lm,mla"), Ok(SPLIT_KDA | SPLIT_LM | SPLIT_MLA));
        assert_eq!(of("sh"), Ok(SPLIT_SHARED));
        assert_eq!(split_names(SPLIT_KDA | SPLIT_LM | SPLIT_MLA | SPLIT_SHARED), "kda,lm,mla,sh");
        assert!(of("kda,mlx").unwrap_err().contains(TP_SPLIT_ENV));
        assert_eq!(split_from_vars(|_| None), Ok(DEFAULT_SPLIT));
        // `lm-held` (opt-in, never the default): the lm split, and the rank holds only its half
        assert_eq!(of("lm-held"), Ok(SPLIT_LM | SPLIT_LM_HELD));
        assert_eq!(of("kda,lm-held,mla,sh"), Ok(DEFAULT_SPLIT | SPLIT_LM_HELD));
        assert_eq!(of("lm,lm-held"), Ok(SPLIT_LM | SPLIT_LM_HELD));
        assert_eq!(split_names(DEFAULT_SPLIT | SPLIT_LM_HELD), "kda,lm-held,mla,sh");
        assert_eq!(of(&split_names(DEFAULT_SPLIT | SPLIT_LM_HELD)), Ok(DEFAULT_SPLIT | SPLIT_LM_HELD));
        assert_eq!(DEFAULT_SPLIT & SPLIT_LM_HELD, 0);
        assert!(of("lm-half").unwrap_err().contains("lm-held"));
    }

    /// w10: unset is `kda,lm,mla,sh`; `none` (and a set-but-empty value) is exactly the builds before it; a list is read as before and
    /// `none` among others adds nothing.
    #[test]
    fn unset_is_all_four_and_none_is_the_opt_out() {
        let of = |v: Option<&str>| split_from_vars(|k| (k == TP_SPLIT_ENV).then(|| v.map(str::to_string)).flatten());
        assert_eq!(DEFAULT_SPLIT, SPLIT_KDA | SPLIT_LM | SPLIT_MLA | SPLIT_SHARED);
        assert_eq!(split_names(DEFAULT_SPLIT), "kda,lm,mla,sh");
        assert_eq!(of(None), Ok(DEFAULT_SPLIT));
        assert_eq!(of(Some("none")), Ok(0));
        assert_eq!(of(Some(" none ")), Ok(0));
        assert_eq!(of(Some("")), Ok(0));
        assert_eq!(of(Some("none,kda")), Ok(SPLIT_KDA));
        assert_eq!(of(Some("kda,lm,mla,sh")), Ok(DEFAULT_SPLIT));
        assert_eq!(of(Some("kda,lm,mla")), Ok(SPLIT_KDA | SPLIT_LM | SPLIT_MLA));
        assert!(of(Some("kda,mlx")).unwrap_err().contains(TP_SPLIT_ENV));
        // a variable other than ours is not ours
        assert_eq!(split_from_vars(|k| (k == "OTHER").then(|| "none".to_string())), Ok(DEFAULT_SPLIT));
    }

    #[test]
    fn no_split_is_refused_under_p2_since_p5b() {
        let all = SPLIT_KDA | SPLIT_LM | SPLIT_MLA;
        for c in [None, Some("p1"), Some("P1"), Some(""), Some("p2"), Some(" P2 ")] {
            for bits in [0, SPLIT_KDA, SPLIT_LM, SPLIT_MLA | SPLIT_LM, all, SPLIT_SHARED | SPLIT_LM, all | SPLIT_SHARED] {
                assert_eq!(split_refused_under_contract(bits, c), None, "{bits} under {c:?}");
            }
        }
    }
}
