//! The load plan: every tensor of a DeepSeek-V4.1-Flash pack, classified into exactly one of three kinds of memory, and the
//! arithmetic that decides whether the result fits a device.
//!
//! ## Why the plan is partial
//!
//! The pack is 332.5 GB of tensors and the device has 121 GiB. No plan that loads it whole exists, so a plan that pretended to
//! would be a plan that lies. [`LoadPlan`] therefore *classifies* rather than enumerates allocations:
//!
//! * [`Placement::Resident`] — in device memory for the life of the process: the embedding and head, the MLA projections, the
//!   compressors and indexers, the shared experts, the routers, the mHC coefficients, the norms, the engram branch's eight
//!   projections, all three NextN layers and the vision tower with its aligner. 18,750,160,200 B (18.750 GB).
//! * [`Placement::ExpertCache`] — the 46,080 routed EXL3 experts: 111,005,061,120 B (111.005 GB) that a *budget* decides how much
//!   of to keep. [`LoadPlan::expert_budget_bytes`] is that budget, and everything above it is streamed per token.
//! * [`Placement::Sidecar`] — the two engram tables: 202,758,032,400 B (202.758 GB) addressed through an `mmap` of the shard, never
//!   copied to the device. They are load *items* in the sense that they must be readable; they are not device bytes.
//!
//! ## Admission
//!
//! [`admit`] takes the plan, the device's available bytes, a context length and a slot count and returns either a full accounting
//! (`resident + slab + window + KV + scratch + rope + runtime reserve + headroom`, every term named and every term an allocation)
//! or an [`AdmissionError`] that says which term overflowed and by how many bytes. Two rules it keeps, because every alternative is
//! a silent failure:
//!
//! * it never shrinks the context unless the caller asked it to ([`AdmissionRequest::shrink_context`]), and when it does the
//!   admitted length is in [`Admission::context`] next to the requested one;
//! * it never shrinks the expert slab silently: a slab that does not fit is [`AdmissionError::ExpertSlabOverflow`] carrying the
//!   largest slab that does, unless the caller asked for [`AdmissionRequest::shrink_slab`].
//!
//! **Every term is a constructor.** The KV is [`gb_cuda::dsv4_attn::Dsv4Planes`] — this pack allocates no paged latent, so the
//! latent/compressed/indexer formulas in [`super::alloc`] are *not* charged on top of it; the scratch is
//! [`super::model::Scratch::bytes_for`], the buffers `Dsv41Model::new` allocates, not the nominal [`super::alloc::scratch_bytes`]
//! workspace nothing allocates; the headroom defaults to zero because the floor already is the margin; and the runtime reserve is
//! the measured [`runtime_reserve_for`] (0.25 GiB + 0.4% of the allocated bytes) rather than a flat guess.
//!
//! The measured starting point (121 GiB device, 32,768-token context, one slot, fp16): resident 18.750 GB + KV 1.446 GB +
//! scratch 0.101 GB + rope 0.017 GB + the measured reserve leaves 108,823,634,856 B of expert slab — 98.0% of the experts, so
//! about 7 experts a layer stream. On the Spark's 118.118 GiB of `MemAvailable` less a 0.5 GiB floor it is 93.98 GiB of slab,
//! 13,665 of the 15,360 experts resident and 9.40 GiB streamed (q4: 94.77 GiB of slab, 13,796 resident, 8.61 GiB streamed).

use super::alloc::{self, KvQuant};
use super::config::DeepseekV41Config;
use super::inventory::Component;
use super::weights::{
    DeepseekV41Weights, Engram, Exl3Linear, Fp8Linear, Hc, Indexer, LmHead, Mla, MtpExpert, MtpLayer, Router, SharedExperts,
    Tensor, TrunkLayer, Vision, VisionLinear, WkvProj,
};
use std::collections::BTreeMap;

/// The default headroom: **zero**. The margin a boot needs at the bottom of the pool is the floor the allocator
/// ([`load::allocate`]) and the planner ([`super::engine::usable_bytes`]) already hold back — 0.5 GiB `MemAvailable` on the
/// Spark, measured — so charging a second gigabyte here double-counted it. Overridable with `REXL3_DSV41_HEADROOM_GIB`.
pub const DEFAULT_HEADROOM_BYTES: u64 = 0;

/// [`DEFAULT_HEADROOM_BYTES`], or `REXL3_DSV41_HEADROOM_GIB` when it is set to a number of GiB.
pub fn headroom_from_env() -> u64 {
    std::env::var("REXL3_DSV41_HEADROOM_GIB")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g >= 0.0)
        .map(|g| (g * (1u64 << 30) as f64) as u64)
        .unwrap_or(DEFAULT_HEADROOM_BYTES)
}

/// Device bytes reserved for the expert streaming window: the slab the non-resident tail is paged through.
///
/// This is the term that makes a DeepSeek-V4.1-Flash pack bootable on a GB10. The tail is 24.7 GiB of experts and it is **disk**; the only
/// device memory it costs is this window, and the window is a constant. 4 GiB is 555 records of 7.72 MB — 13 per layer, against a 384-way
/// router, which is the point at which a decode token's 240 touches stop being pure misses; and on unified memory it is 4 GiB of the same
/// 110 GiB the trunk and the resident experts come out of, so it is priced, not free.
pub const DEFAULT_STREAM_WINDOW_BYTES: u64 = 4 << 30;

/// The **measured** runtime reserve: a boot pays 0.25 GiB plus 0.4% of the device bytes it allocates, for everything no
/// constructor of ours names — the CUDA context and its kernel images, the host heap the pack's bindings are read through (a
/// 167 MB `wq_b` is a `Vec<u8>` before it is a `DevBuf`), the loader's staging slots, and the GPU page tables that scale with
/// the allocation.
///
/// The measurement it comes from (DGX Spark GB10, 32K context, 1 slot, 0.5 GiB `MemAvailable` floor, one boot):
///
/// ```text
/// consumed 116.868 GiB  (stage=admitted 118.118 -> stage=model_built 1.250)
///   regions 105.242 + uploads 6.705 + window 3.303 + planes 0.874 + model 0.086 + rope 0.016 = 116.226
///   residual 0.642 GiB, of which the CUDA context is 0.207
/// ```
///
/// The flat 1.5 GiB this replaces over-charged that boot by 0.86 GiB and under-charged a boot with a bigger slab; the
/// residual tracks the allocated bytes (page tables), which is what the 0.4% is. At the measured boot's own shape (113.62 GiB
/// allocated: 91.182 of slab, 17.462 resident, 4.0 of window, 0.874 of planes, 0.086 of model scratch, 0.016 of rope) the model
/// gives 0.25 + 0.45 = 0.704 GiB against the measured 0.642 — 0.06 GiB of margin, on the safe side.
/// Overridable with `REXL3_DSV41_RUNTIME_RESERVE_GIB`.
///
/// Re-measured on the Spark at a 0.5 GiB floor, window priced as allocated: two boots of the same pack left
/// +0.368 GiB and +0.086 GiB unpriced over a 0.25 GiB base (the boot-to-boot spread is the driver's and the page cache's, not the
/// plan's), and the second ended at 0.43 GiB of `MemAvailable` - under the floor. The base is therefore 0.70 GiB: the worse boot's
/// residual plus 0.1 GiB, so a boot lands on or above the floor.
pub const RUNTIME_RESERVE_BASE_BYTES: u64 = 700 << 20;

/// Basis points of the allocated device bytes the runtime reserve charges: 0.4% (4 per 1,000).
pub const RUNTIME_RESERVE_PER_MILLE: u64 = 4;

/// The runtime reserve for a boot that allocates `allocated_bytes` of device memory
/// ([`RUNTIME_RESERVE_BASE_BYTES`] + 0.4% of it).
pub fn runtime_reserve_for(allocated_bytes: u64) -> u64 {
    RUNTIME_RESERVE_BASE_BYTES + allocated_bytes / 1_000 * RUNTIME_RESERVE_PER_MILLE
}

/// [`runtime_reserve_for`], unless `REXL3_DSV41_RUNTIME_RESERVE_GIB` names a flat reserve in GiB.
pub fn runtime_reserve_from_env() -> Option<u64> {
    std::env::var("REXL3_DSV41_RUNTIME_RESERVE_GIB")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g >= 0.0)
        .map(|g| (g * (1u64 << 30) as f64) as u64)
}

/// Where a tensor lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Placement {
    /// Device memory, allocated up front and held for the life of the process.
    Resident,
    /// A NextN (MTP) routed expert: device memory, always, and **outside the slab budget**. The drafter's 384 experts are 3.18 GiB
    /// of EXL3 records in the re-upload; they are resident by construction and nothing ever streams them, so they are neither
    /// counted in [`LoadPlan::expert_bytes`] (what a budget buys) nor walked by the LFU cache. They live in a region of their own
    /// ([`super::load::RegionKind::MtpExperts`]) so that "always resident" is a placement, not a habit.
    MtpExpert,
    /// A routed expert: device memory if [`LoadPlan::expert_budget_bytes`] reaches it, streamed per token otherwise.
    ExpertCache,
    /// An engram table: `mmap`ped from the shard and addressed in place, never copied to the device.
    Sidecar,
}

impl std::fmt::Display for Placement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Placement::Resident => "resident",
            Placement::MtpExpert => "mtp-expert",
            Placement::ExpertCache => "expert-cache",
            Placement::Sidecar => "sidecar",
        })
    }
}

/// One tensor of the plan, placed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanItem {
    pub name: String,
    pub bytes: u64,
    pub placement: Placement,
    pub component: Component,
}

/// Why a plan could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    /// The expert budget is larger than the experts the pack holds.
    ExpertBudget { requested: u64, available: u64 },
    /// A tensor of the tree was not classified: the walk and the tree disagree (a bug, not a pack problem).
    Unclassified { name: String },
    /// The walk saw a different number of tensors (or bytes) than the tree holds.
    Incomplete { walked: usize, walked_bytes: u64, tree: usize, tree_bytes: u64 },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::ExpertBudget { requested, available } => write!(
                f,
                "deepseek_v41: an expert budget of {requested} B exceeds the {available} B of routed experts the pack holds"
            ),
            PlanError::Unclassified { name } => write!(f, "deepseek_v41: `{name}` is in the tree and the plan did not place it"),
            PlanError::Incomplete { walked, walked_bytes, tree, tree_bytes } => write!(
                f,
                "deepseek_v41: the plan walked {walked} tensors ({walked_bytes} B) of a tree of {tree} ({tree_bytes} B)"
            ),
        }
    }
}

impl std::error::Error for PlanError {}

/// Every tensor of a pack, placed; and the two totals admission needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadPlan {
    /// In tree order, one item a tensor.
    pub items: Vec<PlanItem>,
    /// Device bytes that are resident whatever the context and the slab: 18,750,160,200 B for the real pack.
    pub resident_bytes: u64,
    /// Device bytes of the NextN (MTP) routed experts: always resident, never part of the slab a budget buys.
    pub mtp_expert_bytes: u64,
    /// Device bytes of the routed experts, all of them: 111,005,061,120 B.
    pub expert_bytes: u64,
    /// Bytes addressed through the `mmap` sidecar: 202,758,032,400 B.
    pub sidecar_bytes: u64,
    /// How many of [`LoadPlan::expert_bytes`] the plan keeps resident. Bounded by `expert_bytes`.
    pub expert_budget_bytes: u64,
    /// How many whole experts that budget buys (layer-major order), and how many bytes that leaves streaming.
    pub resident_experts: usize,
    pub streamed_expert_bytes: u64,
}

impl LoadPlan {
    /// Place every tensor of `w`, keeping at most `expert_budget` bytes of routed experts resident.
    ///
    /// The walk is exhaustive by construction and checked: it must see the same number of tensors and the same bytes as the tree,
    /// so a tensor added to [`DeepseekV41Weights`] without a placement here is [`PlanError::Incomplete`] rather than a silent hole.
    pub fn build(w: &DeepseekV41Weights, expert_budget: u64) -> Result<Self, PlanError> {
        Self::build_with(w, expert_budget, None)
    }

    /// [`build`](Self::build) for one rank of a tensor-parallel group: the rank's own routed experts — the trunk's
    /// and every NextN stage's — are planned, and the other ranks' are ABSENT, never read and never allocated (the
    /// mirror of nemotron's `Piece::Skip`, `tp_shard`: the plan is where "absent" becomes "never planned"). Everything
    /// the [`tp::ShardTable`] does not cut is planned exactly as `build` plans it — replicated, whole, on every rank.
    /// The walk stays exhaustive: the absent tensors are COUNTED (not lost), so the incomplete check still proves the
    /// walk saw every tensor of the tree. The budget buys whole experts of the rank's own range only.
    pub fn build_tp(w: &DeepseekV41Weights, expert_budget: u64, shard: &super::tp::ShardTable) -> Result<Self, PlanError> {
        Self::build_with(w, expert_budget, Some(shard))
    }

    /// [`LoadPlan::build_tp`] over an optional table: `None` is [`LoadPlan::build`] (every tensor placed). The engine's
    /// seam — its options may or may not carry a topology — so a boot's planning calls are one call.
    pub fn build_maybe(
        w: &DeepseekV41Weights,
        expert_budget: u64,
        shard: Option<&super::tp::ShardTable>,
    ) -> Result<Self, PlanError> {
        Self::build_with(w, expert_budget, shard)
    }

    /// The full plan: every expert resident. Fails admission on a 121 GiB device (the pack does not fit), which is the point.
    pub fn full(w: &DeepseekV41Weights) -> Result<Self, PlanError> {
        // every expert's aligned record, not just its parts: the padding is device memory too
        let all = super::alloc::expert_records_bytes(w);
        Self::build_with(w, all, None)
    }

    /// [`full`](Self::full) for one rank of a group: every expert THE RANK OWNS, resident; the other half is the
    /// other rank's.
    pub fn full_tp(w: &DeepseekV41Weights, shard: &super::tp::ShardTable) -> Result<Self, PlanError> {
        Self::build_with(w, owned_expert_records(w, Some(shard)), Some(shard))
    }

    /// The walk behind [`LoadPlan::build`] and [`LoadPlan::build_tp`]: `shard` is `None` for the single node (every
    /// tensor placed) and `Some` for one rank of a group (the other ranks' routed experts counted as absent instead).
    fn build_with(w: &DeepseekV41Weights, expert_budget: u64, shard: Option<&super::tp::ShardTable>) -> Result<Self, PlanError> {
        let mut items: Vec<PlanItem> = Vec::with_capacity(w.len());
        // the tensors this rank does not hold: the other ranks' routed experts, (count, bytes)
        let mut absent = (0usize, 0u64);
        push_tensor(&mut items, &w.embed, Placement::Resident, Component::Embed);
        match &w.head {
            LmHead::Bf16(t) => push_tensor(&mut items, t, Placement::Resident, Component::Head),
            LmHead::Exl3(g) => {
                for t in [&g.trellis, &g.mul1, &g.suh, &g.svh] {
                    push_tensor(&mut items, t, Placement::Resident, Component::Head);
                }
            }
        }
        push_tensor(&mut items, &w.norm, Placement::Resident, Component::Norm);
        for l in &w.layers {
            push_trunk_layer(&mut items, l, shard, &mut absent);
        }
        for m in &w.mtp {
            push_mtp_layer(&mut items, m, shard, &mut absent);
        }
        if let Some(v) = &w.vision {
            push_vision(&mut items, v);
        }
        if let Some(a) = &w.aligner {
            push_vision_linear(&mut items, &a.w1, Component::Vision);
            push_vision_linear(&mut items, &a.w2, Component::Vision);
        }

        let walked_bytes: u64 = items.iter().map(|i| i.bytes).sum();
        // the absent tensors are part of the walk (seen and deliberately not placed), so the check counts them
        if items.len() + absent.0 != w.len() || walked_bytes + absent.1 != w.bytes() {
            return Err(PlanError::Incomplete {
                walked: items.len() + absent.0,
                walked_bytes: walked_bytes + absent.1,
                tree: w.len(),
                tree_bytes: w.bytes(),
            });
        }

        let resident_bytes: u64 = items.iter().filter(|i| i.placement == Placement::Resident).map(|i| i.bytes).sum();
        // the drafter's experts: device memory, but not part of the slab a budget buys
        let mtp_expert_bytes: u64 = items.iter().filter(|i| i.placement == Placement::MtpExpert).map(|i| i.bytes).sum();
        let expert_bytes: u64 = items.iter().filter(|i| i.placement == Placement::ExpertCache).map(|i| i.bytes).sum();
        let sidecar_bytes: u64 = items.iter().filter(|i| i.placement == Placement::Sidecar).map(|i| i.bytes).sum();
        // the most a budget can buy is every expert's aligned record (parts plus padding), slightly more than the parts alone
        let expert_records = owned_expert_records(w, shard);
        if expert_budget > expert_records {
            return Err(PlanError::ExpertBudget { requested: expert_budget, available: expert_records });
        }
        // the budget buys whole experts and nothing else: walk them in layer-major order, taking every expert that fits. An expert that does
        // not fit is skipped rather than ending the walk, which is the same rule `Layout::new` uses — the two must agree or the layout's
        // slab is not the plan's. An expert is fitted by its aligned record (what it occupies in the slab) and counted by its parts (what
        // the pack holds), so the budget covers the padding and every byte total still matches the pack's. A rank walks only its own
        // range (`ShardTable::owns_trunk_expert`): the other half is never a candidate.
        let mut left = expert_budget;
        let mut resident_experts = 0usize;
        let mut resident_expert_bytes = 0u64;
        for l in &w.layers {
            for (e, ex) in l.experts.experts.iter().enumerate() {
                if !shard.is_none_or(|s| s.owns_trunk_expert(e)) {
                    continue;
                }
                let rec = ex.record_bytes();
                if rec <= left {
                    left -= rec;
                    resident_experts += 1;
                    resident_expert_bytes += ex.projections().iter().map(|g| g.bytes()).sum::<u64>();
                }
            }
        }
        Ok(Self {
            items,
            resident_bytes,
            mtp_expert_bytes,
            expert_bytes,
            sidecar_bytes,
            expert_budget_bytes: resident_expert_bytes,
            resident_experts,
            streamed_expert_bytes: expert_bytes - resident_expert_bytes,
        })
    }

    /// Every byte that has to be read: the resident set and the experts, but not the `mmap` sidecar.
    ///
    /// 129,755,221,320 B (129.755 GB) for the real pack — 99.87% of the device's 129,922,760,704 B, which is why the expert slab
    /// is a budget and not a given.
    pub fn load_item_bytes(&self) -> u64 {
        self.resident_bytes + self.mtp_expert_bytes + self.expert_bytes
    }

    /// Every byte the pack holds, sidecar included: 332,513,253,720 B.
    pub fn total_bytes(&self) -> u64 {
        self.resident_bytes + self.mtp_expert_bytes + self.expert_bytes + self.sidecar_bytes
    }

    pub fn items_in(&self, p: Placement) -> impl Iterator<Item = &PlanItem> {
        self.items.iter().filter(move |i| i.placement == p)
    }

    /// Bytes by placement, in the order [`Placement`] declares them.
    pub fn by_placement(&self) -> BTreeMap<Placement, u64> {
        let mut m = BTreeMap::new();
        for i in &self.items {
            *m.entry(i.placement).or_insert(0u64) += i.bytes;
        }
        m
    }

    /// Bytes by component, the classification the pack's own inventory uses.
    pub fn by_component(&self) -> BTreeMap<Component, u64> {
        let mut m = BTreeMap::new();
        for i in &self.items {
            *m.entry(i.component).or_insert(0u64) += i.bytes;
        }
        m
    }

    /// The items whose bytes are not device memory: the sidecar, named, so a caller can check it has the shards it needs.
    pub fn sidecar(&self) -> impl Iterator<Item = &PlanItem> {
        self.items_in(Placement::Sidecar)
    }
}

fn push_tensor(items: &mut Vec<PlanItem>, t: &Tensor, placement: Placement, component: Component) {
    items.push(PlanItem { name: t.name.clone(), bytes: t.bytes(), placement, component });
}

fn push_fp8(items: &mut Vec<PlanItem>, g: &Fp8Linear, component: Component) {
    if let Some(ex) = &g.exl3 {
        for t in [&ex.trellis, &ex.mul1, &ex.suh, &ex.svh] {
            push_tensor(items, t, Placement::Resident, component);
        }
        return;
    }
    push_tensor(items, &g.weight, Placement::Resident, component);
    push_tensor(items, &g.scale, Placement::Resident, component);
}

/// `whose` is `None` for a trunk layer (the inventory's `mla_q` / `mla_kv` / `mla_o` classes) and `Some(Mtp)` for a NextN layer,
/// whose whole block the inventory books as `mtp`.
fn push_mla(items: &mut Vec<PlanItem>, m: &Mla, whose: Option<Component>) {
    let (q, kv, o) = (whose.unwrap_or(Component::MlaQ), whose.unwrap_or(Component::MlaKv), whose.unwrap_or(Component::MlaO));
    push_fp8(items, &m.wq_a, q);
    push_fp8(items, &m.wq_b, q);
    push_fp8(items, &m.wkv, kv);
    push_fp8(items, &m.wo_a, o);
    push_fp8(items, &m.wo_b, o);
    push_tensor(items, &m.q_norm, Placement::Resident, q);
    push_tensor(items, &m.kv_norm, Placement::Resident, kv);
    push_tensor(items, &m.attn_sink, Placement::Resident, o);
}

fn push_hc(items: &mut Vec<PlanItem>, h: &Hc, component: Component) {
    push_tensor(items, &h.fn_, Placement::Resident, component);
    push_tensor(items, &h.base, Placement::Resident, component);
    push_tensor(items, &h.scale, Placement::Resident, component);
}

fn push_router(items: &mut Vec<PlanItem>, r: &Router, component: Component) {
    push_tensor(items, &r.weight, Placement::Resident, component);
    push_tensor(items, &r.bias, Placement::Resident, component);
    push_tensor(items, &r.bias_vl, Placement::Resident, component);
}

fn push_shared(items: &mut Vec<PlanItem>, s: &SharedExperts, component: Component) {
    for g in [&s.w1, &s.w2, &s.w3] {
        push_fp8(items, g, component);
    }
}

fn push_indexer(items: &mut Vec<PlanItem>, i: &Indexer) {
    push_fp8(items, &i.wq_b, Component::DsaIndex);
    push_tensor(items, &i.weights_proj, Placement::Resident, Component::DsaIndex);
    for t in [&i.wk, &i.k_norm].into_iter().flatten() {
        push_tensor(items, t, Placement::Resident, Component::DsaIndex);
    }
}

fn push_engram(items: &mut Vec<PlanItem>, e: &Engram) {
    // the table is the sidecar: two tensors addressed through the shard's mapping
    push_tensor(items, &e.table.embed, Placement::Sidecar, Component::Engram);
    push_tensor(items, &e.table.scale, Placement::Sidecar, Component::Engram);
    // the branch around it is ordinary resident weight
    push_tensor(items, &e.q_weight, Placement::Resident, Component::Engram);
    push_tensor(items, &e.k_weight, Placement::Resident, Component::Engram);
    match &e.wkv {
        WkvProj::Plain(w) => push_fp8(items, w, Component::Engram),
        // the EXL3 group form is read once at boot by the engine's own loader (`Checkpoint::load_group`)
        // and decoded into the host `Fp8Matrix` (`Fp8Matrix::from_group`): never device-resident, never
        // uploaded, addressed like the sidecar so the loader leaves it to the engine
        WkvProj::Group(g) => {
            push_tensor(items, &g.trellis, Placement::Sidecar, Component::Engram);
            push_tensor(items, &g.suh, Placement::Sidecar, Component::Engram);
            push_tensor(items, &g.svh, Placement::Sidecar, Component::Engram);
            push_tensor(items, &g.mul1, Placement::Sidecar, Component::Engram);
        }
    }
}

fn push_expert(items: &mut Vec<PlanItem>, g: &Exl3Linear) {
    for t in [&g.trellis, &g.mul1, &g.suh, &g.svh] {
        push_tensor(items, t, Placement::ExpertCache, Component::MoeExperts);
    }
}

/// The aligned record bytes of the routed trunk experts `shard` owns — all of them when it is `None`: the most a
/// budget can buy, and the cap [`PlanError::ExpertBudget`] names. A rank's cap is its own range's records, not the
/// pack's (the other half is never a candidate). `pub(crate)` for the layout's slab cap, which must be the same figure.
pub(crate) fn owned_expert_records(w: &DeepseekV41Weights, shard: Option<&super::tp::ShardTable>) -> u64 {
    match shard {
        None => super::alloc::expert_records_bytes(w),
        Some(s) => w
            .layers
            .iter()
            .flat_map(|l| l.experts.experts.iter().enumerate())
            .filter(|&(e, _)| s.owns_trunk_expert(e))
            .map(|(_, e)| e.record_bytes())
            .sum(),
    }
}

fn push_trunk_layer(items: &mut Vec<PlanItem>, l: &TrunkLayer, shard: Option<&super::tp::ShardTable>, absent: &mut (usize, u64)) {
    push_tensor(items, &l.attn_norm, Placement::Resident, Component::Norm);
    push_tensor(items, &l.ffn_norm, Placement::Resident, Component::Norm);
    push_mla(items, &l.mla, None);
    push_router(items, &l.router, Component::MoeRouter);
    push_shared(items, &l.shared, Component::MoeShared);
    // the routed experts of the OTHER ranks are absent: counted here (the walk's exhaustiveness check needs them),
    // never placed, never read, never allocated
    for (e, ex) in l.experts.experts.iter().enumerate() {
        if !shard.is_none_or(|s| s.owns_trunk_expert(e)) {
            for g in ex.projections() {
                absent.0 += 4;
                absent.1 += g.bytes();
            }
            continue;
        }
        for g in ex.projections() {
            push_expert(items, g);
        }
    }
    push_hc(items, &l.hc_attn, Component::HcAttn);
    push_hc(items, &l.hc_ffn, Component::HcFfn);
    if let Some(c) = &l.compressor {
        push_tensor(items, &c.wkv, Placement::Resident, Component::DsaCompressor);
        push_tensor(items, &c.norm, Placement::Resident, Component::DsaCompressor);
        if let Some(g) = &c.gate {
            push_tensor(items, g, Placement::Resident, Component::DsaCompressor);
        }
    }
    if let Some(i) = &l.indexer {
        push_indexer(items, i);
    }
    if let Some(e) = &l.engram {
        push_engram(items, e);
    }
}

fn push_mtp_layer(items: &mut Vec<PlanItem>, m: &MtpLayer, shard: Option<&super::tp::ShardTable>, absent: &mut (usize, u64)) {
    push_tensor(items, &m.attn_norm, Placement::Resident, Component::Mtp);
    push_tensor(items, &m.ffn_norm, Placement::Resident, Component::Mtp);
    push_mla(items, &m.mla, Some(Component::Mtp));
    push_router(items, &m.router, Component::Mtp);
    push_shared(items, &m.shared, Component::Mtp);
    // the drafter's experts: int8 in the first upload, EXL3 K2 in the re-upload. Either way they are resident — a drafter that
    // streamed its own experts would fault a token of lookahead through a 20 MiB read — so they are booked as `MtpExpert`, which
    // no budget and no LFU walk sees. A rank of a group holds its own range only; the others' are absent (counted, never placed).
    for (e, ex) in m.experts.iter().enumerate() {
        let owned = shard.is_none_or(|s| s.owns_nextn_expert(e));
        match ex {
            MtpExpert::Int4(ex) => {
                for g in ex.projections() {
                    if !owned {
                        absent.0 += 2;
                        absent.1 += g.bytes();
                        continue;
                    }
                    push_tensor(items, &g.weight, Placement::MtpExpert, Component::Mtp);
                    push_tensor(items, &g.scale, Placement::MtpExpert, Component::Mtp);
                }
            }
            MtpExpert::Exl3(ex) => {
                for g in ex.projections() {
                    if !owned {
                        absent.0 += 4;
                        absent.1 += g.bytes();
                        continue;
                    }
                    for t in [&g.trellis, &g.mul1, &g.suh, &g.svh] {
                        push_tensor(items, t, Placement::MtpExpert, Component::Mtp);
                    }
                }
            }
        }
    }
    push_hc(items, &m.hc_attn, Component::Mtp);
    push_hc(items, &m.hc_ffn, Component::Mtp);
    if let Some(g) = &m.main_proj {
        push_fp8(items, g, Component::Mtp);
    }
    for t in [&m.main_norm, &m.markov_embed, &m.markov_head, &m.confidence, &m.norm].into_iter().flatten() {
        push_tensor(items, t, Placement::Resident, Component::Mtp);
    }
}

fn push_vision_linear(items: &mut Vec<PlanItem>, g: &VisionLinear, component: Component) {
    push_tensor(items, &g.weight, Placement::Resident, component);
    push_tensor(items, &g.bias, Placement::Resident, component);
}

fn push_vision(items: &mut Vec<PlanItem>, v: &Vision) {
    push_vision_linear(items, &v.patch_embed, Component::Vision);
    for b in &v.blocks {
        push_tensor(items, &b.norm1, Placement::Resident, Component::Vision);
        push_tensor(items, &b.norm2, Placement::Resident, Component::Vision);
        push_vision_linear(items, &b.qkv, Component::Vision);
        push_vision_linear(items, &b.wo, Component::Vision);
        push_tensor(items, &b.mlp_w1, Placement::Resident, Component::Vision);
        push_tensor(items, &b.mlp_w2, Placement::Resident, Component::Vision);
    }
    for t in [&v.norm, &v.image_start, &v.image_end, &v.image_newline] {
        push_tensor(items, t, Placement::Resident, Component::Vision);
    }
}

// -------------------------------------------------------------------------------------------------------------------- admission

/// What admission is asked, and under what policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionRequest {
    /// Bytes the device has for this process (`121 GiB` on the target box: 129,922,760,704 B).
    pub device_bytes: u64,
    /// The context length asked for, in tokens.
    pub context: usize,
    /// Sequences served concurrently: the KV is priced for `context * slots` tokens.
    pub slots: usize,
    /// The dtype of the *paged* latent. This pack allocates no paged latent — its KV is the DSA planes, which are fp16 by
    /// construction ([`attn_planes_bytes`]) — so this does not move the price today; it is carried so the q4 latent work has
    /// the field it will price itself in.
    pub kv: KvQuant,
    /// How many bytes of routed experts the caller wants resident (whole experts).
    pub expert_slab_bytes: u64,
    /// Device bytes for the cache that streams the rest. Zero when every expert is resident; [`DEFAULT_STREAM_WINDOW_BYTES`] when a tail
    /// streams. The tail itself is never a term here: it is disk.
    pub window_bytes: u64,
    /// Rows of the widest forward call: the model's scratch is priced at this width.
    pub prefill_rows: usize,
    /// Reserved on top of everything else ([`DEFAULT_HEADROOM_BYTES`]: zero — the floor already is the margin).
    pub headroom_bytes: u64,
    /// `None`: the measured model, [`runtime_reserve_for`] — 0.25 GiB plus 0.4% of the device bytes the boot allocates.
    /// `Some`: a flat reserve (`REXL3_DSV41_RUNTIME_RESERVE_GIB`).
    pub runtime_reserve_bytes: Option<u64>,
    /// Admit a shorter context rather than failing, reporting the length. Off by default: a server that silently serves less
    /// context than it was asked for is worse than one that refuses to start.
    pub shrink_context: bool,
    /// Admit a smaller expert slab rather than failing, reporting the slab. Off by default: the slab is a speed decision.
    pub shrink_slab: bool,
}

impl AdmissionRequest {
    /// A request at the target device (121 GiB), one slot, the default headroom, failing closed on both shrink decisions.
    pub fn at_121_gib(context: usize, expert_slab_bytes: u64) -> Self {
        Self {
            device_bytes: 121 << 30,
            context,
            slots: 1,
            kv: KvQuant::F16,
            expert_slab_bytes,
            window_bytes: 0,
            prefill_rows: 512,
            headroom_bytes: headroom_from_env(),
            runtime_reserve_bytes: runtime_reserve_from_env(),
            shrink_context: false,
            shrink_slab: false,
        }
    }

    /// The serve this family runs: the largest whole-expert slab that fits, and the streaming window reserved out of the same device.
    pub fn serving(context: usize, slots: usize, device_bytes: u64) -> Self {
        Self { device_bytes, context, slots, window_bytes: DEFAULT_STREAM_WINDOW_BYTES, ..Self::at_121_gib(context, 0) }
    }
}

/// The full accounting of an admitted plan: every byte, by term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Admission {
    pub device_bytes: u64,
    /// The context actually admitted (equal to the request unless `shrink_context`).
    pub context: usize,
    pub requested_context: usize,
    pub slots: usize,
    /// The cache's format: the one admission priced and the one the model must allocate ([`KvQuant`], from `--kv-dtype` /
    /// `REXL3_DSV41_KV`, fp16 by default). Carried here so the build cannot allocate a different format from the one priced.
    pub kv: KvQuant,
    pub resident_bytes: u64,
    /// The NextN (MTP) routed experts: resident by construction, outside the slab. 3.180 GiB in the re-upload, 6.724 GiB in the
    /// first one.
    pub mtp_expert_bytes: u64,
    pub expert_slab_bytes: u64,
    /// The streaming window: device bytes, and the *only* device charge the non-resident tail carries. It is one region of
    /// the layout ([`super::load::Layout::window_region`]) and the cache's slab is a view of it, so this is charged once.
    pub expert_window_bytes: u64,
    pub streamed_expert_bytes: u64,
    /// The KV the model allocates: `layers x slots` of [`gb_cuda::dsv4_attn::Dsv4Planes`] — the window ring, the compressed
    /// plane, the indexer's keys, the level-1 candidate mask and the published selection. There is no paged latent and therefore
    /// no second KV charge: this *is* the KV.
    pub kv_bytes: u64,
    /// The model's own [`super::model::Scratch`] ([`super::model::Scratch::bytes_for`]): the activations and workspaces
    /// `Dsv41Model::new` allocates. The old nominal [`alloc::scratch_bytes`] is not charged — nothing ever allocated it.
    pub scratch_bytes: u64,
    /// The shared RoPE tables: one `cos`/`sin` pair a distinct rotation, not one a layer.
    pub rope_bytes: u64,
    /// The CUDA context, kernel images, host heap and page tables ([`runtime_reserve_for`]).
    pub runtime_reserve_bytes: u64,
    pub headroom_bytes: u64,
    /// Device bytes the boot allocates: `resident + slab + window + kv`. The runtime reserve is a percentage of this.
    pub allocated_bytes: u64,
    /// `resident + slab + window + kv + scratch + rope + runtime reserve + headroom`.
    pub total_bytes: u64,
    /// `device - total`, the slack left for nothing to fall over into.
    pub free_bytes: u64,
}

impl Admission {
    /// The terms, in the order they are charged. `streamed_expert_bytes` is deliberately not one of them: it is disk.
    pub fn terms(&self) -> [(&'static str, u64); 8] {
        [
            ("resident", self.resident_bytes),
            ("mtp_experts", self.mtp_expert_bytes),
            ("expert_slab", self.expert_slab_bytes),
            ("expert_window", self.expert_window_bytes),
            ("kv", self.kv_bytes),
            ("scratch", self.scratch_bytes),
            ("rope", self.rope_bytes),
            ("runtime_reserve", self.runtime_reserve_bytes),
        ]
    }

    /// `true` when nothing is streamed: the whole expert set is resident.
    pub fn all_experts_resident(&self) -> bool {
        self.streamed_expert_bytes == 0
    }

    /// `true` when the admitted context is the one that was asked for.
    pub fn context_honoured(&self) -> bool {
        self.context == self.requested_context
    }
}

impl std::fmt::Display for Admission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "resident {:.3} GB + MTP experts {:.3} GB + slab {:.3} GB + window {:.3} GB (streaming {:.3} GB from disk) + KV {:.3} GB (the DSA planes) + \
             scratch {:.3} GB + rope {:.3} GB + runtime reserve {:.3} GB + headroom {:.3} GB = {:.3} of {:.3} GB, {:.3} GB free, \
             context {} x {} slot(s)",
            gb(self.resident_bytes),
            gb(self.mtp_expert_bytes),
            gb(self.expert_slab_bytes),
            gb(self.expert_window_bytes),
            gb(self.streamed_expert_bytes),
            gb(self.kv_bytes),
            gb(self.scratch_bytes),
            gb(self.rope_bytes),
            gb(self.runtime_reserve_bytes),
            gb(self.headroom_bytes),
            gb(self.total_bytes),
            gb(self.device_bytes),
            gb(self.free_bytes),
            self.context,
            self.slots,
        )
    }
}

fn gb(b: u64) -> f64 {
    b as f64 / 1e9
}

/// Why a plan cannot be served at the settings asked for. Every variant names the term and the bytes, so a failed start says what
/// to change rather than "out of memory".
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    /// The context is longer than the model's own maximum: a rope/position problem before it is a memory one.
    ContextBeyondModel { context: usize, max_positions: usize },
    /// Zero slots, or zero context, or zero rows: nothing to price.
    Empty { what: &'static str },
    /// The resident set alone does not fit.
    ResidentOverflow { resident_bytes: u64, device_bytes: u64, over_by: u64 },
    /// The context asked for does not fit with the rest of the plan: the KV term is the one that overflowed.
    ContextOverflow {
        context: usize,
        slots: usize,
        device_bytes: u64,
        resident_bytes: u64,
        expert_slab_bytes: u64,
        scratch_bytes: u64,
        headroom_bytes: u64,
        kv_bytes: u64,
        kv_per_token: u64,
        over_by: u64,
        /// The longest context that does fit at these settings (`0` if not even one page does).
        largest_context: usize,
    },
    /// The expert slab asked for does not fit: the slab term overflowed.
    ExpertSlabOverflow { requested: u64, largest: u64, over_by: u64 },
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdmissionError::ContextBeyondModel { context, max_positions } => write!(
                f,
                "deepseek_v41: a context of {context} tokens exceeds the model's {max_positions} positions"
            ),
            AdmissionError::Empty { what } => write!(f, "deepseek_v41: admission was asked for zero {what}"),
            AdmissionError::ResidentOverflow { resident_bytes, device_bytes, over_by } => write!(
                f,
                "deepseek_v41: the resident set ({resident_bytes} B) alone exceeds the device ({device_bytes} B) by {over_by} B"
            ),
            AdmissionError::ContextOverflow {
                context,
                slots,
                kv_bytes,
                kv_per_token,
                over_by,
                largest_context,
                ..
            } => write!(
                f,
                "deepseek_v41: a context of {context} tokens over {slots} slot(s) needs {kv_bytes} B of KV ({kv_per_token} B a \
                 token) and overflows the device by {over_by} B; the longest context that fits is {largest_context} tokens"
            ),
            AdmissionError::ExpertSlabOverflow { requested, largest, over_by } => write!(
                f,
                "deepseek_v41: an expert slab of {requested} B overflows the device by {over_by} B; the largest slab that fits is \
                 {largest} B"
            ),
        }
    }
}

impl std::error::Error for AdmissionError {}

/// The row depth the model's scratch is allocated at: the executor's chunk (capped at the kernel's `MAX_ROWS`) or a window, whichever is
/// wider — the same expression [`super::model::Dsv41Model::new`] uses.
fn model_rows(req: &AdmissionRequest) -> usize {
    req.prefill_rows.clamp(super::model::WINDOW_ROWS, gb_cuda::dsv4_attn::MAX_ROWS)
}

/// The kernel crate's view of this pack's geometry. A config that does not translate is refused by name at bind ([`super::device`]);
/// admission is not the step that does it, and every field it reads is the family config's own.
fn kernel_cfg(cfg: &DeepseekV41Config) -> gb_cuda::dsv4_attn::Dsv4Config {
    super::device::dsv4_config(cfg).unwrap_or_else(|_| gb_cuda::dsv4_attn::Dsv4Config::v41())
}

/// The DSA planes of every layer over every slot: the window ring, the compressed plane, the indexer's keys, the level-1 candidate mask
/// and the published selection, from [`gb_cuda::dsv4_attn::Dsv4Planes::bytes`]. **This is the pack's KV**, and the only term
/// admission charges for it: the full five planes, per layer (a ratio-1 layer's compressed plane, index keys, mask and score
/// rows hold twice what a ratio-2 layer's does), at the cache's format.
pub fn attn_planes_bytes(cfg: &DeepseekV41Config, context: usize, slots: usize, kv: KvQuant) -> u64 {
    let dcfg = kernel_cfg(cfg);
    let fmt = kv.fmt();
    let per_slot: u64 = (0..cfg.layers).map(|l| gb_cuda::dsv4_attn::Dsv4Planes::bytes(&dcfg, context, l, fmt) as u64).sum();
    per_slot * slots as u64
}

/// Bytes a token of context adds to the device, from the same planes formula as [`attn_planes_bytes`]: on every layer one
/// compressed entry (`head_dim` at the cache's format) and one index key, plus one candidate-mask column (`MAX_ROWS` bytes),
/// each `ratio` tokens. The window ring and the published selection are constant in the context. The scheduler prices its KV
/// with this, so `gb_sched`'s admission and ours cannot disagree.
pub fn kv_bytes_per_token(cfg: &DeepseekV41Config, slots: usize, kv: KvQuant) -> u64 {
    let k = kernel_cfg(cfg);
    let fmt = kv.fmt();
    let per_position = gb_cuda::dsv4_attn::Dsv4Planes::bytes_per_position(&k, fmt) as u64;
    let per_slot: u64 = (0..cfg.layers).map(|l| per_position.div_ceil(k.ratio(l).max(1) as u64)).sum();
    per_slot * slots as u64
}

/// The shared RoPE tables: one `cos`/`sin` pair a distinct rotation ([`gb_cuda::dsv4_attn::SharedRope`]).
pub fn rope_bytes(cfg: &DeepseekV41Config, context: usize) -> u64 {
    gb_cuda::dsv4_attn::shared_rope_bytes(&kernel_cfg(cfg), context)
}

/// Every term that moves with the context, in one place: the DSA planes (the KV) and the model's own [`super::model::Scratch`],
/// plus the RoPE tables. Every one of them is a constructor's own formula, and every one of them is allocated.
#[derive(Clone, Copy, Debug)]
struct Moved {
    kv: u64,
    scratch: u64,
    rope: u64,
}

impl Moved {
    fn at(cfg: &DeepseekV41Config, req: &AdmissionRequest, context: usize) -> Self {
        Self {
            kv: attn_planes_bytes(cfg, context, req.slots, req.kv),
            scratch: super::model::Scratch::bytes_for(cfg, model_rows(req), context),
            rope: rope_bytes(cfg, context),
        }
    }

    /// Everything but the slab, the window and the headroom: what a context costs before any expert is resident.
    fn fixed(&self) -> u64 {
        self.kv + self.scratch + self.rope
    }
}

/// The runtime reserve for a boot whose allocations total `allocated_bytes`: the flat environment override if there is one, the
/// measured model ([`runtime_reserve_for`]) otherwise.
fn reserve(req: &AdmissionRequest, allocated_bytes: u64) -> u64 {
    req.runtime_reserve_bytes.unwrap_or_else(|| runtime_reserve_for(allocated_bytes))
}

/// The expert terms: whole resident experts and the streaming window. The window exists only when a tail does, and it is never larger than
/// what the caller asked for.
fn expert_terms(plan: &LoadPlan, slab: u64, window_bytes: u64) -> (u64, u64) {
    let slab = slab.min(plan.expert_bytes);
    let window = if slab >= plan.expert_bytes { 0 } else { window_bytes.min(slab) };
    (slab, window)
}

/// Everything a boot spends that is not the slab and not the window.
fn baseline(plan: &LoadPlan, req: &AdmissionRequest, m: &Moved) -> u64 {
    plan.resident_bytes.saturating_add(plan.mtp_expert_bytes).saturating_add(m.fixed()).saturating_add(req.headroom_bytes)
}

/// The longest context that fits at `req`'s settings with `req.expert_slab_bytes` resident, in whole pages, capped at the model's
/// maximum.
pub fn largest_context(plan: &LoadPlan, cfg: &DeepseekV41Config, req: &AdmissionRequest) -> usize {
    let (slots, rows) = (req.slots, req.prefill_rows);
    if slots == 0 || rows == 0 {
        return 0;
    }
    let device_bytes = req.device_bytes;
    let (slab, window) = expert_terms(plan, req.expert_slab_bytes, req.window_bytes);
    // the reserve is a percentage of the allocation, so price it at this shape's allocation once and then walk down in pages
    let base = |m: &Moved| baseline(plan, req, m).saturating_add(slab).saturating_add(window);
    let per_token = kv_bytes_per_token(cfg, slots, req.kv).max(1);
    let at_zero = Moved::at(cfg, req, 0);
    let fixed = base(&at_zero) + reserve(req, base(&at_zero));
    if fixed >= device_bytes {
        return 0;
    }
    // the KV term is linear in the context and the scratch's block scores are too: divide, then walk down in pages until it fits
    let pages = alloc::PAGE_TOKENS.max(1) as u64;
    let mut ctx = ((device_bytes - fixed) / per_token).min(cfg.max_positions as u64);
    ctx -= ctx % pages; // whole pages: a context is a page count
    let mut ctx = ctx as usize;
    while ctx > 0 {
        let m = Moved::at(cfg, req, ctx);
        let b = base(&m);
        if b + reserve(req, b) <= device_bytes {
            break;
        }
        ctx = ctx.saturating_sub(alloc::PAGE_TOKENS.max(1));
    }
    let _ = rows;
    ctx
}

/// The largest expert slab that fits at `req`'s settings (0 when even the resident set and the cache do not).
///
/// The runtime reserve is 0.4% of the allocation, so the slab is solved for rather than subtracted: with `k = 4/1000`,
/// `total = base + C + k * (resident + window + kv + slab) + slab`, which inverts in one step. The answer is then checked
/// against [`admit`]'s own arithmetic, which floors, so it never over-states.
pub fn largest_slab(plan: &LoadPlan, cfg: &DeepseekV41Config, req: &AdmissionRequest) -> u64 {
    let m = Moved::at(cfg, req, req.context);
    // every expert resident: no streaming window is charged (`expert_terms`), so check `window = 0` first
    let all_b = baseline(plan, req, &m).saturating_add(plan.expert_bytes);
    let all_alloc =
        plan.resident_bytes.saturating_add(plan.mtp_expert_bytes).saturating_add(plan.expert_bytes).saturating_add(m.kv);
    if all_b.saturating_add(reserve(req, all_alloc)) <= req.device_bytes {
        return plan.expert_bytes;
    }
    // a slab that leaves a tail pays for the window; one that holds every expert returned above, with no window
    let window = req.window_bytes;
    let rest = baseline(plan, req, &m).saturating_add(window);
    let allocated = plan.resident_bytes.saturating_add(plan.mtp_expert_bytes).saturating_add(window).saturating_add(m.kv);
    let room = match req.runtime_reserve_bytes {
        Some(flat) => req.device_bytes.saturating_sub(rest).saturating_sub(flat),
        None => {
            let reserve_at_zero = runtime_reserve_for(allocated);
            let num = req.device_bytes.saturating_sub(rest).saturating_sub(reserve_at_zero);
            // total = rest + reserve_at_zero + slab * (1 + k): one division, floored
            (num as u128 * 1_000 / (1_000 + RUNTIME_RESERVE_PER_MILLE as u128)) as u64
        }
    };
    let slab = room.min(plan.expert_bytes);
    // every expert resident: the window is no longer charged, and whatever the slab did not need is free
    if slab >= plan.expert_bytes {
        return plan.expert_bytes;
    }
    // the flooring above can leave the total a few hundred bytes over on a device that is exactly full; walk back by the overage
    let mut slab = slab;
    for _ in 0..4 {
        let (s, w) = expert_terms(plan, slab, req.window_bytes);
        let b = baseline(plan, req, &m).saturating_add(s).saturating_add(w);
        let total = b.saturating_add(reserve(req, b));
        if total <= req.device_bytes {
            break;
        }
        slab = slab.saturating_sub(total - req.device_bytes);
    }
    slab
}

/// Decide whether `plan` can be served at `req` on `cfg`, and return every byte of the answer.
///
/// The order of the checks is the order a failure is worth reporting in: the model's own position limit, the resident set, the
/// context, then the expert slab. Nothing is reduced without being asked to be, and every reduction is in the result.
pub fn admit(plan: &LoadPlan, cfg: &DeepseekV41Config, req: &AdmissionRequest) -> Result<Admission, AdmissionError> {
    if req.context == 0 {
        return Err(AdmissionError::Empty { what: "context" });
    }
    if req.slots == 0 {
        return Err(AdmissionError::Empty { what: "slots" });
    }
    if req.prefill_rows == 0 {
        return Err(AdmissionError::Empty { what: "prefill rows" });
    }
    if req.context > cfg.max_positions {
        return Err(AdmissionError::ContextBeyondModel { context: req.context, max_positions: cfg.max_positions });
    }
    if plan.resident_bytes.saturating_add(plan.mtp_expert_bytes).saturating_add(req.headroom_bytes) > req.device_bytes {
        return Err(AdmissionError::ResidentOverflow {
            resident_bytes: plan.resident_bytes + plan.mtp_expert_bytes,
            device_bytes: req.device_bytes,
            over_by: plan.resident_bytes + plan.mtp_expert_bytes + req.headroom_bytes - req.device_bytes,
        });
    }
    let (slab, window) = expert_terms(plan, req.expert_slab_bytes, req.window_bytes);
    let t = Moved::at(cfg, req, req.context);

    let build = |context: usize, slab: u64, window: u64, t: &Moved| {
        let allocated = plan
            .resident_bytes
            .saturating_add(plan.mtp_expert_bytes)
            .saturating_add(slab)
            .saturating_add(window)
            .saturating_add(t.kv);
        let runtime = reserve(req, allocated);
        let total =
            allocated.saturating_add(t.scratch).saturating_add(t.rope).saturating_add(runtime).saturating_add(req.headroom_bytes);
        Admission {
            device_bytes: req.device_bytes,
            context,
            requested_context: req.context,
            slots: req.slots,
            kv: req.kv,
            resident_bytes: plan.resident_bytes,
            mtp_expert_bytes: plan.mtp_expert_bytes,
            expert_slab_bytes: slab,
            expert_window_bytes: window,
            streamed_expert_bytes: plan.expert_bytes - slab,
            kv_bytes: t.kv,
            scratch_bytes: t.scratch,
            rope_bytes: t.rope,
            runtime_reserve_bytes: runtime,
            headroom_bytes: req.headroom_bytes,
            allocated_bytes: allocated,
            total_bytes: total,
            free_bytes: req.device_bytes.saturating_sub(total),
        }
    };

    // the context is judged on the terms it moves (resident + KV + scratch + rope + reserve + headroom); the slab is judged after
    // it, so a slab that does not fit is never reported as a context problem
    let fits_kv = |context: usize| {
        let b = baseline(plan, req, &Moved::at(cfg, req, context));
        b + reserve(req, b) <= req.device_bytes
    };

    if !fits_kv(req.context) {
        // the term that overflowed is the KV one (the context is the only thing that moved): name it, and the context that would fit
        let per_token = kv_bytes_per_token(cfg, req.slots, req.kv).max(1);
        let b = baseline(plan, req, &t);
        let over_by = b + reserve(req, b) - req.device_bytes;
        let largest = largest_context(plan, cfg, req);
        if req.shrink_context && largest > 0 {
            return Ok(build(largest, slab, window, &Moved::at(cfg, req, largest)));
        }
        return Err(AdmissionError::ContextOverflow {
            context: req.context,
            slots: req.slots,
            device_bytes: req.device_bytes,
            resident_bytes: plan.resident_bytes,
            expert_slab_bytes: slab,
            scratch_bytes: t.scratch,
            headroom_bytes: req.headroom_bytes,
            kv_bytes: t.kv,
            kv_per_token: per_token,
            over_by,
            largest_context: largest,
        });
    }
    let admitted = build(req.context, slab, window, &t);
    if admitted.total_bytes > req.device_bytes {
        let largest = largest_slab(plan, cfg, req);
        if req.shrink_slab {
            let (slab, window) = expert_terms(plan, largest, req.window_bytes);
            return Ok(build(req.context, slab, window, &t));
        }
        return Err(AdmissionError::ExpertSlabOverflow {
            requested: slab,
            largest,
            over_by: admitted.total_bytes - req.device_bytes,
        });
    }
    Ok(admitted)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::deepseek_v41::bind::bind;
    use crate::families::deepseek_v41::inventory::{MtpExpertFormat, PackInventory};
    use crate::families::deepseek_v41::load::{self, Layout};
    use serde_json::Value;

    const CONFIG: &str = include_str!("../../../../../testdata/deepseek_v41/config.json");
    const PACK: &str = "/tmp/dsv41_inventory.json";
    const GIB: u64 = 1 << 30;

    fn real() -> DeepseekV41Config {
        DeepseekV41Config::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    /// The pack's bound tree, or `None` when this machine does not have its listing.
    fn weights() -> Option<DeepseekV41Weights> {
        let text = std::fs::read_to_string(PACK).ok()?;
        let listing: Value = serde_json::from_str(&text).unwrap();
        let cfg = real();
        let inv = PackInventory::from_json(&cfg, &listing).unwrap();
        Some(bind(&inv, &cfg).unwrap())
    }

    fn plan() -> Option<LoadPlan> {
        let w = weights()?;
        Some(LoadPlan::build(&w, 0).unwrap())
    }

    /// The measured budget: every class, and the pack's own 332,513,253,720 B.
    #[test]
    fn the_plan_sums_to_the_measured_pack() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let p = LoadPlan::build(&w, 0).unwrap();
        assert_eq!(p.total_bytes(), 332_513_253_720, "309.7 GiB of tensors");
        assert_eq!(p.items.len(), 188_245, "one item a tensor");
        // the three placements, exactly
        assert_eq!(p.sidecar_bytes, 202_758_032_400, "the two engram tables");
        assert_eq!(p.expert_bytes, 111_005_061_120, "46,080 routed EXL3 experts");
        assert_eq!(p.resident_bytes, 11_530_714_440, "everything but the routed and the NextN experts");
        assert_eq!(p.mtp_expert_bytes, 7_219_445_760, "the first upload's 384 int8 NextN experts");
        // and they add up both ways
        assert_eq!(p.resident_bytes + p.mtp_expert_bytes + p.expert_bytes + p.sidecar_bytes, p.total_bytes());
        assert_eq!(p.load_item_bytes(), 129_755_221_320, "the pack minus its mmap sidecar");
        // 4 sidecar tensors, 184,320 expert tensors, the rest resident
        assert_eq!(p.items_in(Placement::Sidecar).count(), 4);
        assert_eq!(p.items_in(Placement::ExpertCache).count(), 40 * 384 * 3 * 4);
        assert_eq!(p.items_in(Placement::Resident).count(), 188_245 - 4 - 184_320 - 2_304);
        assert_eq!(p.items_in(Placement::MtpExpert).count(), 2_304, "384 int8 experts x 3 projections x 2 parts");
        let by = p.by_placement();
        assert_eq!(by[&Placement::Resident], 11_530_714_440);
        assert_eq!(by[&Placement::MtpExpert], 7_219_445_760);
        assert_eq!(p.sidecar().count(), 4);
        // the classes the alloc arithmetic derives from the config alone agree with the bound tree, item for item
        let c = real();
        let comp = p.by_component();
        let cfg_bytes: u64 = alloc::config_classes(&c, MtpExpertFormat::Int4).iter().map(|i| i.bytes).sum();
        assert_eq!(cfg_bytes + p.expert_bytes, p.total_bytes());
        assert_eq!(comp[&Component::MlaQ] + comp[&Component::MlaKv] + comp[&Component::MlaO], alloc::mla_bytes(&c));
        assert_eq!(comp[&Component::Norm], alloc::norm_bytes(&c));
        assert_eq!(comp[&Component::MoeShared], alloc::shared_expert_bytes(&c));
        assert_eq!(comp[&Component::DsaIndex], alloc::indexer_bytes(&c));
        assert_eq!(comp[&Component::DsaCompressor], alloc::compressor_bytes(&c));
        assert_eq!(comp[&Component::MoeRouter], alloc::router_bytes(&c));
        assert_eq!(comp[&Component::Engram], alloc::engram_table_bytes(&c) + alloc::engram_branch_bytes(&c));
        assert_eq!(comp[&Component::Mtp], alloc::mtp_bytes(&c, MtpExpertFormat::Int4));
    }

    /// **The re-upload, end to end.** The complete header listing of
    /// `vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw` (190,549 tensors, EXL3 `K2` NextN experts) when this machine has it:
    /// inventory, bind, plan and layout, with the drafter's 384 experts resident and outside the slab.
    #[test]
    fn the_reuploaded_pack_inventories_binds_and_prices_as_exl3() {
        const PACK_V2: &str = "/tmp/dsv41_inventory_v2.json";
        let Ok(text) = std::fs::read_to_string(PACK_V2) else {
            eprintln!("not run: {PACK_V2} is not on this machine");
            return;
        };
        let listing: Value = serde_json::from_str(&text).unwrap();
        let c = real();
        let inv = PackInventory::from_json(&c, &listing).unwrap();
        assert_eq!(inv.mtp_experts(), MtpExpertFormat::Exl3, "the re-upload's NextN experts are EXL3 groups");
        assert_eq!(
            inv.len(),
            190_549,
            "2,304 more tensors than the first upload: four parts an EXL3 projection, two an int8 one"
        );
        let w = bind(&inv, &c).unwrap();
        let p = LoadPlan::build(&w, 0).unwrap();
        assert_eq!(p.total_bytes(), 328_708_303_704, "the pack is 3.545 GiB smaller than the first upload");
        assert_eq!(p.mtp_expert_bytes, 3_414_495_744, "384 K2 experts, 3.180 GiB");
        assert_eq!(p.expert_bytes, 111_005_061_120, "the trunk's experts are unchanged");
        assert_eq!(p.resident_bytes, 11_530_714_440, "every non-expert tensor, byte for byte the first upload's");
        // the format is priced as EXL3: the config classes agree with the bound tree
        let classes: u64 = alloc::config_classes(&c, MtpExpertFormat::Exl3).iter().map(|i| i.bytes).sum();
        assert_eq!(classes + p.expert_bytes, p.total_bytes());
        // every NextN expert is resident, in a region of its own, and none of them is a slab decision
        let layout = Layout::new(&w, &p, 0, 0).unwrap();
        load::check_agreement(&w, &p, &layout).unwrap();
        // the regions hold whole records: 252 B of part alignment padding an expert, 290,304 B over the 384
        assert_eq!(layout.mtp_expert_bytes, 3_414_786_048);
        assert_eq!(layout.slab_bytes, 0, "no budget: the trunk's experts all stream, the drafter's do not");
        assert_eq!(layout.streamed_bytes, p.expert_bytes);
        let mtp = layout.regions.iter().filter(|r| matches!(r.kind, load::RegionKind::MtpExperts(_))).count();
        assert_eq!(mtp, 3, "one region a NextN layer");
        let views = layout
            .bindings
            .iter()
            .filter(|(n, _)| n.starts_with("mtp.") && n.contains(".ffn.experts."))
            .filter(|(_, b)| b.is_region())
            .count();
        assert_eq!(views, 4_608, "3 layers x 128 experts x 3 projections x 4 parts are views, not uploads");
    }

    /// **The re-upload's admission table**, printed: every term admission charges at 32K over one slot, and how far short of
    /// all-experts-resident it is. The numbers the boot's `dsv41_slab` line is read against.
    #[test]
    fn the_reuploads_admission_table() {
        const PACK_V2: &str = "/tmp/dsv41_inventory_v2.json";
        let Ok(text) = std::fs::read_to_string(PACK_V2) else {
            eprintln!("not run: {PACK_V2} is not on this machine");
            return;
        };
        let listing: Value = serde_json::from_str(&text).unwrap();
        let c = real();
        let inv = PackInventory::from_json(&c, &listing).unwrap();
        let w = bind(&inv, &c).unwrap();
        // the Spark's idle `MemAvailable` at stage=admitted, less the 0.5 GiB floor
        let device = (118.118f64 * (1u64 << 30) as f64) as u64 - (1u64 << 29);
        let probe = LoadPlan::build(&w, 0).unwrap();
        let mut req = AdmissionRequest::serving(32_768, 1, device);
        req.expert_slab_bytes = largest_slab(&probe, &c, &req);
        let plan = LoadPlan::build(&w, req.expert_slab_bytes).unwrap();
        let a = admit(&plan, &c, &req).unwrap();
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        for (name, bytes) in a.terms() {
            eprintln!("  {name:>16} {:>15} B  {:.3} GiB", bytes, gib(bytes));
        }
        eprintln!(
            "  slab {:.3} GiB, {} of {} experts, streamed {:.3} GiB, window {:.3} GiB",
            gib(a.expert_slab_bytes),
            plan.resident_experts,
            c.layers * c.moe.routed_experts,
            gib(a.streamed_expert_bytes),
            gib(a.expert_window_bytes)
        );
        let all = plan.resident_bytes + plan.mtp_expert_bytes + plan.expert_bytes + a.kv_bytes + a.scratch_bytes + a.rope_bytes;
        eprintln!(
            "  all-resident short by {:.3} GiB",
            (all + runtime_reserve_for(all)) as f64 / (1u64 << 30) as f64 - gib(device)
        );
        let layout = Layout::new(&w, &plan, a.expert_slab_bytes, a.expert_window_bytes).unwrap();
        eprintln!(
            "  layout regions {:.3} GiB = views {:.3} + window {:.3} + padding {:.3} ({:.2}% of the views)",
            gib(layout.device_bytes()),
            gib(layout.view_bytes),
            gib(layout.window_bytes),
            gib(layout.padding_bytes()),
            100.0 * layout.padding_bytes() as f64 / layout.view_bytes as f64
        );
        assert!(a.expert_slab_bytes > 95u64 << 30, "{}", a.expert_slab_bytes);
        assert!(a.mtp_expert_bytes == 3_414_495_744);
    }

    /// A budget buys whole experts and the plan says which and how many.
    #[test]
    fn an_expert_budget_buys_whole_experts_and_reports_what_streams() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let full = LoadPlan::full(&w).unwrap();
        assert_eq!(full.expert_budget_bytes, full.expert_bytes);
        assert_eq!(full.streamed_expert_bytes, 0);
        assert_eq!(full.resident_experts, 40 * 384);

        // a gigabyte: 138 whole experts at the pack's 7.2 MB average, and the rest streams
        let p = LoadPlan::build(&w, 1 << 30).unwrap();
        assert!(p.expert_budget_bytes <= 1 << 30);
        assert!(p.resident_experts > 100 && p.resident_experts < 160, "{} experts", p.resident_experts);
        assert_eq!(p.expert_budget_bytes + p.streamed_expert_bytes, p.expert_bytes);
        // the resident set does not move with the budget
        assert_eq!(p.resident_bytes, full.resident_bytes);
        assert_eq!(p.sidecar_bytes, full.sidecar_bytes);
        // a budget larger than every expert's aligned record is a plan error, not a silent clamp; exactly that much is every expert
        let records = super::super::alloc::expert_records_bytes(&w);
        assert!(records > full.expert_bytes, "the aligned records carry padding");
        assert!(matches!(LoadPlan::build(&w, records + 1), Err(PlanError::ExpertBudget { .. })));
        assert_eq!(LoadPlan::build(&w, records).unwrap().resident_experts, 40 * 384);
    }

    /// **What admission charges is what the boot allocates.** The KV is the DSA planes (there is no paged latent), the scratch is
    /// the model's own `Scratch`, the headroom is the floor's job and so defaults to zero, and the reserve is the measured model.
    #[test]
    fn admission_prices_only_what_is_allocated() {
        let Some(p) = plan() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let c = real();
        let device = 121 * GIB;
        assert_eq!(device, 129_922_760_704);

        let a = admit(&p, &c, &AdmissionRequest::at_121_gib(32_768, 0)).unwrap();
        assert_eq!(a.resident_bytes, 11_530_714_440);
        assert_eq!(a.mtp_expert_bytes, 7_219_445_760);
        assert_eq!(a.expert_slab_bytes, 0);
        // the KV *is* the planes: the five buffers of all 40 layers, charged once, at the cache's format
        let f16 = attn_planes_bytes(&c, 32_768, 1, KvQuant::F16);
        let q4 = attn_planes_bytes(&c, 32_768, 1, KvQuant::Q4);
        assert_eq!(a.kv_bytes, f16, "fp16 is the default");
        // by hand: 40 x (the 128-row ring + the published selection) + 1,015,848 compressed positions x (entry + key + mask byte);
        // a selection row holds the window's slots and then the compressed top-k (`window + index_topk` entries, i32)
        let ring_and_selection =
            (c.sliding_window * c.head_dim) as u64 * 2 + 128 * (c.sliding_window + c.indexer.topk) as u64 * 4;
        assert_eq!(f16, (c.layers as u64) * ring_and_selection + 1_015_848 * (1_024 + 256 + 128));
        assert_eq!(q4, (c.layers as u64) * ring_and_selection + 1_015_848 * (368 + 80 + 128));
        // the format is the only thing that moves: 2.41x over the whole plane, 2.86x over the compressed entries and keys
        // alone (the ring is fp16 and the selection i32 in both layouts)
        assert!(f16 > 2 * q4, "{f16} vs {q4}");
        // the scratch is the model's own Scratch at the model's row depth, not the nominal 270,295,040 B workspace nothing allocates
        assert_eq!(
            a.scratch_bytes,
            super::super::model::Scratch::bytes_for(&c, model_rows(&AdmissionRequest::at_121_gib(32_768, 0)), 32_768)
        );
        // 101,157,376 B before the routed block moved onto the device; +24,584,704 B is the routed block's own arena
        // (`gb_cuda::dsv4_moe::MoeScratch`: the gathered input, the three packed expert activations at `rows * topk`
        // slots, the grouping, and the gemv workspace) net of the fp16 shared expert's gate/up/act and the `gather_idx`
        // buffer it replaces.
        assert_eq!(a.scratch_bytes, 125_742_080);
        assert!(a.scratch_bytes < 270_295_040, "the nominal workspace is gone");
        assert_eq!(a.rope_bytes, 16_777_216);
        // the floor is the margin: no second gigabyte
        assert_eq!(a.headroom_bytes, DEFAULT_HEADROOM_BYTES);
        assert_eq!(a.headroom_bytes, 0);
        // the runtime reserve is the measured model over the bytes this boot allocates
        assert_eq!(
            a.allocated_bytes,
            a.resident_bytes + a.mtp_expert_bytes + a.expert_slab_bytes + a.expert_window_bytes + a.kv_bytes
        );
        assert_eq!(a.runtime_reserve_bytes, runtime_reserve_for(a.allocated_bytes));
        assert_eq!(a.runtime_reserve_bytes, RUNTIME_RESERVE_BASE_BYTES + a.allocated_bytes / 1_000 * 4);
        // 0.70 GiB plus 0.4% of ~19.8 GB, not the flat 1.5 GiB this replaces
        assert!((0.81e9..0.82e9).contains(&(a.runtime_reserve_bytes as f64)), "{} B", a.runtime_reserve_bytes);
        // and the total is exactly the named terms
        assert_eq!(a.terms().iter().map(|(_, b)| b).sum::<u64>() + a.headroom_bytes, a.total_bytes);
        assert_eq!(a.free_bytes, device - a.total_bytes);
        assert!(!a.all_experts_resident());
        assert!(a.context_honoured());
        // no term is charged twice: the KV appears once, and the window is the slab's business
        let names: Vec<&str> = a.terms().iter().map(|(n, _)| *n).collect();
        assert_eq!(
            names,
            ["resident", "mtp_experts", "expert_slab", "expert_window", "kv", "scratch", "rope", "runtime_reserve"]
        );
        for (name, bytes) in a.terms() {
            assert!(bytes != a.streamed_expert_bytes, "`{name}` is charged the tail");
        }
    }

    /// The scheduler prices its KV from the same planes formula admission charges: the per-token figure is the derivative of
    /// [`attn_planes_bytes`], so `gb_sched`'s budget and ours cannot drift.
    #[test]
    fn the_per_token_kv_is_the_planes_derivative() {
        let c = real();
        let dcfg = kernel_cfg(&c);
        for slots in [1usize, 8] {
            for kv in [KvQuant::F16, KvQuant::Q4] {
                let per_token = kv_bytes_per_token(&c, slots, kv);
                // one page of context, through the constructor's own formula
                let a = attn_planes_bytes(&c, 65_536 + alloc::PAGE_TOKENS, slots, kv);
                let b = attn_planes_bytes(&c, 65_536, slots, kv);
                let measured = (a - b) / alloc::PAGE_TOKENS as u64;
                assert_eq!(per_token, measured, "{slots} slot(s), {kv:?}");
                // per layer: one compressed entry + one index key + one mask column, over that layer's own ratio
                let by_hand: u64 = (0..c.layers)
                    .map(|l| {
                        (gb_cuda::dsv4_attn::Dsv4Planes::bytes_per_position(&dcfg, kv.fmt()) as u64) / dcfg.ratio(l).max(1) as u64
                    })
                    .sum();
                assert_eq!(per_token, by_hand * slots as u64);
            }
        }
        // fp16: 18 ratio-2 layers at 704 B a token and 22 at 1,408 (the ratio-1 layers, and the two that pool nothing but
        // still own a plane); q4 is 2.44x less a token
        assert_eq!(kv_bytes_per_token(&c, 1, KvQuant::F16), 18 * 704 + 22 * 1_408);
        assert_eq!(kv_bytes_per_token(&c, 1, KvQuant::Q4), 18 * 288 + 22 * 576);
    }

    /// **Every expert resident: no tail, and therefore no window.** A device that holds the whole bank pays for the bank and
    /// nothing else — the streaming window is not a fixed cost of the model, it is the price of a tail.
    #[test]
    fn a_device_that_holds_every_expert_charges_no_window_and_streams_nothing() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let c = real();
        let full = LoadPlan::full(&w).unwrap();
        let records = super::super::alloc::expert_records_bytes(&w);
        let probe = AdmissionRequest::serving(32_768, 1, 121 * GIB);
        let rest = full.resident_bytes
            + full.mtp_expert_bytes
            + attn_planes_bytes(&c, 32_768, 1, KvQuant::F16)
            + super::super::model::Scratch::bytes_for(&c, model_rows(&probe), 32_768)
            + rope_bytes(&c, 32_768);
        // a device large enough for the whole bank plus every other term
        let device = rest + records + runtime_reserve_for(rest + records) + (1 << 20);
        let req = AdmissionRequest { expert_slab_bytes: records, ..AdmissionRequest::serving(32_768, 1, device) };
        let a = admit(&full, &c, &req).unwrap();
        assert_eq!(a.expert_window_bytes, 0, "no tail, no window");
        assert_eq!(a.streamed_expert_bytes, 0);
        assert!(a.all_experts_resident());
        assert_eq!(a.expert_slab_bytes, full.expert_bytes);
        assert_eq!(a.total_bytes, a.allocated_bytes + a.scratch_bytes + a.rope_bytes + a.runtime_reserve_bytes);
        // and the same request on a device a gigabyte short of the bank streams, and pays for a window
        let short = AdmissionRequest {
            expert_slab_bytes: records - GIB,
            ..AdmissionRequest::serving(32_768, 1, device + DEFAULT_STREAM_WINDOW_BYTES)
        };
        let a = admit(&full, &c, &short).unwrap();
        assert!(a.expert_window_bytes > 0, "a tail needs somewhere to land");
        assert!(!a.all_experts_resident());
    }

    /// The arithmetic admission is judged by, at the target device, written out.
    #[test]
    fn admission_at_121_gib_and_32k_prices_every_term() {
        let Some(p) = plan() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let c = real();
        let device = 121 * GIB;
        let req = AdmissionRequest::at_121_gib(32_768, 0);
        // the shape is admitted with no cache at all
        admit(&p, &c, &req).unwrap();

        // the largest slab that fits at those settings: every term but the slab, solved for the 0.4% reserve
        let slab = largest_slab(&p, &c, &req);
        let a2 = admit(&p, &c, &AdmissionRequest { expert_slab_bytes: slab, ..req }).unwrap();
        assert!(a2.total_bytes <= device, "{} > {device}", a2.total_bytes);
        assert!(device - a2.total_bytes < 1 << 20, "{} B of slack", device - a2.total_bytes);
        assert_eq!(a2.expert_slab_bytes, slab);
        assert_eq!(a2.streamed_expert_bytes, p.expert_bytes - slab);
        // 2.2 GB of experts stream: about 7 of the 384 experts of a layer. With the duplicate window, the double-priced KV, the
        // nominal scratch, the headroom and the flat reserve all gone, 121 GiB holds 98.0% of the bank
        // (the 0.70 GiB reserve base, measured on the Spark, streams ~0.48 GB more than the earlier 0.25 GiB base did)
        assert!((2.2e9..3.1e9).contains(&(a2.streamed_expert_bytes as f64)), "{} B streams", a2.streamed_expert_bytes);

        // and the whole pack resident does not fit: the slab term overflows, by name
        let e = admit(&p, &c, &AdmissionRequest { expert_slab_bytes: p.expert_bytes, ..req }).unwrap_err();
        match e {
            AdmissionError::ExpertSlabOverflow { requested, largest, over_by } => {
                assert_eq!(requested, 111_005_061_120);
                assert_eq!(largest, slab);
                assert!(over_by > 1_000_000_000, "{over_by}");
            }
            other => panic!("expected ExpertSlabOverflow, got {other}"),
        }
        // asked to shrink, it admits and says what it shrank to
        let a = admit(&p, &c, &AdmissionRequest { shrink_slab: true, expert_slab_bytes: p.expert_bytes, ..req }).unwrap();
        assert_eq!(a.expert_slab_bytes, slab);
        assert!(a.total_bytes <= device);
    }

    /// **Why the pack boots at all.** On a GB10 the usable pool is `MemAvailable` less a 0.5 GiB floor, shared with the host. The
    /// tail is 20+ GiB of experts and it is *disk*. Charging it — the shape of the old layout, which staged the tail in a
    /// host-RAM region — does not fit; charging the 4 GiB window does, with room for the slab the tail displaced.
    #[test]
    fn the_gb10_boot_charges_the_window_and_never_the_tail() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let c = real();
        let usable = 110u64 << 30;
        let probe = LoadPlan::build(&w, 0).unwrap();
        let mut req = AdmissionRequest::serving(32_768, 1, usable);
        let slab = largest_slab(&probe, &c, &req);
        req.expert_slab_bytes = slab;
        let plan = LoadPlan::build(&w, slab).unwrap();
        let a = admit(&plan, &c, &req).unwrap();

        // the boot is admitted, and the terms add up exactly
        assert!(a.total_bytes <= usable, "{} > {usable}", a.total_bytes);
        assert_eq!(a.terms().iter().map(|(_, b)| b).sum::<u64>() + a.headroom_bytes, a.total_bytes);
        assert_eq!(a.resident_bytes, 11_530_714_440);
        assert_eq!(a.mtp_expert_bytes, 7_219_445_760);
        assert_eq!(a.expert_window_bytes, DEFAULT_STREAM_WINDOW_BYTES);
        // the tail is what streams from disk, and it is nowhere in the total
        assert!(a.streamed_expert_bytes > 17_000_000_000, "{} B", a.streamed_expert_bytes);
        // AND the old arithmetic — the tail as a staging region — would not have fitted here
        let charged_tail = a.total_bytes - a.expert_window_bytes + a.streamed_expert_bytes;
        assert!(charged_tail > usable, "the tail-sized region fits, so this test proves nothing: {charged_tail}");
        // the message says so in as many words
        let text = a.to_string();
        assert!(text.contains("window") && text.contains("from disk"), "{text}");
    }

    /// 32K fits with no expert cache and with one: the plan is admissible, and the slack is reported rather than spent.
    #[test]
    fn a_32k_context_is_admitted_with_zero_experts_cached() {
        let Some(p) = plan() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let c = real();
        // slab 0 is the minimum: admission succeeds there, and the accounting is complete
        let a = admit(&p, &c, &AdmissionRequest::at_121_gib(32_768, 0)).unwrap();
        assert_eq!(a.terms().iter().map(|(_, b)| b).sum::<u64>() + a.headroom_bytes, a.total_bytes);
        assert!(a.free_bytes > 100_000_000_000);
        // the full 1,048,576-token model context also fits with no expert cache
        let a = admit(&p, &c, &AdmissionRequest::at_121_gib(1_048_576, 0)).unwrap();
        assert!(a.context_honoured());
        // the planes are 42.6 GiB at a 1M context, which is why a 1M-token boot is 61.7 GB before an expert is resident
        assert_eq!(a.kv_bytes, attn_planes_bytes(&c, 1_048_576, 1, KvQuant::F16));
        assert!(a.kv_bytes > 45_000_000_000, "{} B of planes", a.kv_bytes);
        assert!(a.free_bytes > 60_000_000_000, "{} B free", a.free_bytes);
        assert!(a.scratch_bytes < 1_000_000_000, "{} B of scratch", a.scratch_bytes);
    }

    /// 1M tokens over the server's 8 slots does not fit: a named context overflow, and the largest context that does.
    #[test]
    fn a_million_token_context_overflows_by_name_and_is_not_silently_shrunk() {
        let Some(p) = plan() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let c = real();
        let req = AdmissionRequest { slots: 8, ..AdmissionRequest::at_121_gib(1_048_576, 0) };
        let e = admit(&p, &c, &req).unwrap_err();
        match e {
            AdmissionError::ContextOverflow {
                context,
                slots,
                kv_bytes,
                kv_per_token,
                over_by,
                largest_context,
                scratch_bytes,
                ..
            } => {
                assert_eq!((context, slots), (1_048_576, 8));
                assert_eq!(kv_per_token, 8 * 43_648, "43,648 B a token at fp16");
                assert_eq!(kv_bytes, attn_planes_bytes(&c, 1_048_576, 8, KvQuant::F16));
                assert_eq!(scratch_bytes, super::super::model::Scratch::bytes_for(&c, super::model_rows(&req), 1_048_576));
                // every term that is not the slab, plus the reserve over it
                let b = 18_750_160_200 + kv_bytes + scratch_bytes + rope_bytes(&c, 1_048_576);
                assert_eq!(over_by, b + runtime_reserve_for(b) - (121 * GIB));
                // the planes are priced per slot — 42.6 GiB x 8 at a 1M context — so the 8 slots share well under the full context
                assert!((250_000..400_000).contains(&largest_context), "{largest_context}");
                assert_eq!(largest_context % alloc::PAGE_TOKENS, 0);
                // the message names the term and the two numbers
                let text = AdmissionError::ContextOverflow {
                    context,
                    slots,
                    device_bytes: 121 * GIB,
                    resident_bytes: 18_750_160_200,
                    expert_slab_bytes: 0,
                    scratch_bytes,
                    headroom_bytes: 0,
                    kv_bytes,
                    kv_per_token,
                    over_by,
                    largest_context,
                }
                .to_string();
                assert!(text.contains("KV") && text.contains("overflows the device by"), "{text}");
                assert!(text.contains(&largest_context.to_string()), "{text}");
            }
            other => panic!("expected ContextOverflow, got {other}"),
        }
        // asked to shrink, it admits the longest context that fits and says which one it admitted
        let a = admit(&p, &c, &AdmissionRequest { shrink_context: true, slots: 8, ..AdmissionRequest::at_121_gib(1_048_576, 0) })
            .unwrap();
        assert!(a.context < 1_048_576 && a.context > 100_000, "{}", a.context);
        assert_eq!((a.requested_context, a.context_honoured()), (1_048_576, false));
        // on ONE slot, 1M fits with no expert cache; with the slab admitted at 32K it is the slab term that overflows, by name
        let slab = largest_slab(&p, &c, &AdmissionRequest::at_121_gib(32_768, 0));
        assert!(admit(&p, &c, &AdmissionRequest::at_121_gib(1_048_576, 0)).is_ok());
        match admit(&p, &c, &AdmissionRequest::at_121_gib(1_048_576, slab)).unwrap_err() {
            AdmissionError::ExpertSlabOverflow { requested, largest, over_by } => {
                assert_eq!(requested, slab);
                // every term at 1M and one slot, over the device
                let m = Moved::at(&c, &AdmissionRequest::at_121_gib(1_048_576, 0), 1_048_576);
                let allocated = 18_750_160_200 + slab + m.kv;
                assert_eq!(over_by, allocated + m.scratch + m.rope + runtime_reserve_for(allocated) - 121 * GIB);
                assert!(over_by > 0);
                assert_eq!(largest, largest_slab(&p, &c, &AdmissionRequest::at_121_gib(1_048_576, 0)));
                assert!(largest < slab);
            }
            other => panic!("expected ExpertSlabOverflow, got {other}"),
        }
    }

    /// The priced table of this pack, printed: every term admission charges at 32K over one slot, and at the model's full context.
    /// This is the number an operator reads off a boot, and the one the 0.5 GiB floor is spent against.
    #[test]
    fn the_priced_table_of_this_pack() {
        let Some(p) = plan() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let c = real();
        for ctx in [32_768usize, 1_048_576] {
            let req = AdmissionRequest::at_121_gib(ctx, 0);
            let a = admit(&p, &c, &req).unwrap();
            let mut sum = 0u64;
            for (name, bytes) in a.terms() {
                eprintln!("  {name:>16} {bytes:>15} B  {:.3} GiB", bytes as f64 / (1u64 << 30) as f64);
                sum += bytes;
            }
            eprintln!(
                "  ctx {ctx}: terms {sum} + headroom {} = {} (total {}), free {}",
                a.headroom_bytes,
                sum + a.headroom_bytes,
                a.total_bytes,
                a.free_bytes
            );
            // the priced total is exactly the sum of the named terms
            assert_eq!(sum + a.headroom_bytes, a.total_bytes);
            // the helpers are the constructors' own formulas, term by term
            assert_eq!(a.kv_bytes, super::attn_planes_bytes(&c, ctx, 1, KvQuant::F16));
            assert_eq!(a.scratch_bytes, super::super::model::Scratch::bytes_for(&c, super::model_rows(&req), ctx));
            assert_eq!(a.rope_bytes, super::rope_bytes(&c, ctx));
        }
        // 32K, one slot: the rope tables are two distinct rotations, not 43 layers of them
        let a = admit(&p, &c, &AdmissionRequest::at_121_gib(32_768, 0)).unwrap();
        assert_eq!(a.rope_bytes, 2 * 2 * 32_768 * 32 * 4, "two tables of 32K x (64/2) f32, cos and sin");
        assert!(a.kv_bytes > 900_000_000, "{} B of DSA planes", a.kv_bytes);
        assert_eq!(a.runtime_reserve_bytes, runtime_reserve_for(a.allocated_bytes));
    }

    /// The longest context that fits is a function of the slab: each gigabyte of experts costs tokens at one slot.
    #[test]
    fn the_largest_context_trades_against_the_expert_slab() {
        let Some(p) = plan() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let c = real();
        let at = |slab: u64| largest_context(&p, &c, &AdmissionRequest::at_121_gib(32_768, slab));
        // no cache: the model's whole 1,048,576 positions fit
        assert_eq!(at(0), 1_048_576);
        // 100 GB of experts leaves 121 GiB - 18.75 - 100 - scratch - reserve for the planes: 43,648 B a token of compressed
        // entries, indexer keys and mask columns, so ~245,000 tokens
        let ctx = at(100_000_000_000);
        assert!((200_000..300_000).contains(&ctx), "{ctx}");
        assert_eq!(ctx % alloc::PAGE_TOKENS, 0);
        // 10 GB of cache still leaves more than the model's positions
        assert_eq!(at(10_000_000_000), 1_048_576);
        let mut last = usize::MAX;
        for gb in [40u64, 80, 100, 108] {
            let ctx = at(gb * 1_000_000_000);
            assert!(ctx <= last, "{gb} GB: {ctx} against {last}");
            last = ctx;
        }
        // and a device too small for the resident set admits nothing
        assert_eq!(
            largest_context(
                &p,
                &c,
                &AdmissionRequest { device_bytes: 1 << 30, headroom_bytes: 0, ..AdmissionRequest::at_121_gib(4_096, 0) }
            ),
            0
        );
        assert!(matches!(
            admit(&p, &c, &AdmissionRequest { device_bytes: 1 << 30, ..AdmissionRequest::at_121_gib(4_096, 0) }),
            Err(AdmissionError::ResidentOverflow { .. })
        ));
    }

    /// A context longer than the model's positions is refused before any memory is priced.
    #[test]
    fn a_context_past_the_models_positions_is_refused() {
        let Some(p) = plan() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let c = real();
        assert_eq!(c.max_positions, 1_048_576);
        let e = admit(&p, &c, &AdmissionRequest::at_121_gib(1_048_577, 0)).unwrap_err();
        assert!(matches!(e, AdmissionError::ContextBeyondModel { context: 1_048_577, max_positions: 1_048_576 }));
        for bad in [
            AdmissionRequest { context: 0, ..AdmissionRequest::at_121_gib(1_024, 0) },
            AdmissionRequest { slots: 0, ..AdmissionRequest::at_121_gib(1_024, 0) },
            AdmissionRequest { prefill_rows: 0, ..AdmissionRequest::at_121_gib(1_024, 0) },
        ] {
            assert!(matches!(admit(&p, &c, &bad), Err(AdmissionError::Empty { .. })), "{bad:?}");
        }
    }
}
