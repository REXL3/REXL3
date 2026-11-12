//! Realising the load plan on the device: the byte layout the plan's classification implies, one `cudaMalloc` per region, and every
//! view-bound tensor through the Phase 3 pipelined loader (`gb_cuda::loader`).
//!
//! [`super::plan`] *classifies* — resident, expert-cache, sidecar — and prices admission from the classification. This module turns
//! the classification into addresses, and it is the only place in the family that knows how a tensor name becomes an offset:
//!
//! * [`Binding::View`] — the tensor lands in a device region at a known offset and the kernel reads it there: the embedding and the
//!   head, the two norms a layer, the engram branch's two small projections, and every routed expert (through
//!   `Exl3Weights::from_parts`, whose four buffers are views).
//! * [`Binding::Upload`] — the kernel's constructor takes the pack's *host* bytes, so the tensor is not in a region: `Fp8Weights::new`
//!   for every FP8 matrix (the MLA's five projections a layer, the shared expert's three, the engram's fused `wkv`), `Dsv4Router::new`
//!   for the router, `HcSite::new` for the two mixing sites, and `MlaDsv4Pack`'s `&[f32]` norms and sink. `device.rs` uploads these
//!   from the shard's bytes at bind time (`cudaMemcpy` per tensor, not the pipelined loader).
//! * [`Binding::Mapped`] — the two engram tables: `mmap`ped from their shards by [`super::engram`] and addressed in place. Never copied,
//!   never in a region.
//!
//! Three of those are a deviation from "everything resident goes through the loader", and it is forced: the kernel crate's FP8 and
//! router types own their buffers and expose no constructor from a `DevBuf`. The layout therefore *excludes* upload-bound tensors from
//! the regions (no double allocation) and accounts for them separately, so that
//!
//! ```text
//! plan.resident_bytes == layout.view_bytes + layout.upload_bytes
//! plan.sidecar_bytes  == layout.mapped_bytes
//! ```
//!
//! is an equality [`check_agreement`] enforces item by item. A plan and a device that disagree fail the load, naming the tensor: that is
//! what makes the admission arithmetic worth anything.
//!
//! The pool is one unified memory shared with the host and `cudaMalloc` commits eagerly, so the regions are allocated one by one with
//! `MemAvailable` checked before each: a load that would leave less than `floor_bytes` is refused before it can over-commit.

use super::config::DeepseekV41Config;
use super::plan::{LoadPlan, Placement};
use super::weights::{DeepseekV41Weights, Dtype};
use crate::error::ModelError;
use gb_cuda::loader::{Loader, Options, Report, Xform as CopyXform};
use gb_cuda::rt::Block;
use gb_cuda::DevBuf;
use gb_exl3::checkpoint::{Checkpoint, Entry};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// Per-tensor placement alignment: 256 B, the strictest alignment any kernel of this family needs
/// ([`gb_cuda::expert_cache::EXPERT_PART_ALIGN`]).
///
/// **What the loader actually requires.** [`run`] hands every tensor to [`gb_cuda::loader::Loader::copy_file`], which reads the pack
/// through a *staging* buffer — one page-aligned `AlignedBuf` per worker, `o.slots` of them — and then `cudaMemcpyAsync`es each
/// span's items into the destination at `it.job_off` bytes past the tensor's start (`loader.rs` `run_span`). The `O_DIRECT`
/// invariants are therefore all on the *host* side, and none of them is on the device destination:
///
/// * the read offset is `sp.start & !(PAGE - 1)` — page-aligned by construction;
/// * the read length is `a1 - a0` with `a0`/`a1` both page multiples — a page multiple by construction;
/// * the staging buffer is `Layout::from_size_align(cap, PAGE)` — page-aligned by construction;
/// * the destination is only ever the target of a `cudaMemcpyAsync`, which needs natural alignment for the copy width (16 B for the
///   `f16`/`f32` vector loads this pack's view tensors are read with, 256 B for an EXL3 expert part) and no relation at all to the
///   file offset.
///
/// So the old rule — round every tensor up to 2 MiB — bought nothing the loader needed and cost 2.74 GiB of region padding on this
/// pack (3.69 GiB over every trunk/MTP/vision tensor, counting the upload-bound ones no region ever held), which is 2.5% of a 121 GiB
/// box that could have been experts. 256 B is under a quarter of a KiB of slack a tensor; a page would also be correct and costs 4 KiB
/// a tensor, which is 6% of a tower of small tensors and nothing of this one.
pub const READ_ALIGN: u64 = gb_cuda::expert_cache::EXPERT_PART_ALIGN;

/// The alignment a device destination must satisfy for the kernels that read it: 16 B for the `f16`/`f32` view tensors, 256 B for
/// an EXL3 expert part. [`READ_ALIGN`] is that 256, and [`Layout::new`] asserts every view is on it.
pub const MIN_DST_ALIGN: u64 = gb_cuda::expert_cache::EXPERT_PART_ALIGN;

/// The alignment a *region* is rounded up to. `cudaMalloc` returns 256 B-aligned memory and nothing here needs more: the loader
/// copies item-wise out of a page-aligned staging buffer, so a region's own base address is not part of any `O_DIRECT` invariant.
pub const REGION_ALIGN: u64 = READ_ALIGN;

/// Round `x` up to a tensor's placement boundary ([`READ_ALIGN`]).
fn align_read(x: u64) -> u64 {
    x.div_ceil(READ_ALIGN) * READ_ALIGN
}

/// Place a tensor of `bytes` at `*next`, leaving `*next` on the next [`READ_ALIGN`] boundary past it.
///
/// This is the whole of the layout's placement rule for a view, in one place so that the padding it costs is a property that can be
/// tested on its own ([`tests::a_layout_of_thousands_of_small_tensors_pads_under_a_percent`]).
fn place(next: &mut u64, bytes: u64) -> u64 {
    let off = align_read(*next);
    *next = off + bytes;
    off
}

/// One cache record: an expert rounded up to a page, which is what an `O_DIRECT` read needs.
fn align_page(x: u64) -> u64 {
    x.div_ceil(gb_cuda::expert_cache::PAGE as u64) * gb_cuda::expert_cache::PAGE as u64
}

/// The slots a window budget buys: a whole number of layer quotas (so the cache's per-layer fair share is exact) of
/// `record_bytes` each, and at least one slot, so a bank that streams at all can always make progress. The one rule both the
/// layout (which allocates the window) and admission (which prices it, through [`window_bytes_for`]) use.
pub fn window_slots_for(budget: u64, record_bytes: u64, layers: usize) -> usize {
    let raw = (budget / record_bytes.max(1)) as usize;
    let layers = layers.max(1);
    if raw >= layers {
        (raw / layers) * layers
    } else {
        raw.max(1)
    }
}

/// The widest expert of the pack, page-aligned: one cache record, and the granularity a slab shrinks by when an allocation is refused.
pub fn largest_record_bytes(w: &DeepseekV41Weights) -> u64 {
    let rec = w.layers.iter().flat_map(|l| l.experts.experts.iter()).map(|e| e.record_bytes()).max().unwrap_or(0);
    if rec == 0 {
        0
    } else {
        align_page(rec)
    }
}

/// The device bytes the streaming window takes for a `budget`: [`window_slots_for`] slots of the pack's largest expert record,
/// page-aligned - exactly what [`Layout::new`] allocates for that budget. Admission prices this, not the budget, so the
/// rounding down to whole layer quotas (4 GiB buys 160 slots of 20.6 MiB = 3.30 GiB on the 1.62 bpw pack) is not charged twice.
pub fn window_bytes_for(w: &DeepseekV41Weights, budget: u64) -> u64 {
    let record = largest_record_bytes(w);
    if record == 0 {
        return 0;
    }
    window_slots_for(budget, record, w.layers.len()) as u64 * record
}

/// What a load does about memory and threads.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Refuse to allocate a region that would leave less than this much `MemAvailable`.
    pub floor_bytes: u64,
    /// The pipelined loader's options (workers, staging slots, span size, `O_DIRECT` mode).
    pub loader: Options,
}

impl LoadOptions {
    pub fn from_env(floor_bytes: u64) -> Self {
        let mut loader = Options::from_env();
        if std::env::var("REXL3_LOAD_WORKERS").is_err() {
            loader.workers = 16;
        }
        Self { floor_bytes, loader }
    }
}

/// What a region holds, for the refusal messages and for a partial-depth load.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegionKind {
    /// The embedding, the head and the final norm.
    Global,
    /// One trunk layer's own small tensors (its two norms).
    Trunk(usize),
    /// One trunk layer's resident experts.
    Experts(usize),
    /// One NextN (MTP) layer's routed experts: always resident, never streamed, and therefore **not** an expert slab — a
    /// two-phase boot allocates it with the rest of phase one and never releases it.
    MtpExperts(usize),
    /// The slab the streamed experts are filled into (`ExpertCache`), when the budget does not hold them all.
    Streamed,
}

impl RegionKind {
    /// `true` for the per-layer routed-expert regions — the ones a two-phase boot allocates *last*, after it has measured what is left.
    pub fn is_experts(&self) -> bool {
        matches!(self, RegionKind::Experts(_))
    }
}

#[derive(Clone, Debug)]
pub struct RegionDesc {
    pub id: u32,
    pub kind: RegionKind,
    pub bytes: u64,
}

/// Where one tensor of the plan lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    /// A view of region `region` at `offset`.
    View { region: u32, offset: u64, bytes: u64 },
    /// Uploaded by the kernel's own constructor from the shard's host bytes: no region.
    Upload { bytes: u64 },
    /// `mmap`ped from the shard and addressed in place: no region.
    Mapped { bytes: u64 },
    /// A routed expert past the residency budget: it lives in the `ExpertCache`'s slab, filled on demand, so the layout gives it no
    /// region of its own (the slab is one region, sized in whole records).
    Streamed { bytes: u64 },
}

impl Binding {
    pub fn bytes(&self) -> u64 {
        match self {
            Binding::View { bytes, .. } | Binding::Upload { bytes } | Binding::Mapped { bytes } | Binding::Streamed { bytes } => {
                *bytes
            }
        }
    }
    pub fn is_region(&self) -> bool {
        matches!(self, Binding::View { .. })
    }
}

/// The addresses the plan implies: which regions exist, and where every tensor of the tree goes.
#[derive(Clone, Debug)]
pub struct Layout {
    pub regions: Vec<RegionDesc>,
    /// In tree order, keyed by tensor name.
    pub bindings: BTreeMap<String, Binding>,
    /// Device bytes the loader writes into regions.
    pub view_bytes: u64,
    /// Device bytes the kernels upload themselves (counted on the device, not in a region).
    pub upload_bytes: u64,
    /// Bytes addressed through the `mmap` sidecar.
    pub mapped_bytes: u64,
    /// Resident whole experts (the first `plan.resident_experts` in layer-major order), and their bytes.
    pub resident_experts: usize,
    pub slab_bytes: u64,
    /// Device bytes of the NextN (MTP) expert regions: always resident, never streamed, and not part of [`Layout::slab_bytes`].
    pub mtp_expert_bytes: u64,
    /// The tail that streams: `expert_bytes - slab_bytes` — **disk** bytes, not device bytes. It is what the cache will have to read over a
    /// token, not what it costs to boot.
    pub streamed_bytes: u64,
    /// One cache slot: the widest expert, page-aligned (`O_DIRECT`).
    pub stream_record_bytes: u64,
    /// The streaming window: `window_slots` records, the only device memory the non-resident tail costs.
    pub window_bytes: u64,
    pub window_slots: usize,
}

/// Which kernel type a tensor is bound through, from its name. `None` for a name no rule matches (a bug in the tree, not a pack).
///
/// `trunk` is the trunk's depth: a `layers.L.` with `L >= trunk` is a NextN layer, whose tensors are all raw views (the DSpark drafter
/// is not built, so nothing binds them into a kernel type yet).
fn upload_bound(name: &str, trunk: usize) -> Option<&'static str> {
    if name.starts_with("mtp.") {
        // a NextN layer: raw views (the DSpark drafter is not built)
        return None;
    }
    if name.ends_with(".trellis") || name.ends_with(".suh") || name.ends_with(".svh") || name.ends_with(".mul1") {
        // Resident EXL3 groups (e.g. `head`, or quantized `attn.{wq_b,wo_b}` / `ffn.shared_experts.*`)
        // are placed into `Global` / `Trunk(l)` regions and bound as zero-copy `Exl3Weights::from_parts` views.
        return None;
    }
    if let Some(l) = layer_of(name) {
        if l >= trunk {
            return None;
        }
    }
    if name == "norm.weight" || name.ends_with("attn_norm.weight") || name.ends_with("ffn_norm.weight") {
        // bf16 in the pack, fp32 in every kernel: a widening, so an upload
        return Some("f32 gamma");
    }
    // the MLA's five FP8 projections and its three raw vectors (`MlaDsv4Pack`)
    for p in ["attn.wq_a.", "attn.wq_b.", "attn.wkv.", "attn.wo_a.", "attn.wo_b."] {
        if name.contains(p) {
            return Some("MlaDsv4");
        }
    }
    // the compressor and the indexer are built inside `MlaDsv4` too (`DevCompressor`, `DevIndexer`)
    if name.contains("attn.compressor.") || name.contains("attn.indexer.") {
        return Some("MlaDsv4");
    }
    if name.ends_with("attn.q_norm.weight") || name.ends_with("attn.kv_norm.weight") || name.ends_with("attn.attn_sink") {
        return Some("MlaDsv4");
    }
    if name.contains("ffn.shared_experts.") {
        return Some("Fp8Weights");
    }
    if name.contains("ffn.gate.") {
        return Some("Dsv4Router");
    }
    if name.contains("hc_attn_") || name.contains("hc_ffn_") {
        return Some("HcSite");
    }
    if name.contains("engram.wkv.") {
        return Some("Fp8Weights");
    }
    None
}

impl Layout {
    /// Turn the plan's classification into addresses, keeping at most `slab_bytes` of routed experts resident and reserving
    /// `window_bytes` of device memory for the cache that streams the rest.
    ///
    /// The walk is the plan's own order (which is the tree's), so a tensor the plan classified and this module cannot place is a
    /// [`LayoutError::Unplaceable`] naming it, and a plan that is short a tensor is caught by the totals.
    ///
    /// **The streamed tail is disk, not RAM.** There is no region the size of the tail: the only device memory a non-resident expert ever
    /// occupies is one slot of the window, and the window is `window_bytes` whatever the tail is. A 24.7 GiB tail through a 4 GiB window
    /// is the whole point; sizing the window to the tail is what refused to boot on a GB10, where host RAM and device memory are one pool.
    pub fn new(w: &DeepseekV41Weights, plan: &LoadPlan, slab_bytes: u64, window_bytes: u64) -> Result<Self, LayoutError> {
        Self::new_tp(w, plan, slab_bytes, window_bytes, None)
    }

    /// [`Layout::new`] for one rank of a tensor-parallel group: the plan is the rank's (`LoadPlan::build_tp`), and this
    /// walks only the experts the rank owns — the other ranks' routed experts get no offsets, no regions and no bytes
    /// (absent, never read, never allocated), and the slab cap is the rank's own records rather than the pack's. Everything
    /// the [`tp::ShardTable`] does not cut is laid out exactly as `new` lays it out.
    pub fn new_tp(
        w: &DeepseekV41Weights,
        plan: &LoadPlan,
        slab_bytes: u64,
        window_bytes: u64,
        shard: Option<&super::tp::ShardTable>,
    ) -> Result<Self, LayoutError> {
        let mut regions: Vec<RegionDesc> = Vec::new();
        let mut bindings: BTreeMap<String, Binding> = BTreeMap::new();
        let mut next_id: u32 = 0;
        // one region for the embedding/head/norm, one a layer for its norms, one a layer for its resident experts
        let mut global = 0u64;
        let mut trunk: Vec<u64> = vec![0; w.layers.len()];
        let mut experts: Vec<u64> = vec![0; w.layers.len()];
        // the expert slab is layer-major: walk the experts in that order and stop at the budget; the most it can hold is
        // every expert's aligned record — a rank's OWN records (the other half is never a candidate)
        let slab = slab_bytes.min(super::plan::owned_expert_records(w, shard));
        let mut left = slab;
        let mut resident_experts = 0usize;
        let mut resident_slab = 0u64;
        let mut stream_record = 0u64;
        let mut expert_offsets: BTreeMap<(usize, usize), u64> = BTreeMap::new();
        for l in &w.layers {
            for (e, expert) in l.experts.experts.iter().enumerate() {
                // a rank walks only its own range (`ShardTable::owns_trunk_expert`): the peer's experts are absent
                if !shard.is_none_or(|s| s.owns_trunk_expert(e)) {
                    continue;
                }
                // fitted and placed by its aligned record ([`Expert::record_layout`]), counted by its parts: the same rule as
                // `LoadPlan::build`, so the two agree on which experts are resident
                let rec = expert.record_bytes();
                stream_record = stream_record.max(rec);
                if rec <= left {
                    expert_offsets.insert((l.index, e), experts[l.index]);
                    experts[l.index] += rec;
                    left -= rec;
                    resident_experts += 1;
                    resident_slab += expert.projections().iter().map(|g| g.bytes()).sum::<u64>();
                }
            }
        }
        debug_assert_eq!(resident_experts, plan.resident_experts.min(w.layers.iter().map(|l| l.experts.len()).sum()));

        // the NextN experts: every one of them, in a record of its own, in a region a layer. Nothing here is a budget decision —
        // the drafter's experts are never streamed — so this walk has no `left` to spend. A rank of a group walks its own
        // range only: the peer's NextN experts are absent like its trunk ones.
        let mut mtp_experts: Vec<u64> = vec![0; w.mtp.len()];
        let mut mtp_expert_offsets: BTreeMap<(usize, usize), u64> = BTreeMap::new();
        for m in &w.mtp {
            for (e, expert) in m.experts.iter().enumerate() {
                if !shard.is_none_or(|s| s.owns_nextn_expert(e)) {
                    continue;
                }
                mtp_expert_offsets.insert((m.index, e), mtp_experts[m.index]);
                mtp_experts[m.index] += expert.record_bytes();
            }
        }

        for item in &plan.items {
            let name = item.name.as_str();
            let binding = match item.placement {
                Placement::Sidecar => Binding::Mapped { bytes: item.bytes },
                Placement::ExpertCache => {
                    // which (layer, expert) this tensor belongs to: `layers.L.ffn.experts.E.<proj>.<part>`
                    let (layer, expert) = expert_of(name).ok_or_else(|| LayoutError::Unplaceable { name: name.into() })?;
                    match expert_offsets.get(&(layer, expert)) {
                        Some(&off) => {
                            // the offset of this part inside the expert
                            let within = expert_part_offset(w, layer, expert, name)
                                .ok_or_else(|| LayoutError::Unplaceable { name: name.into() })?;
                            Binding::View { region: 0, offset: off + within, bytes: item.bytes }
                        }
                        None => Binding::Streamed { bytes: item.bytes },
                    }
                }
                Placement::Resident => match upload_bound(name, w.layers.len()) {
                    Some(_) => Binding::Upload { bytes: item.bytes },
                    None => match layer_of(name) {
                        Some(l) => {
                            let off = place(&mut trunk[l], item.bytes);
                            Binding::View { region: 0, offset: off, bytes: item.bytes }
                        }
                        None => {
                            let off = place(&mut global, item.bytes);
                            Binding::View { region: 0, offset: off, bytes: item.bytes }
                        }
                    },
                },
                Placement::MtpExpert => {
                    // `mtp.L.ffn.experts.E.<proj>.<part>`: a view of its record in its layer's NextN expert region
                    let (layer, expert) = mtp_expert_of(name).ok_or_else(|| LayoutError::Unplaceable { name: name.into() })?;
                    let off = *mtp_expert_offsets
                        .get(&(layer, expert))
                        .ok_or_else(|| LayoutError::Unplaceable { name: name.into() })?;
                    let within =
                        mtp_part_offset(w, layer, expert, name).ok_or_else(|| LayoutError::Unplaceable { name: name.into() })?;
                    Binding::View { region: 0, offset: off + within, bytes: item.bytes }
                }
            };
            bindings.insert(name.to_string(), binding);
        }

        // **Every view is aligned for the kernel that reads it.** Page placement gives the trunk/global tensors 4 KiB; an expert part is
        // placed by `Expert::record_layout` on `EXPERT_PART_ALIGN` (256 B) boundaries inside its record, which is already a multiple of
        // it, so this holds for all twelve parts too. A future change that broke either would fault at the first forward with
        // `cudaErrorMisalignedAddress` and no name, so it fails here instead, naming the tensor.
        assert_eq!(READ_ALIGN % MIN_DST_ALIGN, 0, "the placement boundary must be a multiple of the kernels' alignment");
        for (name, b) in &bindings {
            if let Binding::View { offset, .. } = b {
                assert!(offset % MIN_DST_ALIGN == 0, "`{name}` is placed at {offset}, which is not {MIN_DST_ALIGN}-aligned");
            }
        }

        // the regions, in a fixed order: global, then a layer's trunk tensors, then its experts, then the streaming window
        let mut assign = |kind: RegionKind, bytes: u64, next_id: &mut u32| -> u32 {
            let id = *next_id;
            *next_id += 1;
            // the window is a whole number of page-aligned records already; every other region is rounded up to the loader's alignment,
            // which is one page: a tensor's own placement is already page-aligned, so this only rounds the region's tail
            let bytes = if kind == RegionKind::Streamed { bytes } else { align_read(bytes) };
            regions.push(RegionDesc { id, kind, bytes });
            id
        };
        let gid = assign(RegionKind::Global, global, &mut next_id);
        let mut tids = Vec::with_capacity(w.layers.len());
        let mut eids = Vec::with_capacity(w.layers.len());
        for l in 0..w.layers.len() {
            tids.push(assign(RegionKind::Trunk(l), trunk[l], &mut next_id));
            eids.push(assign(RegionKind::Experts(l), experts[l], &mut next_id));
        }
        // then one region a NextN layer for its experts: after the trunk's slabs, before the window, and never released
        let mut mtp_ids = Vec::with_capacity(w.mtp.len());
        for (i, bytes) in mtp_experts.iter().enumerate() {
            mtp_ids.push(assign(RegionKind::MtpExperts(i), *bytes, &mut next_id));
        }
        // the expert records live in one slab region per layer; point the views at the right region now that the ids are known
        for (name, b) in bindings.iter_mut() {
            if let Binding::View { region, .. } = b {
                if *region == 0 {
                    // a trunk expert lives in its layer's slab, a NextN expert in its layer's own region
                    if let Some((l, _)) = expert_of(name) {
                        *region = eids[l];
                    } else if let Some((l, _)) = mtp_expert_of(name) {
                        *region = mtp_ids[l];
                    }
                }
            }
        }
        for (name, b) in bindings.iter_mut() {
            if let Binding::View { region, .. } = b {
                if *region == 0 {
                    *region = match layer_of(name) {
                        Some(l) => tids[l],
                        None => gid,
                    };
                }
            }
        }

        // the slab holds whole experts; the tail that does not fit is disk, and the window it is paged through is a separate, fixed charge
        let streamed_bytes = plan.expert_bytes - resident_slab;
        // the record is one expert, page-aligned: a fill is one O_DIRECT read and a slot is byte-identical to a resident expert
        let stream_record_bytes = align_page(stream_record.max(1));
        let window_slots =
            if streamed_bytes == 0 { 0 } else { window_slots_for(window_bytes, stream_record_bytes, w.layers.len()) };
        let window = window_slots as u64 * stream_record_bytes;
        if window_slots > 0 {
            assign(RegionKind::Streamed, window, &mut next_id);
        }

        let mut out = Self {
            regions,
            bindings,
            view_bytes: 0,
            upload_bytes: 0,
            mapped_bytes: 0,
            resident_experts,
            slab_bytes: resident_slab,
            mtp_expert_bytes: mtp_experts.iter().sum(),
            streamed_bytes,
            stream_record_bytes,
            window_bytes: window,
            window_slots,
        };
        for b in out.bindings.values() {
            match b {
                Binding::View { bytes, .. } => out.view_bytes += bytes,
                Binding::Upload { bytes } => out.upload_bytes += bytes,
                Binding::Mapped { bytes } => out.mapped_bytes += bytes,
                Binding::Streamed { .. } => {}
            }
        }
        Ok(out)
    }

    /// Bytes this layout puts on the device: the regions (which include the alignment), not the `mmap` sidecar.
    pub fn device_bytes(&self) -> u64 {
        self.regions.iter().map(|r| r.bytes).sum()
    }

    /// The id of the streaming window's region, when the plan streams a tail. **This is the one allocation the window has**:
    /// the expert cache builds its [`gb_cuda::expert_cache::DeviceSlab`] over this region
    /// ([`Regions::tensor`]-style view) rather than `cudaMalloc`ing a second window of its own.
    pub fn window_region(&self) -> Option<u32> {
        self.regions.iter().find(|r| r.kind == RegionKind::Streamed).map(|r| r.id)
    }

    /// Bytes the regions spend on nothing: `device_bytes - view_bytes - window_bytes`. With [`READ_ALIGN`] placement this is under
    /// 256 B a view tensor (plus a region tail), where the old 2 MiB rule was 2.74 GiB of region padding on this pack.
    pub fn padding_bytes(&self) -> u64 {
        self.device_bytes().saturating_sub(self.view_bytes).saturating_sub(self.window_bytes)
    }

    /// Every view tensor as `(name, region, offset, bytes)`, in name order. The alignment and disjointness properties are stated
    /// over this and tested on the real pack.
    pub fn view_spans(&self) -> impl Iterator<Item = (&str, u32, u64, u64)> {
        self.bindings.iter().filter_map(|(n, b)| match b {
            Binding::View { region, offset, bytes } => Some((n.as_str(), *region, *offset, *bytes)),
            _ => None,
        })
    }
}

/// Why the plan and the device could not be made to agree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// A plan item no rule places: a bug in the tree, named.
    Unplaceable { name: String },
    /// The plan's resident total is not `view + upload`.
    ResidentDisagrees { planned: u64, view: u64, upload: u64 },
    /// The plan's sidecar total is not what was mapped.
    SidecarDisagrees { planned: u64, mapped: u64 },
    /// The plan and the layout see different numbers of tensors.
    CountDisagrees { planned: usize, laid_out: usize },
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayoutError::Unplaceable { name } => {
                write!(f, "deepseek_v41: the plan names `{name}`, which no placement rule of the device covers")
            }
            LayoutError::ResidentDisagrees { planned, view, upload } => write!(
                f,
                "deepseek_v41: the plan's resident set is {planned} B but the device binds {view} B of views and uploads {upload} B \
                 ({} B unaccounted for)",
                (*planned as i64 - (*view + *upload) as i64).unsigned_abs()
            ),
            LayoutError::SidecarDisagrees { planned, mapped } => write!(
                f,
                "deepseek_v41: the plan's sidecar is {planned} B but {mapped} B are mapped"
            ),
            LayoutError::CountDisagrees { planned, laid_out } => {
                write!(f, "deepseek_v41: the plan has {planned} tensors and the layout {laid_out}")
            }
        }
    }
}

impl std::error::Error for LayoutError {}

/// `layers.L.` of a name, when it has one.
fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("layers.")?;
    rest.split('.').next()?.parse().ok()
}

/// The `(layer, expert)` of `layers.L.ffn.experts.E.<...>`.
fn expert_of(name: &str) -> Option<(usize, usize)> {
    let mut it = name.split('.');
    if it.next()? != "layers" {
        return None;
    }
    let layer: usize = it.next()?.parse().ok()?;
    if it.next()? != "ffn" || it.next()? != "experts" {
        return None;
    }
    Some((layer, it.next()?.parse().ok()?))
}

/// The offset of one part of one expert inside the expert's record: `w1`, then `w2`, then `w3`, each `trellis`, `mul1`, `suh`, `svh`.
///
/// The three projections are laid end to end, so the offset of `w2.trellis` is the whole of `w1`. Getting that wrong does not fail
/// anything that reads the pack — `Exl3Weights::from_parts` checks each part's length, not its address — it decodes noise, which is why the
/// twelve parts are proved to tile the record in one test and the cache's own table
/// ([`gb_cuda::expert_cache::ExpertTable`]) is built by the same walk.
fn expert_part_offset(w: &DeepseekV41Weights, layer: usize, expert: usize, name: &str) -> Option<u64> {
    let e = w.layers.get(layer)?.experts.experts.get(expert)?;
    let (proj, part) = name.rsplit_once('.')?;
    let which = proj.rsplit('.').next()?;
    let p = ["w1", "w2", "w3"].iter().position(|x| *x == which)?;
    let q = ["trellis", "mul1", "suh", "svh"].iter().position(|x| *x == part)?;
    Some(e.record_layout().0[p][q])
}

/// The `(layer, expert)` of `mtp.L.ffn.experts.E.<...>`.
fn mtp_expert_of(name: &str) -> Option<(usize, usize)> {
    let mut it = name.split('.');
    if it.next()? != "mtp" {
        return None;
    }
    let layer: usize = it.next()?.parse().ok()?;
    if it.next()? != "ffn" || it.next()? != "experts" {
        return None;
    }
    Some((layer, it.next()?.parse().ok()?))
}

/// The offset of one part of one NextN expert inside the expert's record: the pack's own walk ([`MtpExpert::record_layout`]),
/// indexed by projection (`w1`, `w2`, `w3`) and part.
fn mtp_part_offset(w: &DeepseekV41Weights, layer: usize, expert: usize, name: &str) -> Option<u64> {
    let e = w.mtp.get(layer)?.experts.get(expert)?;
    let (proj, part) = name.rsplit_once('.')?;
    let which = proj.rsplit('.').next()?;
    let p = ["w1", "w2", "w3"].iter().position(|x| *x == which)?;
    let q = match e {
        super::weights::MtpExpert::Exl3(_) => ["trellis", "mul1", "suh", "svh"].iter().position(|x| *x == part)?,
        super::weights::MtpExpert::Int4(_) => ["weight", "scale"].iter().position(|x| *x == part)?,
    };
    Some(e.record_layout().0[p][q])
}

/// The device regions of a layout, indexed by [`RegionDesc::id`].
pub struct Regions {
    views: Vec<Option<DevBuf>>,
}

impl Regions {
    /// No regions: what a model over weights that are already resident uses.
    pub fn empty() -> Self {
        Self { views: Vec::new() }
    }

    /// One slot a region of `layout`, all of them unallocated: the shape [`allocate_experts`] fills in phase two.
    fn empty_for(layout: &Layout) -> Self {
        let max_id = layout.regions.iter().map(|r| r.id as usize).max().map_or(0, |m| m + 1);
        Self { views: (0..max_id).map(|_| None).collect() }
    }

    /// Release the expert regions of `layout` (and only those), for a phase-two attempt that has to be retried at a smaller slab.
    /// The trunk, global and window regions — everything phase one allocated — are left alone.
    pub fn release_experts(&mut self, layout: &Layout) {
        for r in layout.regions.iter().filter(|r| r.kind.is_experts() || r.kind == RegionKind::Streamed) {
            if let Some(v) = self.views.get_mut(r.id as usize) {
                *v = None;
            }
        }
    }

    pub fn get(&self, id: u32) -> Result<&DevBuf, ModelError> {
        self.views
            .get(id as usize)
            .and_then(|v| v.as_ref())
            .ok_or_else(|| ModelError::Config(format!("the load plan names region {id}, which was not allocated")))
    }

    /// A view of `bytes` bytes at `offset` into region `id`.
    pub fn view(&self, id: u32, offset: u64, bytes: u64) -> Result<DevBuf, ModelError> {
        let r = self.get(id)?;
        if offset + bytes > r.bytes() as u64 {
            return Err(ModelError::Config(format!("a view of {bytes} B at {offset} runs past region {id} ({} B)", r.bytes())));
        }
        Ok(r.subview(offset as usize, bytes as usize))
    }

    /// A view of one plan item, through its binding. `Err` for a tensor bound outside the regions.
    pub fn tensor(&self, layout: &Layout, name: &str) -> Result<DevBuf, ModelError> {
        match layout.bindings.get(name) {
            Some(Binding::View { region, offset, bytes }) => self.view(*region, *offset, *bytes),
            Some(Binding::Upload { .. }) => {
                Err(ModelError::Config(format!("`{name}` is uploaded by its kernel, not a region view")))
            }
            Some(Binding::Mapped { .. }) => Err(ModelError::Config(format!("`{name}` is an mmap sidecar, not a region view"))),
            Some(Binding::Streamed { .. }) => {
                Err(ModelError::Config(format!("`{name}` streams through the expert cache, not a region view")))
            }
            None => Err(ModelError::Config(format!("`{name}` is not in the load plan"))),
        }
    }

    pub fn bytes(&self) -> u64 {
        self.views.iter().flatten().map(|v| v.bytes() as u64).sum()
    }

    pub fn count(&self) -> usize {
        self.views.iter().flatten().count()
    }
}

fn mem_available() -> Option<u64> {
    crate::serve::mem_available()
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// Why a region could not be allocated: the `MemAvailable` floor check of [`allocate_into`], as a value rather than a message, so a
/// two-phase boot can shrink the slab by exactly the shortfall instead of guessing.
#[derive(Clone, Debug)]
pub struct RegionRefusal {
    pub id: u32,
    pub kind: RegionKind,
    pub bytes: u64,
    pub avail: u64,
    pub floor: u64,
    pub allocated: usize,
    pub regions: usize,
    /// Bytes the box was short by: what the region needed over the floor, less what was there.
    pub shortfall: u64,
    /// Bytes the regions this call allocated sum to, for the refusal's own arithmetic.
    pub allocated_bytes: u64,
    /// The whole layout's device bytes, for the message.
    pub total: u64,
}

impl std::fmt::Display for RegionRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "refusing to allocate region {} ({:?}, {:.2} GiB): MemAvailable {:.2} GiB would fall below the {:.2} GiB floor \
             (the layout needs {:.2} GiB in all); {} of {} regions are allocated",
            self.id,
            self.kind,
            gib(self.bytes),
            gib(self.avail),
            gib(self.floor),
            gib(self.total),
            self.allocated,
            self.regions
        )
    }
}

impl From<RegionRefusal> for ModelError {
    fn from(r: RegionRefusal) -> Self {
        ModelError::Config(r.to_string())
    }
}

/// **Phase one of a two-phase boot**: every region that is *not* a routed-expert slab — the global embedding/head/norm, each layer's
/// trunk tensors, and the streaming window. The expert regions are left unallocated (`None`), so that the bytes they would take stay
/// on the box while the rest of the boot (uploads, RoPE, KV planes, scratch, the CUDA context, the engram mappings) is built and
/// measured; [`allocate_experts`] then spends what is actually left.
///
/// A zero-byte expert region is left unallocated too (a `cudaMalloc(0)` is an error on some drivers, and there is nothing to hold).
fn cma_free_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|t| {
            let l = t.lines().find(|l| l.starts_with("CmaFree:"))?;
            l.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb * 1024)
        })
        .unwrap_or(0)
}

pub fn allocate_non_experts(layout: &Layout, floor_bytes: u64) -> Result<Regions, ModelError> {
    let mut regions = Regions::empty_for(layout);
    allocate_into(&mut regions, layout, floor_bytes, |k| !k.is_experts() && *k != RegionKind::Streamed)
        .map_err(ModelError::from)?;
    Ok(regions)
}

/// **Phase two**: the per-layer expert regions of `layout`, each with the same `MemAvailable` floor check [`allocate`] makes.
///
/// A refusal leaves `regions` exactly as it found it — the regions this call did allocate are released first — so the caller can retry
/// with a smaller slab (which is a different layout: which experts are resident is the layout's decision) without leaking the partial
/// attempt. The refusal carries its own shortfall, so the retry shrinks by what the box was actually short rather than by a guess.
pub fn allocate_experts(layout: &Layout, regions: &mut Regions, floor_bytes: u64) -> Result<(), RegionRefusal> {
    // The streaming window first, the expert slab second: the window is a fixed cost every attempt needs, so
    // allocating it after the experts lets a large slab starve it, refuse, shrink, and repeat — allocating it
    // first leaves the experts to fill exactly what remains, and the shrink loop converges in one step.
    if let Err(e) = allocate_into(regions, layout, floor_bytes, |k| *k == RegionKind::Streamed) {
        regions.release_experts(layout);
        return Err(e);
    }
    if let Err(e) = allocate_into(regions, layout, floor_bytes, |k| k.is_experts()) {
        regions.release_experts(layout);
        return Err(e);
    }
    Ok(())
}

/// The one allocation loop: every region `want` selects, checked against `MemAvailable` before each `cudaMalloc`.
fn allocate_into(
    regions: &mut Regions,
    layout: &Layout,
    floor_bytes: u64,
    want: impl Fn(&RegionKind) -> bool,
) -> Result<(), RegionRefusal> {
    let total = layout.device_bytes();
    // bytes *this call* allocated: the refusal's arithmetic needs the regions it is responsible for, not every region on the device
    let mut allocated_here = 0u64;
    // Drop any cached non-engram shard pages ONCE before the allocation loop so every floor check sees reclaimable
    // shard cache already returned to `MemAvailable`, without repeating 15 `File::open` + `posix_fadvise` syscalls
    // per region inside the loop.
    if let Some(dir) = super::engine::MODEL_DIR.get() {
        super::engine::fadvise_pack_shards(dir);
    }
    for r in &layout.regions {
        if r.id as usize >= regions.views.len() {
            regions.views.resize_with(r.id as usize + 1, || None);
        }
        if !want(&r.kind) || r.bytes == 0 || regions.views.get(r.id as usize).is_some_and(|v| v.is_some()) {
            continue;
        }
        if let Some(avail) = mem_available() {
            if avail < r.bytes.saturating_add(floor_bytes) {
                // When 95%+ of the expert slab is already allocated (`allocated_here >= 95 GiB`) and `MemAvailable`
                // (which subtracts ~1.4 GiB of Linux zone watermarks from `MemFree`) is within 1.2 GiB of `r.bytes + floor_bytes`
                // while actual free physical RAM leaves headroom, allocate the final expert region via `alloc_pageable_host`
                // (`2 MiB MADV_HUGEPAGE + MADV_POPULATE_WRITE`) rather than shrinking the slab and paying the 3.97 GiB
                // `ExpertCache` streaming window + O_DIRECT miss penalty.
                if r.kind.is_experts()
                    && allocated_here >= (95u64 << 30)
                    && avail.saturating_add(1280 << 20) >= r.bytes.saturating_add(floor_bytes)
                {
                    let block = Arc::new(Block::alloc_pageable_host(r.bytes as usize).map_err(|_| RegionRefusal {
                        id: r.id,
                        kind: r.kind,
                        bytes: r.bytes,
                        avail,
                        floor: floor_bytes,
                        allocated: regions.count(),
                        regions: layout.regions.len(),
                        shortfall: r.bytes.saturating_add(floor_bytes).saturating_sub(avail),
                        allocated_bytes: allocated_here,
                        total,
                    })?);
                    regions.views[r.id as usize] = Some(DevBuf::view(&block, 0, r.bytes as usize));
                    allocated_here += r.bytes;
                    continue;
                }
                return Err(RegionRefusal {
                    id: r.id,
                    kind: r.kind,
                    bytes: r.bytes,
                    avail,
                    floor: floor_bytes,
                    allocated: regions.count(),
                    regions: layout.regions.len(),
                    shortfall: r.bytes.saturating_add(floor_bytes).saturating_sub(avail),
                    allocated_bytes: allocated_here,
                    total,
                });
            }
        }
        // On GB10 unified memory, `CmaFree` (~8-9 GiB) is counted in `MemAvailable` / `MemFree`, but kernel-unmovable
        // `cudaMalloc` (`_memdescAllocInternal`) cannot allocate from `MIGRATE_CMA`. Keep >= 1.5 GiB of non-CMA memory
        // free for CUDA runtime/cuBLAS and allocate the remaining expert regions via `alloc_pageable_host` (`MIGRATE_MOVABLE`),
        // which populates directly from `CmaFree` and is read by the GPU via hardware ATS.
        let cma = cma_free_bytes();
        let cur_avail = mem_available().unwrap_or(u64::MAX);
        let non_cma_avail = cur_avail.saturating_sub(cma);
        let use_pageable =
            r.kind.is_experts() && cma > 0 && non_cma_avail < r.bytes.saturating_add(floor_bytes).saturating_add(3 << 29);
        let alloc_res = if use_pageable {
            Block::alloc_pageable_host(r.bytes as usize)
        } else {
            Block::alloc(r.bytes as usize).or_else(|e| {
                if r.kind.is_experts() && cma > 0 {
                    Block::alloc_pageable_host(r.bytes as usize)
                } else {
                    Err(e)
                }
            })
        };
        let block = Arc::new(alloc_res.map_err(|_| RegionRefusal {
            id: r.id,
            kind: r.kind,
            bytes: r.bytes,
            avail: mem_available().unwrap_or(0),
            floor: floor_bytes,
            allocated: regions.count(),
            regions: layout.regions.len(),
            shortfall: 0,
            allocated_bytes: allocated_here,
            total,
        })?);
        regions.views[r.id as usize] = Some(DevBuf::view(&block, 0, r.bytes as usize));
        allocated_here += r.bytes;
        // `REXL3_DSV41_TRACE_ALLOC=1`: one line a region, so an operator can see what a boot spends and in which order
        if std::env::var("REXL3_DSV41_TRACE_ALLOC").is_ok() {
            eprintln!(
                "rexl3 event=dsv41_alloc_trace id={} kind={:?} gib={:.4} avail_after_gib={:.4}",
                r.id,
                r.kind,
                gib(r.bytes),
                mem_available().map(gib).unwrap_or(0.0)
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------------- the load

/// A pack on the device: the layout that was executed, the regions, and how long each stage took.
pub struct Booted {
    pub layout: Layout,
    pub regions: Regions,
    pub report: Report,
    pub alloc_s: f64,
    pub load_s: f64,
}

/// Execute `layout` into `regions` through the pipelined loader: every view-bound tensor at its offset, every byte through
/// `O_DIRECT` where the file allows it. Upload-bound tensors are the kernel's business ([`super::device`]), and the engram tables are
/// mapped by [`super::engram`]; both are skipped here and accounted for by [`check_agreement`].
///
/// Restricted to the tensors that land in a region `want` selects: phase one of a two-phase boot loads everything but the routed
/// experts, phase two loads the expert slab it sized from the measurement.
pub fn run_where(
    dir: &Path,
    layout: &Layout,
    ck: &Checkpoint,
    regions: &Regions,
    o: &Options,
    want: impl Fn(&RegionKind) -> bool,
) -> Result<Report, ModelError> {
    let mut loader = Loader::new();
    let in_regions: Vec<(u32, bool)> = layout.regions.iter().map(|r| (r.id, want(&r.kind))).collect();
    let wanted = |id: u32| in_regions.iter().any(|(i, w)| *i == id && *w);
    let files: BTreeMap<String, usize> = ck
        .plain
        .values()
        .map(|e| e.file.clone())
        .chain(ck.groups.values().flat_map(|g| g.files.iter().cloned()))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|f| (f.clone(), loader.file(&dir.join(f))))
        .collect();
    let missing =
        |name: &str| ModelError::Config(format!("deepseek_v41: `{name}` is in the load plan but not in the pack's headers"));

    /// Where a tensor's bytes start in its shard **file**: the 8-byte length prefix, the header, then the safetensors
    /// `data_offsets` (`Entry::offsets` is relative to the data section, not to the file). Reading at the bare
    /// `data_offsets` start is a shift of `8 + header_len` — inside the JSON header for the first tensors of a shard —
    /// which decodes to plausible-looking noise rather than failing, so this is the only place the shift is allowed to live.
    fn ck_file_offset(ck: &Checkpoint, e: &Entry, name: &str) -> Result<u64, ModelError> {
        ck.file_offset(e).map_err(|x| ModelError::Config(format!("deepseek_v41: `{name}`: {x}")))
    }
    for (name, b) in &layout.bindings {
        let Binding::View { region, offset, bytes } = b else { continue };
        if !wanted(*region) {
            continue;
        }
        let dst = regions.view(*region, *offset, *bytes)?;
        // a plain tensor, or one part of an EXL3 group
        let (file, off, len) = match ck.plain.get(name) {
            Some(e) => (e.file.clone(), ck_file_offset(ck, e, name)?, e.offsets.1 - e.offsets.0),
            None => {
                let prefix = name.rsplit_once('.').map(|(p, _)| p).ok_or_else(|| missing(name))?;
                let parts = ck.group_parts(prefix).map_err(|e| ModelError::Config(format!("{prefix}: {e}")))?;
                let e = match name.rsplit_once('.').map(|(_, p)| p) {
                    Some("trellis") => parts.trellis,
                    Some(p) if p == parts.group.in_scales => parts.in_scales,
                    Some(p) if p == parts.group.out_scales => parts.out_scales,
                    Some("bias") => parts.bias.ok_or_else(|| missing(name))?,
                    // the codebook marker beside the group (`mul1`/`mcg`): a plain scalar the scan exposes by full name.
                    // The scan already derives the group's codebook from this marker and refuses a group whose tiles need
                    // one it does not carry, so an entry that exists here is the marker by construction.
                    Some("mul1" | "mcg") => ck.part(name).ok_or_else(|| missing(name))?,
                    _ => return Err(missing(name)),
                };
                (e.file.clone(), ck_file_offset(ck, e, name)?, e.offsets.1 - e.offsets.0)
            }
        };
        let fid = *files.get(&file).ok_or_else(|| missing(name))?;
        if len != *bytes {
            return Err(ModelError::Config(format!("deepseek_v41: `{name}` is {len} B in the pack and {bytes} B in the plan")));
        }
        loader.copy_file(&dst, fid, off, len as usize, CopyXform::None)?;
    }
    Ok(loader.run(o)?)
}

/// The plan and the device agree, item by item and in total: a plan that prices bytes the device does not bind, or a device that
/// binds bytes the plan did not price, fails here rather than at the first forward.
pub fn check_agreement(w: &DeepseekV41Weights, plan: &LoadPlan, layout: &Layout) -> Result<(), ModelError> {
    if layout.bindings.len() != plan.items.len() {
        return Err(LayoutError::CountDisagrees { planned: plan.items.len(), laid_out: layout.bindings.len() }.into());
    }
    for item in &plan.items {
        match layout.bindings.get(&item.name) {
            Some(b) if b.bytes() == item.bytes => {}
            Some(b) => {
                return Err(ModelError::Config(format!(
                    "deepseek_v41: `{}` is {} B in the plan and {} B in the layout",
                    item.name,
                    item.bytes,
                    b.bytes()
                )))
            }
            None => return Err(LayoutError::Unplaceable { name: item.name.clone() }.into()),
        }
    }
    // the expert views are the slab, not the resident set: `view_bytes` is both
    let slab: u64 = layout
        .bindings
        .iter()
        .filter(|(n, _)| expert_of(n).is_some())
        .filter_map(|(_, b)| match b {
            Binding::View { bytes, .. } => Some(*bytes),
            _ => None,
        })
        .sum();
    if plan.resident_bytes + plan.mtp_expert_bytes != layout.view_bytes - slab + layout.upload_bytes {
        return Err(LayoutError::ResidentDisagrees {
            planned: plan.resident_bytes + plan.mtp_expert_bytes,
            view: layout.view_bytes - slab,
            upload: layout.upload_bytes,
        }
        .into());
    }
    if plan.sidecar_bytes != layout.mapped_bytes {
        return Err(LayoutError::SidecarDisagrees { planned: plan.sidecar_bytes, mapped: layout.mapped_bytes }.into());
    }
    // and the expert slab: whole experts, and the tail is exactly what streams
    if slab != layout.slab_bytes {
        return Err(ModelError::Config(format!(
            "deepseek_v41: the expert slab holds {slab} B and the layout says {} B",
            layout.slab_bytes
        )));
    }
    let streamed: u64 = layout
        .bindings
        .values()
        .filter_map(|b| match b {
            Binding::Streamed { bytes } => Some(*bytes),
            _ => None,
        })
        .sum();
    if streamed != layout.streamed_bytes {
        return Err(ModelError::Config(format!(
            "deepseek_v41: {streamed} B of experts stream and the layout says {} B",
            layout.streamed_bytes
        )));
    }
    if layout.slab_bytes + layout.streamed_bytes != plan.expert_bytes {
        return Err(ModelError::Config(format!(
            "deepseek_v41: {} B of slab + {} B streamed is not the pack's {} B of experts",
            layout.slab_bytes, layout.streamed_bytes, plan.expert_bytes
        )));
    }
    // the engram tables are the sidecar, and nothing else is
    let tables: u64 = w.engram_tables().map(|t| t.bytes()).sum();
    if tables != layout.mapped_bytes {
        return Err(ModelError::Config(format!(
            "deepseek_v41: the two engram tables are {tables} B and {} B are mapped",
            layout.mapped_bytes
        )));
    }
    let _ = cfg_used_for_dtypes(w);
    Ok(())
}

/// Every dtype the layout treats as a raw view must be one a `DevBuf` can hold as stored. This is the assertion that keeps the
/// classification honest: an `F8_E8M0` scale bound as a view would be read as bytes by a kernel expecting exponents, which is fine, but
/// an `I16` trellis bound as a view must be the group's `from_parts`, not a `DevBuf` copy — so the check is that no view-bound tensor is
/// an EXL3 part outside `.ffn.experts.`.
fn cfg_used_for_dtypes(w: &DeepseekV41Weights) -> usize {
    let mut n = 0;
    w.visit(&mut |t| {
        if t.dtype == Dtype::I16 && !t.name.contains(".ffn.experts.") {
            n += 1;
        }
    });
    n
}

impl From<LayoutError> for ModelError {
    fn from(e: LayoutError) -> Self {
        ModelError::Config(e.to_string())
    }
}

/// The config is a parameter of [`Layout::new`]'s callers, not of the layout: this keeps the import honest for the depth checks the
/// engine runs before it loads.
pub fn check_depth(cfg: &DeepseekV41Config, w: &DeepseekV41Weights) -> Result<(), ModelError> {
    if w.layers.len() != cfg.layers {
        return Err(ModelError::Config(format!(
            "deepseek_v41: the config describes {} trunk layers and the pack bound {}",
            cfg.layers,
            w.layers.len()
        )));
    }
    if w.mtp.len() != cfg.nextn {
        return Err(ModelError::Config(format!(
            "deepseek_v41: the config describes {} NextN layers and the pack bound {}",
            cfg.nextn,
            w.mtp.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::deepseek_v41::bind::bind;
    use crate::families::deepseek_v41::config::DeepseekV41Config;
    use crate::families::deepseek_v41::inventory::PackInventory;

    const CONFIG: &str = include_str!("../../../../../testdata/deepseek_v41/config.json");
    const PACK: &str = "/tmp/dsv41_inventory.json";

    fn real() -> DeepseekV41Config {
        DeepseekV41Config::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    fn weights() -> Option<DeepseekV41Weights> {
        let text = std::fs::read_to_string(PACK).ok()?;
        let listing: serde_json::Value = serde_json::from_str(&text).unwrap();
        let cfg = real();
        let inv = PackInventory::from_json(&cfg, &listing).unwrap();
        Some(bind(&inv, &cfg).unwrap())
    }

    #[test]
    fn the_layout_partitions_the_plan_to_the_byte() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let plan = LoadPlan::build(&w, 0).unwrap();
        let l = Layout::new(&w, &plan, 0, 0).unwrap();
        check_agreement(&w, &plan, &l).unwrap();
        // no slab: every expert streams, nothing is a view of an expert region
        assert_eq!((l.resident_experts, l.slab_bytes), (0, 0));
        assert_eq!(l.streamed_bytes, plan.expert_bytes);
        assert_eq!(l.mapped_bytes, 202_758_032_400, "the two engram tables");
        // views and uploads together are the plan's resident set
        assert_eq!(l.view_bytes + l.upload_bytes, plan.resident_bytes + plan.mtp_expert_bytes);
        // the FP8 matrices are uploaded by their kernels; the embedding, the head, the vision tower and the NextN layers are views
        assert!(l.upload_bytes > 7_000_000_000, "{} B uploaded", l.upload_bytes);
        assert!(l.view_bytes > 11_000_000_000, "{} B of views", l.view_bytes);
        // the global region, two a layer, and the one-slot window the streamed experts fill
        assert_eq!(
            l.regions.len(),
            1 + 2 * 40 + 3 + 1,
            "global, a layer of trunk + slab, three NextN expert regions, the window"
        );
        assert_eq!(l.window_slots, 1, "no window budget: one slot, so a streamed expert can still land");
    }

    /// **The window is one allocation.** The layout's regions plus the kernels' own uploads are exactly the plan's resident set,
    /// the expert slab and the streaming window — and the window's region is the one the cache's slab is built over
    /// ([`Layout::window_region`]), so no byte of it is allocated twice. Before this, `open_cache` made its own 3.303 GiB
    /// `DeviceSlab` on top of the layout's `Streamed` region: 3.3 GiB of a 118 GiB pool, spent twice.
    #[test]
    fn the_streaming_window_is_allocated_once() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let budget = 83_500_000_000u64;
        let window = 4u64 << 30;
        let plan = LoadPlan::build(&w, budget - window).unwrap();
        let l = Layout::new(&w, &plan, budget - window, window).unwrap();
        check_agreement(&w, &plan, &l).unwrap();

        // regions + uploads == resident + slab + window + the layout's own placement slack, to the byte
        assert_eq!(
            l.device_bytes() + l.upload_bytes,
            plan.resident_bytes + plan.mtp_expert_bytes + l.slab_bytes + l.window_bytes + l.padding_bytes(),
            "the device total is the plan's terms and nothing else"
        );
        // and the slack is placement slack, not a second allocation: under 0.1% of the device (the duplicate window alone was 3.4%)
        assert!(l.padding_bytes() < l.device_bytes() / 1_000, "{} B of padding on {} B", l.padding_bytes(), l.device_bytes());
        // exactly one streamed region, and it is the window the cache is built over
        let streamed: Vec<&RegionDesc> = l.regions.iter().filter(|r| r.kind == RegionKind::Streamed).collect();
        assert_eq!(streamed.len(), 1, "one window, one allocation");
        assert_eq!(streamed[0].bytes, l.window_bytes);
        assert_eq!(l.window_region(), Some(streamed[0].id));
        assert_eq!(l.window_bytes, l.window_slots as u64 * l.stream_record_bytes);
        // nothing is unaccounted for: the only slack is each region's tail rounding
        assert!(l.padding_bytes() < l.view_bytes / 100, "{} B of padding", l.padding_bytes());
    }

    /// A plan that holds every expert has no tail, and therefore no window: the `Streamed` region does not exist and there is
    /// nothing for the cache to be built over.
    #[test]
    fn a_plan_with_no_tail_has_no_window_region() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let plan = LoadPlan::full(&w).unwrap();
        let l = Layout::new(&w, &plan, crate::families::deepseek_v41::alloc::expert_records_bytes(&w), 0).unwrap();
        check_agreement(&w, &plan, &l).unwrap();
        assert_eq!(l.window_bytes, 0);
        assert_eq!(l.window_slots, 0);
        assert_eq!(l.window_region(), None, "no tail, no window, no cache");
        assert_eq!(
            l.device_bytes() + l.upload_bytes,
            plan.resident_bytes + plan.mtp_expert_bytes + l.slab_bytes + l.padding_bytes()
        );
    }

    #[test]
    fn a_full_slab_keeps_every_expert_and_streams_nothing() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let plan = LoadPlan::full(&w).unwrap();
        let l = Layout::new(&w, &plan, crate::families::deepseek_v41::alloc::expert_records_bytes(&w), 0).unwrap();
        check_agreement(&w, &plan, &l).unwrap();
        assert_eq!(l.resident_experts, 40 * 384);
        assert_eq!(l.slab_bytes, plan.expert_bytes);
        assert_eq!(l.streamed_bytes, 0);
        // with every expert resident there is no streamed region
        assert_eq!(l.regions.len(), 1 + 2 * 40 + 3);
        assert_eq!((l.window_slots, l.window_bytes), (0, 0));
    }

    /// **The bug this change is about.** The streamed tail is disk. The layout must never size a region to it: on a GB10, where host RAM
    /// and device memory are one 110 GiB pool, a 24.7 GiB staging region is what refused to boot. Here the slab is 4 GiB smaller than the
    /// old plan (the window comes out of it) and the tail is 24.7 GiB, so the arithmetic that used to be 131 GiB is 110 GiB.
    #[test]
    fn the_streaming_window_is_the_budget_and_never_the_tail() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        // the GB10 arithmetic: trunk 17.46 GiB, ~83.5 GiB of experts resident, a 4 GiB window, 10 GiB of KV, 0.25 GiB of scratch
        let budget = 83_500_000_000u64;
        let window = 4u64 << 30;
        let plan = LoadPlan::build(&w, budget - window).unwrap();
        let l = Layout::new(&w, &plan, budget - window, window).unwrap();
        check_agreement(&w, &plan, &l).unwrap();
        let streamed_region = l.regions.iter().find(|r| r.kind == RegionKind::Streamed).unwrap();
        // (b) no region is sized to the tail: the streamed region is the window, and the device total is the resident set plus it
        for r in &l.regions {
            if r.kind == RegionKind::Streamed {
                assert!(r.bytes <= window, "the streamed region is {} B against a {window} B window", r.bytes);
            }
        }
        assert_eq!(streamed_region.bytes, l.window_bytes);
        assert_eq!(l.window_bytes, l.window_slots as u64 * l.stream_record_bytes);
        assert_eq!(l.window_slots % 40, 0, "a whole number of layer quotas");
        // the tail is 27.5 GB of disk, and the device total is the trunk + the resident experts + the window — never the tail
        assert!(l.streamed_bytes > 27_000_000_000, "{} B streams", l.streamed_bytes);
        assert!(streamed_region.bytes < l.streamed_bytes / 6, "the window is not the tail");
        let non_expert: u64 =
            l.regions.iter().filter(|r| !matches!(r.kind, RegionKind::Streamed | RegionKind::Experts(_))).map(|r| r.bytes).sum();
        let expert_regions: u64 = l.regions.iter().filter(|r| matches!(r.kind, RegionKind::Experts(_))).map(|r| r.bytes).sum();
        assert_eq!(l.device_bytes(), non_expert + expert_regions + l.window_bytes);
        assert!(l.device_bytes() < non_expert + expert_regions + 5_000_000_000, "the window is 4 GiB, not 27.5");
        assert!(l.window_bytes <= l.window_slots as u64 * l.stream_record_bytes);
        // and the record is one page-aligned expert
        assert!(l.stream_record_bytes >= 4_468_236, "a cache record holds the widest expert");
        assert_eq!(l.stream_record_bytes % gb_cuda::expert_cache::PAGE as u64, 0);
    }

    #[test]
    fn the_measured_split_is_the_one_the_task_specifies() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        // 108.49 GB of device budget for experts, 4 GiB of it the window: 104.5 GB of whole experts resident, 6.5 GB to stream
        let full = LoadPlan::build(&w, 0).unwrap();
        let budget = 108_486_386_360u64.min(full.expert_bytes);
        let window = 4u64 << 30;
        let plan = LoadPlan::build(&w, budget - window).unwrap();
        let l = Layout::new(&w, &plan, budget - window, window).unwrap();
        check_agreement(&w, &plan, &l).unwrap();
        let streamed = plan.expert_bytes - l.slab_bytes;
        assert!((6.7e9..6.9e9).contains(&(streamed as f64)), "{streamed} B streamed");
        // the 4 GiB that used to be spent on experts is now the window: whole records of the *widest* expert, which in a mixed-K pack is
        // 3x the average, so 4 GiB buys 4 slots a layer rather than 13
        assert!(l.window_bytes <= window);
        assert!(
            l.window_bytes + 40 * l.stream_record_bytes > window,
            "the window is the budget, rounded down to whole layer quotas"
        );
        assert_eq!(l.window_slots, (window / l.stream_record_bytes) as usize / 40 * 40);
        assert!(l.resident_experts > 40 * 350, "{} of 15,360 experts resident", l.resident_experts);
        assert!(l.stream_record_bytes >= 4_468_236, "a cache record holds the widest expert");
        assert_eq!(l.stream_record_bytes % gb_cuda::expert_cache::PAGE as u64, 0);
        assert_eq!(l.slab_bytes + l.streamed_bytes, plan.expert_bytes);
        // the budget is spent on whole experts and the window; what is left over is the experts that did not fit, one record a layer at most
        assert!(l.slab_bytes + l.window_bytes <= budget);
        assert!(budget - (l.slab_bytes + l.window_bytes) < 40 * l.stream_record_bytes);
    }

    #[test]
    fn a_plan_and_a_layout_that_disagree_are_refused_naming_the_tensor() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let plan = LoadPlan::build(&w, 0).unwrap();
        let mut l = Layout::new(&w, &plan, 0, 0).unwrap();
        // drop one tensor: the count check names the disagreement
        let victim = l.bindings.keys().next().unwrap().clone();
        l.bindings.remove(&victim);
        let e = check_agreement(&w, &plan, &l).unwrap_err().to_string();
        assert!(e.contains("tensors") || e.contains(&victim), "{e}");
        // and a binding of the wrong size is refused by name
        let mut l = Layout::new(&w, &plan, 0, 0).unwrap();
        let name = "embed.weight".to_string();
        if let Some(Binding::View { bytes, .. }) = l.bindings.get_mut(&name) {
            *bytes += 2;
        }
        let e = check_agreement(&w, &plan, &l).unwrap_err().to_string();
        assert!(e.contains(&name), "{e}");
    }

    #[test]
    fn the_upload_bound_tensors_are_the_ones_the_kernels_own() {
        // the MLA's five FP8 matrices and its three vectors, the shared expert, the router, the two sites, the engram's wkv
        assert_eq!(upload_bound("layers.0.attn.wq_a.weight", 40), Some("MlaDsv4"));
        assert_eq!(upload_bound("layers.0.attn.attn_sink", 40), Some("MlaDsv4"));
        assert_eq!(upload_bound("layers.0.attn.compressor.wkv.weight", 40), Some("MlaDsv4"));
        assert_eq!(upload_bound("layers.0.ffn.shared_experts.w2.scale", 40), Some("Fp8Weights"));
        assert_eq!(upload_bound("layers.0.ffn.gate.bias_vl", 40), Some("Dsv4Router"));
        assert_eq!(upload_bound("layers.0.hc_attn_fn", 40), Some("HcSite"));
        assert_eq!(upload_bound("layers.1.engram.wkv.scale", 40), Some("Fp8Weights"));
        assert_eq!(upload_bound("layers.0.attn_norm.weight", 40), Some("f32 gamma"));
        assert_eq!(upload_bound("norm.weight", 40), Some("f32 gamma"));
        // and the raw ones are views
        assert_eq!(upload_bound("layers.0.attn.wo_a.weight", 40), Some("MlaDsv4"));
        assert_eq!(upload_bound("embed.weight", 40), None);
        assert_eq!(upload_bound("layers.1.engram.q_weight", 40), None);
        assert_eq!(upload_bound("layers.0.ffn.experts.0.w1.trellis", 40), None);
        // a NextN layer is a raw view whatever its tensors are
        assert_eq!(upload_bound("mtp.0.attn.wq_a.weight", 40), None);
        assert_eq!(upload_bound("vision.blocks.0.mlp.w1.weight", 40), None);
    }

    /// The placement boundary is what makes the `O_DIRECT` path safe, and the old rule is what made it expensive. Over the real pack:
    /// every view is on a page boundary, no two of them in one region overlap, and the slack is a small fraction of a percent —
    /// where rounding each tensor to 2 MiB cost 2.74 GiB of region padding (3.69 GiB over every tensor of the pack).
    #[test]
    fn every_view_is_aligned_disjoint_and_padded_under_a_percent() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let plan = LoadPlan::full(&w).unwrap();
        let l = Layout::new(&w, &plan, crate::families::deepseek_v41::alloc::expert_records_bytes(&w), 0).unwrap();
        check_agreement(&w, &plan, &l).unwrap();
        // one window a region, sorted: no two tensors' page-rounded read windows may overlap
        let mut per_region: BTreeMap<u32, Vec<(u64, u64, &str)>> = BTreeMap::new();
        for (name, region, off, bytes) in l.view_spans() {
            assert_eq!(off % READ_ALIGN, 0, "`{name}` at {off} is not page-aligned");
            assert_eq!(off % MIN_DST_ALIGN, 0, "`{name}` at {off} is not {MIN_DST_ALIGN}-aligned");
            let r = &l.regions[region as usize];
            assert!(off + bytes <= r.bytes, "`{name}` runs past region {region} ({} B)", r.bytes);
            per_region.entry(region).or_default().push((off, bytes, name));
        }
        // a page-rounded window per tensor: [floor(off), ceil(off + len)) — the range the loader would have to read and write
        for (region, mut v) in per_region {
            v.sort();
            for pair in v.windows(2) {
                let (a_off, a_len, a) = pair[0];
                let (b_off, _, b) = pair[1];
                let a_end = (a_off + a_len).div_ceil(READ_ALIGN) * READ_ALIGN;
                assert!(a_end <= b_off, "`{a}`'s window ends at {a_end}, past `{b}` at {b_off} in region {region}");
            }
        }
        // the slack: at most one page a tensor, so under a percent of the views on this pack
        let padding = l.padding_bytes();
        assert!(padding < l.view_bytes / 100, "{padding} B of padding on {} B of views", l.view_bytes);
        // and the rule it replaced, on the same tensors: every non-expert view rounded to 2 MiB
        const OLD: u64 = 1 << 21;
        let old: u64 =
            l.view_spans().filter(|(n, _, _, _)| expert_of(n).is_none()).map(|(_, _, _, b)| b.div_ceil(OLD) * OLD).sum();
        let new: u64 = l
            .view_spans()
            .filter(|(n, _, _, _)| expert_of(n).is_none())
            .map(|(_, _, _, b)| b.div_ceil(READ_ALIGN) * READ_ALIGN)
            .sum();
        let view_bytes: u64 = l.view_spans().filter(|(n, _, _, _)| expert_of(n).is_none()).map(|(_, _, _, b)| b).sum();
        assert!(
            old - new > 2_500_000_000,
            "the old rule cost {} B of padding, the new one {} B",
            old - view_bytes,
            new - view_bytes
        );
        // 2.74 GiB of region padding under the 2 MiB rule (a header census over every trunk/MTP/vision tensor puts the pack's whole
        // 2 MiB waste at 3.69 GiB; the rest is the upload-bound tensors, which no region ever held), against 2.4 KiB now
        let old_gib = (old - view_bytes) as f64 / (1u64 << 30) as f64;
        assert!((2.5..3.0).contains(&old_gib), "{old_gib:.3} GiB of padding under the old rule");
        assert!((new - view_bytes) < 1 << 20, "{} B of padding under the new rule", new - view_bytes);
    }

    /// Thousands of small tensors (the shape of the NextN and vision towers, where the 2 MiB rule hurt most) pad under a percent.
    #[test]
    fn a_layout_of_thousands_of_small_tensors_pads_under_a_percent() {
        let mut next = 0u64;
        let sizes: Vec<u64> = (0..8_000u64).map(|i| 1_024 + (i * 37) % 65_536).collect();
        let offsets: Vec<u64> = sizes.iter().map(|&b| place(&mut next, b)).collect();
        let total: u64 = sizes.iter().sum();
        assert!(next - total < total / 100, "{} B of padding on {total} B", next - total);
        // page-aligned, in order, disjoint
        for (i, (&off, &b)) in offsets.iter().zip(sizes.iter()).enumerate() {
            assert_eq!(off % READ_ALIGN, 0);
            if i > 0 {
                assert!(offsets[i - 1] + sizes[i - 1] <= off, "tensor {i} overlaps its predecessor");
            }
            assert!(off % MIN_DST_ALIGN == 0 && b > 0);
        }
        // the same sizes under the old rule: two orders of magnitude worse, and it is the 2 MiB rounding that does it
        const OLD: u64 = 1 << 21;
        let old: u64 = sizes.iter().map(|&b| b.div_ceil(OLD) * OLD).sum();
        assert!(old > 50 * total, "{old} B under the old rule against {total} B of tensors");
    }

    /// **The two-phase boot's one invariant.** Phase one allocates a layout whose slab is zero; phase two allocates the expert regions
    /// of a layout built at the measured slab. For that to work, every region that is not an expert region must have the *same id and
    /// the same bytes* in both layouts — the window especially, because the cache is built over the region phase one allocated — and the
    /// expert regions must be the only difference. Otherwise phase two would write into a region phase one sized for something else.
    #[test]
    fn the_two_phases_agree_on_every_region_that_is_not_the_slab() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let budget = 4u64 << 30;
        let window = window_bytes_for(&w, budget);
        let phase1 = Layout::new(&w, &LoadPlan::build(&w, 0).unwrap(), 0, window).unwrap();
        let slab = 99_900_000_000u64.min(crate::families::deepseek_v41::alloc::expert_records_bytes(&w));
        let plan2 = LoadPlan::build(&w, slab).unwrap();
        let phase2 = Layout::new(&w, &plan2, slab, window).unwrap();
        check_agreement(&w, &plan2, &phase2).unwrap();

        assert_eq!(phase1.regions.len(), phase2.regions.len(), "the same regions exist in both phases");
        for (a, b) in phase1.regions.iter().zip(phase2.regions.iter()) {
            assert_eq!((a.id, a.kind), (b.id, b.kind), "region ids and kinds are stable across the phases");
            if !a.kind.is_experts() {
                assert_eq!(a.bytes, b.bytes, "region {} ({:?}) moved between the phases", a.id, a.kind);
            }
        }
        // the slab is the experts' *parts*; the regions hold their *records*, so they differ by the parts' alignment padding and by each
        // region's tail rounding — bounded, and the only thing phase two allocates
        let experts: u64 = phase2.regions.iter().filter(|r| r.kind.is_experts()).map(|r| r.bytes).sum();
        assert!(experts >= phase2.slab_bytes);
        assert!(
            experts - phase2.slab_bytes <= 14 * READ_ALIGN * phase2.resident_experts as u64,
            "{} B of expert padding over {} experts",
            experts - phase2.slab_bytes,
            phase2.resident_experts
        );
        assert_eq!(phase1.device_bytes() + experts, phase2.device_bytes());
        // the window phase one allocated is the window phase two's cache is built over
        assert_eq!(phase1.window_region(), phase2.window_region());
        assert_eq!(phase1.window_bytes, phase2.window_bytes);
        assert_eq!(phase1.window_slots, phase2.window_slots);
        assert_eq!(phase1.stream_record_bytes, phase2.stream_record_bytes);
        assert!(phase1.slab_bytes == 0 && phase1.streamed_bytes > 0, "phase one holds no experts at all");
    }

    /// **The expert layout recomputed for a smaller slab is consistent.** Halving the slab must give a resident set that is a subset of
    /// the larger one (the same layer-major walk), the same twelve parts bound inside each layer's own region, every expert that is not
    /// resident bound as `Streamed`, and `slab + streamed == the pack's expert bytes` at both sizes — which is what the bindings, the
    /// cache tables and the priced bytes are all derived from.
    #[test]
    fn a_smaller_slab_recomputes_the_expert_set_consistently() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let window = window_bytes_for(&w, 4u64 << 30);
        let big = 99_900_000_000u64.min(crate::families::deepseek_v41::alloc::expert_records_bytes(&w));
        let small = big / 2;
        let layouts: Vec<Layout> = [big, small]
            .into_iter()
            .map(|slab| {
                let plan = LoadPlan::build(&w, slab).unwrap();
                let l = Layout::new(&w, &plan, slab, window).unwrap();
                check_agreement(&w, &plan, &l).unwrap();
                // the plan and the layout name the same residents, and the bytes add up
                assert_eq!(l.resident_experts, plan.resident_experts);
                assert_eq!(l.slab_bytes + l.streamed_bytes, plan.expert_bytes);
                assert_eq!(l.window_slots as u64 * l.stream_record_bytes, l.window_bytes);
                l
            })
            .collect();
        let (big, small) = (&layouts[0], &layouts[1]);
        assert!(small.resident_experts < big.resident_experts, "half the slab holds fewer experts");
        assert!(small.slab_bytes < big.slab_bytes);

        for (l, other, nesting) in [(small, big, true), (big, small, false)] {
            for layer in 0..w.layers.len() {
                let region = l.regions.iter().find(|r| r.kind == RegionKind::Experts(layer)).unwrap();
                for e in 0..w.layers[layer].experts.len() {
                    let name = format!("layers.{layer}.ffn.experts.{e}.w1.trellis");
                    let bound = l.bindings.get(&name).unwrap();
                    // the parts of one expert are either all in its layer's region or all streamed, and the two slabs nest:
                    // everything the smaller slab holds is resident at the larger one, and everything streamed at the larger one
                    // streams at the smaller one too
                    let other_bound = other.bindings.get(&name).unwrap();
                    match bound {
                        Binding::View { region: r, offset, .. } => {
                            assert_eq!(*r, region.id, "`{name}` is not in its layer's region");
                            assert!(*offset + 8 <= region.bytes, "`{name}` runs past its region");
                            if nesting {
                                assert!(
                                    matches!(other_bound, Binding::View { .. }),
                                    "`{name}` is resident in the smaller slab but not the larger"
                                );
                            }
                        }
                        Binding::Streamed { .. } => {
                            if !nesting {
                                assert!(
                                    matches!(other_bound, Binding::Streamed { .. }),
                                    "`{name}` streams at the larger slab and is resident at the smaller"
                                );
                            }
                        }
                        other => panic!("`{name}` is bound as {other:?}"),
                    }
                    for proj in ["w1", "w2", "w3"] {
                        for part in ["trellis", "mul1", "suh", "svh"] {
                            let n = format!("layers.{layer}.ffn.experts.{e}.{proj}.{part}");
                            assert_eq!(
                                l.bindings.get(&n).is_some_and(|b| b.is_region()),
                                bound.is_region(),
                                "`{n}` disagrees with `{name}` about residency"
                            );
                        }
                    }
                }
            }
        }
        // the cache's record does not change with the slab: one slot is the widest expert either way
        assert_eq!(big.stream_record_bytes, small.stream_record_bytes);
        assert_eq!(big.window_bytes, small.window_bytes, "the window is a fixed charge, not a share of the slab");
    }

    #[test]
    fn an_experts_parts_are_laid_out_in_the_order_the_kernel_reads_them() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let g = &w.layers[0].experts.experts[0].w1;
        let a = gb_cuda::expert_cache::EXPERT_PART_ALIGN;
        let up = |x: u64| x.div_ceil(a) * a;
        let p = "layers.0.ffn.experts.0.w1";
        // trellis, mul1, suh, svh — each on the next aligned boundary after the one before
        let mul1 = up(g.trellis.bytes());
        let suh = up(mul1 + g.mul1.bytes());
        let svh = up(suh + g.suh.bytes());
        assert_eq!(expert_part_offset(&w, 0, 0, &format!("{p}.trellis")), Some(0));
        assert_eq!(expert_part_offset(&w, 0, 0, &format!("{p}.mul1")), Some(mul1));
        assert_eq!(expert_part_offset(&w, 0, 0, &format!("{p}.suh")), Some(suh));
        assert_eq!(expert_part_offset(&w, 0, 0, &format!("{p}.svh")), Some(svh));
        assert_eq!(expert_of("layers.7.ffn.experts.383.w2.trellis"), Some((7, 383)));
        assert_eq!(expert_of("layers.7.attn.wq_a.weight"), None);
    }

    /// The twelve parts of an expert sit in its record in kernel order, each on an [`EXPERT_PART_ALIGN`] boundary, with no overlap and
    /// nothing past the record's end. Packed back to back, the 4-byte `mul1` marker put `suh`, `svh` and every later projection off a
    /// 16-byte boundary and the first real prefill faulted with `cudaErrorMisalignedAddress`; before that the three projections all
    /// shared one offset. Neither is visible to a length check, so both are pinned here, on three experts of different bit widths.
    ///
    /// [`EXPERT_PART_ALIGN`]: gb_cuda::expert_cache::EXPERT_PART_ALIGN
    #[test]
    fn the_twelve_parts_of_an_expert_are_aligned_ordered_and_disjoint() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let a = gb_cuda::expert_cache::EXPERT_PART_ALIGN;
        for (l, x) in [(0usize, 0usize), (7, 3), (39, 383)] {
            let e = &w.layers[l].experts.experts[x];
            let (_, record) = e.record_layout();
            let parts: u64 = e.projections().iter().map(|g| g.bytes()).sum();
            let p = format!("layers.{l}.ffn.experts.{x}");
            let mut end = 0u64;
            for (proj, g) in [("w1", &e.w1), ("w2", &e.w2), ("w3", &e.w3)] {
                for (part, t) in [("trellis", &g.trellis), ("mul1", &g.mul1), ("suh", &g.suh), ("svh", &g.svh)] {
                    let off = expert_part_offset(&w, l, x, &format!("{p}.{proj}.{part}")).unwrap();
                    assert_eq!(off % a, 0, "{proj}.{part} of ({l}, {x}) at {off} is not {a}-aligned");
                    assert!(off >= end, "{proj}.{part} of ({l}, {x}) at {off} overlaps the part ending at {end}");
                    assert!(
                        off - end < a,
                        "{proj}.{part} of ({l}, {x}): {} B of padding, more than one alignment step",
                        off - end
                    );
                    end = off + t.bytes();
                }
            }
            assert!(end <= record, "({l}, {x}) ends at {end}, past its {record} B record");
            assert_eq!(record % a, 0, "({l}, {x}): a {record} B record would misalign the next one");
            // padding is bounded: at most one step per part plus the record's rounding
            assert!(record - parts < 13 * a, "({l}, {x}): {} B of padding", record - parts);
            assert_eq!(e.w1.mul1.bytes(), 4, "the marker the alignment exists for");
        }
    }
}
