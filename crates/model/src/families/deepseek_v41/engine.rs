//! Building a ready-to-serve DeepSeek-V4.1-Flash engine from a model folder.
//!
//! The order is the one that can refuse cheapest first, and every step fails closed by name:
//!
//! 1. **the config** ([`DeepseekV41Config::from_dir`]), which validates the relationships between its fields;
//! 2. **the pack's inventory** — every tensor the config implies, with its dtype and shape, so a pack that is missing one, has an extra one
//!    or has one of the wrong shape is refused before a byte is read;
//! 3. **the typed tree** ([`bind`]) — the inventory as [`DeepseekV41Weights`];
//! 4. **the load plan and admission** ([`super::plan`]) — the classification, and the arithmetic that decides whether the result fits. This
//!    is where a pack that does not fit is refused, and the refusal names the term that overflowed (`resident`, the KV, or the expert slab)
//!    and the bytes: [`AdmissionError`]'s `Display` is the message. **This is a prediction**, made before anything is allocated;
//! 5. **phase one** — the layout at a zero slab ([`super::load::Layout`], checked item by item against the plan), then every allocation
//!    that is *not* the routed-expert slab: the global/trunk regions and the streaming window, the load (one `cudaMalloc` a region, every
//!    view-bound tensor through the pipelined loader), the two engram tables `mmap`ped, the bind (the kernel types over those regions, and
//!    the uploads their constructors insist on), and the model (the DSA planes, the scratch, the block table);
//! 6. **the measurement** — [`settle_and_measure`]: release the process's own transient host pages, let the box settle, and read
//!    `MemAvailable`. What is left, less the floor and a small post-ready margin, is the slab ([`slab_from_measurement`]);
//! 7. **phase two** — the layout recomputed at that slab, which is what decides *which* experts are resident (the same layer-major walk as
//!    before, so the plan, the bindings, the cache tables and the priced bytes are one decision), the per-layer expert regions allocated
//!    with the same floor check (on a refusal the slab is re-solved, never aborted), the resident experts loaded, the `ExpertCache` built
//!    over the window phase one allocated, and [`super::device::bind_experts`] binding both;
//! 8. **the engine** — the backend and the scheduler's admission, charged the measured boot.
//!
//! The expert slab is a *budget*, not a given: [`BuildLimits::expert_slab_bytes`] defaults to the largest slab that fits at the requested
//! context and slot count ([`super::plan::largest_slab`]), which on the 121 GiB box is 108.82 GB of the 111.0 GB bank — 98.0% of the
//! experts resident, ~9 a layer streamed. A caller that wants the whole bank resident asks for it and is refused by name when it does not fit.
//! On the Spark at a 0.5 GiB floor the measured slab is ~94.7 GiB: 13,793 of 15,360 experts resident, 8.65 GiB streamed, and 0.14-0.16 GiB
//! of `MemAvailable` left at ready — boot after boot, because the slab is sized from what the boot actually left rather than from what
//! admission predicted it would.

use super::backend::Dsv41Backend;
use super::bind::bind;
use super::config::DeepseekV41Config;
use super::device;
use super::engram::{EngramFiles, EngramState, Loc, TokenMap, WkvFiles};
use super::inventory::{PackInventory, Row};
use super::load::{self, Layout, LoadOptions, Regions};
use super::model::{Dsv41Model, ModelOptions};
use super::plan::{self, Admission, AdmissionRequest, LoadPlan};
use super::vision;
use super::weights::DeepseekV41Weights;
use crate::error::ModelError;
use crate::serve::{gib, ServeOptions};
use gb_engine::{Engine, EngineConfig, NoDrafts};
use gb_exl3::checkpoint::{scan, Checkpoint};
use gb_sched::CostModel;
use std::sync::Arc;

/// The DeepSeek-V4.1-Flash engine, before its type is erased.
pub type Dsv41Engine = Engine<Dsv41Backend<Dsv41Model>>;

/// Rank 0 of a tensor-parallel group: the backend over the lockstep leader runtime, which forwards every
/// state-changing call to the followers and applies it on both ranks.
pub type Dsv41GroupEngine = Engine<Dsv41Backend<super::tp_runtime::TpLeader<Dsv41Model>>>;

/// Milliseconds per step by rows, for the scheduler's verify-row budget. The measured table of the GLM family is *not* transferable (a
/// different expert count, a different selection); this one is a placeholder that says so: flat, so the budget never trims a draft, and to be
/// replaced by the first measurement on a device (`tests/dsv41_bench.rs`, which does not exist yet).
pub(super) const STEP_MS: &[(usize, f64)] = &[(1, 1.0), (super::model::WINDOW_ROWS, 1.0)];

/// Rows of one chunk as admission prices it (see [`admit_pack`]).
pub const ADMISSION_CHUNK_ROWS: usize = 512;

/// What a build may be told to assume instead of measuring: the two numbers a test drives the refusal with, and nothing else.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuildLimits {
    /// Device bytes admission is judged against. `None`: what the allocator will allow — the device's `cudaMemGetInfo` free
    /// bytes capped by `MemAvailable`, less the floor ([`usable_bytes`]); 121 GiB less the floor when there is no device.
    pub device_bytes: Option<u64>,
    /// Routed-expert bytes to keep resident. `None`: the largest slab that fits.
    pub expert_slab_bytes: Option<u64>,
}

impl BuildLimits {
    pub fn at(device_bytes: u64) -> Self {
        Self { device_bytes: Some(device_bytes), expert_slab_bytes: None }
    }
}

/// The device's memory for admission: `cudaMemGetInfo`'s free bytes when there is a device, 121 GiB otherwise (the target box, and what a
/// CPU-only build is judged against so that a plan can still be priced and refused without one).
pub fn device_bytes() -> u64 {
    if let Ok(v) = std::env::var("REXL3_DSV41_DEVICE_BYTES") {
        if let Ok(n) = v.trim().parse::<u64>() {
            return n;
        }
    }
    if gb_cuda::device_available() {
        if let Ok((free, _)) = gb_cuda::rt::mem_info() {
            if free > 0 {
                return free as u64;
            }
        }
    }
    121 << 30
}

/// What admission may spend: `MemAvailable` less the floor the allocator enforces ([`load::allocate`]), the same number the
/// allocator checks, so the two cannot disagree. Never the device's free figure, nor the smaller of the two: on GB10 that
/// figure is `MemFree`, which leaves out the page cache `cudaMalloc` reclaims on demand (measured: a 100 GiB `cudaMalloc` with
/// 48.6 GiB `MemFree` and 70.6 GiB cached succeeded, reclaiming 53 GiB), so clamping to it hands whatever the box last read to
/// the NVMe tail - one boot after a buffered read streamed 25 GiB of experts instead of ~9. This is the planning rule of
/// [`crate::hostmem::planning_free`]. The device figure is used only where the host reports no `MemAvailable`;
/// `REXL3_DSV41_DEVICE_BYTES` is taken as given.
pub fn usable_bytes(floor_bytes: u64) -> u64 {
    if std::env::var("REXL3_DSV41_DEVICE_BYTES").is_ok() {
        return device_bytes();
    }
    crate::serve::mem_available().unwrap_or_else(device_bytes).saturating_sub(floor_bytes)
}

/// One `rexl3 event=dsv41_mem` line: where a boot stands against the box after a stage, so memory the plan did not price shows
/// up as the stage whose `MemAvailable` drop exceeds what admission charged for it.
fn mem_mark(stage: &str) -> Option<u64> {
    let m = crate::hostmem::MemInfo::read()?;
    let rss_anon = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("RssAnon:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .unwrap_or(0)
        * 1024;
    let cuda_free = if gb_cuda::device_available() { gb_cuda::rt::mem_info().map(|(f, _)| f).unwrap_or(0) } else { 0 };
    eprintln!(
        "rexl3 event=dsv41_mem stage={stage} mem_available_gib={:.3} mem_free_gib={:.3} cached_gib={:.3} rss_anon_gib={:.3} cuda_free_gib={:.3}",
        gib(m.available),
        gib(m.free),
        gib(m.cached),
        gib(rss_anon),
        gib(cuda_free as u64)
    );
    Some(m.available)
}

/// `event=dsv41_mem_check`: the plan against the boot. `consumed` is `MemAvailable` at `stage=admitted` less at `stage=model_built`,
/// and `unpriced` is what the boot took that admission did not charge for — the number that made a 108.01 GiB plan consume 115.28.
/// Over half a GiB is a term nobody priced, and says so at `level=warn`.
fn mem_check(planned: u64, admitted: Option<u64>) {
    let Some(before) = admitted else { return };
    let Some(m) = crate::hostmem::MemInfo::read() else { return };
    let consumed = before.saturating_sub(m.available);
    let unpriced = consumed as i128 - planned as i128;
    let level = if unpriced > (1i128 << 29) / 2 { " level=warn" } else { "" };
    eprintln!(
        "rexl3 event=dsv41_mem_check planned_gib={:.3} consumed_gib={:.3} unpriced_gib={:.3}{level}",
        gib(planned),
        gib(consumed),
        unpriced as f64 / (1u64 << 30) as f64
    );
}

/// The inventory of a scanned pack: every plain tensor and every EXL3 group's four parts, as rows.
fn inventory(cfg: &DeepseekV41Config, ck: &Checkpoint) -> Result<PackInventory, ModelError> {
    let skip_vision = cfg.vision.is_none();
    let mut rows: Vec<Row> = Vec::with_capacity(ck.plain.len() + ck.groups.len() * 4);
    for (name, e) in &ck.plain {
        if skip_vision
            && (name.starts_with("vision.")
                || name.starts_with("aligner.")
                || matches!(name.as_str(), "image_start" | "image_newline" | "image_end"))
        {
            continue;
        }
        rows.push(Row::new(name.clone(), e.dtype.clone(), e.shape.clone(), 0));
    }
    for prefix in ck.groups.keys() {
        if skip_vision && (prefix.starts_with("vision.") || prefix.starts_with("aligner.")) {
            continue;
        }
        let parts = ck.group_parts(prefix).map_err(|e| ModelError::Config(format!("{prefix}: {e}")))?;
        let g = parts.group;
        let mut push = |name: String, e: &gb_exl3::checkpoint::Entry| {
            rows.push(Row::new(name, e.dtype.clone(), e.shape.clone(), 0));
        };
        push(format!("{prefix}.trellis"), parts.trellis);
        push(format!("{prefix}.{}", g.in_scales), parts.in_scales);
        push(format!("{prefix}.{}", g.out_scales), parts.out_scales);
        // the codebook marker (`mul1` or `mcg`) is a plain scalar of the group: the scan already refuses a group whose
        // tiles need a marker it does not carry, and the family's inventory names it, so it is surfaced here too.
        for marker in ["mul1", "mcg"] {
            if let Some(e) = ck.part(&format!("{prefix}.{marker}")) {
                push(format!("{prefix}.{marker}"), e);
            }
        }
        if let Some(b) = parts.bias {
            push(format!("{prefix}.bias"), b);
        }
        // the codebook marker is a 4-byte plain tensor of the group; it is not in the tree, and `bind` does not claim it, so it is
        // checked by the loader instead (`LoadPlan::markers` is the GLM family's mechanism; this pack's markers are read with the
        // group's parameters and need no separate row).
    }
    PackInventory::from_rows(cfg, rows).map_err(|e| ModelError::Config(format!("the pack's inventory: {e}")))
}

/// Resolve this family's floor: `plan::DEFAULT_FLOOR_BYTES` (0.5 GiB) when `o.floor_bytes` is still the 8 GiB Qwen/GLM default,
/// or `REXL3_DSV41_FLOOR_GIB` if set.
pub fn floor_for(o: &ServeOptions) -> u64 {
    if let Ok(v) = std::env::var("REXL3_DSV41_FLOOR_GIB") {
        if let Ok(g) = v.trim().parse::<f64>() {
            if g.is_finite() && g >= 0.0 {
                return (g * (1u64 << 30) as f64) as u64;
            }
        }
    }
    if o.floor_bytes == (8 << 30) {
        1 << 28 // 0.25 GiB default floor on unified-memory GB10
    } else {
        o.floor_bytes
    }
}

/// Whether the prefix cache runs. `Off` and `Mib(0)` never. `On` (`--prefix-cache on`) and `Mib(n)` always: the operator required a
/// cache, and a snapshot that does not fit is skipped when it is taken (`Dsv41Model::can_snapshot_prefix`), not by running the whole
/// process without a cache because admission left little free. Only `Auto`, the default, still needs 2 GiB free after admission.
fn prefix_cache_enabled(asked: crate::serve::PrefixCacheOption, free_bytes: u64) -> bool {
    use crate::serve::PrefixCacheOption as P;
    match asked {
        P::Off | P::Mib(0) => false,
        P::Auto => free_bytes >= (2 << 30),
        P::On | P::Mib(_) => true,
    }
}

/// The plan and the admission for a bound tree, refusing by name when the pack does not fit. `shard` is `None` on the
/// single node and this rank's split table on a group: the probe, the slab and the plan are the rank's own.
pub fn admit_pack(
    cfg: &DeepseekV41Config,
    w: &DeepseekV41Weights,
    o: &ServeOptions,
    limits: BuildLimits,
    shard: Option<&super::tp::ShardTable>,
) -> Result<(LoadPlan, Admission), ModelError> {
    let floor = floor_for(o);
    let device = limits.device_bytes.unwrap_or_else(|| usable_bytes(floor));
    let req0 = request_for(w, o, device)?;
    // the slab is whatever the caller asked for, or the largest that fits at these settings — of the RANK's OWN experts
    let probe = LoadPlan::build_maybe(w, 0, shard).map_err(|e| ModelError::Config(e.to_string()))?;
    let slab = match limits.expert_slab_bytes {
        Some(b) => b.min(probe.expert_bytes),
        None => plan::largest_slab(&probe, cfg, &req0),
    };
    let mut req = req0;
    req.expert_slab_bytes = slab;
    let plan_budget = if slab >= probe.expert_bytes { super::plan::owned_expert_records(w, shard) } else { slab };
    let plan = LoadPlan::build_maybe(w, plan_budget, shard).map_err(|e| ModelError::Config(e.to_string()))?;
    let admission = plan::admit(&plan, cfg, &req).map_err(|e| ModelError::Config(e.to_string()))?;
    Ok((plan, admission))
}

/// Admission's question, less the slab: the context, the slots, the cache format, the chunk it prices, the streaming window, and the
/// device the answer is judged against. Everything admission reads that is not the expert budget.
fn request_for(w: &DeepseekV41Weights, o: &ServeOptions, device: u64) -> Result<AdmissionRequest, ModelError> {
    if o.slots == 0 {
        return Err(ModelError::Config("--slots must be at least 1".into()));
    }
    if o.max_model_len == 0 {
        return Err(ModelError::Config("--max-model-len must be at least 1".into()));
    }
    let mut req = AdmissionRequest::at_121_gib(o.max_model_len, 0);
    req.device_bytes = device;
    req.slots = if o.slots_auto { 1 } else { o.slots };
    // The cache's format, resolved once here: `--kv-dtype q4` (or `REXL3_DSV41_KV`) sets it, fp16 is the default, and the
    // price and the planes the model allocates are the same format because both read it from this request.
    req.kv = super::alloc::KvQuant::from_fmt(
        gb_cuda::dsv4_kvq4::configured_fmt(gb_cuda::dsv4_kvq4::KvFmt::F16).map_err(ModelError::Config)?,
    );
    // The chunk admission prices. It is the 512 rows the serving split was measured at; the executor runs chunks of at most
    // `dsv4_attn::MAX_ROWS` (128) rows, so the scratch term is a conservative over-reserve of about 0.2 GB.
    req.prefill_rows = o.prefill_chunk.clamp(1, ADMISSION_CHUNK_ROWS);
    // The streaming window is device memory and is charged before the slab is sized: a plan that spent the last 4 GiB on experts would
    // leave the tail nowhere to land. It is zero when the whole bank fits, which `expert_terms` decides.
    // priced as the layout will allocate it: whole layer quotas of the largest record, not the raw budget
    req.window_bytes = load::window_bytes_for(w, plan::DEFAULT_STREAM_WINDOW_BYTES);
    Ok(req)
}

pub fn build_engine(o: &ServeOptions) -> Result<Dsv41Engine, ModelError> {
    if std::env::var_os(gb_cuda::dsv4_kvq4::KV_FMT_ENV).is_none() {
        std::env::set_var(gb_cuda::dsv4_kvq4::KV_FMT_ENV, gb_cuda::dsv4_kvq4::KvFmt::Q4.name());
    }
    build_limited(o, BuildLimits::default())
}

/// Rank 0 of a tensor-parallel group (`--tp-ranks 2 --tp-rank 0`), with the rendezvous `listener` the family bound
/// BEFORE the pack was scanned (rank 1 can connect while this rank plans and loads). The engine built here is the
/// backend over the lockstep leader runtime: every state-changing call the backend makes is forwarded to the
/// followers and applied on both ranks, so the group runs one model in lockstep.
pub fn build_engine_group(o: &ServeOptions, listener: std::net::TcpListener) -> Result<Dsv41GroupEngine, ModelError> {
    if std::env::var_os(gb_cuda::dsv4_kvq4::KV_FMT_ENV).is_none() {
        std::env::set_var(gb_cuda::dsv4_kvq4::KV_FMT_ENV, gb_cuda::dsv4_kvq4::KvFmt::Q4.name());
    }
    build_group_limited(o, BuildLimits::default(), listener)
}

/// The head of every rank's boot: the config, the scan, the typed tree, and this rank's admission. Shared by the
/// single-rank build, rank 0 of a group and a follower, so the three boots cannot drift.
struct ScannedRank {
    cfg: DeepseekV41Config,
    ck: Checkpoint,
    w: DeepseekV41Weights,
    admission: plan::Admission,
    shard: Option<super::tp::ShardTable>,
}

fn scan_and_admit(
    o: &ServeOptions,
    limits: BuildLimits,
    tp_group: Option<&crate::serve::TpOptions>,
) -> Result<ScannedRank, ModelError> {
    let t = std::time::Instant::now();
    let _ = MODEL_DIR.set(o.model_dir.clone());
    evict_safetensors_page_cache(&o.model_dir, true);
    trim_heap();
    let window_rows = super::model::parse_window_rows(
        gb_cuda::dsv4_moe::runtime_env("REXL3_DSV41_WINDOW_ROWS").as_deref(),
        super::model::WINDOW_ROWS,
    )
    .map_err(ModelError::Config)?;
    super::spec::SpecLearn::parse(gb_cuda::dsv4_moe::runtime_env("REXL3_DSV41_SPEC_LEARN").as_deref())
        .map_err(ModelError::Config)?;
    if window_rows < o.slots {
        return Err(ModelError::Config(format!(
            "REXL3_DSV41_WINDOW_ROWS={window_rows}: the verify window would hold fewer rows than the server's {} slots (sequences share the window's rows), which would cut serving capacity to {window_rows} concurrent sequences",
            o.slots
        )));
    }
    let mut cfg = DeepseekV41Config::from_dir(&o.model_dir).map_err(ModelError::Config)?;
    if o.vision.is_none() {
        cfg.vision = None;
    }
    if o.max_model_len > cfg.max_positions
        && (o.max_model_len <= 1_048_576 || std::env::var("REXL3_DSV41_ALLOW_LONG_CTX").ok().as_deref() == Some("1"))
    {
        cfg.max_positions = o.max_model_len;
        let orig = cfg.rope.yarn.original_max_positions.max(1) as f64;
        cfg.rope.yarn.factor = (o.max_model_len as f64 / orig).max(cfg.rope.yarn.factor);
    }
    let ck = scan(&o.model_dir, false).map_err(|e| ModelError::Config(format!("{}: {e}", o.model_dir.display())))?;
    let inv = inventory(&cfg, &ck)?;
    let w = bind(&inv, &cfg).map_err(|e| ModelError::Config(e.to_string()))?;
    load::check_depth(&cfg, &w)?;
    eprintln!(
        "rexl3 event=dsv41_pack tensors={} gib={:.2} shards={} scan_s={:.2}",
        w.len(),
        gib(w.bytes()),
        ck.plain.values().map(|e| e.file.as_str()).collect::<std::collections::BTreeSet<_>>().len(),
        t.elapsed().as_secs_f64()
    );
    // ---- a tensor-parallel group: this rank's split table FIRST, fail-closed by name on a config whose counts do
    // not divide, before the pack is scanned or priced.
    let shard = match tp_group {
        Some(t) => Some(super::tp_boot::shard_table_for(t, &cfg, o)?),
        None => None,
    };
    let (_plan, admission) = admit_pack(&cfg, &w, o, limits, shard.as_ref())?;
    Ok(ScannedRank { cfg, ck, w, admission, shard })
}

/// This rank's boot gate: the pack's identity, the geometry its admission planned (the model's own `kv_blocks`, the
/// flags' context, slots and chunk rows — deterministic in the flags, so two ranks started alike compare equal), and
/// the family's facts (every environment knob that changes the arithmetic). Placed at the options' own rank of the
/// group (the default is rank 0 of 1 — the follower MUST be placed, or its Hello claims rank 0 and the leader's
/// roster refuses it by name).
fn gate_for_rank(o: &ServeOptions, sr: &ScannedRank) -> Result<super::tp_runtime::BootGate, ModelError> {
    let pack = super::tp_runtime::pack_identity(&o.model_dir).map_err(ModelError::Config)?;
    let gate = super::tp_runtime::gate_of(
        pack,
        sr.w.layers.len(),
        sr.admission.slots,
        sr.admission.context.div_ceil(crate::kvtables::BLOCK_TOKENS),
        sr.admission.context,
        o.prefill_chunk.clamp(1, gb_cuda::dsv4_attn::MAX_ROWS),
        super::tp_runtime::Dsv41Facts::of(o, sr.admission.kv.fmt()),
    );
    Ok(match o.tp.as_ref() {
        Some(t) => gate.at(t.rank, t.ranks),
        None => gate,
    })
}

/// What a follower's own plan said, for its boot log line and its load.
pub(crate) struct FollowerPlan {
    pub cfg: DeepseekV41Config,
    pub ck: Checkpoint,
    pub w: DeepseekV41Weights,
    pub admission: plan::Admission,
    pub shard: super::tp::ShardTable,
    /// The rank's resident weight bytes (the log's `weights_gib`).
    pub weights_bytes: u64,
    /// What the plan left free (the log's `free_after_gib`).
    pub free_after_weights: u64,
    /// The KV pool's blocks (the log's `kv_blocks`).
    pub kv_blocks: usize,
    /// The plan's spare bytes (the log's `spare_gib`).
    pub spare_bytes: u64,
}

/// A follower's own plan: the same scan and admission rank 0 made, from THIS rank's free memory (the geometry is a
/// pure function of the flags, so equal flags compare equal at the gate; the slab the follower sizes is its own).
pub(crate) fn plan_follower(o: &ServeOptions) -> Result<(FollowerPlan, super::tp_runtime::BootGate), ModelError> {
    let tp = o.tp.as_ref().filter(|t| t.is_group()).ok_or_else(|| {
        ModelError::Config("a tensor-parallel follower needs --tp-ranks 2 --tp-rank 1 (or any rank R above 0)".into())
    })?;
    let sr = scan_and_admit(o, BuildLimits::default(), Some(tp))?;
    let gate = gate_for_rank(o, &sr)?;
    let admission = &sr.admission;
    let fp = FollowerPlan {
        cfg: sr.cfg,
        ck: sr.ck,
        w: sr.w,
        admission: sr.admission,
        shard: sr.shard.expect("a group boot built a split table"),
        weights_bytes: admission.resident_bytes,
        free_after_weights: admission.free_bytes,
        kv_blocks: admission.context.div_ceil(crate::kvtables::BLOCK_TOKENS),
        spare_bytes: admission.headroom_bytes,
    };
    Ok((fp, gate))
}

/// A follower's load: its own scan's plan, built with RANK 0'S pinned geometry (the gate's `model_options` — it may
/// hold more memory, never run other numbers) and its combine arms over the group's communicator.
pub(crate) fn load_follower(
    o: &ServeOptions,
    fp: FollowerPlan,
    pinned: &ModelOptions,
    comm: std::sync::Arc<dyn super::tp::TpComm>,
) -> Result<Dsv41Model, ModelError> {
    let loaded = load_rank(o, &fp.cfg, &fp.w, &fp.ck, &fp.admission, Some(&fp.shard), Some(comm), Some(pinned))?;
    Ok(loaded.model)
}

/// The post-ready margin: device bytes the slab does **not** get, held back for what the boot itself allocates after the measurement.
///
/// It is small because everything large is already allocated when the slab is sized — this is only the boot's own tail: the expert
/// cache's first lease and its staging pool, the block table's lazily-mapped pages, the CUDA driver's page tables for the slab itself
/// (charged as the slab is touched, not at `cudaMalloc`), and whatever the allocator hands back late. **Measured** on the DGX Spark: a
/// boot that sized the slab at exactly `MemAvailable - floor` was killed at `stage=model_built` in three of five boots; the term that
/// matters in practice is the slab's own page tables, which is why [`slab_after_refusal`] solves for the ratio rather than trusting
/// this number to cover it. Overridable with `REXL3_DSV41_POST_READY_MARGIN_GIB`.
pub const DEFAULT_POST_READY_MARGIN_BYTES: u64 = (1u64 << 30) * 15 / 100; // 0.15 GiB

/// Device bytes held back for the **first forward**: the workspaces the kernels' libraries create lazily on first use — cuBLAS's
/// handle and workspace above all — plus the request path's own buffers. Measured on the Spark, first request after a boot:
/// `MemAvailable` 0.499 -> 0.136 GiB, so 0.36 GiB, and a second request costs nothing more. Without this term the floor is respected
/// by the boot and crossed by the first token, which is the failure that matters least visibly and most.
///
/// Overridable with `REXL3_DSV41_FORWARD_RESERVE_GIB`.
pub const DEFAULT_FORWARD_RESERVE_BYTES: u64 = (1u64 << 30) * 45 / 100; // 0.45 GiB

/// Device bytes the expert load itself costs on top of the slab it fills: the loader's staging slots and the cache's tables. Measured on
/// the Spark at 0.12 GiB net of the [`trim_host`] that follows it (0.30 GiB gross), and only for the decision to stop shrinking — the
/// load is not re-run to check it, so this is an allowance, not a reservation.
const LOAD_RESERVE_BYTES: u64 = (1u64 << 30) * 15 / 100;

/// [`DEFAULT_FORWARD_RESERVE_BYTES`], or `REXL3_DSV41_FORWARD_RESERVE_GIB` when it is set to a number of GiB.
pub fn forward_reserve_bytes() -> u64 {
    reserve_from(|k| std::env::var(k).ok(), "REXL3_DSV41_FORWARD_RESERVE_GIB", DEFAULT_FORWARD_RESERVE_BYTES)
}

/// Device bytes held back for **after readiness**: the memory monitor's 1 GiB margin plus what one prefix-snapshot capture can
/// transiently need (about 0.2 GiB at the longest captured prefix). Full expert residency is elected only when it leaves this much
/// free, and every slab sizing keeps it, so a full-residency boot still feeds the snapshot pool and the monitor.
pub const DEFAULT_POST_READY_RESERVE_BYTES: u64 = (1u64 << 30) * 6 / 5; // 1.2 GiB

/// [`DEFAULT_POST_READY_RESERVE_BYTES`]. Fixed and automatic: the reserve exists to feed a gate, so it is not a knob.
pub fn post_ready_reserve_bytes() -> u64 {
    DEFAULT_POST_READY_RESERVE_BYTES
}

/// [`DEFAULT_POST_READY_MARGIN_BYTES`], or `REXL3_DSV41_POST_READY_MARGIN_GIB` when it is set to a number of GiB.
pub fn post_ready_margin_bytes() -> u64 {
    margin_from(|k| std::env::var(k).ok())
}

/// [`post_ready_margin_bytes`] over an injected environment, so the parse (and its refusals) can be tested without setting a variable
/// in a process whose other tests are running.
fn margin_from(env: impl Fn(&str) -> Option<String>) -> u64 {
    reserve_from(env, "REXL3_DSV41_POST_READY_MARGIN_GIB", DEFAULT_POST_READY_MARGIN_BYTES)
}

fn reserve_from(env: impl Fn(&str) -> Option<String>, key: &str, default: u64) -> u64 {
    env(key)
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|g| g.is_finite() && *g >= 0.0)
        .map(|g| (g * (1u64 << 30) as f64) as u64)
        .unwrap_or(default)
}

/// The slab a measurement buys: what is left, less the floor, less the margin — floored at zero and capped at every expert's record.
/// One function so the arithmetic is testable on its own, and so no call site can forget the cap or the floor.
///
/// `margin` is the post-ready margin **plus** the forward reserve: the boot must leave the floor intact both when it declares itself
/// ready and when the first request runs, and the second is the one that used to cross it. `reserve` is the post-ready reserve
/// ([`post_ready_reserve_bytes`]) the snapshot pool and the memory monitor need to stay fed after the boot.
fn slab_from_measurement(measured: u64, floor: u64, margin: u64, reserve: u64, expert_records: u64) -> u64 {
    measured.saturating_sub(floor).saturating_sub(margin).saturating_sub(reserve).min(expert_records)
}

/// Whether every expert fits resident while the post-ready reserve still stays free: the window a full-residency boot does not pay
/// (`win_reclaim`) counts toward the fit, but the floor and the reserve must both remain untouched.
fn full_residency_fits(measured: u64, floor: u64, win_reclaim: u64, expert_records: u64, reserve: u64) -> bool {
    measured.saturating_sub(floor).saturating_sub(reserve).saturating_add(win_reclaim) >= expert_records
}

/// The slab a refusal buys on the retry.
///
/// A `cudaMalloc` of `n` bytes costs more than `n` on a GB10 — the page tables that map it are host memory in the same pool, measured at
/// about 0.7% — so a slab sized as `measured - floor - margin` is short by that fraction and the last regions cross the floor. Rather
/// than step down by a guess, this solves for it: the regions this attempt did allocate (`allocated_bytes`) consumed
/// `budget - (avail - floor)`, so their real overhead ratio is known, and the largest slab that fits is `(avail - floor) / (1 + ratio)`,
/// less one record of slack. One step in the common case, and it lands within a record of the largest slab that fits.
fn slab_after_refusal(budget: u64, floor: u64, refusal: &load::RegionRefusal, record: u64) -> u64 {
    let left = refusal.avail.saturating_sub(floor);
    if refusal.allocated_bytes == 0 {
        // nothing was allocated: the first region did not fit, so the slab is whatever is left
        return left.saturating_sub(record);
    }
    let spent = budget.saturating_sub(left);
    let overhead = spent.saturating_sub(refusal.allocated_bytes);
    let ratio = overhead as f64 / refusal.allocated_bytes as f64;
    // every byte of the slab costs `1 + ratio`, and the whole budget is what may be spent: `S * (1 + ratio) = budget`
    let solved = (budget as f64 / (1.0 + ratio)) as u64;
    solved.saturating_sub(record).max(record)
}

/// Settle the box, then read `MemAvailable`: the number the slab is sized from.
///
/// Two things have to happen before that read is worth anything. The process's own transient host pages must go back — every kernel
/// constructor here reads through a `Vec<u8>` of the shard's bytes (167 MB for one `wq_b`), and glibc keeps a freed arena's pages
/// resident, so a boot that just bound 40 layers is holding hundreds of megabytes it will never touch again — and the box must be
/// allowed to finish the reclaim that the last `cudaMalloc` started. So: [`trim_host`], a device sync, and then sample until two
/// consecutive reads agree within 4 MiB (up to a second), returning the lower of the pair. A reading that never settles still returns
/// its last sample: the margin is what absorbs the difference, and this never fabricates a number.
fn settle_and_measure(model_dir: &std::path::Path) -> Option<u64> {
    trim_host(model_dir);
    let mut last: Option<u64> = None;
    for _ in 0..20 {
        let now = crate::hostmem::MemInfo::read()?.available;
        if let Some(prev) = last {
            if prev.abs_diff(now) < (4 << 20) {
                return Some(now.min(prev));
            }
        }
        last = Some(now);
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    last
}

/// Drop page cache on the pack's non-engram safetensors shards (the two engram shards are mmap'd and read on demand, so
/// their pages are the tables' own). Safe to call at any time: the shards are immutable, and a dropped page costs one
/// re-read, never a wrong byte.
pub(crate) fn fadvise_pack_shards(model_dir: &std::path::Path) {
    evict_safetensors_page_cache(model_dir, false);
}

/// The pack directory, set once at build start so the allocation loop can reclaim shard pages without threading the
/// path through every signature.
pub(crate) static MODEL_DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// Evict cached `.safetensors` shard pages (and adjacent `target/release` build artifacts) from the Linux page cache
/// before admission reads `MemAvailable`.
fn evict_safetensors_page_cache(model_dir: &std::path::Path, include_engram: bool) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        let evict_file = |path: &std::path::Path| {
            if let Ok(f) = std::fs::File::open(path) {
                unsafe {
                    libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
                }
            }
        };
        if let Ok(rd) = std::fs::read_dir(model_dir) {
            for entry in rd.flatten() {
                let path = entry.path();
                let name = entry.file_name();
                let s = name.to_string_lossy();
                if s.ends_with(".safetensors")
                    && (include_engram || (!s.contains("00016-of-00017") && !s.contains("00017-of-00017")))
                {
                    evict_file(&path);
                }
            }
        }
        if include_engram {
            if let Ok(exe) = std::env::current_exe() {
                if let Some(release_dir) = exe.parent() {
                    for sub in ["deps", "build", "incremental"] {
                        let mut stack = vec![release_dir.join(sub)];
                        while let Some(dir) = stack.pop() {
                            if let Ok(rd) = std::fs::read_dir(&dir) {
                                for entry in rd.flatten() {
                                    let p = entry.path();
                                    if p.is_dir() {
                                        stack.push(p);
                                    } else if p.is_file() {
                                        evict_file(&p);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Best-effort glibc heap trim after a large load/evict (`malloc_trim` is glibc-only; no-op on macOS).
#[cfg(not(target_os = "macos"))]
fn trim_heap() {
    // SAFETY: `malloc_trim` only releases free heap pages; it takes no pointers and cannot invalidate live allocations.
    unsafe {
        libc::malloc_trim(0);
    }
}

#[cfg(target_os = "macos")]
fn trim_heap() {}

/// Hand the process's own transient host pages back to the box: `malloc_trim` for the glibc arenas the per-tensor reads and the
/// loader's staging buffers were freed into, drop page cache on non-engram safetensors shards, then a device sync.
fn trim_host(model_dir: &std::path::Path) {
    trim_heap();
    evict_safetensors_page_cache(model_dir, false);
    let _ = gb_cuda::rt::device_sync();
}

/// [`build_engine`], with admission's two assumptions injected ([`BuildLimits`]). A tensor-parallel group is refused
/// here by name: it boots through [`build_engine_group`], where the family has bound the rendezvous listener before
/// the pack was scanned.
pub fn build_limited(o: &ServeOptions, limits: BuildLimits) -> Result<Dsv41Engine, ModelError> {
    if o.tp.as_ref().is_some_and(|t| t.is_group()) {
        return Err(ModelError::Config(
            "a tensor-parallel group boots through the family's build, which binds the --tp-peer listener before the \
             pack is scanned"
                .into(),
        ));
    }
    let sr = scan_and_admit(o, limits, None)?;
    eprintln!("{}", admission_line(&sr.admission));
    let target_fp = crate::fingerprint::fingerprint(&sr.ck);
    let loaded = load_rank(o, &sr.cfg, &sr.w, &sr.ck, &sr.admission, None, None, None)?;
    finish_engine(loaded.model, o, &loaded.final_admission, loaded.per_token, &target_fp)
}

/// [`build_engine_group`], with admission's two assumptions injected.
#[allow(clippy::too_many_arguments)]
pub fn build_group_limited(
    o: &ServeOptions,
    limits: BuildLimits,
    listener: std::net::TcpListener,
) -> Result<Dsv41GroupEngine, ModelError> {
    let tp =
        o.tp.as_ref()
            .filter(|t| t.is_group())
            .ok_or_else(|| ModelError::Config("a group build needs --tp-ranks 2 --tp-rank 0 --tp-peer HOST:PORT".into()))?;
    let sr = scan_and_admit(o, limits, Some(tp))?;
    eprintln!("{}", admission_line(&sr.admission));
    let target_fp = crate::fingerprint::fingerprint(&sr.ck);
    // the gate this rank offers its followers, and the followers under one deadline — each identified by its Hello,
    // gated (geometry, facts, build, NCCL environment) before anything is offered or read
    let gate = gate_for_rank(o, &sr)?;
    eprintln!(
        "rexl3 event=tp_leader_waiting peer={} ranks={} slots={} kv_pages={} max_context={} prefill_rows={} free_after_gib={:.2} nccl=\"{}\"",
        tp.peer,
        tp.ranks,
        gate.slots,
        gate.kv_pages,
        gate.max_context,
        gate.prefill_rows,
        crate::serve::gib(sr.admission.free_bytes),
        crate::lockstep::nccl_summary(&gate.nccl)
    );
    let t = super::tp_boot::timeouts(tp);
    let links = crate::lockstep::accept_group(&listener, &gate, &t)?;
    let topo = super::tp_boot::topology(tp);
    let t0 = std::time::Instant::now();
    let mut loaded_out: Option<(plan::Admission, u64)> = None;
    let leader = super::tp_runtime::boot_leader_group(
        links,
        &gate,
        &t,
        |rv| super::tp_boot::make_comm(topo, rv),
        |comm| {
            let r = load_rank(o, &sr.cfg, &sr.w, &sr.ck, &sr.admission, sr.shard.as_ref(), Some(comm), None)?;
            let _ = loaded_out.insert((r.final_admission, r.per_token));
            Ok(r.model)
        },
    )?;
    leader.on_group_lost(Box::new(super::tp_boot::abort_group_comm));
    leader.on_comm_health(Arc::new(super::tp_boot::group_comm_health));
    gb_engine::boot::mark("tp_group_ready", None);
    eprintln!("rexl3 event=tp_group_ready ranks={} boot_s={:.1}", tp.ranks, t0.elapsed().as_secs_f64());
    let (final_admission, per_token) = loaded_out.expect("the leader's load closure ran");
    finish_engine(leader, o, &final_admission, per_token, &target_fp)
}

/// The one-line admission report every rank's boot prints (a term that overflows is refused before this, naming the
/// term — this is the accounting of what was admitted).
fn admission_line(admission: &plan::Admission) -> String {
    format!(
        "rexl3 event=dsv41_admission resident_gib={:.3} mtp_experts_gib={:.3} slab_gib={:.3} streamed_gib={:.3} kv_gib={:.3} scratch_gib={:.3} \
         planes_gib={:.3} rope_gib={:.3} reserve_gib={:.3} headroom_gib={:.3} context={} total_gib={:.3} free_gib={:.3}",
        gib(admission.resident_bytes),
        gib(admission.mtp_expert_bytes),
        gib(admission.expert_slab_bytes),
        gib(admission.streamed_expert_bytes),
        gib(admission.kv_bytes),
        gib(admission.scratch_bytes),
        // `kv_gib` and `planes_gib` are the same allocation by construction (Admission::kv_bytes *is* Dsv4Planes::bytes); both
        // names are printed so an operator reading an older or newer line sees the same number.
        gib(admission.kv_bytes),
        gib(admission.rope_bytes),
        gib(admission.runtime_reserve_bytes),
        gib(admission.headroom_bytes),
        admission.context,
        gib(admission.total_bytes),
        gib(admission.free_bytes)
    )
}

/// What one rank's load produced: its model (its combine arms enabled over the group's communicator when it is a
/// rank of one) and the admission the engine built over it is charged from.
pub(crate) struct LoadedRank {
    pub model: Dsv41Model,
    pub final_admission: plan::Admission,
    pub per_token: u64,
}

/// One rank's weight load — phases one and two of the boot, the whole device path from the plan to a bound model.
/// Shared by the single-rank build, rank 0 of a group (inside the lockstep leader's load closure) and a follower, so
/// the three cannot drift. `comm` is the group's communicator (the model's combine arms are enabled over it, one
/// exchange a site); `pinned` is RANK 0'S planned geometry for a follower (`None` on the ranks that derive their own
/// — equal flags derive equal geometry, and the boot gate has compared it before any of this runs).
#[allow(clippy::too_many_arguments)]
fn load_rank(
    o: &ServeOptions,
    cfg: &DeepseekV41Config,
    w: &DeepseekV41Weights,
    ck: &Checkpoint,
    admission: &plan::Admission,
    shard_ref: Option<&super::tp::ShardTable>,
    comm: Option<Arc<dyn super::tp::TpComm>>,
    pinned: Option<&ModelOptions>,
) -> Result<LoadedRank, ModelError> {
    // ---- **phase one**: every allocation that is not the routed-expert slab, so the slab can be sized from what is left.
    //
    // The order is the point. The plan above is a prediction made before anything was allocated: it prices `MemAvailable` at admission,
    // and the boot then spends what that prediction did not see — the CUDA context and its images, the per-tensor host `Vec<u8>` the
    // kernel constructors read through, the loader's staging slots, the RoPE tables, the DSA planes, the scratch, the GPU page tables
    // that scale with the allocation. On the Spark that residue varies by ±0.4 GiB boot to boot, which at a 0.5 GiB floor is the
    // difference between a boot that fits and one the kernel kills at `stage=model_built`. So nothing is put in the slab until
    // everything else exists and has been measured.
    let plan0 = LoadPlan::build_maybe(w, 0, shard_ref).map_err(|e| ModelError::Config(e.to_string()))?;
    let layout0 = Layout::new_tp(w, &plan0, 0, 0, shard_ref)?;
    load::check_agreement(w, &plan0, &layout0)?;
    let admitted_at = mem_mark("admitted");
    gb_cuda::rt::init_context()?;
    let _ = mem_mark("cuda_context");
    // **the regions first, the cache over them.** The streaming window is one region of the layout, so the cache's slab is a
    // view of it rather than a second `cudaMalloc` of the same bytes: that duplicate cost 3.303 GiB of a 118 GiB pool. The
    // region's own `MemAvailable` floor check covers the window; the cache allocates nothing.
    let floor = floor_for(o);
    let lo = LoadOptions::from_env(floor);
    let mut regions: Regions = load::allocate_non_experts(&layout0, floor)?;
    let _ = mem_mark("regions_allocated");
    let report = load::run_where(&o.model_dir, &layout0, ck, &regions, &lo.loader, |k| !k.is_experts())?;
    let _ = mem_mark("weights_loaded");
    eprintln!(
        "rexl3 event=dsv41_load regions={} gib={:.3} workers={} spans={} direct_spans={} view_gib={:.3} upload_gib={:.3} mapped_gib={:.3}",
        regions.count(),
        gib(regions.bytes()),
        report.workers,
        report.spans,
        report.direct_spans,
        gib(layout0.view_bytes),
        gib(layout0.upload_bytes),
        gib(layout0.mapped_bytes)
    );
    // ---- the engram tables: mapped, never copied
    let engram = open_engram(cfg, ck, admission.slots)?;
    let _ = mem_mark("engram_opened");
    // the kernels' own uploads (6.7 GiB of FP8 attention/shared tensors), the shared RoPE tables, and every non-expert view
    let dev = device::bind(cfg, w, ck, &layout0, &regions, admission.context, engram, shard_ref)?;
    let _ = mem_mark("bound");
    eprintln!(
        "rexl3 event=dsv41_bound layers={} mtp={} engram_tables={} resident_experts={} device_gib={:.3}",
        dev.layers.len(),
        dev.mtp.len(),
        dev.engram.len(),
        dev.resident_experts(),
        gib(dev.bytes())
    );
    // the scheduler's view of the same plan: the weights are already charged by our own admission, so only the KV is left to price —
    // and it is priced from the planes' own formula (`Dsv4Planes::bytes`' derivative), not from an average of a total
    let per_token = plan::kv_bytes_per_token(cfg, admission.slots, admission.kv).max(1);
    // a follower of a group is built with RANK 0'S planned geometry (it may hold more memory, never run other
    // numbers); every other rank derives the same numbers from the same flags
    let (slots, context) = match pinned {
        Some(p) => (p.slots, p.max_context),
        None => (admission.slots, admission.context),
    };
    if pinned.is_some_and(|p| p.slots > admission.slots || p.max_context > admission.context) {
        return Err(ModelError::Config(format!(
            "rank 0 planned {} slots at a {}-token context and this rank's admission holds only {} at {}",
            pinned.unwrap().slots,
            pinned.unwrap().max_context,
            admission.slots,
            admission.context
        )));
    }
    if pinned.is_some_and(|p| p.kv != admission.kv.fmt()) {
        return Err(ModelError::Config(format!(
            "rank 0 planned the {} KV cache and this rank admitted {} (pass the same --kv-dtype to every rank)",
            kv_name(pinned.unwrap().kv),
            kv_name(admission.kv.fmt())
        )));
    }
    // the model's own state: the DSA planes (~0.56 GiB q4) and the scratch (~0.13 GiB). Before the slab, so the measurement sees them.
    let is_tp = o.tp.as_ref().is_some_and(|t| t.is_group());
    let chunk_rows = match pinned {
        Some(p) => p.chunk_rows,
        None if !is_tp && o.prefill_chunk == 4096 => 512,
        None => o.prefill_chunk.clamp(1, gb_cuda::dsv4_attn::MAX_ROWS),
    };
    let mut model = Dsv41Model::new(
        cfg.clone(),
        dev,
        &ModelOptions {
            slots,
            kv_blocks: context.div_ceil(crate::kvtables::BLOCK_TOKENS),
            max_context: context,
            chunk_rows,
            layers: w.layers.len(),
            kv: admission.kv.fmt(),
        },
    )?;
    let _ = mem_mark("model_built");
    mem_check(
        admission.total_bytes.saturating_sub(admission.expert_slab_bytes).saturating_sub(admission.expert_window_bytes),
        admitted_at,
    );
    // ---- the vision tower, when images are served: its weights were loaded with everything else (they are plain
    // region views of the layout), so this binds the kernels over them, allocates the tower's scratch (before the
    // measurement, like every other boot allocation), and hands the model the span embeddings the splice reads.
    // `--no-vision` (or a pack with no tower) skips the tower; a follower rank still loads the three span delimiter
    // embeddings (`image_start`, `image_end`, `image_newline`, 24 KiB) when present in `layout0` so it can replay
    // `Dsv41Call::PrefillMm` in lockstep with rank 0.
    if let (Ok(s), Ok(e), Ok(n)) = (
        regions.tensor(&layout0, "image_start"),
        regions.tensor(&layout0, "image_end"),
        regions.tensor(&layout0, "image_newline"),
    ) {
        if let Ok(span_emb) = super::vision_serve::SpanEmbeddings::from_device(&s, &e, &n) {
            model.set_span_embeddings(span_emb);
        }
    }
    let mut vision_pub: Option<(&crate::serve::VisionSlot, super::vision_serve::Dsv41Encoder)> = None;
    if let (Some(slot), Some(v)) = (&o.vision, cfg.vision.as_ref()) {
        let cap = o.max_image_tokens.min(v.max_image_tokens).min(vision::MAX_IMAGE_TOKENS);
        let tower =
            vision::Dsv4VisionTower::bind(v, &vision::TowerOptions::new(cap), &|name, _want| regions.tensor(&layout0, name))?;
        let (weights_gib, scratch_gib) = (gib(tower.weight_bytes()), gib(tower.scratch_bytes()));
        let pp = super::image_preprocessor(&o.model_dir, o)?.map(|p| p.id()).unwrap_or_default();
        let features = (cap as u64) * vision::OUT_HIDDEN as u64 * 4 * 8; // eight images in flight, host rows
        let enc = super::vision_serve::Dsv41Encoder::new(tower, crate::serve::FeatureBudget::new(features), &pp, cap)?;
        eprintln!(
            "rexl3 event=dsv41_vision tokens={cap} weights_gib={weights_gib:.3} scratch_gib={scratch_gib:.3} features_mib={:.0}",
            features as f64 / 1048576.0
        );
        vision_pub = Some((slot, enc));
    }
    let _ = mem_mark("vision_bound");
    // ---- **the measurement**, and the slab it buys.
    //
    // Everything above is allocated, uploaded, mapped and built; what is left on the box is the honest budget for routed experts, and it
    // is what this sizes from — not the prediction admission made before any of it existed. `margin` is two measured terms: the boot's
    // own tail ([`DEFAULT_POST_READY_MARGIN_BYTES`], 0.15 GiB) and the first forward's lazy workspaces
    // ([`DEFAULT_FORWARD_RESERVE_BYTES`], 0.45 GiB) — the second is what keeps the floor intact when the first request runs.
    let post_margin = if std::env::var_os("REXL3_DSV41_POST_READY_MARGIN_GIB").is_some() { post_ready_margin_bytes() } else { 0 };
    let fwd_reserve = if std::env::var_os("REXL3_DSV41_FORWARD_RESERVE_GIB").is_some() { forward_reserve_bytes() } else { 0 };
    let margin = post_margin.saturating_add(fwd_reserve);
    // the post-ready reserve: what must still be free after the boot for the prefix-snapshot pool and the memory monitor
    let reserve = post_ready_reserve_bytes();
    let measured = settle_and_measure(&o.model_dir).ok_or_else(|| {
        ModelError::Config(
            "deepseek_v41: /proc/meminfo could not be read, so the expert slab cannot be sized from measurement".into(),
        )
    })?;
    let record = load::largest_record_bytes(w).max(1);
    let expert_records = super::plan::owned_expert_records(w, shard_ref);
    let win_bytes = load::window_bytes_for(w, plan::DEFAULT_STREAM_WINDOW_BYTES);
    let budget = measured.saturating_sub(floor).saturating_sub(margin);
    // Keeping 100% of experts resident (`expert_records`, with `win = 0`) uses less memory than shrinking the slab and allocating
    // the 4 GiB streaming window (`win_bytes ≈ 3.303 GiB`). Always attempt full expert residency first whenever `measured - floor - reserve + win_reclaim >= expert_records`.
    let win_reclaim = if layout0.window_slots > 0 { win_bytes } else { 0 };
    let mut slab = if full_residency_fits(measured, floor, win_reclaim, expert_records, reserve) {
        expert_records
    } else {
        let s = slab_from_measurement(measured, floor, margin, reserve, expert_records);
        let win0 = if admission.expert_window_bytes > 0 { admission.expert_window_bytes } else { win_bytes };
        (s.saturating_sub(win0) * 975) / 1000
    };
    let need = floor.saturating_add(margin).saturating_add(reserve).saturating_add(LOAD_RESERVE_BYTES);
    let (layout, plan2) = loop {
        let plan2 = LoadPlan::build_maybe(w, slab, shard_ref).map_err(|e| ModelError::Config(e.to_string()))?;
        let win = if plan2.streamed_expert_bytes == 0 {
            0
        } else if admission.expert_window_bytes > 0 {
            admission.expert_window_bytes
        } else {
            win_bytes
        };
        let layout = Layout::new_tp(w, &plan2, slab, win, shard_ref)?;
        load::check_agreement(w, &plan2, &layout)?;
        let next = match load::allocate_experts(&layout, &mut regions, floor) {
            Ok(()) => {
                // Never shrink when 100% of experts (`15,360 / 15,360`) are already resident in device memory (`win == 0`),
                // because shrinking would require allocating the 4 GiB streaming window (`win_bytes`) and consume ~3.3 GiB MORE memory.
                if plan2.streamed_expert_bytes == 0 {
                    break (layout, plan2);
                }
                let after = crate::serve::mem_available().unwrap_or(0);
                if after >= need || slab <= record {
                    break (layout, plan2);
                }
                regions.release_experts(&layout);
                let next = slab.saturating_sub((need - after).saturating_add(record).min(slab));
                eprintln!(
                    "rexl3 event=dsv41_slab_shrink slab_gib={:.3} next_gib={:.3} after_alloc_gib={:.3} need_gib={:.3} reason=below_reserve",
                    gib(slab),
                    gib(next),
                    gib(after),
                    gib(need)
                );
                next
            }
            Err(refusal) if slab > record => {
                let effective_budget = if layout.window_bytes == 0 { budget.saturating_sub(win_bytes) } else { budget };
                let next = slab_after_refusal(effective_budget, floor, &refusal, record);
                if next >= slab {
                    return Err(refusal.into());
                }
                eprintln!(
                    "rexl3 event=dsv41_slab_shrink slab_gib={:.3} next_gib={:.3} shortfall_gib={:.3} reason=\"{refusal}\"",
                    gib(slab),
                    gib(next),
                    gib(refusal.shortfall)
                );
                next
            }
            Err(refusal) => return Err(refusal.into()),
        };
        slab = next;
    };
    let _ = mem_mark("experts_allocated");
    let cache = open_cache(cfg, ck, &layout, &regions)?;
    let _ = mem_mark("expert_cache");
    let mut exp_loader = lo.loader.clone();
    if crate::serve::mem_available().is_some_and(|a| a < (2u64 << 30)) {
        exp_loader.workers = exp_loader.workers.min(2);
    }
    let expert_report = load::run_where(&o.model_dir, &layout, ck, &regions, &exp_loader, |k| k.is_experts())?;
    let _ = mem_mark("experts_loaded");
    device::bind_experts(cfg, w, ck, &layout, &regions, cache, &mut model.dev)?;
    // the loader's staging slots and the per-expert host copies are dead now; give them back before the boot calls itself ready
    trim_host(&o.model_dir);
    let _ = mem_mark("experts_bound");
    eprintln!(
        "rexl3 event=dsv41_experts resident={} resident_gib={:.3} streamed_gib={:.3} window_slots={} window_gib={:.3} record_gib={:.6} spans={} \
         mtp_experts={} mtp_experts_gib={:.3} mtp_format={}",
        layout.resident_experts,
        gib(layout.slab_bytes),
        gib(layout.streamed_bytes),
        layout.window_slots,
        gib(layout.window_bytes),
        layout.stream_record_bytes as f64 / (1u64 << 30) as f64,
        expert_report.spans,
        w.mtp.iter().map(|m| m.experts.len()).sum::<usize>(),
        gib(layout.mtp_expert_bytes),
        w.mtp.iter().filter_map(|m| m.expert_format()).next().unwrap_or("none")
    );
    // The one line an operator greps for: what admission predicted, what the box actually had, and what the slab became.
    let ready_at = mem_mark("ready");
    eprintln!(
        "rexl3 event=dsv41_slab planned_gib={:.3} measured_gib={:.3} ready_gib={:.3} floor_gib={:.3} margin_gib={:.3} forward_gib={:.3} \
         reserve_gib={:.3} slab_gib={:.3} resident={} of={} streamed_gib={:.3}",
        gib(admission.expert_slab_bytes),
        gib(measured),
        ready_at.map(gib).unwrap_or(0.0),
        gib(floor),
        gib(post_margin),
        gib(fwd_reserve),
        gib(reserve),
        gib(layout.slab_bytes),
        layout.resident_experts,
        w.layers.iter().map(|l| l.experts.len()).sum::<usize>(),
        gib(layout.streamed_bytes)
    );
    // The scheduler's charge is the *measured* boot, not the predicted one: the runtime reserve is not added back — the measurement
    // already contains it (the CUDA context, the page tables and the host heap were all allocated before it was taken), so charging it
    // twice would hold ~0.7 GiB of experts out of the slab for nothing. The device figure is the one admission was given, so the terms
    // stay comparable with the prediction; the *real* headroom at ready is the `stage=ready` mark, which is what `dsv41_slab` reports.
    let mut req = request_for(w, o, admission.device_bytes)?;
    req.expert_slab_bytes = layout.slab_bytes;
    req.runtime_reserve_bytes = Some(0);
    let needed_dev = plan2.resident_bytes
        + plan2.mtp_expert_bytes
        + layout.slab_bytes
        + layout.window_bytes
        + plan::attn_planes_bytes(cfg, req.context, req.slots, req.kv)
        + super::model::Scratch::bytes_for(
            cfg,
            req.prefill_rows.clamp(super::model::WINDOW_ROWS, gb_cuda::dsv4_attn::MAX_ROWS),
            req.context,
        )
        + plan::rope_bytes(cfg, req.context)
        + req.headroom_bytes;
    req.device_bytes = req.device_bytes.max(needed_dev);
    let final_admission = plan::admit(&plan2, cfg, &req).map_err(|e| ModelError::Config(e.to_string()))?;
    let _ = mem_mark("ready");
    mem_check(final_admission.total_bytes, admitted_at);
    // the group's model: the rank's combine arms over the communicator, one exchange a site — the step from here on is
    // the group's, byte-identical to the single-node one at every combine
    if let (Some(comm), Some(table)) = (comm, shard_ref) {
        model.enable_tp(comm, table.clone())?;
    }
    if let Some((slot, enc)) = vision_pub {
        let tokens = enc.max_tokens();
        let _ = slot.set(std::sync::Arc::new(crate::serve::VisionService::new(Box::new(enc), tokens)));
    }
    Ok(LoadedRank { model, final_admission, per_token })
}

/// A KV format's short name, for this file's refusals.
fn kv_name(kv: gb_cuda::dsv4_kvq4::KvFmt) -> &'static str {
    match kv {
        gb_cuda::dsv4_kvq4::KvFmt::F16 => "fp16",
        gb_cuda::dsv4_kvq4::KvFmt::Q4 => "q4",
    }
}

/// The engine over a loaded runtime — the model on a single rank, the leader on rank 0 of a group: the drafting the
/// options ask for, the scheduler's admission charged the measured boot, and the row budget of the family's window.
fn finish_engine<R: super::tp_runtime::Dsv41Runtime>(
    rt: R,
    o: &ServeOptions,
    final_admission: &plan::Admission,
    per_token: u64,
    target_fp: &str,
) -> Result<Engine<Dsv41Backend<R>>, ModelError> {
    let window_rows = super::model::parse_window_rows(
        gb_cuda::dsv4_moe::runtime_env("REXL3_DSV41_WINDOW_ROWS").as_deref(),
        super::model::WINDOW_ROWS,
    )
    .map_err(ModelError::Config)?;
    let drafting = o.drafting && o.spec.drafts();
    let spec_tree = drafting
        && matches!(o.spec, crate::serve::SpecMode::Auto | crate::serve::SpecMode::Mtp)
        && rt.trees_capable()
        && gb_cuda::dsv4_moe::runtime_env("REXL3_DSV41_SPEC_TREE").is_some_and(|v| v.trim() != "0");
    if spec_tree {
        eprintln!("rexl3 event=dsv41_spec_trees verify=path_windows window_rows={} knob=REXL3_DSV41_SPEC_TREE", window_rows);
    }
    if rt.spec_learn_on() {
        eprintln!("rexl3 event=dsv41_spec_learn learn=on knob=REXL3_DSV41_SPEC_LEARN");
    }
    if window_rows != super::model::WINDOW_ROWS {
        eprintln!(
            "rexl3 event=dsv41_window_rows rows={window_rows} default={} knob=REXL3_DSV41_WINDOW_ROWS",
            super::model::WINDOW_ROWS
        );
    }
    let default_drafts = super::model::MTP_MAX_DRAFTS.min(window_rows - 1);
    let max_drafts = if drafting {
        std::env::var("REXL3_DSV41_MTP_DRAFTS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(default_drafts)
            .min(window_rows - 1)
            .min(super::model::MTP_MAX_DRAFTS)
    } else {
        0
    };
    let chunk_rows = rt.chunk_rows();
    let mut backend = Dsv41Backend::new(rt).with_spec(o.spec, drafting).with_chunk(chunk_rows);
    let cache_on = prefix_cache_enabled(o.prefix_cache, final_admission.free_bytes);
    if cache_on {
        let ns = super::backend::prefix_namespace(target_fp, final_admission.kv.fmt().name(), backend.m.has_mtp());
        backend = backend.with_default_prefix_cache(ns);
    }
    let mm = gb_sched::ModelMemory {
        weights_bytes: 0,
        kv_bytes_per_token: per_token.max(1),
        state_bytes_per_slot: 0,
        activation_reserve: 0,
    };
    let admitted = gb_sched::Admission::new(final_admission.total_bytes, 0, mm, final_admission.slots)
        .map(|a| {
            a.with_max_context(final_admission.context as u64)
                .with_kv_tokens((final_admission.context * final_admission.slots) as u64)
        })
        .map_err(|e| ModelError::Config(e.to_string()))?;
    let cost = CostModel::from_table(STEP_MS.to_vec());
    let make: Box<dyn Fn() -> Box<dyn gb_engine::Drafter>> =
        if drafting && final_admission.slots == 1 && o.spec != crate::serve::SpecMode::Mtp {
            Box::new(move || Box::new(gb_spec::CopyDrafter::new(8, max_drafts)) as Box<dyn gb_engine::Drafter>)
        } else {
            Box::new(|| Box::new(NoDrafts) as Box<dyn gb_engine::Drafter>)
        };
    let mut eng = Engine::new(backend, admitted, EngineConfig { max_rows: window_rows, max_drafts, cost, drafting }, make);
    if spec_tree {
        eng = eng.with_trees(gb_engine::TreeConfig::default());
    }
    if cache_on {
        let mut pcfg = crate::serve::prefix_config_from_env()?;
        if std::env::var_os("REXL3_PREFIX_MIN_CUT").is_none() {
            pcfg.policy.min_cut = 64;
        }
        if std::env::var_os("REXL3_PREFIX_TAIL").is_none() {
            pcfg.policy.tail = 48;
        }
        if std::env::var_os("REXL3_PREFIX_GHOST_GAIN").is_none() {
            pcfg.policy.ghost_min_gain = 128;
        }
        if std::env::var_os("REXL3_PREFIX_MIN_GAP").is_none() {
            pcfg.policy.min_gap = 96;
        }
        eng = eng.with_prefix_cache(pcfg);
    }
    Ok(eng)
}

/// Open the expert cache over the pack, when the layout streams a tail. `None` when every expert is resident: no window, no cache, no
/// NVMe reads on the forward path.
///
/// The table is built from the pack's own headers — the twelve parts of each expert, at their real shard offsets — so a slot filled by
/// [`gb_cuda::expert_cache::PackExpertSource`] is byte-for-byte the record a resident expert would have had. The slab is one allocation of
/// `window_slots` records; the fill threads and the staging pool come from [`gb_cuda::expert_cache::Config`]'s defaults (2 workers, 2 in
/// flight).
fn open_cache(
    cfg: &DeepseekV41Config,
    ck: &Checkpoint,
    layout: &Layout,
    regions: &Regions,
) -> Result<Option<Arc<gb_cuda::expert_cache::ExpertCache>>, ModelError> {
    use gb_cuda::expert_cache::{Config, DeviceSlab, ExpertCache, ExpertTable, PackExpertSource};
    if layout.window_slots == 0 || layout.streamed_bytes == 0 {
        return Ok(None);
    }
    let record = layout.stream_record_bytes as usize;
    let table = ExpertTable::from_checkpoint(ck, record, cfg.layers, cfg.moe.routed_experts)
        .map_err(|e| ModelError::Config(format!("deepseek_v41: the expert table: {e}")))?;
    if table.is_empty() {
        return Err(ModelError::Config("deepseek_v41: the layout streams experts but the pack's headers name none".into()));
    }
    let source = PackExpertSource::open(table)
        .map_err(|e| ModelError::Config(format!("deepseek_v41: the expert pack could not be opened: {e}")))?;
    // the slab **is** the layout's `Streamed` region: one allocation, and the layout stays the single source of addresses
    let id = layout
        .window_region()
        .ok_or_else(|| ModelError::Config("deepseek_v41: the layout streams experts but has no window region".into()))?;
    let window = regions.view(id, 0, layout.window_bytes)?;
    // a device without a CUDA runtime has no slab: the cache is then None, and a boot that reaches the forward without one is what the
    // `no device` self-skips are for. On a CPU-only build the window is still priced and allocated; nothing fills it.
    let slab = match DeviceSlab::over(window, layout.window_slots, record) {
        Ok(s) => s,
        Err(gb_cuda::expert_cache::CacheError::NoDevice) => return Ok(None),
        Err(e) => return Err(ModelError::Config(format!("deepseek_v41: the expert window: {e}"))),
    };
    let o = Config {
        budget_bytes: layout.window_bytes as usize,
        expert_bytes: record,
        layers: cfg.layers,
        experts_per_layer: cfg.moe.routed_experts,
        ..Config::default()
    };
    debug_assert_eq!(o.slots(), layout.window_slots, "the window is a whole number of layer quotas");
    Ok(Some(ExpertCache::new(o, Arc::new(source), Arc::new(slab)).map_err(|e| ModelError::Config(e.to_string()))?))
}

/// An engram tensor's bytes come from the pack's own index first, then — for the two tables, which
/// quantized packs omit — from the [`super::engram::SHARDS_ENV`] sidecar scan (the deepseek-ai base
/// model dir, or an engram-only dir whose index lists the four table tensors). A tensor in neither
/// is an error naming both sources, never a silent skip.
fn engram_entry<'a>(
    name: &str,
    ck: &'a Checkpoint,
    sidecar: Option<&'a Checkpoint>,
) -> Result<(&'a Checkpoint, &'a gb_exl3::checkpoint::Entry), ModelError> {
    if let Some(e) = ck.plain.get(name) {
        return Ok((ck, e));
    }
    if let Some(sc) = sidecar {
        if let Some(e) = sc.plain.get(name) {
            return Ok((sc, e));
        }
    }
    Err(ModelError::Config(format!(
        "deepseek_v41: engram tensor {name} is in neither the pack nor the REXL3_DSV41_ENGRAM_SHARDS sidecar"
    )))
}

/// Scan the sidecar dir named by [`super::engram::SHARDS_ENV`] once per boot; unset or empty is
/// `None`, a set-but-unscannable dir is an error naming the dir (fail-closed).
fn engram_shard_sidecar() -> Result<Option<Checkpoint>, ModelError> {
    let Some(dir) = std::env::var_os(super::engram::SHARDS_ENV).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let path = std::path::Path::new(&dir);
    let sc = gb_exl3::checkpoint::scan_indexed(path, false)
        .map_err(|e| ModelError::Config(format!("deepseek_v41: {}: {e}", path.display())))?;
    Ok(Some(sc))
}

/// Open the engram layers' mappings and their token map. A pack with no engram layers opens nothing. The six tensors of a
/// layer live inside the pack's shards; the scan carries their file and absolute offset, and the lengths stay the config's.
fn open_engram(cfg: &DeepseekV41Config, ck: &Checkpoint, slots: usize) -> Result<Vec<EngramState>, ModelError> {
    if !cfg.has_engram() {
        return Ok(Vec::new());
    }
    let map = Arc::new(TokenMap::from_env_or_dir(&ck.root).map_err(|e| ModelError::Config(e.to_string()))?);
    let mut out = Vec::with_capacity(cfg.engram.layer_ids.len());
    let sidecar = engram_shard_sidecar()?;
    for &layer in &cfg.engram.layer_ids {
        let name = |suffix: &str| format!("layers.{layer}.engram.{suffix}");
        // (path, absolute offset, dtype) for one tensor, pack-first then the sidecar scan
        let entry = |suffix: &str| -> Result<(Loc, String), ModelError> {
            let n = name(suffix);
            let (owner, e) = engram_entry(&n, ck, sidecar.as_ref())?;
            let offset = owner.file_offset(e).map_err(|e| ModelError::Config(format!("deepseek_v41: {n}: {e}")))?;
            Ok((Loc { path: owner.root.join(&e.file), offset }, e.dtype.clone()))
        };
        let loc = |suffix: &str| entry(suffix).map(|(l, _)| l);
        // the wkv's two pack forms: plain FP8 tensors, or the EXL3 group the engine loads itself
        let wkv_prefix = format!("layers.{layer}.engram.wkv");
        let (wkv, wkv_group) = if ck.plain.contains_key(&format!("{wkv_prefix}.weight")) {
            (WkvFiles::Plain { weight: loc("wkv.weight")?, scale: loc("wkv.scale")? }, None)
        } else if ck.groups.contains_key(&wkv_prefix) {
            let g = ck.load_group(&wkv_prefix).map_err(|e| ModelError::Config(format!("deepseek_v41: {wkv_prefix}: {e}")))?;
            (WkvFiles::Group, Some(g))
        } else {
            return Err(ModelError::Config(format!(
                "deepseek_v41: layer {layer}'s engram wkv is neither a plain FP8 tensor nor an EXL3 group \
                 in the pack or the REXL3_DSV41_ENGRAM_SHARDS sidecar"
            )));
        };
        let (q_weight, q_dt) = entry("q_weight")?;
        let (k_weight, k_dt) = entry("k_weight")?;
        let files = EngramFiles {
            embed: loc("embed.weight")?,
            scale: loc("embed.scale")?,
            q_weight,
            q_f32: q_dt == "F32",
            k_weight,
            k_f32: k_dt == "F32",
            wkv,
        };
        out.push(
            EngramState::open(&cfg.engram, layer, cfg.hidden, &files, slots, wkv_group.as_ref(), Arc::clone(&map)).map_err(
                |e| {
                    ModelError::Config(format!(
                "deepseek_v41: the engram table of layer {layer} could not be mapped ({e}); set REXL3_DSV41_ENGRAM_TOKEN_MAP \
                 to the compressed token map"
            ))
                },
            )?,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::alloc::KvQuant;
    use super::*;
    use serde_json::Value;

    const CONFIG: &str = include_str!("../../../../../testdata/deepseek_v41/config.json");
    const PACK: &str = "/tmp/dsv41_inventory.json";

    fn real() -> DeepseekV41Config {
        DeepseekV41Config::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    fn weights() -> Option<DeepseekV41Weights> {
        let text = std::fs::read_to_string(PACK).ok()?;
        let listing: Value = serde_json::from_str(&text).unwrap();
        let cfg = real();
        let inv = PackInventory::from_json(&cfg, &listing).unwrap();
        Some(bind(&inv, &cfg).unwrap())
    }

    fn options() -> ServeOptions {
        ServeOptions::new(std::path::PathBuf::from("/nonexistent"))
    }

    /// At the target device the pack is admitted, and every term is what the plan said it was.
    #[test]
    fn the_pack_is_admitted_at_121_gib_with_the_measured_split() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let cfg = real();
        let mut o = options();
        o.max_model_len = 32_768;
        o.slots = 1;
        let (plan, a) = admit_pack(&cfg, &w, &o, BuildLimits::at(121 << 30), None).unwrap();
        assert_eq!(a.resident_bytes, plan.resident_bytes);
        assert_eq!(a.expert_slab_bytes + a.streamed_expert_bytes, plan.expert_bytes);
        // the slab is the largest that fits with the window reserved: what the 4 GiB window, the KV the pack actually allocates and
        // the measured reserve leave of the 111.0 GB bank, read from disk through the window's records
        assert!(a.free_bytes < 1 << 20, "{} B free", a.free_bytes);
        // the window is priced as the layout allocates it: 160 slots of the largest record (3.30 GiB), not the raw 4 GiB budget
        assert_eq!(a.expert_window_bytes, load::window_bytes_for(&w, plan::DEFAULT_STREAM_WINDOW_BYTES));
        assert!(a.expert_window_bytes < plan::DEFAULT_STREAM_WINDOW_BYTES && a.expert_window_bytes > 3 << 30);
        assert!((4.0e9..7.0e9).contains(&(a.streamed_expert_bytes as f64)), "{} B streams", a.streamed_expert_bytes);
        let charged = a.terms().iter().map(|(_, b)| *b).sum::<u64>() + a.headroom_bytes;
        assert_eq!(charged, a.total_bytes);
        assert!(a.context >= 32_768, "context {}", a.context);
    }

    /// **The admission table of this pack**, printed: the Spark's idle `MemAvailable` (118.1 GiB) less the 0.5 GiB floor, at a
    /// 32K context over one slot. Resident experts and streamed bytes with every charge in place, and how far short of
    /// all-experts-resident the pack is — plus what the MTP routed experts and the bf16 head would buy back if they were EXL3.
    #[test]
    fn the_admission_table_at_the_serving_floor() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let cfg = real();
        let mut o = options();
        o.max_model_len = 32_768;
        o.slots = 1;
        // `MemAvailable` at stage=admitted, measured on the Spark: 118.118 GiB
        let available = (118.118f64 * (1u64 << 30) as f64) as u64;
        let device = available - (1u64 << 29); // the 0.5 GiB floor
                                               // both formats, priced from the same plan: the format is the only term that moves
        let probe = LoadPlan::build(&w, 0).unwrap();
        let total_experts = cfg.layers * cfg.moe.routed_experts;
        for kv in [KvQuant::F16, KvQuant::Q4] {
            let mut req = AdmissionRequest::serving(32_768, 1, device);
            req.kv = kv;
            let slab = plan::largest_slab(&probe, &cfg, &req);
            req.expert_slab_bytes = slab;
            let plan = LoadPlan::build(&w, slab).unwrap();
            let a = plan::admit(&plan, &cfg, &req).unwrap();
            // what must leave if every expert is resident: the window is zero there, so the shortfall is the slab's shortfall
            // plus the window it no longer pays
            let all_resident =
                plan.resident_bytes + plan.mtp_expert_bytes + plan.expert_bytes + a.kv_bytes + a.scratch_bytes + a.rope_bytes;
            let short_by = (all_resident + plan::runtime_reserve_for(all_resident)) as i128 - device as i128;
            eprintln!(
                "  kv={:<3} resident {:.3} GiB, slab {:.3} GiB ({} of {total_experts} experts), streamed {:.3} GiB, \
                 window {:.3} GiB, kv {:.3} GiB, scratch {:.3} GiB, rope {:.3} GiB, reserve {:.3} GiB, short_by {:.3} GiB",
                kv.fmt().name(),
                gib(a.resident_bytes),
                gib(a.expert_slab_bytes),
                plan.resident_experts,
                gib(a.streamed_expert_bytes),
                gib(a.expert_window_bytes),
                gib(a.kv_bytes),
                gib(a.scratch_bytes),
                gib(a.rope_bytes),
                gib(a.runtime_reserve_bytes),
                short_by as f64 / (1u64 << 30) as f64,
            );
            // the slab is the largest that fits: nothing is left over but the flooring of the 0.4% reserve
            assert!(a.total_bytes <= device && device - a.total_bytes < 1 << 20, "{} of {device}", a.total_bytes);
            // every expert is not resident: a tail streams through the window at both formats
            assert!(a.streamed_expert_bytes > 0);
            assert_eq!(a.expert_window_bytes, plan::DEFAULT_STREAM_WINDOW_BYTES);
            // the measured reserve, and no headroom: this is the whole point of the change
            assert_eq!(a.headroom_bytes, 0);
            assert_eq!(a.runtime_reserve_bytes, plan::runtime_reserve_for(a.allocated_bytes));
            assert!(short_by > 0, "all-experts-resident fits, so this table proves nothing");
        }
        // the admission the serve actually builds is the env's format, and it is admitted
        let (_plan, a) = admit_pack(&cfg, &w, &o, BuildLimits::at(device), None).unwrap();
        // with the duplication, the double-priced KV, the nominal scratch, the headroom and the flat reserve all gone, 97% of the
        // 111.0 GB bank is resident; the rest streams through the window
        assert!(a.expert_slab_bytes > 100_000_000_000, "{} B of slab", a.expert_slab_bytes);
    }

    /// `build()` refuses a pack whose admission overflows, and names the term.
    #[test]
    fn build_refuses_a_pack_whose_resident_set_does_not_fit_and_names_the_term() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let cfg = real();
        let o = options();
        // a device smaller than the 18.75 GB resident set
        let e = admit_pack(&cfg, &w, &o, BuildLimits::at(1 << 30), None).unwrap_err().to_string();
        assert!(e.contains("resident set"), "{e}");
        assert!(e.contains("exceeds the device"), "{e}");
        // the message carries the two numbers an operator needs
        assert!(e.contains("18750160200"), "{e}");
        assert!(e.contains(&(1u64 << 30).to_string()), "{e}");
        // `build_engine` is `admit_pack` followed by the load, and the admission runs first: nothing is allocated, and no device is
        // touched, before this refusal is produced.
        assert!(admit_pack(&cfg, &w, &o, BuildLimits::at(1 << 30), None).is_err());
    }

    /// A slab that does not fit is refused as the slab term, not as the context's.
    #[test]
    fn an_oversized_slab_is_refused_by_name_with_the_largest_that_fits() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let cfg = real();
        let o = options();
        let mut o = o;
        o.max_model_len = 32_768;
        o.slots = 1;
        let e = admit_pack(&cfg, &w, &o, BuildLimits { device_bytes: Some(121 << 30), expert_slab_bytes: Some(u64::MAX) }, None)
            .unwrap_err()
            .to_string();
        // every expert resident does not fit at a 32K context over one slot
        assert!(e.contains("expert slab") || e.contains("slab"), "{e}");
        assert!(e.contains("largest slab that fits"), "{e}");
    }

    /// A context longer than the model's positions is refused before any memory is priced.
    #[test]
    fn a_context_past_the_models_positions_is_refused() {
        let Some(w) = weights() else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let cfg = real();
        let mut o = options();
        o.max_model_len = cfg.max_positions + 1;
        let e = admit_pack(&cfg, &w, &o, BuildLimits::default(), None).unwrap_err().to_string();
        assert!(e.contains("positions"), "{e}");
    }

    // ------------------------------------------------------------------ the measured slab (two-phase boot)

    /// The margin is 0.15 GiB, and the environment can move it — but never to something that is not a number of GiB.
    #[test]
    fn the_post_ready_margin_defaults_to_zero_and_is_overridable() {
        // the margin defaults to 0.15 GiB; REXL3_DSV41_POST_READY_MARGIN_GIB overrides it
        assert_eq!(DEFAULT_POST_READY_MARGIN_BYTES, 161_061_273);
        let none = |_: &str| None;
        assert_eq!(margin_from(none), DEFAULT_POST_READY_MARGIN_BYTES);
        let at = |v: &'static str| move |_: &str| Some(v.to_string());
        assert_eq!(margin_from(at("0.25")), 268_435_456);
        assert_eq!(margin_from(at(" 1.5 ")), 1_610_612_736);
        assert_eq!(margin_from(at("0")), 0, "zero is a legal, deliberate margin");
        // an unparseable or negative value is the default, not a silent zero
        assert_eq!(margin_from(at("")), DEFAULT_POST_READY_MARGIN_BYTES);
        assert_eq!(margin_from(at("half")), DEFAULT_POST_READY_MARGIN_BYTES);
        assert_eq!(margin_from(at("-1")), DEFAULT_POST_READY_MARGIN_BYTES);
        assert_eq!(margin_from(at("NaN")), DEFAULT_POST_READY_MARGIN_BYTES);
        assert_eq!(margin_from(at("inf")), DEFAULT_POST_READY_MARGIN_BYTES);
        // the forward reserve is its own term, default 0.45 GiB (measured: a first request costs 0.36)
        assert_eq!(DEFAULT_FORWARD_RESERVE_BYTES, 483_183_820);
        assert_eq!(reserve_from(|_: &str| None, "X", DEFAULT_FORWARD_RESERVE_BYTES), DEFAULT_FORWARD_RESERVE_BYTES);
        assert_eq!(reserve_from(at("1"), "X", DEFAULT_FORWARD_RESERVE_BYTES), 1 << 30);
    }

    /// The slab is `measured - floor - margin - reserve`, capped at every expert's record and floored at zero: the whole arithmetic
    /// of the measurement, including the two ends an operator can reach by moving the floor.
    #[test]
    fn the_slab_is_what_the_measurement_leaves() {
        let gib_ = |g: u64| g << 30;
        // the Spark: 99.9 GiB left after phase one, at a 0.5 GiB floor, a 0.15 GiB margin and the post-ready reserve,
        // against 103.4 GiB of expert records
        let records = 103_400_000_000u64;
        let slab = slab_from_measurement(
            99_900_000_000,
            gib_(1) / 2,
            DEFAULT_POST_READY_MARGIN_BYTES,
            DEFAULT_POST_READY_RESERVE_BYTES,
            records,
        );
        assert_eq!(slab, 99_900_000_000 - gib_(1) / 2 - DEFAULT_POST_READY_MARGIN_BYTES - DEFAULT_POST_READY_RESERVE_BYTES);
        assert_eq!(
            slab + gib_(1) / 2 + DEFAULT_POST_READY_MARGIN_BYTES + DEFAULT_POST_READY_RESERVE_BYTES,
            99_900_000_000,
            "nothing is invented"
        );
        // every expert fits: the slab is the records, and the excess is the window the plan no longer pays
        assert_eq!(slab_from_measurement(u64::MAX / 2, 0, 0, 0, records), records);
        // a box with less left than the floor plus the margin plus the reserve buys no slab at all, rather than underflowing
        assert_eq!(
            slab_from_measurement(
                gib_(1) / 4,
                gib_(1) / 2,
                DEFAULT_POST_READY_MARGIN_BYTES,
                DEFAULT_POST_READY_RESERVE_BYTES,
                records
            ),
            0
        );
        assert_eq!(slab_from_measurement(0, 0, 0, 0, records), 0);
    }

    /// Full residency is elected only when the post-ready reserve stays free: the window a full-residency boot does not pay counts
    /// toward the fit, but the floor and the reserve must both remain untouched.
    #[test]
    fn full_residency_keeps_the_post_ready_reserve_free() {
        let gib_ = |g: u64| g << 30;
        let records = 102_383_000_000u64;
        let floor = gib_(1) / 2;
        let reserve = DEFAULT_POST_READY_RESERVE_BYTES;
        let win = 3_303_000_000u64;
        // the exact boundary, no window: `measured` must clear floor + reserve + records
        assert!(full_residency_fits(records + floor + reserve, floor, 0, records, reserve));
        assert!(!full_residency_fits(records + floor + reserve - 1, floor, 0, records, reserve));
        // the reclaimed window buys exactly `win` of slack: the reserve still must survive
        assert!(full_residency_fits(records + floor + reserve - win, floor, win, records, reserve));
        assert!(!full_residency_fits(records + floor + reserve - win - 1, floor, win, records, reserve));
        // the measured 262K boot (104.3 GiB after phase one) still elects full residency with the reserve left
        assert!(full_residency_fits(104_300_000_000, floor, win, records, reserve));
        // the reserve is independent of the floor: a higher floor and enough measurement still elect
        assert!(full_residency_fits(105_800_000_000, gib_(1), win, records, reserve));
    }

    /// **A refusal is solved, not stepped.** Given a slab that overran by the page-table overhead of what it did allocate, the retry lands
    /// on the largest slab that fits — and it is always strictly smaller, so the loop in [`build_limited`] cannot spin.
    #[test]
    fn a_refused_slab_is_solved_and_always_shrinks() {
        let record = 7_721_984u64;
        let floor = 1u64 << 29;
        let budget = 100_000_000_000u64;
        let refusal = |avail: u64, allocated: u64| load::RegionRefusal {
            id: 72,
            kind: load::RegionKind::Experts(35),
            bytes: 2_383_200_000,
            avail,
            floor,
            allocated: 0,
            regions: 82,
            shortfall: (2_383_200_000u64 + floor).saturating_sub(avail),
            allocated_bytes: allocated,
            total: 109_820_000_000,
        };
        // 38 layers allocated nominally 92 GiB, and the box shows 2.43 GiB left: the real ratio is ~0.7%, so the next slab is
        // `budget / 1.007` — not the slab that failed, and not a sixteenth less
        let allocated = 92u64 << 30;
        let avail = floor + (budget - allocated) - (allocated / 143); // ~0.7% of 92 GiB already spent on page tables
        let next = slab_after_refusal(budget, floor, &refusal(avail, allocated), record);
        let r = (allocated / 143) as f64 / allocated as f64;
        let want = (budget as f64 / (1.0 + r)) as u64;
        assert!(next < budget - floor, "{next} is not below the budget");
        assert!((next as i128 - (want - record) as i128).abs() <= record as i128, "{next} is not {want}");
        assert!((next as f64) * (1.0 + r) <= budget as f64, "the retry does not fit: {next} against {budget}");
        // zero allocated: the first region did not fit, and the step is what is left
        assert_eq!(slab_after_refusal(budget, floor, &refusal(floor + (10u64 << 30), 0), record), (10u64 << 30) - record);
        // a shortfall that solves to nothing still shrinks rather than sticking
        assert!(slab_after_refusal(budget, floor, &refusal(floor, 1), record) < budget);
    }
}

#[cfg(test)]
mod engram_loc_tests {
    use super::*;
    use gb_exl3::checkpoint::scan_indexed;
    use std::io::Write as _;
    use std::path::{Path, PathBuf};

    /// A minimal indexed checkpoint: one shard holding `names` as [8, 256] F8_E4M3 tensors and an
    /// index pointing at it — enough for `scan_indexed` and `file_offset` to run for real.
    fn write_mini_ckpt(dir: &Path, names: &[&str]) {
        std::fs::create_dir_all(dir).unwrap();
        let mut tensors = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for n in names {
            let start = data.len() as u64;
            data.extend_from_slice(&[7u8; 2048]);
            tensors.insert(
                n.to_string(),
                serde_json::json!({"dtype": "F8_E4M3", "shape": [8, 256], "data_offsets": [start, start + 2048]}),
            );
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(tensors)).unwrap();
        let shard = dir.join("shard-00001.safetensors");
        let mut f = std::fs::File::create(&shard).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(&data).unwrap();
        let wm: serde_json::Map<String, serde_json::Value> =
            names.iter().map(|n| (n.to_string(), serde_json::json!("shard-00001.safetensors"))).collect();
        let idx = serde_json::json!({"metadata": {"total_size": data.len()}, "weight_map": wm});
        std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&idx).unwrap()).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("dsv41-eng-{}-{name}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        d
    }

    /// A tensor the pack itself carries resolves from the pack root, sidecar or not, with a real
    /// file offset into the pack's own shard.
    #[test]
    fn engram_entry_prefers_the_pack() {
        let root = tmp("prefers");
        write_mini_ckpt(&root.join("pack"), &["layers.1.engram.embed.weight"]);
        write_mini_ckpt(&root.join("sidecar"), &["layers.1.engram.embed.weight"]);
        let pack = scan_indexed(&root.join("pack"), false).unwrap();
        let sidecar = scan_indexed(&root.join("sidecar"), false).unwrap();
        let (owner, e) = engram_entry("layers.1.engram.embed.weight", &pack, Some(&sidecar)).expect("resolves");
        assert!(owner.root.ends_with("pack"), "owner is the pack: {}", owner.root.display());
        let off = owner.file_offset(e).expect("real header offset");
        assert!(off >= 8, "offset is absolute: {off}");
        std::fs::remove_dir_all(&root).ok();
    }

    /// A tensor only the sidecar carries resolves from the sidecar root — the quantized-pack case.
    #[test]
    fn engram_entry_falls_back_to_the_sidecar() {
        let root = tmp("fallback");
        write_mini_ckpt(&root.join("pack"), &["layers.1.engram.q_weight"]);
        write_mini_ckpt(&root.join("sidecar"), &["layers.1.engram.embed.weight"]);
        let pack = scan_indexed(&root.join("pack"), false).unwrap();
        let sidecar = scan_indexed(&root.join("sidecar"), false).unwrap();
        let (owner, e) = engram_entry("layers.1.engram.embed.weight", &pack, Some(&sidecar)).expect("resolves");
        assert!(owner.root.ends_with("sidecar"), "owner is the sidecar: {}", owner.root.display());
        let off = owner.file_offset(e).expect("real header offset");
        assert!(off >= 8, "offset is absolute: {off}");
        std::fs::remove_dir_all(&root).ok();
    }

    /// Neither source has it: the error names the tensor AND the env var, fail-closed.
    #[test]
    fn engram_entry_names_both_sources_when_missing() {
        let root = tmp("missing");
        write_mini_ckpt(&root.join("pack"), &["layers.1.engram.q_weight"]);
        let pack = scan_indexed(&root.join("pack"), false).unwrap();
        let err = engram_entry("layers.1.engram.embed.weight", &pack, None).expect_err("neither source has it");
        let msg = err.to_string();
        assert!(msg.contains("layers.1.engram.embed.weight"), "names the tensor: {msg}");
        assert!(msg.contains("REXL3_DSV41_ENGRAM_SHARDS"), "names the env var: {msg}");
        std::fs::remove_dir_all(&root).ok();
    }

    /// One GB10 at the full window with vision leaves admission about 0 bytes free. `--prefix-cache on` must still run a cache there
    /// (measured: `rexl3_prefix_cache_enabled` was 0 and no prompt was ever reused).
    #[test]
    fn a_required_prefix_cache_is_not_switched_off_for_lack_of_free_memory() {
        use crate::serve::PrefixCacheOption as P;
        assert!(!prefix_cache_enabled(P::Off, u64::MAX), "off stays off");
        assert!(!prefix_cache_enabled(P::Mib(0), u64::MAX), "zero MiB is off");
        assert!(prefix_cache_enabled(P::On, 0), "on with nothing free still runs a cache");
        assert!(prefix_cache_enabled(P::Mib(1), 0), "an explicit size still runs a cache");
        assert!(!prefix_cache_enabled(P::Auto, 0), "the default may decline when little is free");
        assert!(!prefix_cache_enabled(P::Auto, (2 << 30) - 1));
        assert!(prefix_cache_enabled(P::Auto, 2 << 30));
    }
}
