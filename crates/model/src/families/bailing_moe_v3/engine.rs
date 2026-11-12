//! Building a ready-to-serve Bailing-3.1 engine from a model folder: map the pack, plan memory, load the weights (with the
//! embedding as an mmap sidecar), build the device image, allocate the state and scratch, and wire the backend into the engine.

use super::alloc::AllocPlan;
use super::backend::BailingBackend;
use super::config::BailingConfig;
use super::device::BailingDevice;
use super::inventory::PackInventory;
use super::load::{load_pack, BailingPack};
use super::memory;
use super::model::BailingModel;
use super::spill::{SpillConfig, SpillStore};
use crate::error::ModelError;
use crate::hostmem;
use crate::pack::PackError;
use crate::plan::{CacheRequest, MemoryPlan};
use crate::serve::{gib, memory_options, ServeOptions};
use gb_engine::{boot, Drafter, Engine, EngineConfig, NoDrafts};
use gb_sched::{CostModel, MemoryProfile, ModelMemory};

/// The Bailing engine, before its type is erased.
pub type BailingEngine = Engine<BailingBackend>;

fn cost_model(with_mtp: bool) -> CostModel {
    let rows = super::alloc::WINDOW_ROWS;
    if with_mtp {
        CostModel::from_table(vec![(1, 1.0), (rows, 1.0)])
    } else {
        CostModel::from_table(vec![(1, 1.0), (rows, 1.0)])
    }
}

pub fn check_options(o: &ServeOptions) -> Result<(), ModelError> {
    if o.vision.is_some() {
        return Err(ModelError::Config("bailing_moe_v3 has no vision tower".into()));
    }
    Ok(())
}

pub fn build_engine(o: &ServeOptions) -> Result<BailingEngine, ModelError> {
    gb_engine::boot::checkpoint();
    check_options(o)?;
    let c = super::contract::check_selected().map_err(ModelError::Config)?;
    super::contract::check_built(c).map_err(ModelError::Config)?;
    boot::fact("ling_contract", c.id().to_string());
    let cfg = BailingConfig::from_dir(&o.model_dir).map_err(ModelError::Config)?;
    let with_mtp = o.drafting && o.spec.uses_mtp() && cfg.has_mtp();

    let ck = gb_exl3::checkpoint::scan(&o.model_dir, false).map_err(|e| ModelError::Pack(PackError::Io(e.to_string())))?;
    let inv = PackInventory::from_checkpoint(&ck).map_err(|e| ModelError::Config(format!("pack inventory: {e}")))?;
    let pack = BailingPack::from_inventory(&inv, cfg.clone()).map_err(|e| ModelError::Config(e.to_string()))?;
    boot::fact("target_layers", cfg.layers.len().to_string());
    boot::fact("target_fingerprint", crate::fingerprint::fingerprint(&ck));

    let weights_bytes = pack.plan.device_bytes();
    // What is left once the weights are resident, sized the way every other family sizes its plan (`hostmem::planning_free`: the host's
    // MemAvailable rounded down to a whole GiB, so two boots of one box in one state plan one cache), less the unpriced boot overhead.
    let (cuda_free, _) = gb_cuda::rt::mem_info().unwrap_or((0, 0));
    let mut host = hostmem::MemInfo::read();
    eprintln!("{}", hostmem::memory_view_line(cuda_free as u64, host.as_ref(), o.floor_bytes));
    let avail_at_plan = host.map(|h| h.available);
    if let Some(h) = host.as_mut() {
        h.available = h.available.saturating_sub(weights_bytes);
    }
    let cuda_free = (cuda_free as u64).saturating_sub(weights_bytes);
    let free = hostmem::planning_free(cuda_free, host.as_ref()).saturating_sub(BOOT_OVERHEAD_BYTES);
    let requested_slots = o.slots.max(1);
    let in_place = super::load::embed_in_place();
    let kv_tokens_cap = o.kv_tokens.unwrap_or(o.max_model_len) as u64;
    let floor_bytes = family_floor_bytes(o.floor_bytes);
    let snapshot_bytes = super::backend::snapshot_fixed_bytes(&cfg, with_mtp);
    // The spill store (NVMe prefix cache): probed before the plan so the planner sizes the cache from the disk quota
    // (`--prefix-cache-mib` when given, else `min(free_disk / 4, 64 GiB)`) and prices only its staging buffer in RAM.
    // None keeps today's host-RAM cache (no writable directory, or the cache off).
    let cache_request = o.prefix_cache.request();
    let spill = if !matches!(cache_request, CacheRequest::Off) {
        let ns = super::backend::prefix_namespace(&crate::fingerprint::fingerprint(&ck), with_mtp);
        let explicit = match cache_request {
            CacheRequest::Fixed(bytes) => Some(bytes >> 20),
            _ => None,
        };
        let s = super::spill::SpillConfig::probe(&format!("{ns:016x}"), explicit);
        if s.is_none() {
            eprintln!("rexl3 level=info event=ling31_spill_backend=ram reason=\"no writable cache directory ($XDG_CACHE_HOME/rexl3/prefix)\"");
        }
        s
    } else {
        None
    };
    let window_tokens = o.max_model_len as u64;
    let base = memory_options(o, with_mtp);
    let at = |rows: usize| profile_at(&cfg, &base, rows);
    let inputs = PlanInputs {
        cfg: &cfg,
        base,
        profile_at: &at,
        free_after_weights: free,
        floor_bytes,
        slots: requested_slots,
        kv_tokens_cap,
        cache: cache_request,
        snapshot_bytes,
        window_tokens,
        in_place,
        with_mtp,
        spill: spill.clone(),
    };
    let planned = plan_boot(&inputs).map_err(|e| ModelError::Config(e.to_string()))?;
    for r in &planned.rejected {
        eprintln!("{}", r.line());
    }
    let Chosen { chunk, mem_opts, profile, plan, free, reserve } = planned.chosen;
    let embed = if in_place { "shard" } else { "device" };
    boot::fact("plan_free_gib", format!("{:.1}", gib(free)));
    boot::fact("plan_post_alloc_reserve_gib", format!("{:.1}", gib(reserve)));
    boot::fact("plan_prefill_chunk", chunk.to_string());
    boot::fact("plan_kv_pool_tokens", plan.kv_tokens.to_string());
    boot::fact("embedding_table", embed.to_string());
    let reach_planned = if plan.cache.enabled() { reach_tokens(&plan.cache, window_tokens) } else { 0 };
    eprintln!(
        "rexl3 level=info event=ling31_plan weights_gib={:.1} free_gib={:.1} floor_gib={:.1} post_alloc_reserve_gib={:.1} embed={embed} \
         chunk={chunk} requested_chunk={} window={window_tokens} kv_pool_tokens={} scratch_gib={:.2} engine_reserve_gib={:.2} entries={} reach_tokens={reach_planned}",
        gib(weights_bytes),
        gib(free),
        gib(floor_bytes),
        gib(reserve),
        planned.requested_chunk,
        plan.kv_tokens,
        gib(profile.scratch_bytes()),
        gib(profile.engine_reserve_bytes()),
        plan.cache.snapshot_slots,
    );
    let slots = requested_slots;

    gb_exl3::loadstats::set_component(gb_exl3::loadstats::Component::Target);
    boot::mark("target_loading", None);
    let read0 = gb_cuda::loader::totals();
    let phase = gb_exl3::loadstats::phase_start(gb_exl3::loadstats::Component::Target);
    let avail_before_load = hostmem::MemInfo::read().map(|h| h.available);
    let loaded = load_pack(&o.model_dir, &pack, &mut |_, _| {})?;
    let avail_after_load = hostmem::MemInfo::read().map(|h| h.available);
    let load_s = gb_exl3::loadstats::phase_end(phase);
    boot::mark("target_loaded", Some(gib(loaded.regions.bytes())));
    match &loaded.report {
        Some(r) => {
            let t = gb_cuda::loader::totals().since(&read0);
            eprintln!(
                "rexl3 event=ling31_load regions={} gib={:.2} load_s={load_s:.2} workers={} spans={} direct_spans={} cache={} read_gib={:.2} cold_gib={:.2}",
                loaded.regions.bufs.len(),
                gib(loaded.regions.bytes()),
                r.workers,
                r.spans,
                r.direct_spans,
                t.cache_state(),
                gib(t.bytes),
                gib(t.cold_bytes)
            );
        }
        None => eprintln!(
            "rexl3 event=ling31_load regions={} gib={:.2} load_s={load_s:.2} mode=serial",
            loaded.regions.bufs.len(),
            gib(loaded.regions.bytes())
        ),
    }
    let dev = BailingDevice::build(&cfg, &pack.weights, loaded.regions).map_err(ModelError::Config)?;
    // The KV pool is the plan's `kv_blocks` and no more: a snapshot copies its prefix's MLA pages to host RAM (`MlaCacheSnap`) and
    // hands the engine no pages (`Captured { pages: [] }`), and a restore copies them back into pages of the live pool. So the cache's
    // `page_blocks` are not device pages here but the host budget of those copies: one plan page is exactly one 256-token copy (the
    // test `one_plan_page_is_one_mla_copy_of_256_tokens`). Adding them to the pool, as before, allocated device pages nothing used.
    let kv_tokens = plan.kv_blocks * 256;
    let model = BailingModel::new(cfg.clone(), dev, loaded.embedding, slots, kv_tokens, with_mtp, mem_opts.prefill_rows)?;
    boot::fact("prefill_rows", model.prefill_rows.to_string());
    let avail_after_model = hostmem::MemInfo::read().map(|h| h.available);

    let mut backend = BailingBackend::new(model, with_mtp);
    if let Some(cfg) = &spill {
        match SpillStore::open(cfg.clone()) {
            Ok(store) => {
                backend.model.set_spill_store(Some(store));
                boot::fact("prefix_cache_backend", "disk");
            }
            Err(e) => eprintln!("rexl3 level=warn event=ling31_spill_backend=ram reason={e:?}"),
        }
    }
    if plan.cache.enabled() {
        let ns = super::backend::prefix_namespace(&crate::fingerprint::fingerprint(&ck), with_mtp);
        backend = backend.with_prefix_cache(ns, plan.cache).map_err(|e| ModelError::Config(format!("the prefix cache: {e}")))?;
        let reach = reach_tokens(&plan.cache, window_tokens);
        let where_ = if spill.is_some() { "disk" } else { "host" };
        let disk_quota_gib = spill.as_ref().map_or(0.0, |s| gib(s.quota_bytes));
        eprintln!(
            "rexl3 event=ling31_prefix_plan entries={} pages={} gib={:.2} reach_tokens={reach} window={window_tokens} where={where_} disk_quota_gib={disk_quota_gib:.2}",
            plan.cache.snapshot_slots,
            plan.cache.page_blocks,
            gib(plan.cache.bytes()),
        );
        if reach < window_tokens {
            eprintln!(
                "rexl3 level=warn event=ling31_prefix_reach reach_tokens={reach} window={window_tokens} note=\"a prompt longer than the reach is never cached; a smaller --prefill-chunk or --max-model-len raises it\""
            );
        }
        boot::fact("prefix_cache_reach_tokens", reach.to_string());
    } else {
        // `plan_boot` refuses a requested cache that plans no entries, so only an explicit `off` reaches here
        eprintln!("rexl3 event=ling31_prefix_plan entries=0 reason=off");
    }
    // What the boot really took against what the plan priced: `MemAvailable` before and after each allocating step. The weights and the
    // plan's other bytes are priced exactly (device regions, the AllocPlan items); each stage's `*_drop_gib` less its `*_planned_gib` is the
    // boot overhead the plan does not price (`BOOT_OVERHEAD_BYTES`), so a drift shows up in this line instead of as a node left short of its floor.
    let delta = |a: Option<u64>, b: Option<u64>| match (a, b) {
        (Some(a), Some(b)) => format!("{:.2}", (a as f64 - b as f64) / (1u64 << 30) as f64),
        _ => "n/a".to_string(),
    };
    eprintln!(
        "rexl3 level=info event=ling31_spend plan_avail_gib={} weights_planned_gib={:.2} weights_drop_gib={} model_planned_gib={:.2} model_drop_gib={} after_plan_avail_gib={} floor_gib={:.1}",
        avail_at_plan.map_or("n/a".to_string(), |a| format!("{:.2}", gib(a))),
        gib(weights_bytes),
        delta(avail_before_load, avail_after_load),
        gib(plan.planned_bytes(slots) - plan.cache.bytes() + reserve),
        delta(avail_after_load, avail_after_model),
        hostmem::MemInfo::read().map_or("n/a".to_string(), |h| format!("{:.2}", gib(h.available))),
        gib(floor_bytes),
    );
    let mm = ModelMemory::from_profile(&profile, 0, 0);
    let admission = plan.admission(mm, slots, o.max_model_len).map_err(|e| ModelError::Config(e.to_string()))?;

    let make: Box<dyn Fn() -> Box<dyn Drafter>> = Box::new(|| Box::new(NoDrafts) as Box<dyn Drafter>);
    // Multi-draft MTP+PLD default (tuned on GB10): 3 drafts max (d0 + up to 2 chained).
    let max_drafts = 3usize;
    let engine = Engine::new(
        backend,
        admission,
        EngineConfig { max_rows: super::alloc::WINDOW_ROWS, max_drafts, cost: cost_model(with_mtp), drafting: with_mtp },
        make,
    );
    if plan.cache.enabled() {
        // the engine's prefix machinery (lookups, cuts, index) is independent of the backend's snapshot store; the
        // interactive cut policy is the one that serves a second question about a long document (as the other families),
        // its cuts on the KDA chunk grid (`prefix_config`)
        Ok(engine.with_prefix_cache(prefix_config()?))
    } else {
        Ok(engine)
    }
}

/// The prefix-cache cut grid: `kda::window_chunked` (the prefill's KDA recurrence) steps 16 rows at a time from each call's first row, and
/// its bits depend on where those 16-row chunks start, so a prefix that resumes at a cut gives a cold prefill's bits only when the cut
/// lies on the cold prefill's grid (a multiple of 16 from the prompt's first row; the prefill chunks are multiples of 16 too,
/// `model::PREFILL_ROWS`).
pub const CUT_GRID: usize = gb_cuda::kda::P2_CHUNK_ROWS;

/// The engine's prefix-cache configuration for this family: the interactive cut policy with the environment's overrides
/// (`serve::interactive_prefix_config_from_env`), every planned cut moved down to the [`CUT_GRID`] (`PrefixConfig::cut_align`). A hit
/// is always a cut, so every restored prefix ends on the grid too and every segment a prefill resumes starts on it.
pub fn prefix_config() -> Result<gb_engine::PrefixConfig, ModelError> {
    Ok(gb_engine::PrefixConfig { cut_align: CUT_GRID, ..crate::serve::interactive_prefix_config_from_env()? })
}

/// Device bytes allocated AFTER the plan (by `BailingDevice::build` and `BailingModel::new`) that the family's memory profile does not
/// count, so the plan takes them out of the free bytes first: `AllocPlan::post_alloc_items` (the bf16 embedding table when it is copied
/// to the device, nothing when the kernels read it in place from its shard, `in_place`: see `load::embed_in_place`; the MTP tap rings
/// and the restricted draft head; the RoPE tables and the block table, sized by the KV pool; the KDA low-rank stand-ins; the experts'
/// pointer tables). The pool is priced at the most pages the plan can give it, `kv_tokens_cap` tokens and a page per slot
/// (`plan::plan_memory_cached_windowed`), so the reserve is exact whenever the window caps the pool and never short when memory does.
pub fn post_alloc_reserve_bytes(cfg: &BailingConfig, slots: usize, with_mtp: bool, in_place: bool, kv_tokens_cap: u64) -> u64 {
    let pool_pages = (kv_tokens_cap as usize).div_ceil(crate::kvtables::BLOCK_TOKENS) + slots;
    AllocPlan::post_alloc_items(cfg, slots, pool_pages, with_mtp, in_place).iter().map(|i| i.bytes).sum()
}

/// This family's floor: 1.0 GiB when `floor_bytes` is still the 8 GiB Qwen/GLM default (the 106 GiB pack leaves ~6-10 GiB on a single
/// 121 GiB GB10, and the post-plan device allocations are already priced by [`post_alloc_reserve_bytes`]; the 8 GiB default would
/// crowd out the prefix cache), what was asked for otherwise (`REXL3_MEMORY_FLOOR_GIB`).
pub fn family_floor_bytes(floor_bytes: u64) -> u64 {
    if floor_bytes == (8 << 30) {
        1 << 30
    } else {
        floor_bytes
    }
}

/// The plan `build_engine` boots with, and the free bytes it was made from. `free_after_weights`: what is free once the weights are
/// resident (`MemAvailable` less the plan's device regions); `reserve`: [`post_alloc_reserve_bytes`], taken out before planning. The
/// prefix cache is sized from the window as Qwen3.8, GLM and Flash-Next size theirs (`plan_memory_cached_windowed`,
/// `docs/PREFIX_CACHE_SIZING.md` section 5): its pages first, enough for one `window_tokens` prefix, then its snapshot slots
/// (`snapshot_bytes`: [`super::backend::snapshot_fixed_bytes`]). Both live in host RAM for this family (see `build_engine`).
#[allow(clippy::too_many_arguments)]
pub fn plan_after_weights(
    profile: &gb_sched::MemoryProfile,
    free_after_weights: u64,
    reserve: u64,
    floor_bytes: u64,
    slots: usize,
    kv_tokens_cap: u64,
    cache: crate::plan::CacheRequest,
    snapshot_bytes: u64,
    window_tokens: u64,
) -> Result<(crate::plan::MemoryPlan, u64), crate::plan::PlanError> {
    let free = free_after_weights.saturating_sub(reserve);
    let plan = crate::plan::plan_memory_cached_windowed(
        profile,
        free,
        floor_bytes,
        slots,
        kv_tokens_cap,
        0,
        cache,
        snapshot_bytes,
        0,
        window_tokens,
    )?;
    Ok((plan, free))
}

/// Host memory the boot takes that no item of the plan prices: what `MemAvailable` falls by beyond the device regions and the `AllocPlan` items
/// (the loader's `cudaMalloc` bytes equal the plan's, `rexl3_load_stage_bytes{stage="malloc"}`). It is the host cost of mapping 108 GiB of
/// device allocations on a unified-memory box (the driver's page-table and metadata slabs), the process itself (`AnonPages` 1.2 GiB) and the
/// page cache the kernel does not count as available. Measured as `weights_drop_gib - weights_planned_gib` plus `model_drop_gib -
/// model_planned_gib` of the `event=ling31_spend` line on N2, 2026-11-05, five boots of chunks 128 to 2,048 and windows 16,384 to 65,536:
/// 1.08, 1.44, 1.48, 1.81, 1.99 GiB (the node's page cache at the plan moves it by half a GiB). 2 GiB, the top of that range, is taken out of the
/// free bytes before every plan, so a boot that fits on paper leaves the floor it promised; before this the plan promised a 3 GiB floor and the
/// node was left with 2.3 to 2.9 GiB at ready.
pub const BOOT_OVERHEAD_BYTES: u64 = 2048 << 20;

/// The prefill chunks the planner chooses between, widest first. The boot never plans a chunk wider than `--prefill-chunk` asked for (the
/// default 4096 is a ceiling), and never automatically selects one below 1,024. That is a memory-planning floor, not an
/// exactness limit: the fp32 router (#253, 47a29e7a) made prefill chunking row-invariant down to the shipped 512-row setting.
/// An operator who asks for 512 gets 512 and nothing narrower.
pub const PLAN_CHUNKS: [usize; 3] = [4096, 2048, 1024];

/// The chunks to try for a request of `requested` rows, widest first: the request itself (clamped and snapped as the model does), then every
/// [`PLAN_CHUNKS`] entry narrower than it.
pub fn chunk_candidates(requested: usize) -> Vec<usize> {
    let ceiling = super::alloc::prefill_chunk_for(requested);
    let mut v = vec![ceiling];
    v.extend(PLAN_CHUNKS.iter().copied().filter(|&c| c < ceiling));
    v
}

/// Everything a memory plan of this family depends on besides the chunk.
pub struct PlanInputs<'a> {
    pub cfg: &'a BailingConfig,
    /// The boot's memory options; `prefill_rows` is the requested chunk (the ceiling).
    pub base: crate::memory::MemoryOptions,
    /// The memory options and profile for a prefill chunk of `rows` ([`profile_at`] for a real boot; a test hands in the measured figures,
    /// which the kernels' workspace sizes make unavailable to a build without CUDA).
    pub profile_at: &'a dyn Fn(usize) -> Result<(crate::memory::MemoryOptions, MemoryProfile), ModelError>,
    /// What is free once the weights are resident (`hostmem::planning_free` less [`BOOT_OVERHEAD_BYTES`]).
    pub free_after_weights: u64,
    pub floor_bytes: u64,
    pub slots: usize,
    /// The KV pool the operator asked for, in tokens (`--kv-tokens`, else the window).
    pub kv_tokens_cap: u64,
    pub cache: CacheRequest,
    pub snapshot_bytes: u64,
    /// The NVMe spill store configuration when one was probed at boot (`None`: the host-RAM cache).
    pub spill: Option<SpillConfig>,
    pub window_tokens: u64,
    pub in_place: bool,
    pub with_mtp: bool,
}

/// The plan a boot runs with and the memory profile it was cut from.
pub struct Chosen {
    pub chunk: usize,
    pub mem_opts: crate::memory::MemoryOptions,
    pub profile: MemoryProfile,
    pub plan: MemoryPlan,
    /// The free bytes the plan was made from (after the after-plan reserve).
    pub free: u64,
    pub reserve: u64,
}

/// A chunk the planner tried and refused, with the figures that decided it.
#[derive(Clone, Debug, PartialEq)]
pub struct Rejected {
    pub chunk: usize,
    pub reason: String,
    pub kv_pool_tokens: u64,
    pub scratch_bytes: u64,
    pub entries: usize,
    /// The chunk's plan fits; the boot took another ([`pick_by_reach`]).
    pub fits: bool,
}

impl Rejected {
    /// The log line of the refused chunk.
    pub fn line(&self) -> String {
        format!(
            "rexl3 level=info event=ling31_plan_try chunk={} fits={} kv_pool_tokens={} scratch_gib={:.2} entries={} reason=\"{}\"",
            self.chunk,
            u8::from(self.fits),
            self.kv_pool_tokens,
            gib(self.scratch_bytes),
            self.entries,
            self.reason
        )
    }
}

/// The result of planning a boot.
pub struct Planned {
    pub chosen: Chosen,
    /// The wider chunks that were tried first and did not fit, widest first.
    pub rejected: Vec<Rejected>,
    /// The chunk the options asked for (the ceiling).
    pub requested_chunk: usize,
}

/// A boot no chunk can hold: every refusal, in the order tried.
#[derive(Debug)]
pub struct PlanRefused {
    pub free: u64,
    pub floor: u64,
    pub window: u64,
    pub cache_requested: bool,
    pub tried: Vec<Rejected>,
    /// The longest pool the narrowest chunk could hold, when the pool (not the cache) was what failed.
    pub pool_holds: Option<u64>,
}

impl std::fmt::Display for PlanRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the Ling memory plan cannot hold this configuration on this box: free_gib={:.1} after the weights and {:.2} GiB of boot overhead, floor_gib={:.1}, window={} tokens{}. Tried ",
            gib(self.free),
            gib(BOOT_OVERHEAD_BYTES),
            gib(self.floor),
            self.window,
            if self.cache_requested { ", prefix cache requested" } else { "" }
        )?;
        for (i, r) in self.tried.iter().enumerate() {
            write!(
                f,
                "{}chunk {} (kv_pool_tokens={}, scratch_gib={:.2}, entries={}): {}",
                if i == 0 { "" } else { "; " },
                r.chunk,
                r.kv_pool_tokens,
                gib(r.scratch_bytes),
                r.entries,
                r.reason
            )?;
        }
        match self.pool_holds {
            Some(t) => write!(
                f,
                ". The KV pool holds at most {t} tokens here: pass --max-model-len {} or free host memory",
                t / 1024 * 1024
            ),
            None => write!(f, ". Free host memory, lower --max-model-len, or pass --prefix-cache off"),
        }
    }
}

impl std::error::Error for PlanRefused {}

/// The spare bytes after the KV pool a prefix cache of two entries needs: the smallest `spare` for which `spare - cache_margin_bytes(spare)`
/// reaches the least cache `fit_pages_first` enables (two snapshot slots and the pages of `MIN_CACHE_REACH_TOKENS` tokens).
pub fn spare_needed_for_two_entries(snapshot_bytes: u64, page_bytes: u64) -> u64 {
    let least = gb_sched::prefix::CacheBudget::pages_first_minimum_bytes(snapshot_bytes, page_bytes);
    // margin = min(CACHE_MARGIN_BYTES, spare / 4): a quarter of the spare while the spare is under four margins, a whole margin above
    let by_quarter = (least * 4).div_ceil(3);
    if by_quarter <= 4 * gb_sched::prefix::CACHE_MARGIN_BYTES {
        by_quarter
    } else {
        least + gb_sched::prefix::CACHE_MARGIN_BYTES
    }
}

/// The cache budget when the spill store is active: sized from the disk quota instead of the RAM spare, with the RAM cost of
/// the cache pinned to the staging buffer (the plan's byte accounting counts slots x staging; the pages cost nothing in RAM).
/// The index limits (slots and pages) keep their reach meaning; `page_bytes` is zeroed only so `bytes()` prices the staging.
pub fn spill_cache_budget(
    spill: &SpillConfig,
    page_bytes: u64,
    window_tokens: u64,
    live_slots: usize,
) -> gb_sched::prefix::CacheBudget {
    let rule = gb_sched::prefix::WindowRule { window_tokens, live_slots };
    let mut b = gb_sched::prefix::CacheBudget::fit_pages_first(
        spill.quota_bytes,
        spill.staging_bytes as u64,
        page_bytes.max(1),
        rule,
        true,
    );
    // a disk entry costs a full-window copy (`full_entry_pages` pages plus the staging); cap the slots to what the quota
    // really holds so the plan's RAM accounting (slots x staging) stays the honest cost of the cache
    let entry_cost = rule.full_entry_pages().saturating_mul(page_bytes.max(1)).saturating_add(spill.staging_bytes as u64);
    let max_entries = (spill.quota_bytes / entry_cost.max(1)).max(gb_sched::prefix::MIN_CACHE_SNAPSHOT_SLOTS) as usize;
    b.snapshot_slots = b.snapshot_slots.min(max_entries);
    // the staging buffer is ONE RAM cost shared by every entry: each slot prices its share, so `bytes()` stays the honest
    // cost of the cache (the staging buffer) however many disk entries the quota allows
    b.snapshot_bytes = (spill.staging_bytes as u64 / b.snapshot_slots.max(1) as u64).max(1);
    b.page_bytes = 0;
    b
}

/// Which of the plans that fit to boot with, `reach` being the longest prefix each one's cache holds (`window` when the cache is off): the
/// widest chunk whose cache reaches the whole window, else the plan that reaches furthest (the wider chunk on a tie). A plan that fits is
/// not enough: at 131,072 tokens of window a 2,048-row chunk plans two entries that reach 45,056 tokens, a 1,024-row chunk 85,504, and an entry
/// that cannot hold the prompt it was cut for is no cache for it. `fits` is in the order tried, widest first; `None` when nothing fits.
pub fn pick_by_reach(reach: &[u64], window: u64) -> Option<usize> {
    reach
        .iter()
        .position(|&r| r >= window)
        .or_else(|| reach.iter().enumerate().max_by_key(|&(i, &r)| (r, std::cmp::Reverse(i))).map(|(i, _)| i))
}

/// The memory options and profile of this family at a prefill chunk of `rows`: what `build_engine` plans and boots with.
pub fn profile_at(
    cfg: &BailingConfig,
    base: &crate::memory::MemoryOptions,
    rows: usize,
) -> Result<(crate::memory::MemoryOptions, MemoryProfile), ModelError> {
    let mem_opts = memory::engine_options(&crate::memory::MemoryOptions { prefill_rows: rows, ..*base });
    let profile = memory::memory_profile(cfg, &mem_opts).map_err(|e| ModelError::Config(e.to_string()))?;
    Ok((mem_opts, profile))
}

/// Plan one chunk: the memory profile at that chunk, the plan, and the three conditions a boot needs of it. The KV pool must hold the pool
/// that was asked for (the window, unless `--kv-tokens` says otherwise): a pool short of the window would boot a window no full prompt can
/// use. A requested prefix cache must plan at least two entries. A plan the planner refuses is a refusal of the chunk, with its words.
fn judge(inp: &PlanInputs, chunk: usize) -> Result<Result<Chosen, Rejected>, ModelError> {
    let (mem_opts, profile) = (inp.profile_at)(chunk)?;
    let reserve = post_alloc_reserve_bytes(inp.cfg, inp.slots, inp.with_mtp, inp.in_place, inp.kv_tokens_cap);
    let scratch_bytes = profile.scratch_bytes();
    let refuse = |reason: String, kv_pool_tokens: u64, entries: usize| Rejected {
        chunk,
        reason,
        kv_pool_tokens,
        scratch_bytes,
        entries,
        fits: false,
    };
    let (mut plan, free) = match plan_after_weights(
        &profile,
        inp.free_after_weights,
        reserve,
        inp.floor_bytes,
        inp.slots,
        inp.kv_tokens_cap,
        inp.cache,
        inp.snapshot_bytes,
        inp.window_tokens,
    ) {
        Ok(p) => p,
        Err(e) => return Ok(Err(refuse(e.to_string(), 0, 0))),
    };
    // The spill store sizes the cache from the disk quota and prices only its staging buffer in RAM: replace the RAM-sized
    // budget (which at 262,144 tokens plans no entries) with the disk one; the pool, floor and reserves are untouched.
    if let Some(spill) = inp.spill.as_ref() {
        if !matches!(inp.cache, CacheRequest::Off) {
            let b = spill_cache_budget(spill, plan.block_bytes, inp.window_tokens, inp.slots);
            if b.enabled() {
                plan.cache = b;
            }
        }
    }
    if plan.kv_tokens < inp.kv_tokens_cap {
        let why = format!(
            "the KV pool holds {} tokens, below the {}-token window: a longer prompt could not be served",
            plan.kv_tokens, inp.kv_tokens_cap
        );
        return Ok(Err(refuse(why, plan.kv_tokens, plan.cache.snapshot_slots)));
    }
    if !matches!(inp.cache, CacheRequest::Off) && !plan.cache.enabled() {
        let need = spare_needed_for_two_entries(inp.snapshot_bytes, plan.block_bytes);
        let why = format!(
            "the prefix cache plans 0 entries: {:.2} GiB spare after the KV pool, two entries need {:.2} GiB (two {:.2} GiB snapshots, {} tokens of pages, the planner's margin)",
            gib(plan.spare_bytes),
            gib(need),
            gib(inp.snapshot_bytes),
            gb_sched::prefix::MIN_CACHE_REACH_TOKENS
        );
        return Ok(Err(refuse(why, plan.kv_tokens, 0)));
    }
    Ok(Ok(Chosen { chunk, mem_opts, profile, plan, free, reserve }))
}

/// The plan a boot runs with. Every chunk of [`chunk_candidates`] is planned (at most the requested one, at least 1,024 when the request is
/// wider); a chunk fits when its plan holds the whole window in the KV pool and, when a prefix cache is requested, at least two cache entries.
/// Of those that fit the boot takes the widest whose cache reaches the whole window, else the one that reaches furthest ([`pick_by_reach`]). When
/// none fits the error names each chunk and the exact figure that refused it.
pub fn plan_boot(inp: &PlanInputs) -> Result<Planned, PlanRefused> {
    let candidates = chunk_candidates(inp.base.prefill_rows);
    let requested_chunk = candidates[0];
    let refusal = |tried: Vec<Rejected>, pool_holds: Option<u64>, cache_requested: bool| PlanRefused {
        free: inp.free_after_weights,
        floor: inp.floor_bytes,
        window: inp.window_tokens,
        cache_requested,
        tried,
        pool_holds,
    };
    let mut fits: Vec<Chosen> = Vec::new();
    let mut rejected: Vec<Rejected> = Vec::new();
    for &c in &candidates {
        match judge(inp, c) {
            Ok(Ok(chosen)) => fits.push(chosen),
            Ok(Err(r)) => rejected.push(r),
            // the options themselves are inconsistent (no NextN layer for MTP): not a memory refusal
            Err(e) => {
                let r =
                    Rejected { chunk: c, reason: e.to_string(), kv_pool_tokens: 0, scratch_bytes: 0, entries: 0, fits: false };
                return Err(refusal(vec![r], None, false));
            }
        }
    }
    let reach: Vec<u64> = fits.iter().map(|c| reach_of(c, inp.window_tokens)).collect();
    let Some(at) = pick_by_reach(&reach, inp.window_tokens) else {
        let pool_holds = rejected.last().filter(|r| r.reason.starts_with("the KV pool holds")).map(|r| r.kv_pool_tokens);
        return Err(refusal(rejected, pool_holds, !matches!(inp.cache, CacheRequest::Off)));
    };
    let chosen = fits.remove(at);
    let chosen_reach = reach[at];
    // the chunks not taken, widest first: the ones that did not fit, and the ones that fit but whose cache reaches less than the chosen one's
    for f in &fits {
        let r = reach_of(f, inp.window_tokens);
        let reason = if r >= chosen_reach {
            format!("fits, but the wider chunk {} fits too and reaches {chosen_reach} tokens", chosen.chunk)
        } else {
            format!("fits, but its cache reaches {r} tokens against {chosen_reach} for chunk {}", chosen.chunk)
        };
        rejected.push(Rejected {
            chunk: f.chunk,
            reason,
            kv_pool_tokens: f.plan.kv_tokens,
            scratch_bytes: f.profile.scratch_bytes(),
            entries: f.plan.cache.snapshot_slots,
            fits: true,
        });
    }
    rejected.sort_by_key(|r| std::cmp::Reverse(r.chunk));
    Ok(Planned { chosen, rejected, requested_chunk })
}

fn reach_of(c: &Chosen, window: u64) -> u64 {
    if c.plan.cache.enabled() {
        reach_tokens(&c.plan.cache, window)
    } else {
        window
    }
}

/// The longest prefix the cache holds within its plan, as the boot reports it (`reach_tokens`, `/v1/models`): its pages' tokens, never
/// more than the window. The pages are the host budget of the snapshots' MLA copies, and pages past one window are room for the
/// second entry's copy, not a longer prefix.
pub fn reach_tokens(cache: &gb_sched::prefix::CacheBudget, window_tokens: u64) -> u64 {
    cache.reach_tokens().min(window_tokens)
}

/// The MTP mode the options allow (kept for the server's `--spec` validation path).
pub fn mtp_allowed(o: &ServeOptions) -> bool {
    matches!(o.spec, gb_spec::SpecMode::Auto | gb_spec::SpecMode::Mtp) && o.drafting
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = include_str!("../../../../../testdata/ling31-flash-1019/config.json");
    const GIB: u64 = 1 << 30;

    fn cfg() -> BailingConfig {
        BailingConfig::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    fn gib(b: u64) -> f64 {
        b as f64 / GIB as f64
    }

    #[test]
    fn the_reserve_leaves_out_the_embedding_table_read_in_place() {
        let c = cfg();
        let table = c.vocab as u64 * c.hidden as u64 * 2;
        assert_eq!(table, 1_287_651_328, "157,184 x 4,096 bf16: 1.20 GiB");
        let device = post_alloc_reserve_bytes(&c, 1, true, false, 32_768);
        let shard = post_alloc_reserve_bytes(&c, 1, true, true, 32_768);
        assert_eq!(device - shard, table);
        // without MTP and with the table in place: the pool's RoPE tables (129 pages of 256 positions, 32 cos and 32 sin f32 each) and
        // block-table row, the KDA low-rank stand-ins and the 69 MoE layers' expert tables
        let pool = 129 * 256 * 64 * 4 + 129 * 4;
        let fixed = 2 * 128 * 4096 * 2 + 2 * 8192 * 128 * 2 + 69 * (9 * 513 * 8 + 513 * 12);
        assert_eq!(post_alloc_reserve_bytes(&c, 1, false, true, 32_768), pool + fixed);
        // the MTP tap ring (64 MiB a slot) and the restricted draft head (~0.2 GiB) come on top, either way
        let two = post_alloc_reserve_bytes(&c, 2, true, true, 32_768);
        assert_eq!(
            two - shard,
            8192 * 4096 * 2 + 256 * 64 * 4 + (2 * 130 - 129) * 4,
            "one more slot: a tap ring and a pool page"
        );
        assert_eq!(format!("{:.1}", gib(device)), "1.5", "the stock-GB10 boot's post_alloc_reserve_gib=1.5");
        assert_eq!(format!("{:.1}", gib(shard)), "0.3");
    }

    #[test]
    fn the_boot_prices_every_allocation_of_the_alloc_plan() {
        // what `build_engine` takes out of the free memory for the profile (one slot, the 32,768-token window, MTP, the server's
        // default `--prefill-chunk`): the slot state, the pool's pages (the window and a page for the slot), the window records, the
        // scratch and the after-plan reserve are, together, every item of the backend's `AllocPlan` (`tests/ling31_alloc.rs` checks
        // the plan against the real allocations)
        let c = cfg();
        let o = crate::memory::MemoryOptions {
            prefill_rows: 4096,
            max_context: 32_768,
            mtp: true,
            vision: None,
            ..crate::memory::MemoryOptions::serving_defaults()
        };
        let profile = super::memory::memory_profile(&c, &super::memory::engine_options(&o)).unwrap();
        for in_place in [true, false] {
            let p = super::super::alloc::AllocParams {
                slots: 1,
                kv_blocks: 129,
                prefill_rows: super::super::alloc::prefill_chunk_for(o.prefill_rows),
                mtp: true,
                embed_in_place: in_place,
            };
            let booted = profile.state_bytes_per_slot()
                + 129 * profile.kv_page_bytes()
                + profile.window_record_bytes()
                + profile.scratch_bytes()
                + post_alloc_reserve_bytes(&c, 1, true, in_place, 32_768);
            assert_eq!(booted, AllocPlan::new(&c, &p).total(&p), "in place: {in_place}");
        }
    }

    #[test]
    fn the_cuts_snap_to_the_kda_chunk_grid() {
        let c = prefix_config().unwrap();
        assert_eq!(c.cut_align, 16);
        assert_eq!(super::super::model::PREFILL_ROWS % CUT_GRID, 0, "a prefill chunk is a whole number of KDA chunks");
        // the interactive policy, otherwise as every family serves it; its smallest cut is on the grid
        let base = crate::serve::interactive_prefix_config_from_env().unwrap();
        assert_eq!(c.policy, base.policy);
        assert_eq!(c.policy.min_cut % CUT_GRID, 0);
    }

    #[test]
    fn the_floor_is_the_one_the_profile_asks_for() {
        assert_eq!(family_floor_bytes(3 * GIB), 3 * GIB, "REXL3_MEMORY_FLOOR_GIB=3");
        assert_eq!(family_floor_bytes(8 << 30), GIB, "the 8 GiB server default becomes this family's 1 GiB");
    }

    // ---- the boot planner on the measured figures

    /// This family's memory model with the figures `memory_profile` gives on a GB10 build of `Ling-3.1-flash-1019` (a build without the
    /// kernels cannot size their workspaces, so the CPU gate plans with these; `the_stand_in_is_the_real_profile` checks them against the
    /// real one wherever the kernels are built). The scratch per prefill chunk is `AllocPlan::scratch_items` at 1,024 / 2,048 / 4,096 rows.
    struct Measured;

    impl Measured {
        const SCRATCH: [(u64, u64); 6] = [
            // #256 (the split-key decode attention) added `mla_split_workspace` (`SPLIT_WS_BYTES`, 42,106,880 B) to the
            // real profile's scratch; the stand-in carries it too so `the_stand_in_is_the_real_profile` holds without kernels.
            (128, 353_115_908),
            (512, 733_041_668),
            (1024, 1_239_610_116),
            (2048, 2_252_747_012),
            (4096, 4_279_020_804),
            (u64::MAX, 4_279_020_804),
        ];
    }

    impl gb_sched::MemoryModel for Measured {
        fn family(&self) -> &str {
            "bailing_moe_v3"
        }
        fn kv_layout(&self) -> gb_sched::KvLayout {
            gb_sched::KvLayout::new().with(gb_sched::KvLayerClass {
                name: "mla_gated",
                layers: 10,
                planes: vec![gb_sched::KvPlane::per_token("latent", 512, 2), gb_sched::KvPlane::per_token("rope", 64, 2)],
            })
        }
        fn recurrent_layers(&self) -> u64 {
            63
        }
        fn state_bytes_per_seq(&self) -> u64 {
            537_772_284
        }
        fn window_record_bytes_per_row(&self) -> u64 {
            662_667_264 / 64
        }
        fn scratch_bytes(&self, rows: u64) -> u64 {
            Self::SCRATCH.iter().find(|(r, _)| *r >= rows).map(|&(_, b)| b).unwrap()
        }
        fn activation_reserve(&self) -> u64 {
            memory::ACTIVATION_RESERVE
        }
        fn drafter_state_bytes_per_slot(&self) -> u64 {
            0
        }
        fn vision(&self) -> Option<gb_sched::VisionMemory> {
            None
        }
    }

    fn measured_at(rows: usize) -> Result<(crate::memory::MemoryOptions, MemoryProfile), ModelError> {
        let o =
            crate::memory::MemoryOptions { prefill_rows: rows, mtp: true, ..crate::memory::MemoryOptions::serving_defaults() };
        let shape = gb_sched::StepShape { page_tokens: 256, window_rows: 64, scratch_rows: (rows as u64).max(64) };
        Ok((o, MemoryProfile::new(&Measured, shape).map_err(|e| ModelError::Config(e.to_string()))?))
    }

    /// The boot of the shipped profile (one slot, MTP, `--prefix-cache on`, the 32 GiB... of the 106.3 GiB pack) on a node that has
    /// `after_weights_gib` GiB of `MemAvailable` left once the weights are resident, for a window and a requested chunk. Returns the plan.
    fn boot(after_weights_gib: f64, window: u64, requested_chunk: usize, cache: CacheRequest) -> Result<Planned, PlanRefused> {
        boot_at_floor(after_weights_gib, 3.0, window, requested_chunk, cache)
    }

    fn boot_at_floor(
        after_weights_gib: f64,
        floor_gib: f64,
        window: u64,
        requested_chunk: usize,
        cache: CacheRequest,
    ) -> Result<Planned, PlanRefused> {
        let c = cfg();
        let host = crate::hostmem::MemInfo { available: (after_weights_gib * GIB as f64) as u64, ..Default::default() };
        let free = crate::hostmem::planning_free(0, Some(&host)).saturating_sub(BOOT_OVERHEAD_BYTES);
        let base = crate::memory::MemoryOptions {
            prefill_rows: requested_chunk,
            mtp: true,
            ..crate::memory::MemoryOptions::serving_defaults()
        };
        plan_boot(&PlanInputs {
            cfg: &c,
            base,
            profile_at: &measured_at,
            free_after_weights: free,
            floor_bytes: family_floor_bytes((floor_gib * GIB as f64) as u64),
            slots: 1,
            kv_tokens_cap: window,
            cache,
            snapshot_bytes: super::super::backend::snapshot_fixed_bytes(&c, true),
            window_tokens: window,
            in_place: true,
            with_mtp: true,
            spill: None,
        })
    }

    const ON: CacheRequest = CacheRequest::Auto { cap_bytes: crate::plan::AUTO_CACHE_CAP_BYTES };

    #[test]
    fn the_chunk_candidates_are_the_request_then_the_narrower_planned_chunks() {
        assert_eq!(chunk_candidates(4096), vec![4096, 2048, 1024], "the shipped default is a ceiling");
        assert_eq!(chunk_candidates(0), vec![4096, 2048, 1024], "0 asks for the default");
        assert_eq!(chunk_candidates(8192), vec![4096, 2048, 1024], "a request past the widest chunk is that chunk");
        assert_eq!(chunk_candidates(2048), vec![2048, 1024]);
        assert_eq!(chunk_candidates(1024), vec![1024]);
        assert_eq!(chunk_candidates(512), vec![512], "narrower than the planned range: the operator's choice, never narrowed");
        assert_eq!(chunk_candidates(3000), vec![2992, 2048, 1024], "snapped to the KDA grid, then the narrower planned chunks");
    }

    #[test]
    fn the_boot_takes_the_widest_chunk_whose_cache_reaches_the_window_else_the_furthest_reach() {
        let w = 32_768;
        // widest first: the first whose reach covers the window
        assert_eq!(pick_by_reach(&[10_000, 32_768, 32_768], w), Some(1));
        assert_eq!(pick_by_reach(&[40_000, 32_768, 32_768], w), Some(0));
        // none covers it: the furthest, and the wider chunk on a tie
        assert_eq!(pick_by_reach(&[7_936, 20_000, 20_000], w), Some(1));
        assert_eq!(pick_by_reach(&[20_000, 20_000], w), Some(0));
        // one chunk that fits is the one that boots, however short its reach
        assert_eq!(pick_by_reach(&[4_096], w), Some(0));
        assert_eq!(pick_by_reach(&[], w), None, "nothing fits");
    }

    #[test]
    fn two_entries_need_exactly_the_spare_the_helper_names() {
        let c = cfg();
        let snap = super::super::backend::snapshot_fixed_bytes(&c, true);
        let page = 2_949_120u64;
        let need = spare_needed_for_two_entries(snap, page);
        let enabled = |spare: u64| {
            let budget = spare.saturating_sub(gb_sched::prefix::cache_margin_bytes(spare));
            gb_sched::prefix::CacheBudget::fit_pages_first(
                budget,
                snap,
                page,
                gb_sched::prefix::WindowRule { window_tokens: 32_768, live_slots: 1 },
                false,
            )
            .enabled()
        };
        assert!(enabled(need), "the named spare plans a cache: {:.3} GiB", gib(need));
        assert!(!enabled(need - (1 << 20)), "one MiB less does not");
        assert_eq!(format!("{:.2}", gib(need)), "1.56", "two 576.9 MiB snapshots and 4,096 tokens of pages, plus the margin");
    }

    #[test]
    fn the_stand_in_is_the_real_profile() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels");
            return;
        }
        let c = cfg();
        let base = crate::memory::MemoryOptions { mtp: true, ..crate::memory::MemoryOptions::serving_defaults() };
        for rows in [128, 512, 1024, 2048, 4096] {
            let (_, real) = profile_at(&c, &base, rows).unwrap();
            let (_, stand_in) = measured_at(rows).unwrap();
            assert_eq!(real.scratch_bytes(), stand_in.scratch_bytes(), "scratch at {rows} rows");
            assert_eq!(real.engine_reserve_bytes(), stand_in.engine_reserve_bytes(), "engine reserve at {rows} rows");
            assert_eq!(real.window_record_bytes(), stand_in.window_record_bytes());
            assert_eq!(real.state_bytes_per_slot(), stand_in.state_bytes_per_slot());
            assert_eq!(real.kv_page_bytes(), stand_in.kv_page_bytes());
            assert_eq!(real.kv_bytes_per_token(), stand_in.kv_bytes_per_token());
        }
    }

    #[test]
    fn the_boot_sizes_its_plan_the_way_every_other_family_does() {
        // the GPU boot cannot run here: pin the lines that decide what the plan is sized from (the guard of `serve::tests` for the families
        // that are not in its list). The free bytes go through `hostmem::planning_free` (the host's MemAvailable rounded down to a whole GiB, so
        // two boots of one box plan one cache), never the smaller of two figures; the admission comes from `MemoryPlan::admission` so it
        // declares the pool; the boot plans through `plan_boot`, which refuses a pool short of the window and a cache of fewer than two entries.
        let code = include_str!("engine.rs").split("#[cfg(test)]").next().unwrap();
        for needle in ["hostmem::planning_free(cuda_free", "plan.admission(", "plan_boot(&inputs)"] {
            assert!(code.contains(needle), "build_engine no longer calls {needle}");
        }
        for l in code.lines().filter(|l| !l.trim_start().starts_with("//")) {
            assert!(!(l.contains("cuda_free") && l.contains(".min(")), "the device's free is in a min again: {l}");
            assert!(!l.contains("Admission::new("), "an Admission built by hand, without the pool: {l}");
            assert!(!l.contains("mem_available()"), "the raw MemAvailable, without the planning quantum: {l}");
        }
    }

    /// What `boot` plans, as `(chunk, entries, reach_tokens)` or the refusal.
    fn plan_of(r: Result<Planned, PlanRefused>, window: u64) -> Result<(usize, usize, u64), PlanRefused> {
        r.map(|p| (p.chosen.chunk, p.chosen.plan.cache.snapshot_slots, reach_tokens(&p.chosen.plan.cache, window)))
    }

    // `MemAvailable` once the weights are resident, from the boots' `ling31_plan free_gib` (plus the 0.3 GiB after-plan reserve the line
    // has taken out): N4 9.8 (gate 0, 2026-11-05), N3 8.2 (Q200, 2026-11-02), N2 6.6 (it holds 2 GiB of huge pages).
    const N4: f64 = 10.1;
    const N3: f64 = 8.5;
    const N2: f64 = 6.9;

    #[test]
    fn the_shipped_profile_on_a_stock_node_boots_at_the_widest_chunk_that_keeps_its_cache() {
        // gate 0 booted this profile on N4 with `ling31_prefix_plan entries=0` (the 4,096-row scratch ate the cache). The planner now says so
        // and narrows the chunk: 4,096 and 2,048 rows leave the cache under two entries once the boot overhead is counted, 1,024 plans two
        // entries that reach the whole window.
        let p = boot(N4, 32_768, 4096, ON).expect("a stock GB10 hosts the profile");
        let c = &p.chosen;
        assert_eq!((p.requested_chunk, c.chunk), (4096, 1024));
        assert_eq!(c.plan.kv_tokens, 32_768, "the pool holds the whole window");
        assert!(c.plan.cache.enabled() && c.plan.cache.snapshot_slots >= 2);
        assert_eq!(reach_tokens(&c.plan.cache, 32_768), 32_768);
        assert_eq!(p.rejected.iter().map(|r| r.chunk).collect::<Vec<_>>(), vec![4096, 2048]);
        assert!(p.rejected[0].reason.starts_with("not enough memory"), "{}", p.rejected[0].reason);
        assert!(p.rejected[1].reason.starts_with("the prefix cache plans 0 entries"), "{}", p.rejected[1].reason);
        assert_eq!(format!("{:.2}", gib(p.rejected[0].scratch_bytes)), "3.99");
        assert_eq!(format!("{:.2}", gib(p.rejected[1].scratch_bytes)), "2.10");
        assert_eq!(format!("{:.2}", gib(c.profile.scratch_bytes())), "1.15");
        // the plan never asks for more than it was given: what it plans plus the floor fit in the free bytes (boot overhead already out)
        assert!(
            c.plan.planned_bytes(1) + 3 * GIB <= c.free,
            "planned {:.3} of {:.3} GiB",
            gib(c.plan.planned_bytes(1)),
            gib(c.free)
        );
        assert_eq!(format!("{:.2}", gib(c.free)), "7.72", "10 GiB planning quantum - 2 GiB boot overhead - 0.3 GiB reserve");
    }

    #[test]
    fn a_requested_chunk_is_a_ceiling() {
        for (ask, got) in [(4096, 1024), (2048, 1024), (1024, 1024)] {
            assert_eq!(plan_of(boot(N4, 32_768, ask, ON), 32_768).unwrap().0, got, "--prefill-chunk {ask}");
        }
        // 512 is the operator's: planned as asked, never narrowed, and refused when it does not fit
        let e = plan_of(boot(N2, 16_384, 512, ON), 16_384).unwrap_err();
        assert_eq!(e.tried.iter().map(|r| r.chunk).collect::<Vec<_>>(), vec![512]);
    }

    #[test]
    fn the_wider_the_window_the_less_a_stock_node_can_cache() {
        // N4, the shipped floor of 3 GiB, one slot: the chunk, the cache entries and how far a cache entry reaches
        let w = |window: u64| plan_of(boot(N4, window, 4096, ON), window);
        assert_eq!(w(16_384).unwrap(), (1024, 2, 16_384));
        assert_eq!(w(32_768).unwrap(), (1024, 2, 32_768));
        assert_eq!(w(65_536).unwrap(), (1024, 2, 15_872), "two entries, but they reach under a third of the window");
        // 131,072: the pool fits, the cache does not; 262,144: the pool itself does not
        let e = w(131_072).unwrap_err();
        assert!(e.tried.iter().all(|r| r.kv_pool_tokens == 0 || r.kv_pool_tokens == 131_072), "{e}");
        assert!(e.to_string().contains("the prefix cache plans 0 entries"), "{e}");
        let e = w(262_144).unwrap_err().to_string();
        assert!(e.contains("the KV pool holds 222464 tokens, below the 262144-token window"), "{e}");
        assert!(e.contains("pass --max-model-len 222208 or free host memory"), "{e}");
    }

    #[test]
    fn a_node_with_less_free_is_refused_by_name_or_needs_a_lower_floor() {
        // N3 and N2 at the profile's 3 GiB floor: no chunk plans two entries, and the boot says so (it used to boot with the cache off)
        for node in [N3, N2] {
            let e = boot(node, 32_768, 4096, ON).err().expect("refused").to_string();
            assert!(e.contains("prefix cache requested") && e.contains("chunk 4096") && e.contains("chunk 1024"), "{e}");
            assert!(e.ends_with("Free host memory, lower --max-model-len, or pass --prefix-cache off"), "{e}");
        }
        // N3 with a 1.5 GiB floor: two entries, reaching 8,960 of the 32,768 tokens (the floor is the operator's to lower, REXL3_MEMORY_FLOOR_GIB)
        let p = boot_at_floor(N3, 1.5, 32_768, 4096, ON).unwrap();
        assert_eq!(plan_of(Ok(p), 32_768).unwrap(), (1024, 2, 6_144));
        // N2 holds 2 GiB of huge pages: nothing plans a cache there, whatever the floor down to 1 GiB
        assert!(boot_at_floor(N2, 1.0, 16_384, 4096, ON).is_err());
        // without a cache the same node boots a smaller window
        assert!(boot_at_floor(N2, 1.0, 16_384, 1024, CacheRequest::Off).is_ok());
    }

    #[test]
    fn a_pool_short_of_the_window_is_refused_even_without_a_cache() {
        // N2 at 1 GiB of floor holds a 16,384-token window; 262,144 tokens would leave a pool far short of it (the server used to serve
        // `min(window, pool)` without saying why)
        assert!(boot_at_floor(N2, 1.0, 16_384, 1024, CacheRequest::Off).is_ok());
        let e = boot_at_floor(N2, 1.0, 262_144, 1024, CacheRequest::Off).err().expect("pool short").to_string();
        assert!(e.contains("below the 262144-token window"), "{e}");
        assert!(!e.contains("prefix cache requested"), "{e}");
    }

    #[test]
    fn the_refusal_names_every_chunk_it_tried_with_its_figures() {
        let e = boot(N3, 32_768, 4096, ON).err().unwrap();
        assert_eq!(e.tried.iter().map(|r| r.chunk).collect::<Vec<_>>(), vec![4096, 2048, 1024]);
        let text = e.to_string();
        for needle in [
            "free_gib=6.0 after the weights and 2.00 GiB of boot overhead",
            "floor_gib=3.0",
            "window=32768 tokens",
            "kv_pool_tokens=",
            "scratch_gib=3.99",
            "entries=0",
        ] {
            assert!(text.contains(needle), "{needle} not in {text}");
        }
    }
}
