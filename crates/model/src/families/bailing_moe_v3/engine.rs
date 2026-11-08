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
use crate::error::ModelError;
use crate::hostmem;
use crate::pack::PackError;
use crate::serve::{gib, mem_available, memory_options, ServeOptions};
use gb_engine::{boot, Drafter, Engine, EngineConfig, NoDrafts};
use gb_sched::{CostModel, ModelMemory};

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
    let cfg = BailingConfig::from_dir(&o.model_dir).map_err(ModelError::Config)?;
    let with_mtp = o.drafting && o.spec.uses_mtp() && cfg.has_mtp();

    let mem_opts = memory::engine_options(&memory_options(o, with_mtp));
    let profile = memory::memory_profile(&cfg, &mem_opts).map_err(|e| ModelError::Config(e.to_string()))?;

    let ck = gb_exl3::checkpoint::scan(&o.model_dir, false).map_err(|e| ModelError::Pack(PackError::Io(e.to_string())))?;
    let inv = PackInventory::from_checkpoint(&ck).map_err(|e| ModelError::Config(format!("pack inventory: {e}")))?;
    let pack = BailingPack::from_inventory(&inv, cfg.clone()).map_err(|e| ModelError::Config(e.to_string()))?;
    boot::fact("target_layers", cfg.layers.len().to_string());
    boot::fact("target_fingerprint", crate::fingerprint::fingerprint(&ck));

    let weights_bytes = pack.plan.device_bytes();
    let free = match (mem_available(), gb_cuda::rt::mem_info()) {
        (Some(avail), _) => avail.saturating_sub(weights_bytes),
        (None, Ok((cuda_free, _))) => (cuda_free as u64).saturating_sub(weights_bytes),
        (None, Err(_)) => {
            let host = hostmem::MemInfo::read();
            host.as_ref().map(|h| h.available.saturating_sub(weights_bytes)).unwrap_or(0)
        }
    };
    let requested_slots = o.slots.max(1);
    let in_place = super::load::embed_in_place();
    let kv_tokens_cap = o.kv_tokens.unwrap_or(o.max_model_len) as u64;
    let reserve = post_alloc_reserve_bytes(&cfg, requested_slots, with_mtp, in_place, kv_tokens_cap);
    let floor_bytes = family_floor_bytes(o.floor_bytes);
    let snapshot_bytes = super::backend::snapshot_fixed_bytes(&cfg, with_mtp);
    let window_tokens = o.max_model_len as u64;
    let (plan, free) = plan_after_weights(
        &profile,
        free,
        reserve,
        floor_bytes,
        requested_slots,
        kv_tokens_cap,
        o.prefix_cache.request(),
        snapshot_bytes,
        window_tokens,
    )
    .map_err(|e| ModelError::Config(e.to_string()))?;
    let embed = if in_place { "shard" } else { "device" };
    boot::fact("plan_free_gib", format!("{:.1}", gib(free)));
    boot::fact("plan_post_alloc_reserve_gib", format!("{:.1}", gib(reserve)));
    boot::fact("embedding_table", embed.to_string());
    eprintln!(
        "rexl3 level=info event=ling31_plan weights_gib={:.1} free_gib={:.1} floor_gib={:.1} post_alloc_reserve_gib={:.1} embed={embed}",
        gib(weights_bytes),
        gib(free),
        gib(floor_bytes),
        gib(reserve)
    );
    let slots = requested_slots;

    let loaded = load_pack(&o.model_dir, &pack, &mut |_, _| {})?;
    let dev = BailingDevice::build(&cfg, &pack.weights, loaded.regions).map_err(ModelError::Config)?;
    // The KV pool is the plan's `kv_blocks` and no more: a snapshot copies its prefix's MLA pages to host RAM (`MlaCacheSnap`) and
    // hands the engine no pages (`Captured { pages: [] }`), and a restore copies them back into pages of the live pool. So the cache's
    // `page_blocks` are not device pages here but the host budget of those copies: one plan page is exactly one 256-token copy (the
    // test `one_plan_page_is_one_mla_copy_of_256_tokens`). Adding them to the pool, as before, allocated device pages nothing used.
    let kv_tokens = plan.kv_blocks * 256;
    let model = BailingModel::new(cfg.clone(), dev, loaded.embedding, slots, kv_tokens, with_mtp)?;

    let mut backend = BailingBackend::new(model, with_mtp);
    if plan.cache.enabled() {
        let ns = super::backend::prefix_namespace(&crate::fingerprint::fingerprint(&ck), with_mtp);
        backend = backend.with_prefix_cache(ns, plan.cache).map_err(|e| ModelError::Config(format!("the prefix cache: {e}")))?;
        let reach = reach_tokens(&plan.cache, window_tokens);
        eprintln!(
            "rexl3 event=ling31_prefix_plan entries={} pages={} gib={:.2} reach_tokens={reach} window={window_tokens} where=host",
            plan.cache.snapshot_slots,
            plan.cache.page_blocks,
            gib(plan.cache.bytes()),
        );
        boot::fact("prefix_cache_reach_tokens", reach.to_string());
    } else {
        let reason = if matches!(o.prefix_cache.request(), crate::plan::CacheRequest::Off) {
            "off"
        } else {
            "insufficient spare memory for two entries"
        };
        eprintln!("rexl3 event=ling31_prefix_plan entries=0 reason={reason}");
    }
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
        // interactive cut policy is the one that serves a second question about a long document (as the other families)
        Ok(engine.with_prefix_cache(crate::serve::interactive_prefix_config_from_env()?))
    } else {
        Ok(engine)
    }
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
                prefill_rows: 512,
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
    fn the_floor_is_the_one_the_profile_asks_for() {
        assert_eq!(family_floor_bytes(3 * GIB), 3 * GIB, "REXL3_MEMORY_FLOOR_GIB=3");
        assert_eq!(family_floor_bytes(8 << 30), GIB, "the 8 GiB server default becomes this family's 1 GiB");
    }
}
