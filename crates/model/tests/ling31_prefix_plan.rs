//! The Ling-3.1 profile's memory plan (`runtimes/ling-3.1/ling-3.1-flash-1019/profile.json`: 32,768 tokens, one slot, MTP, the cache on,
//! `REXL3_MEMORY_FLOOR_GIB=3`) by the arithmetic `bailing_moe_v3::engine::build_engine` boots with, on the real config: the prefix
//! cache sized from the window, its pages the host budget of the snapshots' MLA copies, and what that leaves on a stock GB10. CPU only (the
//! stock-GB10 plans need the kernels' workspace sizes, so a build without CUDA kernels skips them).

use gb_model::families::bailing_moe_v3::backend::snapshot_fixed_bytes;
use gb_model::families::bailing_moe_v3::config::BailingConfig;
use gb_model::families::bailing_moe_v3::engine::{
    family_floor_bytes, plan_after_weights, post_alloc_reserve_bytes, reach_tokens,
};
use gb_model::families::bailing_moe_v3::memory;
use gb_model::memory::MemoryOptions;
use gb_model::plan::{CacheRequest, MemoryPlan, AUTO_CACHE_CAP_BYTES};
use gb_sched::prefix::CacheBudget;

const CONFIG: &str = include_str!("../../../testdata/ling31-flash-1019/config.json");
const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
/// The profile's `--prefix-cache on`.
const ON: CacheRequest = CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES };

fn cfg() -> BailingConfig {
    BailingConfig::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
}

/// The profile's memory model: `--max-model-len 32768 --slots 1 --spec mtp`, the server's default `--prefill-chunk` (4096, which this
/// family clamps to its 512-row chunk), no vision.
fn ling_profile() -> gb_sched::MemoryProfile {
    let o =
        MemoryOptions { prefill_rows: 4096, max_context: 32_768, mtp: true, vision: None, ..MemoryOptions::serving_defaults() };
    memory::memory_profile(&cfg(), &memory::engine_options(&o)).unwrap()
}

/// The profile's plan on a node with `free_after_weights` bytes free once the weights are resident (the embedding table read in place
/// or copied to the device), and the free bytes it was made from.
fn plan(free_after_weights: u64, in_place: bool) -> (MemoryPlan, u64) {
    let c = cfg();
    let reserve = post_alloc_reserve_bytes(&c, 1, true, in_place, 32_768);
    let snapshot = snapshot_fixed_bytes(&c, true);
    plan_after_weights(&ling_profile(), free_after_weights, reserve, family_floor_bytes(3 * GIB), 1, 32_768, ON, snapshot, 32_768)
        .unwrap()
}

fn gib(b: u64) -> f64 {
    b as f64 / GIB as f64
}

#[test]
fn one_plan_page_is_one_mla_copy_of_256_tokens() {
    // the cache's pages are the host budget of the snapshots' MLA page copies (`build_engine`): that is exact because a page of the
    // plan (every MLA layer's 256 tokens, the MTP layer's included) is the bytes a snapshot copies for 256 tokens of its prefix
    let p = ling_profile();
    let mla_caches = cfg().mla_layers().len() as u64 + 1;
    assert_eq!(mla_caches, 10);
    let copy = mla_caches * gb_cuda::mla_gated::MlaGatedCache::block_bytes(gb_cuda::exl3_linear::Dtype::BF16) as u64;
    assert_eq!(p.kv_page_bytes(), copy);
    assert_eq!(copy, 2_949_120);
}

#[test]
fn the_snapshot_slot_is_the_kda_pools_and_the_tap_ring() {
    let c = cfg();
    // per KDA layer: two f32 state buffers (64 heads x 128 x 128), the conv history (3 rows of q|k|v, bf16) and the parity word
    let layer = 2 * 64 * 128 * 128 * 4 + 3 * 3 * 64 * 128 * 2 + 4;
    assert_eq!(layer, 8_536_068);
    assert_eq!(c.kda_layers().len(), 63);
    let backend = snapshot_fixed_bytes;
    assert_eq!(backend(&c, false), 63 * layer);
    // with MTP the snapshot also holds the tap ring: 8,192 bf16 rows of 4,096
    assert_eq!(backend(&c, true), 63 * layer + 8192 * 4096 * 2);
    assert_eq!(format!("{:.1}", backend(&c, true) as f64 / MIB as f64), "576.9");
}

#[test]
fn a_stock_gb10_plans_two_entries_with_the_table_in_place() {
    if !gb_cuda::rt::kernels_built() {
        eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes (the engine reserve) are not known");
        return;
    }
    // A boot of this profile on fe729abf, on a stock GB10 (116.3 GiB MemAvailable idle), logged `ling31_plan free_gib=5.8
    // post_alloc_reserve_gib=1.5` and `ling31_prefix_plan entries=0`. `free_gib` is rounded to 0.1, so the node had
    // 5.75 ..= 5.85 GiB plus the device reserve free once the weights were resident; both ends of that range are checked.
    // The engine reserve is the real scratch (1.155 GiB: the shared 512-row prefill KDA record, the 64-row window records, the
    // FrostMoE and EXL3 workspaces, `AllocPlan::scratch_items`), so the pages cover a whole window from the range's high end and
    // 119 pages (30,464 tokens) at its low end.
    let device = post_alloc_reserve_bytes(&cfg(), 1, true, false, 32_768);
    for (logged, pages) in [(5.75, 119), (5.849, 146)] {
        let free_after_weights = (logged * GIB as f64) as u64 + device;
        // the window rule alone does not make room: with the table copied there is still no cache
        let (copied, _) = plan(free_after_weights, false);
        assert!(!copied.cache.enabled(), "free_gib={logged}: {:?}", copied.cache);
        let (after, free) = plan(free_after_weights, true);
        assert_eq!(free, (logged * GIB as f64) as u64 + 1_287_651_328);
        eprintln!(
            "logged free_gib={logged}: in place free {:.3} GiB, engine reserve {:.3}, slot state {:.3}, kv {} pages ({:.3} GiB), \
             spare {:.3}, cache {} entries {} pages {:.3} GiB (host), reach {} tokens, planned {:.3} GiB",
            gib(free),
            gib(after.engine_reserve_bytes),
            gib(after.state_bytes_per_slot),
            after.kv_blocks,
            gib(after.kv_blocks as u64 * after.block_bytes),
            gib(after.spare_bytes),
            after.cache.snapshot_slots,
            after.cache.page_blocks,
            gib(after.cache.bytes()),
            after.cache.reach_tokens(),
            gib(after.planned_bytes(1))
        );
        assert_eq!(after.cache.snapshot_slots, 2);
        assert_eq!(after.cache.page_blocks, pages, "{:?}", after.cache);
        assert_eq!(reach_tokens(&after.cache, 32_768), (pages * 256).min(32_768) as u64, "the boot reports at most the window");
        // the device pool is the window (and a page for the slot), nothing for the cache
        assert_eq!(after.kv_blocks, 129);
        assert_eq!(after.kv_tokens, 32_768);
        assert!(after.planned_bytes(1) + 3 * GIB <= free, "the plan never asks for more than it was given");
    }
}

#[test]
fn with_the_table_in_place_two_entries_survive_a_node_with_less_memory_free() {
    if !gb_cuda::rt::kernels_built() {
        eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes (the engine reserve) are not known");
        return;
    }
    // The least free memory (once the weights are resident) at which the profile still plans two entries, and at which their pages
    // cover one and two whole windows, against the stock GB10's above (5.75 ..= 5.85 GiB plus the device reserve; margins are
    // quoted from the low end, negative when the stock GB10 is short of it).
    let device = post_alloc_reserve_bytes(&cfg(), 1, true, false, 32_768);
    let node_low = (5.75 * GIB as f64) as u64 + device;
    let least = |ok: &dyn Fn(&CacheBudget) -> bool| {
        (0..=8192u64).map(|m| node_low + 2 * GIB - m * MIB).take_while(|&f| ok(&plan(f, true).0.cache)).last().unwrap()
    };
    let enabled = least(&|c| c.enabled());
    let one_window = least(&|c| c.enabled() && c.page_blocks >= 128);
    let two_windows = least(&|c| c.enabled() && c.page_blocks >= 256);
    let margin = |f: u64| (node_low as i64 - f as i64) / MIB as i64;
    for (what, f) in [("2 entries", enabled), ("one full window of pages", one_window), ("two full windows", two_windows)] {
        eprintln!(
            "least free_after_weights for {what}: {:.3} GiB (margin {} MiB against the stock GB10's {:.3} ..= {:.3} GiB): {:?}",
            gib(f),
            margin(f),
            gib(node_low),
            gib(node_low + GIB / 10),
            plan(f, true).0.cache
        );
    }
    assert!(enabled < one_window && one_window < two_windows);
    assert_eq!((margin(enabled), margin(one_window), margin(two_windows)), (387, -33, -513));
    // one window of pages is inside the stock GB10's range, two are not
    assert!(one_window <= node_low + GIB / 10 && two_windows > node_low + GIB / 10);
    // under one window of pages the cache still has two entries, and says its reach (`reach_tokens`): a full-window prefix's MLA
    // copy then runs past the plan, which the floor and `can_snapshot_prefix`'s MemAvailable check absorb
    assert_eq!(plan(enabled, true).0.cache.snapshot_slots, 2);
    assert_eq!(reach_tokens(&plan(enabled, true).0.cache, 32_768), 16 * 256, "under one window the boot says so");
    assert!(!plan(enabled - MIB, true).0.cache.enabled());
}
