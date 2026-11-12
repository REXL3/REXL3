//! The Ling-3.1 profile's memory plan (`runtimes/ling-3.1/ling-3.1-flash-1019/profile.json`: 32,768 tokens, one slot, MTP, the cache on,
//! `REXL3_MEMORY_FLOOR_GIB=3`) by the arithmetic `bailing_moe_v3::engine::build_engine` boots with, on the real config: the prefix
//! cache sized from the window, its pages the host budget of the snapshots' MLA copies, and what that leaves on a stock GB10. CPU only (the
//! stock-GB10 plans need the kernels' workspace sizes, so a build without CUDA kernels skips them).

use gb_model::families::bailing_moe_v3::backend::snapshot_fixed_bytes;
use gb_model::families::bailing_moe_v3::config::BailingConfig;
use gb_model::families::bailing_moe_v3::engine::{
    family_floor_bytes, plan_after_weights, plan_boot, post_alloc_reserve_bytes, profile_at, reach_tokens, PlanInputs,
    PlanRefused, Planned, BOOT_OVERHEAD_BYTES,
};
use gb_model::families::bailing_moe_v3::memory;
use gb_model::families::bailing_moe_v3::spill::{SpillConfig, SpillMeta, SpillStore, SPILL_STAGING_BYTES};
use gb_model::memory::MemoryOptions;
use gb_model::plan::{CacheRequest, MemoryPlan, AUTO_CACHE_CAP_BYTES};

const CONFIG: &str = include_str!("../../../testdata/ling31-flash-1019/config.json");
const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
/// The profile's `--prefix-cache on`.
const ON: CacheRequest = CacheRequest::Auto { cap_bytes: AUTO_CACHE_CAP_BYTES };

fn cfg() -> BailingConfig {
    BailingConfig::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
}

/// The profile's memory model: `--max-model-len 32768 --slots 1 --spec mtp`, the server's default `--prefill-chunk` (4096, which this
/// family uses at its widest chunk), no vision.
fn ling_profile() -> gb_sched::MemoryProfile {
    let o =
        MemoryOptions { prefill_rows: 4096, max_context: 32_768, mtp: true, vision: None, ..MemoryOptions::serving_defaults() };
    memory::memory_profile(&cfg(), &memory::engine_options(&o)).unwrap()
}

/// The spill store's plan at the full 262,144-token window: the disk quota (64 GiB) sizes entries and pages, the RAM cost is
/// only the staging buffer, and the plan fits at the 2 GiB floor with `reach_tokens == 262144`.
#[test]
fn the_spill_store_plan_reaches_the_whole_262k_window_at_the_two_gib_floor() {
    if !gb_cuda::rt::kernels_built() {
        eprintln!("not run: this build has no CUDA kernels");
        return;
    }
    let c = cfg();
    let host = gb_model::hostmem::MemInfo { available: (N4 * GIB as f64) as u64, ..Default::default() };
    let free = gb_model::hostmem::planning_free(0, Some(&host)).saturating_sub(BOOT_OVERHEAD_BYTES);
    let base = MemoryOptions { prefill_rows: 4096, mtp: true, ..MemoryOptions::serving_defaults() };
    let at = |rows: usize| profile_at(&c, &base, rows);
    let spill = SpillConfig {
        dir: std::env::temp_dir().join("ling-plan-spill-none"),
        quota_bytes: 64 << 30,
        staging_bytes: SPILL_STAGING_BYTES,
    };
    let p = plan_boot(&PlanInputs {
        cfg: &c,
        base,
        profile_at: &at,
        free_after_weights: free,
        floor_bytes: family_floor_bytes(2 * GIB),
        slots: 1,
        kv_tokens_cap: 262_144,
        cache: ON,
        snapshot_bytes: snapshot_fixed_bytes(&c, true),
        window_tokens: 262_144,
        in_place: true,
        with_mtp: true,
        spill: Some(spill),
    })
    .expect("the spill cache fits 262,144 tokens at the 2 GiB floor");
    let sel = &p.chosen;
    eprintln!(
        "spill 262K: chunk {} entries {} pages {} reach {} planned {:.3} GiB of free {:.3}",
        sel.chunk,
        sel.plan.cache.snapshot_slots,
        sel.plan.cache.page_blocks,
        reach_tokens(&sel.plan.cache, 262_144),
        gib(sel.plan.planned_bytes(1)),
        gib(sel.free),
    );
    assert_eq!(sel.plan.kv_tokens, 262_144, "the pool holds the whole window");
    assert!(sel.plan.cache.enabled(), "the spill cache is enabled");
    assert!(sel.plan.cache.snapshot_slots >= 2, "entries: {}", sel.plan.cache.snapshot_slots);
    assert!(sel.plan.cache.page_blocks >= 1024, "pages: {}", sel.plan.cache.page_blocks);
    assert_eq!(reach_tokens(&sel.plan.cache, 262_144), 262_144, "reach_tokens == window");
    assert!(
        sel.plan.planned_bytes(1) + 2 * GIB <= sel.free,
        "planned {:.3} GiB of {:.3}",
        gib(sel.plan.planned_bytes(1)),
        gib(sel.free),
    );
}

/// The profile's plan on a node with `free_after_weights` bytes free once the weights are resident (the embedding table read in place
/// or copied to the device), and the free bytes it was made from.
fn plan(free_after_weights: u64, in_place: bool) -> Result<(MemoryPlan, u64), gb_model::plan::PlanError> {
    let c = cfg();
    let reserve = post_alloc_reserve_bytes(&c, 1, true, in_place, 32_768);
    let snapshot = snapshot_fixed_bytes(&c, true);
    plan_after_weights(&ling_profile(), free_after_weights, reserve, family_floor_bytes(3 * GIB), 1, 32_768, ON, snapshot, 32_768)
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

/// Free bytes on a stock GB10 once the 106 GiB pack is resident: 116.3 GiB MemAvailable idle minus the pack. The boot
/// that anchored the old test logged `free_gib=5.8` (a rounded 0.1), so the low end is used.
fn stock_gb10_free_after_weights() -> u64 {
    let device_reserve = post_alloc_reserve_bytes(&cfg(), 1, true, false, 32_768);
    (5.75 * GIB as f64) as u64 + device_reserve
}

#[test]
fn the_wide_scratch_is_priced_and_the_window_pool_survives_when_memory_allows() {
    if !gb_cuda::rt::kernels_built() {
        eprintln!("not run: this build has no CUDA kernels");
        return;
    }
    let p = ling_profile();
    assert_eq!(p.shape().scratch_rows, 4096);
    assert!(p.engine_reserve_bytes() > 2 * GIB, "{:.3} GiB", gib(p.engine_reserve_bytes()));

    // The truthful wide-chunk plan can refuse the stock node's small post-weight remainder; that is safer than
    // oversubscribing it. With additional free memory the same plan must retain the whole window and bound its cache.
    let stock = stock_gb10_free_after_weights();
    if let Err(e) = plan(stock, true) {
        eprintln!("the stock node cannot hold the wide profile (the plan refuses): {e}");
    }
    let (after, free) = plan(stock + 8 * GIB, true).unwrap();
    assert_eq!(after.kv_blocks, 129);
    assert_eq!(after.kv_tokens, 32_768);
    assert!(after.planned_bytes(1) + 3 * GIB <= free);
    if after.cache.enabled() {
        assert!(after.cache.snapshot_slots >= 2);
        assert!(after.cache.bytes() <= after.spare_bytes);
        assert_eq!(reach_tokens(&after.cache, 32_768), after.cache.reach_tokens().min(32_768));
    }
    if let Ok((copied, _)) = plan(stock + 8 * GIB, false) {
        assert!(copied.cache.bytes() <= after.cache.bytes());
        assert_eq!(copied.kv_blocks, 129);
    }
}

#[test]
fn the_cache_shrinks_smoothly_then_disables_or_the_plan_refuses() {
    if !gb_cuda::rt::kernels_built() {
        eprintln!("not run: this build has no CUDA kernels");
        return;
    }
    let high = stock_gb10_free_after_weights() + 8 * GIB;
    let mut last: Option<gb_model::plan::MemoryPlan> = None;
    let mut stopped = false;
    for step in 0..=192u64 {
        let f = high.saturating_sub(step * 64 * MIB);
        let Ok((p, _)) = plan(f, true) else {
            stopped = true;
            break;
        };
        assert_eq!(p.kv_blocks, 129, "the window pool survives at every feasible size");
        if let Some(prev) = &last {
            assert!(p.cache.bytes() <= prev.cache.bytes(), "the cache never grows as memory drops at step {step}");
        }
        if !p.cache.enabled() {
            stopped = true;
            break;
        }
        last = Some(p);
    }
    assert!(stopped, "the cache disables or the plan refuses within 12 GiB");
}

// ---- the boot planner on the real profile: the stock nodes' figures

/// `MemAvailable` once the 106.3 GiB weights are resident, on the nodes the Ling profile was booted on (`ling31_plan free_gib` plus the
/// 0.3 GiB after-plan reserve the line has already taken out): N4 9.8, N3 8.2 (2026-11-02), N2 6.6 (it holds 2 GiB of huge pages).
const N4: f64 = 10.1;
const N3: f64 = 8.5;
const N2: f64 = 6.9;

/// The shipped profile's boot (`--max-model-len`, one slot, MTP, `--prefix-cache on`, floor 3) on a node with `after_weights_gib` free once the
/// weights are resident, for a requested `--prefill-chunk`: what `build_engine` plans, through the same `plan_boot`.
fn boot(after_weights_gib: f64, window: u64, requested_chunk: usize) -> Result<Planned, PlanRefused> {
    let c = cfg();
    let host = gb_model::hostmem::MemInfo { available: (after_weights_gib * GIB as f64) as u64, ..Default::default() };
    let free = gb_model::hostmem::planning_free(0, Some(&host)).saturating_sub(BOOT_OVERHEAD_BYTES);
    let base = MemoryOptions { prefill_rows: requested_chunk, mtp: true, ..MemoryOptions::serving_defaults() };
    let at = |rows: usize| profile_at(&c, &base, rows);
    plan_boot(&PlanInputs {
        cfg: &c,
        base,
        profile_at: &at,
        free_after_weights: free,
        floor_bytes: family_floor_bytes(3 * GIB),
        slots: 1,
        kv_tokens_cap: window,
        cache: ON,
        snapshot_bytes: snapshot_fixed_bytes(&c, true),
        window_tokens: window,
        in_place: true,
        with_mtp: true,
        spill: None,
    })
}

#[test]
fn the_shipped_default_chunk_is_narrowed_until_a_stock_node_plans_its_cache() {
    if !gb_cuda::rt::kernels_built() {
        eprintln!("not run: this build has no CUDA kernels");
        return;
    }
    // the profile as shipped (32,768 tokens, the server's default `--prefill-chunk` 4096). The widest chunk is refused for its scratch; the boot
    // takes the widest one that still plans two entries and the whole window, and says what it refused.
    let p = boot(N4, 32_768, 4096).expect("a stock GB10 hosts the shipped profile");
    let c = &p.chosen;
    eprintln!(
        "N4: chunk {} (asked {}), kv pool {} tokens, scratch {:.2} GiB, {} entries, reach {}, free {:.2} GiB; refused {:?}",
        c.chunk,
        p.requested_chunk,
        c.plan.kv_tokens,
        gib(c.profile.scratch_bytes()),
        c.plan.cache.snapshot_slots,
        reach_tokens(&c.plan.cache, 32_768),
        gib(c.free),
        p.rejected.iter().map(|r| r.chunk).collect::<Vec<_>>()
    );
    assert_eq!(p.requested_chunk, 4096);
    assert!(c.chunk < 4096 && c.chunk >= 1024, "chunk {}", c.chunk);
    assert!(p.rejected.iter().all(|r| r.chunk > c.chunk), "only wider chunks are refused");
    assert!(p.rejected.iter().any(|r| r.chunk == 4096), "the 4,096 default is refused");
    assert!(c.plan.cache.enabled() && c.plan.cache.snapshot_slots >= 2, "two entries: {:?}", c.plan.cache);
    assert_eq!(c.plan.kv_tokens, 32_768, "the pool holds the whole window");
    assert_eq!(reach_tokens(&c.plan.cache, 32_768), 32_768, "and the cache reaches it");
    // the plan never asks for more than it was given: what it plans, the floor, and the unpriced boot overhead fit in what is free
    assert!(c.plan.planned_bytes(1) + 3 * GIB <= c.free, "planned {:.3} GiB of {:.3}", gib(c.plan.planned_bytes(1)), gib(c.free));
}

#[test]
fn a_node_that_cannot_hold_the_profile_is_refused_with_the_exact_reason() {
    if !gb_cuda::rt::kernels_built() {
        eprintln!("not run: this build has no CUDA kernels");
        return;
    }
    // N2 holds 2 GiB of huge pages: no chunk plans two entries at the profile's 3 GiB floor, and the boot says so instead of booting without
    // the cache (the recipe's own `forbid_log` is `ling31_prefix_plan entries=0`)
    let e = boot(N2, 32_768, 4096).err().expect("N2 cannot host the profile").to_string();
    eprintln!("N2: {e}");
    assert!(
        e.contains("prefix cache requested") && e.contains("chunk 4096") && e.contains("chunk 2048") && e.contains("chunk 1024"),
        "{e}"
    );
    assert!(e.contains("the prefix cache plans 0 entries") || e.contains("not enough memory"), "{e}");
    assert!(e.contains("--prefix-cache off"), "{e}");
}

#[test]
fn a_window_the_pool_cannot_hold_is_refused_not_booted_short() {
    if !gb_cuda::rt::kernels_built() {
        eprintln!("not run: this build has no CUDA kernels");
        return;
    }
    // 262,144 tokens of window need a 2.81 GiB pool: a node whose memory leaves less would have booted a window no full prompt could use
    // (the server then served `min(window, pool)` without saying why). Now the boot names the pool it can hold.
    let e = boot(N4, 262_144, 4096).err().expect("N4 cannot hold a 262,144-token window with the cache").to_string();
    eprintln!("N4 262144: {e}");
    assert!(e.contains("below the 262144-token window"), "{e}");
    assert!(e.contains("--max-model-len"), "{e}");
}

#[test]
fn the_wider_the_chunk_the_less_memory_is_left_for_the_pool_and_the_cache() {
    if !gb_cuda::rt::kernels_built() {
        eprintln!("not run: this build has no CUDA kernels");
        return;
    }
    // what each chunk costs (the scratch the plan prices, measured as `AllocPlan::scratch_items`), at the profile's window
    let c = cfg();
    let base = MemoryOptions { mtp: true, ..MemoryOptions::serving_defaults() };
    let reserve = |rows: usize| profile_at(&c, &base, rows).unwrap().1.engine_reserve_bytes();
    assert!(reserve(1024) < reserve(2048) && reserve(2048) < reserve(4096));
    assert!(reserve(4096) - reserve(1024) > 5 * GIB / 2, "4,096 rows cost over 2.5 GiB more than 1,024");
}

const SPILL_META: SpillMeta =
    SpillMeta { namespace: 7, len_tokens: 512, n_pages: 1, n_kda_layers: 1, mtp: false, n_mla_caches: 1, page_block_bytes: 64 };

fn spill_test_dir(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("ling-spill-public-{name}-{}", std::process::id()))
}

fn write_spill_snapshot(store: &mut SpillStore, id: u64, payload: &[u8]) -> std::path::PathBuf {
    store.start_write(id, SPILL_META, payload.len() as u64).unwrap();
    store.push_bytes(payload).unwrap();
    store.finish_write().unwrap();
    store.last_path().to_path_buf()
}

#[test]
fn a_checksum_failure_does_not_wedge_the_next_capture() {
    let dir = spill_test_dir("checksum");
    let _ = std::fs::remove_dir_all(&dir);
    let mut store = SpillStore::open_at(dir.clone(), 1 << 20, 32).unwrap();
    let payload: Vec<u8> = (0..192).map(|i| (i % 31) as u8).collect();
    let corrupt = write_spill_snapshot(&mut store, 1, &payload);
    let mut raw = std::fs::read(&corrupt).unwrap();
    let header_len = 88 + payload.len().div_ceil(32) * 8;
    raw[header_len + 40] ^= 0x80;
    std::fs::write(&corrupt, raw).unwrap();

    store.begin_read(&corrupt, SPILL_META).unwrap();
    let mut out = vec![0; payload.len()];
    assert!(store.read_exact_into(&mut out).is_err());

    let replacement = write_spill_snapshot(&mut store, 2, &payload);
    store.begin_read(&replacement, SPILL_META).unwrap();
    store.read_exact_into(&mut out).unwrap();
    store.finish_read().unwrap();
    assert_eq!(out, payload);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn explicit_abort_paths_return_the_spill_store_to_a_reusable_state() {
    let dir = spill_test_dir("abort");
    let _ = std::fs::remove_dir_all(&dir);
    let mut store = SpillStore::open_at(dir.clone(), 1 << 20, 32).unwrap();
    let payload = vec![42; 96];
    let path = write_spill_snapshot(&mut store, 3, &payload);

    store.begin_read(&path, SPILL_META).unwrap();
    let mut first = [0; 16];
    store.read_exact_into(&mut first).unwrap();
    store.abort_read();
    store.abort_read();

    store.start_write(4, SPILL_META, payload.len() as u64).unwrap();
    store.push_bytes(&payload[..16]).unwrap();
    let partial = store.last_path().to_path_buf();
    store.abort_write();
    store.abort_write();
    assert!(!partial.exists());

    let replacement = write_spill_snapshot(&mut store, 5, &payload);
    store.begin_read(&replacement, SPILL_META).unwrap();
    let mut out = vec![0; payload.len()];
    store.read_exact_into(&mut out).unwrap();
    store.finish_read().unwrap();
    assert_eq!(out, payload);
    std::fs::remove_dir_all(dir).unwrap();
}
