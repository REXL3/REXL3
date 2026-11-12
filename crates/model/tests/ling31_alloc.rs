//! The Ling-3.1 backend's device allocations against its memory model, on the whole model (the real pack, MTP on, the profile's one slot and
//! its 32,768-token window): every buffer `BailingModel::new` and `BailingDevice::build` allocate beside the pack's regions is an item of
//! `alloc::AllocPlan` of the same bytes, and every item is allocated. `#[ignore]`, env-gated (`REXL3_LING_DIR`), one model per process, with
//! a 1 GiB `MemAvailable` guard that aborts the process.

mod ling31_support;

use gb_model::families::bailing_moe_v3::alloc::{AllocParams, AllocPlan};
use gb_model::families::bailing_moe_v3::engine::post_alloc_reserve_bytes;
use gb_model::families::bailing_moe_v3::{load, memory};
use gb_model::memory::MemoryOptions;
use ling31_support::*;

/// The profile's pool: the window and a page for the slot (`plan::plan_memory_cached_windowed` caps it there).
const KV_BLOCKS: usize = 32_768 / 256 + 1;

#[test]
#[ignore]
fn allocations_equal_the_plan() {
    let Some(m) = load(1, KV_BLOCKS * 256, true) else { return };
    let p = AllocParams {
        slots: 1,
        kv_blocks: KV_BLOCKS,
        prefill_rows: m.prefill_rows(),
        mtp: true,
        embed_in_place: load::embed_in_place(),
    };
    let plan = AllocPlan::new(&m.cfg, &p);
    for (name, bytes) in m.allocations() {
        eprintln!("{name:24} {:10.2} MiB", bytes as f64 / 1048576.0);
    }
    m.check_allocations(&plan, &p).unwrap_or_else(|e| panic!("the memory model and the allocations disagree: {e}"));
    // the boot's arithmetic is the same plan: the profile (slot state, pages, window records, scratch) and the after-plan reserve
    let o =
        MemoryOptions { prefill_rows: 4096, max_context: 32_768, mtp: true, vision: None, ..MemoryOptions::serving_defaults() };
    let profile = memory::memory_profile(&m.cfg, &memory::engine_options(&o)).unwrap();
    let booted = profile.state_bytes_per_slot()
        + KV_BLOCKS as u64 * profile.kv_page_bytes()
        + profile.window_record_bytes()
        + profile.scratch_bytes()
        + post_alloc_reserve_bytes(&m.cfg, 1, true, p.embed_in_place, 32_768);
    assert_eq!(booted, plan.total(&p), "the boot prices what the backend allocates");
    let all: u64 = m.allocations().iter().map(|(_, b)| b).sum();
    eprintln!(
        "allocated after the plan {:.3} GiB; MemAvailable {:.2} GiB (low {:.2})",
        gib(all),
        gib(mem_available()),
        gib(guard().low())
    );
}
