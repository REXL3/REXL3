//! The planning-free choice (`hostmem::planning_free_on`): the single figure every memory plan is sized from,
//! and the fix that keeps full-context boots alive on a discrete GPU.
//!
//! On a unified-memory device (the GB10, `cudaDevAttrIntegrated` == 1) the host's `MemAvailable` binds:
//! `cudaMalloc` reclaims the host's page cache, so the device's own free figure is only `MemFree` and must
//! not shrink the plan. On a discrete GPU the device has its own VRAM and a plan sized from `MemAvailable`
//! alone can ask it for more than it has and die in `cudaMalloc`; the plan is sized from the smaller of the
//! device's free and the host's `MemAvailable`. This test pins the choice at the public seam, so a family's
//! boot can be read against the box's class.

use gb_model::hostmem::{memory_view_line_on, planning_free, planning_free_on, MemInfo};
use gb_model::serve::{floor_bytes_for_device, TpOptions, DISCRETE_MEMORY_FLOOR};

const GIB: u64 = 1 << 30;

/// The RTX-class box of the bug: after the weights are resident the device reports ~51 GiB free of its own
/// VRAM while the host still reports ~116 GiB of MemAvailable.
fn rtx_host() -> MemInfo {
    MemInfo { total: 121 * GIB, free: 12 * GIB, available: 116 * GIB, cached: 80 * GIB }
}

#[test]
fn a_discrete_device_plans_from_the_smaller_of_the_device_and_the_host() {
    let h = rtx_host();
    // device free binds: min(24 GiB, 116 GiB) = 24 GiB
    assert_eq!(planning_free_on(24 * GIB, Some(&h), false), 24 * GIB);
    // host binds: min(96 GiB, 40 GiB) = 40 GiB
    let host_bound = MemInfo { available: 40 * GIB, ..h };
    assert_eq!(planning_free_on(96 * GIB, Some(&host_bound), false), 40 * GIB);
    // the discrete rule keeps the exact bytes: the GiB rounding would cost up to a GiB of the small headroom a
    // 24/32/48 GiB card has after the weights (free 2.69 GiB -> planned 2.00 alone costs the 24 GiB card its boot)
    assert_eq!(planning_free_on(51 * GIB - 1, Some(&h), false), 51 * GIB - 1);
    assert_eq!(planning_free_on(51 * GIB + 1, Some(&h), false), 51 * GIB + 1);
    // no host figure: the device's, integrated or not
    assert_eq!(planning_free_on(33 * GIB, None, false), 33 * GIB);
    assert_eq!(planning_free_on(33 * GIB, None, true), 33 * GIB);
}

#[test]
fn a_discrete_device_uses_its_default_floor_but_keeps_an_explicit_override() {
    assert_eq!(floor_bytes_for_device(8 * GIB, false), DISCRETE_MEMORY_FLOOR);
    assert_eq!(floor_bytes_for_device(8 * GIB, true), 8 * GIB);
    std::env::set_var("REXL3_MEMORY_FLOOR_GIB", "2.5");
    assert_eq!(floor_bytes_for_device((5 * GIB) / 2, false), (5 * GIB) / 2);
    std::env::remove_var("REXL3_MEMORY_FLOOR_GIB");
}

#[test]
fn an_integrated_device_keeps_the_host_only_rule_byte_identical() {
    let h = rtx_host();
    // a small device free never binds on the GB10: cudaMalloc takes the page cache back
    assert_eq!(planning_free_on(30 * GIB, Some(&h), true), 116 * GIB);
    // at integrated == true the device-aware rule IS the old rule, to the byte
    assert_eq!(planning_free_on(30 * GIB, Some(&h), true), planning_free(30 * GIB, Some(&h)));
    assert_eq!(planning_free_on(70 * GIB, Some(&h), true), planning_free(70 * GIB, Some(&h)));
}

#[test]
fn the_memory_view_line_names_the_class_and_the_figure_it_planned_from() {
    let h = rtx_host();
    let l = memory_view_line_on(51 * GIB, Some(&h), 8 * GIB, false);
    assert!(l.contains("planned_from=cuda_free_min_mem_available"), "{l}");
    assert!(l.contains("plan_free_gib=51.0"), "{l}");
    assert!(l.contains("cuda_free_gib=51.0"), "{l}");
    let l = memory_view_line_on(51 * GIB, Some(&h), 8 * GIB, true);
    assert!(l.contains("planned_from=mem_available"), "{l}");
    assert!(l.contains("plan_free_gib=116.0"), "{l}");
}

/// Option validation only: the actual leader/follower admission checks live in `serve.rs`'s unit tests.
#[test]
fn a_four_rank_group_validates_rank_two() {
    let tp = TpOptions::new(4, 2, "192.0.2.1:29600");
    assert!(tp.is_group());
    assert!(tp.validate().is_ok());
    assert_eq!((tp.ranks, tp.rank), (4, 2));
}
