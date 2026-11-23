//! The host's account of memory: the figure every memory plan is sized from.
//!
//! The CPU and the GPU of a GB10 draw from one pool, and the two sides count it differently. `cudaMemGetInfo` free is the kernel's `MemFree`: it
//! leaves out the page cache, so after a cargo build, a dump or a buffered read of a pack it can be 12 GiB on a box whose `MemAvailable` is 113.
//! The plan used to be sized from the smaller of the two, so what a server could hold (262,144 KV tokens, the prefix cache, GLM's slots) depended
//! on what the node had done before the boot. The premise behind that (`cudaMalloc` does not take clean cache pages back) is false on this box:
//! `cudaMalloc` is an eager kernel page allocation, direct reclaim runs for it, and it drained 82 GiB of clean, dirty, active and mapped file pages
//! while it allocated 100 GiB (`docs/EVAL_RESULTS.md`, "Memory: capacity that does not depend on the boot history"; `tools/alloc_probe.cu`
//! reproduces it). So what an allocation can get is what the kernel says is available, `MemAvailable`, and that is what the plans use; the
//! device's figure is logged and nothing else. The floor, the engine reserve and the slot state still come out of it exactly once
//! (`plan::plan_memory_cached`), the loader keeps checking it before each region (`families::glm::load::allocate`), and an allocation the
//! driver refuses anyway ends the boot with [`allocation_refusal`], never with a bare CUDA code.
//!
//! That is the unified-memory rule (`cudaDevAttrIntegrated` == 1, the GB10). On a discrete GPU the device has its own VRAM and
//! `cudaMalloc` cannot reclaim the host's page cache, so a plan sized from `MemAvailable` alone can ask the device for more than it has
//! (the boot then dies in `cudaMalloc`). The device-aware [`planning_free_on`] keeps the host's figure binding on an integrated device and
//! sizes from the smaller of the device's free and the host's on a discrete one.

use std::fmt::Display;

/// The lines of `/proc/meminfo` the plans and their logs use, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemInfo {
    pub total: u64,
    /// `MemFree`: what `cudaMemGetInfo` calls free.
    pub free: u64,
    /// `MemAvailable`: the kernel's estimate of what an allocation can get without swapping, the reclaimable page cache included.
    pub available: u64,
    /// `Cached`: the page cache (and tmpfs).
    pub cached: u64,
}

impl MemInfo {
    /// Parse the text of `/proc/meminfo`. `None` when `MemAvailable` or `MemFree` is missing (a kernel that does not report them gives no
    /// figure to plan from, and the caller falls back to the device's).
    pub fn parse(text: &str) -> Option<MemInfo> {
        let kb = |key: &str| -> Option<u64> {
            let l = text.lines().find(|l| l.strip_prefix(key).is_some_and(|r| r.starts_with(':')))?;
            l.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb * 1024)
        };
        Some(MemInfo {
            total: kb("MemTotal").unwrap_or(0),
            free: kb("MemFree")?,
            available: kb("MemAvailable")?,
            cached: kb("Cached").unwrap_or(0),
        })
    }

    /// The host's figures now.
    #[cfg(not(target_os = "macos"))]
    pub fn read() -> Option<MemInfo> {
        std::fs::read_to_string("/proc/meminfo").ok().and_then(|t| MemInfo::parse(&t))
    }

    /// macOS has no `/proc/meminfo`. The same three figures from the VM counters: `free` is free pages, `cached` the file-backed
    /// (external) pageable pages plus speculative and purgeable ones, and `available` their sum -- what the kernel hands back on
    /// demand, the analogue of Linux `MemAvailable`. On Apple Silicon this memory is the GPU's too (unified), so the device
    /// plan and the host plan read the same pool.
    #[cfg(target_os = "macos")]
    pub fn read() -> Option<MemInfo> {
        fn sysctl_u64(name: &str) -> Option<u64> {
            let c = std::ffi::CString::new(name).ok()?;
            let mut v: u64 = 0;
            let mut n = std::mem::size_of::<u64>();
            // SAFETY: sysctlbyname writes at most `n` bytes into `v`; integers narrower than 8 bytes are zero-extended below
            let rc = unsafe { libc::sysctlbyname(c.as_ptr(), (&mut v as *mut u64).cast(), &mut n, std::ptr::null_mut(), 0) };
            (rc == 0).then_some(if n == 4 { v & 0xffff_ffff } else { v })
        }
        let page = sysctl_u64("hw.pagesize")?;
        let pages = |k: &str| sysctl_u64(k).map(|p| p * page);
        let free = pages("vm.page_free_count")?;
        let cached = pages("vm.page_pageable_external_count").unwrap_or(0)
            + pages("vm.page_speculative_count").unwrap_or(0)
            + pages("vm.page_purgeable_count").unwrap_or(0);
        Some(MemInfo { total: sysctl_u64("hw.memsize").unwrap_or(0), free, available: free + cached, cached })
    }

    /// What `MemAvailable` counts and `MemFree` does not: memory the kernel gives back on demand (clean page cache, reclaimable slab).
    pub fn reclaimable(&self) -> u64 {
        self.available.saturating_sub(self.free)
    }
}

/// The plan is sized from a whole number of these: `MemAvailable` moves by a few hundred MiB between two boots of the same box in the same state (the
/// server's own start-up reads, the other tenants of the box: 16.2-17.2 GiB in fourteen cold GLM boots on anonymized host-b), and a plan that follows it to the byte moves
/// with it (GLM's pool at four slots is 77,824 tokens at 17.0 GiB and nothing at 16.0). Rounding down to a GiB makes boots that read within one GiB of each
/// other one boot. The cost is the rounding itself: up to a GiB of what the kernel reports is not planned, so a pinned `--slots N` that fit at 16.3 GiB is
/// refused at 16.0 (the refusal names the count that fits); `docs/EVAL_RESULTS.md` "Memory" says so and leaves the choice of step to the integrator.
/// (An earlier plan that clamped to the device's free moved by the 0.4 GiB between three such boots from 17,664 to 51,456 tokens; that clamp is gone.)
pub const PLAN_QUANTUM: u64 = 1 << 30;

/// The bytes a plan is sized from once the weights are resident: the host's `MemAvailable`, or the device's own free figure where the host reports
/// none, rounded down to [`PLAN_QUANTUM`]. Never the smaller of the two: the device's figure only misses the page cache, which `cudaMalloc` reclaims
/// on demand. This is the integrated-device rule (the GB10): [`planning_free_on`] is the device-aware variant a discrete-GPU family must use.
pub fn planning_free(cuda_free: u64, host: Option<&MemInfo>) -> u64 {
    planning_free_on(cuda_free, host, true)
}

/// [`planning_free`] for an explicit device class. `integrated` is `cudaDevAttrIntegrated` == 1 (the device shares the host's memory, as on the
/// GB10): a plan is sized from the host's `MemAvailable` alone, and the device's own free figure is only the kernel's `MemFree`, which is not the
/// bound. On a discrete device (`integrated` == false) the device has its own VRAM: `cudaMalloc` cannot reclaim the host's page cache, so a plan
/// sized from `MemAvailable` can ask the device for more than it has and die in allocation. The plan is sized from the smaller of the device's
/// free bytes and the host's `MemAvailable`. The GiB rounding of [`PLAN_QUANTUM`] exists for the host-side jitter of the unified-memory rule
/// (whole-GiB swings of `MemAvailable` between boots of one state); a discrete GPU's free figure is its VRAM and is exact, and rounding it down
/// throws away up to a GiB of the small headroom a 24/32/48 GiB card has left after the weights (the 24 GiB card of the small-fit plan: free
/// 2.69 GiB -> planned 2.00), so the discrete path keeps the exact bytes. Byte-identical to [`planning_free`] at `integrated` == true.
pub fn planning_free_on(cuda_free: u64, host: Option<&MemInfo>, integrated: bool) -> u64 {
    let free = match (host, integrated) {
        (Some(h), true) => h.available,
        (Some(h), false) => h.available.min(cuda_free),
        (None, _) => cuda_free,
    };
    if integrated {
        free - free % PLAN_QUANTUM
    } else {
        free
    }
}

/// The integrated-device reading of [`planning_free`] without the [`PLAN_QUANTUM`] rounding: the host's `MemAvailable` to the byte, or the
/// device's free figure where the host reports none. For a family whose plan is a few GiB wide on a full box, where the up-to-a-GiB the
/// rounding leaves unplanned decides whether the shipped window fits at all (Ling-3.1 on one GB10: 6-10 GiB after the 106 GiB pack). Such a
/// plan moves with `MemAvailable` between boots; the family must refuse a pool short of its window rather than boot smaller.
pub fn planning_free_exact(cuda_free: u64, host: Option<&MemInfo>) -> u64 {
    match host {
        Some(h) => h.available,
        None => cuda_free,
    }
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// The `rexl3 event=memory_view` line: what the plan was sized from and what the device would have said, so a boot's capacity can be read against
/// the box's state (the A/B of `docs/EVAL_RESULTS.md` is made of these lines). The integrated-device form ([`planning_free`]).
pub fn memory_view_line(cuda_free: u64, host: Option<&MemInfo>, floor: u64) -> String {
    memory_view_line_on(cuda_free, host, floor, true)
}

/// [`memory_view_line`] for an explicit device class (see [`planning_free_on`] for `integrated`).
pub fn memory_view_line_on(cuda_free: u64, host: Option<&MemInfo>, floor: u64, integrated: bool) -> String {
    let planned = planning_free_on(cuda_free, host, integrated);
    let from = if integrated { "mem_available" } else { "cuda_free_min_mem_available" };
    match host {
        Some(h) => format!(
            "rexl3 event=memory_view cuda_free_gib={:.1} mem_free_gib={:.1} mem_available_gib={:.1} cached_gib={:.1} planned_from={from} plan_free_gib={:.1} floor_gib={:.1}",
            gib(cuda_free),
            gib(h.free),
            gib(h.available),
            gib(h.cached),
            gib(planned),
            gib(floor)
        ),
        None => format!(
            "rexl3 level=warn event=memory_view cuda_free_gib={:.1} planned_from=cuda_free reason=\"/proc/meminfo has no MemAvailable\" plan_free_gib={:.1} floor_gib={:.1}",
            gib(cuda_free),
            gib(planned),
            gib(floor)
        ),
    }
}

/// [`memory_view_line`] for a plan sized by [`planning_free_exact`]: the same fields, with `plan_free_gib` the unrounded figure (two decimals)
/// and `planned_from=mem_available_exact`, so the line states what the plan was really sized from.
pub fn memory_view_line_unrounded(cuda_free: u64, host: Option<&MemInfo>, floor: u64) -> String {
    let planned = planning_free_exact(cuda_free, host);
    match host {
        Some(h) => format!(
            "rexl3 event=memory_view cuda_free_gib={:.1} mem_free_gib={:.1} mem_available_gib={:.2} cached_gib={:.1} planned_from=mem_available_exact plan_free_gib={:.2} floor_gib={:.1}",
            gib(cuda_free),
            gib(h.free),
            gib(h.available),
            gib(h.cached),
            gib(planned),
            gib(floor)
        ),
        None => format!(
            "rexl3 level=warn event=memory_view cuda_free_gib={:.1} planned_from=cuda_free reason=\"/proc/meminfo has no MemAvailable\" plan_free_gib={:.2} floor_gib={:.1}",
            gib(cuda_free),
            gib(planned),
            gib(floor)
        ),
    }
}

/// The `rexl3 event=memory_view_exact` line of a tensor-parallel rank's plan (`glm::tp_boot::plan_rank`): the same reading as
/// [`memory_view_line`] in MiB with three decimals, and what the whole-GiB rounding of [`planning_free`] left out of the plan
/// (`rounding_remainder_mib`: free memory the plan never counts), so a rank's allocations can be read against the plan to the MiB. The
/// single node's `memory_view` line is unchanged.
pub fn memory_view_exact_line(cuda_free: u64, host: Option<&MemInfo>, floor: u64) -> String {
    let planned = planning_free(cuda_free, host);
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    let (from, read) = match host {
        Some(h) => ("mem_available", h.available),
        None => ("cuda_free", cuda_free),
    };
    let host_part = host.map_or(String::new(), |h| {
        format!(" mem_free_mib={:.3} mem_available_mib={:.3} cached_mib={:.3}", mib(h.free), mib(h.available), mib(h.cached))
    });
    format!(
        "rexl3 event=memory_view_exact cuda_free_mib={:.3}{host_part} planned_from={from} plan_free_mib={:.3} rounding_remainder_mib={:.3} floor_mib={:.3}",
        mib(cuda_free),
        mib(planned),
        mib(read - planned),
        mib(floor)
    )
}

/// The boot error for an allocation the device refused although the plan fits: what was being allocated, what the plan asked for, and what the host
/// had when the plan was made and has now, so the operator can tell memory taken by another process (the usual cause) from a driver that does not
/// reclaim. `what` names the allocation ("the engine's buffers", "the prefix cache").
pub fn allocation_refusal(
    what: &str,
    err: &dyn Display,
    planned_bytes: u64,
    floor: u64,
    at_plan: Option<&MemInfo>,
    now: Option<&MemInfo>,
) -> String {
    let (Some(then), Some(now)) = (at_plan, now) else {
        return format!(
            "could not allocate {what} ({:.2} GiB planned): {err}; free memory on this box, or start with fewer --slots or a smaller --kv-tokens",
            gib(planned_bytes)
        );
    };
    let cause = if now.available < planned_bytes.saturating_add(floor) {
        format!(
            "another process took memory in between: it is {:.1} GiB short of the plan and the {:.1} GiB floor",
            gib(planned_bytes.saturating_add(floor) - now.available),
            gib(floor)
        )
    } else {
        "the host still reports enough, so the driver refused memory the kernel says it can reclaim; run tools/alloc_probe.cu to check that cudaMalloc still takes the page cache back".to_string()
    };
    format!(
        "could not allocate {what} ({:.2} GiB planned): {err}; MemAvailable was {:.1} GiB when the plan was made and is {:.1} GiB now (MemFree {:.1}, cache {:.1}): {cause}; free memory on this box, or start with fewer --slots or a smaller --kv-tokens",
        gib(planned_bytes),
        gib(then.available),
        gib(now.available),
        gib(now.free),
        gib(now.cached)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    /// anonymized host-c after 83 GiB of buffered reads (2026-10-03): the device sees 30 GiB free, the kernel 113.6 GiB available.
    const DIRTY: &str = "MemTotal:       127600812 kB\nMemFree:        31498240 kB\nMemAvailable:  119132160 kB\nBuffers:          580808 kB\nCached:         86212608 kB\nSwapCached:            0 kB\n";

    #[test]
    fn meminfo_is_read_in_bytes_by_exact_key() {
        let m = MemInfo::parse(DIRTY).unwrap();
        assert_eq!(m.total, 127_600_812 * 1024);
        assert_eq!(m.free, 31_498_240 * 1024);
        assert_eq!(m.available, 119_132_160 * 1024);
        assert_eq!(m.cached, 86_212_608 * 1024, "Cached, not SwapCached");
        assert_eq!(m.reclaimable(), (119_132_160 - 31_498_240) * 1024);
        // a key that only starts like the one asked for is not it
        let t = "MemFreeable: 5 kB\nMemFree: 7 kB\nMemAvailable: 9 kB\n";
        assert_eq!(MemInfo::parse(t).unwrap().free, 7 * 1024);
        // no MemAvailable (an old kernel, another OS): no figure, the caller falls back to the device's
        assert_eq!(MemInfo::parse("MemTotal: 1 kB\nMemFree: 1 kB\n"), None);
        assert_eq!(MemInfo::parse(""), None);
    }

    #[test]
    fn the_plan_is_sized_from_what_the_kernel_can_give_not_from_the_devices_free() {
        // after buffered I/O: cudaMemGetInfo free 12 GiB (MemFree), MemAvailable 94 GiB. The old rule planned 12 GiB, a clean boot 94 GiB.
        let dirty = MemInfo { total: 121 * GIB, free: 12 * GIB, available: 94 * GIB, cached: 80 * GIB };
        let clean = MemInfo { total: 121 * GIB, free: 87 * GIB, available: 94 * GIB, cached: 8 * GIB };
        assert_eq!(planning_free(dirty.free, Some(&dirty)), 94 * GIB);
        assert_eq!(planning_free(clean.free, Some(&clean)), 94 * GIB);
        assert_eq!(
            planning_free(dirty.free, Some(&dirty)),
            planning_free(clean.free, Some(&clean)),
            "boot history does not move the plan"
        );
        // the device may even see more than the host (nothing to reclaim against): the host's figure still binds
        assert_eq!(planning_free(100 * GIB, Some(&MemInfo { available: 40 * GIB, ..clean })), 40 * GIB);
        // no host figure: the device's
        assert_eq!(planning_free(33 * GIB, None), 33 * GIB);
    }

    #[test]
    fn a_discrete_device_plans_from_the_smaller_of_the_device_and_the_host() {
        // RTX-class box after the weights are resident: the device reports 51 GiB free of its own VRAM,
        // the host still reports 116 GiB of MemAvailable. A plan sized from MemAvailable alone would ask
        // the device for ~115 GiB and die in cudaMalloc; the discrete rule binds the device's figure.
        let h = MemInfo { total: 121 * GIB, free: 12 * GIB, available: 116 * GIB, cached: 80 * GIB };
        assert_eq!(planning_free_on(51 * GIB, Some(&h), false), 51 * GIB);
        // the host is the bound when it is the smaller figure
        assert_eq!(planning_free_on(70 * GIB, Some(&h), false), 70 * GIB);
        // the discrete rule keeps the exact bytes: the GiB rounding would cost up to a GiB of the small
        // headroom a 24/32/48 GiB card has after the weights (free 2.69 GiB -> planned 2.00 alone costs
        // the 24 GiB card its whole boot at the new floor)
        assert_eq!(planning_free_on(51 * GIB - 1, Some(&h), false), 51 * GIB - 1);
        assert_eq!(planning_free_on(51 * GIB + 1, Some(&h), false), 51 * GIB + 1);
        // no host figure: the device's, integrated or not
        assert_eq!(planning_free_on(33 * GIB, None, false), 33 * GIB);
        assert_eq!(planning_free_on(33 * GIB, None, true), 33 * GIB);
    }

    #[test]
    fn an_integrated_device_keeps_the_host_only_rule_byte_identical() {
        // the GB10: cudaMalloc reclaims the host's page cache, so the device's free figure never binds
        let h = MemInfo { total: 121 * GIB, free: 12 * GIB, available: 116 * GIB, cached: 80 * GIB };
        assert_eq!(planning_free_on(30 * GIB, Some(&h), true), 116 * GIB);
        // at integrated == true the device-aware rule IS the old rule, to the byte
        assert_eq!(planning_free_on(30 * GIB, Some(&h), true), planning_free(30 * GIB, Some(&h)));
        assert_eq!(planning_free_on(70 * GIB, Some(&h), true), planning_free(70 * GIB, Some(&h)));
    }

    #[test]
    fn the_sub_gib_jitter_between_boots_of_one_state_does_not_move_the_plan() {
        // GLM after its weights on anonymized host-b, three boots of the same state (2026-10-03): MemAvailable 16.3, 16.6 and 16.7 GiB
        let at = |avail_mib: u64| MemInfo { total: 121 * GIB, free: 10 * GIB, available: avail_mib << 20, cached: 2 * GIB };
        let plans: Vec<u64> = [16_690, 17_000, 17_100].iter().map(|&m| planning_free(0, Some(&at(m)))).collect();
        assert_eq!(plans, vec![16 * GIB; 3]);
        // the rounding is down (never plans memory the host did not report) and exact at a boundary
        assert_eq!(planning_free(0, Some(&at(17 * 1024 - 1))), 16 * GIB);
        assert_eq!(planning_free(0, Some(&at(17 * 1024))), 17 * GIB);
        assert_eq!(planning_free(GIB - 1, None), 0);
    }

    #[test]
    fn a_refusal_says_what_was_planned_and_what_changed() {
        let then = MemInfo { total: 121 * GIB, free: 5 * GIB, available: 60 * GIB, cached: 50 * GIB };
        let took = MemInfo { total: 121 * GIB, free: GIB, available: 20 * GIB, cached: 15 * GIB };
        let t =
            allocation_refusal("the engine's buffers", &"cudaErrorMemoryAllocation", 40 * GIB, 8 * GIB, Some(&then), Some(&took));
        for want in [
            "the engine's buffers",
            "40.00 GiB planned",
            "cudaErrorMemoryAllocation",
            "was 60.0 GiB when the plan was made and is 20.0 GiB now",
            "another process took memory",
            "28.0 GiB short",
            "--slots",
            "--kv-tokens",
        ] {
            assert!(t.contains(want), "{want:?} missing from: {t}");
        }
        // the host still has the memory: the driver is what refused, and the text says how to check it
        let ok = MemInfo { available: 70 * GIB, ..took };
        let t = allocation_refusal("the prefix cache", &"cudaErrorMemoryAllocation", 8 * GIB, 8 * GIB, Some(&then), Some(&ok));
        assert!(t.contains("the driver refused") && t.contains("alloc_probe") && !t.contains("another process"), "{t}");
        // without the host's figures the message still names the plan and the knobs
        let t = allocation_refusal("the engine's buffers", &"cudaErrorMemoryAllocation", 8 * GIB, 8 * GIB, None, None);
        assert!(t.contains("8.00 GiB planned") && t.contains("--slots"), "{t}");
    }

    #[test]
    fn the_memory_view_line_names_both_figures_and_what_the_plan_used() {
        let m = MemInfo::parse(DIRTY).unwrap();
        let l = memory_view_line(30 * GIB, Some(&m), 8 * GIB);
        assert!(
            l.contains("cuda_free_gib=30.0")
                && l.contains("mem_available_gib=113.6")
                && l.contains("planned_from=mem_available")
                && l.contains("plan_free_gib=113.0"),
            "{l}"
        );
        let l = memory_view_line(30 * GIB, None, 8 * GIB);
        assert!(l.contains("planned_from=cuda_free") && l.contains("plan_free_gib=30.0") && l.contains("level=warn"), "{l}");
    }
}
