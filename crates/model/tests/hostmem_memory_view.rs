use gb_model::hostmem::{memory_view_exact_line, MemInfo};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

/// The rank's exact line exposes the reading, the whole-GiB plan, and the
/// remainder omitted by rounding. This integration-test path also keeps the
/// behavior covered by the repository's test-with-code PR rule.
#[test]
fn the_exact_memory_view_line_shows_the_rounding_remainder() {
    let m = MemInfo { total: 128 * GIB, free: 100 * GIB, available: 115 * GIB + 789 * MIB + 512 * 1024, cached: 2 * GIB };
    let line = memory_view_exact_line(30 * GIB, Some(&m), 2 * GIB);
    for want in [
        "event=memory_view_exact",
        "cuda_free_mib=30720.000",
        "mem_available_mib=118549.500",
        "planned_from=mem_available",
        "plan_free_mib=117760.000",
        "rounding_remainder_mib=789.500",
        "floor_mib=2048.000",
    ] {
        assert!(line.contains(want), "{want:?} missing from: {line}");
    }

    let whole = MemInfo { available: 115 * GIB, ..m };
    assert!(memory_view_exact_line(30 * GIB, Some(&whole), 2 * GIB).contains("rounding_remainder_mib=0.000"));

    let line = memory_view_exact_line(30 * GIB + 3 * MIB, None, 2 * GIB);
    assert!(
        line.contains("planned_from=cuda_free")
            && line.contains("rounding_remainder_mib=3.000")
            && !line.contains("mem_available"),
        "{line}"
    );
}
