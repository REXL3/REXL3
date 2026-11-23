//! What a tensor-parallel GLM rank allocates after its plan that the memory profile does not model row by row, priced BEFORE the load so
//! that the plan equals the allocations within a stated tolerance. Opt-in (`memory::PRICE_UNMODELLED_ENV`, separate from the reserve
//! margin): unset, every term is 0 and every shipped GLM plan is unchanged field for field (the RED-SNOW 524,288-token profile at its
//! 4 GiB margin included).
//!
//! The items (each measured on the RED-SNOW ranks at 1,048,576 tokens, `docs/GLM_TP.md` section 8):
//!
//! - `roce_region`: the RoCE transport's registered region (`cudaHostAlloc`, mapped and pinned), the layout's own size for this rank's site
//!   plan (`gb_cuda::roce::region_bytes` on `tp_boot::roce_site_plan_here`). Charged whenever the transport may come up (`roce` or `auto`):
//!   an `auto` rank whose probe falls back to NCCL allocates less than planned, never more.
//! - `split`: the head split's buffers and the P2 exchange's parts (`model::tp::split_planned_bytes`): the KDA heads' weight copies, the
//!   `o_proj` K-split partials, the gathered halves, the shared-expert split, the parts.
//! - `graphs`: the graph caches at their limits (`model::GraphBudget`) times what one graph is charged at ([`WINDOW_GRAPH_BYTES`],
//!   [`DRAFT_GRAPH_BYTES`]).
//! - `driver`: the driver's bookkeeping of what the rank allocates, [`DRIVER_BYTES_PER_GIB`] of every GiB the plan may allocate (measured as
//!   kernel SUnreclaim + VmallocUsed against 8 GiB of `cudaMalloc` / VMM allocations: the same per GiB at every granularity).
//!
//! Not priced (the margin's job): the per-allocation driver slab (~51 KiB a `cudaMalloc`), the process's own anonymous memory, the
//! communicators' internal buffers and the first requests' lazy allocations.

use super::model::tp::SplitBytes;
use super::model::GraphBudget;

/// What one instantiated window graph is charged at (see `graphs::GraphStats::max_window_graph_bytes`, logged as `event=glm_graph_bytes`).
pub const WINDOW_GRAPH_BYTES: u64 = 24 << 20;
/// What one instantiated NextN draft graph is charged at.
pub const DRAFT_GRAPH_BYTES: u64 = 8 << 20;
/// The driver's bookkeeping per GiB allocated (10 MiB: kernel slab and vmalloc measured against 8 GiB of device allocations on a GB10).
pub const DRIVER_BYTES_PER_GIB: u64 = 10 << 20;

/// The rank's priced extras (all 0 when the switch is off).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RankExtras {
    pub roce_region: u64,
    pub split: SplitBytes,
    pub graphs: u64,
    pub driver: u64,
}

impl RankExtras {
    pub fn total(&self) -> u64 {
        self.roce_region + self.split.total() + self.graphs + self.driver
    }

    /// The `rexl3 event=tp_rank_priced` line: every term in MiB (three decimals) and the total.
    pub fn line(&self, rank: usize) -> String {
        let m = |b: u64| b as f64 / (1u64 << 20) as f64;
        let s = &self.split;
        format!(
            "rexl3 event=tp_rank_priced family=glm rank={rank} roce_region_mib={:.3} kda_copies_mib={:.3} partials_mib={:.3} halves_mib={:.3} shared_split_mib={:.3} p2_parts_mib={:.3} graphs_mib={:.3} driver_mib={:.3} total_mib={:.3}",
            m(self.roce_region),
            m(s.kda_copies),
            m(s.partials),
            m(s.halves),
            m(s.shared),
            m(s.p2_parts),
            m(self.graphs),
            m(self.driver),
            m(self.total())
        )
    }
}

/// The graph caches at their limits: `budget` (the default limits when `None`) times the per-graph charge.
pub fn graph_bytes(budget: Option<GraphBudget>) -> u64 {
    let b = budget.unwrap_or(GraphBudget::DEFAULT);
    b.windows as u64 * WINDOW_GRAPH_BYTES + b.drafts as u64 * DRAFT_GRAPH_BYTES
}

/// The driver's bookkeeping of `allocatable` bytes (what the plan may allocate: the free memory before the load less the floor).
pub fn driver_bytes(allocatable: u64) -> u64 {
    ((allocatable as u128 * DRIVER_BYTES_PER_GIB as u128) >> 30) as u64
}

/// The `rexl3 event=tp_rank_allocated` line after the load: what was priced against what the rank holds, item by item (the RoCE region from
/// the transport, the split's buffers from the model), and the difference (`over_mib` > 0: allocated past the price; logged at `warn`).
pub fn allocated_line(rank: usize, priced: &RankExtras, roce_region: Option<u64>, split: Option<SplitBytes>) -> String {
    let m = |b: u64| b as f64 / (1u64 << 20) as f64;
    let s = split.unwrap_or_default();
    let roce = roce_region.unwrap_or(0);
    let (got, want) = (roce + s.total(), priced.roce_region + priced.split.total());
    let over = got as i128 - want as i128;
    format!(
        "rexl3 {}event=tp_rank_allocated family=glm rank={rank} roce_region_mib={:.3} kda_copies_mib={:.3} partials_mib={:.3} halves_mib={:.3} shared_split_mib={:.3} p2_parts_mib={:.3} allocated_mib={:.3} priced_mib={:.3} over_mib={:.3}",
        if over > 0 { "level=warn " } else { "" },
        m(roce),
        m(s.kda_copies),
        m(s.partials),
        m(s.halves),
        m(s.shared),
        m(s.p2_parts),
        m(got),
        m(want),
        over as f64 / (1u64 << 20) as f64
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn the_terms_add_up_and_the_line_names_each() {
        let e = RankExtras {
            roce_region: 300 * MIB,
            split: SplitBytes { kda_copies: 3 * MIB, partials: 24 * MIB, halves: 38 * MIB, shared: 2 * MIB, p2_parts: 64 * MIB },
            graphs: 368 * MIB,
            driver: 1100 * MIB,
        };
        assert_eq!(e.total(), (300 + 3 + 24 + 38 + 2 + 64 + 368 + 1100) * MIB);
        let l = e.line(0);
        for want in [
            "event=tp_rank_priced",
            "rank=0",
            "roce_region_mib=300.000",
            "p2_parts_mib=64.000",
            "graphs_mib=368.000",
            "total_mib=1899.000",
        ] {
            assert!(l.contains(want), "{want:?} missing from {l}");
        }
        assert_eq!(RankExtras::default().total(), 0);
    }

    #[test]
    fn the_graph_and_driver_charges() {
        // the one-slot budget with the NextN layer: 10 window graphs and 16 draft graphs
        assert_eq!(graph_bytes(Some(GraphBudget::one_slot(4))), 10 * WINDOW_GRAPH_BYTES + 16 * DRAFT_GRAPH_BYTES);
        assert_eq!(graph_bytes(None), 96 * WINDOW_GRAPH_BYTES + 32 * DRAFT_GRAPH_BYTES, "no budget: the default limits");
        assert_eq!(driver_bytes(1 << 30), 10 * MIB);
        assert_eq!(driver_bytes(113 << 30), 1130 * MIB);
    }

    #[test]
    fn the_allocated_line_warns_only_past_the_price() {
        let priced = RankExtras {
            roce_region: 100 * MIB,
            split: SplitBytes { p2_parts: 64 * MIB, ..Default::default() },
            ..Default::default()
        };
        let exact = allocated_line(1, &priced, Some(100 * MIB), Some(SplitBytes { p2_parts: 64 * MIB, ..Default::default() }));
        assert!(exact.contains("over_mib=0.000") && !exact.contains("level=warn"), "{exact}");
        let under = allocated_line(1, &priced, None, Some(SplitBytes { p2_parts: 64 * MIB, ..Default::default() }));
        assert!(
            under.contains("over_mib=-100.000") && !under.contains("level=warn"),
            "a fallback to NCCL is under the price: {under}"
        );
        let over = allocated_line(1, &priced, Some(101 * MIB), Some(SplitBytes { p2_parts: 64 * MIB, ..Default::default() }));
        assert!(over.contains("level=warn") && over.contains("over_mib=1.000"), "{over}");
    }
}
