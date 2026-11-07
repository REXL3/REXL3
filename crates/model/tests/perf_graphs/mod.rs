//! The graph switches of the perf harness, in one small file so that the same `glm_perf.rs` builds on a commit that has no graphs (the baseline
//! copy of this file there answers `None` and does nothing).
#![allow(dead_code)]

use gb_model::families::glm::model::GlmModel;

pub struct Stats {
    pub captures: u64,
    pub replays: u64,
    pub eager: u64,
    pub failures: u64,
    pub capture_ms: f64,
    pub last_failure: Option<String>,
}

pub fn set(m: &mut GlmModel, on: bool) {
    m.set_graphs(on);
}

/// Set a deliberate bug of the graph path by name (`stale_rows`, `stale_table`, `wrong_shape`, `sync_in_capture`), or clear it.
pub fn set_mutant(m: &mut GlmModel, name: Option<&str>) {
    use gb_model::families::glm::model::{GraphMutant, Wait};
    m.set_graph_mutant(name.map(|n| match n {
        "stale_rows" => GraphMutant::StaleRows,
        "stale_table" => GraphMutant::StaleTable,
        "wrong_shape" => GraphMutant::WrongShape,
        "sync_in_capture" => GraphMutant::SyncInCapture,
        "slow_side" => GraphMutant::SlowSide,
        "join_late" => GraphMutant::JoinLate,
        "skip_latent" => GraphMutant::SkipWait(Wait::Latent),
        "skip_pool" => GraphMutant::SkipWait(Wait::Pool),
        "skip_q" => GraphMutant::SkipWait(Wait::Q),
        "skip_select" => GraphMutant::SkipWait(Wait::Select),
        other => panic!("{other} is not a graph mutant"),
    }));
}

pub fn clear(m: &mut GlmModel) {
    m.clear_graphs().unwrap();
}

pub fn is_on(m: &GlmModel) -> bool {
    m.graphs_on()
}

pub fn stats(m: &GlmModel) -> Option<Stats> {
    let s = m.graph_stats();
    Some(Stats {
        captures: s.captures,
        replays: s.replays,
        eager: s.eager_windows,
        failures: s.capture_failures,
        capture_ms: s.capture_ms,
        last_failure: s.last_failure,
    })
}
