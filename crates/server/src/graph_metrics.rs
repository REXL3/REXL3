//! `/metrics` text for a backend's CUDA-graph cache (`gb_engine::GraphCounters`): replays (hits), eager windows and captures (misses),
//! evictions, failed captures, capture time, and the draft head's own cache. Emitted only when the engine reports a graph cache, so a
//! backend without one (the mock, a CPU family) serves exactly the text it served before.

use gb_engine::GraphCounters;
use std::fmt::Write;

fn put(out: &mut String, name: &str, kind: &str, help: &str, v: impl std::fmt::Display) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}");
}

/// Prometheus text for the graph cache. Counters only grow; on a tensor-parallel group they are rank 0's.
pub fn render(g: &GraphCounters) -> String {
    let mut out = String::new();
    let c = "counter";
    put(&mut out, "rexl3_graph_cache_replays_total", c, "Windows that replayed a captured CUDA graph (hits).", g.replays);
    put(
        &mut out,
        "rexl3_graph_cache_captures_total",
        c,
        "Windows captured into a CUDA graph (a miss on the second sighting).",
        g.captures,
    );
    put(
        &mut out,
        "rexl3_graph_cache_eager_windows_total",
        c,
        "Windows that ran eagerly (a first sighting, a shape marked eager, a failed capture).",
        g.eager_windows,
    );
    put(
        &mut out,
        "rexl3_graph_cache_evictions_total",
        c,
        "Captured graphs destroyed to stay under the cache's limit (least recently used first).",
        g.evictions,
    );
    put(
        &mut out,
        "rexl3_graph_cache_capture_failures_total",
        c,
        "Captures that failed; the shape then runs eagerly.",
        g.capture_failures,
    );
    put(&mut out, "rexl3_graph_cache_capture_seconds_total", c, "Time spent capturing graphs.", g.capture_ms / 1e3);
    put(
        &mut out,
        "rexl3_draft_graph_cache_replays_total",
        c,
        "Draft-head calls that replayed a captured graph.",
        g.draft_replays,
    );
    put(&mut out, "rexl3_draft_graph_cache_captures_total", c, "Draft-head calls captured into a graph.", g.draft_captures);
    put(&mut out, "rexl3_draft_graph_cache_eager_total", c, "Draft-head calls that ran eagerly.", g.draft_eager);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_counter_is_one_series_with_its_value() {
        let g = GraphCounters {
            replays: 7,
            captures: 2,
            eager_windows: 3,
            capture_failures: 1,
            evictions: 4,
            capture_ms: 1500.0,
            draft_replays: 11,
            draft_captures: 5,
            draft_eager: 6,
        };
        let t = render(&g);
        for (name, v) in [
            ("rexl3_graph_cache_replays_total", "7"),
            ("rexl3_graph_cache_captures_total", "2"),
            ("rexl3_graph_cache_eager_windows_total", "3"),
            ("rexl3_graph_cache_evictions_total", "4"),
            ("rexl3_graph_cache_capture_failures_total", "1"),
            ("rexl3_graph_cache_capture_seconds_total", "1.5"),
            ("rexl3_draft_graph_cache_replays_total", "11"),
            ("rexl3_draft_graph_cache_captures_total", "5"),
            ("rexl3_draft_graph_cache_eager_total", "6"),
        ] {
            assert!(t.contains(&format!("\n{name} {v}\n")), "{name} {v} missing from:\n{t}");
            assert_eq!(t.matches(&format!("# TYPE {name} counter")).count(), 1, "{name}");
        }
    }
}
