//! CUDA graphs of the Qwen decode window: one captured graph per fixed window shape, replayed with new inputs.
//!
//! A window (`Model::window_candidates`: the layers, the head and the top-k) is the same list of launches every
//! round; only what the kernels READ from device memory changes between two windows of one shape: the token ids,
//! positions, row slots, the sequence layout (`upload_rows` writes it into the model's own buffers before every
//! capture and replay), the page table, the owed-replay counts, the recurrent state and the KV cache. A graph
//! bakes in every other argument. The audit of what the captured region hands a launcher that is NOT read from a
//! device buffer:
//!
//! | host value | what it decides | in the key |
//! |---|---|---|
//! | `rows` | every launch's row count and grid | key |
//! | `n_seq` | the GDN kernels' sequence count (`seq_start` / `seq_slot` are device buffers) | key |
//! | `count` | the top-k's candidate count | key |
//! | `par` | which of the two record sets (`recs[par]`) the window fills: the sets alternate every window | key |
//! | `owed_prev` | whether the GDN window kernel folds the previous window's accepted rows in (`window_owed` reads `owed_row0` / `owed_keep` from device memory) | key |
//!
//! Everything else the region touches is constant for the model's life (`max_context`, `vocab`, the layer shapes,
//! `pools.slots`, the buffer addresses). The preamble — the previous window's `flush`, `upload_rows`,
//! `ensure_pages` (its page-table upload) and `fold_owed` (the standalone replay of sequences that sit the window
//! out, and the upload of the owed counts) — stays outside the graph: it is host work and synchronous copies, and
//! a capture would refuse it. So the captured region is exactly `launch_layers` + the head + the top-k, the part
//! that recurs with a fixed shape, which is the scope of this cache (tree windows — a batch with `parents` — and
//! every window under `REXL3_TREE_KERNELS=all` stay eager; their shapes are the depth track's).
//!
//! **Policy.** The first window of a shape runs eagerly (it also warms the kernels' lazy one-time setup, such as
//! the attention decode's `cudaFuncSetAttribute`), the second captures it and replays it, every later one
//! replays. A capture that fails for any non-sticky reason (the runtime refusing a call, a launcher's argument
//! error) marks the shape eager for the rest of the process and runs the window eagerly: a capture never fails a
//! request and never latches the backend fatal; only a sticky device error does, exactly as for a launch. The
//! cache is bounded ([`GRAPH_LIMIT`] instantiated graphs, least recently used out), and a graph is destroyed only
//! after the stream has drained.
//!
//! This module is the decision layer, deliberately free of anything device-shaped so the CPU suite can pin it;
//! `forward.rs` owns the capture itself (`Model::capture_window_graph`).

use crate::rows::RowBatch;
use std::collections::HashMap;

/// Instantiated graphs kept at once.
pub const GRAPH_LIMIT: usize = 64;

/// The environment variable that switches the window graphs: `0` / `off` is off, anything else (and unset) is on.
pub const ENV: &str = "REXL3_QWEN_GRAPHS";

/// `REXL3_QWEN_GRAPH_AT=0` captures a window shape on its FIRST sighting instead of its second (a test knob: the
/// whole GPU test battery then runs through captured graphs, the first window of every shape included).
pub const ENV_AT: &str = "REXL3_QWEN_GRAPH_AT";

/// What identifies a captured window: every host-side value that reaches a launcher of the captured region and is
/// not read from a device buffer (see the module table).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GraphKey {
    pub rows: usize,
    pub n_seq: usize,
    /// The record set the window fills (`recs[par]`): the two sets alternate every window.
    pub par: usize,
    /// Whether the window folds the previous window's owed replay in.
    pub owed: bool,
    /// The top-k's candidate count.
    pub count: usize,
}

impl GraphKey {
    pub fn of(b: &RowBatch, count: usize, par: usize, owed: bool) -> Self {
        Self { rows: b.rows, n_seq: b.n_seq(), par, owed, count }
    }
}

/// A deliberate bug of the graph path, so the tests can show they notice it (`Model::set_window_graph_mutant`).
/// Never set in serving.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowGraphMutant {
    /// A replay does not write this window's token ids, positions, slots and layout into the device buffers: the
    /// graph reads the previous window's.
    StaleRows,
    /// A host synchronisation inside the capture (the capture must refuse, fall back to eager launches and never
    /// fail the window).
    SyncInCapture,
}

/// One shape's place in the cache. `G` is the instantiated graph (the model's `gb_cuda::graph::Graph`; `()` in the
/// CPU tests of this decision layer).
pub enum Slot<G> {
    /// Run once eagerly; the next window of the shape captures.
    Seen,
    Ready(G, u64),
    /// A capture of this shape failed: eager from now on.
    Eager,
}

/// What the cache wants done with a window, from `WindowGraphs::decide`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// The shape's first sighting: run eagerly (warms the kernels' lazy setup); the next one captures.
    Warmup,
    /// Capture this window's launches; on success replay the graph right away.
    Capture,
    /// Replay the captured graph.
    Replay,
    /// Run eagerly (a capture of this shape failed before, or the graphs are off for it).
    Eager,
}

/// What the cache did, for tests and the server's log.
#[derive(Clone, Debug, Default)]
pub struct WindowGraphStats {
    pub replays: u64,
    pub captures: u64,
    pub eager_windows: u64,
    pub capture_failures: u64,
    pub evictions: u64,
    pub capture_ms: f64,
    pub last_failure: Option<String>,
}

/// The decision layer of the window graphs: which windows capture, which replay, which run eagerly. Free of
/// device state on purpose (see the module doc); `forward.rs` drives it.
pub struct WindowGraphs<G = ()> {
    pub on: bool,
    /// Capture a shape at its first sighting (see [`ENV_AT`]).
    pub capture_first: bool,
    pub mutant: Option<WindowGraphMutant>,
    pub(crate) slots: HashMap<GraphKey, Slot<G>>,
    tick: u64,
    pub stats: WindowGraphStats,
}

impl<G> WindowGraphs<G> {
    /// From the environment ([`ENV`], [`ENV_AT`]); everything else defaults off.
    pub fn new() -> Self {
        let on = !matches!(std::env::var(ENV).as_deref(), Ok("0" | "off" | "false" | "no"));
        Self {
            on,
            capture_first: std::env::var(ENV_AT).as_deref() == Ok("0"),
            mutant: None,
            slots: HashMap::new(),
            tick: 0,
            stats: WindowGraphStats::default(),
        }
    }

    /// What to do with a window of this shape: replay the graph, capture it (the second sighting), or run it
    /// eagerly (the first sighting, or a capture that failed before).
    pub fn decide(&mut self, key: GraphKey) -> Decision {
        self.tick += 1;
        let tick = self.tick;
        match self.slots.get_mut(&key) {
            Some(Slot::Ready(_, used)) => {
                *used = tick;
                Decision::Replay
            }
            Some(Slot::Eager) => Decision::Eager,
            Some(Slot::Seen) => Decision::Capture,
            None if self.capture_first => Decision::Capture,
            None => {
                self.slots.insert(key, Slot::Seen);
                Decision::Warmup
            }
        }
    }

    /// Whether a window of this shape would replay (a peek; `decide` is the one that acts). The key must be the
    /// caller's best knowledge before the preamble runs: the mutant that skips an upload asks with it.
    pub fn would_replay(&self, key: GraphKey) -> bool {
        matches!(self.slots.get(&key), Some(Slot::Ready(..)))
    }

    /// A capture of this shape failed: eager from now on.
    pub fn mark_failed(&mut self, key: GraphKey) {
        self.slots.insert(key, Slot::Eager);
    }

    /// The capture succeeded: the graph is ready, last used now.
    pub fn mark_ready(&mut self, key: GraphKey, g: G) {
        self.slots.insert(key, Slot::Ready(g, self.tick));
    }

    /// Instantiated graphs held.
    pub fn ready_len(&self) -> usize {
        self.slots.values().filter(|s| matches!(s, Slot::Ready(..))).count()
    }

    /// Demote the least recently used ready shape back to `Seen` (its graph is destroyed): the caller syncs the
    /// stream first — a graph that is still executing must not be destroyed. Whether one was demoted.
    pub fn demote_oldest(&mut self) -> bool {
        let oldest = self
            .slots
            .iter()
            .filter_map(|(k, s)| match s {
                Slot::Ready(_, used) => Some((*used, *k)),
                _ => None,
            })
            .min_by_key(|x| x.0)
            .map(|(_, k)| k);
        match oldest {
            Some(k) => {
                self.slots.insert(k, Slot::Seen);
                self.stats.evictions += 1;
                true
            }
            None => false,
        }
    }

    /// Forget every captured graph (a test of the capture path; the caller drains the stream first).
    pub fn clear(&mut self) {
        self.slots.clear();
    }
}

impl<G> Default for WindowGraphs<G> {
    fn default() -> Self {
        Self { on: false, capture_first: false, mutant: None, slots: HashMap::new(), tick: 0, stats: WindowGraphStats::default() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> WindowGraphs<()> {
        WindowGraphs { on: true, ..Default::default() }
    }

    fn key(rows: usize) -> GraphKey {
        GraphKey { rows, n_seq: 1, par: 0, owed: false, count: 28 }
    }

    #[test]
    fn a_windows_key_is_every_host_value_the_launches_bake_in() {
        let b = |rows: usize, n_seq: usize| {
            let reqs = (0..n_seq)
                .map(|s| {
                    let per = rows / n_seq + usize::from(s < rows % n_seq);
                    RowRequest { slot: s, tokens: vec![10; per], first_position: 0 }
                })
                .collect::<Vec<_>>();
            RowBatch::from_requests(&reqs)
        };
        let k = GraphKey::of(&b(8, 2), 28, 1, true);
        assert_eq!(k, GraphKey { rows: 8, n_seq: 2, par: 1, owed: true, count: 28 });
        // rows, sequences, candidate count, record parity and the owed fold each make another graph
        assert_ne!(k, GraphKey::of(&b(9, 2), 28, 1, true));
        assert_ne!(k, GraphKey::of(&b(8, 2), 28, 0, true));
        assert_ne!(k, GraphKey::of(&b(8, 2), 20, 1, true));
        assert_ne!(k, GraphKey::of(&b(8, 2), 28, 1, false));
    }

    #[test]
    fn a_shape_warms_up_once_captures_once_and_replays_for_ever() {
        let mut c = cache();
        let k = key(8);
        assert_eq!(c.decide(k), Decision::Warmup, "the first sighting runs eagerly (and warms the kernels)");
        assert_eq!(c.decide(k), Decision::Capture, "the second sighting captures");
        c.mark_ready(k, ());
        assert_eq!(c.decide(k), Decision::Replay, "and every later one replays");
        assert_eq!(c.decide(k), Decision::Replay);
        // another shape has its own life
        assert_eq!(c.decide(key(4)), Decision::Warmup);
        assert_eq!(c.ready_len(), 1);
    }

    #[test]
    fn graph_at_zero_captures_the_first_sighting() {
        let mut c = WindowGraphs::<()> { on: true, capture_first: true, ..Default::default() };
        assert_eq!(c.decide(key(8)), Decision::Capture);
    }

    #[test]
    fn a_failed_capture_makes_the_shape_eager_for_good() {
        let mut c = cache();
        let k = key(8);
        assert_eq!(c.decide(k), Decision::Warmup);
        assert_eq!(c.decide(k), Decision::Capture);
        c.mark_failed(k);
        assert_eq!(c.decide(k), Decision::Eager, "the shape never captures again");
        assert_eq!(c.decide(k), Decision::Eager);
        assert_eq!(c.ready_len(), 0);
    }

    #[test]
    fn the_least_recently_used_graph_is_demoted_at_the_limit() {
        let mut c = cache();
        for rows in 0..GRAPH_LIMIT {
            let k = key(rows);
            assert_eq!(c.decide(k), Decision::Warmup);
            assert_eq!(c.decide(k), Decision::Capture);
            c.mark_ready(k, ());
        }
        assert_eq!(c.ready_len(), GRAPH_LIMIT);
        // touch the oldest (rows 0) so rows 1 becomes the least recently used
        assert_eq!(c.decide(key(0)), Decision::Replay);
        assert!(c.demote_oldest(), "at the limit one graph is demoted");
        assert_eq!(c.ready_len(), GRAPH_LIMIT - 1, "one fewer instantiated graph");
        assert_eq!(c.decide(key(0)), Decision::Replay, "the touched shape survived");
        assert_eq!(c.decide(key(1)), Decision::Capture, "the untouched one captures again");
        c.mark_ready(key(1), ());
        assert_eq!(c.ready_len(), GRAPH_LIMIT, "and the count is back at the limit");
    }

    use gb_engine::RowRequest;

    #[test]
    fn a_ready_shape_reports_itself_to_the_uploads_guard() {
        let mut c = cache();
        assert!(!c.would_replay(key(8)));
        c.decide(key(8));
        c.decide(key(8));
        c.mark_ready(key(8), ());
        assert!(c.would_replay(key(8)), "the mutant's skip-an-upload guard sees the replay coming");
        assert!(!c.would_replay(key(4)), "another shape does not");
    }
}
