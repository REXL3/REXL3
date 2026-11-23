//! The bound on Flash-Next's captured CUDA graphs: a least-recently-used cache with a fixed entry cap, and its counters.
//!
//! A decode window and an MTP draft chain are each captured once per shape (`model.rs`: `WindowKey`, `ChainKey`) and replayed. The
//! shape space is large: a window's key carries the row count of every sequence, the deferred-commit flag, the record parity, the QSA
//! split and the attention grid, so two streams that draft independently visit thousands of shapes (4,252 in 7,050 windows on four
//! streams in the report this bound answers), and an instantiated graph holds host memory for as long as it lives. The cache keeps
//! at most `cap` of them and drops the one used longest ago when a new shape arrives.
//!
//! **What eviction may and may not do.** A graph is only a recording of launches: replaying it, capturing it again and running the
//! same launches eagerly issue the same kernels with the same arguments in the same order, so which shapes are resident never
//! reaches an output bit. The one hazard is lifetime: a launched graph may still be executing when the next shape arrives, so the
//! owner waits for the stream before it drops a victim ([`GraphLru::make_room`] waits, then evicts; `model.rs`
//! `capture_and_launch`). The order of evictions depends on the sequence of keys alone, never on timing, so the ranks of a
//! tensor-parallel group (which see the same windows in the same order) keep the same residents.
//!
//! This module is free of anything device-shaped so the CPU suite pins the policy.

use std::collections::HashMap;
use std::hash::Hash;

/// Window graphs kept at once. Sized from the shipped two-slot profile (`docs/FLASH_NEXT_DFLASH2.md`, "Graph caches"): one
/// stream visits about 35 window shapes per attention grid bucket, two streams about 150, and 128 shapes carry 97.6% of the
/// lookups of a mixed run (one and two streams, greedy and sampled, four grid buckets; 99% of a two-stream run inside one
/// bucket). A graph of a two-stream window costs about 10 MiB of host memory, so the cap holds the cache near 1.3 GiB; a miss
/// costs one capture (about 4 ms, close to what one replay saves over an eager window), so a shape outside the resident set costs
/// little more than running it eagerly.
pub const WINDOW_GRAPH_CAP: usize = 128;

/// MTP draft-chain graphs kept at once. A chain is one sequence (the head drafts for a lone stream), so its shape space is small:
/// 44 shapes over two grid buckets in the same measurement, 24 of them carrying 99% of the lookups.
pub const CHAIN_GRAPH_CAP: usize = 64;

/// At most `cap` values, least recently used out. `get` and `insert` both count as a use.
pub struct GraphLru<K, G> {
    by_key: HashMap<K, (G, u64)>,
    /// The use counter: every entry carries the tick of its last use, and the smallest is the eviction victim.
    tick: u64,
    cap: usize,
    /// What this cache has done; the owner counts (`model.rs` `capture_and_launch`).
    pub stats: CacheStats,
}

impl<K: Copy + Eq + Hash, G> GraphLru<K, G> {
    pub fn new(cap: usize) -> Self {
        assert!(cap > 0, "a graph cache holds at least the graph being launched");
        Self { by_key: HashMap::new(), tick: 0, cap, stats: CacheStats::default() }
    }

    /// Move the cache out, leaving an empty one of the same cap (the owner lends itself to the captured closure meanwhile).
    pub fn take(&mut self) -> Self {
        std::mem::replace(self, Self::new(self.cap))
    }

    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Change the cap (tests shrink it to force evictions). Entries over a smaller cap leave on the next inserts.
    pub fn set_cap(&mut self, cap: usize) {
        assert!(cap > 0, "a graph cache holds at least the graph being launched");
        self.cap = cap;
    }

    pub fn contains(&self, key: &K) -> bool {
        self.by_key.contains_key(key)
    }

    /// Whether an insert must be preceded by an eviction.
    pub fn full(&self) -> bool {
        self.by_key.len() >= self.cap
    }

    /// The value of `key`, now the most recently used.
    pub fn get(&mut self, key: &K) -> Option<&G> {
        self.tick += 1;
        let tick = self.tick;
        self.by_key.get_mut(key).map(|e| {
            e.1 = tick;
            &e.0
        })
    }

    /// Make room for one insert: when the cache is full, call `drain` once (the owner's wait for every launch of a resident graph
    /// to finish) and only then drop least recently used entries until one slot is free. A failed `drain` drops nothing and
    /// returns its error. Returns the number of graphs dropped (also added to [`CacheStats::evictions`]).
    pub fn make_room<E>(&mut self, drain: impl FnOnce() -> Result<(), E>) -> Result<u64, E> {
        if !self.full() {
            return Ok(0);
        }
        drain()?;
        let mut dropped = 0;
        while self.full() {
            drop(self.evict());
            dropped += 1;
        }
        self.stats.evictions += dropped;
        Ok(dropped)
    }

    /// Remove and return the least recently used entry. The caller owns the wait that makes dropping it safe ([`GraphLru::make_room`]
    /// is the only caller outside the tests).
    pub fn evict(&mut self) -> Option<(K, G)> {
        let key = *self.by_key.iter().min_by_key(|(_, e)| e.1)?.0;
        self.by_key.remove(&key).map(|e| (key, e.0))
    }

    /// Add `key` as the most recently used. The cache must have room ([`GraphLru::full`] is false) and must not hold `key`.
    pub fn insert(&mut self, key: K, value: G) {
        assert!(!self.full(), "evict before inserting into a full graph cache");
        self.tick += 1;
        let old = self.by_key.insert(key, (value, self.tick));
        assert!(old.is_none(), "a captured shape is inserted once");
    }

    /// The resident keys, least recently used first (tests and the boot report).
    pub fn keys_by_age(&self) -> Vec<K> {
        let mut v: Vec<(u64, K)> = self.by_key.iter().map(|(k, e)| (e.1, *k)).collect();
        v.sort_by_key(|e| e.0);
        v.into_iter().map(|e| e.1).collect()
    }
}

/// What one of the two caches has done (counters only grow).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CacheStats {
    /// Lookups that replayed a resident graph.
    pub replays: u64,
    /// Lookups that captured a new graph (it is launched in the same call).
    pub captures: u64,
    /// Graphs destroyed to stay at the cap.
    pub evictions: u64,
    /// Milliseconds spent capturing (begin to instantiate; the launch is not in it).
    pub capture_ms: f64,
}

/// The counters of a model's window and chain graph caches.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GraphStats {
    pub window: CacheStats,
    pub chain: CacheStats,
    /// Windows and chains that ran eagerly (graphs off, a rank before its first exchange, a refused capture).
    pub eager: u64,
    /// Captures the runtime refused (the model then runs eagerly for the rest of the process).
    pub capture_failures: u64,
}

impl GraphStats {
    /// The engine's view (`/metrics`: `rexl3_graph_cache_*`). The chain cache is the MTP draft head's, so it reports as the draft
    /// head's cache; evictions and capture time are the sum of both.
    pub fn counters(&self) -> gb_engine::GraphCounters {
        gb_engine::GraphCounters {
            replays: self.window.replays,
            captures: self.window.captures,
            eager_windows: self.eager,
            capture_failures: self.capture_failures,
            evictions: self.window.evictions + self.chain.evictions,
            capture_ms: self.window.capture_ms + self.chain.capture_ms,
            draft_replays: self.chain.replays,
            draft_captures: self.chain.captures,
            draft_eager: 0,
        }
    }
}

/// Whether the `n`-th eviction of a cache is logged: the first, then every power of two (a thrashing cache stays visible in the
/// log without a line per window).
pub fn logs_eviction(n: u64) -> bool {
    n.is_power_of_two()
}

/// Whether the evictions that took a cache's count from `before` to `after` crossed a logged count ([`logs_eviction`]).
pub fn logs_evictions(before: u64, after: u64) -> bool {
    (before + 1..=after).any(logs_eviction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_least_recently_used_entry_is_the_victim_and_a_replay_refreshes_its_entry() {
        let mut c: GraphLru<u32, &str> = GraphLru::new(3);
        for (k, v) in [(1, "a"), (2, "b"), (3, "c")] {
            assert!(!c.full());
            c.insert(k, v);
        }
        assert!(c.full() && c.len() == 3);
        // a replay of the oldest entry makes it the newest: the victim is then the second oldest
        assert_eq!(c.get(&1), Some(&"a"));
        assert_eq!(c.keys_by_age(), vec![2, 3, 1]);
        assert_eq!(c.evict(), Some((2, "b")));
        c.insert(4, "d");
        assert_eq!(c.keys_by_age(), vec![3, 1, 4]);
        // a miss is not a use of anything
        assert_eq!(c.get(&2), None);
        assert_eq!(c.keys_by_age(), vec![3, 1, 4]);
        assert_eq!(c.evict(), Some((3, "c")));
        assert_eq!(c.evict(), Some((1, "a")));
        assert_eq!(c.evict(), Some((4, "d")));
        assert_eq!(c.evict(), None);
        assert!(c.is_empty());
    }

    /// The owner's loop (evict while full, then insert) never holds more than the cap, whatever the key sequence, counts one
    /// eviction per graph dropped, and keeps a hot key resident through any number of cold ones.
    #[test]
    fn the_owner_loop_stays_at_the_cap_and_keeps_hot_keys_resident() {
        let cap = 8;
        let mut c: GraphLru<u64, u64> = GraphLru::new(cap);
        let (mut evictions, mut captures, mut replays) = (0u64, 0u64, 0u64);
        let mut lookup = |c: &mut GraphLru<u64, u64>, key: u64| {
            if !c.contains(&key) {
                while c.full() {
                    assert!(c.evict().is_some());
                    evictions += 1;
                }
                c.insert(key, key * 10);
                captures += 1;
            } else {
                replays += 1;
            }
            assert_eq!(c.get(&key), Some(&(key * 10)));
            assert!(c.len() <= cap);
        };
        // keys 0 and 1 are hot (every other lookup); a stream of 500 cold keys passes between them
        let mut x = 0x9e37_79b9_7f4a_7c15u64;
        for i in 0..1000u64 {
            let key = if i % 2 == 0 {
                i / 2 % 2
            } else {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                100 + x % 100_000
            };
            lookup(&mut c, key);
        }
        assert!(c.contains(&0) && c.contains(&1), "a hot key was evicted by cold ones");
        assert_eq!(c.len(), cap);
        assert_eq!(captures + replays, 1000);
        assert_eq!(evictions, captures - cap as u64, "every capture past the cap evicts exactly one graph");
        assert!(replays >= 498, "the hot keys replay: {replays}");
    }

    #[test]
    fn a_smaller_cap_drains_on_the_next_insert() {
        let mut c: GraphLru<u32, u32> = GraphLru::new(4);
        for k in 0..4 {
            c.insert(k, k);
        }
        c.set_cap(2);
        let mut dropped = Vec::new();
        while c.full() {
            dropped.push(c.evict().unwrap().0);
        }
        c.insert(9, 9);
        assert_eq!(dropped, vec![0, 1, 2]);
        assert_eq!(c.keys_by_age(), vec![3, 9]);
    }

    #[test]
    #[should_panic(expected = "evict before inserting")]
    fn an_insert_into_a_full_cache_is_a_bug() {
        let mut c: GraphLru<u32, u32> = GraphLru::new(1);
        c.insert(1, 1);
        c.insert(2, 2);
    }

    #[test]
    fn the_engine_counters_sum_both_caches_and_report_the_chain_as_the_draft_head() {
        let s = GraphStats {
            window: CacheStats { replays: 10, captures: 4, evictions: 2, capture_ms: 1.5 },
            chain: CacheStats { replays: 7, captures: 3, evictions: 1, capture_ms: 0.5 },
            eager: 5,
            capture_failures: 1,
        };
        let c = s.counters();
        assert_eq!((c.replays, c.captures, c.eager_windows, c.capture_failures), (10, 4, 5, 1));
        assert_eq!((c.evictions, c.capture_ms), (3, 2.0));
        assert_eq!((c.draft_replays, c.draft_captures, c.draft_eager), (7, 3, 0));
    }

    #[test]
    fn eviction_logging_fires_when_a_batch_crosses_a_power_of_two() {
        assert!(logs_evictions(0, 1));
        assert!(!logs_evictions(4, 7));
        assert!(logs_evictions(5, 8));
        assert!(logs_evictions(60, 70));
        assert!(!logs_evictions(70, 70));
    }

    #[test]
    fn evictions_are_logged_at_powers_of_two() {
        let logged: Vec<u64> = (1..=70).filter(|&n| logs_eviction(n)).collect();
        assert_eq!(logged, vec![1, 2, 4, 8, 16, 32, 64]);
    }
}
