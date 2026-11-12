//! `/metrics` text for the prefix cache (`gb_engine::PrefixStats`): lookups, hits, tokens saved, inserts, evictions, and what the cache
//! holds. Counters only grow; a client that wants one workload's hit rate reads them before and after and subtracts.

use gb_engine::PrefixStats;
use std::fmt::Write;

fn put(out: &mut String, name: &str, kind: &str, help: &str, v: impl std::fmt::Display) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}");
}

/// Prometheus text for the cache. `rexl3_prefix_cache_enabled` is always there (0 when the engine runs without one), so a dashboard can tell
/// "off" from "no traffic yet"; the rest is only emitted when it is on.
pub fn render(s: &PrefixStats) -> String {
    let mut out = String::new();
    put(&mut out, "rexl3_prefix_cache_enabled", "gauge", "1 when the engine runs a prefix cache.", s.enabled as u8);
    if !s.enabled {
        return out;
    }
    let c = "counter";
    let g = "gauge";
    put(
        &mut out,
        "rexl3_prefix_cache_lookups_total",
        c,
        "Requests that looked the prefix cache up (above 128 prompt tokens, text, not opted out).",
        s.lookups,
    );
    put(
        &mut out,
        "rexl3_prefix_cache_lookup_tokens_total",
        c,
        "Prompt tokens of the requests that looked the cache up.",
        s.lookup_tokens,
    );
    put(&mut out, "rexl3_prefix_cache_hits_total", c, "Lookups that found a stored prefix and resumed from it.", s.hits);
    put(&mut out, "rexl3_prefix_cache_tokens_saved_total", c, "Prompt tokens a hit did not prefill.", s.tokens_saved);
    put(&mut out, "rexl3_prefix_cache_inserts_total", c, "Snapshots stored.", s.inserts);
    put(
        &mut out,
        "rexl3_prefix_cache_duplicates_total",
        c,
        "Snapshots dropped because the prefix was stored already.",
        s.duplicates,
    );
    put(&mut out, "rexl3_prefix_cache_no_room_total", c, "Snapshots dropped because they could not fit.", s.no_room);
    put(
        &mut out,
        "rexl3_prefix_cache_evictions_total",
        c,
        "Entries evicted (least recently used first, a prefix hit at least twice last).",
        s.evictions,
    );
    put(
        &mut out,
        "rexl3_prefix_cache_evicted_for_pages_total",
        c,
        "Evictions made to fit a new entry's pages in the page budget.",
        s.evicted_for_pages,
    );
    put(
        &mut out,
        "rexl3_prefix_cache_evicted_for_slots_total",
        c,
        "Evictions made to free a snapshot slot for a capture.",
        s.evicted_for_slots,
    );
    put(
        &mut out,
        "rexl3_prefix_cache_restore_failures_total",
        c,
        "Restores that failed (the entry was dropped and the prompt prefilled cold).",
        s.restore_failures,
    );
    put(
        &mut out,
        "rexl3_prefix_cache_captures_discarded_total",
        c,
        "Snapshots released unused because their sequence failed or was cancelled while prefilling.",
        s.captures_discarded,
    );
    if s.long_prefill_policy {
        render_long_prefill(&mut out, s);
    }
    put(&mut out, "rexl3_prefix_cache_entries", g, "Entries stored.", s.entries);
    put(&mut out, "rexl3_prefix_cache_pages", g, "Distinct KV pages the entries hold.", s.pages_held);
    put(&mut out, "rexl3_prefix_cache_snapshot_slots", g, "Snapshot slots reserved for the cache.", s.snapshot_slots);
    put(&mut out, "rexl3_prefix_cache_page_blocks", g, "KV pages reserved for the cache.", s.page_blocks);
    // a pool-page cache reserves no pages (`page_blocks` 0) and may hold any idle page of the pool: said only for it, so every other
    // server serves the /metrics text it did before
    if s.pool_pages > 0 {
        put(
            &mut out,
            "rexl3_prefix_cache_pool_pages",
            g,
            "KV pool pages the cache may hold while idle (--prefix-cache pool: it reserves none of its own).",
            s.pool_pages,
        );
    }
    put(&mut out, "rexl3_prefix_cache_bytes", g, "Bytes the entries hold (snapshot slots plus distinct pages).", s.bytes_held);
    put(&mut out, "rexl3_prefix_cache_budget_bytes", g, "Bytes reserved for the cache at boot.", s.bytes_budget);
    put(&mut out, "rexl3_prefix_cache_hit_ratio", g, "Hits per lookup since boot.", format!("{:.6}", s.hit_rate()));
    out
}

/// The long-prefill policy's counters (`docs/LONG_PREFILL_POLICY.md`), only when it is on.
fn render_long_prefill(out: &mut String, s: &PrefixStats) {
    let c = "counter";
    put(
        out,
        "rexl3_prefix_cache_refused_by_value_total",
        c,
        "Snapshots not stored because the room was held by entries of more value (size-aware eviction).",
        s.refused_by_value,
    );
    put(
        out,
        "rexl3_prefix_cache_checkpoints_total",
        c,
        "Checkpoint snapshots a long prefill took (progressive cuts).",
        s.checkpoints,
    );
    put(
        out,
        "rexl3_prefix_cache_checkpoints_superseded_total",
        c,
        "Checkpoint snapshots released because a later checkpoint of the same prefill replaced them.",
        s.checkpoints_superseded,
    );
    put(
        out,
        "rexl3_prefix_cache_kept_on_cancel_total",
        c,
        "Snapshots of a cancelled prefill offered to the cache instead of discarded (progressive cuts on).",
        s.kept_on_cancel,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_long_prefill_counters_appear_only_with_the_policy_on() {
        let off = render(&PrefixStats { enabled: true, ..Default::default() });
        assert!(!off.contains("checkpoints") && !off.contains("refused_by_value") && !off.contains("kept_on_cancel"));
        let on = render(&PrefixStats {
            enabled: true,
            long_prefill_policy: true,
            refused_by_value: 3,
            checkpoints: 5,
            checkpoints_superseded: 4,
            kept_on_cancel: 1,
            ..Default::default()
        });
        for line in [
            "rexl3_prefix_cache_refused_by_value_total 3",
            "rexl3_prefix_cache_checkpoints_total 5",
            "rexl3_prefix_cache_checkpoints_superseded_total 4",
            "rexl3_prefix_cache_kept_on_cancel_total 1",
        ] {
            assert!(on.contains(&format!("\n{line}\n")), "{line}");
        }
    }

    /// The series a shipped cache (pages of its own) serves, in order: the pool-page gauge is not among them, so its /metrics text is the
    /// text it served before `--prefix-cache pool` existed.
    #[test]
    fn a_cache_with_pages_of_its_own_serves_the_series_it_did_before() {
        let t = render(&PrefixStats { enabled: true, page_blocks: 48, snapshot_slots: 6, ..Default::default() });
        let names: Vec<&str> = t.lines().filter(|l| !l.starts_with('#')).filter_map(|l| l.split(' ').next()).collect();
        assert_eq!(
            names,
            [
                "rexl3_prefix_cache_enabled",
                "rexl3_prefix_cache_lookups_total",
                "rexl3_prefix_cache_lookup_tokens_total",
                "rexl3_prefix_cache_hits_total",
                "rexl3_prefix_cache_tokens_saved_total",
                "rexl3_prefix_cache_inserts_total",
                "rexl3_prefix_cache_duplicates_total",
                "rexl3_prefix_cache_no_room_total",
                "rexl3_prefix_cache_evictions_total",
                "rexl3_prefix_cache_evicted_for_pages_total",
                "rexl3_prefix_cache_evicted_for_slots_total",
                "rexl3_prefix_cache_restore_failures_total",
                "rexl3_prefix_cache_captures_discarded_total",
                "rexl3_prefix_cache_entries",
                "rexl3_prefix_cache_pages",
                "rexl3_prefix_cache_snapshot_slots",
                "rexl3_prefix_cache_page_blocks",
                "rexl3_prefix_cache_bytes",
                "rexl3_prefix_cache_budget_bytes",
                "rexl3_prefix_cache_hit_ratio",
            ]
        );
    }

    /// `--prefix-cache pool` reports through the SAME series as a cache with pages of its own (what the gate's prefix stage reads): a cached
    /// turn moves the hits by one and the saved tokens by the hit's length, a full-window prompt counts its evictions, and the one gauge of
    /// its own says how many pool pages the entries may hold (its `page_blocks` is 0: it reserved none). Driven on the engine's GLM-shaped mock.
    #[test]
    fn a_pool_page_cache_reports_its_hits_and_evictions_through_the_same_series() {
        use gb_engine::mock::{Fault, MockBackend, PrefixFault, MOCK_PAGE};
        use gb_engine::{CutPolicy, Engine, EngineConfig, PrefixConfig, Request};
        const WINDOW: usize = 16 * MOCK_PAGE;
        let pool_pages = WINDOW / MOCK_PAGE + 1;
        let mm = gb_sched::ModelMemory {
            weights_bytes: 1 << 30,
            kv_bytes_per_token: 12_000,
            state_bytes_per_slot: 141 << 20,
            activation_reserve: 1 << 30,
        };
        let adm = gb_sched::Admission::new(64 << 30, 6 << 30, mm, 1)
            .unwrap()
            .with_max_context(WINDOW as u64)
            .with_kv_tokens(WINDOW as u64);
        let cfg = EngineConfig {
            max_rows: 64,
            max_drafts: 7,
            cost: gb_sched::CostModel::from_table(vec![(1, 70.0), (16, 74.0), (32, 110.0), (64, 190.0)]),
            drafting: true,
        };
        let backend =
            MockBackend::new(512, 1, Fault::None).with_glm_shape().with_pool_prefix_cache(4, pool_pages, PrefixFault::None);
        let mut e = Engine::new(backend, adm, cfg, Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn gb_engine::Drafter>))
            .with_prefix_cache(PrefixConfig {
                policy: CutPolicy { min_cut: 64, tail: 8, ghost_min_gain: 256, min_gap: 0 },
                ghost_blocks: 1 << 12,
                ..Default::default()
            });
        let toks = |seed: u32, n: usize| -> Vec<u32> {
            (0..n as u32).map(|i| 1 + (i.wrapping_mul(2_654_435_761) ^ seed) % 511).collect()
        };
        let req = |prompt: Vec<u32>, max_new: usize| Request {
            prompt,
            sampling: gb_sampling::Sampling { seed: 7, temperature: 0.0, top_k: 20, top_p: 0.95, min_p: 0.0 },
            max_new,
            stop: vec![],
            draft: true,
            images: vec![],
            cache: true,
        };
        let mut run = |e: &mut Engine<MockBackend>, r: Request| {
            let id = e.submit(r).unwrap();
            e.run().unwrap();
            e.take_outputs().into_iter().find(|o| o.id == id).unwrap()
        };
        let line = |t: &str, name: &str| -> u64 {
            t.lines()
                .find_map(|l| l.strip_prefix(&format!("{name} ")))
                .unwrap_or_else(|| panic!("{name} missing"))
                .parse()
                .unwrap()
        };
        let a = toks(1, 3000);
        run(&mut e, req(a.clone(), 24));
        let before = render(&e.prefix_stats());
        assert_eq!(line(&before, "rexl3_prefix_cache_enabled"), 1);
        assert_eq!(line(&before, "rexl3_prefix_cache_hits_total"), 0);
        assert_eq!(line(&before, "rexl3_prefix_cache_page_blocks"), 0, "it reserved no pages of its own");
        assert_eq!(line(&before, "rexl3_prefix_cache_pool_pages"), pool_pages as u64);
        assert!(line(&before, "rexl3_prefix_cache_pages") > 0, "its entry holds idle pool pages");
        // the cached turn: hits 0 -> 1, the saved tokens the hit's length (the end cut, 8 tokens short of the prompt)
        let mut b = a.clone();
        b.extend(toks(2, 300));
        let ob = run(&mut e, req(b, 60));
        let after = render(&e.prefix_stats());
        assert_eq!(ob.cached_tokens, 3000 - 8);
        assert_eq!(line(&after, "rexl3_prefix_cache_hits_total"), 1);
        assert_eq!(line(&after, "rexl3_prefix_cache_tokens_saved_total"), (3000 - 8) as u64);
        assert_eq!(line(&after, "rexl3_prefix_cache_evictions_total"), line(&before, "rexl3_prefix_cache_evictions_total"));
        // a full-window prompt evicts the entries in its way: counted as evictions for pages
        let ev0 = line(&after, "rexl3_prefix_cache_evictions_total");
        run(&mut e, req(toks(3, WINDOW - 96), 96));
        let full = render(&e.prefix_stats());
        assert!(line(&full, "rexl3_prefix_cache_evictions_total") > ev0, "{full}");
        assert!(line(&full, "rexl3_prefix_cache_evicted_for_pages_total") >= 1, "{full}");
        assert_eq!(line(&full, "rexl3_prefix_cache_hits_total"), 1);
    }

    #[test]
    fn a_cache_that_is_off_says_so_and_nothing_else() {
        let t = render(&PrefixStats::default());
        assert!(t.contains("rexl3_prefix_cache_enabled 0"));
        assert!(!t.contains("hits_total"));
    }

    #[test]
    fn the_counters_and_the_hit_ratio_are_rendered() {
        let s = PrefixStats {
            enabled: true,
            lookups: 8,
            hits: 6,
            tokens_saved: 20_000,
            entries: 3,
            bytes_held: 1 << 30,
            ..Default::default()
        };
        let t = render(&s);
        for line in [
            "rexl3_prefix_cache_enabled 1",
            "rexl3_prefix_cache_lookups_total 8",
            "rexl3_prefix_cache_hits_total 6",
            "rexl3_prefix_cache_tokens_saved_total 20000",
            "rexl3_prefix_cache_entries 3",
            "rexl3_prefix_cache_bytes 1073741824",
            "rexl3_prefix_cache_hit_ratio 0.750000",
        ] {
            assert!(t.contains(line), "{line}\n{t}");
        }
        assert!(t.contains("# TYPE rexl3_prefix_cache_hits_total counter"));
    }
}
