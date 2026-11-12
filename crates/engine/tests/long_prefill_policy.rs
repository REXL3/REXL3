//! The long-prefill policy of the prefix cache (`docs/LONG_PREFILL_POLICY.md`) at engine level, on the CPU mock with a hybrid model's state
//! (paged KV the logits read, a path-dependent recurrent state, a draft ring): progressive cuts (checkpoints) that a cancelled prefill keeps,
//! and size-aware eviction. Every resumed reply must equal the uncached one byte for byte (tokens, finish, speculation counters), and the
//! books (pages and snapshot slots) must balance. Both features are off by default; the controls show the default behaviour unchanged.

use gb_engine::mock::{ChainDrafter, Fault, MockBackend, PrefixFault};
use gb_engine::{CacheBudget, CutPolicy, Drafter, Engine, EngineConfig, Eviction, Output, PrefixConfig, Request};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};

const VOCAB: usize = 512;
const SLOTS: usize = 8;
const SLICE: usize = 64;

fn admission() -> Admission {
    let mm = ModelMemory {
        weights_bytes: 1 << 30,
        kv_bytes_per_token: 65536,
        state_bytes_per_slot: 150 << 20,
        activation_reserve: 1 << 30,
    };
    Admission::new(64 << 30, 6 << 30, mm, SLOTS).unwrap()
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        max_rows: 64,
        max_drafts: 7,
        cost: CostModel::from_table(vec![(1, 70.0), (16, 74.0), (32, 110.0), (64, 190.0)]),
        drafting: true,
    }
}

fn drafter() -> Box<dyn Fn() -> Box<dyn Drafter>> {
    Box::new(|| Box::new(ChainDrafter { vocab: VOCAB, corrupt_every: 5 }) as Box<dyn Drafter>)
}

/// `units` entries of a snapshot slot (1 MiB) and 8 pages (64 KiB each): a snapshot costs 16 pages.
fn budget(units: usize) -> CacheBudget {
    CacheBudget { snapshot_slots: units, page_blocks: units * 8, snapshot_bytes: 1 << 20, page_bytes: 1 << 16 }
}

fn config(eviction: Eviction, checkpoint_every: usize) -> PrefixConfig {
    PrefixConfig {
        policy: CutPolicy { min_cut: 64, tail: 8, ghost_min_gain: 256, min_gap: 0 },
        ghost_blocks: 1 << 12,
        eviction,
        checkpoint_every,
        ..Default::default()
    }
}

fn mock(cache: Option<CacheBudget>) -> MockBackend {
    let b = MockBackend::new(VOCAB, SLOTS, Fault::None).with_path_dependent_state().with_native_draft();
    match cache {
        Some(bud) => b.with_prefix_cache(bud, PrefixFault::None),
        None => b.with_paged_kv(SLOTS * 64),
    }
    .with_prefill_slice(SLICE)
}

fn cached(units: usize, cfg: PrefixConfig) -> Engine<MockBackend> {
    Engine::new(mock(Some(budget(units))), admission(), engine_config(), drafter()).with_prefix_cache(cfg)
}

fn toks(seed: u64, n: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1 + (x % (VOCAB as u64 - 1)) as u32
        })
        .collect()
}

fn greedy(prompt: Vec<u32>, max_new: usize) -> Request {
    Request {
        prompt,
        sampling: Sampling { seed: 7, temperature: 0.0, top_k: 20, top_p: 0.95, min_p: 0.0 },
        max_new,
        stop: vec![],
        draft: true,
        images: vec![],
        cache: true,
    }
}

fn one(e: &mut Engine<MockBackend>, r: Request) -> Output {
    let id = e.submit(r).unwrap();
    e.run().unwrap();
    let mut outs = e.take_outputs();
    outs.retain(|o| o.id == id);
    assert_eq!(outs.len(), 1);
    outs.remove(0)
}

/// The reply on an engine without a cache.
fn reference(r: &Request) -> Output {
    let mut e = Engine::new(mock(None), admission(), engine_config(), drafter());
    one(&mut e, Request { cache: false, ..r.clone() })
}

fn same(a: &Output, b: &Output) {
    assert_eq!(a.tokens, b.tokens, "tokens");
    assert_eq!((a.finish, a.stopped), (b.finish, b.stopped), "finish");
    assert_eq!((a.drafted, a.accepted, a.rounds), (b.drafted, b.accepted, b.rounds), "speculation counters");
}

fn no_leaks(e: &Engine<MockBackend>) {
    let st = e.prefix_stats();
    assert_eq!(e.backend.pages_in_use(), st.pages_held as usize, "pages in use against the index");
    assert_eq!(e.backend.snapshots_in_use(), st.entries as usize, "snapshot slots in use against the index");
}

/// Prefill `r` until `done` prompt tokens are in, then cancel it (a client that left, a request deadline).
fn prefill_then_cancel(e: &mut Engine<MockBackend>, r: Request, done: usize) {
    let id = e.submit_deferred(r).unwrap();
    for _ in 0..done / SLICE {
        assert!(e.step().unwrap());
    }
    assert_eq!(e.prefilling(), 1, "still prefilling");
    assert!(e.cancel(id));
    let _ = e.take_outputs();
}

/// A 1,600-token prompt cancelled after 1,152 tokens: with checkpoints every 256 tokens the snapshot at 1,024 is kept (the earlier ones
/// were replaced as the prefill went on), and the retry resumes from it with the reply of a cold run. A second question that shares only
/// the first 1,100 tokens resumes from the same checkpoint. Without checkpoints (the default) nothing is kept.
#[test]
fn a_cancelled_long_prefill_keeps_its_last_checkpoint_and_a_retry_resumes_exactly() {
    let doc = toks(1, 1600);
    let r = greedy(doc.clone(), 24);
    // the default: the cancelled prefill leaves nothing
    let mut e = cached(8, config(Eviction::Lru, 0));
    prefill_then_cancel(&mut e, r.clone(), 1152);
    let st = e.prefix_stats();
    assert_eq!((st.entries, st.kept_on_cancel, st.checkpoints), (0, 0, 0));
    no_leaks(&e);
    // checkpoints on
    let mut e = cached(8, config(Eviction::Lru, 256));
    prefill_then_cancel(&mut e, r.clone(), 1152);
    let st = e.prefix_stats();
    assert_eq!((st.checkpoints, st.checkpoints_superseded, st.kept_on_cancel), (4, 3, 1), "{st:?}");
    assert_eq!((st.entries, st.captures_discarded), (1, 0), "{st:?}");
    no_leaks(&e);
    // the retry: resumed from 1,024 tokens, the reply of a cold run
    let out = one(&mut e, r.clone());
    assert_eq!(out.cached_tokens, 1024);
    same(&out, &reference(&r));
    no_leaks(&e);
    // a second question about the same first 1,100 tokens: also from the checkpoint at 1,024 (the retry's own end cut is longer than it)
    let q2 = greedy([&doc[..1100], &toks(2, 300)[..]].concat(), 16);
    let out = one(&mut e, q2.clone());
    assert_eq!(out.cached_tokens, 1024);
    same(&out, &reference(&q2));
    no_leaks(&e);
}

/// Checkpoints never evict an entry: with too few free snapshot slots (the planned cuts plus two) a prompt takes none, and the cache is
/// as it would be without them.
#[test]
fn checkpoints_only_use_free_snapshot_slots() {
    let mut e = cached(2, config(Eviction::Lru, 256));
    // fill both slots with two end cuts
    for s in [3u64, 4] {
        let r = greedy(toks(s, 400), 4);
        let _ = one(&mut e, r);
    }
    assert_eq!(e.prefix_stats().entries, 2);
    let r = greedy(toks(5, 1600), 8);
    let out = one(&mut e, r.clone());
    same(&out, &reference(&r));
    let st = e.prefix_stats();
    assert_eq!(st.checkpoints, 0, "{st:?}");
    no_leaks(&e);
}

/// One entry holds every page of a small cache. LRU (the default) evicts it for the next small prompt; size-aware eviction refuses the
/// small entries a bounded number of times (the credit ratio) and the large entry keeps serving exact hits meanwhile; then it goes.
#[test]
fn a_large_entry_survives_small_inserts_under_size_aware_eviction_and_hits_exactly() {
    // 2 units: 2 snapshot slots and 16 pages; the large prompt's end cut (4,088 tokens) holds all 16 pages
    let big = toks(10, 4096);
    let ask = |i: u64| greedy([&big[..], &toks(100 + i, 40)[..]].concat(), 8);
    let small = |i: u64| greedy(toks(200 + i, 300), 4);
    // LRU: one small prompt evicts it
    let mut e = cached(2, config(Eviction::Lru, 0));
    let _ = one(&mut e, greedy(big.clone(), 4));
    let _ = one(&mut e, small(0));
    assert_eq!(one(&mut e, ask(0)).cached_tokens, 0, "LRU: the large entry was evicted by one small prompt");
    // size-aware: overhead 16 pages; d_big = 4,088 / 32 = 127.75, d_small = 292 / 18 = 16.2: the large entry outlasts 7 small inserts
    let mut e = cached(2, config(Eviction::SizeAware { overhead_pages: 0 }, 0));
    let _ = one(&mut e, greedy(big.clone(), 4));
    for i in 0..7 {
        let _ = one(&mut e, small(i));
        no_leaks(&e);
    }
    let st = e.prefix_stats();
    assert_eq!((st.refused_by_value, st.entries), (7, 1), "{st:?}");
    let q = ask(1);
    let out = one(&mut e, q.clone());
    assert_eq!(out.cached_tokens, 4088);
    same(&out, &reference(&q));
    // the hit renewed its credit (twice the density at the current inflation); small prompts keep being refused, and after enough of them
    // the large entry is the victim: small entries are never starved for good
    let mut stored_at = None;
    for i in 10..60 {
        let _ = one(&mut e, small(i));
        no_leaks(&e);
        if e.prefix_stats().refused_by_value < 7 + (i - 9) {
            stored_at = Some(i);
            break;
        }
    }
    let at = stored_at.expect("a small entry was stored within 50 small prompts");
    assert!(at > 10, "the hit protected the large entry for a while ({at})");
    assert_eq!(one(&mut e, ask(2)).cached_tokens, 0, "then it was evicted");
}

/// The admission's prefill floor counts only what is left after the longest cached prefix, and reading it leaves the cache as it was (no
/// lookup counted, no pin, no recency): a refused request must not change what the cache keeps.
#[test]
fn the_prefill_floor_starts_at_the_cached_prefix_and_peeks_without_side_effects() {
    let mut e = Engine::new(
        mock(Some(budget(8))).with_prefill_floor(std::time::Duration::from_millis(1)),
        admission(),
        engine_config(),
        drafter(),
    )
    .with_prefix_cache(config(Eviction::Lru, 0));
    let doc = toks(30, 2000);
    let ask = greedy([&doc[..], &toks(31, 100)[..]].concat(), 4);
    assert_eq!(e.prefill_floor(&ask), Some(std::time::Duration::from_millis(2100)), "cold: every token");
    let _ = one(&mut e, greedy(doc.clone(), 4));
    let before = e.prefix_stats();
    // the end cut of the document is at 1,992: 108 tokens are left
    assert_eq!(e.prefill_floor(&ask), Some(std::time::Duration::from_millis(108)));
    assert_eq!(e.prefix_stats(), before, "peeking changed the cache's counters");
    // a request that opts out of the cache is charged in full
    assert_eq!(e.prefill_floor(&Request { cache: false, ..ask.clone() }), Some(std::time::Duration::from_millis(2100)));
    assert_eq!(one(&mut e, ask).cached_tokens, 1992);
    // a backend without a floor states none
    let plain = cached(8, config(Eviction::Lru, 0));
    assert_eq!(plain.prefill_floor(&greedy(doc, 4)), None);
}
