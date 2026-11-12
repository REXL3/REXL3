//! The pool-page prefix cache at engine level on the CPU mock in its GLM shape (`MockBackend::with_pool_prefix_cache`): the entries keep the KV
//! pool's IDLE pages (no pages of their own, `page_blocks == 0`), and the engine's room step (`Engine::pool_room`) evicts them, least recently used
//! first and never a pinned one, only for the pages a live sequence may still claim beyond what it holds, the pages a hit shares with it credited.
//!
//! The pool here is what the planner gives one slot: the window's pages plus the slot's one page of rounding (`plan_memory_pool_cached`). The
//! mock panics on a page the pool cannot give, so a room step that keeps too little fails these tests at once, and one that evicts too much
//! (the cache gone exactly where it matters: a follow-up turn longer than half the pool) fails their hit counts.

use gb_engine::mock::{Fault, MockBackend, PrefixFault, MOCK_PAGE};
use gb_engine::{CutPolicy, Drafter, Engine, EngineConfig, Output, PrefixConfig, Request};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};

const VOCAB: usize = 512;
const TAIL: usize = 8;
/// The window and the pool of one slot: 16 pages of tokens, 17 pages.
const WINDOW: usize = 16 * MOCK_PAGE;
const POOL_PAGES: usize = WINDOW / MOCK_PAGE + 1;

fn admission(kv_tokens: usize) -> Admission {
    let mm = ModelMemory {
        weights_bytes: 1 << 30,
        kv_bytes_per_token: 12_000,
        state_bytes_per_slot: 141 << 20,
        activation_reserve: 1 << 30,
    };
    Admission::new(64 << 30, 6 << 30, mm, 1).unwrap().with_max_context(kv_tokens as u64).with_kv_tokens(kv_tokens as u64)
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        max_rows: 64,
        max_drafts: 7,
        cost: CostModel::from_table(vec![(1, 70.0), (16, 74.0), (32, 110.0), (64, 190.0)]),
        drafting: true,
    }
}

fn no_native_drafter() -> Box<dyn Fn() -> Box<dyn Drafter>> {
    Box::new(|| Box::new(gb_engine::NoDrafts) as Box<dyn Drafter>)
}

fn policy() -> PrefixConfig {
    PrefixConfig {
        policy: CutPolicy { min_cut: 64, tail: TAIL, ghost_min_gain: 256, min_gap: 0 },
        ghost_blocks: 1 << 12,
        ..Default::default()
    }
}

fn pooled(snapshot_slots: usize) -> Engine<MockBackend> {
    let b = MockBackend::new(VOCAB, 1, Fault::None).with_glm_shape().with_pool_prefix_cache(
        snapshot_slots,
        POOL_PAGES,
        PrefixFault::None,
    );
    Engine::new(b, admission(WINDOW), engine_config(), no_native_drafter()).with_prefix_cache(policy())
}

fn plain() -> Engine<MockBackend> {
    let b = MockBackend::new(VOCAB, 1, Fault::None).with_glm_shape().with_paged_kv(64);
    Engine::new(b, admission(WINDOW), engine_config(), no_native_drafter())
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

fn request(prompt: Vec<u32>, max_new: usize, seed: u64, temperature: f64) -> Request {
    Request {
        prompt,
        sampling: Sampling { seed, temperature, top_k: 20, top_p: 0.95, min_p: 0.0 },
        max_new,
        stop: vec![],
        draft: true,
        images: vec![],
        cache: true,
    }
}

fn greedy(prompt: Vec<u32>, max_new: usize) -> Request {
    request(prompt, max_new, 7, 0.0)
}

fn one(e: &mut Engine<MockBackend>, r: Request) -> Output {
    let id = e.submit(r).unwrap();
    e.run().unwrap();
    let mut outs = e.take_outputs();
    outs.retain(|o| o.id == id);
    assert_eq!(outs.len(), 1);
    outs.remove(0)
}

fn same(a: &Output, b: &Output) -> Result<(), String> {
    if a.tokens != b.tokens {
        let at = a.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y).unwrap_or(a.tokens.len().min(b.tokens.len()));
        return Err(format!("tokens differ at {at} ({} vs {} tokens)", a.tokens.len(), b.tokens.len()));
    }
    if (a.finish, a.stopped) != (b.finish, b.stopped) {
        return Err(format!("finish {:?} vs {:?}", a.finish, b.finish));
    }
    if (a.drafted, a.accepted, a.rounds) != (b.drafted, b.accepted, b.rounds) {
        return Err("speculation counters differ".into());
    }
    Ok(())
}

fn reference(r: &Request) -> Output {
    one(&mut plain(), Request { cache: false, ..r.clone() })
}

/// With nothing running, the pages in use are exactly the pages the index holds, and the pool's free pages are the rest.
fn no_leaks(e: &Engine<MockBackend>) {
    let st = e.prefix_stats();
    assert_eq!(e.backend.pages_in_use(), st.pages_held as usize, "pages in use vs held by the index");
    assert_eq!(e.backend.free_pages() + e.backend.pages_in_use(), POOL_PAGES, "the pool's pages are all accounted for");
    assert_eq!(e.backend.snapshots_in_use(), st.entries as usize);
}

fn end_cut(total: usize) -> usize {
    total - TAIL
}

/// The tester's case: a follow-up turn whose prompt is above `(pool - max_new) / 2` reserves more pages than the pool has free beside its own
/// entry. Crediting the pages the hit shares, it needs none of them evicted: the turn resumes from its entry, and replies as the uncached run.
#[test]
fn a_follow_up_turn_longer_than_half_the_pool_keeps_its_own_entry() {
    let mut e = pooled(4);
    let a_prompt = toks(1, 3000);
    let a = greedy(a_prompt.clone(), 24);
    let oa = one(&mut e, a.clone());
    same(&oa, &reference(&a)).unwrap();
    no_leaks(&e);
    let mut b_prompt = a_prompt;
    b_prompt.extend(toks(2, 300));
    assert!(b_prompt.len() > (POOL_PAGES * MOCK_PAGE - 60) / 2, "the case: a prompt above half the pool");
    let b = request(b_prompt, 60, 5, 0.8);
    let ob = one(&mut e, b.clone());
    same(&ob, &reference(&b)).unwrap();
    assert_eq!(ob.cached_tokens, end_cut(3000), "the follow-up resumed from its own entry");
    assert_eq!(e.prefix_stats().evicted_for_pages, 0, "nothing had to go: the hit's pages are the turn's own");
    no_leaks(&e);
}

/// The pool full of entries when a hit's restore needs its one fresh page (the copy of the partial last page): the room step evicts another
/// entry, never the hit, which the lookup pinned (by recency alone the hit, the older entry, would have gone first).
#[test]
fn the_hit_stays_pinned_while_the_pool_makes_room_for_its_copy() {
    let mut e = pooled(4);
    let a_prompt = toks(11, 3000);
    one(&mut e, greedy(a_prompt.clone(), 24));
    // an unrelated prompt fills the rest of the pool with its own entry
    let u = greedy(toks(12, 1100), 4);
    let ou = one(&mut e, u.clone());
    same(&ou, &reference(&u)).unwrap();
    assert_eq!(e.backend.free_pages(), 0, "the two entries hold the whole pool");
    let mut b_prompt = a_prompt;
    b_prompt.extend(toks(13, 300));
    let b = request(b_prompt, 60, 5, 0.8);
    let ob = one(&mut e, b.clone());
    same(&ob, &reference(&b)).unwrap();
    assert_eq!(ob.cached_tokens, end_cut(3000), "the pinned hit was not evicted");
    assert!(e.prefix_stats().evicted_for_pages >= 1, "the unrelated entry made the room");
    no_leaks(&e);
}

/// A prompt of the whole window (prompt + max_new = the window) with entries in the way: they are evicted for it, and it replies as uncached.
#[test]
fn a_full_window_prompt_evicts_what_it_needs_and_replies_as_uncached() {
    let mut e = pooled(4);
    one(&mut e, greedy(toks(21, 3000), 24));
    let c = greedy(toks(22, WINDOW - 96), 96);
    let oc = one(&mut e, c.clone());
    same(&oc, &reference(&c)).unwrap();
    assert_eq!(oc.cached_tokens, 0);
    assert!(e.prefix_stats().evicted_for_pages >= 1);
    no_leaks(&e);
}

/// The full-window repeat at one slot with a pool of the window: the second request is served from the first one's entry (its end cut), which
/// holds nearly the whole pool, and replies as uncached. A cache with a region of its own would need a second window of pages for this.
#[test]
fn a_full_window_repeat_is_served_from_the_cache_at_one_slot() {
    let mut e = pooled(4);
    let p = toks(31, WINDOW - 96);
    let r1 = greedy(p.clone(), 96);
    let o1 = one(&mut e, r1.clone());
    same(&o1, &reference(&r1)).unwrap();
    no_leaks(&e);
    let r2 = request(p, 96, 9, 0.7);
    let o2 = one(&mut e, r2.clone());
    same(&o2, &reference(&r2)).unwrap();
    assert_eq!(o2.cached_tokens, end_cut(WINDOW - 96), "the repeat resumed from the cache");
    no_leaks(&e);
    // and a third turn still finds an entry of its own conversation
    let r3 = greedy(toks(31, WINDOW - 96), 96);
    assert_eq!(one(&mut e, r3).cached_tokens, end_cut(WINDOW - 96));
    no_leaks(&e);
}

/// An uncached request (`"prefix_cache": false`) still gets its pages: the room step runs for it too, and the entries it displaces are evicted,
/// not leaked.
#[test]
fn an_uncached_request_gets_its_pages_from_idle_entries() {
    let mut e = pooled(4);
    one(&mut e, greedy(toks(41, 3000), 24));
    let c = Request { cache: false, ..greedy(toks(42, WINDOW - 96), 96) };
    let oc = one(&mut e, c.clone());
    same(&oc, &reference(&c)).unwrap();
    no_leaks(&e);
}

/// A conversation of many turns in a pool of the window: every turn after the first resumes from the previous turn's prompt, and replies as
/// uncached, while the entries keep only idle pages.
#[test]
fn a_long_conversation_resumes_every_turn_from_its_last_entry() {
    let mut e = pooled(4);
    let mut prompt = toks(51, 600);
    let mut cases = 0;
    for turn in 0..8u64 {
        let r = request(prompt.clone(), 40, 100 + turn, if turn % 2 == 0 { 0.0 } else { 0.8 });
        let o = one(&mut e, r.clone());
        same(&o, &reference(&r)).unwrap_or_else(|m| panic!("turn {turn}: {m}"));
        if turn > 0 {
            assert!(o.cached_tokens > 0, "turn {turn} missed");
        }
        no_leaks(&e);
        prompt.extend(&o.tokens);
        prompt.extend(toks(200 + turn, 300));
        cases += 1;
        if prompt.len() + 40 > WINDOW {
            break;
        }
    }
    assert!(cases >= 5, "{cases} turns");
}
