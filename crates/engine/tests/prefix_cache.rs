//! The prefix cache at engine level, on the CPU mock with everything a hybrid model has that the cache must carry: paged KV the logits are
//! read from, a recurrent state that depends on the prefill path, and a draft context ring the drafts are computed from (see
//! `gb_engine::mock`). Cached replies must equal uncached ones byte for byte (tokens, finish reason, and the speculation counters), for every
//! cut point, concurrent sharers, eviction, failed and cancelled sequences and slot reuse. Each invariant also has a negative test: the mock
//! can break exactly that invariant (`PrefixFault`), and the scenario must then fail.

use gb_engine::mock::{ChainDrafter, Fault, MockBackend, PrefixFault};
use gb_engine::{
    Backend, BackendError, CacheBudget, Captured, CutPolicy, CutStep, Drafter, Engine, EngineConfig, FinishReason, ImageSpan,
    Output, PrefillStep, PrefixCaps, PrefixConfig, PrefixHandle, Request, RowOutput, RowRequest, SubmitError,
};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};

const VOCAB: usize = 512;
const SLOTS: usize = 8;
/// The cut policy the tests run: small enough that prompts of a few hundred tokens are cut.
const TAIL: usize = 8;

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

fn policy() -> PrefixConfig {
    PrefixConfig {
        policy: CutPolicy { min_cut: 64, tail: TAIL, ghost_min_gain: 256, min_gap: 0 },
        ghost_blocks: 1 << 12,
        ..Default::default()
    }
}

/// A budget of `units` entries (a snapshot slot and 8 pages each); the byte sizes are arbitrary.
fn budget(units: usize) -> CacheBudget {
    CacheBudget { snapshot_slots: units, page_blocks: units * 8, snapshot_bytes: 1 << 20, page_bytes: 1 << 16 }
}

fn mock(slice: Option<usize>, cache: Option<(CacheBudget, PrefixFault)>) -> MockBackend {
    let b = MockBackend::new(VOCAB, SLOTS, Fault::None).with_path_dependent_state().with_native_draft();
    let b = match cache {
        Some((bud, fault)) => b.with_prefix_cache(bud, fault),
        None => b.with_paged_kv(SLOTS * 64),
    };
    match slice {
        Some(s) => b.with_prefill_slice(s),
        None => b,
    }
}

fn cached_engine(units: usize, fault: PrefixFault, slice: Option<usize>) -> Engine<MockBackend> {
    Engine::new(mock(slice, Some((budget(units), fault))), admission(), engine_config(), drafter()).with_prefix_cache(policy())
}

fn cached_engine_with(units: usize, policy: CutPolicy) -> Engine<MockBackend> {
    Engine::new(mock(None, Some((budget(units), PrefixFault::None))), admission(), engine_config(), drafter())
        .with_prefix_cache(PrefixConfig { policy, ghost_blocks: 1 << 12, ..Default::default() })
}

fn plain_engine(slice: Option<usize>) -> Engine<MockBackend> {
    Engine::new(mock(slice, None), admission(), engine_config(), drafter())
}

/// Deterministic tokens that are not degenerate (no long repeats).
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

/// Submit, prefill and run to the end; the finished output.
fn one<B: Backend>(e: &mut Engine<B>, r: Request) -> Output {
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
        return Err(format!(
            "speculation counters differ: drafted/accepted/rounds {}/{}/{} vs {}/{}/{}",
            a.drafted, a.accepted, a.rounds, b.drafted, b.accepted, b.rounds
        ));
    }
    Ok(())
}

/// Tokens and finish reason only: the speculation counters of a request depend on who else is in the round (the rows the budget hands each
/// stream), so a request that ran beside others is compared on what it answered, not on how many drafts it was offered.
fn same_reply(a: &Output, b: &Output) -> Result<(), String> {
    let (ta, tb) =
        (Output { drafted: 0, accepted: 0, rounds: 0, ..a.clone() }, Output { drafted: 0, accepted: 0, rounds: 0, ..b.clone() });
    same(&ta, &tb)
}

/// The reply of `r` on an engine that has no cache, alone.
fn reference(r: &Request, slice: Option<usize>) -> Output {
    let mut e = plain_engine(slice);
    one(&mut e, Request { cache: false, ..r.clone() })
}

/// Pages and snapshots are all accounted for: with nothing running, what the backend holds is exactly what the index holds.
fn no_leaks(e: &Engine<MockBackend>) -> Result<(), String> {
    let st = e.prefix_stats();
    if e.backend.pages_in_use() != st.pages_held as usize {
        return Err(format!("{} pages in use, the index holds {}", e.backend.pages_in_use(), st.pages_held));
    }
    if e.backend.snapshots_in_use() != st.entries as usize {
        return Err(format!("{} snapshot slots in use, the index holds {} entries", e.backend.snapshots_in_use(), st.entries));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------------------------------
// the contract: cached == uncached, for every cut point
// ------------------------------------------------------------------------------------------------------------------------------------

/// The end cut of a prompt of `total` tokens (what the policy plans for an otherwise cold prompt), or 0.
fn end_cut(total: usize) -> usize {
    let c = total.saturating_sub(TAIL);
    if total > 128 && c >= 64 {
        c
    } else {
        0
    }
}

fn sweep(slice: Option<usize>) -> Result<usize, String> {
    let mut cases = 0;
    for l1 in [129usize, 130, 192, 200, 255, 256, 257, 300, 511, 512, 513, 700, 1000, 2100] {
        for ext in [1usize, 2, 9, 17, 40, 300] {
            let a_prompt = toks(l1 as u64, l1);
            let mut b_prompt = a_prompt.clone();
            b_prompt.extend(toks(1000 + ext as u64, ext));
            let (a, b) = (greedy(a_prompt, 24), request(b_prompt, 40, 5, 0.8));
            let mut e = cached_engine(4, PrefixFault::None, slice);
            let oa = one(&mut e, a.clone());
            same(&oa, &reference(&a, slice)).map_err(|m| format!("A at {l1}: {m}"))?;
            assert_eq!(oa.cached_tokens, 0, "a cold server has nothing to hit");
            let ob = one(&mut e, b.clone());
            same(&ob, &reference(&b, slice)).map_err(|m| format!("B at {l1}+{ext}: {m}"))?;
            if ob.cached_tokens != end_cut(l1) {
                return Err(format!("B at {l1}+{ext} hit {} tokens, expected {}", ob.cached_tokens, end_cut(l1)));
            }
            // and the very same prompt again hits too, and is again byte-identical
            let oc = one(&mut e, a.clone());
            same(&oc, &oa).map_err(|m| format!("A again at {l1}: {m}"))?;
            // the longest stored prefix: A's end cut, or B's when B's prompt ran at most 7 tokens past A's (B's end cut is then inside A)
            let longest = end_cut(l1).max(Some(end_cut(l1 + ext)).filter(|&c| c < l1).unwrap_or(0));
            if oc.cached_tokens != longest {
                return Err(format!("A again at {l1}+{ext} hit {}, expected {longest}", oc.cached_tokens));
            }
            no_leaks(&e).map_err(|m| format!("at {l1}+{ext}: {m}"))?;
            cases += 3;
        }
    }
    Ok(cases)
}

#[test]
fn cached_replies_equal_uncached_replies_at_every_cut_point() {
    for slice in [None, Some(64), Some(100), Some(300)] {
        let n = sweep(slice).unwrap_or_else(|m| panic!("slice {slice:?}: {m}"));
        assert_eq!(n, 14 * 6 * 3);
    }
}

#[test]
fn a_hit_counts_in_the_stats_and_saves_the_prefill() {
    let mut e = cached_engine(4, PrefixFault::None, None);
    let a = greedy(toks(1, 900), 20);
    one(&mut e, a.clone());
    let s = e.prefix_stats();
    assert_eq!((s.lookups, s.hits, s.tokens_saved, s.entries, s.enabled), (1, 0, 0, 1, true));
    assert_eq!(s.lookup_tokens, 900);
    one(&mut e, a);
    let s = e.prefix_stats();
    assert_eq!((s.lookups, s.hits, s.tokens_saved), (2, 1, 892));
    assert!((s.hit_rate() - 0.5).abs() < 1e-9);
    assert_eq!(s.bytes_held, s.entries * (1 << 20) + s.pages_held * (1 << 16));
}

// ------------------------------------------------------------------------------------------------------------------------------------
// the path rule: nothing of 128 tokens or fewer is cached, and nothing crosses the boundary
// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn a_cut_within_the_gap_of_the_state_the_prefill_holds_is_not_taken() {
    // min_gap 256: a turn that adds fewer than 256 tokens past the entry it resumed from leaves no new entry (the earlier state wins), one that adds
    // more does
    let mut e = cached_engine_with(6, CutPolicy { min_cut: 64, tail: TAIL, ghost_min_gain: 256, min_gap: 256 });
    let base = toks(501, 1200);
    one(&mut e, greedy(base.clone(), 8));
    assert_eq!(e.prefix_stats().entries, 1);
    let ext = |n: usize, seed: u64| {
        let mut p = base.clone();
        p.extend(toks(seed, n));
        p
    };
    // +140 tokens: the end cut (1,324) is 132 past the hit (1,192): not taken, and the reply is still exact
    let r = greedy(ext(140, 502), 8);
    let o = one(&mut e, r.clone());
    assert_eq!((o.cached_tokens, e.prefix_stats().entries), (1192, 1));
    same(&o, &reference(&Request { cache: false, ..r }, None)).unwrap();
    // +300 tokens: 300 past the hit, taken
    let o = one(&mut e, greedy(ext(300, 503), 8));
    assert_eq!((o.cached_tokens, e.prefix_stats().entries), (1192, 2));
    // the next turn of that conversation resumes from the new entry
    let mut longer = ext(300, 503);
    longer.extend(toks(504, 30));
    assert_eq!(one(&mut e, greedy(longer, 8)).cached_tokens, 1492);
}

#[test]
fn a_prefix_two_requests_resumed_from_outlives_a_flood_of_one_shot_prompts() {
    // three entries at most. The shared prompt's entry is the oldest and two requests have resumed from it; a flood of one-shot prompts must evict each
    // other, not it. (An entry that has hit once is not protected: that is every conversation's superseded turn, see the index's tests.)
    let mut e = cached_engine(3, PrefixFault::None, None);
    let shared = toks(511, 800);
    one(&mut e, greedy(shared.clone(), 8));
    for _ in 0..2 {
        assert_eq!(one(&mut e, greedy(shared.clone(), 8)).cached_tokens, 792, "it hits");
    }
    for i in 0..6u64 {
        one(&mut e, greedy(toks(520 + i, 700), 8));
    }
    let s = e.prefix_stats();
    assert!(s.evictions >= 4 && s.evicted_for_slots >= 1, "{s:?}");
    assert_eq!(one(&mut e, greedy(shared, 8)).cached_tokens, 792, "the shared prefix survived the flood");
}

#[test]
fn a_prompt_of_128_tokens_or_fewer_never_touches_the_cache() {
    let mut e = cached_engine(4, PrefixFault::None, None);
    for total in [1usize, 17, 100, 127, 128] {
        let r = greedy(toks(total as u64, total), 16);
        let o = one(&mut e, r.clone());
        same(&o, &reference(&r, None)).unwrap();
        assert_eq!(o.cached_tokens, 0);
    }
    let s = e.prefix_stats();
    assert_eq!((s.lookups, s.hits, s.entries, s.inserts), (0, 0, 0, 0), "GEMV-path prompts neither look up nor leave anything");
    no_leaks(&e).unwrap();
}

/// A prompt of exactly 128 tokens, then one of 329 that starts with it: the 129+ one is on another path and must not resume from anything the
/// short one could have left.
fn boundary_scenario(fault: PrefixFault) -> Result<(), String> {
    let short = toks(77, 128);
    let mut long = short.clone();
    long.extend(toks(78, 201));
    let mut e = cached_engine(4, fault, None);
    let o1 = one(&mut e, greedy(short.clone(), 40));
    same(&o1, &reference(&greedy(short, 40), None))?;
    let r2 = request(long, 60, 3, 1.0);
    let o2 = one(&mut e, r2.clone());
    same(&o2, &reference(&r2, None)).map_err(|m| format!("the longer prompt after the short one: {m}"))
}

#[test]
fn a_prefix_is_never_reused_across_the_128_token_boundary() {
    boundary_scenario(PrefixFault::None).unwrap();
}

#[test]
fn mutant_path_ignored_is_caught() {
    let e = boundary_scenario(PrefixFault::PathIgnored).unwrap_err();
    println!("path ignored: {e}");
    assert!(e.contains("tokens differ") || e.contains("speculation counters differ"), "{e}");
}

// ------------------------------------------------------------------------------------------------------------------------------------
// copy-on-write, refcounts, eviction of a referenced entry, draft context, recurrent state
// ------------------------------------------------------------------------------------------------------------------------------------

/// A is decoding while B (sharing A's prompt up to the end cut, a partial page) is prefilled over the same prefix and then both decode.
fn donor_decoding_scenario(fault: PrefixFault, slice: Option<usize>) -> Result<(), String> {
    let a_prompt = toks(11, 700);
    let mut b_prompt = a_prompt[..end_cut(700)].to_vec();
    b_prompt.extend(toks(12, 120));
    let (a, b) = (greedy(a_prompt, 90), request(b_prompt, 80, 9, 0.7));
    let mut e = cached_engine(4, fault, slice);
    let ida = e.submit(a.clone()).unwrap();
    for _ in 0..3 {
        e.step().map_err(|x| x.to_string())?; // A is decoding
    }
    let idb = e.submit(b.clone()).unwrap(); // B restores from A's entry and prefills its suffix while A is mid-decode
    e.run().map_err(|x| x.to_string())?;
    let outs = e.take_outputs();
    let get = |id| outs.iter().find(|o| o.id == id).unwrap();
    same_reply(get(ida), &reference(&a, slice)).map_err(|m| format!("the donor: {m}"))?;
    same_reply(get(idb), &reference(&b, slice)).map_err(|m| format!("the sharer: {m}"))?;
    if get(idb).cached_tokens != 692 {
        return Err(format!("B hit {} tokens, expected 692", get(idb).cached_tokens));
    }
    no_leaks(&e)
}

#[test]
fn a_sharer_prefilling_over_the_partial_page_leaves_the_donor_alone() {
    for slice in [None, Some(100)] {
        donor_decoding_scenario(PrefixFault::None, slice).unwrap();
    }
}

#[test]
fn mutant_no_copy_on_write_is_caught() {
    let e = donor_decoding_scenario(PrefixFault::NoCopyOnWrite, None).unwrap_err();
    println!("no copy on write: {e}");
    assert!(e.contains("the donor") || e.contains("the sharer"), "{e}");
}

/// An entry is evicted while a sequence that restored from it still runs; a newcomer then takes pages from the free list.
fn eviction_while_shared_scenario(fault: PrefixFault) -> Result<(), String> {
    let a = greedy(toks(21, 700), 10);
    let mut b_prompt = toks(21, 700)[..end_cut(700)].to_vec();
    b_prompt.extend(toks(22, 60));
    let b = request(b_prompt, 120, 4, 0.0);
    // one entry at most: B's capture needs A's slot, so A's entry is evicted while B (which restored from it) is decoding on its shared pages
    let mut e = cached_engine(1, fault, None);
    one(&mut e, a);
    let idb = e.submit(b.clone()).unwrap();
    for _ in 0..2 {
        e.step().map_err(|x| x.to_string())?;
    }
    let mut cs = Vec::new();
    for i in 0..2u64 {
        let c = greedy(toks(100 + i, 600), 6);
        cs.push((c.clone(), e.submit(c).unwrap()));
    }
    e.run().map_err(|x| x.to_string())?;
    let outs = e.take_outputs();
    let get = |id| outs.iter().find(|o| o.id == id).unwrap();
    same_reply(get(idb), &reference(&b, None)).map_err(|m| format!("the sequence sharing the evicted entry's pages: {m}"))?;
    for (c, id) in &cs {
        same_reply(get(*id), &reference(c, None)).map_err(|m| format!("a newcomer: {m}"))?;
    }
    if e.prefix_stats().evictions == 0 {
        return Err("the scenario evicted nothing".into());
    }
    no_leaks(&e)
}

#[test]
fn evicting_an_entry_a_running_sequence_shares_frees_nothing_it_uses() {
    eviction_while_shared_scenario(PrefixFault::None).unwrap();
}

#[test]
fn mutant_freeing_pages_that_are_still_referenced_is_caught() {
    // the mock itself may trip over the damage (a double release is a panic): a panic is a detection too
    let r = std::panic::catch_unwind(|| eviction_while_shared_scenario(PrefixFault::FreeWhileReferenced));
    println!("free while referenced: {r:?}");
    assert!(r.map_or(true, |r| r.is_err()), "freeing a referenced page must corrupt a reply or break the books");
}

#[test]
fn mutant_refcount_leak_is_caught() {
    assert!(
        eviction_while_shared_scenario(PrefixFault::RefcountLeak).is_err(),
        "pages that were never given back must show in the books"
    );
}

/// The draft's context after a hit: the speculation counters of the cached run equal the uncached run's only when the window was restored.
fn draft_scenario(fault: PrefixFault) -> Result<(), String> {
    for (l1, ext) in [(300usize, 150usize), (800, 200), (2300, 120), (3000, 500)] {
        let a_prompt = toks(31 + l1 as u64, l1);
        let mut b_prompt = a_prompt.clone();
        b_prompt.extend(toks(32, ext));
        let (a, b) = (greedy(a_prompt, 20), greedy(b_prompt, 120));
        let mut e = cached_engine(4, fault, None);
        one(&mut e, a);
        // an unrelated prompt as long as B's takes the slot (and its draft ring) after A: a ring that is not restored holds its keys
        one(&mut e, greedy(toks(900 + l1 as u64, l1 + ext + 130), 8));
        let ob = one(&mut e, b.clone());
        if ob.cached_tokens == 0 {
            return Err("no hit".into());
        }
        same(&ob, &reference(&b, None)).map_err(|m| format!("at {l1}+{ext}: {m}"))?;
    }
    Ok(())
}

#[test]
fn a_hit_gives_the_draft_the_context_an_uncached_run_has() {
    draft_scenario(PrefixFault::None).unwrap();
}

#[test]
fn mutant_draft_context_not_restored_is_caught() {
    let e = draft_scenario(PrefixFault::DraftNotRestored).unwrap_err();
    println!("draft not restored: {e}");
    assert!(e.contains("speculation counters differ"), "{e}");
}

#[test]
fn mutant_recurrent_state_not_restored_is_caught() {
    let e = draft_scenario(PrefixFault::StateNotRestored).unwrap_err();
    println!("state not restored: {e}");
    assert!(e.contains("tokens differ") || e.contains("speculation counters differ"), "{e}");
}

// ------------------------------------------------------------------------------------------------------------------------------------
// concurrency, pressure, slot reuse, sampling
// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn concurrent_sharers_of_one_prefix_all_equal_their_uncached_replies() {
    for slice in [None, Some(64)] {
        let sys = toks(41, 900);
        let mut e = cached_engine(6, PrefixFault::None, slice);
        // two requests with the same system prompt leave the shared entry (the second one discovers the shared blocks); five more arrive
        // together, each with its own question, sampled with its own seed
        for q in 0..2u64 {
            let mut p = sys.clone();
            p.extend(toks(42 + q, 30));
            one(&mut e, greedy(p, 12));
        }
        let reqs: Vec<Request> = (0..5u64)
            .map(|i| {
                let mut p = sys.clone();
                p.extend(toks(50 + i, 20 + 7 * i as usize));
                request(p, 50 + 5 * i as usize, 100 + i, if i % 2 == 0 { 0.0 } else { 1.0 })
            })
            .collect();
        let ids: Vec<u64> = reqs.iter().map(|r| e.submit_deferred(r.clone()).unwrap()).collect();
        e.run().unwrap();
        let outs = e.take_outputs();
        for (r, id) in reqs.iter().zip(&ids) {
            let o = outs.iter().find(|o| o.id == *id).unwrap();
            same_reply(o, &reference(r, slice)).unwrap_or_else(|m| panic!("slice {slice:?}: {m}"));
            assert_eq!(o.cached_tokens, 768, "every sharer hits the shared blocks (slice {slice:?})");
        }
        no_leaks(&e).unwrap();
    }
}

#[test]
fn a_long_run_of_mixed_traffic_under_a_tight_budget_stays_exact_and_balanced() {
    // 3 entries at most, 24 pages; 40 requests drawn from 5 conversations of growing prompts, so entries are inserted, hit and evicted all along
    let mut e = cached_engine(3, PrefixFault::None, Some(128));
    let mut rng = 0x1234_5678_u64;
    let mut next = |m: u64| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng >> 33) % m
    };
    let mut convo: Vec<Vec<u32>> = (0..5u64).map(|c| toks(200 + c, 300 + 100 * c as usize)).collect();
    let mut hits = 0;
    for i in 0..40u64 {
        let c = next(5) as usize;
        let ext = 5 + next(120) as usize;
        convo[c].extend(toks(300 + i, ext));
        let r = request(convo[c].clone(), 30, i, if next(2) == 0 { 0.0 } else { 0.9 });
        let o = one(&mut e, r.clone());
        same(&o, &reference(&r, Some(128))).unwrap_or_else(|m| panic!("request {i}: {m}"));
        hits += (o.cached_tokens > 0) as usize;
        let s = e.prefix_stats();
        assert!(s.entries <= 3 && s.pages_held <= 24, "{s:?}");
        no_leaks(&e).unwrap();
    }
    assert!(hits >= 8, "the traffic is built to hit ({hits})");
    assert!(e.prefix_stats().evictions > 0, "and to evict");
}

#[test]
fn slot_reuse_never_shows_the_previous_owners_state() {
    // one slot: every request lands in it after another's; the restored state, pages and draft ring must replace the old ones entirely
    let mut e = Engine::new(
        MockBackend::new(VOCAB, 1, Fault::None)
            .with_path_dependent_state()
            .with_native_draft()
            .with_prefix_cache(budget(4), PrefixFault::None),
        {
            let mm =
                ModelMemory { weights_bytes: 0, kv_bytes_per_token: 65536, state_bytes_per_slot: 1 << 20, activation_reserve: 0 };
            Admission::new(1 << 40, 0, mm, 1).unwrap()
        },
        engine_config(),
        drafter(),
    )
    .with_prefix_cache(policy());
    let base = toks(61, 800);
    let mut prompts = vec![base.clone()];
    let mut ext = base.clone();
    ext.extend(toks(62, 90));
    prompts.push(ext);
    prompts.push(toks(63, 500)); // unrelated, takes the slot between
    let mut again = base.clone();
    again.extend(toks(64, 60));
    prompts.push(again);
    prompts.push(base);
    for (i, p) in prompts.into_iter().enumerate() {
        let r = request(p, 40, i as u64, 0.6);
        let o = one(&mut e, r.clone());
        same(&o, &reference(&r, None)).unwrap_or_else(|m| panic!("request {i}: {m}"));
    }
    assert!(e.prefix_stats().hits >= 3);
}

#[test]
fn drafted_equals_serial_with_the_cache() {
    // the same request with speculation on and off (the serial path) on a cached engine: identical tokens, hit or miss
    let sys = toks(71, 1000);
    let mut e = cached_engine(4, PrefixFault::None, None);
    for i in 0..6u64 {
        let mut p = sys.clone();
        p.extend(toks(72 + i, 30));
        let drafted = request(p.clone(), 60, i, 0.0);
        let serial = Request { draft: false, ..drafted.clone() };
        let (od, os) = (one(&mut e, drafted), one(&mut e, serial));
        assert_eq!(od.tokens, os.tokens, "request {i}");
        assert_eq!(os.drafted, 0);
        assert!(i < 2 || od.cached_tokens > 0, "from the third prompt on the shared blocks are stored");
    }
}

#[test]
fn a_request_that_opts_out_neither_looks_up_nor_leaves_anything() {
    let mut e = cached_engine(4, PrefixFault::None, None);
    let r = Request { cache: false, ..greedy(toks(81, 600), 12) };
    let o = one(&mut e, r);
    let s = e.prefix_stats();
    assert_eq!((o.cached_tokens, s.lookups, s.entries), (0, 0, 0));
    one(&mut e, greedy(toks(81, 600), 12));
    assert_eq!(e.prefix_stats().entries, 1);
    let o = one(&mut e, Request { cache: false, ..greedy(toks(81, 600), 12) });
    assert_eq!(o.cached_tokens, 0, "an opted-out request does not hit what is stored either");
}

/// A request whose prompt has `placeholders` image tokens at `start`, the image identified by `key` (the mock treats them as text; what the
/// cache must do is key on the image, not on the placeholder token).
fn with_image(prompt: Vec<u32>, start: usize, placeholders: usize, key: u64) -> Request {
    let side = (placeholders as f64).sqrt() as usize;
    assert_eq!(side * side, placeholders);
    let mut p = prompt;
    p[start..start + placeholders].iter_mut().for_each(|t| *t = 7); // the same `<|image_pad|>` id for every image
    let span = ImageSpan { start, tokens: placeholders, merged: (side, side), features: std::sync::Arc::new(()), key };
    Request { images: vec![span], ..greedy(p, 12) }
}

#[test]
fn an_image_prompt_is_cached_by_the_content_of_its_images() {
    let mut e = cached_engine(6, PrefixFault::None, None);
    let text = toks(91, 900);
    let (a, same_image, other_image) = (
        with_image(text.clone(), 100, 256, 111),
        with_image(text.clone(), 100, 256, 111),
        with_image(text.clone(), 100, 256, 222), // the same text and placeholder ids around another image
    );
    let o1 = one(&mut e, a.clone());
    assert_eq!(o1.cached_tokens, 0);
    let o2 = one(&mut e, same_image);
    assert_eq!(o2.cached_tokens, 892, "the same text and the same image resume from the end cut");
    same(&o2, &reference(&Request { cache: false, ..a }, None)).unwrap();
    let o3 = one(&mut e, other_image);
    assert_eq!(
        o3.cached_tokens, 0,
        "another image under the same text is another prefix: nothing may be shared across the image"
    );
    // and the plain-text prompt with the same ids is another prefix as well (the placeholder ids alone do not name an image)
    let mut plain = text.clone();
    plain[100..356].iter_mut().for_each(|t| *t = 7);
    assert_eq!(one(&mut e, greedy(plain, 12)).cached_tokens, 0);
}

#[test]
fn an_image_with_no_key_is_never_cached() {
    let mut e = cached_engine(4, PrefixFault::None, None);
    let r = with_image(toks(92, 900), 100, 256, 0);
    one(&mut e, r.clone());
    one(&mut e, r);
    let s = e.prefix_stats();
    assert_eq!(
        (s.lookups, s.entries),
        (0, 0),
        "an image of unknown content cannot be named, so its request does not touch the cache"
    );
}

#[test]
fn a_cut_that_would_fall_inside_an_image_moves_in_front_of_it() {
    // 720 tokens, the image covers 200..456 and 456..712 is text: the end cut (712) is outside; with the image reaching to 718 it is inside
    let mut e = cached_engine(6, PrefixFault::None, None);
    let r = with_image(toks(93, 720), 200, 256, 5);
    one(&mut e, r.clone());
    assert_eq!(one(&mut e, r).cached_tokens, 712);
    let r = with_image(toks(94, 720), 234, 484, 6);
    let first = one(&mut e, r.clone());
    assert_eq!(first.cached_tokens, 0);
    let second = one(&mut e, r.clone());
    assert_eq!(
        second.cached_tokens, 234,
        "the end cut (712) is inside the image (234..718): the snapshot is taken in front of it"
    );
    same(&second, &reference(&Request { cache: false, ..r }, None)).unwrap();
    // an image that starts below the minimum cut leaves nothing to cut in front of it
    let early = with_image(toks(95, 720), 40, 676, 7); // 26 x 26 placeholders, 40..716
    one(&mut e, early.clone());
    assert_eq!(one(&mut e, early).cached_tokens, 0);
}

#[test]
fn a_second_conversation_about_the_same_image_leaves_its_own_end_cut() {
    // the image covers 40..716 (below the minimum cut at its start) and 716..760 is text. A second prompt with the same image and another text after the
    // first 740 tokens has its block-aligned shared prefix (512) INSIDE the image, which is dropped; the end cut (752) is within the gap of that shared cut,
    // and must not be suppressed by a cut that was never taken: the second conversation's own follow-up resumes from it
    let policy = CutPolicy { min_cut: 64, tail: TAIL, ghost_min_gain: 256, min_gap: 256 };
    let mut e = cached_engine_with(6, policy);
    let a = with_image(toks(96, 760), 40, 676, 8);
    one(&mut e, a.clone());
    let mut second_tokens = a.prompt.clone();
    second_tokens.truncate(740);
    second_tokens.extend(toks(97, 20));
    let a2 = Request { prompt: second_tokens, ..a.clone() };
    let o2 = one(&mut e, a2.clone());
    assert_eq!(o2.cached_tokens, 0, "the first prompt's entry (752) is not a prefix of this one (they differ from 740)");
    let mut follow = a2.prompt.clone();
    follow.extend(toks(98, 50));
    let r = Request { prompt: follow, ..a2.clone() };
    let o3 = one(&mut e, r.clone());
    assert_eq!(o3.cached_tokens, 752, "the second conversation's follow-up resumes from ITS end cut");
    same(&o3, &reference(&Request { cache: false, ..r }, None)).unwrap();
}

#[test]
fn a_shared_system_prompt_is_found_on_the_second_sighting() {
    // three requests share 3 full blocks (800 tokens: 3 x 256 + 32) and differ after them. The first leaves only its own end cut; the second
    // (a miss) discovers the shared blocks and snapshots at 768; the third hits it.
    let sys = toks(101, 800);
    let q = |i: u64| {
        let mut p = sys.clone();
        p.extend(toks(102 + i, 120));
        greedy(p, 10)
    };
    let mut e = cached_engine(6, PrefixFault::None, None);
    let hits: Vec<usize> = (0..4).map(|i| one(&mut e, q(i)).cached_tokens).collect();
    assert_eq!(hits, vec![0, 0, 768, 768]);
    // the replies equal the uncached ones, hit or not
    let mut e = cached_engine(6, PrefixFault::None, None);
    for i in 0..4 {
        same(&one(&mut e, q(i)), &reference(&q(i), None)).unwrap();
    }
}

#[test]
fn a_multi_turn_conversation_resumes_from_the_previous_prompt_each_turn() {
    // each turn's prompt is the previous prompt's history (without its generation prompt, the last 3 tokens, which the next turn renders
    // differently), the reply, and a new user message
    let mut e = cached_engine(4, PrefixFault::None, Some(128));
    let mut history = toks(111, 1200);
    let mut last_total = 0;
    for turn in 0..6u64 {
        let user = toks(112 + turn, 60);
        let mut prompt = history.clone();
        prompt.extend(&user);
        prompt.extend([3u32, 4, 5]); // the generation prompt
        let r = request(prompt.clone(), 80, turn, 0.0);
        let o = one(&mut e, r.clone());
        same(&o, &reference(&r, Some(128))).unwrap_or_else(|m| panic!("turn {turn}: {m}"));
        if turn > 0 {
            assert_eq!(o.cached_tokens, end_cut(last_total), "turn {turn} resumes from the previous prompt's end cut");
        }
        last_total = prompt.len();
        history = prompt[..prompt.len() - 3].to_vec();
        history.extend(&o.tokens);
    }
    assert_eq!(e.prefix_stats().hits, 5);
}

// ------------------------------------------------------------------------------------------------------------------------------------
// failure isolation
// ------------------------------------------------------------------------------------------------------------------------------------

/// A backend that fails on demand around a mock with a cache: a NaN row for prompts starting with `poison`, a restore that errors.
struct Flaky {
    inner: MockBackend,
    poison: Option<u32>,
    fail_restore: bool,
    error: Option<BackendError>,
}

impl Flaky {
    fn new(inner: MockBackend) -> Self {
        Self { inner, poison: None, fail_restore: false, error: None }
    }
}

impl Backend for Flaky {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn take_error(&mut self) -> Option<BackendError> {
        self.error.take().or_else(|| self.inner.take_error())
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.inner.prefill(slot, start, tokens, candidates)
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.inner.step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.inner.commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.inner.release(slot)
    }
    fn propose(&mut self, slot: usize, tokens: &[u32], s: &Sampling, max: usize) -> Option<Vec<u32>> {
        self.inner.propose(slot, tokens, s, max)
    }
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> PrefillStep {
        self.prefill_slice_cuts(slot, prompt, images, done, candidates, &[]).step
    }
    fn prefill_slice_cuts(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> CutStep {
        let mut st = self.inner.prefill_slice_cuts(slot, prompt, images, done, candidates, cuts);
        if self.poison.is_some() && prompt.first().copied() == self.poison {
            if let Some(r) = st.step.row.as_mut() {
                r.values.iter_mut().for_each(|v| *v = f32::NAN); // the snapshots in `st.captured` stay: the engine must give them back
            }
        }
        st
    }
    fn prefix_caps(&self) -> Option<PrefixCaps> {
        self.inner.prefix_caps()
    }
    fn prefix_path(&self, total: usize) -> Option<u32> {
        self.inner.prefix_path(total)
    }
    fn prefix_free_snapshots(&self) -> usize {
        self.inner.prefix_free_snapshots()
    }
    fn prefix_restore(&mut self, slot: usize, h: &PrefixHandle, len: usize) -> Result<(), String> {
        if self.fail_restore {
            self.fail_restore = false;
            self.inner.prefix_restore(slot, h, len).ok();
            self.error = Some(BackendError::retryable("injected restore failure"));
            return Err("injected restore failure".into());
        }
        self.inner.prefix_restore(slot, h, len)
    }
    fn prefix_release(&mut self, h: PrefixHandle) {
        self.inner.prefix_release(h)
    }
}

fn flaky_engine() -> Engine<Flaky> {
    Engine::new(Flaky::new(mock(None, Some((budget(4), PrefixFault::None)))), admission(), engine_config(), drafter())
        .with_prefix_cache(policy())
}

#[test]
fn a_sequence_that_fails_after_its_snapshots_leaves_no_entry_and_no_leak() {
    let mut e = flaky_engine();
    let bad = greedy(vec![9; 1].into_iter().chain(toks(121, 899)).collect(), 10);
    e.backend.poison = Some(9);
    let err = e.submit(bad.clone()).unwrap_err();
    assert!(matches!(err, SubmitError::Sampling(_)), "{err}");
    let s = e.prefix_stats();
    assert_eq!((s.entries, s.inserts), (0, 0), "nothing a failed sequence computed is stored");
    assert!(s.captures_discarded >= 1, "{s:?}");
    assert_eq!((e.backend.inner.pages_in_use(), e.backend.inner.snapshots_in_use()), (0, 0), "nothing leaks");
    // a good request that shares the bad one's prefix is unaffected (and sees no entry of it)
    e.backend.poison = None;
    let good = Request { prompt: bad.prompt.clone(), ..greedy(vec![], 10) };
    let o = one(&mut e, good.clone());
    assert_eq!(o.cached_tokens, 0);
    same(&o, &reference(&good, None)).unwrap();
}

#[test]
fn a_failed_restore_drops_the_entry_and_prefills_cold() {
    let mut e = flaky_engine();
    let r = greedy(toks(131, 700), 30);
    one(&mut e, r.clone());
    assert_eq!(e.prefix_stats().entries, 1);
    e.backend.fail_restore = true;
    let o = one(&mut e, r.clone());
    assert_eq!(o.cached_tokens, 0, "the failed restore is not a hit");
    same(&o, &reference(&r, None)).unwrap();
    let s = e.prefix_stats();
    assert_eq!(s.restore_failures, 1);
    // the suspect entry was dropped and the cold prefill left fresh ones (the end cut, and the shared blocks this second sighting found)
    assert_eq!(s.entries, 2);
    let o = one(&mut e, r.clone());
    assert_eq!(o.cached_tokens, 692);
    same(&o, &reference(&r, None)).unwrap();
    assert_eq!((e.backend.inner.pages_in_use(), e.backend.inner.snapshots_in_use()), (e.prefix_stats().pages_held as usize, 2));
}

#[test]
fn cancelling_a_prefill_releases_its_snapshots_and_cancelling_a_decode_keeps_the_entry() {
    // a prompt of 1,000 tokens prefilled in slices of 331 (331, 662, 993, 1,000): the end cut (992) is passed in the third slice, and the
    // prompt is still one slice short of complete
    let mut e = cached_engine(4, PrefixFault::None, Some(331));
    let r = greedy(toks(141, 1000), 400);
    let id = e.submit_deferred(r.clone()).unwrap();
    for _ in 0..3 {
        e.step().unwrap();
    }
    assert!(e.prefilling() == 1 && e.backend.snapshots_in_use() == 1, "the cut was passed and is pending");
    assert!(e.cancel(id));
    let s = e.prefix_stats();
    assert_eq!((s.entries, s.captures_discarded), (0, 1));
    assert_eq!((e.backend.pages_in_use(), e.backend.snapshots_in_use()), (0, 0), "a cancelled prefill gives everything back");
    // the same prompt again, cancelled while DECODING: its entry (already in the index) stays and serves the next request
    let id = e.submit(r.clone()).unwrap();
    e.step().unwrap();
    assert!(e.cancel(id));
    // two entries: the end cut, and the shared blocks (768) the cancelled attempt's sighting had recorded
    assert_eq!(e.prefix_stats().entries, 2);
    let o = one(&mut e, Request { max_new: 20, ..r.clone() });
    assert_eq!(o.cached_tokens, 992);
    same(&o, &reference(&Request { max_new: 20, ..r }, Some(331))).unwrap();
    e.take_outputs();
    no_leaks(&e).unwrap();
}

#[test]
fn aborting_everything_gives_back_what_prefills_had_pending() {
    let mut e = cached_engine(4, PrefixFault::None, Some(331));
    for i in 0..3u64 {
        e.submit_deferred(greedy(toks(151 + i, 1000), 50)).unwrap();
    }
    for _ in 0..9 {
        e.step().unwrap(); // each prompt gets three slices in turn: all three have passed their end cut
    }
    assert_eq!(e.backend.snapshots_in_use(), 3);
    e.abort_all();
    let s = e.prefix_stats();
    assert_eq!(s.entries, 0);
    assert_eq!((e.backend.pages_in_use(), e.backend.snapshots_in_use()), (0, 0));
}

#[test]
fn a_backend_without_a_cache_leaves_the_engine_uncached() {
    // the default `Backend` (the GLM backend's situation): `with_prefix_cache` changes nothing
    let mut e = Engine::new(MockBackend::new(VOCAB, SLOTS, Fault::None), admission(), engine_config(), drafter())
        .with_prefix_cache(policy());
    let r = greedy(toks(161, 600), 10);
    one(&mut e, r.clone());
    let o = one(&mut e, r);
    assert_eq!((o.cached_tokens, e.prefix_stats().enabled), (0, false));
    // and a mock whose budget cannot hold two entries disables itself as well
    let e = Engine::new(
        MockBackend::new(VOCAB, SLOTS, Fault::None)
            .with_prefix_cache(CacheBudget { snapshot_slots: 0, ..budget(0) }, PrefixFault::None),
        admission(),
        engine_config(),
        drafter(),
    )
    .with_prefix_cache(policy());
    assert!(!e.prefix_stats().enabled);
}

#[test]
fn an_unfinished_reply_is_still_a_finished_output_with_its_cached_count() {
    // `Output::cached_tokens` is set when the prompt is complete and survives the decode
    let mut e = cached_engine(4, PrefixFault::None, None);
    let r = greedy(toks(171, 500), 30);
    one(&mut e, r.clone());
    let o = one(&mut e, r);
    assert_eq!((o.cached_tokens, o.finish), (492, Some(FinishReason::Length)));
}

#[allow(dead_code)]
fn _captured_is_public(c: Captured) -> usize {
    c.len
}
