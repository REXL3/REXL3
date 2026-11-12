//! The prefix cache at engine level on the CPU mock in its GLM shape (`MockBackend::with_glm_shape`; design: `docs/PREFIX_CACHE.md`, "GLM-5.3-Flash").
//!
//! What differs from the Qwen-shaped mock of `prefix_cache.rs`, and what these tests pin:
//!
//! * **One arithmetic.** GLM has no path rule (every prompt chunk, whatever its size or position, leaves the bits of the same token decoded alone), so a
//!   prompt of ANY length is cacheable and resumes exactly: the mock answers `prefix_path` for every total.
//! * **The raw-row ring.** A GLM slot keeps the last raw rows of each indexer layer outside its pages. A prefix adopted at a length that is not a multiple
//!   of 4 must bring the `len % 4` rows of its incomplete pool group, or the first pool completed after the cut is built from stale rows. The pooled-key
//!   plane lives in the PAGES (so the copy of the partial last page must carry it) and steers the logits (a wrong pool shows in the reply once the
//!   context is sparse; the mock has no dense regime, so it shows at once).
//! * **The draft carry.** The NextN drafter starts from the last committed position's tap. A restore must bring it, or the drafts of the first round
//!   differ: replies cannot change (drafts only propose), the speculation counters do. GLM's drafter cache is row-invariant, so a cached run's counters
//!   equal the uncached run's (Qwen's differ through its non-row-invariant absorb): `same` compares them.
//!
//! Every invariant has a negative test: the mock breaks exactly that invariant (`PrefixFault`) and the scenario must then fail.

use gb_engine::mock::{Fault, MockBackend, PrefixFault};
use gb_engine::{CacheBudget, CutPolicy, Drafter, Engine, EngineConfig, Output, PrefixConfig, Request};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};

const VOCAB: usize = 512;
const SLOTS: usize = 8;
const TAIL: usize = 8;

fn admission(slots: usize) -> Admission {
    let mm = ModelMemory {
        weights_bytes: 1 << 30,
        kv_bytes_per_token: 12_000,
        state_bytes_per_slot: 141 << 20,
        activation_reserve: 1 << 30,
    };
    Admission::new(64 << 30, 6 << 30, mm, slots).unwrap()
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

/// Every cut a test can ask for is allowed: a GLM prompt of 100 tokens is as cacheable as one of 10,000.
fn policy() -> PrefixConfig {
    PrefixConfig {
        policy: CutPolicy { min_cut: 64, tail: TAIL, ghost_min_gain: 256, min_gap: 0 },
        ghost_blocks: 1 << 12,
        ..Default::default()
    }
}

fn budget(units: usize) -> CacheBudget {
    CacheBudget { snapshot_slots: units, page_blocks: units * 8, snapshot_bytes: 141 << 20, page_bytes: 3 << 20 }
}

fn glm_mock(slots: usize, slice: Option<usize>, cache: Option<(CacheBudget, PrefixFault)>) -> MockBackend {
    let b = MockBackend::new(VOCAB, slots, Fault::None).with_glm_shape();
    let b = match cache {
        Some((bud, fault)) => b.with_prefix_cache(bud, fault),
        None => b.with_paged_kv(slots * 64),
    };
    match slice {
        Some(s) => b.with_prefill_slice(s),
        None => b,
    }
}

fn cached(units: usize, fault: PrefixFault, slice: Option<usize>) -> Engine<MockBackend> {
    Engine::new(glm_mock(SLOTS, slice, Some((budget(units), fault))), admission(SLOTS), engine_config(), no_native_drafter())
        .with_prefix_cache(policy())
}

fn plain(slice: Option<usize>) -> Engine<MockBackend> {
    Engine::new(glm_mock(SLOTS, slice, None), admission(SLOTS), engine_config(), no_native_drafter())
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

/// Tokens, finish reason AND the speculation counters: GLM's drafter state is a function of the committed rows alone, so a hit changes none of them.
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

fn reference(r: &Request, slice: Option<usize>) -> Output {
    let mut e = plain(slice);
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

/// The end cut of a prompt of `total` tokens (what the policy plans for an otherwise cold prompt), or 0.
fn end_cut(total: usize) -> usize {
    let c = total.saturating_sub(TAIL);
    if c >= 64 {
        c
    } else {
        0
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------
// the contract: cached == uncached at every cut, every residue mod 4
// ------------------------------------------------------------------------------------------------------------------------------------

/// Prompt lengths whose end cut (`length - 8`) takes every residue mod 4 near the places that matter: the page boundary (256), the first pools of a
/// page, a few pages in, and a long prompt.
const LENGTHS: [usize; 26] = [
    100, 129, 130, 131, 132, 255, 256, 257, 258, 259, 300, 301, 302, 303, 511, 512, 513, 514, 700, 701, 702, 703, 1000, 1001,
    2100, 2103,
];

fn sweep(fault: PrefixFault, slice: Option<usize>) -> Result<usize, String> {
    let mut cases = 0;
    for l1 in LENGTHS {
        for ext in [1usize, 9, 40, 300] {
            let a_prompt = toks(l1 as u64, l1);
            let mut b_prompt = a_prompt.clone();
            b_prompt.extend(toks(1000 + ext as u64, ext));
            let (a, b) = (greedy(a_prompt, 24), request(b_prompt, 60, 5, 0.8));
            let mut e = cached(4, fault, slice);
            let oa = one(&mut e, a.clone());
            same(&oa, &reference(&a, slice)).map_err(|m| format!("A at {l1}: {m}"))?;
            assert_eq!(oa.cached_tokens, 0, "a cold server has nothing to hit");
            // an unrelated prompt takes the slot (its ring, its pages' pools, its carry) between A and B: whatever a restore does not bring back is
            // the stranger's, not A's own
            one(&mut e, greedy(toks(7000 + l1 as u64, 300), 4));
            let ob = one(&mut e, b.clone());
            same(&ob, &reference(&b, slice))
                .map_err(|m| format!("B at {l1}+{ext} (cut {}, residue {}): {m}", end_cut(l1), end_cut(l1) % 4))?;
            if ob.cached_tokens != end_cut(l1) {
                return Err(format!("B at {l1}+{ext} hit {} tokens, expected {}", ob.cached_tokens, end_cut(l1)));
            }
            no_leaks(&e).map_err(|m| format!("at {l1}+{ext}: {m}"))?;
            cases += 2;
        }
    }
    Ok(cases)
}

#[test]
fn glm_cached_replies_and_counters_equal_uncached_at_every_cut_and_residue() {
    for slice in [None, Some(64), Some(100)] {
        let n = sweep(PrefixFault::None, slice).unwrap_or_else(|m| panic!("slice {slice:?}: {m}"));
        assert_eq!(n, LENGTHS.len() * 4 * 2);
    }
}

#[test]
fn mutant_ring_tail_not_restored_is_caught() {
    // only a cut that is not a multiple of 4 has rows in the incomplete pool group: the sweep has all four residues and must fail on the other three
    let e = sweep(PrefixFault::RingTailNotRestored, None).unwrap_err();
    println!("ring tail not restored: {e}");
    assert!(e.contains("tokens differ") || e.contains("speculation counters differ"), "{e}");
    assert!(!e.contains("residue 0)"), "a cut at a multiple of 4 needs no ring rows, so it cannot be what fails: {e}");
}

#[test]
fn a_cut_at_a_multiple_of_four_needs_no_ring_rows() {
    // the same mutant on prompts whose end cut is a multiple of 4: nothing to restore, nothing to break (page-aligned adoption is the special case)
    for l1 in [132usize, 256, 300, 512, 700, 1000] {
        assert_eq!(end_cut(l1) % 4, 0);
        let a_prompt = toks(l1 as u64, l1);
        let mut b_prompt = a_prompt.clone();
        b_prompt.extend(toks(3, 40));
        let mut e = cached(4, PrefixFault::RingTailNotRestored, None);
        one(&mut e, greedy(a_prompt, 8));
        let b = request(b_prompt, 60, 5, 0.8);
        let ob = one(&mut e, b.clone());
        assert_eq!(ob.cached_tokens, end_cut(l1));
        same(&ob, &reference(&b, None)).unwrap_or_else(|m| panic!("{l1}: {m}"));
    }
}

#[test]
fn mutant_pools_not_copied_with_the_partial_page_is_caught() {
    let e = sweep(PrefixFault::PoolsNotCopied, None).unwrap_err();
    println!("pools not copied: {e}");
    assert!(e.contains("tokens differ") || e.contains("speculation counters differ"), "{e}");
}

#[test]
fn mutant_draft_carry_not_restored_is_caught_by_the_counters_alone() {
    // the carry steers the drafts and nothing else: every reply is still right, and the counters of the first round are not
    let mut tokens_differ = false;
    let mut counters_differ = false;
    for l1 in [300usize, 700, 1000, 2100] {
        let a_prompt = toks(l1 as u64, l1);
        let mut b_prompt = a_prompt.clone();
        b_prompt.extend(toks(5, 60));
        let mut e = cached(4, PrefixFault::CarryNotRestored, None);
        one(&mut e, greedy(a_prompt, 8));
        // another prompt takes the slot (and leaves its carry in it) between A and B
        one(&mut e, greedy(toks(900 + l1 as u64, l1 + 100), 8));
        let b = greedy(b_prompt, 120);
        let ob = one(&mut e, b.clone());
        assert!(ob.cached_tokens > 0);
        match same(&ob, &reference(&b, None)) {
            Ok(()) => {}
            Err(m) if m.contains("tokens differ") => tokens_differ = true,
            Err(_) => counters_differ = true,
        }
    }
    assert!(!tokens_differ, "a drafter's carry can never change a reply");
    assert!(counters_differ, "a stale carry must show in the speculation counters of some prompt");
}

#[test]
fn mutant_recurrent_state_not_restored_is_caught() {
    let e = sweep(PrefixFault::StateNotRestored, None).unwrap_err();
    println!("state not restored: {e}");
    assert!(e.contains("tokens differ") || e.contains("speculation counters differ"), "{e}");
}

#[test]
fn mutant_no_copy_on_write_is_caught() {
    // a donor decoding while a sharer prefills over the partial last page: sharing the page clobbers the donor's tokens and its pools
    let scenario = |fault: PrefixFault| -> Result<(), String> {
        let a_prompt = toks(11, 700);
        let mut b_prompt = a_prompt[..end_cut(700)].to_vec();
        b_prompt.extend(toks(12, 120));
        let (a, b) = (greedy(a_prompt, 90), request(b_prompt, 80, 9, 0.7));
        let mut e = cached(4, fault, Some(100));
        let ida = e.submit(a.clone()).unwrap();
        for _ in 0..3 {
            e.step().map_err(|x| x.to_string())?;
        }
        let idb = e.submit(b.clone()).unwrap();
        e.run().map_err(|x| x.to_string())?;
        let outs = e.take_outputs();
        let get = |id| outs.iter().find(|o| o.id == id).unwrap();
        // beside another stream the counters depend on the rows the budget hands each: replies only
        let tokens_only = |o: &Output, r: &Request| -> Result<(), String> {
            let want = reference(r, Some(100));
            if o.tokens != want.tokens {
                return Err("tokens differ".into());
            }
            Ok(())
        };
        tokens_only(get(ida), &a).map_err(|m| format!("the donor: {m}"))?;
        tokens_only(get(idb), &b).map_err(|m| format!("the sharer: {m}"))?;
        no_leaks(&e)
    };
    scenario(PrefixFault::None).unwrap();
    let e = scenario(PrefixFault::NoCopyOnWrite).unwrap_err();
    println!("no copy on write: {e}");
    assert!(e.contains("the donor") || e.contains("the sharer"), "{e}");
}

// ------------------------------------------------------------------------------------------------------------------------------------
// no path rule
// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn glm_prompts_of_128_tokens_or_fewer_are_cached_like_any_other() {
    // the Qwen engine never caches a prompt of 128 tokens or fewer (its decode GEMV leaves other bits than the GEMM) and never resumes a longer one from
    // such a state. GLM has one arithmetic, so a 100-token prompt leaves an entry a 140-token one resumes from, and a 128-token prompt one a 329-token one does
    for (l1, ext) in [(100usize, 40usize), (128, 201), (129, 5), (127, 1)] {
        let a_prompt = toks(l1 as u64 + 5, l1);
        let mut b_prompt = a_prompt.clone();
        b_prompt.extend(toks(78, ext));
        let mut e = cached(4, PrefixFault::None, None);
        let oa = one(&mut e, greedy(a_prompt.clone(), 30));
        same(&oa, &reference(&greedy(a_prompt, 30), None)).unwrap();
        let b = request(b_prompt, 60, 3, 1.0);
        let ob = one(&mut e, b.clone());
        assert_eq!(ob.cached_tokens, end_cut(l1), "{l1}+{ext}");
        assert!(ob.cached_tokens > 0, "{l1}+{ext}: the prefix must be found");
        same(&ob, &reference(&b, None)).unwrap_or_else(|m| panic!("{l1}+{ext}: {m}"));
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------
// slot reuse, concurrency, opt-out
// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn slot_reuse_never_shows_the_previous_owners_ring_pages_or_carry() {
    // one slot: every request lands in it after another's, with the previous owner's raw rows, pools and carry still in it
    let mut e = Engine::new(
        MockBackend::new(VOCAB, 1, Fault::None).with_glm_shape().with_prefix_cache(budget(4), PrefixFault::None),
        admission(1),
        engine_config(),
        no_native_drafter(),
    )
    .with_prefix_cache(policy());
    let base = toks(61, 803);
    let mut prompts = vec![base.clone()];
    let mut ext = base.clone();
    ext.extend(toks(62, 91));
    prompts.push(ext);
    prompts.push(toks(63, 501)); // unrelated, takes the slot between
    let mut again = base.clone();
    again.extend(toks(64, 62));
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
fn concurrent_sharers_each_equal_their_uncached_reply() {
    for slice in [None, Some(64)] {
        let sys = toks(41, 903);
        let mut e = cached(6, PrefixFault::None, slice);
        for q in 0..2u64 {
            let mut p = sys.clone();
            p.extend(toks(50 + q, 20));
            one(&mut e, greedy(p, 12));
        }
        let mut ids = Vec::new();
        let mut reqs = Vec::new();
        for q in 0..5u64 {
            let mut p = sys.clone();
            p.extend(toks(60 + q, 17 + q as usize));
            let r = request(p, 50, 100 + q, 0.7);
            ids.push(e.submit(r.clone()).unwrap());
            reqs.push(r);
        }
        e.run().unwrap();
        let outs = e.take_outputs();
        for (id, r) in ids.iter().zip(&reqs) {
            let o = outs.iter().find(|o| o.id == *id).unwrap();
            assert!(o.cached_tokens >= 768, "a sharer of the system prompt resumes from the shared blocks ({})", o.cached_tokens);
            // beside others the counters depend on the rows the budget hands each stream: the reply is what must match
            assert_eq!(o.tokens, reference(r, slice).tokens, "sharer of {}", r.prompt.len());
        }
        no_leaks(&e).unwrap();
    }
}

#[test]
fn a_request_that_opts_out_is_untouched() {
    let mut e = cached(4, PrefixFault::None, None);
    let r = Request { cache: false, ..greedy(toks(81, 600), 12) };
    let o = one(&mut e, r.clone());
    let s = e.prefix_stats();
    assert_eq!((o.cached_tokens, s.lookups, s.entries), (0, 0, 0));
    same(&o, &reference(&r, None)).unwrap();
    one(&mut e, greedy(toks(81, 600), 12));
    assert_eq!(e.prefix_stats().entries, 1);
    let o = one(&mut e, r);
    assert_eq!(o.cached_tokens, 0, "an opted-out request does not hit what is stored either");
}

#[test]
fn a_differing_prefix_hits_nothing_and_changes_nothing() {
    let mut e = cached(4, PrefixFault::None, None);
    one(&mut e, greedy(toks(91, 700), 12));
    // the same length, one token different early on: no entry is a prefix of it
    let mut other = toks(91, 700);
    other[5] = other[5] % (VOCAB as u32 - 1) + 1;
    let r = greedy(other, 30);
    let o = one(&mut e, r.clone());
    assert_eq!(o.cached_tokens, 0);
    same(&o, &reference(&r, None)).unwrap();
}
