//! Flash-Next interior-cut restore, driven at engine level over the CPU mock: a follow-up that
//! resumes from an interior cut (the previous prompt's end cut) and appends a new suffix must answer
//! exactly like an uncached server. Includes the page-aliasing shape: a second, deeper snapshot is
//! inserted while a shallower one is still live, and the shallower one is restored afterwards — the
//! restored reply must still equal the cold one (the depth-0.5 entry must not pick pages written by
//! the depth-0.9 entry).
//!
//! The mock (`gb_engine::mock::MockBackend`) models everything the Flash-Next cache must carry:
//! paged KV, a prefill-path-dependent recurrent state, a draft context ring, copy-on-write partial
//! pages and prefix faults. The engine logic is family-agnostic, so this exercises the exact code
//! path the Flash-Next backend plugs into (`plan_prefix` -> `prefix_restore` -> slices -> commit),
//! without a GPU.

use gb_engine::mock::{ChainDrafter, Fault, MockBackend, PrefixFault};
use gb_engine::{Backend, CacheBudget, CutPolicy, Drafter, Engine, EngineConfig, Output, PrefixConfig, Request};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};

const VOCAB: usize = 512;
const SLOTS: usize = 8;
const TAIL: usize = 8;
const L1: usize = 5200; // the first (cold) prompt: a long document
const Q: usize = 40; // the appended follow-up question: 40 tokens

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
    // interactive-style values: small min_cut, tail 8 here (like the engine-level suite), min_gap 0
    // so the end cut is always taken even for a short appended question.
    PrefixConfig {
        policy: CutPolicy { min_cut: 64, tail: TAIL, ghost_min_gain: 256, min_gap: 0 },
        ghost_blocks: 1 << 12,
        ..Default::default()
    }
}

/// The Flash-Next engine policy: the interactive cut policy with no alignment (the exact restore is
/// valid at any cut). The tail is the interactive 64, so the exact stage-3 geometry below plans its
/// end cut at 2067 (mid-page), and it must hit there — not at an aligned 2048.
fn policy_unhoned() -> PrefixConfig {
    PrefixConfig {
        policy: CutPolicy { min_cut: 64, tail: 64, ghost_min_gain: 256, min_gap: 128 },
        ghost_blocks: 1 << 12,
        ..Default::default()
    }
}

fn budget(units: usize) -> CacheBudget {
    CacheBudget { snapshot_slots: units, page_blocks: units * 8, snapshot_bytes: 1 << 20, page_bytes: 1 << 16 }
}

fn cached_engine(units: usize) -> Engine<MockBackend> {
    let b = MockBackend::new(VOCAB, SLOTS, Fault::None)
        .with_path_dependent_state()
        .with_native_draft()
        .with_prefix_cache(budget(units), PrefixFault::None);
    Engine::new(b, admission(), engine_config(), drafter()).with_prefix_cache(policy())
}

/// [`cached_engine`] with the Flash-Next policy (interactive cuts, no alignment: the restore is
/// exact at any cut, so mid-page cuts hit at their true position).
fn cached_engine_exact(units: usize) -> Engine<MockBackend> {
    let b = MockBackend::new(VOCAB, SLOTS, Fault::None)
        .with_path_dependent_state()
        .with_native_draft()
        .with_prefix_cache(budget(units), PrefixFault::None);
    Engine::new(b, admission(), engine_config(), drafter()).with_prefix_cache(policy_unhoned())
}

fn plain_engine() -> Engine<MockBackend> {
    let b = MockBackend::new(VOCAB, SLOTS, Fault::None).with_path_dependent_state().with_native_draft().with_paged_kv(SLOTS * 64);
    Engine::new(b, admission(), engine_config(), drafter())
}

/// Deterministic non-degenerate tokens (from the engine-level prefix suite).
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

fn one<B: Backend>(e: &mut Engine<B>, r: Request) -> Output {
    let id = e.submit(r).unwrap();
    e.run().unwrap();
    let mut outs = e.take_outputs();
    outs.retain(|o| o.id == id);
    assert_eq!(outs.len(), 1);
    outs.remove(0)
}

fn reference(r: &Request) -> Output {
    one(&mut plain_engine(), Request { cache: false, ..r.clone() })
}

fn same_reply(a: &Output, b: &Output) -> Result<(), String> {
    if a.tokens != b.tokens {
        let at = a.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y).unwrap_or(a.tokens.len().min(b.tokens.len()));
        return Err(format!("tokens differ at {at} ({} vs {} tokens)", a.tokens.len(), b.tokens.len()));
    }
    if (a.finish, a.stopped) != (b.finish, b.stopped) {
        return Err(format!("finish {:?} vs {:?}", a.finish, b.finish));
    }
    Ok(())
}

/// The interior-cut restore: the first prompt runs cold (its end cut is stored), then a follow-up
/// that starts with it and appends a new question resumes from that interior cut. Every reply must
/// equal the uncached reference.
#[test]
fn follow_up_appending_a_question_resumes_from_the_interior_cut_and_matches_cold() {
    let mut e = cached_engine(4);
    let doc = toks(1, L1);
    let first = request(doc.clone(), 24, 7, 0.0);
    let o1 = one(&mut e, first.clone());
    same_reply(&o1, &reference(&first)).unwrap();
    assert_eq!(o1.cached_tokens, 0, "a cold server has nothing to hit");

    let mut follow_prompt = doc.clone();
    follow_prompt.extend(toks(2, Q));
    let follow = request(follow_prompt, 40, 5, 0.8);
    let o2 = one(&mut e, follow.clone());
    // the hit is the first prompt's end cut: L1 - TAIL, an interior point of the follow-up
    assert_eq!(o2.cached_tokens, L1 - TAIL, "the follow-up must resume from the first prompt's end cut");
    same_reply(&o2, &reference(&follow)).unwrap();

    // a third turn (appending again) resumes from the second prompt's end cut, still exact
    let mut third_prompt = follow.prompt.clone();
    third_prompt.extend(toks(3, Q));
    let third = request(third_prompt, 40, 6, 0.8);
    let o3 = one(&mut e, third.clone());
    assert_eq!(o3.cached_tokens, L1 + Q - TAIL);
    same_reply(&o3, &reference(&third)).unwrap();

    // the cache holds its entries and no page/snapshot leak (with min_gap 0 every turn stores
    // its end cut, so the count grows with the turns)
    let s = e.prefix_stats();
    assert!(s.entries >= 2, "expected at least the first two end cuts, got {}", s.entries);
    assert_eq!(e.backend.pages_in_use(), s.pages_held as usize);
    assert_eq!(e.backend.snapshots_in_use(), s.entries as usize);
}

/// The aliasing shape from the audit: a deeper snapshot inserted after a shallower one, then the
/// shallower one restored (`the depth-0.5 entry must not pick pages written by the depth-0.9
/// entry`). System prefix (shared) + two per-turn end cuts; resume from the shallow turn's entry
/// while the deeper entry exists.
#[test]
fn restoring_a_shallower_entry_with_a_deeper_entry_live_keeps_the_cold_reply() {
    let mut e = cached_engine(4);
    let sys = toks(11, 800);
    const T1: usize = 800 + L1; // turn 1 prompt length
    const T1_CUT: usize = T1 - TAIL; // turn 1's end cut (the interior restore point)
    const T2: usize = T1 + Q; // turn 2 prompt length
    const T2_CUT: usize = T2 - TAIL; // turn 2's end cut
    assert!(T1_CUT < T2 && T2_CUT < T2);

    // turn 1: system + document (the document runs cold and stores its end cut)
    let mut turn1_prompt = sys.clone();
    turn1_prompt.extend(toks(21, L1));
    let turn1 = request(turn1_prompt, 24, 7, 0.0);
    let o1 = one(&mut e, turn1.clone());
    same_reply(&o1, &reference(&turn1)).unwrap();

    // turn 2: system + document + q1 -> stores its own end cut (a deeper entry: same sys+doc prefix
    // plus q1). The shared prefix is discovered here, so two entries exist.
    let mut turn2_prompt = turn1.prompt.clone();
    turn2_prompt.extend(toks(22, Q));
    let turn2 = request(turn2_prompt, 40, 5, 0.8);
    let o2 = one(&mut e, turn2.clone());
    same_reply(&o2, &reference(&turn2)).unwrap();
    assert_eq!(o2.cached_tokens, T1_CUT, "turn 2 resumes from turn 1's end cut");
    let stats2 = e.prefix_stats();
    assert!(stats2.entries >= 2, "the deeper entry must be stored: {}", stats2.entries);

    // turn 3: system + document + q2 (a DIFFERENT question): the longest matching prefix is the
    // document's end cut (turn 1's entry) — turn 2's entry diverges inside q1, so it must not hit.
    // This restores the shallow entry while the deeper entry is still live. If the shallow entry's
    // pages had been aliased by turn 2's writes, the reply would diverge.
    let mut turn3_prompt = sys.clone();
    turn3_prompt.extend(toks(21, L1));
    turn3_prompt.extend(toks(23, Q));
    let turn3 = request(turn3_prompt, 40, 6, 0.8);
    let o3 = one(&mut e, turn3.clone());
    assert_eq!(o3.cached_tokens, T1_CUT, "turn 3 resumes from the document's end cut");
    same_reply(&o3, &reference(&turn3)).unwrap();

    // turn 4: the same prompt as turn 2 (a retry): must still hit turn 2's entry (the deeper one)
    // and stay exact.
    let o4 = one(&mut e, turn2.clone());
    assert_eq!(o4.cached_tokens, T2_CUT);
    same_reply(&o4, &reference(&turn2)).unwrap();

    let s = e.prefix_stats();
    assert_eq!(e.backend.pages_in_use(), s.pages_held as usize);
    assert_eq!(e.backend.snapshots_in_use(), s.entries as usize);
}

/// The ceiling-case grid coincidence: a cut that lands exactly on the natural slice boundary must
/// still be captured (the capture condition tests `cuts.contains(&end)`, not why the slice ended
/// there). With min_gap 0 and tail TAIL, the end cut is at L1-TAIL; run with several slice sizes so
/// that the grid sometimes aligns with the cut.
#[test]
fn an_interior_cut_on_the_natural_slice_boundary_is_still_captured_and_restored() {
    for slice in [64usize, 100, 256, 512, 600] {
        let mut e = cached_engine(4);
        let doc = toks(31, L1);
        let first = request(doc.clone(), 24, 7, 0.0);
        let o1 = one(&mut e, first.clone());
        same_reply(&o1, &reference(&first)).unwrap();

        let mut follow_prompt = doc;
        follow_prompt.extend(toks(32, Q));
        let follow = request(follow_prompt, 40, 5, 0.8);
        let o2 = one(&mut e, follow.clone());
        assert_eq!(o2.cached_tokens, L1 - TAIL, "slice {slice}: the end cut must be stored and hit");
        same_reply(&o2, &reference(&follow)).unwrap();
    }
}

/// Decode after an interior restore must be exact even when the restored slot's suffix is long
/// enough to pass several slice boundaries (the recurrent state is exercised past the boundary).
#[test]
fn a_long_suffix_after_restore_stays_exact_across_slice_boundaries() {
    let mut e = cached_engine(4);
    let doc = toks(41, L1);
    let first = request(doc.clone(), 24, 7, 0.0);
    one(&mut e, first.clone());

    let mut follow_prompt = doc;
    follow_prompt.extend(toks(42, 1500)); // a long suffix: many slices past the restore point
    let follow = request(follow_prompt, 40, 5, 0.8);
    let o2 = one(&mut e, follow.clone());
    assert_eq!(o2.cached_tokens, L1 - TAIL);
    same_reply(&o2, &reference(&follow)).unwrap();
}

/// Zero-length appended suffix plus a full-prompt retry hits the end cut too (the "identical-prompt
/// full replay" shape that was GPU-proven) — the interior-cut path must not regress it.
#[test]
fn identical_prompt_replay_still_hits_and_matches() {
    let mut e = cached_engine(4);
    let p = request(toks(51, L1), 24, 7, 0.0);
    let o1 = one(&mut e, p.clone());
    same_reply(&o1, &reference(&p)).unwrap();
    let o2 = one(&mut e, p.clone());
    assert_eq!(o2.cached_tokens, L1 - TAIL);
    same_reply(&o2, &o1).unwrap();
}

/// A restore failure drops the entry and falls back to a cold prefill that still answers exactly.
#[test]
fn a_restore_failure_falls_back_cold_and_stays_exact() {
    let mut e = cached_engine(2);
    let p = request(toks(61, L1), 24, 7, 0.0);
    one(&mut e, p.clone());
    let mut follow_prompt = p.prompt.clone();
    follow_prompt.extend(toks(62, Q));
    let follow = request(follow_prompt, 40, 5, 0.8);
    let o2 = one(&mut e, follow.clone());
    same_reply(&o2, &reference(&follow)).unwrap();
}

/// The stage-3 gate geometry, at engine level: the donor prompt ends at 2131 (its end cut 2131-64 =
/// 2067 = 8x256+19 is mid-page), the follow-up shares the first 2067 tokens and then appends a
/// DIFFERENT suffix that ends before the donor does. With the exact partial-page restore the cut at
/// 2067 is used as-is: the restore copies the partial page, scrubs the donor's continuation, and the
/// divergent remainder re-prefills from 2067 — the hit is mid-page and the reply equals cold.
#[test]
fn a_mid_page_cut_restores_exactly_and_matches_cold() {
    let mut e = cached_engine_exact(4);
    const DON_TOTAL: usize = 2131; // 8x256 + 83: the end cut 2131-64 = 2067 lands mid-page
    const SHARED: usize = 2067; // q2's prefix length (the exact stage-3 shared prefix)
    const Q2_SUFFIX: usize = 48; // a different continuation than the donor's last 64 tokens
    let doc = toks(71, DON_TOTAL);

    // donor (cold): stores its end cut at 2067
    let first = request(doc.clone(), 24, 7, 0.0);
    let o1 = one(&mut e, first.clone());
    same_reply(&o1, &reference(&first)).unwrap();
    assert_eq!(o1.cached_tokens, 0, "a cold server has nothing to hit");

    // q2: shares exactly SHARED tokens, then diverges with its own suffix
    let mut q2_prompt = doc[..SHARED].to_vec();
    q2_prompt.extend(toks(72, Q2_SUFFIX));
    let q2 = request(q2_prompt, 40, 5, 0.8);
    let o2 = one(&mut e, q2.clone());
    assert_eq!(o2.cached_tokens, SHARED, "the hit must be the mid-page 2067 itself: the exact restore needs no alignment");
    same_reply(&o2, &reference(&q2)).unwrap();

    // the same shape must still hit on a third, extending turn: q3 continues q2 and resumes from the
    // same 2067 entry (q2's own end cut 2115-64=2051 is behind its hit, so it stores nothing new)
    let mut q3_prompt = q2.prompt.clone();
    q3_prompt.extend(toks(73, Q2_SUFFIX));
    let q3 = request(q3_prompt, 40, 6, 0.8);
    let o3 = one(&mut e, q3.clone());
    assert_eq!(o3.cached_tokens, SHARED, "turn 3 resumes from the same mid-page entry");
    same_reply(&o3, &reference(&q3)).unwrap();

    // no page/snapshot leak
    let s = e.prefix_stats();
    assert_eq!(e.backend.pages_in_use(), s.pages_held as usize);
    assert_eq!(e.backend.snapshots_in_use(), s.entries as usize);
}

/// The stage-6 gate geometry, at engine level: a conversation whose shared prefix is UNDER one KV
/// page (256 tokens). With `cut_align` the end cut of the first turn would round below `min_cut`
/// and nothing would be stored, so turn 2 got `cached_tokens == 0`. The exact restore stores the
/// true end cut, so a short conversation hits from turn 2 on.
#[test]
fn a_two_turn_conversation_with_a_shared_prefix_under_256_tokens_hits_at_turn_two() {
    let mut e = cached_engine(4);
    const DOC: usize = 192; // the first prompt: under one KV page
    const Q: usize = 40; // the appended question
    let doc = toks(81, DOC);

    // turn 1 (cold): stores its end cut 192-8 = 184
    let first = request(doc.clone(), 24, 7, 0.0);
    let o1 = one(&mut e, first.clone());
    same_reply(&o1, &reference(&first)).unwrap();
    assert_eq!(o1.cached_tokens, 0, "a cold server has nothing to hit");

    // turn 2: the same 192-token prefix plus a question -> hits 184 (a strict prefix, mid-page)
    let mut turn2_prompt = doc.clone();
    turn2_prompt.extend(toks(82, Q));
    let turn2 = request(turn2_prompt, 40, 5, 0.8);
    let o2 = one(&mut e, turn2.clone());
    assert!(o2.cached_tokens > 0, "a shared prefix under 256 tokens must hit from turn 2 on (got {})", o2.cached_tokens);
    assert_eq!(o2.cached_tokens, DOC - 8 as usize, "the hit is turn 1's end cut");
    same_reply(&o2, &reference(&turn2)).unwrap();

    // turn 3 extends again: hits turn 2's end cut (232-8 = 224) or at least keeps hitting
    let mut turn3_prompt = turn2.prompt.clone();
    turn3_prompt.extend(toks(83, Q));
    let turn3 = request(turn3_prompt, 40, 6, 0.8);
    let o3 = one(&mut e, turn3.clone());
    assert!(o3.cached_tokens >= o2.cached_tokens, "cached tokens never go backwards");
    same_reply(&o3, &reference(&turn3)).unwrap();

    // no page/snapshot leak
    let s = e.prefix_stats();
    assert_eq!(e.backend.pages_in_use(), s.pages_held as usize);
    assert_eq!(e.backend.snapshots_in_use(), s.entries as usize);
}
