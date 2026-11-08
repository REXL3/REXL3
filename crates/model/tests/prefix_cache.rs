//! The prefix cache on the REAL model and the real engine (`serve::build_engine`, the engine as the server runs it): a request that resumes
//! from a cached prefix must be byte-identical to the same request without the cache, and the state a restore leaves behind must equal,
//! bit for bit, the state a cold prefill leaves (`docs/PREFIX_CACHE.md`).
//!
//! Tests (all `#[ignore]`: they need `REXL3_QWEN_DIR`, `REXL3_DRAFT_DIR` and a GPU; the gate runs them; with `REXL3_REQUIRE_GPU=1` a missing
//! requirement is a failure, otherwise they print `not run`):
//! * `snapshot_and_restore_leave_the_state_a_cold_prefill_leaves`: prompts of 129 to 9,000 tokens with snapshots at chosen cuts (inside a chunk,
//!   at a chunk boundary, at the very end, a 1-row part); restore into another slot and prefill the rest in another chunk grid: KV, `S` and conv
//!   hashes at the end and the last row's candidates equal the cold prefill's, the draft window equals the saved one, nothing leaks.
//! * `cached_replies_equal_uncached_replies`: prompts around the 128-token boundary and up to 9,000 tokens, each followed by an extension of it:
//!   the extension's reply with a hit equals its reply without the cache, drafted equals serial, and nothing crosses the boundary.
//! * `a_conversation_and_a_shared_system_prompt`: six-turn conversations that share a system prompt, replies equal turn by turn.
//! * `six_concurrent_sharers`: six requests sharing a 3.4k-token prompt, sampled with their own seeds, run together.
//! * `a_resume_equals_a_fresh_prefill_at_other_chunk_grids`: the rest of the prompt in chunks of 1,024 and of 3 tokens, cuts inside a chunk.
//! * `a_sharer_prefilling_over_the_partial_page_leaves_the_donor_alone`: copy-on-write of the partial last page, a donor decoding beside it.
//! * `a_prompt_of_128_tokens_or_fewer_never_enters_the_cache_on_the_real_model_either`.
//! * `eviction_cancel_and_the_books`: a two-entry cache under pressure, a cancelled prefill, every page and slot accounted for.
//! * `a_snapshot_settles_what_a_window_owes_and_a_restore_leaves_nothing_owed`: the B2 fold.
//! * `an_image_prompt_resumes_from_the_cache_and_answers_the_same`: image-keyed entries, a cut that moves in front of an image, M-RoPE.
//! * `the_snapshot_pool_is_what_the_plan_says`.
//! * `the_burst_policy_with_the_cache_equals_the_uncached_and_the_serial_replies`: 4, 8 and 16 simultaneous requests sharing a prefix (some of
//!   128 tokens or fewer, some opted out) under the shipped burst policy with batched short prefill: each reply equals its uncached solo reply
//!   and its reply under the serial policy, the sharers hit, the books balance.
//! * with draft trees (`--spec-tree`, docs/TREE_DRAFTING.md): `a_snapshot_settles_what_a_tree_window_owes_and_a_restore_leaves_nothing_owed`
//!   (the owed rows are an accepted PATH) and `cached_replies_equal_uncached_replies_with_trees` (a hit followed by tree rounds).
//!
//!     REXL3_QWEN_DIR=... REXL3_DRAFT_DIR=... cargo test --release -p gb-model --test prefix_cache -- --ignored --nocapture --test-threads=1

use gb_engine::{Backend, Engine, ImageSpan, Output, PrefillPolicy, PrefixHandle, Request, RowOutput, RowRequest, TreeRequest};
use gb_model::forward::Model;
use gb_model::rows::RowBatch;
use gb_model::serve::{build_engine, PrefixCacheOption, QwenEngine, ServeOptions, VisionSlot};
use gb_sampling::Sampling;

const CAND: usize = 20;

fn require_gpu() -> bool {
    std::env::var("REXL3_REQUIRE_GPU").is_ok_and(|v| v == "1")
}

/// The packs, or `None` after saying `not run` (a failure when `REXL3_REQUIRE_GPU=1`: the gate cannot pass vacuously).
fn packs() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let q = std::env::var("REXL3_QWEN_DIR").ok().map(std::path::PathBuf::from);
    let d = std::env::var("REXL3_DRAFT_DIR").ok().map(std::path::PathBuf::from);
    match (q, d) {
        (Some(q), Some(d)) if gb_cuda::device_available() => Some((q, d)),
        _ => {
            assert!(!require_gpu(), "REXL3_REQUIRE_GPU=1 but this test needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR and a GPU");
            eprintln!("not run: needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR and a GPU");
            None
        }
    }
}

/// The engine as the server builds it: the shipped planner, namespace and cut policy (small `min_cut` so that prompts of a few hundred tokens
/// are cut), a text-only server with `slots` slots, 48k tokens of KV and a prefill chunk of `chunk` rows.
fn engine(cache: PrefixCacheOption, slots: usize, chunk: usize) -> Option<QwenEngine> {
    engine_with(cache, slots, chunk, false)
}

/// [`engine`] with draft trees (`--spec-tree`) on or off.
fn engine_with(cache: PrefixCacheOption, slots: usize, chunk: usize, trees: bool) -> Option<QwenEngine> {
    let (q, d) = packs()?;
    std::env::set_var("REXL3_PREFIX_MIN_CUT", "64");
    std::env::set_var("REXL3_PREFIX_TAIL", "16");
    std::env::set_var("REXL3_PREFIX_MIN_GAP", "0"); // every cut is wanted here (the gap rule has its own tests)
    let mut o = ServeOptions::new(q);
    o.draft_dir = Some(d);
    o.slots = slots;
    o.max_model_len = 12288;
    o.kv_tokens = Some(49152);
    o.prefill_chunk = chunk;
    o.vision = None;
    o.prefix_cache = cache;
    o.spec_tree = trees;
    Some(build_engine(&o).unwrap_or_else(|e| panic!("cannot build the engine: {e}")))
}

/// [`engine`] with the vision tower loaded: the slot the encoder service is published in.
fn engine_with_vision(cache: PrefixCacheOption, slots: usize, chunk: usize) -> Option<(QwenEngine, VisionSlot)> {
    let (q, d) = packs()?;
    std::env::set_var("REXL3_PREFIX_MIN_CUT", "64");
    std::env::set_var("REXL3_PREFIX_TAIL", "16");
    std::env::set_var("REXL3_PREFIX_MIN_GAP", "0");
    let mut o = ServeOptions::new(q);
    o.draft_dir = Some(d);
    o.slots = slots;
    o.max_model_len = 12288;
    o.kv_tokens = Some(49152);
    o.prefill_chunk = chunk;
    let slot: VisionSlot = Default::default();
    o.vision = Some(slot.clone());
    o.prefix_cache = cache;
    Some((build_engine(&o).unwrap_or_else(|e| panic!("cannot build the engine: {e}")), slot))
}

fn random_prompt(len: usize, seed: u64) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1000 + (x % 99_000) as u32
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

fn one(e: &mut QwenEngine, r: Request) -> Output {
    let id = e.submit(r).unwrap();
    e.run().unwrap();
    let mut outs = e.take_outputs();
    outs.retain(|o| o.id == id);
    assert_eq!(outs.len(), 1);
    outs.remove(0)
}

/// Prefill all of `prompt` into `slot` from `done`, with `cuts` (those above `done`), as the engine would, slice by slice. Returns the last
/// row and the snapshots taken.
fn prefill_all(
    b: &mut gb_model::backend::QwenBackend,
    slot: usize,
    prompt: &[u32],
    mut done: usize,
    cuts: &[usize],
) -> (RowOutput, Vec<(usize, PrefixHandle)>) {
    let mut taken = Vec::new();
    loop {
        let cut = b.prefill_slice_cuts(slot, prompt, &[], done, CAND, cuts);
        assert!(b.take_error().is_none(), "the backend reported an error");
        let step = cut.step;
        done = step.done;
        taken.extend(cut.captured.into_iter().map(|c| (c.len, c.handle)));
        if let Some(row) = step.row {
            assert_eq!(done, prompt.len());
            return (row, taken);
        }
    }
}

/// `REXL3_PREFIX_PROMPTS=257,4097` restricts the bit-identity cases to those prompt lengths (a quick look, a mutation run).
fn wanted(total: usize) -> bool {
    std::env::var("REXL3_PREFIX_PROMPTS").map_or(true, |v| v.split(',').any(|t| t.trim().parse() == Ok(total)))
}

// ------------------------------------------------------------------------------------------------------------------------------------

/// Prompts and the cuts snapshotted in each: the first rows of a chunk, a 1-row part, inside a chunk, the page boundaries, a chunk boundary,
/// the end of the prompt.
const CASES: &[(usize, &[usize])] = &[
    (129, &[1, 64, 100, 128]),
    (192, &[64, 127, 160]),
    (256, &[1, 128, 255, 256]),
    (257, &[200, 256]),
    (4096, &[1000, 2048, 4093, 4096]),
    (4097, &[4000, 4094, 4096]),
    (9000, &[3000, 4096, 5000, 8000]),
];

#[test]
#[ignore]
fn snapshot_and_restore_leave_the_state_a_cold_prefill_leaves() {
    let Some(mut e) = engine(PrefixCacheOption::Mib(6000), 4, 4096) else { return };
    let pages_free = e.backend.m.kv_tab.free_blocks();
    let snaps_total = e.backend.m.snaps.as_ref().unwrap().slots;
    let mut cases = 0;
    for &(p, cuts) in CASES.iter().filter(|(p, _)| wanted(*p)) {
        let prompt = random_prompt(p, p as u64);
        // the cold prefill: the reference, no cuts
        let (ref_row, none) = prefill_all(&mut e.backend, 0, &prompt, 0, &[]);
        assert!(none.is_empty());
        let ref_hash = e.backend.m.state_hashes(0, p).unwrap();
        // the donor: the same prompt with every cut taken on the way. Its end state must equal the reference's (a cut changes no bit) ...
        let (row, taken) = prefill_all(&mut e.backend, 1, &prompt, 0, cuts);
        assert_eq!(taken.len(), cuts.len(), "p{p}: every cut was taken");
        assert!(row.bits_eq(&ref_row), "p{p}: the last row's candidates changed with cuts {cuts:?}");
        assert_eq!(e.backend.m.state_hashes(1, p).unwrap(), ref_hash, "p{p}: the donor's state changed with cuts {cuts:?}");
        // ... and so must the end state of a restore from each cut into another slot, the rest prefilled in another chunk grid
        for (len, h) in &taken {
            e.backend.prefix_restore(2, h, *len).unwrap_or_else(|m| panic!("p{p} cut {len}: {m}"));
            let meta = e.backend.m.snaps.as_ref().unwrap().meta(h.snap).unwrap();
            assert_eq!((meta.len, meta.lo), (*len, len.saturating_sub(2048)), "p{p} cut {len}");
            // the draft window came back as it was saved
            let ring = e.backend.m.draft.as_ref().unwrap().ring_window_hash(2, meta.lo, meta.n).unwrap();
            let saved = e.backend.m.snaps.as_ref().unwrap().window_hash(h.snap, meta.n).unwrap();
            assert_eq!(ring, saved, "p{p} cut {len}: the restored draft window is not the saved one");
            // the recurrent state and the conv history are the snapshot's
            let (ss, sc) = e.backend.m.snaps.as_ref().unwrap().gdn_hashes(h.snap).unwrap();
            let now = e.backend.m.state_hashes(2, *len).unwrap();
            assert_eq!((now.s, now.conv), (ss, sc), "p{p} cut {len}: the restored state is not the snapshot's");
            if *len < p {
                // (a cut at the very end of the prompt leaves nothing to prefill: the engine never resumes from one, but the state is checked)
                let (row, again) = prefill_all(&mut e.backend, 2, &prompt, *len, &[]);
                assert!(again.is_empty());
                assert!(row.bits_eq(&ref_row), "p{p} cut {len}: the last row after a restore differs from the cold prefill's");
            }
            assert_eq!(
                e.backend.m.state_hashes(2, p).unwrap(),
                ref_hash,
                "p{p} cut {len}: KV or state after a restore differs from the cold prefill's"
            );
            cases += 1;
        }
        for (_, h) in taken {
            e.backend.prefix_release(h);
        }
        for slot in 0..3 {
            e.backend.release(slot);
        }
        // nothing leaks: every page and every snapshot slot is back
        assert_eq!(e.backend.m.kv_tab.free_blocks(), pages_free, "p{p}: pages leaked");
        assert_eq!(e.backend.m.snaps.as_ref().unwrap().free_count(), snaps_total, "p{p}: snapshot slots leaked");
    }
    eprintln!("snapshot and restore: {cases} restores bit-identical to the cold prefill ({} prompts)", CASES.len());
}

// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
#[ignore]
fn cached_replies_equal_uncached_replies() {
    let Some(mut e) = engine(PrefixCacheOption::Mib(6000), 4, 4096) else { return };
    let mut cases = 0;
    for (p, exts) in [
        (128usize, &[1usize, 17, 300][..]),
        (129, &[1, 17, 300]),
        (130, &[1, 17]),
        (192, &[17, 300]),
        (256, &[1, 17]),
        (257, &[17, 300]),
        (4096, &[17]),
        (4097, &[1, 17]),
        (9000, &[17]),
    ] {
        for &ext in exts {
            // a prompt of its own per case: prompts that start alike would legitimately hit each other's entries
            let a = random_prompt(p, 7000 + 1000 * p as u64 + ext as u64);
            let mut b = a.clone();
            b.extend(random_prompt(ext, 8000 + ext as u64));
            let oa = one(&mut e, request(a.clone(), 24, 1, 0.0));
            assert_eq!(oa.cached_tokens, 0, "p{p}: a cold prompt resumes from nothing");
            // B without the cache, then B with it
            let rb = request(b.clone(), 48, 2, 0.7);
            let uncached = one(&mut e, Request { cache: false, ..rb.clone() });
            assert_eq!(uncached.cached_tokens, 0);
            let cached = one(&mut e, rb.clone());
            let expect = if p > 128 { p - 16 } else { 0 }; // min_cut 64 is below every cut here
            assert_eq!(
                cached.cached_tokens, expect,
                "p{p}+{ext}: tokens resumed from the cache (0 = the GEMV path, never cached, never reused across 128)"
            );
            assert_eq!(cached.tokens, uncached.tokens, "p{p}+{ext}: the cached reply differs from the uncached one");
            assert_eq!(cached.finish, uncached.finish);
            // drafted == serial with a hit: the serial path (no speculation) gives the same tokens
            let serial = one(&mut e, Request { draft: false, ..rb });
            assert_eq!(serial.tokens, cached.tokens, "p{p}+{ext}: serial differs from drafted");
            assert_eq!(serial.drafted, 0);
            cases += 1;
        }
    }
    let s = e.prefix_stats();
    eprintln!(
        "cached == uncached == serial in {cases} cases; {} hits of {} lookups, {} tokens saved",
        s.hits, s.lookups, s.tokens_saved
    );
    assert!(s.hits >= 12, "{s:?}");
}

// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
#[ignore]
fn a_conversation_and_a_shared_system_prompt() {
    // two slots, so slots are reused all along; three conversations share one system prompt
    let Some(mut e) = engine(PrefixCacheOption::Mib(6000), 2, 4096) else { return };
    let system = random_prompt(1500, 11);
    let gen = [11u32, 12, 13, 14, 15]; // the generation prompt a next turn renders differently
    let mut history: Vec<Vec<u32>> = (0..3).map(|_| system.clone()).collect();
    let mut last_total = [0usize; 3];
    let mut hits = 0;
    for turn in 0..5u64 {
        for c in 0..3usize {
            let mut prompt = history[c].clone();
            prompt.extend(random_prompt(120, 100 * c as u64 + turn));
            prompt.extend(gen);
            let r = request(prompt.clone(), 40, 10 * turn + c as u64, if c == 1 { 0.8 } else { 0.0 });
            let cached = one(&mut e, r.clone());
            let uncached = one(&mut e, Request { cache: false, ..r });
            assert_eq!(cached.tokens, uncached.tokens, "conversation {c}, turn {turn}");
            if turn > 0 {
                assert_eq!(
                    cached.cached_tokens,
                    last_total[c] - 16,
                    "conversation {c}, turn {turn} resumes from the previous prompt's end cut"
                );
            }
            hits += (cached.cached_tokens > 0) as usize;
            last_total[c] = prompt.len();
            history[c] = prompt[..prompt.len() - gen.len()].to_vec();
            history[c].extend(&cached.tokens);
        }
    }
    let s = e.prefix_stats();
    eprintln!(
        "conversations: {hits} of 15 turns hit; {} tokens saved; {} entries, {} MiB held",
        s.tokens_saved,
        s.entries,
        s.bytes_held >> 20
    );
    assert!(hits >= 12);
}

// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
#[ignore]
fn six_concurrent_sharers() {
    let Some(mut e) = engine(PrefixCacheOption::Mib(6000), 6, 4096) else { return };
    let system = random_prompt(3400, 21);
    // two requests with the system prompt leave the shared entry; six more arrive together
    for q in 0..2u64 {
        let mut p = system.clone();
        p.extend(random_prompt(30, 22 + q));
        one(&mut e, request(p, 8, q, 0.0));
    }
    let reqs: Vec<Request> = (0..6u64)
        .map(|i| {
            let mut p = system.clone();
            p.extend(random_prompt(25 + 5 * i as usize, 30 + i));
            request(p, 60 + 10 * i as usize, 50 + i, if i % 2 == 0 { 0.0 } else { 1.0 })
        })
        .collect();
    let ids: Vec<u64> = reqs.iter().map(|r| e.submit_deferred(r.clone()).unwrap()).collect();
    e.run().unwrap();
    let outs = e.take_outputs();
    let free_after = (e.backend.m.kv_tab.free_blocks(), e.backend.m.snaps.as_ref().unwrap().free_count());
    for (r, id) in reqs.iter().zip(&ids) {
        let o = outs.iter().find(|o| o.id == *id).unwrap();
        assert!(o.cached_tokens >= 3328, "every sharer hits the shared blocks, got {}", o.cached_tokens);
        let reference = one(&mut e, Request { cache: false, ..r.clone() });
        assert_eq!(o.tokens, reference.tokens, "sharer {id} differs from its uncached reply");
    }
    let s = e.prefix_stats();
    // the books: with nothing running, the pool and the snapshot slots hold exactly what the entries hold
    assert_eq!(free_after.1 as u64, s.snapshot_slots - s.entries);
    eprintln!("six sharers equal their uncached replies; {} entries, {} pages", s.entries, s.pages_held);
}

// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
#[ignore]
fn eviction_cancel_and_the_books() {
    // two entries at most (700 MiB is two units): five distinct prompts evict, and a cancelled prefill gives back what it had
    let Some(mut e) = engine(PrefixCacheOption::Mib(700), 3, 997) else { return };
    let caps = e.backend.prefix_caps().unwrap();
    assert_eq!(caps.budget.snapshot_slots, 2, "700 MiB is two entries");
    let total_pages = e.backend.m.kv_tab.free_blocks();
    let prompts: Vec<Vec<u32>> = (0..5).map(|i| random_prompt(1500, 40 + i)).collect();
    for round in 0..2 {
        for (i, p) in prompts.iter().enumerate() {
            let r = request(p.clone(), 16, i as u64, 0.0);
            let cached = one(&mut e, r.clone());
            let uncached = one(&mut e, Request { cache: false, ..r });
            assert_eq!(cached.tokens, uncached.tokens, "round {round}, prompt {i}");
            let s = e.prefix_stats();
            assert!(s.entries <= 2 && s.pages_held <= caps.budget.page_blocks as u64, "{s:?}");
        }
    }
    assert!(e.prefix_stats().evictions >= 5);
    // a prompt of 3,000 tokens in slices of 997 (997, 1994, 2991, 3000): its end cut (2984) is taken in the third slice, one short of complete
    let long = request(random_prompt(3000, 99), 8, 1, 0.0);
    let id = e.submit_deferred(long).unwrap();
    for _ in 0..3 {
        e.step().unwrap();
    }
    assert_eq!(e.prefilling(), 1);
    assert!(e.backend.m.snaps.as_ref().unwrap().free_count() < 2, "the cut was taken and is pending");
    assert!(e.cancel(id));
    e.take_outputs();
    let s = e.prefix_stats();
    assert_eq!(
        e.backend.m.snaps.as_ref().unwrap().free_count() as u64,
        s.snapshot_slots - s.entries,
        "the cancelled prefill gave its snapshot slot back"
    );
    assert_eq!(
        total_pages - e.backend.m.kv_tab.free_blocks(),
        s.pages_held as usize,
        "and its pages: the pool holds exactly what the entries hold"
    );
    eprintln!("eviction and cancel: {} evictions, {} entries, {} pages, books balanced", s.evictions, s.entries, s.pages_held);
}

// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
#[ignore]
fn a_sharer_prefilling_over_the_partial_page_leaves_the_donor_alone() {
    // A is decoding when B, which shares A's prompt up to its end cut (a partial page: 700 - 16 = 684 = 2 x 256 + 172), is prefilled over the
    // same prefix with other tokens and then both decode: without the copy-on-write of the partial page B's prefill would overwrite A's keys
    let Some(mut e) = engine(PrefixCacheOption::Mib(3000), 4, 4096) else { return };
    let a_prompt = random_prompt(700, 61);
    let mut b_prompt = a_prompt[..684].to_vec();
    b_prompt.extend(random_prompt(160, 62));
    let (a, b) = (request(a_prompt, 120, 1, 0.0), request(b_prompt, 90, 2, 0.0));
    let ida = e.submit(a.clone()).unwrap();
    for _ in 0..3 {
        e.step().unwrap();
    }
    let idb = e.submit(b.clone()).unwrap();
    e.run().unwrap();
    let outs = e.take_outputs();
    let get = |id| outs.iter().find(|o| o.id == id).unwrap().clone();
    assert_eq!(get(idb).cached_tokens, 684, "B resumed from A's end cut");
    let ref_a = one(&mut e, Request { cache: false, ..a });
    let ref_b = one(&mut e, Request { cache: false, ..b });
    assert_eq!(get(ida).tokens, ref_a.tokens, "the donor's reply changed while a sharer prefilled over its page");
    assert_eq!(get(idb).tokens, ref_b.tokens, "the sharer's reply differs from its uncached reply");
}

#[test]
#[ignore]
fn a_prompt_of_128_tokens_or_fewer_never_enters_the_cache_on_the_real_model_either() {
    let Some(mut e) = engine(PrefixCacheOption::Mib(3000), 4, 4096) else { return };
    for p in [20usize, 100, 128] {
        let r = request(random_prompt(p, 70 + p as u64), 16, 3, 0.0);
        let first = one(&mut e, r.clone());
        let again = one(&mut e, r);
        assert_eq!((first.cached_tokens, again.cached_tokens), (0, 0), "p{p}");
        assert_eq!(first.tokens, again.tokens);
    }
    let s = e.prefix_stats();
    assert_eq!((s.lookups, s.entries, s.inserts), (0, 0, 0), "{s:?}: the GEMV path neither looks up nor leaves anything");
    // and a longer prompt that starts with a 128-token one is a cold prompt of its own path
    let short = random_prompt(128, 90);
    let mut long = short.clone();
    long.extend(random_prompt(200, 91));
    one(&mut e, request(short, 8, 1, 0.0));
    let o = one(&mut e, request(long, 8, 1, 0.0));
    assert_eq!(o.cached_tokens, 0);
}

/// A request with one image: `before` random text tokens, the image's placeholders, `after` more text. The encoder output is shared by every
/// request made from the same `Preprocessed` (it is read-only), the key names its content.
fn image_request(
    svc: &gb_model::serve::VisionService,
    png: &str,
    before: usize,
    after: usize,
    seed: u64,
    max_new: usize,
) -> Request {
    use gb_vision::preprocessor::ImagePreprocessor;
    let pp = gb_vision::Preprocessing::qwen38();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision").join(png);
    let pre = gb_vision::preprocess(&gb_vision::decode::decode(&std::fs::read(path).unwrap()).unwrap(), &pp).unwrap();
    let enc = svc.encode(&pre).unwrap();
    let tokens = enc.tokens;
    let pp_id = pp.id().bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
    let key = gb_sched::image_content_hash(pre.pixel_hash, pre.size.0 as u32, pre.size.1 as u32, pp_id).max(1);
    let mut prompt = random_prompt(before, seed);
    prompt.extend(std::iter::repeat_n(248056u32, tokens)); // <|image_pad|>
    prompt.extend(random_prompt(after, seed + 1));
    let span = ImageSpan { start: before, tokens, merged: enc.merged, features: enc.features, key };
    Request { images: vec![span], ..request(prompt, max_new, seed, 0.0) }
}

#[test]
#[ignore]
fn an_image_prompt_resumes_from_the_cache_and_answers_the_same() {
    // a prompt with an image: text, 256 placeholders, text. The key names the image: the same image under the same text resumes, another
    // image does not, and a snapshot is never taken inside the image (it moves in front of it). M-RoPE positions after a resume are the
    // ones a cold prefill gives (they are a function of the prompt's spans), so the reply is the same. Prefill chunks of 512 put the image
    // across chunk boundaries.
    for chunk in [4096usize, 512] {
        let Some((mut e, slot)) = engine_with_vision(PrefixCacheOption::Mib(3000), 4, chunk) else { return };
        let svc = slot.get().cloned().expect("the vision tower is loaded");
        // (1) the image ends well before the end of the prompt: the end cut is in the text after it
        let a = image_request(&svc, "check_512x512.png", 300, 400, 5, 24);
        let total = a.prompt.len();
        let cold = one(&mut e, a.clone());
        let uncached = one(&mut e, Request { cache: false, ..a.clone() });
        assert_eq!(
            cold.tokens, uncached.tokens,
            "chunk {chunk}: a cold image prompt on the cache path differs from the uncached one"
        );
        let hit = one(&mut e, a.clone());
        assert_eq!(hit.cached_tokens, total - 16, "chunk {chunk}: the same image under the same text resumes from the end cut");
        assert_eq!(hit.tokens, uncached.tokens, "chunk {chunk}: a resumed image prompt differs from the uncached one");
        // the same image and a longer text after it: a hit as well, identical to its uncached twin
        let longer = image_request(&svc, "check_512x512.png", 300, 500, 5, 24);
        let (hit_longer, plain_longer) = (one(&mut e, longer.clone()), one(&mut e, Request { cache: false, ..longer }));
        assert!(hit_longer.cached_tokens >= 300 + 256, "chunk {chunk}: resumed {} tokens", hit_longer.cached_tokens);
        assert_eq!(hit_longer.tokens, plain_longer.tokens, "chunk {chunk}");
        // (2) another image under the same text and the same placeholders is another prefix
        let other = image_request(&svc, "demo_scene.png", 300, 400, 5, 24);
        let o = one(&mut e, other.clone());
        assert_eq!(o.cached_tokens, 0, "chunk {chunk}: another image must not resume from this one's state");
        assert_eq!(o.tokens, one(&mut e, Request { cache: false, ..other }).tokens);
        // (3) the image reaches the end of the prompt: the end cut would be inside it, so the snapshot is taken in front of the image
        let tail = image_request(&svc, "check_512x512.png", 600, 10, 6, 24);
        let (cold, hit) = (one(&mut e, tail.clone()), one(&mut e, tail.clone()));
        assert_eq!(cold.cached_tokens, 0);
        assert_eq!(hit.cached_tokens, 600, "chunk {chunk}: the cut moves in front of the image");
        assert_eq!(hit.tokens, one(&mut e, Request { cache: false, ..tail }).tokens, "chunk {chunk}");
        let s = e.prefix_stats();
        eprintln!("chunk {chunk}: image prompts: {} hits of {} lookups", s.hits, s.lookups);
    }
}

#[test]
#[ignore]
fn a_resume_equals_a_fresh_prefill_at_other_chunk_grids() {
    // TensorFold's check (docs/TENSORFOLD_062_REVIEW.md section 5): resume == fresh with the rest of the prompt prefilled in chunks of 1,024 and of
    // 3 tokens, and with a keep inside a chunk (the donor's 4,096-row chunk holds both cuts)
    let Some(mut e) = engine(PrefixCacheOption::Mib(3000), 4, 4096) else { return };
    let p = 1500;
    let prompt = random_prompt(p, 4242);
    let (ref_row, _) = prefill_all(&mut e.backend, 0, &prompt, 0, &[]);
    let ref_hash = e.backend.m.state_hashes(0, p).unwrap();
    let (_, taken) = prefill_all(&mut e.backend, 1, &prompt, 0, &[400, 1440]);
    assert_eq!(taken.len(), 2);
    let mut runs = 0;
    for (len, h) in &taken {
        for step in [1024usize, 3] {
            if step == 3 && p - len > 100 {
                continue; // 3-token chunks are one W_q reconstruct pass each: only over the 60 tokens behind the second cut
            }
            e.backend.prefix_restore(2, h, *len).unwrap();
            let (mut at, mut last) = (*len, None);
            while at < p {
                let end = (at + step).min(p);
                last = e.backend.m.prefill_chunk(2, at, &prompt[at..end], p, (end == p).then_some(CAND), 0).unwrap();
                at = end;
            }
            let c = last.expect("the last chunk returns the candidates");
            let ids: Vec<u32> = c.ids[..CAND].iter().map(|&x| x as u32).collect();
            let row = RowOutput { ids, values: c.vals[..CAND].to_vec() };
            assert!(row.bits_eq(&ref_row), "cut {len}, chunks of {step}: the last row differs from the fresh prefill's");
            assert_eq!(
                e.backend.m.state_hashes(2, p).unwrap(),
                ref_hash,
                "cut {len}, chunks of {step}: KV or state differs from the fresh prefill's"
            );
            runs += 1;
        }
    }
    assert_eq!(runs, 3);
    eprintln!("chunk grids: {runs} resumes (cuts 400 and 1440 taken inside the donor's chunk; the rest prefilled in chunks of 1,024 and of 3) equal the fresh prefill");
    for (_, h) in taken {
        e.backend.prefix_release(h);
    }
}

fn req(slot: usize, tokens: &[u32], first: usize) -> RowRequest {
    RowRequest { slot, tokens: tokens.to_vec(), first_position: first as u64 }
}

#[test]
#[ignore]
fn a_snapshot_settles_what_a_window_owes_and_a_restore_leaves_nothing_owed() {
    let Some(mut e) = engine(PrefixCacheOption::Mib(2000), 3, 4096) else { return };
    let m: &mut Model = &mut e.backend.m;
    let prompt = random_prompt(400, 5);
    let p = prompt.len();
    m.begin_slot(0).unwrap();
    m.prefill_chunk(0, 0, &prompt, p, Some(CAND), 0).unwrap().unwrap();
    // two rounds of window + commit leave the second one's accepted rows owed to S (B2: the next window kernel folds them in)
    let toks: Vec<u32> = (0..4).map(|i| 2000 + 31 * i).collect();
    m.window_candidates(&RowBatch::from_requests(&[req(0, &toks, p)]), CAND).unwrap();
    m.commit(0, 3);
    m.window_candidates(&RowBatch::from_requests(&[req(0, &toks, p + 3)]), CAND).unwrap();
    m.commit(0, 2);
    m.flush().unwrap();
    assert_eq!(m.owed_slots(), vec![0], "the slot owes accepted rows");
    // the snapshot of a slot that owes rows is the state AFTER them
    let snap = m.snaps.as_mut().unwrap().alloc().unwrap();
    m.snapshot_state(0, snap).unwrap();
    assert_eq!(m.owed_slots(), Vec::<usize>::new(), "taking the snapshot applied what was owed");
    let settled = m.state_hashes(0, p + 5).unwrap();
    let (ss, sc) = m.snaps.as_ref().unwrap().gdn_hashes(snap).unwrap();
    assert_eq!((settled.s, settled.conv), (ss, sc), "the snapshot lacks the rows the slot owed");
    m.snaps.as_mut().unwrap().free_slot(snap);
    // a restore into a slot that owes rows of its previous sequence leaves it owing nothing, and the restored state is what decodes next
    let (row, taken) = prefill_all(&mut e.backend, 1, &prompt, 0, &[300]);
    let _ = row;
    let (len, h) = &taken[0];
    let m = &mut e.backend.m;
    m.begin_slot(2).unwrap();
    m.prefill_chunk(2, 0, &random_prompt(200, 6), 200, Some(CAND), 0).unwrap().unwrap();
    m.window_candidates(&RowBatch::from_requests(&[req(2, &toks, 200)]), CAND).unwrap();
    m.commit(2, 4);
    m.flush().unwrap();
    assert_eq!(m.owed_slots(), vec![2]);
    e.backend.prefix_restore(2, h, *len).unwrap();
    assert!(
        e.backend.m.owed_slots().is_empty(),
        "a restore must leave no owed rows behind: they would be replayed into the restored state"
    );
    let (r2, _) = prefill_all(&mut e.backend, 2, &prompt, *len, &[]);
    let (r1, _) = prefill_all(&mut e.backend, 0, &prompt, 0, &[]);
    assert!(r1.bits_eq(&r2), "the restored slot's prefill differs from a cold one's");
    e.backend.prefix_release(h.clone());
}

/// The tree of the tests below: row 0 the root, rows 1 and 2 its children, row 3 a child of row 2, rows 4 and 5 children of row 3; the accepted
/// path `0 -> 2 -> 3 -> 5` is not a prefix of the window.
const TREE_PARENTS: [i32; 6] = [-1, 0, 0, 2, 3, 3];
const TREE_PATH: [usize; 4] = [0, 2, 3, 5];

fn tree_req(slot: usize, tokens: &[u32], first: usize) -> TreeRequest {
    TreeRequest { slot, tokens: tokens.to_vec(), parents: TREE_PARENTS.to_vec(), first_position: first as u64 }
}

#[test]
#[ignore]
fn a_snapshot_settles_what_a_tree_window_owes_and_a_restore_leaves_nothing_owed() {
    let Some(mut e) = engine_with(PrefixCacheOption::Mib(2000), 4, 4096, true) else { return };
    let m: &mut Model = &mut e.backend.m;
    assert!(m.trees_enabled().0, "the engine was built with --spec-tree");
    let prompt = random_prompt(400, 5);
    let p = prompt.len();
    // slot 0 runs tree windows that commit a PATH (owed to S as a row list), slot 1 the same paths as plain chains: the serial mirror
    for slot in [0usize, 1] {
        m.begin_slot(slot).unwrap();
        m.prefill_chunk(slot, 0, &prompt, p, Some(CAND), 0).unwrap().unwrap();
    }
    let toks: Vec<u32> = (0..6).map(|i| 2000 + 31 * i).collect();
    let chain: Vec<u32> = TREE_PATH.iter().map(|&r| toks[r]).collect();
    let mut first = p;
    for _ in 0..2 {
        m.window_candidates(&RowBatch::from_trees(&[tree_req(0, &toks, first)]), CAND).unwrap();
        m.commit_path(0, &TREE_PATH).unwrap();
        m.window_candidates(&RowBatch::from_requests(&[req(1, &chain, first)]), CAND).unwrap();
        m.commit(1, chain.len());
        first += chain.len();
    }
    // the last tree commit is what the slot owes (the mirror's window above replayed the slot's earlier debt: the slot did not run in that window)
    m.window_candidates(&RowBatch::from_trees(&[tree_req(0, &toks, first)]), CAND).unwrap();
    m.commit_path(0, &TREE_PATH).unwrap();
    m.flush().unwrap();
    assert_eq!(m.owed_slots(), vec![0], "the slot owes the PATH it committed last");
    // the snapshot of a slot that owes a path is the state AFTER it
    let snap = m.snaps.as_mut().unwrap().alloc().unwrap();
    m.snapshot_state(0, snap).unwrap();
    assert_eq!(m.owed_slots(), Vec::<usize>::new(), "taking the snapshot applied what was owed");
    let (ss, sc) = m.snaps.as_ref().unwrap().gdn_hashes(snap).unwrap();
    // the serial mirror runs the same last path as a plain chain
    m.window_candidates(&RowBatch::from_requests(&[req(1, &chain, first)]), CAND).unwrap();
    m.commit(1, chain.len());
    first += chain.len();
    let (h0, h1) = (m.state_hashes(0, first).unwrap(), m.state_hashes(1, first).unwrap());
    assert_eq!((h0.s, h0.conv), (ss, sc), "the snapshot lacks the path the slot owed");
    assert_eq!((h0.s, h0.conv, h0.kv), (h1.s, h1.conv, h1.kv), "the tree slot's state differs from its serial mirror's");
    m.snaps.as_mut().unwrap().free_slot(snap);
    // a restore into a slot that owes a PATH of its previous sequence leaves it owing nothing
    let (_, taken) = prefill_all(&mut e.backend, 3, &prompt, 0, &[300]);
    let (len, h) = &taken[0];
    let m = &mut e.backend.m;
    m.begin_slot(2).unwrap();
    m.prefill_chunk(2, 0, &random_prompt(200, 6), 200, Some(CAND), 0).unwrap().unwrap();
    m.window_candidates(&RowBatch::from_trees(&[tree_req(2, &toks, 200)]), CAND).unwrap();
    m.commit_path(2, &TREE_PATH).unwrap();
    m.flush().unwrap();
    assert!(m.owed_slots().contains(&2));
    e.backend.prefix_restore(2, h, *len).unwrap();
    assert!(
        !e.backend.m.owed_slots().contains(&2),
        "a restore must leave no owed path behind: it would be replayed into the restored state"
    );
    // the restored slot decodes like a cold one: its suffix prefill equals the cold prefill's, and so does the next TREE window
    let (r2, _) = prefill_all(&mut e.backend, 2, &prompt, *len, &[]);
    let (r1, _) = prefill_all(&mut e.backend, 0, &prompt, 0, &[]);
    assert!(r1.bits_eq(&r2), "the restored slot's prefill differs from a cold one's");
    let m = &mut e.backend.m;
    let (a, b) = (
        m.window_candidates(&RowBatch::from_trees(&[tree_req(0, &toks, p)]), CAND).unwrap(),
        m.window_candidates(&RowBatch::from_trees(&[tree_req(2, &toks, p)]), CAND).unwrap(),
    );
    assert!(
        a.ids == b.ids && a.vals.iter().map(|v| v.to_bits()).eq(b.vals.iter().map(|v| v.to_bits())),
        "a tree window after a restore differs from one after a cold prefill"
    );
    e.backend.prefix_release(h.clone());
}

#[test]
#[ignore]
fn cached_replies_equal_uncached_replies_with_trees() {
    let Some(mut e) = engine_with(PrefixCacheOption::Mib(6000), 4, 4096, true) else { return };
    assert!(e.verifies_trees(), "the engine was built with --spec-tree");
    let mut cases = 0;
    for (p, ext) in [(129usize, 17usize), (257, 17), (257, 300), (4097, 17)] {
        let a = random_prompt(p, 9000 + 1000 * p as u64 + ext as u64);
        let mut b = a.clone();
        b.extend(random_prompt(ext, 9500 + ext as u64));
        let oa = one(&mut e, request(a.clone(), 24, 1, 0.0));
        assert_eq!(oa.cached_tokens, 0);
        for (round, (seed, t)) in [(2u64, 0.7f64), (3, 0.0)].into_iter().enumerate() {
            let rb = request(b.clone(), 64, seed, t);
            let uncached = one(&mut e, Request { cache: false, ..rb.clone() });
            let cached = one(&mut e, rb.clone());
            // the first run of B resumes from A's entry; it leaves an entry of its own, which the second run resumes from
            let expect = if round == 0 { p - 16 } else { p + ext - 16 };
            assert_eq!(cached.cached_tokens, expect, "p{p}+{ext} round {round}: tokens resumed from the cache");
            assert_eq!(
                cached.tokens, uncached.tokens,
                "p{p}+{ext} T={t}: the cached reply differs from the uncached one after tree rounds"
            );
            assert_eq!(cached.finish, uncached.finish);
            let serial = one(&mut e, Request { draft: false, ..rb });
            assert_eq!(serial.tokens, cached.tokens, "p{p}+{ext} T={t}: serial differs from the tree-drafted reply");
            cases += 1;
        }
    }
    let trees: u64 = e.spec_stats().trees.iter().map(|t| t.rounds).sum();
    assert!(trees > 0, "no tree round ran");
    let s = e.prefix_stats();
    eprintln!(
        "with trees: cached == uncached == serial in {cases} cases ({trees} tree rounds); {} hits of {} lookups",
        s.hits, s.lookups
    );
    assert!(s.hits >= 4, "{s:?}");
}

// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
#[ignore]
fn the_snapshot_pool_is_what_the_plan_says() {
    let Some(e) = engine(PrefixCacheOption::Mib(3000), 4, 4096) else { return };
    let caps = e.backend.prefix_caps().unwrap();
    let m = &e.backend.m;
    let sn = m.snaps.as_ref().unwrap();
    // the snapshot pool allocates exactly the slots and bytes the plan counted
    assert_eq!(sn.slots, caps.budget.snapshot_slots);
    assert_eq!(sn.allocated_bytes(), caps.budget.snapshot_slots as u64 * caps.budget.snapshot_bytes);
    assert_eq!(caps.budget.snapshot_bytes, 195_887_104, "a slot is the recurrent state plus the draft window");
    // the KV pool is a page of 16 MiB each and holds the cache's pages on top of admission's
    let page = (m.kv.iter().map(|k| (k.k.bytes() + k.v.bytes()) as u64).sum::<u64>()) / m.kv[0].n_blocks as u64;
    assert_eq!(page, caps.budget.page_bytes);
    assert!(m.kv[0].n_blocks > caps.budget.page_blocks + 4);
    assert!(caps.budget.bytes() <= 3000 << 20);
    eprintln!(
        "snapshot pool {} slots = {} MiB; pages {} (+{} for the cache)",
        sn.slots,
        sn.allocated_bytes() >> 20,
        m.kv[0].n_blocks,
        caps.budget.page_blocks
    );
}

// ------------------------------------------------------------------------------------------------------------------------------------

/// Submit every request at once and run to the end. Returns each request's output (in request order), and how many prompts had their first token
/// after the first step (what the policy served in a step: one under `Serial`).
fn run_together(e: &mut QwenEngine, reqs: &[Request]) -> (Vec<Output>, usize) {
    e.take_events(); // what earlier requests reported
    let ids: Vec<u64> = reqs.iter().map(|r| e.submit_deferred(r.clone()).unwrap()).collect();
    e.step().unwrap();
    let served = e.take_events().iter().filter(|ev| !ev.tokens.is_empty()).count();
    e.run().unwrap();
    let outs = e.take_outputs();
    assert!(e.take_failed().is_empty());
    (ids.iter().map(|id| outs.iter().find(|o| o.id == *id).unwrap().clone()).collect(), served)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// The system prompt and a question of its own: hits the shared entry.
    Sharer,
    /// 128 tokens or fewer: batched in one forward, never cached.
    Short,
    /// A sharer that opted out: looks nothing up.
    OptedOut,
}

#[test]
#[ignore]
fn the_burst_policy_with_the_cache_equals_the_uncached_and_the_serial_replies() {
    burst_with_the_cache(false);
}

/// The same with draft trees (`--spec-tree`, the server's default with the DFlash2 draft): the sharers hit, the short prompts go through one
/// batched forward, and every reply that follows comes out of tree rounds (windows of trees, path commits) next to the burst's prefill steps.
#[test]
#[ignore]
fn the_burst_policy_with_the_cache_equals_the_uncached_and_the_serial_replies_with_trees() {
    burst_with_the_cache(true);
}

fn burst_with_the_cache(trees: bool) {
    // 16 slots: the burst is as wide as this box's server bursts get. The engine is the server's (burst policy with batched short prefill, the
    // cache on); the same engine is switched to the serial policy for the reference arm, so one model load serves all three
    let Some(e) = engine_with(PrefixCacheOption::Mib(6000), 16, 4096, trees) else { return };
    assert_eq!(e.verifies_trees(), trees);
    let mut e = e.with_prefill_policy(PrefillPolicy::BATCHED_SHORT).with_prefill_batching(true);
    let total_pages = e.backend.m.kv_tab.free_blocks();
    let system = random_prompt(1400, 301);
    let with_system = |seed: u64, extra: usize| {
        let mut p = system.clone();
        p.extend(random_prompt(extra, seed));
        p
    };
    // two requests with the system prompt leave the shared entry (the second one discovers the shared blocks: 5 x 256 = 1,280 tokens, which
    // saves more than the policy's 1,024), and the third one hits it
    for q in 0..3u64 {
        let o = one(&mut e, request(with_system(310 + q, 40), 8, q, 0.0));
        assert_eq!(o.cached_tokens >= 1280, q == 2, "warm-up request {q} hit {} tokens", o.cached_tokens);
    }
    for n in [4usize, 8, 16] {
        // sharers of the system prompt, short prompts (128 tokens or fewer: batched in one forward, never cached) and opted-out sharers, each
        // with its own seed and sampling
        let kinds: Vec<Kind> = (0..n)
            .map(|i| {
                if i % 5 == 4 {
                    Kind::Short
                } else if i % 7 == 6 {
                    Kind::OptedOut
                } else {
                    Kind::Sharer
                }
            })
            .collect();
        let reqs: Vec<Request> = kinds
            .iter()
            .enumerate()
            .map(|(i, kind)| {
                let seed = 1000 * n as u64 + i as u64;
                let prompt = match kind {
                    Kind::Short => random_prompt(40 + 7 * (i % 4), seed),
                    _ => with_system(seed, 25 + 3 * i),
                };
                let mut r = request(prompt, 40, seed, if i % 2 == 0 { 0.0 } else { 0.9 });
                r.cache = *kind != Kind::OptedOut;
                r
            })
            .collect();
        let refs: Vec<Output> = reqs.iter().map(|r| one(&mut e, Request { cache: false, ..r.clone() })).collect();
        let (burst, served_burst) = run_together(&mut e, &reqs);
        e = e.with_prefill_policy(PrefillPolicy::Serial);
        let (serial, served_serial) = run_together(&mut e, &reqs);
        e = e.with_prefill_policy(PrefillPolicy::BATCHED_SHORT);
        assert_eq!(served_serial, 1, "n={n}: the serial policy serves one prompt in a step");
        assert!(served_burst >= 2, "n={n}: the burst policy served {served_burst} prompts in its first step");
        for (i, r) in reqs.iter().enumerate() {
            let what = format!("n={n} request {i} ({:?}, {} tokens)", kinds[i], r.prompt.len());
            assert_eq!(burst[i].tokens, refs[i].tokens, "{what}: the burst reply differs from its uncached solo reply");
            assert_eq!(burst[i].finish, refs[i].finish, "{what}");
            assert_eq!(serial[i].tokens, refs[i].tokens, "{what}: the serial reply differs from its uncached solo reply");
            assert_eq!(refs[i].cached_tokens, 0);
            if kinds[i] == Kind::Sharer {
                assert!(
                    burst[i].cached_tokens >= 1280,
                    "{what}: a sharer hits the shared blocks, got {}",
                    burst[i].cached_tokens
                );
            } else {
                assert_eq!(
                    burst[i].cached_tokens, 0,
                    "{what}: a prompt of 128 tokens or fewer, or an opted-out one, is never cached"
                );
            }
            assert_eq!(r.prompt.len() <= 128, kinds[i] == Kind::Short, "{what}");
        }
        // the books: with nothing running, the pool and the snapshot slots hold exactly what the entries hold
        let s = e.prefix_stats();
        assert_eq!(
            e.backend.m.snaps.as_ref().unwrap().free_count() as u64,
            s.snapshot_slots - s.entries,
            "n={n}: snapshot slots"
        );
        assert_eq!(total_pages - e.backend.m.kv_tab.free_blocks(), s.pages_held as usize, "n={n}: pages");
        eprintln!(
            "burst n={n}: equal to uncached and to serial, {served_burst} prompts served in the first step (serial {served_serial}), {} entries, {} hits",
            s.entries, s.hits
        );
    }
    let tree_rounds: u64 = e.spec_stats().trees.iter().map(|t| t.rounds).sum();
    assert_eq!(tree_rounds > 0, trees, "{tree_rounds} tree rounds with trees {trees}");
    eprintln!("with trees {trees}: {tree_rounds} tree rounds");
}

#[allow(dead_code)]
fn _engine_type(_: &Engine<gb_model::backend::QwenBackend>) {}
