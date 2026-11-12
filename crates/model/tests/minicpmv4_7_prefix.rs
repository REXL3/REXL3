//! The MiniCPM-V-4.7 prefix cache on a real pack (E6b of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`, section 14).
//!
//! `REXL3_MINICPM_DIR` names the EXL3 pack: the converter's 8-layer smoke pack (6 gated delta net and 2 attention layers: every kind
//! of state the snapshot holds, in seconds) or the full 40-layer pack. Every test is `#[ignore]` (GPU) and prints `not run:` without
//! it. One model per test: run with `--test-threads=1`. The smoke pack takes the shared GPU lock, the full pack the exclusive one.
//!
//! ```text
//! flock -s -w 1800 /tmp/rexl3-gpu.lock env REXL3_MINICPM_DIR=~/models/minicpm-v-4.7-35b-a3b/work/smokeB/o \
//!     cargo test --release -p gb-model --test minicpmv4_7_prefix -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! What runs, and the claim each one makes (every comparison is bit equality, never a tolerance):
//!
//! * `the_cache_is_planned_allocated_and_priced`: the plan's snapshot slots and pages are the arena and the pool the boot made.
//! * `a_snapshot_restores_the_exact_state_at_every_cut`: capture at block-aligned, off-grid and mid-chunk cuts; the snapshot's bytes
//!   are the donor's state bytes; a restored slot holds the donor's GDN state, convolution history and KV rows byte for byte, shares
//!   the full pages by reference (refcounts) and owns a private copy of the partial page; its continuation on another chunk grid
//!   ends at the same candidates bits and the same state as the uncached run.
//! * `cut_points_do_not_depend_on_the_chunk_grid`: the same cut taken under five chunk grids is the same snapshot.
//! * `cached_replies_equal_uncached_replies`: through the real engine, greedy and sampled (keys), drafted and serial, one stream and
//!   several, a burst of identical prompts, a partial-prefix hit, an opt-out, a short prompt; every cached reply equals its uncached
//!   reply, and each scenario asserts that the cache was really used (hits, tokens saved, `Output::cached_tokens`).
//! * `eviction_under_pressure_never_frees_what_a_running_sequence_uses`: entries evicted while a sequence that restored from them
//!   decodes; the sequence's reply equals its uncached reply, and the books balance.
//! * `a_snapshot_never_continues_a_prompt_of_the_other_arithmetic` and `a_failed_restore_ends_only_its_slot`: the guards.

use gb_engine::{Backend, PrefixHandle, Request, RowOutput, RowRequest};
use gb_model::families::minicpmv4_7::backend::MiniCpmBackend;
use gb_model::families::minicpmv4_7::memory::allocations;
use gb_model::families::minicpmv4_7::model::MiniCpmModel;
use gb_model::families::minicpmv4_7::prefix;
use gb_model::families::minicpmv4_7::state::compare_with_plan_cached;
use minicpmv4_7_support::*;
use sha2::{Digest, Sha256};

mod minicpmv4_7_support;

// ------------------------------------------------------------------------------------------------------------ state digests

fn hash_range(h: &mut Sha256, buf: &gb_cuda::DevBuf, at: usize, n: usize) {
    let mut data = vec![0u8; n];
    buf.download_range(at, &mut data).unwrap();
    h.update(data);
}

/// `slot`'s recurrent state in the arena's order: `S` of every GDN layer, then every layer's convolution history.
fn hash_state(m: &mut MiniCpmModel, slot: usize, h: &mut Sha256) {
    m.flush().unwrap();
    m.stream.sync().unwrap();
    let slots = m.opts.slots;
    for s in &m.pools.s {
        hash_range(h, s, slot * (s.bytes() / slots), s.bytes() / slots);
    }
    for c in &m.pools.conv {
        hash_range(h, c, slot * (c.bytes() / slots), c.bytes() / slots);
    }
}

/// The K and V rows of positions `0..len` of every attention layer, through the slot's own page table: only rows a prefill really
/// wrote (the unused tail of a physical page need not match).
fn hash_kv(m: &mut MiniCpmModel, slot: usize, len: usize, h: &mut Sha256) {
    m.stream.sync().unwrap();
    let blocks = m.kv_tab.blocks_of(slot).to_vec();
    assert!(blocks.len() >= len.div_ceil(256), "slot {slot} holds {} pages for {len} tokens", blocks.len());
    for cache in &m.kv {
        let token = cache.kv_heads.token_elems() * 2;
        assert_eq!(cache.block_bytes(), 256 * token);
        for p0 in (0..len).step_by(256) {
            let rows = (len - p0).min(256);
            let at = blocks[p0 / 256] as usize * cache.block_bytes();
            hash_range(h, &cache.k, at, rows * token);
            hash_range(h, &cache.v, at, rows * token);
        }
    }
}

/// Everything a prompt of `len` tokens leaves in `slot`: the GDN state, the convolution history, the KV rows and the rope delta.
fn digest(m: &mut MiniCpmModel, slot: usize, len: usize) -> Vec<u8> {
    let mut h = Sha256::new();
    hash_state(m, slot, &mut h);
    hash_kv(m, slot, len, &mut h);
    h.update(m.rope_delta[slot].to_le_bytes());
    h.finalize().to_vec()
}

fn state_hash(m: &mut MiniCpmModel, slot: usize) -> Vec<u8> {
    let mut h = Sha256::new();
    hash_state(m, slot, &mut h);
    h.finalize().to_vec()
}

fn sha(bytes: &[u8]) -> Vec<u8> {
    Sha256::digest(bytes).to_vec()
}

// -------------------------------------------------------------------------------------------- the backend-level primitives

/// A prefill of `prompt` in `slot` with snapshots at `cuts`: the handles taken, each with the digest of the slot at the cut (read
/// the moment the slice ended there) and the snapshot's own bytes; and the final candidates row.
struct Donor {
    taken: Vec<(usize, PrefixHandle, Vec<u8>, Vec<u8>)>,
    row: RowOutput,
}

fn donor(b: &mut MiniCpmBackend, slot: usize, prompt: &[u32], cuts: &[usize]) -> Donor {
    let mut done = 0;
    let mut taken = Vec::new();
    loop {
        let step = b.prefill_slice_cuts(slot, prompt, &[], done, 16, cuts);
        done = step.step.done;
        for c in step.captured {
            assert_eq!(c.len, done, "a cut is taken exactly where it was asked");
            let at_cut = digest(&mut b.m, slot, c.len);
            let bytes = b.prefix_arena().unwrap().read_slot(c.handle.snap).unwrap();
            taken.push((c.len, c.handle, at_cut, sha(&bytes)));
        }
        if let Some(row) = step.step.row {
            assert!(b.take_error().is_none(), "the prefill failed");
            return Donor { taken, row };
        }
    }
}

/// The rest of `prompt` from `done` in `slot`, whatever chunk grid the backend has now.
fn finish(b: &mut MiniCpmBackend, slot: usize, prompt: &[u32], mut done: usize) -> RowOutput {
    loop {
        let step = b.prefill_slice(slot, prompt, &[], done, 16);
        done = step.done;
        if let Some(row) = step.row {
            assert!(b.take_error().is_none(), "the continuation failed");
            return row;
        }
    }
}

/// A cold prefill of the whole prompt with no cuts on `slot`.
fn cold(b: &mut MiniCpmBackend, slot: usize, prompt: &[u32]) -> RowOutput {
    finish(b, slot, prompt, 0)
}

// ---------------------------------------------------------------------------------------------------------------- the tests

#[test]
#[ignore]
fn the_cache_is_planned_allocated_and_priced() {
    let Some(mut rig) = Rig::load(3, 4096, 512, 3) else { return };
    let (cache, slots, pages) = (rig.plan.memory.cache, rig.report.slots, rig.report.kv_pages);
    assert!(cache.enabled() && cache.snapshot_slots >= 2, "{cache:?}");
    assert_eq!(pages, rig.plan.memory.kv_blocks + cache.page_blocks);
    let cfg = rig.cfg.clone();
    // the arena and the pool are exactly what the plan priced, item by item
    let items = allocations(&cfg, None);
    let held = rig.report.state.clone();
    let diffs = compare_with_plan_cached(&items, &held, slots, pages, 64, cache.snapshot_slots);
    assert!(diffs.is_empty(), "{diffs:?}");
    assert_eq!(held.prefix_snapshots, cache.snapshot_slots as u64 * cache.snapshot_bytes);
    assert_eq!(rig.m().kv[0].n_blocks, pages);
    let arena = rig.eng.backend.prefix_arena().unwrap();
    assert_eq!(arena.allocated_bytes(), held.prefix_snapshots);
    assert_eq!(arena.free_count(), cache.snapshot_slots);
    // the engine runs it, with the plan's numbers, and its namespace is the pack's
    let st = rig.eng.prefix_stats();
    assert!(st.enabled);
    assert_eq!(
        (st.snapshot_slots, st.page_blocks, st.bytes_budget),
        (cache.snapshot_slots as u64, cache.page_blocks as u64, cache.bytes())
    );
    let caps = rig.eng.backend.prefix_caps().unwrap();
    assert_eq!(caps.budget, cache);
    assert_eq!(
        caps.namespace,
        prefix::namespace_of(
            &rig.report.fingerprint,
            &gb_cuda::exl3_linear::prefill_pin_name(),
            "grouped",
            prefix::NO_IMAGES,
            prefix::NO_DRAFT
        )
    );
    assert_eq!(rig.eng.backend.prefix_path(5000), Some(prefix::PATH_ID), "the default prefill MoE arm is the grouped one");
    // the admitted pool is the share without the cache's pages: a request over it is refused whatever the cache holds
    assert_eq!(rig.plan.memory.kv_tokens, ((pages - cache.page_blocks - slots) * 256) as u64);
    // the cache is not counted in the plan's admitted bytes, and the plan stays inside what it was given
    assert!(rig.plan.memory.planned_bytes(slots) <= rig.plan.free_after_weights);
}

/// The core gate: for every kind of cut (a page boundary, off the grid, mid-chunk), the snapshot is the donor's state byte for byte,
/// the restored slot is the donor's slot at the cut, and the suffix, on another chunk grid, ends at the uncached bits.
#[test]
#[ignore]
fn a_snapshot_restores_the_exact_state_at_every_cut() {
    let Some(mut rig) = Rig::load(4, 4096, 1024, 12) else { return };
    let total = 2300;
    let prompt = toks(7, total);
    let cuts = [129usize, 255, 256, 257, 513, 777, 1024, 1536, 2048];
    let b = &mut rig.eng.backend;
    b.chunk = 1024;
    // the uncached run on slot 3: no cuts at all
    let reference = cold(b, 3, &prompt);
    let ref_state = digest(&mut b.m, 3, total);
    // the donor on slot 2 takes one snapshot at each cut in one pass (a cut at 1024 and 2048 is a chunk boundary; the others end
    // their slice early)
    let d = donor(b, 2, &prompt, &cuts);
    assert_eq!(d.taken.iter().map(|t| t.0).collect::<Vec<_>>(), cuts.to_vec(), "every cut is taken once, in order");
    assert!(d.row.bits_eq(&reference), "taking snapshots does not change the prompt's bits");
    assert_eq!(digest(&mut b.m, 2, total), ref_state, "nor its state");
    assert_eq!(b.prefix_free_snapshots(), rig.plan.memory.cache.snapshot_slots - cuts.len());
    for (cut, handle, at_cut, snap_hash) in &d.taken {
        let cut = *cut;
        // the snapshot's bytes are the donor's recurrent state at the cut (S of every layer, then the conv histories)
        let donor_at_cut = at_cut.clone();
        // restore into slot 0 (the engine's call), then compare slot 0 with the donor as it was at the cut
        b.chunk = 257; // the suffix runs on another grid than the donor's
        b.prefix_restore(0, handle, cut).unwrap_or_else(|e| panic!("restore at {cut}: {e}"));
        assert_eq!(digest(&mut b.m, 0, cut), donor_at_cut, "state, conv history and KV rows after the restore at {cut}");
        // the snapshot is what was restored: re-reading the arena slot gives the hash taken at capture
        let bytes = b.prefix_arena().unwrap().read_slot(handle.snap).unwrap();
        assert_eq!(&sha(&bytes), snap_hash, "the arena slot is unchanged by the restore at {cut}");
        // pages: the full pages are the donor's by reference, the partial page is a private copy
        let (full, partial) = (cut / 256, cut % 256 != 0);
        assert_eq!(prefix::page_plan(cut), (full, partial));
        let mine = b.m.kv_tab.blocks_of(0).to_vec();
        assert_eq!(mine.len(), cut.div_ceil(256));
        assert_eq!(&mine[..full], &handle.pages[..full], "full pages are shared");
        for p in &handle.pages[..full] {
            assert!(b.m.kv_tab.refcount(*p) >= 3, "page {p}: the donor, the entry and the restored slot hold it");
        }
        if partial {
            assert_ne!(mine[full], handle.pages[full], "the partial page is copied, not shared");
            assert_eq!(b.m.kv_tab.refcount(mine[full]), 1, "the copy is the restored slot's alone");
        }
        // the suffix on the other grid ends at the uncached bits, with the uncached state
        let row = finish(b, 0, &prompt, cut);
        assert!(row.bits_eq(&reference), "candidates after a hit at {cut}: {:?} against {:?}", row.ids, reference.ids);
        assert_eq!(digest(&mut b.m, 0, total), ref_state, "state after a hit at {cut}");
        b.release(0);
    }
    // every handle goes back exactly once, and the pool is whole
    let held: Vec<PrefixHandle> = d.taken.into_iter().map(|t| t.1).collect();
    b.release(2);
    b.release(3);
    for h in held {
        b.prefix_release(h);
    }
    assert!(b.take_error().is_none());
    assert_eq!(b.prefix_free_snapshots(), rig.plan.memory.cache.snapshot_slots);
    assert_eq!(rig.eng.backend.m.kv_tab.free_blocks(), rig.report.kv_pages, "every page is back in the pool");
}

/// A snapshot is a function of the prompt and the cut only: the same cut taken under five chunk grids is the same snapshot, and the
/// same state in the slot. (`REXL3_MINICPM_PREFIX_TOKENS` runs it at a long context.)
#[test]
#[ignore]
fn cut_points_do_not_depend_on_the_chunk_grid() {
    let total = std::env::var("REXL3_MINICPM_PREFIX_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(2600usize);
    let Some(mut rig) = Rig::load(2, total + 64, 4096, 4) else { return };
    let prompt = toks(21, total);
    let cuts = [777usize, 1024, total - 100];
    // per grid: the grid, the (slot digest, snapshot hash) of every cut, and the final digest
    type PerGrid = (usize, Vec<(Vec<u8>, Vec<u8>)>, Vec<u8>);
    let mut seen: Vec<PerGrid> = Vec::new();
    for grid in [97usize, 256, 257, 1000, 4096] {
        let b = &mut rig.eng.backend;
        b.chunk = grid;
        let d = donor(b, 0, &prompt, &cuts);
        assert_eq!(d.taken.len(), cuts.len(), "grid {grid}");
        let per_cut: Vec<(Vec<u8>, Vec<u8>)> = d.taken.iter().map(|t| (t.2.clone(), t.3.clone())).collect();
        let fin = digest(&mut b.m, 0, total);
        for t in d.taken {
            b.prefix_release(t.1);
        }
        b.release(0);
        seen.push((grid, per_cut, fin));
    }
    for (grid, per_cut, fin) in &seen[1..] {
        for (i, ((state, snap), (state0, snap0))) in per_cut.iter().zip(&seen[0].1).enumerate() {
            assert_eq!(state, state0, "grid {grid} against grid {}: the slot at cut {}", seen[0].0, cuts[i]);
            assert_eq!(snap, snap0, "grid {grid} against grid {}: the snapshot at cut {}", seen[0].0, cuts[i]);
        }
        assert_eq!(fin, &seen[0].2, "grid {grid}: the final state");
    }
}

/// Through the real engine: cached == uncached, token for token, in every way a request meets the cache.
#[test]
#[ignore]
fn cached_replies_equal_uncached_replies() {
    let Some(mut rig) = Rig::load(4, 8192, 512, 16) else { return };
    let e = &mut rig.eng;
    let (a, tail1) = (toks(11, 900), toks(12, 400));
    let b_prompt = cat(&a, &tail1); // extends `a`
    let hits = |e: &Eng| e.prefix_stats().hits;
    let saved = |e: &Eng| e.prefix_stats().tokens_saved;

    // the donor: uncached reference first, then the cached request that leaves the entries
    let ref_a = one(e, request(&a, greedy(), 24, false, false));
    assert_eq!((ref_a.cached_tokens, e.prefix_stats().inserts), (0, 0), "an opted-out request leaves nothing");
    let got_a = one(e, request(&a, greedy(), 24, false, true));
    same("donor", &got_a, &ref_a);
    assert_eq!(got_a.cached_tokens, 0);
    assert!(e.prefix_stats().inserts >= 1, "the donor left an entry");

    // an extension hits the donor's end cut (the prompt's end less the tail: 900 - 16, off the page grid), whatever the sampler
    // (the first extension leaves its own end cut, 1300 - 16, which every later identical request hits instead)
    for (i, (name, sampling, draft)) in [
        ("greedy serial", greedy(), false),
        ("greedy drafted", greedy(), true),
        ("sampled(5) serial", sampled(5), false),
        ("sampled(5) drafted", sampled(5), true),
        ("sampled(6) serial", sampled(6), false),
    ]
    .into_iter()
    .enumerate()
    {
        let want = one(e, request(&b_prompt, sampling, 24, draft, false));
        assert_eq!(want.cached_tokens, 0, "{name}: the reference path looks nothing up");
        let (h0, s0) = (hits(e), saved(e));
        let got = one(e, request(&b_prompt, sampling, 24, draft, true));
        let hit = if i == 0 { 884 } else { 1284 };
        assert_eq!(got.cached_tokens, hit, "{name}: the hit is the longest end cut stored");
        assert_eq!((hits(e), saved(e)), (h0 + 1, s0 + hit as u64), "{name}: the engine counted it");
        same(name, &got, &want);
    }

    // a partial-prefix hit: shares only the first 700 tokens of `a`. The first such prompt finds the shared blocks (the ghost set)
    // and leaves the entry at the shared cut; the second hits it with its first 512 tokens
    let p1 = cat(&a[..700], &toks(14, 200));
    let p2 = cat(&a[..700], &toks(15, 200));
    let want1 = one(e, request(&p1, greedy(), 16, false, false));
    let want2 = one(e, request(&p2, greedy(), 16, false, false));
    let got1 = one(e, request(&p1, greedy(), 16, false, true));
    same("partial, first sighting", &got1, &want1);
    let h0 = hits(e);
    let got2 = one(e, request(&p2, greedy(), 16, false, true));
    assert_eq!(got2.cached_tokens, 512, "the shared prefix is hit at its last whole block");
    assert_eq!(hits(e), h0 + 1);
    same("partial hit", &got2, &want2);

    // several streams at once, each its own continuation of `a` or of `b_prompt`, beside an unrelated prompt: every one equals its
    // own uncached reply
    let rs: Vec<Request> = vec![
        request(&cat(&b_prompt, &toks(31, 100)), greedy(), 20, true, true),
        request(&cat(&a, &toks(32, 250)), sampled(9), 20, false, true),
        request(&toks(33, 700), greedy(), 20, true, true),
        request(&cat(&b_prompt, &toks(34, 60)), sampled(10), 20, true, true),
    ];
    let wants = many(e, rs.iter().map(|r| Request { cache: false, ..r.clone() }).collect());
    let h0 = hits(e);
    let gots = many(e, rs);
    for (i, (g, w)) in gots.iter().zip(&wants).enumerate() {
        same(&format!("stream {i}"), g, w);
    }
    assert!(hits(e) >= h0 + 2, "the continuations hit: {} -> {}", h0, hits(e));
    assert!(gots[0].cached_tokens > 0 && gots[1].cached_tokens > 0 && gots[3].cached_tokens > 0);
    assert_eq!(gots[2].cached_tokens, 0, "an unrelated prompt hits nothing");

    // a burst of identical prompts, then one more: all equal the reference, the later one hits the end cut
    let burst = toks(40, 1000);
    let want = one(e, request(&burst, greedy(), 12, true, false));
    let outs = many(e, (0..4).map(|_| request(&burst, greedy(), 12, true, true)).collect());
    for (i, o) in outs.iter().enumerate() {
        same(&format!("burst member {i}"), o, &want);
    }
    let again = one(e, request(&burst, greedy(), 12, true, true));
    assert_eq!(again.cached_tokens, 984, "the end cut of the identical prompt");
    same("after the burst", &again, &want);

    // a short prompt (the decode-GEMV arm) is never cached, and an opt-out leaves nothing behind
    let entries = e.prefix_stats().entries;
    let short = toks(50, 100);
    let w = one(e, request(&short, greedy(), 12, false, false));
    let g = one(e, request(&short, greedy(), 12, false, true));
    assert_eq!(g.cached_tokens, 0);
    same("short", &g, &w);
    assert_eq!(e.prefix_stats().entries, entries, "a prompt of 128 tokens or fewer leaves no entry");

    // the books balance with nothing running: every snapshot slot is an entry's or free, every page an entry's or free
    let st = e.prefix_stats();
    assert_eq!(st.restore_failures, 0);
    let arena = e.backend.prefix_arena().unwrap();
    assert_eq!(arena.held_count() as u64, st.entries);
    assert_eq!(arena.held_count() + arena.free_count(), rig.plan.memory.cache.snapshot_slots);
    assert_eq!(e.backend.m.kv_tab.free_blocks() as u64 + st.pages_held, rig.report.kv_pages as u64, "pages: free + held = pool");
    eprintln!("prefix stats at the end: {st:?}");
}

/// Eviction while a sequence runs on pages its entry shared: the entry goes, the sequence's pages stay (references), and nothing the
/// sequence reads is recycled under it.
#[test]
#[ignore]
fn eviction_under_pressure_never_frees_what_a_running_sequence_uses() {
    // two snapshot slots: the third entry evicts one
    let Some(mut rig) = Rig::load(3, 8192, 512, 2) else { return };
    assert_eq!(rig.plan.memory.cache.snapshot_slots, 2, "the budget holds two entries");
    let e = &mut rig.eng;
    let (a, b2, c, d) = (toks(61, 900), toks(62, 900), toks(63, 900), toks(64, 900));
    let s_prompt = cat(&a, &toks(65, 300));
    let want = one(e, request(&s_prompt, sampled(3), 120, false, false));
    one(e, request(&a, greedy(), 4, false, true));
    one(e, request(&b2, greedy(), 4, false, true));
    assert_eq!(e.prefix_stats().entries, 2);
    // S restores from A's entry and decodes; while it does, two new prompts arrive and evict both entries
    let sid = e.submit_deferred(request(&s_prompt, sampled(3), 120, false, true)).unwrap();
    for _ in 0..3 {
        e.step().unwrap();
    }
    assert_eq!(e.prefix_stats().hits, 1, "S hit A's entry");
    let (cid, did) = (
        e.submit_deferred(request(&c, greedy(), 4, false, true)).unwrap(),
        e.submit_deferred(request(&d, greedy(), 4, false, true)).unwrap(),
    );
    let mut outs = Vec::new();
    while e.busy() && e.prefix_stats().evictions < 2 {
        e.step().unwrap();
        outs.extend(e.take_outputs());
    }
    // S must still be running when the evictions have happened, or the scenario proved nothing
    assert!(
        e.busy() && outs.iter().all(|o| o.id != sid),
        "S finished before the entries were evicted: the scenario is too short"
    );
    e.run().unwrap();
    assert!(e.take_failed().is_empty());
    outs.extend(e.take_outputs());
    let s_out = outs.iter().find(|o| o.id == sid).expect("S finished");
    same("the sequence that outlived its entry", s_out, &want);
    assert_eq!(s_out.cached_tokens, 884);
    assert!(outs.iter().any(|o| o.id == cid) && outs.iter().any(|o| o.id == did));
    let st = e.prefix_stats();
    assert!(st.evictions >= 2 && st.evicted_for_slots >= 2, "{st:?}");
    assert_eq!(st.restore_failures, 0);
    // nothing leaked: slots and pages balance
    let arena = e.backend.prefix_arena().unwrap();
    assert_eq!(arena.held_count() as u64, st.entries);
    assert_eq!(e.backend.m.kv_tab.free_blocks() as u64 + st.pages_held, rig.report.kv_pages as u64);
    // and the entry that is left still restores exactly: its reply equals the uncached one
    let after = one(e, request(&c, greedy(), 8, false, true));
    let reference = one(e, request(&c, greedy(), 8, false, false));
    same("an entry after the evictions", &after, &reference);
    assert_eq!(after.cached_tokens, 884, "C's entry survived (D, arriving with no free slot, took no cut)");
}

/// The restore hook has no prompt, so the backend keeps the arithmetic path a restored slot was made under and checks it against the
/// prompt that continues it. A prompt of 128 tokens or fewer takes the decode GEMV: it must never continue a pinned-GEMM snapshot.
#[test]
#[ignore]
fn a_snapshot_never_continues_a_prompt_of_the_other_arithmetic() {
    let Some(mut rig) = Rig::load(3, 2048, 512, 3) else { return };
    let long = toks(71, 400);
    let b = &mut rig.eng.backend;
    let d = donor(b, 0, &long, &[100]);
    assert_eq!(d.taken.len(), 1);
    let handle = d.taken[0].1.clone();
    // a 120-token prompt that shares the 100: GEMV arm, refused, and only its slot is ended
    let short = long[..120].to_vec();
    let other_before = state_hash(&mut b.m, 0);
    b.prefix_restore(1, &handle, 100).unwrap();
    let step = b.prefill_slice(1, &short, &[], 100, 8);
    assert!(step.row.unwrap().values.iter().all(|v| v.is_nan()));
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("other arithmetic"), "{}", e.message);
    assert!(b.m.kv_tab.blocks_of(1).is_empty(), "the refused slot gave back its pages");
    assert_eq!(state_hash(&mut b.m, 0), other_before, "the donor's slot is untouched");
    // the engine never offers it: with the entry present, the same short prompt through the engine hits nothing and equals its cold run
    let e = &mut rig.eng;
    one(e, request(&long, greedy(), 4, false, true));
    let want = one(e, request(&short, greedy(), 8, false, false));
    let got = one(e, request(&short, greedy(), 8, false, true));
    assert_eq!(got.cached_tokens, 0);
    same("short prompt beside a long entry", &got, &want);
    // the same snapshot continues a prompt of its own arithmetic
    let b = &mut rig.eng.backend;
    let longer = cat(&long[..100], &toks(72, 150));
    b.prefix_restore(2, &handle, 100).unwrap();
    let hit = finish(b, 2, &longer, 100);
    let reference = cold(b, 1, &longer);
    assert!(hit.bits_eq(&reference));
    b.release(0);
    b.release(1);
    b.release(2);
    b.prefix_release(handle);
}

/// A restore that fails ends its slot and nothing else: another sequence in the middle of decoding is bit-for-bit what it is alone.
#[test]
#[ignore]
fn a_failed_restore_ends_only_its_slot() {
    let Some(mut rig) = Rig::load(4, 2048, 512, 3) else { return };
    let prompt = toks(81, 500);
    let side = toks(82, 300);
    let b = &mut rig.eng.backend;
    let d = donor(b, 0, &prompt, &[200]);
    let handle = d.taken[0].1.clone();
    // the sequence on slot 2: a prefill, then greedy windows; its reference is the same run alone on slot 3
    let run = |b: &mut MiniCpmBackend, slot: usize, with_failures: Option<&PrefixHandle>| -> Vec<RowOutput> {
        let mut rows = vec![cold(b, slot, &side)];
        let mut next = rows[0].ids[0];
        for i in 0..6 {
            if let (Some(h), 2) = (with_failures, i) {
                // a handle of the wrong length, and a released one: both fail, in the middle of slot `slot`'s decoding
                assert!(b.prefix_restore(1, h, 199).is_err());
                let bad = PrefixHandle { snap: h.snap, pages: vec![] };
                assert!(b.prefix_restore(1, &bad, 200).is_err());
                assert!(b.take_error().is_none(), "a refused restore is not a backend error: the engine prefills cold");
            }
            let req = RowRequest { slot, first_position: (side.len() + i) as u64, tokens: vec![next] };
            let out = b.step(&[req], 8);
            b.commit(slot, 1);
            assert!(b.take_error().is_none());
            next = out[0][0].ids[0];
            rows.push(out[0][0].clone());
        }
        rows
    };
    let alone = run(b, 3, None);
    let beside = run(b, 2, Some(&handle));
    for (i, (x, y)) in beside.iter().zip(&alone).enumerate() {
        assert!(x.bits_eq(y), "row {i} of the sequence beside the failed restores differs from its solo run");
    }
    assert!(b.m.kv_tab.blocks_of(1).is_empty(), "the failed restores left slot 1 empty");
    // and the slot that failed restores fine afterwards, from the same handle
    b.prefix_restore(1, &handle, 200).unwrap();
    let hit = finish(b, 1, &prompt, 200);
    assert!(hit.bits_eq(&d.row));
    for s in 0..4 {
        b.release(s);
    }
    b.prefix_release(handle);
}

/// The slot's rope delta (the canvas offset every later text token of a sequence that had an image carries; E5b) is part of what a
/// snapshot holds: it is restored over whatever the destination slot had, and a snapshot of a text prompt restores a delta of 0.
#[test]
#[ignore]
fn a_snapshot_carries_the_slots_rope_delta() {
    let Some(mut rig) = Rig::load(3, 2048, 512, 3) else { return };
    let prompt = toks(91, 300);
    let b = &mut rig.eng.backend;
    cold(b, 0, &prompt);
    b.m.set_rope_delta(0, 9).unwrap();
    let with_delta = b.with_model_and_arena(|m, a| prefix::capture(m, a, 0, 300).unwrap().unwrap()).unwrap();
    b.m.set_rope_delta(0, 0).unwrap();
    let plain = b.with_model_and_arena(|m, a| prefix::capture(m, a, 0, 300).unwrap().unwrap()).unwrap();
    assert_eq!(b.prefix_arena().unwrap().meta(with_delta.handle.snap).unwrap().rope_delta, 9);
    // the destination had another delta: the snapshot's replaces it
    b.m.set_rope_delta(1, 4).unwrap();
    b.prefix_restore(1, &with_delta.handle, 300).unwrap();
    assert_eq!(b.m.rope_delta[1], 9);
    b.m.set_rope_delta(2, 4).unwrap();
    b.prefix_restore(2, &plain.handle, 300).unwrap();
    assert_eq!(b.m.rope_delta[2], 0);
    // and it is part of the digest the other tests compare: the two restored slots differ in nothing else
    let (d1, d2) = (digest(&mut b.m, 1, 300), digest(&mut b.m, 2, 300));
    assert_ne!(d1, d2, "the digest sees the rope delta");
    for s in 0..3 {
        b.release(s);
    }
    b.prefix_release(with_delta.handle);
    b.prefix_release(plain.handle);
}

/// The strongest "uncached": an engine booted with no cache at all (the plain `prefill_slice_inner` route, no arena) answers the
/// requests a cache engine serves from its entries token for token. The two engines are loaded one after the other.
#[test]
#[ignore]
fn an_engine_with_no_cache_answers_like_a_cache_engine_serving_hits() {
    let (a, tail) = (toks(101, 900), toks(102, 400));
    let ext = cat(&a, &tail);
    let reqs = |_: ()| {
        vec![
            request(&a, greedy(), 20, false, true),
            request(&ext, greedy(), 20, true, true),
            request(&ext, sampled(7), 20, false, true),
            request(&cat(&a[..700], &toks(103, 150)), sampled(8), 20, true, true),
        ]
    };
    let mut cached_replies = Vec::new();
    {
        let Some(mut rig) = Rig::load(2, 4096, 512, 6) else { return };
        for r in reqs(()) {
            cached_replies.push(one(&mut rig.eng, r));
        }
        let st = rig.eng.prefix_stats();
        assert!(st.hits >= 2 && st.tokens_saved >= 1_700, "the cache engine served hits: {st:?}");
        assert_eq!(cached_replies[1].cached_tokens, 884);
        assert_eq!(cached_replies[2].cached_tokens, 1284);
    }
    let Some(mut rig) = Rig::load(2, 4096, 512, 0) else { return };
    assert!(rig.eng.backend.prefix_arena().is_none() && !rig.eng.prefix_stats().enabled);
    for (i, r) in reqs(()).into_iter().enumerate() {
        let plain = one(&mut rig.eng, r);
        assert_eq!(plain.cached_tokens, 0);
        same(&format!("request {i}"), &cached_replies[i], &plain);
    }
}

/// Sets an environment variable for the engine a test boots and puts it back (the tests of a binary run one at a time).
struct EnvGuard(&'static str, Option<String>);

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let old = std::env::var(key).ok();
        std::env::set_var(key, value);
        EnvGuard(key, old)
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.1 {
            Some(v) => std::env::set_var(self.0, v),
            None => std::env::remove_var(self.0),
        }
    }
}

/// E6a's prefill MoE arm is part of the arithmetic: the grouped arm and E3's CoopMK arm leave different bits, so a snapshot of one is
/// refused by a model that runs the other (the restore reads the snapshot's path id), by name, and ends only the slot.
#[test]
#[ignore]
fn a_snapshot_of_the_other_prefill_moe_arm_is_refused() {
    let Some(mut rig) = Rig::load(3, 2048, 512, 3) else { return };
    assert!(rig.eng.backend.m.prefill_moe_grouped(), "the default arm is the grouped one");
    let prompt = toks(111, 600);
    let b = &mut rig.eng.backend;
    let d = donor(b, 0, &prompt, &[300]);
    let handle = d.taken[0].1.clone();
    assert_eq!(b.prefix_arena().unwrap().meta(handle.snap).unwrap().path, prefix::PATH_ID);
    // what a CoopMK-arm snapshot looks like to this model
    let was = b.with_model_and_arena(|_, a| a.retag(handle.snap, prefix::PATH_ID_COOPMK)).unwrap().unwrap();
    assert_eq!(was, prefix::PATH_ID);
    let e = b.prefix_restore(1, &handle, 300).unwrap_err();
    assert!(e.contains("coopmk") && e.contains("grouped") && e.contains("bits differ"), "{e}");
    assert!(b.m.kv_tab.blocks_of(1).is_empty(), "the refused slot holds nothing");
    assert!(b.take_error().is_none(), "a refused restore is the engine's cold prefill, not a backend error");
    // the slot beside it is untouched, and the same snapshot of its own arm restores and ends at the uncached bits
    b.with_model_and_arena(|_, a| a.retag(handle.snap, prefix::PATH_ID)).unwrap();
    b.prefix_restore(1, &handle, 300).unwrap();
    let hit = finish(b, 1, &prompt, 300);
    assert!(hit.bits_eq(&d.row));
    for s in 0..3 {
        b.release(s);
    }
    b.prefix_release(handle);
}

/// The oracle arm caches too: with `REXL3_MINICPM_PREFILL_MOE=coopmk` the same round trip is byte-exact, its snapshots carry the
/// CoopMK path id and its namespace is not the grouped arm's.
#[test]
#[ignore]
fn the_coopmk_prefill_arm_restores_exactly_too() {
    let _arm = EnvGuard::set("REXL3_MINICPM_PREFILL_MOE", "coopmk");
    let Some(mut rig) = Rig::load(3, 4096, 1024, 4) else { return };
    assert!(!rig.eng.backend.m.prefill_moe_grouped());
    let ns = rig.eng.backend.prefix_caps().unwrap().namespace;
    assert_eq!(
        ns,
        prefix::namespace_of(
            &rig.report.fingerprint,
            &gb_cuda::exl3_linear::prefill_pin_name(),
            "coopmk",
            prefix::NO_IMAGES,
            prefix::NO_DRAFT
        )
    );
    assert_eq!(rig.eng.backend.prefix_path(5000), Some(prefix::PATH_ID_COOPMK));
    let prompt = toks(112, 1800);
    let b = &mut rig.eng.backend;
    b.chunk = 1024;
    let reference = cold(b, 2, &prompt);
    let ref_state = digest(&mut b.m, 2, prompt.len());
    let d = donor(b, 0, &prompt, &[257, 777, 1024]);
    assert!(d.row.bits_eq(&reference));
    for (cut, handle, at_cut, _) in &d.taken {
        assert_eq!(b.prefix_arena().unwrap().meta(handle.snap).unwrap().path, prefix::PATH_ID_COOPMK);
        b.chunk = 257;
        b.prefix_restore(1, handle, *cut).unwrap();
        assert_eq!(&digest(&mut b.m, 1, *cut), at_cut, "restore at {cut}");
        let row = finish(b, 1, &prompt, *cut);
        assert!(row.bits_eq(&reference), "candidates after a hit at {cut} under the CoopMK arm");
        assert_eq!(digest(&mut b.m, 1, prompt.len()), ref_state, "state after a hit at {cut}");
        b.release(1);
    }
    for t in d.taken {
        b.prefix_release(t.1);
    }
    b.release(0);
    b.release(2);
    // through the engine, a hit under this arm equals its uncached reply
    let e = &mut rig.eng;
    let ext = cat(&prompt, &toks(113, 200));
    one(e, request(&prompt, greedy(), 8, false, true));
    let want = one(e, request(&ext, sampled(4), 12, false, false));
    let got = one(e, request(&ext, sampled(4), 12, false, true));
    assert!(got.cached_tokens > 0, "the CoopMK arm's engine hit its own entry");
    same("coopmk arm, cached against uncached", &got, &want);
}
