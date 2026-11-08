//! The prefix cache through the real engine and backend on the whole GLM-5.3-Flash model with the NextN layer (90.5 GiB resident, one model per process):
//! `Engine<GlmBackend>` with a cache serves requests whose prompts share a prefix, and every reply equals the reply of the same request on the engine's
//! own uncached path (`Request::cache = false`, the server's `"prefix_cache": false`), token for token, with the NextN drafter at a fixed depth: the
//! speculation counters (`drafted`, `accepted`, `rounds`) too, because GLM's drafter state is a function of the committed rows alone, so a hit changes
//! none of them (Qwen's differ through its non-row-invariant draft absorb). `Request::draft = false` is the serial reference.
//!
//! One test, one model load, a sequence of scenarios on ONE engine (the cache's state carries from a scenario to the next, as a server's does):
//! 1. a sweep of prompt lengths whose end cut takes every residue mod 4, each followed by a longer prompt that extends it (the hit), compared with the
//!    uncached run and the serial run, greedy and sampled, drafted and serial, in two prefill chunk sizes;
//! 2. concurrent sharers of one prefix beside other traffic;
//! 3. a request that opts out looks nothing up and leaves nothing, a differing prefix hits nothing, prompts of 128 tokens or fewer cache like any other;
//! 4. an image prefix (synthetic features: bit equality does not depend on them): the image's content key names the prefix, another image or no key hits
//!    nothing, and every reply equals the uncached one;
//! 5. pressure: more prompts than snapshot slots, then the books balance (every page and snapshot slot is the index's or free).
//!
//! `#[ignore]`, env-gated (`REXL3_GLM_DIR`), under the GPU lock. `REXL3_GLM_CHUNK_ROWS` defaults to 1,024 so that the large-row kernels run.

mod glm_support;

use gb_engine::{CacheBudget, CutPolicy, Engine, ImageSpan, Output, PrefixConfig, Request};
use gb_model::families::glm::backend::GlmBackend;
use gb_model::families::glm::engine::engine_over;
use gb_model::families::glm::prefix;
use gb_model::families::glm::spec::SpecConfig;
use gb_model::families::glm::vision_serve::GlmImageFeatures;
use gb_sampling::Sampling;
use gb_sched::{Admission, ModelMemory};
use gb_spec::SpecMode;
use glm_support::*;
use std::sync::Arc;

const SLOTS: usize = 4;
const ADMIT_BLOCKS: usize = 64; // 16,384 tokens admitted
const CACHE_PAGES: usize = 96; // what the cache may hold on top
const KV_BLOCKS: usize = ADMIT_BLOCKS + CACHE_PAGES;
const MAX_CONTEXT: usize = 8192;
const SNAPSHOTS: usize = 5;
const TAIL: usize = 16;
const MIN_CUT: usize = 128;

type Eng = Engine<GlmBackend>;

fn engine(model: gb_model::families::glm::model::GlmModel, chunk: usize) -> Eng {
    let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1, state_bytes_per_slot: 1, activation_reserve: 0 };
    let admission =
        Admission::new((ADMIT_BLOCKS * 256 + SLOTS) as u64, 0, mm, SLOTS).unwrap().with_max_context(MAX_CONTEXT as u64);
    let snapshot_bytes = prefix::snapshot_bytes(&model.cfg, model.has_mtp()).unwrap();
    let budget = CacheBudget { snapshot_slots: SNAPSHOTS, page_blocks: CACHE_PAGES, snapshot_bytes, page_bytes: 3 << 20 };
    let mut cfg = SpecConfig::new(SpecMode::Mtp);
    cfg.mtp_depth = 2;
    let backend = GlmBackend::with_spec(model, cfg)
        .with_chunk(chunk)
        .with_prefix_cache(prefix::namespace_of("glm-prefix-engine-test", true), budget)
        .unwrap();
    engine_over(backend, admission, true).with_prefix_cache(PrefixConfig {
        policy: CutPolicy { min_cut: MIN_CUT, tail: TAIL, ghost_min_gain: 512, min_gap: 64 },
        ghost_blocks: 1 << 12,
        ..Default::default()
    })
}

fn toks(seed: u64, n: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1 + (x % 150_000) as u32
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

fn one(e: &mut Eng, r: Request) -> Output {
    let id = e.submit(r).unwrap();
    e.run().unwrap();
    assert!(e.take_failed().is_empty(), "no request may fail");
    let mut outs = e.take_outputs();
    outs.retain(|o| o.id == id);
    assert_eq!(outs.len(), 1);
    outs.remove(0)
}

/// `a` against `b`: tokens and finish reason always; the speculation counters too when `counters`.
fn same(a: &Output, b: &Output, counters: bool) -> Result<(), String> {
    if a.tokens != b.tokens {
        let at = a.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y).unwrap_or(a.tokens.len().min(b.tokens.len()));
        return Err(format!("tokens differ at {at} ({} vs {} tokens)", a.tokens.len(), b.tokens.len()));
    }
    if (a.finish, a.stopped) != (b.finish, b.stopped) {
        return Err(format!("finish {:?} vs {:?}", a.finish, b.finish));
    }
    if counters && (a.drafted, a.accepted, a.rounds) != (b.drafted, b.accepted, b.rounds) {
        return Err(format!(
            "speculation counters differ: drafted/accepted/rounds {}/{}/{} vs {}/{}/{}",
            a.drafted, a.accepted, a.rounds, b.drafted, b.accepted, b.rounds
        ));
    }
    Ok(())
}

/// The reply of `r` with the cache off (the reference path of the same engine), and with speculation off (the serial path): `r`'s reply must equal both.
fn references(e: &mut Eng, r: &Request) -> (Output, Output) {
    let uncached = one(e, Request { cache: false, ..r.clone() });
    let serial = one(e, Request { cache: false, draft: false, ..r.clone() });
    (uncached, serial)
}

/// The books balance with nothing running: every page is free or an entry's, every snapshot slot is free or an entry's.
fn balanced(e: &Eng) -> Result<(), String> {
    let st = e.prefix_stats();
    let free = e.backend.m.free_pages();
    if free + st.pages_held as usize != KV_BLOCKS {
        return Err(format!("{free} pages free + {} held by the index != {KV_BLOCKS}", st.pages_held));
    }
    let snaps_free = e.backend.m.prefix_free_snapshots();
    if snaps_free + st.entries as usize != SNAPSHOTS {
        return Err(format!("{snaps_free} snapshot slots free + {} entries != {SNAPSHOTS}", st.entries));
    }
    Ok(())
}

fn end_cut(total: usize) -> usize {
    let c = total.saturating_sub(TAIL);
    if c >= MIN_CUT {
        // under the P2 prefill contract the backend moves the planned cut down to its segment's 16-row KDA grid (`prefix::cuts_for`; a fresh
        // prompt's segment starts at 0) and captures it at that length; P1 takes it as planned
        let none = |_: usize| false;
        prefix::cuts_for(gb_model::families::glm::contract::current(), 0, 0, &[c], &none).first().copied().unwrap_or(0)
    } else {
        0
    }
}

/// 1. The sweep.
fn sweep(e: &mut Eng, chunk: usize) {
    let mut cases = 0;
    for (l1, ext, t) in [
        (300usize, 120usize, 0.0f64),
        (301, 9, 0.8),
        (302, 400, 0.0),
        (303, 40, 0.8),
        (700, 200, 0.0),
        (1000, 300, 0.7),
        (2203, 150, 0.0),
        (2100, 500, 0.8),
    ] {
        let a_prompt = toks(l1 as u64 + chunk as u64, l1);
        let mut b_prompt = a_prompt.clone();
        b_prompt.extend(toks(5000 + ext as u64, ext));
        let a = request(a_prompt, 24, 3, t);
        let b = request(b_prompt, 48, 11, t);
        let oa = one(e, a.clone());
        let (ra, _) = references(e, &a);
        same(&oa, &ra, true).unwrap_or_else(|m| panic!("chunk {chunk}: A at {l1}: {m}"));
        let before = e.prefix_stats();
        let ob = one(e, b.clone());
        let after = e.prefix_stats();
        let (rb, sb) = references(e, &b);
        same(&ob, &rb, true).unwrap_or_else(|m| {
            panic!("chunk {chunk}: B at {l1}+{ext} (cut {}, residue {}) cached vs uncached: {m}", end_cut(l1), end_cut(l1) % 4)
        });
        same(&ob, &sb, false).unwrap_or_else(|m| panic!("chunk {chunk}: B at {l1}+{ext} cached vs serial: {m}"));
        assert_eq!(ob.cached_tokens, end_cut(l1), "chunk {chunk}: B at {l1}+{ext} hit {} tokens", ob.cached_tokens);
        assert_eq!(after.hits, before.hits + 1);
        assert_eq!(after.tokens_saved, before.tokens_saved + end_cut(l1) as u64);
        balanced(e).unwrap_or_else(|m| panic!("chunk {chunk}: after {l1}+{ext}: {m}"));
        eprintln!(
            "chunk {chunk}: {l1}+{ext} T{t}: hit {} of {} tokens, cached == uncached == serial, counters equal ({}/{}/{})",
            ob.cached_tokens,
            l1 + ext,
            ob.drafted,
            ob.accepted,
            ob.rounds
        );
        cases += 1;
    }
    assert_eq!(cases, 8);
}

/// 2. Four sharers of one system prompt, each with its own question and seed, together with a stranger.
fn sharers(e: &mut Eng) {
    let sys = toks(41, 1203);
    // two requests leave the shared entry (the second one discovers the shared blocks)
    for q in 0..2u64 {
        let mut p = sys.clone();
        p.extend(toks(50 + q, 33));
        one(e, request(p, 8, q, 0.0));
    }
    let mut reqs = Vec::new();
    for q in 0..4u64 {
        let mut p = sys.clone();
        p.extend(toks(60 + q, 21 + 7 * q as usize));
        reqs.push(request(p, 40, 100 + q, if q % 2 == 0 { 0.0 } else { 0.7 }));
    }
    let mut ids = Vec::new();
    for r in &reqs {
        ids.push(e.submit_deferred(r.clone()).unwrap());
    }
    while e.busy() {
        e.step().unwrap();
    }
    assert!(e.take_failed().is_empty());
    let outs = e.take_outputs();
    for (id, r) in ids.iter().zip(&reqs) {
        let o = outs.iter().find(|o| o.id == *id).unwrap();
        assert!(o.cached_tokens >= 1024, "a sharer of the system prompt resumes from its shared blocks: {}", o.cached_tokens);
        // beside others the counters depend on the rows each stream was handed; the reply is what must match
        let (u, _) = references(e, r);
        same(o, &u, false).unwrap_or_else(|m| panic!("sharer {}: {m}", r.sampling.seed));
    }
    balanced(e).unwrap();
}

/// 3. Opt-out, a differing prefix, short prompts.
fn edges(e: &mut Eng) {
    let base = toks(81, 900);
    one(e, request(base.clone(), 8, 1, 0.0));
    let before = e.prefix_stats();
    let r = Request { cache: false, ..request(base.clone(), 20, 2, 0.0) };
    let o = one(e, r);
    let after = e.prefix_stats();
    assert_eq!(o.cached_tokens, 0, "an opted-out request does not hit what is stored");
    assert_eq!(
        (after.lookups, after.entries, after.inserts),
        (before.lookups, before.entries, before.inserts),
        "and looks nothing up and leaves nothing"
    );
    // a differing prefix: the same length, one token different early on
    let mut other = base.clone();
    other[5] = other[5] % 149_000 + 7;
    let r = request(other, 20, 2, 0.0);
    let o = one(e, r.clone());
    assert_eq!(o.cached_tokens, 0);
    let (u, _) = references(e, &r);
    same(&o, &u, true).unwrap();
    // a prompt of 128 tokens or fewer is cached like any other: no path rule
    for (l1, ext) in [(150usize, 30usize), (200, 10)] {
        let a = toks(90 + l1 as u64, l1);
        let mut b = a.clone();
        b.extend(toks(91, ext));
        one(e, request(a.clone(), 8, 1, 0.0));
        let rb = request(b, 24, 5, 0.0);
        let ob = one(e, rb.clone());
        assert_eq!(ob.cached_tokens, end_cut(l1), "{l1}+{ext}");
        let (u, _) = references(e, &rb);
        same(&ob, &u, true).unwrap_or_else(|m| panic!("{l1}+{ext}: {m}"));
    }
    balanced(e).unwrap();
}

fn features(seed: u64, tokens: usize) -> Arc<GlmImageFeatures> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let f: Vec<f32> = (0..tokens * 4096)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / 16_777_216.0 - 0.5) * 0.1
        })
        .collect();
    let mut buf = gb_cuda::DevBuf::zeroed(tokens * 4096 * 4).unwrap();
    buf.upload(&f).unwrap();
    Arc::new(GlmImageFeatures { buf, tokens, merged: (tokens / 16, 16), key: 0, permit: None })
}

/// A prompt of `pre` text tokens, an image of 368 placeholder tokens, `post` text tokens; the image's content key is `key`.
fn image_request(
    image_token: u32,
    pre: usize,
    post_seed: u64,
    post: usize,
    key: u64,
    feats: &Arc<GlmImageFeatures>,
    max_new: usize,
) -> Request {
    let mut prompt = toks(7, pre);
    prompt.extend(std::iter::repeat_n(image_token, feats.tokens));
    prompt.extend(toks(post_seed, post));
    let span = ImageSpan { start: pre, tokens: feats.tokens, merged: feats.merged, features: feats.clone(), key };
    Request { images: vec![span], ..request(prompt, max_new, 4, 0.0) }
}

/// 4. Images.
fn images(e: &mut Eng) {
    let (f1, f2) = (features(1, 368), features(2, 368));
    let tok = e.backend.m.cfg.vision.as_ref().expect("the pack has a vision config").image_token_id;
    // a first prompt with image 1 leaves its end cut after the image; a longer one with the same image and text resumes from it
    let a = image_request(tok, 200, 31, 60, 0x1111, &f1, 24);
    let oa = one(e, a.clone());
    let (ua, _) = references(e, &a);
    same(&oa, &ua, true).unwrap_or_else(|m| panic!("image A: {m}"));
    let b = image_request(tok, 200, 31, 200, 0x1111, &f1, 40);
    // B's text after the image extends A's: the same first 60 tokens
    assert_eq!(b.prompt[..200 + 368 + 60], a.prompt[..200 + 368 + 60]);
    let ob = one(e, b.clone());
    let want = end_cut(a.prompt.len());
    assert_eq!(ob.cached_tokens, want, "the image prefix is found by the image's content key");
    assert!(want > 200 + 368, "the cut is after the image");
    let (ub, sb) = references(e, &b);
    same(&ob, &ub, true).unwrap_or_else(|m| panic!("image B cached vs uncached: {m}"));
    same(&ob, &sb, false).unwrap_or_else(|m| panic!("image B cached vs serial: {m}"));
    // the same text around ANOTHER image: nothing past the text before it may be reused, and the reply is its own
    let c = image_request(tok, 200, 31, 200, 0x2222, &f2, 40);
    let oc = one(e, c.clone());
    assert!(oc.cached_tokens < 200 + 368, "another image is another prefix: hit {}", oc.cached_tokens);
    let (uc, _) = references(e, &c);
    same(&oc, &uc, true).unwrap_or_else(|m| panic!("image C: {m}"));
    assert_ne!(oc.tokens, ob.tokens, "the two images are not the same prompt (the features steer the reply)");
    // an image with no key (0) is never cached: no lookup, no entry
    let before = e.prefix_stats();
    let d = image_request(tok, 200, 31, 200, 0, &f1, 24);
    let od = one(e, d.clone());
    let after = e.prefix_stats();
    assert_eq!((od.cached_tokens, after.lookups, after.inserts), (0, before.lookups, before.inserts));
    balanced(e).unwrap();
}

/// 5. More prompts than snapshot slots, then everything accounted for.
fn pressure(e: &mut Eng) {
    let before = e.prefix_stats();
    for i in 0..14u64 {
        let p = toks(300 + i, 400 + 37 * i as usize);
        let r = request(p, 6, i, 0.0);
        let o = one(e, r.clone());
        if i % 4 == 0 {
            let (u, _) = references(e, &r);
            same(&o, &u, true).unwrap();
        }
        balanced(e).unwrap_or_else(|m| panic!("after prompt {i}: {m}"));
    }
    let after = e.prefix_stats();
    assert!(after.evictions > before.evictions, "14 prompts into {SNAPSHOTS} snapshot slots must evict");
    assert!(after.entries <= SNAPSHOTS as u64 && after.pages_held <= CACHE_PAGES as u64);
}

#[test]
#[ignore]
fn prefix_cache_through_the_engine() {
    if std::env::var("REXL3_GLM_CHUNK_ROWS").is_err() {
        std::env::set_var("REXL3_GLM_CHUNK_ROWS", "1024");
    }
    let Some(model) = load_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let mut e = engine(model, 1024);
    balanced(&e).unwrap();
    sweep(&mut e, 1024);
    // the same prefix in another chunk size: the slice grid is the backend's and no bit depends on it
    e.backend.set_chunk(256);
    sweep(&mut e, 256);
    sharers(&mut e);
    edges(&mut e);
    images(&mut e);
    pressure(&mut e);
    let st = e.prefix_stats();
    eprintln!(
        "prefix cache: lookups {} hits {} tokens saved {} inserts {} evictions {} (for pages {}, for slots {}) discarded {} restore failures {}",
        st.lookups, st.hits, st.tokens_saved, st.inserts, st.evictions, st.evicted_for_pages, st.evicted_for_slots, st.captures_discarded, st.restore_failures
    );
    assert_eq!(st.restore_failures, 0);
}
