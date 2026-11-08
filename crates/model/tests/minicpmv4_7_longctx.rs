//! Long context on the full pack: the E6b scorecard (`docs/MINICPMV4_7_INTEGRATION_PLAN.md`, section 14).
//!
//! `REXL3_MINICPM_DIR` is the 40-layer pack, `REXL3_MINICPM_NIAH_DIR` a directory `tools/minicpm_v47/make_niah.py` wrote
//! (`prefix.u32`, `niah.json`: one document of `max(cuts) + 1` tokens with needles spliced in at fixed positions, and a question for
//! every (cut, needle) pair whose needle is inside the cut's prefix). Both tests are `#[ignore]`: they boot a 27 GB pack and run for
//! minutes. Hold the GPU lock exclusively (`flock -x`): a full-pack boot costs 28 GiB of `MemAvailable`, and the timings are only
//! numbers on an idle GPU.
//!
//! * `niah_scorecard`: ONE pass of the document through the model with a snapshot at every cut (the prefill time at each length is
//!   the cumulative time at the cut: the prefix of a longer prompt is the shorter prompt, the arithmetic path depends on the total
//!   only), then for every question the snapshot of its cut is restored into a fresh slot, the question prefilled (the suffix),
//!   the answer generated greedily and compared with the needle's value token for token; then 48 more decode steps at that depth.
//!   One cut (`REXL3_MINICPM_COLD_CUT`, 32,000 by default) is also prefilled cold in one piece and its candidates compared bit for
//!   bit with the cached run's. Reports the memory the box had before the boot, after it and after the prefill.
//! * `a_long_prompt_through_the_engine`: the engine's own policy at 32,000 tokens: a cold question leaves its end cut, a second
//!   question about the same document is cold and leaves the shared cut, a repeat hits the first's end cut, and a third phrasing
//!   hits the shared cut; every cached reply equals the uncached reply.
//!
//! ```text
//! flock -x -w 3600 /tmp/rexl3-gpu.lock env REXL3_MINICPM_DIR=~/models/minicpm-v-4.7-35b-a3b/packs/q3-6.00bpw \
//!     REXL3_MINICPM_NIAH_DIR=DIR cargo test --release -p gb-model --test minicpmv4_7_longctx -- --ignored --nocapture niah
//! ```

mod minicpmv4_7_support;

use gb_engine::{Backend, RowOutput, RowRequest};
use gb_model::families::minicpmv4_7::backend::MiniCpmBackend;
use minicpmv4_7_support::*;
use std::path::PathBuf;
use std::time::Instant;

struct Niah {
    ids: Vec<u32>,
    cuts: Vec<usize>,
    eos: Vec<u32>,
    questions: Vec<Question>,
    /// The digits of every needle's passcode: a wrong answer that is another needle's is a retrieval slip of the model, a wrong
    /// answer that is no passcode is a corrupted state.
    values: Vec<Vec<u32>>,
}

struct Question {
    cut: usize,
    needle: usize,
    depth: f64,
    suffix: Vec<u32>,
    alt_suffix: Vec<u32>,
    answer: Vec<u32>,
}

fn load_niah(max_cut: usize) -> Option<Niah> {
    let Some(dir) = std::env::var_os("REXL3_MINICPM_NIAH_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_MINICPM_NIAH_DIR is not set (tools/minicpm_v47/make_niah.py writes it)");
        return None;
    };
    let raw = std::fs::read(dir.join("prefix.u32")).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let ids: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("niah.json")).unwrap()).unwrap();
    let u32s = |v: &serde_json::Value| -> Vec<u32> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() };
    let cuts: Vec<usize> =
        meta["cuts"].as_array().unwrap().iter().map(|c| c.as_u64().unwrap() as usize).filter(|&c| c <= max_cut).collect();
    assert!(!cuts.is_empty(), "no cut at or below {max_cut}");
    let questions = meta["questions"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|q| (q["cut"].as_u64().unwrap() as usize) <= max_cut)
        .map(|q| Question {
            cut: q["cut"].as_u64().unwrap() as usize,
            needle: q["needle"].as_u64().unwrap() as usize,
            depth: q["depth"].as_f64().unwrap(),
            suffix: u32s(&q["suffix_ids"]),
            alt_suffix: u32s(&q["suffix_alt_ids"]),
            answer: u32s(&q["answer_ids"]),
        })
        .collect();
    let values = meta["needles"].as_array().unwrap().iter().map(|v| u32s(&v["value_ids"])).collect();
    Some(Niah { ids, cuts, eos: u32s(&meta["eos_ids"]), questions, values })
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn contains(hay: &[u32], needle: &[u32]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

/// Prefill `prompt` into `slot` from `from`, slice by slice; the final row, and the seconds the slices took.
fn prefill(b: &mut MiniCpmBackend, slot: usize, prompt: &[u32], from: usize) -> (RowOutput, f64) {
    let t = Instant::now();
    let mut done = from;
    loop {
        let step = b.prefill_slice(slot, prompt, &[], done, 8);
        done = step.done;
        if let Some(row) = step.row {
            assert!(b.take_error().is_none(), "the prefill failed");
            return (row, t.elapsed().as_secs_f64());
        }
    }
}

/// `n` greedy steps of one row after the prompt of `slot` (`pos` is the next position), feeding the argmax each time and ignoring
/// the end of the answer: the per-step milliseconds as the engine's window path takes them (graph replayed, candidates read back).
fn decode_steps(b: &mut MiniCpmBackend, slot: usize, mut pos: usize, mut next: u32, n: usize) -> Vec<f64> {
    let mut ms = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        let out = b.step(&[RowRequest { slot, first_position: pos as u64, tokens: vec![next] }], 4);
        b.commit(slot, 1);
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        assert!(b.take_error().is_none(), "a decode step failed at position {pos}");
        next = out[0][0].ids[0];
        pos += 1;
    }
    ms
}

#[test]
#[ignore]
fn niah_scorecard() {
    let max_cut = env_usize("REXL3_MINICPM_NIAH_MAX", usize::MAX);
    let Some(n) = load_niah(max_cut) else { return };
    let top = *n.cuts.iter().max().unwrap();
    assert!(n.ids.len() > top, "the document has {} tokens, the top cut is {top}", n.ids.len());
    let max_ctx = env_usize("REXL3_MINICPM_LONG_CTX", 262_144);
    let slots = env_usize("REXL3_MINICPM_LONG_SLOTS", 4);
    let chunk = env_usize("REXL3_MINICPM_LONG_CHUNK", 4096);
    let cold_cut = env_usize("REXL3_MINICPM_COLD_CUT", 32_000);
    let decode_n = env_usize("REXL3_MINICPM_DECODE_STEPS", 48);
    let max_new = 24;
    // one snapshot slot a cut and one to spare (an entry unit is a snapshot and the planner's eight pages: 101.4 MiB)
    let units = n.cuts.len() as u64 + 1;
    let Some(mut rig) = Rig::load(slots, max_ctx, chunk, units) else { return };
    assert!(rig.plan.memory.cache.snapshot_slots >= n.cuts.len(), "{:?}", rig.plan.memory.cache);
    let (mem0, mem1) = (rig.mem_before.unwrap_or(0), rig.mem_after.unwrap_or(0));
    eprintln!(
        "NIAH boot: {} slots, pool {} pages ({:.2} GiB of KV at {} bytes a token), weights {:.2} GiB, snapshot arena {:.2} GiB, \
         MemAvailable {:.1} -> {:.1} GiB",
        rig.report.slots,
        rig.report.kv_pages,
        gib(rig.report.state.kv_target),
        rig.cfg.full_layers() * 2 * rig.cfg.kv_heads * rig.cfg.head_dim * 2,
        gib(rig.report.weights_planned),
        gib(rig.report.state.prefix_snapshots),
        gib(mem0),
        gib(mem1)
    );
    let b = &mut rig.eng.backend;

    // ---- 1. the document once, a snapshot at every cut
    let prompt = &n.ids[..top + 1];
    let mut done = 0;
    let mut taken: Vec<(usize, gb_engine::PrefixHandle)> = Vec::new();
    let mut curve: Vec<(usize, f64)> = Vec::new();
    let (mut cum, t_prefill) = (0.0f64, Instant::now());
    let mut cum_at_cut = std::collections::BTreeMap::new();
    while done < prompt.len() {
        let t = Instant::now();
        let s = b.prefill_slice_cuts(0, prompt, &[], done, 8, &n.cuts);
        let dt = t.elapsed().as_secs_f64();
        assert!(b.take_error().is_none(), "the prefill failed at {done}");
        let rows = s.step.done - done;
        cum += dt;
        done = s.step.done;
        curve.push((done, dt * 1e3 / rows as f64));
        for c in s.captured {
            eprintln!("NIAH snapshot at {} after {cum:.1} s", c.len);
            cum_at_cut.insert(c.len, cum);
            taken.push((c.len, c.handle));
        }
        eprintln!(
            "NIAH prefill {done}/{} : {:.0} ms for {rows} rows ({:.0} tok/s), {cum:.1} s",
            prompt.len(),
            dt * 1e3,
            rows as f64 / dt
        );
    }
    let prefill_s = t_prefill.elapsed().as_secs_f64();
    let mem2 = gb_model::serve::mem_available().unwrap_or(0);
    assert_eq!(taken.len(), n.cuts.len(), "every cut was taken");
    eprintln!(
        "NIAH prefill of {} tokens: {prefill_s:.1} s ({:.0} tok/s); MemAvailable after {:.1} GiB (fell {:.1} GiB since the boot)",
        prompt.len(),
        prompt.len() as f64 / prefill_s,
        gib(mem2),
        gib(mem1.saturating_sub(mem2))
    );

    // ---- 2. a question at every (cut, needle): restore, prefill the suffix, answer, then decode at the depth
    let mut failures = Vec::new();
    let mut alt_failures: Vec<(usize, usize, f64)> = Vec::new();
    let mut decode_at: std::collections::BTreeMap<usize, f64> = Default::default();
    let mut cached_rows: std::collections::BTreeMap<(usize, usize), RowOutput> = Default::default();
    for (qi, q) in n.questions.iter().enumerate() {
        let handle = &taken.iter().find(|t| t.0 == q.cut).expect("a snapshot at the question's cut").1;
        let question = [&n.ids[..q.cut], &q.suffix[..]].concat();
        let t = Instant::now();
        b.prefix_restore(1, handle, q.cut).unwrap_or_else(|e| panic!("restore at {}: {e}", q.cut));
        b.m.stream.sync().unwrap();
        let restore_ms = t.elapsed().as_secs_f64() * 1e3;
        let (row, suffix_s) = prefill(b, 1, &question, q.cut);
        let ttft_cached = restore_ms / 1e3 + suffix_s;
        // the answer
        let mut gen = vec![row.ids[0]];
        let mut pos = question.len();
        let mut next = row.ids[0];
        while gen.len() < max_new && !n.eos.contains(&next) {
            let out = b.step(&[RowRequest { slot: 1, first_position: pos as u64, tokens: vec![next] }], 4);
            b.commit(1, 1);
            assert!(b.take_error().is_none());
            next = out[0][0].ids[0];
            gen.push(next);
            pos += 1;
        }
        let ok = contains(&gen, &q.answer);
        if !ok {
            failures.push((q.cut, q.needle, q.depth, gen.clone()));
        }
        eprintln!(
            "NIAH question cut={} needle={} depth={:.3} restore={restore_ms:.1} ms suffix={:.0} ms ({} tokens) cached_ttft={ttft_cached:.2} s \
             cold_ttft~{:.1} s answer={} generated={:?}",
            q.cut,
            q.needle,
            q.depth,
            suffix_s * 1e3,
            q.suffix.len(),
            cum_at_cut[&q.cut] + suffix_s,
            if ok { "EXACT" } else { "WRONG" },
            gen
        );
        if let std::collections::btree_map::Entry::Vacant(v) = cached_rows.entry((q.cut, 0)) {
            v.insert(row.clone());
            // decode at this depth, once per cut: the answer's slot continues (past its end: the speed is the point)
            if let std::collections::btree_map::Entry::Vacant(v) = decode_at.entry(q.cut) {
                let mut ms = decode_steps(b, 1, pos, next, decode_n);
                let warm = ms.len().min(8);
                let med = median(&mut ms[warm..]);
                v.insert(med);
                eprintln!("NIAH decode cut={} at depth {}: median {med:.2} ms a step ({:.1} tok/s)", q.cut, pos, 1e3 / med);
            }
        }
        b.release(1);
        // the same question in other words, from the same snapshot (0.3 s): a second sample of the retrieval, so that a wrong answer
        // can be told from a prompt the model is brittle on
        let alt_question = [&n.ids[..q.cut], &q.alt_suffix[..]].concat();
        b.prefix_restore(1, handle, q.cut).unwrap_or_else(|e| panic!("restore at {}: {e}", q.cut));
        let (alt_row, _) = prefill(b, 1, &alt_question, q.cut);
        let mut alt = vec![alt_row.ids[0]];
        let (mut apos, mut anext) = (alt_question.len(), alt_row.ids[0]);
        while alt.len() < max_new && !n.eos.contains(&anext) {
            let out = b.step(&[RowRequest { slot: 1, first_position: apos as u64, tokens: vec![anext] }], 4);
            b.commit(1, 1);
            assert!(b.take_error().is_none());
            anext = out[0][0].ids[0];
            alt.push(anext);
            apos += 1;
        }
        let alt_ok = contains(&alt, &q.answer);
        if !alt_ok {
            alt_failures.push((q.cut, q.needle, q.depth));
        }
        eprintln!(
            "NIAH alt-question cut={} needle={} depth={:.3} answer={} (first phrasing {}) generated={:?}",
            q.cut,
            q.needle,
            q.depth,
            if alt_ok { "EXACT" } else { "WRONG" },
            if ok { "EXACT" } else { "WRONG" },
            alt
        );
        b.release(1);
        let _ = qi;
    }

    // ---- 3. cold == cached, bit for bit, at one cut (a fresh prefill of the whole question prompt in another slot). The document's
    // slot and its snapshots hold the pool's pages (the top cut's entry is a whole context): give them back first, the cold prefill
    // needs a pool of its own.
    for (_, h) in std::mem::take(&mut taken) {
        b.prefix_release(h);
    }
    b.release(0);
    let mut cold_report = None;
    if let Some(q) = n.questions.iter().find(|q| q.cut == cold_cut) {
        let question = [&n.ids[..q.cut], &q.suffix[..]].concat();
        let (row, secs) = prefill(b, 2, &question, 0);
        // the cached run of the same question is the first question of the cut
        let first = n.questions.iter().find(|x| x.cut == cold_cut).unwrap();
        assert_eq!(first.needle, q.needle);
        let cached = cached_rows.get(&(cold_cut, 0)).expect("the cached row of the cut");
        assert!(row.bits_eq(cached), "cold and cached candidates differ at {cold_cut}: {:?} against {:?}", row.ids, cached.ids);
        eprintln!("NIAH cold == cached at {cold_cut}: bit-equal candidates; the cold prefill took {secs:.1} s");
        cold_report = Some((cold_cut, secs));
        b.release(2);
    } else {
        eprintln!("NIAH no cold comparison: no question at cut {cold_cut}");
    }

    // ---- 4. the table
    eprintln!("NIAH TABLE");
    eprintln!("| context | prefill s | tok/s (cumulative) | chunk ms/row at the cut | KV GiB | decode ms/step | decode tok/s |");
    for &c in &n.cuts {
        let tail = curve.iter().rfind(|(d, _)| *d <= c).map(|x| x.1).unwrap_or(0.0);
        let med = decode_at.get(&c).copied().unwrap_or(f64::NAN);
        eprintln!(
            "| {c} | {:.1} | {:.0} | {tail:.2} | {:.2} | {med:.2} | {:.1} |",
            cum_at_cut[&c],
            c as f64 / cum_at_cut[&c],
            gib(c as u64 * rig.plan.memory.block_bytes / 256),
            1e3 / med
        );
    }
    eprintln!("NIAH answers: {} of {} exact", n.questions.len() - failures.len(), n.questions.len());
    eprintln!("NIAH alt-phrasing answers: {} of {} exact", n.questions.len() - alt_failures.len(), n.questions.len());
    for (c, k, d, g) in &failures {
        eprintln!("NIAH WRONG cut={c} needle={k} depth={d:.3}: {g:?}");
    }
    if let Some((c, s)) = cold_report {
        eprintln!("NIAH cold prefill of one question at {c}: {s:.1} s");
    }
    // what the gate holds: a wrong answer is the passcode of ANOTHER needle (the model slipped between two codewords of one long
    // document; a corrupted state would produce digits that belong to no needle), and at least 90 % of the questions are exact
    // (REXL3_MINICPM_NIAH_MIN_EXACT overrides). A truncated pack (the 8-layer smoke pack exercises the harness, not the model)
    // answers nothing: `REXL3_MINICPM_NIAH_ALLOW_WRONG=1` skips the assertions.
    let mut garbled = Vec::new();
    for (c, k, d, g) in &failures {
        let other = n.values.iter().enumerate().find(|(i, v)| *i != *k && contains(g, v)).map(|(i, _)| i);
        eprintln!(
            "NIAH WRONG cut={c} needle={k} depth={d:.3} is {}",
            match other {
                Some(i) => format!("the passcode of needle {i}"),
                None => "no needle's passcode".to_string(),
            }
        );
        if other.is_none() {
            garbled.push((*c, *k));
        }
    }
    if std::env::var("REXL3_MINICPM_NIAH_ALLOW_WRONG").is_ok_and(|v| v == "1") {
        return;
    }
    let min = std::env::var("REXL3_MINICPM_NIAH_MIN_EXACT").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.9);
    let exact = 1.0 - failures.len() as f64 / n.questions.len() as f64;
    assert!(garbled.is_empty(), "answers that are no needle's passcode (a corrupted state?): {garbled:?}");
    assert!(
        exact >= min,
        "{} of {} needles were retrieved exactly, the gate is {min}",
        n.questions.len() - failures.len(),
        n.questions.len()
    );
}

#[test]
#[ignore]
fn a_long_prompt_through_the_engine() {
    let cut = env_usize("REXL3_MINICPM_ENGINE_CUT", 32_000);
    let Some(n) = load_niah(cut) else { return };
    let qs: Vec<&Question> = n.questions.iter().filter(|q| q.cut == cut).collect();
    assert!(qs.len() >= 2, "two needles are inside the cut {cut}");
    // the pool holds one prompt and the cache the same again (pages of an entry are not the admitted share's): eight pages a unit
    let pages = cut.div_ceil(256) + 2;
    // two entries of a prompt this long (the first question's end cut and the shared cut, from separate prefills: no page in common)
    let units = (2 * pages as u64 + 16).div_ceil(8);
    let ctx = (pages + 8) * 256;
    let Some(mut rig) = Rig::load_with(2, ctx, 4096, units, gb_engine::PrefixConfig::default()) else { return };
    let e = &mut rig.eng;
    let doc = &n.ids[..cut];
    let prompt_of = |q: &Question, alt: bool| [doc, if alt { &q.alt_suffix[..] } else { &q.suffix[..] }].concat();
    let (q1, q2) = (qs[0], qs[1]);
    let t = |e: &mut Eng, r| {
        let t0 = Instant::now();
        let o = one(e, r);
        (o, t0.elapsed().as_secs_f64())
    };
    let (a, ta) = t(e, request(&prompt_of(q1, false), greedy(), 16, false, true));
    eprintln!("ENGINE Q1 cold: {ta:.1} s cached_tokens={} entries={}", a.cached_tokens, e.prefix_stats().entries);
    assert_eq!(a.cached_tokens, 0);
    let (b2, tb) = t(e, request(&prompt_of(q2, false), greedy(), 16, false, true));
    eprintln!(
        "ENGINE Q2 (same document, other question): {tb:.1} s cached_tokens={} entries={}",
        b2.cached_tokens,
        e.prefix_stats().entries
    );
    assert_eq!(b2.cached_tokens, 0, "the shared blocks are discovered on the second sighting: this one is cold");
    let (c, tc) = t(e, request(&prompt_of(q1, false), greedy(), 16, false, true));
    eprintln!("ENGINE Q1 again: {tc:.2} s cached_tokens={}", c.cached_tokens);
    assert_eq!(
        c.cached_tokens,
        prompt_of(q1, false).len() - 16,
        "the repeat hits its own end cut (the prompt's end less the tail)"
    );
    same("a repeat of the first question", &c, &a);
    let (d, td) = t(e, request(&prompt_of(q1, true), greedy(), 16, false, true));
    eprintln!("ENGINE Q1 rephrased: {td:.2} s cached_tokens={}", d.cached_tokens);
    assert!(d.cached_tokens >= cut - 256, "the rephrased question hits the shared cut inside the document: {}", d.cached_tokens);
    let (want, tw) = t(e, request(&prompt_of(q1, true), greedy(), 16, false, false));
    eprintln!("ENGINE Q1 rephrased, uncached: {tw:.1} s");
    same("a hit inside the document", &d, &want);
    eprintln!("ENGINE speedup of the hit: {:.1}x ({tw:.1} s -> {td:.2} s); stats {:?}", tw / td, e.prefix_stats());
    // the truncated smoke pack answers nothing (`REXL3_MINICPM_NIAH_ALLOW_WRONG=1`); the full pack must name the needle's value
    if std::env::var("REXL3_MINICPM_NIAH_ALLOW_WRONG").is_ok_and(|v| v == "1") {
        return;
    }
    assert!(contains(&d.tokens, &q1.answer), "the answer is the needle's value: {:?}", d.tokens);
}

/// The engine's side of the retrieval oracle check (`tools/minicpm_v47/niah_fork_check.py` is the exllamav3 fork's): the logits of
/// the answer position (the row that predicts the first token of the reply) of chosen questions, and the greedy reply, written to
/// `REXL3_MINICPM_ANSWER_DIR` as `engine_<cut>_<needle>.f16` (the head's own dtype, `[vocab]`) and `engine_<cut>_<needle>.json`.
/// `REXL3_MINICPM_ANSWER_QUESTIONS` is `cut:needle,...`, e.g. `200000:3,261000:4,261000:5`. The same prompts, token for token, the
/// fork gets.
#[test]
#[ignore]
fn niah_answer_logits() {
    let Ok(spec) = std::env::var("REXL3_MINICPM_ANSWER_QUESTIONS") else {
        return eprintln!("not run: REXL3_MINICPM_ANSWER_QUESTIONS is not set");
    };
    let dir = PathBuf::from(std::env::var("REXL3_MINICPM_ANSWER_DIR").expect("REXL3_MINICPM_ANSWER_DIR"));
    std::fs::create_dir_all(&dir).unwrap();
    let wanted: Vec<(usize, usize)> = spec
        .split(',')
        .map(|p| {
            let (c, k) = p.split_once(':').expect("cut:needle");
            (c.trim().parse().unwrap(), k.trim().parse().unwrap())
        })
        .collect();
    let top = wanted.iter().map(|w| w.0).max().unwrap();
    let Some(n) = load_niah(top) else { return };
    let mut cuts: Vec<usize> = wanted.iter().map(|w| w.0).collect();
    cuts.sort_unstable();
    cuts.dedup();
    let max_ctx = (top + 4096).div_ceil(256) * 256;
    let Some(mut rig) = Rig::load(2, max_ctx, 4096, cuts.len() as u64 + 1) else { return };
    let vocab = rig.cfg.vocab;
    let b = &mut rig.eng.backend;
    let prompt = &n.ids[..top + 1];
    let mut taken: Vec<(usize, gb_engine::PrefixHandle)> = Vec::new();
    let mut done = 0;
    while done < prompt.len() {
        let s = b.prefill_slice_cuts(0, prompt, &[], done, 8, &cuts);
        assert!(b.take_error().is_none(), "the prefill failed at {done}");
        done = s.step.done;
        taken.extend(s.captured.into_iter().map(|c| (c.len, c.handle)));
        eprintln!("ANSWER prefill {done}/{}", prompt.len());
    }
    assert_eq!(taken.len(), cuts.len());
    for (cut, needle) in wanted {
        let q = n.questions.iter().find(|q| q.cut == cut && q.needle == needle).expect("a question for (cut, needle)");
        let handle = &taken.iter().find(|t| t.0 == cut).unwrap().1;
        let question = [&n.ids[..cut], &q.suffix[..]].concat();
        b.prefix_restore(1, handle, cut).unwrap();
        let (row, _) = prefill(b, 1, &question, cut);
        // the row that predicts the first token of the reply is the last row of the suffix chunk
        let mut logits = vec![0u16; vocab];
        b.m.last_rows_logits(q.suffix.len() - 1, 1, &mut logits).unwrap();
        let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("engine_{cut}_{needle}.f16")), bytes).unwrap();
        let f16 = b.m.logits_are_f16();
        let mut gen = vec![row.ids[0]];
        let (mut pos, mut next) = (question.len(), row.ids[0]);
        while gen.len() < 12 && !n.eos.contains(&next) {
            let out = b.step(&[RowRequest { slot: 1, first_position: pos as u64, tokens: vec![next] }], 4);
            b.commit(1, 1);
            assert!(b.take_error().is_none());
            next = out[0][0].ids[0];
            gen.push(next);
            pos += 1;
        }
        let meta = serde_json::json!({
            "cut": cut, "needle": needle, "depth": q.depth, "prompt_tokens": question.len(), "dtype": if f16 { "f16" } else { "bf16" },
            "generated": gen, "answer_ids": q.answer, "exact": contains(&gen, &q.answer),
            "top8_ids": row.ids, "top8_values": row.values,
        });
        std::fs::write(dir.join(format!("engine_{cut}_{needle}.json")), meta.to_string()).unwrap();
        eprintln!("ANSWER cut={cut} needle={needle} depth={:.3} exact={} generated={gen:?}", q.depth, contains(&gen, &q.answer));
        b.release(1);
    }
    for (_, h) in taken {
        b.prefix_release(h);
    }
    b.release(0);
}
