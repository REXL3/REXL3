//! The ceiling of keyed-noise drafting for GLM, measured offline (docs/GLM53.md "Keyed-noise drafting"), the way docs/TREE_DRAFTING.md section 7 did it for the
//! Qwen tree: log what a policy needs once, replay any policy against it.
//!
//! A draft round accepts depth `d` exactly when the head's pick at that depth equals the target's sample there. The sample IS the committed text (the
//! engine's keyed sampler is a pure function of the target's logits and (seed, position, id)), so no target logits are needed to score a draft: the serial
//! text is the truth. A draft chain reaches depth `d` only when the `d - 1` drafts before it were the truth, and then the chain has been fed exactly the
//! truth (the same tokens, the same hidden of its own previous step), so the head's candidates at every depth of every anchor are a function of the
//! committed text alone. One log serves every policy:
//!
//! * pass 1 (`dump_candidates`): serial generation with the engine's keyed sampler (the truth), logging the target's top-64 candidates at each position;
//! * pass 2: the same text again, one row a step, and at EVERY anchor the NextN chain fed the true continuation (`GlmModel::mtp_debug_chain`), logging the
//!   head's top-64 candidates at depths 0..3. The head is the FULL head; the restricted head (the serving default) is its list filtered to the set, exactly
//!   (kept columns are bit-identical), so both are replayed from this one log.
//!
//! `replay_ceiling` (CPU) replays rounds as the engine runs them (a policy's next round starts where its own last one ended) for the argmax drafter and
//! the keyed drafter (the target's full sampler over the head's candidates), at K = 1..4 and for the full and the restricted head, and the expected
//! agreement of the two picks over many seeds (which has no sampling noise of its own).
//!
//!     REXL3_GLM_DIR=... REXL3_NOISE_PROMPTS=prompts.json REXL3_NOISE_OUT=dump cargo test --release -p gb-model --test glm_noise_ceiling dump_candidates -- --ignored --nocapture
//!     REXL3_NOISE_DUMP=dump cargo test --release -p gb-model --test glm_noise_ceiling replay_ceiling -- --nocapture
//!
//! `prompts.json` is `{"class": [[token ids], ...]}` (`tools/glm_noise_prompts.py`). Env-gated: without the pack or the dump a test prints "not run".

mod glm_support;

use gb_engine::RowRequest;
use gb_model::families::glm::model::{vocab_ranges, GlmModel};
use gb_model::rows::RowBatch;
use gb_sampling::{pairwise_sum, try_choose, uniform, Sampling};
use glm_support::*;
use std::io::Write;
use std::path::{Path, PathBuf};

const SLOTS: usize = 2;
const KV_BLOCKS: usize = 64;
const MAX_CONTEXT: usize = 4096;
const COUNT: usize = 64;
const DEPTH: usize = 4;

fn modes() -> Vec<(String, f64)> {
    std::env::var("REXL3_NOISE_MODES")
        .unwrap_or_else(|_| "greedy:0,t0.7:0.7,t1.0:1.0".into())
        .split(',')
        .map(|m| {
            let (l, t) = m.split_once(':').expect("REXL3_NOISE_MODES is label:temperature,...");
            (l.to_string(), t.parse().unwrap())
        })
        .collect()
}

fn sampling_for(temperature: f64, seed: u64) -> Sampling {
    // the server's defaults: top_k 20, top_p 0.95
    Sampling { seed, temperature, top_k: 20, top_p: 0.95, min_p: 0.0 }
}

fn prefill(m: &mut GlmModel, slot: usize, prompt: &[u32], want: bool) -> Option<(Vec<i32>, Vec<f32>)> {
    m.begin_slot(slot).unwrap();
    let mut at = 0;
    let mut out = None;
    while at < prompt.len() {
        let end = (at + m.chunk_rows()).min(prompt.len());
        let w = (want && end == prompt.len()).then_some(COUNT);
        if let Some(c) = m.prefill_chunk(slot, at, &prompt[at..end], w).unwrap() {
            out = Some((c.ids[..COUNT].to_vec(), c.vals[..COUNT].to_vec()));
        }
        at = end;
    }
    out
}

fn step(m: &mut GlmModel, slot: usize, last: u32, first_position: usize, want: usize) -> (Vec<i32>, Vec<f32>) {
    let b = RowBatch::from_requests(&[RowRequest { slot, tokens: vec![last], first_position: first_position as u64 }]);
    let c = m.window_candidates(&b, want).unwrap();
    m.commit(slot, 1);
    (c.ids[..want].to_vec(), c.vals[..want].to_vec())
}

/// A row's top-64 candidates: ids and logits.
type Cands = (Vec<i32>, Vec<f32>);

/// Pass 1: the serial text under `s`, and the target's top-64 at each position (row `j`: the candidates that sampled `gen[j]`).
fn truth(m: &mut GlmModel, slot: usize, prompt: &[u32], s: &Sampling, n_new: usize, eos: &[u32]) -> (Vec<u32>, Vec<Cands>) {
    let cc = s.candidate_count(m.cfg.vocab);
    let mut cands = Vec::new();
    let mut gen: Vec<u32> = Vec::new();
    let pick = |c: &(Vec<i32>, Vec<f32>), pos: usize| {
        let ids: Vec<u32> = c.0[..cc].iter().map(|&i| i as u32).collect();
        try_choose(&c.1[..cc], &ids, pos as u64, s).unwrap()
    };
    let first = prefill(m, slot, prompt, true).unwrap();
    gen.push(pick(&first, prompt.len()));
    cands.push(first);
    while gen.len() < n_new && !eos.contains(gen.last().unwrap()) {
        let len = prompt.len() + gen.len();
        let c = step(m, slot, *gen.last().unwrap(), len - 1, COUNT);
        gen.push(pick(&c, len));
        cands.push(c);
    }
    m.end_slot(slot);
    (gen, cands)
}

/// Pass 2: the head's top-64 at depths `0..depth` of every anchor `j` (the sequence `prompt ++ gen[..=j]`), the chain fed `gen[j + 1 + k]`.
fn head_lists(m: &mut GlmModel, slot: usize, prompt: &[u32], gen: &[u32]) -> Vec<Vec<(Vec<u32>, Vec<f32>)>> {
    prefill(m, slot, prompt, false);
    let mut tokens = prompt.to_vec();
    tokens.push(gen[0]);
    let mut out = Vec::new();
    for j in 0..gen.len().saturating_sub(1) {
        let depth = DEPTH.min(gen.len() - 1 - j);
        let lists = m
            .mtp_debug_chain(slot, &tokens, depth, COUNT, false, true, &mut |k, _ids, _vals| gen[j + 1 + k])
            .unwrap()
            .expect("the slot drafts");
        out.push(lists);
        // advance the target one row: feed gen[j] at position prompt.len() + j
        step(m, slot, gen[j], prompt.len() + j, 1);
        tokens.push(gen[j + 1]);
    }
    m.end_slot(slot);
    out
}

fn w_u32(f: &mut std::fs::File, v: &[u32]) {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    f.write_all(&b).unwrap();
}
fn w_f32(f: &mut std::fs::File, v: &[f32]) {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    f.write_all(&b).unwrap();
}

/// One generation's log: `meta.txt`, `tokens.bin` (prompt then generated, u32), `target.bin` (per generated position: 64 ids u32, 64 vals f32) and
/// `head.bin` (per anchor: the depth count u32, then per depth 64 ids u32 and 64 vals f32).
fn write_dump(
    dir: &Path,
    class: &str,
    s: &Sampling,
    prompt: &[u32],
    gen: &[u32],
    target: &[(Vec<i32>, Vec<f32>)],
    head: &[Vec<(Vec<u32>, Vec<f32>)>],
) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("meta.txt"),
        format!(
            "class={class}\nprompt_len={}\ngenerated={}\nseed={}\ntemperature={}\ntop_k={}\ntop_p={}\nmin_p={}\n",
            prompt.len(),
            gen.len(),
            s.seed,
            s.temperature,
            s.top_k,
            s.top_p,
            s.min_p
        ),
    )
    .unwrap();
    let mut t = std::fs::File::create(dir.join("tokens.bin")).unwrap();
    w_u32(&mut t, prompt);
    w_u32(&mut t, gen);
    let mut f = std::fs::File::create(dir.join("target.bin")).unwrap();
    for (ids, vals) in target {
        w_u32(&mut f, &ids.iter().map(|&i| i as u32).collect::<Vec<_>>());
        w_f32(&mut f, vals);
    }
    let mut f = std::fs::File::create(dir.join("head.bin")).unwrap();
    for anchor in head {
        w_u32(&mut f, &[anchor.len() as u32]);
        for (ids, vals) in anchor {
            w_u32(&mut f, ids);
            w_f32(&mut f, vals);
        }
    }
}

#[test]
#[ignore]
fn dump_candidates() {
    let (Ok(prompts), Ok(out)) = (std::env::var("REXL3_NOISE_PROMPTS"), std::env::var("REXL3_NOISE_OUT")) else {
        eprintln!("REXL3_NOISE_PROMPTS and REXL3_NOISE_OUT are not set: not run");
        return;
    };
    let Some(mut m) = shared_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let n_new: usize = std::env::var("REXL3_NOISE_N").ok().and_then(|v| v.parse().ok()).unwrap_or(384);
    let per_class: usize = std::env::var("REXL3_NOISE_PER_CLASS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let eos = m.cfg.eos.clone();
    let classes: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(prompts).unwrap()).unwrap();
    let out = PathBuf::from(out);
    let t0 = std::time::Instant::now();
    for (class, list) in classes.as_object().unwrap() {
        for (pi, ids) in list.as_array().unwrap().iter().take(per_class).enumerate() {
            let prompt: Vec<u32> = ids.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
            for (label, temperature) in modes() {
                let s = sampling_for(temperature, 1000 + pi as u64 * 17 + class.len() as u64);
                let t = std::time::Instant::now();
                let (gen, target) = truth(&mut m, 0, &prompt, &s, n_new, &eos);
                let head = head_lists(&mut m, 0, &prompt, &gen);
                write_dump(&out.join(format!("{class}-{pi}-{label}")), class, &s, &prompt, &gen, &target, &head);
                eprintln!(
                    "{class}-{pi}-{label}: prompt {} tokens, {} generated, {} anchors, {:.1} s (total {:.0} s)",
                    prompt.len(),
                    gen.len(),
                    head.len(),
                    t.elapsed().as_secs_f64(),
                    t0.elapsed().as_secs_f64()
                );
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------ the replay

struct Gen {
    class: String,
    label: String,
    prompt_len: usize,
    s: Sampling,
    gen: Vec<u32>,
    /// per generated position: the target's ids (u32) and values
    target: Vec<(Vec<u32>, Vec<f32>)>,
    /// per anchor, per depth: the head's ids and values
    head: Vec<Vec<(Vec<u32>, Vec<f32>)>>,
}

fn r_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn load(dir: &Path) -> Option<Gen> {
    let meta = std::fs::read_to_string(dir.join("meta.txt")).ok()?;
    let get = |k: &str| meta.lines().find_map(|l| l.strip_prefix(&format!("{k}=")).map(str::to_string)).unwrap();
    let prompt_len: usize = get("prompt_len").parse().unwrap();
    let n: usize = get("generated").parse().unwrap();
    let s = Sampling {
        seed: get("seed").parse().unwrap(),
        temperature: get("temperature").parse().unwrap(),
        top_k: get("top_k").parse().unwrap(),
        top_p: get("top_p").parse().unwrap(),
        min_p: get("min_p").parse().unwrap(),
    };
    let tok = std::fs::read(dir.join("tokens.bin")).unwrap();
    let gen: Vec<u32> = (0..n).map(|i| r_u32(&tok, (prompt_len + i) * 4)).collect();
    let list = |b: &[u8], at: usize| -> (Vec<u32>, Vec<f32>) {
        let ids = (0..COUNT).map(|i| r_u32(b, at + i * 4)).collect();
        let vals = (0..COUNT).map(|i| f32::from_bits(r_u32(b, at + COUNT * 4 + i * 4))).collect();
        (ids, vals)
    };
    let tb = std::fs::read(dir.join("target.bin")).unwrap();
    let target = (0..n).map(|j| list(&tb, j * COUNT * 8)).collect();
    let hb = std::fs::read(dir.join("head.bin")).unwrap();
    let mut head = Vec::new();
    let mut at = 0;
    while at < hb.len() {
        let d = r_u32(&hb, at) as usize;
        at += 4;
        let mut v = Vec::new();
        for _ in 0..d {
            v.push(list(&hb, at));
            at += COUNT * 8;
        }
        head.push(v);
    }
    let name = dir.file_name()?.to_string_lossy().to_string();
    let mut parts = name.rsplitn(2, '-');
    let label = parts.next()?.to_string();
    let class = parts.next()?.rsplit_once('-')?.0.to_string();
    Some(Gen { class, label, prompt_len, s, gen, target, head })
}

/// How a drafter picks from its list. `Keyed(w)`: the target's sampler with the noise weighted by `w` (1 is the target's own draw, 0 would be the argmax
/// over the kept candidates).
#[derive(Clone, Copy, Debug)]
enum Pick {
    Argmax,
    Keyed(f64),
}

/// `gb_sampling::try_choose` with the Gumbel noise scaled by `w` (a host copy of its pipeline; `w == 1` equals it, checked below).
fn choose_weighted(values: &[f32], ids: &[u32], position: u64, s: &Sampling, w: f64) -> u32 {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| (-values[a]).partial_cmp(&-values[b]).unwrap().then(ids[a].cmp(&ids[b])));
    let k = if s.top_k > 0 { s.top_k.min(idx.len()) } else { idx.len() }.max(1);
    let t = s.temperature.max(1e-6);
    let mut kid: Vec<u32> = idx[..k].iter().map(|&i| ids[i]).collect();
    let mut scaled: Vec<f64> = idx[..k].iter().map(|&i| values[i] as f64 / t).collect();
    if s.top_p > 0.0 && s.top_p < 1.0 {
        let mx = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut probs: Vec<f64> = scaled.iter().map(|v| (v - mx).exp()).collect();
        let total = pairwise_sum(&probs);
        for p in probs.iter_mut() {
            *p /= total;
        }
        let (mut c, mut below) = (0.0, 0usize);
        for &p in &probs {
            c += p;
            if c < s.top_p {
                below += 1;
            }
        }
        let keep = (below + 1).min(k);
        kid.truncate(keep);
        scaled.truncate(keep);
    }
    if s.min_p > 0.0 {
        let floor = scaled[0] + s.min_p.ln();
        let keep = scaled.iter().filter(|&&v| v >= floor).count().max(1);
        kid.truncate(keep);
        scaled.truncate(keep);
    }
    let (mut best, mut best_score) = (0usize, f64::NEG_INFINITY);
    for (i, (&id, &v)) in kid.iter().zip(&scaled).enumerate() {
        let u = uniform(s.seed, position, id as u64);
        let g = -((-(u.ln())).ln());
        let score = v + w * g;
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    kid[best]
}

fn pick(p: Pick, values: &[f32], ids: &[u32], position: u64, s: &Sampling) -> u32 {
    if s.temperature == 0.0 {
        return ids[0]; // the lists are sorted: the first is the argmax, ties to the lowest id
    }
    match p {
        Pick::Argmax => ids[0],
        Pick::Keyed(1.0) => try_choose(values, ids, position, s).unwrap(),
        Pick::Keyed(w) => choose_weighted(values, ids, position, s, w),
    }
}

/// The restricted head's list: the full list's candidates inside the set, in order (its kept columns are bit-identical). None when the set holds fewer than `need`.
fn restrict(list: &(Vec<u32>, Vec<f32>), in_set: &dyn Fn(u32) -> bool, need: usize) -> Option<(Vec<u32>, Vec<f32>)> {
    let (mut ids, mut vals) = (Vec::new(), Vec::new());
    for (&i, &v) in list.0.iter().zip(&list.1) {
        if in_set(i) {
            ids.push(i);
            vals.push(v);
        }
    }
    (ids.len() >= need).then_some((ids, vals))
}

#[derive(Default, Clone, Debug)]
struct Tally {
    rounds: u64,
    committed: u64,
    drafted: u64,
    accepted: u64,
    /// per depth: rounds that reached it, and hits
    reached: [u64; DEPTH],
    hits: [u64; DEPTH],
}

impl Tally {
    fn add(&mut self, o: &Tally) {
        self.rounds += o.rounds;
        self.committed += o.committed;
        self.drafted += o.drafted;
        self.accepted += o.accepted;
        for d in 0..DEPTH {
            self.reached[d] += o.reached[d];
            self.hits[d] += o.hits[d];
        }
    }
    fn tpr(&self) -> f64 {
        self.committed as f64 / self.rounds.max(1) as f64
    }
    fn acc(&self) -> f64 {
        self.accepted as f64 / self.drafted.max(1) as f64
    }
    fn cond(&self, d: usize) -> f64 {
        self.hits[d] as f64 / self.reached[d].max(1) as f64
    }
}

/// Replay rounds of depth `k` over a generation: the round at anchor `a` drafts `k` tokens from the head's lists and commits the accepted prefix and one more.
/// `restricted`: the head's list is the set's (`Adaptive`: where the last 32 tokens are at least 90% inside the set and the set holds `top_k` candidates).
fn replay(g: &Gen, k: usize, p: Pick, restricted: Option<&dyn Fn(u32) -> bool>) -> Tally {
    let mut t = Tally::default();
    let mut a = 0usize;
    let n = g.gen.len();
    while a + 1 < n {
        let len = g.prompt_len + a + 1;
        let avail = g.head[a].len().min(k).min(n - 1 - a);
        // the sequence's last 32 tokens (the adaptive coverage test of `VocabMode::Adaptive(0.9)`)
        let use_set = restricted.filter(|in_set| {
            let all: Vec<u32> = g.gen[..=a].to_vec();
            let tail = &all[all.len().saturating_sub(32)..];
            tail.iter().filter(|&&x| in_set(x)).count() as f32 / tail.len() as f32 >= 0.9
        });
        let mut accepted = 0;
        for d in 0..avail {
            let list = &g.head[a][d];
            let list = match use_set {
                Some(in_set) => match restrict(list, in_set, g.s.top_k.max(1)) {
                    Some(l) => l,
                    None => list.clone(),
                },
                None => list.clone(),
            };
            let draft = pick(p, &list.1, &list.0, (len + d) as u64, &g.s);
            t.drafted += 1;
            t.reached[d] += 1;
            if draft == g.gen[a + 1 + d] {
                t.hits[d] += 1;
                accepted += 1;
            } else {
                break;
            }
        }
        t.accepted += accepted as u64;
        t.rounds += 1;
        t.committed += accepted as u64 + 1;
        a += accepted + 1;
    }
    t
}

/// The expected agreement of the head's pick and the target's sample at depth `d` of anchor `a`, over `seeds` noise seeds (the truth text is one
/// realisation; this has no sampling noise of its own): the head's list at depth `d` against the target's list at the position it predicts.
fn expected_agreement(g: &Gen, d: usize, p: Pick, seeds: u64, restricted: Option<&dyn Fn(u32) -> bool>) -> (f64, u64) {
    let (mut agree, mut total) = (0.0f64, 0u64);
    let cc = g.s.top_k + 8;
    for a in 0..g.head.len() {
        if g.head[a].len() <= d || a + 1 + d >= g.gen.len() {
            continue;
        }
        let list = match restricted {
            Some(in_set) => restrict(&g.head[a][d], in_set, g.s.top_k.max(1)).unwrap_or_else(|| g.head[a][d].clone()),
            None => g.head[a][d].clone(),
        };
        let tl = &g.target[a + 1 + d];
        let pos = (g.prompt_len + a + 1 + d) as u64;
        let mut hits = 0u32;
        for sd in 0..seeds {
            let s = Sampling { seed: sd * 7919 + 13, ..g.s };
            let truth = try_choose(&tl.1[..cc], &tl.0[..cc], pos, &s).unwrap();
            if pick(p, &list.1, &list.0, pos, &s) == truth {
                hits += 1;
            }
        }
        agree += hits as f64 / seeds as f64;
        total += 1;
    }
    (agree / total.max(1) as f64, total)
}

#[test]
fn the_weighted_pick_at_weight_one_is_the_samplers_own() {
    let mut r = 5u64;
    for case in 0..200 {
        let mut next = || {
            r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            r >> 33
        };
        let ids: Vec<u32> = (0..40).map(|i| i * 3 + 1).collect();
        let mut vals: Vec<f32> = (0..40).map(|_| ((next() % 3000) as f32 / 500.0) - 3.0).collect();
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let s = Sampling {
            seed: next(),
            temperature: [0.5, 0.7, 1.0][(next() % 3) as usize],
            top_k: [5usize, 20, 40][(next() % 3) as usize],
            top_p: [1.0, 0.95, 0.7][(next() % 3) as usize],
            min_p: [0.0, 0.1][(next() % 2) as usize],
        };
        let pos = next() % 1000;
        assert_eq!(choose_weighted(&vals, &ids, pos, &s, 1.0), try_choose(&vals, &ids, pos, &s).unwrap(), "case {case}");
    }
}

#[test]
fn replay_ceiling() {
    let Ok(dir) = std::env::var("REXL3_NOISE_DUMP") else {
        eprintln!("REXL3_NOISE_DUMP is not set: not run");
        return;
    };
    let mut gens: Vec<Gen> = std::fs::read_dir(&dir).unwrap().filter_map(|e| load(&e.unwrap().path())).collect();
    gens.sort_by(|a, b| (a.label.clone(), a.class.clone()).cmp(&(b.label.clone(), b.class.clone())));
    assert!(!gens.is_empty(), "no generation in {dir}");
    let ranges = vocab_ranges(512, 154_880);
    let in_set = move |id: u32| {
        let strip = id as usize / 128;
        ranges.iter().any(|&(lo, hi)| strip >= lo && strip < hi)
    };
    let seeds: u64 = std::env::var("REXL3_NOISE_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    // policies: the shipped argmax, the keyed draw, and hedges (the noise at a fraction of its weight, TensorFold's 0.7 among them)
    let policies: Vec<(&str, Pick)> = vec![
        ("argmax", Pick::Argmax),
        ("keyed", Pick::Keyed(1.0)),
        ("w0.7", Pick::Keyed(0.7)),
        ("w0.5", Pick::Keyed(0.5)),
        ("w0.3", Pick::Keyed(0.3)),
    ];
    let labels: Vec<String> = {
        let mut v: Vec<String> = gens.iter().map(|g| g.label.clone()).collect();
        v.dedup();
        v
    };
    let mut report = String::new();
    for head_name in ["full", "restricted"] {
        let restricted: Option<&dyn Fn(u32) -> bool> = (head_name == "restricted").then_some(&in_set as &dyn Fn(u32) -> bool);
        for label in &labels {
            let sel: Vec<&Gen> = gens.iter().filter(|g| &g.label == label).collect();
            let n_gen: usize = sel.iter().map(|g| g.gen.len()).sum();
            report += &format!("\n### head {head_name}, {label}: {} generations, {n_gen} tokens\n\n", sel.len());
            report +=
                "tokens per round (realised rounds), K = 1 / 2 / 3 / 4, and conditional acceptance of depth 1..4 at K = 4:\n\n";
            report += "| policy | K=1 | K=2 | K=3 | K=4 | a1 | a2 | a3 | a4 | accepted/drafted K=4 |\n|---|---|---|---|---|---|---|---|---|---|\n";
            for (pn, p) in &policies {
                let mut tpr = Vec::new();
                let mut t4 = Tally::default();
                for k in 1..=DEPTH {
                    let mut t = Tally::default();
                    for g in &sel {
                        t.add(&replay(g, k, *p, restricted));
                    }
                    tpr.push(t.tpr());
                    if k == DEPTH {
                        t4 = t;
                    }
                }
                report += &format!(
                    "| {pn} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n",
                    tpr[0],
                    tpr[1],
                    tpr[2],
                    tpr[3],
                    t4.cond(0),
                    t4.cond(1),
                    t4.cond(2),
                    t4.cond(3),
                    t4.acc()
                );
            }
            // by class at K = 2 and K = 3
            report += "\nby class, tokens per round at K = 2 / 3 (argmax -> keyed):\n\n| class | K=2 argmax | K=2 keyed | gain | K=3 argmax | K=3 keyed | gain | depth-1 accepted argmax -> keyed |\n|---|---|---|---|---|---|---|---|\n";
            let mut classes: Vec<String> = sel.iter().map(|g| g.class.clone()).collect();
            classes.sort();
            classes.dedup();
            for c in classes {
                let cs: Vec<&&Gen> = sel.iter().filter(|g| g.class == c).collect();
                let run = |k: usize, p: Pick| {
                    let mut t = Tally::default();
                    for g in &cs {
                        t.add(&replay(g, k, p, restricted));
                    }
                    t
                };
                let (a2, k2, a3, k3) =
                    (run(2, Pick::Argmax), run(2, Pick::Keyed(1.0)), run(3, Pick::Argmax), run(3, Pick::Keyed(1.0)));
                report += &format!(
                    "| {c} | {:.3} | {:.3} | {:+.1}% | {:.3} | {:.3} | {:+.1}% | {:.3} -> {:.3} |\n",
                    a2.tpr(),
                    k2.tpr(),
                    (k2.tpr() / a2.tpr() - 1.0) * 100.0,
                    a3.tpr(),
                    k3.tpr(),
                    (k3.tpr() / a3.tpr() - 1.0) * 100.0,
                    a3.cond(0),
                    k3.cond(0)
                );
            }
            // the expected agreement over seeds, pooled
            report += &format!("\nexpected agreement of the head's pick with the target's sample over {seeds} seeds, by depth (no sampling noise of its own):\n\n| policy | depth 1 | depth 2 | depth 3 | depth 4 |\n|---|---|---|---|---|\n");
            for (pn, p) in &policies {
                let mut row = String::new();
                for d in 0..DEPTH {
                    let (mut num, mut den) = (0.0, 0u64);
                    for g in &sel {
                        let (a, n) = expected_agreement(g, d, *p, seeds, restricted);
                        num += a * n as f64;
                        den += n;
                    }
                    row += &format!(" {:.3} |", num / den.max(1) as f64);
                }
                report += &format!("| {pn} |{row}\n");
            }
        }
    }
    println!("{report}");
    if let Ok(path) = std::env::var("REXL3_NOISE_REPORT") {
        std::fs::write(path, &report).unwrap();
    }
}
