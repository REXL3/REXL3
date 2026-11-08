#![allow(clippy::needless_range_loop)]
//! Tree drafting, stage S0 (docs/TREE_DRAFTING.md "S0 results"): the DFlash2 selector's candidate lattice at EVERY position of
//! real generations, for the offline ceiling study (`tools/tree_ceiling.py`).
//!
//! Why every position: a verify round's tree is built at the anchor the previous round ended on, so a different tree policy starts
//! its rounds at different positions than the chain does. The lattice at an anchor depends only on the committed context and the
//! anchor (the draft block runs on `[anchor, MASK x 7]`; only the selector's edge terms depend on earlier picks), and the committed
//! text is the same whatever the speculation (drafted == serial). So the generation is run SERIALLY (one row a step, the keyed sampler
//! of the request), a draft round with `Draft::capture_lattice` is run at every position, and any policy's rounds can then be
//! replayed offline against the true continuation exactly.
//!
//!     REXL3_QWEN_DIR=... REXL3_DRAFT_DIR=... REXL3_LATTICE_CASES=cases.json REXL3_LATTICE_OUT=outdir \
//!       [REXL3_LATTICE_STREAMS=6] [REXL3_LATTICE_TOKENS=384] \
//!       cargo test --release -p gb-model --test lattice_dump -- --ignored --nocapture dump_lattices
//!
//! `cases.json` (written by `tools/tree_prompts.py`): `[{"name", "class", "ids": [prompt token ids], "temperature", "top_p",
//! "top_k", "seed"}]`. Per case the test writes `<name>.lat` (binary records, layout below) and `<name>.json` (prompt and generated
//! ids, sampling). Record layout, little endian: u32 pos, u32 anchor, u32 flags (bit 0: the restricted draft head ran), u32 rows,
//! then `ids u32[rows][16]`, `unary f32[rows][16]`, `edge f32[rows][16 parents][16]`, `picks u32[rows]`, `margin f32[rows]`,
//! `score f32[rows]` (the walk's picks, leads and winning scores). The file starts with `b"LAT1"`, u32 rows, u32 k.

mod refs;
use gb_engine::RowRequest;
use gb_model::config::QwenConfig;
use gb_model::draft::{Draft, Lattice, DRAFTS, SELECTOR_TOP_K};
use gb_model::forward::Model;
use gb_model::pack::Pack;
use gb_model::rows::RowBatch;
use gb_model::weights::Weights;
use gb_sampling::{try_choose, Sampling};
use refs::{have_refs, ref_npy};

/// End-of-turn tokens (`generation_config.json`): a generation stops after one.
const EOS: [u32; 2] = [248046, 248044];

struct Env {
    m: Model,
    chunk: usize,
}

fn load(slots: usize, max_context: usize, chunk: usize) -> Option<Env> {
    let d = std::path::PathBuf::from(std::env::var("REXL3_QWEN_DIR").ok()?);
    let dd = std::path::PathBuf::from(std::env::var("REXL3_DRAFT_DIR").ok()?);
    if !gb_cuda::device_available() {
        return None;
    }
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    let vocab = cfg.vocab;
    let blocks = slots * max_context.div_ceil(256) + 8;
    let mut m = Model::new(cfg, w, slots, blocks, max_context, chunk).unwrap();
    let block = gb_model::draft::draft_block_from_env().unwrap();
    let mut draft = Draft::load_block(&Pack::open(&dd).unwrap(), vocab, slots, chunk, block).unwrap().0;
    // the shipped default head: TensorFold's candidate set where every stream's recent tokens are inside it, the full head otherwise
    if std::env::var("REXL3_DRAFT_VOCAB").as_deref() != Ok("full") {
        let ranges = gb_model::costs::DraftVocab::Auto.ranges().expect("the auto set has ranges");
        draft.restrict_vocab(&m.w.lm_head, &ranges).unwrap();
        draft.set_adaptive(Some(gb_model::costs::AUTO_MIN_IN_SET));
    }
    draft.capture_lattice = true;
    m.draft = Some(draft);
    Some(Env { m, chunk })
}

fn sampling_of(c: &serde_json::Value) -> Sampling {
    Sampling {
        seed: c["seed"].as_u64().unwrap_or(0),
        temperature: c["temperature"].as_f64().unwrap_or(1.0),
        top_k: c["top_k"].as_u64().unwrap_or(20) as usize,
        top_p: c["top_p"].as_f64().unwrap_or(0.95),
        min_p: 0.0,
    }
}

struct Stream {
    name: String,
    slot: usize,
    sampling: Sampling,
    tokens: Vec<u32>,
    prompt_len: usize,
    done: bool,
    cases_meta: serde_json::Value,
    rec: Vec<u8>,
    records: usize,
    head: [usize; 2],
}

fn put_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}
fn put_f32(v: &mut Vec<u8>, x: f32) {
    v.extend_from_slice(&x.to_le_bytes());
}

/// One anchor's record (layout in the module docs).
#[allow(clippy::too_many_arguments)] // a binary record: position, anchor, the block's drafts, then the lattice and the walk
fn record(
    out: &mut Vec<u8>,
    pos: usize,
    anchor: u32,
    drafts: usize,
    lat: &Lattice,
    picks: &[u32],
    margin: &[f32],
    score: &[f32],
) {
    assert_eq!((lat.rows, lat.k), (drafts, SELECTOR_TOP_K), "the lattice rows are the block's drafts");
    put_u32(out, pos as u32);
    put_u32(out, anchor);
    put_u32(out, lat.restricted_head as u32);
    put_u32(out, lat.rows as u32);
    lat.ids.iter().for_each(|&x| put_u32(out, x));
    lat.unary.iter().for_each(|&x| put_f32(out, x));
    lat.edge.iter().for_each(|&x| put_f32(out, x));
    picks.iter().for_each(|&x| put_u32(out, x));
    margin.iter().for_each(|&x| put_f32(out, x));
    score.iter().for_each(|&x| put_f32(out, x));
}

/// Prefill `prompt` into `slot` the way the Qwen backend does (the same chunk grid, the draft's context window) and return the
/// candidates after it.
fn prefill(env: &mut Env, slot: usize, prompt: &[u32], cand: usize) -> (Vec<u32>, Vec<f32>) {
    let n = prompt.len();
    env.m.begin_slot(slot).unwrap();
    let ctx_from = n.saturating_sub(gb_model::draft::CTX_KEEP);
    let (mut at, mut last) = (0, None);
    while at < n {
        let end = (at + env.chunk).min(n);
        let want = (end == n).then_some(cand);
        last = env.m.prefill_chunk(slot, at, &prompt[at..end], n, want, ctx_from).unwrap();
        at = end;
    }
    let c = last.expect("a prefill returns the last row's candidates");
    (c.ids[..cand].iter().map(|&x| x as u32).collect(), c.vals[..cand].to_vec())
}

fn run_group(env: &mut Env, cases: &[serde_json::Value], first_slot: usize, new_tokens: usize, out_dir: &std::path::Path) {
    let vocab = env.m.cfg.vocab;
    let drafts = env.m.draft.as_ref().map(|d| d.block - 1).unwrap_or(DRAFTS);
    let mut streams: Vec<Stream> = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let prompt: Vec<u32> = c["ids"].as_array().expect("ids").iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let sampling = sampling_of(c);
        let cand = sampling.candidate_count(vocab);
        let slot = first_slot + i;
        let (ids, vals) = prefill(env, slot, &prompt, cand);
        let first = try_choose(&vals, &ids, prompt.len() as u64, &sampling).expect("the first token samples");
        let mut tokens = prompt.clone();
        tokens.push(first);
        streams.push(Stream {
            name: c["name"].as_str().unwrap().to_string(),
            slot,
            sampling,
            tokens,
            prompt_len: prompt.len(),
            done: EOS.contains(&first),
            cases_meta: c.clone(),
            rec: Vec::new(),
            records: 0,
            head: [0; 2],
        });
    }
    let t0 = std::time::Instant::now();
    let mut steps = 0usize;
    loop {
        let active: Vec<usize> = (0..streams.len()).filter(|&i| !streams[i].done).collect();
        if active.is_empty() {
            break;
        }
        // the lattice at each active stream's anchor (the last token, not yet fed)
        let asks: Vec<(usize, &[u32], &Sampling, usize)> =
            active.iter().map(|&i| (streams[i].slot, &streams[i].tokens[..], &streams[i].sampling, drafts)).collect();
        let picked = env.m.propose_drafts_batch_want(&asks).unwrap();
        drop(asks);
        for (&i, p) in active.iter().zip(&picked) {
            let p = p.as_ref().expect("a draft");
            let lat = p.lattice.as_ref().expect("lattice capture is on");
            let s = &mut streams[i];
            let pos = s.tokens.len() - 1;
            record(&mut s.rec, pos, s.tokens[pos], drafts, lat, &p.tokens, &p.margin, &p.score);
            s.records += 1;
            s.head[lat.restricted_head as usize] += 1;
        }
        // the serial step: one row per stream (the same kernels and bits a verify window's row has)
        let reqs: Vec<RowRequest> = active
            .iter()
            .map(|&i| {
                let s = &streams[i];
                RowRequest { slot: s.slot, tokens: vec![*s.tokens.last().unwrap()], first_position: (s.tokens.len() - 1) as u64 }
            })
            .collect();
        let cand = active.iter().map(|&i| streams[i].sampling.candidate_count(vocab)).max().unwrap();
        let c = env.m.window_candidates(&RowBatch::from_requests(&reqs), cand).unwrap();
        for (r, &i) in active.iter().enumerate() {
            let s = &mut streams[i];
            let ids: Vec<u32> = c.ids[r * c.count..(r + 1) * c.count].iter().map(|&x| x as u32).collect();
            let t = try_choose(&c.vals[r * c.count..(r + 1) * c.count], &ids, s.tokens.len() as u64, &s.sampling)
                .expect("a row samples");
            env.m.commit(s.slot, 1);
            s.tokens.push(t);
            let generated = s.tokens.len() - s.prompt_len;
            if EOS.contains(&t) || generated >= new_tokens {
                s.done = true;
            }
        }
        steps += 1;
        if steps % 64 == 0 {
            eprintln!("  {steps} steps, {:.1} s", t0.elapsed().as_secs_f64());
        }
    }
    for s in &streams {
        let mut f = Vec::from(&b"LAT1"[..]);
        put_u32(&mut f, drafts as u32);
        put_u32(&mut f, SELECTOR_TOP_K as u32);
        f.extend_from_slice(&s.rec);
        std::fs::write(out_dir.join(format!("{}.lat", s.name)), f).unwrap();
        let mut meta = s.cases_meta.clone();
        meta["gen"] = serde_json::json!(s.tokens[s.prompt_len..]);
        meta["anchors"] = serde_json::json!(s.records);
        meta["head_full_restricted"] = serde_json::json!(s.head);
        meta.as_object_mut().unwrap().remove("ids");
        meta["prompt_len"] = serde_json::json!(s.prompt_len);
        meta["prompt"] = s.cases_meta["ids"].clone();
        std::fs::write(out_dir.join(format!("{}.json", s.name)), serde_json::to_string(&meta).unwrap()).unwrap();
        eprintln!(
            "{}: {} anchors, {} generated, head full/restricted {:?}",
            s.name,
            s.records,
            s.tokens.len() - s.prompt_len,
            s.head
        );
        env.m.end_slot(s.slot);
    }
}

#[test]
#[ignore]
fn dump_lattices() {
    let (Ok(cases), Ok(out)) = (std::env::var("REXL3_LATTICE_CASES"), std::env::var("REXL3_LATTICE_OUT")) else {
        return eprintln!("not run: needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR, REXL3_LATTICE_CASES, REXL3_LATTICE_OUT and a GPU");
    };
    let streams: usize = std::env::var("REXL3_LATTICE_STREAMS").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    let new_tokens: usize = std::env::var("REXL3_LATTICE_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(384);
    let max_ctx: usize = std::env::var("REXL3_LATTICE_CONTEXT").ok().and_then(|v| v.parse().ok()).unwrap_or(12288);
    let cases: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cases).unwrap()).unwrap();
    let mut cases = cases.as_array().expect("a list of cases").clone();
    // REXL3_LATTICE_ONLY=name,name: only these cases (a solo re-run of a few, to compare with the batched dump)
    if let Ok(only) = std::env::var("REXL3_LATTICE_ONLY") {
        let names: Vec<&str> = only.split(',').collect();
        cases.retain(|c| names.contains(&c["name"].as_str().unwrap()));
    }
    let out_dir = std::path::PathBuf::from(out);
    std::fs::create_dir_all(&out_dir).unwrap();
    let Some(mut env) = load(streams, max_ctx, 2048) else {
        return eprintln!("not run: needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR and a GPU");
    };
    for (g, group) in cases.chunks(streams).enumerate() {
        eprintln!("group {g}: {}", group.iter().map(|c| c["name"].as_str().unwrap()).collect::<Vec<_>>().join(" "));
        run_group(&mut env, group, 0, new_tokens, &out_dir);
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// The lattice of a sequence is the same bits whether its draft round runs alone or batched, with the capture on or off for the
/// picks, as long as the same head (full or restricted) ran: the S0 study logs six streams at once and relies on it.
#[test]
#[ignore]
fn batched_lattices_equal_solo_lattices() {
    if std::env::var("REXL3_QWEN_DIR").is_err() || std::env::var("REXL3_DRAFT_DIR").is_err() || !have_refs() {
        return eprintln!("not run: needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR, a GPU and the reference dumps (REXL3_REF_DIR)");
    }
    let slots = 6usize;
    let Some(mut env) = load(slots, 4096, 256) else { return eprintln!("not run: needs a GPU") };
    let long: Vec<u32> = ref_npy("long/ids.npy").to_i64().into_iter().map(|x| x as u32).collect();
    let prompts: Vec<Vec<u32>> = (0..slots).map(|s| long[s * 9..long.len() - 3 * s].to_vec()).collect();
    let samplings = [
        Sampling { seed: 11, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 },
        Sampling { seed: 12, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 },
        Sampling { seed: 13, temperature: 0.7, top_k: 20, top_p: 0.95, min_p: 0.0 },
        Sampling { seed: 14, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 },
        Sampling { seed: 15, temperature: 1.3, top_k: 20, top_p: 0.5, min_p: 0.0 },
        Sampling { seed: 16, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 },
    ];
    let mut seqs = Vec::new();
    for s in 0..slots {
        prefill(&mut env, s, &prompts[s], 28);
        let mut t = prompts[s].clone();
        t.push(1000 + 17 * s as u32);
        seqs.push(t);
    }
    let drafts = env.m.draft.as_ref().map(|d| d.block - 1).unwrap_or(DRAFTS);
    let one = |env: &mut Env, s: usize| {
        env.m.propose_drafts_batch_want(&[(s, &seqs[s][..], &samplings[s], drafts)]).unwrap().pop().unwrap().unwrap()
    };
    let solo: Vec<_> = (0..slots).map(|s| one(&mut env, s)).collect();
    let asks: Vec<(usize, &[u32], &Sampling, usize)> = (0..slots).map(|s| (s, &seqs[s][..], &samplings[s], drafts)).collect();
    let all = env.m.propose_drafts_batch_want(&asks).unwrap();
    for s in 0..slots {
        let (a, b) = (&solo[s], all[s].as_ref().unwrap());
        let (la, lb) = (a.lattice.as_ref().unwrap(), b.lattice.as_ref().unwrap());
        assert_eq!(a.tokens, b.tokens, "slot {s}: picks");
        assert_eq!(bits(&a.margin), bits(&b.margin), "slot {s}: margins");
        if la.restricted_head == lb.restricted_head {
            assert_eq!(la.ids, lb.ids, "slot {s}: candidates");
            assert_eq!(bits(&la.unary), bits(&lb.unary), "slot {s}: unary scores");
            assert_eq!(bits(&la.edge), bits(&lb.edge), "slot {s}: edge terms");
        } else {
            eprintln!("slot {s}: the batch took the other draft head than the solo round; lattices not compared");
        }
        // the lattice reproduces the walk's own scores: a greedy walk through it picks what the draft picked (greedy streams only)
        if samplings[s].temperature == 0.0 {
            let mut parent = 0usize;
            for i in 0..lb.rows {
                let sc = |c: usize| lb.unary[i * lb.k + c] + lb.edge[(i * lb.k + parent) * lb.k + c];
                let best = (1..lb.k).fold(0, |b, c| if sc(c) > sc(b) { c } else { b });
                assert_eq!(lb.ids[i * lb.k + best], b.tokens[i], "slot {s} row {i}: the lattice chain is not the walk");
                assert_eq!(sc(best).to_bits(), b.score[i].to_bits(), "slot {s} row {i}: winning score bits");
                parent = best;
            }
        }
    }
    eprintln!("lattices: {} streams, batched == solo", slots);
}
