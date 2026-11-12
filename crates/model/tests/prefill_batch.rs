#![allow(clippy::needless_range_loop)]
//! Several short prompts through ONE prefill forward (`Model::prefill_batch`) on the real model: each prompt's last-row candidates, the
//! tokens it decodes afterwards (so its recurrent state, conv history and KV rows) and the drafts the draft proposes from the context it
//! absorbed are bit-identical to what the same prompt gives in a slot of its own, whatever else rides in the forward and in whatever order.
//!
//!     REXL3_QWEN_DIR=... REXL3_DRAFT_DIR=... cargo test --release -p gb-model --test prefill_batch -- --ignored --nocapture

mod refs;
use gb_engine::RowRequest;
use gb_model::config::QwenConfig;
use gb_model::draft::Draft;
use gb_model::error::ModelError;
use gb_model::forward::{BatchPrompt, Candidates, Model};
use gb_model::pack::Pack;
use gb_model::rows::RowBatch;
use gb_model::weights::Weights;
use gb_sampling::Sampling;
use refs::{have_refs, ref_npy};

const COUNT: usize = 28;
/// Tokens decoded one row at a time after each prefill.
const DECODE: usize = 6;
const SLOTS: usize = 24;

fn load() -> Option<Model> {
    let d = std::path::PathBuf::from(std::env::var("REXL3_QWEN_DIR").ok()?);
    let dd = std::path::PathBuf::from(std::env::var("REXL3_DRAFT_DIR").ok()?);
    if !gb_cuda::device_available() || !have_refs() {
        return None;
    }
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let vocab = cfg.vocab;
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    let mut m = Model::new(cfg, w, SLOTS, SLOTS * 4 + 64, 4096, 512).unwrap();
    m.draft = Some(Draft::load(&Pack::open(&dd).unwrap(), vocab, SLOTS, 512).unwrap().0);
    Some(m)
}

fn row(c: &Candidates, r: usize) -> (&[i32], &[f32]) {
    (&c.ids[r * c.count..(r + 1) * c.count], &c.vals[r * c.count..(r + 1) * c.count])
}

fn same(a: (&[i32], &[f32]), b: (&[i32], &[f32])) -> bool {
    a.0 == b.0 && a.1.iter().map(|v| v.to_bits()).eq(b.1.iter().map(|v| v.to_bits()))
}

/// What a slot gives after its prompt: the candidates of the prompt's last row, then `DECODE` greedy serial steps (their candidates),
/// then the drafts the draft proposes from the context it holds.
struct Trace {
    first: Candidates,
    steps: Vec<Candidates>,
    drafts: Option<Vec<u32>>,
}

fn sampling(seed: u64) -> Sampling {
    Sampling { seed, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 }
}

/// Continue a slot whose prompt (`prompt`, first candidates `first`) is in: the same serial steps whichever way the prompt got there.
fn trace(m: &mut Model, slot: usize, prompt: &[u32], first: Candidates, seed: u64) -> Trace {
    let p = prompt.len();
    let mut toks = prompt.to_vec();
    toks.push(first.ids[0] as u32);
    // drafts first: they read the context the prefill absorbed, before any decode step adds to it
    let drafts = m.propose_drafts(slot, &toks, &sampling(seed)).unwrap();
    let mut steps = Vec::new();
    for k in 0..DECODE {
        let b = RowBatch::from_requests(&[RowRequest { slot, tokens: vec![toks[p + k]], first_position: (p + k) as u64 }]);
        let c = m.window_candidates(&b, COUNT).unwrap();
        m.commit(slot, 1);
        toks.push(c.ids[0] as u32);
        steps.push(c);
    }
    Trace { first, steps, drafts }
}

fn assert_same_trace(a: &Trace, b: &Trace, what: &str) {
    assert!(same(row(&a.first, 0), row(&b.first, 0)), "{what}: the candidates of the prompt's last row differ");
    for (k, (x, y)) in a.steps.iter().zip(&b.steps).enumerate() {
        assert!(same(row(x, 0), row(y, 0)), "{what}: decode step {k} differs (the state the prefill left is not the solo one)");
    }
    assert_eq!(a.drafts, b.drafts, "{what}: the draft proposes differently from the context it absorbed");
}

fn prompts() -> Vec<Vec<u32>> {
    let long: Vec<u32> = ref_npy("long/ids.npy").to_i64().into_iter().map(|x| x as u32).collect();
    // different content and lengths: 1, 3, 7, 12, 19, 25, 31, 40, 55 and 64 tokens (the reference prompt is 96 tokens long)
    [1usize, 3, 7, 12, 19, 25, 31, 40, 55, 64].iter().enumerate().map(|(i, &n)| long[i * 3..i * 3 + n].to_vec()).collect()
}

#[test]
#[ignore]
fn a_batched_prefill_leaves_every_prompt_exactly_as_a_solo_prefill_does() {
    let Some(mut m) = load() else { return eprintln!("needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR, a GPU and testdata/ref: skipping") };
    let ps = prompts();
    let n = ps.len();
    // solo references, slots 0..n
    let mut solo: Vec<Trace> = Vec::new();
    for k in 0..n {
        m.begin_slot(k).unwrap();
        let ctx_from = ps[k].len().saturating_sub(gb_model::draft::CTX_KEEP);
        let first = m.prefill_chunk(k, 0, &ps[k], ps[k].len(), Some(COUNT), ctx_from).unwrap().unwrap();
        solo.push(trace(&mut m, k, &ps[k], first, 7 + k as u64));
    }
    eprintln!("solo references for {n} prompts of 1..64 tokens");
    // groups of prompts (indices), each within 128 rows; the same prompts in different company and order
    let groups: [&[usize]; 6] = [&[4, 5, 6, 7], &[7, 6, 5, 4], &[0, 1, 2, 3, 4, 5], &[9, 8], &[3], &[1, 9, 0, 2, 7]];
    for (g, idx) in groups.iter().enumerate() {
        // fresh slots for the batch: n.. (the solo slots keep their state)
        let slots: Vec<usize> = (0..idx.len()).map(|j| n + j).collect();
        let bp: Vec<BatchPrompt> = idx
            .iter()
            .zip(&slots)
            .map(|(&k, &slot)| BatchPrompt {
                slot,
                tokens: &ps[k],
                ctx_from: ps[k].len().saturating_sub(gb_model::draft::CTX_KEEP),
            })
            .collect();
        let rows: usize = idx.iter().map(|&k| ps[k].len()).sum();
        let c = m.prefill_batch(&bp, COUNT).unwrap();
        assert_eq!(c.rows, idx.len());
        for (j, &k) in idx.iter().enumerate() {
            // the batch's candidates for this prompt, as a one-row Candidates
            let first = Candidates {
                rows: 1,
                count: c.count,
                ids: c.ids[j * c.count..(j + 1) * c.count].to_vec(),
                vals: c.vals[j * c.count..(j + 1) * c.count].to_vec(),
            };
            let t = trace(&mut m, slots[j], &ps[k], first, 7 + k as u64);
            assert_same_trace(
                &t,
                &solo[k],
                &format!("group {g} ({rows} rows, prompts {idx:?}), prompt {k} ({} tokens)", ps[k].len()),
            );
        }
        eprintln!("group {g}: prompts {idx:?}, {rows} rows in one forward: every prompt equals its solo prefill");
    }
}

#[test]
#[ignore]
fn a_batch_the_model_cannot_take_is_refused_before_it_touches_anything() {
    let Some(mut m) = load() else { return eprintln!("needs REXL3_QWEN_DIR, REXL3_DRAFT_DIR, a GPU and testdata/ref: skipping") };
    let ps = prompts();
    m.begin_slot(0).unwrap();
    let first = m.prefill_chunk(0, 0, &ps[5], ps[5].len(), Some(COUNT), 0).unwrap().unwrap();
    let before = trace(&mut m, 0, &ps[5], first, 3);
    let refuse = |m: &mut Model, bp: Vec<BatchPrompt>, want: usize, what: &str| match m.prefill_batch(&bp, want) {
        Err(ModelError::Request(msg)) => assert!(msg.contains("batched prefill"), "{what}: {msg}"),
        other => panic!("{what}: {:?}", other.map(|c| c.rows)),
    };
    let long129: Vec<u32> = (0..129).map(|i| 1000 + i as u32).collect();
    let two = [ps[9].as_slice(), ps[9].as_slice()];
    refuse(&mut m, vec![], COUNT, "no prompts");
    refuse(&mut m, vec![BatchPrompt { slot: 10, tokens: &long129, ctx_from: 0 }], COUNT, "a prompt past the GEMV route");
    refuse(&mut m, vec![BatchPrompt { slot: 10, tokens: &[], ctx_from: 0 }], COUNT, "an empty prompt");
    refuse(
        &mut m,
        vec![BatchPrompt { slot: 11, tokens: two[0], ctx_from: 0 }, BatchPrompt { slot: 11, tokens: two[1], ctx_from: 0 }],
        COUNT,
        "one slot twice",
    );
    refuse(&mut m, vec![BatchPrompt { slot: SLOTS, tokens: &ps[1], ctx_from: 0 }], COUNT, "a slot that does not exist");
    refuse(&mut m, vec![BatchPrompt { slot: 12, tokens: &ps[1], ctx_from: 0 }], 0, "no candidates asked for");
    // 3 x 64 = 192 rows: past what one decode call takes
    let wide: Vec<BatchPrompt> = (0..3).map(|j| BatchPrompt { slot: 13 + j, tokens: &ps[9], ctx_from: 0 }).collect();
    refuse(&mut m, wide, COUNT, "192 rows");
    // slot 0 was not touched by any of them: its next steps are what the same steps gave before? (it already decoded DECODE tokens, so
    // compare a fresh solo prefill of the same prompt against a rerun of the whole trace)
    m.begin_slot(1).unwrap();
    let first = m.prefill_chunk(1, 0, &ps[5], ps[5].len(), Some(COUNT), 0).unwrap().unwrap();
    let after = trace(&mut m, 1, &ps[5], first, 3);
    assert_same_trace(&before, &after, "after the refused batches");
}
