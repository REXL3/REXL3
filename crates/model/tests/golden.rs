//! Layer goldens on the real model: hashes of the candidate lists (ids and the bits of their logits) after fixed prompts
//! and a few decode steps. Every path that can move an output bit feeds them: the decode GEMV and the cuBLAS prefill
//! linears, the W_q reconstruct, GDN prefill, window and commit, paged attention for prefill and decode, the glue kernels
//! and the head. `scripts/gpu_gate.sh` diffs the output against `testdata/gpu/golden_layers.txt`, so a different cuBLAS, a
//! different nvcc or a kernel change that was meant to keep every bit is caught by name.
//!
//!     REXL3_GOLDEN_OUT=golden_layers.txt REXL3_QWEN_DIR=... cargo test --release -p gb-model --test golden -- --ignored

use gb_engine::RowRequest;
use gb_model::config::QwenConfig;
use gb_model::forward::{Candidates, Model};
use gb_model::pack::Pack;
use gb_model::rows::RowBatch;
use gb_model::weights::Weights;
use std::io::Write;

const COUNT: usize = 28;
const CHUNK: usize = 4096;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn hash(c: &Candidates) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for (i, v) in c.ids.iter().zip(&c.vals) {
        fnv(&mut h, &i.to_le_bytes());
        fnv(&mut h, &v.to_bits().to_le_bytes());
    }
    h
}

/// A fixed pseudo-random prompt of ordinary text tokens (no special tokens).
fn prompt(len: usize, seed: u64) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1000 + (x % 99_000) as u32
        })
        .collect()
}

/// Prefill `tokens` into `slot` the way the server does (chunks of `CHUNK` rows from position 0) and return the last row's
/// candidates.
fn prefill(m: &mut Model, slot: usize, tokens: &[u32]) -> Candidates {
    m.begin_slot(slot).unwrap();
    let mut last = None;
    for start in (0..tokens.len()).step_by(CHUNK) {
        let end = (start + CHUNK).min(tokens.len());
        let want = (end == tokens.len()).then_some(COUNT);
        last = m.prefill_chunk(slot, start, &tokens[start..end], tokens.len(), want, 0).unwrap();
    }
    last.unwrap()
}

#[test]
#[ignore]
fn golden_layer_hashes() {
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    let Ok(d) = std::env::var("REXL3_QWEN_DIR") else { return eprintln!("REXL3_QWEN_DIR not set: skipping") };
    if !gb_cuda::device_available() {
        return eprintln!("no CUDA device: skipping");
    }
    let d = std::path::PathBuf::from(d);
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    let mut m = Model::new(cfg, w, 6, 96, 16384, CHUNK).unwrap();
    let mut out = String::new();
    // (name, prompt length): 100 stays on the GEMV path, 300 takes cuBLAS in one chunk, 5000 and 9000 cross chunk edges
    for (name, len) in [("p100", 100usize), ("p300", 300), ("p5000", 5000), ("p9000", 9000)] {
        let toks = prompt(len, len as u64);
        let c = prefill(&mut m, 0, &toks);
        out += &format!("{name} prefill {:016x} top {:?}\n", hash(&c), &c.ids[..4]);
        // greedy decode: one-row windows, each committed (attention decode, GDN window and commit)
        let mut t = c.ids[0];
        for k in 0..4 {
            let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: vec![t as u32], first_position: (len + k) as u64 }]);
            let c = m.window_candidates(&b, COUNT).unwrap();
            m.commit(0, 1);
            out += &format!("{name} decode{k} {:016x}\n", hash(&c));
            t = c.ids[0];
        }
        // an 8-row verify window of pseudo-drafts, uncommitted (the multi-row decode kernels)
        let drafts = prompt(8, 7 + len as u64);
        let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: drafts, first_position: (len + 4) as u64 }]);
        let c = m.window_candidates(&b, COUNT).unwrap();
        m.commit(0, 1);
        out += &format!("{name} window8 {:016x}\n", hash(&c));
    }
    // batched verify windows across sequences, the multi-stream shapes: 2 x 8 = 16 rows and 6 x 8 = 48 rows (the two- and
    // three-pass decode kernels on real weights)
    let starts: Vec<(Vec<u32>, u32)> = (0..6)
        .map(|s| {
            let toks = prompt(200 + 37 * s, 100 + s as u64);
            let c = prefill(&mut m, s, &toks);
            (toks, c.ids[0] as u32)
        })
        .collect();
    for n_seq in [2usize, 6] {
        let reqs: Vec<RowRequest> = (0..n_seq)
            .map(|s| {
                let mut tokens = vec![starts[s].1];
                tokens.extend(prompt(7, 900 + s as u64));
                RowRequest { slot: s, tokens, first_position: starts[s].0.len() as u64 }
            })
            .collect();
        let c = m.window_candidates(&RowBatch::from_requests(&reqs), COUNT).unwrap();
        out += &format!("batch{} window {:016x}\n", n_seq * 8, hash(&c));
        for s in 0..n_seq {
            m.commit(s, 1);
        }
        // the next window must start from the same state: replay the committed first rows into fresh prefills
        for (s, (toks, _)) in starts.iter().enumerate().take(n_seq) {
            prefill(&mut m, s, toks);
        }
    }
    eprint!("{out}");
    std::fs::File::create(path).unwrap().write_all(out.as_bytes()).unwrap();
}
