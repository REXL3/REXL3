//! Old against new on the server's own chunk grid: the golden hashing of `golden.rs` for prompt lengths on both sides of
//! the changed band (a prompt of more than 4096 tokens whose last chunk has 1..=128 rows), written to a file so a c655bea
//! build and the pinned-GEMM build can be diffed. The one line that differs between the two trees is `pc`.
//!
//!     REXL3_BAND_OUT=band.txt REXL3_QWEN_DIR=... cargo test --release -p gb-model --test golden_band -- --ignored --nocapture

use gb_engine::RowRequest;
use gb_model::config::QwenConfig;
use gb_model::forward::{Candidates, Model};
use gb_model::pack::Pack;
use gb_model::rows::RowBatch;
use gb_model::weights::Weights;
use std::io::Write;

const COUNT: usize = 28;
const CHUNK: usize = 4096;

fn pc(m: &mut Model, start: usize, toks: &[u32], total: usize, want: Option<usize>) -> Option<Candidates> {
    m.prefill_chunk(0, start, toks, total, want, 0).unwrap()
}

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

#[test]
#[ignore]
fn golden_on_both_sides_of_the_band() {
    let (Ok(d), Ok(out)) = (std::env::var("REXL3_QWEN_DIR"), std::env::var("REXL3_BAND_OUT")) else {
        return eprintln!("REXL3_QWEN_DIR and REXL3_BAND_OUT not set: skipping");
    };
    if !gb_cuda::device_available() {
        return eprintln!("no CUDA device: skipping");
    }
    let d = std::path::PathBuf::from(d);
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    let mut m = Model::new(cfg, w, 2, 128, 16384, CHUNK).unwrap();
    let lengths: Vec<usize> = std::env::var("REXL3_BAND_LENGTHS")
        .ok()
        .map(|s| s.split(',').map(|t| t.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![100, 129, 300, 4096, 4097, 4099, 4200, 4224, 4225, 5000, 8193, 8300, 9000]);
    let mut text = String::new();
    for len in lengths {
        let toks = prompt(len, len as u64);
        m.begin_slot(0).unwrap();
        let mut last = None;
        for start in (0..len).step_by(CHUNK) {
            let end = (start + CHUNK).min(len);
            last = pc(&mut m, start, &toks[start..end], len, (end == len).then_some(COUNT));
        }
        let c = last.unwrap();
        let band = len > CHUNK && (1..=128).contains(&(len % CHUNK));
        text += &format!("p{len} prefill {:016x} top {:?}  # band: {band}\n", hash(&c), &c.ids[..4]);
        let mut t = c.ids[0];
        let mut greedy = vec![t];
        for k in 0..12 {
            let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: vec![t as u32], first_position: (len + k) as u64 }]);
            let c = m.window_candidates(&b, COUNT).unwrap();
            m.commit(0, 1);
            if k < 4 {
                text += &format!("p{len} decode{k} {:016x}\n", hash(&c));
            }
            t = c.ids[0];
            greedy.push(t);
        }
        text += &format!("p{len} greedy {greedy:?}\n");
    }
    eprint!("{text}");
    std::fs::File::create(out).unwrap().write_all(text.as_bytes()).unwrap();
}
