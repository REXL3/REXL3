//! Prefill time on the real model by prompt length: whole-prompt time to the first token's candidates (the chunks of the
//! server's grid, 4096 rows) and tokens per second, median of several runs. The same file builds against c655bea (where
//! `prefill_chunk` has no `total`) and against the pinned-GEMM tree, so the two can be run back to back on an idle GPU.
//!
//!     REXL3_QWEN_DIR=... cargo test --release -p gb-model --test prefill_bench -- --ignored --nocapture
//!
//! `REXL3_BENCH_LENGTHS` (comma separated) and `REXL3_BENCH_REPS` override the defaults.

use gb_model::config::QwenConfig;
use gb_model::forward::{Candidates, Model};
use gb_model::pack::Pack;
use gb_model::weights::Weights;
use std::time::Instant;

const CHUNK: usize = 4096;

/// The one line that differs between the tree under test and c655bea.
fn pc(m: &mut Model, start: usize, toks: &[u32], total: usize, want: Option<usize>) -> Option<Candidates> {
    m.prefill_chunk(0, start, toks, total, want, 0).unwrap()
}

fn prompt(len: usize) -> Vec<u32> {
    let mut x = (len as u64).wrapping_mul(0x9E3779B97F4A7C15) | 1;
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
fn bench_prefill_time_by_prompt_length() {
    let Ok(d) = std::env::var("REXL3_QWEN_DIR") else { return eprintln!("REXL3_QWEN_DIR not set: skipping") };
    if !gb_cuda::device_available() {
        return eprintln!("no GPU: skipping");
    }
    let d = std::path::PathBuf::from(d);
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    let mut m = Model::new(cfg, w, 2, 160, 40960, CHUNK).unwrap();
    let lengths: Vec<usize> = std::env::var("REXL3_BENCH_LENGTHS")
        .ok()
        .map(|s| s.split(',').map(|t| t.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![100, 129, 300, 1000, 2000, 4096, 4097, 4200, 8192, 16384]);
    let reps: usize = std::env::var("REXL3_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let run = |m: &mut Model, toks: &[u32]| -> f64 {
        m.begin_slot(0).unwrap();
        let t = Instant::now();
        let mut start = 0;
        while start < toks.len() {
            let end = (start + CHUNK).min(toks.len());
            pc(m, start, &toks[start..end], toks.len(), (end == toks.len()).then_some(28));
            start = end;
        }
        t.elapsed().as_secs_f64() * 1000.0
    };
    run(&mut m, &prompt(300)); // warm up: cuBLAS handles, kernels, allocator
    println!("{:>7} {:>10} {:>10} {:>10} {:>9}", "tokens", "median ms", "min ms", "max ms", "tok/s");
    for &t in &lengths {
        let toks = prompt(t);
        let mut ms: Vec<f64> = (0..reps).map(|_| run(&mut m, &toks)).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ms[ms.len() / 2];
        println!("{:>7} {:>10.1} {:>10.1} {:>10.1} {:>9.0}", t, med, ms[0], ms[ms.len() - 1], t as f64 / (med / 1000.0));
    }
}
