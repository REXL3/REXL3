//! What a prefix-cache snapshot and a restore cost on the GB10 (numbers for `docs/PREFIX_CACHE.md` and `docs/EVAL_RESULTS.md`): the prefill of one
//! prompt with no cut, with a snapshot taken INSIDE the chunk (the GDN recurrence in two parts, a copy of the state between them), and with the
//! chunk split at the cut (the way a cut would be taken without that: a second chunk, a second W_q reconstruct pass); and the time of a restore.
//! Medians of `REXL3_BENCH_RUNS` runs (default 5). A measurement without assertions, skipped by the gate (the name has `bench`).
//!
//!     REXL3_QWEN_DIR=... REXL3_DRAFT_DIR=... cargo test --release -p gb-model --test prefix_cache_bench -- --ignored --nocapture   (the test name has `bench`, which the gate skips)

use gb_model::forward::{Model, SnapCut};
use gb_model::serve::{build_engine, PrefixCacheOption, ServeOptions};
use std::time::Instant;

const CAND: usize = 20;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
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

/// Run `f` (which ends with a synchronising read of the candidates) and return its milliseconds.
fn ms(m: &mut Model, f: impl FnOnce(&mut Model)) -> f64 {
    m.sync().unwrap();
    let t = Instant::now();
    f(m);
    m.sync().unwrap();
    t.elapsed().as_secs_f64() * 1e3
}

#[test]
#[ignore]
fn bench_snapshot_and_restore_cost() {
    let (Ok(q), Ok(d)) = (std::env::var("REXL3_QWEN_DIR"), std::env::var("REXL3_DRAFT_DIR")) else {
        return eprintln!("not run: needs REXL3_QWEN_DIR and REXL3_DRAFT_DIR");
    };
    if !gb_cuda::device_available() {
        return eprintln!("not run: no GPU");
    }
    let runs: usize = std::env::var("REXL3_BENCH_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let mut o = ServeOptions::new(q.into());
    o.draft_dir = Some(d.into());
    o.slots = 2;
    o.max_model_len = 12288;
    o.kv_tokens = Some(24576);
    o.vision = None;
    o.prefix_cache = PrefixCacheOption::Mib(3000);
    let mut e = build_engine(&o).unwrap();
    let m = &mut e.backend.m;
    let snap = m.snaps.as_mut().unwrap().alloc().unwrap();
    println!("\n| prompt | cut | no cut (ms) | cut inside the chunk (ms) | chunk split at the cut (ms) | inside vs none | split vs none | restore (ms) |");
    println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for p in [200usize, 600, 1000, 2000, 3400, 4096] {
        let prompt = random_prompt(p, p as u64);
        let c = p - 16;
        let ctx_plain = p.saturating_sub(gb_model::draft::CTX_KEEP);
        let ctx_cut = ctx_plain.min(c.saturating_sub(2048));
        let (mut plain, mut inside, mut split, mut restore) = (vec![], vec![], vec![], vec![]);
        for run in 0..runs + 1 {
            let t_plain = ms(m, |m| {
                m.begin_slot(0).unwrap();
                m.prefill_chunk(0, 0, &prompt, p, Some(CAND), ctx_plain).unwrap().unwrap();
            });
            let t_inside = ms(m, |m| {
                m.begin_slot(0).unwrap();
                m.prefill_chunk_snap(0, 0, &prompt, p, Some(CAND), ctx_cut, None, &[], &[SnapCut { at: c, snap }])
                    .unwrap()
                    .unwrap();
            });
            let pages = m.retain_prefix_pages(0, c);
            let t_split = ms(m, |m| {
                m.begin_slot(1).unwrap();
                m.prefill_chunk(1, 0, &prompt[..c], p, None, ctx_plain).unwrap();
                m.prefill_chunk(1, c, &prompt[c..], p, Some(CAND), ctx_plain).unwrap().unwrap();
            });
            let t_restore = ms(m, |m| m.restore_prefix(1, snap, &pages, c).unwrap());
            m.release_pages(&pages);
            if run > 0 {
                // the first run warms the clocks, the allocator and the pinned GEMM's workspaces
                plain.push(t_plain);
                inside.push(t_inside);
                split.push(t_split);
                restore.push(t_restore);
            }
        }
        let (a, b, s, r) = (median(plain), median(inside), median(split), median(restore));
        println!(
            "| {p} | {c} | {a:.1} | {b:.1} | {s:.1} | {:+.1} ms ({:+.2}%) | {:+.1} ms ({:+.1}%) | {r:.2} |",
            b - a,
            (b - a) / a * 100.0,
            s - a,
            (s - a) / a * 100.0
        );
    }
}
