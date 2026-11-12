//! The P4-TAIL ceiling probe on the real pack (`docs/TAIL_FLOOR_CEILING.md`): the W_q reconstruct floor of the tail of a
//! prefill, measured where it is paid. Three things, medians of `REXL3_BENCH_RUNS` runs (default 5):
//!
//! 1. the wall time of every chunk of a 32,768-token prompt (chunk 4,096, the served default): the first chunks are the
//!    steady state, the last is the tail;
//! 2. a tail-sized chunk alone at 16/64/128/256/512/4096 rows (the same `total`, so the same pinned path): the floor's
//!    shape, and what the 400 reconstructs cost against the rows that need them;
//! 3. a cached re-ask: restore from a snapshot taken 16 tokens before the end of a 32k prompt, then the 16-row tail
//!    chunk — the path whose TTFT the floor sits under.
//!
//! A measurement without assertions, skipped by the gate (the name has `bench`). No `nsys` here: the chunk walls are
//! cudaEvent-free host walls around synchronising calls, the same method `prefix_cache_bench.rs` uses.
//!
//!     REXL3_QWEN_DIR=... REXL3_DRAFT_DIR=... flock /tmp/rexl3-gpu.lock cargo test --release -p gb-model \
//!         --test tail_floor_bench -- --ignored --nocapture

use gb_model::forward::Model;
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
fn bench_tail_floor() {
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
    o.max_model_len = 36_864;
    o.kv_tokens = Some(40_960);
    o.vision = None;
    o.prefix_cache = PrefixCacheOption::Mib(3000);
    let mut e = build_engine(&o).unwrap();
    let m = &mut e.backend.m;
    let chunk = e.backend.chunk;
    eprintln!("chunk {chunk}, runs {runs}");
    // the W_q residency (REXL3_WQ_RESIDENT_GIB, off unless set): its counters after the runs tell what was saved
    let wq_on = m.wq_stats().budget_bytes > 0;
    if wq_on {
        eprintln!(
            "W_q residency: budget {:.2} GiB ({} of the 400 matrices' 45.31 GiB)",
            m.wq_stats().budget_bytes as f64 / (1u64 << 30) as f64,
            m.wq_stats().resident
        );
    }

    // 1. every chunk of a 32,768-token prompt (the draft context is the last CTX_KEEP tokens: the first chunk feeds it too)
    let p32 = 32_768usize;
    let prompt = random_prompt(p32, 32);
    let ctx_from = p32 - gb_model::draft::CTX_KEEP;
    let mut per_chunk: Vec<Vec<f64>> = vec![Vec::new(); p32.div_ceil(chunk)];
    for run in 0..runs + 1 {
        m.begin_slot(0).unwrap();
        let mut at = 0;
        let mut ci = 0;
        while at < p32 {
            let end = (at + chunk).min(p32);
            let want = (end == p32).then_some(CAND);
            let t = ms(m, |m| {
                let c = m.prefill_chunk(0, at, &prompt[at..end], p32, want, ctx_from).unwrap();
                if want.is_some() {
                    c.unwrap();
                }
            });
            if run > 0 {
                per_chunk[ci].push(t);
            }
            at = end;
            ci += 1;
        }
        m.end_slot(0);
    }
    eprintln!("\n| chunk | rows | ms (median) |");
    eprintln!("| ---: | ---: | ---: |");
    for (ci, v) in per_chunk.iter().enumerate() {
        let rows = chunk.min(p32 - ci * chunk);
        eprintln!("| {} | {rows} | {:.1} |", ci + 1, median(v.clone()));
    }
    let walls: Vec<f64> = per_chunk.iter().map(|v| median(v.clone())).collect();
    eprintln!(
        "32,768-token prefill: {:.0} ms over {} chunks; steady chunks (1..n) {:.1} ms, tail {:.1} ms",
        walls.iter().sum::<f64>(),
        walls.len(),
        walls[..walls.len() - 1].iter().sum::<f64>() / (walls.len() - 1) as f64,
        walls[walls.len() - 1]
    );

    // 2. a tail-sized chunk alone (total still 32,768, the chunk at the prompt's end, so the pinned path a 32k prompt's
    //    tail really takes; `PrefillPath` routes by the prompt's total length, not the chunk's rows)
    eprintln!("\n| tail rows | ms | ms/row |");
    eprintln!("| ---: | ---: | ---: |");
    for rows in [16usize, 64, 128, 256, 512, 4096] {
        let start = p32 - rows;
        let mut v = Vec::new();
        for run in 0..runs + 1 {
            let t = ms(m, |m| {
                m.begin_slot(0).unwrap();
                m.prefill_chunk(0, start, &prompt[start..], p32, Some(CAND), ctx_from).unwrap().unwrap();
            });
            if run > 0 {
                v.push(t);
            }
        }
        eprintln!("| {rows} | {:.1} | {:.3} |", median(v.clone()), median(v) / rows as f64);
    }

    // 3. a cached re-ask: snapshot 16 tokens before the end of a 32k prompt, restore, run the 16-row tail
    let cut = p32 - 16;
    let snap = m.snaps.as_mut().unwrap().alloc().unwrap();
    let ctx_cut = ctx_from.min(cut.saturating_sub(2048));
    {
        m.begin_slot(0).unwrap();
        // the cut lies inside the LAST chunk (rows 28,672..32,768): the snapshot is taken there, as the server's
        // cut policy does, so no second W_q reconstruct is paid for it
        let mut at = 0;
        while at < p32 {
            let end = (at + chunk).min(p32);
            let want = (end == p32).then_some(CAND);
            let cuts = if end == p32 { vec![gb_model::forward::SnapCut { at: cut, snap }] } else { Vec::new() };
            let c = m
                .prefill_chunk_snap(
                    0,
                    at,
                    &prompt[at..end],
                    p32,
                    want,
                    if end == p32 { ctx_cut } else { ctx_from },
                    None,
                    &[],
                    &cuts,
                )
                .unwrap();
            if want.is_some() {
                c.unwrap();
            }
            at = end;
        }
    }
    let mut restore_v = Vec::new();
    let mut reask_v = Vec::new();
    for run in 0..runs + 1 {
        let pages = m.retain_prefix_pages(0, cut);
        let t_restore = ms(m, |m| {
            m.restore_prefix(1, snap, &pages, cut).unwrap();
        });
        let t_reask = ms(m, |m| {
            m.prefill_chunk(1, cut, &prompt[cut..], p32, Some(CAND), ctx_from).unwrap().unwrap();
        });
        m.end_slot(1);
        m.release_pages(&pages);
        if run > 0 {
            restore_v.push(t_restore);
            reask_v.push(t_reask);
        }
    }
    let (restore_med, reask_med) = (median(restore_v), median(reask_v));
    eprintln!(
        "\ncached re-ask after a 32k prompt: restore {restore_med:.1} ms + 16-row tail chunk {reask_med:.1} ms = {:.1} ms",
        restore_med + reask_med
    );
    if wq_on {
        let s = m.wq_stats();
        eprintln!(
            "W_q residency at the end: hits {} fills {} evictions {} resident {} ({} GiB) of a {:.2} GiB budget",
            s.hits,
            s.misses,
            s.evictions,
            s.resident,
            s.resident_bytes as f64 / (1u64 << 30) as f64,
            s.budget_bytes as f64 / (1u64 << 30) as f64
        );
    }
}
