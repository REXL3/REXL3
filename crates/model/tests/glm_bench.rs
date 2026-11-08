//! Speed of the whole GLM-5.3-Flash model on the GB10: single-stream decode, the window step table (what `engine::STEP_MS` is read from),
//! prefill, and where a forward's time goes by op class. All `#[ignore]`, env-gated (REXL3_GLM_DIR for the pack, REXL3_GLM_ORACLE_DIR only for
//! real token ids to feed), one model per process, `--test-threads=1`, under the GPU lock. Numbers go to stderr; nothing here asserts a speed.

mod glm_support;

use gb_model::families::glm::model::GlmModel;
use gb_model::rows::RowBatch;
use glm_support::*;
use std::path::PathBuf;

const SLOTS: usize = 4;
const KV_BLOCKS: usize = 80; // 20,480 tokens
const MAX_CONTEXT: usize = 8192;

fn model() -> Option<std::sync::MutexGuard<'static, GlmModel>> {
    shared_model(SLOTS, KV_BLOCKS, MAX_CONTEXT)
}

/// `n` real token ids (the oracle's 2,278-token prompt, tiled): the speed of a token does not depend on its identity except through the MoE
/// routing, which real text exercises as it would be in serving.
fn tiled(n: usize) -> Option<Vec<u32>> {
    let dir = match std::env::var("REXL3_GLM_ORACLE_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            eprintln!("REXL3_GLM_ORACLE_DIR is not set: not run");
            return None;
        }
    };
    let ids = gb_model::families::glm::npy::read(&dir.join("long_2300").join("ids.npy")).unwrap().to_i32().unwrap();
    Some((0..n).map(|i| ids[i % ids.len()] as u32).collect())
}

fn req(slot: usize, pos: usize, toks: &[u32]) -> gb_engine::RowRequest {
    gb_engine::RowRequest { slot, tokens: toks.to_vec(), first_position: pos as u64 }
}

/// Prefill `ids` into `slot` in 128-row chunks; (seconds in all, milliseconds of the slowest chunk).
fn timed_prefill(m: &mut GlmModel, slot: usize, ids: &[u32]) -> (f64, f64) {
    m.begin_slot(slot).unwrap();
    let (mut at, mut worst) = (0, 0f64);
    let t0 = std::time::Instant::now();
    while at < ids.len() {
        let end = (at + m.chunk_rows()).min(ids.len());
        let t = std::time::Instant::now();
        m.prefill_chunk(slot, at, &ids[at..end], None).unwrap();
        worst = worst.max(t.elapsed().as_secs_f64() * 1e3);
        at = end;
    }
    (t0.elapsed().as_secs_f64(), worst)
}

/// Mean milliseconds of `n` windows after two warm-up ones; `reqs(i)` the requests of the i-th, each committed in full (a commit is applied
/// inside the next forward, so its cost is in the mean).
fn time_windows(m: &mut GlmModel, n: usize, reqs: &dyn Fn(usize) -> Vec<gb_engine::RowRequest>) -> f64 {
    let step = |m: &mut GlmModel, i: usize| {
        let rs = reqs(i);
        let b = RowBatch::from_requests(&rs);
        m.window_candidates(&b, 8).unwrap();
        for r in &rs {
            m.commit(r.slot, r.tokens.len());
        }
    };
    for i in 0..2 {
        step(m, i);
    }
    let t = std::time::Instant::now();
    for i in 2..2 + n {
        step(m, i);
    }
    t.elapsed().as_secs_f64() * 1e3 / n as f64
}

/// Decode speed, the window step table and prefill speed at three context depths: dense attention, just past the 2,048 regime boundary, deep.
#[test]
#[ignore]
fn bench_steps() {
    let Some(all) = tiled(6400) else { return };
    let Some(mut m) = model() else { return };
    for ctx in [128usize, 2300, 6144] {
        let ids = &all[..ctx];
        let (secs, worst) = timed_prefill(&mut m, 0, ids);
        eprintln!(
            "ctx {ctx}: prefill {:.1} s = {:.0} tok/s ({} chunks of 128, slowest {:.0} ms)",
            secs,
            ctx as f64 / secs,
            ctx.div_ceil(128),
            worst
        );
        // single-stream decode: real greedy tokens, one row at a time
        let mut tok = all[ctx];
        let mut times = Vec::new();
        for i in 0..28 {
            let t = std::time::Instant::now();
            let b = RowBatch::from_requests(&[req(0, ctx + i, &[tok])]);
            let c = m.window_candidates(&b, 8).unwrap();
            m.commit(0, 1);
            tok = c.ids[0] as u32;
            if i >= 4 {
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        eprintln!(
            "ctx {ctx}: decode R=1 {:.2} ms/step mean ({:.2} median, {:.2} min, {:.2} max) = {:.2} tok/s",
            mean,
            times[times.len() / 2],
            times[0],
            times[times.len() - 1],
            1e3 / mean
        );
        // the step table: one sequence, R rows per window
        let mut line = String::new();
        let mut next = ctx + 28;
        for r in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 32, 48, 64] {
            let base = next;
            let ms = time_windows(&mut m, 5, &|i| {
                let at = (base + i * r) % 6000;
                vec![req(0, base + i * r, &all[at..at + r])]
            });
            next += 7 * r;
            line += &format!(" R{r}={ms:.1}");
        }
        eprintln!("ctx {ctx}: window ms by rows:{line}");
    }
    // several sequences in one step: S slots at 256 tokens each, R rows apiece
    for s in 0..4 {
        timed_prefill(&mut m, s, &all[..256]);
    }
    let mut line = String::new();
    let mut base = [256usize; 4];
    for (s, r) in [(1usize, 1usize), (2, 1), (3, 1), (4, 1), (2, 2), (4, 2), (4, 4), (4, 8)] {
        let start = base;
        let ms = time_windows(&mut m, 5, &|i| {
            (0..s)
                .map(|k| {
                    let at = (start[k] + i * r) % 5000;
                    req(k, start[k] + i * r, &all[at..at + r])
                })
                .collect()
        });
        for b in base.iter_mut().take(s) {
            *b += 7 * r;
        }
        line += &format!(" {s}x{r}={ms:.1}");
    }
    eprintln!("ctx ~256: window ms by slots x rows:{line}");
}

/// Where a forward's time goes by op class (CUDA events around every op of the schedule): decode at two depths, a window of 8, a prefill.
#[test]
#[ignore]
fn bench_profile() {
    let Some(all) = tiled(2600) else { return };
    let Some(mut m) = model() else { return };
    for ctx in [128usize, 2300] {
        timed_prefill(&mut m, 0, &all[..ctx]);
        for rows in [1usize, 8] {
            m.profile = Some(Default::default());
            let mut pos = ctx;
            for _ in 0..6 {
                let b = RowBatch::from_requests(&[req(0, pos, &all[pos..pos + rows])]);
                m.window_candidates(&b, 8).unwrap();
                m.commit(0, rows);
                pos += rows;
            }
            report_profile(&mut m, &format!("decode window of {rows} row(s) at context {ctx}"));
        }
        m.profile = Some(Default::default());
        timed_prefill(&mut m, 0, &all[..ctx]);
        report_profile(&mut m, &format!("prefill of a {ctx}-token prompt in 128-row chunks"));
    }
}

/// For `nsys profile --capture-range=cudaProfilerApi`: 40 single-row decode steps at context ~256 and 20 windows of 8, inside the profiler range
/// (the prefill, the warm-up and the weight load are outside it). Analyse with `tools/nsys_decode_gaps.py` (kernel count and busy time per step).
#[test]
#[ignore]
fn bench_decode_trace() {
    let Some(all) = tiled(400) else { return };
    let Some(mut m) = model() else { return };
    timed_prefill(&mut m, 0, &all[..256]);
    let mut pos = 256;
    for rows in [1usize, 8] {
        for _ in 0..6 {
            let b = RowBatch::from_requests(&[req(0, pos, &all[pos..pos + rows])]);
            m.window_candidates(&b, 8).unwrap();
            m.commit(0, rows);
            pos += rows;
        }
        let steps = if rows == 1 { 40 } else { 20 };
        // nsys needs a moment after a range ends to arm the next one: without the pause the second range misses its first 14 steps
        std::thread::sleep(std::time::Duration::from_secs(5));
        gb_cuda::rt::profiler_start();
        let t = std::time::Instant::now();
        for _ in 0..steps {
            let b = RowBatch::from_requests(&[req(0, pos, &all[pos..pos + rows])]);
            m.window_candidates(&b, 8).unwrap();
            m.commit(0, rows);
            pos += rows;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
        gb_cuda::rt::profiler_stop();
        eprintln!("traced {steps} windows of {rows} row(s): {ms:.2} ms each (under the profiler)");
    }
}

fn report_profile(m: &mut GlmModel, what: &str) {
    let p = m.profile.take().unwrap();
    let total: f64 = p.ms.values().sum();
    eprintln!("profile: {what}: {} forwards, {:.2} ms each", p.forwards, total / p.forwards as f64);
    let mut v: Vec<_> = p.ms.iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (k, ms) in v {
        eprintln!("    {k:32} {:8.2} ms  {:5.1} %", ms / p.forwards as f64, 100.0 * ms / total);
    }
    // the steps inside the classes
    for (k, ms) in &p.sub {
        eprintln!("      . {k:62} {:8.2} ms  {:5.1} %", ms / p.forwards as f64, 100.0 * ms / total);
    }
}
