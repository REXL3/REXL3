//! Prefill of large chunks on the whole GLM-5.3-Flash model (anonymized host-b or anonymized host-d, 87 GiB resident): a chunk of more than 128 rows runs the large-row kernels
//! (`gb_cuda::exl3_big`, `gb_cuda::moe_prefill`) and slabbed attention, and every row gets the bits the decode path gives it. All `#[ignore]` and env-gated:
//! REXL3_GLM_DIR (the pack), REXL3_GLM_ORACLE_DIR (real token ids: `long_2300`), REXL3_GLM_CHUNK_ROWS (the chunk the model is built for, at least the
//! largest chunk of the test: 1024 default), REXL3_GLM_SLOTS (default 3). One model per process: `--test-threads=1`, under the GPU lock.
//!
//! * `big_chunks_equal_the_decode_path`: the same prompt through many chunk grids (small chunks run the decode kernels, large ones the large-row
//!   path) leaves identical state and identical next-token logits, and decodes on identically.
//! * `bench_prefill_by_chunk`: tokens per second and the time per op class for chunks of 128 .. 2,048 rows.

mod glm_support;

use gb_model::families::glm::model::{GlmModel, Head, Mode};
use gb_model::families::glm::npy;
use gb_model::rows::RowBatch;
use glm_support::*;
use std::path::PathBuf;

const KV_BLOCKS: usize = 80; // 20,480 tokens
const MAX_CONTEXT: usize = 8192;

fn slots() -> usize {
    std::env::var("REXL3_GLM_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(3)
}

fn model() -> Option<std::sync::MutexGuard<'static, GlmModel>> {
    let mut m = shared_model(slots(), KV_BLOCKS, MAX_CONTEXT)?;
    m.set_prefill_contract(gb_model::families::glm::contract::PrefillContract::P1)
        .expect("the big-chunk test runs the P1 large-row path");
    Some(m)
}

fn all_ids() -> Option<Vec<u32>> {
    let dir = match std::env::var("REXL3_GLM_ORACLE_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            eprintln!("REXL3_GLM_ORACLE_DIR is not set: not run");
            return None;
        }
    };
    let ids = npy::read(&dir.join("long_2300").join("ids.npy")).unwrap().to_i32().unwrap();
    Some(ids.iter().map(|&x| x as u32).collect())
}

/// `n` real token ids (the oracle's 2,278-token prompt, tiled).
fn tiled(n: usize) -> Option<Vec<u32>> {
    let ids = all_ids()?;
    Some((0..n).map(|i| ids[i % ids.len()]).collect())
}

type KdaState = Vec<(Vec<u32>, Vec<u16>)>;

/// Everything a prompt leaves: the last row's logits bits, the committed state of every layer, and what decoding on from it gives.
#[derive(PartialEq)]
struct Snap {
    last_logits: Vec<u16>,
    top: (Vec<i32>, Vec<u32>),
    kda: KdaState,
    latent: Vec<Vec<u16>>,
    pools: Vec<Vec<u16>>,
    decoded: Vec<Vec<u16>>,
}

fn state(m: &mut GlmModel, slot: usize, committed: usize) -> (KdaState, Vec<Vec<u16>>, Vec<Vec<u16>>) {
    let (mut kda, mut latent, mut pools) = (Vec::new(), Vec::new(), Vec::new());
    for l in 0..45 {
        if l % 4 == 3 {
            latent.push(m.latent_rows(l, slot, committed).unwrap());
            pools.push(m.pool_rows(l, slot, committed / 4).unwrap());
        } else {
            let (s, c) = m.kda_state(l, slot).unwrap();
            kda.push((s.iter().map(|x| x.to_bits()).collect(), c));
        }
    }
    (kda, latent, pools)
}

/// Prefill `ids` into `slot` cutting it into chunks of the sizes in `grid` (the last size repeats), then decode `decode` tokens greedily.
fn run_grid(m: &mut GlmModel, slot: usize, ids: &[u32], grid: &[usize], decode: usize) -> Snap {
    m.begin_slot(slot).unwrap();
    let (mut at, mut i) = (0, 0);
    let mut last = None;
    while at < ids.len() {
        let size = grid[i.min(grid.len() - 1)];
        i += 1;
        let end = (at + size).min(ids.len());
        let want = (end == ids.len()).then_some(8usize);
        last = m.prefill_chunk(slot, at, &ids[at..end], want).unwrap();
        at = end;
    }
    let c = last.unwrap();
    let last_logits = m.logits_host(0, 1).unwrap();
    let (kda, latent, pools) = state(m, slot, ids.len());
    let mut decoded = Vec::new();
    let mut tok = c.ids[0] as u32;
    for j in 0..decode {
        let b =
            RowBatch::from_requests(&[gb_engine::RowRequest { slot, tokens: vec![tok], first_position: (ids.len() + j) as u64 }]);
        m.forward(&b, Mode::Window, Head::AllRows).unwrap();
        decoded.push(m.logits_host(0, 1).unwrap());
        let cand = m.candidates_of(1, 1).unwrap();
        m.commit(slot, 1);
        tok = cand.ids[0] as u32;
    }
    Snap { last_logits, top: (c.ids.clone(), c.vals.iter().map(|v| v.to_bits()).collect()), kda, latent, pools, decoded }
}

fn describe(a: &Snap, b: &Snap) -> String {
    let mut d = Vec::new();
    if a.last_logits != b.last_logits {
        d.push(format!(
            "last-row logits differ in {} of {} elements",
            a.last_logits.iter().zip(&b.last_logits).filter(|(x, y)| x != y).count(),
            a.last_logits.len()
        ));
    }
    if a.top != b.top {
        d.push("top candidates differ".into());
    }
    for (l, (x, y)) in a.kda.iter().zip(&b.kda).enumerate() {
        if x.0 != y.0 {
            d.push(format!("KDA state of KDA layer #{l} differs"));
            break;
        }
        if x.1 != y.1 {
            d.push(format!("KDA conv of KDA layer #{l} differs"));
            break;
        }
    }
    if a.latent != b.latent {
        let l = a.latent.iter().zip(&b.latent).position(|(x, y)| x != y).unwrap();
        d.push(format!("MLA latent rows of MLA layer #{l} differ"));
    }
    if a.pools != b.pools {
        let l = a.pools.iter().zip(&b.pools).position(|(x, y)| x != y).unwrap();
        d.push(format!("DSA pool rows of MLA layer #{l} differ"));
    }
    if a.decoded != b.decoded {
        let j = a.decoded.iter().zip(&b.decoded).position(|(x, y)| x != y).unwrap();
        d.push(format!("the logits of decode step {j} after the prompt differ"));
    }
    d.join("; ")
}

#[test]
#[ignore]
fn big_chunks_equal_the_decode_path() {
    let Some(all) = all_ids() else { return };
    let Some(mut m) = model() else { return };
    let max = m.chunk_rows();
    // 1,500: dense attention throughout; 2,278: the later rows are past the 2,048 boundary where the attention turns sparse
    for n in [1500usize, 2278] {
        let ids = &all[..n];
        let reference = run_grid(&mut m, 0, ids, &[128], 6);
        eprintln!("prompt of {n} tokens: reference = 128-row chunks (the decode kernels)");
        for grid in [
            vec![max.min(1024)],
            vec![max.min(2048)],
            vec![384],
            vec![129],
            vec![256],
            vec![max.min(700), 129, 300],
            vec![1, max.min(1023), 129],
            vec![100, max.min(900), 5, 128],
        ] {
            let t = std::time::Instant::now();
            let got = run_grid(&mut m, 0, ids, &grid, 6);
            let why = describe(&got, &reference);
            assert!(why.is_empty(), "prompt of {n}, chunk grid {grid:?}: {why}");
            eprintln!("  grid {grid:?}: bit-identical to the reference (state of 45 layers, last-row logits, 6 decode steps) in {:.1} s", t.elapsed().as_secs_f64());
        }
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
    for (k, ms) in &p.sub {
        eprintln!("      . {k:62} {:8.2} ms  {:5.1} %", ms / p.forwards as f64, 100.0 * ms / total);
    }
}

/// Tokens per second of a prompt of REXL3_GLM_BENCH_TOKENS (default 4,096) tokens by chunk size, and where a chunk's time goes.
#[test]
#[ignore]
fn bench_prefill_by_chunk() {
    let n: usize = std::env::var("REXL3_GLM_BENCH_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(4096);
    let Some(ids) = tiled(n) else { return };
    let Some(mut m) = model() else { return };
    let max = m.chunk_rows();
    let sizes: Vec<usize> = [128usize, 256, 512, 1024, 2048].into_iter().filter(|&c| c <= max).collect();
    for &c in &sizes {
        // one warm-up chunk, then the whole prompt timed
        m.begin_slot(0).unwrap();
        m.prefill_chunk(0, 0, &ids[..c.min(n)], None).unwrap();
        m.begin_slot(0).unwrap();
        let t = std::time::Instant::now();
        let mut at = 0;
        while at < n {
            let end = (at + c).min(n);
            m.prefill_chunk(0, at, &ids[at..end], None).unwrap();
            at = end;
        }
        let secs = t.elapsed().as_secs_f64();
        eprintln!(
            "BENCH chunk {c:5}: {n} tokens in {secs:6.2} s = {:7.1} tok/s ({:.0} ms a chunk of {c})",
            n as f64 / secs,
            secs * 1e3 / (n as f64 / c as f64)
        );
        // the profile of one chunk past the first (position 2,048 or later: sparse attention in the later rows)
        m.begin_slot(0).unwrap();
        let warm = (2048 / c) * c;
        let mut at = 0;
        while at < warm.min(n) {
            let end = (at + c).min(n);
            m.prefill_chunk(0, at, &ids[at..end], None).unwrap();
            at = end;
        }
        if at < n {
            m.profile = Some(Default::default());
            for _ in 0..2 {
                if at >= n {
                    break;
                }
                let end = (at + c).min(n);
                m.prefill_chunk(0, at, &ids[at..end], None).unwrap();
                at = end;
            }
            report_profile(&mut m, &format!("a prefill chunk of {c} rows at positions {}..{at}", warm));
        }
    }
}

/// For `nsys profile --capture-range=cudaProfilerApi` (`tools/glm_prefill_nsys.sh`): two chunks of `REXL3_GLM_TRACE_CHUNK` rows (default: the model's) at
/// positions 2,048 and on (the sparse attention regime) inside the profiler range; the prompt before them is outside it. Analyse with
/// `tools/nsys_decode_gaps.py OUT.sqlite --steps 2` (kernel time by name).
#[test]
#[ignore]
fn bench_prefill_trace() {
    let m0 = model();
    let Some(mut m) = m0 else { return };
    let c: usize =
        std::env::var("REXL3_GLM_TRACE_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(m.chunk_rows()).min(m.chunk_rows());
    let warm = (2048 / c).max(1) * c;
    let Some(ids) = tiled(warm + 2 * c) else { return };
    m.begin_slot(0).unwrap();
    let mut at = 0;
    while at < warm {
        m.prefill_chunk(0, at, &ids[at..at + c], None).unwrap();
        at += c;
    }
    // nsys needs a moment to arm the range
    std::thread::sleep(std::time::Duration::from_secs(5));
    gb_cuda::rt::profiler_start();
    let t = std::time::Instant::now();
    for _ in 0..2 {
        m.prefill_chunk(0, at, &ids[at..at + c], None).unwrap();
        at += c;
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / 2.0;
    gb_cuda::rt::profiler_stop();
    eprintln!("traced 2 chunks of {c} rows at positions {warm}..{at}: {ms:.1} ms each (under the profiler)");
}
