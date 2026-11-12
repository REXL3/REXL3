//! Phase 4: the prefill's arithmetic is picked by the prompt's TOTAL length and is row invariant, so a prompt's bits do not
//! depend on how it is cut into chunks. Real model, real pack.
//!
//! Before this change two things made a row's bits depend on the chunk it was in: `Model::lin` routed a prefill linear on the
//! chunk's row count (chunks of 128 rows or fewer took the decode GEMV, larger ones W_q reconstruct plus `cublasGemmEx`, so the
//! 1..=128-row tail of a prompt of 4096 q + r tokens took another arithmetic than the rest of the prompt), and `cublasGemmEx`
//! itself left the large-M bits for k_proj/v_proj (K=5120, N=1024) at 193..=248 and 385..=391 rows. The measurements are in the
//! commit messages and in docs/EVAL_RESULTS.md ("Phase 4").
//!
//! Tests (all `#[ignore]`, they need `REXL3_QWEN_DIR` and a GPU; the gate runs them):
//! * `prefill_rows_do_not_depend_on_the_chunk_grid`: every layer's output of every row, whole prompt against many chunk
//!   grids, tails of 1 to 128 rows included, bit for bit.
//! * `chunked_prefill_equals_whole_on_the_real_model`: the engine harness's `chunked_prefill_equals_whole` on the real
//!   backend at arbitrary split points (non-multiples of 4096, pieces and tails of 1..=129 rows, a piece longer than a chunk),
//!   last prefill row and decoded tokens.
//! * `pinned_prefill_is_row_invariant_on_the_real_pack`: the M = 1..4096 sweep on the pack's own weights for every prefill
//!   shape, and the check that the shape list the cuda crate tests is the pack's.
//!
//!     REXL3_QWEN_DIR=... [REXL3_PROMPT_IDS=ids.json] cargo test --release -p gb-model --test prefill_route -- --ignored --nocapture

use gb_cuda::exl3_linear::sweep;
use gb_engine::harness::chunked_prefill_equals_whole;
use gb_engine::{Backend, RowOutput, RowRequest};
use gb_model::backend::QwenBackend;
use gb_model::config::QwenConfig;
use gb_model::forward::{Candidates, Model};
use gb_model::pack::Pack;
use gb_model::weights::Weights;
use gb_sampling::Sampling;
use std::cell::RefCell;
use std::rc::Rc;

const COUNT: usize = 28;
const CHUNK: usize = 4096;

fn dir() -> Option<std::path::PathBuf> {
    let d = std::path::PathBuf::from(std::env::var("REXL3_QWEN_DIR").ok()?);
    gb_cuda::device_available().then_some(d)
}

fn load_weights(d: &std::path::Path) -> (QwenConfig, Weights) {
    let cfg = QwenConfig::from_dir(d).unwrap();
    let w = Weights::load(&Pack::open(d).unwrap(), &cfg, |_, _| {}).unwrap();
    (cfg, w)
}

fn load_model() -> Option<Model> {
    let d = dir()?;
    let (cfg, w) = load_weights(&d);
    Some(Model::new(cfg, w, 2, 160, 24576, CHUNK).unwrap())
}

/// A fixed pseudo-random prompt of ordinary text tokens.
fn random_prompt(len: usize, seed: u64) -> Vec<u32> {
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

/// Real text, tokenised outside (a JSON array of ids), when `REXL3_PROMPT_IDS` names a file with enough of it.
fn real_prompt(len: usize) -> Option<Vec<u32>> {
    let text = std::fs::read_to_string(std::env::var("REXL3_PROMPT_IDS").ok()?).ok()?;
    let ids: Vec<u32> =
        text.trim().trim_matches(|c| c == '[' || c == ']').split(',').filter_map(|s| s.trim().parse().ok()).collect();
    (ids.len() >= len).then(|| ids[..len].to_vec())
}

/// Every layer's output rows of a prefill (concatenated over its chunks) and the last row's candidates.
struct Run {
    /// `[layer][row * hidden + i]`, bf16 bits
    layers: Vec<Vec<u16>>,
    cand: Candidates,
}

/// Prefill `tokens` (a prompt of `tokens.len()` tokens) into slot 0 in chunks of the given lengths, capturing every layer's
/// output rows.
fn run(m: &mut Model, tokens: &[u32], chunks: &[usize]) -> Run {
    assert_eq!(chunks.iter().sum::<usize>(), tokens.len());
    m.begin_slot(0).unwrap();
    m.capture = true;
    let mut layers: Vec<Vec<u16>> = Vec::new();
    let mut start = 0;
    let mut cand = None;
    for (i, &n) in chunks.iter().enumerate() {
        let last = i + 1 == chunks.len();
        cand = m.prefill_chunk(0, start, &tokens[start..start + n], tokens.len(), last.then_some(COUNT), 0).unwrap();
        if layers.is_empty() {
            layers = vec![Vec::new(); m.captured.len()];
        }
        for (l, rows) in layers.iter_mut().zip(&m.captured) {
            l.extend_from_slice(rows);
        }
        start += n;
    }
    m.capture = false;
    Run { layers, cand: cand.unwrap() }
}

fn same_cand(a: &Candidates, b: &Candidates) -> bool {
    a.ids == b.ids && a.vals.iter().map(|v| v.to_bits()).eq(b.vals.iter().map(|v| v.to_bits()))
}

/// (layers whose rows differ, the first such layer, rows that differ in the first such layer, the first row that does).
fn diff(a: &Run, b: &Run, hidden: usize) -> (usize, Option<usize>, usize, Option<usize>) {
    let (mut layers, mut first) = (0, None);
    let mut rows_in_first = 0;
    let mut first_row = None;
    for (i, (x, y)) in a.layers.iter().zip(&b.layers).enumerate() {
        if x != y {
            layers += 1;
            if first.is_none() {
                first = Some(i);
                let bad: Vec<usize> = (0..x.len() / hidden)
                    .filter(|&r| x[r * hidden..(r + 1) * hidden] != y[r * hidden..(r + 1) * hidden])
                    .collect();
                rows_in_first = bad.len();
                first_row = bad.first().copied();
            }
        }
    }
    (layers, first, rows_in_first, first_row)
}

/// `REXL3_ROUTE_TOTALS=300,4200` restricts the cases to those prompt lengths (a quick look, a mutation run).
fn wanted(total: usize) -> bool {
    std::env::var("REXL3_ROUTE_TOTALS").map_or(true, |v| v.split(',').any(|t| t.trim().parse() == Ok(total)))
}

/// The server's grid for `total` tokens: chunks of CHUNK and a tail.
fn server_grid(total: usize) -> Vec<usize> {
    let mut g = vec![CHUNK; total / CHUNK];
    if total % CHUNK != 0 {
        g.push(total % CHUNK);
    }
    g
}

/// Whole prompt and many grids of the same prompt must give the same bits for every row of every layer.
#[test]
#[ignore]
fn prefill_rows_do_not_depend_on_the_chunk_grid() {
    let Some(mut m) = load_model() else { return eprintln!("needs REXL3_QWEN_DIR and a GPU: skipping") };
    let d = m.cfg.hidden;
    let mut failures = Vec::new();
    // (total, the reference grid, the grids that must equal it). Tails of 1..=128 rows, cuts at non-multiples, one row first.
    let cases: Vec<(usize, Vec<usize>, Vec<Vec<usize>>)> = vec![
        (
            1300,
            vec![1300],
            vec![
                vec![1299, 1],
                vec![1, 1299],
                vec![1284, 16],
                vec![1267, 33],
                vec![1236, 64],
                vec![1172, 128],
                vec![1171, 129],
                vec![129, 1171],
                vec![17, 33, 64, 128, 129, 929],
            ],
        ),
        (
            4400,
            server_grid(4400), // [4096, 304]
            vec![vec![2200, 2200], vec![4000, 399, 1], vec![1, 4095, 304]],
        ),
        (
            4200,
            server_grid(4200), // [4096, 104]: a tail of 104 rows
            vec![vec![4000, 200], vec![4096, 64, 40], vec![4096, 103, 1]],
        ),
    ];
    println!("{:>6} {:>22} {:>7} {:>22} {:>8}", "total", "reference grid", "layers", "this grid", "result");
    for (total, reference, others) in cases.into_iter().filter(|c| wanted(c.0)) {
        let toks = random_prompt(total, total as u64);
        let a = run(&mut m, &toks, &reference);
        for g in others {
            let b = run(&mut m, &toks, &g);
            let (layers, first, rows, row0) = diff(&a, &b, d);
            let cand = same_cand(&a.cand, &b.cand);
            let ok = layers == 0 && cand;
            println!(
                "{:>6} {:>22} {:>7} {:>22} {:>8}",
                total,
                format!("{reference:?}"),
                layers,
                format!("{g:?}"),
                if ok {
                    "same".to_string()
                } else {
                    format!(
                        "DIFF first layer {first:?}, {rows} rows from row {row0:?}, candidates {}",
                        if cand { "same" } else { "differ" }
                    )
                }
            );
            if !ok {
                failures.push(format!("total {total}: grid {g:?} differs from {reference:?} (layers {layers}, first {first:?}, candidates same: {cand})"));
            }
        }
    }
    assert!(failures.is_empty(), "a prompt's bits depend on its chunk grid:\n{}", failures.join("\n"));
}

/// One real backend shared by the runs of the harness (it builds a fresh backend per run; reloading 14.65 GiB for each is
/// not an option, and `QwenBackend::prefill_piece` at position 0 resets the slot).
#[derive(Clone)]
struct Shared(Rc<RefCell<QwenBackend>>);

impl Backend for Shared {
    fn vocab_size(&self) -> usize {
        self.0.borrow().vocab_size()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.0.borrow_mut().prefill(slot, start, tokens, candidates)
    }
    fn prefill_piece(&mut self, slot: usize, start: u64, tokens: &[u32], total: usize, candidates: usize) -> RowOutput {
        self.0.borrow_mut().prefill_piece(slot, start, tokens, total, candidates)
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.0.borrow_mut().step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.0.borrow_mut().commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.0.borrow_mut().release(slot)
    }
}

#[test]
#[ignore]
fn chunked_prefill_equals_whole_on_the_real_model() {
    let Some(m) = load_model() else { return eprintln!("needs REXL3_QWEN_DIR and a GPU: skipping") };
    let backend = Shared(Rc::new(RefCell::new(QwenBackend::new(m, CHUNK))));
    let make = || backend.clone();
    let s = Sampling { seed: 5, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 };
    // (total, cut points): pieces and tails of 1, 16, 17, 33, 64, 104, 128 and 129 rows, a cut at 4096 and others that are not
    // multiples of it, a piece that the backend itself has to cut again (4204 -> 4096 + 108)
    let cases: Vec<(usize, Vec<Vec<usize>>)> = vec![
        (100, vec![vec![50], vec![99], vec![1]]), // the GEMV path throughout
        (128, vec![vec![64], vec![127]]),
        (129, vec![vec![128], vec![1], vec![64, 65]]), // 129 is the shortest GEMM prompt: its pieces are all GEMM too
        (
            300,
            vec![
                vec![1],
                vec![16],
                vec![17],
                vec![33],
                vec![64],
                vec![128],
                vec![129],
                vec![299],
                vec![284],
                vec![236],
                vec![16, 64],
                vec![1, 2, 3, 130],
            ],
        ),
        (4200, vec![vec![4096], vec![4000], vec![4199], vec![4183], vec![4136], vec![4167], vec![1000, 4100], vec![17, 4097]]),
        (8300, vec![vec![4096], vec![8192], vec![2000, 8200], vec![1, 4097]]),
    ];
    let mut failures = Vec::new();
    for (total, cut_sets) in cases.into_iter().filter(|c| wanted(c.0)) {
        let prompt = random_prompt(total, 0xC0FFEE + total as u64);
        for cuts in cut_sets {
            let t = std::time::Instant::now();
            let r = chunked_prefill_equals_whole(&make, &prompt, &cuts, &s, 12);
            println!(
                "total {total:>5} cuts {:<26} {} ({:.1} s)",
                format!("{cuts:?}"),
                if r.is_ok() { "equal" } else { "DIFFERENT" },
                t.elapsed().as_secs_f64()
            );
            if let Err(e) = r {
                failures.push(format!("total {total}, cuts {cuts:?}: {e}"));
            }
        }
    }
    // the same on real text, when the ids are given
    if let Some(prompt) = real_prompt(4200) {
        for cuts in [vec![4096], vec![4183]] {
            let r = chunked_prefill_equals_whole(&make, &prompt, &cuts, &s, 12);
            println!("real text total 4200 cuts {cuts:?}: {}", if r.is_ok() { "equal" } else { "DIFFERENT" });
            if let Err(e) = r {
                failures.push(format!("real text, cuts {cuts:?}: {e}"));
            }
        }
    }
    assert!(failures.is_empty(), "chunked prefill differs from whole:\n{}", failures.join("\n"));
}

/// The sweep on the pack's own weights: for each distinct prefill shape of the pack, the first layer that has it and the last.
#[test]
#[ignore]
fn pinned_prefill_is_row_invariant_on_the_real_pack() {
    let Some(d) = dir() else { return eprintln!("needs REXL3_QWEN_DIR and a GPU: skipping") };
    let (_, w) = load_weights(&d);
    gb_cuda::exl3_linear::prefill_init().unwrap();
    let st = gb_cuda::Stream::new().unwrap();
    let linears = w.prefill_linears();
    let mut shapes: Vec<(usize, usize)> = linears.iter().map(|l| (l.k, l.n)).collect();
    shapes.sort_unstable();
    shapes.dedup();
    // the list the cuda crate tests with synthetic weights must be the pack's
    let mut expect = vec![(5120, 17408), (17408, 5120), (5120, 10240), (5120, 6144), (6144, 5120), (5120, 12288), (5120, 1024)];
    expect.sort_unstable();
    assert_eq!(shapes, expect, "the prefill shapes of the pack are not the ones the synthetic sweep covers");
    let full = std::env::var("REXL3_SWEEP_FULL").is_ok_and(|v| v != "0");
    let ms = sweep::grid(full);
    // sizes where cublasGemmEx was stable for every shape (k_proj/v_proj deviate at 193..=248 and 385..=391, see exl3_linear.rs)
    let legacy_ms = [129usize, 130, 255, 256, 257, 300, 1000, 4095, 4096];
    for (k, n) in shapes {
        let of_shape: Vec<_> = linears.iter().filter(|l| (l.k, l.n) == (k, n)).collect();
        for l in [of_shape[0], of_shape[of_shape.len() - 1]] {
            let bad = sweep::row_invariance(l, sweep::Path::Pinned, &ms, &st).unwrap();
            assert_eq!(bad, 0, "pack layer K={k} N={n}: {bad} (m, offset) cases differ from the 4096-row call");
        }
        let differ = sweep::pinned_vs_legacy(of_shape[0], &legacy_ms, &st).unwrap();
        assert!(differ.is_empty(), "pack layer K={k} N={n}: pinned differs from cublasGemmEx at m = {differ:?}");
        println!(
            "K={k:<6} N={n:<6} {} layers: {} row counts row invariant, equal to cublasGemmEx at the stable sizes",
            of_shape.len(),
            ms.len()
        );
    }
}
