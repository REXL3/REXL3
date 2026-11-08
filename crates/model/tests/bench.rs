//! Step-time sweep on the real model: the cost table the speculation budget needs, and the step part of the perf gate.
//!
//!     REXL3_QWEN_DIR=... cargo test --release -p gb-model --test bench -- --ignored --nocapture
//!
//! `REXL3_BENCH_CONTEXT` sets the model's max context (default 262144, what the server runs with: the attention decode grid
//! is sized from it, so a smaller value measures a different kernel launch). `REXL3_BENCH_OUT` writes the table as JSON.
//! `REXL3_BENCH_SHAPES` (for example `6x8,8x8`: sequences x rows per sequence) restricts the table to those shapes, which keeps a
//! profiler run short. `REXL3_BENCH_ITERS` sets the timed rounds per shape (default 12).
//! `REXL3_BENCH_KEEP` is how many rows of each sequence's window every timed round commits: a number N keeps min(window, N) (default
//! 3, about what a live round accepts, so the commit it times is a real one; 0 commits nothing), `half` keeps about half of each
//! window (`evals/b6_bench.sh` times the cost grid with it: a round commits as many rows as it produces tokens, 3 to 5 of a window
//! of 5 to 8).
//! `REXL3_BENCH_FILL=N` prefills N tokens into every slot first (default: the reference prompt, a few hundred tokens), so the
//! table also shows what attention over a real context costs; the step table then runs 1, 2 and 6 sequences only.
//!
//! `REXL3_BENCH_GRID=1` times every (sequences, rows per sequence) combination up to 8 x 8 instead of the 8 standard ones;
//! `REXL3_BENCH_GRID=full` times every row total from 1 to 64 for 1 to 8 sequences (about 260 steps): the budget's cost table.
//! `REXL3_BENCH_GRID=wide` is `full` plus 9, 10, 12, 14, 16, 20, 24, 28 and 32 sequences with row totals up to 128 (`wide_grid`: every
//! multiple of the sequence count to 8 rows each, and the two sides of every row count where the kernels change plan, 32/33, 48/49,
//! 64/65, 96/97 and 128): the cost table for the 32-stream and wide-window work. It needs `REXL3_BENCH_SLOTS=32` and
//! `REXL3_BENCH_WINDOW=128`.
//! `REXL3_BENCH_SLOTS=N` sequences the model is built (and prefilled) for (default 8, 6 with a fill) and `REXL3_BENCH_WINDOW=N` the
//! rows of its window (default 64, the served value; at most 128). `REXL3_BENCH_SHAPES` may name more rows per sequence than 8.
//!
//! `REXL3_BENCH_TREE_BRANCH=B` times TREE windows instead: every sequence's window is a random tree of its row count with branching 1 to B
//! (`gb_cuda::tree::random_parents`), run with `REXL3_SPEC_TREE=1` (so the tree buffers exist), and a commit keeps the root path to the
//! first node `keep - 1` deep. Run it next to the same shapes with `REXL3_BENCH_TREE_BRANCH` unset (plain chain windows) in one GPU hold: the
//! difference is what a tree window costs over a chain of the same rows (the cost table the tree planner prices rows with is the chain's).
//!
//! `draft_time_by_streams` (needs `REXL3_DRAFT_DIR` too) times one draft round for 1..6 sequences and the draft's vocabulary head
//! alone by rows: what the draft costs, and how much of it is the head. `REXL3_BENCH_DRAFT_SLOTS=32` runs it for up to 32 sequences
//! (the draft runs them in groups of `MAX_DRAFT_SEQS` = 16, so 17 and more cost a second draft round) and the head for up to 128 rows.

mod refs;
use gb_cuda::tree::{random_parents, Tree};
use gb_engine::{RowRequest, TreeRequest};
use gb_model::config::QwenConfig;
use gb_model::forward::Model;
use gb_model::pack::Pack;
use gb_model::rows::RowBatch;
use gb_model::weights::Weights;
use refs::{have_refs, ref_npy};
use std::time::Instant;

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().map(|v| v.parse().unwrap_or_else(|_| panic!("{name} is a number")))
}

/// The wide grid: the 260 points of `full` (1 to 8 sequences, every row total from one per sequence to eight) and, for each of the
/// larger sequence counts the 32-stream engine meets, a coarser set of row totals up to 128 that is dense where the kernels change
/// plan. Rows are spread over the sequences as evenly as they allow, like `full`.
fn wide_grid() -> Vec<Vec<usize>> {
    const EDGES: [usize; 21] = [16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 64, 65, 80, 81, 96, 97, 112, 113, 128];
    let spread = |s: usize, rows: usize| -> Vec<usize> { (0..s).map(|i| rows / s + usize::from(i < rows % s)).collect() };
    let mut grid: Vec<Vec<usize>> = (1usize..=8).flat_map(|s| (s..=8 * s).map(move |rows| spread(s, rows))).collect();
    for s in [9usize, 10, 12, 14, 16, 20, 24, 28, 32] {
        let hi = (8 * s).min(128);
        let mut totals: Vec<usize> =
            (1..=8).map(|j| s * j).chain(std::iter::once(s + 1)).chain(EDGES).filter(|&r| r >= s && r <= hi).collect();
        totals.sort_unstable();
        totals.dedup();
        grid.extend(totals.into_iter().map(|rows| spread(s, rows)));
    }
    grid
}

#[test]
fn the_wide_grid_covers_the_32_stream_surface() {
    let g = wide_grid();
    let pts: std::collections::BTreeSet<(usize, usize)> = g.iter().map(|v| (v.len(), v.iter().sum())).collect();
    assert_eq!(pts.len(), g.len(), "every (sequences, rows) point once");
    assert!((1..=8).all(|s| (s..=8 * s).all(|r| pts.contains(&(s, r)))), "the old grid is inside the new one, point for point");
    for s in [9usize, 16, 24, 32] {
        for r in [s, s + 1, 32, 33, 48, 49, 64, 65, 96, 97, 128] {
            if r >= s && r <= (8 * s).min(128) {
                assert!(pts.contains(&(s, r)), "{s} sequences at {r} rows");
            }
        }
    }
    assert_eq!(pts.iter().map(|p| p.1).max(), Some(128));
    assert_eq!(pts.iter().map(|p| p.0).max(), Some(32));
    // no sequence carries more than the 8 rows of a DFlash2 block, and every sequence has a row
    assert!(g.iter().all(|v| v.iter().all(|&x| (1..=8).contains(&x))));
}

#[test]
#[ignore]
fn step_time_by_rows() {
    let Ok(d) = std::env::var("REXL3_QWEN_DIR") else { return eprintln!("REXL3_QWEN_DIR not set: skipping") };
    if !gb_cuda::device_available() || !have_refs() {
        return eprintln!("no GPU or refs: skipping");
    }
    let d = std::path::PathBuf::from(d);
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    let ctx: usize = std::env::var("REXL3_BENCH_CONTEXT").ok().and_then(|v| v.parse().ok()).unwrap_or(262144);
    let fill: Option<usize> = std::env::var("REXL3_BENCH_FILL").ok().and_then(|v| v.parse().ok());
    let slots = env_usize("REXL3_BENCH_SLOTS").unwrap_or(if fill.is_some() { 6 } else { 8 });
    let window = env_usize("REXL3_BENCH_WINDOW").unwrap_or(gb_model::forward::WINDOW_ROWS);
    let long: Vec<u32> = ref_npy("long/ids.npy").to_i64().into_iter().map(|x| x as u32).collect();
    // a fill of N tokens is the reference prompt repeated (the token values do not change what attention or the GDN layers cost)
    let prompt: Vec<u32> = match fill {
        Some(n) => (0..n).map(|i| long[i % long.len()]).collect(),
        None => long,
    };
    let p = prompt.len();
    let kv_blocks = if fill.is_some() { slots * (p.div_ceil(256) + 2) } else { 128.max(slots * (p.div_ceil(256) + 2)) };
    let mut m = Model::with_window_rows(cfg, w, slots, kv_blocks, ctx.max(p + 64), 512, window).unwrap();
    let mut table = Vec::new();
    for s in 0..slots {
        m.begin_slot(s).unwrap();
        for (i, c) in prompt.chunks(512).enumerate() {
            m.prefill_chunk(s, i * 512, c, prompt.len(), None, 0).unwrap();
        }
    }
    m.sync().unwrap();
    let toks: Vec<u32> = (0..128).map(|i| (1000 + i * 7) as u32).collect();
    // How many rows of its window each sequence's commit keeps: min(window rows, N) for REXL3_BENCH_KEEP=N (default 3, the perf gate's
    // baseline), about half of the window for REXL3_BENCH_KEEP=half
    let keep_of = |rows: usize| -> usize {
        match std::env::var("REXL3_BENCH_KEEP").as_deref() {
            Ok("half") => rows.div_ceil(2).max(1),
            Ok(n) => rows.min(n.parse().expect("REXL3_BENCH_KEEP is a number or `half`")),
            Err(_) => rows.min(3),
        }
    };
    eprintln!("context {p} tokens per slot");
    eprintln!("{:>6} {:>5} {:>10} {:>10}", "rows", "seqs", "ms/step", "ms/commit+step");
    // rows per sequence for each timed step
    let uniform = |s: usize, r: usize| vec![r; s];
    let configs: Vec<Vec<usize>> = match std::env::var("REXL3_BENCH_SHAPES") {
        Ok(v) => v
            .split(',')
            .map(|x| {
                let (a, b) = x.split_once('x').expect("REXL3_BENCH_SHAPES entries look like 6x8");
                uniform(a.parse().unwrap(), b.parse().unwrap())
            })
            .collect(),
        Err(_) => match std::env::var("REXL3_BENCH_GRID").as_deref() {
            // every total from one row per sequence to eight, spread as evenly as the sequences allow, for 1 to 8 sequences: a round's
            // cost is not a function of its row count alone, and the kernels change plan at 16, 32 and 48 rows
            Ok("full") => (1usize..=8)
                .flat_map(|s| {
                    (s..=8 * s).map(move |rows| (0..s).map(|i| rows / s + usize::from(i < rows % s)).collect::<Vec<usize>>())
                })
                .collect(),
            Ok("wide") => wide_grid(),
            Ok(_) => (1usize..=8).flat_map(|s| (1..=8).map(move |r| uniform(s, r))).collect(),
            Err(_) if fill.is_some() => vec![uniform(1, 1), uniform(1, 8), uniform(2, 8), uniform(6, 8)],
            Err(_) => {
                [(1, 1), (1, 2), (1, 4), (1, 8), (2, 8), (4, 8), (6, 8), (8, 8)].iter().map(|&(s, r)| uniform(s, r)).collect()
            }
        },
    };
    let n: usize = std::env::var("REXL3_BENCH_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
    for per_seq in configs {
        let seqs = per_seq.len();
        let rows: usize = per_seq.iter().sum();
        let reqs: Vec<RowRequest> = per_seq
            .iter()
            .enumerate()
            .map(|(s, &per)| RowRequest { slot: s, tokens: toks[..per].to_vec(), first_position: p as u64 })
            .collect();
        // tree windows (REXL3_BENCH_TREE_BRANCH): a random tree per sequence and the root path a commit of `keep` rows follows
        let branch: Option<usize> = std::env::var("REXL3_BENCH_TREE_BRANCH").ok().and_then(|v| v.parse().ok());
        let trees: Vec<Tree> = match branch {
            Some(br) => per_seq
                .iter()
                .enumerate()
                .map(|(s, &per)| Tree::new(&random_parents(1000 + (rows * 7 + s) as u64, per, br)).unwrap())
                .collect(),
            None => Vec::new(),
        };
        let b = match branch {
            Some(_) => {
                let treqs: Vec<TreeRequest> = per_seq
                    .iter()
                    .enumerate()
                    .map(|(s, &per)| TreeRequest {
                        slot: s,
                        tokens: toks[..per].to_vec(),
                        parents: trees[s].parents().to_vec(),
                        first_position: p as u64,
                    })
                    .collect();
                RowBatch::from_trees(&treqs)
            }
            None => RowBatch::from_requests(&reqs),
        };
        // warm up, then time with the commit included (it is part of every real round)
        // the commit's kept rows (see `keep_of`): a commit that keeps nothing skips the recurrent-state replay, which is most of what
        // a commit costs
        let keep: Vec<usize> = per_seq.iter().map(|&per| keep_of(per)).collect();
        // a tree commit follows a root path of `keep` rows (or the deepest there is)
        let paths: Vec<Option<Vec<usize>>> = trees
            .iter()
            .zip(&keep)
            .map(|(t, &k)| {
                let node = (0..t.len()).find(|&i| t.depth(i) + 1 >= k).unwrap_or(t.len() - 1);
                Some(t.path(node))
            })
            .collect();
        let commit_all = |m: &mut Model| {
            for (s, &k) in keep.iter().enumerate() {
                match paths.get(s) {
                    Some(Some(path)) if k > 0 => m.commit_path(s, path).unwrap(),
                    _ => m.commit(s, k),
                }
            }
        };
        for _ in 0..3 {
            m.window_candidates(&b, 28).unwrap();
            commit_all(&mut m);
        }
        let t = Instant::now();
        for _ in 0..n {
            m.window_candidates(&b, 28).unwrap();
            commit_all(&mut m);
        }
        m.flush().unwrap();
        m.sync().unwrap();
        let per_round = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let t = Instant::now();
        for _ in 0..n {
            m.window_candidates(&b, 28).unwrap();
        }
        let per_step = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
        eprintln!("{:>6} {:>5} {:>10.1} {:>10.1}", rows, seqs, per_step, per_round);
        table.push(format!("{{\"rows\": {}, \"seqs\": {seqs}, \"ms_step\": {per_step:.2}, \"ms_round\": {per_round:.2}}}", rows));
    }
    if let Ok(path) = std::env::var("REXL3_BENCH_OUT") {
        let doc = format!("{{\"max_context\": {ctx}, \"fill\": {p}, \"steps\": [\n  {}\n]}}\n", table.join(",\n  "));
        std::fs::write(path, doc).unwrap();
    }
}

fn mean_ms(n: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    t.elapsed().as_secs_f64() * 1000.0 / n as f64
}

#[test]
#[ignore]
fn draft_time_by_streams() {
    use gb_cuda::exl3_linear::{linear_workspace_bytes, Dtype};
    use gb_cuda::DevBuf;
    use gb_model::draft::Draft;
    use gb_sampling::Sampling;
    let (Ok(d), Ok(dd)) = (std::env::var("REXL3_QWEN_DIR"), std::env::var("REXL3_DRAFT_DIR")) else {
        return eprintln!("REXL3_QWEN_DIR / REXL3_DRAFT_DIR not set: skipping");
    };
    if !gb_cuda::device_available() || !have_refs() {
        return eprintln!("no GPU or refs: skipping");
    }
    let (d, dd) = (std::path::PathBuf::from(d), std::path::PathBuf::from(dd));
    let cfg = QwenConfig::from_dir(&d).unwrap();
    let vocab = cfg.vocab;
    let w = Weights::load(&Pack::open(&d).unwrap(), &cfg, |_, _| {}).unwrap();
    let slots_n = env_usize("REXL3_BENCH_DRAFT_SLOTS").unwrap_or(6);
    assert!((1..=64).contains(&slots_n), "REXL3_BENCH_DRAFT_SLOTS is 1..=64");
    let wide = slots_n > 6;
    let mut m = Model::new(cfg, w, slots_n, 128.max(slots_n * 3), 4096, 512).unwrap();
    m.draft = Some(Draft::load(&Pack::open(&dd).unwrap(), vocab, slots_n, 512).unwrap().0);
    let long: Vec<u32> = ref_npy("long/ids.npy").to_i64().into_iter().map(|x| x as u32).collect();
    let seqs: Vec<Vec<u32>> = (0..slots_n)
        .map(|s| {
            // the reference prompt is 96 tokens: sequences past the 16th reuse an earlier one's start (their slots differ)
            let prompt = &long[(s % 16) * 5..];
            m.begin_slot(s).unwrap();
            m.prefill_chunk(s, 0, prompt, prompt.len(), Some(28), 0).unwrap();
            let mut t = prompt.to_vec();
            t.push(1000 + 17 * s as u32);
            t
        })
        .collect();
    m.sync().unwrap();
    let samplings = vec![Sampling { seed: 5, temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.0 }; slots_n];
    eprintln!("draft round, one block of 8 rows per sequence (wall ms, including the sync):");
    eprintln!("{:>5} {:>10} {:>10}", "seqs", "ms/round", "ms/seq");
    // past 16 sequences the draft runs a second group (`MAX_DRAFT_SEQS`), each with its own sync: the rows 17 and up are the point
    let counts: Vec<usize> = if wide {
        [1usize, 2, 3, 4, 5, 6, 8, 12, 16, 17, 20, 24, 28, 32].into_iter().filter(|&n| n <= slots_n).collect()
    } else {
        vec![1, 2, 3, 4, 5, 6]
    };
    for &n in &counts {
        let asks: Vec<(usize, &[u32], &Sampling)> = (0..n).map(|s| (s, &seqs[s][..], &samplings[s])).collect();
        let ms = mean_ms(12, || {
            m.propose_drafts_batch(&asks).unwrap();
        });
        eprintln!("{:>5} {:>10.2} {:>10.2}", n, ms, ms / n as f64);
    }
    // the draft's head (the target's 6-bit lm_head) on `nd` rows: 7 per sequence in a round
    let head = &m.w.lm_head;
    let stream = &m.stream;
    let mut ws = DevBuf::zeroed(linear_workspace_bytes(128, 5120, vocab, 6)).unwrap();
    let mut y = DevBuf::zeroed(128 * vocab * 2).unwrap();
    let x = DevBuf::zeroed(128 * 5120 * 2).unwrap();
    eprintln!("draft head alone ({vocab} columns, 6-bit), wall ms:");
    eprintln!("{:>5} {:>10}", "rows", "ms");
    let head_rows: Vec<usize> = if wide {
        vec![1, 7, 8, 14, 16, 17, 21, 24, 28, 32, 33, 35, 42, 48, 49, 56, 64, 65, 80, 96, 97, 112, 128]
    } else {
        vec![1, 7, 8, 14, 16, 17, 21, 24, 28, 32, 33, 35, 42, 48]
    };
    for nd in head_rows {
        let ms = mean_ms(20, || {
            head.forward(&x, &mut y, nd, Dtype::BF16, Dtype::BF16, &mut ws, stream).unwrap();
            stream.sync().unwrap();
        });
        eprintln!("{:>5} {:>10.2}", nd, ms);
    }
    // a draft round asked for fewer picks per sequence, over the full vocabulary and over TensorFold's candidate set
    let strips = gb_model::draft::vocab_strips(&gb_model::draft::TF_DRAFT_VOCAB, vocab).unwrap();
    let kept: usize = strips.iter().map(|&(lo, hi)| (hi - lo) * 128).sum();
    for set in ["full", "tf"] {
        if set == "tf" {
            let gb_model::forward::Model { draft, w, .. } = &mut m;
            let added = draft.as_mut().unwrap().restrict_vocab(&w.lm_head, &gb_model::draft::TF_DRAFT_VOCAB).unwrap();
            eprintln!("restricted draft vocabulary: {kept} of {vocab} columns, {:.0} MiB added", added as f64 / (1 << 20) as f64);
            let sub = m.w.lm_head.select_strips(&strips).unwrap();
            let mut ys = DevBuf::zeroed(128 * kept * 2).unwrap();
            let mut wss = DevBuf::zeroed(linear_workspace_bytes(128, 5120, kept, 6)).unwrap();
            let restricted_rows: &[usize] =
                if wide { &[1, 7, 16, 26, 32, 33, 42, 48, 64, 65, 96, 128] } else { &[1, 7, 16, 26, 32, 33, 42, 48] };
            for &nd in restricted_rows {
                let ms = mean_ms(20, || {
                    sub.forward(&x, &mut ys, nd, Dtype::BF16, Dtype::BF16, &mut wss, &m.stream).unwrap();
                    m.stream.sync().unwrap();
                });
                eprintln!("restricted head alone, {nd:>3} rows: {ms:.2} ms");
            }
        }
        eprintln!("draft round, vocabulary {set}: ms by sequences (rows) and picks asked per sequence");
        eprintln!("{:>5} {:>8} {:>8} {:>8} {:>8} {:>8}", "seqs", "7", "5", "4", "2", "1");
        for &n in &counts {
            let mut line = format!("{n:>5}");
            for want in [7usize, 5, 4, 2, 1] {
                let asks: Vec<(usize, &[u32], &Sampling, usize)> =
                    (0..n).map(|s| (s, &seqs[s][..], &samplings[s], want)).collect();
                let ms = mean_ms(12, || {
                    m.propose_drafts_batch_want(&asks).unwrap();
                });
                line.push_str(&format!(" {ms:>8.2}"));
            }
            eprintln!("{line}");
        }
    }
}
