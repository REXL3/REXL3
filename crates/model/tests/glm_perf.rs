//! Decode-step performance of the GLM-5.3-Flash model on the GB10, and the proof that it costs no bit. All `#[ignore]`, env-gated
//! (REXL3_GLM_DIR: the pack, REXL3_GLM_ORACLE_DIR: the oracle's dumps for real token ids), one model (with the NextN layer) per process, under
//! the GPU lock: `nrun NAME --gpu -- 'cargo test --release -p gb-model --test glm_perf -- --ignored --nocapture --test-threads=1 perf_digest perf_steps'`.
//!
//! * `perf_digest`: a scripted run (prefill, serial steps, verify windows of 2..9 rows with partial keeps, the DSA regime boundary, several
//!   sequences in one window, MTP rounds) hashed segment by segment (every row's fp16 logits, the candidates, the drafts, the KDA state and the
//!   caches): printed as `DIGEST <graphs> <segment> <hex>`. The script runs once with graphs off and once with graphs on in the same process and the
//!   two must be identical; `REXL3_DIGEST_EXPECT=<file>` also compares every line with a file written by an earlier build (the commit before a change).
//! * `perf_steps`: step time of the decode (R = 1, context 2,300), the window table R = 1..9, and the MTP draft call, graphs off and on, as medians
//!   of several runs with the GPU clock recorded.

mod glm_support;
mod perf_graphs;

use gb_engine::RowRequest;
use gb_model::families::glm::model::{DraftAsk, GlmModel};
use gb_model::families::glm::npy;
use gb_model::rows::RowBatch;
use glm_support::*;
use std::path::{Path, PathBuf};

const SLOTS: usize = 8;
const KV_BLOCKS: usize = 80; // 20,480 tokens
const MAX_CONTEXT: usize = 8192;

fn model() -> Option<std::sync::MutexGuard<'static, GlmModel>> {
    shared_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT)
}

fn oracle_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_GLM_ORACLE_DIR") {
        Ok(d) if Path::new(&d).is_dir() => Some(PathBuf::from(d)),
        Ok(d) => panic!("REXL3_GLM_ORACLE_DIR={d} is not a directory"),
        Err(_) => {
            eprintln!("REXL3_GLM_ORACLE_DIR is not set: not run");
            None
        }
    }
}

fn ids_of(oracle: &Path, prompt: &str) -> Vec<u32> {
    npy::read(&oracle.join(prompt).join("ids.npy")).unwrap().to_i32().unwrap().iter().map(|&x| x as u32).collect()
}

fn graphs(m: &mut GlmModel, on: bool) {
    perf_graphs::set(m, on);
}

fn clocks() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=clocks.sm,clocks.mem,utilization.gpu,power.draw,temperature.gpu", "--format=csv,noheader"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|e| format!("nvidia-smi: {e}"))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

// ---------------------------------------------------------------------------------------------------------------------------- the digest

/// FNV-1a over the bytes fed to it.
#[derive(Clone, Copy)]
struct Dg(u64);

impl Dg {
    fn new() -> Self {
        Dg(0xcbf29ce484222325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
    fn u16s(&mut self, v: &[u16]) {
        for x in v {
            self.bytes(&x.to_le_bytes());
        }
    }
    fn i32s(&mut self, v: &[i32]) {
        for x in v {
            self.bytes(&x.to_le_bytes());
        }
    }
    fn f32s(&mut self, v: &[f32]) {
        for x in v {
            self.bytes(&x.to_bits().to_le_bytes());
        }
    }
}

/// What the script holds per slot: the tokens so far (the last one is pending: not yet fed).
struct Seq {
    slot: usize,
    tokens: Vec<u32>,
}

/// Prefill `prompt` into `slot` as the backend does (128-row chunks); the sequence ends with the greedy token after it.
fn start(m: &mut GlmModel, slot: usize, prompt: &[u32]) -> Seq {
    m.begin_slot(slot).unwrap();
    let mut at = 0;
    let mut first = 0;
    while at < prompt.len() {
        let end = (at + 128).min(prompt.len());
        let want = (end == prompt.len()).then_some(8);
        if let Some(c) = m.prefill_chunk(slot, at, &prompt[at..end], want).unwrap() {
            first = c.ids[0] as u32;
        }
        at = end;
    }
    let mut tokens = prompt.to_vec();
    tokens.push(first);
    Seq { slot, tokens }
}

/// One window over several sequences: `rows[i]` tokens for sequence i, the first the pending token and the rest `extra` (drafts); hashes every row's
/// logits and candidates; commits `keeps[i]` rows of each and extends the sequences by the committed rows' greedy successors.
fn window(m: &mut GlmModel, dg: &mut Dg, seqs: &mut [&mut Seq], drafts: &[Vec<u32>], keeps: &[usize]) {
    let reqs: Vec<RowRequest> = seqs
        .iter()
        .zip(drafts)
        .map(|(s, d)| {
            let mut toks = vec![*s.tokens.last().unwrap()];
            toks.extend_from_slice(d);
            RowRequest { slot: s.slot, tokens: toks, first_position: (s.tokens.len() - 1) as u64 }
        })
        .collect();
    let b = RowBatch::from_requests(&reqs);
    let c = m.window_candidates(&b, 8).unwrap();
    let logits = m.logits_host(0, b.rows).unwrap();
    dg.u16s(&logits);
    dg.i32s(&c.ids);
    dg.f32s(&c.vals);
    for (i, s) in seqs.iter_mut().enumerate() {
        let rows = b.seq_rows(i);
        let keep = keeps[i].min(rows.len());
        m.commit(s.slot, keep);
        // the committed rows' own tokens (the drafts) join the sequence, then the greedy token of the last kept row is the new pending one
        for j in 1..keep {
            s.tokens.push(reqs[i].tokens[j]);
        }
        s.tokens.push(c.ids[(rows.start + keep - 1) * c.count] as u32);
    }
}

fn segment(name: &str, dg: &Dg, tag: &str, out: &mut Vec<String>) {
    let line = format!("DIGEST {tag} {name} {:016x}", dg.0);
    println!("{line}");
    out.push(line);
}

/// The scripted run. `tag` labels the output lines.
fn script(m: &mut GlmModel, oracle: &Path, tag: &str) -> Vec<String> {
    let long = ids_of(oracle, "long_2300");
    let code = ids_of(oracle, "code_700");
    let chat = ids_of(oracle, "chat_300");
    let mut out = Vec::new();

    // 1. serial decode at context 2,278 (the sparse regime), 16 steps
    let mut s0 = start(m, 0, &long);
    let mut dg = Dg::new();
    for _ in 0..16 {
        window(m, &mut dg, &mut [&mut s0], &[vec![]], &[1]);
    }
    segment("serial_2300", &dg, tag, &mut out);

    // 2. verify windows of 2..=9 rows with drafts that are partly right (the greedy tokens of a serial run would be right; these are filler) and
    //    partial keeps; three passes so that every shape is replayed; a serial step after each
    let mut dg = Dg::new();
    for pass in 0..3usize {
        for r in 2..=9usize {
            let d: Vec<u32> = (0..r - 1).map(|j| code[(pass * 131 + r * 17 + j * 7) % code.len()]).collect();
            let keep = [r, 1, r / 2 + 1, r - 1, r][(pass + r) % 5];
            window(m, &mut dg, &mut [&mut s0], &[d], &[keep]);
            window(m, &mut dg, &mut [&mut s0], &[vec![]], &[1]);
        }
    }
    segment("windows_2300", &dg, tag, &mut out);

    // 3. the DSA regime boundary: a sequence of 2,040 tokens, windows of 9 across 2,047/2,048, then serial
    let mut s1 = start(m, 1, &long[..2040]);
    let mut dg = Dg::new();
    for (r, keep) in [(9usize, 9usize), (9, 5), (5, 2), (3, 3), (4, 1), (1, 1), (1, 1), (9, 9), (6, 6), (2, 2)] {
        let d: Vec<u32> = (0..r - 1).map(|j| long[(300 + r * 11 + j * 5) % long.len()]).collect();
        window(m, &mut dg, &mut [&mut s1], &[d], &[keep]);
    }
    for _ in 0..4 {
        window(m, &mut dg, &mut [&mut s1], &[vec![]], &[1]);
    }
    segment("boundary_2040", &dg, tag, &mut out);

    // 4. several sequences in one window: two sequences of 3 rows (kept 3 and 1), three of one row, the same shapes three times
    let mut s2 = start(m, 2, &chat);
    let mut dg = Dg::new();
    for pass in 0..3usize {
        let d0: Vec<u32> = vec![code[(pass * 3) % 100], code[(pass * 3 + 1) % 100]];
        let d1: Vec<u32> = vec![code[(pass * 5 + 7) % 100], code[(pass * 5 + 8) % 100]];
        window(m, &mut dg, &mut [&mut s0, &mut s1], &[d0, d1], &[3, 1]);
        window(m, &mut dg, &mut [&mut s0, &mut s1, &mut s2], &[vec![], vec![], vec![]], &[1, 1, 1]);
        window(m, &mut dg, &mut [&mut s2, &mut s0], &[vec![code[11]], vec![]], &[2, 1]);
    }
    segment("multi", &dg, tag, &mut out);

    // 5. MTP rounds: the drafter proposes (depth 3), a window verifies, the accepted prefix is committed
    let mut s3 = start(m, 3, &chat[..200]);
    let mut dg = Dg::new();
    for round in 0..10usize {
        let depth = 1 + round % 3;
        let ds = m.mtp_draft(&[DraftAsk { slot: 3, tokens: &s3.tokens, depth }]).unwrap().remove(0);
        let d = ds.map(|d| d.ids).unwrap_or_default();
        dg.i32s(&d.iter().map(|&x| x as i32).collect::<Vec<_>>());
        // accept the prefix the next window's greedy tokens agree with: run the window, count agreement, commit
        let mut toks = vec![*s3.tokens.last().unwrap()];
        toks.extend_from_slice(&d);
        let b = RowBatch::from_requests(&[RowRequest {
            slot: 3,
            tokens: toks.clone(),
            first_position: (s3.tokens.len() - 1) as u64,
        }]);
        let c = m.window_candidates(&b, 8).unwrap();
        dg.u16s(&m.logits_host(0, b.rows).unwrap());
        dg.i32s(&c.ids);
        let acc = d.iter().enumerate().take_while(|(j, &x)| c.ids[j * c.count] as u32 == x).count();
        m.commit(3, acc + 1);
        s3.tokens.extend_from_slice(&d[..acc]);
        s3.tokens.push(c.ids[acc * c.count] as u32);
    }
    segment("mtp", &dg, tag, &mut out);

    // 6. the state: KDA layers (both buffers, parity, conv), the latent rows and pooled keys of an MLA layer, for the slots that ran
    let mut dg = Dg::new();
    for slot in [0usize, 1, 3] {
        for l in [0usize, 5, 30, 44] {
            // the committed buffer and the conv history (the other buffer of the pair holds whatever the last window left in it, which depends on
            // what ran in the slot before this script)
            let (st, conv) = m.kda_state(l, slot).unwrap();
            dg.f32s(&st);
            dg.u16s(&conv);
        }
        dg.u16s(&m.latent_rows(3, slot, 64).unwrap());
        dg.u16s(&m.latent_rows(43, slot, 64).unwrap());
        dg.u16s(&m.pool_rows(7, slot, 16).unwrap());
    }
    segment("state", &dg, tag, &mut out);
    out
}

#[test]
#[ignore]
fn perf_digest() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    graphs(&mut m, false);
    let off = script(&mut m, &oracle, "graphs_off");
    graphs(&mut m, true);
    let before = perf_graphs::stats(&m);
    let avail0 = mem_available();
    let on = script(&mut m, &oracle, "graphs_on");
    if let (Some(b), Some(st)) = (before, perf_graphs::stats(&m)) {
        println!(
            "GRAPHS captures {} replays {} eager {} failures {} (before the script: {} captures) last failure {:?}; MemAvailable {:.2} -> {:.2} GiB",
            st.captures,
            st.replays,
            st.eager,
            st.failures,
            b.captures,
            st.last_failure,
            gib(avail0),
            gib(mem_available())
        );
        assert_eq!(st.failures, 0, "a capture failed: {:?}", st.last_failure);
        assert!(st.replays > b.replays || !perf_graphs::is_on(&m), "graphs are on but nothing was replayed");
    }
    for (a, b) in off.iter().zip(&on) {
        let (a, b) = (a.splitn(3, ' ').nth(2).unwrap(), b.splitn(3, ' ').nth(2).unwrap());
        assert_eq!(a, b, "graphs on and off differ");
    }
    if let Ok(path) = std::env::var("REXL3_DIGEST_EXPECT") {
        let want = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let got: Vec<String> = off.iter().map(|l| l.splitn(3, ' ').nth(2).unwrap().to_string()).collect();
        let want: Vec<String> = want
            .lines()
            .filter(|l| l.starts_with("DIGEST graphs_off "))
            .map(|l| l.splitn(3, ' ').nth(2).unwrap().to_string())
            .collect();
        assert_eq!(got, want, "the digests differ from {path}");
        println!("DIGEST matches {path}");
    }
    if let Ok(path) = std::env::var("REXL3_DIGEST_WRITE") {
        std::fs::write(&path, off.join("\n") + "\n").unwrap();
    }
}

/// The graph path's invariants, each broken on purpose (`GraphMutant`): a replay that leaves the window's rows stale, one that leaves the page table
/// stale, a window that takes the graph of another row count, must change the digest (the test fails if one does not); a capture with a host
/// synchronisation inside must be refused and fall back to eager launches with the bits unchanged, without a fatal error.
#[test]
#[ignore]
fn perf_graph_mutants() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    graphs(&mut m, false);
    let want: Vec<String> =
        script(&mut m, &oracle, "graphs_off").iter().map(|l| l.splitn(3, ' ').nth(2).unwrap().to_string()).collect();
    graphs(&mut m, true);
    let mut missed = Vec::new();
    for name in [
        "stale_rows",
        "stale_table",
        "wrong_shape",
        "sync_in_capture",
        "slow_side",
        "join_late",
        "skip_latent",
        "skip_pool",
        "skip_q",
        "skip_select",
    ] {
        // REXL3_GRAPH_MUTANTS="skip_latent,skip_pool" runs only those (a re-check of one change)
        if std::env::var("REXL3_GRAPH_MUTANTS").is_ok_and(|v| !v.split(',').any(|x| x.trim() == name)) {
            continue;
        }
        perf_graphs::clear(&mut m);
        perf_graphs::set_mutant(&mut m, Some(name));
        let before = perf_graphs::stats(&m);
        let got: Vec<String> =
            script(&mut m, &oracle, name).iter().map(|l| l.splitn(3, ' ').nth(2).unwrap().to_string()).collect();
        perf_graphs::set_mutant(&mut m, None);
        let st = perf_graphs::stats(&m);
        let differs = got != want;
        if let (Some(b), Some(st)) = (before, st) {
            println!(
                "MUTANT {name}: digest {} (captures {}, replays {}, failures {})",
                if differs { "DIFFERS (detected)" } else { "equal" },
                st.captures - b.captures,
                st.replays - b.replays,
                st.failures - b.failures
            );
            if name == "slow_side" {
                // the control: a slow side stream with the join in place must not move a bit
                assert!(!differs, "a slow shared expert changed the bits: the join does not hold");
            } else if name == "sync_in_capture" {
                assert!(!differs, "a refused capture changed the bits");
                assert!(
                    st.failures > b.failures && st.captures == b.captures,
                    "every capture with a host sync inside must be refused"
                );
            } else if !differs {
                missed.push(name);
            }
        }
    }
    assert!(missed.is_empty(), "graph mutants not detected by the digest: {missed:?}");
}

/// The window table the arbiter prices verification with (`spec::VERIFY_MS`): one sequence at context ~2,300, `R` rows of real text a window, every window
/// committed in full (the commit is applied inside the next forward, so it is in the time), the production configuration; printed as `VERIFY rows ms`.
#[test]
#[ignore]
fn perf_verify_table() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    let long = ids_of(&oracle, "long_2300");
    let code = ids_of(&oracle, "code_700");
    graphs(&mut m, true);
    let mut s = start(&mut m, 0, &long);
    for r in [1usize, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 24, 32, 48, 64] {
        let mut t = Vec::new();
        for i in 0..14usize {
            let mut toks = vec![*s.tokens.last().unwrap()];
            toks.extend_from_slice(&code[(i * 37 + r) % 600..(i * 37 + r) % 600 + r - 1]);
            let b = RowBatch::from_requests(&[RowRequest {
                slot: 0,
                tokens: toks.clone(),
                first_position: (s.tokens.len() - 1) as u64,
            }]);
            let t0 = std::time::Instant::now();
            let c = m.window_candidates(&b, 8).unwrap();
            m.commit(0, r);
            if i >= 3 {
                t.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            s.tokens.extend_from_slice(&toks[1..]);
            s.tokens.push(c.ids[(r - 1) * c.count] as u32);
        }
        println!("VERIFY {r} {:.2}", median(&mut t));
    }
}

// ---------------------------------------------------------------------------------------------------------------------------- tuning

/// The router GEMV's blocking (rows a warp, warps a block, loads in flight a lane) in the real model, weights cold: AR step time per shape (graphs
/// on; the graphs are rebuilt for each shape). Every shape gives the same bits (the digest test checks the one that becomes the default).
#[test]
#[ignore]
fn perf_router_shapes() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    let long = ids_of(&oracle, "long_2300");
    graphs(&mut m, true);
    let mut s = start(&mut m, 0, &long);
    let shapes: Vec<Option<(usize, usize, usize)>> = vec![
        None,
        Some((1, 8, 1)),
        Some((1, 8, 4)),
        Some((1, 8, 8)),
        Some((1, 4, 8)),
        Some((1, 2, 8)),
        Some((1, 1, 8)),
        Some((1, 8, 16)),
        Some((1, 4, 16)),
        Some((1, 2, 16)),
        Some((1, 1, 16)),
        None,
    ];
    for shape in shapes {
        gb_cuda::moe::set_router_shape(shape);
        m.clear_graphs().unwrap();
        ar_steps(&mut m, &mut s, 4);
        let mut meds = Vec::new();
        for _ in 0..3 {
            let mut t = ar_steps(&mut m, &mut s, 20);
            meds.push(median(&mut t));
        }
        println!("ROUTER shape {shape:?}: AR median {:.2} ms", median(&mut meds));
    }
    gb_cuda::moe::set_router_shape(None);
}

/// The window table (R = 1..9 rows, one sequence at context ~2,300, real text after the pending token, nothing committed so every window is the very
/// same work) for two configurations of the model, A and B, alternating in blocks (the graphs are rebuilt at each switch): per R the median of
/// three blocks of 8 windows each. `set(m, false)` puts the model in configuration A, `set(m, true)` in B.
fn ab_windows(m: &mut GlmModel, oracle: &Path, a: &str, b: &str, set: &dyn Fn(&mut GlmModel, bool, usize)) {
    let long = ids_of(oracle, "long_2300");
    let code = ids_of(oracle, "code_700");
    graphs(m, true);
    let s = start(m, 0, &long);
    println!("AB {a} vs {b}, context {}: window ms by rows (the same window replayed, nothing committed)", s.tokens.len());
    let (mut sum_a, mut sum_b) = (0.0, 0.0);
    for r in 1..=9usize {
        let mut toks = vec![*s.tokens.last().unwrap()];
        toks.extend_from_slice(&code[40 + r * 9..40 + r * 9 + r - 1]);
        let batch = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks, first_position: (s.tokens.len() - 1) as u64 }]);
        let mut t: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        for round in 0..3 {
            for k in 0..2 {
                // alternate the order each round so a drift does not favour one side
                let which = (k + round) % 2;
                set(m, which == 1, r);
                m.clear_graphs().unwrap();
                let mut blk = Vec::new();
                for i in 0..11usize {
                    let t0 = std::time::Instant::now();
                    m.window_candidates(&batch, 8).unwrap();
                    m.commit(0, 0);
                    if i >= 3 {
                        blk.push(t0.elapsed().as_secs_f64() * 1e3);
                    }
                }
                t[which].push(median(&mut blk));
            }
        }
        let (ma, mb) = (median(&mut t[0]), median(&mut t[1]));
        sum_a += ma;
        sum_b += mb;
        println!("AB rows={r}: {a} {ma:.2} ms, {b} {mb:.2} ms (gain {:.2} ms)", ma - mb);
    }
    println!("AB sum over R=1..9: {a} {sum_a:.1} ms, {b} {sum_b:.1} ms");
    set(m, true, 0);
}

/// The shared expert on a side stream (concurrent with the router and the routed experts) against the same graphs without it, the window table.
#[test]
#[ignore]
fn perf_overlap_windows() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    ab_windows(&mut m, &oracle, "shared expert on the main stream", "shared expert on a side stream", &|m, b, _| {
        m.set_overlap(b).unwrap();
    });
}

/// The MLA layers' branches (the latent chain and the indexer's two sides beside the query chain): A: the shared expert's side stream only, B: and the
/// MLA branches, the window table.
#[test]
#[ignore]
fn perf_branch_windows() {
    use gb_model::families::glm::model::graphs::{PART_ALL, PART_SHARED};
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    ab_windows(&mut m, &oracle, "shared only", "shared + MLA branches", &|m, b, _| {
        m.set_overlap_parts(if b { PART_ALL } else { PART_SHARED }).unwrap();
    });
    m.set_overlap(true).unwrap();
}

/// The router GEMV's loads in flight: the old default (one load in flight a lane) against the new (four), the window table.
#[test]
#[ignore]
fn perf_router_windows() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    ab_windows(&mut m, &oracle, "un=1 (before)", "un=4 (default)", &|_, b, r| {
        // the old default: rows a warp 1 below 8 rows and 2 below 32, 8 warps, one load in flight; the new one is the launcher's own
        gb_cuda::moe::set_router_shape(if b { None } else { Some((if r >= 8 { 2 } else { 1 }, 8, 1)) });
    });
    gb_cuda::moe::set_router_shape(None);
}

// ---------------------------------------------------------------------------------------------------------------------------- trace

/// For `nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --cuda-graph-trace=node` (`tools/glm_perf_nsys.sh`): at context
/// ~2,300 (the sparse DSA regime), 30 single-row steps, then 20 windows of 3 rows, each range inside the profiler (the prefill and the warm-up are
/// outside it). `REXL3_TRACE_GRAPHS=0` traces the eager launches. Analyse with `tools/nsys_decode_gaps.py --steps 30,20`.
#[test]
#[ignore]
fn perf_trace() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    let long = ids_of(&oracle, "long_2300");
    let code = ids_of(&oracle, "code_700");
    graphs(&mut m, std::env::var("REXL3_TRACE_GRAPHS").as_deref() != Ok("0"));
    let mut s = start(&mut m, 0, &long);
    ar_steps(&mut m, &mut s, 6);
    std::thread::sleep(std::time::Duration::from_secs(5));
    gb_cuda::rt::profiler_start();
    let ar = ar_steps(&mut m, &mut s, 30);
    gb_cuda::rt::profiler_stop();
    println!("traced 30 steps of 1 row: {:.2} ms each (under the profiler)", ar.iter().sum::<f64>() / 30.0);
    std::thread::sleep(std::time::Duration::from_secs(5));
    // windows of 3 rows (a pending token and two drafts of real text), nothing committed so that every window is the same work
    let mut toks = vec![*s.tokens.last().unwrap()];
    toks.extend_from_slice(&code[100..102]);
    let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks, first_position: (s.tokens.len() - 1) as u64 }]);
    for _ in 0..6 {
        m.window_candidates(&b, 8).unwrap();
        m.commit(0, 0);
    }
    std::thread::sleep(std::time::Duration::from_secs(5));
    gb_cuda::rt::profiler_start();
    let t = std::time::Instant::now();
    for _ in 0..20 {
        m.window_candidates(&b, 8).unwrap();
        m.commit(0, 0);
    }
    let ms = t.elapsed().as_secs_f64() * 1e3 / 20.0;
    gb_cuda::rt::profiler_stop();
    println!("traced 20 windows of 3 rows: {ms:.2} ms each (under the profiler)");
}

// ---------------------------------------------------------------------------------------------------------------------------- speed

/// Milliseconds of `n` consecutive serial decode steps of `s` (each: window + candidates + commit, the previous commit applied inside).
fn ar_steps(m: &mut GlmModel, s: &mut Seq, n: usize) -> Vec<f64> {
    let mut times = Vec::new();
    for _ in 0..n {
        let t = std::time::Instant::now();
        let b = RowBatch::from_requests(&[RowRequest {
            slot: s.slot,
            tokens: vec![*s.tokens.last().unwrap()],
            first_position: (s.tokens.len() - 1) as u64,
        }]);
        let c = m.window_candidates(&b, 8).unwrap();
        m.commit(s.slot, 1);
        times.push(t.elapsed().as_secs_f64() * 1e3);
        s.tokens.push(c.ids[0] as u32);
    }
    times
}

#[test]
#[ignore]
fn perf_steps() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = model() else { return };
    let long = ids_of(&oracle, "long_2300");
    let code = ids_of(&oracle, "code_700");
    let runs = 7usize;
    println!("CLOCKS start {}", clocks());
    let mut s = start(&mut m, 0, &long);
    // warm both modes (the first graphs-on window of a shape runs eagerly, the second captures)
    for on in [false, true] {
        graphs(&mut m, on);
        ar_steps(&mut m, &mut s, 4);
    }
    let mut table: Vec<(bool, Vec<f64>)> = vec![(false, Vec::new()), (true, Vec::new())];
    for run in 0..runs {
        for (on, meds) in table.iter_mut() {
            graphs(&mut m, *on);
            let mut t = ar_steps(&mut m, &mut s, 30);
            let med = median(&mut t);
            println!(
                "AR graphs={} run={run} median {med:.2} ms (min {:.2}, max {:.2}) {:.2} tok/s",
                *on,
                t[0],
                t[t.len() - 1],
                1e3 / med
            );
            meds.push(med);
        }
        println!("CLOCKS run {run} {}", clocks());
    }
    for (on, meds) in table.iter_mut() {
        let med = median(meds);
        println!(
            "AR_MEDIAN graphs={} {med:.2} ms = {:.2} tok/s (median of {runs} runs of 30 steps, context ~{})",
            *on,
            1e3 / med,
            s.tokens.len()
        );
    }
    // the window table: ONE window per R (real text tokens after the pending one) run again and again with nothing committed (keep 0 changes no state),
    // alternating graphs off and on, so both modes time the very same work
    for r in 1..=9usize {
        let mut toks = vec![*s.tokens.last().unwrap()];
        toks.extend_from_slice(&code[40 + r * 9..40 + r * 9 + r - 1]);
        let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks, first_position: (s.tokens.len() - 1) as u64 }]);
        let mut t: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        for i in 0..24usize {
            let k = i % 2;
            graphs(&mut m, k == 1);
            let t0 = std::time::Instant::now();
            m.window_candidates(&b, 8).unwrap();
            m.commit(0, 0);
            if i >= 6 {
                t[k].push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        let (off, on) = (median(&mut t[0]), median(&mut t[1]));
        println!("WINDOW rows={r} off {off:.2} ms on {on:.2} ms (gain {:.2} ms)", off - on);
    }
    // the draft call: a window of 3 rows (committed) then the draft call that absorbs them and chains 2 drafts, on a slot that drafts
    let chat = ids_of(&oracle, "chat_300");
    let mut s = start(&mut m, 4, &long);
    let _ = chat;
    for on in [false, true] {
        graphs(&mut m, on);
        let mut dr = Vec::new();
        let mut win = Vec::new();
        for i in 0..16usize {
            let d: Vec<u32> = vec![code[(i * 5) % code.len()], code[(i * 5 + 1) % code.len()]];
            let mut toks = vec![*s.tokens.last().unwrap()];
            toks.extend_from_slice(&d);
            let b = RowBatch::from_requests(&[RowRequest { slot: 4, tokens: toks, first_position: (s.tokens.len() - 1) as u64 }]);
            let t0 = std::time::Instant::now();
            let c = m.window_candidates(&b, 8).unwrap();
            let w = t0.elapsed().as_secs_f64() * 1e3;
            m.commit(4, 3);
            s.tokens.extend_from_slice(&d);
            s.tokens.push(c.ids[2 * c.count] as u32);
            let t0 = std::time::Instant::now();
            let out = m.mtp_draft(&[DraftAsk { slot: 4, tokens: &s.tokens, depth: 2 }]).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            assert!(out[0].is_some(), "the slot drafts");
            if i >= 4 {
                dr.push(ms);
                win.push(w);
            }
        }
        println!(
            "ROUND graphs={on}: window of 3 rows {:.2} ms, draft call (absorb 3, K=2) {:.2} ms",
            median(&mut win),
            median(&mut dr)
        );
    }
    println!("CLOCKS end {}", clocks());
    if let Some(st) = perf_graphs::stats(&m) {
        println!(
            "GRAPHS captures {} replays {} eager {} failures {} capture_ms {:.1}",
            st.captures, st.replays, st.eager, st.failures, st.capture_ms
        );
    }
}
