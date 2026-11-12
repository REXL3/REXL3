//! Task TP-M: the two-rank TP-exact layer loop on one GPU: two `NemotronModel` shards in one process (two threads, an in-process
//! loopback `TpComm`), the same scripted calls fed to both in lockstep, and the digest of every call compared bit for bit with the
//! single-node engine's (`docs/NEMOTRON_TP.md` section 7, step 3).
//!
//!     REXL3_NEMO_OMNI_DIR=<pack> REXL3_NEMO_ORACLE_DIR=<ref-text-1k> [REXL3_NEMO_TP_PARTS=prefix,full] [REXL3_NEMO_TP_GRAPHS=1] \
//!       [REXL3_NEMO_TP_RANKS=2..8] \
//!       cargo test --release -p gb-model --test nemotron_omni_tp -- --ignored --nocapture --test-threads=1
//!
//! `REXL3_NEMO_TP_RANKS` (default 2, the verified group) runs the same tests as an R-way loopback group (`LoopbackGroup::new(R)`): R
//! threads, each its own `NemotronModel` from `load_tp` with rank r of R, the serving shard plan, every rank's digest compared with the
//! single node's, and the collectives each rank issued compared with the plan (`docs/NEMOTRON_TP.md` section 13, the N-way loopback of
//! `docs/CLUSTER_N_NODE_PLAN_2026-10-11.md` phase E). The 8-layer prefix runs at every R up to 8; `full` (all 88 layers against the
//! golden) is about 25 GiB a rank at three, 20 at four, 17 at five: R = 3, 4 and 5 fit a 128 GiB machine with the 10 GiB spare the
//! test's own check keeps, R = 8 (about 14 GiB a rank) does not. Other switches of this file: `REXL3_TP_ATTN=slice` (q and o sliced at
//! R >= 3, k and v whole), `REXL3_TP_EXPERT_MAP` (an ownership table made for R), `REXL3_NEMO_TP_GOLDEN` (the repaired pack's golden),
//! `REXL3_NEMO_TP_CORRUPT=1` (the negative control: one wrong word on the last rank must make the digest test fail; it then reports
//! `TP NEGATIVE CONTROL OK`), `REXL3_NEMO_TP_HEAD_FULL=1` (the head test over the whole trunk instead of the prefix: about 64 GiB for
//! the single-node reference, then R x 25 GiB). `nemotron_tp_group_rank_window_captures_into_a_graph_and_replays` captures and replays
//! the windows of rank 0 of an R-rank group (a communicator that sums nothing: the capture of the arms, not the numbers).
//!
//! * `prefix` (default): the 8-layer prefix `MEMEMEM*` (`layers: Some(8)`), fast. The reference is a single-node run of the same
//!   prefix, in this process, before the shards load.
//! * `full`: all 88 layers. The reference is the committed golden `testdata/gpu/nemotron_step_digest.txt` (121 lines, the original
//!   pack). Two shards of the full model share one device: each holds half the routed experts, about 2 x 36 GiB; check the plan's
//!   numbers against the device's free memory before running it.
//! * both ranks' lines must equal the reference (a replicated engine's two ranks hold the same bits), with graphs off. The loopback
//!   group synchronises its stream and the host inside a collective, so it cannot be captured: `REXL3_NEMO_TP_GRAPHS=1` (three more
//!   passes with graphs on: eager, capture, replay) is for a capturable group only.
//! * the collectives of a 1-row and a 5-row window are counted as the model issues them and printed (`TP COLLECTIVES`).
//!
//! Both shards run the identical call sequence on their own thread; the collectives inside the calls synchronise them. The shards of the
//! digest test do not load the MTP head (the head follows the whole trunk, two whole-trunk shards on one device are 95 GiB, and a loopback
//! of two shards on one device must not take a device-wide synchronisation while the other rank's stream waits in a collective); the
//! head has its own test at the end of this file, `nemotron_tp_head_loopback_is_the_single_node_head`, which puts it on an 8-layer
//! prefix (`REXL3_NEMO_HEAD_ON_PREFIX=1`, set by the test) and runs the production draft call, the debug chain and windows in the
//! loopback in every `REXL3_TP_HEAD` mode, comparing every digest line with the single node's (about 15 GiB, 20 s).
//!
//!     REXL3_NEMO_OMNI_DIR=<pack> REXL3_NEMO_ORACLE_DIR=<ref-text-1k> [REXL3_NEMO_TP_HEAD_MODES=split,replicate,experts,lm_head] \
//!       cargo test --release -p gb-model --test nemotron_omni_tp -- --ignored --nocapture --test-threads=1 nemotron_tp_head
#![allow(clippy::type_complexity)]

use gb_cuda::loader::Options;
use gb_cuda::{CudaError, DevBuf, Stream};
use gb_engine::RowRequest;
use gb_model::families::nemotron_h_omni::backend::ModelOptions;
use gb_model::families::nemotron_h_omni::bridge;
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::device::{HeadLinear, Mixer, NemotronWeights};
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::model::NemotronModel;
use gb_model::families::nemotron_h_omni::oracle::Dump;
use gb_model::families::nemotron_h_omni::tp::{SingleRank, TpComm, TpTopology};
use gb_model::families::nemotron_h_omni::tp_arm::{
    gather_head_rows, model_collectives, planned_collectives, serving_shard_plan, step_collectives, CountingComm,
};
use gb_model::families::nemotron_h_omni::tp_comm::{HostBarrier, LoopbackGroup};
use gb_model::families::nemotron_h_omni::tp_shard::head_row_cut;
use gb_model::families::nemotron_h_omni::weights::{self, LayerPlan, UnitId};
use gb_model::forward::Candidates;
use gb_model::rows::RowBatch;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

const SLOTS: usize = 8;
const KV_PAGES: usize = 64;
const MAX_CONTEXT: usize = 4096;
/// Candidates per row a window asks for (what the harness and the sampler use).
const COUNT: usize = 28;

fn env_parts() -> Vec<String> {
    std::env::var("REXL3_NEMO_TP_PARTS")
        .unwrap_or_else(|_| "prefix".into())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// The group size of the loopback tests: `REXL3_NEMO_TP_RANKS`, 2 when unset (the verified pair). 2 to 8.
fn env_ranks() -> usize {
    match std::env::var("REXL3_NEMO_TP_RANKS") {
        Err(_) => 2,
        Ok(v) => {
            let r: usize = v.trim().parse().unwrap_or_else(|_| panic!("REXL3_NEMO_TP_RANKS must be an integer, got `{v}`"));
            assert!((2..=8).contains(&r), "REXL3_NEMO_TP_RANKS must be 2 to 8, got {r}");
            r
        }
    }
}

fn cycle(src: &[u32], p: usize) -> u32 {
    src[p % src.len()]
}

fn reset(m: &mut NemotronModel) {
    for s in 0..SLOTS {
        m.end_slot(s);
    }
}

// ------------------------------------------------------------------------------------------------------------------ the digest

/// A 64-bit mix over the bytes fed to it (8 at a time). Not a cryptographic hash: a change detector.
#[derive(Clone, Copy)]
struct Dg(u64);

impl Dg {
    fn new() -> Self {
        Dg(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        let mut h = self.0;
        let mut it = b.chunks_exact(8);
        for c in &mut it {
            let w = u64::from_le_bytes(c.try_into().unwrap());
            h = (h ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            h ^= h >> 29;
        }
        for &x in it.remainder() {
            h = (h ^ x as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = h ^ (b.len() as u64);
    }
    fn f32s(&mut self, v: &[f32]) {
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
        self.bytes(b);
    }
    fn u16s(&mut self, v: &[u16]) {
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
        self.bytes(b);
    }
    fn i32s(&mut self, v: &[i32]) {
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
        self.bytes(b);
    }
}

struct Digest {
    lines: Vec<(String, u64)>,
    /// Print every line (the run whose lines are the reference).
    print: bool,
}

impl Digest {
    fn emit(&mut self, name: String, d: Dg) {
        if self.print {
            println!("DIGEST {name} {:016x}", d.0);
        }
        self.lines.push((name, d.0));
    }
}

fn hash_logits(m: &NemotronModel, rows: usize) -> Dg {
    let vocab = m.vocab_size();
    let mut v = vec![0f32; rows * vocab];
    m.core.bufs.logits.download_range(0, &mut v).unwrap();
    let mut d = Dg::new();
    d.f32s(&v);
    d
}

fn hash_cands(c: &Candidates) -> Dg {
    let mut d = Dg::new();
    d.i32s(&c.ids);
    d.f32s(&c.vals);
    d
}

/// The S and conv pools of `slot` in every Mamba layer.
fn hash_state(m: &mut NemotronModel, slot: usize) -> Dg {
    let mut d = Dg::new();
    for ord in 0..m.core.mamba.len() {
        let (s, conv) = m.debug_mamba_state(ord, slot).unwrap();
        d.f32s(&s);
        d.u16s(&conv);
    }
    d
}

/// The fp16 K and V rows `first..first + n` of `slot` in every attention layer.
fn hash_kv(m: &mut NemotronModel, slot: usize, first: usize, n: usize) -> Dg {
    let mut d = Dg::new();
    for ord in 0..m.core.kv_geom.layers {
        let (k, v) = m.debug_kv_rows(ord, slot, first, n).unwrap();
        d.u16s(&k);
        d.u16s(&v);
    }
    d
}

/// The first `rows` rows of every Mamba layer's window record and retained xBC rows (what a commit replays).
fn hash_records(m: &NemotronModel, rows: usize) -> Dg {
    use gb_cuda::nemotron_ssm::{CONV_DIM, HEADS};
    let mut d = Dg::new();
    let c = &m.core;
    for ord in 0..c.mamba.len() {
        let r = &c.win_rec[ord];
        let mut xbc = vec![0u16; rows * CONV_DIM];
        let mut dt = vec![0u16; rows * HEADS];
        let mut g = vec![0f32; rows * HEADS];
        let mut decay = vec![0f32; rows * HEADS];
        let mut xin = vec![0f32; rows * CONV_DIM];
        r.xbc.download_range(0, &mut xbc).unwrap();
        r.dt.download_range(0, &mut dt).unwrap();
        r.g.download_range(0, &mut g).unwrap();
        r.decay.download_range(0, &mut decay).unwrap();
        c.win_xbc[ord].download_range(0, &mut xin).unwrap();
        d.u16s(&xbc);
        d.u16s(&dt);
        d.f32s(&g);
        d.f32s(&decay);
        d.f32s(&xin);
    }
    d
}

/// One sequence of a scripted window.
struct Part<'a> {
    slot: usize,
    first: usize,
    src: &'a [u32],
    off: usize,
    rows: usize,
    keep: usize,
}

fn part_tokens(p: &Part) -> Vec<u32> {
    (0..p.rows).map(|j| cycle(p.src, p.off + p.first + j)).collect()
}

/// A verify window of several sequences, then the commits; digests the logits of every row, the candidates, the window records, and
/// after the commits the state and the KV rows of each sequence.
fn window_seg(m: &mut NemotronModel, dg: &mut Digest, name: &str, parts: &[Part]) {
    let reqs: Vec<RowRequest> =
        parts.iter().map(|p| RowRequest { slot: p.slot, tokens: part_tokens(p), first_position: p.first as u64 }).collect();
    let b = RowBatch::from_requests(&reqs);
    let c = m.window_candidates(&b, COUNT).unwrap_or_else(|e| panic!("{name}: {e}"));
    dg.emit(format!("{name}.logits"), hash_logits(m, b.rows));
    dg.emit(format!("{name}.cands"), hash_cands(&c));
    dg.emit(format!("{name}.records"), hash_records(m, b.rows));
    for p in parts {
        m.commit(p.slot, p.keep);
    }
    m.flush().unwrap_or_else(|e| panic!("{name}: {e}"));
    for p in parts {
        dg.emit(format!("{name}.state.s{}", p.slot), hash_state(m, p.slot));
        dg.emit(format!("{name}.kv.s{}", p.slot), hash_kv(m, p.slot, p.first, p.rows));
    }
}

/// A prompt prefilled in slices of 64 rows (the last slice asks for candidates); digests the last row's logits and candidates, and the
/// state and KV rows at the end. `mid` also hashes the state after the slice that ends at that position.
#[allow(clippy::too_many_arguments)]
fn prefill_seg(
    m: &mut NemotronModel,
    dg: &mut Digest,
    name: &str,
    slot: usize,
    src: &[u32],
    off: usize,
    n: usize,
    mid: Option<usize>,
) {
    prefill_seg_w(m, dg, name, slot, src, off, n, mid, 64)
}

/// [`prefill_seg`] in slices of `width` rows.
#[allow(clippy::too_many_arguments)]
fn prefill_seg_w(
    m: &mut NemotronModel,
    dg: &mut Digest,
    name: &str,
    slot: usize,
    src: &[u32],
    off: usize,
    n: usize,
    mid: Option<usize>,
    width: usize,
) {
    m.begin_slot(slot).unwrap_or_else(|e| panic!("{name}: begin_slot: {e}"));
    let mut at = 0;
    while at < n {
        let end = (at + width).min(n);
        let toks: Vec<u32> = (at..end).map(|p| cycle(src, off + p)).collect();
        let b = RowBatch::from_requests(&[RowRequest { slot, tokens: toks, first_position: at as u64 }]);
        let want = (end == n).then_some(end - at - 1);
        let c = m.prefill_chunk(&b, want).unwrap_or_else(|e| panic!("{name}: {e}"));
        if let Some(c) = c {
            m.flush().unwrap();
            dg.emit(format!("{name}.cands"), hash_cands(&c));
            dg.emit(format!("{name}.logits"), hash_logits(m, 1));
        }
        if mid == Some(end) {
            m.flush().unwrap();
            dg.emit(format!("{name}.state@{end}"), hash_state(m, slot));
        }
        at = end;
    }
    m.flush().unwrap_or_else(|e| panic!("{name}: {e}"));
    dg.emit(format!("{name}.state"), hash_state(m, slot));
    dg.emit(format!("{name}.kv"), hash_kv(m, slot, 0, n));
}

/// The scripted run, once. See the header of this file for what it covers.
fn digest_script(m: &mut NemotronModel, text: &[Vec<u32>; 2], print: bool) -> Vec<(String, u64)> {
    reset(m);
    let mut dg = Digest { lines: Vec::new(), print };
    let (a, b, c) = (0usize, 1usize, 2usize);
    let sa: &[u32] = &text[0];
    let sb: &[u32] = &text[1];
    // positions committed so far, per slot
    let mut pa = 1020usize;
    let mut pb = 505usize;
    let mut pc = 70usize;

    // 1. prompts: A crosses the 512-key chunk inside its prefill and ends 4 keys short of 1024; B ends 7 keys short of 512; C is short
    prefill_seg(m, &mut dg, "pA", a, sa, 0, pa, Some(512));
    prefill_seg(m, &mut dg, "pB", b, sb, 0, pb, None);
    prefill_seg(m, &mut dg, "pC", c, sb, 400, pc, None);

    // 2. serial decode steps on A, across the 1024-key chunk edge (positions 1020 .. 1025)
    for i in 0..5 {
        window_seg(m, &mut dg, &format!("serial{i}"), &[Part { slot: a, first: pa, src: sa, off: 0, rows: 1, keep: 1 }]);
        pa += 1;
    }

    // 3. windows of 2 to 9 rows with partial keeps
    for (i, (rows, keep)) in [(2usize, 1usize), (3, 3), (5, 2), (8, 8), (9, 4)].into_iter().enumerate() {
        window_seg(m, &mut dg, &format!("win{i}_{rows}"), &[Part { slot: a, first: pa, src: sa, off: 0, rows, keep }]);
        pa += keep;
    }

    // 4. B across its 512-key edge: the rows before the edge read one chunk and the rows after it two
    for (i, (rows, keep)) in [(9usize, 6usize), (3, 2), (4, 4)].into_iter().enumerate() {
        window_seg(m, &mut dg, &format!("edgeB{i}_{rows}"), &[Part { slot: b, first: pb, src: sb, off: 0, rows, keep }]);
        pb += keep;
    }

    // 5. three sequences in one window, in an order that is not the slot order
    window_seg(
        m,
        &mut dg,
        "multi3",
        &[
            Part { slot: c, first: pc, src: sb, off: 400, rows: 3, keep: 1 },
            Part { slot: a, first: pa, src: sa, off: 0, rows: 4, keep: 3 },
            Part { slot: b, first: pb, src: sb, off: 0, rows: 2, keep: 2 },
        ],
    );
    pc += 1;
    pa += 3;
    pb += 2;

    // 6. the widest windows: 64 rows of one sequence, a serial step, then 64 rows of two sequences
    window_seg(m, &mut dg, "w64_a", &[Part { slot: a, first: pa, src: sa, off: 0, rows: 64, keep: 37 }]);
    pa += 37;
    window_seg(m, &mut dg, "serial_after64", &[Part { slot: a, first: pa, src: sa, off: 0, rows: 1, keep: 1 }]);
    pa += 1;
    window_seg(
        m,
        &mut dg,
        "w64_ab",
        &[
            Part { slot: a, first: pa, src: sa, off: 0, rows: 32, keep: 32 },
            Part { slot: b, first: pb, src: sb, off: 0, rows: 32, keep: 5 },
        ],
    );
    pa += 32;

    // 7. a prefill slice of two sequences in one batch (the slice of a batch of fresh prompts)
    {
        m.begin_slot(3).unwrap();
        m.begin_slot(4).unwrap();
        let t3: Vec<u32> = (0..40).map(|p| cycle(sa, 100 + p)).collect();
        let t4: Vec<u32> = (0..24).map(|p| cycle(sb, 700 + p)).collect();
        let bt = RowBatch::from_requests(&[
            RowRequest { slot: 3, tokens: t3, first_position: 0 },
            RowRequest { slot: 4, tokens: t4, first_position: 0 },
        ]);
        let cands = m.prefill_chunk(&bt, Some(63)).unwrap().expect("candidates");
        m.flush().unwrap();
        dg.emit("pM.cands".into(), hash_cands(&cands));
        dg.emit("pM.logits".into(), hash_logits(m, 1));
        dg.emit("pM.state.s3".into(), hash_state(m, 3));
        dg.emit("pM.state.s4".into(), hash_state(m, 4));
        dg.emit("pM.kv.s3".into(), hash_kv(m, 3, 0, 40));
        dg.emit("pM.kv.s4".into(), hash_kv(m, 4, 0, 24));
    }

    // 8. a window over the slots of the slice and a long sequence together
    window_seg(
        m,
        &mut dg,
        "multi4",
        &[
            Part { slot: 3, first: 40, src: sa, off: 100, rows: 2, keep: 2 },
            Part { slot: 4, first: 24, src: sb, off: 700, rows: 1, keep: 1 },
            Part { slot: a, first: pa, src: sa, off: 0, rows: 1, keep: 1 },
            Part { slot: c, first: pc, src: sb, off: 400, rows: 6, keep: 6 },
        ],
    );

    reset(m);
    dg.lines
}
// ------------------------------------------------------------------------------------------------------------------ the shards

fn texts() -> [Vec<u32>; 2] {
    let dir = PathBuf::from(std::env::var_os("REXL3_NEMO_ORACLE_DIR").expect("REXL3_NEMO_ORACLE_DIR"));
    let get = |s: usize| Dump::open(&dir, s).and_then(|d| d.ids()).unwrap_or_else(|e| panic!("oracle ids of sample {s}: {e}"));
    [get(0), get(1)]
}

fn config(dir: &Path) -> OmniConfig {
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| panic!("config.json: {e}"));
    OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"))
}

fn options(layers: Option<usize>) -> ModelOptions {
    // The model's prefill cap, as a server and the single-node digest (nemotron_omni_perf) run it (512 by default; 256 under
    // REXL3_NEMO_PREFILL_ROWS_MAX=256, bit-identical). NOT 64, although the digest's own slices are 64 rows: with a cap of 64 or less the
    // model allocates neither the large-row MoE prefill scratch nor the P2 tensor-core block's (model.rs, `rows_cap > WINDOW_ROWS`), so
    // under P2 (the default) every prefill slice silently ran the MoE on P1's fp32 kernels beside P2's dense and Mamba arithmetic
    // (arm_moe.rs `p2_prefill` needs the large-row scratch): a hybrid no server runs, and the P2 golden taken from this test was that
    // hybrid's (lane D, 2026-10-15). Under P1 the cap changes no bit (64-row slices take the decode path). No MTP head (see the header).
    ModelOptions {
        slots: SLOTS,
        kv_pages: KV_PAGES,
        max_context: MAX_CONTEXT,
        prefill_rows: gb_model::families::nemotron_h_omni::model::prefill_rows_cap(),
        layers,
        mtp: false,
    }
}

/// The two comms of an in-process loopback group on this device (TP-N's `LoopbackGroup`: host barriers and stream syncs inside the
/// call, so it is not capturable: the shards run with graphs off).
fn loopback_pair() -> Result<[Arc<dyn TpComm>; 2], String> {
    let (a, b) = LoopbackGroup::pair();
    Ok([Arc::new(a), Arc::new(b)])
}

/// A comm that is the inner one except that after its `at`-th allreduce (counting from 0) it overwrites four bytes of the reduced range
/// with another value: a rank that holds one wrong word. The negative control of the digest test (`REXL3_NEMO_TP_CORRUPT=1` on the last
/// rank): the test must then FAIL for that rank's digest, and only for it, or it proves nothing.
struct CorruptComm {
    inner: Arc<dyn TpComm>,
    at: usize,
    calls: std::sync::atomic::AtomicUsize,
}

impl TpComm for CorruptComm {
    fn topology(&self) -> TpTopology {
        self.inner.topology()
    }

    fn allreduce_sum_f32(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
        self.inner.allreduce_sum_f32(buf, off_bytes, count, st)?;
        if self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == self.at {
            buf.memset_range_async(off_bytes, 4, 0x3f, st)?;
        }
        Ok(())
    }

    fn abort(&self) {
        self.inner.abort();
    }
}

/// The `ranks` comms of an in-process loopback group (`LoopbackGroup::new`, rank 0 first): one per shard thread.
/// Two ranks under P2 with the shared-expert split on (`tp_arm::shared_split_wanted`, as a boot decides) get a second communicator each,
/// as `tp_boot::make_comm` gives them (lane S).
fn loopback_group(ranks: usize) -> Vec<Arc<dyn TpComm>> {
    if ranks == 2 && gb_model::families::nemotron_h_omni::tp_arm::shared_split_wanted(2) {
        eprintln!("loopback group: two ranks with a second communicator (the shared-expert split)");
        let (a, b) = LoopbackGroup::pair_with_aux();
        return vec![Arc::new(a) as Arc<dyn TpComm>, Arc::new(b) as Arc<dyn TpComm>];
    }
    LoopbackGroup::new(ranks).into_iter().map(|c| Arc::new(c) as Arc<dyn TpComm>).collect()
}

/// Prints the lane-S path counters when a test ends (a split that fell back to the whole path would match too: the gate needs them > 0).
struct CountersAtEnd;

impl Drop for CountersAtEnd {
    fn drop(&mut self) {
        eprintln!("{}", gb_model::families::nemotron_h_omni::tp_arm::p2tp_counters_line());
    }
}

/// What a shard hands back.
struct ShardRun {
    /// The digest with graphs off.
    eager: Vec<(String, u64)>,
    /// The three graph passes' lines (empty when graphs are off).
    graph_passes: Vec<Vec<(String, u64)>>,
    stats: gb_model::families::nemotron_h_omni::model::GraphStats,
    /// `(rows, calls, elements, planned calls)` of the windows measured (rank 0 only); the plan is `model_collectives` of the weights
    /// this rank holds.
    collectives: Vec<(usize, usize, usize, usize)>,
    /// Device bytes of the shard's buffers and pools.
    pools: u64,
}

#[allow(clippy::too_many_arguments)]
fn run_shard(
    dir: &Path,
    cfg: &OmniConfig,
    plan: &LayerPlan,
    layers: Option<usize>,
    comm: Arc<dyn TpComm>,
    counting: Option<Arc<CountingComm>>,
    ready: Arc<HostBarrier>,
    text: &[Vec<u32>; 2],
    graphs: bool,
) -> ShardRun {
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let rank = comm.topology().rank;
    let t = Instant::now();
    let comm: Arc<dyn TpComm> = match &counting {
        Some(c) => c.clone(),
        None => comm,
    };
    // a rank that fails to load must not leave the other waiting at the barrier (or in a collective) while this test holds the GPU lock
    let loaded = catch_unwind(AssertUnwindSafe(|| {
        NemotronModel::load_tp(dir, cfg, plan, options(layers), comm).unwrap_or_else(|e| panic!("rank {rank}: load: {e}"))
    }));
    let mut m = match loaded {
        Ok(m) => m,
        Err(p) => {
            ready.break_now();
            resume_unwind(p)
        }
    };
    let pools = m.pool_bytes().total();
    eprintln!("rank {rank}: loaded in {:.1} s, pools {:.0} MiB", t.elapsed().as_secs_f64(), pools as f64 / 1048576.0);
    // no shard starts a step while the other is still in a device-wide synchronisation of its own load
    ready.wait(Duration::from_secs(1800)).unwrap_or_else(|e| panic!("rank {rank}: the other rank did not finish loading: {e:?}"));

    m.set_graphs(false);
    let t = Instant::now();
    let eager = digest_script(&mut m, text, rank == 0);
    eprintln!("rank {rank}: digest script {} lines in {:.1} s", eager.len(), t.elapsed().as_secs_f64());

    let mut graph_passes = Vec::new();
    if graphs {
        m.set_graphs(true);
        m.clear_graphs().unwrap();
        for pass in 1..=3 {
            let t = Instant::now();
            graph_passes.push(digest_script(&mut m, text, false));
            let s = m.graph_stats();
            eprintln!(
                "rank {rank}: graph pass {pass}: {:.1} s; captures {}, replays {}, eager {}, capture failures {}",
                t.elapsed().as_secs_f64(),
                s.captures,
                s.replays,
                s.eager_windows,
                s.capture_failures
            );
        }
        m.set_graphs(false);
    }

    // the collectives of a 1-row and a 5-row window, as the model issues them
    let mut collectives = Vec::new();
    {
        let src: &[u32] = &text[0];
        m.begin_slot(0).unwrap();
        let toks: Vec<u32> = (0..60).map(|p| cycle(src, p)).collect();
        let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks, first_position: 0 }]);
        m.prefill_chunk(&b, None).unwrap();
        m.flush().unwrap();
        let mut at = 60usize;
        for rows in [1usize, 5] {
            if let Some(c) = &counting {
                c.take();
            }
            let toks: Vec<u32> = (0..rows).map(|p| cycle(src, at + p)).collect();
            let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks, first_position: at as u64 }]);
            m.window_candidates(&b, COUNT).unwrap();
            m.commit(0, rows);
            m.flush().unwrap();
            at += rows;
            if let Some(c) = &counting {
                let calls = c.take();
                let plan: usize =
                    model_collectives(&m.core.w, m.core.tp.topology(), rows, rows, cfg.vocab).iter().map(|c| c.calls).sum();
                collectives.push((rows, calls.len(), calls.iter().map(|(n, _)| n).sum::<usize>(), plan));
            }
        }
        m.end_slot(0);
    }
    ShardRun { eager, graph_passes, stats: m.graph_stats(), collectives, pools }
}

/// `want` against `got`, line by line; the mismatches by name.
fn compare(what: &str, want: &[(String, u64)], got: &[(String, u64)]) -> Vec<String> {
    let mut bad = Vec::new();
    if want.len() != got.len() {
        bad.push(format!("{what}: {} lines, the reference has {}", got.len(), want.len()));
    }
    for ((n, h), (n2, h2)) in want.iter().zip(got) {
        if n != n2 || h != h2 {
            bad.push(format!("{what}: {n}: reference {h:016x}, got {n2} {h2:016x}"));
        }
    }
    bad
}

/// The committed 121-line golden of the ORIGINAL pack, or the file `REXL3_NEMO_TP_GOLDEN` names (the repaired pack's
/// `testdata/gpu/nemotron_step_digest_repaired.txt`: a path relative to the repository root or absolute).
fn golden() -> Vec<(String, u64)> {
    let root = repo_root();
    let path = match std::env::var_os("REXL3_NEMO_TP_GOLDEN") {
        Some(p) => root.join(p),
        None => root.join("testdata/gpu/nemotron_step_digest.txt"),
    };
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .filter_map(|l| l.split_once(' '))
        .map(|(n, h)| (n.to_string(), u64::from_str_radix(h.trim(), 16).unwrap()))
        .collect()
}

const GIB: f64 = (1u64 << 30) as f64;

/// `MemAvailable` of the host, bytes (page cache that can be dropped counts: weights are read once and mapped by the loader).
fn mem_available() -> u64 {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    text.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map_or(0, |kb| kb * 1024)
}

/// The bytes of the trunk (the first `layers`, or all of it) and the globals that rank `rank` of `ranks` holds under the serving plan
/// (the experts of its range, the dense matrices it slices or holds whole, the embedding and the `lm_head` rows): what the memory plan
/// prices. No MTP head.
fn rank_planned_bytes(plan: &LayerPlan, cfg: &OmniConfig, layers: Option<usize>, rank: usize, ranks: usize) -> u64 {
    let n = layers.unwrap_or(usize::MAX);
    let shard = serving_shard_plan(TpTopology { rank, ranks }, cfg);
    shard.rank_bytes(plan, cfg.experts, |u| match u {
        UnitId::Trunk(i) => i < n,
        UnitId::Embed | UnitId::NormF | UnitId::LmHead => true,
        UnitId::Mtp(_) => false,
    })
}

/// What the `ranks` shards take together, bytes: the pack's planned tensors (the experts split between the ranks, everything else counted
/// once; a rank also holds the replicated parts, a few GiB each, which the per-rank plan counts), the pools of the models and 16 GiB
/// left for the host. A prefix is priced by its share of the layers. The larger of the old pair estimate and the sum of the ranks' own
/// planned bytes plus a GiB of pools each.
fn group_bytes(plan: &LayerPlan, inv: &Inventory, cfg: &OmniConfig, layers: Option<usize>, ranks: usize) -> u64 {
    let total = plan.bytes(inv);
    let share = match layers {
        Some(n) => total / 88 * n as u64 + (2 << 30),
        None => total,
    };
    let held: u64 = (0..ranks).map(|r| rank_planned_bytes(plan, cfg, layers, r, ranks) + (1 << 30)).sum();
    share.max(held) + (8 << 30) + (16 << 30)
}

#[allow(clippy::too_many_arguments)]
fn run_part(dir: &Path, cfg: &OmniConfig, plan: &LayerPlan, inv: &Inventory, part: &str, text: &[Vec<u32>; 2], ranks: usize) {
    let layers = match part {
        "prefix" => Some(8),
        "full" => None,
        other => panic!("REXL3_NEMO_TP_PARTS: unknown part `{other}`"),
    };
    // the loopback group is not capturable (it synchronises its stream and the host inside a collective): graphs are opt-in, for a
    // capturable group
    let graphs = std::env::var("REXL3_NEMO_TP_GRAPHS").is_ok_and(|v| v == "1");
    if !NemotronModel::TP_WEIGHTS_BUILT {
        eprintln!(
            "not run: this tree has no per-rank weights (nemotron/tp-weights is not merged: `NemotronModel::TP_WEIGHTS_BUILT`)"
        );
        return;
    }
    // the shards of a model share one device and its memory (a GB10's device memory is host memory): refuse before loading anything
    // when the group, the pools and a margin do not fit what the host can give
    let need = group_bytes(plan, inv, cfg, layers, ranks);
    let have = mem_available();
    let held: Vec<f64> =
        (0..ranks).map(|r| (rank_planned_bytes(plan, cfg, layers, r, ranks) as f64 / GIB * 100.0).round() / 100.0).collect();
    eprintln!(
        "TP {part}: {ranks} shards; planned weights a rank (GiB) {held:?}; need about {:.1} GiB, the host has {:.1} GiB available",
        need as f64 / GIB,
        have as f64 / GIB
    );
    if have < need {
        eprintln!(
            "not run: the {ranks} shards need about {:.1} GiB and only {:.1} GiB is available",
            need as f64 / GIB,
            have as f64 / GIB
        );
        return;
    }
    // the reference: the single-node engine's digest of the same prefix, or the committed golden of the whole model
    let reference = if layers.is_some() {
        let mut m = NemotronModel::load(dir, cfg, plan, options(layers)).unwrap_or_else(|e| panic!("single-node load: {e}"));
        m.set_graphs(false);
        let lines = digest_script(&mut m, text, false);
        drop(m);
        eprintln!("reference: single-node, {} lines", lines.len());
        lines
    } else {
        let g = golden();
        eprintln!("reference: the golden, {} lines", g.len());
        g
    };

    let group = loopback_group(ranks);
    let ready = Arc::new(HostBarrier::new(ranks));
    let corrupt = std::env::var("REXL3_NEMO_TP_CORRUPT").is_ok_and(|v| v == "1");
    let t = Instant::now();
    let runs: Vec<ShardRun> = std::thread::scope(|s| {
        let handles: Vec<_> = group
            .into_iter()
            .map(|comm| {
                // the negative control: the last rank holds one wrong word after its 8th collective (index 7) (`REXL3_NEMO_TP_CORRUPT=1`)
                let comm: Arc<dyn TpComm> = if corrupt && comm.topology().rank == ranks - 1 {
                    Arc::new(CorruptComm { inner: comm, at: 7, calls: std::sync::atomic::AtomicUsize::new(0) })
                } else {
                    comm
                };
                // every rank's collectives are counted: the plan holds for each of them, and they agree with each other
                let counting = Arc::new(CountingComm::new(comm.clone()));
                let ready = ready.clone();
                s.spawn(move || run_shard(dir, cfg, plan, layers, comm, Some(counting), ready, text, graphs))
            })
            .collect();
        handles.into_iter().enumerate().map(|(r, h)| h.join().unwrap_or_else(|_| panic!("rank {r} panicked"))).collect()
    });
    eprintln!(
        "TP {part}: {ranks} shards in {:.1} s; pools {:?} MiB",
        t.elapsed().as_secs_f64(),
        runs.iter().map(|r| (r.pools as f64 / 1048576.0).round()).collect::<Vec<_>>()
    );

    let mut bad = Vec::new();
    for (rank, r) in runs.iter().enumerate() {
        bad.extend(compare(&format!("rank {rank} eager"), &reference, &r.eager));
        for (p, lines) in r.graph_passes.iter().enumerate() {
            bad.extend(compare(&format!("rank {rank} graph pass {}", p + 1), &reference, lines));
        }
    }
    let n_layers = layers.unwrap_or_else(|| cfg.pattern.chars().count());
    let pattern: String = cfg.pattern.chars().take(n_layers).collect();
    let sig = |r: &ShardRun| r.collectives.iter().map(|&(a, b, c, _)| (a, b, c)).collect::<Vec<_>>();
    for (rank, run) in runs.iter().enumerate() {
        for (rows, calls, elements, planned) in &run.collectives {
            // every sharded shape sliced is the most the model can issue: shapes TP-W keeps replicated issue fewer
            let all: usize = step_collectives(&pattern, *rows, *rows, cfg.vocab).iter().map(|c| c.calls).sum();
            if rank == 0 {
                eprintln!(
                    "TP COLLECTIVES {part} R={ranks}: a {rows}-row window issues {calls} allreduces of {elements} f32 ({:.2} MB); \
                     planned from the weights {planned}, with every shape sliced {all}",
                    (*elements * 4) as f64 / 1e6
                );
            }
            assert_eq!(
                calls, planned,
                "{part} R={ranks} rank {rank}: a {rows}-row window issued {calls} collectives, the weights' plan says {planned}"
            );
            // the same count from the shard plan and the config alone (no weights): the plan, the weights and what was issued agree
            let mut pcfg = cfg.clone();
            pcfg.pattern = pattern.clone();
            let shard_plan = serving_shard_plan(TpTopology { rank, ranks }, cfg);
            let from_plan: usize = planned_collectives(&shard_plan, &pcfg, *rows, *rows).iter().map(|c| c.calls).sum();
            assert_eq!(
                calls, &from_plan,
                "{part} R={ranks} rank {rank}: a {rows}-row window issued {calls} collectives, the shard plan says {from_plan}"
            );
        }
        // every rank issues what rank 0 does (a rank that skipped a collective its peers run would hold the group in it)
        assert_eq!(sig(run), sig(&runs[0]), "{part} R={ranks}: rank {rank} issued other collectives than rank 0");
    }
    if graphs {
        let st = &runs[0].stats;
        assert!(st.captures > 0 && st.replays > 0, "the graph passes captured {} and replayed {}", st.captures, st.replays);
        assert_eq!(st.capture_failures, 0, "a capture failed: {:?}", st.last_failure);
    }
    if corrupt {
        // the negative control: one wrong word on one rank must show in the digest lines
        let first = bad.first().cloned().unwrap_or_default();
        assert!(!bad.is_empty(), "negative control: one corrupted word on rank {} went undetected", ranks - 1);
        eprintln!("TP NEGATIVE CONTROL OK ({part}, R={ranks}): {} digest lines differ; the first: {first}", bad.len());
        return;
    }
    for b in &bad {
        eprintln!("TP DIGEST MISMATCH {b}");
    }
    assert!(bad.is_empty(), "{} digest lines differ from the single-node engine's", bad.len());
    eprintln!(
        "TP DIGEST OK ({part}, R={ranks}): {} lines, all {ranks} ranks, graphs {}: identical to the single-node engine",
        reference.len(),
        if graphs { "off and on" } else { "off" }
    );
}

#[test]
#[ignore]
fn nemotron_tp_loopback_is_the_single_node_engine() {
    let _counters = CountersAtEnd;
    let Some(dir) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set");
        return;
    };
    if std::env::var_os("REXL3_NEMO_ORACLE_DIR").is_none() {
        eprintln!("not run: REXL3_NEMO_ORACLE_DIR is not set");
        return;
    }
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let text = texts();
    let ranks = env_ranks();
    for part in env_parts() {
        run_part(&dir, &cfg, &plan, &inv, &part, &text, ranks);
    }
}

// ------------------------------------------------------------------------------------------------------------------ hand-sliced, host loopback

// The one-layer test below slices the weights by hand (`Exl3Weights::select_strips` for the `in_proj`, the rows of the `lm_head`
// copied) and keeps the rest of the model whole, so the arms, the scatter and the head split are proved without the other workers'
// sliced weights.

/// Trunk layer 0 (a Mamba layer) and the globals on the device, whole.
fn whole_weights(dir: &Path, cfg: &OmniConfig, plan: &LayerPlan, inv: &Inventory) -> NemotronWeights {
    let pick = |u: UnitId| match u {
        UnitId::Trunk(i) => i < 1,
        UnitId::Embed | UnitId::NormF | UnitId::LmHead => true,
        UnitId::Mtp(_) => false,
    };
    let mut up = weights::upload(dir, inv, plan, pick, &Options::from_env()).unwrap_or_else(|e| panic!("upload: {e}"));
    let st = Stream::new_nonblocking().unwrap();
    let w = bridge::into_device(&mut up, cfg, plan, 0..1, &st).unwrap_or_else(|e| panic!("bridge: {e}"));
    st.sync().unwrap();
    w
}

/// Rank `rank` of `ranks`: the rows `head_row_cut(vocab, rank, ranks)` of the `lm_head` (what the loader hands a rank), the rest of `w`
/// whole. At two ranks the halves of [`slice_weights`].
fn slice_head_rows(mut w: NemotronWeights, rank: usize, ranks: usize, vocab: usize) -> NemotronWeights {
    let row = 4096 * 2;
    let (lo, hi) = head_row_cut(vocab, rank, ranks);
    let st = Stream::new_nonblocking().unwrap();
    let mut head = DevBuf::alloc((hi - lo) * row).unwrap();
    head.copy_range_from(0, w.lm_head.bf16().expect("a bf16 lm_head"), lo * row, (hi - lo) * row, &st).unwrap();
    st.sync().unwrap();
    w.lm_head = HeadLinear::Bf16(head);
    w
}

/// Rank `rank`'s share of `w`: the Mamba layer's `in_proj` columns `[0, 9216)` or `[9216, 18560)` and half the rows of the `lm_head`.
fn slice_weights(mut w: NemotronWeights, rank: usize, vocab: usize) -> NemotronWeights {
    let Mixer::Mamba(m) = &mut w.layers[0].mixer else { panic!("layer 0 is not a Mamba layer") };
    let strips = if rank == 0 { (0, 72) } else { (72, 145) };
    m.in_proj = m.in_proj.select_strips(&[strips]).unwrap_or_else(|e| panic!("select_strips: {e}"));
    let row = 4096 * 2;
    let half = vocab / 2;
    let st = Stream::new_nonblocking().unwrap();
    let mut head = DevBuf::alloc(half * row).unwrap();
    head.copy_range_from(0, w.lm_head.bf16().expect("a bf16 lm_head"), rank * half * row, half * row, &st).unwrap();
    st.sync().unwrap();
    w.lm_head = HeadLinear::Bf16(head);
    w
}

#[test]
#[ignore]
fn nemotron_tp_one_layer_hand_sliced_host_loopback_is_the_single_node_engine() {
    let Some(dir) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set");
        return;
    };
    if std::env::var_os("REXL3_NEMO_ORACLE_DIR").is_none() {
        eprintln!("not run: REXL3_NEMO_ORACLE_DIR is not set");
        return;
    }
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let text = texts();
    let opts = options(Some(1));

    // the single-node reference: the same layer and globals, whole
    let mut m = NemotronModel::new(whole_weights(&dir, &cfg, &plan, &inv), &cfg, opts.clone()).unwrap_or_else(|e| panic!("{e}"));
    m.set_graphs(false);
    let reference = digest_script(&mut m, &text, false);
    drop(m);
    eprintln!("reference: single-node, one Mamba layer, {} lines", reference.len());

    // two ranks of one group, each with its slices, in lockstep
    let [c0, c1] = loopback_pair().unwrap();
    // both ranks go through a CountingComm, as in the full test (`nemotron_tp_loopback_is_the_single_node_engine`): the wrapper forwards
    // the allreduce only, not the byte swap, so a rank behind it runs P2's `in_proj` exchange in its allreduce form (`linear_tp`, the
    // same bits by `linear_tp_p2`'s contract) while a bare LoopbackComm takes the swap form; with only rank 0 wrapped the two ranks
    // issued different collectives under P2 ("rank 0 reduces 1187840 elements, rank 1 1212416": an allreduce against a swap). So the
    // loopback tests exercise the allreduce form of that exchange; its swap form runs on the NCCL comm of the two-node server legs
    // (scripts/nemotron/check_tp.sh, tools/nemotron_tp_verify.sh), whose replies equal the single node's
    let counting = Arc::new(CountingComm::new(c0));
    let comms: [Arc<dyn TpComm>; 2] = [counting.clone(), Arc::new(CountingComm::new(c1))];
    let ready = Arc::new(HostBarrier::new(2));
    let runs: Vec<(Vec<(String, u64)>, usize)> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                let (dir, cfg, plan, inv, text, opts, ready, counting) =
                    (&dir, &cfg, &plan, &inv, &text, opts.clone(), ready.clone(), counting.clone());
                s.spawn(move || {
                    gb_cuda::rt::init_context().unwrap();
                    let w = slice_weights(whole_weights(dir, cfg, plan, inv), rank, cfg.vocab);
                    let mut m = NemotronModel::new_tp(w, cfg, opts, comm).unwrap_or_else(|e| panic!("rank {rank}: {e}"));
                    m.set_graphs(false);
                    ready.wait(Duration::from_secs(600)).expect("the other rank did not finish loading");
                    let lines = digest_script(&mut m, text, false);
                    let calls = if rank == 0 { counting.take().len() } else { 0 };
                    (lines, calls)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a rank panicked")).collect()
    });
    let mut bad = Vec::new();
    for (rank, (lines, _)) in runs.iter().enumerate() {
        bad.extend(compare(&format!("rank {rank}"), &reference, lines));
    }
    for b in &bad {
        eprintln!("TP DIGEST MISMATCH {b}");
    }
    assert!(bad.is_empty(), "{} digest lines differ from the single-node engine's", bad.len());
    eprintln!(
        "TP ONE-LAYER OK: {} lines identical on both ranks; rank 0 issued {} allreduces over the script",
        reference.len(),
        runs[0].1
    );
}

/// Rank 0 of a group of two whose collective does nothing: not exact (nothing is summed), but a launch like any other, so a window that
/// runs through it can be captured into a CUDA graph. It is what a capture test of the arms needs on one GPU, where the NCCL group
/// cannot run (NCCL refuses two ranks on one device) and the loopback group is not capturable.
struct NoopPair;

impl TpComm for NoopPair {
    fn topology(&self) -> TpTopology {
        TpTopology { rank: 0, ranks: 2 }
    }

    fn allreduce_sum_f32(&self, _buf: &mut DevBuf, _off: usize, _count: usize, _st: &Stream) -> Result<(), CudaError> {
        Ok(())
    }
}

#[test]
#[ignore]
fn nemotron_tp_one_layer_window_captures_into_a_graph_and_replays() {
    let Some(dir) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set");
        return;
    };
    if std::env::var_os("REXL3_NEMO_ORACLE_DIR").is_none() {
        eprintln!("not run: REXL3_NEMO_ORACLE_DIR is not set");
        return;
    }
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let text = texts();
    let w = slice_weights(whole_weights(&dir, &cfg, &plan, &inv), 0, cfg.vocab);
    let mut m = NemotronModel::new_tp(w, &cfg, options(Some(1)), Arc::new(NoopPair)).unwrap_or_else(|e| panic!("{e}"));
    m.set_graphs(true);
    m.begin_slot(0).unwrap();
    let src: &[u32] = &text[0];
    let toks: Vec<u32> = (0..16).map(|p| cycle(src, p)).collect();
    let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks, first_position: 0 }]);
    m.prefill_chunk(&b, None).unwrap();
    // the same window shape three times: eager, capture + launch, replay. A slice's scatter (memset + copy), the collective and the
    // head's split are all launches of the captured stream.
    let mut at = 16usize;
    for _ in 0..3 {
        let toks: Vec<u32> = (0..2).map(|p| cycle(src, at + p)).collect();
        let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks, first_position: at as u64 }]);
        m.window_candidates(&b, COUNT).unwrap();
        m.commit(0, 2);
        m.flush().unwrap();
        at += 2;
    }
    let s = m.graph_stats();
    eprintln!(
        "TP CAPTURE: captures {}, replays {}, eager windows {}, capture failures {}",
        s.captures, s.replays, s.eager_windows, s.capture_failures
    );
    assert_eq!(s.capture_failures, 0, "a capture failed: {:?}", s.last_failure);
    assert!(s.captures == 1 && s.replays >= 1, "captured {} and replayed {}", s.captures, s.replays);
}

/// Rank 0 of a group of `ranks` whose collective does nothing (see [`NoopPair`]): the plan's shard of this rank (`load_tp` is generic
/// over the group), and a launch like any other for a capture.
struct NoopGroup {
    ranks: usize,
}

impl TpComm for NoopGroup {
    fn topology(&self) -> TpTopology {
        TpTopology { rank: 0, ranks: self.ranks }
    }

    fn allreduce_sum_f32(&self, _buf: &mut DevBuf, _off: usize, _count: usize, _st: &Stream) -> Result<(), CudaError> {
        Ok(())
    }
}

/// Windows of rank 0 of an R-rank group (`REXL3_NEMO_TP_RANKS`, 8-layer prefix, the serving plan's shard) captured into a CUDA graph and
/// replayed: the arms at R >= 3 (`in_proj` whole, the uneven `lm_head` cut and its scatter, the experts' ranges, `check_buffers`) launch
/// nothing that cannot be captured. Not exact (nothing is summed), so only the capture is asserted.
#[test]
#[ignore]
fn nemotron_tp_group_rank_window_captures_into_a_graph_and_replays() {
    let Some(dir) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set");
        return;
    };
    if std::env::var_os("REXL3_NEMO_ORACLE_DIR").is_none() || !gb_cuda::device_available() {
        eprintln!("not run: no oracle ids or no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let ranks = env_ranks();
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let text = texts();
    let mut m = NemotronModel::load_tp(&dir, &cfg, &plan, options(Some(8)), Arc::new(NoopGroup { ranks }))
        .unwrap_or_else(|e| panic!("rank 0 of {ranks}: {e}"));
    m.set_graphs(true);
    m.begin_slot(0).unwrap();
    let src: &[u32] = &text[0];
    let toks: Vec<u32> = (0..16).map(|p| cycle(src, p)).collect();
    let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks, first_position: 0 }]);
    m.prefill_chunk(&b, None).unwrap();
    // the same window shape three times: eager, capture + launch, replay
    let mut at = 16usize;
    for _ in 0..3 {
        let toks: Vec<u32> = (0..2).map(|p| cycle(src, at + p)).collect();
        let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks, first_position: at as u64 }]);
        m.window_candidates(&b, COUNT).unwrap();
        m.commit(0, 2);
        m.flush().unwrap();
        at += 2;
    }
    let s = m.graph_stats();
    eprintln!(
        "TP CAPTURE R={ranks}: captures {}, replays {}, eager windows {}, capture failures {}",
        s.captures, s.replays, s.eager_windows, s.capture_failures
    );
    assert_eq!(s.capture_failures, 0, "a capture failed: {:?}", s.last_failure);
    assert!(s.captures == 1 && s.replays >= 1, "captured {} and replayed {}", s.captures, s.replays);
}

/// One window of three rows after a 40-token prompt, and the `in_proj` output (`bufs.proj`, `[3, 18560]`) it left: the buffer the TP
/// scatter and allreduce fill. `None` for the model on a single rank.
fn proj_after_window(m: &mut NemotronModel, text: &[Vec<u32>; 2]) -> Vec<f32> {
    let src: &[u32] = &text[0];
    m.begin_slot(0).unwrap();
    let toks: Vec<u32> = (0..40).map(|p| cycle(src, p)).collect();
    let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks, first_position: 0 }]);
    m.prefill_chunk(&b, None).unwrap();
    let toks: Vec<u32> = (0..3).map(|p| cycle(src, 40 + p)).collect();
    let b = RowBatch::from_requests(&[RowRequest { slot: 0, tokens: toks, first_position: 40 }]);
    m.window_candidates(&b, COUNT).unwrap();
    m.flush().unwrap();
    let mut v = vec![0f32; 3 * 18560];
    m.core.bufs.proj.download_range(0, &mut v).unwrap();
    m.end_slot(0);
    v
}

#[test]
#[ignore]
fn nemotron_tp_one_layer_diagnose_in_proj_bits() {
    let Some(dir) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set");
        return;
    };
    if std::env::var_os("REXL3_NEMO_ORACLE_DIR").is_none() || !gb_cuda::device_available() {
        eprintln!("not run: no oracle ids or no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let text = texts();
    let opts = options(Some(1));
    let mut m = NemotronModel::new(whole_weights(&dir, &cfg, &plan, &inv), &cfg, opts.clone()).unwrap();
    m.set_graphs(false);
    let single = proj_after_window(&mut m, &text);
    drop(m);

    let (a, b) = LoopbackGroup::pair();
    let comms: [Arc<dyn TpComm>; 2] = [Arc::new(a), Arc::new(b)];
    let ready = Arc::new(HostBarrier::new(2));
    let outs: Vec<Vec<f32>> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                let (dir, cfg, plan, inv, text, opts, ready) = (&dir, &cfg, &plan, &inv, &text, opts.clone(), ready.clone());
                s.spawn(move || {
                    gb_cuda::rt::init_context().unwrap();
                    let w = slice_weights(whole_weights(dir, cfg, plan, inv), rank, cfg.vocab);
                    let mut m = NemotronModel::new_tp(w, cfg, opts, comm).unwrap();
                    m.set_graphs(false);
                    ready.wait(Duration::from_secs(600)).unwrap();
                    proj_after_window(&mut m, text)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a rank panicked")).collect()
    });
    let tp = &outs[0];
    let (mut bits, mut vals, mut neg_zero_single, mut neg_zero_tp) = (0usize, 0usize, 0usize, 0usize);
    let mut first = Vec::new();
    for (i, (x, y)) in single.iter().zip(tp).enumerate() {
        if x.to_bits() != y.to_bits() {
            bits += 1;
            if x != y && !(x.is_nan() && y.is_nan()) {
                vals += 1;
            }
            if first.len() < 8 {
                first.push((i, x.to_bits(), y.to_bits()));
            }
        }
        if x.to_bits() == 0x8000_0000 {
            neg_zero_single += 1;
        }
        if y.to_bits() == 0x8000_0000 {
            neg_zero_tp += 1;
        }
    }
    eprintln!(
        "TP DIAG in_proj output of 3 rows ({} f32): {bits} differ in bits, {vals} differ in value; -0.0 in the single-node row {neg_zero_single}, in the TP row {neg_zero_tp}",
        single.len()
    );
    for (i, a, b) in first {
        eprintln!("TP DIAG element {i} (row {}, column {}): single {a:08x}, tp {b:08x}", i / 18560, i % 18560);
    }
    assert_eq!(single.len(), tp.len());
}

/// The restricted draft head's table under TP: the rows `ids` of the trunk's `lm_head`, gathered through the group from each rank's half
/// (widen, allreduce, narrow), are the whole head's rows byte for byte, on both ranks, across the cut at 65,536, in runs and singly,
/// in more than one staging chunk.
#[test]
#[ignore]
fn nemotron_tp_head_rows_gather_is_the_whole_heads_rows() {
    let Some(dir) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set");
        return;
    };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let vocab = cfg.vocab;
    let ranks = env_ranks();
    // runs at both ends and across every cut of the group's row split (the pair's, 65,536, always among them), singles every 31st id:
    // more than 2,048 rows, so more than one staging chunk
    let mut ids: Vec<u32> = (0..300).chain(65_500..65_700).chain(vocab as u32 - 72..vocab as u32).collect();
    for r in 1..ranks {
        let cut = head_row_cut(vocab, r, ranks).0 as u32;
        ids.extend(cut - 100..cut + 100);
    }
    ids.extend((0..vocab as u32).step_by(31));
    ids.sort_unstable();
    ids.dedup();
    assert!(ids.len() > 4096, "{} ids", ids.len());
    let row = 4096 * 2;
    let st = Stream::new_nonblocking().unwrap();

    let whole = whole_weights(&dir, &cfg, &plan, &inv);
    let want = gather_head_rows(&SingleRank, &whole.lm_head, vocab, &ids, &st).unwrap();
    st.sync().unwrap();
    let mut want_bytes = vec![0u8; ids.len() * row];
    want.download_range(0, &mut want_bytes).unwrap();
    drop(whole);

    let comms = loopback_group(ranks);
    let ready = Arc::new(HostBarrier::new(ranks));
    let got: Vec<Vec<u8>> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                let (dir, cfg, plan, inv, ids, ready) = (&dir, &cfg, &plan, &inv, &ids, ready.clone());
                s.spawn(move || {
                    gb_cuda::rt::init_context().unwrap();
                    let w = slice_head_rows(whole_weights(dir, cfg, plan, inv), rank, ranks, cfg.vocab);
                    ready.wait(Duration::from_secs(600)).unwrap();
                    let st = Stream::new_nonblocking().unwrap();
                    let t = gather_head_rows(&*comm, &w.lm_head, cfg.vocab, ids, &st).unwrap();
                    st.sync().unwrap();
                    let mut bytes = vec![0u8; ids.len() * 8192];
                    t.download_range(0, &mut bytes).unwrap();
                    bytes
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a rank panicked")).collect()
    });
    for (rank, g) in got.iter().enumerate() {
        let bad = g.iter().zip(&want_bytes).position(|(x, y)| x != y);
        assert!(bad.is_none(), "rank {rank}: the gathered table differs from the whole head's rows at byte {}", bad.unwrap_or(0));
    }
    eprintln!(
        "TP HEAD ROWS OK: {} rows ({:.1} MB) gathered through the group are the whole head's, on all {ranks} ranks",
        ids.len(),
        want_bytes.len() as f64 / 1e6
    );
}

/// Wide prefill slices under TP: slices of 256, 128 and an odd 100 rows run the MoE layers through the large-row prefill path
/// (`PrefillMoe::forward_prefill_in_tp`, half the experts absent, one allreduce of `y` a layer) and must give, row for row, the bits of
/// the single-node engine's: the last row's candidates and logits, the S and conv pools and the KV rows of the whole prompt, then a
/// verify window on top. 8-layer prefix, both ranks, graphs off.
fn wide_lines(m: &mut NemotronModel, text: &[Vec<u32>; 2]) -> Vec<(String, u64)> {
    reset(m);
    let src: &[u32] = &text[0];
    let mut dg = Digest { lines: Vec::new(), print: false };
    // 300 tokens: a 256-row slice and a 44-row one; 128-row slices (128 + 128 + 44); an odd 100 (100 + 100 + 100)
    prefill_seg_w(m, &mut dg, "w256", 0, src, 0, 300, None, 256);
    prefill_seg_w(m, &mut dg, "w128", 1, src, 10, 300, Some(128), 128);
    prefill_seg_w(m, &mut dg, "w100", 2, src, 20, 300, None, 100);
    window_seg(m, &mut dg, "after", &[Part { slot: 0, first: 300, src, off: 0, rows: 5, keep: 3 }]);
    reset(m);
    dg.lines
}

#[test]
#[ignore]
fn nemotron_tp_wide_prefill_slices_are_the_single_node_engine() {
    let _counters = CountersAtEnd;
    let Some(dir) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set");
        return;
    };
    if std::env::var_os("REXL3_NEMO_ORACLE_DIR").is_none() || !gb_cuda::device_available() {
        eprintln!("not run: no oracle ids or no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let text = texts();
    let opts = ModelOptions { prefill_rows: 256, ..options(Some(8)) };

    let mut m = NemotronModel::load(&dir, &cfg, &plan, opts.clone()).unwrap_or_else(|e| panic!("single-node load: {e}"));
    m.set_graphs(false);
    let reference = wide_lines(&mut m, &text);
    drop(m);

    let ranks = env_ranks();
    let comms = loopback_group(ranks);
    let ready = Arc::new(HostBarrier::new(ranks));
    let runs: Vec<Vec<(String, u64)>> = std::thread::scope(|s| {
        let handles: Vec<_> = comms
            .into_iter()
            .enumerate()
            .map(|(rank, comm)| {
                let (dir, cfg, plan, text, opts, ready) = (&dir, &cfg, &plan, &text, opts.clone(), ready.clone());
                s.spawn(move || {
                    gb_cuda::rt::init_context().unwrap();
                    // a rank that fails to load must not leave the others waiting at the barrier
                    let loaded = catch_unwind(AssertUnwindSafe(|| {
                        NemotronModel::load_tp(dir, cfg, plan, opts, comm).unwrap_or_else(|e| panic!("rank {rank}: {e}"))
                    }));
                    let mut m = match loaded {
                        Ok(m) => m,
                        Err(p) => {
                            ready.break_now();
                            resume_unwind(p)
                        }
                    };
                    m.set_graphs(false);
                    ready.wait(Duration::from_secs(600)).unwrap();
                    wide_lines(&mut m, text)
                })
            })
            .collect();
        handles.into_iter().enumerate().map(|(r, h)| h.join().unwrap_or_else(|_| panic!("rank {r} panicked"))).collect()
    });
    let mut bad = Vec::new();
    for (rank, lines) in runs.iter().enumerate() {
        bad.extend(compare(&format!("rank {rank}"), &reference, lines));
    }
    for b in &bad {
        eprintln!("TP DIGEST MISMATCH {b}");
    }
    assert!(bad.is_empty(), "{} lines differ from the single-node engine's", bad.len());
    eprintln!(
        "TP WIDE PREFILL OK: {} lines (slices of 256, 128 and 100 rows, a window after), all {ranks} ranks",
        reference.len()
    );
}

// ------------------------------------------------------------------------------------------------------------------ the MTP head

// The draft head under tensor parallelism (`docs/NEMOTRON_TP.md` section 9.2): its routed experts split between the ranks (the other
// half `Proj::Absent`, `forward_in_tp` with one allreduce of `y`) and its restricted `lm_head` scored by half of the rows on each rank
// (the `-0.0` scatter and one allreduce of the logits). The head follows the whole trunk in production; here it follows an 8-layer
// prefix (`REXL3_NEMO_HEAD_ON_PREFIX=1`, a test hook) so that two shards, the head's bf16 experts and a single-node reference fit in
// about 15 GiB and not in 95. What it drafts says nothing about acceptance; every bit of it must be the single-node engine's.

use gb_model::families::nemotron_h_omni::mtp::DraftAsk;
use gb_sampling::Sampling;

fn head_options(layers: Option<usize>) -> ModelOptions {
    ModelOptions { mtp: true, ..options(layers) }
}

/// The collectives one draft call is expected to issue on a rank of a group, by `REXL3_TP_HEAD` mode: the head's MoE reduce once per
/// head block (the absorb and each chain step) when its experts are split, and one logits combine per pick when the lm_head is split
/// (the restricted table) or is the rank's half of the trunk's head (the full-vocabulary head, always).
fn expected_collectives(experts: bool, lm_head: bool, restricted: bool, depth: usize) -> usize {
    if depth == 0 {
        return 0;
    }
    let moe = if experts { depth } else { 0 };
    let lm = if !restricted || lm_head { depth } else { 0 };
    moe + lm
}

/// The scripted head run: three streams (two sampled, one greedy) through prompts, then rounds of draft call, debug chain, one window of
/// all three with partial keeps, commits. Digests every draft id and logit bits, the head's top-64 candidates at every chain depth, the
/// windows' candidates and the head's KV rows. Returns the lines and, per draft call, `(depth, collectives issued)` when `counting`.
fn head_script(
    m: &mut NemotronModel,
    text: &[Vec<u32>; 2],
    restricted: bool,
    counting: Option<&CountingComm>,
) -> (Vec<(String, u64)>, Vec<(usize, usize)>) {
    reset(m);
    m.set_mtp_restricted(restricted);
    let mut dg = Digest { lines: Vec::new(), print: false };
    let mut calls = Vec::new();
    let prompts: [Vec<u32>; 3] = [text[0][..70].to_vec(), text[1][300..340].to_vec(), text[0][500..566].to_vec()];
    let samplers = [
        Some(Sampling { seed: 5, temperature: 0.8, top_k: 20, top_p: 0.95, min_p: 0.0 }),
        None,
        Some(Sampling { seed: 6, temperature: 1.0, top_k: 8, top_p: 1.0, min_p: 0.0 }),
    ];
    let mut toks: Vec<Vec<u32>> = Vec::new();
    for (slot, p) in prompts.iter().enumerate() {
        m.begin_slot(slot).unwrap();
        let mut at = 0;
        let mut first = 0u32;
        while at < p.len() {
            let end = (at + 64 - at % 64).min(p.len());
            let b = RowBatch::from_requests(&[RowRequest { slot, tokens: p[at..end].to_vec(), first_position: at as u64 }]);
            let want = (end == p.len()).then_some(b.rows - 1);
            if let Some(c) = m.prefill_chunk(&b, want).unwrap() {
                first = c.ids[0] as u32;
                dg.emit(format!("s{slot}.prompt.cands"), hash_cands(&c));
            }
            at = end;
        }
        let mut t = p.clone();
        t.push(first);
        toks.push(t);
    }
    let depths = [3usize, 3, 2, 4, 1, 3, 3, 0, 3];
    for (round, &depth) in depths.iter().enumerate() {
        // stream 1 asks for fewer drafts, and in round 5 only absorbs (a depth-0 ask)
        let ask_depth = |s: usize| match s {
            1 if round == 5 => 0,
            1 => depth.min(2),
            _ => depth,
        };
        let asks: Vec<DraftAsk> = (0..3).map(|s| DraftAsk { slot: s, tokens: &toks[s], depth: ask_depth(s) }).collect();
        if let Some(c) = counting {
            c.take();
        }
        let got = m.mtp_draft(&asks, &samplers).unwrap_or_else(|e| panic!("round {round}: draft: {e}"));
        let max_depth = asks.iter().map(|a| a.depth).max().unwrap();
        if let Some(c) = counting {
            calls.push((max_depth, c.take().len()));
        }
        let mut drafts: Vec<Vec<u32>> = Vec::new();
        for (s, d) in got.iter().enumerate() {
            let mut h = Dg::new();
            match d {
                Some(d) => {
                    h.bytes(&d.ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>());
                    h.f32s(&d.vals);
                    drafts.push(d.ids.clone());
                }
                None => {
                    h.bytes(b"none");
                    drafts.push(Vec::new());
                }
            }
            dg.emit(format!("r{round}.s{s}.drafts"), h);
        }
        // the head's own candidates at each depth for stream 0, the chain fed with the first candidate (the debug path: it takes the
        // device-wide synchronisations the production call does not)
        if round % 2 == 0 {
            let mut f = |_k: usize, ids: &[u32], _v: &[f32]| ids[0];
            if let Some(lists) = m.mtp_debug_chain(0, &toks[0], depth.max(1), 64, restricted, false, &mut f).unwrap() {
                let mut h = Dg::new();
                for (ids, vals) in &lists {
                    h.bytes(&ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>());
                    h.f32s(vals);
                }
                dg.emit(format!("r{round}.chain0"), h);
            }
        }
        // one window of all three streams: [last, drafts..], keep a prefix of it
        let reqs: Vec<RowRequest> = (0..3)
            .map(|s| {
                let mut w = vec![*toks[s].last().unwrap()];
                w.extend_from_slice(&drafts[s]);
                RowRequest { slot: s, tokens: w, first_position: (toks[s].len() - 1) as u64 }
            })
            .collect();
        let b = RowBatch::from_requests(&reqs);
        let c = m.window_candidates(&b, 8).unwrap_or_else(|e| panic!("round {round}: window: {e}"));
        dg.emit(format!("r{round}.window.cands"), hash_cands(&c));
        let mut row = 0usize;
        for (s, r) in reqs.iter().enumerate() {
            let rows = r.tokens.len();
            let keep = 1 + (round + s) % rows;
            m.commit(s, keep);
            let next = c.ids[(row + keep - 1) * c.count] as u32;
            toks[s].extend_from_slice(&r.tokens[1..keep]);
            toks[s].push(next);
            row += rows;
        }
        m.flush().unwrap();
    }
    for s in 0..3 {
        let built = m.mtp_book(s).map_or(0, |b| b.first).saturating_sub(1);
        if built > 0 {
            let (k, v) = m.mtp_kv_rows(s, 0, built).unwrap();
            let mut h = Dg::new();
            h.u16s(&k);
            h.u16s(&v);
            dg.emit(format!("s{s}.head_kv"), h);
        }
    }
    reset(m);
    (dg.lines, calls)
}

/// `REXL3_TP_HEAD` modes the loopback test runs, as the variable's value and the `(experts, lm_head)` it splits.
const HEAD_MODES: [(&str, bool, bool); 4] =
    [("split", true, true), ("replicate", false, false), ("experts", true, false), ("lm_head", false, true)];

#[test]
#[ignore]
fn nemotron_tp_head_loopback_is_the_single_node_head() {
    let Some(dir) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set");
        return;
    };
    if std::env::var_os("REXL3_NEMO_ORACLE_DIR").is_none() || !gb_cuda::device_available() {
        eprintln!("not run: no oracle ids or no CUDA device");
        return;
    }
    let ranks = env_ranks();
    let have = mem_available();
    if have < (24u64 << 30) {
        eprintln!("not run: the shards of the head test need about 16 GiB and only {:.1} GiB is available", have as f64 / GIB);
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    // `REXL3_NEMO_TP_HEAD_FULL=1`: the head over the WHOLE trunk (88 layers, the production configuration) instead of the 8-layer prefix;
    // about 64 GiB for the single-node reference and the group's R x 25 GiB after it (the reference is dropped first)
    let full = std::env::var("REXL3_NEMO_TP_HEAD_FULL").is_ok_and(|v| v == "1");
    let layers = if full { None } else { Some(8) };
    if !full {
        std::env::set_var("REXL3_NEMO_HEAD_ON_PREFIX", "1");
    }
    let modes: Vec<String> = match std::env::var("REXL3_NEMO_TP_HEAD_MODES") {
        Ok(v) => v.split(',').map(|s| s.trim().to_string()).collect(),
        Err(_) => HEAD_MODES.iter().map(|m| m.0.to_string()).collect(),
    };
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let text = texts();
    let opts = head_options(layers);
    // the group's weights by plan before anything loads (the trunk's 8 layers and the globals, and the head, rank by rank, in the
    // worst of the modes asked for: a mode that keeps the head's experts whole prices the whole head on every rank), a GiB of pools a
    // rank, and 16 GiB left for the host; refuse when they do not fit
    {
        let head_whole = modes.iter().any(|m| HEAD_MODES.iter().any(|h| h.0 == m.as_str() && !h.1));
        let per_rank: Vec<u64> = (0..ranks)
            .map(|r| {
                let mut shard = serving_shard_plan(TpTopology { rank: r, ranks }, &cfg);
                if head_whole {
                    shard = shard.replicating_head();
                }
                let n = layers.unwrap_or(usize::MAX);
                let trunk = shard.rank_bytes(&plan, cfg.experts, |u| match u {
                    UnitId::Trunk(i) => i < n,
                    UnitId::Embed | UnitId::NormF | UnitId::LmHead => true,
                    UnitId::Mtp(_) => false,
                });
                let head = shard.rank_bytes(&plan, cfg.experts, |u| matches!(u, UnitId::Mtp(_)));
                trunk + head + (1 << 30)
            })
            .collect();
        let need = per_rank.iter().sum::<u64>() + (16 << 30);
        eprintln!(
            "TP head test: {ranks} shards, planned weights + pools a rank (GiB) {:?}; need about {:.1} GiB, the host has {:.1} GiB",
            per_rank.iter().map(|b| (*b as f64 / GIB * 100.0).round() / 100.0).collect::<Vec<_>>(),
            need as f64 / GIB,
            have as f64 / GIB
        );
        if (ranks > 2 || full) && have < need {
            eprintln!(
                "not run: the {ranks} shards need about {:.1} GiB and only {:.1} GiB is available",
                need as f64 / GIB,
                have as f64 / GIB
            );
            return;
        }
    }

    // the single-node reference: the restricted head and the full head, both
    std::env::remove_var("REXL3_TP_HEAD");
    let mut m = NemotronModel::load(&dir, &cfg, &plan, opts.clone()).unwrap_or_else(|e| panic!("single-node load: {e}"));
    assert!(m.mtp_loaded() && m.mtp_sub_head().is_some(), "the reference carries the head and its restricted table");
    m.set_graphs(false);
    let single_head_bytes = m.mtp_allocations().iter().find(|(k, _)| *k == "mtp_weights").map(|(_, b)| *b).unwrap();
    let mut reference = Vec::new();
    for restricted in [true, false] {
        let (lines, _) = head_script(&mut m, &text, restricted, None);
        let tag = if restricted { "restricted" } else { "full" };
        reference.extend(lines.into_iter().map(|(n, h)| (format!("{tag}.{n}"), h)));
    }
    drop(m);
    eprintln!(
        "reference: single-node {} with the head, {} lines; head weights {:.2} GiB",
        if full { "88 layers" } else { "8-layer prefix" },
        reference.len(),
        single_head_bytes as f64 / GIB
    );

    for mode in &modes {
        let (_, experts, lm_head) =
            *HEAD_MODES.iter().find(|m| m.0 == mode.as_str()).unwrap_or_else(|| panic!("unknown head mode `{mode}`"));
        std::env::set_var("REXL3_TP_HEAD", mode);
        let group = loopback_group(ranks);
        // every rank's draft calls are counted (rank 0's list is the one printed): each rank issues what the mode says
        let countings: Vec<Arc<CountingComm>> = group.iter().map(|c| Arc::new(CountingComm::new(c.clone()))).collect();
        let ready = Arc::new(HostBarrier::new(ranks));
        type HeadRun = (Vec<(String, u64)>, Vec<(usize, usize, bool)>, u64);
        let runs: Vec<HeadRun> = std::thread::scope(|s| {
            let handles: Vec<_> = countings
                .iter()
                .enumerate()
                .map(|(rank, counting)| {
                    let comm: Arc<dyn TpComm> = counting.clone();
                    let (dir, cfg, plan, text, opts, ready, counting) =
                        (&dir, &cfg, &plan, &text, opts.clone(), ready.clone(), counting.clone());
                    s.spawn(move || {
                        gb_cuda::rt::init_context().unwrap();
                        let loaded = catch_unwind(AssertUnwindSafe(|| {
                            NemotronModel::load_tp(dir, cfg, plan, opts, comm)
                                .unwrap_or_else(|e| panic!("rank {rank}: load: {e}"))
                        }));
                        let mut m = match loaded {
                            Ok(m) => m,
                            Err(p) => {
                                ready.break_now();
                                resume_unwind(p)
                            }
                        };
                        m.set_graphs(false);
                        let head_bytes = m.mtp_allocations().iter().find(|(k, _)| *k == "mtp_weights").map(|(_, b)| *b).unwrap();
                        ready.wait(Duration::from_secs(900)).expect("the other rank did not finish loading");
                        let mut lines = Vec::new();
                        let mut issued = Vec::new();
                        for restricted in [true, false] {
                            let (l, calls) = head_script(&mut m, text, restricted, Some(&*counting));
                            let tag = if restricted { "restricted" } else { "full" };
                            lines.extend(l.into_iter().map(|(n, h)| (format!("{tag}.{n}"), h)));
                            issued.extend(calls.into_iter().map(|(d, n)| (d, n, restricted)));
                        }
                        (lines, issued, head_bytes)
                    })
                })
                .collect();
            handles.into_iter().enumerate().map(|(r, h)| h.join().unwrap_or_else(|_| panic!("rank {r} panicked"))).collect()
        });
        let mut bad = Vec::new();
        for (rank, (lines, _, _)) in runs.iter().enumerate() {
            bad.extend(compare(&format!("head mode {mode} R={ranks} rank {rank}"), &reference, lines));
        }
        // the collectives of every draft call are the ones the mode says, on every rank, and the experts' bytes shrink (or not)
        let mut total = 0usize;
        for (rank, run) in runs.iter().enumerate() {
            for &(depth, got, restricted) in &run.1 {
                let want = expected_collectives(experts, lm_head, restricted, depth);
                if got != want {
                    bad.push(format!(
                        "head mode {mode} R={ranks} rank {rank}: a draft call of depth {depth} (restricted {restricted}) issued {got} collectives, expected {want}"
                    ));
                }
                if rank == 0 {
                    total += got;
                }
            }
        }
        let planned: Vec<u64> = (0..ranks)
            .map(|rank| {
                let shard = gb_model::families::nemotron_h_omni::tp_arm::serving_shard_plan(TpTopology { rank, ranks }, &cfg);
                shard.rank_bytes(&plan, cfg.experts, |u| matches!(u, UnitId::Mtp(_)))
            })
            .collect();
        for rank in 0..ranks {
            if runs[rank].2 != planned[rank] {
                bad.push(format!(
                    "head mode {mode} R={ranks} rank {rank}: holds {} head bytes, the shard plan says {}",
                    runs[rank].2, planned[rank]
                ));
            }
            if experts && runs[rank].2 >= single_head_bytes {
                bad.push(format!(
                    "head mode {mode} R={ranks} rank {rank}: the head did not shrink ({} of {single_head_bytes})",
                    runs[rank].2
                ));
            }
        }
        eprintln!(
            "TP HEAD mode {mode} R={ranks}: {} lines per rank; head weights per rank {:?} GiB (single {:.2}); rank 0 issued {total} collectives over {} draft calls: {:?}",
            reference.len(),
            runs.iter().map(|r| (r.2 as f64 / GIB * 100.0).round() / 100.0).collect::<Vec<_>>(),
            single_head_bytes as f64 / GIB,
            runs[0].1.len(),
            runs[0].1.iter().take(9).map(|(d, n, _)| format!("d{d}:{n}")).collect::<Vec<_>>()
        );
        for b in &bad {
            eprintln!("TP HEAD MISMATCH {b}");
        }
        assert!(bad.is_empty(), "{} head lines or counts differ ({mode}, {ranks} ranks)", bad.len());
    }
    std::env::remove_var("REXL3_TP_HEAD");
    eprintln!("TP HEAD OK: R={ranks}, modes {modes:?}: every digest line (drafts, logit bits, candidates, windows, head KV) identical to the single node, all ranks");
}

/// The repository root: `REXL3_REPO_ROOT` when set (a prebuilt test binary run from an installed bin directory that carries the
/// tree's `testdata/`, e.g. on a node without this checkout), else the checkout this binary was built from.
fn repo_root() -> std::path::PathBuf {
    std::env::var_os("REXL3_REPO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}
