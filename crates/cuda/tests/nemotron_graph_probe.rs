//! G0: can the Nemotron GPU paths be captured into a CUDA graph and replayed? For each path the probe runs it eagerly, captures the same
//! launches into a `gb_cuda::graph::Graph` on one non-blocking stream, replays the graph twice from the same starting state and compares
//! the bits with the eager run, then changes the inputs in place on the same device buffers and checks the replay follows them (what the
//! engine's decode loop does between steps). The table is printed with `--nocapture` and written up in `docs/NEMOTRON_GRAPH_PROBE.md`.
//!
//! Everything a launch needs (buffers, scratch, index arrays) is allocated and filled before the capture: an allocation, a pageable
//! copy or a host wait inside a capture fails it. `a_sync_inside_a_capture_is_refused` is the control: the harness does report a failure.
//! GPU tests skip when there is no device.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::exl3_synth::synth_group;
use common::Rng;
use core::ffi::c_void;
use gb_cuda::attention::BlockTable;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::gdn::BatchBuf;
use gb_cuda::graph::{CaptureError, Graph};
use gb_cuda::nemotron_attn::{self as na, NemoKv, RowIndex, Scratch};
use gb_cuda::nemotron_attn_prefill as nf;
use gb_cuda::nemotron_dense as nd;
use gb_cuda::nemotron_glue as ng;
use gb_cuda::nemotron_moe::{MoeScratch, NemotronMoe, Proj, Weights as MoeWeights, EXPERTS, EXPERT_INTER, HIDDEN, LATENT};
use gb_cuda::nemotron_ssm as ns;
use gb_cuda::{device_available, CudaError, DevBuf, Stream};
use gb_exl3::Codebook;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

mod ev {
    use core::ffi::c_void;
    extern "C" {
        pub fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
        pub fn cudaEventDestroy(e: *mut c_void) -> i32;
    }
}

/// Microseconds per call of `f`, `iters` calls queued back to back between two events (after `warm` untimed calls).
fn batch_us(st: &Stream, warm: usize, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..warm {
        f();
    }
    st.sync().unwrap();
    let (mut a, mut b) = (core::ptr::null_mut::<c_void>(), core::ptr::null_mut::<c_void>());
    let mut ms = 0f32;
    unsafe {
        assert_eq!(ev::cudaEventCreateWithFlags(&mut a, 0), 0);
        assert_eq!(ev::cudaEventCreateWithFlags(&mut b, 0), 0);
        assert_eq!(ev::cudaEventRecord(a, st.raw()), 0);
    }
    for _ in 0..iters {
        f();
    }
    unsafe {
        assert_eq!(ev::cudaEventRecord(b, st.raw()), 0);
    }
    st.sync().unwrap();
    unsafe {
        assert_eq!(ev::cudaEventElapsedTime(&mut ms, a, b), 0);
        ev::cudaEventDestroy(a);
        ev::cudaEventDestroy(b);
    }
    ms as f64 * 1000.0 / iters as f64
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The harness

/// One probed path: its buffers, and what the harness needs to run it eagerly, capture it and compare.
trait Case {
    /// The launches. Everything they touch is already allocated; nothing here may synchronise, allocate or copy from pageable memory.
    fn launch(&mut self, st: &Stream) -> Result<(), CudaError>;
    /// Outside a capture: puts the outputs and any state the launches change back to the start (the current initial state).
    fn reset(&mut self);
    /// Outside a capture: new inputs (and a new initial state) written in place into the same device buffers.
    fn perturb(&mut self);
    /// The eager run on the perturbed inputs, when it needs more than `launch` (a host-side value that is stale after `perturb`).
    fn eager_after_perturb(&mut self, st: &Stream) -> Result<(), CudaError> {
        self.launch(st)
    }
    /// The bits of everything the path produced.
    fn read(&mut self) -> Vec<u32>;
}

struct Report {
    name: String,
    /// `Ok` or the capture's refusal.
    capture: Result<(), String>,
    replay_1: bool,
    replay_2: bool,
    follows_inputs: bool,
    eager_us: f64,
    replay_us: f64,
}

impl Report {
    fn passed(&self) -> bool {
        self.capture.is_ok() && self.replay_1 && self.replay_2 && self.follows_inputs
    }
    fn row(&self) -> String {
        let yn = |b: bool| if b { "yes" } else { "NO" };
        match &self.capture {
            Ok(()) => format!(
                "| {} | ok | {} | {} | {} | {:.1} | {:.1} |",
                self.name,
                yn(self.replay_1),
                yn(self.replay_2),
                yn(self.follows_inputs),
                self.eager_us,
                self.replay_us
            ),
            Err(e) => format!("| {} | REFUSED: {e} | - | - | - | - | - |", self.name),
        }
    }
}

fn f32_bits(b: &DevBuf, n: usize) -> Vec<u32> {
    let mut v = vec![0f32; n];
    b.download(&mut v).unwrap();
    v.iter().map(|x| x.to_bits()).collect()
}

fn u16_bits(b: &DevBuf, n: usize) -> Vec<u32> {
    let mut v = vec![0u16; n];
    b.download(&mut v).unwrap();
    v.iter().map(|&x| x as u32).collect()
}

/// `perturb`, `reset` and the case constructors write the buffers on the legacy stream (`DevBuf::upload`, `from_slice`,
/// `zeroed`), which a non-blocking probe stream does not wait for (`st.sync()` does not either). Every `reset` ends with
/// this, and every `perturb` is followed by a `reset`, so the next launch reads what was written.
fn settle() {
    gb_cuda::rt::device_sync().unwrap();
}

fn run_case(name: &str, st: &Stream, c: &mut dyn Case) -> Report {
    let mut r = Report {
        name: name.to_string(),
        capture: Ok(()),
        replay_1: false,
        replay_2: false,
        follows_inputs: false,
        eager_us: 0.0,
        replay_us: 0.0,
    };
    // eager reference (twice: the second run must equal the first, or "replay equals eager" would mean nothing)
    st.sync().unwrap();
    c.reset();
    c.launch(st).unwrap_or_else(|e| panic!("{name}: the eager run failed: {e}"));
    st.sync().unwrap();
    let e1 = c.read();
    c.reset();
    c.launch(st).unwrap();
    st.sync().unwrap();
    assert!(c.read() == e1, "{name}: two eager runs from the same state differ");
    // capture
    c.reset();
    let g = match Graph::capture(st, || c.launch(st)) {
        Ok(g) => g,
        Err(CaptureError::Refused(e)) => {
            r.capture = Err(format!("runtime refused: {e}"));
            return r;
        }
        Err(CaptureError::Closure(e)) => {
            r.capture = Err(format!("a launch failed while capturing: {e}"));
            return r;
        }
    };
    // two replays, each from the start state
    for k in 0..2 {
        c.reset();
        g.launch(st).unwrap();
        st.sync().unwrap();
        let same = c.read() == e1;
        if k == 0 {
            r.replay_1 = same;
        } else {
            r.replay_2 = same;
        }
    }
    // new inputs in place: the replay must produce what a fresh eager run on them produces (and that must differ from before)
    c.perturb();
    c.reset();
    c.eager_after_perturb(st).unwrap();
    st.sync().unwrap();
    let e2 = c.read();
    c.reset();
    g.launch(st).unwrap();
    st.sync().unwrap();
    r.follows_inputs = c.read() == e2 && e2 != e1;
    // launch cost: 20 eager calls against 20 replays
    r.eager_us = batch_us(st, 3, 20, || c.launch(st).unwrap());
    r.replay_us = batch_us(st, 3, 20, || g.launch(st).unwrap());
    r
}

fn print_table(title: &str, reports: &[Report]) {
    eprintln!("\n{title}");
    eprintln!("| path | capture | replay 1 == eager | replay 2 == eager | replay follows new inputs | eager us | replay us |");
    eprintln!("|---|---|---|---|---|---|---|");
    for r in reports {
        eprintln!("{}", r.row());
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// (a) the latent MoE block on one shared scratch

fn bf16_bits(r: &mut Rng, n: usize) -> Vec<u16> {
    (0..n).map(|_| (r.uniform(-0.05, 0.05).to_bits() >> 16) as u16).collect()
}

/// A synthetic layer as `tests/nemotron_moe.rs` builds it: random gate and bias, EXL3 dense projections and the listed experts
/// (EXL3, except `bf16`), every other expert absent.
fn synthetic_moe(seed: u64, experts: &[usize], bf16: Option<usize>) -> MoeWeights {
    const SHARED_INTER: usize = 5376;
    let mut r = Rng(seed);
    let gate = bf16_bits(&mut r, EXPERTS * HIDDEN);
    let bias: Vec<f32> = r.vec(EXPERTS, -0.1, 0.1);
    let fc1 = synth_group(HIDDEN, LATENT, 4, Codebook::Mul1, seed + 1);
    let fc2 = synth_group(LATENT, HIDDEN, 4, Codebook::Mul1, seed + 2);
    let su = synth_group(HIDDEN, SHARED_INTER, 4, Codebook::Mul1, seed + 3);
    let sd = synth_group(SHARED_INTER, HIDDEN, 4, Codebook::Mul1, seed + 4);
    let mut up: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    let mut down: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    for &e in experts {
        if Some(e) == bf16 {
            up[e] = Proj::bf16(LATENT, EXPERT_INTER, &bf16_bits(&mut r, EXPERT_INTER * LATENT)).unwrap();
            down[e] = Proj::bf16(EXPERT_INTER, LATENT, &bf16_bits(&mut r, LATENT * EXPERT_INTER)).unwrap();
        } else {
            let ug = synth_group(LATENT, EXPERT_INTER, 4, Codebook::Mul1, seed + 1000 + 2 * e as u64);
            let dg = synth_group(EXPERT_INTER, LATENT, 4, Codebook::Mul1, seed + 1001 + 2 * e as u64);
            up[e] = Proj::exl3_from_group(&ug).unwrap();
            down[e] = Proj::exl3_from_group(&dg).unwrap();
        }
    }
    MoeWeights {
        gate: DevBuf::from_slice(&gate).unwrap(),
        gate_dtype: gb_cuda::nemotron_moe::GateDtype::Bf16,
        router_bias: DevBuf::from_slice(&bias).unwrap(),
        scale: 5.0,
        fc1: Proj::exl3_from_group(&fc1).unwrap(),
        fc2: Proj::exl3_from_group(&fc2).unwrap(),
        shared_up: Proj::exl3_from_group(&su).unwrap(),
        shared_down: Proj::exl3_from_group(&sd).unwrap(),
        up,
        down,
    }
}

/// `layers` blocks (their own tables, one shared scratch) chained: block i reads the output of block i - 1.
struct MoeCase {
    layers: Vec<NemotronMoe>,
    scratch: MoeScratch,
    h: DevBuf,
    mids: Vec<DevBuf>,
    rows: usize,
    rng: Rng,
}

impl MoeCase {
    fn new(rows: usize, layers: usize, scratch_rows: usize) -> Self {
        let experts: Vec<usize> = (0..EXPERTS).step_by(37).collect();
        let layers_v: Vec<NemotronMoe> =
            (0..layers).map(|i| NemotronMoe::tables(synthetic_moe(61 + i as u64, &experts, Some(experts[3]))).unwrap()).collect();
        let mut rng = Rng(77);
        let h = DevBuf::from_slice(&rng.vec(rows * HIDDEN, -1.0, 1.0)).unwrap();
        let mids = (0..layers).map(|_| DevBuf::zeroed(rows * HIDDEN * 4).unwrap()).collect();
        Self { layers: layers_v, scratch: MoeScratch::new(scratch_rows).unwrap(), h, mids, rows, rng }
    }
}

impl Case for MoeCase {
    fn launch(&mut self, st: &Stream) -> Result<(), CudaError> {
        for i in 0..self.layers.len() {
            let (done, rest) = self.mids.split_at_mut(i);
            let input = if i == 0 { &self.h } else { &done[i - 1] };
            self.layers[i].forward_in(&mut self.scratch, input, self.rows, &mut rest[0], st)?;
        }
        Ok(())
    }
    fn reset(&mut self) {
        for m in &mut self.mids {
            m.upload(&vec![0f32; self.rows * HIDDEN]).unwrap();
        }
        settle();
    }
    fn perturb(&mut self) {
        self.h.upload(&self.rng.vec(self.rows * HIDDEN, -1.0, 1.0)).unwrap();
    }
    fn read(&mut self) -> Vec<u32> {
        self.mids.iter().flat_map(|m| f32_bits(m, self.rows * HIDDEN)).collect()
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// (b) one EXL3 linear, decode path

struct ExlCase {
    w: Exl3Weights,
    x: DevBuf,
    y: DevBuf,
    ws: DevBuf,
    m: usize,
    rng: Rng,
}

impl ExlCase {
    fn new(m: usize) -> Self {
        let g = synth_group(4096, 1024, 4, Codebook::Mul1, 9);
        let w = Exl3Weights::from_loaded(&g).unwrap();
        let mut rng = Rng(3);
        let x = DevBuf::from_slice(&rng.vec(m * 4096, -1.0, 1.0)).unwrap();
        let y = DevBuf::zeroed(m * 1024 * 4).unwrap();
        let ws = DevBuf::zeroed(w.workspace_bytes(m)).unwrap();
        Self { w, x, y, ws, m, rng }
    }
}

impl Case for ExlCase {
    fn launch(&mut self, st: &Stream) -> Result<(), CudaError> {
        self.w.forward(&self.x, &mut self.y, self.m, Dtype::F32, Dtype::F32, &mut self.ws, st)
    }
    fn reset(&mut self) {
        self.y.upload(&vec![0f32; self.m * 1024]).unwrap();
        settle();
    }
    fn perturb(&mut self) {
        self.x.upload(&self.rng.vec(self.m * 4096, -1.0, 1.0)).unwrap();
    }
    fn read(&mut self) -> Vec<u32> {
        f32_bits(&self.y, self.m * 1024)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// (c) the glue kernels, all in one graph

struct GlueCase {
    rows: usize,
    vocab: usize,
    table: DevBuf,
    ids: DevBuf,
    embed_out: DevBuf,
    norm_x: DevBuf,
    norm_w: DevBuf,
    norm_out: DevBuf,
    inproj: DevBuf,
    z: DevBuf,
    xbc: DevBuf,
    dt: DevBuf,
    logits: DevBuf,
    ws: DevBuf,
    top_ids: DevBuf,
    top_vals: DevBuf,
    src16: DevBuf,
    widened_bf: DevBuf,
    widened_h: DevBuf,
    feats: DevBuf,
    spliced: DevBuf,
    rng: Rng,
}

const GLUE_VOCAB: usize = 131072;
const GLUE_COUNT: usize = 40;
const GLUE_TABLE_ROWS: usize = 256;
const GLUE_WIDEN: usize = 4099;

impl GlueCase {
    fn new(rows: usize) -> Self {
        let mut rng = Rng(5);
        let u16s = |rng: &mut Rng, n: usize| -> Vec<u16> { (0..n).map(|_| (rng.next_u64() >> 40) as u16).collect() };
        let table = DevBuf::from_slice(&u16s(&mut rng, GLUE_TABLE_ROWS * ng::HIDDEN)).unwrap();
        let ids: Vec<i32> = (0..rows).map(|_| (rng.next_u64() % GLUE_TABLE_ROWS as u64) as i32).collect();
        let norm_w = DevBuf::from_slice(&bf16_bits(&mut rng, ng::HIDDEN)).unwrap();
        let n_ws = ng::topk_workspace_bytes(rows, GLUE_VOCAB, GLUE_COUNT);
        let fz = |n: usize| DevBuf::zeroed(n * 4).unwrap();
        Self {
            rows,
            vocab: GLUE_VOCAB,
            table,
            ids: DevBuf::from_slice(&ids).unwrap(),
            embed_out: fz(rows * ng::HIDDEN),
            norm_x: DevBuf::from_slice(&rng.vec(rows * ng::HIDDEN, -2.0, 2.0)).unwrap(),
            norm_w,
            norm_out: fz(rows * ng::HIDDEN),
            inproj: DevBuf::from_slice(&rng.vec(rows * ng::INPROJ_COLS, -3.0, 3.0)).unwrap(),
            z: fz(rows * ng::Z_COLS),
            xbc: fz(rows * ng::XBC_COLS),
            dt: fz(rows * ng::DT_COLS),
            logits: DevBuf::from_slice(&rng.vec(rows * GLUE_VOCAB, -8.0, 8.0)).unwrap(),
            ws: DevBuf::zeroed(n_ws).unwrap(),
            top_ids: fz(rows * GLUE_COUNT),
            top_vals: fz(rows * GLUE_COUNT),
            src16: DevBuf::from_slice(&u16s(&mut rng, GLUE_WIDEN)).unwrap(),
            widened_bf: fz(GLUE_WIDEN),
            widened_h: fz(GLUE_WIDEN),
            feats: DevBuf::from_slice(&u16s(&mut rng, 4 * ng::HIDDEN)).unwrap(),
            spliced: fz(rows.max(4) * ng::HIDDEN),
            rng,
        }
    }
    fn outputs(&self) -> Vec<(&DevBuf, usize)> {
        let r = self.rows;
        vec![
            (&self.embed_out, r * ng::HIDDEN),
            (&self.norm_out, r * ng::HIDDEN),
            (&self.z, r * ng::Z_COLS),
            (&self.xbc, r * ng::XBC_COLS),
            (&self.dt, r * ng::DT_COLS),
            (&self.top_ids, r * GLUE_COUNT),
            (&self.top_vals, r * GLUE_COUNT),
            (&self.widened_bf, GLUE_WIDEN),
            (&self.widened_h, GLUE_WIDEN),
            (&self.spliced, r.max(4) * ng::HIDDEN),
        ]
    }
}

impl Case for GlueCase {
    fn launch(&mut self, st: &Stream) -> Result<(), CudaError> {
        let r = self.rows;
        ng::embed_gather_f32(&self.table, &self.ids, r, &mut self.embed_out, st)?;
        ng::rmsnorm_bf16w_f32(&self.norm_x, &self.norm_w, r, ng::HIDDEN, ng::HIDDEN, 1e-5, &mut self.norm_out, ng::HIDDEN, st)?;
        ng::split_inproj(&self.inproj, r, &mut self.z, &mut self.xbc, &mut self.dt, st)?;
        ng::topk_f32(&self.logits, r, self.vocab, GLUE_COUNT, &mut self.ws, &mut self.top_ids, &mut self.top_vals, st)?;
        ng::widen_bf16_f32(&self.src16, GLUE_WIDEN, &mut self.widened_bf, st)?;
        ng::widen_f16_f32(&self.src16, GLUE_WIDEN, &mut self.widened_h, st)?;
        ng::splice_rows(&self.feats, 1, &mut self.spliced, 0, 3, st)
    }
    fn reset(&mut self) {
        let r = self.rows;
        for (b, n) in [
            (&mut self.embed_out, r * ng::HIDDEN),
            (&mut self.norm_out, r * ng::HIDDEN),
            (&mut self.z, r * ng::Z_COLS),
            (&mut self.xbc, r * ng::XBC_COLS),
            (&mut self.dt, r * ng::DT_COLS),
            (&mut self.top_ids, r * GLUE_COUNT),
            (&mut self.top_vals, r * GLUE_COUNT),
            (&mut self.widened_bf, GLUE_WIDEN),
            (&mut self.widened_h, GLUE_WIDEN),
            (&mut self.spliced, r.max(4) * ng::HIDDEN),
        ] {
            b.upload(&vec![0u32; n]).unwrap();
        }
        settle();
    }
    fn perturb(&mut self) {
        let r = self.rows;
        let ids: Vec<i32> = (0..r).map(|_| (self.rng.next_u64() % GLUE_TABLE_ROWS as u64) as i32).collect();
        self.ids.upload(&ids).unwrap();
        self.norm_x.upload(&self.rng.vec(r * ng::HIDDEN, -2.0, 2.0)).unwrap();
        self.inproj.upload(&self.rng.vec(r * ng::INPROJ_COLS, -3.0, 3.0)).unwrap();
        self.logits.upload(&self.rng.vec(r * GLUE_VOCAB, -8.0, 8.0)).unwrap();
        let s: Vec<u16> = (0..GLUE_WIDEN).map(|_| (self.rng.next_u64() >> 40) as u16).collect();
        self.src16.upload(&s).unwrap();
    }
    fn read(&mut self) -> Vec<u32> {
        let mut out = Vec::new();
        for (b, n) in self.outputs() {
            out.extend(f32_bits(b, n));
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// (d) the dense bf16 GEMV

struct DenseCase {
    w: DevBuf,
    x: DevBuf,
    y: DevBuf,
    m: usize,
    n: usize,
    k: usize,
    rng: Rng,
}

impl DenseCase {
    fn new(m: usize) -> Self {
        let (n, k) = (2048usize, 4096usize);
        let mut rng = Rng(21);
        let w = DevBuf::from_slice(&bf16_bits(&mut rng, n * k)).unwrap();
        let x = DevBuf::from_slice(&rng.vec(m * k, -1.0, 1.0)).unwrap();
        Self { w, x, y: DevBuf::zeroed(m * n * 4).unwrap(), m, n, k, rng }
    }
}

impl Case for DenseCase {
    fn launch(&mut self, st: &Stream) -> Result<(), CudaError> {
        nd::gemv_bf16(&self.w, &self.x, &mut self.y, self.m, self.n, self.k, st)
    }
    fn reset(&mut self) {
        self.y.upload(&vec![0f32; self.m * self.n]).unwrap();
        settle();
    }
    fn perturb(&mut self) {
        self.x.upload(&self.rng.vec(self.m * self.k, -1.0, 1.0)).unwrap();
    }
    fn read(&mut self) -> Vec<u32> {
        f32_bits(&self.y, self.m * self.n)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// (e) KV write and decode attention behind a RowIndex

const PAGE: usize = 256;
const KVD: usize = 2 * 128;
const QD: usize = 32 * 128;

struct AttnCase {
    kv: NemoKv,
    table: BlockTable,
    idx: RowIndex,
    scratch: Scratch,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    out: DevBuf,
    rows: usize,
    /// (slot, position) per row of the next perturbation, applied in place to `idx`'s device arrays.
    next: (Vec<i32>, Vec<i32>),
    /// A fresh index for the eager run after a perturbation (`RowIndex::from_host` is the only way to set its host-side `max_pos`).
    fresh: Option<RowIndex>,
    rng: Rng,
}

impl AttnCase {
    fn new(rows: usize) -> Self {
        let (slots, filled) = (2usize, 1100usize);
        let pages = 1100usize.div_ceil(PAGE);
        let n_blocks = pages * slots + 2;
        let mut rng = Rng(0x5eed);
        // a shuffled block pool, so the table is not the identity
        let mut ids: Vec<u32> = (0..n_blocks as u32).collect();
        for i in (1..ids.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            ids.swap(i, j);
        }
        let tables: Vec<Vec<u32>> = (0..slots).map(|s| ids[s * pages..(s + 1) * pages].to_vec()).collect();
        let mut kv = NemoKv::new(n_blocks).unwrap();
        let table = BlockTable::from_host(&tables, pages).unwrap();
        let st = Stream::new().unwrap();
        // prefill positions 0..filled of both slots with random K and V
        let (mut ps, mut pp) = (Vec::new(), Vec::new());
        for s in 0..slots {
            for p in 0..filled {
                ps.push(s as i32);
                pp.push(p as i32);
            }
        }
        let pre = RowIndex::from_host(&ps, &pp).unwrap();
        let kd = DevBuf::from_slice(&rng.vec(ps.len() * KVD, -1.0, 1.0)).unwrap();
        let vd = DevBuf::from_slice(&rng.vec(ps.len() * KVD, -1.0, 1.0)).unwrap();
        na::kv_write(&mut kv, &table, &pre, &kd, &vd, &st).unwrap();
        st.sync().unwrap();
        // the decode rows: slot 0 at 1000, slot 1 at 640, then more of the same
        let slot: Vec<i32> = (0..rows).map(|r| (r % slots) as i32).collect();
        let pos: Vec<i32> = (0..rows).map(|r| if r % 2 == 0 { 1000 - r as i32 } else { 640 + r as i32 }).collect();
        let idx = RowIndex::from_host(&slot, &pos).unwrap();
        // room for three chunks: the perturbed positions below stay inside the chunks the graph was captured with
        let scratch = Scratch::new(rows, 3).unwrap();
        let q = DevBuf::from_slice(&rng.vec(rows * QD, -1.0, 1.0)).unwrap();
        let k = DevBuf::from_slice(&rng.vec(rows * KVD, -1.0, 1.0)).unwrap();
        let v = DevBuf::from_slice(&rng.vec(rows * KVD, -1.0, 1.0)).unwrap();
        let out = DevBuf::zeroed(rows * QD * 4).unwrap();
        // the perturbation: the rows swap slots and move to other positions with the same or fewer chunks (512-key chunks)
        let nslot: Vec<i32> = (0..rows).map(|r| ((r + 1) % slots) as i32).collect();
        let npos: Vec<i32> = (0..rows).map(|r| if r % 2 == 0 { 530 + 3 * r as i32 } else { 1005 - 2 * r as i32 }).collect();
        Self { kv, table, idx, scratch, q, k, v, out, rows, next: (nslot, npos), fresh: None, rng }
    }
}

impl Case for AttnCase {
    fn launch(&mut self, st: &Stream) -> Result<(), CudaError> {
        let idx = self.fresh.as_ref().unwrap_or(&self.idx);
        na::kv_write(&mut self.kv, &self.table, idx, &self.k, &self.v, st)?;
        na::attn_decode(&self.q, &self.kv, &self.table, idx, &mut self.scratch, &mut self.out, st)
    }
    fn reset(&mut self) {
        self.out.upload(&vec![0f32; self.rows * QD]).unwrap();
        settle();
    }
    fn perturb(&mut self) {
        // in place, through the index's public device arrays: what a pre-sized index would do each step
        self.idx.slot.upload(&self.next.0).unwrap();
        self.idx.pos.upload(&self.next.1).unwrap();
        self.q.upload(&self.rng.vec(self.rows * QD, -1.0, 1.0)).unwrap();
        self.k.upload(&self.rng.vec(self.rows * KVD, -1.0, 1.0)).unwrap();
        self.v.upload(&self.rng.vec(self.rows * KVD, -1.0, 1.0)).unwrap();
        self.fresh = None;
    }
    fn eager_after_perturb(&mut self, st: &Stream) -> Result<(), CudaError> {
        // `idx.max_pos` still describes the captured positions: the eager reference needs an index built for the new ones
        self.fresh = Some(RowIndex::from_host(&self.next.0, &self.next.1).unwrap());
        self.launch(st)
    }
    fn read(&mut self) -> Vec<u32> {
        f32_bits(&self.out, self.rows * QD)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// (e2) the same under the one position rule: rows at and above 8,192 take the flash kernel (plan, partial and merge launches that read
// device buffers only), rows below it the tree, in one launch list built for a chunk bucket

const LONG_FILLED: usize = 17_000;
/// The bucket the graph is captured for: 32 tree chunks = positions below 16,384 = 2 flash splits.
const LONG_BUCKET: usize = 32;

struct AttnLongCase {
    kv: NemoKv,
    table: BlockTable,
    idx: RowIndex,
    scratch: Scratch,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    out: DevBuf,
    rows: usize,
    n_seq: usize,
    next: (Vec<i32>, Vec<i32>),
    /// The eager reference after a perturbation: an index with the host rows of the new positions and NO bucket floor, so the
    /// launch is the natural one (the chunks the deepest row has), against which the bucketed replay must be bit-equal.
    fresh: Option<RowIndex>,
    rng: Rng,
}

impl AttnLongCase {
    fn new(rows: usize) -> Self {
        let slots = 2usize;
        let pages = LONG_FILLED.div_ceil(PAGE);
        let n_blocks = pages * slots + 2;
        let mut rng = Rng(0x10_9C7);
        let mut ids: Vec<u32> = (0..n_blocks as u32).collect();
        for i in (1..ids.len()).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            ids.swap(i, j);
        }
        let tables: Vec<Vec<u32>> = (0..slots).map(|s| ids[s * pages..(s + 1) * pages].to_vec()).collect();
        let mut kv = NemoKv::new(n_blocks).unwrap();
        let table = BlockTable::from_host(&tables, pages).unwrap();
        let st = Stream::new().unwrap();
        for s in 0..slots {
            let (mut ps, mut pp) = (Vec::new(), Vec::new());
            for p in 0..LONG_FILLED {
                ps.push(s as i32);
                pp.push(p as i32);
            }
            let pre = RowIndex::from_host(&ps, &pp).unwrap();
            let kd = DevBuf::from_slice(&rng.vec(ps.len() * KVD, -1.0, 1.0)).unwrap();
            let vd = DevBuf::from_slice(&rng.vec(ps.len() * KVD, -1.0, 1.0)).unwrap();
            na::kv_write(&mut kv, &table, &pre, &kd, &vd, &st).unwrap();
            st.sync().unwrap();
        }
        // the rows: sequence 0 holds the first ceil(rows / 2) rows at consecutive positions across the threshold, sequence 1 the rest,
        // far below it (a call that holds both arithmetics)
        let n0 = rows.div_ceil(2);
        let n_seq = if rows > n0 { 2 } else { 1 };
        let slot: Vec<i32> = (0..rows).map(|r| if r < n0 { 0 } else { 1 }).collect();
        let pos: Vec<i32> = (0..rows).map(|r| if r < n0 { 8190 + r as i32 } else { 700 + r as i32 }).collect();
        let mut idx = RowIndex::from_host(&slot, &pos).unwrap();
        idx.set_min_chunks(LONG_BUCKET);
        let scratch = Scratch::new(rows.max(8), 40).unwrap();
        let q = DevBuf::from_slice(&rng.vec(rows * QD, -1.0, 1.0)).unwrap();
        let k = DevBuf::from_slice(&rng.vec(rows * KVD, -1.0, 1.0)).unwrap();
        let v = DevBuf::from_slice(&rng.vec(rows * KVD, -1.0, 1.0)).unwrap();
        let out = DevBuf::zeroed(rows * QD * 4).unwrap();
        // the perturbation: sequence 1 now holds the deep rows (up to the bucket's last position), sequence 0 the shallow ones; rows move
        // to other splits of the bucket and the row blocks change shape
        let nslot: Vec<i32> = (0..rows).map(|r| if r < n0 { 1 } else { 0 }).collect();
        let npos: Vec<i32> = (0..rows).map(|r| if r < n0 { 16_370 + r as i32 } else { 9_000 + 100 * r as i32 }).collect();
        Self { kv, table, idx, scratch, q, k, v, out, rows, n_seq, next: (nslot, npos), fresh: None, rng }
    }
}

impl Case for AttnLongCase {
    fn launch(&mut self, st: &Stream) -> Result<(), CudaError> {
        let idx = self.fresh.as_ref().unwrap_or(&self.idx);
        na::kv_write(&mut self.kv, &self.table, idx, &self.k, &self.v, st)?;
        nf::attn_long(&self.q, &self.kv, &self.table, idx, self.n_seq, &mut self.scratch, &mut self.out, st)
    }
    fn reset(&mut self) {
        self.out.upload(&vec![0f32; self.rows * QD]).unwrap();
        settle();
    }
    fn perturb(&mut self) {
        self.idx.slot.upload(&self.next.0).unwrap();
        self.idx.pos.upload(&self.next.1).unwrap();
        self.q.upload(&self.rng.vec(self.rows * QD, -1.0, 1.0)).unwrap();
        self.k.upload(&self.rng.vec(self.rows * KVD, -1.0, 1.0)).unwrap();
        self.v.upload(&self.rng.vec(self.rows * KVD, -1.0, 1.0)).unwrap();
        self.fresh = None;
    }
    fn eager_after_perturb(&mut self, st: &Stream) -> Result<(), CudaError> {
        self.fresh = Some(RowIndex::from_host(&self.next.0, &self.next.1).unwrap());
        self.launch(st)
    }
    fn read(&mut self) -> Vec<u32> {
        f32_bits(&self.out, self.rows * QD)
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// (f) the Mamba2 verify pass (pre, window), and the rest of a step (gated norm and the two commits)

struct SsmCase {
    batch: BatchBuf,
    rows: usize,
    n_seq: usize,
    conv_w: DevBuf,
    conv_b: DevBuf,
    a_log: DevBuf,
    dt_bias: DevBuf,
    d: DevBuf,
    xbc_in: DevBuf,
    dt_raw: DevBuf,
    conv_pool: DevBuf,
    s_pool: DevBuf,
    rec: ns::Record,
    o: DevBuf,
    // the rest of the step
    z: DevBuf,
    norm_w: DevBuf,
    normed: DevBuf,
    keep: DevBuf,
    full_step: bool,
    // host copies of the initial state the commits change
    conv_host: Vec<u16>,
    s_host: Vec<f32>,
    rng: Rng,
}

impl SsmCase {
    fn new(full_step: bool) -> Self {
        let (lens, slots) = ([1usize, 4], [0usize, 1]);
        let rows: usize = lens.iter().sum();
        let n_seq = lens.len();
        let mut rng = Rng(0x55a);
        let conv_host: Vec<u16> = bf16_bits(&mut rng, n_seq * ns::CONV_SLOT_ELEMS);
        let s_host: Vec<f32> = rng.vec(n_seq * ns::S_SLOT_ELEMS, -0.5, 0.5);
        let batch = BatchBuf::new(&lens, &slots).unwrap();
        Self {
            batch,
            rows,
            n_seq,
            conv_w: DevBuf::from_slice(&bf16_bits(&mut rng, ns::CONV_DIM * ns::CONV_TAPS)).unwrap(),
            conv_b: DevBuf::from_slice(&bf16_bits(&mut rng, ns::CONV_DIM)).unwrap(),
            a_log: DevBuf::from_slice(&rng.vec(ns::HEADS, 0.0, 1.0)).unwrap(),
            dt_bias: DevBuf::from_slice(&rng.vec(ns::HEADS, -1.0, 1.0)).unwrap(),
            d: DevBuf::from_slice(&rng.vec(ns::HEADS, -1.0, 1.0)).unwrap(),
            xbc_in: DevBuf::from_slice(&rng.vec(rows * ns::CONV_DIM, -1.0, 1.0)).unwrap(),
            dt_raw: DevBuf::from_slice(&rng.vec(rows * ns::HEADS, -2.0, 2.0)).unwrap(),
            conv_pool: DevBuf::from_slice(&conv_host).unwrap(),
            s_pool: DevBuf::from_slice(&s_host).unwrap(),
            rec: ns::Record::alloc(rows).unwrap(),
            o: DevBuf::zeroed(rows * ns::INNER * 2).unwrap(),
            z: DevBuf::from_slice(&rng.vec(rows * ns::INNER, -2.0, 2.0)).unwrap(),
            norm_w: DevBuf::from_slice(&rng.vec(ns::INNER, 0.5, 1.5)).unwrap(),
            normed: DevBuf::zeroed(rows * ns::INNER * 2).unwrap(),
            keep: DevBuf::from_slice(&[1i32, 3]).unwrap(),
            full_step,
            conv_host,
            s_host,
            rng,
        }
    }
}

impl Case for SsmCase {
    fn launch(&mut self, st: &Stream) -> Result<(), CudaError> {
        let b = self.batch.view();
        let w =
            ns::Weights { conv_w: &self.conv_w, conv_b: &self.conv_b, a_log: &self.a_log, dt_bias: &self.dt_bias, d: &self.d };
        ns::pre(st, &self.xbc_in, &self.dt_raw, &w, &self.conv_pool, &b, &mut self.rec)?;
        ns::window(st, &self.rec, &self.s_pool, &self.d, &b, &mut self.o)?;
        if self.full_step {
            ns::gated_norm(st, &self.o, &self.z, &self.norm_w, self.rows, &mut self.normed)?;
            ns::commit_state(st, &self.rec, &mut self.s_pool, &b, &self.keep)?;
            ns::commit_conv(st, &self.xbc_in, &mut self.conv_pool, &b, &self.keep)?;
        }
        Ok(())
    }
    fn reset(&mut self) {
        self.o.upload(&vec![0u16; self.rows * ns::INNER]).unwrap();
        self.normed.upload(&vec![0u16; self.rows * ns::INNER]).unwrap();
        self.conv_pool.upload(&self.conv_host).unwrap();
        self.s_pool.upload(&self.s_host).unwrap();
        self.rec.xbc.upload(&vec![0u16; self.rows * ns::CONV_DIM]).unwrap();
        self.rec.dt.upload(&vec![0u16; self.rows * ns::HEADS]).unwrap();
        self.rec.g.upload(&vec![0f32; self.rows * ns::HEADS]).unwrap();
        self.rec.decay.upload(&vec![0f32; self.rows * ns::HEADS]).unwrap();
        settle();
    }
    fn perturb(&mut self) {
        self.xbc_in.upload(&self.rng.vec(self.rows * ns::CONV_DIM, -1.0, 1.0)).unwrap();
        self.dt_raw.upload(&self.rng.vec(self.rows * ns::HEADS, -2.0, 2.0)).unwrap();
        self.z.upload(&self.rng.vec(self.rows * ns::INNER, -2.0, 2.0)).unwrap();
        self.s_host = self.rng.vec(self.n_seq * ns::S_SLOT_ELEMS, -0.5, 0.5);
        self.conv_host = bf16_bits(&mut self.rng, self.n_seq * ns::CONV_SLOT_ELEMS);
    }
    fn read(&mut self) -> Vec<u32> {
        let r = self.rows;
        let mut out = u16_bits(&self.o, r * ns::INNER);
        out.extend(u16_bits(&self.rec.xbc, r * ns::CONV_DIM));
        out.extend(f32_bits(&self.rec.g, r * ns::HEADS));
        out.extend(f32_bits(&self.rec.decay, r * ns::HEADS));
        if self.full_step {
            out.extend(u16_bits(&self.normed, r * ns::INNER));
            out.extend(f32_bits(&self.s_pool, self.n_seq * ns::S_SLOT_ELEMS));
            out.extend(u16_bits(&self.conv_pool, self.n_seq * ns::CONV_SLOT_ELEMS));
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The tests

/// The control: a host wait inside a capture is refused and the harness says so, and the stream is usable afterwards. A pageable copy
/// and an allocation inside a capture are reported (the runtime decides whether its capture mode refuses them), not asserted.
#[test]
fn a_sync_inside_a_capture_is_refused() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let r = Graph::capture(&st, || st.sync());
    assert!(r.is_err(), "a stream sync inside a capture must fail it");
    eprintln!("control: sync inside a capture -> {}", r.err().map(|e| e.to_string()).unwrap());
    // a pageable upload is not refused: it runs at once on the legacy stream (not part of the graph), so a replay does not repeat it
    let mut buf = DevBuf::zeroed(1024).unwrap();
    match Graph::capture(&st, || buf.upload(&[7u8; 1024])) {
        Ok(g) => {
            let ran_at_capture = buf.to_vec::<u8>().unwrap().iter().all(|&b| b == 7);
            buf.upload(&[0u8; 1024]).unwrap();
            g.launch(&st).unwrap();
            st.sync().unwrap();
            let repeated = buf.to_vec::<u8>().unwrap().contains(&7);
            eprintln!("control: pageable upload inside a capture -> captured; ran at capture time: {ran_at_capture}; repeated by a replay: {repeated}");
        }
        Err(e) => eprintln!("control: pageable upload inside a capture -> {e}"),
    }
    let r = Graph::capture(&st, || DevBuf::zeroed(1 << 20).map(|_| ()));
    eprintln!("control: allocation inside a capture -> {}", r.err().map(|e| e.to_string()).unwrap_or_else(|| "captured".into()));
    // the stream still captures and replays a launch
    let src = DevBuf::zeroed(64).unwrap();
    let mut dst = DevBuf::zeroed(128).unwrap();
    st.sync().unwrap();
    let g = Graph::capture(&st, || ng::widen_bf16_f32(&src, 16, &mut dst, &st))
        .unwrap_or_else(|e| panic!("capture after the control failed: {e}"));
    g.launch(&st).unwrap();
    st.sync().unwrap();
}

/// What a replay cannot follow: the chunk count baked into the attention launch. A graph captured with rows at most in chunk 2 (keys
/// below 1024) is replayed with a row at position 1030 (chunk 3), and its output is compared with a fresh eager run. The result for
/// that position is reported, not asserted: it is the reason an engine keys its attention graphs by `max_chunks` (and by rows).
#[test]
fn the_attention_graph_is_bounded_by_the_chunks_it_was_captured_with() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let mut c = AttnCase::new(2);
    st.sync().unwrap();
    c.reset();
    let g = Graph::capture(&st, || c.launch(&st)).unwrap_or_else(|e| panic!("{e}"));
    // inside the captured chunks first: the replay follows (the main table checks this too)
    c.next = (vec![0, 1], vec![1020, 300]);
    c.perturb();
    c.reset();
    c.eager_after_perturb(&st).unwrap();
    st.sync().unwrap();
    let inside_eager = c.read();
    c.reset();
    g.launch(&st).unwrap();
    st.sync().unwrap();
    let inside = c.read() == inside_eager;
    // now a position in a third chunk. The replay goes first: the eager run would leave the third chunk's partial in the scratch, and
    // the replay (which never computes it) would then read a correct one by accident.
    c.next = (vec![0, 1], vec![1030, 600]);
    c.perturb();
    c.reset();
    g.launch(&st).unwrap();
    st.sync().unwrap();
    let replayed = c.read();
    c.reset();
    c.eager_after_perturb(&st).unwrap();
    st.sync().unwrap();
    let outside = c.read() == replayed;
    eprintln!(
        "attention graph captured at max_chunks = 2: replay equals a fresh eager run for positions 1020 and 300 (2 and 1 chunks): {inside}; \
         for position 1030 (chunk 3): {outside}"
    );
    assert!(inside, "a replay inside the captured chunks must match a fresh eager run");
}

/// The flash path under capture. A graph captured for the 32-chunk bucket (positions below 16,384: two splits) is replayed with the
/// rows anywhere in the bucket, and equals a fresh eager run of the natural launch; a row past the bucket (a third split) is the
/// reason the model keys its attention graphs by the chunk bucket (reported, not asserted, as for the tree).
#[test]
fn the_flash_graph_is_bounded_by_the_splits_of_its_bucket_and_follows_inside_it() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let mut c = AttnLongCase::new(5);
    // the first window of a shape runs eagerly in the model before it is captured: the kernel's shared-memory attribute is set by then
    c.launch(&st).unwrap();
    st.sync().unwrap();
    c.reset();
    let g = Graph::capture(&st, || c.launch(&st)).unwrap_or_else(|e| panic!("capturing the flash launches: {e}"));
    // inside the bucket: deep rows in the first and in the second split, shallow rows, another slot assignment
    for (slots, pos) in [
        (vec![0, 0, 0, 1, 1], vec![8192, 8193, 12_000, 400, 401]),
        (vec![1, 1, 1, 0, 0], vec![16_380, 16_381, 16_382, 16_383, 9000]),
        (vec![0, 1, 0, 1, 0], vec![8191, 8192, 100, 15_000, 10_000]),
    ] {
        c.next = (slots.clone(), pos.clone());
        c.perturb();
        c.reset();
        c.eager_after_perturb(&st).unwrap();
        st.sync().unwrap();
        let eager = c.read();
        c.reset();
        g.launch(&st).unwrap();
        st.sync().unwrap();
        assert!(c.read() == eager, "the replay of the bucket's graph differs from a fresh eager run at {slots:?} {pos:?}");
    }
    // past the bucket: a third split. The replay goes first (the eager run would leave the third split's partial in the scratch)
    c.next = (vec![0, 0, 0, 1, 1], vec![16_384, 16_900, 700, 701, 702]);
    c.perturb();
    c.reset();
    g.launch(&st).unwrap();
    st.sync().unwrap();
    let replayed = c.read();
    c.reset();
    c.eager_after_perturb(&st).unwrap();
    st.sync().unwrap();
    let beyond = c.read() == replayed;
    eprintln!(
        "flash graph captured for the 32-chunk bucket (2 splits): replay equals a fresh eager run for rows below 16,384 (three \
         rearrangements, deep and shallow rows in one call): true; for rows at 16,384 and 16,900 (split 3): {beyond}"
    );
}

/// Where the time of a MoE block goes (reported; launch overhead is in the main table): the router, fc1 alone, one routed expert alone
/// and the whole block, on synthetic weights with 14 of 512 experts present.
#[test]
fn moe_block_time_split() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let experts: Vec<usize> = (0..EXPERTS).step_by(37).collect();
    let mut owned = NemotronMoe::new(synthetic_moe(61, &experts, Some(experts[3])), 64).unwrap();
    let mut rng = Rng(8);
    for rows in [1usize, 8] {
        let h = DevBuf::from_slice(&rng.vec(rows * HIDDEN, -1.0, 1.0)).unwrap();
        let x = DevBuf::from_slice(&rng.vec(rows * LATENT, -1.0, 1.0)).unwrap();
        let mut lat = DevBuf::zeroed(rows * LATENT * 4).unwrap();
        let mut one = DevBuf::zeroed(rows * LATENT * 4).unwrap();
        let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
        let route = batch_us(&st, 3, 20, || owned.route(&h, rows, &st).unwrap());
        let fc1 = batch_us(&st, 3, 20, || owned.latent_in(&h, rows, &mut lat, &st).unwrap());
        let expert = batch_us(&st, 3, 20, || owned.expert(experts[1], &x, rows, &mut one, &st).unwrap());
        let whole = batch_us(&st, 3, 20, || owned.forward(&h, rows, &mut out, &st).unwrap());
        eprintln!(
            "moe block, {rows} row(s), synthetic weights (14 of 512 experts present): route {route:.1} us, fc1 {fc1:.1} us, one expert {expert:.1} us, whole block {whole:.1} us"
        );
    }
}

#[test]
fn graph_probe_table() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let mut reports: Vec<Report> = Vec::new();

    // (a) the MoE block on one shared scratch (64 rows of scratch, as the engine would size it)
    for rows in [1usize, 8] {
        let mut c = MoeCase::new(rows, 1, 64);
        reports.push(run_case(&format!("(a) NemotronMoe::forward_in, {rows} row(s), shared MoeScratch"), &st, &mut c));
    }
    {
        let mut c = MoeCase::new(8, 2, 64);
        reports.push(run_case("(a) two MoE layers chained on one MoeScratch, 8 rows, one graph", &st, &mut c));
    }
    // (b) one EXL3 linear
    for m in [1usize, 8] {
        let mut c = ExlCase::new(m);
        reports.push(run_case(&format!("(b) Exl3Weights::forward 4096 -> 1024, {m} row(s)"), &st, &mut c));
    }
    // (c) the glue kernels
    for rows in [1usize, 8] {
        let mut c = GlueCase::new(rows);
        reports.push(run_case(
            &format!("(c) glue: embed, norm, split, top-k (131072, 40), widens, splice, {rows} row(s)"),
            &st,
            &mut c,
        ));
    }
    // (d) the dense GEMV
    for m in [1usize, 8] {
        let mut c = DenseCase::new(m);
        reports.push(run_case(&format!("(d) nemotron_dense::gemv_bf16 4096 -> 2048, {m} row(s)"), &st, &mut c));
    }
    // (e) KV write and decode attention
    for rows in [1usize, 2, 8] {
        let mut c = AttnCase::new(rows);
        reports.push(run_case(
            &format!("(e) nemotron_attn kv_write + attn_decode, {rows} row(s), RowIndex updated in place"),
            &st,
            &mut c,
        ));
    }
    // (e2) the same under the one position rule: flash rows beside tree rows in one launch list built for the 32-chunk bucket
    for rows in [1usize, 5, 8] {
        let mut c = AttnLongCase::new(rows);
        reports.push(run_case(
            &format!("(e2) nemotron_attn kv_write + attn_long (flash at >= 8192, bucket 32 chunks), {rows} row(s)"),
            &st,
            &mut c,
        ));
    }
    // (f) Mamba2: pre + window, then the whole step
    {
        let mut c = SsmCase::new(false);
        reports.push(run_case("(f) nemotron_ssm pre + window, 5 rows in 2 sequences", &st, &mut c));
        let mut c = SsmCase::new(true);
        reports.push(run_case("(f) nemotron_ssm pre + window + gated_norm + commit_state + commit_conv", &st, &mut c));
    }

    print_table("G0 graph probe", &reports);
    let failed: Vec<&str> = reports.iter().filter(|r| !r.passed()).map(|r| r.name.as_str()).collect();
    assert!(failed.is_empty(), "paths that do not capture and replay bit-identically: {failed:?}");
}
