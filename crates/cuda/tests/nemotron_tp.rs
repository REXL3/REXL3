//! The kernel-side pieces of two-rank TP-exact for Nemotron (`csrc/nemotron_tp.cu`, `NemotronMoe::forward_in_tp`,
//! `PrefillMoe::forward_prefill_in_tp`; design in `docs/NEMOTRON_TP.md`). Synthetic weights, no pack needed.
//!
//! The rule under test: every element is computed by one rank with the single-node kernel and the other rank receives it by a sum over
//! buffers of disjoint support (`x + 0.0 == x`), so the TP result is the single-node result bit for bit.
//!
//! * `scatter_cols_f32`: a bit-exact copy into a zeroed full-width row, against a CPU model, at 1 to 64 rows, on the Nemotron splits
//!   (multiples of 128 columns) and on odd widths and misaligned base addresses (the element-wise path).
//! * The MoE expert-parallel hook: the block built whole, and as two ranks (rank 0 owns experts `[0, 256)`, rank 1 `[256, 512)`, the
//!   others absent). Per (row, slot) pair the owner's `y` row equals the whole block's bit for bit and the other rank's is exact `+0.0`; the
//!   sum of the two `y` is the whole `y`; the rest of the block run on the summed `y` gives the whole block's output bit for bit; a
//!   no-op hook is `forward_in` (`forward_prefill_in`) bit for bit. Natural, skewed (every pick in one rank's half, both halves) and
//!   checkerboard routing; the fast path, the reference path and the prefill path; with and without bf16 experts; graph capture.
//!
//!     cargo test -p gb-cuda --release --test nemotron_tp -- --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::exl3_synth::synth_group;
use common::Rng;
use core::ffi::c_void;
use gb_cuda::graph::Graph;
use gb_cuda::nemotron_moe::{MoeScratch, Proj, ReduceHook, Weights, EXPERTS, EXPERT_INTER, HIDDEN, LATENT, SHARED_INTER, TOPK};
use gb_cuda::nemotron_moe_prefill::{PrefillMoe, PrefillScratch};
use gb_cuda::nemotron_tp::{scatter_cols_f32, scatter_cols_f32_cpu};
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

const SCALE: f32 = 5.0;
/// Experts of rank 0: `[0, HALF)`; rank 1 owns `[HALF, 512)`.
const HALF: usize = EXPERTS / 2;

// ---------------------------------------------------------------------------------------------------------------------------------
// scatter_cols_f32

/// Random f32 bit patterns of every kind (NaN payloads, infinities, denormals, signed zeros): a copy must keep all of them.
fn rand_bits(seed: u64, n: usize) -> Vec<f32> {
    let mut r = Rng(seed);
    let mut v: Vec<f32> = (0..n).map(|_| f32::from_bits(r.next_u64() as u32)).collect();
    // make sure the special values are there whatever n is
    for (i, bits) in [0x8000_0000u32, 0x7fc0_1234, 0xffff_ffff, 0x7f80_0000, 0xff80_0000, 0x0000_0001, 0x8000_0001, 0x0000_0000]
        .iter()
        .enumerate()
    {
        if i < n {
            v[i * (n / 8).max(1)] = f32::from_bits(*bits);
        }
    }
    v
}

fn bits_equal(a: &[f32], b: &[f32]) -> Option<usize> {
    assert_eq!(a.len(), b.len());
    (0..a.len()).find(|&i| a[i].to_bits() != b[i].to_bits())
}

/// One scatter on the device against the CPU model. `mis`: the buffers start 4 bytes into an allocation (not 16-byte aligned).
fn check_scatter(st: &Stream, rows: usize, n_local: usize, n_total: usize, col_off: usize, mis: bool, seed: u64) {
    let src_h = rand_bits(seed, rows * n_local);
    let want = scatter_cols_f32_cpu(&src_h, rows, n_local, n_total, col_off);
    let poison = vec![f32::from_bits(0xdead_beef); rows * n_total];
    let (src, mut dst);
    if mis {
        let sb = DevBuf::zeroed_view(rows * n_local * 4 + 16).unwrap();
        let db = DevBuf::zeroed_view(rows * n_total * 4 + 16).unwrap();
        let (mut s, mut d) = (sb.subview(4, rows * n_local * 4), db.subview(4, rows * n_total * 4));
        s.upload(&src_h).unwrap();
        d.upload(&poison).unwrap();
        // keep the parent views alive through the views' own owner Arc
        src = s;
        dst = d;
    } else {
        src = DevBuf::from_slice(&src_h).unwrap();
        dst = DevBuf::from_slice(&poison).unwrap();
    }
    scatter_cols_f32(&src, &mut dst, rows, n_local, n_total, col_off, st).unwrap();
    st.sync().unwrap();
    let got = dst.to_vec::<f32>().unwrap();
    let got = &got[..rows * n_total];
    if let Some(i) = bits_equal(got, &want) {
        panic!(
            "scatter rows {rows} n_local {n_local} n_total {n_total} off {col_off} mis {mis}: element {i} (row {}, col {}) is {:08x}, want {:08x}",
            i / n_total,
            i % n_total,
            got[i].to_bits(),
            want[i].to_bits()
        );
    }
}

#[test]
fn scatter_matches_the_cpu_model_at_1_to_64_rows() {
    gpu!();
    let st = Stream::new().unwrap();
    // (n_local, n_total, col_off): the Nemotron splits (multiples of 128) and the edges of the window
    let shapes: &[(usize, usize, usize)] = &[
        (128, 256, 0),
        (128, 256, 128),
        (128, 128, 0),    // the whole row (no zeros)
        (256, 1024, 384), // inside
        (128, 4096, 3968),
        (2048, 4096, 0), // out_proj / q / o: half of 4096
        (2048, 4096, 2048),
        (9216, 18560, 0),    // in_proj cut: 72 tiles
        (9344, 18560, 9216), // 73 tiles
        (128, 256, 0),
    ];
    for &(nl, nt, off) in shapes {
        for rows in 1..=64usize {
            check_scatter(&st, rows, nl, nt, off, false, 0x5ca7 + rows as u64 * 31 + nl as u64);
        }
        eprintln!("scatter {nl} of {nt} at {off}: rows 1..=64 bitwise the CPU model");
    }
    // lm_head: the vocabulary split 65536 / 65536 (33 MB of destination at 64 rows)
    for &rows in &[1usize, 2, 5, 22, 33, 64] {
        check_scatter(&st, rows, 65536, 131072, 0, false, 11 + rows as u64);
        check_scatter(&st, rows, 65536, 131072, 65536, false, 12 + rows as u64);
    }
    // kv: 128 of 256 columns
    for rows in 1..=64usize {
        check_scatter(&st, rows, 128, 256, 128, false, 99 + rows as u64);
    }
    eprintln!("scatter lm_head 65536 of 131072 and kv 128 of 256: bitwise the CPU model");
}

#[test]
fn scatter_odd_widths_and_misaligned_buffers_take_the_scalar_path_with_the_same_bits() {
    gpu!();
    let st = Stream::new().unwrap();
    for &(nl, nt, off) in
        &[(7usize, 19usize, 3usize), (66, 130, 2), (1, 1, 0), (1, 5, 4), (5, 5, 0), (0, 12, 0), (128, 256, 1), (129, 257, 128)]
    {
        for rows in [1usize, 2, 7, 64] {
            check_scatter(&st, rows, nl, nt, off, false, 7 + rows as u64);
        }
    }
    // aligned shapes in buffers that are not 16-byte aligned
    for &(nl, nt, off) in &[(128usize, 256usize, 128usize), (2048, 4096, 2048), (9344, 18560, 9216)] {
        for rows in [1usize, 5, 64] {
            check_scatter(&st, rows, nl, nt, off, true, 21 + rows as u64);
        }
    }
    // out-of-range windows are refused without a launch
    let src = DevBuf::zeroed(4 * 4 * 8).unwrap();
    let mut dst = DevBuf::zeroed(4 * 8 * 8).unwrap();
    assert!(scatter_cols_f32(&src, &mut dst, 4, 8, 16, 9, &st).is_err());
    assert!(scatter_cols_f32(&src, &mut dst, 4, 8, 0, 0, &st).is_err());
    // zero rows is a no-op
    assert!(scatter_cols_f32(&src, &mut dst, 0, 8, 16, 0, &st).is_ok());
}

/// Two ranks' scatters summed on the host are the gathered row, bit for bit: the identity the allreduce relies on (here with the sum
/// done on the host, and including signed zeros and NaN payloads in the data: the scatter's filler is -0.0, the true additive identity,
/// so the sum keeps the sign of a zero too, which the step digest's raw-bit hashes of the S pool and the window records need).
#[test]
fn two_scatters_summed_are_the_gathered_row() {
    gpu!();
    let st = Stream::new().unwrap();
    let (rows, nt, cut) = (7usize, 18560usize, 9216usize);
    let whole: Vec<f32> = rand_bits(0x6a7e, rows * nt);
    let halves: [(usize, usize); 2] = [(0, cut), (cut, nt)];
    let mut sum: Option<Vec<f32>> = None;
    for (lo, hi) in halves {
        let nl = hi - lo;
        let mut local = Vec::with_capacity(rows * nl);
        for r in 0..rows {
            local.extend_from_slice(&whole[r * nt + lo..r * nt + hi]);
        }
        let src = DevBuf::from_slice(&local).unwrap();
        let mut dst = DevBuf::zeroed(rows * nt * 4).unwrap();
        scatter_cols_f32(&src, &mut dst, rows, nl, nt, lo, &st).unwrap();
        st.sync().unwrap();
        let d = dst.to_vec::<f32>().unwrap();
        match sum.as_mut() {
            None => sum = Some(d),
            Some(s) => {
                for (a, b) in s.iter_mut().zip(&d) {
                    *a += *b;
                }
            }
        }
    }
    let sum = sum.unwrap();
    // NaNs: x + id keeps a NaN's payload (the quiet bit may be set on a signalling one by the host add, so compare as NaN class)
    let bad = (0..sum.len()).find(|&i| !(sum[i].to_bits() == whole[i].to_bits() || (sum[i].is_nan() && whole[i].is_nan())));
    assert!(bad.is_none(), "the sum of the two scatters differs from the whole row at {}", bad.unwrap_or(0));
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The MoE expert-parallel hook

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Routing {
    /// Random gate and bias: picks spread over both halves.
    Natural,
    /// Every pick in rank 0's half; rank 1's `y` is all zeros.
    LowOnly,
    /// Every pick in rank 1's half; rank 0's `y` is all zeros.
    HighOnly,
    /// Even rows alternate low, high, low, ... in rank order; odd rows start with the high half: pair (row, slot) belongs to rank
    /// (row + slot) % 2.
    Checker,
}

/// The router tables of a routing: the bf16 gate bits `[512, 4096]` and the bias `[512]`.
fn router_tables(routing: Routing, seed: u64) -> (Vec<u16>, Vec<f32>) {
    let mut r = Rng(seed);
    let bf = |v: f32| (v.to_bits() >> 16) as u16;
    match routing {
        Routing::Natural => {
            let gate = (0..EXPERTS * HIDDEN).map(|_| bf(r.uniform(-0.05, 0.05))).collect();
            (gate, r.vec(EXPERTS, -0.1, 0.1))
        }
        // logits are zero (gate zero), so the keys are 0.5 + bias: the largest 22 biases are picked, in order
        Routing::LowOnly => {
            let bias = (0..EXPERTS).map(|e| if e < HALF { 10.0 - 0.01 * e as f32 } else { -0.01 * e as f32 }).collect();
            (vec![0; EXPERTS * HIDDEN], bias)
        }
        Routing::HighOnly => {
            let bias = (0..EXPERTS).map(|e| if e >= HALF { 10.0 - 0.01 * (e - HALF) as f32 } else { -0.01 * e as f32 }).collect();
            (vec![0; EXPERTS * HIDDEN], bias)
        }
        // logit_e = h[0] * (+-A): sigmoid differs by d ~ 0.01 between the halves, in favour of the half h[0]'s sign selects; the biases
        // step by 0.02 per index inside a half, so the order is low0, high0, low1, high1, ... (h[0] > 0) or the other way round
        Routing::Checker => {
            let a = 0.02f32;
            let mut gate = vec![0u16; EXPERTS * HIDDEN];
            for e in 0..EXPERTS {
                gate[e * HIDDEN] = bf(if e < HALF { a } else { -a });
            }
            let bias = (0..EXPERTS).map(|e| -0.02 * (e % HALF) as f32).collect();
            (gate, bias)
        }
    }
}

/// The input rows of a routing (`h[row][0]` carries the sign the checkerboard needs).
fn make_h(routing: Routing, rows: usize, seed: u64) -> Vec<f32> {
    let mut h = Rng(seed).vec(rows * HIDDEN, -1.0, 1.0);
    if routing == Routing::Checker {
        for r in 0..rows {
            h[r * HIDDEN] = if r % 2 == 0 { 1.0 } else { -1.0 };
        }
    }
    h
}

fn bf16_bits(r: &mut Rng, n: usize) -> Vec<u16> {
    (0..n).map(|_| (r.uniform(-0.05, 0.05).to_bits() >> 16) as u16).collect()
}

/// The block built whole, and as the two ranks of an expert-parallel group (the same weights; a rank's other experts are absent).
struct Trio {
    whole: PrefillMoe,
    r0: PrefillMoe,
    r1: PrefillMoe,
    present: Vec<usize>,
}

/// `present`: the experts that exist (the others are absent in all three blocks and must not be routed to). `bf16`: those of them that
/// are bf16 pairs (the rest are 4-bit EXL3).
fn build_trio(seed: u64, routing: Routing, present: &[usize], bf16: &[usize]) -> Trio {
    let (gate_bits, bias) = router_tables(routing, seed);
    let fc1 = synth_group(HIDDEN, LATENT, 4, Codebook::Mul1, seed + 1);
    let fc2 = synth_group(LATENT, HIDDEN, 4, Codebook::Mul1, seed + 2);
    let su = synth_group(HIDDEN, SHARED_INTER, 4, Codebook::Mul1, seed + 3);
    let sd = synth_group(SHARED_INTER, HIDDEN, 4, Codebook::Mul1, seed + 4);
    let mut up: [Vec<Proj>; 3] = [0, 1, 2].map(|_| (0..EXPERTS).map(|_| Proj::Absent).collect());
    let mut down: [Vec<Proj>; 3] = [0, 1, 2].map(|_| (0..EXPERTS).map(|_| Proj::Absent).collect());
    for &e in present {
        let owner = 1 + e / HALF; // index into the three tables: 0 whole, 1 rank 0, 2 rank 1
        if bf16.contains(&e) {
            let mut r = Rng(seed + 5000 + e as u64);
            let uw = bf16_bits(&mut r, EXPERT_INTER * LATENT);
            let dw = bf16_bits(&mut r, LATENT * EXPERT_INTER);
            for t in [0, owner] {
                up[t][e] = Proj::bf16(LATENT, EXPERT_INTER, &uw).unwrap();
                down[t][e] = Proj::bf16(EXPERT_INTER, LATENT, &dw).unwrap();
            }
        } else {
            let ug = synth_group(LATENT, EXPERT_INTER, 4, Codebook::Mul1, seed + 1000 + 2 * e as u64);
            let dg = synth_group(EXPERT_INTER, LATENT, 4, Codebook::Mul1, seed + 1001 + 2 * e as u64);
            for t in [0, owner] {
                up[t][e] = Proj::exl3_from_group(&ug).unwrap();
                down[t][e] = Proj::exl3_from_group(&dg).unwrap();
            }
        }
    }
    let mut blocks = Vec::new();
    for (u, d) in up.into_iter().zip(down) {
        let w = Weights {
            gate: DevBuf::from_slice(&gate_bits).unwrap(),
            gate_dtype: gb_cuda::nemotron_moe::GateDtype::Bf16,
            router_bias: DevBuf::from_slice(&bias).unwrap(),
            scale: SCALE,
            fc1: Proj::exl3_from_group(&fc1).unwrap(),
            fc2: Proj::exl3_from_group(&fc2).unwrap(),
            shared_up: Proj::exl3_from_group(&su).unwrap(),
            shared_down: Proj::exl3_from_group(&sd).unwrap(),
            up: u,
            down: d,
        };
        let moe = PrefillMoe::tables(w).unwrap();
        assert!(moe.supports_prefill());
        blocks.push(moe);
    }
    let r1 = blocks.pop().unwrap();
    let r0 = blocks.pop().unwrap();
    let whole = blocks.pop().unwrap();
    Trio { whole, r0, r1, present: present.to_vec() }
}

/// The experts a controlled routing can pick (so the whole block need only hold these).
fn controlled_present(routing: Routing) -> Vec<usize> {
    match routing {
        Routing::Natural => (0..EXPERTS).collect(),
        Routing::LowOnly => (0..24).collect(),
        Routing::HighOnly => (HALF..HALF + 24).collect(),
        Routing::Checker => (0..12).chain(HALF..HALF + 12).collect(),
    }
}

/// A hook that leaves `y` alone.
fn noop(_y: &mut DevBuf, _n: usize, _st: &Stream) -> Result<(), CudaError> {
    Ok(())
}

/// A hook that records `y[..n]` on the host (a host sync: for tests, never inside a capture).
fn record<'a>(store: &'a mut Vec<f32>) -> impl FnMut(&mut DevBuf, usize, &Stream) -> Result<(), CudaError> + 'a {
    move |y, n, st| {
        st.sync()?;
        store.resize(n, 0.0);
        y.download_range(0, &mut store[..])
    }
}

/// A hook that is the allreduce of a two-rank group, done on the host: `y += peer` (the peer's recorded `y`), written back; records the
/// rank's own `y` before the sum.
fn sum_with<'a>(peer: &'a [f32], own: &'a mut Vec<f32>) -> impl FnMut(&mut DevBuf, usize, &Stream) -> Result<(), CudaError> + 'a {
    move |y, n, st| {
        st.sync()?;
        assert_eq!(peer.len(), n, "the peer's y has {} elements, this rank's {n}", peer.len());
        own.resize(n, 0.0);
        y.download_range(0, &mut own[..])?;
        let sum: Vec<f32> = own.iter().zip(peer).map(|(a, b)| a + b).collect();
        y.upload(&sum)
    }
}

/// One run of a block: its output rows and the picks of the route.
struct BlockRun {
    out: Vec<f32>,
    pick: Vec<i32>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Path {
    Fast,
    Reference,
    Prefill,
}

struct Scratches {
    dec: MoeScratch,
    pre: PrefillScratch,
}

/// `moe` on `rows` rows of `h` along `path`, plain (`hook == None`: the single-node entry point) or through the `_tp` entry point with
/// `hook`. The scratch is poisoned first, so a stage that does not write what it should cannot pass on a previous call's values.
fn run_block(
    moe: &PrefillMoe,
    sc: &mut Scratches,
    path: Path,
    h: &DevBuf,
    rows: usize,
    hook: Option<&mut ReduceHook>,
    st: &Stream,
) -> BlockRun {
    let mut out = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * HIDDEN]).unwrap();
    // a non-blocking `st` does not wait for the legacy-stream uploads (`h`, and this NaN poison, which a late DMA would write
    // over the block's output)
    gb_cuda::rt::device_sync().unwrap();
    let pick = match path {
        Path::Fast | Path::Reference => {
            let s = &mut sc.dec;
            s.poison(st).unwrap();
            let res = match (path, hook) {
                (Path::Fast, None) => moe.forward_in(s, h, rows, &mut out, st),
                (Path::Fast, Some(r)) => moe.forward_in_tp(s, h, rows, &mut out, st, r),
                (_, None) => moe.forward_reference(s, h, rows, &mut out, st),
                (_, Some(r)) => moe.forward_reference_in_tp(s, h, rows, &mut out, st, r),
            };
            res.unwrap();
            st.sync().unwrap();
            s.download_route(rows, st).unwrap().0
        }
        Path::Prefill => {
            let s = &mut sc.pre;
            s.poison(st).unwrap();
            let res = match hook {
                None => moe.forward_prefill_in(s, h, rows, &mut out, st),
                Some(r) => moe.forward_prefill_in_tp(s, h, rows, &mut out, st, r),
            };
            res.unwrap();
            st.sync().unwrap();
            s.download_route(rows, st).unwrap().0
        }
    };
    BlockRun { out: out.to_vec::<f32>().unwrap(), pick }
}

fn assert_bits(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    if let Some(i) = bits_equal(got, want) {
        let bad = (0..got.len()).filter(|&j| got[j].to_bits() != want[j].to_bits()).count();
        panic!(
            "{what}: {bad} of {} values differ in bits, first at {i} ({:08x} against {:08x})",
            got.len(),
            got[i].to_bits(),
            want[i].to_bits()
        );
    }
}

/// What one comparison found, for the report line.
struct Found {
    pairs: [usize; 2],
    sign_flips: usize,
}

/// The whole battery for one (block trio, path, rows).
fn check_case(label: &str, trio: &Trio, sc: &mut Scratches, path: Path, routing: Routing, rows: usize, st: &Stream) -> Found {
    let h_host = make_h(routing, rows, 0x7e57 + rows as u64);
    let h = DevBuf::from_slice(&h_host).unwrap();
    let what = |s: &str| format!("{label} {path:?} rows {rows}: {s}");

    // the single-node block, and the hook entry point with a no-op hook on each of the three blocks
    let plain_w = run_block(&trio.whole, sc, path, &h, rows, None, st);
    assert!(plain_w.out.iter().all(|v| v.is_finite()), "{}", what("the whole block's output is not finite"));
    assert!(plain_w.out.iter().any(|v| *v != 0.0), "{}", what("the whole block's output is all zero"));
    for p in &plain_w.pick {
        assert!(trio.present.contains(&(*p as usize)), "{}", what(&format!("routed to expert {p}, which is absent")));
    }
    let mut nop = noop;
    let nop_w = run_block(&trio.whole, sc, path, &h, rows, Some(&mut nop), st);
    assert_bits(&what("whole: no-op hook output against the plain entry point"), &nop_w.out, &plain_w.out);
    assert_eq!(nop_w.pick, plain_w.pick);
    let plain_0 = run_block(&trio.r0, sc, path, &h, rows, None, st);
    let nop_0 = run_block(&trio.r0, sc, path, &h, rows, Some(&mut nop), st);
    assert_bits(&what("rank 0: no-op hook output against the plain entry point"), &nop_0.out, &plain_0.out);
    let plain_1 = run_block(&trio.r1, sc, path, &h, rows, None, st);
    let nop_1 = run_block(&trio.r1, sc, path, &h, rows, Some(&mut nop), st);
    assert_bits(&what("rank 1: no-op hook output against the plain entry point"), &nop_1.out, &plain_1.out);
    // the router is replicated: every block picks the same experts
    assert_eq!(plain_0.pick, plain_w.pick, "{}", what("rank 0's route differs from the whole block's"));
    assert_eq!(plain_1.pick, plain_w.pick, "{}", what("rank 1's route differs from the whole block's"));

    // y of each block
    let (mut yw, mut y0, mut y1) = (Vec::new(), Vec::new(), Vec::new());
    {
        let mut rec = record(&mut yw);
        let r = run_block(&trio.whole, sc, path, &h, rows, Some(&mut rec), st);
        assert_bits(&what("whole: recording hook output"), &r.out, &plain_w.out);
    }
    {
        let mut rec = record(&mut y0);
        run_block(&trio.r0, sc, path, &h, rows, Some(&mut rec), st);
    }
    {
        let mut rec = record(&mut y1);
        run_block(&trio.r1, sc, path, &h, rows, Some(&mut rec), st);
    }
    let n = rows * TOPK * LATENT;
    assert_eq!((yw.len(), y0.len(), y1.len()), (n, n, n));
    assert!(yw.iter().all(|v| v.is_finite()), "{}", what("the whole y is not finite"));

    // per pair: the owner's row is the whole block's row (same kernel, same bits, signed zeros too), the other rank's is exact +0.0
    let mut pairs = [0usize; 2];
    for p in 0..rows * TOPK {
        let owner = if (plain_w.pick[p] as usize) < HALF { 0 } else { 1 };
        pairs[owner] += 1;
        let rg = p * LATENT..(p + 1) * LATENT;
        let (own, other) = if owner == 0 { (&y0, &y1) } else { (&y1, &y0) };
        assert_bits(
            &what(&format!(
                "pair {p} (row {}, slot {}, expert {}): the owner's y against the whole's",
                p / TOPK,
                p % TOPK,
                plain_w.pick[p]
            )),
            &own[rg.clone()],
            &yw[rg.clone()],
        );
        let nz = other[rg.clone()].iter().position(|v| v.to_bits() != 0);
        assert!(
            nz.is_none(),
            "{}",
            what(&format!(
                "pair {p}: the non-owner's y element {} is {:08x}, not +0.0",
                nz.unwrap_or(0),
                other[p * LATENT + nz.unwrap_or(0)].to_bits()
            ))
        );
    }
    // the sum of the two ranks' y is the whole y (the sign of a zero may flip: x + 0.0)
    let mut sign_flips = 0;
    for i in 0..n {
        let s = y0[i] + y1[i];
        if s.to_bits() != yw[i].to_bits() {
            assert!(
                s == 0.0 && yw[i] == 0.0,
                "{}",
                what(&format!(
                    "the summed y differs from the whole y at {i}: {:08x} against {:08x}",
                    s.to_bits(),
                    yw[i].to_bits()
                ))
            );
            sign_flips += 1;
        }
    }

    // the rest of the block on the summed y: combine, fc2, shared expert and the add give the whole block's output, on both ranks
    let (mut own0, mut own1) = (Vec::new(), Vec::new());
    let tp0 = {
        let mut hook = sum_with(&y1, &mut own0);
        run_block(&trio.r0, sc, path, &h, rows, Some(&mut hook), st)
    };
    assert_bits(&what("rank 0 with the summed y: block output against the whole block's"), &tp0.out, &plain_w.out);
    let tp1 = {
        let mut hook = sum_with(&y0, &mut own1);
        run_block(&trio.r1, sc, path, &h, rows, Some(&mut hook), st)
    };
    assert_bits(&what("rank 1 with the summed y: block output against the whole block's"), &tp1.out, &plain_w.out);
    // the two ranks computed the same y before the sum that the recording runs saw (determinism of the absent-expert paths)
    assert_bits(&what("rank 0's y is the same in two runs"), &own0, &y0);
    assert_bits(&what("rank 1's y is the same in two runs"), &own1, &y1);
    Found { pairs, sign_flips }
}

fn scratches() -> Scratches {
    Scratches { dec: MoeScratch::new(64).unwrap(), pre: PrefillScratch::new(256).unwrap() }
}

fn run_config(label: &str, routing: Routing, present: Vec<usize>, bf16: &[usize], seed: u64) {
    let st = Stream::new_nonblocking().unwrap();
    let t0 = std::time::Instant::now();
    let trio = build_trio(seed, routing, &present, bf16);
    eprintln!(
        "{label}: built whole + 2 ranks in {:.1}s ({} present experts, {} bf16, slow experts whole/rank0/rank1 = {}/{}/{})",
        t0.elapsed().as_secs_f64(),
        present.len(),
        bf16.len(),
        trio.whole.slow_experts(),
        trio.r0.slow_experts(),
        trio.r1.slow_experts()
    );
    let mut sc = scratches();
    for &rows in &[1usize, 5, 22, 64] {
        for path in [Path::Fast, Path::Reference] {
            let f = check_case(label, &trio, &mut sc, path, routing, rows, &st);
            eprintln!(
                "{label}: {path:?} {rows:>3} rows: owner rows == whole, non-owner +0.0, sum == whole, combine+fc2+shared == whole (pairs rank0/rank1 {}/{}; {} zero sign flips in the sum)",
                f.pairs[0], f.pairs[1], f.sign_flips
            );
        }
    }
    for &rows in &[128usize, 256] {
        let f = check_case(label, &trio, &mut sc, Path::Prefill, routing, rows, &st);
        eprintln!(
            "{label}: Prefill {rows:>3} rows: owner rows == whole, non-owner +0.0, sum == whole, combine+fc2+shared == whole (pairs rank0/rank1 {}/{}; {} zero sign flips in the sum)",
            f.pairs[0], f.pairs[1], f.sign_flips
        );
    }
    match routing {
        Routing::LowOnly => {
            let h = DevBuf::from_slice(&make_h(routing, 5, 3)).unwrap();
            let mut y1 = Vec::new();
            {
                let mut rec = record(&mut y1);
                run_block(&trio.r1, &mut sc, Path::Fast, &h, 5, Some(&mut rec), &st);
            }
            assert!(y1.iter().all(|v| v.to_bits() == 0), "{label}: rank 1 owns no pick, its y must be all +0.0");
        }
        Routing::HighOnly => {
            let h = DevBuf::from_slice(&make_h(routing, 5, 3)).unwrap();
            let mut y0 = Vec::new();
            {
                let mut rec = record(&mut y0);
                run_block(&trio.r0, &mut sc, Path::Fast, &h, 5, Some(&mut rec), &st);
            }
            assert!(y0.iter().all(|v| v.to_bits() == 0), "{label}: rank 0 owns no pick, its y must be all +0.0");
        }
        _ => {}
    }
}

/// Natural routing, 512 experts, two of them bf16 (one per rank): a rank block is slow (absent and bf16) in both paths.
#[test]
fn expert_parallel_exact_natural_routing_with_bf16_experts() {
    gpu!();
    run_config("natural+bf16", Routing::Natural, controlled_present(Routing::Natural), &[5, 300], 0x7a11);
}

/// Natural routing, 512 4-bit experts.
#[test]
fn expert_parallel_exact_natural_routing_4bit() {
    gpu!();
    run_config("natural", Routing::Natural, controlled_present(Routing::Natural), &[], 0x7a12);
}

/// Skewed: every pick of every row in rank 0's half, then in rank 1's half.
#[test]
fn expert_parallel_exact_skewed_routing() {
    gpu!();
    run_config("skew-low", Routing::LowOnly, controlled_present(Routing::LowOnly), &[], 0x7a13);
    run_config("skew-high", Routing::HighOnly, controlled_present(Routing::HighOnly), &[], 0x7a14);
}

/// Checkerboard: slot s of row r belongs to rank (r + s) % 2.
#[test]
fn expert_parallel_exact_checkerboard_routing() {
    gpu!();
    let present = controlled_present(Routing::Checker);
    run_config("checker", Routing::Checker, present.clone(), &[], 0x7a15);
    // and the same ownership with a bf16 pair on each side
    run_config("checker+bf16", Routing::Checker, present, &[2, HALF + 3], 0x7a16);
}

/// The checkerboard really is one: the picks of the routing alternate between the halves slot by slot (a sanity check of the test's
/// own routing tables, so that "checkerboard" in the report means what it says).
#[test]
fn the_checkerboard_routing_alternates_ownership() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let trio = build_trio(0x7a15, Routing::Checker, &controlled_present(Routing::Checker), &[]);
    let mut sc = scratches();
    let rows = 8;
    let h = DevBuf::from_slice(&make_h(Routing::Checker, rows, 5)).unwrap();
    let r = run_block(&trio.whole, &mut sc, Path::Fast, &h, rows, None, &st);
    for row in 0..rows {
        for slot in 0..TOPK {
            let owner = if (r.pick[row * TOPK + slot] as usize) < HALF { 0 } else { 1 };
            assert_eq!(owner, (row + slot) % 2, "row {row} slot {slot}: expert {}", r.pick[row * TOPK + slot]);
        }
    }
    eprintln!("checkerboard routing: pair (row, slot) belongs to rank (row + slot) % 2 for 8 rows x 22 slots");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Graph capture

/// `forward_in_tp` and `forward_prefill_in_tp` capture into a CUDA graph (the side stream's fork and join included) with a hook that
/// queues a device copy, and the replay equals the eager run and the whole block, bit for bit.
#[test]
fn the_tp_block_captures_into_a_graph_and_replays_exactly() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let routing = Routing::Checker;
    let trio = build_trio(0x7a17, routing, &controlled_present(routing), &[]);
    let mut sc = scratches();
    for &(rows, prefill) in &[(1usize, false), (5, false), (16, false), (22, false), (64, false), (128, true), (256, true)] {
        let h_host = make_h(routing, rows, 0x6a + rows as u64);
        let h = DevBuf::from_slice(&h_host).unwrap();
        let path = if prefill { Path::Prefill } else { Path::Fast };
        // the whole block's output, and both ranks' y, eagerly
        let whole = run_block(&trio.whole, &mut sc, path, &h, rows, None, &st);
        let (mut y0, mut y1) = (Vec::new(), Vec::new());
        {
            let mut rec = record(&mut y0);
            run_block(&trio.r0, &mut sc, path, &h, rows, Some(&mut rec), &st);
        }
        {
            let mut rec = record(&mut y1);
            run_block(&trio.r1, &mut sc, path, &h, rows, Some(&mut rec), &st);
        }
        let n = y0.len();
        let summed_host: Vec<f32> = y0.iter().zip(&y1).map(|(a, b)| a + b).collect();
        let summed = DevBuf::from_slice(&summed_host).unwrap();
        // rank 0 through a graph: the hook replaces y with the summed buffer by a device copy (what an allreduce on the stream is, to the
        // block's dependencies)
        let mut out = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * HIDDEN]).unwrap();
        gb_cuda::rt::device_sync().unwrap();
        let graph = {
            let (dec, pre) = (&mut sc.dec, &mut sc.pre);
            let moe = &trio.r0;
            let mut hook =
                |y: &mut DevBuf, count: usize, s: &Stream| -> Result<(), CudaError> { y.copy_from(&summed, count * 4, s) };
            assert_eq!(n, rows * TOPK * LATENT);
            Graph::capture(&st, || {
                if prefill {
                    moe.forward_prefill_in_tp(pre, &h, rows, &mut out, &st, &mut hook)
                } else {
                    moe.forward_in_tp(dec, &h, rows, &mut out, &st, &mut hook)
                }
            })
            .unwrap_or_else(|e| panic!("rows {rows}: the capture failed: {e}"))
        };
        for rep in 0..3 {
            out.memset_async(0xFF, &st).unwrap();
            graph.launch(&st).unwrap();
            st.sync().unwrap();
            assert_bits(
                &format!("rows {rows} graph replay {rep} against the whole block"),
                &out.to_vec::<f32>().unwrap(),
                &whole.out,
            );
        }
        eprintln!(
            "graph: rows {rows} {}: captured (hook = device copy of the summed y), 3 replays bitwise the whole block",
            if prefill { "prefill" } else { "decode" }
        );
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// What a rank's absent experts cost

mod ev {
    use core::ffi::c_void;
    extern "C" {
        pub fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
        pub fn cudaEventDestroy(e: *mut c_void) -> i32;
    }
}

/// Microseconds per block of a captured graph of `layers` consecutive blocks, replayed `reps` times (the median).
fn graph_us(st: &Stream, g: &Graph, layers: usize, reps: usize) -> f64 {
    for _ in 0..3 {
        g.launch(st).unwrap();
    }
    st.sync().unwrap();
    let mut v = Vec::new();
    for _ in 0..reps {
        let (mut a, mut b) = (core::ptr::null_mut::<c_void>(), core::ptr::null_mut::<c_void>());
        let mut ms = 0f32;
        unsafe {
            assert_eq!(ev::cudaEventCreateWithFlags(&mut a, 0), 0);
            assert_eq!(ev::cudaEventCreateWithFlags(&mut b, 0), 0);
            assert_eq!(ev::cudaEventRecord(a, st.raw()), 0);
        }
        g.launch(st).unwrap();
        unsafe {
            assert_eq!(ev::cudaEventRecord(b, st.raw()), 0);
        }
        st.sync().unwrap();
        unsafe {
            assert_eq!(ev::cudaEventElapsedTime(&mut ms, a, b), 0);
            ev::cudaEventDestroy(a);
            ev::cudaEventDestroy(b);
        }
        v.push(ms as f64 * 1000.0 / layers as f64);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Prints the GPU time of a block with half its experts absent (the TP rank's block, no-op hook) against the whole block, in a graph of
/// 10 consecutive blocks. No assertion on time. Natural routing, 4-bit experts.
#[test]
fn what_the_absent_experts_cost_per_block() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let routing = Routing::Natural;
    let trio = build_trio(0x7a12, routing, &controlled_present(routing), &[]);
    let mut sc = scratches();
    eprintln!("per block, microseconds, median of 9 graph replays of 10 blocks (natural routing, 4-bit experts):");
    for &rows in &[1usize, 5, 22, 64] {
        let h = DevBuf::from_slice(&make_h(routing, rows, 9 + rows as u64)).unwrap();
        let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
        let layers = 10;
        let mut times = Vec::new();
        for (name, moe, tp) in [("whole forward_in", &trio.whole, false), ("rank0 forward_in_tp(no-op)", &trio.r0, true)] {
            let g = {
                let s = &mut sc.dec;
                let mut hook = |_: &mut DevBuf, _: usize, _: &Stream| -> Result<(), CudaError> { Ok(()) };
                Graph::capture(&st, || {
                    for _ in 0..layers {
                        if tp {
                            moe.forward_in_tp(s, &h, rows, &mut out, &st, &mut hook)?;
                        } else {
                            moe.forward_in(s, &h, rows, &mut out, &st)?;
                        }
                    }
                    Ok::<(), CudaError>(())
                })
                .unwrap_or_else(|e| panic!("capture: {e}"))
            };
            times.push((name, graph_us(&st, &g, layers, 9)));
        }
        eprintln!("  rows {rows:>3}: {} {:.1} us, {} {:.1} us", times[0].0, times[0].1, times[1].0, times[1].1);
    }
    // the large-row prefill path: a rank's 256 absent experts are its "slow" class (zeros by gb_nmp_slow)
    eprintln!("prefill path, per block, microseconds, median of 9 graph replays of 5 blocks:");
    for &rows in &[64usize, 128, 256] {
        let h = DevBuf::from_slice(&make_h(routing, rows, 19 + rows as u64)).unwrap();
        let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
        let layers = 5;
        let mut times = Vec::new();
        for (name, moe, tp) in
            [("whole forward_prefill_in", &trio.whole, false), ("rank0 forward_prefill_in_tp(no-op)", &trio.r0, true)]
        {
            let g = {
                let s = &mut sc.pre;
                let mut hook = |_: &mut DevBuf, _: usize, _: &Stream| -> Result<(), CudaError> { Ok(()) };
                Graph::capture(&st, || {
                    for _ in 0..layers {
                        if tp {
                            moe.forward_prefill_in_tp(s, &h, rows, &mut out, &st, &mut hook)?;
                        } else {
                            moe.forward_prefill_in(s, &h, rows, &mut out, &st)?;
                        }
                    }
                    Ok::<(), CudaError>(())
                })
                .unwrap_or_else(|e| panic!("capture: {e}"))
            };
            times.push((name, graph_us(&st, &g, layers, 9)));
        }
        eprintln!("  rows {rows:>3}: {} {:.1} us, {} {:.1} us", times[0].0, times[0].1, times[1].0, times[1].1);
    }
}

#[test]
fn the_scatter_costs_microseconds() {
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    eprintln!("scatter_cols_f32 per call, microseconds (graph of 20 calls, median of 9):");
    for &(rows, nl, nt) in &[
        (1usize, 9216usize, 18560usize),
        (5, 9344, 18560),
        (5, 2048, 4096),
        (5, 65536, 131072),
        (64, 9344, 18560),
        (64, 65536, 131072),
    ] {
        let src = DevBuf::from_slice(&rand_bits(1, rows * nl)).unwrap();
        let mut dst = DevBuf::zeroed(rows * nt * 4).unwrap();
        let g = Graph::capture(&st, || {
            for _ in 0..20 {
                scatter_cols_f32(&src, &mut dst, rows, nl, nt, nt - nl, &st)?;
            }
            Ok::<(), CudaError>(())
        })
        .unwrap_or_else(|e| panic!("capture: {e}"));
        eprintln!("  rows {rows:>3} {nl:>6} of {nt:>6}: {:.1} us", graph_us(&st, &g, 20, 9));
    }
}
