//! The large-row prefill path of the Nemotron 3.5 Super VL latent-MoE block (`csrc/nemotron_moe_prefill.cu`, package P1) against the
//! decode path (`NemotronMoe::forward_in`, which `tests/nemotron_moe.rs` and `tests/nemotron_moe_perf.rs` gate against the CPU oracle and
//! the original kernels) and against a CPU block reference.
//!
//! Gates (all bitwise unless said): a row's output, router logits, picks and weights are the decode path's whether it runs in a 1-row, a
//! 64-row, a 128-row or a 256-row call, across chunk boundaries; the scratch is poisoned with a NaN pattern before every run; the block
//! matches the CPU reference at the tolerance of `tests/nemotron_moe.rs` (1e-3); the pack's layers 1 (512 4-bit experts) and 26 (bf16
//! expert pairs among them) agree with the decode path; so do layers whose dense projections (fc1, fc2, the shared expert's) are bf16, up
//! to the MTP head's all-bf16 block. Timing (ignored, `bench` in the names): rows per second and GB/s per layer.
//!
//!     cargo test -p gb-cuda --release --test nemotron_moe_prefill -- --test-threads=1 --nocapture
//!     REXL3_NEMO_OMNI_DIR=<pack> cargo test -p gb-cuda --release --test nemotron_moe_prefill -- --ignored --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::exl3_synth::synth_group;
use common::Rng;
use core::ffi::c_void;
use gb_cuda::nemotron_moe::{MoeScratch, Proj, Weights, EXPERTS, EXPERT_INTER, HIDDEN, LATENT, SHARED_INTER, TOPK};
use gb_cuda::nemotron_moe_prefill::{PrefillMoe, PrefillScratch};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::{scan, Checkpoint, LoadedGroup};
use gb_exl3::{codebook, forward, Codebook};
use std::collections::BTreeSet;
use std::path::PathBuf;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

const SCALE: f32 = 5.0;
/// The achievable DRAM bandwidth of the GB10 the floors are taken at (about 273 GB/s peak; PERF-MoE's floor).
const FLOOR_GBPS: f64 = 240.0;

// ---------------------------------------------------------------------------------------------------------------------------------
// Layers

fn bf16_bits(r: &mut Rng, n: usize) -> Vec<u16> {
    (0..n).map(|_| (r.uniform(-0.05, 0.05).to_bits() >> 16) as u16).collect()
}

/// `n` bf16 weights in (-0.05, 0.05), four to a draw (the all-bf16 layers have 2.8 G of them).
fn bf16_bits_fast(r: &mut Rng, n: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(n + 3);
    while out.len() < n {
        let w = r.next_u64();
        for j in 0..4 {
            let u = ((w >> (16 * j)) & 0xffff) as f32 / 65536.0;
            out.push(((-0.05 + 0.1 * u).to_bits() >> 16) as u16);
        }
    }
    out.truncate(n);
    out
}

/// The host side of a synthetic layer (for the CPU reference): groups and bf16 weights of the experts present.
struct Host {
    gate: Vec<f32>,
    bias: Vec<f32>,
    table: Vec<u16>,
    fc1: DenseRef,
    fc2: DenseRef,
    shared_up: DenseRef,
    shared_down: DenseRef,
    experts: std::collections::BTreeMap<usize, ExpertRef>,
}

enum ExpertRef {
    Exl3(Box<LoadedGroup>, Box<LoadedGroup>),
    Bf16(Vec<f32>, Vec<f32>),
}

/// A dense projection of the host copy: 4-bit EXL3, or bf16 `[n, k]` as floats.
enum DenseRef {
    Exl3(LoadedGroup),
    Bf16(Vec<f32>),
}

fn bf16_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
}

/// A synthetic layer: a random gate and bias, EXL3 dense projections, and the experts in `present` (those also in `bf16` are bf16, the
/// rest EXL3, the others absent). Every expert has its own random trellis. Returns the weights and the host copy.
fn synthetic_layer(seed: u64, present: &[usize], bf16: &[usize]) -> (Weights, Host) {
    synthetic_layer_dense(seed, present, bf16, [false; 4])
}

/// `synthetic_layer` with the dense projections `dense_bf16` (fc1, fc2, shared up, shared down) bf16 instead of 4-bit.
fn synthetic_layer_dense(seed: u64, present: &[usize], bf16: &[usize], dense_bf16: [bool; 4]) -> (Weights, Host) {
    let mut r = Rng(seed);
    let gate_bits = bf16_bits(&mut r, EXPERTS * HIDDEN);
    let bias: Vec<f32> = r.vec(EXPERTS, -0.1, 0.1);
    let mut dr = Rng(seed ^ 0xDE45E);
    let dims = [(HIDDEN, LATENT), (LATENT, HIDDEN), (HIDDEN, SHARED_INTER), (SHARED_INTER, HIDDEN)];
    let mut dense_w: Vec<Proj> = Vec::new();
    let mut dense_h: Vec<DenseRef> = Vec::new();
    for (i, &(k, n)) in dims.iter().enumerate() {
        if dense_bf16[i] {
            let bits = bf16_bits_fast(&mut dr, k * n);
            dense_w.push(Proj::bf16(k, n, &bits).unwrap());
            dense_h.push(DenseRef::Bf16(bf16_f32(&bits)));
        } else {
            let g = synth_group(k, n, 4, Codebook::Mul1, seed + 1 + i as u64);
            dense_w.push(Proj::exl3_from_group(&g).unwrap());
            dense_h.push(DenseRef::Exl3(g));
        }
    }
    let mut up: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    let mut down: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    let mut map = std::collections::BTreeMap::new();
    for &e in present {
        if bf16.contains(&e) {
            let uw = bf16_bits_fast(&mut r, EXPERT_INTER * LATENT);
            let dw = bf16_bits_fast(&mut r, LATENT * EXPERT_INTER);
            up[e] = Proj::bf16(LATENT, EXPERT_INTER, &uw).unwrap();
            down[e] = Proj::bf16(EXPERT_INTER, LATENT, &dw).unwrap();
            map.insert(e, ExpertRef::Bf16(bf16_f32(&uw), bf16_f32(&dw)));
        } else {
            let ug = synth_group(LATENT, EXPERT_INTER, 4, Codebook::Mul1, seed + 1000 + 2 * e as u64);
            let dg = synth_group(EXPERT_INTER, LATENT, 4, Codebook::Mul1, seed + 1001 + 2 * e as u64);
            up[e] = Proj::exl3_from_group(&ug).unwrap();
            down[e] = Proj::exl3_from_group(&dg).unwrap();
            map.insert(e, ExpertRef::Exl3(Box::new(ug), Box::new(dg)));
        }
    }
    let mut dw = dense_w.into_iter();
    let (fc1_w, fc2_w, su_w, sd_w) = (dw.next().unwrap(), dw.next().unwrap(), dw.next().unwrap(), dw.next().unwrap());
    let mut dh = dense_h.into_iter();
    let (fc1, fc2, shared_up, shared_down) = (dh.next().unwrap(), dh.next().unwrap(), dh.next().unwrap(), dh.next().unwrap());
    let w = Weights {
        gate: DevBuf::from_slice(&gate_bits).unwrap(),
        gate_dtype: gb_cuda::nemotron_moe::GateDtype::Bf16,
        router_bias: DevBuf::from_slice(&bias).unwrap(),
        scale: SCALE,
        fc1: fc1_w,
        fc2: fc2_w,
        shared_up: su_w,
        shared_down: sd_w,
        up,
        down,
    };
    let host = Host {
        gate: bf16_f32(&gate_bits),
        bias,
        table: codebook(Codebook::Mul1),
        fc1,
        fc2,
        shared_up,
        shared_down,
        experts: map,
    };
    (w, host)
}

/// The synthetic cases: (name, seed, present experts, bf16 experts). `sparse14`: 14 of 512 present, one bf16, the rest absent (a pair
/// routed to an absent expert is zeros); `all512`: all present, two bf16; `all512_exl3`: all present and 4-bit.
fn synthetic_cases() -> Vec<(&'static str, u64, Vec<usize>, Vec<usize>)> {
    let sparse: Vec<usize> = (0..EXPERTS).step_by(37).collect();
    let bf = sparse[3];
    vec![
        ("sparse14", 61, sparse, vec![bf]),
        ("all512", 101, (0..EXPERTS).collect(), vec![5, 300]),
        ("all512_exl3", 131, (0..EXPERTS).collect(), vec![]),
    ]
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Running both paths

/// What a block call produces: the output rows, and the router's logits, picks and weights.
struct Run {
    out: Vec<f32>,
    logits: Vec<f32>,
    pick: Vec<i32>,
    wts: Vec<f32>,
}

/// The decode path on `rows` rows in calls of at most 64 rows (what a 64-row-slice engine runs).
fn run_decode(moe: &PrefillMoe, h: &[f32], rows: usize, st: &Stream) -> Run {
    let mut s = MoeScratch::new(64).unwrap();
    let mut r = Run { out: Vec::new(), logits: Vec::new(), pick: Vec::new(), wts: Vec::new() };
    let mut start = 0;
    while start < rows {
        let n = (rows - start).min(64);
        let hd = DevBuf::from_slice(&h[start * HIDDEN..(start + n) * HIDDEN]).unwrap();
        let mut out = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); n * HIDDEN]).unwrap();
        s.poison(st).unwrap();
        moe.forward_in(&mut s, &hd, n, &mut out, st).unwrap();
        st.sync().unwrap();
        r.out.extend(out.to_vec::<f32>().unwrap());
        r.logits.extend(s.download_logits(n, st).unwrap());
        let (p, w) = s.download_route(n, st).unwrap();
        r.pick.extend(p);
        r.wts.extend(w);
        start += n;
    }
    r
}

/// The prefill path on `rows` rows as one call, on a scratch poisoned beforehand.
fn run_prefill(moe: &PrefillMoe, s: &mut PrefillScratch, h: &[f32], rows: usize, st: &Stream) -> Run {
    let hd = DevBuf::from_slice(&h[..rows * HIDDEN]).unwrap();
    let mut out = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * HIDDEN]).unwrap();
    s.poison(st).unwrap();
    moe.forward_prefill_in(s, &hd, rows, &mut out, st).unwrap();
    st.sync().unwrap();
    let o = out.to_vec::<f32>().unwrap();
    let logits = s.download_logits(rows, st).unwrap();
    let (pick, wts) = s.download_route(rows, st).unwrap();
    Run { out: o, logits, pick, wts }
}

fn assert_same(what: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    let bad: Vec<usize> = (0..a.len()).filter(|&i| a[i].to_bits() != b[i].to_bits()).collect();
    assert!(
        bad.is_empty(),
        "{what}: {} of {} values differ in bits, first at {} ({:e} against {:e})",
        bad.len(),
        a.len(),
        bad[0],
        a[bad[0]],
        b[bad[0]]
    );
}

fn assert_runs_equal(label: &str, a: &Run, b: &Run) {
    assert!(a.out.iter().any(|v| *v != 0.0 && v.is_finite()), "{label}: the output is all zero or not finite");
    assert_same(&format!("{label}: output"), &a.out, &b.out);
    assert_same(&format!("{label}: router logits"), &a.logits, &b.logits);
    assert_same(&format!("{label}: router weights"), &a.wts, &b.wts);
    assert_eq!(a.pick, b.pick, "{label}: router picks");
}

fn layer(seed: u64, present: &[usize], bf16: &[usize]) -> (PrefillMoe, Host) {
    layer_dense(seed, present, bf16, [false; 4])
}

fn layer_dense(seed: u64, present: &[usize], bf16: &[usize], dense_bf16: [bool; 4]) -> (PrefillMoe, Host) {
    let (w, h) = synthetic_layer_dense(seed, present, bf16, dense_bf16);
    let moe = PrefillMoe::tables(w).unwrap();
    assert!(moe.supports_prefill());
    (moe, h)
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Gate 1: bits

/// Output, router logits, picks and weights of the prefill call equal the decode path's at 1 to 256 rows, on three synthetic layers (a
/// sparse one with absent and bf16 experts, a full one with two bf16 experts, a full 4-bit one).
#[test]
fn prefill_bits_equal_the_decode_path_on_synthetic_layers() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut s = PrefillScratch::new(512).unwrap();
    for (name, seed, present, bf16) in synthetic_cases() {
        let (moe, _) = layer(seed, &present, &bf16);
        eprintln!("{name}: {} slow experts", moe.slow_experts());
        // 300, 512: the default wide slice since 2026-10-15 (REXL3_NEMO_PREFILL_ROWS_MAX, lane S)
        for &rows in &[1usize, 2, 7, 33, 64, 65, 100, 128, 200, 256, 300, 512] {
            let h = Rng(0xB17 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
            let d = run_decode(&moe, &h, rows, &st);
            let p = run_prefill(&moe, &mut s, &h, rows, &st);
            assert_runs_equal(&format!("{name}, {rows} rows"), &p, &d);
            eprintln!("{name}, {rows} rows: output, logits, picks and weights are bitwise the decode path's");
        }
    }
    // the launch alternatives give the same bits: the experts in two launches by class, one launch of items of up to 32 rows or of up to
    // 16, the shared expert in line instead of on the side stream
    use gb_cuda::nemotron_moe_prefill::{set_bf16_kernel, set_interleave, set_schedule, set_side_stream};
    let (name, seed, present, bf16) = synthetic_cases().remove(1);
    let (moe, _) = layer(seed, &present, &bf16);
    for &rows in &[65usize, 200, 256] {
        let h = Rng(0xB17 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
        let d = run_decode(&moe, &h, rows, &st);
        // (schedule, shared expert on the side stream, bf16 kernel)
        for (schedule, side, bf16k) in [
            (0u8, true, 0u8),
            (1, true, 0),
            (2, true, 0),
            (1, false, 0),
            (0, false, 0),
            (2, false, 0),
            (1, true, 1),
            (2, true, 1),
            (0, false, 1),
        ] {
            set_schedule(schedule);
            set_side_stream(side);
            set_bf16_kernel(bf16k);
            let p = run_prefill(&moe, &mut s, &h, rows, &st);
            assert_runs_equal(
                &format!("{name}, {rows} rows, schedule {schedule}, side stream {side}, bf16 kernel {bf16k}"),
                &p,
                &d,
            );
        }
    }
    set_interleave(true);
    set_side_stream(true);
    set_bf16_kernel(0);
}

/// An F16 router gate (as stored, decision D1 of v0.1.2): the prefill call's output, logits, picks and weights equal the
/// decode path's at 1 to 512 rows (the full layer with two bf16 experts).
#[test]
fn prefill_bits_equal_the_decode_path_with_an_f32_gate() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut s = PrefillScratch::new(512).unwrap();
    let (name, seed, present, bf16) = synthetic_cases().remove(1);
    let (mut w, _) = synthetic_layer(seed, &present, &bf16);
    let mut r = Rng(0xF32F);
    let bits: Vec<u16> = (0..EXPERTS * HIDDEN).map(|_| gb_exl3::f16::f64_to_f16(r.uniform(-0.05, 0.05) as f64)).collect();
    w.gate = DevBuf::from_slice(&bits).unwrap();
    w.gate_dtype = gb_cuda::nemotron_moe::GateDtype::F16;
    let moe = PrefillMoe::tables(w).unwrap();
    assert!(moe.supports_prefill());
    assert_eq!(moe.gate_dtype(), gb_cuda::nemotron_moe::GateDtype::F16);
    for &rows in &[1usize, 13, 64, 65, 200, 512] {
        let h = Rng(0xB17 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
        let d = run_decode(&moe, &h, rows, &st);
        let p = run_prefill(&moe, &mut s, &h, rows, &st);
        assert_runs_equal(&format!("{name}, F16 gate, {rows} rows"), &p, &d);
    }
    eprintln!("{name}, f16 gate widened: prefill == decode path bitwise at 1, 13, 64, 65, 200, 512 rows");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// MixedK (v0.1.2 lane K, P1-5): routed experts and dense projections at every integer width, fp16 dense projections

/// A MixedK synthetic layer (as tests/nemotron_moe_perf.rs's): each present expert's up and down at `up_bits(e)`, `down_bits(e)` (Mul1,
/// 2 to 8 bits; those in `bf16` bf16), the dense projections at `dense[i]` bits (0: fp16 `[N, K]`).
fn mixed_layer(
    seed: u64,
    present: &[usize],
    bf16: &[usize],
    up_bits: impl Fn(usize) -> usize,
    down_bits: impl Fn(usize) -> usize,
    dense: [usize; 4],
) -> Weights {
    let mut r = Rng(seed);
    let gate_bits = bf16_bits(&mut r, EXPERTS * HIDDEN);
    let bias: Vec<f32> = r.vec(EXPERTS, -0.1, 0.1);
    let g = |k, n, b, s| Proj::exl3_from_group(&synth_group(k, n, b, Codebook::Mul1, s)).unwrap();
    let mut up: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    let mut down: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    for &e in present {
        if bf16.contains(&e) {
            up[e] = Proj::bf16(LATENT, EXPERT_INTER, &bf16_bits_fast(&mut r, EXPERT_INTER * LATENT)).unwrap();
            down[e] = Proj::bf16(EXPERT_INTER, LATENT, &bf16_bits_fast(&mut r, LATENT * EXPERT_INTER)).unwrap();
        } else {
            up[e] = g(LATENT, EXPERT_INTER, up_bits(e), seed + 1000 + 2 * e as u64);
            down[e] = g(EXPERT_INTER, LATENT, down_bits(e), seed + 1001 + 2 * e as u64);
        }
    }
    let shapes = [(HIDDEN, LATENT), (LATENT, HIDDEN), (HIDDEN, SHARED_INTER), (SHARED_INTER, HIDDEN)];
    let mut d = shapes.iter().enumerate().map(|(i, &(k, n))| {
        if dense[i] == 0 {
            let bits: Vec<u16> = (0..k * n).map(|_| gb_exl3::f16::f64_to_f16(r.uniform(-0.05, 0.05) as f64)).collect();
            Proj::f16(k, n, &bits).unwrap()
        } else {
            g(k, n, dense[i], seed + 1 + i as u64)
        }
    });
    let (fc1, fc2, shared_up, shared_down) = (d.next().unwrap(), d.next().unwrap(), d.next().unwrap(), d.next().unwrap());
    Weights {
        gate: DevBuf::from_slice(&gate_bits).unwrap(),
        gate_dtype: gb_cuda::nemotron_moe::GateDtype::Bf16,
        router_bias: DevBuf::from_slice(&bias).unwrap(),
        scale: SCALE,
        fc1,
        fc2,
        shared_up,
        shared_down,
        up,
        down,
    }
}

/// A width of {2, 3, 4, 5, 6} bits for expert `e` and stream `salt` (every (up, down) pair of widths occurs).
fn width_of(e: usize, salt: u64) -> usize {
    let h = (e as u64 ^ salt).wrapping_mul(0x9E3779B97F4A7C15) >> 59;
    [2, 3, 4, 5, 6][(h % 5) as usize]
}

/// The MixedK cases: his pack's shapes (all 512 experts at 2 to 6 bits, up and down apart; fp16 fc1 and fc2; shared expert 5 and 4
/// bits), the wide widths (7 and 8 bits; dense 8/6/7/6; two bf16 experts), a TP rank's half (absent experts) with 4-bit dense
/// projections, and a shared expert of 4 and 6 bits (a 4-bit up before an any-width down).
fn mixed_cases() -> Vec<(&'static str, Weights)> {
    let all: Vec<usize> = (0..EXPERTS).collect();
    let sparse: Vec<usize> = (0..EXPERTS).step_by(5).collect();
    let even: Vec<usize> = (0..EXPERTS).step_by(2).collect();
    vec![
        ("mixedk512 (his shapes)", mixed_layer(201, &all, &[], |e| width_of(e, 1), |e| width_of(e, 2), [0, 0, 5, 4])),
        (
            "wide (7, 8; dense 8/6/7/6; bf16)",
            mixed_layer(202, &sparse, &[sparse[3], sparse[40]], |e| [7, 8, 4][e % 3], |e| [8, 7, 5][e % 3], [8, 6, 7, 6]),
        ),
        ("tp half mixed, dense 4", mixed_layer(203, &even, &[], |e| width_of(e, 3), |e| width_of(e, 4), [4, 4, 4, 4])),
        ("shared 4 -> 6, fc1/fc2 fp16", mixed_layer(204, &all, &[], |e| width_of(e, 5), |e| width_of(e, 6), [0, 0, 4, 6])),
    ]
}

/// MixedK layers on the prefill path (`pf_gemmk` for the routed experts and the any-width dense projections, `pf_slow` / `pf_slow4` for
/// the fp16 ones): output, router logits, picks and weights are the decode path's (which tests/nemotron_moe_perf.rs gates bit for bit
/// against the reference launches) at 1 to 512 rows, on both bf16/fp16 kernels and with the shared expert in line.
#[test]
fn prefill_bits_equal_the_decode_path_on_mixedk_layers() {
    gpu!();
    use gb_cuda::nemotron_moe_prefill::{set_any_width_prefill, set_bf16_kernel, set_side_stream};
    set_any_width_prefill(true);
    let st = Stream::new().unwrap();
    let mut s = PrefillScratch::new(512).unwrap();
    for (name, w) in mixed_cases() {
        let moe = PrefillMoe::tables(w).unwrap();
        assert!(moe.supports_prefill(), "{name}: a MixedK layer is on the prefill path with the any-width switch on");
        for &rows in &[1usize, 2, 7, 16, 17, 33, 64, 65, 100, 128, 200, 256, 300, 512] {
            let h = Rng(0xB17 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
            let d = run_decode(&moe, &h, rows, &st);
            let p = run_prefill(&moe, &mut s, &h, rows, &st);
            assert_runs_equal(&format!("{name}, {rows} rows"), &p, &d);
        }
        for (side, bf16k) in [(false, 0u8), (true, 1), (false, 1)] {
            set_side_stream(side);
            set_bf16_kernel(bf16k);
            for &rows in &[65usize, 200] {
                let h = Rng(0xB17 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
                let d = run_decode(&moe, &h, rows, &st);
                let p = run_prefill(&moe, &mut s, &h, rows, &st);
                assert_runs_equal(&format!("{name}, {rows} rows, side stream {side}, bf16 kernel {bf16k}"), &p, &d);
            }
        }
        set_side_stream(true);
        set_bf16_kernel(0);
        eprintln!("{name}: prefill == decode path bitwise at 1 to 512 rows (and in line, pf_slow4)");
    }
}

/// `pf_gemmk` at 4 bits gives `pf_gemm4`'s bits: the 4-bit synthetic layers (with bf16 and absent experts among them) built as MixedK
/// tables (`set_force_any_width_prefill`: their fast experts on `pf_gemmk`, items of 16 rows) equal their default tables (`pf_gemm4`)
/// and the decode path, output, logits, picks and weights, at 1 to 512 rows.
#[test]
fn pf_gemmk_at_4_bits_gives_pf_gemm4s_bits() {
    gpu!();
    use gb_cuda::nemotron_moe_prefill::set_force_any_width_prefill;
    let st = Stream::new().unwrap();
    let mut s = PrefillScratch::new(512).unwrap();
    for (name, seed, present, bf16) in synthetic_cases() {
        let (moe, _) = layer(seed, &present, &bf16);
        set_force_any_width_prefill(true);
        let (wk, _) = synthetic_layer(seed, &present, &bf16);
        let any = PrefillMoe::tables(wk).unwrap();
        set_force_any_width_prefill(false);
        for &rows in &[1usize, 16, 17, 33, 65, 256, 512] {
            let h = Rng(0xB17 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
            let a = run_prefill(&moe, &mut s, &h, rows, &st);
            let b = run_prefill(&any, &mut s, &h, rows, &st);
            assert_runs_equal(&format!("{name}, {rows} rows: pf_gemmk vs pf_gemm4"), &b, &a);
            if rows <= 64 {
                assert_runs_equal(
                    &format!("{name}, {rows} rows: pf_gemmk vs the decode path"),
                    &b,
                    &run_decode(&moe, &h, rows, &st),
                );
            }
        }
        eprintln!("{name}: pf_gemmk at 4 bits == pf_gemm4 bitwise at 1 to 512 rows");
    }
}

/// Until the any-width switch is on (P1-7), a MixedK layer is not on the prefill path (`pf_slow` / `pf_zero_absent` would zero its
/// experts): built with the switch off, the same layers are not `supports_prefill`.
#[test]
fn a_mixedk_layer_is_off_the_prefill_path_until_the_any_width_switch() {
    gpu!();
    // a fresh process state is not guaranteed (another test may have turned it on): the check is on tables built with it off
    gb_cuda::nemotron_moe_prefill::set_any_width_prefill(false);
    for (name, w) in mixed_cases() {
        assert!(!PrefillMoe::tables(w).unwrap().supports_prefill(), "{name}");
    }
    gb_cuda::nemotron_moe_prefill::set_any_width_prefill(true);
}

/// The absent-expert fast path (`REXL3_NEMO_PREFILL_ABSENT_FAST`, on by default since 2026-10-15, `0` opts out; lane S): a layer whose slow experts are all absent (a TP rank's
/// non-owned experts) skips `pf_slow` and zeroes the absent pairs with one kernel; output, logits, picks and weights are bitwise the
/// decode path's and the path without the switch, at 1 to 512 rows, on a scratch poisoned before every call. A layer with a bf16 pair
/// keeps `pf_slow` (the switch must not change it either).
#[test]
fn the_absent_fast_path_has_the_bits_of_pf_slow() {
    gpu!();
    use gb_cuda::nemotron_moe_prefill::set_absent_fast;
    let st = Stream::new().unwrap();
    let mut s = PrefillScratch::new(512).unwrap();
    // a TP rank's table: half the experts present (4-bit), half absent; the sparse case with no bf16; and one with a bf16 pair
    let half: Vec<usize> = (0..EXPERTS).filter(|e| e % 2 == 1).collect();
    let sparse: Vec<usize> = (0..EXPERTS).step_by(37).collect();
    let cases: Vec<(&str, u64, Vec<usize>, Vec<usize>)> = vec![
        ("half_absent", 171, half.clone(), vec![]),
        ("sparse_absent", 61, sparse, vec![]),
        ("half_absent_bf16", 173, half, vec![3]),
    ];
    for (name, seed, present, bf16) in cases {
        let (moe, _) = layer(seed, &present, &bf16);
        for &rows in &[1usize, 65, 200, 256, 512] {
            let h = Rng(0xAB5 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
            set_absent_fast(false);
            let off = run_prefill(&moe, &mut s, &h, rows, &st);
            set_absent_fast(true);
            let on = run_prefill(&moe, &mut s, &h, rows, &st);
            assert_runs_equal(&format!("{name}, {rows} rows: switch on against off"), &on, &off);
            if rows <= 256 {
                let d = run_decode(&moe, &h, rows, &st);
                assert_runs_equal(&format!("{name}, {rows} rows: switch on against the decode path"), &on, &d);
            }
            eprintln!("{name}, {rows} rows: the absent fast path is bitwise pf_slow's and the decode path's");
        }
    }
    // back to the default (on)
    set_absent_fast(true);
}

/// A row has the same bits whatever call it runs in: rows of a 256-row prefill call equal the same rows in calls of 1, 64 and 128 rows
/// (the cuts are not on the 64-row grid), and equal the 1-row decode path.
#[test]
fn a_row_has_the_same_bits_in_any_call() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut s = PrefillScratch::new(256).unwrap();
    let (name, seed, present, bf16) = synthetic_cases().remove(1);
    let (moe, _) = layer(seed, &present, &bf16);
    let h = Rng(0x5EED).vec(256 * HIDDEN, -1.0, 1.0);
    let whole = run_prefill(&moe, &mut s, &h, 256, &st);
    // Every cut of the 256 rows into calls: (start, rows).
    let cuts: &[(usize, usize)] =
        &[(0, 64), (64, 64), (128, 128), (0, 1), (37, 1), (255, 1), (10, 117), (100, 100), (3, 250), (192, 64)];
    for &(start, rows) in cuts {
        let part = run_prefill(&moe, &mut s, &h[start * HIDDEN..(start + rows) * HIDDEN], rows, &st);
        assert_same(
            &format!("{name}: rows {start}..{} in a {rows}-row call: output", start + rows),
            &part.out,
            &whole.out[start * HIDDEN..(start + rows) * HIDDEN],
        );
        assert_same(
            &format!("{name}: rows {start}..: logits"),
            &part.logits,
            &whole.logits[start * EXPERTS..(start + rows) * EXPERTS],
        );
        assert_eq!(part.pick, whole.pick[start * TOPK..(start + rows) * TOPK], "{name}: picks of rows {start}..");
        assert_same(&format!("{name}: rows {start}..: weights"), &part.wts, &whole.wts[start * TOPK..(start + rows) * TOPK]);
    }
    // The decode path's 1-row calls (rows 0, 63, 64, 200) are the same bits as the rows of the 256-row call.
    for &row in &[0usize, 63, 64, 200] {
        let d = run_decode(&moe, &h[row * HIDDEN..(row + 1) * HIDDEN], 1, &st);
        assert_same(&format!("{name}: row {row}, decode 1-row call"), &d.out, &whole.out[row * HIDDEN..(row + 1) * HIDDEN]);
    }
    eprintln!("{name}: {} cuts of 256 rows and 4 one-row decode calls give the 256-row call's bits", cuts.len());
}

/// Rows that pick the same experts (the same 4 rows tiled to 256): each routed expert gets 64 to 256 rows, past the 32 rows of a block, so
/// an expert's rows are split in several blocks. The bits are the decode path's, and every tile of a row is the same bits.
#[test]
fn an_expert_with_many_rows_is_split_in_blocks_without_changing_bits() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut s = PrefillScratch::new(256).unwrap();
    let (name, seed, present, bf16) = synthetic_cases().remove(1);
    let (moe, _) = layer(seed, &present, &bf16);
    let base = Rng(0x7117).vec(4 * HIDDEN, -1.0, 1.0);
    let h: Vec<f32> = (0..256).flat_map(|r| base[(r % 4) * HIDDEN..(r % 4 + 1) * HIDDEN].to_vec()).collect();
    let p = run_prefill(&moe, &mut s, &h, 256, &st);
    let d = run_decode(&moe, &h, 256, &st);
    assert_runs_equal(&format!("{name}, 4 rows tiled to 256"), &p, &d);
    for r in 0..256 {
        assert_same(
            &format!("tile row {r}"),
            &p.out[r * HIDDEN..(r + 1) * HIDDEN],
            &p.out[(r % 4) * HIDDEN..(r % 4 + 1) * HIDDEN],
        );
    }
    let items = s.download_items(256, &st).unwrap();
    let max_rows = items.iter().flatten().map(|it| it[2]).max().unwrap();
    let experts: BTreeSet<i32> = p.pick.iter().copied().collect();
    eprintln!(
        "{name}: 4 distinct rows x 64: {} experts, {} items (small {}, large {}, slow {}), at most {max_rows} rows an item",
        experts.len(),
        items.iter().map(Vec::len).sum::<usize>(),
        items[0].len(),
        items[1].len(),
        items[2].len()
    );
    assert!(max_rows <= 32, "an item has more than 32 rows");
    // every pair is in exactly one item, and an expert with c rows is at least ceil(c / 32) items
    let mut rows_of: std::collections::BTreeMap<i32, (usize, usize)> = std::collections::BTreeMap::new();
    for it in items.iter().flatten() {
        let e = rows_of.entry(it[0]).or_default();
        e.0 += it[2] as usize;
        e.1 += 1;
    }
    assert_eq!(rows_of.values().map(|v| v.0).sum::<usize>(), 256 * TOPK, "the items do not cover every pair once");
    for (e, (rows, n)) in &rows_of {
        assert!(*n >= rows.div_ceil(32), "expert {e}: {rows} rows in {n} items");
    }
    assert!(rows_of.values().any(|v| v.0 >= 64 && v.1 >= 2), "no expert had 64 or more rows in several items");
}

/// Dense projections in bf16: any subset of fc1, fc2 and the shared expert's up and down (the repaired pack has bf16 shared
/// projections; the MTP head has all four), with 4-bit and bf16 routed experts mixed. Bits equal the decode path's, the scratch poisoned.
#[test]
fn bf16_dense_projections_equal_the_decode_path() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut s = PrefillScratch::new(256).unwrap();
    let sparse: Vec<usize> = (0..EXPERTS).step_by(37).collect();
    let mixed = vec![sparse[3], sparse[8]];
    let cases: [(&str, u64, [bool; 4]); 5] = [
        ("shared up and down bf16", 301, [false, false, true, true]),
        ("shared up bf16, down 4-bit (rotated in place)", 311, [false, false, true, false]),
        ("shared up 4-bit, down bf16 (plain output)", 321, [false, false, false, true]),
        ("fc1 and fc2 bf16", 331, [true, true, false, false]),
        ("all four bf16", 341, [true; 4]),
    ];
    use gb_cuda::nemotron_moe_prefill::set_bf16_kernel;
    for (name, seed, dense_bf16) in cases {
        let (moe, _) = layer_dense(seed, &sparse, &mixed, dense_bf16);
        for &rows in &[1usize, 65, 100, 256] {
            let h = Rng(0xD3 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
            let d = run_decode(&moe, &h, rows, &st);
            for bf16k in [0u8, 1] {
                set_bf16_kernel(bf16k);
                let p = run_prefill(&moe, &mut s, &h, rows, &st);
                assert_runs_equal(&format!("{name}, {rows} rows, bf16 kernel {bf16k}"), &p, &d);
            }
        }
        eprintln!("{name}: 1, 65, 100 and 256 rows are bitwise the decode path's, with both bf16 kernels");
    }
    set_bf16_kernel(0);
    // the side stream off: the shared expert in line gives the same bits
    use gb_cuda::nemotron_moe_prefill::set_side_stream;
    let (moe, _) = layer_dense(341, &sparse, &mixed, [true; 4]);
    let h = Rng(0xD3 + 200).vec(200 * HIDDEN, -1.0, 1.0);
    let d = run_decode(&moe, &h, 200, &st);
    set_side_stream(false);
    let p = run_prefill(&moe, &mut s, &h, 200, &st);
    set_side_stream(true);
    assert_runs_equal("all four bf16, 200 rows, shared expert in line", &p, &d);
}

/// The MTP head's block: all 512 experts bf16 and the four dense projections bf16 (5.6 GB). 65 to 256 rows on the large-row path equal
/// the decode path in calls of 64 rows, and a row has the same bits in any call.
#[test]
fn an_all_bf16_block_runs_on_the_prefill_path_with_the_decode_paths_bits() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut s = PrefillScratch::new(256).unwrap();
    let all: Vec<usize> = (0..EXPERTS).collect();
    let (moe, _) = layer_dense(401, &all, &all, [true; 4]);
    assert_eq!(moe.slow_experts(), EXPERTS);
    let h = Rng(0xAB1).vec(256 * HIDDEN, -1.0, 1.0);
    use gb_cuda::nemotron_moe_prefill::set_bf16_kernel;
    let whole = run_prefill(&moe, &mut s, &h, 256, &st);
    for &rows in &[1usize, 65, 128, 256] {
        let d = run_decode(&moe, &h[..rows * HIDDEN], rows, &st);
        for bf16k in [0u8, 1] {
            set_bf16_kernel(bf16k);
            let p = run_prefill(&moe, &mut s, &h[..rows * HIDDEN], rows, &st);
            assert_runs_equal(&format!("all bf16, {rows} rows, bf16 kernel {bf16k}"), &p, &d);
        }
        eprintln!("all bf16, {rows} rows: output, logits, picks and weights are bitwise the decode path's (both bf16 kernels)");
    }
    // rows 100 .. 200 as a call of their own, and the 256-row call of the other kernel
    for bf16k in [0u8, 1] {
        set_bf16_kernel(bf16k);
        let part = run_prefill(&moe, &mut s, &h[100 * HIDDEN..200 * HIDDEN], 100, &st);
        assert_same("rows 100..200 of a 256-row call against a 100-row call", &part.out, &whole.out[100 * HIDDEN..200 * HIDDEN]);
        let all = run_prefill(&moe, &mut s, &h, 256, &st);
        assert_same("a 256-row call of each bf16 kernel", &all.out, &whole.out);
    }
    set_bf16_kernel(0);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Gate 2: the CPU block reference

fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "length {} against {}", got.len(), want.len());
    let scale = want.iter().fold(0f64, |m, &v| m.max((v as f64).abs()));
    let diff = got.iter().zip(want).fold(0f64, |m, (&g, &w)| m.max(((g - w) as f64).abs()));
    diff / scale
}

fn logits_cpu(h: &[f32], gate: &[f32]) -> Vec<f32> {
    (0..EXPERTS).map(|j| h.iter().zip(&gate[j * HIDDEN..(j + 1) * HIDDEN]).map(|(a, b)| a * b).sum()).collect()
}

/// moe_ref's route: top-k of sigmoid + bias (ties to the lower index), weights = unbiased sigmoid over the selected set, times `scale`.
fn route_cpu(logits: &[f32], bias: &[f32], scale: f32) -> (Vec<usize>, Vec<f32>, f32) {
    let sig: Vec<f32> = logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    let sel: Vec<f32> = sig.iter().zip(bias).map(|(s, b)| s + b).collect();
    let mut order: Vec<usize> = (0..sel.len()).collect();
    order.sort_by(|&a, &b| sel[b].partial_cmp(&sel[a]).unwrap().then(a.cmp(&b)));
    let mut sorted = sel.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let gap = sorted[TOPK - 1] - sorted[TOPK];
    order.truncate(TOPK);
    let denom = order.iter().map(|&i| sig[i]).sum::<f32>() + 1e-20;
    let w = order.iter().map(|&i| sig[i] / denom * scale).collect();
    (order, w, gap)
}

fn exl3_cpu(g: &LoadedGroup, table: &[u16], x: &[f32], rows: usize) -> Vec<f32> {
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    forward(&x64, rows, &g.trellis, g.kt, g.nt, g.group.bits, table, &g.suh, &g.svh, g.bias.as_deref())
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

/// A dense projection on the host: `x` is `[rows, k]`.
fn dense_cpu(d: &DenseRef, table: &[u16], x: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    match d {
        DenseRef::Exl3(g) => exl3_cpu(g, table, x, rows),
        DenseRef::Bf16(w) => bf16_cpu(w, k, n, x, rows),
    }
}

fn bf16_cpu(w: &[f32], k: usize, n: usize, x: &[f32], rows: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * n];
    for r in 0..rows {
        for j in 0..n {
            y[r * n + j] = x[r * k..(r + 1) * k].iter().zip(&w[j * k..(j + 1) * k]).map(|(a, b)| a * b).sum();
        }
    }
    y
}

fn relu2(v: &mut [f32]) {
    for x in v.iter_mut() {
        let r = x.max(0.0);
        *x = r * r;
    }
}

/// The block on the host (as moe_ref's `latent_moe`, the routed sum in slot order): `rows` rows of `h` with the routes given.
fn block_cpu(s: &Host, h: &[f32], rows: usize, picks: &[Vec<usize>], wts: &[Vec<f32>]) -> Vec<f32> {
    let x = dense_cpu(&s.fc1, &s.table, h, rows, HIDDEN, LATENT);
    let mut y_pair = vec![vec![0f32; LATENT]; rows * TOPK];
    let used: BTreeSet<usize> = picks.iter().flatten().copied().collect();
    for e in used {
        let pairs: Vec<(usize, usize)> =
            (0..rows).flat_map(|t| (0..TOPK).map(move |sl| (t, sl))).filter(|&(t, sl)| picks[t][sl] == e).collect();
        let mut batch = Vec::with_capacity(pairs.len() * LATENT);
        for &(t, _) in &pairs {
            batch.extend_from_slice(&x[t * LATENT..(t + 1) * LATENT]);
        }
        let y = match &s.experts[&e] {
            ExpertRef::Exl3(ug, dg) => {
                let mut a = exl3_cpu(ug, &s.table, &batch, pairs.len());
                relu2(&mut a);
                exl3_cpu(dg, &s.table, &a, pairs.len())
            }
            ExpertRef::Bf16(uw, dw) => {
                let mut a = bf16_cpu(uw, LATENT, EXPERT_INTER, &batch, pairs.len());
                relu2(&mut a);
                bf16_cpu(dw, EXPERT_INTER, LATENT, &a, pairs.len())
            }
        };
        for (i, &(t, sl)) in pairs.iter().enumerate() {
            y_pair[t * TOPK + sl].copy_from_slice(&y[i * LATENT..(i + 1) * LATENT]);
        }
    }
    let mut routed = vec![0f32; rows * LATENT];
    for t in 0..rows {
        for sl in 0..TOPK {
            let w = wts[t][sl];
            for (d, &v) in routed[t * LATENT..(t + 1) * LATENT].iter_mut().zip(&y_pair[t * TOPK + sl]) {
                *d += w * v;
            }
        }
    }
    let routed_out = dense_cpu(&s.fc2, &s.table, &routed, rows, LATENT, HIDDEN);
    let mut sa = dense_cpu(&s.shared_up, &s.table, h, rows, HIDDEN, SHARED_INTER);
    relu2(&mut sa);
    let shared_out = dense_cpu(&s.shared_down, &s.table, &sa, rows, SHARED_INTER, HIDDEN);
    routed_out.iter().zip(&shared_out).map(|(a, b)| a + b).collect()
}

/// The full block, 4 distinct rows tiled to 128 (so each used expert has 32 to 128 rows: several blocks, more than one class), with a
/// bf16 expert among EXL3 experts, against the CPU reference at 1e-3; every row of the tiled call has the bits of its base row.
#[test]
fn prefill_matches_the_cpu_reference_with_a_bf16_expert() {
    cpu_reference_case([false; 4], false);
}

/// The same with the head's shape: every used expert bf16 and the four dense projections bf16.
#[test]
fn prefill_matches_the_cpu_reference_on_an_all_bf16_block() {
    for bf16k in [0u8, 1] {
        gb_cuda::nemotron_moe_prefill::set_bf16_kernel(bf16k);
        cpu_reference_case([true; 4], true);
    }
    gb_cuda::nemotron_moe_prefill::set_bf16_kernel(0);
}

fn cpu_reference_case(dense_bf16: [bool; 4], all_bf16: bool) {
    gpu!();
    let st = Stream::new().unwrap();
    let mut r = Rng(31);
    let base = r.vec(4 * HIDDEN, -1.0, 1.0);
    // The routes on these rows, from the layer's gate (the gate and bias come first in the seeded stream, so a layer with no experts has
    // the same ones), so that the loaded experts are the ones the block uses.
    let (_, probe) = synthetic_layer(41, &[], &[]);
    let mut picks: Vec<Vec<usize>> = Vec::new();
    let mut wts: Vec<Vec<f32>> = Vec::new();
    for t in 0..4 {
        let (idx, w, gap) = route_cpu(&logits_cpu(&base[t * HIDDEN..(t + 1) * HIDDEN], &probe.gate), &probe.bias, SCALE);
        assert!(gap > 1e-5, "row {t} has a near-tie at the boundary: pick other rows");
        picks.push(idx);
        wts.push(w);
    }
    let used: Vec<usize> = picks.iter().flatten().copied().collect::<BTreeSet<usize>>().into_iter().collect();
    let bf16 = used[used.len() / 2];
    let bf16_set = if all_bf16 { used.clone() } else { vec![bf16] };
    eprintln!("CPU reference: {} distinct experts for 4 rows; {} bf16, dense bf16 {dense_bf16:?}", used.len(), bf16_set.len());
    let (w, host) = synthetic_layer_dense(41, &used, &bf16_set, dense_bf16);
    let moe = PrefillMoe::tables(w).unwrap();
    let mut s = PrefillScratch::new(128).unwrap();
    let h: Vec<f32> = (0..128).flat_map(|r| base[(r % 4) * HIDDEN..(r % 4 + 1) * HIDDEN].to_vec()).collect();
    let got = run_prefill(&moe, &mut s, &h, 128, &st);
    let want = block_cpu(&host, &base, 4, &picks, &wts);
    let e = rel_err(&got.out[..4 * HIDDEN], &want);
    eprintln!("prefill block (synthetic, 128 rows = 4 x 32, {} bf16 experts, dense bf16 {dense_bf16:?}) vs CPU reference: max rel {e:.3e}", bf16_set.len());
    assert!(e <= 1e-3, "the block is off the CPU reference by {e:e}");
    for r in 4..128 {
        assert_same(
            &format!("tile row {r}"),
            &got.out[r * HIDDEN..(r + 1) * HIDDEN],
            &got.out[(r % 4) * HIDDEN..(r % 4 + 1) * HIDDEN],
        );
    }
    let d = run_decode(&moe, &h, 128, &st);
    assert_runs_equal("tiled rows against the decode path", &got, &d);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The pack's layers

fn pack_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set; the pack-backed checks need the weights");
    }
    dir
}

fn plain_u16(ck: &Checkpoint, name: &str) -> Vec<u16> {
    let e = ck.plain.get(name).unwrap_or_else(|| panic!("the pack has no plain tensor {name}"));
    assert_eq!(e.dtype, "BF16", "{name} is {}, not BF16", e.dtype);
    ck.read_u16(e).unwrap()
}

fn plain_f32(ck: &Checkpoint, name: &str) -> Vec<f32> {
    let e = ck.plain.get(name).unwrap_or_else(|| panic!("the pack has no plain tensor {name}"));
    ck.read_floats(e).unwrap().into_iter().map(|v| v as f32).collect()
}

/// Bytes the kernels read of one projection: the trellis (or the bf16 weight) and the f32 scales and bias.
fn proj_bytes(p: &Proj) -> u64 {
    match p {
        Proj::Exl3 { k, n, trellis, bias, .. } => {
            trellis.bytes() as u64 + 4 * (*k as u64 + *n as u64) + bias.as_ref().map_or(0, |_| 4 * *n as u64)
        }
        Proj::Bf16 { w, .. } | Proj::F16 { w, .. } => w.bytes() as u64,
        Proj::Absent => 0,
    }
}

/// A pack layer's routed projection: the EXL3 group when the pack has one under `prefix`, else the bf16 `{prefix}.weight`.
fn pack_proj(ck: &Checkpoint, prefix: &str, k: usize, n: usize) -> Proj {
    if ck.groups.contains_key(prefix) {
        Proj::exl3_from_group(&ck.load_group(prefix).unwrap_or_else(|e| panic!("{prefix}: {e}"))).unwrap()
    } else {
        Proj::bf16(k, n, &plain_u16(ck, &format!("{prefix}.weight"))).unwrap()
    }
}

/// Text layer `layer`'s block from the pack and the bytes of its projections: gate, bias, dense EXL3 projections, every routed expert
/// (EXL3 or bf16 as the pack stores it).
fn pack_layer(ck: &Checkpoint, layer: usize) -> (Weights, Bytes) {
    let mp = format!("language_model.backbone.layers.{layer}.mixer");
    let up: Vec<Proj> = (0..EXPERTS).map(|e| pack_proj(ck, &format!("{mp}.experts.{e}.up_proj"), LATENT, EXPERT_INTER)).collect();
    let down: Vec<Proj> =
        (0..EXPERTS).map(|e| pack_proj(ck, &format!("{mp}.experts.{e}.down_proj"), EXPERT_INTER, LATENT)).collect();
    // a dense projection: the EXL3 group, or the bf16 weight the repaired pack keeps for a shared expert's matrix
    let dense = |name: &str, k: usize, n: usize| pack_proj(ck, &format!("{mp}.{name}"), k, n);
    let w = Weights {
        gate: DevBuf::from_slice(&plain_u16(ck, &format!("{mp}.gate.weight"))).unwrap(),
        gate_dtype: gb_cuda::nemotron_moe::GateDtype::Bf16,
        router_bias: DevBuf::from_slice(&plain_f32(ck, &format!("{mp}.gate.e_score_correction_bias"))).unwrap(),
        scale: SCALE,
        fc1: dense("fc1_latent_proj", HIDDEN, LATENT),
        fc2: dense("fc2_latent_proj", LATENT, HIDDEN),
        shared_up: dense("shared_experts.up_proj", HIDDEN, SHARED_INTER),
        shared_down: dense("shared_experts.down_proj", SHARED_INTER, HIDDEN),
        up,
        down,
    };
    let bytes = Bytes {
        fixed: w.gate.bytes() as u64
            + proj_bytes(&w.fc1)
            + proj_bytes(&w.fc2)
            + proj_bytes(&w.shared_up)
            + proj_bytes(&w.shared_down),
        expert: w.up.iter().zip(&w.down).map(|(u, d)| proj_bytes(u) + proj_bytes(d)).collect(),
    };
    (w, bytes)
}

struct Bytes {
    fixed: u64,
    expert: Vec<u64>,
}

/// Layers of the pack with a bf16 dense projection (the repaired pack's shared experts): the prefill call equals the decode path in bits
/// at 65 and 256 rows. The original pack has none: the test then says so and passes.
#[test]
fn pack_layers_with_a_bf16_dense_projection_equal_the_decode_path() {
    gpu!();
    let Some(dir) = pack_dir() else { return };
    let ck = scan(&dir, true).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let mut layers: Vec<usize> = Vec::new();
    for l in 0..88usize {
        let mp = format!("language_model.backbone.layers.{l}.mixer");
        let any = ["fc1_latent_proj", "fc2_latent_proj", "shared_experts.up_proj", "shared_experts.down_proj"]
            .iter()
            .any(|n| ck.plain.contains_key(&format!("{mp}.{n}.weight")));
        if any {
            layers.push(l);
        }
    }
    eprintln!("pack layers with a bf16 dense projection: {layers:?}");
    if layers.is_empty() {
        return;
    }
    let st = Stream::new().unwrap();
    let mut s = PrefillScratch::new(256).unwrap();
    // the first, the middle and the last such layer
    let picked: Vec<usize> =
        [0, layers.len() / 2, layers.len() - 1].iter().map(|&i| layers[i]).collect::<BTreeSet<_>>().into_iter().collect();
    for l in picked {
        let (w, _) = pack_layer(&ck, l);
        let moe = PrefillMoe::tables(w).unwrap();
        assert!(moe.supports_prefill(), "pack layer {l} is not run by the prefill path");
        for &rows in &[65usize, 256] {
            let h = Rng(0xFA58 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
            let d = run_decode(&moe, &h, rows, &st);
            let p = run_prefill(&moe, &mut s, &h, rows, &st);
            assert_runs_equal(&format!("pack layer {l}, {rows} rows"), &p, &d);
        }
        eprintln!(
            "pack layer {l} ({} bf16 experts, a bf16 dense projection): 65 and 256 rows are bitwise the decode path's",
            moe.slow_experts()
        );
    }
}

impl Bytes {
    /// Bytes one call reads: the dense weights, and each distinct picked expert's pair once.
    fn of_picks(&self, pick: &[i32]) -> u64 {
        let distinct: BTreeSet<i32> = pick.iter().copied().collect();
        self.fixed + distinct.iter().map(|&e| self.expert[e as usize]).sum::<u64>()
    }
}

/// The pack's layer 1 (all 512 experts 4-bit Mul1) and layer 26 (bf16 expert pairs among the 4-bit ones): the prefill call equals the
/// decode path in bits at 1, 65, 128 and 256 rows. Reads the pack only.
#[test]
fn prefill_equals_the_decode_path_on_the_pack_layers() {
    gpu!();
    let Some(dir) = pack_dir() else { return };
    let ck = scan(&dir, true).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let st = Stream::new().unwrap();
    let mut s = PrefillScratch::new(256).unwrap();
    for layer_idx in [1usize, 26] {
        let (w, _) = pack_layer(&ck, layer_idx);
        let moe = PrefillMoe::tables(w).unwrap();
        assert!(moe.supports_prefill(), "pack layer {layer_idx} is not run by the prefill path");
        eprintln!("pack layer {layer_idx}: {} bf16 experts", moe.slow_experts());
        for &rows in &[1usize, 65, 128, 256] {
            let h = Rng(0xFA57 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
            let d = run_decode(&moe, &h, rows, &st);
            let p = run_prefill(&moe, &mut s, &h, rows, &st);
            assert_runs_equal(&format!("pack layer {layer_idx}, {rows} rows"), &p, &d);
            eprintln!("pack layer {layer_idx}, {rows} rows: output, logits, picks and weights are bitwise the decode path's");
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Timing

mod ev {
    use core::ffi::c_void;
    extern "C" {
        pub fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
        pub fn cudaEventSynchronize(e: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
        pub fn cudaEventDestroy(e: *mut c_void) -> i32;
    }
}

struct Ev(*mut c_void);

impl Ev {
    fn new() -> Self {
        let mut e = core::ptr::null_mut();
        assert_eq!(unsafe { ev::cudaEventCreateWithFlags(&mut e, 0) }, 0);
        Ev(e)
    }
    fn record(&self, st: &Stream) {
        assert_eq!(unsafe { ev::cudaEventRecord(self.0, st.raw()) }, 0);
    }
    fn us_to(&self, later: &Ev) -> f64 {
        let mut ms = 0f32;
        assert_eq!(unsafe { ev::cudaEventSynchronize(later.0) }, 0);
        assert_eq!(unsafe { ev::cudaEventElapsedTime(&mut ms, self.0, later.0) }, 0);
        ms as f64 * 1000.0
    }
}

impl Drop for Ev {
    fn drop(&mut self) {
        unsafe {
            ev::cudaEventDestroy(self.0);
        }
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// The row counts of the timing tests: 64, 128, 256, or `REXL3_NEMO_PERF_ROWS="128,256"`.
fn bench_rows() -> Vec<usize> {
    match std::env::var("REXL3_NEMO_PERF_ROWS") {
        Ok(v) => v
            .split(',')
            .map(|s| s.trim().parse().expect("REXL3_NEMO_PERF_ROWS is a comma-separated list of row counts"))
            .collect(),
        Err(_) => vec![64, 128, 256],
    }
}

/// Whole-block time of `layer` at each row count, on the prefill path and on the decode path (one call at that row count, a scratch for
/// it), another layer run before every timed call so that L2 holds clean lines of other weights (as in a prefill slice). Bytes are the
/// weights the picks touch, once each; the floor is one read of them at `FLOOR_GBPS`.
fn bench_layer(label: &str, layer: &PrefillMoe, other: &PrefillMoe, bytes: &Bytes, rows_list: &[usize], reps: usize) {
    // `REXL3_NEMO_PERF_SCHEDULE=0|1|2`: the routed projections' launch schedule (see `set_schedule`)
    if let Some(m) = std::env::var("REXL3_NEMO_PERF_SCHEDULE").ok().and_then(|v| v.parse::<u8>().ok()) {
        gb_cuda::nemotron_moe_prefill::set_schedule(m);
        eprintln!("schedule {m}");
    }
    // `REXL3_NEMO_PERF_BF16=0|1`: the bf16 projections' kernel (see `set_bf16_kernel`)
    if let Some(k) = std::env::var("REXL3_NEMO_PERF_BF16").ok().and_then(|v| v.parse::<u8>().ok()) {
        gb_cuda::nemotron_moe_prefill::set_bf16_kernel(k);
        eprintln!("bf16 kernel {k}");
    }
    let st = Stream::new_nonblocking().unwrap();
    let max = *rows_list.iter().max().unwrap();
    let mut ps = PrefillScratch::new(max).unwrap();
    let mut ds = MoeScratch::new(max).unwrap();
    let (a, b) = (Ev::new(), Ev::new());
    eprintln!("\n=== {label}: whole block, another layer before each call, median of {reps}; floor = one read of the picked weights at {FLOOR_GBPS} GB/s ===");
    eprintln!(
        "{:>5} {:>10} {:>9} {:>9} {:>9} {:>12} {:>14} {:>10} {:>14} {:>16}",
        "rows", "experts", "MB", "floor us", "us", "x floor", "GB/s", "rows/s", "decode path us", "64-row calls us"
    );
    for &rows in rows_list {
        let hs: Vec<DevBuf> = (0..reps.max(2))
            .map(|i| DevBuf::from_slice(&Rng(8000 + 31 * i as u64 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0)).unwrap())
            .collect();
        let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
        let (mut t_p, mut t_d, mut t_c) = (Vec::new(), Vec::new(), Vec::new());
        // warm-up
        layer.forward_prefill_in(&mut ps, &hs[0], rows, &mut out, &st).unwrap();
        layer.forward_in(&mut ds, &hs[0], rows, &mut out, &st).unwrap();
        st.sync().unwrap();
        for i in 0..reps {
            other.forward_in(&mut ds, &hs[(i + 1) % hs.len()], rows, &mut out, &st).unwrap();
            a.record(&st);
            layer.forward_prefill_in(&mut ps, &hs[i % hs.len()], rows, &mut out, &st).unwrap();
            b.record(&st);
            t_p.push(a.us_to(&b));
            other.forward_in(&mut ds, &hs[(i + 1) % hs.len()], rows, &mut out, &st).unwrap();
            a.record(&st);
            layer.forward_in(&mut ds, &hs[i % hs.len()], rows, &mut out, &st).unwrap();
            b.record(&st);
            t_d.push(a.us_to(&b));
            // what a 64-row-slice engine runs for these rows: calls of 64 rows (the same input each: only the time is wanted)
            other.forward_in(&mut ds, &hs[(i + 1) % hs.len()], rows.min(64), &mut out, &st).unwrap();
            a.record(&st);
            for _ in 0..rows.div_ceil(64) {
                layer.forward_in(&mut ds, &hs[i % hs.len()], rows.min(64), &mut out, &st).unwrap();
            }
            b.record(&st);
            t_c.push(a.us_to(&b));
        }
        let (pick, _) = ps.download_route(rows, &st).unwrap();
        let mb = bytes.of_picks(&pick) as f64 / 1e6;
        let experts = pick.iter().collect::<BTreeSet<_>>().len();
        let floor = mb * 1e6 / (FLOOR_GBPS * 1e9) * 1e6;
        let (tp, td, tc) = (median(&mut t_p), median(&mut t_d), median(&mut t_c));
        eprintln!(
            "{rows:>5} {experts:>10} {mb:>9.1} {floor:>9.0} {tp:>9.0} {:>12.2} {:>14.1} {:>10.0} {td:>14.0} {tc:>16.0}",
            tp / floor,
            mb * 1e6 / (tp * 1e-6) / 1e9,
            rows as f64 / (tp * 1e-6)
        );
    }
}

/// The MTP head's block, synthetic: all 512 experts and the four dense projections bf16 (what the head's absorb of a prefill slice runs).
/// The floor is one read of the picked experts' bf16 weights (2 x 5.5 MB an expert) at `FLOOR_GBPS`.
#[test]
#[ignore = "timing"]
fn prefill_bench_head_like_block() {
    gpu!();
    let all: Vec<usize> = (0..EXPERTS).collect();
    let (w, _) = synthetic_layer_dense(401, &all, &all, [true; 4]);
    let bytes = Bytes {
        fixed: w.gate.bytes() as u64
            + proj_bytes(&w.fc1)
            + proj_bytes(&w.fc2)
            + proj_bytes(&w.shared_up)
            + proj_bytes(&w.shared_down),
        expert: w.up.iter().zip(&w.down).map(|(u, d)| proj_bytes(u) + proj_bytes(d)).collect(),
    };
    let l = PrefillMoe::tables(w).unwrap();
    bench_layer("MTP head block (512 bf16 experts, bf16 dense; synthetic)", &l, &l, &bytes, &bench_rows(), 9);
}

/// Per-width prefill throughput (v0.1.2 lane K, P1-5, the P1 kernels): a layer of 512 experts all at K bits (K = 2 .. 8, dense 4-bit)
/// on `pf_gemmk` (and the 4-bit layer on `pf_gemm4`), then his pack's shape (mixed widths, fp16 fc1 and fc2). `REXL3_NEMO_PERF_WIDTHS`
/// picks the widths, `REXL3_NEMO_PERF_ROWS` the rows (64, 128, 256).
#[test]
#[ignore = "timing"]
fn prefill_bench_widths() {
    gpu!();
    gb_cuda::nemotron_moe_prefill::set_any_width_prefill(true);
    let all: Vec<usize> = (0..EXPERTS).collect();
    let bytes_of = |w: &Weights| Bytes {
        fixed: w.gate.bytes() as u64
            + proj_bytes(&w.fc1)
            + proj_bytes(&w.fc2)
            + proj_bytes(&w.shared_up)
            + proj_bytes(&w.shared_down),
        expert: w.up.iter().zip(&w.down).map(|(u, d)| proj_bytes(u) + proj_bytes(d)).collect(),
    };
    let other = PrefillMoe::tables(mixed_layer(499, &all, &[], |_| 4, |_| 4, [4; 4])).unwrap();
    let widths: Vec<usize> = match std::env::var("REXL3_NEMO_PERF_WIDTHS") {
        Ok(v) => v.split(',').map(|s| s.trim().parse().unwrap()).collect(),
        Err(_) => (2..=8).collect(),
    };
    for k in widths {
        let w = mixed_layer(500 + k as u64, &all, &[], |_| k, |_| k, [4; 4]);
        let b = bytes_of(&w);
        let l = PrefillMoe::tables(w).unwrap();
        bench_layer(&format!("512 experts at {k} bits, dense 4-bit (synthetic)"), &l, &other, &b, &bench_rows(), 7);
    }
    let (name, w) = mixed_cases().remove(0);
    let b = bytes_of(&w);
    let l = PrefillMoe::tables(w).unwrap();
    bench_layer(&format!("{name} (synthetic)"), &l, &other, &b, &bench_rows(), 7);
}

/// The pack's layers 1 and 26 at 64, 128 and 256 rows: time, bytes read against the floor, rows/s and GB/s./// The pack's layers 1 and 26 at 64, 128 and 256 rows: time, bytes read against the floor, rows/s and GB/s.
#[test]
#[ignore = "timing"]
fn prefill_bench_pack_layers() {
    gpu!();
    let Some(dir) = pack_dir() else { return };
    let ck = scan(&dir, true).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let (w1, b1) = pack_layer(&ck, 1);
    let (w26, b26) = pack_layer(&ck, 26);
    let (w3, _) = pack_layer(&ck, 3);
    let l1 = PrefillMoe::tables(w1).unwrap();
    let l26 = PrefillMoe::tables(w26).unwrap();
    let l3 = PrefillMoe::tables(w3).unwrap();
    let rows = bench_rows();
    bench_layer("pack layer 1 (512 experts, 4-bit Mul1)", &l1, &l3, &b1, &rows, 9);
    bench_layer("pack layer 26 (bf16 expert pairs among 4-bit)", &l26, &l3, &b26, &rows, 9);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Lane S3, contract P2: the block's four dense projections (fc1, fc2, the shared expert's) on the tensor-core prefill GEMM
// (`nemotron_dense_mma`), which they take only inside a prefill scope with the opt-in on.

/// With the opt-in on and a prefill scope open, a 256-row block stays within 2e-3 (max, relative to the largest output) of the frozen
/// path, the router's logits, picks and weights are untouched (it reads the block's input), a row's bits do not depend on the call it
/// is in (slices of 65, 100, 91, 64 and 1 rows: P2(a), at any width), and outside a scope (a wide decode window) the block is the
/// frozen path's bit for bit.
#[test]
fn p2_dense_projections_stay_close_and_row_invariant() {
    gpu!();
    use gb_cuda::nemotron_dense_mma as ndm;
    let st = Stream::new().unwrap();
    let (moe, _) = layer(131, &(0..EXPERTS).collect::<Vec<usize>>(), &[]);
    let rows = 256;
    let h = Rng(77).vec(rows * HIDDEN, -1.0, 1.0);
    let mut s = PrefillScratch::new(rows).unwrap();
    ndm::set_enabled(true);
    let frozen = run_prefill(&moe, &mut s, &h, rows, &st);
    let p2 = {
        let _scope = ndm::PrefillScope::enter();
        run_prefill(&moe, &mut s, &h, rows, &st)
    };
    assert_same("router logits", &p2.logits, &frozen.logits);
    assert_same("router weights", &p2.wts, &frozen.wts);
    assert_eq!(p2.pick, frozen.pick, "router picks");
    let e = rel_err(&p2.out, &frozen.out);
    let differ = p2.out.iter().zip(&frozen.out).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    eprintln!(
        "P2 block (256 rows) against the frozen fp32 dense projections: max rel {e:.3e}, {differ} of {} values differ",
        p2.out.len()
    );
    assert!(e <= 2e-3 && differ > 0, "P2 block off the frozen path by {e:e} ({differ} values differ)");
    {
        let _scope = ndm::PrefillScope::enter();
        let mut start = 0;
        for n in [65usize, 100, 91] {
            let part = run_prefill(&moe, &mut s, &h[start * HIDDEN..], n, &st);
            assert_same(
                &format!("P2 rows {start}..{} in a {n}-row call", start + n),
                &part.out,
                &p2.out[start * HIDDEN..(start + n) * HIDDEN],
            );
            start += n;
        }
        let narrow = run_prefill(&moe, &mut s, &h, 64, &st);
        assert_same("a 64-row P2 call", &narrow.out, &p2.out[..64 * HIDDEN]);
        let one = run_prefill(&moe, &mut s, &h[200 * HIDDEN..], 1, &st);
        assert_same("a 1-row P2 call", &one.out, &p2.out[200 * HIDDEN..201 * HIDDEN]);
    }
    let again = run_prefill(&moe, &mut s, &h, rows, &st);
    assert_same("no scope: the frozen path", &again.out, &frozen.out);
    ndm::set_enabled(false);
}
