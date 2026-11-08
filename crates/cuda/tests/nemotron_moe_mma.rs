//! The tensor-core prefill path of the Nemotron latent-MoE block under contract P2 (`csrc/nemotron_moe_mma.cu`, lane S2;
//! `docs/NEMOTRON_PREFILL_MMA.md`), on SYNTHETIC EXL3 weights of the real shapes (and, ignored, on the pack's layers).
//!
//! Gates (GPU): (1) every projection shape against the default path's fp32 kernel and a host f64 reference, fp16 and bf16, max and rms
//! relative error, plus the exact scale equivariance of the ScaledRows format; (2) column halves (a TP rank's split of N) equal one call
//! bit for bit; (3) the block is deterministic and a row's bits do not depend on the other rows: 5 runs, slice widths 256 / 512 / 1024
//! / 2048, a permutation of the rows, padding (odd row counts), every item tile (16 to 128 rows) and the scratch poisoned before every
//! run; (4) the block against the fp32 path and the CPU block reference; (5) expert-parallel ranks summed equal the single call.
//! Ignored: timing (`bench` in the names) and the quality on the pack's layers with real layer inputs (`quality`).
//!
//!     cargo test -p gb-cuda --release --test nemotron_moe_mma -- --test-threads=1 --nocapture
//!     REXL3_NEMO_PERF_ROWS=256,512,1024,2048,4096 cargo test -p gb-cuda --release --test nemotron_moe_mma bench -- --ignored --nocapture
//!     REXL3_NEMO_OMNI_DIR=<pack> REXL3_NEMO_ORACLE_DIR=<ref-text-1k> cargo test -p gb-cuda --release --test nemotron_moe_mma quality -- --ignored --nocapture
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop, clippy::type_complexity)]
mod common;

use common::exl3_synth::synth_group;
use common::Rng;
use core::ffi::c_void;
use gb_cuda::nemotron_moe::{Proj, Weights, EXPERTS, EXPERT_INTER, HIDDEN, LATENT, SHARED_INTER, TOPK};
use gb_cuda::nemotron_moe_prefill::mma::{self, raw, MmaScratch, ProjTable};
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

// ---------------------------------------------------------------------------------------------------------------------------------
// Metrics

/// (max |d| / max |want|, rms d / rms want).
fn stats(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let (mut md, mut mw, mut sd, mut sw) = (0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let d = (g as f64 - w as f64).abs();
        md = md.max(d);
        mw = mw.max((w as f64).abs());
        sd += d * d;
        sw += (w as f64) * (w as f64);
    }
    (md / mw.max(1e-30), (sd / sw.max(1e-300)).sqrt())
}

fn assert_bits(what: &str, a: &[f32], b: &[f32]) {
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

// ---------------------------------------------------------------------------------------------------------------------------------
// Synthetic layers (4-bit Mul1 everywhere; experts outside `present` absent)

struct Host {
    gate: Vec<f32>,
    bias: Vec<f32>,
    table: Vec<u16>,
    dense: Vec<LoadedGroup>, // fc1, fc2, shared up, shared down
    experts: std::collections::BTreeMap<usize, (LoadedGroup, LoadedGroup)>,
}

fn bf16_bits(r: &mut Rng, n: usize) -> Vec<u16> {
    (0..n).map(|_| (r.uniform(-0.05, 0.05).to_bits() >> 16) as u16).collect()
}

fn bf16_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
}

/// The same seeds give the same gate, bias, dense projections and expert weights whatever `present` is (so the ranks of an
/// expert-parallel pair are slices of the same layer).
fn synthetic_layer(seed: u64, present: &[usize], keep_host: bool) -> (Weights, Option<Host>) {
    let mut r = Rng(seed);
    let gate_bits = bf16_bits(&mut r, EXPERTS * HIDDEN);
    let bias: Vec<f32> = r.vec(EXPERTS, -0.1, 0.1);
    let dims = [(HIDDEN, LATENT), (LATENT, HIDDEN), (HIDDEN, SHARED_INTER), (SHARED_INTER, HIDDEN)];
    let dense_g: Vec<LoadedGroup> =
        dims.iter().enumerate().map(|(i, &(k, n))| synth_group(k, n, 4, Codebook::Mul1, seed + 1 + i as u64)).collect();
    let dense_w: Vec<Proj> = dense_g.iter().map(|g| Proj::exl3_from_group(g).unwrap()).collect();
    let mut up: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    let mut down: Vec<Proj> = (0..EXPERTS).map(|_| Proj::Absent).collect();
    let mut map = std::collections::BTreeMap::new();
    for &e in present {
        let ug = synth_group(LATENT, EXPERT_INTER, 4, Codebook::Mul1, seed + 1000 + 2 * e as u64);
        let dg = synth_group(EXPERT_INTER, LATENT, 4, Codebook::Mul1, seed + 1001 + 2 * e as u64);
        up[e] = Proj::exl3_from_group(&ug).unwrap();
        down[e] = Proj::exl3_from_group(&dg).unwrap();
        if keep_host {
            map.insert(e, (ug, dg));
        }
    }
    let mut dw = dense_w.into_iter();
    let w = Weights {
        gate: DevBuf::from_slice(&gate_bits).unwrap(),
        gate_dtype: gb_cuda::nemotron_moe::GateDtype::Bf16,
        router_bias: DevBuf::from_slice(&bias).unwrap(),
        scale: SCALE,
        fc1: dw.next().unwrap(),
        fc2: dw.next().unwrap(),
        shared_up: dw.next().unwrap(),
        shared_down: dw.next().unwrap(),
        up,
        down,
    };
    let host = keep_host.then(|| Host {
        gate: bf16_f32(&gate_bits),
        bias,
        table: codebook(Codebook::Mul1),
        dense: dense_g,
        experts: map,
    });
    (w, host)
}

fn all_experts() -> Vec<usize> {
    (0..EXPERTS).collect()
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Running the paths

fn run_fma(moe: &PrefillMoe, s: &mut PrefillScratch, h: &[f32], rows: usize, st: &Stream) -> Vec<f32> {
    let hd = DevBuf::from_slice(&h[..rows * HIDDEN]).unwrap();
    let mut out = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * HIDDEN]).unwrap();
    s.poison(st).unwrap();
    moe.forward_prefill_in(s, &hd, rows, &mut out, st).unwrap();
    st.sync().unwrap();
    out.to_vec::<f32>().unwrap()
}

fn run_mma(moe: &PrefillMoe, s: &mut PrefillScratch, m: &mut MmaScratch, h: &[f32], rows: usize, st: &Stream) -> Vec<f32> {
    let hd = DevBuf::from_slice(&h[..rows * HIDDEN]).unwrap();
    let mut out = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * HIDDEN]).unwrap();
    s.poison(st).unwrap();
    m.poison(st).unwrap();
    moe.forward_prefill_mma(s, m, &hd, rows, &mut out, st).unwrap();
    st.sync().unwrap();
    out.to_vec::<f32>().unwrap()
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Host reference

fn exl3_cpu(g: &LoadedGroup, table: &[u16], x: &[f32], rows: usize) -> Vec<f32> {
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    forward(&x64, rows, &g.trellis, g.kt, g.nt, g.group.bits, table, &g.suh, &g.svh, g.bias.as_deref())
        .into_iter()
        .map(|v| v as f32)
        .collect()
}

fn relu2(v: &mut [f32]) {
    for x in v.iter_mut() {
        let r = x.max(0.0);
        *x = r * r;
    }
}

fn logits_cpu(h: &[f32], gate: &[f32]) -> Vec<f32> {
    (0..EXPERTS).map(|j| h.iter().zip(&gate[j * HIDDEN..(j + 1) * HIDDEN]).map(|(a, b)| a * b).sum()).collect()
}

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

fn block_cpu(s: &Host, h: &[f32], rows: usize, picks: &[Vec<usize>], wts: &[Vec<f32>]) -> Vec<f32> {
    let x = exl3_cpu(&s.dense[0], &s.table, h, rows);
    let mut routed = vec![0f32; rows * LATENT];
    for t in 0..rows {
        for sl in 0..TOPK {
            let (ug, dg) = &s.experts[&picks[t][sl]];
            let mut a = exl3_cpu(ug, &s.table, &x[t * LATENT..(t + 1) * LATENT], 1);
            relu2(&mut a);
            let y = exl3_cpu(dg, &s.table, &a, 1);
            for (d, &v) in routed[t * LATENT..(t + 1) * LATENT].iter_mut().zip(&y) {
                *d += wts[t][sl] * v;
            }
        }
    }
    let ro = exl3_cpu(&s.dense[1], &s.table, &routed, rows);
    let mut sa = exl3_cpu(&s.dense[2], &s.table, h, rows);
    relu2(&mut sa);
    let so = exl3_cpu(&s.dense[3], &s.table, &sa, rows);
    ro.iter().zip(&so).map(|(a, b)| a + b).collect()
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Gate 1: every projection shape against the fp32 kernel and the host

/// (name, k, n, relu2) of the six 4-bit projections of a MoE layer.
const SHAPES: [(&str, usize, usize, bool); 6] = [
    ("routed up", LATENT, EXPERT_INTER, true),
    ("routed down", EXPERT_INTER, LATENT, false),
    ("fc1", HIDDEN, LATENT, false),
    ("fc2", LATENT, HIDDEN, false),
    ("shared up", HIDDEN, SHARED_INTER, true),
    ("shared down", SHARED_INTER, HIDDEN, false),
];

/// Inputs of `rows` x `k`: `wide` spreads the row magnitudes over 1e-3 .. 1e3 and makes a few values 50x their row (relu2-like spikes).
fn inputs(seed: u64, rows: usize, k: usize, wide: bool) -> Vec<f32> {
    let mut r = Rng(seed);
    let mut x = r.vec(rows * k, -1.0, 1.0);
    if wide {
        for row in 0..rows {
            let mag = 10f32.powf(r.uniform(-3.0, 3.0));
            for v in &mut x[row * k..(row + 1) * k] {
                *v *= mag;
                if r.next_u64() % 97 == 0 {
                    *v *= 50.0;
                }
            }
        }
    }
    x
}

struct Proj1 {
    tab: ProjTable,
    _p: Proj,
    g: LoadedGroup,
}

fn proj1(k: usize, n: usize, seed: u64) -> Proj1 {
    let g = synth_group(k, n, 4, Codebook::Mul1, seed);
    let p = Proj::exl3_from_group(&g).unwrap();
    let tab = ProjTable::new(&[&p]).unwrap();
    Proj1 { tab, _p: p, g }
}

/// MMA projection of plain rows to fp32 (rotate, then the GEMM).
fn mma_proj(p: &Proj1, x: &[f32], rows: usize, k: usize, n: usize, relu2: bool, mt: usize, bf: bool, st: &Stream) -> Vec<f32> {
    let xd = DevBuf::from_slice(x).unwrap();
    let a16 = DevBuf::zeroed(rows * k * 2).unwrap();
    let ainv = DevBuf::zeroed(rows * k / 128 * 4).unwrap();
    let out = DevBuf::from_slice(&vec![f32::NAN; rows * n]).unwrap();
    unsafe {
        raw::rotate(xd.ptr(), k, 1, core::ptr::null(), p.tab.entry(0), rows, k, a16.ptr(), ainv.ptr(), bf, st).unwrap();
        raw::gemm_dense(
            a16.ptr(),
            ainv.ptr(),
            rows,
            mt,
            p.tab.entry(0),
            core::ptr::null(),
            k,
            n,
            0,
            n / 128,
            out.ptr(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            relu2,
            core::ptr::null(),
            bf,
            st,
        )
        .unwrap();
    }
    st.sync().unwrap();
    out.to_vec::<f32>().unwrap()
}

fn fma_proj(p: &Proj1, x: &[f32], rows: usize, k: usize, n: usize, relu2: bool, st: &Stream) -> Vec<f32> {
    let xd = DevBuf::from_slice(x).unwrap();
    let rot = DevBuf::zeroed(rows * k * 4).unwrap();
    let out = DevBuf::from_slice(&vec![f32::NAN; rows * n]).unwrap();
    unsafe { raw::fma_dense(xd.ptr(), rot.ptr(), rows, p.tab.entry(0), k, n, out.ptr(), relu2, st).unwrap() };
    st.sync().unwrap();
    out.to_vec::<f32>().unwrap()
}

#[test]
fn every_projection_shape_against_the_fp32_kernel_and_the_host() {
    gpu!();
    let st = Stream::new().unwrap();
    let table = codebook(Codebook::Mul1);
    let rows = 48;
    let host_rows = 6;
    eprintln!(
        "\n{:<12} {:>5} {:>5} {:<7} | {:>10} {:>10} | {:>10} {:>10} | {:>10} {:>10} | {:>10} {:>10}",
        "projection",
        "K",
        "N",
        "inputs",
        "fp16 max",
        "fp16 rms",
        "bf16 max",
        "bf16 rms",
        "fp32 max",
        "fp32 rms",
        "16-vs-32 mx",
        "16-vs-32 rms"
    );
    for (i, &(name, k, n, act)) in SHAPES.iter().enumerate() {
        let p = proj1(k, n, 7000 + i as u64);
        for wide in [false, true] {
            let x = inputs(900 + i as u64 + wide as u64 * 50, rows, k, wide);
            let mut want = exl3_cpu(&p.g, &table, &x[..host_rows * k], host_rows);
            if act {
                relu2(&mut want);
            }
            let m16 = mma_proj(&p, &x, rows, k, n, act, 4, false, &st);
            let mbf = mma_proj(&p, &x, rows, k, n, act, 4, true, &st);
            let f32k = fma_proj(&p, &x, rows, k, n, act, &st);
            let hn = host_rows * n;
            // per row: rows of very different magnitudes are compared each against its own scale
            let per_row = |got: &[f32]| -> (f64, f64) {
                let (mut mx, mut rms2) = (0f64, 0f64);
                for r in 0..host_rows {
                    let (a, b) = stats(&got[r * n..(r + 1) * n], &want[r * n..(r + 1) * n]);
                    mx = mx.max(a);
                    rms2 = rms2.max(b);
                }
                (mx, rms2)
            };
            let (a16, r16) = per_row(&m16[..hn]);
            let (abf, rbf) = per_row(&mbf[..hn]);
            let (a32, r32) = per_row(&f32k[..hn]);
            let (mut amf, mut rmf) = (0f64, 0f64);
            for r in 0..rows {
                let (a, b) = stats(&m16[r * n..(r + 1) * n], &f32k[r * n..(r + 1) * n]);
                amf = amf.max(a);
                rmf = rmf.max(b);
            }
            eprintln!(
                "{name:<12} {k:>5} {n:>5} {:<7} | {a16:>10.3e} {r16:>10.3e} | {abf:>10.3e} {rbf:>10.3e} | {a32:>10.3e} {r32:>10.3e} | {amf:>10.3e} {rmf:>10.3e}",
                if wide { "wide" } else { "uniform" }
            );
            assert!(m16.iter().all(|v| v.is_finite()), "{name}: a non-finite output");
            assert!(r16 < 3e-3 && a16 < 2e-2, "{name}: fp16 MMA off the host by max {a16:e} rms {r16:e}");
            assert!(rmf < 3e-3, "{name}: fp16 MMA off the fp32 kernel by rms {rmf:e}");
        }
        // exact scale equivariance of ScaledRows: x * 2^20 gives out * 2^20 bit for bit (no relu2, no bias), and no overflow
        if !act {
            let x = inputs(77 + i as u64, 32, k, true);
            let xs: Vec<f32> = x.iter().map(|v| v * 1048576.0).collect();
            let a = mma_proj(&p, &x, 32, k, n, false, 4, false, &st);
            let b = mma_proj(&p, &xs, 32, k, n, false, 4, false, &st);
            let a_scaled: Vec<f32> = a.iter().map(|v| v * 1048576.0).collect();
            assert_bits(&format!("{name}: scale equivariance"), &b, &a_scaled);
        }
    }
    eprintln!("(per row, worst row of {host_rows}: max |d| / max |ref| and rms d / rms ref against the host f64 reference; 16-vs-32: the fp16 MMA against the default path's fp32 kernel, worst of {rows} rows)");
}

/// The router logits on tensor cores (bf16 gate, h as two bf16 terms) against a host f64 reference, and a row's logits in any call.
#[test]
fn router_logits_against_the_host() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut r = Rng(5150);
    let gate_bits = bf16_bits(&mut r, EXPERTS * HIDDEN);
    let gate = bf16_f32(&gate_bits);
    let rows = 200;
    let h = inputs(77, rows, HIDDEN, true);
    let gd = DevBuf::from_slice(&gate_bits).unwrap();
    let hd = DevBuf::from_slice(&h).unwrap();
    let out = DevBuf::from_slice(&vec![f32::NAN; rows * EXPERTS]).unwrap();
    unsafe { raw::router(hd.ptr(), gd.ptr(), out.ptr(), rows, HIDDEN, &st).unwrap() };
    st.sync().unwrap();
    let got = out.to_vec::<f32>().unwrap();
    let mut worst = 0f64;
    for t in 0..rows {
        let want: Vec<f32> = (0..EXPERTS)
            .map(|e| {
                h[t * HIDDEN..(t + 1) * HIDDEN]
                    .iter()
                    .zip(&gate[e * HIDDEN..(e + 1) * HIDDEN])
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        worst = worst.max(stats(&got[t * EXPERTS..(t + 1) * EXPERTS], &want).0);
    }
    eprintln!(
        "router logits on tensor cores, {rows} rows (row magnitudes 1e-3..1e3): worst row max |d| / max |ref| = {worst:.3e}"
    );
    assert!(worst < 1e-4, "router logits off by {worst:e}");
    // a row's logits in a 7-row call starting at row 100 equal its logits in the 200-row call
    let o7 = DevBuf::from_slice(&vec![f32::NAN; 7 * EXPERTS]).unwrap();
    let h7 = DevBuf::from_slice(&h[100 * HIDDEN..107 * HIDDEN]).unwrap();
    unsafe { raw::router(h7.ptr(), gd.ptr(), o7.ptr(), 7, HIDDEN, &st).unwrap() };
    st.sync().unwrap();
    assert_bits("rows 100..107 alone", &o7.to_vec::<f32>().unwrap(), &got[100 * EXPERTS..107 * EXPERTS]);
}

/// The f32-gate routers of this path (an F16 gate widened, decision D1 of v0.1.2): the tiled fp32 kernel gives the decode path's f32-gate
/// logits bitwise; the tensor-core one (h and the gate split in two bf16 terms) matches a host f64 reference as the bf16 one does, a
/// row's logits do not depend on the call, and on a gate of bf16 values (the low terms are zero) it gives the bf16 tensor-core router's
/// bits.
#[test]
fn f32_gate_routers_against_the_decode_kernel_and_the_host() {
    gpu!();
    use gb_cuda::nemotron_moe::{gate_f32_from_f16, GateDtype};
    let st = Stream::new().unwrap();
    let mut r = Rng(0xF16A);
    let fbits: Vec<u16> = (0..EXPERTS * HIDDEN).map(|_| gb_exl3::f16::f64_to_f16(r.uniform(-0.05, 0.05) as f64)).collect();
    let gate = gate_f32_from_f16(&fbits);
    let rows = 200;
    let h = inputs(78, rows, HIDDEN, true);
    let gd = DevBuf::from_slice(&gate).unwrap();
    let hd = DevBuf::from_slice(&h).unwrap();
    let nan = |n: usize| DevBuf::from_slice(&vec![f32::NAN; n]).unwrap();
    // tiled fp32 against the decode path's f32-gate kernel (a router-only block's fast route)
    let tiled = nan(rows * EXPERTS);
    unsafe { raw::router_tiled(hd.ptr(), gd.ptr(), GateDtype::F32, tiled.ptr(), rows, &st).unwrap() };
    let (mut w, _) = synthetic_layer(1, &[], false);
    w.gate = DevBuf::from_slice(&gate).unwrap();
    w.gate_dtype = GateDtype::F32;
    let blk = gb_cuda::nemotron_moe::NemotronMoe::tables(w).unwrap();
    let mut ms = gb_cuda::nemotron_moe::MoeScratch::new(rows).unwrap();
    blk.route_in(&mut ms, &hd, rows, &st).unwrap();
    let dec = ms.download_logits(rows, &st).unwrap();
    st.sync().unwrap();
    assert_bits("tiled f32-gate router vs the decode path's", &tiled.to_vec::<f32>().unwrap(), &dec);
    // tensor cores against the host
    let out = nan(rows * EXPERTS);
    unsafe { raw::router_g32(hd.ptr(), gd.ptr(), out.ptr(), rows, HIDDEN, &st).unwrap() };
    st.sync().unwrap();
    let got = out.to_vec::<f32>().unwrap();
    let mut worst = 0f64;
    for t in 0..rows {
        let want: Vec<f32> = (0..EXPERTS)
            .map(|e| {
                h[t * HIDDEN..(t + 1) * HIDDEN]
                    .iter()
                    .zip(&gate[e * HIDDEN..(e + 1) * HIDDEN])
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        worst = worst.max(stats(&got[t * EXPERTS..(t + 1) * EXPERTS], &want).0);
    }
    eprintln!("f32-gate router on tensor cores, {rows} rows: worst row max |d| / max |ref| = {worst:.3e}; tiled == decode kernel bitwise");
    assert!(worst < 1e-4, "f32-gate tensor-core router logits off by {worst:e}");
    let o7 = nan(7 * EXPERTS);
    let h7 = DevBuf::from_slice(&h[100 * HIDDEN..107 * HIDDEN]).unwrap();
    unsafe { raw::router_g32(h7.ptr(), gd.ptr(), o7.ptr(), 7, HIDDEN, &st).unwrap() };
    st.sync().unwrap();
    assert_bits("f32-gate tensor cores, rows 100..107 alone", &o7.to_vec::<f32>().unwrap(), &got[100 * EXPERTS..107 * EXPERTS]);
    // a gate of bf16 values: the bf16 tensor-core router's bits
    let bbits = bf16_bits(&mut r, EXPERTS * HIDDEN);
    let (gb, gf) = (DevBuf::from_slice(&bbits).unwrap(), DevBuf::from_slice(&bf16_f32(&bbits)).unwrap());
    let (ob, of) = (nan(rows * EXPERTS), nan(rows * EXPERTS));
    unsafe { raw::router(hd.ptr(), gb.ptr(), ob.ptr(), rows, HIDDEN, &st).unwrap() };
    unsafe { raw::router_g32(hd.ptr(), gf.ptr(), of.ptr(), rows, HIDDEN, &st).unwrap() };
    st.sync().unwrap();
    assert_bits(
        "f32-gate tensor cores on bf16 values vs the bf16 one",
        &of.to_vec::<f32>().unwrap(),
        &ob.to_vec::<f32>().unwrap(),
    );
    // the F16 gate as stored: the f32 kernels' bits on the widened gate, tiled and on tensor cores
    let g16 = DevBuf::from_slice(&fbits).unwrap();
    let (t16, c16) = (nan(rows * EXPERTS), nan(rows * EXPERTS));
    unsafe { raw::router_tiled(hd.ptr(), g16.ptr(), GateDtype::F16, t16.ptr(), rows, &st).unwrap() };
    unsafe { raw::router_g16(hd.ptr(), g16.ptr(), c16.ptr(), rows, HIDDEN, &st).unwrap() };
    st.sync().unwrap();
    assert_bits("F16 gate tiled vs the widened f32 gate", &t16.to_vec::<f32>().unwrap(), &tiled.to_vec::<f32>().unwrap());
    assert_bits("F16 gate on tensor cores vs the widened f32 gate", &c16.to_vec::<f32>().unwrap(), &got);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// MixedK (v0.1.2 lane K, P1-5): routed experts at 2 to 8 bits on tensor cores (one launch a width), fp16 and any-width dense projections
// on the fp32 kernels inside the block

/// A MixedK synthetic layer (as tests/nemotron_moe_perf.rs's): each present expert's up and down at `up_bits(e)`, `down_bits(e)` (Mul1),
/// the dense projections at `dense[i]` bits (0: fp16 `[N, K]`).
fn mixed_layer(
    seed: u64,
    present: &[usize],
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
        up[e] = g(LATENT, EXPERT_INTER, up_bits(e), seed + 1000 + 2 * e as u64);
        down[e] = g(EXPERT_INTER, LATENT, down_bits(e), seed + 1001 + 2 * e as u64);
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

fn width_of(e: usize, salt: u64) -> usize {
    let h = (e as u64 ^ salt).wrapping_mul(0x9E3779B97F4A7C15) >> 59;
    [2, 3, 4, 5, 6][(h % 5) as usize]
}

/// MixedK layers on tensor cores (his shapes: experts at 2 to 6 bits, up and down apart, fp16 fc1 and fc2, shared 5 / 4; and 7 / 8
/// bits with dense 8/6/7/6): against the fp32 path (bit-gated against the decode path) within the 4-bit block's limits, deterministic
/// over runs, and a row's bits do not depend on the slice, its width, the row order, the padding, the item tile or the stream.
#[test]
fn mixedk_blocks_on_tensor_cores_stay_close_to_the_fp32_path_and_row_independent() {
    gpu!();
    gb_cuda::nemotron_moe_prefill::set_any_width_prefill(true);
    let st = Stream::new().unwrap();
    let all = all_experts();
    let fifth: Vec<usize> = (0..EXPERTS).step_by(5).collect();
    let even: Vec<usize> = (0..EXPERTS).step_by(2).collect();
    let cases = [
        ("his shapes", mixed_layer(211, &all, |e| width_of(e, 1), |e| width_of(e, 2), [0, 0, 5, 4])),
        ("7 / 8 bits, dense 8/6/7/6", mixed_layer(212, &fifth, |e| [7, 8, 4][e % 3], |e| [8, 7, 5][e % 3], [8, 6, 7, 6])),
        (
            "a TP rank's half (absent experts), mixed widths",
            mixed_layer(213, &even, |e| width_of(e, 3), |e| width_of(e, 4), [4, 4, 4, 4]),
        ),
    ];
    let rows = 1024;
    let mut s = PrefillScratch::new(rows).unwrap();
    let mut m = MmaScratch::new(rows).unwrap();
    let h = Rng(0x5EEE).vec(rows * HIDDEN, -1.0, 1.0);
    for (name, w) in cases {
        let moe = PrefillMoe::tables(w).unwrap();
        assert!(moe.supports_prefill(), "{name}");
        mma::set_tile(4);
        let base = run_mma(&moe, &mut s, &mut m, &h, rows, &st);
        assert!(base.iter().all(|v| v.is_finite()) && base.iter().any(|v| *v != 0.0), "{name}: not finite or all zero");
        let fma = run_fma(&moe, &mut s, &h, rows, &st);
        let (mx, rms) = stats(&base, &fma);
        eprintln!("{name}, {rows} rows: tensor cores against the fp32 path max {mx:.3e} rms {rms:.3e}");
        assert!(mx < 1e-2 && rms < 3e-3, "{name}: the tensor-core block is off the fp32 path: max {mx:e} rms {rms:e}");
        assert_bits(&format!("{name}: run 1 against run 0"), &run_mma(&moe, &mut s, &mut m, &h, rows, &st), &base);
        for width in [256usize, 512] {
            let mut cat = Vec::with_capacity(rows * HIDDEN);
            for c in 0..rows / width {
                cat.extend(run_mma(&moe, &mut s, &mut m, &h[c * width * HIDDEN..], width, &st));
            }
            assert_bits(&format!("{name}: slices of {width}"), &cat, &base);
        }
        let n = 512;
        let mut perm: Vec<usize> = (0..n).collect();
        let mut r = Rng(98);
        for i in (1..n).rev() {
            perm.swap(i, (r.next_u64() % (i as u64 + 1)) as usize);
        }
        let hp: Vec<f32> = perm.iter().flat_map(|&i| h[i * HIDDEN..(i + 1) * HIDDEN].to_vec()).collect();
        let op = run_mma(&moe, &mut s, &mut m, &hp, n, &st);
        for (j, &i) in perm.iter().enumerate() {
            assert_bits(
                &format!("{name}: permuted row {j}"),
                &op[j * HIDDEN..(j + 1) * HIDDEN],
                &base[i * HIDDEN..(i + 1) * HIDDEN],
            );
        }
        for cnt in [1usize, 17, 333] {
            assert_bits(&format!("{name}: {cnt} rows"), &run_mma(&moe, &mut s, &mut m, &h, cnt, &st), &base[..cnt * HIDDEN]);
        }
        for mt in [1u8, 2, 8] {
            mma::set_tile(mt);
            assert_bits(&format!("{name}: tile {mt}"), &run_mma(&moe, &mut s, &mut m, &h, rows, &st), &base);
        }
        mma::set_tile(4);
        gb_cuda::nemotron_moe_prefill::set_side_stream(false);
        let o = run_mma(&moe, &mut s, &mut m, &h, rows, &st);
        gb_cuda::nemotron_moe_prefill::set_side_stream(true);
        assert_bits(&format!("{name}: shared expert in line"), &o, &base);
        eprintln!("{name}: deterministic; slices, permutation, padding, tiles 1/2/8 and the in-line shared expert bit-equal");
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// fp16 dense projections on tensor cores (v0.1.2 lane K, CP2: the MixedK pack's fc1 / fc2, `nmm_dense_h16`)

/// An fp16 projection's tensor-core output for plain rows `x` (`nmm_plain_rows`, then `nmm_dense_h16` at tile `mt`), with the
/// epilogue's relu2, `add` and `add2` (`add2_is_out`: `out` preloaded with `add2` and passed as it, the residual's in-place form).
#[allow(clippy::too_many_arguments)]
fn h16_proj(
    tab: &ProjTable,
    x: &[f32],
    rows: usize,
    k: usize,
    n: usize,
    mt: usize,
    relu2: bool,
    add: Option<&[f32]>,
    add2: Option<&[f32]>,
    add2_is_out: bool,
    st: &Stream,
) -> Vec<f32> {
    let xd = DevBuf::from_slice(&x[..rows * k]).unwrap();
    let a16 = DevBuf::from_slice(&vec![0x7e00u16; rows * k]).unwrap();
    let ainv = DevBuf::from_slice(&vec![f32::NAN; rows * k / 128]).unwrap();
    let init = match (add2, add2_is_out) {
        (Some(b), true) => b.to_vec(),
        _ => vec![f32::from_bits(0x7fc0_dead); rows * n],
    };
    let out = DevBuf::from_slice(&init).unwrap();
    let addd = add.map(|v| DevBuf::from_slice(v).unwrap());
    let add2d = if add2_is_out { None } else { add2.map(|v| DevBuf::from_slice(v).unwrap()) };
    let ptr = |b: &Option<DevBuf>| b.as_ref().map_or(core::ptr::null(), |b| b.ptr() as *const c_void);
    let add2p = if add2_is_out { out.ptr() as *const c_void } else { ptr(&add2d) };
    unsafe {
        raw::plain_rows(xd.ptr(), k, rows, k, a16.ptr(), ainv.ptr(), st).unwrap();
        raw::dense_h16(a16.ptr(), ainv.ptr(), rows, mt, tab.entry(0), k, n, out.ptr(), relu2, ptr(&addd), add2p, st).unwrap();
    }
    st.sync().unwrap();
    out.to_vec::<f32>().unwrap()
}

/// The fp32 chain of an fp16 dense projection (`pf_slow4`: the decode path's bits).
fn h16_fp32(tab: &ProjTable, x: &[f32], rows: usize, k: usize, n: usize, st: &Stream) -> Vec<f32> {
    let xd = DevBuf::from_slice(&x[..rows * k]).unwrap();
    let out = DevBuf::from_slice(&vec![f32::NAN; rows * n]).unwrap();
    unsafe { raw::slow_dense(xd.ptr(), rows, tab.entry(0), k, n, out.ptr(), false, 1, st).unwrap() };
    st.sync().unwrap();
    out.to_vec::<f32>().unwrap()
}

/// fp16 dense projections (the MixedK pack's fc1 `[1024, 4096]` and fc2 `[4096, 1024]`, `Proj::F16`) on tensor cores (`nmm_dense_h16`,
/// the P2 arithmetic of a plain weight): against a host f64 reference and against the fp32 chain within the 4-bit projections' P2 limits
/// (rms 3e-3, max 2e-2 a row), uniform inputs and rows over six decades with spikes; exactly scale-equivariant (ScaledRows); a row's
/// bits independent of the call (the rows with it, the tile, the row order); the epilogue's relu2, + add, + add2 (in place too) in order.
#[test]
fn fp16_dense_projections_on_tensor_cores_against_the_fp32_chain_and_the_host() {
    gpu!();
    let st = Stream::new().unwrap();
    let (rows, host_rows) = (48usize, 6usize);
    for (i, &(name, k, n)) in [("fc1", HIDDEN, LATENT), ("fc2", LATENT, HIDDEN)].iter().enumerate() {
        let mut r = Rng(8800 + i as u64);
        let bits: Vec<u16> = (0..k * n).map(|_| gb_exl3::f16::f64_to_f16(r.uniform(-0.05, 0.05) as f64)).collect();
        let wq: Vec<f64> = bits.iter().map(|&b| gb_exl3::f16::f16_to_f64(b)).collect();
        let p = Proj::f16(k, n, &bits).unwrap();
        let tab = ProjTable::new(&[&p]).unwrap();
        for wide in [false, true] {
            let x = inputs(950 + i as u64 + wide as u64 * 50, rows, k, wide);
            let want: Vec<f32> = (0..host_rows * n)
                .map(|o| {
                    let (row, c) = (o / n, o % n);
                    (0..k).map(|j| x[row * k + j] as f64 * wq[c * k + j]).sum::<f64>() as f32
                })
                .collect();
            let m16 = h16_proj(&tab, &x, rows, k, n, 4, false, None, None, false, &st);
            let f32k = h16_fp32(&tab, &x, rows, k, n, &st);
            assert!(m16.iter().all(|v| v.is_finite()), "{name}: a non-finite output");
            let (mut a16, mut r16, mut amf, mut rmf) = (0f64, 0f64, 0f64, 0f64);
            for row in 0..rows {
                let (a, b) = stats(&m16[row * n..(row + 1) * n], &f32k[row * n..(row + 1) * n]);
                (amf, rmf) = (amf.max(a), rmf.max(b));
                if row < host_rows {
                    let (a, b) = stats(&m16[row * n..(row + 1) * n], &want[row * n..(row + 1) * n]);
                    (a16, r16) = (a16.max(a), r16.max(b));
                }
            }
            eprintln!(
                "{name} [{n}, {k}] {}: tensor cores against the host max {a16:.3e} rms {r16:.3e}, against the fp32 chain max {amf:.3e} rms {rmf:.3e}",
                if wide { "wide" } else { "uniform" }
            );
            assert!(r16 < 3e-3 && a16 < 2e-2, "{name}: off the host by max {a16:e} rms {r16:e}");
            assert!(rmf < 3e-3 && amf < 2e-2, "{name}: off the fp32 chain by max {amf:e} rms {rmf:e}");
        }
        // ScaledRows: x * 2^20 gives out * 2^20 bit for bit
        let x = inputs(77 + i as u64, rows, k, true);
        let xs: Vec<f32> = x.iter().map(|v| v * 1048576.0).collect();
        let base = h16_proj(&tab, &x, rows, k, n, 4, false, None, None, false, &st);
        let scaled: Vec<f32> = base.iter().map(|v| v * 1048576.0).collect();
        assert_bits(
            &format!("{name}: scale equivariance"),
            &h16_proj(&tab, &xs, rows, k, n, 4, false, None, None, false, &st),
            &scaled,
        );
        // a row's bits: in fewer rows, at every tile, in another order
        for cnt in [1usize, 17, 33] {
            assert_bits(
                &format!("{name}: {cnt} rows"),
                &h16_proj(&tab, &x, cnt, k, n, 4, false, None, None, false, &st),
                &base[..cnt * n],
            );
        }
        for mt in [1usize, 2, 8] {
            if raw::dense_h16_smem(mt, k) <= 101_376 {
                assert_bits(
                    &format!("{name}: tile {mt}"),
                    &h16_proj(&tab, &x, rows, k, n, mt, false, None, None, false, &st),
                    &base,
                );
            }
        }
        let perm: Vec<usize> = (0..rows).map(|j| (j * 29 + 5) % rows).collect();
        let xp: Vec<f32> = perm.iter().flat_map(|&j| x[j * k..(j + 1) * k].to_vec()).collect();
        let op = h16_proj(&tab, &xp, rows, k, n, 4, false, None, None, false, &st);
        for (j, &src) in perm.iter().enumerate() {
            assert_bits(&format!("{name}: permuted row {j}"), &op[j * n..(j + 1) * n], &base[src * n..(src + 1) * n]);
        }
        // the epilogue: relu2, then + add, then + add2 (also with add2 the output buffer itself)
        let mut rr = Rng(31 + i as u64);
        let (ad, ad2) = (rr.vec(rows * n, -1.0, 1.0), rr.vec(rows * n, -1.0, 1.0));
        let expect: Vec<f32> = (0..rows * n)
            .map(|o| {
                let q = base[o].max(0.0);
                (q * q + ad[o]) + ad2[o]
            })
            .collect();
        assert_bits(
            &format!("{name}: relu2 + add + add2"),
            &h16_proj(&tab, &x, rows, k, n, 4, true, Some(&ad), Some(&ad2), false, &st),
            &expect,
        );
        assert_bits(
            &format!("{name}: add2 in place"),
            &h16_proj(&tab, &x, rows, k, n, 4, true, Some(&ad), Some(&ad2), true, &st),
            &expect,
        );
        eprintln!("{name}: scale-equivariant; rows 1/17/33, tiles, permutation bit-equal; epilogue order exact");
    }
}

/// His shape's block with the residual fused into an fp16 fc2's epilogue (`forward_prefill_mma_resid`) is bit for bit
/// `forward_prefill_mma` followed by `x + out`; with the fp16 tensor-core switch off the block keeps the fp32 chain's dense projections
/// (and is then off the tensor-core block by no more than the P2 block limits).
#[test]
fn an_fp16_fc2_fuses_the_residual_bit_for_bit_and_the_switch_selects_the_chain() {
    gpu!();
    gb_cuda::nemotron_moe_prefill::set_any_width_prefill(true);
    let st = Stream::new().unwrap();
    let all = all_experts();
    let moe = PrefillMoe::tables(mixed_layer(214, &all, |e| width_of(e, 5), |e| width_of(e, 6), [0, 0, 4, 4])).unwrap();
    assert!(moe.supports_prefill_mma_resid(), "an fp16 fc2 on tensor cores takes the fused residual");
    let rows = 300;
    let mut s = PrefillScratch::new(rows).unwrap();
    let mut m = MmaScratch::new(rows).unwrap();
    let h = Rng(0x7E5).vec(rows * HIDDEN, -1.0, 1.0);
    let x0 = Rng(0x7E6).vec(rows * HIDDEN, -2.0, 2.0);
    mma::set_tile(4);
    let mix = run_mma(&moe, &mut s, &mut m, &h, rows, &st);
    let hd = DevBuf::from_slice(&h).unwrap();
    let mut xd = DevBuf::from_slice(&x0).unwrap();
    s.poison(&st).unwrap();
    m.poison(&st).unwrap();
    moe.forward_prefill_mma_resid(&mut s, &mut m, &hd, rows, &mut xd, &st).unwrap();
    st.sync().unwrap();
    let want: Vec<f32> = x0.iter().zip(&mix).map(|(a, b)| a + b).collect();
    assert_bits("fused residual against x + block", &xd.to_vec::<f32>().unwrap(), &want);
    mma::set_fp16_dense(false);
    assert!(!moe.supports_prefill_mma_resid(), "off: the fp16 fc2 is the fp32 chain's, not fused");
    let chain = run_mma(&moe, &mut s, &mut m, &h, rows, &st);
    mma::set_fp16_dense(true);
    let (mx, rms) = stats(&mix, &chain);
    eprintln!("fp16 dense on tensor cores against the fp32 chain inside the block: max {mx:.3e} rms {rms:.3e}");
    assert!(mx < 1e-2 && rms < 3e-3, "max {mx:e} rms {rms:e}");
    assert!(mix.iter().zip(&chain).any(|(a, b)| a.to_bits() != b.to_bits()), "the switch selects another arithmetic");
}

/// The tensor-core path's width launches at 4 bits give its 4-bit launch's bits: the 4-bit synthetic layer built as a MixedK table
/// (`set_force_any_width_prefill`: the any-width items, rotation and the K2 = 8 width launch) equals its default table, at 256 and
/// 1,024 rows.
#[test]
fn the_width_launches_at_4_bits_give_the_4_bit_launchs_bits() {
    gpu!();
    use gb_cuda::nemotron_moe_prefill::set_force_any_width_prefill;
    let st = Stream::new().unwrap();
    let rows = 1024;
    let moe = PrefillMoe::tables(synthetic_layer(161, &all_experts(), false).0).unwrap();
    set_force_any_width_prefill(true);
    let any = PrefillMoe::tables(synthetic_layer(161, &all_experts(), false).0).unwrap();
    set_force_any_width_prefill(false);
    let mut s = PrefillScratch::new(rows).unwrap();
    let mut m = MmaScratch::new(rows).unwrap();
    let h = Rng(0x4B17).vec(rows * HIDDEN, -1.0, 1.0);
    for n in [256usize, rows] {
        let a = run_mma(&moe, &mut s, &mut m, &h, n, &st);
        let b = run_mma(&any, &mut s, &mut m, &h, n, &st);
        assert_bits(&format!("{n} rows: the width launches against the 4-bit launch"), &b, &a);
    }
    eprintln!("tensor cores: the width launches at 4 bits == the 4-bit launch bitwise at 256 and 1024 rows");
}

/// MixedK, expert parallel (TP = 2 on one device): two ranks holding the even and the odd experts of his pack's shapes (mixed widths,
/// fp16 fc1 / fc2, shared 5 / 4), each the other's experts absent, the routed `y` summed between the down projection and the combine:
/// the summed block equals the single call bit for bit, on the tensor-core block (P2) and on the fp32 prefill path (P1).
#[test]
fn mixedk_expert_parallel_ranks_sum_to_the_single_call() {
    gpu!();
    gb_cuda::nemotron_moe_prefill::set_any_width_prefill(true);
    let st = Stream::new().unwrap();
    let rows = 256;
    let even: Vec<usize> = (0..EXPERTS).filter(|e| e % 2 == 0).collect();
    let odd: Vec<usize> = (0..EXPERTS).filter(|e| e % 2 == 1).collect();
    let mk = |present: &[usize]| {
        PrefillMoe::tables(mixed_layer(221, present, |e| width_of(e, 1), |e| width_of(e, 2), [0, 0, 5, 4])).unwrap()
    };
    let (full, r0, r1) = (mk(&all_experts()), mk(&even), mk(&odd));
    let mut s = PrefillScratch::new(rows).unwrap();
    let mut m = MmaScratch::new(rows).unwrap();
    let h = Rng(4243).vec(rows * HIDDEN, -1.0, 1.0);
    let hd = DevBuf::from_slice(&h).unwrap();
    for p2 in [true, false] {
        let single = if p2 { run_mma(&full, &mut s, &mut m, &h, rows, &st) } else { run_fma(&full, &mut s, &h, rows, &st) };
        let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
        let mut y1: Vec<f32> = Vec::new();
        {
            let mut keep = |y: &mut DevBuf, n: usize, st: &Stream| -> Result<(), gb_cuda::CudaError> {
                st.sync()?;
                let mut v = vec![0f32; n];
                y.download_range(0, &mut v)?;
                y1 = v;
                Ok(())
            };
            if p2 {
                r1.forward_prefill_mma_tp(&mut s, &mut m, &hd, rows, &mut out, &st, &mut keep).unwrap();
            } else {
                r1.forward_prefill_in_tp(&mut s, &hd, rows, &mut out, &st, &mut keep).unwrap();
            }
            st.sync().unwrap();
        }
        {
            let mut add = |y: &mut DevBuf, n: usize, st: &Stream| -> Result<(), gb_cuda::CudaError> {
                st.sync()?;
                let mut v = vec![0f32; n];
                y.download_range(0, &mut v)?;
                for (a, b) in v.iter_mut().zip(&y1) {
                    *a += *b;
                }
                y.upload(&v)?;
                Ok(())
            };
            if p2 {
                r0.forward_prefill_mma_tp(&mut s, &mut m, &hd, rows, &mut out, &st, &mut add).unwrap();
            } else {
                r0.forward_prefill_in_tp(&mut s, &hd, rows, &mut out, &st, &mut add).unwrap();
            }
            st.sync().unwrap();
        }
        let what = if p2 { "P2 tensor cores" } else { "P1 fp32" };
        assert_bits(&format!("MixedK {what}: rank 0 + rank 1 against the single call"), &out.to_vec::<f32>().unwrap(), &single);
        eprintln!(
            "MixedK expert parallel ({what}), 2 ranks (even / odd experts), {rows} rows: the summed block equals the single call"
        );
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Gate 2: a column split (a TP rank's range of N) is exact

#[test]
fn column_halves_equal_one_call() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 100;
    for (i, &(name, k, n, act)) in SHAPES.iter().enumerate() {
        let p = proj1(k, n, 7100 + i as u64);
        let x = inputs(31 + i as u64, rows, k, true);
        let xd = DevBuf::from_slice(&x).unwrap();
        let a16 = DevBuf::zeroed(rows * k * 2).unwrap();
        let ainv = DevBuf::zeroed(rows * k / 128 * 4).unwrap();
        let full = DevBuf::from_slice(&vec![f32::NAN; rows * n]).unwrap();
        let halves = DevBuf::from_slice(&vec![f32::NAN; rows * n]).unwrap();
        let ncb = n / 128;
        let cut = ncb / 2;
        for mt in [1usize, 4, 8] {
            unsafe {
                raw::rotate(xd.ptr(), k, 1, core::ptr::null(), p.tab.entry(0), rows, k, a16.ptr(), ainv.ptr(), false, &st)
                    .unwrap();
                let g = |nb0: usize, nc: usize, out: &DevBuf| {
                    raw::gemm_dense(
                        a16.ptr(),
                        ainv.ptr(),
                        rows,
                        mt,
                        p.tab.entry(0),
                        core::ptr::null(),
                        k,
                        n,
                        nb0,
                        nc,
                        out.ptr(),
                        core::ptr::null_mut(),
                        core::ptr::null_mut(),
                        act,
                        core::ptr::null(),
                        false,
                        &st,
                    )
                    .unwrap()
                };
                g(0, ncb, &full);
                g(0, cut, &halves);
                g(cut, ncb - cut, &halves);
            }
            st.sync().unwrap();
            assert_bits(
                &format!("{name}, tile {mt}: columns 0..{} + {}..{} against one call", cut * 128, cut * 128, n),
                &halves.to_vec::<f32>().unwrap(),
                &full.to_vec::<f32>().unwrap(),
            );
        }
        eprintln!("{name} (K {k}, N {n}): two column ranges cut at {} equal one call bit for bit (tiles 1, 4, 8)", cut * 128);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Gate 3: the block is deterministic and row-independent

#[test]
fn the_block_is_deterministic_and_row_independent() {
    gpu!();
    let st = Stream::new().unwrap();
    let (w, _) = synthetic_layer(131, &all_experts(), false);
    let moe = PrefillMoe::tables(w).unwrap();
    let rows = 2048;
    let mut s = PrefillScratch::new(rows).unwrap();
    let mut m = MmaScratch::new(rows).unwrap();
    let h = Rng(0x5EED).vec(rows * HIDDEN, -1.0, 1.0);
    mma::set_tile(4);
    let base = run_mma(&moe, &mut s, &mut m, &h, rows, &st);
    assert!(base.iter().all(|v| v.is_finite()) && base.iter().any(|v| *v != 0.0), "the output is not finite or all zero");
    let items = m.download_items(&st).unwrap();
    let most = items.iter().map(|it| it[2]).max().unwrap();
    eprintln!("2048 rows: {} items of at most 64 rows, the largest {most}", items.len());
    // 5 runs
    for run in 1..5 {
        let again = run_mma(&moe, &mut s, &mut m, &h, rows, &st);
        assert_bits(&format!("run {run} against run 0"), &again, &base);
    }
    eprintln!("5 runs at 2048 rows: bit-equal");
    // slice widths
    for width in [256usize, 512, 1024] {
        let mut cat = Vec::with_capacity(rows * HIDDEN);
        for c in 0..rows / width {
            cat.extend(run_mma(&moe, &mut s, &mut m, &h[c * width * HIDDEN..], width, &st));
        }
        assert_bits(&format!("slices of {width} against one slice of 2048"), &cat, &base);
        eprintln!("slices of {width} rows concatenated: bit-equal to one 2048-row slice");
    }
    // a permutation of 1024 rows
    let n = 1024;
    let mut perm: Vec<usize> = (0..n).collect();
    let mut r = Rng(99);
    for i in (1..n).rev() {
        perm.swap(i, (r.next_u64() % (i as u64 + 1)) as usize);
    }
    let hp: Vec<f32> = perm.iter().flat_map(|&i| h[i * HIDDEN..(i + 1) * HIDDEN].to_vec()).collect();
    let op = run_mma(&moe, &mut s, &mut m, &hp, n, &st);
    for (j, &i) in perm.iter().enumerate() {
        assert_bits(
            &format!("permuted row {j} (row {i})"),
            &op[j * HIDDEN..(j + 1) * HIDDEN],
            &base[i * HIDDEN..(i + 1) * HIDDEN],
        );
    }
    eprintln!("1024 rows permuted: every row has its bits");
    // padding: odd row counts and duplicated rows
    for cnt in [1usize, 17, 333, 1000] {
        let o = run_mma(&moe, &mut s, &mut m, &h, cnt, &st);
        assert_bits(&format!("{cnt} rows against the first {cnt} of 2048"), &o, &base[..cnt * HIDDEN]);
    }
    let dup: Vec<f32> = (0..256).flat_map(|r| h[(r % 3) * HIDDEN..(r % 3 + 1) * HIDDEN].to_vec()).collect();
    let od = run_mma(&moe, &mut s, &mut m, &dup, 256, &st);
    for r in 0..256 {
        assert_bits(
            &format!("duplicate row {r}"),
            &od[r * HIDDEN..(r + 1) * HIDDEN],
            &base[(r % 3) * HIDDEN..(r % 3 + 1) * HIDDEN],
        );
    }
    eprintln!("1, 17, 333, 1000 rows and 256 copies of 3 rows (experts of 256 rows: several items each): bit-equal");
    // every item tile
    for mt in [1u8, 2, 8] {
        mma::set_tile(mt);
        let o = run_mma(&moe, &mut s, &mut m, &h, rows, &st);
        assert_bits(&format!("tile {mt} against tile 4"), &o, &base);
    }
    mma::set_tile(4);
    eprintln!("item tiles of 16, 32, 128 rows: bit-equal to 64");

    // the shared expert in line instead of on the side stream
    gb_cuda::nemotron_moe_prefill::set_side_stream(false);
    let o = run_mma(&moe, &mut s, &mut m, &h, rows, &st);
    gb_cuda::nemotron_moe_prefill::set_side_stream(true);
    assert_bits("shared expert in line", &o, &base);
    eprintln!("shared expert in line: bit-equal");
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Gate 4: the block against the fp32 path and the CPU reference

#[test]
fn the_block_against_the_fp32_path_and_the_cpu_reference() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut r = Rng(31);
    let base = r.vec(4 * HIDDEN, -1.0, 1.0);
    let (_, probe) = synthetic_layer(41, &[], true);
    let probe = probe.unwrap();
    let mut picks = Vec::new();
    let mut wts = Vec::new();
    for t in 0..4 {
        let (idx, w, gap) = route_cpu(&logits_cpu(&base[t * HIDDEN..(t + 1) * HIDDEN], &probe.gate), &probe.bias, SCALE);
        assert!(gap > 1e-5, "row {t}: a near tie at the boundary");
        picks.push(idx);
        wts.push(w);
    }
    let used: Vec<usize> = picks.iter().flatten().copied().collect::<BTreeSet<usize>>().into_iter().collect();
    let (w, host) = synthetic_layer(41, &used, true);
    let host = host.unwrap();
    let moe = PrefillMoe::tables(w).unwrap();
    let rows = 128;
    let mut s = PrefillScratch::new(rows).unwrap();
    let mut m = MmaScratch::new(rows).unwrap();
    let h: Vec<f32> = (0..rows).flat_map(|r| base[(r % 4) * HIDDEN..(r % 4 + 1) * HIDDEN].to_vec()).collect();
    let want = block_cpu(&host, &base, 4, &picks, &wts);
    let got = run_mma(&moe, &mut s, &mut m, &h, rows, &st);
    let fma = run_fma(&moe, &mut s, &h, rows, &st);
    let (a, b) = stats(&got[..4 * HIDDEN], &want);
    let (c, d) = stats(&fma[..4 * HIDDEN], &want);
    let (e, f) = stats(&got, &fma);
    mma::set_bf16(true);
    let gbf = run_mma(&moe, &mut s, &mut m, &h, rows, &st);
    mma::set_bf16(false);
    let (g, hh) = stats(&gbf[..4 * HIDDEN], &want);
    eprintln!("block, {} experts, 128 rows (4 x 32) against the CPU reference: fp16 MMA max {a:.3e} rms {b:.3e}; bf16 MMA max {g:.3e} rms {hh:.3e}; fp32 path max {c:.3e} rms {d:.3e}; fp16 MMA against the fp32 path max {e:.3e} rms {f:.3e}", used.len());
    assert!(a < 1e-2 && b < 3e-3, "the MMA block is off the CPU reference: max {a:e} rms {b:e}");
    for r in 4..rows {
        assert_bits(&format!("tile row {r}"), &got[r * HIDDEN..(r + 1) * HIDDEN], &got[(r % 4) * HIDDEN..(r % 4 + 1) * HIDDEN]);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Gate 5: expert-parallel ranks summed equal the single call

#[test]
fn expert_parallel_ranks_sum_to_the_single_call() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 256;
    let even: Vec<usize> = (0..EXPERTS).filter(|e| e % 2 == 0).collect();
    let odd: Vec<usize> = (0..EXPERTS).filter(|e| e % 2 == 1).collect();
    let full = PrefillMoe::tables(synthetic_layer(151, &all_experts(), false).0).unwrap();
    let r0 = PrefillMoe::tables(synthetic_layer(151, &even, false).0).unwrap();
    let r1 = PrefillMoe::tables(synthetic_layer(151, &odd, false).0).unwrap();
    let mut s = PrefillScratch::new(rows).unwrap();
    let mut m = MmaScratch::new(rows).unwrap();
    let h = Rng(4242).vec(rows * HIDDEN, -1.0, 1.0);
    let single = run_mma(&full, &mut s, &mut m, &h, rows, &st);
    let hd = DevBuf::from_slice(&h).unwrap();
    let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
    // rank 1 first: keep its y
    let mut y1: Vec<f32> = Vec::new();
    {
        let mut keep = |y: &mut DevBuf, n: usize, st: &Stream| -> Result<(), gb_cuda::CudaError> {
            st.sync()?;
            let mut v = vec![0f32; n];
            y.download_range(0, &mut v)?;
            y1 = v;
            Ok(())
        };
        r1.forward_prefill_mma_tp(&mut s, &mut m, &hd, rows, &mut out, &st, &mut keep).unwrap();
        st.sync().unwrap();
    }
    // rank 0: y += y1 (the all-reduce of two ranks)
    {
        let mut add = |y: &mut DevBuf, n: usize, st: &Stream| -> Result<(), gb_cuda::CudaError> {
            st.sync()?;
            let mut v = vec![0f32; n];
            y.download_range(0, &mut v)?;
            for (a, b) in v.iter_mut().zip(&y1) {
                *a += *b;
            }
            y.upload(&v)?;
            Ok(())
        };
        r0.forward_prefill_mma_tp(&mut s, &mut m, &hd, rows, &mut out, &st, &mut add).unwrap();
        st.sync().unwrap();
    }
    assert_bits("rank 0 + rank 1 against the single call", &out.to_vec::<f32>().unwrap(), &single);
    eprintln!("expert parallel, 2 ranks (even / odd experts), 256 rows: the summed block equals the single call bit for bit");
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
    fn ms_to(&self, later: &Ev) -> f64 {
        let mut ms = 0f32;
        assert_eq!(unsafe { ev::cudaEventSynchronize(later.0) }, 0);
        assert_eq!(unsafe { ev::cudaEventElapsedTime(&mut ms, self.0, later.0) }, 0);
        ms as f64
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

fn env_list(name: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(name) {
        Ok(v) => v.split(',').map(|s| s.trim().parse().unwrap_or_else(|_| panic!("{name}: a comma-separated list"))).collect(),
        Err(_) => default.to_vec(),
    }
}

/// Whole-block time at each row count: the default fp32 path, and the tensor-core path for each tile, with the dense projections on
/// tensor cores and (`/fp32 dense`) on the default kernels. L2 is flushed (64 MB written) before every timed call; median of `reps`.
/// "tok/s 40 layers" is rows / (40 x block ms): the MoE blocks of a slice alone.
fn bench_block(label: &str, moe: &PrefillMoe, hsrc: &dyn Fn(usize, usize) -> Vec<f32>, reps: usize) {
    let rows_list = env_list("REXL3_NEMO_PERF_ROWS", &[256, 512, 1024, 2048, 4096]);
    if let Ok(v) = std::env::var("REXL3_NEMO_MMA_ROUTER") {
        mma::set_router_mma(v == "1");
        eprintln!("router on tensor cores: {}", v == "1");
    }
    let tiles = env_list("REXL3_NEMO_MMA_TILES", &[4]);
    let st = Stream::new_nonblocking().unwrap();
    let max = *rows_list.iter().max().unwrap();
    let mut s = PrefillScratch::new(max).unwrap();
    let mut m = MmaScratch::new(max).unwrap();
    let flush = DevBuf::zeroed(64 << 20).unwrap();
    let (a, b) = (Ev::new(), Ev::new());
    eprintln!("\n=== {label}: whole MoE block, L2 flushed before each call, median of {reps} ===");
    eprintln!("{:>5} {:<22} {:>9} {:>11} {:>14} {:>9}", "rows", "path", "ms", "us/row", "tok/s 40 lay.", "x fp32");
    for &rows in &rows_list {
        let hs: Vec<DevBuf> = (0..2).map(|i| DevBuf::from_slice(&hsrc(rows, i)).unwrap()).collect();
        let mut out = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
        let mut time = |f: &mut dyn FnMut(&DevBuf, &mut DevBuf)| -> f64 {
            f(&hs[0], &mut out);
            st.sync().unwrap();
            let mut t = Vec::new();
            for i in 0..reps {
                flush.memset_async_view(&st);
                a.record(&st);
                f(&hs[i % 2], &mut out);
                b.record(&st);
                t.push(a.ms_to(&b));
            }
            median(&mut t)
        };
        let t_fma = time(&mut |h, o| moe.forward_prefill_in(&mut s, h, rows, o, &st).unwrap());
        let line = |name: &str, ms: f64| {
            eprintln!(
                "{rows:>5} {name:<22} {ms:>9.3} {:>11.2} {:>14.0} {:>9.2}",
                ms * 1000.0 / rows as f64,
                rows as f64 / (40.0 * ms * 1e-3),
                t_fma / ms
            );
        };
        line("fp32 (default)", t_fma);
        for &mt in &tiles {
            mma::set_tile(mt as u8);
            mma::set_dense(true);
            let t = time(&mut |h, o| moe.forward_prefill_mma(&mut s, &mut m, h, rows, o, &st).unwrap());
            line(&format!("mma tile {mt}"), t);
            mma::set_dense(false);
            let t = time(&mut |h, o| moe.forward_prefill_mma(&mut s, &mut m, h, rows, o, &st).unwrap());
            line(&format!("mma tile {mt} /fp32 dense"), t);
            mma::set_dense(true);
        }
        mma::set_tile(mma::DEFAULT_TILE);
    }
}

trait FlushView {
    fn memset_async_view(&self, st: &Stream);
}

impl FlushView for DevBuf {
    fn memset_async_view(&self, st: &Stream) {
        extern "C" {
            fn cudaMemsetAsync(p: *mut c_void, v: i32, n: usize, s: *mut c_void) -> i32;
        }
        assert_eq!(unsafe { cudaMemsetAsync(self.ptr(), 1, self.bytes(), st.raw()) }, 0);
    }
}

#[test]
#[ignore = "timing"]
fn mma_bench_synthetic_block() {
    gpu!();
    let (w, _) = synthetic_layer(401, &all_experts(), false);
    let moe = PrefillMoe::tables(w).unwrap();
    bench_block(
        "synthetic layer (512 experts, 4-bit Mul1; random gate: near-uniform routing)",
        &moe,
        &|rows, i| Rng(8000 + 31 * i as u64 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0),
        7,
    );
}

/// Per-width P2 throughput (v0.1.2 lane K): the whole block with all 512 experts at K bits (K = 2 .. 8; dense 4-bit, so only the
/// routed experts' width moves) on the tensor-core path's width launch, against the fp32 path; then his pack's shape (experts at 2 to 6
/// bits, fp16 fc1 and fc2 on the fp32 kernels, shared 5 / 4). `REXL3_NEMO_PERF_WIDTHS` picks the widths, `REXL3_NEMO_PERF_ROWS` the rows.
#[test]
#[ignore = "timing"]
fn mma_bench_widths() {
    gpu!();
    gb_cuda::nemotron_moe_prefill::set_any_width_prefill(true);
    let all = all_experts();
    let widths = env_list("REXL3_NEMO_PERF_WIDTHS", &[2, 3, 4, 5, 6, 7, 8]);
    let h = |rows: usize, i: usize| Rng(8100 + 31 * i as u64 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0);
    for k in widths {
        let moe = PrefillMoe::tables(mixed_layer(600 + k as u64, &all, |_| k, |_| k, [4; 4])).unwrap();
        bench_block(&format!("512 experts at {k} bits, dense 4-bit (synthetic)"), &moe, &h, 7);
    }
    let moe = PrefillMoe::tables(mixed_layer(611, &all, |e| width_of(e, 1), |e| width_of(e, 2), [0, 0, 5, 4])).unwrap();
    bench_block("his shapes: experts at 2-6 bits, fp16 fc1 / fc2, shared 5 / 4 (synthetic)", &moe, &h, 7);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The pack's layers (ignored; REXL3_NEMO_OMNI_DIR, and REXL3_NEMO_ORACLE_DIR for real layer inputs)

fn plain_u16(ck: &Checkpoint, name: &str) -> Vec<u16> {
    let e = ck.plain.get(name).unwrap_or_else(|| panic!("the pack has no plain tensor {name}"));
    ck.read_u16(e).unwrap()
}

fn plain_f32(ck: &Checkpoint, name: &str) -> Vec<f32> {
    let e = ck.plain.get(name).unwrap_or_else(|| panic!("the pack has no plain tensor {name}"));
    ck.read_floats(e).unwrap().into_iter().map(|v| v as f32).collect()
}

fn pack_proj(ck: &Checkpoint, prefix: &str, k: usize, n: usize) -> Proj {
    if ck.groups.contains_key(prefix) {
        Proj::exl3_from_group(&ck.load_group(prefix).unwrap_or_else(|e| panic!("{prefix}: {e}"))).unwrap()
    } else {
        Proj::bf16(k, n, &plain_u16(ck, &format!("{prefix}.weight"))).unwrap()
    }
}

fn pack_layer(ck: &Checkpoint, layer: usize) -> Weights {
    let mp = format!("language_model.backbone.layers.{layer}.mixer");
    Weights {
        gate: DevBuf::from_slice(&plain_u16(ck, &format!("{mp}.gate.weight"))).unwrap(),
        gate_dtype: gb_cuda::nemotron_moe::GateDtype::Bf16,
        router_bias: DevBuf::from_slice(&plain_f32(ck, &format!("{mp}.gate.e_score_correction_bias"))).unwrap(),
        scale: SCALE,
        fc1: pack_proj(ck, &format!("{mp}.fc1_latent_proj"), HIDDEN, LATENT),
        fc2: pack_proj(ck, &format!("{mp}.fc2_latent_proj"), LATENT, HIDDEN),
        shared_up: pack_proj(ck, &format!("{mp}.shared_experts.up_proj"), HIDDEN, SHARED_INTER),
        shared_down: pack_proj(ck, &format!("{mp}.shared_experts.down_proj"), SHARED_INTER, HIDDEN),
        up: (0..EXPERTS).map(|e| pack_proj(ck, &format!("{mp}.experts.{e}.up_proj"), LATENT, EXPERT_INTER)).collect(),
        down: (0..EXPERTS).map(|e| pack_proj(ck, &format!("{mp}.experts.{e}.down_proj"), EXPERT_INTER, LATENT)).collect(),
    }
}

fn env_dir(name: &str) -> Option<PathBuf> {
    let d = std::env::var_os(name).map(PathBuf::from);
    if d.is_none() {
        eprintln!("not run: {name} is not set");
    }
    d
}

/// The residual stream entering text layer `layer` (`layers0/m{layer:03}.f16`, 1024 tokens of the oracle) and its RMS norm with the
/// layer's norm weight (eps 1e-5): (residual, h), both `[1024, 4096]`.
fn real_inputs(ck: &Checkpoint, oracle: &std::path::Path, layer: usize) -> (Vec<f32>, Vec<f32>) {
    let b = std::fs::read(oracle.join("layers0").join(format!("m{layer:03}.f16"))).unwrap();
    let x: Vec<f32> = b.chunks_exact(2).map(|c| gb_exl3::f16::f16_to_f64(u16::from_le_bytes([c[0], c[1]])) as f32).collect();
    assert_eq!(x.len(), 1024 * HIDDEN);
    let wn = bf16_f32(&plain_u16(ck, &format!("language_model.backbone.layers.{layer}.norm.weight")));
    let mut h = vec![0f32; x.len()];
    for r in 0..1024 {
        let row = &x[r * HIDDEN..(r + 1) * HIDDEN];
        let ms = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / HIDDEN as f64;
        let inv = 1.0 / (ms + 1e-5).sqrt();
        for j in 0..HIDDEN {
            h[r * HIDDEN + j] = (row[j] as f64 * inv) as f32 * wn[j];
        }
    }
    (x, h)
}

/// Quality on real layer inputs: pack MoE layers, the oracle's residual stream entering them (1024 tokens), RMS-normed. The fp32
/// default path is the reference; the tensor-core block in fp16 and in bf16. Reports the block output's error and the induced error
/// on the next block's input (residual + block output).
#[test]
#[ignore = "pack and oracle"]
fn mma_quality_on_real_layer_inputs() {
    gpu!();
    let Some(dir) = env_dir("REXL3_NEMO_OMNI_DIR") else { return };
    let Some(oracle) = env_dir("REXL3_NEMO_ORACLE_DIR") else { return };
    let ck = scan(&dir, true).unwrap();
    let st = Stream::new().unwrap();
    let rows = 1024;
    let mut s = PrefillScratch::new(rows).unwrap();
    let mut m = MmaScratch::new(rows).unwrap();
    let layers = env_list("REXL3_NEMO_MMA_LAYERS", &[1, 26, 52, 85]);
    // REXL3_NEMO_MMA_ROUTER=0|1: the router logits on the fp32 kernel or on tensor cores
    if let Ok(v) = std::env::var("REXL3_NEMO_MMA_ROUTER") {
        mma::set_router_mma(v == "1");
        eprintln!("router on tensor cores: {}", v == "1");
    }
    eprintln!(
        "\n{:>5} {:<5} | {:>10} {:>10} {:>9} | {:>10} {:>10} | {:>10} {:>10} | {:>12}",
        "layer", "ops", "out max", "out rms", "cos", "next max", "next rms", "h absmax", "same picks", "worst row rms"
    );
    for &l in &layers {
        let moe = PrefillMoe::tables(pack_layer(&ck, l)).unwrap();
        let (x, h) = real_inputs(&ck, &oracle, l);
        let reference = run_fma(&moe, &mut s, &h, rows, &st);
        let (pick_ref, _) = s.download_route(rows, &st).unwrap();
        for bfm in [false, true] {
            mma::set_bf16(bfm);
            let got = run_mma(&moe, &mut s, &mut m, &h, rows, &st);
            let (pick, _) = s.download_route(rows, &st).unwrap();
            let rows_same = (0..rows).filter(|&t| pick[t * TOPK..(t + 1) * TOPK] == pick_ref[t * TOPK..(t + 1) * TOPK]).count();
            let (a, b) = stats(&got, &reference);
            let (cos, _) = common::cos_rel_rms(&got, &reference);
            let next_ref: Vec<f32> = x.iter().zip(&reference).map(|(a, b)| a + b).collect();
            let next_got: Vec<f32> = x.iter().zip(&got).map(|(a, b)| a + b).collect();
            let (c, d) = stats(&next_got, &next_ref);
            let worst = (0..rows)
                .map(|r| stats(&got[r * HIDDEN..(r + 1) * HIDDEN], &reference[r * HIDDEN..(r + 1) * HIDDEN]).1)
                .fold(0f64, f64::max);
            let ha = h.iter().fold(0f32, |m, v| m.max(v.abs()));
            eprintln!(
                "{l:>5} {:<5} | {a:>10.3e} {b:>10.3e} {cos:>9.6} | {c:>10.3e} {d:>10.3e} | {ha:>10.3e} {:>10} | {worst:>12.3e}",
                if bfm { "bf16" } else { "fp16" },
                format!("{rows_same}/{rows}")
            );
        }
        mma::set_bf16(false);
    }
}

#[test]
#[ignore = "timing, pack"]
fn mma_bench_pack_layer() {
    gpu!();
    let Some(dir) = env_dir("REXL3_NEMO_OMNI_DIR") else { return };
    let ck = scan(&dir, true).unwrap();
    let layer = env_list("REXL3_NEMO_MMA_LAYERS", &[1])[0];
    let moe = PrefillMoe::tables(pack_layer(&ck, layer)).unwrap();
    // real layer inputs when the oracle is there (1024 tokens, repeated with a small perturbation per copy beyond 1024 rows)
    let real = std::env::var_os("REXL3_NEMO_ORACLE_DIR").map(|o| real_inputs(&ck, std::path::Path::new(&o), layer).1);
    let src = move |rows: usize, i: usize| -> Vec<f32> {
        match &real {
            Some(h) => {
                let mut r = Rng(17 + i as u64);
                (0..rows)
                    .flat_map(|t| {
                        let base = &h[(t % 1024) * HIDDEN..(t % 1024 + 1) * HIDDEN];
                        let f = if t < 1024 && i == 0 { 1.0 } else { 1.0 + r.uniform(-0.05, 0.05) };
                        base.iter().map(|v| v * f).collect::<Vec<f32>>()
                    })
                    .collect()
            }
            None => Rng(8000 + 31 * i as u64 + rows as u64).vec(rows * HIDDEN, -1.0, 1.0),
        }
    };
    bench_block(&format!("pack layer {layer} ({} slow experts)", moe.slow_experts()), &moe, &src, 7);
}
