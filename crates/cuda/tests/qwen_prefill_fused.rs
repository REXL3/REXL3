//! Qwen3.8-27B prefill on the fused tensor-core dense GEMM (`gb_cuda::glm_p2`, driven by `gb_model::prefill_p2`), on SYNTHETIC Mul1 4-bit EXL3
//! weights of the 27B pack's shapes (`common::exl3_synth`): the pinned cuBLAS path (`Exl3Weights::forward_prefill_pinned`, what a Qwen prompt
//! chunk ran before) against the fused kernel, with the bf16 activations and bf16 outputs the model passes.
//!
//! * `fused_rows_are_bit_identical_in_any_call_on_the_qwen_shapes`: a row's bits do not depend on the call's width, its boundary or the row order.
//! * `the_chain_order_equals_the_pinned_cublas_path_bit_for_bit_on_the_qwen_shapes`: the fused GEMM (single-chain K order, what the model
//!   runs) gives the pinned path's bf16 output bit for bit on every shape at 1 to 4,096 rows: the switch changes speed and nothing else.
//! * `both_k_orders_are_as_close_to_fp64_as_the_decode_path_on_the_qwen_shapes`: against an fp64 reference, the rms error and the signed
//!   shrink of the chain and of the per-128 fold (the alternative that would move bits) beside the decode path's.
//! * `bench_*` (ignored): ms and TFLOP/s per call, pinned (reconstruct + cuBLAS + rotations) against fused, at the server's chunk widths.
//!
//! Run:
//!
//!     cargo test -p gb-cuda --release --test qwen_prefill_fused -- --test-threads=1 --nocapture
//!     cargo test -p gb-cuda --release --test qwen_prefill_fused -- --ignored --test-threads=1 --nocapture bench
#![cfg(not(gb_no_cuda))]
mod common;

use common::exl3_synth::synth_group;
use common::Rng;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::glm_p2::{self, Variant};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::Codebook;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

/// (name, K, N): the dense EXL3 linears of the 4.00 bpw pack's text layers (all Mul1, 4 bits).
const SHAPES: &[(&str, usize, usize)] = &[
    ("mlp gate/up", 5120, 17408),
    ("mlp down", 17408, 5120),
    ("attn q", 5120, 12288),
    ("gdn qkv", 5120, 10240),
    ("gdn z", 5120, 6144),
    ("out/o", 6144, 5120),
    ("attn k/v", 5120, 1024),
];

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn layer(k: usize, n: usize, seed: u64) -> Exl3Weights {
    Exl3Weights::from_loaded(&synth_group(k, n, 4, Codebook::Mul1, seed)).unwrap()
}

fn acts(seed: u64, rows: usize, k: usize) -> Vec<u16> {
    let mut r = Rng(seed);
    (0..rows * k).map(|_| bf16_bits(r.uniform(-1.5, 1.5))).collect()
}

fn fused(w: &Exl3Weights, x: &DevBuf, rows: usize, st: &Stream) -> Vec<u16> {
    let mut y = DevBuf::from_slice(&vec![0x7fc1u16; rows * w.n]).unwrap();
    let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, w.k)).unwrap();
    glm_p2::linear_variant(w, x, &mut y, rows, Dtype::BF16, Dtype::BF16, &mut ws, Variant::Chain, 0, st).unwrap();
    st.sync().unwrap();
    y.to_vec::<u16>().unwrap()
}

fn pinned(w: &Exl3Weights, x: &DevBuf, rows: usize, st: &Stream) -> Vec<u16> {
    let mut y = DevBuf::zeroed(rows * w.n * 2).unwrap();
    let mut scratch = DevBuf::alloc(w.prefill_scratch_bytes(rows)).unwrap();
    w.forward_prefill_pinned(x, &mut y, rows, Dtype::BF16, &mut scratch, st).unwrap();
    st.sync().unwrap();
    y.to_vec::<u16>().unwrap()
}

fn upload(x: &[u16]) -> DevBuf {
    let mut d = DevBuf::zeroed_view(x.len() * 2).unwrap();
    d.upload(x).unwrap();
    d
}

#[test]
fn fused_rows_are_bit_identical_in_any_call_on_the_qwen_shapes() {
    gpu!();
    let st = Stream::new().unwrap();
    for (i, &(name, k, n)) in SHAPES.iter().enumerate() {
        assert!(glm_p2::supported(&layer(k, n, 1)), "{name}");
        let w = layer(k, n, 100 + i as u64);
        let rows = 300usize;
        let xb = acts(7 + i as u64, rows, k);
        let x = upload(&xb);
        let whole = fused(&w, &x, rows, &st);
        assert_eq!(whole, fused(&w, &x, rows, &st), "{name}: repeat");
        for piece in [256usize, 129, 128, 100, 17, 1] {
            let mut at = 0;
            while at < rows {
                let m = piece.min(rows - at);
                let xs = upload(&xb[at * k..(at + m) * k]);
                let got = fused(&w, &xs, m, &st);
                assert_eq!(got[..], whole[at * n..(at + m) * n], "{name}: piece {piece} at row {at}");
                at += m;
            }
        }
        // a row permutation: the same rows in another order and position
        let mut xr = Vec::with_capacity(xb.len());
        for r in (0..rows).rev() {
            xr.extend_from_slice(&xb[r * k..(r + 1) * k]);
        }
        let rev = fused(&w, &upload(&xr), rows, &st);
        for r in 0..rows {
            assert_eq!(rev[(rows - 1 - r) * n..(rows - r) * n], whole[r * n..(r + 1) * n], "{name}: reversed row {r}");
        }
    }
}

fn fwht(v: &mut [f64]) {
    let mut h = 1;
    while h < v.len() {
        for i in (0..v.len()).step_by(2 * h) {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// y rows of the exact layer in f64: ((x * suh) H / sqrt 128) . W_q . H / sqrt 128 * svh, W_q the exact dequantized weights.
fn reference(g: &gb_exl3::checkpoint::LoadedGroup, wq: &[u16], x: &[f64], rows: &[usize]) -> Vec<f64> {
    let (k, n) = (g.group.k, g.group.n);
    let lut: Vec<f64> = (0..=u16::MAX).map(f16_to_f64).collect();
    let s = 1.0 / 128f64.sqrt();
    let mut out = Vec::with_capacity(rows.len() * n);
    for &r in rows {
        let mut xr: Vec<f64> = (0..k).map(|i| x[r * k + i] * g.suh[i]).collect();
        xr.chunks_mut(128).for_each(fwht);
        let mut c = vec![0f64; n];
        for i in 0..k {
            let a = xr[i] * s;
            let wrow = &wq[i * n..(i + 1) * n];
            for j in 0..n {
                c[j] += a * lut[wrow[j] as usize];
            }
        }
        c.chunks_mut(128).for_each(fwht);
        out.extend((0..n).map(|j| c[j] * s * g.svh[j]));
    }
    out
}

/// (rms error / rms ref, signed shrink: mean of err x sign(ref) / rms ref; negative = outputs shrunk toward zero) over the sampled rows.
fn err(refv: &[f64], got: &[f32], rows: &[usize], n: usize) -> (f64, f64) {
    let (mut ee, mut rr, mut shrink) = (0f64, 0f64, 0f64);
    let cnt = (rows.len() * n) as f64;
    for (i, &r) in rows.iter().enumerate() {
        for j in 0..n {
            let (w, g) = (refv[i * n + j], got[r * n + j] as f64);
            ee += (g - w) * (g - w);
            rr += w * w;
            shrink += (g - w) * w.signum();
        }
    }
    ((ee / rr).sqrt(), shrink / cnt / (rr / cnt).sqrt())
}

/// fp16 in, fp32 out: the K order in isolation (no bf16 rounding of the output), against the decode path and an fp64 reference.
#[test]
fn both_k_orders_are_as_close_to_fp64_as_the_decode_path_on_the_qwen_shapes() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 160usize;
    let sample: Vec<usize> = (0..rows).step_by(53).collect();
    for (i, &(name, k, n)) in SHAPES.iter().enumerate().filter(|(_, s)| s.1 * s.2 >= 5120 * 6144) {
        let g = synth_group(k, n, 4, Codebook::Mul1, 1000 + i as u64);
        let w = Exl3Weights::from_loaded(&g).unwrap();
        let mut wqd = DevBuf::alloc(k * n * 2).unwrap();
        w.reconstruct(&mut wqd, &st).unwrap();
        st.sync().unwrap();
        let wq = wqd.to_vec::<u16>().unwrap();
        let mut r = Rng(77 + i as u64);
        let xb: Vec<u16> = (0..rows * k)
            .map(|_| {
                let v = r.uniform(-1.0, 1.0) as f64;
                f64_to_f16(if r.next_u64() % 32 == 0 { 4.0 * v } else { v })
            })
            .collect();
        let xv: Vec<f64> = xb.iter().map(|&b| f16_to_f64(b)).collect();
        let x = upload(&xb);
        let refv = reference(&g, &wq, &xv, &sample);
        // the decode path in calls of at most 128 rows
        let yd = DevBuf::zeroed_view(rows * n * 4).unwrap();
        let mut dws = DevBuf::alloc(w.workspace_bytes(128)).unwrap();
        for s0 in (0..rows).step_by(128) {
            let m = (rows - s0).min(128);
            let xs = x.subview(s0 * k * 2, m * k * 2);
            let mut ys = yd.subview(s0 * n * 4, m * n * 4);
            w.forward(&xs, &mut ys, m, Dtype::F16, Dtype::F32, &mut dws, &st).unwrap();
        }
        st.sync().unwrap();
        let (dr, dk) = err(&refv, &yd.to_vec::<f32>().unwrap(), &sample, n);
        let mut line = format!("{name:12} K {k:5} N {n:5}: decode path rms {dr:.2e} shrink {dk:+.1e}");
        for v in [Variant::Chain, Variant::Fold128] {
            let mut y = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * n]).unwrap();
            let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, k)).unwrap();
            glm_p2::linear_variant(&w, &x, &mut y, rows, Dtype::F16, Dtype::F32, &mut ws, v, 0, &st).unwrap();
            st.sync().unwrap();
            let (pr, pk) = err(&refv, &y.to_vec::<f32>().unwrap(), &sample, n);
            line += &format!(" | {} rms {pr:.2e} shrink {pk:+.1e}", v.id());
            assert!(pr <= 1.5 * dr + 1e-7 && pr <= 5e-4, "{name} {v:?}: rms error {pr:e} against the decode path's {dr:e}");
        }
        eprintln!("{line}");
    }
}

/// MEASURED finding, gated: the single-chain K order (`Variant::Chain`) gives the pinned cuBLAS path's bf16 output bit for bit on every
/// shape of the pack at every row count tried (cuBLAS's pinned algorithm is itself one fp16 MMA chain over K with fp32 accumulation, and the
/// rotations round at the same points). Only the per-128 fold (`Variant::Fold128`) moves bits.
#[test]
fn the_chain_order_equals_the_pinned_cublas_path_bit_for_bit_on_the_qwen_shapes() {
    gpu!();
    gb_cuda::exl3_linear::prefill_init().unwrap();
    let st = Stream::new().unwrap();
    let mut diffs = Vec::new();
    for (i, &(name, k, n)) in SHAPES.iter().enumerate() {
        let w = layer(k, n, 400 + i as u64);
        for rows in [1usize, 17, 129, 300, 1000, 4096] {
            let xb = acts(51 + i as u64 + rows as u64, rows, k);
            let x = upload(&xb);
            let mut y = DevBuf::from_slice(&vec![0x7fc1u16; rows * n]).unwrap();
            let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, k)).unwrap();
            glm_p2::linear_variant(&w, &x, &mut y, rows, Dtype::BF16, Dtype::BF16, &mut ws, Variant::Chain, 0, &st).unwrap();
            st.sync().unwrap();
            let (a, b) = (y.to_vec::<u16>().unwrap(), pinned(&w, &x, rows, &st));
            let bad = a.iter().zip(&b).filter(|(p, q)| p != q).count();
            eprintln!("{name:12} K {k:5} N {n:5} rows {rows:5}: {bad} of {} outputs differ", a.len());
            if bad != 0 {
                diffs.push(format!("{name} rows {rows}: {bad}"));
            }
        }
    }
    assert!(diffs.is_empty(), "chain != pinned: {diffs:?}");
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore]
fn bench_fused_against_the_pinned_path_on_the_qwen_shapes() {
    gpu!();
    gb_cuda::exl3_linear::prefill_init().unwrap();
    let st = Stream::new().unwrap();
    let mut flush = DevBuf::alloc(256 << 20).unwrap();
    let mut tot = std::collections::BTreeMap::<usize, (f64, f64, f64)>::new();
    for (i, &(name, k, n)) in SHAPES.iter().enumerate() {
        let w = layer(k, n, 300 + i as u64);
        // per-chunk multiplicity of the layer in the 64-layer model
        let count = match name {
            "mlp gate/up" => 128.0,
            "mlp down" => 64.0,
            "attn q" => 16.0,
            "gdn qkv" => 48.0,
            "gdn z" => 48.0,
            "out/o" => 64.0,
            _ => 32.0,
        };
        for rows in [512usize, 1024, 2048, 4096] {
            let xb = acts(3 + i as u64, rows, k);
            let x = upload(&xb);
            let mut y = DevBuf::alloc(rows * n * 2).unwrap();
            let mut ws = DevBuf::alloc(w.prefill_scratch_bytes(rows).max(glm_p2::workspace_bytes(rows, k))).unwrap();
            // what: 0 pinned, 1 fused fold128 (default tile), 2 fused chain
            let mut time = |what: u8| -> f64 {
                let mut v = Vec::new();
                for it in 0..8 {
                    flush.memset_async(it as u8, &st).unwrap();
                    st.sync().unwrap();
                    let t = std::time::Instant::now();
                    match what {
                        0 => w.forward_prefill_pinned(&x, &mut y, rows, Dtype::BF16, &mut ws, &st).unwrap(),
                        1 => glm_p2::linear_variant(
                            &w,
                            &x,
                            &mut y,
                            rows,
                            Dtype::BF16,
                            Dtype::BF16,
                            &mut ws,
                            Variant::Fold128,
                            0,
                            &st,
                        )
                        .unwrap(),
                        _ => glm_p2::linear_variant(
                            &w,
                            &x,
                            &mut y,
                            rows,
                            Dtype::BF16,
                            Dtype::BF16,
                            &mut ws,
                            Variant::Chain,
                            0,
                            &st,
                        )
                        .unwrap(),
                    }
                    st.sync().unwrap();
                    if it > 0 {
                        v.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                }
                median(v)
            };
            let tf = 2.0 * rows as f64 * k as f64 * n as f64 / 1e9;
            let (tp, tfo, tc) = (time(0), time(1), time(2));
            eprintln!(
                "BENCH {name:12} K {k:5} N {n:5} rows {rows:5}: pinned {tp:7.3} ms {:5.1} TF | fold128 {tfo:7.3} ms {:5.1} TF x{:.2} (tile {}) | chain {tc:7.3} ms {:5.1} TF x{:.2}",
                tf / tp,
                tf / tfo,
                tp / tfo,
                glm_p2::pick_tile(rows, k, n, Variant::Fold128),
                tf / tc,
                tp / tc
            );
            let e = tot.entry(rows).or_default();
            e.0 += count * tp;
            e.1 += count * tfo;
            e.2 += count * tc;
        }
    }
    for (rows, (p, f, c)) in tot {
        eprintln!("BENCH chunk of {rows:5} rows, all dense linears of one chunk: pinned {p:8.1} ms | fold128 {f:8.1} ms (x{:.2}) | chain {c:8.1} ms (x{:.2})", p / f, p / c);
    }
}
