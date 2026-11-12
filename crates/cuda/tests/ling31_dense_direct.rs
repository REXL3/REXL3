//! Lane LP3 (Ling-3.1-flash-1019 `bailing_moe_v3`): the dense EXL3 linears of a prompt chunk on the direct-to-MMA
//! tensor-core GEMM (`gb_cuda::glm_p2::ling_linear_parts`, the same fold128 kernel as GLM's P2 dense path) at
//! LING's shapes and widths — including the ODD `k2` half-integer widths (3 = 1.5 bits, 5 = 2.5, 7 = 3.5) the GLM
//! path never sees — and over the decomposed `Lin` parts (trellis / suh / svh) exactly where the pack put them
//! (the multi-part `suh`/`svh` slices), so no fp16 `W_q` is ever reconstructed in global memory.
//!
//! Gates (not ignored), over synthetic Mul1 EXL3 weights at every Ling dense projection's shape and at every odd
//! and even width the pack uses:
//! * `ling_direct_mma_matches_the_k2_decode_path`: per shape, the direct-to-MMA prefill against the row-invariant
//!   `k2` decode path (`Exl3Weights::k2_forward`, the very decoder the MoE kernels use) on the same trellis, fp16
//!   inputs, fp32 outputs. Fold128 must be within 1.5x of the decode path's fp64-rms error and under 5e-4 (the
//!   GLM P2 bound), so the direct path is numerically equivalent to the decode path's own dequantization.
//! * `ling_direct_mma_rows_are_bit_identical_in_any_call`: repeat, slice widths (whole, 129, 17, 1), a row
//!   permutation, NaN rows after the call's end: bit for bit.
//! * `ling_direct_mma_supports_every_dense_projection`: `glm_p2::ling_supported` accepts every (k, n, k2) of the
//!   Ling pack's dense projections (KDA qkv/f/g/o, MLA q_a/q_b/kv_a/o, dense FFN gate/up/down, MTP L72).
//! * `ling_fp32_router_logits_keep_the_bf16_reduction_order`: rounding the fp32 router linear's output to bf16 is
//!   bit-identical to the bf16 kernel's output, while the fp32 result retains its unrounded low bits.
//!
//!     cargo test -p gb-cuda --release --test ling31_dense_direct -- --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]
#![allow(clippy::needless_range_loop)]
mod common;

use common::Rng;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::glm_p2;
use gb_cuda::mla_gated;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::{Group, LoadedGroup};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::{Bits, Codebook};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

/// (name, K, N, k2): every dense EXL3 projection of the Ling-3.1-flash pack at the widths it ships. `k2` is twice
/// the bits (the trellis shape's `Bits::twice`), so 3 = 1.5 bits, 5 = 2.5, 7 = 3.5. The pack mixes widths across
/// projections and layers, so the table covers every k2 in 2..=8 even and odd.
const LING_SHAPES: &[(&str, usize, usize, usize)] = &[
    ("kda qkv", 4096, 24576, 6),
    ("kda qkv 3.5b", 4096, 24576, 7),
    ("kda f", 4096, 8192, 6),
    ("kda g", 4096, 8192, 6),
    ("kda o", 8192, 4096, 6),
    ("kda o 3.5b", 8192, 4096, 7),
    ("mla q_a", 4096, 1536, 6),
    ("mla q_b", 1536, 12288, 6),
    ("mla kv_a", 4096, 640, 6),
    ("mla o", 8192, 4096, 6),
    ("dense gate", 4096, 10240, 6),
    ("dense up", 4096, 10240, 6),
    ("dense down", 10240, 4096, 6),
    ("dense down 3.5b", 10240, 4096, 7),
    ("qkv 1.5b", 4096, 24576, 3),
    ("o 2.5b", 8192, 4096, 5),
    ("down 2b", 10240, 4096, 4),
    ("down 4b", 10240, 4096, 8),
];

fn synth_group_k2(k: usize, n: usize, k2: usize, seed: u64) -> LoadedGroup {
    let mut r = Rng(seed);
    let bits2 = k2 as u8;
    // 16 * (k2 / 2) values' worth of int16 words per 16x16 tile: the odd k2 pack two values into three bits, so
    // the word count is k * n * k2 / 32 int16s (k * n * k2 / 16 bytes).
    let words = k * n * k2 / 32;
    let trellis: Vec<i16> = (0..words).map(|_| r.next_u64() as i16).collect();
    let group = Group {
        prefix: "synthetic".into(),
        bits: Bits::from_twice(bits2).unwrap(),
        codebook: Codebook::Mul1,
        k,
        n,
        in_scales: "suh",
        out_scales: "svh",
        bias: false,
        files: vec![],
        bytes: 0,
    };
    let scale = |r: &mut Rng, len: usize, mag: f64| -> Vec<f64> {
        (0..len)
            .map(|_| {
                f16_to_f64(f64_to_f16(
                    (if r.next_u64() & 1 == 1 { -1.0 } else { 1.0 }) * mag * (0.5 + (r.next_u64() >> 40) as f64 / 16777216.0),
                ))
            })
            .collect()
    };
    let svh = scale(&mut r, n, 1.0 / (k as f64).sqrt());
    let suh = scale(&mut r, k, 1.0);
    LoadedGroup { group, trellis, kt: k / 16, nt: n / 16, suh, svh, bias: None }
}

/// The `Lin` image: the raw parts as device buffers (exactly what `bailing_moe_v3::device::Lin` holds).
struct Parts {
    k: usize,
    n: usize,
    k2: usize,
    trellis: DevBuf,
    suh: DevBuf,
    svh: DevBuf,
}

fn parts_of(g: &LoadedGroup) -> Parts {
    let f16_bits = |v: &[f64]| -> Vec<u16> { v.iter().map(|&x| f64_to_f16(x)).collect() };
    Parts {
        k: g.group.k,
        n: g.group.n,
        k2: g.group.bits.twice() as usize,
        trellis: DevBuf::from_slice(&g.trellis).unwrap(),
        suh: DevBuf::from_slice(&f16_bits(&g.suh)).unwrap(),
        svh: DevBuf::from_slice(&f16_bits(&g.svh)).unwrap(),
    }
}

fn activations(seed: u64, rows: usize, k: usize, scale: f64) -> (Vec<u16>, Vec<f64>) {
    let mut r = Rng(seed);
    let bits: Vec<u16> = (0..rows * k)
        .map(|_| {
            let v = r.uniform(-1.0, 1.0) as f64;
            let v = if r.next_u64() % 32 == 0 { 4.0 * v } else { v };
            f64_to_f16(v * scale)
        })
        .collect();
    let vals = bits.iter().map(|&b| f16_to_f64(b)).collect();
    (bits, vals)
}

fn upload(x: &[u16]) -> DevBuf {
    let mut d = DevBuf::zeroed_view(x.len() * 2).unwrap();
    d.upload(x).unwrap();
    d
}

/// The direct-to-MMA prefill over the `Lin` parts (what `Lin::forward_prefill_pinned` runs), fp32 out.
fn direct(p: &Parts, x: &DevBuf, rows: usize, st: &Stream) -> Vec<f32> {
    let mut y = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * p.n]).unwrap();
    let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, p.k)).unwrap();
    glm_p2::ling_linear_parts(
        x,
        &mut y,
        rows,
        Dtype::F16,
        Dtype::F32,
        p.k,
        p.n,
        p.k2,
        Codebook::Mul1,
        &p.trellis,
        &p.suh,
        &p.svh,
        &mut ws,
        st,
    )
    .unwrap();
    st.sync().unwrap();
    y.to_vec::<f32>().unwrap()
}

/// The row-invariant `k2` decode path over the same parts (the oracle: the very decoder the MoE kernels use).
fn decode(p: &Parts, x: &DevBuf, rows: usize, st: &Stream) -> Vec<f32> {
    let y = DevBuf::zeroed_view(rows * p.n * 4).unwrap();
    let mut ws = DevBuf::alloc(gb_cuda::exl3_linear::linear_workspace_bytes(128, p.k, p.n, 1)).unwrap();
    let mut s = 0;
    while s < rows {
        let m = (rows - s).min(128);
        let xs = x.subview(s * p.k * 2, m * p.k * 2);
        let mut ys = y.subview(s * p.n * 4, m * p.n * 4);
        Exl3Weights::k2_forward(
            &xs,
            &mut ys,
            m,
            Dtype::F16,
            Dtype::F32,
            p.k,
            p.n,
            p.k2,
            Codebook::Mul1,
            &p.trellis,
            &p.suh,
            &p.svh,
            None,
            &mut ws,
            st,
        )
        .unwrap();
        s += m;
    }
    st.sync().unwrap();
    y.to_vec::<f32>().unwrap()
}

/// The exact dequantized W_q [K][N] fp16 bits through the MoE decoder (a pure function of the trellis, odd k2
/// included), for the fp64 reference.
fn weights_f16(p: &Parts, st: &Stream) -> Vec<u16> {
    let mut wq = DevBuf::alloc(p.k * p.n * 2).unwrap();
    gb_cuda::fmoe::unpack(&p.trellis, &mut wq, p.k, p.n, p.k2, st).unwrap();
    st.sync().unwrap();
    wq.to_vec::<u16>().unwrap()
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

/// y rows of the exact layer in f64: ((x * suh) H / sqrt 128) . W_q . H / sqrt 128 * svh.
fn reference(g: &LoadedGroup, wq: &[u16], x: &[f64], rows: &[usize]) -> Vec<f64> {
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

/// (max abs / max |ref|, rms / rms ref, mean / rms ref, mean of err x sign(ref) / rms ref).
fn err(refv: &[f64], got: &[f32], rows: &[usize], n: usize) -> (f64, f64, f64, f64) {
    let (mut ee, mut rr, mut mx, mut mr, mut sum, mut shrink) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let cnt = (rows.len() * n) as f64;
    for (i, &r) in rows.iter().enumerate() {
        for j in 0..n {
            let (w, g) = (refv[i * n + j], got[r * n + j] as f64);
            let e = g - w;
            ee += e * e;
            rr += w * w;
            mx = mx.max(e.abs());
            mr = mr.max(w.abs());
            sum += e;
            shrink += e * w.signum();
        }
    }
    let rms = (rr / cnt).sqrt();
    (mx / mr, (ee / rr).sqrt(), sum / cnt / rms, shrink / cnt / rms)
}

#[test]
fn ling_direct_mma_supports_every_dense_projection() {
    gpu!();
    for &(name, k, n, k2) in LING_SHAPES {
        assert!(glm_p2::ling_supported(k, n, k2, Codebook::Mul1), "{name}: k2 = {k2} must be supported");
    }
    // the odd widths the GLM integer-bits entry point cannot name are the point of the Ling entry point
    for k2 in [3usize, 5, 7, 9] {
        assert!(glm_p2::ling_supported(4096, 4096, k2, Codebook::Mul1), "odd k2 = {k2} must be supported");
    }
}

#[test]
fn ling_direct_mma_matches_the_k2_decode_path() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 64;
    let sample: Vec<usize> = (0..rows).step_by(17).collect();
    for (i, &(name, k, n, k2)) in LING_SHAPES.iter().enumerate() {
        let g = synth_group_k2(k, n, k2, 4100 + i as u64);
        let p = parts_of(&g);
        let wq = weights_f16(&p, &st);
        let (xb, xv) = activations(71 + i as u64, rows, k, 1.0);
        let xd = upload(&xb);
        let refv = reference(&g, &wq, &xv, &sample);
        let (dm, dr, ds, dk) = err(&refv, &decode(&p, &xd, rows, &st), &sample, n);
        let (pm, pr, ps, pk) = err(&refv, &direct(&p, &xd, rows, &st), &sample, n);
        eprintln!(
            "{name:16} K {k:5} N {n:5} k2 {k2}: decode max {dm:.2e} rms {dr:.2e} mean {ds:+.1e} shrink {dk:+.1e} | direct max {pm:.2e} rms {pr:.2e} mean {ps:+.1e} shrink {pk:+.1e}"
        );
        assert!(pr <= 1.5 * dr + 1e-7 && pr <= 5e-4, "{name}: direct rms error {pr:e} against the decode path's {dr:e}");
        assert!(pm <= 1.5 * dm + 1e-6, "{name}: direct max error {pm:e} against the decode path's {dm:e}");
    }
}

#[test]
fn ling_direct_mma_rows_are_bit_identical_in_any_call() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 200;
    // a subset of the shapes at both an even and an odd k2 (the whole table is covered by the accuracy gate)
    for (i, &(name, k, n, k2)) in LING_SHAPES.iter().take(6).enumerate() {
        let g = synth_group_k2(k, n, k2, 5100 + i as u64);
        let p = parts_of(&g);
        let (xb, _) = activations(97 + i as u64, rows, k, 1.0);
        let xd = upload(&xb);
        let whole = direct(&p, &xd, rows, &st);
        assert!(whole.iter().all(|v| v.is_finite()) && whole.iter().any(|v| *v != 0.0), "{name}: output not finite or all zero");
        let again = direct(&p, &xd, rows, &st);
        assert!(whole.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: run to run");
        for &wd in &[129usize, 17, 1] {
            let mut got = Vec::with_capacity(rows * n);
            let mut s = 0;
            while s < rows {
                let m = wd.min(rows - s);
                let xs = upload(&xb[s * k..(s + m) * k]);
                got.extend(direct(&p, &xs, m, &st));
                s += m;
            }
            let bad = got.iter().zip(&whole).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            assert_eq!(bad, 0, "{name}: slices of {wd}: {bad} values differ from the whole call");
        }
        // a permutation of the rows, and NaN rows after the call's last row (never read)
        let perm: Vec<usize> = {
            let mut r = Rng(5 + i as u64);
            let mut p: Vec<usize> = (0..rows).collect();
            for j in (1..rows).rev() {
                p.swap(j, (r.next_u64() % (j as u64 + 1)) as usize);
            }
            p
        };
        let m = rows - 37;
        let mut xp: Vec<u16> = perm[..m].iter().flat_map(|&r| xb[r * k..(r + 1) * k].to_vec()).collect();
        xp.extend(std::iter::repeat_n(0x7e00u16, 37 * k));
        let yp = direct(&p, &upload(&xp), m, &st);
        for (j, &r) in perm[..m].iter().enumerate() {
            assert!(
                yp[j * n..(j + 1) * n].iter().zip(&whole[r * n..(r + 1) * n]).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name}: row {r} at position {j} of a permuted, NaN-padded call"
            );
        }
        eprintln!("{name:16} k2 {k2}: bit-identical over runs, slices [129, 17, 1], a permutation and NaN padding");
    }
}

/// The multi-part `suh`/`svh` case: a `Lin` whose scales sit at arbitrary offsets inside larger device regions
/// (as `bailing_moe_v3::device::Lin::new` subviews them out of the pack's regions). The direct path must read
/// exactly the slice it is handed, not the region's start.
#[test]
fn ling_direct_mma_reads_the_part_slices_not_the_region() {
    gpu!();
    let st = Stream::new().unwrap();
    let (k, n, k2) = (4096usize, 8192usize, 7usize); // 3.5 bits
    let g = synth_group_k2(k, n, k2, 6100);
    let p = parts_of(&g);
    // embed suh and svh at odd offsets inside larger buffers filled with poison
    let embed = |src: &DevBuf, bytes: usize, off: usize| -> DevBuf {
        let mut big = DevBuf::from_slice(&vec![0x7e00u16; (off + bytes + 512) / 2]).unwrap();
        big.subview(off, bytes).copy_from(src, bytes, &st).unwrap();
        big.subview(off, bytes)
    };
    let suh = embed(&p.suh, k * 2, 256);
    let svh = embed(&p.svh, n * 2, 512);
    // subview borrows nothing, but `big` must outlive the copies above (they are stream-ordered and synced below)
    st.sync().unwrap();
    let rows = 33;
    let (xb, _) = activations(131, rows, k, 1.0);
    let xd = upload(&xb);
    let mut y = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * n]).unwrap();
    let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, k)).unwrap();
    glm_p2::ling_linear_parts(
        &xd,
        &mut y,
        rows,
        Dtype::F16,
        Dtype::F32,
        k,
        n,
        k2,
        Codebook::Mul1,
        &p.trellis,
        &suh,
        &svh,
        &mut ws,
        &st,
    )
    .unwrap();
    st.sync().unwrap();
    let got = y.to_vec::<f32>().unwrap();
    let want = direct(&p, &xd, rows, &st);
    let bad = got.iter().zip(&want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(bad, 0, "the offset slices must give exactly the in-place parts' bits: {bad} differ");
}

/// fp32 router logits (Ling's `router_dtype = "fp32"`): on the small-row scalar path the fp32 linear retains the
/// unrounded accumulation; rounding each result to bf16 must reproduce the bf16 kernel's output bit for bit. On wide
/// rows the bf16-output linear and the fp32 router linear may be different kernels with different accumulation orders
/// (fp32 noise far below a bf16 ulp), so their results must agree to within one bf16 ulp.
#[test]
fn ling_fp32_router_logits_keep_the_bf16_reduction_order() {
    gpu!();
    let st = Stream::new().unwrap();
    let (h, e) = (4096usize, 512usize);
    // bf16 inputs and weights: the linear reads bf16, so generate the values and keep their bf16 bit patterns.
    let mut r = Rng(211);
    let bf = |v: f64| -> u16 { ((v as f32).to_bits() >> 16) as u16 };
    let xb: Vec<u16> = (0..4 * h).map(|_| bf(r.uniform(-1.0, 1.0) as f64)).collect();
    let wb: Vec<u16> = (0..e * h).map(|_| bf(r.uniform(-0.02, 0.02) as f64)).collect();
    let x = upload(&xb);
    let w = upload(&wb);
    // small rows: the scalar reference path (rows <= 4)
    let rows = 4;
    let mut y32 = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * e]).unwrap();
    mla_gated::bf16_linear_f32(&x, &w, &mut y32, rows, h, e, &st).unwrap();
    let mut y16 = DevBuf::from_slice(&vec![0u16; rows * e]).unwrap();
    mla_gated::bf16_linear(&x, &w, &mut y16, rows, h, e, &st).unwrap();
    st.sync().unwrap();
    let a = y32.to_vec::<f32>().unwrap();
    let b = y16.to_vec::<u16>().unwrap();
    let mut retained_low_bits = 0usize;
    for i in 0..rows * e {
        let bits = a[i].to_bits();
        retained_low_bits += usize::from(bits & 0xffff != 0);
        // Round-to-nearest-even, matching the bf16 output kernel.
        let rounded = ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16;
        assert_eq!(b[i], rounded, "logit {i}: fp32 {} vs bf16 {:04x}", a[i], b[i]);
    }
    assert!(retained_low_bits > rows * e / 2, "fp32 logits were unexpectedly pre-rounded");

    // wide rows (rows > 4): the fp32 router and the bf16-output linear agree to within one bf16 ulp
    let rows = 4096;
    let xb: Vec<u16> = (0..rows * h).map(|_| bf(r.uniform(-1.0, 1.0) as f64)).collect();
    let x = upload(&xb);
    let mut y32 = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * e]).unwrap();
    mla_gated::bf16_linear_f32(&x, &w, &mut y32, rows, h, e, &st).unwrap();
    let mut y16 = DevBuf::from_slice(&vec![0u16; rows * e]).unwrap();
    mla_gated::bf16_linear(&x, &w, &mut y16, rows, h, e, &st).unwrap();
    st.sync().unwrap();
    let a = y32.to_vec::<f32>().unwrap();
    let b = y16.to_vec::<u16>().unwrap();
    let mut max_rel = 0.0f32;
    for i in 0..rows * e {
        let rounded = ((a[i].to_bits().wrapping_add(0x7fff + ((a[i].to_bits() >> 16) & 1))) >> 16) as u16;
        // both are bf16 bit patterns: bf16 is the top half of an f32 (`f16_to_f64` reads IEEE half, a different layout)
        let rb = f32::from_bits((rounded as u32) << 16) as f64;
        let bb = f32::from_bits((b[i] as u32) << 16) as f64;
        // relative to the logit's magnitude, floored at 0.05 so a value near zero (where a different accumulation order moves many ulps of
        // the tiny result but nothing visible) does not dominate; one bf16 ulp is at most 2^-7 of the larger value
        let base = rb.abs().max(bb.abs()).max(0.05);
        max_rel = max_rel.max(((rb - bb).abs() / base) as f32);
    }
    assert!(
        max_rel <= 1.0 / 128.0,
        "wide-row bf16 output differs from the rounded fp32 output by more than one bf16 ulp: {max_rel}"
    );

    let mut fast = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * e]).unwrap();
    mla_gated::bf16_router_f32(&x, &w, &mut fast, rows, h, e, &st).unwrap();
    st.sync().unwrap();
    let c = fast.to_vec::<f32>().unwrap();
    let max_abs = a.iter().zip(&c).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(max_abs < 5e-4, "tensor-op router differs from the fp32 linear by {max_abs}");
    eprintln!("fp32 router logits: small-row exactness kept; tensor-op max abs diff {max_abs}; bf16 rounded rel diff {max_rel}");
}

/// The fp32 router linears' bits must not depend on how many rows share the call. Two row-invariant paths: the scalar warp-per-output
/// kernel for decode / verify rows (`bf16_linear_f32`, every row count), and the tensor-core kernel for prompt rows (`router_rows_f32`,
/// every row count including 1). Each row of a call of R rows (R across the old cuBLAS cut at 4, odd tails, the snapshot-cut slices of a
/// 585-token prompt, a full 4096-row chunk, at offsets inside a larger buffer) equals the same row computed alone by the same path. The
/// former `rows > 4` cuBLAS path failed this (483 of 512 logits of row 0 differed between a 1-row and a 5-row call).
#[test]
fn ling_fp32_router_logits_are_row_count_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let (h, e) = (4096usize, 512usize);
    let mut r = Rng(977);
    let bf = |v: f64| -> u16 { ((v as f32).to_bits() >> 16) as u16 };
    let maxrows = 4096usize;
    let xb: Vec<u16> = (0..maxrows * h).map(|_| bf(r.uniform(-3.0, 3.0) as f64)).collect();
    let wb: Vec<u16> = (0..e * h).map(|_| bf(r.uniform(-0.05, 0.05) as f64)).collect();
    let x = upload(&xb);
    let w = upload(&wb);
    let run = |mma: bool, rows: usize, off: usize| -> Vec<f32> {
        let xs = x.subview(off * h * 2, rows * h * 2);
        let mut y = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * e]).unwrap();
        if mma {
            mla_gated::router_rows_f32(&xs, &w, &mut y, rows, h, e, &st).unwrap();
        } else {
            mla_gated::bf16_linear_f32(&xs, &w, &mut y, rows, h, e, &st).unwrap();
        }
        st.sync().unwrap();
        y.to_vec::<f32>().unwrap()
    };
    let probe: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 15, 16, 17, 63, 64, 65, 72, 73, 511, 512, 584, 1023, 2047, 4095];
    for mma in [false, true] {
        let name = if mma { "tensor-core prompt router" } else { "scalar decode router" };
        // the reference: every probed row alone
        let want: Vec<Vec<f32>> = probe.iter().map(|&i| run(mma, 1, i)).collect();
        for &(rows, off) in &[
            (4usize, 0usize),
            (5, 0),
            (8, 0),
            (17, 0),
            (73, 512),
            (73, 0),
            (585, 0),
            (1024, 0),
            (4096, 0),
            (3, 4093),
            (1337, 2000),
        ] {
            let got = run(mma, rows, off);
            for (pi, &i) in probe.iter().enumerate() {
                if i < off || i >= off + rows {
                    continue;
                }
                let g = &got[(i - off) * e..(i - off + 1) * e];
                let bad = g.iter().zip(&want[pi]).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                assert_eq!(
                    bad, 0,
                    "{name}: logits of row {i} in a call of {rows} rows at offset {off} differ from the row alone: {bad} of {e}"
                );
            }
        }
    }
    // the two paths compute the same function to fp32 accumulation-order noise (a sanity check of the tensor-core kernel's indexing)
    let a = run(false, 129, 700);
    let b = run(true, 129, 700);
    let max_abs = a.iter().zip(&b).fold(0.0f32, |m, (p, q)| m.max((p - q).abs()));
    let scale = a.iter().fold(0.0f32, |m, p| m.max(p.abs()));
    assert!(
        max_abs <= 1e-4 * scale.max(1.0),
        "tensor-core router differs from the scalar router by {max_abs} (logit scale {scale})"
    );
}

/// Time (not a gate): the three router linears at a 4,096-row prefill chunk (`--ignored bench_ling_router --nocapture`).
#[test]
#[ignore]
fn bench_ling_router() {
    gpu!();
    let st = Stream::new().unwrap();
    let (h, e) = (4096usize, 512usize);
    let mut r = Rng(5);
    let bf = |v: f64| -> u16 { ((v as f32).to_bits() >> 16) as u16 };
    for &rows in &[73usize, 585, 2048, 4096] {
        let xb: Vec<u16> = (0..rows * h).map(|_| bf(r.uniform(-3.0, 3.0) as f64)).collect();
        let wb: Vec<u16> = (0..e * h).map(|_| bf(r.uniform(-0.05, 0.05) as f64)).collect();
        let (x, w) = (upload(&xb), upload(&wb));
        let mut y = DevBuf::zeroed(rows * e * 4).unwrap();
        let time = |f: &dyn Fn(&mut DevBuf)| -> f64 {
            let mut y = DevBuf::zeroed(rows * e * 4).unwrap();
            for _ in 0..3 {
                f(&mut y);
            }
            st.sync().unwrap();
            let t = std::time::Instant::now();
            for _ in 0..20 {
                f(&mut y);
            }
            st.sync().unwrap();
            t.elapsed().as_secs_f64() / 20.0 * 1e3
        };
        let cublas = time(&|y| mla_gated::bf16_router_f32(&x, &w, y, rows, h, e, &st).unwrap());
        let mma = time(&|y| mla_gated::router_rows_f32(&x, &w, y, rows, h, e, &st).unwrap());
        let scalar = time(&|y| mla_gated::bf16_linear_f32(&x, &w, y, rows, h, e, &st).unwrap());
        let _ = &mut y;
        eprintln!("router {rows:5} rows: cuBLAS {cublas:.3} ms | tensor-core row-invariant {mma:.3} ms | scalar warp-per-output {scalar:.3} ms");
    }
}

#[test]
#[ignore]
fn bench_ling_direct_tiles() {
    gpu!();
    let st = Stream::new().unwrap();
    // the heaviest dense projections of the Ling pack, at the actual prefill chunk widths
    for &rows in &[2048usize, 4096] {
        for &(name, k, n, k2) in &[LING_SHAPES[0], LING_SHAPES[2], LING_SHAPES[4], LING_SHAPES[12]] {
            let g = synth_group_k2(k, n, k2, 0x9911 + k2 as u64);
            let p = parts_of(&g);
            let (xb, _) = activations(717, rows, k, 1.0);
            let x = upload(&xb);
            let mut y = DevBuf::zeroed(rows * n * 4).unwrap();
            let mut ws = DevBuf::alloc(glm_p2::workspace_bytes(rows, k)).unwrap();
            for tile in [0, 641, 642, 1281, 1282] {
                glm_p2::set_ling_tile_force(tile);
                let mut direct = || {
                    glm_p2::ling_linear_parts(
                        &x,
                        &mut y,
                        rows,
                        Dtype::F16,
                        Dtype::F32,
                        k,
                        n,
                        k2,
                        Codebook::Mul1,
                        &p.trellis,
                        &p.suh,
                        &p.svh,
                        &mut ws,
                        &st,
                    )
                    .unwrap();
                    st.sync().unwrap();
                };
                direct();
                let t0 = std::time::Instant::now();
                for _ in 0..3 {
                    direct();
                }
                let dm = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;
                eprintln!("{name:16} rows={rows} tile={tile}: {dm:.3} ms");
            }
            glm_p2::set_ling_tile_force(0);
        }
    }
}
