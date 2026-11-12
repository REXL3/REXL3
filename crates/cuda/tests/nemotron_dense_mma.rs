//! Lane S3: the tensor-core prefill GEMM of the dense EXL3 linears (`csrc/nemotron_dense_mma.cu`, contract P2) on SYNTHETIC 4-bit
//! Mul1 EXL3 weights of the Nemotron 3.5 Super VL shapes (`common::exl3_synth`; shapes from the pack's config: hidden 4,096, Mamba
//! in_proj 18,560 = 2 x 8,192 + 2 x 8 x 128 + 128, out_proj 8,192, q / o 4,096, k / v 2 x 128, MoE latent 1,024, shared expert 5,376).
//!
//! Gates (not ignored):
//! * `p2_is_as_close_to_fp64_as_the_decode_path`: per shape, the fp64 reference of the exact dequantized weights on sampled rows; the
//!   P2 output's max and rms relative error are printed next to the decode path's tensor-core kernel (`Exl3Weights::forward`, fp16
//!   activations as well) and, for the MoE block's four projections, next to the frozen fp32 FFMA kernel (`pf_gemm4`, the path P2
//!   replaces there). P2 must be within 1.5x of the decode path's error and under 5e-4 rms.
//! * `p2_rows_are_bit_identical_in_any_call`: repeat, slice widths (whole, 256, 100, 65, 17, 1), every tile shape, row permutation, and
//!   NaN rows past the call's end (padding is never read): bit for bit.
//! * `p2_tp_column_cuts_are_exact`: the two ranks' 128-column strips (the TP=2 cut of each N-split projection) give the single layer's
//!   columns bit for bit.
//!
//! Timing (ignored, `bench` in the name): ms per call at 256 to 4,096 rows against the decode path in 128-row calls (what
//! `linear_rows` runs today) and the fp32 pf path, cold (a 256 MiB memset before each call).
//!
//!     cargo test -p gb-cuda --release --test nemotron_dense_mma -- --test-threads=1 --nocapture
//!     cargo test -p gb-cuda --release --test nemotron_dense_mma -- --ignored --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]
#![allow(clippy::needless_range_loop)]
mod common;

use common::exl3_synth::synth_group;
use common::Rng;
use core::ffi::c_void;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::nemotron_dense_mma as ndm;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::f16_to_f64;
use gb_exl3::Codebook;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

/// (name, K, N, input dtype as the engine passes it, MoE block projection)
const SHAPES: &[(&str, usize, usize, Dtype, bool)] = &[
    ("mamba in_proj", 4096, 18560, Dtype::F32, false),
    ("mamba out_proj", 8192, 4096, Dtype::BF16, false),
    ("attn q", 4096, 4096, Dtype::F32, false),
    ("attn k", 4096, 256, Dtype::F32, false),
    ("attn v", 4096, 256, Dtype::F32, false),
    ("attn o", 4096, 4096, Dtype::F32, false),
    ("moe fc1", 4096, 1024, Dtype::F32, true),
    ("moe fc2", 1024, 4096, Dtype::F32, true),
    ("moe shared up", 4096, 5376, Dtype::F32, true),
    ("moe shared down", 5376, 4096, Dtype::F32, true),
];

/// The TP=2 column split of each shape in strips of 128 (`tp::splits`: in_proj rank 0 is [0, 9216)).
fn tp_split(n: usize) -> usize {
    if n == 18560 {
        72
    } else {
        n / 256
    }
}

fn bf16_round(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits(((b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16) << 16)
}

/// Activations with the heavier tails of real hidden states (a few percent of entries four times larger).
fn activations(seed: u64, rows: usize, k: usize, dt: Dtype) -> Vec<f32> {
    let mut r = Rng(seed);
    (0..rows * k)
        .map(|_| {
            let v = r.uniform(-1.0, 1.0);
            let v = if r.next_u64() % 32 == 0 { 4.0 * v } else { v };
            if dt == Dtype::BF16 {
                bf16_round(v)
            } else {
                v
            }
        })
        .collect()
}

/// The rows on the device in the engine's dtype, as a view (so that the 128-row calls of the decode path can cut it).
fn upload(x: &[f32], dt: Dtype) -> DevBuf {
    let mut d = DevBuf::zeroed_view(x.len() * dt.size()).unwrap();
    match dt {
        Dtype::F32 => d.upload(x).unwrap(),
        Dtype::BF16 => d.upload(&x.iter().map(|v| (v.to_bits() >> 16) as u16).collect::<Vec<u16>>()).unwrap(),
        Dtype::F16 => unreachable!(),
    }
    d
}

/// The layer of the 128-column strips [lo, hi) of `g` (what a TP rank holds of an N-split projection), built on the host.
fn strips_of(g: &gb_exl3::checkpoint::LoadedGroup, lo: usize, hi: usize) -> Exl3Weights {
    let (k, n) = (g.group.k, g.group.n);
    // the trellis is int16: a 16 x 16 tile is 16 * bits of them (64 at 4 bits)
    let bits = g.group.bits.twice() as usize / 2;
    let (tn, words) = (n / 16, 16 * bits);
    let mut sub = g.clone();
    sub.group.n = (hi - lo) * 128;
    sub.nt = sub.group.n / 16;
    sub.trellis = (0..k / 16).flat_map(|kt| g.trellis[(kt * tn + lo * 8) * words..(kt * tn + hi * 8) * words].to_vec()).collect();
    sub.svh = g.svh[lo * 128..hi * 128].to_vec();
    Exl3Weights::from_loaded(&sub).unwrap()
}

fn layer(k: usize, n: usize, seed: u64) -> (Exl3Weights, gb_exl3::checkpoint::LoadedGroup) {
    layer_bits(k, n, 4, seed)
}

/// A synthetic Mul1 layer at `bits` (whole widths 2 to 8).
fn layer_bits(k: usize, n: usize, bits: usize, seed: u64) -> (Exl3Weights, gb_exl3::checkpoint::LoadedGroup) {
    let g = synth_group(k, n, bits, Codebook::Mul1, seed);
    (Exl3Weights::from_loaded(&g).unwrap(), g)
}

fn p2(w: &Exl3Weights, x: &DevBuf, dt: Dtype, rows: usize, tile: i32, st: &Stream) -> Vec<f32> {
    let mut y = DevBuf::from_slice(&vec![f32::from_bits(0x7fc0_dead); rows * w.n]).unwrap();
    let mut ws = DevBuf::alloc(ndm::workspace_bytes(rows, w.k)).unwrap();
    ndm::linear_tiled(w, x, &mut y, rows, dt, Dtype::F32, &mut ws, tile, st).unwrap();
    st.sync().unwrap();
    y.to_vec::<f32>().unwrap()
}

/// The decode path in calls of at most 128 rows (`linear_rows`).
fn decode_path(w: &Exl3Weights, x: &DevBuf, dt: Dtype, rows: usize, st: &Stream) -> Vec<f32> {
    let y = DevBuf::zeroed_view(rows * w.n * 4).unwrap();
    let mut ws = DevBuf::alloc(w.workspace_bytes(128)).unwrap();
    let mut s = 0;
    while s < rows {
        let m = (rows - s).min(128);
        let xs = x.subview(s * w.k * dt.size(), m * w.k * dt.size());
        let mut ys = y.subview(s * w.n * 4, m * w.n * 4);
        w.forward(&xs, &mut ys, m, dt, Dtype::F32, &mut ws, st).unwrap();
        s += m;
    }
    st.sync().unwrap();
    y.to_vec::<f32>().unwrap()
}

// ---- the frozen fp32 path of the MoE block's dense projections (pf_gemm4), called as nemotron_moe_prefill.rs calls it ----

#[repr(C)]
#[derive(Clone, Copy)]
struct PfProj {
    w: u64,
    suh: u64,
    svh: u64,
    bias: u64,
    cb: u64,
    fmt: i32,
    twice: i32,
    k: i32,
    n: i32,
}

extern "C" {
    fn gb_nmp_rotate(
        src: *const c_void,
        ksrc: i32,
        div: i32,
        pick: *const c_void,
        tab: *const c_void,
        rows: i32,
        k: i32,
        out: *mut c_void,
        s: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn gb_nmp_gemm(
        src: *const c_void,
        ksrc: i32,
        lists: *const c_void,
        lcap: i32,
        work: *const c_void,
        nwork: *const c_void,
        dense_rows: i32,
        zitems: i32,
        rbmax: i32,
        tab: *const c_void,
        ntab: *const c_void,
        k: i32,
        n: i32,
        s: i32,
        out: *mut c_void,
        relu2: i32,
        add: *const c_void,
        stream: *mut c_void,
    ) -> i32;
}

struct Fp32Path {
    _bufs: Vec<DevBuf>,
    tab: DevBuf,
}

fn fp32_path(g: &gb_exl3::checkpoint::LoadedGroup, w: &Exl3Weights) -> Fp32Path {
    let suh: Vec<f32> = g.suh.iter().map(|&v| v as f32).collect();
    let svh: Vec<f32> = g.svh.iter().map(|&v| v as f32).collect();
    let tr = DevBuf::from_slice(&g.trellis).unwrap();
    let (s, v) = (DevBuf::from_slice(&suh).unwrap(), DevBuf::from_slice(&svh).unwrap());
    let p = PfProj {
        w: tr.ptr() as u64,
        suh: s.ptr() as u64,
        svh: v.ptr() as u64,
        bias: 0,
        cb: 0,
        fmt: 2 << 4,
        twice: 8,
        k: w.k as i32,
        n: w.n as i32,
    };
    let tab = DevBuf::from_slice(&[p]).unwrap();
    Fp32Path { _bufs: vec![tr, s, v], tab }
}

fn slices(k: usize) -> usize {
    let kt = k / 16;
    [16usize, 8, 4, 2, 1].into_iter().find(|&s| kt % s == 0 && kt / s >= 16).unwrap_or(1)
}

fn fp32_run(f: &Fp32Path, shape: (usize, usize), x: &DevBuf, y: &mut DevBuf, rot: &mut DevBuf, rows: usize, st: &Stream) {
    let (k, n) = shape;
    unsafe {
        assert_eq!(
            gb_nmp_rotate(x.ptr(), k as i32, 1, core::ptr::null(), f.tab.ptr(), rows as i32, k as i32, rot.ptr(), st.raw()),
            0
        );
        assert_eq!(
            gb_nmp_gemm(
                rot.ptr(),
                k as i32,
                core::ptr::null(),
                0,
                core::ptr::null(),
                core::ptr::null(),
                rows as i32,
                rows.div_ceil(32) as i32,
                32,
                f.tab.ptr(),
                core::ptr::null(),
                k as i32,
                n as i32,
                slices(k) as i32,
                y.ptr(),
                0,
                core::ptr::null(),
                st.raw()
            ),
            0
        );
    }
}

// ---- the fp64 reference ----

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

/// y rows of the exact layer in f64: ((x * suh) H / sqrt 128) . W_q (the fp16 values the kernels decode) . H / sqrt 128 * svh.
fn reference(g: &gb_exl3::checkpoint::LoadedGroup, wq: &[f64], x: &[f32], rows: &[usize]) -> Vec<f64> {
    let (k, n) = (g.group.k, g.group.n);
    let s = 1.0 / 128f64.sqrt();
    let mut out = Vec::with_capacity(rows.len() * n);
    for &r in rows {
        let mut xr: Vec<f64> = (0..k).map(|i| x[r * k + i] as f64 * g.suh[i]).collect();
        xr.chunks_mut(128).for_each(fwht);
        let mut c = vec![0f64; n];
        for i in 0..k {
            let a = xr[i] * s;
            let wrow = &wq[i * n..(i + 1) * n];
            for j in 0..n {
                c[j] += a * wrow[j];
            }
        }
        c.chunks_mut(128).for_each(fwht);
        out.extend((0..n).map(|j| c[j] * s * g.svh[j]));
    }
    out
}

/// (max abs error / max |ref|, rms error / rms ref) over the sampled rows.
fn err(refv: &[f64], got: &[f32], rows: &[usize], n: usize) -> (f64, f64) {
    let (mut ee, mut rr, mut mx, mut mr) = (0f64, 0f64, 0f64, 0f64);
    for (i, &r) in rows.iter().enumerate() {
        for j in 0..n {
            let (w, g) = (refv[i * n + j], got[r * n + j] as f64);
            ee += (g - w) * (g - w);
            rr += w * w;
            mx = mx.max((g - w).abs());
            mr = mr.max(w.abs());
        }
    }
    (mx / mr, (ee / rr).sqrt())
}

#[test]
fn p2_is_as_close_to_fp64_as_the_decode_path() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 256;
    let sample: Vec<usize> = (0..rows).step_by(51).collect();
    for (i, &(name, k, n, dt, moe)) in SHAPES.iter().enumerate() {
        let (w, g) = layer(k, n, 1000 + i as u64);
        let mut wq = DevBuf::alloc(k * n * 2).unwrap();
        w.reconstruct(&mut wq, &st).unwrap();
        st.sync().unwrap();
        let wq: Vec<f64> = wq.to_vec::<u16>().unwrap().into_iter().map(f16_to_f64).collect();
        let x = activations(77 + i as u64, rows, k, dt);
        let xd = upload(&x, dt);
        let refv = reference(&g, &wq, &x, &sample);
        let (pm, pr) = err(&refv, &p2(&w, &xd, dt, rows, 0, &st), &sample, n);
        let (dm, dr) = err(&refv, &decode_path(&w, &xd, dt, rows, &st), &sample, n);
        let fp32 = if moe {
            let f = fp32_path(&g, &w);
            let mut y = DevBuf::alloc(rows * n * 4).unwrap();
            let mut rot = DevBuf::alloc(rows * k * 4).unwrap();
            fp32_run(&f, (k, n), &xd, &mut y, &mut rot, rows, &st);
            st.sync().unwrap();
            let e = err(&refv, &y.to_vec::<f32>().unwrap(), &sample, n);
            format!(" | fp32 pf path max {:.2e} rms {:.2e}", e.0, e.1)
        } else {
            String::new()
        };
        eprintln!("{name:16} K {k:5} N {n:5}: P2 max {pm:.2e} rms {pr:.2e} | decode-path tc max {dm:.2e} rms {dr:.2e}{fp32}");
        assert!(pr <= 1.5 * dr + 1e-7 && pr <= 5e-4, "{name}: P2 rms error {pr:e} against the decode path's {dr:e}");
        assert!(pm <= 1.5 * dm + 1e-6, "{name}: P2 max error {pm:e} against the decode path's {dm:e}");
    }
}

#[test]
fn p2_rows_are_bit_identical_in_any_call() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 512;
    for (i, &(name, k, n, dt, _)) in SHAPES.iter().enumerate() {
        let (w, _) = layer(k, n, 2000 + i as u64);
        let x = activations(91 + i as u64, rows, k, dt);
        let xd = upload(&x, dt);
        let whole = p2(&w, &xd, dt, rows, 0, &st);
        assert!(whole.iter().all(|v| v.is_finite()) && whole.iter().any(|v| *v != 0.0), "{name}: output not finite or all zero");
        let again = p2(&w, &xd, dt, rows, 0, &st);
        assert!(whole.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}: run to run");
        let tiles: &[i32] = if n >= 4096 { &[641, 642, 1281, 1282] } else { &[641, 1282] };
        let widths: &[usize] = if k * n > 40_000_000 { &[256, 65, 1] } else { &[256, 100, 65, 17, 1] };
        for &tile in tiles {
            for &wd in widths {
                let mut got = Vec::with_capacity(rows * n);
                let mut s = 0;
                while s < rows {
                    let m = wd.min(rows - s);
                    let xs = upload(&x[s * k..(s + m) * k], dt);
                    got.extend(p2(&w, &xs, dt, m, tile, &st));
                    s += m;
                }
                let bad = got.iter().zip(&whole).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                assert_eq!(bad, 0, "{name}: tile {tile}, slices of {wd}: {bad} values differ from the whole call");
            }
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
        let mut xp: Vec<f32> = perm[..m].iter().flat_map(|&r| x[r * k..(r + 1) * k].to_vec()).collect();
        xp.extend(std::iter::repeat_n(f32::NAN, 37 * k));
        let yp = p2(&w, &upload(&xp, dt), dt, m, 0, &st);
        for (j, &r) in perm[..m].iter().enumerate() {
            assert!(
                yp[j * n..(j + 1) * n].iter().zip(&whole[r * n..(r + 1) * n]).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name}: row {r} at position {j} of a permuted, NaN-padded call"
            );
        }
        eprintln!("{name:16}: bit-identical over runs, tiles {tiles:?}, slices {widths:?}, a permutation and NaN padding");
    }
}

#[test]
fn p2_tp_column_cuts_are_exact() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 300;
    for (i, &(name, k, n, dt, _)) in SHAPES.iter().enumerate() {
        let strips = n / 128;
        if strips < 2 {
            continue;
        }
        let (w, g) = layer(k, n, 3000 + i as u64);
        let xd = upload(&activations(13 + i as u64, rows, k, dt), dt);
        let whole = p2(&w, &xd, dt, rows, 0, &st);
        let cut = tp_split(n);
        for (lo, hi) in [(0, cut), (cut, strips)] {
            let part = strips_of(&g, lo, hi);
            let y = p2(&part, &xd, dt, rows, 0, &st);
            let nn = (hi - lo) * 128;
            for r in 0..rows {
                assert!(
                    y[r * nn..(r + 1) * nn]
                        .iter()
                        .zip(&whole[r * n + lo * 128..r * n + hi * 128])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{name}: rank strips [{lo}, {hi}) row {r} differ from the single layer"
                );
            }
        }
        eprintln!("{name:16}: TP=2 column cut at strip {cut} of {strips} exact");
    }
}

/// P1-6 (v0.1.2 lane K): the tensor-core prefill GEMM at every whole width of a MixedK pack's dense linears (2, 3, 5, 6, 7, 8 bits,
/// Mul1; 4 bits is the tests above): as close to the fp64 reference as the decode path, bit-identical over runs, tiles, slices and a
/// permutation, and a TP column cut exact. The shapes are the pack's: attention q / k / v / o and Mamba out_proj (K 8,192).
#[test]
fn p2_at_every_whole_width_is_as_close_as_the_decode_path_and_bit_identical_in_any_call() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 200;
    let sample: Vec<usize> = (0..rows).step_by(41).collect();
    let shapes = [("attn q", 4096, 4096), ("attn k/v", 4096, 256), ("attn o", 4096, 4096), ("mamba out", 8192, 4096)];
    for bits in [2usize, 3, 5, 6, 7, 8] {
        for (i, &(name, k, n)) in shapes.iter().enumerate() {
            let (w, g) = layer_bits(k, n, bits, 7000 + 10 * bits as u64 + i as u64);
            assert!(w.bits == bits, "{name}: {bits} bits");
            let mut wq = DevBuf::alloc(k * n * 2).unwrap();
            w.reconstruct(&mut wq, &st).unwrap();
            st.sync().unwrap();
            let wq: Vec<f64> = wq.to_vec::<u16>().unwrap().into_iter().map(f16_to_f64).collect();
            let dt = Dtype::F32;
            let x = activations(55 + i as u64, rows, k, dt);
            let xd = upload(&x, dt);
            let refv = reference(&g, &wq, &x, &sample);
            let whole = p2(&w, &xd, dt, rows, 0, &st);
            let (pm, pr) = err(&refv, &whole, &sample, n);
            let (dm, dr) = err(&refv, &decode_path(&w, &xd, dt, rows, &st), &sample, n);
            eprintln!(
                "{bits} bits {name:10} K {k:5} N {n:5}: P2 max {pm:.2e} rms {pr:.2e} | decode path max {dm:.2e} rms {dr:.2e}"
            );
            assert!(pr <= 1.5 * dr + 1e-7 && pr <= 5e-4, "{bits} bits {name}: P2 rms {pr:e} against the decode path's {dr:e}");
            assert!(pm <= 1.5 * dm + 1e-6, "{bits} bits {name}: P2 max {pm:e} against the decode path's {dm:e}");
            let again = p2(&w, &xd, dt, rows, 0, &st);
            assert!(whole.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()), "{bits} bits {name}: run to run");
            for &tile in &[641, 642, 1281, 1282] {
                for &wd in &[64usize, 17, 1] {
                    let mut got = Vec::with_capacity(rows * n);
                    let mut s = 0;
                    while s < rows {
                        let m = wd.min(rows - s);
                        got.extend(p2(&w, &upload(&x[s * k..(s + m) * k], dt), dt, m, tile, &st));
                        s += m;
                    }
                    let bad = got.iter().zip(&whole).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    assert_eq!(
                        bad, 0,
                        "{bits} bits {name}: tile {tile}, slices of {wd}: {bad} values differ from the whole call"
                    );
                }
            }
            let strips = n / 128;
            if strips >= 2 {
                let cut = tp_split(n);
                for (lo, hi) in [(0, cut), (cut, strips)] {
                    let y = p2(&strips_of(&g, lo, hi), &xd, dt, rows, 0, &st);
                    let nn = (hi - lo) * 128;
                    for r in 0..rows {
                        assert!(
                            y[r * nn..(r + 1) * nn]
                                .iter()
                                .zip(&whole[r * n + lo * 128..r * n + hi * 128])
                                .all(|(a, b)| a.to_bits() == b.to_bits()),
                            "{bits} bits {name}: TP strips [{lo}, {hi}) row {r}"
                        );
                    }
                }
            }
        }
        eprintln!("{bits} bits: every shape as close as the decode path, bit-identical over runs, tiles, slices; TP cut exact");
    }
}

// ---- timing ----// ---- timing ----

mod ev {
    use core::ffi::c_void;
    extern "C" {
        pub fn cudaEventCreate(e: *mut *mut c_void) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
        pub fn cudaEventSynchronize(e: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
        pub fn cudaEventDestroy(e: *mut c_void) -> i32;
    }
}

/// Median ms of `reps` calls of `f`, each after a 256 MiB memset (no weight line left in L2).
fn time_cold(st: &Stream, flush: &mut DevBuf, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut t = Vec::new();
    f();
    st.sync().unwrap();
    unsafe {
        let (mut a, mut b) = (core::ptr::null_mut(), core::ptr::null_mut());
        ev::cudaEventCreate(&mut a);
        ev::cudaEventCreate(&mut b);
        for i in 0..reps {
            flush.memset_async((i & 0xff) as u8, st).unwrap();
            ev::cudaEventRecord(a, st.raw());
            f();
            ev::cudaEventRecord(b, st.raw());
            ev::cudaEventSynchronize(b);
            let mut ms = 0f32;
            ev::cudaEventElapsedTime(&mut ms, a, b);
            t.push(ms as f64);
        }
        ev::cudaEventDestroy(a);
        ev::cudaEventDestroy(b);
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[t.len() / 2]
}

/// Per-width P2 throughput of the trunk's dense projections (v0.1.2 lane K, P1-6: his pack's attention and Mamba linears at 4 to 8 bits,
/// the any-width instantiations of 400ece9e): P2 ms and TFLOP/s at 1,024 and 4,096 rows for each whole width, against the decode-path
/// kernel in 128-row calls (what such a projection took before). `REXL3_NEMO_PERF_WIDTHS` picks the widths.
#[test]
#[ignore = "timing"]
fn bench_p2_widths() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut flush = DevBuf::alloc(256 << 20).unwrap();
    let reps = 7;
    let widths: Vec<usize> = match std::env::var("REXL3_NEMO_PERF_WIDTHS") {
        Ok(v) => v.split(',').map(|s| s.trim().parse().unwrap()).collect(),
        Err(_) => (2..=8).collect(),
    };
    let shapes =
        [("mamba in_proj", 4096, 18560), ("mamba out_proj", 8192, 4096), ("attn q / o", 4096, 4096), ("attn k / v", 4096, 256)];
    println!("| shape | bits | rows | P2 tile | P2 ms | TFLOP/s | decode-path tc ms (128-row calls) | P2 speed-up |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for &bits in &widths {
        for (i, &(name, k, n)) in shapes.iter().enumerate() {
            let (w, _) = layer_bits(k, n, bits, 9000 + 10 * bits as u64 + i as u64);
            let dt = Dtype::F32;
            let rmax = 4096;
            let xd = upload(&activations(70 + i as u64, rmax, k, dt), dt);
            let mut y = DevBuf::zeroed_view(rmax * n * 4).unwrap();
            let mut ws = DevBuf::alloc(ndm::workspace_bytes(rmax, k)).unwrap();
            let mut dws = DevBuf::alloc(w.workspace_bytes(128)).unwrap();
            for rows in [1024usize, 4096] {
                let tp = time_cold(&st, &mut flush, reps, || {
                    ndm::linear(&w, &xd, &mut y, rows, dt, Dtype::F32, &mut ws, &st).unwrap()
                });
                let td = time_cold(&st, &mut flush, reps, || {
                    let mut s = 0;
                    while s < rows {
                        let m = (rows - s).min(128);
                        let xs = xd.subview(s * k * dt.size(), m * k * dt.size());
                        let mut ys = y.subview(s * n * 4, m * n * 4);
                        w.forward(&xs, &mut ys, m, dt, Dtype::F32, &mut dws, &st).unwrap();
                        s += m;
                    }
                });
                println!(
                    "| {name} | {bits} | {rows} | {} | {tp:.3} | {:.1} | {td:.3} | {:.2}x |",
                    ndm::pick_tile(rows, n),
                    2.0 * rows as f64 * k as f64 * n as f64 / tp / 1e9,
                    td / tp
                );
            }
        }
    }
}

#[test]
#[ignore = "timing"]
fn bench_p2_against_the_existing_kernels() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut flush = DevBuf::alloc(256 << 20).unwrap();
    let reps = 9;
    println!("| shape | rows | P2 tile | P2 ms | TFLOP/s | decode-path tc ms (128-row calls) | fp32 pf ms | P2 speed-up |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|");
    for (i, &(name, k, n, dt, moe)) in SHAPES.iter().enumerate() {
        let (w, g) = layer(k, n, 4000 + i as u64);
        let rmax = 4096;
        let xd = upload(&activations(7 + i as u64, rmax, k, dt), dt);
        let mut y = DevBuf::zeroed_view(rmax * n * 4).unwrap();
        let mut ws = DevBuf::alloc(ndm::workspace_bytes(rmax, k)).unwrap();
        let mut dws = DevBuf::alloc(w.workspace_bytes(128)).unwrap();
        let f = if moe { Some(fp32_path(&g, &w)) } else { None };
        let mut rot = DevBuf::alloc(rmax * k * 4).unwrap();
        for rows in [256usize, 512, 1024, 2048, 4096] {
            let tp =
                time_cold(&st, &mut flush, reps, || ndm::linear(&w, &xd, &mut y, rows, dt, Dtype::F32, &mut ws, &st).unwrap());
            let td = time_cold(&st, &mut flush, reps, || {
                let mut s = 0;
                while s < rows {
                    let m = (rows - s).min(128);
                    let xs = xd.subview(s * k * dt.size(), m * k * dt.size());
                    let mut ys = y.subview(s * n * 4, m * n * 4);
                    w.forward(&xs, &mut ys, m, dt, Dtype::F32, &mut dws, &st).unwrap();
                    s += m;
                }
            });
            let tf = match &f {
                Some(f) => {
                    format!("{:.3}", time_cold(&st, &mut flush, reps, || fp32_run(f, (k, n), &xd, &mut y, &mut rot, rows, &st)))
                }
                None => "-".into(),
            };
            let base = if moe { tf.parse::<f64>().unwrap() } else { td };
            println!(
                "| {name} | {rows} | {} | {tp:.3} | {:.1} | {td:.3} | {tf} | {:.2}x |",
                ndm::pick_tile(rows, n),
                2.0 * rows as f64 * k as f64 * n as f64 / tp / 1e9,
                base / tp
            );
        }
    }
}
