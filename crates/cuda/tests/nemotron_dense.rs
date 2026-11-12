//! Dense bf16 GEMV: agreement with a CPU float64 reference at every supported k, bitwise row invariance, and the lm_head
//! shape's bandwidth. GPU tests skip when there is no device.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::Rng;
use core::ffi::c_void;
use gb_cuda::nemotron_dense as nd;
use gb_cuda::{device_available, DevBuf, Stream};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

/// CUDA events with timing (the crate's `rt::Event` is created without timing).
mod ev {
    use core::ffi::c_void;
    extern "C" {
        pub fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: u32) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
        pub fn cudaEventDestroy(e: *mut c_void) -> i32;
    }
}

/// Median milliseconds per call of `f` over `iters` timed calls, after `warm` untimed calls. Each call is timed by its own
/// pair of CUDA events on `st`.
fn median_ms(st: &Stream, warm: usize, iters: usize, mut f: impl FnMut()) -> f32 {
    for _ in 0..warm {
        f();
    }
    st.sync().unwrap();
    let mut pairs = Vec::with_capacity(iters);
    for _ in 0..iters {
        let (mut a, mut b) = (core::ptr::null_mut::<c_void>(), core::ptr::null_mut::<c_void>());
        unsafe {
            assert_eq!(ev::cudaEventCreateWithFlags(&mut a, 0), 0);
            assert_eq!(ev::cudaEventCreateWithFlags(&mut b, 0), 0);
            assert_eq!(ev::cudaEventRecord(a, st.raw()), 0);
        }
        f();
        unsafe { assert_eq!(ev::cudaEventRecord(b, st.raw()), 0) };
        pairs.push((a, b));
    }
    st.sync().unwrap();
    let mut ms: Vec<f32> = pairs
        .iter()
        .map(|&(a, b)| {
            let mut t = 0f32;
            unsafe { assert_eq!(ev::cudaEventElapsedTime(&mut t, a, b), 0) };
            t
        })
        .collect();
    for (a, b) in pairs {
        unsafe {
            ev::cudaEventDestroy(a);
            ev::cudaEventDestroy(b);
        }
    }
    ms.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let h = iters / 2;
    if iters % 2 == 0 {
        (ms[h - 1] + ms[h]) / 2.0
    } else {
        ms[h]
    }
}

/// f32 -> bf16 bits, round to nearest even (inputs are finite).
fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn bf16_val(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// Random x `[m, k]` in [-1, 1) and W `[n, k]` as bf16 bits.
fn random_case(m: usize, n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<u16>) {
    let mut rng = Rng(seed);
    let x = rng.vec(m * k, -1.0, 1.0);
    let w = rng.vec(n * k, -1.0, 1.0).iter().map(|&v| bf16_bits(v)).collect();
    (x, w)
}

/// fp64 reference `[m, cols.len()]` (row-major) for the listed columns of W, from the exact bf16 and f32 inputs.
fn reference(x: &[f32], w: &[u16], m: usize, k: usize, cols: &[usize]) -> Vec<f64> {
    let mut out = vec![0f64; m * cols.len()];
    for (ci, &c) in cols.iter().enumerate() {
        let wr: Vec<f64> = w[c * k..(c + 1) * k].iter().map(|&h| bf16_val(h) as f64).collect();
        for r in 0..m {
            let xr = &x[r * k..(r + 1) * k];
            out[r * cols.len() + ci] = xr.iter().zip(&wr).map(|(&a, &b)| a as f64 * b).sum();
        }
    }
    out
}

/// Worst `|got - want| / max(|want|, 0.01 * rms(want))` and the worst `|got - want| / max|want|`.
fn errors(got: &[f32], want: &[f64]) -> (f64, f64) {
    let rms = (want.iter().map(|v| v * v).sum::<f64>() / want.len() as f64).sqrt();
    let amax = want.iter().fold(0f64, |a, v| a.max(v.abs()));
    let (mut rel, mut norm) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let d = (*g as f64 - w).abs();
        rel = rel.max(d / w.abs().max(0.01 * rms));
        norm = norm.max(d / amax);
    }
    (rel, norm)
}

fn run(x: &[f32], w: &[u16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let st = Stream::new().unwrap();
    let dx = DevBuf::from_slice(x).unwrap();
    let dw = DevBuf::from_slice(w).unwrap();
    let mut dy = DevBuf::zeroed(m * n * 4).unwrap();
    nd::gemv_bf16(&dw, &dx, &mut dy, m, n, k, &st).unwrap();
    st.sync().unwrap();
    dy.to_vec::<f32>().unwrap()
}

#[test]
fn matches_a_float64_reference_at_every_supported_k() {
    gpu!();
    let cases = [
        (1, 300, 1024),
        (64, 300, 1024),
        (1, 257, 2688),
        (64, 130, 2688),
        (1, 512, 4096),
        (64, 96, 4096),
        (3, 100, 5376),
        (64, 100, 5376),
        (1, 4096, 8192),
        (64, 64, 8192),
        (5, 70, 4096),   // partial row tile (MT = 8, rows 5..8 clamped)
        (100, 40, 1024), // two row tiles: 64 + 36 (clamped)
    ];
    for (i, &(m, n, k)) in cases.iter().enumerate() {
        let (x, w) = random_case(m, n, k, 1000 + i as u64);
        let got = run(&x, &w, m, n, k);
        let cols: Vec<usize> = (0..n).collect();
        let want = reference(&x, &w, m, k, &cols);
        let (rel, norm) = errors(&got, &want);
        eprintln!("m={m:3} n={n:6} k={k:5}: max rel err {rel:.2e} (floor 0.01*rms), max err / max|y| {norm:.2e}");
        assert!(rel <= 1e-4, "m={m} n={n} k={k}: max relative error {rel}");
    }
}

#[test]
fn row_r_is_bitwise_the_same_at_m1_and_m64() {
    gpu!();
    for (n, k, m) in [(512usize, 4096usize, 64usize), (257, 2688, 5), (300, 1024, 100), (96, 5376, 64), (64, 8192, 33)] {
        let (x, w) = random_case(m, n, k, 77 + k as u64);
        let batched = run(&x, &w, m, n, k);
        let mut mismatches = 0usize;
        for r in 0..m {
            let row = run(&x[r * k..(r + 1) * k], &w, 1, n, k);
            let a: Vec<u32> = row.iter().map(|v| v.to_bits()).collect();
            let b: Vec<u32> = batched[r * n..(r + 1) * n].iter().map(|v| v.to_bits()).collect();
            mismatches += a.iter().zip(&b).filter(|(p, q)| p != q).count();
        }
        eprintln!("n={n} k={k}: rows 0..{m} at m=1 vs m={m}: {mismatches} differing f32 bit patterns");
        assert_eq!(mismatches, 0, "n={n} k={k}: row invariance broken");
    }
}

#[test]
fn lm_head_shape_bandwidth() {
    gpu!();
    let (n, k) = (131072usize, 4096usize);
    let st = Stream::new().unwrap();
    let mut rng = Rng(4242);
    // a fast deterministic bf16 fill for the 1 GiB weight (only the spot check below reads these values back)
    let w: Vec<u16> = (0..n * k)
        .map(|i| {
            let v = (rng.next_u64() ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 40;
            bf16_bits((v as f32 / 8388608.0) - 1.0)
        })
        .collect();
    let dw = DevBuf::from_slice(&w).unwrap();
    let wbytes = (n * k * 2) as f64;

    // device copy bandwidth of the same buffer, in both byte conventions
    let mut dcopy = DevBuf::zeroed(n * k * 2).unwrap();
    let copy_ms = median_ms(&st, 3, 10, || dcopy.copy_from(&dw, n * k * 2, &st).unwrap());
    eprintln!(
        "device D2D copy of {:.2} GiB: {:.1} GB/s counting bytes copied once, {:.1} GB/s counting read + write",
        wbytes / (1u64 << 30) as f64,
        wbytes / (copy_ms as f64 * 1e-3) / 1e9,
        2.0 * wbytes / (copy_ms as f64 * 1e-3) / 1e9
    );

    // 64 spread columns for the float64 spot check
    let cols: Vec<usize> = (0..64).map(|i| (i * 2039 + 11) % n).collect();
    for m in [1usize, 64] {
        let x = rng.vec(m * k, -1.0, 1.0);
        let dx = DevBuf::from_slice(&x).unwrap();
        let mut dy = DevBuf::zeroed(m * n * 4).unwrap();
        let ms = median_ms(&st, 3, 10, || nd::gemv_bf16(&dw, &dx, &mut dy, m, n, k, &st).unwrap());
        let gbps = wbytes / (ms as f64 * 1e-3) / 1e9;
        st.sync().unwrap();
        let got: Vec<f32> = dy.to_vec().unwrap();
        let mut sub = Vec::with_capacity(m * cols.len());
        for r in 0..m {
            sub.extend(cols.iter().map(|&c| got[r * n + c]));
        }
        let want = reference(&x, &w, m, k, &cols);
        let (rel, norm) = errors(&sub, &want);
        eprintln!(
            "lm_head n={n} k={k} m={m:2}: median {ms:.3} ms/call, {gbps:.1} GB/s of bf16 weights read once; \
             spot check of 64 columns: max rel err {rel:.2e}, max err / max|y| {norm:.2e}"
        );
        assert!(rel <= 1e-4, "lm_head m={m}: max relative error {rel}");
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// Bit-exactness gates (additive). The kernel's contract is not "close to a float64 reference" but one fixed order of f32 operations per
// output (see the header of csrc/nemotron_dense.cu), so that every launch shape, every row-block layout and every later rewrite give the
// same bits. `exact_dot` is that order written out on the CPU.
// ---------------------------------------------------------------------------------------------------------------------------------

/// The row counts the gates cover (they include the dispatch edges of the kernel: 4 | 5, 64 | 65).
const GATE_MS: [usize; 11] = [1, 2, 3, 4, 7, 8, 16, 33, 64, 65, 128];

/// FNV-1a 64 over the little-endian bytes of the f32 bit patterns.
fn digest(y: &[f32]) -> u64 {
    gb_exl3::fnv1a(y.iter().flat_map(|v| v.to_bits().to_le_bytes()))
}

/// xor tree over 16 stripes (offsets 8, 4, 2, 1), the value every stripe ends with (they are equal: the sums commute).
fn stripe_tree(a: &[f32; 16]) -> f32 {
    let mut v = *a;
    for off in [8usize, 4, 2, 1] {
        let mut nv = v;
        for s in 0..16 {
            nv[s] = v[s] + v[s ^ off];
        }
        v = nv;
    }
    v[0]
}

/// One output of the kernel: stripe `s` of a row owns the 8-element chunks `s + 16 i`; a chunk is two chains of four products
/// (`fmul`, then three `fma`), the sum of the chains is added to the stripe's accumulator; every 8 iterations the stripes are combined by
/// the xor tree and stripe 0 keeps the total (the others restart at zero); the result is the xor tree of the stripes at the end.
fn exact_dot(x: &[f32], w: &[u16]) -> f32 {
    let steps = x.len() / 128;
    let mut acc = [0f32; 16];
    for i in 0..steps {
        for s in 0..16 {
            let b = (s + 16 * i) * 8;
            let wv = |e: usize| bf16_val(w[b + e]);
            let mut p = x[b] * wv(0);
            p = x[b + 1].mul_add(wv(1), p);
            p = x[b + 2].mul_add(wv(2), p);
            p = x[b + 3].mul_add(wv(3), p);
            let mut q = x[b + 4] * wv(4);
            q = x[b + 5].mul_add(wv(5), q);
            q = x[b + 6].mul_add(wv(6), q);
            q = x[b + 7].mul_add(wv(7), q);
            acc[s] += p + q;
        }
        if (i + 1) % 8 == 0 {
            let t = stripe_tree(&acc);
            acc = [0f32; 16];
            acc[0] = t;
        }
    }
    stripe_tree(&acc)
}

/// Like `random_case`, but a third of the entries are exact zeros (some of them negative zeros in W), so that signed zeros and
/// zero chains go through the reductions.
fn zero_laden_case(m: usize, n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<u16>) {
    let mut rng = Rng(seed);
    let x = rng.vec(m * k, -1.0, 1.0).into_iter().map(|v| if rng.next_u64() % 3 == 0 { 0.0 } else { v }).collect();
    let w = rng
        .vec(n * k, -1.0, 1.0)
        .into_iter()
        .map(|v| match rng.next_u64() % 6 {
            0 => 0x0000u16,
            1 => 0x8000u16,
            _ => bf16_bits(v),
        })
        .collect();
    (x, w)
}

/// A weight uploaded once, called at many shapes.
struct Weight {
    dw: DevBuf,
    st: Stream,
}

impl Weight {
    fn new(w: &[u16]) -> Self {
        Self { dw: DevBuf::from_slice(w).unwrap(), st: Stream::new().unwrap() }
    }
    fn call(&self, x: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        assert_eq!(x.len(), m * k);
        let dx = DevBuf::from_slice(x).unwrap();
        let mut dy = DevBuf::zeroed(m * n * 4).unwrap();
        nd::gemv_bf16(&self.dw, &dx, &mut dy, m, n, k, &self.st).unwrap();
        self.st.sync().unwrap();
        dy.to_vec::<f32>().unwrap()
    }
}

#[test]
fn every_output_equals_the_exact_cpu_model() {
    gpu!();
    // n is not a multiple of the 16- and 64-column tiles; m straddles the 4 | 5 and 64 | 65 dispatch edges
    let cases = [
        (1usize, 77usize, 1024usize),
        (2, 33, 2688),
        (3, 130, 4096),
        (4, 131, 5376),
        (5, 65, 8192),
        (8, 70, 1024),
        (9, 129, 4096),
        (16, 100, 2688),
        (17, 90, 5376),
        (33, 71, 8192),
        (48, 66, 1024),
        (49, 130, 4096),
        (64, 100, 2688),
        (65, 70, 5376),
        (100, 40, 8192),
        (128, 64, 4096),
    ];
    for (i, &(m, n, k)) in cases.iter().enumerate() {
        for zeros in [false, true] {
            let (x, w) = if zeros { zero_laden_case(m, n, k, 5000 + i as u64) } else { random_case(m, n, k, 4000 + i as u64) };
            let got = Weight::new(&w).call(&x, m, n, k);
            let mut bad = 0usize;
            for r in 0..m {
                for c in 0..n {
                    let want = exact_dot(&x[r * k..(r + 1) * k], &w[c * k..(c + 1) * k]);
                    if got[r * n + c].to_bits() != want.to_bits() {
                        bad += 1;
                    }
                }
            }
            eprintln!("m={m:3} n={n:3} k={k:4} zeros={zeros}: {bad} of {} outputs differ from the exact CPU model", m * n);
            assert_eq!(bad, 0, "m={m} n={n} k={k} zeros={zeros}");
        }
    }
}

#[test]
fn rows_are_bitwise_independent_of_m_at_every_shape() {
    gpu!();
    // (n, k): the engine's widths, n not a multiple of the tile (16, 64, 128 columns)
    let mut shapes = vec![];
    for &k in &nd::SUPPORTED_K {
        shapes.push((37usize, k));
        shapes.push((101, k));
        shapes.push((197, k));
    }
    shapes.push((4096 + 37, 4096));
    shapes.push((256, 4096));
    for &(n, k) in &shapes {
        let mmax = *GATE_MS.last().unwrap();
        let (x, w) = random_case(mmax, n, k, 31 + (n * 7 + k) as u64);
        let dev = Weight::new(&w);
        let full = dev.call(&x, mmax, n, k);
        let mut mismatches = 0usize;
        for &m in &GATE_MS {
            let y = dev.call(&x[..m * k], m, n, k);
            for r in 0..m {
                mismatches += (0..n).filter(|&c| y[r * n + c].to_bits() != full[r * n + c].to_bits()).count();
            }
        }
        // single rows taken from the middle and the end of the big call
        for r in [0usize, 5, 63, 64, 100, mmax - 1] {
            let y = dev.call(&x[r * k..(r + 1) * k], 1, n, k);
            mismatches += (0..n).filter(|&c| y[c].to_bits() != full[r * n + c].to_bits()).count();
        }
        eprintln!(
            "n={n:5} k={k:4}: rows at m = {GATE_MS:?} and single rows vs the m = {mmax} call: {mismatches} differing outputs"
        );
        assert_eq!(mismatches, 0, "n={n} k={k}: row invariance broken");
    }
}

/// Digests of whole outputs, captured from the kernel as it was before the row-blocked rewrite (the 16-stripe kernel of commit
/// `e6ac52e`); the exact CPU model above reproduces the same bits, this keeps the check of every tile edge cheap and pack free.
/// Set `REXL3_DENSE_PRINT_DIGESTS=1` to print the current values (to re-capture after an intended change of the arithmetic).
const GOLDEN: [(usize, usize, usize, u64); 9] = [
    (1, 301, 1024, 0x1dd59b87e3596afa),
    (3, 257, 2688, 0xd051fe22aae71d0c),
    (5, 129, 4096, 0x0cdc718bccf9ed13),
    (16, 300, 5376, 0xa1414431eac82c53),
    (33, 100, 8192, 0x38f82a6ddae47c68),
    (64, 257, 4096, 0x26cb53d8cb050bde),
    (65, 70, 2688, 0xce350beef54c6a70),
    (100, 130, 1024, 0x0fa2b0407f99f02e),
    (128, 64, 8192, 0x289ec401ed4a0758),
];

#[test]
fn outputs_keep_their_golden_digests() {
    gpu!();
    let print = std::env::var("REXL3_DENSE_PRINT_DIGESTS").is_ok();
    let mut bad = 0;
    for (i, &(m, n, k, want)) in GOLDEN.iter().enumerate() {
        let (x, w) = random_case(m, n, k, 9000 + i as u64);
        let got = digest(&Weight::new(&w).call(&x, m, n, k));
        if print {
            eprintln!("    ({m}, {n}, {k}, {got:#018x}),");
        } else if got != want {
            eprintln!("m={m} n={n} k={k}: digest {got:#018x}, golden {want:#018x}");
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "outputs moved: the arithmetic of the kernel changed");
}
