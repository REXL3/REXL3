//! Performance table and bit-identity gate of the dense bf16 GEMV at the shapes the Nemotron 3.5 Super VL engine calls it with
//! (`csrc/nemotron_dense.cu`): the 1.07 GB `lm_head`, the MTP head's restricted sub-head (32k to 64k rows), `eh_proj` and its q / k / v / o.
//!
//! One 1 GiB weight serves every shape (the first `n * k` bf16 of it), so the cost of a shape is only its kernel. The gate (not ignored)
//! checks that on the lm_head shape a row of `y` is bitwise the same at m = 1, 4, 16, 33, 64 and that the whole m = 64 output keeps the
//! digest the pre-change kernel produced (`LM_HEAD_DIGEST`).
//!
//! The table (ignored, `bench` in the name): median and minimum of 15 calls per shape and row count, **cold** (a 128 MiB device copy
//! before every timed call, so L2 holds no lines of the weight, as in a decode step where other layers run in between), with the
//! achieved GB/s of bf16 weight bytes read once. m = 1 is the decode row, 4 a verify window, 16 a draft tree, 64 a prefill slice.
//!
//!     cargo test -p gb-cuda --release --test nemotron_dense_perf -- --test-threads=1 --nocapture
//!     cargo test -p gb-cuda --release --test nemotron_dense_perf -- --ignored --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]
#![allow(clippy::needless_range_loop)]
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

/// The lm_head: vocabulary 131,072 rows of the 4,096-wide hidden state.
const LM_N: usize = 131_072;
const LM_K: usize = 4_096;

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

fn bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// The 1 GiB weight of the lm_head shape: a fast deterministic bf16 fill in [-1, 1).
fn lm_head_weight() -> DevBuf {
    let mut rng = Rng(4242);
    let w: Vec<u16> = (0..LM_N * LM_K)
        .map(|i| {
            let v = (rng.next_u64() ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 40;
            bf16_bits((v as f32 / 8388608.0) - 1.0)
        })
        .collect();
    DevBuf::from_slice(&w).unwrap()
}

fn digest(y: &[f32]) -> u64 {
    gb_exl3::fnv1a(y.iter().flat_map(|v| v.to_bits().to_le_bytes()))
}

/// FNV-1a of the f32 bits of `y` at m = 64 on the lm_head shape with `x = Rng(777).vec(64 * 4096, -1, 1)`, captured from the kernel as
/// it was before the row-blocked rewrite (commit `e6ac52e`). Set `REXL3_DENSE_PRINT_DIGESTS=1` to print the current value.
const LM_HEAD_DIGEST: u64 = 0xa345a5c173248927;

#[test]
fn lm_head_rows_are_bitwise_the_same_at_every_m() {
    gpu!();
    let st = Stream::new().unwrap();
    let dw = lm_head_weight();
    let m64 = 64usize;
    let x = Rng(777).vec(m64 * LM_K, -1.0, 1.0);
    let call = |rows: &[f32], m: usize| -> Vec<f32> {
        let dx = DevBuf::from_slice(rows).unwrap();
        let mut dy = DevBuf::zeroed(m * LM_N * 4).unwrap();
        nd::gemv_bf16(&dw, &dx, &mut dy, m, LM_N, LM_K, &st).unwrap();
        st.sync().unwrap();
        dy.to_vec::<f32>().unwrap()
    };
    let full = call(&x, m64);
    let got = digest(&full);
    if std::env::var("REXL3_DENSE_PRINT_DIGESTS").is_ok() {
        eprintln!("LM_HEAD_DIGEST = {got:#018x}");
    } else {
        assert_eq!(got, LM_HEAD_DIGEST, "the m = 64 lm_head output moved: the arithmetic of the kernel changed");
    }
    let mut bad = 0usize;
    for m in [1usize, 4, 16, 33] {
        let y = call(&x[..m * LM_K], m);
        bad += y.iter().zip(&full).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    }
    for r in [17usize, 63] {
        let y = call(&x[r * LM_K..(r + 1) * LM_K], 1);
        bad += (0..LM_N).filter(|&c| y[c].to_bits() != full[r * LM_N + c].to_bits()).count();
    }
    eprintln!("lm_head: rows at m = 1, 4, 16, 33 and single rows 17, 63 vs the m = 64 call: {bad} differing outputs");
    assert_eq!(bad, 0, "row invariance broken on the lm_head shape");
}

/// Median and minimum milliseconds of `iters` calls of `f`, each preceded by `flush` (not timed).
fn cold_ms(st: &Stream, warm: usize, iters: usize, mut flush: impl FnMut(), mut f: impl FnMut()) -> (f32, f32) {
    for _ in 0..warm {
        flush();
        f();
    }
    st.sync().unwrap();
    let mk = || {
        let mut e = core::ptr::null_mut::<c_void>();
        assert_eq!(unsafe { ev::cudaEventCreateWithFlags(&mut e, 0) }, 0);
        e
    };
    let mut v = Vec::with_capacity(iters);
    for _ in 0..iters {
        flush();
        let (a, b) = (mk(), mk());
        unsafe { assert_eq!(ev::cudaEventRecord(a, st.raw()), 0) };
        f();
        unsafe { assert_eq!(ev::cudaEventRecord(b, st.raw()), 0) };
        unsafe { assert_eq!(ev::cudaEventSynchronize(b), 0) };
        let mut ms = 0f32;
        unsafe { assert_eq!(ev::cudaEventElapsedTime(&mut ms, a, b), 0) };
        unsafe {
            ev::cudaEventDestroy(a);
            ev::cudaEventDestroy(b);
        }
        v.push(ms);
    }
    v.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let h = iters / 2;
    let med = if iters % 2 == 0 { (v[h - 1] + v[h]) / 2.0 } else { v[h] };
    (med, v[0])
}

#[test]
#[ignore]
fn bench_dense_gemv_shapes() {
    gpu!();
    let st = Stream::new().unwrap();
    let dw = lm_head_weight();
    let xmax = 128usize;
    let x = Rng(99).vec(xmax * 8192, -1.0, 1.0);
    let dx = DevBuf::from_slice(&x).unwrap();
    let mut dy = DevBuf::zeroed(xmax * LM_N * 4).unwrap();
    let flush_src = DevBuf::zeroed(128 << 20).unwrap();
    let mut flush_dst = DevBuf::zeroed(128 << 20).unwrap();

    // (name, n, k, row counts)
    let kv = 2 * 128; // N_KV_HEADS * HEAD_DIM
    let shapes: [(&str, usize, usize, &[usize]); 7] = [
        ("lm_head", LM_N, LM_K, &[1, 2, 3, 4, 5, 8, 16, 32, 48, 64, 65, 128]),
        ("sub-head 64k", 65_536, LM_K, &[1, 4, 16, 64]),
        ("sub-head 48k", 49_152, LM_K, &[1, 4, 16, 64]),
        ("sub-head 32k", 32_768, LM_K, &[1, 4, 16, 64]),
        ("eh_proj", 4_096, 8_192, &[1, 4, 16, 64]),
        ("q / o", 4_096, 4_096, &[1, 4, 16, 64]),
        ("k / v", kv, 4_096, &[1, 4, 16, 64]),
    ];
    eprintln!("{:14} {:>7} {:>5} {:>4} {:>10} {:>9} {:>9}", "shape", "n", "k", "m", "median ms", "min ms", "GB/s");
    for (name, n, k, ms) in shapes {
        let wbytes = (n * k * 2) as f64;
        for &m in ms {
            let (med, min) = cold_ms(
                &st,
                3,
                15,
                || flush_dst.copy_from(&flush_src, 128 << 20, &st).unwrap(),
                || nd::gemv_bf16(&dw, &dx, &mut dy, m, n, k, &st).unwrap(),
            );
            eprintln!("{name:14} {n:7} {k:5} {m:4} {med:10.4} {min:9.4} {:9.1}", wbytes / (med as f64 * 1e-3) / 1e9);
        }
    }
}
