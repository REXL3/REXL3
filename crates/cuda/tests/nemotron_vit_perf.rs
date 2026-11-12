//! Speed of the Nemotron 3.5 Super VL vision tower (`csrc/nemotron_vit.cu`): a per-stage table at 1,024, 4,096 and 13,312 patches
//! with the achieved TFLOP/s of every GEMM and attention call and the GB/s of every memory-bound stage, and the ceilings it is
//! read against (the device's `mma.sync` peak and cuBLAS on the tower's GEMM shapes, both through `glm_vision`).
//!
//! The tests are `#[ignore]`d benchmarks (`bench` in the name). Run on a GB10 under the device lock:
//!
//!     REXL3_NEMOTRON_PACK=<pack> cargo test --release -p gb-cuda --test nemotron_vit_perf -- --ignored --test-threads=1 --nocapture
//!
//! `REXL3_VIT_PERF_PATCHES` (default `1024,4096,13312`) and `REXL3_VIT_PERF_REPS` (default 3) tune the table. The pack is only read
//! for the vision tensors; the pixels are synthetic, so no pack-derived data is printed beyond timings.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use gb_cuda::glm_vision as gv;
use gb_cuda::nemotron_vit as nv;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::f64_to_f16;
use std::path::PathBuf;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

/// Random fp16 bit patterns uniform in [lo, hi).
fn rand_f16_bits(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            f64_to_f16(lo + (hi - lo) * u)
        })
        .collect()
}

/// An even patch grid `(gh, gw)` of `n` patches (a multiple of 4) closest to square, `gh <= gw`.
fn grid_of(n: usize) -> (usize, usize) {
    let mut best = None;
    let mut gh = 2;
    while gh * gh <= n {
        if n % gh == 0 && (n / gh) % 2 == 0 {
            best = Some((gh, n / gh));
        }
        gh += 2;
    }
    best.unwrap_or_else(|| panic!("{n} patches has no even grid"))
}

fn env_list(name: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(name) {
        Ok(v) => v.split(',').map(|x| x.trim().parse().unwrap()).collect(),
        Err(_) => default.to_vec(),
    }
}

/// (FLOP of one call, bytes of one call) of a stage at `n` patches, 0 where not meaningful.
fn stage_cost(name: &str, n: usize) -> (f64, f64) {
    let (t, np) = ((n + nv::PREFIX) as f64, (n / 4) as f64);
    let (h, mlp) = (nv::HIDDEN as f64, nv::MLP_HIDDEN as f64);
    let n = n as f64;
    match name {
        "patchify" => (0.0, n * 768.0 * 2.0 * 2.0),
        "patch_gemm" => (2.0 * n * h * 768.0, 0.0),
        "pos_embed" => (0.0, n * h * 4.0 * 3.0),
        "ln1" | "ln2" => (0.0, t * h * (4.0 + 2.0)),
        "qkv" => (2.0 * t * 3.0 * h * h, 0.0),
        "attn" => (4.0 * t * t * h, 0.0),
        "proj" => (2.0 * t * h * h, 0.0),
        "fc1" => (2.0 * t * mlp * h, 0.0),
        "fc2" => (2.0 * t * h * mlp, 0.0),
        "final_ln" => (0.0, n * h * (4.0 + 2.0)),
        "pixel_shuffle" => (0.0, n * h * 2.0 * 2.0),
        "mlp1_rms" => (0.0, np * 4.0 * h * 2.0 * 2.0),
        "mlp1_fc1" => (2.0 * np * nv::MLP1_HIDDEN as f64 * 4.0 * h, 0.0),
        "mlp1_fc2" => (2.0 * np * nv::LLM_HIDDEN as f64 * nv::MLP1_HIDDEN as f64, 0.0),
        _ => (0.0, 0.0),
    }
}

/// FLOP of the whole forward at `n` patches.
fn total_flop(n: usize) -> f64 {
    let mut f = 0.0;
    for (name, calls) in [
        ("patch_gemm", 1.0),
        ("qkv", 32.0),
        ("attn", 32.0),
        ("proj", 32.0),
        ("fc1", 32.0),
        ("fc2", 32.0),
        ("mlp1_fc1", 1.0),
        ("mlp1_fc2", 1.0),
    ] {
        f += stage_cost(name, n).0 * calls;
    }
    f
}

/// The per-stage table of the forward at each patch count (best of the reps by total time). One line per stage: calls, ms, share
/// of the total, GFLOP, TFLOP/s (GEMM and attention), GB/s (the memory-bound stages, the bytes their kernels must move).
#[test]
#[ignore = "benchmark"]
fn bench_stage_table() {
    gpu!();
    let Some(pack) = std::env::var_os("REXL3_NEMOTRON_PACK").map(PathBuf::from) else {
        eprintln!("REXL3_NEMOTRON_PACK is not set: skipping");
        return;
    };
    let patches = env_list("REXL3_VIT_PERF_PATCHES", &[1024, 4096, 13312]);
    let reps = env_list("REXL3_VIT_PERF_REPS", &[3])[0];
    let st = Stream::new().unwrap();
    let w = nv::Weights::load(&pack).unwrap();
    let maxn = *patches.iter().max().unwrap();
    let mut ws = nv::Workspace::new(maxn).unwrap();
    eprintln!("workspace {} B for {maxn} patches, weights {} B", ws.bytes(), w.bytes());
    for &n in &patches {
        let (gh, gw) = grid_of(n);
        let (height, width) = (gh * nv::PATCH, gw * nv::PATCH);
        assert_eq!(nv::patch_grid(height, width), Some((gh, gw)));
        let pix = DevBuf::from_slice(&rand_f16_bits(3 * height * width, 17 + n as u64, -2.0, 2.0)).unwrap();
        // one untimed forward warms the clocks, the L2 and the module
        nv::tower_forward(&w, &mut ws, &pix, height, width, None, &st).unwrap();
        st.sync().unwrap();
        let mut best: Option<nv::Profile> = None;
        let mut walls = Vec::new();
        for _ in 0..reps {
            let t0 = std::time::Instant::now();
            let p = nv::tower_forward_profiled(&w, &mut ws, &pix, height, width, &st).unwrap();
            walls.push(t0.elapsed().as_secs_f64() * 1e3);
            if best.as_ref().is_none_or(|b| p.total_us < b.total_us) {
                best = Some(p);
            }
        }
        let p = best.unwrap();
        println!("== {n} patches (grid {gh}x{gw}, {} tokens) ==", n / 4);
        println!("{:<14} {:>5} {:>10} {:>7} {:>10} {:>9} {:>8}", "stage", "calls", "ms", "share", "GFLOP", "TFLOP/s", "GB/s");
        for s in &p.stages {
            let (flop, bytes) = stage_cost(s.name, n);
            let (flop, bytes) = (flop * s.calls as f64, bytes * s.calls as f64);
            let secs = s.us * 1e-6;
            println!(
                "{:<14} {:>5} {:>10.3} {:>6.1}% {:>10.1} {:>9} {:>8}",
                s.name,
                s.calls,
                s.us / 1e3,
                100.0 * s.us / p.total_us,
                flop / 1e9,
                if flop > 0.0 { format!("{:.1}", flop / 1e12 / secs) } else { "-".into() },
                if bytes > 0.0 { format!("{:.0}", bytes / 1e9 / secs) } else { "-".into() },
            );
        }
        let flop = total_flop(n);
        println!(
            "TOTAL {n} patches: {:.2} ms (wall {:.2} ms), {:.1} GFLOP of GEMM and attention, {:.1} TFLOP/s",
            p.total_us / 1e3,
            walls.iter().cloned().fold(f64::MAX, f64::min),
            flop / 1e9,
            flop / 1e12 / (p.total_us * 1e-6)
        );
    }
}

/// The device's `mma.sync` peak and cuBLAS (the GLM tower's GEMM path) on the tower's GEMM shapes: the bars the tower's own kernels are
/// read against. `c_is_f32` rows are the residual GEMMs' output type.
#[test]
#[ignore = "benchmark"]
fn bench_ceilings() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut sink = DevBuf::zeroed(1024).unwrap();
    for blocks in [96usize, 192, 384] {
        let iters = 20_000usize;
        gv::mma_peak(&mut sink, blocks, 1000, &st).unwrap();
        st.sync().unwrap();
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            gv::mma_peak(&mut sink, blocks, iters, &st).unwrap();
            st.sync().unwrap();
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let flop = blocks as f64 * 8.0 * iters as f64 * 8.0 * 4096.0;
        println!("mma.sync m16n8k16 peak, {blocks} CTAs of 8 warps: {:.1} TFLOP/s", flop / 1e12 / best);
    }
    let mut cbws = DevBuf::zeroed(32 << 20).unwrap();
    for &n in &[1024usize, 4096, 13312] {
        let t = n + nv::PREFIX;
        for (name, m, k, nn, f32out) in [
            ("qkv", t, nv::HIDDEN, 3 * nv::HIDDEN, false),
            ("proj", t, nv::HIDDEN, nv::HIDDEN, true),
            ("fc1", t, nv::HIDDEN, nv::MLP_HIDDEN, false),
            ("fc2", t, nv::MLP_HIDDEN, nv::HIDDEN, true),
        ] {
            let a = DevBuf::from_slice(&rand_f16_bits(m * k, 1, -1.0, 1.0)).unwrap();
            let b = DevBuf::from_slice(&rand_f16_bits(nn * k, 2, -0.1, 0.1)).unwrap();
            let mut c = DevBuf::zeroed(m * nn * if f32out { 4 } else { 2 }).unwrap();
            gv::gemm(&a, k, &b, &mut c, nn, m, k, nn, f32out, &mut cbws, &st).unwrap();
            st.sync().unwrap();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t0 = std::time::Instant::now();
                gv::gemm(&a, k, &b, &mut c, nn, m, k, nn, f32out, &mut cbws, &st).unwrap();
                st.sync().unwrap();
                best = best.min(t0.elapsed().as_secs_f64());
            }
            let flop = 2.0 * m as f64 * k as f64 * nn as f64;
            println!("cublas {name:<5} M={m:<6} N={nn:<5} K={k:<5}: {:.3} ms, {:.1} TFLOP/s", best * 1e3, flop / 1e12 / best);
        }
    }
}
