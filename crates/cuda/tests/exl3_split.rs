//! Exact tensor-parallel cuts of an EXL3 layer (`gb_cuda::exl3_split`, lane E GLM Phase 2 item 4), bit for bit against the full layer:
//!
//! * S1 column strips with the parent's plan: [`Exl3Weights::forward_strip`] (in place) and [`Exl3Weights::select_strips_planned`] (a
//!   copy) give the full layer's columns of those strips, on the decode path at 1-128 rows, and [`Exl3Weights::forward_big_strip`] /
//!   [`PlannedExl3::forward_big`] on the big path up to 2,048 rows; also for strips narrower than 64 strips (the shared expert's halves),
//!   where the strip's own shape would pick another plan;
//! * S2 K ranges: [`Exl3Weights::forward_split_partials`] of the ranges of a split of the parent's plan (halves, uneven, one split),
//!   gathered into `[sk][m][N]` and finished by [`Exl3Weights::finish_partials`], give the full forward bit for bit.
//!
//! Shapes: every GLM-5.3-Flash cut of the plan (KDA qkv / o_proj, MLA q_b / o_proj, the shared expert, lm_head vocab halves) at the
//! sc_4.10bpw widths (8 / 6 bits) and the 2.25bpw pack's (2-4 bits), and a K split of two (one split a rank).
//!
//!     cargo test --release -p gb-cuda --test exl3_split -- --test-threads=1 --nocapture
#![cfg(not(gb_no_cuda))]
mod common;

use common::*;
use gb_cuda::exl3_linear::{Dtype, Exl3Weights};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::{Group, LoadedGroup};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use gb_exl3::{Bits, Codebook};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: not run");
            return;
        }
    };
}

fn normal(r: &mut Rng) -> f64 {
    let (a, b) = ((r.next_u64() >> 11) as f64 / (1u64 << 53) as f64, (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64);
    (-2.0 * a.max(1e-300).ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
}

fn scale_vec(r: &mut Rng, n: usize, mag: f64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            f16_to_f64(f64_to_f16(
                (if r.next_u64() & 1 == 1 { -1.0 } else { 1.0 }) * mag * (0.5 + (r.next_u64() >> 40) as f64 / 16777216.0),
            ))
        })
        .collect()
}

fn synth(k: usize, n: usize, bits: usize, cb: Codebook, bias: bool, seed: u64) -> Exl3Weights {
    let mut r = Rng(seed);
    let words = (k / 16) * (n / 16) * 16 * bits;
    let trellis: Vec<i16> = (0..words).map(|_| r.next_u64() as i16).collect();
    let group = Group {
        prefix: "synthetic".into(),
        bits: Bits::from_twice(2 * bits as u8).unwrap(),
        codebook: cb,
        k,
        n,
        in_scales: "suh",
        out_scales: "svh",
        bias,
        files: vec![],
        bytes: 0,
    };
    let svh = scale_vec(&mut r, n, 1.0 / (k as f64).sqrt());
    let b = bias.then(|| scale_vec(&mut r, n, 0.5));
    let g = LoadedGroup { group, trellis, kt: k / 16, nt: n / 16, suh: scale_vec(&mut r, k, 1.0), svh, bias: b };
    Exl3Weights::from_loaded(&g).unwrap()
}

/// fp16 input rows.
fn input(r: &mut Rng, rows: usize, k: usize) -> Vec<u16> {
    (0..rows * k).map(|_| f64_to_f16(normal(r))).collect()
}

fn full_decode(w: &Exl3Weights, st: &Stream, x: &DevBuf, m: usize) -> Vec<f32> {
    let mut ws = DevBuf::zeroed(w.workspace_bytes(m).max(16)).unwrap();
    let mut y = DevBuf::zeroed(m * w.n * 4).unwrap();
    w.forward(x, &mut y, m, Dtype::F16, Dtype::F32, &mut ws, st).unwrap();
    st.sync().unwrap();
    y.to_vec().unwrap()
}

fn full_big(w: &Exl3Weights, st: &Stream, x: &DevBuf, m: usize) -> Vec<f32> {
    let mut ws = DevBuf::zeroed(w.big_workspace_bytes(m).max(16)).unwrap();
    let mut y = DevBuf::zeroed(m * w.n * 4).unwrap();
    w.forward_big(x, &mut y, m, Dtype::F16, Dtype::F32, &mut ws, st).unwrap();
    st.sync().unwrap();
    y.to_vec().unwrap()
}

/// The columns of strips `ranges` of `[m, n]` rows.
fn cols(y: &[f32], m: usize, n: usize, ranges: &[(usize, usize)]) -> Vec<f32> {
    let mut out = Vec::new();
    for r in 0..m {
        for &(lo, hi) in ranges {
            out.extend_from_slice(&y[r * n + lo * 128..r * n + hi * 128]);
        }
    }
    out
}

fn same(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits()) {
        panic!("{what}: element {i} differs ({:e} vs {:e})", got[i], want[i]);
    }
}

const DECODE_ROWS: [usize; 10] = [1, 2, 7, 8, 16, 17, 33, 64, 65, 128];

/// S1 on the decode path: every strip range in place and as a planned copy equals the full layer's columns.
fn check_strips(w: &Exl3Weights, sets: &[Vec<(usize, usize)>], rows: &[usize], st: &Stream, seed: u64) -> usize {
    let (sk, wk) = w.plan().unwrap();
    let copies: Vec<_> = sets.iter().map(|s| w.select_strips_planned(s).unwrap()).collect();
    let mut r = Rng(seed);
    let mut n_checked = 0;
    for &m in rows {
        let xh = input(&mut r, m, w.k);
        let x = DevBuf::from_slice(&xh).unwrap();
        let full = full_decode(w, st, &x, m);
        for (si, set) in sets.iter().enumerate() {
            let want = cols(&full, m, w.n, set);
            // the planned copy
            let c = &copies[si];
            assert_eq!(c.plan, (sk, wk));
            let mut ws = DevBuf::zeroed(c.workspace_bytes(m).max(16)).unwrap();
            let mut y = DevBuf::zeroed(m * c.w.n * 4).unwrap();
            c.forward(&x, &mut y, m, Dtype::F16, Dtype::F32, &mut ws, st).unwrap();
            st.sync().unwrap();
            same(&format!("{}x{} {}-bit copy {set:?} at {m} rows", w.k, w.n, w.bits), &y.to_vec::<f32>().unwrap(), &want);
            // each range in place
            let mut got = vec![0f32; 0];
            let mut per: Vec<Vec<f32>> = Vec::new();
            for &(lo, hi) in set {
                let mut ws = DevBuf::zeroed(w.strip_workspace_bytes(hi - lo, m).max(16)).unwrap();
                let mut y = DevBuf::zeroed(m * (hi - lo) * 128 * 4).unwrap();
                w.forward_strip(lo, hi, &x, &mut y, m, Dtype::F16, Dtype::F32, &mut ws, st).unwrap();
                st.sync().unwrap();
                per.push(y.to_vec().unwrap());
            }
            for row in 0..m {
                for (j, &(lo, hi)) in set.iter().enumerate() {
                    let wd = (hi - lo) * 128;
                    got.extend_from_slice(&per[j][row * wd..(row + 1) * wd]);
                }
            }
            same(&format!("{}x{} {}-bit in place {set:?} at {m} rows", w.k, w.n, w.bits), &got, &want);
            n_checked += 1;
        }
    }
    n_checked
}

/// S2 on the decode path: the partials of each range gathered and finished equal the full forward.
fn check_splits(w: &Exl3Weights, cuts: &[Vec<(usize, usize)>], rows: &[usize], st: &Stream, seed: u64) -> usize {
    let (sk, _) = w.plan().unwrap();
    let mut r = Rng(seed);
    let mut n_checked = 0;
    for &m in rows {
        let xh = input(&mut r, m, w.k);
        let x = DevBuf::from_slice(&xh).unwrap();
        let full = full_decode(w, st, &x, m);
        for cut in cuts {
            let mut z = DevBuf::zeroed(sk * m * w.n * 4).unwrap();
            z.memset_async(0xCD, st).unwrap(); // every slot must be written by its range
            for &(s0, s1) in cut {
                let (k0, k1) = w.split_k_range(s0, s1).unwrap();
                let xr: Vec<u16> = (0..m).flat_map(|row| xh[row * w.k + k0..row * w.k + k1].to_vec()).collect();
                let xd = DevBuf::from_slice(&xr).unwrap();
                let mut ws = DevBuf::zeroed(w.split_partials_workspace_bytes(s0, s1, m).max(16)).unwrap();
                ws.memset_async(0xAB, st).unwrap();
                w.forward_split_partials(s0, s1, &xd, Dtype::F16, &mut z, s0 * m * w.n * 4, m, &mut ws, st).unwrap();
            }
            let mut y = DevBuf::zeroed(m * w.n * 4).unwrap();
            w.finish_partials(&z, m, &mut y, Dtype::F32, st).unwrap();
            st.sync().unwrap();
            same(
                &format!("{}x{} {}-bit (sk {sk}) K cut {cut:?} at {m} rows", w.k, w.n, w.bits),
                &y.to_vec::<f32>().unwrap(),
                &full,
            );
            n_checked += 1;
        }
    }
    n_checked
}

/// S1 on the big path: the strips in place and as a planned copy equal the full big forward's columns.
fn check_big_strips(w: &Exl3Weights, sets: &[Vec<(usize, usize)>], rows: &[usize], st: &Stream, seed: u64) -> usize {
    let copies: Vec<_> = sets.iter().map(|s| w.select_strips_planned(s).unwrap()).collect();
    let mut r = Rng(seed);
    let mut n_checked = 0;
    for &m in rows {
        let xh = input(&mut r, m, w.k);
        let x = DevBuf::from_slice(&xh).unwrap();
        let full = full_big(w, st, &x, m);
        for (si, set) in sets.iter().enumerate() {
            let want = cols(&full, m, w.n, set);
            let c = &copies[si];
            let mut ws = DevBuf::zeroed(c.big_workspace_bytes(m).max(16)).unwrap();
            let mut y = DevBuf::zeroed(m * c.w.n * 4).unwrap();
            c.forward_big(&x, &mut y, m, Dtype::F16, Dtype::F32, &mut ws, st).unwrap();
            st.sync().unwrap();
            same(&format!("{}x{} {}-bit big copy {set:?} at {m} rows", w.k, w.n, w.bits), &y.to_vec::<f32>().unwrap(), &want);
            if set.len() == 1 {
                let (lo, hi) = set[0];
                let mut ws = DevBuf::zeroed(w.big_workspace_bytes(m).max(16)).unwrap();
                let mut y = DevBuf::zeroed(m * (hi - lo) * 128 * 4).unwrap();
                w.forward_big_strip(lo, hi, &x, &mut y, m, Dtype::F16, Dtype::F32, &mut ws, st).unwrap();
                st.sync().unwrap();
                same(
                    &format!("{}x{} {}-bit big in place {set:?} at {m} rows", w.k, w.n, w.bits),
                    &y.to_vec::<f32>().unwrap(),
                    &want,
                );
            }
            n_checked += 1;
        }
    }
    n_checked
}

fn halves(strips: usize) -> Vec<Vec<(usize, usize)>> {
    vec![vec![(0, strips / 2)], vec![(strips / 2, strips)]]
}

/// KDA qkv `[q | k | v]` of 64 heads x 128: rank r's 32 heads are the strips (32 r .. 32 r + 32) of each third.
fn qkv_ranks(strips: usize) -> Vec<Vec<(usize, usize)>> {
    let t = strips / 3;
    (0..2).map(|r| (0..3).map(|j| (j * t + r * t / 2, j * t + (r + 1) * t / 2)).collect()).collect()
}

#[test]
fn column_strips_equal_the_full_layer_on_the_decode_path() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut total = 0;
    // (K, N, bits, what): sc_4.10bpw's widths and the 2.25bpw pack's
    for (i, &(k, n, bits, what)) in [
        (4096usize, 24576usize, 8usize, "KDA qkv"),
        (4096, 24576, 2, "KDA qkv 2.25bpw"),
        (1536, 16384, 8, "MLA q_b"),
        (1536, 16384, 4, "MLA q_b 2.25bpw"),
        (4096, 2048, 8, "shared gate/up (8 strips a half: another plan of its own)"),
        (4096, 2048, 3, "shared gate/up 2.25bpw"),
    ]
    .iter()
    .enumerate()
    {
        let w = synth(k, n, bits, Codebook::Mul1, false, 100 + i as u64);
        let mut sets = halves(n / 128);
        if what.starts_with("KDA qkv") {
            sets = qkv_ranks(n / 128);
        }
        sets.push(vec![(1, 3), (5, 6)]); // an odd selection
        let c = check_strips(&w, &sets, &DECODE_ROWS, &st, 200 + i as u64);
        eprintln!("{what} {k}x{n} {bits}-bit (plan {:?}): {c} strip cases bit-identical", w.plan().unwrap());
        total += c;
    }
    // lm_head vocab halves (6 bits, 1,210 strips): the head's rows only
    let w = synth(4096, 154880, 6, Codebook::Mul1, false, 300);
    let c = check_strips(&w, &halves(1210), &[1, 4, 8], &st, 301);
    eprintln!("lm_head 4096x154880 6-bit (plan {:?}): {c} strip cases bit-identical", w.plan().unwrap());
    total += c;
    // a bias, other codebooks
    for (i, cb) in [Codebook::Inst3, Codebook::Mcg].into_iter().enumerate() {
        let w = synth(4096, 2048, 4, cb, true, 400 + i as u64);
        total += check_strips(&w, &halves(16), &[1, 17, 128], &st, 410 + i as u64);
    }
    eprintln!("{total} strip cases bit-identical to the full layer's columns");
}

#[test]
fn k_split_partials_finish_to_the_full_layer() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut total = 0;
    for (i, &(k, n, bits, what)) in [
        (8192usize, 4096usize, 8usize, "KDA o_proj"),
        (8192, 4096, 3, "KDA o_proj 2.25bpw"),
        (16384, 4096, 8, "MLA o_proj"),
        (16384, 4096, 2, "MLA o_proj 2.25bpw"),
        (2048, 4096, 8, "shared down"),
        (2048, 4096, 2, "shared down 2.25bpw"),
        (1024, 2048, 4, "a K split of two (one split a rank)"),
    ]
    .iter()
    .enumerate()
    {
        let w = synth(k, n, bits, Codebook::Mul1, i % 2 == 1, 500 + i as u64);
        let (sk, _) = w.plan().unwrap();
        assert!(sk >= 2, "{what}: plan {:?} has no K split", w.plan());
        let mut cuts = vec![vec![(0, sk / 2), (sk / 2, sk)], vec![(0, 1), (1, sk)], vec![(0, sk)]];
        if sk > 2 {
            cuts.push((0..sk).map(|s| (s, s + 1)).collect()); // every split alone
        }
        let c = check_splits(&w, &cuts, &DECODE_ROWS, &st, 600 + i as u64);
        eprintln!("{what} {k}x{n} {bits}-bit (plan {:?}): {c} K-cut cases bit-identical", w.plan().unwrap());
        total += c;
    }
    eprintln!("{total} K-cut cases bit-identical to the full forward");
}

#[test]
fn column_strips_equal_the_full_layer_on_the_big_path() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = [1usize, 64, 129, 512, 2048];
    let mut total = 0;
    for (i, &(k, n, bits, what)) in [
        (4096usize, 24576usize, 8usize, "KDA qkv"),
        (4096, 24576, 2, "KDA qkv 2.25bpw"),
        (1536, 16384, 8, "MLA q_b"),
        (4096, 2048, 8, "shared gate/up"),
        (4096, 2048, 3, "shared gate/up 2.25bpw"),
    ]
    .iter()
    .enumerate()
    {
        let w = synth(k, n, bits, Codebook::Mul1, false, 700 + i as u64);
        let mut sets = halves(n / 128);
        if what.starts_with("KDA qkv") {
            sets.extend(qkv_ranks(n / 128));
        }
        let c = check_big_strips(&w, &sets, &rows, &st, 800 + i as u64);
        eprintln!("{what} {k}x{n} {bits}-bit (plan {:?}): {c} big-path strip cases bit-identical", w.plan().unwrap());
        total += c;
    }
    eprintln!("{total} big-path strip cases bit-identical to the full big forward's columns");
}

/// Lane E, the decode chain rot_in -> linear -> rot_out with and without the programmatic-dependent-launch attribute (`set_pdl`), in
/// captured graphs as a decode step runs: `pairs` pairs of layers (up then down, 1 row, 8 bits) replayed. `warm`: one pair of weights reused
/// (they stay in the L2); `cold`: a distinct pair each time (200 MB of weights, streamed from DRAM like a decode step's). Microseconds a
/// layer, and the last buffer asserted bit-equal between every run.
#[test]
#[ignore]
fn bench_pdl_chain() {
    use gb_cuda::graph::Graph;
    use gb_cuda::moe::EventTimer;
    gpu!();
    let st = Stream::new_nonblocking().unwrap();
    let t = EventTimer::new(2).unwrap();
    for &(k, n, bits, what) in [
        (4096usize, 12288usize, 8usize, "4096 <-> 12288"),
        (4096, 2048, 8, "4096 <-> 2048 (the shared expert's shape)"),
        (4096, 2048, 4, "4096 <-> 2048"),
    ]
    .iter()
    {
        for cold in [false, true] {
            let nw = if cold { 12 / (n / 2048).min(6) } else { 1 };
            let ups: Vec<Exl3Weights> = (0..nw).map(|i| synth(k, n, bits, Codebook::Mul1, false, 1000 + 2 * i as u64)).collect();
            let downs: Vec<Exl3Weights> =
                (0..nw).map(|i| synth(n, k, bits, Codebook::Mul1, false, 1001 + 2 * i as u64)).collect();
            let mut r = Rng(1002);
            let x0 = DevBuf::from_slice(&input(&mut r, 1, k)).unwrap();
            let ws_bytes = ups[0].workspace_bytes(1).max(downs[0].workspace_bytes(1)).max(16);
            let pairs = if cold { nw } else { 24 };
            let mut results: Vec<Vec<u16>> = Vec::new();
            let mut times = Vec::new();
            for on in [false, true, false, true] {
                gb_cuda::exl3_linear::set_pdl(on).unwrap();
                let mut a = DevBuf::zeroed(k * 2).unwrap();
                let mut mid = DevBuf::zeroed(n * 2).unwrap();
                let mut ws = DevBuf::zeroed(ws_bytes).unwrap();
                let g = Graph::capture(&st, || -> Result<(), gb_cuda::CudaError> {
                    let mut src = &x0;
                    for i in 0..pairs {
                        ups[i % nw].forward(src, &mut mid, 1, Dtype::F16, Dtype::F16, &mut ws, &st)?;
                        downs[i % nw].forward(&mid, &mut a, 1, Dtype::F16, Dtype::F16, &mut ws, &st)?;
                        src = &a;
                    }
                    Ok(())
                })
                .unwrap_or_else(|_| panic!("capture failed"));
                for _ in 0..3 {
                    g.launch(&st).unwrap();
                }
                st.sync().unwrap();
                let mut v = Vec::new();
                for _ in 0..40 {
                    t.record(0, &st).unwrap();
                    g.launch(&st).unwrap();
                    t.record(1, &st).unwrap();
                    st.sync().unwrap();
                    v.push(t.ms(0, 1).unwrap() as f64 * 1e3 / (2 * pairs) as f64);
                }
                v.sort_by(|x, y| x.partial_cmp(y).unwrap());
                times.push(v[v.len() / 2]);
                results.push(a.to_vec::<u16>().unwrap());
            }
            gb_cuda::exl3_linear::set_pdl(true).unwrap();
            for w in results.windows(2) {
                assert!(w[0] == w[1], "{what}: PDL changes the bits");
            }
            println!(
                "PDLCHAIN {what} {bits}-bit {}: off {:.2} / {:.2} us a layer, PDL on {:.2} / {:.2} us (bits identical) -> {:+.2} us a layer",
                if cold { "COLD (distinct weights)" } else { "warm" },
                times[0],
                times[2],
                times[1],
                times[3],
                (times[1] + times[3]) / 2.0 - (times[0] + times[2]) / 2.0
            );
        }
    }
}
