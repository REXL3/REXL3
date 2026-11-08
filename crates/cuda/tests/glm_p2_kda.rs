//! The P2 chunked KDA recurrence (`csrc/glm_p2_kda.cu`, `kda::prefill_chunked`; `docs/GLM_PREFILL_P2.md` section 8.1), against a float64
//! sequential reference of the same inputs with P1's own row-by-row state kernel beside it, and its bit rules: one call == any split at
//! multiples of 16 rows, a partial chunk's rows == the same rows of a longer call, both block shapes alike, two runs alike.
//!
//! Inputs are synthetic with GLM's layout and statistics: q, k l2-normalised per head (q scaled by 1/sqrt(128), as `prep` leaves them),
//! v = bf16(silu(.)), beta = bf16(sigmoid(.)), the per-channel decay g = exp(-5 sigmoid(z)) with per-channel offsets that span slow
//! (~0.999) to fast (~e^-4.9) channels, a non-zero initial state. Run under the GPU lock:
//!
//!     flock -x /tmp/rexl3-gpu.lock cargo test --release -p gb-cuda --test glm_p2_kda -- --nocapture --test-threads=1
#![cfg(not(gb_no_cuda))]
#![allow(clippy::needless_range_loop)]

mod common;

use common::Rng;
use gb_cuda::kda::*;
use gb_cuda::num::{bf16_to_f32, f32_to_bf16};
use gb_cuda::{device_available, DevBuf, Stream};
use std::time::Instant;

const H: usize = GLM_HEADS;
const D: usize = HEAD_DIM;

macro_rules! need_gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

#[derive(Clone)]
struct Inp {
    rows: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<u16>,
    g: Vec<f32>,
    beta: Vec<f32>,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn normal(rng: &mut Rng) -> f32 {
    // the sum of 4 uniforms: close enough to a normal for test inputs
    (0..4).map(|_| rng.uniform(-1.0, 1.0)).sum::<f32>() * 0.866
}

fn unit(rng: &mut Rng, scale: f32) -> Vec<f32> {
    let x: Vec<f32> = (0..D).map(|_| normal(rng)).collect();
    let n = x.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>().sqrt();
    x.iter().map(|&a| ((a as f64 / n) * scale as f64) as f32).collect()
}

/// `rows` rows of every head; `offsets` [heads * 128] the per-channel decay offsets (the same for every row of a test).
fn inputs(rng: &mut Rng, rows: usize, offsets: &[f32]) -> Inp {
    let (mut q, mut k, mut v, mut g, mut beta) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let qs = 1.0 / (D as f32).sqrt();
    for _ in 0..rows {
        for _ in 0..H {
            q.extend(unit(rng, qs));
            k.extend(unit(rng, 1.0));
            for _ in 0..D {
                let a = rng.uniform(-3.0, 3.0) as f64;
                v.push(f32_to_bf16((a * sigmoid(a)) as f32));
            }
            beta.push(bf16_to_f32(f32_to_bf16(sigmoid(rng.uniform(-3.0, 3.0) as f64) as f32)));
        }
        for c in 0..H * D {
            let z = offsets[c] as f64 + normal(rng) as f64;
            g.push((-5.0 * sigmoid(z)).exp() as f32);
        }
    }
    Inp { rows, q, k, v, g, beta }
}

fn offsets(rng: &mut Rng) -> Vec<f32> {
    (0..H * D).map(|_| rng.uniform(-8.0, 4.0)).collect()
}

fn slice(inp: &Inp, from: usize, to: usize) -> Inp {
    let e = H * D;
    Inp {
        rows: to - from,
        q: inp.q[from * e..to * e].to_vec(),
        k: inp.k[from * e..to * e].to_vec(),
        v: inp.v[from * e..to * e].to_vec(),
        g: inp.g[from * e..to * e].to_vec(),
        beta: inp.beta[from * H..to * H].to_vec(),
    }
}

struct Dev {
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    g: DevBuf,
    beta: DevBuf,
    y: DevBuf,
}

fn upload(inp: &Inp) -> Dev {
    Dev {
        q: DevBuf::from_slice(&inp.q).unwrap(),
        k: DevBuf::from_slice(&inp.k).unwrap(),
        v: DevBuf::from_slice(&inp.v).unwrap(),
        g: DevBuf::from_slice(&inp.g).unwrap(),
        beta: DevBuf::from_slice(&inp.beta).unwrap(),
        y: DevBuf::zeroed(inp.rows.max(1) * H * D * 2).unwrap(),
    }
}

fn initial_state(rng: &mut Rng) -> Vec<f32> {
    (0..H * D * D).map(|_| rng.uniform(-0.05, 0.05)).collect()
}

const SLOT: usize = 1;

fn pool_with(state: &[f32]) -> Pool {
    let mut pool = Pool::new(Dims::glm(), 2).unwrap();
    pool.write_slot(SLOT, state, &vec![0u16; Dims::glm().conv_slot_elems()]).unwrap();
    pool
}

/// The buffer a state kernel wrote (S[1 - cur]; the parity is never flipped in these tests).
fn written(pool: &Pool) -> Vec<f32> {
    let cur = (pool.read_parity().unwrap()[SLOT] & 1) as usize;
    let both = pool.read_both(SLOT).unwrap();
    let e = H * D * D;
    both[(1 - cur) * e..(2 - cur) * e].to_vec()
}

/// Flip the slot's parity on the host (what the commit tail does after a piece): the state just written becomes the one read next.
fn flip(pool: &mut Pool) {
    let mut p = pool.read_parity().unwrap();
    p[SLOT] ^= 1;
    pool.parity.upload(&p).unwrap();
}

struct Out {
    y: Vec<u16>,
    state: Vec<f32>,
    ms: f64,
}

/// The chunked kernel over `calls` consecutive row counts of `inp` (a parity flip between calls), from `state`.
fn p2(st: &Stream, inp: &Inp, state: &[f32], calls: &[usize], vs: usize) -> Out {
    let mut pool = pool_with(state);
    let mut y = Vec::new();
    let mut at = 0;
    let mut ms = 0.0;
    for (i, &n) in calls.iter().enumerate() {
        let part = slice(inp, at, at + n);
        let mut d = upload(&part);
        st.sync().unwrap();
        let t = Instant::now();
        chunk_raw(st, H, &d.q, &d.k, &d.v, &d.g, &d.beta, &mut pool, SLOT, n, vs, &mut d.y).unwrap();
        st.sync().unwrap();
        ms += t.elapsed().as_secs_f64() * 1e3;
        y.extend(d.y.to_vec::<u16>().unwrap()[..n * H * D].to_vec());
        if i + 1 < calls.len() {
            flip(&mut pool);
        }
        at += n;
    }
    assert_eq!(at, inp.rows);
    Out { y, state: written(&pool), ms }
}

/// P1's row-by-row state kernel over all of `inp` in one window, from `state`.
fn p1(st: &Stream, inp: &Inp, state: &[f32]) -> Out {
    let mut pool = pool_with(state);
    let mut d = upload(inp);
    let ss = DevBuf::from_slice(&[0i32, inp.rows as i32]).unwrap();
    let sl = DevBuf::from_slice(&[SLOT as i32]).unwrap();
    st.sync().unwrap();
    let t = Instant::now();
    p1_state_raw(st, H, &Params::glm(), &d.q, &d.k, &d.v, &d.g, &d.beta, &mut pool, &ss, &sl, &mut d.y).unwrap();
    st.sync().unwrap();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    Out { y: d.y.to_vec::<u16>().unwrap()[..inp.rows * H * D].to_vec(), state: written(&pool), ms }
}

/// The float64 sequential recurrence of heads `heads` (P1's `update()` exactly, in f64): y (before rounding) and the end state.
fn reference(inp: &Inp, state: &[f32], heads: &[usize]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut ys = Vec::new();
    let mut ss = Vec::new();
    for &h in heads {
        let mut s: Vec<f64> = state[h * D * D..(h + 1) * D * D].iter().map(|&x| x as f64).collect();
        let mut y = Vec::with_capacity(inp.rows * D);
        for r in 0..inp.rows {
            let o = (r * H + h) * D;
            let (q, k, g) = (&inp.q[o..o + D], &inp.k[o..o + D], &inp.g[o..o + D]);
            let beta = inp.beta[r * H + h] as f64;
            for j in 0..D {
                let row = &mut s[j * D..(j + 1) * D];
                let mut mem = 0.0;
                for i in 0..D {
                    row[i] *= g[i] as f64;
                    mem += row[i] * k[i] as f64;
                }
                let delta = (bf16_to_f32(inp.v[o + j]) as f64 - mem) * beta;
                let mut out = 0.0;
                for i in 0..D {
                    row[i] += k[i] as f64 * delta;
                    out += row[i] * q[i] as f64;
                }
                y.push(out);
            }
        }
        ys.push(y);
        ss.push(s);
    }
    (ys, ss)
}

/// (relative RMS, max abs) of `got` against `want` over heads `heads` (`got` laid out [rows][H][128] for y, [H][128][128] for a state).
fn err_y(got: &[u16], want: &[Vec<f64>], heads: &[usize], rows: usize) -> (f64, f64) {
    let (mut num, mut den, mut mx) = (0.0f64, 0.0f64, 0.0f64);
    for (hi, &h) in heads.iter().enumerate() {
        for r in 0..rows {
            for j in 0..D {
                let a = bf16_to_f32(got[(r * H + h) * D + j]) as f64;
                let b = want[hi][r * D + j];
                num += (a - b) * (a - b);
                den += b * b;
                mx = mx.max((a - b).abs());
            }
        }
    }
    ((num / den).sqrt(), mx)
}

fn err_s(got: &[f32], want: &[Vec<f64>], heads: &[usize]) -> (f64, f64) {
    let (mut num, mut den, mut mx) = (0.0f64, 0.0f64, 0.0f64);
    for (hi, &h) in heads.iter().enumerate() {
        for e in 0..D * D {
            let a = got[h * D * D + e] as f64;
            let b = want[hi][e];
            num += (a - b) * (a - b);
            den += b * b;
            mx = mx.max((a - b).abs());
        }
    }
    ((num / den).sqrt(), mx)
}

fn same_bits_f32(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn the_chunked_recurrence_is_as_close_to_f64_as_p1s_row_by_row_step() {
    need_gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(0x5eed_0e01);
    let off = offsets(&mut rng);
    let state = initial_state(&mut rng);
    let heads = [0usize, 7, 21, 42, 63];
    for rows in [16usize, 37, 1024] {
        let inp = inputs(&mut rng, rows, &off);
        let (yr, sr) = reference(&inp, &state, &heads);
        let a = p2(&st, &inp, &state, &[rows], 128);
        let b = p1(&st, &inp, &state);
        let (ey2, my2) = err_y(&a.y, &yr, &heads, rows);
        let (ey1, my1) = err_y(&b.y, &yr, &heads, rows);
        let (es2, ms2) = err_s(&a.state, &sr, &heads);
        let (es1, ms1) = err_s(&b.state, &sr, &heads);
        eprintln!(
            "{rows:5} rows vs f64: y rel RMS P2 {ey2:.3e} (max {my2:.2e}) P1 {ey1:.3e} (max {my1:.2e}); end state rel RMS P2 {es2:.3e} \
             (max {ms2:.2e}) P1 {es1:.3e} (max {ms1:.2e})"
        );
        // y is bf16 in both (the rounding dominates: ~1e-3); the state is fp32: the chunked form may not be visibly worse than P1's
        assert!(ey2 < 4e-3 && ey2 < 1.5 * ey1 + 1e-4, "y: P2 {ey2:.3e} vs P1 {ey1:.3e}");
        assert!(es2 < 1e-5 && es2 < 8.0 * es1 + 1e-7, "state: P2 {es2:.3e} vs P1 {es1:.3e}");
    }
}

#[test]
fn the_error_does_not_grow_over_32k_rows() {
    need_gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(0x5eed_0e02);
    let off = offsets(&mut rng);
    let state = initial_state(&mut rng);
    let heads = [3usize, 50];
    let rows = 32768;
    let inp = inputs(&mut rng, rows, &off);
    let mut prev = None;
    for n in [4096usize, 32768] {
        let part = slice(&inp, 0, n);
        let (_, sr) = reference(&part, &state, &heads);
        let calls: Vec<usize> = vec![512; n / 512];
        let a = p2(&st, &part, &state, &calls, 128);
        let b = p1(&st, &part, &state);
        let (es2, _) = err_s(&a.state, &sr, &heads);
        let (es1, _) = err_s(&b.state, &sr, &heads);
        eprintln!("{n:6} rows (512-row calls): end state rel RMS vs f64 P2 {es2:.3e} P1 {es1:.3e}");
        assert!(es2 < 1e-5 && es2 < 8.0 * es1 + 1e-7, "state at {n}: P2 {es2:.3e} vs P1 {es1:.3e}");
        if let Some(p) = prev {
            assert!(es2 < 4.0 * p + 1e-7, "the error grew from {p:.3e} at 4,096 rows to {es2:.3e} at {n}");
        }
        prev = Some(es2);
    }
}

#[test]
fn any_split_at_16_rows_both_block_shapes_and_two_runs_give_the_same_bits() {
    need_gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(0x5eed_0e03);
    let off = offsets(&mut rng);
    let state = initial_state(&mut rng);
    let rows = 1024;
    let inp = inputs(&mut rng, rows, &off);
    let one = p2(&st, &inp, &state, &[rows], 128);
    for (calls, vs) in [
        (vec![rows], 128usize),
        (vec![rows], 64),
        (vec![512, 512], 128),
        (vec![160, 864], 64),
        (vec![16; 64], 128),
        (vec![48, 16, 448, 512], 64),
    ] {
        let o = p2(&st, &inp, &state, &calls, vs);
        assert!(
            o.y == one.y,
            "y differs for calls {calls:?} vs {vs}: {} of {} values",
            o.y.iter().zip(&one.y).filter(|(a, b)| a != b).count(),
            o.y.len()
        );
        assert!(same_bits_f32(&o.state, &one.state), "the end state differs for calls {calls:?} vs {vs}");
    }
    // a partial chunk: the rows of a 1,000-row call (and of 1,000 = 992 + 8) are those of the 1,024-row call
    for calls in [vec![1000usize], vec![992, 8], vec![512, 488]] {
        let part = slice(&inp, 0, 1000);
        let o = p2(&st, &part, &state, &calls, 128);
        assert!(o.y[..] == one.y[..1000 * H * D], "y of a partial last chunk differs for calls {calls:?}");
    }
    // the state after a partial chunk is that chunk's: deterministic, and equal whatever the 16-aligned grid before it
    let a = p2(&st, &slice(&inp, 0, 1000), &state, &[1000], 128);
    let b = p2(&st, &slice(&inp, 0, 1000), &state, &[512, 480, 8], 64);
    assert!(same_bits_f32(&a.state, &b.state), "the state after a partial chunk depends on the grid");
    eprintln!("bit rules hold: 1,024 rows in 6 grids and 2 block shapes, partial chunks, the state after a partial chunk");
}

#[test]
fn bench_chunked_against_p1() {
    need_gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(0x5eed_0e04);
    let off = offsets(&mut rng);
    let state = initial_state(&mut rng);
    let rows: usize = std::env::var("REXL3_P2KDA_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    let inp = inputs(&mut rng, rows, &off);
    let (mut t1, mut t2a, mut t2b) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..5 {
        t1.push(p1(&st, &inp, &state).ms);
        t2a.push(p2(&st, &inp, &state, &[rows], 128).ms);
        t2b.push(p2(&st, &inp, &state, &[rows], 64).ms);
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (a, b, c) = (med(&mut t1), med(&mut t2a), med(&mut t2b));
    let per = |t: f64| t * 1e3 / rows as f64;
    eprintln!(
        "{rows} rows, one layer (64 heads), median of 5 (host-timed, synced): P1 state kernel {a:.3} ms ({:.2} us/row), P2 chunked VS=128 \
         {b:.3} ms ({:.2} us/row, x{:.2}), VS=64 {c:.3} ms ({:.2} us/row, x{:.2})",
        per(a),
        per(b),
        a / b,
        per(c),
        a / c
    );
}

/// The kernel's golden hashes (`REXL3_GOLDEN_OUT`, `scripts/gpu_gate.sh`'s GLM line; `testdata/gpu/golden_glm_p2_kda.txt`): y and the end
/// state of the chunked recurrence on fixed synthetic inputs, a whole number of chunks and a partial one, both block shapes (one line each:
/// the shapes must agree), from a non-zero state.
#[test]
#[ignore]
fn golden_glm_p2_kda_hashes() {
    need_gpu!();
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    let st = Stream::new().unwrap();
    let h16 = |v: &[u16]| gb_exl3::fnv1a(v.iter().flat_map(|x| x.to_le_bytes()));
    let h32 = |v: &[f32]| gb_exl3::fnv1a(v.iter().flat_map(|x| x.to_bits().to_le_bytes()));
    let mut lines = vec![format!("arith {} chunk {}", P2_ARITH_ID, P2_CHUNK_ROWS)];
    let mut rng = Rng(0x601d_0e05);
    let off = offsets(&mut rng);
    let state = initial_state(&mut rng);
    for rows in [16usize, 37, 512] {
        let inp = inputs(&mut rng, rows, &off);
        let a = p2(&st, &inp, &state, &[rows], 128);
        let b = p2(&st, &inp, &state, &[rows], 64);
        assert!(a.y == b.y && same_bits_f32(&a.state, &b.state), "the block shapes disagree at {rows} rows");
        lines.push(format!("p2kda rows={rows} y {:016x} state {:016x}", h16(&a.y), h32(&a.state)));
    }
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    eprintln!("wrote {} golden lines to {path}", lines.len());
}
