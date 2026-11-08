//! The GLM glue kernels (`csrc/glm_glue.cu`): the fp16 RMSNorm against a host mirror (bit for bit) and the top-k with the
//! valid-vocabulary mask against a full sort. GPU tests print "not run" when there is no device.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::needless_range_loop)]
mod common;

use common::*;
use gb_cuda::{device_available, glm_glue, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: not run");
            return;
        }
    };
}

fn h(x: f32) -> u16 {
    f64_to_f16(x as f64)
}
fn f(b: u16) -> f32 {
    f16_to_f64(b) as f32
}

/// The kernel's arithmetic, one row: thread t owns the 8-wide vectors t, t + 256, ... and adds their squares in element order, the
/// 32 lanes of each warp are summed by the xor butterfly, the 8 warps in order; then `(x * inv) * w` rounded once to fp16.
fn rms_host(x: &[u16], w: &[f32], eps: f32) -> Vec<u16> {
    let dim = x.len();
    let nvec = dim / 8;
    let mut thread = vec![0f32; 256];
    for (t, acc) in thread.iter_mut().enumerate() {
        let mut v = t;
        while v < nvec {
            for i in 0..8 {
                let e = f(x[v * 8 + i]);
                *acc += e * e;
            }
            v += 256;
        }
    }
    let mut warp = [0f32; 8];
    for (wi, out) in warp.iter_mut().enumerate() {
        let mut a = [0f32; 32];
        a.copy_from_slice(&thread[wi * 32..wi * 32 + 32]);
        for m in [16usize, 8, 4, 2, 1] {
            let b = a;
            for l in 0..32 {
                a[l] = b[l] + b[l ^ m];
            }
        }
        *out = a[0];
    }
    let mut ss = warp[0];
    for &s in &warp[1..] {
        ss += s;
    }
    let inv = 1.0f32 / (ss / dim as f32 + eps).sqrt();
    (0..dim).map(|i| h((f(x[i]) * inv) * w[i])).collect()
}

fn rand_f16(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| {
            let s = rng.uniform(0.0, 1.0);
            h(rng.uniform(-1.0, 1.0) * (0.02 + 20.0 * s * s * s))
        })
        .collect()
}

#[test]
fn rmsnorm_f16_equals_the_host_mirror_bit_for_bit_and_is_row_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(7);
    let eps = 1e-5f32;
    for dim in [512usize, 1536, 4096, 8] {
        let rows = 37;
        let x = rand_f16(&mut rng, rows * dim);
        let w = rng.vec(dim, 0.05, 2.0);
        let (xd, wd) = (DevBuf::from_slice(&x).unwrap(), DevBuf::from_slice(&w).unwrap());
        let mut yd = DevBuf::zeroed(rows * dim * 2).unwrap();
        glm_glue::rmsnorm_f16(&xd, &wd, &mut yd, rows, dim, eps, &st).unwrap();
        st.sync().unwrap();
        let y: Vec<u16> = yd.to_vec().unwrap();
        for r in 0..rows {
            assert_eq!(y[r * dim..(r + 1) * dim], rms_host(&x[r * dim..(r + 1) * dim], &w, eps), "dim {dim} row {r}");
        }
        // a row alone gives the same bits as inside the batch, whatever the row count of the call
        for n in [1usize, 2, 3, 9, 16] {
            let mut yn = DevBuf::zeroed(n * dim * 2).unwrap();
            glm_glue::rmsnorm_f16(&xd, &wd, &mut yn, n, dim, eps, &st).unwrap();
            st.sync().unwrap();
            let got: Vec<u16> = yn.to_vec().unwrap();
            assert_eq!(got[..], y[..n * dim], "dim {dim}: {n} rows");
        }
    }
}

#[test]
fn rmsnorm_f16_refuses_bad_shapes_and_zero_rows_is_a_no_op() {
    gpu!();
    let st = Stream::new().unwrap();
    let x = DevBuf::zeroed(4096).unwrap();
    let w = DevBuf::zeroed(4096).unwrap();
    let mut y = DevBuf::zeroed(4096).unwrap();
    assert!(glm_glue::rmsnorm_f16(&x, &w, &mut y, 1, 12, 1e-5, &st).is_err(), "dim not a multiple of 8");
    assert!(glm_glue::rmsnorm_f16(&x, &w, &mut y, 4, 2048, 1e-5, &st).is_err(), "buffers too small");
    assert!(glm_glue::rmsnorm_f16(&x, &w, &mut y, 0, 64, 1e-5, &st).is_ok());
}

/// Host reference: the first `count` columns of `0..valid` by (value desc, id asc); NaN is -inf and -0 ties with +0.
fn topk_host(row: &[u16], valid: usize, count: usize) -> (Vec<i32>, Vec<f32>) {
    let key = |b: u16| -> f32 {
        let v = f(b);
        if v.is_nan() {
            f32::NEG_INFINITY
        } else if v == 0.0 {
            0.0
        } else {
            v
        }
    };
    let mut idx: Vec<usize> = (0..valid).collect();
    idx.sort_by(|&a, &b| key(row[b]).partial_cmp(&key(row[a])).unwrap().then(a.cmp(&b)));
    idx.truncate(count);
    (idx.iter().map(|&i| i as i32).collect(), idx.iter().map(|&i| key(row[i])).collect())
}

fn run_topk(logits: &[u16], rows: usize, stride: usize, valid: usize, count: usize) -> (Vec<i32>, Vec<f32>) {
    let st = Stream::new().unwrap();
    let ld = DevBuf::from_slice(logits).unwrap();
    let mut ids = DevBuf::zeroed(rows * count * 4).unwrap();
    let mut vals = DevBuf::zeroed(rows * count * 4).unwrap();
    let mut ws = DevBuf::zeroed(glm_glue::topk_workspace_bytes(rows, valid, count)).unwrap();
    glm_glue::topk_f16(&ld, rows, stride, valid, count, &mut ids, &mut vals, &mut ws, &st).unwrap();
    st.sync().unwrap();
    (ids.to_vec().unwrap(), vals.to_vec().unwrap())
}

#[test]
fn topk_f16_matches_a_full_sort_and_never_reads_past_the_valid_columns() {
    gpu!();
    let mut rng = Rng(11);
    // (rows, stride, valid): small and awkward shapes, then the model's: 154,880 columns of which 154,856 are tokens
    for (rows, stride, valid) in [(1usize, 64usize, 64usize), (3, 1000, 987), (5, 20_000, 19_999), (2, 154_880, 154_856)] {
        for count in [1usize, 2, 8, 16, 17, 33, 64] {
            let mut logits = rand_f16(&mut rng, rows * stride);
            for r in 0..rows {
                let row = &mut logits[r * stride..(r + 1) * stride];
                // ties (several equal maxima), a NaN, a -0 against a +0, and huge values in the masked padding
                let top = h(9.5);
                for k in 0..5 {
                    row[(r * 37 + k * 211) % valid] = top;
                }
                row[(r * 11 + 3) % valid] = 0x7E00;
                row[1 % valid] = 0x8000;
                row[2 % valid] = 0;
                for p in valid..stride {
                    row[p] = h(60000.0);
                }
            }
            let (ids, vals) = run_topk(&logits, rows, stride, valid, count);
            for r in 0..rows {
                let (wi, wv) = topk_host(&logits[r * stride..(r + 1) * stride], valid, count);
                assert_eq!(
                    ids[r * count..(r + 1) * count],
                    wi[..],
                    "rows {rows} stride {stride} valid {valid} count {count} row {r}"
                );
                assert_eq!(
                    vals[r * count..(r + 1) * count].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    wv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "values: rows {rows} count {count} row {r}"
                );
            }
        }
    }
}

#[test]
fn topk_f16_is_row_invariant() {
    gpu!();
    let mut rng = Rng(13);
    let (stride, valid, count) = (154_880usize, 154_856usize, 16usize);
    let rows = 9;
    let logits = rand_f16(&mut rng, rows * stride);
    let (ids, vals) = run_topk(&logits, rows, stride, valid, count);
    for r in 0..rows {
        let (i1, v1) = run_topk(&logits[r * stride..(r + 1) * stride], 1, stride, valid, count);
        assert_eq!(ids[r * count..(r + 1) * count], i1[..], "row {r} alone");
        assert_eq!(
            vals[r * count..(r + 1) * count].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            v1.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn topk_f16_refuses_what_it_cannot_do() {
    gpu!();
    let st = Stream::new().unwrap();
    let l = DevBuf::zeroed(2 * 100 * 2).unwrap();
    let mut ids = DevBuf::zeroed(1024).unwrap();
    let mut vals = DevBuf::zeroed(1024).unwrap();
    let mut ws = DevBuf::zeroed(1 << 20).unwrap();
    // count above 64, above valid, valid above stride, buffers too small
    assert!(glm_glue::topk_f16(&l, 2, 100, 100, 65, &mut ids, &mut vals, &mut ws, &st).is_err());
    assert!(glm_glue::topk_f16(&l, 2, 100, 10, 11, &mut ids, &mut vals, &mut ws, &st).is_err());
    assert!(glm_glue::topk_f16(&l, 2, 100, 101, 4, &mut ids, &mut vals, &mut ws, &st).is_err());
    assert!(glm_glue::topk_f16(&l, 3, 100, 100, 4, &mut ids, &mut vals, &mut ws, &st).is_err());
    assert_eq!(glm_glue::topk_workspace_bytes(1, 100, 65), 0);
}
