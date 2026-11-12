//! The NextN (MTP) layer's glue kernels (`csrc/glm_mtp.cu`) against host mirrors, bit for bit, and their row invariance. GPU tests
//! print "not run" when there is no device.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::needless_range_loop)]
mod common;

use common::*;
use gb_cuda::{device_available, glm_mtp, DevBuf, Stream};
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
/// bf16 bits to f32 (the high half of the f32)
fn bf(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
/// f32 to the nearest bf16 bits (ties to even)
fn to_bf(x: f32) -> u16 {
    let u = x.to_bits();
    let r = u.wrapping_add(0x7FFF + ((u >> 16) & 1));
    (r >> 16) as u16
}

/// The sum of squares in the kernels' one order: thread t owns the 8-wide vectors t, t + 256, ..., adds their squares in element order;
/// the 32 lanes of a warp are summed by the xor butterfly; the 8 warps in order.
fn sumsq(x: &[f32]) -> f32 {
    let dim = x.len();
    let nvec = dim / 8;
    let mut thread = vec![0f32; 256];
    for (t, acc) in thread.iter_mut().enumerate() {
        let mut v = t;
        while v < nvec {
            for i in 0..8 {
                let e = x[v * 8 + i];
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
    ss
}

fn norm_host(x: &[f32], w: &[f32], eps: f32) -> Vec<u16> {
    let dim = x.len();
    let inv = 1.0f32 / (sumsq(x) / dim as f32 + eps).sqrt();
    (0..dim).map(|i| h((x[i] * inv) * w[i])).collect()
}

fn rand_bf16(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n).map(|_| to_bf(rng.uniform(-1.0, 1.0) * 0.05)).collect()
}

fn rand_f16(rng: &mut Rng, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| {
            let s = rng.uniform(0.0, 1.0);
            h(rng.uniform(-1.0, 1.0) * (0.02 + 20.0 * s * s * s))
        })
        .collect()
}

/// One row of `mtp_input` on the host.
fn input_host(table: &[u16], id: i32, vocab: usize, hid: &[u16], we: &[f32], wh: &[f32], eps: f32) -> Vec<u16> {
    let dim = hid.len();
    let e: Vec<f32> = if id >= 0 && (id as usize) < vocab {
        // the bf16 embedding row rounded to fp16, as the fork widens it
        table[id as usize * dim..(id as usize + 1) * dim].iter().map(|&b| f(h(bf(b)))).collect()
    } else {
        vec![0.0; dim]
    };
    let g: Vec<f32> = hid.iter().map(|&b| f(b)).collect();
    let mut out = norm_host(&e, we, eps);
    out.extend(norm_host(&g, wh, eps));
    out
}

#[test]
fn mtp_input_equals_the_host_mirror_bit_for_bit_and_is_row_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(11);
    let (dim, vocab, rows, eps) = (4096usize, 64usize, 29usize, 1e-5f32);
    let table = rand_bf16(&mut rng, vocab * dim);
    let hid = rand_f16(&mut rng, rows * dim);
    let (we, wh) = (rng.vec(dim, 0.05, 2.0), rng.vec(dim, 0.05, 2.0));
    // ids include one past the vocabulary and one negative: both give a zero embedding row (and a zero normed half)
    let ids: Vec<i32> = (0..rows).map(|r| ((r * 7 + 3) % vocab) as i32).collect::<Vec<_>>();
    let mut ids = ids;
    ids[5] = vocab as i32;
    ids[6] = -1;
    let (td, idd, hd, wed, whd) = (
        DevBuf::from_slice(&table).unwrap(),
        DevBuf::from_slice(&ids).unwrap(),
        DevBuf::from_slice(&hid).unwrap(),
        DevBuf::from_slice(&we).unwrap(),
        DevBuf::from_slice(&wh).unwrap(),
    );
    let mut out = DevBuf::zeroed(rows * 2 * dim * 2).unwrap();
    glm_mtp::mtp_input(&td, &idd, &hd, &wed, &whd, &mut out, rows, dim, vocab, eps, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<u16> = out.to_vec().unwrap();
    for r in 0..rows {
        let want = input_host(&table, ids[r], vocab, &hid[r * dim..(r + 1) * dim], &we, &wh, eps);
        assert_eq!(got[r * 2 * dim..(r + 1) * 2 * dim], want[..], "row {r} (id {})", ids[r]);
    }
    assert!(got[5 * 2 * dim..5 * 2 * dim + dim].iter().all(|&b| b == 0), "an id past the vocabulary is a zero embedding row");
    // a prefix of the rows gives the same bits whatever the row count of the call
    for n in [1usize, 2, 3, 9, 16] {
        let mut o = DevBuf::zeroed(n * 2 * dim * 2).unwrap();
        glm_mtp::mtp_input(&td, &idd, &hd, &wed, &whd, &mut o, n, dim, vocab, eps, &st).unwrap();
        st.sync().unwrap();
        let v: Vec<u16> = o.to_vec().unwrap();
        assert_eq!(v[..], got[..n * 2 * dim], "{n} rows");
    }
}

#[test]
fn an_image_row_normalises_like_a_text_row_of_the_same_values_and_leaves_the_rest_of_the_row_alone() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(23);
    let (dim, vocab, rows, eps) = (4096usize, 32usize, 11usize, 1e-5f32);
    let we = rng.vec(dim, 0.05, 2.0);
    let wh = rng.vec(dim, 0.05, 2.0);
    let hid = rand_f16(&mut rng, rows * dim);
    // a feature table (fp32, the vision tower's output): rows 0..4 sit at cat rows 3..7 (a splice in the middle of a chunk)
    let nfeat = 6;
    let feat: Vec<f32> = rng.vec(nfeat * dim, -2.5, 2.5);
    let table = rand_bf16(&mut rng, vocab * dim);
    let ids: Vec<i32> = (0..rows).map(|r| (r % vocab) as i32).collect();
    let (td, idd, hd, wed, whd, fd) = (
        DevBuf::from_slice(&table).unwrap(),
        DevBuf::from_slice(&ids).unwrap(),
        DevBuf::from_slice(&hid).unwrap(),
        DevBuf::from_slice(&we).unwrap(),
        DevBuf::from_slice(&wh).unwrap(),
        DevBuf::from_slice(&feat).unwrap(),
    );
    let mut cat = DevBuf::zeroed(rows * 2 * dim * 2).unwrap();
    glm_mtp::mtp_input(&td, &idd, &hd, &wed, &whd, &mut cat, rows, dim, vocab, eps, &st).unwrap();
    st.sync().unwrap();
    let before: Vec<u16> = cat.to_vec().unwrap();
    // feature rows 1..5 over cat rows 3..7
    glm_mtp::mtp_embed_f32(&fd, 1, &wed, &mut cat, 3, 4, dim, eps, &st).unwrap();
    st.sync().unwrap();
    let after: Vec<u16> = cat.to_vec().unwrap();
    for r in 0..rows {
        let (a, b) = (&after[r * 2 * dim..(r + 1) * 2 * dim], &before[r * 2 * dim..(r + 1) * 2 * dim]);
        if (3..7).contains(&r) {
            // the embedding half: the fp32 feature rounded to fp16, normed in the kernels' order; the hidden half untouched
            let e: Vec<f32> = feat[(r - 3 + 1) * dim..(r - 3 + 2) * dim].iter().map(|&x| f(h(x))).collect();
            assert_eq!(a[..dim], norm_host(&e, &we, eps)[..], "row {r}: the embedding half");
            assert_eq!(a[dim..], b[dim..], "row {r}: the hidden half is untouched");
        } else {
            assert_eq!(a, b, "row {r} is not in the splice and must not change");
        }
    }
    // the same values as a table row give the same bits as `mtp_input` makes of that table row (fp16 values: no rounding left to differ)
    let id = 5usize;
    let as_f32: Vec<f32> = table[id * dim..(id + 1) * dim].iter().map(|&b| f(h(bf(b)))).collect();
    let fd2 = DevBuf::from_slice(&as_f32).unwrap();
    let mut one = DevBuf::zeroed(2 * dim * 2).unwrap();
    glm_mtp::mtp_embed_f32(&fd2, 0, &wed, &mut one, 0, 1, dim, eps, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<u16> = one.to_vec().unwrap();
    let want = input_host(&table, id as i32, vocab, &hid[..dim], &we, &wh, eps);
    assert_eq!(got[..dim], want[..dim], "an image row holding a token's embedding normalises exactly like the token's row");
    // a prefix of the splice gives the same bits whatever the row count of the call
    for n in [1usize, 2, 3] {
        let mut c2 = DevBuf::zeroed(rows * 2 * dim * 2).unwrap();
        glm_mtp::mtp_embed_f32(&fd, 1, &wed, &mut c2, 3, n, dim, eps, &st).unwrap();
        st.sync().unwrap();
        let v: Vec<u16> = c2.to_vec().unwrap();
        for r in 0..n {
            assert_eq!(
                v[(3 + r) * 2 * dim..(3 + r) * 2 * dim + dim],
                after[(3 + r) * 2 * dim..(3 + r) * 2 * dim + dim],
                "{n} rows, row {r}"
            );
        }
    }
    // bad extents are refused
    assert!(glm_mtp::mtp_embed_f32(&fd, 4, &wed, &mut cat, 3, 4, dim, eps, &st).is_err(), "rows past the feature table");
    assert!(glm_mtp::mtp_embed_f32(&fd, 0, &wed, &mut cat, rows - 1, 4, dim, eps, &st).is_err(), "rows past the cat buffer");
}

#[test]
fn the_fp32_norms_and_adds_equal_the_host_mirror_bit_for_bit_and_are_row_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(5);
    let eps = 1e-5f32;
    for dim in [4096usize, 512, 8] {
        let rows = 23;
        let x: Vec<f32> = rng.vec(rows * dim, -3.0, 3.0);
        let y: Vec<f32> = rng.vec(rows * dim, -2.0, 2.0);
        let w = rng.vec(dim, 0.05, 2.0);
        let (xd, yd, wd) = (DevBuf::from_slice(&x).unwrap(), DevBuf::from_slice(&y).unwrap(), DevBuf::from_slice(&w).unwrap());

        // y = rmsnorm(x)
        let mut o = DevBuf::zeroed(rows * dim * 2).unwrap();
        glm_mtp::rmsnorm_f32_f16(&xd, &wd, &mut o, rows, dim, eps, &st).unwrap();
        st.sync().unwrap();
        let got: Vec<u16> = o.to_vec().unwrap();
        for r in 0..rows {
            assert_eq!(got[r * dim..(r + 1) * dim], norm_host(&x[r * dim..(r + 1) * dim], &w, eps), "norm dim {dim} row {r}");
        }
        for n in [1usize, 2, 9] {
            let mut on = DevBuf::zeroed(n * dim * 2).unwrap();
            glm_mtp::rmsnorm_f32_f16(&xd, &wd, &mut on, n, dim, eps, &st).unwrap();
            st.sync().unwrap();
            let v: Vec<u16> = on.to_vec().unwrap();
            assert_eq!(v[..], got[..n * dim], "norm dim {dim}: {n} rows");
        }

        // x += y then the norm of the sum
        let mut xs = DevBuf::from_slice(&x).unwrap();
        let mut o2 = DevBuf::zeroed(rows * dim * 2).unwrap();
        glm_mtp::add_rmsnorm_f32(&mut xs, &yd, &wd, &mut o2, rows, dim, eps, &st).unwrap();
        st.sync().unwrap();
        let (sum_got, norm_got): (Vec<f32>, Vec<u16>) = (xs.to_vec().unwrap(), o2.to_vec().unwrap());
        for r in 0..rows {
            let want: Vec<f32> = (0..dim).map(|i| x[r * dim + i] + y[r * dim + i]).collect();
            assert_eq!(sum_got[r * dim..(r + 1) * dim], want[..], "sum dim {dim} row {r}");
            assert_eq!(norm_got[r * dim..(r + 1) * dim], norm_host(&want, &w, eps), "add+norm dim {dim} row {r}");
        }

        // x += y alone
        let mut xa = DevBuf::from_slice(&x).unwrap();
        glm_mtp::add_f32(&mut xa, &yd, rows * dim, &st).unwrap();
        st.sync().unwrap();
        let added: Vec<f32> = xa.to_vec().unwrap();
        assert_eq!(added, sum_got, "dim {dim}: the plain add is the sum the fused kernel stores");
    }
}

#[test]
fn the_mtp_glue_refuses_bad_shapes_and_zero_rows_is_a_no_op() {
    gpu!();
    let st = Stream::new().unwrap();
    let x = DevBuf::zeroed(4096 * 4).unwrap();
    let w = DevBuf::zeroed(4096).unwrap();
    let mut y = DevBuf::zeroed(4096).unwrap();
    assert!(glm_mtp::rmsnorm_f32_f16(&x, &w, &mut y, 1, 12, 1e-5, &st).is_err(), "dim not a multiple of 8");
    assert!(glm_mtp::rmsnorm_f32_f16(&x, &w, &mut y, 4, 1024, 1e-5, &st).is_err(), "output buffer too small");
    assert!(glm_mtp::rmsnorm_f32_f16(&x, &w, &mut y, 0, 64, 1e-5, &st).is_ok());
    let mut xm = DevBuf::zeroed(4096 * 4).unwrap();
    assert!(glm_mtp::add_f32(&mut xm, &x, 3, &st).is_err(), "n % 4");
    assert!(glm_mtp::add_f32(&mut xm, &x, 1 << 20, &st).is_err(), "buffers too small");
}
