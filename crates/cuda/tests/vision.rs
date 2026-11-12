//! Vision-tower kernels against straightforward CPU references. GPU tests skip when there is no device.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::Rng;
use gb_cuda::vision as vis;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

fn up<T: Copy>(v: &[T]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

fn h(v: f32) -> u16 {
    f64_to_f16(v as f64)
}

fn hv(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| h(x)).collect()
}

fn bf16_round(v: f32) -> f32 {
    let b = v.to_bits();
    f32::from_bits(b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000)
}

fn fetch_f16(b: &DevBuf, n: usize) -> Vec<f64> {
    let v: Vec<u16> = b.to_vec().unwrap();
    v[..n].iter().map(|&x| f16_to_f64(x)).collect()
}

#[test]
fn layernorm_matches_the_formula_for_f32_and_f16_inputs() {
    gpu!();
    let st = Stream::new().unwrap();
    let (rows, dim) = (9usize, 1152usize);
    let mut rng = Rng(5);
    let x = rng.vec(rows * dim, -3.0, 5.0);
    let w = rng.vec(dim, 0.5, 1.5);
    let b = rng.vec(dim, -0.5, 0.5);
    let (wh, bh) = (hv(&w), hv(&b));
    let xh = hv(&x);
    for f32_in in [true, false] {
        let dx = if f32_in { up(&x) } else { up(&xh) };
        let mut dy = DevBuf::zeroed(rows * dim * 2).unwrap();
        vis::layernorm(&dx, f32_in, &up(&wh), &up(&bh), &mut dy, rows, dim, 1e-6, &st).unwrap();
        st.sync().unwrap();
        let got = fetch_f16(&dy, rows * dim);
        let mut worst = 0f64;
        for r in 0..rows {
            let xr: Vec<f64> =
                (0..dim).map(|c| if f32_in { x[r * dim + c] as f64 } else { f16_to_f64(xh[r * dim + c]) }).collect();
            let mean = xr.iter().sum::<f64>() / dim as f64;
            let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / dim as f64;
            for c in 0..dim {
                let want = (xr[c] - mean) / (var + 1e-6).sqrt() * f16_to_f64(wh[c]) + f16_to_f64(bh[c]);
                worst = worst.max((got[r * dim + c] - want).abs() / want.abs().max(1.0));
            }
        }
        assert!(worst < 2e-3, "f32_in {f32_in}: worst relative error {worst}");
    }
}

#[test]
fn rope2d_rotates_pairs_d_and_d_plus_36() {
    gpu!();
    let st = Stream::new().unwrap();
    let (p, heads) = (11usize, 4usize);
    let mut rng = Rng(7);
    let q = rng.vec(p * heads * 72, -2.0, 2.0);
    let k = rng.vec(p * heads * 72, -2.0, 2.0);
    let ang = rng.vec(p * 36, 0.0, 60.0);
    let mut cs = Vec::with_capacity(p * 72);
    for r in 0..p {
        cs.extend((0..36).map(|d| ang[r * 36 + d].cos()));
        cs.extend((0..36).map(|d| ang[r * 36 + d].sin()));
    }
    let (qh, kh) = (hv(&q), hv(&k));
    let (mut dq, mut dk) = (up(&qh), up(&kh));
    vis::rope2d(&mut dq, &mut dk, &up(&cs), p, heads, &st).unwrap();
    st.sync().unwrap();
    let (gq, gk) = (fetch_f16(&dq, p * heads * 72), fetch_f16(&dk, p * heads * 72));
    let mut worst = 0f64;
    for r in 0..p {
        for hd in 0..heads {
            for d in 0..36 {
                let at = (r * heads + hd) * 72 + d;
                let (c, s) = (cs[r * 72 + d] as f64, cs[r * 72 + 36 + d] as f64);
                for (src, got) in [(&qh, &gq), (&kh, &gk)] {
                    let (a, b) = (f16_to_f64(src[at]), f16_to_f64(src[at + 36]));
                    worst = worst.max((got[at] - (a * c - b * s)).abs()).max((got[at + 36] - (b * c + a * s)).abs());
                }
            }
        }
    }
    assert!(worst < 2e-3, "worst absolute error {worst}");
}

#[test]
fn pos_add_follows_the_bf16_chain() {
    gpu!();
    let st = Stream::new().unwrap();
    let (p, hdim, n) = (13usize, 1152usize, 50usize);
    let mut rng = Rng(9);
    let table: Vec<f32> = rng.vec(n * hdim, -2.0, 2.0).iter().map(|&v| bf16_round(v)).collect();
    let idx: Vec<i32> = (0..p * 4).map(|_| (rng.next_u64() % n as u64) as i32).collect();
    let w: Vec<f32> = rng.vec(p * 4, 0.0, 1.0).iter().map(|&v| bf16_round(v)).collect();
    let x0 = rng.vec(p * hdim, -4.0, 4.0);
    let bias = rng.vec(hdim, -1.0, 1.0);
    let biash = hv(&bias);
    let mut dx = up(&x0);
    vis::pos_add(&mut dx, Some(&up(&biash)), &up(&table), &up(&idx), &up(&w), p, hdim, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<f32> = dx.to_vec().unwrap();
    for r in 0..p {
        for c in 0..hdim {
            let t = |i: usize| bf16_round(table[idx[r * 4 + i] as usize * hdim + c] * w[r * 4 + i]);
            let s = bf16_round(bf16_round(bf16_round(t(0) + t(1)) + t(2)) + t(3));
            let want = x0[r * hdim + c] + f16_to_f64(biash[c]) as f32 + s;
            assert_eq!(got[r * hdim + c], want, "patch {r} col {c}");
        }
    }
}

#[test]
fn elementwise_kernels() {
    gpu!();
    let st = Stream::new().unwrap();
    let n = 10_007usize;
    let mut rng = Rng(11);
    // gelu (tanh form), f16 in place
    let v = rng.vec(n, -6.0, 6.0);
    let vh = hv(&v);
    let mut dv = up(&vh);
    vis::gelu_tanh(&mut dv, n, &st).unwrap();
    // residual add f32 += f16
    let x = rng.vec(n, -3.0, 3.0);
    let mut dx = up(&x);
    vis::add_f16(&mut dx, &up(&vh), n, &st).unwrap();
    // f32 -> bf16
    let mut db = DevBuf::zeroed(n * 2).unwrap();
    vis::f32_to_bf16(&dx, &mut db, n, &st).unwrap();
    st.sync().unwrap();
    let g = fetch_f16(&dv, n);
    for i in 0..n {
        let a = f16_to_f64(vh[i]);
        let want = 0.5 * a * (1.0 + (0.7978845608028654 * (a + 0.044715 * a * a * a)).tanh());
        assert!((g[i] - want).abs() < 1e-3 * want.abs().max(1.0), "gelu {a}: {} vs {want}", g[i]);
    }
    let sum: Vec<f32> = dx.to_vec().unwrap();
    let bits: Vec<u16> = db.to_vec().unwrap();
    for i in 0..n {
        assert_eq!(sum[i], x[i] + f16_to_f64(vh[i]) as f32, "add {i}");
        assert_eq!(bits[i], (bf16_round(sum[i]).to_bits() >> 16) as u16, "bf16 {i}");
    }
}

fn attention_case(p: usize, heads: usize, seed: u64, scale_in: f32) {
    let st = Stream::new().unwrap();
    let mut rng = Rng(seed);
    let n = p * heads * 72;
    let (q, k, v) = (rng.vec(n, -scale_in, scale_in), rng.vec(n, -scale_in, scale_in), rng.vec(n, -1.5, 1.5));
    let (qh, kh, vh) = (hv(&q), hv(&k), hv(&v));
    let mut out = DevBuf::zeroed(n * 2).unwrap();
    vis::attention(&up(&qh), &up(&kh), &up(&vh), &mut out, p, heads, &st).unwrap();
    st.sync().unwrap();
    let got = fetch_f16(&out, n);
    let stride = heads * 72;
    let scale = 1.0 / (72f64).sqrt();
    let (mut worst, mut rms) = (0f64, 0f64);
    // check a spread of (row, head) pairs in f64: every row for small p, a stride of rows for large p
    let step = (p / 97).max(1);
    let mut checked = 0;
    for r in (0..p).step_by(step).chain([p - 1]) {
        for head in 0..heads {
            let qr: Vec<f64> = (0..72).map(|d| f16_to_f64(qh[r * stride + head * 72 + d])).collect();
            let mut sc: Vec<f64> = (0..p)
                .map(|j| (0..72).map(|d| qr[d] * f16_to_f64(kh[j * stride + head * 72 + d])).sum::<f64>() * scale)
                .collect();
            let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            sc.iter_mut().for_each(|s| *s = (*s - m).exp());
            let l: f64 = sc.iter().sum();
            for d in 0..72 {
                let want: f64 = (0..p).map(|j| sc[j] * f16_to_f64(vh[j * stride + head * 72 + d])).sum::<f64>() / l;
                let e = (got[r * stride + head * 72 + d] - want).abs();
                worst = worst.max(e);
                rms += want * want;
            }
            checked += 1;
        }
    }
    let rms = (rms / (checked * 72) as f64).sqrt();
    eprintln!("attention p={p} heads={heads} q-scale {scale_in}: worst abs error {worst:.2e}, output rms {rms:.3}");
    assert!(worst < 4e-3 * rms.max(0.05) + 1e-3, "p={p}: worst error {worst} against output rms {rms}");
}

#[test]
fn attention_matches_a_float64_softmax_attention() {
    gpu!();
    for (p, heads) in [(1, 2), (5, 3), (64, 2), (65, 2), (100, 16), (257, 4), (1000, 16), (4100, 2)] {
        attention_case(p, heads, 100 + p as u64, 1.0);
    }
    // sharper softmax (larger logits), where the running max and rescale matter
    attention_case(777, 4, 5, 3.0);
}
