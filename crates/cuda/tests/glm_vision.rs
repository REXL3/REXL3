//! The GLM-5.3-Flash vision-tower kernels (`csrc/glm_vision.cu`) against straightforward host references, including the inputs on which each
//! deliberate bug of the tower's mutation table changes the output (an eps that only matters for a near-constant row, a clamp that only matters above
//! 10, a GELU that differs from the tanh form by a fraction of a fp16 ulp). GPU tests print "not run" when there is no device.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::Rng;
use gb_cuda::glm_vision as gv;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: not run");
            return;
        }
    };
}

fn up<T: Copy>(v: &[T]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}
fn h(v: f64) -> u16 {
    f64_to_f16(v)
}
fn f(b: u16) -> f64 {
    f16_to_f64(b)
}
/// fp16 round trip
fn r16(v: f64) -> f64 {
    f(h(v))
}
fn bf16(v: f32) -> u16 {
    // round to nearest even
    let b = v.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
fn bf(b: u16) -> f64 {
    f32::from_bits((b as u32) << 16) as f64
}
fn fetch16(b: &DevBuf, n: usize) -> Vec<f64> {
    let v: Vec<u16> = b.to_vec().unwrap();
    v[..n].iter().map(|&x| f(x)).collect()
}
fn fetch32(b: &DevBuf, n: usize) -> Vec<f32> {
    let v: Vec<f32> = b.to_vec().unwrap();
    v[..n].to_vec()
}

/// Largest `|got - want| / max(|want|, floor)`.
fn worst_rel(got: &[f64], want: &[f64], floor: f64) -> f64 {
    got.iter().zip(want).map(|(g, w)| (g - w).abs() / w.abs().max(floor)).fold(0.0, f64::max)
}

fn rel_l2(got: &[f64], want: &[f64]) -> f64 {
    let d: f64 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
    let n: f64 = want.iter().map(|w| w * w).sum();
    (d / n.max(1e-300)).sqrt()
}

// ------------------------------------------------------------------------------------------------------------------ norms

fn rms_host(x: &[f32], w: &[u16], eps: f64) -> Vec<f64> {
    let ss: f64 = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
    let rmf = 1.0 / (ss + eps).sqrt();
    x.iter().zip(w).map(|(&v, &wb)| r16(v as f64 * bf(wb) * rmf)).collect()
}

#[test]
fn rmsnorm_matches_the_formula_and_its_eps_is_the_forks() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 6usize;
    let mut rng = Rng(11);
    let mut x = rng.vec(rows * 1024, -4.0, 4.0);
    // row 3: tiny values (mean square about 1e-6, where eps 1e-5 and 1e-6 give outputs 2.4x apart); row 4: constant; row 5: one huge outlier
    for c in 0..1024 {
        x[3 * 1024 + c] = rng.uniform(-1.7e-3, 1.7e-3);
        x[4 * 1024 + c] = 0.25;
    }
    x[5 * 1024 + 7] = 900.0;
    let w: Vec<u16> = (0..1024).map(|_| bf16(rng.uniform(0.2, 1.8))).collect();
    for eps in [1e-5f32, 1e-6] {
        let mut dx = up(&x);
        let mut dy = DevBuf::zeroed(rows * 1024 * 2).unwrap();
        gv::rmsnorm(&mut dx, None, None, &up(&w), &mut dy, rows, eps, &st).unwrap();
        st.sync().unwrap();
        let got = fetch16(&dy, rows * 1024);
        for r in 0..rows {
            let want = rms_host(&x[r * 1024..(r + 1) * 1024], &w, eps as f64);
            let wr = worst_rel(&got[r * 1024..(r + 1) * 1024], &want, 1e-3);
            assert!(wr < 2.5e-3, "eps {eps} row {r}: worst relative error {wr}");
        }
    }
    // the eps matters on the tiny row: the two epsilons give outputs that differ by far more than a fp16 rounding
    let a = rms_host(&x[3 * 1024..4 * 1024], &w, 1e-5);
    let b = rms_host(&x[3 * 1024..4 * 1024], &w, 1e-6);
    assert!(worst_rel(&a, &b, 1e-3) > 0.5, "the tiny row does not separate eps 1e-5 from 1e-6");
}

#[test]
fn the_fused_residual_add_writes_the_stream_and_normalises_it() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 5usize;
    let mut rng = Rng(12);
    let x = rng.vec(rows * 1024, -3.0, 3.0);
    let raw: Vec<u16> = rng.vec(rows * 1024, -2.0, 2.0).iter().map(|&v| h(v as f64)).collect();
    let bias: Vec<u16> = rng.vec(1024, -1.0, 1.0).iter().map(|&v| h(v as f64)).collect();
    let w: Vec<u16> = (0..1024).map(|_| bf16(rng.uniform(0.2, 1.8))).collect();
    for with_bias in [true, false] {
        let mut dx = up(&x);
        let mut dy = DevBuf::zeroed(rows * 1024 * 2).unwrap();
        let (draw, dbias, dw) = (up(&raw), up(&bias), up(&w));
        gv::rmsnorm(&mut dx, Some(&draw), with_bias.then_some(&dbias), &dw, &mut dy, rows, 1e-5, &st).unwrap();
        st.sync().unwrap();
        let got_x = fetch32(&dx, rows * 1024);
        let got_y = fetch16(&dy, rows * 1024);
        for r in 0..rows {
            let xr: Vec<f32> = (0..1024)
                .map(|c| {
                    let i = r * 1024 + c;
                    // `y = mlp(..)` is an fp16 tensor with the bias already added (a second fp16 rounding), then `x += y` in fp32
                    let y = if with_bias { r16(f(raw[i]) + f(bias[c])) } else { f(raw[i]) };
                    (x[i] as f64 + y) as f32
                })
                .collect();
            assert!(xr.iter().zip(&got_x[r * 1024..]).all(|(a, b)| a == b), "row {r}: the stream is x + f16(raw + bias)");
            let want = rms_host(&xr, &w, 1e-5);
            assert!(worst_rel(&got_y[r * 1024..(r + 1) * 1024], &want, 1e-3) < 2.5e-3);
        }
    }
}

#[test]
fn layernorm_matches_the_formula_and_its_eps_is_the_forks() {
    gpu!();
    let st = Stream::new().unwrap();
    let (rows, dim) = (5usize, 4096usize);
    let mut rng = Rng(13);
    let mut x: Vec<u16> = rng.vec(rows * dim, -3.0, 5.0).iter().map(|&v| h(v as f64)).collect();
    // row 3: nearly constant, variance about 1e-6 (eps 1e-6 and 1e-5 differ by 3x here)
    for c in 0..dim {
        x[3 * dim + c] = h(2.0 + rng.uniform(-1.7e-3, 1.7e-3) as f64);
    }
    let w: Vec<u16> = rng.vec(dim, 0.5, 1.5).iter().map(|&v| h(v as f64)).collect();
    let b: Vec<u16> = rng.vec(dim, -0.5, 0.5).iter().map(|&v| h(v as f64)).collect();
    let host = |r: usize, eps: f64| -> Vec<f64> {
        let xr: Vec<f64> = (0..dim).map(|c| f(x[r * dim + c])).collect();
        let mean = xr.iter().sum::<f64>() / dim as f64;
        let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / dim as f64;
        (0..dim).map(|c| r16((xr[c] - mean) / (var + eps).sqrt() * f(w[c]) + f(b[c]))).collect()
    };
    for eps in [1e-6f32, 1e-5] {
        let mut dy = DevBuf::zeroed(rows * dim * 2).unwrap();
        gv::layernorm(&up(&x), &up(&w), &up(&b), &mut dy, rows, dim, eps, &st).unwrap();
        st.sync().unwrap();
        let got = fetch16(&dy, rows * dim);
        for r in 0..rows {
            let wr = worst_rel(&got[r * dim..(r + 1) * dim], &host(r, eps as f64), 1.0);
            assert!(wr < 2.5e-3, "eps {eps} row {r}: worst error {wr}");
        }
    }
    assert!(worst_rel(&host(3, 1e-6), &host(3, 1e-5), 1.0) > 0.2, "the near-constant row does not separate eps 1e-6 from 1e-5");
}

// ------------------------------------------------------------------------------------------------------------------ activations

/// erf to better than 1e-9 (series for small arguments, the continued fraction of erfc for large ones).
fn erf(x: f64) -> f64 {
    let a = x.abs();
    let v = if a < 2.5 {
        let (mut term, mut sum) = (a, a);
        for n in 1..200 {
            term *= -a * a / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        // erfc(a) = exp(-a^2) / (a sqrt(pi)) * 1 / (1 + 1/(2a^2) / (1 + 2/(2a^2) / (1 + ...)))
        let mut cf = 0.0;
        for k in (1..60).rev() {
            cf = (k as f64 / 2.0) / (a + cf);
        }
        1.0 - (-a * a).exp() / std::f64::consts::PI.sqrt() / (a + cf)
    };
    v.copysign(x)
}

#[test]
fn gelu_is_the_exact_erf_form_and_the_tanh_form_is_not_it() {
    gpu!();
    let st = Stream::new().unwrap();
    // every fp16 value in [-9, 9]
    let xs: Vec<u16> = (0u16..=0xFFFF).filter(|&b| f(b).is_finite() && f(b).abs() <= 9.0).collect();
    let exact = |v: f64| r16(0.5 * v * (1.0 + erf(v / std::f64::consts::SQRT_2)));
    let mut worst_exact_ulp = 0f64;
    let mut tanh_differs = 0usize;
    for flags in [0, 1] {
        let mut d = up(&xs);
        gv::gelu(&mut d, xs.len(), flags, &st).unwrap();
        st.sync().unwrap();
        let got = fetch16(&d, xs.len());
        for (i, &b) in xs.iter().enumerate() {
            let want = exact(f(b));
            if flags == 0 {
                // within one fp16 step of the correctly rounded exact value; ATen's formula `x * 0.5 * (1 + erf(x / sqrt 2))` in fp32 (mirrored here) loses
                // digits in `1 + erf` for x below -3 (the sum is below 6e-5 there), so up to two steps are allowed beyond that
                let ulp = (want.abs().max(6.1e-5)) * 2f64.powi(-10);
                let allowed = if f(b) < -3.0 { 2.1 } else { 1.01 };
                let steps = (got[i] - want).abs() / ulp;
                assert!(steps <= allowed, "x {} got {} want {want} ({steps} steps)", f(b), got[i]);
                worst_exact_ulp = worst_exact_ulp.max(steps);
            } else if got[i] != want {
                tanh_differs += 1;
            }
        }
    }
    assert!(worst_exact_ulp <= 2.1);
    assert!(
        tanh_differs > 100,
        "the tanh form must differ from the exact one on many fp16 inputs (it differed on {tanh_differs})"
    );
}

fn silu_h(g: f64) -> f64 {
    r16(g / (1.0 + (-g).exp()))
}

#[test]
fn the_block_activation_clamps_the_silu_and_the_up_and_the_two_orders_are_the_same_in_fp16() {
    gpu!();
    let st = Stream::new().unwrap();
    // gate and up over a sweep that crosses the limit of 10 on both sides
    let vals: Vec<f64> = (0..4096).map(|i| -40.0 + 80.0 * (i as f64) / 4095.0).collect();
    let n = vals.len();
    let g: Vec<u16> = vals.iter().map(|&v| h(v)).collect();
    let u: Vec<u16> = vals.iter().rev().map(|&v| h(v * 0.9)).collect();
    let (bg, bu) = (vec![0u16; n], vec![0u16; n]);
    let mut gu = vec![0u16; 2 * n];
    gu[..n].copy_from_slice(&g);
    gu[n..].copy_from_slice(&u);
    let mut outs = Vec::new();
    for flags in [0, 1, 2, 4] {
        let mut d = up(&gu);
        gv::swiglu_block(&mut d, &up(&bg), &up(&bu), n, 2 * n, 10.0, flags, 1, &st).unwrap();
        st.sync().unwrap();
        let all: Vec<u16> = d.to_vec().unwrap();
        outs.push(all[..n].iter().map(|&b| f(b)).collect::<Vec<f64>>());
    }
    // flags 0: min(silu(g), 10) * clamp(u, -10, 10), silu in half precision: within a few fp16 steps of the exact value
    for i in 0..n {
        let gv_ = f(g[i]);
        let s = silu_h(gv_).min(10.0);
        let uc = f(u[i]).clamp(-10.0, 10.0);
        let want = r16(s * uc);
        let tol = want.abs().max(1e-3) * 6e-3 + 2e-3;
        assert!((outs[0][i] - want).abs() <= tol, "i {i}: g {gv_} u {} got {} want {want}", f(u[i]), outs[0][i]);
    }
    // the clamps matter: without them (flags 1) values beyond +-10 are not clamped
    assert!((0..n).any(|i| (outs[1][i] - outs[0][i]).abs() > 1.0), "dropping the clamps changes nothing on a sweep to 40");
    // clamping g before the silu (flags 2, the merger's order) gives the same fp16 values: silu(10) = 9.99955 rounds to 10.0 and min(silu(g), 10) is 10
    // above it, so the fork's order and the HF order are one function on fp16 (this mutation of the table is an equivalent one)
    let differing = (0..n).filter(|&i| outs[2][i] != outs[0][i]).count();
    assert!(differing <= n / 200, "the two clamp orders differ on {differing} of {n} fp16 values");
    // without biases (flags 4) the result is what zero biases give, and with a bias it is not
    assert!((0..n).all(|i| outs[3][i] == outs[0][i]));
    let bias: Vec<u16> = vec![h(0.5); n];
    let mut d = up(&gu);
    gv::swiglu_block(&mut d, &up(&bias), &up(&bias), n, 2 * n, 10.0, 0, 1, &st).unwrap();
    st.sync().unwrap();
    let with: Vec<u16> = d.to_vec().unwrap();
    assert!((0..n).any(|i| (f(with[i]) - outs[0][i]).abs() > 0.1), "the biases are not applied");
}

#[test]
fn the_merger_activation_clamps_before_the_silu_in_fp32() {
    gpu!();
    let st = Stream::new().unwrap();
    let n = 4096usize;
    let vals: Vec<f64> = (0..n).map(|i| -40.0 + 80.0 * (i as f64) / (n - 1) as f64).collect();
    let g: Vec<u16> = vals.iter().map(|&v| h(v)).collect();
    let u: Vec<u16> = vals.iter().rev().map(|&v| h(v * 0.9)).collect();
    let mut gu = vec![0u16; 2 * n];
    gu[..n].copy_from_slice(&g);
    gu[n..].copy_from_slice(&u);
    let run = |flags: i32| -> Vec<f64> {
        let mut d = up(&gu);
        gv::swiglu_merger(&mut d, n, 2 * n, 10.0, 1, flags, &st).unwrap();
        st.sync().unwrap();
        let all: Vec<u16> = d.to_vec().unwrap();
        all[..n].iter().map(|&b| f(b)).collect()
    };
    let got = run(0);
    for i in 0..n {
        let gc = f(g[i]).min(10.0);
        let uc = f(u[i]).clamp(-10.0, 10.0);
        let s = r16(gc / (1.0 + (-gc).exp()));
        let want = r16(uc * s);
        assert!((got[i] - want).abs() <= want.abs().max(1e-3) * 2e-3 + 1e-4, "i {i}: got {} want {want}", got[i]);
    }
    let unclamped = run(1);
    assert!((0..n).any(|i| (unclamped[i] - got[i]).abs() > 1.0), "dropping the merger's clamps changes nothing on a sweep to 40");
}

// ------------------------------------------------------------------------------------------------------------------ qkv, attention

fn inv_freq() -> Vec<f32> {
    (0..16).map(|j| 1.0f32 / 10000f32.powf((2 * j) as f32 / 32.0)).collect()
}

/// Host mirror of `qkv_prep` for one patch: returns the new row `[3072]`.
fn qkv_host(raw: &[u16], bias: &[u16], qn: &[u16], kn: &[u16], inv: &[f32], gw: usize, p: usize, flags: i32) -> Vec<f64> {
    let (bw, blk, w4) = (gw / 2, p / 4, p % 4);
    let (r, c) = ((blk / bw) * 2 + w4 / 2, (blk % bw) * 2 + w4 % 2);
    let (ra, ca) = if flags & 1 != 0 { (c, r) } else { (r, c) };
    let mut out = vec![0.0; 3072];
    for head in 0..32 {
        let nw = if head < 16 { qn } else { kn };
        let a: Vec<f64> = (0..64)
            .map(|d| {
                let v = f(raw[head * 64 + d]);
                if flags & 2 != 0 {
                    v
                } else {
                    r16(v + f(bias[head * 64 + d]))
                }
            })
            .collect();
        let rmf = if flags & 4 != 0 { 1.0 } else { 1.0 / (a.iter().map(|v| v * v).sum::<f64>() / 64.0 + 1e-5).sqrt() };
        let nrm: Vec<f64> = (0..64).map(|d| if flags & 4 != 0 { a[d] } else { r16(a[d] * rmf * bf(nw[d])) }).collect();
        for d in 0..32 {
            let ang = if d < 16 { ra as f64 * inv[d] as f64 } else { ca as f64 * inv[d - 16] as f64 };
            let (s, co) = if flags & 8 != 0 { (0.0, 1.0) } else { (ang.sin(), ang.cos()) };
            out[head * 64 + d] = r16(nrm[d] * co - nrm[d + 32] * s);
            out[head * 64 + d + 32] = r16(nrm[d + 32] * co + nrm[d] * s);
        }
    }
    for i in 0..1024 {
        out[2048 + i] = if flags & 2 != 0 { f(raw[2048 + i]) } else { r16(f(raw[2048 + i]) + f(bias[2048 + i])) };
    }
    out
}

#[test]
fn qkv_prep_is_bias_head_norm_and_the_2d_rope_and_each_flag_changes_it() {
    gpu!();
    let st = Stream::new().unwrap();
    let (gh, gw) = (6usize, 8usize);
    let p = gh * gw;
    let mut rng = Rng(17);
    let raw: Vec<u16> = rng.vec(p * 3072, -3.0, 3.0).iter().map(|&v| h(v as f64)).collect();
    let bias: Vec<u16> = rng.vec(3072, -1.0, 1.0).iter().map(|&v| h(v as f64)).collect();
    let qn: Vec<u16> = (0..64).map(|_| bf16(rng.uniform(0.3, 1.7))).collect();
    let kn: Vec<u16> = (0..64).map(|_| bf16(rng.uniform(0.3, 1.7))).collect();
    let inv = inv_freq();
    let mut base: Option<Vec<f64>> = None;
    for flags in [0, gv::qkv::SWAP_AXES, gv::qkv::NO_BIAS, gv::qkv::NO_HEAD_NORM, gv::qkv::NO_ROPE] {
        let mut d = up(&raw);
        gv::qkv_prep(&mut d, &up(&bias), &up(&qn), &up(&kn), &up(&inv), p, gw, 1e-5, flags, &st).unwrap();
        st.sync().unwrap();
        let got = fetch16(&d, p * 3072);
        let mut want = Vec::new();
        for pi in 0..p {
            want.extend(qkv_host(&raw[pi * 3072..(pi + 1) * 3072], &bias, &qn, &kn, &inv, gw, pi, flags));
        }
        // __sinf / __cosf at angles up to ~ 7 radians, a norm and two roundings: a few fp16 steps
        let wr = worst_rel(&got, &want, 0.05);
        assert!(wr < 8e-3, "flags {flags}: worst relative error {wr}");
        match &base {
            None => base = Some(got),
            Some(b) => assert!(rel_l2(&got, b) > 1e-2, "flag {flags} changes the output by only {}", rel_l2(&got, b)),
        }
    }
}

fn attention_host(qkv: &[u16], p: usize, causal: bool) -> Vec<f64> {
    let mut out = vec![0.0; p * 1024];
    for head in 0..16 {
        let q = |i: usize, d: usize| f(qkv[i * 3072 + head * 64 + d]);
        let k = |i: usize, d: usize| f(qkv[i * 3072 + 1024 + head * 64 + d]);
        let v = |i: usize, d: usize| f(qkv[i * 3072 + 2048 + head * 64 + d]);
        for i in 0..p {
            let n = if causal { i + 1 } else { p };
            let s: Vec<f64> = (0..n).map(|j| (0..64).map(|d| q(i, d) * k(j, d)).sum::<f64>() / 8.0).collect();
            let m = s.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for d in 0..64 {
                out[i * 1024 + head * 64 + d] = (0..n).map(|j| e[j] * v(j, d)).sum::<f64>() / z;
            }
        }
    }
    out
}

#[test]
fn attention_is_bidirectional_softmax_attention_of_head_dim_64() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(19);
    for p in [1usize, 2, 63, 64, 65, 130, 300] {
        let qkv: Vec<u16> = (0..p * 3072)
            .map(|i| h(if (i % 3072) < 2048 { rng.uniform(-1.5, 1.5) } else { rng.uniform(-2.0, 2.0) } as f64))
            .collect();
        for mask in [gv::mask::NONE, gv::mask::CAUSAL] {
            let mut out = DevBuf::zeroed(p * 1024 * 2).unwrap();
            gv::attention(&up(&qkv), &mut out, p, mask, &st).unwrap();
            st.sync().unwrap();
            let got = fetch16(&out, p * 1024);
            let want = attention_host(&qkv, p, mask == gv::mask::CAUSAL);
            let e = rel_l2(&got, &want);
            assert!(e < 2e-3, "P {p} mask {mask}: rel L2 {e}");
        }
        if p >= 64 {
            // causal and bidirectional differ
            let mut a = DevBuf::zeroed(p * 1024 * 2).unwrap();
            let mut b = DevBuf::zeroed(p * 1024 * 2).unwrap();
            let q = up(&qkv);
            gv::attention(&q, &mut a, p, gv::mask::NONE, &st).unwrap();
            gv::attention(&q, &mut b, p, gv::mask::CAUSAL, &st).unwrap();
            st.sync().unwrap();
            assert!(rel_l2(&fetch16(&a, p * 1024), &fetch16(&b, p * 1024)) > 1e-2);
        }
    }
}

#[test]
fn attention_v2_is_bit_identical_to_v1() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(41);
    for p in [1usize, 2, 7, 63, 64, 65, 127, 128, 129, 200, 257, 1000, 1472, 4100] {
        let qkv: Vec<u16> = (0..p * 3072)
            .map(|i| h(if (i % 3072) < 2048 { rng.uniform(-3.0, 3.0) } else { rng.uniform(-2.0, 2.0) } as f64))
            .collect();
        let q = up(&qkv);
        for mask in [gv::mask::NONE, gv::mask::CAUSAL] {
            let (mut a, mut b) = (DevBuf::zeroed(p * 1024 * 2).unwrap(), DevBuf::zeroed(p * 1024 * 2).unwrap());
            gv::attention_variant(&q, &mut a, p, mask, 1, &st).unwrap();
            gv::attention_variant(&q, &mut b, p, mask, 2, &st).unwrap();
            st.sync().unwrap();
            let (va, vb): (Vec<u16>, Vec<u16>) = (a.to_vec().unwrap(), b.to_vec().unwrap());
            assert!(va[..p * 1024] == vb[..p * 1024], "P {p} mask {mask}: v2 differs from v1");
        }
    }
}

/// The device's mma.sync peak (fp16 in, fp32 accumulate): the ceiling of every tensor-core kernel of the tower. Printed.
#[test]
#[ignore = "benchmark: cargo test --release -p gb-cuda --test glm_vision -- --ignored --nocapture mma_peak attention_speed"]
fn mma_peak() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut sink = DevBuf::zeroed(1024).unwrap();
    for blocks in [96usize, 192, 384, 768] {
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
        eprintln!("mma.sync m16n8k16 peak, {blocks} CTAs of 8 warps: {:.1} TFLOP/s", flop / 1e12 / best);
    }
}

/// Speed of the two attention kernels at the sizes of an image (one layer; the tower runs 24): printed, not asserted.
#[test]
#[ignore = "benchmark: cargo test --release -p gb-cuda --test glm_vision -- --ignored --nocapture mma_peak attention_speed"]
fn attention_speed() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(43);
    for p in [1472usize, 8000, 31828] {
        let qkv: Vec<u16> = (0..p * 3072).map(|_| h(rng.uniform(-2.0, 2.0) as f64)).collect();
        let q = up(&qkv);
        let mut out = DevBuf::zeroed(p * 1024 * 2).unwrap();
        for variant in [1, 2] {
            gv::attention_variant(&q, &mut out, p, gv::mask::NONE, variant, &st).unwrap();
            st.sync().unwrap();
            let mut times = Vec::new();
            for _ in 0..5 {
                let t0 = std::time::Instant::now();
                gv::attention_variant(&q, &mut out, p, gv::mask::NONE, variant, &st).unwrap();
                st.sync().unwrap();
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let flops = 4.0 * (p as f64) * (p as f64) * 1024.0;
            eprintln!("attention v{variant}, P = {p}: {:.2} ms, {:.1} TFLOP/s", times[2], flops / 1e12 / (times[2] / 1e3));
        }
    }
}

#[test]
fn attention_is_deterministic_and_sees_every_key() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(23);
    let p = 1000usize;
    let mut qkv: Vec<u16> = (0..p * 3072).map(|_| h(rng.uniform(-1.0, 1.0) as f64)).collect();
    let run = |qkv: &[u16]| -> Vec<u16> {
        let mut out = DevBuf::zeroed(p * 1024 * 2).unwrap();
        gv::attention(&up(qkv), &mut out, p, gv::mask::NONE, &st).unwrap();
        st.sync().unwrap();
        out.to_vec().unwrap()
    };
    let a = run(&qkv);
    assert_eq!(a, run(&qkv), "two runs differ");
    // changing the value of the LAST key moves the output of the FIRST query
    for d in 0..64 {
        qkv[(p - 1) * 3072 + 2048 + d] = h(f(qkv[(p - 1) * 3072 + 2048 + d]) + 30.0);
    }
    let b = run(&qkv);
    assert!(a[..64] != b[..64], "the first query does not see the last key");
}

// ------------------------------------------------------------------------------------------------------------------ weight preparation, splice, gemm

#[test]
fn the_conversions_fold_the_temporal_slots_and_order_the_merge() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(29);
    // patch fold: [O, 3, 2, 14, 14] bf16
    let o = 8usize;
    let src: Vec<u16> = (0..o * 1176).map(|_| bf16(rng.uniform(-0.2, 0.2))).collect();
    for slots in [2usize, 1] {
        let mut dst = DevBuf::zeroed(o * 592 * 2).unwrap();
        gv::fold_patch(&up(&src), &mut dst, o, 592, slots, &st).unwrap();
        st.sync().unwrap();
        let got: Vec<u16> = dst.to_vec().unwrap();
        for oo in 0..o {
            for k in 0..592 {
                let want = if k < 588 {
                    let (c, r) = (k / 196, k % 196);
                    let base = oo * 1176 + c * 392 + r;
                    let w0 = r16(bf(src[base]));
                    let w1 = if slots == 2 { r16(bf(src[base + 196])) } else { 0.0 };
                    h(w0 + w1)
                } else {
                    0
                };
                assert_eq!(got[oo * 592 + k], want, "o {oo} k {k} slots {slots}");
            }
        }
    }
    // downsample permutation: [O, C, 2, 2] -> [O, (j, c)]
    let (o, c) = (4usize, 16usize);
    let src: Vec<u16> = (0..o * c * 4).map(|_| bf16(rng.uniform(-1.0, 1.0))).collect();
    for mode in [0, 1, 2] {
        let mut dst = DevBuf::zeroed(o * c * 4 * 2).unwrap();
        let mut bad = DevBuf::zeroed(4).unwrap();
        gv::permute_downsample(&up(&src), &mut dst, o, c, mode, &mut bad, &st).unwrap();
        st.sync().unwrap();
        let got: Vec<u16> = dst.to_vec().unwrap();
        for oo in 0..o {
            for cc in 0..c {
                for j in 0..4usize {
                    let (kh, kw) = (j / 2, j % 2);
                    let dk = match mode {
                        0 => j * c + cc,
                        1 => cc * 4 + j,
                        _ => (kw * 2 + kh) * c + cc,
                    };
                    assert_eq!(got[oo * c * 4 + dk], h(bf(src[oo * c * 4 + cc * 4 + j])), "mode {mode} o {oo} c {cc} j {j}");
                }
            }
        }
        assert_eq!(bad.to_vec::<i32>().unwrap()[0], 0);
    }
}

#[test]
fn a_bf16_beyond_fp16_range_is_counted() {
    gpu!();
    let st = Stream::new().unwrap();
    let vals: Vec<u16> = vec![bf16(1.0), bf16(70000.0), bf16(-0.5), bf16(1e30), bf16(3.0e-9)];
    let mut d = up(&vals);
    let mut bad = DevBuf::zeroed(4).unwrap();
    gv::bf16_to_f16(&mut d, vals.len(), &mut bad, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<u16> = d.to_vec().unwrap();
    assert_eq!(bad.to_vec::<i32>().unwrap()[0], 2, "70000 and 1e30 are not finite in fp16");
    assert_eq!((f(got[0]), f(got[2])), (1.0, -0.5));
    assert_eq!(got[4], h(bf(vals[4])), "a bf16 that is a fp16 subnormal rounds like torch does");
}

#[test]
fn the_splice_writes_the_row_into_all_four_streams() {
    gpu!();
    let st = Stream::new().unwrap();
    let (rows, hidden, ns) = (10usize, 64usize, 4usize);
    let streams: Vec<f32> = (0..rows * ns * hidden).map(|i| -(i as f32) - 1.0).collect();
    let feats: Vec<f32> = (0..6 * hidden).map(|i| i as f32 * 0.5).collect();
    let ds = up(&streams);
    unsafe { gv::splice(ds.ptr(), &up(&feats), 3, 1, 4, hidden, ns, &st).unwrap() };
    st.sync().unwrap();
    let got = fetch32(&ds, rows * ns * hidden);
    for r in 0..rows {
        for s in 0..ns {
            for c in 0..hidden {
                let i = (r * ns + s) * hidden + c;
                let want = if (3..7).contains(&r) { feats[(r - 3 + 1) * hidden + c] } else { streams[i] };
                assert_eq!(got[i], want, "row {r} stream {s} col {c}");
            }
        }
    }
}

#[test]
fn the_gemm_is_a_times_w_transposed_with_strides_and_both_output_types() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(31);
    let (m, k, n) = (37usize, 96usize, 80usize);
    let (lda, ldc) = (k + 16, n + 24);
    let a: Vec<u16> = rng.vec(m * lda, -1.0, 1.0).iter().map(|&v| h(v as f64)).collect();
    let w: Vec<u16> = rng.vec(n * k, -1.0, 1.0).iter().map(|&v| h(v as f64)).collect();
    let want: Vec<f64> = (0..m * n).map(|i| (0..k).map(|d| f(a[(i / n) * lda + d]) * f(w[(i % n) * k + d])).sum()).collect();
    let mut ws = DevBuf::zeroed(1 << 20).unwrap();
    for f32_out in [true, false] {
        let mut c = DevBuf::zeroed(m * ldc * 4).unwrap();
        gv::gemm(&up(&a), lda, &up(&w), &mut c, ldc, m, k, n, f32_out, &mut ws, &st).unwrap();
        st.sync().unwrap();
        let got: Vec<f64> = if f32_out {
            let v: Vec<f32> = c.to_vec().unwrap();
            (0..m * n).map(|i| v[(i / n) * ldc + i % n] as f64).collect()
        } else {
            let v: Vec<u16> = c.to_vec().unwrap();
            (0..m * n).map(|i| f(v[(i / n) * ldc + i % n])).collect()
        };
        let e = rel_l2(&got, &want);
        assert!(e < if f32_out { 1e-6 } else { 1e-3 }, "f32_out {f32_out}: rel L2 {e}");
    }
}
