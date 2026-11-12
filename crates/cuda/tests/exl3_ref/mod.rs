//! The f64 reference of an EXL3 linear layer, for the tests that measure a kernel's error against the exact dequantized weights.
#![allow(dead_code)]

use gb_cuda::exl3_linear::Exl3Weights;
use gb_cuda::{DevBuf, Stream};
use gb_exl3::checkpoint::LoadedGroup;
use gb_exl3::f16::f16_to_f64;

pub fn fwht(v: &mut [f64]) {
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

/// The exact dequantized weights [K][N] of a group, as f32 (fp16 values: exact).
pub fn weights(g: &LoadedGroup, st: &Stream) -> Vec<f32> {
    let w = Exl3Weights::from_loaded(g).unwrap();
    let mut wq = DevBuf::alloc(w.k * w.n * 2).unwrap();
    w.reconstruct(&mut wq, st).unwrap();
    st.sync().unwrap();
    wq.to_vec::<u16>().unwrap().iter().map(|&b| f16_to_f64(b) as f32).collect()
}

/// (x * suh) H / sqrt 128 . W . H / sqrt 128 * svh in f64.
pub fn linear_ref(g: &LoadedGroup, w: &[f32], x: &[f64]) -> Vec<f64> {
    let (k, n) = (g.group.k, g.group.n);
    let s = 1.0 / 128f64.sqrt();
    let mut xr: Vec<f64> = (0..k).map(|i| x[i] * g.suh[i]).collect();
    xr.chunks_mut(128).for_each(fwht);
    let mut c = vec![0f64; n];
    for i in 0..k {
        let a = xr[i] * s;
        let wrow = &w[i * n..(i + 1) * n];
        for j in 0..n {
            c[j] += a * wrow[j] as f64;
        }
    }
    c.chunks_mut(128).for_each(fwht);
    (0..n).map(|j| c[j] * s * g.svh[j]).collect()
}

/// Relative errors: (max abs / max |ref|, rms / rms ref, mean / rms ref, mean of err x sign(ref) / rms ref).
pub fn err(refv: &[f64], got: &[f32]) -> (f64, f64, f64, f64) {
    let (mut ee, mut rr, mut mx, mut mr, mut sum, mut shrink) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for (&w, &g) in refv.iter().zip(got) {
        let e = g as f64 - w;
        ee += e * e;
        rr += w * w;
        mx = mx.max(e.abs());
        mr = mr.max(w.abs());
        sum += e;
        shrink += e * w.signum();
    }
    let cnt = refv.len() as f64;
    let rms = (rr / cnt).sqrt();
    (mx / mr, (ee / rr).sqrt(), sum / cnt / rms, shrink / cnt / rms)
}

/// One expert's pair output in f64 for the fp16 input row `x`: gate and up, the clamped SwiGLU (`limit`), down. `w*` are the groups'
/// [`weights`].
pub fn expert_ref(g: [&LoadedGroup; 3], w: [&[f32]; 3], x: &[f64], limit: f64) -> Vec<f64> {
    let gg = linear_ref(g[0], w[0], x);
    let u = linear_ref(g[1], w[1], x);
    let a: Vec<f64> = gg.iter().zip(&u).map(|(&g, &u)| (g / (1.0 + (-g).exp())).min(limit) * u.clamp(-limit, limit)).collect();
    linear_ref(g[2], w[2], &a)
}
