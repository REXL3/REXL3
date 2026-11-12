//! Regularization: output and input channel scales, the sign flips and the rotation into the quantizer's domain.
//!
//! 1. Output channel RMS c_j of W, normalized to mean 1 (an all-zero tensor is flagged `zeros`).
//! 2. Output scales on, except with a per-channel output sensitivity in auto mode: on iff E[s e] / (E[s] E[e]) <= the
//!    max ratio (e = c^2).
//! 3. If on: zero channels get c = 0.1, sv <- sv c + 1e-10. W <- W / sv; zero channels then get sv = 0. W <- W P_n.
//! 4. Input channel RMS r_i of that W (zeros -> 0.1); su <- su r / (-divisor) + 1e-10; W <- P_k (W / su).
//!
//! RMS and means accumulate in fp64 and round once to fp32.

use super::dev::{mean32, reduce, scale_mat, Gpu, Mat, Red, Result, Scale};
use super::hadamard::{rotate_cols_blockwise, rotate_rows_blockwise, scratch_for};
use super::{Options, OutScales};

pub struct Regularized {
    /// (k) full input scales
    pub su: Vec<f32>,
    /// (n) full output scales, 0 on zero channels
    pub sv: Vec<f32>,
    pub apply_out_scales: bool,
    pub zeros: bool,
    pub ratio: Option<f64>,
}

/// E[s e] / (E[s] E[e]) with e = c^2, in fp64; the sensitivity is cut or zero-padded to the channel count.
pub fn out_scales_ratio(c: &[f32], sensitivity: &[f32]) -> Option<f64> {
    let n = c.len();
    let e: Vec<f64> = c.iter().map(|v| (*v as f64) * (*v as f64)).collect();
    let s: Vec<f64> = (0..n).map(|i| sensitivity.get(i).map(|v| *v as f64).unwrap_or(0.0)).collect();
    if sensitivity.iter().take(n).any(|v| !v.is_finite()) {
        return None;
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let den = mean(&s) * mean(&e);
    if !den.is_finite() || den <= 0.0 {
        return None;
    }
    let se: Vec<f64> = s.iter().zip(&e).map(|(a, b)| a * b).collect();
    Some(mean(&se) / den)
}

/// Regularizes w (k x n, fp32, in place) with the signs su (k) and sv (n).
pub fn regularize(
    g: &Gpu,
    w: &mut Mat,
    su_signs: &[f32],
    sv_signs: &[f32],
    opts: &Options,
    out_sensitivity: Option<&[f32]>,
    fallback: bool,
) -> Result<Regularized> {
    let (k, n) = (w.rows, w.cols);
    let force = match opts.out_scales {
        OutScales::Auto => None,
        OutScales::Always => Some(true),
        OutScales::Never => Some(false),
    };
    let mut apply = force.unwrap_or(true);
    let mut ratio = None;

    let sums = reduce(g, w.ptr(), w.ptr(), n, k, n, 1, Red::SumSq)?;
    let mut c: Vec<f32> = sums.iter().map(|s| mean32(*s, k).sqrt()).collect();
    let mean = mean32(c.iter().map(|v| *v as f64).sum::<f64>(), n) as f64;
    let zeros;
    if mean > 1e-30 {
        let inv = (1.0 / mean) as f32;
        for v in c.iter_mut() {
            *v *= inv;
        }
        zeros = false;
        if let Some(s) = out_sensitivity {
            if !fallback {
                ratio = out_scales_ratio(&c, s);
                if force.is_none() {
                    if let Some(r) = ratio {
                        apply = r <= opts.out_scales_max_ratio;
                    }
                }
            }
        }
    } else {
        zeros = true;
        if force.is_some() {
            apply = true;
        }
    }
    let zero: Vec<bool> = c.iter().map(|v| v.abs() < 1e-30).collect();
    let mut sv: Vec<f32> = sv_signs.to_vec();
    if apply {
        for j in 0..n {
            if zero[j] {
                c[j] = 0.1;
            }
            sv[j] = sv[j] * c[j] + 1e-10f32;
        }
    }
    scale_mat(g, w, &sv, Scale::DivCols)?;
    for j in 0..n {
        if zero[j] {
            sv[j] = 0.0;
        }
    }
    let tmp = scratch_for(w)?;
    rotate_cols_blockwise(g, w, &tmp)?;

    let sums = reduce(g, w.ptr(), w.ptr(), k, n, 1, n, Red::SumSq)?;
    let inv_div = (1.0 / -opts.input_scale_divisor) as f32;
    let su: Vec<f32> = sums
        .iter()
        .zip(su_signs)
        .map(|(s, sign)| {
            let mut r = mean32(*s, n).sqrt();
            if r.abs() < 1e-30 {
                r = 0.1;
            }
            sign * r * inv_div + 1e-10f32
        })
        .collect();
    scale_mat(g, w, &su, Scale::DivRows)?;
    rotate_rows_blockwise(g, w, &tmp)?;
    g.sync()?;
    Ok(Regularized { su, sv, apply_out_scales: apply, zeros, ratio })
}

/// W <- g W, su <- su / g.
pub fn apply_global_scale(g: &Gpu, w: &mut Mat, su: &mut [f32], gs: f64) -> Result<()> {
    super::dev::mul_scalar(g, w.ptr(), w.ptr(), w.len(), gs as f32)?;
    let inv = (1.0 / gs) as f32;
    for v in su.iter_mut() {
        *v *= inv;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_is_one_for_flat_sensitivity_and_pads_with_zeros() {
        let c = [1.0f32, 2.0, 3.0, 4.0];
        let r = out_scales_ratio(&c, &[1.0, 1.0, 1.0, 1.0]).unwrap();
        assert!((r - 1.0).abs() < 1e-15);
        // sensitivity on the large channels only: above 1
        let r = out_scales_ratio(&c, &[0.0, 0.0, 1.0, 1.0]).unwrap();
        let e = [1.0, 4.0, 9.0, 16.0];
        let want = ((9.0 + 16.0) / 4.0) / ((2.0 / 4.0) * (30.0 / 4.0));
        assert!((r - want).abs() < 1e-15 && r > 1.0, "{r}");
        let _ = e;
        // shorter sensitivity is zero-padded; longer is cut
        assert_eq!(out_scales_ratio(&c, &[0.0, 0.0, 1.0]), out_scales_ratio(&c, &[0.0, 0.0, 1.0, 0.0]));
        assert_eq!(out_scales_ratio(&c, &[1.0, 1.0, 1.0, 1.0, 9.0]), out_scales_ratio(&c, &[1.0; 4]));
        assert_eq!(out_scales_ratio(&c, &[0.0; 4]), None);
        assert_eq!(out_scales_ratio(&c, &[f32::NAN, 1.0, 1.0, 1.0]), None);
    }
}
