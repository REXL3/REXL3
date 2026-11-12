//! Input Hessian preparation, once per Hessian (shared by every tensor fed the same activation):
//! H = H_sum / count; m = mean(diag H) (fallback if count is 0, m is not finite or m < 1e-20); H += sigma m I;
//! H <- P S H S P with the input signs S; block LDL (unless in fallback). With the options' energy order and refit,
//! also the energy-ordered factor and the refit H, which every tensor of the Hessian shares.

use super::dev::{diag, mul_scalar, Diag, Gpu, Mat, QuantError, Result};
use super::hadamard::{rotate_sym_interleaved, unrotate_sym};
use super::ldl::{block_ldl, diag_mean};
use super::ldlq::energy_factor;
use super::Options;

pub struct Prepared {
    pub k: usize,
    pub fallback: bool,
    /// the input signs (k)
    pub su: Vec<f32>,
    /// the rotated, damped H (absent when uncalibrated)
    pub h_rot: Option<Mat>,
    /// block LDL factor with zero diagonal blocks (absent in fallback)
    pub l: Option<Mat>,
    pub diag_mean: f64,
    pub count: u64,
    /// the damped H in the original basis (the rotated H rotated back), for the refit; when `opts.refit`
    pub refit_h: Option<Mat>,
    /// the energy block order (None: the identity) and the factor of the permuted H; when `opts.energy_order`
    pub energy: Option<(Option<super::ldlq::BlockOrder>, Option<Mat>)>,
}

/// `h_sum` is the (k x k) sum of x^T x over `count` rows, `su` the input signs.
pub fn prepare(g: &Gpu, h_sum: Option<&[f32]>, k: usize, count: u64, su: &[f32], opts: &Options) -> Result<Prepared> {
    if su.len() != k {
        return Err(QuantError::Shape(format!("{} input signs for k = {k}", su.len())));
    }
    let base = |fallback, h_rot, l, diag_mean| Prepared {
        k,
        fallback,
        su: su.to_vec(),
        h_rot,
        l,
        diag_mean,
        count,
        refit_h: None,
        energy: None,
    };
    let Some(h_sum) = h_sum else {
        return Ok(base(true, None, None, 0.0));
    };
    let mut h = Mat::from_host(h_sum, k, k)?;
    let (fallback, dm) = if count == 0 {
        (true, 0.0)
    } else {
        mul_scalar(g, h.ptr(), h.ptr(), k * k, (1.0 / count as f64) as f32)?;
        let m = diag_mean(g, &h)? as f64;
        if !m.is_finite() {
            (true, 0.0)
        } else {
            (m < 1e-20, m)
        }
    };
    diag(g, &mut h, (opts.sigma_reg * dm) as f32, Diag::Add)?;
    rotate_sym_interleaved(g, &mut h, su)?;
    let l = if fallback { None } else { Some(block_ldl(g, &mut h, opts.cholesky_retries, opts.cholesky_retry_damping)?) };
    let mut p = base(fallback, None, l, dm);
    if !fallback {
        if opts.energy_order {
            p.energy = Some(energy_factor(g, &h, opts.cholesky_retries, opts.cholesky_retry_damping)?);
        }
        if opts.refit {
            p.refit_h = Some(unrotate_sym(g, &h, su)?);
        }
    }
    p.h_rot = Some(h);
    g.sync()?;
    Ok(p)
}
