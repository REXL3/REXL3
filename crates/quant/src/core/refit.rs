//! Scale refit with the trellis fixed, under tr(E^T H E) with H the damped input Hessian (original basis):
//!
//! ```text
//! for each round:  c = column scales of Q;  Q <- Q diag(c);  sv <- sv c
//!                  r = row scales of Q;     Q <- diag(r) Q;  su <- su r
//! ```
//!
//! - columns: c_j = (q_j^T H w_j) / (q_j^T H q_j) (1 where the denominator is <= 1e-30);
//! - rows: solve ((Q Q^T) o H) r = rowsum(Q o (H W)) with a 1e-6 x mean-diagonal ridge (LU); non-finite or
//!   non-positive r_i become 1; a singular system keeps the row scales.
//!
//! The reported error tr(E^T H' E) / tr(W^T H' W) uses H' = H minus the damping share of its mean diagonal.
//!
//! The optional guard clamps every round's scales to [0.25, 4] and keeps the refit only if the error with the
//! fp16-rounded scales is no worse than before it.

use super::dev::{
    binary, diag, download, reduce, reduce_all, scale_mat, transpose, upload, Bin, Diag, Gpu, Mat, Red, Result, Scale,
};
use super::ldl::diag_mean;
use crate::gpu::cublas::Op;
use gb_cuda::rt::DevBuf;

pub struct Metric<'a> {
    pub h: &'a Mat,
    pub h_err: Mat,
    pub w: &'a Mat,
    pub hw: Mat,
    pub den: f64,
}

/// out = a (k x k) b (k x n), row-major.
fn mm(g: &Gpu, a: &Mat, b: &Mat, out: &Mat) -> Result<()> {
    let (k, n) = (b.rows, b.cols);
    unsafe { g.sgemm(Op::N, Op::N, n, k, k, 1.0, b.ptr(), n, a.ptr(), k, 0.0, out.ptr(), n) }
}

fn sum_dot(g: &Gpu, a: &Mat, b: &Mat) -> Result<f64> {
    Ok(reduce_all(g, a.ptr(), b.ptr(), a.len(), Red::Dot)? as f32 as f64)
}

impl<'a> Metric<'a> {
    pub fn new(g: &Gpu, h: &'a Mat, w: &'a Mat, damping_share: f64) -> Result<Metric<'a>> {
        let mut h_err = h.clone_on(g)?;
        if damping_share > 0.0 {
            let d_off = damping_share * diag_mean(g, h)? as f64;
            diag(g, &mut h_err, d_off as f32, Diag::Sub)?;
        }
        let hw = Mat::alloc(w.rows, w.cols)?;
        mm(g, h, w, &hw)?;
        let den = if damping_share > 0.0 {
            let t = Mat::alloc(w.rows, w.cols)?;
            mm(g, &h_err, w, &t)?;
            sum_dot(g, w, &t)?
        } else {
            sum_dot(g, w, &hw)?
        };
        Ok(Metric { h, h_err, w, hw, den })
    }

    pub fn error(&self, g: &Gpu, q: &Mat) -> Result<f64> {
        let e = Mat::alloc(q.rows, q.cols)?;
        binary(g, e.ptr(), q.cols, self.w.ptr(), q.cols, q.ptr(), q.cols, q.rows, q.cols, Bin::Sub)?;
        let t = Mat::alloc(q.rows, q.cols)?;
        mm(g, &self.h_err, &e, &t)?;
        Ok(sum_dot(g, &e, &t)? / self.den.max(1e-30))
    }

    pub fn column_scales(&self, g: &Gpu, q: &Mat) -> Result<Vec<f32>> {
        let (k, n) = (q.rows, q.cols);
        let hq = Mat::alloc(k, n)?;
        mm(g, self.h, q, &hq)?;
        let num = reduce(g, q.ptr(), self.hw.ptr(), n, k, n, 1, Red::Dot)?;
        let dn = reduce(g, q.ptr(), hq.ptr(), n, k, n, 1, Red::Dot)?;
        Ok(num
            .iter()
            .zip(&dn)
            .map(|(a, b)| {
                let (a, b) = (*a as f32, *b as f32);
                if b > 1e-30 {
                    a / b.max(1e-30)
                } else {
                    1.0
                }
            })
            .collect())
    }

    pub fn row_scales(&self, g: &Gpu, q: &Mat) -> Result<Vec<f32>> {
        let (k, n) = (q.rows, q.cols);
        let mut a = Mat::alloc(k, k)?;
        unsafe { g.sgemm(Op::T, Op::N, k, k, n, 1.0, q.ptr(), n, q.ptr(), n, 0.0, a.ptr(), k)? };
        binary(g, a.ptr(), k, a.ptr(), k, self.h.ptr(), k, k, k, Bin::Mul)?;
        let rhs: Vec<f32> = reduce(g, q.ptr(), self.hw.ptr(), k, n, 1, n, Red::Dot)?.iter().map(|v| *v as f32).collect();
        let m = diag_mean(g, &a)?;
        diag(g, &mut a, m * 1e-6f32, Diag::Add)?;
        let r = match solve(g, &a, &rhs)? {
            Some(r) => r,
            None => return Ok(vec![1.0; k]),
        };
        Ok(r.into_iter().map(|v| if v.is_finite() && v > 0.0 { v } else { 1.0 }).collect())
    }
}

/// x with a x = b (LU with partial pivoting on a column-major copy, then two triangular solves); None if singular.
pub fn solve(g: &Gpu, a: &Mat, b: &[f32]) -> Result<Option<Vec<f32>>> {
    let k = a.rows;
    let cm = transpose(g, a)?;
    let lwork = unsafe { g.solver.sgetrf_workspace(k as i32, k as i32, cm.ptr(), k as i32)? };
    let work = DevBuf::alloc((lwork.max(1) as usize) * 4)?;
    let ipiv = DevBuf::alloc(k * 4)?;
    let info = DevBuf::zeroed(4)?;
    unsafe {
        g.solver.sgetrf(
            k as i32,
            k as i32,
            cm.ptr(),
            k as i32,
            work.ptr() as *mut f32,
            ipiv.ptr() as *mut i32,
            info.ptr() as *mut i32,
        )?
    };
    if download::<i32>(g, &info, 1)?[0] != 0 {
        return Ok(None);
    }
    let piv: Vec<i32> = download(g, &ipiv, k)?;
    let mut x = b.to_vec();
    for (i, p) in piv.iter().enumerate() {
        x.swap(i, (*p - 1) as usize);
    }
    let dx = upload(&x)?;
    unsafe {
        g.blas.strsm_left(false, true, k as i32, 1, 1.0, cm.ptr(), k as i32, dx.ptr() as *mut f32, k as i32)?;
        g.blas.strsm_left(true, false, k as i32, 1, 1.0, cm.ptr(), k as i32, dx.ptr() as *mut f32, k as i32)?;
    }
    Ok(Some(download(g, &dx, k)?))
}

pub struct Refit {
    pub q: Mat,
    pub su: Vec<f32>,
    pub sv: Vec<f32>,
    pub e_before: f64,
    pub e_after: f64,
    /// false when the guard rejected the refit (the scales were kept)
    pub kept: bool,
}

/// The guard's per-round scale clamp.
pub const GUARD_CLAMP: (f32, f32) = (0.25, 4.0);

fn f16_round(v: f32) -> f32 {
    gb_exl3::f16::f16_to_f64(gb_exl3::f16::f64_to_f16(v as f64)) as f32
}

/// Refits su (k) and sv (n) to the fixed Q (k x n, original basis); with `guard`, clamped and no-worse checked.
pub fn refit_scales(g: &Gpu, q: Mat, su: &[f32], sv: &[f32], metric: &Metric, rounds: usize, guard: bool) -> Result<Refit> {
    let e_before = metric.error(g, &q)?;
    let q0 = if guard { Some(q.clone_on(g)?) } else { None };
    let (mut q, mut su_f, mut sv_f) = (q, su.to_vec(), sv.to_vec());
    let clamp = |v: f32| if guard { v.clamp(GUARD_CLAMP.0, GUARD_CLAMP.1) } else { v };
    for _ in 0..rounds {
        let c: Vec<f32> = metric.column_scales(g, &q)?.into_iter().map(clamp).collect();
        scale_mat(g, &mut q, &c, Scale::MulCols)?;
        for (s, c) in sv_f.iter_mut().zip(&c) {
            *s *= c;
        }
        let r: Vec<f32> = metric.row_scales(g, &q)?.into_iter().map(clamp).collect();
        scale_mat(g, &mut q, &r, Scale::MulRows)?;
        for (s, r) in su_f.iter_mut().zip(&r) {
            *s *= r;
        }
    }
    let Some(q0) = q0 else {
        let e_after = metric.error(g, &q)?;
        return Ok(Refit { q, su: su_f, sv: sv_f, e_before, e_after, kept: true });
    };
    // the guard: Q re-expressed with the fp16-rounded scales, kept only if no worse
    let su_h: Vec<f32> = su_f.iter().map(|v| f16_round(*v)).collect();
    let sv_h: Vec<f32> = sv_f.iter().map(|v| f16_round(*v)).collect();
    let ratio = |new: &[f32], old: &[f32]| -> Vec<f32> {
        new.iter().zip(old).map(|(a, b)| if b.abs() > 1e-8 { a / b } else { *a }).collect()
    };
    let mut qh = q0.clone_on(g)?;
    scale_mat(g, &mut qh, &ratio(&su_h, su), Scale::MulRows)?;
    scale_mat(g, &mut qh, &ratio(&sv_h, sv), Scale::MulCols)?;
    let e_after = metric.error(g, &qh)?;
    if e_after <= e_before {
        Ok(Refit { q: qh, su: su_h, sv: sv_h, e_before, e_after, kept: true })
    } else {
        Ok(Refit { q: q0, su: su.to_vec(), sv: sv.to_vec(), e_before, e_after: e_before, kept: false })
    }
}
