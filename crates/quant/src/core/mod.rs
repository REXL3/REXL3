//! The per-tensor core: Hessian finalize and damping, sign flips and the block-128 Hadamard rotation, block-16 LDL,
//! LDLQ rounding (one-sided, energy order by default), the global scale search, the scale refit (with an optional
//! clamp and no-worse guard) and packing. Strict fp32 (no TF32); reductions accumulate in fp64 and round once.
//!
//! ```text
//! let prep = prepare(&g, Some(&h_sum), k, count, &su_signs, &opts)?;      // once per input Hessian
//! let out = quantize(&mut g, &prep, &tensor, &sv_signs, &opts)?;          // per tensor
//! ```

pub mod dev;
pub mod fixture;
pub mod gscale;
pub mod hadamard;
pub mod ldl;
pub mod ldlq;
pub mod prep;
pub mod refit;
pub mod regularize;

use dev::{binary, download, scale_mat, Bin, Gpu, Mat, QuantError, Result, Scale};
use gb_exl3::Bits;
use std::time::Instant;

pub use prep::{prepare, Prepared};

/// Rates the format stores: 1..8 bits, and 1.5 / 2.5 / 3.5 (mul1 codebook).
pub fn bits_of(k: f64) -> Option<Bits> {
    let t = k * 2.0;
    if t.fract() != 0.0 || !(2.0..=16.0).contains(&t) {
        return None;
    }
    let b = Bits::from_twice(t as u8)?;
    (b.is_integer() || [3, 5, 7].contains(&(t as u8))).then_some(b)
}

/// The mul1 codebook marker (the multiplier's bit pattern as int32).
pub const MUL1_MARKER: i32 = 0x83DCD12Du32 as i32;

/// The RMS drift LDLQ feedback causes, by rate (1.0 for rates not listed).
pub fn ldlq_drift(k: f64) -> f64 {
    match (k * 2.0) as u32 {
        2 => 1.08,
        3 => 1.035,
        4 => 1.018,
        5 => 1.009,
        6 => 1.004,
        _ => 1.0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutScales {
    Auto,
    Always,
    Never,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub k: f64,
    pub sigma_reg: f64,
    pub cholesky_retries: usize,
    pub cholesky_retry_damping: f64,
    pub out_scales: OutScales,
    pub out_scales_max_ratio: f64,
    pub input_scale_divisor: f64,
    pub gscale: bool,
    pub gscale_width: usize,
    pub gscale_max_tiles: usize,
    pub buf_rows: usize,
    /// visit the LDLQ blocks in descending tr(H_bb); off = the reference order
    pub energy_order: bool,
    pub refit: bool,
    pub refit_rounds: usize,
    /// clamp the refit's scales to [0.25, 4] per round and keep the refit only if no worse
    pub refit_guard: bool,
}

impl Options {
    /// The defaults at rate `k`: energy order on.
    pub fn new(k: f64) -> Options {
        Options {
            k,
            sigma_reg: 0.025,
            cholesky_retries: 10,
            cholesky_retry_damping: 0.05,
            out_scales: OutScales::Auto,
            out_scales_max_ratio: 1.05,
            input_scale_divisor: 1.24371088,
            gscale: true,
            gscale_width: 3,
            gscale_max_tiles: 65536,
            buf_rows: 128,
            energy_order: true,
            refit: true,
            refit_rounds: 2,
            refit_guard: false,
        }
    }

    /// The parity profile: the reference block order (energy order off), otherwise the defaults.
    pub fn parity(k: f64) -> Options {
        Options { energy_order: false, ..Options::new(k) }
    }

    pub fn bits(&self) -> Result<Bits> {
        bits_of(self.k).ok_or_else(|| QuantError::Shape(format!("unsupported rate {}", self.k)))
    }

    /// sigma / (1 + sigma): the share of the damped mean diagonal the damping added.
    pub fn damping_share(&self) -> f64 {
        self.sigma_reg / (1.0 + self.sigma_reg)
    }
}

/// n random signs (+1 / -1) from `seed` and a stream id (SplitMix64; the top bit of each draw).
pub fn signs(seed: u64, stream: u64, n: usize) -> Vec<f32> {
    let mut s = seed ^ stream.wrapping_mul(0xD1B5_4A32_D192_ED03);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            if z >> 63 == 1 {
                -1.0
            } else {
                1.0
            }
        })
        .collect()
}

/// n signs of a named stream: [`signs`] with the stream id FNV-1a 64 of `label` (the converter names the input
/// signs of a qmap `su:<qmap>` and the output signs of a tensor `sv:<key>`).
pub fn signs_named(seed: u64, label: &str, n: usize) -> Vec<f32> {
    signs(seed, gb_exl3::fnv1a(label.bytes()), n)
}

/// One weight in quantizer orientation: W (k = in_features, n = out_features), row-major fp32.
pub struct Tensor<'a> {
    pub weight: &'a [f32],
    pub k: usize,
    pub n: usize,
    pub out_sensitivity: Option<&'a [f32]>,
}

/// Seconds per stage (each boundary synchronizes the stream).
#[derive(Clone, Debug, Default)]
pub struct Times {
    pub stages: Vec<(&'static str, f64)>,
}

impl Times {
    fn add(&mut self, name: &'static str, t: &mut Instant) {
        let now = Instant::now();
        let dt = (now - *t).as_secs_f64();
        *t = now;
        if let Some(e) = self.stages.iter_mut().find(|(n, _)| *n == name) {
            e.1 += dt;
        } else {
            self.stages.push((name, dt));
        }
    }
    pub fn merge(&mut self, other: &Times) {
        for (n, v) in &other.stages {
            if let Some(e) = self.stages.iter_mut().find(|(m, _)| m == n) {
                e.1 += v;
            } else {
                self.stages.push((n, *v));
            }
        }
    }
}

pub struct Output {
    /// [k/16, n/16, 256] states, trellis order
    pub states: Vec<i16>,
    /// [k/16, n/16, 16 K] packed words
    pub trellis: Vec<i16>,
    /// fp16 bits
    pub suh: Vec<u16>,
    pub svh: Vec<u16>,
    pub g: f64,
    /// after refit, original basis
    pub proxy_err: f64,
    /// regularized domain, before refit
    pub proxy_err_rounding: f64,
    pub refit_err_before: Option<f64>,
    pub refit_kept: bool,
    pub apply_out_scales: bool,
    pub out_scales_ratio: Option<f64>,
    pub zeros: bool,
    pub times: Times,
}

/// tr(A^T B A), summed over 1024-column blocks of B (k x k): per block T = (A^T B[:, j0:j1])^T, then T . A[j0:j1]
/// as one cuBLAS dot product.
pub fn block_trace(g: &Gpu, a: &Mat, b: &Mat) -> Result<f64> {
    let (k, n) = (a.rows, a.cols);
    let mut total = 0.0;
    let out = gb_cuda::rt::DevBuf::alloc(4)?;
    let t = Mat::alloc(1024.min(k), n)?;
    for j0 in (0..k).step_by(1024) {
        let jb = 1024.min(k - j0);
        unsafe {
            g.sgemm(
                crate::gpu::cublas::Op::N,
                crate::gpu::cublas::Op::T,
                n,
                jb,
                k,
                1.0,
                a.ptr(),
                n,
                b.at(0, j0),
                k,
                0.0,
                t.ptr(),
                n,
            )?;
            g.sgemm(
                crate::gpu::cublas::Op::T,
                crate::gpu::cublas::Op::T,
                1,
                1,
                jb * n,
                1.0,
                t.ptr(),
                jb * n,
                a.at(j0, 0),
                1,
                0.0,
                out.ptr() as *mut f32,
                1,
            )?;
        }
        total += download::<f32>(g, &out, 1)?[0] as f64;
    }
    Ok(total)
}

/// diag(su) P_k Q_r P_n diag(sv): the weight the stored tensors decode to (before fp16 rounding of su / sv).
pub fn reconstruct(g: &Gpu, q_r: &Mat, su: &[f32], sv: &[f32]) -> Result<Mat> {
    let mut q = hadamard::rotate_rows(g, q_r)?;
    scale_mat(g, &mut q, su, Scale::MulRows)?;
    let mut q = hadamard::rotate_cols(g, &q)?;
    scale_mat(g, &mut q, sv, Scale::MulCols)?;
    Ok(q)
}

fn f16_bits(v: f32) -> u16 {
    gb_exl3::f16::f64_to_f16(v as f64)
}

/// Quantizes one tensor with its prepared input Hessian and output signs `sv_signs` (n).
pub fn quantize(g: &mut Gpu, prep: &Prepared, t: &Tensor, sv_signs: &[f32], opts: &Options) -> Result<Output> {
    let (k, n) = (t.k, t.n);
    if prep.k != k || t.weight.len() != k * n || sv_signs.len() != n {
        return Err(QuantError::Shape(format!("weight {k} x {n}, Hessian k = {}, {} output signs", prep.k, sv_signs.len())));
    }
    if prep.fallback {
        return Err(QuantError::Shape("uncalibrated (fallback) tensors are not supported yet".into()));
    }
    let bits = opts.bits()?;
    let mut times = Times::default();
    let mut clock = Instant::now();
    let w_orig = Mat::from_host(t.weight, k, n)?;
    let mut w = w_orig.clone_on(g)?;
    let reg = regularize::regularize(g, &mut w, &prep.su, sv_signs, opts, t.out_sensitivity, prep.fallback)?;
    let (mut su, sv) = (reg.su, reg.sv);
    times.add("regularize", &mut clock);

    let gs = if opts.gscale {
        let gs = gscale::search(g, &w, bits, opts.gscale_width, ldlq_drift(opts.k), opts.gscale_max_tiles)?;
        regularize::apply_global_scale(g, &mut w, &mut su, gs)?;
        gs
    } else {
        1.0
    };
    g.sync()?;
    times.add("gscale", &mut clock);

    if (opts.energy_order && prep.energy.is_none()) || (opts.refit && prep.refit_h.is_none()) {
        return Err(QuantError::Shape("the Hessian was prepared with other energy-order / refit options".into()));
    }
    let (q_r, states) = {
        let base = prep.l.as_ref().expect("calibrated");
        match (&prep.energy, opts.energy_order) {
            (Some((Some(order), Some(l))), true) => ldlq::one_sided_ordered(g, &w, l, Some(order), bits, opts.buf_rows)?,
            _ => ldlq::one_sided(g, &w, base, bits, opts.buf_rows)?,
        }
    };
    times.add("ldlq", &mut clock);

    let h_rot = prep.h_rot.as_ref().expect("calibrated");
    let e = Mat::alloc(k, n)?;
    binary(g, e.ptr(), n, w.ptr(), n, q_r.ptr(), n, k, n, Bin::Sub)?;
    let num = block_trace(g, &e, h_rot)?;
    let den = block_trace(g, &w, h_rot)?;
    drop(e);
    let proxy_rounding = num / den.max(1e-8);
    times.add("proxy", &mut clock);

    let q = reconstruct(g, &q_r, &su, &sv)?;
    drop(q_r);
    times.add("reconstruct", &mut clock);

    let (su_f, sv_f, proxy, e_before, kept) = if opts.refit {
        let h_fit = prep.refit_h.as_ref().expect("checked above");
        let metric = refit::Metric::new(g, h_fit, &w_orig, opts.damping_share())?;
        let r = refit::refit_scales(g, q, &su, &sv, &metric, opts.refit_rounds, opts.refit_guard)?;
        (r.su, r.sv, r.e_after, Some(r.e_before), r.kept)
    } else {
        (su, sv, proxy_rounding, None, true)
    };
    times.add("refit", &mut clock);

    let tk = k / 16;
    let tn = n / 16;
    let words = bits.tile_words();
    let mut packed = gb_cuda::rt::DevBuf::alloc(tk * tn * words * 2)?;
    gb_cuda::exl3_encode::pack_trellis(&mut packed, &states, tk as i32, tn as i32, bits.twice() as i32, &g.stream)?;
    let trellis: Vec<i16> = download(g, &packed, tk * tn * words)?;
    let states: Vec<i16> = download(g, &states, tk * tn * 256)?;
    times.add("pack", &mut clock);
    Ok(Output {
        states,
        trellis,
        suh: su_f.iter().map(|v| f16_bits(*v)).collect(),
        svh: sv_f.iter().map(|v| f16_bits(*v)).collect(),
        g: gs,
        proxy_err: proxy,
        proxy_err_rounding: proxy_rounding,
        refit_err_before: e_before,
        refit_kept: kept,
        apply_out_scales: reg.apply_out_scales,
        out_scales_ratio: reg.ratio,
        zeros: reg.zeros,
        times,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_drift_and_defaults() {
        for k in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0] {
            assert_eq!(bits_of(k).unwrap().as_f64(), k);
        }
        for k in [0.5, 4.5, 9.0, 1.25] {
            assert!(bits_of(k).is_none(), "{k}");
        }
        assert_eq!(ldlq_drift(1.0), 1.08);
        assert_eq!(ldlq_drift(2.5), 1.009);
        assert_eq!(ldlq_drift(4.0), 1.0);
        let o = Options::new(2.0);
        assert!(o.energy_order && !o.refit_guard && o.refit);
        assert!(!Options::parity(2.0).energy_order);
        assert!((o.damping_share() - 0.025 / 1.025).abs() < 1e-18);
        assert_eq!(MUL1_MARKER as u32, 0x83DCD12D);
        let s = signs(7, 0, 4096);
        assert_eq!(s, signs(7, 0, 4096));
        assert_ne!(s, signs(7, 1, 4096));
        // the converter oracle's sign source (forward_parity.py) draws the same values
        assert_eq!(signs_named(7, "su:a", 8), vec![-1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0]);
        assert!(s.iter().all(|v| *v == 1.0 || *v == -1.0));
        let neg = s.iter().filter(|v| **v < 0.0).count();
        assert!((1848..2248).contains(&neg), "{neg}");
    }
}
