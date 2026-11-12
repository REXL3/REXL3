//! The block-128 Sylvester Hadamard rotation, H_128[i][j] = (-1)^popcount(i & j) as fp32 times fp32(1 / sqrt(128)).
//!
//! Blockwise forms (one GEMM per 128-wide block, written back in place) and batched forms (one GEMM over all blocks)
//! round differently and are used where the Python core uses each.

use super::dev::{copy2d, scale_mat, Gpu, Mat, Result, Scale};
use crate::gpu::cublas::Op;

pub const HAD: usize = 128;

pub fn matrix() -> Vec<f32> {
    let s = (1.0 / (HAD as f64).sqrt()) as f32;
    let mut h = vec![0f32; HAD * HAD];
    for i in 0..HAD {
        for j in 0..HAD {
            h[i * HAD + j] = if (i & j).count_ones() % 2 == 1 { -1.0 } else { 1.0 } * s;
        }
    }
    h
}

/// x <- x P_n, one GEMM per 128-column block.
pub fn rotate_cols_blockwise(g: &Gpu, x: &mut Mat, tmp: &Mat) -> Result<()> {
    let (k, n) = (x.rows, x.cols);
    for c0 in (0..n).step_by(HAD) {
        unsafe { g.sgemm(Op::N, Op::N, HAD, k, HAD, 1.0, g.had.ptr(), HAD, x.at(0, c0), n, 0.0, tmp.ptr(), HAD)? };
        copy2d(g, x.at(0, c0), n, tmp.ptr(), HAD, k, HAD)?;
    }
    Ok(())
}

/// x <- P_k x, one GEMM per 128-row block.
pub fn rotate_rows_blockwise(g: &Gpu, x: &mut Mat, tmp: &Mat) -> Result<()> {
    let (k, n) = (x.rows, x.cols);
    for r0 in (0..k).step_by(HAD) {
        unsafe { g.sgemm(Op::N, Op::N, n, HAD, HAD, 1.0, x.at(r0, 0), n, g.had.ptr(), HAD, 0.0, tmp.ptr(), n)? };
        copy2d(g, x.at(r0, 0), n, tmp.ptr(), n, HAD, n)?;
    }
    Ok(())
}

/// A scratch matrix big enough for both blockwise rotations of x.
pub fn scratch_for(x: &Mat) -> Result<Mat> {
    Mat::alloc(x.rows.max(x.cols), HAD)
}

/// P_k x as a new matrix, all row blocks in one batched GEMM.
pub fn rotate_rows(g: &Gpu, x: &Mat) -> Result<Mat> {
    let (k, n) = (x.rows, x.cols);
    let out = Mat::alloc(k, n)?;
    let stride = (HAD * n) as i64;
    unsafe {
        g.blas.sgemm_strided_batched(
            Op::N,
            Op::N,
            n as i32,
            HAD as i32,
            HAD as i32,
            1.0,
            x.ptr(),
            n as i32,
            stride,
            g.had.ptr(),
            HAD as i32,
            0,
            0.0,
            out.ptr(),
            n as i32,
            stride,
            (k / HAD) as i32,
        )?
    };
    Ok(out)
}

/// Rows per GEMM of [`rotate_cols`]: at most 2^19 rows of 128.
const MAX_GEMM_ROWS: usize = 1 << 19;

/// x P_n as a new matrix, all column blocks folded into one GEMM (chunked along k above 2^19 GEMM rows).
pub fn rotate_cols(g: &Gpu, x: &Mat) -> Result<Mat> {
    let (k, n) = (x.rows, x.cols);
    let out = Mat::alloc(k, n)?;
    let rows = (MAX_GEMM_ROWS * HAD / n).max(1);
    let step = if rows >= k { k } else { rows };
    for i in (0..k).step_by(step) {
        let r = step.min(k - i);
        unsafe { g.sgemm(Op::N, Op::N, HAD, r * n / HAD, HAD, 1.0, g.had.ptr(), HAD, x.at(i, 0), HAD, 0.0, out.at(i, 0), HAD)? };
    }
    Ok(out)
}

/// H <- P (S H S) P for the input Hessian: scale columns, rotate column blocks, scale rows, rotate row blocks.
pub fn rotate_sym_interleaved(g: &Gpu, h: &mut Mat, s: &[f32]) -> Result<()> {
    let tmp = scratch_for(h)?;
    scale_mat(g, h, s, Scale::MulCols)?;
    rotate_cols_blockwise(g, h, &tmp)?;
    scale_mat(g, h, s, Scale::MulRows)?;
    rotate_rows_blockwise(g, h, &tmp)?;
    g.sync()
}

/// S P H_rot P S as a new matrix: undoes the input rotation for a sign vector.
pub fn unrotate_sym(g: &Gpu, h_rot: &Mat, signs: &[f32]) -> Result<Mat> {
    let mut h = h_rot.clone_on(g)?;
    let tmp = scratch_for(&h)?;
    rotate_cols_blockwise(g, &mut h, &tmp)?;
    rotate_rows_blockwise(g, &mut h, &tmp)?;
    let s: Vec<f32> = signs
        .iter()
        .map(|v| {
            if *v > 0.0 {
                1.0
            } else if *v < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect();
    scale_mat(g, &mut h, &s, Scale::MulCols)?;
    scale_mat(g, &mut h, &s, Scale::MulRows)?;
    g.sync()?;
    Ok(h)
}

#[cfg(test)]
mod tests {
    #[test]
    fn matrix_is_orthonormal_and_symmetric() {
        let h = super::matrix();
        let s = (1.0 / 128f64.sqrt()) as f32;
        assert_eq!(h[0], s);
        assert_eq!(h[128 + 1], -s);
        assert_eq!(h[3 * 128 + 5], -s); // popcount(3 & 5) = 1
        assert_eq!(h[3 * 128 + 4], s);
        for i in 0..128 {
            for j in 0..128 {
                assert_eq!(h[i * 128 + j], h[j * 128 + i]);
                let d: f64 = (0..128).map(|t| h[i * 128 + t] as f64 * h[j * 128 + t] as f64).sum();
                assert!((d - if i == j { 1.0 } else { 0.0 }).abs() < 1e-6);
            }
        }
    }
}
