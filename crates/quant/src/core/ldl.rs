//! Block LDL with 16 x 16 blocks: H = (I + L0) D (I + L0)^T, L0 strictly block-lower. Returns L0 (zero diagonal blocks).
//!
//! 1. Cholesky H = C C^T (cuSOLVER, lower, on a column-major copy). On failure, add `retry_damping` x mean(diag H) to
//!    H's diagonal (in place) and retry, up to `retries` times.
//! 2. Invert C's diagonal blocks (batched LU and solve against I).
//! 3. L = C blockdiag(C_ii^-1), one GEMM per block column; then the diagonal blocks are zeroed.

use super::dev::{
    diag, diag_blocks_colmajor, download, identity_blocks, mean32, reduce, transpose, tril_from_colmajor, upload,
    zero_diag_blocks, Diag, Gpu, Mat, QuantError, Red, Result,
};
use crate::gpu::cublas::Op;
use crate::gpu::cusolver::Fill;
use gb_cuda::rt::DevBuf;

pub const BLOCK: usize = 16;

/// f32(mean of the diagonal), accumulated in fp64.
pub fn diag_mean(g: &Gpu, h: &Mat) -> Result<f32> {
    let s = reduce(g, h.ptr(), h.ptr(), 1, h.rows, h.cols + 1, 0, Red::Sum)?;
    Ok(mean32(s[0], h.rows))
}

/// The lower Cholesky factor of h as a row-major matrix (zeros above the diagonal); damps h in place on retries.
pub fn cholesky(g: &Gpu, h: &mut Mat, retries: usize, retry_damping: f64) -> Result<Mat> {
    let n = h.rows;
    for attempt in 0..=retries {
        let cm = transpose(g, h)?;
        let (dev_bytes, host_bytes) = g.solver.potrf_workspace(Fill::Lower, n as i64, n as i64)?;
        let work = DevBuf::alloc(dev_bytes.max(1))?;
        let mut host = vec![0u8; host_bytes];
        let info = DevBuf::zeroed(4)?;
        unsafe {
            g.solver.potrf(Fill::Lower, n as i64, cm.ptr(), n as i64, work.ptr(), dev_bytes, &mut host, info.ptr() as *mut i32)?
        };
        let info: i32 = download::<i32>(g, &info, 1)?[0];
        if info == 0 {
            return tril_from_colmajor(g, &cm);
        }
        if attempt == retries {
            break;
        }
        let m = diag_mean(g, h)?;
        diag(g, h, m * retry_damping as f32, Diag::Add)?;
    }
    Err(QuantError::Cholesky(format!("failed after {retries} damping retries")))
}

/// Block LDL factor L0 of h (n x n, n % 16 == 0).
pub fn block_ldl(g: &Gpu, h: &mut Mat, retries: usize, retry_damping: f64) -> Result<Mat> {
    let n = h.rows;
    if h.cols != n || n % BLOCK != 0 {
        return Err(QuantError::Shape(format!("block LDL of a {} x {} matrix", h.rows, h.cols)));
    }
    let m = n / BLOCK;
    let c = cholesky(g, h, retries, retry_damping)?;
    // inverses of the diagonal blocks, column-major, as torch.linalg.inv computes them: getrf + getrs against I
    let a = diag_blocks_colmajor(g, &c)?;
    let inv = identity_blocks(g, m)?;
    let a_ptrs: Vec<u64> = (0..m).map(|b| a.at(b, 0) as u64).collect();
    let b_ptrs: Vec<u64> = (0..m).map(|b| inv.at(b, 0) as u64).collect();
    let da = upload(&a_ptrs)?;
    let db = upload(&b_ptrs)?;
    let piv = DevBuf::alloc(m * BLOCK * 4)?;
    let info = DevBuf::alloc(m * 4)?;
    unsafe {
        g.blas.sgetrf_batched(
            BLOCK as i32,
            da.ptr() as *const *mut f32,
            BLOCK as i32,
            piv.ptr() as *mut i32,
            info.ptr() as *mut i32,
            m as i32,
        )?;
        g.blas.sgetrs_batched(
            BLOCK as i32,
            BLOCK as i32,
            da.ptr() as *const *const f32,
            BLOCK as i32,
            piv.ptr() as *const i32,
            db.ptr() as *const *mut f32,
            BLOCK as i32,
            m as i32,
        )?;
    }
    // L = C blockdiag(inv): per block column, tmp (n x 16) = C[:, block] inv_b, written back
    let tmp = Mat::alloc(n, BLOCK)?;
    for b in 0..m {
        unsafe {
            g.sgemm(Op::T, Op::N, BLOCK, n, BLOCK, 1.0, inv.at(b, 0), BLOCK, c.at(0, b * BLOCK), n, 0.0, tmp.ptr(), BLOCK)?;
        }
        super::dev::copy2d(g, c.at(0, b * BLOCK), n, tmp.ptr(), BLOCK, n, BLOCK)?;
    }
    let mut l = c;
    zero_diag_blocks(g, &mut l, BLOCK)?;
    g.sync()?;
    Ok(l)
}
