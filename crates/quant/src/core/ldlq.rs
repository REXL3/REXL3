//! One-sided LDLQ rounding in the regularized domain.
//!
//! 16-row blocks from the last up to row 0, in spans of `buf_rows`. The input of block b is
//! X_b + sum over r > b of L[r, b]^T (X_r - Q_r): within a span it is accumulated block by block, and the finished
//! spans' contribution comes from a cache updated once per span.
//!
//! Energy order: the blocks are visited in descending tr(H_bb) of the rotated input Hessian (ties: the higher block
//! index first). H and X are permuted block-wise, the recursion runs on the permuted problem, and Q's rows and the
//! states' block rows are put back afterwards; the stored format does not change.

use super::dev::{
    binary, gather_blocks_i16, gather_rows, gather_sym, rows_to_tiles, tiles_to_rows, upload, Bin, Gpu, Mat, QuantError, Result,
};
use super::ldl::block_ldl;
use crate::encode::encode_into;
use crate::exl3_trellis::Encoded;
use crate::gpu::cublas::Op;
use gb_cuda::rt::DevBuf;
use gb_exl3::Bits;

/// A block permutation: position i of the permuted problem holds original block `blocks[i]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockOrder {
    pub blocks: Vec<usize>,
}

impl BlockOrder {
    /// Ascending tr(H_bb) (fp64), stable: the recursion visits the last position first, so the highest-energy block
    /// goes first and ties keep the reference order.
    pub fn by_energy(diag: &[f32]) -> BlockOrder {
        let m = diag.len() / 16;
        let energy: Vec<f64> = (0..m).map(|b| diag[16 * b..16 * b + 16].iter().map(|v| *v as f64).sum()).collect();
        let mut blocks: Vec<usize> = (0..m).collect();
        blocks.sort_by(|a, b| energy[*a].total_cmp(&energy[*b]));
        BlockOrder { blocks }
    }
    pub fn is_identity(&self) -> bool {
        self.blocks.iter().enumerate().all(|(i, b)| i == *b)
    }
    fn inverse(&self) -> Vec<usize> {
        let mut inv = vec![0; self.blocks.len()];
        for (i, b) in self.blocks.iter().enumerate() {
            inv[*b] = i;
        }
        inv
    }
    fn rows(blocks: &[usize]) -> Vec<i32> {
        blocks.iter().flat_map(|b| (0..16).map(move |r| (16 * b + r) as i32)).collect()
    }
}

/// The energy order of a rotated H (None when it is the identity) and the factor of the permuted H.
pub fn energy_factor(g: &Gpu, h_rot: &Mat, retries: usize, retry_damping: f64) -> Result<(Option<BlockOrder>, Option<Mat>)> {
    let d = super::dev::reduce(g, h_rot.ptr(), h_rot.ptr(), h_rot.rows, 1, 1, h_rot.cols + 1, super::dev::Red::Sum)?;
    let diag: Vec<f32> = d.iter().map(|v| *v as f32).collect();
    let order = BlockOrder::by_energy(&diag);
    if order.is_identity() {
        return Ok((None, None));
    }
    let idx = upload(&BlockOrder::rows(&order.blocks))?;
    let mut hp = gather_sym(g, h_rot, &idx)?;
    let l = block_ldl(g, &mut hp, retries, retry_damping)?;
    Ok((Some(order), Some(l)))
}

/// Rounds X (k x n) with the zero-diagonal factor L (k x k): (Q, states [k/16, n/16, 256] int16 on the device).
pub fn one_sided(g: &mut Gpu, x: &Mat, l: &Mat, bits: Bits, buf_rows: usize) -> Result<(Mat, DevBuf)> {
    let (k, n) = (x.rows, x.cols);
    if k % 16 != 0 || n % 128 != 0 || k % buf_rows != 0 || l.rows != k || l.cols != k {
        return Err(QuantError::Shape(format!("one-sided rounding of {k} x {n} (buf_rows {buf_rows}, L {}x{})", l.rows, l.cols)));
    }
    let tn = n / 16;
    let cache = Mat::zeros(k, n)?;
    let q = Mat::zeros(k, n)?;
    let states = DevBuf::zeroed(k / 16 * tn * 256 * 2)?;
    let err = Mat::alloc(buf_rows, n)?;
    let inp = Mat::alloc(16, n)?;
    let tiles = DevBuf::alloc(tn * 256 * 4)?;
    let mut enc = Encoded::alloc(tn)?;
    let mut states_out = states;
    for j in (buf_rows..=k).rev().step_by(buf_rows) {
        let i = j - buf_rows;
        for bj in (16..=buf_rows).rev().step_by(16) {
            let bi = bj - 16;
            let r = buf_rows - bj;
            if r > 0 {
                binary(g, err.ptr(), n, x.at(i + bj, 0), n, q.at(i + bj, 0), n, r, n, Bin::Sub)?;
                unsafe {
                    g.sgemm(Op::N, Op::T, n, 16, r, 1.0, err.ptr(), n, l.at(i + bj, i + bi), k, 1.0, cache.at(i + bi, 0), n)?;
                }
            }
            binary(g, inp.ptr(), n, x.at(i + bi, 0), n, cache.at(i + bi, 0), n, 16, n, Bin::Add)?;
            rows_to_tiles(g, &tiles, inp.ptr(), n, tn)?;
            encode_into(g, &tiles, tn, bits, &mut enc)?;
            tiles_to_rows(g, q.at(i + bi, 0), n, &enc.values, tn)?;
            states_out.copy_from_at((i + bi) / 16 * tn * 512, &enc.states, 0, tn * 512, &g.stream)?;
        }
        binary(g, err.ptr(), n, x.at(i, 0), n, q.at(i, 0), n, buf_rows, n, Bin::Sub)?;
        unsafe { g.sgemm(Op::N, Op::T, n, k, buf_rows, 1.0, err.ptr(), n, l.at(i, 0), k, 1.0, cache.ptr(), n)? };
    }
    g.sync()?;
    Ok((q, states_out))
}

/// [`one_sided`] in an optional block order (`l` the factor of the permuted H); results in the original row order.
pub fn one_sided_ordered(
    g: &mut Gpu,
    x: &Mat,
    l: &Mat,
    order: Option<&BlockOrder>,
    bits: Bits,
    buf_rows: usize,
) -> Result<(Mat, DevBuf)> {
    let Some(order) = order else {
        return one_sided(g, x, l, bits, buf_rows);
    };
    let xp = gather_rows(g, x, &upload(&BlockOrder::rows(&order.blocks))?)?;
    let (qp, sp) = one_sided(g, &xp, l, bits, buf_rows)?;
    let inv = order.inverse();
    let q = gather_rows(g, &qp, &upload(&BlockOrder::rows(&inv))?)?;
    let width = x.cols / 16 * 256;
    let states = DevBuf::alloc(sp.bytes())?;
    let inv32: Vec<i32> = inv.iter().map(|v| *v as i32).collect();
    gather_blocks_i16(g, &states, &sp, &upload(&inv32)?, inv.len(), width)?;
    g.sync()?;
    Ok((q, states))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_order_is_ascending_and_stable() {
        let mut diag = vec![0f32; 64];
        for (b, e) in [3.0f32, 1.0, 3.0, 2.0].iter().enumerate() {
            for v in &mut diag[16 * b..16 * b + 16] {
                *v = *e / 16.0;
            }
        }
        let o = BlockOrder::by_energy(&diag);
        // visited from the last position: block 2 (3.0, the later of the tie) first, then block 0, 3, 1
        assert_eq!(o.blocks, vec![1, 3, 0, 2]);
        assert!(!o.is_identity());
        let inv = o.inverse();
        for (i, b) in o.blocks.iter().enumerate() {
            assert_eq!(inv[*b], i);
        }
        assert_eq!(&BlockOrder::rows(&[1, 0])[..3], &[16, 17, 18]);
        assert_eq!(BlockOrder::rows(&[1, 0])[16], 0);
        let flat = vec![1f32; 64];
        assert!(BlockOrder::by_energy(&flat).is_identity());
    }
}
