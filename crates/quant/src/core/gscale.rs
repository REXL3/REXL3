//! Global scale search. Sample tiles of the regularized weight (a wrapped diagonal of `width` tiles, then the
//! highest- and lowest-RMS tiles), times the LDLQ drift factor of the rate. For each scale s the error is
//! mean((Q(s t) / s - t)^2) with Q the tile encoder: coarse s = 0.1, 0.3, ..., 1.9 on every third sample, fine 5 points
//! 0.075 apart around the coarse best on all samples, then a parabola through the best fine point and its neighbours
//! (offset clamped to half a step), g >= 0.01.

use super::dev::{
    gather_rows_raw, gather_tiles, gs_err, mean32, mul_scalar, scaled_copies, tile_sumsq, upload, Gpu, Mat, Result,
};
use crate::encode::encode_tiles;
use gb_cuda::rt::DevBuf;
use gb_exl3::Bits;

pub const FINE_STEP: f64 = 0.075;
pub const FINE_POINTS: usize = 5;
pub const MIN_SCALE: f64 = 0.01;

pub fn coarse() -> Vec<f64> {
    (0..10).map(|i| 0.1 + 0.2 * i as f64).collect()
}

/// (tile row, tile column) of the wrapped diagonal, position-major.
pub fn diagonal_tiles(tk: usize, tn: usize, width: usize) -> (Vec<i32>, Vec<i32>) {
    let len = tk.max(tn);
    let mut a = Vec::with_capacity(len * width);
    let mut c = Vec::with_capacity(len * width);
    for i in 0..len {
        for w in 0..width {
            a.push((i % tk) as i32);
            c.push(((i + w) % tn) as i32);
        }
    }
    (a, c)
}

pub fn extreme_count(tk: usize, tn: usize, width: usize) -> usize {
    let len = tk.max(tn);
    (8usize.max(len * width / 16)).min((tk * tn).div_ceil(2))
}

/// The best of the fine points refined by a parabola through its neighbours: (scale, index of the best point).
pub fn refine(mses: &[f64], centers: &[f64], step: f64) -> (f64, usize) {
    let mut best = 0;
    for i in 1..mses.len() {
        if mses[i] < mses[best] {
            best = i;
        }
    }
    let mut offset = 0.0;
    if best > 0 && best < mses.len() - 1 {
        let (y0, y1, y2) = (mses[best - 1], mses[best], mses[best + 1]);
        let denom = y0 - 2.0 * y1 + y2;
        offset = if denom > 0.0 { 0.5 * (y0 - y2) / denom } else { 0.0 };
        offset = offset.clamp(-0.5, 0.5);
    }
    ((centers[best] + offset * step).max(MIN_SCALE), best)
}

pub fn fine_points(center: f64) -> Vec<f64> {
    let half = (FINE_POINTS / 2) as f64;
    (0..FINE_POINTS).map(|i| center + FINE_STEP * (i as f64 - half)).collect()
}

/// The sample tiles of w (k x n) in trellis order, times `drift`: (device tiles, count).
pub fn sample_tiles(g: &Gpu, w: &Mat, width: usize, drift: f64) -> Result<(DevBuf, usize)> {
    let (tk, tn) = (w.rows / 16, w.cols / 16);
    let (mut ta, mut tc) = diagonal_tiles(tk, tn, width);
    let num_x = extreme_count(tk, tn, width);
    let ms: Vec<f32> = tile_sumsq(g, w)?.iter().map(|s| mean32(*s, 256)).collect();
    let mut idx: Vec<usize> = (0..ms.len()).collect();
    idx.sort_by(|a, b| ms[*b].total_cmp(&ms[*a]).then(a.cmp(b)));
    let hi: Vec<usize> = idx[..num_x].to_vec();
    idx.sort_by(|a, b| ms[*a].total_cmp(&ms[*b]).then(a.cmp(b)));
    let lo: Vec<usize> = idx[..num_x].to_vec();
    for t in hi.iter().chain(&lo) {
        ta.push((t / tn) as i32);
        tc.push((t % tn) as i32);
    }
    let count = ta.len();
    let out = DevBuf::alloc(count * 256 * 4)?;
    gather_tiles(g, &out, w, &upload(&ta)?, &upload(&tc)?, count)?;
    mul_scalar(g, out.ptr() as *mut f32, out.ptr() as *const f32, count * 256, drift as f32)?;
    Ok((out, count))
}

/// Errors of the tile set `t` (count tiles) at each scale, encoder calls packed up to `max_tiles` tiles.
pub fn evaluate(g: &mut Gpu, t: &DevBuf, count: usize, scales: &[f64], bits: Bits, max_tiles: usize) -> Result<Vec<f64>> {
    let per = (max_tiles / count.max(1)).max(1);
    let mut out = Vec::with_capacity(scales.len());
    for chunk in scales.chunks(per) {
        let sets = chunk.len();
        let mul: Vec<f32> = chunk.iter().map(|s| *s as f32).collect();
        let inv: Vec<f32> = chunk.iter().map(|s| (1.0 / s) as f32).collect();
        let batch = DevBuf::alloc(sets * count * 256 * 4)?;
        scaled_copies(g, &batch, t, &upload(&mul)?, count * 256, sets)?;
        let enc = encode_tiles(g, &batch, sets * count, bits)?;
        let sums = gs_err(g, &enc.values, t, &upload(&inv)?, sets, count * 256)?;
        out.extend(sums.iter().map(|s| mean32(*s, count * 256) as f64));
    }
    Ok(out)
}

/// The global scale of the regularized weight w.
pub fn search(g: &mut Gpu, w: &Mat, bits: Bits, width: usize, drift: f64, max_tiles: usize) -> Result<f64> {
    let (samples, count) = sample_tiles(g, w, width, drift)?;
    // every third sample tile for the coarse grid
    let sub_idx: Vec<i32> = (0..count).step_by(3).map(|i| i as i32).collect();
    let subs = gather_rows_raw(g, samples.ptr() as *const f32, &upload(&sub_idx)?, sub_idx.len(), 256)?;
    let grid = coarse();
    let err = evaluate(g, &subs, sub_idx.len(), &grid, bits, max_tiles)?;
    let mut c = 0;
    for i in 1..err.len() {
        if err[i] < err[c] {
            c = i;
        }
    }
    let centers = fine_points(grid[c]);
    let err2 = evaluate(g, &samples, count, &centers, bits, max_tiles)?;
    Ok(refine(&err2, &centers, FINE_STEP).0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_and_sampling_geometry() {
        let c = coarse();
        assert_eq!(c.len(), 10);
        assert_eq!(c[0], 0.1);
        assert_eq!(c[9], 0.1 + 0.2 * 9.0);
        assert_eq!(fine_points(1.0), vec![1.0 - 0.15, 1.0 - 0.075, 1.0, 1.0 + 0.075, 1.0 + 0.15]);
        let (a, c) = diagonal_tiles(2, 5, 3);
        assert_eq!(a.len(), 15);
        assert_eq!(&a[..6], &[0, 0, 0, 1, 1, 1]);
        assert_eq!(&c[..6], &[0, 1, 2, 1, 2, 3]);
        assert_eq!((a[12], c[12]), (0, 4));
        assert_eq!(c[14], 1);
        assert_eq!(extreme_count(128, 512, 3), 96);
        assert_eq!(extreme_count(2, 2, 3), 2);
        assert_eq!(extreme_count(32, 32, 3), 8);
    }

    #[test]
    fn refine_takes_the_parabola_vertex_clamped_to_half_a_step() {
        let centers = fine_points(1.0);
        // symmetric around index 2: no offset
        assert_eq!(refine(&[4.0, 1.0, 0.0, 1.0, 4.0], &centers, FINE_STEP), (1.0, 2));
        // vertex at -0.25 steps from index 2
        let (g, b) = refine(&[2.25, 0.25, 0.25 * 0.25 + 0.0, 1.0 - 0.75 + 0.0625 + 0.5, 4.0], &centers, FINE_STEP);
        assert_eq!(b, 2);
        assert!(g < 1.0 && g > 1.0 - 0.5 * FINE_STEP);
        // best at an end: no refinement; first of equal minima
        assert_eq!(refine(&[0.0, 1.0, 2.0, 3.0, 4.0], &centers, FINE_STEP), (centers[0], 0));
        assert_eq!(refine(&[1.0, 0.5, 0.5, 2.0, 3.0], &centers, FINE_STEP).1, 1);
        // a flat or concave neighbourhood: no offset; the floor
        assert_eq!(refine(&[1.0, 0.5, 0.0, -0.5, -0.4], &centers, FINE_STEP).1, 3);
        assert_eq!(refine(&[1.0, 0.0, 1.0, 2.0, 3.0], &[0.0, 0.0, 0.0, 0.0, 0.0], FINE_STEP).0, MIN_SCALE);
    }
}
