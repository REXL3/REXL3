//! Per-image lookup tables the encoder needs, computed on the CPU in the same float32 arithmetic as the ExLlamaV3 modules
//! (`Qwen3VLPosEmbedding.fast_pos_embed_interpolate` and `qwen2_position_embedding_grid_2d`), in block-major patch order:
//! the position-embedding taps (4 table indices and 4 bf16-rounded weights a patch) and the 2D rotary angles.

/// Side of the position-embedding grid (`num_position_embeddings` = 48 x 48).
pub const POS_GRID: usize = 48;
pub const HEAD_DIM: usize = 72;

/// `torch.linspace(0, end, steps)` in float32: `start + step * i` for the first half, `end - step * (steps - i - 1)` after it.
fn linspace(end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![0.0];
    }
    let step = end / (steps - 1) as f32;
    let half = steps / 2;
    (0..steps).map(|i| if i < half { step * i as f32 } else { end - step * (steps - i - 1) as f32 }).collect()
}

/// float32 -> bfloat16 -> float32, round to nearest even.
pub fn bf16_round(v: f32) -> f32 {
    let b = v.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return v; // NaN
    }
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1));
    f32::from_bits(r & 0xffff_0000)
}

/// Raster index of patch `p` in block-major order for a `gh x gw` grid with 2x2 merge blocks.
fn raster_of(p: usize, gw: usize) -> (usize, usize) {
    let bw = gw / 2;
    let (blk, within) = (p / 4, p % 4);
    let (br, bc) = (blk / bw, blk % bw);
    (br * 2 + within / 2, bc * 2 + within % 2)
}

/// Position-embedding plan for a `gh x gw` patch grid: `idx[p*4 + i]` table rows and `w[p*4 + i]` weights (bf16-rounded).
pub fn pos_embed_plan(gh: usize, gw: usize) -> (Vec<i32>, Vec<f32>) {
    let last = (POS_GRID - 1) as f32;
    let (hs, ws) = (linspace(last, gh), linspace(last, gw));
    let n = POS_GRID as i32;
    let side = |v: f32| -> (i32, i32, f32) {
        let f = v as i32; // .int() truncates toward zero
        (f, (f + 1).min(n - 1), v - f as f32)
    };
    let p_count = gh * gw;
    let (mut idx, mut w) = (Vec::with_capacity(p_count * 4), Vec::with_capacity(p_count * 4));
    for p in 0..p_count {
        let (r, c) = raster_of(p, gw);
        let (hf, hc, dh) = side(hs[r]);
        let (wf, wc, dw) = side(ws[c]);
        idx.extend([hf * n + wf, hf * n + wc, hc * n + wf, hc * n + wc]);
        w.extend([(1.0 - dh) * (1.0 - dw), (1.0 - dh) * dw, dh * (1.0 - dw), dh * dw].map(bf16_round));
    }
    (idx, w)
}

/// Rotary angles `[P, 36]` (float32): for a patch at grid (row, col), `row * inv_freq[0..18]` then `col * inv_freq[0..18]`,
/// with `inv_freq[j] = 1 / theta^(2j/36)`.
pub fn rope_angles(gh: usize, gw: usize, theta: f32) -> Vec<f32> {
    let dim = HEAD_DIM / 2; // 36
    let inv: Vec<f32> = (0..dim / 2).map(|j| 1.0f32 / theta.powf((2 * j) as f32 / dim as f32)).collect();
    let mut out = Vec::with_capacity(gh * gw * dim);
    for p in 0..gh * gw {
        let (r, c) = raster_of(p, gw);
        out.extend(inv.iter().map(|f| r as f32 * f));
        out.extend(inv.iter().map(|f| c as f32 * f));
    }
    out
}

/// `[P, 2, 36]` cosines then sines of [`rope_angles`], the layout of the `rope2d` kernel.
pub fn rope_cos_sin(gh: usize, gw: usize, theta: f32) -> Vec<f32> {
    let a = rope_angles(gh, gw, theta);
    let mut out = Vec::with_capacity(a.len() * 2);
    for row in a.chunks_exact(36) {
        out.extend(row.iter().map(|x| x.cos()));
        out.extend(row.iter().map(|x| x.sin()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_rounding_is_nearest_even() {
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(1.0 + 1.0 / 256.0), 1.0, "a tie rounds to the even mantissa");
        assert_eq!(bf16_round(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
        assert_eq!(bf16_round(0.0), 0.0);
    }

    #[test]
    fn block_major_order_visits_each_2x2_block_in_raster_order() {
        // 4x4 grid: patches 0..3 are the top-left block (0,0) (0,1) (1,0) (1,1), patches 4..7 the block to its right
        let got: Vec<(usize, usize)> = (0..8).map(|p| raster_of(p, 4)).collect();
        assert_eq!(got, vec![(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]);
    }

    #[test]
    fn weights_of_each_patch_sum_to_about_one() {
        let (_, w) = pos_embed_plan(16, 16);
        for c in w.chunks_exact(4) {
            assert!((c.iter().sum::<f32>() - 1.0).abs() < 0.02, "{c:?}");
        }
    }
}
