//! The 2D rotary angle table of the Qwen2/GLM4V-style vision towers, generic in the head dimension.
//!
//! The ExLlamaV3 fork builds it in `qwen2_position_embedding_grid_2d(grid_thw, head_dim, spatial_merge_size, theta)` for Qwen and GLM alike:
//! `inv_freq[j] = 1 / theta^(2j / dim)` for `j < dim / 2` where `dim = head_dim / 2`, one row per patch in block-major order (the order of
//! the patch tensor: 2x2 merge blocks in raster order, raster order inside a block) holding `row * inv_freq` then `col * inv_freq`
//! (`head_dim / 2` angles; the rotation applies them to the pairs of the head dimension, `cos`/`sin` of `cat(angles, angles)`).
//! `tables::rope_angles` is the same table with the Qwen3.8 head dimension (72) fixed; GLM-5.3-Flash's tower has head dimension
//! 1024 / 16 = 64 (32 angles a patch) and no learned position embedding at all (`pos_embedding = False`), so this table is its only
//! positional input from the CPU. `tests/glm_fixtures.rs` compares it with the fork's function.

/// `inv_freq` of the table: `dim / 2` float32 values for `head_dim` (`dim = head_dim / 2`), computed in float32 like the fork
/// (`1.0 / theta ** (arange(0, dim, 2) / dim)`).
pub fn inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    let dim = head_dim / 2;
    (0..dim / 2).map(|j| 1.0f32 / theta.powf((2 * j) as f32 / dim as f32)).collect()
}

/// The `(row, col)` of block-major patch `p` in a `gh x gw` grid merged `merge x merge`.
pub fn block_major_rc(p: usize, gw: usize, merge: usize) -> (usize, usize) {
    let (block, within) = (p / (merge * merge), p % (merge * merge));
    let blocks_per_row = gw / merge;
    ((block / blocks_per_row) * merge + within / merge, (block % blocks_per_row) * merge + within % merge)
}

/// `[gh * gw, 2 * inv.len()]` angles from a given `inv_freq`: `row * inv` then `col * inv` per patch, block-major.
pub fn angles_with(inv: &[f32], gh: usize, gw: usize, merge: usize) -> Vec<f32> {
    assert!(merge > 0 && gh % merge == 0 && gw % merge == 0, "the grid {gh}x{gw} is not a multiple of the merge size {merge}");
    let mut out = Vec::with_capacity(gh * gw * inv.len() * 2);
    for p in 0..gh * gw {
        let (r, c) = block_major_rc(p, gw, merge);
        out.extend(inv.iter().map(|f| r as f32 * f));
        out.extend(inv.iter().map(|f| c as f32 * f));
    }
    out
}

/// The angle table for `head_dim` and `theta`: `[gh * gw, head_dim / 2]`.
pub fn angles(head_dim: usize, theta: f32, gh: usize, gw: usize, merge: usize) -> Vec<f32> {
    angles_with(&inv_freq(head_dim, theta), gh, gw, merge)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_major_order_visits_each_merge_block_in_raster_order() {
        let got: Vec<(usize, usize)> = (0..8).map(|p| block_major_rc(p, 4, 2)).collect();
        assert_eq!(got, vec![(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]);
        // every patch of a grid is visited exactly once
        let mut seen = [false; 6 * 10];
        for p in 0..60 {
            let (r, c) = block_major_rc(p, 10, 2);
            assert!(!std::mem::replace(&mut seen[r * 10 + c], true));
        }
    }

    #[test]
    fn head_dim_72_reproduces_the_qwen_table() {
        let (gh, gw) = (6usize, 10usize);
        let a = angles(72, 10000.0, gh, gw, 2);
        let q = crate::tables::rope_angles(gh, gw, 10000.0);
        assert_eq!(a.len(), q.len());
        assert!(a.iter().zip(&q).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn glm_head_dim_has_32_angles_per_patch() {
        let a = angles(64, 10000.0, 4, 6, 2);
        assert_eq!(a.len(), 4 * 6 * 32);
        let inv = inv_freq(64, 10000.0);
        assert_eq!(inv.len(), 16);
        assert_eq!(inv[0], 1.0);
        // patch 3 is (row 1, col 1) of the first block: its row angles and column angles are equal
        assert_eq!(&a[3 * 32..3 * 32 + 16], &a[3 * 32 + 16..3 * 32 + 32]);
    }
}
