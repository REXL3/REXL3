//! A step's rows laid out as the kernels want them: concatenated rows of several sequences with per-row and
//! per-sequence index arrays.

use gb_engine::{RowRequest, TreeRequest};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RowBatch {
    pub rows: usize,
    /// Token id of each row.
    pub ids: Vec<i32>,
    /// Absolute position of each row's token.
    pub positions: Vec<i32>,
    /// Slot of the sequence each row belongs to.
    pub row_slot: Vec<i32>,
    /// Row offsets of each sequence into the concatenated arrays (n_seq + 1 entries).
    pub seq_start: Vec<i32>,
    /// Slot of each sequence.
    pub seq_slot: Vec<i32>,
    /// Interleaved M-RoPE positions `[rows, 3]` (T, H, W) for prefill rows inside an image; `None` for text rows, which use
    /// `positions` plus the slot's rope delta (all three streams equal).
    pub rope3: Option<Vec<i32>>,
    /// The parent of every row as a GLOBAL row index (-1 for a sequence's root), for a window whose sequences are trees; `None`
    /// for plain chains (row i's parent is row i - 1 within its sequence). `positions` then hold the LOGICAL position of each row
    /// (root position + depth).
    pub parents: Option<Vec<i32>>,
}

impl RowBatch {
    pub fn from_requests(reqs: &[RowRequest]) -> Self {
        let mut b = RowBatch { seq_start: vec![0], ..Default::default() };
        for r in reqs {
            for (j, &t) in r.tokens.iter().enumerate() {
                b.ids.push(t as i32);
                b.positions.push((r.first_position + j as u64) as i32);
                b.row_slot.push(r.slot as i32);
            }
            b.rows += r.tokens.len();
            b.seq_start.push(b.rows as i32);
            b.seq_slot.push(r.slot as i32);
        }
        b
    }

    /// A window of trees: row i of a request sits at `first_position + depth(i)`; `parents` become global row indices.
    pub fn from_trees(reqs: &[TreeRequest]) -> Self {
        let mut b = RowBatch { seq_start: vec![0], parents: Some(Vec::new()), ..Default::default() };
        for r in reqs {
            let base = b.rows as i32;
            let depth = r.depths();
            for (j, &t) in r.tokens.iter().enumerate() {
                b.ids.push(t as i32);
                b.positions.push((r.first_position + depth[j] as u64) as i32);
                b.row_slot.push(r.slot as i32);
                b.parents.as_mut().unwrap().push(if r.parents[j] < 0 { -1 } else { base + r.parents[j] });
            }
            b.rows += r.tokens.len();
            b.seq_start.push(b.rows as i32);
            b.seq_slot.push(r.slot as i32);
        }
        b
    }

    pub fn n_seq(&self) -> usize {
        self.seq_slot.len()
    }

    /// Rows of sequence `i`.
    pub fn seq_rows(&self, i: usize) -> std::ops::Range<usize> {
        self.seq_start[i] as usize..self.seq_start[i + 1] as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_laid_out_per_sequence() {
        let reqs = vec![
            RowRequest { slot: 3, tokens: vec![10, 11, 12], first_position: 100 },
            RowRequest { slot: 0, tokens: vec![20], first_position: 5 },
        ];
        let b = RowBatch::from_requests(&reqs);
        assert_eq!(b.rows, 4);
        assert_eq!(b.ids, vec![10, 11, 12, 20]);
        assert_eq!(b.positions, vec![100, 101, 102, 5]);
        assert_eq!(b.row_slot, vec![3, 3, 3, 0]);
        assert_eq!(b.seq_start, vec![0, 3, 4]);
        assert_eq!(b.seq_slot, vec![3, 0]);
        assert_eq!(b.seq_rows(1), 3..4);
        assert_eq!(b.n_seq(), 2);
        assert_eq!(b.parents, None, "a chain batch carries no parents");
    }

    #[test]
    fn a_tree_window_sits_at_depth_positions_with_global_parents() {
        let reqs = vec![
            // 0 -> 1 -> 3, 0 -> 2: rows 1 and 2 share depth 1
            TreeRequest { slot: 3, tokens: vec![10, 11, 12, 13], parents: vec![-1, 0, 0, 1], first_position: 100 },
            TreeRequest { slot: 0, tokens: vec![20, 21], parents: vec![-1, 0], first_position: 5 },
        ];
        let b = RowBatch::from_trees(&reqs);
        assert_eq!(b.rows, 6);
        assert_eq!(b.ids, vec![10, 11, 12, 13, 20, 21]);
        // the position is the depth, not the row index: the sibling at row 2 is at 101, the grandchild at row 3 at 102
        assert_eq!(b.positions, vec![100, 101, 101, 102, 5, 6]);
        assert_eq!(b.parents, Some(vec![-1, 0, 0, 1, -1, 4]), "parents are global rows");
        assert_eq!(b.seq_start, vec![0, 4, 6]);
        assert_eq!(b.row_slot, vec![3, 3, 3, 3, 0, 0]);
        assert!(!reqs[0].is_chain() && reqs[1].is_chain());
        assert_eq!(reqs[0].depths(), vec![0, 1, 1, 2]);
    }
}
