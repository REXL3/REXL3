//! The deep-position attention path of the Nemotron 3.5 Super VL NoPE layers (`csrc/nemotron_attn_prefill.cu`): a flash-style
//! tensor-core kernel that serves every row at absolute position `>= FLASH_FROM_POS`, whichever entry point the row arrives by
//! (decode, verify window or prefill), beside the exact tree of [`crate::nemotron_attn`], which serves the rows below it.
//!
//! **One rule, one arithmetic per position, one path id** ([`FLASH_PATH_ID`]). The tree sums 512-key chunks of scalar fp32 products
//! in a fixed order. This path sums the same attention on the tensor cores (`mma.sync` m16n8k16, fp16 in, fp32 out) with an online
//! softmax over absolute 64-key tiles inside absolute 8,192-key splits, and merges the splits in index order. The two are not
//! bit-equal, by design; they are each equal to the oracle within fp16 terms (`docs/NEMOTRON_ATTN_LONG.md`). Because the choice is
//! made per row, from its absolute position alone, decode == verify window == prefill holds at every position: a row has the same
//! bits whichever call carried it and whichever rows were beside it.
//!
//! What it guarantees (and what `tests/nemotron_attn_long.rs` pins):
//! * **row invariance**: a row's output depends only on its own query, its slot's keys up to its position and the arithmetic above;
//!   not on how a prompt was sliced, on the other rows of the call, their number or their sequences, or on where the pages sit;
//! * **graph safety**: [`attn_long`] launches only kernels that read device buffers (the row plan is a kernel too), with a grid that is
//!   a function of `(rows, sequences, chunk bucket)`, which the model's graph key already holds; a launch made for more splits or row
//!   blocks than a call needs gives the same bits;
//! * the tree's kernels and its row index are as before for every row below the threshold. The scratch ([`Scratch::new_serving`])
//!   holds the tree's partials for the 16 chunks a row below the threshold can have, whatever the context, and the flash path's own,
//!   `rows x ceil(context / 8192)` groups of 32 heads of `(m, l, o)`: what scales with the context is one group per 8,192 keys.

use crate::attention::BlockTable;
use crate::nemotron_attn::Q_ROW_ELEMS;
use crate::nemotron_attn::{self as tree, NemoKv, RowIndex, Scratch};
pub use crate::nemotron_attn::{flash_splits, FLASH_CHUNK_KEYS, FLASH_FROM_POS, TREE_CHUNKS_BELOW};
use crate::{CudaError, DevBuf, Stream};

// `FLASH_FROM_POS` (8,192): the first absolute position (0-based) the flash path serves, a multiple of `FLASH_CHUNK_KEYS` (the keys of
// a split, `FL_KC` in the kernel source) and of the tree's chunk, so the first split of the first flash row is full and the tree never
// needs more than `TREE_CHUNKS_BELOW` (16) chunks for a row it serves. A call whose launch is built for at most that many chunks is the
// tree's alone, kernel for kernel.
/// Keys of one flash tile (the online softmax step and the unit P is rounded to fp16 in).
pub const FLASH_TILE_KEYS: usize = 64;
/// Rows of one slot a CTA serves, one warp each (`FL_ROWS` in the kernel source).
pub const FLASH_ROWS_PER_CTA: usize = 8;
/// The prefix-cache path id of the arithmetic of the whole model: part of what a cached state is valid for. Bump it with any change of
/// the split, the tile, the rounding points, the merge order or the position threshold.
pub const FLASH_PATH_ID: &str = "nemo-attn-flash-v1/split8192/tile64/from8192";

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_nemo_flash_prefill_i8(
            q: *const c_void,
            kc: *const c_void,
            vc: *const c_void,
            table: *const c_void,
            max_blocks: i32,
            slot: *const c_void,
            pos: *const c_void,
            blocks: *mut c_void,
            rows: i32,
            from: i32,
            row_blocks: i32,
            chunks: i32,
            stride: i32,
            part_ml: *mut c_void,
            part_o: *mut c_void,
            out: *mut c_void,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_nemo_flash_prefill(
            q: *const c_void,
            kc: *const c_void,
            vc: *const c_void,
            table: *const c_void,
            max_blocks: i32,
            slot: *const c_void,
            pos: *const c_void,
            blocks: *mut c_void,
            rows: i32,
            from: i32,
            row_blocks: i32,
            chunks: i32,
            stride: i32,
            part_ml: *mut c_void,
            part_o: *mut c_void,
            out: *mut c_void,
            s: *mut c_void,
        ) -> i32;
    }
}

fn i(v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| CudaError(1))
}

/// Whether a row at absolute position `pos` is served by the flash path.
pub const fn uses_flash(pos: usize) -> bool {
    pos >= FLASH_FROM_POS
}

/// An upper bound of the row blocks a call of `rows` rows of `n_seq` sequences can make (`0` for unknown): a sequence's rows at and
/// above the threshold are consecutive, so each sequence makes `ceil(len / 8)` blocks, which sum to at most
/// `n_seq + (rows - n_seq) / 8`. A launch is made for this many blocks; the plan kernel's count says how many are real.
pub fn max_row_blocks(rows: usize, n_seq: usize) -> usize {
    let runs = if n_seq == 0 { rows } else { n_seq.min(rows) };
    (runs + (rows - runs) / FLASH_ROWS_PER_CTA).clamp(1, rows.max(1))
}

/// Flash attention for the rows `first..first + rows` of `idx` whose position is at least `from_pos` (rows below it are left alone:
/// their output belongs to the tree), launched for `splits` splits and `row_blocks` row blocks, both upper bounds a call may be built
/// for. `q` and `out` are fp32 `[idx.rows][32][128]`; the rows' K and V must already be in the cache ([`tree::kv_write`]). Launches on
/// `st` and returns without synchronising; reads only device buffers, so it can be captured. The scratch's flash buffers hold the
/// partials of `scratch.rows()` rows of `scratch.flash_splits()` splits and the row plan; a call of more rows is run in passes of rows
/// (a row's bits do not depend on the pass).
#[allow(clippy::too_many_arguments)]
pub fn attn_flash(
    q: &DevBuf,
    kv: &NemoKv,
    table: &BlockTable,
    idx: &RowIndex,
    first: usize,
    rows: usize,
    from_pos: usize,
    splits: usize,
    row_blocks: usize,
    scratch: &mut Scratch,
    out: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    assert!(first + rows <= idx.rows, "nemotron_attn_prefill: rows {first}..{} of {}", first + rows, idx.rows);
    assert!(table.slots > 0 && kv.n_blocks > 0, "nemotron_attn_prefill needs a non-empty cache and block table");
    assert!(
        q.bytes() >= (first + rows) * Q_ROW_ELEMS * 4 && out.bytes() >= (first + rows) * Q_ROW_ELEMS * 4,
        "q or out too small"
    );
    assert!(q.ptr() as usize % 16 == 0 && out.ptr() as usize % 16 == 0, "q and out must be 16-byte aligned");
    let (rows_cap, f_splits) = (scratch.rows(), scratch.flash_splits());
    let (ml, po, plan) = scratch.flash_buffers();
    let splits = splits.max(1);
    assert!(f_splits >= splits, "nemotron_attn_prefill: a launch for {splits} splits, the scratch holds {f_splits} a row");
    // a call of more rows than the scratch holds goes in passes of its rows; a row's bits do not depend on the pass
    let pass_rows = rows.min(rows_cap).max(1);
    let launch = match kv.dtype {
        tree::KvDtype::F16 => ffi::gb_nemo_flash_prefill,
        tree::KvDtype::Int8 => ffi::gb_nemo_flash_prefill_i8,
    };
    let mut done = 0;
    while done < rows {
        let n = pass_rows.min(rows - done);
        let r0 = first + done;
        // SAFETY: the plan buffer holds a count and a pair per row of a pass, the partial buffers `rows_cap x f_splits` groups, and the
        // other pointers are the callers' device buffers offset to the pass's first row
        CudaError::check(unsafe {
            launch(
                q.ptr_at(r0 * Q_ROW_ELEMS * 4),
                kv.k.ptr(),
                kv.v.ptr(),
                table.buf.ptr(),
                i(table.max_blocks)?,
                idx.slot.ptr_at(r0 * 4),
                idx.pos.ptr_at(r0 * 4),
                plan.ptr(),
                i(n)?,
                i(from_pos)?,
                i(row_blocks.min(n).max(1))?,
                i(splits)?,
                i(splits)?,
                ml.ptr(),
                po.ptr(),
                out.ptr_at(r0 * Q_ROW_ELEMS * 4),
                st.raw(),
            )
        })?;
        done += n;
    }
    Ok(())
}

/// Attention for any call, by the one rule: a row at absolute position `>= FLASH_FROM_POS` is served by the flash kernel and every
/// other row by the exact tree, whatever the call is (decode, verify window or prefill slice, any rows of any sequences).
///
/// A call whose launch is built for at most [`TREE_CHUNKS_BELOW`] chunks (`idx.max_chunks()`: the deepest row's, or a captured
/// graph's bucket) cannot hold a deep row and is exactly the call [`tree::attn_decode`] or [`tree::attn_prefill`] would have made
/// (the same kernels, the same launch). Above it both are launched for every row: the tree skips the rows at and above the threshold
/// (its grid is built for [`TREE_CHUNKS_BELOW`] chunks), the flash path skips those below, and which is which is read from the
/// device's position array, so the launch list is a function of `(rows, n_seq, idx.max_chunks())` only and a CUDA graph of it serves
/// every call of that shape and bucket. `n_seq` is the number of sequences of the call (a bound for its row blocks; `0` unknown).
#[allow(clippy::too_many_arguments)]
pub fn attn_long(
    q: &DevBuf,
    kv: &NemoKv,
    table: &BlockTable,
    idx: &RowIndex,
    n_seq: usize,
    scratch: &mut Scratch,
    out: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    let max_chunks = idx.max_chunks();
    // a call of more rows than the scratch holds (a prefill slice of up to 256 rows against a window's 64) is run in passes of its rows;
    // a call that fits is one pass, the launch of before
    let cap = scratch.rows().max(1);
    // Lane S3, contract P2 (prefill only: opted in, inside a prefill scope, any width): every row on the flash kernel, the rows
    // below FLASH_FROM_POS as well (the tree runs them at about 5 TFLOP/s, the flash kernel at about 60). A row's bits then depend on
    // its position, K and V only (the flash kernel's own invariance), but differ from the decode path's tree below 8,192: accepted
    // under P2, which changes the prefill bits anyway. Needs a scratch with flash buffers; otherwise the rule below.
    if crate::nemotron_dense_mma::active(idx.rows) {
        let splits = flash_splits(max_chunks.max(1));
        if scratch.flash_splits() >= splits {
            let mut a = 0;
            while a < idx.rows {
                let n = cap.min(idx.rows - a);
                attn_flash(q, kv, table, idx, a, n, 0, splits, max_row_blocks(n, n_seq), scratch, out, st)?;
                a += n;
            }
            return Ok(());
        }
    }
    if max_chunks <= TREE_CHUNKS_BELOW {
        let mut a = 0;
        while a < idx.rows {
            let n = cap.min(idx.rows - a);
            tree::attn_range(q, kv, table, idx, a, n, max_chunks, i32::MAX, scratch, out, st)?;
            a += n;
        }
        return Ok(());
    }
    let splits = flash_splits(max_chunks);
    let mut a = 0;
    while a < idx.rows {
        let n = cap.min(idx.rows - a);
        let tree_chunks = TREE_CHUNKS_BELOW.min(scratch.max_chunks());
        tree::attn_range(q, kv, table, idx, a, n, tree_chunks, FLASH_FROM_POS as i32, scratch, out, st)?;
        attn_flash(q, kv, table, idx, a, n, FLASH_FROM_POS, splits, max_row_blocks(n, n_seq), scratch, out, st)?;
        a += n;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nemotron_attn::CHUNK_KEYS;

    #[test]
    fn the_threshold_is_a_split_and_a_chunk_boundary_and_a_row_at_it_is_flash() {
        assert_eq!(FLASH_FROM_POS % FLASH_CHUNK_KEYS, 0);
        assert_eq!(FLASH_FROM_POS % CHUNK_KEYS, 0);
        assert_eq!(TREE_CHUNKS_BELOW, 16);
        assert_eq!(FLASH_CHUNK_KEYS % FLASH_TILE_KEYS, 0);
        assert_eq!(256 % FLASH_TILE_KEYS, 0, "a tile never straddles a page");
        assert!(!uses_flash(FLASH_FROM_POS - 1));
        assert!(uses_flash(FLASH_FROM_POS));
    }

    #[test]
    fn the_serving_scratch_holds_sixteen_tree_chunks_whatever_the_context_and_a_split_a_row_per_8192_keys() {
        use crate::nemotron_attn::GROUP_BYTES;
        assert_eq!(GROUP_BYTES, 32 * 520);
        // 64 rows (a window's) at 1M tokens (2,048 chunks): 16 chunks of tree, 128 splits of flash, the row plan
        let (tree, flash) = (64 * 16 * GROUP_BYTES, 64 * 128 * GROUP_BYTES + 65 * 8);
        assert_eq!(Scratch::serving_bytes(64, 2048), tree + flash);
        assert_eq!(Scratch::serving_bytes(64, 2048), 153_354_760);
        // the same rows with the tree sized for the context (`Scratch::new`, for direct use of the tree's entry points) are 2.18 GB
        assert_eq!(Scratch::tree_bytes(64, 2048), 2_181_038_080);
        assert_eq!(Scratch::bytes_for(64, 2048), 2_181_038_080 + flash);
        // a context of at most 8,192 tokens has at most 16 chunks and one split
        assert_eq!(Scratch::serving_bytes(64, 16), 64 * 16 * GROUP_BYTES + 64 * GROUP_BYTES + 65 * 8);
        assert_eq!(Scratch::serving_bytes(64, 5), 64 * 5 * GROUP_BYTES + 64 * GROUP_BYTES + 65 * 8);
        // the tree part does not grow with the context, the flash part is linear in it
        let t = |c| Scratch::serving_bytes(256, c);
        assert_eq!(t(2048) - t(1024), 256 * 64 * GROUP_BYTES);
        assert_eq!(t(1024) - t(512), 256 * 32 * GROUP_BYTES);
    }

    #[test]
    fn the_flash_splits_follow_the_trees_chunk_bucket() {
        // a row at the last position below the threshold has 16 chunks: no flash launch is made for it; the first row above has 17
        assert_eq!((8191 + 1usize).div_ceil(CHUNK_KEYS), TREE_CHUNKS_BELOW);
        assert_eq!((8192 + 1usize).div_ceil(CHUNK_KEYS), TREE_CHUNKS_BELOW + 1);
        // a row at position p needs p / 8192 + 1 splits; a launch for its chunk count has exactly that many
        for p in [8192usize, 8193, 12_287, 16_383, 16_384, 100_000, 524_287, 524_288, 1_048_575] {
            let chunks = (p + 1).div_ceil(CHUNK_KEYS);
            assert_eq!(flash_splits(chunks), p / FLASH_CHUNK_KEYS + 1, "position {p}");
        }
        // a power-of-two bucket (a captured graph) never has fewer splits than any row of it needs
        for bucket in [32usize, 64, 128, 256, 512, 1024, 2048] {
            for real in (bucket / 2 + 1)..=bucket {
                assert!(flash_splits(bucket) >= flash_splits(real));
            }
            assert_eq!(flash_splits(bucket), bucket / 16);
        }
    }

    #[test]
    fn the_row_block_bound_covers_every_way_to_split_rows_into_sequences() {
        // brute force: every composition of `rows` into `seqs` runs gives at most max_row_blocks blocks
        fn blocks(lens: &[usize]) -> usize {
            lens.iter().map(|l| l.div_ceil(FLASH_ROWS_PER_CTA)).sum()
        }
        fn walk(rows: usize, seqs: usize, acc: &mut Vec<usize>, worst: &mut usize) {
            if seqs == 1 {
                acc.push(rows);
                *worst = (*worst).max(blocks(acc));
                acc.pop();
                return;
            }
            for l in 1..=rows - (seqs - 1) {
                acc.push(l);
                walk(rows - l, seqs - 1, acc, worst);
                acc.pop();
            }
        }
        for rows in 1..=24usize {
            for seqs in 1..=rows.min(5) {
                let mut worst = 0;
                walk(rows, seqs, &mut Vec::new(), &mut worst);
                assert!(
                    max_row_blocks(rows, seqs) >= worst,
                    "rows {rows}, sequences {seqs}: bound {} < {worst}",
                    max_row_blocks(rows, seqs)
                );
                assert!(max_row_blocks(rows, seqs) <= rows);
            }
            assert_eq!(max_row_blocks(rows, 0), rows);
        }
        assert_eq!(max_row_blocks(1, 1), 1);
        assert_eq!(max_row_blocks(8, 1), 1);
        assert_eq!(max_row_blocks(9, 1), 2);
        assert_eq!(max_row_blocks(64, 1), 8);
    }
}
