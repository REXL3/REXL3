//! Device-side greedy verification gather for Flash-Next MTP (`csrc/qfn_verify.cu`).
//!
//! The verify window already computes the exact per-row top-`count` of the target LM head
//! (`glue::topk`, sorted by (value DESC, id ASC)). The host greedy accept only ever looks at the
//! winner of each row, so a greedy round can read back just the packed argmax column instead of the
//! whole `[rows, count]` candidate table. This module is that pack: one tiny kernel copies column 0
//! of `ids` / `vals` into dense per-row arrays (a row whose candidates contain a NaN is reported as
//! NaN, mirroring `gb_sampling::try_choose`'s `NanLogit` refusal exactly), plus the pure-Rust mirror
//! [`greedy_gather_host`] the tests pin it against.
//!
//! This only *reduces the readback*: it does not change what is read. The source buffers are the same
//! ones the host path downloads, so a greedy stream fed the gathered rows produces bit-identical
//! candidates to the full download (column 0 IS the argmax, ties by lowest id, per `glue::topk`).

use crate::rt::{CudaError, DevBuf, Stream};

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        /// `out_ids[r] = ids[r * count]`, `out_vals[r] = vals[r * count]` (NaN when row `r` has any
        /// NaN candidate). One thread per row.
        pub fn gb_qfn_greedy_gather(
            ids: *const c_void,
            vals: *const c_void,
            count: i32,
            rows: i32,
            out_ids: *mut c_void,
            out_vals: *mut c_void,
            stream: *mut c_void,
        ) -> i32;
    }
}

/// Pack the argmax column of `ids` / `vals` (`[rows, count]` each, as produced by `glue::topk`) into
/// the dense per-row buffers `out_ids` / `out_vals` (each at least `[rows]`). A row with any NaN candidate
/// gets a NaN value (its id is still copied), which is exactly what the host greedy path would see and fail
/// on. `count` must be `1..=64`; `rows` is the LOGICAL number of rows of the window (the live rows the
/// top-k filled), not the capacity of the output buffers — the kernel touches exactly `rows` rows, so
/// buffer space past row `rows` is never read or written. It is the caller's job to sync `st` before
/// reading `out_ids` / `out_vals` back (they are written asynchronously).
pub unsafe fn greedy_gather(
    ids: &DevBuf,
    vals: &DevBuf,
    count: usize,
    rows: usize,
    out_ids: &mut DevBuf,
    out_vals: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    if count == 0 || count > 64 {
        return Err(CudaError(1));
    }
    if rows == 0 {
        return Ok(());
    }
    if out_ids.bytes() / 4 < rows || out_vals.bytes() / 4 < rows {
        return Err(CudaError(1));
    }
    if ids.bytes() < rows * count * 4 || vals.bytes() < rows * count * 4 {
        return Err(CudaError(1));
    }
    CudaError::check(unsafe {
        ffi::gb_qfn_greedy_gather(ids.ptr(), vals.ptr(), count as i32, rows as i32, out_ids.ptr(), out_vals.ptr(), st.raw())
    })
}

/// CPU mirror of [`greedy_gather`]: column 0 of every row of a (value DESC, id ASC) candidate table,
/// NaN-propagating exactly like the kernel. `ids` / `vals` are `[rows * count]`.
pub fn greedy_gather_host(ids: &[i32], vals: &[f32], count: usize) -> (Vec<i32>, Vec<f32>) {
    assert!(count > 0, "count is zero");
    assert_eq!(ids.len(), vals.len(), "ids and vals differ in length");
    assert!(ids.len() % count == 0, "ids length is not a multiple of count");
    let rows = ids.len() / count;
    let mut out_ids = Vec::with_capacity(rows);
    let mut out_vals = Vec::with_capacity(rows);
    for r in 0..rows {
        let base = r * count;
        out_ids.push(ids[base]);
        if vals[base..base + count].iter().any(|v| v.is_nan()) {
            out_vals.push(f32::NAN);
        } else {
            out_vals.push(vals[base]);
        }
    }
    (out_ids, out_vals)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gather is the identity on the argmax column of a sorted candidate table, NaN row included.
    #[test]
    fn gather_matches_the_argmax_column_of_sorted_candidates() {
        // rows of 4 candidates, sorted by (value DESC, id ASC); column 0 is the winner
        let ids = vec![7, 1, 2, 3, 100, 0, 9, 8, 42, 41, 40, 39];
        let vals = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -1.0, -2.0];
        let (oid, oval) = greedy_gather_host(&ids, &vals, 4);
        assert_eq!(oid, vec![7, 100, 42]);
        assert_eq!(oval, vec![0.9, 0.5, 0.1]);
    }

    /// A NaN anywhere in a row poisons the row's value (the id is still the first column's), matching
    /// `try_choose`'s `NanLogit` refusal. A plain -inf is NOT a NaN and passes through.
    #[test]
    fn nan_rows_are_flagged_and_inf_rows_are_not() {
        let ids = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let vals = vec![0.5, f32::NAN, 0.4, 0.3, 0.2, 0.1, f32::NEG_INFINITY, -1.0, 0.0, -1.0, -2.0, -3.0];
        let (oid, oval) = greedy_gather_host(&ids, &vals, 4);
        assert_eq!(oid, vec![1, 5, 9]);
        assert!(oval[0].is_nan(), "row with a NaN candidate must report NaN");
        assert_eq!(oval[1], 0.2, "-inf is a valid value, not a NaN");
        assert_eq!(oval[2], 0.0);
    }

    /// A single candidate per row degenerates to plain copies.
    #[test]
    fn count_one_is_a_plain_copy() {
        let ids = vec![11, 22, 33];
        let vals = vec![1.0, 2.0, 3.0];
        let (oid, oval) = greedy_gather_host(&ids, &vals, 1);
        assert_eq!(oid, ids);
        assert_eq!(oval, vals);
    }

    /// Ties are already resolved by the topk sort (lowest id first); the gather must not re-argmax.
    #[test]
    fn ties_are_taken_from_the_sort_order_not_rescanned() {
        // two rows whose columns 0 and 1 tie: column 0 stays the winner whichever id is first
        let ids = vec![5, 3, 1, 0, 9, 2, 2, 9];
        let vals = vec![0.5, 0.5, 0.4, 0.3, 0.7, 0.7, 0.1, 0.0];
        let (oid, _) = greedy_gather_host(&ids, &vals, 4);
        assert_eq!(oid, vec![5, 9]);
    }

    // ------------------------------------------------------------------ GPU-vs-host differential
    //
    // Run the REAL kernel (`gb_qfn_greedy_gather`) against the host mirror `greedy_gather_host` over the
    // shapes the verify window actually produces, including the observed production shape (rows=3,
    // candidates_per_row=28, see reports/qwen-perf-recovery/device-verify-ab.md). These tests are gated
    // on a live CUDA device: they are skipped by the stub build and on GPU-less hosts, and run on the GB10
    // campaign host (a `#[ignore]` fence would keep them out of `cargo test --workspace` on the GPU box).
    //
    // Non-vacuous: the kernel must treat `rows` as the LOGICAL number of live rows, not the capacity of the
    // output buffers. The original wrapper derived rows from the output-buffer size, so the model's
    // capacity-sized dv buffers made it gather 64 rows from a table with only 3 live rows — reading stale
    // previous-window rows and writing them into `out[3..64]`. `gather_never_touches_rows_past_the_logical_count`
    // pins that contract: before the fix it failed (the tail came back clobbered), after the fix the tail
    // keeps its sentinel.

    /// A deterministic pseudo-random table of `rows` x `count` for the shape (rows, count).
    fn table(rows: usize, count: usize, sorted: bool) -> (Vec<i32>, Vec<f32>) {
        let mut state = (rows * 131 + count * 17 + 7) as u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut ids = Vec::with_capacity(rows * count);
        let mut vals = Vec::with_capacity(rows * count);
        for _ in 0..rows {
            let mut row: Vec<(i32, f32)> = (0..count)
                .map(|c| {
                    let id = (next() % 1000) as i32;
                    // a logit in [-4, 4); sprinkle -inf, +0/-0 and (on a few rows) a NaN
                    let raw = (next() % 131072) as f32 / 16384.0 - 4.0;
                    let v = match (c, next() % 17) {
                        (0, 0) => f32::NAN,
                        (_, 1) => f32::NEG_INFINITY,
                        (_, 2) => -0.0,
                        (_, 3) => 0.0,
                        _ => raw,
                    };
                    (id, v)
                })
                .collect();
            if sorted {
                // the real topk order: (value DESC, id ASC), NaN canonicalized to -inf
                row.sort_by(|a, b| {
                    let av = a.1;
                    let bv = b.1;
                    let an = av.is_nan();
                    let bn = bv.is_nan();
                    let (a0, b0) = (if an { f32::NEG_INFINITY } else { av }, if bn { f32::NEG_INFINITY } else { bv });
                    b0.total_cmp(&a0).then(a.0.cmp(&b.0))
                });
            }
            for (id, v) in row {
                ids.push(id);
                vals.push(v);
            }
        }
        (ids, vals)
    }

    /// Run the kernel with the wrapper for `(rows, count)` and return the full output buffers (capacity 64).
    fn gather_on_device(ids: &[i32], vals: &[f32], count: usize, rows: usize) -> (Vec<i32>, Vec<f32>) {
        let st = crate::rt::Stream::new_nonblocking().expect("stream");
        let ids_dev = crate::rt::DevBuf::from_slice(ids).expect("ids");
        let vals_dev = crate::rt::DevBuf::from_slice(vals).expect("vals");
        let mut out_ids = crate::rt::DevBuf::alloc(64 * 4).expect("out_ids");
        let mut out_vals = crate::rt::DevBuf::alloc(64 * 4).expect("out_vals");
        // Zero the outputs on the LAUNCH stream. `DevBuf::zeroed` runs `cudaMemset` on the legacy
        // default stream, which does not synchronize with a non-blocking stream: under GPU contention
        // the memset can complete after the kernel writes (wiping the results). A stream-ordered
        // memset closes that race, and the inputs (`from_slice`) are safe because a pageable `cudaMemcpy`
        // is synchronous and completes before the launch.
        out_ids.memset_async(0, &st).expect("zero ids");
        out_vals.memset_async(0, &st).expect("zero vals");
        unsafe { greedy_gather(&ids_dev, &vals_dev, count, rows, &mut out_ids, &mut out_vals, &st) }.expect("gather");
        st.sync().expect("sync");
        (out_ids.to_vec().expect("ids back"), out_vals.to_vec().expect("vals back"))
    }

    /// The mirror's value for row `r` (NaN-aware compare: CUDA's `NAN` is a different payload than Rust's).
    fn same_value(got: f32, exp: f32) -> bool {
        (got.is_nan() && exp.is_nan()) || got.to_bits() == exp.to_bits()
    }

    /// Every shape the verify window produces (observed production shape rows=3, count=28 included), sorted
    /// exactly like `glue::topk` (value DESC, id ASC): the kernel must return the mirror's column 0.
    #[test]
    fn kernel_matches_the_host_mirror_on_the_topk_layout() {
        if !crate::rt::device_available() {
            eprintln!("skip: no CUDA device");
            return;
        }
        for (rows, count) in [(1, 1), (1, 28), (3, 28), (28, 28), (64, 64), (64, 1), (64, 28)] {
            let (ids, vals) = table(rows, count, true);
            let (exp_ids, exp_vals) = greedy_gather_host(&ids, &vals, count);
            let (got_ids, got_vals) = gather_on_device(&ids, &vals, count, rows);
            assert_eq!(got_ids[..rows], exp_ids[..], "rows={rows} count={count}: ids");
            for r in 0..rows {
                assert!(
                    same_value(got_vals[r], exp_vals[r]),
                    "rows={rows} count={count}: row {r} value {} != {}",
                    got_vals[r],
                    exp_vals[r]
                );
            }
        }
    }

    /// Column 0 of an UNSORTED table is taken literally (the gather packs, it does not re-sort): the kernel
    /// must agree with the mirror even when the table is not in topk order.
    #[test]
    fn kernel_takes_column_zero_of_unsorted_ids_literally_like_the_mirror() {
        if !crate::rt::device_available() {
            eprintln!("skip: no CUDA device");
            return;
        }
        let (rows, count) = (3usize, 28usize);
        let (ids, vals) = table(rows, count, false);
        let (exp_ids, exp_vals) = greedy_gather_host(&ids, &vals, count);
        let (got_ids, got_vals) = gather_on_device(&ids, &vals, count, rows);
        assert_eq!(got_ids[..rows], exp_ids[..], "unsorted ids");
        for r in 0..rows {
            assert!(same_value(got_vals[r], exp_vals[r]), "unsorted row {r} value {} != {}", got_vals[r], exp_vals[r]);
        }
    }

    /// The kernel must touch exactly `rows` live rows: buffer space past them keeps its sentinel. Before the
    /// row-count fix the wrapper derived rows from the output-buffer capacity (64), so this test was red —
    /// the tail came back clobbered with stale table rows.
    #[test]
    fn gather_never_touches_rows_past_the_logical_count() {
        if !crate::rt::device_available() {
            eprintln!("skip: no CUDA device");
            return;
        }
        let (rows, count) = (3usize, 28usize);
        let (ids, vals) = table(rows, count, true);
        let st = crate::rt::Stream::new_nonblocking().expect("stream");
        let ids_dev = crate::rt::DevBuf::from_slice(&ids).expect("ids");
        let vals_dev = crate::rt::DevBuf::from_slice(&vals).expect("vals");
        let sentinel = 0x7F7F_7F7Fi32;
        let mut out_ids = vec![sentinel; 64];
        let mut out_vals = vec![sentinel as f32; 64];
        let mut out_ids_dev = crate::rt::DevBuf::zeroed(64 * 4).expect("out_ids");
        let mut out_vals_dev = crate::rt::DevBuf::zeroed(64 * 4).expect("out_vals");
        out_ids_dev.upload(&out_ids).expect("upload ids");
        out_vals_dev.upload(&out_vals).expect("upload vals");
        unsafe { greedy_gather(&ids_dev, &vals_dev, count, rows, &mut out_ids_dev, &mut out_vals_dev, &st) }.expect("gather");
        st.sync().expect("sync");
        out_ids_dev.download(&mut out_ids).expect("download ids");
        out_vals_dev.download(&mut out_vals).expect("download vals");
        // the live slice equals the mirror
        let (exp_ids, exp_vals) = greedy_gather_host(&ids, &vals, count);
        assert_eq!(out_ids[..rows], exp_ids[..], "live ids");
        for r in 0..rows {
            assert_eq!(out_vals[r].to_bits(), exp_vals[r].to_bits(), "live row {r} value bits");
        }
        // everything past the logical rows is untouched
        assert!(out_ids[rows..].iter().all(|&x| x == sentinel), "tail ids clobbered: {:?}", &out_ids[rows..]);
        assert!(out_vals[rows..].iter().all(|&x| x.to_bits() == (sentinel as f32).to_bits()), "tail vals clobbered");
    }
}
