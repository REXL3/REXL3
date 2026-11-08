//! Kernel-side pieces of the two-rank tensor-parallel Nemotron engine (`csrc/nemotron_tp.cu`; design in `docs/NEMOTRON_TP.md`).
//!
//! The rule of that design is that every output element is computed by exactly one rank with the single-node kernel, and the other rank
//! receives it through a sum over buffers of disjoint support (a rank holds its own elements and exact `-0.0` elsewhere, the true additive
//! identity: `x + (-0.0) == x` for every `x`, the sign of a zero included, where `x + (+0.0)` would turn a `-0.0` into `+0.0` and the step
//! digest, which hashes raw bits, would see it). The functions here build such buffers. They do no arithmetic: a bit-exact copy and the
//! identity, and an element's value
//! depends on its own (row, column) only, so every function is row invariant by construction.
//!
//! The MoE expert-parallel hook is not here: it is `NemotronMoe::forward_in_tp` and `PrefillMoe::forward_prefill_in_tp`
//! (`nemotron_moe.rs`, `nemotron_moe_prefill.rs`), which call a caller-supplied reduce between the routed down projection and the combine.

//!
//! # Cost of the expert-parallel `y` reduce (per MoE layer; 40 layers a step)
//!
//! The buffer the hook reduces is f32 `[rows, 22, 1024]`: 90,112 bytes a row. Per layer: 1 row 90 KB, 5 rows (a verify window) 450 KB,
//! 22 rows 2.0 MB, 64 rows 5.8 MB, 256 rows (a prefill slice) 23 MB. Per step (x 40): 18 MB at 5 rows, 231 MB at 64, 923 MB at 256.
//! Against the fabric numbers of `docs/NEMOTRON_TP.md` (256 KB in 52 to 62 us, 8 MB at 23 GB/s bus) a 5-row reduce is latency bound
//! (about 60 to 80 us a layer, 2.5 to 3 ms a step, projected) and a 64-row one bandwidth bound (about 250 us a layer, 10 ms a step).
//!
//! Leaner layouts that keep every bit (no arithmetic ever touches a value, and the combine keeps its slot order):
//!
//! * Reduce-scatter instead of allreduce: lay `y` out column-half major (`[2][rows][22][512]`, a relayout in the down projection's
//!   output side or one extra launch), reduce-scatter it so that each rank receives the summed half of the columns, run the combine on
//!   its 512 columns (every element is still the 22-slot sum in slot order, computed by one rank), and all-gather `routed`
//!   (`rows * 1024` floats). Bytes sent per rank: 45 KB + 2 KB a row against 90 KB: 1.9x fewer. Needs `reduce_scatter` and
//!   `all_gather` on the `TpComm` trait (TP-C) and a `combine` over a column range (a new launcher, not in `nemotron_moe.cu` today).
//! * Compacting the owned slots (each rank sends only its pairs, about 11 of 22 on average, positions known on both sides because the
//!   router is replicated): another 2x on average, but the owned count is routing dependent and lives on the device, so a fixed-capacity
//!   buffer (the only kind a CUDA graph can carry) must hold all 22 slots for the skewed case: no saving without a conditional graph
//!   node or a host sync. Not worth it.
//!
//! Measured, the block itself (node 4, 2026-10-11, graph of 10 consecutive blocks, natural routing, 4-bit experts, per block):
//! the whole block `forward_in` takes 422 / 1251 / 3374 / 5322 us at 1 / 5 / 22 / 64 rows and a rank's block (half of the experts absent,
//! no-op hook) 320 / 748 / 1980 / 3158 us: 0.76x at 1 row and 0.59 to 0.60x from 5 rows up, so the absent experts' extra output-side
//! launches cost little (a variant that zeroed `y` with one launch and skipped them was built and measured: 336 / 768 / 1957 / 3102 us,
//! slower at 1 to 5 rows and 1 to 2 percent faster at 22 to 64, so it was not kept). The large-row prefill path (`forward_prefill_in`,
//! whole / rank block, per block): 5282 / 3431 us at 64 rows, 6889 / 4805 us at 128, 10214 / 7470 us at 256 (0.73x: the shared expert,
//! fc1, fc2 and the router are replicated, and the rank's 256 absent experts run through the slow class, which writes their zeros), against
//! 4 x 3145 = 12580 us for a 256-row slice cut into four 64-row decode-path calls on a rank.
//!
//! Neither pays at 1 to 5 rows (latency bound, 60 to 80 us either way). Reduce-scatter is worth about half of the reduce time from
//! roughly 22 rows up (5 ms a step at 64 rows, 20 ms at 256): revisit it for the aggregate arm and for prefill under TP, not for the
//! single-stream 75 tok/s target.

use crate::{CudaError, DevBuf, Stream};

/// What a call outside its contract returns, without launching: `cudaErrorInvalidValue`.
const INVALID: CudaError = CudaError(1);

mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_nemo_tp_scatter_cols_f32(src: *const c_void, dst: *mut c_void, rows: i32, n_local: i32, n_total: i32, col_off: i32, s: *mut c_void) -> i32;
        pub fn gb_nemo_tp_pairs(cnt: *const c_void, lists: *const c_void, lcap: i32, lo: i32, hi: i32, n: i32, y: *mut c_void, packed: *mut c_void, to_packed: i32, s: *mut c_void) -> i32;
        pub fn gb_nemo_tp_place_split(own: *const c_void, nl: i32, off: i32, peer: *const c_void, pnl: i32, poff: i32, bend: i32, z: *mut c_void, xbc: *mut c_void, dt: *mut c_void, rows: i32, s: *mut c_void) -> i32;
        pub fn gb_nemo_tp_pair_map(cnt: *const c_void, lists: *const c_void, lcap: i32, lo: i32, hi: i32, map: *mut c_void, s: *mut c_void) -> i32;
        pub fn gb_nemo_tp_combine_gather(y: *const c_void, recv: *const c_void, map: *const c_void, wts: *const c_void, rows: i32, n: i32, routed: *mut c_void, s: *mut c_void) -> i32;
        pub fn gb_nemo_tp_split_hg(src: *const c_void, rows: i32, nz: i32, nx: i32, nbc: i32, zo: i32, xo: i32, bo: i32, z: *mut c_void, xbc: *mut c_void, dt: *mut c_void, s: *mut c_void) -> i32;
        pub fn gb_nemo_tp_pack_cols(src: *const c_void, rows: i32, nl: i32, off: i32, bend: i32, out: *mut c_void, s: *mut c_void) -> i32;
        pub fn gb_nemo_tp_place_cols(own: *const c_void, nl: i32, off: i32, peer: *const c_void, pnl: i32, poff: i32, bend: i32, dst: *mut c_void, rows: i32, nt: i32, s: *mut c_void) -> i32;
    }
}

/// Panics when `b` holds fewer than `bytes` bytes (a short buffer must not fault on the device).
fn need(b: &DevBuf, bytes: usize, what: &str) {
    assert!(b.bytes() >= bytes, "nemotron_tp {what}: buffer of {} bytes, need {bytes}", b.bytes());
}

fn i(v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| INVALID)
}

/// Writes `src` (f32 `[rows, n_local]`) into the columns `[col_off, col_off + n_local)` of `dst` (f32 `[rows, n_total]`) and exact `-0.0`
/// into every other column of every row, in one fused launch on `st`. A bit-exact copy (NaN payloads and signed zeros included): the
/// result is the disjoint-support buffer a following `allreduce_sum_f32` turns into the gathered row.
///
/// `src` and `dst` must be different, non-overlapping buffers. Needs `col_off + n_local <= n_total`; `n_local == 0` is a pure zero
/// fill. Runs vectorised (float4) when the widths, the offset and both base addresses are multiples of 4 elements / 16 bytes (always
/// the case for the Nemotron splits, which are multiples of 128 columns), and element-wise otherwise; the bits are the same.
/// `rows == 0` is a no-op.
pub fn scatter_cols_f32(
    src: &DevBuf,
    dst: &mut DevBuf,
    rows: usize,
    n_local: usize,
    n_total: usize,
    col_off: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    if n_total == 0 || col_off.checked_add(n_local).is_none_or(|e| e > n_total) {
        return Err(INVALID);
    }
    need(src, rows * n_local * 4, "scatter_cols_f32 src");
    need(dst, rows * n_total * 4, "scatter_cols_f32 dst");
    CudaError::check(unsafe {
        ffi::gb_nemo_tp_scatter_cols_f32(src.ptr(), dst.ptr(), i(rows)?, i(n_local)?, i(n_total)?, i(col_off)?, st.raw())
    })
}

/// Bytes of one packed row of the columns `[off, off + nl)` (lane S, contract P2): the columns below `bend` as bf16, the rest as f32.
pub fn packed_row_bytes(nl: usize, off: usize, bend: usize) -> usize {
    let nb = bend.saturating_sub(off).min(nl);
    nb * 2 + (nl - nb) * 4
}

/// Packs `src` (f32 `[rows, nl]`, the columns `[off, off + nl)` of a wider row) for the peer of a two-rank exchange: the columns below
/// `bend` rounded to bf16 (round to nearest even: use it ONLY for columns whose every consumer rounds to bf16 itself), the others as f32
/// bit-exact. `out` gets `rows x packed_row_bytes(nl, off, bend)` bytes.
pub fn pack_cols(
    src: &DevBuf,
    rows: usize,
    nl: usize,
    off: usize,
    bend: usize,
    out: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    need(src, rows * nl * 4, "pack_cols src");
    need(out, rows * packed_row_bytes(nl, off, bend), "pack_cols out");
    CudaError::check(unsafe { ffi::gb_nemo_tp_pack_cols(src.ptr(), i(rows)?, i(nl)?, i(off)?, i(bend)?, out.ptr(), st.raw()) })
}

/// `dst` (f32 `[rows, nt]`): this rank's columns `[off, off + nl)` from `own` (f32 `[rows, nl]`, bit-exact), the peer's `[poff, poff +
/// pnl)` from its packed rows (bf16 widened exactly below `bend`, f32 from it). The two ranges must cover the row (two ranks).
#[allow(clippy::too_many_arguments)]
pub fn place_cols(
    own: &DevBuf,
    nl: usize,
    off: usize,
    peer: &DevBuf,
    pnl: usize,
    poff: usize,
    bend: usize,
    dst: &mut DevBuf,
    rows: usize,
    nt: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    if off + nl > nt || poff + pnl > nt || nl + pnl != nt {
        return Err(INVALID);
    }
    need(own, rows * nl * 4, "place_cols own");
    need(peer, rows * packed_row_bytes(pnl, poff, bend), "place_cols peer");
    need(dst, rows * nt * 4, "place_cols dst");
    CudaError::check(unsafe {
        ffi::gb_nemo_tp_place_cols(
            own.ptr(),
            i(nl)?,
            i(off)?,
            peer.ptr(),
            i(pnl)?,
            i(poff)?,
            i(bend)?,
            dst.ptr(),
            i(rows)?,
            i(nt)?,
            st.raw(),
        )
    })
}

/// The routed pairs of the experts `[lo, hi)` (lane S, contract P2): `to_packed` copies their `y` rows (`n` floats each) into `packed`,
/// in expert order then list order; otherwise copies `packed` back into their rows. `cnt` (i32 [512]) and `lists` (i32 [512 * lcap]) are
/// device addresses of the prefill path's bucket outputs (`PrefillScratch::route_ptrs`). A bit-exact copy.
///
/// # Safety
/// `cnt` and `lists` must be live device buffers of those shapes, and every listed pair a row of `y`; `packed` must hold the range's pairs.
#[allow(clippy::too_many_arguments)]
pub unsafe fn route_pairs(
    cnt: usize,
    lists: usize,
    lcap: usize,
    lo: usize,
    hi: usize,
    n: usize,
    y: &mut DevBuf,
    packed: &mut DevBuf,
    to_packed: bool,
    st: &Stream,
) -> Result<(), CudaError> {
    CudaError::check(ffi::gb_nemo_tp_pairs(
        cnt as *const core::ffi::c_void,
        lists as *const core::ffi::c_void,
        i(lcap)?,
        i(lo)?,
        i(hi)?,
        i(n)?,
        y.ptr(),
        packed.ptr(),
        to_packed as i32,
        st.raw(),
    ))
}

/// `map` (i32 `[pairs]`): -1 for every pair (0xFF fill), then for the experts `[lo, hi)` (the peer's) the row of the packed buffer that
/// carries each pair, in [`route_pairs`]'s order.
///
/// # Safety
/// As [`route_pairs`]; `map` holds the call's `rows x 22` pairs.
pub unsafe fn pair_map(
    cnt: usize,
    lists: usize,
    lcap: usize,
    lo: usize,
    hi: usize,
    map: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    map.memset_async(0xFF, st)?;
    CudaError::check(ffi::gb_nemo_tp_pair_map(
        cnt as *const core::ffi::c_void,
        lists as *const core::ffi::c_void,
        i(lcap)?,
        i(lo)?,
        i(hi)?,
        map.ptr(),
        st.raw(),
    ))
}

/// The routed combine (`nemotron_moe::gb_nm_combine`'s arithmetic) reading each slot's row from `y` (map < 0) or from `recv` (the
/// peer's gathered rows, row `map[pair]`): no unpack copy. Device addresses: `y`, `recv` `[.., n]` f32, `map` i32, `wts` f32 `[rows, 22]`,
/// `routed` f32 `[rows, n]`.
///
/// # Safety
/// Every address must be a live device buffer of that shape.
#[allow(clippy::too_many_arguments)]
pub unsafe fn combine_gather(
    y: usize,
    recv: usize,
    map: usize,
    wts: usize,
    rows: usize,
    n: usize,
    routed: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    use core::ffi::c_void;
    CudaError::check(ffi::gb_nemo_tp_combine_gather(
        y as *const c_void,
        recv as *const c_void,
        map as *const c_void,
        wts as *const c_void,
        i(rows)?,
        i(n)?,
        routed as *mut c_void,
        st.raw(),
    ))
}

/// Downloads `out.len()` i32 from the device address `src` (synchronous).
///
/// # Safety
/// `src` must address at least `out.len()` live i32 on the device.
pub unsafe fn download_i32_raw(src: usize, out: &mut [i32]) -> Result<(), CudaError> {
    crate::rt::download_raw(src, out.as_mut_ptr() as *mut core::ffi::c_void, std::mem::size_of_val(out))
}

/// [`place_cols`] fused with the in_proj split (`nemotron_glue::split_inproj`): z `[rows, 8192]` (rounded to bf16 as the split does),
/// xBC `[rows, 10240]` and dt `[rows, 128]` written straight from this rank's columns and the peer's packed rows. The same bits as
/// `place_cols` then `split_inproj`.
#[allow(clippy::too_many_arguments)]
pub fn place_split(
    own: &DevBuf,
    nl: usize,
    off: usize,
    peer: &DevBuf,
    pnl: usize,
    poff: usize,
    bend: usize,
    z: &mut DevBuf,
    xbc: &mut DevBuf,
    dt: &mut DevBuf,
    rows: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    need(own, rows * nl * 4, "place_split own");
    need(peer, rows * packed_row_bytes(pnl, poff, bend), "place_split peer");
    need(z, rows * 8192 * 4, "place_split z");
    need(xbc, rows * 10240 * 4, "place_split xbc");
    need(dt, rows * 128 * 4, "place_split dt");
    CudaError::check(unsafe {
        ffi::gb_nemo_tp_place_split(
            own.ptr(),
            i(nl)?,
            i(off)?,
            peer.ptr(),
            i(pnl)?,
            i(poff)?,
            i(bend)?,
            z.ptr(),
            xbc.ptr(),
            dt.ptr(),
            i(rows)?,
            st.raw(),
        )
    })
}

/// The head-group in_proj split (lane S, P2 at two ranks): `src` f32 `[rows, nz + nx + 2 nbc + 128]` (the rank's merged in_proj output:
/// z, x of its heads, B, C of its groups, dt of every head) written where `split_inproj` writes those columns: z `[rows, 8192]` at `zo`
/// (rounded to bf16 as the split does), xBC `[rows, 10240]` (x at `xo`, B at 8192 + `bo`, C at 9216 + `bo`), dt `[rows, 128]`.
#[allow(clippy::too_many_arguments)]
pub fn split_hg(
    src: &DevBuf,
    rows: usize,
    nz: usize,
    nx: usize,
    nbc: usize,
    zo: usize,
    xo: usize,
    bo: usize,
    z: &mut DevBuf,
    xbc: &mut DevBuf,
    dt: &mut DevBuf,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    need(src, rows * (nz + nx + 2 * nbc + 128) * 4, "split_hg src");
    need(z, rows * 8192 * 4, "split_hg z");
    need(xbc, rows * 10240 * 4, "split_hg xbc");
    need(dt, rows * 128 * 4, "split_hg dt");
    CudaError::check(unsafe {
        ffi::gb_nemo_tp_split_hg(
            src.ptr(),
            i(rows)?,
            i(nz)?,
            i(nx)?,
            i(nbc)?,
            i(zo)?,
            i(xo)?,
            i(bo)?,
            z.ptr(),
            xbc.ptr(),
            dt.ptr(),
            st.raw(),
        )
    })
}

/// The CPU model of [`scatter_cols_f32`] (host f32, the same layout), for tests and for stub builds.
pub fn scatter_cols_f32_cpu(src: &[f32], rows: usize, n_local: usize, n_total: usize, col_off: usize) -> Vec<f32> {
    assert!(col_off + n_local <= n_total, "columns [{col_off}, {}) do not fit in {n_total}", col_off + n_local);
    assert_eq!(src.len(), rows * n_local, "src is not [{rows}, {n_local}]");
    let mut dst = vec![-0.0f32; rows * n_total];
    for r in 0..rows {
        dst[r * n_total + col_off..r * n_total + col_off + n_local].copy_from_slice(&src[r * n_local..(r + 1) * n_local]);
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_model_places_the_slice_and_zeros_the_rest() {
        let src: Vec<f32> = (0..2 * 3).map(|v| v as f32 + 1.0).collect();
        let d = scatter_cols_f32_cpu(&src, 2, 3, 8, 4);
        assert_eq!(d.len(), 16);
        assert_eq!(&d[0..8], &[-0.0, -0.0, -0.0, -0.0, 1.0, 2.0, 3.0, -0.0]);
        assert_eq!(&d[8..16], &[-0.0, -0.0, -0.0, -0.0, 4.0, 5.0, 6.0, -0.0]);
        // the filler is -0.0 (0x80000000), the additive identity
        assert!(d.iter().filter(|v| **v == 0.0).all(|v| v.to_bits() == 0x8000_0000));
    }

    #[test]
    fn two_scatters_summed_are_the_gathered_row() {
        // the TP identity: rank 0 and rank 1 scatter their halves, the sum is the whole row, bit for bit (x + (-0.0) == x)
        let mut whole: Vec<f32> = (0..3 * 8).map(|v| f32::from_bits(0x3f80_0000 + 7919 * v as u32)).collect();
        // signed zeros are values of these buffers: both must survive the sum (the filler is -0.0, so x + filler == x for both)
        whole[1] = -0.0;
        whole[5] = 0.0;
        whole[10] = -0.0;
        whole[22] = 0.0;
        let half = |c0: usize| -> Vec<f32> { (0..3).flat_map(|r| whole[r * 8 + c0..r * 8 + c0 + 4].to_vec()).collect() };
        let a = scatter_cols_f32_cpu(&half(0), 3, 4, 8, 0);
        let b = scatter_cols_f32_cpu(&half(4), 3, 4, 8, 4);
        let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert!(sum.iter().zip(&whole).all(|(s, w)| s.to_bits() == w.to_bits()));
    }

    #[test]
    fn a_stub_build_refuses_instead_of_launching() {
        // with no device the launcher is the stub (cudaErrorNoDevice) and a real build returns an error for an out-of-range column
        // window before launching; either way the call is an Err
        let src = DevBuf::zeroed(16);
        if let Ok(src) = src {
            let mut dst = DevBuf::zeroed(64).unwrap();
            let st = Stream::new().unwrap();
            assert!(scatter_cols_f32(&src, &mut dst, 1, 4, 8, 6, &st).is_err());
        }
    }
}
