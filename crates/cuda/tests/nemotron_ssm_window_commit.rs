//! `nemotron_ssm::window_commit` (a prefill slice's pass that reads S and stores the state after every row) against what it replaces:
//! `window` followed by `commit_state` with keep = each sequence's length. The core outputs and every byte of the S pool must be
//! identical, for sequences of unequal lengths (one row, many rows) in non-identity slots, with the slots outside the batch untouched.

#![cfg(not(gb_no_cuda))]

use gb_cuda::gdn::BatchBuf;
use gb_cuda::nemotron_ssm::*;
use gb_cuda::num::f32_to_bf16;
use gb_cuda::{device_available, DevBuf, Stream};

fn splitmix64(z: u64) -> u64 {
    let z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Rng(u64);

impl Rng {
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let v = splitmix64(self.0);
                self.0 = self.0.wrapping_add(1);
                lo + (hi - lo) * ((v >> 40) as f32 / 16_777_216.0)
            })
            .collect()
    }
}

fn bf16(v: &[f32]) -> DevBuf {
    let u: Vec<u16> = v.iter().map(|&x| f32_to_bf16(x)).collect();
    DevBuf::from_slice(&u).unwrap()
}

#[test]
fn window_commit_is_window_then_commit_state_of_every_row() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let mut rng = Rng(0x5eed);
    let lens = [5usize, 1, 17, 64, 2];
    let slots = [3usize, 0, 5, 1, 6];
    let n_slots = 8;
    let rows: usize = lens.iter().sum();

    let conv_w = bf16(&rng.vec(CONV_DIM * CONV_TAPS, -0.5, 0.5));
    let conv_b = bf16(&rng.vec(CONV_DIM, -0.5, 0.5));
    let a_log = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let dt_bias = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let d = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let w = Weights { conv_w: &conv_w, conv_b: &conv_b, a_log: &a_log, dt_bias: &dt_bias, d: &d };

    let s0: Vec<f32> = rng.vec(n_slots * S_SLOT_ELEMS, -1.0, 1.0);
    let conv0: Vec<u16> = rng.vec(n_slots * CONV_SLOT_ELEMS, -2.0, 2.0).iter().map(|&x| f32_to_bf16(x)).collect();
    let xin = DevBuf::from_slice(&rng.vec(rows * CONV_DIM, -2.0, 2.0)).unwrap();
    let dt_raw = DevBuf::from_slice(&rng.vec(rows * HEADS, -2.0, 2.0)).unwrap();
    let conv_pool = DevBuf::from_slice(&conv0).unwrap();

    let bb = BatchBuf::with_pool_slots(&lens, &slots, n_slots).unwrap();
    let batch = bb.view();
    let keep = DevBuf::from_slice(&lens.iter().map(|&l| l as i32).collect::<Vec<_>>()).unwrap();

    // the two calls it replaces
    let mut rec_a = Record::alloc(rows).unwrap();
    let mut s_a = DevBuf::from_slice(&s0).unwrap();
    let mut o_a = DevBuf::zeroed(rows * INNER * 2).unwrap();
    pre(&st, &xin, &dt_raw, &w, &conv_pool, &batch, &mut rec_a).unwrap();
    window(&st, &rec_a, &s_a, &d, &batch, &mut o_a).unwrap();
    commit_state(&st, &rec_a, &mut s_a, &batch, &keep).unwrap();

    // the fused pass
    let mut rec_b = Record::alloc(rows).unwrap();
    let mut s_b = DevBuf::from_slice(&s0).unwrap();
    let mut o_b = DevBuf::zeroed(rows * INNER * 2).unwrap();
    pre(&st, &xin, &dt_raw, &w, &conv_pool, &batch, &mut rec_b).unwrap();
    window_commit(&st, &rec_b, &mut s_b, &d, &batch, &mut o_b).unwrap();
    st.sync().unwrap();

    let (oa, ob) = (o_a.to_vec::<u16>().unwrap(), o_b.to_vec::<u16>().unwrap());
    assert!(oa == ob, "core outputs differ at element {:?}", oa.iter().zip(&ob).position(|(x, y)| x != y));
    let (sa, sb) = (s_a.to_vec::<u32>().unwrap(), s_b.to_vec::<u32>().unwrap());
    assert!(sa == sb, "S pools differ at element {:?}", sa.iter().zip(&sb).position(|(x, y)| x != y));
    // and the commit really moved the used slots and left the others alone
    let init: Vec<u32> = s0.iter().map(|x| x.to_bits()).collect();
    for s in 0..n_slots {
        let r = s * S_SLOT_ELEMS..(s + 1) * S_SLOT_ELEMS;
        let moved = sb[r.clone()] != init[r];
        assert_eq!(moved, slots.contains(&s), "slot {s}: moved = {moved}");
    }
}

/// A prefill slice of 128 or 256 rows (the width the large-row MoE prefill allows) is the same bits as the same rows cut into slices of
/// 64 or any other cut: the core output of every row, the S pool and the conv pool after the last row. A slice is `pre`, `window_commit`
/// and `commit_conv` with keep = its rows, and a later slice reads the pools the earlier one left.
#[test]
fn a_wide_prefill_slice_equals_the_same_rows_cut_into_narrow_slices() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let mut rng = Rng(0xFEED);
    let n_slots = 3;
    let slot = 2usize;
    let total = 256usize;

    let conv_w = bf16(&rng.vec(CONV_DIM * CONV_TAPS, -0.5, 0.5));
    let conv_b = bf16(&rng.vec(CONV_DIM, -0.5, 0.5));
    let a_log = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let dt_bias = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let d = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let w = Weights { conv_w: &conv_w, conv_b: &conv_b, a_log: &a_log, dt_bias: &dt_bias, d: &d };

    let s0: Vec<f32> = rng.vec(n_slots * S_SLOT_ELEMS, -1.0, 1.0);
    let conv0: Vec<u16> = rng.vec(n_slots * CONV_SLOT_ELEMS, -2.0, 2.0).iter().map(|&x| f32_to_bf16(x)).collect();
    let xin_h = rng.vec(total * CONV_DIM, -2.0, 2.0);
    let dt_h = rng.vec(total * HEADS, -2.0, 2.0);

    // The rows `start..start + len` as one slice on the pools (s, conv); returns the core output of the rows.
    let run_slice = |s: &mut DevBuf, conv: &mut DevBuf, start: usize, len: usize| -> Vec<u16> {
        let xin = DevBuf::from_slice(&xin_h[start * CONV_DIM..(start + len) * CONV_DIM]).unwrap();
        let dt_raw = DevBuf::from_slice(&dt_h[start * HEADS..(start + len) * HEADS]).unwrap();
        let bb = BatchBuf::with_pool_slots(&[len], &[slot], n_slots).unwrap();
        let batch = bb.view();
        let keep = DevBuf::from_slice(&[len as i32]).unwrap();
        let mut rec = Record::alloc(len).unwrap();
        let mut o = DevBuf::zeroed(len * INNER * 2).unwrap();
        pre(&st, &xin, &dt_raw, &w, conv, &batch, &mut rec).unwrap();
        window_commit(&st, &rec, s, &d, &batch, &mut o).unwrap();
        commit_conv(&st, &xin, conv, &batch, &keep).unwrap();
        st.sync().unwrap();
        o.to_vec::<u16>().unwrap()
    };
    let run_cut = |cuts: &[usize]| -> (Vec<u16>, Vec<u32>, Vec<u16>) {
        assert_eq!(cuts.iter().sum::<usize>(), total);
        let mut s = DevBuf::from_slice(&s0).unwrap();
        let mut conv = DevBuf::from_slice(&conv0).unwrap();
        let (mut o, mut start) = (Vec::new(), 0);
        for &len in cuts {
            o.extend(run_slice(&mut s, &mut conv, start, len));
            start += len;
        }
        (o, s.to_vec::<u32>().unwrap(), conv.to_vec::<u16>().unwrap())
    };

    let whole = run_cut(&[256]);
    for cuts in [vec![64, 64, 64, 64], vec![128, 128], vec![100, 156], vec![1, 255], vec![37, 64, 128, 27], vec![1, 1, 1, 253]] {
        let got = run_cut(&cuts);
        assert!(got.0 == whole.0, "{cuts:?}: core output differs at {:?}", got.0.iter().zip(&whole.0).position(|(a, b)| a != b));
        assert!(got.1 == whole.1, "{cuts:?}: S pool differs at {:?}", got.1.iter().zip(&whole.1).position(|(a, b)| a != b));
        assert!(got.2 == whole.2, "{cuts:?}: conv pool differs at {:?}", got.2.iter().zip(&whole.2).position(|(a, b)| a != b));
    }
}

/// `window_commit_sel`: a decode window in which the single-row sequences are applied in the pass that reads their state and the others
/// (several rows: they may be rolled back) are left to a later commit. For every sequence the core output equals `window`'s; the S pool
/// equals `window` + `commit_state` with keep = the length for an applied sequence and the initial state for the others; the conv
/// history moved by `commit_conv` with `apply` as `keep` equals the same with keep = the length for an applied sequence and 0 for the
/// others. Non-identity slots, slots outside the batch untouched, and the one-row case at every position of the batch.
#[test]
fn window_commit_sel_applies_exactly_the_flagged_sequences() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let mut rng = Rng(0xA991);
    let lens = [1usize, 1, 5, 1, 17, 1, 2, 1];
    let apply_h = [1i32, 1, 0, 1, 0, 1, 0, 1];
    let slots = [3usize, 0, 5, 1, 6, 7, 2, 4];
    let n_slots = 9;
    let rows: usize = lens.iter().sum();

    let conv_w = bf16(&rng.vec(CONV_DIM * CONV_TAPS, -0.5, 0.5));
    let conv_b = bf16(&rng.vec(CONV_DIM, -0.5, 0.5));
    let a_log = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let dt_bias = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let d = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let w = Weights { conv_w: &conv_w, conv_b: &conv_b, a_log: &a_log, dt_bias: &dt_bias, d: &d };

    let s0: Vec<f32> = rng.vec(n_slots * S_SLOT_ELEMS, -1.0, 1.0);
    let conv0: Vec<u16> = rng.vec(n_slots * CONV_SLOT_ELEMS, -2.0, 2.0).iter().map(|&x| f32_to_bf16(x)).collect();
    let xin = DevBuf::from_slice(&rng.vec(rows * CONV_DIM, -2.0, 2.0)).unwrap();
    let dt_raw = DevBuf::from_slice(&rng.vec(rows * HEADS, -2.0, 2.0)).unwrap();
    let bb = BatchBuf::with_pool_slots(&lens, &slots, n_slots).unwrap();
    let batch = bb.view();
    let apply = DevBuf::from_slice(&apply_h).unwrap();
    let keep_ref: Vec<i32> = lens.iter().zip(&apply_h).map(|(&l, &a)| if a > 0 { l as i32 } else { 0 }).collect();
    let keep = DevBuf::from_slice(&keep_ref).unwrap();

    // reference: window, then the state and conv commits with keep = the applied sequences' lengths and 0 for the rest
    let conv_a = DevBuf::from_slice(&conv0).unwrap();
    let mut conv_a_new = DevBuf::from_slice(&conv0).unwrap();
    let mut rec_a = Record::alloc(rows).unwrap();
    let mut s_a = DevBuf::from_slice(&s0).unwrap();
    let mut o_a = DevBuf::zeroed(rows * INNER * 2).unwrap();
    pre(&st, &xin, &dt_raw, &w, &conv_a, &batch, &mut rec_a).unwrap();
    window(&st, &rec_a, &s_a, &d, &batch, &mut o_a).unwrap();
    commit_state(&st, &rec_a, &mut s_a, &batch, &keep).unwrap();
    commit_conv(&st, &xin, &mut conv_a_new, &batch, &keep).unwrap();

    // the selecting pass, with `apply` as the conv commit's keep
    let conv_b = DevBuf::from_slice(&conv0).unwrap();
    let mut conv_b_new = DevBuf::from_slice(&conv0).unwrap();
    let mut rec_b = Record::alloc(rows).unwrap();
    let mut s_b = DevBuf::from_slice(&s0).unwrap();
    let mut o_b = DevBuf::zeroed(rows * INNER * 2).unwrap();
    pre(&st, &xin, &dt_raw, &w, &conv_b, &batch, &mut rec_b).unwrap();
    window_commit_sel(&st, &rec_b, &mut s_b, &d, &batch, &apply, &mut o_b).unwrap();
    commit_conv(&st, &xin, &mut conv_b_new, &batch, &apply).unwrap();
    st.sync().unwrap();

    let (oa, ob) = (o_a.to_vec::<u16>().unwrap(), o_b.to_vec::<u16>().unwrap());
    assert!(oa == ob, "core outputs differ at element {:?}", oa.iter().zip(&ob).position(|(x, y)| x != y));
    let (sa, sb) = (s_a.to_vec::<u32>().unwrap(), s_b.to_vec::<u32>().unwrap());
    assert!(sa == sb, "S pools differ at element {:?}", sa.iter().zip(&sb).position(|(x, y)| x != y));
    let (ca, cb) = (conv_a_new.to_vec::<u16>().unwrap(), conv_b_new.to_vec::<u16>().unwrap());
    assert!(ca == cb, "conv pools differ at element {:?}", ca.iter().zip(&cb).position(|(x, y)| x != y));
    // exactly the applied sequences' slots moved
    let init: Vec<u32> = s0.iter().map(|x| x.to_bits()).collect();
    for s in 0..n_slots {
        let r = s * S_SLOT_ELEMS..(s + 1) * S_SLOT_ELEMS;
        let moved = sb[r.clone()] != init[r];
        let applied = slots.iter().zip(&apply_h).any(|(&sl, &a)| sl == s && a > 0);
        assert_eq!(moved, applied, "slot {s}: moved = {moved}, applied = {applied}");
    }
}

/// Lane S3: with the prefill opt-in the slice's pass runs on the four-threads-a-column kernel (`window_commit4_kernel`). It must be the
/// one-thread-a-column kernel's bits: core outputs and every byte of the S pool against `window` + `commit_state`, for several
/// sequences of unequal lengths (1 to 300 rows: more than one staging chunk of 16, and a partial last chunk) in non-identity slots, and
/// a wide slice against the same rows cut in narrow slices on the fast kernel.
#[test]
fn the_four_lane_scan_has_the_one_lane_scans_bits() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let mut rng = Rng(0xC0FFEE);
    let lens = [300usize, 1, 17, 16, 65];
    let slots = [2usize, 0, 5, 1, 6];
    let n_slots = 8;
    let rows: usize = lens.iter().sum();
    let conv_w = bf16(&rng.vec(CONV_DIM * CONV_TAPS, -0.5, 0.5));
    let conv_b = bf16(&rng.vec(CONV_DIM, -0.5, 0.5));
    let a_log = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let dt_bias = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let d = DevBuf::from_slice(&rng.vec(HEADS, -1.0, 1.0)).unwrap();
    let w = Weights { conv_w: &conv_w, conv_b: &conv_b, a_log: &a_log, dt_bias: &dt_bias, d: &d };
    let s0: Vec<f32> = rng.vec(n_slots * S_SLOT_ELEMS, -1.0, 1.0);
    let conv0: Vec<u16> = rng.vec(n_slots * CONV_SLOT_ELEMS, -2.0, 2.0).iter().map(|&x| f32_to_bf16(x)).collect();
    let xin_h = rng.vec(rows * CONV_DIM, -2.0, 2.0);
    let dt_h = rng.vec(rows * HEADS, -2.0, 2.0);
    let xin = DevBuf::from_slice(&xin_h).unwrap();
    let dt_raw = DevBuf::from_slice(&dt_h).unwrap();
    let conv_pool = DevBuf::from_slice(&conv0).unwrap();
    let bb = BatchBuf::with_pool_slots(&lens, &slots, n_slots).unwrap();
    let batch = bb.view();
    let keep = DevBuf::from_slice(&lens.iter().map(|&l| l as i32).collect::<Vec<_>>()).unwrap();

    let mut rec = Record::alloc(rows).unwrap();
    pre(&st, &xin, &dt_raw, &w, &conv_pool, &batch, &mut rec).unwrap();
    let mut s_a = DevBuf::from_slice(&s0).unwrap();
    let mut o_a = DevBuf::zeroed(rows * INNER * 2).unwrap();
    window(&st, &rec, &s_a, &d, &batch, &mut o_a).unwrap();
    commit_state(&st, &rec, &mut s_a, &batch, &keep).unwrap();
    gb_cuda::nemotron_dense_mma::set_enabled(true);
    let mut s_b = DevBuf::from_slice(&s0).unwrap();
    let mut o_b = DevBuf::zeroed(rows * INNER * 2).unwrap();
    window_commit(&st, &rec, &mut s_b, &d, &batch, &mut o_b).unwrap();
    st.sync().unwrap();
    let (oa, ob) = (o_a.to_vec::<u16>().unwrap(), o_b.to_vec::<u16>().unwrap());
    assert!(oa == ob, "core outputs differ at element {:?}", oa.iter().zip(&ob).position(|(x, y)| x != y));
    let (sa, sb) = (s_a.to_vec::<u32>().unwrap(), s_b.to_vec::<u32>().unwrap());
    assert!(sa == sb, "S pools differ at element {:?}", sa.iter().zip(&sb).position(|(x, y)| x != y));

    // one sequence of 300 rows as one slice, and cut in slices of 100, 17 and 1 rows, on the fast kernel
    let one = BatchBuf::with_pool_slots(&[300], &[2], n_slots).unwrap();
    let mut s_w = DevBuf::from_slice(&s0).unwrap();
    let mut o_w = DevBuf::zeroed(300 * INNER * 2).unwrap();
    window_commit(&st, &rec, &mut s_w, &d, &one.view(), &mut o_w).unwrap();
    st.sync().unwrap();
    let (ow, sw) = (o_w.to_vec::<u16>().unwrap(), s_w.to_vec::<u32>().unwrap());
    assert!(ow[..] == oa[..300 * INNER], "the 300-row sequence alone differs from its rows in the batch");
    for cut in [100usize, 17, 1] {
        let mut s_c = DevBuf::from_slice(&s0).unwrap();
        let mut conv = DevBuf::from_slice(&conv0).unwrap();
        let mut got: Vec<u16> = Vec::new();
        let mut start = 0;
        while start < 300 {
            let len = cut.min(300 - start);
            let mut r = Record::alloc(len).unwrap();
            let xs = DevBuf::from_slice(&xin_h[start * CONV_DIM..(start + len) * CONV_DIM]).unwrap();
            let ds = DevBuf::from_slice(&dt_h[start * HEADS..(start + len) * HEADS]).unwrap();
            // a later slice's conv history is what commit_conv left in the pool
            let b = BatchBuf::with_pool_slots(&[len], &[2], n_slots).unwrap();
            pre(&st, &xs, &ds, &w, &conv, &b.view(), &mut r).unwrap();
            let mut o = DevBuf::zeroed(len * INNER * 2).unwrap();
            window_commit(&st, &r, &mut s_c, &d, &b.view(), &mut o).unwrap();
            let k = DevBuf::from_slice(&[len as i32]).unwrap();
            commit_conv(&st, &xs, &mut conv, &b.view(), &k).unwrap();
            st.sync().unwrap();
            got.extend(o.to_vec::<u16>().unwrap());
            start += len;
        }
        assert!(got == ow, "slices of {cut}: core output differs at {:?}", got.iter().zip(&ow).position(|(a, b)| a != b));
        let sc = s_c.to_vec::<u32>().unwrap();
        assert!(sc == sw, "slices of {cut}: S pool differs at {:?}", sc.iter().zip(&sw).position(|(a, b)| a != b));
    }
    gb_cuda::nemotron_dense_mma::set_enabled(false);
}
