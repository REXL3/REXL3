//! The log-probability readout kernel against a host f64 reference: random and adversarial rows, row invariance, a golden
//! hash. GPU tests skip when there is no device.
#![cfg(not(gb_no_cuda))]
mod common;

use common::Rng;
use core::ffi::c_void;
use gb_cuda::num::{bf16_to_f32, f32_to_bf16};
use gb_cuda::{device_available, logprob, DevBuf, Stream};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

/// The largest |device lse - host f64 lse| the tests accept. The device sums at most 248,320 fp32 terms in (0, 1] as 1,024
/// strided partial sums folded pairwise and adds the fp32 logarithm to the maximum, so the result is a few fp32 units in
/// the last place of a value near 30 (1.9e-6 each). Measured on a GB10 (sm_121, CUDA 13.0) over the rows of
/// `random_rows_match_the_host_f64_reference`, which prints it: 9.7e-7. The bound is four times that.
const LSE_TOL: f64 = 4e-6;

struct Out {
    /// `[rows, 2]`: lse, chosen logit.
    norm: Vec<f32>,
    ids: Vec<i32>,
    vals: Vec<f32>,
}

fn run(st: &Stream, logits: &[u16], rows: usize, stride: usize, valid: usize, k: usize, chosen: &[i32]) -> Out {
    assert_eq!((logits.len(), chosen.len()), (rows * stride, rows));
    let dev = DevBuf::from_slice(logits).unwrap();
    let ch = DevBuf::from_slice(chosen).unwrap();
    let norm = DevBuf::zeroed(rows * 8).unwrap();
    let ids = DevBuf::zeroed(rows * k * 4).unwrap();
    let vals = DevBuf::zeroed(rows * k * 4).unwrap();
    let wsb = logprob::workspace_bytes(rows, valid, k);
    let ws = DevBuf::zeroed(wsb).unwrap();
    unsafe {
        logprob::rows(
            dev.ptr() as *const c_void,
            rows,
            stride,
            valid,
            k,
            ch.ptr() as *const c_void,
            norm.ptr(),
            ids.ptr(),
            vals.ptr(),
            ws.ptr(),
            wsb,
            st,
        )
        .unwrap()
    };
    st.sync().unwrap();
    Out { norm: norm.to_vec().unwrap(), ids: ids.to_vec().unwrap(), vals: vals.to_vec().unwrap() }
}

/// log(sum exp) of a row's valid columns in f64, max-shifted.
fn lse_ref(row: &[u16]) -> f64 {
    let x: Vec<f64> = row.iter().map(|&u| bf16_to_f32(u) as f64).collect();
    let m = x.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v));
    if !m.is_finite() {
        return m;
    }
    m + x.iter().map(|&v| (v - m).exp()).sum::<f64>().ln()
}

/// The top `k` by (value descending, id ascending), a NaN counted as -inf: the order `glue::topk` documents.
fn top_ref(row: &[u16], k: usize) -> Vec<(i32, f32)> {
    let mut v: Vec<(f32, i32)> = row
        .iter()
        .enumerate()
        .map(|(i, &u)| {
            let f = bf16_to_f32(u);
            (if f.is_nan() { f32::NEG_INFINITY } else { f }, i as i32)
        })
        .collect();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    v.iter().take(k).map(|&(f, i)| (i, f)).collect()
}

/// Checks every output of `rows` finite rows against the references; returns the largest lse difference.
fn check(st: &Stream, logits: &[u16], rows: usize, stride: usize, valid: usize, k: usize, chosen: &[i32], what: &str) -> f64 {
    let o = run(st, logits, rows, stride, valid, k, chosen);
    let mut worst = 0f64;
    for r in 0..rows {
        let row = &logits[r * stride..r * stride + valid];
        let d = (o.norm[2 * r] as f64 - lse_ref(row)).abs();
        assert!(d <= LSE_TOL, "{what}: row {r}: lse {} differs from the f64 reference by {d:e}", o.norm[2 * r]);
        worst = worst.max(d);
        let want = if (0..valid as i32).contains(&chosen[r]) { bf16_to_f32(row[chosen[r] as usize]) } else { f32::NEG_INFINITY };
        assert_eq!(o.norm[2 * r + 1].to_bits(), want.to_bits(), "{what}: row {r}: the chosen logit");
        let top: Vec<(i32, f32)> = (0..k).map(|i| (o.ids[r * k + i], o.vals[r * k + i])).collect();
        assert_eq!(top, top_ref(row, k), "{what}: row {r}: top {k}");
        // the chosen token is among the top k exactly when fewer than k columns rank ahead of it
        if (0..valid as i32).contains(&chosen[r]) {
            let c = chosen[r];
            let ahead = row.iter().enumerate().filter(|&(i, &u)| {
                let (a, b) = (bf16_to_f32(u), want);
                a > b || (a == b && (i as i32) < c)
            });
            assert_eq!(top.iter().any(|t| t.0 == c), ahead.count() < k, "{what}: row {r}: chosen in top {k} iff rank < {k}");
        }
    }
    worst
}

/// Logits shaped like a language-model head's: a wide bulk and a few high columns.
fn head_like(rng: &mut Rng, rows: usize, stride: usize) -> Vec<u16> {
    let mut v: Vec<u16> = (0..rows * stride).map(|_| f32_to_bf16(rng.uniform(-12.0, 4.0))).collect();
    for r in 0..rows {
        for _ in 0..8 {
            v[r * stride + (rng.next_u64() as usize) % stride] = f32_to_bf16(rng.uniform(8.0, 30.0));
        }
    }
    v
}

#[test]
fn random_rows_match_the_host_f64_reference() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(0x10_9b0b);
    let mut worst = 0f64;
    // the served vocabulary; a row shorter than the block; lengths that are not multiples of the block or of 4; a padded stride
    for &(rows, stride, valid) in
        &[(6usize, 248_320usize, 248_320usize), (3, 1000, 1000), (5, 5003, 5003), (4, 248_192, 248_144), (2, 1024, 1)]
    {
        let logits = head_like(&mut rng, rows, stride);
        let chosen: Vec<i32> = (0..rows).map(|_| (rng.next_u64() as usize % valid) as i32).collect();
        for &k in &[1usize, 5, 20] {
            let k = k.min(valid);
            worst = worst.max(check(&st, &logits, rows, stride, valid, k, &chosen, &format!("{rows}x{stride}/{valid} k={k}")));
        }
        // a flat row (every column near the maximum) is the largest sum: the worst case for the fp32 accumulation
        let flat: Vec<u16> = (0..rows * stride).map(|_| f32_to_bf16(rng.uniform(-0.01, 0.01))).collect();
        worst = worst.max(check(&st, &flat, rows, stride, valid, 1, &chosen, &format!("flat {rows}x{stride}/{valid}")));
    }
    eprintln!("logprob kernel: max |lse - f64 reference| = {worst:e} (tolerance {LSE_TOL:e})");
}

#[test]
fn fp16_rows_match_the_host_and_exclude_padding() {
    gpu!();
    let st = Stream::new().unwrap();
    let (rows, stride, valid, k) = (3usize, 5003usize, 4999usize, logprob::MAX_TOP);
    let chosen = [-1i32, 17, 4998];
    let mut rng = Rng(0xf16);
    let logits: Vec<u16> = (0..rows * stride)
        .map(|i| {
            if i % stride >= valid {
                gb_exl3::f16::f64_to_f16(1000.0)
            } else {
                gb_exl3::f16::f64_to_f16(rng.uniform(-12.0, 20.0) as f64)
            }
        })
        .collect();
    let dev = DevBuf::from_slice(&logits).unwrap();
    let ch = DevBuf::from_slice(&chosen).unwrap();
    let norm = DevBuf::zeroed(rows * 8).unwrap();
    let ids = DevBuf::zeroed(rows * k * 4).unwrap();
    let vals = DevBuf::zeroed(rows * k * 4).unwrap();
    let wsb = logprob::workspace_bytes_f16(rows, valid, k);
    let ws = DevBuf::zeroed(wsb).unwrap();
    unsafe {
        logprob::rows_f16(
            dev.ptr().cast_const(),
            rows,
            stride,
            valid,
            k,
            ch.ptr().cast_const(),
            norm.ptr(),
            ids.ptr(),
            vals.ptr(),
            ws.ptr(),
            wsb,
            &st,
        )
        .unwrap();
    }
    st.sync().unwrap();
    let (norm, ids, vals): (Vec<f32>, Vec<i32>, Vec<f32>) =
        (norm.to_vec().unwrap(), ids.to_vec().unwrap(), vals.to_vec().unwrap());
    for r in 0..rows {
        let row: Vec<f32> = logits[r * stride..r * stride + valid].iter().map(|&v| gb_exl3::f16::f16_to_f64(v) as f32).collect();
        let peak = row.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v as f64));
        let lse = peak + row.iter().map(|&v| (v as f64 - peak).exp()).sum::<f64>().ln();
        assert!((norm[2 * r] as f64 - lse).abs() <= LSE_TOL);
        let want_chosen = if chosen[r] < 0 { f32::NEG_INFINITY } else { row[chosen[r] as usize] };
        assert_eq!(norm[2 * r + 1].to_bits(), want_chosen.to_bits());
        let mut order: Vec<u32> = (0..valid as u32).collect();
        order.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]).then(a.cmp(&b)));
        assert_eq!(&ids[r * k..(r + 1) * k], &order[..k].iter().map(|&v| v as i32).collect::<Vec<_>>());
        for j in 0..k {
            assert!(ids[r * k + j] >= 0 && (ids[r * k + j] as usize) < valid, "padding entered the top-k");
            assert_eq!(vals[r * k + j].to_bits(), row[order[j] as usize].to_bits());
        }
    }
}

#[test]
fn adversarial_rows() {
    gpu!();
    let st = Stream::new().unwrap();
    let vocab = 70_001usize;
    let k = logprob::MAX_TOP;
    // all equal: lse = v + ln(vocab), the top k is ids 0..k (more entries asked for than there are distinct values)
    let eq = vec![f32_to_bf16(1.5); vocab];
    check(&st, &eq, 1, vocab, vocab, k, &[vocab as i32 - 1], "all equal");
    let o = run(&st, &eq, 1, vocab, vocab, k, &[7]);
    assert_eq!(o.ids, (0..k as i32).collect::<Vec<_>>());
    // two distinct values and k above the count of the larger: ties broken by the lower id on both sides of the step
    let mut two = vec![f32_to_bf16(-3.0); vocab];
    for i in [65_000usize, 9, 40_000] {
        two[i] = f32_to_bf16(2.0);
    }
    check(&st, &two, 1, vocab, vocab, k, &[9], "two values");
    // one huge column: every other term underflows to 0, lse is that logit exactly
    let mut huge = vec![f32_to_bf16(-5.0); vocab];
    huge[31_337] = f32_to_bf16(3.0e4);
    let o = run(&st, &huge, 1, vocab, vocab, k, &[31_337]);
    assert_eq!((o.norm[0], o.norm[1], o.ids[0]), (bf16_to_f32(huge[31_337]), bf16_to_f32(huge[31_337]), 31_337));
    check(&st, &huge, 1, vocab, vocab, k, &[0], "one huge");
    // -inf columns contribute nothing; a chosen -inf column reads back -inf
    let mut rng = Rng(77);
    let mut masked = head_like(&mut rng, 2, vocab);
    for i in (0..2 * vocab).filter(|i| i % 3 != 0) {
        masked[i] = 0xFF80;
    }
    check(&st, &masked, 2, vocab, vocab, k, &[1, 3], "-inf columns");
    // every column -inf: lse is -inf, not NaN
    let o = run(&st, &vec![0xFF80u16; vocab], 1, vocab, vocab, 1, &[0]);
    assert_eq!((o.norm[0], o.norm[1]), (f32::NEG_INFINITY, f32::NEG_INFINITY));
    // a NaN poisons the row's lse (whichever thread reads it); +inf makes it +inf; the neighbouring rows are untouched
    let mut bad = head_like(&mut rng, 4, vocab);
    bad[vocab + 1024 * 5 + 17] = 0x7FC1;
    bad[3 * vocab + 9] = 0x7F80;
    let o = run(&st, &bad, 4, vocab, vocab, k, &[0, 0, 0, 0]);
    assert!(o.norm[2].is_nan() && o.norm[6] == f32::INFINITY, "{:?}", &o.norm);
    for r in [0usize, 2] {
        assert!((o.norm[2 * r] as f64 - lse_ref(&bad[r * vocab..(r + 1) * vocab])).abs() <= LSE_TOL);
    }
    // a chosen id outside the row (the last prompt row has no next token) reads -inf and disturbs nothing else
    let o = run(&st, &eq, 1, vocab, vocab, 1, &[-1]);
    assert_eq!(o.norm[1], f32::NEG_INFINITY);
    let o = run(&st, &eq, 1, vocab, vocab - 1, 1, &[vocab as i32 - 1]);
    assert_eq!(o.norm[1], f32::NEG_INFINITY, "a column in the pad is outside the row");
    // refused shapes
    let (d, c, n) = (DevBuf::from_slice(&eq).unwrap(), DevBuf::from_slice(&[0i32]).unwrap(), DevBuf::zeroed(8).unwrap());
    let call = |k: usize, valid: usize| unsafe {
        logprob::rows(
            d.ptr() as *const c_void,
            1,
            vocab,
            valid,
            k,
            c.ptr() as *const c_void,
            n.ptr(),
            n.ptr(),
            n.ptr(),
            n.ptr(),
            0,
            &st,
        )
    };
    assert!(call(0, vocab).is_err() && call(logprob::MAX_TOP + 1, vocab).is_err() && call(1, vocab + 1).is_err());
    st.sync().unwrap();
}

/// A row's bits do not depend on its index, on the rows beside it, or on how many rows the call has: the property that makes
/// a readout the same however a prompt is cut into chunks and the head into pieces.
#[test]
fn a_rows_output_does_not_depend_on_the_batch() {
    gpu!();
    let st = Stream::new().unwrap();
    let (vocab, k) = (248_320usize, logprob::MAX_TOP);
    let mut rng = Rng(0xbeef);
    let all = head_like(&mut rng, 7, vocab);
    let chosen: Vec<i32> = (0..7).map(|_| (rng.next_u64() as usize % vocab) as i32).collect();
    let whole = run(&st, &all, 7, vocab, vocab, k, &chosen);
    let bits = |o: &Out, r: usize| -> (Vec<u32>, Vec<i32>, Vec<u32>) {
        (
            o.norm[2 * r..2 * r + 2].iter().map(|v| v.to_bits()).collect(),
            o.ids[r * k..(r + 1) * k].to_vec(),
            o.vals[r * k..(r + 1) * k].iter().map(|v| v.to_bits()).collect(),
        )
    };
    for r in 0..7 {
        // alone
        let one = run(&st, &all[r * vocab..(r + 1) * vocab], 1, vocab, vocab, k, &chosen[r..r + 1]);
        assert_eq!(bits(&one, 0), bits(&whole, r), "row {r} alone");
        // at another index among other rows
        let mut mixed = head_like(&mut rng, 3, vocab);
        let at = r % 3;
        mixed[at * vocab..(at + 1) * vocab].copy_from_slice(&all[r * vocab..(r + 1) * vocab]);
        let mut ch = [5i32, 6, 7];
        ch[at] = chosen[r];
        assert_eq!(bits(&run(&st, &mixed, 3, vocab, vocab, k, &ch), at), bits(&whole, r), "row {r} at index {at} of 3");
    }
    // and a repeat of the same call is the same bits
    let again = run(&st, &all, 7, vocab, vocab, k, &chosen);
    assert_eq!(
        (again.norm.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), again.ids),
        (whole.norm.iter().map(|v| v.to_bits()).collect(), whole.ids)
    );
}

fn fnv<T: Copy>(v: &[T]) -> u64 {
    let p = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) };
    p.iter().fold(0xcbf29ce484222325u64, |h, x| (h ^ u64::from(*x)).wrapping_mul(0x100000001b3))
}

/// The kernel's bits on a fixed input (sm_121, the toolchain of `rust-toolchain.toml` and CUDA 13.0): a change of the
/// reduction order, of the exponential or of the tie order moves a hash.
#[test]
fn golden_logprob_hashes() {
    gpu!();
    let st = Stream::new().unwrap();
    let (rows, vocab, k) = (5usize, 248_320usize, logprob::MAX_TOP);
    let mut rng = Rng(0x601d);
    let logits = head_like(&mut rng, rows, vocab);
    let chosen: Vec<i32> = (0..rows).map(|_| (rng.next_u64() as usize % vocab) as i32).collect();
    let o = run(&st, &logits, rows, vocab, vocab, k, &chosen);
    let got = [("norm", fnv(&o.norm)), ("ids", fnv(&o.ids)), ("vals", fnv(&o.vals))];
    eprintln!("golden_logprob_hashes: {got:x?}");
    let want = [("norm", GOLDEN_NORM), ("ids", GOLDEN_IDS), ("vals", GOLDEN_VALS)];
    assert_eq!(got, want, "got {got:x?}");
}

const GOLDEN_NORM: u64 = 0xd60aea9f15518c69;
const GOLDEN_IDS: u64 = 0xd6c68419ccd258aa;
const GOLDEN_VALS: u64 = 0xdae38c88b92a45e7;
