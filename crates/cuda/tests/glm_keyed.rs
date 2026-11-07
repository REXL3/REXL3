//! The GLM NextN head's keyed pick (`csrc/glm_mtp.cu::keyed_pick_k`, header `csrc/keyed.cuh`): the head's top candidates become a draft by the TARGET's keyed
//! draw (seed, absolute position, token id), against `gb_sampling::try_choose` on the same candidates, its row invariance, and a golden hash.
//! GPU tests print "not run" when there is no device.
//!
//!     REXL3_GOLDEN_OUT=testdata/gpu/golden_glm_keyed.txt cargo test --release -p gb-cuda --test glm_keyed golden_glm_keyed_hashes -- --ignored
#![cfg(not(gb_no_cuda))]
mod common;

use common::*;
use gb_cuda::glm_mtp::{keyed_pick, KeyedRow};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_sampling::{try_choose, Sampling};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: not run");
            return;
        }
    };
}

/// A candidate list as the head's top-k produces it: `count` (id, value) pairs by (value desc, id asc), the values fp16-exact (the head's logits are
/// fp16), a few exact ties, a few strong tokens so that top_p and min_p cut something.
fn candidates(rng: &mut Rng, count: usize) -> (Vec<i32>, Vec<f32>) {
    let mut ids: Vec<i32> = Vec::new();
    while ids.len() < count {
        let id = (rng.next_u64() % 154_856) as i32;
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    let mut pairs: Vec<(i32, f32)> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| {
            let base = rng.uniform(-3.0, 3.0);
            let v = if i % 7 == 3 {
                1.0
            } else if i % 11 == 0 {
                base + 4.0
            } else {
                base
            };
            // fp16 rounding: the value as the head's fp16 logit would be
            (id, gb_exl3::f16::f16_to_f64(gb_exl3::f16::f64_to_f16(v as f64)) as f32)
        })
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    (pairs.iter().map(|p| p.0).collect(), pairs.iter().map(|p| p.1).collect())
}

fn settings(rng: &mut Rng, count: usize) -> Sampling {
    let pick = |rng: &mut Rng, n: usize| (rng.next_u64() % n as u64) as usize;
    let top_k = [1usize, 2, 8, 20, 40, 56][pick(rng, 6)].min(count);
    Sampling {
        seed: rng.next_u64() & ((1 << 63) - 1),
        temperature: [0.3, 0.7, 1.0, 1.3][pick(rng, 4)],
        top_k,
        top_p: [1.0, 0.95, 0.8, 0.5][pick(rng, 4)],
        min_p: [0.0, 0.05, 0.2][pick(rng, 3)],
    }
}

fn row_of(s: &Sampling, position: u64, keyed: bool) -> KeyedRow {
    KeyedRow {
        seed: s.seed,
        first_position: position,
        temperature: s.temperature,
        top_p: s.top_p,
        min_p: s.min_p,
        top_k: s.top_k as i32,
        keyed: keyed as i32,
    }
}

/// Run the kernel over `rows` (lists and records) at chain depth `depth`; (ids, values) of the picks.
fn run(st: &Stream, lists: &[(Vec<i32>, Vec<f32>)], recs: &[KeyedRow], count: usize, depth: usize) -> (Vec<i32>, Vec<f32>) {
    let n = lists.len();
    let ids: Vec<i32> = lists.iter().flat_map(|l| l.0.iter().copied()).collect();
    let vals: Vec<f32> = lists.iter().flat_map(|l| l.1.iter().copied()).collect();
    let (dids, dvals, dp) =
        (DevBuf::from_slice(&ids).unwrap(), DevBuf::from_slice(&vals).unwrap(), DevBuf::from_slice(recs).unwrap());
    let (mut oi, mut ov) = (DevBuf::zeroed(n * 4).unwrap(), DevBuf::zeroed(n * 4).unwrap());
    keyed_pick(&dids, &dvals, count, n, &dp, depth, &mut oi, &mut ov, st).unwrap();
    st.sync().unwrap();
    let (mut a, mut b) = (vec![0i32; n], vec![0f32; n]);
    oi.download(&mut a).unwrap();
    ov.download(&mut b).unwrap();
    (a, b)
}

#[test]
fn the_device_pick_is_the_hosts_try_choose_on_the_same_candidates() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(0x6B657965);
    let (mut rows_checked, mut keyed_rows, mut mismatches) = (0usize, 0usize, 0usize);
    for count in [8usize, 28, 32, 48, 64] {
        for depth in [0usize, 1, 2, 3] {
            let n = 200;
            let mut lists = Vec::new();
            let mut sets = Vec::new();
            let mut recs = Vec::new();
            let mut pos = Vec::new();
            for r in 0..n {
                lists.push(candidates(&mut rng, count));
                let s = settings(&mut rng, count);
                let p = rng.next_u64() % 6000;
                // one row in five is not keyed: the first candidate, as the argmax drafter takes it
                let keyed = r % 5 != 0;
                recs.push(row_of(&s, p, keyed));
                sets.push((s, keyed));
                pos.push(p);
            }
            let (ids, vals) = run(&st, &lists, &recs, count, depth);
            for r in 0..n {
                let (s, keyed) = &sets[r];
                let want = if *keyed {
                    try_choose(&lists[r].1, &lists[r].0.iter().map(|&i| i as u32).collect::<Vec<_>>(), pos[r] + depth as u64, s)
                        .unwrap() as i32
                } else {
                    lists[r].0[0]
                };
                if ids[r] != want {
                    mismatches += 1;
                    // a mismatch is only ever a near-tie of two noised scores that the device's and the host's libm round apart: tell it
                    // from a real bug by the gap between the two
                    eprintln!("row {r} count {count} depth {depth}: device {} host {want} ({s:?})", ids[r]);
                }
                let at = lists[r].0.iter().position(|&i| i == ids[r]).expect("the pick is one of the candidates");
                assert_eq!(vals[r].to_bits(), lists[r].1[at].to_bits(), "the value returned is the pick's own logit");
                rows_checked += 1;
                keyed_rows += usize::from(*keyed);
            }
        }
    }
    assert_eq!(mismatches, 0, "{mismatches} of {rows_checked} rows ({keyed_rows} keyed) differ from the host's try_choose");
}

#[test]
fn a_rows_pick_depends_on_its_own_list_and_record_alone() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(77);
    let count = 28;
    let n = 128;
    let lists: Vec<_> = (0..n).map(|_| candidates(&mut rng, count)).collect();
    let recs: Vec<KeyedRow> = (0..n).map(|_| row_of(&settings(&mut rng, count), rng.next_u64() % 4000, true)).collect();
    let all = run(&st, &lists, &recs, count, 2);
    // each row alone, in a batch of 2 with a stranger either side of it, at the last place, and in every window size
    for r in (0..n).step_by(5) {
        let alone = run(&st, &lists[r..r + 1], &recs[r..r + 1], count, 2);
        assert_eq!((alone.0[0], alone.1[0].to_bits()), (all.0[r], all.1[r].to_bits()), "row {r} alone");
        let other = (r + 17) % n;
        let pair = run(&st, &[lists[other].clone(), lists[r].clone()], &[recs[other], recs[r]], count, 2);
        assert_eq!(pair.0[1], all.0[r], "row {r} behind another row");
        let pair = run(&st, &[lists[r].clone(), lists[other].clone()], &[recs[r], recs[other]], count, 2);
        assert_eq!(pair.0[0], all.0[r], "row {r} before another row");
    }
    for w in [1usize, 2, 7, 33, 64, 127] {
        let part = run(&st, &lists[..w], &recs[..w], count, 2);
        assert_eq!(part.0[..], all.0[..w], "the first {w} rows");
    }
    // the record's noise is the row's own: the same list and record at the same place in a different row of the batch draws the same, and
    // moving the position (or the depth, which moves it by the same amount) moves the draw
    let mut moved = 0;
    for r in 0..n {
        let mut shifted = recs[r];
        shifted.first_position += 1;
        let a = run(&st, &lists[r..r + 1], &[shifted], count, 2).0[0];
        let b = run(&st, &lists[r..r + 1], &[recs[r]], count, 3).0[0];
        assert_eq!(a, b, "row {r}: position + 1 at depth 2 is position at depth 3");
        if a != all.0[r] {
            moved += 1;
        }
    }
    assert!(moved > 10, "a position one on drew other noise in only {moved} of {n} rows");
}

#[test]
fn the_noise_of_a_candidate_is_its_token_ids_not_its_columns() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(5150);
    // a restricted head's list is a subset of the full list's candidates with the same ids: when every candidate the sampler keeps is inside it, the pick
    // is the full list's (the full head draws the same noise for the same ids). Compare the device on both lists.
    let (mut checked, mut differ_by_column) = (0, 0);
    for _ in 0..300 {
        let count = 64;
        let (ids, vals) = candidates(&mut rng, count);
        let s = Sampling { top_k: [5usize, 20, 40][(rng.next_u64() % 3) as usize], ..settings(&mut rng, count) };
        let rec = row_of(&s, rng.next_u64() % 3000, true);
        let full = run(&st, &[(ids.clone(), vals.clone())], &[rec], count, 1).0[0];
        // the subset keeps the first top_k (what the sampler keeps) and every candidate beyond them with probability 1/2, padded to the same count
        let keep: Vec<usize> = (0..count).filter(|&i| i < s.top_k || rng.next_u64() % 2 == 0).collect();
        let m = keep.len().min(count);
        let sub_ids: Vec<i32> = keep[..m].iter().map(|&i| ids[i]).collect();
        let sub_vals: Vec<f32> = keep[..m].iter().map(|&i| vals[i]).collect();
        let sub = run(&st, &[(sub_ids.clone(), sub_vals.clone())], &[rec], m, 1).0[0];
        assert_eq!(sub, full, "the pick over a subset that holds every kept candidate is the full list's");
        // control: the same subset with column numbers in place of the ids draws other noise
        let cols: Vec<i32> = (0..m as i32).collect();
        let by_col = run(&st, &[(cols, sub_vals)], &[rec], m, 1).0[0];
        if sub_ids[by_col as usize] != full {
            differ_by_column += 1;
        }
        checked += 1;
    }
    assert_eq!(checked, 300);
    assert!(
        differ_by_column > 30,
        "the control (noise keyed by column) must be able to fail: it differed {differ_by_column} times"
    );
}

#[test]
fn greedy_rows_and_unkeyed_rows_take_the_first_candidate_whatever_the_record() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(3);
    let count = 32;
    let lists: Vec<_> = (0..64).map(|_| candidates(&mut rng, count)).collect();
    let mut recs: Vec<KeyedRow> = (0..64).map(|_| row_of(&settings(&mut rng, count), rng.next_u64() % 999, true)).collect();
    for (i, r) in recs.iter_mut().enumerate() {
        if i % 2 == 0 {
            r.temperature = 0.0; // greedy: no noise
        } else {
            r.keyed = 0; // a request the drafter does not key
        }
    }
    let (ids, vals) = run(&st, &lists, &recs, count, 3);
    for r in 0..64 {
        assert_eq!(ids[r], lists[r].0[0], "row {r}");
        assert_eq!(vals[r].to_bits(), lists[r].1[0].to_bits(), "row {r}");
    }
}

#[test]
fn bad_shapes_are_refused_and_no_rows_is_a_no_op() {
    gpu!();
    let st = Stream::new().unwrap();
    let (ids, vals) = (DevBuf::zeroed(64 * 4 * 2).unwrap(), DevBuf::zeroed(64 * 4 * 2).unwrap());
    let params = DevBuf::zeroed(48 * 2).unwrap();
    let (mut oi, mut ov) = (DevBuf::zeroed(8).unwrap(), DevBuf::zeroed(8).unwrap());
    assert!(keyed_pick(&ids, &vals, 65, 1, &params, 0, &mut oi, &mut ov, &st).is_err(), "more than 64 candidates");
    assert!(keyed_pick(&ids, &vals, 0, 1, &params, 0, &mut oi, &mut ov, &st).is_err(), "no candidates");
    assert!(keyed_pick(&ids, &vals, 64, 3, &params, 0, &mut oi, &mut ov, &st).is_err(), "buffers too small for 3 rows");
    assert!(keyed_pick(&ids, &vals, 64, 2, &params, 0, &mut oi, &mut ov, &st).is_ok());
    assert!(keyed_pick(&ids, &vals, 64, 0, &params, 0, &mut oi, &mut ov, &st).is_ok(), "zero rows is a no-op");
    st.sync().unwrap();
}

fn fnv(b: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Hashes of the picks for a spread of row counts, list lengths, depths and samplings on seeded lists: a change that must keep every bit diffs this
/// file against one written by the old build (a different nvcc or libm, a changed order of the candidates or the cut, noise on another key).
///
///     REXL3_GOLDEN_OUT=testdata/gpu/golden_glm_keyed.txt cargo test --release -p gb-cuda --test glm_keyed golden_glm_keyed_hashes -- --ignored
#[test]
#[ignore]
fn golden_glm_keyed_hashes() {
    use std::io::Write;
    gpu!();
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    let mut f = std::fs::File::create(path).unwrap();
    let st = Stream::new().unwrap();
    for count in [8usize, 28, 32, 64] {
        for rows in [1usize, 2, 3, 8, 17, 64, 128] {
            for depth in [0usize, 1, 3] {
                let mut rng = Rng(9000 + (count * 1000 + rows * 10 + depth) as u64);
                let lists: Vec<_> = (0..rows).map(|_| candidates(&mut rng, count)).collect();
                let recs: Vec<KeyedRow> = (0..rows)
                    .map(|r| {
                        let s = settings(&mut rng, count);
                        let mut rec = row_of(&s, rng.next_u64() % 8000, r % 6 != 5);
                        if r % 9 == 4 {
                            rec.temperature = 0.0;
                        }
                        rec
                    })
                    .collect();
                let (ids, vals) = run(&st, &lists, &recs, count, depth);
                let vb: Vec<u8> = vals.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
                let ib: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
                writeln!(f, "count={count} rows={rows} depth={depth} ids {:016x} vals {:016x}", fnv(&ib), fnv(&vb)).unwrap();
            }
        }
    }
}
