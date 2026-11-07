//! Long-context prefill attention of the Nemotron 3.5 Super VL NoPE layers (`csrc/nemotron_attn.cu`, `csrc/nemotron_attn_prefill.cu`):
//! what the exact reduction tree costs at 4k to 1M keys, and what the flash-style tensor-core path costs and guarantees.
//!
//! Everything here is synthetic (random q, k and v). The cache is REAL at every depth: one attention layer's K and V for 1M tokens is
//! 1 GiB (2 kv heads x 128 x fp16 x 2 x 1M), so the timings read genuine DRAM, nothing is aliased. Run on a GB10 under the device
//! lock, release build, one test at a time:
//!
//!     cargo test --release -p gb-cuda --test nemotron_attn_long -- --ignored --test-threads=1 --nocapture <name>
//!
//! * `bench_tree_long_context`: the CURRENT kernels (`attn_prefill` and `attn_decode`): per-slice time, achieved TFLOP/s and
//!   effective KV GB/s for 64- and 256-row slices ending at 4k, 32k, 128k, 512k and 1M keys, and the decode step at 1, 4 and 8 rows
//!   at 128k, 512k and 1M. It ends with the 1M-token prefill projection from the measured slice times.
//! * the flash tests (`flash_*`) are further down; they are skipped without a device.
//!
//! `REXL3_ATTN_LONG_POS` (default `4096,32768,131072,524288,1048576`) and `REXL3_ATTN_LONG_REPS` (default 5) tune the table.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use gb_cuda::attention::BlockTable;
use gb_cuda::moe::EventTimer;
use gb_cuda::nemotron_attn::{self as na, NemoKv, RowIndex, Scratch};
use gb_cuda::{device_available, DevBuf, Stream};

/// fp32 elements of one query or output row (32 heads of 128).
const QD: usize = 32 * 128;
/// fp32 elements of one projected K or V row (2 kv heads of 128).
const KVD: usize = 2 * 128;
const PAGE: usize = 256;

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

/// A xorshift stream: the test data is deterministic and needs no crate.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[-scale, scale)`.
    fn f32(&mut self, scale: f32) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32(scale)).collect()
    }
}

fn env_list(name: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(name) {
        Ok(v) => v.split(',').map(|x| x.trim().parse().unwrap()).collect(),
        Err(_) => default.to_vec(),
    }
}

fn fmt_pos(p: usize) -> String {
    if p % 1024 == 0 {
        format!("{}k", p / 1024)
    } else {
        p.to_string()
    }
}

/// One layer's cache of `tokens` positions of slot 0 on the identity block table, filled with random K and V (every 8,192-position
/// block of the cache is its own draw), and the table.
fn long_cache(tokens: usize, st: &Stream) -> (NemoKv, BlockTable) {
    let pages = tokens.div_ceil(PAGE);
    let kv_blocks = pages + 1;
    let mut kv = NemoKv::new(kv_blocks).unwrap();
    let table = BlockTable::from_host(&[(0..pages as u32).collect()], pages).unwrap();
    const BATCH: usize = 8192;
    let mut rng = Rng(0x5EED_10C0);
    let mut start = 0;
    while start < tokens {
        let n = BATCH.min(tokens - start);
        let k = DevBuf::from_slice(&rng.vec(n * KVD, 1.0)).unwrap();
        let v = DevBuf::from_slice(&rng.vec(n * KVD, 1.0)).unwrap();
        let pos: Vec<i32> = (start..start + n).map(|p| p as i32).collect();
        let idx = RowIndex::from_host(&vec![0; n], &pos).unwrap();
        na::kv_write(&mut kv, &table, &idx, &k, &v, st).unwrap();
        st.sync().unwrap();
        start += n;
    }
    (kv, table)
}

/// Median over `reps` of the stream time of `launch` (event to event), after one warm-up call.
fn median_ms(st: &Stream, reps: usize, mut launch: impl FnMut()) -> f64 {
    let t = EventTimer::new(2).unwrap();
    launch();
    st.sync().unwrap();
    let mut v = Vec::new();
    for _ in 0..reps {
        t.record(0, st).unwrap();
        launch();
        t.record(1, st).unwrap();
        v.push(t.ms(0, 1).unwrap() as f64);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// FLOP of attention for rows at `positions`: 512 per (query head, key): 128 MACs for the score and 128 for the weighted sum.
fn flops(positions: &[i32]) -> f64 {
    positions.iter().map(|&p| (p as f64 + 1.0) * 32.0 * 512.0).sum()
}

/// One slice: `rows` rows ending at key count `end` (positions `end - rows .. end`), at slot 0.
fn slice_rows(end: usize, rows: usize) -> (Vec<i32>, Vec<i32>) {
    let pos: Vec<i32> = (end - rows..end).map(|p| p as i32).collect();
    (vec![0; rows], pos)
}

/// Linear interpolation of `(x, y)` samples (sorted by x), extrapolated flat below and linearly above with the last segment's slope.
fn interp(samples: &[(f64, f64)], x: f64) -> f64 {
    if samples.len() == 1 {
        return samples[0].1;
    }
    let i = samples.iter().position(|s| s.0 >= x).unwrap_or(samples.len() - 1).max(1);
    let (a, b) = (samples[i - 1], samples[i]);
    a.1 + (b.1 - a.1) * (x - a.0) / (b.0 - a.0)
}

/// Seconds of attention (all `layers`) of a whole prefill of `n` tokens on a grid of `slice`-row slices, from measured per-slice
/// times `samples` = `(end position, ms)` of one layer.
fn prefill_attention_seconds(samples: &[(f64, f64)], n: usize, slice: usize, layers: usize) -> f64 {
    let mut ms = 0.0;
    let mut end = slice;
    while end < n + slice {
        ms += interp(samples, end.min(n) as f64);
        end += slice;
    }
    ms * layers as f64 / 1000.0
}

#[test]
#[ignore]
fn bench_tree_long_context() {
    gpu!();
    let st = Stream::new().unwrap();
    let positions = env_list("REXL3_ATTN_LONG_POS", &[4096, 32768, 131072, 524288, 1048576]);
    let reps = env_list("REXL3_ATTN_LONG_REPS", &[5])[0];
    let max_pos = *positions.iter().max().unwrap();
    eprintln!("cache: {} tokens of one layer = {:.2} GiB of K and V", max_pos, max_pos as f64 * 1024.0 / (1u64 << 30) as f64);
    let (kv, table) = long_cache(max_pos, &st);
    let mut rng = Rng(0xA77E);
    let mut q_host = rng.vec(256 * QD, 1.0);
    // a decode-like softmax: some structure so that the running max moves (q scaled up a little)
    for x in q_host.iter_mut() {
        *x *= 2.0;
    }
    let mut out = DevBuf::zeroed(256 * QD * 4).unwrap();

    println!(
        "\n== exact reduction tree (attn_prefill / attn_decode), one layer, synthetic fp16 cache of {} keys ==",
        fmt_pos(max_pos)
    );
    println!("{:<10} {:>5} {:>9} {:>11} {:>9} {:>10} {:>10}", "call", "rows", "end pos", "ms", "TFLOP/s", "KV GB/s", "L2 GB/s");
    let mut by_rows: std::collections::BTreeMap<usize, Vec<(f64, f64)>> = Default::default();
    for &rows in &[64usize, 256] {
        for &end in &positions {
            if end < rows {
                continue;
            }
            let (slots, pos) = slice_rows(end, rows);
            let mut scratch = Scratch::new(rows, end.div_ceil(512)).unwrap();
            let idx = RowIndex::from_host(&slots, &pos).unwrap();
            let qv = DevBuf::from_slice(&q_host[..rows * QD]).unwrap();
            let ms = median_ms(&st, reps, || {
                na::attn_prefill(&qv, &kv, &table, &idx, &mut scratch, &mut out, &st).unwrap();
            });
            let fl = flops(&pos);
            // the slice's rows share one cache: the K and V bytes it needs are the deepest row's; each row's CTAs re-read them
            let uniq = end as f64 * 1024.0;
            let l2 = rows as f64 * end as f64 * 1024.0;
            println!(
                "{:<10} {:>5} {:>9} {:>11.3} {:>9.2} {:>10.1} {:>10.1}",
                "prefill",
                rows,
                fmt_pos(end),
                ms,
                fl / ms / 1e9,
                uniq / ms / 1e6,
                l2 / ms / 1e6
            );
            by_rows.entry(rows).or_default().push((end as f64, ms));
        }
    }
    for &rows in &[1usize, 4, 8] {
        for &end in positions.iter().filter(|&&p| p >= 131072) {
            let (slots, pos) = slice_rows(end, rows);
            let mut scratch = Scratch::new(rows, end.div_ceil(512)).unwrap();
            let idx = RowIndex::from_host(&slots, &pos).unwrap();
            let qv = DevBuf::from_slice(&q_host[..rows * QD]).unwrap();
            let ms = median_ms(&st, reps, || {
                na::attn_decode(&qv, &kv, &table, &idx, &mut scratch, &mut out, &st).unwrap();
            });
            let fl = flops(&pos);
            let uniq = end as f64 * 1024.0;
            println!(
                "{:<10} {:>5} {:>9} {:>11.3} {:>9.2} {:>10.1} {:>10}",
                "decode",
                rows,
                fmt_pos(end),
                ms,
                fl / ms / 1e9,
                uniq / ms / 1e6,
                "-"
            );
        }
    }

    // the projection of a 1M prefill: the measured 64-row slice times, interpolated in the end position, summed over the slices of the
    // grid and the 8 layers; the dense part is PFX's 280 tok/s (a 64-row slice is 227 ms, measured on node 1)
    if let Some(s64) = by_rows.get(&64) {
        let dense = |n: usize| n as f64 / 280.0;
        println!("\nprojection from the measured 64-row slices (8 layers, interpolated between the measured end positions):");
        println!("{:>10} {:>12} {:>14} {:>12}", "tokens", "dense s", "attention s", "total");
        for &n in &[32768usize, 131072, 524288, 1048576] {
            let att = prefill_attention_seconds(s64, n, 64, 8);
            let tot = dense(n) + att;
            println!("{:>10} {:>12.0} {:>14.0} {:>9.0} s = {:.2} h", fmt_pos(n), dense(n), att, tot, tot / 3600.0);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The flash path (`nemotron_attn_prefill`): accuracy against an f64 reference, row and chunk invariance, the position split, speed.
// ---------------------------------------------------------------------------------------------------------------------------------

use gb_cuda::nemotron_attn_prefill as nf;

fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits()) {
        panic!(
            "{what}: first bit difference at element {i} (row {}, head {}): {} vs {}",
            i / QD,
            (i % QD) / 128,
            got[i],
            want[i]
        );
    }
    assert!(got.iter().all(|x| x.is_finite()), "{what}: a non-finite output");
}

/// Caches for `slots` sequences of up to `tokens` tokens on a shuffled pool of pages (the table is not the identity).
fn layout(tokens: usize, slots: usize, seed: u64) -> (NemoKv, BlockTable) {
    let pages = tokens.div_ceil(PAGE);
    let n_blocks = pages * slots + 2;
    let mut ids: Vec<u32> = (0..n_blocks as u32).collect();
    let mut rng = Rng(seed | 1);
    for i in (1..ids.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        ids.swap(i, j);
    }
    let tables: Vec<Vec<u32>> = (0..slots).map(|s| ids[s * pages..(s + 1) * pages].to_vec()).collect();
    (NemoKv::new(n_blocks).unwrap(), BlockTable::from_host(&tables, pages).unwrap())
}

fn write(kv: &mut NemoKv, table: &BlockTable, slots: &[i32], pos: &[i32], k: &[f32], v: &[f32], st: &Stream) {
    let idx = RowIndex::from_host(slots, pos).unwrap();
    let kd = DevBuf::from_slice(k).unwrap();
    let vd = DevBuf::from_slice(v).unwrap();
    na::kv_write(kv, table, &idx, &kd, &vd, st).unwrap();
    st.sync().unwrap();
}

/// The flash kernel over every row of the call, whatever the positions.
fn flash(kv: &NemoKv, table: &BlockTable, slots: &[i32], pos: &[i32], q: &[f32], scratch: &mut Scratch, st: &Stream) -> Vec<f32> {
    let idx = RowIndex::from_host(slots, pos).unwrap();
    let qd = DevBuf::from_slice(q).unwrap();
    let mut out = DevBuf::zeroed(q.len() * 4).unwrap();
    let (rows, splits) = (pos.len(), nf::flash_splits(idx.max_chunks()));
    nf::attn_flash(&qd, kv, table, &idx, 0, rows, 0, splits, rows, scratch, &mut out, st).unwrap();
    st.sync().unwrap();
    out.to_vec::<f32>().unwrap()
}

/// The engine's attention entry (the one position rule) over every row of the call.
fn long(kv: &NemoKv, table: &BlockTable, slots: &[i32], pos: &[i32], q: &[f32], scratch: &mut Scratch, st: &Stream) -> Vec<f32> {
    let idx = RowIndex::from_host(slots, pos).unwrap();
    let qd = DevBuf::from_slice(q).unwrap();
    let mut out = DevBuf::zeroed(q.len() * 4).unwrap();
    nf::attn_long(&qd, kv, table, &idx, 0, scratch, &mut out, st).unwrap();
    st.sync().unwrap();
    out.to_vec::<f32>().unwrap()
}

/// The tree over every row of the call.
fn tree(kv: &NemoKv, table: &BlockTable, slots: &[i32], pos: &[i32], q: &[f32], scratch: &mut Scratch, st: &Stream) -> Vec<f32> {
    let idx = RowIndex::from_host(slots, pos).unwrap();
    let qd = DevBuf::from_slice(q).unwrap();
    let mut out = DevBuf::zeroed(q.len() * 4).unwrap();
    na::attn_prefill(&qd, kv, table, &idx, scratch, &mut out, st).unwrap();
    st.sync().unwrap();
    out.to_vec::<f32>().unwrap()
}

fn f16_round(x: f32) -> f32 {
    gb_exl3::f16::f16_to_f64(gb_exl3::f16::f64_to_f16(x as f64)) as f32
}

/// The attention of one row at position `p` in f64 over fp16-rounded K and V (`k` and `v` hold `[tokens][2][128]`): the exact softmax,
/// no probability rounding.
fn ref_row(q: &[f32], k: &[f32], v: &[f32], p: usize) -> Vec<f64> {
    let n = p + 1;
    let scale = 1.0 / (128f64).sqrt();
    let mut out = vec![0f64; QD];
    let mut s = vec![0f64; n];
    for h in 0..32 {
        let kvh = h / 16;
        let qh = &q[h * 128..(h + 1) * 128];
        let mut m = f64::NEG_INFINITY;
        for j in 0..n {
            let kr = &k[j * KVD + kvh * 128..][..128];
            let mut d = 0f64;
            for e in 0..128 {
                d += qh[e] as f64 * kr[e] as f64;
            }
            s[j] = d * scale;
            m = m.max(s[j]);
        }
        let mut l = 0f64;
        let o = &mut out[h * 128..(h + 1) * 128];
        for j in 0..n {
            let w = (s[j] - m).exp();
            l += w;
            let vr = &v[j * KVD + kvh * 128..][..128];
            for e in 0..128 {
                o[e] += w * vr[e] as f64;
            }
        }
        for x in o.iter_mut() {
            *x /= l;
        }
    }
    out
}

/// The worst `max |gpu - ref| / max |ref|` over the heads of a row.
fn worst_rel(got: &[f32], want: &[f64]) -> f64 {
    let mut worst = 0f64;
    for h in 0..32 {
        let w = &want[h * 128..(h + 1) * 128];
        let g = &got[h * 128..(h + 1) * 128];
        let scale = w.iter().fold(0f64, |a, &x| a.max(x.abs()));
        let err = g.iter().zip(w).fold(0f64, |a, (&x, &y)| a.max((x as f64 - y).abs()));
        worst = worst.max(err / scale.max(1e-30));
    }
    worst
}

#[test]
#[ignore]
fn flash_matches_the_f64_reference_within_fp16_terms() {
    gpu!();
    let st = Stream::new().unwrap();
    let tokens = 33_000;
    let (mut kv, table) = layout(tokens, 1, 0xF1A5);
    let mut rng = Rng(0xACC0);
    let k = rng.vec(tokens * KVD, 1.0);
    let v = rng.vec(tokens * KVD, 1.0);
    let all: Vec<i32> = (0..tokens as i32).collect();
    write(&mut kv, &table, &vec![0; tokens], &all, &k, &v, &st);
    let kr: Vec<f32> = k.iter().map(|&x| f16_round(x)).collect();
    let vr: Vec<f32> = v.iter().map(|&x| f16_round(x)).collect();
    // rows at and around the threshold, the split edges and tile edges, and deep
    let rows: Vec<usize> = vec![0, 63, 64, 8191, 8192, 8193, 8255, 8256, 12345, 16383, 16384, 16385, 24575, 24576, 30001, 32999];
    for q_scale in [1.0f32, 6.0, 16.0] {
        let q = rng.vec(rows.len() * QD, q_scale);
        let pos: Vec<i32> = rows.iter().map(|&p| p as i32).collect();
        let mut scratch = Scratch::new(64, 66).unwrap();
        let got = flash(&kv, &table, &vec![0; rows.len()], &pos, &q, &mut scratch, &st);
        assert!(got.iter().all(|x| x.is_finite()));
        // the exact tree on the same rows, for scale: its error is the fp16 rounding of P alone
        let mut tscratch = Scratch::new(rows.len(), 66).unwrap();
        let tgot = tree(&kv, &table, &vec![0; rows.len()], &pos, &q, &mut tscratch, &st);
        let (mut worst, mut tworst) = (0f64, 0f64);
        for (i, &p) in rows.iter().enumerate() {
            let want = ref_row(&q[i * QD..(i + 1) * QD], &kr, &vr, p);
            let e = worst_rel(&got[i * QD..(i + 1) * QD], &want);
            let te = worst_rel(&tgot[i * QD..(i + 1) * QD], &want);
            eprintln!("q scale {q_scale}: row at position {p}: max error / head max: flash {e:.3e}, tree {te:.3e}");
            worst = worst.max(e);
            tworst = tworst.max(te);
        }
        eprintln!("q scale {q_scale}: worst: flash {worst:.3e}, tree {tworst:.3e}");
        assert!(worst <= 1e-3, "flash vs f64 reference at q scale {q_scale}: {worst:.3e} of the head's max");
    }
}

/// The prompt rows `p0..p0 + n` of slot 0 through slicings of 1, 7, 64, 100 and 256 rows give the bits of one whole call, and so do
/// the same rows batched beside another sequence, on another page placement.
#[test]
#[ignore]
fn flash_is_row_and_chunk_invariant() {
    gpu!();
    let st = Stream::new().unwrap();
    let (p0, n) = (8400usize, 1024usize);
    let tokens = p0 + n;
    let mut rng = Rng(0x1A71);
    let q = rng.vec(n * QD, 2.0);
    let k0 = rng.vec(tokens * KVD, 1.0);
    let v0 = rng.vec(tokens * KVD, 1.0);
    let all: Vec<i32> = (0..tokens as i32).collect();

    let (mut kv, table) = layout(tokens, 2, 0xB10C);
    write(&mut kv, &table, &vec![0; tokens], &all, &k0, &v0, &st);
    // slot 1: a different sequence, the same positions
    let k1 = rng.vec(tokens * KVD, 1.0);
    let v1 = rng.vec(tokens * KVD, 1.0);
    write(&mut kv, &table, &vec![1; tokens], &all, &k1, &v1, &st);

    let pos: Vec<i32> = (p0..p0 + n).map(|p| p as i32).collect();
    let mut scratch = Scratch::new_serving(1024, 24).unwrap();
    let whole = flash(&kv, &table, &vec![0; n], &pos, &q, &mut scratch, &st);
    let again = flash(&kv, &table, &vec![0; n], &pos, &q, &mut scratch, &st);
    assert_bits_eq(&again, &whole, "the same call twice");
    // a scratch too small for the rows in one pass: the same rows in several passes
    let mut small = Scratch::new_serving(4, 24).unwrap();
    let passes = flash(&kv, &table, &vec![0; n], &pos, &q, &mut small, &st);
    assert_bits_eq(&passes, &whole, "in passes of a small scratch");

    for slice in [1usize, 7, 64, 100, 256] {
        let mut got = Vec::with_capacity(n * QD);
        let mut a = 0;
        while a < n {
            let b = (a + slice).min(n);
            got.extend(flash(&kv, &table, &vec![0; b - a], &pos[a..b], &q[a * QD..b * QD], &mut scratch, &st));
            a = b;
        }
        assert_bits_eq(&got, &whole, &format!("slices of {slice} rows vs one call"));
        eprintln!("slices of {slice} rows: bit-identical to the whole call");
    }

    // batched with another sequence: the slot-1 rows (other keys, other queries) interleaved in runs of 5 and 3
    let q1 = rng.vec(n * QD, 2.0);
    let mut slots = Vec::new();
    let mut pp = Vec::new();
    let mut qq = Vec::new();
    let (mut a, mut b) = (0usize, 0usize);
    let mut which = 0;
    while a < n || b < n {
        let take = if which % 2 == 0 { 5 } else { 3 };
        if which % 2 == 0 {
            let e = (a + take).min(n);
            for r in a..e {
                slots.push(0);
                pp.push(pos[r]);
                qq.extend_from_slice(&q[r * QD..(r + 1) * QD]);
            }
            a = e;
        } else {
            let e = (b + take).min(n);
            for r in b..e {
                slots.push(1);
                pp.push(pos[r]);
                qq.extend_from_slice(&q1[r * QD..(r + 1) * QD]);
            }
            b = e;
        }
        which += 1;
    }
    let mut big = Scratch::new_serving(2 * n, 24).unwrap();
    let mixed = flash(&kv, &table, &slots, &pp, &qq, &mut big, &st);
    let solo1 = flash(&kv, &table, &vec![1; n], &pos, &q1, &mut scratch, &st);
    let (mut ia, mut ib) = (0usize, 0usize);
    for (r, &s) in slots.iter().enumerate() {
        let row = &mixed[r * QD..(r + 1) * QD];
        let want = if s == 0 {
            ia += 1;
            &whole[(ia - 1) * QD..ia * QD]
        } else {
            ib += 1;
            &solo1[(ib - 1) * QD..ib * QD]
        };
        assert_bits_eq(row, want, &format!("batched row {r} of slot {s}"));
    }
    eprintln!("rows batched with another sequence: bit-identical to their solo runs");

    // another page placement of the same keys: the bits do not depend on where the pages sit
    let (mut kv2, table2) = layout(tokens, 1, 0xDEAD);
    write(&mut kv2, &table2, &vec![0; tokens], &all, &k0, &v0, &st);
    let moved = flash(&kv2, &table2, &vec![0; n], &pos, &q, &mut scratch, &st);
    assert_bits_eq(&moved, &whole, "another page placement");
}

/// The engine's entry (`attn_prefill_long`) over a prompt that crosses the threshold: the slicing is invisible, the rows below the
/// threshold are the tree's bits and the rows at and above it are the flash kernel's.
#[test]
#[ignore]
fn the_position_split_is_invisible_to_chunking() {
    gpu!();
    let st = Stream::new().unwrap();
    let (p0, n) = (7000usize, 2400usize);
    let tokens = p0 + n;
    let mut rng = Rng(0x5911);
    let q = rng.vec(n * QD, 2.0);
    let k = rng.vec(tokens * KVD, 1.0);
    let v = rng.vec(tokens * KVD, 1.0);
    let all: Vec<i32> = (0..tokens as i32).collect();
    let (mut kv, table) = layout(tokens, 1, 0x5917);
    write(&mut kv, &table, &vec![0; tokens], &all, &k, &v, &st);
    let pos: Vec<i32> = (p0..p0 + n).map(|p| p as i32).collect();
    let mut scratch = Scratch::new(1200, tokens.div_ceil(512)).unwrap();
    let whole = long(&kv, &table, &vec![0; n], &pos, &q, &mut scratch, &st);
    // a scratch of 64 rows serves any slice: the tree run goes in passes of 64 rows, the flash run in passes of what fits
    let mut small = Scratch::new(64, tokens.div_ceil(512)).unwrap();
    let passes = long(&kv, &table, &vec![0; n], &pos, &q, &mut small, &st);
    assert_bits_eq(&passes, &whole, "a 64-row scratch");
    let split = nf::FLASH_FROM_POS - p0;
    let t = tree(&kv, &table, &vec![0; split], &pos[..split], &q[..split * QD], &mut scratch, &st);
    assert_bits_eq(&whole[..split * QD], &t, "rows below the threshold are the tree's");
    let f = flash(&kv, &table, &vec![0; n - split], &pos[split..], &q[split * QD..], &mut scratch, &st);
    assert_bits_eq(&whole[split * QD..], &f, "rows at and above the threshold are the flash kernel's");
    // and they differ from the tree there (it is another arithmetic)
    let tdeep = tree(&kv, &table, &vec![0; 64], &pos[split..split + 64], &q[split * QD..(split + 64) * QD], &mut scratch, &st);
    assert!(
        tdeep.iter().zip(&whole[split * QD..(split + 64) * QD]).any(|(a, b)| a.to_bits() != b.to_bits()),
        "flash == tree bitwise?"
    );
    let worst = (0..64)
        .map(|r| {
            let (a, b) = (&tdeep[r * QD..(r + 1) * QD], &whole[(split + r) * QD..(split + r + 1) * QD]);
            let want: Vec<f64> = a.iter().map(|&x| x as f64).collect();
            worst_rel(b, &want)
        })
        .fold(0f64, f64::max);
    eprintln!("flash vs tree on the first 64 rows at and above the threshold: worst {worst:.3e} of the head's max");
    assert!(worst <= 1e-3);

    // slicings, with slice edges at, before and after the threshold row
    for slice in [1usize, 7, 33, 64, 100, 256] {
        let mut got = Vec::with_capacity(n * QD);
        let mut a = 0;
        while a < n {
            let b = (a + slice).min(n);
            got.extend(long(&kv, &table, &vec![0; b - a], &pos[a..b], &q[a * QD..b * QD], &mut scratch, &st));
            a = b;
        }
        assert_bits_eq(&got, &whole, &format!("slices of {slice} rows vs one call"));
        eprintln!("slices of {slice} rows through the position split: bit-identical to the whole call");
    }
}

/// A call whose rows are all below the threshold is the tree's call, bit for bit and launch for launch.
#[test]
#[ignore]
fn below_the_threshold_the_entry_is_the_tree() {
    gpu!();
    let st = Stream::new().unwrap();
    let n = 300;
    let tokens = nf::FLASH_FROM_POS;
    let mut rng = Rng(0xB010);
    let q = rng.vec(n * QD, 2.0);
    let k = rng.vec(tokens * KVD, 1.0);
    let v = rng.vec(tokens * KVD, 1.0);
    let all: Vec<i32> = (0..tokens as i32).collect();
    let (mut kv, table) = layout(tokens, 1, 0xB011);
    write(&mut kv, &table, &vec![0; tokens], &all, &k, &v, &st);
    let pos: Vec<i32> = (tokens - n..tokens).map(|p| p as i32).collect();
    let mut scratch = Scratch::new(n, tokens.div_ceil(512)).unwrap();
    let a = long(&kv, &table, &vec![0; n], &pos, &q, &mut scratch, &st);
    let b = tree(&kv, &table, &vec![0; n], &pos, &q, &mut scratch, &st);
    assert_bits_eq(&a, &b, "below the threshold");
    // a scratch narrower than the call (the model's: a window's 64 rows against a prefill slice of 256): the same rows in passes
    let mut narrow = Scratch::new(64, tokens.div_ceil(512)).unwrap();
    let c = long(&kv, &table, &vec![0; n], &pos, &q, &mut narrow, &st);
    assert_bits_eq(&c, &b, "below the threshold, in passes of a 64-row scratch");
}

/// The one rule, by entry shape: the same rows through a one-call prefill, single-row decode calls, verify windows of 2 to 24 rows,
/// calls shared with another sequence and launches built for a graph bucket (more chunks than the rows need) are bit-identical, row
/// by row, across the threshold; the rows below it are the tree's bits and the rows at and above it the flash kernel's.
#[test]
#[ignore]
fn one_rule_decode_window_and_prefill_give_a_row_the_same_bits() {
    gpu!();
    let st = Stream::new().unwrap();
    let t = nf::FLASH_FROM_POS;
    let tokens = 2 * t + 4000;
    let mut rng = Rng(0xD0D0);
    let (k0, v0) = (rng.vec(tokens * KVD, 1.0), rng.vec(tokens * KVD, 1.0));
    let (k1, v1) = (rng.vec(tokens * KVD, 1.0), rng.vec(tokens * KVD, 1.0));
    let all: Vec<i32> = (0..tokens as i32).collect();
    let (mut kv, table) = layout(tokens, 2, 0xD0D1);
    write(&mut kv, &table, &vec![0; tokens], &all, &k0, &v0, &st);
    write(&mut kv, &table, &vec![1; tokens], &all, &k1, &v1, &st);
    // rows around the threshold, the split edge of the second split and deep
    let mut rows: Vec<usize> = (t - 12..t + 12).collect();
    rows.extend([2 * t - 1, 2 * t, 2 * t + 1, tokens - 1]);
    let n = rows.len();
    let pos: Vec<i32> = rows.iter().map(|&p| p as i32).collect();
    let q = rng.vec(n * QD, 2.0);
    let mut scratch = Scratch::new(64, tokens.div_ceil(512)).unwrap();

    // the reference: one call; below the threshold the tree's bits, above it the flash kernel's
    let whole = long(&kv, &table, &vec![0; n], &pos, &q, &mut scratch, &st);
    let below = rows.iter().filter(|&&p| p < t).count();
    let t_out = tree(&kv, &table, &vec![0; below], &pos[..below], &q[..below * QD], &mut scratch, &st);
    assert_bits_eq(&whole[..below * QD], &t_out, "rows below the threshold are the tree's");
    let f_out = flash(&kv, &table, &vec![0; n - below], &pos[below..], &q[below * QD..], &mut scratch, &st);
    assert_bits_eq(&whole[below * QD..], &f_out, "rows at and above the threshold are the flash kernel's");

    // decode: every row alone
    for r in 0..n {
        let one = long(&kv, &table, &[0], &pos[r..r + 1], &q[r * QD..(r + 1) * QD], &mut scratch, &st);
        assert_bits_eq(&one, &whole[r * QD..(r + 1) * QD], &format!("row at {} alone", rows[r]));
    }
    eprintln!("{n} rows, each alone: bit-identical to the one call");
    // verify windows: every window of K rows of the run (consecutive positions), K = 2, 3, 5, 8, 9, 24
    for kk in [2usize, 3, 5, 8, 9, 24] {
        for a in 0..=(24 - kk) {
            let w = long(&kv, &table, &vec![0; kk], &pos[a..a + kk], &q[a * QD..(a + kk) * QD], &mut scratch, &st);
            assert_bits_eq(&w, &whole[a * QD..(a + kk) * QD], &format!("window of {kk} rows from {}", rows[a]));
        }
    }
    eprintln!("windows of 2..24 rows across the threshold: bit-identical to the one call");
    // another sequence in the call (other keys and queries), its rows before, between and after: slot 0 rows keep their bits
    let q1 = rng.vec(n * QD, 2.0);
    let (mut slots, mut pp, mut qq) = (Vec::new(), Vec::new(), Vec::new());
    for (i, r) in (0..24usize).enumerate() {
        if i % 5 == 0 {
            // a slot-1 row at another position
            slots.push(1);
            pp.push(pos[(r + 7) % 24]);
            qq.extend_from_slice(&q1[r * QD..(r + 1) * QD]);
        }
        slots.push(0);
        pp.push(pos[r]);
        qq.extend_from_slice(&q[r * QD..(r + 1) * QD]);
    }
    let mixed = long(&kv, &table, &slots, &pp, &qq, &mut scratch, &st);
    let (mut j, mut other) = (0usize, 0usize);
    for (r, &s) in slots.iter().enumerate() {
        if s == 0 {
            assert_bits_eq(
                &mixed[r * QD..(r + 1) * QD],
                &whole[j * QD..(j + 1) * QD],
                &format!("slot 0 row {j} beside another sequence"),
            );
            j += 1;
        } else {
            let want = long(&kv, &table, &[1], &pp[r..r + 1], &qq[r * QD..(r + 1) * QD], &mut scratch, &st);
            assert_bits_eq(&mixed[r * QD..(r + 1) * QD], &want, &format!("slot 1 row {other} alone vs in the call"));
            other += 1;
        }
    }
    eprintln!("rows beside another sequence: bit-identical to their solo calls");
    // a launch built for a graph bucket (the chunk count rounded up to a power of two, here 64 chunks = 32,768 keys, and the cap)
    for bucket in [32usize, 64, 1024] {
        for (a, kk) in [(0usize, 1usize), (11, 5), (8, 8), (20, 4)] {
            let mut idx = RowIndex::from_host(&vec![0; kk], &pos[a..a + kk]).unwrap();
            idx.set_min_chunks(bucket.min(scratch.max_chunks()));
            let qd = DevBuf::from_slice(&q[a * QD..(a + kk) * QD]).unwrap();
            let mut out = DevBuf::zeroed(kk * QD * 4).unwrap();
            nf::attn_long(&qd, &kv, &table, &idx, 1, &mut scratch, &mut out, &st).unwrap();
            st.sync().unwrap();
            assert_bits_eq(
                &out.to_vec::<f32>().unwrap(),
                &whole[a * QD..(a + kk) * QD],
                &format!("bucket {bucket}, {kk} rows from {}", rows[a]),
            );
        }
    }
    eprintln!("launches built for the chunk buckets 32, 64 and 1024: bit-identical");
}

/// The decode shape: 1 to 8 rows of one sequence at the end of a deep cache, per layer: the tree's `attn_decode` against the one rule's
/// `attn_long` (the flash kernel above 8,192), and the same launch built for a graph bucket twice the rows' chunks (what a captured
/// window runs just above a power of two).
#[test]
#[ignore]
fn bench_decode_shapes() {
    gpu!();
    let st = Stream::new().unwrap();
    let ends = env_list("REXL3_ATTN_LONG_POS", &[131072, 524288, 1048576]);
    let reps = env_list("REXL3_ATTN_LONG_REPS", &[9])[0];
    let max_pos = *ends.iter().max().unwrap() + 64;
    let (kv, table) = long_cache(max_pos, &st);
    let mut rng = Rng(0xDEC0);
    let q_host = rng.vec(8 * QD, 2.0);
    let mut out = DevBuf::zeroed(8 * QD * 4).unwrap();
    let mut scratch = Scratch::new(64, max_pos.div_ceil(512)).unwrap();
    println!("\n== decode shapes, one layer, synthetic fp16 cache of {} keys ==", fmt_pos(max_pos));
    println!("{:<22} {:>5} {:>9} {:>9} {:>9}", "call", "rows", "end pos", "ms", "KV GB/s");
    for &end in &ends {
        for &rows in &[1usize, 2, 4, 5, 8] {
            let (slots, pos) = slice_rows(end, rows);
            let qv = DevBuf::from_slice(&q_host[..rows * QD]).unwrap();
            let idx = RowIndex::from_host(&slots, &pos).unwrap();
            let t_tree = median_ms(&st, reps, || {
                na::attn_decode(&qv, &kv, &table, &idx, &mut scratch, &mut out, &st).unwrap();
            });
            let t_long = median_ms(&st, reps, || {
                nf::attn_long(&qv, &kv, &table, &idx, 1, &mut scratch, &mut out, &st).unwrap();
            });
            let gbs = |ms: f64| end as f64 * 1024.0 / ms / 1e6;
            println!("{:<22} {:>5} {:>9} {:>9.3} {:>9.1}", "tree attn_decode", rows, fmt_pos(end), t_tree, gbs(t_tree));
            println!("{:<22} {:>5} {:>9} {:>9.3} {:>9.1}", "one rule attn_long", rows, fmt_pos(end), t_long, gbs(t_long));
        }
    }
    // the padded launch: rows just above a power of two of chunks run in the next bucket
    for &end in &ends[..1] {
        let end = end + 8;
        for &rows in &[1usize, 5, 8] {
            let (slots, pos) = slice_rows(end, rows);
            let qv = DevBuf::from_slice(&q_host[..rows * QD]).unwrap();
            let mut idx = RowIndex::from_host(&slots, &pos).unwrap();
            let natural = median_ms(&st, reps, || {
                nf::attn_long(&qv, &kv, &table, &idx, 1, &mut scratch, &mut out, &st).unwrap();
            });
            let bucket = idx.real_chunks().next_power_of_two();
            idx.set_min_chunks(bucket);
            let padded = median_ms(&st, reps, || {
                nf::attn_long(&qv, &kv, &table, &idx, 1, &mut scratch, &mut out, &st).unwrap();
            });
            println!(
                "{:<22} {:>5} {:>9} {:>9.3} natural, {:.3} built for {} chunks",
                "one rule, bucket pad",
                rows,
                fmt_pos(end),
                natural,
                padded,
                bucket
            );
        }
    }
}

#[test]
#[ignore]
fn bench_flash_long_context() {
    gpu!();
    let st = Stream::new().unwrap();
    let positions = env_list("REXL3_ATTN_LONG_POS", &[4096, 32768, 131072, 524288, 1048576]);
    let reps = env_list("REXL3_ATTN_LONG_REPS", &[5])[0];
    let max_pos = *positions.iter().max().unwrap();
    let (kv, table) = long_cache(max_pos, &st);
    let mut rng = Rng(0xA77E);
    let mut q_host = rng.vec(256 * QD, 2.0);
    for x in q_host.iter_mut() {
        *x *= 1.0;
    }
    let mut out = DevBuf::zeroed(256 * QD * 4).unwrap();
    // the model's scratch: a window's 64 rows, the tree for 16 chunks and the flash path for the context's splits; a call of more rows
    // runs in passes
    let mut scratch = Scratch::new_serving(64, max_pos.div_ceil(512)).unwrap();
    println!(
        "\n== flash path (attn_flash), one layer, synthetic fp16 cache of {} keys, path id {} ==",
        fmt_pos(max_pos),
        nf::FLASH_PATH_ID
    );
    println!("{:<10} {:>5} {:>9} {:>11} {:>9} {:>10} {:>10}", "call", "rows", "end pos", "ms", "TFLOP/s", "KV GB/s", "L2 GB/s");
    let mut by_rows: std::collections::BTreeMap<usize, Vec<(f64, f64)>> = Default::default();
    for &rows in &[1usize, 8, 64, 256] {
        for &end in &positions {
            if end < rows {
                continue;
            }
            let (slots, pos) = slice_rows(end, rows);
            let idx = RowIndex::from_host(&slots, &pos).unwrap();
            let qv = DevBuf::from_slice(&q_host[..rows * QD]).unwrap();
            let ms = median_ms(&st, reps, || {
                nf::attn_flash(
                    &qv,
                    &kv,
                    &table,
                    &idx,
                    0,
                    rows,
                    0,
                    nf::flash_splits(idx.max_chunks()),
                    rows,
                    &mut scratch,
                    &mut out,
                    &st,
                )
                .unwrap();
            });
            let fl = flops(&pos);
            let uniq = end as f64 * 1024.0;
            let l2 = (rows.div_ceil(8)) as f64 * end as f64 * 1024.0;
            println!(
                "{:<10} {:>5} {:>9} {:>11.3} {:>9.2} {:>10.1} {:>10.1}",
                "flash",
                rows,
                fmt_pos(end),
                ms,
                fl / ms / 1e9,
                uniq / ms / 1e6,
                l2 / ms / 1e6
            );
            by_rows.entry(rows).or_default().push((end as f64, ms));
        }
    }
    println!("(L2 GB/s: K and V bytes the CTAs fetch, one fetch per 8-row block)");
    // projection: the tree below the threshold (its measured time per slice, from bench_tree_long_context at 4k: 0.82 ms a 64-row
    // slice, linear in the keys), the flash path above; 8 layers; PFX's 280 tok/s dense
    if let Some(s64) = by_rows.get(&64) {
        let tree_ms = |end: f64| 0.819 * end / 4096.0;
        let mut samples: Vec<(f64, f64)> = vec![(0.0, 0.0)];
        samples.extend(s64.iter().copied().filter(|s| s.0 > nf::FLASH_FROM_POS as f64));
        let dense = |n: usize| n as f64 / 280.0;
        println!(
            "\nprojection (8 layers; tree below {} keys at 0.819 ms x end/4k, flash above, measured 64-row slices interpolated):",
            nf::FLASH_FROM_POS
        );
        println!("{:>10} {:>12} {:>14} {:>12}", "tokens", "dense s", "attention s", "total");
        for &n in &[32768usize, 131072, 524288, 1048576] {
            let mut ms = 0.0;
            let mut end = 64usize;
            while end < n + 64 {
                let e = end.min(n);
                ms += if e <= nf::FLASH_FROM_POS { tree_ms(e as f64) } else { interp(&samples, e as f64).max(0.0) };
                end += 64;
            }
            let att = ms * 8.0 / 1000.0;
            let tot = dense(n) + att;
            println!("{:>10} {:>12.0} {:>14.0} {:>9.0} s = {:.2} h", fmt_pos(n), dense(n), att, tot, tot / 3600.0);
        }
    }
}

/// Lane S3, contract P2: inside a prefill scope with the opt-in, `attn_long` serves the rows below 8,192 by the flash kernel. The rows
/// get the flash kernel's bits (and the flash kernel's error against the f64 reference, printed next to the tree's); outside a scope
/// the call is the tree's bit for bit (decode and verify windows keep the one rule). Then the speed of a 256-row slice ending at 1k to
/// 8k keys: tree against flash.
#[test]
#[ignore]
fn p2_prefill_rows_below_the_threshold_take_the_flash_kernel() {
    gpu!();
    use gb_cuda::nemotron_dense_mma as ndm;
    let st = Stream::new().unwrap();
    let tokens = nf::FLASH_FROM_POS;
    let mut rng = Rng(0x5352);
    let k = rng.vec(tokens * KVD, 1.0);
    let v = rng.vec(tokens * KVD, 1.0);
    let all: Vec<i32> = (0..tokens as i32).collect();
    let (mut kv, table) = layout(tokens, 1, 0x5353);
    write(&mut kv, &table, &vec![0; tokens], &all, &k, &v, &st);
    let kr: Vec<f32> = k.iter().map(|&x| f16_round(x)).collect();
    let vr: Vec<f32> = v.iter().map(|&x| f16_round(x)).collect();
    let n = 256;
    let mut scratch = Scratch::new(64, 66).unwrap();
    ndm::set_enabled(true);
    for end in [n, 3000, tokens] {
        for q_scale in [1.0f32, 6.0] {
            let q = rng.vec(n * QD, q_scale);
            let (slots, pos) = slice_rows(end, n);
            let p2 = {
                let _scope = ndm::PrefillScope::enter();
                long(&kv, &table, &slots, &pos, &q, &mut scratch, &st)
            };
            let f = flash(&kv, &table, &slots, &pos, &q, &mut scratch, &st);
            assert_bits_eq(&p2, &f, "P2 rows below the threshold: the flash kernel's bits");
            // a short tail slice inside the scope: the same rows' bits (P2(a): the width never picks the arithmetic)
            let tail = {
                let _scope = ndm::PrefillScope::enter();
                long(&kv, &table, &slots[n - 17..], &pos[n - 17..], &q[(n - 17) * QD..], &mut scratch, &st)
            };
            assert_bits_eq(&tail, &p2[(n - 17) * QD..], "P2: a 17-row tail slice");
            let t = long(&kv, &table, &slots, &pos, &q, &mut scratch, &st);
            let tr = tree(&kv, &table, &slots, &pos, &q, &mut Scratch::new(n, 66).unwrap(), &st);
            assert_bits_eq(&t, &tr, "no scope: the tree's bits");
            let (mut ef, mut et) = (0f64, 0f64);
            for i in (0..n).step_by(37) {
                let want = ref_row(&q[i * QD..(i + 1) * QD], &kr, &vr, pos[i] as usize);
                ef = ef.max(worst_rel(&f[i * QD..(i + 1) * QD], &want));
                et = et.max(worst_rel(&tr[i * QD..(i + 1) * QD], &want));
            }
            eprintln!(
                "slice of {n} ending at {end}, q scale {q_scale}: worst max error / head max: flash {ef:.3e}, tree {et:.3e}"
            );
            assert!(ef <= 1e-3, "flash below the threshold off the f64 reference by {ef:e}");
        }
    }
    ndm::set_enabled(false);
    println!("| 256-row slice ending at | tree ms | tree TFLOP/s | flash ms | flash TFLOP/s |");
    println!("|---:|---:|---:|---:|---:|");
    for end in [1024usize, 2048, 4096, 8192] {
        let q = DevBuf::from_slice(&rng.vec(n * QD, 2.0)).unwrap();
        let (slots, pos) = slice_rows(end, n);
        let idx = RowIndex::from_host(&slots, &pos).unwrap();
        let mut out = DevBuf::zeroed(n * QD * 4).unwrap();
        let mut ts = Scratch::new(n, 66).unwrap();
        let tt = median_ms(&st, 7, || na::attn_prefill(&q, &kv, &table, &idx, &mut ts, &mut out, &st).unwrap());
        let splits = nf::flash_splits(idx.max_chunks());
        let tf =
            median_ms(&st, 7, || nf::attn_flash(&q, &kv, &table, &idx, 0, n, 0, splits, n, &mut scratch, &mut out, &st).unwrap());
        let fl = flops(&pos);
        println!("| {end} | {tt:.3} | {:.1} | {tf:.3} | {:.1} |", fl / tt / 1e9, fl / tf / 1e9);
    }
}
