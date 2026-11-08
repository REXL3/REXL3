//! The int8 KV cache of the Nemotron 3.5 Super VL attention layers (`--kv-dtype int8`, `csrc/nemotron_kv8.cuh`,
//! `docs/NEMOTRON_KV_INT8.md`). Synthetic data, no model. On a GB10 under the device lock, release build:
//!
//!     cargo test --release -p gb-cuda --test nemotron_kv_int8 -- --include-ignored --test-threads=1 --nocapture
//!
//! * `the_quantiser_*`: the device's page bytes (values and exponents) equal the host model `nemotron_attn::kv8` bit for bit.
//! * `int8_attention_is_the_fp16_attention_of_the_dequantised_cache`: the tree and the flash path over an int8 cache give, for every
//!   row (decode rows, windows, prefill slices of 1 to 256 rows, two sequences, both sides of the 8,192 threshold), the bits the fp16
//!   kernels give over an fp16 cache holding the dequantised values. So every invariance the fp16 path has (decode == window ==
//!   prefill, chunked == whole, placement) holds for int8, and int8 differs from fp16 by the quantisation of K and V alone.
//! * `int8_error_against_the_f64_reference`: what the quantisation costs in the attention output, against the f64 reference of the
//!   UNquantised K and V, beside the fp16 path's error on the same rows (printed; bounded).
//! * `bench_kv_int8_*`: decode at depth and prefill slices, fp16 against int8, one layer
//!   (`REXL3_ATTN_LONG_POS`, `REXL3_ATTN_LONG_REPS`).
#![cfg(not(gb_no_cuda))]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use gb_cuda::attention::BlockTable;
use gb_cuda::moe::EventTimer;
use gb_cuda::nemotron_attn::{self as na, kv8, KvDtype, NemoKv, RowIndex, Scratch};
use gb_cuda::nemotron_attn_prefill as nf;
use gb_cuda::{device_available, DevBuf, Stream};

const QD: usize = 32 * 128;
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

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f32(&mut self, scale: f32) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32(scale)).collect()
    }
}

/// K-like rows: a per-token magnitude drawn over six decades, a few outlier channels per kv head 20x the rest, and some zero rows.
fn k_like(rng: &mut Rng, tokens: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(tokens * KVD);
    for t in 0..tokens {
        let mag = 10f32.powf(rng.f32(3.0));
        for h in 0..2 {
            for d in 0..128 {
                let outlier = d % 37 == 5 + h;
                let x = rng.f32(mag) * if outlier { 20.0 } else { 1.0 };
                out.push(if t % 97 == 13 { 0.0 } else { x });
            }
        }
    }
    out
}

/// The env list or a default.
fn env_list(name: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(name) {
        Ok(v) => v.split(',').map(|x| x.trim().parse().unwrap()).collect(),
        Err(_) => default.to_vec(),
    }
}

/// Caches of `dtype` for `slots` sequences of up to `tokens` tokens on a shuffled pool of pages.
fn layout(tokens: usize, slots: usize, seed: u64, dtype: KvDtype) -> (NemoKv, BlockTable, Vec<Vec<u32>>) {
    let pages = tokens.div_ceil(PAGE);
    let n_blocks = pages * slots + 2;
    let mut ids: Vec<u32> = (0..n_blocks as u32).collect();
    let mut rng = Rng(seed | 1);
    for i in (1..ids.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        ids.swap(i, j);
    }
    let tables: Vec<Vec<u32>> = (0..slots).map(|s| ids[s * pages..(s + 1) * pages].to_vec()).collect();
    // zeroed, as the model's pool is (a masked key is never NaN)
    let bytes = n_blocks * dtype.page_bytes();
    let kv = NemoKv { k: DevBuf::zeroed(bytes).unwrap(), v: DevBuf::zeroed(bytes).unwrap(), n_blocks, dtype };
    (kv, BlockTable::from_host(&tables, pages).unwrap(), tables)
}

fn write(kv: &mut NemoKv, table: &BlockTable, slots: &[i32], pos: &[i32], k: &[f32], v: &[f32], st: &Stream) {
    for c in (0..pos.len()).step_by(8192) {
        let e = (c + 8192).min(pos.len());
        let idx = RowIndex::from_host(&slots[c..e], &pos[c..e]).unwrap();
        let kd = DevBuf::from_slice(&k[c * KVD..e * KVD]).unwrap();
        let vd = DevBuf::from_slice(&v[c * KVD..e * KVD]).unwrap();
        na::kv_write(kv, table, &idx, &kd, &vd, st).unwrap();
        st.sync().unwrap();
    }
}

/// The host model's dequantised values of rows `x` (`[tokens][2][128]`): what an int8 cache holds, as fp32 (exact fp16 values).
fn dequantised(x: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(128) {
        let (e, b) = kv8::quantise_row_i8(row);
        out.extend(b.iter().map(|&b| kv8::value(b, e)));
    }
    out
}

fn long(kv: &NemoKv, table: &BlockTable, slots: &[i32], pos: &[i32], q: &[f32], scratch: &mut Scratch, st: &Stream) -> Vec<f32> {
    let idx = RowIndex::from_host(slots, pos).unwrap();
    let qd = DevBuf::from_slice(q).unwrap();
    let mut out = DevBuf::zeroed(q.len() * 4).unwrap();
    nf::attn_long(&qd, kv, table, &idx, 0, scratch, &mut out, st).unwrap();
    st.sync().unwrap();
    out.to_vec::<f32>().unwrap()
}

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

#[test]
fn the_host_quantiser_and_the_flag() {
    assert_eq!(KvDtype::Int8.page_bytes(), 66_048);
    assert_eq!(KvDtype::F16.page_bytes(), 131_072);
    assert_eq!(KvDtype::F16.token_bytes(), 1024);
    assert_eq!(KvDtype::parse("FP16"), Ok(KvDtype::F16));
    assert!(KvDtype::parse("bf16").is_err());
    // fp8 was evaluated and rejected: a named refusal, never a silent fallback
    for v in ["fp8", "e4m3", "FP8"] {
        assert_eq!(KvDtype::parse(v), Err(na::FP8_REJECTED.to_string()), "{v}");
    }
    assert_eq!(KvDtype::parse("int8"), Ok(KvDtype::Int8));
    assert_eq!(KvDtype::Int8.token_bytes(), 516);
    // int8: the exponent puts the row maximum in (63.5, 127]; the dequantised values are exact fp16 values at both ends of the range
    for amax in [1e-3f32, 0.75, 1.0, 1.984375, 1.99, 127.0, 127.5, 5000.0] {
        let e = kv8::exponent_i8(amax);
        let r = amax / 2f32.powi(e);
        assert!(r <= 127.0 && r > 63.5, "amax {amax}: e {e}, {r}");
    }
    for e in [kv8::I8_EMIN as i8, 0, kv8::I8_EMAX as i8] {
        for b in 0u8..=255 {
            if b != 0x80 {
                let _ = kv8::dequant_i8_f16_bits(b, e);
            }
        }
    }
    let (e, q) = kv8::quantise_row_i8(&[2.5f32; 128]);
    assert_eq!((e, q[0] as i8), (-5, 80)); // 2.5 x 32 = 80
    let (_, q) = kv8::quantise_row_i8(&(0..128).map(|i| if i == 0 { 127.0 } else { 0.5 }).collect::<Vec<f32>>());
    assert_eq!((q[0] as i8, q[1] as i8), (127, 0)); // 0.5 rounds to the even 0
}

#[test]
fn the_quantiser_on_the_device_is_the_host_model_bit_for_bit() {
    gpu!();
    quantiser_case(KvDtype::Int8);
}

fn quantiser_case(dtype: KvDtype) {
    let st = Stream::new().unwrap();
    let tokens = 3 * PAGE + 77;
    let (mut kv, table, tables) = layout(tokens, 2, 0x8E4, dtype);
    let mut rng = Rng(0xF8F8);
    let mut k = k_like(&mut rng, tokens);
    let mut v = rng.vec(tokens * KVD, 3.0);
    // edge rows: a row maximum exactly 448 x 2^e, just above it, huge, tiny, ties between E4M3 neighbours, a single non-zero value
    for (t, val) in
        [(1usize, 448.0f32), (2, 449.0), (3, 6.0e4), (4, 1e-6), (5, 1.0625), (6, -3.5), (7, 127.0), (8, 127.5), (9, 0.75)]
    {
        for d in 0..KVD {
            k[t * KVD + d] = if d % 3 == 0 { val } else { val * (d as f32 / 300.0) };
            v[t * KVD + d] = if d == 7 { val } else { 0.0 };
        }
    }
    let all: Vec<i32> = (0..tokens as i32).collect();
    // slot 1, written in two calls (the order of writes is not part of the bytes)
    write(&mut kv, &table, &vec![1; tokens], &all, &k, &v, &st);
    let pages_k = kv.k.to_vec::<u8>().unwrap();
    let pages_v = kv.v.to_vec::<u8>().unwrap();
    let blocks: Vec<u32> = tables[1].clone();
    let mut checked = 0;
    for t in 0..tokens {
        let pg = blocks[t / PAGE] as usize * kv8::PAGE_BYTES;
        for h in 0..2 {
            for (src, pages, what) in [(&k, &pages_k, "K"), (&v, &pages_v, "V")] {
                let row = &src[t * KVD + h * 128..][..128];
                let (e, b) = kv8::quantise_row_i8(row);
                let got_e = pages[pg + kv8::EXP_OFF + (t % PAGE) * 2 + h] as i8;
                assert_eq!(got_e, e, "{what} exponent of token {t}, head {h}");
                let got = &pages[pg + (t % PAGE) * KVD + h * 128..][..128];
                if let Some(d) = got.iter().zip(&b).position(|(a, b)| a != b) {
                    panic!(
                        "{what} token {t} head {h} dim {d}: device {:#04x}, host {:#04x} (x = {})",
                        got[d],
                        b[d],
                        src[t * KVD + h * 128 + d]
                    );
                }
                checked += 1;
            }
        }
    }
    eprintln!("{dtype:?}: {checked} (token, kv head, K|V) rows: device == host model");
}

#[test]
fn int8_attention_is_the_fp16_attention_of_the_dequantised_cache() {
    gpu!();
    equivalence_case(KvDtype::Int8);
}

fn equivalence_case(dtype: KvDtype) {
    let st = Stream::new().unwrap();
    let tokens = 20_000;
    let mut rng = Rng(0xD0D0);
    let k = k_like(&mut rng, tokens);
    let v = rng.vec(tokens * KVD, 1.0);
    let (kq, vq) = (dequantised(&k), dequantised(&v));
    let all: Vec<i32> = (0..tokens as i32).collect();
    // two sequences: slot 0 the data, slot 1 the data reversed (rows of both in one call)
    let rev = |x: &[f32]| -> Vec<f32> { x.chunks(KVD).rev().flatten().copied().collect() };
    let (mut k8, table, _) = layout(tokens, 2, 0x5157, dtype);
    write(&mut k8, &table, &vec![0; tokens], &all, &k, &v, &st);
    write(&mut k8, &table, &vec![1; tokens], &all, &rev(&k), &rev(&v), &st);
    let (mut k16, table16, _) = layout(tokens, 2, 0x5157, KvDtype::F16);
    write(&mut k16, &table16, &vec![0; tokens], &all, &kq, &vq, &st);
    write(&mut k16, &table16, &vec![1; tokens], &all, &rev(&kq), &rev(&vq), &st);

    let mut scratch = Scratch::new_serving(64, tokens.div_ceil(512)).unwrap();
    // decode rows and windows at and around the threshold and deep, a 256-row prefill slice across 8,192, rows of both slots
    let mut calls: Vec<(Vec<i32>, Vec<i32>)> = vec![];
    for p in [0usize, 511, 512, 8191, 8192, 8193, 12_345, 16_384, 19_999] {
        calls.push((vec![0], vec![p as i32]));
        calls.push((vec![1], vec![p as i32]));
    }
    for (start, n) in [(8188usize, 5usize), (16_380, 8), (100, 64), (8100, 256), (19_744, 256), (8191, 1)] {
        calls.push((vec![0; n], (start..start + n).map(|p| p as i32).collect()));
    }
    let mixed: Vec<i32> = (8180..8200).chain(19_990..20_000).collect();
    calls.push(((0..mixed.len()).map(|i| (i % 2) as i32).collect(), mixed));
    let mut rows_checked = 0;
    for (slots, pos) in &calls {
        let q = rng.vec(pos.len() * QD, 4.0);
        let a = long(&k8, &table, slots, pos, &q, &mut scratch, &st);
        let b = long(&k16, &table16, slots, pos, &q, &mut scratch, &st);
        assert_bits_eq(&a, &b, &format!("{dtype:?} vs fp16-of-dequantised, rows {:?}..", &pos[..pos.len().min(3)]));
        rows_checked += pos.len();
    }
    // and an int8 row is the same bits whichever call carries it: one 256-row slice against the same rows one at a time
    let pos: Vec<i32> = (8000..8256).collect();
    let q = rng.vec(pos.len() * QD, 4.0);
    let whole = long(&k8, &table, &vec![0; 256], &pos, &q, &mut scratch, &st);
    for (i, &p) in pos.iter().enumerate().step_by(17) {
        let one = long(&k8, &table, &[0], &[p], &q[i * QD..(i + 1) * QD], &mut scratch, &st);
        assert_bits_eq(&one, &whole[i * QD..(i + 1) * QD], &format!("int8 row {p}: decode vs prefill slice"));
    }
    eprintln!("{dtype:?}: {rows_checked} rows == fp16 of the dequantised cache, bit for bit; decode == prefill");
}

/// f64 attention of one row over `k`, `v` (`[tokens][2][128]`, already whatever precision is meant).
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
            s[j] = (0..128).map(|e| qh[e] as f64 * kr[e] as f64).sum::<f64>() * scale;
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
        o.iter_mut().for_each(|x| *x /= l);
    }
    out
}

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
fn int8_error_against_the_f64_reference() {
    gpu!();
    let st = Stream::new().unwrap();
    let tokens = 17_000;
    let rows: Vec<usize> = vec![0, 63, 511, 4095, 8191, 8192, 8255, 12_345, 16_383, 16_384, 16_999];
    for (dist, scale_q) in [("uniform K and V", 2.0f32), ("K with 20x outlier channels", 1.0), ("uniform, q scale 8", 8.0)] {
        let mut rng = Rng(0xE88 + scale_q as u64);
        let k = if dist.starts_with("K with") {
            (0..tokens * KVD).map(|i| rng.f32(1.0) * if (i % 128) % 37 == 5 { 20.0 } else { 1.0 }).collect()
        } else {
            rng.vec(tokens * KVD, 1.0)
        };
        let v = rng.vec(tokens * KVD, 1.0);
        let all: Vec<i32> = (0..tokens as i32).collect();
        let (mut c8, t8, _) = layout(tokens, 1, 0x77, KvDtype::Int8);
        let (mut c16, t16, _) = layout(tokens, 1, 0x77, KvDtype::F16);
        write(&mut c8, &t8, &vec![0; tokens], &all, &k, &v, &st);
        write(&mut c16, &t16, &vec![0; tokens], &all, &k, &v, &st);
        let pos: Vec<i32> = rows.iter().map(|&p| p as i32).collect();
        let q = rng.vec(rows.len() * QD, scale_q);
        let mut scratch = Scratch::new_serving(64, tokens.div_ceil(512)).unwrap();
        let g8 = long(&c8, &t8, &vec![0; rows.len()], &pos, &q, &mut scratch, &st);
        let g16 = long(&c16, &t16, &vec![0; rows.len()], &pos, &q, &mut scratch, &st);
        let (mut w8, mut w16) = (0f64, 0f64);
        for (i, &p) in rows.iter().enumerate() {
            let want = ref_row(&q[i * QD..(i + 1) * QD], &k, &v, p);
            let (e8, e16) = (worst_rel(&g8[i * QD..(i + 1) * QD], &want), worst_rel(&g16[i * QD..(i + 1) * QD], &want));
            w8 = w8.max(e8);
            w16 = w16.max(e16);
            eprintln!("  {dist}: row at {p}: fp16 {e16:.3e}, int8 {e8:.3e}");
        }
        eprintln!(
            "{dist}: worst max |out - f64(unquantised)| / head max over {} rows: fp16 cache {w16:.3e}, int8 cache {w8:.3e}",
            rows.len()
        );
        // a report, not a gate: on random data the output is an average of noise and so is its error (the ratio does not shrink with
        // depth). The quality gate is the model's logits (docs/NEMOTRON_KV_INT8.md)
        assert!(g8.iter().all(|x| x.is_finite()) && w8 < 0.25, "{dist}: int8 attention error {w8:.3e}");
    }
}

/// One layer's cache of `tokens` positions of slot 0 on the identity table, random K and V in 8,192-token draws.
fn long_cache(tokens: usize, dtype: KvDtype, st: &Stream) -> (NemoKv, BlockTable) {
    let pages = tokens.div_ceil(PAGE);
    let mut kv = NemoKv::new_typed(pages + 1, dtype).unwrap();
    let table = BlockTable::from_host(&[(0..pages as u32).collect()], pages).unwrap();
    let mut rng = Rng(0x5EED_10C0);
    let mut start = 0;
    while start < tokens {
        let n = 8192.min(tokens - start);
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

#[test]
#[ignore]
fn bench_kv_int8_decode_and_prefill() {
    gpu!();
    let st = Stream::new().unwrap();
    let ends = env_list("REXL3_ATTN_LONG_POS", &[131072, 524288, 1048576]);
    let reps = env_list("REXL3_ATTN_LONG_REPS", &[9])[0];
    let max_pos = *ends.iter().max().unwrap() + 256;
    let mut rng = Rng(0xDEC8);
    let q_host = rng.vec(256 * QD, 2.0);
    let mut out = DevBuf::zeroed(256 * QD * 4).unwrap();
    let mut scratch = Scratch::new_serving(64, max_pos.div_ceil(512)).unwrap();
    println!("\n== fp16 vs int8 KV, one layer, synthetic cache of {max_pos} keys, attn_long (the one rule) ==");
    println!(
        "{:<6} {:>5} {:>9} {:>10} {:>10} {:>8} {:>11} {:>11}",
        "call", "rows", "end", "fp16 ms", "int8 ms", "speedup", "fp16 GB/s", "int8 GB/s"
    );
    let mut caches = vec![];
    for dtype in [KvDtype::F16, KvDtype::Int8] {
        caches.push(long_cache(max_pos, dtype, &st));
    }
    for &end in &ends {
        for &rows in &[1usize, 4, 5, 8, 64, 256] {
            let (slots, pos): (Vec<i32>, Vec<i32>) =
                ((0..rows).map(|_| 0).collect(), (end - rows..end).map(|p| p as i32).collect());
            let idx = RowIndex::from_host(&slots, &pos).unwrap();
            let qv = DevBuf::from_slice(&q_host[..rows * QD]).unwrap();
            let mut t = [0f64; 2];
            for (i, (kv, table)) in caches.iter().enumerate() {
                t[i] = median_ms(&st, reps, || {
                    nf::attn_long(&qv, kv, table, &idx, 1, &mut scratch, &mut out, &st).unwrap();
                });
            }
            let gbs = |ms: f64, b: usize| end as f64 * b as f64 / ms / 1e6;
            println!(
                "{:<6} {:>5} {:>9} {:>10.3} {:>10.3} {:>7.2}x {:>11.1} {:>11.1}",
                if rows <= 8 { "decode" } else { "slice" },
                rows,
                end,
                t[0],
                t[1],
                t[0] / t[1],
                gbs(t[0], KvDtype::F16.token_bytes()),
                gbs(t[1], KvDtype::Int8.token_bytes())
            );
        }
    }
    // the cache write of a 256-row slice and of one decode row
    for rows in [1usize, 256] {
        let k = DevBuf::from_slice(&rng.vec(rows * KVD, 1.0)).unwrap();
        let v = DevBuf::from_slice(&rng.vec(rows * KVD, 1.0)).unwrap();
        let pos: Vec<i32> = (0..rows as i32).collect();
        let idx = RowIndex::from_host(&vec![0; rows], &pos).unwrap();
        let mut t = [0f64; 2];
        for (i, (kv, table)) in caches.iter_mut().enumerate() {
            t[i] = median_ms(&st, reps, || na::kv_write(kv, table, &idx, &k, &v, &st).unwrap());
        }
        println!("kv_write {rows:>4} rows: fp16 {:.4} ms, int8 {:.4} ms", t[0], t[1]);
    }
}
