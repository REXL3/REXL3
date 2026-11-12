//! The split-key decode attention of Ling-3.1 (`ling_mla_attn::attend_split`, `csrc/ling_mla_attn.cu`'s `attn_split_kernel` +
//! `attn_combine_kernel`, arithmetic id `ling-mla-attn-tc32-bf16-v2-decode-split1024`): absolute 1,024-key chunks on separate CTAs, merged
//! in ascending chunk order.
//!
//! What this file pins (the correctness contract of the decode path; every comparison is on raw bits unless it says oracle):
//!   * a row with fewer than 1,024 keys has one chunk and equals the walk kernel (`attend`) bit for bit;
//!   * ROW INVARIANCE: a (slot, position) query row gives the same bits computed alone (a 1-row decode step) as inside 2-, 3-, 4- and
//!     70-row calls, at any index of the call, beside rows of other positions and of another slot with a different page placement;
//!     the 70-row call is launched in several row batches (the workspace holds 320 (row, chunk) slots), so this also pins batching;
//!   * the chunk boundary is ABSOLUTE: positions 1023 / 1024 / 1025 / 2047 / 2048 / 2049 / 3071 / 3072 / 4095 / 8191 and others;
//!   * GRID INVARIANCE: the same rows over block tables of 40, 80 and 320 pages (so 10, 20 and 80 chunk-grid planes, the grid a
//!     CUDA-graph capture bakes in) give the same bits, and so do two runs;
//!   * a window of pages (`attend_split_from`) equals the same pages as a table of their own;
//!   * oracle: the output matches an f64 softmax attention (cosine >= 0.9999 per row) at chunked lengths.
//!
//! Run on a GB10 box:
//!   cargo test --release -p gb-cuda --test ling_mla_split -- --nocapture --test-threads=1
#![cfg(not(gb_no_cuda))]
#![allow(dead_code)]

mod common;

use common::Rng;
use gb_cuda::attention::{BlockTable, RowIndex};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::ling_mla_attn as tc;
use gb_cuda::mla_gated::{MlaGatedCache, LATENT_DIM, N_HEADS, ROPE_DIM, SM_SCALE};
use gb_cuda::rt::{DevBuf, Stream};

const PAGES: usize = 40; // 10,240 keys
const KEYS: usize = PAGES * 256;
const CW: usize = 576;
const ROW_LAT: usize = N_HEADS * LATENT_DIM; // o_lat / q_lat elements of a row
const ROW_ROT: usize = N_HEADS * ROPE_DIM;

fn bf(rng: &mut Rng, n: usize, scale: f32) -> Vec<u16> {
    (0..n).map(|_| gb_cuda::num::f32_to_bf16(rng.uniform(-1.0, 1.0) * scale)).collect()
}
fn f64s(v: &[u16]) -> Vec<f64> {
    v.iter().map(|&x| gb_cuda::num::bf16_to_f32(x) as f64).collect()
}

/// (slot, position) of the bank's query rows. Slot 1 reads the same pages in the opposite order (a different block placement).
const BANK: [(i32, i32); 16] = [
    (0, 0),
    (0, 31),
    (0, 700),
    (0, 1023),
    (0, 1024),
    (1, 1025),
    (0, 2047),
    (1, 2048),
    (0, 2049),
    (1, 3071),
    (0, 3072),
    (1, 4095),
    (0, 5000),
    (1, 8191),
    (0, 9000),
    (1, 10239),
];

struct Fixture {
    st: Stream,
    cache: MlaGatedCache,
    host_cache: Vec<u16>, // [PAGES][256][576]
    tables: Vec<Vec<u32>>,
    q_lat: Vec<Vec<u16>>, // per bank row
    q_rot: Vec<Vec<u16>>,
    ws: DevBuf,
}

impl Fixture {
    fn new(seed: u64) -> Self {
        let mut rng = Rng(seed);
        let st = Stream::new().unwrap();
        // keys with a per-key scale in [0.2, 2.6], so the softmax maximum moves between chunks
        let mut host_cache = vec![0u16; KEYS * CW];
        for k in 0..KEYS {
            let sc = 0.2 + 2.4 * rng.uniform(0.0, 1.0);
            let row = bf(&mut rng, CW, sc);
            host_cache[k * CW..(k + 1) * CW].copy_from_slice(&row);
        }
        let mut cache = MlaGatedCache::new(PAGES, Dtype::BF16).unwrap();
        cache.buf.upload(&host_cache).unwrap();
        let fwd: Vec<u32> = (0..PAGES as u32).collect();
        let rev: Vec<u32> = (0..PAGES as u32).rev().collect();
        // slot 1 is the same cache read through reversed pages: its logical key j lives on page PAGES-1-j/256. Its host copy differs, so
        // the bank rows of slot 1 are compared with their own oracle through `logical_keys`.
        let q_lat = BANK.iter().map(|_| bf(&mut rng, ROW_LAT, 4.0)).collect();
        let q_rot = BANK.iter().map(|_| bf(&mut rng, ROW_ROT, 4.0)).collect();
        let ws = DevBuf::zeroed(tc::SPLIT_WS_BYTES).unwrap();
        Fixture { st, cache, host_cache, tables: vec![fwd, rev], q_lat, q_rot, ws }
    }

    /// The keys slot `slot` sees, logically ordered, as f64 `[KEYS][576]`.
    fn logical_keys(&self, slot: usize, upto: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(upto * CW);
        for j in 0..upto {
            let page = self.tables[slot][j / 256] as usize;
            let base = (page * 256 + j % 256) * CW;
            out.extend(f64s(&self.host_cache[base..base + CW]));
        }
        out
    }

    /// Runs the split attention on the bank rows `idx` (in that order) over a block table `width` pages wide; the row of bank entry
    /// `i` has the query, slot and position of `BANK[i]` wherever it sits in the call.
    fn run(&mut self, idx: &[usize], width: usize) -> Vec<Vec<u16>> {
        self.run_with(idx, width, 0, true)
    }

    fn run_with(&mut self, idx: &[usize], width: usize, first_block: usize, split: bool) -> Vec<Vec<u16>> {
        let n = idx.len();
        let tabs: Vec<Vec<u32>> = self.tables.iter().map(|t| t.clone()).collect();
        let table = BlockTable::from_host(&tabs, width).unwrap();
        let slots: Vec<i32> = idx.iter().map(|&i| BANK[i].0).collect();
        let pos: Vec<i32> = idx.iter().map(|&i| BANK[i].1 - 256 * first_block as i32).collect();
        let ri = RowIndex::from_host(&slots, &pos).unwrap();
        let ql: Vec<u16> = idx.iter().flat_map(|&i| self.q_lat[i].iter().copied()).collect();
        let qr: Vec<u16> = idx.iter().flat_map(|&i| self.q_rot[i].iter().copied()).collect();
        let q_lat = DevBuf::from_slice(&ql).unwrap();
        let q_rot = DevBuf::from_slice(&qr).unwrap();
        let mut o = DevBuf::alloc(n * ROW_LAT * 2).unwrap();
        // poison the output: a row the kernel forgets to write must not pass by accident
        o.upload(&vec![0x7fc1u16; n * ROW_LAT]).unwrap();
        if split {
            tc::attend_split_from(&q_lat, &q_rot, &self.cache, &table, first_block, &ri, &mut o, &mut self.ws, n, &self.st)
                .unwrap();
        } else {
            tc::attend(&q_lat, &q_rot, &self.cache, &table, &ri, &mut o, n, false, &self.st).unwrap();
        }
        self.st.sync().unwrap();
        let mut got = vec![0u16; n * ROW_LAT];
        o.download(&mut got).unwrap();
        got.chunks(ROW_LAT).map(|c| c.to_vec()).collect()
    }

    /// f64 reference of one bank row: softmax attention of its 64 heads over keys 0..=pos.
    fn oracle(&self, i: usize) -> Vec<f64> {
        let (slot, pos) = BANK[i];
        let n = pos as usize + 1;
        let keys = self.logical_keys(slot as usize, n);
        let ql = f64s(&self.q_lat[i]);
        let qr = f64s(&self.q_rot[i]);
        let mut out = vec![0f64; ROW_LAT];
        for h in 0..N_HEADS {
            let mut s = vec![0f64; n];
            for j in 0..n {
                let k = &keys[j * CW..(j + 1) * CW];
                let mut a = 0.0;
                for c in 0..LATENT_DIM {
                    a += ql[h * LATENT_DIM + c] * k[c];
                }
                for d in 0..ROPE_DIM {
                    a += qr[h * ROPE_DIM + d] * k[LATENT_DIM + d];
                }
                s[j] = a * (SM_SCALE as f64);
            }
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut l = 0.0;
            for v in s.iter_mut() {
                *v = (*v - m).exp();
                l += *v;
            }
            for j in 0..n {
                let w = s[j] / l;
                let k = &keys[j * CW..j * CW + LATENT_DIM];
                for c in 0..LATENT_DIM {
                    out[h * LATENT_DIM + c] += w * k[c];
                }
            }
        }
        out
    }
}

fn cos_sim(a: &[f64], b: &[f64]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt()
}

fn bit_eq(a: &[u16], b: &[u16], what: &str) {
    if a != b {
        let bad = a.iter().zip(b).filter(|(x, y)| x != y).count();
        let first = a.iter().zip(b).position(|(x, y)| x != y).unwrap();
        panic!("{what}: {bad} of {} elements differ (first at {first}: {:#06x} vs {:#06x})", a.len(), a[first], b[first]);
    }
}

#[cfg(not(gb_no_cuda))]
#[test]
fn split_constants_match_the_kernel() {
    // the Rust constants the memory plan and the id are built from are the CUDA file's
    extern "C" {
        fn gb_ling_mla_split_ws_bytes() -> usize;
        fn gb_ling_mla_split_keys() -> i32;
        fn gb_ling_mla_split_ws_slots() -> i32;
    }
    unsafe {
        assert_eq!(gb_ling_mla_split_keys() as usize, tc::SPLIT_KEYS);
        assert_eq!(gb_ling_mla_split_ws_slots() as usize, tc::SPLIT_WS_SLOTS);
        assert_eq!(gb_ling_mla_split_ws_bytes(), tc::SPLIT_WS_BYTES);
    }
}

#[cfg(not(gb_no_cuda))]
#[test]
fn one_chunk_equals_the_walk_kernel_bit_for_bit() {
    // pos < 1024: one chunk, w_0 = 2^0 = 1, so the combine is the walk kernel's epilogue
    let mut f = Fixture::new(1);
    let idx = [0usize, 1, 2, 3]; // positions 0, 31, 700, 1023 (slot 0)
    let split = f.run_with(&idx, PAGES, 0, true);
    let walk = f.run_with(&idx, PAGES, 0, false);
    for (k, &i) in idx.iter().enumerate() {
        bit_eq(&split[k], &walk[k], &format!("pos {} split vs walk", BANK[i].1));
    }
}

#[cfg(not(gb_no_cuda))]
#[test]
fn row_is_bit_identical_alone_and_inside_wider_calls() {
    let mut f = Fixture::new(2);
    let all: Vec<usize> = (0..BANK.len()).collect();
    // each row alone: a 1-row decode step
    let single: Vec<Vec<u16>> = all.iter().map(|&i| f.run(&[i], PAGES).remove(0)).collect();
    // the whole bank in one call (16 rows: one batch at 40 pages; positions ascending)
    let whole = f.run(&all, PAGES);
    for i in 0..BANK.len() {
        bit_eq(&whole[i], &single[i], &format!("bank {i} (slot {} pos {}) in the 16-row call", BANK[i].0, BANK[i].1));
    }
    // windows of 2, 3 and 4 rows: every bank row at every index of the window, beside rows of other positions and the other slot
    let mut checked = 0;
    for width in 2..=4usize {
        for i in 0..BANK.len() {
            for at in 0..width {
                let mut idx: Vec<usize> = (0..width).map(|k| (i + 1 + 5 * k) % BANK.len()).collect();
                idx[at] = i;
                // distinct neighbours are not needed (duplicates are legal rows), only that row `at` is bank row `i`
                let got = f.run(&idx, PAGES);
                bit_eq(&got[at], &single[i], &format!("bank {i} at index {at} of {idx:?}"));
                checked += 1;
            }
        }
    }
    // reversed and rotated orders of the whole bank
    let mut rev = all.clone();
    rev.reverse();
    let got = f.run(&rev, PAGES);
    for (k, &i) in rev.iter().enumerate() {
        bit_eq(&got[k], &single[i], &format!("bank {i} in the reversed call"));
    }
    println!(
        "row invariance: {} windows of 2-4 rows, the 16-row call and its reverse: all rows bit-equal to the 1-row step",
        checked
    );
}

#[cfg(not(gb_no_cuda))]
#[test]
fn a_call_wider_than_the_workspace_batch_gives_the_same_bits() {
    // 70 rows over a 40-page table: 10 chunks a row, 32 rows a batch, so three batches (32 + 32 + 6)
    let mut f = Fixture::new(3);
    let single: Vec<Vec<u16>> = (0..BANK.len()).map(|i| f.run(&[i], PAGES).remove(0)).collect();
    let idx: Vec<usize> = (0..70).map(|k| (k * 7 + 3) % BANK.len()).collect();
    let got = f.run(&idx, PAGES);
    for (k, &i) in idx.iter().enumerate() {
        bit_eq(&got[k], &single[i], &format!("row {k} (bank {i}) of the 70-row call"));
    }
    // and over a 160-page table (40 chunks a row, 8 rows a batch): the same rows, the same bits
    let got = f.run(&idx, 160);
    for (k, &i) in idx.iter().enumerate() {
        bit_eq(&got[k], &single[i], &format!("row {k} (bank {i}) of the 70-row call, 160-page table"));
    }
}

#[cfg(not(gb_no_cuda))]
#[test]
fn the_chunk_grid_width_does_not_change_a_bit() {
    // the launch spans every page of the table; a CUDA graph captured at one width replays at every position. Widths 40 / 80 / 320
    // pages are 10 / 20 / 80 chunk planes (320 pages is the workspace's 80-chunk edge for a 4-row batch).
    let mut f = Fixture::new(4);
    let idx = [4usize, 5, 8, 13];
    let base = f.run(&idx, PAGES);
    for width in [80usize, 320, 1280] {
        let got = f.run(&idx, width);
        for k in 0..idx.len() {
            bit_eq(&got[k], &base[k], &format!("width {width} pages, row {k}"));
        }
    }
    let again = f.run(&idx, PAGES);
    for k in 0..idx.len() {
        bit_eq(&again[k], &base[k], &format!("second run, row {k}"));
    }
}

#[cfg(not(gb_no_cuda))]
#[test]
fn a_page_window_equals_a_table_of_its_own() {
    // attend_split_from(first_block = 2): chunks and positions count from the window's first page (the draft layer's clamp)
    let mut f = Fixture::new(5);
    let idx = [8usize, 10, 12]; // slot 0, positions 2049, 3072, 5000: all past page 2
    let windowed = f.run_with(&idx, PAGES, 2, true);
    // the same logical keys as a table that starts at page 2: rebuild the fixture's slot-0 table without its first two pages
    let tabs = vec![f.tables[0][2..].to_vec(), f.tables[1].clone()];
    let table = BlockTable::from_host(&tabs, PAGES).unwrap();
    let slots = vec![0i32; idx.len()];
    let pos: Vec<i32> = idx.iter().map(|&i| BANK[i].1 - 512).collect();
    let ri = RowIndex::from_host(&slots, &pos).unwrap();
    let ql: Vec<u16> = idx.iter().flat_map(|&i| f.q_lat[i].iter().copied()).collect();
    let qr: Vec<u16> = idx.iter().flat_map(|&i| f.q_rot[i].iter().copied()).collect();
    let (q_lat, q_rot) = (DevBuf::from_slice(&ql).unwrap(), DevBuf::from_slice(&qr).unwrap());
    let mut o = DevBuf::alloc(idx.len() * ROW_LAT * 2).unwrap();
    tc::attend_split(&q_lat, &q_rot, &f.cache, &table, &ri, &mut o, &mut f.ws, idx.len(), &f.st).unwrap();
    f.st.sync().unwrap();
    let mut got = vec![0u16; idx.len() * ROW_LAT];
    o.download(&mut got).unwrap();
    for k in 0..idx.len() {
        bit_eq(&got[k * ROW_LAT..(k + 1) * ROW_LAT], &windowed[k], &format!("window row {k}"));
    }
}

#[cfg(not(gb_no_cuda))]
#[test]
fn a_padding_row_writes_zeros_and_disturbs_no_neighbour() {
    let mut f = Fixture::new(6);
    let alone = f.run(&[8], PAGES).remove(0);
    let table = BlockTable::from_host(&f.tables.clone(), PAGES).unwrap();
    let ri = RowIndex::from_host(&[0, 0, 0], &[-1, 2049, -1]).unwrap();
    let ql: Vec<u16> = [8usize, 8, 8].iter().flat_map(|&i| f.q_lat[i].iter().copied()).collect();
    let qr: Vec<u16> = [8usize, 8, 8].iter().flat_map(|&i| f.q_rot[i].iter().copied()).collect();
    let (q_lat, q_rot) = (DevBuf::from_slice(&ql).unwrap(), DevBuf::from_slice(&qr).unwrap());
    let mut o = DevBuf::alloc(3 * ROW_LAT * 2).unwrap();
    o.upload(&vec![0x7fc1u16; 3 * ROW_LAT]).unwrap();
    tc::attend_split(&q_lat, &q_rot, &f.cache, &table, &ri, &mut o, &mut f.ws, 3, &f.st).unwrap();
    f.st.sync().unwrap();
    let mut got = vec![0u16; 3 * ROW_LAT];
    o.download(&mut got).unwrap();
    assert!(got[..ROW_LAT].iter().all(|&x| x == 0) && got[2 * ROW_LAT..].iter().all(|&x| x == 0), "padding rows must be zeros");
    bit_eq(&got[ROW_LAT..2 * ROW_LAT], &alone, "the live row between two padding rows");
}

#[cfg(not(gb_no_cuda))]
#[test]
fn split_matches_the_f64_oracle_across_chunk_boundaries() {
    let mut f = Fixture::new(7);
    // positions 1023 (one full chunk), 1024 (a second chunk of one key), 2049, 5000, 10239 (ten chunks, the last one 1,024 keys)
    for &i in &[3usize, 4, 8, 12, 15] {
        let got = f.run(&[i], PAGES).remove(0);
        let want = f.oracle(i);
        let got_f = f64s(&got);
        let cos = cos_sim(&got_f, &want);
        let mut max_abs = 0f64;
        let mut max_ref = 0f64;
        for (&g, &w) in got_f.iter().zip(&want) {
            max_abs = max_abs.max((g - w).abs());
            max_ref = max_ref.max(w.abs());
        }
        // the walk kernel (what the split replaces) against the same oracle, for comparison: the split must not be the less accurate of the two
        let walk = f64s(&f.run_with(&[i], PAGES, 0, false).remove(0));
        let (mut w_abs, mut e_split2, mut e_walk2) = (0f64, 0f64, 0f64);
        for ((&w, &g), &r) in walk.iter().zip(&got_f).zip(&want) {
            w_abs = w_abs.max((w - r).abs());
            e_split2 += (g - r) * (g - r);
            e_walk2 += (w - r) * (w - r);
        }
        println!(
            "oracle pos {:5}: split cos {cos:.7} max |err| {max_abs:.4e} rms {:.4e} | walk max |err| {w_abs:.4e} rms {:.4e} (max |ref| {max_ref:.3})",
            BANK[i].1,
            (e_split2 / want.len() as f64).sqrt(),
            (e_walk2 / want.len() as f64).sqrt()
        );
        assert!(e_split2 <= e_walk2 * 1.05, "pos {}: the split is less accurate than the walk it replaces", BANK[i].1);
        assert!(cos >= 0.9999, "pos {}: cos {cos}", BANK[i].1);
        assert!(max_abs <= 0.02 * max_ref, "pos {}: max |err| {max_abs} vs max |ref| {max_ref}", BANK[i].1);
    }
}

/// Pins the arithmetic ([`tc::ARITH_ATTN`]): the FNV-1a of every bank row's o_lat on the fixed fixture. A change to the chunk length, the
/// merge order or formula, or the fold of a chunk's tiles moves it, and then the id must move with it. (The row-invariance tests above hold
/// for any fixed-order merge, so they would not notice a reordering; this does.)
#[cfg(not(gb_no_cuda))]
#[test]
fn split_arithmetic_is_pinned() {
    const PINNED: u64 = 0x6b36_444b_ef77_cb0a;
    let mut f = Fixture::new(2026);
    let all: Vec<usize> = (0..BANK.len()).collect();
    let got = f.run(&all, PAGES);
    let h = gb_exl3::fnv1a(got.iter().flat_map(|r| r.iter().flat_map(|x| x.to_le_bytes())));
    println!("split arithmetic hash {h:016x} ({})", tc::ARITH_ATTN);
    assert_eq!(h, PINNED, "the split attention's bits moved: if intended, bump ARITH_ATTN and re-pin ({h:016x})");
}

/// Per-call time of the walk kernel and the split path at decode shapes (1, 2, 4 rows) over 1K..30K keys of a 120-page cache, 64 heads.
/// `cargo test --release -p gb-cuda --test ling_mla_split -- --ignored --nocapture bench_split_vs_walk`
#[cfg(not(gb_no_cuda))]
#[test]
#[ignore]
fn bench_split_vs_walk() {
    const BP: usize = 120;
    let st = Stream::new().unwrap();
    let mut rng = Rng(77);
    let tile = bf(&mut rng, 4096 * CW, 1.0);
    let mut host = Vec::with_capacity(BP * 256 * CW);
    while host.len() < BP * 256 * CW {
        let n = (BP * 256 * CW - host.len()).min(tile.len());
        host.extend_from_slice(&tile[..n]);
    }
    let mut cache = MlaGatedCache::new(BP, Dtype::BF16).unwrap();
    cache.buf.upload(&host).unwrap();
    let table = BlockTable::from_host(&[(0..BP as u32).collect()], BP).unwrap();
    let mut ws = DevBuf::zeroed(tc::SPLIT_WS_BYTES).unwrap();
    let ql = bf(&mut rng, 4 * ROW_LAT, 4.0);
    let qr = bf(&mut rng, 4 * ROW_ROT, 4.0);
    let (q_lat, q_rot) = (DevBuf::from_slice(&ql).unwrap(), DevBuf::from_slice(&qr).unwrap());
    let mut o = DevBuf::alloc(4 * ROW_LAT * 2).unwrap();
    println!("keys rows | walk ms/call | split ms/call | speedup");
    for keys in [1024usize, 4096, 8192, 16384, 30000] {
        for rows in [1usize, 2, 4] {
            let pos: Vec<i32> = (0..rows as i32).map(|r| keys as i32 - 1 - (rows as i32 - 1 - r)).collect();
            let ri = RowIndex::from_host(&vec![0; rows], &pos).unwrap();
            let mut t = [0f64; 2];
            for (k, split) in [false, true].into_iter().enumerate() {
                let run = |o: &mut DevBuf, ws: &mut DevBuf| {
                    if split {
                        tc::attend_split(&q_lat, &q_rot, &cache, &table, &ri, o, ws, rows, &st).unwrap();
                    } else {
                        tc::attend(&q_lat, &q_rot, &cache, &table, &ri, o, rows, false, &st).unwrap();
                    }
                };
                for _ in 0..5 {
                    run(&mut o, &mut ws);
                }
                st.sync().unwrap();
                let n = 40;
                let t0 = std::time::Instant::now();
                for _ in 0..n {
                    run(&mut o, &mut ws);
                }
                st.sync().unwrap();
                t[k] = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
            }
            println!("{keys:6} {rows} | {:9.4} | {:9.4} | {:5.1}x", t[0], t[1], t[0] / t[1]);
        }
    }
}
