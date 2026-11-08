//! Nemotron 3.5 Super VL NoPE attention on the GPU (`csrc/nemotron_attn.cu`): the KV write rounds once into the block table's pages,
//! decode of token t equals prefill row t bit for bit at every length, a verify window equals single-row decodes bit for bit, and
//! placement (slot and blocks) changes no bit. The GPU against the CPU oracle is in `crates/model/.../attn_ref.rs`, which can import
//! the oracle (this crate cannot: gb-model depends on it). GPU tests skip when there is no device.
#![cfg(not(gb_no_cuda))]

use gb_cuda::attention::BlockTable;
use gb_cuda::nemotron_attn::{self as na, NemoKv, RowIndex, Scratch};
use gb_cuda::{device_available, DevBuf, Stream};

/// fp32 elements of one query or output row (32 heads of 128).
const QD: usize = 32 * 128;
/// fp32 elements of one projected K or V row (2 kv heads of 128).
const KVD: usize = 2 * 128;
const PAGE: usize = 256;

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

/// Caches for `slots` sequences of up to `tokens` tokens. Each slot's pages are drawn from a shuffled pool of blocks, so the table
/// is not the identity and the kernels must follow it. Returns the cache, the table and the host copy of the table.
fn layout(tokens: usize, slots: usize, seed: u64) -> (NemoKv, BlockTable, Vec<Vec<u32>>) {
    let pages = tokens.div_ceil(PAGE);
    let n_blocks = pages * slots + 2;
    let mut ids: Vec<u32> = (0..n_blocks as u32).collect();
    let mut rng = Rng(seed | 1);
    for i in (1..ids.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        ids.swap(i, j);
    }
    let tables: Vec<Vec<u32>> = (0..slots).map(|s| ids[s * pages..(s + 1) * pages].to_vec()).collect();
    let kv = NemoKv::new(n_blocks).unwrap();
    let table = BlockTable::from_host(&tables, pages).unwrap();
    (kv, table, tables)
}

/// Writes the K and V rows of `pos` (in `slots`) into the cache.
fn write(kv: &mut NemoKv, table: &BlockTable, slots: &[i32], pos: &[i32], k: &[f32], v: &[f32], st: &Stream) {
    let idx = RowIndex::from_host(slots, pos).unwrap();
    let kd = DevBuf::from_slice(k).unwrap();
    let vd = DevBuf::from_slice(v).unwrap();
    na::kv_write(kv, table, &idx, &kd, &vd, st).unwrap();
    st.sync().unwrap();
}

/// Runs attention for the rows (`slots`, `pos`) with queries `q` (fp32 `[rows][32][128]`) and returns the output rows.
fn attend(
    kv: &NemoKv,
    table: &BlockTable,
    rows: (&[i32], &[i32]),
    q: &[f32],
    scratch: &mut Scratch,
    prefill: bool,
    st: &Stream,
) -> Vec<f32> {
    let (slots, pos) = rows;
    let idx = RowIndex::from_host(slots, pos).unwrap();
    let qd = DevBuf::from_slice(q).unwrap();
    let mut out = DevBuf::zeroed(q.len() * 4).unwrap();
    if prefill {
        na::attn_prefill(&qd, kv, table, &idx, scratch, &mut out, st).unwrap();
    } else {
        na::attn_decode(&qd, kv, table, &idx, scratch, &mut out, st).unwrap();
    }
    st.sync().unwrap();
    out.to_vec::<f32>().unwrap()
}

fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits()) {
        panic!("{what}: first bit difference at {i}: {} vs {}", got[i], want[i]);
    }
    assert!(got.iter().all(|x| x.is_finite()), "{what}: a non-finite output");
}

#[test]
fn kv_write_rounds_once_into_the_block_table_pages() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let tokens = 600;
    let (mut kv, table, tables) = layout(tokens, 1, 0x1111);
    let mut rng = Rng(0x2222);
    let k = rng.vec(tokens * KVD, 3.0);
    let v = rng.vec(tokens * KVD, 3.0);
    let pos: Vec<i32> = (0..tokens as i32).collect();
    write(&mut kv, &table, &vec![0; tokens], &pos, &k, &v, &st);
    let kc: Vec<u16> = kv.k.to_vec().unwrap();
    let vc: Vec<u16> = kv.v.to_vec().unwrap();
    for (r, &p) in pos.iter().enumerate() {
        let p = p as usize;
        for e in 0..KVD {
            let at = (tables[0][p / PAGE] as usize * PAGE + p % PAGE) * KVD + e;
            let want_k = gb_exl3::f16::f64_to_f16(k[r * KVD + e] as f64);
            let want_v = gb_exl3::f16::f64_to_f16(v[r * KVD + e] as f64);
            assert_eq!(kc[at], want_k, "K row {r} element {e}");
            assert_eq!(vc[at], want_v, "V row {r} element {e}");
        }
    }
}

#[test]
fn decode_of_token_t_equals_prefill_row_t_bit_for_bit() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    for len in [1usize, 2, 63, 64, 511, 512, 513, 1300] {
        let mut rng = Rng(0x3000 + len as u64);
        let q = rng.vec(len * QD, 1.0);
        let k = rng.vec(len * KVD, 1.0);
        let v = rng.vec(len * KVD, 1.0);
        let all: Vec<i32> = (0..len as i32).collect();
        let zeros = vec![0i32; len];

        // prefill: every row in one call
        let (mut kv, table, _) = layout(len, 1, 0x4000 + len as u64);
        write(&mut kv, &table, &zeros, &all, &k, &v, &st);
        let mut scratch = Scratch::new(len, len.div_ceil(512)).unwrap();
        let prefill = attend(&kv, &table, (&zeros, &all), &q, &mut scratch, true, &st);

        // decode: a fresh cache filled with NaN bit patterns, one token at a time, on another block placement. Reading a
        // position past the row's own key would show up as NaN or as a changed bit.
        let (mut kv2, table2, _) = layout(len, 1, 0x5000 + len as u64);
        kv2.k.memset_async(0xFF, &st).unwrap();
        kv2.v.memset_async(0xFF, &st).unwrap();
        st.sync().unwrap();
        let mut scratch1 = Scratch::new(1, len.div_ceil(512)).unwrap();
        for t in 0..len {
            let pos = [t as i32];
            write(&mut kv2, &table2, &[0], &pos, &k[t * KVD..(t + 1) * KVD], &v[t * KVD..(t + 1) * KVD], &st);
            let out = attend(&kv2, &table2, (&[0], &pos), &q[t * QD..(t + 1) * QD], &mut scratch1, false, &st);
            assert_bits_eq(&out, &prefill[t * QD..(t + 1) * QD], &format!("len {len}: decode of token {t} vs prefill row {t}"));
        }
    }
}

#[test]
fn verify_window_of_8_equals_8_single_row_decodes_bit_for_bit() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let len = 1300;
    let mut rng = Rng(0x6000);
    let q = rng.vec(len * QD, 1.0);
    let k = rng.vec(len * KVD, 1.0);
    let v = rng.vec(len * KVD, 1.0);
    let all: Vec<i32> = (0..len as i32).collect();
    let (mut kv, table, _) = layout(len, 1, 0x6001);
    write(&mut kv, &table, &vec![0; len], &all, &k, &v, &st);
    let mut scratch8 = Scratch::new(8, len.div_ceil(512)).unwrap();
    let mut scratch1 = Scratch::new(1, len.div_ceil(512)).unwrap();
    // windows that sit inside a chunk, across the 512 boundary, across 1024, and at the end of the sequence
    for start in [0usize, 1, 5, 505, 508, 1000, 1017, 1292] {
        let pos: Vec<i32> = (start..start + 8).map(|p| p as i32).collect();
        let window = attend(&kv, &table, (&[0; 8], &pos), &q[start * QD..(start + 8) * QD], &mut scratch8, false, &st);
        for (i, &p) in pos.iter().enumerate() {
            let single = attend(&kv, &table, (&[0], &[p]), &q[(start + i) * QD..(start + i + 1) * QD], &mut scratch1, false, &st);
            assert_bits_eq(&window[i * QD..(i + 1) * QD], &single, &format!("window at {start}, row {i} (position {p})"));
        }
    }
}

#[test]
fn slot_and_block_placement_change_no_bit() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let len = 700;
    let mut rng = Rng(0x7000);
    let q = rng.vec(len * QD, 1.0);
    let k = rng.vec(len * KVD, 1.0);
    let v = rng.vec(len * KVD, 1.0);
    // The same sequence in slot 0 and in slot 1, with disjoint blocks, and both slots in one prefill call.
    let (mut kv, table, _) = layout(len, 2, 0x7001);
    let mut slots = vec![0i32; len];
    slots.extend(vec![1i32; len]);
    let mut pos: Vec<i32> = (0..len as i32).collect();
    pos.extend(0..len as i32);
    let mut kk = k.clone();
    kk.extend_from_slice(&k);
    let mut vv = v.clone();
    vv.extend_from_slice(&v);
    let mut qq = q.clone();
    qq.extend_from_slice(&q);
    write(&mut kv, &table, &slots, &pos, &kk, &vv, &st);
    let mut scratch = Scratch::new(2 * len, len.div_ceil(512)).unwrap();
    let out = attend(&kv, &table, (&slots, &pos), &qq, &mut scratch, true, &st);
    assert_bits_eq(&out[len * QD..], &out[..len * QD], "slot 1 against slot 0");
}

/// Prefill slices of 128 and 256 rows (the widths the large-row MoE prefill allows) are the same bits as the same rows cut into slices of
/// 64 or other cuts: each row's output depends on its own keys only. The rows are one sequence at positions 700.., so every slice reads
/// the previous slices' keys, across the 512-key chunk edges.
#[test]
fn wide_prefill_slices_equal_narrow_ones_bit_for_bit() {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let st = Stream::new().unwrap();
    let (base, rows) = (700usize, 256usize);
    let len = base + rows;
    let mut rng = Rng(0x8000);
    let q = rng.vec(len * QD, 1.0);
    let k = rng.vec(len * KVD, 1.0);
    let v = rng.vec(len * KVD, 1.0);
    let all: Vec<i32> = (0..len as i32).collect();
    let (mut kv, table, _) = layout(len, 1, 0x8001);
    write(&mut kv, &table, &vec![0; len], &all, &k, &v, &st);
    let chunks = len.div_ceil(512);
    // one call of 256 rows
    let mut scratch = Scratch::new(rows, chunks).unwrap();
    let pos: Vec<i32> = (base as i32..len as i32).collect();
    let whole = attend(&kv, &table, (&vec![0; rows], &pos), &q[base * QD..], &mut scratch, true, &st);
    for cuts in [vec![64usize, 64, 64, 64], vec![128, 128], vec![100, 156], vec![9, 247], vec![1, 63, 64, 128]] {
        let mut out = Vec::new();
        let mut start = 0;
        for &n in &cuts {
            let mut sc = Scratch::new(n, chunks).unwrap();
            let p: Vec<i32> = ((base + start) as i32..(base + start + n) as i32).collect();
            let got =
                attend(&kv, &table, (&vec![0; n], &p), &q[(base + start) * QD..(base + start + n) * QD], &mut sc, true, &st);
            out.extend(got);
            start += n;
        }
        assert_bits_eq(&out, &whole, &format!("cuts {cuts:?}"));
    }
}
