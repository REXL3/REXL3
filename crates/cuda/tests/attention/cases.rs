// The paged KV cache and attention cases, for the attention shape `SHAPE` (query heads x KV heads) the including test target sets
// (tests/attention.rs: 24/4, tests/attention_kv2.rs: 24/2, tests/attention_q16kv2.rs: 16/2, tests/attention_q8kv2.rs: 8/2).
// Included, not a module, so every target keeps the plain test names; each declares `mod common` and `SHAPE`.

use common::{cos_rel_rms, have_refs, ref_npy, Rng};
use gb_cuda::attention::*;
use gb_cuda::num::{bf16_to_f32, f32_to_bf16};
use gb_cuda::{device_available, DevBuf, Stream};

/// bf16 elements of one token's K (or V) row, the query heads, the query heads per KV head and the elements of one query (and
/// output) row, for `SHAPE`.
const TE: usize = SHAPE.token_elems();
const NQ: usize = SHAPE.q_heads();
const GRP: usize = SHAPE.group();
const QE: usize = SHAPE.q_row_elems();

const POISON: u32 = 0x3fff_fff0; // an unmapped table entry: reading it would fault, so a kernel that does fails loudly

fn gpu() -> Option<Stream> {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return None;
    }
    Some(Stream::new().unwrap())
}

// ---------------------------------------------------------------------------------------------------- data

/// Deterministic K/V of a sequence: bf16 bits `[len][kv heads][256]` each (a per-token key amplitude keeps softmax peaked).
fn gen_kv(sid: u64, len: usize) -> (Vec<u16>, Vec<u16>) {
    let mut r = Rng(0x1234_5678 ^ sid.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut k = Vec::with_capacity(len * TE);
    let mut v = Vec::with_capacity(len * TE);
    for _ in 0..len {
        let amp = r.uniform(0.5, 2.0);
        k.extend((0..TE).map(|_| f32_to_bf16(r.uniform(-1.0, 1.0) * amp)));
        v.extend((0..TE).map(|_| f32_to_bf16(r.uniform(-1.0, 1.0))));
    }
    (k, v)
}

/// The query row (24 heads x 256, bf16 bits) of sequence `sid` at `pos`.
fn qvec(sid: u64, pos: usize) -> Vec<u16> {
    let mut r = Rng(0xABCD ^ sid.wrapping_mul(0x1000_0000_01B3) ^ (pos as u64).wrapping_mul(0x9E37_79B9));
    (0..QE).map(|_| f32_to_bf16(r.uniform(-1.5, 1.5))).collect()
}

struct Seq {
    sid: u64,
    len: usize,
    k: Vec<u16>,
    v: Vec<u16>,
}

struct World {
    cache: PagedKv,
    table: BlockTable,
    seqs: Vec<Seq>,
    stream: Stream,
}

fn shuffle<T>(v: &mut [T], r: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = (r.next_u64() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// Fills the cache with NaN bits, places each sequence's pages at shuffled blocks (`placement` seeds the shuffle),
/// poisons the unused table entries and writes all K/V through `kv_write` in one launch.
fn make_world(seqs: Vec<Seq>, placement: u64, spare_blocks: usize) -> World {
    let stream = Stream::new().unwrap();
    let need: usize = seqs.iter().map(|s| s.len.div_ceil(BLOCK_TOKENS)).sum();
    let n_blocks = need + spare_blocks;
    let mut cache = PagedKv::with_shape(n_blocks, SHAPE).unwrap();
    cache.k.memset_async(0xFF, &stream).unwrap();
    cache.v.memset_async(0xFF, &stream).unwrap();
    let mut pool: Vec<u32> = (0..n_blocks as u32).collect();
    shuffle(&mut pool, &mut Rng(placement));
    let max_blocks = seqs.iter().map(|s| s.len.div_ceil(BLOCK_TOKENS)).max().unwrap() + 4;
    let mut tables = Vec::new();
    let mut next = 0;
    for s in &seqs {
        let nb = s.len.div_ceil(BLOCK_TOKENS);
        let mut t: Vec<u32> = pool[next..next + nb].to_vec();
        next += nb;
        t.resize(max_blocks, POISON);
        tables.push(t);
    }
    let table = BlockTable::from_host(&tables, max_blocks).unwrap();
    let (mut ks, mut vs, mut slot, mut pos) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (si, s) in seqs.iter().enumerate() {
        ks.extend_from_slice(&s.k);
        vs.extend_from_slice(&s.v);
        slot.extend((0..s.len).map(|_| si as i32));
        pos.extend(0..s.len as i32);
    }
    let (kd, vd) = (DevBuf::from_slice(&ks).unwrap(), DevBuf::from_slice(&vs).unwrap());
    let ri = RowIndex::from_host(&slot, &pos).unwrap();
    kv_write(&kd, &vd, &ri, &table, &mut cache, slot.len(), &stream).unwrap();
    stream.sync().unwrap();
    World { cache, table, seqs, stream }
}

fn seqs_of(lens: &[usize]) -> Vec<Seq> {
    lens.iter()
        .enumerate()
        .map(|(i, &len)| {
            let (k, v) = gen_kv(i as u64 + 1, len);
            Seq { sid: i as u64 + 1, len, k, v }
        })
        .collect()
}

/// Decode over `rows = [(seq index, position)]`; q rows come from `qvec(sid, pos)`.
fn run_decode(w: &World, rows: &[(usize, usize)]) -> Vec<u16> {
    let r: Vec<(usize, i32)> = rows.iter().map(|&(s, p)| (s, p as i32)).collect();
    run_decode_pad(w, &r)
}

/// As `run_decode`, but a negative position is a padding row (its q is a dummy).
fn run_decode_pad(w: &World, rows: &[(usize, i32)]) -> Vec<u16> {
    let n = rows.len();
    let mut q = Vec::new();
    for &(si, p) in rows {
        q.extend(qvec(w.seqs[si].sid, p.max(0) as usize));
    }
    let max_ctx = rows.iter().map(|r| (r.1 + 1).max(1) as usize).max().unwrap();
    // exercise a padded grid: the launch may cover a longer context than any row needs
    let max_ctx = max_ctx + (max_ctx % 3) * 700;
    let max_ctx = max_ctx.min(w.table.max_blocks * BLOCK_TOKENS);
    let slot: Vec<i32> = rows.iter().map(|r| r.0 as i32).collect();
    let pos: Vec<i32> = rows.iter().map(|r| r.1).collect();
    let ri = RowIndex::from_host(&slot, &pos).unwrap();
    let qd = DevBuf::from_slice(&q).unwrap();
    let mut out = DevBuf::from_slice(&vec![0xABCDu16; n * QE]).unwrap();
    let mut ws = DevBuf::zeroed(decode_workspace_bytes_for(SHAPE, n, max_ctx)).unwrap();
    attn_decode(&qd, &w.cache, &w.table, &ri, n, &mut out, max_ctx, &mut ws, &w.stream).unwrap();
    w.stream.sync().unwrap();
    out.to_vec::<u16>().unwrap()
}

/// Prefill of one sequence in the given `(start, rows)` chunks (consecutive); returns all rows' outputs.
fn run_prefill(w: &World, si: usize, chunks: &[(usize, usize)]) -> Vec<u16> {
    let mut all = Vec::new();
    for &(start, n) in chunks {
        let mut q = Vec::new();
        for p in start..start + n {
            q.extend(qvec(w.seqs[si].sid, p));
        }
        let qd = DevBuf::from_slice(&q).unwrap();
        let mut out = DevBuf::zeroed(n * QE * 2).unwrap();
        attn_prefill(&qd, &w.cache, &w.table, si, start, n, &mut out, &w.stream).unwrap();
        w.stream.sync().unwrap();
        all.extend(out.to_vec::<u16>().unwrap());
    }
    all
}

/// f64 reference of one query row (24 heads) at `pos` against the sequence's bf16 K/V.
fn ref_row(s: &Seq, pos: usize, q: &[u16]) -> Vec<f32> {
    let n = pos + 1;
    let mut out = vec![0f32; QE];
    for h in 0..NQ {
        let hk = h / GRP;
        let qh: Vec<f64> = q[h * HEAD_DIM..(h + 1) * HEAD_DIM].iter().map(|&b| bf16_to_f32(b) as f64).collect();
        let mut sc = vec![0f64; n];
        for (j, scj) in sc.iter_mut().enumerate() {
            let kr = &s.k[j * TE + hk * HEAD_DIM..][..HEAD_DIM];
            *scj = qh.iter().zip(kr).map(|(a, &b)| a * bf16_to_f32(b) as f64).sum::<f64>() / 16.0;
        }
        let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut l = 0f64;
        let mut acc = vec![0f64; HEAD_DIM];
        for (j, &scj) in sc.iter().enumerate() {
            let p = (scj - mx).exp();
            l += p;
            let vr = &s.v[j * TE + hk * HEAD_DIM..][..HEAD_DIM];
            for (a, &x) in acc.iter_mut().zip(vr) {
                *a += p * bf16_to_f32(x) as f64;
            }
        }
        for d in 0..HEAD_DIM {
            out[h * HEAD_DIM + d] = (acc[d] / l) as f32;
        }
    }
    out
}

fn f32s(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| bf16_to_f32(b)).collect()
}

fn assert_bits_eq(what: &str, got: &[u16], want: &[u16]) {
    assert_eq!(got.len(), want.len());
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        panic!(
            "{what}: bits differ at element {i} (head {}, dim {}): {:#06x} vs {:#06x}",
            i / HEAD_DIM,
            i % HEAD_DIM,
            got[i],
            want[i]
        );
    }
}

const CONTEXTS: [usize; 9] = [1, 255, 256, 257, 511, 512, 513, 2000, 5000];

static WORST: std::sync::Mutex<(f64, f64)> = std::sync::Mutex::new((1.0, 0.0));

fn check_vs_ref(what: &str, s: &Seq, pos: usize, got: &[u16]) {
    let want = ref_row(s, pos, &qvec(s.sid, pos));
    let (cos, rel) = cos_rel_rms(&f32s(got), &want);
    {
        let mut w = WORST.lock().unwrap();
        *w = (w.0.min(cos), w.1.max(rel));
    }
    assert!(cos >= 0.99995 && rel <= 5e-3, "{what}: len {} pos {pos}: cos {cos} relRMS {rel}", s.len);
}

// ---------------------------------------------------------------------------------------------------- (a)

#[test]
fn decode_matches_f64_reference() {
    if gpu().is_none() {
        return;
    }
    let w = make_world(seqs_of(&CONTEXTS), 7, 5);
    // the last position of every sequence, all in one launch (nine streams, nine slots)
    let rows: Vec<(usize, usize)> = CONTEXTS.iter().enumerate().map(|(i, &l)| (i, l - 1)).collect();
    let out = run_decode(&w, &rows);
    for (r, &(si, p)) in rows.iter().enumerate() {
        check_vs_ref("last pos", &w.seqs[si], p, &out[r * QE..(r + 1) * QE]);
    }
    eprintln!("decode vs f64 (so far): worst (cos, relRMS) = {:?}", *WORST.lock().unwrap());
    // interior positions around the chunk / page edges
    let mut rows = Vec::new();
    for (si, &len) in CONTEXTS.iter().enumerate() {
        for p in [0, 255, 256, 511, 512, 1023, 1500, 4999] {
            if p < len {
                rows.push((si, p));
            }
        }
    }
    for part in rows.chunks(97) {
        let out = run_decode(&w, part);
        for (r, &(si, p)) in part.iter().enumerate() {
            check_vs_ref("interior", &w.seqs[si], p, &out[r * QE..(r + 1) * QE]);
        }
    }
}

#[test]
fn prefill_matches_f64_reference() {
    if gpu().is_none() {
        return;
    }
    let w = make_world(seqs_of(&CONTEXTS), 11, 3);
    for (si, &len) in CONTEXTS.iter().enumerate() {
        let chunks: Vec<(usize, usize)> = if len > 4096 { vec![(0, 4096), (4096, len - 4096)] } else { vec![(0, len)] };
        let out = run_prefill(&w, si, &chunks);
        let mut rows: Vec<usize> = vec![
            0,
            len - 1,
            len / 2,
            len.saturating_sub(17),
            255.min(len - 1),
            256.min(len - 1),
            4095.min(len - 1),
            4096.min(len - 1),
        ];
        let mut r = Rng(len as u64);
        rows.extend((0..6).map(|_| (r.next_u64() % len as u64) as usize));
        for p in rows {
            check_vs_ref("prefill", &w.seqs[si], p, &out[p * QE..(p + 1) * QE]);
        }
    }
    eprintln!("prefill vs f64 (so far): worst (cos, relRMS) = {:?}", *WORST.lock().unwrap());
}

// ---------------------------------------------------------------------------------------------------- (b)

#[test]
fn decode_row_invariance_bit_exact() {
    if gpu().is_none() {
        return;
    }
    let lens = [2000usize, 5000, 700, 513];
    let w1 = make_world(seqs_of(&lens), 1, 4);
    let w2 = make_world(seqs_of(&lens), 99, 11); // different block placement, same content
    let alone = |w: &World, si: usize, p: usize| run_decode(w, &[(si, p)]);
    let probes = [(0usize, 1999usize), (0, 1500), (0, 511), (1, 4999), (1, 4000), (1, 600), (2, 699), (2, 0), (3, 512), (3, 3)];
    let base: Vec<Vec<u16>> = probes.iter().map(|&(s, p)| alone(&w1, s, p)).collect();
    // different block placement
    for (i, &(s, p)) in probes.iter().enumerate() {
        assert_bits_eq("placement", &alone(&w2, s, p), &base[i]);
    }
    // windows of 2..128 consecutive rows of one sequence, the probe at different offsets inside
    // (a CTA serves 4, 6, 8 or 12 rows at groups of 12, 8, 6 or 4: every size and its neighbours are in the list)
    for &wn in &[2usize, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 31, 64, 100, 128] {
        for &(si, p) in &[(1usize, 4999usize), (0, 1999), (1, 4000), (0, 511)] {
            for off in [0usize, wn / 2, wn - 1] {
                if p < off || p - off + wn > lens[si] {
                    continue;
                }
                let p0 = p - off;
                let rows: Vec<(usize, usize)> = (0..wn).map(|r| (si, p0 + r)).collect();
                let out = run_decode(&w1, &rows);
                let want = alone(&w1, si, p);
                assert_bits_eq(&format!("window {wn} off {off}"), &out[off * QE..(off + 1) * QE], &want);
                // and the same window in the other placement
                let out2 = run_decode(&w2, &rows);
                assert_bits_eq("window placement", &out2[off * QE..(off + 1) * QE], &want);
            }
        }
    }
    // mixed batches: random rows of all sequences in random order
    let mut r = Rng(5);
    for trial in 0..12 {
        let n = [1usize, 2, 7, 8, 9, 24, 50, 128][trial % 8];
        let mut rows: Vec<(usize, usize)> = (0..n)
            .map(|_| {
                let si = (r.next_u64() % lens.len() as u64) as usize;
                (si, (r.next_u64() % lens[si] as u64) as usize)
            })
            .collect();
        if trial % 3 == 0 {
            rows.sort(); // contiguous runs per slot
        }
        let out = run_decode(&w1, &rows);
        for (i, &(si, p)) in rows.iter().enumerate() {
            assert_bits_eq(&format!("batch {trial} row {i}"), &out[i * QE..(i + 1) * QE], &alone(&w1, si, p));
        }
    }
}

/// A verify window written to the cache at P..P+W-1 over stale data attends exactly like serial rows.
#[test]
fn verify_window_equals_serial() {
    if gpu().is_none() {
        return;
    }
    let len = 3000usize;
    let serial_world = make_world(seqs_of(&[len]), 3, 2);
    for &(p, wn) in &[(1000usize, 8usize), (505, 8), (2043, 8), (255, 16), (2990, 10), (0, 8), (511, 128)] {
        // the window cache: sequence 1's true keys below P, stale (different) data from P on, window then written
        let (k, v) = gen_kv(1, len);
        let (sk, sv) = gen_kv(777, len);
        let mut mixed = Seq { sid: 1, len, k: k.clone(), v: v.clone() };
        mixed.k[p * TE..].copy_from_slice(&sk[p * TE..]);
        mixed.v[p * TE..].copy_from_slice(&sv[p * TE..]);
        let ww = make_world(vec![mixed], 21, 6);
        // write the window rows' true K/V over positions P..P+W-1
        let hi = p + wn.min(len - p);
        let nw = hi - p;
        let kd = DevBuf::from_slice(&k[p * TE..hi * TE]).unwrap();
        let vd = DevBuf::from_slice(&v[p * TE..hi * TE]).unwrap();
        let ri = RowIndex::from_host(&vec![0; nw], &(p as i32..hi as i32).collect::<Vec<_>>()).unwrap();
        let mut ww = ww;
        kv_write(&kd, &vd, &ri, &ww.table, &mut ww.cache, nw, &ww.stream).unwrap();
        ww.stream.sync().unwrap();
        let rows: Vec<(usize, usize)> = (p..hi).map(|x| (0, x)).collect();
        let out = run_decode(&ww, &rows);
        for (r, &(_, pos)) in rows.iter().enumerate() {
            let want = run_decode(&serial_world, &[(0, pos)]);
            assert_bits_eq(&format!("window P={p} row {r}"), &out[r * QE..(r + 1) * QE], &want);
        }
    }
}

/// A padding row (position -1, what a CUDA-graph bucket uses) is inert: real rows keep their bits, the padding row's
/// output is zeros, and `kv_write` of a padding row leaves the cache alone.
#[test]
fn padding_rows_are_inert() {
    if gpu().is_none() {
        return;
    }
    let lens = [2000usize, 700];
    let mut w = make_world(seqs_of(&lens), 17, 3);
    let real = [(0usize, 1999i32), (0, 1500), (1, 699), (0, 3)];
    let alone: Vec<Vec<u16>> = real.iter().map(|&r| run_decode_pad(&w, &[r])).collect();
    for pad_at in 0..=real.len() {
        for pad_slot in [0usize, 1] {
            let mut rows = real.to_vec();
            rows.insert(pad_at, (pad_slot, -1));
            let out = run_decode_pad(&w, &rows);
            let mut ri = 0;
            for (i, &(_, p)) in rows.iter().enumerate() {
                let got = &out[i * QE..(i + 1) * QE];
                if p < 0 {
                    assert!(got.iter().all(|&b| b == 0), "padding row {i} must be zeros");
                } else {
                    assert_bits_eq(&format!("pad at {pad_at}"), got, &alone[ri]);
                    ri += 1;
                }
            }
        }
    }
    // kv_write skips a padding row: write garbage "at position -1" and re-run
    let (gk, gv) = gen_kv(555, 2);
    let (kd, vd) = (DevBuf::from_slice(&gk).unwrap(), DevBuf::from_slice(&gv).unwrap());
    let ri = RowIndex::from_host(&[0, 0], &[-1, -1]).unwrap();
    kv_write(&kd, &vd, &ri, &w.table, &mut w.cache, 2, &w.stream).unwrap();
    w.stream.sync().unwrap();
    for (i, &r) in real.iter().enumerate() {
        assert_bits_eq("after padding kv_write", &run_decode_pad(&w, &[r]), &alone[i]);
    }
}

/// ~100k tokens: 196 chunks of 512 merged in order, block ids past 1000, prefill at a large start.
#[test]
fn long_context_matches_f64_reference() {
    if gpu().is_none() {
        return;
    }
    let len = 100_000usize;
    let w = make_world(seqs_of(&[len]), 23, 700);
    assert!(w.cache.n_blocks > 1000);
    let rows: Vec<(usize, usize)> = vec![(0, len - 1), (0, len - 2), (0, 65_535), (0, 511)];
    let out = run_decode(&w, &rows);
    for (r, &(si, p)) in rows.iter().enumerate() {
        check_vs_ref("long decode", &w.seqs[si], p, &out[r * QE..(r + 1) * QE]);
    }
    let start = len - 200;
    let pre = run_prefill(&w, 0, &[(start, 64)]);
    for p in [0usize, 17, 63] {
        check_vs_ref("long prefill", &w.seqs[0], start + p, &pre[p * QE..(p + 1) * QE]);
    }
    // a prefill chunk at a large start equals the same rows inside one longer call
    let whole = run_prefill(&w, 0, &[(start - 100, 164)]);
    assert_bits_eq("long prefill chunk invariance", &pre, &whole[100 * QE..164 * QE]);
}

/// The DSpark draft's block (and any window whose rows all see the same keys): R query rows of ONE slot with the same visibility
/// bound, distinct queries. Each row equals the row run alone (wherever it sits in the block, whatever the others are, with another
/// stream's rows mixed into the launch) and matches the f64 reference at that bound.
#[test]
fn block_rows_sharing_one_visibility_bound_equal_their_solo_rows() {
    if gpu().is_none() {
        return;
    }
    let lens = [3000usize, 700];
    let w = make_world(seqs_of(&lens), 29, 3);
    let mut checked = 0;
    for &(si, bound, rows) in
        &[(0usize, 2999i32, 15usize), (0, 1500, 15), (1, 699, 12), (1, 100, 7), (0, 511, 15), (0, 0, 15), (0, 2048, 1)]
    {
        let q: Vec<u16> = (0..rows).flat_map(|r| qvec(0x5000 + r as u64, bound as usize)).collect();
        let rws: Vec<(usize, i32)> = vec![(si, bound); rows];
        let out = decode_with_q(&w, &rws, &q);
        // the same block with another stream's rows interleaved (positions of its own), in one launch
        let other_si = 1 - si;
        let mut mixed_rows = Vec::new();
        let mut mixed_q = Vec::new();
        for r in 0..rows {
            mixed_rows.push((si, bound));
            mixed_q.extend_from_slice(&q[r * QE..(r + 1) * QE]);
            if r % 4 == 1 {
                let p = (17 * r + 5) as i32 % lens[other_si] as i32;
                mixed_rows.push((other_si, p));
                mixed_q.extend(qvec(0x6000 + r as u64, p as usize));
            }
        }
        let mixed = decode_with_q(&w, &mixed_rows, &mixed_q);
        let mut at = 0;
        for r in 0..rows {
            let qr = &q[r * QE..(r + 1) * QE];
            let solo = decode_with_q(&w, &[(si, bound)], qr);
            assert_bits_eq(
                &format!("seq {si} bound {bound} row {r} of {rows}: block vs solo"),
                &out[r * QE..(r + 1) * QE],
                &solo,
            );
            while mixed_rows[at] != (si, bound) || mixed_q[at * QE..(at + 1) * QE] != *qr {
                at += 1;
            }
            assert_bits_eq(
                &format!("seq {si} bound {bound} row {r}: mixed launch vs solo"),
                &mixed[at * QE..(at + 1) * QE],
                &solo,
            );
            at += 1;
            let want = ref_row(&w.seqs[si], bound as usize, qr);
            let (cos, rel) = cos_rel_rms(&f32s(&out[r * QE..(r + 1) * QE]), &want);
            assert!(cos >= 0.99995 && rel <= 5e-3, "block row {r} at bound {bound}: cos {cos} relRMS {rel}");
            checked += 1;
        }
    }
    assert!(checked >= 80, "{checked}");
}

/// Every prefill CTA shape the build has gives the f64 reference's answer, and one it does not have (6 or 3 heads a CTA at groups of
/// 8 and 4: they divide neither) is refused with an error instead of launching a kernel that was never instantiated.
#[test]
fn prefill_variants_match_the_f64_reference_or_are_refused() {
    let Some(_) = gpu() else { return };
    let len = 700usize;
    let w = make_world(seqs_of(&[len]), 41, 2);
    let q: Vec<u16> = (0..len).flat_map(|p| qvec(w.seqs[0].sid, p)).collect();
    let qd = DevBuf::from_slice(&q).unwrap();
    let mut supported = 0;
    for variant in 0..5i32 {
        let mut out = DevBuf::from_slice(&vec![0xABCDu16; len * QE]).unwrap();
        let r = attn_prefill_variant(&qd, &w.cache, &w.table, 0, 0, len, &mut out, variant, &w.stream);
        w.stream.sync().unwrap();
        if SHAPE.prefill_variant_supported(variant) {
            r.unwrap_or_else(|e| panic!("variant {variant} is a shape of this build but failed: {e}"));
            let got = out.to_vec::<u16>().unwrap();
            for p in [0usize, 1, 63, 64, 255, 256, 511, 699] {
                check_vs_ref(&format!("prefill variant {variant}"), &w.seqs[0], p, &got[p * QE..(p + 1) * QE]);
            }
            supported += 1;
        } else {
            assert!(r.is_err(), "variant {variant} does not divide the query group of {GRP}: it must be refused");
            assert!(out.to_vec::<u16>().unwrap().iter().all(|&b| b == 0xABCD), "a refused call wrote its output");
        }
    }
    assert_eq!(supported, (0..5).filter(|&v| SHAPE.prefill_variant_supported(v)).count());
    assert!(supported >= 3);
    // and the stream is healthy after a refusal
    let again = run_decode(&w, &[(0, len - 1)]);
    check_vs_ref("decode after a refused variant", &w.seqs[0], len - 1, &again);
}

// ---------------------------------------------------------------------------------------------------- (c)

#[test]
fn prefill_chunk_invariant_and_agrees_with_decode() {
    if gpu().is_none() {
        return;
    }
    let len = 4000usize;
    let w = make_world(seqs_of(&[len, 900]), 13, 3);
    let whole = run_prefill(&w, 0, &[(0, len)]);
    for split in [
        vec![(0usize, 1000usize), (1000, 1777), (2777, 1223)],
        vec![(0, 1), (1, 4095 - 1000), (3096, 904)],
        vec![(0, 63), (63, 64), (127, 3873)],
    ] {
        let chunked = run_prefill(&w, 0, &split);
        assert_bits_eq("chunked prefill", &chunked, &whole);
    }
    // single-row calls at assorted positions
    for p in [0usize, 1, 15, 16, 63, 64, 255, 256, 511, 512, 1999, 3999] {
        let one = run_prefill(&w, 0, &[(p, 1)]);
        assert_bits_eq("single row", &one, &whole[p * QE..(p + 1) * QE]);
    }
    // the second sequence is unaffected by where it sits in the batch of slots
    let a = run_prefill(&w, 1, &[(0, 900)]);
    let b = run_prefill(&w, 1, &[(0, 400), (400, 500)]);
    assert_bits_eq("seq 1 chunked", &a, &b);
    // prefill vs decode: close, within bf16 tolerance (they are different arithmetic)
    let rows: Vec<(usize, usize)> = (0..128).map(|i| (0, len - 128 + i)).collect();
    let dec = run_decode(&w, &rows);
    let pre = &whole[(len - 128) * QE..];
    let (cos, rel) = cos_rel_rms(&f32s(pre), &f32s(&dec));
    eprintln!("prefill vs decode: cos {cos} relRMS {rel}");
    assert!(cos >= 0.99998 && rel <= 4e-3, "cos {cos} relRMS {rel}");
}

// ---------------------------------------------------------------------------------------------------- (d)

fn dump_case(prompt: &str, prefill: bool) {
    if SHAPE != AttnShape::Q24Kv4 {
        eprintln!("not run: the layer-3 dumps are Qwen3.8-27B's (4 kv heads)");
        return;
    }
    if !have_refs() {
        eprintln!("no reference dumps: skipping");
        return;
    }
    let Some(_) = gpu() else { return };
    let rd = |n: &str| ref_npy(&format!("{prompt}/sub/L03.{n}.npy")).to_f32();
    let (qo, ko, vo, qp, oin) = (rd("rope.q_out"), rd("rope.k_out"), rd("v_proj.out"), rd("q_proj.out"), rd("o_proj.in"));
    let t = ko.len() / TE;
    let bf = |v: &[f32]| -> Vec<u16> { v.iter().map(|&x| f32_to_bf16(x)).collect() };
    let seq = Seq { sid: 1, len: t, k: bf(&ko), v: bf(&vo) };
    let w = make_world(vec![seq], 5, 2);
    let qb = bf(&qo);
    let qd = DevBuf::from_slice(&qb).unwrap();
    let mut out = DevBuf::zeroed(t * QE * 2).unwrap();
    if prefill {
        attn_prefill(&qd, &w.cache, &w.table, 0, 0, t, &mut out, &w.stream).unwrap();
    } else {
        let ri = RowIndex::from_host(&vec![0; t], &(0..t as i32).collect::<Vec<_>>()).unwrap();
        let mut ws = DevBuf::zeroed(decode_workspace_bytes_for(SHAPE, t, t)).unwrap();
        attn_decode(&qd, &w.cache, &w.table, &ri, t, &mut out, t, &mut ws, &w.stream).unwrap();
    }
    w.stream.sync().unwrap();
    let o = f32s(&out.to_vec::<u16>().unwrap());
    // o * sigmoid(gate): the gate is the second half of each head's 512 of q_proj.out
    let mut gated = vec![0f32; t * QE];
    for ti in 0..t {
        for h in 0..NQ {
            for d in 0..HEAD_DIM {
                let g = qp[ti * 12288 + h * 512 + 256 + d];
                gated[ti * QE + h * HEAD_DIM + d] = o[ti * QE + h * HEAD_DIM + d] / (1.0 + (-g).exp());
            }
        }
    }
    let (cos, rel) = cos_rel_rms(&gated, &oin);
    eprintln!("{prompt} {}: cos {cos:.7} relRMS {rel:.3e}", if prefill { "prefill" } else { "decode" });
    assert!(cos >= 0.9999 && rel <= 1e-2, "{prompt}: cos {cos} relRMS {rel}");
}

#[test]
fn layer3_dump_short_prefill() {
    dump_case("short", true);
}
#[test]
fn layer3_dump_long_prefill() {
    dump_case("long", true);
}
#[test]
fn layer3_dump_short_decode() {
    dump_case("short", false);
}
#[test]
fn layer3_dump_long_decode() {
    dump_case("long", false);
}

// ---------------------------------------------------------------------------------------------------- (e)

/// A cache of `ctx` tokens of repeated random K/V in one sequence (contiguous blocks), for timing.
fn bench_world(ctx: usize) -> (PagedKv, BlockTable, Stream) {
    let stream = Stream::new().unwrap();
    let nb = ctx.div_ceil(BLOCK_TOKENS);
    let mut cache = PagedKv::with_shape(nb, SHAPE).unwrap();
    let table = BlockTable::from_host(&[(0..nb as u32).collect()], nb).unwrap();
    let (k, v) = gen_kv(42, 4096.min(ctx));
    let (kd, vd) = (DevBuf::from_slice(&k).unwrap(), DevBuf::from_slice(&v).unwrap());
    let mut done = 0;
    while done < ctx {
        let n = 4096.min(ctx - done);
        let pos: Vec<i32> = (done as i32..(done + n) as i32).collect();
        let ri = RowIndex::from_host(&vec![0; n], &pos).unwrap();
        kv_write(&kd, &vd, &ri, &table, &mut cache, n, &stream).unwrap();
        stream.sync().unwrap();
        done += n;
    }
    (cache, table, stream)
}

fn time_ms(stream: &Stream, iters: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..3 {
        f();
    }
    stream.sync().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    stream.sync().unwrap();
    t0.elapsed().as_secs_f64() * 1e3 / iters as f64
}

#[test]
#[ignore]
fn bench_decode() {
    if gpu().is_none() {
        return;
    }
    for ctx in [1024usize, 8192, 32768, 131072, 262144] {
        let (cache, table, stream) = bench_world(ctx);
        for rows in [1usize, 8] {
            let pos: Vec<i32> = (0..rows).map(|r| (ctx - rows + r) as i32).collect();
            let ri = RowIndex::from_host(&vec![0; rows], &pos).unwrap();
            let q: Vec<u16> = (0..rows).flat_map(|r| qvec(1, ctx - rows + r)).collect();
            let qd = DevBuf::from_slice(&q).unwrap();
            let mut out = DevBuf::zeroed(rows * QE * 2).unwrap();
            let mut ws = DevBuf::zeroed(decode_workspace_bytes_for(SHAPE, rows, ctx)).unwrap();
            let ms =
                time_ms(&stream, 20, || attn_decode(&qd, &cache, &table, &ri, rows, &mut out, ctx, &mut ws, &stream).unwrap());
            let bytes = (ctx * TE * 2 * 2) as f64;
            eprintln!(
                "decode ctx {ctx:>6} rows {rows}: {ms:8.3} ms  {:6.1} GB/s of unique KV ({:.1} MiB)",
                bytes / ms / 1e6,
                bytes / 1048576.0
            );
        }
    }
}

fn prefill_flops(start: usize, rows: usize) -> f64 {
    // 4 flops per (q.k + p.v) mac pair x head dim x heads x causal key count
    let keys: f64 = (start..start + rows).map(|p| (p + 1) as f64).sum();
    4.0 * HEAD_DIM as f64 * NQ as f64 * keys
}

#[test]
#[ignore]
fn bench_prefill() {
    if gpu().is_none() {
        return;
    }
    for (ctx, rows) in [(8192usize, 4096usize), (32768, 4096), (131072, 4096), (262144, 4096)] {
        let (cache, table, stream) = bench_world(ctx);
        let start = ctx - rows;
        let q: Vec<u16> = (0..rows).flat_map(|r| qvec(1, start + r)).collect();
        let qd = DevBuf::from_slice(&q).unwrap();
        let mut out = DevBuf::zeroed(rows * QE * 2).unwrap();
        let variants: Vec<i32> =
            if ctx == 8192 { (0..5).filter(|&v| SHAPE.prefill_variant_supported(v)).collect() } else { vec![0] };
        for variant in variants {
            let iters = if ctx > 100_000 { 4 } else { 8 };
            let ms = time_ms(&stream, iters, || {
                attn_prefill_variant(&qd, &cache, &table, 0, start, rows, &mut out, variant, &stream).unwrap()
            });
            eprintln!(
                "prefill ctx {ctx:>6} rows {rows} variant {variant}: {ms:9.3} ms  {:6.2} TFLOPS (causal)",
                prefill_flops(start, rows) / ms / 1e9
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------------- tree windows

use gb_cuda::tree::{random_parents, Tree, PATH_STRIDE};

/// bf16 K/V rows `[rows][kv heads][256]` of a tree window's nodes (a different stream from any committed sequence's).
fn node_kv(seed: u64, rows: usize) -> (Vec<u16>, Vec<u16>) {
    gen_kv(0x7000 + seed, rows)
}

/// Decode of explicit q rows: `rows = [(seq index, position)]`, `q` their query rows (the explicit-q twin of `run_decode_pad`).
fn decode_with_q(w: &World, rows: &[(usize, i32)], q: &[u16]) -> Vec<u16> {
    let n = rows.len();
    let max_ctx = rows.iter().map(|r| (r.1 + 1).max(1) as usize).max().unwrap();
    let max_ctx = (max_ctx + (max_ctx % 3) * 700).min(w.table.max_blocks * BLOCK_TOKENS);
    let slot: Vec<i32> = rows.iter().map(|r| r.0 as i32).collect();
    let pos: Vec<i32> = rows.iter().map(|r| r.1).collect();
    let ri = RowIndex::from_host(&slot, &pos).unwrap();
    let qd = DevBuf::from_slice(q).unwrap();
    let mut out = DevBuf::from_slice(&vec![0xABCDu16; n * QE]).unwrap();
    let mut ws = DevBuf::zeroed(decode_workspace_bytes_for(SHAPE, n, max_ctx)).unwrap();
    attn_decode(&qd, &w.cache, &w.table, &ri, n, &mut out, max_ctx, &mut ws, &w.stream).unwrap();
    w.stream.sync().unwrap();
    out.to_vec::<u16>().unwrap()
}

/// The tree attention over `trees = [(seq index, tree)]` of a world whose committed length of sequence `s` is `w.seqs[s].len`: q rows
/// are `qvec(0x4000 + global row, depth)`, window K/V `node_kv`. Returns every row's output.
struct TreeRun {
    out: Vec<u16>,
    q: Vec<u16>,
    kn: Vec<u16>,
    vn: Vec<u16>,
    starts: Vec<usize>,
}

fn run_tree_attention(w: &World, trees: &[(usize, Tree)], seed: u64, q_override: Option<&[u16]>) -> TreeRun {
    let rows: usize = trees.iter().map(|t| t.1.len()).sum();
    let (kn, vn) = node_kv(seed, rows);
    let (mut slot, mut pos, mut base, mut paths, mut q, mut starts) =
        (vec![], vec![], vec![], vec![-1i32; rows * PATH_STRIDE], vec![], vec![]);
    let mut r0 = 0usize;
    for (si, t) in trees {
        starts.push(r0);
        let l = w.seqs[*si].len;
        for i in 0..t.len() {
            let r = r0 + i;
            slot.push(*si as i32);
            pos.push((l + t.depth(i)) as i32);
            base.push(l as i32);
            for (j, a) in t.path(i).into_iter().enumerate() {
                paths[r * PATH_STRIDE + j] = (r0 + a) as i32;
            }
            q.extend(qvec(0x4000 + r as u64, t.depth(i)));
        }
        r0 += t.len();
    }
    if let Some(o) = q_override {
        q = o.to_vec();
    }
    let max_ctx = (pos.iter().max().unwrap() + 1) as usize;
    let max_ctx = (max_ctx + (max_ctx % 3) * 700).min(w.table.max_blocks * BLOCK_TOKENS);
    let ri = RowIndex::from_host(&slot, &pos).unwrap();
    let (qd, knd, vnd) = (DevBuf::from_slice(&q).unwrap(), DevBuf::from_slice(&kn).unwrap(), DevBuf::from_slice(&vn).unwrap());
    let (based, pathsd) = (DevBuf::from_slice(&base).unwrap(), DevBuf::from_slice(&paths).unwrap());
    let mut out = DevBuf::from_slice(&vec![0xABCDu16; rows * QE]).unwrap();
    let mut ws = DevBuf::zeroed(decode_workspace_bytes_for(SHAPE, rows, max_ctx)).unwrap();
    let tr = TreeRows { base: &based, paths: &pathsd, stride: PATH_STRIDE };
    attn_decode_tree(&qd, &w.cache, &w.table, &ri, &tr, &knd, &vnd, rows, &mut out, max_ctx, &mut ws, &w.stream).unwrap();
    w.stream.sync().unwrap();
    TreeRun { out: out.to_vec::<u16>().unwrap(), q, kn, vn, starts }
}

/// Every row of every tree equals the chain decode of its own path: the committed prefix in the cache, the path's window rows written
/// at L..L+depth, a plain `attn_decode` over those rows. The tree world's cache holds NaN at and above L (never written), so a tree
/// kernel that read a position it should not would fail. `l` positions straddle chunk (512), page (256) and tile (64) boundaries.
fn check_tree_attention(lens: &[usize], trees: &[(usize, Tree)], seed: u64, placement: u64) -> usize {
    let tw = make_world(seqs_of(lens), placement, 4);
    let run = run_tree_attention(&tw, trees, seed, None);
    // the reference world: the same committed prefix, rewritten from L per leaf path
    let mut rw = make_world(seqs_of(lens), placement ^ 0x55, 4 + 130);
    let mut checked = 0;
    for (ti, (si, t)) in trees.iter().enumerate() {
        let l = lens[*si];
        let leaves: Vec<usize> = (0..t.len()).filter(|&i| !t.parents().contains(&(i as i32))).collect();
        for &leaf in &leaves {
            let path = t.path(leaf);
            let pr: Vec<usize> = path.iter().map(|&i| run.starts[ti] + i).collect();
            // write the path's window rows at L..L+depth (bf16 rows of the node buffers)
            let mut k = Vec::new();
            let mut v = Vec::new();
            for &r in &pr {
                k.extend_from_slice(&run.kn[r * TE..(r + 1) * TE]);
                v.extend_from_slice(&run.vn[r * TE..(r + 1) * TE]);
            }
            let (kd, vd) = (DevBuf::from_slice(&k).unwrap(), DevBuf::from_slice(&v).unwrap());
            let ri = RowIndex::from_host(&vec![*si as i32; path.len()], &(l as i32..(l + path.len()) as i32).collect::<Vec<_>>())
                .unwrap();
            // the block table needs pages for the window: extend the sequence's table (spare blocks exist)
            ensure_pages(&mut rw, *si, l + path.len());
            kv_write(&kd, &vd, &ri, &rw.table, &mut rw.cache, path.len(), &rw.stream).unwrap();
            rw.stream.sync().unwrap();
            let rows: Vec<(usize, i32)> = (0..path.len()).map(|j| (*si, (l + j) as i32)).collect();
            let mut q = Vec::new();
            for &r in &pr {
                q.extend_from_slice(&run.q[r * QE..(r + 1) * QE]);
            }
            let want = decode_with_q(&rw, &rows, &q);
            for (j, &r) in pr.iter().enumerate() {
                assert_bits_eq(
                    &format!("seq {si} L {l} leaf {leaf} node {} (depth {j}) of {} rows", path[j], t.len()),
                    &run.out[r * QE..(r + 1) * QE],
                    &want[j * QE..(j + 1) * QE],
                );
                checked += 1;
            }
        }
    }
    checked
}

/// Give sequence `si` of the world pages up to `len` tokens (a spare block per missing page).
fn ensure_pages(w: &mut World, si: usize, len: usize) {
    let have = w.seqs[si].len.div_ceil(BLOCK_TOKENS);
    let need = len.div_ceil(BLOCK_TOKENS);
    if need <= have {
        return;
    }
    let mut tables: Vec<Vec<u32>> = Vec::new();
    let flat = w.table.buf.to_vec::<i32>().unwrap();
    for s in 0..w.seqs.len() {
        tables.push(flat[s * w.table.max_blocks..(s + 1) * w.table.max_blocks].iter().map(|&b| b as u32).collect());
    }
    // blocks 0..used + spare are shuffled ids; find ids not in any table
    let taken: std::collections::HashSet<u32> = tables.iter().flat_map(|t| t.iter().copied()).filter(|&b| b != POISON).collect();
    let mut free = (0..w.cache.n_blocks as u32).filter(|b| !taken.contains(b));
    for p in have..need {
        tables[si][p] = free.next().expect("a spare block");
    }
    w.table.upload(&tables).unwrap();
}

#[test]
fn tree_attention_equals_per_path_chain_decode_bit_for_bit() {
    if gpu().is_none() {
        return;
    }
    let mut checked = 0;
    // committed lengths that put the window across a chunk (512), page (256) and tile (64) boundary, and deep inside one
    let lens = [2000usize, 511, 513, 256, 5100, 63, 65, 1];
    for (case, &l) in lens.iter().enumerate() {
        // a tree of up to 40 rows, branching 1..4, in sequence 0 of its own world together with a second sequence
        let n = [1usize, 2, 8, 9, 17, 24, 40, 33][case];
        let b = [1usize, 2, 3, 4][case % 4];
        let t = Tree::new(&random_parents(case as u64 + 1, n, b)).unwrap();
        let other = Tree::new(&random_parents(case as u64 + 50, 7, 2)).unwrap();
        checked += check_tree_attention(&[l, 700], &[(0, t), (1, other)], case as u64, 31 + case as u64);
    }
    assert!(checked >= 100, "the test must cover many rows ({checked})");
}

/// The window spills across a chunk boundary (rows at L..L+depth with L = 508 .. 511 and a deep tree) and a whole 64-row tree in one
/// sequence: the shape of the biggest single-stream tree.
#[test]
fn tree_attention_across_a_chunk_boundary_and_at_64_rows() {
    if gpu().is_none() {
        return;
    }
    let mut checked = 0;
    for (case, &l) in [508usize, 509, 510, 511, 512, 1020, 1023, 4095].iter().enumerate() {
        let t = Tree::new(&random_parents(100 + case as u64, 64, 3)).unwrap();
        checked += check_tree_attention(&[l], &[(0, t)], 9 + case as u64, 7 + case as u64);
    }
    // a chain of 8 rows hanging across the boundary
    let c = Tree::chain(8).unwrap();
    checked += check_tree_attention(&[509], &[(0, c)], 77, 5);
    assert!(checked >= 200, "{checked}");
}

/// Chain == tree through the attention: the window the decode kernel reads from the cache (written at P..P+W-1) and the same window
/// as node buffers give the same bits.
#[test]
fn a_chain_through_the_tree_attention_equals_the_decode_kernel() {
    if gpu().is_none() {
        return;
    }
    let len = 3000usize;
    for &(p, wn) in &[(1000usize, 8usize), (505, 8), (2043, 8), (255, 16), (2990, 10), (511, 32), (1, 8)] {
        let hi = p + wn.min(len - p);
        let nw = hi - p;
        let (k, v) = gen_kv(1, len);
        // the cache world: true keys below p, then the window written at p.. (what the decode kernel reads)
        let mut full = make_world(seqs_of(&[len]), 3, 2);
        let rows: Vec<(usize, i32)> = (p..hi).map(|x| (0, x as i32)).collect();
        let mut q = Vec::new();
        for r in 0..nw {
            q.extend(qvec(0x4000 + r as u64, r));
        }
        // the decode kernel's window: sequence content up to p is the committed prefix; positions p.. hold the node rows
        let (kn, vn) = node_kv(5, nw);
        let (kd, vd) = (DevBuf::from_slice(&kn).unwrap(), DevBuf::from_slice(&vn).unwrap());
        let ri = RowIndex::from_host(&vec![0; nw], &(p as i32..hi as i32).collect::<Vec<_>>()).unwrap();
        kv_write(&kd, &vd, &ri, &full.table, &mut full.cache, nw, &full.stream).unwrap();
        full.stream.sync().unwrap();
        let want = decode_with_q(&full, &rows, &q);
        // the tree world: only the committed prefix (positions below p) of the same sequence; the window rows come from the node buffers
        let prefix = Seq { sid: 1, len: p, k: k[..p * TE].to_vec(), v: v[..p * TE].to_vec() };
        let tw = make_world(vec![prefix], 21, 6);
        let t = Tree::chain(nw).unwrap();
        // the node buffers `run_tree_attention` generates must be the same rows: it asks node_kv(seed, rows); use seed 5
        let run = run_tree_attention(&tw, &[(0, t)], 5, Some(&q));
        for r in 0..nw {
            assert_bits_eq(&format!("chain P={p} W={wn} row {r}"), &run.out[r * QE..(r + 1) * QE], &want[r * QE..(r + 1) * QE]);
        }
    }
}

/// `kv_write_rows` puts exactly the chosen node rows at the chosen positions (the commit of an accepted path) and nothing else.
#[test]
fn kv_write_rows_commits_exactly_the_accepted_path() {
    if gpu().is_none() {
        return;
    }
    let lens = [700usize, 300];
    let mut w = make_world(seqs_of(&lens), 11, 8);
    ensure_pages(&mut w, 0, 700 + 16);
    ensure_pages(&mut w, 1, 300 + 16);
    let before_k = w.cache.k.to_vec::<u16>().unwrap();
    let (kn, vn) = node_kv(3, 20);
    let (kd, vd) = (DevBuf::from_slice(&kn).unwrap(), DevBuf::from_slice(&vn).unwrap());
    // sequence 0 commits window rows 0, 3, 7 at 700, 701, 702; sequence 1 rows 12, 13 at 300, 301
    let src = [0i32, 3, 7, 12, 13];
    let slot = [0i32, 0, 0, 1, 1];
    let pos = [700i32, 701, 702, 300, 301];
    let (sd, sl, ps) = (DevBuf::from_slice(&src).unwrap(), DevBuf::from_slice(&slot).unwrap(), DevBuf::from_slice(&pos).unwrap());
    kv_write_rows(&kd, &vd, &sd, &sl, &ps, &w.table, &mut w.cache, 5, &w.stream).unwrap();
    w.stream.sync().unwrap();
    let after_k = w.cache.k.to_vec::<u16>().unwrap();
    let after_v = w.cache.v.to_vec::<u16>().unwrap();
    let tables = w.table.buf.to_vec::<i32>().unwrap();
    let mut written = std::collections::HashSet::new();
    for i in 0..5 {
        let blk = tables[slot[i] as usize * w.table.max_blocks + pos[i] as usize / BLOCK_TOKENS] as usize;
        let at = (blk * BLOCK_TOKENS + pos[i] as usize % BLOCK_TOKENS) * TE;
        let r = src[i] as usize;
        assert_eq!(&after_k[at..at + TE], &kn[r * TE..(r + 1) * TE], "K of entry {i}");
        assert_eq!(&after_v[at..at + TE], &vn[r * TE..(r + 1) * TE], "V of entry {i}");
        written.insert(at);
    }
    for (i, (a, b)) in after_k.iter().zip(&before_k).enumerate() {
        if a != b {
            assert!(written.contains(&(i / TE * TE)), "K word {i} changed outside the committed rows");
        }
    }
}

// ---------------------------------------------------------------------------------------------------- golden

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Golden hashes of the kernels' outputs on seeded synthetic inputs (no pack needed), for this target's KV head count: the cache
/// after `kv_write`, decode windows of several widths and contexts, every prefill variant, and a tree window. A change that must
/// keep every bit diffs this file against one written by the old build.
///
///     REXL3_GOLDEN_OUT=golden_attention.txt cargo test --release -p gb-cuda --test attention golden_attention_hashes -- --ignored
///     REXL3_GOLDEN_OUT=golden_attention_kv2.txt cargo test --release -p gb-cuda --test attention_kv2 golden_attention_hashes -- --ignored
#[test]
#[ignore]
fn golden_attention_hashes() {
    use std::io::Write;
    let Some(_st) = gpu() else { return };
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    let w = make_world(seqs_of(&[1, 300, 1000, 5000]), 77, 3);
    let mut out = String::new();
    // the cache as kv_write left it, page by page in table order (the unused tail of a last page is NaN fill)
    let (kc, vc) = (w.cache.k.to_vec::<u16>().unwrap(), w.cache.v.to_vec::<u16>().unwrap());
    out += &format!("cache k {:016x} v {:016x}\n", fnv(&u16_bytes(&kc)), fnv(&u16_bytes(&vc)));
    for (name, rows) in [
        ("decode R1 seq0 p0", vec![(0usize, 0usize)]),
        ("decode R1 seq3 p4999", vec![(3, 4999)]),
        ("decode R4 seq2 p996..999", (996..1000).map(|p| (2, p)).collect()),
        ("decode R7 mixed", vec![(1, 299), (2, 511), (2, 512), (3, 513), (0, 0), (3, 4095), (1, 7)]),
        ("decode R64 seq3", (0..64).map(|i| (3, 2000 + 31 * i)).collect::<Vec<_>>()),
        ("decode R128 seq3", (0..128).map(|i| (3, 4872 + i)).collect()),
    ] {
        out += &format!("{name} {:016x}\n", fnv(&u16_bytes(&run_decode(&w, &rows))));
    }
    // every CTA shape this build has (at groups of 8 and 4 variants 2 and 3 do not exist: see the refusal test)
    for variant in (0..5).filter(|&v| SHAPE.prefill_variant_supported(v)) {
        let mut q = Vec::new();
        for p in 0..1000 {
            q.extend(qvec(w.seqs[2].sid, p));
        }
        let qd = DevBuf::from_slice(&q).unwrap();
        let mut o = DevBuf::zeroed(1000 * QE * 2).unwrap();
        attn_prefill_variant(&qd, &w.cache, &w.table, 2, 0, 1000, &mut o, variant, &w.stream).unwrap();
        w.stream.sync().unwrap();
        out += &format!("prefill seq2 1000 rows variant {variant} {:016x}\n", fnv(&u16_bytes(&o.to_vec::<u16>().unwrap())));
    }
    out += &format!(
        "prefill seq3 chunks 0+4096, 4096+904 {:016x}\n",
        fnv(&u16_bytes(&run_prefill(&w, 3, &[(0, 4096), (4096, 904)])))
    );
    let trees = vec![(2usize, Tree::new(&random_parents(9, 9, 3)).unwrap()), (3, Tree::new(&random_parents(17, 17, 4)).unwrap())];
    let run = run_tree_attention(&w, &trees, 5, None);
    out += &format!("tree windows 9 + 17 nodes {:016x}\n", fnv(&u16_bytes(&run.out)));
    std::fs::File::create(path).unwrap().write_all(out.as_bytes()).unwrap();
}
