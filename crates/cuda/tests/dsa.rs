//! The DSA indexer (`gb_cuda::dsa`): exactness of the selection against the ExLlamaV3 fork's `dsa_topk` and a host reference, scoring
//! and key-side numerics against the fork on real layer weights and against float64, row invariance (a row of a window equals the row
//! alone, across regimes, pool edges, slots and window sizes), rollback, and golden hashes. GPU tests skip without a device; the oracle
//! tests skip without `REXL3_DSA_REF_DIR` (dumps from tools/dsa_oracle.py), and fail instead when `REXL3_DSA_REQUIRE_REF=1`.
#![cfg(not(gb_no_cuda))]

mod common;

use common::{read_npy, Rng};
use gb_cuda::attention::{BlockTable, RowIndex};
use gb_cuda::dsa::{self, Planes, Scratch, D, H, K_OUT, K_STRIDE, MAX_ROWS, P, SEL_POOLS};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use std::path::PathBuf;

// ------------------------------------------------------------------------------------------------ plumbing

fn gpu() -> Option<Stream> {
    if !device_available() {
        eprintln!("no CUDA device: skipping");
        return None;
    }
    Some(Stream::new().unwrap())
}

/// The oracle dump directory (`tools/dsa_oracle.py`), or None (skip). With REXL3_DSA_REQUIRE_REF=1 a missing dump is a failure, so a gate
/// that has the dumps cannot pass vacuously.
fn ref_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_DSA_REF_DIR") {
        Ok(d) if PathBuf::from(&d).join("adv").is_dir() => Some(PathBuf::from(d)),
        _ => {
            assert!(
                std::env::var("REXL3_DSA_REQUIRE_REF").as_deref() != Ok("1"),
                "REXL3_DSA_REF_DIR is required (tools/dsa_oracle.py)"
            );
            eprintln!("REXL3_DSA_REF_DIR not set or empty: oracle comparison not run");
            None
        }
    }
}

fn h2f(u: u16) -> f64 {
    f16_to_f64(u)
}
fn f2h(x: f64) -> u16 {
    f64_to_f16(x)
}
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3))
}
fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// The ordered 16-bit key (-0 == +0); None for -inf and NaN, which are never selected.
fn key(mut u: u16) -> Option<u16> {
    if u == 0x8000 {
        u = 0;
    }
    if (u & 0x7fff) > 0x7c00 || u == 0xfc00 {
        return None;
    }
    Some(if u & 0x8000 != 0 { !u } else { u | 0x8000 })
}

/// Distance in representable fp16 values between two finite bit patterns (+-0 equal).
fn ulps(a: u16, b: u16) -> i64 {
    match (key(a), key(b)) {
        (Some(x), Some(y)) => (x as i64 - y as i64).abs(),
        (None, None) if a == b => 0,
        _ => i64::MAX,
    }
}

#[derive(Default, Debug)]
struct UlpStats {
    n: usize,
    exact: usize,
    max: i64,
}
impl UlpStats {
    fn add(&mut self, got: u16, want: u16) {
        let u = ulps(got, want);
        self.n += 1;
        self.exact += (u == 0) as usize;
        self.max = self.max.max(u);
    }
    fn frac_exact(&self) -> f64 {
        self.exact as f64 / self.n.max(1) as f64
    }
}

fn normal(r: &mut Rng) -> f32 {
    // Box-Muller from the shared uniform generator
    let (u1, u2) = (r.uniform(1e-7, 1.0), r.uniform(0.0, 1.0));
    (-2.0 * (u1 as f64).ln()).sqrt() as f32 * (std::f64::consts::TAU * u2 as f64).cos() as f32
}
fn rand_f16(r: &mut Rng, n: usize, mean: f32, sd: f32) -> Vec<u16> {
    (0..n).map(|_| f2h((mean + sd * normal(r)) as f64)).collect()
}

struct Weights {
    ln_w: Vec<u16>,
    ln_b: Vec<u16>,
    ape: Vec<f32>,
}
impl Weights {
    fn synthetic(seed: u64) -> Self {
        let mut r = Rng(seed);
        Self {
            ln_w: rand_f16(&mut r, D, 1.0, 0.2),
            ln_b: rand_f16(&mut r, D, 0.0, 0.1),
            ape: (0..P * D).map(|_| 0.5 * normal(&mut r)).collect(),
        }
    }
    fn device(&self) -> dsa::IndexerWeights {
        dsa::IndexerWeights::new(&self.ln_w, &self.ln_b, &self.ape, dsa::LN_EPS).unwrap()
    }
}

/// Planes + table + scratch for `slots` slots of `max_context` tokens, blocks shuffled across slots, every unwritten byte NaN (a read
/// of an unwritten pool or ring row turns its score NaN, which the selection never picks, so the tests see it).
struct Rig {
    st: Stream,
    w: dsa::IndexerWeights,
    planes: Planes,
    scratch: Scratch,
    table: BlockTable,
    host_table: Vec<Vec<u32>>,
}

impl Rig {
    fn new(slots: usize, max_context: usize, ring_len: usize, w: &Weights, seed: u64) -> Self {
        let st = Stream::new().unwrap();
        let bps = max_context.div_ceil(256);
        let n_blocks = slots * bps;
        let mut perm: Vec<u32> = (0..n_blocks as u32).collect();
        let mut r = Rng(seed);
        for i in (1..perm.len()).rev() {
            perm.swap(i, (r.next_u64() % (i as u64 + 1)) as usize);
        }
        let host_table: Vec<Vec<u32>> = (0..slots).map(|s| perm[s * bps..(s + 1) * bps].to_vec()).collect();
        let table = BlockTable::from_host(&host_table, bps).unwrap();
        let mut planes = Planes::new(n_blocks, slots, ring_len).unwrap();
        planes.pool.memset_async(0x7f, &st).unwrap();
        planes.ring.memset_async(0x7f, &st).unwrap();
        st.sync().unwrap();
        Self { st, w: w.device(), planes, scratch: Scratch::new(max_context).unwrap(), table, host_table }
    }
    fn pool_row(&self, slot: usize, p: usize) -> usize {
        (self.host_table[slot][p / 64] as usize * 64 + p % 64) * D
    }
    fn pool_plane(&self) -> Vec<u16> {
        self.planes.pool.to_vec::<u16>().unwrap()
    }
    fn pool_key(&self, plane: &[u16], slot: usize, p: usize) -> Vec<u16> {
        let o = self.pool_row(slot, p);
        plane[o..o + D].to_vec()
    }
    /// Uploads whole pool arrays of one slot at once (`pools[p]` is 128 values).
    fn load_pools(&mut self, slot: usize, pools: &[u16]) {
        let mut plane = self.pool_plane();
        for p in 0..pools.len() / D {
            let o = self.pool_row(slot, p);
            plane[o..o + D].copy_from_slice(&pools[p * D..(p + 1) * D]);
        }
        self.planes.pool.upload(&plane).unwrap();
    }
    /// The key-side update for rows of one slot at `pos0..`: `k_raw`, `gate` are `[n][128]` fp16 bits.
    fn append(&mut self, slot: usize, pos0: usize, k_raw: &[u16], gate: &[u16]) {
        let n = k_raw.len() / D;
        let (slots, poss): (Vec<i32>, Vec<i32>) = (0..n).map(|i| (slot as i32, (pos0 + i) as i32)).unzip();
        let rows = RowIndex::from_host(&slots, &poss).unwrap();
        let (k, g) = (DevBuf::from_slice(k_raw).unwrap(), DevBuf::from_slice(gate).unwrap());
        dsa::append(&self.w, &mut self.planes, &mut self.scratch, &k, &g, &rows, n, &self.table, &self.st).unwrap();
        self.st.sync().unwrap();
    }
    /// Scores and selects for rows `(slot, pos)`; returns `(scores [n][s_stride], indices [n][K_STRIDE], meta [n][2])`.
    fn select(&mut self, rows: &[(usize, usize)], q: &[u16], w: &[u16]) -> (Vec<u16>, Vec<i32>, Vec<i32>) {
        let n = rows.len();
        let (slots, poss): (Vec<i32>, Vec<i32>) = rows.iter().map(|&(s, p)| (s as i32, p as i32)).unzip();
        let ri = RowIndex::from_host(&slots, &poss).unwrap();
        let (qd, wd) = (DevBuf::from_slice(q).unwrap(), DevBuf::from_slice(w).unwrap());
        let n_pools_max = rows.iter().map(|&(_, p)| dsa::row_pools(p)).max().unwrap().max(1);
        // poison the outputs: a row the kernel does not write must not look like a result
        self.scratch.indices.memset_async(0xee, &self.st).unwrap();
        self.scratch.meta.memset_async(0xee, &self.st).unwrap();
        dsa::select(&self.planes, &mut self.scratch, &qd, &wd, &ri, n, n_pools_max, &self.table, &self.st).unwrap();
        self.st.sync().unwrap();
        let s_stride = self.scratch.s_stride;
        let scores = self.scratch.scores.to_vec::<u16>().unwrap()[..n * s_stride].to_vec();
        let idx = self.scratch.indices.to_vec::<i32>().unwrap()[..n * K_STRIDE].to_vec();
        let meta = self.scratch.meta.to_vec::<i32>().unwrap()[..n * 2].to_vec();
        (scores, idx, meta)
    }
    /// Selection from an arbitrary score matrix (the kernels' select stage alone).
    fn select_matrix(&mut self, scores: &[u16], s_stride: usize, poss: &[usize]) -> (Vec<i32>, Vec<i32>) {
        let n = poss.len();
        let slots: Vec<i32> = vec![0; n];
        let p: Vec<i32> = poss.iter().map(|&x| x as i32).collect();
        let ri = RowIndex::from_host(&slots, &p).unwrap();
        let sc = DevBuf::from_slice(scores).unwrap();
        let mut idx = DevBuf::zeroed(MAX_ROWS * K_STRIDE * 4).unwrap();
        let mut meta = DevBuf::zeroed(MAX_ROWS * 8).unwrap();
        idx.memset_async(0xee, &self.st).unwrap();
        meta.memset_async(0xee, &self.st).unwrap();
        dsa::select_scores(&sc, s_stride, &ri, n, &mut idx, &mut meta, &self.st).unwrap();
        self.st.sync().unwrap();
        (idx.to_vec::<i32>().unwrap()[..n * K_STRIDE].to_vec(), meta.to_vec::<i32>().unwrap()[..n * 2].to_vec())
    }
}

// ------------------------------------------------------------------------------------------------ host references

/// Float64 index-key LayerNorm of one row (fp16 weights), rounded once to fp16, and per element the magnitude of its terms
/// (`|(x - mean) * rstd * w| + |b|`: an element near zero from cancelling terms carries fp32 error of that scale, not of its own).
fn ref_layernorm(x: &[u16], w: &[u16], b: &[u16], eps: f64) -> (Vec<u16>, Vec<f64>) {
    let xs: Vec<f64> = x.iter().map(|&v| h2f(v)).collect();
    let mean = xs.iter().sum::<f64>() / D as f64;
    let var = xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / D as f64;
    let rstd = 1.0 / (var + eps).sqrt();
    let y: Vec<u16> = (0..D).map(|i| f2h((xs[i] - mean) * rstd * h2f(w[i]) + h2f(b[i]))).collect();
    let mag: Vec<f64> = (0..D).map(|i| ((xs[i] - mean) * rstd * h2f(w[i])).abs() + h2f(b[i]).abs()).collect();
    (y, mag)
}

/// Float64 pooled key from the four members' `[k | gate]` rows, and per dimension the magnitude of its terms.
fn ref_pool(members: &[&[u16]], ape: &[f32]) -> (Vec<u16>, Vec<f64>) {
    let mut out = Vec::new();
    let mut mag = Vec::new();
    for d in 0..D {
        let g: Vec<f64> = (0..P).map(|j| h2f(members[j][D + d]) + ape[j * D + d] as f64).collect();
        let m = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = g.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = e.iter().sum();
        out.push(f2h((0..P).map(|j| e[j] / s * h2f(members[j][d])).sum::<f64>()));
        mag.push((0..P).map(|j| e[j] / s * h2f(members[j][d]).abs()).sum::<f64>());
    }
    (out, mag)
}

/// Float64 score of one (row, pool): 2^-6 * sum_h relu(q_h . K) * w_h (not yet rounded), and the magnitude it is accumulated from,
/// 2^-6 * sum_h relu(q_h . K) * |w_h| (the scale of the fp32 rounding error: the weights have both signs, so a score can be far
/// smaller than its terms and a fixed ulp bound on the result would be meaningless there).
fn ref_score_l1(q: &[u16], w: &[u16], pool: &[u16]) -> (f64, f64) {
    let (mut s, mut l1) = (0.0, 0.0);
    for h in 0..H {
        let dot: f64 = (0..D).map(|d| h2f(q[h * D + d]) * h2f(pool[d])).sum();
        if dot > 0.0 {
            s += dot * h2f(w[h]);
            l1 += dot * h2f(w[h]).abs();
        }
    }
    (s * 0.015625, l1 * 0.015625)
}
/// Spacing of fp16 values at |x| (the ulp of the larger of two compared values).
fn spacing16(x: f64) -> f64 {
    let a = x.abs().max(2f64.powi(-14));
    2f64.powi(a.log2().floor() as i32 - 10)
}
/// Accepts `got` for `want` (both fp16 bits) when they differ by at most one fp16 spacing at their larger magnitude plus
/// `TOL` of the accumulation magnitude `l1` (fp32 sums of 128-term dots and a 32-head sum: the measured worst is far below TOL).
const TOL: f64 = 2e-6;
fn score_close(got: u16, want: u16, l1: f64) -> bool {
    let (a, b) = (h2f(got), h2f(want));
    if got == want || ulps(got, want) == 0 {
        return true;
    }
    (a - b).abs() <= spacing16(a.abs().max(b.abs())) + TOL * l1
}

/// Agreement of two fp16 arrays whose elements are fp32 sums of terms of magnitude `mag`: the share that is bit-identical, the worst
/// distance in fp16 values over elements whose sum does not cancel (|value| >= mag / 8, where an ulp bound means something), and the
/// worst error beyond one fp16 spacing as a fraction of `mag` (the fp32 accumulation error; bounded by TOL).
#[derive(Default, Debug)]
struct Closeness {
    n: usize,
    exact: usize,
    clean_max_ulp: i64,
    excess: f64,
}
impl Closeness {
    fn add(&mut self, got: u16, want: u16, mag: f64) {
        self.n += 1;
        let u = ulps(got, want);
        if u == 0 {
            self.exact += 1;
            return;
        }
        let (a, b) = (h2f(got), h2f(want));
        if b.abs() >= mag / 8.0 {
            self.clean_max_ulp = self.clean_max_ulp.max(u);
        }
        self.excess = self.excess.max(((a - b).abs() - spacing16(a.abs().max(b.abs()))).max(0.0) / mag.max(1e-30));
    }
    fn frac_exact(&self) -> f64 {
        self.exact as f64 / self.n.max(1) as f64
    }
}

/// The 512 best pools of `scores[..t]` (fp16 bits), ties to the lowest index, ascending.
fn ref_topk(scores: &[u16], t: usize, k: usize) -> Vec<usize> {
    let mut v: Vec<(u16, usize)> = (0..t).filter_map(|i| key(scores[i]).map(|kk| (kk, i))).collect();
    v.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    v.truncate(k);
    let mut out: Vec<usize> = v.into_iter().map(|(_, i)| i).collect();
    out.sort();
    out
}

/// The full expected selection row for a sparse row at `pos`: pools expanded to tokens, then the tail.
fn expect_row(pools: &[usize], pos: usize) -> Vec<i32> {
    let mut v: Vec<i32> = pools.iter().flat_map(|&p| (0..P).map(move |j| (p * P + j) as i32)).collect();
    let tail = dsa::row_tail(pos);
    v.extend((0..tail).map(|j| (pos + 1 - tail + j) as i32));
    v
}

/// The fork's selection row as a sorted list without the -1 padding. The fork emits the entries above the threshold in ascending order
/// and then the threshold ties in ascending order; REXL3 emits the same set in one ascending run (tail tokens, all larger than any
/// pooled token, stay last), which is the order the attention gather reads in.
fn fork_row(row: &[i32]) -> Vec<i32> {
    let mut v: Vec<i32> = row.iter().cloned().filter(|&x| x >= 0).collect();
    v.sort();
    v
}

fn check_row(idx: &[i32], meta: &[i32], r: usize, expect: &[i32], what: &str) {
    let row = &idx[r * K_STRIDE..(r + 1) * K_STRIDE];
    assert_eq!(meta[2 * r], 1, "{what}: row {r} not sparse");
    assert_eq!(meta[2 * r + 1] as usize, expect.len(), "{what}: row {r} count");
    assert_eq!(&row[..expect.len()], expect, "{what}: row {r} selection");
    assert!(row[expect.len()..].iter().all(|&x| x == -1), "{what}: row {r} padding");
}

/// A position whose row sees exactly `t` complete pools and a tail of `tail` raw tokens (the regime needs pos + 1 > 2048, so the
/// 512-pool rows always have a tail of at least one token).
fn pos_for(t: usize, row: usize) -> usize {
    let tail = if t > SEL_POOLS { row % P } else { 1 + row % (P - 1) };
    P * t + tail - 1
}

// ------------------------------------------------------------------------------------------------ oracle access

/// The layer's indexer tensors read from the GLM pack itself through `gb_exl3::Checkpoint` (only the shard headers are scanned and the
/// three small tensors read): `REXL3_GLM_PACK`, else None. `layer` 3 is the first trunk MLA layer, 45 the MTP layer.
fn pack_weights(layer: usize) -> Option<Weights> {
    let dir = std::env::var("REXL3_GLM_PACK").ok()?;
    let ck = gb_exl3::checkpoint::scan(std::path::Path::new(&dir), false).expect("scan the GLM pack");
    let get = |n: &str, dtype: &str| {
        let key = format!("model.language_model.layers.{layer}.self_attn.indexer.{n}");
        let e = ck.plain.get(&key).unwrap_or_else(|| panic!("{key} is not a plain tensor of the pack"));
        assert_eq!(e.dtype, dtype, "{key}");
        e
    };
    let ape_bytes = ck.read_tensor(get("index_kpool_compress_ape", "F32")).unwrap();
    Some(Weights {
        ln_w: ck.read_u16(get("k_norm.weight", "F16")).unwrap(),
        ln_b: ck.read_u16(get("k_norm.bias", "F16")).unwrap(),
        ape: ape_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
    })
}

struct Oracle {
    dir: PathBuf,
    ctx: usize,
    /// The weights the kernels run with: the pack's (through the Checkpoint) when `REXL3_GLM_PACK` is set, else the oracle's dump.
    w: Weights,
    /// The weights the fork itself loaded, as the oracle script dumped them.
    w_dump: Weights,
}
impl Oracle {
    fn open(root: &std::path::Path, layer: usize, ctx: usize) -> Option<Self> {
        let dir = root.join(format!("L{layer}_S{ctx}"));
        if !dir.join("manifest.json").exists() {
            assert!(
                std::env::var("REXL3_DSA_REQUIRE_REF").as_deref() != Ok("1"),
                "{}: the oracle dump is missing",
                dir.display()
            );
            eprintln!("{}: no oracle dump: oracle comparison not run", dir.display());
            return None;
        }
        let h16 = |n: &str| -> Vec<u16> {
            read_npy(&dir.join(n)).data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
        };
        let ape: Vec<f32> =
            read_npy(&dir.join("ape.npy")).data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let w_dump = Weights { ln_w: h16("ln_w.npy"), ln_b: h16("ln_b.npy"), ape };
        let w = pack_weights(layer).unwrap_or_else(|| Weights {
            ln_w: w_dump.ln_w.clone(),
            ln_b: w_dump.ln_b.clone(),
            ape: w_dump.ape.clone(),
        });
        Some(Self { w, w_dump, dir, ctx })
    }
    fn h16(&self, name: &str) -> Vec<u16> {
        read_npy(&self.dir.join(name)).data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }
    fn i32s(&self, name: &str) -> Vec<i32> {
        read_npy(&self.dir.join(name)).data.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }
    /// `(p0, R, T, selected)` of every case in the manifest (json.dump with indent=1: the keys of a case appear in this order).
    fn cases(&self) -> Vec<(usize, usize, usize, bool)> {
        let text = std::fs::read_to_string(self.dir.join("manifest.json")).unwrap();
        let num = |chunk: &str, key: &str| -> usize {
            let i = chunk.find(key).unwrap_or_else(|| panic!("manifest: no {key}")) + key.len();
            chunk[i..].trim_start().split(|c: char| !c.is_ascii_digit()).next().unwrap().parse().unwrap()
        };
        text.split("\"p0\":")
            .skip(1)
            .map(|c| {
                let c = format!("\"p0\":{c}");
                (num(&c, "\"p0\":"), num(&c, "\"R\":"), num(&c, "\"T\":"), c.contains("\"selected\": true"))
            })
            .collect()
    }
}

fn oracle_layers(root: &std::path::Path) -> Vec<Oracle> {
    [(3, 16384), (45, 8192), (3, 262144)].iter().filter_map(|&(l, s)| Oracle::open(root, l, s)).collect()
}

// ================================================================================================ tests

/// The tensors the kernels need, read from the pack through `gb_exl3::Checkpoint`, equal what the fork loaded (the oracle's dump) bit for bit:
/// the key-side and scoring oracle tests run the kernels on the pack's values when `REXL3_GLM_PACK` is set.
#[test]
fn pack_tensors_equal_the_oracles_weights() {
    let Some(root) = ref_dir() else { return };
    if std::env::var("REXL3_GLM_PACK").is_err() {
        assert!(std::env::var("REXL3_DSA_REQUIRE_REF").as_deref() != Ok("1"), "REXL3_GLM_PACK is required");
        eprintln!("REXL3_GLM_PACK not set: pack tensors not compared");
        return;
    }
    for (layer, ctx) in [(3usize, 16384usize), (45, 8192)] {
        let Some(o) = Oracle::open(&root, layer, ctx) else { continue };
        let p = pack_weights(layer).unwrap();
        assert_eq!(p.ln_w, o.w_dump.ln_w, "layer {layer}: k_norm.weight");
        assert_eq!(p.ln_b, o.w_dump.ln_b, "layer {layer}: k_norm.bias");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&p.ape), bits(&o.w_dump.ape), "layer {layer}: ape");
        assert_eq!((p.ln_w.len(), p.ln_b.len(), p.ape.len()), (D, D, P * D));
    }
}

/// The key side on real layer weights: LayerNorm against the fork's `idx_k_norm` and against float64; the pooled-key plane built through
/// the ring in 128-row calls against the fork's `_update_pool_plane` and against float64.
#[test]
fn key_side_matches_the_fork_and_float64() {
    let Some(_st) = gpu() else { return };
    let Some(root) = ref_dir() else { return };
    for o in oracle_layers(&root).iter() {
        let (k_raw, gate, k_norm, pool) = (o.h16("k_raw.npy"), o.h16("gate.npy"), o.h16("k_norm.npy"), o.h16("pool.npy"));
        let s = o.ctx;
        let mut rig = Rig::new(1, s, 256, &o.w, 5);
        let (mut ln_vs_fork, mut ln_vs_f64) = (Closeness::default(), Closeness::default());
        let mut kg_rows: Vec<u16> = Vec::new();
        let mut pos = 0;
        while pos < s {
            let n = (MAX_ROWS - (pos % 7)).min(s - pos); // call sizes 122..128: pools straddle calls at every offset
            rig.append(0, pos, &k_raw[pos * D..(pos + n) * D], &gate[pos * D..(pos + n) * D]);
            let kg = rig.scratch.kg.to_vec::<u16>().unwrap();
            for i in 0..n {
                let got = &kg[i * 2 * D..i * 2 * D + D];
                assert_eq!(
                    &kg[i * 2 * D + D..(i + 1) * 2 * D],
                    &gate[(pos + i) * D..(pos + i + 1) * D],
                    "gate must pass through"
                );
                let (r, mag) = ref_layernorm(&k_raw[(pos + i) * D..(pos + i + 1) * D], &o.w.ln_w, &o.w.ln_b, dsa::LN_EPS as f64);
                for d in 0..D {
                    ln_vs_fork.add(got[d], k_norm[(pos + i) * D + d], mag[d]);
                    ln_vs_f64.add(got[d], r[d], mag[d]);
                }
                kg_rows.extend_from_slice(&kg[i * 2 * D..(i + 1) * 2 * D]);
            }
            pos += n;
        }
        eprintln!(
            "S{s}: LayerNorm vs torch F.layer_norm: {:.5} bit-identical, worst {} ulp where the result does not cancel, excess {:.1e}; vs float64: {:.5}, worst {} ulp, excess {:.1e}",
            ln_vs_fork.frac_exact(),
            ln_vs_fork.clean_max_ulp,
            ln_vs_fork.excess,
            ln_vs_f64.frac_exact(),
            ln_vs_f64.clean_max_ulp,
            ln_vs_f64.excess
        );
        assert!(ln_vs_f64.clean_max_ulp <= 1 && ln_vs_f64.excess < TOL, "LayerNorm vs float64: {ln_vs_f64:?}");
        assert!(
            ln_vs_fork.clean_max_ulp <= 1 && ln_vs_fork.excess < TOL,
            "LayerNorm vs the fork (torch F.layer_norm): {ln_vs_fork:?}"
        );
        assert!(ln_vs_fork.frac_exact() > 0.99, "LayerNorm bit-identical fraction {}", ln_vs_fork.frac_exact());

        // the pool plane my pipeline built (from my LayerNorm rows) against the fork's, and against float64 on the same members
        let plane = rig.pool_plane();
        let (mut vs_fork, mut vs_f64) = (Closeness::default(), Closeness::default());
        for p in 0..s / P {
            let got = rig.pool_key(&plane, 0, p);
            let members: Vec<&[u16]> = (0..P).map(|j| &kg_rows[(P * p + j) * 2 * D..(P * p + j + 1) * 2 * D]).collect();
            let (r, mag) = ref_pool(&members, &o.w.ape);
            for d in 0..D {
                vs_fork.add(got[d], pool[p * D + d], mag[d]);
                vs_f64.add(got[d], r[d], mag[d]);
            }
        }
        eprintln!(
            "S{s}: pool plane vs fork: {:.5} bit-identical, worst {} ulp where the sum does not cancel, excess {:.1e}; vs float64: {:.5}, worst {} ulp, excess {:.1e}",
            vs_fork.frac_exact(), vs_fork.clean_max_ulp, vs_fork.excess, vs_f64.frac_exact(), vs_f64.clean_max_ulp, vs_f64.excess
        );
        // the fork's pool is computed from ITS LayerNorm rows, mine from mine (up to 1 ulp apart): a pooled key may differ by about one ulp
        // of the key magnitude (bounded here by 1e-3, two fp16 spacings of the terms), more than that in ulps of a small pooled value
        assert!(vs_f64.clean_max_ulp <= 1 && vs_f64.excess < TOL, "pooled keys vs float64: {vs_f64:?}");
        assert!(vs_fork.frac_exact() > 0.99 && vs_fork.excess < 1e-3, "pooled keys vs the fork: {vs_fork:?}");

        // the pool kernel alone on the fork's own rows ([k_norm | gate]) must match the fork's pool bits closely
        let mut rig2 = Rig::new(1, s, 256, &o.w, 6);
        let mut st2 = Closeness::default();
        let mut pos = 0;
        while pos < s {
            let n = MAX_ROWS.min(s - pos);
            let mut kg = vec![0u16; MAX_ROWS * 2 * D];
            for i in 0..n {
                kg[i * 2 * D..i * 2 * D + D].copy_from_slice(&k_norm[(pos + i) * D..(pos + i + 1) * D]);
                kg[i * 2 * D + D..(i + 1) * 2 * D].copy_from_slice(&gate[(pos + i) * D..(pos + i + 1) * D]);
            }
            rig2.scratch.kg.upload(&kg).unwrap();
            let (slots, poss): (Vec<i32>, Vec<i32>) = (0..n).map(|i| (0, (pos + i) as i32)).unzip();
            let rows = RowIndex::from_host(&slots, &poss).unwrap();
            dsa::pool_update(&rig2.w, &mut rig2.planes, &rig2.scratch, &rows, n, &rig2.table, &rig2.st).unwrap();
            rig2.st.sync().unwrap();
            pos += n;
        }
        let plane2 = rig2.pool_plane();
        for p in 0..s / P {
            let got = rig2.pool_key(&plane2, 0, p);
            let members: Vec<Vec<u16>> = (0..P)
                .map(|j| {
                    let t = P * p + j;
                    [&k_norm[t * D..(t + 1) * D], &gate[t * D..(t + 1) * D]].concat()
                })
                .collect();
            let (_, mag) = ref_pool(&members.iter().map(|m| &m[..]).collect::<Vec<_>>(), &o.w.ape);
            for d in 0..D {
                st2.add(got[d], pool[p * D + d], mag[d]);
            }
        }
        eprintln!("S{s}: pool kernel on the fork's rows vs fork: {:.6} bit-identical, worst {} ulp where the sum does not cancel, excess {:.1e}", st2.frac_exact(), st2.clean_max_ulp, st2.excess);
        assert!(
            st2.clean_max_ulp <= 1 && st2.frac_exact() > 0.999 && st2.excess < TOL,
            "pool kernel vs fork on identical rows: {st2:?}"
        );
    }
}

/// Scoring on the fork's pool plane and the fork's queries: fp16 scores against the fork's scores and against float64.
#[test]
fn scores_match_the_fork_and_float64() {
    let Some(_st) = gpu() else { return };
    let Some(root) = ref_dir() else { return };
    for o in oracle_layers(&root) {
        let pool = o.h16("pool.npy");
        let mut rig = Rig::new(1, o.ctx, 256, &o.w, 9);
        rig.load_pools(0, &pool);
        // exact: bit-identical; the rest must be within one fp16 spacing plus TOL of the accumulation magnitude (normwise)
        let (mut vs_fork, mut vs_f64) = (Closeness::default(), Closeness::default());
        let mut n_cases = 0;
        for (p0, rr, t, _) in o.cases() {
            let tag = format!("c{p0}_{rr}");
            let (q, w, sc) =
                (o.h16(&format!("{tag}_q.npy")), o.h16(&format!("{tag}_w.npy")), o.h16(&format!("{tag}_scores.npy")));
            let rows: Vec<(usize, usize)> = (0..rr).map(|r| (0, p0 + r)).collect();
            let (got, _, meta) = rig.select(&rows, &q, &w);
            let st = rig.scratch.s_stride;
            for r in 0..rr {
                let pos = p0 + r;
                let bound = dsa::row_pools(pos);
                if !dsa::is_sparse(pos) {
                    assert_eq!(meta[2 * r], 0, "dense row {pos} must be reported dense");
                    continue; // the kernel scores sparse rows only
                }
                let (qr, wr) = (&q[r * H * D..(r + 1) * H * D], &w[r * H..(r + 1) * H]);
                for p in 0..bound.min(t) {
                    let (g, f) = (got[r * st + p], sc[r * t + p]);
                    let fork_exact = ulps(g, f) == 0;
                    let sampled = (p * 7 + r) % 11 == 0 || p + 4 >= bound;
                    if fork_exact && !sampled {
                        vs_fork.n += 1;
                        vs_fork.exact += 1;
                        continue;
                    }
                    let (refv, l1) = ref_score_l1(qr, wr, &pool[p * D..(p + 1) * D]);
                    vs_fork.add(g, f, l1);
                    assert!(
                        score_close(g, f, l1),
                        "{tag} row {r} pool {p}: mine {} fork {} float64 {refv} (l1 {l1})",
                        h2f(g),
                        h2f(f)
                    );
                    if sampled {
                        vs_f64.add(g, f2h(refv), l1);
                        assert!(
                            score_close(g, f2h(refv), l1),
                            "{tag} row {r} pool {p}: mine {} float64 {refv} (l1 {l1})",
                            h2f(g)
                        );
                    }
                }
            }
            n_cases += 1;
        }
        eprintln!(
            "S{} ({n_cases} cases): scores vs fork: {:.5} bit-identical of {}, worst {} ulp where the sum does not cancel, error beyond one spacing {:.1e} of the term magnitude; vs float64: {:.5} bit-identical of {}, worst {} ulp, excess {:.1e}",
            o.ctx, vs_fork.frac_exact(), vs_fork.n, vs_fork.clean_max_ulp, vs_fork.excess, vs_f64.frac_exact(), vs_f64.n, vs_f64.clean_max_ulp, vs_f64.excess
        );
        assert!(
            vs_fork.frac_exact() > 0.99 && vs_fork.clean_max_ulp <= 2 && vs_fork.excess < TOL,
            "scores vs the fork: {vs_fork:?}"
        );
        assert!(vs_f64.clean_max_ulp <= 1 && vs_f64.excess < TOL, "scores vs float64: {vs_f64:?}");
    }
}

/// The selection stage on the fork's own fp16 score matrices must reproduce the fork's complete selection (pools, tokens, tail) exactly;
/// the whole pipeline (my scores, my selection) must pick the same set of pools in every case, and where it differs the difference is
/// counted and attributed.
#[test]
fn selection_equals_the_forks_on_real_scores() {
    let Some(_st) = gpu() else { return };
    let Some(root) = ref_dir() else { return };
    for o in oracle_layers(&root) {
        let pool = o.h16("pool.npy");
        let mut rig = Rig::new(1, o.ctx, 256, &o.w, 11);
        rig.load_pools(0, &pool);
        let (mut n_rows, mut n_set_equal) = (0, 0);
        let mut diffs = Vec::new();
        for (p0, rr, t, selected) in o.cases() {
            if !selected {
                continue;
            }
            let tag = format!("c{p0}_{rr}");
            let sc = o.h16(&format!("{tag}_scores.npy"));
            let fork_sel = o.i32s(&format!("{tag}_sel.npy"));
            let kpad = fork_sel.len() / rr;
            // (1) identical scores in, identical selection out
            let st = (t.div_ceil(8) * 8).max(8);
            let mut mat = vec![0xfc00u16; rr * st];
            for r in 0..rr {
                mat[r * st..r * st + t].copy_from_slice(&sc[r * t..(r + 1) * t]);
            }
            let poss: Vec<usize> = (0..rr).map(|r| p0 + r).collect();
            let (idx, meta) = rig.select_matrix(&mat, st, &poss);
            for r in 0..rr {
                let want = fork_row(&fork_sel[r * kpad..r * kpad + K_OUT]);
                check_row(&idx, &meta, r, &want, &format!("{tag} fork scores"));
            }
            // (2) the whole pipeline
            let (q, w) = (o.h16(&format!("{tag}_q.npy")), o.h16(&format!("{tag}_w.npy")));
            let rows: Vec<(usize, usize)> = (0..rr).map(|r| (0, p0 + r)).collect();
            let (_, idx2, meta2) = rig.select(&rows, &q, &w);
            for r in 0..rr {
                n_rows += 1;
                let want = fork_row(&fork_sel[r * kpad..r * kpad + K_OUT]);
                let got = &idx2[r * K_STRIDE..r * K_STRIDE + meta2[2 * r + 1] as usize];
                if got == &want[..] {
                    n_set_equal += 1;
                } else {
                    let (a, b): (std::collections::BTreeSet<_>, std::collections::BTreeSet<_>) =
                        (got.iter().collect(), want.iter().collect());
                    diffs.push(format!(
                        "{tag} row {r}: only mine {:?}, only fork's {:?}",
                        a.difference(&b).take(8).collect::<Vec<_>>(),
                        b.difference(&a).take(8).collect::<Vec<_>>()
                    ));
                }
            }
        }
        eprintln!(
            "S{}: selection on fork scores identical in every case; full pipeline identical in {n_set_equal}/{n_rows} rows",
            o.ctx
        );
        for d in &diffs {
            eprintln!("  {d}");
        }
        assert_eq!(n_set_equal, n_rows, "the full pipeline's selection differs from the fork's: {diffs:?}");
    }
}

/// Adversarial score matrices (ties at the threshold and across a radix byte, exactly k valid, fp16 extremes) through the fork's
/// `dsa_topk` (dumped) and a host reference: the selections must be identical.
#[test]
fn selection_is_exact_on_adversarial_rows_against_the_fork() {
    let Some(_st) = gpu() else { return };
    let Some(root) = ref_dir() else { return };
    let adv = root.join("adv");
    let mut names: Vec<String> = std::fs::read_dir(&adv)
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().to_str()?.strip_suffix("_scores.npy").map(String::from))
        .collect();
    names.sort();
    assert!(names.len() >= 15, "adversarial dumps missing");
    let w = Weights::synthetic(1);
    let mut rig = Rig::new(1, 4096, 256, &w, 3);
    for name in names {
        let sc_npy = read_npy(&adv.join(format!("{name}_scores.npy")));
        let (rr, t) = (sc_npy.shape[0], sc_npy.shape[1]);
        let sc: Vec<u16> = sc_npy.data.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let fork: Vec<i32> = read_npy(&adv.join(format!("{name}_idx.npy")))
            .data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let st = t.div_ceil(8) * 8;
        let mut mat = vec![0xfc00u16; rr * st];
        for r in 0..rr {
            mat[r * st..r * st + t].copy_from_slice(&sc[r * t..(r + 1) * t]);
        }
        // positions: T pools and a tail of 0..3 tokens (every tail length is exercised)
        let poss: Vec<usize> = (0..rr).map(|r| pos_for(t, r)).collect();
        let (idx, meta) = rig.select_matrix(&mat, st, &poss);
        for r in 0..rr {
            let pos = poss[r];
            assert_eq!(dsa::row_pools(pos), t, "{name}: test positions");
            let host = ref_topk(&sc[r * t..(r + 1) * t], t, SEL_POOLS);
            let mut fork_pools: Vec<usize> =
                fork[r * SEL_POOLS..(r + 1) * SEL_POOLS].iter().filter(|&&x| x >= 0).map(|&x| x as usize).collect();
            fork_pools.sort(); // the fork emits the entries above the threshold ascending and THEN the ties ascending; the SET is the contract
            assert_eq!(
                host, fork_pools,
                "{name} row {r}: the host reference disagrees with the fork (a test bug or a fork finding)"
            );
            check_row(&idx, &meta, r, &expect_row(&host, pos), &name);
        }
    }
}

/// Selection edge cases only a host reference can judge: -0 and +0 tie, NaN and -inf are never selected, rows with fewer than 512 valid
/// scores, every length around tile edges, random rows with heavy ties.
#[test]
fn selection_is_exact_on_host_generated_rows() {
    let Some(_st) = gpu() else { return };
    let w = Weights::synthetic(1);
    let mut rig = Rig::new(1, 4096, 256, &w, 3);
    let mut r = Rng(0xD5A);
    let mut total = 0;
    for &t in &[
        512usize, 513, 600, 1023, 1024, 2047, 4096, 8191, 8192, 8193, 12345, 16384, 16385, 20000, 65535, 65536, 65537, 70000,
        131077, 262144,
    ] {
        let st = t.div_ceil(8) * 8;
        let rows = 4;
        let mut mat = vec![0xfc00u16; rows * st];
        for row in 0..rows {
            for i in 0..t {
                let v = match row {
                    0 => f2h((r.next_u64() % 24) as f64 / 4.0 - 3.0), // heavy ties incl. 0 and negatives
                    1 => match r.next_u64() % 5 {
                        0 => 0x8000, // -0
                        1 => 0x0000, // +0
                        2 => 0x7e00, // NaN
                        3 => 0xfc00, // -inf
                        _ => f2h((r.next_u64() % 3) as f64),
                    },
                    2 => f2h(normal(&mut r) as f64 * 9.0),
                    _ => f2h(1.0), // all equal
                };
                mat[row * st + i] = v;
            }
        }
        let poss: Vec<usize> = (0..rows).map(|row| pos_for(t, row)).collect();
        let (idx, meta) = rig.select_matrix(&mat, st, &poss);
        for row in 0..rows {
            let valid = (0..t).filter(|&i| key(mat[row * st + i]).is_some()).count();
            let host = ref_topk(&mat[row * st..], t, SEL_POOLS);
            assert_eq!(host.len(), SEL_POOLS.min(valid));
            // a row with fewer than 512 valid scores selects them all, and the tail follows them directly
            check_row(&idx, &meta, row, &expect_row(&host, poss[row]), &format!("T={t} kind {row}"));
            total += 1;
        }
    }
    eprintln!("{total} host-generated rows exact");
}

/// Row invariance: a window's row `r` equals the same row alone, for every window size, at positions around the dense/sparse switch and
/// around pool edges, with several slots sharing one call and rows reordered across slots.
#[test]
fn row_r_of_a_window_equals_row_r_alone() {
    let Some(_st) = gpu() else { return };
    let w = Weights::synthetic(77);
    let slots = 3;
    let lens = [2300usize, 2500, 6001];
    let mut rig = Rig::new(slots, 8192, 256, &w, 13);
    let mut rng = Rng(31);
    // build each slot's planes through the real key-side pipeline
    for (s, &len) in lens.iter().enumerate() {
        let (k, g) = (rand_f16(&mut rng, len * D, 0.0, 3.0), rand_f16(&mut rng, len * D, 0.0, 1.5));
        let mut pos = 0;
        while pos < len {
            let n = 100.min(len - pos);
            rig.append(s, pos, &k[pos * D..(pos + n) * D], &g[pos * D..(pos + n) * D]);
            pos += n;
        }
    }
    // per-position query material, a pure function of (slot, pos) so the same row has the same inputs in every call shape
    let qrow = |s: usize, p: usize| -> (Vec<u16>, Vec<u16>) {
        let mut r = Rng(0x51 ^ (s as u64 * 1_000_003) ^ (p as u64).wrapping_mul(0x9E37_79B9));
        (rand_f16(&mut r, H * D, 0.0, 1.0), rand_f16(&mut r, H, 0.3, 1.0))
    };
    let run = |rig: &mut Rig, rows: &[(usize, usize)]| {
        let (mut q, mut w) = (Vec::new(), Vec::new());
        for &(s, p) in rows {
            let (a, b) = qrow(s, p);
            q.extend(a);
            w.extend(b);
        }
        rig.select(rows, &q, &w)
    };
    let mut checked = 0;
    let mut compare = |rig: &mut Rig, rows: &[(usize, usize)], what: &str| {
        let (sc, idx, meta) = run(rig, rows);
        let st = rig.scratch.s_stride;
        for (r, &row) in rows.iter().enumerate() {
            let (sc1, idx1, meta1) = run(rig, &[row]);
            assert_eq!(&meta[2 * r..2 * r + 2], &meta1[0..2], "{what}: row {r} {row:?} meta");
            if dsa::is_sparse(row.1) {
                let t = dsa::row_pools(row.1);
                assert_eq!(&sc[r * st..r * st + t], &sc1[..t], "{what}: row {r} {row:?} scores differ between window and solo");
                assert_eq!(&idx[r * K_STRIDE..(r + 1) * K_STRIDE], &idx1[..K_STRIDE], "{what}: row {r} {row:?} selection");
            } else {
                assert_eq!(meta1[0], 0);
            }
            checked += 1;
        }
    };
    // single-slot windows of every size, starting around the regime switch and at every pool phase
    for &rr in &[1usize, 2, 3, 4, 5, 8, 9, 16, 17, 64, 128] {
        for &p0 in &[2030usize, 2037, 2038, 2039, 2040, 2041, 2044, 2045, 2046, 2047, 2048, 2049, 2150] {
            let rows: Vec<(usize, usize)> = (0..rr).map(|i| (0, p0 + i)).collect();
            compare(&mut rig, &rows, &format!("slot 0 window R={rr} p0={p0}"));
        }
    }
    // windows deep in the sparse regime and across pool pages (page = 64 pools = 256 tokens)
    for &(s, p0, rr) in &[(2usize, 5000usize, 9usize), (2, 5115, 9), (2, 255, 8), (1, 2000, 70), (2, 3071, 6), (2, 5990, 11)] {
        let rows: Vec<(usize, usize)> = (0..rr).map(|i| (s, p0 + i)).collect();
        compare(&mut rig, &rows, &format!("slot {s} window R={rr} p0={p0}"));
    }
    // several slots in one call, in different phases, windows consecutive per slot
    let mut rows = Vec::new();
    rows.extend((0..3).map(|i| (0usize, 2046 + i)));
    rows.extend((0..5).map(|i| (2usize, 4001 + i)));
    rows.extend((0..2).map(|i| (1usize, 2499 + i)));
    rows.extend((0..4).map(|i| (2usize, 6000 - 3 + i)));
    compare(&mut rig, &rows, "mixed slots");
    // the same rows with the slot groups in another order and another window size around them
    let mut rows2 = Vec::new();
    rows2.extend((0..2).map(|i| (1usize, 2499 + i)));
    rows2.extend((0..4).map(|i| (2usize, 6000 - 3 + i)));
    rows2.extend((0..3).map(|i| (0usize, 2046 + i)));
    let (sc_a, idx_a, _) = run(&mut rig, &rows);
    let (sc_b, idx_b, _) = run(&mut rig, &rows2);
    let st = rig.scratch.s_stride;
    // row (0, 2048) is row 2 of slot 0's group in `rows` (index 2) and row 6 in `rows2` (index 8)
    let (ia, ib) = (2usize, 8usize);
    assert_eq!(&sc_a[ia * st..ia * st + 512], &sc_b[ib * st..ib * st + 512]);
    assert_eq!(&idx_a[ia * K_STRIDE..(ia + 1) * K_STRIDE], &idx_b[ib * K_STRIDE..(ib + 1) * K_STRIDE]);
    eprintln!("{checked} rows of windows equal the same rows alone");
}

/// The regime is decided per row: in a window straddling 2,048 the early rows are dense (untouched, (0, 0)) and the later rows sparse
/// exactly as if decoded alone (count 2048 + the row's own tail).
#[test]
fn regime_is_decided_per_row() {
    let Some(_st) = gpu() else { return };
    let w = Weights::synthetic(5);
    let mut rig = Rig::new(1, 4096, 256, &w, 17);
    let mut rng = Rng(8);
    let n = 2200;
    let (k, g) = (rand_f16(&mut rng, n * D, 0.0, 3.0), rand_f16(&mut rng, n * D, 0.0, 1.5));
    for c in 0..n.div_ceil(100) {
        let (lo, hi) = (c * 100, ((c + 1) * 100).min(n));
        rig.append(0, lo, &k[lo * D..hi * D], &g[lo * D..hi * D]);
    }
    let rows: Vec<(usize, usize)> = (0..9).map(|i| (0, 2043 + i)).collect(); // positions 2043..=2051
    let q = rand_f16(&mut rng, 9 * H * D, 0.0, 1.0);
    let wt = rand_f16(&mut rng, 9 * H, 0.3, 1.0);
    let (_, idx, meta) = rig.select(&rows, &q, &wt);
    for (r, &(_, pos)) in rows.iter().enumerate() {
        if pos < 2048 {
            assert_eq!(&meta[2 * r..2 * r + 2], &[0, 0], "pos {pos} is dense");
            assert!(
                idx[r * K_STRIDE..(r + 1) * K_STRIDE].iter().all(|&x| x == 0xeeeeeeeeu32 as i32),
                "a dense row's selection row must be untouched"
            );
        } else {
            assert_eq!(meta[2 * r], 1, "pos {pos} is sparse");
            assert_eq!(meta[2 * r + 1] as usize, 2048 + (pos + 1) % 4, "pos {pos} count");
            let (_, idx1, meta1) = rig.select(&[(0, pos)], &q[r * H * D..(r + 1) * H * D], &wt[r * H..(r + 1) * H]);
            assert_eq!(&idx[r * K_STRIDE..(r + 1) * K_STRIDE], &idx1[..K_STRIDE], "pos {pos}: window row == solo");
            assert_eq!(&meta[2 * r..2 * r + 2], &meta1[..2]);
        }
    }
    // the first sparse rows (512 pools visible) select every pool: 2048 tokens plus the tail, i.e. all tokens 0..=pos
    let r = rows.iter().position(|&(_, p)| p == 2048).unwrap();
    let all: Vec<i32> = (0..=2048).collect();
    assert_eq!(&idx[r * K_STRIDE..r * K_STRIDE + 2049], &all[..]);
}

/// A padding row (position -1, what a CUDA-graph bucket pads a call with) is inert: it is not scored or selected, it does not touch the
/// ring or the pool plane, and the real rows around it get exactly the bits they get without it.
#[test]
fn padding_rows_are_inert() {
    let Some(_st) = gpu() else { return };
    let w = Weights::synthetic(61);
    let mut rng = Rng(62);
    let n = 2300;
    let (k, g) = (rand_f16(&mut rng, n * D, 0.0, 3.0), rand_f16(&mut rng, n * D, 0.0, 1.5));
    let mk = || Rig::new(1, 4096, 256, &w, 63);
    let (mut a, mut b) = (mk(), mk());
    for rig in [&mut a, &mut b] {
        for c in 0..(n - 8).div_ceil(100) {
            let (lo, hi) = (c * 100, ((c + 1) * 100).min(n - 8));
            rig.append(0, lo, &k[lo * D..hi * D], &g[lo * D..hi * D]);
        }
    }
    // the last 8 tokens as a window of real rows, once alone (rig a) and once interleaved with padding rows carrying junk (rig b)
    let (kw, gw) = (&k[(n - 8) * D..], &g[(n - 8) * D..]);
    a.append(0, n - 8, kw, gw);
    let junk_k = rand_f16(&mut rng, 3 * D, 50.0, 9.0);
    let junk_g = rand_f16(&mut rng, 3 * D, 50.0, 9.0);
    let layout: Vec<Option<usize>> =
        vec![None, Some(0), Some(1), Some(2), None, Some(3), Some(4), Some(5), Some(6), Some(7), None]; // None = padding
    let (mut kk, mut gg, mut slots, mut poss) = (vec![], vec![], vec![], vec![]);
    for (i, e) in layout.iter().enumerate() {
        match e {
            Some(j) => {
                kk.extend_from_slice(&kw[j * D..(j + 1) * D]);
                gg.extend_from_slice(&gw[j * D..(j + 1) * D]);
                slots.push(0);
                poss.push((n - 8 + j) as i32);
            }
            None => {
                kk.extend_from_slice(&junk_k[(i % 3) * D..(i % 3 + 1) * D]);
                gg.extend_from_slice(&junk_g[(i % 3) * D..(i % 3 + 1) * D]);
                slots.push(0);
                poss.push(-1);
            }
        }
    }
    let ri = RowIndex::from_host(&slots, &poss).unwrap();
    let (kd, gd) = (DevBuf::from_slice(&kk).unwrap(), DevBuf::from_slice(&gg).unwrap());
    dsa::append(&b.w, &mut b.planes, &mut b.scratch, &kd, &gd, &ri, layout.len(), &b.table, &b.st).unwrap();
    b.st.sync().unwrap();
    let ring_a = a.planes.ring.to_vec::<u16>().unwrap();
    let ring_b = b.planes.ring.to_vec::<u16>().unwrap();
    assert!(ring_a.iter().zip(&ring_b).all(|(x, y)| x == y), "padding rows changed the ring");
    let pool_a = a.pool_plane();
    let pool_b = b.pool_plane();
    for p in 0..n / P {
        assert_eq!(a.pool_key(&pool_a, 0, p), b.pool_key(&pool_b, 0, p), "padding rows changed pool {p}");
    }
    // selection: the real rows of the padded call equal the unpadded call's rows; the padding rows report dense and write nothing
    let q = rand_f16(&mut rng, layout.len() * H * D, 0.0, 1.0);
    let wt = rand_f16(&mut rng, layout.len() * H, 0.3, 1.0);
    let ri_sel = RowIndex::from_host(&slots, &poss).unwrap();
    let (qd, wd) = (DevBuf::from_slice(&q).unwrap(), DevBuf::from_slice(&wt).unwrap());
    b.scratch.indices.memset_async(0xee, &b.st).unwrap();
    dsa::select(&b.planes, &mut b.scratch, &qd, &wd, &ri_sel, layout.len(), n / P, &b.table, &b.st).unwrap();
    b.st.sync().unwrap();
    let idx = b.scratch.indices.to_vec::<i32>().unwrap();
    let meta = b.scratch.meta.to_vec::<i32>().unwrap();
    for (i, e) in layout.iter().enumerate() {
        match e {
            None => {
                assert_eq!(&meta[2 * i..2 * i + 2], &[0, 0], "padding row {i}");
                assert!(
                    idx[i * K_STRIDE..(i + 1) * K_STRIDE].iter().all(|&x| x == 0xeeeeeeeeu32 as i32),
                    "padding row {i} wrote a selection"
                );
            }
            Some(j) => {
                let pos = n - 8 + j;
                let (_, idx1, meta1) = a.select(&[(0, pos)], &q[i * H * D..(i + 1) * H * D], &wt[i * H..(i + 1) * H]);
                assert_eq!(&meta[2 * i..2 * i + 2], &meta1[..2], "real row {i}");
                assert_eq!(&idx[i * K_STRIDE..(i + 1) * K_STRIDE], &idx1[..K_STRIDE], "real row {i} next to padding rows");
            }
        }
    }
}

/// Appends, partial acceptance, rollback and re-appends of different rows leave the planes exactly as feeding the surviving rows one at a
/// time would: the ring and the always-recompute pool rule. Run with the minimum ring and with a ring as long as the context (the oracle's
/// full per-token plane).
#[test]
fn rollback_and_windows_leave_the_planes_as_serial_does() {
    let Some(_st) = gpu() else { return };
    let w = Weights::synthetic(21);
    for &ring_len in &[dsa::RING_MIN, 256usize, 1024] {
        let mut rig = Rig::new(2, 1024, ring_len, &w, 19);
        let mut serial = Rig::new(2, 1024, 256, &w, 23);
        let mut rng = Rng(99);
        // logical token streams per slot, with the rows that were ever appended
        let mut streams: Vec<(Vec<u16>, Vec<u16>)> = vec![(Vec::new(), Vec::new()); 2];
        let mut accepted_total = 0;
        for round in 0..700 {
            let s = (rng.next_u64() % 2) as usize;
            let len = streams[s].0.len() / D;
            if len + 20 >= 1024 {
                continue;
            }
            let n = 1 + (rng.next_u64() % 9) as usize; // a verify window of 1..9 rows
            let (k, g) = (rand_f16(&mut rng, n * D, 0.0, 3.0), rand_f16(&mut rng, n * D, 0.0, 1.5));
            rig.append(s, len, &k, &g);
            // accept a random prefix of the window (the rest is rolled back by simply not advancing the length)
            let acc = (rng.next_u64() % (n as u64 + 1)) as usize;
            let acc = if round % 7 == 0 { n } else { acc };
            streams[s].0.extend_from_slice(&k[..acc * D]);
            streams[s].1.extend_from_slice(&g[..acc * D]);
            accepted_total += acc;
        }
        for (s, st) in streams.iter().enumerate() {
            let len = st.0.len() / D;
            for pos in 0..len {
                serial.append(s, pos, &st.0[pos * D..(pos + 1) * D], &st.1[pos * D..(pos + 1) * D]);
            }
        }
        let (pa, pb) = (rig.pool_plane(), serial.pool_plane());
        let mut checked = 0;
        for (s, st) in streams.iter().enumerate() {
            for p in 0..st.0.len() / D / P {
                assert_eq!(
                    rig.pool_key(&pa, s, p),
                    serial.pool_key(&pb, s, p),
                    "ring_len {ring_len}: slot {s} pool {p} differs from the serial plane"
                );
                checked += 1;
            }
        }
        eprintln!(
            "ring_len {ring_len}: {checked} pools equal the serial plane after {accepted_total} accepted rows with rollbacks"
        );
        assert!(checked > 150, "only {checked} pools compared");
    }
}

/// 262,144 tokens of context (65,536 pools): scores against float64 on a sample, the selection against the host reference on the GPU's own
/// scores, for 1-row and 4-row windows, with a page table shuffled across 1,024 blocks.
#[test]
fn full_context_262144() {
    let Some(_st) = gpu() else { return };
    let w = Weights::synthetic(2);
    let ctx = 262_144;
    let mut rig = Rig::new(1, ctx, 256, &w, 29);
    let mut rng = Rng(4);
    let n_pools = ctx / P;
    // pool keys with structure: a mix of scales so that fp16 score ties occur at the threshold
    let pools = rand_f16(&mut rng, n_pools * D, 0.0, 1.0);
    rig.load_pools(0, &pools);
    for &(p0, rr) in &[(ctx - 1, 1usize), (ctx - 4, 4), (200_001, 3), (131_071, 1)] {
        let q = rand_f16(&mut rng, rr * H * D, 0.0, 1.0);
        let wt = rand_f16(&mut rng, rr * H, 0.5, 0.5);
        let rows: Vec<(usize, usize)> = (0..rr).map(|r| (0, p0 + r)).collect();
        let (sc, idx, meta) = rig.select(&rows, &q, &wt);
        let st = rig.scratch.s_stride;
        let mut worst = UlpStats::default();
        for r in 0..rr {
            let pos = p0 + r;
            let t = dsa::row_pools(pos);
            for p in (0..t).step_by(101).chain(t - 3..t) {
                let (refv, l1) =
                    ref_score_l1(&q[r * H * D..(r + 1) * H * D], &wt[r * H..(r + 1) * H], &pools[p * D..(p + 1) * D]);
                worst.add(sc[r * st + p], f2h(refv));
                assert!(score_close(sc[r * st + p], f2h(refv), l1), "ctx {ctx} p0 {p0} row {r} pool {p}");
            }
            let host = ref_topk(&sc[r * st..], t, SEL_POOLS);
            check_row(&idx, &meta, r, &expect_row(&host, pos), &format!("ctx {ctx} p0 {p0}"));
        }
        eprintln!(
            "ctx {ctx} p0 {p0} R {rr}: scores vs float64 bit-identical {:.5} (max {} ulp); selection exact",
            worst.frac_exact(),
            worst.max
        );
    }
}

/// Golden hashes of the kernels' outputs on seeded synthetic inputs (no pack needed): the key side, the plane, the scores and the
/// selection for several call shapes. A change that must keep every bit diffs this file against one written by the old build.
///
///     REXL3_GOLDEN_OUT=golden_dsa.txt cargo test --release -p gb-cuda --test dsa golden_dsa_hashes -- --ignored
#[test]
#[ignore]
fn golden_dsa_hashes() {
    use std::io::Write;
    let Some(_st) = gpu() else { return };
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    let mut f = std::fs::File::create(path).unwrap();
    let w = Weights::synthetic(1234);
    let slots = 3;
    let lens = [2600usize, 4111, 9000];
    let mut rig = Rig::new(slots, 9216, 256, &w, 41);
    let mut rng = Rng(555);
    let mut out = String::new();
    for (s, &len) in lens.iter().enumerate() {
        let (k, g) = (rand_f16(&mut rng, len * D, 0.0, 3.0), rand_f16(&mut rng, len * D, 0.0, 1.5));
        let mut pos = 0;
        while pos < len {
            let n = (97 + 5 * s).min(len - pos);
            rig.append(s, pos, &k[pos * D..(pos + n) * D], &g[pos * D..(pos + n) * D]);
            if pos == 0 {
                let kg = rig.scratch.kg.to_vec::<u16>().unwrap();
                out += &format!("slot {s} key side first call {:016x}\n", fnv(bytes_of(&kg[..n * 2 * D])));
            }
            pos += n;
        }
        let plane = rig.pool_plane();
        let mut h = 0xcbf29ce484222325u64;
        for p in 0..len / P {
            for &b in bytes_of(&rig.pool_key(&plane, s, p)) {
                h = (h ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
        out += &format!("slot {s} pool plane ({} pools) {h:016x}\n", len / P);
    }
    for (name, rows) in [
        ("R1 slot0 p2599", vec![(0usize, 2599usize)]),
        ("R1 slot2 p8999", vec![(2, 8999)]),
        ("R3 slot1 p2046", vec![(1, 2046), (1, 2047), (1, 2048)]),
        ("R9 slot1 p4100", (0..9).map(|i| (1, 4100 + i)).collect()),
        ("mixed slots", vec![(0, 2047), (0, 2048), (0, 2049), (2, 8990), (2, 8991), (1, 2050)]),
        ("R128 slot2", (0..128).map(|i| (2, 3000 + i)).collect()),
    ] {
        let n = rows.len();
        let q = rand_f16(&mut rng, n * H * D, 0.0, 1.0);
        let wt = rand_f16(&mut rng, n * H, 0.3, 1.0);
        let (sc, idx, meta) = rig.select(&rows, &q, &wt);
        let st = rig.scratch.s_stride;
        let mut hs = 0xcbf29ce484222325u64;
        for (r, &(_, pos)) in rows.iter().enumerate() {
            let t = if dsa::is_sparse(pos) { dsa::row_pools(pos) } else { 0 };
            for &b in bytes_of(&sc[r * st..r * st + t]) {
                hs = (hs ^ b as u64).wrapping_mul(0x100000001b3);
            }
        }
        out += &format!(
            "{name} scores {hs:016x} selection {:016x} meta {:016x}\n",
            fnv(bytes_of(&idx[..n * K_STRIDE])),
            fnv(bytes_of(&meta))
        );
    }
    eprint!("{out}");
    f.write_all(out.as_bytes()).unwrap();
}

// ------------------------------------------------------------------------------------------------ benchmark

mod cuda_events {
    use core::ffi::c_void;
    extern "C" {
        pub fn cudaEventCreate(e: *mut *mut c_void) -> i32;
        pub fn cudaEventRecord(e: *mut c_void, s: *mut c_void) -> i32;
        pub fn cudaEventSynchronize(e: *mut c_void) -> i32;
        pub fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
        pub fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, n: usize, kind: i32, s: *mut c_void) -> i32;
    }
}

/// Scoring and selection time against the bytes the indexer must read, per layer, at 8k, 32k, 131k and 262k tokens of context, for 1-row
/// decode and verify windows. Cold iterations start with a 100 MB read of another plane (four L2s) so the pool plane comes from DRAM, as in a real step
/// where the 11 layers' planes and everything else evict each other; the "warm" columns skip the flush.
///
///     cargo test --release -p gb-cuda --test dsa bench_dsa_indexer -- --ignored --nocapture
#[test]
#[ignore]
fn bench_dsa_indexer() {
    use cuda_events::*;
    let Some(st) = gpu() else { return };
    let w = Weights::synthetic(2);
    // a read-only flush (a memset would leave dirty lines whose write-back is charged to the measured kernel): scoring one row over a
    // 1,572,864-token plane reads 100 MB, four L2s
    let mut fl = Rig::new(1, 1_572_864, 256, &w, 5);
    let (flq, flw) = (DevBuf::zeroed(H * D * 2).unwrap(), DevBuf::zeroed(H * 2).unwrap());
    let flri = RowIndex::from_host(&[0], &[1_572_863]).unwrap();
    let flush = |st: &Stream, fl: &mut Rig| {
        dsa::score(&fl.planes, &mut fl.scratch, &flq, &flw, &flri, 1, 393_216, &fl.table, st).unwrap()
    };
    let mk = || unsafe {
        let mut e = std::ptr::null_mut();
        cudaEventCreate(&mut e);
        e
    };
    let (e0, e1, e2) = (mk(), mk(), mk());
    let ms = |a, b| unsafe {
        let mut t = 0f32;
        cudaEventElapsedTime(&mut t, a, b);
        t as f64
    };
    let median = |mut v: Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    // a pure streaming reference: a device copy of 16 MiB (reads 16 MiB, writes 16 MiB), cold
    {
        let src = DevBuf::zeroed(16 << 20).unwrap();
        let dst = DevBuf::zeroed(16 << 20).unwrap();
        let mut t = Vec::new();
        for _ in 0..30 {
            flush(&st, &mut fl);
            unsafe {
                cudaEventRecord(e0, st.raw());
                cudaMemcpyAsync(dst.ptr(), src.ptr(), 16 << 20, 3, st.raw());
                cudaEventRecord(e1, st.raw());
                cudaEventSynchronize(e1);
            }
            t.push(ms(e0, e1));
        }
        let m = median(t);
        eprintln!("reference: D2D copy of 16 MiB (cold) {:.1} us = {:.0} GB/s (read + write)", m * 1e3, 2.0 * 16.777216 / m);
    }
    eprintln!(
        "{:>8} {:>3} | {:>9} {:>9} {:>9} | {:>9} {:>9} | {:>9} {:>7}",
        "ctx", "R", "score us", "select us", "total us", "warm sc", "warm tot", "plane MB", "GB/s"
    );
    let mut rng = Rng(7);
    for &ctx in &[8192usize, 32768, 131072, 262144] {
        let mut rig = Rig::new(1, ctx, 256, &w, 3);
        let n_pools = ctx / P;
        let pools = rand_f16(&mut rng, n_pools * D, 0.0, 1.0);
        rig.load_pools(0, &pools);
        for &rr in &[1usize, 2, 4, 9, 16] {
            let q = rand_f16(&mut rng, rr * H * D, 0.0, 1.0);
            let wt = rand_f16(&mut rng, rr * H, 0.5, 0.5);
            let (slots, poss): (Vec<i32>, Vec<i32>) = (0..rr).map(|r| (0, (ctx - rr + r) as i32)).unzip();
            let ri = RowIndex::from_host(&slots, &poss).unwrap();
            let (qd, wd) = (DevBuf::from_slice(&q).unwrap(), DevBuf::from_slice(&wt).unwrap());
            let (mut cold_s, mut cold_t, mut warm_s, mut warm_t) = (vec![], vec![], vec![], vec![]);
            for it in 0..60 {
                let cold = it % 2 == 0;
                if cold {
                    flush(&st, &mut fl);
                }
                unsafe { cudaEventRecord(e0, st.raw()) };
                dsa::score(&rig.planes, &mut rig.scratch, &qd, &wd, &ri, rr, n_pools, &rig.table, &st).unwrap();
                unsafe { cudaEventRecord(e1, st.raw()) };
                let s_stride = rig.scratch.s_stride;
                let dsa::Scratch { scores, indices, meta, .. } = &mut rig.scratch;
                dsa::select_scores(scores, s_stride, &ri, rr, indices, meta, &st).unwrap();
                unsafe {
                    cudaEventRecord(e2, st.raw());
                    cudaEventSynchronize(e2);
                }
                if it >= 4 {
                    let (a, b) = (ms(e0, e1) * 1e3, ms(e1, e2) * 1e3);
                    if cold {
                        cold_s.push(a);
                        cold_t.push(a + b);
                    } else {
                        warm_s.push(a);
                        warm_t.push(a + b);
                    }
                }
            }
            let (cs, ct) = (median(cold_s), median(cold_t));
            let plane_mb = n_pools as f64 * 256.0 / 1e6;
            eprintln!(
                "{:>8} {:>3} | {:>9.1} {:>9.1} {:>9.1} | {:>9.1} {:>9.1} | {:>9.2} {:>7.0}",
                ctx,
                rr,
                cs,
                ct - cs,
                ct,
                median(warm_s),
                median(warm_t),
                plane_mb,
                plane_mb * 1e-3 / (cs * 1e-6)
            );
        }
    }
}
