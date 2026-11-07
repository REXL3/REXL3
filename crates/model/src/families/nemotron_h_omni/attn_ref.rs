//! The attention mixer of Nemotron 3.5 Super VL's `*` layers, as an fp32 CPU reference.
//!
//! These layers are GQA with no rotary embedding (NoPE), no biases, no q or k norm, and full causal attention with no sliding window.
//! This follows the ExLlamaV3 fork's `Attention` (`exllamav3/modules/attn.py`) with `rope_settings = None`, `sm_scale = head_dim ** -0.5`
//! and `sliding_window = -1`, and `nemotronh_block` (`exllamav3/architecture/nemotronh.py`), which passes exactly those.
//!
//! The input rows are already normalized (the block's `attn_norm` runs first), and the output is the o_proj result before the residual
//! add. Neither the norm nor the residual is here. Weights are PyTorch `Linear` layout, `[out, in]` row-major, already dequantized to
//! fp32; the EXL3 unpacking is not here either.
//!
//! The reduction order is fixed by the key index and is the tree of `tools/attn_gpu_design.md` (decisions 1 to 5). Each output element
//! is one sequential sum over the input dimension. Each query at absolute position `p` attends to keys `0..=p`, which split into
//! 512-key chunks at absolute multiples of 512; each chunk runs an online softmax over its 64-key tiles in ascending order, and the
//! chunk states merge in a fixed balanced binary tree over the chunk index. No loop shape depends on how a call was chunked, so a
//! prefill in one call and the same prefill in any chunks with the cache agree bit for bit, and so do the GPU's paths.
//!
//! Numerics of the cache and the tree: K and V are rounded once to fp16 when they are appended (the cache dtype), and P is rounded
//! to fp16 once per 64-key tile for the weighted sum of values. Scores and the softmax denominator stay fp32. The oracle keeps these
//! roundings on purpose, because the GPU kernels store the same fp16 K and V and round P the same way. Queries and the projections
//! stay fp32.

/// The shape of one attention layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geom {
    /// The residual width: `q_proj` and `o_proj` input or output, and the `k_proj` and `v_proj` input.
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
}

impl Geom {
    /// The measured `*` layers: `q_proj` and `o_proj` are `[4096, 4096]`, and `k_proj` and `v_proj` are `[256, 4096]`.
    pub const OMNI_STAR: Geom = Geom { hidden: 4096, n_heads: 32, n_kv: 2, head_dim: 128 };

    /// The query width, `n_heads * head_dim`.
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// The key or value width, `n_kv * head_dim`.
    pub fn kv_dim(&self) -> usize {
        self.n_kv * self.head_dim
    }

    /// The logit scale, `head_dim^-0.5`.
    pub fn scale(&self) -> f32 {
        head_scale(self.head_dim)
    }
}

/// `head_dim^-0.5`, computed in f64 and rounded once to f32.
fn head_scale(head_dim: usize) -> f32 {
    (1.0 / (head_dim as f64).sqrt()) as f32
}

/// Keys per chunk. Chunk `c` covers the absolute keys `[512 c, 512 c + 512)`, so chunk edges never move with a call.
pub const CHUNK_KEYS: usize = 512;
/// Keys per tile inside a chunk. P is rounded to fp16 once per tile.
pub const TILE_KEYS: usize = 64;

/// Rounds to the nearest fp16 (ties to even) and widens back. It is the crate's one-rounding conversion (`gb_exl3::f16`), which
/// rounds once from f64 and never through f32.
pub fn round_f16(x: f32) -> f32 {
    gb_exl3::f16::f16_to_f64(gb_exl3::f16::f64_to_f16(x as f64)) as f32
}

/// `y[t][o] = sum_i w[o][i] * x[t][i]`, with `w` row-major `[out_dim, in_dim]` and `x` row-major `[rows, in_dim]`. Each output is one
/// sequential sum over `i`, so a row's result does not depend on how many rows the call carries.
fn linear(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    assert!(in_dim > 0 && x.len() % in_dim == 0, "input is not whole rows of {in_dim}");
    assert_eq!(w.len(), out_dim * in_dim, "weight is not [{out_dim}, {in_dim}]");
    let rows = x.len() / in_dim;
    let mut y = vec![0f32; rows * out_dim];
    for t in 0..rows {
        let xr = &x[t * in_dim..(t + 1) * in_dim];
        for o in 0..out_dim {
            let wr = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0f32;
            for (a, b) in wr.iter().zip(xr) {
                acc += a * b;
            }
            y[t * out_dim + o] = acc;
        }
    }
    y
}

/// The q, k and v projections of `h` (`[rows, hidden]`): `q` is `[rows, q_dim]`, and `k` and `v` are `[rows, kv_dim]`. No bias, no norm,
/// no rotation.
pub fn project_qkv(h: &[f32], g: &Geom, wq: &[f32], wk: &[f32], wv: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (linear(h, wq, g.hidden, g.q_dim()), linear(h, wk, g.hidden, g.kv_dim()), linear(h, wv, g.hidden, g.kv_dim()))
}

/// The state of one chunk for one query head: the running max `m`, the sum of exponentials `l` and the unnormalized output `o`, all
/// fp32. The attention output is `o / l` after every chunk is merged.
#[derive(Clone, Debug)]
struct Partial {
    m: f32,
    l: f32,
    o: Vec<f32>,
}

/// The state of the keys `[c_start, min(c_start + 512, n))` for one query head `qh`, over the tiles of 64 keys in ascending order.
/// Each tile raises the running max to `m_new`, rescales the old state by `alpha = exp(m_old - m_new)` (zero before the first tile),
/// then adds its own terms. The denominator takes `p = exp(s - m_new)` in fp32; the weighted sum of values takes `fp16(p)`. Within a
/// tile, the sum over keys is sequential. A tile holds only keys below `n`, so a masked key is never read and never contributes.
#[allow(clippy::too_many_arguments)]
fn chunk_partial(
    qh: &[f32],
    k: &[f32],
    v: &[f32],
    kd: usize,
    kv_off: usize,
    scale: f32,
    c_start: usize,
    n: usize,
    round_p: bool,
) -> Partial {
    let hd = qh.len();
    let c_end = (c_start + CHUNK_KEYS).min(n);
    let mut m = f32::NEG_INFINITY;
    let mut l = 0f32;
    let mut o = vec![0f32; hd];
    let mut acc = vec![0f32; hd];
    let mut s = [0f32; TILE_KEYS];
    let mut ts = c_start;
    while ts < c_end {
        let w = (c_end - ts).min(TILE_KEYS);
        let mut tile_max = f32::NEG_INFINITY;
        for (t, st) in s[..w].iter_mut().enumerate() {
            let kr = &k[(ts + t) * kd + kv_off..][..hd];
            let mut dot = 0f32;
            for (a, b) in qh.iter().zip(kr) {
                dot += a * b;
            }
            *st = dot * scale;
            tile_max = tile_max.max(*st);
        }
        let m_new = m.max(tile_max);
        let alpha = if m == f32::NEG_INFINITY { 0.0 } else { (m - m_new).exp() };
        let mut tile_sum = 0f32;
        for st in &mut s[..w] {
            let p = (*st - m_new).exp();
            tile_sum += p;
            *st = if round_p { round_f16(p) } else { p };
        }
        acc.iter_mut().for_each(|a| *a = 0.0);
        for (t, &p) in s[..w].iter().enumerate() {
            let vr = &v[(ts + t) * kd + kv_off..][..hd];
            for (a, x) in acc.iter_mut().zip(vr) {
                *a += p * x;
            }
        }
        l = l * alpha + tile_sum;
        for (oi, a) in o.iter_mut().zip(&acc) {
            *oi = *oi * alpha + a;
        }
        m = m_new;
        ts += w;
    }
    Partial { m, l, o }
}

/// Merges two adjacent chunk states, `a` the lower one: `m = max(ma, mb)`, `sa = exp(ma - m)`, `sb = exp(mb - m)`, `l = la sa + lb sb`,
/// `o = oa sa + ob sb`. An empty side (`m == -inf`, the identity) returns the other side bit for bit.
fn merge(a: Partial, b: Partial) -> Partial {
    if b.m == f32::NEG_INFINITY {
        return a;
    }
    if a.m == f32::NEG_INFINITY {
        return b;
    }
    let m = a.m.max(b.m);
    let sa = (a.m - m).exp();
    let sb = (b.m - m).exp();
    let l = a.l * sa + b.l * sb;
    let o = a.o.iter().zip(&b.o).map(|(x, y)| x * sa + y * sb).collect();
    Partial { m, l, o }
}

/// The fixed balanced binary tree over chunk indices. The leaves are padded to the next power of two with the identity, and node
/// `(lo, len)` merges its two halves; a half that starts past the last chunk is the identity, so it is skipped. A query's chunk count
/// is its own, `ceil((p + 1) / 512)`, so its tree depends only on its position and not on how the call was split.
fn merge_tree(parts: &[Partial]) -> Partial {
    fn node(parts: &[Partial], lo: usize, len: usize) -> Option<Partial> {
        if lo >= parts.len() {
            return None;
        }
        if len == 1 {
            return Some(parts[lo].clone());
        }
        let half = len / 2;
        let left = node(parts, lo, half);
        match node(parts, lo + half, half) {
            Some(right) => Some(merge(left.expect("a node's left half holds a chunk"), right)),
            None => left,
        }
    }
    node(parts, 0, parts.len().next_power_of_two()).expect("a query has at least one chunk")
}

/// Causal GQA over a cache. `q` holds the `t` new query rows, `[t, n_heads * head_dim]`. `k` and `v` hold the whole sequence,
/// `[past_len + t, n_kv * head_dim]`: the cached past first, then the new rows. Query row `i` has absolute position `p = past_len + i`
/// and attends to keys `0..=p`. Query head `h` reads kv head `h / (n_heads / n_kv)`, and logits are `(q . k) * head_dim^-0.5`.
///
/// Each query runs the tree of the module docs: 512-key chunks of 64-key tiles, merged over the chunk index, with P rounded to fp16
/// once per tile. This is the oracle the GPU kernels are checked against. The result for a row depends only on that row's query and
/// on `k` and `v` up to its position. For the f64 formula checks use [`causal_gqa_fp32`].
pub fn causal_gqa(q: &[f32], k: &[f32], v: &[f32], n_heads: usize, n_kv: usize, head_dim: usize, past_len: usize) -> Vec<f32> {
    causal_gqa_with(q, k, v, n_heads, n_kv, head_dim, past_len, true)
}

/// The same tree and the same chunk and tile order as [`causal_gqa`], with P left in fp32. It differs from the exact formula only by
/// fp32 accumulation and `exp`, so it is the reference for the f64 checks. `k` and `v` are used as given (the cache's fp16 rounding is
/// applied where the cache is written, [`KvCache`]).
pub fn causal_gqa_fp32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    past_len: usize,
) -> Vec<f32> {
    causal_gqa_with(q, k, v, n_heads, n_kv, head_dim, past_len, false)
}

#[allow(clippy::too_many_arguments)]
fn causal_gqa_with(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    past_len: usize,
    round_p: bool,
) -> Vec<f32> {
    assert!(n_kv > 0 && n_heads % n_kv == 0 && head_dim > 0, "GQA needs whole groups of query heads");
    let qd = n_heads * head_dim;
    let kd = n_kv * head_dim;
    assert!(q.len() % qd == 0, "q is not whole rows of {qd}");
    let t_new = q.len() / qd;
    let total = past_len + t_new;
    assert_eq!(k.len(), total * kd, "k must hold the past and the new keys");
    assert_eq!(v.len(), total * kd, "v must hold the past and the new values");
    let group = n_heads / n_kv;
    let scale = head_scale(head_dim);
    let mut out = vec![0f32; t_new * qd];
    for i in 0..t_new {
        let n = past_len + i + 1;
        for h in 0..n_heads {
            let kv = h / group;
            let qh = &q[i * qd + h * head_dim..][..head_dim];
            let parts: Vec<Partial> = (0..n.div_ceil(CHUNK_KEYS))
                .map(|c| chunk_partial(qh, k, v, kd, kv * head_dim, scale, c * CHUNK_KEYS, n, round_p))
                .collect();
            let state = merge_tree(&parts);
            for (dst, x) in out[i * qd + h * head_dim..][..head_dim].iter_mut().zip(&state.o) {
                *dst = x / state.l;
            }
        }
    }
    out
}

/// One layer's keys and values: a row of `n_kv * head_dim` values per cached position, appended in order. They are fp16 values
/// ([`round_f16`]) held as f32.
#[derive(Clone, Debug)]
pub struct KvCache {
    n_kv: usize,
    head_dim: usize,
    k: Vec<f32>,
    v: Vec<f32>,
}

impl KvCache {
    pub fn new(g: &Geom) -> Self {
        assert!(g.n_kv > 0 && g.head_dim > 0, "a KV cache needs a non-empty geometry");
        Self { n_kv: g.n_kv, head_dim: g.head_dim, k: Vec::new(), v: Vec::new() }
    }

    /// The number of cached positions.
    pub fn len(&self) -> usize {
        self.k.len() / (self.n_kv * self.head_dim)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// All cached keys, `[len, n_kv * head_dim]`.
    pub fn keys(&self) -> &[f32] {
        &self.k
    }

    /// All cached values, `[len, n_kv * head_dim]`.
    pub fn values(&self) -> &[f32] {
        &self.v
    }

    /// Appends the new keys and values, each rounded once to fp16 (the cache dtype) from the fp32 projection.
    fn append(&mut self, k: &[f32], v: &[f32]) {
        self.k.extend(k.iter().map(|&x| round_f16(x)));
        self.v.extend(v.iter().map(|&x| round_f16(x)));
    }
}

/// One `*` layer's weights, in PyTorch `Linear` layout: `wq` is `[q_dim, hidden]`, `wk` and `wv` are `[kv_dim, hidden]`, and `wo` is
/// `[hidden, q_dim]`. None of them has a bias.
#[derive(Clone, Copy, Debug)]
pub struct Weights<'a> {
    pub geom: Geom,
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub wo: &'a [f32],
}

/// The attention mixer on `h_rows` (`[t, hidden]`, already normalized): project to q, k and v, append the new k and v to `kv_cache`,
/// attend each new query causally over the whole cache, and project the heads with `wo`. The result is `[t, hidden]`, before the residual.
pub fn attention_block(h_rows: &[f32], w: &Weights<'_>, kv_cache: &mut KvCache) -> Vec<f32> {
    let g = w.geom;
    assert!(kv_cache.n_kv == g.n_kv && kv_cache.head_dim == g.head_dim, "the KV cache geometry differs from the layer");
    let past = kv_cache.len();
    let (q, k, v) = project_qkv(h_rows, &g, w.wq, w.wk, w.wv);
    kv_cache.append(&k, &v);
    let heads = causal_gqa(&q, kv_cache.keys(), kv_cache.values(), g.n_heads, g.n_kv, g.head_dim, past);
    linear(&heads, w.wo, g.q_dim(), g.hidden)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::nemotron_h_omni::config::OmniConfig;

    /// The published pack's `config.json`, committed as a fixture (`testdata/nemotron_omni/config.json`).
    const PACK: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");
    const G: Geom = Geom::OMNI_STAR;
    /// The measured head geometry with a small residual width, for the property tests. Projections cost `hidden` per output, and the
    /// unoptimized test build is slow; the attention itself is unchanged.
    const SMALL: Geom = Geom { hidden: 256, ..G };

    /// A xorshift stream: the test data is deterministic and needs no crate.
    struct Rng(u64);

    impl Rng {
        /// Uniform in `[-scale, scale)`.
        fn next_f32(&mut self, scale: f32) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        }

        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.next_f32(scale)).collect()
        }
    }

    /// Projection weights for one geometry, scaled so the projections are O(1) for O(1) inputs.
    struct Owned {
        geom: Geom,
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
    }

    impl Owned {
        fn random(geom: Geom, seed: u64) -> Self {
            let mut r = Rng(seed);
            let s_in = 1.0 / (geom.hidden as f32).sqrt();
            let s_q = 1.0 / (geom.q_dim() as f32).sqrt();
            Owned {
                geom,
                wq: r.vec(geom.q_dim() * geom.hidden, s_in),
                wk: r.vec(geom.kv_dim() * geom.hidden, s_in),
                wv: r.vec(geom.kv_dim() * geom.hidden, s_in),
                wo: r.vec(geom.hidden * geom.q_dim(), s_q),
            }
        }

        fn view(&self) -> Weights<'_> {
            Weights { geom: self.geom, wq: &self.wq, wk: &self.wk, wv: &self.wv, wo: &self.wo }
        }
    }

    /// Row lengths for a prefill of `total` rows in chunks of `chunk` (the last one takes the remainder).
    fn chunked(total: usize, chunk: usize) -> Vec<usize> {
        let mut sizes = vec![chunk; total / chunk];
        if total % chunk != 0 {
            sizes.push(total % chunk);
        }
        sizes
    }

    fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        if let Some(i) = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits()) {
            panic!("{what}: first bit difference at {i}: {} vs {}", got[i], want[i]);
        }
    }

    fn widen(x: &[f32]) -> Vec<f64> {
        x.iter().map(|&a| a as f64).collect()
    }

    fn max_abs_diff(got: &[f32], want: &[f64]) -> f64 {
        assert_eq!(got.len(), want.len());
        got.iter().zip(want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max)
    }

    /// `y[t][o] = sum_i w[o][i] * x[t][i]` in f64, written out per element.
    fn naive_linear(x: &[f64], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f64> {
        let rows = x.len() / in_dim;
        let mut y = Vec::with_capacity(rows * out_dim);
        for t in 0..rows {
            for o in 0..out_dim {
                let mut acc = 0f64;
                for i in 0..in_dim {
                    acc += w[o * in_dim + i] as f64 * x[t * in_dim + i];
                }
                y.push(acc);
            }
        }
        y
    }

    /// The naive loop in f64 over given q, k and v. Row `i` is at position `past + i` and sees keys `0..=past+i`. Query head `h` reads
    /// kv head `h / 16`, with the group size of the measured model written as the literal 16. The logit scale is `128^-0.5` and nothing
    /// rotates the queries or keys.
    fn naive_causal(q: &[f64], k: &[f64], v: &[f64], past: usize, scale: f64) -> Vec<f64> {
        let (n_heads, n_kv, hd) = (32usize, 2usize, 128usize);
        let (qd, kd) = (n_heads * hd, n_kv * hd);
        let t = q.len() / qd;
        let mut out = vec![0f64; t * qd];
        for i in 0..t {
            let pos = past + i;
            for h in 0..n_heads {
                let kv = h / 16;
                let logits: Vec<f64> = (0..=pos)
                    .map(|j| {
                        let dot: f64 = (0..hd).map(|d| q[i * qd + h * hd + d] * k[j * kd + kv * hd + d]).sum();
                        dot * scale
                    })
                    .collect();
                let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = logits.iter().map(|x| (x - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for d in 0..hd {
                    out[i * qd + h * hd + d] = (0..=pos).map(|j| e[j] / z * v[j * kd + kv * hd + d]).sum();
                }
            }
        }
        out
    }

    /// The whole mixer for a prefill from an empty cache, in f64: projections, the naive loop, and o_proj.
    /// K and V are rounded to fp16 here, as the cache stores them, so the only difference from the oracle's fp32 path is accumulation
    /// order. P is exact: the comparison is with [`causal_gqa_fp32`], and the fp16 rounding of P has its own test.
    fn naive_mixer(h: &[f32], w: &Owned, scale: f64) -> Vec<f64> {
        let g = w.geom;
        let hx = widen(h);
        let q = naive_linear(&hx, &w.wq, g.hidden, g.q_dim());
        let fp16 = |x: Vec<f64>| -> Vec<f64> { x.into_iter().map(|a| round_f16(a as f32) as f64).collect() };
        let k = fp16(naive_linear(&hx, &w.wk, g.hidden, g.kv_dim()));
        let v = fp16(naive_linear(&hx, &w.wv, g.hidden, g.kv_dim()));
        let o = naive_causal(&q, &k, &v, 0, scale);
        naive_linear(&o, &w.wo, g.q_dim(), g.hidden)
    }

    #[test]
    fn the_geometry_is_the_published_one() {
        let cfg = OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap();
        assert_eq!((cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim), (G.hidden, G.n_heads, G.n_kv, G.head_dim));
        assert_eq!((G.q_dim(), G.kv_dim()), (4096, 256));
    }

    #[test]
    fn a_future_token_does_not_change_an_earlier_output() {
        let w = Owned::random(SMALL, 1);
        let t = 37;
        let h = Rng(0x5EED_0001).vec(t * SMALL.hidden, 1.0);
        let base = attention_block(&h, &w.view(), &mut KvCache::new(&SMALL));
        let hd = SMALL.hidden;
        for changed in [1usize, 16, 36] {
            let mut h2 = h.clone();
            for x in &mut h2[changed * hd..(changed + 1) * hd] {
                *x += 1.0;
            }
            let out = attention_block(&h2, &w.view(), &mut KvCache::new(&SMALL));
            let split = changed * hd;
            assert_bits_eq(&out[..split], &base[..split], &format!("rows before token {changed}"));
            assert!(
                out[split..].iter().zip(&base[split..]).any(|(a, b)| a.to_bits() != b.to_bits()),
                "token {changed} changed nothing"
            );
        }
    }

    #[test]
    fn prefill_in_chunks_is_bitwise_equal_to_one_call() {
        let w = Owned::random(SMALL, 2);
        let t = 37;
        let h = Rng(0x5EED_0002).vec(t * SMALL.hidden, 1.0);
        let mut whole_cache = KvCache::new(&SMALL);
        let whole = attention_block(&h, &w.view(), &mut whole_cache);
        assert_eq!(whole_cache.len(), t);
        let plans = [vec![1; t], chunked(t, 5), chunked(t, 16), vec![t], vec![3, 1, 20, 13]];
        for plan in plans {
            assert_eq!(plan.iter().sum::<usize>(), t);
            let mut cache = KvCache::new(&SMALL);
            let mut out = Vec::with_capacity(whole.len());
            let mut start = 0;
            for rows in plan.iter().copied() {
                let block = &h[start * SMALL.hidden..(start + rows) * SMALL.hidden];
                out.extend(attention_block(block, &w.view(), &mut cache));
                start += rows;
            }
            let what = format!("chunks {plan:?}");
            assert_bits_eq(&out, &whole, &format!("{what}: output"));
            assert_bits_eq(cache.keys(), whole_cache.keys(), &format!("{what}: keys"));
            assert_bits_eq(cache.values(), whole_cache.values(), &format!("{what}: values"));
        }
    }

    /// Row lengths that sum to `total`, cycling through `pattern` (the last one is cut short).
    fn cycled(total: usize, pattern: &[usize]) -> Vec<usize> {
        let mut sizes = Vec::new();
        let mut left = total;
        for &p in pattern.iter().cycle() {
            if left == 0 {
                break;
            }
            let take = p.min(left);
            sizes.push(take);
            left -= take;
        }
        sizes
    }

    #[test]
    fn chunk_plans_across_the_512_boundary_are_bitwise_equal() {
        // 700 keys: one full 512-key chunk and a ragged second one. A small head count keeps the debug build fast; the tree is the
        // same for any head count.
        const T: usize = 700;
        let g = Geom { hidden: 64, n_heads: 4, n_kv: 2, head_dim: 32 };
        let w = Owned::random(g, 6);
        let h = Rng(0x5EED_0006).vec(T * g.hidden, 1.0);
        let mut whole_cache = KvCache::new(&g);
        let whole = attention_block(&h, &w.view(), &mut whole_cache);
        assert_eq!(whole_cache.len(), T);
        assert!(whole_cache.keys().iter().all(|&x| round_f16(x).to_bits() == x.to_bits()), "the cache holds fp16 values");
        let plans =
            [chunked(T, 1), chunked(T, 5), chunked(T, 16), chunked(T, 37), cycled(T, &[3, 1, 20, 13]), vec![512, 188], vec![T]];
        for plan in plans {
            assert_eq!(plan.iter().sum::<usize>(), T);
            let mut cache = KvCache::new(&g);
            let mut out = Vec::with_capacity(whole.len());
            let mut start = 0;
            for rows in plan.iter().copied() {
                let block = &h[start * g.hidden..(start + rows) * g.hidden];
                out.extend(attention_block(block, &w.view(), &mut cache));
                start += rows;
            }
            let what = format!("chunks {plan:?}");
            assert_bits_eq(&out, &whole, &format!("{what}: output"));
            assert_bits_eq(cache.keys(), whole_cache.keys(), &format!("{what}: keys"));
            assert_bits_eq(cache.values(), whole_cache.values(), &format!("{what}: values"));
        }
    }

    #[test]
    fn fp16_p_changes_the_result_by_at_most_the_weight_rounding() {
        // P is rounded to fp16 once per tile, so each weight moves by at most 2^-11 of itself. The output is a weighted sum of values
        // in [-1, 1] with weights that sum to one, so the change is a few 2^-11 at most. Small geometry: the debug build stays fast.
        let g = Geom { hidden: 64, n_heads: 4, n_kv: 2, head_dim: 32 };
        let total = 700;
        let mut r = Rng(0x5EED_0009);
        let q = r.vec(total * g.q_dim(), 1.0);
        let k = r.vec(total * g.kv_dim(), 1.0);
        let v = r.vec(total * g.kv_dim(), 1.0);
        let rounded = causal_gqa(&q, &k, &v, g.n_heads, g.n_kv, g.head_dim, 0);
        let exact = causal_gqa_fp32(&q, &k, &v, g.n_heads, g.n_kv, g.head_dim, 0);
        let change = max_abs_diff(&rounded, &widen(&exact));
        eprintln!("fp16 P: max abs change against the fp32 P tree {change:.3e} (2^-11 = {:.3e})", 2f64.powi(-11));
        assert!(change > 0.0, "the rounding of P changed nothing: it is not applied");
        assert!(change < 2e-3, "fp16 P moved the output by {change}");
    }

    #[test]
    #[allow(clippy::excessive_precision)] // the exact decimal of the fp16 value
    fn round_f16_is_the_fp16_grid_with_ties_to_even() {
        assert_eq!(round_f16(0.1).to_bits(), 0.0999755859375f32.to_bits());
        assert_eq!(round_f16(1.0 + 2f32.powi(-11)), 1.0, "tie between 1 and 1 + 2^-10 keeps the even 1");
        assert_eq!(round_f16(1.0 + 3.0 * 2f32.powi(-11)), 1.0 + 2f32.powi(-9), "tie keeps the even neighbour");
        assert_eq!(round_f16(65504.0), 65504.0);
        assert_eq!(round_f16(65519.0), 65504.0);
        assert!(round_f16(65520.0).is_infinite(), "65520 is the tie that overflows");
        assert_eq!(round_f16(2f32.powi(-24)), 2f32.powi(-24), "the smallest subnormal");
        assert_eq!(round_f16(2f32.powi(-25)), 0.0, "a tie to zero");
        assert_eq!(round_f16(-1.5), -1.5);
    }

    #[test]
    fn query_head_h_reads_kv_head_h_over_16() {
        let (past, t) = (3usize, 5usize);
        let mut r = Rng(0x5EED_0003);
        let q = r.vec(t * G.q_dim(), 1.0);
        let k = r.vec((past + t) * G.kv_dim(), 1.0);
        let v = r.vec((past + t) * G.kv_dim(), 1.0);
        let got = causal_gqa_fp32(&q, &k, &v, G.n_heads, G.n_kv, G.head_dim, past);
        let want = naive_causal(&widen(&q), &widen(&k), &widen(&v), past, 1.0 / 128f64.sqrt());
        let err = max_abs_diff(&got, &want);
        eprintln!("GQA mapping: causal_gqa vs naive loop, max abs error {err:.3e}");
        assert!(err < 1e-5, "max abs error {err}");

        // Structure, independent of the naive loop: changing kv head 1 alone must move query heads 16..32 and nothing else.
        let hd = G.head_dim;
        for (kv, moved) in [(1usize, 16..32), (0usize, 0..16)] {
            let mut v2 = v.clone();
            for j in 0..past + t {
                for d in 0..hd {
                    v2[j * G.kv_dim() + kv * hd + d] += 1.0;
                }
            }
            let out2 = causal_gqa_fp32(&q, &k, &v2, G.n_heads, G.n_kv, hd, past);
            for i in 0..t {
                for h in 0..G.n_heads {
                    let span = i * G.q_dim() + h * hd..i * G.q_dim() + (h + 1) * hd;
                    let same = got[span.clone()].iter().zip(&out2[span.clone()]).all(|(a, b)| a.to_bits() == b.to_bits());
                    assert_eq!(!same, moved.contains(&h), "kv head {kv} changed query head {h} = {same}");
                }
            }
        }
    }

    /// The GPU kernels (`gb_cuda::nemotron_attn`) against this oracle, on the measured geometry. The whole sequence is prefilled on
    /// the GPU and twelve rows are checked, the ones at chunk and tile edges and at the end. The oracle rounds K, V and P to fp16 where
    /// the kernels do, so the difference is fp32 accumulation order and `exp`. The error is `max |gpu - cpu|` over each (row, head),
    /// divided by that head's `max |cpu|`, and its worst value is the number to report. Skipped without a device.
    #[test]
    fn the_gpu_kernels_match_this_oracle_within_fp16_terms() {
        use gb_cuda::attention::BlockTable;
        use gb_cuda::nemotron_attn::{self as na, NemoKv, RowIndex, Scratch};
        use gb_cuda::{device_available, DevBuf, Stream};
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
        const L: usize = 1300;
        let (qd, kd) = (G.q_dim(), G.kv_dim());
        let mut r = Rng(0x5EED_0008);
        let q = r.vec(L * qd, 1.0);
        let k = r.vec(L * kd, 1.0);
        let v = r.vec(L * kd, 1.0);
        let kr: Vec<f32> = k.iter().map(|&x| round_f16(x)).collect();
        let vr: Vec<f32> = v.iter().map(|&x| round_f16(x)).collect();

        let pages = L.div_ceil(256);
        let table = BlockTable::from_host(&[(0..pages as u32).collect::<Vec<u32>>()], pages).unwrap();
        let mut kv = NemoKv::new(pages + 2).unwrap();
        let st = Stream::new().unwrap();
        let slots = vec![0i32; L];
        let pos: Vec<i32> = (0..L as i32).collect();
        let idx = RowIndex::from_host(&slots, &pos).unwrap();
        let kdev = DevBuf::from_slice(&k).unwrap();
        let vdev = DevBuf::from_slice(&v).unwrap();
        na::kv_write(&mut kv, &table, &idx, &kdev, &vdev, &st).unwrap();
        let qdev = DevBuf::from_slice(&q).unwrap();
        let mut out = DevBuf::zeroed(L * qd * 4).unwrap();
        let mut scratch = Scratch::new(L, L.div_ceil(512)).unwrap();
        na::attn_prefill(&qdev, &kv, &table, &idx, &mut scratch, &mut out, &st).unwrap();
        st.sync().unwrap();
        let got: Vec<f32> = out.to_vec().unwrap();
        assert!(got.iter().all(|x| x.is_finite()), "the GPU output has a non-finite value");

        let (mut worst_norm, mut worst_abs) = (0f64, 0f64);
        for i in [0usize, 1, 63, 64, 511, 512, 513, 575, 576, 1023, 1024, 1299] {
            let want =
                causal_gqa(&q[i * qd..(i + 1) * qd], &kr[..(i + 1) * kd], &vr[..(i + 1) * kd], G.n_heads, G.n_kv, G.head_dim, i);
            for h in 0..G.n_heads {
                let hd = G.head_dim;
                let w = &want[h * hd..][..hd];
                let g = &got[i * qd + h * hd..][..hd];
                let scale = w.iter().fold(0f64, |a, &x| a.max((x as f64).abs()));
                let abs = max_abs_diff(g, &widen(w));
                worst_abs = worst_abs.max(abs);
                worst_norm = worst_norm.max(abs / scale);
            }
        }
        eprintln!(
            "GPU vs oracle, L = {L}, 12 rows x 32 heads: max abs error {worst_abs:.3e}, max error / head max {worst_norm:.3e}"
        );
        assert!(worst_norm <= 1e-3, "GPU vs oracle: {worst_norm:.3e} of the head's max");
    }

    #[test]
    fn the_logit_scale_is_head_dim_to_the_minus_half_and_nothing_rotates() {
        assert!((G.scale() as f64 - 128f64.powf(-0.5)).abs() < 1e-7);

        let w = Owned::random(G, 4);
        let t = 3;
        let h = Rng(0x5EED_0004).vec(t * G.hidden, 1.0);
        // The mixer with the cache's fp16 K and V (as `attention_block` stores them) and the fp32 tree, so the f64 formula is the reference
        // at 1e-5. The fp16 rounding of P is tested on its own.
        let (q, k, v) = project_qkv(&h, &G, &w.wq, &w.wk, &w.wv);
        let k: Vec<f32> = k.iter().map(|&x| round_f16(x)).collect();
        let v: Vec<f32> = v.iter().map(|&x| round_f16(x)).collect();
        let heads = causal_gqa_fp32(&q, &k, &v, G.n_heads, G.n_kv, G.head_dim, 0);
        let got = linear(&heads, &w.wo, G.q_dim(), G.hidden);
        let want = naive_mixer(&h, &w, 128f64.powf(-0.5));
        let err = max_abs_diff(&got, &want);
        eprintln!("full block (hidden 4096): fp32-P mixer vs naive formula, max abs error {err:.3e}");
        assert!(err < 1e-5, "max abs error {err}");

        // The comparison must be able to fail: a unit scale moves the result far past the tolerance.
        let unscaled = naive_mixer(&h, &w, 1.0);
        let sensitivity = max_abs_diff(&got, &unscaled);
        eprintln!("sensitivity: unit scale differs by {sensitivity:.3e}");
        assert!(sensitivity > 1e-2, "the test cannot tell a unit scale from head_dim^-0.5");

        // NoPE: identical tokens give identical rows at every position. A rotation would make the rows differ.
        let row = Rng(0x5EED_0005).vec(G.hidden, 1.0);
        let same = attention_block(&row.repeat(6), &w.view(), &mut KvCache::new(&G));
        for i in 1..6 {
            for c in 0..G.hidden {
                assert!((same[i * G.hidden + c] - same[c]).abs() < 1e-5, "row {i} differs from row 0 at {c}");
            }
        }
    }
}
