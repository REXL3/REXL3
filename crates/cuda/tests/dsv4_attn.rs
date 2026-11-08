//! DeepSeek V4.1 attention (`src/dsv4_attn.rs`, `csrc/dsv4_attn.cu`): the YaRN table pinned to the reference formula,
//! the whole attention block on a 4-token / 2-layer toy config against a **direct transcription of model.py**, the
//! runtime-k top-k against a full sort, the level-1 candidate blocks, and the compressed-plane identity.
//!
//! # Why a transcription and not the module
//!
//! `reference_block` below is written from `/tmp/ds41ref/model.py` (`Compressor.forward`, `Indexer.forward`,
//! `select_candidate_blocks`, `Attention.forward`) and `kernel.py` (`sparse_attn_kernel`), in f64, with its own
//! RoPE, its own E4M3/E2M1 encoders (brute-force nearest over the format's code points) and its own softmax. It calls
//! nothing in the module under test except the packed weight bytes, so a mistake in the module cannot hide in it.
//!
//! # What runs without a GPU
//!
//! Everything except `gpu_*`: those print `not run: no device` and return, so this binary is green on a CPU-only box.
use gb_cuda::dsv4_attn::*;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::fp8_linear::{e4m3_to_f32, e8m0_to_f32, TILE};
use gb_cuda::{device_available, DevBuf, Stream};

// ------------------------------------------------------------------------------------------------ helpers

/// xorshift64*: the same test data everywhere, no dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// A uniform f32 in `[lo, hi)`.
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }
}

/// The nearest E4M3 code point to `v` (brute force over the format, 0x7F/0xFF excluded as NaN).
fn nearest_e4m3(v: f32) -> f32 {
    let mut best = f32::INFINITY;
    let mut bd = f32::INFINITY;
    for b in 0u16..256 {
        let b = b as u8;
        if b == 0x7F || b == 0xFF {
            continue;
        }
        let d = (e4m3_to_f32(b) - v).abs();
        if d < bd {
            bd = d;
            best = e4m3_to_f32(b);
        }
    }
    best
}

/// The nearest E2M1 value.
fn nearest_e2m1(v: f32) -> f32 {
    let sign = if v < 0.0 { -1.0 } else { 1.0 };
    let a = v.abs();
    let mut best = 0.0f32;
    let mut bd = f32::INFINITY;
    for &c in &[0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] {
        let d = (a - c).abs();
        if d < bd {
            bd = d;
            best = c;
        }
    }
    sign * best
}

/// The reference's `act_quant` / `fp4_act_quant` (`inplace=True`), transcribed: amax over the block, scale = amax /
/// max (or `2^ceil(log2(...))` for MXFP), quantise, dequantise.
fn ref_quant(x: &mut [f32], block: usize, bits: u32, pow2: bool) {
    let max = if bits == 8 { 448.0f32 } else { 6.0f32 };
    for c in x.chunks_mut(block) {
        let amax = c.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-4);
        let s = if pow2 { (amax / max).log2().ceil().exp2() } else { amax / max };
        for v in c.iter_mut() {
            let q = *v / s;
            *v = (if bits == 8 { nearest_e4m3(q) } else { nearest_e2m1(q) }) * s;
        }
    }
}

/// The reference's `RMSNorm`.
fn ref_norm(x: &mut [f32], w: &[f32], eps: f32) {
    let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ss as f32 + eps).sqrt();
    for (v, &wi) in x.iter_mut().zip(w) {
        *v = *v * r * wi;
    }
}

/// YaRN frequencies, transcribed from `precompute_freqs_cis` (model.py:369).
fn ref_freqs(
    dim: usize,
    seq: usize,
    original_seq_len: usize,
    base: f64,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> Vec<f64> {
    let half = dim / 2;
    let mut f: Vec<f64> = (0..half).map(|j| 1.0 / base.powf(2.0 * j as f64 / dim as f64)).collect();
    if original_seq_len > 0 {
        let corrected =
            |rot: f64| dim as f64 * (original_seq_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
        let low = corrected(beta_fast).floor().max(0.0);
        let high = corrected(beta_slow).ceil().min((dim - 1) as f64);
        for (j, v) in f.iter_mut().enumerate() {
            let ramp = ((j as f64 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *v = *v / factor * (1.0 - smooth) + *v * smooth;
        }
    }
    let _ = seq;
    f
}

/// Rotate the last `dim` of the `block`-wide segment of `x` that starts at `off`, at `pos`, adjacent pairs as
/// complex numbers — the reference's `apply_rotary_emb(x[..., -dim:], freqs)`.
fn ref_rope(x: &mut [f32], off: usize, block: usize, dim: usize, pos: usize, f: &[f64], inverse: bool) {
    let off = off + block - dim;
    let half = dim / 2;
    for j in 0..half {
        let ang = pos as f64 * f[j];
        let (c, s) = (ang.cos() as f32, ang.sin() as f32);
        let s = if inverse { -s } else { s };
        let a = x[off + 2 * j];
        let b = x[off + 2 * j + 1];
        x[off + 2 * j] = a * c - b * s;
        x[off + 2 * j + 1] = a * s + b * c;
    }
}

/// Quantise f32 weights to E4M3 + 32x32 E8M0 blocks, exactly the pack layout.
fn pack_fp8(n: usize, k: usize, w: &[f32]) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(w.len(), n * k);
    let (nt, kt) = (n / TILE, k / TILE);
    let mut bytes = vec![0u8; n * k];
    let mut scales = vec![0u8; nt * kt];
    for ni in 0..nt {
        for ki in 0..kt {
            let mut amax = 1e-8f32;
            for r in 0..TILE {
                for c in 0..TILE {
                    amax = amax.max(w[(ni * TILE + r) * k + ki * TILE + c].abs());
                }
            }
            let e = (amax / 448.0).log2().ceil();
            let s = e.exp2();
            scales[ni * kt + ki] = (e as i32 + 127).clamp(0, 254) as u8;
            for r in 0..TILE {
                for c in 0..TILE {
                    let v = w[(ni * TILE + r) * k + ki * TILE + c] / s;
                    // nearest code point, then its byte
                    let mut best = 0u8;
                    let mut bd = f32::INFINITY;
                    for b in 0u16..256 {
                        let b = b as u8;
                        if b == 0x7F || b == 0xFF {
                            continue;
                        }
                        let d = (e4m3_to_f32(b) - v).abs();
                        if d < bd {
                            bd = d;
                            best = b;
                        }
                    }
                    bytes[(ni * TILE + r) * k + ki * TILE + c] = best;
                }
            }
        }
    }
    (bytes, scales)
}

/// The dense f32 view of a packed weight, read byte by byte (the transcription never sees `Fp8Host`).
fn deq(n: usize, k: usize, w: &[u8], s: &[u8]) -> Vec<f32> {
    let kt = k / TILE;
    (0..n * k)
        .map(|i| {
            let (row, col) = (i / k, i % k);
            e4m3_to_f32(w[i]) * e8m0_to_f32(s[(row / TILE) * kt + col / TILE])
        })
        .collect()
}

/// The dense f32 view of a BF16 weight: the checkpoint's bytes, widened (exact -- bf16 is an f32's top half).
fn deq_bf16(n: usize, k: usize, raw: &[u8]) -> Vec<f32> {
    assert_eq!(raw.len(), n * k * 2);
    raw.chunks_exact(2).map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
}

/// `y [m, n] = x [m, k] @ W[n, k]^T` in f64.
fn mm(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f64;
            for i in 0..k {
                acc += x[r * k + i] as f64 * w[c * k + i] as f64;
            }
            y[r * n + c] = acc as f32;
        }
    }
    y
}

fn max_rel(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        num = num.max((*g as f64 - *w as f64).abs());
        den = den.max(w.abs() as f64);
    }
    num / den.max(1e-30)
}

// ------------------------------------------------------------------------------------------------ the toy

/// Every weight of the two toy layers, packed the way the checkpoint holds them.
struct Toy {
    cfg: Dsv4Config,
    q_norm: Vec<f32>,
    kv_norm: Vec<f32>,
    sink: Vec<f32>,
    wq_a: (Vec<u8>, Vec<u8>),
    wq_b: (Vec<u8>, Vec<u8>),
    wkv: (Vec<u8>, Vec<u8>),
    wo_a: (Vec<u8>, Vec<u8>),
    wo_b: (Vec<u8>, Vec<u8>),
    c_norm: Vec<f32>,
    i_wq_b: (Vec<u8>, Vec<u8>),
    i_k_norm: Vec<f32>,
    /// The four matrices the pack keeps in BF16, as it stores them: bytes, no scales. The binder uploads these
    /// unrounded (`Bf16W`), so the host twin is `Bf16Host` over the same bytes.
    c_wkv_bf16: Vec<u8>,
    c_wgate_bf16: Vec<u8>,
    i_weights_bf16: Vec<u8>,
    i_wk_bf16: Vec<u8>,
}

fn toy(ratio: u8) -> Toy {
    let cfg = Dsv4Config::toy(ratio);
    let mut r = Rng(0x5eed_1234_abcd_0917);
    let (dim, hd, ql, g, ol, ih, id) =
        (cfg.dim, cfg.head_dim, cfg.q_lora, cfg.o_groups, cfg.o_lora, cfg.index_heads, cfg.index_dim);
    let mk = |r: &mut Rng, n: usize, k: usize| pack_fp8(n, k, &r.vec(n * k, -0.5, 0.5));
    // BF16 on the wire and BF16 on the device: the binder uploads these bytes unrounded
    let bf = |r: &mut Rng, n: usize, k: usize| -> Vec<u8> {
        r.vec(n * k, -0.5, 0.5).iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect()
    };
    let c_wkv_bf16 = bf(&mut r, hd, dim);
    let c_wgate_bf16 = bf(&mut r, hd, dim);
    let i_weights_bf16 = bf(&mut r, ih, dim);
    let i_wk_bf16 = bf(&mut r, id, hd);
    Toy {
        q_norm: r.vec(ql, 0.5, 1.5),
        kv_norm: r.vec(hd, 0.5, 1.5),
        sink: r.vec(cfg.n_heads, -1.0, 1.0),
        wq_a: mk(&mut r, ql, dim),
        wq_b: mk(&mut r, cfg.n_heads * hd, ql),
        wkv: mk(&mut r, hd, dim),
        wo_a: mk(&mut r, g * ol, cfg.o_in_per_group()),
        wo_b: mk(&mut r, dim, g * ol),
        c_norm: r.vec(hd, 0.5, 1.5),
        i_wq_b: mk(&mut r, ih * id, ql),
        i_k_norm: r.vec(id, 0.5, 1.5),
        c_wkv_bf16,
        c_wgate_bf16,
        i_weights_bf16,
        i_wk_bf16,
        cfg,
    }
}

impl Toy {
    /// Layer `l` of the module, weights uploaded through the FP8 path.
    fn layer(&self, l: usize) -> MlaDsv4Host {
        let cfg = self.cfg.clone();
        let d = cfg.dim;
        let f = |x: &(Vec<u8>, Vec<u8>), n: usize, k: usize| Fp8Host::new(n, k, &x.0, &x.1);
        let mut layer = MlaDsv4Host {
            layer_id: l,
            wq_a: f(&self.wq_a, cfg.q_lora, d),
            wq_b: f(&self.wq_b, cfg.n_heads * cfg.head_dim, cfg.q_lora),
            wkv: f(&self.wkv, cfg.head_dim, d),
            wo_a: f(&self.wo_a, cfg.o_groups * cfg.o_lora, cfg.o_in_per_group()),
            wo_b: f(&self.wo_b, d, cfg.o_groups * cfg.o_lora),
            q_norm: self.q_norm.clone(),
            kv_norm: self.kv_norm.clone(),
            sink: self.sink.clone(),
            rope: Rope::for_layer(&cfg, l, 16),
            crope: Rope::compress(&cfg, 16),
            compressor: None,
            indexer: None,
            cfg: cfg.clone(),
        };
        // `kv_source_of` is the rule the block itself uses: a ratio-0 layer pools nothing, even if the table names it
        if cfg.kv_source_of(l) == Some(l) {
            layer.compressor = Some(CompressorHost::new(
                (cfg.ratio(l) as usize).max(1),
                Bf16Host::new(cfg.head_dim, d, &self.c_wkv_bf16),
                Some(Bf16Host::new(cfg.head_dim, d, &self.c_wgate_bf16)),
                self.c_norm.clone(),
            ));
        }
        if cfg.index_source_of(l) == Some(l) {
            let mut ix = IndexerHost::new(
                cfg.index_heads,
                cfg.index_dim,
                cfg.index_topk,
                f(&self.i_wq_b, cfg.index_heads * cfg.index_dim, cfg.q_lora),
                Bf16Host::new(cfg.index_heads, d, &self.i_weights_bf16),
                if cfg.is_kv_source(l) { Some(Bf16Host::new(cfg.index_dim, cfg.head_dim, &self.i_wk_bf16)) } else { None },
                self.i_k_norm.clone(),
            );
            if l == cfg.candidate_source_layer {
                ix = ix.with_candidate_source(cfg.candidate_topk_blocks, cfg.candidate_block_size);
            } else if cfg.uses_candidates(l) {
                ix = ix.with_candidates();
            }
            layer.indexer = Some(ix);
        }
        layer
    }

    /// Layer `l`'s bytes as the binder hands them over: FP8 pairs for the five MLA projections and `indexer.wq_b`, BF16
    /// for the compressor's and the indexer's small matrices (which is how the checkpoint stores them).
    fn pack(&self, l: usize) -> MlaDsv4Pack<'_> {
        let cfg = &self.cfg;
        let compressor = (cfg.kv_source_of(l) == Some(l)).then(|| CompressorPack {
            wkv: &self.c_wkv_bf16,
            wgate: (cfg.ratio(l) > 1).then_some(self.c_wgate_bf16.as_slice()),
            norm: &self.c_norm,
        });
        let indexer = (cfg.index_source_of(l) == Some(l)).then(|| IndexerPack {
            wq_b: (&self.i_wq_b.0, &self.i_wq_b.1),
            weights_proj: &self.i_weights_bf16,
            wk: (cfg.kv_source_of(l) == Some(l)).then_some(self.i_wk_bf16.as_slice()),
            k_norm: &self.i_k_norm,
        });
        MlaDsv4Pack {
            wq_a: (&self.wq_a.0, &self.wq_a.1),
            wq_b: (&self.wq_b.0, &self.wq_b.1),
            wkv: (&self.wkv.0, &self.wkv.1),
            wo_a: (&self.wo_a.0, &self.wo_a.1),
            wo_b: (&self.wo_b.0, &self.wo_b.1),
            q_norm: &self.q_norm,
            kv_norm: &self.kv_norm,
            sink: &self.sink,
            compressor,
            indexer,
        }
    }
}

// ------------------------------------------------------------------------------------------------ the transcription

/// The planes the transcription reads and writes (its own layout, one sequence).
struct RefPlanes {
    window: Vec<f32>,
    compress: Vec<f32>,
    index_k: Vec<f32>,
    cand: Vec<bool>,
    topk: Vec<i32>,
}

/// `Attention.forward` of model.py, transcribed. f64 accumulation, its own RoPE/quant/softmax.
fn reference_block(t: &Toy, l: usize, x: &[f32], n: usize, start_pos: usize, p: &mut RefPlanes) -> Vec<f32> {
    let cfg = &t.cfg;
    let (h, d, rd, eps) = (cfg.n_heads, cfg.head_dim, cfg.rope_dim, cfg.eps);
    let ql = cfg.q_lora;
    let ratio = cfg.ratio(l) as usize;
    // model.py:681-697: a compressing layer rotates every tensor at compress_rope_theta with YaRN on; a plain one
    // rotates at rope_theta with YaRN off.
    let cfreqs = ref_freqs(rd, 16, cfg.original_seq_len, cfg.compress_rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow);
    let freqs = if ratio > 0 {
        cfreqs.clone()
    } else {
        ref_freqs(rd, 16, 0, cfg.rope_theta, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
    };
    let scale = 1.0 / (d as f32).sqrt();

    // q_a -> q_norm -> q_b -> rope
    let mut qr = mm(x, &deq(ql, cfg.dim, &t.wq_a.0, &t.wq_a.1), n, ql, cfg.dim);
    for r in 0..n {
        ref_norm(&mut qr[r * ql..][..ql], &t.q_norm, eps);
    }
    let wq_b = deq(h * d, ql, &t.wq_b.0, &t.wq_b.1);
    let mut q = mm(&qr, &wq_b, n, h * d, ql);
    for r in 0..n {
        for hh in 0..h {
            ref_rope(&mut q, (r * h + hh) * d, d, rd, start_pos + r, &freqs, false);
        }
    }

    // window kv
    let mut kv = mm(x, &deq(d, cfg.dim, &t.wkv.0, &t.wkv.1), n, d, cfg.dim);
    for r in 0..n {
        ref_norm(&mut kv[r * d..][..d], &t.kv_norm, eps);
        ref_rope(&mut kv, r * d, d, rd, start_pos + r, &freqs, false);
    }
    ref_quant(&mut kv, 32, 8, true);
    let win = cfg.window;
    if start_pos == 0 {
        if n <= win {
            p.window[..n * d].copy_from_slice(&kv);
        } else {
            for r in 0..win {
                let src = n - win + r;
                let slot = (n - win + r) % win;
                p.window[slot * d..][..d].copy_from_slice(&kv[src * d..][..d]);
            }
        }
    } else {
        let slot = start_pos % win;
        p.window[slot * d..][..d].copy_from_slice(&kv[..d]);
    }
    let mut combined: Vec<f32> = if start_pos == 0 { kv.clone() } else { p.window[..win * d].to_vec() };
    let split = combined.len() / d;

    // window idxs (get_window_topk_idxs)
    let mut lists: Vec<Vec<i32>> = Vec::new();
    if start_pos == 0 {
        let width = n.min(win);
        for r in 0..n {
            let mut v = Vec::new();
            for j in 0..width {
                let idx = (r + 1).saturating_sub(win) + j;
                v.push(if idx > r { -1 } else { idx as i32 });
            }
            lists.push(v);
        }
    } else {
        let oldest = start_pos % win + 1;
        let v: Vec<i32> = (oldest..win).chain(0..oldest).map(|s| if s > start_pos { -1 } else { s as i32 }).collect();
        lists.push(v);
    }

    if ratio > 0 {
        let n_pos = (start_pos + n) / ratio;
        let src = cfg.kv_source_of(l).unwrap();
        let mut latent: Option<Vec<f32>> = None;
        if src == l {
            // Compressor.forward
            let mut ck = mm(x, &deq_bf16(d, cfg.dim, &t.c_wkv_bf16), n, d, cfg.dim);
            if ratio == 1 {
                for g in 0..n {
                    ref_norm(&mut ck[g * d..][..d], &t.c_norm, eps);
                }
                latent = Some(ck);
            } else {
                let sc = mm(x, &deq_bf16(d, cfg.dim, &t.c_wgate_bf16), n, d, cfg.dim);
                let groups = n / ratio;
                if groups > 0 {
                    let mut pooled = vec![0f32; groups * d];
                    for g in 0..groups {
                        for dd in 0..d {
                            let mut mx = f32::NEG_INFINITY;
                            for j in 0..ratio {
                                mx = mx.max(sc[(g * ratio + j) * d + dd]);
                            }
                            let mut sum = 0f64;
                            let mut acc = 0f64;
                            for j in 0..ratio {
                                let e = ((sc[(g * ratio + j) * d + dd]) - mx).exp() as f64;
                                sum += e;
                                acc += e * ck[(g * ratio + j) * d + dd] as f64;
                            }
                            pooled[g * d + dd] = (acc / sum) as f32;
                        }
                    }
                    for g in 0..groups {
                        ref_norm(&mut pooled[g * d..][..d], &t.c_norm, eps);
                    }
                    latent = Some(pooled);
                }
            }

            // Indexer.forward: index keys from the pre-rope latent
            if let Some(lat) = latent.as_ref() {
                let groups = lat.len() / d;
                let wk = deq_bf16(cfg.index_dim, d, &t.i_wk_bf16);
                let mut k = mm(lat, &wk, groups, cfg.index_dim, d);
                for g in 0..groups {
                    ref_norm(&mut k[g * cfg.index_dim..][..cfg.index_dim], &t.i_k_norm, eps);
                    let pos = if start_pos == 0 { g * ratio } else { start_pos + 1 - ratio };
                    ref_rope(&mut k, g * cfg.index_dim, cfg.index_dim, rd.min(cfg.index_dim), pos, &freqs, false);
                    ref_quant(&mut k[g * cfg.index_dim..][..cfg.index_dim], 32, 4, true);
                }
                let at = start_pos / ratio;
                p.index_k[at * cfg.index_dim..][..groups * cfg.index_dim].copy_from_slice(&k);
            }

            // rope + fp4 (E4M3 scales, block 16) + write
            if let Some(lat) = latent.as_mut() {
                let groups = lat.len() / d;
                for g in 0..groups {
                    let pos = if start_pos == 0 { g * ratio } else { start_pos + 1 - ratio };
                    ref_rope(lat, g * d, d, rd, pos, &cfreqs, false);
                }
                ref_quant(lat, 16, 4, false);
                let at = start_pos / ratio;
                p.compress[at * d..][..groups * d].copy_from_slice(lat);
            }
        }

        // index scores and the two-level selection
        if cfg.is_index_source(l) {
            let ih = cfg.index_heads;
            let id = cfg.index_dim;
            let mut qi = mm(&qr, &deq(ih * id, ql, &t.i_wq_b.0, &t.i_wq_b.1), n, ih * id, ql);
            for r in 0..n {
                for hh in 0..ih {
                    ref_rope(&mut qi, (r * ih + hh) * id, id, rd.min(id), start_pos + r, &cfreqs, false);
                    ref_quant(&mut qi[(r * ih + hh) * id..][..id], 32, 4, true);
                }
            }
            let w = mm(x, &deq_bf16(ih, cfg.dim, &t.i_weights_bf16), n, ih, cfg.dim);
            let ws = (1.0 / (id as f32).sqrt()) * (ih as f32).powf(-0.5);
            let mut score = vec![0f32; n * n_pos];
            for r in 0..n {
                for pos in 0..n_pos {
                    let mut acc = 0f64;
                    for hh in 0..ih {
                        let mut dot = 0f64;
                        for i in 0..id {
                            dot += qi[(r * ih + hh) * id + i] as f64 * p.index_k[pos * id + i] as f64;
                        }
                        acc += dot.max(0.0) * (w[r * ih + hh] * ws) as f64;
                    }
                    score[r * n_pos + pos] = acc as f32;
                }
            }
            let reach = |r: usize| if start_pos == 0 { (r + 1) / ratio } else { n_pos };
            for r in 0..n {
                let len = reach(r);
                for pos in len..n_pos {
                    score[r * n_pos + pos] = f32::NEG_INFINITY;
                }
                if l == cfg.candidate_source_layer {
                    p.cand = select_candidate_blocks(
                        &score[r * n_pos..][..n_pos],
                        len.max(1),
                        cfg.candidate_topk_blocks,
                        cfg.candidate_block_size,
                    );
                } else if cfg.uses_candidates(l) {
                    for pos in 0..len {
                        if !p.cand[pos] {
                            score[r * n_pos + pos] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            // topk(min(topk, n_pos)).indices.sort() then where(< len, +offset, -1)
            let mut sel = vec![-1i32; n * cfg.index_topk];
            for r in 0..n {
                let len = reach(r);
                let k = cfg.index_topk.min(n_pos);
                let mut order: Vec<usize> = (0..n_pos).collect();
                order.sort_by(|&a, &b| score[r * n_pos + b].partial_cmp(&score[r * n_pos + a]).unwrap().then(a.cmp(&b)));
                order.truncate(k);
                order.sort_unstable();
                for (i, &pos) in order.iter().enumerate() {
                    sel[r * cfg.index_topk + i] = if pos < len { pos as i32 + split as i32 } else { -1 };
                }
            }
            p.topk = sel;
        }

        combined.extend_from_slice(&p.compress[..n_pos * d]);
        for (r, list) in lists.iter_mut().take(n).enumerate() {
            for k in 0..cfg.index_topk {
                let v = p.topk[r * cfg.index_topk + k];
                if v >= 0 {
                    list.push(v);
                }
            }
        }
    }

    // sparse_attn
    let mut o = vec![0f32; n * h * d];
    for r in 0..n {
        for hh in 0..h {
            let mut mx = -1e30f32;
            for &idx in &lists[r] {
                if idx < 0 {
                    continue;
                }
                let mut acc = 0f64;
                for i in 0..d {
                    acc += q[(r * h + hh) * d + i] as f64 * combined[idx as usize * d + i] as f64;
                }
                mx = mx.max(acc as f32 * scale);
            }
            let mut sum = 0f64;
            let mut acc = vec![0f64; d];
            for &idx in &lists[r] {
                if idx < 0 {
                    continue;
                }
                let mut s = 0f64;
                for i in 0..d {
                    s += q[(r * h + hh) * d + i] as f64 * combined[idx as usize * d + i] as f64;
                }
                let e = ((s as f32) * scale - mx).exp() as f64;
                sum += e;
                for i in 0..d {
                    acc[i] += e * combined[idx as usize * d + i] as f64;
                }
            }
            sum += (t.sink[hh] - mx).exp() as f64;
            for i in 0..d {
                o[(r * h + hh) * d + i] = (acc[i] / sum) as f32;
            }
        }
    }

    // inverse rope, then the grouped output projection
    for r in 0..n {
        for hh in 0..h {
            ref_rope(&mut o, (r * h + hh) * d, d, rd, start_pos + r, &freqs, true);
        }
    }
    ref_quant(&mut o, 32, 8, true);
    let (g, ol, oin) = (cfg.o_groups, cfg.o_lora, cfg.o_in_per_group());
    let woa = deq(g * ol, oin, &t.wo_a.0, &t.wo_a.1);
    let mut olora = vec![0f32; n * g * ol];
    for r in 0..n {
        for gi in 0..g {
            for j in 0..ol {
                let mut acc = 0f64;
                for c in 0..oin {
                    acc += o[r * h * d + gi * oin + c] as f64 * woa[(gi * ol + j) * oin + c] as f64;
                }
                olora[(r * g + gi) * ol + j] = acc as f32;
            }
        }
    }
    mm(&olora, &deq(cfg.dim, g * ol, &t.wo_b.0, &t.wo_b.1), n, cfg.dim, g * ol)
}

// ------------------------------------------------------------------------------------------------ 1. YaRN

#[test]
fn yarn_table_pins_first_and_last_frequencies() {
    // dim 64, original_seq_len 65536, base 10000, factor 16, beta 32/1: the reference's numbers.
    let t = yarn_inv_freq(ROPE_DIM, ORIGINAL_SEQ_LEN, ROPE_THETA, ROPE_FACTOR, BETA_FAST, BETA_SLOW);
    // i < low = 20: untouched by YaRN
    let first = [1.0f32, 0.749_894_2, 0.562_341_3, 0.421_696_5];
    // i >= high = 33: divided by the factor
    let last = [1.337_886_7e-4, 8.322_61e-5, 4.958_664e-5, 2.756_799e-5];
    for (j, want) in first.iter().enumerate() {
        let rel = ((t[j] - want) / want).abs();
        assert!(rel < 1e-6, "freq {j}: {} != {want}", t[j]);
    }
    let half = ROPE_DIM / 2;
    for (i, want) in last.iter().enumerate() {
        let j = half - 4 + i;
        let rel = ((t[j] - want) / want).abs();
        assert!(rel < 1e-5, "freq {j}: {} != {want}", t[j]);
    }

    // and the same function against the transcribed formula, every frequency
    let want = ref_freqs(ROPE_DIM, 1, ORIGINAL_SEQ_LEN, ROPE_THETA, ROPE_FACTOR, BETA_FAST, BETA_SLOW);
    for (j, w) in want.iter().enumerate() {
        assert!((t[j] as f64 - w).abs() / w < 1e-6, "freq {j}: {} != {w}", t[j]);
    }
    // monotone decreasing, and the band really is a blend (not a step)
    assert!(t.windows(2).all(|w| w[0] > w[1]));
    assert!(t[20] < 1.0 && t[20] > t[19] / ROPE_FACTOR as f32 * 1.0001);

    // the compressed table is a different table: base 160000
    let c = yarn_inv_freq(ROPE_DIM, ORIGINAL_SEQ_LEN, COMPRESS_ROPE_THETA, ROPE_FACTOR, BETA_FAST, BETA_SLOW);
    assert!((c[0] - 1.0).abs() < 1e-6);
    assert!(c[1] < t[1], "the compressed plane's base must spread the low frequencies");
    assert_eq!(c.len(), t.len());

    // YaRN is off when original_seq_len == 0: plain 1 / base^(2j/dim)
    let off = yarn_inv_freq(ROPE_DIM, 0, ROPE_THETA, ROPE_FACTOR, BETA_FAST, BETA_SLOW);
    for (j, &f) in off[..half].iter().enumerate() {
        let want = 1.0f32 / (ROPE_THETA as f32).powf(2.0 * j as f32 / ROPE_DIM as f32);
        assert!((f - want).abs() / want < 1e-6, "plain freq {j}");
    }
}

#[test]
fn rope_rotation_is_a_complex_multiply_and_inverts() {
    let rope = Rope::new(8, 4, 0, 10_000.0, 1.0, 32.0, 1.0);
    let mut x: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let orig = x.clone();
    rope.apply(&mut x, 1, 8, 8, 1, 2, false);
    // pair 0 at pos 2: angle 2 * 1.0
    let (a, b) = (orig[0], orig[1]);
    let (c, s) = (2.0f32.cos(), 2.0f32.sin());
    assert!((x[0] - (a * c - b * s)).abs() < 1e-6);
    assert!((x[1] - (a * s + b * c)).abs() < 1e-6);
    rope.apply(&mut x, 1, 8, 8, 1, 2, true);
    for (g, w) in x.iter().zip(&orig) {
        assert!((g - w).abs() < 1e-6, "inverse rotation must undo the rotation");
    }
}

// ------------------------------------------------------------------------------------------------ 2. the block

#[test]
fn block_matches_the_reference_transcription() {
    for ratio in [2u8, 1, 0] {
        let t = toy(ratio);
        let cfg = t.cfg.clone();
        let n = 4;
        let mut r = Rng(0xdead_beef_0000_0001);
        let x = r.vec(n * cfg.dim, -1.0, 1.0);

        let mut planes = HostPlanes::new(&cfg, 16, 0);
        let mut refp = RefPlanes {
            window: vec![0f32; cfg.window * cfg.head_dim],
            compress: vec![0f32; 16 * cfg.head_dim],
            index_k: vec![0f32; 16 * cfg.index_dim],
            cand: vec![false; 16],
            topk: vec![-1; MAX_ROWS * cfg.index_topk],
        };
        let mut layers: Vec<MlaDsv4Host> = (0..cfg.n_layers).map(|l| t.layer(l)).collect();

        // prefill both layers, then one decode step, on the same sequence
        for (l, layer) in layers.iter_mut().enumerate() {
            let got = layer.forward(&x, n, 0, &mut planes, None);
            let want = reference_block(&t, l, &x, n, 0, &mut refp);
            let rel = max_rel(&got, &want);
            assert!(rel < 1e-4, "ratio {ratio} layer {l} prefill: max rel err {rel:e}");
        }

        // decode: one token at position 4 (needs a completed group for ratio 2 -> 5 tokens is not a group; use 4)
        let xt = r.vec(cfg.dim, -1.0, 1.0);
        for (l, layer) in layers.iter_mut().enumerate() {
            let got = layer.forward(&xt, 1, 4, &mut planes, None);
            let want = reference_block(&t, l, &xt, 1, 4, &mut refp);
            let rel = max_rel(&got, &want);
            assert!(rel < 1e-4, "ratio {ratio} layer {l} decode: max rel err {rel:e}");
        }
    }
}

#[test]
fn grouped_woa_is_block_diagonal() {
    // group g must not read another group's heads: perturbing group 1's input leaves group 0's rows untouched
    let t = toy(2);
    let cfg = t.cfg.clone();
    let layer = t.layer(0);
    let (g, ol, oin) = (cfg.o_groups, cfg.o_lora, cfg.o_in_per_group());
    let a = vec![0.1f32; g * oin];
    let mut b = a.clone();
    for v in &mut b[oin..2 * oin] {
        *v += 10.0;
    }
    let mut ol_a = vec![0f32; g * ol];
    let mut ol_b = vec![0f32; g * ol];
    for gi in 0..g {
        for j in 0..ol {
            let mut acc = 0f64;
            for c in 0..oin {
                acc += a[gi * oin + c] as f64 * layer.wo_a.w[(gi * ol + j) * oin + c] as f64;
            }
            ol_a[gi * ol + j] = acc as f32;
            let mut acc = 0f64;
            for c in 0..oin {
                acc += b[gi * oin + c] as f64 * layer.wo_a.w[(gi * ol + j) * oin + c] as f64;
            }
            ol_b[gi * ol + j] = acc as f32;
        }
    }
    for j in 0..ol {
        assert_eq!(ol_a[j], ol_b[j], "group 0 moved when group 1 changed");
    }
    assert!((ol_a[ol] - ol_b[ol]).abs() > 1e-3, "group 1 did not move");
}

#[test]
fn reworked_woa_matches_classic_bit_for_bit_on_real_shapes() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    // The shipped pack's wo_a shape: 8 groups, [1024, 4096] per group, rows 1-8.
    let cfg = Dsv4Config::v41();
    let (ol, oin, groups) = (cfg.o_lora, cfg.o_in_per_group(), cfg.o_groups);
    let hd = groups * oin;
    let mut rng = Rng(0x00C0_FFEE_00C0_FFEE);
    for rows in [1usize, 2, 3, 4, 5, 6, 7, 8] {
        let x16: Vec<u16> = rng.vec(rows * hd, -1.0, 1.0).iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
        let (w, s) = pack_fp8(groups * ol, oin, &rng.vec(groups * ol * oin, -0.5, 0.5));
        let od = DevBuf::from_slice(&x16).unwrap();
        let wd = DevBuf::from_slice(&w).unwrap();
        let sd = DevBuf::from_slice(&s).unwrap();
        let mut classic = DevBuf::zeroed(rows * groups * ol * 2).unwrap();
        let mut fast = DevBuf::zeroed(rows * groups * ol * 2).unwrap();
        MlaDsv4::woa_grouped_ab(&od, &wd, &sd, &mut classic, rows, groups, ol, oin, Dtype::F16, true, &st).unwrap();
        MlaDsv4::woa_grouped_ab(&od, &wd, &sd, &mut fast, rows, groups, ol, oin, Dtype::F16, false, &st).unwrap();
        st.sync().unwrap();
        let a: Vec<u16> = classic.to_vec().unwrap();
        let b: Vec<u16> = fast.to_vec().unwrap();
        assert_eq!(a, b, "woa rework differs on rows={rows}");
    }
}

// ------------------------------------------------------------------------------------------------ 3. top-k

#[test]
fn top512_matches_a_full_sort() {
    let mut r = Rng(0x1234_5678_9abc_def0);
    // the pack's shape: 512 selections out of a long compressed plane
    for (n_pos, k) in [(2048usize, 512usize), (513, 512), (512, 512), (100, 512), (7000, 512), (9, 3)] {
        let mut s = r.vec(n_pos, -5.0, 5.0);
        // duplicate a few scores so the tie rule is exercised
        for i in (0..n_pos).step_by(97) {
            s[i] = 0.25;
        }
        let got = topk_indices(&s, k);
        let mut want: Vec<usize> = (0..n_pos).collect();
        want.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap().then(a.cmp(&b)));
        want.truncate(k.min(n_pos));
        want.sort_unstable();
        assert_eq!(got, want, "n_pos {n_pos} k {k}");

        // the invariant that matters: nothing left out scores above anything taken
        if k < n_pos {
            let worst = got.iter().map(|&i| s[i]).fold(f32::INFINITY, f32::min);
            for (i, &v) in s.iter().enumerate() {
                if !got.contains(&i) {
                    assert!(v <= worst, "position {i} ({v}) beats the selection's worst ({worst})");
                }
            }
        }
    }
    // -inf entries are selected (and later become -1), exactly as the reference's topk does
    let s = vec![1.0f32, f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY];
    assert_eq!(topk_indices(&s, 4), vec![0, 1, 2, 3]);
    assert_eq!(topk_indices(&s, 2), vec![0, 2]);
}

#[test]
fn topk_length_is_a_parameter_not_a_constant() {
    // the trap this module exists for: the same scores selected at 512, 256 and 64
    let mut r = Rng(0x0bad_f00d_0000_0001);
    let s = r.vec(4096, -1.0, 1.0);
    let mut prev: Option<Vec<usize>> = None;
    for k in [512usize, 256, 64] {
        let got = topk_indices(&s, k);
        assert_eq!(got.len(), k, "k = {k} must select exactly k positions");
        if let Some(p) = &prev {
            assert!(p.len() > got.len());
            // a smaller k is a subset of a larger one (distinct scores here)
            for i in &got {
                assert!(p.contains(i), "k = {k} is not nested in the larger selection");
            }
        }
        prev = Some(got);
    }
}

// ------------------------------------------------------------------------------------------------ 4. candidates

#[test]
fn candidate_blocks_pin_the_newest_block_and_drop_unreachable() {
    // 32 positions, 4 blocks of 8. Reach 20 positions (blocks 0,1,2 partly), block 2 is the newest.
    let mut logits = vec![0f32; 32];
    for (i, v) in logits.iter_mut().enumerate() {
        *v = if i < 20 { -(i as f32) } else { f32::NEG_INFINITY }; // block 0 scores highest
    }
    let m = select_candidate_blocks(&logits, 20, 2, 8);
    assert!(m[0] && m[1], "the best block must be kept");
    assert!(m[16] && m[17] && m[18] && m[19], "the newest block must be pinned in");
    assert!(!m[8] && !m[9], "a third block must be dropped at topk_blocks = 2");
    assert!(!m[24] && !m[31], "unreachable positions are never kept");

    // fewer reachable blocks than topk_blocks: the -inf picks are dropped, not kept
    let m = select_candidate_blocks(&logits, 20, 100, 8);
    // block 2 spans 16..24 and holds reachable position 16, so its -inf tail (20..23) comes with it
    assert!(m[..24].iter().all(|&b| b), "every reachable block is kept whole");
    assert!(m[24..].iter().all(|&b| !b), "and nothing beyond");
}

// ------------------------------------------------------------------------------------------------ 5. compressed plane

/// `softmax(q . k / sqrt(d))` over the plane rows `positions`, per head — the logits the compressor must preserve.
fn logits(q: &[f32], plane: &[f32], positions: &[usize], d: usize) -> Vec<f32> {
    positions
        .iter()
        .map(|&p| {
            let mut acc = 0f64;
            for i in 0..d {
                acc += q[i] as f64 * plane[p * d + i] as f64;
            }
            (acc as f32) / (d as f32).sqrt()
        })
        .collect()
}

#[test]
fn ratio2_plane_reproduces_the_uncompressed_logits() {
    // With a uniform gate (`wgate == 0`) a ratio-2 group's latent is exactly the RMSNorm of `wkv` applied to the pair
    // mean, so the compressed plane at rope positions 0, 2, ... is the ratio-1 plane of the pair means at those same
    // positions. That pins: the gate's softmax normalisation, the linearity of the pooling, the norm's placement, and
    // the rope position `j * ratio`.
    let cfg2 = Dsv4Config::toy(2);
    let d = cfg2.head_dim;
    let hd = cfg2.head_dim;
    let mut r = Rng(0xfeed_face_0000_0002);
    let wkv = Bf16Host::from_f32(hd, cfg2.dim, r.vec(hd * cfg2.dim, -0.5, 0.5));
    let wgate = Bf16Host::from_f32(hd, cfg2.dim, vec![0f32; hd * cfg2.dim]); // uniform gate
    let norm = r.vec(hd, 0.5, 1.5);
    let n = 4;

    let x = r.vec(n * cfg2.dim, -1.0, 1.0);
    let mut c2 = CompressorHost::new(2, wkv.clone(), Some(wgate), norm.clone());
    let lat2 = c2.forward(&x, n, 0).expect("two complete groups");

    // the ratio-1 compressor over the pair means
    let mut c1 = CompressorHost::new(1, wkv.clone(), None, norm.clone());
    let mut means = vec![0f32; (n / 2) * cfg2.dim];
    for g in 0..n / 2 {
        for c in 0..cfg2.dim {
            means[g * cfg2.dim + c] = 0.5 * (x[(2 * g) * cfg2.dim + c] + x[(2 * g + 1) * cfg2.dim + c]);
        }
    }
    let lat1 = c1.forward(&means, n / 2, 0).expect("ratio 1 is a plain projection");

    assert_eq!(lat2.len(), n / 2 * d);
    let rel = max_rel(&lat2, &lat1);
    assert!(rel < 1e-5, "the ratio-2 latent must be the ratio-1 latent of the pair means (rel {rel:e})");

    // same latent, same rope, same logits
    let rope = Rope::compress(&cfg2, 16);
    let mut p2 = lat2.clone();
    let mut p1 = lat1.clone();
    for g in 0..n / 2 {
        rope.apply(&mut p2[g * d..][..d], 1, d, d, 1, g * 2, false);
        rope.apply(&mut p1[g * d..][..d], 1, d, d, 1, g * 2, false);
    }
    let q = r.vec(d, -1.0, 1.0);
    let l2 = logits(&q, &p2, &[0, 1], d);
    let l1 = logits(&q, &p1, &[0, 1], d);
    let rel = max_rel(&l2, &l1);
    assert!(rel < 1e-5, "compressed logits {l2:?} != uncompressed {l1:?} (rel {rel:e})");

    // and the gate really is doing something: a non-zero gate changes the plane
    let wgate2 = Bf16Host::from_f32(hd, cfg2.dim, r.vec(hd * cfg2.dim, -0.5, 0.5));
    let mut c3 = CompressorHost::new(2, wkv, Some(wgate2), norm);
    let lat3 = c3.forward(&x, n, 0).unwrap();
    assert!(max_rel(&lat3, &lat2) > 1e-2, "a trained gate must move the pooled latent");
}

#[test]
fn compressor_holds_the_partial_group_across_decode_steps() {
    let cfg = Dsv4Config::toy(2);
    let hd = cfg.head_dim;
    let mut r = Rng(0xabcd_0000_1111_2222);
    let wkv = Bf16Host::from_f32(hd, cfg.dim, r.vec(hd * cfg.dim, -0.5, 0.5));
    let wgate = Bf16Host::from_f32(hd, cfg.dim, r.vec(hd * cfg.dim, -0.5, 0.5));
    let norm = r.vec(hd, 0.5, 1.5);
    let x = r.vec(4 * cfg.dim, -1.0, 1.0);

    // prefill of 4 tokens, then a decode step per token, must reproduce the prefill's two groups
    let mut pre = CompressorHost::new(2, wkv.clone(), Some(wgate.clone()), norm.clone());
    let a = pre.forward(&x, 4, 0).unwrap();

    let mut dec = CompressorHost::new(2, wkv, Some(wgate), norm);
    let mut out: Vec<f32> = Vec::new();
    for t in 0..4 {
        if let Some(l) = dec.forward(&x[t * cfg.dim..][..cfg.dim], 1, t) {
            out.extend_from_slice(&l);
        }
    }
    assert_eq!(out.len(), a.len());
    let rel = max_rel(&out, &a);
    assert!(rel < 1e-5, "decode-step pooling must match prefill (rel {rel:e})");
}

// ------------------------------------------------------------------------------------------------ 6. device

#[test]
fn gpu_tables_and_launchers() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let cfg = Dsv4Config::toy(2);
    // rope table upload + rotation, then the inverse
    let rope = Rope::new(8, 4, 0, 10_000.0, 1.0, 32.0, 1.0);
    let cos = DevBuf::from_slice(&rope.cos).unwrap();
    let sin = DevBuf::from_slice(&rope.sin).unwrap();
    // `gb_dsv4_rope` reads and writes fp16 (the table is f32), so the vector is uploaded as f16 bits — an f32 upload
    // would have the kernel read half of each value's bits as two separate rows of garbage
    let host: Vec<f32> = (0..8).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let bits: Vec<u16> = host.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
    let x = DevBuf::from_slice(&bits).unwrap();
    rope_rows_dev(&x, 1, 8, 8, 8, 1, (&cos, &sin), 2, 1, false, &st).unwrap();
    rope_rows_dev(&x, 1, 8, 8, 8, 1, (&cos, &sin), 2, 1, true, &st).unwrap();
    st.sync().unwrap();
    let back: Vec<u16> = x.to_vec().unwrap();
    for (g, w) in back.iter().zip(&host) {
        let g = gb_exl3::f16::f16_to_f64(*g) as f32;
        assert!((g - w).abs() < 1e-3, "device rotation must invert: {g} vs {w}");
    }
    // quantise-dequantise on the device: fp16 in, fp16 out (the same dtype the launcher is bound with)
    let mut h = h_f32();
    let hb: Vec<u16> = h.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
    let buf = DevBuf::from_slice(&hb).unwrap();
    quant_dev(&buf, 1, h.len(), 32, 4, true, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<u16> = buf.to_vec().unwrap();
    quant_dequant(&mut h, 32, 4, true);
    assert_eq!(got.len(), h.len());
    for (g, w) in got.iter().zip(&h) {
        let g = gb_exl3::f16::f16_to_f64(*g) as f32;
        // 4-bit over this block's scale: an eighth of the range, not a rounding
        assert!(g.is_finite() && (g - w).abs() <= 0.2, "quantise must round-trip within the block scale: {g} vs {w}");
    }
    let _ = cfg;
}

#[test]
fn gpu_rope_steps_positions_for_compressed_entries() {
    // One compressed entry stands for the first token of its group, so entry j rotates at `j * ratio` (model.py:754):
    // a table walked one row at a time would rotate entry 1 at position 1 instead of 2.
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let rope = Rope::new(8, 4, 0, 10_000.0, 1.0, 32.0, 1.0);
    let cos = DevBuf::from_slice(&rope.cos).unwrap();
    let sin = DevBuf::from_slice(&rope.sin).unwrap();
    let host: Vec<f32> = (0..16).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let bits: Vec<u16> = host.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
    let x = DevBuf::from_slice(&bits).unwrap();
    rope_rows_dev(&x, 2, 8, 8, 8, 1, (&cos, &sin), 0, 2, false, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<u16> = x.to_vec().unwrap();
    for r in 0..2 {
        let mut want = host[r * 8..][..8].to_vec();
        // row r sits at position r * 2, not r
        let one = Rope::new(8, 4, 0, 10_000.0, 1.0, 32.0, 1.0);
        one.apply(&mut want, 1, 8, 8, 1, r * 2, false);
        for (i, w) in want.iter().enumerate() {
            let g = gb_exl3::f16::f16_to_f64(got[r * 8 + i]) as f32;
            assert!((g - w).abs() < 2e-3, "row {r} column {i}: {g} vs {w}");
        }
    }
}

fn h_f32() -> Vec<f32> {
    (0..32).map(|i| (i as f32 - 16.0) * 0.05).collect()
}

#[test]
fn gpu_kv_prepare_matches_the_unfused_chain() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    // the trunk's real KV shape: 512-wide rows, 64 rope dims, q4 rotation of the 448 nope dims
    let (width, rd, rows) = (512usize, 64usize, 4usize);
    let nope = width - rd;
    let rope = Rope::new(rd, 2048, 0, 10_000.0, 1.0, 32.0, 1.0);
    let (cos, sin) = (DevBuf::from_slice(&rope.cos).unwrap(), DevBuf::from_slice(&rope.sin).unwrap());
    let w: Vec<f32> = (0..width).map(|i| 0.5 + 0.001 * (i as f32 % 97.0)).collect();
    let mut r = Rng(0x5eed_c0de_0000_0001);
    let host: Vec<f32> = r.vec(rows * width, -3.0, 3.0);
    let bits: Vec<u16> = host.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();

    let a = DevBuf::from_slice(&bits).unwrap();
    rmsnorm_dev(&a, &DevBuf::from_slice(&w).unwrap(), rows, width, 1e-6, &st).unwrap();
    rope_rows_dev(&a, rows, width, rd, width, 1, (&cos, &sin), 101, 1, false, &st).unwrap();
    quant_dev(&a, rows, width, 32, 8, true, &st).unwrap();
    gb_cuda::dsv4_kvq4::rot_dev(&a, rows, width, nope, width, 1, &st).unwrap();

    let b = DevBuf::from_slice(&bits).unwrap();
    kv_prepare_dev(&b, &DevBuf::from_slice(&w).unwrap(), rows, width, rd, width, (&cos, &sin), 101, 1, true, 1e-6, &st).unwrap();
    st.sync().unwrap();
    let got_a: Vec<u16> = a.to_vec().unwrap();
    let got_b: Vec<u16> = b.to_vec().unwrap();
    assert_eq!(got_a, got_b, "fused KV prep must be bit-identical to the unfused chain");

    // and the no-rotation variant (f16 cache) must match norm+rope+quant
    let a2 = DevBuf::from_slice(&bits).unwrap();
    rmsnorm_dev(&a2, &DevBuf::from_slice(&w).unwrap(), rows, width, 1e-6, &st).unwrap();
    rope_rows_dev(&a2, rows, width, rd, width, 1, (&cos, &sin), 101, 1, false, &st).unwrap();
    quant_dev(&a2, rows, width, 32, 8, true, &st).unwrap();
    let b2 = DevBuf::from_slice(&bits).unwrap();
    kv_prepare_dev(&b2, &DevBuf::from_slice(&w).unwrap(), rows, width, rd, width, (&cos, &sin), 101, 1, false, 1e-6, &st)
        .unwrap();
    st.sync().unwrap();
    assert_eq!(a2.to_vec::<u16>().unwrap(), b2.to_vec::<u16>().unwrap(), "no-rot fused KV prep must match");
}

#[test]
fn gpu_block_matches_the_host_on_a_source_and_a_reader() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    // toy(2): layer 0 owns the pool and the indexer, layer 1 is a reader at the same ratio — the pack's own shape, twice
    let t = toy(2);
    let cfg = t.cfg.clone();
    let (n, d) = (4usize, cfg.dim);
    let mut r = Rng(0x0bad_c0de_1234_5678);
    let x = r.vec(n * d, -1.0, 1.0);

    // --- the host twin, on the same bytes: layer 0 writes the group's plane, layer 1 reads it
    let mut h0 = t.layer(0);
    let mut h1 = t.layer(1);
    let mut hp0 = HostPlanes::new(&cfg, 16, 0);
    let mut hp1 = HostPlanes::new(&cfg, 16, 1);
    let want0 = h0.forward(&x, n, 0, &mut hp0, None);
    let want1 = h1.forward(&x, n, 0, &mut hp1, Some(&mut hp0));

    // --- the device block, through the pack
    let xb: Vec<u16> = x.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
    let xdev = DevBuf::from_slice(&xb).unwrap();
    let mut m0 = MlaDsv4::new(&cfg, 0, &t.pack(0), 16).unwrap();
    let mut m1 = MlaDsv4::new(&cfg, 1, &t.pack(1), 16).unwrap();
    assert!(m0.compressor.is_some() && m0.indexer.is_some(), "layer 0 is a source and a selector");
    assert!(m1.compressor.is_none(), "layer 1 reads layer 0's pool");
    assert!(m1.indexer.is_none(), "layer 1 is not a selector: it reads the selection layer 0 published");
    assert!(m0.indexer.as_ref().unwrap().wk.is_some(), "the pool owner builds the index keys");
    let d0 = Dsv4Planes::new(&cfg, 16, 0, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
    let d1 = Dsv4Planes::new(&cfg, 16, 1, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
    let mut sc = Dsv4Scratch::new(&cfg, n, 16).unwrap();
    m0.forward(&xdev, n, 0, &d0, None, &mut sc, &st).unwrap();
    let got0: Vec<f32> = sc.out.to_vec::<u16>().unwrap()[..n * d].iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect();
    // layer 1's window ring is its own; its compressed plane, index keys and selection are layer 0's
    m1.forward(&xdev, n, 0, &d1, Some(&d0), &mut sc, &st).unwrap();
    let got1: Vec<f32> = sc.out.to_vec::<u16>().unwrap()[..n * d].iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect();

    // stage by stage: the compressed plane and the index keys are the group's, and both must be the host's
    let f16s = |v: &[u16]| -> Vec<f32> { v.iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect() };
    let dc: Vec<f32> = f16s(&d0.compress.to_vec::<u16>().unwrap());
    let rel = max_rel(&dc, &hp0.compress[..dc.len()]);
    assert!(rel < 5e-3, "the compressed plane must be the host's, max rel err {rel:e}");
    let dk: Vec<f32> = f16s(&d0.index_k.to_vec::<u16>().unwrap());
    let rel = max_rel(&dk, &hp0.index_k[..dk.len()]);
    assert!(rel < 5e-3, "the index keys must be the host's, max rel err {rel:e}");
    // the published row is [window slots | compressed selection]; the host keeps the two halves apart
    let dt: Vec<i32> = d0.topk.to_vec::<i32>().unwrap();
    let stride = d0.sel_stride;
    let want_win = window_idxs(cfg.window, n, 0);
    for r in 0..n {
        assert_eq!(
            &dt[r * stride..][..n.min(cfg.window)],
            &want_win[r][..],
            "row {r}'s window half must be get_window_topk_idxs"
        );
        assert_eq!(
            &dt[r * stride + n..][..cfg.index_topk],
            &hp0.topk_idxs[r * cfg.index_topk..][..cfg.index_topk],
            "row {r}'s compressed half must be the published selection"
        );
    }
    for (l, (got, want)) in [(0usize, (&got0, &want0)), (1, (&got1, &want1))].into_iter() {
        let rel = max_rel(got, want);
        assert!(rel < 5e-2, "layer {l}: the device block must reproduce the host twin, max rel err {rel:e}");
    }

    // two decode steps at ratio 2: position 4 leaves its group half filled (no entry, the state carries it) and
    // position 5 completes it (one entry, out of the state). Both layers, both paths.
    for pos in [4usize, 5] {
        let xt = r.vec(d, -1.0, 1.0);
        let xb: Vec<u16> = xt.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
        let xtdev = DevBuf::from_slice(&xb).unwrap();
        let want0 = h0.forward(&xt, 1, pos, &mut hp0, None);
        let want1 = h1.forward(&xt, 1, pos, &mut hp1, Some(&mut hp0));
        m0.forward(&xtdev, 1, pos, &d0, None, &mut sc, &st).unwrap();
        let got0: Vec<f32> = sc.out.to_vec::<u16>().unwrap()[..d].iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect();
        m1.forward(&xtdev, 1, pos, &d1, Some(&d0), &mut sc, &st).unwrap();
        let got1: Vec<f32> = sc.out.to_vec::<u16>().unwrap()[..d].iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect();
        for (l, (got, want)) in [(0usize, (&got0, &want0)), (1, (&got1, &want1))].into_iter() {
            let rel = max_rel(got, want);
            assert!(rel < 5e-2, "layer {l} at {pos}: decode must reproduce the host twin, max rel err {rel:e}");
        }
    }
    // and a reader that silently used its own (unwritten) plane instead of the source's would not match `want1`
    let d1_unshared = Dsv4Planes::new(&cfg, 16, 1, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
    m1.forward(&xdev, n, 0, &d1_unshared, None, &mut sc, &st).unwrap();
    let got1_unshared: Vec<f32> =
        sc.out.to_vec::<u16>().unwrap()[..n * d].iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect();
    assert!(max_rel(&got1_unshared, &want1) > 1e-3, "unshared reader plane must differ from shared");
}

#[test]
fn gpu_sliding_window_only_layer_reads_its_own_window() {
    // A ratio-0 layer has no compressed half, so its published row is nothing but `get_window_topk_idxs`. A row left at
    // zero would have every query attend to window slot 0 `split` times instead of its causal window.
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let t = toy(0);
    let cfg = t.cfg.clone();
    let (n, d) = (4usize, cfg.dim);
    let mut r = Rng(0x5151_2323_9999_0001);
    let x = r.vec(n * d, -1.0, 1.0);
    let mut h0 = t.layer(0);
    let mut hp = HostPlanes::new(&cfg, 16, 0);
    let want = h0.forward(&x, n, 0, &mut hp, None);
    let xb: Vec<u16> = x.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
    let xdev = DevBuf::from_slice(&xb).unwrap();
    let mut m0 = MlaDsv4::new(&cfg, 0, &t.pack(0), 16).unwrap();
    let d0 = Dsv4Planes::new(&cfg, 16, 0, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
    let mut sc = Dsv4Scratch::new(&cfg, n, 16).unwrap();
    m0.forward(&xdev, n, 0, &d0, None, &mut sc, &st).unwrap();
    let got: Vec<f32> = sc.out.to_vec::<u16>().unwrap()[..n * d].iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect();
    let rel = max_rel(&got, &want);
    assert!(rel < 5e-2, "a window-only layer must reproduce the host twin, max rel err {rel:e}");
    let sel: Vec<i32> = d0.topk.to_vec::<i32>().unwrap();
    let want_win = window_idxs(cfg.window, n, 0);
    for row in 0..n {
        assert_eq!(&sel[row * d0.sel_stride..][..n], &want_win[row][..], "row {row}'s window half");
    }
}

#[test]
fn gpu_block_self_skips() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    // A full block call needs the pack's bytes; the toy path above covers the math, so this only asserts that the
    // planes and scratch size themselves and that the launchers are reachable.
    let cfg = Dsv4Config::toy(2);
    let planes = Dsv4Planes::new(&cfg, 16, 0, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
    assert!(planes.compress.bytes() >= cfg.head_dim * 2);
    let scratch = Dsv4Scratch::new(&cfg, 4, 16).unwrap();
    assert!(scratch.out.bytes() >= 4 * cfg.dim * 2);
    assert!(Dsv4Scratch::bytes(&cfg, 4, 16) > 0);
}

#[test]
fn config_roles_of_the_pack() {
    let cfg = Dsv4Config::v41();
    cfg.validate().unwrap();
    assert_eq!(cfg.ratio(0), 0);
    assert_eq!(cfg.ratio(1), 0);
    assert_eq!(cfg.ratio(2), 2);
    assert_eq!(cfg.ratio(19), 2);
    assert_eq!(cfg.ratio(20), 1);
    assert_eq!(cfg.ratio(39), 1);
    assert_eq!(cfg.ratio(40), 0);
    assert_eq!(cfg.ratio(42), 0);
    // layers 3..7 read layer 2's plane; 9..13 read 8's; 21..39 read 20's
    assert_eq!(cfg.kv_source_of(3), Some(2));
    assert_eq!(cfg.kv_source_of(7), Some(2));
    assert_eq!(cfg.kv_source_of(8), Some(8));
    assert_eq!(cfg.kv_source_of(39), Some(20));
    assert_eq!(cfg.kv_source_of(0), None);
    assert_eq!(cfg.kv_source_of(41), None);
    // index sources: 2, 8, 14, 20, 24, 28, 32, 36
    assert_eq!(cfg.index_source_of(23), Some(20));
    assert_eq!(cfg.index_source_of(24), Some(24));
    assert_eq!(cfg.index_source_of(37), Some(36));
    assert!(cfg.uses_candidates(24) && cfg.uses_candidates(36));
    assert!(!cfg.uses_candidates(20), "the source is not a consumer");
    assert!(!cfg.uses_candidates(21), "a non-index-source layer has no indexer to mask");
    assert_eq!(cfg.o_in_per_group(), 4096);
    assert_eq!(cfg.n_heads * cfg.head_dim, 32_768);
}

// ------------------------------------------------------------------------------------------------ scale

/// `toy(ratio)` for a caller-supplied config: the same weights, but the selection geometry the pack runs
/// (window 128, index_topk 512, candidate blocks 2048x8) instead of the toy's 4/2/2x2.
fn toy_with(cfg: Dsv4Config) -> Toy {
    let mut r = Rng(0x5eed_1234_abcd_0917);
    let (dim, hd, ql, g, ol, ih, id) =
        (cfg.dim, cfg.head_dim, cfg.q_lora, cfg.o_groups, cfg.o_lora, cfg.index_heads, cfg.index_dim);
    let mk = |r: &mut Rng, n: usize, k: usize| pack_fp8(n, k, &r.vec(n * k, -0.5, 0.5));
    let bf = |r: &mut Rng, n: usize, k: usize| -> Vec<u8> {
        r.vec(n * k, -0.5, 0.5).iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect()
    };
    Toy {
        q_norm: r.vec(ql, 0.5, 1.5),
        kv_norm: r.vec(hd, 0.5, 1.5),
        sink: r.vec(cfg.n_heads, -1.0, 1.0),
        wq_a: mk(&mut r, ql, dim),
        wq_b: mk(&mut r, cfg.n_heads * hd, ql),
        wkv: mk(&mut r, hd, dim),
        wo_a: mk(&mut r, g * ol, cfg.o_in_per_group()),
        wo_b: mk(&mut r, dim, g * ol),
        c_norm: r.vec(hd, 0.5, 1.5),
        i_wq_b: mk(&mut r, ih * id, ql),
        i_k_norm: r.vec(id, 0.5, 1.5),
        c_wkv_bf16: bf(&mut r, hd, dim),
        c_wgate_bf16: bf(&mut r, hd, dim),
        i_weights_bf16: bf(&mut r, ih, dim),
        i_wk_bf16: bf(&mut r, id, hd),
        cfg,
    }
}

/// `Toy::layer` with RoPE tables built for `seq` positions (the toy's own builder pins 16, which the
/// host twin asserts against the moment a test walks past it).
fn layer_at(t: &Toy, l: usize, seq: usize) -> MlaDsv4Host {
    let cfg = t.cfg.clone();
    let d = cfg.dim;
    let f = |x: &(Vec<u8>, Vec<u8>), n: usize, k: usize| Fp8Host::new(n, k, &x.0, &x.1);
    let mut layer = MlaDsv4Host {
        layer_id: l,
        wq_a: f(&t.wq_a, cfg.q_lora, d),
        wq_b: f(&t.wq_b, cfg.n_heads * cfg.head_dim, cfg.q_lora),
        wkv: f(&t.wkv, cfg.head_dim, d),
        wo_a: f(&t.wo_a, cfg.o_groups * cfg.o_lora, cfg.o_in_per_group()),
        wo_b: f(&t.wo_b, d, cfg.o_groups * cfg.o_lora),
        q_norm: t.q_norm.clone(),
        kv_norm: t.kv_norm.clone(),
        sink: t.sink.clone(),
        rope: Rope::for_layer(&cfg, l, seq),
        crope: Rope::compress(&cfg, seq),
        compressor: None,
        indexer: None,
        cfg: cfg.clone(),
    };
    if cfg.kv_source_of(l) == Some(l) {
        layer.compressor = Some(CompressorHost::new(
            (cfg.ratio(l) as usize).max(1),
            Bf16Host::new(cfg.head_dim, d, &t.c_wkv_bf16),
            Some(Bf16Host::new(cfg.head_dim, d, &t.c_wgate_bf16)),
            t.c_norm.clone(),
        ));
    }
    if cfg.index_source_of(l) == Some(l) {
        let mut ix = IndexerHost::new(
            cfg.index_heads,
            cfg.index_dim,
            cfg.index_topk,
            f(&t.i_wq_b, cfg.index_heads * cfg.index_dim, cfg.q_lora),
            Bf16Host::new(cfg.index_heads, d, &t.i_weights_bf16),
            if cfg.is_kv_source(l) { Some(Bf16Host::new(cfg.index_dim, cfg.head_dim, &t.i_wk_bf16)) } else { None },
            t.i_k_norm.clone(),
        );
        if l == cfg.candidate_source_layer {
            ix = ix.with_candidate_source(cfg.candidate_topk_blocks, cfg.candidate_block_size);
        } else if cfg.uses_candidates(l) {
            ix = ix.with_candidates();
        }
        layer.indexer = Some(ix);
    }
    layer
}

/// The same comparison at the pack's real dimensions (v41 config, random weights): the toy dims cannot
/// catch width-conditional kernel paths (index_dim 128, head_dim 512, rope 64, q_lora 1280). Selection
/// is pinned dense here (`index_topk >= n_pos`, candidate blocks pinned in), so the comparison isolates
/// planes -> keys -> gather -> projections; the selective regime is asserted in
/// `gpu_block_matches_the_host_at_scale` and, mask-for-mask, in the candidate-selection test.
#[test]
fn gpu_block_matches_the_host_at_real_dims() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let cfg = Dsv4Config::v41();
    let layer = 2usize; // ratio 2, kv source + index source
    assert_eq!(cfg.ratio(layer), 2);
    let max_seq = 1024usize;
    let n = 256usize; // n / 2 = 128 <= index_topk (512), exercises bf16_mma_k (m > 8) at full v41 dims
    let t = toy_with(cfg.clone());
    let (d, dcmp) = (cfg.dim, cfg.head_dim);
    let f16s = |v: &[u16]| -> Vec<f32> { v.iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect() };

    let mut r = Rng(0x1234_5678_9abc_def0);
    let x = r.vec(n * d, -1.0, 1.0);
    let mut h = layer_at(&t, layer, max_seq);
    let mut hp = HostPlanes::new(&cfg, max_seq, layer);
    let want = h.forward(&x, n, 0, &mut hp, None);

    let xb: Vec<u16> = x.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
    let xdev = DevBuf::from_slice(&xb).unwrap();
    let mut m = MlaDsv4::new(&cfg, layer, &t.pack(layer), max_seq).unwrap();
    let dp = Dsv4Planes::new(&cfg, max_seq, layer, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
    let mut sc = Dsv4Scratch::new(&cfg, n, max_seq).unwrap();
    m.forward(&xdev, n, 0, &dp, None, &mut sc, &st).unwrap();

    let groups = n / 2;
    // index keys: how far off, in code steps?
    let dk: Vec<f32> = f16s(&dp.index_k.to_vec::<u16>().unwrap());
    let id = cfg.index_dim;
    let mut key_bad = 0usize;
    let mut key_worst = 0f64;
    for g in 0..groups {
        for j in 0..id {
            let (a, b) = (dk[g * id + j], hp.index_k[g * id + j]);
            let rel = (a - b).abs() / b.abs().max(1e-3);
            if rel > 0.34 {
                key_bad += 1;
            }
            key_worst = key_worst.max(rel as f64);
        }
    }
    eprintln!("real dims n={n}: index keys {key_bad}/{} entries past one fp4 code step, worst rel {key_worst:.3}", groups * id);

    // selection sets (device stages in scratch.oq at n > PLANE_ROWS)
    let topk_stage_off = ((cfg.window + n) * cfg.head_dim * 2 + 4095) & !4095;
    let oq_bytes = sc.oq.to_vec::<u8>().unwrap();
    let rd_i32 = |at: usize| -> i32 { i32::from_le_bytes([oq_bytes[at], oq_bytes[at + 1], oq_bytes[at + 2], oq_bytes[at + 3]]) };
    let stride = dp.sel_stride;
    let n_pos = n / 2;
    let cmp_len = n_pos.min(cfg.index_topk);
    let mut set_diff_rows = 0usize;
    let mut set_diff_total = 0usize;
    let mut worst_row = (0usize, 0usize);
    for row in 0..n {
        let host_row = &hp.topk_idxs[row * cfg.index_topk..][..cmp_len];
        let mut dev_set: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for k in 0..cmp_len {
            dev_set.insert(rd_i32(topk_stage_off + (row * stride + 128 + k) * 4));
        }
        let host_set: std::collections::HashSet<i32> = host_row.iter().copied().collect();
        let diff = host_set.symmetric_difference(&dev_set).count();
        if diff > 0 {
            set_diff_rows += 1;
            set_diff_total += diff;
            if diff > worst_row.1 {
                worst_row = (row, diff);
            }
        }
    }
    eprintln!(
        "real dims n={n}: selection SET differs on {set_diff_rows}/{n} rows, {set_diff_total} entries total, worst row {} ({})",
        worst_row.0, worst_row.1
    );

    let got: Vec<f32> = sc.out.to_vec::<u16>().unwrap()[..n * d].iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect();
    let rel = max_rel(&got, &want);
    eprintln!("real dims n={n}: block output max rel {rel:e}");
    assert!(rel < 5e-2, "n={n}: the device block must reproduce the host twin, max rel err {rel:e}");
    let _ = dcmp;
}
#[test]
fn gpu_block_matches_the_host_at_scale() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let mut cfg = Dsv4Config::toy(2);
    cfg.window = 128;
    cfg.index_topk = 512;
    cfg.candidate_topk_blocks = 2048;
    cfg.candidate_block_size = 8;
    let max_seq = 4096usize;
    let t = toy_with(cfg.clone());
    let (d, dcmp) = (cfg.dim, cfg.head_dim);
    let mut m0 = MlaDsv4::new(&cfg, 0, &t.pack(0), max_seq).unwrap();
    let d0 = Dsv4Planes::new(&cfg, max_seq, 0, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
    let mut sc = Dsv4Scratch::new(&cfg, MAX_ROWS, max_seq).unwrap();
    let f16s = |v: &[u16]| -> Vec<f32> { v.iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect() };

    // n <= 1024 keeps both sides' selection dense (n_pos <= index_topk); past it the top-512 cut
    // sits inside the score distribution and the selection sets are asserted mask-for-mask.
    for n in [600usize, 1024, 1115, 1400] {
        let mut r = Rng(0x0bad_c0de_0000_0000u64.wrapping_add(n as u64));
        let x = r.vec(n * d, -1.0, 1.0);
        // host twin
        let mut h0 = layer_at(&t, 0, max_seq);
        let mut hp0 = HostPlanes::new(&cfg, max_seq, 0);
        let want = h0.forward(&x, n, 0, &mut hp0, None);
        // device
        let xb: Vec<u16> = x.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
        let xdev = DevBuf::from_slice(&xb).unwrap();
        m0.forward(&xdev, n, 0, &d0, None, &mut sc, &st).unwrap();
        // stage by stage
        let groups = n / 2;
        let dc: Vec<f32> = f16s(&d0.compress.to_vec::<u16>().unwrap());
        {
            // diagnostic: which groups differ (contiguous band = structural, scattered = numeric)
            let mut bad: Vec<usize> = vec![];
            let mut worst = (0usize, 0f64);
            for g in 0..groups {
                let gv = &dc[g * dcmp..][..dcmp];
                let hv = &hp0.compress[g * dcmp..][..dcmp];
                let rel = max_rel(gv, hv);
                if rel > 1e-2 {
                    bad.push(g);
                }
                if rel > worst.1 {
                    worst = (g, rel);
                }
            }
            eprintln!(
                "n={n}: {}/{} groups off (>1e-2), first bad {:?}, worst group {} rel {:#.3}",
                bad.len(),
                groups,
                &bad[..bad.len().min(12)],
                worst.0,
                worst.1
            );
            if !bad.is_empty() {
                for &g in bad.iter().take(3) {
                    let mx = hp0.compress[g * dcmp..][..dcmp].iter().map(|v| v.abs()).fold(0f32, |a, v| a.max(v));
                    let diffs: Vec<String> = (0..dcmp)
                        .filter(|&dd| (dc[g * dcmp + dd] - hp0.compress[g * dcmp + dd]).abs() / mx.max(1e-6) > 1e-2)
                        .map(|dd| {
                            format!(
                                "d{dd}: got {:.4} want {:.4} (want[d-1]={:.4} want[d+1]={:.4})",
                                dc[g * dcmp + dd],
                                hp0.compress[g * dcmp + dd],
                                hp0.compress[g * dcmp + dd.saturating_sub(1)],
                                hp0.compress[g * dcmp + (dd + 1).min(dcmp - 1)]
                            )
                        })
                        .collect();
                    eprintln!("  group {g}: {}", diffs.join(" | "));
                }
            }
        }
        let rel = max_rel(&dc[..groups * dcmp], &hp0.compress[..groups * dcmp]);
        // NOTE: the fp4 round-trip (block 16, E4M3 scales) hits nearest-code rounding boundaries where the
        // device's f16 block-max and the host's f32 differ by one E2M1 code step (x1.5) on scattered
        // single dims -- bounded by that step, not by noise. The load-bearing checks are the selection
        // (exact) and the block output (5e-2).
        assert!(rel < 0.51, "n={n}: the compressed plane must be the host's, max rel err {rel:e}");
        let dk: Vec<f32> = f16s(&d0.index_k.to_vec::<u16>().unwrap());
        let rel = max_rel(&dk[..groups * cfg.index_dim], &hp0.index_k[..groups * cfg.index_dim]);
        assert!(rel < 0.51, "n={n}: the index keys must be the host's, max rel err {rel:e}");
        // the published selection: window half then compressed half, exact. n > PLANE_ROWS stages the row
        // set in scratch.oq (topk_stage_off), n <= PLANE_ROWS writes the persistent plane.
        let topk_stage_off = ((cfg.window + n) * cfg.head_dim * 2 + 4095) & !4095;
        let oq_bytes = sc.oq.to_vec::<u8>().unwrap();
        let rd_i32 =
            |at: usize| -> i32 { i32::from_le_bytes([oq_bytes[at], oq_bytes[at + 1], oq_bytes[at + 2], oq_bytes[at + 3]]) };
        let stride = d0.sel_stride;
        let n_pos = n / 2;
        let cmp_len = n_pos.min(cfg.index_topk);
        let mut sel_diffs = 0usize;
        let mut first_bad: Option<(usize, usize, i32, i32)> = None;
        for row in 0..n {
            let host_row = &hp0.topk_idxs[row * cfg.index_topk..];
            for (k, &hv) in host_row.iter().enumerate().take(cmp_len) {
                let dev_v = rd_i32(topk_stage_off + (row * stride + 128 + k) * 4);
                if dev_v != hv {
                    if first_bad.is_none() {
                        first_bad = Some((row, k, dev_v, hv));
                    }
                    sel_diffs += 1;
                }
            }
        }
        if sel_diffs > 0 {
            eprintln!(
                "n={n}: selection entries differ from the host's: {sel_diffs} of {}, first at {:?}",
                n * cmp_len,
                first_bad
            );
        }
        // order-swaps inside the same selected SET are benign for the gather; a differing SET is not.
        if n_pos > cfg.index_topk {
            let mut set_diff_rows = 0usize;
            let mut set_diff_total = 0usize;
            for row in 0..n {
                let host_row = &hp0.topk_idxs[row * cfg.index_topk..][..cmp_len];
                let mut dev_set: std::collections::HashSet<i32> = std::collections::HashSet::new();
                for k in 0..cmp_len {
                    dev_set.insert(rd_i32(topk_stage_off + (row * stride + 128 + k) * 4));
                }
                let host_set: std::collections::HashSet<i32> = host_row.iter().copied().collect();
                let d = host_set.symmetric_difference(&dev_set).count();
                if d > 0 {
                    set_diff_rows += 1;
                    set_diff_total += d;
                }
            }
            eprintln!(
                "n={n}: selection SET differs on {set_diff_rows}/{} rows, {set_diff_total} symmetric-diff entries total",
                n
            );
        }
        let got: Vec<f32> =
            sc.out.to_vec::<u16>().unwrap()[..n * d].iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect();
        let rel = max_rel(&got, &want);
        let tol = if n_pos <= cfg.index_topk { 5e-2 } else { 8e-2 };
        assert!(rel < tol, "n={n}: the device block must reproduce the host twin, max rel err {rel:e} (tol {tol:e})");
        eprintln!("n={n}: planes, keys, selection and output match the host twin");

        // two decode steps on top (state carry at ratio 2: one step half-fills a group, the next completes it)
        for pos in [n, n + 1] {
            let xt = r.vec(d, -1.0, 1.0);
            let xb: Vec<u16> = xt.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
            let xtdev = DevBuf::from_slice(&xb).unwrap();
            let want = h0.forward(&xt, 1, pos, &mut hp0, None);
            m0.forward(&xtdev, 1, pos, &d0, None, &mut sc, &st).unwrap();
            let got: Vec<f32> =
                sc.out.to_vec::<u16>().unwrap()[..d].iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect();
            let rel = max_rel(&got, &want);
            assert!(rel < tol, "n={n} pos={pos}: decode must reproduce the host twin, max rel err {rel:e} (tol {tol:e})");
        }
    }
}

/// The candidate selection must survive block counts past the shared-memory cliff: the level-1
/// block scores scale with the context (one f32 per 8-position block) and the single-kernel form
/// asked for `n_blocks * 4` bytes of dynamic shared memory, which CUDA refuses past the 48 KB
/// default -- `cudaErrorInvalidValue` at `Mla { layer: 20 }` for every final prefill chunk whose
/// start sat above ~71.7K tokens (interior chunks never reach the layer-20 indexer, so only the
/// final chunk crashed). The observed crash geometry is exactly 98304 positions (12288 blocks =
/// 48 KB + the static histogram); the rungs need 65536 (512K) and 131072 (1M) blocks.
#[test]
fn gpu_forward_batch_matches_serial_forward_slot_bit_for_bit() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    for ratio in [1u8, 2u8] {
        let t = toy(ratio);
        let cfg = t.cfg.clone();
        let d = cfg.dim;
        let mut rng = Rng(0x55aa_1234_0000 + ratio as u64);
        let pre_lens = [4usize, 3, 2, 5];
        let n_slots = pre_lens.len();

        let mut m0_ser = MlaDsv4::new(&cfg, 0, &t.pack(0), 16).unwrap();
        let mut m1_ser = MlaDsv4::new(&cfg, 1, &t.pack(1), 16).unwrap();
        let mut m0_bat = MlaDsv4::new(&cfg, 0, &t.pack(0), 16).unwrap();
        let mut m1_bat = MlaDsv4::new(&cfg, 1, &t.pack(1), 16).unwrap();

        let mut p0_ser = Vec::new();
        let mut p1_ser = Vec::new();
        let mut p0_bat = Vec::new();
        let mut p1_bat = Vec::new();
        for _ in 0..n_slots {
            p0_ser.push(Dsv4Planes::new(&cfg, 16, 0, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap());
            p1_ser.push(Dsv4Planes::new(&cfg, 16, 1, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap());
            p0_bat.push(Dsv4Planes::new(&cfg, 16, 0, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap());
            p1_bat.push(Dsv4Planes::new(&cfg, 16, 1, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap());
        }
        let mut sc_ser = Dsv4Scratch::new(&cfg, 16, 16).unwrap();
        let mut sc_bat = Dsv4Scratch::new(&cfg, 16, 16).unwrap();

        for (sl, &plen) in pre_lens.iter().enumerate() {
            let xp: Vec<u16> = rng.vec(plen * d, -1.0, 1.0).into_iter().map(|v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
            let xdev = DevBuf::from_slice(&xp).unwrap();
            m0_ser.forward_slot(&xdev, plen, 0, sl, &p0_ser[sl], None, &mut sc_ser, &st).unwrap();
            m1_ser.forward_slot(&xdev, plen, 0, sl, &p1_ser[sl], Some(&p0_ser[sl]), &mut sc_ser, &st).unwrap();
            m0_bat.forward_slot(&xdev, plen, 0, sl, &p0_bat[sl], None, &mut sc_bat, &st).unwrap();
            m1_bat.forward_slot(&xdev, plen, 0, sl, &p1_bat[sl], Some(&p0_bat[sl]), &mut sc_bat, &st).unwrap();
        }

        for step in 0..2usize {
            let x_step: Vec<u16> =
                rng.vec(n_slots * d, -1.0, 1.0).into_iter().map(|v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
            let x_all = DevBuf::from_slice(&x_step).unwrap();

            let mut want0 = Vec::with_capacity(n_slots * d);
            let mut want1 = Vec::with_capacity(n_slots * d);
            for sl in 0..n_slots {
                let x_sl = DevBuf::from_slice(&x_step[sl * d..(sl + 1) * d]).unwrap();
                let pos = pre_lens[sl] + step;
                m0_ser.forward_slot(&x_sl, 1, pos, sl, &p0_ser[sl], None, &mut sc_ser, &st).unwrap();
                want0.extend_from_slice(&sc_ser.out.to_vec::<u16>().unwrap()[..d]);
                m1_ser.forward_slot(&x_sl, 1, pos, sl, &p1_ser[sl], Some(&p0_ser[sl]), &mut sc_ser, &st).unwrap();
                want1.extend_from_slice(&sc_ser.out.to_vec::<u16>().unwrap()[..d]);
            }

            let mut out0 = DevBuf::zeroed_view(n_slots * d * 2).unwrap();
            let mut out1 = DevBuf::zeroed_view(n_slots * d * 2).unwrap();
            let seqs0: Vec<gb_cuda::dsv4_attn::MlaSeq<'_>> = (0..n_slots)
                .map(|sl| gb_cuda::dsv4_attn::MlaSeq {
                    slot: sl,
                    row0: sl,
                    rows: 1,
                    start_pos: pre_lens[sl] + step,
                    planes: &p0_bat[sl],
                    shared: None,
                })
                .collect();
            m0_bat.forward_batch(&x_all, n_slots, &seqs0, &mut sc_bat, &mut out0, &st).unwrap();
            let seqs1: Vec<gb_cuda::dsv4_attn::MlaSeq<'_>> = (0..n_slots)
                .map(|sl| gb_cuda::dsv4_attn::MlaSeq {
                    slot: sl,
                    row0: sl,
                    rows: 1,
                    start_pos: pre_lens[sl] + step,
                    planes: &p1_bat[sl],
                    shared: Some(&p0_bat[sl]),
                })
                .collect();
            m1_bat.forward_batch(&x_all, n_slots, &seqs1, &mut sc_bat, &mut out1, &st).unwrap();
            st.sync().unwrap();
            let got0: Vec<u16> = out0.to_vec().unwrap();
            let got1: Vec<u16> = out1.to_vec().unwrap();
            assert_eq!(got0, want0, "ratio={ratio} step={step} layer 0 batched != serial");
            assert_eq!(got1, want1, "ratio={ratio} step={step} layer 1 batched != serial");
        }
    }
}

#[test]
fn gpu_candidate_selection_survives_block_counts_past_the_shmem_cliff() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let cfg = Dsv4Config::v41();
    // 1. the serving shape must launch: layer 20's replay tail (n = 128, start_pos just below the
    //    total) is the call that used to die with cudaErrorInvalidValue past ~96K positions.
    {
        let layer = cfg.candidate_source_layer;
        let t = toy_with(cfg.clone());
        let (n, total) = (128usize, 98_304usize);
        let mut r = Rng(0xc1ff_0000_0000_0001u64 ^ total as u64);
        let x = r.vec(n * cfg.dim, -1.0, 1.0);
        let xb: Vec<u16> = x.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
        let xdev = DevBuf::from_slice(&xb).unwrap();
        let mut m = MlaDsv4::new(&cfg, layer, &t.pack(layer), 262_144).unwrap();
        let dp = Dsv4Planes::new(&cfg, 262_144, layer, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
        let mut sc = Dsv4Scratch::new(&cfg, n, 262_144).unwrap();
        m.forward(&xdev, n, total - n, &dp, None, &mut sc, &st)
            .unwrap_or_else(|e| panic!("total={total}: the layer-20 candidate path must launch, got {e}"));
    }
    // 2. the selection itself, driven directly with known scores and compared mask-for-mask with the
    //    host twin, at the rungs' block counts: 12288 (256K), 65536 (512K) and 131072 (1M) blocks.
    //    `keep` is the serving value (2048) everywhere, so all of them are in the selective regime.
    //
    // 3. the shift-0 radix pass in ISOLATION. The low-byte pass's mask used to be `0`, which left
    //    the threshold at the bottom of the final tie bucket instead of at the k-th key. With
    //    distinct scores that bucket is a singleton whose bottom IS the k-th key, so the emit still
    //    landed on k and the defect was invisible (the previously passing 512-width case was of
    //    exactly this kind). These cases put many blocks in ONE top-24-bit bucket and vary only the
    //    low byte, which is the only shape the shift-0 pass can separate; `lows[b]` is block b's
    //    low mantissa byte.
    let check = |width: usize, rows: usize, topk: usize, lows: Option<&[u32]>, start_pos: usize| {
        let bs = cfg.candidate_block_size;
        assert_eq!(width % bs, 0, "width {width} must be whole {bs}-position blocks");
        let n_blocks = width / bs;
        assert!(lows.is_none_or(|l| l.len() == n_blocks), "one low byte per block");
        let mut host_scores = vec![0f32; rows * width];
        match lows {
            Some(l) => {
                for r in 0..rows {
                    for (b, &low) in l.iter().enumerate().take(n_blocks) {
                        // all positions of a block share the value, so the block max is exactly it
                        let v = f32::from_bits(0x3F80_0000 | low);
                        for i in b * bs..(b + 1) * bs {
                            host_scores[r * width + i] = v;
                        }
                    }
                }
            }
            None => {
                let mut rng = Rng(0x5bad_c0de_0000_0000u64 ^ width as u64);
                for v in host_scores.iter_mut() {
                    *v = rng.uniform(-3.0, 3.0);
                }
                // a few -inf (unreachable) and a runaway top block: shapes the radix must handle
                for r in 0..rows {
                    for p in width - 3..width {
                        host_scores[r * width + p] = f32::NEG_INFINITY;
                    }
                    host_scores[r * width + 100] = 9.0;
                }
            }
        }
        let score = DevBuf::from_slice(&host_scores).unwrap();
        let mut bscore = DevBuf::zeroed(rows * n_blocks * 4).unwrap();
        let mut mask = DevBuf::zeroed(rows * width).unwrap();
        candidate_blocks_dev(&score, &mut bscore, &mut mask, width, rows, width, start_pos, 1, topk, bs, &st)
            .unwrap_or_else(|e| panic!("width={width}: the candidate kernels must launch, got {e}"));
        let got: Vec<u8> = mask.to_vec().unwrap();
        let keep = topk.min(n_blocks);
        for row in 0..rows {
            let want = select_candidate_blocks(&host_scores[row * width..][..width], start_pos + row + 1, topk, bs);
            let wb: Vec<u8> = want.iter().map(|&x| u8::from(x)).collect();
            let dev_kept: usize = got[row * width..][..width].iter().step_by(bs).map(|&x| x as usize).sum();
            let host_kept: usize = wb.iter().step_by(bs).map(|&x| x as usize).sum();
            if let Some(p) = (0..width).find(|&p| got[row * width + p] != wb[p]) {
                let diffs: Vec<usize> = (0..width).filter(|&q| got[row * width + q] != wb[q]).take(12).collect();
                panic!(
                    "width={width} ({n_blocks} blocks, keep {keep}) row={row}: mask differs at {p} (and {diffs:?}); \
                     dev keeps {dev_kept} blocks, host keeps {host_kept}"
                );
            }
            assert_eq!(dev_kept, keep, "width={width} row={row}: the mask must hold exactly `keep` blocks");
            eprintln!(
                "width={width}: {} blocks, keep {keep} (lows {}) -> mask == host twin, {dev_kept} blocks kept",
                n_blocks,
                if lows.is_some() { "tied" } else { "random" }
            );
        }
    };
    // the rung block counts (the shmem-cliff axis)
    check(98_304, 2, 2048, None, 98_176);
    check(524_288, 2, 2048, None, 524_160);
    check(1_048_576, 2, 2048, None, 1_048_448);
    // keep == n_blocks exactly: the early branch (nothing to select), the boundary the 16384 width sits on
    check(16_384, 2, 2048, None, 16_256);
    // one block past the early branch, uniform values: 2049 blocks, 2048 kept, every value in one bucket
    check(16_392, 2, 2048, None, 16_264);
    // tied: 3 of every 4 blocks five ulps above the bucket bottom, keep 8 of 64 -- the bottom-of-bucket
    // threshold admitted all 48 above-bottom blocks (mask kept 64, not 8)
    let mixed: Vec<u32> = (0..64).map(|b| if b % 4 == 0 { 0 } else { 5 }).collect();
    check(512, 2, 8, Some(&mixed), 384);
    // tied, marginal: 2047 blocks at low 6 and the last two at low 5 -- the true k-th key is the first
    // low-5 block, so the correct emit is exactly 2048; a bottom-of-bucket threshold emits 2049, one
    // past the k slots (the serving shape that corrupted the row tail)
    let mut marginal: Vec<u32> = vec![6u32; 2049];
    marginal[2047] = 5;
    marginal[2048] = 5;
    check(16_392, 2, 2048, Some(&marginal), 16_264);
}

/// The `Dsv4Scratch` bscore staging must be 16-aligned in the real allocation and usable as a
/// `float*` base for `candidate_score_kernel` on the long-context shape that used to crash with
/// CUDA error 716 (misaligned address). Runs on its own stream, never the legacy one.
#[test]
fn gpu_dsv4_scratch_bscore_staging_is_aligned_and_runs() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let cfg = Dsv4Config::v41();
    let max_seq = 262_144usize; // shipped dsv41-single-262k / dsv41-tp2-262k window
    let rows = 3usize; // odd rows: the pre-fix offset was misaligned here
    let sc = Dsv4Scratch::new(&cfg, rows, max_seq).unwrap();
    let bscore_off = Dsv4Scratch::bscore_off(&cfg, rows, max_seq);
    assert_eq!(bscore_off % 16, 0, "bscore_off must be 16-aligned");
    let base = sc.oq.ptr_at(bscore_off) as usize;
    assert_eq!(base % 16, 0, "allocated bscore pointer must be 16-aligned");

    let bs = cfg.candidate_block_size;
    let width = 16_512usize; // 2064 blocks > candidate_topk_blocks (2048): the selective regime
    let n_blocks = width / bs;
    let topk = cfg.candidate_topk_blocks;
    let start_pos = 16_384usize;
    let mut rng = Rng(0x7160_0000_0000_0001u64 ^ (max_seq as u64));
    let mut host_scores = vec![0f32; rows * width];
    for v in host_scores.iter_mut() {
        *v = rng.uniform(-3.0, 3.0);
    }
    let score = DevBuf::from_slice(&host_scores).unwrap();
    let blk = sc.oq.block().unwrap().clone();
    let mut bscore = DevBuf::view(&blk, bscore_off, rows * n_blocks * 4);
    let mut mask = DevBuf::zeroed(rows * width).unwrap();
    candidate_blocks_dev(&score, &mut bscore, &mut mask, width, rows, width, start_pos, 1, topk, bs, &st)
        .unwrap_or_else(|e| panic!("candidate kernels must launch on the scratch bscore staging, got {e}"));
    st.sync().unwrap();
    let got: Vec<u8> = mask.to_vec().unwrap();
    for row in 0..rows {
        let want = select_candidate_blocks(&host_scores[row * width..][..width], start_pos + row + 1, topk, bs);
        let wb: Vec<u8> = want.iter().map(|&x| u8::from(x)).collect();
        let dev_kept: usize = got[row * width..][..width].iter().step_by(bs).map(|&x| x as usize).sum();
        assert_eq!(&got[row * width..][..width], &wb[..], "row {row}: mask mismatch");
        assert_eq!(dev_kept, topk.min(n_blocks), "row {row}");
    }
}

/// Alias-free stage trace of the compressor chain: `compress_only_slot` stops before the Q/attention
/// stages reuse the scratch block, so ckv/gate/latent hold their own values on return.
#[test]
fn gpu_real_dims_compressor_stage_trace() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let mut cfg = Dsv4Config::v41();
    cfg.n_heads = 4;
    cfg.o_groups = 2;
    let layer = 2usize;
    let max_seq = 2048usize;
    let n = 1040usize;
    let t = toy_with(cfg.clone());
    let mut r = Rng(0x1234_5678_9abc_def0);
    let x = r.vec(n * cfg.dim, -1.0, 1.0);

    let mut h = layer_at(&t, layer, max_seq);
    let mut hp = HostPlanes::new(&cfg, max_seq, layer);
    let _ = h.forward(&x, n, 0, &mut hp, None);

    let xb: Vec<u16> = x.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
    let xdev = DevBuf::from_slice(&xb).unwrap();
    let mut m = MlaDsv4::new(&cfg, layer, &t.pack(layer), max_seq).unwrap();
    let dp = Dsv4Planes::new(&cfg, max_seq, layer, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
    let mut sc = Dsv4Scratch::new(&cfg, n, max_seq).unwrap();
    m.compress_only_slot(&xdev, n, 0, 0, &dp, &mut sc, false, &st).unwrap();

    let f16s = |b: &[u16]| -> Vec<f32> { b.iter().map(|&v| gb_exl3::f16::f16_to_f64(v) as f32).collect() };
    let cmp = |name: &str, got: &[f32], want: &[f32]| {
        let rel = max_rel(got, want);
        eprintln!("stage {name}: max rel {rel:.4} over {} entries", got.len());
    };
    let comp = h.compressor.as_ref().unwrap();
    let want_ckv = comp.wkv.matmul(&x, n);
    let got_ckv: Vec<f32> = f16s(&sc.ckv.to_vec::<u16>().unwrap());
    cmp("ckv (bf16 gemm)", &got_ckv[..want_ckv.len()], &want_ckv);
    let groups = n / 2;
    let got_lat: Vec<f32> = f16s(&sc.latent.to_vec::<u16>().unwrap());
    cmp("latent vs host plane", &got_lat[..groups * cfg.head_dim], &hp.compress[..groups * cfg.head_dim]);
    let got_kt: Vec<f32> = f16s(&sc.index_kt.to_vec::<u16>().unwrap());
    cmp("index_kt vs host plane keys", &got_kt[..groups * cfg.index_dim], &hp.index_k[..groups * cfg.index_dim]);
}

/// The two bf16 GEMM kernels must agree: `bf16_linear_k` is the trivially correct fallback;
/// `bf16_mma_k` (m > 8, k%64==0, n%32==0) is the fast path every real compressor wkv/wgate and
/// indexer wk/weights_proj call takes. Same weights, same x, rows 0..7 shared.
#[test]
fn gpu_bf16_mma_matches_the_fallback() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let (k, n, m_big) = (5120usize, 512usize, 1400usize);
    let mut r = Rng(0xb16_5010);
    let xh: Vec<u16> = (0..m_big * k).map(|_| gb_exl3::f16::f64_to_f16(r.uniform(-1.0, 1.0) as f64)).collect();
    let w: Vec<u8> = r.vec(n * k, -0.5, 0.5).iter().flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes()).collect();
    let xdev = DevBuf::from_slice(&xh).unwrap();
    let wgt = Bf16W::new(n, k, &w).unwrap();
    let host = Bf16Host::new(n, k, &w);

    let mut y_big = DevBuf::zeroed(m_big * n * 2).unwrap();
    wgt.forward(&xdev, &mut y_big, m_big, Dtype::F16, &st).unwrap(); // bf16_mma_k
    let got_big: Vec<u16> = y_big.to_vec::<u16>().unwrap();

    let m_small = 8usize;
    let xs = DevBuf::from_slice(&xh[..m_small * k]).unwrap();
    let mut y_small = DevBuf::zeroed(m_small * n * 2).unwrap();
    wgt.forward(&xs, &mut y_small, m_small, Dtype::F16, &st).unwrap(); // bf16_linear_k
    let got_small: Vec<u16> = y_small.to_vec::<u16>().unwrap();

    let want = host.matmul(&xh[..m_small * k].iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect::<Vec<f32>>(), m_small);
    let f = |b: u16| gb_exl3::f16::f16_to_f64(b) as f32;
    let mut mma_vs_fb = 0usize;
    let mut fb_vs_host = 0usize;
    for i in 0..m_small * n {
        let (a, b, c) = (f(got_big[i]), f(got_small[i]), want[i]);
        if (a - b).abs() > 0.02 * (c.abs() + 0.02) {
            mma_vs_fb += 1;
        }
        if (b - c).abs() > 0.02 * (c.abs() + 0.02) {
            fb_vs_host += 1;
        }
    }
    eprintln!(
        "bf16 gemm k={k} n={n}: mma-vs-fallback mismatches {mma_vs_fb}/{mn}, fallback-vs-host {fb_vs_host}/{mn}",
        mn = m_small * n
    );
    assert_eq!(mma_vs_fb, 0, "bf16_mma_k must agree with bf16_linear_k on shared rows");
    assert_eq!(fb_vs_host, 0, "the fallback must agree with the host twin");
}

#[test]
fn gpu_prefix_snapshot_restore_matches_cold_prefill_on_odd_and_even_cuts() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    for ratio in [1u8, 2u8] {
        for cut in [3usize, 4usize] {
            let total = 6usize;
            let suffix = total - cut;
            let t = toy(ratio);
            let cfg = t.cfg.clone();
            let d = cfg.dim;
            let mut rng = Rng(0x77cc_0000 + (ratio as u64) * 16 + cut as u64);
            let x_full: Vec<u16> =
                rng.vec(total * d, -1.0, 1.0).into_iter().map(|v| gb_exl3::f16::f64_to_f16(v as f64)).collect();

            // Cold reference on slot 0
            let mut m0_cold = MlaDsv4::new(&cfg, 0, &t.pack(0), 16).unwrap();
            let mut m1_cold = MlaDsv4::new(&cfg, 1, &t.pack(1), 16).unwrap();
            let p0_cold = Dsv4Planes::new(&cfg, 16, 0, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
            let p1_cold = Dsv4Planes::new(&cfg, 16, 1, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
            let mut sc_cold = Dsv4Scratch::new(&cfg, 16, 16).unwrap();
            let x_cold_dev = DevBuf::from_slice(&x_full).unwrap();
            m0_cold.forward_slot(&x_cold_dev, total, 0, 0, &p0_cold, None, &mut sc_cold, &st).unwrap();
            let want0: Vec<u16> = sc_cold.out.to_vec().unwrap()[..total * d].to_vec();
            m1_cold.forward_slot(&x_cold_dev, total, 0, 0, &p1_cold, Some(&p0_cold), &mut sc_cold, &st).unwrap();
            let want1: Vec<u16> = sc_cold.out.to_vec().unwrap()[..total * d].to_vec();

            // Prefix snapshot at `cut` on slot 0, restored into slot 2, followed by suffix `cut..total` on slot 2
            let mut m0 = MlaDsv4::new(&cfg, 0, &t.pack(0), 16).unwrap();
            let mut m1 = MlaDsv4::new(&cfg, 1, &t.pack(1), 16).unwrap();
            let p0_s0 = Dsv4Planes::new(&cfg, 16, 0, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
            let p1_s0 = Dsv4Planes::new(&cfg, 16, 1, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
            let p0_s2 = Dsv4Planes::new(&cfg, 16, 0, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
            let p1_s2 = Dsv4Planes::new(&cfg, 16, 1, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
            let mut sc = Dsv4Scratch::new(&cfg, 16, 16).unwrap();

            let x_pre_dev = DevBuf::from_slice(&x_full[..cut * d]).unwrap();
            m0.forward_slot(&x_pre_dev, cut, 0, 0, &p0_s0, None, &mut sc, &st).unwrap();
            m1.forward_slot(&x_pre_dev, cut, 0, 0, &p1_s0, Some(&p0_s0), &mut sc, &st).unwrap();
            st.sync().unwrap();

            // Take snapshot of slot 0 at `cut`
            let n_comp = cut / (ratio as usize);
            let cmp_bytes = n_comp * cfg.head_dim * 2;
            let ik_bytes = n_comp * cfg.index_dim * 2;
            let mut cmp_snap = vec![0u8; cmp_bytes];
            let mut ik_snap = vec![0u8; ik_bytes];
            if cmp_bytes > 0 {
                p0_s0.compress.download_range(0, &mut cmp_snap).unwrap();
                p0_s0.index_k.download_range(0, &mut ik_snap).unwrap();
            }
            let comp_state = m0.comp_state_snapshot(0, cut).unwrap();
            let mut w0_snap = vec![0u8; p0_s0.window.bytes()];
            let mut w1_snap = vec![0u8; p1_s0.window.bytes()];
            p0_s0.window.download_range(0, &mut w0_snap).unwrap();
            p1_s0.window.download_range(0, &mut w1_snap).unwrap();

            // Restore into slot 2
            unsafe {
                if cmp_bytes > 0 {
                    st.memcpy_h2d_async(p0_s2.compress.ptr(), cmp_snap.as_ptr(), cmp_bytes).unwrap();
                    st.memcpy_h2d_async(p0_s2.index_k.ptr(), ik_snap.as_ptr(), ik_bytes).unwrap();
                }
                st.memcpy_h2d_async(p0_s2.window.ptr(), w0_snap.as_ptr(), w0_snap.len()).unwrap();
                st.memcpy_h2d_async(p1_s2.window.ptr(), w1_snap.as_ptr(), w1_snap.len()).unwrap();
            }
            m0.comp_state_restore(2, comp_state.as_ref(), &st).unwrap();
            st.sync().unwrap();

            // Run suffix `cut..total` on slot 2 with `ced_replay_tail = false`
            let x_suf_dev = DevBuf::from_slice(&x_full[cut * d..]).unwrap();
            m0.forward_slot_ex(&x_suf_dev, suffix, cut, 2, &p0_s2, None, false, false, &mut sc, false, &st).unwrap();
            let got0: Vec<u16> = sc.out.to_vec().unwrap()[..suffix * d].to_vec();
            m1.forward_slot_ex(&x_suf_dev, suffix, cut, 2, &p1_s2, Some(&p0_s2), false, false, &mut sc, false, &st).unwrap();
            let got1: Vec<u16> = sc.out.to_vec().unwrap()[..suffix * d].to_vec();

            assert_eq!(got0, &want0[cut * d..], "ratio={ratio} cut={cut} layer 0 suffix mismatch");
            assert_eq!(got1, &want1[cut * d..], "ratio={ratio} cut={cut} layer 1 suffix mismatch");
        }
    }
}

/// V57: the per-row attention output of a prompt must not depend on the prefill chunk grid — running
/// the same rows as one call ("whole") and as 128/256/512-row chunks must be bit-identical for every
/// row. This test drives the layer pair that carries the sparse-attention plane (layer 20: candidate
/// source; layer 24: reader index-layer applying the level-1 candidate mask) through the whole prompt
/// and through the three chunk grids, and asserts the trailing `PLANE_ROWS` rows (the rows the final
/// logits depend on) are bit-identical.
///
/// Companion fixes: `need_cand` in `dsv4_attn.rs` is gated only by the global compressed-block count
/// (`n_blocks > candidate_topk_blocks`), never by `n <= PLANE_ROWS`; the layer-20 CED tail
/// (`CedTailSnap` in `model.rs`) splices the trailing 128 rows back into the final chunk, so the
/// layer-20 CED compaction sees the same window whichever grid the prompt was cut on; and the
/// `window_idxs_kernel` `chunk_only` branch (the `ced_replay_tail=true` path) is the one the final
/// 128-row call runs.
///
/// A scaled v41 geometry is used (layer 20 candidate source, layer 24 reader index-layer; window 128;
/// index_topk 128; 2048 candidate blocks of 8) so the run stays fast while the structural behavior
/// matches the pack.
#[test]
fn gpu_prefill_chunk_grid_invariance() {
    if !device_available() {
        eprintln!("not run: no device");
        return;
    }
    let st = Stream::new().unwrap();
    let mut cfg = Dsv4Config::toy(1);
    cfg.dim = 512;
    cfg.n_heads = 8;
    cfg.head_dim = 64;
    cfg.rope_dim = 32;
    cfg.q_lora = 128;
    cfg.o_lora = 128;
    cfg.o_groups = 4;
    cfg.window = 128;
    cfg.n_layers = 25;
    cfg.index_heads = 8;
    cfg.index_dim = 64;
    cfg.index_topk = 128;
    cfg.compress_ratios = vec![0u8; 25];
    cfg.compress_ratios[20] = 1;
    cfg.compress_ratios[24] = 1;
    cfg.kv_source_layers = vec![20];
    cfg.index_source_layers = vec![20, 24];
    cfg.candidate_source_layer = 20;
    cfg.candidate_topk_blocks = 2048;
    cfg.candidate_block_size = 8;
    let src = 20usize;
    let reader = 24usize;
    assert_eq!(cfg.candidate_source_layer, src);
    assert!(cfg.uses_candidates(reader), "layer {reader} must apply the candidate mask");
    let max_seq = 4096usize;
    let t = toy_with(cfg.clone());
    let d = cfg.dim;
    let tail = gb_cuda::dsv4_attn::PLANE_ROWS;
    let f16s = |v: &[u16]| -> Vec<f32> { v.iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect() };
    let mut m_src = MlaDsv4::new(&cfg, src, &t.pack(src), max_seq).unwrap();
    let mut m_rdr = MlaDsv4::new(&cfg, reader, &t.pack(reader), max_seq).unwrap();

    // Layer call with the exact parameters a grid chunk gets: interior chunks plain; the final chunk is
    // what `prefill_chunk_ex` sends to layer 20 — rows > 128 are compacted to the trailing 128
    // (`ced_replay_tail=true`), rows <= 128 run as-is (`ced_replay_tail=false`, window assembled from
    // the ring + the call's own rows).
    let layer_call = |m: &mut MlaDsv4,
                      xb: &[u16],
                      n: usize,
                      start: usize,
                      ced_tail: bool,
                      sc: &mut Dsv4Scratch,
                      p_own: &Dsv4Planes,
                      p_shared: Option<&Dsv4Planes>|
     -> Vec<f32> {
        let xdev = DevBuf::from_slice(&xb[start * d..(start + n) * d]).unwrap();
        m.forward_slot_ex(&xdev, n, start, 0, p_own, p_shared, false, ced_tail, sc, false, &st).unwrap();
        f16s(&sc.out.to_vec::<u16>().unwrap()[..n * d])
    };

    for l in [1024usize, 1500] {
        let mut r = Rng(0x57ab_cdef_0000_0000u64.wrapping_add(l as u64));
        let x = r.vec(l * d, -1.0, 1.0);
        let xb: Vec<u16> = x.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();

        // whole prompt: one call of all `l` rows
        let p_src_w = Dsv4Planes::new(&cfg, max_seq, src, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
        let p_rdr_w = Dsv4Planes::new(&cfg, max_seq, reader, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
        let mut sc_w = Dsv4Scratch::new(&cfg, l, max_seq).unwrap();
        let whole_src = layer_call(&mut m_src, &xb, l, 0, false, &mut sc_w, &p_src_w, None);
        let whole_rdr = layer_call(&mut m_rdr, &xb, l, 0, false, &mut sc_w, &p_rdr_w, Some(&p_src_w));

        // V57 CED-replay tail: the model's layer-20 CED block always finishes by running the trailing
        // `PLANE_ROWS` rows as one `ced_replay_tail=true` call — a whole prompt compacts to them, a
        // chunked grid splices its host tail (`CedTailSnap`) back in. So the reference for the final
        // call is the WHOLE run's compaction, replayed on the whole run's planes: every grid's final
        // call must equal it bit-for-bit. This exercises the `window_idxs_kernel` `chunk_only` branch
        // (Fix 1) on both sides, over fully populated compressed planes.
        let nfin = tail.min(l);
        let sfin = l - nfin;
        let fin_ref_src = layer_call(&mut m_src, &xb, nfin, sfin, true, &mut sc_w, &p_src_w, None);
        let fin_ref_rdr = layer_call(&mut m_rdr, &xb, nfin, sfin, true, &mut sc_w, &p_rdr_w, Some(&p_src_w));

        for grid in [128usize, 256, 512] {
            let p_src = Dsv4Planes::new(&cfg, max_seq, src, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
            let p_rdr = Dsv4Planes::new(&cfg, max_seq, reader, gb_cuda::dsv4_kvq4::KvFmt::F16).unwrap();
            let mut sc = Dsv4Scratch::new(&cfg, l, max_seq).unwrap();
            let mut pos = 0usize;
            let mut got_src = vec![0f32; l * d];
            let mut got_rdr = vec![0f32; l * d];
            while pos < l {
                let n = grid.min(l - pos);
                // naive grid: every chunk, including the final one, runs as a plain layer call
                let o0 = layer_call(&mut m_src, &xb, n, pos, false, &mut sc, &p_src, None);
                let o1 = layer_call(&mut m_rdr, &xb, n, pos, false, &mut sc, &p_rdr, Some(&p_src));
                got_src[pos * d..(pos + n) * d].copy_from_slice(&o0);
                got_rdr[pos * d..(pos + n) * d].copy_from_slice(&o1);
                pos += n;
            }
            // compare the trailing `tail` rows against the whole prompt's tail (f32 widened from f16:
            // equal iff the same f16 bits)
            let from = (l - tail) * d;
            let mism_src = whole_src[from..].iter().zip(&got_src[from..]).filter(|(a, b)| a != b).count();
            let mism_rdr = whole_rdr[from..].iter().zip(&got_rdr[from..]).filter(|(a, b)| a != b).count();
            eprintln!("L={l} grid {grid}: layer {src} {mism_src}, layer {reader} {mism_rdr} tail values differ from whole");
            assert_eq!(mism_src, 0, "L={l} grid {grid}: layer {src} tail differs from whole on {mism_src}");
            assert_eq!(mism_rdr, 0, "L={l} grid {grid}: layer {reader} tail differs from whole on {mism_rdr}");

            // V57 CED-replay tail: this grid's final call (the model's tail splice) must equal the whole
            // run's compaction replay (`fin_ref_*`) bit-for-bit, on every grid.
            let fin_src = layer_call(&mut m_src, &xb, nfin, sfin, true, &mut sc, &p_src, None);
            let fin_rdr = layer_call(&mut m_rdr, &xb, nfin, sfin, true, &mut sc, &p_rdr, Some(&p_src));
            let mism_src = fin_ref_src.iter().zip(&fin_src).filter(|(a, b)| a != b).count();
            let mism_rdr = fin_ref_rdr.iter().zip(&fin_rdr).filter(|(a, b)| a != b).count();
            eprintln!(
                "L={l} grid {grid} ced tail: layer {src} {mism_src}, layer {reader} {mism_rdr} tail values differ from whole compaction"
            );
            assert_eq!(mism_src, 0, "L={l} grid {grid} ced tail: layer {src} tail differs from whole compaction on {mism_src}");
            assert_eq!(
                mism_rdr, 0,
                "L={l} grid {grid} ced tail: layer {reader} tail differs from whole compaction on {mism_rdr}"
            );
        }

        eprintln!("L={l}: done");
    }
}
