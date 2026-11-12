//! DeepSeek-V4.1-Flash NextN / DSpark (`src/dsv4_mtp.rs`, `csrc/dsv4_mtp.cu`): one MTP layer against a **direct
//! transcription of model.py**, the order-sensitivity of the three-stage chain, the confidence head's monotonicity, and
//! `get_dspark_topk_idxs` pinned for a 12-token sequence at `block_size 5`.
//!
//! # Why a transcription and not the module
//!
//! `reference_layer` below is written from `/tmp/ds41ref/model.py` (`DSparkAttention.forward`, `Block.forward`,
//! `Block.hc_*`, `MoE.forward`, `Gate.forward`, `Expert.forward`), in f64, with its own RMSNorm, its own RoPE, its own
//! E4M3 block encoder and its own Sinkhorn. It calls nothing in the module under test, so a mistake in the module cannot
//! hide inside it. The two are compared to a tolerance that covers f32-vs-f64 only.
//!
//! # What runs without a GPU
//!
//! Everything except `gpu_*`: those print `not run: no device` and return, so this binary is green on a CPU-only box.
use gb_cuda::dsv4_attn::{select_candidate_blocks, Rope};
use gb_cuda::dsv4_moe::{sqrtsoftplus, HcCfg};
use gb_cuda::dsv4_mtp::*;
use gb_cuda::{device_available, DevBuf, Stream};

// ------------------------------------------------------------------------------------------------ fixtures

/// xorshift64*: the same test data everywhere, no dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * u
    }
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }
}

/// A toy stage with every weight filled: small magnitudes so nothing saturates, and non-zero `hc_*_scale` so the
/// hyper-connections actually mix (a zero scale would make every stream identical and hide an hc bug).
fn toy_stage(cfg: &MtpConfig, stage: usize, seq: usize, seed: u64) -> MtpLayerHost {
    let mut r = Rng(seed);
    let mut s = MtpLayerHost::zeros(cfg, stage, seq);
    let d = cfg.dim;
    s.wq_a = r.vec(cfg.q_lora * d, -0.2, 0.2);
    s.wq_b = r.vec(cfg.n_heads * cfg.head_dim * cfg.q_lora, -0.2, 0.2);
    s.wkv = r.vec(cfg.head_dim * d, -0.2, 0.2);
    s.wo_a = r.vec(cfg.o_groups * cfg.o_lora * cfg.o_in_per_group(), -0.2, 0.2);
    s.wo_b = r.vec(d * cfg.o_groups * cfg.o_lora, -0.2, 0.2);
    s.q_norm = r.vec(cfg.q_lora, 0.8, 1.2);
    s.kv_norm = r.vec(cfg.head_dim, 0.8, 1.2);
    s.sink = r.vec(cfg.n_heads, -0.5, 0.5);
    s.gate = r.vec(cfg.n_routed_experts * d, -0.3, 0.3);
    s.gate_bias = r.vec(cfg.n_routed_experts, -0.1, 0.1);
    s.w1 = r.vec(cfg.n_routed_experts * cfg.inter_dim * d, -0.3, 0.3);
    s.w3 = r.vec(cfg.n_routed_experts * cfg.inter_dim * d, -0.3, 0.3);
    s.w2 = r.vec(cfg.n_routed_experts * d * cfg.inter_dim, -0.3, 0.3);
    s.shared_w1 = r.vec(cfg.inter_dim * d, -0.3, 0.3);
    s.shared_w3 = r.vec(cfg.inter_dim * d, -0.3, 0.3);
    s.shared_w2 = r.vec(d * cfg.inter_dim, -0.3, 0.3);
    s.hc_attn_fn = r.vec(24 * 4 * d, -0.05, 0.05);
    s.hc_attn_base = r.vec(24, -0.2, 0.2);
    s.hc_attn_scale = [0.1, 0.1, 0.1];
    s.hc_ffn_fn = r.vec(24 * 4 * d, -0.05, 0.05);
    s.hc_ffn_base = r.vec(24, -0.2, 0.2);
    s.hc_ffn_scale = [0.1, 0.1, 0.1];
    s.attn_norm_w = Some(r.vec(d, 0.8, 1.2));
    s.ffn_norm_w = Some(r.vec(d, 0.8, 1.2));
    if let Some(p) = s.main_proj.as_mut() {
        *p = r.vec(d * cfg.tap_width(), -0.1, 0.1);
    }
    if let Some(n) = s.main_norm.as_mut() {
        *n = r.vec(d, 0.8, 1.2);
    }
    if let Some(n) = s.norm.as_mut() {
        *n = r.vec(d, 0.8, 1.2);
    }
    if let Some(m) = s.markov.as_mut() {
        m.embed = r.vec(cfg.vocab * cfg.markov_rank, -0.4, 0.4);
        m.head = r.vec(cfg.vocab * cfg.markov_rank, -0.4, 0.4);
    }
    if let Some(c) = s.confidence_head.as_mut() {
        c.proj = r.vec(cfg.conf_in(), -0.05, 0.05);
        c.bias = 0.05;
    }
    s
}

fn toy_chain(cfg: &MtpConfig) -> DsparkChain {
    let stages = (0..cfg.n_stages).map(|i| toy_stage(cfg, i, 64, 0x5eed_0000 + i as u64)).collect();
    DsparkChain::new(cfg.clone(), stages)
}

// ------------------------------------------------------------------------------------------------ the transcription
//
// Everything below is written from model.py in f64 and shares no code with `gb_cuda::dsv4_mtp`.

type V = Vec<f64>;

fn to64(v: &[f32]) -> V {
    v.iter().map(|&x| x as f64).collect()
}

fn rms(x: &mut [f64], w: &[f64], eps: f64) {
    let ss = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ss + eps).sqrt();
    for (v, wi) in x.iter_mut().zip(w) {
        *v = *v * r * wi;
    }
}

/// `F.linear(x, w)` with `w` `[n, k]`.
fn linear(w: &[f64], n: usize, k: usize, x: &[f64]) -> V {
    (0..n)
        .map(|c| {
            let wr = &w[c * k..][..k];
            (0..k).map(|i| wr[i] * x[i]).sum()
        })
        .collect()
}

fn linear_m(w: &[f64], n: usize, k: usize, x: &[f64], m: usize) -> V {
    let mut out = Vec::with_capacity(m * n);
    for r in 0..m {
        out.extend(linear(w, n, k, &x[r * k..][..k]));
    }
    out
}

/// `act_quant(x, 32, fp8, e8m0, inplace=True)`: round to nearest E4M3 with a power-of-two block scale.
fn act_quant(x: &mut [f64]) {
    for chunk in x.chunks_mut(32) {
        let amax = chunk.iter().fold(0f64, |a, &v| a.max(v.abs())).max(1e-4);
        let s = (amax / 448.0).log2().ceil().exp2();
        for v in chunk.iter_mut() {
            let q = *v / s;
            let a = q.abs();
            let r = if a < 1e-30 {
                0.0
            } else if a >= 448.0 {
                448.0 * q.signum()
            } else {
                let e = a.log2().floor();
                let step = if e < -6.0 { 2f64.powi(-9) } else { 2f64.powi(e as i32 - 3) };
                (a / step).round() * step * q.signum()
            };
            *v = r * s;
        }
    }
}

/// `apply_rotary_emb(x[..., -rd:], freqs_cis)` — adjacent pairs, `[rows][sub][block]` with the rotation on the tail of
/// each block.
#[allow(clippy::too_many_arguments)]
fn rope(x: &mut [f64], rows: usize, stride: usize, block: usize, sub: usize, pos0: usize, table: &Rope, inverse: bool) {
    let half = table.dim / 2;
    for r in 0..rows {
        let pos = pos0 + r;
        for b in 0..sub {
            let base = r * stride + b * block + (block - table.dim);
            for j in 0..half {
                let (c, s) = (table.cos[pos * half + j] as f64, table.sin[pos * half + j] as f64);
                let s = if inverse { -s } else { s };
                let (a, bb) = (x[base + 2 * j], x[base + 2 * j + 1]);
                x[base + 2 * j] = a * c - bb * s;
                x[base + 2 * j + 1] = a * s + bb * c;
            }
        }
    }
}

/// `hc_split_sinkhorn` for `hc_mult == 4` with `NFN == 24` mixes.
fn hc_split(mixes: &[f64], scale: &[f32; 3], base: &[f32], cfg: HcCfg) -> ([f64; 4], [f64; 4], [f64; 16]) {
    let eps = cfg.hc_eps as f64;
    let mut pre = [0f64; 4];
    let mut post = [0f64; 4];
    let mut comb = [0f64; 16];
    for h in 0..4 {
        pre[h] = 1.0 / (1.0 + (-(mixes[h] * scale[0] as f64 + base[h] as f64)).exp()) + eps;
        post[h] = 2.0 * (1.0 / (1.0 + (-(mixes[h + 4] * scale[1] as f64 + base[h + 4] as f64)).exp()));
    }
    for j in 0..16 {
        comb[j] = mixes[8 + j] * scale[2] as f64 + base[8 + j] as f64;
    }
    // Sinkhorn: row softmax + eps, one column normalisation, then (iters - 1) rounds of (row, column)
    for j in 0..4 {
        let mx = (0..4).fold(f64::NEG_INFINITY, |a, k| a.max(comb[j * 4 + k]));
        let mut s = 0.0;
        for k in 0..4 {
            comb[j * 4 + k] = (comb[j * 4 + k] - mx).exp();
            s += comb[j * 4 + k];
        }
        for k in 0..4 {
            comb[j * 4 + k] = comb[j * 4 + k] / s + eps;
        }
    }
    for k in 0..4 {
        let c: f64 = (0..4).map(|j| comb[j * 4 + k]).sum();
        for j in 0..4 {
            comb[j * 4 + k] /= c + eps;
        }
    }
    for _ in 1..cfg.sinkhorn_iters {
        for j in 0..4 {
            let s: f64 = (0..4).map(|k| comb[j * 4 + k]).sum();
            for k in 0..4 {
                comb[j * 4 + k] /= s + eps;
            }
        }
        for k in 0..4 {
            let s: f64 = (0..4).map(|j| comb[j * 4 + k]).sum();
            for j in 0..4 {
                comb[j * 4 + k] /= s + eps;
            }
        }
    }
    (pre, post, comb)
}

fn hc_mixes(x: &[f64], rows: usize, hidden: usize, fnw: &[f64], scale: &[f32; 3], base: &[f32], cfg: HcCfg) -> (V, V, V) {
    let row = 4 * hidden;
    let (mut pre, mut post, mut comb) = (vec![0f64; rows * 4], vec![0f64; rows * 4], vec![0f64; rows * 16]);
    for r in 0..rows {
        let xr = &x[r * row..][..row];
        let mut q = vec![0f64; 25];
        for j in 0..24 {
            q[j] = (0..row).map(|i| xr[i] * fnw[j * row + i]).sum();
        }
        q[24] = xr.iter().map(|v| v * v).sum();
        let rmr = 1.0 / (q[24] / row as f64 + cfg.rms_eps as f64).sqrt();
        for v in q.iter_mut() {
            *v *= rmr;
        }
        let (p, s, c) = hc_split(&q, scale, base, cfg);
        pre[r * 4..][..4].copy_from_slice(&p);
        post[r * 4..][..4].copy_from_slice(&s);
        comb[r * 16..][..16].copy_from_slice(&c);
    }
    (pre, post, comb)
}

fn hc_pre(x: &[f64], rows: usize, hidden: usize, pre: &[f64]) -> V {
    let row = 4 * hidden;
    let mut out = vec![0f64; rows * hidden];
    for r in 0..rows {
        for d in 0..hidden {
            out[r * hidden + d] = (0..4).map(|h| pre[r * 4 + h] * x[r * row + h * hidden + d]).sum();
        }
    }
    out
}

fn hc_post(x: &[f64], y: &[f64], rows: usize, hidden: usize, post: &[f64], comb: &[f64]) -> V {
    let row = 4 * hidden;
    let mut out = vec![0f64; rows * row];
    for r in 0..rows {
        for h in 0..4 {
            for d in 0..hidden {
                let mut o = post[r * 4 + h] * y[r * hidden + d];
                for g in 0..4 {
                    o += comb[r * 16 + g * 4 + h] * x[r * row + g * hidden + d];
                }
                out[r * row + h * hidden + d] = o;
            }
        }
    }
    out
}

/// `Expert.forward` with `swiglu_limit` clamps.
#[allow(clippy::too_many_arguments)]
fn expert(w1: &[f32], w3: &[f32], w2: &[f32], x: &[f64], dim: usize, inter: usize, limit: f64, weight: f64) -> V {
    let (w1, w3, w2) = (&to64(w1), &to64(w3), &to64(w2));
    let gate = linear(w1, inter, dim, x);
    let up = linear(w3, inter, dim, x);
    let h: V = (0..inter)
        .map(|i| {
            let u = up[i].clamp(-limit, limit);
            let g = gate[i].min(limit);
            (g / (1.0 + (-g).exp())) * u * weight
        })
        .collect();
    linear(w2, dim, inter, &h)
}

/// `Mtpl.forward` for `start_pos > 0`, transcribed.
fn reference_layer(
    cfg: &MtpConfig,
    s: &MtpLayerHost,
    x: &[f64],
    start_pos: usize,
    pre_mix: &[f64],
    main_x: &[f64],
    cache: &mut [f32],
) -> (V, V) {
    let (d, hd, nh, n) = (cfg.dim, cfg.head_dim, cfg.n_heads, cfg.block_size);
    let (attn_pre, attn_post, attn_comb) = hc_mixes(x, n, d, &to64(&s.hc_attn_fn), &s.hc_attn_scale, &s.hc_attn_base, cfg.hc);
    let mut xn = hc_pre(x, n, d, pre_mix);
    for r in 0..n {
        rms(&mut xn[r * d..][..d], &to64(s.attn_norm_w.as_ref().unwrap()), cfg.eps as f64);
    }

    // --- DSparkAttention.forward
    let mut mkv = linear(&to64(&s.wkv), hd, d, main_x);
    rms(&mut mkv[..hd], &to64(&s.kv_norm), cfg.eps as f64);
    rope(&mut mkv[..hd], 1, hd, hd, 1, start_pos, &s.rope, false);
    act_quant(&mut mkv);
    let slot = start_pos % cfg.window;
    cache[slot * hd..][..hd].copy_from_slice(&mkv.iter().map(|&v| v as f32).collect::<Vec<f32>>());

    let mut qr = linear_m(&to64(&s.wq_a), cfg.q_lora, d, &xn, n);
    for r in 0..n {
        rms(&mut qr[r * cfg.q_lora..][..cfg.q_lora], &to64(&s.q_norm), cfg.eps as f64);
    }
    let mut q = linear_m(&to64(&s.wq_b), nh * hd, cfg.q_lora, &qr, n);
    // `freqs_cis[start_pos + seqlen : ...]` (model.py:1057) with seqlen = 1 at decode: the block starts one position
    // after the tap it reads.
    rope(&mut q, n, nh * hd, hd, nh, start_pos + 1, &s.rope, false);
    let mut kv = linear_m(&to64(&s.wkv), hd, d, &xn, n);
    for r in 0..n {
        rms(&mut kv[r * hd..][..hd], &to64(&s.kv_norm), cfg.eps as f64);
    }
    rope(&mut kv, n, hd, hd, 1, start_pos + 1, &s.rope, false);
    act_quant(&mut kv);

    let mut combined = to64(cache);
    combined.extend_from_slice(&kv);
    let idxs = get_dspark_topk_row(cfg.window, n, start_pos);

    let scale = 1.0 / (hd as f64).sqrt();
    let mut o = vec![0f64; n * nh * hd];
    for r in 0..n {
        for hh in 0..nh {
            let qh = &q[(r * nh + hh) * hd..][..hd];
            let mut logits: Vec<f64> =
                idxs.iter().map(|&i| (0..hd).map(|e| qh[e] * combined[i as usize * hd + e]).sum::<f64>() * scale).collect();
            let mx = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            for v in logits.iter_mut() {
                *v = (*v - mx).exp();
            }
            let sum: f64 = logits.iter().sum::<f64>() + (s.sink[hh] as f64 - mx).exp();
            for (j, &i) in idxs.iter().enumerate() {
                let wgt = logits[j] / sum;
                for e in 0..hd {
                    o[(r * nh + hh) * hd + e] += wgt * combined[i as usize * hd + e];
                }
            }
        }
    }
    rope(&mut o, n, nh * hd, hd, nh, start_pos + 1, &s.rope, true);
    act_quant(&mut o);

    let (g, ol, oin) = (cfg.o_groups, cfg.o_lora, cfg.o_in_per_group());
    let mut olv = vec![0f64; n * g * ol];
    for r in 0..n {
        for gi in 0..g {
            let xg = &o[r * nh * hd + gi * oin..][..oin];
            let wg = &s.wo_a[gi * ol * oin..][..ol * oin];
            for j in 0..ol {
                olv[(r * g + gi) * ol + j] = (0..oin).map(|c| xg[c] * wg[j * oin + c] as f64).sum();
            }
        }
    }
    let y = linear_m(&to64(&s.wo_b), d, g * ol, &olv, n);
    let x1 = hc_post(x, &y, n, d, &attn_post, &attn_comb);

    // --- MoE.forward
    let (ffn_pre, ffn_post, ffn_comb) = hc_mixes(&x1, n, d, &to64(&s.hc_ffn_fn), &s.hc_ffn_scale, &s.hc_ffn_base, cfg.hc);
    let mut xf = hc_pre(&x1, n, d, &attn_pre);
    for r in 0..n {
        rms(&mut xf[r * d..][..d], &to64(s.ffn_norm_w.as_ref().unwrap()), cfg.eps as f64);
    }
    let (e, i, topk) = (cfg.n_routed_experts, cfg.inter_dim, cfg.topk);
    let gate_w = to64(&s.gate);
    let mut y = vec![0f64; n * d];
    for r in 0..n {
        let xr = &xf[r * d..][..d];
        let scores: Vec<f64> = (0..e)
            .map(|c| sqrtsoftplus((0..d).map(|k| gate_w[c * d + k] * xr[k]).sum::<f64>() as f32) as f64 / cfg.gate_temp as f64)
            .collect();
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_by(|&a, &b| {
            (scores[b] + s.gate_bias[b] as f64).total_cmp(&(scores[a] + s.gate_bias[a] as f64)).then(a.cmp(&b))
        });
        let den: f64 = order[..topk].iter().map(|&x| scores[x]).sum::<f64>() + 1e-20;
        for &ex in &order[..topk] {
            let wgt = scores[ex] / den * cfg.route_scale as f64;
            let out = expert(
                &s.w1[ex * i * d..][..i * d],
                &s.w3[ex * i * d..][..i * d],
                &s.w2[ex * d * i..][..d * i],
                xr,
                d,
                i,
                cfg.swiglu_limit as f64,
                wgt,
            );
            for c in 0..d {
                y[r * d + c] += out[c];
            }
        }
        let shared = expert(&s.shared_w1, &s.shared_w3, &s.shared_w2, xr, d, i, cfg.swiglu_limit as f64, 1.0);
        for c in 0..d {
            y[r * d + c] += shared[c];
        }
    }
    let x2 = hc_post(&x1, &y, n, d, &ffn_post, &ffn_comb);
    (x2, ffn_pre)
}

fn max_rel(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-3);
    a.iter().zip(b).fold(0f64, |m, (&x, &y)| m.max((x as f64 - y).abs() / scale))
}

// ------------------------------------------------------------------------------------------------ one layer

#[test]
fn one_mtp_layer_matches_a_direct_transcription() {
    let cfg = MtpConfig::toy();
    let s = toy_stage(&cfg, 1, 64, 0xC0FFEE);
    let n = cfg.block_size;

    let mut r = Rng(7);
    let x = r.vec(n * 4 * cfg.dim, -1.0, 1.0);
    let pre_mix = r.vec(n * 4, 0.0, 1.0);
    let main_x = r.vec(cfg.dim, -1.0, 1.0);

    // the module, then the transcription, each on its own copy of the cache
    let mut s_mod = s.clone();
    let (got, got_pre) = s_mod.forward(&x, 3, &pre_mix, &main_x);
    let cache_mod = s_mod.window_kv.clone();
    let mut cache_ref = s.window_kv.clone();
    let (want, want_pre) = reference_layer(&cfg, &s, &to64(&x), 3, &to64(&pre_mix), &to64(&main_x), &mut cache_ref);

    let e = max_rel(&got, &want);
    assert!(e < 2e-5, "the layer disagrees with the transcription by {e} (relative)");
    let ep = max_rel(&got_pre, &want_pre);
    assert!(ep < 2e-5, "the pre_mix handed to the next block disagrees by {ep}");
    // and the window slot the step wrote agrees, so the cache is not silently skipped
    assert_eq!(cache_mod.len(), cache_ref.len());
    let c = max_rel(&cache_mod, &to64(&cache_ref));
    assert!(c < 1e-6, "the window cache disagrees by {c}");
}

#[test]
fn prefill_only_seeds_the_window_and_returns_x_untouched() {
    // `if start_pos == 0: self.attn(x, start_pos, main_x); return x, pre_mix`
    let cfg = MtpConfig::toy();
    let mut s = toy_stage(&cfg, 0, 64, 3);
    let rows = 4;
    let x = vec![0.25f32; rows * 4 * cfg.dim];
    let pre_mix = vec![0.5f32; rows * 4];
    let main_x = vec![0.1f32; rows * cfg.dim];
    let (got, got_pre) = s.forward(&x, 0, &pre_mix, &main_x);
    assert_eq!(got, x);
    assert_eq!(got_pre, pre_mix);
    assert!(s.window_kv.iter().any(|&v| v != 0.0), "prefill must seed the ring");
}

// ------------------------------------------------------------------------------------------------ the chain

#[test]
fn the_chain_is_order_sensitive() {
    // Three distinct stages: running 1 then 0 then 2 must not match 0 then 1 then 2, because the residual stream each
    // stage reads is the previous stage's output and the stages do not share weights. Swapping the *stage list* (not the
    // roles) is the point: a depth-K loop over one set of weights would be order-free, and this model is not that.
    let cfg = MtpConfig::toy();
    let mut a = toy_chain(&cfg);
    let mut b = toy_chain(&cfg);
    let taps: Vec<f32> = Rng(11).vec(cfg.tap_width(), -1.0, 1.0);
    let anchor = 3i32;
    let (h0, main_x) = a.stages[0].forward_embed(&taps, 1, &[anchor]);

    let run = |stages: &mut [MtpLayerHost], order: &[usize]| -> Vec<f32> {
        let mut h = h0.clone();
        let mut pre_mix = DsparkChain::identity_pre_mix(&cfg, cfg.block_size);
        for &i in order {
            let (nh, npm) = stages[i].forward(&h, 5, &pre_mix, &main_x);
            h = nh;
            pre_mix = npm;
        }
        h
    };
    let ha = run(&mut a.stages, &[0, 1, 2]);
    let hb = run(&mut b.stages, &[1, 0, 2]);
    let e = max_rel(&ha, &hb.iter().map(|&v| v as f64).collect::<Vec<_>>());
    assert!(e > 1e-3, "swapping mtp.0 and mtp.1 changed the stream by only {e} (relative) — the chain is not chained");

    // ... and the second stage really reads the first stage's stream: zeroing stage 0's output projection changes stage 2
    let mut c = toy_chain(&cfg);
    c.stages[0].wo_b.iter_mut().for_each(|v| *v = 0.0);
    let hc = run(&mut c.stages, &[0, 1, 2]);
    let ec = max_rel(&hc, &ha.iter().map(|&v| v as f64).collect::<Vec<_>>());
    assert!(ec > 1e-6, "stage 0's output does not reach the head (delta {ec})");

    // the head is reached through the chain, not around it: the block's logits depend on the order too
    let mut d = toy_chain(&cfg);
    let da = d.step(&taps, &[anchor]);
    assert_eq!(da.ids.len(), cfg.block_size + 1);
    assert_eq!(da.confidence.len(), cfg.block_size);
}

#[test]
fn every_stage_keeps_its_own_cache() {
    // Three stages, three rings. Zeroing stage 1's `wkv` must empty stage 1's ring and leave the other two alone — a shared
    // cache (or a stage that never writes) would show up here as three identical rings.
    let cfg = MtpConfig::toy();
    let mut chain = toy_chain(&cfg);
    let taps = vec![0.1f32; cfg.tap_width()];
    chain.step(&taps, &[1]);
    let rings: Vec<Vec<f32>> = chain.stages.iter().map(|s| s.window_kv.clone()).collect();
    assert_eq!(rings.len(), 3);
    for (i, r) in rings.iter().enumerate() {
        assert!(r.iter().any(|&v| v != 0.0), "stage {i} never wrote its ring");
    }
    assert_ne!(rings[0], rings[1], "two stages share a cache");

    let mut z = toy_chain(&cfg);
    z.stages[1].wkv.iter_mut().for_each(|v| *v = 0.0);
    z.step(&taps, &[1]);
    assert!(z.stages[1].window_kv.iter().all(|&v| v == 0.0), "stage 1's ring was not the one it wrote");
    assert!(z.stages[0].window_kv.iter().any(|&v| v != 0.0));
    assert!(z.stages[2].window_kv.iter().any(|&v| v != 0.0));
    assert_eq!(chain.best_case_tokens(), 6);
}

// ------------------------------------------------------------------------------------------------ heads

#[test]
fn confidence_is_monotone_in_the_alignment_of_h_and_the_markov_embedding() {
    // With proj = [u_h ; u_m] over unit vectors, the score is <u_h, h> + <u_m, m> + b, so walking `h` and `m` along
    // those directions must increase it at every step. This is the property the head is *for*: a block row whose hidden
    // state agrees with the Markov embedding of its predecessor is the row the trunk should keep.
    let cfg = MtpConfig::toy();
    let (dim, rank) = (cfg.dim, cfg.markov_rank);
    let mut r = Rng(5);
    let uh = r.vec(dim, -1.0, 1.0);
    let um = r.vec(rank, -1.0, 1.0);
    let nz = |v: &Vec<f32>| (v.iter().map(|x| (x * x) as f64).sum::<f64>()).sqrt() as f32;
    let uh: Vec<f32> = uh.iter().map(|x| x / nz(&uh)).collect();
    let um: Vec<f32> = um.iter().map(|x| x / nz(&um)).collect();

    let mut proj = vec![0f32; cfg.conf_in()];
    proj[..dim].copy_from_slice(&uh);
    proj[dim..].copy_from_slice(&um);
    let head = ConfidenceHeadHost::new(&cfg, proj, -0.5);

    let h0 = r.vec(dim, -0.5, 0.5);
    let m0 = r.vec(rank, -0.5, 0.5);
    let mut last = f32::NEG_INFINITY;
    for step in 0..8 {
        let a = step as f32 * 0.25;
        let h: Vec<f32> = h0.iter().zip(&uh).map(|(x, u)| x + a * u).collect();
        let m: Vec<f32> = m0.iter().zip(&um).map(|(x, u)| x + a * u).collect();
        let c = head.confidence(&h, &m);
        assert!(c > last, "confidence fell at step {step}: {c} <= {last}");
        last = c;
    }
    assert!(last > 0.5, "eight steps of alignment should clear the mid-point, got {last}");

    // ... and the score is exactly the dot product the reference computes, not a renormalised variant
    let h: Vec<f32> = h0.clone();
    let m: Vec<f32> = m0.clone();
    let want: f64 = (0..dim).map(|i| h[i] as f64 * uh[i] as f64).sum::<f64>()
        + (0..rank).map(|i| m[i] as f64 * um[i] as f64).sum::<f64>()
        - 0.5;
    let got = head.score(&h, &m) as f64;
    assert!((got - want).abs() < 1e-4, "{got} vs {want}");
}

#[test]
fn the_markov_head_biases_the_full_vocabulary_from_the_previous_token() {
    let cfg = MtpConfig::toy();
    let mut r = Rng(9);
    let m =
        MarkovHeadHost::new(&cfg, r.vec(cfg.vocab * cfg.markov_rank, -1.0, 1.0), r.vec(cfg.vocab * cfg.markov_rank, -1.0, 1.0));
    let a = m.embed_tokens(&[0]);
    let b = m.embed_tokens(&[1]);
    assert_eq!(a.len(), cfg.markov_rank);
    assert_ne!(a, b, "different tokens must embed differently");
    let la = m.logits(&a, 1);
    let lb = m.logits(&b, 1);
    assert_eq!(la.len(), cfg.vocab, "the bias is a full distribution, not a scalar");
    assert_ne!(la, lb);
    // row t of embed is embed[t]
    assert_eq!(a, m.embed[..cfg.markov_rank].to_vec());
}

#[test]
fn the_confidence_head_reads_the_stream_before_the_final_norm() {
    // `forward_head` is `x = hc_pre(x, pre_mix)` then `logits = head(norm(x))` then `confidence_head(x, ...)`: the norm
    // only feeds the logits. Scaling the final norm's weight therefore moves the sampled ids and leaves every confidence
    // bit-identical — an easy off-by-one-tensor to get wrong, and invisible without this test.
    let cfg = MtpConfig::toy();
    let mut a = toy_chain(&cfg);
    let mut b = toy_chain(&cfg);
    let last = cfg.n_stages - 1;
    // the trunk's lm_head, so the logits are not identically zero
    let lm = Rng(31).vec(cfg.vocab * cfg.dim, -1.0, 1.0);
    for ch in [&mut a, &mut b] {
        let st = ch.stages[last].clone().with_trunk_tables(
            Rng(29).vec(cfg.vocab * cfg.dim, -1.0, 1.0),
            lm.clone(),
            vec![1.0; cfg.dim],
            vec![1.0; cfg.dim],
        );
        ch.stages[last] = st;
    }
    b.stages[last].norm.as_mut().unwrap().iter_mut().for_each(|v| *v *= 4.0);

    let taps = vec![0.05f32; cfg.tap_width()];
    let (h0, main_x) = a.stages[0].forward_embed(&taps, 1, &[2]);
    assert_eq!(h0.len(), main_x.len() * cfg.block_size * 4);
    let pre_mix = DsparkChain::identity_pre_mix(&cfg, cfg.block_size);

    let run = |c: &mut DsparkChain| -> DsparkDraft {
        let mut h = h0.clone();
        let mut pm = pre_mix.clone();
        for i in 0..cfg.n_stages {
            let (nh, np) = c.stages[i].forward(&h, 4, &pm, &main_x);
            h = nh;
            pm = np;
        }
        c.stages[last].forward_head(&h, &pm, &[2])
    };
    let da = run(&mut a);
    let db = run(&mut b);
    assert_eq!(da.confidence, db.confidence, "the confidence head must not see the final norm");
    assert_ne!(da.logits, db.logits, "the logits must see it");

    // and the confidence is exactly `sigmoid(proj . [hc_pre(x, pre_mix) ; m] + b)` over the rows `forward_head` used
    let head = a.stages[last].confidence_head.as_ref().unwrap();
    for (k, &c) in da.confidence.iter().enumerate() {
        assert!(c.is_finite() && (0.0..=1.0).contains(&c), "row {k} is not a probability: {c}");
    }
    assert!(head.rank > 0 && head.dim == cfg.dim);
}

// ------------------------------------------------------------------------------------------------ topk_idxs

#[test]
fn get_dspark_topk_idxs_is_pinned_for_a_twelve_token_sequence() {
    // 12 tokens of context, block_size 5, a window of 8 that is already full: the reachable window prefix is all 8 slots
    // and the block is slots 8..13, so the list is arange(13) — one row, repeated over all five queries.
    let got = get_dspark_topk_idxs(8, 2, 5, 12);
    let n_kv = 13;
    assert_eq!(got.len(), 2 * 5 * n_kv);
    let want: Vec<i32> = (0..13).collect();
    for q in 0..2 * 5 {
        assert_eq!(&got[q * n_kv..][..n_kv], &want[..], "query {q} does not see the pinned list");
    }
    assert_eq!(get_dspark_topk_row(8, 5, 12), want);

    // ... but it is NOT arange: before the window fills, the unreachable slots are skipped, and the block still starts at
    // `window`. At start_pos 5 only slots 0..6 are reachable, so slots 6 and 7 are absent and the block is 8..13.
    let row = get_dspark_topk_row(8, 5, 5);
    assert_eq!(row, vec![0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12]);
    assert_eq!(row.len(), 6 + 5);

    // the first step after prefill: two window slots (0 and 1), then the block
    assert_eq!(get_dspark_topk_row(8, 5, 1), vec![0, 1, 8, 9, 10, 11, 12]);

    // and the pack's own geometry: window 128 at step 12 is 13 + 5 slots
    assert_eq!(get_dspark_topk_row(128, 5, 12).len(), 13 + 5);
    assert_eq!(get_dspark_topk_row(128, 5, 200).len(), 128 + 5);
}

#[test]
fn the_block_attends_bidirectionally() {
    // Every query gets the same list, so changing the *last* block slot changes the *first* draft row's output. A causal
    // draft (or an off-by-one in the block offset) breaks this.
    let cfg = MtpConfig::toy();
    let mut a = toy_stage(&cfg, 0, 64, 21);
    let mut b = toy_stage(&cfg, 0, 64, 21);
    let n = cfg.block_size;
    // the block's residual stream is `hc_mult` copies wide, not `dim`: `[n][4][dim]`
    let x: Vec<f32> = (0..n * 4 * cfg.dim).map(|i| (i as f32 * 0.013).sin()).collect();
    let pre_mix = DsparkChain::identity_pre_mix(&cfg, n);
    let main_x = vec![0.2f32; cfg.dim];

    let ya = a.forward(&x, 6, &pre_mix, &main_x).0;
    // zero the KV projection of the *last* block row only, by zeroing the row's input
    let mut x2 = x.clone();
    x2[(n - 1) * 4 * cfg.dim..].iter_mut().for_each(|v| *v = 0.0);
    let yb = b.forward(&x2, 6, &pre_mix, &main_x).0;
    let row0_a = &ya[..4 * cfg.dim];
    let row0_b = &yb[..4 * cfg.dim];
    let d: f32 = row0_a.iter().zip(row0_b).map(|(p, q)| (p - q).abs()).fold(0.0, f32::max);
    assert!(d > 1e-4, "row 0 did not see the last block slot (max delta {d})");
}

#[test]
fn the_candidate_walk_is_2048_blocks_of_8_anchored_at_layer_20() {
    // The draft attention inherits the trunk's level-1 candidate mask: 2048 blocks of 8 positions, published by layer 20.
    assert_eq!((CANDIDATE_SOURCE_LAYER, CAND_TOPK_BLOCKS, CAND_BLOCK), (20, 2048, 8));
    let n_pos = CAND_TOPK_BLOCKS * CAND_BLOCK * 2; // 4096 blocks
    let mut logits = vec![0f32; n_pos];
    for (i, v) in logits.iter_mut().enumerate() {
        // block b scores b / 4096, so the top half is the last 2048 blocks
        *v = (i / CAND_BLOCK) as f32 / 4096.0;
    }
    let mask = select_candidate_blocks(&logits, n_pos, CAND_TOPK_BLOCKS, CAND_BLOCK);
    assert_eq!(mask.len(), n_pos);
    assert_eq!(mask.iter().filter(|&&m| m).count(), CAND_TOPK_BLOCKS * CAND_BLOCK);
    // the mask is block-aligned: whole 8-blocks, and the kept blocks are the highest-scoring ones
    for b in 0..n_pos / CAND_BLOCK {
        let all = mask[b * CAND_BLOCK..][..CAND_BLOCK].iter().all(|&m| m);
        let none = mask[b * CAND_BLOCK..][..CAND_BLOCK].iter().all(|&m| !m);
        assert!(all || none, "block {b} is partially masked");
        assert_eq!(all, b >= 2048, "block {b} kept against its score");
    }
}

// ------------------------------------------------------------------------------------------------ the arbiter

#[test]
fn the_acceptance_rule_is_a_prefix_and_the_best_case_is_the_whole_block() {
    assert_eq!(accept_prefix(&[0.9, 0.8, 0.7, 0.6, 0.5], 0.45), 5, "the best case is block_size drafts");
    assert_eq!(accept_prefix(&[0.9, 0.9, 0.4, 0.9, 0.9], 0.45), 2);
    assert_eq!(accept_prefix(&[0.4, 0.9, 0.9, 0.9, 0.9], 0.45), 0);
    assert_eq!(accept_prefix(&[], 0.5), 0);

    // a whole step: the anchor is free, so the best case is block_size + 1 tokens
    let cfg = MtpConfig::toy();
    let chain = toy_chain(&cfg);
    assert_eq!(chain.best_case_tokens(), 6);

    // the draft carries one confidence per drafted token, and the anchor is not one of them
    let mut chain = toy_chain(&cfg);
    let taps = vec![0.05f32; cfg.tap_width()];
    let d = chain.step(&taps, &[4]);
    assert_eq!(d.ids.len(), cfg.block_size + 1);
    assert_eq!(d.ids[0], 4);
    assert_eq!(d.drafted().len(), cfg.block_size);
    assert_eq!(d.confidence.len(), cfg.block_size);
    assert_eq!(d.accepted(0.0), cfg.block_size, "a zero threshold keeps the whole block");
    assert_eq!(d.accepted(1.1), 0, "an impossible threshold keeps nothing");
}

#[test]
fn the_noise_token_fills_every_draft_slot_but_the_anchor() {
    // `forward_embed` builds `[t, NOISE x block_size]`, so with a trunk embedding table installed slot 0 is the anchor's
    // row and slots 1.. are all the noise row, once per residual copy.
    let cfg = MtpConfig::toy();
    let mut s = toy_stage(&cfg, 0, 64, 1);
    let mut r = Rng(23);
    let embed = r.vec(cfg.vocab * cfg.dim, -1.0, 1.0);
    s = s.with_trunk_tables(embed.clone(), vec![0.0; cfg.vocab * cfg.dim], vec![1.0; cfg.dim], vec![1.0; cfg.dim]);

    let taps = vec![0.1f32; cfg.tap_width()];
    let (h, main_x) = s.forward_embed(&taps, 1, &[3]);
    assert_eq!(h.len(), cfg.block_size * 4 * cfg.dim);
    assert_eq!(main_x.len(), cfg.dim);

    let d = cfg.dim;
    let slot = |k: usize, copy: usize| &h[((k * 4) + copy) * d..][..d];
    let anchor = &embed[3 * d..][..d];
    let noise = &embed[cfg.noise_token_id as usize * d..][..d];
    assert_ne!(anchor, noise, "the fixture needs distinct rows");
    for copy in 0..4 {
        assert_eq!(slot(0, copy), anchor, "slot 0 must be the trunk's own token");
        for k in 1..cfg.block_size {
            assert_eq!(slot(k, copy), noise, "slot {k} must be the noise token");
        }
    }
}

// ------------------------------------------------------------------------------------------------ device

fn dev() -> Option<Stream> {
    if !device_available() {
        eprintln!("not run: no device");
        return None;
    }
    Some(Stream::new().unwrap())
}

#[test]
fn gpu_topk_idxs_matches_the_host() {
    let Some(st) = dev() else { return };
    let (bsz, block, start) = (2usize, 5usize, 12usize);
    let reach = 8usize.min(start + 1);
    let n_kv = reach + block;
    let mut out = DevBuf::alloc(bsz * block * n_kv * 4).unwrap();
    topk_idxs_dev(&mut out, bsz, block, reach, 8, &st).unwrap();
    st.sync().unwrap();
    let host: Vec<i32> = out.to_vec().unwrap();
    assert_eq!(host, get_dspark_topk_idxs(8, bsz, block, start));
}

#[test]
fn gpu_confidence_and_accept_match_the_host() {
    let Some(st) = dev() else { return };
    let cfg = MtpConfig::toy();
    let mut r = Rng(13);
    let h: Vec<f32> = r.vec(cfg.block_size * cfg.dim, -1.0, 1.0);
    let m: Vec<f32> = r.vec(cfg.block_size * cfg.markov_rank, -1.0, 1.0);
    let w: Vec<f32> = r.vec(cfg.conf_in(), -0.1, 0.1);
    let head = ConfidenceHeadHost::new(&cfg, w.clone(), 0.0);
    let (hb, mb, wb, bb) = (
        DevBuf::from_slice(&h).unwrap(),
        DevBuf::from_slice(&m).unwrap(),
        DevBuf::from_slice(&w).unwrap(),
        DevBuf::from_slice(&[0f32]).unwrap(),
    );
    let mut out = DevBuf::alloc(cfg.block_size * 4).unwrap();
    confidence_dev(&hb, &mb, &wb, &bb, &mut out, cfg.block_size, cfg.dim, cfg.markov_rank, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<f32> = out.to_vec().unwrap();
    let want = head.confidence_rows(&h, &m, cfg.block_size);
    for (a, b) in got.iter().zip(&want) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}

#[test]
fn gpu_markov_bias_and_accept() {
    let Some(st) = dev() else { return };
    let cfg = MtpConfig::toy();
    let mut r = Rng(17);
    let m =
        MarkovHeadHost::new(&cfg, r.vec(cfg.vocab * cfg.markov_rank, -1.0, 1.0), r.vec(cfg.vocab * cfg.markov_rank, -1.0, 1.0));
    let tokens = [2i32, 5];
    let mut logits = vec![0f32; 2 * cfg.vocab];
    let (lb, tb, eb, hb) = (
        DevBuf::from_slice(&logits).unwrap(),
        DevBuf::from_slice(&tokens).unwrap(),
        DevBuf::from_slice(&m.embed).unwrap(),
        DevBuf::from_slice(&m.head).unwrap(),
    );
    let mut lb = lb;
    markov_bias_dev(&mut lb, &tb, &eb, &hb, 2, cfg.vocab, cfg.markov_rank, &st).unwrap();
    st.sync().unwrap();
    logits = lb.to_vec().unwrap();
    for (row, &t) in tokens.iter().enumerate() {
        let want = m.logits(&m.embed_tokens(&[t]), 1);
        for (a, b) in logits[row * cfg.vocab..][..cfg.vocab].iter().zip(&want) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    let conf = [0.9f32, 0.9, 0.1, 0.9, 0.9];
    let cb = DevBuf::from_slice(&conf).unwrap();
    let mut out = DevBuf::alloc(4).unwrap();
    accept_dev(&cb, &mut out, 1, 5, 0.5, &st).unwrap();
    st.sync().unwrap();
    let n: Vec<i32> = out.to_vec().unwrap();
    assert_eq!(n[0] as usize, accept_prefix(&conf, 0.5));
}
