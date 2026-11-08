//! The MTP (multi-token prediction) head of Nemotron 3.5 Super VL, as an fp32 CPU reference.
//!
//! The head is one block pair `*E` (`mtp_layers_block_type`) that drafts one token ahead of the trunk. It is built from the two
//! references that already exist, so it adds only its own wiring: the attention block is [`attn_ref::attention_block`] on the head's
//! own [`KvCache`], and the MoE block is [`moe_ref::latent_moe`]. The dense maps (`eh_proj`, the latent projections, the shared expert,
//! and the logits) run through [`moe_ref::Dense`], so there is one dense arithmetic in the family's references.
//!
//! ```text
//! e  = enorm(embed(next_token))           // RMSNorm, eps 1e-5
//! hn = hnorm(h)                           // h is the trunk's post-norm_f hidden state
//! x  = eh_proj(cat([e, hn]))              // [4096, 8192]; the embedding is the FIRST half. x is the head's residual stream
//! x  = x + attn(norm(x))                  // q/k/v/o, NoPE, 32 heads, 2 KV heads, head dim 128, the head's own KV cache
//! x  = x + latent_moe(norm(x))            // router, latent width 1024, shared expert, bf16 experts
//! out = final_layernorm(x)                // the head's output hidden; logits = lm_head(out) with the TRUNK's lm_head
//! ```
//!
//! One fusion row `r` of a call pairs the embedding of the token that follows trunk position `p` with the trunk's post-`norm_f`
//! hidden state at `p`; its output scores the token at `p + 2`. Pairing the rows is the caller's job: [`forward`] sees only the
//! two row blocks. Rows of one call are consecutive cache positions of one sequence and are causal among themselves, and across
//! calls through the cache, so feeding rows one at a time equals one call (bit for bit, as in `attn_ref`).
//!
//! What is pinned by the pack and what is not. The pack's tensor inventory ([`super::weights::units`]) fixes every name and shape
//! here ([`tensor_shapes`] is checked against it), and it holds no embedding and no `lm_head` of the head's own, so the head reuses
//! the trunk's (`tie_word_embeddings` is false, so the `lm_head` is not the embedding either). The order of the concat, the norms'
//! placement and the `eps` of 1e-5 come from the task statement and the HF structure it describes; they were not checked against the
//! HF code or a real-weight dump. The audit file `mtp-reference.md` was not available when this was written.
//!
//! Precision: fp32 throughout, like the other references. Weights are the pack's bf16 values widened to f32 by the caller (exact).
//! The HF bf16 model rounds at its own points, which are not modelled, so a real-weight fixture is judged with a tolerance.
//! The RMSNorm sum of squares is sequential here and a fixed tree on the GPU (`csrc/nemotron_glue.cu`); the two agree to fp32
//! accumulation order, not bitwise. The attention keeps `attn_ref`'s roundings (K and V rounded once to fp16 at the cache write,
//! P once per 64-key tile), so a fixture of the head sees the same cache dtype the GPU stores.

use super::attn_ref::{self, Geom, KvCache};
use super::config::OmniConfig;
use super::moe_ref::{self, Dense, Mlp, Proj, Router};
use super::weights::LM;
use std::collections::BTreeSet;

/// The RMSNorm epsilon of every norm in the head (`layer_norm_epsilon` of the published config).
pub const RMS_EPS: f32 = 1e-5;

/// The shape of the head. The attention part is [`Geom`], whose `hidden` is also the fusion width and the residual width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtpGeom {
    pub attn: Geom,
    /// Routed experts of the MoE block (512 on the pack).
    pub experts: usize,
    /// Experts selected per token (22 on the pack).
    pub topk: usize,
    /// The latent width the routed experts run in (1024).
    pub latent: usize,
    /// A routed expert's inner width (2688).
    pub expert_inter: usize,
    /// The shared expert's inner width (5376).
    pub shared_inter: usize,
    /// `routed_scaling_factor` (5.0).
    pub routed_scale: f32,
    /// The RMSNorm epsilon, [`RMS_EPS`] on the pack.
    pub eps: f32,
}

impl MtpGeom {
    /// The measured head.
    pub const OMNI: MtpGeom = MtpGeom {
        attn: Geom::OMNI_STAR,
        experts: 512,
        topk: 22,
        latent: 1024,
        expert_inter: 2688,
        shared_inter: 5376,
        routed_scale: 5.0,
        eps: RMS_EPS,
    };

    /// The residual width, which is also half the width of `eh_proj`'s input.
    pub fn hidden(&self) -> usize {
        self.attn.hidden
    }

    /// The head's geometry as the parsed config states it (the trunk's MoE geometry is the head's).
    pub fn from_config(cfg: &OmniConfig) -> Self {
        Self {
            attn: Geom { hidden: cfg.hidden, n_heads: cfg.heads, n_kv: cfg.kv_heads, head_dim: cfg.head_dim },
            experts: cfg.experts,
            topk: cfg.topk,
            latent: cfg.latent,
            expert_inter: cfg.expert_inter,
            shared_inter: cfg.shared_inter,
            routed_scale: cfg.routed_scaling as f32,
            eps: cfg.eps as f32,
        }
    }
}

/// One routed expert's projections in the pack's `Linear` layout: `up` is `[expert_inter, latent]` and `down` is `[latent, expert_inter]`.
#[derive(Clone, Debug)]
pub struct ExpertWeights {
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

/// The head's weights, f32, in the pack's `[out, in]` row-major layout. The doc of each field names its tensor under
/// `language_model.mtp.layers.0.` (the attention block `*`) or `.1.` (the MoE block `E`).
#[derive(Clone, Debug)]
pub struct MtpWeights {
    pub geom: MtpGeom,
    /// `0.eh_proj.weight`, `[hidden, 2 * hidden]`: the fusion of `cat([enorm(embed), hnorm(h)])`.
    pub eh_proj: Vec<f32>,
    /// `0.enorm.weight`, `[hidden]`: the norm of the embedding half.
    pub enorm: Vec<f32>,
    /// `0.hnorm.weight`, `[hidden]`: the norm of the trunk hidden half.
    pub hnorm: Vec<f32>,
    /// `0.norm.weight`, `[hidden]`: the attention block's pre-norm.
    pub attn_norm: Vec<f32>,
    /// `0.mixer.q_proj.weight`, `[q_dim, hidden]`.
    pub wq: Vec<f32>,
    /// `0.mixer.k_proj.weight`, `[kv_dim, hidden]`.
    pub wk: Vec<f32>,
    /// `0.mixer.v_proj.weight`, `[kv_dim, hidden]`.
    pub wv: Vec<f32>,
    /// `0.mixer.o_proj.weight`, `[hidden, q_dim]`.
    pub wo: Vec<f32>,
    /// `1.norm.weight`, `[hidden]`: the MoE block's pre-norm.
    pub moe_norm: Vec<f32>,
    /// `1.mixer.gate.weight`, `[experts, hidden]` (the router, fp32 in the reference).
    pub gate: Vec<f32>,
    /// `1.mixer.gate.e_score_correction_bias`, `[experts]`.
    pub gate_bias: Vec<f32>,
    /// `1.mixer.fc1_latent_proj.weight`, `[latent, hidden]`.
    pub fc1: Vec<f32>,
    /// `1.mixer.fc2_latent_proj.weight`, `[hidden, latent]`.
    pub fc2: Vec<f32>,
    /// `1.mixer.shared_experts.up_proj.weight`, `[shared_inter, hidden]`.
    pub shared_up: Vec<f32>,
    /// `1.mixer.shared_experts.down_proj.weight`, `[hidden, shared_inter]`.
    pub shared_down: Vec<f32>,
    /// `1.mixer.experts.{e}.up_proj.weight` and `.down_proj.weight`: one slot per routed expert. As in [`moe_ref::latent_moe`], a slot
    /// may be `None` when no row of the call selects that expert, so a fixture loads only the experts it needs. A selected expert that
    /// is `None` panics by number.
    pub experts: Vec<Option<ExpertWeights>>,
    /// `1.final_layernorm.weight`, `[hidden]`: the head's output norm.
    pub final_norm: Vec<f32>,
}

/// Every tensor of the head, by the pack's name and shape, in the order of [`super::weights::units`] (block 0, block 1, then the
/// experts' up and down projections). The experts are all listed, whatever the caller loaded.
pub fn tensor_shapes(g: &MtpGeom) -> Vec<(String, Vec<usize>)> {
    let (h, a) = (g.hidden(), &g.attn);
    let (b0, b1) = (format!("{LM}.mtp.layers.0"), format!("{LM}.mtp.layers.1"));
    let mut out = vec![
        (format!("{b0}.eh_proj.weight"), vec![h, 2 * h]),
        (format!("{b0}.enorm.weight"), vec![h]),
        (format!("{b0}.hnorm.weight"), vec![h]),
        (format!("{b0}.norm.weight"), vec![h]),
        (format!("{b0}.mixer.q_proj.weight"), vec![a.q_dim(), h]),
        (format!("{b0}.mixer.k_proj.weight"), vec![a.kv_dim(), h]),
        (format!("{b0}.mixer.v_proj.weight"), vec![a.kv_dim(), h]),
        (format!("{b0}.mixer.o_proj.weight"), vec![h, a.q_dim()]),
        (format!("{b1}.final_layernorm.weight"), vec![h]),
        (format!("{b1}.norm.weight"), vec![h]),
        (format!("{b1}.mixer.gate.weight"), vec![g.experts, h]),
        (format!("{b1}.mixer.gate.e_score_correction_bias"), vec![g.experts]),
        (format!("{b1}.mixer.fc1_latent_proj.weight"), vec![g.latent, h]),
        (format!("{b1}.mixer.fc2_latent_proj.weight"), vec![h, g.latent]),
        (format!("{b1}.mixer.shared_experts.up_proj.weight"), vec![g.shared_inter, h]),
        (format!("{b1}.mixer.shared_experts.down_proj.weight"), vec![h, g.shared_inter]),
    ];
    for e in 0..g.experts {
        out.push((format!("{b1}.mixer.experts.{e}.up_proj.weight"), vec![g.expert_inter, g.latent]));
        out.push((format!("{b1}.mixer.experts.{e}.down_proj.weight"), vec![g.latent, g.expert_inter]));
    }
    out
}

impl MtpWeights {
    /// The tensors the struct holds, by the pack's name, in the order of [`tensor_shapes`]. An unloaded expert is `None`.
    pub fn tensors(&self) -> Vec<(String, Option<&[f32]>)> {
        fn some(v: &[f32]) -> Option<&[f32]> {
            Some(v)
        }
        let shapes = tensor_shapes(&self.geom);
        let mut data: Vec<Option<&[f32]>> = vec![
            some(&self.eh_proj),
            some(&self.enorm),
            some(&self.hnorm),
            some(&self.attn_norm),
            some(&self.wq),
            some(&self.wk),
            some(&self.wv),
            some(&self.wo),
            some(&self.final_norm),
            some(&self.moe_norm),
            some(&self.gate),
            some(&self.gate_bias),
            some(&self.fc1),
            some(&self.fc2),
            some(&self.shared_up),
            some(&self.shared_down),
        ];
        for slot in &self.experts {
            data.push(slot.as_ref().map(|x| x.up.as_slice()));
            data.push(slot.as_ref().map(|x| x.down.as_slice()));
        }
        shapes.into_iter().zip(data).map(|((name, _), d)| (name, d)).collect()
    }

    /// Whether every held tensor has the number of values its shape needs. The refusal names the pack tensor.
    pub fn check(&self) -> Result<(), String> {
        let g = &self.geom;
        if self.experts.len() != g.experts {
            return Err(format!(
                "MTP head: {} expert slots are held, the geometry has {} routed experts",
                self.experts.len(),
                g.experts
            ));
        }
        if g.topk == 0 || g.topk > g.experts {
            return Err(format!("MTP head: topk {} is outside 1..={}", g.topk, g.experts));
        }
        for ((name, data), (_, shape)) in self.tensors().into_iter().zip(tensor_shapes(g)) {
            let want: usize = shape.iter().product();
            if let Some(d) = data {
                if d.len() != want {
                    return Err(format!("MTP head: `{name}` holds {} values, its shape {shape:?} needs {want}", d.len()));
                }
            }
        }
        Ok(())
    }
}

/// The head's own KV cache: the keys and values of its attention layer, one row per fusion row. They are separate from the trunk's
/// (`tools/attn_gpu_design.md`, decision 4): their keys come from `eh_proj` of the fusion rows, a different sequence from the trunk's.
/// The values are fp16 held as f32, as `attn_ref::KvCache` stores them.
///
/// `Clone` is the rollback: a caller that speculates on top of the cache keeps a clone and puts it back. `attn_ref::KvCache` has no
/// in-place truncate, and this file may not add one.
#[derive(Clone, Debug)]
pub struct MtpKv {
    attn: Geom,
    cache: KvCache,
}

impl MtpKv {
    pub fn new(g: &MtpGeom) -> Self {
        Self { attn: g.attn, cache: KvCache::new(&g.attn) }
    }

    /// The number of cached fusion rows.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// All cached keys, `[len, n_kv * head_dim]`.
    pub fn keys(&self) -> &[f32] {
        self.cache.keys()
    }

    /// All cached values, `[len, n_kv * head_dim]`.
    pub fn values(&self) -> &[f32] {
        self.cache.values()
    }

    /// What one token's K and V take in the cache dtype (fp16): K and V, `n_kv * head_dim` each, 2 bytes. 1 KiB on the pack.
    pub fn bytes_per_token(g: &Geom) -> usize {
        2 * g.kv_dim() * 2
    }

    /// What the cached rows take in the cache dtype, `len * bytes_per_token`. (The host holds them as f32.)
    pub fn bytes(&self) -> usize {
        self.len() * Self::bytes_per_token(&self.attn)
    }
}

/// RMSNorm over rows of `w.len()` values: `out = (x * inv) * w` with `inv = 1 / sqrt(mean(x^2) + eps)`, in f32. This is the
/// expression of `nemotron_glue::rmsnorm_bf16w_f32`; the sum of squares is sequential here and a fixed tree there. The norm of a zero
/// row is zero (the `eps` keeps `inv` finite).
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = w.len();
    assert!(d > 0 && x.len() % d == 0, "rms_norm: input is not whole rows of {d}");
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(d) {
        let mut sq = 0f32;
        for &v in row {
            sq += v * v;
        }
        let mean = sq / d as f32;
        let inv = 1.0 / (mean + eps).sqrt();
        out.extend(row.iter().zip(w).map(|(&v, &wc)| (v * inv) * wc));
    }
    out
}

/// The rows of the trunk's embedding table (`[vocab, hidden]`, f32) for `tokens`: the head's `embed(next_token)`.
pub fn embed_rows(table: &[f32], hidden: usize, tokens: &[u32]) -> Vec<f32> {
    assert!(hidden > 0 && table.len() % hidden == 0, "the embedding table is not whole rows of {hidden}");
    let vocab = table.len() / hidden;
    let mut out = Vec::with_capacity(tokens.len() * hidden);
    for &t in tokens {
        assert!((t as usize) < vocab, "token {t} is outside the vocabulary of {vocab}");
        out.extend_from_slice(&table[t as usize * hidden..(t as usize + 1) * hidden]);
    }
    out
}

/// The residual stream after each stage of the head, for `t` rows (each `[t, hidden]`). The stages are the points a real-weight
/// fixture or a GPU layer can be compared at.
#[derive(Clone, Debug)]
pub struct MtpTaps {
    /// `eh_proj(cat([enorm(embed), hnorm(h)]))`: the head's input residual.
    pub fused: Vec<f32>,
    /// `fused + attention(attn_norm(fused))`.
    pub after_attn: Vec<f32>,
    /// `after_attn + latent_moe(moe_norm(after_attn))`.
    pub after_moe: Vec<f32>,
    /// `final_layernorm(after_moe)`: the head's output hidden, which the trunk's `lm_head` scores.
    pub out: Vec<f32>,
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "residual add: lengths differ");
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// The MoE block on its normed input. The router picks the experts first, and only those experts are widened and handed to
/// `latent_moe`, so a weight set that holds all 512 experts does not transpose all of them per call. The selection is the one
/// `latent_moe` makes (same router, same function), so a selected expert that was not loaded still panics there, by number.
fn moe_block(w: &MtpWeights, m_in: &[f32]) -> Vec<f32> {
    let g = &w.geom;
    let h = g.hidden();
    let rows = m_in.len() / h;
    let router = Router { gate: w.gate.clone(), bias: w.gate_bias.clone(), hidden: h, topk: g.topk, scale: g.routed_scale };
    let logits = router.logits(m_in, rows);
    let mut needed = BTreeSet::new();
    for r in 0..rows {
        let (idx, _) = moe_ref::route(&logits[r * g.experts..(r + 1) * g.experts], &router.bias, router.topk, router.scale);
        needed.extend(idx);
    }
    let experts: Vec<Option<Mlp>> = w
        .experts
        .iter()
        .enumerate()
        .map(|(e, slot)| match slot {
            Some(x) if needed.contains(&e) => Some(Mlp {
                up: Box::new(Dense::from_hf(g.expert_inter, g.latent, &x.up)),
                down: Box::new(Dense::from_hf(g.latent, g.expert_inter, &x.down)),
            }),
            _ => None,
        })
        .collect();
    let fc1 = Dense::from_hf(g.latent, h, &w.fc1);
    let fc2 = Dense::from_hf(h, g.latent, &w.fc2);
    let shared = Mlp {
        up: Box::new(Dense::from_hf(g.shared_inter, h, &w.shared_up)),
        down: Box::new(Dense::from_hf(h, g.shared_inter, &w.shared_down)),
    };
    moe_ref::latent_moe(m_in, &router, &fc1, &fc2, &experts, &shared)
}

/// The head over `t` fusion rows, returning every stage ([`MtpTaps`]).
///
/// `embed_rows` is `[t, hidden]`: the raw (not yet normed) embeddings of the next tokens. `hidden_rows` is `[t, hidden]`: the trunk's
/// hidden states after `norm_f`, one per row (the same tap the trunk's `lm_head` reads; the head norms it again with `hnorm`).
/// `positions[i]` is the cache slot of row `i`: it must be `kv.len() + i`. The attention has no rotary embedding, so a position is
/// never an input of the arithmetic; it only states where the row lands, and a gap or a repeat is refused instead of silently
/// reordering the cache. Rows are appended to `kv` and attend over the whole cache causally.
pub fn forward_taps(w: &MtpWeights, embed_rows: &[f32], hidden_rows: &[f32], positions: &[usize], kv: &mut MtpKv) -> MtpTaps {
    if let Err(e) = w.check() {
        panic!("{e}");
    }
    let g = &w.geom;
    let h = g.hidden();
    assert!(embed_rows.len() % h == 0, "embed_rows is not whole rows of {h}");
    assert_eq!(hidden_rows.len(), embed_rows.len(), "embed_rows and hidden_rows must hold the same rows");
    let t = embed_rows.len() / h;
    assert_eq!(positions.len(), t, "one position per fusion row");
    assert_eq!(kv.attn, g.attn, "the KV cache geometry differs from the head");
    for (i, &p) in positions.iter().enumerate() {
        assert_eq!(
            p,
            kv.len() + i,
            "fusion row {i}: position {p}, but the head's cache holds {} rows (rows are consecutive)",
            kv.len()
        );
    }
    if t == 0 {
        let empty = Vec::new();
        return MtpTaps { fused: empty.clone(), after_attn: empty.clone(), after_moe: empty.clone(), out: empty };
    }

    // the embedding is the first half of every fused row, the trunk hidden state the second
    let e = rms_norm(embed_rows, &w.enorm, g.eps);
    let n = rms_norm(hidden_rows, &w.hnorm, g.eps);
    let mut cat = Vec::with_capacity(t * 2 * h);
    for r in 0..t {
        cat.extend_from_slice(&e[r * h..(r + 1) * h]);
        cat.extend_from_slice(&n[r * h..(r + 1) * h]);
    }
    let fused = Dense::from_hf(h, 2 * h, &w.eh_proj).apply(&cat, t);

    let a_in = rms_norm(&fused, &w.attn_norm, g.eps);
    let aw = attn_ref::Weights { geom: g.attn, wq: &w.wq, wk: &w.wk, wv: &w.wv, wo: &w.wo };
    let after_attn = add(&fused, &attn_ref::attention_block(&a_in, &aw, &mut kv.cache));

    let m_in = rms_norm(&after_attn, &w.moe_norm, g.eps);
    let after_moe = add(&after_attn, &moe_block(w, &m_in));

    let out = rms_norm(&after_moe, &w.final_norm, g.eps);
    MtpTaps { fused, after_attn, after_moe, out }
}

/// The head's output hidden over `t` fusion rows, `[t, hidden]`. See [`forward_taps`] for the inputs.
pub fn forward(w: &MtpWeights, embed_rows: &[f32], hidden_rows: &[f32], positions: &[usize], kv: &mut MtpKv) -> Vec<f32> {
    forward_taps(w, embed_rows, hidden_rows, positions, kv).out
}

/// A head projection the pack stores as EXL3 (the 4-bit head), decoded on the host into the f32 `[n, k]` (HF `[out, in]`) layout
/// [`MtpWeights`] holds: `gb_exl3::decode_weight` (the host EXL3 path: `forward`'s weight, to float64 rounding) rounded once to f32. The
/// reference then runs the quantized head through the same dense maps as the bf16 one. `trellis` is `[k/16, n/16, 16 * bits]`, `suh` and
/// `svh` the scales as stored (fp16, widened exactly).
pub fn hf_from_exl3(
    trellis: &[i16],
    k: usize,
    n: usize,
    bits: gb_exl3::Bits,
    codebook: gb_exl3::Codebook,
    suh: &[f64],
    svh: &[f64],
) -> Vec<f32> {
    let w = gb_exl3::decode_weight(trellis, k / 16, n / 16, bits, &gb_exl3::codebook(codebook), suh, svh);
    let mut out = vec![0f32; n * k];
    for (i, row) in w.chunks_exact(n).enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[j * k + i] = v as f32;
        }
    }
    out
}

/// The head's logits: `hidden_rows` (`[t, hidden]`, the output of [`forward`]) through `lm_head` (`[vocab, hidden]`). The head has no
/// output projection of its own in the pack, so this is the TRUNK's `lm_head`, the one the trunk's `norm_f` output meets.
pub fn logits(hidden_rows: &[f32], lm_head: &[f32], hidden: usize) -> Vec<f32> {
    assert!(hidden > 0 && lm_head.len() % hidden == 0, "lm_head is not whole rows of {hidden}");
    assert!(hidden_rows.len() % hidden == 0, "hidden_rows is not whole rows of {hidden}");
    let vocab = lm_head.len() / hidden;
    Dense::from_hf(vocab, hidden, lm_head).apply(hidden_rows, hidden_rows.len() / hidden)
}

#[cfg(test)]
mod tests {
    use super::super::weights::{units, Storages, UnitId};
    use super::*;

    /// The published pack's `config.json`, committed as a fixture (`testdata/nemotron_omni/config.json`).
    const PACK: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");

    /// The measured attention heads (32, 2 KV, 128) with a small residual width and a scaled-down MoE, through the same code path.
    /// Projections cost `hidden` per output and the unoptimized test build is slow; the head's wiring is unchanged.
    fn small() -> MtpGeom {
        MtpGeom {
            attn: Geom { hidden: 64, ..Geom::OMNI_STAR },
            experts: 16,
            topk: 4,
            latent: 16,
            expert_inter: 24,
            shared_inter: 32,
            routed_scale: 5.0,
            eps: RMS_EPS,
        }
    }

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

        /// Norm weights around 1, so enorm, hnorm and the other norms differ from each other.
        fn norm(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| 1.0 + self.next_f32(0.5)).collect()
        }
    }

    /// Seeded weights scaled so the projections are O(1) for O(1) inputs. Every expert is loaded.
    fn synth(g: MtpGeom, seed: u64) -> MtpWeights {
        let mut r = Rng(seed);
        let h = g.hidden();
        let s = |fan_in: usize| 1.0 / (fan_in as f32).sqrt();
        let (q, kv) = (g.attn.q_dim(), g.attn.kv_dim());
        MtpWeights {
            geom: g,
            eh_proj: r.vec(h * 2 * h, s(2 * h)),
            enorm: r.norm(h),
            hnorm: r.norm(h),
            attn_norm: r.norm(h),
            wq: r.vec(q * h, s(h)),
            wk: r.vec(kv * h, s(h)),
            wv: r.vec(kv * h, s(h)),
            wo: r.vec(h * q, s(q)),
            moe_norm: r.norm(h),
            gate: r.vec(g.experts * h, s(h)),
            gate_bias: r.vec(g.experts, 0.1),
            fc1: r.vec(g.latent * h, s(h)),
            fc2: r.vec(h * g.latent, s(g.latent)),
            shared_up: r.vec(g.shared_inter * h, s(h)),
            shared_down: r.vec(h * g.shared_inter, s(g.shared_inter)),
            experts: (0..g.experts)
                .map(|_| {
                    Some(ExpertWeights {
                        up: r.vec(g.expert_inter * g.latent, s(g.latent)),
                        down: r.vec(g.latent * g.expert_inter, s(g.expert_inter)),
                    })
                })
                .collect(),
            final_norm: r.norm(h),
        }
    }

    /// `t` fusion rows: raw embeddings and post-norm_f hidden states.
    fn rows(g: &MtpGeom, t: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut r = Rng(seed);
        (r.vec(t * g.hidden(), 1.0), r.vec(t * g.hidden(), 1.0))
    }

    fn at(start: usize, n: usize) -> Vec<usize> {
        (start..start + n).collect()
    }

    /// One call on a cache: the rows land at the cache's next positions.
    fn run(w: &MtpWeights, e: &[f32], hd: &[f32], kv: &mut MtpKv) -> Vec<f32> {
        let n = e.len() / w.geom.hidden();
        forward(w, e, hd, &at(kv.len(), n), kv)
    }

    fn fresh(w: &MtpWeights, e: &[f32], hd: &[f32]) -> Vec<f32> {
        run(w, e, hd, &mut MtpKv::new(&w.geom))
    }

    fn assert_bits_eq(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        if let Some(i) = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits()) {
            panic!("{what}: first bit difference at {i}: {} vs {}", got[i], want[i]);
        }
    }

    fn differs(a: &[f32], b: &[f32]) -> bool {
        a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits())
    }

    fn widen(x: &[f32]) -> Vec<f64> {
        x.iter().map(|&a| a as f64).collect()
    }

    fn max_abs_diff(got: &[f32], want: &[f64]) -> f64 {
        assert_eq!(got.len(), want.len());
        got.iter().zip(want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max)
    }

    fn max_abs_diff32(a: &[f32], b: &[f32]) -> f64 {
        max_abs_diff(a, &widen(b))
    }

    /// RMSNorm in f64, written out per element, with the epsilon given as a literal by the caller.
    fn naive_rms(x: &[f64], w: &[f32], eps: f64) -> Vec<f64> {
        let d = w.len();
        let mut out = Vec::with_capacity(x.len());
        for row in x.chunks(d) {
            let ms = row.iter().map(|v| v * v).sum::<f64>() / d as f64;
            let inv = 1.0 / (ms + eps).sqrt();
            out.extend(row.iter().zip(w).map(|(v, &wc)| v * inv * wc as f64));
        }
        out
    }

    /// The fusion rows in f64: `eh_proj(cat([enorm(e), hnorm(h)]))`, with `first_is_embedding` false for the swapped order.
    fn naive_fused(w: &MtpWeights, e: &[f32], hd: &[f32], first_is_embedding: bool) -> Vec<f64> {
        let h = w.geom.hidden();
        let ne = naive_rms(&widen(e), &w.enorm, 1e-5);
        let nh = naive_rms(&widen(hd), &w.hnorm, 1e-5);
        let mut out = Vec::new();
        for r in 0..e.len() / h {
            let (er, hr) = (&ne[r * h..(r + 1) * h], &nh[r * h..(r + 1) * h]);
            let cat: Vec<f64> =
                if first_is_embedding { er.iter().chain(hr).copied().collect() } else { hr.iter().chain(er).copied().collect() };
            for o in 0..h {
                out.push((0..2 * h).map(|i| w.eh_proj[o * 2 * h + i] as f64 * cat[i]).sum());
            }
        }
        out
    }

    /// Row lengths for `total` rows in chunks of `chunk` (the last one takes the remainder).
    fn chunked(total: usize, chunk: usize) -> Vec<usize> {
        let mut sizes = vec![chunk; total / chunk];
        if total % chunk != 0 {
            sizes.push(total % chunk);
        }
        sizes
    }

    #[test]
    fn the_geometry_is_the_published_one() {
        let cfg = OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap();
        assert_eq!(MtpGeom::from_config(&cfg), MtpGeom::OMNI);
        assert_eq!(RMS_EPS, 1e-5);
        assert_eq!(cfg.eps as f32, RMS_EPS, "the config's layer_norm_epsilon is the head's eps");
        let g = MtpGeom::OMNI;
        assert_eq!((g.hidden(), g.attn.n_heads, g.attn.n_kv, g.attn.head_dim), (4096, 32, 2, 128));
        assert_eq!((g.experts, g.topk, g.latent), (512, 22, 1024));
        assert_eq!(MtpKv::bytes_per_token(&g.attn), 1024, "K and V of 2 heads x 128 in fp16");
    }

    #[test]
    fn the_tensor_names_and_shapes_are_the_loaders_plan() {
        let cfg = OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap();
        let g = MtpGeom::from_config(&cfg);
        let planned: Vec<(String, Vec<usize>)> = units(&cfg, &Storages::new())
            .into_iter()
            .filter(|u| matches!(u.id, UnitId::Mtp(_)))
            .flat_map(|u| u.tensors.into_iter().map(|w| (w.name, w.shape)))
            .collect();
        assert_eq!(planned.len(), 1040);
        assert_eq!(tensor_shapes(&g), planned, "the head's tensors are the plan's MTP tensors, in order");
        // eh_proj is [hidden, 2 * hidden] and the router bias holds one value per expert
        assert_eq!(planned[0].1, vec![4096, 8192]);
        assert_eq!(planned[11].1, vec![512]);
    }

    #[test]
    fn a_weight_of_the_wrong_size_is_refused_by_its_pack_name() {
        let g = small();
        let mut w = synth(g, 1);
        assert!(w.check().is_ok());
        w.wk.pop();
        let err = w.check().unwrap_err();
        assert!(err.contains("language_model.mtp.layers.0.mixer.k_proj.weight"), "{err}");
        let mut w = synth(g, 1);
        w.experts[3].as_mut().unwrap().down.push(0.0);
        let err = w.check().unwrap_err();
        assert!(err.contains("language_model.mtp.layers.1.mixer.experts.3.down_proj.weight"), "{err}");
        let mut w = synth(g, 1);
        w.experts.pop();
        assert!(w.check().unwrap_err().contains("expert slots"));
    }

    #[test]
    #[should_panic(expected = "language_model.mtp.layers.0.eh_proj.weight")]
    fn forward_refuses_a_wrong_sized_weight() {
        let g = small();
        let mut w = synth(g, 1);
        w.eh_proj.truncate(10);
        let (e, hd) = rows(&g, 1, 2);
        fresh(&w, &e, &hd);
    }

    #[test]
    fn output_shapes_follow_the_rows() {
        let g = small();
        let w = synth(g, 2);
        let h = g.hidden();
        for t in [1usize, 2, 7] {
            let (e, hd) = rows(&g, t, 10 + t as u64);
            let mut kv = MtpKv::new(&g);
            let taps = forward_taps(&w, &e, &hd, &at(0, t), &mut kv);
            for (name, v) in
                [("fused", &taps.fused), ("after_attn", &taps.after_attn), ("after_moe", &taps.after_moe), ("out", &taps.out)]
            {
                assert_eq!(v.len(), t * h, "{name} for {t} rows");
                assert!(v.iter().all(|x| x.is_finite()), "{name} has a non-finite value");
            }
            assert_eq!(kv.len(), t);
        }
        let mut kv = MtpKv::new(&g);
        assert!(forward(&w, &[], &[], &[], &mut kv).is_empty());
        assert!(kv.is_empty());
    }

    #[test]
    fn the_measured_dimensions_run_through_the_same_path() {
        // eh_proj [4096, 8192], q and o [4096, 4096], k and v [256, 4096], with a scaled-down MoE (the real experts are 512 x 2.75 M).
        let g = MtpGeom { experts: 8, topk: 2, latent: 32, expert_inter: 48, shared_inter: 64, ..MtpGeom::OMNI };
        let w = synth(g, 3);
        assert_eq!((w.eh_proj.len(), w.wq.len(), w.wk.len(), w.wo.len()), (4096 * 8192, 4096 * 4096, 256 * 4096, 4096 * 4096));
        let (e, hd) = rows(&g, 2, 4);
        let mut kv = MtpKv::new(&g);
        let out = run(&w, &e, &hd, &mut kv);
        assert_eq!(out.len(), 2 * 4096);
        assert!(out.iter().all(|x| x.is_finite()));
        assert_eq!(kv.keys().len(), 2 * 256);
        assert_eq!(kv.bytes(), 2 * 1024);
    }

    #[test]
    fn rms_norm_is_the_epsilon_1e_5_formula() {
        let w = vec![1.0f32; 64];
        // a row with mean square 1e-8, far below eps: out = x / sqrt(1e-8 + 1e-5) = 0.0316..., not x / sqrt(ms) = 1
        let tiny = vec![1e-4f32; 64];
        let got = rms_norm(&tiny, &w, RMS_EPS);
        let want = 1e-4 / (1e-8f64 + 1e-5).sqrt();
        assert!((got[0] as f64 - want).abs() < 1e-7, "{} vs {want}", got[0]);
        let with_1e_6 = 1e-4 / (1e-8f64 + 1e-6).sqrt();
        assert!(
            (got[0] as f64 - with_1e_6).abs() > 0.05,
            "eps 1e-6 would give {with_1e_6}: the test cannot tell the epsilons apart"
        );
        // a row of unit scale: out = x / sqrt(ms + eps), times the weight, per channel
        let x: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();
        let wt: Vec<f32> = (0..8).map(|i| 0.5 + i as f32 * 0.25).collect();
        let ms = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 8.0;
        let got = rms_norm(&x, &wt, RMS_EPS);
        for i in 0..8 {
            let want = x[i] as f64 / (ms + 1e-5).sqrt() * wt[i] as f64;
            assert!((got[i] as f64 - want).abs() < 1e-6, "channel {i}");
        }
        // rows are independent, and a zero row stays zero
        let mut two = x.clone();
        two.extend(vec![0.0f32; 8]);
        let both = rms_norm(&two, &wt, RMS_EPS);
        assert_bits_eq(&both[..8], &got, "row 0 is the same alone and beside another row");
        assert!(both[8..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn the_fusion_row_is_eh_proj_of_enorm_embedding_then_hnorm_hidden() {
        let g = small();
        let w = synth(g, 5);
        // O(1) rows, and rows of mean square near 1e-6 where the eps of 1e-5 changes the answer by a factor of 3 or more
        for (scale, seed) in [(1.0f32, 6u64), (1e-3, 7)] {
            let (e, hd) = rows(&g, 3, seed);
            let (e, hd): (Vec<f32>, Vec<f32>) = (e.iter().map(|v| v * scale).collect(), hd.iter().map(|v| v * scale).collect());
            let taps = forward_taps(&w, &e, &hd, &at(0, 3), &mut MtpKv::new(&g));
            let want = naive_fused(&w, &e, &hd, true);
            let err = max_abs_diff(&taps.fused, &want);
            eprintln!("fusion rows at scale {scale}: max abs error {err:.3e}");
            assert!(err < 1e-5, "scale {scale}: max abs error {err}");
            // the comparison can fail: the other concat order is far from the answer
            let swapped = naive_fused(&w, &e, &hd, false);
            let sens = max_abs_diff(&taps.fused, &swapped);
            assert!(sens > 1e-2 * scale as f64, "scale {scale}: the swapped order differs by only {sens}");
        }
    }

    #[test]
    fn each_stage_is_the_documented_block_on_the_stage_before() {
        let g = small();
        let w = synth(g, 8);
        let h = g.hidden();
        let (e, hd) = rows(&g, 5, 9);
        let taps = forward_taps(&w, &e, &hd, &at(0, 5), &mut MtpKv::new(&g));

        // attention: pre-norm, the block on a fresh cache, a residual add onto the fused rows
        let aw = attn_ref::Weights { geom: g.attn, wq: &w.wq, wk: &w.wk, wv: &w.wv, wo: &w.wo };
        let a = attn_ref::attention_block(&rms_norm(&taps.fused, &w.attn_norm, 1e-5), &aw, &mut KvCache::new(&g.attn));
        assert_bits_eq(&taps.after_attn, &add(&taps.fused, &a), "after_attn");

        // MoE: pre-norm, the trunk's latent MoE with every expert built, a residual add
        let m_in = rms_norm(&taps.after_attn, &w.moe_norm, 1e-5);
        let router = Router { gate: w.gate.clone(), bias: w.gate_bias.clone(), hidden: h, topk: g.topk, scale: g.routed_scale };
        let experts: Vec<Option<Mlp>> = w
            .experts
            .iter()
            .map(|x| {
                let x = x.as_ref().unwrap();
                Some(Mlp {
                    up: Box::new(Dense::from_hf(g.expert_inter, g.latent, &x.up)),
                    down: Box::new(Dense::from_hf(g.latent, g.expert_inter, &x.down)),
                })
            })
            .collect();
        let shared = Mlp {
            up: Box::new(Dense::from_hf(g.shared_inter, h, &w.shared_up)),
            down: Box::new(Dense::from_hf(h, g.shared_inter, &w.shared_down)),
        };
        let m = moe_ref::latent_moe(
            &m_in,
            &router,
            &Dense::from_hf(g.latent, h, &w.fc1),
            &Dense::from_hf(h, g.latent, &w.fc2),
            &experts,
            &shared,
        );
        assert_bits_eq(&taps.after_moe, &add(&taps.after_attn, &m), "after_moe");

        // the output norm is the head's own `final_layernorm`, with eps 1e-5
        let want = naive_rms(&widen(&taps.after_moe), &w.final_norm, 1e-5);
        let err = max_abs_diff(&taps.out, &want);
        eprintln!("final_layernorm: max abs error {err:.3e}");
        assert!(err < 1e-5, "max abs error {err}");
        assert_bits_eq(&taps.out, &rms_norm(&taps.after_moe, &w.final_norm, RMS_EPS), "out");
        // each block changes the stream, so the wiring above is not vacuous
        assert!(differs(&taps.fused, &taps.after_attn) && differs(&taps.after_attn, &taps.after_moe));
    }

    #[test]
    fn a_later_row_does_not_change_an_earlier_output() {
        let g = small();
        let w = synth(g, 11);
        let (t, h) = (9, g.hidden());
        let (e, hd) = rows(&g, t, 12);
        let base = fresh(&w, &e, &hd);
        for changed in [1usize, 4, t - 1] {
            for which in ["embedding", "hidden"] {
                let (mut e2, mut h2) = (e.clone(), hd.clone());
                let target = if which == "embedding" { &mut e2 } else { &mut h2 };
                for x in &mut target[changed * h..(changed + 1) * h] {
                    *x += 1.0;
                }
                let out = fresh(&w, &e2, &h2);
                let split = changed * h;
                assert_bits_eq(&out[..split], &base[..split], &format!("rows before {which} row {changed}"));
                assert!(
                    differs(&out[split..(changed + 1) * h], &base[split..(changed + 1) * h]),
                    "{which} row {changed} changed nothing"
                );
            }
        }
        // the same fact as a prefix property: the first r + 1 rows alone give the first r + 1 outputs of the whole call
        for r in [0usize, 3, t - 2] {
            let n = (r + 1) * h;
            assert_bits_eq(&fresh(&w, &e[..n], &hd[..n]), &base[..n], &format!("prefix of {} rows", r + 1));
        }
    }

    #[test]
    fn the_embedding_is_first_and_it_is_the_half_that_meets_enorm() {
        let g = small();
        let w = synth(g, 13);
        let h = g.hidden();
        let (e, hd) = rows(&g, 4, 14);
        let (e2, h2) = rows(&g, 4, 15);
        let base = fresh(&w, &e, &hd);

        // swapping the two inputs changes the output (the halves are not interchangeable)
        assert!(max_abs_diff32(&fresh(&w, &hd, &e), &base) > 1e-3, "the head ignores which input is the embedding");

        // enorm = 0 zeroes the first half of the concat: nothing of the embedding is left, bit for bit
        let mut z = w.clone();
        z.enorm = vec![0.0; h];
        assert_bits_eq(&fresh(&z, &e, &hd), &fresh(&z, &e2, &hd), "enorm = 0: output independent of the embedding");
        assert!(differs(&fresh(&z, &e, &hd), &fresh(&z, &e, &h2)), "enorm = 0: the hidden half still matters");
        assert!(differs(&fresh(&z, &e, &hd), &base), "enorm has no effect");

        // hnorm = 0 zeroes the second half: nothing of the trunk hidden state is left
        let mut z = w.clone();
        z.hnorm = vec![0.0; h];
        assert_bits_eq(&fresh(&z, &e, &hd), &fresh(&z, &e, &h2), "hnorm = 0: output independent of the hidden state");
        assert!(differs(&fresh(&z, &e, &hd), &fresh(&z, &e2, &hd)), "hnorm = 0: the embedding half still matters");

        // eh_proj columns [hidden, 2 * hidden) are the second half: zeroing them removes the hidden state, and zeroing [0, hidden)
        // removes the embedding. Together with the two checks above this pins the order of the concat and the norm of each half.
        let (mut second, mut first) = (w.clone(), w.clone());
        for o in 0..h {
            second.eh_proj[o * 2 * h + h..(o + 1) * 2 * h].iter_mut().for_each(|x| *x = 0.0);
            first.eh_proj[o * 2 * h..o * 2 * h + h].iter_mut().for_each(|x| *x = 0.0);
        }
        assert_bits_eq(&fresh(&second, &e, &hd), &fresh(&second, &e, &h2), "second half zero: independent of the hidden state");
        assert!(differs(&fresh(&second, &e, &hd), &fresh(&second, &e2, &hd)));
        assert_bits_eq(&fresh(&first, &e, &hd), &fresh(&first, &e2, &hd), "first half zero: independent of the embedding");
        assert!(differs(&fresh(&first, &e, &hd), &fresh(&first, &e, &h2)));
    }

    #[test]
    fn the_head_normalizes_the_post_norm_f_tap_again_with_hnorm() {
        let g = small();
        let w = synth(g, 16);
        let (t, h) = (4, g.hidden());
        let mut r = Rng(0x7A9);
        // the trunk's last residual rows, unnormalized and large; norm_f has a non-constant weight
        let resid = r.vec(t * h, 30.0);
        let norm_f = r.norm(h);
        let tap = rms_norm(&resid, &norm_f, RMS_EPS);
        let want = naive_rms(&widen(&resid), &norm_f, 1e-5);
        assert!(max_abs_diff(&tap, &want) < 1e-5, "the tap is norm_f of the residual with eps 1e-5");
        let (e, _) = rows(&g, t, 17);
        let fused = |hidden: &[f32]| forward_taps(&w, &e, hidden, &at(0, t), &mut MtpKv::new(&g)).fused;

        // the raw residual is not the tap: norm_f's weights are not a scale, so skipping norm_f changes the fusion rows
        assert!(max_abs_diff32(&fused(&resid), &fused(&tap)) > 1e-2, "the head cannot tell the tap from the raw stream");
        // but hnorm normalizes whatever scale arrives: a tap scaled by 7 fuses to the same rows up to the eps
        let scaled: Vec<f32> = tap.iter().map(|v| v * 7.0).collect();
        let err = max_abs_diff32(&fused(&scaled), &fused(&tap));
        eprintln!("hnorm scale invariance on a tap scaled by 7: max abs change {err:.3e}");
        assert!(err < 1e-3, "{err}");
    }

    #[test]
    fn the_cache_grows_one_kib_of_k_and_v_per_token() {
        let g = small();
        let w = synth(g, 18);
        assert_eq!(MtpKv::bytes_per_token(&g.attn), 1024);
        let mut kv = MtpKv::new(&g);
        assert_eq!((kv.len(), kv.bytes()), (0, 0));
        let mut done = 0;
        for (i, n) in [3usize, 1, 4].into_iter().enumerate() {
            let (e, hd) = rows(&g, n, 20 + i as u64);
            run(&w, &e, &hd, &mut kv);
            done += n;
            assert_eq!(kv.len(), done);
            assert_eq!(kv.keys().len(), done * 256, "K: 2 KV heads x 128 per token");
            assert_eq!(kv.values().len(), done * 256, "V: 2 KV heads x 128 per token");
            assert_eq!(kv.bytes(), done * 1024);
        }
        // the cache holds the cache dtype: every value is on the fp16 grid, so 2 bytes each is exact
        assert!(kv.keys().iter().chain(kv.values()).all(|&x| attn_ref::round_f16(x).to_bits() == x.to_bits()));
    }

    #[test]
    fn rows_one_at_a_time_through_the_cache_equal_one_call() {
        let g = small();
        let w = synth(g, 21);
        let t = 11;
        let (e, hd) = rows(&g, t, 22);
        let h = g.hidden();
        let mut whole_kv = MtpKv::new(&g);
        let whole = run(&w, &e, &hd, &mut whole_kv);
        assert_eq!(whole_kv.len(), t);
        let plans = [vec![1; t], chunked(t, 2), chunked(t, 4), vec![t], vec![3, 1, 5, 2]];
        for plan in plans {
            assert_eq!(plan.iter().sum::<usize>(), t);
            let mut kv = MtpKv::new(&g);
            let mut out = Vec::with_capacity(whole.len());
            let mut start = 0;
            for rows in plan.iter().copied() {
                out.extend(run(&w, &e[start * h..(start + rows) * h], &hd[start * h..(start + rows) * h], &mut kv));
                start += rows;
            }
            let what = format!("chunks {plan:?}");
            assert_bits_eq(&out, &whole, &format!("{what}: output"));
            assert_bits_eq(kv.keys(), whole_kv.keys(), &format!("{what}: keys"));
            assert_bits_eq(kv.values(), whole_kv.values(), &format!("{what}: values"));
        }
    }

    #[test]
    fn the_same_inputs_give_the_same_bits() {
        let g = small();
        let (e, hd) = rows(&g, 6, 24);
        let a = forward_taps(&synth(g, 23), &e, &hd, &at(0, 6), &mut MtpKv::new(&g));
        let b = forward_taps(&synth(g, 23), &e, &hd, &at(0, 6), &mut MtpKv::new(&g));
        for (name, x, y) in [
            ("fused", &a.fused, &b.fused),
            ("after_attn", &a.after_attn, &b.after_attn),
            ("after_moe", &a.after_moe, &b.after_moe),
            ("out", &a.out, &b.out),
        ] {
            assert_bits_eq(x, y, name);
        }
        // a different seed is a different head, so the equality above is not an artifact of constant output
        assert!(differs(&a.out, &fresh(&synth(g, 25), &e, &hd)));
    }

    #[test]
    fn restoring_a_cache_clone_rolls_a_speculative_row_back() {
        let g = small();
        let w = synth(g, 26);
        let (e, hd) = rows(&g, 5, 27);
        let h = g.hidden();
        let mut kv = MtpKv::new(&g);
        run(&w, &e[..3 * h], &hd[..3 * h], &mut kv);
        let saved = kv.clone();
        let (se, sh) = rows(&g, 1, 28);
        run(&w, &se, &sh, &mut kv); // a speculative row that gets rejected
        assert_eq!(kv.len(), 4);
        kv = saved;
        assert_eq!(kv.len(), 3);
        let out = run(&w, &e[3 * h..], &hd[3 * h..], &mut kv);
        assert_bits_eq(&out, &fresh(&w, &e, &hd)[3 * h..], "rows after the rollback equal the straight path");
    }

    #[test]
    #[should_panic(expected = "position 5, but the head's cache holds 0 rows")]
    fn positions_that_do_not_continue_the_cache_are_refused() {
        let g = small();
        let w = synth(g, 29);
        let (e, hd) = rows(&g, 2, 30);
        forward(&w, &e, &hd, &[5, 6], &mut MtpKv::new(&g));
    }

    #[test]
    fn only_the_selected_experts_are_needed() {
        let g = small();
        let w = synth(g, 31);
        let h = g.hidden();
        let (e, hd) = rows(&g, 3, 32);
        let taps = forward_taps(&w, &e, &hd, &at(0, 3), &mut MtpKv::new(&g));
        // the experts the router picks for these rows, found with the trunk's router
        let m_in = rms_norm(&taps.after_attn, &w.moe_norm, RMS_EPS);
        let router = Router { gate: w.gate.clone(), bias: w.gate_bias.clone(), hidden: h, topk: g.topk, scale: g.routed_scale };
        let logits = router.logits(&m_in, 3);
        let mut picked = BTreeSet::new();
        for r in 0..3 {
            picked.extend(moe_ref::route(&logits[r * g.experts..(r + 1) * g.experts], &router.bias, g.topk, g.routed_scale).0);
        }
        assert!(picked.len() < g.experts, "every expert is picked: the test cannot show a partial load");
        let mut partial = w.clone();
        for (i, slot) in partial.experts.iter_mut().enumerate() {
            if !picked.contains(&i) {
                *slot = None;
            }
        }
        assert!(partial.check().is_ok());
        assert_bits_eq(&fresh(&partial, &e, &hd), &taps.out, "only the picked experts loaded");
    }

    #[test]
    #[should_panic(expected = "is selected but was not loaded")]
    fn a_selected_expert_that_is_not_loaded_is_refused() {
        let g = small();
        let mut w = synth(g, 31);
        w.experts.iter_mut().for_each(|slot| *slot = None);
        let (e, hd) = rows(&g, 1, 33);
        fresh(&w, &e, &hd);
    }

    #[test]
    fn the_logits_are_the_trunk_lm_head_on_the_head_output() {
        let g = small();
        let w = synth(g, 34);
        let h = g.hidden();
        let vocab = 50;
        let (e, hd) = rows(&g, 3, 35);
        let out = fresh(&w, &e, &hd);
        let lm_head = Rng(36).vec(vocab * h, 1.0 / (h as f32).sqrt());
        let got = logits(&out, &lm_head, h);
        assert_eq!(got.len(), 3 * vocab);
        let mut want = Vec::new();
        for r in 0..3 {
            for v in 0..vocab {
                want.push((0..h).map(|i| lm_head[v * h + i] as f64 * out[r * h + i] as f64).sum::<f64>());
            }
        }
        assert!(max_abs_diff(&got, &want) < 1e-5);
        // the embedding lookup is a row gather of the same table layout
        let table = Rng(37).vec(vocab * h, 1.0);
        let looked = embed_rows(&table, h, &[7, 0, 49]);
        assert_bits_eq(&looked[h..2 * h], &table[..h], "token 0");
        assert_bits_eq(&looked[2 * h..], &table[49 * h..], "token 49");
    }

    /// A 4-bit head projection decoded by [`hf_from_exl3`] is the EXL3 layer: through the reference's dense map (`Dense::from_hf`) a row
    /// gives what `moe_ref::Exl3` (`gb_exl3::forward`, the host EXL3 path) gives, to f32 rounding of the weight.
    #[test]
    fn an_exl3_head_projection_decodes_to_the_dense_map_of_the_same_layer() {
        use gb_exl3::{Bits, Codebook};
        let (k, n, bits) = (256usize, 384usize, Bits::from_twice(8).unwrap());
        let mut r = Rng(0x5eed_1234);
        let trellis: Vec<i16> =
            (0..(k / 16) * (n / 16) * bits.tile_words()).map(|_| (r.next_f32(1.0) * 32767.0) as i16).collect();
        let suh: Vec<f64> = (0..k).map(|_| r.next_f32(0.3) as f64).collect();
        let svh: Vec<f64> = (0..n).map(|_| r.next_f32(0.05) as f64).collect();
        let w = hf_from_exl3(&trellis, k, n, bits, Codebook::Mul1, &suh, &svh);
        assert_eq!(w.len(), n * k);
        let rows = 3;
        let x = r.vec(rows * k, 1.0);
        let got = Dense::from_hf(n, k, &w).apply(&x, rows);
        let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let want =
            gb_exl3::forward(&xf, rows, &trellis, k / 16, n / 16, bits, &gb_exl3::codebook(Codebook::Mul1), &suh, &svh, None);
        let (mut d2, mut t2) = (0f64, 0f64);
        for (a, b) in got.iter().zip(&want) {
            d2 += (*a as f64 - b).powi(2);
            t2 += b * b;
        }
        assert!(t2 > 0.0 && (d2 / t2).sqrt() < 1e-6, "rel {}", (d2 / t2).sqrt());
    }
}
