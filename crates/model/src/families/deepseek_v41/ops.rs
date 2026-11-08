//! The DeepSeek-V4.1-Flash forward as data: the list of operations the executor (`model.rs`) walks, and the reference arithmetic the
//! zero-GPU forward is checked against.
//!
//! Building the list is a pure function of the config, so every invariant the numbers depend on is checked on the CPU, and mutated,
//! without a device. The schedule mirrors `Transformer.forward` / `Block.forward` in the reference (`model.py`, 1186-1287 and
//! 909-998) exactly, including the two things about this block that are easiest to get wrong:
//!
//! * **`pre_mix` is carried, not recomputed.** A block's `hc_mixes` runs on the streams as they arrive and produces the
//!   coefficients of the *next* sublayer: attention collapses with the `pre` of the previous block's FFN (`make_identity_pre_mix` at
//!   layer 0), the FFN collapses with the `pre` this block's attention computed. [`PreMix`] is that carry, spelled out, so the
//!   collapse cannot silently take its own site's coefficients.
//! * **The engram comes before the block.** `Transformer.forward` calls `layer.engram(h, ...)` on the residual streams *before*
//!   `layer(h, ...)`, on the layers in `engram_layer_ids` (1 and 14), so its contribution is in the streams the block's first
//!   `hc_mixes` sees.
//!
//! ```text
//! Embed
//! per layer l:  [Engram(l)]
//!               HcMixes(attn l)  HcCollapse(attn l, mix)  RmsNorm(attn l)  Mla(l)  HcPost(attn l)
//!               HcMixes(ffn l)   HcCollapse(ffn l, attn l) RmsNorm(ffn l)  MoeShared(l) MoeRoute(l) MoeExperts(l)  HcPost(ffn l)
//!               (mix for layer l + 1 is ffn l)
//! HeadCollapse(mix)  FinalNorm  LmHead
//! ```
//!
//! The MoE is three ops rather than one so that its internal order — the shared expert computed before the router, the routed
//! experts added last — is a property of the schedule a test can read, not a comment in the executor.

use super::config::DeepseekV41Config;

/// Which of a layer's two hyper-connection sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SiteKind {
    /// Around the mixer (`hc_attn_*`), normed by `attn_norm`.
    Attn,
    /// Around the feed-forward block (`hc_ffn_*`), normed by `ffn_norm`.
    Ffn,
}

impl SiteKind {
    pub fn name(self) -> &'static str {
        match self {
            SiteKind::Attn => "attn",
            SiteKind::Ffn => "ffn",
        }
    }
}

/// The `pre` coefficients a collapse uses. `Identity` is `make_identity_pre_mix`: every copy weighted 1, which is what layer 0's
/// attention starts from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreMix {
    Identity,
    Site { layer: usize, kind: SiteKind },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Token ids -> the `hc_mult` fp32 residual streams (the embedding of each row, broadcast).
    Embed,
    /// The layer's n-gram lookup: replaces the streams with the gated sum. Layers in `engram_layer_ids` only.
    Engram { layer: usize },
    /// `hc_mixes`: one projection of the flattened streams, split into `pre` / `post` / `comb` (Sinkhorn on the comb).
    HcMixes { layer: usize, kind: SiteKind },
    /// `hc_pre`: collapse the copies into one sublayer input, weighted by `mix`.
    HcCollapse { layer: usize, kind: SiteKind, mix: PreMix },
    /// The site's RMSNorm gamma.
    RmsNorm { layer: usize, kind: SiteKind },
    /// The mixer: MLA over the window and the compressed plane, selected by the DSA indexer.
    Mla { layer: usize },
    /// The shared expert's gated MLP (`down(act(gate(x)) * up(x))`), computed before the router.
    MoeShared { layer: usize },
    /// The router: `sqrtsoftplus` logits, `noaux_tc` selection with the correction bias, six picks and their weights.
    MoeRoute { layer: usize },
    /// The six routed experts on the picks, weighted and summed, then added to the shared expert's output.
    MoeExperts { layer: usize },
    /// `hc_post`: expand the sublayer output and mix the residual in through `comb`.
    HcPost { layer: usize, kind: SiteKind },
    /// The last block's collapse, on the streams as they leave layer `layers - 1`.
    HeadCollapse { mix: PreMix },
    /// The final RMSNorm.
    FinalNorm,
    /// The untied `lm_head`.
    LmHead,
}

impl Op {
    /// The op's name in a log line or a test failure.
    pub fn name(&self) -> &'static str {
        match self {
            Op::Embed => "embed",
            Op::Engram { .. } => "engram",
            Op::HcMixes { .. } => "hc_mixes",
            Op::HcCollapse { .. } => "hc_collapse",
            Op::RmsNorm { .. } => "rms_norm",
            Op::Mla { .. } => "mla",
            Op::MoeShared { .. } => "moe_shared",
            Op::MoeRoute { .. } => "moe_route",
            Op::MoeExperts { .. } => "moe_experts",
            Op::HcPost { .. } => "hc_post",
            Op::HeadCollapse { .. } => "head_collapse",
            Op::FinalNorm => "final_norm",
            Op::LmHead => "lm_head",
        }
    }
}

/// Whether `layer` carries an engram table.
pub fn has_engram(cfg: &DeepseekV41Config, layer: usize) -> bool {
    cfg.engram.layer_ids.contains(&layer)
}

/// The forward of the first `layers` layers, followed (`head`) by the collapse, the final norm and the `lm_head`.
pub fn plan_forward(cfg: &DeepseekV41Config, layers: usize, head: bool) -> Vec<Op> {
    let n = layers.min(cfg.layers);
    let mut ops = vec![Op::Embed];
    let mut mix = PreMix::Identity;
    for l in 0..n {
        if has_engram(cfg, l) {
            ops.push(Op::Engram { layer: l });
        }
        ops.push(Op::HcMixes { layer: l, kind: SiteKind::Attn });
        ops.push(Op::HcCollapse { layer: l, kind: SiteKind::Attn, mix });
        ops.push(Op::RmsNorm { layer: l, kind: SiteKind::Attn });
        ops.push(Op::Mla { layer: l });
        ops.push(Op::HcPost { layer: l, kind: SiteKind::Attn });

        ops.push(Op::HcMixes { layer: l, kind: SiteKind::Ffn });
        ops.push(Op::HcCollapse { layer: l, kind: SiteKind::Ffn, mix: PreMix::Site { layer: l, kind: SiteKind::Attn } });
        ops.push(Op::RmsNorm { layer: l, kind: SiteKind::Ffn });
        // the router first, so its selection can go to the cache's prefetch queue before the shared expert and the post run: the fills
        // are NVMe reads, and the only way they are not on the critical path is if something else is
        ops.push(Op::MoeRoute { layer: l });
        ops.push(Op::MoeShared { layer: l });
        ops.push(Op::MoeExperts { layer: l });
        ops.push(Op::HcPost { layer: l, kind: SiteKind::Ffn });
        mix = PreMix::Site { layer: l, kind: SiteKind::Ffn };
    }
    if head {
        ops.push(Op::HeadCollapse { mix });
        ops.push(Op::FinalNorm);
        ops.push(Op::LmHead);
    }
    ops
}

/// The reference's `make_identity_pre_mix` (`model.py:1138`): stream 0 hot (`[1.0, 0.0, ...]`).
pub fn identity_pre_mix(hc: usize) -> Vec<f32> {
    let mut v = vec![0.0; hc];
    if hc > 0 {
        v[0] = 1.0;
    }
    v
}

// ----------------------------------------------------------------------------------------------------------------- the invariants

/// A deliberate bug in the schedule, so the tests can show the checker catches each one. Production passes none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutant {
    /// The attention collapse takes its own site's `pre` instead of the previous block's FFN.
    OwnPreMix,
    /// The FFN collapse takes the previous block's `pre` instead of this block's attention.
    StalePreMix,
    /// The engram runs after the block instead of before it.
    EngramAfterBlock,
    /// The engram is scheduled on layer 0 instead of the layers the config names.
    EngramShifted,
    /// The routed experts are computed before the router.
    ExpertsBeforeRoute,
    /// The shared expert is added after the routed experts instead of before.
    SharedAfterExperts,
    /// The head skips the collapse.
    NoHeadCollapse,
    /// A global norm is applied where the site's should be.
    NormAfterPost,
}

/// [`plan_forward`] with a deliberate bug, for [`check_plan`]'s tests.
pub fn plan_forward_mutated(cfg: &DeepseekV41Config, layers: usize, head: bool, m: Mutant) -> Vec<Op> {
    let mut ops = plan_forward(cfg, layers, head);
    let n = layers.min(cfg.layers);
    match m {
        Mutant::OwnPreMix => {
            for op in ops.iter_mut() {
                if let Op::HcCollapse { layer, kind: SiteKind::Attn, .. } = op {
                    *op = Op::HcCollapse {
                        layer: *layer,
                        kind: SiteKind::Attn,
                        mix: PreMix::Site { layer: *layer, kind: SiteKind::Attn },
                    };
                }
            }
        }
        Mutant::StalePreMix => {
            for op in ops.iter_mut() {
                if let Op::HcCollapse { layer, kind: SiteKind::Ffn, .. } = op {
                    let mix =
                        if *layer == 0 { PreMix::Identity } else { PreMix::Site { layer: *layer - 1, kind: SiteKind::Ffn } };
                    *op = Op::HcCollapse { layer: *layer, kind: SiteKind::Ffn, mix };
                }
            }
        }
        Mutant::EngramAfterBlock => {
            // move every Engram op to just after its block's last HcPost
            for l in 0..n {
                if let Some(at) = ops.iter().position(|o| *o == Op::Engram { layer: l }) {
                    let e = ops.remove(at);
                    if let Some(post) = ops.iter().rposition(|o| *o == Op::HcPost { layer: l, kind: SiteKind::Ffn }) {
                        ops.insert(post + 1, e);
                    }
                }
            }
        }
        Mutant::EngramShifted => {
            for op in ops.iter_mut() {
                if let Op::Engram { layer } = op {
                    *layer = layer.saturating_sub(1);
                }
            }
            if !cfg.engram.layer_ids.is_empty() && !has_engram(cfg, 0) {
                if let Some(Op::Engram { layer }) = ops.iter_mut().find(|o| matches!(o, Op::Engram { .. })) {
                    *layer = 0;
                }
            }
        }
        Mutant::ExpertsBeforeRoute => {
            for l in 0..n {
                let (r, e) = (Op::MoeRoute { layer: l }, Op::MoeExperts { layer: l });
                if let (Some(ri), Some(ei)) = (ops.iter().position(|o| *o == r), ops.iter().position(|o| *o == e)) {
                    if ri < ei {
                        ops.swap(ri, ei);
                    }
                }
            }
        }
        Mutant::SharedAfterExperts => {
            for l in 0..n {
                let (s, e) = (Op::MoeShared { layer: l }, Op::MoeExperts { layer: l });
                if let (Some(si), Some(ei)) = (ops.iter().position(|o| *o == s), ops.iter().position(|o| *o == e)) {
                    if si < ei {
                        ops.swap(si, ei);
                    }
                }
            }
        }
        Mutant::NoHeadCollapse => ops.retain(|o| !matches!(o, Op::HeadCollapse { .. })),
        Mutant::NormAfterPost => {
            for l in 0..n {
                for kind in [SiteKind::Attn, SiteKind::Ffn] {
                    let nrm = Op::RmsNorm { layer: l, kind };
                    let post = Op::HcPost { layer: l, kind };
                    if let Some(ni) = ops.iter().position(|o| *o == nrm) {
                        ops.remove(ni);
                        if let Some(pi) = ops.iter().position(|o| *o == post) {
                            ops.insert(pi + 1, nrm);
                        }
                    }
                }
            }
        }
    }
    ops
}

/// Every invariant of a schedule, written independently of the builder: `Err` names the first violated one.
pub fn check_plan(cfg: &DeepseekV41Config, layers: usize, head: bool, ops: &[Op]) -> Result<(), String> {
    let n = layers.min(cfg.layers);
    if ops.first() != Some(&Op::Embed) {
        return Err("the forward must start with the embedding".into());
    }
    if ops.iter().filter(|o| **o == Op::Embed).count() != 1 {
        return Err("exactly one embedding".into());
    }
    let mut i = 1usize;
    let mut mix = PreMix::Identity;
    let at = |k: usize| ops.get(k).copied();
    for l in 0..n {
        // the engram, when this layer has one, runs on the streams before the block
        let want_engram = has_engram(cfg, l);
        let has = at(i) == Some(Op::Engram { layer: l });
        if want_engram != has {
            return Err(match (want_engram, at(i)) {
                (true, Some(o)) => format!("op {i}: layer {l} carries an engram table, but the schedule has {}", o.name()),
                (false, Some(o)) => {
                    format!("op {i}: layer {l} has no engram table, but the schedule runs {} on it", o.name())
                }
                (_, None) => format!("op {i}: the schedule ends inside layer {l}"),
            });
        }
        if has {
            i += 1;
        }
        // attention site: mixes, collapse(pre_mix), norm, mixer, post
        let want = [
            Op::HcMixes { layer: l, kind: SiteKind::Attn },
            Op::HcCollapse { layer: l, kind: SiteKind::Attn, mix },
            Op::RmsNorm { layer: l, kind: SiteKind::Attn },
            Op::Mla { layer: l },
            Op::HcPost { layer: l, kind: SiteKind::Attn },
            // ffn site: mixes, collapse(this block's attn pre), norm
            Op::HcMixes { layer: l, kind: SiteKind::Ffn },
            Op::HcCollapse { layer: l, kind: SiteKind::Ffn, mix: PreMix::Site { layer: l, kind: SiteKind::Attn } },
            Op::RmsNorm { layer: l, kind: SiteKind::Ffn },
            // the MoE: the router first, then the shared expert, then the routed experts (the route is what the cache prefetches from)
            Op::MoeRoute { layer: l },
            Op::MoeShared { layer: l },
            Op::MoeExperts { layer: l },
            Op::HcPost { layer: l, kind: SiteKind::Ffn },
        ];
        for w in want {
            match at(i) {
                Some(got) if got == w => i += 1,
                Some(got) => return Err(format!("op {i}: expected {} on layer {l}, found {} — {:?}", w.name(), got.name(), w)),
                None => return Err(format!("op {i}: the schedule ends inside layer {l} (expected {})", w.name())),
            }
        }
        mix = PreMix::Site { layer: l, kind: SiteKind::Ffn };
    }
    if head {
        if at(i) != Some(Op::HeadCollapse { mix }) {
            return Err(format!("op {i}: the head collapses with the last block's FFN pre_mix ({mix:?}), found {:?}", at(i)));
        }
        i += 1;
        if at(i) != Some(Op::FinalNorm) {
            return Err(format!("op {i}: the final norm follows the head collapse"));
        }
        i += 1;
        if at(i) != Some(Op::LmHead) {
            return Err(format!("op {i}: the lm_head is last"));
        }
        i += 1;
    }
    if i != ops.len() {
        return Err(format!("op {i}: {} ops after the end of the schedule ({:?})", ops.len() - i, &ops[i..]));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ the reference arithmetic
//
// The host primitives every op is defined by. They are the CPU transcription of the reference's torch, written once here so that
// the toy forward in `model.rs` and its test agree by construction rather than by two people reading the same Python.

/// `x * rsqrt(mean(x^2) + eps) * w`, in place. The mean is over one row of `width`.
pub fn rms_norm(x: &mut [f32], w: &[f32], width: usize, eps: f32) {
    for row in x.chunks_exact_mut(width) {
        let mean = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let r = (mean + eps).sqrt().recip();
        for (v, g) in row.iter_mut().zip(w) {
            *v = *v * r * g;
        }
    }
}

/// `y = W x` for `W` `[n, k]` row major and `x` `[rows, k]`.
pub fn linear(x: &[f32], w: &[f32], rows: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * n];
    for r in 0..rows {
        for j in 0..n {
            let mut acc = 0f32;
            for c in 0..k {
                acc = w[j * k + c].mul_add(x[r * k + c], acc);
            }
            y[r * n + j] = acc;
        }
    }
    y
}

/// The fused SwiGLU with the training clamp (`Expert.forward` without the two GEMMs).
pub fn swiglu(gate: &[f32], up: &[f32], w: Option<&[f32]>, rows: usize, inter: usize, limit: f32) -> Vec<f32> {
    let mut out = vec![0f32; rows * inter];
    for r in 0..rows {
        let wr = w.map_or(1.0, |w| w[r]);
        for i in 0..inter {
            let g = gate[r * inter + i];
            let u = up[r * inter + i];
            let (g, u) = if limit > 0.0 { (g.min(limit), u.clamp(-limit, limit)) } else { (g, u) };
            out[r * inter + i] = (g / (1.0 + (-g).exp())) * u * wr;
        }
    }
    out
}

/// `softplus(x).sqrt()`: the router's score function.
pub fn sqrtsoftplus(x: f32) -> f32 {
    // log1p(exp(x)) is what torch's softplus is; for large x the exp overflows, so take the linear form there
    let s = if x > 20.0 { x } else { x.exp().ln_1p() };
    s.max(0.0).sqrt()
}

/// The router's selection: `topk` of `score + bias` (ties to the lower id), weights `score / sum(score)` times `scale`.
pub fn select(logits: &[f32], rows: usize, experts: usize, topk: usize, bias: &[f32], scale: f32) -> (Vec<i32>, Vec<f32>) {
    let mut pick = vec![0i32; rows * topk];
    let mut wts = vec![0f32; rows * topk];
    for r in 0..rows {
        let lg = &logits[r * experts..(r + 1) * experts];
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&a, &b| {
            let ka = sqrtsoftplus(lg[a]) + bias[a];
            let kb = sqrtsoftplus(lg[b]) + bias[b];
            kb.total_cmp(&ka).then(a.cmp(&b))
        });
        let sum: f32 = order[..topk].iter().map(|&e| sqrtsoftplus(lg[e])).sum();
        let den = sum + 1e-20;
        for k in 0..topk {
            let e = order[k];
            pick[r * topk + k] = e as i32;
            wts[r * topk + k] = sqrtsoftplus(lg[e]) / den * scale;
        }
    }
    (pick, wts)
}

/// The Sinkhorn tail of `hc_split_sinkhorn` on one `[hc, hc]` comb, in place: row softmax, `+ eps`, one column normalisation,
/// then `iters - 1` rounds of (row, column), every division `/(sum + eps)`.
pub fn sinkhorn(c: &mut [f32], hc: usize, iters: usize, eps: f32) {
    for j in 0..hc {
        let mx = c[j * hc..(j + 1) * hc].iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut s = 0f32;
        for k in 0..hc {
            c[j * hc + k] = (c[j * hc + k] - mx).exp();
            s += c[j * hc + k];
        }
        for k in 0..hc {
            c[j * hc + k] = c[j * hc + k] / s + eps;
        }
    }
    let mut col = vec![0f32; hc];
    for k in 0..hc {
        for j in 0..hc {
            col[k] += c[j * hc + k];
        }
        for j in 0..hc {
            c[j * hc + k] /= col[k] + eps;
        }
    }
    for _ in 1..iters {
        for j in 0..hc {
            let s: f32 = c[j * hc..(j + 1) * hc].iter().sum();
            for k in 0..hc {
                c[j * hc + k] /= s + eps;
            }
        }
        for k in 0..hc {
            let s: f32 = (0..hc).map(|j| c[j * hc + k]).sum();
            for j in 0..hc {
                c[j * hc + k] /= s + eps;
            }
        }
    }
}

/// `hc_split_sinkhorn`: 25 mixes into `pre` `[hc]`, `post` `[hc]` and `comb` `[hc, hc]`.
pub fn hc_split(mixes: &[f32], scale: &[f32], base: &[f32], hc: usize, iters: usize, eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let nfn = (2 + hc) * hc;
    assert_eq!(mixes.len(), nfn);
    assert_eq!(base.len(), nfn);
    let mut pre = vec![0f32; hc];
    let mut post = vec![0f32; hc];
    for h in 0..hc {
        pre[h] = 1.0 / (1.0 + (-mixes[h].mul_add(scale[0], base[h])).exp()) + eps;
        post[h] = 2.0 * (1.0 / (1.0 + (-mixes[h + hc].mul_add(scale[1], base[h + hc])).exp()));
    }
    let mut comb = vec![0f32; hc * hc];
    for j in 0..hc * hc {
        comb[j] = mixes[2 * hc + j].mul_add(scale[2], base[2 * hc + j]);
    }
    sinkhorn(&mut comb, hc, iters, eps);
    (pre, post, comb)
}

/// `Block.hc_mixes`: `mixes = (x.flatten . fn^T) * rsqrt(mean(x^2) + rms_eps)` then [`hc_split`].
#[allow(clippy::too_many_arguments)] // the mHC mix sites' fixed geometry; grouping would rename, not reduce
pub fn hc_mixes(
    x: &[f32],
    rows: usize,
    hc: usize,
    hidden: usize,
    fnw: &[f32],
    scale: &[f32],
    base: &[f32],
    iters: usize,
    rms_eps: f32,
    hc_eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let row = hc * hidden;
    let nfn = (2 + hc) * hc;
    let mut pre = vec![0f32; rows * hc];
    let mut post = vec![0f32; rows * hc];
    let mut comb = vec![0f32; rows * hc * hc];
    for r in 0..rows {
        let xr = &x[r * row..(r + 1) * row];
        let rstd = (xr.iter().map(|v| v * v).sum::<f32>() / row as f32 + rms_eps).sqrt().recip();
        let mut mixes = vec![0f32; nfn];
        for (q, m) in mixes.iter_mut().enumerate() {
            let mut acc = 0f32;
            for d in 0..row {
                acc = fnw[q * row + d].mul_add(xr[d], acc);
            }
            *m = acc * rstd;
        }
        let (p, s, c) = hc_split(&mixes, scale, base, hc, iters, hc_eps);
        pre[r * hc..(r + 1) * hc].copy_from_slice(&p);
        post[r * hc..(r + 1) * hc].copy_from_slice(&s);
        comb[r * hc * hc..(r + 1) * hc * hc].copy_from_slice(&c);
    }
    (pre, post, comb)
}

/// `Block.hc_pre`: `sum_h pre[h] x[h][d]`.
pub fn hc_pre(x: &[f32], rows: usize, hc: usize, hidden: usize, pre: &[f32]) -> Vec<f32> {
    let row = hc * hidden;
    let mut out = vec![0f32; rows * hidden];
    for r in 0..rows {
        for d in 0..hidden {
            let mut acc = pre[r * hc] * x[r * row + d];
            for h in 1..hc {
                acc = pre[r * hc + h].mul_add(x[r * row + h * hidden + d], acc);
            }
            out[r * hidden + d] = acc;
        }
    }
    out
}

/// `Block.hc_post`: `post[h] y[d] + sum_g comb[g][h] x[g][d]`.
pub fn hc_post(x: &[f32], y: &[f32], rows: usize, hc: usize, hidden: usize, post: &[f32], comb: &[f32]) -> Vec<f32> {
    let row = hc * hidden;
    let mut out = vec![0f32; rows * row];
    for r in 0..rows {
        for h in 0..hc {
            for d in 0..hidden {
                let mut o = post[r * hc + h] * y[r * hidden + d];
                for g in 0..hc {
                    o = comb[r * hc * hc + g * hc + h].mul_add(x[r * row + g * hidden + d], o);
                }
                out[r * row + h * hidden + d] = o;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const CONFIG: &str = include_str!("../../../../../testdata/deepseek_v41/config.json");

    fn real() -> DeepseekV41Config {
        DeepseekV41Config::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    #[test]
    fn the_full_schedule_by_hand() {
        let c = real();
        let ops = plan_forward(&c, c.layers, true);
        // embed, then 12 ops a layer (plus one engram on each of the two layers that carry one), then the head's three
        assert_eq!(ops.len(), 1 + 12 * c.layers + c.engram.layer_ids.len() + 3);
        assert_eq!(ops[0], Op::Embed);
        // layer 0: no engram, identity pre_mix on the attention collapse, the attention's pre on the FFN collapse
        assert_eq!(ops[1], Op::HcMixes { layer: 0, kind: SiteKind::Attn });
        assert_eq!(ops[2], Op::HcCollapse { layer: 0, kind: SiteKind::Attn, mix: PreMix::Identity });
        // layer 1 carries an engram, and it comes before the block
        let e1 = ops.iter().position(|o| *o == Op::Engram { layer: 1 }).unwrap();
        assert_eq!(ops[e1 + 1], Op::HcMixes { layer: 1, kind: SiteKind::Attn });
        assert_eq!(
            ops[e1 + 2],
            Op::HcCollapse { layer: 1, kind: SiteKind::Attn, mix: PreMix::Site { layer: 0, kind: SiteKind::Ffn } },
            "layer 1's attention collapses with layer 0's FFN pre_mix"
        );
        // the engram layers are the config's
        let engram: Vec<usize> = ops
            .iter()
            .filter_map(|o| match o {
                Op::Engram { layer } => Some(*layer),
                _ => None,
            })
            .collect();
        assert_eq!(engram, c.engram.layer_ids);
        assert_eq!(engram, vec![1, 14]);
        // the MoE order inside one layer: the router first, so its selection can be prefetched while the shared expert runs
        let m = ops.iter().position(|o| *o == Op::MoeRoute { layer: 3 }).unwrap();
        assert_eq!(&ops[m..m + 3], &[Op::MoeRoute { layer: 3 }, Op::MoeShared { layer: 3 }, Op::MoeExperts { layer: 3 }]);
        // the head
        assert_eq!(
            &ops[ops.len() - 3..],
            &[Op::HeadCollapse { mix: PreMix::Site { layer: c.layers - 1, kind: SiteKind::Ffn } }, Op::FinalNorm, Op::LmHead]
        );
        check_plan(&c, c.layers, true, &ops).unwrap();
    }

    #[test]
    fn every_prefix_of_the_model_is_a_valid_schedule() {
        let c = real();
        for n in 1..=c.layers {
            for head in [false, true] {
                let ops = plan_forward(&c, n, head);
                check_plan(&c, n, head, &ops).unwrap_or_else(|e| panic!("{n} layers, head {head}: {e}"));
                // without the head the schedule ends on the last block's FFN post
                if !head {
                    assert_eq!(ops.last(), Some(&Op::HcPost { layer: n - 1, kind: SiteKind::Ffn }));
                }
            }
        }
    }

    #[test]
    fn each_mutant_breaks_an_invariant() {
        let c = real();
        for m in [
            Mutant::OwnPreMix,
            Mutant::StalePreMix,
            Mutant::EngramAfterBlock,
            Mutant::EngramShifted,
            Mutant::ExpertsBeforeRoute,
            Mutant::SharedAfterExperts,
            Mutant::NoHeadCollapse,
            Mutant::NormAfterPost,
        ] {
            let ops = plan_forward_mutated(&c, c.layers, true, m);
            let e = check_plan(&c, c.layers, true, &ops).err().unwrap_or_else(|| panic!("{m:?} was not caught"));
            eprintln!("{m:?}: caught: {e}");
        }
    }

    #[test]
    fn a_schedule_that_ends_early_is_refused_naming_the_op() {
        let c = real();
        let mut ops = plan_forward(&c, c.layers, true);
        ops.truncate(ops.len() - 2);
        let e = check_plan(&c, c.layers, true, &ops).unwrap_err();
        assert!(e.contains("final norm") || e.contains("lm_head"), "{e}");
    }

    #[test]
    fn the_router_scores_and_selects_like_the_reference() {
        // softplus(0).sqrt() = ln2.sqrt(); a bias that lifts expert 3 must pick it first despite a lower logit
        let experts = 8;
        let logits = vec![0.0f32, 0.1, 0.2, -1.0, 0.4, 0.3, -0.5, 0.05];
        let mut bias = vec![0f32; experts];
        bias[3] = 3.0;
        let (pick, wts) = select(&logits, 1, experts, 6, &bias, 1.5);
        assert_eq!(pick[0], 3, "the correction bias steers selection: {pick:?}");
        // the weights are over the *unbiased* scores and sum to the routing scale
        assert!((wts.iter().sum::<f32>() - 1.5).abs() < 1e-5, "{wts:?}");
        // ties go to the lower id
        let flat = vec![0.5f32; experts];
        let (pick, _) = select(&flat, 1, experts, 6, &vec![0f32; experts], 1.0);
        assert_eq!(&pick[..3], &[0, 1, 2]);
        assert_eq!(sqrtsoftplus(0.0), std::f32::consts::LN_2.sqrt());
    }

    #[test]
    fn the_sinkhorn_comb_is_doubly_stochastic() {
        let hc = 4usize;
        let mixes: Vec<f32> = (0..(2 + hc) * hc).map(|i| ((i * 37) % 11) as f32 * 0.1 - 0.5).collect();
        let scale = [0.5f32, 0.25, 0.125];
        let base = vec![0.1f32; (2 + hc) * hc];
        let (pre, post, comb) = hc_split(&mixes, &scale, &base, hc, 20, 1e-6);
        // pre is a sigmoid plus eps, post twice a sigmoid
        assert!(pre.iter().all(|&v| v > 1e-6 && v < 1.0 + 1e-6), "{pre:?}");
        assert!(post.iter().all(|&v| v > 0.0 && v < 2.0), "{post:?}");
        for j in 0..hc {
            let rs: f32 = (0..hc).map(|k| comb[j * hc + k]).sum();
            assert!((rs - (1.0 + hc as f32 * 1e-6)).abs() < 1e-3, "row {j} sums to {rs}");
        }
        for k in 0..hc {
            let cs: f32 = (0..hc).map(|j| comb[j * hc + k]).sum();
            assert!((cs - (1.0 + hc as f32 * 1e-6)).abs() < 1e-3, "column {k} sums to {cs}");
        }
    }

    #[test]
    fn hc_pre_and_hc_post_are_the_reference_pair() {
        // hc_post(hc_pre(x, p), x, post, comb) with post = 1 and comb = I is the identity on x
        let (rows, hc, hidden) = (2usize, 4usize, 8usize);
        let x: Vec<f32> = (0..rows * hc * hidden).map(|i| (i % 17) as f32 * 0.125 - 1.0).collect();
        let pre = vec![0.5f32; rows * hc];
        let y = hc_pre(&x, rows, hc, hidden, &pre);
        let post = vec![1.0f32; rows * hc];
        let mut comb = vec![0f32; rows * hc * hc];
        for r in 0..rows {
            for g in 0..hc {
                comb[r * hc * hc + g * hc + g] = 1.0;
            }
        }
        let back = hc_post(&x, &y, rows, hc, hidden, &post, &comb);
        let want: Vec<f32> = (0..rows)
            .flat_map(|r| {
                x[r * hc * hidden..(r + 1) * hc * hidden].iter().map(|v| v + y[r * hidden..][0] * 0.0).collect::<Vec<_>>()
            })
            .collect();
        // stream 0 of the result is pre[0] * x[0] + x[0] for every row (comb is the identity, so each stream keeps its own x)
        for r in 0..rows {
            for h in 0..hc {
                for d in 0..hidden {
                    let got = back[r * hc * hidden + h * hidden + d];
                    let expect = x[r * hc * hidden + h * hidden + d] + y[r * hidden + d];
                    assert!((got - expect).abs() < 1e-5, "row {r} stream {h} d {d}: {got} vs {expect}");
                }
            }
        }
        assert_eq!(want.len(), rows * hc * hidden);
    }

    #[test]
    fn swiglu_clamps_both_branches_of_the_up_projection_and_the_gate_from_above() {
        let limit = 10.0;
        let gate = [100.0f32, -100.0];
        let up = [100.0f32, -100.0];
        let out = swiglu(&gate, &up, None, 1, 2, limit);
        // gate is min(g, 10); up is clamped to +-10
        let g0 = 10.0f32;
        assert!((out[0] - (g0 / (1.0 + (-g0).exp())) * 10.0).abs() < 1e-4);
        let g1 = -100.0f32;
        assert!((out[1] - (g1 / (1.0 + (-g1).exp())) * -10.0).abs() < 1e-3);
        // a per-row weight multiplies
        let w = [0.5f32];
        let weighted = swiglu(&[1.0], &[2.0], Some(&w), 1, 1, limit);
        let plain = swiglu(&[1.0], &[2.0], None, 1, 1, limit);
        assert!((weighted[0] - plain[0] * 0.5).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_matches_its_definition() {
        let mut x = vec![3.0f32, 4.0, 0.0, 1.0];
        let w = vec![1.0f32, 2.0, 1.0, 0.5];
        rms_norm(&mut x, &w, 4, 0.0);
        let mean: f32 = (9.0 + 16.0 + 0.0 + 1.0) / 4.0;
        let r = mean.sqrt().recip();
        assert!((x[0] - 3.0 * r).abs() < 1e-6);
        assert!((x[1] - 4.0 * r * 2.0).abs() < 1e-6);
        assert!((x[3] - 1.0 * r * 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_linear_is_w_x() {
        let w = [1.0f32, 2.0, 3.0, 4.0]; // [2, 2]
        let y = linear(&[1.0f32, 1.0], &w, 1, 2, 2);
        assert_eq!(y, vec![3.0, 7.0]);
    }

    /// A toy config: the pack's shape, shrunk to 2 layers / 4 experts / hidden 64, with the engram on layers 1 and 14 mapped into
    /// range (the config's own list is filtered to the layers that exist).
    pub(super) fn toy_json() -> Value {
        let mut j: Value = serde_json::from_str(CONFIG).unwrap();
        let t = j["text_config"].as_object_mut().unwrap();
        let mut set = |k: &str, v: Value| {
            t.insert(k.to_string(), v);
        };
        set("vocab_size", 128.into());
        set("hidden_size", 64.into());
        set("num_hidden_layers", 2.into());
        set("num_attention_heads", 2.into());
        set("num_key_value_heads", 1.into());
        set("head_dim", 32.into());
        set("qk_rope_head_dim", 8.into());
        set("q_lora_rank", 16.into());
        set("o_lora_rank", 8.into());
        set("o_groups", 2.into());
        set("moe_intermediate_size", 32.into());
        set("n_routed_experts", 4.into());
        set("n_shared_experts", 1.into());
        set("num_experts_per_tok", 2.into());
        set("sliding_window", 8.into());
        set("max_position_embeddings", 64.into());
        set("compress_ratios", serde_json::json!([0, 0]));
        set("kv_source_layer_ids", serde_json::json!([]));
        set("index_source_layer_ids", serde_json::json!([]));
        set("index_n_heads", 2.into());
        set("index_head_dim", 16.into());
        set("index_topk", 8.into());
        set("candidate_source_layer_id", 0.into());
        set("engram_layer_ids", serde_json::json!([1]));
        set("engram_num_embeddings", serde_json::json!([64]));
        set("engram_max_ngram_size", 4.into());
        set("engram_vocab_size", 128.into());
        set("engram_n_heads", 2.into());
        set("engram_head_dim", 32.into());
        set("num_nextn_predict_layers", 0.into());
        set("dspark_target_layer_ids", serde_json::json!([]));
        set("dspark_noise_token_id", 5.into());
        t.remove("vision_config");
        j
    }

    pub(super) fn toy() -> DeepseekV41Config {
        DeepseekV41Config::from_json(&toy_json()).unwrap()
    }

    #[test]
    fn the_toy_config_is_what_the_forward_test_needs() {
        let c = toy();
        assert_eq!((c.layers, c.hidden, c.moe.routed_experts), (2, 64, 4));
        assert_eq!(c.engram.layer_ids, vec![1]);
        // ratio 0 everywhere: pure sliding-window attention, so the toy needs no compressed plane
        assert!((0..c.layers).all(|l| c.compress.ratio(l) == 0));
        let ops = plan_forward(&c, c.layers, true);
        check_plan(&c, c.layers, true, &ops).unwrap();
        assert!(ops.contains(&Op::Engram { layer: 1 }));
        assert!(!ops.contains(&Op::Engram { layer: 0 }));
    }
}
