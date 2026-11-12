//! The latent-MoE block of the Nemotron 3.5 Super VL text backbone (`E` layers), as a pure-Rust CPU reference.
//!
//! This is the arithmetic the fork runs (`exllamav3/modules/block_sparse_mlp.py` and `block_sparse_mlp_routing.py`, `routing_dots`):
//!
//! 1. The router sees the full-width block input `h`: `logits = h @ gate^T`, `scores = sigmoid(logits)`.
//! 2. Selection is the top-k of `scores + e_score_correction_bias`; a lower expert index wins a tie. The mixture weights are the
//!    unbiased `scores` of the selected experts, normalized over that set, times `routed_scaling_factor`.
//! 3. `x = fc1_latent_proj(h)` (width 1024) is the routed experts' input. Each selected expert computes `down(relu2(up(x)))`, and
//!    the weighted expert outputs of a token are summed.
//! 4. `fc2_latent_proj` projects that sum back to the hidden width.
//! 5. The shared expert (`up`, relu2, `down`) runs on the full `h`, and its output is added.
//!
//! Precision: the dense arithmetic here is fp32. EXL3 projections run through `gb_exl3::forward`, the repo's float64 reference layer,
//! and come back as fp32. The fork's production path keeps fp16 intermediates (the router, the latent projection, the expert
//! activations); this reference does not, so it measures the math, not the production rounding.

use gb_exl3::checkpoint::LoadedGroup;
use gb_exl3::{codebook, forward, Codebook};
use std::sync::OnceLock;

/// A linear map `y = x @ W` over `rows` row-major inputs of width `k()`, returning `rows * n()` values.
pub trait Proj {
    fn k(&self) -> usize;
    fn n(&self) -> usize;
    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32>;
}

/// A dense weight stored as `[k, n]` row-major, so `apply` computes `x @ W`.
pub struct Dense {
    k: usize,
    n: usize,
    w: Vec<f32>,
}

impl Dense {
    /// `w` is `[k, n]` row-major.
    pub fn new(k: usize, n: usize, w: Vec<f32>) -> Self {
        assert_eq!(w.len(), k * n, "dense weight is not [{k}, {n}]");
        Self { k, n, w }
    }

    /// From a Hugging Face `nn.Linear` weight, `[out, in]` row-major, transposed to `[in, out]`.
    pub fn from_hf(out: usize, inp: usize, w: &[f32]) -> Self {
        assert_eq!(w.len(), out * inp, "dense weight is not [{out}, {inp}]");
        let mut t = vec![0f32; inp * out];
        for o in 0..out {
            for i in 0..inp {
                t[i * out + o] = w[o * inp + i];
            }
        }
        Self::new(inp, out, t)
    }
}

impl Proj for Dense {
    fn k(&self) -> usize {
        self.k
    }
    fn n(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        assert_eq!(x.len(), rows * self.k, "input is not [{rows}, {}]", self.k);
        let (k, n) = (self.k, self.n);
        let mut y = vec![0f32; rows * n];
        for r in 0..rows {
            let xr = &x[r * k..(r + 1) * k];
            let yr = &mut y[r * n..(r + 1) * n];
            for (i, &xi) in xr.iter().enumerate() {
                let wr = &self.w[i * n..(i + 1) * n];
                for (acc, &wj) in yr.iter_mut().zip(wr) {
                    *acc += xi * wj;
                }
            }
        }
        y
    }
}

/// The 65536-entry fp16 codebook for `cb`, built once per process.
fn codebook_table(cb: Codebook) -> &'static [u16] {
    static TABLES: [OnceLock<Vec<u16>>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    let i = match cb {
        Codebook::Inst3 => 0,
        Codebook::Mcg => 1,
        Codebook::Mul1 => 2,
    };
    TABLES[i].get_or_init(|| codebook(cb))
}

/// An EXL3 linear layer, applied through `gb_exl3::forward`: `y = ((x * suh) H W_q H) * svh (+ bias)`, with `W_q` decoded from the
/// trellis by the repo's host decoder. The decode runs on every call, so batch the rows of one projection into one call.
pub struct Exl3 {
    pub group: LoadedGroup,
}

impl Proj for Exl3 {
    fn k(&self) -> usize {
        self.group.group.k
    }
    fn n(&self) -> usize {
        self.group.group.n
    }
    fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        assert_eq!(x.len(), rows * self.k(), "EXL3 input is not [{rows}, {}]", self.k());
        let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let g = &self.group;
        forward(
            &xf,
            rows,
            &g.trellis,
            g.kt,
            g.nt,
            g.group.bits,
            codebook_table(g.group.codebook),
            &g.suh,
            &g.svh,
            g.bias.as_deref(),
        )
        .into_iter()
        .map(|v| v as f32)
        .collect()
    }
}

/// A gateless relu2 MLP, `down(relu2(up(x)))`. Routed experts and the shared expert have this shape.
pub struct Mlp {
    pub up: Box<dyn Proj>,
    pub down: Box<dyn Proj>,
}

impl Mlp {
    /// `x` is `[rows, up.k()]`; returns `[rows, down.n()]`.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        expert_forward(x, self.up.as_ref(), self.down.as_ref())
    }
}

/// The router: `gate` is the checkpoint's `gate.weight` as fp32, `[experts, hidden]` row-major; `bias` is
/// `e_score_correction_bias` (one per expert).
pub struct Router {
    pub gate: Vec<f32>,
    pub bias: Vec<f32>,
    pub hidden: usize,
    pub topk: usize,
    pub scale: f32,
}

impl Router {
    pub fn experts(&self) -> usize {
        self.bias.len()
    }

    /// Pre-sigmoid logits for `rows` tokens of width `hidden`, as `[rows, experts]`.
    pub fn logits(&self, h: &[f32], rows: usize) -> Vec<f32> {
        let (hid, e) = (self.hidden, self.experts());
        assert_eq!(h.len(), rows * hid, "router input is not [{rows}, {hid}]");
        assert_eq!(self.gate.len(), e * hid, "router gate is not [{e}, {hid}]");
        let mut out = vec![0f32; rows * e];
        for r in 0..rows {
            let hr = &h[r * hid..(r + 1) * hid];
            for j in 0..e {
                let gr = &self.gate[j * hid..(j + 1) * hid];
                out[r * e + j] = hr.iter().zip(gr).map(|(a, b)| a * b).sum();
            }
        }
        out
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Top-k routing for one token. `scores_logits` are the router's pre-sigmoid logits, one per expert.
///
/// Selection is by `sigmoid(logit) + bias`, and a lower expert index wins an equal selection value. The weights are the unbiased
/// sigmoid scores of the selected experts divided by their sum (plus `1e-20`, as `routing_ds3` does), times `scale`. Returns the
/// expert indices in selection order (largest selection value first) with their weights.
pub fn route(scores_logits: &[f32], bias: &[f32], topk: usize, scale: f32) -> (Vec<usize>, Vec<f32>) {
    assert_eq!(scores_logits.len(), bias.len(), "one bias per expert");
    assert!((1..=bias.len()).contains(&topk), "topk {topk} is outside 1..={}", bias.len());
    let scores: Vec<f32> = scores_logits.iter().map(|&x| sigmoid(x)).collect();
    let sel: Vec<f32> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
    assert!(sel.iter().all(|v| v.is_finite()), "router: a selection value is not finite");
    let mut order: Vec<usize> = (0..sel.len()).collect();
    order.sort_by(|&a, &b| sel[b].partial_cmp(&sel[a]).unwrap().then(a.cmp(&b)));
    order.truncate(topk);
    let denom = order.iter().map(|&i| scores[i]).sum::<f32>() + 1e-20;
    let weights = order.iter().map(|&i| scores[i] / denom * scale).collect();
    (order, weights)
}

/// One expert on `x_latent` (`[rows, up.k()]`): `down(relu2(up(x)))`, with `relu2(u) = max(u, 0)^2`. Returns `[rows, down.n()]`.
pub fn expert_forward(x_latent: &[f32], up: &dyn Proj, down: &dyn Proj) -> Vec<f32> {
    let k = up.k();
    assert!(k > 0 && x_latent.len() % k == 0, "expert input is not a multiple of {k}");
    let rows = x_latent.len() / k;
    let mut a = up.apply(x_latent, rows);
    for v in a.iter_mut() {
        let r = v.max(0.0);
        *v = r * r;
    }
    down.apply(&a, rows)
}

/// The whole latent-MoE block for `h` (`[rows, router.hidden]`). Returns `[rows, router.hidden]`.
///
/// `experts[e]` must be `Some` for every expert the router selects for some token; the rest may be `None`, so a caller can load only
/// the experts it needs. `fc1` is `[hidden -> latent]`, `fc2` is `[latent -> hidden]`, and `shared` is the shared expert on `h`.
pub fn latent_moe(h: &[f32], router: &Router, fc1: &dyn Proj, fc2: &dyn Proj, experts: &[Option<Mlp>], shared: &Mlp) -> Vec<f32> {
    let hid = router.hidden;
    assert!(hid > 0 && h.len() % hid == 0, "block input is not a multiple of {hid}");
    let rows = h.len() / hid;
    let e_count = router.experts();
    assert_eq!(experts.len(), e_count, "one expert slot per routed expert");
    assert_eq!((fc1.k(), fc2.n()), (hid, hid), "fc1 must map the hidden width in and fc2 must map it out");
    assert_eq!(fc1.n(), fc2.k(), "fc1 and fc2 must meet at the latent width");
    let latent = fc1.n();

    let logits = router.logits(h, rows);
    let mut per_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); e_count];
    for t in 0..rows {
        let (idx, w) = route(&logits[t * e_count..(t + 1) * e_count], &router.bias, router.topk, router.scale);
        for (e, we) in idx.into_iter().zip(w) {
            per_expert[e].push((t, we));
        }
    }

    let x = fc1.apply(h, rows);
    let mut routed = vec![0f32; rows * latent];
    for (e, list) in per_expert.iter().enumerate() {
        if list.is_empty() {
            continue;
        }
        let mlp = experts[e].as_ref().unwrap_or_else(|| panic!("latent_moe: expert {e} is selected but was not loaded"));
        let mut batch = Vec::with_capacity(list.len() * latent);
        for &(t, _) in list {
            batch.extend_from_slice(&x[t * latent..(t + 1) * latent]);
        }
        let y = mlp.forward(&batch);
        for (r, &(t, w)) in list.iter().enumerate() {
            let dst = &mut routed[t * latent..(t + 1) * latent];
            for (d, s) in dst.iter_mut().zip(&y[r * latent..(r + 1) * latent]) {
                *d += w * s;
            }
        }
    }

    let routed_out = fc2.apply(&routed, rows);
    let shared_out = shared.forward(h);
    routed_out.iter().zip(&shared_out).map(|(a, b)| a + b).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(n: usize) -> Dense {
        let mut w = vec![0f32; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        Dense::new(n, n, w)
    }

    fn scaled_identity(n: usize, s: f32) -> Dense {
        let mut w = vec![0f32; n * n];
        for i in 0..n {
            w[i * n + i] = s;
        }
        Dense::new(n, n, w)
    }

    #[test]
    fn a_tie_goes_to_the_lower_index() {
        let (idx, w) = route(&[0.0; 6], &[0.0; 6], 3, 1.0);
        assert_eq!(idx, vec![0, 1, 2]);
        for x in &w {
            assert!((x - 1.0 / 3.0).abs() < 1e-7);
        }
    }

    #[test]
    fn selection_uses_the_bias_and_the_weights_do_not() {
        // scores = sigmoid(logit): 0.9, 0.5, 0.6, 0.1. With bias [0, 0, 0.5, 0] the selection values are 0.9, 0.5, 1.1, 0.1, so the
        // top two are experts 2 and 0. Their weights come from the unbiased scores 0.6 and 0.9.
        let logit = |s: f32| (s / (1.0 - s)).ln();
        let logits = [logit(0.9), logit(0.5), logit(0.6), logit(0.1)];
        let (idx, w) = route(&logits, &[0.0, 0.0, 0.5, 0.0], 2, 5.0);
        assert_eq!(idx, vec![2, 0]);
        assert!((w[0] - 5.0 * 0.6 / 1.5).abs() < 1e-5, "{w:?}");
        assert!((w[1] - 5.0 * 0.9 / 1.5).abs() < 1e-5, "{w:?}");
    }

    #[test]
    fn weights_sum_to_the_scale() {
        let logits: Vec<f32> = (0..16).map(|i| (i as f32 - 7.5) * 0.3).collect();
        let (_, w) = route(&logits, &[0.1; 16], 4, 5.0);
        assert!((w.iter().sum::<f32>() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn relu2_then_down() {
        // up = I, down = diag(1, 3): x = [-1, 2] -> relu2 -> [0, 4] -> [0, 12].
        let y = expert_forward(&[-1.0, 2.0], &identity(2), &Dense::new(2, 2, vec![1.0, 0.0, 0.0, 3.0]));
        assert_eq!(y, vec![0.0, 12.0]);
    }

    #[test]
    fn hf_weights_are_transposed_to_in_by_out() {
        // HF [out=2, in=3]: rows are outputs. x = [1, 1, 1] gives [1 + 2 + 3, 4 + 5 + 6].
        let d = Dense::from_hf(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!((d.k(), d.n()), (3, 2));
        assert_eq!(d.apply(&[1.0, 1.0, 1.0], 1), vec![6.0, 15.0]);
    }

    #[test]
    fn latent_moe_on_a_two_expert_block() {
        // fc1 = fc2 = I, shared expert = 0, scale 2, top-1. Token 0 (logits [1, 0]) and token 1 (logits [0, -1]) both select expert 1
        // through the bias 0.6. Expert 1 is relu2 then 3 * I, so token 0 gives 2 * 3 * relu2([1, 0]) = [6, 0] and token 1 gives 0.
        let router = Router { gate: vec![1.0, 0.0, 0.0, 1.0], bias: vec![0.0, 0.6], hidden: 2, topk: 1, scale: 2.0 };
        let experts = vec![None, Some(Mlp { up: Box::new(identity(2)), down: Box::new(scaled_identity(2, 3.0)) })];
        let shared = Mlp { up: Box::new(Dense::new(2, 2, vec![0.0; 4])), down: Box::new(Dense::new(2, 2, vec![0.0; 4])) };
        let out = latent_moe(&[1.0, 0.0, 0.0, -1.0], &router, &identity(2), &identity(2), &experts, &shared);
        assert_eq!(out.len(), 4);
        assert!((out[0] - 6.0).abs() < 1e-5 && out[1].abs() < 1e-5, "{out:?}");
        assert!(out[2].abs() < 1e-5 && out[3].abs() < 1e-5, "{out:?}");
    }

    #[test]
    #[should_panic(expected = "expert 0 is selected but was not loaded")]
    fn a_selected_expert_that_is_not_loaded_is_refused() {
        let router = Router { gate: vec![1.0, 0.0, 0.0, 1.0], bias: vec![0.0, 0.0], hidden: 2, topk: 1, scale: 1.0 };
        let shared = Mlp { up: Box::new(identity(2)), down: Box::new(identity(2)) };
        latent_moe(&[1.0, 0.5], &router, &identity(2), &identity(2), &[None, None], &shared);
    }
}
