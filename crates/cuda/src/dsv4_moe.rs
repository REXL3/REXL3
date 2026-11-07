//! DeepSeek-V4.1's MoE router and hyper-connection mix, at this pack's shapes: hidden 5120, 384 routed experts + 1 shared,
//! top-6, `sqrtsoftplus` scoring, `noaux_tc` selection, `norm_topk_prob`, `routed_scaling_factor` 1.5, `swiglu_limit` 10.0,
//! `hc_mult` 4, `hc_sinkhorn_iters` 20, `hc_eps` 1e-6. Kernels: `csrc/dsv4_moe.cu` (design and the row-invariance
//! argument are in that file's header).
//!
//! The reference is `model.py` lines 794-996 (`Gate`, `Expert`, `MoE`, `Block`) and `kernel.py` lines 407-474
//! (`hc_split_sinkhorn`); every host function here is a direct transcription of those, and the tests pin them against one.
//!
//! # The interface the model code calls (nothing here synchronizes with the host)
//!
//! ```text
//! let tab  = SqrtSoftplusTable::new()?;                       // 65,536 fp32 (256 KiB), fp16-logit scoring
//! let rtr  = Dsv4Router::new(&gate_bf16, &bias, &bias_vl, Dsv4Dims::V41)?;
//! rtr.logits(&x, Dtype::BF16, rows, &mut logits, &st)?;       // [rows, 384] fp32
//! rtr.select(&logits, rows, Some(&image_mask), &mut pick, &mut wts, &st)?;   // [rows, 6] i32 / fp16
//! hc.partials(&x, &site.fn_buf, rows, &mut work, &st)?;       // 25 dots per atom, [rows][25][40]
//! hc.finalize(&x, &site, rows, &mut work, &st)?;              // pre [rows,4], post [rows,4], comb [rows,4,4]
//! hc.apply(&mut x, &y, Dtype::BF16, rows, &work, &st)?;       // x[h] = post[h] y + sum_g comb[g][h] x[g]
//! ```
//!
//! # The router, exactly (Gate.forward)
//!
//! ```text
//! scores  = (x.float() @ W.float()^T) / gate_temp                                  gate_temp = 1.0 here
//! scores  = sqrt(softplus(scores))
//! bias    = bias_vl if (image_mask is not None and this token is in an image span) else bias
//! indices = topk(scores + bias, 6)[1]        -- the dual bias STEERS SELECTION ONLY
//! weights = scores.gather(indices)           -- the weight is the UNBIASED score
//! weights = weights / (weights.sum(-1) + 1e-20)   -- norm_topk_prob; the 1e-20 is not norm_eps
//! weights = weights * 1.5                    -- routed_scaling_factor
//! ```
//!
//! `bias_vl` is a second `[384]` fp32 vector (`noaux_tc_for_vl` in training): for a token inside an image span it replaces
//! `bias` in the selection sum, and for every other token it is not read at all. It never touches the returned weight.
//!
//! # Hyper-connections
//!
//! Four fp32 residual streams of 5120 per row, `x[row][stream][d]`. One site computes 24 dots of the flattened 20480-value
//! row against `hc_attn_fn` / `hc_ffn_fn` (`[24][20480]`) plus the row's sum of squares, scales them by
//! `rsqrt(ss/20480 + norm_eps)`, and splits them into `pre` (4, `sigmoid + hc_eps`), `post` (4, `2 sigmoid`) and `comb`
//! (16, row softmax then a 20-iteration Sinkhorn to doubly stochastic).
//!
//! The 4096/16384/32-atom instantiation of the same arithmetic is [`crate::mhc`] (GLM-5.3-Flash). Both are the same code
//! with a different `hidden`: [`hc_mixes_host`] takes it as an argument, and the test
//! `hc_4096_and_5120_agree_under_zero_padding` proves the two agree bit for bit on zero-padded tensors.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use core::ffi::c_void;

use crate::rt::{CudaError, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};

/// Model width (gate/up K, down N).
pub const HIDDEN: usize = 5120;
/// Expert width (gate/up N, down K).
pub const INTER: usize = 2304;
/// Routed experts.
pub const EXPERTS: usize = 384;
/// Experts a token picks.
pub const TOPK: usize = 6;
/// `routed_scaling_factor`.
pub const ROUTE_SCALE: f32 = 1.5;
/// `swiglu_limit` (`<= 0` disables the clamps).
pub const SWIGLU_LIMIT: f32 = 10.0;
/// Residual streams per row (`hc_mult`).
pub const STREAMS: usize = 4;
/// Values per row: `STREAMS * HIDDEN`.
pub const ROW: usize = STREAMS * HIDDEN;
/// Rows of `fn`: 4 pre, 4 post, 16 comb.
pub const NFN: usize = 24;
/// `d` per atom.
pub const ATOM_D: usize = 128;
/// Atoms per row: `HIDDEN / ATOM_D` = 40.
pub const ATOMS: usize = HIDDEN / ATOM_D;
/// Floats of partials per row (`25 * ATOMS`).
pub const PART_FLOATS_PER_ROW: usize = (NFN + 1) * ATOMS;
/// Rows a single launch accepts.
pub const MAX_ROWS: usize = 65535;

#[allow(non_snake_case)]
mod ffi {
    use core::ffi::c_void;
    crate::launchers! {
        pub fn gb_dsv4_router_logits(x: *const c_void, x_type: i32, w: *const c_void, logits: *mut c_void, rows: i32, s: *mut c_void) -> i32;
        pub fn gb_dsv4_router_select(
            logits: *const c_void,
            tab: *const c_void,
            bias: *const c_void,
            bias_vl: *const c_void,
            mask: *const c_void,
            pick: *mut c_void,
            wts: *mut c_void,
            rows: i32,
            f16_log: i32,
            scale: f32,
            top_p: f32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_hc_partials(x: *const c_void, fnw: *const c_void, part: *mut c_void, rows: i32, s: *mut c_void) -> i32;
        pub fn gb_dsv4_hc_partials_and_collapse(
            x: *const c_void,
            fnw: *const c_void,
            part: *mut c_void,
            pre_in: *const f32,
            collapsed_out: *mut f32,
            rows: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_hc_apply_partials_and_collapse(
            x: *mut c_void,
            fnw: *const c_void,
            part: *mut c_void,
            pre_in: *const f32,
            collapsed_out: *mut f32,
            y_prev: *const c_void,
            post_prev: *const c_void,
            comb_prev: *const c_void,
            rows: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_hc_finalize(
            x: *const c_void,
            part: *const c_void,
            base: *const c_void,
            scale: *const c_void,
            pre_out: *mut c_void,
            post: *mut c_void,
            comb: *mut c_void,
            rows: i32,
            rms_eps: f32,
            hc_eps: f32,
            iters: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_hc_apply(
            x: *mut c_void,
            y: *const c_void,
            y_type: i32,
            post: *const c_void,
            comb: *const c_void,
            rows: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_swiglu(
            g: *const c_void,
            u: *const c_void,
            w: *const c_void,
            out: *mut c_void,
            n: i32,
            inter: i32,
            limit: f32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_swiglu_f16(
            g: *const c_void,
            u: *const c_void,
            w: *const c_void,
            out: *mut c_void,
            n: i32,
            inter: i32,
            limit: f32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_moe_group(
            picks: *const i32,
            wts: *const c_void,
            counts: *mut i32,
            offsets: *mut i32,
            perm: *mut i32,
            pinv: *mut i32,
            wpack: *mut c_void,
            slots: i32,
            n_exp: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_moe_gather(
            x: *const c_void,
            perm: *const i32,
            xp: *mut c_void,
            slots: i32,
            topk: i32,
            h: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_moe_scatter(
            shared: *const c_void,
            yp: *const c_void,
            pinv: *const i32,
            wpack: *const c_void,
            out_f32: *mut c_void,
            out_f16: *mut c_void,
            rows: i32,
            topk: i32,
            h: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_moe_fast_experts(
            hidden_in: *const c_void,
            shared_f16: *mut c_void,
            picks: *const i32,
            wts: *const c_void,
            moe_out_f32: *mut c_void,
            descs: *const c_void,
            counts: *mut i32,
            offsets: *mut i32,
            perm: *mut i32,
            pinv: *mut i32,
            wpack: *mut c_void,
            xh_w2: *mut c_void,
            y: *mut c_void,
            ws: *mut c_void,
            ws_bytes: usize,
            rows: i32,
            topk: i32,
            h: i32,
            inter: i32,
            n_exp: i32,
            gemm_only: i32,
            limit: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_moe_fast_experts_tp_pre(
            hidden_in: *const c_void,
            picks: *const i32,
            wts: *const c_void,
            descs: *const c_void,
            counts: *mut i32,
            offsets: *mut i32,
            perm: *mut i32,
            pinv: *mut i32,
            wpack: *mut c_void,
            xh_w2: *mut c_void,
            y: *mut c_void,
            ws: *mut c_void,
            ws_bytes: usize,
            rows: i32,
            topk: i32,
            h: i32,
            inter: i32,
            n_exp: i32,
            e_lo: i32,
            e_hi: i32,
            gemm_only: i32,
            limit: f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_moe_fast_experts_tp_post(
            shared_f16: *mut c_void,
            y: *const c_void,
            y_peer: *const c_void,
            offsets: *const i32,
            pinv: *const i32,
            wpack: *const c_void,
            moe_out_f32: *mut c_void,
            rows: i32,
            topk: i32,
            h: i32,
            e_lo: i32,
            e_hi: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_interleave_o_halves(
            left: *const c_void,
            right: *const c_void,
            dst: *mut c_void,
            rows: i32,
            half_f16: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_sqrtsoftplus(x: *const c_void, y: *mut c_void, n: i32, s: *mut c_void) -> i32;
        pub fn gb_dsv4_engram_gate_add(
            x: *mut f32,
            kv: *const c_void,
            qw: *const c_void,
            kw: *const c_void,
            rows: i32,
            hc: i32,
            dim: i32,
            eps: f32,
            s: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_hc_collapse_f32(
            x: *const f32,
            pre: *const f32,
            y: *mut f32,
            rows: i32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
        pub fn gb_dsv4_mtp_hc_pre(
            x: *const f32,
            pre: *const f32,
            y: *mut c_void,
            rows: i32,
            dim: i32,
            stream: *mut c_void,
        ) -> i32;
    }
}

fn i(v: usize) -> Result<i32, CudaError> {
    if v > MAX_ROWS {
        return Err(CudaError(1));
    }
    Ok(v as i32)
}

fn i32_of(v: usize) -> Result<i32, CudaError> {
    i32::try_from(v).map_err(|_| CudaError(1))
}

// ================================================================================================================================
// 1. sqrtsoftplus
// ================================================================================================================================

/// `sqrt(softplus(x))` in its stable form, fp32. `softplus(x) = log1p(exp(-|x|)) + max(x, 0)`: the exponent never
/// overflows (its argument is `<= 0`), and the `log1p` keeps full relative precision as `x -> -inf` instead of cancelling
/// in `log(1 + exp(x))`. The same bits as `sqrtsoftplus_f` in `csrc/dsv4_moe.cu`.
#[inline]
pub fn sqrtsoftplus(x: f32) -> f32 {
    let m = x.abs();
    ((-m).exp().ln_1p() + x.max(0.0)).sqrt()
}

/// [`sqrtsoftplus`] in fp64: the reference the fp32 path is measured against.
#[inline]
pub fn sqrtsoftplus_f64(x: f64) -> f64 {
    let m = x.abs();
    ((-m).exp().ln_1p() + x.max(0.0)).sqrt()
}

/// The 65,536-entry fp32 table indexed by fp16 bits (finite entries only; +inf maps to +inf, NaN to NaN). One per process.
pub fn sqrtsoftplus_table_host() -> Vec<f32> {
    (0u32..65536)
        .map(|h| {
            let x = f16_to_f64(h as u16);
            if x.is_nan() {
                f32::NAN
            } else {
                sqrtsoftplus_f64(x) as f32
            }
        })
        .collect()
}

/// The table on the device, for the fp16-logit scoring path.
pub struct SqrtSoftplusTable {
    #[allow(dead_code)]
    buf: DevBuf,
}

impl SqrtSoftplusTable {
    pub fn new() -> Result<Self, CudaError> {
        Ok(Self { buf: DevBuf::from_slice(&sqrtsoftplus_table_host())? })
    }

    /// The host bits of the table this object was built from (the tests compare against it).
    pub fn host(&self) -> Vec<f32> {
        sqrtsoftplus_table_host()
    }
}

/// `y = sqrtsoftplus(x)` elementwise, on the device. Asynchronous on `st`.
pub fn sqrtsoftplus_dev(x: &DevBuf, y: &mut DevBuf, n: usize, st: &Stream) -> Result<(), CudaError> {
    if n == 0 {
        return Ok(());
    }
    if x.bytes() < n * 4 || y.bytes() < n * 4 {
        return Err(CudaError(1));
    }
    unsafe { CudaError::check(ffi::gb_dsv4_sqrtsoftplus(x.ptr(), y.ptr(), i32_of(n)?, st.raw())) }
}

// ================================================================================================================================
// 2. the router
// ================================================================================================================================

/// Shapes and constants of one MoE layer's router.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dsv4Dims {
    /// Model width: 5120.
    pub hidden: usize,
    /// Routed experts: 384.
    pub experts: usize,
    /// Experts a token picks: 6.
    pub topk: usize,
    /// `routed_scaling_factor`: 1.5.
    pub scale: f32,
    /// `gate_temp`: 1.0 (folded into the logits by the caller when it is not).
    pub gate_temp: f32,
}

impl Dsv4Dims {
    pub const V41: Dsv4Dims = Dsv4Dims { hidden: HIDDEN, experts: EXPERTS, topk: TOPK, scale: ROUTE_SCALE, gate_temp: 1.0 };
}

/// The host transcription of `Gate.forward`: `logits` is `[rows, experts]` fp32 (already divided by `gate_temp`), and the
/// result is the picks `[rows, topk]` plus the fp32 weights `[rows, topk]` (unrounded; the engine stores fp16).
///
/// `image_mask[i] == true` selects `bias_vl[i]`-style: the whole `bias_vl` vector replaces `bias` for that row, exactly as
/// `torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)` does. Ties in `score + bias` go to the lower expert id
/// (`torch.topk` makes no such promise, so the engine fixes one; the device kernel and this function agree).
pub fn select_host(
    logits: &[f32],
    rows: usize,
    d: Dsv4Dims,
    bias: &[f32],
    bias_vl: Option<&[f32]>,
    image_mask: Option<&[bool]>,
) -> (Vec<i32>, Vec<f32>) {
    assert!(bias.len() == d.experts, "bias must be [experts]");
    assert!(logits.len() == rows * d.experts, "logits must be [rows, experts]");
    if let Some(m) = image_mask {
        assert!(m.len() == rows, "image_mask must be [rows]");
    }
    if bias_vl.is_some() {
        assert!(bias_vl.unwrap().len() == d.experts, "bias_vl must be [experts]");
    }
    let mut pick = vec![0i32; rows * d.topk];
    let mut wts = vec![0f32; rows * d.topk];
    let mut order: Vec<usize> = vec![0; d.experts];
    for r in 0..rows {
        let lg = &logits[r * d.experts..(r + 1) * d.experts];
        let b = match (image_mask, bias_vl) {
            (Some(m), Some(vl)) if m[r] => vl,
            _ => bias,
        };
        // scores = softplus(logit).sqrt(), then the bias steers selection only
        for (e, o) in order.iter_mut().enumerate() {
            *o = e;
        }
        let key = |e: usize| sqrtsoftplus(lg[e]) + b[e];
        order.sort_by(|&a, &c| key(c).total_cmp(&key(a)).then(a.cmp(&c)));
        let mut sum = 0f32;
        for k in 0..d.topk {
            sum += sqrtsoftplus(lg[order[k]]);
        }
        let den = sum + 1e-20;
        for k in 0..d.topk {
            let e = order[k];
            pick[r * d.topk + k] = e as i32;
            wts[r * d.topk + k] = (sqrtsoftplus(lg[e]) / den) * d.scale;
        }
    }
    (pick, wts)
}

/// [`select_host`] with the weights rounded to fp16 as the engine stores them.
pub fn select_host_f16(
    logits: &[f32],
    rows: usize,
    d: Dsv4Dims,
    bias: &[f32],
    bias_vl: Option<&[f32]>,
    image_mask: Option<&[bool]>,
) -> (Vec<i32>, Vec<u16>) {
    let (p, w) = select_host(logits, rows, d, bias, bias_vl, image_mask);
    (p, w.iter().map(|&v| f64_to_f16(v as f64)).collect())
}

/// Routed-expert top-p pruning threshold for decode (`rows <= 8`): defaults to `0.85` (overridable via
/// `REXL3_DSV41_MOE_TOPP`; set `REXL3_DSV41_MOE_TOPP=1.0` to disable). For `rows > 8` (prefill), returns `1.0`.
pub fn runtime_env(key: &str) -> Option<String> {
    static KNOBS_PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    if let Some(path) = KNOBS_PATH.get_or_init(|| std::env::var("REXL3_DSV41_KNOBS_FILE").ok()) {
        if let Ok(txt) = std::fs::read_to_string(path) {
            for line in txt.lines() {
                let line = line.trim();
                if let Some((k, v)) = line.split_once('=') {
                    if k.trim() == key {
                        return Some(v.trim().to_string());
                    }
                }
            }
        }
    }
    std::env::var(key).ok()
}

pub fn moe_top_p(rows: usize) -> f32 {
    if rows > 16 {
        return 1.0;
    }
    let default_top_p = 1.0;
    runtime_env("REXL3_DSV41_MOE_TOPP")
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default_top_p)
}

/// Host reference for [`Dsv4Router::select_top_p`]: when `top_p < 1.0`, sorts the `topk` selected `(pick, unscaled_prob)`
/// pairs in descending order of `unscaled_prob`, keeps at least `min_k = 3` experts (`j = 0, 1, 2`) and any subsequent
/// expert while cumulative probability `cum_p < top_p` (setting pruned picks to `-1` and weights to `0.0`), and
/// renormalizes kept weights to sum to `d.scale`.
pub fn select_top_p_host(
    logits: &[f32],
    rows: usize,
    d: Dsv4Dims,
    bias: &[f32],
    bias_vl: Option<&[f32]>,
    image_mask: Option<&[bool]>,
    top_p: f32,
) -> (Vec<i32>, Vec<f32>) {
    // NaN is excluded upstream by the config parse; >= is the readable form of !(< 1.0)
    if top_p >= 1.0 {
        return select_host(logits, rows, d, bias, bias_vl, image_mask);
    }
    assert!(bias.len() == d.experts, "bias must be [experts]");
    assert!(logits.len() == rows * d.experts, "logits must be [rows, experts]");
    if let Some(m) = image_mask {
        assert!(m.len() == rows, "image_mask must be [rows]");
    }
    if bias_vl.is_some() {
        assert!(bias_vl.unwrap().len() == d.experts, "bias_vl must be [experts]");
    }
    let mut pick = vec![-1i32; rows * d.topk];
    let mut wts = vec![0f32; rows * d.topk];
    let mut order: Vec<usize> = vec![0; d.experts];
    for r in 0..rows {
        let lg = &logits[r * d.experts..(r + 1) * d.experts];
        let b = match (image_mask, bias_vl) {
            (Some(m), Some(vl)) if m[r] => vl,
            _ => bias,
        };
        for (e, o) in order.iter_mut().enumerate() {
            *o = e;
        }
        let key = |e: usize| sqrtsoftplus(lg[e]) + b[e];
        order.sort_by(|&a, &c| key(c).total_cmp(&key(a)).then(a.cmp(&c)));
        let mut chosen: Vec<(usize, f32)> = (0..d.topk).map(|k| (order[k], sqrtsoftplus(lg[order[k]]))).collect();
        chosen.sort_by(|&(ea, sa), &(eb, sb)| sb.total_cmp(&sa).then(ea.cmp(&eb)));
        let sum: f32 = chosen.iter().map(|&(_, s)| s).sum();
        let sum_den = sum + 1e-20;
        let mut cum_p = 0f32;
        let mut kept_sum = 0f32;
        let mut kept = vec![false; d.topk];
        for (j, &(_, s)) in chosen.iter().enumerate() {
            if j < 3 || cum_p < top_p {
                kept[j] = true;
                cum_p += s / sum_den;
                kept_sum += s;
            }
        }
        let den = kept_sum + 1e-20;
        for j in 0..d.topk {
            if kept[j] {
                pick[r * d.topk + j] = chosen[j].0 as i32;
                wts[r * d.topk + j] = (chosen[j].1 / den) * d.scale;
            } else {
                pick[r * d.topk + j] = -1;
                wts[r * d.topk + j] = 0.0;
            }
        }
    }
    (pick, wts)
}

/// One layer's router on the device: the bf16 gate `[experts, hidden]`, and the two fp32 `[experts]` biases.
pub struct Dsv4Router {
    gate: DevBuf,
    bias: DevBuf,
    bias_vl: DevBuf,
    pub d: Dsv4Dims,
}

impl Dsv4Router {
    /// `gate_bf16` is the pack's `ffn.gate.weight` (bf16 `[experts, hidden]` as raw bits), `bias` / `bias_vl` its
    /// `ffn.gate.bias` / `ffn.gate.bias_vl` (fp32 `[experts]`).
    pub fn new(gate_bf16: &[u16], bias: &[f32], bias_vl: &[f32], d: Dsv4Dims) -> Result<Self, CudaError> {
        if d.experts == 0
            || d.topk == 0
            || d.topk > d.experts
            || d.topk > 16
            || d.hidden == 0
            || d.experts > 512
            || gate_bf16.len() != d.experts * d.hidden
            || bias.len() != d.experts
            || bias_vl.len() != d.experts
        {
            return Err(CudaError(1));
        }
        Ok(Self {
            gate: DevBuf::from_slice(gate_bf16)?,
            bias: DevBuf::from_slice(bias)?,
            bias_vl: DevBuf::from_slice(bias_vl)?,
            d,
        })
    }

    /// logits `[rows, experts]` fp32 = `x @ gate^T` (bf16 weight, f32 accumulate). Asynchronous on `st`.
    pub fn logits(
        &self,
        x: &DevBuf,
        x_dtype: crate::exl3_linear::Dtype,
        rows: usize,
        logits: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let es = x_dtype.size();
        if rows == 0 {
            return Ok(());
        }
        if x.bytes() < rows * self.d.hidden * es || logits.bytes() < rows * self.d.experts * 4 {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_dsv4_router_logits(
                x.ptr(),
                x_dtype as i32,
                self.gate.ptr(),
                logits.ptr(),
                i(rows)?,
                st.raw(),
            ))
        }
    }

    /// picks `[rows, topk]` i32 and weights `[rows, topk]` fp16 from fp32 logits. `image_mask` is `[rows]` u8 (0 or 1) or
    /// `None`; with a mask the row's `bias_vl` replaces its `bias` in the selection sum. Uses [`moe_top_p`]`(rows)` for
    /// decode top-p expert pruning when `rows <= 8`.
    pub fn select(
        &self,
        logits: &DevBuf,
        rows: usize,
        image_mask: Option<&DevBuf>,
        pick: &mut DevBuf,
        wts: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.select_top_p(logits, rows, image_mask, 1.0, pick, wts, st)
    }

    /// [`Self::select`] with an explicit `top_p` threshold (`>= 1.0` disables pruning).
    pub fn select_top_p(
        &self,
        logits: &DevBuf,
        rows: usize,
        image_mask: Option<&DevBuf>,
        top_p: f32,
        pick: &mut DevBuf,
        wts: &mut DevBuf,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        if logits.bytes() < rows * self.d.experts * 4
            || pick.bytes() < rows * self.d.topk * 4
            || wts.bytes() < rows * self.d.topk * 2
        {
            return Err(CudaError(1));
        }
        if let Some(m) = image_mask {
            if m.bytes() < rows {
                return Err(CudaError(1));
            }
        }
        let mask = image_mask.map_or(core::ptr::null_mut(), |m| m.ptr());
        unsafe {
            CudaError::check(ffi::gb_dsv4_router_select(
                logits.ptr(),
                core::ptr::null(),
                self.bias.ptr(),
                self.bias_vl.ptr(),
                mask,
                pick.ptr(),
                wts.ptr(),
                i(rows)?,
                0,
                self.d.scale,
                top_p,
                st.raw(),
            ))
        }
    }
}

// ================================================================================================================================
// 3. hyper-connections
// ================================================================================================================================

/// Numeric constants of the mixer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HcCfg {
    /// The epsilon of the unweighted RMS over the 20480 stream values (`args.norm_eps`, 1e-20 for this pack: `hc_mixes`
    /// normalises by `self.norm_eps`, not by a separate `rms_norm_eps`).
    pub rms_eps: f32,
    /// `hc_eps`: added to `pre` and to the softmax / Sinkhorn denominators (1e-6).
    pub hc_eps: f32,
    /// `hc_sinkhorn_iters` (20): one column normalisation, then `iters - 1` rounds of (row, column).
    pub sinkhorn_iters: i32,
}

impl HcCfg {
    pub const V41: HcCfg = HcCfg { rms_eps: 1e-20, hc_eps: 1e-6, sinkhorn_iters: 20 };
}

/// The Sinkhorn tail of `hc_split_sinkhorn_kernel` on a `[4, 4]` comb in place: row softmax, `+ eps`, one column
/// normalisation, then `iters - 1` rounds of (row, column), every division `/(sum + eps)`.
pub fn sinkhorn_host(c: &mut [f32; 16], iters: i32, eps: f32) {
    for j in 0..STREAMS {
        let mut mx = c[j * STREAMS];
        for k in 1..STREAMS {
            mx = mx.max(c[j * STREAMS + k]);
        }
        let mut s = 0f32;
        for k in 0..STREAMS {
            c[j * STREAMS + k] = (c[j * STREAMS + k] - mx).exp();
            s += c[j * STREAMS + k];
        }
        for k in 0..STREAMS {
            c[j * STREAMS + k] = c[j * STREAMS + k] / s + eps;
        }
    }
    let mut col = [0f32; STREAMS];
    for k in 0..STREAMS {
        for j in 0..STREAMS {
            col[k] += c[j * STREAMS + k];
        }
        for j in 0..STREAMS {
            c[j * STREAMS + k] /= col[k] + eps;
        }
    }
    for _ in 1..iters {
        for j in 0..STREAMS {
            let mut s = 0f32;
            for k in 0..STREAMS {
                s += c[j * STREAMS + k];
            }
            for k in 0..STREAMS {
                c[j * STREAMS + k] /= s + eps;
            }
        }
        for k in 0..STREAMS {
            let mut s = 0f32;
            for j in 0..STREAMS {
                s += c[j * STREAMS + k];
            }
            for j in 0..STREAMS {
                c[j * STREAMS + k] /= s + eps;
            }
        }
    }
}

/// Split 25 `mixes` into `pre` / `post` / `comb` (`hc_split_sinkhorn`): the three scale/base pairs, then the Sinkhorn.
pub fn hc_split_host(mixes: &[f32], scale: &[f32; 3], base: &[f32], cfg: HcCfg) -> ([f32; 4], [f32; 4], [f32; 16]) {
    assert!(mixes.len() == NFN + 1 && base.len() == NFN);
    let mut pre = [0f32; STREAMS];
    let mut post = [0f32; STREAMS];
    let mut comb = [0f32; 16];
    for h in 0..STREAMS {
        pre[h] = 1.0 / (1.0 + (-mixes[h].mul_add(scale[0], base[h])).exp()) + cfg.hc_eps;
        post[h] = 2.0 * (1.0 / (1.0 + (-mixes[h + STREAMS].mul_add(scale[1], base[h + STREAMS])).exp()));
    }
    for j in 0..16 {
        comb[j] = mixes[STREAMS * 2 + j].mul_add(scale[2], base[STREAMS * 2 + j]);
    }
    sinkhorn_host(&mut comb, cfg.sinkhorn_iters, cfg.hc_eps);
    (pre, post, comb)
}

/// The 25 atom partials of one row, in the device's fixed order: atom `a`, lane `l` owns `d = 128a + 4l .. +3` of every
/// stream, its quantity is an fma chain over its 16 values in (stream, d) order, and the 32 lane partials fold through the
/// xor butterfly `16, 8, 4, 2, 1`. `hidden` must be a multiple of 128.
pub fn hc_dots_host(x_row: &[f32], hidden: usize, fnw: &[f32]) -> [f32; NFN + 1] {
    let row = STREAMS * hidden;
    assert!(x_row.len() == row && fnw.len() == NFN * row && hidden % ATOM_D == 0);
    let natom = hidden / ATOM_D;
    let mut q = [0f32; NFN + 1];
    for a in 0..natom {
        for (qi, out) in q.iter_mut().enumerate() {
            let mut lane = [0f32; 32];
            for (l, lv) in lane.iter_mut().enumerate() {
                let d0 = a * ATOM_D + l * 4;
                let mut acc = 0f32;
                for h in 0..STREAMS {
                    for e in 0..4 {
                        let xv = x_row[h * hidden + d0 + e];
                        acc = if qi == NFN { xv.mul_add(xv, acc) } else { fnw[qi * row + h * hidden + d0 + e].mul_add(xv, acc) };
                    }
                }
                *lv = acc;
            }
            let mut m = 16;
            while m >= 1 {
                for l in 0..m {
                    lane[l] += lane[l + m];
                }
                m >>= 1;
            }
            *out += lane[0];
        }
    }
    q
}

/// `Block.hc_mixes` on the host: `mixes = (x . fn^T) * rsqrt(sum x^2 / mean_len + rms_eps)` then [`hc_split_host`].
///
/// `mean_len` is the length of the RMS mean. It is `STREAMS * hidden` in the model; it is a separate argument so the
/// zero-padding test can hold it fixed while it changes the layout.
pub fn hc_mixes_host(
    x: &[f32],
    rows: usize,
    hidden: usize,
    fnw: &[f32],
    scale: &[f32; 3],
    base: &[f32],
    mean_len: usize,
    cfg: HcCfg,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let row = STREAMS * hidden;
    assert!(x.len() == rows * row);
    let mut pre = vec![0f32; rows * STREAMS];
    let mut post = vec![0f32; rows * STREAMS];
    let mut comb = vec![0f32; rows * 16];
    for r in 0..rows {
        let mut q = hc_dots_host(&x[r * row..(r + 1) * row], hidden, fnw);
        let rmr = (q[NFN] / mean_len as f32 + cfg.rms_eps).sqrt().recip();
        for v in q.iter_mut() {
            *v *= rmr;
        }
        let (p, s, c) = hc_split_host(&q, scale, base, cfg);
        pre[r * STREAMS..(r + 1) * STREAMS].copy_from_slice(&p);
        post[r * STREAMS..(r + 1) * STREAMS].copy_from_slice(&s);
        comb[r * 16..(r + 1) * 16].copy_from_slice(&c);
    }
    (pre, post, comb)
}

/// `Block.hc_pre` on the host: `sum_h pre[h] x[h][d]` (fma chain in `h` order).
pub fn hc_pre_host(x: &[f32], rows: usize, hidden: usize, pre: &[f32]) -> Vec<f32> {
    let row = STREAMS * hidden;
    let mut out = vec![0f32; rows * hidden];
    for r in 0..rows {
        for d in 0..hidden {
            let mut acc = pre[r * STREAMS] * x[r * row + d];
            for h in 1..STREAMS {
                acc = pre[r * STREAMS + h].mul_add(x[r * row + h * hidden + d], acc);
            }
            out[r * hidden + d] = acc;
        }
    }
    out
}

/// `Block.hc_post` on the host: `post[h] y[d] + sum_g comb[g][h] x[g][d]`.
pub fn hc_post_host(x: &[f32], y: &[f32], rows: usize, hidden: usize, post: &[f32], comb: &[f32]) -> Vec<f32> {
    let row = STREAMS * hidden;
    let mut out = vec![0f32; rows * row];
    for r in 0..rows {
        for h in 0..STREAMS {
            for d in 0..hidden {
                let mut o = post[r * STREAMS + h] * y[r * hidden + d];
                for g in 0..STREAMS {
                    o = comb[r * 16 + g * STREAMS + h].mul_add(x[r * row + g * hidden + d], o);
                }
                out[r * row + h * hidden + d] = o;
            }
        }
    }
    out
}

/// One mixing site's weights (`layers.N.hc_attn` or `hc_ffn`): `fn` `[24][20480]` fp32, `base` `[24]`, `scale` `[3]`.
pub struct HcSite {
    pub fn_buf: DevBuf,
    pub base: DevBuf,
    pub scale: DevBuf,
}

impl HcSite {
    pub fn new(fnw: &[f32], base: &[f32], scale: &[f32; 3]) -> Result<Self, CudaError> {
        if fnw.len() != NFN * ROW || base.len() != NFN {
            return Err(CudaError(1));
        }
        Ok(Self { fn_buf: DevBuf::from_slice(fnw)?, base: DevBuf::from_slice(base)?, scale: DevBuf::from_slice(scale)? })
    }
}

/// Per-stream scratch of the mixer: the 25 atom partials, `pre`, `post` and `comb` of up to `max_rows` rows.
pub struct HcWork {
    part: DevBuf,
    pre: DevBuf,
    post: DevBuf,
    comb: DevBuf,
    pub max_rows: usize,
}

impl HcWork {
    /// The four buffers [`Self::new`] allocates, from the same formulas: what admission charges for the hyper-connection workspace.
    pub fn bytes_for(max_rows: usize) -> u64 {
        let r = max_rows as u64;
        r * PART_FLOATS_PER_ROW as u64 * 4 + r * STREAMS as u64 * 4 * 2 + r * 16 * 4
    }

    /// The bytes this workspace holds — what [`Self::bytes_for`] predicts.
    pub fn bytes(&self) -> u64 {
        (self.part.bytes() + self.pre.bytes() + self.post.bytes() + self.comb.bytes()) as u64
    }

    pub fn new(max_rows: usize) -> Result<Self, CudaError> {
        if max_rows == 0 || max_rows > MAX_ROWS {
            return Err(CudaError(1));
        }
        Ok(Self {
            part: DevBuf::alloc(max_rows * PART_FLOATS_PER_ROW * 4)?,
            pre: DevBuf::alloc(max_rows * STREAMS * 4)?,
            post: DevBuf::alloc(max_rows * STREAMS * 4)?,
            comb: DevBuf::alloc(max_rows * 16 * 4)?,
            max_rows,
        })
    }

    /// `pre [rows, 4]` fp32 (the collapsed sublayer input after the [RMSNorm] the caller applies).
    pub fn pre(&self) -> &DevBuf {
        &self.pre
    }
    pub fn post(&self) -> &DevBuf {
        &self.post
    }
    pub fn comb(&self) -> &DevBuf {
        &self.comb
    }
}

/// The hyper-connection mixer at this pack's shape: `x` is `[rows][4][5120]` fp32 and is updated in place by [`Self::apply`].
pub struct Dsv4Hc {
    pub cfg: HcCfg,
}

impl Dsv4Hc {
    pub const V41: Dsv4Hc = Dsv4Hc { cfg: HcCfg::V41 };

    /// The 25 dots per atom of every row (into `work`). One launch, 40 blocks of 8 warps per row.
    pub fn partials(&self, x: &DevBuf, site: &HcSite, rows: usize, work: &mut HcWork, st: &Stream) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        if rows > work.max_rows
            || x.bytes() < rows * ROW * 4
            || site.fn_buf.bytes() < NFN * ROW * 4
            || work.part.bytes() < rows * PART_FLOATS_PER_ROW * 4
        {
            return Err(CudaError(1));
        }
        unsafe { CudaError::check(ffi::gb_dsv4_hc_partials(x.ptr(), site.fn_buf.ptr(), work.part.ptr(), i(rows)?, st.raw())) }
    }

    /// [`Self::partials`] fused with [`Self::collapse_f32`] in the same CTA pass over `x`.
    pub fn partials_and_collapse(
        &self,
        x: &DevBuf,
        site: &HcSite,
        pre_in: &DevBuf,
        collapsed_out: &mut DevBuf,
        rows: usize,
        work: &mut HcWork,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        if rows > work.max_rows
            || x.bytes() < rows * ROW * 4
            || site.fn_buf.bytes() < NFN * ROW * 4
            || work.part.bytes() < rows * PART_FLOATS_PER_ROW * 4
            || pre_in.bytes() < rows * STREAMS * 4
            || collapsed_out.bytes() < rows * HIDDEN * 4
        {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_dsv4_hc_partials_and_collapse(
                x.ptr(),
                site.fn_buf.ptr(),
                work.part.ptr(),
                pre_in.ptr() as *const f32,
                collapsed_out.ptr() as *mut f32,
                i(rows)?,
                st.raw(),
            ))
        }
    }

    /// [`Self::apply`] (FP16 `y_prev`) fused with the next site's [`Self::partials_and_collapse`] in one CTA pass over `x`.
    pub fn apply_partials_and_collapse(
        &self,
        x: &mut DevBuf,
        y_prev: &DevBuf,
        site: &HcSite,
        pre_in: &DevBuf,
        collapsed_out: &mut DevBuf,
        rows: usize,
        work: &mut HcWork,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        if rows > work.max_rows
            || x.bytes() < rows * ROW * 4
            || y_prev.bytes() < rows * HIDDEN * 2
            || site.fn_buf.bytes() < NFN * ROW * 4
            || work.part.bytes() < rows * PART_FLOATS_PER_ROW * 4
            || pre_in.bytes() < rows * STREAMS * 4
            || collapsed_out.bytes() < rows * HIDDEN * 4
        {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_dsv4_hc_apply_partials_and_collapse(
                x.ptr(),
                site.fn_buf.ptr(),
                work.part.ptr(),
                pre_in.ptr() as *const f32,
                collapsed_out.ptr() as *mut f32,
                y_prev.ptr(),
                work.post.ptr(),
                work.comb.ptr(),
                i(rows)?,
                st.raw(),
            ))
        }
    }

    /// Reduce the partials, then `pre` / `post` / `comb` (with the 20-iteration Sinkhorn), then collapse the streams.
    pub fn finalize(&self, x: &DevBuf, site: &HcSite, rows: usize, work: &mut HcWork, st: &Stream) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        if rows > work.max_rows || x.bytes() < rows * ROW * 4 || work.part.bytes() < rows * PART_FLOATS_PER_ROW * 4 {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_dsv4_hc_finalize(
                x.ptr(),
                work.part.ptr(),
                site.base.ptr(),
                site.scale.ptr(),
                work.pre.ptr(),
                work.post.ptr(),
                work.comb.ptr(),
                i(rows)?,
                self.cfg.rms_eps,
                self.cfg.hc_eps,
                self.cfg.sinkhorn_iters,
                st.raw(),
            ))
        }
    }

    /// `streams` `[rows, 4, dim]` f32 + `pre` `[rows, 4]` f32 -> `out` `[rows, dim]` f32, on `st`.
    pub fn collapse_f32(
        &self,
        streams: &DevBuf,
        pre: &DevBuf,
        out: &mut DevBuf,
        rows: usize,
        dim: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        if streams.bytes() < rows * STREAMS * dim * 4 || pre.bytes() < rows * STREAMS * 4 || out.bytes() < rows * dim * 4 {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_dsv4_hc_collapse_f32(
                streams.ptr() as *const f32,
                pre.ptr() as *const f32,
                out.ptr() as *mut f32,
                i32_of(rows)?,
                i32_of(dim)?,
                st.raw(),
            ))
        }
    }

    /// `streams` `[rows, 4, dim]` f32 + `pre` `[rows, 4]` f32 -> `out` `[rows, dim]` f16, on `st`.
    pub fn collapse(
        &self,
        streams: &DevBuf,
        pre: &DevBuf,
        out: &mut DevBuf,
        rows: usize,
        dim: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        if streams.bytes() < rows * STREAMS * dim * 4 || pre.bytes() < rows * STREAMS * 4 || out.bytes() < rows * dim * 2 {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_dsv4_mtp_hc_pre(
                streams.ptr() as *const f32,
                pre.ptr() as *const f32,
                out.ptr(),
                i32_of(rows)?,
                i32_of(dim)?,
                st.raw(),
            ))
        }
    }

    /// `x[h][d] <- post[h] y[d] + sum_g comb[g][h] x[g][d]`, in place.
    pub fn apply(
        &self,
        x: &mut DevBuf,
        y: &DevBuf,
        y_dtype: crate::exl3_linear::Dtype,
        rows: usize,
        work: &HcWork,
        st: &Stream,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        if rows > work.max_rows || x.bytes() < rows * ROW * 4 || y.bytes() < rows * HIDDEN * y_dtype.size() {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_dsv4_hc_apply(
                x.ptr(),
                y.ptr(),
                y_dtype as i32,
                work.post.ptr(),
                work.comb.ptr(),
                i(rows)?,
                st.raw(),
            ))
        }
    }
}

// ================================================================================================================================
// 4. fused SwiGLU with the 10.0 clamp
// ================================================================================================================================

/// `Expert.forward` without the two GEMMs: `silu(min(gate, limit)) * clamp(up, -limit, limit)`, times `w[token]` when the
/// expert is weighted. The clamps are from training (they keep fp8/fp4 activations in range): the up branch is clamped on
/// both sides, the gate branch only from above.
#[inline]
pub fn swiglu_scalar(gate: f32, up: f32, w: f32, limit: f32) -> f32 {
    let (g, u) = if limit > 0.0 { (gate.min(limit), up.clamp(-limit, limit)) } else { (gate, up) };
    (g / (1.0 + (-g).exp())) * u * w
}

/// [`swiglu_scalar`] over `[rows, inter]`, with one weight per row (`None` = unweighted, as for the shared expert).
pub fn swiglu_host(gate: &[f32], up: &[f32], w: Option<&[f32]>, rows: usize, inter: usize, limit: f32) -> Vec<f32> {
    assert!(gate.len() == rows * inter && up.len() == rows * inter);
    if let Some(w) = w {
        assert!(w.len() == rows);
    }
    let mut out = vec![0f32; rows * inter];
    for r in 0..rows {
        let wr = w.map_or(1.0, |w| w[r]);
        for i in 0..inter {
            out[r * inter + i] = swiglu_scalar(gate[r * inter + i], up[r * inter + i], wr, limit);
        }
    }
    out
}

/// The fused device activation: `out = silu(min(gate, limit)) * clamp(up, -limit, limit) * w[token]`. Asynchronous on `st`.
pub fn swiglu(
    gate: &DevBuf,
    up: &DevBuf,
    w: Option<&DevBuf>,
    out: &mut DevBuf,
    rows: usize,
    inter: usize,
    limit: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    let n = rows * inter;
    if n == 0 {
        return Ok(());
    }
    if gate.bytes() < n * 4 || up.bytes() < n * 4 || out.bytes() < n * 4 {
        return Err(CudaError(1));
    }
    if let Some(w) = w {
        if w.bytes() < rows * 4 {
            return Err(CudaError(1));
        }
    }
    let wp: *const c_void = w.map_or(core::ptr::null(), |w| w.ptr());
    unsafe {
        CudaError::check(ffi::gb_dsv4_swiglu(gate.ptr(), up.ptr(), wp, out.ptr(), i32_of(n)?, i32_of(inter)?, limit, st.raw()))
    }
}

/// [`swiglu`] on the MoE block's native dtype: `gate`, `up` and `out` are fp16 `[rows, inter]`, `w` (when given) fp16
/// `[rows]`. The activation is computed in fp32 and rounded once, on the way out. Asynchronous on `st`.
pub fn swiglu_f16(
    gate: &DevBuf,
    up: &DevBuf,
    w: Option<&DevBuf>,
    out: &mut DevBuf,
    rows: usize,
    inter: usize,
    limit: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    let n = rows * inter;
    if n == 0 {
        return Ok(());
    }
    if gate.bytes() < n * 2 || up.bytes() < n * 2 || out.bytes() < n * 2 {
        return Err(CudaError(1));
    }
    if let Some(w) = w {
        if w.bytes() < rows * 2 {
            return Err(CudaError(1));
        }
    }
    let wp: *const c_void = w.map_or(core::ptr::null(), |w| w.ptr());
    unsafe {
        CudaError::check(ffi::gb_dsv4_swiglu_f16(
            gate.ptr(),
            up.ptr(),
            wp,
            out.ptr(),
            i32_of(n)?,
            i32_of(inter)?,
            limit,
            st.raw(),
        ))
    }
}

/// Apply the Engram signed-sqrt gate and add `gate * value` into `streams` (`[rows, hc, dim]` f32) in place on `st`.
pub fn engram_gate_add(
    streams: &mut DevBuf,
    kv: &DevBuf,
    q_weight: &DevBuf,
    k_weight: &DevBuf,
    rows: usize,
    hc: usize,
    dim: usize,
    eps: f32,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    if streams.bytes() < rows * hc * dim * 4
        || kv.bytes() < rows * (hc + 1) * dim * 2
        || q_weight.bytes() < hc * dim * 2
        || k_weight.bytes() < hc * dim * 2
    {
        return Err(CudaError(1));
    }
    unsafe {
        CudaError::check(ffi::gb_dsv4_engram_gate_add(
            streams.ptr() as *mut f32,
            kv.ptr(),
            q_weight.ptr(),
            k_weight.ptr(),
            i32_of(rows)?,
            i32_of(hc)?,
            i32_of(dim)?,
            eps,
            st.raw(),
        ))
    }
}

// ================================================================================================================================
// 5. the routed experts: group on the device, one segment a launch, scatter-add on the device
// ================================================================================================================================

use crate::exl3_linear::{linear_workspace_bytes, Dtype, Exl3Weights, MAX_DECODE_ROWS};
use crate::rt::Block;
use std::sync::Arc;

/// The alignment every segment of [`MoeScratch`]'s arena starts at. 512 B: a multiple of the 16 B every kernel here
/// requires of a pointer, with room to spare so a segment's start is never a cache-line boundary of its neighbour.
const SEG: usize = 512;

/// One active expert's three groups (`w1`, `w2`, `w3`, in the model's order), however they are stored, plus — for an
/// expert that streams — the lease that keeps its cache slot from being evicted while the three GEMMs read it.
pub enum MoeGroups<'a> {
    /// In the slab: the loader bound these in place, and nothing has to be pinned.
    Resident(&'a [Exl3Weights]),
    /// Past the residency budget: a cache slot, held by the lease for as long as this value lives.
    Streamed { groups: Vec<Exl3Weights>, lease: crate::expert_cache::Lease<'a> },
}

impl MoeGroups<'_> {
    /// The three groups, in `w1`, `w2`, `w3` order.
    pub fn get(&self) -> &[Exl3Weights] {
        match self {
            MoeGroups::Resident(g) => g,
            MoeGroups::Streamed { groups, .. } => groups,
        }
    }
}

/// What one layer's routed block did, for the boot log and the tests: how many experts were active and how many launches
/// that cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoeStats {
    /// Experts with at least one row (`> 0`).
    pub active: usize,
    /// Launches issued: `group`, `gather`, `scatter` plus four an active expert (`w1`, `w3`, `swiglu`, `w2`).
    pub launches: usize,
    /// Packed slots (`rows * topk`).
    pub slots: usize,
}

/// Every buffer one layer's routed block needs, at one chunk shape, in **one** device arena: nothing here allocates
/// during a request, and every segment is a [`DevBuf::view`] into the arena, so a per-expert segment is a pointer
/// offset rather than an allocation.
///
/// | segment | shape | dtype |
/// |---|---|---|
/// | `counts` | `[experts]` | i32 |
/// | `offsets` | `[experts + 1]` | i32 |
/// | `perm`, `pinv` | `[slots]` | i32 |
/// | `wpack` | `[slots]` | f16 |
/// | `x` | `[slots, hidden]` | f16 |
/// | `g`, `u`, `act` | `[slots, inter]` | f16 |
/// | `y` | `[slots, hidden]` | f16 |
/// | `ws` | the EXL3 decode workspace for the widest projection at `rows` rows | bytes |
///
/// `slots = rows * topk`. `x` is the gathered input, `g` / `u` the two projections, `act` the swiglu output, `y` the
/// down projection; all four are fp16 because that is the EXL3 gemv's operand type, and the accumulation that matters
/// (the weighted sum over `k`) is fp32 in [`moe_experts`]'s scatter.
pub struct MoeScratch {
    rows: usize,
    topk: usize,
    hidden: usize,
    inter: usize,
    experts: usize,
    /// The trunk layer this scratch belongs to, set by the model before each call. Only the debug dump
    /// (`REXL3_DSV41_MOE_DUMP=<layer>`) reads it, so it takes no part in the arithmetic.
    pub layer: usize,
    arena: Arc<Block>,
    counts: DevBuf,
    offsets: DevBuf,
    perm: DevBuf,
    pinv: DevBuf,
    wpack: DevBuf,
    x: DevBuf,
    g: DevBuf,
    u: DevBuf,
    act: DevBuf,
    y: DevBuf,
    ws: DevBuf,
}

/// Segment offsets inside [`MoeScratch`]'s arena, and its total size.
struct Layout {
    counts: usize,
    offsets: usize,
    perm: usize,
    pinv: usize,
    wpack: usize,
    x: usize,
    g: usize,
    u: usize,
    act: usize,
    y: usize,
    ws: usize,
    total: usize,
}

/// The gemv workspace [`moe_experts`] needs: that of the larger of the two projections at the decode path's row cap
/// ([`MAX_DECODE_ROWS`]), because a segment longer than the cap is cut into windows of it — the decode path is row
/// invariant, so the bits do not depend on the window.
fn gemv_workspace_bytes(slots: usize, hidden: usize, inter: usize) -> usize {
    let m = slots.clamp(1, MAX_DECODE_ROWS);
    linear_workspace_bytes(m, hidden, inter, 6).max(linear_workspace_bytes(m, inter, hidden, 6)).max(1)
}

impl Layout {
    fn of(rows: usize, topk: usize, hidden: usize, inter: usize, experts: usize) -> Self {
        let slots = rows * topk;
        let mut o = 0usize;
        let mut at = |n: usize| {
            let here = o;
            o += n.div_ceil(SEG) * SEG;
            here
        };
        let counts = at(experts * 4);
        let offsets = at((experts + 1) * 4);
        let perm = at(slots * 4);
        let pinv = at(slots * 4);
        let wpack = at(slots * 2);
        let act = at(slots * inter * 2);
        let y = at(slots * hidden * 2);
        let x = at(slots * hidden * 2);
        let g = at(slots * inter * 2);
        let u = at(slots * inter * 2);
        let align256 = |v: usize| (v + 255) & !255;
        let z_slots = slots.min(16 * topk);
        let fast_need = if hidden == 5120 {
            2 * align256(slots * 4) + 256 + align256(2 * slots * hidden * 2) + align256(4 * z_slots * inter * 4)
        } else {
            0
        };
        let xgu_bytes = (u + (slots * inter * 2).div_ceil(SEG) * SEG) - x;
        let ws_bytes = gemv_workspace_bytes(slots, hidden, inter).max(fast_need.saturating_sub(xgu_bytes));
        let ws = at(ws_bytes);
        Self { counts, offsets, perm, pinv, wpack, x, g, u, act, y, ws, total: o }
    }
}

impl MoeScratch {
    /// The bytes [`Self::new`] allocates, from the same layout: what admission charges for the routed block.
    pub fn bytes_for(rows: usize, topk: usize, hidden: usize, inter: usize, experts: usize) -> u64 {
        Layout::of(rows, topk, hidden, inter, experts).total as u64
    }

    /// The bytes this scratch holds — what [`Self::bytes_for`] predicts. `new` checks they are equal.
    pub fn bytes(&self) -> u64 {
        self.arena.bytes() as u64
    }

    /// Allocate the arena once, for `rows` rows of one chunk.
    pub fn new(rows: usize, topk: usize, hidden: usize, inter: usize, experts: usize) -> Result<Self, CudaError> {
        if rows == 0 || topk == 0 || hidden == 0 || inter == 0 || experts == 0 || experts > 1024 {
            return Err(CudaError(1));
        }
        let l = Layout::of(rows, topk, hidden, inter, experts);
        let arena = Arc::new(Block::alloc(l.total)?);
        let v = |off: usize, bytes: usize| DevBuf::view(&arena, off, bytes);
        let slots = rows * topk;
        let offsets = Block::alloc_mapped_host((experts + 1) * 4)
            .map(|b| DevBuf::view(&Arc::new(b), 0, (experts + 1) * 4))
            .unwrap_or_else(|_| v(l.offsets, (experts + 1) * 4));
        let s = Self {
            rows,
            topk,
            hidden,
            inter,
            experts,
            layer: usize::MAX,
            counts: v(l.counts, experts * 4),
            offsets,
            perm: v(l.perm, slots * 4),
            pinv: v(l.pinv, slots * 4),
            wpack: v(l.wpack, slots * 2),
            x: v(l.x, slots * hidden * 2),
            g: v(l.g, slots * inter * 2),
            u: v(l.u, slots * inter * 2),
            act: v(l.act, slots * inter * 2),
            y: v(l.y, slots * hidden * 2),
            ws: v(l.ws, gemv_workspace_bytes(slots, hidden, inter)),
            arena: arena.clone(),
        };
        debug_assert_eq!(
            s.bytes(),
            Self::bytes_for(rows, topk, hidden, inter, experts),
            "the MoE scratch is not the bytes admission prices for it"
        );

        Ok(s)
    }

    /// Rows of one call, `topk`, and the two widths.
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        (self.rows, self.topk, self.hidden, self.inter)
    }

    pub fn arena_block(&self) -> &Arc<Block> {
        &self.arena
    }

    /// The last grouping's `[experts + 1]` offsets: the one host-visible product of [`Self::group`]. One 1,540-byte copy
    /// per layer at this pack's 384 experts.
    pub fn offsets_host(&self) -> Result<Vec<i32>, CudaError> {
        self.offsets.to_vec()
    }

    /// The scratch's `perm` as packed by the last grouping (or [`Self::pack_stable`]): the tests' read of the
    /// placement the kernels will scatter through.
    pub fn perm_host(&self) -> Result<Vec<i32>, CudaError> {
        self.perm.to_vec()
    }

    /// [`Self::perm_host`]'s inverse.
    pub fn pinv_host(&self) -> Result<Vec<i32>, CudaError> {
        self.pinv.to_vec()
    }

    /// [`Self::perm_host`]'s packed weights.
    pub fn wpack_host(&self) -> Result<Vec<u16>, CudaError> {
        self.wpack.to_vec()
    }

    /// Repack this scratch's `perm`/`pinv`/`wpack` on the host in the STABLE order ([`pack_stable_host`]'s, the
    /// contract `group_k`'s header documents but its atomic placement cannot guarantee): the device grouping stands
    /// (its `counts`/`offsets` are order-free), the permutation is recomputed from the picks alone. Two small
    /// downloads and three uploads, on a path whose step already syncs for [`Self::offsets_host`]; it runs only on a
    /// rank of a group — the single-rank result is packing-invariant, the exchanged one is not.
    pub fn pack_stable(
        &mut self,
        picks: &DevBuf,
        wts: &DevBuf,
        offsets: &[i32],
        rows: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let slots = rows * self.topk;
        if rows == 0 || picks.bytes() < slots * 4 || wts.bytes() < slots * 2 || offsets.len() != self.experts + 1 {
            return Err(CudaError(1));
        }
        let mut picks_h = vec![0i32; slots];
        let mut wts_h = vec![0u16; slots];
        picks.download_range(0, &mut picks_h)?;
        wts.download_range(0, &mut wts_h)?;
        let (perm, pinv, wpack) = pack_stable_host(offsets, &picks_h, &wts_h);
        unsafe {
            st.memcpy_h2d_async(self.perm.ptr(), perm.as_ptr() as *const u8, perm.len() * 4)?;
            st.memcpy_h2d_async(self.pinv.ptr(), pinv.as_ptr() as *const u8, pinv.len() * 4)?;
            st.memcpy_h2d_async(self.wpack.ptr(), wpack.as_ptr() as *const u8, wpack.len() * 2)?;
        }
        st.sync()?;
        Ok(())
    }

    /// Borrow the EXL3 decode workspace, growing it if `need` exceeds the current size.
    pub fn ensure_workspace(&mut self, need: usize) -> Result<&mut DevBuf, CudaError> {
        if self.ws.bytes() < need {
            self.ws = DevBuf::zeroed(need)?;
        }
        Ok(&mut self.ws)
    }

    /// Build the grouping: `counts`, `offsets`, the permutation `perm` (packed position -> slot), its inverse `pinv`
    /// (slot -> packed position) and the packed weights `wpack`. One launch, one block; no host round trip.
    pub fn group(&mut self, picks: &DevBuf, wts: &DevBuf, rows: usize) -> Result<(), CudaError> {
        self.group_raw(picks, wts, rows, core::ptr::null_mut())
    }

    /// [`Self::group`] on `st`.
    pub fn group_on(&mut self, picks: &DevBuf, wts: &DevBuf, rows: usize, st: &Stream) -> Result<(), CudaError> {
        self.group_raw(picks, wts, rows, st.raw())
    }

    fn group_raw(&mut self, picks: &DevBuf, wts: &DevBuf, rows: usize, stream: *mut core::ffi::c_void) -> Result<(), CudaError> {
        if rows == 0 || rows > self.rows {
            return Err(CudaError(1));
        }
        let slots = rows * self.topk;
        if picks.bytes() < slots * 4 || wts.bytes() < slots * 2 {
            return Err(CudaError(1));
        }
        unsafe {
            CudaError::check(ffi::gb_dsv4_moe_group(
                picks.ptr() as *const i32,
                wts.ptr(),
                self.counts.ptr() as *mut i32,
                self.offsets.ptr() as *mut i32,
                self.perm.ptr() as *mut i32,
                self.pinv.ptr() as *mut i32,
                self.wpack.ptr(),
                i32_of(slots)?,
                i32_of(self.experts)?,
                stream,
            ))
        }
    }
}

/// Run one layer's routed experts: group the rows by expert on the device, run each active expert's `w1` / `w3` ->
/// swiglu -> `w2` on its own segment with its own bit width and codebook, then scatter-add the weighted outputs back per
/// row on the device.
///
/// `groups_of(e)` supplies expert `e`'s three groups (`w1`, `w2`, `w3`) — resident slab views, or a lease on its
/// ExpertCache slot; the returned [`MoeGroups`] must be kept alive for as long as the GEMMs read it, which is why the
/// lease rides in it. `picks` / `wts` are the router's `[rows, topk]` i32 / f16 selection, on the device.
///
/// `shared_f16` is the shared expert's `[rows, hidden]` f16 output, and is **replaced in place** by the block's f16
/// result (`shared[r] + sum_k w y`): the scatter reads row `r` and writes row `r`, so no second f16 buffer is needed.
/// `moe_out_f32` is the same sum at fp32, which is what the mHC post's coefficient math wants.
///
/// Launch count: `4 * active + 3` (group, gather, scatter, and `w1`, `w3`, swiglu, `w2` a active expert). The one host
/// step is the `offsets` copy that says which experts are active and where their segments start — see [`MoeStats`].
/// `REXL3_DSV41_MOE_DUMP=<layer>`: dump the intermediates of that trunk layer's routed block (and of the shared expert it
/// enters with) once per call. A debug probe only: the buffers are squashed to `(absmax of the finite entries, count of
/// non-finite entries)`, so the numbers are one line an expert and never a tensor.
fn moe_dump_layer(layer: usize) -> bool {
    static D: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    D.get_or_init(|| {
        std::env::var("REXL3_DSV41_MOE_DUMP")
            .map(|s| s.split(',').filter_map(|p| p.trim().parse().ok()).collect())
            .unwrap_or_default()
    })
    .contains(&layer)
}

/// `(absmax over the finite entries, count of non-finite entries)` of an fp16 buffer, read back to the host.
fn f16_extent(v: &DevBuf) -> (f32, usize) {
    match v.to_vec::<u16>() {
        Ok(b) => {
            let (mut mx, mut bad) = (0f32, 0usize);
            for &h in &b {
                let x = gb_exl3::f16::f16_to_f64(h) as f32;
                if x.is_finite() {
                    if x.abs() > mx {
                        mx = x.abs();
                    }
                } else {
                    bad += 1;
                }
            }
            (mx, bad)
        }
        Err(_) => (f32::NAN, usize::MAX),
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Dsv4ExpertDesc {
    pub w1_trellis: u64,
    pub w1_suh: u64,
    pub w1_svh: u64,
    pub w1_bits: i32,
    pub w3_bits: i32,
    pub w3_trellis: u64,
    pub w3_suh: u64,
    pub w3_svh: u64,
    pub w2_bits: i32,
    pub pad: i32,
    pub w2_trellis: u64,
    pub w2_suh: u64,
    pub w2_svh: u64,
}

/// Build a device descriptor table `[experts]` for `moe_experts_resident` when all experts in a layer are resident,
/// un-biased, `Codebook::Mul1`, and `bits in 1..=4`, and `hidden` / `inter` satisfy the fast kernel's alignment.
pub fn build_expert_descs(experts: &[&[Exl3Weights]], hidden: usize, inter: usize) -> Result<Option<DevBuf>, CudaError> {
    let opts: Vec<Option<&[Exl3Weights]>> = experts.iter().map(|&g| Some(g)).collect();
    build_expert_descs_sparse(&opts, hidden, inter)
}

/// Build a device descriptor table `[experts]` where non-owned experts (on a TP rank) are `None` (zeroed descriptors)
/// and every owned expert is resident (`Some`), un-biased, `Codebook::Mul1`, and `bits in 1..=4`.
pub fn build_expert_descs_sparse(
    experts: &[Option<&[Exl3Weights]>],
    hidden: usize,
    inter: usize,
) -> Result<Option<DevBuf>, CudaError> {
    if hidden == 0
        || inter == 0
        || (hidden % 128) != 0
        || (inter % 128) != 0
        || ((hidden / 16) % 8) != 0
        || ((inter / 16) % 4) != 0
    {
        return Ok(None);
    }
    let mut descs = Vec::with_capacity(experts.len());
    for g in experts {
        let Some(g) = g else {
            descs.push(Dsv4ExpertDesc::default());
            continue;
        };
        if g.len() != 3 {
            return Ok(None);
        }
        let (w1, w2, w3) = (&g[0], &g[1], &g[2]);
        for w in [w1, w2, w3] {
            if !(1..=4).contains(&w.bits) || w.codebook != gb_exl3::Codebook::Mul1 || w.bias.is_some() {
                return Ok(None);
            }
        }
        descs.push(Dsv4ExpertDesc {
            w1_trellis: w1.trellis.ptr() as u64,
            w1_suh: w1.suh.ptr() as u64,
            w1_svh: w1.svh.ptr() as u64,
            w1_bits: w1.bits as i32,
            w3_bits: w3.bits as i32,
            w3_trellis: w3.trellis.ptr() as u64,
            w3_suh: w3.suh.ptr() as u64,
            w3_svh: w3.svh.ptr() as u64,
            w2_bits: w2.bits as i32,
            pad: 0,
            w2_trellis: w2.trellis.ptr() as u64,
            w2_suh: w2.suh.ptr() as u64,
            w2_svh: w2.svh.ptr() as u64,
        });
    }
    Ok(Some(DevBuf::from_slice(&descs)?))
}

/// Interleave two compact column halves `left` and `right` (`[rows, half_f16]` f16 each) into `dst` (`[rows, 2 * half_f16]` f16)
/// in a single kernel launch across all `rows`.
pub fn interleave_o_halves(
    left: &DevBuf,
    right: &DevBuf,
    dst: &mut DevBuf,
    rows: usize,
    half_f16: usize,
    st: &Stream,
) -> Result<(), CudaError> {
    if rows == 0 {
        return Ok(());
    }
    if left.bytes() < rows * half_f16 * 2 || right.bytes() < rows * half_f16 * 2 || dst.bytes() < rows * half_f16 * 4 {
        return Err(CudaError(1));
    }
    unsafe {
        CudaError::check(ffi::gb_dsv4_interleave_o_halves(
            left.ptr(),
            right.ptr(),
            dst.ptr(),
            i32_of(rows)?,
            i32_of(half_f16)?,
            st.raw(),
        ))
    }
}

/// Fast grouped MoE for one rank of a tensor-parallel group whose owned experts `own_range` (`[e_lo, e_hi)`) are all resident:
/// zero host stream syncs, 5 compute launches + 1 fixed-size `slots * h * 2` byte swap into `y_peer` + 1 scatter launch.
#[allow(clippy::too_many_arguments)]
pub fn moe_experts_resident_tp<F>(
    s: &mut MoeScratch,
    descs: &DevBuf,
    own_range: std::ops::Range<usize>,
    mut swap_y: F,
    y_peer: &mut DevBuf,
    rows: usize,
    gemm_only: bool,
    hidden_in: &DevBuf,
    shared_f16: &mut DevBuf,
    picks: &DevBuf,
    wts: &DevBuf,
    moe_out_f32: &mut DevBuf,
    limit: f32,
    st: &Stream,
) -> Result<MoeStats, CudaError>
where
    F: FnMut(&DevBuf, &mut DevBuf, usize, &Stream) -> Result<(), CudaError>,
{
    let (topk, h, inter, experts) = (s.topk, s.hidden, s.inter, s.experts);
    let slots = rows * topk;
    if rows == 0 || rows > s.rows || y_peer.bytes() < slots * h * 2 {
        return Err(CudaError(1));
    }
    let align256 = |x: usize| (x + 255) & !255;
    let z_slots = if rows <= 16 && !gemm_only { slots } else { 0 };
    let need = 2 * align256(slots * 4) + 256 + align256(2 * slots * h * 2) + align256(4 * z_slots * inter * 4);
    let x_off = Layout::of(s.rows, s.topk, s.hidden, s.inter, s.experts).x;
    let tail_avail = s.arena.bytes().saturating_sub(x_off);
    let (ws_ptr, ws_len) = if tail_avail >= need {
        (s.x.ptr(), tail_avail)
    } else {
        s.ensure_workspace(need)?;
        (s.ws.ptr(), s.ws.bytes())
    };
    unsafe {
        CudaError::check(ffi::gb_dsv4_moe_fast_experts_tp_pre(
            hidden_in.ptr(),
            picks.ptr() as *const i32,
            wts.ptr(),
            descs.ptr(),
            s.counts.ptr() as *mut i32,
            s.offsets.ptr() as *mut i32,
            s.perm.ptr() as *mut i32,
            s.pinv.ptr() as *mut i32,
            s.wpack.ptr(),
            s.act.ptr(),
            s.y.ptr(),
            ws_ptr,
            ws_len,
            i32_of(rows)?,
            i32_of(topk)?,
            i32_of(h)?,
            i32_of(inter)?,
            i32_of(experts)?,
            i32_of(own_range.start)?,
            i32_of(own_range.end)?,
            i32_of(gemm_only as usize)?,
            limit,
            st.raw(),
        ))?;
    }
    swap_y(&s.y, y_peer, slots * h * 2, st)?;
    unsafe {
        CudaError::check(ffi::gb_dsv4_moe_fast_experts_tp_post(
            shared_f16.ptr(),
            s.y.ptr(),
            y_peer.ptr(),
            s.offsets.ptr() as *const i32,
            s.pinv.ptr() as *const i32,
            s.wpack.ptr(),
            moe_out_f32.ptr(),
            i32_of(rows)?,
            i32_of(topk)?,
            i32_of(h)?,
            i32_of(own_range.start)?,
            i32_of(own_range.end)?,
            st.raw(),
        ))?;
    }
    Ok(MoeStats { active: slots.min(own_range.len()), launches: 6, slots })
}

#[allow(clippy::too_many_arguments)]
pub fn moe_experts_resident(
    s: &mut MoeScratch,
    descs: &DevBuf,
    rows: usize,
    hidden_in: &DevBuf,
    shared_f16: &mut DevBuf,
    picks: &DevBuf,
    wts: &DevBuf,
    moe_out_f32: &mut DevBuf,
    limit: f32,
    st: &Stream,
) -> Result<MoeStats, CudaError> {
    moe_experts_resident_with_join(s, descs, None, rows, false, hidden_in, shared_f16, picks, wts, moe_out_f32, limit, st)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn moe_experts_resident_with_join(
    s: &mut MoeScratch,
    descs: &DevBuf,
    join_shared: Option<&mut dyn FnMut(&Stream) -> Result<(), CudaError>>,
    rows: usize,
    gemm_only: bool,
    hidden_in: &DevBuf,
    shared_f16: &mut DevBuf,
    picks: &DevBuf,
    wts: &DevBuf,
    moe_out_f32: &mut DevBuf,
    limit: f32,
    st: &Stream,
) -> Result<MoeStats, CudaError> {
    let (topk, h, inter, experts) = (s.topk, s.hidden, s.inter, s.experts);
    let slots = rows * topk;
    if rows == 0 || rows > s.rows {
        return Err(CudaError(1));
    }
    let align256 = |x: usize| (x + 255) & !255;
    let z_slots = if rows <= 16 && !gemm_only { slots } else { 0 };
    let need = 2 * align256(slots * 4) + 256 + align256(2 * slots * h * 2) + align256(4 * z_slots * inter * 4);
    let x_off = Layout::of(s.rows, s.topk, s.hidden, s.inter, s.experts).x;
    let tail_avail = s.arena.bytes().saturating_sub(x_off);
    let (ws_ptr, ws_len) = if tail_avail >= need {
        (s.x.ptr(), tail_avail)
    } else {
        s.ensure_workspace(need)?;
        (s.ws.ptr(), s.ws.bytes())
    };
    if let Some(join) = join_shared {
        unsafe {
            CudaError::check(ffi::gb_dsv4_moe_fast_experts_tp_pre(
                hidden_in.ptr(),
                picks.ptr() as *const i32,
                wts.ptr(),
                descs.ptr(),
                s.counts.ptr() as *mut i32,
                s.offsets.ptr() as *mut i32,
                s.perm.ptr() as *mut i32,
                s.pinv.ptr() as *mut i32,
                s.wpack.ptr(),
                s.act.ptr(),
                s.y.ptr(),
                ws_ptr,
                ws_len,
                i32_of(rows)?,
                i32_of(topk)?,
                i32_of(h)?,
                i32_of(inter)?,
                i32_of(experts)?,
                0,
                i32_of(experts)?,
                i32_of(gemm_only as usize)?,
                limit,
                st.raw(),
            ))?;
        }
        join(st)?;
        unsafe {
            CudaError::check(ffi::gb_dsv4_moe_fast_experts_tp_post(
                shared_f16.ptr(),
                s.y.ptr(),
                s.y.ptr(),
                s.offsets.ptr() as *const i32,
                s.pinv.ptr() as *const i32,
                s.wpack.ptr(),
                moe_out_f32.ptr(),
                i32_of(rows)?,
                i32_of(topk)?,
                i32_of(h)?,
                0,
                i32_of(experts)?,
                st.raw(),
            ))?;
        }
        return Ok(MoeStats { active: slots.min(experts), launches: 6, slots });
    }
    unsafe {
        CudaError::check(ffi::gb_dsv4_moe_fast_experts(
            hidden_in.ptr(),
            shared_f16.ptr(),
            picks.ptr() as *const i32,
            wts.ptr(),
            moe_out_f32.ptr(),
            descs.ptr(),
            s.counts.ptr() as *mut i32,
            s.offsets.ptr() as *mut i32,
            s.perm.ptr() as *mut i32,
            s.pinv.ptr() as *mut i32,
            s.wpack.ptr(),
            s.act.ptr(),
            s.y.ptr(),
            ws_ptr,
            ws_len,
            i32_of(rows)?,
            i32_of(topk)?,
            i32_of(h)?,
            i32_of(inter)?,
            i32_of(experts)?,
            i32_of(gemm_only as usize)?,
            limit,
            st.raw(),
        ))?;
    }
    Ok(MoeStats { active: slots.min(experts), launches: 6, slots })
}

pub fn moe_experts<'a, F>(
    s: &mut MoeScratch,
    groups_of: F,
    rows: usize,
    hidden_in: &DevBuf,
    shared_f16: &mut DevBuf,
    picks: &DevBuf,
    wts: &DevBuf,
    moe_out_f32: &mut DevBuf,
    limit: f32,
    st: &Stream,
) -> Result<MoeStats, CudaError>
where
    F: FnMut(usize) -> Result<MoeGroups<'a>, CudaError>,
{
    moe_experts_with(s, groups_of, None, None, None, rows, hidden_in, shared_f16, picks, wts, moe_out_f32, limit, st)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn moe_experts_with_join<'a, F>(
    s: &mut MoeScratch,
    groups_of: F,
    before_scatter: Option<&mut dyn FnMut(&Stream) -> Result<(), CudaError>>,
    rows: usize,
    hidden_in: &DevBuf,
    shared_f16: &mut DevBuf,
    picks: &DevBuf,
    wts: &DevBuf,
    moe_out_f32: &mut DevBuf,
    limit: f32,
    st: &Stream,
) -> Result<MoeStats, CudaError>
where
    F: FnMut(usize) -> Result<MoeGroups<'a>, CudaError>,
{
    moe_experts_with(s, groups_of, None, None, before_scatter, rows, hidden_in, shared_f16, picks, wts, moe_out_f32, limit, st)
}

/// The exchange between the routed experts' down projections and the slot combine of [`moe_experts_tp`]: `(y, the rank's own
/// slot range, hidden)`. The hook makes `y` the WHOLE buffer — this rank's segments it computed, the peer's it receives — by
/// whatever transport the group's collective offers; no arithmetic is wanted (a byte exchange of the two disjoint slot ranges
/// is bit-exact), and on return every element of `y` must be the single-node engine's, because the unchanged scatter reads all
/// of it on every rank.
pub type YExchange<'a> = dyn FnMut(&mut DevBuf, std::ops::Range<usize>, usize, usize, &Stream) -> Result<(), CudaError> + 'a;
// (y, own slot range, SLOTS OF THIS PACKING, hidden, stream)

/// [`moe_experts`] for one rank of an expert-parallel group, the mirror of the nemotron block's `forward_in_tp`: exactly
/// [`moe_experts`]'s launches, with two differences the one rule allows — the loop computes only the experts `own` claims
/// (their segments of `y`; the pack's router runs replicated, so the grouping, the offsets and the slot layout are the
/// single-node ones, identical on every rank), and `exchange` runs once between the down projections and the scatter to make
/// `y` whole. With a no-op `exchange` and an `own` that claims every expert this is [`moe_experts`], bit for bit.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn moe_experts_tp<'a, F>(
    s: &mut MoeScratch,
    groups_of: F,
    own: &dyn Fn(usize) -> bool,
    exchange: &mut YExchange<'_>,
    rows: usize,
    hidden_in: &DevBuf,
    shared_f16: &mut DevBuf,
    picks: &DevBuf,
    wts: &DevBuf,
    moe_out_f32: &mut DevBuf,
    limit: f32,
    st: &Stream,
) -> Result<MoeStats, CudaError>
where
    F: FnMut(usize) -> Result<MoeGroups<'a>, CudaError>,
{
    moe_experts_with(
        s,
        groups_of,
        Some(own),
        Some(exchange),
        None,
        rows,
        hidden_in,
        shared_f16,
        picks,
        wts,
        moe_out_f32,
        limit,
        st,
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn moe_experts_tp_with_join<'a, F>(
    s: &mut MoeScratch,
    groups_of: F,
    own: &dyn Fn(usize) -> bool,
    exchange: &mut YExchange<'_>,
    before_scatter: Option<&mut dyn FnMut(&Stream) -> Result<(), CudaError>>,
    rows: usize,
    hidden_in: &DevBuf,
    shared_f16: &mut DevBuf,
    picks: &DevBuf,
    wts: &DevBuf,
    moe_out_f32: &mut DevBuf,
    limit: f32,
    st: &Stream,
) -> Result<MoeStats, CudaError>
where
    F: FnMut(usize) -> Result<MoeGroups<'a>, CudaError>,
{
    moe_experts_with(
        s,
        groups_of,
        Some(own),
        Some(exchange),
        before_scatter,
        rows,
        hidden_in,
        shared_f16,
        picks,
        wts,
        moe_out_f32,
        limit,
        st,
    )
}

/// The slot range of `y` the experts `own` claims hold: `[offsets[first owned], offsets[last owned + 1])`. Ownership is
/// contiguous by construction (a rank holds `[E r / R, E (r + 1) / R)` of the layer's experts), so the union of its segments
/// is one range; a non-contiguous claim is refused by name here rather than exchanging the wrong slots silently.
/// The STABLE packing of one routed block: `perm`/`pinv`/`wpack` with every expert's segment in ASCENDING SLOT
/// order — the contract `group_k`'s header documents but its `atomicAdd` placement cannot guarantee (arrival
/// order). Pure function of `picks` and `wts` alone, so two ranks holding the same routing hold the same bytes.
///
/// Why only the tensor-parallel path needs it: the single-rank result is packing-INVARIANT (each slot's `y` is
/// row-independent and the scatter sums in `k` order, so any consistent permutation gives the same bits), but the
/// TP y-exchange swaps `y` BY PACKED POSITION — a race-packed `y` arrives under the peer's different intra-segment
/// order and the receiver's scatter assigns the wrong slots (the first live gate's three-way `layer_03`
/// divergence: rows whose segments' orders coincided stayed exact, the rest shifted), and the divergence then
/// feeds the router until the two ranks' exchanges stop pairing in size and the collective spins forever.
pub fn pack_stable_host(offsets: &[i32], picks: &[i32], wts: &[u16]) -> (Vec<i32>, Vec<i32>, Vec<u16>) {
    let slots = picks.len();
    assert_eq!(wts.len(), slots, "one weight a slot");
    assert!(!offsets.is_empty(), "offsets are [experts + 1]");
    let mut cursor: Vec<i32> = offsets[..offsets.len() - 1].to_vec();
    let mut perm = vec![-1i32; slots];
    let mut pinv = vec![-1i32; slots];
    let mut wpack = vec![0u16; slots];
    for (i, &e) in picks.iter().enumerate() {
        if e < 0 {
            continue; // a pruned slot (top-p): unplaced, its `pinv` stays -1 and the scatter skips it
        }
        let eu = e as usize;
        if eu >= cursor.len() {
            continue;
        }
        let p = cursor[eu];
        cursor[eu] += 1;
        perm[p as usize] = i as i32;
        pinv[i] = p;
        wpack[p as usize] = wts[i];
    }
    (perm, pinv, wpack)
}

fn own_slot_range(offsets: &[i32], own: &dyn Fn(usize) -> bool) -> Result<std::ops::Range<usize>, CudaError> {
    let mut range: Option<std::ops::Range<usize>> = None;
    for (e, &a) in offsets.iter().enumerate() {
        if !own(e) {
            continue;
        }
        let start = a as usize;
        let end = offsets.get(e + 1).copied().unwrap_or(a) as usize;
        match &mut range {
            Some(r) if r.end == start => r.end = end,
            Some(r) => {
                eprintln!("dsv4 moe: the owned experts' slots are not contiguous ([{}, {}) then [{start}, {end})): the split table is wrong", r.start, r.end);
                return Err(CudaError(1));
            }
            None => range = Some(start..end),
        }
    }
    Ok(range.unwrap_or(0..0))
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn moe_experts_with<'a, F>(
    s: &mut MoeScratch,
    mut groups_of: F,
    own: Option<&dyn Fn(usize) -> bool>,
    exchange: Option<&mut YExchange<'_>>,
    before_scatter: Option<&mut dyn FnMut(&Stream) -> Result<(), CudaError>>,
    rows: usize,
    hidden_in: &DevBuf,
    shared_f16: &mut DevBuf,
    picks: &DevBuf,
    wts: &DevBuf,
    moe_out_f32: &mut DevBuf,
    limit: f32,
    st: &Stream,
) -> Result<MoeStats, CudaError>
where
    F: FnMut(usize) -> Result<MoeGroups<'a>, CudaError>,
{
    // `rows` is this call's, not the arena's depth: a decode step of 1 row must group 1 x topk slots and read nothing
    // past them. The arena is sized for the deepest call the model can make.
    let (topk, h, inter) = (s.topk, s.hidden, s.inter);
    let slots = rows * topk;
    if rows == 0 || rows > s.rows {
        return Err(CudaError(1));
    }
    if hidden_in.bytes() < rows * h * 2
        || shared_f16.bytes() < rows * h * 2
        || moe_out_f32.bytes() < rows * h * 4
        || picks.bytes() < slots * 4
        || wts.bytes() < slots * 2
    {
        return Err(CudaError(1));
    }
    let mut stats = MoeStats { slots, ..MoeStats::default() };

    // 1. group (device): counts, offsets, perm, pinv, wpack.
    s.group_on(picks, wts, rows, st)?;
    stats.launches += 1;

    // 2. gather (device): x[p] = hidden_in[perm[p] / topk]. Launched before `st.sync()` so it runs on the GPU
    // ahead of the stream barrier while `shared_stream` continues running the shared expert uninterrupted.
    unsafe {
        CudaError::check(ffi::gb_dsv4_moe_gather(
            hidden_in.ptr(),
            s.perm.ptr() as *const i32,
            s.x.ptr(),
            i32_of(slots)?,
            i32_of(topk)?,
            i32_of(h)?,
            st.raw(),
        ))?;
    }
    stats.launches += 1;

    // 3. the one host-visible number: which experts are active and where their segments start (`group_k` places
    // every expert's segment in ascending slot order on thread 0, so both ranks hold the identical permutation;
    // `s.offsets` lives in mapped host memory, so `offsets_host` is a pure CPU read with no legacy-stream D2H copy).
    st.sync()?;
    let offsets = s.offsets_host()?;
    if offsets.len() != s.experts + 1 {
        return Err(CudaError(1));
    }

    // 4. one segment a launch, at the expert's own width, with its lease held for its three GEMMs.
    let experts = s.experts;
    let layer_no = s.layer;
    let dump = moe_dump_layer(layer_no);
    if dump {
        st.sync()?;
        let (xs, xb) = f16_extent(hidden_in);
        let (ss, sb) = f16_extent(shared_f16);
        let picks_h: Vec<i32> = picks.to_vec().unwrap_or_default();
        let wts_h: Vec<u16> = wts.to_vec().unwrap_or_default();
        eprintln!(
            "rexl3 event=dsv41_moe_dump layer={} rows={rows} slots={slots} x_absmax={xs:e} x_bad={xb} shared_absmax={ss:e} shared_bad={sb} picks={picks_h:?} wts={:?}",
            layer_no,
            wts_h.iter().map(|&b| gb_exl3::f16::f16_to_f64(b)).collect::<Vec<_>>()
        );
    }
    let MoeScratch { x, g, u, act, y, ws, wpack, pinv, .. } = s;
    for e in 0..experts {
        let (a, b) = (offsets[e] as usize, offsets[e + 1] as usize);
        let cnt = b - a;
        if cnt == 0 {
            continue;
        }
        if cnt > slots {
            return Err(CudaError(1));
        }
        // a rank of a group computes only its own experts: the peer's picks stay in the grouping (the router is replicated,
        // so the slot layout is the single-node one), their segments of `y` are the peer's to fill — by the exchange below
        if own.is_some_and(|o| !o(e)) {
            continue;
        }
        stats.active += 1;
        let mg = groups_of(e)?;
        let groups = mg.get();
        if groups.len() != 3 {
            return Err(CudaError(1));
        }
        // A segment is at most one slot a row, but a chunk deeper than the decode path's row cap would still overflow it;
        // the decode path is row invariant, so a long segment runs as windows of the cap with the same bits.
        let mut at = 0usize;
        while at < cnt {
            let n = (cnt - at).min(MAX_DECODE_ROWS);
            let xv = x.subview((a + at) * h * 2, n * h * 2);
            let mut gv = g.subview((a + at) * inter * 2, n * inter * 2);
            let mut uv = u.subview((a + at) * inter * 2, n * inter * 2);
            let mut av = act.subview((a + at) * inter * 2, n * inter * 2);
            let mut yv = y.subview((a + at) * h * 2, n * h * 2);
            groups[0].forward(&xv, &mut gv, n, Dtype::F16, Dtype::F16, &mut *ws, st)?;
            groups[2].forward(&xv, &mut uv, n, Dtype::F16, Dtype::F16, &mut *ws, st)?;
            swiglu_f16(&gv, &uv, None, &mut av, n, inter, limit, st)?;
            groups[1].forward(&av, &mut yv, n, Dtype::F16, Dtype::F16, &mut *ws, st)?;
            if dump {
                st.sync()?;
                let (xs, xb) = f16_extent(&xv);
                let (gs, gb) = f16_extent(&gv);
                let (us, ub) = f16_extent(&uv);
                let (as_, ab) = f16_extent(&av);
                let (ys, yb) = f16_extent(&yv);
                // the record itself: an all-inf projection with a finite input is a scale vector, not an arithmetic overflow
                let suh: Vec<u16> = groups[0].suh.to_vec().unwrap_or_default();
                let svh: Vec<u16> = groups[0].svh.to_vec().unwrap_or_default();
                let tre: Vec<i16> = groups[0].trellis.to_vec().unwrap_or_default();
                let f = |h: u16| gb_exl3::f16::f16_to_f64(h);
                let bad_of = |v: &[u16]| v.iter().filter(|&&h| !f(h).is_finite()).count();
                let mx_of = |v: &[u16]| v.iter().map(|&h| f(h).abs()).filter(|x| x.is_finite()).fold(0.0, f64::max);
                let tmx = tre.iter().map(|&t| (t as i64).unsigned_abs()).max().unwrap_or(0);
                eprintln!(
                    "rexl3 event=dsv41_moe_rec layer={layer_no} expert={e} suh_n={} suh_bad={} suh_absmax={:e} suh_head=[{:04x},{:04x},{:04x},{:04x}] svh_n={} svh_bad={} svh_absmax={:e} svh_head=[{:04x},{:04x},{:04x},{:04x}] trellis_n={} trellis_max={} trellis_head=[{},{},{},{}]",
                    suh.len(),
                    bad_of(&suh),
                    mx_of(&suh),
                    suh.first().copied().unwrap_or(0),
                    suh.get(1).copied().unwrap_or(0),
                    suh.get(2).copied().unwrap_or(0),
                    suh.get(3).copied().unwrap_or(0),
                    svh.len(),
                    bad_of(&svh),
                    mx_of(&svh),
                    svh.first().copied().unwrap_or(0),
                    svh.get(1).copied().unwrap_or(0),
                    svh.get(2).copied().unwrap_or(0),
                    svh.get(3).copied().unwrap_or(0),
                    tre.len(),
                    tmx,
                    tre.first().copied().unwrap_or(0),
                    tre.get(1).copied().unwrap_or(0),
                    tre.get(2).copied().unwrap_or(0),
                    tre.get(3).copied().unwrap_or(0)
                );
                eprintln!(
                    "rexl3 event=dsv41_moe_expert layer={} expert={e} cnt={cnt} n={n} bits=({},{},{}) k=({},{},{}) cb=({:?},{:?},{:?}) x={xs:e}/{xb} g={gs:e}/{gb} u={us:e}/{ub} act={as_:e}/{ab} y={ys:e}/{yb}",
                    layer_no,
                    groups[0].bits,
                    groups[1].bits,
                    groups[2].bits,
                    groups[0].k,
                    groups[1].k,
                    groups[2].k,
                    groups[0].codebook,
                    groups[1].codebook,
                    groups[2].codebook
                );
            }
            stats.launches += 4;
            at += n;
        }
        // `mg`'s lease is released here, after the last window of this expert.
    }

    // 5. the tensor-parallel combine, before the scatter: the peer's slot range of `y` still holds scratch garbage — one call
    // of the hook makes `y` the whole buffer, and the scatter below then runs identically on every rank over it.
    if let Some(exchange) = exchange {
        let own_slots = own_slot_range(&offsets, own.expect("an exchange without an ownership filter"))?;
        // the packing's OWN slot count — not the scratch buffer's capacity: the peer's range is the complement of
        // `own` within THIS call's packing, and the two ranks' exchanges must pair exactly (an exchange sized by
        // the buffer's capacity waits for bytes the peer never sends)
        let slots = offsets.last().copied().unwrap_or(0) as usize;
        exchange(y, own_slots, slots, h, st)?;
    }

    // 6. scatter-add (device): out[r] = shared[r] + sum_k w y, fp32 accumulate, both dtypes written.
    // Join `shared_stream` right here — the first and only op in `MoeExperts` that reads `shared_f16` — so the
    // shared expert's FP8 GEMMs overlap with grouping, gather, host offset readback, routed expert GEMMs and the
    // TP y-exchange.
    if let Some(join) = before_scatter {
        join(st)?;
    }
    unsafe {
        CudaError::check(ffi::gb_dsv4_moe_scatter(
            shared_f16.ptr(),
            y.ptr(),
            pinv.ptr() as *const i32,
            wpack.ptr(),
            moe_out_f32.ptr(),
            shared_f16.ptr(),
            i32_of(rows)?,
            i32_of(topk)?,
            i32_of(h)?,
            st.raw(),
        ))?;
    }
    stats.launches += 1;
    Ok(stats)
}
