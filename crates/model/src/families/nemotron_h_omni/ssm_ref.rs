//! The Mamba2 mixer's CPU reference in pure fp32: the depthwise causal conv over x|B|C, the discretization, the recurrence, and the
//! gate with the grouped RMSNorm. It is the oracle the SSM kernels are checked against, so it favours the plain formula over speed.
//!
//! Semantics (from the fork, `exllamav3/modules/mamba2.py`, `gated_delta_net_fn/conv1d.py`, `gated_rmsnorm.py`, and `mamba2_dt_op`
//! in `gdn.cu`), with every intermediate kept in fp32:
//! - conv: `out[c] = silu(sum_k w[c][k] * x[t-3+k][c] + bias[c])`, taps `k = 0..4`, so `w[c][3]` multiplies the current token. Only
//!   x|B|C are convolved. z and dt are not.
//! - dt: `dt = clamp(softplus(dt_raw + dt_bias), 0, +inf)`, with softplus linear above 20; `g = -exp(A_log) * dt`.
//! - recurrence, fp32 state laid out `[head][state][head_dim]`: `state = state * exp(g) + B outer (x * dt)`, then `y = C . state + D * x`.
//!   Head `h` reads B and C of group `h / (heads / groups)`.
//! - gate: `v = y * silu(z)`, then per group of `heads * head_dim / groups` channels, `out = w * v / sqrt(mean(v^2) + eps)`.
//!
//! The output stops before `out_proj`. The fork's CUDA path rounds conv output, dt, z and the core output to bf16, and this
//! reference does not, so the bf16 choice is left to the engine kernels.

/// The Mamba2 geometry: heads, width of a head, B/C groups, and the state width per group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub heads: usize,
    pub head_dim: usize,
    pub groups: usize,
    pub state: usize,
}

impl Dims {
    /// The measured geometry of the Omni text backbone (`config.rs`, `GEOMETRY`).
    pub const OMNI: Dims = Dims { heads: 128, head_dim: 64, groups: 8, state: 128 };

    /// Width of the z and x streams (`heads * head_dim`), also the number of channels in the gate.
    pub fn inner(&self) -> usize {
        self.heads * self.head_dim
    }
    /// Width of B, and of C: `groups * state`.
    pub fn bc(&self) -> usize {
        self.groups * self.state
    }
    /// Channels that the causal conv runs over: x | B | C.
    pub fn conv_channels(&self) -> usize {
        self.inner() + 2 * self.bc()
    }
    /// Heads per B/C group.
    pub fn heads_per_group(&self) -> usize {
        self.heads / self.groups
    }
    /// Elements of the fp32 recurrent state for one sequence.
    pub fn state_len(&self) -> usize {
        self.heads * self.state * self.head_dim
    }
}

/// Taps of the depthwise causal conv.
pub const CONV_TAPS: usize = 4;
/// Pre-conv rows kept between tokens (`CONV_TAPS - 1`).
pub const CONV_HISTORY: usize = CONV_TAPS - 1;

#[inline]
pub fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// `x` for `x > 20`, else `ln(1 + e^x)`: the threshold of the fork's `softplus` in `gdn.cu`.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// `(dt, g)` for one head: `dt = clamp(softplus(dt_raw + dt_bias), 0, +inf)` and `g = -exp(A_log) * dt`. A NaN input stays NaN.
#[inline]
pub fn discretize(dt_raw: f32, dt_bias: f32, a_log: f32) -> (f32, f32) {
    let dt = softplus(dt_raw + dt_bias);
    let dt = if dt < 0.0 { 0.0 } else { dt };
    (dt, -a_log.exp() * dt)
}

/// Splits one token's conv output into `(x, B, C)`.
pub fn split_xbc(dims: Dims, xbc: &[f32]) -> (&[f32], &[f32], &[f32]) {
    let inner = dims.inner();
    let bc = dims.bc();
    (&xbc[..inner], &xbc[inner..inner + bc], &xbc[inner + bc..inner + 2 * bc])
}

/// The conv before the activation: `history` holds the three previous pre-conv rows, oldest first (`history[j * C + c]`), and is
/// advanced past `x`. `w` is `[C][4]` and `bias` is `[C]`.
pub fn conv_pre_step(history: &mut [f32], w: &[f32], bias: &[f32], x: &[f32], pre: &mut [f32]) {
    let c = x.len();
    assert_eq!(history.len(), CONV_HISTORY * c, "conv history is [3][C]");
    assert_eq!(w.len(), c * CONV_TAPS, "conv weight is [C][4]");
    assert_eq!(bias.len(), c, "conv bias is [C]");
    assert_eq!(pre.len(), c, "conv output is [C]");
    for ch in 0..c {
        let h0 = history[ch];
        let h1 = history[c + ch];
        let h2 = history[2 * c + ch];
        let wc = &w[ch * CONV_TAPS..(ch + 1) * CONV_TAPS];
        pre[ch] = wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * x[ch] + bias[ch];
    }
    history.copy_within(c..3 * c, 0);
    history[2 * c..].copy_from_slice(x);
}

/// One token of the depthwise causal conv over x|B|C, with silu. `history` is `[3][C]` and advances as in [`conv_pre_step`].
pub fn conv_step(history: &mut [f32], w: &[f32], bias: &[f32], x: &[f32], out: &mut [f32]) {
    conv_pre_step(history, w, bias, x, out);
    for v in out.iter_mut() {
        *v = silu(*v);
    }
}

/// One token of the recurrence. `state` is fp32 `[heads][state][head_dim]` and is updated in place. `x` is `[heads * head_dim]`,
/// `b` and `c` are `[groups * state]`, and `dt`, `g` and `d` are `[heads]`. `y` is `[heads * head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn recurrent_step(
    dims: Dims,
    state: &mut [f32],
    x: &[f32],
    b: &[f32],
    c: &[f32],
    dt: &[f32],
    g: &[f32],
    d: &[f32],
    y: &mut [f32],
) {
    let (heads, p, n) = (dims.heads, dims.head_dim, dims.state);
    assert_eq!(state.len(), dims.state_len(), "state is [heads][state][head_dim]");
    assert_eq!(x.len(), heads * p, "x is [heads * head_dim]");
    assert_eq!(b.len(), dims.bc(), "B is [groups * state]");
    assert_eq!(c.len(), dims.bc(), "C is [groups * state]");
    assert!(dt.len() == heads && g.len() == heads && d.len() == heads, "dt, g and D are [heads]");
    assert_eq!(y.len(), heads * p, "y is [heads * head_dim]");
    for h in 0..heads {
        let grp = h / dims.heads_per_group();
        let decay = g[h].exp();
        let xh = &x[h * p..(h + 1) * p];
        let sh = &mut state[h * n * p..(h + 1) * n * p];
        let bg = &b[grp * n..(grp + 1) * n];
        for i in 0..n {
            let bi = bg[i];
            let row = &mut sh[i * p..(i + 1) * p];
            for j in 0..p {
                row[j] = row[j] * decay + bi * (xh[j] * dt[h]);
            }
        }
        let cg = &c[grp * n..(grp + 1) * n];
        let yh = &mut y[h * p..(h + 1) * p];
        for j in 0..p {
            yh[j] = d[h] * xh[j];
        }
        for i in 0..n {
            let ci = cg[i];
            let row = &sh[i * p..(i + 1) * p];
            for j in 0..p {
                yh[j] += row[j] * ci;
            }
        }
    }
}

/// `t` tokens of [`recurrent_step`], in order. Per token: `xs` is `[t][heads * head_dim]`, `bs` and `cs` are `[t][groups * state]`,
/// `dts` and `gs` are `[t][heads]`, and `ys` is `[t][heads * head_dim]`. `d` is `[heads]`, shared by every token.
#[allow(clippy::too_many_arguments)]
pub fn prefill_scan(
    dims: Dims,
    state: &mut [f32],
    t: usize,
    xs: &[f32],
    bs: &[f32],
    cs: &[f32],
    dts: &[f32],
    gs: &[f32],
    d: &[f32],
    ys: &mut [f32],
) {
    let inner = dims.inner();
    let bc = dims.bc();
    let h = dims.heads;
    assert!(xs.len() == t * inner && ys.len() == t * inner, "x and y are [t][heads * head_dim]");
    assert!(bs.len() == t * bc && cs.len() == t * bc, "B and C are [t][groups * state]");
    assert!(dts.len() == t * h && gs.len() == t * h, "dt and g are [t][heads]");
    for step in 0..t {
        recurrent_step(
            dims,
            state,
            &xs[step * inner..(step + 1) * inner],
            &bs[step * bc..(step + 1) * bc],
            &cs[step * bc..(step + 1) * bc],
            &dts[step * h..(step + 1) * h],
            &gs[step * h..(step + 1) * h],
            d,
            &mut ys[step * inner..(step + 1) * inner],
        );
    }
}

/// The gate and the grouped RMSNorm for one token: `out = w * (y * silu(z)) / sqrt(mean_group((y * silu(z))^2) + eps)`, with the
/// mean taken over each group of `inner / groups` channels. `y`, `z`, `w` and `out` are `[inner]`.
pub fn gated_norm(dims: Dims, y: &[f32], z: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let inner = dims.inner();
    let gw = inner / dims.groups;
    assert!(y.len() == inner && z.len() == inner && w.len() == inner && out.len() == inner, "gate and norm are [inner]");
    for grp in 0..dims.groups {
        let r = grp * gw..(grp + 1) * gw;
        let mut sq = 0.0f32;
        for i in r.clone() {
            let v = y[i] * silu(z[i]);
            out[i] = v;
            sq += v * v;
        }
        let inv = 1.0 / (sq / gw as f32 + eps).sqrt();
        for i in r {
            out[i] *= inv;
            out[i] *= w[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic fill in [-1, 1): a 64-bit LCG, so the tests need no RNG crate.
    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    const SMALL: Dims = Dims { heads: 4, head_dim: 3, groups: 2, state: 5 };

    #[test]
    fn the_omni_geometry_is_the_measured_one() {
        let d = Dims::OMNI;
        assert_eq!((d.inner(), d.bc(), d.conv_channels()), (8192, 1024, 10240));
        assert_eq!(d.heads_per_group(), 16);
        assert_eq!(d.state_len(), 128 * 128 * 64);
    }

    #[test]
    fn softplus_is_linear_above_the_threshold_and_ln2_at_zero() {
        assert_eq!(softplus(25.0), 25.0);
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        assert!((softplus(-30.0) - (-30.0f32).exp()).abs() < 1e-12);
    }

    #[test]
    fn discretize_is_the_clamped_softplus_and_a_negative_log_decay() {
        let (dt, g) = discretize(0.0, 0.0, 0.0);
        assert!((dt - std::f32::consts::LN_2).abs() < 1e-7);
        assert!((g + dt).abs() < 1e-7, "exp(A_log) is 1 at A_log = 0");
        let (dt, g) = discretize(-50.0, 0.0, 1.0);
        assert!((0.0..1e-6).contains(&dt) && g <= 0.0);
        assert!(discretize(f32::NAN, 0.0, 0.0).0.is_nan(), "NaN stays NaN, as torch clamp does");
    }

    #[test]
    fn conv_taps_run_oldest_to_current_and_the_history_advances() {
        // One channel: w = [1, 10, 100, 1000], history oldest first = [1, 2, 3], x = 5, so the taps give 1 + 20 + 300 + 5000.
        let w = [1.0f32, 10.0, 100.0, 1000.0];
        let bias = [0.0f32];
        let mut hist = vec![1.0f32, 2.0, 3.0];
        let mut pre = [0.0f32];
        conv_pre_step(&mut hist, &w, &bias, &[5.0], &mut pre);
        assert_eq!(pre[0], 5321.0);
        assert_eq!(hist, vec![2.0, 3.0, 5.0]);
        conv_pre_step(&mut hist, &w, &bias, &[7.0], &mut pre);
        assert_eq!(pre[0], 2.0 + 30.0 + 500.0 + 7000.0);
    }

    #[test]
    fn conv_step_is_silu_of_the_pre_activation() {
        let w = fill(8, 1);
        let bias = fill(2, 2);
        let x = fill(2, 3);
        let mut hist_a = fill(6, 4);
        let mut hist_b = hist_a.clone();
        let mut pre = [0.0f32; 2];
        let mut out = [0.0f32; 2];
        conv_pre_step(&mut hist_a, &w, &bias, &x, &mut pre);
        conv_step(&mut hist_b, &w, &bias, &x, &mut out);
        for c in 0..2 {
            assert_eq!(out[c], silu(pre[c]));
        }
        assert_eq!(hist_a, hist_b);
    }

    #[test]
    fn a_token_from_zero_state_is_x_times_d_plus_dt_times_b_dot_c() {
        let dims = SMALL;
        let x = fill(dims.inner(), 10);
        let b = fill(dims.bc(), 11);
        let c = fill(dims.bc(), 12);
        let dt = [0.3f32, 0.7, 0.1, 2.0];
        let g = [-0.1f32, -0.2, -0.3, -0.4];
        let d = [0.5f32, -1.0, 2.0, 0.25];
        let mut state = vec![0.0f32; dims.state_len()];
        let mut y = vec![0.0f32; dims.inner()];
        recurrent_step(dims, &mut state, &x, &b, &c, &dt, &g, &d, &mut y);
        for h in 0..dims.heads {
            let grp = h / dims.heads_per_group();
            let bc: f32 = (0..dims.state).map(|n| b[grp * dims.state + n] * c[grp * dims.state + n]).sum();
            for p in 0..dims.head_dim {
                let xv = x[h * dims.head_dim + p];
                let want = d[h] * xv + dt[h] * xv * bc;
                assert!((y[h * dims.head_dim + p] - want).abs() < 1e-5, "head {h} channel {p}");
            }
        }
    }

    #[test]
    fn the_decay_is_exp_g_per_token_on_a_single_state_element() {
        // One head, one state element, one channel: S1 = B1 * x1 * dt1, then S2 = S1 * exp(g) + B2 * x2 * dt2.
        let dims = Dims { heads: 1, head_dim: 1, groups: 1, state: 1 };
        let mut state = vec![0.0f32];
        let mut y = vec![0.0f32];
        let (dt1, g1) = (0.5f32, -0.25f32);
        recurrent_step(dims, &mut state, &[2.0], &[3.0], &[1.0], &[dt1], &[g1], &[0.0], &mut y);
        let s1 = 3.0 * 2.0 * dt1;
        assert!((state[0] - s1).abs() < 1e-6);
        assert!((y[0] - s1).abs() < 1e-6);
        let (dt2, g2) = (1.5f32, -0.5f32);
        recurrent_step(dims, &mut state, &[-1.0], &[4.0], &[2.0], &[dt2], &[g2], &[0.0], &mut y);
        let s2 = s1 * g2.exp() - 4.0 * dt2;
        assert!((state[0] - s2).abs() < 1e-6);
        assert!((y[0] - 2.0 * s2).abs() < 1e-6);
    }

    #[test]
    fn head_h_reads_the_b_and_c_of_group_h_over_heads_per_group() {
        // Only group 1 has non-zero B and C, so only heads 2 and 3 (heads_per_group = 2) may change.
        let dims = SMALL;
        let x = fill(dims.inner(), 20);
        let mut b = vec![0.0f32; dims.bc()];
        let mut c = vec![0.0f32; dims.bc()];
        for n in 0..dims.state {
            b[dims.state + n] = 1.0;
            c[dims.state + n] = 1.0;
        }
        let mut state = vec![0.0f32; dims.state_len()];
        let mut y = vec![0.0f32; dims.inner()];
        recurrent_step(dims, &mut state, &x, &b, &c, &[1.0; 4], &[-0.1; 4], &[0.0; 4], &mut y);
        let per_head = dims.state * dims.head_dim;
        for h in 0..2 {
            assert!(state[h * per_head..(h + 1) * per_head].iter().all(|v| *v == 0.0), "head {h} is group 0");
        }
        for h in 2..4 {
            assert!(state[h * per_head..(h + 1) * per_head].iter().any(|v| *v != 0.0), "head {h} is group 1");
        }
    }

    #[test]
    fn prefill_scan_equals_repeated_recurrent_steps_exactly() {
        let dims = SMALL;
        let t = 7;
        let xs = fill(t * dims.inner(), 30);
        let bs = fill(t * dims.bc(), 31);
        let cs = fill(t * dims.bc(), 32);
        let dts: Vec<f32> = fill(t * dims.heads, 33).iter().map(|v| v.abs() * 0.5).collect();
        let gs: Vec<f32> = fill(t * dims.heads, 34).iter().map(|v| -v.abs()).collect();
        let d = fill(dims.heads, 35);

        let mut scan_state = vec![0.0f32; dims.state_len()];
        let mut scan_y = vec![0.0f32; t * dims.inner()];
        prefill_scan(dims, &mut scan_state, t, &xs, &bs, &cs, &dts, &gs, &d, &mut scan_y);

        let mut loop_state = vec![0.0f32; dims.state_len()];
        let mut loop_y = vec![0.0f32; t * dims.inner()];
        for s in 0..t {
            let ys = &mut loop_y[s * dims.inner()..(s + 1) * dims.inner()];
            recurrent_step(
                dims,
                &mut loop_state,
                &xs[s * dims.inner()..(s + 1) * dims.inner()],
                &bs[s * dims.bc()..(s + 1) * dims.bc()],
                &cs[s * dims.bc()..(s + 1) * dims.bc()],
                &dts[s * dims.heads..(s + 1) * dims.heads],
                &gs[s * dims.heads..(s + 1) * dims.heads],
                &d,
                ys,
            );
        }
        assert_eq!(scan_y, loop_y);
        assert_eq!(scan_state, loop_state);
    }

    #[test]
    fn a_zero_gate_zeroes_the_output_and_the_norm_is_unit_rms_per_group() {
        let dims = Dims { heads: 4, head_dim: 4, groups: 2, state: 2 };
        let y = fill(dims.inner(), 40);
        let w = vec![1.0f32; dims.inner()];
        let mut out = vec![0.0f32; dims.inner()];
        gated_norm(dims, &y, &vec![0.0; dims.inner()], &w, 1e-5, &mut out);
        assert!(out.iter().all(|v| *v == 0.0), "silu(0) = 0 closes the gate");

        let z = fill(dims.inner(), 41);
        gated_norm(dims, &y, &z, &w, 1e-5, &mut out);
        let gw = dims.inner() / dims.groups;
        for grp in 0..dims.groups {
            let ms: f32 = out[grp * gw..(grp + 1) * gw].iter().map(|v| v * v).sum::<f32>() / gw as f32;
            assert!((ms - 1.0).abs() < 1e-3, "group {grp} mean square {ms}");
        }
    }

    #[test]
    fn the_norm_weight_is_per_channel_and_applied_after_the_normalization() {
        let dims = Dims { heads: 2, head_dim: 2, groups: 1, state: 1 };
        let y = [1.0f32, 2.0, 3.0, 4.0];
        let z = [100.0f32; 4]; // silu(100) ~ 100, so the gated value is ~ y * 100
        let mut ones = [0.0f32; 4];
        gated_norm(dims, &y, &z, &[1.0; 4], 0.0, &mut ones);
        let mut scaled = [0.0f32; 4];
        gated_norm(dims, &y, &z, &[2.0, 2.0, 0.5, 0.5], 0.0, &mut scaled);
        for i in 0..4 {
            let expect = ones[i] * if i < 2 { 2.0 } else { 0.5 };
            assert!((scaled[i] - expect).abs() < 1e-6, "channel {i}");
        }
    }
}
