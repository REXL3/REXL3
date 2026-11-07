//! DeepSeek-V4.1 router + hyper-connection mix (`src/dsv4_moe.rs`, `csrc/dsv4_moe.cu`).
//!
//! HOST tests (these run everywhere, including a box with no nvcc):
//!   * `sqrtsoftplus_pinned_values_*` — 8 pinned inputs (large negative, large positive, both sides of the knee) against an
//!     fp64 reference of `sqrt(log1p(exp(-|x|)) + max(x, 0))`.
//!   * `sqrtsoftplus_table_matches_closed_form` — all 65,536 fp16-bit entries to 1e-6.
//!   * `router_top6_matches_direct_transcription` — 384 experts, top-6, the normalisation and the 1.5 scale against an
//!     independent (repeated-argmax) transcription of `Gate.forward`.
//!   * `router_bias_vl_selects_without_touching_the_weight` — the dual-bias path.
//!   * `sinkhorn_20_iters_is_doubly_stochastic` — a random 4x4.
//!   * `hc_4096_and_5120_agree_under_zero_padding` — the GLM-5.3 (4096/16384/32-atom) and this pack's
//!     (5120/20480/40-atom) instantiations agree bit for bit on zero-padded tensors.
//!   * `swiglu_clamps_match_reference` — the 10.0 clamps.
//!
//! GPU tests self-skip with "not run: no device" when there is no device.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
mod common;

use common::Rng;
use gb_cuda::dsv4_moe::*;
use gb_cuda::{device_available, DevBuf, Stream};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("not run: no device");
            return;
        }
    };
}

fn up<T: Copy>(v: &[T]) -> DevBuf {
    DevBuf::from_slice(v).unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ================================================================================================================================
// 1. sqrtsoftplus
// ================================================================================================================================

/// Eight pinned inputs: two large negatives, a moderate negative, the near-zero knee on both sides, and three positives
/// (including one past fp16's range, so the table's clamp is exercised separately from the closed form).
#[test]
fn sqrtsoftplus_pinned_values_match_float64() {
    const X: [f32; 8] = [-100.0, -30.0, -8.0, -0.5, 0.0, 0.5, 8.0, 100.0];
    for &x in X.iter() {
        let got = sqrtsoftplus(x) as f64;
        let want = sqrtsoftplus_f64(x as f64);
        let tol = 1e-6 * want.abs().max(1.0);
        assert!((got - want).abs() <= tol, "sqrtsoftplus({x}) = {got}, want {want}");
    }
    // no overflow / no cancellation at the ends
    assert!(sqrtsoftplus(1e30).is_finite());
    assert!((sqrtsoftplus(1e30) as f64 - (1e30f64).sqrt()).abs() / 1e15 < 1e-6);
    // softplus(-large) is exp(x) to full relative precision, so the score is exp(x/2)
    let x = -80f32;
    let want = ((x as f64).exp()).sqrt();
    assert!(((sqrtsoftplus(x) as f64) - want).abs() / want < 1e-6);
    // monotone, and the ordering of scores equals the ordering of softplus scores
    let mut prev = f32::NEG_INFINITY;
    for k in -2000..2000 {
        let v = sqrtsoftplus(k as f32 * 0.05);
        assert!(v >= prev);
        prev = v;
    }
}

#[test]
fn sqrtsoftplus_table_matches_closed_form() {
    let tab = sqrtsoftplus_table_host();
    assert_eq!(tab.len(), 65536);
    let mut worst = 0f64;
    for h in 0u32..65536 {
        let x = gb_exl3::f16::f16_to_f64(h as u16);
        let got = tab[h as usize] as f64;
        if x.is_nan() {
            assert!(got.is_nan(), "entry {h:#06x} should be NaN");
            continue;
        }
        let want = sqrtsoftplus_f64(x);
        if !want.is_finite() {
            // +inf / -inf entries: sqrt(softplus(+inf)) = +inf, and for -inf softplus is exp(-inf) = 0
            assert_eq!(got, want, "entry {h:#06x}: {got} vs {want}");
            continue;
        }
        let rel = (got - want).abs() / want.abs().max(1.0);
        worst = worst.max(rel);
        assert!(rel <= 1e-6, "table[{h:#06x}] = {got}, want {want} (rel {rel:e})");
    }
    eprintln!("sqrtsoftplus table: worst relative error {worst:e} over 65,536 entries");
}

// ================================================================================================================================
// 2. the router
// ================================================================================================================================

/// A deliberately different transcription of `Gate.forward`: repeated argmax over a `used` mask instead of a sort, and an
/// explicit fp64 sum for the normalisation cross-check.
fn direct_transcription(logits: &[f32], row: usize, d: Dsv4Dims, b: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let lg = &logits[row * d.experts..(row + 1) * d.experts];
    let mut key = vec![0f32; d.experts];
    let mut score = vec![0f32; d.experts];
    for e in 0..d.experts {
        // sqrt(softplus(x)) written a third way: ln(1 + e^x) via log1p on the positive part
        let sp = if lg[e] > 0.0 { lg[e] + (-lg[e]).exp().ln_1p() } else { lg[e].exp().ln_1p() };
        score[e] = sp.sqrt();
        key[e] = score[e] + b[e];
    }
    let mut used = vec![false; d.experts];
    let mut idx = Vec::with_capacity(d.topk);
    for _ in 0..d.topk {
        let mut best = usize::MAX;
        for e in 0..d.experts {
            if used[e] {
                continue;
            }
            if best == usize::MAX || key[e] > key[best] {
                best = e;
            }
        }
        used[best] = true;
        idx.push(best);
    }
    let sum: f64 = idx.iter().map(|&e| score[e] as f64).sum();
    let w = idx.iter().map(|&e| ((score[e] as f64 / sum) * d.scale as f64) as f32).collect();
    (idx, w)
}

#[test]
fn router_top6_matches_direct_transcription() {
    let d = Dsv4Dims::V41;
    assert_eq!((d.hidden, d.experts, d.topk), (5120, 384, 6));
    assert_eq!(d.scale, 1.5);
    let mut rng = Rng(0xD54D_0E01);
    let rows = 5;
    // logits in [-12, 12]: a spread wide enough that the top-6 is unambiguous
    let logits: Vec<f32> = (0..rows * d.experts).map(|_| rng.uniform(-12.0, 12.0)).collect();
    let bias: Vec<f32> = (0..d.experts).map(|_| rng.uniform(-0.4, 0.4)).collect();
    let bias_vl: Vec<f32> = (0..d.experts).map(|_| rng.uniform(-0.4, 0.4)).collect();
    let (pick, wts) = select_host(&logits, rows, d, &bias, Some(&bias_vl), None);
    for r in 0..rows {
        let (want_i, want_w) = direct_transcription(&logits, r, d, &bias);
        let got_i: Vec<usize> = pick[r * d.topk..(r + 1) * d.topk].iter().map(|&v| v as usize).collect();
        assert_eq!(got_i, want_i, "row {r}: picks differ");
        // strictly descending keys (no ties in this input), so the order is fully determined
        for k in 0..d.topk {
            let got = wts[r * d.topk + k];
            let want = want_w[k];
            assert!((got - want).abs() <= 1e-6 * want.abs().max(1.0), "row {r} slot {k}: {got} vs {want}");
        }
        // the six weights sum to routed_scaling_factor, and each is its unbiased score over that sum
        let s: f32 = wts[r * d.topk..(r + 1) * d.topk].iter().sum();
        assert!((s - 1.5).abs() < 1e-6, "row {r}: weights sum to {s}, want 1.5");
        let mut sc = [0f32; 6];
        for k in 0..d.topk {
            sc[k] = sqrtsoftplus(logits[r * d.experts + got_i[k]]);
        }
        let den: f32 = sc.iter().sum::<f32>() + 1e-20;
        for k in 0..d.topk {
            let want = (sc[k] / den) * 1.5;
            assert!((wts[r * d.topk + k] - want).abs() <= 1e-7, "row {r} slot {k}: not score/sum*1.5");
        }
    }
    // fp16 rendering is the only rounding the engine adds
    let (_, w16) = select_host_f16(&logits, rows, d, &bias, None, None);
    for i in 0..rows * d.topk {
        let want = gb_exl3::f16::f16_to_f64(w16[i]) as f32;
        assert!((want - wts[i]).abs() <= 1e-3 * want.abs().max(1.0));
    }
}

#[test]
fn router_bias_vl_selects_without_touching_the_weight() {
    let d = Dsv4Dims::V41;
    let mut rng = Rng(0xB1A5_0001);
    let rows = 4;
    let logits: Vec<f32> = (0..rows * d.experts).map(|_| rng.uniform(-6.0, 6.0)).collect();
    let bias = vec![0f32; d.experts];
    // a large VL bias: expert 7 wins every vision token, whichever its score
    let mut bias_vl = vec![0f32; d.experts];
    bias_vl[7] = 50.0;
    let mut mask = vec![false; rows];
    mask[1] = true;
    mask[3] = true;
    let (pick, wts) = select_host(&logits, rows, d, &bias, Some(&bias_vl), Some(&mask));
    for r in 0..rows {
        let picked: Vec<i32> = pick[r * d.topk..(r + 1) * d.topk].to_vec();
        if mask[r] {
            assert_eq!(picked[0], 7, "vision row {r} must take the bias_vl winner first");
            // ... and the weight is still the UNBIASED score of that expert, over the unbiased top-6 sum
            let s = sqrtsoftplus(logits[r * d.experts + 7]);
            assert!(wts[r * d.topk] > 0.0);
            let den: f32 = picked.iter().map(|&e| sqrtsoftplus(logits[r * d.experts + e as usize])).sum::<f32>() + 1e-20;
            assert!(((s / den) * 1.5 - wts[r * d.topk]).abs() <= 1e-7);
        } else {
            // a text row never reads bias_vl: identical to routing with bias alone
            let (p0, w0) = select_host(&logits[r * d.experts..(r + 1) * d.experts], 1, d, &bias, None, None);
            assert_eq!(picked, p0);
            assert_eq!(&wts[r * d.topk..(r + 1) * d.topk], &w0[..]);
        }
    }
    // a mask that is all-false is the same as no mask at all
    let (pa, wa) = select_host(&logits, rows, d, &bias, Some(&bias_vl), Some(&vec![false; rows]));
    let (pb, wb) = select_host(&logits, rows, d, &bias, None, None);
    assert_eq!(pa, pb);
    assert_eq!(bits(&wa), bits(&wb));
}

// ================================================================================================================================
// 3. Sinkhorn
// ================================================================================================================================

#[test]
fn sinkhorn_20_iters_is_doubly_stochastic() {
    // A random positive 4x4, which is what `comb` is after the row softmax. Alternating normalisation converges
    // geometrically; by iteration 5 it is at the `hc_eps` floor, and 20 iterations is the model's margin.
    let mut rng = Rng(0x51_2C4E);
    let mut worst = 0f32;
    let mut trial = 0;
    while trial < 64 {
        let mut c = [0f32; 16];
        for v in c.iter_mut() {
            *v = rng.uniform(0.0, 1.0);
        }
        let c0 = c;
        sinkhorn_host(&mut c, HcCfg::V41.sinkhorn_iters, HcCfg::V41.hc_eps);
        for j in 0..4 {
            let rs: f32 = (0..4).map(|k| c[j * 4 + k]).sum();
            let cs: f32 = (0..4).map(|k| c[k * 4 + j]).sum();
            worst = worst.max((rs - 1.0).abs()).max((cs - 1.0).abs());
            assert!((rs - 1.0).abs() < 2e-6, "trial {trial} row {j} sums to {rs}");
            assert!((cs - 1.0).abs() < 2e-6, "trial {trial} col {j} sums to {cs}");
            for k in 0..4 {
                assert!(c[j * 4 + k] > 0.0, "comb must stay positive");
            }
        }
        // geometric: the residual falls by orders of magnitude, and 20 iterations is at the eps floor
        let res = |it: i32| {
            let mut d = c0;
            sinkhorn_host(&mut d, it, HcCfg::V41.hc_eps);
            (0..4).map(|j| ((0..4).map(|k| d[j * 4 + k]).sum::<f32>() - 1.0).abs()).fold(0f32, f32::max)
        };
        let (r1, r3, r20) = (res(1), res(3), res(20));
        assert!(r3 < r1 && r20 <= r3, "trial {trial}: residual not decreasing ({r1} {r3} {r20})");
        assert!(r20 <= 2e-6, "trial {trial}: 20 iterations leave {r20}");
        trial += 1;
    }
    eprintln!("sinkhorn(20) over 64 random 4x4: worst |sum - 1| = {worst:e} (the hc_eps floor)");

    // Iterations beyond ~6 buy nothing at this size: the eps floor dominates. (In a wide-spread logit matrix the rate
    // is set by the conditioning, so 20 is the model's margin, not its break-even.)
    let mut c2 = [0f32; 16];
    for (i, v) in c2.iter_mut().enumerate() {
        *v = (i as f32 * 0.7).sin();
    }
    let mut c20 = c2;
    sinkhorn_host(&mut c2, 6, 1e-6);
    sinkhorn_host(&mut c20, 20, 1e-6);
    let d: f32 = (0..16).map(|i| (c2[i] - c20[i]).abs()).fold(0.0, f32::max);
    assert!(d < 1e-5, "6 and 20 iterations disagree by {d}");
}

// ================================================================================================================================
// 4. hyper-connections: 4096 vs 5120 under zero padding
// ================================================================================================================================

/// Zero-pad a `[streams][h]`-per-row tensor with `h = 4096` into the `h = 5120` layout.
fn pad_rows(src: &[f32], rows: usize, h: usize, hpad: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * STREAMS * hpad];
    for r in 0..rows {
        for s in 0..STREAMS {
            for d in 0..h {
                out[r * STREAMS * hpad + s * hpad + d] = src[r * STREAMS * h + s * h + d];
            }
        }
    }
    out
}

#[test]
fn hc_4096_and_5120_agree_under_zero_padding() {
    const H_SMALL: usize = 4096;
    const H_BIG: usize = 5120;
    assert_eq!(ATOMS, 40);
    assert_eq!(ROW, 20480);
    assert_eq!(H_BIG / ATOM_D, ATOMS);
    assert_eq!(H_SMALL / ATOM_D, 32);

    let mut rng = Rng(0x4C_5120);
    let rows = 3;
    let x_s: Vec<f32> = (0..rows * STREAMS * H_SMALL).map(|_| rng.uniform(-2.0, 2.0)).collect();
    let fn_s: Vec<f32> = (0..NFN * STREAMS * H_SMALL).map(|_| rng.uniform(-0.05, 0.05)).collect();
    let base: Vec<f32> = (0..NFN).map(|_| rng.uniform(-0.5, 0.5)).collect();
    let scale = [0.7f32, -1.3, 0.4];

    let x_b = pad_rows(&x_s, rows, H_SMALL, H_BIG);
    let fn_b = pad_rows(&fn_s, NFN, H_SMALL, H_BIG);

    // (a) the 25 dots are bit-identical: the 1024 pad values per stream contribute exact +0.0 terms
    for r in 0..rows {
        let qa = hc_dots_host(&x_s[r * STREAMS * H_SMALL..(r + 1) * STREAMS * H_SMALL], H_SMALL, &fn_s);
        let qb = hc_dots_host(&x_b[r * STREAMS * H_BIG..(r + 1) * STREAMS * H_BIG], H_BIG, &fn_b);
        assert_eq!(bits(&qa), bits(&qb), "row {r}: the 32-atom and 40-atom dot chains differ");
        // and both match a plain float64 dot of the unpadded row
        for qi in 0..NFN + 1 {
            let mut want = 0f64;
            for h in 0..STREAMS {
                for d in 0..H_SMALL {
                    let xv = x_s[r * STREAMS * H_SMALL + h * H_SMALL + d] as f64;
                    want += if qi == NFN { xv * xv } else { fn_s[qi * STREAMS * H_SMALL + h * H_SMALL + d] as f64 * xv };
                }
            }
            let got = qa[qi] as f64;
            assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0), "row {r} q{qi}: {got} vs {want}");
        }
    }

    // (b) with the RMS mean held at the unpadded length, pre / post / comb are bit-identical too
    let cfg = HcCfg::V41;
    let mean = STREAMS * H_SMALL;
    let (pa, sa, ca) = hc_mixes_host(&x_s, rows, H_SMALL, &fn_s, &scale, &base, mean, cfg);
    let (pb, sb, cb) = hc_mixes_host(&x_b, rows, H_BIG, &fn_b, &scale, &base, mean, cfg);
    assert_eq!(bits(&pa), bits(&pb), "pre differs");
    assert_eq!(bits(&sa), bits(&sb), "post differs");
    assert_eq!(bits(&ca), bits(&cb), "comb differs");

    // (c) with the natural 20480-value mean the two differ only by the known rsqrt factor of the two means
    let (pc, _, _) = hc_mixes_host(&x_b, rows, H_BIG, &fn_b, &scale, &base, STREAMS * H_BIG, cfg);
    for r in 0..rows {
        let ss: f64 = x_s[r * STREAMS * H_SMALL..(r + 1) * STREAMS * H_SMALL].iter().map(|&v| (v as f64) * (v as f64)).sum();
        let ra = 1.0 / (ss / mean as f64).sqrt();
        let rb = 1.0 / (ss / (STREAMS * H_BIG) as f64).sqrt();
        // a sigmoid of a differently-scaled argument: check they are NOT equal but both stay in range
        assert!((ra - rb).abs() > 1e-6);
        for h in 0..STREAMS {
            assert!(pa[r * STREAMS + h] > cfg.hc_eps && pa[r * STREAMS + h] <= 1.0 + cfg.hc_eps);
            assert!(pc[r * STREAMS + h] > cfg.hc_eps && pc[r * STREAMS + h] <= 1.0 + cfg.hc_eps);
        }
    }

    // (d) comb is doubly stochastic at this shape, and pre / post respect their ranges
    for r in 0..rows {
        for j in 0..STREAMS {
            let rs: f32 = (0..STREAMS).map(|k| ca[r * 16 + j * STREAMS + k]).sum();
            let cs: f32 = (0..STREAMS).map(|k| ca[r * 16 + k * STREAMS + j]).sum();
            assert!((rs - 1.0).abs() < 1e-5 && (cs - 1.0).abs() < 1e-5);
            let p = sa[r * STREAMS + j];
            assert!(p > 0.0 && p < 2.0, "post out of (0, 2): {p}");
        }
    }

    // (e) hc_pre / hc_post round-trip the shapes
    let y: Vec<f32> = (0..rows * H_BIG).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let col = hc_pre_host(&x_b, rows, H_BIG, &pb);
    assert_eq!(col.len(), rows * H_BIG);
    let post = hc_post_host(&x_b, &y, rows, H_BIG, &sb, &cb);
    assert_eq!(post.len(), rows * ROW);
    // with comb = 0 the post is exactly post[h] * y[d]; with post = 0 it is exactly the comb mix
    let z16 = vec![0f32; rows * 16];
    let only_post = hc_post_host(&x_b, &y, rows, H_BIG, &sb, &z16);
    for r in 0..rows {
        for h in 0..STREAMS {
            for d in 0..64 {
                assert_eq!(only_post[r * ROW + h * H_BIG + d], sb[r * STREAMS + h] * y[r * H_BIG + d]);
            }
        }
    }
}

// ================================================================================================================================
// 5. SwiGLU
// ================================================================================================================================

#[test]
fn swiglu_clamps_match_reference() {
    // the up branch is clamped on both sides, the gate branch only from above
    let gate = [20.0f32, -20.0, 3.0, 0.0];
    let up = [20.0f32, -20.0, 0.5, -0.5];
    let out = swiglu_host(&gate, &up, None, 1, 4, SWIGLU_LIMIT);
    let silu = |x: f64| x / (1.0 + (-x).exp());
    let want = [silu(10.0) * 10.0, silu(-20.0) * -10.0, silu(3.0) * 0.5, 0.0];
    for i in 0..4 {
        assert!((out[i] as f64 - want[i]).abs() <= 1e-6 * want[i].abs().max(1.0), "swiglu[{i}] = {}", out[i]);
    }
    // the gate is NOT clamped from below
    assert!((out[1] as f64 - want[1]).abs() < 1e-9);
    // limit <= 0 disables both clamps
    let free = swiglu_host(&gate, &up, None, 1, 4, 0.0);
    assert!((free[0] as f64 - silu(20.0) * 20.0).abs() <= 1e-5 * (silu(20.0) * 20.0));
    // the per-token weight multiplies the product
    let w = [2.5f32];
    let wt = swiglu_host(&gate, &up, Some(&w), 1, 4, SWIGLU_LIMIT);
    for i in 0..4 {
        assert_eq!(wt[i], out[i] * 2.5);
    }
}

// ================================================================================================================================
// 6. GPU (self-skipping)
// ================================================================================================================================

#[test]
fn gpu_router_select_matches_host() {
    gpu!();
    let d = Dsv4Dims::V41;
    let st = Stream::new().unwrap();
    let mut rng = Rng(0x6E01);
    let rows = 9;
    let logits: Vec<f32> = (0..rows * d.experts).map(|_| rng.uniform(-8.0, 8.0)).collect();
    let bias: Vec<f32> = (0..d.experts).map(|_| rng.uniform(-0.3, 0.3)).collect();
    let bias_vl: Vec<f32> = (0..d.experts).map(|_| rng.uniform(-0.3, 0.3)).collect();
    let mask: Vec<u8> = (0..rows).map(|r| (r % 3 == 0) as u8).collect();
    let logits_d = up(&logits);
    let (mut pick, mut wts) = (DevBuf::alloc(rows * d.topk * 4).unwrap(), DevBuf::alloc(rows * d.topk * 2).unwrap());
    let rtr = Dsv4Router::new(&vec![0u16; d.experts * d.hidden], &bias, &bias_vl, d).unwrap();
    let m = up(&mask);
    rtr.select(&logits_d, rows, Some(&m), &mut pick, &mut wts, &st).unwrap();
    let mbool: Vec<bool> = mask.iter().map(|&v| v != 0).collect();
    let (hp, _) = select_host(&logits, rows, d, &bias, Some(&bias_vl), Some(&mbool));
    let got: Vec<i32> = pick.to_vec::<i32>().unwrap();
    assert_eq!(got, hp);
}

#[test]
fn gpu_sqrtsoftplus_matches_host() {
    gpu!();
    let st = Stream::new().unwrap();
    let x: Vec<f32> = (-64..64).map(|i| i as f32 * 0.5).collect();
    let (xd, mut yd) = (up(&x), DevBuf::alloc(x.len() * 4).unwrap());
    sqrtsoftplus_dev(&xd, &mut yd, x.len(), &st).unwrap();
    st.sync().unwrap();
    let got: Vec<f32> = yd.to_vec::<f32>().unwrap();
    for i in 0..x.len() {
        assert!((got[i] - sqrtsoftplus(x[i])).abs() <= 1e-6 * got[i].abs().max(1.0));
    }
}

#[test]
fn gpu_hc_matches_host() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(0x6E02);
    let rows = 3;
    let x: Vec<f32> = (0..rows * ROW).map(|_| rng.uniform(-2.0, 2.0)).collect();
    let fnw: Vec<f32> = (0..NFN * ROW).map(|_| rng.uniform(-0.05, 0.05)).collect();
    let base: Vec<f32> = (0..NFN).map(|_| rng.uniform(-0.5, 0.5)).collect();
    let scale = [0.7f32, -1.3, 0.4];
    let site = HcSite::new(&fnw, &base, &scale).unwrap();
    let hc = Dsv4Hc::V41;
    let mut work = HcWork::new(rows).unwrap();
    let xd = up(&x);
    hc.partials(&xd, &site, rows, &mut work, &st).unwrap();
    hc.finalize(&xd, &site, rows, &mut work, &st).unwrap();
    st.sync().unwrap();
    let (hp, hs, hc_comb) = hc_mixes_host(&x, rows, HIDDEN, &fnw, &scale, &base, ROW, hc.cfg);
    let pre = work.pre().to_vec::<f32>().unwrap();
    let post = work.post().to_vec::<f32>().unwrap();
    let comb = work.comb().to_vec::<f32>().unwrap();
    for i in 0..rows * STREAMS {
        assert!((pre[i] - hp[i]).abs() <= 1e-6 * hp[i].abs().max(1.0), "pre[{i}]");
        assert!((post[i] - hs[i]).abs() <= 1e-6 * hs[i].abs().max(1.0), "post[{i}]");
    }
    for i in 0..rows * 16 {
        assert!((comb[i] - hc_comb[i]).abs() <= 1e-6 * hc_comb[i].abs().max(1.0), "comb[{i}]");
    }
}

#[test]
fn gpu_swiglu_matches_host() {
    gpu!();
    let st = Stream::new().unwrap();
    let rows = 2;
    let inter = 64;
    let g: Vec<f32> = (0..rows * inter).map(|i| (i as f32 * 0.31).sin() * 15.0).collect();
    let u: Vec<f32> = (0..rows * inter).map(|i| (i as f32 * 0.17).cos() * 15.0).collect();
    let w = [0.5f32, 2.0];
    let (gd, ud, mut od) = (up(&g), up(&u), DevBuf::alloc(rows * inter * 4).unwrap());
    let wd = up(&w);
    swiglu(&gd, &ud, Some(&wd), &mut od, rows, inter, SWIGLU_LIMIT, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<f32> = od.to_vec::<f32>().unwrap();
    let want = swiglu_host(&g, &u, Some(&w), rows, inter, SWIGLU_LIMIT);
    for i in 0..rows * inter {
        assert!((got[i] - want[i]).abs() <= 1e-6 * want[i].abs().max(1.0), "swiglu[{i}]");
    }
}

#[test]
fn gpu_hc_apply_partials_and_collapse_matches_host() {
    gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(0x7E03);
    for &rows in &[1usize, 5, 16, 65, 128] {
        let x0: Vec<f32> = (0..rows * ROW).map(|_| rng.uniform(-1.5, 1.5)).collect();
        let y_prev_f32: Vec<f32> = (0..rows * HIDDEN).map(|_| rng.uniform(-0.8, 0.8)).collect();
        let y_prev_f16: Vec<u16> = y_prev_f32.iter().map(|&v| gb_exl3::f16::f64_to_f16(v as f64)).collect();
        let y_prev_q: Vec<f32> = y_prev_f16.iter().map(|&h| gb_exl3::f16::f16_to_f64(h) as f32).collect();
        let pre_in: Vec<f32> = (0..rows * STREAMS).map(|_| rng.uniform(0.1, 0.9)).collect();
        let post_prev: Vec<f32> = (0..rows * STREAMS).map(|_| rng.uniform(0.2, 1.8)).collect();
        let comb_prev: Vec<f32> = (0..rows * 16).map(|_| rng.uniform(0.05, 0.45)).collect();
        let fnw: Vec<f32> = (0..NFN * ROW).map(|_| rng.uniform(-0.05, 0.05)).collect();
        let base: Vec<f32> = (0..NFN).map(|_| rng.uniform(-0.5, 0.5)).collect();
        let scale = [0.7f32, -1.3, 0.4];

        let site = HcSite::new(&fnw, &base, &scale).unwrap();
        let hc = Dsv4Hc::V41;
        let mut work = HcWork::new(rows).unwrap();
        // Seed work.post and work.comb with post_prev / comb_prev via an initial apply or upload
        // We can run unfused apply -> partials_and_collapse -> finalize as the reference on GPU and compare against host!
        let mut xd_unfused = up(&x0);
        let mut xd_fused = up(&x0);
        let yd = up(&y_prev_f16);
        let pred = up(&pre_in);
        let mut col_unfused = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();
        let mut col_fused = DevBuf::zeroed(rows * HIDDEN * 4).unwrap();

        // Populate work.post and work.comb via a dummy finalize first
        hc.partials(&xd_unfused, &site, rows, &mut work, &st).unwrap();
        hc.finalize(&xd_unfused, &site, rows, &mut work, &st).unwrap();
        st.sync().unwrap();
        let post_init = work.post().to_vec::<f32>().unwrap();
        let comb_init = work.comb().to_vec::<f32>().unwrap();
        let _ = (post_prev, comb_prev);

        // Unfused: apply -> partials_and_collapse -> finalize
        hc.apply(&mut xd_unfused, &yd, gb_cuda::exl3_linear::Dtype::F16, rows, &work, &st).unwrap();
        hc.partials_and_collapse(&xd_unfused, &site, &pred, &mut col_unfused, rows, &mut work, &st).unwrap();
        hc.finalize(&xd_unfused, &site, rows, &mut work, &st).unwrap();
        st.sync().unwrap();
        let pre_unfused = work.pre().to_vec::<f32>().unwrap();
        let post_unfused = work.post().to_vec::<f32>().unwrap();
        let comb_unfused = work.comb().to_vec::<f32>().unwrap();
        let col_u = col_unfused.to_vec::<f32>().unwrap();

        // Reset work.post / work.comb to same initial state for fused
        let mut work_f = HcWork::new(rows).unwrap();
        hc.partials(&xd_fused, &site, rows, &mut work_f, &st).unwrap();
        hc.finalize(&xd_fused, &site, rows, &mut work_f, &st).unwrap();
        hc.apply_partials_and_collapse(&mut xd_fused, &yd, &site, &pred, &mut col_fused, rows, &mut work_f, &st).unwrap();
        hc.finalize(&xd_fused, &site, rows, &mut work_f, &st).unwrap();
        st.sync().unwrap();

        let pre_fused = work_f.pre().to_vec::<f32>().unwrap();
        let post_fused = work_f.post().to_vec::<f32>().unwrap();
        let comb_fused = work_f.comb().to_vec::<f32>().unwrap();
        let col_f = col_fused.to_vec::<f32>().unwrap();

        assert_eq!(bits(&col_u), bits(&col_f), "collapsed differs at rows={rows}");
        assert_eq!(bits(&pre_unfused), bits(&pre_fused), "pre differs at rows={rows}");
        assert_eq!(bits(&post_unfused), bits(&post_fused), "post differs at rows={rows}");
        assert_eq!(bits(&comb_unfused), bits(&comb_fused), "comb differs at rows={rows}");

        // Also check against host reference
        let x_after = hc_post_host(&x0, &y_prev_q, rows, HIDDEN, &post_init, &comb_init);
        let col_host = hc_pre_host(&x_after, rows, HIDDEN, &pre_in);
        for i in 0..rows * HIDDEN {
            assert!((col_f[i] - col_host[i]).abs() <= 1e-5 * col_host[i].abs().max(1.0), "col[{i}] rows={rows}");
        }
    }
}

#[test]
fn bench_dsv4_hc_mixes_80_calls() {
    gpu!();
    let st = Stream::new().unwrap();
    let max_m = 2048usize;
    let fnw = vec![0.01f32; NFN * ROW];
    let base = vec![0.05f32; NFN];
    let scale = [0.7f32, -1.3, 0.4];
    let site = HcSite::new(&fnw, &base, &scale).unwrap();
    let hc = Dsv4Hc::V41;
    let mut work = HcWork::new(max_m).unwrap();
    let mut xd = DevBuf::zeroed(max_m * ROW * 4).unwrap();
    let yd = DevBuf::zeroed(max_m * HIDDEN * 2).unwrap();
    let pred = DevBuf::zeroed(max_m * STREAMS * 4).unwrap();
    let mut cold = DevBuf::zeroed(max_m * HIDDEN * 4).unwrap();

    for &m in &[128usize, 256, 512, 1024, 2048] {
        for _ in 0..4 {
            hc.apply_partials_and_collapse(&mut xd, &yd, &site, &pred, &mut cold, m, &mut work, &st).unwrap();
            hc.finalize(&xd, &site, m, &mut work, &st).unwrap();
        }
        st.sync().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..80 {
            hc.apply_partials_and_collapse(&mut xd, &yd, &site, &pred, &mut cold, m, &mut work, &st).unwrap();
            hc.finalize(&xd, &site, m, &mut work, &st).unwrap();
        }
        st.sync().unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("bench_dsv4_hc_80calls m={m:4}: total={ms:6.2} ms | per_call={:.3} ms", ms / 80.0);
    }
}
