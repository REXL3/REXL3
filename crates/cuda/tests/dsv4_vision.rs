//! DeepSeek-V4.1-Flash vision tower (`src/dsv4_vision.rs`, `csrc/dsv4_vision.cu`): the 588 flattening order, one block
//! against a direct transcription of `vision.py`, the aligner's `(C, kH, kW)` window order, and the image-span splice.
//!
//! # Why a transcription and not the module
//!
//! `reference_block` and `reference_aligner` below are written from `/tmp/ds41ref/vision.py` (`Attention.forward`,
//! `MLP.forward`, `Block.forward`, `Aligner.forward`) in **f64**, with their own RoPE, their own softmax and their own
//! `erf`, and they call nothing in the module under test. A mistake in the module cannot hide in them.
//!
//! # What runs without a GPU
//!
//! Everything except `gpu_*`: those print `not run: no device` and return, so this binary is green on a CPU-only box.
use gb_cuda::dsv4_vision::*;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::f64_to_f16;

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
    /// A uniform f64 in `[lo, hi)`.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u
    }
    fn vec(&mut self, n: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..n).map(|_| self.uniform(lo, hi)).collect()
    }
    fn f32(&mut self, n: usize, lo: f64, hi: f64) -> Vec<f32> {
        self.vec(n, lo, hi).into_iter().map(|v| v as f32).collect()
    }
}

/// `erf` to f64: a Taylor series inside `|x| < 2.5` and the Laplace continued fraction beyond it, so the transcription is
/// accurate to ~1e-15 and its own error can never be mistaken for the module's.
fn erf(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 1e-12 {
        return 0.0;
    }
    if ax < 2.5 {
        let x2 = x * x;
        let mut term = x; // (-1)^n x^(2n+1) / n!
        let mut sum = x;
        for n in 1..200 {
            term *= -x2 / n as f64;
            let add = term / (2 * n + 1) as f64;
            sum += add;
            if add.abs() < 1e-18 * sum.abs().max(1e-300) {
                break;
            }
        }
        return 2.0 / std::f64::consts::PI.sqrt() * sum;
    }
    // erfc(x) = exp(-x^2) / (sqrt(pi) * (x + (1/2)/(x + 1/(x + (3/2)/(x + ...)))))
    let mut t = 0.0;
    for n in (1..=200).rev() {
        t = (n as f64 / 2.0) / (ax + t);
    }
    let erfc = (-ax * ax).exp() / (std::f64::consts::PI.sqrt() * (ax + t));
    x.signum() * (1.0 - erfc)
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x / 2f64.sqrt()))
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

// ------------------------------------------------------------------------------------------------ the transcription

/// `RMSNorm.forward`: f32 mean and rsqrt, then the weight.
fn ref_rmsnorm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let d = w.len();
    x.chunks_exact(d)
        .flat_map(|r| {
            let ms = r.iter().map(|v| v * v).sum::<f64>() / d as f64;
            let s = 1.0 / (ms + eps).sqrt();
            (0..d).map(move |i| w[i] * (r[i] * s))
        })
        .collect()
}

/// `nn.Linear`: `x @ w^T + b`, `w` row-major `[n, k]`.
fn ref_linear(x: &[f64], rows: usize, k: usize, w: &[f64], b: &[f64]) -> Vec<f64> {
    let n = if b.is_empty() { w.len() / k } else { b.len() };
    let mut out = vec![0f64; rows * n];
    for r in 0..rows {
        for o in 0..n {
            let mut acc = 0f64;
            for i in 0..k {
                acc += x[r * k + i] * w[o * k + i];
            }
            out[r * n + o] = acc + if b.is_empty() { 0.0 } else { b[o] };
        }
    }
    out
}

/// `get_vision_cos_sin` + `apply_rotary`, from the formula: `dim` frequencies over `theta`, the row angle in the first
/// half of the angle vector and the column angle in the second, each pair applied to the two halves of a head.
fn ref_rope(x: &mut [f64], rows: usize, n_w: usize, heads: usize, hd: usize, rope: usize, theta: f64) {
    let half = rope / 2;
    for r in 0..rows {
        let (hpos, wpos) = ((r / n_w) as f64, (r % n_w) as f64);
        for h in 0..heads {
            for j in 0..rope {
                let (pos, f) = if j < half { (hpos, j) } else { (wpos, j - half) };
                let ang = pos / theta.powf((2 * f) as f64 / rope as f64);
                let (c, s) = (ang.cos(), ang.sin());
                let base = (r * heads + h) * hd;
                let x1 = x[base + j];
                let x2 = x[base + rope + j];
                x[base + j] = x1 * c - x2 * s;
                x[base + rope + j] = x2 * c + x1 * s;
            }
        }
    }
}

struct RefBlock {
    norm1: Vec<f64>,
    norm2: Vec<f64>,
    qkv_w: Vec<f64>,
    qkv_b: Vec<f64>,
    wo_w: Vec<f64>,
    wo_b: Vec<f64>,
    w1: Vec<f64>,
    w2: Vec<f64>,
}

/// `Block.forward` in f64: `x + attn(norm1(x))`, then `x + mlp(norm2(x))`, bidirectional over the `p` patches.
fn reference_block(x: &[f64], p: usize, n_w: usize, w: &RefBlock, eps: f64) -> Vec<f64> {
    let (d, heads, hd, inter) = (HIDDEN, HEADS, HEAD_DIM, INTER);
    let rope = ROPE_DIM;

    let h = ref_rmsnorm(x, &w.norm1, eps);
    let qkv = ref_linear(&h, p, d, &w.qkv_w, &w.qkv_b);
    let mut q = vec![0f64; p * d];
    let mut k = vec![0f64; p * d];
    let mut v = vec![0f64; p * d];
    for r in 0..p {
        q[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d..r * 3 * d + d]);
        k[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d + d..r * 3 * d + 2 * d]);
        v[r * d..(r + 1) * d].copy_from_slice(&qkv[r * 3 * d + 2 * d..(r + 1) * 3 * d]);
    }
    ref_rope(&mut q, p, n_w, heads, hd, rope, ROPE_THETA as f64);
    ref_rope(&mut k, p, n_w, heads, hd, rope, ROPE_THETA as f64);

    let scale = 1.0 / (hd as f64).sqrt();
    let mut att = vec![0f64; p * d];
    for hh in 0..heads {
        for i in 0..p {
            let mut sc = vec![0f64; p];
            for j in 0..p {
                let mut acc = 0f64;
                for c in 0..hd {
                    acc += q[(i * heads + hh) * hd + c] * k[(j * heads + hh) * hd + c];
                }
                sc[j] = acc * scale;
            }
            let m = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
            let sum: f64 = ex.iter().sum();
            for j in 0..p {
                let a = ex[j] / sum;
                for c in 0..hd {
                    att[(i * heads + hh) * hd + c] += a * v[(j * heads + hh) * hd + c];
                }
            }
        }
    }
    let o = ref_linear(&att, p, d, &w.wo_w, &w.wo_b);
    let mut x1: Vec<f64> = x.iter().zip(&o).map(|(a, b)| a + b).collect();

    let h2 = ref_rmsnorm(&x1, &w.norm2, eps);
    let gu = ref_linear(&h2, p, d, &w.w1, &Vec::new());
    let act: Vec<f64> = (0..p)
        .flat_map(|r| (0..inter).map(move |i| (r, i)))
        .map(|(r, i)| silu(gu[r * 2 * inter + i]) * gu[r * 2 * inter + inter + i])
        .collect();
    let down = ref_linear(&act, p, inter, &w.w2, &Vec::new());
    for i in 0..x1.len() {
        x1[i] += down[i];
    }
    x1
}

/// `Aligner.forward`: `F.pad`, `F.unfold(3, stride=3)`, `w2(gelu(w1(x)))`.
fn reference_aligner(x: &[f64], n_h: usize, n_w: usize, w1: &[f64], b1: &[f64], w2: &[f64], b2: &[f64]) -> Vec<f64> {
    let r = MERGE;
    let oh = n_h.div_ceil(r);
    let ow = n_w.div_ceil(r);
    let mut rows = vec![0f64; oh * ow * ALIGN_IN];
    for oy in 0..oh {
        for ox in 0..ow {
            for c in 0..HIDDEN {
                for kh in 0..r {
                    for kw in 0..r {
                        let (y, xx) = (oy * r + kh, ox * r + kw);
                        // the pad is on the right and the bottom, with zeros
                        if y < n_h && xx < n_w {
                            rows[(oy * ow + ox) * ALIGN_IN + c * r * r + kh * r + kw] = x[(y * n_w + xx) * HIDDEN + c];
                        }
                    }
                }
            }
        }
    }
    let h = ref_linear(&rows, oh * ow, ALIGN_IN, w1, b1);
    let g: Vec<f64> = h.iter().map(|&v| gelu(v)).collect();
    ref_linear(&g, oh * ow, OUT_HIDDEN, w2, b2)
}

// ------------------------------------------------------------------------------------------------ tests

/// **The 588.** `PatchEmbed.forward` is `self.proj(x.flatten(1))` on `[P, 3, 14, 14]`, so a row is channel-major then
/// row then column. This pins that against the reference's own `reshape`/`permute`, and against the two orders it is *not*
/// (patch-major, and channel-innermost) so that a transposition cannot pass.
#[test]
fn the_588_dim_patch_embed_matches_the_references_flattening_order() {
    let (n_h, n_w) = (2, 3);
    let (ph, pw) = (n_h * PATCH, n_w * PATCH);
    assert_eq!(PATCH_K, 3 * PATCH * PATCH);

    // every pixel is its own coordinate: c * 100000 + y * 1000 + x
    let mut pix = vec![0f32; 3 * ph * pw];
    for c in 0..3 {
        for y in 0..ph {
            for x in 0..pw {
                pix[c * ph * pw + y * pw + x] = (c * 100_000 + y * 1000 + x) as f32;
            }
        }
    }
    let mut got = vec![0f32; n_h * n_w * PATCH_K];
    patchify(&pix, n_h, n_w, &mut got);

    // the reference: x.reshape(3, n_h, p, n_w, p).permute(1, 3, 0, 2, 4).reshape(-1, 3, p, p), then flatten(1)
    for a in 0..n_h {
        for b in 0..n_w {
            for c in 0..3 {
                for i in 0..PATCH {
                    for j in 0..PATCH {
                        let y = a * PATCH + i;
                        let x = b * PATCH + j;
                        let want = pix[c * ph * pw + y * pw + x];
                        let at = (a * n_w + b) * PATCH_K + c * PATCH * PATCH + i * PATCH + j;
                        assert_eq!(got[at], want, "patch ({a},{b}) c={c} i={i} j={j}");
                        assert_eq!(flatten_index(c, i, j), c * PATCH * PATCH + i * PATCH + j);
                    }
                }
            }
        }
    }

    // the two wrong orders, made explicit on the first patch
    let p0 = &got[..PATCH_K];
    assert_eq!(p0[0], 0.0, "c=0, top-left");
    assert_eq!(p0[1], 1.0, "one column right is +1 (not one channel down)");
    assert_eq!(p0[PATCH], 1000.0, "one row down is y+1 in the encoded value");
    assert_eq!(p0[PATCH * PATCH], 100_000.0, "channel 1 starts at 196, so c is outermost");
    assert_eq!(p0[196 + 1], 100_001.0);

    // and the second patch starts where the first ends, in row-major over the grid
    assert_eq!(got[PATCH_K], pix[PATCH], "patch (0,1) starts 14 columns in");
}

/// One block, the real dims, against the f64 transcription. `block_host` is the module's f32 twin of the kernels; the two
/// share no code, so a wrong rope axis, a dropped bias, a causal mask or a wrong MLP split all show up here.
#[test]
fn one_block_matches_a_direct_transcription_of_vision_py() {
    let (n_h, n_w) = (2, 2);
    let p = n_h * n_w;
    let mut rng = Rng(0x5eed_1234_abcd_0001);

    // The weights are scaled so that *every* sublayer moves the stream by O(1): a bound relative to an O(1) value is then
    // tight enough that a swapped gate/up split, a dropped bias or a missing rope all fail, and loose enough that f32 and
    // f64 agree (they differ by ~1e-6 here).
    let eps = RMS_EPS as f64;
    let scale = 0.03;
    let r = |rng: &mut Rng, n: usize, s: f64| rng.vec(n, -s, s);
    let w = RefBlock {
        norm1: rng.vec(HIDDEN, 0.5, 1.5),
        norm2: rng.vec(HIDDEN, 0.5, 1.5),
        qkv_w: r(&mut rng, 3 * HIDDEN * HIDDEN, scale),
        qkv_b: r(&mut rng, 3 * HIDDEN, 0.02),
        wo_w: r(&mut rng, HIDDEN * HIDDEN, scale),
        wo_b: r(&mut rng, HIDDEN, 0.02),
        w1: r(&mut rng, 2 * INTER * HIDDEN, scale),
        w2: r(&mut rng, HIDDEN * INTER, scale),
    };
    let x: Vec<f64> = rng.vec(p * HIDDEN, -1.0, 1.0);

    let want = reference_block(&x, p, n_w, &w, eps);

    let to32 = |v: &[f64]| -> Vec<f32> { v.iter().map(|&a| a as f32).collect() };
    let got = block_host(
        &to32(&x),
        p,
        n_w,
        &BlockHostW {
            norm1: &to32(&w.norm1),
            norm2: &to32(&w.norm2),
            qkv_w: &to32(&w.qkv_w),
            qkv_b: &to32(&w.qkv_b),
            wo_w: &to32(&w.wo_w),
            wo_b: &to32(&w.wo_b),
            w1: &to32(&w.w1),
            w2: &to32(&w.w2),
        },
        RMS_EPS,
    );

    let mut worst = 0f64;
    for (a, b) in got.iter().zip(&want) {
        let d = (*a as f64 - b).abs();
        worst = worst.max(d / b.abs().max(1.0));
    }
    assert!(worst < 1e-3, "block_host differs from the transcription by {worst:e}");
    // and the block is not the identity: it moved the stream
    let moved = got.iter().zip(&x).map(|(a, b)| (*a as f64 - b).abs()).fold(0f64, f64::max);
    assert!(moved > 1e-3, "the block did nothing");
}

/// The aligner's 9216 is `F.unfold`'s `(C, kH, kW)` window with `C` outermost — **not** the patch-major order a 2 x 2
/// merge would give, which is the mistake this pins.
#[test]
fn the_aligner_window_is_channel_major() {
    let (n_h, n_w) = (3, 3); // one output token, no padding
    let mut x = vec![0f32; n_h * n_w * HIDDEN];
    for y in 0..n_h {
        for xx in 0..n_w {
            for c in 0..HIDDEN {
                x[(y * n_w + xx) * HIDDEN + c] = (y * 9 + xx) as f32 * 10_000.0 + c as f32;
            }
        }
    }
    // the gather the kernel runs, reproduced on the host
    let mut got = vec![0f32; ALIGN_IN];
    for c in 0..HIDDEN {
        for kh in 0..MERGE {
            for kw in 0..MERGE {
                got[aligner_index(c, kh, kw)] = x[(kh * n_w + kw) * HIDDEN + c];
            }
        }
    }
    assert_eq!(got[0], 0.0, "c=0, window top-left");
    assert_eq!(got[1], 10_000.0, "the window moves before the channel does");
    assert_eq!(got[9], 1.0, "channel 1 is 9 values in, not 1");
    assert_eq!(got[ALIGN_IN - 1], 20.0 * 10_000.0 + 1023.0, "c=1023, window (2,2) is patch (2,2)");

    // and the aligner itself, against the transcription, on a 3 x 3 grid (one token)
    let mut rng = Rng(0x51ee_d000_0000_0002);
    let s = 0.02 / (ALIGN_IN as f64).sqrt();
    let w1 = rng.vec(OUT_HIDDEN * ALIGN_IN, -s, s);
    let b1 = rng.vec(OUT_HIDDEN, -0.01, 0.01);
    let w2 = rng.vec(OUT_HIDDEN * OUT_HIDDEN, 0.02 / (OUT_HIDDEN as f64).sqrt(), 0.02 / (OUT_HIDDEN as f64).sqrt());
    let b2 = rng.vec(OUT_HIDDEN, -0.01, 0.01);
    let xf: Vec<f64> = x.iter().map(|&v| (v / 10_000.0) as f64).collect();
    let want = reference_aligner(&xf, n_h, n_w, &w1, &b1, &w2, &b2);
    let got = aligner_host(
        &xf.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        n_h,
        n_w,
        &w1.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        &b1.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        &w2.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        &b2.iter().map(|&v| v as f32).collect::<Vec<_>>(),
    );
    let mut worst = 0f64;
    for (a, b) in got.iter().zip(&want) {
        worst = worst.max((*a as f64 - b).abs() / b.abs().max(1.0));
    }
    assert!(worst < 1e-3, "aligner_host differs from the transcription by {worst:e}");
}

/// The span layout and the engram mask it implies: `[START] ([IMAGE]*w [NEWLINE])*h [END]`, every position of which is
/// `false` in the mask, so an n-gram never crosses an image — and `None` for a text-only prompt.
#[test]
fn an_image_span_is_the_processor_layout_and_its_engram_mask() {
    let (n_h, n_w) = (6, 9); // 2 x 3 llm grid
    assert_eq!((llm_grid(n_h), llm_grid(n_w)), (2, 3));
    let types = token_types(2, 3);
    assert_eq!(types.len(), num_image_tokens(2, 3));
    assert_eq!(types, vec![0, 1, 1, 1, 2, 1, 1, 1, 2, 3]);

    // text only: no token_types at all, so no mask — the fast path
    assert_eq!(engram_mask(None), None);

    let mut all = vec![TokenType::Text as i8; 3];
    all.extend(&types);
    all.extend([TokenType::Text as i8; 2]);
    let m = engram_mask(Some(&all)).unwrap();
    assert_eq!(m[..3], [true, true, true]);
    assert!(m[3..3 + types.len()].iter().all(|&b| !b));
    assert_eq!(m[3 + types.len()..], [true, true]);
    assert_eq!(image_mask(&all), m.iter().map(|b| !b).collect::<Vec<_>>());

    // the splice order the model's `splice_into` and the kernel both walk
    let order = splice_order(2, 3);
    assert_eq!(order.len(), types.len());
    assert_eq!(order[0], (0, SpliceSrc::Start));
    assert_eq!(order[order.len() - 1], (types.len() - 1, SpliceSrc::End));
    let rows: Vec<usize> =
        order.iter().filter_map(|&(_, s)| if let SpliceSrc::Aligner(r) = s { Some(r) } else { None }).collect();
    assert_eq!(rows, (0..6).collect::<Vec<_>>(), "the aligner's rows in reading order");
}

// ------------------------------------------------------------------------------------------------ device

/// The 588-K GEMM is the one shape in the tower that is not a multiple of 8, so it is the one worth running first: the
/// kernel predicates its last K tile, and a bug there is a wrong patch embedding with no other symptom.
#[test]
fn gpu_patch_embed_matches_the_host_linear() {
    if !device_available() {
        println!("not run: no device");
        return;
    }
    let (n_h, n_w) = (2, 2);
    let p = n_h * n_w;
    let mut rng = Rng(0x600d_0000_0000_0003);
    let pix = rng.f32(3 * (n_h * PATCH) * (n_w * PATCH), -1.0, 1.0);
    let mut patches = vec![0f32; p * PATCH_K];
    patchify(&pix, n_h, n_w, &mut patches);

    let wf = rng.f32(HIDDEN * PATCH_K, -0.02, 0.02);
    let bf = rng.f32(HIDDEN, -0.01, 0.01);
    let want = linear_host(&patches, p, PATCH_K, &wf, &bf);

    let st = Stream::new().unwrap();
    let a = DevBuf::from_slice(&patches.iter().map(|&v| f64_to_f16(v as f64)).collect::<Vec<u16>>()).unwrap();
    let w = DevBuf::from_slice(&wf.iter().map(|&v| f64_to_f16(v as f64)).collect::<Vec<u16>>()).unwrap();
    let b = DevBuf::from_slice(&bf.iter().map(|&v| f64_to_f16(v as f64)).collect::<Vec<u16>>()).unwrap();
    let mut c = DevBuf::zeroed(p * HIDDEN * 4).unwrap();
    gemm(Epi::StoreF32, &a, &w, Some(&b), &mut c, 0, p, HIDDEN, PATCH_K, PATCH_K, HIDDEN, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<f32> = c.to_vec().unwrap();

    let mut worst = 0f64;
    for (g, e) in got.iter().zip(&want) {
        worst = worst.max((*g as f64 - *e as f64).abs() / (*e as f64).abs().max(1.0));
    }
    // fp16 operands: the honest bound is the operand rounding, not f32
    assert!(worst < 5e-3, "patch-embed GEMM differs from the host by {worst:e}");
}

/// A multi-tile GEMM with `m != n`: 4 row tiles by 5 column tiles, so most tiles sit off the grid diagonal — the shape
/// every tower GEMM has (1515 patches against 1024/3072/9216-wide weights) and the one a single-row-tile probe cannot
/// see. The B tile must come from the *column* block's rows of `b`; loading it from the row block's rows reads valid but
/// unrelated weight rows, which is finite, deterministic garbage — exactly the failure that blanks image replies while
/// text serves fine.
#[test]
fn gpu_gemm_matches_the_host_linear_when_both_indices_span_multiple_tiles() {
    if !device_available() {
        println!("not run: no device");
        return;
    }
    let (m, n, k) = (200, 300, 80); // 3.125 row tiles, 4.7 column tiles, 5 K tiles
    let mut rng = Rng(0x600d_0000_0000_0005);
    let af = rng.f32(m * k, -1.0, 1.0);
    let wf = rng.f32(n * k, -0.05, 0.05);
    let bf = rng.f32(n, -0.01, 0.01);
    let want = linear_host(&af, m, k, &wf, &bf);

    let st = Stream::new().unwrap();
    let h16 = |v: &[f32]| -> Vec<u16> { v.iter().map(|&x| f64_to_f16(x as f64)).collect() };
    let a = DevBuf::from_slice(&h16(&af)).unwrap();
    let w = DevBuf::from_slice(&h16(&wf)).unwrap();
    let b = DevBuf::from_slice(&h16(&bf)).unwrap();
    let mut c = DevBuf::zeroed(m * n * 4).unwrap();
    gemm(Epi::StoreF32, &a, &w, Some(&b), &mut c, 0, m, n, k, k, n, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<f32> = c.to_vec().unwrap();

    let mut worst = 0f64;
    for (g, e) in got.iter().zip(&want) {
        worst = worst.max((*g as f64 - *e as f64).abs() / (*e as f64).abs().max(1.0));
    }
    assert!(worst < 5e-3, "multi-tile GEMM differs from the host by {worst:e}");
}

/// `rmsnorm` against `rmsnorm_host` with a realistic gamma: the kernel's block reduction must sum EVERY lane's
/// partial (a dropped half is a clean sqrt(2) scale error that end-to-end block tests can hide inside small sublayer
/// outputs — exactly how the `=`-for-`+=` in the first shuffle shipped).
#[test]
fn gpu_rmsnorm_matches_the_host_summing_every_lane() {
    if !device_available() {
        println!("not run: no device");
        return;
    }
    let rows = 130; // more than one tile of nothing in particular — the reduction is per row
    let mut rng = Rng(0x600d_0000_0000_0006);
    let x = rng.f32(rows * HIDDEN, -3.0, 3.0);
    let w = rng.f32(HIDDEN, 0.05, 1.2); // f32 gamma, as the bind widens it
    let want = rmsnorm_host(&x, &w, RMS_EPS);
    let st = Stream::new().unwrap();
    let xd = DevBuf::from_slice(&x).unwrap();
    let wd = DevBuf::from_slice(&w).unwrap();
    let mut y = DevBuf::zeroed(rows * HIDDEN * 2).unwrap();
    rmsnorm(&xd, &wd, &mut y, rows, HIDDEN, RMS_EPS, &st).unwrap();
    st.sync().unwrap();
    let raw: Vec<u16> = y.to_vec().unwrap();
    let got: Vec<f32> = raw.iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect();
    let mut worst = 0f64;
    for (g, e) in got.iter().zip(&want) {
        worst = worst.max((*g as f64 - *e as f64).abs() / (*e as f64).abs().max(1.0));
    }
    assert!(worst < 2e-3, "rmsnorm differs from the host by {worst:e} (a ~0.41 rel error is the dropped-half signature)");
}

/// The block's kernel sequence on the device against the module's f32 host twin (`block_host`): every kernel of
/// `Block.forward` at once, so a wrong K offset in attention, a missing `lda` on the down projection or a norm writing
/// the wrong dtype cannot hide behind "finite and deterministic" — the state this test shipped in, unrun, while image
/// replies came back blank. Also deterministic across runs (one image, fixed kernels, one answer).
#[test]
fn gpu_block_pipeline_matches_the_host_twin_and_is_deterministic() {
    if !device_available() {
        println!("not run: no device");
        return;
    }
    let p = 70; // a 7 x 10 patch grid: two row tiles in every GEMM, off-diagonal column tiles everywhere
    let n_w = 10;
    let mut rng = Rng(0x600d_0000_0000_0004);
    let s = 0.05 / (HIDDEN as f64).sqrt();
    // the values first, so the host twin sees exactly what the device sees (f16-rounded GEMM operands, f32 norms)
    let val = |rng: &mut Rng, n: usize, lo: f64, hi: f64| rng.f32(n, lo, hi);
    let norm1 = val(&mut rng, HIDDEN, 0.5, 1.5);
    let norm2 = val(&mut rng, HIDDEN, 0.5, 1.5);
    let qkv_w = val(&mut rng, 3 * HIDDEN * HIDDEN, -s, s);
    let qkv_b = val(&mut rng, 3 * HIDDEN, -0.01, 0.01);
    let wo_w = val(&mut rng, HIDDEN * HIDDEN, -s, s);
    let wo_b = val(&mut rng, HIDDEN, -0.01, 0.01);
    let w1 = val(&mut rng, 2 * INTER * HIDDEN, -s, s);
    let w2 = val(&mut rng, HIDDEN * INTER, -s, s);
    let x0 = val(&mut rng, p * HIDDEN, -1.0, 1.0);

    let h16 = |v: &[f32]| -> Vec<u16> { v.iter().map(|&x| f64_to_f16(x as f64)).collect() };
    let narrowed = |v: &[f32]| -> Vec<f32> {
        let bits = h16(v);
        bits.iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect()
    };
    let want = block_host(
        &x0,
        p,
        n_w,
        &BlockHostW {
            norm1: &norm1,
            norm2: &norm2,
            qkv_w: &narrowed(&qkv_w),
            qkv_b: &narrowed(&qkv_b),
            wo_w: &narrowed(&wo_w),
            wo_b: &narrowed(&wo_b),
            w1: &narrowed(&w1),
            w2: &narrowed(&w2),
        },
        RMS_EPS,
    );

    let n1 = DevBuf::from_slice(&norm1).unwrap();
    let n2 = DevBuf::from_slice(&norm2).unwrap();
    let mk = |v: &[f32]| -> DevBuf { DevBuf::from_slice(&h16(v)).unwrap() };
    let (qkv_wd, qkv_bd) = (mk(&qkv_w), mk(&qkv_b));
    let (wo_wd, wo_bd) = (mk(&wo_w), mk(&wo_b));
    let (w1d, w2d) = (mk(&w1), mk(&w2));

    let run = || -> Vec<f32> {
        let st = Stream::new().unwrap();
        let mut x = DevBuf::zeroed(p * HIDDEN * 4).unwrap();
        x.upload(&x0).unwrap();
        let mut h = DevBuf::zeroed(p * HIDDEN * 2).unwrap();
        let mut qkv = DevBuf::zeroed(p * 3 * HIDDEN * 2).unwrap();
        let mut att = DevBuf::zeroed(p * HIDDEN * 2).unwrap();
        let mut gu = DevBuf::zeroed(p * MLP * 2).unwrap();
        let mut ang = DevBuf::zeroed(p * ROPE_DIM * 4).unwrap();
        rmsnorm(&x, &n1, &mut h, p, HIDDEN, RMS_EPS, &st).unwrap();
        gemm(Epi::BiasF16, &h, &qkv_wd, Some(&qkv_bd), &mut qkv, 0, p, 3 * HIDDEN, HIDDEN, HIDDEN, 3 * HIDDEN, &st).unwrap();
        rope_table(&mut ang, p / n_w, n_w, ROPE_DIM, ROPE_THETA, &st).unwrap();
        rope(&mut qkv, &ang, p, n_w, HEADS, &st).unwrap();
        attention(&qkv, &mut att, p, &st).unwrap();
        gemm(Epi::BiasAddF32, &att, &wo_wd, Some(&wo_bd), &mut x, 0, p, HIDDEN, HIDDEN, HIDDEN, HIDDEN, &st).unwrap();
        rmsnorm(&x, &n2, &mut h, p, HIDDEN, RMS_EPS, &st).unwrap();
        gemm(Epi::BiasF16, &h, &w1d, None, &mut gu, 0, p, MLP, HIDDEN, HIDDEN, MLP, &st).unwrap();
        silu_mul(&mut gu, INTER, MLP, p, &st).unwrap();
        // the activation is the first INTER columns of an MLP-strided row: lda = MLP
        gemm(Epi::BiasAddF32, &gu, &w2d, None, &mut x, 0, p, HIDDEN, INTER, MLP, HIDDEN, &st).unwrap();
        st.sync().unwrap();
        x.to_vec().unwrap()
    };

    let a = run();
    assert!(a.iter().all(|v| v.is_finite()), "the block produced a non-finite value");
    let mut worst = 0f64;
    for (g, e) in a.iter().zip(&want) {
        worst = worst.max((*g as f64 - *e as f64).abs() / (*e as f64).abs().max(1.0));
    }
    assert!(worst < 1e-2, "the block pipeline differs from the host twin by {worst:e}");
    let moved = a.iter().zip(&x0).map(|(u, v)| (u - v).abs()).fold(0f32, f32::max);
    assert!(moved > 1e-4, "the block did not move the stream");
    assert_eq!(a, run(), "the block is not deterministic");
}
