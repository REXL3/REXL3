//! The Ling-3.1-flash-1019 Gated MLA (`crates/cuda/src/mla_gated.rs`, `csrc/mla_gated.cu`) against a CPU
//! reference transcription of `BailingMoeV3MultiLatentAttention` (64 heads, `nope=128, rope=64, v=128,
//! latent=512`, full causal, interleaved RoPE with `theta=6e6`, the `576->640` padded `kv_a_proj_with_mqa` stride,
//! and the head-wise sigmoid `g_proj` gate), and the row-invariance contract across prefill/decode chunkings.
//!
//! Run on a GB10 box (see the spec's box rules):
//!   gpujob.sh cargo test --release -p gb-cuda --test mla_gated -- --nocapture --test-threads=1
#![allow(dead_code)]

mod common;

use common::Rng;
use gb_cuda::attention::{BlockTable, RowIndex};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::mla_gated::*;
use gb_cuda::rt::{DevBuf, Stream};

fn f16s(rng: &mut Rng, n: usize, scale: f32) -> Vec<u16> {
    (0..n).map(|_| gb_exl3::f16::f64_to_f16((rng.uniform(-1.0, 1.0) * scale) as f64)).collect()
}

fn bf16s(rng: &mut Rng, n: usize, scale: f32) -> Vec<u16> {
    (0..n).map(|_| gb_cuda::num::f32_to_bf16(rng.uniform(-1.0, 1.0) * scale)).collect()
}

fn to_f64_f16(v: &[u16]) -> Vec<f64> {
    v.iter().map(|&x| gb_exl3::f16::f16_to_f64(x)).collect()
}

fn to_f64_bf16(v: &[u16]) -> Vec<f64> {
    v.iter().map(|&x| gb_cuda::num::bf16_to_f32(x) as f64).collect()
}

fn to_u16(dtype: Dtype, v: &[f64]) -> Vec<u16> {
    match dtype {
        Dtype::F16 => v.iter().map(|&x| gb_exl3::f16::f64_to_f16(x)).collect(),
        Dtype::BF16 => v.iter().map(|&x| gb_cuda::num::f32_to_bf16(x as f32)).collect(),
        Dtype::F32 => panic!(),
    }
}

/// f64 reference of `BailingMoeV3MultiLatentAttention.forward` on rows with positions `pos` (ascending, one slot),
/// given the rounded `q` `[rows][64*192]` (head h at h*192), `kv` `[rows][640]` (cols 0..511 latent, 512..575
/// rope), the layer input `x` `[rows][4096]`, the packed `uk` `[64][128][512]`, `uvt` `[64][512][128]`,
/// `norm_w` `[512]` and `g_proj` `[64][4096]` in f64.
fn ref_mla(x: &[f64], q: &[f64], kv: &[f64], norm_w: &[f64], uk: &[f64], uvt: &[f64], g: &[f64], pos: &[i32]) -> Vec<f64> {
    let rows = pos.len();
    let theta = ROPE_THETA as f64;
    let inv_freq: Vec<f64> = (0..32).map(|i| 1.0 / theta.powf(2.0 * i as f64 / 64.0)).collect();
    let mut out = vec![0.0f64; rows * 64 * 128];
    for r in 0..rows {
        let pp = pos[r] as usize;
        let cs: Vec<f64> = (0..32).map(|i| (pp as f64 * inv_freq[i]).cos()).collect();
        let sn: Vec<f64> = (0..32).map(|i| (pp as f64 * inv_freq[i]).sin()).collect();
        let rot = |src: &[f64]| -> Vec<f64> {
            let mut y = vec![0.0f64; 64];
            for i in 0..32 {
                y[i] = src[2 * i] * cs[i] - src[2 * i + 1] * sn[i];
                y[i + 32] = src[2 * i] * sn[i] + src[2 * i + 1] * cs[i];
            }
            y
        };
        // per-token normalised latents and rotated rope keys of this row's prefix
        let np = pp + 1;
        let mut cnorm = vec![vec![0.0f64; 512]; np];
        let mut krot = vec![vec![0.0f64; 64]; np];
        for j in 0..np {
            let mean_sq = (0..512).map(|c| kv[j * 640 + c].powi(2)).sum::<f64>() / 512.0;
            let inv = 1.0 / (mean_sq + 1e-6f64).sqrt();
            for c in 0..512 {
                cnorm[j][c] = kv[j * 640 + c] * inv * norm_w[c];
            }
            krot[j] = rot(&kv[j * 640 + 512..j * 640 + 576]);
        }
        for h in 0..64 {
            let qh = &q[r * 12288 + h * 192..r * 12288 + h * 192 + 192];
            let qpass = &qh[0..128];
            let qrot_h = rot(&qh[128..192]);
            // absorb
            let mut qlat = vec![0.0f64; 512];
            for d in 0..128 {
                for c in 0..512 {
                    qlat[c] += qpass[d] * uk[(h * 128 + d) * 512 + c];
                }
            }
            // scores + online softmax over keys 0..=pos
            let mut scores = vec![0.0f64; np];
            let mut mx = -1e300f64;
            for j in 0..np {
                let mut s = 0.0f64;
                for c in 0..512 {
                    s += qlat[c] * cnorm[j][c];
                }
                for i in 0..64 {
                    s += qrot_h[i] * krot[j][i];
                }
                s *= SM_SCALE as f64;
                scores[j] = s;
                mx = mx.max(s);
            }
            let mut sum = 0.0f64;
            for j in 0..np {
                scores[j] = (scores[j] - mx).exp();
                sum += scores[j];
            }
            let mut olat = vec![0.0f64; 512];
            for j in 0..np {
                let pj = scores[j] / sum;
                for c in 0..512 {
                    olat[c] += pj * cnorm[j][c];
                }
            }
            for n in 0..128 {
                let mut v = 0.0f64;
                for c in 0..512 {
                    v += olat[c] * uvt[(h * 512 + c) * 128 + n];
                }
                out[r * 8192 + h * 128 + n] = v;
            }
        }
        // head-wise sigmoid gate on the layer input
        for h in 0..64 {
            let mut dot = 0.0f64;
            for c in 0..4096 {
                dot += x[r * 4096 + c] * g[h * 4096 + c];
            }
            let gate = 1.0 / (1.0 + (-dot).exp());
            for n in 0..128 {
                out[r * 8192 + h * 128 + n] *= gate;
            }
        }
    }
    out
}

struct Harness {
    dtype: Dtype,
    x: Vec<u16>,
    q: Vec<u16>,
    kv: Vec<u16>,
    norm_w: Vec<u16>,
    uk: Vec<u16>,
    uvt: Vec<u16>,
    g: Vec<u16>,
    pos: Vec<i32>,
    n: usize,
}

fn make(rows: usize, dtype: Dtype, scale: f32, seed: u64) -> Harness {
    let mut rng = Rng(seed);
    let n = rows;
    // Activations (x, q, kv, norm_w) are stored in the active dtype; the kv_b checkpoint source and g_proj
    // stay f16 / bf16 respectively regardless of dtype.
    let acts = |rng: &mut Rng, n: usize, s: f32| match dtype {
        Dtype::F16 => f16s(rng, n, s),
        Dtype::BF16 => bf16s(rng, n, s),
        Dtype::F32 => unreachable!(),
    };
    let x = acts(&mut rng, n * 4096, scale);
    let q = acts(&mut rng, n * 12288, scale * 0.5);
    let kv = acts(&mut rng, n * 640, scale * 0.5);
    let norm_w = acts(&mut rng, 512, 1.0);
    let uk = f16s(&mut rng, 64 * 128 * 512, 0.05);
    let uvt = f16s(&mut rng, 64 * 512 * 128, 0.05);
    let g = bf16s(&mut rng, 64 * 4096, 0.05);
    let pos: Vec<i32> = (0..rows as i32).collect();
    Harness { dtype, x, q, kv, norm_w, uk, uvt, g, pos, n }
}

impl Harness {
    fn ref_out(&self) -> Vec<f64> {
        let (x, q, kv, norm_w) = match self.dtype {
            Dtype::F16 => (to_f64_f16(&self.x), to_f64_f16(&self.q), to_f64_f16(&self.kv), to_f64_f16(&self.norm_w)),
            Dtype::BF16 => (to_f64_bf16(&self.x), to_f64_bf16(&self.q), to_f64_bf16(&self.kv), to_f64_bf16(&self.norm_w)),
            Dtype::F32 => unreachable!(),
        };
        ref_mla(&x, &q, &kv, &norm_w, &to_f64_f16(&self.uk), &to_f64_f16(&self.uvt), &to_f64_bf16(&self.g), &self.pos)
    }
    fn kv_b_full(&self) -> Vec<u16> {
        // Reconstruct the checkpoint's `kv_b_proj.weight` `[64 * 256][512]`: rows `h*256 + d` (d < 128) are
        // `W_UK`, rows `h*256 + 128 + n` are `W_UV` (with `W_UV[h][c][n]` = `uvt[h][c][n]`).
        let mut kv_b = vec![0u16; N_HEADS * KV_B_ROWS_PER_HEAD * LATENT_DIM];
        for h in 0..N_HEADS {
            for d in 0..NOPE_DIM {
                for c in 0..LATENT_DIM {
                    kv_b[(h * KV_B_ROWS_PER_HEAD + d) * LATENT_DIM + c] = self.uk[(h * NOPE_DIM + d) * LATENT_DIM + c];
                }
            }
            for n in 0..NOPE_DIM {
                for c in 0..LATENT_DIM {
                    kv_b[(h * KV_B_ROWS_PER_HEAD + NOPE_DIM + n) * LATENT_DIM + c] =
                        self.uvt[(h * LATENT_DIM + c) * NOPE_DIM + n];
                }
            }
        }
        kv_b
    }
    fn weights(&self) -> MlaGatedWeights {
        let kv_b = self.kv_b_full();
        // `g_proj` is always bf16 bits; the gate kernel reads it as bf16 regardless of the activation dtype.
        MlaGatedWeights::from_checkpoint(&kv_b, Dtype::F16, &self.g, self.dtype).unwrap()
    }
}

/// One device pipeline over a shared cache: rows are appended in calls of any size (prefill: one call with all
/// rows; decode: many calls), and a row's gated output is bit-identical no matter how its neighbours were chunked.
struct Runner {
    dtype: Dtype,
    h: Harness,
    st: Stream,
    w: MlaGatedWeights,
    cache: MlaGatedCache,
    table: BlockTable,
    cos_b: DevBuf,
    sin_b: DevBuf,
    qlat: DevBuf,
    qrot: DevBuf,
    olat: DevBuf,
    out: DevBuf,
    gated: DevBuf,
}

impl Runner {
    fn new(h: Harness) -> Self {
        let dtype = h.dtype;
        let st = Stream::new().unwrap();
        let w = h.weights();
        let cache = MlaGatedCache::new(8, dtype).unwrap();
        let table = BlockTable::from_host(&[vec![0, 1, 2, 3, 4, 5, 6, 7]], 8).unwrap();
        let (cos, sin) = rope_tables(1024, ROPE_THETA);
        let cos_b = DevBuf::from_slice(&cos).unwrap();
        let sin_b = DevBuf::from_slice(&sin).unwrap();
        let qlat = DevBuf::alloc(h.n * 64 * 512 * dtype.size()).unwrap();
        let qrot = DevBuf::alloc(h.n * 64 * 64 * dtype.size()).unwrap();
        let olat = DevBuf::alloc(h.n * 64 * 512 * dtype.size()).unwrap();
        let out = DevBuf::alloc(h.n * 64 * 128 * dtype.size()).unwrap();
        let gated = DevBuf::alloc(h.n * 64 * 128 * dtype.size()).unwrap();
        Runner { dtype, h, st, w, cache, table, cos_b, sin_b, qlat, qrot, olat, out, gated }
    }
    fn run(&mut self, first: usize, count: usize) -> Vec<u16> {
        let n = count;
        let slots: Vec<i32> = vec![0; n];
        let pos: Vec<i32> = self.h.pos[first..first + n].to_vec();
        let ri = RowIndex::from_host(&slots, &pos).unwrap();
        let x_b = DevBuf::from_slice(&self.h.x[first * 4096..first * 4096 + n * 4096]).unwrap();
        let q_b = DevBuf::from_slice(&self.h.q[first * 12288..first * 12288 + n * 12288]).unwrap();
        let kv_b = DevBuf::from_slice(&self.h.kv[first * 640..first * 640 + n * 640]).unwrap();
        let norm_b = DevBuf::from_slice(&self.h.norm_w).unwrap();
        kv_prep(&kv_b, &norm_b, &self.cos_b, &self.sin_b, &ri, &self.table, &mut self.cache, n, &self.st).unwrap();
        q_prep(&q_b, &self.w, &self.cos_b, &self.sin_b, &ri, &mut self.qlat, &mut self.qrot, n, &self.st).unwrap();
        attend(&self.qlat, &self.qrot, &self.cache, &self.table, &ri, &mut self.olat, n, &self.st).unwrap();
        expand(&self.olat, &self.w, &mut self.out, n, &self.st).unwrap();
        gate_apply(&x_b, &self.w, &self.out, &mut self.gated, n, &self.st).unwrap();
        let mut got = vec![0u16; n * 8192];
        self.gated.download(&mut got).unwrap();
        got
    }
}

fn cos_sim(a: &[f64], b: &[f64]) -> f64 {
    let mut ab = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for (&x, &y) in a.iter().zip(b) {
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt()
}

#[cfg(not(gb_no_cuda))]
#[test]
fn mla_gated_matches_cpu_reference() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        for (rows, seed) in [(1u64, 11u64), (7, 22), (33, 33), (127, 44)] {
            let h = make(rows as usize, dtype, 0.5, seed);
            let want = h.ref_out();
            let mut runner = Runner::new(h);
            let got = runner.run(0, rows as usize);
            let got_f: Vec<f64> = match dtype {
                Dtype::F16 => to_f64_f16(&got),
                Dtype::BF16 => to_f64_bf16(&got),
                Dtype::F32 => unreachable!(),
            };
            let cos = cos_sim(&got_f, &want);
            let mut rel = 0.0f64;
            for (&g, &w) in got_f.iter().zip(&want) {
                let d = (g - w).abs();
                let den = w.abs().max(1e-6);
                rel = rel.max(d / den);
            }
            assert!(cos >= 0.9999, "dtype {dtype:?} rows {rows}: cos {cos} < 0.9999 (relErr {rel})");
        }
    }
}

#[cfg(not(gb_no_cuda))]
#[test]
fn mla_gated_row_invariant_across_chunkings() {
    for dtype in [Dtype::F16, Dtype::BF16] {
        let h = make(40, dtype, 0.5, 77);
        // prefill: the whole batch in one call
        let mut runner = Runner::new(h);
        let all = runner.run(0, 40);
        // decode-style: append chunks of different sizes, one call each over the shared cache
        for chunks in [vec![1usize; 40], vec![7, 6, 5, 4, 3, 2, 1, 12], vec![13, 27]] {
            let mut runner = Runner::new(make(40, dtype, 0.5, 77));
            let mut first = 0usize;
            let mut cumulative = Vec::new();
            for c in chunks {
                cumulative.extend(runner.run(first, c));
                first += c;
            }
            assert_eq!(cumulative, all, "dtype {dtype:?} chunking differs bit-for-bit");
        }
    }
}

#[cfg(not(gb_no_cuda))]
#[test]
fn bf16_linear_matches_cpu_reference() {
    let mut rng = Rng(99);
    for (rows, k, n) in [(1usize, 256usize, 1usize), (3, 1024, 8), (17, 4096, 512)] {
        let x: Vec<u16> = bf16s(&mut rng, rows * k, 1.0);
        let w: Vec<u16> = bf16s(&mut rng, n * k, 1.0);
        let xf = to_f64_bf16(&x);
        let wf = to_f64_bf16(&w);
        let mut want = vec![0.0f64; rows * n];
        for r in 0..rows {
            for j in 0..n {
                let mut s = 0.0f64;
                for c in 0..k {
                    s += xf[r * k + c] * wf[j * k + c];
                }
                want[r * n + j] = s;
            }
        }
        let st = Stream::new().unwrap();
        let xb = DevBuf::from_slice(&x).unwrap();
        let wb = DevBuf::from_slice(&w).unwrap();
        let mut yb = DevBuf::alloc(rows * n * 2).unwrap();
        bf16_linear(&xb, &wb, &mut yb, rows, k, n, &st).unwrap();
        let mut got = vec![0u16; rows * n];
        yb.download(&mut got).unwrap();
        let got_f = to_f64_bf16(&got);
        let cos = cos_sim(&got_f, &want);
        assert!(cos >= 0.9999, "rows {rows} k {k} n {n}: cos {cos}");
    }
}

#[cfg(not(gb_no_cuda))]
#[test]
fn eh_proj_cat_bf16_matches_cpu_reference() {
    let mut rng = Rng(123);
    for (rows, h) in [(1usize, 256usize), (5, 1024), (13, 4096)] {
        let e: Vec<u16> = bf16s(&mut rng, rows * h, 1.0);
        let hn: Vec<u16> = bf16s(&mut rng, rows * h, 1.0);
        let w: Vec<u16> = bf16s(&mut rng, h * 2 * h, 1.0);
        let ef = to_f64_bf16(&e);
        let hf = to_f64_bf16(&hn);
        let wf = to_f64_bf16(&w);
        let mut want = vec![0.0f64; rows * h];
        for r in 0..rows {
            for j in 0..h {
                let mut s = 0.0f64;
                for c in 0..h {
                    s += ef[r * h + c] * wf[j * 2 * h + c];
                }
                for c in 0..h {
                    s += hf[r * h + c] * wf[j * 2 * h + h + c];
                }
                want[r * h + j] = s;
            }
        }
        let st = Stream::new().unwrap();
        let eb = DevBuf::from_slice(&e).unwrap();
        let hb = DevBuf::from_slice(&hn).unwrap();
        let wb = DevBuf::from_slice(&w).unwrap();
        let mut yb = DevBuf::alloc(rows * h * 2).unwrap();
        eh_proj_cat_bf16(&eb, &hb, &wb, &mut yb, rows, h, &st).unwrap();
        let mut got = vec![0u16; rows * h];
        yb.download(&mut got).unwrap();
        let got_f = to_f64_bf16(&got);
        let cos = cos_sim(&got_f, &want);
        assert!(cos >= 0.9999, "rows {rows} h {h}: cos {cos}");
    }
}

#[cfg(not(gb_no_cuda))]
#[test]
fn bf16_linear_is_bit_exact_to_the_documented_tree_order() {
    // Replicates the kernel's exact f32 FMA chains (ascending c8, element order) and the
    // __shfl_down_sync tree order (delta 16, 8, 4, 2, 1) and requires bit-identical bf16 outputs.
    let mut rng = Rng(7);
    let (rows, k, n) = (4usize, 1024usize, 33usize);
    let x: Vec<u16> = bf16s(&mut rng, rows * k, 0.8);
    let w: Vec<u16> = bf16s(&mut rng, n * k, 0.8);
    let xf: Vec<f32> = x.iter().map(|&v| gb_cuda::num::bf16_to_f32(v)).collect();
    let wf: Vec<f32> = w.iter().map(|&v| gb_cuda::num::bf16_to_f32(v)).collect();
    let mut want = vec![0u16; rows * n];
    for r in 0..rows {
        for j in 0..n {
            let mut parts = [0.0f32; 32];
            for lane in 0..32 {
                let mut acc = 0.0f32;
                let mut c8 = lane;
                while c8 < k / 8 {
                    let base = c8 * 8;
                    for e in 0..8 {
                        acc = xf[r * k + base + e].mul_add(wf[j * k + base + e], acc);
                    }
                    c8 += 32;
                }
                parts[lane] = acc;
            }
            for d in [16usize, 8, 4, 2, 1] {
                for i in 0..(32 - d) {
                    parts[i] += parts[i + d];
                }
            }
            want[r * n + j] = gb_cuda::num::f32_to_bf16(parts[0]);
        }
    }
    let st = Stream::new().unwrap();
    let xb = DevBuf::from_slice(&x).unwrap();
    let wb = DevBuf::from_slice(&w).unwrap();
    let mut yb = DevBuf::alloc(rows * n * 2).unwrap();
    bf16_linear(&xb, &wb, &mut yb, rows, k, n, &st).unwrap();
    let mut got = vec![0u16; rows * n];
    yb.download(&mut got).unwrap();
    assert_eq!(got, want, "bf16_linear bits differ from the documented f32 tree order");
}
