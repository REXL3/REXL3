//! The Ling-3.1-flash-1019 Gated-MLA under the LP2 tensor-core contract (`crates/cuda/src/ling_mla_attn.rs`,
//! `csrc/ling_mla_attn.cu`) against the same CPU reference transcription of `BailingMoeV3MultiLatentAttention`
//! that `tests/mla_gated.rs` uses, plus the row/chunk-invariance and determinism contract of P2.
//!
//! P2 is NOT bit-equal to P1 (the contraction orders differ by design), so the assertions are:
//!   * f64-oracle bounded error (cosine >= 0.9999, the same bar as the scalar path's own test),
//!   * row/chunk independence: a row's gated output is bit-identical across prefill chunkings (P2 (a)),
//!   * determinism: two runs of the same launch produce identical bits (P2 (d)),
//!   * P1 vs P2 agreement within the bf16 rounding budget (the qualification gate for the contract flip).
//!
//! Run on a GB10 box:
//!   gpujob.sh cargo test --release -p gb-cuda --test ling_mla_attn -- --nocapture --test-threads=1
#![cfg(not(gb_no_cuda))]
#![allow(dead_code)]

mod common;

use common::Rng;
use gb_cuda::attention::{BlockTable, RowIndex};
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::ling_mla_attn as tc;
use gb_cuda::mla_gated as sc;
use gb_cuda::mla_gated::*;
use gb_cuda::rt::{DevBuf, Stream};

fn bf16s(rng: &mut Rng, n: usize, scale: f32) -> Vec<u16> {
    (0..n).map(|_| gb_cuda::num::f32_to_bf16(rng.uniform(-1.0, 1.0) * scale)).collect()
}
fn f16s(rng: &mut Rng, n: usize, scale: f32) -> Vec<u16> {
    (0..n).map(|_| gb_exl3::f16::f64_to_f16((rng.uniform(-1.0, 1.0) * scale) as f64)).collect()
}
fn to_f64_bf16(v: &[u16]) -> Vec<f64> {
    v.iter().map(|&x| gb_cuda::num::bf16_to_f32(x) as f64).collect()
}
fn to_f64_f16(v: &[u16]) -> Vec<f64> {
    v.iter().map(|&x| gb_exl3::f16::f16_to_f64(x)).collect()
}

/// f64 reference of `BailingMoeV3MultiLatentAttention.forward` on rows with positions `pos` (one slot),
/// identical to `tests/mla_gated.rs::ref_mla` (the oracle both contracts are checked against).
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
            let mut qlat = vec![0.0f64; 512];
            for d in 0..128 {
                for c in 0..512 {
                    qlat[c] += qpass[d] * uk[(h * 128 + d) * 512 + c];
                }
            }
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

fn make(rows: usize, scale: f32, seed: u64) -> Harness {
    let mut rng = Rng(seed);
    let n = rows;
    let x = bf16s(&mut rng, n * 4096, scale);
    let q = bf16s(&mut rng, n * 12288, scale * 0.5);
    let kv = bf16s(&mut rng, n * 640, scale * 0.5);
    let norm_w = bf16s(&mut rng, 512, 1.0);
    let uk = f16s(&mut rng, 64 * 128 * 512, 0.05);
    let uvt = f16s(&mut rng, 64 * 512 * 128, 0.05);
    let g = bf16s(&mut rng, 64 * 4096, 0.05);
    let pos: Vec<i32> = (0..rows as i32).collect();
    Harness { x, q, kv, norm_w, uk, uvt, g, pos, n }
}

impl Harness {
    fn ref_out(&self) -> Vec<f64> {
        let (x, q, kv, norm_w) = (to_f64_bf16(&self.x), to_f64_bf16(&self.q), to_f64_bf16(&self.kv), to_f64_bf16(&self.norm_w));
        ref_mla(&x, &q, &kv, &norm_w, &to_f64_f16(&self.uk), &to_f64_f16(&self.uvt), &to_f64_bf16(&self.g), &self.pos)
    }
    fn kv_b_full(&self) -> Vec<u16> {
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
        // bf16 dst: the tensor-core path is bf16 only (Ling-3.1 binds kv_b as bf16).
        MlaGatedWeights::from_checkpoint(&kv_b, Dtype::F16, &self.g, Dtype::BF16).unwrap()
    }
}

/// One device pipeline over a shared cache, parameterized by contract (P1 scalar / P2 tensor core).
struct Runner {
    p2: bool,
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
    fn new(h: Harness, p2: bool) -> Self {
        let st = Stream::new().unwrap();
        let w = h.weights();
        let cache = MlaGatedCache::new(8, Dtype::BF16).unwrap();
        let table = BlockTable::from_host(&[vec![0, 1, 2, 3, 4, 5, 6, 7]], 8).unwrap();
        let (cos, sin) = rope_tables(1024, ROPE_THETA);
        let cos_b = DevBuf::from_slice(&cos).unwrap();
        let sin_b = DevBuf::from_slice(&sin).unwrap();
        let qlat = DevBuf::alloc(h.n * 64 * 512 * 2).unwrap();
        let qrot = DevBuf::alloc(h.n * 64 * 64 * 2).unwrap();
        let olat = DevBuf::alloc(h.n * 64 * 512 * 2).unwrap();
        let out = DevBuf::alloc(h.n * 64 * 128 * 2).unwrap();
        let gated = DevBuf::alloc(h.n * 64 * 128 * 2).unwrap();
        Runner { p2, h, st, w, cache, table, cos_b, sin_b, qlat, qrot, olat, out, gated }
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
        if self.p2 {
            tc::mla_gated_rows_tc(
                &x_b,
                &q_b,
                &kv_b,
                &norm_b,
                &self.cos_b,
                &self.sin_b,
                &self.w,
                &mut self.cache,
                &self.table,
                &ri,
                n,
                &mut self.qlat,
                &mut self.qrot,
                &mut self.olat,
                &mut self.gated,
                false,
                &self.st,
            )
            .unwrap();
        } else {
            sc::mla_gated_rows(
                &x_b,
                &q_b,
                &kv_b,
                &norm_b,
                &self.cos_b,
                &self.sin_b,
                &self.w,
                &mut self.cache,
                &self.table,
                &ri,
                n,
                &mut self.qlat,
                &mut self.qrot,
                &mut self.olat,
                &mut self.out,
                &mut self.gated,
                &self.st,
            )
            .unwrap();
        }
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
fn tc_mla_matches_cpu_reference() {
    for (rows, seed) in [(1u64, 11u64), (7, 22), (33, 33), (127, 44)] {
        let h = make(rows as usize, 0.5, seed);
        let want = h.ref_out();
        let mut runner = Runner::new(h, true);
        let got = runner.run(0, rows as usize);
        let got_f = to_f64_bf16(&got);
        let cos = cos_sim(&got_f, &want);
        let mut rel = 0.0f64;
        for (&g, &w) in got_f.iter().zip(&want) {
            let d = (g - w).abs();
            let den = w.abs().max(1e-6);
            rel = rel.max(d / den);
        }
        assert!(cos >= 0.9999, "tc rows {rows}: cos {cos} < 0.9999 (relErr {rel})");
    }
}

#[cfg(not(gb_no_cuda))]
#[test]
fn tc_mla_row_invariant_across_chunkings() {
    // P2 (a): a row's gated output is bit-identical no matter how its neighbours were chunked.
    let h = make(40, 0.5, 77);
    let mut runner = Runner::new(h, true);
    let all = runner.run(0, 40);
    for chunks in [vec![1usize; 40], vec![7, 6, 5, 4, 3, 2, 1, 12], vec![13, 27]] {
        let mut runner = Runner::new(make(40, 0.5, 77), true);
        let mut first = 0usize;
        let mut cumulative = Vec::new();
        for c in chunks {
            cumulative.extend(runner.run(first, c));
            first += c;
        }
        assert_eq!(cumulative, all, "tc chunking differs bit-for-bit");
    }
}

#[cfg(not(gb_no_cuda))]
#[test]
fn tc_mla_deterministic_run_to_run() {
    // P2 (d): the same launch replays the same bits.
    let h = make(33, 0.5, 99);
    let mut r1 = Runner::new(h, true);
    let a = r1.run(0, 33);
    let b = r1.run(0, 33); // second run over the same cache/positions
    assert_eq!(a, b, "tc attention is not deterministic run to run");
}

#[cfg(not(gb_no_cuda))]
#[test]
fn p1_and_p2_agree_within_bf16_rounding() {
    // The qualification gate for the contract flip: P1 (scalar) and P2 (tensor core) consume the same
    // weights and cache and must agree to within the bf16 rounding budget of the contraction reorder.
    for (rows, seed) in [(1u64, 5u64), (16, 6), (40, 7)] {
        let want_p1 = {
            let mut r = Runner::new(make(rows as usize, 0.5, seed), false);
            r.run(0, rows as usize)
        };
        let got_p2 = {
            let mut r = Runner::new(make(rows as usize, 0.5, seed), true);
            r.run(0, rows as usize)
        };
        let a = to_f64_bf16(&want_p1);
        let b = to_f64_bf16(&got_p2);
        let cos = cos_sim(&a, &b);
        assert!(cos >= 0.9999, "P1 vs P2 rows {rows}: cos {cos} < 0.9999");
        // bounded absolute deviation (bf16 quantum on the gated output scale)
        let mut max_d = 0.0f64;
        for (&x, &y) in a.iter().zip(&b) {
            max_d = max_d.max((x - y).abs());
        }
        assert!(max_d < 0.05, "P1 vs P2 rows {rows}: max |diff| {max_d} too large");
    }
}

/// The windowed attention (`mla_gated::attend_from`, `ling_mla_attn::attend_from`): a query row attends the keys of a window that starts at a
/// page of its slot, `first_block` pages in, with its position counted from there. Ling-3.1's MTP drafter uses it past 8K tokens, where the
/// entries below the window were never written.
///
/// * `first_block == 0` is the plain attention, bit for bit;
/// * the window ignores everything before its first page: rewriting those pages changes no bit of the output, while the plain attention
///   over the same pages does change;
/// * the window is the same attention over fewer keys: the output equals the plain attention over a cache that holds the window's keys at
///   positions from 0 (the rope is relative, so shifting keys and query alike leaves the scores alone), to the rounding of the cached keys.
mod window {
    use crate::common::Rng;
    use gb_cuda::attention::{BlockTable, RowIndex};
    use gb_cuda::exl3_linear::Dtype;
    use gb_cuda::ling_mla_attn as tc;
    use gb_cuda::mla_gated::{self as sc, *};
    use gb_cuda::rt::{DevBuf, Stream};

    const PAGE: usize = 256;
    /// Rows written: page 0 and 44 rows of page 1.
    const ROWS: usize = 300;
    /// The window starts at page 1; the query is the last row (position 299, 43 counted from the window).
    const WINDOW_BLOCK: usize = 1;

    fn bf16s(rng: &mut Rng, n: usize, scale: f32) -> Vec<u16> {
        (0..n).map(|_| gb_cuda::num::f32_to_bf16(rng.uniform(-1.0, 1.0) * scale)).collect()
    }

    fn cos_sim(a: &[u16], b: &[u16]) -> f64 {
        let f = |v: &[u16]| v.iter().map(|&x| gb_cuda::num::bf16_to_f32(x) as f64).collect::<Vec<_>>();
        let (a, b) = (f(a), f(b));
        let dot: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        dot / (a.iter().map(|x| x * x).sum::<f64>() * b.iter().map(|x| x * x).sum::<f64>()).sqrt()
    }

    struct Env {
        st: Stream,
        norm_w: DevBuf,
        cos: DevBuf,
        sin: DevBuf,
        q_lat: DevBuf,
        /// The raw query of the last row `[64][192]`.
        q: DevBuf,
        kv: Vec<u16>,
    }

    fn env() -> Env {
        let mut rng = Rng(2026);
        let (cos, sin) = rope_tables(1024, ROPE_THETA);
        Env {
            st: Stream::new().unwrap(),
            norm_w: DevBuf::from_slice(&bf16s(&mut rng, 512, 1.0)).unwrap(),
            cos: DevBuf::from_slice(&cos).unwrap(),
            sin: DevBuf::from_slice(&sin).unwrap(),
            q_lat: DevBuf::from_slice(&bf16s(&mut rng, 64 * 512, 0.5)).unwrap(),
            q: DevBuf::from_slice(&bf16s(&mut rng, 64 * 192, 0.5)).unwrap(),
            kv: bf16s(&mut rng, ROWS * 640, 0.5),
        }
    }

    /// Write the cache entries of `kv` (rows of 640) at positions `at..at + rows` of slot 0 into `cache`.
    fn write(e: &Env, cache: &mut MlaGatedCache, table: &BlockTable, kv: &[u16], at: usize) {
        let n = kv.len() / 640;
        let pos: Vec<i32> = (at..at + n).map(|p| p as i32).collect();
        let ri = RowIndex::from_host(&vec![0; n], &pos).unwrap();
        let kv_b = DevBuf::from_slice(kv).unwrap();
        kv_prep(&kv_b, &e.norm_w, &e.cos, &e.sin, &ri, table, cache, n, &e.st).unwrap();
    }

    fn fresh(e: &Env, kv: &[u16], at: usize) -> (MlaGatedCache, BlockTable) {
        let mut cache = MlaGatedCache::new(2, Dtype::BF16).unwrap();
        let table = BlockTable::from_host(&[vec![0, 1]], 2).unwrap();
        write(e, &mut cache, &table, kv, at);
        (cache, table)
    }

    /// The query's `q_rot` at position `pos` (the rope of the raw query).
    fn q_rot_at(e: &Env, pos: usize) -> DevBuf {
        let ri = RowIndex::from_host(&[0], &[pos as i32]).unwrap();
        let mut q_rot = DevBuf::alloc(64 * 64 * 2).unwrap();
        sc::q_rope_only(&e.q, &e.cos, &e.sin, &ri, &mut q_rot, 1, &e.st).unwrap();
        q_rot
    }

    #[derive(Clone, Copy, Debug)]
    enum Kernel {
        Scalar,
        Tensor,
    }

    /// One query row at `pos` (counted from `first_block`) over `cache`/`table`, through the plain or the windowed entry point.
    fn attend_row(
        e: &Env,
        k: Kernel,
        cache: &MlaGatedCache,
        table: &BlockTable,
        first_block: Option<usize>,
        pos: usize,
        q_rot: &DevBuf,
    ) -> Vec<u16> {
        let ri = RowIndex::from_host(&[0], &[pos as i32]).unwrap();
        let mut o = DevBuf::alloc(64 * 512 * 2).unwrap();
        match (k, first_block) {
            (Kernel::Scalar, None) => sc::attend(&e.q_lat, q_rot, cache, table, &ri, &mut o, 1, &e.st).unwrap(),
            (Kernel::Scalar, Some(b)) => sc::attend_from(&e.q_lat, q_rot, cache, table, b, &ri, &mut o, 1, &e.st).unwrap(),
            (Kernel::Tensor, None) => tc::attend(&e.q_lat, q_rot, cache, table, &ri, &mut o, 1, false, &e.st).unwrap(),
            (Kernel::Tensor, Some(b)) => tc::attend_from(&e.q_lat, q_rot, cache, table, b, &ri, &mut o, 1, false, &e.st).unwrap(),
        }
        let mut out = vec![0u16; 64 * 512];
        o.download(&mut out).unwrap();
        out
    }

    #[cfg(not(gb_no_cuda))]
    #[test]
    fn a_window_from_block_zero_is_the_plain_attention() {
        let e = env();
        let (cache, table) = fresh(&e, &e.kv, 0);
        let q_rot = q_rot_at(&e, ROWS - 1);
        for k in [Kernel::Scalar, Kernel::Tensor] {
            let plain = attend_row(&e, k, &cache, &table, None, ROWS - 1, &q_rot);
            let from0 = attend_row(&e, k, &cache, &table, Some(0), ROWS - 1, &q_rot);
            assert_eq!(plain, from0, "{k:?}: first_block 0 differs from the plain attention");
        }
    }

    #[cfg(not(gb_no_cuda))]
    #[test]
    fn the_window_ignores_the_pages_before_it() {
        let e = env();
        let (mut cache, table) = fresh(&e, &e.kv, 0);
        let q_rot = q_rot_at(&e, ROWS - 1);
        let pos = ROWS - 1 - WINDOW_BLOCK * PAGE;
        for k in [Kernel::Scalar, Kernel::Tensor] {
            let plain_before = attend_row(&e, k, &cache, &table, None, ROWS - 1, &q_rot);
            let win_before = attend_row(&e, k, &cache, &table, Some(WINDOW_BLOCK), pos, &q_rot);
            assert_ne!(plain_before, win_before, "{k:?}: the window must drop keys (control)");
            // rewrite page 0 with other keys: positions 0..PAGE
            let mut rng = Rng(99);
            let other = bf16s(&mut rng, PAGE * 640, 2.0);
            write(&e, &mut cache, &table, &other, 0);
            let win_after = attend_row(&e, k, &cache, &table, Some(WINDOW_BLOCK), pos, &q_rot);
            let plain_after = attend_row(&e, k, &cache, &table, None, ROWS - 1, &q_rot);
            assert_eq!(win_before, win_after, "{k:?}: the window read a page before its start");
            assert_ne!(plain_before, plain_after, "{k:?}: the plain attention must read page 0 (control)");
            // put the original page back for the next kernel
            write(&e, &mut cache, &table, &e.kv[..PAGE * 640], 0);
        }
    }

    #[cfg(not(gb_no_cuda))]
    #[test]
    fn the_window_is_the_attention_over_its_keys_alone() {
        let e = env();
        let (cache, table) = fresh(&e, &e.kv, 0);
        // the window's keys alone, at positions from 0: rows PAGE..ROWS written at 0..ROWS-PAGE
        let (small, small_table) = fresh(&e, &e.kv[PAGE * 640..], 0);
        let pos = ROWS - 1 - PAGE;
        let q_rot_abs = q_rot_at(&e, ROWS - 1);
        let q_rot_rel = q_rot_at(&e, pos);
        for k in [Kernel::Scalar, Kernel::Tensor] {
            let win = attend_row(&e, k, &cache, &table, Some(WINDOW_BLOCK), pos, &q_rot_abs);
            let alone = attend_row(&e, k, &small, &small_table, None, pos, &q_rot_rel);
            let cs = cos_sim(&win, &alone);
            assert!(cs >= 0.999, "{k:?}: the window differs from the attention over its keys alone (cosine {cs})");
            eprintln!("{k:?}: window vs keys-alone cosine {cs:.6}");
        }
    }
}

/// The row-batched Ling MTP `eh_proj` (`ling_mla_attn::eh_proj_cat_bf16`, the end of `csrc/ling_mla_attn.cu`) against the shared kernel it replaces on the
/// Ling path (`mla_gated::eh_proj_cat_bf16`, `csrc/mla_gated.cu`'s `gb_mlag_eh_proj_cat_bf16_kernel`).
///
/// The contract is bit identity per row, whatever the tile (`j` columns a warp x `r` rows a weight read), the rows a CTA walks, the number of rows
/// in the call and the row's index in it: one output is one lane-strided fp32 fma chain (e half, then h half), a 5-step shfl_down tree, one bf16
/// rounding, and no other row ever joins that chain. Every comparison below is on raw bits:
///   * the production launch equals the shared kernel for every row count 1..=70 (the decode / chain / absorb widths) on random operands;
///   * every instantiated tile, with slabs that split the call at 1, 5, 24, 25 and 1,024 rows, equals the shared kernel at the R boundaries
///     (r-1, r, r+1, 2r-1, 2r, 2r+1) and across a 1,100-row call (the 1,024-row slab boundary);
///   * a row computed alone (a 1-row call, the chain step) equals the same row inside the 70-row and the 1,100-row calls;
///   * operand regimes: wide dynamic range (per-column weight scales 2^-8 .. 2^8, heavy-tailed activations, exact zeros), so rounding order bites;
///   * widths other than 4096 (256, 1024, 2048);
///   * a CPU transcription of the documented chain (lane stride, fma order, tree, one rounding) equals both kernels on a block of outputs;
///   * the real L72 `eh_proj` weight of the pack (`REXL3_LING_DIR`, ignored by default) with random normalised activations.
/// The oracle's own sensitivity (a different order or a skipped rounding gives different bits on this fixture) is a CPU-only test, so a
/// GPU mutation of the kernel is known to be catchable.
///
/// Run on a GB10 box (names are `eh_proj_rows::...`):
///   cargo test --release -p gb-cuda --test ling_mla_attn eh_proj_rows -- --nocapture --test-threads=1
///   REXL3_LING_DIR=<pack> cargo test --release -p gb-cuda --test ling_mla_attn eh_proj_rows -- --ignored --nocapture --test-threads=1
mod eh_proj_rows {
    use crate::common::Rng;
    use gb_cuda::ling_mla_attn as tc;
    use gb_cuda::mla_gated as sc;
    use gb_cuda::num::{bf16_to_f32, f32_to_bf16};
    use gb_cuda::rt::{DevBuf, Stream};

    /// bf16 values: uniform in [-1, 1) x scale x 2^(-6 .. +6), with exact zeros, so the fp32 partial sums round and a reduction order is visible.
    fn acts(rng: &mut Rng, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|i| {
                let mut v = rng.uniform(-1.0, 1.0) * scale;
                if i % 11 == 0 {
                    v = 0.0;
                } else {
                    v *= (2.0f32).powi(((rng.next_u64() >> 33) % 13) as i32 - 6);
                }
                f32_to_bf16(v)
            })
            .collect()
    }

    /// `w` bf16 `[h][2h]`: each output column j has its own scale 2^(-8 .. +8), so partial sums span many binades. The h half is the negated
    /// e half except at every 64th element (+-1 ulp): in even columns element for element (so a row whose h operand equals its e operand sums,
    /// inside each lane, two huge almost-opposite partial sums to a tiny one), in odd columns across neighbouring 8-element blocks (block b
    /// negates block b^1, so the cancellation is between lane l and lane l^1 and only the final tree sum sees it). The fp32 rounding of those
    /// sums is then visible in the bf16 output: a reduction-order or rounding mutation moves hundreds of outputs, not one in thirty thousand.
    fn weights(rng: &mut Rng, h: usize) -> Vec<u16> {
        let mut w = Vec::with_capacity(h * 2 * h);
        for j in 0..h {
            let sc = (2.0f32).powi((j % 17) as i32 - 8) * 0.05;
            let e_half: Vec<u16> = (0..h).map(|_| f32_to_bf16(rng.uniform(-1.0, 1.0) * sc)).collect();
            w.extend_from_slice(&e_half);
            for c in 0..h {
                let src = if j % 2 == 0 { c } else { (c ^ 8) % h };
                let neg = e_half[src] ^ 0x8000;
                w.push(if c % 64 == 5 { neg.wrapping_add(1) } else { neg });
            }
        }
        w
    }

    /// h operand rows: `r % 3 == 1` repeats its e row (cancels in the even columns), `r % 3 == 2` the e row with neighbouring 8-element blocks
    /// swapped (cancels across lanes in the odd columns), `r % 3 == 0` is an independent draw.
    fn cancelling(rng: &mut Rng, e: &[u16], h: usize, scale: f32) -> Vec<u16> {
        let mut hn = acts(rng, e.len(), scale);
        for r in 0..e.len() / h {
            for c in 0..h {
                match r % 3 {
                    1 => hn[r * h + c] = e[r * h + c],
                    2 => hn[r * h + c] = e[r * h + (c ^ 8) % h],
                    _ => {}
                }
            }
        }
        hn
    }

    /// The documented chain for output (r, j) in plain Rust: lane `l` runs `acc = fma(x, w, acc)` over its `c8 = l, l+32, ..` uint4s (eight
    /// ascending elements each), e half then h half, then the shfl_down tree 16, 8, 4, 2, 1, then one bf16 rounding.
    fn cpu_ref(e: &[u16], hn: &[u16], w: &[u16], h: usize, r: usize, j: usize) -> u16 {
        let h8 = h / 8;
        let mut lane_acc = [0.0f32; 32];
        for (lane, acc_out) in lane_acc.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (x, off) in [(e, 0usize), (hn, h)] {
                let mut c8 = lane;
                while c8 < h8 {
                    for k in 0..8 {
                        let c = c8 * 8 + k;
                        acc = bf16_to_f32(x[r * h + c]).mul_add(bf16_to_f32(w[j * 2 * h + off + c]), acc);
                    }
                    c8 += 32;
                }
            }
            *acc_out = acc;
        }
        for d in [16usize, 8, 4, 2, 1] {
            for i in 0..(32 - d) {
                lane_acc[i] += lane_acc[i + d];
            }
        }
        f32_to_bf16(lane_acc[0])
    }

    struct Fix {
        st: Stream,
        h: usize,
        rows: usize,
        e: DevBuf,
        hn: DevBuf,
        w: DevBuf,
        e_host: Vec<u16>,
        h_host: Vec<u16>,
        w_host: Vec<u16>,
    }

    impl Fix {
        fn new(seed: u64, h: usize, rows: usize) -> Self {
            let mut rng = Rng(seed);
            let e_host = acts(&mut rng, rows * h, 1.0);
            let h_host = cancelling(&mut rng, &e_host, h, 0.7);
            let w_host = weights(&mut rng, h);
            Fix::with(h, rows, e_host, h_host, w_host)
        }

        fn with(h: usize, rows: usize, e_host: Vec<u16>, h_host: Vec<u16>, w_host: Vec<u16>) -> Self {
            Fix {
                st: Stream::new().unwrap(),
                h,
                rows,
                e: DevBuf::from_slice(&e_host).unwrap(),
                hn: DevBuf::from_slice(&h_host).unwrap(),
                w: DevBuf::from_slice(&w_host).unwrap(),
                e_host,
                h_host,
                w_host,
            }
        }

        fn out(&self, n: usize) -> DevBuf {
            DevBuf::alloc(n * self.h * 2).unwrap()
        }

        fn read(&self, y: &DevBuf, n: usize) -> Vec<u16> {
            self.st.sync().unwrap();
            let mut v = vec![0u16; n * self.h];
            y.download(&mut v).unwrap();
            v
        }

        /// The shared kernel on the first `n` rows.
        fn old(&self, n: usize) -> Vec<u16> {
            let mut y = self.out(n);
            sc::eh_proj_cat_bf16(&self.e, &self.hn, &self.w, &mut y, n, self.h, &self.st).unwrap();
            self.read(&y, n)
        }

        /// The production launch on the first `n` rows.
        fn new_rows(&self, n: usize) -> Vec<u16> {
            let mut y = self.out(n);
            tc::eh_proj_cat_bf16(&self.e, &self.hn, &self.w, &mut y, n, self.h, &self.st).unwrap();
            self.read(&y, n)
        }

        /// An explicit tile on the first `n` rows; `None` when the tile is not instantiated.
        fn tile(&self, n: usize, j: usize, r: usize, slab: usize) -> Option<Vec<u16>> {
            let mut y = self.out(n);
            match tc::eh_proj_cat_bf16_tile(&self.e, &self.hn, &self.w, &mut y, n, self.h, j, r, slab, &self.st) {
                Ok(()) => Some(self.read(&y, n)),
                Err(_) => None,
            }
        }

        /// Rows `[lo, lo + n)` as a call of their own (the operands are sliced on the host and uploaded again).
        fn new_slice(&self, lo: usize, n: usize) -> Vec<u16> {
            let h = self.h;
            let f = Fix::with(
                h,
                n,
                self.e_host[lo * h..(lo + n) * h].to_vec(),
                self.h_host[lo * h..(lo + n) * h].to_vec(),
                self.w_host.clone(),
            );
            f.new_rows(n)
        }
    }

    fn bit_eq(a: &[u16], b: &[u16], h: usize, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        if let Some(i) = (0..a.len()).find(|&i| a[i] != b[i]) {
            panic!("{what}: first differing bits at row {} column {}: {:#06x} vs {:#06x}", i / h, i % h, a[i], b[i]);
        }
    }

    const TILES: &[(usize, usize)] = &[(1, 1), (1, 3), (1, 8), (2, 1), (2, 4), (2, 8), (2, 16), (8, 4)];

    #[cfg(not(gb_no_cuda))]
    #[test]
    fn production_launch_equals_the_shared_kernel_for_every_row_count_1_to_70() {
        let fx = Fix::new(11, 4096, 70);
        let want = fx.old(70);
        for n in 1..=70usize {
            let got = fx.new_rows(n);
            bit_eq(&got, &want[..n * 4096], 4096, &format!("production launch, {n} rows"));
        }
        println!("70 row counts: bit-equal to the shared kernel");
    }

    #[cfg(not(gb_no_cuda))]
    #[test]
    fn every_tile_and_slab_equals_the_shared_kernel_at_the_r_boundaries() {
        let fx = Fix::new(12, 4096, 1100);
        let want = fx.old(1100);
        let mut checked = 0;
        for &(j, r) in TILES {
            for slab in [1usize, 5, 24, 25, 1024] {
                let mut counts: Vec<usize> = vec![1, 2, 7, 70, 1100];
                for k in [r.saturating_sub(1), r, r + 1, 2 * r - 1, 2 * r, 2 * r + 1, slab.saturating_sub(1), slab, slab + 1] {
                    if (1..=1100).contains(&k) {
                        counts.push(k);
                    }
                }
                counts.sort_unstable();
                counts.dedup();
                for n in counts {
                    let got = fx.tile(n, j, r, slab).unwrap_or_else(|| panic!("tile j{j} r{r} is not instantiated"));
                    bit_eq(&got, &want[..n * 4096], 4096, &format!("tile j{j} r{r} slab {slab}, {n} rows"));
                    checked += 1;
                }
            }
        }
        println!("{} tile/slab/row-count combinations bit-equal", checked);
    }

    #[cfg(not(gb_no_cuda))]
    #[test]
    fn a_row_alone_equals_the_same_row_inside_wider_calls() {
        let fx = Fix::new(13, 4096, 1100);
        let wide = fx.new_rows(1100);
        let seventy = fx.new_rows(70);
        for i in (0..70).chain([127, 128, 511, 1023, 1024, 1025, 1099]) {
            let alone = fx.new_slice(i, 1);
            bit_eq(&alone, &wide[i * 4096..(i + 1) * 4096], 4096, &format!("row {i} alone vs the 1,100-row call"));
            if i < 70 {
                bit_eq(&alone, &seventy[i * 4096..(i + 1) * 4096], 4096, &format!("row {i} alone vs the 70-row call"));
            }
        }
        // a window of rows starting mid-call (different tile phase) equals the same rows of the whole call
        for (lo, n) in [(3usize, 24usize), (23, 25), (1000, 100)] {
            let win = fx.new_slice(lo, n);
            bit_eq(&win, &wide[lo * 4096..(lo + n) * 4096], 4096, &format!("rows {lo}..{} as a call of their own", lo + n));
        }
    }

    #[cfg(not(gb_no_cuda))]
    #[test]
    fn other_widths_equal_the_shared_kernel() {
        for h in [256usize, 1024, 2048] {
            let fx = Fix::new(14 + h as u64, h, 70);
            let want = fx.old(70);
            for n in [1usize, 5, 24, 25, 48, 49, 70] {
                bit_eq(&fx.new_rows(n), &want[..n * h], h, &format!("h {h}, {n} rows"));
            }
        }
    }

    #[cfg(not(gb_no_cuda))]
    #[test]
    fn both_kernels_equal_the_documented_cpu_chain() {
        let h = 4096;
        let fx = Fix::new(15, h, 3);
        let old = fx.old(3);
        let new = fx.new_rows(3);
        let mut n = 0;
        for r in 0..3 {
            for j in (0..h).step_by(53).chain([0, 1, 7, 8, 4094, 4095]) {
                let want = cpu_ref(&fx.e_host, &fx.h_host, &fx.w_host, h, r, j);
                assert_eq!(old[r * h + j], want, "shared kernel vs the CPU chain at ({r}, {j})");
                assert_eq!(new[r * h + j], want, "row-batched kernel vs the CPU chain at ({r}, {j})");
                n += 1;
            }
        }
        println!("{n} outputs equal the CPU transcription");
    }

    #[cfg(not(gb_no_cuda))]
    #[test]
    fn runs_are_deterministic_and_an_empty_call_is_a_no_op() {
        let fx = Fix::new(16, 4096, 200);
        bit_eq(&fx.new_rows(200), &fx.new_rows(200), 4096, "two runs");
        let mut y = fx.out(1);
        tc::eh_proj_cat_bf16(&fx.e, &fx.hn, &fx.w, &mut y, 0, 4096, &fx.st).unwrap();
    }

    /// The CPU oracle sees the mutations a GPU kernel could suffer: another k order, the h half first, a reduction tree that is not the documented
    /// one, and a missing bf16 rounding all give different bits on this fixture (hundreds of outputs, not one or two), so a kernel mutation is
    /// catchable by the bit comparisons above.
    #[test]
    fn the_oracle_is_sensitive_to_the_orders_it_pins() {
        let h = 1024;
        let rows = 12;
        let mut rng = Rng(17);
        let e = acts(&mut rng, rows * h, 1.0);
        let hn = cancelling(&mut rng, &e, h, 0.7);
        let w = weights(&mut rng, h);
        let chain = |order: u8| -> Vec<u16> {
            let h8 = h / 8;
            let mut out = vec![];
            for r in 0..rows {
                for j in 0..128 {
                    let mut lanes = [0.0f32; 32];
                    for (lane, la) in lanes.iter_mut().enumerate() {
                        let mut acc = 0.0f32;
                        let halves: [(&[u16], usize); 2] = if order == 1 { [(&hn, h), (&e, 0)] } else { [(&e, 0), (&hn, h)] };
                        for (x, off) in halves {
                            let mut c8 = lane;
                            while c8 < h8 {
                                let ks: Vec<usize> = if order == 2 { (0..8).rev().collect() } else { (0..8).collect() };
                                for k in ks {
                                    let c = c8 * 8 + k;
                                    acc = bf16_to_f32(x[r * h + c]).mul_add(bf16_to_f32(w[j * 2 * h + off + c]), acc);
                                }
                                c8 += 32;
                            }
                        }
                        *la = acc;
                    }
                    let deltas: [usize; 5] = if order == 3 { [1, 2, 4, 8, 16] } else { [16, 8, 4, 2, 1] };
                    for d in deltas {
                        for i in 0..(32 - d) {
                            lanes[i] += lanes[i + d];
                        }
                    }
                    out.push(if order == 4 { (lanes[0].to_bits() >> 16) as u16 } else { f32_to_bf16(lanes[0]) });
                }
            }
            out
        };
        let good = chain(0);
        for (order, name) in
            [(1u8, "h half first"), (2, "descending elements"), (3, "ascending tree"), (4, "truncation for the rounding")]
        {
            let bad = chain(order);
            let diff = good.iter().zip(&bad).filter(|(a, b)| a != b).count();
            println!("mutation `{name}`: {diff} of {} outputs differ", good.len());
            assert!(diff >= 20, "mutation `{name}` moves only {diff} outputs on the fixture");
        }
    }

    /// The real L72 eh_proj weight (`model.layers.72.eh_proj.weight`, bf16 [4096, 8192]) with random normalised activations.
    #[cfg(not(gb_no_cuda))]
    #[test]
    #[ignore = "needs the Ling pack: REXL3_LING_DIR"]
    fn the_real_pack_weight_gives_the_same_bits() {
        let Ok(dir) = std::env::var("REXL3_LING_DIR") else {
            eprintln!("REXL3_LING_DIR is not set: not run");
            return;
        };
        let ck = gb_exl3::checkpoint::scan(std::path::Path::new(&dir), false).expect("scan the pack");
        let ent = ck.plain.get("model.layers.72.eh_proj.weight").expect("the L72 eh_proj tensor");
        assert_eq!(ent.shape, vec![4096, 8192]);
        let w = ck.read_u16(ent).expect("read eh_proj");
        let mut rng = Rng(18);
        let rows = 300;
        let e = acts(&mut rng, rows * 4096, 1.0);
        let hn = acts(&mut rng, rows * 4096, 1.0);
        let fx = Fix::with(4096, rows, e, hn, w);
        let want = fx.old(rows);
        for n in [1usize, 2, 3, 4, 23, 24, 25, 70, 299, 300] {
            bit_eq(&fx.new_rows(n), &want[..n * 4096], 4096, &format!("real weight, {n} rows"));
        }
        println!("real L72 eh_proj weight: bit-equal at 10 row counts up to {rows}");
    }

    /// Old vs new cost per row on a 4,096-wide call (`cargo test --release ... bench -- --ignored --nocapture`).
    #[cfg(not(gb_no_cuda))]
    #[test]
    #[ignore = "a measurement, not a check"]
    fn bench_old_vs_new_per_row() {
        let fx = Fix::new(19, 4096, 8191);
        for n in [1usize, 2, 4, 16, 70, 512, 2048, 4096, 8191] {
            let reps = if n <= 16 { 20 } else { 3 };
            let mut y = fx.out(n);
            let mut run = |old: bool| {
                let once = |y: &mut DevBuf| {
                    if old {
                        sc::eh_proj_cat_bf16(&fx.e, &fx.hn, &fx.w, y, n, 4096, &fx.st).unwrap()
                    } else {
                        tc::eh_proj_cat_bf16(&fx.e, &fx.hn, &fx.w, y, n, 4096, &fx.st).unwrap()
                    }
                };
                once(&mut y);
                fx.st.sync().unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    once(&mut y);
                }
                fx.st.sync().unwrap();
                t0.elapsed().as_secs_f64() * 1e3 / reps as f64
            };
            let old = if n <= 2048 { Some(run(true)) } else { None };
            let new = run(false);
            println!(
                "rows {n:5}: old {} ms, new {new:.3} ms ({:.1} us/row)",
                old.map_or("-".into(), |o| format!("{o:.3}")),
                new * 1e3 / n as f64
            );
        }
    }
}
