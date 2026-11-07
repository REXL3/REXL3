//! The P2 MLA absorb / expand (`csrc/glm_p2_mla.cu`, `gb_cuda::glm_p2_mla`; `docs/GLM_PREFILL_P2.md` section 12) against a float64
//! reference of the same fp16 operands with P1's fp32 chains (`mla::absorb` / `mla::expand`) beside it, and its bit rules: a row's bits
//! do not depend on the rows of the call, where the row sits in a tile, the row stride of q or the run.
//!
//! Inputs are synthetic with GLM's shapes (64 heads, 256 nope dims, a 512-wide latent), fp16, a few rows scaled up 16x (a wide dynamic
//! range inside one call). Run under the GPU lock:
//!
//!     flock -x /tmp/rexl3-gpu.lock cargo test --release -p gb-cuda --test glm_p2_mla -- --nocapture --test-threads=1
#![cfg(not(gb_no_cuda))]
#![allow(clippy::needless_range_loop)]

mod common;

use common::Rng;
use gb_cuda::exl3_linear::Dtype;
use gb_cuda::glm_p2_mla;
use gb_cuda::mla::{self, MlaWeights, LATENT_DIM, NOPE_DIM, N_HEADS};
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use std::time::Instant;

macro_rules! need_gpu {
    () => {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

const QD: usize = N_HEADS * NOPE_DIM; // a q row (absorb input) and an output row of the expand
const LD: usize = N_HEADS * LATENT_DIM; // a q_lat / o_lat row

fn normal(rng: &mut Rng) -> f32 {
    (0..4).map(|_| rng.uniform(-1.0, 1.0)).sum::<f32>() * 0.866
}

/// `n` fp16 values ~ N(0, scale), rows of `row` values, every 7th row 16x larger.
fn acts(rng: &mut Rng, rows: usize, row: usize, scale: f32) -> Vec<u16> {
    let mut v = Vec::with_capacity(rows * row);
    for r in 0..rows {
        let s = if r % 7 == 3 { scale * 16.0 } else { scale };
        for _ in 0..row {
            v.push(f64_to_f16((normal(rng) * s) as f64));
        }
    }
    v
}

/// A device copy of `v` that [`DevBuf::subview`] can cut (a view of a block of its own).
fn up(v: &[u16]) -> DevBuf {
    let mut b = DevBuf::zeroed_view(v.len() * 2).unwrap();
    b.upload(v).unwrap();
    b
}

struct W {
    dev: MlaWeights,
    uk: Vec<u16>,
    uvt: Vec<u16>,
}

fn weights(rng: &mut Rng) -> W {
    let uk: Vec<u16> = (0..N_HEADS * NOPE_DIM * LATENT_DIM).map(|_| f64_to_f16((normal(rng) * 0.05) as f64)).collect();
    let uvt: Vec<u16> = (0..N_HEADS * LATENT_DIM * NOPE_DIM).map(|_| f64_to_f16((normal(rng) * 0.05) as f64)).collect();
    let dev = MlaWeights { uk: DevBuf::from_slice(&uk).unwrap(), uvt: DevBuf::from_slice(&uvt).unwrap(), dtype: Dtype::F16 };
    W { dev, uk, uvt }
}

/// float64 absorb: q `[rows][qs]`, out `[rows][64][512]`.
fn ref_absorb(q: &[u16], qs: usize, w: &W, rows: usize) -> Vec<f64> {
    let mut out = vec![0f64; rows * LD];
    for r in 0..rows {
        for h in 0..N_HEADS {
            let qr: Vec<f64> = (0..NOPE_DIM).map(|d| f16_to_f64(q[r * qs + h * NOPE_DIM + d])).collect();
            let o = &mut out[(r * N_HEADS + h) * LATENT_DIM..][..LATENT_DIM];
            for d in 0..NOPE_DIM {
                let wr = &w.uk[(h * NOPE_DIM + d) * LATENT_DIM..][..LATENT_DIM];
                for c in 0..LATENT_DIM {
                    o[c] += qr[d] * f16_to_f64(wr[c]);
                }
            }
        }
    }
    out
}

/// float64 expand: o_lat `[rows][64][512]`, out `[rows][64 * 256]`.
fn ref_expand(ol: &[u16], w: &W, rows: usize) -> Vec<f64> {
    let mut out = vec![0f64; rows * QD];
    for r in 0..rows {
        for h in 0..N_HEADS {
            let o = &mut out[r * QD + h * NOPE_DIM..][..NOPE_DIM];
            for c in 0..LATENT_DIM {
                let x = f16_to_f64(ol[(r * N_HEADS + h) * LATENT_DIM + c]);
                let wr = &w.uvt[(h * LATENT_DIM + c) * NOPE_DIM..][..NOPE_DIM];
                for n in 0..NOPE_DIM {
                    o[n] += x * f16_to_f64(wr[n]);
                }
            }
        }
    }
    out
}

/// (relative RMS error, fraction of outputs equal to the reference rounded to fp16)
fn err(got: &[u16], want: &[f64]) -> (f64, f64) {
    let (mut e, mut n, mut hit) = (0f64, 0f64, 0usize);
    for (g, w) in got.iter().zip(want) {
        let d = f16_to_f64(*g) - w;
        e += d * d;
        n += w * w;
        hit += (*g == f64_to_f16(*w)) as usize;
    }
    ((e / n).sqrt(), hit as f64 / want.len() as f64)
}

fn absorb(st: &Stream, p2: bool, q: &DevBuf, qs: usize, w: &W, rows: usize) -> Vec<u16> {
    let mut out = DevBuf::zeroed(rows.max(1) * LD * 2).unwrap();
    if p2 {
        glm_p2_mla::absorb(q, qs, &w.dev, &mut out, rows, st).unwrap();
    } else {
        mla::absorb(q, qs, &w.dev, &mut out, rows, st).unwrap();
    }
    st.sync().unwrap();
    out.to_vec::<u16>().unwrap()[..rows * LD].to_vec()
}

fn expand(st: &Stream, p2: bool, ol: &DevBuf, w: &W, rows: usize) -> Vec<u16> {
    let mut out = DevBuf::zeroed(rows.max(1) * QD * 2).unwrap();
    if p2 {
        glm_p2_mla::expand(ol, &w.dev, &mut out, rows, st).unwrap();
    } else {
        mla::expand(ol, &w.dev, &mut out, rows, st).unwrap();
    }
    st.sync().unwrap();
    out.to_vec::<u16>().unwrap()[..rows * QD].to_vec()
}

#[test]
fn p2_absorb_and_expand_are_as_close_to_f64_as_p1s_chains() {
    need_gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(0x0d_ab_50_4b);
    let w = weights(&mut rng);
    for rows in [1usize, 37, 128] {
        let q = acts(&mut rng, rows, QD, 1.0);
        let qd = up(&q);
        let want = ref_absorb(&q, QD, &w, rows);
        let (e1, h1) = err(&absorb(&st, false, &qd, QD, &w, rows), &want);
        let (e2, h2) = err(&absorb(&st, true, &qd, QD, &w, rows), &want);
        eprintln!("absorb rows {rows:>3}: rel RMS P1 {e1:.3e} P2 {e2:.3e}; == fp16(f64) P1 {:.4} P2 {:.4}", h1, h2);
        assert!(e2 <= e1 * 1.05 + 1e-9, "absorb: P2 {e2:e} against P1 {e1:e}");
        let ol = acts(&mut rng, rows, LD, 0.5);
        let od = up(&ol);
        let want = ref_expand(&ol, &w, rows);
        let (e1, h1) = err(&expand(&st, false, &od, &w, rows), &want);
        let (e2, h2) = err(&expand(&st, true, &od, &w, rows), &want);
        eprintln!("expand rows {rows:>3}: rel RMS P1 {e1:.3e} P2 {e2:.3e}; == fp16(f64) P1 {:.4} P2 {:.4}", h1, h2);
        assert!(e2 <= e1 * 1.05 + 1e-9, "expand: P2 {e2:e} against P1 {e1:e}");
    }
}

#[test]
fn a_rows_bits_depend_on_that_row_alone() {
    need_gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(0x0d_b175);
    let w = weights(&mut rng);
    let rows = 128;
    let q = acts(&mut rng, rows, QD, 1.0);
    let ol = acts(&mut rng, rows, LD, 0.5);
    let (qd, od) = (up(&q), up(&ol));
    let a_all = absorb(&st, true, &qd, QD, &w, rows);
    let e_all = expand(&st, true, &od, &w, rows);
    assert_eq!(a_all, absorb(&st, true, &qd, QD, &w, rows), "absorb: two runs");
    assert_eq!(e_all, expand(&st, true, &od, &w, rows), "expand: two runs");
    // calls of other row counts over the same rows (a row lands in another tile, at another place in it, padded otherwise)
    for splits in [vec![37usize, 91], vec![1, 63, 64], vec![65, 63], vec![16; 8], vec![127, 1]] {
        let mut a0 = 0usize;
        for &n in &splits {
            let qv = qd.subview(a0 * QD * 2, n * QD * 2);
            let ov = od.subview(a0 * LD * 2, n * LD * 2);
            assert_eq!(
                absorb(&st, true, &qv, QD, &w, n),
                a_all[a0 * LD..(a0 + n) * LD],
                "absorb rows {a0}..{} of {splits:?}",
                a0 + n
            );
            assert_eq!(
                expand(&st, true, &ov, &w, n),
                e_all[a0 * QD..(a0 + n) * QD],
                "expand rows {a0}..{} of {splits:?}",
                a0 + n
            );
            a0 += n;
        }
    }
    // the same rows in another order: each row's bits follow it
    let perm: Vec<usize> = (0..rows).map(|i| (i * 53 + 11) % rows).collect();
    let qp: Vec<u16> = perm.iter().flat_map(|&r| q[r * QD..(r + 1) * QD].to_vec()).collect();
    let ap = absorb(&st, true, &up(&qp), QD, &w, rows);
    for (i, &r) in perm.iter().enumerate() {
        assert_eq!(ap[i * LD..(i + 1) * LD], a_all[r * LD..(r + 1) * LD], "absorb: row {r} at {i}");
    }
    // q with a wider row stride (P1's absorb takes one): the same bits
    let qs = QD + 64;
    let mut qw = vec![0u16; rows * qs];
    for r in 0..rows {
        qw[r * qs..r * qs + QD].copy_from_slice(&q[r * QD..(r + 1) * QD]);
        for j in QD..qs {
            qw[r * qs + j] = f64_to_f16(1000.0);
        }
    }
    assert_eq!(absorb(&st, true, &up(&qw), qs, &w, rows), a_all, "absorb: row stride {qs}");
}

#[test]
fn bench_p2_against_p1() {
    need_gpu!();
    let st = Stream::new().unwrap();
    let mut rng = Rng(0x0d_be4c);
    let w = weights(&mut rng);
    for rows in [128usize, 512] {
        let qd = up(&acts(&mut rng, rows, QD, 1.0));
        let od = up(&acts(&mut rng, rows, LD, 0.5));
        let mut ql = DevBuf::zeroed(rows * LD * 2).unwrap();
        let mut out = DevBuf::zeroed(rows * QD * 2).unwrap();
        let flops = 2.0 * rows as f64 * N_HEADS as f64 * NOPE_DIM as f64 * LATENT_DIM as f64;
        let iters = 200;
        for p2 in [false, true] {
            for which in ["absorb", "expand"] {
                let mut run = |st: &Stream| {
                    if which == "absorb" {
                        if p2 {
                            glm_p2_mla::absorb(&qd, QD, &w.dev, &mut ql, rows, st).unwrap()
                        } else {
                            mla::absorb(&qd, QD, &w.dev, &mut ql, rows, st).unwrap()
                        }
                    } else if p2 {
                        glm_p2_mla::expand(&od, &w.dev, &mut out, rows, st).unwrap()
                    } else {
                        mla::expand(&od, &w.dev, &mut out, rows, st).unwrap()
                    }
                };
                for _ in 0..20 {
                    run(&st);
                }
                st.sync().unwrap();
                let t = Instant::now();
                for _ in 0..iters {
                    run(&st);
                }
                st.sync().unwrap();
                let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
                eprintln!(
                    "{} {which} {rows} rows: {us:.1} us ({:.1} us a 128 rows), {:.1} TFLOP/s",
                    if p2 { "P2" } else { "P1" },
                    us * 128.0 / rows as f64,
                    flops / us / 1e6
                );
            }
        }
    }
}

/// The kernel's golden hashes (`REXL3_GOLDEN_OUT`, `scripts/gpu_gate.sh`'s GLM line; `testdata/gpu/golden_glm_p2_mla.txt`): absorb and
/// expand on fixed synthetic inputs at 1, 37 and 128 rows.
#[test]
#[ignore]
fn golden_glm_p2_mla_hashes() {
    need_gpu!();
    let path = std::env::var("REXL3_GOLDEN_OUT").expect("REXL3_GOLDEN_OUT");
    let st = Stream::new().unwrap();
    let h16 = |v: &[u16]| gb_exl3::fnv1a(v.iter().flat_map(|x| x.to_le_bytes()));
    let mut lines = vec![format!("arith {}", glm_p2_mla::ARITH_ID)];
    let mut rng = Rng(0x601d_0d0d);
    let w = weights(&mut rng);
    for rows in [1usize, 37, 128] {
        let qd = up(&acts(&mut rng, rows, QD, 1.0));
        let od = up(&acts(&mut rng, rows, LD, 0.5));
        let a = absorb(&st, true, &qd, QD, &w, rows);
        let e = expand(&st, true, &od, &w, rows);
        lines.push(format!("p2mla rows={rows} absorb {:016x} expand {:016x}", h16(&a), h16(&e)));
    }
    std::fs::write(&path, lines.join("\n") + "\n").unwrap();
    eprintln!("wrote {} golden lines to {path}", lines.len());
}
