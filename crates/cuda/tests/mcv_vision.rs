//! The MiniCPM-V-4.7 vision path (`csrc/mcv_vision.cu`, `src/mcv_vision.rs`): kernel tests against f64 references, crop invariance,
//! and the per-stage gates against the Hugging Face model (`tools/minicpmv_vision_oracle.py`).
//!
//! GPU tests print `not run: ...` and pass when there is no device. The checkpoint tests also need `REXL3_MINICPMV47_DIR` (the bf16
//! checkpoint folder, or an EXL3 pack: the tower tensors are copied through unchanged) and, for the oracle gates,
//! `REXL3_MCV_ORACLE_DIR` (default `~/projects/minicpm-vision-oracle`, written by `tools/minicpmv_vision_oracle.py refs`).
//!
//! Metrics: `rel_l2 = ||a - b|| / ||b||`, `rel_max = max|a - b| / max|b|`, `cosine`, `b` the reference.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use gb_cuda::mcv_vision as mv;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use std::path::{Path, PathBuf};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("not run: no CUDA device");
            return;
        }
    };
}

#[derive(Clone, Copy, Debug)]
struct Metric {
    rel_max: f64,
    rel_l2: f64,
    cosine: f64,
}

fn metric(a: &[f32], b: &[f32]) -> Metric {
    assert_eq!(a.len(), b.len(), "metric: lengths differ ({} vs {})", a.len(), b.len());
    assert!(a.iter().all(|x| x.is_finite()), "candidate has non-finite values");
    let (mut dmax, mut bmax, mut d2, mut b2, mut ab, mut a2) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dmax = dmax.max((x - y).abs());
        bmax = bmax.max(y.abs());
        d2 += (x - y) * (x - y);
        b2 += y * y;
        ab += x * y;
        a2 += x * x;
    }
    Metric {
        rel_max: dmax / bmax.max(1e-30),
        rel_l2: d2.sqrt() / b2.sqrt().max(1e-30),
        cosine: ab / (a2.sqrt() * b2.sqrt()).max(1e-30),
    }
}

/// Random fp16-representable values in [lo, hi): bits and widened values.
fn rand_f16(n: usize, seed: u64, lo: f32, hi: f32) -> (Vec<u16>, Vec<f32>) {
    let mut s = seed;
    let mut bits = Vec::with_capacity(n);
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
        let b = f64_to_f16(lo as f64 + (hi - lo) as f64 * u);
        bits.push(b);
        vals.push(f16_to_f64(b) as f32);
    }
    (bits, vals)
}

fn f16s_to_f32(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| f16_to_f64(b) as f32).collect()
}

fn upload_f16(bits: &[u16]) -> DevBuf {
    DevBuf::from_slice(bits).unwrap()
}

// ---------------------------------------------------------------------------------------------------------------- kernels

fn gemm_reference(a: &[f32], b: &[f32], bias: Option<&[f32]>, m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0f64; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut s = bias.map_or(0.0, |bi| bi[col] as f64);
            for kk in 0..k {
                s += a[r * k + kk] as f64 * b[col * k + kk] as f64;
            }
            c[r * n + col] = s;
        }
    }
    c
}

fn gelu_tanh(x: f64) -> f64 {
    0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())
}

fn gelu_erf(x: f64) -> f64 {
    // erf by a series good to ~1e-12 over the range used here
    fn erf(z: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.5 * z.abs());
        let y = 1.0
            - t * (-z * z - 1.26551223
                + t * (1.00002368
                    + t * (0.37409196
                        + t * (0.09678418
                            + t * (-0.18628806
                                + t * (0.27886807
                                    + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
                .exp();
        if z >= 0.0 {
            y
        } else {
            -y
        }
    }
    0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))
}

#[test]
fn gemm_matches_an_f64_reference_for_every_epilogue_and_the_model_shapes() {
    gpu!();
    let st = Stream::new().unwrap();
    // (m, n, k): partial row tiles, the model's real K and N values
    for &(m, n, k) in &[
        (37usize, 1152usize, 608usize),
        (520, 3456, 1152),
        (130, 3456, 1152),
        (9, 4320, 1152),
        (70, 1152, 4320),
        (5, 4608, 4608),
        (3, 2048, 4608),
    ] {
        let (abits, a) = rand_f16(m * k, (m + n) as u64, -1.0, 1.0);
        let (bbits, b) = rand_f16(n * k, (n + k) as u64, -0.1, 0.1);
        let bias: Vec<f32> = rand_f16(n, 5, -0.5, 0.5).1;
        let da = upload_f16(&abits);
        let db = upload_f16(&bbits);
        let dbias = DevBuf::from_slice(&bias).unwrap();
        let want = gemm_reference(&a, &b, Some(&bias), m, n, k);
        for epi in [mv::Epi::BiasF16, mv::Epi::BiasGeluTanhF16, mv::Epi::BiasGeluF16] {
            let mut dc = DevBuf::zeroed(m * n * 2).unwrap();
            mv::gemm(epi, &da, &db, Some(&dbias), &mut dc, 0, m, n, k, n, &st).unwrap();
            st.sync().unwrap();
            let got = f16s_to_f32(&dc.to_vec::<u16>().unwrap());
            let want_e: Vec<f32> = want
                .iter()
                .map(|&v| match epi {
                    mv::Epi::BiasF16 => v as f32,
                    mv::Epi::BiasGeluTanhF16 => gelu_tanh(v) as f32,
                    _ => gelu_erf(v) as f32,
                })
                .collect();
            let mt = metric(&got, &want_e);
            assert!(mt.rel_max < 2e-3 && mt.rel_l2 < 6e-4, "{epi:?} {m}x{n}x{k}: {mt:?}");
        }
        // fp32 outputs: store and in-place residual add
        let init: Vec<f32> = rand_f16(m * n, 7, -2.0, 2.0).1;
        let mut dc = DevBuf::from_slice(&init).unwrap();
        mv::gemm(mv::Epi::BiasAddF32, &da, &db, Some(&dbias), &mut dc, 0, m, n, k, n, &st).unwrap();
        st.sync().unwrap();
        let got = dc.to_vec::<f32>().unwrap();
        let want_add: Vec<f32> = want.iter().zip(&init).map(|(&w, &i)| (w + i as f64) as f32).collect();
        let mt = metric(&got, &want_add);
        assert!(mt.rel_max < 1e-5, "BiasAddF32 {m}x{n}x{k}: {mt:?}");
        let mut dc = DevBuf::zeroed(m * n * 4).unwrap();
        mv::gemm(mv::Epi::StoreF32, &da, &db, Some(&dbias), &mut dc, 0, m, n, k, n, &st).unwrap();
        st.sync().unwrap();
        let mt = metric(&dc.to_vec::<f32>().unwrap(), &want.iter().map(|&v| v as f32).collect::<Vec<_>>());
        assert!(mt.rel_max < 1e-5, "StoreF32 {m}x{n}x{k}: {mt:?}");
    }
}

#[test]
fn layernorm_with_bias_matches_the_formula_at_both_widths() {
    gpu!();
    let st = Stream::new().unwrap();
    for dim in [mv::HIDDEN, mv::WINDOW_HIDDEN, 1000] {
        let rows = 11usize;
        let x: Vec<f32> = rand_f16(rows * dim, 3, -4.0, 6.0).1;
        let w: Vec<f32> = rand_f16(dim, 5, 0.5, 1.5).1;
        let b: Vec<f32> = rand_f16(dim, 9, -0.5, 0.5).1;
        let (dx, dw, db) = (DevBuf::from_slice(&x).unwrap(), DevBuf::from_slice(&w).unwrap(), DevBuf::from_slice(&b).unwrap());
        let mut dy = DevBuf::zeroed(rows * dim * 2).unwrap();
        mv::layernorm(&dx, &dw, &db, &mut dy, rows, dim, mv::LN_EPS, &st).unwrap();
        st.sync().unwrap();
        let got = f16s_to_f32(&dy.to_vec::<u16>().unwrap());
        let mut want = vec![0f32; rows * dim];
        for r in 0..rows {
            let row = &x[r * dim..(r + 1) * dim];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / dim as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / dim as f64;
            for c in 0..dim {
                want[r * dim + c] =
                    (((row[c] as f64 - mean) / (var + mv::LN_EPS as f64).sqrt()) * w[c] as f64 + b[c] as f64) as f32;
            }
        }
        let mt = metric(&got, &want);
        assert!(mt.rel_max < 2e-3, "fp32 input, dim {dim}: {mt:?}");
        // the fp16-input variant on the same values (they are exactly representable)
        let xb: Vec<u16> = x.iter().map(|&v| f64_to_f16(v as f64)).collect();
        let dxh = upload_f16(&xb);
        let mut dy2 = DevBuf::zeroed(rows * dim * 2).unwrap();
        mv::layernorm_f16(&dxh, &dw, &db, &mut dy2, rows, dim, mv::LN_EPS, &st).unwrap();
        st.sync().unwrap();
        let got2 = f16s_to_f32(&dy2.to_vec::<u16>().unwrap());
        assert!(metric(&got2, &want).rel_max < 2e-3, "fp16 input, dim {dim}");
    }
}

#[test]
fn patchify_reads_the_processor_layout() {
    gpu!();
    let st = Stream::new().unwrap();
    let patches = 5usize;
    let w = 14 * patches;
    // bf16 bits of small distinct values: value = (c, i, column) code
    let mut pix = vec![0u16; 3 * 14 * w];
    for c in 0..3 {
        for i in 0..14 {
            for x in 0..w {
                let v = (c * 1000 + i * 100 + x) as f32 / 4096.0;
                pix[(c * 14 + i) * w + x] = (v.to_bits() >> 16) as u16;
            }
        }
    }
    let dp = DevBuf::from_slice(&pix).unwrap();
    let mut out = DevBuf::zeroed(patches * mv::PATCH_PADDED * 2).unwrap();
    mv::patchify(&dp, &mut out, patches, &st).unwrap();
    st.sync().unwrap();
    let got = f16s_to_f32(&out.to_vec::<u16>().unwrap());
    for p in 0..patches {
        for k in 0..mv::PATCH_PADDED {
            let want = if k < mv::PATCH_DIM {
                let (c, r) = (k / 196, k % 196);
                let (i, j) = (r / 14, r % 14);
                let v = f32::from_bits((pix[(c * 14 + i) * w + p * 14 + j] as u32) << 16);
                f16_to_f64(f64_to_f16(v as f64)) as f32
            } else {
                0.0
            };
            assert_eq!(got[p * mv::PATCH_PADDED + k], want, "patch {p} column {k}");
        }
    }
}

/// f64 attention of one crop, head dim 72: qkv [rows, 3456] values -> out [rows, 1152].
fn attention_reference(qkv: &[f32], rows: usize) -> Vec<f32> {
    let scale = 1.0 / (mv::HEAD_DIM as f64).sqrt();
    let ld = 3 * mv::HIDDEN;
    let mut out = vec![0f32; rows * mv::HIDDEN];
    for h in 0..mv::HEADS {
        for i in 0..rows {
            let q = &qkv[i * ld + h * 72..i * ld + h * 72 + 72];
            let mut s: Vec<f64> = (0..rows)
                .map(|j| {
                    let k = &qkv[j * ld + mv::HIDDEN + h * 72..j * ld + mv::HIDDEN + h * 72 + 72];
                    q.iter().zip(k).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() * scale
                })
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut l = 0.0;
            for v in s.iter_mut() {
                *v = (*v - m).exp();
                l += *v;
            }
            for d in 0..72 {
                let mut o = 0.0;
                for j in 0..rows {
                    o += s[j] * qkv[j * ld + 2 * mv::HIDDEN + h * 72 + d] as f64;
                }
                out[i * mv::HIDDEN + h * 72 + d] = (o / l) as f32;
            }
        }
    }
    out
}

#[test]
fn attention_is_per_crop_and_matches_f64() {
    gpu!();
    let st = Stream::new().unwrap();
    // crops of awkward lengths (partial key tiles, a single key, exact tiles)
    let lens = [1usize, 7, 64, 65, 130, 33];
    let rows: usize = lens.iter().sum();
    let (qbits, q) = rand_f16(rows * 3 * mv::HIDDEN, 11, -3.0, 3.0);
    let dq = upload_f16(&qbits);
    let mut crops = Vec::new();
    let mut at = 0i32;
    for &l in &lens {
        crops.extend_from_slice(&[at, l as i32]);
        at += l as i32;
    }
    let dc = DevBuf::from_slice(&crops).unwrap();
    let mut out = DevBuf::zeroed(rows * mv::HIDDEN * 2).unwrap();
    mv::attention(&dq, &mut out, &dc, lens.len(), *lens.iter().max().unwrap(), rows, &st).unwrap();
    st.sync().unwrap();
    let got = f16s_to_f32(&out.to_vec::<u16>().unwrap());
    let mut off = 0usize;
    for (z, &l) in lens.iter().enumerate() {
        let want = attention_reference(&q[off * 3 * mv::HIDDEN..(off + l) * 3 * mv::HIDDEN], l);
        let mt = metric(&got[off * mv::HIDDEN..(off + l) * mv::HIDDEN], &want);
        println!("attention crop {z} ({l} rows): rel_max {:.3e} rel_l2 {:.3e}", mt.rel_max, mt.rel_l2);
        assert!(mt.rel_max < 3e-3 && mt.rel_l2 < 1e-3, "crop {z} ({l} rows): {mt:?}");
        off += l;
    }
    // a crop alone gives the same bits as the crop inside the launch
    let z = 4usize; // the 130-row crop
    let start: usize = lens[..z].iter().sum();
    let alone = &qbits[start * 3 * mv::HIDDEN..(start + lens[z]) * 3 * mv::HIDDEN];
    let da = upload_f16(alone);
    let dc1 = DevBuf::from_slice(&[0i32, lens[z] as i32]).unwrap();
    let mut out1 = DevBuf::zeroed(lens[z] * mv::HIDDEN * 2).unwrap();
    mv::attention(&da, &mut out1, &dc1, 1, lens[z], lens[z], &st).unwrap();
    st.sync().unwrap();
    let a = out1.to_vec::<u16>().unwrap();
    let all = out.to_vec::<u16>().unwrap();
    assert_eq!(a[..], all[start * mv::HIDDEN..(start + lens[z]) * mv::HIDDEN], "crop output depends on the crops beside it");
}

// ---------------------------------------------------------------------------------------------------------------- the tower

fn checkpoint_dir() -> Option<PathBuf> {
    match std::env::var_os("REXL3_MINICPMV47_DIR") {
        Some(v) => Some(PathBuf::from(v)),
        None => {
            eprintln!("not run: REXL3_MINICPMV47_DIR is not set");
            None
        }
    }
}

fn oracle_dir() -> Option<PathBuf> {
    let d = std::env::var_os("REXL3_MCV_ORACLE_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("projects/minicpm-vision-oracle")))?;
    if d.join("manifest.json").exists() {
        Some(d)
    } else {
        eprintln!("not run: no oracle manifest in {}", d.display());
        None
    }
}

fn read_f32(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

struct Case {
    name: String,
    mode: mv::Mode,
    crops: Vec<(usize, usize)>,
    pixels: Vec<u16>,
    dir: PathBuf,
}

fn load_case(dir: &Path, name: &str) -> Case {
    let d = dir.join(name);
    let plan = std::fs::read_to_string(d.join("plan.txt")).unwrap_or_else(|e| panic!("{}: {e}", d.join("plan.txt").display()));
    let mut lines = plan.lines();
    let mode = mv::Mode::parse(lines.next().unwrap().trim()).unwrap();
    let nums: Vec<usize> = lines.next().unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
    let crops = nums.chunks_exact(2).map(|c| (c[0], c[1])).collect();
    let b = std::fs::read(d.join("pixels.u16")).unwrap();
    let pixels = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    Case { name: name.to_string(), mode, crops, pixels, dir: d }
}

fn case_names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("plan.txt").exists())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    v.sort();
    v
}

/// Stages every case is compared at (16x has the merger; 4x does not), the names of the oracle's files.
const STAGES: [&str; 6] = ["embed", "layer6", "merger", "layer26", "post_ln", "projector"];

/// `tools/minicpmv_vision_oracle.py`'s `bitsum64`: the little-endian u32 words (the tail zero padded), the sum of
/// `(word + 1) * (MIX * (index + 1))` modulo 2^64, xor the byte length.
fn bitsum64(bytes: &[u8]) -> u64 {
    const MIX: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut total = 0u64;
    let mut idx = 0u64;
    let mut chunks = bytes.chunks_exact(4);
    for c in &mut chunks {
        idx += 1;
        let w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64 + 1;
        total = total.wrapping_add(w.wrapping_mul(idx.wrapping_mul(MIX)));
    }
    let rest = chunks.remainder();
    if !rest.is_empty() {
        idx += 1;
        let mut w = [0u8; 4];
        w[..rest.len()].copy_from_slice(rest);
        total = total.wrapping_add((u32::from_le_bytes(w) as u64 + 1).wrapping_mul(idx.wrapping_mul(MIX)));
    }
    total ^ bytes.len() as u64
}

/// The committed manifest of the oracle references (`testdata/minicpmv4_7/vision/oracle_manifest.txt`): the checksums that keep a
/// regenerated oracle (another box, another torch build) from silently changing what the gates mean. `REXL3_MCV_ORACLE_SHA=off`
/// reports a mismatch and lets the numeric gates decide.
struct Manifest {
    pixels: std::collections::BTreeMap<String, u64>,
    stages: std::collections::BTreeMap<(String, String), u64>,
}

fn manifest() -> Manifest {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/minicpmv4_7/vision/oracle_manifest.txt");
    let text = std::fs::read_to_string(path).unwrap();
    let mut m = Manifest { pixels: Default::default(), stages: Default::default() };
    for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f[0] {
            "case" => {
                m.pixels.insert(f[1].to_string(), u64::from_str_radix(f[4], 16).unwrap());
            }
            "stage" => {
                m.stages.insert((f[1].to_string(), f[2].to_string()), u64::from_str_radix(f[4], 16).unwrap());
            }
            other => panic!("oracle_manifest.txt: unknown line kind {other}"),
        }
    }
    m
}

fn check_reference(man: &Manifest, c: &Case, stage: &str, bytes: &[u8], problems: &mut Vec<String>) {
    let want = match stage {
        "pixels" => man.pixels.get(&c.name),
        _ => man.stages.get(&(c.name.clone(), stage.to_string())),
    };
    match want {
        None => problems.push(format!("{}/{stage}: not in the manifest", c.name)),
        Some(&w) if w != bitsum64(bytes) => problems.push(format!(
            "{}/{stage}: reference bits differ from the manifest ({:016x} against {w:016x})",
            c.name,
            bitsum64(bytes)
        )),
        Some(_) => {}
    }
}

fn run(w: &mv::Weights, ws: &mut mv::Workspace, c: &Case, st: &Stream) -> (mv::Plan, mv::Trace, Vec<f32>) {
    let plan = mv::Plan::new(&c.crops, c.mode, ws.max_patches()).unwrap();
    assert_eq!(c.pixels.len(), 3 * 14 * 14 * plan.patches, "{}: pixel count", c.name);
    let dp = DevBuf::from_slice(&c.pixels).unwrap();
    let mut tr = mv::Trace::default();
    mv::forward(w, ws, &dp, &plan, Some(&mut tr), st).unwrap();
    st.sync().unwrap();
    let mut out = vec![0f32; plan.tokens * mv::LLM_HIDDEN];
    ws.out().download_range(0, &mut out).unwrap();
    (plan, tr, out)
}

#[test]
fn the_weights_convert_without_overflow() {
    gpu!();
    let Some(dir) = checkpoint_dir() else { return };
    let w = mv::Weights::load(&dir).unwrap();
    let r = w.report();
    println!("convert report: {r:?}; device bytes {}", w.bytes());
    assert_eq!(r.overflow, 0, "a weight overflows fp16");
    assert_eq!(r.tensors, 27 * 6 + 1 + 6 + 2, "matrices converted: 27 layers x (q, k, v, out, fc1, fc2), the patch embedding, the merger (q, k, v, out, linear_1, linear_2) and the projector (2)");
}

#[test]
fn per_stage_against_the_hugging_face_model() {
    gpu!();
    let (Some(dir), Some(odir)) = (checkpoint_dir(), oracle_dir()) else { return };
    let st = Stream::new().unwrap();
    let w = mv::Weights::load(&dir).unwrap();
    let mut ws = mv::Workspace::new(16 * 1024).unwrap();
    let names = case_names(&odir);
    assert!(names.len() >= 10, "{} oracle cases in {}", names.len(), odir.display());
    // gates: the engine's distance to the fp32 truth, per stage (see docs/MINICPMV4_7_INTEGRATION_PLAN.md "E5a results")
    let gate_fp32 = |stage: &str| -> f64 {
        match stage {
            // about twice the worst of the 17 oracle cases (4.0e-7, 4.6e-4, 5.3e-4, 1.4e-3, 3.8e-4, 2.0e-3), against bf16 floors of
            // 2e-3 to 4e-2: a tower 5x less accurate than this one passes the floors and fails these
            "embed" => 1.0e-6,
            "layer6" => 1.0e-3,
            "merger" => 1.1e-3,
            "layer26" => 3.0e-3,
            "post_ln" => 8.0e-4,
            "projector" => 4.0e-3,
            _ => unreachable!(),
        }
    };
    let mut worst = std::collections::BTreeMap::<&str, f64>::new();
    let mut failures = Vec::new();
    let man = manifest();
    let mut ref_problems: Vec<String> = Vec::new();
    assert_eq!(names.len(), man.pixels.len(), "the oracle directory and the committed manifest list different cases");
    println!("{:<28} {:>8} {:>10} {:>10} {:>10} {:>10}", "case/stage", "rows", "eng|fp32", "eng|bf16", "bf16|fp32", "cosine");
    for n in &names {
        let c = load_case(&odir, n);
        check_reference(&man, &c, "pixels", &std::fs::read(c.dir.join("pixels.u16")).unwrap(), &mut ref_problems);
        let (plan, tr, out) = run(&w, &mut ws, &c, &st);
        for stage in STAGES {
            let f32p = c.dir.join(format!("{stage}.fp32.f32"));
            if !f32p.exists() {
                assert!(stage == "merger" && c.mode == mv::Mode::X4, "{n}: no {stage}");
                continue;
            }
            check_reference(&man, &c, stage, &std::fs::read(&f32p).unwrap(), &mut ref_problems);
            let want32 = read_f32(&f32p);
            let want16 = read_f32(&c.dir.join(format!("{stage}.bf16.f32")));
            let got: &[f32] =
                if stage == "projector" { &out } else { tr.stage(stage).unwrap_or_else(|| panic!("{n}: engine has no {stage}")) };
            let (m32, m16, floor) = (metric(got, &want32), metric(got, &want16), metric(&want16, &want32));
            println!(
                "{:<28} {:>8} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.6}",
                format!("{n}/{stage}"),
                got.len() / if stage == "projector" { 2048 } else { 1152 },
                m32.rel_l2,
                m16.rel_l2,
                floor.rel_l2,
                m32.cosine
            );
            let e = worst.entry(stage).or_insert(0.0);
            *e = e.max(m32.rel_l2);
            if m32.rel_l2 > gate_fp32(stage) {
                failures.push(format!("{n}/{stage}: rel_l2 vs fp32 {:.3e} > gate {:.1e}", m32.rel_l2, gate_fp32(stage)));
            }
            // against the bf16 model the engine must sit inside the bf16 floor's neighbourhood (never more than 1.5x of it and a margin)
            if m16.rel_l2 > 1.5 * floor.rel_l2 + 2e-3 {
                failures.push(format!(
                    "{n}/{stage}: rel_l2 vs bf16 {:.3e} exceeds 1.5 x floor {:.3e} + 2e-3",
                    m16.rel_l2, floor.rel_l2
                ));
            }
        }
        assert_eq!(plan.tokens * mv::LLM_HIDDEN, out.len());
        println!("{n}: absmax of the fp16 intermediates {:?}", tr.absmax);
        for (k, v) in &tr.absmax {
            assert!(
                *v < 30000.0 || k == "h",
                "{n}: fp16 tensor {k} reaches {v}: too close to 65504, the fp32 fallback is needed"
            );
        }
    }
    println!("worst engine|fp32 per stage: {worst:?}");
    if std::env::var("REXL3_MCV_ORACLE_SHA").map(|v| v == "off").unwrap_or(false) {
        for p in &ref_problems {
            println!("reference manifest (reported only): {p}");
        }
    } else {
        assert!(ref_problems.is_empty(), "{}", ref_problems.join("\n"));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn a_crop_does_not_depend_on_the_crops_beside_it() {
    gpu!();
    let Some(dir) = checkpoint_dir() else { return };
    let st = Stream::new().unwrap();
    let w = mv::Weights::load(&dir).unwrap();
    let mut ws = mv::Workspace::new(8 * 1024).unwrap();
    // pseudo-random bf16 pixels in [-1, 1] for four crops of different grids
    let grids = [(32usize, 32usize), (16, 24), (40, 12), (8, 8)];
    for mode in [mv::Mode::X16, mv::Mode::X4] {
        let crop_pixels = |g: (usize, usize), seed: u64| -> Vec<u16> {
            let n = g.0 * g.1;
            let mut s = seed;
            // packed layout [3, 14, 14 * n]: build per crop then splice below
            (0..3 * 14 * 14 * n)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let v = (((s >> 33) % 255) as f32 - 127.5) / 127.5;
                    (v.to_bits() >> 16) as u16
                })
                .collect()
        };
        // packed tensors of several crops concatenate along the width: [c][i][crop0 cols | crop1 cols | ...]
        let pack = |parts: &[(Vec<u16>, usize)]| -> Vec<u16> {
            let total: usize = parts.iter().map(|p| p.1 * 14).sum();
            let mut out = vec![0u16; 3 * 14 * total];
            let mut col = 0;
            for (px, n) in parts {
                let w = n * 14;
                for ci in 0..3 * 14 {
                    out[ci * total + col..ci * total + col + w].copy_from_slice(&px[ci * w..(ci + 1) * w]);
                }
                col += w;
            }
            out
        };
        let parts: Vec<(Vec<u16>, usize)> =
            grids.iter().enumerate().map(|(k, &g)| (crop_pixels(g, 100 + k as u64), g.0 * g.1)).collect();
        let all = {
            let plan = mv::Plan::new(&grids, mode, ws.max_patches()).unwrap();
            let dp = DevBuf::from_slice(&pack(&parts)).unwrap();
            mv::forward(&w, &mut ws, &dp, &plan, None, &st).unwrap();
            st.sync().unwrap();
            let mut out = vec![0f32; plan.tokens * mv::LLM_HIDDEN];
            ws.out().download_range(0, &mut out).unwrap();
            (plan, out)
        };
        for (k, &g) in grids.iter().enumerate() {
            let plan = mv::Plan::new(&[g], mode, ws.max_patches()).unwrap();
            let dp = DevBuf::from_slice(&pack(&parts[k..k + 1])).unwrap();
            mv::forward(&w, &mut ws, &dp, &plan, None, &st).unwrap();
            st.sync().unwrap();
            let mut alone = vec![0f32; plan.tokens * mv::LLM_HIDDEN];
            ws.out().download_range(0, &mut alone).unwrap();
            let t0 = all.0.token0[k] * mv::LLM_HIDDEN;
            let inside = &all.1[t0..t0 + alone.len()];
            let same = alone.iter().zip(inside).all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "{mode:?}: crop {k} {g:?} differs when it shares the launch with other crops");
        }
    }
}

/// The latency table of `docs/MINICPMV4_7_INTEGRATION_PLAN.md` ("E5a results"): GPU time of the whole vision path per image shape and
/// mode, best of five after two warm-ups, with the achieved fraction of the `mma.sync` peak (125 TFLOP/s measured on this box by
/// `nemotron_vit_perf::bench_ceilings`) and of the weight-streaming bound (the tower's bytes at 273 GB/s). Run alone under the
/// exclusive lock: `REXL3_MINICPMV47_DIR=... cargo test --release -p gb-cuda --test mcv_vision -- --ignored --nocapture bench_timing`.
#[test]
#[ignore]
fn bench_timing_table() {
    gpu!();
    let Some(dir) = checkpoint_dir() else { return };
    let st = Stream::new().unwrap();
    let w = mv::Weights::load(&dir).unwrap();
    let mut ws = mv::Workspace::new(12 * 1024).unwrap();
    let bw_ms = w.bytes() as f64 / 273e9 * 1e3;
    println!("weights {} bytes: {bw_ms:.2} ms at 273 GB/s", w.bytes());
    println!(
        "{:<6} {:>5} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "mode", "crops", "patches", "embed", "early", "merger", "late", "proj", "total ms", "TFLOP/s", "of peak"
    );
    for mode in [mv::Mode::X16, mv::Mode::X4] {
        for ncrops in [1usize, 4, 10] {
            let crops = vec![(32usize, 32usize); ncrops];
            let plan = mv::Plan::new(&crops, mode, ws.max_patches()).unwrap();
            let px: Vec<u16> =
                (0..3 * 14 * 14 * plan.patches).map(|i| (((i * 2654435761usize) >> 12) as u16 & 0x007f) | 0x3e80).collect();
            let dp = DevBuf::from_slice(&px).unwrap();
            for _ in 0..2 {
                mv::forward_timed(&w, &mut ws, &dp, &plan, &st).unwrap();
            }
            let mut best: Option<mv::Timing> = None;
            for _ in 0..5 {
                let t = mv::forward_timed(&w, &mut ws, &dp, &plan, &st).unwrap();
                if best.as_ref().is_none_or(|b| t.total_ms < b.total_ms) {
                    best = Some(t);
                }
            }
            let t = best.unwrap();
            let g = |n: &str| t.groups.iter().filter(|(k, _)| *k == n).map(|(_, v)| *v).sum::<f64>();
            let tflops = plan.flops() / (t.total_ms * 1e-3) / 1e12;
            println!(
                "{:<6} {:>5} {:>8} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>8.1} {:>7.0}%",
                format!("{mode:?}"),
                ncrops,
                plan.patches,
                g("embed"),
                g("early"),
                g("merger"),
                g("late"),
                g("projector"),
                t.total_ms,
                tflops,
                100.0 * tflops / 125.0
            );
        }
    }
}
