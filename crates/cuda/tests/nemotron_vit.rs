//! The Nemotron 3.5 Super VL vision tower (`csrc/nemotron_vit.cu`) against the pack's reference dumps.
//!
//! GPU tests print `not run: ...` and pass when there is no device. The reference tests also need the pack and the dumps:
//! `REXL3_NEMO_OMNI_DIR` (or `REXL3_NEMOTRON_PACK`) is the pack directory (the vision tensors are read through its index), and
//! `REXL3_NEMO_VIT_REF` (or `REXL3_NEMOTRON_VIT_REF`) is a dump directory (`ref_512x512.safetensors`, `ref_320x448.safetensors`, from
//! `tools/dump_radio_reference.py`; node 4 has them under `~/projects/nemo-vision-ref/run1`). The committed fixture
//! `testdata/nemotron_omni/vision/radio_ref_fixture.safetensors` supplies the per-stage rows.
//!
//! Metrics follow the reference's: `rel_max = max|a - b| / max|b|`, `rel_l2 = ||a - b|| / ||b||`, with `b` the reference.
#![cfg(not(gb_no_cuda))]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use gb_cuda::nemotron_vit as nv;
use gb_cuda::{device_available, DevBuf, Stream};
use gb_exl3::checkpoint::read_header_checked;
use gb_exl3::f16::{f16_to_f64, f64_to_f16};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

macro_rules! gpu {
    () => {
        if !device_available() {
            eprintln!("not run: no CUDA device");
            return;
        }
    };
}

/// The first variable that is set (the first name is the primary one). A test that cannot run says `not run: REXL3_NEMO_...`, the form
/// scripts/gpu_gate.sh does not count as a skip (the Nemotron pack tests run in scripts/gpu_gate_nemotron.sh).
macro_rules! env_or_skip {
    ($first:expr $(, $alt:expr)*) => {
        match [$first $(, $alt)*].iter().find_map(|n| std::env::var_os(n)) {
            Some(v) => PathBuf::from(v),
            None => {
                eprintln!("not run: {} is not set", $first);
                return;
            }
        }
    };
}

const FIXTURE_DIR: &str = "testdata/nemotron_omni/vision";

/// Accept thresholds, each about twice the measured bf16-vs-fp32 noise floor of the reference (README, "Noise floor"): the
/// floor is rel_l2 1.4e-2 to 1.6e-2 at the tower output, and rel_max 4.0e-2 / 5.0e-2 and rel_l2 3.9e-2 at mlp1.
const GATE_TOWER_REL_L2: f64 = 3.2e-2;
const GATE_MLP1_REL_MAX: f64 = 1.0e-1;
const GATE_MLP1_REL_L2: f64 = 8.0e-2;
/// Per-stage gate on the fixture rows (the stages are the same computation, so they sit near the tower's floor).
const GATE_STAGE_REL_L2: f64 = 4.0e-2;

#[derive(Clone, Copy, Debug)]
struct Metric {
    rel_max: f64,
    rel_l2: f64,
}

fn metric(a: &[f32], b: &[f32]) -> Metric {
    assert_eq!(a.len(), b.len(), "metric: lengths differ");
    assert!(a.iter().all(|x| x.is_finite()), "candidate has non-finite values");
    let (mut dmax, mut bmax, mut d2, mut b2) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dmax = dmax.max((x - y).abs());
        bmax = bmax.max(y.abs());
        d2 += (x - y) * (x - y);
        b2 += y * y;
    }
    Metric { rel_max: dmax / bmax.max(1e-30), rel_l2: d2.sqrt() / b2.sqrt().max(1e-30) }
}

/// One tensor of a single safetensors file as f32 (BF16 and F32), with its shape. Only that tensor's bytes are read. The dump
/// files store tensor names without the shape tag (the tag is the file name), so each file is read on its own.
fn st_f32(path: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    let (hs, _data_len, header) = read_header_checked(path).unwrap();
    let v = header.get(name).unwrap_or_else(|| panic!("{name} is not in {}", path.display()));
    let dtype = v.get("dtype").and_then(|d| d.as_str()).unwrap().to_string();
    let shape: Vec<usize> =
        v.get("shape").and_then(|s| s.as_array()).unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
    let off = v.get("data_offsets").and_then(|o| o.as_array()).unwrap();
    let (a, b) = (off[0].as_u64().unwrap(), off[1].as_u64().unwrap());
    let mut f = std::fs::File::open(path).unwrap();
    f.seek(SeekFrom::Start(8 + hs + a)).unwrap();
    let mut bytes = vec![0u8; (b - a) as usize];
    f.read_exact(&mut bytes).unwrap();
    let vals = match dtype.as_str() {
        "BF16" => bytes.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        "F32" => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        other => panic!("{name}: unexpected dtype {other}"),
    };
    (vals, shape)
}

fn f16_bits_to_f32(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| f16_to_f64(b) as f32).collect()
}

/// Random fp16-representable values in [lo, hi).
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

#[test]
fn layernorm_with_bias_matches_the_formula() {
    gpu!();
    let st = Stream::new().unwrap();
    let (rows, dim) = (7usize, nv::HIDDEN);
    let (_, x) = rand_f16(rows * dim, 3, -4.0, 6.0);
    let (_, w) = rand_f16(dim, 5, 0.5, 1.5);
    let (_, b) = rand_f16(dim, 9, -0.5, 0.5);
    let dx = DevBuf::from_slice(&x).unwrap();
    let dw = DevBuf::from_slice(&w).unwrap();
    let db = DevBuf::from_slice(&b).unwrap();
    let mut dy = DevBuf::zeroed(rows * dim * 2).unwrap();
    nv::layernorm(&dx, 0, &dw, &db, &mut dy, rows, dim, nv::LN_EPS, &st).unwrap();
    st.sync().unwrap();
    let got = f16_bits_to_f32(&dy.to_vec::<u16>().unwrap());
    let mut want = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / dim as f64;
        let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / dim as f64;
        for c in 0..dim {
            want[r * dim + c] = (((row[c] as f64 - mean) / (var + nv::LN_EPS as f64).sqrt()) * w[c] as f64 + b[c] as f64) as f32;
        }
    }
    let m = metric(&got, &want);
    println!("layernorm rel_max {:.3e} rel_l2 {:.3e}", m.rel_max, m.rel_l2);
    assert!(m.rel_max < 2e-3, "{m:?}");
}

#[test]
fn attention_matches_a_cpu_reference_with_a_partial_tile() {
    gpu!();
    let st = Stream::new().unwrap();
    // 75 tokens: one full 64-key tile and a partial one; the query tail is partly past the end too
    let t = 75usize;
    let ld = 3 * nv::HIDDEN;
    let (_, qkv) = rand_f16(t * ld, 11, -1.5, 1.5);
    let dq = DevBuf::from_slice(&qkv.iter().map(|&v| f64_to_f16(v as f64)).collect::<Vec<u16>>()).unwrap();
    let mut dout = DevBuf::zeroed(t * nv::HIDDEN * 2).unwrap();
    nv::attention(&dq, &mut dout, t, &st).unwrap();
    st.sync().unwrap();
    let got = f16_bits_to_f32(&dout.to_vec::<u16>().unwrap());
    let scale = 1.0 / (nv::HEAD_DIM as f64).sqrt();
    let mut want = vec![0f32; t * nv::HIDDEN];
    for head in 0..nv::HEADS {
        for i in 0..t {
            let q = |d: usize| qkv[i * ld + head * nv::HEAD_DIM + d] as f64;
            let mut s = vec![0f64; t];
            for j in 0..t {
                s[j] = (0..nv::HEAD_DIM).map(|d| q(d) * qkv[j * ld + nv::HIDDEN + head * nv::HEAD_DIM + d] as f64).sum::<f64>()
                    * scale;
            }
            let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let p: Vec<f64> = s.iter().map(|&v| (v - mx).exp()).collect();
            let z: f64 = p.iter().sum();
            for d in 0..nv::HEAD_DIM {
                let o: f64 =
                    (0..t).map(|j| p[j] * qkv[j * ld + 2 * nv::HIDDEN + head * nv::HEAD_DIM + d] as f64).sum::<f64>() / z;
                want[i * nv::HIDDEN + head * nv::HEAD_DIM + d] = o as f32;
            }
        }
    }
    let m = metric(&got, &want);
    println!("attention rel_max {:.3e} rel_l2 {:.3e}", m.rel_max, m.rel_l2);
    assert!(m.rel_max < 2e-3, "{m:?}");
}

/// The tower against the full dumps (tower features, mlp1) and the committed fixture (cls, first patch and last token at every
/// stage, the final layernorm, the pixel shuffle and mlp1 rows), at the two shapes the dumps hold.
#[test]
fn tower_matches_the_reference_dumps_and_fixture() {
    gpu!();
    let pack = env_or_skip!("REXL3_NEMO_OMNI_DIR", "REXL3_NEMOTRON_PACK");
    let refd = env_or_skip!("REXL3_NEMO_VIT_REF", "REXL3_NEMOTRON_VIT_REF");
    let fixture = repo_root().join(FIXTURE_DIR).join("radio_ref_fixture.safetensors");
    let st = Stream::new().unwrap();
    let w = nv::Weights::load(&pack).unwrap();
    let mut ws = nv::Workspace::new(nv::MAX_PATCHES).unwrap();
    for (tag, height, width) in [("512x512", 512usize, 512usize), ("320x448", 320, 448)] {
        let (gh, gw) = nv::patch_grid(height, width).unwrap();
        let n = gh * gw;
        let np = n / 4;
        let dump = refd.join(format!("ref_{tag}.safetensors"));
        let (px, pshape) = st_f32(&dump, "pixels_bf16");
        assert_eq!(pshape, [1, 3, height, width], "{tag} pixels");
        let px16: Vec<u16> = px.iter().map(|&v| f64_to_f16(v as f64)).collect();
        let pix = DevBuf::from_slice(&px16).unwrap();

        let mut tr = nv::Trace::default();
        nv::tower_forward(&w, &mut ws, &pix, height, width, Some(&mut tr), &st).unwrap();
        st.sync().unwrap();

        // full tensors from the dump
        let (ref_feat, fshape) = st_f32(&dump, "tower_features");
        assert_eq!(fshape, [1, n, nv::HIDDEN]);
        let mut feat = vec![0f32; n * nv::HIDDEN];
        ws.h().download_range(nv::FEATURE_OFFSET, &mut feat).unwrap();
        let m_feat = metric(&feat, &ref_feat);

        let (ref_lnf, _) = st_f32(&dump, "final_ln");
        let mut lnf16 = vec![0u16; n * nv::HIDDEN];
        ws.final_ln().download_range(0, &mut lnf16).unwrap();
        let m_lnf = metric(&f16_bits_to_f32(&lnf16), &ref_lnf);

        let (ref_mlp1, mshape) = st_f32(&dump, "mlp1");
        assert_eq!(mshape, [1, np, nv::LLM_HIDDEN]);
        let mut mlp1 = vec![0f32; np * nv::LLM_HIDDEN];
        ws.mlp1().download_range(0, &mut mlp1).unwrap();
        let m_mlp1 = metric(&mlp1, &ref_mlp1);

        // per-stage rows from the committed fixture: block_rows [33, 3, 1280] (embeddings, then block outputs)
        let (fx_blocks, bshape) = st_f32(&fixture, &format!("{tag}.block_rows"));
        assert_eq!(bshape, [nv::BLOCKS + 1, 3, nv::HIDDEN]);
        let mut worst_stage = Metric { rel_max: 0.0, rel_l2: 0.0 };
        let mut stage_l2 = Vec::new();
        for (k, got) in tr.blocks.iter().enumerate() {
            let want = &fx_blocks[k * 3 * nv::HIDDEN..(k + 1) * 3 * nv::HIDDEN];
            let m = metric(got, want);
            stage_l2.push(m.rel_l2);
            worst_stage.rel_max = worst_stage.rel_max.max(m.rel_max);
            worst_stage.rel_l2 = worst_stage.rel_l2.max(m.rel_l2);
        }
        let (fx_lnf, _) = st_f32(&fixture, &format!("{tag}.final_ln_rows"));
        let m_fx_lnf = metric(&tr.final_ln, &fx_lnf);
        let (fx_ps, _) = st_f32(&fixture, &format!("{tag}.pixel_shuffle_rows"));
        let m_fx_ps = metric(&tr.pixel_shuffle, &fx_ps);
        let (fx_mlp1, _) = st_f32(&fixture, &format!("{tag}.mlp1_rows"));
        let m_fx_mlp1 = metric(&tr.mlp1, &fx_mlp1);

        println!("== {tag}: {n} patches (grid {gh}x{gw}) ==");
        println!("tower_features (full)  rel_max {:.3e} rel_l2 {:.3e}", m_feat.rel_max, m_feat.rel_l2);
        println!("final_ln (full, f16)   rel_max {:.3e} rel_l2 {:.3e}", m_lnf.rel_max, m_lnf.rel_l2);
        println!("mlp1 (full)            rel_max {:.3e} rel_l2 {:.3e}", m_mlp1.rel_max, m_mlp1.rel_l2);
        println!("stages (fixture rows, 33): worst rel_max {:.3e} worst rel_l2 {:.3e}", worst_stage.rel_max, worst_stage.rel_l2);
        println!("  rel_l2 by stage: {}", stage_l2.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>().join(" "));
        println!("fixture final_ln rows  rel_max {:.3e} rel_l2 {:.3e}", m_fx_lnf.rel_max, m_fx_lnf.rel_l2);
        println!("fixture pixel_shuffle  rel_max {:.3e} rel_l2 {:.3e}", m_fx_ps.rel_max, m_fx_ps.rel_l2);
        println!("fixture mlp1 rows      rel_max {:.3e} rel_l2 {:.3e}", m_fx_mlp1.rel_max, m_fx_mlp1.rel_l2);

        assert!(m_feat.rel_l2 < GATE_TOWER_REL_L2, "{tag}: tower features {m_feat:?}");
        assert!(m_mlp1.rel_max < GATE_MLP1_REL_MAX, "{tag}: mlp1 {m_mlp1:?}");
        assert!(m_mlp1.rel_l2 < GATE_MLP1_REL_L2, "{tag}: mlp1 {m_mlp1:?}");
        assert!(worst_stage.rel_l2 < GATE_STAGE_REL_L2, "{tag}: a stage {worst_stage:?}");
        assert!(m_fx_mlp1.rel_max < GATE_MLP1_REL_MAX && m_fx_mlp1.rel_l2 < GATE_MLP1_REL_L2, "{tag}: mlp1 rows {m_fx_mlp1:?}");
    }
}

/// Non-square sizes in both orientations run (the position resample crops the square max(h, w) grid), and the output is finite
/// and of the right size. Taller than wide exercises the row crop, which the dumps do not cover.
#[test]
fn non_square_sizes_run_and_stay_finite() {
    gpu!();
    let pack = env_or_skip!("REXL3_NEMO_OMNI_DIR", "REXL3_NEMOTRON_PACK");
    let st = Stream::new().unwrap();
    let w = nv::Weights::load(&pack).unwrap();
    let mut ws = nv::Workspace::new(2240).unwrap();
    for (height, width) in [(448usize, 320usize), (640, 896)] {
        let (gh, gw) = nv::patch_grid(height, width).unwrap();
        let n = gh * gw;
        let (_, px) = rand_f16(3 * height * width, 17, -2.0, 2.0);
        let pix = DevBuf::from_slice(&px.iter().map(|&v| f64_to_f16(v as f64)).collect::<Vec<u16>>()).unwrap();
        nv::tower_forward(&w, &mut ws, &pix, height, width, None, &st).unwrap();
        st.sync().unwrap();
        let mut feat = vec![0f32; n * nv::HIDDEN];
        ws.h().download_range(nv::FEATURE_OFFSET, &mut feat).unwrap();
        let mut mlp1 = vec![0f32; n / 4 * nv::LLM_HIDDEN];
        ws.mlp1().download_range(0, &mut mlp1).unwrap();
        assert!(feat.iter().all(|v| v.is_finite()), "{height}x{width}: features");
        assert!(mlp1.iter().all(|v| v.is_finite()), "{height}x{width}: mlp1");
        let rms = (mlp1.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / mlp1.len() as f64).sqrt();
        println!("{height}x{width}: {gh}x{gw} patches, mlp1 rms {rms:.3e}");
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
// Kernel-level tests of the tensor-core path (GEMM epilogues and tile edges, attention at a size with partial tiles, the warp
// LayerNorm, the channel-generic patchify, determinism).
// ---------------------------------------------------------------------------------------------------------------------------

/// erf to about 1.5e-7 absolute (Abramowitz and Stegun 7.1.26): enough for a 2e-3 gate.
fn erf64(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}

fn gemm_case(epi: nv::Epi, m: usize, n: usize, k: usize, ldc: usize, seed: u64, with_bias: bool) -> Metric {
    let st = Stream::new().unwrap();
    let (abits, a) = rand_f16(m * k, seed, -1.0, 1.0);
    let (bbits, b) = rand_f16(n * k, seed + 1, -0.25, 0.25);
    let (_, bias) = rand_f16(n, seed + 2, -0.5, 0.5);
    let f32out = matches!(epi, nv::Epi::BiasAddF32 | nv::Epi::StoreF32);
    let (_, c_init) = rand_f16(m * ldc, seed + 3, -2.0, 2.0);
    let da = DevBuf::from_slice(&abits).unwrap();
    let db = DevBuf::from_slice(&bbits).unwrap();
    let dbias = DevBuf::from_slice(&bias).unwrap();
    let esz = if f32out { 4 } else { 2 };
    let mut dc = DevBuf::zeroed(m * ldc * esz).unwrap();
    if epi == nv::Epi::BiasAddF32 {
        dc.upload(&c_init).unwrap();
    }
    let bias_arg = if with_bias { Some(&dbias) } else { None };
    nv::gemm(epi, &da, &db, bias_arg, &mut dc, 0, m, n, k, ldc, &st).unwrap();
    st.sync().unwrap();
    let got: Vec<f32> = if f32out { dc.to_vec::<f32>().unwrap() } else { f16_bits_to_f32(&dc.to_vec::<u16>().unwrap()) };
    let mut got_used = Vec::with_capacity(m * n);
    let mut want_used = Vec::with_capacity(m * n);
    for r in 0..m {
        for c in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                acc += a[r * k + kk] as f64 * b[c * k + kk] as f64;
            }
            let v = acc + if with_bias { bias[c] as f64 } else { 0.0 };
            let o = match epi {
                nv::Epi::BiasF16 | nv::Epi::StoreF32 => v,
                nv::Epi::BiasGeluF16 => 0.5 * v * (1.0 + erf64(v * std::f64::consts::FRAC_1_SQRT_2)),
                nv::Epi::BiasAddF32 => c_init[r * ldc + c] as f64 + v,
                nv::Epi::Relu2F16 => v.max(0.0).powi(2),
            };
            got_used.push(got[r * ldc + c]);
            want_used.push(o as f32);
        }
        // the columns past n of every row are untouched (zero, or the initial value for the in-place add)
        for c in n..ldc {
            let expect = if epi == nv::Epi::BiasAddF32 { c_init[r * ldc + c] } else { 0.0 };
            assert_eq!(got[r * ldc + c], expect, "{epi:?} {m}x{n}x{k} ldc {ldc}: column {c} of row {r} was written");
        }
    }
    metric(&got_used, &want_used)
}

#[test]
fn gemm_matches_a_cpu_reference_for_every_epilogue_and_tile_edge() {
    gpu!();
    // (m, n, k, ldc): the tensor-core path with a partial row tile and a partial column tile, a strided output, and the shapes the
    // simple kernel takes (n not a multiple of 8, k not a multiple of 32)
    let shapes = [
        (130usize, 384usize, 64usize, 384usize),
        (75, 200, 96, 208),
        (33, 1280, 1280, 1280),
        (5, 24, 32, 24),
        (50, 100, 48, 100),
        (9, 72, 40, 72),
    ];
    for (si, &(m, n, k, ldc)) in shapes.iter().enumerate() {
        for (ei, epi) in [nv::Epi::BiasF16, nv::Epi::BiasGeluF16, nv::Epi::BiasAddF32, nv::Epi::StoreF32, nv::Epi::Relu2F16]
            .into_iter()
            .enumerate()
        {
            let with_bias = epi != nv::Epi::Relu2F16;
            let mt = gemm_case(epi, m, n, k, ldc, 100 + 10 * si as u64 + ei as u64, with_bias);
            println!("gemm {epi:?} m={m} n={n} k={k} ldc={ldc}: rel_max {:.2e} rel_l2 {:.2e}", mt.rel_max, mt.rel_l2);
            let tol = if matches!(epi, nv::Epi::BiasAddF32 | nv::Epi::StoreF32) { 1e-5 } else { 2e-3 };
            assert!(mt.rel_max < tol, "{epi:?} {m}x{n}x{k}: {mt:?}");
        }
    }
}

#[test]
fn gemm_and_attention_are_deterministic() {
    gpu!();
    let st = Stream::new().unwrap();
    let (m, n, k) = (333usize, 640usize, 256usize);
    let (abits, _) = rand_f16(m * k, 7, -1.0, 1.0);
    let (bbits, _) = rand_f16(n * k, 8, -0.3, 0.3);
    let (_, bias) = rand_f16(n, 9, -0.5, 0.5);
    let da = DevBuf::from_slice(&abits).unwrap();
    let db = DevBuf::from_slice(&bbits).unwrap();
    let dbias = DevBuf::from_slice(&bias).unwrap();
    let run = |epi: nv::Epi| -> Vec<u8> {
        let esz = if matches!(epi, nv::Epi::BiasAddF32 | nv::Epi::StoreF32) { 4 } else { 2 };
        let mut c = DevBuf::zeroed(m * n * esz).unwrap();
        nv::gemm(epi, &da, &db, Some(&dbias), &mut c, 0, m, n, k, n, &st).unwrap();
        st.sync().unwrap();
        c.to_vec::<u8>().unwrap()
    };
    for epi in [nv::Epi::BiasF16, nv::Epi::BiasGeluF16, nv::Epi::StoreF32] {
        assert!(run(epi) == run(epi), "{epi:?} is not deterministic");
    }
    let t = 777usize;
    let (qbits, _) = rand_f16(t * 3 * nv::HIDDEN, 21, -2.0, 2.0);
    let dq = DevBuf::from_slice(&qbits).unwrap();
    let att = || -> Vec<u16> {
        let mut o = DevBuf::zeroed(t * nv::HIDDEN * 2).unwrap();
        nv::attention(&dq, &mut o, t, &st).unwrap();
        st.sync().unwrap();
        o.to_vec::<u16>().unwrap()
    };
    assert!(att() == att(), "attention is not deterministic");
}

/// Attention at a size with a partial key tile (1034 = 16 * 64 + 10) and a partial query tile, against an f64 reference on three
/// heads (first, middle, last).
#[test]
fn attention_matches_a_cpu_reference_at_the_size_of_an_image() {
    gpu!();
    let st = Stream::new().unwrap();
    let t = 1034usize;
    let ld = 3 * nv::HIDDEN;
    let (qbits, qkv) = rand_f16(t * ld, 31, -2.0, 2.0);
    let dq = DevBuf::from_slice(&qbits).unwrap();
    let mut dout = DevBuf::zeroed(t * nv::HIDDEN * 2).unwrap();
    nv::attention(&dq, &mut dout, t, &st).unwrap();
    st.sync().unwrap();
    let got = f16_bits_to_f32(&dout.to_vec::<u16>().unwrap());
    let scale = 1.0 / (nv::HEAD_DIM as f64).sqrt();
    let (mut g, mut w) = (Vec::new(), Vec::new());
    for head in [0usize, 7, 15] {
        for i in 0..t {
            let mut s = vec![0f64; t];
            for j in 0..t {
                s[j] = (0..nv::HEAD_DIM)
                    .map(|d| {
                        qkv[i * ld + head * nv::HEAD_DIM + d] as f64 * qkv[j * ld + nv::HIDDEN + head * nv::HEAD_DIM + d] as f64
                    })
                    .sum::<f64>()
                    * scale;
            }
            let mx = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let p: Vec<f64> = s.iter().map(|&v| (v - mx).exp()).collect();
            let z: f64 = p.iter().sum();
            for d in 0..nv::HEAD_DIM {
                let o: f64 =
                    (0..t).map(|j| p[j] * qkv[j * ld + 2 * nv::HIDDEN + head * nv::HEAD_DIM + d] as f64).sum::<f64>() / z;
                w.push(o as f32);
                g.push(got[i * nv::HIDDEN + head * nv::HEAD_DIM + d]);
            }
        }
    }
    let m = metric(&g, &w);
    println!("attention T={t} rel_max {:.3e} rel_l2 {:.3e}", m.rel_max, m.rel_l2);
    assert!(m.rel_max < 2e-3 && m.rel_l2 < 1e-3, "{m:?}");
}

/// LayerNorm on a row range that starts mid-buffer (the tower's `vision_final_layernorm` starts at the first patch) and on a width
/// other than 1280 (the generic kernel).
#[test]
fn layernorm_handles_an_offset_and_other_widths() {
    gpu!();
    let st = Stream::new().unwrap();
    for (rows, dim, skip) in [(37usize, nv::HIDDEN, 10usize), (9, 1000, 0)] {
        let (_, x) = rand_f16((rows + skip) * dim, 41, -3.0, 5.0);
        let (_, w) = rand_f16(dim, 43, 0.5, 1.5);
        let (_, b) = rand_f16(dim, 47, -0.5, 0.5);
        let dx = DevBuf::from_slice(&x).unwrap();
        let dw = DevBuf::from_slice(&w).unwrap();
        let db = DevBuf::from_slice(&b).unwrap();
        let mut dy = DevBuf::zeroed(rows * dim * 2).unwrap();
        nv::layernorm(&dx, skip * dim * 4, &dw, &db, &mut dy, rows, dim, nv::LN_EPS, &st).unwrap();
        st.sync().unwrap();
        let got = f16_bits_to_f32(&dy.to_vec::<u16>().unwrap());
        let mut want = vec![0f32; rows * dim];
        for r in 0..rows {
            let row = &x[(r + skip) * dim..(r + skip + 1) * dim];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / dim as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / dim as f64;
            for c in 0..dim {
                want[r * dim + c] =
                    (((row[c] as f64 - mean) / (var + nv::LN_EPS as f64).sqrt()) * w[c] as f64 + b[c] as f64) as f32;
            }
        }
        let m = metric(&got, &want);
        println!("layernorm rows {rows} dim {dim}: rel_max {:.3e}", m.rel_max);
        assert!(m.rel_max < 2e-3, "{rows}x{dim}: {m:?}");
    }
}

/// Patchify for 3 channels (an image) and 6 (two stacked frames): every output element is the pixel the column order names.
#[test]
fn patchify_reads_the_channel_row_column_order_for_3_and_6_channels() {
    gpu!();
    let st = Stream::new().unwrap();
    let (gh, gw) = (3usize, 5usize);
    let (h, w) = (gh * 16, gw * 16);
    for channels in [3usize, 6] {
        let (bits, px) = rand_f16(channels * h * w, 51, -1.0, 1.0);
        let dp = DevBuf::from_slice(&bits).unwrap();
        let mut out = DevBuf::zeroed(gh * gw * channels * 256 * 2).unwrap();
        nv::patchify_channels(&dp, &mut out, h, w, gh, gw, channels, &st).unwrap();
        st.sync().unwrap();
        let got = f16_bits_to_f32(&out.to_vec::<u16>().unwrap());
        for p in 0..gh * gw {
            let (gy, gx) = (p / gw, p % gw);
            for i in 0..channels * 256 {
                let (c, py, pxx) = (i / 256, (i / 16) % 16, i % 16);
                assert_eq!(
                    got[p * channels * 256 + i],
                    px[(c * h + gy * 16 + py) * w + gx * 16 + pxx],
                    "channels {channels} patch {p} col {i}"
                );
            }
        }
    }
}

/// The repository root: `REXL3_REPO_ROOT` when set (a prebuilt test binary run from an installed bin directory that carries the
/// tree's `testdata/`, e.g. on a node without this checkout), else the checkout this binary was built from.
fn repo_root() -> std::path::PathBuf {
    std::env::var_os("REXL3_REPO_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}
