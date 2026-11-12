//! The video path of the Nemotron 3.5 Super VL tower (`nemotron_vit::video_embed`, `video_forward`): the six-plane patch projection with the
//! pack's `video_embedder.weight`, and tubelets batched through the blocks.
//!
//! GPU tests skip when there is no device. All of them need the pack (`REXL3_NEMOTRON_PACK`, the vision tensors are read through its index);
//! the reference test also needs `REXL3_NEMOTRON_VIDEO_REF`, the directory `tools/nemotron_omni/dump_video_reference.py` wrote
//! (`video_<name>.safetensors`: the three fixture videos' tubelets and the pack's own bf16 and fp32 stages). Nothing derived from a weight is
//! committed; the numbers are printed as scalars.
//!
//! Metrics follow `nemotron_vit.rs`: `rel_max = max|a - b| / max|b|`, `rel_l2 = ||a - b|| / ||b||`, `b` the reference.
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
            eprintln!("no CUDA device: skipping");
            return;
        }
    };
}

macro_rules! env_or_skip {
    ($name:expr) => {
        match std::env::var_os($name) {
            Some(v) => PathBuf::from(v),
            None => {
                eprintln!("{} is not set: skipping", $name);
                return;
            }
        }
    };
}

/// The image gates of `nemotron_vit.rs`, not relaxed for video: the tower's features, mlp1, a stage.
const GATE_TOWER_REL_L2: f64 = 3.2e-2;
const GATE_MLP1_REL_MAX: f64 = 1.0e-1;
const GATE_MLP1_REL_L2: f64 = 8.0e-2;
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

/// One tensor of a safetensors file as f32 (BF16 and F32), with its shape.
fn st_f32(path: &Path, name: &str) -> (Vec<f32>, Vec<usize>) {
    let (hs, _len, header) = read_header_checked(path).unwrap();
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

fn rand_f16(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            f64_to_f16(lo as f64 + (hi - lo) as f64 * u)
        })
        .collect()
}

/// What one tubelet's run leaves, as raw bits (the comparison is bit for bit).
#[derive(PartialEq, Eq, Debug)]
struct Bits {
    features: Vec<u32>,
    final_ln: Vec<u16>,
    mlp1: Vec<u32>,
}

/// Run `order` (indices into `pix`) as one batch and read each tubelet's rows back.
fn run_batch(
    w: &nv::Weights,
    ws: &mut nv::Workspace,
    st: &Stream,
    pix: &[DevBuf],
    order: &[usize],
    h: usize,
    wd: usize,
) -> Vec<Bits> {
    let (gh, gw) = nv::patch_grid(h, wd).unwrap();
    let (n, np) = (gh * gw, gh * gw / 4);
    let t = nv::PREFIX + n;
    for (b, &g) in order.iter().enumerate() {
        nv::video_embed(w, ws, &pix[g], h, wd, b, st).unwrap();
    }
    nv::video_forward(w, ws, h, wd, order.len(), None, st).unwrap();
    st.sync().unwrap();
    (0..order.len())
        .map(|b| {
            let mut f = vec![0f32; n * nv::HIDDEN];
            ws.h().download_range((b * t + nv::PREFIX) * nv::HIDDEN * 4, &mut f).unwrap();
            let mut ln = vec![0u16; n * nv::HIDDEN];
            ws.final_ln().download_range(b * n * nv::HIDDEN * 2, &mut ln).unwrap();
            let mut m = vec![0f32; np * nv::LLM_HIDDEN];
            ws.mlp1().download_range(b * np * nv::LLM_HIDDEN * 4, &mut m).unwrap();
            Bits {
                features: f.iter().map(|v| v.to_bits()).collect(),
                final_ln: ln,
                mlp1: m.iter().map(|v| v.to_bits()).collect(),
            }
        })
        .collect()
}

/// A batch of tubelets equals the same tubelets run one at a time, bit for bit, whatever their order in the batch (so whatever 128-row
/// GEMM tile their rows share with a neighbour): features, the final LayerNorm and mlp1. The tile is 960 patches, so a tubelet is 970
/// rows and a batch cuts the GEMM tiles through the tubelets.
#[test]
fn a_video_batch_equals_its_tubelets_one_at_a_time_bit_for_bit() {
    gpu!();
    let pack = env_or_skip!("REXL3_NEMOTRON_PACK");
    let st = Stream::new().unwrap();
    let w = nv::Weights::load(&pack).unwrap();
    let (h, wd) = (384usize, 640usize);
    let n = (h / 16) * (wd / 16);
    let k = 5usize;
    let cap = nv::video_batch_capacity(4 * 1280 + 1024, n);
    assert!(cap >= 5, "capacity {cap}");
    let mut ws = nv::Workspace::new(4 * 1280 + 1024).unwrap();
    let pix: Vec<DevBuf> =
        (0..k).map(|g| DevBuf::from_slice(&rand_f16(nv::VIDEO_PLANES * h * wd, 700 + g as u64, -2.0, 2.0)).unwrap()).collect();
    // the references: each tubelet alone
    let alone: Vec<Bits> = (0..k).map(|g| run_batch(&w, &mut ws, &st, &pix, &[g], h, wd).pop().unwrap()).collect();
    assert!(alone.iter().all(|b| b.features.iter().all(|v| f32::from_bits(*v).is_finite())), "non-finite features");
    assert!(alone[0] != alone[1], "tubelets must differ for the test to mean anything");
    for order in [vec![0usize, 1], vec![2, 0, 3], vec![4, 3, 2, 1, 0], vec![1, 1, 4], vec![3, 0, 1, 2, 4]] {
        let got = run_batch(&w, &mut ws, &st, &pix, &order, h, wd);
        for (b, &g) in order.iter().enumerate() {
            assert!(got[b].features == alone[g].features, "batch {order:?}: tubelet {g} at {b}: tower features differ");
            assert!(got[b].final_ln == alone[g].final_ln, "batch {order:?}: tubelet {g} at {b}: final LayerNorm differs");
            assert!(got[b].mlp1 == alone[g].mlp1, "batch {order:?}: tubelet {g} at {b}: mlp1 differs");
        }
        println!("batch {order:?}: every tubelet's features, final LN and mlp1 are bit-identical to its own run");
    }
    // and the run is repeatable (the kernels are deterministic)
    let again = run_batch(&w, &mut ws, &st, &pix, &[2], h, wd).pop().unwrap();
    assert!(again == alone[2], "a tubelet run twice differs");
}

/// A frame pair that is the same frame twice (what an odd count pads) takes the same path: the tubelet `[f, f]` is an ordinary tubelet,
/// and its features differ from the single-frame image path's (another embedder), which is the point of the video projection.
#[test]
fn the_video_projection_is_not_the_image_projection() {
    gpu!();
    let pack = env_or_skip!("REXL3_NEMOTRON_PACK");
    let st = Stream::new().unwrap();
    let w = nv::Weights::load(&pack).unwrap();
    let (h, wd) = (256usize, 256usize);
    let mut ws = nv::Workspace::new(2 * 256 + 4).unwrap();
    let frame = rand_f16(3 * h * wd, 31, -2.0, 2.0);
    let both: Vec<u16> = frame.iter().chain(frame.iter()).copied().collect();
    let img = DevBuf::from_slice(&frame).unwrap();
    let tub = DevBuf::from_slice(&both).unwrap();
    let n = (h / 16) * (wd / 16);
    nv::tower_forward(&w, &mut ws, &img, h, wd, None, &st).unwrap();
    st.sync().unwrap();
    let mut as_image = vec![0f32; n * nv::HIDDEN];
    ws.h().download_range(nv::FEATURE_OFFSET, &mut as_image).unwrap();
    nv::video_embed(&w, &mut ws, &tub, h, wd, 0, &st).unwrap();
    nv::video_forward(&w, &mut ws, h, wd, 1, None, &st).unwrap();
    st.sync().unwrap();
    let mut as_video = vec![0f32; n * nv::HIDDEN];
    ws.h().download_range(nv::FEATURE_OFFSET, &mut as_video).unwrap();
    let m = metric(&as_video, &as_image);
    println!("[f, f] as a tubelet against f as an image: rel_l2 {:.3e}", m.rel_l2);
    assert!(m.rel_l2 > 1e-3, "the video embedder gave the image path's features: {m:?}");
}

/// The three fixture videos' tubelets (the pack's own processor and `forward_video`'s stacking) through the engine's tower against the
/// pack's own bf16 run and an fp32 run of the same code: the image gates, and the engine's distance to the fp32 truth against the bf16
/// run's, by stage.
#[test]
fn the_video_tower_matches_the_pack_on_the_fixture_videos() {
    gpu!();
    let pack = env_or_skip!("REXL3_NEMOTRON_PACK");
    let refd = env_or_skip!("REXL3_NEMOTRON_VIDEO_REF");
    let st = Stream::new().unwrap();
    let w = nv::Weights::load(&pack).unwrap();
    let mut ws = nv::Workspace::new(nv::MAX_PATCHES).unwrap();
    let mut found = 0;
    for name in ["portrait_1frame", "wide_2frames", "square_5frames"] {
        let dump = refd.join(format!("video_{name}.safetensors"));
        if !dump.is_file() {
            eprintln!("{} is missing: skipping {name}", dump.display());
            continue;
        }
        found += 1;
        let (px, shape) = st_f32(&dump, "pixels_bf16");
        assert_eq!(shape.len(), 4);
        let (g, c, h, wd) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(c, nv::VIDEO_PLANES);
        let (gh, gw) = nv::patch_grid(h, wd).unwrap();
        let (n, np) = (gh * gw, gh * gw / 4);
        let per = c * h * wd;
        let pix: Vec<DevBuf> = (0..g)
            .map(|i| {
                let v: Vec<u16> = px[i * per..(i + 1) * per].iter().map(|&x| f64_to_f16(x as f64)).collect();
                DevBuf::from_slice(&v).unwrap()
            })
            .collect();
        // the whole video as one batch (the capacity at 13,312 patches holds 12 tubelets of ~1,000 patches)
        assert!(g <= nv::video_batch_capacity(nv::MAX_PATCHES, n));
        for b in 0..g {
            nv::video_embed(&w, &mut ws, &pix[b], h, wd, b, &st).unwrap();
        }
        let mut traces = Vec::new();
        nv::video_forward(&w, &mut ws, h, wd, g, Some(&mut traces), &st).unwrap();
        st.sync().unwrap();
        let t = nv::PREFIX + n;
        println!("== {name}: {g} tubelets of {n} patches ({gh}x{gw}), {np} tokens each ==");
        let mut feat = vec![0f32; g * t * nv::HIDDEN];
        ws.h().download_range(0, &mut feat).unwrap();
        let mut lnf16 = vec![0u16; g * n * nv::HIDDEN];
        ws.final_ln().download_range(0, &mut lnf16).unwrap();
        let lnf: Vec<f32> = lnf16.iter().map(|&b| f16_to_f64(b) as f32).collect();
        let mut mlp1 = vec![0f32; g * np * nv::LLM_HIDDEN];
        ws.mlp1().download_range(0, &mut mlp1).unwrap();
        // the engine's tower features without the prefix rows, per tubelet
        let mut tower = Vec::with_capacity(g * n * nv::HIDDEN);
        for b in 0..g {
            tower.extend_from_slice(&feat[(b * t + nv::PREFIX) * nv::HIDDEN..(b + 1) * t * nv::HIDDEN]);
        }
        for tag in ["bf16", "fp32"] {
            let (rt, s) = st_f32(&dump, &format!("{tag}.tower_features"));
            assert_eq!(s, [g, n, nv::HIDDEN]);
            let (rl, _) = st_f32(&dump, &format!("{tag}.final_ln"));
            let (rm, s) = st_f32(&dump, &format!("{tag}.mlp1"));
            assert_eq!(s, [g, np, nv::LLM_HIDDEN]);
            let (mt, ml, mm) = (metric(&tower, &rt), metric(&lnf, &rl), metric(&mlp1, &rm));
            println!(
                "engine vs {tag}: tower rel_l2 {:.3e} (rel_max {:.3e})  final_ln rel_l2 {:.3e}  mlp1 rel_l2 {:.3e} rel_max {:.3e}",
                mt.rel_l2, mt.rel_max, ml.rel_l2, mm.rel_l2, mm.rel_max
            );
            if tag == "bf16" {
                assert!(mt.rel_l2 < GATE_TOWER_REL_L2, "{name}: tower features {mt:?}");
                assert!(mm.rel_max < GATE_MLP1_REL_MAX && mm.rel_l2 < GATE_MLP1_REL_L2, "{name}: mlp1 {mm:?}");
            }
        }
        // the engine against the truth, and the pack's own bf16 run against it
        let (t32, _) = st_f32(&dump, "fp32.tower_features");
        let (t16, _) = st_f32(&dump, "bf16.tower_features");
        let (m32, _) = st_f32(&dump, "fp32.mlp1");
        let (m16, _) = st_f32(&dump, "bf16.mlp1");
        let (e_t, b_t) = (metric(&tower, &t32).rel_l2, metric(&t16, &t32).rel_l2);
        let (e_m, b_m) = (metric(&mlp1, &m32).rel_l2, metric(&m16, &m32).rel_l2);
        println!(
            "against fp32: tower engine {e_t:.3e} bf16 {b_t:.3e} (engine {:.1}x closer); mlp1 engine {e_m:.3e} bf16 {b_m:.3e} (engine {:.1}x closer)",
            b_t / e_t,
            b_m / e_m
        );
        assert!(e_t < b_t && e_m < b_m, "{name}: the engine is farther from the fp32 truth than the pack's bf16 run");

        // per-stage rows (tokens 0, 10 and the last of every block) of every tubelet
        let mut worst = 0f64;
        let mut by_stage = vec![0f64; nv::BLOCKS + 1];
        let (rows, s) = st_f32(&dump, "bf16.block_rows");
        assert_eq!(s, [g, nv::BLOCKS + 1, 3, nv::HIDDEN]);
        for (b, tr) in traces.iter().enumerate() {
            assert_eq!(tr.blocks.len(), nv::BLOCKS + 1);
            for (k, got) in tr.blocks.iter().enumerate() {
                let o = ((b * (nv::BLOCKS + 1) + k) * 3) * nv::HIDDEN;
                let m = metric(got, &rows[o..o + 3 * nv::HIDDEN]);
                by_stage[k] = by_stage[k].max(m.rel_l2);
                worst = worst.max(m.rel_l2);
            }
        }
        println!(
            "stage rows vs bf16 (every tubelet): worst rel_l2 {worst:.3e}; by stage: {}",
            by_stage.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>().join(" ")
        );
        assert!(worst < GATE_STAGE_REL_L2, "{name}: a stage {worst:.3e}");
        for b in 0..g {
            let (fl, _) = st_f32(&dump, "bf16.final_ln_rows");
            let (ps, _) = st_f32(&dump, "bf16.pixel_shuffle_rows");
            let (ml, _) = st_f32(&dump, "bf16.mlp1_rows");
            let m_fl = metric(&traces[b].final_ln, &fl[b * 2 * nv::HIDDEN..(b + 1) * 2 * nv::HIDDEN]);
            let m_ps = metric(&traces[b].pixel_shuffle, &ps[b * 2 * 4 * nv::HIDDEN..(b + 1) * 2 * 4 * nv::HIDDEN]);
            let m_ml = metric(&traces[b].mlp1, &ml[b * 2 * nv::LLM_HIDDEN..(b + 1) * 2 * nv::LLM_HIDDEN]);
            println!(
                "  tubelet {b}: final_ln rows {:.2e}  pixel_shuffle rows {:.2e}  mlp1 rows rel_l2 {:.2e} rel_max {:.2e}",
                m_fl.rel_l2, m_ps.rel_l2, m_ml.rel_l2, m_ml.rel_max
            );
            assert!(m_ml.rel_max < GATE_MLP1_REL_MAX && m_ml.rel_l2 < GATE_MLP1_REL_L2, "{name} tubelet {b}: mlp1 rows {m_ml:?}");
        }
    }
    assert!(found > 0, "no fixture video dump under REXL3_NEMOTRON_VIDEO_REF");
}
