//! GPU tests of the NextN (MTP) layer of GLM-5.3-Flash. All `#[ignore]` and env-gated (REXL3_GLM_DIR: the pack; REXL3_GLM_ORACLE_DIR: the
//! oracle's dumps, anonymized host-d `/srv/oracles/glm53/final4-A (example path)`); without them a test prints "not run". They share one model per process and must run with
//! `--test-threads=1`, under the GPU lock (`nrun NAME --gpu -- ...`).
//!
//! * `mtp_stage_in_isolation_matches_the_oracle`: a SMALL model (one trunk layer, the embedding and head, the NextN layer: about 6 GiB) runs the
//!   NextN block over a whole prompt from the oracle's own final-norm hidden rows (`final_norm.npy`, every position for the prompts of at most 512
//!   rows) and the oracle's tokens, and the last 9 rows of the input fusion, the block, `shared_head.norm` and the logits are judged against the
//!   oracle's `mtp_*.npy` with the manifest's "MTP stage in isolation" bounds (rel L2 <= 4.4e-3, logits max abs <= 0.07, top-1 >= 8 of 9 rows).

mod glm_support;

use gb_model::families::glm::npy;
use glm_support::*;
use std::path::{Path, PathBuf};

fn oracle_dir() -> Option<PathBuf> {
    match std::env::var("REXL3_GLM_ORACLE_DIR") {
        Ok(d) if Path::new(&d).is_dir() => Some(PathBuf::from(d)),
        Ok(d) => panic!("REXL3_GLM_ORACLE_DIR={d} is not a directory"),
        Err(_) => {
            eprintln!("REXL3_GLM_ORACLE_DIR is not set: not run");
            None
        }
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        num += ((*x - *y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-300)).sqrt()
}

fn f16_to_f32(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| gb_exl3::f16::f16_to_f64(b) as f32).collect()
}

/// The manifest's tolerance for the MTP stage in isolation: (rel L2, logits max abs, top-1 fraction).
fn mtp_tolerance() -> (f64, f64, f64) {
    let m: serde_json::Value = serde_json::from_str(include_str!("../../../testdata/glm53/ref_manifest.json")).unwrap();
    let t = &m["tolerances"]["mtp"];
    (t["rel_l2"].as_f64().unwrap(), t["logits_max_abs"].as_f64().unwrap(), t["top1_min"].as_f64().unwrap())
}

const SLOTS: usize = 2;
const KV_BLOCKS: usize = 12; // 3,072 tokens
const MAX_CONTEXT: usize = 3072;

#[test]
#[ignore]
fn mtp_stage_in_isolation_matches_the_oracle() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = shared_mtp_model(1, SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let (tol_l2, tol_logits, tol_top1) = mtp_tolerance();
    let h = m.cfg.hidden;
    let v = m.cfg.vocab;
    let mut judged = 0;
    for prompt in ["one_token", "short_20", "chat_300", "think_prefix"] {
        let d = oracle.join(prompt);
        let ids: Vec<u32> = npy::read(&d.join("ids.npy")).unwrap().to_i32().unwrap().iter().map(|&x| x as u32).collect();
        let t = ids.len();
        let fnorm = npy::read(&d.join("final_norm.npy")).unwrap();
        if fnorm.shape[0] != t {
            eprintln!("{prompt}: the oracle dumped {} of {t} final-norm rows: not used here", fnorm.shape[0]);
            continue;
        }
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("meta.json")).unwrap()).unwrap();
        let first = meta["greedy_next_token"].as_u64().unwrap() as u32;
        // the oracle's MTP window: T + 1 rows, row i takes the embedding of token i (the greedy next token for row T) and the tapped hidden of
        // position i - 1 (zeros for row 0): `final_norm.npy` holds fp16 values widened to fp32, so narrowing them is exact
        let fv = fnorm.to_f32().unwrap();
        let mut tokens = ids.clone();
        tokens.push(first);
        let mut hidden = vec![0u16; (t + 1) * h];
        for i in 1..=t {
            for k in 0..h {
                hidden[i * h + k] = gb_exl3::f16::f64_to_f16(fv[(i - 1) * h + k] as f64);
            }
        }
        let u = 9.min(t + 1);
        m.begin_slot(0).unwrap();
        let cap = m.mtp_debug_forward(0, 0, &tokens, &hidden, u).unwrap();
        let load = |name: &str| npy::read(&d.join(name)).unwrap().to_f32().unwrap();
        // the oracle's last u rows of each stage
        let (want_in, want_block, want_norm, want_logits) =
            (load("mtp_in.npy"), load("mtp_block.npy"), load("mtp_norm.npy"), load("mtp_logits.npy"));
        assert_eq!(want_in.len(), u * h, "{prompt}: the oracle's MTP rows");
        let stages = [
            ("mtp_in (eh_proj)", rel_l2(&cap.x_in, &want_in)),
            ("mtp_block", rel_l2(&cap.block, &want_block)),
            ("mtp_norm (shared_head.norm)", rel_l2(&f16_to_f32(&cap.norm), &want_norm)),
        ];
        for (name, e) in stages {
            eprintln!("{prompt}: {name:30} rel L2 {e:.3e} (bound {tol_l2:.3e})");
            assert!(e <= tol_l2, "{prompt} {name}: rel L2 {e:.3e} above the bound {tol_l2:.3e}");
        }
        let got = f16_to_f32(&cap.logits);
        assert_eq!(got.len(), u * v);
        let mad = got.iter().zip(&want_logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) as f64;
        let argmax =
            |x: &[f32]| x.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &y)| if y > a.1 { (i, y) } else { a }).0;
        let agree = (0..u).filter(|&r| argmax(&got[r * v..(r + 1) * v]) == argmax(&want_logits[r * v..(r + 1) * v])).count();
        eprintln!(
            "{prompt}: logits max abs {mad:.4} (bound {tol_logits:.4}), top-1 agree {agree} of {u} rows (bound {:.0})",
            tol_top1 * u as f64
        );
        assert!(mad <= tol_logits, "{prompt}: logits differ by {mad:.4}, above {tol_logits:.4}");
        assert!(agree as f64 >= (tol_top1 * u as f64).floor(), "{prompt}: top-1 agrees on {agree} of {u} rows");
        // the draft the generator would take: the argmax of the last row, against the oracle's own
        let top = npy::read(&d.join("mtp_top64_ids.npy")).unwrap().to_i32().unwrap();
        eprintln!("{prompt}: draft {} (oracle {})", argmax(&got[(u - 1) * v..u * v]), top[(u - 1) * 64]);
        judged += 1;
    }
    assert!(judged >= 3, "at least three prompts have every final-norm row dumped; judged {judged}");
}

/// The image rows of the oracle's MTP window: `img_small` is 391 rows of which 368 are image (the demo scene: 32 x 46 patches, 16 x 23 merged blocks),
/// and the oracle's drafter builds its cache over them from the vision features (`indexed_embeddings` of its prefill), not from the placeholder
/// token's embedding. The NextN block over the whole prompt from the oracle's own taps and features must give the oracle's last 9 rows within the same
/// bounds as a text prompt; the same run with the placeholder's embedding in those rows (the text-only drafter) is the control that shows the check can
/// tell the two apart.
#[test]
#[ignore]
fn mtp_image_rows_from_the_vision_features_match_the_oracle() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = shared_mtp_model(1, SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let (tol_l2, tol_logits, tol_top1) = mtp_tolerance();
    let (h, v) = (m.cfg.hidden, m.cfg.vocab);
    let prompt = "img_small";
    let d = oracle.join(prompt);
    let (tokens, hidden) = mtp_window_of(&oracle, prompt, h).expect("img_small has every final-norm row dumped");
    let vm: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("vision/meta.json")).unwrap()).unwrap();
    let (start, rows) = (vm["image_start_position"].as_u64().unwrap() as usize, vm["image_rows"].as_u64().unwrap() as usize);
    let g = vm["grid_thw"].as_array().unwrap();
    let merged = (g[1].as_u64().unwrap() as usize / 2, g[2].as_u64().unwrap() as usize / 2);
    let feats = npy::read(&d.join("vision/features.npy")).unwrap().to_f32().unwrap();
    assert_eq!(feats.len(), rows * h);
    let mut buf = gb_cuda::DevBuf::zeroed(rows * h * 4).unwrap();
    buf.upload(&feats).unwrap();
    let f = std::sync::Arc::new(gb_model::families::glm::vision_serve::GlmImageFeatures {
        buf,
        tokens: rows,
        merged,
        key: 0,
        permit: None,
    });
    let u = 9;
    let load = |name: &str| npy::read(&d.join(name)).unwrap().to_f32().unwrap();
    let (want_in, want_block, want_norm, want_logits) =
        (load("mtp_in.npy"), load("mtp_block.npy"), load("mtp_norm.npy"), load("mtp_logits.npy"));
    let argmax =
        |x: &[f32]| x.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &y)| if y > a.1 { (i, y) } else { a }).0;
    let judge = |cap: &gb_model::families::glm::model::MtpCapture| {
        let got = f16_to_f32(&cap.logits);
        let mad = got.iter().zip(&want_logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) as f64;
        let agree = (0..u).filter(|&r| argmax(&got[r * v..(r + 1) * v]) == argmax(&want_logits[r * v..(r + 1) * v])).count();
        (rel_l2(&cap.x_in, &want_in), rel_l2(&cap.block, &want_block), rel_l2(&f16_to_f32(&cap.norm), &want_norm), mad, agree)
    };
    // the drafter as built: the image rows from the features
    m.begin_slot(0).unwrap();
    let cap = m.mtp_debug_forward_mm(0, 0, &tokens, &hidden, u, &[(f.clone(), start, rows)]).unwrap();
    let (e_in, e_block, e_norm, mad, agree) = judge(&cap);
    eprintln!("{prompt} (image rows from the features): mtp_in {e_in:.3e}, block {e_block:.3e}, norm {e_norm:.3e} (bound {tol_l2:.3e}); logits max abs {mad:.4} (bound {tol_logits:.4}); top-1 {agree} of {u}");
    // the control: a text-only drafter feeds the placeholder id's embedding in every image row
    m.begin_slot(1).unwrap();
    let ctl = m.mtp_debug_forward(1, 0, &tokens, &hidden, u).unwrap();
    let (c_in, c_block, c_norm, c_mad, c_agree) = judge(&ctl);
    eprintln!("{prompt} (control, the placeholder's embedding in the image rows): mtp_in {c_in:.3e}, block {c_block:.3e}, norm {c_norm:.3e}; logits max abs {c_mad:.4}; top-1 {c_agree} of {u}");
    for (name, e) in [("mtp_in", e_in), ("mtp_block", e_block), ("mtp_norm", e_norm)] {
        assert!(e <= tol_l2, "{prompt} {name}: rel L2 {e:.3e} above the bound {tol_l2:.3e}");
    }
    assert!(mad <= tol_logits, "{prompt}: logits differ by {mad:.4}, above {tol_logits:.4}");
    assert!(agree as f64 >= (tol_top1 * u as f64).floor(), "{prompt}: top-1 agrees on {agree} of {u} rows");
    // the control must be visibly worse somewhere (the block output attends over the image rows' cache entries); if it were not, this test would
    // not tell the splice from its absence
    assert!(
        c_block > e_block * 2.0 || c_norm > e_norm * 2.0 || c_mad > mad * 2.0,
        "the placeholder control is indistinguishable from the feature run: block {c_block:.3e} vs {e_block:.3e}, norm {c_norm:.3e} vs {e_norm:.3e}, logits {c_mad:.4} vs {mad:.4}"
    );
}

/// The hidden rows and tokens of the oracle's MTP window for `prompt` (see the isolation test), when its final-norm rows are all dumped.
fn mtp_window_of(oracle: &Path, prompt: &str, h: usize) -> Option<(Vec<u32>, Vec<u16>)> {
    let d = oracle.join(prompt);
    let ids: Vec<u32> = npy::read(&d.join("ids.npy")).unwrap().to_i32().unwrap().iter().map(|&x| x as u32).collect();
    let t = ids.len();
    let fnorm = npy::read(&d.join("final_norm.npy")).unwrap();
    if fnorm.shape[0] != t {
        return None;
    }
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(d.join("meta.json")).unwrap()).unwrap();
    let first = meta["greedy_next_token"].as_u64().unwrap() as u32;
    let fv = fnorm.to_f32().unwrap();
    let mut tokens = ids;
    tokens.push(first);
    let mut hidden = vec![0u16; (t + 1) * h];
    for i in 1..=t {
        for k in 0..h {
            hidden[i * h + k] = gb_exl3::f16::f64_to_f16(fv[(i - 1) * h + k] as f64);
        }
    }
    Some((tokens, hidden))
}

/// The layer is row-invariant: the same rows fed in windows of 128, one row at a time, and in an odd mix give the same bits (the last rows'
/// block output, norm and logits) and leave the same bits in the layer's latent cache and DSA pools. Drafts and the rows of a verify window
/// never depend on how the rows were batched.
#[test]
#[ignore]
fn mtp_window_equals_row_by_row_bit_for_bit() {
    let Some(oracle) = oracle_dir() else { return };
    let Some(mut m) = shared_mtp_model(1, SLOTS, KV_BLOCKS, MAX_CONTEXT) else { return };
    let h = m.cfg.hidden;
    let (tokens, hidden) = mtp_window_of(&oracle, "chat_300", h).expect("chat_300 has every final-norm row dumped");
    let n = tokens.len();
    // run A: the production grid (128-row windows) in slot 0; run B: one row per call in slot 1; run C: windows of 1, 2, 3, 5, 9, ... in slot 0 again
    m.begin_slot(0).unwrap();
    let a = m.mtp_debug_forward(0, 0, &tokens, &hidden, 9).unwrap();
    let (lat_a, pool_a) = (m.mtp_latent_rows(0, n).unwrap(), m.mtp_pool_rows(0, n / 4).unwrap());
    m.begin_slot(1).unwrap();
    let mut b_last = None;
    for i in 0..n {
        let cap = m.mtp_debug_forward(1, i, &tokens[i..i + 1], &hidden[i * h..(i + 1) * h], 1).unwrap();
        b_last = Some(cap);
    }
    let (lat_b, pool_b) = (m.mtp_latent_rows(1, n).unwrap(), m.mtp_pool_rows(1, n / 4).unwrap());
    m.begin_slot(0).unwrap();
    let mut at = 0;
    let mut c_last = None;
    for (k, w) in [1usize, 2, 3, 5, 9, 17, 33, 64].iter().cycle().enumerate() {
        if at >= n {
            break;
        }
        let end = (at + w).min(n);
        let tail = if end == n { 9.min(end - at) } else { 1 };
        let cap = m.mtp_debug_forward(0, at, &tokens[at..end], &hidden[at * h..end * h], tail).unwrap();
        if end == n {
            c_last = Some(cap);
        }
        at = end;
        assert!(k < 200);
    }
    let (lat_c, pool_c) = (m.mtp_latent_rows(0, n).unwrap(), m.mtp_pool_rows(0, n / 4).unwrap());
    let (b, c) = (b_last.unwrap(), c_last.unwrap());
    // the last row: block output, norm and logits, bit for bit (the `x` buffers hold fp32 and compare by bits)
    let last_x = |v: &[f32]| v[v.len() - h..].iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    assert_eq!(last_x(&a.block), last_x(&b.block), "window vs row by row: the last row's block output");
    assert_eq!(last_x(&a.block), last_x(&c.block), "window vs mixed windows: the last row's block output");
    assert_eq!(a.norm[a.norm.len() - h..], b.norm[..], "norm: window vs row by row");
    assert_eq!(a.norm[a.norm.len() - h..], c.norm[c.norm.len() - h..], "norm: window vs mixed windows");
    let v = m.cfg.vocab;
    assert_eq!(a.logits[a.logits.len() - v..], b.logits[..], "logits: window vs row by row");
    assert_eq!(a.logits[a.logits.len() - v..], c.logits[c.logits.len() - v..], "logits: window vs mixed windows");
    assert!(lat_a == lat_b && lat_a == lat_c, "the layer's latent cache differs by how the rows were batched");
    assert!(pool_a == pool_b && pool_a == pool_c, "the layer's DSA pools differ by how the rows were batched");
    eprintln!("MTP layer: {n} rows in windows of 128, one by one and in a mix of 1..64: block output, norm, logits, latent cache and pools bit-identical");
}
