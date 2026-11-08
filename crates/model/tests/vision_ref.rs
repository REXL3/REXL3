#![allow(clippy::needless_range_loop)]
//! The vision tower on the GPU against ExLlamaV3's own activations (`tools/dump_vision_reference.py`), stage by stage, fed by
//! the Rust preprocessing (`gb-vision`) of the PNG fixtures, so the whole chain image bytes -> LM embeddings is covered.
//!
//!     REXL3_QWEN_DIR=... cargo test --release -p gb-model --test vision_ref -- --ignored --nocapture
//!
//! ExLlamaV3 is the oracle, not a bit-exact target (fp16 GEMMs with its own tiling), so stages are compared by cosine and
//! relative RMS; the bounds below were set from the measured numbers printed by this test.

mod refs;
use gb_cuda::{DevBuf, Stream};
use gb_model::pack::Pack;
use gb_model::vision::{self, VisionTower};
use gb_vision::{decode, preprocess, Preprocessing};
use refs::{cos_rel_rms, ref_dir};
use std::collections::HashMap;
use std::path::PathBuf;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision")
}

fn f16_to_f32(b: u16) -> f32 {
    gb_exl3::f16::f16_to_f64(b) as f32
}

fn names() -> Vec<String> {
    let t = std::fs::read_to_string(fixtures().join("fixtures.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&t).unwrap();
    v["cases"].as_array().unwrap().iter().map(|c| c["name"].as_str().unwrap().to_string()).collect()
}

#[test]
#[ignore]
fn encoder_matches_exllamav3_stage_by_stage() {
    let Ok(dir) = std::env::var("REXL3_QWEN_DIR") else { return eprintln!("REXL3_QWEN_DIR not set: skipping") };
    let vdir = ref_dir().join("vision");
    if !gb_cuda::device_available() || !vdir.join("manifest.json").exists() {
        return eprintln!("no GPU or vision reference dump (tools/dump_vision_reference.py): skipping");
    }
    let dir = PathBuf::from(dir);
    vision::check_config(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let pack = Pack::open(&dir).unwrap();
    let mut tower = VisionTower::load(&pack, 4096).unwrap();
    eprintln!(
        "tower: {:.2} GiB weights, {:.2} GiB scratch",
        tower.weight_bytes as f64 / 2f64.powi(30),
        tower.scratch_bytes() as f64 / 2f64.powi(30)
    );
    let st = Stream::new().unwrap();
    let pp = Preprocessing::qwen38();
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(vdir.join("manifest.json")).unwrap()).unwrap();
    let cases: Vec<String> = manifest["cases"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect();
    assert!(!cases.is_empty());
    let all = names();
    let mut worst_final = (1.0f64, 0.0f64);
    eprintln!("{:24} {:>7} {:>9} {:>9}", "case / stage", "cos", "rel rms", "");
    // JPEG fixtures are skipped here: zune-jpeg differs from libjpeg-turbo by up to 4/255, so their patches are not the oracle's
    for case in cases.iter().filter(|c| all.contains(c) && !c.starts_with("jpeg")) {
        let file = std::fs::read_dir(fixtures())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .find(|f| f.starts_with(&format!("{case}.")) && !f.ends_with(".json"))
            .unwrap();
        let pre = preprocess(&decode::decode(&std::fs::read(fixtures().join(&file)).unwrap()).unwrap(), &pp).unwrap();
        let rdir = vdir.join(case);
        let want_patches = refs::read_npy(&rdir.join("patches.npy")).to_f32();
        let got_patches: Vec<f32> = pre.patches.iter().map(|&b| f16_to_f32(b)).collect();
        assert_eq!(got_patches, want_patches, "{case}: preprocessing differs from ExLlamaV3");
        let p = pre.n_patches();
        let mut out = DevBuf::zeroed(p / 4 * vision::OUT_HIDDEN * 2).unwrap();
        let mut got: HashMap<String, Vec<f32>> = HashMap::new();
        tower
            .encode_observed(&pre, &mut out, &st, &mut |name, buf, s| {
                s.sync().unwrap();
                let v = if name.starts_with("b00_") {
                    buf.to_vec::<u16>().unwrap()[..p * vision::HIDDEN].iter().map(|&b| f16_to_f32(b)).collect()
                } else {
                    buf.to_vec::<f32>().unwrap()[..p * vision::HIDDEN].to_vec()
                };
                got.insert(name.to_string(), v);
            })
            .unwrap();
        st.sync().unwrap();
        let bits: Vec<u16> = out.to_vec().unwrap();
        let fin: Vec<f32> = bits[..p / 4 * vision::OUT_HIDDEN].iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
        for stage in [
            "pos_embed",
            "b00_norm1_out",
            "b00_attn_out",
            "b00_norm2_out",
            "b00_mlp_out",
            "block_00",
            "block_06",
            "block_13",
            "block_20",
            "block_26",
        ] {
            let want =
                refs::read_npy(&rdir.join(format!("{}.npy", if stage == "pos_embed" { "pos_embed" } else { stage }))).to_f32();
            let (cos, rel) = cos_rel_rms(&got[stage], &want);
            eprintln!("{case:16} {stage:12} {cos:>9.6} {rel:>9.5}");
            // bounds set from the measured numbers (worst cosine 0.99994 at block 13, rel rms <= 0.012): fp16 GEMMs and its own tiling
            let (min_cos, max_rel) = match stage {
                "pos_embed" | "b00_norm1_out" => (0.9999999, 1e-5),
                s if s.starts_with("b00_") => (0.99999, 0.002),
                _ => (0.9999, 0.02),
            };
            assert!(cos >= min_cos && rel <= max_rel, "{case} {stage}: cosine {cos}, relative rms {rel}");
        }
        let want = refs::read_npy(&rdir.join("merger.npy")).to_f32();
        let (cos, rel) = cos_rel_rms(&fin, &want);
        eprintln!("{case:16} {:12} {cos:>9.6} {rel:>9.5}", "merger");
        assert!(cos >= 0.9999 && rel <= 0.015, "{case} merger: cosine {cos}, relative rms {rel}");
        if cos < worst_final.0 {
            worst_final = (cos, rel);
        }
    }
    eprintln!("worst final cosine {:.6} (rel rms {:.5})", worst_final.0, worst_final.1);
}

/// Encode time by image size (random patches: the cost does not depend on the content). Prints ms per image and the tower's
/// arithmetic rate. Run on a quiet GPU.
#[test]
#[ignore]
fn encoder_speed() {
    let Ok(dir) = std::env::var("REXL3_QWEN_DIR") else { return eprintln!("REXL3_QWEN_DIR not set: skipping") };
    if !gb_cuda::device_available() {
        return eprintln!("no GPU: skipping");
    }
    let dir = PathBuf::from(dir);
    let pack = Pack::open(&dir).unwrap();
    let max_patches = 16384;
    let mut tower = VisionTower::load(&pack, max_patches).unwrap();
    let st = Stream::new().unwrap();
    let mut rng = refs::Rng(3);
    eprintln!("{:>10} {:>8} {:>10} {:>10} {:>9}", "grid", "patches", "LM tokens", "ms", "TFLOPS");
    for (gh, gw) in [(16usize, 16usize), (32, 32), (48, 48), (64, 64), (90, 90), (128, 128)] {
        let p = gh * gw;
        let patches: Vec<u16> = (0..p * 1536).map(|_| gb_exl3::f16::f64_to_f16(rng.uniform(-1.0, 1.0) as f64)).collect();
        let pre =
            gb_vision::Preprocessed { size: (gw * 16, gh * 16), grid: (1, gh, gw), patches, patch_dim: 1536, pixel_hash: 0 };
        let mut out = DevBuf::zeroed(p / 4 * vision::OUT_HIDDEN * 2).unwrap();
        tower.encode(&pre, &mut out, &st).unwrap(); // warm up (cuBLAS handle, caches)
        st.sync().unwrap();
        let mut times = Vec::new();
        for _ in 0..5 {
            let t = std::time::Instant::now();
            tower.encode(&pre, &mut out, &st).unwrap();
            st.sync().unwrap();
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = times[times.len() / 2];
        // linear part 0.85 GFLOP per patch, attention 4 * 1152 * P^2 per layer * 27
        let flops = p as f64 * 0.85e9 + 4.0 * 1152.0 * (p as f64).powi(2) * 27.0;
        eprintln!("{:>10} {:>8} {:>10} {:>10.1} {:>9.1}", format!("{gh}x{gw}"), p, p / 4, ms, flops / (ms * 1e-3) / 1e12);
    }
}
