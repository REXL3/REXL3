//! The MiniCPM-V-4.7 image path end to end, stand-alone (E5a): an image file through the family's preprocessing, the tower, the window
//! merger and the projector, against the Hugging Face model (`tools/minicpmv_vision_oracle.py`).
//!
//! Needs a device, `REXL3_MINICPMV47_DIR` (the bf16 checkpoint folder or an EXL3 pack) and the oracle directory
//! (`REXL3_MCV_ORACLE_DIR`, default `~/projects/minicpm-vision-oracle`; `tools/minicpmv_vision_oracle.py refs`, then `images`). Skips with
//! `not run: ...` otherwise. The per-stage gates (embed, layer 6, merger, layer 26, post_layernorm) are `crates/cuda/tests/mcv_vision.rs`
//! over the oracle's own pixels; this test adds the Rust preprocessing in front (its bytes must equal the HF processor's) and the
//! family's encoder seam behind (`ImageEncoder::encode` on the `Preprocessed`).

use gb_cuda::device_available;
use gb_model::families::minicpmv4_7::vision::{
    mode_of, MiniCpmEncoder, MiniCpmFeatures, MiniCpmTower, DEFAULT_MAX_PATCHES, OUT_HIDDEN,
};
use gb_model::serve::{feature_ref, FeatureBudget, ImageEncoder};
use gb_vision::minicpmv::{Downsample, MiniCpmPreprocessing, ResizeProfile};
use gb_vision::preprocessor::ImagePreprocessor;
use std::path::{Path, PathBuf};

fn dirs() -> Option<(PathBuf, PathBuf)> {
    if !device_available() {
        eprintln!("not run: no CUDA device");
        return None;
    }
    let Some(ck) = std::env::var_os("REXL3_MINICPMV47_DIR").map(PathBuf::from) else {
        eprintln!("not run: REXL3_MINICPMV47_DIR is not set");
        return None;
    };
    let odir = std::env::var_os("REXL3_MCV_ORACLE_DIR")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("projects/minicpm-vision-oracle")))?;
    if !odir.join("manifest.json").exists() {
        eprintln!("not run: no oracle manifest in {}", odir.display());
        return None;
    }
    Some((ck, odir))
}

fn read_f32(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn rel_l2_cos(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let (mut d2, mut b2, mut a2, mut ab) = (0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        d2 += (x - y) * (x - y);
        b2 += y * y;
        a2 += x * x;
        ab += x * y;
    }
    ((d2 / b2).sqrt(), ab / (a2.sqrt() * b2.sqrt()))
}

fn bf16_round(v: f32) -> u16 {
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

#[test]
fn files_through_the_preprocessing_the_tower_and_the_projector_match_hugging_face() {
    let Some((ck, odir)) = dirs() else { return };
    let mut names: Vec<String> = std::fs::read_dir(&odir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("input.png").exists() && e.path().join("plan.txt").exists())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    assert!(names.len() >= 10, "{} oracle images in {} (run the oracle's `images` command)", names.len(), odir.display());
    let tower = MiniCpmTower::load(&ck, DEFAULT_MAX_PATCHES).unwrap();
    let mut enc = MiniCpmEncoder::new(
        tower,
        MiniCpmPreprocessing::from_dir(&ck).unwrap().with_profile(ResizeProfile::Pil),
        FeatureBudget::new(1 << 30),
    );
    println!(
        "{:<28} {:>5} {:>6} {:>7} {:>10} {:>10} {:>10}",
        "case", "mode", "crops", "tokens", "rel_l2|f32", "rel_l2|b16", "cosine"
    );
    let mut worst = 0f64;
    for n in &names {
        let d = odir.join(n);
        let plan_txt = std::fs::read_to_string(d.join("plan.txt")).unwrap();
        let mut lines = plan_txt.lines();
        let mode_name = lines.next().unwrap().trim().to_string();
        let sizes: Vec<u32> = lines.next().unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
        let want_sizes: Vec<[u32; 2]> = sizes.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
        // the preprocessing under this case's mode, with the PIL backend the oracle venv runs
        let pre = MiniCpmPreprocessing::from_dir(&ck)
            .unwrap()
            .with_profile(ResizeProfile::Pil)
            .with_downsample(Downsample::parse(&mode_name).unwrap());
        let rgb = ImagePreprocessor::decode(&pre, &std::fs::read(d.join("input.png")).unwrap()).unwrap();
        let input = pre.preprocess_crops(&rgb).unwrap();
        // 1. the bytes the tower reads are the processor's
        let want_px: Vec<u16> =
            std::fs::read(d.join("pixels.u16")).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(input.target_sizes, want_sizes, "{n}: target sizes");
        assert!(
            input.pixels == want_px,
            "{n}: the preprocessed pixels differ from the HF processor's ({} values)",
            want_px.len()
        );
        // 2. the tower on them
        let mode = mode_of(pre.downsample);
        let got = enc.tower().encode_packed_f32(&input.pixels, &input.target_sizes, mode).unwrap();
        let w32 = read_f32(&d.join("projector.fp32.f32"));
        let w16 = read_f32(&d.join("projector.bf16.f32"));
        let ((r32, cos), (r16, _)) = (rel_l2_cos(&got, &w32), rel_l2_cos(&got, &w16));
        println!(
            "{n:<28} {mode_name:>5} {:>6} {:>7} {r32:>10.3e} {r16:>10.3e} {cos:>10.6}",
            input.target_sizes.len(),
            got.len() / OUT_HIDDEN
        );
        assert!(r32 < 4.0e-3, "{n}: projector rel_l2 vs fp32 {r32:.3e}");
        assert!(cos > 0.99999, "{n}: cosine {cos}");
        worst = worst.max(r32);
        // 3. the seam: Preprocessed -> EncodedImage, features bf16 on the device equal the rounded fp32 rows
        let seam = ImagePreprocessor::preprocess(&pre, &rgb).unwrap();
        enc.set_preprocessing(pre.clone());
        let e = enc.encode(&seam).unwrap();
        assert_eq!((e.tokens, e.merged), (got.len() / OUT_HIDDEN, (1, got.len() / OUT_HIDDEN)), "{n}");
        let f = feature_ref::<MiniCpmFeatures>(e.features.as_ref()).expect("MiniCpmFeatures");
        assert_eq!(f.tokens, e.tokens);
        let mut rows = vec![0u16; e.tokens * OUT_HIDDEN];
        f.buf.download_range(0, &mut rows).unwrap();
        let want_rows: Vec<u16> = got.iter().map(|&v| bf16_round(v)).collect();
        assert!(rows == want_rows, "{n}: the device bf16 rows are not the rounded fp32 rows");
        let covered: usize = f.crops.iter().map(|c| c.grid.0 * c.grid.1).sum();
        assert_eq!(covered, e.tokens, "{n}: crop rows cover the features");
    }
    println!("worst projector rel_l2 vs fp32: {worst:.3e}");
}

#[test]
fn an_encode_that_does_not_match_its_plan_is_refused_by_name() {
    let Some((ck, _)) = dirs() else { return };
    let tower = MiniCpmTower::load(&ck, DEFAULT_MAX_PATCHES).unwrap();
    let pre = MiniCpmPreprocessing::from_dir(&ck).unwrap();
    let mut enc = MiniCpmEncoder::new(tower, pre.clone(), FeatureBudget::new(1 << 30));
    let rgb = gb_vision::Rgb { width: 448, height: 448, data: vec![127; 448 * 448 * 3] };
    let mut seam = ImagePreprocessor::preprocess(&pre, &rgb).unwrap();
    assert!(enc.encode(&seam).is_ok());
    seam.patches.pop();
    let e = enc.encode(&seam).err().expect("a short pixel tensor is refused").to_string();
    assert!(e.contains("does not match the crop plan"), "{e}");
    // a feature budget that cannot hold the image is Busy, not a crash
    let tiny = MiniCpmTower::load(&ck, 1024).unwrap();
    let mut enc2 = MiniCpmEncoder::new(tiny, pre.clone(), FeatureBudget::new(1000));
    let ok = ImagePreprocessor::preprocess(&pre, &rgb).unwrap();
    let e = enc2.encode(&ok).err().expect("busy").to_string();
    assert!(e.contains("memory for image features"), "{e}");
}

#[test]
fn the_memory_arithmetic_equals_the_allocations() {
    use gb_model::families::minicpmv4_7::vision::{weight_bytes, workspace_bytes};
    let Some((ck, _)) = dirs() else { return };
    for max_patches in [1024usize, DEFAULT_MAX_PATCHES] {
        let tower = MiniCpmTower::load(&ck, max_patches).unwrap();
        let (w, s) = tower.bytes();
        assert_eq!(w, weight_bytes(), "weights");
        assert_eq!(s, workspace_bytes(max_patches), "workspace of {max_patches} patches");
    }
}
