//! Byte-for-byte comparison with PIL and ExLlamaV3 on the fixtures from `tools/gen_vision_fixtures.py`.

use gb_vision::{decode, preprocess, resize, smart_resize, Preprocessing};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision")
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn sha(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

fn cases() -> Vec<Value> {
    let t = std::fs::read_to_string(dir().join("fixtures.json")).expect("run tools/gen_vision_fixtures.py");
    serde_json::from_str::<Value>(&t).unwrap()["cases"].as_array().unwrap().clone()
}

#[test]
fn smart_resize_and_resized_pixels_match_pil_exactly() {
    let p = Preprocessing::qwen38();
    let mut worst_jpeg = 0u8;
    for c in cases() {
        let name = c["name"].as_str().unwrap();
        let bytes = std::fs::read(dir().join(c["file"].as_str().unwrap())).unwrap();
        let img = decode::decode(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            [img.width as u64, img.height as u64],
            [c["src_size"][0].as_u64().unwrap(), c["src_size"][1].as_u64().unwrap()],
            "{name}"
        );
        let (nw, nh) = smart_resize(img.width, img.height, &p).unwrap();
        assert_eq!(
            [nw as u64, nh as u64],
            [c["resized_size"][0].as_u64().unwrap(), c["resized_size"][1].as_u64().unwrap()],
            "{name}: smart_resize"
        );
        let resized = if (nw, nh) != (img.width, img.height) {
            resize::bicubic_rgb(&img.data, img.width, img.height, nw, nh)
        } else {
            img.data.clone()
        };
        if name.starts_with("jpeg") {
            // libjpeg-turbo (Pillow) and zune-jpeg may differ in the last bit of the decode: compare the decoded pixels directly
            let unhex = |h: &str| -> Vec<u8> {
                (0..h.len() / 2).map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap()).collect()
            };
            if let Some(h) = c.get("decoded_hex") {
                let want = unhex(h.as_str().unwrap());
                let d = img.data.iter().zip(&want).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
                eprintln!("{name}: decoded pixels differ from PIL by at most {d}/255");
                worst_jpeg = worst_jpeg.max(d);
            } else {
                eprintln!("{name}: decoded bytes identical to PIL: {}", sha(&img.data) == c["decoded_sha256"].as_str().unwrap());
            }
            continue;
        }
        assert_eq!(sha(&resized), c["resized_sha256"].as_str().unwrap(), "{name}: resized pixels differ from PIL");
    }
    eprintln!("worst JPEG decode difference vs PIL: {worst_jpeg}/255");
    // zune-jpeg is not bit-identical to libjpeg-turbo (upsampling and IDCT rounding): observed up to 4/255 per channel. PNG is exact.
    assert!(worst_jpeg <= 4, "JPEG decode differs from libjpeg-turbo by {worst_jpeg}");
}

/// The same resized pixels as above for the kernel on exact thread counts (the fixtures are small, so the default policy would run them inline):
/// every committed Qwen fixture that resizes must equal PIL's bytes however its rows are split, and so must the entry with the policy and a cap.
#[test]
fn resized_pixels_match_pil_on_every_thread_count() {
    let p = Preprocessing::qwen38();
    let mut resized = 0;
    for c in cases() {
        let name = c["name"].as_str().unwrap();
        if name.starts_with("jpeg") {
            continue; // the decoder is not byte-exact; the resize of its pixels is covered by the other fixtures
        }
        let bytes = std::fs::read(dir().join(c["file"].as_str().unwrap())).unwrap();
        let img = decode::decode(&bytes).unwrap();
        let (nw, nh) = smart_resize(img.width, img.height, &p).unwrap();
        if (nw, nh) == (img.width, img.height) {
            continue;
        }
        resized += 1;
        let want = c["resized_sha256"].as_str().unwrap();
        for t in [1usize, 2, 3, 5, 8, 16, 64] {
            let out = resize::bicubic_rgb_threads(&img.data, img.width, img.height, nw, nh, t);
            assert_eq!(sha(&out), want, "{name}: {t} threads differ from PIL");
        }
        for cap in [0usize, 1, 4] {
            let out = resize::bicubic_rgb_with(&img.data, img.width, img.height, nw, nh, cap);
            assert_eq!(sha(&out), want, "{name}: policy with a cap of {cap} differs from PIL");
        }
    }
    assert!(resized >= 8, "{resized} resizing fixtures");
}

#[test]
fn patches_match_exllamav3_preprocess_exactly() {
    let p = Preprocessing::qwen38();
    for c in cases() {
        let name = c["name"].as_str().unwrap();
        if name.starts_with("jpeg") {
            continue; // decoder-dependent; covered by the tolerance above
        }
        let bytes = std::fs::read(dir().join(c["file"].as_str().unwrap())).unwrap();
        let out = preprocess(&decode::decode(&bytes).unwrap(), &p).unwrap();
        assert_eq!(
            out.grid,
            (
                c["grid_thw"][0].as_u64().unwrap() as usize,
                c["grid_thw"][1].as_u64().unwrap() as usize,
                c["grid_thw"][2].as_u64().unwrap() as usize
            ),
            "{name}"
        );
        assert_eq!(out.n_patches() as u64, c["n_patches"].as_u64().unwrap());
        let raw: Vec<u8> = out.patches.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(sha(&raw), c["patches_sha256"].as_str().unwrap(), "{name}: patch tensor differs from ExLlamaV3");
        assert_eq!(out.patch_dim, 1536);
    }
}

#[test]
fn hash_depends_on_pixels_and_size_only() {
    let p = Preprocessing::qwen38();
    let bytes = std::fs::read(dir().join("check_512x512.png")).unwrap();
    let img = decode::decode(&bytes).unwrap();
    let a = preprocess(&img, &p).unwrap();
    let b = preprocess(&img, &p).unwrap();
    assert_eq!(a.pixel_hash, b.pixel_hash);
    let mut other = img.clone();
    other.data[100] ^= 1;
    assert_ne!(a.pixel_hash, preprocess(&other, &p).unwrap().pixel_hash);
}

#[test]
fn position_embedding_plan_and_rope_angles_match_exllamav3() {
    let t = std::fs::read_to_string(dir().join("tables.json")).expect("run tools/gen_vision_fixtures.py");
    let v: Value = serde_json::from_str(&t).unwrap();
    let (mut idx_bad, mut w_bad, mut worst_ulp) = (0usize, 0usize, 0u32);
    let mut total = 0usize;
    for c in v["cases"].as_array().unwrap() {
        let (gh, gw) = (c["grid"][0].as_u64().unwrap() as usize, c["grid"][1].as_u64().unwrap() as usize);
        let (idx, w) = gb_vision::tables::pos_embed_plan(gh, gw);
        let want_idx: Vec<i64> = c["idx"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect();
        let want_w: Vec<f64> = c["w"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        assert_eq!(idx.len(), want_idx.len());
        for (a, b) in idx.iter().zip(&want_idx) {
            idx_bad += (*a as i64 != *b) as usize;
        }
        for (a, b) in w.iter().zip(&want_w) {
            w_bad += (*a as f64 != *b) as usize;
        }
        total += idx.len();
        let ang = gb_vision::tables::rope_angles(gh, gw, 10000.0);
        let want: Vec<u32> = c["angles_bits"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        assert_eq!(ang.len(), want.len());
        for (a, b) in ang.iter().zip(&want) {
            worst_ulp = worst_ulp.max(a.to_bits().abs_diff(*b));
        }
    }
    eprintln!("position plan: {idx_bad} index and {w_bad} weight mismatches of {total}; rope angles worst {worst_ulp} ulp");
    assert_eq!(idx_bad, 0, "table indices must match exactly");
    assert_eq!(w_bad, 0, "bf16 weights must match exactly");
    assert!(worst_ulp <= 2, "rope angles differ by {worst_ulp} ulp");
}
