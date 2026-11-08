//! WebP decode against Pillow (libwebp): every fixture under `testdata/vision/webp/` decodes to the very bytes Pillow gives, under both alpha
//! policies (`testdata/vision/webp/pil.json`, written by `tools/gen_webp_fixtures.py`: lossy at quality 0, 75 and 100 and methods 0, 4 and
//! 6, lossy with alpha, lossless with and without alpha, a 4-colour palette (pixel bundling), grayscale, EXIF orientations 6 and 8). The
//! wider sweep (1,077 files, every quality 0 to 100, the RealWorldQA rows, hand-built filtered alpha planes) is in `docs/VISION.md`, section "WebP"; it found no differing
//! byte. Also here: the format is sniffed from the bytes (a data URI's declared type does not matter), animation is refused by name, and the
//! decoded pixels reach the Nemotron preprocessing exactly as a PNG of the same pixels does.

use gb_vision::decode::{self, data_uri_bytes, UNSUPPORTED_FORMAT};
use gb_vision::nemotron::NemotronPreprocessing;
use gb_vision::preprocessor::ImagePreprocessor;
use gb_vision::rgb_decode::{decode_rgb, AlphaPolicy};
use gb_vision::VisionError;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision/webp")
}

fn read(name: &str) -> Vec<u8> {
    std::fs::read(dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn manifest() -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir().join("pil.json")).unwrap()).unwrap()
}

fn sha(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

fn b64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            s.push(if i <= c.len() { T[(n >> (18 - 6 * i)) as usize & 63] as char } else { '=' });
        }
    }
    s
}

#[test]
fn every_fixture_is_pillows_bytes_under_both_alpha_policies() {
    let m = manifest();
    let fx = m["fixtures"].as_object().unwrap();
    assert!(fx.len() >= 12);
    let kinds: std::collections::BTreeSet<&str> = fx.values().map(|v| v["chunk"].as_str().unwrap()).collect();
    assert_eq!(
        kinds.into_iter().collect::<Vec<_>>(),
        ["VP8 ", "VP8L", "VP8X"],
        "simple lossy, simple lossless and extended files"
    );
    for (name, want) in fx {
        let b = read(name);
        let size = (want["width"].as_u64().unwrap() as usize, want["height"].as_u64().unwrap() as usize);
        let drop = decode_rgb(&b, AlphaPolicy::Drop).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!((drop.width, drop.height), size, "{name}");
        assert_eq!(sha(&drop.data), want["drop_sha256"].as_str().unwrap(), "{name}: alpha dropped (convert(\"RGB\"))");
        let white = decode_rgb(&b, AlphaPolicy::CompositeWhite).unwrap();
        assert_eq!(sha(&white.data), want["white_sha256"].as_str().unwrap(), "{name}: pasted onto white");
        // the Qwen path composites onto white, like the PNG path
        assert_eq!(decode::decode(&b).unwrap().data, white.data, "{name}: decode::decode");
    }
}

#[test]
fn the_exif_orientation_is_applied_like_exif_transpose() {
    // 37 x 21 sources: orientations 6 and 8 swap the sides (the manifest's sizes are those of `exif_transpose`)
    for n in ["lossless_exif6_37x21.webp", "lossy_exif8_37x21.webp"] {
        let img = decode_rgb(&read(n), AlphaPolicy::Drop).unwrap();
        assert_eq!((img.width, img.height), (21, 37), "{n}");
    }
}

#[test]
fn animation_is_refused_by_name() {
    let m = manifest();
    for n in m["animated"].as_array().unwrap() {
        let b = read(n.as_str().unwrap());
        for r in [decode::decode(&b), decode_rgb(&b, AlphaPolicy::Drop), decode_rgb(&b, AlphaPolicy::CompositeWhite)] {
            assert_eq!(r.map(|i| i.width), Err(VisionError::Unsupported("animated images are not supported".into())));
        }
    }
}

#[test]
fn the_bytes_decide_the_format_not_the_declared_media_type() {
    let b = read("lossy_q75_97x61.webp");
    let want = decode::decode(&b).unwrap();
    for mime in ["image/webp", "image/png", "image/jpeg", "application/octet-stream"] {
        let bytes = data_uri_bytes(&format!("data:{mime};base64,{}", b64(&b))).unwrap();
        assert_eq!(decode::decode(&bytes).unwrap(), want, "{mime}");
    }
    // a RIFF file of another kind, and the refusal text
    let mut wave = b.clone();
    wave[8..12].copy_from_slice(b"WAVE");
    for r in [decode::decode(&wave), decode_rgb(&wave, AlphaPolicy::Drop)] {
        assert_eq!(r.map(|i| i.width), Err(VisionError::Unsupported(UNSUPPORTED_FORMAT.into())));
    }
    assert_eq!(UNSUPPORTED_FORMAT, "only PNG, JPEG and WebP images are supported");
}

/// A PNG (8-bit RGB) of `img`'s pixels.
fn png_of(img: &gb_vision::Rgb) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut e = png::Encoder::new(&mut out, img.width as u32, img.height as u32);
        e.set_color(png::ColorType::Rgb);
        e.set_depth(png::BitDepth::Eight);
        e.write_header().unwrap().write_image_data(&img.data).unwrap();
    }
    out
}

#[test]
fn the_nemotron_preprocessing_sees_the_same_pixels_as_from_a_png() {
    // what reaches the RADIO tower: the canvas, its float pixels and its hash, for a WebP and for a PNG of Pillow's decode of it
    let pre = NemotronPreprocessing::default();
    for n in
        ["lossy_q75_97x61.webp", "lossy_alpha_q80_63x65.webp", "lossless_alpha_exact_63x65.webp", "lossless_exif6_37x21.webp"]
    {
        let b = read(n);
        let from_webp = pre.decode(&b).unwrap();
        let png = png_of(&from_webp);
        let from_png = pre.decode(&png).unwrap();
        assert_eq!(from_webp, from_png, "{n}");
        let (a, c) = (pre.preprocess(&from_webp).unwrap(), pre.preprocess(&from_png).unwrap());
        assert_eq!((a.canvas(), a.tokens, a.pixel_hash), (c.canvas(), c.tokens, c.pixel_hash), "{n}");
        assert_eq!(a.pixels, c.pixels, "{n}");
    }
}
