//! Decode PNG, JPEG and WebP bytes to 8-bit RGB under an explicit alpha policy, the way the two GLM references convert to RGB.
//!
//! `decode::decode` (the Qwen path, untouched) always composites transparent pixels onto white, which is what ExLlamaV3's `convert_to_rgb`
//! does. HuggingFace's `convert_to_rgb` is `image.convert("RGB")`, which throws the alpha channel away: a fully transparent pixel keeps the
//! colour stored under it. The two give different model inputs for any image with transparency, so the policy is a parameter here
//! ([`AlphaPolicy`]) and each GLM preprocessing profile names the one of its reference. Pillow details reproduced (checked against Pillow 12.3
//! on the fixture images, `testdata/vision/glm`):
//!
//! - `Drop`: RGBA, LA and palette images lose alpha and `tRNS` entirely (`transparency` in `info` is ignored by `convert("RGB")`).
//! - `CompositeWhite`: `convert("RGBA")` honours `tRNS` for palette and gray images (the key colour becomes alpha 0) and then pastes onto white
//!   with Pillow's `MULDIV255`; an RGB image is returned untouched, so an RGB `tRNS` key is ignored.
//! - 16-bit samples: RGB, RGBA and gray+alpha keep the high byte; 16-bit gray (Pillow mode `I;16`) is converted with Pillow's `I;16 -> L` rule,
//!   `high != 0 ? 255 : low`, which saturates every sample of 256 or more (so a 16-bit gray photograph becomes nearly white, in both references).
//! - 1, 2 and 4-bit gray are scaled to 0..255; palettes are expanded.
//!
//! JPEG goes through `decode::decode` (zune-jpeg, EXIF orientation applied; within a few levels of libjpeg-turbo, not bit-exact: see
//! `docs/VISION.md`). WebP goes through `webp::decode` with the same policy (alpha as libwebp delivers it, un-premultiplied; EXIF
//! orientation applied). Animated PNG and WebP, GIF, BMP and the rest are refused, as for Qwen.

use crate::decode::MAX_DECODED_PIXELS;
use crate::{Rgb, VisionError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaPolicy {
    /// HuggingFace `image.convert("RGB")`: the alpha channel is dropped.
    Drop,
    /// ExLlamaV3 `convert_to_rgb`: composited onto white.
    CompositeWhite,
}

fn bad(m: impl Into<String>) -> VisionError {
    VisionError::Decode(m.into())
}

const PNG_MAGIC: [u8; 4] = [0x89, b'P', b'N', b'G'];

/// Decode PNG, JPEG or WebP bytes to RGB under `policy`.
pub fn decode_rgb(bytes: &[u8], policy: AlphaPolicy) -> Result<Rgb, VisionError> {
    if bytes.starts_with(&PNG_MAGIC) {
        decode_png(bytes, policy)
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        crate::decode::decode(bytes)
    } else if crate::webp::is_webp(bytes) {
        crate::webp::decode(bytes, policy)
    } else {
        Err(VisionError::Unsupported(crate::decode::UNSUPPORTED_FORMAT.into()))
    }
}

/// Pillow's `MULDIV255`-based paste of `x` with alpha `a` onto white (the same function as `decode::over_white`).
fn over_white(x: u8, a: u8) -> u8 {
    let mul = |p: u32, q: u32| -> u32 {
        let t = p * q + 128;
        ((t >> 8) + t) >> 8
    };
    let (m, inv) = (a as u32, 255 - a as u32);
    (mul(255, inv) + mul(x as u32, m)).min(255) as u8
}

fn apply(policy: AlphaPolicy, x: u8, a: u8) -> u8 {
    match policy {
        AlphaPolicy::Drop => x,
        AlphaPolicy::CompositeWhite => over_white(x, a),
    }
}

/// Pillow's `I;16 -> L` conversion of one big-endian sample.
fn gray16(hi: u8, lo: u8) -> u8 {
    if hi != 0 {
        255
    } else {
        lo
    }
}

fn decode_png(bytes: &[u8], policy: AlphaPolicy) -> Result<Rgb, VisionError> {
    // IHDR: 8 signature bytes, length, "IHDR", width, height, bit depth (offset 24), colour type (offset 25)
    if bytes.len() < 33 || &bytes[12..16] != b"IHDR" {
        return Err(bad("png: missing IHDR"));
    }
    let (bit_depth, color_type) = (bytes[24], bytes[25]);
    let gray16_image = color_type == 0 && bit_depth == 16;
    // RGB never carries alpha in Pillow (a tRNS key is ignored), so it is not expanded; 16-bit gray keeps both bytes for `gray16`
    let tf = if gray16_image {
        png::Transformations::EXPAND
    } else if color_type == 2 {
        png::Transformations::STRIP_16
    } else {
        png::Transformations::EXPAND | png::Transformations::STRIP_16
    };
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(tf);
    let mut reader = dec.read_info().map_err(|e| bad(format!("png: {e}")))?;
    let info = reader.info();
    if info.is_animated() {
        return Err(VisionError::Unsupported("animated images are not supported".into()));
    }
    let (w, h) = (info.width as usize, info.height as usize);
    if w == 0 || h == 0 || w.checked_mul(h).is_none_or(|p| p > MAX_DECODED_PIXELS) {
        return Err(VisionError::Unsupported(format!("{w}x{h} exceeds the decoded size limit")));
    }
    let mut buf = vec![0u8; reader.output_buffer_size().ok_or_else(|| bad("png: size overflow"))?];
    let fr = reader.next_frame(&mut buf).map_err(|e| bad(format!("png: {e}")))?;
    let data = &buf[..fr.buffer_size()];
    let mut out = Vec::with_capacity(w * h * 3);
    let sixteen = fr.bit_depth == png::BitDepth::Sixteen;
    match (fr.color_type, sixteen) {
        (png::ColorType::Rgb, false) => out.extend_from_slice(data),
        (png::ColorType::Rgba, false) => {
            for p in data.chunks_exact(4) {
                out.extend([apply(policy, p[0], p[3]), apply(policy, p[1], p[3]), apply(policy, p[2], p[3])]);
            }
        }
        (png::ColorType::Grayscale, false) => {
            for &g in data {
                out.extend([g, g, g]);
            }
        }
        (png::ColorType::GrayscaleAlpha, false) => {
            for p in data.chunks_exact(2) {
                let g = apply(policy, p[0], p[1]);
                out.extend([g, g, g]);
            }
        }
        (png::ColorType::Grayscale, true) => {
            for p in data.chunks_exact(2) {
                let g = gray16(p[0], p[1]);
                out.extend([g, g, g]);
            }
        }
        // 16-bit gray with a tRNS key: Pillow's I;16 image plus a `transparency` entry; the sample rule is the same, the key adds alpha
        (png::ColorType::GrayscaleAlpha, true) => {
            for p in data.chunks_exact(4) {
                let g = apply(policy, gray16(p[0], p[1]), p[2]);
                out.extend([g, g, g]);
            }
        }
        (c, s) => return Err(bad(format!("png: unexpected decoded layout {c:?} (16-bit: {s})"))),
    }
    if out.len() != w * h * 3 {
        return Err(bad("png: unexpected sample count"));
    }
    Ok(Rgb { width: w, height: h, data: out })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pillow_rules_for_gray16_and_paste() {
        assert_eq!(
            (gray16(0x00, 0x34), gray16(0x01, 0x00), gray16(0x12, 0x34), gray16(0xff, 0xff), gray16(0, 0)),
            (0x34, 255, 255, 255, 0)
        );
        // a = 0 gives white, a = 255 keeps the value, in between Pillow's rounding
        assert_eq!((over_white(10, 0), over_white(10, 255), over_white(0, 128), over_white(255, 7)), (255, 10, 127, 255));
        assert_eq!((apply(AlphaPolicy::Drop, 10, 0), apply(AlphaPolicy::CompositeWhite, 10, 0)), (10, 255));
    }

    #[test]
    fn refuses_what_it_cannot_read() {
        assert!(matches!(decode_rgb(b"GIF89a....", AlphaPolicy::Drop), Err(VisionError::Unsupported(_))));
        assert!(matches!(decode_rgb(&PNG_MAGIC, AlphaPolicy::Drop), Err(VisionError::Decode(_))));
        let mut truncated = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13];
        truncated.extend_from_slice(b"IHDR");
        truncated.extend_from_slice(&[0; 17]);
        assert!(decode_rgb(&truncated, AlphaPolicy::CompositeWhite).is_err());
    }
}
