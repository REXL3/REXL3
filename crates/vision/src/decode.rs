//! Decode PNG, JPEG and WebP bytes (and `data:` URIs) to 8-bit RGB the way ExLlamaV3's `convert_to_rgb` leaves them: transparent
//! regions composited onto white, grayscale and palette images expanded, and the EXIF orientation of a JPEG or WebP applied.

use crate::{Rgb, VisionError};

fn bad(m: impl Into<String>) -> VisionError {
    VisionError::Decode(m.into())
}

/// Decoded images may not exceed this many pixels before any resize (a decompression-bomb guard).
pub const MAX_DECODED_PIXELS: usize = 1 << 26;

/// Standard base64 (RFC 4648, `=` padding optional, whitespace ignored).
pub fn base64_decode(s: &str) -> Result<Vec<u8>, VisionError> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for b in s.bytes() {
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => return Err(bad("invalid base64")),
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

/// The bytes of a `data:image/...;base64,...` URI. Any other URL (http, file, ...) is refused.
pub fn data_uri_bytes(url: &str) -> Result<Vec<u8>, VisionError> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| VisionError::Unsupported("only data: URIs are accepted for images (remote fetch is disabled)".into()))?;
    let (head, body) = rest.split_once(',').ok_or_else(|| bad("malformed data URI"))?;
    if !head.ends_with(";base64") {
        return Err(VisionError::Unsupported("only base64 data URIs are accepted".into()));
    }
    base64_decode(body)
}

/// The refusal for any other format (GIF, BMP, TIFF, HEIC, SVG, a RIFF file that is not WebP, ...).
pub const UNSUPPORTED_FORMAT: &str = "only PNG, JPEG and WebP images are supported";

/// Decode an image file's bytes. The format is sniffed from the bytes (PNG and JPEG signatures, `RIFF....WEBP`); the media type a data URI
/// or an HTTP response declares is never consulted.
pub fn decode(bytes: &[u8]) -> Result<Rgb, VisionError> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        decode_png(bytes)
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        decode_jpeg(bytes)
    } else if crate::webp::is_webp(bytes) {
        crate::webp::decode(bytes, crate::rgb_decode::AlphaPolicy::CompositeWhite)
    } else {
        Err(VisionError::Unsupported(UNSUPPORTED_FORMAT.into()))
    }
}

/// `x` over white with alpha `a`, Pillow's `paste` onto a white RGBA canvas then `convert("RGB")`: `(x*a + 255*(255-a) + 127) / 255` style
/// rounding is Pillow's `MULDIV255`, reproduced here.
fn over_white(x: u8, a: u8) -> u8 {
    // Pillow ImagingPaste with a mask on an RGBA canvas: out = BLEND(canvas, src, mask) = (canvas*(255-m) + src*m) / 255 rounded,
    // via MULDIV255(a, b) = (tmp = a*b + 128, ((tmp >> 8) + tmp) >> 8)
    let mul = |p: u32, q: u32| -> u32 {
        let t = p * q + 128;
        ((t >> 8) + t) >> 8
    };
    let (m, inv) = (a as u32, 255 - a as u32);
    (mul(255, inv) + mul(x as u32, m)).min(255) as u8
}

fn decode_png(bytes: &[u8]) -> Result<Rgb, VisionError> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    // expand palettes and low bit depths, and strip 16-bit samples to 8 (Pillow keeps the high byte)
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
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
    match fr.color_type {
        png::ColorType::Rgb => out.extend_from_slice(data),
        png::ColorType::Rgba => {
            for p in data.chunks_exact(4) {
                out.extend([over_white(p[0], p[3]), over_white(p[1], p[3]), over_white(p[2], p[3])]);
            }
        }
        png::ColorType::Grayscale => {
            for &g in data {
                out.extend([g, g, g]);
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for p in data.chunks_exact(2) {
                let g = over_white(p[0], p[1]);
                out.extend([g, g, g]);
            }
        }
        png::ColorType::Indexed => return Err(bad("png: palette was not expanded")),
    }
    Ok(Rgb { width: w, height: h, data: out })
}

/// EXIF orientation (1..8) of a JPEG, 1 when absent.
fn jpeg_orientation(bytes: &[u8]) -> u16 {
    let mut i = 2;
    while i + 4 <= bytes.len() && bytes[i] == 0xff {
        let marker = bytes[i + 1];
        let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
        if len < 2 {
            break; // a length that cannot cover its own two bytes: no segment to read (the slice below would start after it ends)
        }
        if marker == 0xe1 && i + 4 + len.saturating_sub(2) <= bytes.len() {
            let seg = &bytes[i + 4..i + 2 + len];
            if seg.len() > 14 && &seg[..6] == b"Exif\0\0" {
                return tiff_orientation(&seg[6..]);
            }
            return 1;
        }
        if marker == 0xda {
            break;
        }
        i += 2 + len;
    }
    1
}

/// The orientation tag (0x0112, 1..8) of a TIFF-structured EXIF block (`II*\0` or `MM\0*` then IFD0), 1 when absent or malformed. Total on
/// any bytes: every read is bounds checked. Shared by the JPEG APP1 segment and the WebP `EXIF` chunk.
pub(crate) fn tiff_orientation(t: &[u8]) -> u16 {
    if t.len() < 8 {
        return 1;
    }
    let le = &t[..2] == b"II";
    let r16 = |o: usize| -> Option<u16> {
        let b = t.get(o..o.checked_add(2)?)?;
        Some(if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) })
    };
    let r32 = |o: usize| -> Option<u32> {
        let b = t.get(o..o.checked_add(4)?)?;
        Some(if le { u32::from_le_bytes([b[0], b[1], b[2], b[3]]) } else { u32::from_be_bytes([b[0], b[1], b[2], b[3]]) })
    };
    if let (Some(off), Some(_)) = (r32(4), r16(2)) {
        let off = off as usize;
        if let Some(n) = r16(off) {
            for e in 0..n as usize {
                let Some(at) = off.checked_add(2 + e * 12) else { break };
                if at >= t.len() {
                    break;
                }
                if r16(at) == Some(0x0112) {
                    return r16(at + 8).filter(|v| (1..=8).contains(v)).unwrap_or(1);
                }
            }
        }
    }
    1
}

/// Apply EXIF orientation `o` (Pillow `ImageOps.exif_transpose` semantics) to an RGB image.
pub(crate) fn orient(img: Rgb, o: u16) -> Rgb {
    if o == 1 {
        return img;
    }
    let (w, h) = (img.width, img.height);
    let (nw, nh) = if matches!(o, 5..=8) { (h, w) } else { (w, h) };
    let mut out = vec![0u8; img.data.len()];
    for y in 0..nh {
        for x in 0..nw {
            // source coordinate for each orientation
            let (sx, sy) = match o {
                2 => (w - 1 - x, y),
                3 => (w - 1 - x, h - 1 - y),
                4 => (x, h - 1 - y),
                5 => (y, x),
                6 => (y, h - 1 - x),
                7 => (w - 1 - y, h - 1 - x),
                8 => (w - 1 - y, x),
                _ => (x, y),
            };
            let s = (sy * w + sx) * 3;
            let d = (y * nw + x) * 3;
            out[d..d + 3].copy_from_slice(&img.data[s..s + 3]);
        }
    }
    Rgb { width: nw, height: nh, data: out }
}

fn decode_jpeg(bytes: &[u8]) -> Result<Rgb, VisionError> {
    use zune_jpeg::zune_core::{colorspace::ColorSpace, options::DecoderOptions};
    let opts = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB).set_max_width(16384).set_max_height(16384);
    let mut dec = zune_jpeg::JpegDecoder::new_with_options(zune_jpeg::zune_core::bytestream::ZCursor::new(bytes), opts);
    // the start-of-frame size decides what `decode` allocates (width x height x 3 bytes, plus the component planes): read it first and refuse an
    // image over the pixel limit before any of that exists. The decoder's own cap (16,384 per side) alone lets a 1 KB file cost 770 MiB and 2 s.
    dec.decode_headers().map_err(|e| bad(format!("jpeg: {e:?}")))?;
    let (w, h) = dec.dimensions().ok_or_else(|| bad("jpeg: no dimensions"))?;
    if w.checked_mul(h).is_none_or(|p| p > MAX_DECODED_PIXELS) {
        return Err(VisionError::Unsupported(format!("{w}x{h} exceeds the decoded size limit")));
    }
    let data = dec.decode().map_err(|e| bad(format!("jpeg: {e:?}")))?;
    if data.len() != w * h * 3 {
        return Err(bad("jpeg: unexpected sample count"));
    }
    Ok(orient(Rgb { width: w, height: h, data }, jpeg_orientation(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_and_data_uris() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVsbG8").unwrap(), b"hello");
        assert_eq!(base64_decode("aGVs\nbG8=").unwrap(), b"hello");
        assert!(base64_decode("a$b").is_err());
        assert_eq!(data_uri_bytes("data:image/png;base64,aGk=").unwrap(), b"hi");
        assert!(matches!(data_uri_bytes("https://example.com/x.png"), Err(VisionError::Unsupported(_))));
        assert!(matches!(data_uri_bytes("data:image/png,raw"), Err(VisionError::Unsupported(_))));
    }

    #[test]
    fn garbage_is_a_decode_or_unsupported_error_not_a_panic() {
        assert!(decode(b"not an image").is_err());
        assert!(decode(&[0x89, b'P', b'N', b'G', 0, 0]).is_err());
        assert!(decode(&[0xff, 0xd8, 0xff]).is_err());
    }

    #[test]
    fn a_segment_length_under_two_is_not_a_slice_and_the_orientation_stays_one() {
        // an APP1 whose length field cannot even cover itself: `jpeg_orientation` runs on every image the JPEG decoder accepted, so it must be
        // total on any bytes (the decoder rejects such a segment today; this keeps it that way if it ever skips one)
        for len in [0u8, 1] {
            let b = [0xff, 0xd8, 0xff, 0xe1, 0, len, b'E', b'x', b'i', b'f', 0, 0, b'I', b'I', 42, 0, 8, 0, 0, 0, 0xff, 0xd9];
            assert_eq!(jpeg_orientation(&b), 1, "APP1 length {len}");
        }
        // a truncated header of every length, and an honest orientation block still read
        let mut ok = vec![0xff, 0xd8, 0xff, 0xe1, 0, 34, b'E', b'x', b'i', b'f', 0, 0, b'I', b'I', 42, 0, 8, 0, 0, 0];
        ok.extend_from_slice(&[1, 0, 0x12, 0x01, 3, 0, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(jpeg_orientation(&ok), 6);
        for cut in 0..ok.len() {
            let _ = jpeg_orientation(&ok[..cut]);
        }
    }

    #[test]
    fn orientation_6_rotates_clockwise() {
        // 2x1 image [A B] with EXIF 6 (rotate 90 cw) becomes a 1x2 image [A; B]
        let img = Rgb { width: 2, height: 1, data: vec![1, 1, 1, 2, 2, 2] };
        let o = orient(img, 6);
        assert_eq!((o.width, o.height), (1, 2));
        assert_eq!(o.data, vec![1, 1, 1, 2, 2, 2]);
    }
}
