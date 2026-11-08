//! WebP (`RIFF....WEBP`): lossy VP8, lossless VP8L, with or without alpha, still images only, decoded to 8-bit RGB under an
//! [`AlphaPolicy`] like the PNG path.
//!
//! The pixel decoder is the `image-webp` crate (image-rs, MIT OR Apache-2.0, `#![forbid(unsafe_code)]`). Everything it is handed has first
//! been checked here, by a RIFF walker that is total on any bytes and allocates nothing in proportion to what a header claims:
//!
//! - the RIFF size must fit the input (a truncated file is a decode error, as libwebp and so Pillow refuse it); every chunk must fit the
//!   RIFF payload;
//! - the image size comes from the VP8 / VP8L bitstream header itself (and, for the extended `VP8X` layout, must equal the canvas): the
//!   crate sizes its VP8 planes from the bitstream header and compares with the canvas only afterwards, so a `VP8X` declaring 1 x 1 around
//!   a 16,383 x 16,383 VP8 frame would cost 400 MB before its error. Over [`MAX_DECODED_PIXELS`] is refused here, from the header;
//! - animation (the `VP8X` animation flag, an `ANIM` or `ANMF` chunk) is refused by name, as an animated PNG is: the crate would quietly
//!   return the first frame;
//! - the crate's metadata readers (`icc_profile`, `exif_metadata`) are never called (they allocate a chunk's claimed size): the `EXIF` chunk
//!   is read here from the validated slice, its orientation applied as for a JPEG (Pillow `ImageOps.exif_transpose`, which HF `load_image`
//!   runs on every format); an ICC profile is ignored, as in the PNG and JPEG paths and in `convert("RGB")`.
//!
//! Peak heap of a decode, per pixel: lossy 1.5 (Y, U, V planes) + 3 (the output), + 1 with alpha (the plane, decoded first: a lossless
//! alpha stream needs 4 while it is decoded, alone); lossless 4 (the ARGB stream, decoded straight into the output). An RGBA result is turned
//! into RGB in place, so no second full-size buffer exists. Measured in `tests/alloc_bounds.rs`.
//!
//! Pixel parity with libwebp (Pillow's decoder): see `docs/VISION.md`, section "WebP".

use crate::decode::{orient, tiff_orientation, MAX_DECODED_PIXELS};
use crate::rgb_decode::AlphaPolicy;
use crate::{Rgb, VisionError};
use std::ops::Range;

fn bad(m: impl Into<String>) -> VisionError {
    VisionError::Decode(format!("webp: {}", m.into()))
}

/// The largest side a VP8 or VP8L bitstream can describe (14 bits).
pub const MAX_SIDE: usize = 16383;

/// `RIFF` + size + `WEBP`.
pub fn is_webp(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP"
}

/// What the walker learned from a still WebP's headers.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Header {
    pub width: usize,
    pub height: usize,
    pub lossless: bool,
    /// Whether the file declares alpha (the VP8L hint bit, the VP8X flag): Pillow's mode RGBA, else RGB.
    pub alpha: bool,
    /// The EXIF orientation (1 when absent).
    pub orientation: u16,
    /// The RIFF slice (`8 + riff size` bytes): what the pixel decoder is given (trailing bytes after it are ignored, as libwebp does).
    pub riff_len: usize,
    /// A lossy image in the extended layout whose `VP8X` declares alpha: the `VP8 ` payload and the `ALPH` payload (absent: opaque, as
    /// libwebp decodes it), as byte ranges of the input. Decoded in two steps (see [`decode`]).
    pub lossy_alpha: Option<(Range<usize>, Option<Range<usize>>)>,
}

fn le16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn le24(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], 0])
}

/// Width and height of a VP8 (lossy) key frame: frame tag (3 bytes), start code `9d 01 2a`, then 14-bit width and height (the two scale
/// bits above them are ignored, by libwebp too).
fn vp8_size(p: &[u8]) -> Result<(usize, usize), VisionError> {
    if p.len() < 10 {
        return Err(bad("VP8 chunk too short"));
    }
    if p[0] & 1 != 0 {
        return Err(bad("VP8 frame is not a key frame"));
    }
    if p[3..6] != [0x9d, 0x01, 0x2a] {
        return Err(bad("bad VP8 start code"));
    }
    // the first partition must fit the chunk (19 bits of the frame tag)
    let first = (le24(p, 0) >> 5) as usize;
    if first > p.len() - 10 {
        return Err(bad("VP8 first partition runs past the chunk"));
    }
    Ok(((le16(p, 6) & 0x3fff) as usize, (le16(p, 8) & 0x3fff) as usize))
}

/// Width and height of a VP8L (lossless) stream: signature 0x2f, then 14-bit width - 1, 14-bit height - 1, alpha hint, 3-bit version 0.
fn vp8l_size(p: &[u8]) -> Result<(usize, usize), VisionError> {
    if p.len() < 5 || p[0] != 0x2f {
        return Err(bad("bad VP8L signature"));
    }
    let h = u32::from_le_bytes([p[1], p[2], p[3], p[4]]);
    if h >> 29 != 0 {
        return Err(bad("unknown VP8L version"));
    }
    Ok(((h & 0x3fff) as usize + 1, ((h >> 14) & 0x3fff) as usize + 1))
}

fn check_size(w: usize, h: usize) -> Result<(), VisionError> {
    if w == 0 || h == 0 {
        return Err(bad("zero width or height"));
    }
    if w > MAX_SIDE || h > MAX_SIDE || w * h > MAX_DECODED_PIXELS {
        return Err(VisionError::Unsupported(format!("{w}x{h} exceeds the decoded size limit")));
    }
    Ok(())
}

fn animated() -> VisionError {
    VisionError::Unsupported("animated images are not supported".into())
}

/// Walk the RIFF container of a still WebP and read its size, kind and orientation, refusing what the decoder must not see.
pub(crate) fn header(bytes: &[u8]) -> Result<Header, VisionError> {
    if !is_webp(bytes) {
        return Err(VisionError::Unsupported(crate::decode::UNSUPPORTED_FORMAT.into()));
    }
    let riff = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if riff < 4 + 8 {
        return Err(bad("RIFF size too small"));
    }
    let riff_len = riff
        .checked_add(8)
        .filter(|&n| n <= bytes.len())
        .ok_or_else(|| bad("truncated file (the RIFF size runs past the data)"))?;
    let data = &bytes[..riff_len];
    // the chunks: (fourcc, payload range)
    let mut chunks: Vec<([u8; 4], Range<usize>)> = Vec::new();
    let mut at = 12;
    while at < data.len() {
        if data.len() - at < 8 {
            return Err(bad("truncated chunk header"));
        }
        let fourcc = [data[at], data[at + 1], data[at + 2], data[at + 3]];
        let size = u32::from_le_bytes([data[at + 4], data[at + 5], data[at + 6], data[at + 7]]) as usize;
        let start = at + 8;
        if size > data.len() - start {
            return Err(bad(format!("chunk {} runs past the RIFF data", String::from_utf8_lossy(&fourcc))));
        }
        chunks.push((fourcc, start..start + size));
        if chunks.len() > 4096 {
            return Err(bad("too many chunks"));
        }
        // chunks are padded to an even size; a missing pad byte at the very end is tolerated (libwebp does)
        at = start + size + (size & 1);
    }
    let Some((first, payload)) = chunks.first().map(|(c, r)| (*c, &data[r.clone()])) else { return Err(bad("no image chunk")) };
    let find = |want: &[u8; 4]| chunks.iter().find(|(c, _)| c == want).map(|(_, r)| r.clone());
    let (w, h, lossless, alpha, orientation, lossy_alpha) = match &first {
        b"VP8 " => {
            let (w, h) = vp8_size(payload)?;
            (w, h, false, false, 1, None)
        }
        b"VP8L" => {
            let (w, h) = vp8l_size(payload)?;
            // the alpha hint bit: what libwebp's features (and so Pillow's mode, RGBA or RGB) report
            (w, h, true, payload[4] & 0x10 != 0, 1, None)
        }
        b"VP8X" => {
            if payload.len() < 10 {
                return Err(bad("VP8X chunk too short"));
            }
            let flags = payload[0];
            let (cw, ch) = (le24(payload, 4) as usize + 1, le24(payload, 7) as usize + 1);
            if flags & 0x02 != 0 || chunks.iter().any(|(c, _)| c == b"ANIM" || c == b"ANMF") {
                return Err(animated());
            }
            // the canvas is checked before the bitstream is looked at: over the limit is a refusal whatever follows
            check_size(cw, ch)?;
            let image = chunks.iter().skip(1).find(|(c, _)| c == b"VP8 " || c == b"VP8L");
            let Some((kind, image)) = image.cloned() else { return Err(bad("no VP8 or VP8L chunk")) };
            let p = &data[image.clone()];
            let lossless = &kind == b"VP8L";
            let (w, h) = if lossless { vp8l_size(p)? } else { vp8_size(p)? };
            if (w, h) != (cw, ch) {
                return Err(bad(format!("bitstream {w}x{h} differs from the canvas {cw}x{ch}")));
            }
            if chunks.iter().filter(|(c, _)| c == b"VP8 " || c == b"VP8L").count() > 1 {
                return Err(bad("more than one image chunk in a still image"));
            }
            // Pillow reads the EXIF chunk with or without the JPEG-style `Exif\0\0` prefix
            let orientation = find(b"EXIF")
                .map(|r| {
                    let e = &data[r];
                    tiff_orientation(e.strip_prefix(b"Exif\0\0").unwrap_or(e))
                })
                .unwrap_or(1);
            // libwebp (and so Pillow) gives an RGBA image only when the VP8X alpha flag is set; an ALPH chunk without it is not used
            let alpha = flags & 0x10 != 0;
            let lossy_alpha = (!lossless && alpha).then(|| (image, find(b"ALPH")));
            (w, h, lossless, alpha, orientation, lossy_alpha)
        }
        other => return Err(bad(format!("unexpected first chunk {}", String::from_utf8_lossy(other)))),
    };
    check_size(w, h)?;
    Ok(Header { width: w, height: h, lossless, alpha, orientation, riff_len, lossy_alpha })
}

/// Decode a still WebP to RGB: alpha (when the file declares it) dropped or composited onto white per `policy`, EXIF orientation applied.
pub fn decode(bytes: &[u8], policy: AlphaPolicy) -> Result<Rgb, VisionError> {
    let hd = header(bytes)?;
    let n = hd.width * hd.height;
    let rgb = match &hd.lossy_alpha {
        // Lossy with alpha, in two steps so that no more than one full-size buffer besides the alpha plane exists at a time (image-webp's
        // own path holds the Y/U/V planes, the RGBA output, the decoded alpha stream as ARGB and its alpha plane together: 10.5 bytes a
        // pixel): the alpha plane (1 byte a pixel), then the colour frame alone as RGB, then the composite in place. The colour of a
        // lossy+alpha file does not depend on its alpha (libwebp writes un-premultiplied RGBA), so the result is the same bytes.
        Some((vp8, alph)) => {
            let alpha = match alph {
                Some(a) => Some(alpha_plane(&bytes[a.clone()], hd.width, hd.height)?),
                None => None, // the flag without the chunk: opaque
            };
            let mut riff = Vec::with_capacity(20 + vp8.len());
            riff.extend_from_slice(b"RIFF");
            riff.extend_from_slice(&((12 + vp8.len() + (vp8.len() & 1)) as u32).to_le_bytes());
            riff.extend_from_slice(b"WEBPVP8 ");
            riff.extend_from_slice(&(vp8.len() as u32).to_le_bytes());
            riff.extend_from_slice(&bytes[vp8.clone()]);
            if vp8.len() & 1 == 1 {
                riff.push(0);
            }
            let mut buf = pixels(&riff, hd.width, hd.height, false)?;
            drop(riff);
            if let (AlphaPolicy::CompositeWhite, Some(alpha)) = (policy, alpha) {
                for (p, &a) in buf.chunks_exact_mut(3).zip(&alpha) {
                    for x in p {
                        *x = over_white(*x, a);
                    }
                }
            }
            buf
        }
        // Lossless without alpha: image-webp decodes the ARGB stream into a buffer of its own and copies RGB out (4 + 3 bytes a pixel).
        // The alpha hint bit does not change the stream's decode, so a copy of the file with the bit set decodes straight into one RGBA
        // buffer (4 bytes a pixel), and the alpha it carries is ignored, as Pillow's RGB mode ignores it.
        None if hd.lossless && !hd.alpha => {
            let mut file = bytes[..hd.riff_len].to_vec();
            if &file[12..16] == b"VP8L" {
                file[24] |= 0x10;
            } else {
                file[20] |= 0x10;
            }
            let mut buf = pixels(&file, hd.width, hd.height, true)?;
            drop(file);
            rgba_to_rgb(&mut buf, n, |r, g, b, _| [r, g, b]);
            buf
        }
        None => {
            let mut buf = pixels(&bytes[..hd.riff_len], hd.width, hd.height, hd.alpha)?;
            if hd.alpha {
                rgba_to_rgb(&mut buf, n, |r, g, b, a| match policy {
                    AlphaPolicy::Drop => [r, g, b],
                    AlphaPolicy::CompositeWhite => [over_white(r, a), over_white(g, a), over_white(b, a)],
                });
            }
            buf
        }
    };
    Ok(orient(Rgb { width: hd.width, height: hd.height, data: rgb }, hd.orientation))
}

/// image-webp's decode of a (validated) file of `w x h`: RGBA when it reports alpha, else RGB. `want_alpha` must be what it reports.
fn pixels(file: &[u8], w: usize, h: usize, want_alpha: bool) -> Result<Vec<u8>, VisionError> {
    let mut dec = image_webp::WebPDecoder::new(std::io::Cursor::new(file)).map_err(|e| bad(e.to_string()))?;
    let (dw, dh) = dec.dimensions();
    if (dw as usize, dh as usize) != (w, h) || dec.is_animated() || dec.has_alpha() != want_alpha {
        return Err(bad("the container and the decoder disagree on the image"));
    }
    let size =
        dec.output_buffer_size().filter(|&s| s == w * h * if want_alpha { 4 } else { 3 }).ok_or_else(|| bad("size overflow"))?;
    let mut buf = vec![0u8; size];
    dec.read_image(&mut buf).map_err(|e| bad(e.to_string()))?;
    Ok(buf)
}

/// RGBA -> RGB in place through `f`: pixel i is read before bytes 3i..3i+3 (all at or below 4i+2) are written; then the tail is freed.
fn rgba_to_rgb(buf: &mut Vec<u8>, n: usize, f: impl Fn(u8, u8, u8, u8) -> [u8; 3]) {
    for i in 0..n {
        let px = f(buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]);
        buf[3 * i..3 * i + 3].copy_from_slice(&px);
    }
    buf.truncate(3 * n);
    buf.shrink_to_fit();
}

/// The alpha plane of an `ALPH` chunk (WebP container spec; libwebp `alpha_dec.c`): one header byte (reserved 2 bits, which must be 0;
/// pre-processing 2 bits, 0 or 1, informative only: libwebp dithers only when asked to, and Pillow does not ask; filter 2 bits;
/// compression 2 bits, 0 raw or 1 lossless), then the data. A lossless alpha stream is a VP8L image stream without its 5-byte header
/// whose green channel is the alpha: it is decoded here as a VP8L file with that header put back (40 bits, so the stream stays byte
/// aligned). Then the filter is undone as libwebp's `HorizontalUnfilter_C` / `VerticalUnfilter_C` / `GradientUnfilter_C` do.
fn alpha_plane(chunk: &[u8], w: usize, h: usize) -> Result<Vec<u8>, VisionError> {
    let (&head, data) = chunk.split_first().ok_or_else(|| bad("empty ALPH chunk"))?;
    let (reserved, pre, filter, method) = (head >> 6, (head >> 4) & 3, (head >> 2) & 3, head & 3);
    if reserved != 0 || pre > 1 || method > 1 {
        return Err(bad("unsupported ALPH header"));
    }
    let n = w * h;
    let mut a = if method == 0 {
        data.get(..n).ok_or_else(|| bad("ALPH data shorter than the image"))?.to_vec()
    } else {
        let mut vp8l = Vec::with_capacity(5 + data.len());
        vp8l.push(0x2f);
        vp8l.extend_from_slice(&((w as u32 - 1) | ((h as u32 - 1) << 14) | (1 << 28)).to_le_bytes());
        vp8l.extend_from_slice(data);
        let mut riff = Vec::with_capacity(20 + vp8l.len() + 1);
        riff.extend_from_slice(b"RIFF");
        riff.extend_from_slice(&((12 + vp8l.len() + (vp8l.len() & 1)) as u32).to_le_bytes());
        riff.extend_from_slice(b"WEBPVP8L");
        riff.extend_from_slice(&(vp8l.len() as u32).to_le_bytes());
        riff.extend_from_slice(&vp8l);
        if vp8l.len() & 1 == 1 {
            riff.push(0);
        }
        drop(vp8l);
        let mut argb = pixels(&riff, w, h, true)?;
        drop(riff);
        for i in 0..n {
            argb[i] = argb[4 * i + 1];
        }
        argb.truncate(n);
        argb.shrink_to_fit();
        argb
    };
    unfilter(&mut a, w, h, filter);
    Ok(a)
}

/// Undo an alpha filter in place (0 none, 1 horizontal, 2 vertical, 3 gradient), libwebp's rules: the first row is always predicted
/// from the left (its first pixel from 0); in later rows the first pixel is predicted from the one above.
fn unfilter(a: &mut [u8], w: usize, h: usize, filter: u8) {
    if filter == 0 {
        return;
    }
    for y in 0..h {
        let row = y * w;
        if y == 0 {
            for x in 1..w {
                a[x] = a[x].wrapping_add(a[x - 1]);
            }
            continue;
        }
        let prev = row - w;
        match filter {
            1 => {
                a[row] = a[row].wrapping_add(a[prev]);
                for x in 1..w {
                    a[row + x] = a[row + x].wrapping_add(a[row + x - 1]);
                }
            }
            2 => {
                for x in 0..w {
                    a[row + x] = a[row + x].wrapping_add(a[prev + x]);
                }
            }
            _ => {
                a[row] = a[row].wrapping_add(a[prev]);
                for x in 1..w {
                    let (left, top, top_left) = (a[row + x - 1] as i32, a[prev + x] as i32, a[prev + x - 1] as i32);
                    a[row + x] = a[row + x].wrapping_add((left + top - top_left).clamp(0, 255) as u8);
                }
            }
        }
    }
}

/// Pillow's `MULDIV255` paste onto white (the PNG paths' rule).
fn over_white(x: u8, a: u8) -> u8 {
    let mul = |p: u32, q: u32| -> u32 {
        let t = p * q + 128;
        ((t >> 8) + t) >> 8
    };
    (mul(255, 255 - a as u32) + mul(x as u32, a as u32)).min(255) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn riff(chunks: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut body = b"WEBP".to_vec();
        for (c, p) in chunks {
            body.extend_from_slice(*c);
            body.extend_from_slice(&(p.len() as u32).to_le_bytes());
            body.extend_from_slice(p);
            if p.len() % 2 == 1 {
                body.push(0);
            }
        }
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    fn vp8_header(w: u16, h: u16) -> Vec<u8> {
        let mut p = vec![0x10, 0x00, 0x00, 0x9d, 0x01, 0x2a];
        p.extend_from_slice(&w.to_le_bytes());
        p.extend_from_slice(&h.to_le_bytes());
        p.extend_from_slice(&[0; 32]);
        p
    }

    fn vp8x(flags: u8, w: u32, h: u32) -> Vec<u8> {
        let mut p = vec![flags, 0, 0, 0];
        p.extend_from_slice(&(w - 1).to_le_bytes()[..3]);
        p.extend_from_slice(&(h - 1).to_le_bytes()[..3]);
        p
    }

    #[test]
    fn sizes_come_from_the_bitstream_and_are_checked_before_any_decode() {
        let simple = riff(&[(b"VP8 ", vp8_header(640, 480))]);
        assert_eq!(header(&simple).map(|h| (h.width, h.height, h.lossless)), Ok((640, 480, false)));
        let over = riff(&[(b"VP8 ", vp8_header(16383, 16383))]);
        assert!(matches!(header(&over), Err(VisionError::Unsupported(_))));
        // a 1 x 1 canvas around a 16,383 x 16,383 frame: refused by the walker, not discovered by the decoder after allocating
        let lie = riff(&[(b"VP8X", vp8x(0, 1, 1)), (b"VP8 ", vp8_header(16383, 16383))]);
        assert!(matches!(header(&lie), Err(VisionError::Decode(_))), "{:?}", header(&lie));
        // a canvas of 2^24 x 2^24
        let huge = riff(&[(b"VP8X", vp8x(0, 1 << 24, 1 << 24)), (b"VP8 ", vp8_header(100, 100))]);
        assert!(matches!(header(&huge), Err(VisionError::Unsupported(_))));
    }

    #[test]
    fn animation_and_wrong_containers_are_refused_by_name() {
        let anim = riff(&[(b"VP8X", vp8x(0x02, 8, 8)), (b"ANIM", vec![0; 6]), (b"ANMF", vec![0; 24])]);
        assert_eq!(header(&anim), Err(animated()));
        // an ANMF without the flag is still an animation
        let anmf = riff(&[(b"VP8X", vp8x(0, 8, 8)), (b"ANMF", vec![0; 24]), (b"VP8 ", vp8_header(8, 8))]);
        assert_eq!(header(&anmf), Err(animated()));
        let mut wave = riff(&[(b"fmt ", vec![0; 16])]);
        wave[8..12].copy_from_slice(b"WAVE");
        assert!(matches!(header(&wave), Err(VisionError::Unsupported(_))));
        assert!(!is_webp(&wave));
    }

    #[test]
    fn lying_lengths_are_errors() {
        let mut b = riff(&[(b"VP8 ", vp8_header(8, 8))]);
        b[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(header(&b), Err(VisionError::Decode(_))));
        let mut b = riff(&[(b"VP8 ", vp8_header(8, 8))]);
        b[16..20].copy_from_slice(&0xffff_fff0u32.to_le_bytes());
        assert!(matches!(header(&b), Err(VisionError::Decode(_))));
        for cut in 0..b.len() {
            let _ = header(&b[..cut]);
        }
    }

    #[test]
    fn exif_orientation_is_read_with_or_without_the_prefix() {
        let mut tiff = b"II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0".to_vec();
        tiff.extend_from_slice(&[0, 0, 0, 0]);
        for e in [tiff.clone(), [b"Exif\0\0".to_vec(), tiff.clone()].concat()] {
            let b = riff(&[(b"VP8X", vp8x(0x08, 8, 8)), (b"VP8 ", vp8_header(8, 8)), (b"EXIF", e)]);
            assert_eq!(header(&b).map(|h| h.orientation), Ok(6));
        }
    }
}
