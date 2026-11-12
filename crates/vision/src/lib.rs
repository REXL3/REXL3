//! CPU image preprocessing for the Qwen3.8 vision tower (`docs/VISION.md`).
//!
//! The pipeline is ExLlamaV3's `Qwen3VLVisionModel.preprocess` (MIT) reimplemented byte for byte: decode, composite alpha on
//! white, `qwen2_smart_resize`, PIL `Image.resize(BICUBIC)` (see `resize`), `x * (1/255)` then `(x - mean) / std` in float32,
//! and the block-major patch layout with inner order (channel, temporal, row, col), cast to fp16. The fixtures under
//! `testdata/vision` come from PIL and ExLlamaV3 themselves (`tools/gen_vision_fixtures.py`).
#![forbid(unsafe_code)]

pub mod decode;
pub mod dsv4;
pub mod glm;
pub mod layout;
pub mod minicpmv;
pub mod nemotron;
pub mod nemotron_video;
pub mod preprocessor;
pub mod resample;
pub mod resize;
pub mod rgb_decode;
pub mod rope2d;
pub mod tables;
pub mod torch_resample;
pub mod webp;

use std::fmt;

/// Preprocessor settings (`preprocessor_config.json` of the pack).
#[derive(Clone, Debug, PartialEq)]
pub struct Preprocessing {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl Preprocessing {
    /// Qwen3.8-27B: patch 16, temporal 2, merge 2, 65,536 to 16,777,216 pixels, mean = std = 0.5.
    pub fn qwen38() -> Self {
        Self {
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            min_pixels: 65_536,
            max_pixels: 16_777_216,
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }
    /// Pixels per side must be a multiple of this.
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum VisionError {
    /// The bytes are not a supported, well-formed image (message says why).
    Decode(String),
    /// Valid image the model cannot take (too small, absurd aspect ratio, too many pixels).
    Unsupported(String),
}

impl fmt::Display for VisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VisionError::Decode(m) => write!(f, "cannot decode the image: {m}"),
            VisionError::Unsupported(m) => write!(f, "unsupported image: {m}"),
        }
    }
}

impl std::error::Error for VisionError {}

/// A decoded 8-bit RGB image, row-major, 3 bytes a pixel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// Python's `round` is half-to-even; so is this.
fn py_round(x: f64) -> f64 {
    x.round_ties_even()
}

/// ExLlamaV3 `qwen2_smart_resize` for an image of `width` x `height`: the (width, height) to resize to. Both are multiples of
/// `factor`, and the pixel count lies within `[min_pixels, max_pixels]` (rounding up to the minimum, down to the maximum).
pub fn smart_resize(width: usize, height: usize, p: &Preprocessing) -> Result<(usize, usize), VisionError> {
    let factor = p.factor();
    if height < factor || width < factor {
        return Err(VisionError::Unsupported(format!("{width}x{height}: both sides must be at least {factor} pixels")));
    }
    let (hi, lo) = (height.max(width) as f64, height.min(width) as f64);
    if hi / lo > 200.0 {
        return Err(VisionError::Unsupported(format!("aspect ratio {:.1} exceeds 200", hi / lo)));
    }
    let f = factor as f64;
    let mut h_bar = (py_round(height as f64 / f) * f) as usize;
    let mut w_bar = (py_round(width as f64 / f) * f) as usize;
    if h_bar * w_bar > p.max_pixels {
        let beta = ((height * width) as f64 / p.max_pixels as f64).sqrt();
        h_bar = ((height as f64 / beta / f).floor() * f) as usize;
        w_bar = ((width as f64 / beta / f).floor() * f) as usize;
    } else if h_bar * w_bar < p.min_pixels {
        let beta = (p.min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ((height as f64 * beta / f).ceil() * f) as usize;
        w_bar = ((width as f64 * beta / f).ceil() * f) as usize;
    }
    Ok((w_bar, h_bar))
}

/// The preprocessed image: fp16 patches `[patches, 3 * temporal * patch * patch]` (block-major over 2x2 merge blocks, inner order
/// (channel, temporal, row, col), the single frame repeated over the temporal axis), the patch grid and a content hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Preprocessed {
    /// Resized pixel size (width, height).
    pub size: (usize, usize),
    /// Patch grid (t, h, w); t is 1 for an image.
    pub grid: (usize, usize, usize),
    /// fp16 bit patterns, row-major `[grid.0 * grid.1 * grid.2, patch_dim]`.
    pub patches: Vec<u16>,
    pub patch_dim: usize,
    /// FNV-1a over the resized RGB bytes and the size: equal for equal model inputs.
    pub pixel_hash: u64,
}

impl Preprocessed {
    pub fn n_patches(&self) -> usize {
        self.grid.0 * self.grid.1 * self.grid.2
    }
    /// LM tokens the image occupies: one per merged 2x2 block of patches.
    pub fn n_tokens(&self, p: &Preprocessing) -> usize {
        self.n_patches() / (p.merge_size * p.merge_size)
    }
}

/// float32 -> float16 bits, round to nearest even (what `torch.half()` / numpy `astype(float16)` do).
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x0200 | (man >> 13) as u16 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut h = (m >> shift) as u16;
        let rem = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = sign | ((e as u16) << 10) | (man >> 13) as u16;
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

fn fnv(h: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(FNV_PRIME))
}

/// `FNV_PRIME^n` modulo 2^64: what FNV-1a does to its state over `n` zero bytes (`(h ^ 0) * prime` each), so a run of black padding costs a
/// logarithmic number of multiplications instead of a byte-at-a-time chain.
fn prime_pow(mut n: usize) -> u64 {
    let (mut base, mut acc) = (FNV_PRIME, 1u64);
    while n > 0 {
        if n & 1 == 1 {
            acc = acc.wrapping_mul(base);
        }
        base = base.wrapping_mul(base);
        n >>= 1;
    }
    acc
}

/// `Preprocessed::pixel_hash` of both preprocessors: FNV-1a over the size (width, then height, each as 8 little-endian bytes) and the RGB bytes of
/// the image the tower sees. A serial chain, about 2 ns a byte on a GB10 core, so it runs beside the layout (`layout::beside`).
pub fn pixel_hash(width: usize, height: usize, rgb: &[u8]) -> u64 {
    fnv(fnv_header(width, height), rgb)
}

fn fnv_header(width: usize, height: usize) -> u64 {
    fnv(fnv(FNV_OFFSET, &(width as u64).to_le_bytes()), &(height as u64).to_le_bytes())
}

/// [`pixel_hash`] of a `tw x th` canvas that holds `content` (`cw x ch` RGB bytes) at its top left and black everywhere else, without building
/// the canvas: the rows of the content, each followed by its padding (`tw - cw` pixels of zeros), then the padding rows. The same value as
/// `pixel_hash(tw, th, canvas)` (zero bytes advance the FNV-1a state by a multiplication, see [`prime_pow`]; the tests compare the two), so the hash
/// can start as soon as the resize is done, while the canvas is still being padded.
pub fn canvas_hash(tw: usize, th: usize, content: &[u8], cw: usize, ch: usize) -> u64 {
    assert!(
        cw <= tw && ch <= th && content.len() == cw * ch * 3,
        "{cw}x{ch} content ({} bytes) in a {tw}x{th} canvas",
        content.len()
    );
    let mut h = fnv_header(tw, th);
    if cw == tw {
        h = fnv(h, content);
    } else if cw == 0 {
        // nothing of the content: the whole canvas is black
        return h.wrapping_mul(prime_pow(tw * th * 3));
    } else {
        let right = prime_pow((tw - cw) * 3);
        for row in content.chunks_exact(cw * 3) {
            h = fnv(h, row).wrapping_mul(right);
        }
    }
    // the padding rows below
    h.wrapping_mul(prime_pow(tw * (th - ch) * 3))
}

impl Preprocessing {
    /// The tensor element of one sample: `x * float32(1/255)`, then `(x - mean) / std` in float32 (what numpy does), cast to fp16 bits. The scalar
    /// function [`Preprocessing::lut_f16`] tabulates; a tensor element is this and nothing else.
    fn normalise(&self, v: u8, c: usize) -> u16 {
        let rescale = (1.0f64 / 255.0) as f32;
        f32_to_f16_bits(((v as f32) * rescale - self.mean[c]) / self.std[c])
    }

    /// [`Preprocessing::normalise`] of every (channel, byte value): the 3 x 256 table the layout looks samples up in.
    pub fn lut_f16(&self) -> [[u16; 256]; 3] {
        std::array::from_fn(|c| std::array::from_fn(|v| self.normalise(v as u8, c)))
    }

    /// The layout of this preprocessing (`layout::Shape`).
    pub fn shape(&self) -> layout::Shape {
        layout::Shape { patch: self.patch_size, merge: self.merge_size, temporal: self.temporal_patch_size }
    }
}

/// Resize (when the size changes), normalize and patchify an RGB image. Mirrors ExLlamaV3's `preprocess` for one image. The resize, the layout and
/// the hash run on up to `resize::default_workers()` threads (`REXL3_VISION_THREADS`), fewer for small images; the result does not depend on it.
pub fn preprocess(img: &Rgb, p: &Preprocessing) -> Result<Preprocessed, VisionError> {
    preprocess_with(img, p, 0)
}

/// [`preprocess`] on at most `max_workers` threads (0: `resize::default_workers`, 1: everything inline). The bytes are the same for every value.
pub fn preprocess_with(img: &Rgb, p: &Preprocessing, max_workers: usize) -> Result<Preprocessed, VisionError> {
    let (nw, nh) = smart_resize(img.width, img.height, p)?;
    let resized;
    let rgb: &Rgb = if (nw, nh) != (img.width, img.height) {
        resized =
            Rgb { width: nw, height: nh, data: resize::bicubic_rgb_with(&img.data, img.width, img.height, nw, nh, max_workers) };
        &resized
    } else {
        img
    };
    let cap = if max_workers == 0 { resize::default_workers() } else { max_workers };
    let pixels = nw * nh;
    let lut = p.lut_f16();
    let plan = layout::plan(cap, pixels, true);
    let (patches, pixel_hash) = layout::beside(
        plan.hash_aside,
        || pixel_hash(nw, nh, &rgb.data),
        || layout::patchify(&rgb.data, nw, nh, &lut, p.shape(), plan.workers),
    );
    let (gh, gw) = (nh / p.patch_size, nw / p.patch_size);
    Ok(Preprocessed { size: (nw, nh), grid: (1, gh, gw), patches, patch_dim: p.shape().patch_dim(), pixel_hash })
}

/// Today's Qwen preprocessing, copied verbatim (the loops, their indexing, the per-sample float32 normalisation and the serial FNV-1a) from
/// the commit that preceded the table-driven, threaded layout: the oracle the new code is compared with byte for byte. It shares only
/// `smart_resize`, `resize::bicubic_rgb` and `f32_to_f16_bits` with the new code, which have their own tests against PIL and the fixtures.
#[cfg(test)]
pub(crate) mod reference {
    use super::*;

    fn fnv(h: u64, bytes: &[u8]) -> u64 {
        bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
    }

    /// The hash lines of `preprocess` below, as a function of a canvas.
    pub(crate) fn pixel_hash(width: usize, height: usize, canvas: &[u8]) -> u64 {
        let mut h = fnv(0xcbf2_9ce4_8422_2325, &(width as u64).to_le_bytes());
        h = fnv(h, &(height as u64).to_le_bytes());
        fnv(h, canvas)
    }

    pub(crate) fn preprocess(img: &Rgb, p: &Preprocessing) -> Result<Preprocessed, VisionError> {
        let (nw, nh) = smart_resize(img.width, img.height, p)?;
        let resized;
        let rgb: &Rgb = if (nw, nh) != (img.width, img.height) {
            resized = Rgb { width: nw, height: nh, data: resize::bicubic_rgb(&img.data, img.width, img.height, nw, nh) };
            &resized
        } else {
            img
        };
        let ps = p.patch_size;
        let (gh, gw) = (nh / ps, nw / ps);
        let m = p.merge_size;
        let t = p.temporal_patch_size;
        let patch_dim = 3 * t * ps * ps;
        let mut patches = vec![0u16; gh * gw * patch_dim];
        // float32 normalization exactly as numpy does it: x * float32(1/255), then (x - mean) / std
        let rescale = (1.0f64 / 255.0) as f32;
        let norm = |v: u8, c: usize| -> u16 { f32_to_f16_bits(((v as f32) * rescale - p.mean[c]) / p.std[c]) };
        // block-major patch order: (block row, block col, row in block, col in block); inner (channel, temporal, row, col)
        let mut idx = 0usize;
        for br in 0..gh / m {
            for bc in 0..gw / m {
                for mr in 0..m {
                    for mc in 0..m {
                        let (py, px) = ((br * m + mr) * ps, (bc * m + mc) * ps);
                        for c in 0..3 {
                            for _ti in 0..t {
                                for r in 0..ps {
                                    let row = ((py + r) * rgb.width + px) * 3 + c;
                                    for k in 0..ps {
                                        patches[idx] = norm(rgb.data[row + k * 3], c);
                                        idx += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        debug_assert_eq!(idx, patches.len());
        let mut h = fnv(0xcbf2_9ce4_8422_2325, &(nw as u64).to_le_bytes());
        h = fnv(h, &(nh as u64).to_le_bytes());
        h = fnv(h, &rgb.data);
        Ok(Preprocessed { size: (nw, nh), grid: (1, gh, gw), patches, patch_dim, pixel_hash: h })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversion_matches_known_values() {
        for (v, bits) in [
            (0.0f32, 0x0000u16),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-1.0, 0xbc00),
            (0.5, 0x3800),
            (65504.0, 0x7bff),
            (1e-8, 0x0000),
            (6.1035156e-5, 0x0400),
            (5.9604645e-8, 0x0001),
            (f32::INFINITY, 0x7c00),
            (70000.0, 0x7c00),
        ] {
            assert_eq!(f32_to_f16_bits(v), bits, "{v}");
        }
        // round to nearest even at a tie: 2049 is halfway between 2048 and 2050 (spacing 2) -> even mantissa 2048
        assert_eq!(f32_to_f16_bits(2049.0), f32_to_f16_bits(2048.0));
        assert_eq!(f32_to_f16_bits(2051.0), f32_to_f16_bits(2052.0));
    }

    #[test]
    fn smart_resize_rules() {
        let p = Preprocessing::qwen38();
        assert_eq!(smart_resize(64, 64, &p).unwrap(), (256, 256), "below the minimum: ceil up to 65,536 pixels");
        assert_eq!(smart_resize(512, 512, &p).unwrap(), (512, 512));
        assert_eq!(smart_resize(2000, 2160, &p).unwrap(), (1984, 2176), "2160/32 = 67.5 rounds to the even 68");
        assert_eq!(smart_resize(6000, 4000, &p).unwrap(), (4992, 3328), "above the maximum: floor down");
        assert!(smart_resize(31, 400, &p).is_err());
        assert!(smart_resize(32, 6400, &p).is_ok());
        assert!(smart_resize(32, 6500, &p).is_err(), "aspect ratio above 200");
    }

    #[test]
    fn the_hash_of_a_padded_canvas_needs_no_canvas() {
        use resize::reference::{noise, Rng};
        let mut r = Rng::new(17);
        let mut cases: Vec<(usize, usize, usize, usize)> =
            vec![(0, 0, 0, 0), (0, 5, 0, 3), (4, 0, 0, 0), (4, 3, 0, 3), (4, 3, 4, 0), (1, 1, 1, 1), (2, 1, 1, 1)];
        // (tw, th, cw, ch): every mix of equal and smaller content sides, then random ones
        for (tw, cw) in [(1usize, 1usize), (2, 1), (5, 5), (5, 4), (28, 27), (28, 1), (64, 33)] {
            for (th, ch) in [(1usize, 1usize), (2, 1), (5, 5), (5, 4), (28, 27), (28, 1), (64, 33)] {
                cases.push((tw, th, cw, ch));
            }
        }
        for _ in 0..260 {
            let (cw, ch) = (1 + r.below(100), 1 + r.below(100));
            cases.push((cw + r.below(50), ch + r.below(50), cw, ch));
        }
        for (tw, th, cw, ch) in &cases {
            let (tw, th, cw, ch) = (*tw, *th, *cw, *ch);
            let content = noise(cw * ch * 3, (tw * 7 + th * 3 + cw + ch) as u64);
            // the canvas as the old code built it: zeros with the content rows copied in
            let mut canvas = vec![0u8; tw * th * 3];
            for y in 0..ch {
                canvas[y * tw * 3..y * tw * 3 + cw * 3].copy_from_slice(&content[y * cw * 3..(y + 1) * cw * 3]);
            }
            assert_eq!(canvas_hash(tw, th, &content, cw, ch), reference::pixel_hash(tw, th, &canvas), "{cw}x{ch} in {tw}x{th}");
            assert_eq!(pixel_hash(tw, th, &canvas), reference::pixel_hash(tw, th, &canvas), "{tw}x{th}");
        }
        assert!(cases.len() >= 300, "{} cases", cases.len());
        // FNV-1a of n zero bytes is a multiplication by the prime to the n
        for n in [0usize, 1, 2, 3, 7, 8, 100, 4097, 1 << 20] {
            let (with_zeros, without) = (reference::pixel_hash(0, 0, &vec![0u8; n]), reference::pixel_hash(0, 0, &[]));
            assert_eq!(with_zeros, without.wrapping_mul(prime_pow(n)), "{n} zero bytes");
        }
    }

    #[test]
    fn the_entry_equals_the_verbatim_reference_on_a_few_images() {
        let p = Preprocessing::qwen38();
        for (w, h) in [(64usize, 64usize), (256, 256), (333, 517), (100, 300), (640, 64), (96, 96)] {
            let img = Rgb { width: w, height: h, data: resize::reference::noise(w * h * 3, (w * 31 + h) as u64) };
            assert_eq!(preprocess(&img, &p), reference::preprocess(&img, &p), "{w}x{h}");
        }
    }
}
