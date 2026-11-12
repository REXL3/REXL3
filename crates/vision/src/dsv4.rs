//! DeepSeek-V4.1-Flash image preprocessing (`ds41ref/image_processor.py`, the reference transcription).
//!
//! An image is padded (not stretched) onto a canvas whose sides are multiples of the 14-pixel patch, normalised to
//! `[-1, 1]`, and patchified channel-major (`x.reshape(3, n_h, p, n_w, p).permute(1, 3, 0, 2, 4).reshape(P, 3, p, p)`,
//! then `flatten(1)`: within a patch the order is channel, then row, then column — [`gb_cuda`-side
//! `dsv4_vision::patchify`]'s order, which the ViT's 588-wide patch GEMM reads). The token grid the language model
//! sees is the 3 x 3 aligner's: `ceil(n_vit_h / 3)` by `ceil(n_vit_w / 3)`, costing
//! [`num_image_tokens`](crate::layout) `= n_llm_h * (n_llm_w + 1) + 2` positions.
//!
//! The resize policy is the reference's `plan_image_grid`: an image under `min_pixels` (295,936 for this pack) is
//! scaled up by its square root ratio; both sides round **up** to a patch multiple (the canvas); and a canvas whose
//! token count would exceed the budget is shrunk by `solve_resize_ratio` — the largest aspect-preserving size whose
//! token grid still fits — never below one token row or column. With `vision_max_wh_ratio` set (this pack's is
//! `None`), an image wider than the ratio is first cropped to it and then stretched, not padded.

use crate::preprocessor::{ImagePlan, ImagePreprocessor};
use crate::{decode, Preprocessed, Rgb, VisionError};

/// Patch side in pixels (`vision_patch_size`).
pub const PATCH: usize = 14;
/// Channels of an RGB image.
pub const CH: usize = 3;
/// Values of one patch: `3 * 14 * 14 = 588`, the ViT patch GEMM's `k`.
pub const PATCH_K: usize = CH * PATCH * PATCH;
/// The aligner's window side (`vision_downsample_ratio`).
pub const MERGE: usize = 3;
/// `vision_min_pixels` of this pack: a smaller image is scaled up to it.
pub const MIN_PIXELS: usize = 295_936;

/// `image_processor.llm_grid`: the token grid a patch grid of this pixel size produces.
pub fn llm_grid(best_height: usize, best_width: usize, patch: usize, ratio: usize) -> (usize, usize) {
    ((best_height / patch).div_ceil(ratio), (best_width / patch).div_ceil(ratio))
}

/// `image_processor.num_image_tokens`: the positions an image of this token grid occupies.
pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

/// `image_processor.solve_resize_ratio`: the largest aspect-preserving pixel size whose token grid fits `max_tokens`.
/// Returns `(best_height, best_width)`.
fn solve_resize_ratio(height: usize, width: usize, patch: usize, ratio: usize, max_tokens: usize) -> (usize, usize) {
    let r = height as f64 / width as f64;
    let max_w_float = ((max_tokens - 2) as f64 / r + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * r;
    let cell = patch * ratio;
    if max_w_float < 1.0 {
        // very tall: collapse to a single column of tokens
        return ((max_tokens - 2) / 2 * cell, cell);
    }
    if max_h_float < 1.0 {
        // very wide: collapse to a single row of tokens
        return (cell, (max_tokens - 3) * cell);
    }
    let beta = (max_w_float.floor() * cell as f64 / width as f64).min(max_h_float.floor() * cell as f64 / height as f64);
    // `floor(v / p) * p`, on both axes
    let side = |v: usize| (v as f64 * beta / patch as f64).floor() as usize * patch;
    (side(height), side(width))
}

/// `image_processor.safe_resize`: shrink the pixel size until the image costs at most `max_tokens` LLM tokens.
/// Returns `(n_llm_h, n_llm_w, best_height, best_width)`.
fn safe_resize(
    height: usize,
    width: usize,
    best_height: usize,
    best_width: usize,
    patch: usize,
    ratio: usize,
    max_tokens: usize,
) -> (usize, usize, usize, usize) {
    let (mut nh, mut nw) = llm_grid(best_height, best_width, patch, ratio);
    let (mut bh, mut bw) = (best_height, best_width);
    if num_image_tokens(nh, nw) > max_tokens {
        let s = solve_resize_ratio(height, width, patch, ratio, max_tokens);
        bh = s.0;
        bw = s.1;
        let g = llm_grid(bh, bw, patch, ratio);
        nh = g.0;
        nw = g.1;
        assert!(num_image_tokens(nh, nw) <= max_tokens, "solve_resize_ratio overshot the token budget");
    }
    (nh, nw, bh, bw)
}

/// `image_processor.plan_image_grid`: the resize plan of a `width x height` image; a pure function of its arguments.
/// Returns `(n_llm_h, n_llm_w, best_height, best_width)`.
pub fn plan_grid(
    width: usize,
    height: usize,
    min_pixels: usize,
    max_tokens: usize,
    max_wh_ratio: Option<f64>,
) -> (usize, usize, usize, usize) {
    let (mut width, mut height) = (width, height);
    if let Some(ratio) = max_wh_ratio {
        if width as f64 > height as f64 * ratio {
            width = (height as f64 * ratio) as usize;
        }
    }
    if 0 < width * height && width * height < min_pixels {
        let s = (min_pixels as f64 / (width * height) as f64).sqrt();
        width = (width as f64 * s) as usize;
        height = (height as f64 * s) as usize;
    }
    let (bw, bh) = (width.div_ceil(PATCH) * PATCH, height.div_ceil(PATCH) * PATCH);
    safe_resize(height, width, bh, bw, PATCH, MERGE, max_tokens)
}

/// The preprocessing of the DeepSeek-V4.1-Flash vision tower: everything [`Dsv4Preprocessing`] is configured with is
/// the pack's `vision_config` plus the served token budget.
#[derive(Clone, Debug, PartialEq)]
pub struct Dsv4Preprocessing {
    /// `vision_min_pixels`: an image with fewer pixels is scaled up (to it, by the square-root ratio).
    pub min_pixels: usize,
    /// The most LLM tokens one image may cost: `min(vision_config.max_image_tokens, the served cap)`.
    pub max_tokens: usize,
    /// `vision_max_wh_ratio` (`None` for this pack): an image wider than the ratio is cropped to it before anything
    /// else, and then stretched to its canvas rather than padded.
    pub max_wh_ratio: Option<f64>,
}

impl Dsv4Preprocessing {
    /// The preprocessing of this pack's `vision_config`, with the served per-image token cap.
    pub fn new(min_pixels: usize, max_tokens: usize, max_wh_ratio: Option<f64>) -> Self {
        Self { min_pixels: min_pixels.max(1), max_tokens: max_tokens.max(num_image_tokens(1, 1)), max_wh_ratio }
    }

    /// The pack's own numbers (the constants this module documents), for the family's default.
    pub fn pack_defaults(max_tokens: usize) -> Self {
        Self::new(MIN_PIXELS, max_tokens, None)
    }

    /// `image_processor.load_image`'s pixel half: the canvas of a decoded image — padded with PIL's neutral gray
    /// `(127, 127, 127)` unless the aspect clamp had already cropped it, in which case stretched.
    fn canvas(&self, img: &Rgb, best_width: usize, best_height: usize) -> Vec<u8> {
        let stretched = self.max_wh_ratio.is_some_and(|ratio| img.width as f64 >= ratio * img.height as f64);
        let (w, h) = (best_width, best_height);
        if img.width == w && img.height == h {
            return img.data.clone();
        }
        if stretched {
            return crate::resize::bicubic_rgb(&img.data, img.width, img.height, w, h);
        }
        // ImageOps.pad: the image keeps its aspect at the canvas's center, surrounded by neutral gray
        let scale = (w as f64 / img.width as f64).min(h as f64 / img.height as f64);
        let (cw, ch) = (
            ((img.width as f64 * scale).floor() as usize).clamp(1, w),
            ((img.height as f64 * scale).floor() as usize).clamp(1, h),
        );
        let content = if cw == img.width && ch == img.height {
            img.data.clone()
        } else {
            crate::resize::bicubic_rgb(&img.data, img.width, img.height, cw, ch)
        };
        let (x0, y0) = ((w - cw) / 2, (h - ch) / 2);
        let mut c = vec![127u8; w * h * CH];
        for y in 0..ch {
            let dst = ((y0 + y) * w + x0) * CH;
            let src = y * cw * CH;
            c[dst..dst + cw * CH].copy_from_slice(&content[src..src + cw * CH]);
        }
        c
    }
}

impl ImagePreprocessor for Dsv4Preprocessing {
    fn family(&self) -> &'static str {
        "deepseek_v4_1_vision"
    }

    fn id(&self) -> String {
        format!(
            "dsv4.1-vision-v1:p{}m{}:px{}:tok{}:ratio{}",
            PATCH,
            MERGE,
            self.min_pixels,
            self.max_tokens,
            match self.max_wh_ratio {
                None => "none".to_string(),
                Some(r) => format!("{r}"),
            }
        )
    }

    fn decode(&self, bytes: &[u8]) -> Result<Rgb, VisionError> {
        decode::decode(bytes)
    }

    fn plan(&self, width: usize, height: usize) -> Result<ImagePlan, VisionError> {
        if width == 0 || height == 0 {
            return Err(VisionError::Unsupported(format!("an image of {width}x{height} has no pixels")));
        }
        let (nh, nw, bh, bw) = plan_grid(width, height, self.min_pixels, self.max_tokens, self.max_wh_ratio);
        Ok(ImagePlan { canvas: (bw, bh), grid: (1, bh / PATCH, bw / PATCH), tokens: num_image_tokens(nh, nw) })
    }

    fn preprocess(&self, img: &Rgb) -> Result<Preprocessed, VisionError> {
        let p = self.plan(img.width, img.height)?;
        let (n_vit_h, n_vit_w) = (p.grid.1, p.grid.2);
        let (w, h) = p.canvas;
        let canvas = self.canvas(img, w, h);
        // permute(2, 0, 1) / 255, then ((x - 0.5) / 0.5), as [3, H, W] planes
        let mut pix = vec![0f32; CH * w * h];
        for c in 0..CH {
            for y in 0..h {
                for x in 0..w {
                    let v = canvas[(y * w + x) * CH + c] as f32 / 255.0;
                    pix[c * w * h + y * w + x] = (v - 0.5) / 0.5;
                }
            }
        }
        // patchify: patch row-major, channel-major within a patch (the ViT's 588-wide order)
        let n = n_vit_h * n_vit_w;
        let mut patches = vec![0u16; n * PATCH_K];
        let mut at = 0usize;
        for ph in 0..n_vit_h {
            for pw in 0..n_vit_w {
                for c in 0..CH {
                    for i in 0..PATCH {
                        for j in 0..PATCH {
                            let v = pix[c * h * w + (ph * PATCH + i) * w + pw * PATCH + j];
                            patches[at + c * PATCH * PATCH + i * PATCH + j] = crate::f32_to_f16_bits(v);
                        }
                    }
                }
                at += PATCH_K;
            }
        }
        // FNV-1a over the canvas's RGB bytes and its size: equal for equal model inputs
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for b in canvas.iter() {
            hash = (hash ^ *b as u64).wrapping_mul(0x0100_0000_01b3);
        }
        for v in [w, h] {
            hash ^= v as u64;
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        Ok(Preprocessed { size: p.canvas, grid: p.grid, patches, patch_dim: PATCH_K, pixel_hash: hash })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pre() -> Dsv4Preprocessing {
        Dsv4Preprocessing::pack_defaults(1000)
    }

    /// The reference's own numbers (`ds41ref/image_processor.py`, `plan_image_grid` transcribed and run against the
    /// same inputs): the canvas, the patch grid, the token grid and the token count must all agree, for an image that
    /// pads, one that scales up, one that shrinks, a square, and a very tall one.
    #[test]
    fn the_resize_plan_is_the_references() {
        // (w, h) -> (canvas_w, canvas_h, n_vit_h, n_vit_w, tokens), from the reference
        let cases = [
            (800, 600, (812, 602, 43, 58, 317), "pads to the patch grid"),
            (100, 80, (616, 490, 35, 44, 194), "a small image scales up to min_pixels"),
            (640, 640, (644, 644, 46, 46, 274), "a square"),
            (4096, 3072, (1456, 1092, 78, 104, 938), "a big image shrinks inside the budget"),
            (1920, 1080, (1708, 966, 69, 122, 968), "a wide one"),
            (200, 1000, (252, 1218, 87, 18, 205), "a tall one"),
            (544, 544, (546, 546, 39, 39, 184), "just under min_pixels (295,936 = 544^2)"),
        ];
        for (w, h, want, why) in cases {
            let p = pre().plan(w, h).unwrap();
            let got = (p.canvas.0, p.canvas.1, p.grid.1, p.grid.2, p.tokens);
            assert_eq!(got, want, "{why}: {w}x{h}");
            assert_eq!(p.canvas.0 % PATCH, 0, "the canvas width is a patch multiple");
            assert_eq!(p.canvas.1 % PATCH, 0, "the canvas height is a patch multiple");
            assert_eq!(p.tokens, num_image_tokens(p.grid.1.div_ceil(MERGE), p.grid.2.div_ceil(MERGE)), "{why}");
        }
    }

    /// Nothing the planner returns may cost more than the budget, for every size that matters, including the
    /// aspect-collapse cases `solve_resize_ratio` has branches for.
    #[test]
    fn no_image_exceeds_the_token_budget() {
        for (w, h) in [(1, 1), (14, 14), (100, 10000), (10000, 100), (295936, 1), (1, 295936), (8192, 8192), (777, 555)] {
            let p = pre().plan(w, h).unwrap();
            assert!(p.tokens <= 1000, "{w}x{h} costs {} tokens", p.tokens);
            assert!(p.grid.1 >= 1 && p.grid.2 >= 1, "{w}x{h} has a grid");
        }
    }

    /// A decoded image patchifies channel-major: patch (0, 0) of a canvas of known values holds the top-left 14 x 14
    /// of every channel, channel-major — the order `PatchEmbed`'s 588-wide GEMM reads. The 644 x 644 image needs no
    /// padding and no resize (644 = 46 patches, 644^2 >= min_pixels), so the canvas is the image.
    #[test]
    fn the_patches_are_channel_major_within_a_patch() {
        let (w, h) = (644, 644); // 46 x 46 patches
        let img = Rgb { width: w, height: h, data: (0..w * h * CH).map(|i| (i % 251) as u8).collect() };
        let p = pre().preprocess(&img).unwrap();
        assert_eq!(p.grid, (1, 46, 46));
        assert_eq!(p.patch_dim, PATCH_K);
        assert_eq!(p.patches.len(), 46 * 46 * PATCH_K);
        assert_eq!(p.size, (w, h));
        let f16 = |i: usize| f16_to_f32(p.patches[i]);
        let want_px = |y: usize, x: usize, c: usize| (img.data[(y * w + x) * CH + c] as f32 / 255.0 - 0.5) / 0.5;
        for c in 0..CH {
            for i in 0..PATCH {
                for j in 0..PATCH {
                    let got = f16(c * PATCH * PATCH + i * PATCH + j);
                    let want = want_px(i, j, c);
                    assert!((got - want).abs() < 1e-3, "({c},{i},{j}): {got} vs {want}");
                }
            }
        }
        // channel-major, then row, then column: (0, 0, 13) is the pixel right of (0, 0, 0), and (1, 0, 0) starts the
        // second channel's plane
        assert!((f16(13) - want_px(0, 13, 0)).abs() < 1e-3);
        assert!((f16(PATCH * PATCH) - want_px(0, 0, 1)).abs() < 1e-3);
        assert_ne!(f16(13), f16(PATCH * PATCH));
        // patch (0, 1) starts at PATCH_K and holds the canvas's columns 14..28
        assert!((f16(PATCH_K) - want_px(0, PATCH, 0)).abs() < 1e-3);
        // patch (1, 0) holds rows 14..28
        assert!((f16(46 * PATCH_K) - want_px(PATCH, 0, 0)).abs() < 1e-3);
    }

    /// The fp16 bits the patchifier writes, back to f32 (the normal range the tests use).
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) as u32) << 31;
        let exp = ((bits >> 10) & 0x1f) as i32;
        let man = (bits & 0x3ff) as u32;
        if exp == 0 {
            // subnormal (the tests' values are not; zero is exact)
            return f32::from_bits(sign);
        }
        f32::from_bits(sign | (((exp - 15 + 127) as u32) << 23) | (man << 13))
    }
}
