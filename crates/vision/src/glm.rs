//! CPU image preprocessing for the GLM-5.3-Flash vision tower (`Glm5NextForConditionalGeneration`).
//!
//! Two references define the model's input, and they are not the same function, so there are two [`Profile`]s:
//!
//! | step | `Profile::Fork` (ExLlamaV3 `Glm5NextVisionModel.preprocess`) | `Profile::Hf` (`Glm5NextProcessor`, transformers 5.17, torchvision backend) |
//! | --- | --- | --- |
//! | RGB conversion | alpha composited on white | `image.convert("RGB")`: alpha dropped |
//! | canvas, content scale, zero pad, patch layout | identical | identical |
//! | resize | PIL `BICUBIC`, 22-bit coefficients (`resample::bicubic_rgb_pil`: `resize::bicubic_rgb` plus Pillow 12.2's two-step for very tall images) | torch's uint8 antialiased bicubic, `i16` coefficients at a per-axis precision (`torch_resample::bicubic_rgb_torch_with`), what `tvF.resize` runs |
//! | rescale + normalise | `x * (1/255)` then `(x - mean) / std` in float32 (numpy) | `(x - mean*255) / (std*255)` in float32 (HF's fused path) |
//! | tensor | fp16 `[P, 1176]` (`.half()`) | float32 `pixel_values` `[P, 1176]` (and its fp16 cast) |
//!
//! **The default is `Fork`** ([`Profile::default`], [`GlmPreprocessing::default`]), and it is byte-exact against the fork (canvas bytes and fp16
//! tensor), resized or not: the fork is the oracle the vision tower was validated against.
//!
//! **`Hf` is byte-exact against the HF processor**, resized or not: the canvas bytes, the float32 `pixel_values` and their fp16 cast, for the 59 fixture
//! images and the 253 images of the pixel sweep (`tests/glm_fixtures.rs`). Where a resize happens (raw pixels under the 16-token minimum, or a canvas
//! over the 8000-token budget) it is the port of the kernel the processor runs (`torch_resample`, BSD-3-Clause, see `THIRD_PARTY_NOTICES.md`), not
//! Pillow's, which differs from it by one or two levels in a few percent of the resized samples (`tools/gen_glm_vision_fixtures.py` records the
//! deviation of every case as `pil_vs_hf`). The same processor has a second, Pillow-based backend (`Glm5NextImageProcessorPil`, what a CPU-only HF
//! install runs); the fp16 tensor of `Hf` equals that backend's exactly where HF's two backends agree with each other (always when no resize happens).
//!
//! The two profiles also differ in alpha (any image with transparency), in the resize kernel, and in 3 of the 768 (channel, byte) values of the
//! normalisation in fp16 (about 0.4% of the elements of a noise image, one fp16 ulp each). `testdata/vision/glm` holds both references for 59 images.
//!
//! The pipeline (all of it is `Glm5NextImageProcessor.resize` / `patchify` and the fork's `preprocess`, which agree):
//!
//! 1. **Canvas.** `smart_resize`: an aligned canvas (`factor = patch * merge = 28`) inside a token budget, counted in tokens of
//!    `temporal * factor^2 = 1568` pixels: [`min_image_tokens`, `max_image_tokens`] = [16, 8000] from `processor_config.json`. A still image is
//!    processed as `temporal = 2` frames. Align up; if the aligned budget is under the minimum scale up by `sqrt(min/budget)` and realign; if over the
//!    maximum binary-search the largest content height whose aligned canvas fits (width follows the aspect ratio).
//! 2. **Content.** The image is scaled by `min(canvas_h/h, canvas_w/w)`; once its raw pixels reach the minimum budget the scale is clamped to 1 (an image
//!    inside the budget is never resized, only padded); `content = floor(size * scale)` clipped to the canvas. Resized (bicubic) only when the content
//!    size differs from the image size.
//! 3. **Pad** the right and bottom with black (uint8 zero, *before* normalisation) up to the canvas.
//! 4. **Normalise**, then **patchify**: `[gh/2, gw/2, 2, 2, C, T, 14, 14]` block-major over 2x2 merge blocks, inner order (channel, temporal, row,
//!    col), a still image repeated over the two temporal slots: `[gh * gw, 3 * 2 * 14 * 14 = 1176]`. Grid `(1, gh, gw)`; the LM sees `gh * gw / 4`
//!    placeholder tokens.
//!
//! The two temporal halves of a still image are identical, so a tower can fold the temporal convolution (`W[:, :, 0] + W[:, :, 1]`) and take
//! [`still_patches`] (588 wide, half the memory and half the GEMM) instead.
//!
//! Video (not implemented): `Glm5NextVideoProcessor` samples frames at 2 fps from the video metadata (it needs fps and duration; `sample_frames`),
//! pads an odd count by repeating the last frame, runs the same canvas rule on the whole stack (`num_frames` = the frame count, budget up to 240,000
//! tokens), and patchifies in pairs of frames (`grid_t = frames / 2`). The prompt gets one `<|begin_of_image|>..<|end_of_image|>` run per frame pair
//! followed by its timestamp text. Frame decoding and sampling would have to be done before this module (no video decoder here).

use std::borrow::Cow;
use std::fmt;

use serde_json::Value;

use crate::layout::{self, Shape};
use crate::preprocessor::{ImagePlan, ImagePreprocessor};
use crate::rgb_decode::{decode_rgb, AlphaPolicy};
use crate::{f32_to_f16_bits, resample, torch_resample, Preprocessed, Rgb, VisionError};

/// Which reference the preprocessing follows (see the module documentation for what is exact).
///
/// `Fork` (the default): the ExLlamaV3 fork's `Glm5NextVisionModel.preprocess`, alpha composited on white, PIL resize, numpy normalisation, fp16.
/// `Hf`: the HF processor, alpha dropped (`image.convert("RGB")`), torch's uint8 resize kernel (`torch_resample`) and its fused float32 normalisation:
/// exact against HF for every image, resized or not.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Profile {
    Hf,
    #[default]
    Fork,
}

impl Profile {
    pub fn alpha_policy(self) -> AlphaPolicy {
        match self {
            Profile::Hf => AlphaPolicy::Drop,
            Profile::Fork => AlphaPolicy::CompositeWhite,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Profile::Hf => "hf",
            Profile::Fork => "fork",
        }
    }
}

/// A `processor_config.json` the preprocessing cannot honour.
#[derive(Debug, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "processor_config.json: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

/// GLM-5.3-Flash image preprocessing parameters (`processor_config.json` `image_processor`, plus the serving limits).
#[derive(Clone, Debug, PartialEq)]
pub struct GlmPreprocessing {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub profile: Profile,
    /// Refuse images with a side beyond this many pixels (a bound on the arithmetic, not a model property).
    pub max_side: usize,
    /// Refuse images with more pixels than this before any pixel work (a decompression-bomb guard).
    pub max_pixels: usize,
    /// Worker threads for the resize, the patchify and the hash of a large image: 0 picks `resize::default_workers()` (the environment variable
    /// `REXL3_VISION_THREADS` when it is a positive integer, else `min(available cores, 8)`), a positive value is the most it may use (it wins over
    /// the variable), 1 runs inline. Rows are independent in all of them, so the output is identical for every value (the tests compare them).
    /// Small work gets fewer: one patchify worker per 65,536 pixels (`layout::layout_workers`), one resize worker per `resize::MIN_WORK_PER_WORKER`
    /// multiply-accumulates (`resize::resize_workers`, for the Pillow kernel of `Fork` and the torch kernel of `Hf` alike), and the hash runs beside
    /// the patchify only from 65,536 pixels (`layout::hash_on_a_thread`), on a thread of its own that counts against `threads` (the patchify then has
    /// one worker fewer, so `threads` is the most threads busy at once).
    pub threads: usize,
}

/// Everything the geometry of one image depends on, computed from its size alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// Canvas (width, height) in pixels, a multiple of `patch * merge`.
    pub canvas: (usize, usize),
    /// Content (width, height) after the aspect-preserving scale, inside the canvas; the rest is black padding.
    pub content: (usize, usize),
    /// Patch grid (t, h, w); t is 1 for an image.
    pub grid: (usize, usize, usize),
    /// `<|image|>` placeholder tokens: one per merged block of patches.
    pub tokens: usize,
}

/// GLM-5.3-Flash with the default profile ([`Profile::Fork`]).
impl Default for GlmPreprocessing {
    fn default() -> Self {
        Self::glm53_flash(Profile::default())
    }
}

/// The reference's `align`: `ceil(value / factor) * factor`.
fn align(value: usize, factor: usize) -> usize {
    value.div_ceil(factor) * factor
}

impl GlmPreprocessing {
    /// GLM-5.3-Flash: patch 14, merge 2, temporal patch 2, 16 to 8000 image tokens, CLIP mean and std (`processor_config.json` of the pack), with
    /// the given profile ([`GlmPreprocessing::default`] picks [`Profile::Fork`]).
    pub fn glm53_flash(profile: Profile) -> Self {
        Self {
            patch_size: 14,
            merge_size: 2,
            temporal_patch_size: 2,
            min_image_tokens: 16,
            max_image_tokens: 8000,
            mean: [0.48145466f64 as f32, 0.4578275f64 as f32, 0.40821073f64 as f32],
            std: [0.26862954f64 as f32, 0.26130258f64 as f32, 0.27577711f64 as f32],
            profile,
            max_side: 1 << 16,
            max_pixels: crate::decode::MAX_DECODED_PIXELS,
            threads: 0,
        }
    }

    /// The most threads the patchify and the hash may use: `threads`, or the default when that is 0.
    fn cap(&self) -> usize {
        if self.threads == 0 {
            crate::resize::default_workers()
        } else {
            self.threads
        }
    }

    /// The layout of this preprocessing (`layout::Shape`).
    pub fn shape(&self) -> Shape {
        Shape { patch: self.patch_size, merge: self.merge_size, temporal: self.temporal_patch_size }
    }

    /// Read the `image_processor` block of a `processor_config.json`. A missing key takes the `Glm5NextImageProcessor` default (as
    /// transformers does); a wrong type, a zero, a non-unit `patch_expand_factor` or `do_rescale: false` is an error.
    pub fn from_processor_config(cfg: &Value, profile: Profile) -> Result<Self, ConfigError> {
        let mut p = Self::glm53_flash(profile);
        let Some(ip) = cfg.get("image_processor") else { return Err(ConfigError("no `image_processor` block".into())) };
        if let Some(t) = ip.get("image_processor_type") {
            if t.as_str() != Some("Glm5NextImageProcessor") {
                return Err(ConfigError(format!("image_processor_type is {t}, expected \"Glm5NextImageProcessor\"")));
            }
        }
        for (key, slot) in [
            ("patch_size", &mut p.patch_size),
            ("merge_size", &mut p.merge_size),
            ("temporal_patch_size", &mut p.temporal_patch_size),
            ("min_image_tokens", &mut p.min_image_tokens),
            ("max_image_tokens", &mut p.max_image_tokens),
        ] {
            if let Some(v) = ip.get(key) {
                *slot = v
                    .as_u64()
                    .filter(|&n| n > 0 && n < 1 << 32)
                    .ok_or_else(|| ConfigError(format!("image_processor.{key} must be a positive integer, got {v}")))?
                    as usize;
            }
        }
        if let Some(v) = ip.get("patch_expand_factor") {
            if v.as_u64() != Some(1) {
                return Err(ConfigError(format!("patch_expand_factor {v} is not supported (only 1)")));
            }
        }
        if ip.get("do_rescale").is_some_and(|v| v.as_bool() != Some(true)) {
            return Err(ConfigError("do_rescale must be true".into()));
        }
        for (key, slot) in [("image_mean", &mut p.mean), ("image_std", &mut p.std)] {
            if let Some(v) = ip.get(key) {
                let a = v.as_array().filter(|a| a.len() == 3);
                let vals: Option<Vec<f64>> = a.and_then(|a| a.iter().map(Value::as_f64).collect());
                let vals = vals.ok_or_else(|| ConfigError(format!("image_processor.{key} must be 3 numbers, got {v}")))?;
                for (s, x) in slot.iter_mut().zip(vals) {
                    *s = x as f32;
                }
            }
        }
        p.validate()?;
        Ok(p)
    }

    /// Lower (or raise) the token budget, e.g. to the server's `--max-image-tokens`. The canvas rule is HF's with `max_image_tokens` overridden, which
    /// the HF processor accepts as a keyword; the model input for an image that fits is unchanged.
    pub fn with_max_image_tokens(mut self, n: usize) -> Result<Self, ConfigError> {
        self.max_image_tokens = n;
        self.validate()?;
        Ok(self)
    }

    pub fn with_profile(mut self, profile: Profile) -> Self {
        self.profile = profile;
        self
    }

    fn validate(&self) -> Result<(), ConfigError> {
        let e = |m: &str| Err(ConfigError(m.into()));
        if self.patch_size == 0 || self.merge_size == 0 || self.temporal_patch_size == 0 {
            return e("patch_size, merge_size and temporal_patch_size must be positive");
        }
        if self.min_image_tokens == 0 || self.max_image_tokens == 0 {
            return e("min_image_tokens and max_image_tokens must be positive");
        }
        if self.min_image_tokens > self.max_image_tokens {
            return e("min_image_tokens exceeds max_image_tokens");
        }
        if self.std.iter().any(|s| !s.is_finite() || *s <= 0.0) || self.mean.iter().any(|m| !m.is_finite()) {
            return e("image_mean must be finite and image_std positive");
        }
        if self.max_side == 0 || self.max_pixels == 0 {
            return e("size limits must be positive");
        }
        Ok(())
    }

    /// Pixels per side of the canvas must be a multiple of this (`patch * merge`).
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// Elements per patch: `3 * temporal * patch^2` (1176).
    pub fn patch_dim(&self) -> usize {
        3 * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    /// `smart_resize` of `Glm5NextImageProcessor` (`glm5_vision_canvas` of the fork): the canvas `(height, width)` for `num_frames` frames of
    /// `height x width` pixels.
    pub fn smart_resize(&self, num_frames: usize, height: usize, width: usize) -> Result<(usize, usize), VisionError> {
        if num_frames == 0 {
            return Err(VisionError::Unsupported("no frames".into()));
        }
        self.check_size(width, height)?;
        let factor = self.factor();
        let tf = self.temporal_patch_size;
        let pixels_per_token = tf * factor * factor;
        let min_pixels = self.min_image_tokens * pixels_per_token;
        let max_pixels = self.max_image_tokens * pixels_per_token;
        let aligned_frames = tf.max(((num_frames as f64 / tf as f64).round_ties_even() as usize) * tf);
        let mut ah = align(height, factor);
        let mut aw = align(width, factor);
        let mut budget = aligned_frames * ah * aw;
        if budget < min_pixels {
            let scale = (min_pixels as f64 / (num_frames * height * width) as f64).sqrt();
            ah = align(((height as f64 * scale).ceil() as usize).max(1), factor);
            aw = align(((width as f64 * scale).ceil() as usize).max(1), factor);
            budget = aligned_frames * ah * aw;
        }
        if budget > max_pixels {
            let minimum = aligned_frames * factor * factor;
            if max_pixels < minimum {
                return Err(VisionError::Unsupported(format!(
                    "max_image_tokens leaves {max_pixels} pixels, fewer than the {minimum} one aligned patch needs"
                )));
            }
            let (mut low, mut high) = (1usize, height);
            (ah, aw) = (factor, factor);
            while low <= high {
                let ch = (low + high) / 2;
                let cw = (((width * ch) as f64 / height as f64).floor() as usize).max(1);
                let (cand_h, cand_w) = (align(ch, factor), align(cw, factor));
                if aligned_frames * cand_h * cand_w <= max_pixels {
                    (ah, aw) = (cand_h, cand_w);
                    low = ch + 1;
                } else {
                    high = ch - 1;
                }
            }
        }
        Ok((ah, aw))
    }

    fn check_size(&self, width: usize, height: usize) -> Result<(), VisionError> {
        if width == 0 || height == 0 {
            return Err(VisionError::Unsupported(format!("an image of {width}x{height} pixels has no content")));
        }
        if width > self.max_side || height > self.max_side {
            return Err(VisionError::Unsupported(format!(
                "{width}x{height}: a side exceeds the limit of {} pixels",
                self.max_side
            )));
        }
        if width.checked_mul(height).is_none_or(|p| p > self.max_pixels) {
            return Err(VisionError::Unsupported(format!("{width}x{height} exceeds the limit of {} pixels", self.max_pixels)));
        }
        Ok(())
    }

    /// Canvas, content size, grid and placeholder count of a `width x height` image, without touching pixels.
    pub fn geometry(&self, width: usize, height: usize) -> Result<Geometry, VisionError> {
        self.check_size(width, height)?;
        let factor = self.factor();
        let tf = self.temporal_patch_size;
        // HF resizes a still image as `temporal_patch_size` frames (the single frame is duplicated into the temporal patch)
        let (th, tw) = self.smart_resize(tf, height, width)?;
        let pixels_per_token = tf * factor * factor;
        let mut scale = (th as f64 / height as f64).min(tw as f64 / width as f64);
        if tf * height * width >= pixels_per_token * self.min_image_tokens {
            scale = scale.min(1.0);
        }
        let ch = ((height as f64 * scale).floor() as usize).clamp(1, th);
        let cw = ((width as f64 * scale).floor() as usize).clamp(1, tw);
        let (gh, gw) = (th / self.patch_size, tw / self.patch_size);
        Ok(Geometry {
            canvas: (tw, th),
            content: (cw, ch),
            grid: (1, gh, gw),
            tokens: gh * gw / (self.merge_size * self.merge_size),
        })
    }

    /// Placeholder tokens for an image of this size (`Glm5NextProcessor.replace_image_token`: `grid.prod() // merge^2`).
    pub fn tokens(&self, width: usize, height: usize) -> Result<usize, VisionError> {
        Ok(self.geometry(width, height)?.tokens)
    }

    /// The 256-entry value table of each channel for the profile's normalisation, in float32.
    pub fn lut_f32(&self) -> [[f32; 256]; 3] {
        let mut lut = [[0f32; 256]; 3];
        match self.profile {
            Profile::Hf => {
                // HF's fused path: mean and std are multiplied by 1/rescale_factor = 255 (float32), then `(x - mean) / std`
                for (c, table) in lut.iter_mut().enumerate() {
                    let (m, s) = (self.mean[c] * 255.0f32, self.std[c] * 255.0f32);
                    for (v, o) in table.iter_mut().enumerate() {
                        *o = (v as f32 - m) / s;
                    }
                }
            }
            Profile::Fork => {
                // numpy: `x * rescale_factor` (float32) then `(x - mean) / std`
                let rescale = (1.0f64 / 255.0) as f32;
                for (c, table) in lut.iter_mut().enumerate() {
                    for (v, o) in table.iter_mut().enumerate() {
                        *o = ((v as f32) * rescale - self.mean[c]) / self.std[c];
                    }
                }
            }
        }
        lut
    }

    /// The same table as fp16 bit patterns (round to nearest even, what `torch.half()` / `astype(float16)` do).
    pub fn lut_f16(&self) -> [[u16; 256]; 3] {
        self.lut_f32().map(|table| table.map(f32_to_f16_bits))
    }

    fn validate_image(&self, img: &Rgb) -> Result<(), VisionError> {
        self.check_size(img.width, img.height)?;
        if img.data.len() != img.width * img.height * 3 {
            return Err(VisionError::Decode(format!(
                "pixel buffer holds {} bytes, a {}x{} RGB image needs {}",
                img.data.len(),
                img.width,
                img.height,
                img.width * img.height * 3
            )));
        }
        Ok(())
    }

    /// The content of the canvas: the image resized to its content size (a view of the input when that is its size), and the geometry.
    fn content_rgb<'a>(&self, img: &'a Rgb) -> Result<(Cow<'a, Rgb>, Geometry), VisionError> {
        self.validate_image(img)?;
        let g = self.geometry(img.width, img.height)?;
        let (cw, ch) = g.content;
        let content: Cow<'a, Rgb> = if (cw, ch) == (img.width, img.height) {
            Cow::Borrowed(img)
        } else {
            // each profile resizes with its reference's kernel (see the module documentation)
            let data = match self.profile {
                Profile::Hf => torch_resample::bicubic_rgb_torch_with(&img.data, img.width, img.height, cw, ch, self.threads),
                Profile::Fork => resample::bicubic_rgb_pil_with(&img.data, img.width, img.height, cw, ch, self.threads),
            };
            Cow::Owned(Rgb { width: cw, height: ch, data })
        };
        Ok((content, g))
    }

    /// The canvas the tower sees, as RGB bytes: the image resized to its content size (when that differs) and padded with black.
    pub fn canvas_rgb<'a>(&self, img: &'a Rgb) -> Result<(Cow<'a, Rgb>, Geometry), VisionError> {
        let (content, g) = self.content_rgb(img)?;
        if g.content == g.canvas {
            return Ok((content, g));
        }
        Ok((Cow::Owned(pad_to_canvas(&content, &g)), g))
    }

    /// Normalise and patchify with a value table per channel: `[gh * gw, 3 * temporal * patch^2]`, block-major over merge blocks
    /// (`layout::patchify`), on the workers the size of the canvas is worth (at most [`GlmPreprocessing::cap`]).
    fn patchify<T: Copy + Default + Send + Sync>(&self, canvas: &Rgb, lut: &[[T; 256]; 3]) -> Vec<T> {
        let workers = layout::plan(self.cap(), canvas.width * canvas.height, false).workers;
        layout::patchify(&canvas.data, canvas.width, canvas.height, lut, self.shape(), workers)
    }

    /// The float32 tensor of the HF processor's `pixel_values` (profile `Hf`; exactly its values where no resize happens) or the numpy tensor before
    /// the fork's `.half()` (profile `Fork`): `[gh * gw, 1176]` plus the geometry.
    pub fn preprocess_f32(&self, img: &Rgb) -> Result<(Vec<f32>, Geometry), VisionError> {
        let (canvas, g) = self.canvas_rgb(img)?;
        Ok((self.patchify(&canvas, &self.lut_f32()), g))
    }

    /// Preprocess one image into the fp16 patch tensor the tower consumes (`Preprocessed.patches`, `[gh * gw, 1176]`). The hash of the canvas is
    /// taken from the resized content (`canvas_hash`: the padding costs nothing) on a thread of its own, which starts as soon as the resize is
    /// done and runs while the canvas is padded and laid out (`layout::beside`).
    pub fn preprocess(&self, img: &Rgb) -> Result<Preprocessed, VisionError> {
        let (content, g) = self.content_rgb(img)?;
        let ((cw, ch), (tw, th)) = (g.content, g.canvas);
        let (pixels, cap) = (tw * th, self.cap());
        let lut = self.lut_f16();
        let plan = layout::plan(cap, pixels, true);
        let (patches, pixel_hash) = layout::beside(
            plan.hash_aside,
            || crate::canvas_hash(tw, th, &content.data, cw, ch),
            || {
                let padded;
                let canvas: &Rgb = if g.content == g.canvas {
                    &content
                } else {
                    padded = pad_to_canvas(&content, &g);
                    &padded
                };
                layout::patchify(&canvas.data, tw, th, &lut, self.shape(), plan.workers)
            },
        );
        Ok(Preprocessed { size: g.canvas, grid: g.grid, patches, patch_dim: self.patch_dim(), pixel_hash })
    }

    /// Decode PNG or JPEG bytes with the alpha handling of this profile's reference.
    pub fn decode(&self, bytes: &[u8]) -> Result<Rgb, VisionError> {
        decode_rgb(bytes, self.profile.alpha_policy())
    }

    /// LM placeholder tokens of a preprocessed image.
    pub fn n_tokens(&self, pre: &Preprocessed) -> usize {
        pre.n_patches() / (self.merge_size * self.merge_size)
    }
}

/// `content` (`cw x ch` RGB bytes, `cw` at least 1) at the top left of a black `tw x th` canvas (`tw >= cw`, `th >= ch`): every byte of the canvas is
/// written once, the rows of the content and the padding after each, then the padding rows below (the canvas is not zero-filled first and then
/// overwritten; byte for byte the same canvas).
fn pad_black(content: &[u8], cw: usize, ch: usize, tw: usize, th: usize) -> Vec<u8> {
    debug_assert!(cw >= 1 && cw <= tw && ch <= th && content.len() == cw * ch * 3);
    let mut data = Vec::with_capacity(tw * th * 3);
    for row in content.chunks_exact(cw * 3).take(ch) {
        data.extend_from_slice(row);
        data.resize(data.len() + (tw - cw) * 3, 0);
    }
    data.resize(tw * th * 3, 0);
    data
}

/// The black-padded canvas of `content` (the content size and the canvas of `g` differ).
fn pad_to_canvas(content: &Rgb, g: &Geometry) -> Rgb {
    let ((cw, ch), (tw, th)) = (g.content, g.canvas);
    Rgb { width: tw, height: th, data: pad_black(&content.data, cw, ch, tw, th) }
}

/// The still image's patches without the temporal repeat: `[P, 3 * patch^2]` (588 for GLM-5.3-Flash), the first temporal slot of each channel.
/// Exact when the tower folds the temporal convolution (`W[:, :, 0] + W[:, :, 1]`), because both slots hold the same pixels. Panics when the
/// layout is not a still image's (`patch_dim` not divisible by `3 * temporal`, or the two slots differ). Threaded over patches for a large image
/// ([`still_patches_with`] with the default worker count).
pub fn still_patches(pre: &Preprocessed, temporal_patch_size: usize) -> Vec<u16> {
    still_patches_with(pre, temporal_patch_size, 0)
}

/// [`still_patches`] on at most `max_workers` threads (0: `resize::default_workers`; small tensors run inline). The output and the panic are the
/// same for every value: a worker that finds differing slots records the patch and stops, and the calling thread repeats the check on the
/// first such patch, in the order of the serial loop, so the message (and the two slices it prints) is the serial one's.
pub fn still_patches_with(pre: &Preprocessed, temporal_patch_size: usize, max_workers: usize) -> Vec<u16> {
    let cap = if max_workers == 0 { crate::resize::default_workers() } else { max_workers };
    still_patches_by(pre, temporal_patch_size, |pixels| layout::plan(cap, pixels, false).workers)
}

/// [`still_patches`] on exactly `workers` threads however little the work (the tests compare thread counts with it).
#[cfg(test)]
pub(crate) fn still_patches_exact(pre: &Preprocessed, temporal_patch_size: usize, workers: usize) -> Vec<u16> {
    still_patches_by(pre, temporal_patch_size, |_| workers)
}

/// The loop of [`still_patches_with`]; `workers_for(pixels)` says how many threads `pixels` pixels (patches times pixels a patch) get.
fn still_patches_by(pre: &Preprocessed, temporal_patch_size: usize, workers_for: impl Fn(usize) -> usize) -> Vec<u16> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let per_slot = pre.patch_dim / (3 * temporal_patch_size);
    assert_eq!(per_slot * 3 * temporal_patch_size, pre.patch_dim, "patch_dim is not 3 * temporal * pixels");
    let tps = temporal_patch_size;
    let n_patches = pre.patches.chunks_exact(pre.patch_dim).len();
    let mut out = vec![0u16; n_patches * 3 * per_slot];
    let workers = workers_for(n_patches * per_slot);
    let first_bad = AtomicUsize::new(usize::MAX);
    // the output row of patch `i` is its 3 channels' first slots; the source patch is `patch_dim` elements, a channel `tps` slots of `per_slot`
    resample::par_rows(&mut out, 3 * per_slot, workers, |i, dst| {
        let patch = &pre.patches[i * pre.patch_dim..(i + 1) * pre.patch_dim];
        for (c, d) in dst.chunks_exact_mut(per_slot).enumerate() {
            let slots = &patch[c * tps * per_slot..(c + 1) * tps * per_slot];
            let (first, rest) = slots.split_at(per_slot);
            if rest.chunks_exact(per_slot).any(|slot| slot != first) {
                first_bad.fetch_min(i, Ordering::Relaxed);
                return;
            }
            d.copy_from_slice(first);
        }
    });
    let bad = first_bad.load(Ordering::Relaxed);
    if bad != usize::MAX {
        let patch = &pre.patches[bad * pre.patch_dim..(bad + 1) * pre.patch_dim];
        for c in 0..3 {
            let first = &patch[c * tps * per_slot..c * tps * per_slot + per_slot];
            for t in 1..tps {
                let s = c * tps * per_slot + t * per_slot;
                assert_eq!(first, &patch[s..s + per_slot], "temporal slots differ: not a still image");
            }
        }
        unreachable!("a worker saw differing temporal slots in patch {bad} and the serial check did not");
    }
    out
}

impl ImagePreprocessor for GlmPreprocessing {
    fn family(&self) -> &'static str {
        "glm5-next"
    }

    fn id(&self) -> String {
        let bits = |a: &[f32; 3]| a.iter().map(|v| format!("{:08x}", v.to_bits())).collect::<Vec<_>>().join(",");
        format!(
            "glm5-next-vision-v3:{}:p{}m{}t{}:tok{}-{}:mean{}:std{}",
            self.profile.name(),
            self.patch_size,
            self.merge_size,
            self.temporal_patch_size,
            self.min_image_tokens,
            self.max_image_tokens,
            bits(&self.mean),
            bits(&self.std)
        )
    }

    fn decode(&self, bytes: &[u8]) -> Result<Rgb, VisionError> {
        GlmPreprocessing::decode(self, bytes)
    }

    fn plan(&self, width: usize, height: usize) -> Result<ImagePlan, VisionError> {
        let g = self.geometry(width, height)?;
        Ok(ImagePlan { canvas: g.canvas, grid: g.grid, tokens: g.tokens })
    }

    fn preprocess(&self, img: &Rgb) -> Result<Preprocessed, VisionError> {
        GlmPreprocessing::preprocess(self, img)
    }
}

/// Today's GLM canvas, patchify, preprocess and `still_patches`, copied verbatim from the commit that preceded the layout rewrite (the zero-filled
/// canvas with the row copies, the per-element table lookups, FNV-1a), except that the patchify runs on one thread (`par_rows` with 1: the old call
/// took its count from `workers`) and that the resize dispatches on the profile as `GlmPreprocessing::content_rgb` does, with the serial torch reference
/// (`torch_resample::reference`) for `Hf`: the oracle the new code is compared with byte for byte.
#[cfg(test)]
pub(crate) mod reference {
    use super::*;

    pub(crate) fn canvas_rgb<'a>(p: &GlmPreprocessing, img: &'a Rgb) -> Result<(Cow<'a, Rgb>, Geometry), VisionError> {
        p.validate_image(img)?;
        let g = p.geometry(img.width, img.height)?;
        let (cw, ch) = g.content;
        let (tw, th) = g.canvas;
        let content: Cow<'a, Rgb> = if (cw, ch) == (img.width, img.height) {
            Cow::Borrowed(img)
        } else {
            let data = match p.profile {
                Profile::Hf => crate::torch_resample::reference::bicubic_rgb_torch(&img.data, img.width, img.height, cw, ch),
                Profile::Fork => resample::bicubic_rgb_pil_with(&img.data, img.width, img.height, cw, ch, p.threads),
            };
            Cow::Owned(Rgb { width: cw, height: ch, data })
        };
        if (cw, ch) == (tw, th) {
            return Ok((content, g));
        }
        let mut data = vec![0u8; tw * th * 3];
        for y in 0..ch {
            data[y * tw * 3..y * tw * 3 + cw * 3].copy_from_slice(&content.data[y * cw * 3..(y + 1) * cw * 3]);
        }
        Ok((Cow::Owned(Rgb { width: tw, height: th, data }), g))
    }

    pub(crate) fn patchify<T: Copy + Default + Send + Sync>(p: &GlmPreprocessing, canvas: &Rgb, lut: &[[T; 256]; 3]) -> Vec<T> {
        let (ps, m, tf) = (p.patch_size, p.merge_size, p.temporal_patch_size);
        let (gh, gw) = (canvas.height / ps, canvas.width / ps);
        let per_frame = ps * ps;
        let mut out = vec![T::default(); gh * gw * p.patch_dim()];
        // one row of merge blocks (m rows of patches) is a contiguous run of the output, and rows are independent
        let block_row = (gw / m) * m * m * p.patch_dim();
        resample::par_rows(&mut out, block_row, 1, |br, dst| {
            let mut idx = 0usize;
            for bc in 0..gw / m {
                for mr in 0..m {
                    for mc in 0..m {
                        let (py, px) = ((br * m + mr) * ps, (bc * m + mc) * ps);
                        for (c, table) in lut.iter().enumerate() {
                            // the first temporal slot, then its copies (a still image repeats its single frame)
                            let start = idx;
                            for r in 0..ps {
                                let row = ((py + r) * canvas.width + px) * 3 + c;
                                for k in 0..ps {
                                    dst[idx] = table[canvas.data[row + k * 3] as usize];
                                    idx += 1;
                                }
                            }
                            for ti in 1..tf {
                                dst.copy_within(start..start + per_frame, start + ti * per_frame);
                            }
                            idx += (tf - 1) * per_frame;
                        }
                    }
                }
            }
            debug_assert_eq!(idx, dst.len());
        });
        out
    }

    fn fnv(h: u64, bytes: &[u8]) -> u64 {
        bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
    }

    pub(crate) fn preprocess_f32(p: &GlmPreprocessing, img: &Rgb) -> Result<(Vec<f32>, Geometry), VisionError> {
        let (canvas, g) = canvas_rgb(p, img)?;
        Ok((patchify(p, &canvas, &p.lut_f32()), g))
    }

    pub(crate) fn preprocess(p: &GlmPreprocessing, img: &Rgb) -> Result<Preprocessed, VisionError> {
        let (canvas, g) = canvas_rgb(p, img)?;
        let patches = patchify(p, &canvas, &p.lut_f16());
        let mut h = fnv(0xcbf2_9ce4_8422_2325, &(canvas.width as u64).to_le_bytes());
        h = fnv(h, &(canvas.height as u64).to_le_bytes());
        h = fnv(h, &canvas.data);
        Ok(Preprocessed { size: g.canvas, grid: g.grid, patches, patch_dim: p.patch_dim(), pixel_hash: h })
    }

    pub(crate) fn still_patches(pre: &Preprocessed, temporal_patch_size: usize) -> Vec<u16> {
        let per_slot = pre.patch_dim / (3 * temporal_patch_size);
        assert_eq!(per_slot * 3 * temporal_patch_size, pre.patch_dim, "patch_dim is not 3 * temporal * pixels");
        let mut out = Vec::with_capacity(pre.n_patches() * 3 * per_slot);
        for patch in pre.patches.chunks_exact(pre.patch_dim) {
            for c in 0..3 {
                let first = &patch[c * temporal_patch_size * per_slot..c * temporal_patch_size * per_slot + per_slot];
                for t in 1..temporal_patch_size {
                    let s = c * temporal_patch_size * per_slot + t * per_slot;
                    assert_eq!(first, &patch[s..s + per_slot], "temporal slots differ: not a still image");
                }
                out.extend_from_slice(first);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default preprocessing (profile `Fork`).
    fn p() -> GlmPreprocessing {
        GlmPreprocessing::default()
    }

    #[test]
    fn the_default_profile_is_the_forks() {
        assert_eq!(Profile::default(), Profile::Fork);
        assert_eq!(p().profile, Profile::Fork);
        assert_eq!(p(), GlmPreprocessing::glm53_flash(Profile::Fork));
        assert_ne!(p(), GlmPreprocessing::glm53_flash(Profile::Hf));
        // the default composites alpha on white (the fork's convert_to_rgb) and is not the HF profile in the cache key
        assert_eq!(p().profile.alpha_policy(), AlphaPolicy::CompositeWhite);
        assert!(p().id().contains(":fork:"), "{}", p().id());
        assert!(!p().id().contains(":hf:"));
        // the cache key carries the version of the preprocessing: v3 since `Hf` resizes with torch's kernel (v2 resized it with Pillow's)
        assert!(p().id().starts_with("glm5-next-vision-v3:fork:"), "{}", p().id());
        assert!(p().with_profile(Profile::Hf).id().starts_with("glm5-next-vision-v3:hf:"));
        // a transparent pixel decodes to white by default and keeps its colour under `Hf`
        let mut png = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut png, 1, 1);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header().unwrap().write_image_data(&[200, 30, 40, 0]).unwrap();
        }
        assert_eq!(p().decode(&png).unwrap().data, [255, 255, 255]);
        assert_eq!(p().with_profile(Profile::Hf).decode(&png).unwrap().data, [200, 30, 40]);
    }

    /// A deterministic noise image of `w x h` pixels.
    fn noise(w: usize, h: usize) -> Rgb {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let data = (0..w * h * 3)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect();
        Rgb { width: w, height: h, data }
    }

    #[test]
    fn fork_resizes_with_pillow_including_its_two_step_and_hf_with_the_torch_kernel_in_one_call() {
        // 10 x 1500 under a 16-token cap: over the budget, so the height shrinks, and taller than 100x its width: Pillow >= 12.2 resizes it in
        // two steps (vertical pass first), which differs from the one-call result; torch's kernel has no such step. The geometry is asserted so the
        // test cannot drift off the rule
        let img = noise(10, 1500);
        for profile in [Profile::Hf, Profile::Fork] {
            let pre = GlmPreprocessing::glm53_flash(profile).with_max_image_tokens(16).unwrap();
            let g = pre.geometry(10, 1500).unwrap();
            let (cw, ch) = g.content;
            assert!(ch < 1500 && 1500 > 100 * 10, "{g:?}");
            let (canvas, _) = pre.canvas_rgb(&img).unwrap();
            let pillow = resample::bicubic_rgb_pil(&img.data, 10, 1500, cw, ch);
            let pillow_one_step = crate::resize::bicubic_rgb(&img.data, 10, 1500, cw, ch);
            assert_ne!(pillow, pillow_one_step, "the image must exercise the two-step rule");
            let torch = torch_resample::bicubic_rgb_torch(&img.data, 10, 1500, cw, ch);
            assert_ne!(torch, pillow, "the two kernels must give different pixels here");
            let want = if profile == Profile::Hf { &torch } else { &pillow };
            for y in 0..ch {
                assert_eq!(
                    &canvas.data[y * g.canvas.0 * 3..y * g.canvas.0 * 3 + cw * 3],
                    &want[y * cw * 3..(y + 1) * cw * 3],
                    "{profile:?} row {y}"
                );
            }
        }
    }

    #[test]
    fn hf_and_fork_resize_with_different_kernels_and_differ_in_alpha_and_normalisation() {
        // a resized opaque image: the canvases are each profile's resize of the same pixels (torch's kernel for Hf, Pillow's for Fork: a few percent of
        // the samples differ by a level or two); the tensors differ by the normalisation arithmetic as well
        let img = noise(96, 96); // 96 x 96 x 2 pixels are under the 16-token minimum: upscaled to 112 x 112
        let (hf, fork) = (p().with_profile(Profile::Hf), p().with_profile(Profile::Fork));
        let (a, g) = hf.canvas_rgb(&img).unwrap();
        assert_eq!(g.content, (112, 112));
        let b = fork.canvas_rgb(&img).unwrap().0;
        assert_eq!(a.data, torch_resample::bicubic_rgb_torch(&img.data, 96, 96, 112, 112));
        assert_eq!(b.data, resample::bicubic_rgb_pil(&img.data, 96, 96, 112, 112));
        let n_diff = a.data.iter().zip(&b.data).filter(|(x, y)| x != y).count();
        let worst = a.data.iter().zip(&b.data).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
        assert!(n_diff > 0 && worst <= 3, "{n_diff} of {} samples differ, by at most {worst}", a.data.len());
        assert_ne!(hf.preprocess(&img).unwrap().pixel_hash, fork.preprocess(&img).unwrap().pixel_hash);
        assert_ne!(hf.preprocess(&img).unwrap().patches, fork.preprocess(&img).unwrap().patches);
        // where nothing is resized the canvas, and so the hash, is the same under both
        let flat = noise(150, 90);
        assert_eq!(hf.geometry(150, 90).unwrap().content, (150, 90));
        assert_eq!(hf.preprocess(&flat).unwrap().pixel_hash, fork.preprocess(&flat).unwrap().pixel_hash);
    }

    #[test]
    fn the_threads_field_reaches_the_resize_of_each_profile_and_never_changes_its_bytes() {
        // 1600x1200 under a 1000-token cap is reduced (about 1045x784): plenty of work for every cap below
        let img = noise(1600, 1200);
        for profile in [Profile::Fork, Profile::Hf] {
            let mut p = GlmPreprocessing { max_image_tokens: 1000, ..GlmPreprocessing::glm53_flash(profile) };
            let (cw, ch) = p.geometry(1600, 1200).unwrap().content;
            assert!((cw, ch) != (1600, 1200));
            let work = match profile {
                Profile::Fork => crate::resize::resize_work(1600, 1200, cw, ch),
                Profile::Hf => torch_resample::torch_resize_work(1600, 1200, cw, ch),
            };
            let mut first: Option<Vec<u8>> = None;
            for threads in [1usize, 2, 5, 0] {
                p.threads = threads;
                let (canvas, _) = p.canvas_rgb(&img).unwrap();
                let cap = if threads == 0 { crate::resize::default_workers() } else { threads };
                let workers = crate::resize::resize_workers(cap, work);
                let chosen = match profile {
                    Profile::Fork => crate::resize::last_workers(),
                    Profile::Hf => torch_resample::last_workers(),
                };
                assert_eq!(chosen, workers, "{profile:?} threads {threads}");
                if threads > 0 {
                    assert_eq!(workers, threads, "this much work is worth {threads} workers");
                }
                assert_eq!(&canvas.data, first.get_or_insert_with(|| canvas.data.clone()), "{profile:?} threads {threads}");
            }
        }
    }

    #[test]
    fn canvas_rule_on_the_branches() {
        let p = p();
        // inside the budget: align up, no resize
        assert_eq!(p.geometry(280, 196).unwrap().canvas, (280, 196));
        assert_eq!(p.geometry(85, 112).unwrap().canvas, (112, 112));
        assert_eq!(p.geometry(85, 112).unwrap().content, (85, 112));
        // under the minimum (raw pixels): scaled up to 16 tokens, 100x100 is upscaled to the 112x112 canvas without padding
        let g = p.geometry(100, 100).unwrap();
        assert_eq!((g.canvas, g.content, g.tokens), ((112, 112), (112, 112), 16));
        // far under: 1x1 -> 112x112 canvas
        assert_eq!(p.geometry(1, 1).unwrap().canvas, (112, 112));
        // the budget edge: 2800x2240 is exactly 8000 tokens and is not touched, one column more is searched down
        let g = p.geometry(2800, 2240).unwrap();
        assert_eq!((g.canvas, g.content, g.tokens), ((2800, 2240), (2800, 2240), 8000));
        let g = p.geometry(2801, 2240).unwrap();
        assert_eq!((g.canvas, g.tokens), ((2800, 2240), 8000));
        assert_eq!(g.content, (2800, 2239), "scale 0.99964 floors the height");
        // extreme aspect needs no resize while it fits
        assert_eq!(p.geometry(1, 8000).unwrap().canvas, (28, 8008));
    }

    #[test]
    fn config_parsing_is_strict_where_it_must_be() {
        let ok: Value = serde_json::from_str(
            r#"{"image_processor":{"patch_size":14,"merge_size":2,"temporal_patch_size":2,"min_image_tokens":16,"max_image_tokens":8000,
            "image_mean":[0.48145466,0.4578275,0.40821073],"image_std":[0.26862954,0.26130258,0.27577711],"patch_expand_factor":1,
            "do_rescale":true,"image_processor_type":"Glm5NextImageProcessor"}}"#,
        )
        .unwrap();
        assert_eq!(GlmPreprocessing::from_processor_config(&ok, Profile::default()).unwrap(), p());
        let bad = |edit: &dyn Fn(&mut Value)| {
            let mut v = ok.clone();
            edit(&mut v);
            GlmPreprocessing::from_processor_config(&v, Profile::default()).unwrap_err().0
        };
        assert!(bad(&|v| v["image_processor"]["patch_size"] = 0.into()).contains("patch_size"));
        assert!(bad(&|v| v["image_processor"]["max_image_tokens"] = "x".into()).contains("max_image_tokens"));
        assert!(bad(&|v| v["image_processor"]["min_image_tokens"] = 9000.into()).contains("exceeds"));
        assert!(bad(&|v| v["image_processor"]["patch_expand_factor"] = 2.into()).contains("patch_expand_factor"));
        assert!(bad(&|v| v["image_processor"]["do_rescale"] = false.into()).contains("do_rescale"));
        assert!(bad(&|v| v["image_processor"]["image_std"] = serde_json::json!([0.0, 1.0, 1.0])).contains("positive"));
        assert!(bad(&|v| v["image_processor"]["image_mean"] = serde_json::json!([0.5, 0.5])).contains("3 numbers"));
        assert!(bad(&|v| v["image_processor"]["image_processor_type"] = "Qwen2VLImageProcessor".into()).contains("Glm5Next"));
        assert!(GlmPreprocessing::from_processor_config(&serde_json::json!({}), Profile::default()).is_err());
        // a missing key takes the class default
        let min = serde_json::json!({"image_processor": {"max_image_tokens": 1024}});
        assert_eq!(
            GlmPreprocessing::from_processor_config(&min, Profile::default()).unwrap(),
            p().with_max_image_tokens(1024).unwrap()
        );
    }

    #[test]
    fn degenerate_inputs_are_refused_with_a_reason() {
        let p = p();
        let msg = |r: Result<Preprocessed, VisionError>| match r {
            Err(e) => e.to_string(),
            Ok(_) => panic!("accepted"),
        };
        assert!(msg(p.preprocess(&Rgb { width: 0, height: 5, data: vec![] })).contains("no content"));
        assert!(msg(p.preprocess(&Rgb { width: 5, height: 0, data: vec![] })).contains("no content"));
        assert!(msg(p.preprocess(&Rgb { width: 4, height: 4, data: vec![0; 47] })).contains("needs 48"));
        assert!(msg(p.preprocess(&Rgb { width: 4, height: 4, data: vec![0; 49] })).contains("needs 48"));
        assert!(p.geometry(70_000, 2).unwrap_err().to_string().contains("side exceeds"));
        assert!(p.geometry(10_000, 10_000).unwrap_err().to_string().contains("exceeds the limit"));
        assert!(p.geometry(usize::MAX, usize::MAX).is_err());
        assert!(p.tokens(0, 0).is_err());
        // a one-token budget still yields a one-token canvas; a budget below one aligned patch is refused rather than looping or returning a
        // zero canvas (only reachable with many frames: video)
        let mut tiny = p.clone();
        tiny.max_image_tokens = 1;
        tiny.min_image_tokens = 1;
        assert_eq!(tiny.geometry(5000, 5000).unwrap().tokens, 1);
        assert!(tiny.smart_resize(64, 5000, 5000).unwrap_err().to_string().contains("one aligned patch needs"));
        // the public arithmetic does not trust its caller either: no division by zero, no overflow
        assert!(p.smart_resize(2, 0, 5).is_err() && p.smart_resize(2, 5, 0).is_err() && p.smart_resize(0, 5, 5).is_err());
        assert!(p.smart_resize(2, usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn fork_and_hf_normalisation_differ_in_exactly_the_recorded_values() {
        let (hf, fork) = (p().with_profile(Profile::Hf), p().with_profile(Profile::Fork));
        let (a, b) = (hf.lut_f16(), fork.lut_f16());
        let n16: usize = (0..3).map(|c| (0..256).filter(|&v| a[c][v] != b[c][v]).count()).sum();
        assert_eq!(n16, 3, "norm_luts.json records 3 differing fp16 values of 768");
        let (a, b) = (hf.lut_f32(), fork.lut_f32());
        let n32: usize = (0..3).map(|c| (0..256).filter(|&v| a[c][v].to_bits() != b[c][v].to_bits()).count()).sum();
        assert_eq!(n32, 661, "norm_luts.json records 661 differing fp32 values of 768");
    }

    #[test]
    fn the_entries_equal_the_verbatim_reference_on_a_few_images() {
        for profile in [Profile::Fork, Profile::Hf] {
            let p = GlmPreprocessing::glm53_flash(profile);
            for (w, h) in [(1usize, 1usize), (30, 20), (85, 112), (112, 85), (300, 200), (640, 480), (37, 29)] {
                let img = noise(w, h);
                assert_eq!(p.preprocess(&img), reference::preprocess(&p, &img), "{profile:?} {w}x{h}");
                assert_eq!(p.preprocess_f32(&img), reference::preprocess_f32(&p, &img), "{profile:?} {w}x{h} f32");
                let (a, ga) = p.canvas_rgb(&img).unwrap();
                let (b, gb) = reference::canvas_rgb(&p, &img).unwrap();
                assert_eq!((a.into_owned(), ga), (b.into_owned(), gb), "{profile:?} {w}x{h} canvas");
            }
        }
        let pre = p().preprocess(&noise(50, 30)).unwrap();
        assert_eq!(still_patches(&pre, 2), reference::still_patches(&pre, 2));
    }

    use crate::resize::reference::{noise as noise_bytes, Rng};

    const THREADS: [usize; 7] = [1, 2, 3, 5, 8, 16, 64];

    fn random_image(r: &mut Rng, w: usize, h: usize) -> Rgb {
        Rgb { width: w, height: h, data: noise_bytes(w * h * 3, r.next()) }
    }

    #[test]
    fn the_entries_and_the_canvas_equal_the_oracle_on_320_random_images_in_both_profiles() {
        let mut r = Rng::new(5);
        // pad kinds seen: (right only, bottom only, both, none), and resized
        let mut kinds = [0usize; 4];
        let mut resized = 0;
        for i in 0..320 {
            let (w, h) = match i % 10 {
                0 => (1, 1),
                1 => (1, 1 + r.below(700)),
                2 => (1 + r.below(700), 1),
                3 => (28 * (1 + r.below(12)), 28 * (1 + r.below(12))),
                4 => (28 * (1 + r.below(12)) + 1 + r.below(27), 28 * (1 + r.below(12)) + 1 + r.below(27)),
                5 => (10 + r.below(20), 28 * (1 + r.below(30)) + r.below(28)),
                _ => (1 + r.below(300), 1 + r.below(300)),
            };
            let image = random_image(&mut r, w, h);
            for profile in [Profile::Fork, Profile::Hf] {
                // the second profile, and thread caps, on every other image: the result must not depend on either
                let p = GlmPreprocessing { threads: [0, 1, 2, 7][i % 4], ..GlmPreprocessing::glm53_flash(profile) };
                let (canvas, g) = p.canvas_rgb(&image).unwrap();
                let (want_canvas, want_g) = reference::canvas_rgb(&p, &image).unwrap();
                assert_eq!((canvas.as_ref(), g), (want_canvas.as_ref(), want_g), "{profile:?} {w}x{h} canvas");
                assert_eq!(p.preprocess(&image), reference::preprocess(&p, &image), "{profile:?} {w}x{h}");
                assert_eq!(p.preprocess_f32(&image), reference::preprocess_f32(&p, &image), "{profile:?} {w}x{h} f32");
                if profile == Profile::Fork {
                    let (cw, ch) = g.content;
                    let (tw, th) = g.canvas;
                    kinds[match (cw < tw, ch < th) {
                        (true, false) => 0,
                        (false, true) => 1,
                        (true, true) => 2,
                        (false, false) => 3,
                    }] += 1;
                    resized += (g.content != (w, h)) as usize;
                }
            }
        }
        eprintln!("320 random images, pad kinds (right, bottom, both, none) {kinds:?}, {resized} resized");
        assert!(kinds.iter().all(|&k| k >= 20) && resized >= 40, "pad kinds {kinds:?}, {resized} resized");
    }

    #[test]
    fn the_padded_canvas_is_the_zero_filled_and_copied_one_for_every_shape_of_pad() {
        let mut r = Rng::new(9);
        let mut shapes = 0;
        // content size (cw, ch) and canvas (tw, th); every combination of equal and larger sides is in the first loops, random ones after
        let mut cases = Vec::new();
        for (cw, tw) in [(1usize, 1usize), (1, 2), (1, 28), (5, 5), (5, 6), (27, 28), (28, 28), (28, 56)] {
            for (ch, th) in [(1usize, 1usize), (1, 2), (1, 28), (5, 5), (5, 6), (27, 28), (28, 28), (28, 56)] {
                cases.push((cw, ch, tw, th));
            }
        }
        for _ in 0..260 {
            let (cw, ch) = (1 + r.below(120), 1 + r.below(120));
            cases.push((cw, ch, cw + r.below(60), ch + r.below(60)));
        }
        for (cw, ch, tw, th) in cases {
            let content = noise_bytes(cw * ch * 3, shapes as u64 + 1);
            // today's canvas: zero-fill, then copy the rows of the content
            let mut want = vec![0u8; tw * th * 3];
            for y in 0..ch {
                want[y * tw * 3..y * tw * 3 + cw * 3].copy_from_slice(&content[y * cw * 3..(y + 1) * cw * 3]);
            }
            assert_eq!(pad_black(&content, cw, ch, tw, th), want, "{cw}x{ch} in {tw}x{th}");
            shapes += 1;
        }
        eprintln!("{shapes} (content, canvas) shapes");
        assert!(shapes >= 300, "{shapes} shapes");
    }

    /// A GLM preprocessing whose layout is `(ps, m, t)` (the config accepts any positive sizes) with random normalisation statistics.
    fn glm_layout(ps: usize, m: usize, t: usize, profile: Profile, r: &mut Rng) -> GlmPreprocessing {
        let mut v = || (r.below(1000) as f32) / 1000.0;
        GlmPreprocessing {
            patch_size: ps,
            merge_size: m,
            temporal_patch_size: t,
            mean: [v(), v(), v()],
            std: [0.05 + v(), 0.05 + v(), 0.05 + v()],
            ..GlmPreprocessing::glm53_flash(profile)
        }
    }

    #[test]
    fn the_patchify_equals_the_oracle_in_f16_and_f32_for_both_profiles_and_every_thread_count() {
        let mut r = Rng::new(13);
        let layouts = [(14usize, 2usize, 2usize), (14, 2, 2), (1, 1, 1), (2, 3, 3), (3, 2, 1), (5, 1, 4), (7, 2, 2)];
        let mut n = 0;
        let mut tall = 0;
        for round in 0..210 {
            let (ps, m, t) = layouts[round % layouts.len()];
            let profile = if round % 2 == 0 { Profile::Fork } else { Profile::Hf };
            let p = glm_layout(ps, m, t, profile, &mut r);
            let f = ps * m;
            let (wu, hu) = match round % 7 {
                0 => (1, 64 + r.below(8)),
                1 => (64 + r.below(8), 1),
                _ => (1 + r.below(8), 1 + r.below(8)),
            };
            let (w, h) = (wu * f, hu * f);
            let canvas = random_image(&mut r, w, h);
            let (l16, l32) = (p.lut_f16(), p.lut_f32());
            let (want16, want32) = (reference::patchify(&p, &canvas, &l16), reference::patchify(&p, &canvas, &l32));
            for workers in THREADS {
                let a = layout::patchify(&canvas.data, w, h, &l16, p.shape(), workers);
                assert_eq!(a, want16, "{w}x{h} layout {ps}/{m}/{t} {profile:?} f16, {workers} threads");
                let b = layout::patchify(&canvas.data, w, h, &l32, p.shape(), workers);
                assert!(
                    b.iter().zip(&want32).all(|(x, y)| x.to_bits() == y.to_bits()) && b.len() == want32.len(),
                    "{w}x{h} layout {ps}/{m}/{t} {profile:?} f32, {workers} threads"
                );
            }
            // the policy entry (the threads field as the cap) and the whole call
            for threads in [0usize, 1, 3] {
                let q = GlmPreprocessing { threads, ..p.clone() };
                assert_eq!(q.patchify(&canvas, &l16), want16);
            }
            n += 1;
            tall += (hu >= 64) as usize;
        }
        eprintln!("{n} canvas shapes, {tall} with a row of blocks for each of 64 threads");
        assert!(n >= 200 && tall >= 20, "{n} shapes, {tall} with a row of blocks for each of 64 threads");
    }

    #[test]
    fn the_threads_field_is_the_cap_of_the_patchify_and_small_canvases_run_inline() {
        // the policy: one worker per 65,536 pixels up to the cap, the cap the field or the default
        let big = GlmPreprocessing { threads: 5, ..p() };
        assert_eq!(big.cap(), 5);
        assert_eq!(GlmPreprocessing { threads: 0, ..p() }.cap(), crate::resize::default_workers());
        assert_eq!(layout::layout_workers(big.cap(), 10 * layout::MIN_PIXELS_PER_WORKER), 5);
        assert_eq!(layout::layout_workers(big.cap(), 3 * layout::MIN_PIXELS_PER_WORKER), 3);
        assert_eq!(layout::layout_workers(big.cap(), layout::MIN_PIXELS_PER_WORKER), 1);
        // and the entries ask exactly that: a 600x500 image (canvas 616x504, 310,464 pixels: 4 workers' worth) and a 64x64 one (canvas 112x112)
        let (large, small) = (noise(600, 500), noise(64, 64));
        for threads in [1usize, 2, 3, 4, 64] {
            let p = GlmPreprocessing { threads, ..p() };
            p.preprocess(&large).unwrap();
            // the hash thread counts against the cap, the float32 tensor has no hash
            let pieces = 616 * 504 / layout::MIN_PIXELS_PER_WORKER;
            let workers = pieces.clamp(1, if threads > 1 { threads - 1 } else { 1 });
            assert_eq!(layout::last_plan(), Some(layout::Plan { workers, hash_aside: threads > 1 }), "threads {threads}");
            p.preprocess_f32(&large).unwrap();
            assert_eq!(
                layout::last_plan(),
                Some(layout::Plan { workers: pieces.clamp(1, threads), hash_aside: false }),
                "the f32 entry, threads {threads}"
            );
            p.preprocess(&small).unwrap();
            assert_eq!(
                layout::last_plan(),
                Some(layout::Plan { workers: 1, hash_aside: false }),
                "a thumbnail with threads {threads}"
            );
        }
        // still_patches: one worker per block of pixels (patches times pixels a patch) up to the cap
        let pre = still(3000, 196, 2, 5);
        for cap in [1usize, 2, 3, 64] {
            still_patches_with(&pre, 2, cap);
            let workers = (3000 * 196 / layout::MIN_PIXELS_PER_WORKER).clamp(1, cap);
            assert_eq!(layout::last_plan().map(|l| l.workers), Some(workers), "still_patches, cap {cap}");
        }
    }

    /// A still-image `Preprocessed` of `n` patches whose `tps` temporal slots of each channel are the same `per_slot` values.
    fn still(n: usize, per_slot: usize, tps: usize, seed: u64) -> Preprocessed {
        let first = noise_bytes(n * 3 * per_slot * 2, seed);
        let first: Vec<u16> = first.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        let mut patches = Vec::with_capacity(n * 3 * tps * per_slot);
        for patch in first.chunks_exact(3 * per_slot) {
            for channel in patch.chunks_exact(per_slot) {
                for _ in 0..tps {
                    patches.extend_from_slice(channel);
                }
            }
        }
        Preprocessed { size: (0, 0), grid: (1, n, 1), patches, patch_dim: 3 * tps * per_slot, pixel_hash: 0 }
    }

    #[test]
    fn still_patches_equals_the_oracle_on_every_thread_count() {
        let mut shapes = 0;
        for tps in [1usize, 2, 3, 4] {
            for per_slot in [1usize, 2, 7, 196, 588] {
                for n in [0usize, 1, 2, 5, 63, 64, 65, 200, 1000] {
                    let pre = still(n, per_slot, tps, (n + per_slot * 7 + tps) as u64);
                    let want = reference::still_patches(&pre, tps);
                    assert_eq!(want.len(), n * 3 * per_slot);
                    for workers in THREADS {
                        assert_eq!(
                            still_patches_exact(&pre, tps, workers),
                            want,
                            "{n} patches, {per_slot} per slot, {tps} slots, {workers} threads"
                        );
                    }
                    for cap in [0usize, 1, 2, 9] {
                        assert_eq!(still_patches_with(&pre, tps, cap), want);
                    }
                    assert_eq!(still_patches(&pre, tps), want);
                    shapes += 1;
                }
            }
        }
        eprintln!("{shapes} still-image tensors");
        assert!(shapes >= 150);
        // the real thing: a preprocessed image folds to half its size
        let pre = p().preprocess(&noise(300, 200)).unwrap();
        assert_eq!(still_patches(&pre, 2), reference::still_patches(&pre, 2));
    }

    /// The panic message of `f`, or `None` when it returns.
    fn panic_message(f: impl FnOnce() -> Vec<u16>) -> Option<String> {
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).err()?;
        Some(
            err.downcast_ref::<String>()
                .cloned()
                .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default(),
        )
    }

    #[test]
    fn still_patches_panics_on_differing_slots_with_the_serial_message_for_the_first_such_patch() {
        let (per_slot, tps) = (5usize, 3usize);
        let n = 130;
        let good = still(n, per_slot, tps, 77);
        // corrupt one (patch, channel, slot) at a time: first, middle, last patch; each channel; each repeated slot
        let mut cases = 0;
        for &patch in &[0usize, 1, 64, 65, 128, 129] {
            for c in 0..3 {
                for t in 1..tps {
                    let mut pre = good.clone();
                    let at = patch * pre.patch_dim + (c * tps + t) * per_slot + 2;
                    pre.patches[at] ^= 1;
                    let want = panic_message(|| reference::still_patches(&pre, tps)).expect("the oracle panics");
                    assert!(want.contains("temporal slots differ: not a still image"), "{want}");
                    for workers in THREADS {
                        let got = panic_message(|| still_patches_exact(&pre, tps, workers)).expect("the new code panics");
                        assert_eq!(got, want, "patch {patch} channel {c} slot {t}, {workers} threads");
                    }
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 6 * 3 * 2);
        // two bad patches: the message is the first one's (in patch order), whichever thread sees which
        let mut pre = good.clone();
        for patch in [100usize, 40] {
            let at = patch * pre.patch_dim + 7;
            pre.patches[at] ^= 0x80;
        }
        let want = panic_message(|| reference::still_patches(&pre, tps)).unwrap();
        for workers in THREADS {
            assert_eq!(panic_message(|| still_patches_exact(&pre, tps, workers)).unwrap(), want, "{workers} threads");
        }
        // the layout check comes first, as before
        let bad_dim = Preprocessed { patch_dim: 10, ..good.clone() };
        assert!(panic_message(|| still_patches(&bad_dim, tps)).unwrap().contains("patch_dim is not 3 * temporal * pixels"));
        // a slot that is the same in only some channels is still caught: a difference in the last element of the last slot
        let mut pre = good;
        let last = pre.patches.len() - 1;
        pre.patches[last] ^= 1;
        assert!(panic_message(|| still_patches_exact(&pre, tps, 4)).is_some());
    }

    #[test]
    fn still_patches_halves_the_tensor_exactly() {
        let p = p();
        let img = Rgb { width: 30, height: 20, data: (0..30 * 20 * 3).map(|i| (i * 31 % 256) as u8).collect() };
        let pre = p.preprocess(&img).unwrap();
        assert_eq!(pre.patch_dim, 1176);
        let half = still_patches(&pre, 2);
        assert_eq!(half.len(), pre.patches.len() / 2);
        // channel 0 of patch 0 is its first 196 values, channel 1 starts at 392 in the full layout and at 196 in the half
        assert_eq!(&half[..196], &pre.patches[..196]);
        assert_eq!(&half[196..392], &pre.patches[392..588]);
    }
}
