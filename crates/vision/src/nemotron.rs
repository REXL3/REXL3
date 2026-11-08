//! Image preprocessing of Nemotron 3.5 Super VL (`NemotronH_Omni_Reasoning_V3`): `NemotronH_Omni_Reasoning_V3ImageProcessor` of the pack
//! (`image_processing_nemotron_h_omni.py`), image path and video-frame path, on the CPU.
//!
//! **The processor.** An image is decoded and converted with `image.convert("RGB")` (alpha dropped, `rgb_decode::AlphaPolicy::Drop`), a single
//! tile is chosen from the size alone ([`target_patches`]: the closest patch grid, scaled down to the token budget, scaled up to
//! `min_num_patches`, both sides rounded to a multiple of the pixel-shuffle factor), and the image is resized to `(h_patches * 16, w_patches * 16)`
//! with `torch.nn.functional.interpolate(float32 [1,3,H,W] of 0..255 values, mode="bicubic", align_corners=False, antialias=True)`, only when the size
//! differs, then normalised as `(x / 255 - mean) / std` in float32. The model casts the result to bf16 (`pixel_values.to(vision_model.config.torch_dtype)`)
//! and runs the RADIO tower; pixel shuffle then folds 2 x 2 patches into one language-model token, so an image of `w_patches x h_patches` patches
//! takes `w_patches * h_patches / 4` placeholder tokens.
//!
//! **What is exact.** Everything that is an integer is the processor's, bit for bit (the tile, the token count; the Python `round` is banker's
//! rounding and is replicated, and the Python exceptions of the same inputs are errors here). The resize is a port of the float32 branch of PyTorch's
//! antialiased bicubic CPU kernel in the operation order torch 2.14.0 (aarch64 wheel, SVE128 dispatch) executes it, and was found bit-identical to it
//! on every shape tried (`tests/nemotron_preprocess.rs` states the numbers it measured). What that order is, because it is not what the source reads:
//!
//! - the scale, the support and the inverse scale are `float`; the centre is the double product of the float scale and `i + 0.5` rounded to float; the
//!   window bounds truncate `float(centre -+ support) + 0.5` (the float difference first, then the double sum); the filter argument is
//!   `float((float(float(j + xmin) - centre) + 0.5) * invscale)`;
//! - the cubic-convolution polynomial (`a = -0.5`) is evaluated in float with fused multiply-adds as the compiler contracted it; the weights are summed
//!   in float in tap order and divided by the sum (`wt /= total_w`);
//! - a pass accumulates in float in tap order: `acc = t0 * w0`, then the first `4 * floor((n - 1) / 4)` taps as `acc += round(t * w)` (the in-order
//!   reduction of a 4-lane SVE vector), then the remaining taps as one fused multiply-add each;
//! - the horizontal pass runs first, then the vertical one, and an axis whose size does not change is skipped.
//!
//! On a CPU that dispatches torch differently (AVX2, a wider SVE vector) torch's own last bits differ from these; the gate of the tests is 1e-5 on the
//! normalised values, which is about 45 float32 steps of the 0..255 values and far above that spread.
//!
//! **Attribution.** The weight and window logic re-expresses `HelperInterpBase::_compute_index_ranges_weights`,
//! `_compute_indices_min_size_weights_aa` and `cpu_upsample_generic_aa` of PyTorch's `aten/src/ATen/native/cpu/UpSampleKernel.cpp` and the
//! `cubic_convolution1/2` of `aten/src/ATen/native/UpSample.h`. PyTorch is BSD-3-Clause, Copyright (c) Facebook, Inc. and its affiliates, and the other
//! holders named in its licence; the notice and the full licence text are in `THIRD_PARTY_NOTICES.md` ("PyTorch license text"), as `docs/UPSTREAMS.md`
//! requires of a BSD-3-Clause port. The threading, the loop structure, the policy and the tests are REXL3's own.
//!
//! **Output.** [`Preprocessed::pixels`] is the processor's `pixel_values[0]`: float32 `[3][H][W]`. [`Preprocessed::bf16_bits`] is what the model feeds
//! the tower. The [`ImagePreprocessor`] impl at the end carries those bf16 bits (channel planes, not patch-major) in `gb_vision::Preprocessed::patches`.

use crate::preprocessor::{ImagePlan, ImagePreprocessor};
use crate::resample::par_rows;
use crate::resize::{default_workers, resize_workers};
use crate::rgb_decode::{decode_rgb, AlphaPolicy};
use crate::{Rgb, VisionError};
use serde_json::Value;
use std::fmt;

/// Pixels per patch side: the tower's patch embedding.
pub const PATCH_SIZE: usize = 16;
/// Pixel-shuffle factor (`downsample_ratio` 0.5): `MERGE x MERGE` patches make one language-model token.
pub const MERGE: usize = 2;
/// Most patches the tower's workspace is sized for (`cuda::nemotron_vit::MAX_PATCHES`): 3,328 tokens.
pub const TOWER_MAX_PATCHES: usize = 13_312;
/// Most placeholder tokens one image can take (`TOWER_MAX_PATCHES / MERGE^2`; `VisionSpec::max_image_tokens`).
pub const MAX_IMAGE_TOKENS: usize = TOWER_MAX_PATCHES / (MERGE * MERGE);
/// `max_model_len - RESERVED_TOKENS` is the processor's token allowance (`num_tokens_available = self.max_model_len - 4`).
const RESERVED_TOKENS: i64 = 4;

/// A `preprocessor_config.json` the preprocessing cannot honour; the message names the key.
#[derive(Debug, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "preprocessor_config.json: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

/// The image processor's parameters (`preprocessor_config.json` of the pack; the defaults are those of the class).
#[derive(Clone, Debug, PartialEq)]
pub struct NemotronPreprocessing {
    /// Pixels per patch side. The tower implements 16.
    pub patch_size: usize,
    /// `int(round(1 / downsample_ratio))`: both patch-grid sides are made multiples of it. The tower implements 2.
    pub downsample_factor: usize,
    pub min_num_patches: usize,
    pub max_num_patches: usize,
    /// The processor's own `max_model_len` (16,384 in the pack): it bounds the patches of one image, not the server's context window.
    pub max_model_len: usize,
    pub video_target_num_patches: usize,
    pub video_maintain_aspect_ratio: bool,
    pub norm_mean: [f32; 3],
    pub norm_std: [f32; 3],
    /// Worker threads of the resize and the normalisation: 0 picks `resize::default_workers()` (`REXL3_VISION_THREADS`), a positive value is the most
    /// used. The output does not depend on it.
    pub threads: usize,
}

impl Default for NemotronPreprocessing {
    /// The values of the published pack's `preprocessor_config.json`.
    fn default() -> Self {
        Self {
            patch_size: PATCH_SIZE,
            downsample_factor: MERGE,
            min_num_patches: 1024,
            max_num_patches: TOWER_MAX_PATCHES,
            max_model_len: 16_384,
            video_target_num_patches: 1024,
            video_maintain_aspect_ratio: true,
            norm_mean: [0.48145466f64 as f32, 0.4578275f64 as f32, 0.40821073f64 as f32],
            norm_std: [0.26862954f64 as f32, 0.26130258f64 as f32, 0.27577711f64 as f32],
            threads: 0,
        }
    }
}

/// Python's `round` (to nearest, ties to even) of a float, as an integer.
fn py_round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

fn bad_size(m: impl Into<String>) -> VisionError {
    VisionError::Unsupported(m.into())
}

/// `tokens = w_patches * h_patches / factor^2` (`(wp * hp) // (self._downsample_factor ** 2)`).
pub fn num_tokens(w_patches: usize, h_patches: usize, cfg: &NemotronPreprocessing) -> usize {
    w_patches * h_patches / (cfg.downsample_factor * cfg.downsample_factor)
}

/// The patch budget of one image when `n_images` images are processed together (the budget rule of `_preprocess`):
/// `budget = (max_model_len - 4) * factor^2`, raised to `min_num_patches * n_images`, then
/// `max(min(budget, max_num_patches), min_num_patches)` for every image. (The `max_num_patches` of the pack is always positive; `0` meaning
/// "unbounded" is refused when the config is read.)
pub fn per_image_budget(cfg: &NemotronPreprocessing, n_images: usize) -> usize {
    let factor = cfg.downsample_factor as i64;
    let budget = (cfg.max_model_len as i64 - RESERVED_TOKENS) * factor * factor;
    let budget = budget.max(cfg.min_num_patches as i64 * n_images as i64);
    budget.min(cfg.max_num_patches as i64).max(cfg.min_num_patches as i64) as usize
}

/// `_compute_target_patches`: the `(w_patches, h_patches)` tile for a `width x height` image when `tokens_available` patches may be used (the
/// processor calls its parameter `tokens_available`; it counts patches, before the pixel shuffle).
///
/// Errors where Python raises: a zero side (`ZeroDivisionError` in `sqrt(tokens_available / patches)`), a scaled-down tile with a zero side when the
/// scale-up branch divides by it, and a tile with a zero side after the rounding (the processor would then ask torch for an empty output).
pub fn target_patches(
    width: usize,
    height: usize,
    tokens_available: usize,
    cfg: &NemotronPreprocessing,
) -> Result<(usize, usize), VisionError> {
    let ps = cfg.patch_size as f64;
    // `round(x + 0.5)`: banker's rounding, so for a multiple of 16 an even quotient rounds down and an odd one up (the comment in the reference says
    // "x for integer", which holds only for even x)
    let closest_h = py_round(height as f64 / ps + 0.5);
    let closest_w = py_round(width as f64 / ps + 0.5);
    let patches = closest_h * closest_w;
    if patches <= 0 {
        return Err(bad_size(format!("{width}x{height}: no patches")));
    }
    let available = tokens_available as i64;
    // downscale to fit the budget
    let factor = (available as f64 / patches as f64).sqrt().min(1.0);
    let mut target_h = (factor * closest_h as f64).floor() as i64;
    let mut target_w = (factor * closest_w as f64).floor() as i64;
    // scale up to the per-image minimum
    let min = cfg.min_num_patches as i64;
    if available > min && target_h * target_w < min {
        if target_h * target_w == 0 {
            return Err(bad_size(format!(
                "{width}x{height}: the tile collapses to nothing at a budget of {tokens_available} patches"
            )));
        }
        let up = (min as f64 / (target_h * target_w) as f64).sqrt();
        target_h = (up * target_h as f64).ceil() as i64;
        target_w = (up * target_w as f64).ceil() as i64;
    }
    // both sides to a multiple of the pixel-shuffle factor, up when the budget allows, else down (not below the factor)
    let divisor = cfg.downsample_factor as i64;
    let rem_h = target_h % divisor;
    if rem_h != 0 {
        let inc = divisor - rem_h;
        if (target_h + inc) * target_w <= available {
            target_h += inc;
        } else {
            target_h = divisor.max(target_h - rem_h);
        }
    }
    let rem_w = target_w % divisor;
    if rem_w != 0 {
        let inc = divisor - rem_w;
        if target_h * (target_w + inc) <= available {
            target_w += inc;
        } else {
            target_w = divisor.max(target_w - rem_w);
        }
    }
    if target_h <= 0 || target_w <= 0 {
        return Err(bad_size(format!("{width}x{height}: the tile would be empty ({target_w}x{target_h} patches)")));
    }
    Ok((target_w as usize, target_h as usize))
}

/// `_compute_target_patches_video`: the aspect-preserving `(w_patches, h_patches)` tile of a video frame, near `video_target_num_patches`.
/// Errors where Python raises (a zero width: `ZeroDivisionError`).
pub fn target_patches_video(width: usize, height: usize, cfg: &NemotronPreprocessing) -> Result<(usize, usize), VisionError> {
    let target = cfg.video_target_num_patches as i64;
    let divisor = cfg.downsample_factor as i64;
    let (mut ph, mut pw);
    if cfg.video_maintain_aspect_ratio {
        if width == 0 {
            return Err(bad_size(format!("{width}x{height}: a frame needs a width")));
        }
        let aspect = width as f64 / height.max(1) as f64;
        ph = py_round((target as f64 / aspect).sqrt()).max(1);
        pw = py_round((target as f64 * aspect).sqrt()).max(1);
        if divisor > 1 {
            let (rem_h, rem_w) = (ph % divisor, pw % divisor);
            let ph_up = ph + if rem_h != 0 { divisor - rem_h } else { 0 };
            let ph_down = ph - rem_h;
            let pw_up = pw + if rem_w != 0 { divisor - rem_w } else { 0 };
            let pw_down = pw - rem_w;
            if ph_up * pw_up <= target {
                ph = ph_up;
                pw = pw_up;
            } else {
                ph = divisor.max(ph_down);
                pw = divisor.max(pw_down);
            }
        }
    } else {
        let side = (target as f64).sqrt() as i64;
        let side = divisor.max(side / divisor * divisor);
        ph = side;
        pw = side;
    }
    Ok((pw as usize, ph as usize))
}

/// The tile chosen for an image: what the prompt and the scheduler need before any pixel is touched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TilePlan {
    pub w_patches: usize,
    pub h_patches: usize,
    /// Placeholder tokens: `w_patches * h_patches / 4`.
    pub tokens: usize,
}

impl TilePlan {
    /// The canvas the tower sees, `(width, height)` in pixels.
    pub fn canvas(&self) -> (usize, usize) {
        (self.w_patches * PATCH_SIZE, self.h_patches * PATCH_SIZE)
    }
    /// The merged grid `(rows, columns)` after the pixel shuffle; `tokens == rows * columns`.
    pub fn merged(&self) -> (usize, usize) {
        (self.h_patches / MERGE, self.w_patches / MERGE)
    }
}

/// An image through the processor: float32 `pixel_values[0]`, the tile and the content hash of the canvas.
#[derive(Clone, Debug, PartialEq)]
pub struct Preprocessed {
    /// `[3][H][W]` (channel planes), `(x / 255 - mean) / std`; `H = h_patches * 16`, `W = w_patches * 16`.
    pub pixels: Vec<f32>,
    pub h_patches: usize,
    pub w_patches: usize,
    /// `w_patches * h_patches / 4`.
    pub tokens: usize,
    /// FNV-1a over the canvas size and the bytes of `pixels` (see [`canvas_hash`]): equal for equal model inputs.
    pub pixel_hash: u64,
}

impl Preprocessed {
    /// Canvas `(width, height)` in pixels.
    pub fn canvas(&self) -> (usize, usize) {
        (self.w_patches * PATCH_SIZE, self.h_patches * PATCH_SIZE)
    }
    pub fn n_patches(&self) -> usize {
        self.w_patches * self.h_patches
    }
    /// The tower's input as the model makes it: `pixels` rounded to bf16, round to nearest even (`pixel_values.to(torch.bfloat16)`).
    pub fn bf16_bits(&self) -> Vec<u16> {
        self.pixels.iter().map(|&v| f32_to_bf16_bits(v)).collect()
    }
}

/// float32 -> bfloat16 bits, round to nearest even (what `.to(torch.bfloat16)` does); NaN stays a NaN.
pub fn f32_to_bf16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    if v.is_nan() {
        return ((x >> 16) as u16) | 0x0040;
    }
    let rounding = 0x7fff + ((x >> 16) & 1);
    (x.wrapping_add(rounding) >> 16) as u16
}

impl NemotronPreprocessing {
    /// Read a `preprocessor_config.json` (or a `processor_config.json`, whose `image_processor` block is used). A missing key takes the default of the
    /// class, as transformers does; a key the tower cannot honour is an error that names it.
    pub fn from_json(root: &Value) -> Result<Self, ConfigError> {
        let cfg = root.get("image_processor").filter(|v| v.is_object()).unwrap_or(root);
        if !cfg.is_object() {
            return Err(ConfigError("not a JSON object".into()));
        }
        if let Some(t) = cfg.get("image_processor_type") {
            if t.as_str() != Some("NemotronH_Omni_Reasoning_V3ImageProcessor") {
                return Err(ConfigError(format!(
                    "image_processor_type is {t}, expected \"NemotronH_Omni_Reasoning_V3ImageProcessor\""
                )));
            }
        }
        let mut p = Self::default();
        let int = |key: &str| -> Result<Option<usize>, ConfigError> {
            match cfg.get(key) {
                None => Ok(None),
                Some(v) => v
                    .as_u64()
                    .filter(|&n| n > 0 && n < 1 << 32)
                    .map(|n| Some(n as usize))
                    .ok_or_else(|| ConfigError(format!("{key} must be a positive integer, got {v}"))),
            }
        };
        if let Some(n) = int("patch_size")? {
            if n != PATCH_SIZE {
                return Err(ConfigError(format!("patch_size is {n}; the RADIO tower implements {PATCH_SIZE}")));
            }
        }
        if let Some(v) = cfg.get("downsample_ratio") {
            let r = v
                .as_f64()
                .filter(|r| r.is_finite() && *r > 0.0)
                .ok_or_else(|| ConfigError(format!("downsample_ratio must be a positive number, got {v}")))?;
            // `int(round(1.0 / downsample_ratio))`
            let factor = py_round(1.0 / r);
            if factor != MERGE as i64 {
                return Err(ConfigError(format!(
                    "downsample_ratio {r} is a pixel-shuffle factor of {factor}; the tower implements {MERGE} (0.5)"
                )));
            }
        }
        if let Some(v) = cfg.get("_downsample_factor") {
            if v.as_u64() != Some(MERGE as u64) {
                return Err(ConfigError(format!("_downsample_factor is {v}; the tower implements {MERGE}")));
            }
        }
        if let Some(n) = int("min_num_patches")? {
            p.min_num_patches = n;
        }
        match cfg.get("max_num_patches") {
            None => {}
            Some(v) => {
                // the processor reads a non-positive value as "unbounded", which the tower's workspace cannot take
                let n = v.as_i64().filter(|&n| n > 0 && n as usize <= TOWER_MAX_PATCHES).ok_or_else(|| {
                    ConfigError(format!("max_num_patches must be between 1 and {TOWER_MAX_PATCHES} (the tower's limit), got {v}"))
                })?;
                p.max_num_patches = n as usize;
            }
        }
        if p.min_num_patches > p.max_num_patches {
            return Err(ConfigError(format!(
                "min_num_patches {} exceeds max_num_patches {}",
                p.min_num_patches, p.max_num_patches
            )));
        }
        if let Some(n) = int("max_model_len")? {
            p.max_model_len = n;
        }
        if let Some(n) = int("video_target_num_patches")? {
            if n > TOWER_MAX_PATCHES {
                return Err(ConfigError(format!("video_target_num_patches {n} exceeds the tower's {TOWER_MAX_PATCHES} patches")));
            }
            p.video_target_num_patches = n;
        }
        if let Some(v) = cfg.get("video_maintain_aspect_ratio") {
            p.video_maintain_aspect_ratio =
                v.as_bool().ok_or_else(|| ConfigError(format!("video_maintain_aspect_ratio must be a boolean, got {v}")))?;
        }
        for (key, slot, positive) in [("norm_mean", &mut p.norm_mean, false), ("norm_std", &mut p.norm_std, true)] {
            let v = cfg.get(key).ok_or_else(|| ConfigError(format!("{key} is missing (the processor has no default)")))?;
            let vals: Option<Vec<f64>> =
                v.as_array().filter(|a| a.len() == 3).and_then(|a| a.iter().map(Value::as_f64).collect());
            let vals = vals.ok_or_else(|| ConfigError(format!("{key} must be 3 numbers, got {v}")))?;
            for (s, x) in slot.iter_mut().zip(vals) {
                // `torch.tensor([floats])` is float32
                let f = x as f32;
                if !f.is_finite() || (positive && f <= 0.0) {
                    return Err(ConfigError(format!(
                        "{key} must be finite{}, got {v}",
                        if positive { " and positive" } else { "" }
                    )));
                }
                *s = f;
            }
        }
        Ok(p)
    }

    /// [`NemotronPreprocessing::from_json`] of the text of a config file.
    pub fn from_json_str(text: &str) -> Result<Self, ConfigError> {
        let v: Value = serde_json::from_str(text).map_err(|e| ConfigError(format!("not valid JSON: {e}")))?;
        Self::from_json(&v)
    }

    /// The `preprocessor_config.json` in a model folder, else the `image_processor` block of its `processor_config.json`.
    pub fn from_dir(dir: &std::path::Path) -> Result<Self, ConfigError> {
        for name in ["preprocessor_config.json", "processor_config.json"] {
            if let Ok(text) = std::fs::read_to_string(dir.join(name)) {
                return Self::from_json_str(&text).map_err(|e| ConfigError(format!("{name}: {}", e.0)));
            }
        }
        Err(ConfigError(format!("{} has neither preprocessor_config.json nor processor_config.json", dir.display())))
    }

    /// `max_num_patches` as a token budget: the most placeholder tokens one image can take under this config.
    pub fn max_tokens(&self) -> usize {
        self.max_num_patches / (self.downsample_factor * self.downsample_factor)
    }

    /// Lower the per-image token limit (the server's `--max-image-tokens`) the way the processor takes it, as `max_num_patches = tokens * factor^2`
    /// (a keyword of the processor): a larger image is resized to fit instead of being refused, and the model input of an image that fits is
    /// unchanged. Errors above the tower's limit and below the minimum tile.
    pub fn with_max_image_tokens(mut self, tokens: usize) -> Result<Self, ConfigError> {
        let patches = tokens.saturating_mul(self.downsample_factor * self.downsample_factor);
        if patches > TOWER_MAX_PATCHES {
            return Err(ConfigError(format!("max image tokens {tokens} exceeds the tower's {MAX_IMAGE_TOKENS}")));
        }
        if patches < self.min_num_patches {
            return Err(ConfigError(format!(
                "max image tokens {tokens} is below the {} tokens of the minimum tile (min_num_patches {})",
                self.min_num_patches / (self.downsample_factor * self.downsample_factor),
                self.min_num_patches
            )));
        }
        self.max_num_patches = patches;
        Ok(self)
    }

    /// The tile, the canvas and the token count of a `width x height` image when `n_images` are processed together, from the size alone.
    pub fn plan_for(&self, width: usize, height: usize, n_images: usize) -> Result<TilePlan, VisionError> {
        if width == 0 || height == 0 {
            return Err(bad_size(format!("{width}x{height}: an image needs both sides")));
        }
        let (w_patches, h_patches) = target_patches(width, height, per_image_budget(self, n_images.max(1)), self)?;
        if w_patches * h_patches > TOWER_MAX_PATCHES {
            return Err(bad_size(format!(
                "{width}x{height}: a tile of {w_patches}x{h_patches} patches exceeds the tower's {TOWER_MAX_PATCHES}"
            )));
        }
        Ok(TilePlan { w_patches, h_patches, tokens: num_tokens(w_patches, h_patches, self) })
    }

    /// The tile of a video frame (`_is_video_mode`): [`target_patches_video`] with the token count.
    pub fn plan_video(&self, width: usize, height: usize) -> Result<TilePlan, VisionError> {
        let (w_patches, h_patches) = target_patches_video(width, height, self)?;
        if w_patches * h_patches > TOWER_MAX_PATCHES {
            return Err(bad_size(format!(
                "{width}x{height}: a tile of {w_patches}x{h_patches} patches exceeds the tower's {TOWER_MAX_PATCHES}"
            )));
        }
        Ok(TilePlan { w_patches, h_patches, tokens: num_tokens(w_patches, h_patches, self) })
    }

    /// A stable identity of everything that changes the model input: the family, every numeric parameter (the floats as their bits), the resize kernel.
    /// Part of an image's cache key (`sched::image_content_hash`); bump the version suffix when the semantics change.
    pub fn identity(&self) -> String {
        let bits = |a: &[f32; 3]| a.iter().map(|v| format!("{:08x}", v.to_bits())).collect::<Vec<_>>().join(",");
        format!(
            "nemotron-h-omni-v1:p{}f{}:np{}-{}:mml{}:vid{}{}:mean{}:std{}:bicubic-aa-f32",
            self.patch_size,
            self.downsample_factor,
            self.min_num_patches,
            self.max_num_patches,
            self.max_model_len,
            self.video_target_num_patches,
            if self.video_maintain_aspect_ratio { "a" } else { "s" },
            bits(&self.norm_mean),
            bits(&self.norm_std)
        )
    }

    fn cap(&self) -> usize {
        if self.threads == 0 {
            default_workers()
        } else {
            self.threads
        }
    }

    /// Resize (when the tile differs from the image), normalise and hash one image, as the processor does for a batch of `n_images` images (the
    /// budget rule depends on the count when `max_model_len` is small).
    pub fn preprocess_for(&self, img: &Rgb, n_images: usize) -> Result<Preprocessed, VisionError> {
        let plan = self.plan_for(img.width, img.height, n_images)?;
        self.preprocess_to(img, plan)
    }

    /// [`NemotronPreprocessing::preprocess_for`] for one image.
    pub fn preprocess(&self, img: &Rgb) -> Result<Preprocessed, VisionError> {
        self.preprocess_for(img, 1)
    }

    /// A video frame through the video rule (`_is_video_mode`), for the frame path.
    pub fn preprocess_video_frame(&self, img: &Rgb) -> Result<Preprocessed, VisionError> {
        let plan = self.plan_video(img.width, img.height)?;
        self.preprocess_to(img, plan)
    }

    fn preprocess_to(&self, img: &Rgb, plan: TilePlan) -> Result<Preprocessed, VisionError> {
        let (sw, sh) = (img.width, img.height);
        if sw == 0 || sh == 0 || img.data.len() != sw * sh * 3 {
            return Err(VisionError::Decode(format!("{sw}x{sh}: {} bytes are not an RGB image", img.data.len())));
        }
        let (tw, th) = plan.canvas();
        let cap = self.cap();
        // 0..255 values as float32, resized when the size differs
        let resized;
        let hwc: &[f32];
        if (tw, th) != (sw, sh) {
            resized = resize_bicubic_aa_f32(&img.data, sw, sh, tw, th, cap);
            hwc = &resized;
        } else {
            resized = img.data.iter().map(|&b| b as f32).collect::<Vec<f32>>();
            hwc = &resized;
        }
        // `(t / 255.0 - mean) / std`: three float32 operations, each correctly rounded
        let mut pixels = vec![0f32; 3 * tw * th];
        let workers = resize_workers(cap, pixels.len() * 3);
        par_rows(&mut pixels, tw, workers, |r, row| {
            let (c, y) = (r / th, r % th);
            let (mean, std) = (self.norm_mean[c], self.norm_std[c]);
            let src = &hwc[y * tw * 3..(y + 1) * tw * 3];
            for (o, px) in row.iter_mut().zip(src.chunks_exact(3)) {
                *o = (px[c] / 255.0 - mean) / std;
            }
        });
        let pixel_hash = canvas_hash(tw, th, &pixels, cap);
        Ok(Preprocessed { pixels, h_patches: plan.h_patches, w_patches: plan.w_patches, tokens: plan.tokens, pixel_hash })
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

fn fnv(h: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(FNV_PRIME))
}

fn fnv_f32(h: u64, plane: &[f32]) -> u64 {
    plane.iter().fold(h, |h, v| fnv(h, &v.to_le_bytes()))
}

/// The content hash of a `[3][H][W]` canvas: FNV-1a over the width and the height (8 little-endian bytes each), then over the three channel planes'
/// own FNV-1a hashes (each over the little-endian bytes of its values, from the standard offset), so the planes hash on three threads (`threads` > 1
/// and a canvas of 64 Ki values or more) and the value does not depend on how many ran.
pub fn canvas_hash(width: usize, height: usize, pixels: &[f32], threads: usize) -> u64 {
    let plane = width * height;
    assert_eq!(pixels.len(), 3 * plane, "{width}x{height} canvas");
    let planes: [u64; 3] = if threads > 1 && plane >= 1 << 16 {
        let mut out = [0u64; 3];
        std::thread::scope(|s| {
            let handles: Vec<_> = pixels.chunks_exact(plane).map(|p| s.spawn(move || fnv_f32(FNV_OFFSET, p))).collect();
            for (o, h) in out.iter_mut().zip(handles) {
                *o = h.join().expect("hash thread");
            }
        });
        out
    } else {
        let mut out = [0u64; 3];
        for (o, p) in out.iter_mut().zip(pixels.chunks_exact(plane.max(1))) {
            *o = fnv_f32(FNV_OFFSET, p);
        }
        out
    };
    let mut h = fnv(FNV_OFFSET, &(width as u64).to_le_bytes());
    h = fnv(h, &(height as u64).to_le_bytes());
    for p in planes {
        h = fnv(h, &p.to_le_bytes());
    }
    h
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// The resize: float32 antialiased bicubic, torch's order of operations (see the module documentation).

/// `cubic_convolution1/2` of ATen with `a = -0.5` in float, fused as the compiler contracted them (`aa_filter<float>`).
fn aa_filter(x: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        // ((A + 2) * x - (A + 3)) * x * x + 1
        let t = 1.5f32.mul_add(x, -2.5);
        (t * x).mul_add(x, 1.0)
    } else if x < 2.0 {
        // ((A * x - 5 * A) * x + 8 * A) * x - 4 * A
        let t = (-0.5f32).mul_add(x, 2.5);
        let t = t.mul_add(x, -4.0);
        t.mul_add(x, 2.0)
    } else {
        0.0
    }
}

/// One axis of the resize: per output pixel the first source index, the tap count and the float weights (`ksize` slots a pixel).
struct Axis {
    ksize: usize,
    xmin: Vec<usize>,
    xsize: Vec<usize>,
    w: Vec<f32>,
}

/// The weights of an axis resized from `in_size` to `out_size` (both at least 1, different): `_compute_indices_min_size_weights_aa<float>`.
fn axis(in_size: usize, out_size: usize) -> Axis {
    let scale = in_size as f32 / out_size as f32;
    let (support, invscale) =
        if scale >= 1.0 { ((scale as f64 * 2.0) as f32, (1.0f64 / scale as f64) as f32) } else { (2.0f32, 1.0f32) };
    let ksize = support.ceil() as usize * 2 + 1;
    let mut xmin = Vec::with_capacity(out_size);
    let mut xsize = Vec::with_capacity(out_size);
    let mut w = vec![0f32; out_size * ksize];
    for i in 0..out_size {
        // `scalar_t center = scale * (i + 0.5 - align_corners_delta)`: the product is taken in double, then stored as float
        let center = (scale as f64 * (i as f64 + 0.5)) as f32;
        // `static_cast<int64_t>(center - support + 0.5 + delta)`: the float difference, then a double sum, truncated toward zero
        let lo = ((center - support) as f64 + 0.5) as i64;
        let hi = ((center + support) as f64 + 0.5) as i64;
        let first = lo.max(0);
        let size = (hi.min(in_size as i64) - first).clamp(0, ksize as i64) as usize;
        let row = &mut w[i * ksize..(i + 1) * ksize];
        let mut total = 0f32;
        for (j, slot) in row.iter_mut().take(size).enumerate() {
            let d = (j as i64 + first) as f32 - center;
            let x = ((d as f64 + 0.5) * invscale as f64) as f32;
            let v = aa_filter(x);
            *slot = v;
            total += v;
        }
        if total != 0.0 {
            for slot in row.iter_mut().take(size) {
                *slot /= total;
            }
        }
        xmin.push(first as usize);
        xsize.push(size);
    }
    Axis { ksize, xmin, xsize, w }
}

/// How many leading taps (after the first) a pass adds as separately rounded products: `4 * floor((n - 1) / 4)`. The rest are fused.
#[inline]
fn plain_taps(n: usize) -> usize {
    n.saturating_sub(1) / 4 * 4
}

/// The horizontal pass over one source row (`sw` pixels of three u8 samples): `dw` pixels of three floats.
fn horizontal_row(row: &[u8], hz: &Axis, dst: &mut [f32]) {
    for (xx, d) in dst.chunks_exact_mut(3).enumerate() {
        let (first, n) = (hz.xmin[xx], hz.xsize[xx]);
        let k = &hz.w[xx * hz.ksize..xx * hz.ksize + n];
        let px = &row[first * 3..(first + n) * 3];
        if n == 0 {
            d.fill(0.0);
            continue;
        }
        let plain = plain_taps(n);
        let mut acc = [px[0] as f32 * k[0], px[1] as f32 * k[0], px[2] as f32 * k[0]];
        for j in 1..=plain {
            let (p, w) = (&px[j * 3..j * 3 + 3], k[j]);
            acc[0] += p[0] as f32 * w;
            acc[1] += p[1] as f32 * w;
            acc[2] += p[2] as f32 * w;
        }
        for j in plain + 1..n {
            let (p, w) = (&px[j * 3..j * 3 + 3], k[j]);
            acc[0] = (p[0] as f32).mul_add(w, acc[0]);
            acc[1] = (p[1] as f32).mul_add(w, acc[1]);
            acc[2] = (p[2] as f32).mul_add(w, acc[2]);
        }
        d.copy_from_slice(&acc);
    }
}

/// A sample of a source row: a byte (the width is kept, so the vertical pass reads the original) or a float (the horizontal pass's output).
trait Sample: Copy + Sync {
    fn value(self) -> f32;
}
impl Sample for u8 {
    #[inline(always)]
    fn value(self) -> f32 {
        self as f32
    }
}
impl Sample for f32 {
    #[inline(always)]
    fn value(self) -> f32 {
        self
    }
}

/// One output row of the vertical pass over `src` (rows of `line` samples): the taps are the outer loop and the columns the inner one; every column
/// is the same sequence of float operations `horizontal_row` makes for its pixel (the first product, the plain taps, the fused tail).
fn vertical_row<T: Sample>(src: &[T], line: usize, vt: &Axis, yy: usize, dst: &mut [f32]) {
    let (first, n) = (vt.xmin[yy], vt.xsize[yy]);
    if n == 0 {
        dst.fill(0.0);
        return;
    }
    let k = &vt.w[yy * vt.ksize..yy * vt.ksize + n];
    let plain = plain_taps(n);
    let row = |j: usize| &src[(first + j) * line..(first + j + 1) * line];
    for (d, &t) in dst.iter_mut().zip(row(0)) {
        *d = t.value() * k[0];
    }
    for (j, &w) in k.iter().enumerate().skip(1) {
        if j <= plain {
            for (d, &t) in dst.iter_mut().zip(row(j)) {
                *d += t.value() * w;
            }
        } else {
            for (d, &t) in dst.iter_mut().zip(row(j)) {
                *d = t.value().mul_add(w, *d);
            }
        }
    }
}

/// Multiply-accumulates of the passes (three channels): the unit `resize::resize_workers` divides into threads.
fn work(sh: usize, dw: usize, hz: Option<&Axis>, vt: Option<&Axis>) -> usize {
    let taps = |a: &Axis| a.xsize.iter().sum::<usize>();
    3 * (hz.map_or(0, |a| sh * taps(a)) + vt.map_or(0, |a| dw * taps(a)))
}

/// `torch.nn.functional.interpolate(x, size=(dh, dw), mode="bicubic", align_corners=False, antialias=True)` of a float32 `[1, 3, sh, sw]` tensor
/// of 0..255 values given as interleaved bytes: the result as interleaved floats, `[dh][dw][3]`. At least one of the two sizes must change (the
/// processor does not call the kernel otherwise); an axis that keeps its size is not resampled.
pub fn resize_bicubic_aa_f32(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, max_workers: usize) -> Vec<f32> {
    assert!(sw >= 1 && sh >= 1 && dw >= 1 && dh >= 1 && src.len() == sw * sh * 3, "{sw}x{sh} -> {dw}x{dh}, {} bytes", src.len());
    let hz = (dw != sw).then(|| axis(sw, dw));
    let vt = (dh != sh).then(|| axis(sh, dh));
    let workers = resize_workers(max_workers, work(sh, dw, hz.as_ref(), vt.as_ref()));
    // horizontal pass first: every source row, floats
    let mid: Option<Vec<f32>> = hz.as_ref().map(|hz| {
        let mut mid = vec![0f32; dw * sh * 3];
        par_rows(&mut mid, dw * 3, workers, |y, dst| horizontal_row(&src[y * sw * 3..(y + 1) * sw * 3], hz, dst));
        mid
    });
    let line = dw * 3;
    let Some(vt) = vt.as_ref() else {
        // only the width changed
        return mid.expect("one axis changes");
    };
    let mut out = vec![0f32; line * dh];
    match &mid {
        Some(mid) => par_rows(&mut out, line, workers, |yy, dst| vertical_row(mid, line, vt, yy, dst)),
        None => par_rows(&mut out, line, workers, |yy, dst| vertical_row(src, line, vt, yy, dst)),
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// The seam to the server: `ImagePreprocessor`.

impl ImagePreprocessor for NemotronPreprocessing {
    fn family(&self) -> &'static str {
        "nemotron-h-omni"
    }

    fn id(&self) -> String {
        self.identity()
    }

    /// `image.convert("RGB")`: the alpha channel is dropped (PNG; a JPEG goes through `decode::decode`, which applies its EXIF orientation).
    fn decode(&self, bytes: &[u8]) -> Result<Rgb, VisionError> {
        decode_rgb(bytes, AlphaPolicy::Drop)
    }

    /// The tile of an image processed alone (`n_images = 1`; with the pack's `max_model_len` the count does not change the tile).
    fn plan(&self, width: usize, height: usize) -> Result<ImagePlan, VisionError> {
        let t = self.plan_for(width, height, 1)?;
        Ok(ImagePlan { canvas: t.canvas(), grid: (1, t.h_patches, t.w_patches), tokens: t.tokens })
    }

    /// `crate::Preprocessed` with the tower's input as bf16 bits in channel planes (`[3][H][W]`, not patch-major): `patches.len() == 3 * H * W`,
    /// `patch_dim == 3 * 16 * 16`, `grid == (1, h_patches, w_patches)`. The float32 tensor of the processor is [`NemotronPreprocessing::preprocess`].
    fn preprocess(&self, img: &Rgb) -> Result<crate::Preprocessed, VisionError> {
        let p = NemotronPreprocessing::preprocess(self, img)?;
        Ok(crate::Preprocessed {
            size: p.canvas(),
            grid: (1, p.h_patches, p.w_patches),
            patches: p.bf16_bits(),
            patch_dim: 3 * PATCH_SIZE * PATCH_SIZE,
            pixel_hash: p.pixel_hash,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> NemotronPreprocessing {
        NemotronPreprocessing::default()
    }

    /// A deterministic byte stream.
    fn noise(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn python_round_is_bankers_so_even_multiples_of_16_round_down() {
        // round(h / 16 + 0.5): 512 -> round(32.5) = 32 (even), 528 -> round(33.5) = 34 (even), 48 -> round(3.5) = 4, 32 -> round(2.5) = 2
        let c = cfg();
        for (side, patches) in [(16usize, 2usize), (32, 2), (48, 4), (64, 4), (512, 32), (528, 34), (511, 32), (513, 34)] {
            // a square image at a budget large enough to leave it alone and small enough not to scale it up (min patches 1 here)
            let mut c1 = c.clone();
            c1.min_num_patches = 4;
            c1.max_num_patches = TOWER_MAX_PATCHES;
            let (w, h) = target_patches(side, side, 13_312, &c1).unwrap();
            // both sides are then rounded to a multiple of 2
            assert_eq!((w, h), (patches, patches), "side {side}");
        }
    }

    #[test]
    fn tiles_of_known_images() {
        let c = cfg();
        // (w, h) -> (w_patches, h_patches), computed with the HF processor (see the fixture table for the exhaustive version)
        for ((w, h), want) in [((512, 512), (32, 32)), ((64, 64), (32, 32)), ((6000, 4000), (140, 94)), ((257, 255), (34, 32))] {
            let t = c.plan_for(w, h, 1).unwrap();
            assert_eq!((t.w_patches, t.h_patches), want, "{w}x{h}");
            assert_eq!(t.tokens, want.0 * want.1 / 4);
        }
    }

    #[test]
    fn the_budget_rule() {
        let mut c = cfg();
        // the pack: (16384 - 4) * 4 = 65520, capped at 13312 whatever the count
        assert_eq!((1..=64).map(|n| per_image_budget(&c, n)).collect::<std::collections::BTreeSet<_>>(), [13_312].into());
        // a small window: the budget is the window's, and rises with the image count to the minimum per image
        c.max_model_len = 1028;
        assert_eq!(per_image_budget(&c, 1), 4096);
        assert_eq!(per_image_budget(&c, 5), 5120);
        c.max_model_len = 260;
        assert_eq!((per_image_budget(&c, 1), per_image_budget(&c, 2)), (1024, 2048));
        c.max_model_len = 2;
        assert_eq!(per_image_budget(&c, 1), 1024, "a window below the reserve still gives the minimum");
    }

    #[test]
    fn the_scale_up_runs_only_above_the_minimum_budget() {
        let mut c = cfg();
        c.max_model_len = 260; // 1024 patches: the minimum itself
                               // a 64x64 image is 4x4 patches, and stays that small (the rounding keeps it even)
        assert_eq!(c.plan_for(64, 64, 1).unwrap().tokens, 4);
        c.max_model_len = 16_384;
        assert_eq!(c.plan_for(64, 64, 1).unwrap().tokens, 256);
    }

    #[test]
    fn what_python_raises_is_an_error() {
        let c = cfg();
        assert!(target_patches(0, 10, 13_312, &c).is_err());
        assert!(target_patches(10, 0, 13_312, &c).is_err());
        // 1 x 100000: 1 x 6250 patches over a budget of 1024 patches floors the height to 0, and the scale-up divides by 0
        assert!(target_patches(100_000, 1, 1025, &c).is_err());
        // at exactly the minimum there is no scale-up: a tile with a zero side is returned by the processor and torch then refuses it
        assert!(target_patches(100_000, 1, 1024, &c).is_err());
        assert!(target_patches_video(0, 10, &c).is_err());
    }

    #[test]
    fn video_tiles() {
        let c = cfg();
        // 1920x1080: aspect 16/9, ph = round(sqrt(1024 * 9 / 16)) = 24, pw = round(sqrt(1024 * 16 / 9)) = 43 -> up 44: 24 * 44 = 1056 > 1024, so down: 24 x 42
        assert_eq!(target_patches_video(1920, 1080, &c).unwrap(), (42, 24));
        assert_eq!(target_patches_video(512, 512, &c).unwrap(), (32, 32));
        let mut s = c.clone();
        s.video_maintain_aspect_ratio = false;
        assert_eq!(target_patches_video(1920, 1080, &s).unwrap(), (32, 32));
    }

    #[test]
    fn a_tile_is_even_within_bounds_and_the_pure_function_of_its_inputs() {
        let c = cfg();
        for (w, h) in [(1usize, 1usize), (3, 4000), (4000, 3), (33, 47), (640, 480), (1920, 1080), (8000, 8000), (5000, 70)] {
            let a = c.plan_for(w, h, 1).unwrap();
            assert_eq!(a, c.plan_for(w, h, 1).unwrap());
            assert!(a.w_patches % 2 == 0 && a.h_patches % 2 == 0, "{w}x{h}: {a:?}");
            assert!(a.w_patches * a.h_patches <= 13_312, "{w}x{h}: {a:?}");
            assert!(a.w_patches * a.h_patches >= 1024, "{w}x{h}: {a:?}");
            // idempotent: an image of exactly the canvas keeps its tile
            let (cw, ch) = a.canvas();
            assert_eq!(c.plan_for(cw, ch, 1).unwrap(), a, "{w}x{h} -> {cw}x{ch}");
        }
    }

    #[test]
    fn the_config_is_read_and_a_key_the_tower_cannot_take_is_named() {
        let ok = r#"{"image_processor_type":"NemotronH_Omni_Reasoning_V3ImageProcessor","_downsample_factor":2,"downsample_ratio":0.5,
            "max_model_len":16384,"max_num_patches":13312,"min_num_patches":1024,"norm_mean":[0.48145466,0.4578275,0.40821073],
            "norm_std":[0.26862954,0.26130258,0.27577711],"patch_size":16,"rescale_factor":0.00392156862745098,
            "video_maintain_aspect_ratio":true,"video_target_num_patches":1024}"#;
        assert_eq!(NemotronPreprocessing::from_json_str(ok).unwrap(), cfg());
        let wrap = format!(r#"{{"image_processor": {ok}, "video_temporal_patch_dim": 2}}"#);
        assert_eq!(NemotronPreprocessing::from_json_str(&wrap).unwrap(), cfg());
        for (find, replace, key) in [
            (r#""patch_size":16"#, r#""patch_size":14"#, "patch_size"),
            (r#""downsample_ratio":0.5"#, r#""downsample_ratio":0.25"#, "downsample_ratio"),
            (r#""_downsample_factor":2"#, r#""_downsample_factor":4"#, "_downsample_factor"),
            (r#""max_num_patches":13312"#, r#""max_num_patches":16384"#, "max_num_patches"),
            (r#""max_num_patches":13312"#, r#""max_num_patches":0"#, "max_num_patches"),
            (r#""min_num_patches":1024"#, r#""min_num_patches":20000"#, "min_num_patches"),
            (r#""max_model_len":16384"#, r#""max_model_len":0"#, "max_model_len"),
            (r#""norm_std":[0.26862954,0.26130258,0.27577711]"#, r#""norm_std":[0.2,0.0,0.3]"#, "norm_std"),
            (r#""norm_mean":[0.48145466,0.4578275,0.40821073]"#, r#""norm_mean":[0.1,0.2]"#, "norm_mean"),
            (r#""video_target_num_patches":1024"#, r#""video_target_num_patches":99999"#, "video_target_num_patches"),
            (r#""NemotronH_Omni_Reasoning_V3ImageProcessor""#, r#""Other""#, "image_processor_type"),
        ] {
            let e = NemotronPreprocessing::from_json_str(&ok.replace(find, replace)).unwrap_err();
            assert!(e.0.contains(key), "{key}: {e}");
        }
        assert!(NemotronPreprocessing::from_json_str("{}").unwrap_err().0.contains("norm_mean"));
    }

    #[test]
    fn a_lower_token_limit_shrinks_the_tile_instead_of_refusing() {
        let c = cfg().with_max_image_tokens(1000).unwrap();
        assert_eq!(c.max_num_patches, 4000);
        let t = c.plan_for(6000, 4000, 1).unwrap();
        assert!(t.tokens <= 1000 && t.tokens > 900, "{t:?}");
        // an image that fits keeps its tile
        assert_eq!(c.plan_for(512, 512, 1).unwrap(), cfg().plan_for(512, 512, 1).unwrap());
        assert!(cfg().with_max_image_tokens(4000).is_err(), "above the tower");
        assert!(cfg().with_max_image_tokens(100).is_err(), "below the minimum tile");
        assert_eq!(cfg().with_max_image_tokens(3328).unwrap(), cfg());
        // the limit is part of the identity: another tile for the same image is another cache key
        assert_ne!(c.identity(), cfg().identity());
    }

    #[test]
    fn the_identity_changes_with_every_parameter() {
        let base = cfg().identity();
        let mut seen = std::collections::BTreeSet::new();
        seen.insert(base.clone());
        let mut c = cfg();
        c.min_num_patches = 512;
        seen.insert(c.identity());
        let mut c = cfg();
        c.max_model_len = 8192;
        seen.insert(c.identity());
        let mut c = cfg();
        c.norm_mean[0] = 0.5;
        seen.insert(c.identity());
        let mut c = cfg();
        c.video_maintain_aspect_ratio = false;
        seen.insert(c.identity());
        let mut c = cfg();
        c.threads = 3; // not part of the model input
        assert_eq!(c.identity(), base);
        assert_eq!(seen.len(), 5);
    }

    /// A slow, plain re-statement of the kernel: the dense weight matrix of each axis, applied with the tap order the fast code uses.
    fn reference_resize(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
        let mut cur: Vec<f32> = src.iter().map(|&b| b as f32).collect();
        let (mut cw, mut ch) = (sw, sh);
        let dot = |taps: &[(f32, f32)]| -> f32 {
            let n = taps.len();
            let mut acc = taps[0].0 * taps[0].1;
            let plain = plain_taps(n);
            for (j, &(t, w)) in taps.iter().enumerate().skip(1) {
                if j <= plain {
                    acc += t * w;
                } else {
                    acc = t.mul_add(w, acc);
                }
            }
            acc
        };
        if dw != cw {
            let a = axis(cw, dw);
            let mut out = vec![0f32; dw * ch * 3];
            for y in 0..ch {
                for x in 0..dw {
                    for c in 0..3 {
                        let taps: Vec<(f32, f32)> =
                            (0..a.xsize[x]).map(|j| (cur[(y * cw + a.xmin[x] + j) * 3 + c], a.w[x * a.ksize + j])).collect();
                        out[(y * dw + x) * 3 + c] = dot(&taps);
                    }
                }
            }
            cur = out;
            cw = dw;
        }
        if dh != ch {
            let a = axis(ch, dh);
            let mut out = vec![0f32; cw * dh * 3];
            for y in 0..dh {
                for x in 0..cw {
                    for c in 0..3 {
                        let taps: Vec<(f32, f32)> =
                            (0..a.xsize[y]).map(|j| (cur[((a.xmin[y] + j) * cw + x) * 3 + c], a.w[y * a.ksize + j])).collect();
                        out[(y * cw + x) * 3 + c] = dot(&taps);
                    }
                }
            }
            cur = out;
            ch = dh;
        }
        let _ = ch;
        cur
    }

    #[test]
    fn the_threaded_resize_is_the_plain_one_on_every_thread_count() {
        // (sw, sh, dw, dh): both axes down, both up, one axis only, a strip, a 1-pixel destination
        let shapes = [
            (97usize, 61usize, 64usize, 80usize),
            (53, 37, 80, 64),
            (130, 40, 130, 50),
            (40, 130, 20, 130),
            (300, 17, 77, 33),
            (7, 5, 64, 64),
            (64, 64, 512, 512),
            (200, 3, 1, 3),
            (3, 200, 3, 1),
        ];
        for (i, &(sw, sh, dw, dh)) in shapes.iter().enumerate() {
            let src = noise(sw * sh * 3, 40 + i as u64);
            let want = reference_resize(&src, sw, sh, dw, dh);
            for t in [1usize, 2, 3, 8, 64] {
                assert_eq!(resize_bicubic_aa_f32(&src, sw, sh, dw, dh, t), want, "{sw}x{sh} -> {dw}x{dh} on {t} threads");
            }
        }
    }

    #[test]
    fn weights_sum_to_one_and_a_constant_image_stays_constant() {
        for (i, o) in [(100usize, 37usize), (37, 100), (4000, 140), (16, 512)] {
            let a = axis(i, o);
            for x in 0..o {
                let s: f32 = a.w[x * a.ksize..x * a.ksize + a.xsize[x]].iter().sum();
                assert!((s - 1.0).abs() < 1e-5, "{i}->{o} pixel {x}: {s}");
            }
        }
        let src = vec![200u8; 90 * 60 * 3];
        let out = resize_bicubic_aa_f32(&src, 90, 60, 33, 100, 1);
        assert!(out.iter().all(|&v| (v - 200.0).abs() < 1e-3), "constant image drifted");
    }

    #[test]
    fn an_image_of_its_tile_size_is_only_normalised() {
        let c = cfg();
        let src = noise(512 * 512 * 3, 3);
        let img = Rgb { width: 512, height: 512, data: src.clone() };
        let p = c.preprocess(&img).unwrap();
        assert_eq!((p.w_patches, p.h_patches, p.tokens), (32, 32, 256));
        let (x, y, ch) = (100usize, 37usize, 1usize);
        let v = src[(y * 512 + x) * 3 + ch] as f32;
        assert_eq!(p.pixels[ch * 512 * 512 + y * 512 + x], (v / 255.0 - c.norm_mean[ch]) / c.norm_std[ch]);
        assert_eq!(p.pixels.len(), 3 * 512 * 512);
    }

    #[test]
    fn the_hash_is_the_same_on_every_thread_count_and_changes_with_size_and_values() {
        let (w, h) = (300usize, 260usize);
        let px: Vec<f32> = noise(3 * w * h * 4, 9)
            .chunks_exact(4)
            .map(|b| f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0x7fff_ffff))
            .collect();
        let serial = canvas_hash(w, h, &px, 1);
        assert_eq!(canvas_hash(w, h, &px, 8), serial);
        let mut other = px.clone();
        other[5] = f32::from_bits(other[5].to_bits() ^ 1);
        assert_ne!(canvas_hash(w, h, &other, 1), serial, "one bit of one value");
        assert_ne!(canvas_hash(h, w, &px, 1), serial, "the same values as another shape");
    }

    #[test]
    fn bf16_is_round_to_nearest_even() {
        for (v, want) in [
            (1.0f32, 0x3f80u16),
            (-1.0, 0xbf80),
            (0.0, 0),
            (f32::from_bits(0x3f80_8000), 0x3f80),
            (f32::from_bits(0x3f81_8000), 0x3f82),
            (f32::from_bits(0x3f80_8001), 0x3f81),
        ] {
            assert_eq!(f32_to_bf16_bits(v), want, "{:#010x}", v.to_bits());
        }
        assert!(f32::from_bits((f32_to_bf16_bits(f32::NAN) as u32) << 16).is_nan());
    }

    #[test]
    fn the_trait_carries_the_bf16_planes_and_the_plan_matches() {
        let c = cfg();
        let img = Rgb { width: 200, height: 150, data: noise(200 * 150 * 3, 17) };
        let plan = ImagePreprocessor::plan(&c, 200, 150).unwrap();
        let pre = ImagePreprocessor::preprocess(&c, &img).unwrap();
        assert_eq!(pre.size, plan.canvas);
        assert_eq!(pre.grid, plan.grid);
        assert_eq!(pre.patches.len(), 3 * plan.canvas.0 * plan.canvas.1);
        assert_eq!(pre.n_patches() * pre.patch_dim, pre.patches.len());
        assert_eq!(pre.n_patches() / 4, plan.tokens);
        let f = c.preprocess(&img).unwrap();
        assert_eq!(pre.patches, f.bf16_bits());
        assert_eq!(pre.pixel_hash, f.pixel_hash);
    }
}
