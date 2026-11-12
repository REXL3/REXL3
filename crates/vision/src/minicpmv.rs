//! MiniCPM-V-4.7 image preprocessing (`docs/MINICPMV4_7_INTEGRATION_PLAN.md`), CPU only.
//!
//! The reference is transformers 5.18's `MiniCPMV4_6ImageProcessor` (the class `models/auto/image_processing_auto.py` maps `minicpmv4_7` to), in
//! both of its backends, and the model's own cast of `pixel_values` to bf16. Transformers is Apache-2.0; this is a re-expression of its slicing
//! arithmetic and its two normalisation paths, checked byte for byte against the real classes by `tools/gen_minicpmv_vision_fixtures.py` and
//! `tests/minicpmv_fixtures.rs` (no code is copied).
//!
//! **What the processor does.** An image of `h x w` pixels is cut into a *thumbnail* (the whole image at about 448 x 448 pixels) and, when it is
//! larger than one 448-pixel slice, a `rows x cols` grid of *slices* of a second, larger resize of the same source:
//!
//! 1. [`get_sliced_grid`]: `ratio = w * h / 448^2`, `multiple = min(ceil(ratio), max_slice_nums)`; one slice or none (`multiple <= 1`) means no grid.
//!    Otherwise the divisors of `multiple - 1`, `multiple` and `multiple + 1` (not 1, not above `max_slice_nums`) are the candidate `rows x cols`,
//!    the winner has the smallest `|ln(w / h) - ln(cols / rows)|` (a later candidate must be strictly smaller; an exact tie goes to more rows).
//! 2. The thumbnail is `find_best_resize(h, w, allow_upscale = no grid)`: when the area exceeds 448^2, or an upscale is allowed, the aspect is kept
//!    and `height = int(448 / sqrt(w / h))`, `width = int(height * w / h)`; each side then goes to `max(round(side / 56) * 56, 56)` (`patch * 4`:
//!    the tower merges 2 x 2 twice). A grid also takes [`get_refine_size`]: each side is first rounded to a multiple of its grid count, one
//!    slice's size is `find_best_resize` of that (upscale allowed), and the refined image is `slice * grid` pixels.
//! 3. Both resizes start from the ORIGINAL image (bicubic, antialias). The refined image is cut row-major into `rows * cols` equal slices.
//! 4. `(x / 255 - 0.5) / 0.5` in float32 (two different spellings, below), then `reshape_by_patch`: every crop becomes `[3, 14, gh * gw * 14]`
//!    with element `[c, i, l * 14 + j]` = channel `c`, row `i`, column `j` of patch `l` (row-major in the crop), and the crops are concatenated
//!    along the last axis, thumbnail first: `pixel_values` is `[1, 3, 14, total_patches * 14]` plus `target_sizes` `[crops, 2]` (patch rows,
//!    patch columns of each crop). The tower's conv (kernel = stride = 14) then reads one patch per 14 columns.
//!
//! **Two backends, two functions.** HF picks the torchvision backend when torchvision is installed and the PIL backend otherwise, and they differ
//! (`ResizeProfile`; [`ResizeProfile::Torch`] is the default, as for any HF install with torchvision):
//!
//! | | `Torch` (`MiniCPMV4_6ImageProcessor`) | `Pil` (`MiniCPMV4_6ImageProcessorPil`) |
//! | --- | --- | --- |
//! | resize | `tvF.resize` of a uint8 channels-last tensor, bicubic + antialias: torch's int16 fixed-point kernel (`torch_resample`) | `PIL.Image.resize(BICUBIC)`, Pillow 12.3 (`resample::bicubic_rgb_pil`) |
//! | normalise | `(x - mean * 255) / (std * 255)`, float32 (`_fuse_mean_std_and_rescale_factor`) | `float32(x * (1/255 in f64))`, then `(v - mean) / std` float32 |
//!
//! Both drop the alpha channel (`image.convert("RGB")`: [`AlphaPolicy::Drop`]) and neither applies EXIF orientation.
//!
//! **Output.** The model casts `pixel_values` to the tower's dtype (bf16) before the patch conv, so [`MiniCpmInput::pixels`] holds the packed
//! tensor as bf16 bits (round to nearest even from the exact float32: [`f32_to_bf16_bits`]); [`MiniCpmPreprocessing::preprocess_crops_f32`] is
//! the float32 tensor itself (HF's `pixel_values`), for oracles that run the reference in fp32 on the bf16-rounded pixels.
//!
//! **Through the `ImagePreprocessor` seam** the shared [`Preprocessed`] carries `patches` = the packed bf16 bits, `patch_dim` = 588, `grid` =
//! `(1, rows, cols)` of the slice grid (`(1, 0, 0)` without slices) and, unlike Qwen's, `size` = the ORIGINAL image size `(width, height)`:
//! the encoder re-derives the [`CropPlan`] (and so every `target_sizes` row) from it with the same configuration and checks the tensor length.
//! The trait's `plan(w, h)` is the placeholder count (`tokens` over all crops) and the thumbnail as `canvas`; the inherent
//! [`MiniCpmPreprocessing::plan`] is the full [`CropPlan`] (call the trait's as `ImagePreprocessor::plan(&p, w, h)`).

use crate::layout;
use crate::nemotron::f32_to_bf16_bits;
use crate::preprocessor::{ImagePlan, ImagePreprocessor};
use crate::resample::{self, par_rows};
use crate::rgb_decode::{decode_rgb, AlphaPolicy};
use crate::{torch_resample, Preprocessed, Rgb, VisionError};
use serde_json::Value;
use std::borrow::Cow;
use std::fmt;
use std::path::Path;

/// The slice scale of the checkpoint (`MiniCPMV4_6ImageProcessor.scale_resolution`): a slice is about 448 x 448 pixels.
pub const SCALE_RESOLUTION: usize = 448;
/// Patch side in pixels (`vision_config.patch_size`).
pub const PATCH_SIZE: usize = 14;
/// Elements of one patch: `3 * 14 * 14`, the K of the tower's patch-embedding GEMM.
pub const PATCH_DIM: usize = 3 * PATCH_SIZE * PATCH_SIZE;
/// `preprocessor_config.json`'s `max_slice_nums` and the class default.
pub const DEFAULT_MAX_SLICE_NUMS: usize = 9;
/// The most slices a configuration may ask for (a bound on the arithmetic; the processor has none).
pub const MAX_SLICE_NUMS_LIMIT: usize = 64;
/// Largest crop side the preprocessing takes, in patches (an image of 1 x 20,000 pixels would otherwise become a 60,032 x 56 crop).
pub const MAX_CROP_SIDE_PATCHES: usize = 4096;
/// Largest patch count of one image over all its crops (thumbnail and slices).
pub const MAX_TOTAL_PATCHES: usize = 65_536;

/// A `preprocessor_config.json` the preprocessing cannot honour.
#[derive(Debug, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "preprocessor_config.json: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Which HF backend the resize and the normalisation follow (see the module documentation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResizeProfile {
    /// The torchvision backend (`MiniCPMV4_6ImageProcessor`): the default of an HF install that has torchvision.
    #[default]
    Torch,
    /// The PIL backend (`MiniCPMV4_6ImageProcessorPil`): what runs without torchvision.
    Pil,
}

impl ResizeProfile {
    pub fn name(self) -> &'static str {
        match self {
            ResizeProfile::Torch => "torch",
            ResizeProfile::Pil => "pil",
        }
    }
}

/// `downsample_mode`: how many patches one LM token covers (`"16x"`: 4 x 4, the tower's window merger then the projector's 2 x 2; `"4x"`: 2 x 2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Downsample {
    #[default]
    X16,
    X4,
}

impl Downsample {
    /// Patches per LM token (16 or 4).
    pub fn divisor(self) -> usize {
        match self {
            Downsample::X16 => 16,
            Downsample::X4 => 4,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Downsample::X16 => "16x",
            Downsample::X4 => "4x",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "16x" => Some(Downsample::X16),
            "4x" => Some(Downsample::X4),
            _ => None,
        }
    }
}

/// Python `round` (half to even) of a float, as the processor's `ensure_divide` uses it.
fn py_round(x: f64) -> f64 {
    x.round_ties_even()
}

/// `ensure_divide(length, divisor) = max(round(length / divisor) * divisor, divisor)` for a float length (the refine size passes floats).
fn ensure_divide_f(length: f64, divisor: usize) -> usize {
    let n = py_round(length / divisor as f64) as usize * divisor;
    n.max(divisor)
}

/// `max(round(length / divisor) * divisor, divisor)`: the nearest multiple of `divisor`, at least `divisor` (a tie goes to the even quotient).
pub fn ensure_divide(length: usize, divisor: usize) -> usize {
    ensure_divide_f(length as f64, divisor)
}

fn find_best_resize_f(
    height: f64,
    width: f64,
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> (usize, usize) {
    let (mut h, mut w) = (height, width);
    if h * w > (scale_resolution * scale_resolution) as f64 || allow_upscale {
        let aspect = w / h;
        // `int(scale_resolution / math.sqrt(aspect_ratio))` then `int(height * aspect_ratio)`: truncation toward zero
        let ih = (scale_resolution as f64 / aspect.sqrt()) as u64;
        h = ih as f64;
        w = (h * aspect) as u64 as f64;
    }
    // factor 4 = two successive 2 x 2 merges (the tower's window merger and the projector)
    (ensure_divide_f(h, patch_size * 4), ensure_divide_f(w, patch_size * 4))
}

/// `find_best_resize(image_size = (height, width), ...)` -> `(best_height, best_width)`.
pub fn find_best_resize(
    image_size: (usize, usize),
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> (usize, usize) {
    find_best_resize_f(image_size.0 as f64, image_size.1 as f64, scale_resolution, patch_size, allow_upscale)
}

/// `get_refine_size(image_size = (height, width), grid = (rows, cols), ...)` -> `(refine_height, refine_width)`: the size of the image the slices
/// are cut from.
pub fn get_refine_size(
    image_size: (usize, usize),
    grid: (usize, usize),
    scale_resolution: usize,
    patch_size: usize,
    allow_upscale: bool,
) -> (usize, usize) {
    let (height, width) = image_size;
    let (grid_y, grid_x) = grid;
    let refine_width = ensure_divide(width, grid_x);
    let refine_height = ensure_divide(height, grid_y);
    // the processor hands `find_best_resize` the (float) size of one slice
    let (bh, bw) = find_best_resize_f(
        refine_height as f64 / grid_y as f64,
        refine_width as f64 / grid_x as f64,
        scale_resolution,
        patch_size,
        allow_upscale,
    );
    (bh * grid_y, bw * grid_x)
}

/// `get_sliced_grid(image_size = (height, width), max_slice_nums, scale_resolution)` -> `Some((rows, cols))`, or `None` when the image is
/// at most one slice. The tie rule (`error == min_error and num_rows > best_grid[0]`) and the float `ln` are the processor's.
pub fn get_sliced_grid(image_size: (usize, usize), max_slice_nums: usize, scale_resolution: usize) -> Option<(usize, usize)> {
    let (original_height, original_width) = image_size;
    let log_ratio = (original_width as f64 / original_height as f64).ln();
    let ratio = (original_width as f64 * original_height as f64) / (scale_resolution as f64 * scale_resolution as f64);
    let multiple = (ratio.ceil() as usize).min(max_slice_nums);
    if multiple <= 1 {
        return None;
    }
    let mut best_grid = (1usize, 1usize);
    let mut min_error = f64::INFINITY;
    for num_slices in [multiple - 1, multiple, multiple + 1] {
        if num_slices == 1 || num_slices > max_slice_nums {
            continue;
        }
        for num_rows in 1..=num_slices {
            if num_slices % num_rows == 0 {
                let num_cols = num_slices / num_rows;
                let error = (log_ratio - (num_cols as f64 / num_rows as f64).ln()).abs();
                if error < min_error {
                    best_grid = (num_rows, num_cols);
                    min_error = error;
                } else if error == min_error && num_rows > best_grid.0 {
                    best_grid = (num_rows, num_cols);
                }
            }
        }
    }
    Some(best_grid)
}

/// One crop of an image: the thumbnail, or one slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropSpec {
    /// Pixel size of the crop (a multiple of 56).
    pub px_w: usize,
    pub px_h: usize,
    /// Patch grid of the crop (`target_sizes` row: `[gh, gw]`).
    pub gh: usize,
    pub gw: usize,
    /// Placeholder tokens the crop takes: `gh * gw / 16` (`/ 4` in `4x` mode).
    pub tokens: usize,
}

impl CropSpec {
    pub fn patches(&self) -> usize {
        self.gh * self.gw
    }
}

/// The geometry of one image, from its size alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CropPlan {
    /// The thumbnail first, then `rows * cols` equal slices row-major (none without a grid).
    pub crops: Vec<CropSpec>,
    /// `(rows, cols)` of the slice grid; `(0, 0)` is no slices (HF's `grids` entry `[0, 0]`).
    pub grid: (usize, usize),
    pub total_patches: usize,
    pub total_tokens: usize,
}

impl CropPlan {
    /// Pixel size `(width, height)` of the refined image the slices are cut from; `None` without slices.
    pub fn refine_size(&self) -> Option<(usize, usize)> {
        (self.grid.0 > 0).then(|| (self.crops[1].px_w * self.grid.1, self.crops[1].px_h * self.grid.0))
    }

    /// HF's `target_sizes`: `[patch rows, patch columns]` of every crop, in order.
    pub fn target_sizes(&self) -> Vec<[u32; 2]> {
        self.crops.iter().map(|c| [c.gh as u32, c.gw as u32]).collect()
    }

    /// Slices (the crops after the thumbnail).
    pub fn slices(&self) -> usize {
        self.crops.len() - 1
    }
}

/// An image through the preprocessing: the packed tower input and its geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MiniCpmInput {
    pub plan: CropPlan,
    /// `pixel_values` as bf16 bits: `[3, 14, total_patches * 14]` row-major (channel, row, then the patches of all crops side by side).
    pub pixels: Vec<u16>,
    /// HF's `target_sizes`: `[patch rows, patch columns]` per crop.
    pub target_sizes: Vec<[u32; 2]>,
    /// FNV-1a over the geometry and the resized RGB bytes of the thumbnail and the refined image: equal for equal model inputs (under one `id()`).
    pub pixel_hash: u64,
}

impl MiniCpmInput {
    /// The `Preprocessed` of the `ImagePreprocessor` seam. `original` is the source image's `(width, height)` (see the module documentation).
    pub fn into_preprocessed(self, original: (usize, usize)) -> Preprocessed {
        Preprocessed {
            size: original,
            grid: (1, self.plan.grid.0, self.plan.grid.1),
            patches: self.pixels,
            patch_dim: PATCH_DIM,
            pixel_hash: self.pixel_hash,
        }
    }
}

/// MiniCPM-V-4.7 image preprocessing parameters (`preprocessor_config.json` plus the serving limits).
#[derive(Clone, Debug, PartialEq)]
pub struct MiniCpmPreprocessing {
    pub scale_resolution: usize,
    pub patch_size: usize,
    /// Most slices of one image (the checkpoint says 9; 1 gives thumbnail-only images, as a video frame uses; 36 is the card's high-resolution setting).
    pub max_slice_nums: usize,
    /// `slice_mode`: false never slices (HF's video processor has its own default).
    pub slice_mode: bool,
    pub downsample: Downsample,
    pub profile: ResizeProfile,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// `rescale_factor` (1/255 as a float64, as in HF).
    pub rescale_factor: f64,
    /// Refuse images with more pixels than this before any pixel work (a decompression-bomb guard).
    pub max_pixels: usize,
    /// Worker threads of the resize, the layout and the hash: 0 picks `resize::default_workers()` (`REXL3_VISION_THREADS`, else
    /// `min(8, cores)`), a positive value is the most it may use, 1 runs inline. Rows are independent, so the output does not depend on it.
    pub threads: usize,
}

impl Default for MiniCpmPreprocessing {
    fn default() -> Self {
        Self::minicpmv47()
    }
}

fn bad(m: impl Into<String>) -> VisionError {
    VisionError::Unsupported(m.into())
}

impl MiniCpmPreprocessing {
    /// The checkpoint's settings: 448-pixel slices of 14-pixel patches, at most 9 slices, 16x downsampling, mean = std = 0.5, torch backend.
    pub fn minicpmv47() -> Self {
        Self {
            scale_resolution: SCALE_RESOLUTION,
            patch_size: PATCH_SIZE,
            max_slice_nums: DEFAULT_MAX_SLICE_NUMS,
            slice_mode: true,
            downsample: Downsample::X16,
            profile: ResizeProfile::Torch,
            mean: [0.5; 3],
            std: [0.5; 3],
            rescale_factor: 1.0 / 255.0,
            max_pixels: crate::decode::MAX_DECODED_PIXELS,
            threads: 0,
        }
    }

    /// Read a `preprocessor_config.json` (or a `processor_config.json`, whose `image_processor` block is used). A missing key takes the class
    /// default (as transformers does); a key the preprocessing cannot honour is an error that names it. `downsample_mode` is not a file key of
    /// the checkpoint (the model config has it); it is read when present.
    pub fn from_preprocessor_config(root: &Value) -> Result<Self, ConfigError> {
        let cfg = root.get("image_processor").filter(|v| v.is_object()).unwrap_or(root);
        let mut p = Self::minicpmv47();
        let e = |m: String| Err(ConfigError(m));
        for (key, slot, lo, hi) in [
            ("max_slice_nums", &mut p.max_slice_nums, 1usize, MAX_SLICE_NUMS_LIMIT),
            ("scale_resolution", &mut p.scale_resolution, 56, 4096),
            ("patch_size", &mut p.patch_size, 1, 256),
        ] {
            if let Some(v) = cfg.get(key) {
                match v.as_u64() {
                    Some(n) if (lo as u64..=hi as u64).contains(&n) => *slot = n as usize,
                    _ => return e(format!("{key} must be an integer in {lo}..={hi}, got {v}")),
                }
            }
        }
        if let Some(v) = cfg.get("slice_mode") {
            p.slice_mode = v.as_bool().ok_or_else(|| ConfigError(format!("slice_mode must be a boolean, got {v}")))?;
        }
        if let Some(v) = cfg.get("downsample_mode") {
            p.downsample = v
                .as_str()
                .and_then(Downsample::parse)
                .ok_or_else(|| ConfigError(format!("downsample_mode must be \"16x\" or \"4x\", got {v}")))?;
        }
        for key in ["do_rescale", "do_normalize", "do_resize", "do_convert_rgb"] {
            if cfg.get(key).is_some_and(|v| v.as_bool() != Some(true)) {
                return e(format!("{key} must be true (the processor's fused path is the only one built)"));
            }
        }
        if let Some(v) = cfg.get("rescale_factor") {
            match v.as_f64() {
                Some(f) if f.is_finite() && f > 0.0 => p.rescale_factor = f,
                _ => return e(format!("rescale_factor must be a positive number, got {v}")),
            }
        }
        for (key, slot) in [("image_mean", &mut p.mean), ("image_std", &mut p.std)] {
            if let Some(v) = cfg.get(key) {
                let vals: Option<Vec<f64>> =
                    v.as_array().filter(|a| a.len() == 3).and_then(|a| a.iter().map(Value::as_f64).collect());
                let vals = vals.ok_or_else(|| ConfigError(format!("{key} must be 3 numbers, got {v}")))?;
                for (s, x) in slot.iter_mut().zip(vals) {
                    *s = x as f32;
                }
            }
        }
        p.validate()?;
        Ok(p)
    }

    /// Read `preprocessor_config.json` of a checkpoint directory (the 4.7 checkpoint's holds `max_slice_nums`, `image_mean`, `image_std` and
    /// `mrope_mode`; `processor_config.json` is `{}` and is not needed).
    pub fn from_dir(dir: &Path) -> Result<Self, ConfigError> {
        let file = dir.join("preprocessor_config.json");
        let text = std::fs::read_to_string(&file).map_err(|e| ConfigError(format!("{}: {e}", file.display())))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| ConfigError(format!("{}: {e}", file.display())))?;
        Self::from_preprocessor_config(&v)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        let e = |m: &str| Err(ConfigError(m.into()));
        if self.patch_size == 0 || self.scale_resolution < self.patch_size * 4 {
            return e("patch_size must be positive and scale_resolution at least 4 patches");
        }
        if !(1..=MAX_SLICE_NUMS_LIMIT).contains(&self.max_slice_nums) {
            return e("max_slice_nums out of range");
        }
        if self.std.iter().any(|s| !s.is_finite() || *s == 0.0) || self.mean.iter().any(|m| !m.is_finite()) {
            return e("image_mean must be finite and image_std finite and non-zero");
        }
        if !(self.rescale_factor.is_finite() && self.rescale_factor > 0.0) {
            return e("rescale_factor must be positive");
        }
        if self.max_pixels == 0 {
            return e("max_pixels must be positive");
        }
        Ok(())
    }

    pub fn with_downsample(mut self, d: Downsample) -> Self {
        self.downsample = d;
        self
    }

    pub fn with_profile(mut self, profile: ResizeProfile) -> Self {
        self.profile = profile;
        self
    }

    /// `max_slice_nums` (1: thumbnail only; 36: the card's high-resolution setting).
    pub fn with_max_slice_nums(mut self, n: usize) -> Result<Self, ConfigError> {
        self.max_slice_nums = n;
        self.validate()?;
        Ok(self)
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// A stable identity of everything that changes the model input (the cache key's preprocessing id; bump the `v1` when the semantics change).
    pub fn id(&self) -> String {
        let bits = |a: &[f32; 3]| a.iter().map(|v| format!("{:08x}", v.to_bits())).collect::<Vec<_>>().join(",");
        format!(
            "minicpmv4.7-vision-v1:{}:s{}p{}m{}:{}:{}:mean{}:std{}:rf{:016x}",
            self.profile.name(),
            self.scale_resolution,
            self.patch_size,
            self.max_slice_nums,
            if self.slice_mode { "slice" } else { "noslice" },
            self.downsample.name(),
            bits(&self.mean),
            bits(&self.std),
            self.rescale_factor.to_bits(),
        )
    }

    fn cap(&self) -> usize {
        if self.threads == 0 {
            crate::resize::default_workers()
        } else {
            self.threads
        }
    }

    fn check_size(&self, width: usize, height: usize) -> Result<(), VisionError> {
        if width == 0 || height == 0 {
            return Err(bad(format!("an image of {width}x{height} pixels has no content")));
        }
        if width.checked_mul(height).is_none_or(|p| p > self.max_pixels) {
            return Err(bad(format!("{width}x{height} exceeds the limit of {} pixels", self.max_pixels)));
        }
        Ok(())
    }

    /// The thumbnail, the slice grid and every crop's size and token count of a `width x height` image, from the size alone. Fails by name for an
    /// image over the pixel limit or whose crops would exceed [`MAX_CROP_SIDE_PATCHES`] or [`MAX_TOTAL_PATCHES`] (an extreme aspect ratio).
    pub fn plan(&self, width: usize, height: usize) -> Result<CropPlan, VisionError> {
        self.check_size(width, height)?;
        let (scale, patch) = (self.scale_resolution, self.patch_size);
        let size = (height, width);
        let grid = if self.slice_mode { get_sliced_grid(size, self.max_slice_nums, scale) } else { None };
        // the source image is always resized (the model resizes "based on auto-inferred size at run-time"); an upscale is allowed without a grid
        let (th, tw) = find_best_resize(size, scale, patch, grid.is_none());
        let div = self.downsample.divisor();
        let crop = |px_w: usize, px_h: usize| -> Result<CropSpec, VisionError> {
            let (gh, gw) = (px_h / patch, px_w / patch);
            if gh > MAX_CROP_SIDE_PATCHES || gw > MAX_CROP_SIDE_PATCHES {
                return Err(bad(format!(
                    "{width}x{height}: a crop of {px_w}x{px_h} pixels ({gw}x{gh} patches) exceeds {MAX_CROP_SIDE_PATCHES} patches a side"
                )));
            }
            Ok(CropSpec { px_w, px_h, gh, gw, tokens: gh * gw / div })
        };
        let mut crops = vec![crop(tw, th)?];
        let mut total = crops[0].patches();
        let mut out_grid = (0, 0);
        if let Some((rows, cols)) = grid {
            let (rh, rw) = get_refine_size(size, (rows, cols), scale, patch, true);
            let (sh, sw) = (rh / rows, rw / cols);
            let one = crop(sw, sh)?;
            total = total.saturating_add(one.patches().saturating_mul(rows * cols));
            if total > MAX_TOTAL_PATCHES {
                return Err(bad(format!("{width}x{height}: {total} patches exceed the limit of {MAX_TOTAL_PATCHES}")));
            }
            crops.extend(std::iter::repeat_n(one, rows * cols));
            out_grid = (rows, cols);
        }
        let total_tokens = crops.iter().map(|c| c.tokens).sum();
        Ok(CropPlan { crops, grid: out_grid, total_patches: total, total_tokens })
    }

    /// The 256-entry value table of each channel for the profile's normalisation, in float32 (HF's `pixel_values` values).
    pub fn lut_f32(&self) -> [[f32; 256]; 3] {
        let mut lut = [[0f32; 256]; 3];
        match self.profile {
            ResizeProfile::Torch => {
                // `_fuse_mean_std_and_rescale_factor`: float32 mean and std times (1.0 / rescale_factor) (a Python float), then `(x - mean) / std`
                let inv = (1.0f64 / self.rescale_factor) as f32;
                for (c, table) in lut.iter_mut().enumerate() {
                    let (m, s) = (self.mean[c] * inv, self.std[c] * inv);
                    for (v, o) in table.iter_mut().enumerate() {
                        *o = (v as f32 - m) / s;
                    }
                }
            }
            ResizeProfile::Pil => {
                // `np.float64(x) * rescale_factor` cast to float32, then `(v - mean) / std` in float32
                for (c, table) in lut.iter_mut().enumerate() {
                    for (v, o) in table.iter_mut().enumerate() {
                        let r = (v as f64 * self.rescale_factor) as f32;
                        *o = (r - self.mean[c]) / self.std[c];
                    }
                }
            }
        }
        lut
    }

    /// [`MiniCpmPreprocessing::lut_f32`] rounded to bf16 (round to nearest even), the values the tower reads.
    pub fn lut_bf16(&self) -> [[u16; 256]; 3] {
        self.lut_f32().map(|t| t.map(f32_to_bf16_bits))
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

    /// The image resized to `dw x dh` with the profile's kernel (a view of the source when that is its size).
    fn resize<'a>(&self, img: &'a Rgb, dw: usize, dh: usize) -> Cow<'a, [u8]> {
        if (dw, dh) == (img.width, img.height) {
            return Cow::Borrowed(&img.data);
        }
        Cow::Owned(match self.profile {
            ResizeProfile::Torch => {
                torch_resample::bicubic_rgb_torch_with(&img.data, img.width, img.height, dw, dh, self.threads)
            }
            ResizeProfile::Pil => resample::bicubic_rgb_pil_with(&img.data, img.width, img.height, dw, dh, self.threads),
        })
    }

    /// The resized sources of a plan: the thumbnail, and the refined image when there is a grid.
    fn sources<'a>(&self, img: &'a Rgb, plan: &CropPlan) -> (Cow<'a, [u8]>, Option<Cow<'a, [u8]>>) {
        let t = &plan.crops[0];
        let thumb = self.resize(img, t.px_w, t.px_h);
        let refine = plan.refine_size().map(|(rw, rh)| self.resize(img, rw, rh));
        (thumb, refine)
    }

    fn content_hash(plan: &CropPlan, thumb: &[u8], refine: Option<&[u8]>) -> u64 {
        let mut h = FNV_OFFSET;
        for v in [plan.crops.len(), plan.grid.0, plan.grid.1] {
            h = fnv(h, &(v as u64).to_le_bytes());
        }
        for c in &plan.crops {
            h = fnv(h, &(c.px_w as u64).to_le_bytes());
            h = fnv(h, &(c.px_h as u64).to_le_bytes());
        }
        h = fnv(h, thumb);
        if let Some(r) = refine {
            h = fnv(h, r);
        }
        h
    }

    fn pack<T: Copy + Default + Send + Sync>(
        &self,
        plan: &CropPlan,
        thumb: &[u8],
        refine: Option<&[u8]>,
        lut: &[[T; 256]; 3],
        workers: usize,
    ) -> Vec<T> {
        pack_crops(plan, self.patch_size, thumb, refine, lut, workers)
    }

    /// Preprocess one image: the crops of [`MiniCpmPreprocessing::plan`] resized, normalised, rounded to bf16 and packed as HF's `pixel_values`.
    /// The hash of the resized pixels runs beside the layout on its own thread for large images (`layout::beside`).
    pub fn preprocess_crops(&self, img: &Rgb) -> Result<MiniCpmInput, VisionError> {
        self.validate_image(img)?;
        let plan = self.plan(img.width, img.height)?;
        let (thumb, refine) = self.sources(img, &plan);
        let lut = self.lut_bf16();
        let source_pixels = plan.total_patches * PATCH_SIZE * PATCH_SIZE;
        let lp = layout::plan(self.cap(), source_pixels, true);
        let (pixels, pixel_hash) = layout::beside(
            lp.hash_aside,
            || Self::content_hash(&plan, &thumb, refine.as_deref()),
            || self.pack(&plan, &thumb, refine.as_deref(), &lut, lp.workers),
        );
        let target_sizes = plan.target_sizes();
        Ok(MiniCpmInput { plan, pixels, target_sizes, pixel_hash })
    }

    /// [`MiniCpmPreprocessing::preprocess_crops`] without the bf16 rounding: HF's float32 `pixel_values` `[3, 14, total_patches * 14]` (for the
    /// tests, and for an fp32 reference that runs the tower on the bf16-rounded pixels).
    pub fn preprocess_crops_f32(&self, img: &Rgb) -> Result<(CropPlan, Vec<f32>), VisionError> {
        self.validate_image(img)?;
        let plan = self.plan(img.width, img.height)?;
        let (thumb, refine) = self.sources(img, &plan);
        let source_pixels = plan.total_patches * PATCH_SIZE * PATCH_SIZE;
        let lp = layout::plan(self.cap(), source_pixels, false);
        let out = self.pack(&plan, &thumb, refine.as_deref(), &self.lut_f32(), lp.workers);
        Ok((plan, out))
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

fn fnv(h: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(FNV_PRIME))
}

/// HF's `reshape_by_patch` of every crop and the concatenation along the last axis: `[3, ps, total_patches * ps]`, element
/// `[c, i, l * ps + j]` of crop `k` (patches before it: `off`) is `lut[c][pixel (row by * ps + i, column bx * ps + j) of the crop][c]` with
/// `l = off + by * gw + bx`. The thumbnail is `thumb` (`px_w x px_h` RGB); the slices are cut row-major from `refine` (`cols * slice_w` wide).
/// One output row (channel, `i`) is independent of the others, so the rows are spread over `workers` threads.
fn pack_crops<T: Copy + Default + Send + Sync>(
    plan: &CropPlan,
    ps: usize,
    thumb: &[u8],
    refine: Option<&[u8]>,
    lut: &[[T; 256]; 3],
    workers: usize,
) -> Vec<T> {
    let row_len = plan.total_patches * ps;
    let mut out = vec![T::default(); 3 * ps * row_len];
    if row_len == 0 {
        return out;
    }
    // where each crop's pixels are: the source bytes and their row stride (pixels), the crop's top-left corner in them, and its first patch
    struct Job<'a> {
        src: &'a [u8],
        stride: usize,
        x0: usize,
        y0: usize,
        gh: usize,
        gw: usize,
        off: usize,
    }
    let mut jobs: Vec<Job> = Vec::with_capacity(plan.crops.len());
    let t = &plan.crops[0];
    jobs.push(Job { src: thumb, stride: t.px_w, x0: 0, y0: 0, gh: t.gh, gw: t.gw, off: 0 });
    if let (Some(r), Some((rw, _))) = (refine, plan.refine_size()) {
        let mut off = t.patches();
        let s = &plan.crops[1];
        for k in 0..plan.slices() {
            let (row, col) = (k / plan.grid.1, k % plan.grid.1);
            jobs.push(Job { src: r, stride: rw, x0: col * s.px_w, y0: row * s.px_h, gh: s.gh, gw: s.gw, off });
            off += s.patches();
        }
    }
    par_rows(&mut out, row_len, workers, |r, dst| {
        let (c, i) = (r / ps, r % ps);
        let table = &lut[c];
        for j in &jobs {
            for by in 0..j.gh {
                let line = (j.y0 + by * ps + i) * j.stride + j.x0;
                for bx in 0..j.gw {
                    let px = &j.src[(line + bx * ps) * 3..(line + bx * ps + ps) * 3];
                    let d = &mut dst[(j.off + by * j.gw + bx) * ps..][..ps];
                    for (o, p) in d.iter_mut().zip(px.chunks_exact(3)) {
                        *o = table[p[c] as usize];
                    }
                }
            }
        }
    });
    out
}

impl ImagePreprocessor for MiniCpmPreprocessing {
    fn family(&self) -> &'static str {
        "minicpmv4.7"
    }

    fn id(&self) -> String {
        MiniCpmPreprocessing::id(self)
    }

    /// `image.convert("RGB")`: the alpha channel is dropped (PNG; a JPEG goes through `decode::decode`, which applies its EXIF orientation).
    fn decode(&self, bytes: &[u8]) -> Result<Rgb, VisionError> {
        decode_rgb(bytes, AlphaPolicy::Drop)
    }

    /// `canvas` is the thumbnail (width, height), `grid` is `(1, rows, cols)` of the slice grid, `tokens` the placeholders of all crops.
    fn plan(&self, width: usize, height: usize) -> Result<ImagePlan, VisionError> {
        let p = MiniCpmPreprocessing::plan(self, width, height)?;
        let t = p.crops[0];
        Ok(ImagePlan { canvas: (t.px_w, t.px_h), grid: (1, p.grid.0, p.grid.1), tokens: p.total_tokens })
    }

    /// `Preprocessed` with `patches` = the packed bf16 bits and `size` = the ORIGINAL `(width, height)` (see the module documentation).
    fn preprocess(&self, img: &Rgb) -> Result<Preprocessed, VisionError> {
        let original = (img.width, img.height);
        Ok(self.preprocess_crops(img)?.into_preprocessed(original))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_divide_rounds_half_to_even_and_never_below_the_divisor() {
        // 84 / 56 = 1.5 -> 2 (even), 28 / 56 = 0.5 -> 0 -> 56, 140 / 56 = 2.5 -> 2 -> 112
        assert_eq!(ensure_divide(84, 56), 112);
        assert_eq!(ensure_divide(28, 56), 56);
        assert_eq!(ensure_divide(140, 56), 112);
        assert_eq!(ensure_divide(0, 56), 56);
        assert_eq!(ensure_divide(448, 56), 448);
        assert_eq!(ensure_divide(447, 56), 448);
        assert_eq!(ensure_divide(420, 56), 448, "7.5 -> 8");
        assert_eq!(ensure_divide(476, 56), 448, "8.5 -> 8 (even)");
    }

    #[test]
    fn a_small_image_is_one_thumbnail_upscaled_to_about_a_slice() {
        let p = MiniCpmPreprocessing::minicpmv47();
        let plan = p.plan(1, 1).unwrap();
        assert_eq!(plan.grid, (0, 0));
        assert_eq!(plan.crops, vec![CropSpec { px_w: 448, px_h: 448, gh: 32, gw: 32, tokens: 64 }]);
        assert_eq!((plan.total_patches, plan.total_tokens), (1024, 64));
        // 4x keeps four times the tokens
        let p4 = p.clone().with_downsample(Downsample::X4);
        assert_eq!(p4.plan(1, 1).unwrap().total_tokens, 256);
    }

    #[test]
    fn grids_follow_the_log_aspect_rule_and_ties_go_to_more_rows() {
        // 1344x896: ratio 6.0, log ratio ln 1.5: of {5, 6, 7} slices 2 rows x 3 columns is the exact fit
        assert_eq!(get_sliced_grid((896, 1344), 9, 448), Some((2, 3)));
        // a square: 4 slices 2x2 (error 0), not 1x4 or 4x1
        assert_eq!(get_sliced_grid((896, 896), 9, 448), Some((2, 2)));
        // one slice or less: none
        assert_eq!(get_sliced_grid((448, 448), 9, 448), None);
        assert_eq!(get_sliced_grid((300, 300), 9, 448), None);
        // max_slice_nums 1 never slices
        assert_eq!(get_sliced_grid((4000, 4000), 1, 448), None);
        // the cap bites: 6000x4000 wants 120 slices, 9 are allowed
        let g = get_sliced_grid((4000, 6000), 9, 448).unwrap();
        assert!(g.0 * g.1 <= 9 && g.0 * g.1 >= 8, "{g:?}");
    }

    #[test]
    fn the_refined_image_is_the_slice_size_times_the_grid() {
        let p = MiniCpmPreprocessing::minicpmv47();
        let plan = p.plan(1344, 896).unwrap();
        assert_eq!(plan.grid, (2, 3));
        assert_eq!(plan.crops.len(), 7);
        let s = plan.crops[1];
        assert_eq!((s.px_w, s.px_h), (448, 448));
        assert_eq!(plan.refine_size(), Some((1344, 896)));
        assert!(plan.crops[1..].iter().all(|c| *c == s));
        // the thumbnail keeps the 3:2 aspect: int(448 / sqrt(1.5)) = 365 -> 392, int(365 * 1.5) = 547 -> 560
        assert_eq!(plan.target_sizes()[0], [28, 40]);
    }

    #[test]
    fn extreme_aspect_ratios_are_refused_by_name() {
        let p = MiniCpmPreprocessing::minicpmv47();
        let e = p.plan(20000, 1).unwrap_err().to_string();
        assert!(e.contains("patches a side"), "{e}");
        assert!(p.plan(0, 5).is_err());
        assert!(p.plan(1 << 14, 1 << 14).unwrap_err().to_string().contains("exceeds the limit"));
    }

    #[test]
    fn config_reads_the_checkpoint_file_and_refuses_what_it_cannot_honour() {
        let v: Value = serde_json::from_str(
            r#"{"max_slice_nums": 9, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "mrope_mode": "canvas"}"#,
        )
        .unwrap();
        assert_eq!(MiniCpmPreprocessing::from_preprocessor_config(&v).unwrap(), MiniCpmPreprocessing::minicpmv47());
        let v36: Value = serde_json::from_str(r#"{"max_slice_nums": 36, "downsample_mode": "4x"}"#).unwrap();
        let p = MiniCpmPreprocessing::from_preprocessor_config(&v36).unwrap();
        assert_eq!((p.max_slice_nums, p.downsample), (36, Downsample::X4));
        for bad in [
            r#"{"max_slice_nums": 0}"#,
            r#"{"max_slice_nums": 65}"#,
            r#"{"downsample_mode": "8x"}"#,
            r#"{"image_std": [0.5, 0.0, 0.5]}"#,
            r#"{"do_normalize": false}"#,
            r#"{"image_mean": [0.5]}"#,
        ] {
            let v: Value = serde_json::from_str(bad).unwrap();
            assert!(MiniCpmPreprocessing::from_preprocessor_config(&v).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_id_names_every_input_that_changes_the_tensor() {
        let p = MiniCpmPreprocessing::minicpmv47();
        let ids = [
            p.id(),
            p.clone().with_profile(ResizeProfile::Pil).id(),
            p.clone().with_downsample(Downsample::X4).id(),
            p.clone().with_max_slice_nums(36).unwrap().id(),
        ];
        for (i, a) in ids.iter().enumerate() {
            for b in &ids[i + 1..] {
                assert_ne!(a, b);
            }
        }
        assert!(p.id().starts_with("minicpmv4.7-vision-v1:torch:"));
    }

    #[test]
    fn the_packed_layout_puts_patch_l_row_i_column_j_at_l_times_14_plus_j() {
        // a 56x56 source (no grid, upscaled to 448x448): the thumbnail's pixels come from a resize, so use the layout function directly on a
        // known crop of 28x14 pixels: two patches side by side
        let (w, h) = (28usize, 14usize);
        let data: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        let plan = CropPlan {
            crops: vec![CropSpec { px_w: w, px_h: h, gh: 1, gw: 2, tokens: 0 }],
            grid: (0, 0),
            total_patches: 2,
            total_tokens: 0,
        };
        let lut: [[u16; 256]; 3] = std::array::from_fn(|c| std::array::from_fn(|v| (c * 1000 + v) as u16));
        let out = pack_crops(&plan, 14, &data, None, &lut, 1);
        assert_eq!(out.len(), 3 * 14 * 28);
        for c in 0..3 {
            for i in 0..14 {
                for l in 0..2 {
                    for j in 0..14 {
                        let px = data[(i * w + l * 14 + j) * 3 + c] as usize;
                        assert_eq!(out[(c * 14 + i) * 28 + l * 14 + j], lut[c][px], "c{c} i{i} l{l} j{j}");
                    }
                }
            }
        }
    }
}
