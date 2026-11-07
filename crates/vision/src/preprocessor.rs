//! The seam between a model family and its image preprocessing.
//!
//! A family supplies one [`ImagePreprocessor`]: how its bytes become RGB (alpha policy, formats), how many placeholder tokens an image of a given
//! size takes (known from the size alone, so the prompt can be expanded and admission can be charged before any pixel is touched), and how the
//! pixels become the tensor its tower reads. Qwen3.8's existing `Preprocessing` is the first implementation (below, behaviour unchanged: it calls
//! the same functions the server calls today); GLM-5.3-Flash's `GlmPreprocessing` is the second.
//!
//! What stays outside the trait on purpose: the tower itself (it reads `Preprocessed.patches` and `grid`), the splice and position rule (Qwen
//! uses M-RoPE over the grid, GLM's text attention is NoPE), and the chat format's placeholder rendering.

use std::fmt;

use crate::{decode, preprocess, smart_resize, Preprocessed, Preprocessing, Rgb, VisionError};

/// What a prompt and the scheduler need to know about an image before its pixels are processed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImagePlan {
    /// The (width, height) of the tensor the tower sees, in pixels.
    pub canvas: (usize, usize),
    /// Patch grid (t, h, w).
    pub grid: (usize, usize, usize),
    /// Placeholder tokens the image occupies in the prompt.
    pub tokens: usize,
}

pub trait ImagePreprocessor: Send + Sync + fmt::Debug {
    /// Family name (`"qwen3.8-vl"`, `"glm5-next"`), for messages.
    fn family(&self) -> &'static str;

    /// A stable identity of everything that changes the model input: the family, the reference profile and every numeric parameter. Part of an
    /// image's cache key (`sched::image_content_hash(pixel_hash, w, h, preprocessing_id)`): bump its version suffix when the semantics change.
    fn id(&self) -> String;

    /// PNG, JPEG or WebP bytes to 8-bit RGB with the family's alpha handling and orientation rules; every other format is refused.
    fn decode(&self, bytes: &[u8]) -> Result<Rgb, VisionError>;

    /// Canvas, grid and placeholder count of a `width x height` image, from the size alone. Fails for images the model cannot take
    /// (too small, absurd aspect, over the pixel limit); the same failures `preprocess` reports, without the pixel work.
    fn plan(&self, width: usize, height: usize) -> Result<ImagePlan, VisionError>;

    /// Resize, normalise and patchify. `plan(img.width, img.height)` describes the result without running this.
    fn preprocess(&self, img: &Rgb) -> Result<Preprocessed, VisionError>;
}

impl ImagePreprocessor for Preprocessing {
    fn family(&self) -> &'static str {
        "qwen3.8-vl"
    }

    fn id(&self) -> String {
        let bits = |a: &[f32; 3]| a.iter().map(|v| format!("{:08x}", v.to_bits())).collect::<Vec<_>>().join(",");
        format!(
            "qwen3.8-vl-v1:p{}m{}t{}:px{}-{}:mean{}:std{}",
            self.patch_size,
            self.merge_size,
            self.temporal_patch_size,
            self.min_pixels,
            self.max_pixels,
            bits(&self.mean),
            bits(&self.std)
        )
    }

    fn decode(&self, bytes: &[u8]) -> Result<Rgb, VisionError> {
        decode::decode(bytes)
    }

    fn plan(&self, width: usize, height: usize) -> Result<ImagePlan, VisionError> {
        let (w, h) = smart_resize(width, height, self)?;
        let (gh, gw) = (h / self.patch_size, w / self.patch_size);
        Ok(ImagePlan { canvas: (w, h), grid: (1, gh, gw), tokens: gh * gw / (self.merge_size * self.merge_size) })
    }

    fn preprocess(&self, img: &Rgb) -> Result<Preprocessed, VisionError> {
        preprocess(img, self)
    }
}
