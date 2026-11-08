//! How many `<|image|>` tokens one image takes in GLM-5.3-Flash, from the image's size alone.
//!
//! The chat template renders `<|begin_of_image|><|image|><|end_of_image|>` per image part, and the HF processor
//! (`Glm5NextProcessor.replace_image_token`) then repeats the single `<|image|>` `image_grid_thw.prod() // merge_size**2` times. The grid
//! comes from `Glm5NextImageProcessor`: the image is resized to a canvas aligned to `patch_size * merge_size` (28) pixels (padded, not
//! stretched) by `smart_resize`, a search that keeps the canvas within a budget of `min_image_tokens..=max_image_tokens` tokens counted
//! with the temporal patch of 2 (one still image fills both temporal slots, so `t = 1`). Nothing here reads pixels; the encoder and the
//! pixel preprocessing are a separate piece of work. `tests/chat_format.rs` checks the count against the real HF image processor (`tests/fixtures/glm53_image_count_cases.json`).
//!
//! The constants come from the pack's `processor_config.json` (`image_processor`): patch 14, merge 2, temporal patch 2, 16 to 8000 tokens.

use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlmImageGeometry {
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
}

impl Default for GlmImageGeometry {
    /// The values `r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121` ships in `processor_config.json`.
    fn default() -> Self {
        Self { patch_size: 14, merge_size: 2, temporal_patch_size: 2, min_image_tokens: 16, max_image_tokens: 8000 }
    }
}

impl GlmImageGeometry {
    /// Read `processor_config.json`'s `image_processor` block; a missing field keeps the default, a wrong type or a zero is an error.
    pub fn from_processor_config(cfg: &Value) -> Result<Self, String> {
        let mut g = Self::default();
        let Some(ip) = cfg.get("image_processor") else { return Ok(g) };
        for (key, slot) in [
            ("patch_size", &mut g.patch_size),
            ("merge_size", &mut g.merge_size),
            ("temporal_patch_size", &mut g.temporal_patch_size),
            ("min_image_tokens", &mut g.min_image_tokens),
            ("max_image_tokens", &mut g.max_image_tokens),
        ] {
            if let Some(v) = ip.get(key) {
                *slot =
                    v.as_u64().filter(|&n| n > 0).ok_or_else(|| {
                        format!("processor_config.json: image_processor.{key} must be a positive integer, got {v}")
                    })? as usize;
            }
        }
        if g.min_image_tokens > g.max_image_tokens {
            return Err("processor_config.json: min_image_tokens exceeds max_image_tokens".into());
        }
        Ok(g)
    }

    /// `Glm5NextImageProcessor`'s `smart_resize` for a still image (`num_frames == temporal_patch_size`): the canvas `(height, width)`
    /// in pixels, a multiple of `patch_size * merge_size` in both directions.
    pub fn canvas(&self, height: usize, width: usize) -> Result<(usize, usize), String> {
        if height == 0 || width == 0 {
            return Err(format!("an image of {width}x{height} pixels has no content"));
        }
        if height > 1 << 24 || width > 1 << 24 {
            return Err(format!("an image of {width}x{height} pixels is far beyond any supported size"));
        }
        let factor = self.patch_size * self.merge_size;
        let temporal = self.temporal_patch_size;
        let per_token = temporal * factor * factor;
        let (min_pixels, max_pixels) = (self.min_image_tokens * per_token, self.max_image_tokens * per_token);
        let align = |v: usize| v.div_ceil(factor) * factor;
        let frames = temporal; // `num_frames` of a still image; round(frames / temporal) * temporal == temporal
        let (mut h, mut w) = (align(height), align(width));
        let mut budget = frames * h * w;
        if budget < min_pixels {
            let scale = (min_pixels as f64 / (frames * height * width) as f64).sqrt();
            h = align(((height as f64 * scale).ceil() as usize).max(1));
            w = align(((width as f64 * scale).ceil() as usize).max(1));
            budget = frames * h * w;
        }
        if budget > max_pixels {
            let minimum = frames * factor * factor;
            if max_pixels < minimum {
                return Err(format!(
                    "max_image_tokens leaves {max_pixels} pixels, fewer than the {minimum} one aligned patch needs"
                ));
            }
            // the largest content height whose aligned canvas fits (binary search over the height, width following the aspect ratio)
            let (mut low, mut high) = (1usize, height);
            (h, w) = (factor, factor);
            while low <= high {
                let content_h = (low + high) / 2;
                let content_w = (((width * content_h) as f64 / height as f64).floor() as usize).max(1);
                let (ch, cw) = (align(content_h), align(content_w));
                if frames * ch * cw <= max_pixels {
                    (h, w) = (ch, cw);
                    low = content_h + 1;
                } else {
                    high = content_h - 1;
                }
            }
        }
        Ok((h, w))
    }

    /// The patch grid `(grid_h, grid_w)` the vision tower sees (`t` is 1 for a still image).
    pub fn grid(&self, height: usize, width: usize) -> Result<(usize, usize), String> {
        let (h, w) = self.canvas(height, width)?;
        Ok((h / self.patch_size, w / self.patch_size))
    }

    /// `<|image|>` tokens the image occupies in the prompt: patches merged `merge_size x merge_size` into one.
    pub fn tokens(&self, height: usize, width: usize) -> Result<usize, String> {
        let (gh, gw) = self.grid(height, width)?;
        Ok(gh * gw / (self.merge_size * self.merge_size))
    }
}
