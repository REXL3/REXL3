//! Video input of Nemotron 3.5 Super VL (`NemotronH_Omni_Reasoning_V3`), on the CPU: what the HF processor and the model do between "a list of
//! frames" and "the rows that replace the placeholders". The reference behaviour, with the evidence for each claim, is `docs/NEMOTRON_VIDEO.md`;
//! this module is its executable form and VS-a's `nemotron.rs` is where a single frame goes through the processor.
//!
//! **What is here**
//!
//! - *Arithmetic* ([`tubelet_count`], [`tubelet_frame_indices`], [`evs_keep_count`], [`square_grid`], [`plan_video`]): tubelets are made by the
//!   MODEL (`forward_video`), not the processor: frames are padded to an even count by repeating the last one and taken two at a time. A
//!   tubelet has the tokens of one frame. The number of rows EVS keeps depends on the frame count, the tile and the rate alone.
//! - *Pixels* ([`VideoBuilder`], [`PreprocessedVideo`], [`patchify_tubelet`]): every frame through `NemotronPreprocessing::preprocess_video_frame`
//!   (VS-a, bit-exact to the processor), rounded to bf16 as `forward_video`'s `.to(dtype)` does, then stacked into the `[6][H][W]` planes the
//!   video patch embedder reads (`[f0 R, f0 G, f0 B, f1 R, f1 G, f1 B]`). **Added here, not in the processor**: the processor emits per-frame
//!   tensors and `forward_video` stacks them; [`PreprocessedVideo::tubelet`] is that stack, [`patchify_tubelet`] the layout `_image_to_patches`
//!   gives it (column `plane * 256 + py * 16 + px`, 1,536 values a patch) and the CPU specification of the device's patchify.
//! - *Labels* ([`frame_labels`]): the processor's `Frame 1 and frame 2: ` prompt labels, with its timestamp rules (`int(1000 / fps)`, the
//!   fallbacks), only when metadata is given.
//! - *EVS* ([`dissimilarity_bf16`], [`retention_mask`], [`retention_mask_bits`]): `compute_retention_mask` of the model, step for step, on the
//!   bf16 `video_embeds`. Torch evaluates `cosine_similarity` on bf16 as a composite (each operation rounds to bf16: the norm, the clamp, the two
//!   divisions, the product, the sum), so this does too, and the fp32 accumulations are done in the order torch 2.14's CPU kernels use on
//!   aarch64 with 8-wide bf16 vectors ([`SumOrder::TorchCpu`]): the result is bit-identical to `torch.nn.functional.cosine_similarity` on that
//!   CPU, and so is the mask (the fixtures of `testdata/nemotron_omni/video_pre` pin it, and the doc gives the measured agreement). Another
//!   device accumulates in another order: where a bf16 rounding sits on a boundary the dissimilarity differs by one bf16 step, which can move a
//!   token across the cut among the many ties (`docs/NEMOTRON_VIDEO.md` section 5 measures how often).
//!
//! **Attribution.** The accumulation orders re-express `vectorized_inner_sum`, `row_sum` and `multi_row_sum` of PyTorch's
//! `aten/src/ATen/native/cpu/SumKernel.cpp` and the 2-norm's last-dimension path of `ReduceOpsKernel.cpp` (BSD-3-Clause, Copyright (c)
//! Facebook, Inc. and its affiliates and the other holders named in its licence; the notice and licence text are in `THIRD_PARTY_NOTICES.md`).
//! The tests and everything else are REXL3's own.

use crate::nemotron::{f32_to_bf16_bits, NemotronPreprocessing, TilePlan, MERGE, PATCH_SIZE};
use crate::{Rgb, VisionError};
use std::cmp::Ordering;
use std::fmt;

/// Frames per tubelet (`video_temporal_patch_size`; `video_temporal_patch_dim` of the processor).
pub const TEMPORAL_PATCH: usize = 2;
/// Colour planes of one frame.
pub const COLOR_PLANES: usize = 3;
/// Planes of a tubelet: the channels `forward_video` reshapes `[G * 2, 3, H, W]` into (`T * C`).
pub const TUBELET_PLANES: usize = TEMPORAL_PATCH * COLOR_PLANES;
/// Values of one video patch: 6 planes of 16 x 16 (`video_embedder.weight` is `[1280, 1536]`).
pub const VIDEO_PATCH_DIM: usize = TUBELET_PLANES * PATCH_SIZE * PATCH_SIZE;
/// Width of one feature row EVS ranks: the language model's hidden size (mlp1's output).
pub const FEATURE_WIDTH: usize = 4096;
/// The 255 the first tubelet's dissimilarity is set to: larger than any `1 - similarity`, exact in bf16.
pub const FIRST_TUBELET_SCORE: f32 = 255.0;

fn bad(m: impl Into<String>) -> VisionError {
    VisionError::Unsupported(m.into())
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Arithmetic.

/// Tubelets of a video of `frames` frames: `ceil(frames / 2)` (`forward_video` pads an odd count).
pub fn tubelet_count(frames: usize) -> usize {
    frames.div_ceil(TEMPORAL_PATCH)
}

/// The two real frames tubelet `g` is made of: `[2g, 2g + 1]`, and for the last tubelet of an odd count `[last, last]` (the padding repeats
/// the last frame: `pixel_values_videos[-1:].expand(...)`).
pub fn tubelet_frame_indices(frames: usize, g: usize) -> [usize; TEMPORAL_PATCH] {
    assert!(frames >= 1 && g < tubelet_count(frames), "tubelet {g} of a video of {frames} frames");
    let last = frames - 1;
    [(TEMPORAL_PATCH * g).min(last), (TEMPORAL_PATCH * g + 1).min(last)]
}

/// How many of `tubelets * tokens` rows EVS keeps at pruning rate `rate`: `max(tokens, int(tubelets * tokens * (1 - q)))` in the model's own
/// order of evaluation (the integer product, then the double `1 - q`; Python's `int` truncates), at most all of them.
pub fn evs_keep_count(tubelets: usize, tokens: usize, rate: f64) -> usize {
    let total = tubelets * tokens;
    let evs = (total as f64 * (1.0 - rate)).trunc();
    let evs = if evs.is_nan() { 0 } else { evs.clamp(0.0, usize::MAX as f64) as usize };
    tokens.max(evs).min(total)
}

/// `int(tokens ** 0.5)` when its square is `tokens`: the grid the model's forward derives (`h = w = int(video_embeds.shape[1] ** 0.5)`) and
/// the only token counts for which `compute_retention_mask`'s reshape does not raise. `None` for the others (252, 247, 240, ...).
pub fn square_grid(tokens: usize) -> Option<usize> {
    let h = (tokens as f64).powf(0.5) as usize;
    (h * h == tokens && tokens > 0).then_some(h)
}

/// What a pruned video's prompt holds, per tubelet: the kept counts of a mask over `tubelets * tokens` rows.
pub fn kept_per_tubelet(mask: &[bool], tubelets: usize, tokens: usize) -> Vec<usize> {
    assert_eq!(mask.len(), tubelets * tokens, "mask of {} rows for {tubelets} tubelets of {tokens}", mask.len());
    mask.chunks(tokens.max(1)).take(tubelets).map(|c| c.iter().filter(|&&k| k).count()).collect()
}

/// The video settings the server chooses (not part of the pack's processor): the pruning rate and what to do with a tile EVS cannot take.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoConfig {
    /// `video_pruning_rate`; 0 is off (the model then never calls `compute_retention_mask`).
    pub pruning_rate: f64,
    /// The pack's `forward` raises when the tile's token count is not a perfect square (`square_grid`). `true` refuses such a video by name when
    /// pruning is on; `false` (default) computes EVS on the tile's true grid, which `compute_retention_mask` itself takes (only `H * W` matters
    /// with `spatial_merge_size = 1`), and for which the reference has no run to compare with.
    pub require_square_grid: bool,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self { pruning_rate: 0.0, require_square_grid: false }
    }
}

impl VideoConfig {
    /// Whether EVS runs at all.
    pub fn pruning(&self) -> bool {
        self.pruning_rate > 0.0
    }

    /// `Err` names what is out of range.
    pub fn validate(&self) -> Result<(), String> {
        if !self.pruning_rate.is_finite() || !(0.0..=1.0).contains(&self.pruning_rate) {
            return Err(format!("video pruning rate {} is not in [0, 1]", self.pruning_rate));
        }
        Ok(())
    }

    /// Part of a video's cache key: everything here that changes the rows the model sees.
    pub fn identity(&self) -> String {
        format!("video-v1:t{}:q{:016x}", TEMPORAL_PATCH, self.pruning_rate.to_bits())
    }
}

/// What a video of `frames` frames of `width` x `height` pixels takes, from the sizes alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoPlan {
    /// The tile every frame is resized to (all frames of a video share it).
    pub tile: TilePlan,
    pub frames: usize,
    pub tubelets: usize,
    /// Placeholder rows of one tubelet: the tile's tokens (not doubled by the two frames).
    pub tokens_per_tubelet: usize,
    /// Rows of the video before pruning: `tubelets * tokens_per_tubelet`.
    pub tokens: usize,
    /// Rows left after EVS at the configured rate (equal to `tokens` when it is off or there is one tubelet).
    pub kept_tokens: usize,
}

/// The plan of a video, with the tile rule of the video path ([`NemotronPreprocessing::plan_video`]). Errors name what Python raises: no
/// frames, a tile the tower cannot run, and (with `require_square_grid`) a pruned video whose tile the model's EVS could not reshape.
pub fn plan_video(
    pre: &NemotronPreprocessing,
    cfg: &VideoConfig,
    width: usize,
    height: usize,
    frames: usize,
) -> Result<VideoPlan, VisionError> {
    cfg.validate().map_err(bad)?;
    if frames == 0 {
        return Err(bad("a video needs at least one frame"));
    }
    let tile = pre.plan_video(width, height)?;
    let tubelets = tubelet_count(frames);
    let tokens_per_tubelet = tile.tokens;
    if cfg.pruning() && cfg.require_square_grid && square_grid(tokens_per_tubelet).is_none() {
        return Err(bad(format!(
            "pruning is on and the {}x{} patch tile has {tokens_per_tubelet} tokens a tubelet: the model's EVS reshapes each tubelet as \
             int(sqrt(tokens))^2 = {}, which raises for a count that is not a perfect square",
            tile.w_patches,
            tile.h_patches,
            ((tokens_per_tubelet as f64).powf(0.5) as usize).pow(2)
        )));
    }
    let tokens = tubelets * tokens_per_tubelet;
    let kept_tokens = if cfg.pruning() { evs_keep_count(tubelets, tokens_per_tubelet, cfg.pruning_rate) } else { tokens };
    Ok(VideoPlan { tile, frames, tubelets, tokens_per_tubelet, tokens, kept_tokens })
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Pixels.

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

fn fnv(h: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(FNV_PRIME))
}

/// The content hash of a video: FNV-1a over a marker, the tubelet width, the frame count, the canvas and every frame's canvas hash in order. A
/// still image of the same pixels never has this value (different marker and different structure), so the two never share a cache key.
pub fn video_hash(frame_hashes: &[u64], canvas: (usize, usize)) -> u64 {
    let mut h = fnv(FNV_OFFSET, b"nemotron-video-v1");
    h = fnv(h, &(TEMPORAL_PATCH as u64).to_le_bytes());
    h = fnv(h, &(frame_hashes.len() as u64).to_le_bytes());
    h = fnv(h, &(canvas.0 as u64).to_le_bytes());
    h = fnv(h, &(canvas.1 as u64).to_le_bytes());
    for f in frame_hashes {
        h = fnv(h, &f.to_le_bytes());
    }
    h
}

/// The frames of a video through the processor's video rule, in bf16 (what `forward_video` feeds the tower), with their canvas hashes.
#[derive(Clone, Debug, PartialEq)]
pub struct PreprocessedVideo {
    pub tile: TilePlan,
    /// Per real frame (no padding): bf16 bits, channel planes `[3][H][W]`.
    pub frames: Vec<Vec<u16>>,
    /// `canvas_hash` of each frame's float32 canvas (`Preprocessed::pixel_hash`).
    pub frame_hashes: Vec<u64>,
    /// [`video_hash`] of the frames.
    pub pixel_hash: u64,
}

impl PreprocessedVideo {
    /// Canvas `(width, height)` in pixels.
    pub fn canvas(&self) -> (usize, usize) {
        self.tile.canvas()
    }

    pub fn tubelets(&self) -> usize {
        tubelet_count(self.frames.len())
    }

    /// bf16 bits of tubelet `g`, planes `[6][H][W]`: frame `2g`'s three planes then frame `2g + 1`'s (the last frame twice when padded).
    pub fn tubelet(&self, g: usize) -> Vec<u16> {
        let [a, b] = tubelet_frame_indices(self.frames.len(), g);
        let mut out = Vec::with_capacity(self.frames[a].len() + self.frames[b].len());
        out.extend_from_slice(&self.frames[a]);
        out.extend_from_slice(&self.frames[b]);
        out
    }

    /// `forward_video`'s `x`: all tubelets, `[G][6][H][W]`.
    pub fn stack(&self) -> Vec<u16> {
        (0..self.tubelets()).flat_map(|g| self.tubelet(g)).collect()
    }
}

/// Frames in one at a time (so that a decoded frame can be dropped as soon as it is preprocessed), the video out.
pub struct VideoBuilder<'a> {
    pre: &'a NemotronPreprocessing,
    tile: Option<TilePlan>,
    frames: Vec<Vec<u16>>,
    hashes: Vec<u64>,
}

impl<'a> VideoBuilder<'a> {
    pub fn new(pre: &'a NemotronPreprocessing) -> Self {
        Self { pre, tile: None, frames: Vec::new(), hashes: Vec::new() }
    }

    /// Frames added so far.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The tile `img` would be resized to, checked against the first frame's: the model takes one tile for the whole video (the processor uses
    /// the first frame's token count for every tubelet, and `forward_video` cannot take frames of different sizes). Nothing is stored.
    pub fn check(&self, img: &Rgb) -> Result<TilePlan, VisionError> {
        let t = self.pre.plan_video(img.width, img.height)?;
        if let Some(t0) = self.tile {
            if (t.w_patches, t.h_patches) != (t0.w_patches, t0.h_patches) {
                return Err(bad(format!(
                    "frame {}: {}x{} resolves to a {}x{} patch tile, frame 1 to {}x{}: every frame of a video must resolve to one tile",
                    self.frames.len() + 1,
                    img.width,
                    img.height,
                    t.w_patches,
                    t.h_patches,
                    t0.w_patches,
                    t0.h_patches
                )));
            }
        }
        Ok(t)
    }

    /// Preprocess one more frame (the processor's video rule: tile, antialiased bicubic when the size differs, normalisation, bf16).
    pub fn push(&mut self, img: &Rgb) -> Result<(), VisionError> {
        let t = self.check(img)?;
        let p = self.pre.preprocess_video_frame(img)?;
        debug_assert_eq!((p.w_patches, p.h_patches), (t.w_patches, t.h_patches));
        self.tile.get_or_insert(t);
        self.frames.push(p.bf16_bits());
        self.hashes.push(p.pixel_hash);
        Ok(())
    }

    pub fn finish(self) -> Result<PreprocessedVideo, VisionError> {
        let tile = self.tile.ok_or_else(|| bad("a video needs at least one frame"))?;
        let pixel_hash = video_hash(&self.hashes, tile.canvas());
        Ok(PreprocessedVideo { tile, frames: self.frames, frame_hashes: self.hashes, pixel_hash })
    }
}

/// Every frame through the video rule: [`VideoBuilder`] over a slice.
pub fn preprocess_video(pre: &NemotronPreprocessing, frames: &[Rgb]) -> Result<PreprocessedVideo, VisionError> {
    let mut b = VideoBuilder::new(pre);
    for f in frames {
        b.push(f)?;
    }
    b.finish()
}

/// `RadioPatchEmbeddings._image_to_patches` of a tubelet with six planes: `planes` is `[6][height][width]`, the result `[rows * cols][1536]`
/// with row `r * cols + c` the patch at `(r, c)` and column `plane * 256 + py * 16 + px`. The CPU specification of the device's patchify (the
/// image path's is the same with three planes and 768 columns).
pub fn patchify_tubelet<T: Copy>(planes: &[T], height: usize, width: usize) -> Vec<T> {
    assert!(height % PATCH_SIZE == 0 && width % PATCH_SIZE == 0, "{width}x{height} is not a whole number of patches");
    assert_eq!(planes.len(), TUBELET_PLANES * height * width, "planes of a {width}x{height} tubelet");
    let (rows, cols) = (height / PATCH_SIZE, width / PATCH_SIZE);
    let mut out = Vec::with_capacity(planes.len());
    for r in 0..rows {
        for c in 0..cols {
            for plane in 0..TUBELET_PLANES {
                for py in 0..PATCH_SIZE {
                    let at = plane * height * width + (r * PATCH_SIZE + py) * width + c * PATCH_SIZE;
                    out.extend_from_slice(&planes[at..at + PATCH_SIZE]);
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Labels.

/// What the caller knows about where the frames came from (`video_metadata`: `fps` and `frames_indices` of the processor's `VideoMetadata`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VideoMetadata {
    pub fps: Option<f64>,
    /// The index of each sampled frame in the source video (needs `fps` to have an effect, as in the processor).
    pub frame_indices: Option<Vec<i64>>,
}

/// The label of each tubelet's `<img>` chunk, `"Frame 1 and frame 2: "`: the processor's loop. `"Frame"` for the first frame of a tubelet and
/// `"frame"` for the second, the 1-based frame number (the padding frame has none), and with `fps` ` sampled at {ts:.2f} seconds` where
/// `ts = int(frame_indices[i]) * int(1000.0 / fps) / 1000.0` when `frame_indices` has an entry for frame `i` and `ts = i / fps` otherwise. A
/// missing, zero or absent `fps` gives plain labels, as the processor's `source_fps = ... or None` does.
pub fn frame_labels(frames: usize, meta: &VideoMetadata) -> Result<Vec<String>, VisionError> {
    let fps = meta.fps.filter(|&f| f != 0.0);
    let frame_ms = match fps {
        Some(f) => {
            let ms = (1000.0 / f).trunc();
            if !ms.is_finite() || ms.abs() > 9e15 {
                return Err(bad(format!("fps {f} gives no whole frame duration in milliseconds")));
            }
            Some(ms as i64)
        }
        None => None,
    };
    let mut out = Vec::with_capacity(tubelet_count(frames));
    for g in 0..tubelet_count(frames) {
        let mut parts = Vec::with_capacity(TEMPORAL_PATCH);
        for j in 0..TEMPORAL_PATCH {
            let fi = g * TEMPORAL_PATCH + j;
            if fi >= frames {
                break;
            }
            let prefix = if j == 0 { "Frame" } else { "frame" };
            let label = match (fps, frame_ms) {
                (Some(f), Some(ms)) => {
                    let idx = meta.frame_indices.as_ref().and_then(|v| v.get(fi));
                    let ts = match idx {
                        Some(&i) => (i as i128 * ms as i128) as f64 / 1000.0,
                        None => fi as f64 / f,
                    };
                    format!("{prefix} {} sampled at {ts:.2} seconds", fi + 1)
                }
                _ => format!("{prefix} {}", fi + 1),
            };
            parts.push(label);
        }
        out.push(format!("{}: ", parts.join(" and ")));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// EVS: `compute_retention_mask` of `modeling_nemotron_h_omni.py`.

/// The order in which fp32 accumulations of the bf16 composite are done.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SumOrder {
    /// torch 2.14 CPU on aarch64 (8-wide bf16 vectors, 4-lane fp32): the 2-norm accumulates two fused multiply-adds per vector into one 4-lane
    /// accumulator and sums the lanes in order; `sum` is `vectorized_inner_sum` (the vector's two float halves added, four rows, four cascade
    /// levels of 16, the scalar tail first). Bit-identical to torch's results there for hidden sizes that are a multiple of 8 (tails: see tests).
    #[default]
    TorchCpu,
    /// Plain left to right in fp32, no fusion: the simplest order and the one a naive port makes. For measuring how often the order matters.
    Sequential,
}

/// What EVS refuses.
#[derive(Debug, PartialEq, Eq)]
pub struct EvsError(pub String);

impl fmt::Display for EvsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EVS: {}", self.0)
    }
}

impl std::error::Error for EvsError {}

/// bf16 bits to float32 (exact).
#[inline]
pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// float32 rounded to the nearest bf16 (ties to even) and back.
#[inline]
fn rbf(v: f32) -> f32 {
    bf16_bits_to_f32(f32_to_bf16_bits(v))
}

const LANES: usize = 4; // fp32 lanes of one accumulator vector
const VEC: usize = 2 * LANES; // bf16 elements of one vector load

/// `c10::utils::CeilLog2`.
fn ceil_log2(x: usize) -> usize {
    if x <= 2 {
        1
    } else {
        (usize::BITS - (x - 1).leading_zeros()) as usize
    }
}

/// Sum of squares as torch's CPU 2-norm accumulates it on a bf16 row (the last-dimension fast path of `norm_kernel_tensor_iterator_impl`).
fn sumsq(x: &[f32], order: SumOrder) -> f32 {
    match order {
        SumOrder::Sequential => x.iter().fold(0.0f32, |s, &v| s + v * v),
        SumOrder::TorchCpu => {
            let body = x.len() - x.len() % VEC;
            let mut acc = [0f32; LANES];
            for blk in x[..body].chunks_exact(VEC) {
                for l in 0..LANES {
                    acc[l] = blk[l].mul_add(blk[l], acc[l]);
                }
                for l in 0..LANES {
                    acc[l] = blk[LANES + l].mul_add(blk[LANES + l], acc[l]);
                }
            }
            let mut s = acc[0];
            for &a in &acc[1..] {
                s += a;
            }
            for &v in &x[body..] {
                s += v * v;
            }
            s
        }
    }
}

/// `acc += v`, lane by lane in fp32.
#[inline]
fn add_lanes(acc: &mut [f32; LANES], v: &[f32; LANES]) {
    for (a, x) in acc.iter_mut().zip(v) {
        *a += x;
    }
}

/// Sum of a bf16 row's values (already rounded products) in fp32 in torch's `vectorized_inner_sum` order.
fn sum_row(p: &[f32], order: SumOrder) -> f32 {
    const ILP: usize = 4;
    const LEVELS: usize = 4;
    let n = p.len();
    if order == SumOrder::Sequential || n < VEC {
        return p.iter().fold(0.0f32, |s, &v| s + v);
    }
    let loads = n / VEC;
    // one vector load: the bf16 vector widened to two float vectors, added lane-wise
    let load = |s: usize| -> [f32; LANES] {
        let b = &p[s * VEC..(s + 1) * VEC];
        let mut v = [0f32; LANES];
        for l in 0..LANES {
            v[l] = b[l] + b[LANES + l];
        }
        v
    };
    let size = loads / ILP; // rows of ILP loads
    let mut acc = [[[0f32; LANES]; ILP]; LEVELS];
    if size > 0 {
        let level_power = 4.max(ceil_log2(size) / LEVELS);
        let level_step = 1usize << level_power;
        let level_mask = level_step - 1;
        let mut i = 0usize;
        while i + level_step <= size {
            for _ in 0..level_step {
                for k in 0..ILP {
                    add_lanes(&mut acc[0][k], &load(i * ILP + k));
                }
                i += 1;
            }
            for j in 1..LEVELS {
                for k in 0..ILP {
                    for l in 0..LANES {
                        acc[j][k][l] += acc[j - 1][k][l];
                        acc[j - 1][k][l] = 0.0;
                    }
                }
                if (i & (level_mask << (j * level_power))) != 0 {
                    break;
                }
            }
        }
        while i < size {
            for k in 0..ILP {
                add_lanes(&mut acc[0][k], &load(i * ILP + k));
            }
            i += 1;
        }
        for j in 1..LEVELS {
            for k in 0..ILP {
                for l in 0..LANES {
                    acc[0][k][l] += acc[j][k][l];
                }
            }
        }
    }
    let mut part = acc[0];
    for s in size * ILP..loads {
        add_lanes(&mut part[0], &load(s));
    }
    for k in 1..ILP {
        for l in 0..LANES {
            part[0][l] += part[k][l];
        }
    }
    let mut fin = 0f32;
    for &v in &p[loads * VEC..] {
        fin += v;
    }
    for l in 0..LANES {
        fin += part[0][l];
    }
    fin
}

/// `torch.nn.functional.cosine_similarity(x1, x2, dim=-1)` of bf16 tensors, for the `tokens` rows of one tubelet against the rows of the
/// previous one, as bf16 values in float32: `x / max(bf16(||x||), bf16(1e-8))` per operand (bf16 each), the product (bf16), the sum (bf16).
struct Cosine {
    hidden: usize,
    order: SumOrder,
    eps: f32,
}

impl Cosine {
    fn new(hidden: usize, order: SumOrder) -> Self {
        Self { hidden, order, eps: rbf(1e-8) }
    }

    /// `x / clamp_min(norm(x), eps)` of every row of a tubelet, in bf16.
    fn normalise(&self, rows: &[f32], out: &mut [f32]) {
        for (x, y) in rows.chunks_exact(self.hidden).zip(out.chunks_exact_mut(self.hidden)) {
            // `clamp_min_` propagates a NaN (f32::max would return the other operand)
            let norm = rbf(sumsq(x, self.order).sqrt());
            let norm = if norm.is_nan() { norm } else { norm.max(self.eps) };
            for (o, &v) in y.iter_mut().zip(x) {
                *o = rbf(v / norm);
            }
        }
    }

    /// The bf16 similarity of a row of normalised values with another.
    fn dot(&self, a: &[f32], b: &[f32], scratch: &mut [f32]) -> f32 {
        for ((s, &x), &y) in scratch.iter_mut().zip(a).zip(b) {
            *s = rbf(x * y);
        }
        rbf(sum_row(scratch, self.order))
    }
}

/// The dissimilarity `compute_retention_mask` ranks by, as bf16 bits: `[t][n]` for `embeds` `[t][n][hidden]` (bf16 bits). The first tubelet's
/// scores are 255 and tubelet `g >= 1`'s token `i` is `bf16(1 - cos(embeds[g][i], embeds[g - 1][i]))`.
pub fn dissimilarity_bf16(embeds: &[u16], t: usize, n: usize, hidden: usize, order: SumOrder) -> Vec<u16> {
    assert_eq!(embeds.len(), t * n * hidden, "embeddings of {t} x {n} x {hidden}");
    assert!(hidden > 0, "hidden size 0");
    let cos = Cosine::new(hidden, order);
    let mut out = vec![0u16; t * n];
    let first = f32_to_bf16_bits(FIRST_TUBELET_SCORE);
    out[..n].fill(first);
    if t < 2 {
        return out;
    }
    let row = n * hidden;
    let mut raw = vec![0f32; row];
    let mut prev = vec![0f32; row];
    let mut cur = vec![0f32; row];
    let mut scratch = vec![0f32; hidden];
    for (r, &b) in raw.iter_mut().zip(&embeds[..row]) {
        *r = bf16_bits_to_f32(b);
    }
    cos.normalise(&raw, &mut prev);
    for g in 1..t {
        for (r, &b) in raw.iter_mut().zip(&embeds[g * row..(g + 1) * row]) {
            *r = bf16_bits_to_f32(b);
        }
        cos.normalise(&raw, &mut cur);
        for i in 0..n {
            let s = cos.dot(&cur[i * hidden..(i + 1) * hidden], &prev[i * hidden..(i + 1) * hidden], &mut scratch);
            out[g * n + i] = f32_to_bf16_bits(1.0 - s);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    out
}

/// `torch.argsort(d, descending=True, stable=True)[:keep]` as a mask: larger first, ties by lower index, NaN larger than any number.
fn top_mask(d: &[f32], keep: usize) -> Vec<bool> {
    let mut idx: Vec<u32> = (0..d.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        let (x, y) = (d[a as usize], d[b as usize]);
        match (x.is_nan(), y.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => y.partial_cmp(&x).unwrap_or(Ordering::Equal),
        }
    });
    let mut mask = vec![false; d.len()];
    for &i in idx.iter().take(keep) {
        mask[i as usize] = true;
    }
    mask
}

/// `compute_retention_mask(video_embeds, thw=(t, h, w), spatial_merge_size=1, q=rate)` on bf16 bits `[t][h * w][hidden]`: `t * h * w` flags,
/// tubelet-major, true for a kept row. Errors when the buffer does not have `t * h * w * hidden` values (where torch's `reshape` raises).
pub fn retention_mask_bits(
    embeds: &[u16],
    t: usize,
    h: usize,
    w: usize,
    hidden: usize,
    rate: f64,
    order: SumOrder,
) -> Result<Vec<bool>, EvsError> {
    let n = h * w;
    if hidden == 0 || embeds.len() != t * n * hidden {
        return Err(EvsError(format!(
            "{} values cannot be reshaped to [{t}, {h}, {w}, {hidden}] ({} needed)",
            embeds.len(),
            t * n * hidden
        )));
    }
    if t == 0 {
        return Ok(Vec::new());
    }
    let d: Vec<f32> = dissimilarity_bf16(embeds, t, n, hidden, order).into_iter().map(bf16_bits_to_f32).collect();
    Ok(top_mask(&d, evs_keep_count(t, n, rate)))
}

/// [`retention_mask_bits`] with the model's default: torch's CPU accumulation order, `embeds` float32 holding bf16 values (the model's
/// `vit_embeds` are bf16; a value that is not exactly representable is rounded to the nearest bf16 first). Panics on a wrong length.
pub fn retention_mask(embeds: &[f32], t: usize, h: usize, w: usize, hidden: usize, rate: f64) -> Vec<bool> {
    let bits: Vec<u16> = embeds.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    retention_mask_bits(&bits, t, h, w, hidden, rate, SumOrder::TorchCpu).unwrap_or_else(|e| panic!("{e}"))
}

/// The pixel shuffle's merge factor, for the callers that turn a tile into the `(rows, cols)` grid of its tokens.
pub const fn merged_grid(h_patches: usize, w_patches: usize) -> (usize, usize) {
    (h_patches / MERGE, w_patches / MERGE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> NemotronPreprocessing {
        NemotronPreprocessing::default()
    }

    /// A deterministic float stream in [-1, 1).
    fn stream(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32) / (1u64 << 23) as f32 - 1.0
        }
    }

    fn bits(v: &[f32]) -> Vec<u16> {
        v.iter().map(|&x| f32_to_bf16_bits(x)).collect()
    }

    #[test]
    fn tubelets_pad_the_last_frame_and_a_tubelet_has_one_frames_tokens() {
        assert_eq!((1..=7).map(tubelet_count).collect::<Vec<_>>(), [1, 1, 2, 2, 3, 3, 4]);
        assert_eq!(tubelet_frame_indices(1, 0), [0, 0]);
        assert_eq!(tubelet_frame_indices(2, 0), [0, 1]);
        assert_eq!([0, 1, 2].map(|g| tubelet_frame_indices(5, g)), [[0, 1], [2, 3], [4, 4]]);
        assert_eq!([0, 1].map(|g| tubelet_frame_indices(4, g)), [[0, 1], [2, 3]]);
        let p = plan_video(&cfg(), &VideoConfig::default(), 1920, 1080, 32).unwrap();
        assert_eq!((p.tile.w_patches, p.tile.h_patches, p.tokens_per_tubelet, p.tubelets, p.tokens), (42, 24, 252, 16, 4032));
        assert_eq!(p.kept_tokens, p.tokens, "EVS off");
        let sq = plan_video(&cfg(), &VideoConfig::default(), 512, 512, 5).unwrap();
        assert_eq!((sq.tokens_per_tubelet, sq.tubelets, sq.tokens), (256, 3, 768));
        assert_eq!(plan_video(&cfg(), &VideoConfig::default(), 640, 480, 1).unwrap().tubelets, 1);
        assert!(plan_video(&cfg(), &VideoConfig::default(), 640, 480, 0).is_err(), "no frames");
        assert!(plan_video(&cfg(), &VideoConfig::default(), 0, 480, 4).is_err(), "no width");
    }

    #[test]
    fn the_keep_count_is_the_models_expression() {
        // int(T * N * (1 - q)), the double 1 - 0.7 = 0.30000000000000004 included
        assert_eq!(evs_keep_count(16, 252, 0.7), 1209);
        assert_eq!(evs_keep_count(3, 256, 0.7), 256, "int(230.4) is below one tubelet: the first tubelet is the floor");
        assert_eq!(evs_keep_count(10, 100, 0.7), 300);
        assert_eq!(evs_keep_count(8, 256, 0.5), 1024);
        assert_eq!(
            evs_keep_count(10, 10, 0.9),
            10,
            "1 - 0.9 is 0.09999999999999998 and int(9.999999999999998) is 9, below one tubelet"
        );
        // 1000 * 0.09999999999999998 is 99.99999999999997: Python's int() gives 99, not the 100 of exact arithmetic
        assert_eq!(evs_keep_count(100, 10, 0.9), 99);
        assert_eq!(evs_keep_count(4, 9, 1.0), 9, "rate 1 keeps the first tubelet only");
        assert_eq!(evs_keep_count(4, 9, 0.0), 36);
        assert_eq!(evs_keep_count(1, 256, 0.7), 256);
        // the plan carries it
        let on = VideoConfig { pruning_rate: 0.7, ..Default::default() };
        assert_eq!(plan_video(&cfg(), &on, 1920, 1080, 32).unwrap().kept_tokens, 1209);
    }

    #[test]
    fn the_square_rule_is_the_models_reshape() {
        assert_eq!(square_grid(256), Some(16));
        assert_eq!(square_grid(225), Some(15));
        for n in [252, 247, 240, 242, 260, 255, 257] {
            assert_eq!(square_grid(n), None, "{n}");
        }
        assert_eq!(square_grid(0), None);
        let strict = VideoConfig { pruning_rate: 0.7, require_square_grid: true };
        let e = plan_video(&cfg(), &strict, 1920, 1080, 4).unwrap_err().to_string();
        assert!(e.contains("252 tokens") && e.contains("perfect square"), "{e}");
        assert!(plan_video(&cfg(), &strict, 512, 512, 4).is_ok());
        // strict only matters with pruning on
        assert!(plan_video(&cfg(), &VideoConfig { pruning_rate: 0.0, require_square_grid: true }, 1920, 1080, 4).is_ok());
        assert!(VideoConfig { pruning_rate: 1.5, ..Default::default() }.validate().is_err());
        assert!(VideoConfig { pruning_rate: f64::NAN, ..Default::default() }.validate().is_err());
        assert!(VideoConfig { pruning_rate: -0.1, ..Default::default() }.validate().is_err());
    }

    #[test]
    fn a_tubelet_is_two_frames_of_planes_and_patches_are_plane_major() {
        let (h, w) = (32usize, 48usize);
        let frame = |f: u16| (0..3 * h * w).map(|i| f * 10_000 + i as u16).collect::<Vec<u16>>();
        let v = PreprocessedVideo {
            tile: TilePlan { w_patches: 3, h_patches: 2, tokens: 1 },
            frames: (0..5).map(frame).collect(),
            frame_hashes: vec![0; 5],
            pixel_hash: 0,
        };
        assert_eq!(v.tubelets(), 3);
        let t1 = v.tubelet(1);
        assert_eq!((t1.len(), t1[0], t1[3 * h * w]), (6 * h * w, 2 * 10_000, 3 * 10_000));
        // the padded tubelet is [last, last]
        let t2 = v.tubelet(2);
        assert_eq!(&t2[..3 * h * w], &t2[3 * h * w..]);
        assert_eq!(t2[0], 4 * 10_000);
        assert_eq!(v.stack().len(), 3 * 6 * h * w);
        // patch (r, c), column plane * 256 + py * 16 + px
        let p = patchify_tubelet(&t1, h, w);
        assert_eq!(p.len(), 6 * h * w);
        let (r, c, plane, py, px) = (1usize, 2usize, 4usize, 5usize, 9usize);
        assert_eq!(
            p[(r * 3 + c) * VIDEO_PATCH_DIM + plane * 256 + py * 16 + px],
            t1[plane * h * w + (r * 16 + py) * w + c * 16 + px]
        );
        // every value lands exactly once
        let mut a = p.clone();
        let mut b = t1.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }

    #[test]
    fn labels_are_the_processors() {
        let none = VideoMetadata::default();
        assert_eq!(frame_labels(5, &none).unwrap(), ["Frame 1 and frame 2: ", "Frame 3 and frame 4: ", "Frame 5: "]);
        assert_eq!(frame_labels(1, &none).unwrap(), ["Frame 1: "]);
        assert_eq!(frame_labels(2, &none).unwrap(), ["Frame 1 and frame 2: "]);
        // 30 fps is 33 ms a frame: index 15 is 0.495 s printed 0.49, 30 is 0.99 s
        let m = VideoMetadata { fps: Some(30.0), frame_indices: Some(vec![0, 15, 30, 45, 60]) };
        assert_eq!(
            frame_labels(5, &m).unwrap(),
            [
                "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.49 seconds: ",
                "Frame 3 sampled at 0.99 seconds and frame 4 sampled at 1.49 seconds: ",
                "Frame 5 sampled at 1.98 seconds: "
            ]
        );
        // 8 fps is 125 ms: 0.125 prints 0.12 and 0.375 prints 0.38 (ties to even, as Python)
        let m = VideoMetadata { fps: Some(8.0), frame_indices: Some(vec![0, 1, 2, 3, 4]) };
        assert_eq!(
            frame_labels(5, &m).unwrap(),
            [
                "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.12 seconds: ",
                "Frame 3 sampled at 0.25 seconds and frame 4 sampled at 0.38 seconds: ",
                "Frame 5 sampled at 0.50 seconds: "
            ]
        );
        // fps without indices: i / fps
        let m = VideoMetadata { fps: Some(30.0), frame_indices: None };
        assert_eq!(frame_labels(5, &m).unwrap()[0], "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 0.03 seconds: ");
        // a short index list falls back for the missing frames (2.5 fps is 400 ms)
        let m = VideoMetadata { fps: Some(2.5), frame_indices: Some(vec![0, 3]) };
        assert_eq!(
            frame_labels(5, &m).unwrap(),
            [
                "Frame 1 sampled at 0.00 seconds and frame 2 sampled at 1.20 seconds: ",
                "Frame 3 sampled at 0.80 seconds and frame 4 sampled at 1.20 seconds: ",
                "Frame 5 sampled at 1.60 seconds: "
            ]
        );
        // indices without fps, or a zero fps, are ignored
        let m = VideoMetadata { fps: None, frame_indices: Some(vec![0, 1, 2, 3, 4]) };
        assert_eq!(frame_labels(3, &m).unwrap(), ["Frame 1 and frame 2: ", "Frame 3: "]);
        assert_eq!(frame_labels(2, &VideoMetadata { fps: Some(0.0), frame_indices: None }).unwrap(), ["Frame 1 and frame 2: "]);
        assert!(frame_labels(2, &VideoMetadata { fps: Some(f64::NAN), frame_indices: None }).is_err());
    }

    #[test]
    fn rounding_and_the_torch_order_are_what_they_claim() {
        // bf16 round to nearest even
        assert_eq!(rbf(1.0), 1.0);
        assert_eq!(rbf(f32::from_bits(0x3f80_8000)), 1.0, "tie to even");
        assert_eq!(rbf(f32::from_bits(0x3f81_8000)), f32::from_bits(0x3f82_0000));
        assert_eq!(f32_to_bf16_bits(255.0), 0x437f, "255 is exact");
        assert_eq!(ceil_log2(1), 1);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(4), 2);
        assert_eq!(ceil_log2(5), 3);
        assert_eq!(ceil_log2(128), 7);
        assert_eq!(ceil_log2(129), 8);
        // a sum of exactly representable small integers is the same in every order
        let p: Vec<f32> = (0..4096).map(|i| (i % 7) as f32 - 3.0).collect();
        assert_eq!(sum_row(&p, SumOrder::TorchCpu), sum_row(&p, SumOrder::Sequential));
        assert_eq!(sumsq(&p, SumOrder::TorchCpu), sumsq(&p, SumOrder::Sequential));
        // a tail (hidden 12) and a short row (hidden 5) are summed too
        for n in [5usize, 8, 12, 20, 36, 100, 4100] {
            let p: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
            let want: f32 = p.iter().sum();
            assert_eq!(sum_row(&p, SumOrder::TorchCpu), want, "hidden {n}");
        }
    }

    #[test]
    fn the_first_tubelet_is_kept_and_a_tie_keeps_the_lower_index() {
        // two identical tubelets: every dissimilarity is the same value, so the second tubelet's score is a tie everywhere
        let (t, n, hidden) = (3usize, 6usize, 16usize);
        let mut rnd = stream(1);
        let one: Vec<f32> = (0..n * hidden).map(|_| rnd()).collect();
        let e: Vec<f32> = (0..t).flat_map(|_| one.clone()).collect();
        let d = dissimilarity_bf16(&bits(&e), t, n, hidden, SumOrder::TorchCpu);
        assert!(d[..n].iter().all(|&b| b == f32_to_bf16_bits(255.0)));
        let second: std::collections::BTreeSet<u16> = d[n..].iter().copied().collect();
        assert!(second.len() <= 3, "identical tubelets score alike: {second:?}");
        // keep = max(6, int(18 * 0.3)) = 6: the first tubelet only
        let m = retention_mask(&e, t, 2, 3, hidden, 0.7);
        assert_eq!(m.iter().filter(|&&k| k).count(), 6);
        assert!(m[..n].iter().all(|&k| k));
        // rate 0.3: keep int(18 * 0.7) = 12: the first tubelet and the 6 lowest indices of the tied rest, i.e. all of tubelet 1 when the scores are equal
        let m = retention_mask(&e, t, 2, 3, hidden, 0.3);
        assert_eq!(m.iter().filter(|&&k| k).count(), 12);
        if d[n..2 * n].iter().all(|&b| b == d[n]) && d[2 * n..].iter().all(|&b| b == d[n]) {
            assert!(m[n..2 * n].iter().all(|&k| k) && m[2 * n..].iter().all(|&k| !k), "{m:?}");
        }
    }

    #[test]
    fn the_changed_tokens_are_kept_over_the_unchanged() {
        // tubelet 1 changes tokens 2 and 5 completely and copies the rest of tubelet 0
        let (n, hidden) = (8usize, 32usize);
        let mut rnd = stream(7);
        let a: Vec<f32> = (0..n * hidden).map(|_| rnd()).collect();
        let mut b = a.clone();
        for tok in [2usize, 5] {
            for k in 0..hidden {
                b[tok * hidden + k] = -a[tok * hidden + k] + 0.1 * rnd();
            }
        }
        let e: Vec<f32> = a.iter().chain(&b).copied().collect();
        // keep = max(8, int(16 * 0.5)) = 8 would be the first tubelet only; rate 0.25 keeps int(12) = 12: 8 + the 4 most changed
        let m = retention_mask(&e, 2, 2, 4, hidden, 0.25);
        assert_eq!(m.iter().filter(|&&k| k).count(), 12);
        assert!(m[8 + 2] && m[8 + 5], "the changed tokens");
        // the first tubelet whole, then the 4 highest scores of the rest with the lower index first among equals: an independent selection
        let d: Vec<f32> =
            dissimilarity_bf16(&bits(&e), 2, 8, hidden, SumOrder::TorchCpu).into_iter().map(bf16_bits_to_f32).collect();
        let mut order: Vec<usize> = (0..16).collect();
        order.sort_by(|&x, &y| d[y].partial_cmp(&d[x]).unwrap().then(x.cmp(&y)));
        let mut want = vec![false; 16];
        for &i in &order[..12] {
            want[i] = true;
        }
        assert_eq!(m, want);
        assert!(m[..8].iter().all(|&k| k));
    }

    #[test]
    fn a_nan_outranks_the_first_tubelet_and_a_zero_vector_scores_zero_similarity() {
        // NaN is the largest value of the sort: with keep == one tubelet, it displaces the last first-tubelet token
        let (t, n, hidden) = (2usize, 6usize, 8usize);
        let mut rnd = stream(3);
        let mut e: Vec<f32> = (0..t * n * hidden).map(|_| rnd()).collect();
        e[n * hidden + 2 * hidden + 3] = f32::NAN;
        let m = retention_mask(&e, t, 2, 3, hidden, 0.7);
        assert_eq!(m.iter().filter(|&&k| k).count(), 6);
        assert!(m[n + 2], "the NaN row is first in the order");
        assert!(m[..5].iter().all(|&k| k) && !m[5], "the sixth first-tubelet token is the one displaced: {m:?}");
        // a zero row: norm 0 is clamped to bf16(1e-8), the quotient is 0 and so is the similarity: dissimilarity 1
        let mut z: Vec<f32> = (0..t * n * hidden).map(|_| rnd()).collect();
        for k in 0..hidden {
            z[n * hidden + k] = 0.0;
        }
        let d = dissimilarity_bf16(&bits(&z), t, n, hidden, SumOrder::TorchCpu);
        assert_eq!(bf16_bits_to_f32(d[n]), 1.0);
    }

    #[test]
    fn errors_name_the_shape_and_the_function_is_total_on_small_inputs() {
        let e = retention_mask_bits(&[0u16; 10], 2, 2, 2, 3, 0.7, SumOrder::TorchCpu).unwrap_err();
        assert!(e.0.contains("[2, 2, 2, 3]") && e.0.contains("24 needed"), "{e}");
        assert_eq!(retention_mask_bits(&[], 0, 2, 2, 4, 0.7, SumOrder::TorchCpu).unwrap(), Vec::<bool>::new());
        // one tubelet: everything kept at any rate
        let e: Vec<f32> = (0..4 * 8).map(|i| (i % 5) as f32).collect();
        assert!(retention_mask(&e, 1, 2, 2, 8, 0.99).iter().all(|&k| k));
        // one token a tubelet
        let e: Vec<f32> = (0..5 * 8).map(|i| ((i * 7) % 11) as f32).collect();
        let m = retention_mask(&e, 5, 1, 1, 8, 0.7);
        assert_eq!(m.len(), 5);
        assert!(m[0]);
    }

    #[test]
    fn the_video_hash_depends_on_frames_order_and_canvas_and_is_not_an_images() {
        let a = video_hash(&[1, 2, 3], (448, 576));
        assert_ne!(a, video_hash(&[1, 3, 2], (448, 576)), "order");
        assert_ne!(a, video_hash(&[1, 2], (448, 576)), "count");
        assert_ne!(a, video_hash(&[1, 2, 3], (576, 448)), "canvas");
        assert_eq!(a, video_hash(&[1, 2, 3], (448, 576)));
        let one = video_hash(&[7], (448, 576));
        assert_ne!(
            one,
            crate::nemotron::canvas_hash(448, 576, &vec![0.0; 3 * 448 * 576], 1),
            "a one-frame video is not a still image"
        );
    }

    #[test]
    fn the_builder_checks_one_tile_for_all_frames() {
        let pre = cfg();
        let img = |w: usize, h: usize| Rgb { width: w, height: h, data: (0..w * h * 3).map(|i| (i * 31 % 251) as u8).collect() };
        let mut b = VideoBuilder::new(&pre);
        b.push(&img(64, 64)).unwrap();
        b.push(&img(64, 64)).unwrap();
        // 70 x 64 (aspect 1.09) resolves to a 32 x 30 patch tile, not the 32 x 32 of 64 x 64
        let e = b.check(&img(70, 64)).unwrap_err().to_string();
        assert!(e.contains("frame 3") && e.contains("one tile"), "{e}");
        assert_eq!(b.len(), 2);
        let v = b.finish().unwrap();
        assert_eq!((v.tubelets(), v.tile.tokens), (1, 256));
        assert_eq!(v.frames[0].len(), 3 * 512 * 512);
        assert_eq!(v.frame_hashes.len(), 2);
        assert!(VideoBuilder::new(&pre).finish().is_err());
        // a still frame equal to the other: same hash, same bits
        assert_eq!(v.frames[0], v.frames[1]);
        assert_eq!(v.frame_hashes[0], v.frame_hashes[1]);
    }
}
