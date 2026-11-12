//! The MiniCPM-V-4.7 vision tower, window merger and projector as the family's image encoder (E5a of
//! `docs/MINICPMV4_7_INTEGRATION_PLAN.md`), stand-alone: it turns the preprocessing's packed pixels into the projector's embedding rows. The
//! splice into the text forward (the `ImageSpan` per frame, the canvas positions of `mrope_canvas`, the chat format) is E3/E5b's.
//!
//! **What an image is here.** `gb_vision::minicpmv::MiniCpmPreprocessing` (byte-exact to the Hugging Face processors, both resize
//! backends) hands the encoder the packed pixel tensor `[3, 14, 14 * patches]` as bf16 bits, the thumbnail's patches first and then the
//! slices' (each crop's patches row-major), and the crops' patch grids (`target_sizes`). Through the `ImageEncoder` seam that is a
//! `gb_vision::Preprocessed` whose `size` is the ORIGINAL image's `(width, height)` (the encoder re-derives the crop plan from it and checks
//! the pixel count and the grid), `grid == (1, rows, cols)` of the slice grid and `patches` the packed bits.
//!
//! **What comes out.** [`MiniCpmFeatures`]: bf16 `[tokens, 2048]` rows on the device, crop-major (the thumbnail's `gh/4 x gw/4` tokens
//! row-major, then each slice's), the rows that replace the image's pad tokens in the prompt; the model's `inputs_embeds` are bf16 and the
//! projector's fp32 result is rounded once, to nearest even, exactly where the reference casts. One `EncodedImage` carries them with
//! `merged == (1, tokens)`: the canvas positions of the frame come from the prompt (`mrope_canvas`), not from a per-image grid.
//!
//! **Crops** are independent (a crop's rows do not depend on the other crops of the launch: `crates/cuda/tests/mcv_vision.rs`), so an
//! image with more patches than the workspace holds runs as several launches of whole crops and the result is the same bytes.

use crate::error::ModelError;
use crate::serve::{EncodedImage, FeatureBudget, ImageEncoder, LeasedFeatures};
use gb_cuda::mcv_vision as mv;
use gb_cuda::{DevBuf, Stream};
use gb_vision::minicpmv::{Downsample, MiniCpmPreprocessing, ResizeProfile, PATCH_DIM, PATCH_SIZE};
use gb_vision::Preprocessed;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Width of a feature row: the language model's hidden size.
pub const OUT_HIDDEN: usize = mv::LLM_HIDDEN;
/// Bytes of one bf16 feature row.
pub const FEATURE_ROW_BYTES: usize = OUT_HIDDEN * 2;
/// Patches the workspace holds by default: 16 crops of 448 pixels (an image at the default `max_slice_nums` of 9 is 10 crops and one launch;
/// the thumbnail and 36 slices of the high-resolution setting run as three).
pub const DEFAULT_MAX_PATCHES: usize = 16 * 1024;

/// The environment variable that sets the patches one tower launch holds (the workspace is sized for it; a multiple of 16, at most
/// `mcv_vision::MAX_PATCHES`). An image whose crops exceed it runs as several launches of whole crops, so it bounds memory, not the
/// image size; one crop must fit (a 448-pixel crop is 1,024 patches, the extreme-aspect crop of a 4,096-patch side needs more).
pub const MAX_PATCHES_ENV: &str = "REXL3_MINICPM_VISION_MAX_PATCHES";

/// The launch size from [`MAX_PATCHES_ENV`], or [`DEFAULT_MAX_PATCHES`]; an invalid value is an error by name (the boot checks it before
/// anything is read).
pub fn check_max_patches() -> Result<usize, ModelError> {
    match std::env::var(MAX_PATCHES_ENV) {
        Err(_) => Ok(DEFAULT_MAX_PATCHES),
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(n) if n > 0 && n % 16 == 0 && n <= mv::MAX_PATCHES => Ok(n),
            _ => Err(ModelError::Config(format!(
                "{MAX_PATCHES_ENV}={v}: a multiple of 16 between 16 and {} patches",
                mv::MAX_PATCHES
            ))),
        },
    }
}

/// [`check_max_patches`] for the memory model, which cannot fail: an invalid setting prices the default (the boot has refused it).
pub fn max_patches_setting() -> usize {
    check_max_patches().unwrap_or(DEFAULT_MAX_PATCHES)
}

/// `REXL3_MINICPM_RESIZE_PROFILE`: `torch` (default: what a Hugging Face install with torchvision runs, `tvF.resize`) or `pil` (the
/// Pillow backend, what an install without torchvision runs; the oracle venv of the plan has none, so its Leg C asks for this).
pub const RESIZE_PROFILE_ENV: &str = "REXL3_MINICPM_RESIZE_PROFILE";
/// `REXL3_MINICPM_MAX_SLICE_NUMS`: the most slices of an image (default the pack's `preprocessor_config.json`, 9; the model card's
/// high-resolution setting is 36). It is part of the cache key (`MiniCpmPreprocessing::id`).
pub const MAX_SLICE_NUMS_ENV: &str = "REXL3_MINICPM_MAX_SLICE_NUMS";
/// `REXL3_MINICPM_DOWNSAMPLE`: `16x` (default: the pack's `downsample_mode`, 64 tokens a 448-pixel crop) or `4x` (256).
pub const DOWNSAMPLE_ENV: &str = "REXL3_MINICPM_DOWNSAMPLE";
/// `REXL3_MINICPM_VIDEO_MAX_SLICE_NUMS`: the most slices of a video frame (default 1: a frame is its thumbnail, as the model card
/// recommends for video; the Hugging Face video processor's own class default is 9).
pub const VIDEO_MAX_SLICE_NUMS_ENV: &str = "REXL3_MINICPM_VIDEO_MAX_SLICE_NUMS";

fn env_choice(key: &str) -> Option<String> {
    std::env::var(key).ok().map(|v| v.trim().to_ascii_lowercase()).filter(|v| !v.is_empty())
}

/// The image preprocessing of a pack folder: `preprocessor_config.json`, the model config's `downsample_mode`, then the environment
/// options above. One function for the server's preprocessing and the encoder's crop plan, so they cannot differ.
pub fn image_preprocessing(dir: &Path) -> Result<MiniCpmPreprocessing, ModelError> {
    let cfg = super::config::MiniCpmConfig::from_dir(dir).map_err(ModelError::Config)?;
    let mut p = MiniCpmPreprocessing::from_dir(dir).map_err(|e| ModelError::Config(e.to_string()))?;
    p = p.with_downsample(match cfg.vision.downsample_mode {
        super::config::DownsampleMode::X16 => Downsample::X16,
        super::config::DownsampleMode::X4 => Downsample::X4,
    });
    if let Some(v) = env_choice(DOWNSAMPLE_ENV) {
        p = p.with_downsample(
            Downsample::parse(&v).ok_or_else(|| ModelError::Config(format!("{DOWNSAMPLE_ENV}={v}: 16x or 4x")))?,
        );
    }
    if let Some(v) = env_choice(RESIZE_PROFILE_ENV) {
        p = p.with_profile(match v.as_str() {
            "torch" => ResizeProfile::Torch,
            "pil" => ResizeProfile::Pil,
            _ => return Err(ModelError::Config(format!("{RESIZE_PROFILE_ENV}={v}: torch or pil"))),
        });
    }
    if let Some(v) = env_choice(MAX_SLICE_NUMS_ENV) {
        let n: usize = v.parse().map_err(|_| ModelError::Config(format!("{MAX_SLICE_NUMS_ENV}={v}: a number")))?;
        p = p.with_max_slice_nums(n).map_err(|e| ModelError::Config(format!("{MAX_SLICE_NUMS_ENV}={v}: {e}")))?;
    }
    Ok(p)
}

/// The preprocessing of a video's frames: the image preprocessing at [`VIDEO_MAX_SLICE_NUMS_ENV`] slices (default 1: no slicing).
pub fn video_preprocessing(image: &MiniCpmPreprocessing) -> Result<MiniCpmPreprocessing, ModelError> {
    let n = match env_choice(VIDEO_MAX_SLICE_NUMS_ENV) {
        None => 1,
        Some(v) => v.parse().map_err(|_| ModelError::Config(format!("{VIDEO_MAX_SLICE_NUMS_ENV}={v}: a number")))?,
    };
    image.clone().with_max_slice_nums(n).map_err(|e| ModelError::Config(format!("{VIDEO_MAX_SLICE_NUMS_ENV}={n}: {e}")))
}

/// The mode of the tower for a preprocessing downsample setting.
pub fn mode_of(d: Downsample) -> mv::Mode {
    match d {
        Downsample::X16 => mv::Mode::X16,
        Downsample::X4 => mv::Mode::X4,
    }
}

/// Where one crop's rows are among an image's features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropRows {
    /// First feature row of the crop.
    pub first_token: usize,
    /// The crop's output grid `(rows, columns)` of tokens; `rows * columns` rows follow `first_token`.
    pub grid: (usize, usize),
    /// The crop's patch grid `(gh, gw)`.
    pub patches: (usize, usize),
}

/// One image's projector output on the device: bf16 `[tokens, 2048]`, crop-major.
pub struct MiniCpmFeatures {
    pub buf: DevBuf,
    pub tokens: usize,
    pub crops: Vec<CropRows>,
    /// The slice grid `(rows, columns)` of the image the crops came from (`(0, 0)`: a thumbnail alone). Set by the encoder, which knows the
    /// crop plan; the chat format lays the slices out in this grid (`\n` between rows) and the splice reads the canvas from the prompt.
    pub slice_grid: (usize, usize),
}

impl MiniCpmFeatures {
    /// Patches per side of the tower's output grid: 4 at `16x`, 2 at `4x` (`None` for features with no crop).
    pub fn merge_factor(&self) -> Option<usize> {
        let c = self.crops.first()?;
        (c.grid.0 != 0).then(|| c.patches.0 / c.grid.0)
    }
}

// SAFETY: the features are immutable device rows once built; the engine thread only reads them, on streams of its own (the same contract
// as the Qwen, GLM and Nemotron feature types).
unsafe impl Sync for MiniCpmFeatures {}

/// Bytes of one image's features.
pub fn feature_bytes(tokens: usize) -> u64 {
    tokens as u64 * FEATURE_ROW_BYTES as u64
}

/// The tower on the device with its scratch: the weights (1.12 GB, fp16 and fp32), the workspace and a stream of its own that does not
/// synchronise with the default one (an encode overlaps the engine thread's work).
pub struct MiniCpmTower {
    w: mv::Weights,
    ws: mv::Workspace,
    pix: DevBuf,
    st: Stream,
}

impl MiniCpmTower {
    /// Loads the tower, merger and projector from a MiniCPM-V-4.7 checkpoint folder (or an EXL3 pack: the 459 tensors are copied
    /// through unchanged) and allocates a workspace for up to `max_patches` patches (a multiple of 16).
    pub fn load(dir: &Path, max_patches: usize) -> Result<MiniCpmTower, ModelError> {
        if max_patches == 0 || max_patches % 16 != 0 || max_patches > mv::MAX_PATCHES {
            return Err(ModelError::Config(format!("max_patches {max_patches}: a multiple of 16, at most {}", mv::MAX_PATCHES)));
        }
        let w = mv::Weights::load(dir).map_err(|e| ModelError::Config(format!("vision tower of {}: {e}", dir.display())))?;
        if w.report().overflow != 0 {
            return Err(ModelError::Config(format!(
                "vision tower of {}: {} weights exceed the fp16 range (the fp32 fallback is not built)",
                dir.display(),
                w.report().overflow
            )));
        }
        Ok(MiniCpmTower {
            w,
            ws: mv::Workspace::new(max_patches)?,
            pix: DevBuf::zeroed(max_patches * PATCH_DIM * 2)?,
            st: Stream::new_nonblocking()?,
        })
    }

    /// Patches one launch holds.
    pub fn max_patches(&self) -> usize {
        self.ws.max_patches()
    }

    /// Device bytes of the weights and of the workspace (the memory model's vision component).
    pub fn bytes(&self) -> (u64, u64) {
        (self.w.bytes(), self.ws.bytes() + self.pix.bytes() as u64)
    }

    /// What the bf16 -> fp16 conversion of the weights did.
    pub fn convert_report(&self) -> &mv::ConvertReport {
        self.w.report()
    }

    /// Runs `groups` of whole crops through the tower and calls `sink(group index, plan, device output fp32 [tokens, 2048])` after each,
    /// with the stream drained so the output is readable.
    fn run_groups(
        &mut self,
        pixels: &[u16],
        target_sizes: &[[u32; 2]],
        mode: mv::Mode,
        mut sink: impl FnMut(&mv::Plan, &DevBuf, &Stream) -> Result<(), ModelError>,
    ) -> Result<Vec<mv::Plan>, ModelError> {
        let crops: Vec<(usize, usize)> = target_sizes.iter().map(|t| (t[0] as usize, t[1] as usize)).collect();
        let total: usize = crops.iter().map(|&(h, w)| h * w).sum();
        if pixels.len() != 3 * PATCH_SIZE * PATCH_SIZE * total {
            return Err(ModelError::Request(format!(
                "image pixels: {} values for crops of {total} patches (want {})",
                pixels.len(),
                3 * PATCH_SIZE * PATCH_SIZE * total
            )));
        }
        // whole crops per launch, in order, while they fit the workspace
        let cap = self.ws.max_patches();
        let mut groups: Vec<(usize, usize)> = Vec::new(); // crop ranges
        let (mut a, mut run) = (0usize, 0usize);
        for (k, &(h, w)) in crops.iter().enumerate() {
            let n = h * w;
            if n > cap {
                return Err(ModelError::Request(format!(
                    "a crop of {h} x {w} patches ({n}) exceeds the vision workspace of {cap} patches"
                )));
            }
            if run + n > cap {
                groups.push((a, k));
                a = k;
                run = 0;
            }
            run += n;
        }
        groups.push((a, crops.len()));
        let mut plans = Vec::with_capacity(groups.len());
        let width_all = total * PATCH_SIZE;
        for &(a, b) in &groups {
            let group_crops = &crops[a..b];
            let patches: usize = group_crops.iter().map(|&(h, w)| h * w).sum();
            let plan = mv::Plan::new(group_crops, mode, cap).map_err(|e| ModelError::Request(format!("vision plan: {e}")))?;
            let first_patch: usize = crops[..a].iter().map(|&(h, w)| h * w).sum();
            let host: std::borrow::Cow<[u16]> = if groups.len() == 1 {
                std::borrow::Cow::Borrowed(pixels)
            } else {
                // the packed tensor is [3 * 14 rows][all crops' columns]: take this group's columns of every row
                let (c0, wg) = (first_patch * PATCH_SIZE, patches * PATCH_SIZE);
                let mut v = vec![0u16; 3 * PATCH_SIZE * wg];
                for r in 0..3 * PATCH_SIZE {
                    v[r * wg..(r + 1) * wg].copy_from_slice(&pixels[r * width_all + c0..r * width_all + c0 + wg]);
                }
                std::borrow::Cow::Owned(v)
            };
            mv::h2d(&mut self.pix, &host, &self.st)?;
            mv::forward(&self.w, &mut self.ws, &self.pix, &plan, None, &self.st)?;
            // an fp16 overflow anywhere in the tower ends as inf or NaN in the projector's output: refuse it by name instead of serving it
            let m = self.ws.output_absmax(plan.tokens, &self.st)?;
            if !m.is_finite() {
                return Err(ModelError::Request(
                    "the vision tower produced non-finite embeddings (an activation left the fp16 range)".into(),
                ));
            }
            sink(&plan, self.ws.out(), &self.st)?;
            plans.push(plan);
        }
        Ok(plans)
    }

    /// The projector's output of an image as host fp32 `[tokens, 2048]` (before the bf16 rounding): what the oracle gates compare.
    pub fn encode_packed_f32(
        &mut self,
        pixels: &[u16],
        target_sizes: &[[u32; 2]],
        mode: mv::Mode,
    ) -> Result<Vec<f32>, ModelError> {
        let mut out: Vec<f32> = Vec::new();
        let r = self.run_groups(pixels, target_sizes, mode, |plan, dev, st| {
            st.sync()?;
            let at = out.len();
            out.resize(at + plan.tokens * OUT_HIDDEN, 0.0);
            dev.download_range(0, &mut out[at..])?;
            Ok(())
        });
        if r.is_err() {
            let _ = self.st.sync();
        }
        r?;
        Ok(out)
    }

    /// The projector's output of an image as bf16 device rows (`tokens * 2048 * 2` bytes), with where each crop's rows are.
    pub fn encode_packed(
        &mut self,
        pixels: &[u16],
        target_sizes: &[[u32; 2]],
        mode: mv::Mode,
    ) -> Result<MiniCpmFeatures, ModelError> {
        self.encode_packed_into(pixels, target_sizes, mode, None)
    }

    fn encode_packed_into(
        &mut self,
        pixels: &[u16],
        target_sizes: &[[u32; 2]],
        mode: mv::Mode,
        buf: Option<DevBuf>,
    ) -> Result<MiniCpmFeatures, ModelError> {
        // the token count from the geometry alone, before any device work
        let per = mode.patches_per_token();
        let tokens: usize = target_sizes.iter().map(|t| t[0] as usize * t[1] as usize / per).sum();
        let mut out = match buf {
            Some(b) => b,
            None => DevBuf::alloc(tokens * FEATURE_ROW_BYTES)?,
        };
        if out.bytes() < tokens * FEATURE_ROW_BYTES {
            return Err(ModelError::Request("feature buffer too small".into()));
        }
        let mut scratch: Option<DevBuf> = None;
        let mut row = 0usize;
        let mut crops_out = Vec::new();
        let r = self.run_groups(pixels, target_sizes, mode, |plan, dev, st| {
            let n = plan.tokens * OUT_HIDDEN;
            let s = match scratch.take() {
                Some(s) if s.bytes() >= n * 2 => s,
                _ => DevBuf::alloc(n * 2)?,
            };
            let mut s = s;
            mv::f32_to_bf16(dev, &mut s, n, st)?;
            out.copy_range_from(row * FEATURE_ROW_BYTES, &s, 0, n * 2, st)?;
            for (k, &(gh, gw)) in plan.crops.iter().enumerate() {
                crops_out.push(CropRows { first_token: row + plan.token0[k], grid: plan.out_grid[k], patches: (gh, gw) });
            }
            row += plan.tokens;
            // the scratch is reused by the next group's rounding on the same stream; keep it
            scratch = Some(s);
            Ok(())
        });
        // always drain: the copies read buffers that are about to be dropped
        let synced = self.st.sync();
        r?;
        synced?;
        debug_assert_eq!(row, tokens);
        Ok(MiniCpmFeatures { buf: out, tokens, crops: crops_out, slice_grid: (0, 0) })
    }
}

/// `REXL3_MINICPM_FEATURE_CACHE_MIB`: bytes of encoded image features the encoder keeps for the next request that sends the same image
/// (default a quarter of the feature budget, at most 64 MiB; `0` turns the cache off). The rows are immutable device memory shared by
/// every request that holds them, and they stay inside the feature budget: an entry holds its lease until it is evicted and the last
/// request using it ends.
pub const FEATURE_CACHE_ENV: &str = "REXL3_MINICPM_FEATURE_CACHE_MIB";

/// What decides an image's feature rows: its resized pixels and geometry (`Preprocessed::pixel_hash` hashes the thumbnail's and the
/// refined image's bytes and the crop geometry), the original size and slice grid, the preprocessing's whole identity (backend, scale,
/// patch, `max_slice_nums`, normalisation: `MiniCpmPreprocessing::id`) and, spelled out because it also changes the rows, the
/// downsample mode and the slice limit. Two images with equal keys have equal feature rows; a key that left one of these out would
/// serve rows of another image or of another mode (`feature_key_sensitivity`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FeatureKey {
    pub pixel_hash: u64,
    pub size: (usize, usize),
    pub grid: (usize, usize, usize),
    pub pre_id: String,
    pub downsample: &'static str,
    pub max_slice_nums: usize,
}

impl FeatureKey {
    pub fn of(p: &Preprocessed, pre: &MiniCpmPreprocessing) -> FeatureKey {
        FeatureKey {
            pixel_hash: p.pixel_hash,
            size: p.size,
            grid: p.grid,
            pre_id: pre.id(),
            downsample: pre.downsample.name(),
            max_slice_nums: pre.max_slice_nums,
        }
    }
}

struct CacheEntry {
    features: Arc<dyn std::any::Any + Send + Sync>,
    tokens: usize,
    bytes: u64,
    last: u64,
}

/// A least-recently-used cache of encoded features, bounded in bytes.
pub struct FeatureCache {
    cap: u64,
    held: u64,
    tick: u64,
    map: HashMap<FeatureKey, CacheEntry>,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl FeatureCache {
    pub fn new(cap_bytes: u64) -> Self {
        FeatureCache { cap: cap_bytes, held: 0, tick: 0, map: HashMap::new(), hits: 0, misses: 0, evictions: 0 }
    }

    /// The default capacity for a feature budget of `budget_total` bytes, from [`FEATURE_CACHE_ENV`].
    pub fn capacity_for(budget_total: u64) -> u64 {
        match std::env::var(FEATURE_CACHE_ENV).ok().and_then(|v| v.trim().parse::<u64>().ok()) {
            Some(mib) => mib << 20,
            None => (budget_total / 4).min(64 << 20),
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn held_bytes(&self) -> u64 {
        self.held
    }

    pub fn get(&mut self, k: &FeatureKey) -> Option<(Arc<dyn std::any::Any + Send + Sync>, usize)> {
        self.tick += 1;
        let tick = self.tick;
        match self.map.get_mut(k) {
            Some(e) => {
                e.last = tick;
                self.hits += 1;
                Some((e.features.clone(), e.tokens))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Keep `features` under `k` (evicting the least recently used until it fits); an entry larger than the whole cache is not kept.
    pub fn put(&mut self, k: FeatureKey, features: Arc<dyn std::any::Any + Send + Sync>, tokens: usize, bytes: u64) {
        if bytes > self.cap {
            return;
        }
        if let Some(old) = self.map.remove(&k) {
            self.held -= old.bytes;
        }
        while self.held + bytes > self.cap && self.evict_one() {}
        self.tick += 1;
        self.held += bytes;
        self.map.insert(k, CacheEntry { features, tokens, bytes, last: self.tick });
    }

    /// Drop the least recently used entry (its lease returns to the budget once no request holds its rows); `false` when empty.
    pub fn evict_one(&mut self) -> bool {
        let Some(k) = self.map.iter().min_by_key(|(_, e)| e.last).map(|(k, _)| k.clone()) else { return false };
        let e = self.map.remove(&k).expect("the key was just found");
        self.held -= e.bytes;
        self.evictions += 1;
        true
    }
}

/// The server-facing encoder: preprocessing identity, the tower and a feature budget.
pub struct MiniCpmEncoder {
    tower: MiniCpmTower,
    pre: MiniCpmPreprocessing,
    /// The preprocessing of a video's frames (E5b: `max_slice_nums` 1 by default): a frame's crop plan is re-derived from it when the
    /// image preprocessing's grid is not the one the preprocessed pixels carry.
    video: Option<MiniCpmPreprocessing>,
    budget: Arc<FeatureBudget>,
    cache: FeatureCache,
}

impl MiniCpmEncoder {
    /// `pre` must be the preprocessing that produced the images (its `max_slice_nums` and downsample decide the crop plan the encoder
    /// re-derives from the original size).
    pub fn new(tower: MiniCpmTower, pre: MiniCpmPreprocessing, budget: Arc<FeatureBudget>) -> Self {
        let cache = FeatureCache::new(FeatureCache::capacity_for(budget.total()));
        MiniCpmEncoder { tower, pre, video: None, budget, cache }
    }

    /// Replaces the feature cache (its capacity in bytes; 0 keeps nothing).
    pub fn with_cache_bytes(mut self, bytes: u64) -> Self {
        self.cache = FeatureCache::new(bytes);
        self
    }

    pub fn cache(&self) -> &FeatureCache {
        &self.cache
    }

    /// The preprocessing of video frames, when it differs from the image preprocessing in the slices it allows (it must agree in
    /// everything else: the downsample mode decides the tower's mode).
    pub fn with_video_preprocessing(mut self, video: MiniCpmPreprocessing) -> Self {
        self.video = Some(video);
        self
    }

    pub fn tower(&mut self) -> &mut MiniCpmTower {
        &mut self.tower
    }

    pub fn preprocessing(&self) -> &MiniCpmPreprocessing {
        &self.pre
    }

    /// Replaces the preprocessing the encoder re-derives its crop plans from (a per-request `downsample_mode` or `max_slice_nums`).
    pub fn set_preprocessing(&mut self, pre: MiniCpmPreprocessing) {
        self.pre = pre;
    }
}

impl ImageEncoder for MiniCpmEncoder {
    fn encode(&mut self, p: &Preprocessed) -> Result<EncodedImage, ModelError> {
        let mut plan = self
            .pre
            .plan(p.size.0, p.size.1)
            .map_err(|e| ModelError::Request(format!("image of {}x{}: {e}", p.size.0, p.size.1)))?;
        let mut used_video = false;
        if p.grid != (1, plan.grid.0, plan.grid.1) {
            // a frame of a video is cut by its own preprocessing: take its plan when that is the grid the pixels carry
            if let Some(v) = self.video.as_ref().filter(|v| v.downsample == self.pre.downsample) {
                if let Ok(alt) = v.plan(p.size.0, p.size.1) {
                    if p.grid == (1, alt.grid.0, alt.grid.1) {
                        plan = alt;
                        used_video = true;
                    }
                }
            }
        }
        if p.patch_dim != PATCH_DIM
            || p.grid != (1, plan.grid.0, plan.grid.1)
            || p.patches.len() != PATCH_DIM * plan.total_patches
        {
            return Err(ModelError::Request(format!(
                "preprocessed image does not match the crop plan of a {}x{} image: grid {:?} (want (1, {}, {})), {} values (want {})",
                p.size.0,
                p.size.1,
                p.grid,
                plan.grid.0,
                plan.grid.1,
                p.patches.len(),
                PATCH_DIM * plan.total_patches
            )));
        }
        let tokens = plan.total_tokens;
        // the rows of an image already encoded under this exact key are shared, not made again (the key is `FeatureKey`'s)
        let key = FeatureKey::of(p, if used_video { self.video.as_ref().expect("used") } else { &self.pre });
        if let Some((features, cached_tokens)) = self.cache.get(&key) {
            if cached_tokens == tokens {
                return Ok(EncodedImage { tokens, merged: (1, tokens), features });
            }
        }
        let bytes = feature_bytes(tokens);
        let permit = loop {
            match self.budget.acquire(bytes) {
                Some(p) => break p,
                // the cache is the budget's idle part: give entries back before refusing an image
                None if self.cache.evict_one() => continue,
                None => {
                    return Err(ModelError::Busy(format!(
                        "the memory for image features is in use ({:.0} MiB free of {:.0} MiB, this image needs {:.0} MiB); retry shortly",
                        self.budget.free() as f64 / 1048576.0,
                        self.budget.total() as f64 / 1048576.0,
                        bytes as f64 / 1048576.0
                    )))
                }
            }
        };
        let mut f = self.tower.encode_packed(&p.patches, &plan.target_sizes(), mode_of(self.pre.downsample))?;
        f.slice_grid = plan.grid;
        if f.tokens != tokens {
            return Err(ModelError::Request(format!("the tower produced {} rows for {tokens} placeholders", f.tokens)));
        }
        let features: Arc<dyn std::any::Any + Send + Sync> = Arc::new(LeasedFeatures::new(f, permit));
        self.cache.put(key, features.clone(), tokens, bytes);
        Ok(EncodedImage { tokens, merged: (1, tokens), features })
    }
}

/// Device bytes of the tower's weights as `mcv_vision::Weights::load` holds them (matrices fp16 `[out, in]`, the MLP and the patch matrix
/// zero padded, vectors and the position table fp32), from the geometry alone.
pub fn weight_bytes() -> u64 {
    const F16: u64 = 2;
    const F32: u64 = 4;
    let (h, mlpp) = (mv::HIDDEN as u64, mv::MLP_PADDED as u64);
    let layer = 4 * h * F32 // two LayerNorms
        + 3 * h * h * F16 + 3 * h * F32 // q, k, v and biases
        + h * h * F16 + h * F32 // out_proj
        + mlpp * h * F16 + mlpp * F32 // fc1 (padded)
        + h * mlpp * F16 + h * F32; // fc2 (padded)
    let (w4, m4) = (mv::WINDOW_HIDDEN as u64, mv::MERGER_MLP as u64);
    let merger = 2 * h * F32 // layer_norm1
        + 3 * h * h * F16 + 3 * h * F32
        + h * h * F16 + h * F32
        + 2 * w4 * F32 // pre_norm
        + m4 * w4 * F16 + m4 * F32 // linear_1
        + h * m4 * F16 + h * F32; // linear_2
    let l = mv::LLM_HIDDEN as u64;
    let projector = 2 * w4 * F32 + w4 * w4 * F16 + w4 * F32 + l * w4 * F16 + l * F32;
    h * mv::PATCH_PADDED as u64 * F16 + h * F32 // patch embedding
        + (mv::POS_SIDE * mv::POS_SIDE) as u64 * h * F32 // position table
        + mv::LAYERS as u64 * layer
        + 2 * h * F32 // post_layernorm
        + merger
        + projector
}

/// Device bytes of `mcv_vision::Workspace::new(max_patches)` plus the encoder's packed-pixel buffer.
pub fn workspace_bytes(max_patches: usize) -> u64 {
    let n = max_patches as u64;
    let (h, mlpp) = (mv::HIDDEN as u64, mv::MLP_PADDED as u64);
    n * mv::PATCH_PADDED as u64 * 2 // patch matrix
        + n * h * 4 // residual stream
        + n * h * 2 // normed
        + n * 3 * h * 2 // qkv
        + n * h * 2 // attention out
        + n * mlpp * 2 // MLP hidden
        + n * h * 4 // window-order stream
        + n * h * 2 // window-order normed
        + n / 4 * mv::LLM_HIDDEN as u64 * 4 // projector out
        + 3 * n * 4 // index tables
        + 2 * mv::MAX_CROPS as u64 * 8 // crop tables
        + 64 // absmax scratch
        + n * PATCH_DIM as u64 * 2 // packed pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pre_of(mode: Downsample, slices: usize) -> MiniCpmPreprocessing {
        MiniCpmPreprocessing::minicpmv47().with_downsample(mode).with_max_slice_nums(slices).unwrap()
    }

    fn image(hash: u64) -> Preprocessed {
        Preprocessed { size: (1344, 896), grid: (1, 2, 3), patches: vec![], pixel_hash: hash, patch_dim: PATCH_DIM }
    }

    #[test]
    fn feature_key_sensitivity() {
        let base = FeatureKey::of(&image(7), &pre_of(Downsample::X16, 9));
        assert_eq!(base, FeatureKey::of(&image(7), &pre_of(Downsample::X16, 9)));
        // the pixels, the mode, the slice limit, the geometry and the preprocessing's own identity each give another key
        assert_ne!(base, FeatureKey::of(&image(8), &pre_of(Downsample::X16, 9)), "pixels");
        assert_ne!(base, FeatureKey::of(&image(7), &pre_of(Downsample::X4, 9)), "downsample mode");
        assert_ne!(base, FeatureKey::of(&image(7), &pre_of(Downsample::X16, 36)), "max_slice_nums");
        let mut other = image(7);
        other.size = (1344, 897);
        assert_ne!(base, FeatureKey::of(&other, &pre_of(Downsample::X16, 9)), "size");
        let mut other = image(7);
        other.grid = (1, 3, 2);
        assert_ne!(base, FeatureKey::of(&other, &pre_of(Downsample::X16, 9)), "slice grid");
        let torch = pre_of(Downsample::X16, 9);
        let pil = torch.clone().with_profile(ResizeProfile::Pil);
        assert_ne!(FeatureKey::of(&image(7), &torch), FeatureKey::of(&image(7), &pil), "resize backend");
        // and each field is read by `FeatureKey` itself, not only through the id string
        let mut k = base.clone();
        k.downsample = "4x";
        assert_ne!(base, k);
        k = base.clone();
        k.max_slice_nums = 36;
        assert_ne!(base, k);
    }

    #[test]
    fn the_cache_is_lru_bounded_in_bytes_and_counts_hits() {
        let mut c = FeatureCache::new(100);
        let key = |h: u64| FeatureKey::of(&image(h), &pre_of(Downsample::X16, 9));
        let f = || -> Arc<dyn std::any::Any + Send + Sync> { Arc::new(()) };
        assert!(c.get(&key(1)).is_none());
        c.put(key(1), f(), 10, 40);
        c.put(key(2), f(), 10, 40);
        assert_eq!((c.len(), c.held_bytes()), (2, 80));
        assert!(c.get(&key(1)).is_some(), "1 is now the most recent");
        c.put(key(3), f(), 10, 40);
        assert!(c.get(&key(2)).is_none(), "2 was the least recently used");
        assert!(c.get(&key(1)).is_some() && c.get(&key(3)).is_some());
        assert_eq!((c.hits, c.misses, c.evictions), (3, 2, 1));
        // an entry that cannot fit at all is not kept, and an off cache keeps nothing
        c.put(key(9), f(), 10, 101);
        assert!(c.get(&key(9)).is_none());
        let mut off = FeatureCache::new(0);
        off.put(key(1), f(), 10, 1);
        assert!(off.is_empty());
        // evicting under pressure empties it one entry at a time
        assert!(c.evict_one() && c.evict_one() && !c.evict_one());
        assert_eq!(c.held_bytes(), 0);
    }

    #[test]
    fn the_arithmetic_of_the_weights_is_the_checkpoints_size_plus_padding() {
        // the checkpoint holds 1,105,972,064 bytes of bf16 in these 459 tensors; fp16 matrices are as large, the position table (22.6 MB)
        // and the vectors are widened to fp32, and the MLP and patch matrix are padded
        let w = weight_bytes();
        assert!(w > 1_105_972_064 && w < 1_160_000_000, "{w}");
    }

    #[test]
    fn the_default_workspace_is_a_few_hundred_megabytes() {
        let b = workspace_bytes(DEFAULT_MAX_PATCHES);
        assert!((400_000_000..800_000_000).contains(&b), "{b}");
    }
}
