//! The RADIO tower and the pixel-shuffle projector as the server's image encoder for Nemotron 3.5 Super VL (milestone M3, task VS-b).
//!
//! **What an image is here.** The server's preprocessing (`gb_vision::nemotron::NemotronPreprocessing`, bit-exact to the HF processor)
//! hands the encoder `gb_vision::Preprocessed`: the normalised tile as bf16 bits in channel planes `[3][H][W]` (not patch-major; the
//! tower's `patchify` cuts the 16 x 16 patches on the device), `grid == (1, H / 16, W / 16)` and `size == (W, H)`. The encoder uploads
//! those bits, widens them to fp16 in place on the device (`glm_vision::bf16_to_f16`: round to nearest even, exact for every value a
//! normalised pixel takes, and a count of results that are not finite is checked), runs [`gb_cuda::nemotron_vit::tower_forward`] (32 blocks,
//! `vision_final_layernorm`, pixel shuffle v2, mlp1) and rounds mlp1's fp32 output to bf16 (`vision::f32_to_bf16`, round to nearest even):
//! the model's `vit_embeds` are bf16 and are scattered over the bf16 embedding rows, so bf16 is the boundary at which the engine's splice
//! (`nemotron_glue::splice_rows`, bf16 rows over the f32 embedding rows) is exact. One image is `tokens = patches / 4` rows of 4,096.
//!
//! **Memory.** [`weight_bytes`] and [`workspace_bytes`] are the arithmetic the plan prices (`memory.rs`, `VisionMemory`); the tower checks
//! its real allocations against them at load and refuses to start when they differ. The features of the images in flight are the one
//! cost that is not allocated up front: the plan holds back `slots x images-per-request` images of the largest size and a
//! [`FeatureBudget`] of exactly that many bytes bounds the encodes (an encode that finds it exhausted is answered 503 with Retry-After, as
//! GLM's is). The encoder runs on the caller's thread with a stream of its own that does not synchronise with the default stream: the
//! tower uses no cuBLAS handle, so nothing here is per host thread.
//!
//! **Video (milestone M4).** A video is a list of frames; two consecutive frames make a *tubelet* (an odd count repeats the last frame), whose
//! six bf16 planes go through the same tower with the video patch projection (`video_embedder.weight`, K = 1,536; everything after the first
//! GEMM is the image path's). [`NemotronTower::encode_video_into`] runs the tubelets in batches the workspace holds
//! (`gb_cuda::nemotron_vit::video_batch_capacity`): the row-wise stages run once over the batch, attention per tubelet, so a tubelet's rows
//! are bit-identical to its run alone. A video needs no buffer of its own: a tubelet's six planes are staged in the encoder's `pix` and the
//! workspace's `patches`, both sized for three planes, so a tile must be at most half the patches the workspace holds (the memory model's
//! numbers are the image path's plus the 3.9 MB video embedder). EVS ([`NemotronEncoder::encode_video`]'s `keep`) runs on the host over the
//! downloaded bf16 features and the kept rows are gathered on the host: the device never decides which rows are kept.

use crate::error::ModelError;
use crate::serve::{EncodedImage, FeatureBudget, FeaturePermit, ImageEncoder, KeepRows, VideoInput};
use gb_cuda::nemotron_vit as nv;
use gb_cuda::{DevBuf, Stream};
use std::path::Path;
use std::sync::Arc;

/// Width of an image feature row: the language model's hidden size.
pub const OUT_HIDDEN: usize = nv::LLM_HIDDEN;
/// Patches folded into one language-model token (the 2 x 2 pixel shuffle).
pub const PATCHES_PER_TOKEN: usize = 4;
/// The most tokens one image can take: the tower's `MAX_PATCHES` over [`PATCHES_PER_TOKEN`] (3,328).
pub const MAX_IMAGE_TOKENS: usize = nv::MAX_PATCHES / PATCHES_PER_TOKEN;
/// Values of one patch: 3 channels of 16 x 16.
pub const PATCH_DIM: usize = 3 * nv::PATCH * nv::PATCH;
/// Bytes of one feature row: bf16 `[4096]`.
pub const FEATURE_ROW_BYTES: usize = OUT_HIDDEN * 2;

/// The per-image token limit this engine runs: what the server asked for (`--max-image-tokens`), at most what the tower holds. The ONE place
/// the cap is taken, so the preprocessing, the memory model and the workspace size never disagree about it (the library default of 4,096 is
/// above the tower's 3,328 and is clamped here, as the family's own `VisionSpec` cap is the server's default).
pub fn image_token_cap(requested: usize) -> usize {
    requested.min(MAX_IMAGE_TOKENS)
}

// ------------------------------------------------------------------------------------------------------------------ the arithmetic

const F16: u64 = 2;
const F32: u64 = 4;

/// Device bytes of the tower's weights as `nemotron_vit::Weights::load` holds them (matrices fp16 `[out, in]`, vectors and the position
/// table fp32), from the geometry alone. The pack's 1.3 GB of bf16 is these, widened where the kernels read f32 (the position table is the
/// largest of those: 128 x 128 x 1,280 f32).
pub fn weight_bytes() -> u64 {
    let (h, mlp, pre) = (nv::HIDDEN as u64, nv::MLP_HIDDEN as u64, nv::PREFIX as u64);
    let patch_in = (3 * nv::PATCH * nv::PATCH) as u64;
    let global = h * patch_in * F16 // patch embedder
        + h * nv::VIDEO_PATCH_DIM as u64 * F16 // video patch embedder (video_embedder.weight [1280, 1536])
        + (nv::POS_GRID * nv::POS_GRID) as u64 * h * F32 // position embedding
        + pre * h * F32 // cls and register tokens
        + 2 * h * F32 // vision_final_layernorm
        + 4 * h * F32 // mlp1.0 RMSNorm
        + nv::MLP1_HIDDEN as u64 * 4 * h * F16 // mlp1.1
        + nv::LLM_HIDDEN as u64 * nv::MLP1_HIDDEN as u64 * F16; // mlp1.3
    let block = 4 * h * F32 // two LayerNorms, weight and bias
        + 3 * h * h * F16 + 3 * h * F32 // fused qkv and its bias
        + h * h * F16 + h * F32 // proj and its bias
        + mlp * h * F16 + mlp * F32 // fc1 and its bias
        + h * mlp * F16 + h * F32; // fc2 and its bias
    global + nv::BLOCKS as u64 * block
}

/// Device bytes of `nemotron_vit::Workspace::new(max_patches)`.
pub fn workspace_bytes(max_patches: usize) -> u64 {
    let n = max_patches as u64;
    let t = n + nv::PREFIX as u64;
    let np = n / PATCHES_PER_TOKEN as u64;
    let (h, mlp) = (nv::HIDDEN as u64, nv::MLP_HIDDEN as u64);
    n * PATCH_DIM as u64 * F16 // patches
        + t * h * F32 // residual stream
        + t * h * F16 // normed
        + t * 3 * h * F16 // qkv
        + t * h * F16 // attention out
        + t * mlp * F16 // fc1 out
        + n * h * F16 // vision_final_layernorm out
        + np * 4 * h * F16 * 2 // pixel shuffle, its RMSNorm
        + np * nv::MLP1_HIDDEN as u64 * F16 // mlp1 hidden
        + np * nv::LLM_HIDDEN as u64 * F32 // mlp1 out
}

/// Bytes of the encoder's own buffers on top of the tower's workspace: the fp16 pixels `[3, H, W]` of the largest image and the counter of
/// non-finite widened pixels.
pub fn input_bytes(max_patches: usize) -> u64 {
    max_patches as u64 * PATCH_DIM as u64 * F16 + BAD_BYTES
}

/// The counter of results the bf16 to fp16 widen could not represent.
const BAD_BYTES: u64 = 4;

/// What the tower holds besides its weights for images of up to `max_tokens` tokens: its workspace and the encoder's input buffers. The
/// memory model's `VisionMemory::encoder_scratch_bytes`.
pub fn scratch_bytes(max_tokens: usize) -> u64 {
    let patches = max_tokens * PATCHES_PER_TOKEN;
    workspace_bytes(patches) + input_bytes(patches)
}

/// Bytes of one image's features (bf16 `[tokens, 4096]`): what a [`FeatureBudget`] permit is for.
pub fn feature_bytes(tokens: usize) -> u64 {
    tokens as u64 * FEATURE_ROW_BYTES as u64
}

/// What one video tile runs as: its patch grid, the tokens of a tubelet, and how many tubelets go through the blocks at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoShape {
    pub gh: usize,
    pub gw: usize,
    /// Rows of one tubelet's features: `gh * gw / 4`.
    pub tokens: usize,
    /// Tubelets per batch (`gb_cuda::nemotron_vit::video_batch_capacity`), at least 1.
    pub batch: usize,
}

/// The shape of a video whose frames are `canvas` (`(width, height)`) pixels on a workspace of `max_patches` patches, or the reason it
/// cannot run. The video projection stages one tubelet's six planes where the image path stages three, so the tile must be at most half of
/// what the workspace holds.
pub fn video_shape(canvas: (usize, usize), max_patches: usize) -> Result<VideoShape, String> {
    let (w, h) = canvas;
    let (gh, gw) = nv::patch_grid(h, w).ok_or_else(|| {
        format!(
            "{w}x{h} pixels: both sides must be multiples of {} and the grid at most {} patches",
            2 * nv::PATCH,
            nv::MAX_PATCHES
        )
    })?;
    let n = gh * gw;
    let batch = nv::video_batch_capacity(max_patches, n);
    if batch == 0 {
        return Err(format!(
            "a {w}x{h} video tile is {n} patches ({} tokens); a tubelet's patch projection stages twice that in the encoder's workspace, which \
             holds {max_patches} patches ({} tokens: --max-image-tokens must be at least {})",
            n / PATCHES_PER_TOKEN,
            max_patches / PATCHES_PER_TOKEN,
            2 * n / PATCHES_PER_TOKEN
        ));
    }
    Ok(VideoShape { gh, gw, tokens: n / PATCHES_PER_TOKEN, batch })
}

// ------------------------------------------------------------------------------------------------------------------ the features

/// An image's encoder output on the device: bf16 `[tokens, 4096]`, one row per merged 2 x 2 block of patches in row-major order of the merged
/// grid. Written once by the encoder and only read afterwards (by the engine thread's prefill), so sharing it between the encoding thread
/// and the engine thread is sound.
pub struct NemotronImageFeatures {
    pub buf: DevBuf,
    pub tokens: usize,
    /// The merged grid (rows, columns); `tokens == merged.0 * merged.1`.
    pub merged: (usize, usize),
    /// The share of the image-feature budget this image holds until it is dropped (`None` outside the server's encoder).
    pub permit: Option<FeaturePermit>,
}

// SAFETY: the buffer is immutable after encoding and device memory is not tied to a host thread.
unsafe impl Sync for NemotronImageFeatures {}

/// One image's rows of features to write over the embedding rows of a prefill chunk: rows `src_row..src_row + n` of `feats` onto batch rows
/// `dst_row..dst_row + n`. Holds the features so they outlive the forward that reads them.
#[derive(Clone)]
pub struct SpliceOp {
    pub feats: Arc<NemotronImageFeatures>,
    pub dst_row: usize,
    pub src_row: usize,
    pub n: usize,
}

impl std::fmt::Debug for SpliceOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SpliceOp {{ dst_row: {}, src_row: {}, n: {} }}", self.dst_row, self.src_row, self.n)
    }
}

/// Which rows of which image fall inside the slice `at..end` of a prompt: for each image `(start, tokens)` of `spans` that overlaps it,
/// `(image index, dst_row, src_row, n)` with `dst_row` a row of the slice and `src_row` a row of the image's features. A slice may start
/// or end inside an image: its share is the overlap, and the shares of the slices that cover a prompt cover every image row exactly once.
pub fn plan_splice(spans: &[(usize, usize)], at: usize, end: usize) -> Vec<(usize, usize, usize, usize)> {
    spans
        .iter()
        .enumerate()
        .filter_map(|(i, &(start, tokens))| {
            let (lo, hi) = (start.max(at), start.saturating_add(tokens).min(end));
            (lo < hi).then(|| (i, lo - at, lo - start, hi - lo))
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------------------------ the tower

/// The tower's weights, its workspace and the encoder's buffers: everything resident for images, before any budget or service is wrapped
/// around it. Loaded once (before the memory plan is measured, so it comes out of the room the KV pool is planned from).
pub struct NemotronTower {
    w: nv::Weights,
    ws: nv::Workspace,
    pix: DevBuf,
    bad: DevBuf,
    st: Stream,
    max_tokens: usize,
}

impl NemotronTower {
    /// Reads the vision tensors of the pack at `pack` and allocates the workspace for images of up to `max_tokens` tokens (at most
    /// [`MAX_IMAGE_TOKENS`]). What was allocated is checked against [`weight_bytes`] and [`scratch_bytes`], item by item.
    pub fn load(pack: &Path, max_tokens: usize) -> Result<Self, ModelError> {
        if max_tokens == 0 || max_tokens > MAX_IMAGE_TOKENS {
            return Err(ModelError::Config(format!(
                "vision: images of {max_tokens} tokens; the tower takes 1 to {MAX_IMAGE_TOKENS}"
            )));
        }
        let patches = max_tokens * PATCHES_PER_TOKEN;
        let w =
            nv::Weights::load(pack).map_err(|e| ModelError::Config(format!("the vision tower of {}: {e}", pack.display())))?;
        let ws = nv::Workspace::new(patches)?;
        let pix = DevBuf::zeroed(patches * PATCH_DIM * 2)?;
        let bad = DevBuf::zeroed(BAD_BYTES as usize)?;
        let t = Self { w, ws, pix, bad, st: Stream::new_nonblocking()?, max_tokens };
        if t.w.bytes() != weight_bytes()
            || t.ws.bytes() + t.pix.bytes() as u64 + t.bad.bytes() as u64 != scratch_bytes(max_tokens)
        {
            return Err(ModelError::Config(format!(
                "the vision memory model and the tower's allocations disagree: weights {} B (model {}), scratch {} B (model {})",
                t.w.bytes(),
                weight_bytes(),
                t.ws.bytes() + t.pix.bytes() as u64 + t.bad.bytes() as u64,
                scratch_bytes(max_tokens)
            )));
        }
        Ok(t)
    }

    /// Device bytes of the weights.
    pub fn weight_bytes(&self) -> u64 {
        self.w.bytes()
    }

    /// Device bytes of the workspace and the encoder's input buffers.
    pub fn scratch_bytes(&self) -> u64 {
        self.ws.bytes() + self.pix.bytes() as u64 + self.bad.bytes() as u64
    }

    /// The most tokens one image may take.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// The tile `pre` describes: `(gh, gw, tokens)`, or the reason the tower cannot run it (a request error: nothing was launched).
    pub fn check(&self, pre: &gb_vision::Preprocessed) -> Result<(usize, usize, usize), ModelError> {
        let bad = |m: String| Err(ModelError::Request(format!("image not encodable: {m}")));
        let (w, h) = pre.size;
        let (t, gh, gw) = pre.grid;
        if t != 1 || pre.patch_dim != PATCH_DIM {
            return bad(format!(
                "grid {:?} with patches of {} values (the tower takes one still image of {PATCH_DIM}-value patches)",
                pre.grid, pre.patch_dim
            ));
        }
        if gh * nv::PATCH != h || gw * nv::PATCH != w {
            return bad(format!("a {w}x{h} canvas for a {gh}x{gw} patch grid"));
        }
        if pre.patches.len() != 3 * h * w {
            return bad(format!("{} pixel values for a 3 x {h} x {w} canvas", pre.patches.len()));
        }
        if nv::patch_grid(h, w) != Some((gh, gw)) {
            return bad(format!(
                "{w}x{h} pixels: both sides must be multiples of {} and the grid at most {} patches",
                2 * nv::PATCH,
                nv::MAX_PATCHES
            ));
        }
        if gh * gw > self.ws.max_patches() {
            return bad(format!(
                "{} patches; this engine's encoder is sized for {} ({} tokens: --max-image-tokens)",
                gh * gw,
                self.ws.max_patches(),
                self.max_tokens
            ));
        }
        Ok((gh, gw, gh * gw / PATCHES_PER_TOKEN))
    }

    /// The shape a video of `canvas` `(width, height)` runs as on this tower, or the request error that says why it cannot.
    pub fn check_video(&self, canvas: (usize, usize)) -> Result<VideoShape, ModelError> {
        video_shape(canvas, self.ws.max_patches()).map_err(|m| ModelError::Request(format!("video not encodable: {m}")))
    }

    /// Runs a video's tubelets (two consecutive frames each, the last frame repeated for an odd count) through the tower with the video
    /// patch projection, in batches of `shape.batch`, and writes tubelet `g`'s features, rounded to bf16, to the first
    /// `tokens * 4096` elements of `outs[g]`. Synchronises per batch. Returns the shape it ran.
    pub fn encode_video_into(
        &mut self,
        frames: &[Vec<u16>],
        canvas: (usize, usize),
        outs: &mut [DevBuf],
    ) -> Result<VideoShape, ModelError> {
        let shape = self.check_video(canvas)?;
        let (w, h) = canvas;
        let tubelets = gb_vision::nemotron_video::tubelet_count(frames.len());
        if tubelets == 0 || outs.len() != tubelets {
            return Err(ModelError::Request(format!(
                "{} frames make {tubelets} tubelets, {} output buffers given",
                frames.len(),
                outs.len()
            )));
        }
        let plane = 3 * h * w;
        if let Some(i) = frames.iter().position(|f| f.len() != plane) {
            return Err(ModelError::Request(format!("frame {}: {} values for a 3 x {h} x {w} canvas", i + 1, frames[i].len())));
        }
        let row = shape.tokens * OUT_HIDDEN;
        if let Some(o) = outs.iter().find(|o| o.bytes() < row * 2) {
            return Err(ModelError::Request(format!(
                "a tubelet's features are {} bytes, an output buffer holds {}",
                row * 2,
                o.bytes()
            )));
        }
        let mut done = 0;
        while done < tubelets {
            let batch = shape.batch.min(tubelets - done);
            let mut run = || -> Result<(), ModelError> {
                let st = &self.st;
                self.bad.memset_range_async(0, BAD_BYTES as usize, 0, st)?;
                for j in 0..batch {
                    let [a, c] = gb_vision::nemotron_video::tubelet_frame_indices(frames.len(), done + j);
                    // the tubelet's six planes: frame a's three, then frame c's, straight from the caller's frames (alive for the whole
                    // call); the bits widen to fp16 in place, as an image's do
                    // SAFETY: `pix` holds `max_patches * 768` fp16 elements and `2 * plane` is at most that (checked by `video_shape`).
                    unsafe {
                        st.memcpy_h2d_async(self.pix.ptr(), frames[a].as_ptr() as *const u8, plane * 2)?;
                        st.memcpy_h2d_async(self.pix.ptr_at(plane * 2), frames[c].as_ptr() as *const u8, plane * 2)?;
                    }
                    gb_cuda::glm_vision::bf16_to_f16(&mut self.pix, 2 * plane, &mut self.bad, st)?;
                    nv::video_embed(&self.w, &mut self.ws, &self.pix, h, w, j, st)?;
                }
                nv::video_forward(&self.w, &mut self.ws, h, w, batch, None, st)?;
                for j in 0..batch {
                    nv::f32_to_bf16_at(self.ws.mlp1(), j * row * 4, &mut outs[done + j], 0, row, st)?;
                }
                st.sync()?;
                Ok(())
            };
            if let Err(e) = run() {
                let _ = self.st.sync();
                return Err(e);
            }
            let mut bad = [0i32; 1];
            self.bad.download_range(0, &mut bad)?;
            if bad[0] != 0 {
                return Err(ModelError::Request(format!("{} pixel values are not finite in fp16", bad[0])));
            }
            done += batch;
        }
        Ok(shape)
    }

    /// Runs the tower on `pre` and writes its features, rounded to bf16, to the first `tokens * 4096` elements of `out`. Synchronises.
    pub fn encode_into(&mut self, pre: &gb_vision::Preprocessed, out: &mut DevBuf) -> Result<usize, ModelError> {
        let (gh, gw, tokens) = self.check(pre)?;
        let (h, w) = (gh * nv::PATCH, gw * nv::PATCH);
        if out.bytes() < tokens * FEATURE_ROW_BYTES {
            return Err(ModelError::Request("output buffer too small for the image's features".into()));
        }
        let n = 3 * h * w;
        let mut run = || -> Result<(), ModelError> {
            let st = &self.st;
            // the bf16 bits as the preprocessing made them, widened to fp16 in place on the device (the tower's input type)
            // SAFETY: `pix` holds `max_patches * 768` fp16 elements and `n` is at most that; the host slice outlives the stream's sync below.
            unsafe { st.memcpy_h2d_async(self.pix.ptr(), pre.patches.as_ptr() as *const u8, n * 2)? };
            self.bad.memset_range_async(0, BAD_BYTES as usize, 0, st)?;
            gb_cuda::glm_vision::bf16_to_f16(&mut self.pix, n, &mut self.bad, st)?;
            nv::tower_forward(&self.w, &mut self.ws, &self.pix, h, w, None, st)?;
            gb_cuda::vision::f32_to_bf16(self.ws.mlp1(), out, tokens * OUT_HIDDEN, st)?;
            st.sync()?;
            Ok(())
        };
        if let Err(e) = run() {
            let _ = self.st.sync();
            return Err(e);
        }
        let mut bad = [0i32; 1];
        self.bad.download_range(0, &mut bad)?;
        if bad[0] != 0 {
            return Err(ModelError::Request(format!("{} pixel values are not finite in fp16", bad[0])));
        }
        Ok(tokens)
    }
}

/// The tower as the server's image encoder (`serve::ImageEncoder`): the caller takes a permit for the features' bytes before any device work,
/// the features come back in a fresh bf16 buffer that holds the permit until it is dropped.
pub struct NemotronEncoder {
    tower: NemotronTower,
    budget: Arc<FeatureBudget>,
}

impl NemotronEncoder {
    pub fn new(tower: NemotronTower, budget: Arc<FeatureBudget>) -> Self {
        Self { tower, budget }
    }

    pub fn tower(&self) -> &NemotronTower {
        &self.tower
    }
}

impl ImageEncoder for NemotronEncoder {
    fn encode(&mut self, pre: &gb_vision::Preprocessed) -> Result<EncodedImage, ModelError> {
        let (gh, gw, tokens) = self.tower.check(pre)?;
        let bytes = feature_bytes(tokens);
        let permit = self.budget.acquire(bytes).ok_or_else(|| {
            ModelError::Busy(format!(
                "the memory for image features is in use ({:.0} MiB free of {:.0} MiB, this image needs {:.0} MiB); retry shortly",
                self.budget.free() as f64 / 1048576.0,
                self.budget.total() as f64 / 1048576.0,
                bytes as f64 / 1048576.0
            ))
        })?;
        let mut buf = DevBuf::alloc(bytes as usize)?;
        self.tower.encode_into(pre, &mut buf)?;
        let merged = (gh / 2, gw / 2);
        Ok(EncodedImage {
            tokens,
            merged,
            features: Arc::new(NemotronImageFeatures { buf, tokens, merged, permit: Some(permit) }),
        })
    }

    /// A video's tubelets, one feature handle each. Without `keep` every row of every tubelet is kept. With it, the bf16 rows of all the
    /// tubelets (tubelet-major) go to the caller's function on the host, which says which rows survive (EVS:
    /// `gb_vision::nemotron_video::retention_mask_bits`), and the survivors are gathered on the host into a buffer per tubelet: the
    /// unpruned rows are dropped (and their budget given back) once the kept ones are resident. A tubelet that keeps all its rows has the
    /// tile's merged grid, any other `(1, kept)`, as the server's spans say (`ImageSpan::merged`); one that keeps none has
    /// `tokens == 0` and no device rows.
    fn encode_video(&mut self, video: &VideoInput<'_>, keep: Option<KeepRows<'_>>) -> Result<Vec<EncodedImage>, ModelError> {
        let shape = self.tower.check_video(video.canvas)?;
        let tubelets = gb_vision::nemotron_video::tubelet_count(video.frames.len());
        if tubelets == 0 {
            return Err(ModelError::Request("a video needs at least one frame".into()));
        }
        let n = shape.tokens;
        let full = feature_bytes(n);
        let unpruned = full * tubelets as u64;
        // pruning holds the unpruned rows and the kept ones at once (at most as many again)
        let peak = if keep.is_some() { unpruned.saturating_mul(2) } else { unpruned };
        if peak > self.budget.total() {
            return Err(ModelError::Request(format!(
                "the video's features need up to {:.0} MiB of the image-feature memory ({tubelets} tubelets of {n} rows{}) and the plan holds \
                 back {:.0} MiB (slots x images per request x --max-image-tokens): send fewer frames or raise --max-image-tokens",
                peak as f64 / 1048576.0,
                if keep.is_some() { ", twice while pruning" } else { "" },
                self.budget.total() as f64 / 1048576.0
            )));
        }
        let busy = |budget: &FeatureBudget, need: u64| {
            ModelError::Busy(format!(
                "the memory for image features is in use ({:.0} MiB free of {:.0} MiB, this video needs {:.0} MiB); retry shortly",
                budget.free() as f64 / 1048576.0,
                budget.total() as f64 / 1048576.0,
                need as f64 / 1048576.0
            ))
        };
        // all of it or nothing: a permit for every tubelet (or one for the whole unpruned set while pruning) before any device work
        let mut permits: Vec<FeaturePermit> = Vec::new();
        if keep.is_some() {
            permits.push(self.budget.acquire(unpruned).ok_or_else(|| busy(&self.budget, unpruned))?);
        } else {
            for _ in 0..tubelets {
                permits.push(self.budget.acquire(full).ok_or_else(|| busy(&self.budget, unpruned))?);
            }
        }
        let mut outs: Vec<DevBuf> = Vec::with_capacity(tubelets);
        for _ in 0..tubelets {
            outs.push(DevBuf::alloc(full as usize)?);
        }
        self.tower.encode_video_into(video.frames, video.canvas, &mut outs)?;
        let merged_full = (shape.gh / 2, shape.gw / 2);
        let Some(keep) = keep else {
            return Ok(outs
                .into_iter()
                .zip(permits)
                .map(|(buf, permit)| EncodedImage {
                    tokens: n,
                    merged: merged_full,
                    features: Arc::new(NemotronImageFeatures { buf, tokens: n, merged: merged_full, permit: Some(permit) }),
                })
                .collect());
        };
        // the rows on the host, the mask, the survivors
        let mut host = vec![0u16; tubelets * n * OUT_HIDDEN];
        for (g, out) in outs.iter().enumerate() {
            out.download_range(0, &mut host[g * n * OUT_HIDDEN..(g + 1) * n * OUT_HIDDEN])?;
        }
        let mask = keep(&host).map_err(|e| ModelError::Request(format!("video pruning: {e}")))?;
        if mask.len() != tubelets * n {
            return Err(ModelError::Request(format!("video pruning returned {} flags for {} rows", mask.len(), tubelets * n)));
        }
        let kept: Vec<usize> = mask.chunks(n).map(|c| c.iter().filter(|&&k| k).count()).collect();
        let mut kept_permits: Vec<Option<FeaturePermit>> = Vec::with_capacity(tubelets);
        for &k in &kept {
            if k == 0 {
                kept_permits.push(None);
                continue;
            }
            let need = feature_bytes(k);
            kept_permits.push(Some(self.budget.acquire(need).ok_or_else(|| busy(&self.budget, unpruned + need))?));
        }
        let mut result = Vec::with_capacity(tubelets);
        for (g, (out, permit)) in outs.into_iter().zip(kept_permits).enumerate() {
            let k = kept[g];
            let (buf, merged) = if k == n {
                (out, merged_full)
            } else if k == 0 {
                (DevBuf::alloc(0)?, (0, 0))
            } else {
                let mut rows = Vec::with_capacity(k * OUT_HIDDEN);
                for (r, &keep_row) in mask[g * n..(g + 1) * n].iter().enumerate() {
                    if keep_row {
                        rows.extend_from_slice(&host[(g * n + r) * OUT_HIDDEN..(g * n + r + 1) * OUT_HIDDEN]);
                    }
                }
                (DevBuf::from_slice(&rows)?, (1, k))
            };
            result.push(EncodedImage {
                tokens: k,
                merged,
                features: Arc::new(NemotronImageFeatures { buf, tokens: k, merged, permit }),
            });
        }
        drop(permits); // the unpruned rows' share, given back once the kept rows are resident
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cap_is_the_towers_and_never_more() {
        assert_eq!(MAX_IMAGE_TOKENS, 3328);
        assert_eq!(image_token_cap(4096), 3328, "the library default is above the tower's");
        assert_eq!(image_token_cap(3328), 3328);
        assert_eq!(image_token_cap(256), 256);
    }

    #[test]
    fn the_weights_are_the_published_towers_bytes() {
        // 32 blocks of 39,388,160 B (qkv 9,830,400 + proj 3,276,800 + fc1 13,107,200 + fc2 13,107,200 in fp16, the vectors in fp32), the
        // fp32 position table (83,886,080), mlp1 (209,715,200 + 167,772,160) and the small globals
        let block = 2 * 2 * 1280 * 4
            + 3 * 1280 * 1280 * 2
            + 3 * 1280 * 4
            + 1280 * 1280 * 2
            + 1280 * 4
            + 5120 * 1280 * 2
            + 5120 * 4
            + 1280 * 5120 * 2
            + 1280 * 4;
        assert_eq!(block, 39_388_160);
        // the image embedder 1,966,080 and the video embedder 3,932,160 (1280 x 1536 x 2)
        assert_eq!(
            weight_bytes(),
            32 * 39_388_160 + 1_966_080 + 3_932_160 + 83_886_080 + 51_200 + 10_240 + 20_480 + 209_715_200 + 167_772_160
        );
        assert_eq!(weight_bytes(), 1_727_774_720);
    }

    #[test]
    fn a_slice_takes_the_overlap_with_each_image_and_the_slices_of_a_prompt_cover_every_image_row_once() {
        // an image of 100 rows from position 11, another of 7 from 200
        let spans = [(11usize, 100usize), (200, 7)];
        // a slice that ends inside the first image, one that starts inside it, one that holds none, one holding the second whole
        assert_eq!(plan_splice(&spans, 0, 64), vec![(0, 11, 0, 53)]);
        assert_eq!(plan_splice(&spans, 64, 128), vec![(0, 0, 53, 47)]);
        assert_eq!(plan_splice(&spans, 111, 200), vec![]);
        assert_eq!(plan_splice(&spans, 192, 256), vec![(1, 8, 0, 7)]);
        // exactly on the image's edges
        assert_eq!(plan_splice(&spans, 0, 11), vec![]);
        assert_eq!(plan_splice(&spans, 111, 112), vec![]);
        assert_eq!(plan_splice(&spans, 10, 12), vec![(0, 1, 0, 1)]);
        // a slice inside one image
        assert_eq!(plan_splice(&spans, 40, 41), vec![(0, 0, 29, 1)]);
        // every grid of cuts: each image row lands in exactly one slice, at the destination row its position says
        for chunk in [1usize, 5, 16, 37, 64, 100, 300] {
            let mut seen = [vec![0u32; 100], vec![0u32; 7]];
            let mut at = 0;
            while at < 300 {
                let end = (at + chunk).min(300);
                for (i, dst, src, n) in plan_splice(&spans, at, end) {
                    for k in 0..n {
                        seen[i][src + k] += 1;
                        assert_eq!(at + dst + k, spans[i].0 + src + k, "chunk {chunk}: row {k} of a share of image {i}");
                    }
                }
                at = end;
            }
            assert!(seen.iter().flatten().all(|&c| c == 1), "chunk {chunk}");
        }
    }

    #[test]
    fn a_video_tile_runs_when_the_workspace_holds_twice_its_patches() {
        let max = MAX_IMAGE_TOKENS * PATCHES_PER_TOKEN; // 13,312
                                                        // 16:9 and 4:3 tiles (42 x 24, 36 x 28: 252 tokens), the square tile (32 x 32: 256) and a portrait one (26 x 40: 260)
        assert_eq!(video_shape((42 * 16, 24 * 16), max), Ok(VideoShape { gh: 24, gw: 42, tokens: 252, batch: 13 }));
        assert_eq!(video_shape((512, 512), max), Ok(VideoShape { gh: 32, gw: 32, tokens: 256, batch: 12 }));
        assert_eq!(video_shape((26 * 16, 40 * 16), max).unwrap().tokens, 260);
        // --max-image-tokens 256 is 1,024 patches: a video tile needs 2,048 (--max-image-tokens 512)
        let e = video_shape((512, 512), 1024).unwrap_err();
        assert!(e.contains("--max-image-tokens must be at least 512"), "{e}");
        assert!(video_shape((512, 512), 2048).is_ok());
        // a tile the tower cannot run at all
        assert!(video_shape((500, 500), max).unwrap_err().contains("multiples of 32"));
        // the batch is what the stream's rows hold
        assert_eq!(video_shape((512, 512), 2048).unwrap().batch, 1);
    }

    #[test]
    fn the_workspace_at_the_towers_largest_image() {
        // 13,312 patches: the fp16 patches, the fp32 stream over 13,322 tokens, and the projector's buffers over 3,328
        assert_eq!(workspace_bytes(13_312), 688_671_744);
        assert_eq!(input_bytes(13_312), 13_312 * 768 * 2 + 4);
        assert_eq!(scratch_bytes(MAX_IMAGE_TOKENS), 688_671_744 + 20_447_232 + 4);
        // it grows with the image limit, so a lower --max-image-tokens buys memory
        assert!(scratch_bytes(1024) < scratch_bytes(2048));
        assert_eq!(feature_bytes(3328), 3328 * 8192);
    }
}
