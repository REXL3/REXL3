//! The image ingress of the Nemotron format without a model: everything between the request's image parts and the engine's `ImageSpan`s that
//! needs no GPU, in one place the vision tower sits behind.
//!
//! For one image: decode (`image.convert("RGB")`: alpha dropped), the tile and its token count from the size alone (refused before any pixel work
//! when it exceeds the per-image limit), the float32 canvas the processor makes (`gb_vision::nemotron`), and the content key the prefix cache
//! uses (`sched::image_content_hash` over the canvas hash, the canvas size and the preprocessing's identity: the same value `http.rs` computes
//! for the other families). For a request: the images in template order, the rendered and tokenized prompt, the `<img>` + `<image>` x N +
//! `</img>` expansion, and one [`gb_engine::ImageSpan`] per image.
//!
//! **Where the model plugs in.** [`NemotronEncoder`] is the one seam: it takes a [`PreparedImage`] (float32 `[3][H][W]`, the grid, the key) and
//! returns whatever the backend splices (`ImageSpan::features`). Without an encoder the span carries the prepared image itself
//! ([`Ingested::prepared`] and `features`), so a backend can run the tower lazily; the request, the ids and the spans are the same either way.
//! The server's generic path (`http.rs`: `ImagePreprocessor` + `VisionService` + `ImageEncoder`) reaches the same preprocessing through
//! `NemotronPreprocessing`'s `ImagePreprocessor` impl and computes the same keys.

use super::video::{self, VideoRef, VideoSettings, VIDEO_TAG};
use super::{OmniFormat, IMAGE_TOKEN, SCRATCH};
use crate::api::ApiError;
use crate::chat_format::ChatFormat;
use crate::preparation::RequestWork;
use crate::tokenizer::Tok;
use crate::vision::{ImageRef, Limits};
use gb_model::error::ModelError;
use gb_vision::nemotron::{NemotronPreprocessing, TilePlan};
use gb_vision::nemotron_video::{
    frame_labels, kept_per_tubelet, plan_video, retention_mask_bits, PreprocessedVideo, SumOrder, VideoBuilder, VideoPlan,
    FEATURE_WIDTH,
};
use gb_vision::preprocessor::ImagePreprocessor;
use gb_vision::VisionError;
use serde_json::{Map, Value};
use std::any::Any;
use std::sync::{Arc, Mutex};

/// The rendered-prompt byte limit [`NemotronImagePrep::ingest`] renders under (the `--max-rendered-prompt-mib` default for the 262,144-token
/// window: `budget::default_rendered_bytes`).
pub const RENDER_LIMIT_BYTES: usize = 4 << 20;

/// One image after everything that needs no model.
pub struct PreparedImage {
    /// The processor's `pixel_values[0]`: float32 `[3][H][W]`, `(x / 255 - mean) / std`.
    pub pixels: Vec<f32>,
    /// The decoded image's size `(width, height)`.
    pub source: (usize, usize),
    /// The canvas `(width, height)` in pixels: `(16 * w_patches, 16 * h_patches)`.
    pub canvas: (usize, usize),
    /// The patch grid `(h_patches, w_patches)`.
    pub grid: (usize, usize),
    /// The grid after the pixel shuffle `(rows, columns)`; `tokens == rows * columns`.
    pub merged: (usize, usize),
    /// Placeholder tokens: the rows of encoder output.
    pub tokens: usize,
    /// FNV-1a over the canvas size and the bytes of `pixels` (`gb_vision::nemotron::canvas_hash`).
    pub pixel_hash: u64,
    /// `ImageSpan::key`: `image_content_hash(pixel_hash, canvas, preprocessing id)`, never 0.
    pub key: u64,
}

impl PreparedImage {
    /// The tower's input as the model makes it (bf16, round to nearest even).
    pub fn bf16_bits(&self) -> Vec<u16> {
        self.pixels.iter().map(|&v| gb_vision::nemotron::f32_to_bf16_bits(v)).collect()
    }
}

impl std::fmt::Debug for PreparedImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PreparedImage {{ source: {:?}, canvas: {:?}, grid: {:?}, tokens: {}, key: {:#x} }}",
            self.source, self.canvas, self.grid, self.tokens, self.key
        )
    }
}

/// The vision tower and projector behind the ingress (the model side's, `M3`): a prepared image in, the features the backend splices out.
/// Runs on the caller's thread; the implementation owns its own concurrency and memory accounting.
pub trait NemotronEncoder: Send + Sync {
    fn encode(&self, image: &PreparedImage) -> Result<Arc<dyn Any + Send + Sync>, ModelError>;
}

/// What an ingress of one request may add: chat-template variables, the encoder behind the seam, and the request's cancellation.
#[derive(Default)]
pub struct IngestOptions<'a> {
    pub kwargs: Option<&'a Map<String, Value>>,
    pub encoder: Option<&'a dyn NemotronEncoder>,
    pub work: Option<&'a RequestWork>,
    /// The tower behind the video seam. Required when pruning is on (which rows are kept depends on the features); without it a video's spans
    /// carry [`DeferredTubelet`]s and the backend runs the tower when it needs the rows.
    pub video_encoder: Option<&'a dyn NemotronVideoEncoder>,
}

/// A request's prompt with its images: ids with every `<image>` run expanded, one span per image, and the prepared images in order.
pub struct Ingested {
    pub ids: Vec<u32>,
    pub starts_in_reasoning: bool,
    /// One span per image, or per non-empty tubelet of the video (a tubelet whose rows were all pruned has no span).
    pub images: Vec<gb_engine::ImageSpan>,
    pub prepared: Vec<Arc<PreparedImage>>,
    /// The request's video, when it had one (then `prepared` is empty: images and video do not mix).
    pub video: Option<Arc<PreparedVideo>>,
}

/// What a failed encode of image `r` means to the client: the same mapping `http.rs` makes for the other families.
pub fn encode_error(r: &ImageRef, e: ModelError) -> ApiError {
    let at = format!("messages[{}].content[{}]", r.message, r.part);
    match e {
        ModelError::Busy(m) => ApiError::status(503, "server_overloaded", format!("{at}: {m}")).retry_after(5),
        ModelError::Cuda(_) | ModelError::StateLost(_) => {
            ApiError::status(500, "server_error", format!("{at}: image encoding failed: {e}"))
        }
        e => ApiError::bad(format!("{at}: image encoding failed: {e}")),
    }
}

/// The preprocessing, its limits and its identity: what turns a request's images into [`PreparedImage`]s and spans.
pub struct NemotronImagePrep {
    pre: Arc<NemotronPreprocessing>,
    limits: Limits,
    /// FNV-1a of the preprocessing's `identity()`, as `http.rs` computes `pp_id`.
    id_key: u64,
    /// Video input (limits, pruning), when on: `ingest` then takes a request with a video part.
    video: Option<Arc<VideoSettings>>,
}

fn fnv_str(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

impl NemotronImagePrep {
    pub fn new(pre: Arc<NemotronPreprocessing>, limits: Limits) -> Self {
        let id_key = fnv_str(&pre.identity());
        Self { pre, limits, id_key, video: None }
    }

    /// The parameters the format read from the model folder, or `None` when it had none. The format's video settings (`OmniFormat::with_video`)
    /// come along: one source of truth for what a video may be.
    pub fn from_format(format: &OmniFormat, limits: Limits) -> Option<Self> {
        format.preprocessing().map(|p| Self { video: format.video().cloned(), ..Self::new(p.clone(), limits) })
    }

    /// Take video requests under these settings.
    pub fn with_video(mut self, settings: Arc<VideoSettings>) -> Self {
        self.video = Some(settings);
        self
    }

    pub fn video_settings(&self) -> Option<&Arc<VideoSettings>> {
        self.video.as_ref()
    }

    pub fn preprocessing(&self) -> &Arc<NemotronPreprocessing> {
        &self.pre
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// The preprocessing's identity (`ImagePreprocessor::id`): part of every image's cache key.
    pub fn id(&self) -> String {
        self.pre.identity()
    }

    /// The tile of a `width x height` image sent with `n_images - 1` others, from the size alone, within the per-image token limit.
    pub fn plan(&self, width: usize, height: usize, n_images: usize) -> Result<TilePlan, ApiError> {
        let t = self.pre.plan_for(width, height, n_images).map_err(|e| ApiError::bad(e.to_string()))?;
        if t.tokens > self.limits.max_tokens_per_image {
            let (cw, ch) = t.canvas();
            return Err(ApiError::bad(format!(
                "the image would occupy {} tokens ({cw}x{ch} after resizing); the limit is {}",
                t.tokens, self.limits.max_tokens_per_image
            )));
        }
        Ok(t)
    }

    /// The cache key of a canvas: what `http.rs` computes as `image_content_hash(pixel_hash, w, h, pp_id)` (0 reads as "unknown", so never 0).
    pub fn key(&self, pixel_hash: u64, canvas: (usize, usize)) -> u64 {
        gb_sched::image_content_hash(pixel_hash, canvas.0 as u32, canvas.1 as u32, self.id_key).max(1)
    }

    /// Decode and preprocess one image of a request that carries `n_images` of them, under the limits: bytes, then the tile and its tokens
    /// (before the resize), then the pixels. Cooperative cancellation is checked between the steps.
    pub fn prepare(
        &self,
        r: &ImageRef,
        bytes: &[u8],
        n_images: usize,
        work: Option<&RequestWork>,
    ) -> Result<PreparedImage, ApiError> {
        let at = format!("messages[{}].content[{}]", r.message, r.part);
        let check = || work.map_or(Ok(()), |w| w.check());
        check()?;
        if bytes.len() > self.limits.max_bytes {
            return Err(ApiError::status(
                413,
                "invalid_request_error",
                format!("{at}: image is larger than {} bytes", self.limits.max_bytes),
            ));
        }
        let img = self.pre.decode(bytes).map_err(|e| ApiError::bad(format!("{at}: {e}")))?;
        check()?;
        let plan = self.plan(img.width, img.height, n_images).map_err(|mut e| {
            e.message = format!("{at}: {}", e.message);
            e
        })?;
        check()?;
        let p = self.pre.preprocess_for(&img, n_images).map_err(|e| ApiError::bad(format!("{at}: {e}")))?;
        check()?;
        debug_assert_eq!((p.w_patches, p.h_patches, p.tokens), (plan.w_patches, plan.h_patches, plan.tokens));
        let canvas = p.canvas();
        let key = self.key(p.pixel_hash, canvas);
        Ok(PreparedImage {
            source: (img.width, img.height),
            canvas,
            grid: (p.h_patches, p.w_patches),
            merged: plan.merged(),
            tokens: p.tokens,
            pixel_hash: p.pixel_hash,
            key,
            pixels: p.pixels,
        })
    }

    /// One [`gb_engine::ImageSpan`] per prepared image, placed by `format.expand_images` on the tokenized prompt `ids` (one `<image>` per image):
    /// the expanded ids and the spans, each carrying `features[i]`.
    pub fn assemble(
        &self,
        format: &dyn ChatFormat,
        tok: &Tok,
        ids: &[u32],
        prepared: &[Arc<PreparedImage>],
        features: Vec<Arc<dyn Any + Send + Sync>>,
    ) -> Result<(Vec<u32>, Vec<gb_engine::ImageSpan>), ApiError> {
        assert_eq!(prepared.len(), features.len(), "one feature handle per image");
        let counts: Vec<usize> = prepared.iter().map(|p| p.tokens).collect();
        let (ids, spans) = format.expand_images(tok, ids, &counts)?;
        let images = spans
            .iter()
            .zip(prepared.iter().zip(features))
            .map(|(sp, (p, features))| gb_engine::ImageSpan {
                start: sp.start,
                tokens: p.tokens,
                merged: p.merged,
                features,
                key: p.key,
            })
            .collect();
        Ok((ids, images))
    }

    /// The whole ingress of a chat request, without a model call unless an `encoder` is given: refuse what the format refuses, find the images in
    /// template order, fetch their bytes with `fetch` (a data URI by default, see [`data_uri`]; a remote fetch is the caller's), prepare them
    /// under the limits, render and tokenize the prompt, expand the placeholders and build the spans.
    pub fn ingest(
        &self,
        format: &dyn ChatFormat,
        tok: &Tok,
        messages: &Value,
        opts: IngestOptions<'_>,
        mut fetch: impl FnMut(&ImageRef) -> Result<Vec<u8>, ApiError>,
    ) -> Result<Ingested, ApiError> {
        let IngestOptions { kwargs, encoder, work, video_encoder } = opts;
        let no_kwargs = Map::new();
        let kwargs = kwargs.unwrap_or(&no_kwargs);
        format.check_media(messages, true)?;
        // a video part is a request of its own kind (the model takes no image beside a video): frames, tubelets, the labelled prompt
        if video::has_video(messages) {
            return self.ingest_video(format, tok, messages, kwargs, video_encoder, work, fetch);
        }
        let refs = format.extract_images(messages, &self.limits)?;
        let n = refs.len();
        let mut prepared = Vec::with_capacity(n);
        let mut total = 0usize;
        for r in &refs {
            let bytes = fetch(r)?;
            total = total.checked_add(bytes.len()).ok_or_else(|| ApiError::bad("image aggregate size overflow"))?;
            if total > self.limits.max_total_bytes {
                return Err(ApiError::status(
                    413,
                    "invalid_request_error",
                    format!(
                        "messages[{}].content[{}]: aggregate image bytes exceed {}",
                        r.message, r.part, self.limits.max_total_bytes
                    ),
                ));
            }
            prepared.push(Arc::new(self.prepare(r, &bytes, n, work)?));
        }
        let rendered = format.render_bounded(messages, kwargs, RENDER_LIMIT_BYTES).map_err(|e| ApiError::bad(e.to_string()))?;
        let ids = tok.encode_bounded(&rendered.text, RENDER_LIMIT_BYTES)?;
        let mut features: Vec<Arc<dyn Any + Send + Sync>> = Vec::with_capacity(n);
        for (r, p) in refs.iter().zip(&prepared) {
            features.push(match encoder {
                Some(e) => e.encode(p).map_err(|err| encode_error(r, err))?,
                // the prepared image itself: the backend runs the tower when it needs the rows
                None => p.clone(),
            });
        }
        let (ids, images) = self.assemble(format, tok, &ids, &prepared, features)?;
        Ok(Ingested { ids, starts_in_reasoning: rendered.starts_in_reasoning, images, prepared, video: None })
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Video (`docs/NEMOTRON_VIDEO.md`).

/// A video after everything that needs no model: its frames in bf16 (what `forward_video` feeds the tower), the tile and the counts, the
/// processor's labels and the content key.
pub struct PreparedVideo {
    pub pre: PreprocessedVideo,
    pub plan: VideoPlan,
    /// The first frame's decoded size `(width, height)`.
    pub source: (usize, usize),
    /// One label per tubelet (`Frame 1 and frame 2: `).
    pub labels: Vec<String>,
    /// The video's content key (never 0): the frames' canvas hashes in order, the canvas, the preprocessing and the video settings.
    pub key: u64,
}

impl PreparedVideo {
    pub fn tubelets(&self) -> usize {
        self.plan.tubelets
    }

    /// bf16 bits of tubelet `g`, planes `[6][H][W]`: the tower's video input.
    pub fn tubelet_bf16(&self, g: usize) -> Vec<u16> {
        self.pre.tubelet(g)
    }
}

impl std::fmt::Debug for PreparedVideo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PreparedVideo {{ frames: {}, tubelets: {}, tokens_per_tubelet: {}, canvas: {:?}, key: {:#x} }}",
            self.plan.frames,
            self.plan.tubelets,
            self.plan.tokens_per_tubelet,
            self.pre.canvas(),
            self.key
        )
    }
}

/// What an unencoded tubelet's span carries (no encoder given): the video and which tubelet, so a backend can run the tower when it needs the
/// rows.
#[derive(Clone, Debug)]
pub struct DeferredTubelet {
    pub video: Arc<PreparedVideo>,
    pub index: usize,
}

/// One tubelet's encoder output: the rows it keeps (all of them without pruning) and the handle the backend splices.
pub struct EncodedTubelet {
    pub tokens: usize,
    pub features: Arc<dyn Any + Send + Sync>,
}

pub struct EncodedVideo {
    pub tubelets: Vec<EncodedTubelet>,
}

/// The mask EVS hands the encoder to compact with: `[tubelets x tokens]` flags over the bf16 features, tubelet-major, 4,096 values a row.
pub type KeepRows<'a> = &'a dyn Fn(&[u16]) -> Result<Vec<bool>, String>;

/// The vision tower behind the video seam (the model side's, M4): the prepared tubelets in, one feature handle per tubelet out.
///
/// With pruning on, `keep` is given and the encoder must call it ONCE with every tubelet's rows (bf16, tubelet-major, `tokens x 4096` each,
/// before any is dropped), keep exactly the rows it flags in order, and report each tubelet's kept count. The host function is the reference
/// (`gb_vision::nemotron_video::retention_mask_bits`), so the rows kept never depend on the device. Without it, every row is kept.
pub trait NemotronVideoEncoder: Send + Sync {
    fn encode_video(&self, video: &PreparedVideo, keep: Option<KeepRows<'_>>) -> Result<EncodedVideo, ModelError>;
}

fn fnv_bytes(h: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// What a vision error says, without its "unsupported image:" lead (the request part and the frame are named instead).
fn vision_message(e: VisionError) -> String {
    match e {
        VisionError::Unsupported(m) => m,
        VisionError::Decode(m) => format!("cannot decode the image: {m}"),
    }
}

impl NemotronImagePrep {
    fn video_id_key(&self, v: &VideoSettings) -> u64 {
        fnv_str(&format!("{}|{}", self.pre.identity(), v.config.identity()))
    }

    /// The plan of a video of `frames` frames of `width` x `height`, from the size alone, within the limits: the per-image token limit for the
    /// tile, the video's token limit for the whole (before pruning), and (when configured) the square-tile rule of the model's EVS.
    pub fn plan_video(&self, width: usize, height: usize, frames: usize) -> Result<VideoPlan, ApiError> {
        let v = self.video.as_ref().ok_or_else(|| ApiError::bad("video input is not enabled on this server"))?;
        let p = plan_video(&self.pre, &v.config, width, height, frames).map_err(|e| ApiError::bad(vision_message(e)))?;
        if p.tokens_per_tubelet > self.limits.max_tokens_per_image {
            let (cw, ch) = p.tile.canvas();
            return Err(ApiError::bad(format!(
                "a {width}x{height} frame would occupy {} tokens ({cw}x{ch} after resizing); the limit is {} (--max-image-tokens)",
                p.tokens_per_tubelet, self.limits.max_tokens_per_image
            )));
        }
        // the tower stages a tubelet's two frames' patches where it stages one image's: the encoder's workspace must hold twice the tile
        if 2 * p.tokens_per_tubelet > self.limits.max_tokens_per_image {
            return Err(ApiError::bad(format!(
                "a {width}x{height} frame would occupy {} tokens a tubelet; the video projection stages two frames' patches in the image \
                 encoder's workspace, so --max-image-tokens must be at least {} for this video (it is {})",
                p.tokens_per_tubelet,
                2 * p.tokens_per_tubelet,
                self.limits.max_tokens_per_image
            )));
        }
        if p.tokens > v.limits.max_tokens {
            return Err(ApiError::bad(format!(
                "the video would occupy {} tokens ({} tubelets of {}, {frames} frames); the limit is {} (--max-video-tokens)",
                p.tokens, p.tubelets, p.tokens_per_tubelet, v.limits.max_tokens
            )));
        }
        Ok(p)
    }

    /// Decode and preprocess the frames of `r`, one at a time (a decoded frame is dropped as soon as it is in bf16), under the limits: bytes
    /// per frame and in all, then the plan from the first frame's size (before any resize), then one tile for every frame.
    pub fn prepare_video(
        &self,
        r: &VideoRef,
        mut fetch: impl FnMut(&ImageRef) -> Result<Vec<u8>, ApiError>,
        work: Option<&RequestWork>,
    ) -> Result<PreparedVideo, ApiError> {
        let v = self.video.clone().ok_or_else(|| ApiError::bad("video input is not enabled on this server"))?;
        let at = r.at();
        let n = r.frames.len();
        let check = || work.map_or(Ok(()), |w| w.check());
        let mut builder = VideoBuilder::new(&self.pre);
        let mut total = 0usize;
        let mut source = (0, 0);
        for (k, url) in r.frames.iter().enumerate() {
            check()?;
            let fref = ImageRef { message: r.message, part: r.part, url: url.clone() };
            let bytes = fetch(&fref).map_err(|mut e| {
                // the part's own prefix is replaced by the part and the frame
                let plain = e.message.strip_prefix(&format!("{at}: ")).unwrap_or(&e.message).to_string();
                e.message = format!("{at}: frame {}: {plain}", k + 1);
                e
            })?;
            if bytes.len() > self.limits.max_bytes {
                return Err(ApiError::status(
                    413,
                    "invalid_request_error",
                    format!("{at}: frame {}: image is larger than {} bytes", k + 1, self.limits.max_bytes),
                ));
            }
            total = total.checked_add(bytes.len()).ok_or_else(|| ApiError::bad("video aggregate size overflow"))?;
            if total > self.limits.max_total_bytes {
                return Err(ApiError::status(
                    413,
                    "invalid_request_error",
                    format!("{at}: frame {}: the frames' bytes exceed {} in all", k + 1, self.limits.max_total_bytes),
                ));
            }
            let img =
                self.pre.decode(&bytes).map_err(|e| ApiError::bad(format!("{at}: frame {}: {}", k + 1, vision_message(e))))?;
            drop(bytes);
            check()?;
            if k == 0 {
                source = (img.width, img.height);
                // refuse a video over its limits from the first frame's size alone, before any frame is resized
                self.plan_video(img.width, img.height, n).map_err(|mut e| {
                    e.message = format!("{at}: {}", e.message);
                    e
                })?;
            }
            builder.push(&img).map_err(|e| ApiError::bad(format!("{at}: {}", vision_message(e))))?;
        }
        check()?;
        let pre = builder.finish().map_err(|e| ApiError::bad(format!("{at}: {}", vision_message(e))))?;
        let plan = self.plan_video(source.0, source.1, n)?;
        let labels = frame_labels(n, &r.metadata).map_err(|e| ApiError::bad(format!("{at}: {}", vision_message(e))))?;
        let (cw, ch) = pre.canvas();
        let key = gb_sched::image_content_hash(pre.pixel_hash, cw as u32, ch as u32, self.video_id_key(&v)).max(1);
        Ok(PreparedVideo { pre, plan, source, labels, key })
    }

    /// The identity of tubelet `g`'s features. Without pruning they are a function of its two frames alone (so two videos that share a pair of
    /// frames share it, and a video with frames appended keeps its first tubelets' keys); with pruning they depend on the whole video, which
    /// the video's key covers. Never equal to a still image's key: the marker differs and the image path's hash is over one canvas.
    pub fn tubelet_key(&self, video: &PreparedVideo, g: usize, kept: usize) -> u64 {
        let v = self.video.as_ref();
        let pruned = v.is_some_and(|v| v.config.pruning()) && video.tubelets() > 1;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        if pruned {
            h = fnv_bytes(h, b"nemotron-tubelet-evs-v1");
            h = fnv_bytes(h, &video.key.to_le_bytes());
            h = fnv_bytes(h, &(g as u64).to_le_bytes());
            h = fnv_bytes(h, &(kept as u64).to_le_bytes());
        } else {
            h = fnv_bytes(h, b"nemotron-tubelet-v1");
            let [a, b] = gb_vision::nemotron_video::tubelet_frame_indices(video.pre.frames.len(), g);
            h = fnv_bytes(h, &video.pre.frame_hashes[a].to_le_bytes());
            h = fnv_bytes(h, &video.pre.frame_hashes[b].to_le_bytes());
        }
        let (cw, ch) = video.pre.canvas();
        let id = v.map_or(self.id_key, |v| self.video_id_key(v));
        gb_sched::image_content_hash(h, cw as u32, ch as u32, id).max(1)
    }

    /// The ingress of a request with a video: fetch and prepare the frames under the limits, render and check the prompt (exactly one
    /// `<video>`), encode (the pruning decides the kept rows with the features), expand the text and tokenize, and build one span per non-empty
    /// tubelet.
    #[allow(clippy::too_many_arguments)]
    pub fn ingest_video(
        &self,
        format: &dyn ChatFormat,
        tok: &Tok,
        messages: &Value,
        kwargs: &Map<String, Value>,
        encoder: Option<&dyn NemotronVideoEncoder>,
        work: Option<&RequestWork>,
        fetch: impl FnMut(&ImageRef) -> Result<Vec<u8>, ApiError>,
    ) -> Result<Ingested, ApiError> {
        self.ingest_video_bounded(format, tok, messages, kwargs, RENDER_LIMIT_BYTES, encoder, work, fetch)
    }

    /// [`NemotronImagePrep::ingest_video`] under a rendered-prompt limit of `render_limit` bytes (the server's `--max-rendered-prompt-mib`).
    #[allow(clippy::too_many_arguments)]
    pub fn ingest_video_bounded(
        &self,
        format: &dyn ChatFormat,
        tok: &Tok,
        messages: &Value,
        kwargs: &Map<String, Value>,
        render_limit: usize,
        encoder: Option<&dyn NemotronVideoEncoder>,
        work: Option<&RequestWork>,
        fetch: impl FnMut(&ImageRef) -> Result<Vec<u8>, ApiError>,
    ) -> Result<Ingested, ApiError> {
        let settings = self.video.clone().ok_or_else(|| ApiError::bad("video input is not enabled on this server"))?;
        let r = video::the_video(messages, &settings.limits)?.ok_or_else(|| ApiError::bad("the request has no video part"))?;
        let at = r.at();
        let prepared = Arc::new(self.prepare_video(&r, fetch, work)?);
        let pruning = settings.config.pruning() && prepared.tubelets() > 1;
        if pruning && encoder.is_none() {
            return Err(ApiError::status(
                500,
                "server_error",
                format!(
                    "{at}: pruning is on and no video encoder is available: the rows kept are chosen from the encoder's features"
                ),
            ));
        }
        // the prompt: exactly one `<video>` (the part's, or the one typed in a text part), no image placeholder beside it
        let rendered = format.render_bounded(messages, kwargs, render_limit).map_err(|e| ApiError::bad(e.to_string()))?;
        let text = &rendered.text;
        if text.contains(SCRATCH) {
            return Err(ApiError::bad(format!(
                "the literal text {SCRATCH} is not allowed in messages that carry a video (the processor turns it into an image token)"
            )));
        }
        let tags = text.matches(VIDEO_TAG).count();
        if tags != 1 {
            return Err(ApiError::bad(format!(
                "the prompt has {tags} {VIDEO_TAG} placeholders but one video was sent (a literal {VIDEO_TAG} typed in a message takes the place \
                 of the video part; the processor expands one)"
            )));
        }
        if text.contains(IMAGE_TOKEN) {
            return Err(ApiError::bad(format!(
                "the prompt has a literal {IMAGE_TOKEN} beside the video: the placeholders of a video are the same token and are filled from one \
                 source (the reference model raises)"
            )));
        }
        let tubelets = prepared.tubelets();
        let tokens = prepared.plan.tokens_per_tubelet;
        // the encoder, with the host EVS function when pruning is on
        let mask_seen: Mutex<Option<Vec<bool>>> = Mutex::new(None);
        let encoded = match encoder {
            None => None,
            Some(e) => {
                let (rows, cols) = prepared.plan.tile.merged();
                let rate = settings.config.pruning_rate;
                let keep = |features: &[u16]| -> Result<Vec<bool>, String> {
                    let m = retention_mask_bits(features, tubelets, rows, cols, FEATURE_WIDTH, rate, SumOrder::TorchCpu)
                        .map_err(|e| e.to_string())?;
                    *mask_seen.lock().unwrap() = Some(m.clone());
                    Ok(m)
                };
                let keep_ref: Option<KeepRows<'_>> = if pruning { Some(&keep) } else { None };
                let enc = e.encode_video(&prepared, keep_ref).map_err(|err| {
                    let fake = ImageRef { message: r.message, part: r.part, url: String::new() };
                    encode_error(&fake, err)
                })?;
                if let Some(w) = work {
                    w.check()?;
                }
                Some(enc)
            }
        };
        // how many rows each tubelet keeps
        let counts: Vec<usize> = match &encoded {
            Some(enc) => {
                if enc.tubelets.len() != tubelets {
                    return Err(ApiError::status(
                        500,
                        "server_error",
                        format!("{at}: the video encoder returned {} tubelets for {tubelets}", enc.tubelets.len()),
                    ));
                }
                let got: Vec<usize> = enc.tubelets.iter().map(|t| t.tokens).collect();
                if pruning {
                    let mask = mask_seen.lock().unwrap().clone().ok_or_else(|| {
                        ApiError::status(500, "server_error", format!("{at}: the video encoder did not ask which rows to keep"))
                    })?;
                    let want = kept_per_tubelet(&mask, tubelets, tokens);
                    if got != want {
                        return Err(ApiError::status(
                            500,
                            "server_error",
                            format!("{at}: the video encoder kept {got:?} rows, the mask keeps {want:?}"),
                        ));
                    }
                } else if got.iter().any(|&t| t != tokens) {
                    return Err(ApiError::status(
                        500,
                        "server_error",
                        format!("{at}: the video encoder returned {got:?} rows for tubelets of {tokens}"),
                    ));
                }
                got
            }
            None => vec![tokens; tubelets],
        };
        // the text: one `<image>` per tubelet after its label; the id expansion makes each the `<img>` + k x `<image>` + `</img>` chunk
        let single = text.replacen(VIDEO_TAG, &video::one_placeholder_per_tubelet(&prepared.labels), 1);
        let ids = tok.encode_bounded(&single, render_limit)?;
        let (ids, spans) = format.expand_images(tok, &ids, &counts)?;
        let merged_full = prepared.plan.tile.merged();
        let mut images = Vec::with_capacity(tubelets);
        for (g, (sp, &kept)) in spans.iter().zip(&counts).enumerate() {
            if kept == 0 {
                continue; // `<img></img>` stays as text: the engine takes no empty span
            }
            let features: Arc<dyn Any + Send + Sync> = match &encoded {
                Some(enc) => enc.tubelets[g].features.clone(),
                None => Arc::new(DeferredTubelet { video: prepared.clone(), index: g }),
            };
            images.push(gb_engine::ImageSpan {
                start: sp.start,
                tokens: kept,
                merged: if kept == tokens { merged_full } else { (1, kept) },
                features,
                key: self.tubelet_key(&prepared, g, kept),
            });
        }
        Ok(Ingested {
            ids,
            starts_in_reasoning: rendered.starts_in_reasoning,
            images,
            prepared: Vec::new(),
            video: Some(prepared),
        })
    }
}

/// The model side's video encoder (`gb_model::serve::VisionService`) behind the seam: the prepared frames in, a feature handle per tubelet
/// out, under the service's lock and the request's cancellation. `encode_ms` is what the last encode took.
pub struct ServiceVideoEncoder<'a> {
    pub service: &'a gb_model::serve::VisionService,
    pub work: Option<&'a RequestWork>,
    pub encode_us: std::sync::atomic::AtomicU64,
}

impl<'a> ServiceVideoEncoder<'a> {
    pub fn new(service: &'a gb_model::serve::VisionService, work: Option<&'a RequestWork>) -> Self {
        Self { service, work, encode_us: std::sync::atomic::AtomicU64::new(0) }
    }
}

impl NemotronVideoEncoder for ServiceVideoEncoder<'_> {
    fn encode_video(&self, video: &PreparedVideo, keep: Option<KeepRows<'_>>) -> Result<EncodedVideo, ModelError> {
        let input = gb_model::serve::VideoInput { frames: &video.pre.frames, canvas: video.pre.canvas() };
        let t0 = std::time::Instant::now();
        let encoded = self.service.encode_video_checked(&input, keep, || {
            self.work.map_or(Ok(()), |w| w.check().map_err(|_| ModelError::Request("video work cancelled".into())))
        })?;
        self.encode_us.store(t0.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(EncodedVideo {
            tubelets: encoded.into_iter().map(|e| EncodedTubelet { tokens: e.tokens, features: e.features }).collect(),
        })
    }
}

/// What the HTTP layer asks of a format that takes video (`ChatFormat::video`): it never names a family. A request's video parts are found,
/// sized for the preparation gate before any frame is copied, and ingested (frames fetched, prepared, encoded, the prompt expanded) in one
/// call that returns the ids and one span per kept tubelet.
pub trait VideoIngress: Send + Sync {
    /// Whether `messages` has a part the template renders as a video.
    fn has_video(&self, messages: &Value) -> bool;
    /// Bytes the request's frame URLs retain while it is prepared (the gate's `retained`), or `None` without a video part. Refuses a frame
    /// count or a URL over its limit by name, before anything is copied.
    fn reserve(&self, messages: &Value, limits: &Limits) -> Result<Option<usize>, ApiError>;
    /// Bytes of decoder, resampler and retained bf16 frames one video of at most the video limits needs (the gate's `bytes`).
    fn work_bytes(&self, limits: &Limits) -> Result<usize, ApiError>;
    /// The whole ingress of a request with a video part: see [`NemotronImagePrep::ingest_video`].
    #[allow(clippy::too_many_arguments)]
    fn ingest(
        &self,
        tok: &Tok,
        messages: &Value,
        kwargs: &Map<String, Value>,
        limits: &Limits,
        render_limit: usize,
        encoder: &dyn NemotronVideoEncoder,
        work: Option<&RequestWork>,
        fetch: &mut dyn FnMut(&ImageRef) -> Result<Vec<u8>, ApiError>,
    ) -> Result<Ingested, ApiError>;
    /// Whether this format's video is frames through the family's IMAGE encoder ([`VideoIngress::ingest_frames`]) rather than tubelets
    /// through a video encoder ([`VideoIngress::ingest`]). `false` for Nemotron.
    fn takes_frames(&self) -> bool {
        false
    }
    /// [`VideoIngress::ingest`] for a format whose frames go through the vision service's image encoder one at a time (MiniCPM-V): the
    /// service replaces the tubelet encoder. The default refuses by name, so a format that says `takes_frames` must override it.
    #[allow(clippy::too_many_arguments)]
    fn ingest_frames(
        &self,
        _tok: &Tok,
        _messages: &Value,
        _kwargs: &Map<String, Value>,
        _limits: &Limits,
        _render_limit: usize,
        _service: &gb_model::serve::VisionService,
        _work: Option<&RequestWork>,
        _fetch: &mut dyn FnMut(&ImageRef) -> Result<Vec<u8>, ApiError>,
    ) -> Result<Ingested, ApiError> {
        Err(ApiError::bad("this chat format does not take video as frames"))
    }
}

/// The bytes of a `data:` URI image under the byte limit (`vision::prepare`'s first steps).
pub fn data_uri(r: &ImageRef, limits: &Limits) -> Result<Vec<u8>, ApiError> {
    if r.url.len() > limits.max_bytes / 3 * 4 + 64 {
        return Err(ApiError::status(
            413,
            "invalid_request_error",
            format!("messages[{}].content[{}]: image is larger than {} bytes", r.message, r.part, limits.max_bytes),
        ));
    }
    gb_vision::decode::data_uri_bytes(&r.url)
        .map_err(|e| ApiError::bad(format!("messages[{}].content[{}]: {e}", r.message, r.part)))
}

/// A `<image>` placeholder of a rendered prompt, for tests and tools: how many a text holds.
pub fn placeholders(text: &str) -> usize {
    text.matches(IMAGE_TOKEN).count()
}
