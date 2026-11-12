//! The MiniCPM-V-4.7 chat format: `<|im_start|>` framing, a `<think>` generation prompt that is open by default (`enable_thinking`
//! false closes it), Qwen-style `<tool_call><function=NAME><parameter=KEY>` calls, and images and video (E5b of
//! `docs/MINICPMV4_7_INTEGRATION_PLAN.md`): the template shipped in the pack (`chat_template.jinja`,
//! `testdata/minicpmv4_7/chat_template.jinja`, read 2026-10-12: the Qwen3.5/3.6 template with the media macro rendering one
//! `<|image_pad|>` per image part and one `<|video_pad|>` per video part, parts of a message joined by `\n`).
//!
//! # The image layout (`processing_minicpmv4_7.py`, `replace_image_token` and `_prepend_local_ids`)
//!
//! The template's one placeholder per image becomes, once the image is encoded and its crop plan is known,
//!
//! ```text
//! <image_id> i </image_id>  <image> pad x t </image>  { <slice> pad x s </slice> x cols  joined by "\n" between rows }
//! ```
//!
//! `i` is the image's index among the images of the request (digits, as the tokenizer splits them; videos take none), `t` the
//! thumbnail's tokens, `s` one slice's, and the slice rows are `rows x cols` of the preprocessing's grid (none: the thumbnail alone).
//! A video frame is `<image> video_pad x t </image>` plus its slices (a frame is its thumbnail by default), the frames of a clip
//! adjacent with nothing between them, and no image id.
//!
//! **The span.** One engine `ImageSpan` per frame: `start` is the frame's `<image>` (the `<image_id>` prefix is plain text before it),
//! `tokens` runs through its last marker and the `\n` tokens that follow it, which the reference model's position rule absorbs into
//! the frame (so a prefix cut cannot fall between a frame and its newline), and `merged == (1, tokens)`. The pad tokens inside it are
//! where the feature rows go (`families::minicpmv4_7::splice`).
//!
//! # Video
//!
//! A video is a list of FRAMES (no container is decoded here: nothing is linked): the payload under `video_url`, `video` or
//! `input_video` is an array of image URLs (strings, or objects with `url` or `image_url`), an object holding that array under `url` or
//! `frames`, or one `data:video/jpeg;base64,<f1>,<f2>,...` string. The part is rewritten to `{"type": "video"}` before the template
//! renders it, because the pack's template only knows `video` as a type (a `video_url` part would raise). Each frame goes through the
//! same decoder, preprocessing and tower as an image (`MiniCpmPreprocessing` at `REXL3_MINICPM_VIDEO_MAX_SLICE_NUMS` slices, default
//! 1). The processor's `stack_frames` tiling (several sub-frames of a second composed into one image) is not built: a frame is one
//! picture. Images and videos may share a request (the reference scatters each modality from its own tensor).

use crate::api::ApiError;
use crate::chat_format::{render_normalized, ChatFormat, EncodedRef, FormatEntry, Placed, RenderedChat, VideoOptions};
use crate::formats::nemotron_omni::prep::{encode_error, Ingested, VideoIngress};
use crate::preparation::RequestWork;
use crate::template::{ChatTemplate, TemplateError};
use crate::tokenizer::Tok;
use crate::tools::{ToolSyntax, FUNCTION_BLOCK_TOOL_SYNTAX};
use crate::vision::{self, ImageRef, Limits, Span};
use gb_model::families::minicpmv4_7::splice::{layout_of, FrameLayout};
use gb_model::families::minicpmv4_7::vision::{image_preprocessing, video_preprocessing};
use gb_vision::minicpmv::MiniCpmPreprocessing;
use gb_vision::preprocessor::ImagePreprocessor;
use serde_json::{Map, Value};
use std::path::Path;
use std::sync::Arc;

pub const ENTRY: FormatEntry =
    FormatEntry { id: "minicpmv", names: &["MiniCPMV4_7ForConditionalGeneration", "minicpmv4_7"], open };

/// `<|image_pad|>`: what the template renders for an image and the feature rows replace.
pub const IMAGE_PAD: u32 = 248_056;
/// `<|video_pad|>`: the same for a video frame.
pub const VIDEO_PAD: u32 = 248_057;
const NEWLINE: u32 = 198;

/// The most frames of one video (`max_num_frames` of the reference's video processor).
pub const MAX_VIDEO_FRAMES: usize = 128;

fn open(dir: &Path) -> Result<Box<dyn ChatFormat>, String> {
    let src = std::fs::read_to_string(dir.join("chat_template.jinja")).map_err(|e| format!("chat_template.jinja: {e}"))?;
    let mut f = MiniCpmFormat::new(&src).map_err(|e| e.to_string())?;
    // the preprocessing the server and the encoder both use (images), and the frames' (video): known to the format so that a video
    // request can be prepared and encoded in one place
    // a pack folder without `preprocessor_config.json` serves text: the error of a bad option still surfaces (the family's own
    // `image_preprocessor` and the encoder read the same function when images are on)
    if dir.join("preprocessor_config.json").exists() {
        let image = image_preprocessing(dir).map_err(|e| e.to_string())?;
        let video = video_preprocessing(&image).map_err(|e| e.to_string())?;
        f.preprocessing = Some(Preprocessing { image: Arc::new(image), video: Arc::new(video) });
    }
    Ok(Box::new(f))
}

struct Preprocessing {
    image: Arc<MiniCpmPreprocessing>,
    video: Arc<MiniCpmPreprocessing>,
}

/// What the server admits of one video, from the request and the options (`--max-video-frames`, `--max-video-tokens`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct VideoLimits {
    max_frames: usize,
    max_tokens: usize,
}

pub struct MiniCpmFormat {
    template: ChatTemplate,
    preprocessing: Option<Preprocessing>,
    video: Option<VideoLimits>,
}

impl MiniCpmFormat {
    pub fn new(template_source: &str) -> Result<Self, TemplateError> {
        Ok(Self { template: ChatTemplate::new(template_source)?, preprocessing: None, video: None })
    }

    /// The same format with the preprocessing a pack folder would give it (tests and tools).
    pub fn with_preprocessing(mut self, image: MiniCpmPreprocessing, video: MiniCpmPreprocessing) -> Self {
        self.preprocessing = Some(Preprocessing { image: Arc::new(image), video: Arc::new(video) });
        self
    }
}

// ---------------------------------------------------------------------------------------------------------------- layout

/// The ids of the marker tokens of the layout, looked up by name in the tokenizer.
struct Markers {
    image_id_open: u32,
    image_id_close: u32,
    image_open: u32,
    image_close: u32,
    slice_open: u32,
    slice_close: u32,
}

impl Markers {
    fn new(tok: &Tok) -> Result<Markers, ApiError> {
        let id = |t: &str| tok.token_to_id(t).ok_or_else(|| ApiError::bad(format!("the tokenizer has no {t} token")));
        Ok(Markers {
            image_id_open: id("<image_id>")?,
            image_id_close: id("</image_id>")?,
            image_open: id("<image>")?,
            image_close: id("</image>")?,
            slice_open: id("<slice>")?,
            slice_close: id("</slice>")?,
        })
    }
}

/// One frame's tokens appended to `out`: `<image>` pad x thumbnail `</image>`, then the slices (rows joined by a newline).
fn push_frame(out: &mut Vec<u32>, pad: u32, layout: &FrameLayout, m: &Markers, merge: usize) -> Result<(), ApiError> {
    let tokens_of = |c: &[u32; 2]| (c[0] as usize / merge) * (c[1] as usize / merge);
    let thumb = layout.crops.first().ok_or_else(|| ApiError::bad("an image with no crop"))?;
    out.push(m.image_open);
    out.extend(std::iter::repeat_n(pad, tokens_of(thumb)));
    out.push(m.image_close);
    let (rows, cols) = layout.slice_grid;
    if rows > 0 && cols > 0 {
        if layout.crops.len() != 1 + rows * cols {
            return Err(ApiError::bad(format!(
                "an image of {rows} x {cols} slices has {} crops (want {})",
                layout.crops.len(),
                1 + rows * cols
            )));
        }
        let per = tokens_of(&layout.crops[1]);
        for r in 0..rows {
            if r > 0 {
                out.push(NEWLINE);
            }
            for _ in 0..cols {
                out.push(m.slice_open);
                out.extend(std::iter::repeat_n(pad, per));
                out.push(m.slice_close);
            }
        }
    }
    Ok(())
}

fn merge_of(layout: &FrameLayout) -> usize {
    layout.downsample.merge_factor()
}

/// The span of a frame that starts at `start` in `out`: its tokens through its markers, plus the newlines that follow (absorbed by the
/// reference's position rule). Called after the whole prompt is expanded, so the newline that follows is in `out`.
fn span_end(out: &[u32], mut end: usize) -> usize {
    while out.get(end) == Some(&NEWLINE) {
        end += 1;
    }
    end
}

// ---------------------------------------------------------------------------------------------------------------- parts

/// Whether a content part is one the template renders as a video: a video type or a payload key the video types carry.
fn is_video_part(part: &Value) -> bool {
    let ty = part.get("type").and_then(Value::as_str).unwrap_or("");
    ["video", "video_url", "input_video"].contains(&ty)
        || ["video", "video_url", "input_video"].iter().any(|k| part.get(*k).is_some())
}

fn is_image(part: &Value) -> bool {
    if is_video_part(part) {
        return false;
    }
    let ty = part.get("type").and_then(Value::as_str).unwrap_or("");
    ["image_url", "image", "input_image"].contains(&ty) || part.get("image_url").is_some() || part.get("image").is_some()
}

fn frame_url(at: &str, i: usize, f: &Value) -> Result<String, ApiError> {
    let url = match f {
        Value::String(s) => Some(s.as_str()),
        Value::Object(o) => match o.get("url").or_else(|| o.get("image_url")) {
            Some(Value::String(s)) => Some(s.as_str()),
            Some(Value::Object(inner)) => inner.get("url").and_then(Value::as_str),
            _ => None,
        },
        _ => None,
    };
    url.map(str::to_string).ok_or_else(|| {
        ApiError::bad(format!("{at}: frame {} is not a URL (a string, or an object with `url` or `image_url`)", i + 1))
    })
}

/// The frames of a video part: URLs, in order. A container is refused by name.
fn video_frames(message: usize, part_index: usize, part: &Value, max_frames: usize) -> Result<Vec<String>, ApiError> {
    let at = format!("messages[{message}].content[{part_index}]");
    let payload = ["video_url", "video", "input_video"].iter().find_map(|k| part.get(*k)).ok_or_else(|| {
        ApiError::bad(format!("{at}: a video part needs its frames under `video_url` (an array of image data URLs)"))
    })?;
    let frames_value = match payload {
        Value::Object(o) => o
            .get("url")
            .or_else(|| o.get("frames"))
            .ok_or_else(|| ApiError::bad(format!("{at}: the video has no `url` (an array of frames)")))?,
        other => other,
    };
    let frames: Vec<String> = match frames_value {
        Value::Array(a) => a.iter().enumerate().map(|(i, f)| frame_url(&at, i, f)).collect::<Result<_, _>>()?,
        Value::String(s) => match s.strip_prefix("data:video/") {
            Some(rest) => {
                let (head, data) = rest.split_once(',').unwrap_or((rest, ""));
                let mut kinds = head.split(';');
                let sub = kinds.next().unwrap_or("").to_ascii_lowercase();
                let b64 = kinds.any(|k| k.eq_ignore_ascii_case("base64"));
                if matches!(sub.as_str(), "jpeg" | "jpg" | "png") && b64 {
                    let image = if sub == "png" { "png" } else { "jpeg" };
                    data.split(',').map(|f| format!("data:image/{image};base64,{f}")).collect()
                } else {
                    return Err(ApiError::bad(format!(
                        "{at}: a `data:video/{sub}` container is not decoded by this server (no codec is linked); send the frames: an \
                         array of image data URLs, or `data:video/jpeg;base64,<frame>,<frame>,...`"
                    )));
                }
            }
            None => {
                return Err(ApiError::bad(format!(
                "{at}: a video is a list of frames, not a single URL or file (no container is decoded by this server); send an \
                     array of image data URLs"
            )))
            }
        },
        _ => return Err(ApiError::bad(format!("{at}: the frames of a video are an array of image data URLs"))),
    };
    if frames.is_empty() {
        return Err(ApiError::bad(format!("{at}: a video needs at least one frame")));
    }
    if frames.len() > max_frames {
        return Err(ApiError::bad(format!(
            "{at}: the video has {} frames; the limit is {max_frames} (--max-video-frames, at most {MAX_VIDEO_FRAMES})",
            frames.len()
        )));
    }
    if let Some(i) = frames.iter().position(|f| f.is_empty()) {
        return Err(ApiError::bad(format!("{at}: frame {} is empty", i + 1)));
    }
    Ok(frames)
}

/// The messages with every video part replaced by `{"type": "video"}` (the one spelling the pack's template renders as a video).
fn render_view(messages: &Value) -> Value {
    let mut m = messages.clone();
    for msg in m.as_array_mut().into_iter().flatten() {
        for part in msg.get_mut("content").and_then(Value::as_array_mut).into_iter().flatten() {
            if is_video_part(part) {
                *part = serde_json::json!({"type": "video"});
            }
        }
    }
    m
}

fn fnv_str(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

// ---------------------------------------------------------------------------------------------------------------- format

impl ChatFormat for MiniCpmFormat {
    fn id(&self) -> &'static str {
        ENTRY.id
    }

    fn render(&self, messages: &Value, kwargs: &Map<String, Value>) -> Result<RenderedChat, TemplateError> {
        let text = render_normalized(&self.template, messages, kwargs)?;
        let starts_in_reasoning = self.starts_in_reasoning(&text);
        Ok(RenderedChat { text, starts_in_reasoning })
    }

    fn render_bounded(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        max_bytes: usize,
    ) -> Result<RenderedChat, TemplateError> {
        crate::template::with_render_limit(max_bytes, || self.render(messages, kwargs))
    }

    /// A thinking prompt ends with `<think>` and not the closed empty block; the same rule as Qwen's.
    fn starts_in_reasoning(&self, prompt: &str) -> bool {
        prompt.trim_end_matches('\n').ends_with("<think>") && !prompt.ends_with("</think>\n\n")
    }

    fn tool_syntax(&self) -> ToolSyntax {
        FUNCTION_BLOCK_TOOL_SYNTAX
    }

    /// Image parts when the server has the tower, video parts when video is on, nothing else: audio and the rest are refused by name.
    fn check_media(&self, messages: &Value, allow_images: bool) -> Result<(), ApiError> {
        let Some(list) = messages.as_array() else { return Ok(()) };
        for (i, m) in list.iter().enumerate() {
            let Some(parts) = m.get("content").and_then(Value::as_array) else { continue };
            for (j, part) in parts.iter().enumerate() {
                let ty = part.get("type").and_then(Value::as_str).unwrap_or("");
                let at = format!("messages[{i}].content[{j}]");
                if is_video_part(part) {
                    if !allow_images {
                        return Err(ApiError::bad(format!(
                            "{at}: `{}` input is not supported by this server (it was started without image support); send text only",
                            if ty.is_empty() { "video" } else { ty }
                        )));
                    }
                    if self.video.is_none() {
                        return Err(ApiError::bad(format!(
                            "{at}: video input is not enabled on this server (--no-video, or the family encodes no video); send images or text"
                        )));
                    }
                } else if is_image(part) {
                    if !allow_images {
                        return Err(ApiError::bad(format!(
                            "{at}: `{}` input is not supported by this server (it was started without image support); send text only",
                            if ty.is_empty() { "image" } else { ty }
                        )));
                    }
                } else if ty != "text" && !ty.is_empty() && part.get("text").is_none() {
                    return Err(ApiError::bad(format!(
                        "{at}: `{ty}` input is not supported by this server; send text, images{}",
                        if self.video.is_some() { " and video frames" } else { "" }
                    )));
                }
            }
        }
        Ok(())
    }

    fn is_image_part(&self, part: &Value) -> bool {
        is_image(part)
    }

    fn image_placeholder(&self, tok: &Tok) -> Result<(u32, &'static str), String> {
        let id = tok.token_to_id("<|image_pad|>").ok_or("tokenizer has no image placeholder")?;
        Ok((id, "<|image_pad|>"))
    }

    /// Placeholder tokens of one image sent alone, from its size and the preprocessing's crop plan.
    fn image_tokens(&self, height: usize, width: usize) -> Option<Result<usize, String>> {
        let p = self.preprocessing.as_ref()?;
        Some(ImagePreprocessor::plan(&*p.image, width, height).map(|plan| plan.tokens).map_err(|e| e.to_string()))
    }

    fn serves_images(&self) -> bool {
        true
    }

    fn video(&self) -> Option<&dyn VideoIngress> {
        self.video.as_ref().map(|_| self as &dyn VideoIngress)
    }

    fn enable_video(&mut self, o: &VideoOptions) -> Result<(), String> {
        if o.pruning_rate != 0.0 {
            return Err(
                "--video-pruning-rate: MiniCPM-V-4.7 keeps every token of a frame (there is no pruning); leave it at 0".into()
            );
        }
        if o.max_frames == 0 || o.max_frames > MAX_VIDEO_FRAMES {
            return Err(format!("--max-video-frames {}: 1 to {MAX_VIDEO_FRAMES} for MiniCPM-V-4.7", o.max_frames));
        }
        if o.max_tokens == 0 {
            return Err("--max-video-tokens must be positive".into());
        }
        self.video = Some(VideoLimits { max_frames: o.max_frames, max_tokens: o.max_tokens });
        Ok(())
    }

    /// The one placeholder of each image becomes its frame: `<image_id>i</image_id>` and the layout of its crops
    /// ([module documentation](self)). The span the engine gets covers the frame from its `<image>` marker, its markers and the
    /// newlines that follow it; `merged` is `(1, tokens)`.
    fn place_images(&self, tok: &Tok, ids: &[u32], encoded: &[EncodedRef<'_>]) -> Result<(Vec<u32>, Vec<Placed>), ApiError> {
        let (pad, literal) = self.image_placeholder(tok).map_err(ApiError::bad)?;
        let pads = ids.iter().filter(|&&t| t == pad).count();
        if pads != encoded.len() {
            return Err(ApiError::bad(format!(
                "the prompt has {pads} image placeholders but {} images were sent (the literal text {literal} is not allowed in messages)",
                encoded.len()
            )));
        }
        let m = Markers::new(tok)?;
        let mut out = Vec::with_capacity(ids.len() + encoded.iter().map(|e| e.tokens + 64).sum::<usize>());
        let mut frames: Vec<(usize, usize)> = Vec::with_capacity(encoded.len()); // (start, end of the last marker)
        let mut k = 0;
        for &t in ids {
            if t != pad {
                out.push(t);
                continue;
            }
            let layout = layout_of(encoded[k].features)
                .ok_or_else(|| ApiError::bad("the image encoder returned features this chat format does not know"))?;
            if layout.rows != encoded[k].tokens {
                return Err(ApiError::bad(format!(
                    "image {k}: the encoder returned {} placeholder rows but its crops make {}",
                    encoded[k].tokens, layout.rows
                )));
            }
            out.push(m.image_id_open);
            out.extend(tok.encode(&k.to_string())?);
            out.push(m.image_id_close);
            let start = out.len();
            push_frame(&mut out, IMAGE_PAD, &layout, &m, merge_of(&layout))?;
            frames.push((start, out.len()));
            k += 1;
        }
        // a span runs from its frame's `<image>` through its last marker and the newlines that follow it
        let placed = frames
            .iter()
            .map(|&(s, e)| {
                let end = span_end(&out, e);
                Placed { start: s, tokens: end - s, merged: (1, end - s) }
            })
            .collect();
        Ok((out, placed))
    }

    fn expand_images(&self, tok: &Tok, ids: &[u32], counts: &[usize]) -> Result<(Vec<u32>, Vec<Span>), ApiError> {
        // the count-only expansion has no layout: a caller that has the encoder's output goes through `place_images`
        let (token, literal) = self.image_placeholder(tok).map_err(ApiError::bad)?;
        vision::expand_placeholders(ids, token, literal, counts)
    }
}

impl VideoIngress for MiniCpmFormat {
    fn has_video(&self, messages: &Value) -> bool {
        self.video.is_some()
            && messages
                .as_array()
                .into_iter()
                .flatten()
                .any(|m| m.get("content").and_then(Value::as_array).is_some_and(|ps| ps.iter().any(is_video_part)))
    }

    /// The bytes the request's frame and image URLs retain while it is prepared: their lengths, one fetched image in flight and a
    /// little per frame. Refuses a frame count or a URL over its limit by name before anything is copied.
    fn reserve(&self, messages: &Value, limits: &Limits) -> Result<Option<usize>, ApiError> {
        let v = self.video.ok_or_else(|| ApiError::bad("video input is not enabled on this server"))?;
        let mut urls = 0usize;
        let mut frames = 0usize;
        let mut images = 0usize;
        let data_max = limits.max_bytes.div_ceil(3).saturating_mul(4).saturating_add(64);
        for (i, m) in messages.as_array().into_iter().flatten().enumerate() {
            for (j, part) in m.get("content").and_then(Value::as_array).into_iter().flatten().enumerate() {
                let at = format!("messages[{i}].content[{j}]");
                let check = |url: &str, what: &str| -> Result<(), ApiError> {
                    let max = if url.starts_with("data:video/") {
                        data_max.saturating_mul(v.max_frames)
                    } else if url.starts_with("data:") {
                        data_max
                    } else {
                        2048
                    };
                    if url.len() > max {
                        return Err(ApiError::status(
                            413,
                            "invalid_request_error",
                            format!("{at}: {what}: image URL exceeds size limit"),
                        ));
                    }
                    Ok(())
                };
                if is_video_part(part) {
                    let list = video_frames_unchecked(part);
                    if list.len() > v.max_frames {
                        return Err(ApiError::bad(format!(
                            "{at}: the video has {} frames; the limit is {} (--max-video-frames)",
                            list.len(),
                            v.max_frames
                        )));
                    }
                    for (k, u) in list.iter().enumerate() {
                        check(u, &format!("frame {}", k + 1))?;
                        urls = urls.saturating_add(u.len());
                    }
                    frames = frames.saturating_add(list.len().max(1));
                } else if is_image(part) {
                    images += 1;
                    if images > limits.max_images {
                        return Err(ApiError::bad(format!("at most {} images per request", limits.max_images)));
                    }
                    if let Some(u) = image_url_str(part) {
                        check(u, "image")?;
                        urls = urls.saturating_add(u.len());
                    }
                }
            }
        }
        urls.checked_add(limits.max_bytes)
            .and_then(|n| n.checked_add(frames.saturating_add(images).saturating_mul(64)))
            .map(Some)
            .ok_or_else(|| ApiError::bad("video retained size overflow"))
    }

    fn work_bytes(&self, limits: &Limits) -> Result<usize, ApiError> {
        crate::preparation::image_work_bytes(limits.max_tokens_per_image)
    }

    fn ingest(
        &self,
        _tok: &Tok,
        _messages: &Value,
        _kwargs: &Map<String, Value>,
        _limits: &Limits,
        _render_limit: usize,
        _encoder: &dyn crate::formats::nemotron_omni::prep::NemotronVideoEncoder,
        _work: Option<&RequestWork>,
        _fetch: &mut dyn FnMut(&ImageRef) -> Result<Vec<u8>, ApiError>,
    ) -> Result<Ingested, ApiError> {
        Err(ApiError::bad("MiniCPM-V-4.7 video goes through the image encoder as frames (ingest_frames)"))
    }

    fn takes_frames(&self) -> bool {
        true
    }

    fn ingest_frames(
        &self,
        tok: &Tok,
        messages: &Value,
        kwargs: &Map<String, Value>,
        limits: &Limits,
        render_limit: usize,
        service: &gb_model::serve::VisionService,
        work: Option<&RequestWork>,
        fetch: &mut dyn FnMut(&ImageRef) -> Result<Vec<u8>, ApiError>,
    ) -> Result<Ingested, ApiError> {
        let v = self.video.ok_or_else(|| ApiError::bad("video input is not enabled on this server"))?;
        let pp = self.preprocessing.as_ref().ok_or_else(|| ApiError::bad("the MiniCPM-V-4.7 preprocessing is not set up"))?;
        let check = || work.map_or(Ok(()), |w| w.check());
        check()?;
        // what the request holds, in template order: image parts and the frames of video parts
        enum Item {
            Image(ImageRef),
            Video(Vec<ImageRef>),
        }
        let mut items: Vec<Item> = Vec::new();
        let (mut images, mut frames_total) = (0usize, 0usize);
        for (i, m) in messages.as_array().into_iter().flatten().enumerate() {
            for (j, part) in m.get("content").and_then(Value::as_array).into_iter().flatten().enumerate() {
                if is_video_part(part) {
                    let urls = video_frames(i, j, part, v.max_frames)?;
                    frames_total += urls.len();
                    if frames_total > v.max_frames {
                        return Err(ApiError::bad(format!(
                            "messages[{i}].content[{j}]: {frames_total} video frames in the request; the limit is {} (--max-video-frames)",
                            v.max_frames
                        )));
                    }
                    items.push(Item::Video(urls.into_iter().map(|url| ImageRef { message: i, part: j, url }).collect()));
                } else if is_image(part) {
                    images += 1;
                    if images > limits.max_images {
                        return Err(ApiError::bad(format!("at most {} images per request", limits.max_images)));
                    }
                    let url = image_url_str(part)
                        .ok_or_else(|| {
                            ApiError::bad(format!("messages[{i}].content[{j}]: an image part needs a `url` (a data: URI)"))
                        })?
                        .to_string();
                    items.push(Item::Image(ImageRef { message: i, part: j, url }));
                }
            }
        }
        // the prompt: the template over the messages with each video part spelled the way it renders as `<|video_pad|>`
        let view = render_view(messages);
        let rendered = self.render_bounded(&view, kwargs, render_limit).map_err(|e| match e {
            TemplateError::Limit => crate::budget::too_large("rendered prompt byte limit exceeded"),
            TemplateError::Raised(m) => ApiError::bad(m),
            TemplateError::Render(m) => ApiError::bad(format!("chat template failed: {m}")),
        })?;
        let ids = tok.encode_bounded(&rendered.text, render_limit)?;
        // prepare and encode every picture, in template order
        struct Done {
            layout: FrameLayout,
            features: Arc<dyn std::any::Any + Send + Sync>,
            key: u64,
        }
        let id_of = |p: &MiniCpmPreprocessing| fnv_str(&ImagePreprocessor::id(p));
        let (image_id, video_id) = (id_of(&pp.image), id_of(&pp.video));
        let mut total_bytes = 0usize;
        let mut video_tokens = 0usize;
        let mut pictures: Vec<Done> = Vec::new(); // images
        let mut clips: Vec<Vec<Done>> = Vec::new(); // videos
        for item in &items {
            let (refs, is_video): (Vec<&ImageRef>, bool) = match item {
                Item::Image(r) => (vec![r], false),
                Item::Video(rs) => (rs.iter().collect(), true),
            };
            let mut group = Vec::with_capacity(refs.len());
            for r in refs {
                check()?;
                let bytes = fetch(r)?;
                total_bytes =
                    total_bytes.checked_add(bytes.len()).ok_or_else(|| ApiError::bad("image aggregate size overflow"))?;
                if total_bytes > limits.max_total_bytes {
                    return Err(ApiError::status(
                        413,
                        "invalid_request_error",
                        format!(
                            "messages[{}].content[{}]: aggregate image bytes exceed {}",
                            r.message, r.part, limits.max_total_bytes
                        ),
                    ));
                }
                let (frame_pp, key_id) = if is_video { (&pp.video, video_id) } else { (&pp.image, image_id) };
                let pre = vision::prepare_bytes(r, &bytes, limits, &**frame_pp, work)?;
                if is_video {
                    let tokens = frame_pp.plan(pre.size.0, pre.size.1).map_or(0, |p| p.total_tokens);
                    video_tokens = video_tokens.saturating_add(tokens);
                    if video_tokens > v.max_tokens {
                        return Err(ApiError::bad(format!(
                            "messages[{}].content[{}]: the video's frames would occupy {video_tokens} tokens; the limit is {} \
                             (--max-video-tokens)",
                            r.message, r.part, v.max_tokens
                        )));
                    }
                }
                check()?;
                let enc = service
                    .encode_checked(&pre, || {
                        check().map_err(|_| gb_model::error::ModelError::Request("image work cancelled".into()))
                    })
                    .map_err(|e| check().err().unwrap_or_else(|| encode_error(r, e)))?;
                let layout = layout_of(&*enc.features)
                    .ok_or_else(|| ApiError::bad("the image encoder returned features this chat format does not know"))?;
                let key = gb_sched::image_content_hash(pre.pixel_hash, pre.size.0 as u32, pre.size.1 as u32, key_id);
                group.push(Done { layout, features: enc.features, key });
            }
            if is_video {
                clips.push(group);
            } else {
                pictures.extend(group);
            }
        }
        check()?;
        // expand: the i-th `<|image_pad|>` is the i-th image, the j-th `<|video_pad|>` the frames of the j-th video
        let m = Markers::new(tok)?;
        let n_image_pads = ids.iter().filter(|&&t| t == IMAGE_PAD).count();
        let n_video_pads = ids.iter().filter(|&&t| t == VIDEO_PAD).count();
        if n_image_pads != pictures.len() || n_video_pads != clips.len() {
            return Err(ApiError::bad(format!(
                "the prompt has {n_image_pads} image and {n_video_pads} video placeholders but {} images and {} videos were sent \
                 (the literal text <|image_pad|> or <|video_pad|> is not allowed in messages)",
                pictures.len(),
                clips.len()
            )));
        }
        let mut out = Vec::with_capacity(ids.len() + 64);
        // (first token, end of the last marker, features, key) of every frame in prompt order
        let mut frames: Vec<(usize, usize, Arc<dyn std::any::Any + Send + Sync>, u64)> = Vec::new();
        let (mut ni, mut nv) = (0usize, 0usize);
        for &t in &ids {
            match t {
                IMAGE_PAD => {
                    let d = &pictures[ni];
                    out.push(m.image_id_open);
                    out.extend(tok.encode(&ni.to_string())?);
                    out.push(m.image_id_close);
                    let s = out.len();
                    push_frame(&mut out, IMAGE_PAD, &d.layout, &m, merge_of(&d.layout))?;
                    frames.push((s, out.len(), d.features.clone(), d.key));
                    ni += 1;
                }
                VIDEO_PAD => {
                    for d in &clips[nv] {
                        let s = out.len();
                        push_frame(&mut out, VIDEO_PAD, &d.layout, &m, merge_of(&d.layout))?;
                        frames.push((s, out.len(), d.features.clone(), d.key));
                    }
                    nv += 1;
                }
                other => out.push(other),
            }
        }
        let images: Vec<gb_engine::ImageSpan> = frames
            .into_iter()
            .map(|(start, frame_end, features, key)| {
                let tokens = span_end(&out, frame_end) - start;
                gb_engine::ImageSpan { start, tokens, merged: (1, tokens), features, key: key.max(1) }
            })
            .collect();
        Ok(Ingested { ids: out, starts_in_reasoning: rendered.starts_in_reasoning, images, prepared: Vec::new(), video: None })
    }
}

fn image_url_str(part: &Value) -> Option<&str> {
    let v = part.get("image_url").or_else(|| part.get("image"))?;
    match v {
        Value::String(s) => Some(s.as_str()),
        Value::Object(o) => o.get("url").and_then(Value::as_str),
        _ => None,
    }
}

/// The URL strings of a video part's frames, without copying and without checking the shape (`video_frames` does that later, on the
/// copies): an array of strings or objects, an object holding it under `url` or `frames`, or one frame-sequence string.
fn video_frames_unchecked(part: &Value) -> Vec<&str> {
    let payload = ["video_url", "video", "input_video"].iter().find_map(|k| part.get(*k));
    let frames = match payload {
        Some(Value::Object(o)) => o.get("url").or_else(|| o.get("frames")),
        other => other,
    };
    fn one(f: &Value) -> Option<&str> {
        match f {
            Value::String(s) => Some(s.as_str()),
            Value::Object(o) => match o.get("url").or_else(|| o.get("image_url")) {
                Some(Value::String(s)) => Some(s.as_str()),
                Some(Value::Object(inner)) => inner.get("url").and_then(Value::as_str),
                _ => None,
            },
            _ => None,
        }
    }
    match frames {
        Some(Value::Array(a)) => a.iter().filter_map(one).collect(),
        Some(Value::String(s)) => vec![s.as_str()],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const TEMPLATE: &str = include_str!("../../../../testdata/minicpmv4_7/chat_template.jinja");

    fn format() -> MiniCpmFormat {
        MiniCpmFormat::new(TEMPLATE).unwrap()
    }

    fn render(messages: Value, kwargs: Value) -> RenderedChat {
        format().render(&messages, kwargs.as_object().unwrap()).unwrap()
    }

    #[test]
    fn the_registry_entry_names_the_architecture_and_the_model_type() {
        assert_eq!(ENTRY.id, "minicpmv");
        assert!(ENTRY.names.contains(&"MiniCPMV4_7ForConditionalGeneration") && ENTRY.names.contains(&"minicpmv4_7"));
        assert_eq!(format().id(), "minicpmv");
    }

    #[test]
    fn thinking_is_open_by_default_and_closed_when_asked() {
        let messages = json!([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]);
        let on = render(messages.clone(), json!({}));
        assert_eq!(
            on.text,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        assert!(on.starts_in_reasoning);
        let off = render(messages, json!({"enable_thinking": false}));
        assert_eq!(
            off.text,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert!(!off.starts_in_reasoning);
    }

    #[test]
    fn a_tool_call_in_the_history_is_written_in_the_xml_form_and_the_answer_is_a_tool_response_turn() {
        let messages = json!([
            {"role": "user", "content": "Weather?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
            {"role": "tool", "content": "sunny"}
        ]);
        let r = render(messages, json!({}));
        assert!(
            r.text.contains("<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n"),
            "{}",
            r.text
        );
        assert!(r.text.contains("<|im_start|>user\n<tool_response>\nsunny\n</tool_response><|im_end|>\n"), "{}", r.text);
        assert!(r.text.ends_with("<|im_start|>assistant\n<think>\n"), "{}", r.text);
    }

    #[test]
    fn images_are_served_and_audio_and_unlisted_parts_are_refused_by_name() {
        let f = format();
        assert!(f.serves_images());
        let part = json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}});
        assert!(f.is_image_part(&part));
        assert!(!f.is_image_part(&json!({"type": "text", "text": "x"})));
        let messages = json!([{"role": "user", "content": [part.clone(), {"type": "text", "text": "what is this?"}]}]);
        assert!(f.check_media(&messages, true).is_ok());
        // a server started with --no-vision refuses the image by name
        let e = f.check_media(&messages, false).unwrap_err().message;
        assert!(e.contains("messages[0].content[0]") && e.contains("without image support"), "{e}");
        let audio =
            json!([{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "AA", "format": "wav"}}]}]);
        let e = f.check_media(&audio, true).unwrap_err().message;
        assert!(e.contains("input_audio"), "{e}");
        // video is refused until it is enabled, then admitted
        let video =
            json!([{"role": "user", "content": [{"type": "video_url", "video_url": {"url": ["data:image/png;base64,AA=="]}}]}]);
        let e = f.check_media(&video, true).unwrap_err().message;
        assert!(e.contains("video input is not enabled"), "{e}");
        let mut f = format();
        f.enable_video(&VideoOptions { max_frames: 64, max_tokens: 8192, pruning_rate: 0.0, require_square_grid: false })
            .unwrap();
        assert!(f.check_media(&video, true).is_ok());
        assert!(f.has_video(&video) && !f.has_video(&messages));
        let e = f.enable_video(&VideoOptions { pruning_rate: 0.5, ..VideoOptions::default() }).unwrap_err();
        assert!(e.contains("no pruning"), "{e}");
        let e = f.enable_video(&VideoOptions { max_frames: 129, ..VideoOptions::default() }).unwrap_err();
        assert!(e.contains("1 to 128"), "{e}");
    }

    #[test]
    fn a_video_part_renders_as_the_video_pad_in_every_spelling_and_an_image_as_the_image_pad() {
        let f = format();
        for part in [
            json!({"type": "video", "video": ["data:image/png;base64,AA=="]}),
            json!({"type": "video_url", "video_url": {"url": ["data:image/png;base64,AA=="]}}),
            json!({"type": "input_video", "input_video": ["data:image/png;base64,AA=="]}),
        ] {
            let messages = json!([{"role": "user", "content": [part, {"type": "text", "text": "what happens?"}]}]);
            let view = render_view(&messages);
            let r = f.render(&view, &Map::new()).unwrap();
            assert!(r.text.contains("<|im_start|>user\n<|video_pad|>\nwhat happens?<|im_end|>"), "{}", r.text);
        }
        let messages = json!([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "text", "text": "compare"}]}]);
        let r = f.render(&render_view(&messages), &Map::new()).unwrap();
        assert!(r.text.contains("<|im_start|>user\n<|image_pad|>\n<|image_pad|>\ncompare<|im_end|>"), "{}", r.text);
    }

    #[test]
    fn video_frames_are_read_from_each_payload_shape_and_containers_are_refused_by_name() {
        let frames = vec!["data:image/png;base64,AA==", "data:image/png;base64,BB=="];
        let shapes = [
            json!({"type": "video", "video": frames}),
            json!({"type": "video_url", "video_url": {"url": frames}}),
            json!({"type": "video_url", "video_url": {"frames": [{"url": frames[0]}, {"image_url": {"url": frames[1]}}]}}),
            json!({"type": "video_url", "video_url": "data:video/png;base64,AA==,BB=="}),
        ];
        for s in &shapes {
            let got = video_frames(0, 0, s, 8).unwrap();
            assert_eq!(got.len(), 2, "{s}");
        }
        assert_eq!(video_frames(0, 0, &shapes[3], 8).unwrap()[1], "data:image/png;base64,BB==");
        let e = video_frames(0, 3, &json!({"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}), 8)
            .unwrap_err()
            .message;
        assert!(e.contains("messages[0].content[3]") && e.contains("container"), "{e}");
        let e = video_frames(1, 0, &json!({"type": "video", "video": "https://example.com/v.mp4"}), 8).unwrap_err().message;
        assert!(e.contains("list of frames"), "{e}");
        let e = video_frames(0, 0, &shapes[0], 1).unwrap_err().message;
        assert!(e.contains("2 frames") && e.contains("limit is 1"), "{e}");
        assert!(video_frames(0, 0, &json!({"type": "video", "video": []}), 8)
            .unwrap_err()
            .message
            .contains("at least one frame"));
    }

    fn layout(grid: (usize, usize), per_crop_patches: [u32; 2]) -> FrameLayout {
        let n = 1 + grid.0 * grid.1;
        let merge = 4usize;
        let tokens = (per_crop_patches[0] as usize / merge) * (per_crop_patches[1] as usize / merge);
        FrameLayout {
            rows: n * tokens,
            crops: vec![per_crop_patches; n],
            downsample: gb_model::mrope_canvas::Downsample::X16,
            slice_grid: grid,
        }
    }

    fn markers() -> Markers {
        Markers {
            image_id_open: 248_090,
            image_id_close: 248_091,
            image_open: 248_078,
            image_close: 248_079,
            slice_open: 248_088,
            slice_close: 248_089,
        }
    }

    #[test]
    fn a_frame_is_the_thumbnail_then_the_slice_rows_joined_by_a_newline() {
        let m = markers();
        let mut out = Vec::new();
        push_frame(&mut out, IMAGE_PAD, &layout((2, 3), [32, 32]), &m, 4).unwrap();
        // thumbnail: 64 pads between <image> and </image>
        assert_eq!(out[0], m.image_open);
        assert!(out[1..65].iter().all(|&t| t == IMAGE_PAD) && out[65] == m.image_close);
        // 2 rows of 3 slices: <slice> 64 pads </slice> each, a newline between the rows only
        let slices = &out[66..];
        assert_eq!(slices.len(), 6 * 66 + 1);
        assert_eq!(slices.iter().filter(|&&t| t == NEWLINE).count(), 1);
        assert_eq!(slices[3 * 66], NEWLINE);
        assert_eq!(out.iter().filter(|&&t| t == IMAGE_PAD).count(), 7 * 64);
        // a thumbnail alone has no slice markers
        let mut one = Vec::new();
        push_frame(&mut one, VIDEO_PAD, &layout((0, 0), [32, 32]), &m, 4).unwrap();
        assert_eq!(one.len(), 66);
        assert!(one.iter().all(|&t| t != m.slice_open && t != NEWLINE));
        // a layout whose crops do not make its grid is refused by name
        let mut bad = layout((2, 2), [32, 32]);
        bad.crops.pop();
        assert!(push_frame(&mut Vec::new(), IMAGE_PAD, &bad, &m, 4).unwrap_err().message.contains("crops"));
    }

    #[test]
    fn a_span_absorbs_the_newlines_after_its_frame_and_nothing_else() {
        let out = vec![1, 2, 3, NEWLINE, 4, 5];
        assert_eq!(span_end(&out, 3), 4);
        assert_eq!(span_end(&out, 2), 2, "an end that is not before a newline stays");
        let two = vec![NEWLINE, NEWLINE, 7];
        assert_eq!(span_end(&two, 0), 2);
        assert_eq!(span_end(&two, 3), 3, "the end of the prompt");
    }

    #[test]
    fn fnv_names_a_preprocessing_the_way_the_image_path_of_the_server_does() {
        // http.rs: `pp.id().bytes().fold(0xcbf29ce484222325, |h, b| (h ^ b).wrapping_mul(0x100000001b3))`
        assert_eq!(fnv_str(""), 0xcbf2_9ce4_8422_2325);
        assert_ne!(fnv_str("a"), fnv_str("b"));
        let a = MiniCpmPreprocessing::minicpmv47();
        let b = a.clone().with_max_slice_nums(1).unwrap();
        assert_ne!(fnv_str(&ImagePreprocessor::id(&a)), fnv_str(&ImagePreprocessor::id(&b)), "slices are part of the key");
    }
}
