//! The video part of a Nemotron chat request, before any pixel: what a `video_url` / `video` / `input_video` part may carry, what is refused and
//! by what name, and the text the processor turns `<video>` into. The reference behaviour is `docs/NEMOTRON_VIDEO.md`.
//!
//! **The request format.** A video is a list of FRAMES (the processor samples nothing: what it is given is what the model sees):
//!
//! ```json
//! {"type": "video_url", "video_url": {"url": ["data:image/png;base64,...", {"url": "data:image/jpeg;base64,..."}], "fps": 2.0, "frame_indices": [0, 15]}}
//! ```
//!
//! - the payload is under `video_url`, `video` or `input_video`, and is an array of frames, an object with `url` (or `frames`) holding that
//!   array, or one string `data:video/jpeg;base64,<f1>,<f2>,...` (the frame-sequence convention of other OpenAI-compatible servers; `png`
//!   and `webp` too: each frame is a still image, decoded and bounded like one);
//! - a frame is a URL string, or an object with `url` or `image_url` (a string or `{"url": ...}`); the bytes go through the image path's
//!   decoders, so PNG, JPEG and WebP frames work (the format is sniffed from each frame's bytes; an animated WebP frame is refused);
//! - `fps` (a number above 0) and `frame_indices` (one non-negative integer per frame; needs `fps`) become the processor's `VideoMetadata`: the
//!   labels then carry `sampled at X seconds`. Without them the labels are `Frame N`, the processor's default;
//! - containers (`data:video/mp4`, `webm`, `gif`, a bare URL) are refused by name: no codec is linked (the proposal for an `ffmpeg` subprocess
//!   is in the document).
//!
//! Refusals name the part (`messages[i].content[j]`) and, for a frame, its number.

use super::{part_kind, Part, VIDEO_TYPES};
use crate::api::ApiError;
use gb_vision::nemotron_video::{frame_labels, VideoConfig, VideoMetadata};
use serde_json::Value;

/// The text the template renders once for a video part, and the processor replaces (the first one only) with the labelled chunks.
pub const VIDEO_TAG: &str = "<video>";

/// What the server admits of one video, decided from the request and the frame sizes alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoLimits {
    /// Frames in the request (two make a tubelet).
    pub max_frames: usize,
    /// Placeholder rows of the video BEFORE pruning (tubelets x the tile's tokens): the tower's work and the features' memory scale with it.
    pub max_tokens: usize,
}

impl Default for VideoLimits {
    /// 32 frames (16 tubelets) and 4,096 tokens: exactly 32 square frames (16 x 256), 32 frames of 16:9 are 4,032.
    fn default() -> Self {
        Self { max_frames: 32, max_tokens: 4096 }
    }
}

/// Everything a server decides about video: the limits and the pruning (off unless the owner turns it on).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VideoSettings {
    pub limits: VideoLimits,
    pub config: VideoConfig,
}

/// One video part, parsed: where it is, its frames' URLs in order, and the metadata the labels use.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoRef {
    pub message: usize,
    pub part: usize,
    /// The part's `type` (or the payload key the template would not render).
    pub kind: String,
    pub frames: Vec<String>,
    pub metadata: VideoMetadata,
}

impl VideoRef {
    /// `messages[i].content[j]`.
    pub fn at(&self) -> String {
        format!("messages[{}].content[{}]", self.message, self.part)
    }
}

fn bad(m: impl Into<String>) -> ApiError {
    ApiError::bad(m)
}

/// The URL of one frame: a string, or an object with `url` or `image_url`.
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
    url.map(str::to_string)
        .ok_or_else(|| bad(format!("{at}: frame {} is not a URL (a string, or an object with `url` or `image_url`)", i + 1)))
}

/// `data:video/jpeg;base64,<f1>,<f2>,...`: the frames as image data URIs. Other containers and bare URLs are refused by name.
fn frame_sequence(at: &str, s: &str) -> Result<Vec<String>, ApiError> {
    if let Some(rest) = s.strip_prefix("data:video/") {
        let (head, data) = rest.split_once(',').unwrap_or((rest, ""));
        let mut kinds = head.split(';');
        let sub = kinds.next().unwrap_or("").to_ascii_lowercase();
        let b64 = kinds.any(|k| k.eq_ignore_ascii_case("base64"));
        if matches!(sub.as_str(), "jpeg" | "jpg" | "png" | "webp") && b64 {
            let image = match sub.as_str() {
                "png" | "webp" => sub.as_str(),
                _ => "jpeg",
            };
            return Ok(data.split(',').map(|f| format!("data:image/{image};base64,{f}")).collect());
        }
        return Err(bad(format!(
            "{at}: a `data:video/{sub}` container is not decoded by this server (no codec is linked); send the frames: an array of image data \
             URLs, or `data:video/jpeg;base64,<frame>,<frame>,...`"
        )));
    }
    Err(bad(format!(
        "{at}: a video is a list of frames, not a single URL or file (no container is decoded by this server); send an array of image data URLs"
    )))
}

/// A video part's payload, parsed and checked against the limits that do not need the frames' pixels.
pub fn parse_part(message: usize, part_index: usize, part: &Value, limits: &VideoLimits) -> Result<VideoRef, ApiError> {
    let at = format!("messages[{message}].content[{part_index}]");
    let kind = part
        .get("type")
        .and_then(Value::as_str)
        .filter(|t| VIDEO_TYPES.contains(t))
        .map(str::to_string)
        .unwrap_or_else(|| super::media_name(part));
    let payload = ["video_url", "video", "input_video"]
        .iter()
        .find_map(|k| part.get(*k))
        .ok_or_else(|| bad(format!("{at}: a `{kind}` part needs its frames under `video_url` (an array of image data URLs)")))?;
    let (frames_value, fps, indices) = match payload {
        Value::Object(o) => (
            o.get("url")
                .or_else(|| o.get("frames"))
                .ok_or_else(|| bad(format!("{at}: `{kind}` has no `url` (an array of frames)")))?,
            o.get("fps").filter(|v| !v.is_null()),
            o.get("frame_indices").filter(|v| !v.is_null()),
        ),
        other => (other, None, None),
    };
    let frames: Vec<String> = match frames_value {
        Value::Array(a) => a.iter().enumerate().map(|(i, f)| frame_url(&at, i, f)).collect::<Result<_, _>>()?,
        Value::String(s) => frame_sequence(&at, s)?,
        _ => return Err(bad(format!("{at}: the frames of a video are an array of image data URLs"))),
    };
    if frames.is_empty() {
        return Err(bad(format!("{at}: a video needs at least one frame")));
    }
    if frames.len() > limits.max_frames {
        return Err(bad(format!(
            "{at}: the video has {} frames; the limit is {} (--max-video-frames)",
            frames.len(),
            limits.max_frames
        )));
    }
    if let Some(i) = frames.iter().position(|f| f.is_empty()) {
        return Err(bad(format!("{at}: frame {} is empty", i + 1)));
    }
    let fps = match fps {
        None => None,
        Some(v) => Some(
            v.as_f64()
                .filter(|f| f.is_finite() && *f > 0.0)
                .ok_or_else(|| bad(format!("{at}: `fps` must be a number above 0, got {v}")))?,
        ),
    };
    let frame_indices = match indices {
        None => None,
        Some(v) => {
            if fps.is_none() {
                return Err(bad(format!("{at}: `frame_indices` needs `fps` (the processor ignores indices without it)")));
            }
            let a =
                v.as_array().ok_or_else(|| bad(format!("{at}: `frame_indices` must be an array of non-negative integers")))?;
            if a.len() != frames.len() {
                return Err(bad(format!("{at}: `frame_indices` has {} entries for {} frames", a.len(), frames.len())));
            }
            let idx: Option<Vec<i64>> =
                a.iter().map(|x| x.as_u64().filter(|&n| n <= u32::MAX as u64).map(|n| n as i64)).collect();
            Some(idx.ok_or_else(|| bad(format!("{at}: `frame_indices` must be non-negative integers")))?)
        }
    };
    let metadata = VideoMetadata { fps, frame_indices };
    // the labels are built from it: refuse what cannot be (an fps that gives no whole millisecond count) here, by name
    frame_labels(frames.len(), &metadata).map_err(|e| bad(format!("{at}: {e}")))?;
    Ok(VideoRef { message, part: part_index, kind, frames, metadata })
}

/// What a request holds of media the video path cares about.
#[derive(Debug, Default)]
pub struct Found {
    pub videos: Vec<VideoRef>,
    /// `messages[i].content[j]` of the first still image.
    pub first_image: Option<String>,
}

/// Every video part of `messages` (parsed) and the first image part, by the template's own test of each.
pub fn find(messages: &Value, limits: &VideoLimits) -> Result<Found, ApiError> {
    let mut out = Found::default();
    for (i, m) in messages.as_array().into_iter().flatten().enumerate() {
        for (j, part) in m.get("content").and_then(Value::as_array).into_iter().flatten().enumerate() {
            match part_kind(part) {
                Part::Video => out.videos.push(parse_part(i, j, part, limits)?),
                Part::Image if out.first_image.is_none() => out.first_image = Some(format!("messages[{i}].content[{j}]")),
                _ => {}
            }
        }
    }
    Ok(out)
}

/// Whether `messages` has a part the template renders as a video (without parsing it).
pub fn has_video(messages: &Value) -> bool {
    messages
        .as_array()
        .into_iter()
        .flatten()
        .any(|m| m.get("content").and_then(Value::as_array).is_some_and(|ps| ps.iter().any(|p| part_kind(p) == Part::Video)))
}

/// The one video of a request, or the refusal of anything else: more than one video, or an image beside it (the model's scatter fills every
/// `<image>` placeholder from one source, so the reference raises for both).
pub fn the_video(messages: &Value, limits: &VideoLimits) -> Result<Option<VideoRef>, ApiError> {
    let found = find(messages, limits)?;
    let mut videos = found.videos.into_iter();
    let Some(video) = videos.next() else { return Ok(None) };
    if let Some(second) = videos.next() {
        return Err(bad(format!(
            "{}: at most one video per request (the processor expands one `<video>`; the first is at {})",
            second.at(),
            video.at()
        )));
    }
    if let Some(image) = found.first_image {
        return Err(bad(format!(
            "{image}: an image and a video ({}) in one request are not supported (images and video fill the same placeholder tokens, which the \
             reference model cannot do); send them in separate requests",
            video.at()
        )));
    }
    Ok(Some(video))
}

/// The URL strings of the frames of a video part's payload, without copying them: an array of strings or of objects with `url` or
/// `image_url`, an object holding that array under `url` or `frames`, or one frame-sequence string.
fn frame_url_strs<'a>(part: &'a Value) -> Vec<&'a str> {
    let payload = ["video_url", "video", "input_video"].iter().find_map(|k| part.get(*k));
    let frames = match payload {
        Some(Value::Object(o)) => o.get("url").or_else(|| o.get("frames")),
        other => other,
    };
    let one = |f: &'a Value| -> Option<&'a str> {
        match f {
            Value::String(s) => Some(s.as_str()),
            Value::Object(o) => match o.get("url").or_else(|| o.get("image_url")) {
                Some(Value::String(s)) => Some(s.as_str()),
                Some(Value::Object(inner)) => inner.get("url").and_then(Value::as_str),
                _ => None,
            },
            _ => None,
        }
    };
    match frames {
        Some(Value::Array(a)) => a.iter().filter_map(one).collect(),
        Some(Value::String(s)) => vec![s.as_str()],
        _ => Vec::new(),
    }
}

/// Bytes the frame URLs of the request's video parts retain while it is prepared, from the request alone and without copying a URL (the
/// preparation gate is charged with it BEFORE the frames are copied into [`VideoRef`]s): the URLs' lengths, one frame's fetched bytes in
/// flight (`max_bytes`: frames are decoded one at a time) and a little per frame. `None` without a video part. A frame count over
/// `video.max_frames` and a URL over its size limit are refused here by name, the same refusals `parse_part` and the decode make later.
pub fn retained_bytes(messages: &Value, video: &VideoLimits, limits: &crate::vision::Limits) -> Result<Option<usize>, ApiError> {
    let mut urls = 0usize;
    let mut frames = 0usize;
    let mut any = false;
    for (i, m) in messages.as_array().into_iter().flatten().enumerate() {
        for (j, part) in m.get("content").and_then(Value::as_array).into_iter().flatten().enumerate() {
            if part_kind(part) != Part::Video {
                continue;
            }
            any = true;
            let at = format!("messages[{i}].content[{j}]");
            let list = frame_url_strs(part);
            if list.len() > video.max_frames {
                return Err(bad(format!(
                    "{at}: the video has {} frames; the limit is {} (--max-video-frames)",
                    list.len(),
                    video.max_frames
                )));
            }
            let data_max = limits.max_bytes.div_ceil(3).saturating_mul(4).saturating_add(64);
            for (k, url) in list.iter().enumerate() {
                // a frame sequence is one string holding every frame: its limit is the frames' together
                let max = if url.starts_with("data:video/") {
                    data_max.saturating_mul(video.max_frames)
                } else if url.starts_with("data:") {
                    data_max
                } else {
                    2048
                };
                if url.len() > max {
                    return Err(ApiError::status(
                        413,
                        "invalid_request_error",
                        format!("{at}: frame {}: image URL exceeds size limit", k + 1),
                    ));
                }
                urls = urls.saturating_add(url.len());
            }
            frames = frames.saturating_add(list.len().max(1));
        }
    }
    if !any {
        return Ok(None);
    }
    urls.checked_add(limits.max_bytes)
        .and_then(|n| n.checked_add(frames.saturating_mul(64)))
        .map(Some)
        .ok_or_else(|| bad("video retained size overflow"))
}

/// The text that replaces the single `<video>`: one label and one `<image>` per tubelet, joined by `\n`. Each `<image>` is later expanded to
/// `<img>` + k x `<image>` + `</img>` by the format's id expansion, which gives exactly the processor's per-tubelet chunk.
pub fn one_placeholder_per_tubelet(labels: &[String]) -> String {
    labels.iter().map(|l| format!("{l}<image>")).collect::<Vec<_>>().join("\n")
}

/// The text the processor hands the tokenizer for a video: `<video>` of `text` (the first one) replaced by, per tubelet,
/// `label + <img> + <image> x counts[g] + </img>`, the chunks joined by `\n` (`processing_nemotron_h_omni.py`). `counts` is the full count of
/// each tubelet for the processor itself; a pruned prompt is the same text with each tubelet's kept count.
pub fn expand_video_text(text: &str, labels: &[String], counts: &[usize]) -> Result<String, ApiError> {
    if labels.len() != counts.len() {
        return Err(bad(format!("{} labels for {} tubelets", labels.len(), counts.len())));
    }
    let at = text.find(VIDEO_TAG).ok_or_else(|| bad("the prompt has no <video> placeholder"))?;
    let mut chunk = String::new();
    for (g, (label, &n)) in labels.iter().zip(counts).enumerate() {
        if g > 0 {
            chunk.push('\n');
        }
        chunk.push_str(label);
        chunk.push_str(super::IMAGE_START);
        for _ in 0..n {
            chunk.push_str(super::IMAGE_TOKEN);
        }
        chunk.push_str(super::IMAGE_END);
    }
    let mut out = String::with_capacity(text.len() + chunk.len());
    out.push_str(&text[..at]);
    out.push_str(&chunk);
    out.push_str(&text[at + VIDEO_TAG.len()..]);
    Ok(out)
}
