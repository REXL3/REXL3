//! The Nemotron 3.5 Super VL chat format: `<|im_start|>` framing, `<think>` blocks, and function-block `<tool_call><function=NAME>`
//! calls, from the pack's own `chat_template.jinja`.
//!
//! **Images.** The template renders each image part as `<image>` (and, when a message holds several, as `<image N><image>`, the label staying
//! text). The HF processor then replaces every `<image>` by `<img>` + `<image>` x N + `</img>`, N being the image's token count
//! (`processing_nemotron_h_omni.py`); the tokens are the ids the tower's features replace. This format does the same on the ids after
//! tokenization ([`OmniFormat::expand_images`]: `<img>`, `<image>`, `</img>` are special, unnormalised added tokens with no strip, so the ids are
//! those of the expanded text, which [`expand_text`] gives and the fixtures pin), and [`prep::NemotronImagePrep`] does everything else that needs
//! no model: decode, the tile, the pixels, the content key, the spans. A literal `<image>` typed in a text part takes the place of the image
//! parts of its message (the template renders no tag for them): the prompt's placeholders and the request's images are paired in order, and a
//! prompt where they do not pair up is refused. Audio parts are refused by name (the pack has no sound tower).
//!
//! **Video.** Off in this format until [`OmniFormat::with_video`] turns on the video path of `video` (the request format, the limits, the
//! refusals) and `prep::NemotronImagePrep::ingest` (frames to tubelets, the labelled prompt, one span per tubelet); without it a video part is
//! refused by name. The server routes video requests through `http.rs::video_prompt` and enables the path unless `--no-video` is given; see
//! `docs/NEMOTRON_VIDEO.md`.

pub mod prep;
pub mod video;

use crate::api::{self, ApiError};
use crate::chat_format::{render_normalized, ChatFormat, FormatEntry, RenderedChat};
use crate::reasoning_caps::{AliasTable, EffortLevel, Markers, ReasoningCapabilities, TemplateVar, ThinkingSwitch};
use crate::template::{ChatTemplate, TemplateError};
use crate::tokenizer::Tok;
use crate::tools::{ToolSyntax, FUNCTION_BLOCK_TOOL_SYNTAX};
use crate::vision::Span;
use gb_vision::nemotron::NemotronPreprocessing;
use serde_json::{Map, Value};
use std::path::Path;
use std::sync::Arc;

pub const ENTRY: FormatEntry =
    FormatEntry { id: "nemotron_omni", names: &["NemotronH_Omni_Reasoning_V3", "nemotron_h_omni"], open };

/// The placeholder the template renders once per image part, and the id the tower's features replace.
pub const IMAGE_TOKEN: &str = "<image>";
/// The tokens that bracket an image's run of placeholders.
pub const IMAGE_START: &str = "<img>";
pub const IMAGE_END: &str = "</img>";
/// The processor's own scratch marker for the expansion (`<|placeholder|>`, turned into `<image>` at the end of its replacement).
const SCRATCH: &str = "<|placeholder|>";

/// What the template renders as an image: `type` is `image`, `image_url` or `input_image` (`render_content` in `chat_template.jinja`).
const IMAGE_TYPES: [&str; 3] = ["image", "image_url", "input_image"];
const VIDEO_TYPES: [&str; 3] = ["video", "video_url", "input_video"];
const AUDIO_TYPES: [&str; 3] = ["audio", "audio_url", "input_audio"];

/// The reasoning tokens a request with `reasoning_effort: "low"` may spend when it names no `max_reasoning_tokens`.
pub const LOW_EFFORT_REASONING_TOKENS: usize = 1024;

/// The reasoning capabilities of the Nemotron 3.5 Super VL template (`chat_template.jinja` lines 65-67 and 241-244): `enable_thinking`
/// (default true; false renders the closed `<think></think>` block), `medium_effort` (default false: when true, the last user message gets
/// `\n\n{reasoning effort: efficient}` appended), `truncate_history_thinking` (default true). `preserve_thinking` is accepted and passed
/// through as before, unread. `reasoning_effort` (top level or in the kwargs) maps onto them: none -> off; low -> on + medium_effort + a
/// 1,024-token budget; medium -> on + medium_effort; high, xhigh -> on (the template's default effort). `max_reasoning_tokens` caps the
/// thought (`</think>` is one token in the pack). See `docs/NEMOTRON_REASONING_API.md`.
pub fn reasoning_capabilities() -> ReasoningCapabilities {
    let think = |name: &str| EffortLevel::new(name, true).sets("enable_thinking", true);
    ReasoningCapabilities {
        reasons: true,
        markers: Some(Markers::new("<think>", "</think>")),
        switch: Some(ThinkingSwitch { var: "enable_thinking".into(), closes_block: true }),
        template_thinks_by_default: true,
        variables: vec![
            TemplateVar::boolean("medium_effort").default_value(false.into()),
            TemplateVar::boolean("truncate_history_thinking").default_value(true.into()),
            TemplateVar::boolean("preserve_thinking").inert(),
        ],
        levels: vec![
            EffortLevel::new("none", false).sets("enable_thinking", false).requires("medium_effort", false),
            think("low").sets("medium_effort", true).budget(LOW_EFFORT_REASONING_TOKENS),
            think("medium").sets("medium_effort", true),
            think("high").requires("medium_effort", false),
            think("xhigh").requires("medium_effort", false),
        ],
        budget: true,
        aliases: vec![AliasTable::openai()],
    }
}

fn open(dir: &Path) -> Result<Box<dyn ChatFormat>, String> {
    let src = std::fs::read_to_string(dir.join("chat_template.jinja")).map_err(|e| format!("chat_template.jinja: {e}"))?;
    let mut f = OmniFormat::new(&src).map_err(|e| e.to_string())?;
    // the image processor's parameters, when the folder has them: a text-only folder does not need them, a wrong one is refused here
    if ["preprocessor_config.json", "processor_config.json"].iter().any(|n| dir.join(n).is_file()) {
        let p = NemotronPreprocessing::from_dir(dir).map_err(|e| e.to_string())?;
        f = f.with_preprocessing(p);
    }
    Ok(Box::new(f))
}

pub struct OmniFormat {
    template: ChatTemplate,
    /// The image processor's parameters of the pack (`preprocessor_config.json`), when it was read.
    preprocessing: Option<Arc<NemotronPreprocessing>>,
    /// Video input (limits and pruning), when the server turned it on: `None` refuses video parts by name as every format did before.
    video: Option<Arc<video::VideoSettings>>,
}

impl OmniFormat {
    pub fn new(template_source: &str) -> Result<Self, TemplateError> {
        Ok(Self { template: ChatTemplate::new(template_source)?, preprocessing: None, video: None })
    }

    pub fn with_preprocessing(mut self, p: NemotronPreprocessing) -> Self {
        self.preprocessing = Some(Arc::new(p));
        self
    }

    /// The image processor's parameters, when the model folder had them.
    pub fn preprocessing(&self) -> Option<&Arc<NemotronPreprocessing>> {
        self.preprocessing.as_ref()
    }

    /// Accept video parts (frames as image data URLs) under these limits, with this pruning. The settings are checked here: an out-of-range
    /// pruning rate or a zero limit is an `Err` that names it.
    pub fn with_video(mut self, settings: video::VideoSettings) -> Result<Self, String> {
        settings.config.validate()?;
        if settings.limits.max_frames == 0 || settings.limits.max_tokens == 0 {
            return Err("video limits must be at least 1 frame and 1 token".into());
        }
        self.video = Some(Arc::new(settings));
        Ok(self)
    }

    /// The video settings, when video input is on.
    pub fn video(&self) -> Option<&Arc<video::VideoSettings>> {
        self.video.as_ref()
    }

    /// [`OmniFormat::with_video`] in place: turn video input on before the format is shared (`ChatFormat::enable_video`).
    pub fn set_video(&mut self, settings: video::VideoSettings) -> Result<(), String> {
        settings.config.validate()?;
        if settings.limits.max_frames == 0 || settings.limits.max_tokens == 0 {
            return Err("video limits must be at least 1 frame and 1 token".into());
        }
        self.video = Some(Arc::new(settings));
        Ok(())
    }

    /// Whether video parts are accepted (the pixel side is `prep`, the model's is the tower's).
    pub fn serves_video(&self) -> bool {
        self.video.is_some()
    }
}

/// `api::refuse_media` for every part of a request except video parts, without copying a frame: each part is reduced to its `type` and the
/// media keys it carries before the shared rule sees it.
fn refuse_residual(messages: &Value, allow_images: bool) -> Result<(), ApiError> {
    let Some(list) = messages.as_array() else { return Ok(()) };
    for (i, m) in list.iter().enumerate() {
        let Some(parts) = m.get("content").and_then(Value::as_array) else { continue };
        for (j, part) in parts.iter().enumerate() {
            if part_kind(part) == Part::Video {
                continue;
            }
            let mut slim = Map::new();
            if let Some(t) = part.get("type") {
                slim.insert("type".into(), t.clone());
            }
            for k in ["image", "image_url", "video", "video_url", "input_audio"] {
                if part.get(k).is_some() {
                    slim.insert(k.into(), Value::Bool(true));
                }
            }
            if let Err(mut e) =
                api::refuse_media(&Value::Array(vec![serde_json::json!({"content": [Value::Object(slim)]})]), allow_images)
            {
                e.message = e.message.replacen("messages[0].content[0]", &format!("messages[{i}].content[{j}]"), 1);
                return Err(e);
            }
        }
    }
    Ok(())
}

/// What the Nemotron template does with a content part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Image,
    Video,
    Audio,
    /// An image, video or audio payload under a `type` the template does not render: it would be dropped without a trace.
    Dropped,
    Other,
}

pub fn part_kind(part: &Value) -> Part {
    let ty = part.get("type").and_then(Value::as_str).unwrap_or("");
    if IMAGE_TYPES.contains(&ty) {
        return Part::Image;
    }
    if VIDEO_TYPES.contains(&ty) {
        return Part::Video;
    }
    if AUDIO_TYPES.contains(&ty) {
        return Part::Audio;
    }
    if ty == "text" || ty == "input_text" {
        return Part::Other;
    }
    let has = |keys: &[&str]| keys.iter().any(|k| part.get(*k).is_some());
    if has(&["image", "image_url"]) || has(&["video", "video_url", "input_video"]) || has(&["audio", "audio_url", "input_audio"])
    {
        Part::Dropped
    } else {
        Part::Other
    }
}

/// `video`, `video_url`, `input_audio`...: the name of a video or audio part, from its `type` or else its payload key.
fn media_name(part: &Value) -> String {
    if let Some(t) = part.get("type").and_then(Value::as_str).filter(|t| VIDEO_TYPES.contains(t) || AUDIO_TYPES.contains(t)) {
        return t.to_string();
    }
    ["video", "video_url", "input_video", "audio", "audio_url", "input_audio"]
        .iter()
        .find(|k| part.get(**k).is_some())
        .map_or("media", |k| *k)
        .to_string()
}

fn unsupported_media(at: &str, name: &str, audio: bool) -> ApiError {
    ApiError::bad(format!(
        "{at}: `{name}` input is not supported by this server: {}; send text and images only",
        if audio { "this model has no audio input" } else { "video input is not available yet" }
    ))
}

/// The text the processor hands the tokenizer: every `<image>` of `text` replaced, in order, by `<img>` + `<image>` x N + `</img>` with the
/// N of `counts` (`processing_nemotron_h_omni.py`, `__call__`, whose replacement goes through a `<|placeholder|>` marker so that the expansion is
/// not itself expanded). The tags and the counts must pair up one to one: the processor raises when there are more tags than images and
/// silently ignores images without a tag, and a `<|placeholder|>` of the text becomes an image token there; all three are errors here.
pub fn expand_text(text: &str, counts: &[usize]) -> Result<String, ApiError> {
    if text.contains(SCRATCH) {
        return Err(ApiError::bad(format!("the literal text {SCRATCH} is not allowed in messages that carry images (the processor turns it into an image token)")));
    }
    let tags = text.matches(IMAGE_TOKEN).count();
    if tags != counts.len() {
        return Err(ApiError::bad(format!(
            "the prompt has {tags} {IMAGE_TOKEN} placeholders but {} images were sent (a literal {IMAGE_TOKEN} typed in a message takes the place of the image parts of that message)",
            counts.len()
        )));
    }
    let run: usize = counts.iter().sum();
    let mut out =
        String::with_capacity(text.len() + run * IMAGE_TOKEN.len() + counts.len() * (IMAGE_START.len() + IMAGE_END.len()));
    let mut rest = text;
    for &n in counts {
        let at = rest.find(IMAGE_TOKEN).expect("counted above");
        out.push_str(&rest[..at]);
        out.push_str(IMAGE_START);
        for _ in 0..n {
            out.push_str(IMAGE_TOKEN);
        }
        out.push_str(IMAGE_END);
        rest = &rest[at + IMAGE_TOKEN.len()..];
    }
    out.push_str(rest);
    Ok(out)
}

impl ChatFormat for OmniFormat {
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

    /// A thinking prompt ends with an open `<think>` and not a closed empty block.
    fn starts_in_reasoning(&self, prompt: &str) -> bool {
        prompt.trim_end_matches('\n').ends_with("<think>") && !prompt.ends_with("</think>\n\n")
    }

    /// [`reasoning_capabilities`]: see `docs/NEMOTRON_REASONING_API.md`.
    fn reasoning_capabilities(&self) -> Option<crate::reasoning_caps::ReasoningCapabilities> {
        Some(reasoning_capabilities())
    }

    fn chat_template(&self) -> Option<&ChatTemplate> {
        Some(&self.template)
    }

    fn tool_syntax(&self) -> ToolSyntax {
        FUNCTION_BLOCK_TOOL_SYNTAX
    }

    /// Image parts are accepted when the server has vision (`allow_images`). Video and audio parts are refused by name: the frame path is not
    /// built and the pack has no sound tower. A part that carries an image under a `type` the template does not render is refused too, since
    /// the template would drop it without a trace.
    fn check_media(&self, messages: &Value, allow_images: bool) -> Result<(), ApiError> {
        let Some(list) = messages.as_array() else { return Ok(()) };
        for (i, m) in list.iter().enumerate() {
            let Some(parts) = m.get("content").and_then(Value::as_array) else { continue };
            for (j, part) in parts.iter().enumerate() {
                let at = format!("messages[{i}].content[{j}]");
                match part_kind(part) {
                    // video is accepted only when the server turned the video path on (`with_video`), and then only with the vision tower
                    Part::Video if self.video.is_some() => {
                        if !allow_images {
                            return Err(ApiError::bad(format!(
                                "{at}: `{}` input is not supported by this server (it was started without image support: the video path \
                                 needs the vision tower); send text only",
                                media_name(part)
                            )));
                        }
                    }
                    Part::Video => return Err(unsupported_media(&at, &media_name(part), false)),
                    Part::Audio => return Err(unsupported_media(&at, &media_name(part), true)),
                    Part::Dropped => {
                        let key =
                            ["image", "image_url", "video", "video_url", "input_video", "audio", "audio_url", "input_audio"]
                                .iter()
                                .find(|k| part.get(**k).is_some())
                                .copied()
                                .unwrap_or("media");
                        // a video or audio payload under no type the template knows is the same refusal, by its key
                        if key.contains("video") && self.video.is_some() {
                            return Err(ApiError::bad(format!(
                                "{at}: a part with a `{key}` payload is not a content part the chat template renders (it would be dropped \
                                 silently); use `video_url`, `video` or `input_video` as its type"
                            )));
                        }
                        if key.contains("video") || key.contains("audio") {
                            return Err(unsupported_media(&at, key, key.contains("audio")));
                        }
                        let name = part.get("type").and_then(Value::as_str).unwrap_or("(no type)");
                        return Err(ApiError::bad(format!(
                            "{at}: a part of type `{name}` with a `{key}` payload is not a content part the chat template renders (it would be dropped silently); use `image_url`, `image` or `input_image` as its type"
                        )));
                    }
                    Part::Image if !allow_images => {
                        let ty = part.get("type").and_then(Value::as_str).unwrap_or("image");
                        return Err(ApiError::bad(format!(
                            "{at}: `{ty}` input is not supported by this server (it was started without image support); send text only"
                        )));
                    }
                    Part::Image | Part::Other => {}
                }
            }
        }
        // anything the shared rule still names (a payload key outside the part kinds above); with video on, the video parts are checked as a
        // whole instead: one video, no image beside it, the limits
        match &self.video {
            None => api::refuse_media(messages, allow_images),
            Some(v) => {
                refuse_residual(messages, allow_images)?;
                video::the_video(messages, &v.limits).map(|_| ())
            }
        }
    }

    /// Video input is on (`with_video`): the HTTP layer routes a request with a video part through [`prep::VideoIngress`].
    fn video(&self) -> Option<&dyn prep::VideoIngress> {
        self.video.as_ref().map(|_| self as &dyn prep::VideoIngress)
    }

    fn enable_video(&mut self, options: &crate::chat_format::VideoOptions) -> Result<(), String> {
        self.set_video(video::VideoSettings {
            limits: video::VideoLimits { max_frames: options.max_frames, max_tokens: options.max_tokens },
            config: gb_vision::nemotron_video::VideoConfig {
                pruning_rate: options.pruning_rate,
                require_square_grid: options.require_square_grid,
            },
        })
    }

    /// The template's own test (`type` is `image`, `image_url` or `input_image`), so the images found in a request and the `<image>` tags it
    /// renders are the same set in the same order.
    fn is_image_part(&self, part: &Value) -> bool {
        part_kind(part) == Part::Image
    }

    fn image_placeholder(&self, tok: &Tok) -> Result<(u32, &'static str), String> {
        tok.token_to_id(IMAGE_TOKEN)
            .map(|id| (id, IMAGE_TOKEN))
            .ok_or_else(|| format!("the tokenizer has no {IMAGE_TOKEN} token"))
    }

    /// The tokens of an image of `height` x `width` pixels sent alone, from the processor's parameters (`None` when the folder had none).
    fn image_tokens(&self, height: usize, width: usize) -> Option<Result<usize, String>> {
        let p = self.preprocessing.as_ref()?;
        Some(p.plan_for(width, height, 1).map(|t| t.tokens).map_err(|e| e.to_string()))
    }

    /// The RADIO tower and its projector are the model side's; this format and its pixel preprocessing are complete. Starting a server with
    /// images on a build whose tower is not built is refused by the family's preflight (`refuse_unbuilt_vision`), before this is asked.
    fn serves_images(&self) -> bool {
        true
    }

    /// Replace each `<image>` of the tokenized prompt by `<img>` + `<image>` x `counts[i]` + `</img>` (the ids of [`expand_text`]) and report where
    /// each run of `<image>` landed: `Span::start` is the index of its first `<image>`, after the `<img>`.
    fn expand_images(&self, tok: &Tok, ids: &[u32], counts: &[usize]) -> Result<(Vec<u32>, Vec<Span>), ApiError> {
        let (pad, literal) = self.image_placeholder(tok).map_err(ApiError::bad)?;
        let special = |t: &str| tok.token_to_id(t).ok_or_else(|| ApiError::bad(format!("the tokenizer has no {t} token")));
        let (open, close) = (special(IMAGE_START)?, special(IMAGE_END)?);
        let pads = ids.iter().filter(|&&t| t == pad).count();
        if pads != counts.len() {
            return Err(ApiError::bad(format!(
                "the prompt has {pads} image placeholders but {} images were sent (a literal {literal} typed in a message takes the place of the image parts of that message)",
                counts.len()
            )));
        }
        let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>() + 2 * counts.len());
        let mut spans = Vec::with_capacity(counts.len());
        let mut k = 0;
        for &t in ids {
            if t == pad {
                out.push(open);
                spans.push(Span { start: out.len(), tokens: counts[k] });
                out.extend(std::iter::repeat_n(pad, counts[k]));
                out.push(close);
                k += 1;
            } else {
                out.push(t);
            }
        }
        Ok((out, spans))
    }
}

impl prep::VideoIngress for OmniFormat {
    fn has_video(&self, messages: &Value) -> bool {
        self.video.is_some() && video::has_video(messages)
    }

    fn reserve(&self, messages: &Value, limits: &crate::vision::Limits) -> Result<Option<usize>, ApiError> {
        let v = self.video.as_ref().ok_or_else(|| ApiError::bad("video input is not enabled on this server"))?;
        video::retained_bytes(messages, &v.limits, limits)
    }

    /// The decoder and resampler of one frame at the largest tile the limits allow (`image_work_bytes`), and the frames kept as bf16 planes
    /// until the tubelets are encoded: `max_frames` of a tile of at most `max_tokens_per_image` tokens (4 patches of 3 x 256 values a token).
    fn work_bytes(&self, limits: &crate::vision::Limits) -> Result<usize, ApiError> {
        let v = self.video.as_ref().ok_or_else(|| ApiError::bad("video input is not enabled on this server"))?;
        let frames = limits
            .max_tokens_per_image
            .checked_mul(4 * 3 * 256 * 2)
            .and_then(|n| n.checked_mul(v.limits.max_frames))
            .ok_or_else(|| ApiError::bad("video preparation size overflow"))?;
        crate::preparation::image_work_bytes(limits.max_tokens_per_image)?
            .checked_add(frames)
            .ok_or_else(|| ApiError::bad("video preparation size overflow"))
    }

    fn ingest(
        &self,
        tok: &Tok,
        messages: &Value,
        kwargs: &Map<String, Value>,
        limits: &crate::vision::Limits,
        render_limit: usize,
        encoder: &dyn prep::NemotronVideoEncoder,
        work: Option<&crate::preparation::RequestWork>,
        fetch: &mut dyn FnMut(&crate::vision::ImageRef) -> Result<Vec<u8>, ApiError>,
    ) -> Result<prep::Ingested, ApiError> {
        let p = prep::NemotronImagePrep::from_format(self, *limits)
            .ok_or_else(|| ApiError::bad("this model folder has no image processor parameters (preprocessor_config.json)"))?;
        p.ingest_video_bounded(self, tok, messages, kwargs, render_limit, Some(encoder), work, fetch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const TEMPLATE: &str = include_str!("../../../../testdata/nemotron_omni/chat_template.jinja");

    /// A tokenizer folder with the Nemotron ids of the image tokens and a whitespace word-level vocabulary.
    pub(crate) fn mini_tok(tag: &str) -> Tok {
        let added = |id: u32, content: &str| json!({"id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true});
        let tj = json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [added(2, "<|im_start|>"), added(3, "<|im_end|>"), added(18, "<image>"), added(19, "<img>"), added(20, "</img>")],
            "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "a": 1}, "unk_token": "[UNK]"}
        });
        let dir = std::env::temp_dir().join(format!("rexl3-omni-mini-tok-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tj.to_string()).unwrap();
        Tok::from_dir(&dir).unwrap()
    }

    #[test]
    fn a_text_conversation_renders_with_the_pack_template_and_opens_a_thought() {
        let f = OmniFormat::new(TEMPLATE).unwrap();
        let msgs = serde_json::json!([{"role": "user", "content": "hi"}]);
        let out = f.render(&msgs, &Map::new()).unwrap();
        assert!(out.text.contains("<|im_start|>user\nhi<|im_end|>"), "{}", out.text);
        assert!(out.text.trim_end().ends_with("<|im_start|>assistant\n<think>") || out.text.contains("<think>"), "{}", out.text);
    }

    #[test]
    fn images_are_refused_until_the_tower_is_built() {
        let f = OmniFormat::new(TEMPLATE).unwrap();
        let msgs = serde_json::json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]);
        assert!(f.check_media(&msgs, false).is_err());
    }

    #[test]
    fn images_are_accepted_with_vision_and_refused_without_it_naming_the_part() {
        let f = OmniFormat::new(TEMPLATE).unwrap();
        for ty in ["image_url", "image", "input_image"] {
            let msgs = json!([{"role": "user", "content": [{"type": "text", "text": "x"}, {"type": ty, "image_url": "data:image/png;base64,AA=="}]}]);
            assert!(f.check_media(&msgs, true).is_ok(), "{ty}");
            let e = f.check_media(&msgs, false).unwrap_err();
            assert!(
                e.message.contains("messages[0].content[1]")
                    && e.message.contains(ty)
                    && e.message.contains("without image support"),
                "{}",
                e.message
            );
            assert!(f.is_image_part(&msgs[0]["content"][1]));
            assert!(!f.is_image_part(&msgs[0]["content"][0]));
        }
        // text-only requests never trip it
        assert!(f
            .check_media(
                &json!([{"role": "user", "content": "hi"}, {"role": "user", "content": [{"type": "text", "text": "x"}]}]),
                false
            )
            .is_ok());
    }

    #[test]
    fn video_and_audio_are_refused_by_name_whatever_the_vision_flag() {
        let f = OmniFormat::new(TEMPLATE).unwrap();
        for (part, name) in [
            (json!({"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}}), "video_url"),
            (json!({"type": "video", "video": "x"}), "`video`"),
            (json!({"type": "input_video", "video_url": "x"}), "input_video"),
            (json!({"type": "input_audio", "input_audio": {"data": "AA", "format": "wav"}}), "input_audio"),
            (json!({"type": "audio_url", "audio_url": {"url": "x"}}), "audio_url"),
            (json!({"video": "x"}), "`video`"),
        ] {
            let msgs = json!([{"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": "data:y"}, part]}]);
            for allow in [true, false] {
                let e = f.check_media(&msgs, allow).unwrap_err();
                if allow {
                    assert!(e.message.contains("messages[0].content[2]") && e.message.contains(name), "{}", e.message);
                    assert!(e.message.contains("not supported"), "{}", e.message);
                } else {
                    // without vision the first refused part is the image before it
                    assert!(e.message.contains("messages[0].content[1]") && e.message.contains("image support"), "{}", e.message);
                }
            }
        }
    }

    #[test]
    fn an_image_the_template_would_drop_is_refused_not_ignored() {
        let f = OmniFormat::new(TEMPLATE).unwrap();
        // a payload without one of the types the template renders
        for part in [json!({"image_url": {"url": "data:image/png;base64,AA"}}), json!({"type": "picture", "image": "x"})] {
            let msgs = json!([{"role": "user", "content": [part]}]);
            let e = f.check_media(&msgs, true).unwrap_err();
            assert!(e.message.contains("messages[0].content[0]") && e.message.contains("dropped silently"), "{}", e.message);
            assert!(!f.is_image_part(&msgs[0]["content"][0]));
        }
        // an unknown part without a media payload is the template's business, as before
        assert!(f.check_media(&json!([{"role": "user", "content": [{"type": "thing", "x": 1}]}]), true).is_ok());
    }

    #[test]
    fn the_processors_expansion_is_reproduced_on_text() {
        // one image, two in a message (the label stays text), a typed tag, none
        assert_eq!(expand_text("a<image>b", &[3]).unwrap(), "a<img><image><image><image></img>b");
        assert_eq!(
            expand_text("<image 1><image>x\n<image 2><image>", &[1, 2]).unwrap(),
            "<image 1><img><image></img>x\n<image 2><img><image><image></img>"
        );
        assert_eq!(expand_text("no images", &[]).unwrap(), "no images");
        // more tags than images, more images than tags, and the scratch marker are errors here (and unsafe there)
        assert!(expand_text("<image><image>", &[1]).unwrap_err().message.contains("2 <image> placeholders but 1 images"));
        assert!(expand_text("<image>", &[1, 1]).is_err());
        assert!(expand_text("<|placeholder|><image>", &[1]).is_err());
    }

    #[test]
    fn the_ids_are_expanded_the_way_the_expanded_text_tokenizes() {
        let tok = mini_tok("expand");
        let f = OmniFormat::new(TEMPLATE).unwrap();
        let rendered = "<|im_start|>user\nlook <image 1><image> and <image 2><image> a<|im_end|>";
        let counts = [3usize, 2];
        // the library numbers added tokens after the vocabulary, whatever ids the file gives: ask the tokenizer
        let (pad, open, close) =
            (tok.token_to_id("<image>").unwrap(), tok.token_to_id("<img>").unwrap(), tok.token_to_id("</img>").unwrap());
        let ids = tok.encode(rendered).unwrap();
        assert_eq!(ids.iter().filter(|&&t| t == pad).count(), 2, "{ids:?}");
        let (expanded, spans) = f.expand_images(&tok, &ids, &counts).unwrap();
        // the same ids as the text the processor builds
        assert_eq!(expanded, tok.encode(&expand_text(rendered, &counts).unwrap()).unwrap());
        assert_eq!(spans.iter().map(|s| s.tokens).collect::<Vec<_>>(), [3, 2]);
        for s in &spans {
            assert_eq!(expanded[s.start - 1], open, "<img> before the run");
            assert!(expanded[s.start..s.start + s.tokens].iter().all(|&t| t == pad));
            assert_eq!(expanded[s.start + s.tokens], close, "</img> after the run");
        }
        assert_eq!(expanded.len(), ids.len() + counts.iter().sum::<usize>() + 2 * counts.len() - counts.len());
        // text without images is untouched, and a count that does not pair up is refused
        let plain = tok.encode("<|im_start|>user\nhello<|im_end|>").unwrap();
        assert_eq!(f.expand_images(&tok, &plain, &[]).unwrap(), (plain.clone(), vec![]));
        assert!(f.expand_images(&tok, &plain, &[4]).is_err());
        assert!(f.expand_images(&tok, &ids, &[4]).is_err());
    }

    #[test]
    fn the_format_reports_its_image_capabilities() {
        let f = OmniFormat::new(TEMPLATE).unwrap();
        assert!(f.serves_images());
        assert_eq!(f.image_tokens(480, 640), None, "no preprocessing parameters were read");
        let f = f.with_preprocessing(NemotronPreprocessing::default());
        assert_eq!(f.image_tokens(480, 640), Some(Ok(300)));
        assert!(f.image_tokens(8000, 8000).unwrap().unwrap() <= 3328);
        assert!(f.image_tokens(0, 640).unwrap().is_err());
    }
}
