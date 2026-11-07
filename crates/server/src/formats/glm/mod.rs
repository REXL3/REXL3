//! The GLM-5.x chat format (GLM-5.3-Flash): `[gMASK]<sop>` prefix, `<|assistant|><think>` generation prompt, tool calls as
//! `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>` ([`tools`]), `<|begin_of_image|><|image|><|end_of_image|>`
//! images whose token count follows from the image size ([`vision`]).

pub mod tools;
pub mod vision;

use crate::api::ApiError;
use crate::chat_format::{render_normalized, ChatFormat, FormatEntry, RenderedChat};
use crate::template::{ChatTemplate, TemplateError};
use crate::tokenizer::Tok;
use crate::tools::ToolSyntax;
use serde_json::{Map, Value};
use std::path::Path;
use vision::GlmImageGeometry;

/// The GLM image placeholder the template renders once per image part, between `<|begin_of_image|>` and `<|end_of_image|>`.
pub const IMAGE_TOKEN: &str = "<|image|>";

pub const ENTRY: FormatEntry = FormatEntry { id: "glm5_next", names: &["Glm5NextForConditionalGeneration", "glm5_next"], open };

fn open(dir: &Path) -> Result<Box<dyn ChatFormat>, String> {
    let src = std::fs::read_to_string(dir.join("chat_template.jinja")).map_err(|e| format!("chat_template.jinja: {e}"))?;
    let mut f = GlmFormat::new(&src).map_err(|e| e.to_string())?;
    if let Ok(t) = std::fs::read_to_string(dir.join("processor_config.json")) {
        let cfg: Value = serde_json::from_str(&t).map_err(|e| format!("processor_config.json: {e}"))?;
        f.image = GlmImageGeometry::from_processor_config(&cfg)?;
    }
    Ok(Box::new(f))
}

pub struct GlmFormat {
    template: ChatTemplate,
    /// The image canvas and token budget (`processor_config.json`; the published pack's values by default).
    pub image: GlmImageGeometry,
}

impl GlmFormat {
    pub fn new(template_source: &str) -> Result<Self, TemplateError> {
        Ok(Self { template: ChatTemplate::new(template_source)?, image: GlmImageGeometry::default() })
    }
}

impl ChatFormat for GlmFormat {
    fn id(&self) -> &'static str {
        ENTRY.id
    }

    fn render(&self, messages: &Value, kwargs: &Map<String, Value>) -> Result<RenderedChat, TemplateError> {
        if messages.as_array().is_some_and(|m| m.is_empty()) {
            // the Qwen template raises for this itself; the GLM template renders a prompt with no turn, and HF refuses before rendering
            return Err(TemplateError::Raised(
                "Cannot apply chat template to an empty conversation. Provide at least one message.".into(),
            ));
        }
        let text = render_normalized(&self.template, messages, kwargs)?;
        let starts_in_reasoning = self.starts_in_reasoning(&text);
        Ok(RenderedChat { text, starts_in_reasoning })
    }

    /// Every GLM prompt ends with `<think>` (no newline), and no template switch closes it.
    fn render_bounded(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        max_bytes: usize,
    ) -> Result<RenderedChat, TemplateError> {
        crate::template::with_render_limit(max_bytes, || self.render(messages, kwargs))
    }

    fn starts_in_reasoning(&self, prompt: &str) -> bool {
        prompt.ends_with("<think>")
    }

    fn tool_syntax(&self) -> ToolSyntax {
        ToolSyntax { open: "<tool_call>", close: "</tool_call>", parse: tools::parse_call }
    }

    fn check_media(&self, messages: &Value, allow_images: bool) -> Result<(), ApiError> {
        check_media(messages, allow_images)
    }

    fn is_image_part(&self, part: &Value) -> bool {
        part_kind(part) == Part::Image
    }

    fn image_placeholder(&self, tok: &Tok) -> Result<(u32, &'static str), String> {
        tok.token_to_id(IMAGE_TOKEN)
            .map(|id| (id, IMAGE_TOKEN))
            .ok_or_else(|| format!("the tokenizer has no {IMAGE_TOKEN} token"))
    }

    fn image_tokens(&self, height: usize, width: usize) -> Option<Result<usize, String>> {
        Some(self.image.tokens(height, width))
    }

    /// The GLM tower (`gb_model::families::glm::vision`) and the family's pixel preprocessing (canvas padding, the fork's normalisation, `gb_vision::glm`)
    /// exist: images are served when the server starts with a vision slot.
    fn serves_images(&self) -> bool {
        true
    }
}

/// What the GLM template does with a content part (`visible_text`, in `chat_template.jinja`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// `type` is `image` or `image_url`: `<|begin_of_image|><|image|><|end_of_image|>`.
    Image,
    /// `type` is `video` or `video_url`: `<|begin_of_video|><|video|><|end_of_video|>`.
    Video,
    /// `type` is `audio`, `audio_url` or `input_audio`: `<|begin_of_audio|><|end_of_audio|>`.
    Audio,
    /// Media by its look (`input_image`, a bare `image` or `video` key, ...) that the template does not recognize and drops without a trace.
    Dropped,
    /// Text, or anything else the template handles as text.
    Other,
}

pub fn part_kind(part: &Value) -> Part {
    let ty = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match ty {
        "image" | "image_url" => Part::Image,
        "video" | "video_url" => Part::Video,
        "audio" | "audio_url" | "input_audio" => Part::Audio,
        "input_image" | "input_video" => Part::Dropped,
        "text" => Part::Other,
        _ if ["image", "image_url", "video", "video_url", "audio", "audio_url", "input_audio"]
            .iter()
            .any(|k| part.get(*k).is_some()) =>
        {
            Part::Dropped
        }
        _ => Part::Other,
    }
}

fn refuse_part(part: &Value, path: &str, in_tool_result: bool, allow_images: bool) -> Result<(), ApiError> {
    let ty = part.get("type").and_then(|t| t.as_str()).unwrap_or("media");
    let refuse = |why: &str| Err(ApiError::bad(format!("{path}: `{ty}` {why}")));
    match part_kind(part) {
        Part::Other => Ok(()),
        Part::Image if !allow_images => {
            refuse("input is not supported by this server (it was started without image support); send text only")
        }
        // the template emits tool results sorted by the order of the calls, so a placeholder in a result would not sit where the image was
        // found in the request: refuse rather than pair an image with the wrong placeholder
        Part::Image if in_tool_result => {
            refuse("input inside a tool result is not supported by this server; send the image in a user message")
        }
        Part::Image => Ok(()),
        Part::Video | Part::Audio => refuse("input is not supported by this server; send text and images only"),
        Part::Dropped => refuse(
            "is not a content part the GLM chat template renders (it would be dropped silently); use `image_url` or `image`",
        ),
    }
}

/// Refuse what the GLM template would drop, pair with the wrong image, or render as a placeholder for data that is never supplied.
pub fn check_media(messages: &Value, allow_images: bool) -> Result<(), ApiError> {
    let Some(list) = messages.as_array() else { return Ok(()) };
    for (i, m) in list.iter().enumerate() {
        let in_tool_result = m.get("role").and_then(|r| r.as_str()) == Some("tool");
        let Some(parts) = m.get("content").and_then(|c| c.as_array()) else { continue };
        for (j, part) in parts.iter().enumerate() {
            refuse_part(part, &format!("messages[{i}].content[{j}]"), in_tool_result, allow_images)?;
            // a tool message may hold `{"tool_call_id": .., "output": [parts]}` entries; the template renders those parts as well
            if in_tool_result {
                for (k, p) in part.get("output").and_then(|o| o.as_array()).into_iter().flatten().enumerate() {
                    refuse_part(p, &format!("messages[{i}].content[{j}].output[{k}]"), true, allow_images)?;
                }
            }
        }
    }
    Ok(())
}
