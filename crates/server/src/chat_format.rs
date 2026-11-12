//! The chat format of a model family: everything the server needs to know about how one family talks, behind one trait, found through a
//! registry. The HTTP layer asks a [`ChatFormat`] instead of knowing any family.
//!
//! A format binds the Jinja template's conventions to the rest of the request path: whether the reply starts inside an open `<think>`
//! block and what closes it, how the model writes a tool call (markers, block syntax, and how argument values are typed from the
//! request's schemas), which content parts the template renders as images, and how many placeholder tokens one image takes.
//!
//! # Adding a family
//!
//! 1. Write a module under `formats/` with a type that implements [`ChatFormat`] (only `id`, `render`, `starts_in_reasoning` and
//!    `tool_syntax` are required; media defaults to "none") and a `pub const ENTRY: FormatEntry`.
//! 2. Add one line, `<module>::ENTRY`, to `formats::BUILTIN`.
//!
//! Optionally, 3. declare the family's reasoning capabilities (`reasoning_capabilities`, `chat_template`): see
//! `docs/REASONING_CAPABILITIES.md`.
//!
//! Nothing else changes: `template.rs`, `tools.rs`, `reasoning.rs`, `reasoning_controls.rs`, `vision.rs`, `http.rs` and `main.rs` never name a
//! family. The
//! lookup key is the model folder's `config.json`: `architectures[0]`, else `model_type`, matched exactly against the `names` of the
//! entries; `--chat-format ID` names an entry directly. A model nothing matches is refused with the list of what is registered, never
//! given another family's format.

use crate::api::{self, ApiError};
use crate::template::{ChatTemplate, TemplateError};
use crate::tokenizer::Tok;
use crate::tools::{self, ToolSchemas, ToolSplitter, ToolSyntax};
use crate::vision::{self, ImageRef, Limits, Span};
use serde_json::{Map, Value};
use std::path::Path;

/// A rendered chat prompt and what the reply that follows it starts in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RenderedChat {
    pub text: String,
    /// The prompt ends inside an open `<think>` block, so the reply begins as reasoning and a `</think>` closes it.
    pub starts_in_reasoning: bool,
}

/// One family's conventions, bound to a model folder (its template and whatever else the family reads there).
pub trait ChatFormat: Send + Sync {
    /// The id the family is registered under (`FormatEntry::id`).
    fn id(&self) -> &'static str;
    /// Optional bounded, closed-answer rendering. Unsupported families fail without rendering.
    fn render_decision(
        &self,
        _messages: &Value,
        _kwargs: &Map<String, Value>,
        _limit: usize,
    ) -> Result<RenderedChat, TemplateError> {
        Err(TemplateError::Raised("decisions scoring is not supported by this model family".into()))
    }

    /// Render the conversation (OpenAI-style messages) with the generation prompt, and say what the reply starts in. Implementations use
    /// [`render_normalized`] so tool-call history is renderable (`arguments` as a JSON string becomes an object).
    fn render(&self, messages: &Value, kwargs: &Map<String, Value>) -> Result<RenderedChat, TemplateError>;

    /// HTTP admission requires a renderer that bounds construction. New formats fail closed until they opt in.
    fn render_bounded(
        &self,
        _messages: &Value,
        _kwargs: &Map<String, Value>,
        _max_bytes: usize,
    ) -> Result<RenderedChat, TemplateError> {
        Err(TemplateError::Render("bounded rendering is not implemented for this chat format".into()))
    }

    /// Whether a reply to this rendered prompt begins inside a reasoning block.
    fn starts_in_reasoning(&self, prompt: &str) -> bool;

    /// The text that ends the reasoning block in the reply, when the family declares no [`ReasoningCapabilities`] (the markers of a
    /// declared descriptor win; an inferred descriptor uses this one when the template writes it).
    ///
    /// [`ReasoningCapabilities`]: crate::reasoning_caps::ReasoningCapabilities
    fn reasoning_close(&self) -> &'static str {
        "</think>"
    }

    /// The reasoning/thinking controls this family's template offers, as data (`docs/REASONING_CAPABILITIES.md`): which
    /// `chat_template_kwargs`, effort levels and reasoning budget a request may use, the markers, the template's default. The server
    /// checks it against the template at load (`reasoning_caps::derive`). `None` (the default): inferred from [`ChatFormat::chat_template`]
    /// when the format exposes one, else the server's original policy.
    fn reasoning_capabilities(&self) -> Option<crate::reasoning_caps::ReasoningCapabilities> {
        None
    }

    /// The chat template, for the load-time check of the reasoning capabilities (the variables it reads, its source). `None`: the
    /// format does not expose it, and nothing is checked.
    fn chat_template(&self) -> Option<&ChatTemplate> {
        None
    }

    /// How the family writes tool calls in its reply.
    fn tool_syntax(&self) -> ToolSyntax;

    /// Refuse content parts this server cannot honour, naming the first one: a part that is refused must never reach the template, which
    /// would silently drop the media or render a placeholder for data that is never supplied. The default is a text-only family.
    fn check_media(&self, messages: &Value, _allow_images: bool) -> Result<(), ApiError> {
        api::refuse_media(messages, false)
    }

    /// Whether the template renders `part` as an image placeholder (so the images found in a request and the placeholders rendered are
    /// always the same set, in the same order).
    fn is_image_part(&self, _part: &Value) -> bool {
        false
    }

    /// The token id the template renders once per image, and how it is spelled in text.
    fn image_placeholder(&self, _tok: &Tok) -> Result<(u32, &'static str), String> {
        Err(format!("the `{}` chat format has no image placeholder", self.id()))
    }

    /// Placeholder tokens one image of `height` x `width` pixels takes, when the format can say from the size alone. `None` for families
    /// whose count comes with their pixel preprocessing.
    fn image_tokens(&self, _height: usize, _width: usize) -> Option<Result<usize, String>> {
        None
    }

    /// Whether this server can encode this family's images end to end (pixel preprocessing and the vision tower). When it cannot, the
    /// server refuses to start with image input enabled.
    fn serves_images(&self) -> bool {
        false
    }

    /// The video ingress of this format, when it takes video and it is on (`enable_video`). `None` for every format that does not: the HTTP
    /// layer then never routes a video part, and `check_media` refuses it by name.
    fn video(&self) -> Option<&dyn crate::formats::nemotron_omni::prep::VideoIngress> {
        None
    }

    /// Turn video input on under `options` before the format is shared. A format without video refuses by name; the server calls this only
    /// when vision is on and the family's tower encodes video.
    fn enable_video(&mut self, _options: &VideoOptions) -> Result<(), String> {
        Err(format!("the `{}` chat format takes no video", self.id()))
    }

    /// The streaming splitter for this family's tool-call syntax.
    fn tool_splitter(&self, schemas: ToolSchemas) -> ToolSplitter {
        ToolSplitter::with_syntax(schemas, self.tool_syntax())
    }

    /// Every image part of `messages`, in the order the template renders their placeholders.
    fn extract_images(&self, messages: &Value, limits: &Limits) -> Result<Vec<ImageRef>, ApiError> {
        vision::extract_images_with(messages, limits, |p| self.is_image_part(p))
    }

    /// Expand the one placeholder per image of a tokenized prompt to `counts[i]` tokens for the i-th image, reporting where each run
    /// landed. Refused when the prompt's placeholders and the images do not pair up one to one.
    fn expand_images(&self, tok: &Tok, ids: &[u32], counts: &[usize]) -> Result<(Vec<u32>, Vec<Span>), ApiError> {
        let (token, literal) = self.image_placeholder(tok).map_err(ApiError::bad)?;
        vision::expand_placeholders(ids, token, literal, counts)
    }

    /// [`ChatFormat::expand_images`] for a family whose expansion needs more than a count: each image's encoder output is given (its
    /// `features` are the family's own type, behind `Any`), and the format says where the span the ENGINE sees begins and how long it
    /// is (a family whose image is a frame with markers in it, MiniCPM-V, reports the whole frame, not the run of placeholders).
    /// The default is the count-only expansion with the encoder's own token count and grid, which is what every family before had.
    fn place_images(&self, tok: &Tok, ids: &[u32], encoded: &[EncodedRef<'_>]) -> Result<(Vec<u32>, Vec<Placed>), ApiError> {
        let counts: Vec<usize> = encoded.iter().map(|e| e.tokens).collect();
        let (ids, spans) = self.expand_images(tok, ids, &counts)?;
        let placed =
            spans.iter().zip(encoded).map(|(sp, e)| Placed { start: sp.start, tokens: e.tokens, merged: e.merged }).collect();
        Ok((ids, placed))
    }
}

/// One encoded image as [`ChatFormat::place_images`] sees it: what the encoder returned.
pub struct EncodedRef<'a> {
    /// Placeholder tokens (rows of encoder output).
    pub tokens: usize,
    pub merged: (usize, usize),
    pub features: &'a (dyn std::any::Any + Send + Sync),
}

/// Where an image sits in the expanded prompt, as the engine's `ImageSpan` wants it: its first token, its length in tokens and the
/// `merged` grid (`tokens == merged.0 * merged.1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placed {
    pub start: usize,
    pub tokens: usize,
    pub merged: (usize, usize),
}

/// What a server decides about video input, for the formats that take it (`ChatFormat::enable_video`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VideoOptions {
    /// Frames in one request (two make a tubelet): `--max-video-frames`.
    pub max_frames: usize,
    /// Placeholder rows of one video before pruning: `--max-video-tokens`.
    pub max_tokens: usize,
    /// The share of a video's rows pruned after the first tubelet (the reference model's `video_pruning_rate`); 0 is off:
    /// `--video-pruning-rate`.
    pub pruning_rate: f64,
    /// Refuse, with pruning on, a tile whose tubelet token count is not a perfect square (where the reference model raises), instead of
    /// pruning on the true grid: `--video-evs-strict`.
    pub require_square_grid: bool,
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self { max_frames: 32, max_tokens: 4096, pruning_rate: 0.0, require_square_grid: false }
    }
}

/// Make assistant tool-call history renderable, then render with the generation prompt: the body shared by the formats' `render`.
pub fn render_normalized(
    template: &ChatTemplate,
    messages: &Value,
    kwargs: &Map<String, Value>,
) -> Result<String, TemplateError> {
    let mut messages = messages.clone();
    tools::normalize_messages(&mut messages).map_err(TemplateError::Raised)?;
    template.render(&messages, true, kwargs)
}

/// One registered family.
#[derive(Clone, Copy)]
pub struct FormatEntry {
    /// The id (`--chat-format ID`, the `chat_format` log line): `qwen`, `glm5_next`.
    pub id: &'static str,
    /// The HF names that select it, matched exactly against `architectures[0]` and `model_type` of `config.json`.
    pub names: &'static [&'static str],
    /// Build the format for the model in `dir` (reads `chat_template.jinja` and whatever else the family needs there).
    pub open: fn(dir: &Path) -> Result<Box<dyn ChatFormat>, String>,
}

/// The registered families, looked up by model config or by id.
#[derive(Clone)]
pub struct FormatRegistry {
    entries: Vec<FormatEntry>,
}

impl FormatRegistry {
    /// The formats the server ships (`formats::BUILTIN`).
    pub fn builtin() -> Self {
        Self::new(crate::formats::BUILTIN).expect("the builtin chat formats have distinct ids and names")
    }

    /// A registry of `entries`; an id or a name claimed twice is an error, since a lookup could not tell the two apart.
    pub fn new(entries: &[FormatEntry]) -> Result<Self, String> {
        let mut r = Self { entries: Vec::new() };
        for e in entries {
            r = r.with(*e)?;
        }
        Ok(r)
    }

    /// This registry plus one more family.
    pub fn with(mut self, entry: FormatEntry) -> Result<Self, String> {
        for e in &self.entries {
            if e.id == entry.id {
                return Err(format!("chat format id `{}` is registered twice", entry.id));
            }
            if let Some(n) = entry.names.iter().find(|n| e.names.contains(n)) {
                return Err(format!("`{n}` is claimed by chat formats `{}` and `{}`", e.id, entry.id));
            }
        }
        self.entries.push(entry);
        Ok(self)
    }

    pub fn ids(&self) -> Vec<&'static str> {
        self.entries.iter().map(|e| e.id).collect()
    }

    fn listing(&self) -> String {
        self.entries.iter().map(|e| format!("{} ({})", e.id, e.names.join(", "))).collect::<Vec<_>>().join("; ")
    }

    pub fn by_id(&self, id: &str) -> Result<&FormatEntry, String> {
        self.entries
            .iter()
            .find(|e| e.id == id)
            .ok_or_else(|| format!("unknown chat format `{id}`; registered: {}", self.listing()))
    }

    /// The entry for a model config: `architectures[0]`, else `model_type`.
    pub fn for_config(&self, config: &Value) -> Result<&FormatEntry, String> {
        let arch = config.get("architectures").and_then(|a| a.get(0)).and_then(|a| a.as_str());
        let model_type = config.get("model_type").and_then(|t| t.as_str());
        [arch, model_type]
            .into_iter()
            .flatten()
            .find_map(|name| self.entries.iter().find(|e| e.names.contains(&name)))
            .ok_or_else(|| {
                format!(
                    "no chat format is registered for architecture `{}` (model_type `{}`); registered: {}; \
                     add the family's ChatFormat to `formats::BUILTIN`, or name one with --chat-format ID",
                    arch.unwrap_or("-"),
                    model_type.unwrap_or("-"),
                    self.listing()
                )
            })
    }

    /// The format of the model in `dir`: the entry named by `forced` (an id), else the one its `config.json` selects.
    pub fn open(&self, dir: &Path, forced: Option<&str>) -> Result<Box<dyn ChatFormat>, String> {
        let entry = match forced {
            Some(id) => self.by_id(id)?,
            None => {
                let text = std::fs::read_to_string(dir.join("config.json"))
                    .map_err(|e| format!("config.json: {e} (it selects the chat format; --chat-format ID names one directly)"))?;
                self.for_config(&serde_json::from_str(&text).map_err(|e| format!("config.json: {e}"))?)?
            }
        };
        (entry.open)(dir)
    }
}
