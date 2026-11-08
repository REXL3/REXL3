//! The MiMo-V2.6 chat format: `<|im_start|>` framing, `<think>` blocks and Qwen-style
//! `<tool_call><function=NAME><parameter=KEY>` calls — the template shipped in the pack
//! (`chat_template.jinja`, verified 2026-10-04: same structure and tool syntax as Qwen3.x).
//! Vision and audio placeholders exist in the template but the tower is not built (M9): a request that
//! needs them is refused, naming the missing component.

use crate::api::{self, ApiError};
use crate::chat_format::{render_normalized, ChatFormat, FormatEntry, RenderedChat};
use crate::template::{ChatTemplate, TemplateError};
use crate::tokenizer::Tok;
use crate::tools::{ToolSyntax, FUNCTION_BLOCK_TOOL_SYNTAX};
use serde_json::{Map, Value};
use std::path::Path;

pub const ENTRY: FormatEntry = FormatEntry { id: "mimo", names: &["MiMoV2ForCausalLM", "mimo_v2"], open };

fn open(dir: &Path) -> Result<Box<dyn ChatFormat>, String> {
    let src = std::fs::read_to_string(dir.join("chat_template.jinja")).map_err(|e| format!("chat_template.jinja: {e}"))?;
    Ok(Box::new(MiMoFormat::new(&src).map_err(|e| e.to_string())?))
}

pub struct MiMoFormat {
    template: ChatTemplate,
}

impl MiMoFormat {
    pub fn new(template_source: &str) -> Result<Self, TemplateError> {
        Ok(Self { template: ChatTemplate::new(template_source)? })
    }
}

impl ChatFormat for MiMoFormat {
    fn id(&self) -> &'static str {
        ENTRY.id
    }

    fn render(&self, messages: &Value, kwargs: &Map<String, Value>) -> Result<RenderedChat, TemplateError> {
        let text = render_normalized(&self.template, messages, kwargs)?;
        let starts_in_reasoning = self.starts_in_reasoning(&text);
        Ok(RenderedChat { text, starts_in_reasoning })
    }

    /// A thinking prompt ends with `<think>` and not the closed empty block; same rule as Qwen's.
    fn render_bounded(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        max_bytes: usize,
    ) -> Result<RenderedChat, TemplateError> {
        crate::template::with_render_limit(max_bytes, || self.render(messages, kwargs))
    }

    fn starts_in_reasoning(&self, prompt: &str) -> bool {
        prompt.trim_end_matches('\n').ends_with("<think>") && !prompt.ends_with("</think>\n\n")
    }

    fn tool_syntax(&self) -> ToolSyntax {
        FUNCTION_BLOCK_TOOL_SYNTAX
    }

    fn check_media(&self, messages: &Value, allow_images: bool) -> Result<(), ApiError> {
        api::refuse_media(messages, allow_images)
    }

    fn is_image_part(&self, part: &Value) -> bool {
        // text-only until the tower lands (M9): the part types `refuse_media` treats as images
        let ty = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
        ["image_url", "image", "input_image"].contains(&ty) || part.get("image_url").is_some() || part.get("image").is_some()
    }

    fn image_placeholder(&self, _tok: &Tok) -> Result<(u32, &'static str), String> {
        Err("the MiMo vision tower is not built (M9): image requests are refused".to_string())
    }

    fn serves_images(&self) -> bool {
        false
    }
}
