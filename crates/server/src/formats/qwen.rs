//! The Qwen3.x chat format (Qwen3.8-27B today): `<think>\n` generation prompt, `<tool_call><function=NAME><parameter=KEY>` calls,
//! `<|vision_start|><|image_pad|><|vision_end|>` images. This is what the server did before formats existed, unchanged.

use crate::api::{self, ApiError};
use crate::chat_format::{render_normalized, ChatFormat, FormatEntry, RenderedChat};
use crate::template::{ChatTemplate, TemplateError};
use crate::tokenizer::Tok;
use crate::tools::{ToolSyntax, FUNCTION_BLOCK_TOOL_SYNTAX};
use crate::vision;
use serde_json::{Map, Value};
use std::path::Path;

pub const ENTRY: FormatEntry = FormatEntry {
    id: "qwen",
    names: &[
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
    ],
    open,
};

fn open(dir: &Path) -> Result<Box<dyn ChatFormat>, String> {
    let src = std::fs::read_to_string(dir.join("chat_template.jinja")).map_err(|e| format!("chat_template.jinja: {e}"))?;
    Ok(Box::new(QwenFormat::new(&src).map_err(|e| e.to_string())?))
}

pub struct QwenFormat {
    template: ChatTemplate,
}

impl QwenFormat {
    pub fn new(template_source: &str) -> Result<Self, TemplateError> {
        Ok(Self { template: ChatTemplate::new(template_source)? })
    }
}

impl ChatFormat for QwenFormat {
    fn render_decision(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        limit: usize,
    ) -> Result<RenderedChat, TemplateError> {
        let text = self.template.render_decision_bounded(messages, kwargs, limit)?;
        let starts_in_reasoning = self.starts_in_reasoning(&text);
        Ok(RenderedChat { text, starts_in_reasoning })
    }
    fn id(&self) -> &'static str {
        ENTRY.id
    }

    fn render(&self, messages: &Value, kwargs: &Map<String, Value>) -> Result<RenderedChat, TemplateError> {
        let text = render_normalized(&self.template, messages, kwargs)?;
        let starts_in_reasoning = self.starts_in_reasoning(&text);
        Ok(RenderedChat { text, starts_in_reasoning })
    }

    /// A thinking prompt ends with `<think>\n`; a non-thinking one with the closed `<think>\n\n</think>\n\n`.
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
        vision::is_image_part(part)
    }

    fn image_placeholder(&self, tok: &Tok) -> Result<(u32, &'static str), String> {
        let id = tok.token_to_id("<|image_pad|>").ok_or("tokenizer has no image placeholder")?;
        Ok((id, "<|image_pad|>"))
    }

    fn serves_images(&self) -> bool {
        true
    }
}
