//! The Bailing-3.1 chat format (`Ling-3.1-flash-1019`): `<role>SYSTEM|HUMAN|ASSISTANT|OBSERVATION</role>` turns, `<|role_end|>`
//! stops, `<think>...</think>` reasoning, and `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>` tool calls
//! (the same XML shape as GLM's, parsed by `formats::glm::tools`).

use crate::api::{self, ApiError};
use crate::chat_format::{ChatFormat, FormatEntry, RenderedChat};
use crate::reasoning_caps::{EffortLevel, Markers, ReasoningCapabilities, TemplateVar, ThinkingSwitch};
use crate::template::{ChatTemplate, TemplateError};
use crate::tokenizer::Tok;
use crate::tools::ToolSyntax;
use serde_json::{Map, Value};
use std::path::Path;

pub const ENTRY: FormatEntry =
    FormatEntry { id: "bailing_moe_v3", names: &["BailingMoeV3ForCausalLM", "bailing_hybrid", "bailing_moe_v3"], open };

pub const THINK_OPEN: &str = "<think>";
pub const THINK_END: &str = "</think>";
pub const ROLE_END: &str = "<|role_end|>";

fn open(dir: &Path) -> Result<Box<dyn ChatFormat>, String> {
    let src = std::fs::read_to_string(dir.join("chat_template.jinja")).map_err(|e| format!("chat_template.jinja: {e}"))?;
    Ok(Box::new(BailingFormat::new(&src).map_err(|e| e.to_string())?))
}

pub struct BailingFormat {
    template: ChatTemplate,
}

impl BailingFormat {
    pub fn new(template_source: &str) -> Result<Self, TemplateError> {
        Ok(Self { template: ChatTemplate::new(template_source)? })
    }
}

pub fn reasoning_capabilities() -> ReasoningCapabilities {
    ReasoningCapabilities {
        reasons: true,
        markers: Some(Markers::new(THINK_OPEN, THINK_END)),
        switch: Some(ThinkingSwitch { var: "enable_thinking".into(), closes_block: true }),
        // the bundled jinja defaults to thinking on (`thinking_option = 'on'` unless disabled)
        template_thinks_by_default: true,
        variables: vec![TemplateVar::boolean("enable_thinking").choice()],
        levels: vec![
            EffortLevel::new("on", true).sets("enable_thinking", true),
            EffortLevel::new("off", false).sets("enable_thinking", false),
        ],
        budget: false,
        aliases: vec![crate::reasoning_caps::AliasTable::openai()],
    }
}

impl ChatFormat for BailingFormat {
    fn id(&self) -> &'static str {
        ENTRY.id
    }
    fn render(&self, messages: &Value, kwargs: &Map<String, Value>) -> Result<RenderedChat, TemplateError> {
        if messages.as_array().is_some_and(|m| m.is_empty()) {
            return Err(TemplateError::Raised(
                "Cannot apply chat template to an empty conversation. Provide at least one message.".into(),
            ));
        }
        let text = crate::chat_format::render_normalized(&self.template, messages, kwargs)?;
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

    fn starts_in_reasoning(&self, prompt: &str) -> bool {
        prompt.ends_with(THINK_OPEN)
    }

    fn reasoning_capabilities(&self) -> Option<ReasoningCapabilities> {
        Some(reasoning_capabilities())
    }

    fn tool_syntax(&self) -> ToolSyntax {
        ToolSyntax { open: "<tool_call>", close: "</tool_call>", parse: super::glm::tools::parse_call }
    }

    fn check_media(&self, messages: &Value, allow_images: bool) -> Result<(), ApiError> {
        api::refuse_media(messages, allow_images)
    }

    fn is_image_part(&self, part: &Value) -> bool {
        let obj = part.as_object();
        obj.is_some_and(|o| {
            let ty = o.get("type").and_then(|t| t.as_str()).unwrap_or("");
            matches!(ty, "image" | "image_url" | "input_image") || o.get("image").is_some() || o.get("image_url").is_some()
        })
    }

    fn image_placeholder(&self, _tok: &Tok) -> Result<(u32, &'static str), String> {
        Err("bailing_moe_v3 has no image placeholder".into())
    }

    fn serves_images(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const JINJA: &str = include_str!("../../../../testdata/ling31/chat_template.jinja");

    fn fmt() -> BailingFormat {
        BailingFormat::new(JINJA).unwrap()
    }

    #[test]
    fn renders_roles_thinking_and_tool_calls() {
        let f = fmt();
        let msgs = serde_json::json!([
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": "What is the weather in SF?" },
            { "role": "assistant", "content": "", "reasoning_content": "The user asks about SF weather.", "tool_calls": [
                { "function": { "name": "get_weather", "arguments": { "city": "San Francisco" } } }
            ] },
            { "role": "tool", "tool_call_id": "t1", "content": "72F, sunny" },
            { "role": "user", "content": "Thanks!" },
        ]);
        let mut kwargs = Map::new();
        kwargs.insert("add_generation_prompt".into(), Value::Bool(true));
        kwargs.insert("enable_thinking".into(), Value::Bool(true));
        let r = f.render(&msgs, &kwargs).unwrap();
        let t = &r.text;
        assert!(t.contains("<role>SYSTEM</role>"), "{t}");
        assert!(t.contains("detailed thinking on<|role_end|>"), "{t}");
        assert!(t.contains("<role>HUMAN</role>What is the weather in SF?<|role_end|>"), "{t}");
        assert!(t.contains("<role>ASSISTANT</role>\n<think>The user asks about SF weather.</think>"), "{t}");
        assert!(
            t.contains("<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>San Francisco</arg_value>\n</tool_call>"),
            "{t}"
        );
        assert!(t.contains("<role>OBSERVATION</role>\n<tool_response>\n72F, sunny\n</tool_response><|role_end|>"), "{t}");
        // the generation prompt ends inside an open think block
        assert!(r.starts_in_reasoning, "{t}");
        assert!(t.ends_with("<role>ASSISTANT</role>\n<think>"), "{t}");
        assert!(t.ends_with("<role>ASSISTANT</role>\n<think>"), "got: {t}");
    }

    #[test]
    fn thinking_off_renders_closed_think() {
        let f = fmt();
        let msgs = serde_json::json!([
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": "Hi" },
        ]);
        let mut kwargs = Map::new();
        kwargs.insert("add_generation_prompt".into(), Value::Bool(true));
        kwargs.insert("enable_thinking".into(), Value::Bool(false));
        let r = f.render(&msgs, &kwargs).unwrap();
        assert!(r.text.contains("detailed thinking off<|role_end|>"), "{}", r.text);
        assert!(!r.starts_in_reasoning);
        assert!(r.text.ends_with("<role>ASSISTANT</role>\n<think></think>"), "{}", r.text);
    }
}
