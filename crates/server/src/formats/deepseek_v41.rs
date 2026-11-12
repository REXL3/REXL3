//! DeepSeek-V4.1-Flash chat format: `<｜begin▁of▁sentence｜>`, `<｜System｜>`, `<｜User｜>`, `<｜Assistant｜>`,
//! `<think>` / `</think>`, `Reasoning Effort: N`, and `<｜DSML｜ calls>` tool calling.

use crate::api::{self, ApiError};
use crate::chat_format::{ChatFormat, FormatEntry, RenderedChat};
use crate::reasoning_caps::{AliasTable, EffortLevel, Markers, ReasoningCapabilities, TemplateVar, ThinkingSwitch};
use crate::template::TemplateError;
use crate::tokenizer::Tok;
use crate::tools::{ToolSyntax, DSML_TOOL_SYNTAX};
use serde_json::{Map, Value};
use std::path::Path;

pub const ENTRY: FormatEntry = FormatEntry {
    id: "deepseek_v41",
    names: &[
        "DeepseekV4ForCausalLM",
        "DeepseekV41ForCausalLM",
        "DeepseekV41ForConditionalGeneration",
        "deepseek_v4",
        "deepseek_v41",
        "deepseek_v41_text",
    ],
    open,
};

pub const BUNDLED_CHAT_TEMPLATE: &str = include_str!("../../../../testdata/deepseek_v41/chat_template.jinja");

const BOS: &str = "<｜begin▁of▁sentence｜>";
const EOS: &str = "<｜end▁of▁sentence｜>";
const THINK_OPEN: &str = "<think>";
const THINK_END: &str = "</think>";
const DSML: &str = "｜DSML｜";
const SYSTEM: &str = "<｜System｜>";
const USER: &str = "<｜User｜>";
const ASSISTANT: &str = "<｜Assistant｜>";
const REMINDER: &str = "<｜latest_reminder｜>";

fn open(_dir: &Path) -> Result<Box<dyn ChatFormat>, String> {
    Ok(Box::new(DeepseekV41Format))
}

pub struct DeepseekV41Format;

pub fn reasoning_capabilities() -> ReasoningCapabilities {
    let think = |name: &str| EffortLevel::new(name, true).sets("enable_thinking", true).sets("reasoning_effort", name);
    ReasoningCapabilities {
        reasons: true,
        markers: Some(Markers::new(THINK_OPEN, THINK_END)),
        switch: Some(ThinkingSwitch { var: "enable_thinking".into(), closes_block: true }),
        template_thinks_by_default: false,
        variables: vec![
            TemplateVar::boolean("preserve_thinking").inert(),
            TemplateVar::one_of("reasoning_effort", &["none", "off", "minimal", "low", "medium", "high", "max", "xhigh"])
                .choice(),
        ],
        levels: vec![
            EffortLevel::new("none", false).sets("enable_thinking", false),
            EffortLevel::new("off", false).sets("enable_thinking", false),
            think("minimal"),
            think("low"),
            think("medium"),
            think("high"),
            think("max"),
            think("xhigh"),
        ],
        budget: true,
        aliases: vec![AliasTable::openai()],
    }
}

fn effort_value(kwargs: &Map<String, Value>) -> i64 {
    match kwargs.get("reasoning_effort") {
        Some(Value::Number(n)) => n.as_i64().unwrap_or(75).clamp(1, 100),
        Some(Value::String(s)) => match s.as_str() {
            "minimal" | "low" => 50,
            "medium" => 65,
            "high" => 75,
            "max" | "xhigh" => 100,
            _ => 75,
        },
        _ => 75,
    }
}

/// The one token an image part renders as (`encoding.py`'s `IMAGE_PLACEHOLDER`; the pack's tokenizer carries it at 129264).
const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";

/// `encoding.py`'s `_is_image_block`: a part that renders as one image placeholder.
fn part_is_image(obj: &Map<String, Value>) -> bool {
    let ty = obj.get("type").and_then(Value::as_str).unwrap_or("");
    matches!(ty, "image_url" | "image" | "input_image") || obj.get("image_url").is_some() || obj.get("image").is_some()
}

fn extract_text(content: Option<&Value>) -> Result<String, TemplateError> {
    match content {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => {
            let mut out = Vec::new();
            for p in parts {
                if let Some(s) = p.as_str() {
                    out.push(s.to_string());
                } else if let Some(obj) = p.as_object() {
                    let ty = obj.get("type").and_then(Value::as_str).unwrap_or("");
                    if matches!(ty, "text" | "input_text" | "output_text") {
                        let text = obj.get("text").and_then(Value::as_str).unwrap_or("");
                        if text.contains(IMAGE_PLACEHOLDER) {
                            // the reference raises: images are separate blocks, never typed into text
                            return Err(TemplateError::Raised(format!(
                                "text part contains the image placeholder '{IMAGE_PLACEHOLDER}': images are separate content parts"
                            )));
                        }
                        out.push(text.to_string());
                    } else if part_is_image(obj) {
                        out.push(IMAGE_PLACEHOLDER.to_string());
                    } else {
                        return Err(TemplateError::Raised(format!("unsupported content part type: {ty}")));
                    }
                }
            }
            Ok(out.join("\n\n"))
        }
        Some(other) => Ok(other.to_string()),
    }
}

fn tool_name(tool: &Map<String, Value>) -> String {
    let mut ns = tool
        .get("namespace")
        .and_then(|v| v.as_str().map(String::from).or_else(|| v.get("name").and_then(Value::as_str).map(String::from)));
    let raw_name = tool.get("name").and_then(Value::as_str).unwrap_or("");
    let name = if let Some((pre, bare)) = raw_name.split_once("::") {
        ns = Some(pre.to_string());
        bare
    } else {
        raw_name
    };
    match ns {
        Some(n) if !n.is_empty() => format!("{n}::{name}"),
        _ => name.to_string(),
    }
}

fn render_tools(tools: &[Value]) -> String {
    let mut schemas = Vec::with_capacity(tools.len());
    for t in tools {
        let mut fn_obj =
            t.get("function").and_then(Value::as_object).cloned().or_else(|| t.as_object().cloned()).unwrap_or_default();
        if let Some(ns) = t.get("namespace") {
            fn_obj.insert("namespace".into(), ns.clone());
        }
        let name = tool_name(&fn_obj);
        if let Some(ns_desc) =
            fn_obj.get("namespace").and_then(Value::as_object).and_then(|o| o.get("description")).and_then(Value::as_str)
        {
            let cur = fn_obj.get("description").and_then(Value::as_str).unwrap_or("");
            fn_obj.insert("description".into(), Value::String(format!("{ns_desc}\n{cur}")));
        }
        fn_obj.remove("namespace");
        fn_obj.insert("name".into(), Value::String(name));
        schemas.push(serde_json::to_string(&Value::Object(fn_obj)).unwrap_or_default());
    }
    let schemas_str = schemas.join("\n");
    format!(
        "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<{DSML} calls>\" block like the following:\n\n<{DSML} calls>\n<{DSML} invoke name=\"$TOOL_NAME\">\n<{DSML} parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{DSML} parameter>\n...\n</{DSML} invoke>\n<{DSML} invoke name=\"$TOOL_NAME2\">\n...\n</{DSML} invoke>\n</{DSML} calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by {THINK_OPEN}), you MUST output your complete reasoning inside {THINK_OPEN}...{THINK_END} BEFORE any tool calls or final response.\n\nOtherwise, output directly after {THINK_END} with tool calls or final response.\n\n### Available Tool Schemas\n\n{schemas_str}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"
    )
}

fn parse_args_map(args: Option<&Value>) -> Map<String, Value> {
    let Some(mut cur) = args.cloned() else { return Map::new() };
    for _ in 0..2 {
        if let Value::String(s) = &cur {
            if s.trim().is_empty() {
                return Map::new();
            }
            if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                cur = parsed;
            } else {
                break;
            }
        }
    }
    match cur {
        Value::Object(m) => m,
        other => {
            let mut m = Map::new();
            m.insert("arguments".into(), other);
            m
        }
    }
}

fn render_calls(calls: &[Value]) -> String {
    let mut invokes = Vec::with_capacity(calls.len());
    for c in calls {
        let f = c.get("function").and_then(Value::as_object).or_else(|| c.as_object());
        let Some(f) = f else { continue };
        let name = tool_name(f);
        let args = parse_args_map(f.get("arguments"));
        let mut params = Vec::with_capacity(args.len());
        for (k, v) in &args {
            let (is_str, val_str) = match v {
                Value::String(s) => ("true", s.clone()),
                other => ("false", serde_json::to_string(other).unwrap_or_default()),
            };
            params.push(format!("<{DSML} parameter name=\"{k}\" string=\"{is_str}\">{val_str}</{DSML} parameter>"));
        }
        invokes.push(format!("<{DSML} invoke name=\"{name}\">\n{}\n</{DSML} invoke>", params.join("\n")));
    }
    format!("\n\n<{DSML} calls>\n{}\n</{DSML} calls>", invokes.join("\n"))
}

#[derive(Clone)]
struct UserBlock {
    is_tool: bool,
    text: String,
    rank: usize,
}

#[derive(Clone)]
struct NormMsg {
    role: String,
    content: String,
    reasoning_content: Option<String>,
    tool_calls: Vec<Value>,
    tools: Vec<Value>,
    blocks: Vec<UserBlock>,
    has_order: bool,
}

fn render_deepseek_v41(messages: &Value, kwargs: &Map<String, Value>) -> Result<RenderedChat, TemplateError> {
    let raw_msgs = messages.as_array().ok_or_else(|| TemplateError::Raised("messages must be an array".into()))?;
    let thinking = kwargs
        .get("enable_thinking")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| kwargs.get("reasoning_effort").and_then(Value::as_str).is_some_and(|s| !matches!(s, "none" | "off")));
    let effort = effort_value(kwargs);
    let req_tools = kwargs.get("tools").and_then(Value::as_array).cloned().unwrap_or_default();

    let mut norm: Vec<NormMsg> = Vec::with_capacity(raw_msgs.len() + 1);
    for m in raw_msgs {
        let obj = m.as_object().ok_or_else(|| TemplateError::Raised("message must be an object".into()))?;
        let mut role = obj.get("role").and_then(Value::as_str).unwrap_or("user").to_string();
        if role == "developer" {
            role = "system".into();
        }
        let content = extract_text(obj.get("content"))?;
        let rc = obj
            .get("reasoning_content")
            .or_else(|| obj.get("reasoning"))
            .and_then(|v| if v.is_null() { None } else { Some(extract_text(Some(v))) })
            .transpose()?;
        let tcs = obj.get("tool_calls").and_then(Value::as_array).cloned().unwrap_or_default();
        let tc_id = obj.get("tool_call_id").and_then(Value::as_str).unwrap_or("").to_string();
        norm.push(NormMsg {
            role,
            content,
            reasoning_content: rc,
            tool_calls: tcs,
            tools: Vec::new(),
            blocks: vec![UserBlock { is_tool: false, text: tc_id, rank: 0 }],
            has_order: false,
        });
    }

    if !req_tools.is_empty() {
        if norm.is_empty() || norm[0].role != "system" {
            norm.insert(
                0,
                NormMsg {
                    role: "system".into(),
                    content: String::new(),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                    tools: req_tools.clone(),
                    blocks: Vec::new(),
                    has_order: false,
                },
            );
        } else {
            norm[0].tools = req_tools.clone();
        }
    }

    // Merge consecutive user and tool messages into user turns with ordered <tool_result> blocks
    let mut merged: Vec<NormMsg> = Vec::with_capacity(norm.len());
    let mut order: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in norm {
        if m.role == "assistant" && !m.tool_calls.is_empty() {
            order.clear();
            for (k, tc) in m.tool_calls.iter().enumerate() {
                let id =
                    tc.get("id").or_else(|| tc.get("function").and_then(|f| f.get("id"))).and_then(Value::as_str).unwrap_or("");
                if !id.is_empty() {
                    order.insert(id.to_string(), k);
                }
            }
        }
        if m.role == "tool" || m.role == "user" {
            let is_tool = m.role == "tool";
            let tc_id = m.blocks.first().map(|b| b.text.as_str()).unwrap_or("");
            let rank = if is_tool { order.get(tc_id).copied().unwrap_or(0) } else { 0 };
            let blk = UserBlock { is_tool, text: m.content.clone(), rank };
            if let Some(last) = merged.last_mut().filter(|l| l.role == "user") {
                last.blocks.push(blk);
            } else {
                merged.push(NormMsg {
                    role: "user".into(),
                    content: String::new(),
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                    tools: Vec::new(),
                    blocks: vec![blk],
                    has_order: !order.is_empty(),
                });
            }
        } else {
            merged.push(m);
        }
    }

    for m in &mut merged {
        if m.role == "user" && m.has_order {
            let mut tool_blks: Vec<UserBlock> = m.blocks.iter().filter(|b| b.is_tool).cloned().collect();
            if tool_blks.len() > 1 {
                tool_blks.sort_by_key(|b| b.rank);
                let mut it = tool_blks.into_iter();
                for b in &mut m.blocks {
                    if b.is_tool {
                        if let Some(next_t) = it.next() {
                            *b = next_t;
                        }
                    }
                }
            }
        }
    }

    let has_tools = merged.iter().any(|m| !m.tools.is_empty());
    let drop_thinking = !has_tools;
    let last_user: isize = merged
        .iter()
        .enumerate()
        .rev()
        .find(|&(idx, m)| m.role == "user" || (m.role == "system" && idx > 0))
        .map(|(idx, _)| idx as isize)
        .unwrap_or(-1);

    let mut out = String::from(BOS);
    for (idx, m) in merged.iter().enumerate() {
        let idx_isize = idx as isize;
        let effort_text = if idx == 0 && thinking {
            format!("Reasoning Effort: {effort} (range 1-100, the higher the value, the more thorough the reasoning)\n\n")
        } else {
            String::new()
        };
        if idx == 0 && (!effort_text.is_empty() || m.role == "system") {
            out.push_str(SYSTEM);
        }
        out.push_str(&effort_text);
        match m.role.as_str() {
            "system" => {
                if idx > 0 {
                    out.push_str(SYSTEM);
                }
                out.push_str(&m.content);
                if !m.tools.is_empty() {
                    out.push_str("\n\n");
                    out.push_str(&render_tools(&m.tools));
                }
            }
            "user" => {
                out.push_str(USER);
                let parts: Vec<String> = m
                    .blocks
                    .iter()
                    .map(|b| if b.is_tool { format!("<tool_result>{}</tool_result>", b.text) } else { b.text.clone() })
                    .collect();
                out.push_str(&parts.join("\n\n"));
            }
            "latest_reminder" => {
                out.push_str(REMINDER);
                out.push_str(&m.content);
            }
            "assistant" => {
                if thinking && (!drop_thinking || idx_isize > last_user) {
                    let rc = m.reasoning_content.as_deref().unwrap_or("");
                    out.push_str(rc);
                    out.push_str(THINK_END);
                }
                out.push_str(&m.content);
                if !m.tool_calls.is_empty() {
                    out.push_str(&render_calls(&m.tool_calls));
                }
                out.push_str(EOS);
            }
            _ => {}
        }
        let nxt = merged.get(idx + 1).map(|n| n.role.as_str());
        if let Some(nr) = nxt {
            if nr != "assistant" && nr != "latest_reminder" {
                continue;
            }
        }
        if m.role == "user" || (m.role == "system" && idx > 0) {
            out.push_str(ASSISTANT);
            if thinking && (!drop_thinking || idx_isize >= last_user) {
                out.push_str(THINK_OPEN);
            } else {
                out.push_str(THINK_END);
            }
        }
    }
    let starts_in_reasoning = out.ends_with(THINK_OPEN);
    Ok(RenderedChat { text: out, starts_in_reasoning })
}

impl ChatFormat for DeepseekV41Format {
    fn id(&self) -> &'static str {
        ENTRY.id
    }

    fn render(&self, messages: &Value, kwargs: &Map<String, Value>) -> Result<RenderedChat, TemplateError> {
        render_deepseek_v41(messages, kwargs)
    }

    fn render_bounded(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        max_bytes: usize,
    ) -> Result<RenderedChat, TemplateError> {
        let res = render_deepseek_v41(messages, kwargs)?;
        if res.text.len() > max_bytes {
            return Err(TemplateError::Render(format!("rendered prompt exceeds {max_bytes} bytes")));
        }
        Ok(res)
    }

    fn starts_in_reasoning(&self, prompt: &str) -> bool {
        prompt.ends_with(THINK_OPEN)
    }

    fn reasoning_capabilities(&self) -> Option<ReasoningCapabilities> {
        Some(reasoning_capabilities())
    }

    fn tool_syntax(&self) -> ToolSyntax {
        DSML_TOOL_SYNTAX
    }

    fn check_media(&self, messages: &Value, allow_images: bool) -> Result<(), ApiError> {
        api::refuse_media(messages, allow_images)
    }

    fn is_image_part(&self, part: &Value) -> bool {
        part.as_object().is_some_and(part_is_image)
    }

    fn image_placeholder(&self, tok: &Tok) -> Result<(u32, &'static str), String> {
        let id = tok.token_to_id(IMAGE_PLACEHOLDER).ok_or("the tokenizer has no <｜deepseek_image｜> image placeholder")?;
        Ok((id, IMAGE_PLACEHOLDER))
    }

    /// The span's token count comes with the pixel preprocessing (the aligner grid), not the size alone.
    fn image_tokens(&self, _height: usize, _width: usize) -> Option<Result<usize, String>> {
        None
    }

    fn serves_images(&self) -> bool {
        true
    }
}
