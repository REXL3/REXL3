//! Tool calls for the chat templates: parsing the model's XML-style calls into OpenAI `tool_calls`, and normalizing
//! tool-call history so the template can render it. This file holds the shared machinery (the request's tool schemas, the
//! streaming splitter that finds the tool-call blocks) and the function-block syntax; a family with other markers or another block
//! syntax plugs into the same splitter through [`ToolSyntax`].
//!
//! A model that uses the function-block syntax is told (by its template) to answer with
//!
//! ```text
//! <tool_call>
//! <function=NAME>
//! <parameter=KEY>
//! VALUE
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Parameter values are text; they are typed from the request's tool schema (`integer`, `number`, `boolean`, `object`, `array`,
//! `null`, `string`), falling back to JSON when the parameter is not in the schema.

use serde_json::{json, Map, Value};
use std::collections::HashMap;

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// One parsed call: the function name and its arguments as a JSON object text (the OpenAI `arguments` string).
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
}

/// Parameter types by function and parameter name, from the request's `tools`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ToolSchemas {
    /// The simple `type` of each parameter (the function-block typing rule).
    types: HashMap<String, HashMap<String, String>>,
    /// Each function's whole `parameters` schema (a typing rule that reads unions, enums and `$ref`s uses it).
    params: HashMap<String, Value>,
}

impl ToolSchemas {
    /// Schemas of the request's tools; `None` when the request offers none or sets `tool_choice` to `"none"`.
    pub fn from_request(body: &Value) -> Option<Self> {
        let tools = body.get("tools")?.as_array()?;
        if tools.is_empty() || body.get("tool_choice").and_then(|c| c.as_str()) == Some("none") {
            return None;
        }
        let mut types: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut params: HashMap<String, Value> = HashMap::new();
        for t in tools {
            let f = t.get("function").unwrap_or(t);
            let Some(name) = f.get("name").and_then(|n| n.as_str()) else { continue };
            let props = f.get("parameters").and_then(|p| p.get("properties")).and_then(|p| p.as_object());
            let m = types.entry(name.to_string()).or_default();
            for (k, v) in props.into_iter().flatten() {
                if let Some(ty) = v.get("type").and_then(|t| t.as_str()) {
                    m.insert(k.clone(), ty.to_string());
                }
            }
            if let Some(p) = f.get("parameters") {
                params.insert(name.to_string(), p.clone());
            }
        }
        Some(Self { types, params })
    }

    /// The declared schema of parameter `key` of function `func`, and the function's whole `parameters` schema (the root that a
    /// `$ref` such as `#/$defs/Unit` points into). `None` when the function or the parameter is not declared.
    pub fn resolve_name(&self, func: &str) -> Option<String> {
        if self.types.contains_key(func) || self.params.contains_key(func) {
            return Some(func.to_string());
        }
        let low = func.to_ascii_lowercase();
        let bare = low.rsplit("::").next().unwrap_or(&low);
        for k in self.types.keys().chain(self.params.keys()) {
            let kl = k.to_ascii_lowercase();
            if kl == low || kl == bare {
                return Some(k.clone());
            }
        }
        None
    }

    pub fn param_schema(&self, func: &str, key: &str) -> Option<(&Value, &Value)> {
        let root = self.params.get(func)?;
        Some((root.get("properties")?.get(key)?, root))
    }

    fn typed(&self, func: &str, key: &str, raw: &str) -> Value {
        let ty = self.types.get(func).and_then(|m| m.get(key)).map(String::as_str);
        let parsed = || serde_json::from_str::<Value>(raw.trim()).ok();
        match ty {
            Some("string") => Value::String(raw.to_string()),
            Some("integer") => parsed().filter(|v| v.is_i64() || v.is_u64()).unwrap_or_else(|| Value::String(raw.to_string())),
            Some("number") => parsed().filter(|v| v.is_number()).unwrap_or_else(|| Value::String(raw.to_string())),
            Some("boolean") => match raw.trim().to_ascii_lowercase().as_str() {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                _ => Value::String(raw.to_string()),
            },
            Some("null") => Value::Null,
            Some("object") | Some("array") => parsed().unwrap_or_else(|| Value::String(raw.to_string())),
            // not in the schema (or an exotic type): JSON when it parses to something structured, else the text
            _ => parsed().filter(|v| !v.is_string()).unwrap_or_else(|| Value::String(raw.to_string())),
        }
    }
}

/// Parse the inside of one `<tool_call>` block. `None` when it has no `<function=NAME>`.
pub fn parse_call(block: &str, schemas: &ToolSchemas) -> Option<ToolCall> {
    let at = block.find("<function=")?;
    let rest = &block[at + "<function=".len()..];
    let end = rest.find('>')?;
    let name = rest[..end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let body = &rest[end + 1..];
    let body = body.split("</function>").next().unwrap_or(body);
    let mut args = Map::new();
    let mut cur = body;
    while let Some(p) = cur.find("<parameter=") {
        let after = &cur[p + "<parameter=".len()..];
        let Some(close) = after.find('>') else { break };
        let key = after[..close].trim().to_string();
        let val_start = &after[close + 1..];
        let (val, next) = match val_start.find("</parameter>") {
            Some(e) => (&val_start[..e], &val_start[e + "</parameter>".len()..]),
            None => (val_start, ""), // an unterminated value runs to the end of the block
        };
        // the template puts exactly one newline on each side of the value
        let val = val.strip_prefix('\n').unwrap_or(val);
        let val = val.strip_suffix('\n').unwrap_or(val);
        args.insert(key.clone(), schemas.typed(&name, &key, val));
        cur = next;
    }
    Some(ToolCall { arguments: Value::Object(args).to_string(), name })
}

/// Splits streamed assistant text into visible text and completed tool calls, holding back anything that might still turn
/// into a marker.
pub struct ToolSplitter {
    schemas: ToolSchemas,
    syntax: ToolSyntax,
    buf: String,
    in_call: bool,
    closed: bool,
}

/// A family's parser for the inside of one tool-call block: `None` when it is not a call the family's template would produce.
pub type ParseCall = fn(&str, &ToolSchemas) -> Option<ToolCall>;

/// How a family writes tool calls in its reply: the text that opens and closes one call, and the reader of what is between them. Families
/// may share the `<tool_call>`..`</tool_call>` markers and differ inside, or use other markers.
#[derive(Clone, Copy)]
pub struct ToolSyntax {
    pub open: &'static str,
    pub close: &'static str,
    pub parse: ParseCall,
}

/// The function-block syntax (`<tool_call><function=NAME><parameter=KEY>`).
pub const FUNCTION_BLOCK_TOOL_SYNTAX: ToolSyntax = ToolSyntax { open: OPEN, close: CLOSE, parse: parse_call };

pub const DSML_OPEN: &str = "<｜DSML｜ invoke";
pub const DSML_CLOSE: &str = "</｜DSML｜ invoke>";

/// DeepSeek-V4.1 DSML tool syntax (`<｜DSML｜ calls><｜DSML｜ invoke name="..."><｜DSML｜ parameter name="..." string="true|false">...`).
pub const DSML_TOOL_SYNTAX: ToolSyntax = ToolSyntax { open: DSML_OPEN, close: DSML_CLOSE, parse: parse_dsml_call };

pub fn parse_dsml_call(raw_block: &str, schemas: &ToolSchemas) -> Option<ToolCall> {
    let norm_block = raw_block
        .replace("<parameter ", "<｜DSML｜ parameter ")
        .replace("</parameter>", "</｜DSML｜ parameter>")
        .replace("<｜DSML｜parameter", "<｜DSML｜ parameter")
        .replace("</｜DSML｜parameter>", "</｜DSML｜ parameter>");
    let block = norm_block.as_str();
    let name_pos = block.find("name=\"")?;
    let after_name = &block[name_pos + "name=\"".len()..];
    let name_end = after_name.find('"')?;
    let raw_name = after_name[..name_end].trim();
    if raw_name.is_empty() {
        return None;
    }
    let name = schemas.resolve_name(raw_name)?;
    let close_tag = after_name[name_end..].find('>')?;
    let mut cur = &after_name[name_end + close_tag + 1..];
    let mut args = Map::new();
    while let Some(p) = cur.find("<｜DSML｜") {
        let after_dsml = &cur[p + "<｜DSML｜".len()..];
        let trimmed = after_dsml.trim_start_matches(' ');
        if !trimmed.starts_with("parameter") {
            cur = after_dsml;
            continue;
        }
        let param_rest = &trimmed["parameter".len()..];
        let Some(gt) = param_rest.find('>') else { break };
        let header = &param_rest[..gt];
        let Some(k_pos) = header.find("name=\"") else {
            cur = &param_rest[gt + 1..];
            continue;
        };
        let k_after = &header[k_pos + "name=\"".len()..];
        let Some(k_end) = k_after.find('"') else { break };
        let key = k_after[..k_end].to_string();
        let is_str = if let Some(s_pos) = header.find("string=\"") {
            let s_after = &header[s_pos + "string=\"".len()..];
            s_after.starts_with("true")
        } else {
            true
        };
        let val_start = &param_rest[gt + 1..];
        let (val, next_cur) = if let Some(e1) = val_start.find("</｜DSML｜ parameter>") {
            (&val_start[..e1], &val_start[e1 + "</｜DSML｜ parameter>".len()..])
        } else if let Some(e2) = val_start.find("</｜DSML｜parameter>") {
            (&val_start[..e2], &val_start[e2 + "</｜DSML｜parameter>".len()..])
        } else {
            (val_start, "")
        };
        if !args.contains_key(&key) {
            let typed_val = if is_str {
                schemas.typed(&name, &key, val)
            } else {
                serde_json::from_str::<Value>(val.trim()).unwrap_or_else(|_| schemas.typed(&name, &key, val))
            };
            args.insert(key, typed_val);
        }
        cur = next_cur;
    }
    Some(ToolCall { name, arguments: Value::Object(args).to_string() })
}

/// Length of the longest suffix of `s` that is a proper prefix of `marker`.
fn partial_marker(s: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|&n| s.len() >= n && s.is_char_boundary(s.len() - n) && marker.starts_with(&s[s.len() - n..]))
        .unwrap_or(0)
}

impl ToolSplitter {
    /// A splitter for the function-block syntax (`<function=NAME><parameter=KEY>`).
    pub fn new(schemas: ToolSchemas) -> Self {
        Self::with_syntax(schemas, FUNCTION_BLOCK_TOOL_SYNTAX)
    }

    /// A splitter for a family's own markers and block syntax.
    pub fn with_syntax(schemas: ToolSchemas, syntax: ToolSyntax) -> Self {
        Self { schemas, syntax, buf: String::new(), in_call: false, closed: false }
    }

    /// True once an outer DSML calls container (`</｜DSML｜function_calls>` / `</｜DSML｜ calls>`) has closed.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Feed more text; returns the text that is now known to be plain content and the calls completed by it.
    pub fn push(&mut self, text: &str) -> (String, Vec<ToolCall>) {
        if self.closed {
            return (String::new(), Vec::new());
        }
        let ToolSyntax { open, close, parse } = self.syntax;
        self.buf.push_str(text);
        if open == DSML_OPEN {
            for cw in [
                "</｜DSML｜ calls>",
                "</｜DSML｜calls>",
                "</｜DSML｜ tool_calls>",
                "</｜DSML｜tool_calls>",
                "</｜DSML｜ function_calls>",
                "</｜DSML｜function_calls>",
                "</tool calls>",
                "</function_calls>",
                "</calls>",
            ] {
                if let Some(pos) = self.buf.find(cw) {
                    self.buf.truncate(pos);
                    self.closed = true;
                    break;
                }
            }
            for w in [
                "<｜DSML｜ calls>",
                "</｜DSML｜ calls>",
                "<｜DSML｜calls>",
                "</｜DSML｜calls>",
                "<｜DSML｜ tool_calls>",
                "</｜DSML｜ tool_calls>",
                "<｜DSML｜tool_calls>",
                "</｜DSML｜tool_calls>",
                "<｜DSML｜ function_calls>",
                "</｜DSML｜ function_calls>",
                "<｜DSML｜function_calls>",
                "</｜DSML｜function_calls>",
                "<tool calls>",
                "</tool calls>",
                "<function_calls>",
                "</function_calls>",
                "</call>",
                "</calls>",
            ] {
                if self.buf.contains(w) {
                    self.buf = self.buf.replace(w, "");
                }
            }
            for (from, to) in [
                ("<｜DSML｜invoke", "<｜DSML｜ invoke"),
                ("<invoke ", "<｜DSML｜ invoke "),
                ("<tool ", "<｜DSML｜ invoke "),
                ("</｜DSML｜invoke>", "</｜DSML｜ invoke>"),
                ("</invoke>", "</｜DSML｜ invoke>"),
                ("</tool>", "</｜DSML｜ invoke>"),
                ("<｜DSML｜parameter", "<｜DSML｜ parameter"),
                ("<parameter ", "<｜DSML｜ parameter "),
                ("</｜DSML｜parameter>", "</｜DSML｜ parameter>"),
                ("</parameter>", "</｜DSML｜ parameter>"),
            ] {
                if self.buf.contains(from) {
                    self.buf = self.buf.replace(from, to);
                }
            }
        }
        let (mut out, mut calls) = (String::new(), Vec::new());
        loop {
            if self.in_call {
                let Some(e) = self.buf.find(close) else {
                    if self.closed && self.syntax.open == DSML_OPEN {
                        let call = (self.syntax.parse)(&self.buf, &self.schemas);
                        self.buf.clear();
                        self.in_call = false;
                        if let Some(c) = call {
                            calls.push(c);
                        }
                    }
                    break;
                };
                let block: String = self.buf[..e].to_string();
                self.buf = self.buf[e + close.len()..].to_string();
                self.in_call = false;
                match parse(&block, &self.schemas) {
                    Some(c) => calls.push(c),
                    None => out.push_str(&format!("{open}{block}{close}")), // not a call we understand: keep it visible
                }
            } else if let Some(o) = self.buf.find(open) {
                // trailing whitespace before a call is separator, not content
                out.push_str(self.buf[..o].trim_end());
                self.buf = self.buf[o + open.len()..].to_string();
                self.in_call = true;
            } else {
                // hold back a possible marker prefix and any trailing whitespace (it may precede a marker)
                let mut partial = partial_marker(&self.buf, open);
                if open == DSML_OPEN {
                    for w in [
                        "<｜DSML｜ calls>",
                        "</｜DSML｜ calls>",
                        "<｜DSML｜calls>",
                        "</｜DSML｜calls>",
                        "<｜DSML｜invoke",
                        "<｜DSML｜ tool_calls>",
                        "<｜DSML｜ function_calls>",
                        "<invoke ",
                        "<tool ",
                        "<tool calls>",
                        "</call>",
                        "</calls>",
                        "</invoke>",
                    ] {
                        partial = partial.max(partial_marker(&self.buf, w));
                    }
                }
                let before = &self.buf[..self.buf.len() - partial];
                let hold = partial + (before.len() - before.trim_end().len());
                let cut = self.buf.len() - hold;
                if self.closed {
                    self.buf.clear();
                } else {
                    out.push_str(&self.buf[..cut]);
                    self.buf = self.buf[cut..].to_string();
                }
                break;
            }
        }
        (out, calls)
    }

    /// End of the text: whatever is still held is plain content (an unterminated call is shown as written).
    pub fn finish(&mut self) -> (String, Vec<ToolCall>) {
        if self.closed {
            self.buf.clear();
            return (String::new(), Vec::new());
        }
        let rest = std::mem::take(&mut self.buf);
        if self.in_call {
            self.in_call = false;
            if self.syntax.open == DSML_OPEN {
                if let Some(c) = (self.syntax.parse)(&rest, &self.schemas) {
                    return (String::new(), vec![c]);
                }
            }
            return (format!("{}{rest}", self.syntax.open), Vec::new());
        }
        (rest, Vec::new())
    }
}

/// Make assistant tool-call history renderable by the chat template, which iterates `tool_call.arguments|items`: OpenAI clients send
/// `arguments` as a JSON string. Also turns a null assistant `content` into an empty string.
pub fn normalize_messages(messages: &mut Value) -> Result<(), String> {
    let Some(list) = messages.as_array_mut() else { return Ok(()) };
    for (i, m) in list.iter_mut().enumerate() {
        if m.get("role").and_then(|r| r.as_str()) != Some("assistant") {
            continue;
        }
        if m.get("content").is_some_and(|c| c.is_null()) {
            m["content"] = json!("");
        }
        let Some(calls) = m.get_mut("tool_calls").and_then(|c| c.as_array_mut()) else { continue };
        for (j, c) in calls.iter_mut().enumerate() {
            let f = if c.get("function").is_some() { c.get_mut("function").unwrap() } else { c };
            if let Some(Value::String(s)) = f.get("arguments") {
                let parsed: Value = serde_json::from_str(s)
                    .map_err(|e| format!("messages[{i}].tool_calls[{j}].function.arguments is not JSON: {e}"))?;
                if !parsed.is_object() {
                    return Err(format!("messages[{i}].tool_calls[{j}].function.arguments must be a JSON object"));
                }
                f["arguments"] = parsed;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schemas() -> ToolSchemas {
        ToolSchemas::from_request(&json!({"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"},
            "ratio": {"type": "number"}, "tags": {"type": "array"}, "opts": {"type": "object"}}}}}]}))
        .unwrap()
    }

    const CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=metric>\ntrue\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n</function>\n</tool_call>";

    #[test]
    fn parses_a_call_and_types_parameters_from_the_schema() {
        let c = parse_call(&CALL[OPEN.len()..CALL.len() - CLOSE.len()], &schemas()).unwrap();
        assert_eq!(c.name, "get_weather");
        let a: Value = serde_json::from_str(&c.arguments).unwrap();
        assert_eq!(a, json!({"city": "Paris", "days": 3, "metric": true, "tags": ["a", "b"]}));
    }

    #[test]
    fn a_string_parameter_stays_a_string_even_when_it_looks_like_a_number() {
        let b = "\n<function=get_weather>\n<parameter=city>\n12345\n</parameter>\n</function>\n";
        let a: Value = serde_json::from_str(&parse_call(b, &schemas()).unwrap().arguments).unwrap();
        assert_eq!(a["city"], json!("12345"));
        // an integer parameter that is not an integer is kept as text rather than coerced
        let b = "\n<function=get_weather>\n<parameter=days>\nthree\n</parameter>\n</function>\n";
        assert_eq!(serde_json::from_str::<Value>(&parse_call(b, &schemas()).unwrap().arguments).unwrap()["days"], json!("three"));
    }

    #[test]
    fn multiline_values_keep_their_inner_newlines() {
        let b = "\n<function=f>\n<parameter=code>\nline one\nline two\n</parameter>\n</function>\n";
        let c = parse_call(b, &ToolSchemas::default()).unwrap();
        assert_eq!(serde_json::from_str::<Value>(&c.arguments).unwrap()["code"], json!("line one\nline two"));
    }

    #[test]
    fn splits_text_and_calls_however_the_stream_is_chunked() {
        let full = format!("Let me check.\n\n{CALL}\n\n{CALL}");
        for step in [1usize, 2, 3, 5, 7, 16, 1000] {
            let mut sp = ToolSplitter::new(schemas());
            let (mut text, mut calls) = (String::new(), Vec::new());
            let chars: Vec<char> = full.chars().collect();
            for ch in chars.chunks(step) {
                let (t, c) = sp.push(&ch.iter().collect::<String>());
                text.push_str(&t);
                calls.extend(c);
            }
            let (t, c) = sp.finish();
            text.push_str(&t);
            calls.extend(c);
            assert_eq!(text, "Let me check.", "step {step}");
            assert_eq!(calls.len(), 2, "step {step}");
            assert_eq!(calls[0], calls[1]);
        }
    }

    #[test]
    fn text_that_only_resembles_a_marker_is_not_lost() {
        let mut sp = ToolSplitter::new(schemas());
        let (t, c) = sp.push("a <tool b <tool_c");
        let (t2, c2) = sp.finish();
        assert_eq!(format!("{t}{t2}"), "a <tool b <tool_c");
        assert!(c.is_empty() && c2.is_empty());
    }

    #[test]
    fn an_unterminated_or_nameless_call_stays_visible() {
        let mut sp = ToolSplitter::new(schemas());
        let (t, c) = sp.push("hi <tool_call>\n<function=get_weather>\n<parameter=city>\nPar");
        assert_eq!((t.as_str(), c.len()), ("hi", 0));
        let (t, c) = sp.finish();
        assert!(t.starts_with("<tool_call>") && t.contains("Par") && c.is_empty());
        let mut sp = ToolSplitter::new(schemas());
        let (t, c) = sp.push("<tool_call>\nnot a function\n</tool_call>");
        assert!(t.contains("not a function") && c.is_empty());
    }

    #[test]
    fn schemas_need_tools_and_respect_tool_choice_none() {
        assert!(ToolSchemas::from_request(&json!({})).is_none());
        assert!(ToolSchemas::from_request(&json!({"tools": []})).is_none());
        let t = json!([{"type": "function", "function": {"name": "f", "parameters": {"properties": {}}}}]);
        assert!(ToolSchemas::from_request(&json!({"tools": t, "tool_choice": "none"})).is_none());
        assert!(ToolSchemas::from_request(&json!({"tools": t, "tool_choice": "auto"})).is_some());
    }

    #[test]
    fn history_arguments_become_objects_for_the_template() {
        let mut m = json!([{"role": "user", "content": "hi"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}]);
        normalize_messages(&mut m).unwrap();
        assert_eq!(m[1]["tool_calls"][0]["function"]["arguments"], json!({"a": 1}));
        assert_eq!(m[1]["content"], json!(""));
        let mut bad = json!([{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "not json"}}]}]);
        assert!(normalize_messages(&mut bad).unwrap_err().contains("messages[0].tool_calls[0]"));
    }
}
