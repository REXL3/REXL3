//! Explicit field/type policy. Unknown output knobs are refused, not silently treated as defaults.
use crate::api::ApiError;
use crate::budget::{too_large, validate_json, BoundedWriter, Limits};
use serde_json::Value;

pub const MAX_TOOLS: usize = 64;
pub const MAX_TOOL_SCHEMA_BYTES: usize = 1 << 20;
pub const MAX_MESSAGES: usize = 10_000;
pub fn bad(field: &str, message: impl Into<String>) -> ApiError {
    ApiError { param: Some(field.into()), ..ApiError::bad(message) }
}
fn large(field: &'static str, message: impl Into<String>) -> ApiError {
    ApiError { param: Some(field.into()), ..too_large(message) }
}
pub fn boolean(body: &Value, key: &'static str, default: bool) -> Result<bool, ApiError> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        _ => Err(bad(key, format!("{key} must be a boolean"))),
    }
}
pub fn non_null<'a>(body: &'a Value, key: &str) -> Option<&'a Value> {
    body.get(key).filter(|v| !v.is_null())
}
pub fn validate_model(body: &Value, served: &str, aliases: &[String]) -> Result<String, ApiError> {
    match body.get("model") {
        None => Ok(served.to_string()),
        Some(Value::String(s)) if s.is_empty() || s.len() > 256 => Err(bad("model", "model must contain 1..256 bytes")),
        Some(Value::String(s)) if s == served || aliases.contains(s) => Ok(s.clone()),
        Some(Value::String(_)) => Err(ApiError {
            code: Some("model_not_found"),
            ..ApiError::status(404, "invalid_request_error", "The requested model does not exist")
        }),
        _ => Err(bad("model", "model must be a string")),
    }
}
/// [`validate_fields_with`] with the OpenAI-compatible top-level reasoning fields.
pub fn validate_fields(body: &Value) -> Result<(), ApiError> {
    let names: Vec<String> = crate::reasoning_controls::TOP_LEVEL_FIELDS.iter().map(|s| s.to_string()).collect();
    validate_fields_with(body, &names)
}

/// Refuse unknown request fields. `reasoning`: the top-level reasoning fields the model's reasoning capabilities know
/// (`reasoning_controls::known_top_level`): they pass here, and the chat path accepts or refuses each by name for this model.
pub fn validate_fields_with(body: &Value, reasoning: &[String]) -> Result<(), ApiError> {
    let obj = body.as_object().ok_or_else(|| bad("request", "request must be a JSON object"))?;
    for field in [
        "parallel_tool_calls",
        "functions",
        "function_call",
        "echo",
        "suffix",
        "best_of",
        "verbosity",
        "audio",
        "modalities",
        "prediction",
        "grammar",
        "guided_json",
        "guided_regex",
    ] {
        if non_null(body, field).is_some() {
            return Err(bad(field, format!("{field} is not supported by this server")));
        }
    }
    if let Some(v) = non_null(body, "response_format") {
        if v.as_object().is_none_or(|o| o.len() != 1 || o.get("type") != Some(&Value::String("text".into()))) {
            return Err(bad(
                "response_format",
                "response_format supports only {\"type\":\"text\"}; json_object and json_schema are not supported",
            ));
        }
    }
    // `logprobs` is accepted only in the shape `prompt_logprobs` implements (a bool or an integer 0..=20): the sampler-side
    // per-generated-token logprobs it otherwise names are not served (see `prompt_logprobs` on /v1/completions).
    if let Some(v) = non_null(body, "logprobs") {
        let ok = match v {
            Value::Bool(_) => true,
            Value::Number(n) => n.as_u64().is_some_and(|k| (k as usize) <= crate::api::MAX_PROMPT_LOGPROBS),
            _ => false,
        };
        if !ok {
            return Err(bad("logprobs", "logprobs must be a boolean or an integer from 0 to 20"));
        }
    }
    if let Some(v) = non_null(body, "logit_bias") {
        if !matches!(v,Value::Object(m) if m.is_empty()) {
            return Err(bad("logit_bias", "logit_bias is not supported by this server (only an empty object is accepted)"));
        }
    }
    if let Some(v) = non_null(body, "stream_options") {
        let o = v.as_object().ok_or_else(|| bad("stream_options", "stream_options must be an object"))?;
        if o.keys().any(|k| k != "include_usage") {
            return Err(bad("stream_options", "only stream_options.include_usage is supported"));
        }
        boolean(v, "include_usage", false).map_err(|mut e| {
            e.param = Some("stream_options".into());
            e
        })?;
    }
    for k in ["stream", "skip_special_tokens", "return_token_ids", "draft", "prefix_cache", "store"] {
        boolean(body, k, false)?;
    }
    for k in ["user", "service_tier"] {
        if non_null(body, k).is_some_and(|v| !v.is_string()) {
            return Err(bad(k, format!("{k} must be a string")));
        }
    }
    if non_null(body, "metadata").is_some_and(|v| !v.is_object()) {
        return Err(bad("metadata", "metadata must be an object"));
    }
    const KNOWN: &[&str] = &[
        "model",
        "messages",
        "prompt",
        "n",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "stream",
        "stream_options",
        "skip_special_tokens",
        "return_token_ids",
        "draft",
        "prefix_cache",
        "tools",
        "tool_choice",
        "chat_template_kwargs",
        "response_format",
        "logprobs",
        "prompt_logprobs",
        "top_logprobs",
        "logit_bias",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "user",
        "metadata",
        "store",
        "service_tier",
    ];
    for k in obj.keys() {
        if k.len() > 256 {
            return Err(bad("request", "request field name exceeds 256 bytes"));
        }
        if !KNOWN.contains(&k.as_str()) && !reasoning.iter().any(|r| r == k) {
            return Err(bad(k, format!("unsupported request field: {k}")));
        }
    }
    Ok(())
}

pub fn validate_completions(body: &Value) -> Result<(), ApiError> {
    let names: Vec<String> = crate::reasoning_controls::TOP_LEVEL_FIELDS.iter().map(|s| s.to_string()).collect();
    validate_completions_with(body, &names)
}

/// [`validate_completions`] refusing by name every top-level reasoning field in `reasoning` (there is no template to give it to).
pub fn validate_completions_with(body: &Value, reasoning: &[String]) -> Result<(), ApiError> {
    for field in ["messages", "tools", "tool_choice", "chat_template_kwargs"] {
        if non_null(body, field).is_some() {
            return Err(bad(field, format!("{field} is not supported by the plain completions endpoint")));
        }
    }
    if boolean(body, "return_token_ids", false)? {
        return Err(bad("return_token_ids", "return_token_ids is supported only by chat completions"));
    }
    crate::reasoning_controls::refuse_top_level(body, reasoning, "by the plain completions endpoint")
}

/// [`validate_chat_with`] under the server's original reasoning policy (`ReasoningCapabilities::legacy`).
pub fn validate_chat(body: &Value, limits: &Limits) -> Result<(), ApiError> {
    static LEGACY: std::sync::OnceLock<crate::reasoning_caps::ReasoningCapabilities> = std::sync::OnceLock::new();
    validate_chat_with(body, limits, LEGACY.get_or_init(crate::reasoning_caps::ReasoningCapabilities::legacy))
}

/// Validate a chat request; its reasoning fields under the model's checked reasoning capabilities (`AppState::reasoning`).
pub fn validate_chat_with(
    body: &Value,
    limits: &Limits,
    caps: &crate::reasoning_caps::ReasoningCapabilities,
) -> Result<(), ApiError> {
    if non_null(body, "prompt").is_some() {
        return Err(bad("prompt", "prompt is not supported by the chat endpoint; use messages"));
    }
    let messages = body
        .get("messages")
        .ok_or_else(|| bad("messages", "messages is required"))?
        .as_array()
        .ok_or_else(|| bad("messages", "messages must be a list"))?;
    if messages.len() > MAX_MESSAGES {
        return Err(large("messages", format!("messages is limited to {MAX_MESSAGES} entries")));
    }
    // Count nested JSON tool-call argument strings too, before normalization allocates Value trees.
    let mut argument_nodes = 0usize;
    for (i, m) in messages.iter().enumerate() {
        if !m.is_object() {
            return Err(bad("messages", format!("messages[{i}] must be an object")));
        }
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("messages", format!("messages[{i}].role must be a string")))?;
        if !["system", "user", "assistant", "tool"].contains(&role) {
            return Err(bad("messages", format!("messages[{i}].role is unsupported")));
        }
        for k in ["name", "tool_call_id", "reasoning_content", "reasoning"] {
            if non_null(m, k).is_some_and(|v| !v.is_string()) {
                return Err(bad("messages", format!("messages[{i}].{k} must be a string")));
            }
        }
        match m.get("content") {
            Some(Value::String(_)) => {}
            None | Some(Value::Null) if role == "assistant" => {}
            Some(Value::Array(parts)) => {
                for (j, p) in parts.iter().enumerate() {
                    if !p.is_object() {
                        return Err(bad("messages", format!("messages[{i}].content[{j}] must be an object")));
                    }
                    let ty = p.get("type").and_then(Value::as_str).unwrap_or("");
                    match ty {
                        "text" if p.get("text").is_some_and(Value::is_string) => {}
                        "image_url" | "image" | "input_image" | "video" | "video_url" | "input_video" | "input_audio" => {} // family/decode remains authoritative
                        "" if p.get("image").is_some() || p.get("image_url").is_some() => {}
                        _ => {
                            return Err(bad("messages", format!("messages[{i}].content[{j}]: unsupported part or invalid text")))
                        }
                    }
                }
            }
            _ => return Err(bad("messages", format!("messages[{i}].content must be text or a list of parts"))),
        }
        if let Some(calls) = non_null(m, "tool_calls") {
            if role != "assistant" {
                return Err(bad("messages", "tool_calls requires an assistant message"));
            }
            let calls = calls.as_array().ok_or_else(|| bad("messages", "tool_calls must be a list"))?;
            if calls.len() > MAX_TOOLS {
                return Err(large("messages", "too many tool calls in one message"));
            }
            for c in calls {
                if !c.is_object() {
                    return Err(bad("messages", "tool call must be an object"));
                }
                let f = c.get("function").unwrap_or(c);
                if !f.get("name").is_some_and(Value::is_string) {
                    return Err(bad("messages", "tool call function.name must be a string"));
                }
                match f.get("arguments") {
                    Some(Value::String(s)) => {
                        if !s.trim_start().starts_with('{') {
                            return Err(bad("messages", "tool call arguments JSON string must contain an object"));
                        }
                        let n = validate_json(s.as_bytes(), limits.json_nodes.saturating_sub(argument_nodes), limits.json_depth)?;
                        argument_nodes =
                            argument_nodes.checked_add(n).ok_or_else(|| large("messages", "tool argument node overflow"))?;
                    }
                    Some(Value::Object(_)) => {}
                    _ => return Err(bad("messages", "tool call arguments must be an object or a JSON object string")),
                }
            }
        }
    }
    validate_tools(body.get("tools"))?;
    crate::reasoning_controls::validate_kwargs(body, caps)
}
pub fn validate_tools(tools: Option<&Value>) -> Result<(), ApiError> {
    let Some(t) = tools.filter(|t| !t.is_null()) else {
        return Ok(());
    };
    let list = t.as_array().ok_or_else(|| bad("tools", "tools must be a list"))?;
    if list.len() > MAX_TOOLS {
        return Err(large("tools", format!("tools is limited to {MAX_TOOLS} entries")));
    }
    let mut w = BoundedWriter::new(MAX_TOOL_SCHEMA_BYTES);
    if serde_json::to_writer(&mut w, t).is_err() {
        return Err(large("tools", format!("serialized tools exceed {MAX_TOOL_SCHEMA_BYTES} bytes")));
    }
    let mut names = std::collections::HashSet::new();
    for t in list {
        if t.get("type").and_then(Value::as_str) != Some("function") {
            return Err(bad("tools", "tools[].type must be function"));
        }
        let f = t.get("function").and_then(Value::as_object).ok_or_else(|| bad("tools", "tools[].function must be an object"))?;
        let name = f
            .get("name")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty() && s.len() <= 64)
            .ok_or_else(|| bad("tools", "function.name must be a non-empty string of at most 64 bytes"))?;
        if !names.insert(name) {
            return Err(bad("tools", "duplicate tool function names are not supported"));
        }
        if f.get("description").is_some_and(|v| !v.is_null() && !v.is_string()) {
            return Err(bad("tools", "function.description must be a string"));
        }
        if f.get("parameters").is_some_and(|v| !v.is_null() && !v.is_object()) {
            return Err(bad("tools", "function.parameters must be an object"));
        }
        // BFCL supplies optional return-schema documentation. It is passed
        // verbatim to the family template, not enforced as an output schema.
        // Existing request node/depth/render and serialized-tool byte bounds
        // cover it; unknown function fields and strict execution stay refused.
        if f.get("response").is_some_and(|v| !v.is_null() && !v.is_object()) {
            return Err(bad("tools", "function.response must be an object"));
        }
        if f.get("strict").is_some_and(|v| !v.is_null() && v != &Value::Bool(false)) {
            return Err(bad("tools", "strict tool schemas are not supported"));
        }
        if t.as_object().is_none_or(|o| o.keys().any(|k| !["type", "function"].contains(&k.as_str())))
            || f.keys().any(|k| !["name", "description", "parameters", "response", "strict"].contains(&k.as_str()))
        {
            return Err(bad("tools", "unsupported tool definition field"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn bfcl_return_schema_is_bounded_prompt_documentation_not_an_output_constraint() {
        let tools = json!([{"type":"function","function":{
            "name":"FileSystemAPI_read_file","description":"Read a simulated file.",
            "parameters":{"type":"object","properties":{"path":{"type":"string"}}},
            "response":{"type":"string","description":"The file contents."}
        }}]);
        validate_tools(Some(&tools)).expect("BFCL return-schema documentation must reach the template unchanged");
        let body = json!({"tools":tools});
        let kwargs = crate::api::chat_template_kwargs(&body).unwrap();
        assert_eq!(kwargs["tools"], body["tools"]);
        assert_eq!(kwargs["tools"][0]["function"]["response"], body["tools"][0]["function"]["response"]);
    }

    #[test]
    fn malformed_return_docs_unknown_fields_and_strict_execution_stay_refused() {
        for response in [json!("not a schema"), json!(42), json!(true), json!([])] {
            let tools = json!([{"type":"function","function":{"name":"f","response":response}}]);
            assert!(validate_tools(Some(&tools)).is_err());
        }
        for extra in [json!({"name":"f","execute":true}), json!({"name":"f","response":{},"strict":true})] {
            let tools = json!([{"type":"function","function":extra}]);
            assert!(validate_tools(Some(&tools)).is_err());
        }
    }
}
