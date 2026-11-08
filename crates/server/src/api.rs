//! Request parsing and response shaping for the OpenAI-compatible endpoints.
//!
//! Requests are read as JSON and checked field by field. Unknown/output-changing unsupported fields are refused explicitly rather than
//! silently changing the request; the documented bookkeeping fields remain accepted-and-ignored.

use gb_sampling::Sampling;
use serde_json::{json, Map, Value};

/// Sampling defaults when a request leaves a field out: the model's `generation_config.json`.
#[derive(Debug, Clone)]
pub struct GenDefaults {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
}

impl Default for GenDefaults {
    fn default() -> Self {
        Self { temperature: 1.0, top_p: 0.95, top_k: 20 }
    }
}

impl GenDefaults {
    pub fn from_dir(dir: &std::path::Path) -> Self {
        let mut d = Self::default();
        if let Ok(t) = std::fs::read_to_string(dir.join("generation_config.json")) {
            if let Ok(v) = serde_json::from_str::<Value>(&t) {
                d.temperature = v.get("temperature").and_then(|x| x.as_f64()).unwrap_or(d.temperature);
                d.top_p = v.get("top_p").and_then(|x| x.as_f64()).unwrap_or(d.top_p);
                d.top_k = v.get("top_k").and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(d.top_k);
            }
        }
        d
    }
}

/// Stop strings a request may carry, and the longest of them in bytes. The reply task searches the held-back text for every stop string after every token, on
/// an async worker: 700,000 strings of 90 bytes (a 62 MiB body) kept one worker busy for 6.8 s of a 64-token reply, and 32 such requests stalled `/health`
/// for a second. OpenAI allows 4; real clients send a handful.
pub const MAX_STOP_STRINGS: usize = 16;
pub const MAX_STOP_BYTES: usize = 1024;

/// `top_k` of 0 or -1 means "no cutoff" in the OpenAI-style APIs. The engine's candidate window is bounded, so such a
/// request gets this many candidates (top_p and min_p still apply inside it).
pub const TOP_K_CAP: usize = 56;

#[derive(Debug, Clone)]
pub struct Params {
    pub sampling: Sampling,
    pub max_tokens: Option<usize>,
    pub stop: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
    pub skip_special: bool,
    pub seed: Option<u64>,
    pub return_token_ids: bool,
    /// `"draft": false` asks for the serial reference path (no speculation).
    pub draft: bool,
    /// `"prefix_cache": false` opts this request out of the prefix cache (nothing is looked up, nothing is stored): the uncached reference path.
    pub prefix_cache: bool,
    /// `prompt_logprobs: k` (0 = off, at most [`MAX_PROMPT_LOGPROBS`]): a teacher-forced pass over the prompt that answers, for
    /// each prompt token after the first, its exact full-vocabulary log-softmax given the tokens before it and that position's
    /// top-`k` token ids and values (vLLM's convention: entry 0 is null, entry `i` scores token `i` from the logits after
    /// tokens `0..i`), and samples nothing. The correctness oracle's readout: it is the model's own head, not the sampler's
    /// candidates.
    pub prompt_logprobs: usize,
}

/// OpenAI's ceiling for `logprobs`/`prompt_logprobs`/`top_logprobs`, and this server's.
pub const MAX_PROMPT_LOGPROBS: usize = 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
    /// Seconds for a `Retry-After` header (429 and the 503s a client should retry).
    pub retry_after: Option<u32>,
    /// OpenAI's machine-readable `error.code` where one is defined (`context_length_exceeded`); `body()` falls back to the HTTP status
    /// number for the others, as it always has.
    pub code: Option<&'static str>,
    /// The request field the error is about (`messages` for a chat prompt, `prompt` for a completion one), reported when set.
    pub param: Option<String>,
}

/// The `code` OpenAI clients compact their history on (and the one TensorFold sends for every prompt past the window).
pub const CONTEXT_LENGTH_EXCEEDED: &str = "context_length_exceeded";

/// Media content parts. Image parts need the vision path (the chat template renders each as one `<|image_pad|>` token that the
/// request handler expands and fills with encoder output); video and audio are never supported. A part that is refused must
/// never reach the template, which would silently drop the media.
const IMAGE_PART_TYPES: [&str; 3] = ["image_url", "image", "input_image"];
const UNSUPPORTED_PART_TYPES: [&str; 4] = ["video", "video_url", "input_video", "input_audio"];

/// Refuse a chat request whose message contents carry video or audio parts (and image parts too unless `allow_images`), naming the
/// first one found.
pub fn refuse_media(messages: &Value, allow_images: bool) -> Result<(), ApiError> {
    let Some(list) = messages.as_array() else { return Ok(()) };
    for (i, m) in list.iter().enumerate() {
        let Some(parts) = m.get("content").and_then(|c| c.as_array()) else { continue };
        for (j, part) in parts.iter().enumerate() {
            let ty = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
            let image = IMAGE_PART_TYPES.contains(&ty) || part.get("image_url").is_some() || part.get("image").is_some();
            let other = UNSUPPORTED_PART_TYPES.contains(&ty)
                || ["video", "video_url", "input_audio"].iter().any(|k| part.get(*k).is_some());
            if other || (image && !allow_images) {
                return Err(ApiError::bad(format!(
                    "messages[{i}].content[{j}]: `{}` input is not supported by this server{}; send text{}",
                    if ty.is_empty() { "media" } else { ty },
                    if image { " (it was started without image support)" } else { "" },
                    if image { " only" } else { " and images only" }
                )));
            }
        }
    }
    Ok(())
}

impl ApiError {
    pub fn bad(message: impl Into<String>) -> Self {
        Self { status: 400, kind: "invalid_request_error", message: message.into(), retry_after: None, code: None, param: None }
    }
    /// A 5xx (or 429) with a machine-readable `type`.
    pub fn status(status: u16, kind: &'static str, message: impl Into<String>) -> Self {
        Self { status, kind, message: message.into(), retry_after: None, code: None, param: None }
    }
    /// A prompt that cannot fit the context window: 400 with OpenAI's `context_length_exceeded` code, naming `messages` as the field
    /// that is too long (the chat endpoint's; see [`ApiError::for_field`]). `message` begins "This model's maximum context length is
    /// N tokens".
    pub fn context_length(message: impl Into<String>) -> Self {
        Self { code: Some(CONTEXT_LENGTH_EXCEEDED), param: Some("messages".into()), ..Self::bad(message) }
    }
    /// The same error for an endpoint whose prompt arrives in another field (`/v1/completions` reads `prompt`): an error that names no
    /// field is left alone.
    pub fn for_field(mut self, field: &'static str) -> Self {
        if self.param.is_some() {
            self.param = Some(field.into());
        }
        self
    }
    /// A machine-readable `code` (the body's `code` is the status otherwise).
    pub fn with_code(mut self, code: &'static str) -> Self {
        self.code = Some(code);
        self
    }
    pub fn retry_after(mut self, secs: u32) -> Self {
        self.retry_after = Some(secs);
        self
    }
    pub fn body(&self) -> Value {
        let code = match self.code {
            Some(c) => json!(c),
            None => json!(self.status),
        };
        let mut error = json!({"message": self.message, "type": self.kind, "code": code});
        if let Some(p) = &self.param {
            error["param"] = json!(p);
        }
        json!({ "error": error })
    }
}

fn num(body: &Value, key: &str) -> Result<Option<f64>, ApiError> {
    match body.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_f64().map(Some).ok_or_else(|| ApiError::bad(format!("{key} must be a number"))),
    }
}

fn zero_or_absent(body: &Value, key: &str) -> Result<(), ApiError> {
    match num(body, key)? {
        None => Ok(()),
        Some(0.0) => Ok(()),
        Some(_) => Err(ApiError::bad(format!("{key} is not supported by this server"))),
    }
}

/// Read the sampling and delivery options common to chat and plain completions.
pub fn parse_params(body: &Value, d: &GenDefaults) -> Result<Params, ApiError> {
    let names: Vec<String> = crate::reasoning_controls::TOP_LEVEL_FIELDS.iter().map(|s| s.to_string()).collect();
    parse_params_with(body, d, &names)
}

/// [`parse_params`] for a model whose reasoning capabilities know the top-level fields `reasoning`
/// (`reasoning_controls::known_top_level`).
pub fn parse_params_with(body: &Value, d: &GenDefaults, reasoning: &[String]) -> Result<Params, ApiError> {
    crate::field_policy::validate_fields_with(body, reasoning)?;
    // `n` is 1 or absent. Anything else is refused: `2.5`, `"3"`, `-1`, `true` and `0` used to be read as "not a number above 1" and answered with one
    // choice, as if the client had asked for 1
    match body.get("n") {
        None | Some(Value::Null) => {}
        Some(v) if v.as_f64() == Some(1.0) => {}
        Some(v) if v.as_u64().is_some_and(|n| n > 1) => return Err(ApiError::bad("n > 1 is not supported by this server")),
        Some(_) => return Err(ApiError::bad("n must be 1 (this server answers one choice per request)")),
    }
    if matches!(body.get("logit_bias"), Some(Value::Object(m)) if !m.is_empty()) {
        return Err(ApiError::bad("logit_bias is not supported by this server"));
    }
    for k in ["presence_penalty", "frequency_penalty"] {
        zero_or_absent(body, k)?;
    }
    if let Some(r) = num(body, "repetition_penalty")? {
        if r != 1.0 {
            return Err(ApiError::bad("repetition_penalty is not supported by this server"));
        }
    }
    let temperature = num(body, "temperature")?.unwrap_or(d.temperature);
    let top_p = num(body, "top_p")?.unwrap_or(d.top_p);
    let min_p = num(body, "min_p")?.unwrap_or(0.0);
    let top_k = match num(body, "top_k")? {
        None => d.top_k,
        Some(k) if k.fract() != 0.0 => return Err(crate::field_policy::bad("top_k", "top_k must be an integer")),
        Some(k) if k <= 0.0 => TOP_K_CAP,
        Some(k) => (k as usize).min(TOP_K_CAP),
    };
    let seed = match body.get("seed") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_u64().ok_or_else(|| ApiError::bad("seed must be a non-negative integer"))? & ((1u64 << 63) - 1)),
    };
    let sampling = Sampling { seed: seed.unwrap_or(0), temperature, top_k, top_p, min_p };
    sampling.validate().map_err(|e| ApiError::bad(e.to_string()))?;
    // Validate both aliases even if one wins. Huge integral counts are clipped later to the actual context room.
    let count = |k: &'static str| -> Result<Option<usize>, ApiError> {
        match num(body, k)? {
            Some(v) if v < 1.0 || v.fract() != 0.0 => Err(crate::field_policy::bad(k, format!("{k} must be a positive integer"))),
            Some(v) => Ok(Some(v as usize)),
            None => Ok(None),
        }
    };
    let completion_count = count("max_completion_tokens")?;
    let legacy_count = count("max_tokens")?;
    let max_tokens = completion_count.or(legacy_count);
    let stop = match body.get("stop") {
        None | Some(Value::Null) => vec![],
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| x.as_str().map(str::to_string).ok_or_else(|| ApiError::bad("stop must be strings")))
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(ApiError::bad("stop must be a string or a list of strings")),
    };
    let stop: Vec<String> = stop.into_iter().filter(|s| !s.is_empty()).collect();
    if stop.len() > MAX_STOP_STRINGS {
        return Err(ApiError::bad(format!("stop takes at most {MAX_STOP_STRINGS} strings, got {}", stop.len())));
    }
    if let Some(s) = stop.iter().find(|s| s.len() > MAX_STOP_BYTES) {
        return Err(ApiError::bad(format!("a stop string is at most {MAX_STOP_BYTES} bytes, got one of {}", s.len())));
    }
    let stream = crate::field_policy::boolean(body, "stream", false)?;
    let include_usage = crate::field_policy::boolean(body.get("stream_options").unwrap_or(&Value::Null), "include_usage", false)?;
    let skip_special = crate::field_policy::boolean(body, "skip_special_tokens", true)?;
    let return_token_ids = crate::field_policy::boolean(body, "return_token_ids", false)?;
    let draft = crate::field_policy::boolean(body, "draft", true)?;
    let prefix_cache = crate::field_policy::boolean(body, "prefix_cache", true)?;
    // `prompt_logprobs` is the oracle readout; `top_logprobs` and `logprobs` name the same k for OpenAI clients that set them
    // together (`logprobs: true` alone means k = 1). The sampler-side logprobs of GENERATED tokens are not served.
    let k_of = |v: &Value| -> Option<usize> {
        match v {
            Value::Bool(true) => Some(1),
            Value::Bool(false) | Value::Null => None,
            Value::Number(n) => n.as_u64().filter(|&k| (k as usize) <= MAX_PROMPT_LOGPROBS).map(|k| k as usize),
            _ => None,
        }
    };
    let prompt_logprobs = match crate::field_policy::non_null(body, "prompt_logprobs") {
        Some(v) => match k_of(v) {
            Some(k) => k,
            None => {
                return Err(crate::field_policy::bad(
                    "prompt_logprobs",
                    format!("prompt_logprobs must be an integer from 0 to {MAX_PROMPT_LOGPROBS}"),
                ))
            }
        },
        None => match crate::field_policy::non_null(body, "top_logprobs") {
            Some(v) => match k_of(v) {
                Some(k) => k,
                None => {
                    return Err(crate::field_policy::bad(
                        "top_logprobs",
                        format!("top_logprobs must be an integer from 0 to {MAX_PROMPT_LOGPROBS}"),
                    ))
                }
            },
            None => crate::field_policy::non_null(body, "logprobs").and_then(k_of).unwrap_or(0),
        },
    };
    Ok(Params {
        sampling,
        max_tokens,
        stop,
        stream,
        include_usage,
        skip_special,
        seed,
        return_token_ids,
        draft,
        prefix_cache,
        prompt_logprobs,
    })
}

/// The kwargs of a chat request under the server's original policy (`reasoning_caps::ReasoningCapabilities::legacy` with the default
/// `--default-thinking off`): [`chat_template_kwargs_base`] plus `enable_thinking: false` when neither it nor `reasoning_effort` is given.
pub fn chat_template_kwargs(body: &Value) -> Result<Map<String, Value>, ApiError> {
    let mut kw = chat_template_kwargs_base(body)?;
    // Real-work default: an omitted effort does not open a think block. Explicit reasoning_effort or enable_thinking still wins.
    if !kw.contains_key("enable_thinking") && !kw.contains_key("reasoning_effort") {
        kw.insert("enable_thinking".into(), Value::Bool(false));
    }
    Ok(kw)
}

/// The request's own `chat_template_kwargs` with its `tools` added (unless `tool_choice` is `"none"`), `tool_choice` checked; no
/// thinking default (that is `reasoning_controls::resolve`'s).
pub fn chat_template_kwargs_base(body: &Value) -> Result<Map<String, Value>, ApiError> {
    crate::field_policy::validate_tools(body.get("tools"))?;
    let mut kw = match body.get("chat_template_kwargs") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(m)) => m.clone(),
        Some(_) => return Err(ApiError::bad("chat_template_kwargs must be an object")),
    };
    match body.get("tool_choice") {
        None | Some(Value::Null) => {}
        Some(Value::String(c)) if c == "auto" || c == "none" => {}
        Some(_) => {
            return Err(ApiError::bad(
                "tool_choice must be \"auto\" or \"none\"; forcing a specific or any tool (\"required\" or a function object) is not supported by this server",
            ))
        }
    }
    let tools_off = body.get("tool_choice").and_then(|c| c.as_str()) == Some("none");
    if let Some(t) = body.get("tools") {
        if !t.is_null() && !tools_off {
            kw.insert("tools".into(), t.clone());
        }
    }
    Ok(kw)
}

/// The token ids of a `/v1/completions` prompt given as an array: a flat list of integers, or a list that holds exactly one such list (OpenAI's batch form
/// with a single prompt). `None` when it is empty or any element is not an integer in `0..=u32::MAX`: such an element used to be cut to 32 bits (an id of
/// 2^32 + 5 became token 5) or silently dropped (`[[1, "a", 2.5]]` became `[1]`), so the model answered a prompt the client did not send.
pub fn token_prompt(a: &[Value]) -> Option<Vec<u32>> {
    let flat = |a: &[Value]| -> Option<Vec<u32>> { a.iter().map(|x| x.as_u64().and_then(|v| u32::try_from(v).ok())).collect() };
    let ids = match a {
        [Value::Array(inner)] => flat(inner)?,
        _ => flat(a)?,
    };
    (!ids.is_empty()).then_some(ids)
}

pub fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub fn finish_str(f: Option<gb_engine::FinishReason>, stop_string: bool) -> &'static str {
    match (f, stop_string) {
        (_, true) => "stop",
        (Some(gb_engine::FinishReason::Length), _) => "length",
        _ => "stop",
    }
}

/// A reply that ended after calling tools finishes as `"tool_calls"`; one cut short by the length limit stays `"length"`.
pub fn finish_with_calls(finish: &'static str, calls: usize) -> &'static str {
    if finish == "stop" && calls > 0 {
        "tool_calls"
    } else {
        finish
    }
}

/// The `usage` block of a reply. `cached`: prompt tokens a prefix-cache hit supplied (they were not prefilled for this request); `reasoning`: the
/// completion tokens spent thinking, through the closing marker (see `reasoning::ReasoningCount`).
pub fn usage_json(
    prompt: usize,
    completion: usize,
    drafted: u64,
    accepted: u64,
    rounds: u64,
    cached: usize,
    reasoning: usize,
) -> Value {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        // OpenAI's detail objects: clients and dashboards read them. `cached_tokens` is the prompt tokens a prefix-cache hit supplied (they were
        // not prefilled for this request); `reasoning_tokens` is the completion tokens spent thinking
        "prompt_tokens_details": {"cached_tokens": cached},
        "completion_tokens_details": {"reasoning_tokens": reasoning},
        // speculation counters, so a client can prove which decoding path ran (tools/gates.py reads these)
        "spec": {"drafted": drafted, "accepted": accepted, "rounds": rounds},
    })
}

#[cfg(test)]
mod media_tests {
    use super::*;

    #[test]
    fn text_parts_pass_and_media_parts_are_refused_by_name() {
        let ok = json!([{"role": "user", "content": "hi"}, {"role": "user", "content": [{"type": "text", "text": "hi"}]}]);
        assert!(refuse_media(&ok, false).is_ok());
        let img = json!([{"role": "user", "content": [{"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]);
        assert!(refuse_media(&img, true).is_ok(), "images pass when the server can encode them");
        let e = refuse_media(&img, false).unwrap_err();
        assert_eq!(e.status, 400);
        assert!(e.message.contains("messages[0].content[1]") && e.message.contains("image_url"), "{}", e.message);
        // a part identified only by its key (the template accepts `image` / `image_url` keys without a type)
        let keyed = json!([{"role": "user", "content": [{"image": "x"}]}]);
        assert!(refuse_media(&keyed, false).is_err());
        assert!(
            refuse_media(&json!([{"role": "user", "content": [{"type": "video", "video": "x"}]}]), true).is_err(),
            "video is refused either way"
        );
        assert!(refuse_media(&json!("not a list"), false).is_ok(), "shape errors belong to the caller");
    }
}

#[cfg(test)]
mod error_body_tests {
    use super::*;

    #[test]
    fn the_context_error_has_openais_code_type_and_param_and_the_rest_keep_the_status_as_code() {
        let b = ApiError::context_length("This model's maximum context length is 8 tokens.").body();
        assert_eq!(
            b,
            json!({"error": {"message": "This model's maximum context length is 8 tokens.", "type": "invalid_request_error",
                             "code": "context_length_exceeded", "param": "messages"}})
        );
        let b = ApiError::bad("n > 1 is not supported by this server").body();
        assert_eq!(b["error"]["code"], 400);
        assert!(b["error"].get("param").is_none());
        let b = ApiError::status(429, "server_overloaded", "full").body();
        assert_eq!((b["error"]["code"].clone(), b["error"]["type"].clone()), (json!(429), json!("server_overloaded")));
    }
}

#[cfg(test)]
mod usage_tests {
    use super::*;

    #[test]
    fn usage_has_openais_detail_objects_beside_the_totals_and_the_spec_block() {
        let u = usage_json(10, 5, 8, 3, 2, 4, 5);
        assert_eq!(
            u,
            json!({
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 5},
                "spec": {"drafted": 8, "accepted": 3, "rounds": 2},
            })
        );
    }
}

#[cfg(test)]
mod finish_tests {
    use super::*;

    #[test]
    fn a_reply_that_called_tools_finishes_as_tool_calls_unless_it_was_cut_off() {
        assert_eq!(finish_with_calls("stop", 1), "tool_calls");
        assert_eq!(finish_with_calls("stop", 3), "tool_calls");
        assert_eq!(finish_with_calls("stop", 0), "stop");
        assert_eq!(finish_with_calls("length", 2), "length", "a call cut off by the length limit is still a length finish");
        assert_eq!(finish_with_calls("length", 0), "length");
        // composed with the engine's own reason, as the server does
        assert_eq!(finish_with_calls(finish_str(None, false), 1), "tool_calls");
        assert_eq!(finish_with_calls(finish_str(Some(gb_engine::FinishReason::Length), false), 1), "length");
        assert_eq!(
            finish_with_calls(finish_str(Some(gb_engine::FinishReason::Length), true), 1),
            "tool_calls",
            "a stop string wins over the limit"
        );
    }
}

#[cfg(test)]
mod tool_choice_tests {
    use super::*;

    #[test]
    fn tool_choice_auto_and_none_are_supported_and_forcing_is_refused() {
        let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]);
        let auto = chat_template_kwargs(&json!({"tools": tools, "tool_choice": "auto"})).unwrap();
        assert!(auto.contains_key("tools"));
        let none = chat_template_kwargs(&json!({"tools": tools, "tool_choice": "none"})).unwrap();
        assert!(!none.contains_key("tools"), "tool_choice none hides the tools from the model");
        for forced in [json!("required"), json!({"type": "function", "function": {"name": "f"}})] {
            let e = chat_template_kwargs(&json!({"tools": tools, "tool_choice": forced})).unwrap_err();
            assert!(e.message.contains("tool_choice"), "{}", e.message);
        }
    }
}

/// Draft trees (docs/TREE_DRAFTING.md, "Sampler state that depends on the path"): every row of a verify window is sampled with the keyed
/// sampler at its own absolute position, from its own logits, so siblings share the draw of their depth and nothing the sampler reads
/// differs between a chain and a tree. Any sampling feature whose result depends on the tokens generated so far (penalties over the
/// history, logit bias by history, a grammar state) would make a node's sample depend on its OWN path, which the engine does not carry
/// per node: those requests are refused here, so none can reach a tree. Whoever supports one must give each tree node its path's history
/// (or send such a request down a chain with no tree) and replace this test.
#[cfg(test)]
mod tree_safety_tests {
    use super::*;

    #[test]
    fn requests_whose_sampling_depends_on_the_generated_history_are_refused() {
        let d = GenDefaults::default();
        for (field, value) in [
            ("logit_bias", json!({"1234": 5})),
            ("presence_penalty", json!(0.5)),
            ("frequency_penalty", json!(0.5)),
            ("repetition_penalty", json!(1.1)),
        ] {
            let e = parse_params(&json!({ field: value }), &d).unwrap_err();
            assert!(e.message.contains(field), "{field}: {}", e.message);
        }
        // neutral values are accepted (they change nothing)
        assert!(parse_params(
            &json!({"presence_penalty": 0, "frequency_penalty": 0.0, "repetition_penalty": 1.0, "logit_bias": {}}),
            &d
        )
        .is_ok());
    }
}
