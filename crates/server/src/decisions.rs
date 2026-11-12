//! Pure server-side preparation of `/v1/decisions` requests: strict validation, the closed Qwen chat prompt, the
//! one-token label gate, and the label-scoring math over a full-vocabulary logits row.
//!
//! This module is deliberately pure: it touches no engine, no runtime and no HTTP. The runtime hands it the served
//! tokenizer and the family's [`ChatFormat`], receives back one [`PreparedDecision`] per question (prompt token ids,
//! label token ids, names), and hands each one to the engine's score path; [`build_response`] turns the returned
//! full rows into the v1 wire response. The wire format is the SGLang decision format v1; the implementation is
//! native (no upstream source, behavior matched only).
//!
//! Math (all host-side f64, compensated summation):
//! - label-conditioned probabilities `p_i = softmax(label_logits / temperature)` over the K label logits only,
//!   shifted by their max before exponentiation, so a tiny positive temperature stays finite and exact;
//! - per-label logprobs `logit_i - logsumexp(full_row)` against the full vocabulary, never a top-k subset —
//!   `label_mass = Σ exp(logprob_i)` must be exact for labels ranked far below any candidate window;
//! - `choice` picks the argmax probability with the first index winning ties; `score` is `Σ index·p_index` over
//!   the 0-based levels; `yes_no` adds nothing; `completion_tokens` is always 0.

use crate::api::{ApiError, CONTEXT_LENGTH_EXCEEDED};
use crate::chat_format::ChatFormat;
use crate::decision_casefold;
use crate::tokenizer::Tok;
use serde_json::{json, Map, Value};

/// The wire contract version this module serves.
pub const PROMPT_FORMAT_VERSION: u64 = 1;

/// Top-level request fields (`model` optional and informational; unknown names are refused).
const TOP_FIELDS: [&str; 7] =
    ["input", "questions", "temperature", "chat_template_kwargs", "prompt_format_version", "return_prompt_token_ids", "model"];
const CHOICE_FIELDS: [&str; 4] = ["id", "type", "question", "options"];
const SCORE_FIELDS: [&str; 4] = ["id", "type", "question", "levels"];
const YES_NO_FIELDS: [&str; 5] = ["id", "type", "question", "yes", "no"];
const OPTION_FIELDS: [&str; 2] = ["name", "description"];

/// A question's upper bound on options (`A`..`Z`).
const MAX_OPTIONS: usize = 26;
/// A score question's upper bound on levels (`0`..`9`).
const MAX_LEVELS: usize = 10;
pub const MAX_PROMPT_BYTES: usize = 8 << 20;

/// One question, fully prepared: the closed chat prompt, its encoding, and the one-token label ids.
///
/// The runtime HTTP layer reads `prompt` and `label_ids` to drive the engine's score path and maps `names` to the
/// returned probabilities; `labels` keeps the per-label spelling (`A`/`B`…, `0`/`1`…, `yes`/`no`).
#[derive(Clone, Debug)]
pub struct PreparedDecision {
    /// The question's `id`, as the request named it (the response's `answers` key).
    pub id: String,
    /// `choice` | `score` | `yes_no`.
    pub kind: &'static str,
    /// Response keys per label: option names, level indices as strings, or `yes`,`no`.
    pub names: Vec<String>,
    /// The label spellings the prompt itself shows (`A`.., `0`.., `yes`,`no`).
    pub labels: Vec<String>,
    /// The rendered chat prompt text (closed thinking; the answer position is the end of the text).
    pub prompt_text: String,
    /// The full prompt encoding, ready for the engine's score prefill.
    pub prompt: Vec<u32>,
    /// One verified single-token id per label, in `names` order.
    pub label_ids: Vec<u32>,
    /// Whether the request asked the prompt ids echoed back.
    pub return_prompt_token_ids: bool,
}

/// One question's score-job result, as the engine's score path will hand it over (the full host row).
#[derive(Clone, Debug)]
pub struct ScoredQuestion {
    /// FULL vocabulary logits of the prompt's last position, host-side f32.
    pub logits_row: Vec<f32>,
    /// Prompt tokens a prefix-cache hit supplied (they were not prefilled for this request).
    pub cached_tokens: usize,
}

/// `logsumexp` of a full logits row in f64 with a max shift and compensated summation. An empty row is refused by
/// the callers; a row of `-inf` gives `-inf`.
pub fn logsumexp_full_row(row: &[f32]) -> f64 {
    let peak = row.iter().copied().fold(f64::NEG_INFINITY, |a: f64, b: f32| a.max(b as f64));
    if !peak.is_finite() {
        // all -inf (or a NaN somewhere: fold keeps NaN), nothing to exponentiate
        return peak;
    }
    let sum = compensated(&mut row.iter().map(|&v| (v as f64 - peak).exp()));
    peak + sum.ln()
}

/// The label-conditioned softmax at `temperature`. Zero is greedy: one-hot on the first maximum logit, the same
/// tie rule as `choice`. A positive temperature shifts by the max label logit before dividing.
pub fn softmax_labels(label_logits: &[f32], temperature: f64) -> Vec<f64> {
    if label_logits.is_empty() {
        return Vec::new();
    }
    if temperature == 0.0 {
        let mut best = 0;
        for (i, &logit) in label_logits.iter().enumerate() {
            if logit > label_logits[best] {
                best = i;
            }
        }
        let mut out = vec![0.0; label_logits.len()];
        out[best] = 1.0;
        return out;
    }
    let peak = label_logits.iter().copied().fold(f64::NEG_INFINITY, |a: f64, b: f32| a.max(b as f64));
    let weights: Vec<f64> = label_logits.iter().map(|&l| ((l as f64 - peak) / temperature).exp()).collect();
    let total = compensated(&mut weights.iter().copied());
    weights.into_iter().map(|w| w / total).collect()
}

/// Kahan-Neumaier compensated summation: the same accumulation every response is built from, so a change of order
/// (never happens here — the sums are over fixed slices) or a long tail of tiny terms cannot drift the result.
fn compensated(items: &mut dyn Iterator<Item = f64>) -> f64 {
    let (mut sum, mut c) = (0.0_f64, 0.0_f64);
    for x in items {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }
    sum + c
}

/// Validate the body and prepare every question against `tok`/`format`: per-question failures carry the id
/// (`question "topic": ...`, or `question at position 2: ...` when the id itself is unusable).
pub fn prepare(tok: &Tok, format: &dyn ChatFormat, body: Value, context_len: usize) -> Result<Vec<PreparedDecision>, ApiError> {
    prepare_capped(tok, format, body, context_len, MAX_PROMPT_BYTES)
}

/// [`prepare`] with each question's prompt held to `prompt_cap` bytes (the server's `--max-rendered-prompt-mib`; never above
/// [`MAX_PROMPT_BYTES`], the aggregate of all the questions' prompts). A prompt is tokenized whole, so its size, not the aggregate, is
/// what one preparation costs in memory.
pub fn prepare_capped(
    tok: &Tok,
    format: &dyn ChatFormat,
    body: Value,
    context_len: usize,
    prompt_cap: usize,
) -> Result<Vec<PreparedDecision>, ApiError> {
    let prompt_cap = prompt_cap.min(MAX_PROMPT_BYTES);
    validate_body(&body)?;
    if format.id() != "qwen" {
        return Err(ApiError::bad("decisions scoring is not supported by this model family"));
    }
    let text = render_text(body.get("input").unwrap_or(&Value::Null))?;
    if text.trim().is_empty() {
        return Err(ApiError::bad("input must not be blank"));
    }
    // every question's prompt carries the whole input
    if text.len() > prompt_cap {
        return Err(crate::budget::too_large("decision input exceeds the rendered prompt byte limit"));
    }
    let questions = body.get("questions").and_then(|q| q.as_array()).map(|v| v.as_slice()).unwrap_or(&[]);
    let mut kwargs = Map::new();
    kwargs.insert("enable_thinking".into(), Value::Bool(false));
    let mut out = Vec::with_capacity(questions.len());
    let mut bytes_left = MAX_PROMPT_BYTES;
    for (index, question) in questions.iter().enumerate() {
        let obj = question.as_object().ok_or_else(|| ApiError::bad("each question must be an object"))?;
        let err = |e: ApiError| prefix_question_error(e, obj, index);
        let mut p = prepare_question(tok, format, &text, obj, &kwargs, context_len, bytes_left.min(prompt_cap)).map_err(err)?;
        bytes_left = bytes_left
            .checked_sub(p.prompt_text.len())
            .ok_or_else(|| ApiError::bad("aggregate decision prompts exceed byte limit"))?;
        p.return_prompt_token_ids = body.get("return_prompt_token_ids").and_then(Value::as_bool).unwrap_or(false);
        out.push(p);
    }
    Ok(out)
}

/// The closed-thinking refusal, factored so tests (and a family audit) can run it on any render: an open
/// `<think>` at the answer position would score a reasoning token, not the answer.
pub fn refuse_open_reasoning(rendered: &crate::chat_format::RenderedChat) -> Result<(), ApiError> {
    // the family already knows (`starts_in_reasoning`); the marker scan is the template-independent backstop
    let open = rendered.starts_in_reasoning
        || rendered.text.rfind("<think>").is_some_and(|t| rendered.text.rfind("</think>").is_none_or(|c| t > c));
    if open {
        return Err(ApiError::bad(
            "decisions need the chat template to leave no reasoning block open at the answer position; \
             this model's template always opens one, so it is not supported (decisions need enable_thinking false or unset)",
        ));
    }
    Ok(())
}

/// The one-token label gate: `encode(prompt + label)` must be `prompt_ids` plus one new id, distinct per label.
pub fn check_labels(tok: &Tok, prompt_text: &str, prompt_ids: &[u32], labels: &[String]) -> Result<Vec<u32>, ApiError> {
    let mut found = Vec::with_capacity(labels.len());
    for label in labels {
        if label.len() > 32 || prompt_text.len() > MAX_PROMPT_BYTES.saturating_sub(label.len()) {
            return Err(ApiError::bad("decision label/prompt exceeds byte limit"));
        }
        let ids = tok.encode(&format!("{prompt_text}{label}"))?;
        if ids.len() != prompt_ids.len() + 1 || ids[..prompt_ids.len()] != *prompt_ids || found.contains(&ids[prompt_ids.len()]) {
            return Err(ApiError::bad(format!(
                "the answer label {label:?} is not one distinct token after the chat prompt for this tokenizer, \
                 so this model is not supported"
            )));
        }
        found.push(ids[prompt_ids.len()]);
    }
    Ok(found)
}

/// Turn prepared questions plus full rows into the v1 response. `scored[i]` answers `prepared[i]`; a short or
/// non-finite result is a 500 (the engine failed the request), never a fabricated answer.
pub fn build_response(body: &Value, prepared: &[PreparedDecision], scored: &[ScoredQuestion]) -> Result<Value, ApiError> {
    if scored.len() != prepared.len() {
        return Err(ApiError::status(
            500,
            "server_error",
            format!("the engine returned {} scored rows for {} questions", scored.len(), prepared.len()),
        ));
    }
    let temperature = body.get("temperature").and_then(|t| t.as_f64()).unwrap_or(1.0);
    let return_ids = body.get("return_prompt_token_ids").and_then(|v| v.as_bool()).unwrap_or(false);
    let mut answers = Map::new();
    let mut prompt_tokens = 0_usize;
    let mut cached_tokens = 0_usize;
    for (q, s) in prepared.iter().zip(scored) {
        prompt_tokens += q.prompt.len();
        cached_tokens += s.cached_tokens;
        if s.logits_row.is_empty() {
            return Err(ApiError::status(
                500,
                "server_error",
                format!("question {:?}: the engine returned an empty logit row", q.id),
            ));
        }
        if s.logits_row.iter().any(|v| !v.is_finite()) {
            return Err(ApiError::status(500, "server_error", "non-finite score vocabulary row"));
        }
        let lse = logsumexp_full_row(&s.logits_row);
        let mut logits = Vec::with_capacity(q.label_ids.len());
        for (_name, &id) in q.names.iter().zip(&q.label_ids) {
            let id = id as usize;
            if id >= s.logits_row.len() {
                return Err(ApiError::status(
                    500,
                    "server_error",
                    format!("question {:?}: label token {id} is outside the vocabulary of the returned row", q.id),
                ));
            }
            let v = s.logits_row[id];
            if !v.is_finite() {
                return Err(ApiError::status(
                    500,
                    "server_error",
                    format!("question {:?}: label scoring produced a non-finite logit", q.id),
                ));
            }
            logits.push(v);
        }
        if !lse.is_finite() {
            return Err(ApiError::status(
                500,
                "server_error",
                format!("question {:?}: label scoring produced a non-finite logsumexp", q.id),
            ));
        }
        let probabilities = softmax_labels(&logits, temperature);
        // Algebraically exp(logit - logsumexp(full)), but subtract the common peak BEFORE
        // accumulation to preserve mass even when all logits have a very large common offset.
        let peak = s.logits_row.iter().map(|&v| v as f64).fold(f64::NEG_INFINITY, f64::max);
        let denominator = compensated(&mut s.logits_row.iter().map(|&v| (v as f64 - peak).exp()));
        let mass = (compensated(&mut logits.iter().map(|&l| (l as f64 - peak).exp())) / denominator).min(1.0);
        // names in option order on the wire, then the derived fields
        let probs_value: Value = {
            let pairs = q.names.iter().zip(&probabilities).map(|(n, p)| (n.clone(), json!(p))).collect();
            Value::Object(pairs)
        };
        let mut answer = Map::new();
        answer.insert("type".into(), json!(q.kind));
        answer.insert("probabilities".into(), probs_value.clone());
        answer.insert("label_mass".into(), json!(mass));
        match q.kind {
            "choice" => {
                // the first index wins ties (option order): a strict `>` keeps the earliest maximum
                let mut top = 0;
                for i in 0..logits.len() {
                    if logits[i] > logits[top] {
                        top = i;
                    }
                }
                answer.insert("choice".into(), json!(q.names[top]));
            }
            "score" => {
                let expected = compensated(&mut probabilities.iter().enumerate().map(|(i, p)| i as f64 * *p));
                answer.insert("score".into(), json!(expected));
            }
            _ => {}
        }
        if return_ids || q.return_prompt_token_ids {
            answer.insert("prompt_token_ids".into(), json!(q.prompt));
            let label_map: Map<String, Value> = q.labels.iter().cloned().zip(q.label_ids.iter().map(|&i| json!(i))).collect();
            answer.insert("label_token_ids".into(), Value::Object(label_map));
        }
        answers.insert(q.id.clone(), Value::Object(answer));
    }
    Ok(json!({
        "object": "decisions",
        "model": body.get("model").and_then(|m| m.as_str()).unwrap_or("default"),
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "answers": Value::Object(answers),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }))
}

/// Response keys per label in `f64` order, for a runtime caller that wants the numbers back out of the wire form.
#[allow(dead_code)]
fn probabilities_by(q: &PreparedDecision, response: &Value) -> Vec<f64> {
    let obj = response.as_object().unwrap();
    q.names.iter().map(|n| obj.get(n).and_then(|v| v.as_f64()).unwrap_or(0.0)).collect()
}

fn prefix_question_error(e: ApiError, question: &Map<String, Value>, index: usize) -> ApiError {
    let id = question.get("id").and_then(|i| i.as_str());
    let where_ = match id {
        Some(id) if !id.trim().is_empty() => {
            format!("question {id:?}")
        }
        _ => format!("question at position {}", index + 1),
    };
    ApiError { message: format!("{where_}: {}", e.message), ..e }
}

// ---------------------------------------------------------------- validation

fn validate_body(body: &Value) -> Result<(), ApiError> {
    let obj = body.as_object().ok_or_else(|| ApiError::bad("the request body must be a JSON object"))?;
    if let Some(unknown) = obj.keys().find(|k| !TOP_FIELDS.contains(&k.as_str())) {
        return Err(ApiError::bad(format!("unknown field {unknown:?}")));
    }
    match obj.get("prompt_format_version") {
        None => {}
        Some(v) => {
            let n = v.as_u64().filter(|_| v.is_u64());
            if n != Some(PROMPT_FORMAT_VERSION) {
                return Err(ApiError::bad(format!(
                    "prompt_format_version {v} is not served, this server uses version {PROMPT_FORMAT_VERSION}"
                )));
            }
        }
    }
    match obj.get("temperature") {
        None => {}
        Some(v) => match v.as_f64() {
            Some(t) if v.is_number() && !v.is_boolean() && t.is_finite() && t >= 0.0 => {}
            _ => return Err(ApiError::bad("temperature must be a finite number of at least 0")),
        },
    }
    match obj.get("chat_template_kwargs") {
        None => {}
        Some(Value::Object(m)) => {
            if let Some(unknown) = m.keys().find(|k| k != &"enable_thinking") {
                return Err(ApiError::bad(format!("unknown field {unknown:?} in chat_template_kwargs")));
            }
            match m.get("enable_thinking") {
                None | Some(Value::Bool(false)) => {}
                _ => return Err(ApiError::bad("decisions need enable_thinking false or unset")),
            }
        }
        Some(_) => return Err(ApiError::bad("chat_template_kwargs must be an object")),
    }
    match obj.get("model") {
        None | Some(Value::String(_)) => {}
        Some(_) => return Err(ApiError::bad("model must be a string")),
    }
    match obj.get("return_prompt_token_ids") {
        None | Some(Value::Bool(_)) => {}
        Some(_) => return Err(ApiError::bad("return_prompt_token_ids must be a boolean")),
    }
    match obj.get("questions") {
        Some(Value::Array(qs)) if !qs.is_empty() && qs.len() <= 10000 => {
            let mut seen = std::collections::HashSet::new();
            for q in qs {
                let obj = q.as_object().ok_or_else(|| ApiError::bad("each question must be an object"))?;
                let id = obj.get("id").and_then(|i| i.as_str()).filter(|i| !i.trim().is_empty());
                let id = id.ok_or_else(|| ApiError::bad("a question id must be a non-blank string"))?;
                if !seen.insert(id.to_string()) {
                    return Err(ApiError::bad(format!("question id {id:?} is repeated")));
                }
            }
        }
        Some(Value::Array(_)) => return Err(ApiError::bad("questions must contain at least one question")),
        Some(_) => return Err(ApiError::bad("questions must be a list of questions")),
        None => return Err(ApiError::bad("questions must contain at least one question")),
    }
    Ok(())
}

/// One question: wording, render, encode, label gate, context check.
fn prepare_question(
    tok: &Tok,
    format: &dyn ChatFormat,
    input: &str,
    question: &Map<String, Value>,
    kwargs: &Map<String, Value>,
    context_len: usize,
    bytes_left: usize,
) -> Result<PreparedDecision, ApiError> {
    let (kind, names, labels, wording) = wording(input, question)?;
    let id = question.get("id").and_then(|i| i.as_str()).unwrap_or_default().to_string();
    let mut messages = Map::new();
    messages.insert("role".into(), json!("user"));
    messages.insert("content".into(), json!(wording));
    let messages = Value::Array(vec![Value::Object(messages)]);
    let rendered = format.render_decision(&messages, kwargs, bytes_left).map_err(|e| match e {
        crate::template::TemplateError::Limit => crate::budget::too_large("decision rendered prompt byte limit exceeded"),
        crate::template::TemplateError::Raised(m) => ApiError::bad(m),
        crate::template::TemplateError::Render(m) => ApiError::bad(format!("chat template failed: {m}")),
    })?;
    refuse_open_reasoning(&rendered)?;
    let prompt_ids = tok.encode(&rendered.text)?;
    if prompt_ids.len() > context_len {
        return Err(ApiError {
            code: Some(CONTEXT_LENGTH_EXCEEDED),
            param: Some("input".into()),
            ..ApiError::bad(format!(
                "the prompt has {} tokens, but this server's maximum context length is {} tokens",
                prompt_ids.len(),
                context_len
            ))
        });
    }
    let label_ids = check_labels(tok, &rendered.text, &prompt_ids, &labels)?;
    Ok(PreparedDecision {
        id,
        kind,
        names,
        labels,
        prompt_text: rendered.text,
        prompt: prompt_ids,
        label_ids,
        return_prompt_token_ids: false,
    })
}

/// The TF `_wording` layout, matched: input, blank line, `Question: …` (or the yes/no lead), option/level lines,
/// the closing instruction.
type Wording = (&'static str, Vec<String>, Vec<String>, String);
fn wording(input: &str, question: &Map<String, Value>) -> Result<Wording, ApiError> {
    // Bound all wording inputs before cloning descriptions/levels or constructing line strings.
    // At most 26 labels plus template/instruction overhead fit inside this reserved margin.
    fn text_bytes(v: &Value) -> Option<usize> {
        match v {
            Value::String(s) => Some(s.len()),
            Value::Array(a) => a.iter().try_fold(0usize, |n, v| n.checked_add(text_bytes(v)?)),
            Value::Object(m) => m.values().try_fold(0usize, |n, v| n.checked_add(text_bytes(v)?)),
            _ => Some(0),
        }
    }
    let mut text_size = input.len();
    for key in ["question", "options", "levels", "yes", "no"] {
        if let Some(v) = question.get(key) {
            text_size = text_size
                .checked_add(text_bytes(v).ok_or_else(|| ApiError::bad("decision wording byte limit"))?)
                .ok_or_else(|| ApiError::bad("decision wording byte limit"))?;
        }
    }
    if text_size > MAX_PROMPT_BYTES.saturating_sub(4096) {
        return Err(ApiError::bad("decision wording exceeds byte limit"));
    }
    let kind = match question.get("type").and_then(|t| t.as_str()) {
        Some("choice") => "choice",
        Some("score") => "score",
        Some("yes_no") => "yes_no",
        other => return Err(ApiError::bad(format!("unknown question type {:?}", other))),
    };
    let question_line = |q: &Map<String, Value>| -> Result<String, ApiError> {
        let text = strict_text(q.get("question"), "question")?;
        if text.trim().is_empty() {
            return Err(ApiError::bad("a question must not be blank"));
        }
        Ok(format!("Question: {text}"))
    };
    let mut lines: Vec<String> = Vec::new();
    let (names, labels): (Vec<String>, Vec<String>) = match kind {
        "choice" => {
            only_fields(question, &CHOICE_FIELDS)?;
            let options = question
                .get("options")
                .and_then(|o| o.as_array())
                .ok_or_else(|| ApiError::bad("a choice needs 2 to 26 options"))?;
            if options.len() < 2 || options.len() > MAX_OPTIONS {
                return Err(ApiError::bad(format!("a choice needs 2 to {} options", MAX_OPTIONS)));
            }
            let mut names = Vec::with_capacity(options.len());
            let mut details = Vec::with_capacity(options.len());
            let mut seen = std::collections::HashSet::new();
            for option in options {
                let obj = option.as_object().ok_or_else(|| ApiError::bad("each option must be an object"))?;
                only_fields(obj, &OPTION_FIELDS)?;
                let name = obj
                    .get("name")
                    .and_then(|n| n.as_str())
                    .filter(|n| {
                        let s = n.trim();
                        !s.is_empty()
                            && n.len() <= MAX_PROMPT_BYTES
                            && !s.chars().any(|c| c == '\n' || c == '\r' || c.is_control())
                    })
                    .ok_or_else(|| ApiError::bad("option names must not be blank or contain line breaks"))?;
                let folded = decision_casefold::fold(name.trim());
                if !seen.insert(folded) {
                    return Err(ApiError::bad(format!("option name {name:?} repeats another name")));
                }
                names.push(name.to_string());
                details.push(optional_text(obj.get("description"))?);
            }
            lines.push(question_line(question)?);
            for (i, name) in names.iter().enumerate() {
                let label = (b'A' + i as u8) as char;
                lines.push(match details[i].as_deref().unwrap_or("") {
                    "" => format!("{label}: {name}"),
                    d => format!("{label}: {name} - {d}"),
                });
            }
            let labels = (0..names.len()).map(|i| ((b'A' + i as u8) as char).to_string()).collect();
            (names, labels)
        }
        "score" => {
            only_fields(question, &SCORE_FIELDS)?;
            let levels =
                question.get("levels").and_then(|l| l.as_array()).ok_or_else(|| ApiError::bad("a score needs 2 to 10 levels"))?;
            if levels.len() < 2 || levels.len() > MAX_LEVELS {
                return Err(ApiError::bad(format!("a score needs 2 to {} levels", MAX_LEVELS)));
            }
            let mut details = Vec::with_capacity(levels.len());
            for level in levels {
                let text = strict_text(Some(level), "level")?;
                if text.trim().is_empty() {
                    return Err(ApiError::bad("a level must not be blank"));
                }
                details.push(text);
            }
            let names: Vec<String> = (0..details.len()).map(|i| i.to_string()).collect();
            lines.push(question_line(question)?);
            for (label, detail) in names.iter().zip(&details) {
                lines.push(format!("{label}: {detail}"));
            }
            (names.clone(), names)
        }
        _ => {
            only_fields(question, &YES_NO_FIELDS)?;
            let lead = strict_text(question.get("question"), "question")?;
            if lead.trim().is_empty() {
                return Err(ApiError::bad("a question must not be blank"));
            }
            lines.push(format!("Is the following true? {lead}"));
            for (label, key) in [("yes", "yes"), ("no", "no")] {
                if let Some(d) = optional_text(question.get(key))? {
                    lines.push(format!("{label}: {d}"));
                }
            }
            (vec!["yes".into(), "no".into()], vec!["yes".into(), "no".into()])
        }
    };
    lines.push(match kind {
        "choice" => "Answer with the letter of one option only.".to_string(),
        "score" => "Answer with the number of one level only.".to_string(),
        _ => "Answer with yes or no only.".to_string(),
    });
    let mut text = String::new();
    for part in std::iter::once(input).chain(std::iter::once("\n")).chain(lines.iter().map(String::as_str)) {
        if part.len().saturating_add(1) > MAX_PROMPT_BYTES.saturating_sub(text.len()) {
            return Err(ApiError::bad("decision wording exceeds byte limit"));
        }
        text.push_str(part);
        text.push('\n');
    }
    Ok((kind, names, labels, text))
}

fn only_fields(obj: &Map<String, Value>, allowed: &[&str]) -> Result<(), ApiError> {
    if let Some(unknown) = obj.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(ApiError::bad(format!("unknown field {unknown:?}")));
    }
    Ok(())
}

/// `input`, question texts, levels and descriptions: string as is; object/array JSON-compacted; anything else
/// refused. `None` for an absent/null optional text (a description may be omitted, a question may not).
fn render_text(value: &Value) -> Result<String, ApiError> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(s) if s.len() <= MAX_PROMPT_BYTES => Ok(s.clone()),
        Value::Object(_) | Value::Array(_) => {
            struct W(Vec<u8>);
            impl std::io::Write for W {
                fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                    if b.len() > MAX_PROMPT_BYTES.saturating_sub(self.0.len()) {
                        return Err(std::io::Error::other("input byte limit"));
                    }
                    self.0.extend_from_slice(b);
                    Ok(b.len())
                }
                fn flush(&mut self) -> std::io::Result<()> {
                    Ok(())
                }
            }
            let mut w = W(Vec::new());
            serde_json::to_writer(&mut w, value).map_err(|_| ApiError::bad("decision input exceeds byte limit"))?;
            String::from_utf8(w.0).map_err(|e| ApiError::bad(e.to_string()))
        }
        _ => Err(ApiError::bad("text must be a string, object, or array")),
    }
}

fn strict_text(value: Option<&Value>, field: &str) -> Result<String, ApiError> {
    match value {
        Some(Value::String(s)) if s.len() <= MAX_PROMPT_BYTES => Ok(s.clone()),
        _ => Err(ApiError::bad(format!("{field} must be a bounded string"))),
    }
}
fn optional_text(value: Option<&Value>) -> Result<Option<String>, ApiError> {
    match value {
        None => Ok(None),
        v => strict_text(v, "description").map(Some),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensated_sums_exact_where_a_plain_loop_drifts() {
        let c = compensated(&mut [1e16, 1.0, -1e16].into_iter());
        assert_eq!(c, 1.0);
        assert_eq!(compensated(&mut std::iter::empty()), 0.0);
    }

    #[test]
    fn softmax_is_stable_at_extremes() {
        // a huge spread would overflow without the shift
        let p = softmax_labels(&[1e30, 0.0], 1.0);
        assert_eq!(p[0], 1.0);
        // subnormal-scale logits do not lose the distribution
        let p = softmax_labels(&[-1000.0, -1001.0], 1.0);
        let ratio = p[0] / p[1];
        assert!((ratio - std::f64::consts::E).abs() < 1e-6, "{p:?}");
        let greedy = softmax_labels(&[1.0, 3.0, 3.0], 0.0);
        assert_eq!(greedy, vec![0.0, 1.0, 0.0]);
    }
}
