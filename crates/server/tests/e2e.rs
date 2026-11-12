//! The whole HTTP path on the CPU mock backend: real tokenizer and chat template, real axum server, raw sockets.
//!
//! Runs everywhere: with no `REXL3_NEMO_OMNI_DIR` it uses the committed fixture (a byte-level tokenizer with the `<|im_start|>` chat tokens
//! and the Nemotron 3.5 Super VL chat template, see `tests/common`); with `REXL3_NEMO_OMNI_DIR` set it uses that pack's tokenizer instead.
//! The GLM tests need REXL3_GLM_DIR (the HF or EXL3 folder with tokenizer.json, config.json, chat_template.jinja and
//! generation_config.json) and skip without it.

mod common;

use common::{chat_body as chat, http, sse, start as start_with, Conn, Opts, Server};
use serde_json::{json, Value};
use std::time::Duration;

fn start(max_model_len: usize) -> Server {
    let dir = common::model_dir();
    // a real pack's tokenizer has ids far beyond the fixture's: the mock then takes the tokenizer's own vocabulary
    let vocab = if dir == common::fixture_dir() { 4096 } else { 0 };
    start_with(Opts {
        max_model_len,
        dir: Some(dir),
        mock: gb_engine::mock::MockOptions { vocab, ..Default::default() },
        ..Default::default()
    })
}

fn glm_dir() -> Option<std::path::PathBuf> {
    std::env::var("REXL3_GLM_DIR").ok().map(std::path::PathBuf::from).filter(|p| p.join("tokenizer.json").exists())
}

/// A server on the model folder `d` (the format comes from its `config.json`), the mock sized by that folder's tokenizer.
fn start_in(d: &std::path::Path, max_model_len: usize) -> Server {
    start_with(Opts {
        max_model_len,
        dir: Some(d.to_path_buf()),
        mock: gb_engine::mock::MockOptions { vocab: 0, ..Default::default() },
        ..Default::default()
    })
}

#[test]
fn health_models_and_metrics() {
    let s = start(4096);
    assert_eq!(http(s.port, "GET", "/health", None), (200, "{\"status\":\"ok\"}".into()));
    assert_eq!(http(s.port, "GET", "/health/ready", None), (200, "{\"status\":\"ok\"}".into()));
    let (st, b) = http(s.port, "GET", "/v1/models", None);
    assert_eq!(st, 200);
    let m: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(m["data"][0]["id"], "mock");
    assert_eq!(m["data"][0]["max_model_len"], 4096);
    // the pack's validation is reported (the mock reads no pack: both null; a real unvalidated pack: `gb_model::serve::check_pack_dirs`)
    for k in ["pack_validated", "pack_warning"] {
        assert!(m["data"][0].get(k).is_some(), "/v1/models lacks {k}: {b}");
    }
    // the prefix cache's reach is an additive field: present, and null for an engine that does not report it (the mock)
    assert!(m["data"][0]["prefix_cache_reach_tokens"].is_null(), "/v1/models: {b}");
    let (_, met) = http(s.port, "GET", "/metrics", None);
    assert!(met.contains("# HELP") && met.contains("kv_cache_size_tokens=\"4096\""));
    // the boot timeline and the weight loader's phase timers are exported beside each other
    for family in ["rexl3_boot_phase_seconds", "rexl3_load_stage_seconds", "rexl3_load_stage_bytes", "rexl3_load_stage_calls"] {
        assert!(met.contains(&format!("# TYPE {family} gauge")), "/metrics lacks {family}");
    }
    // a recorded stage shows up as a labelled sample
    gb_model::loadstats::add_for(
        gb_model::loadstats::Component::Draft,
        gb_model::loadstats::Stage::Memcpy,
        std::time::Duration::from_millis(250),
        4096,
    );
    let (_, met) = http(s.port, "GET", "/metrics", None);
    assert!(
        met.lines().any(|l| l.starts_with("rexl3_load_stage_bytes{component=\"draft\",stage=\"memcpy\"} ")),
        "a recorded loader stage is not exported:\n{}",
        met.lines().filter(|l| l.contains("load_stage")).collect::<Vec<_>>().join("\n")
    );
}

#[test]
fn chat_completion_non_streaming() {
    let s = start(4096);
    let (st, b) = http(s.port, "POST", "/v1/chat/completions", Some(&chat(json!({}))));
    assert_eq!(st, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 12);
    let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap();
    // the prompt the server tokenized is exactly what /render says
    let (_, r) = http(s.port, "POST", "/v1/chat/completions/render", Some(&chat(json!({}))));
    let r: Value = serde_json::from_str(&r).unwrap();
    assert_eq!(r["token_ids"].as_array().unwrap().len() as u64, prompt_tokens);
    assert_eq!(v["usage"]["total_tokens"].as_u64().unwrap(), prompt_tokens + 12);
    // an omitted request does not open a think block
    assert!(v["choices"][0]["message"].get("reasoning_content").is_none(), "{v}");
    // same request, same reply (seed derived from the prompt)
    let (_, b2) = http(s.port, "POST", "/v1/chat/completions", Some(&chat(json!({}))));
    let v2: Value = serde_json::from_str(&b2).unwrap();
    assert_eq!(v["choices"][0]["message"], v2["choices"][0]["message"]);
    // thinking off: the prompt already holds the closed block, so there is no reasoning field
    let (_, b3) =
        http(s.port, "POST", "/v1/chat/completions", Some(&chat(json!({"chat_template_kwargs": {"enable_thinking": false}}))));
    let v3: Value = serde_json::from_str(&b3).unwrap();
    assert!(v3["choices"][0]["message"].get("reasoning_content").is_none(), "{v3}");
}

#[test]
fn chat_completion_streaming_with_usage_and_token_ids() {
    let s = start(4096);
    let body = chat(json!({"stream": true, "stream_options": {"include_usage": true}, "return_token_ids": true}));
    let (st, b) = http(s.port, "POST", "/v1/chat/completions", Some(&body));
    assert_eq!(st, 200, "{b}");
    assert!(b.trim_end().ends_with("data: [DONE]"), "{b}");
    let chunks = sse(&b);
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let finish: Vec<&Value> = chunks
        .iter()
        .filter(|c| c["choices"].as_array().map(|a| !a.is_empty() && !a[0]["finish_reason"].is_null()).unwrap_or(false))
        .collect();
    assert_eq!(finish.len(), 1);
    assert_eq!(finish[0]["choices"][0]["finish_reason"], "length");
    let usage = chunks.last().unwrap();
    assert_eq!(usage["usage"]["completion_tokens"], 12);
    assert_eq!(usage["usage"]["spec"]["drafted"], 0);
    // streamed token ids add up to the completion
    let ids: usize = chunks
        .iter()
        .filter_map(|c| c["choices"].get(0).and_then(|ch| ch.get("token_ids")))
        .map(|t| t.as_array().unwrap().len())
        .sum();
    assert_eq!(ids, 12);
}

#[test]
fn bad_requests_get_clear_errors() {
    let s = start(512);
    let post = |b: Value| http(s.port, "POST", "/v1/chat/completions", Some(&b));
    for (name, body, needle) in [
        ("n=2", chat(json!({"n": 2})), "n > 1"),
        ("logprobs", chat(json!({"logprobs": true})), "logprobs"),
        ("penalty", chat(json!({"presence_penalty": 0.5})), "presence_penalty"),
        ("min_p", chat(json!({"min_p": 1.5})), "min_p"),
        ("max_tokens 0", chat(json!({"max_tokens": 0})), "positive integer"),
        ("no messages", json!({"model": "mock"}), "messages is required"),
        ("empty messages", json!({"messages": []}), "chat template failed"),
        (
            "bad role",
            json!({"messages": [{"role": "user", "content": "hi"}, {"role": "robot", "content": "x"}]}),
            "role is unsupported",
        ),
        ("too long", json!({"messages": [{"role": "user", "content": "word ".repeat(2000)}]}), "maximum context length is 512"),
    ] {
        let (st, b) = post(body);
        assert_eq!(st, 400, "{name}: {b}");
        assert!(b.contains(needle), "{name}: {b}");
        let v: Value = serde_json::from_str(&b).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error", "{name}: {b}");
        // every refusal carries the HTTP status as its code, except a prompt past the window: OpenAI's context_length_exceeded
        let want = if name == "too long" { json!("context_length_exceeded") } else { json!(400) };
        assert_eq!(v["error"]["code"], want, "{name}: {b}");
    }
    // the engine is still fine afterwards
    assert_eq!(post(chat(json!({}))).0, 200);
}

#[test]
fn completions_accepts_text_and_token_ids() {
    let s = start(4096);
    let (st, b) =
        http(s.port, "POST", "/v1/completions", Some(&json!({"prompt": "Hello there", "max_tokens": 5, "temperature": 0.0})));
    assert_eq!(st, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["usage"]["completion_tokens"], 5);
    let ids = s.state.tok.encode("Hello there").unwrap();
    let (_, b2) = http(s.port, "POST", "/v1/completions", Some(&json!({"prompt": ids, "max_tokens": 5, "temperature": 0.0})));
    let v2: Value = serde_json::from_str(&b2).unwrap();
    assert_eq!(v["choices"][0]["text"], v2["choices"][0]["text"], "text and token-id prompts must agree");
    assert_eq!(http(s.port, "POST", "/v1/completions", Some(&json!({"prompt": 5}))).0, 400);
}

#[test]
fn concurrent_requests_all_finish_and_memory_comes_back() {
    let s = start(4096);
    let port = s.port;
    let threads: Vec<_> = (0..6)
        .map(|i| {
            std::thread::spawn(move || {
                let body = json!({"messages": [{"role": "user", "content": format!("question number {i}")}], "max_tokens": 20, "temperature": 0.0, "stream": i % 2 == 0});
                let (st, b) = http(port, "POST", "/v1/chat/completions", Some(&body));
                assert_eq!(st, 200, "{b}");
                b
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    s.wait(Duration::from_secs(5), "the engine to give everything back", |s| s.in_flight() == 0);
    assert_eq!(s.metric(|m| &m.requests), 6);
    assert_eq!(s.metric(|m| &m.rejected), 0);
}

#[test]
fn a_client_that_disconnects_is_cancelled() {
    let s = start_with(Opts {
        mock: gb_engine::mock::MockOptions { step_ms: 5, vocab: 4096, ..Default::default() },
        ..Default::default()
    });
    let mut c = Conn::post(s.port, "/v1/chat/completions", &chat(json!({"stream": true, "max_tokens": 2000})));
    assert!(c.read_until(Duration::from_secs(10), |t| t.contains("data: ")), "the stream never started");
    // SSE role headers can arrive before the engine has entered admission. Zero in-flight at that instant is not cancellation.
    s.wait(Duration::from_secs(10), "the stream to enter engine admission", |s| s.in_flight() > 0);
    drop(c);
    // Wait for cancellation itself, not a transient queued -> prefilling zero in aggregate gauges.
    s.wait(Duration::from_secs(10), "the abandoned sequence to be cancelled", |s| {
        s.in_flight() == 0 && s.metric(|m| &m.cancelled) == 1
    });
    assert_eq!(s.metric(|m| &m.cancelled), 1);
}

// ---------------------------------------------------------------------------------------------------------------------
// GLM-5.3-Flash through the same HTTP path (needs REXL3_GLM_DIR)

fn glm_fixture() -> Value {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/glm53_chat_template_cases.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

/// The request body for a fixture case, the way an OpenAI client sends it (`tools` at the top level, the rest as chat_template_kwargs).
fn glm_body(case: &Value) -> Value {
    let mut kwargs = case["kwargs"].as_object().unwrap().clone();
    kwargs.remove("add_generation_prompt");
    let tools = kwargs.remove("tools");
    let messages = case.get("wire_messages").unwrap_or(&case["messages"]).clone();
    let mut body = json!({"model": "mock", "messages": messages, "chat_template_kwargs": kwargs});
    if let Some(t) = tools.filter(|t| !t.is_null()) {
        body["tools"] = t;
    }
    body
}

#[test]
fn glm_prompts_over_http_are_the_token_ids_hf_produces() {
    let Some(d) = glm_dir() else { return eprintln!("REXL3_GLM_DIR not set: skipping") };
    let s = start_in(&d, 65536);
    assert_eq!(s.state.format.id(), "glm5_next", "config.json of the folder selects the GLM format");
    let (mut checked, mut refused) = (0, 0);
    let fx = glm_fixture();
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        if c.get("images").is_some() || c.get("hf_refuses").is_some() || c["kwargs"].get("add_generation_prompt").is_some() {
            continue;
        }
        let (st, b) = http(s.port, "POST", "/v1/chat/completions/render", Some(&glm_body(c)));
        let media_refused = ["video", "audio", "input_image", "keyed image"].iter().any(|k| name.contains(k));
        match c["ids"].as_array() {
            Some(want) if !media_refused => {
                assert_eq!(st, 200, "{name}: {b}");
                let got: Value = serde_json::from_str(&b).unwrap();
                assert_eq!(got["token_ids"].as_array().unwrap(), want, "{name}");
                assert_eq!(got["count"].as_u64().unwrap() as usize, want.len());
                checked += 1;
            }
            // what the template cannot render, and the media the server refuses by name, are 400s with a reason
            _ => {
                if name.contains("JSON string") {
                    assert_eq!(st, 200, "{name}: {b}"); // an OpenAI client's string `arguments` is repaired
                } else {
                    assert_eq!(st, 400, "{name}: {b}");
                    refused += 1;
                }
            }
        }
    }
    assert!(checked >= 80 && refused >= 8, "{checked} prompts checked, {refused} refused");
    // the chat endpoint tokenizes the same prompt: usage says how many tokens
    let case = fx["cases"].as_array().unwrap().iter().find(|c| c["name"] == "tools").unwrap().clone();
    let mut body = glm_body(&case);
    body["max_tokens"] = json!(3);
    body["temperature"] = json!(0.0);
    let (st, b) = http(s.port, "POST", "/v1/chat/completions", Some(&body));
    assert_eq!(st, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["usage"]["prompt_tokens"].as_u64().unwrap() as usize, case["ids"].as_array().unwrap().len());
    assert!(v["choices"][0]["message"].get("reasoning_content").is_some(), "a GLM reply starts inside <think>");
}

#[test]
fn glm_refuses_what_its_template_would_drop_or_cannot_back() {
    let Some(d) = glm_dir() else { return eprintln!("REXL3_GLM_DIR not set: skipping") };
    let s = start_in(&d, 4096);
    let post = |b: Value| http(s.port, "POST", "/v1/chat/completions", Some(&b));
    let user = |parts: Value| json!({"model": "mock", "messages": [{"role": "user", "content": parts}], "max_tokens": 2});
    let img = json!([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]);
    for (name, body, needle) in [
        ("empty messages", json!({"messages": []}), "empty conversation"),
        ("input_image", user(json!([{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}])), "dropped silently"),
        ("keyed image", user(json!([{"image": "data:image/png;base64,AAAA"}])), "dropped silently"),
        ("video", user(json!([{"type": "video", "video": "x"}])), "not supported"),
        ("audio", user(json!([{"type": "input_audio", "input_audio": {}}])), "not supported"),
        ("image without a vision tower", user(img), "without image support"),
    ] {
        let (st, b) = post(body);
        assert_eq!(st, 400, "{name}: {b}");
        assert!(b.contains(needle), "{name}: {b}");
    }
    assert_eq!(post(user(json!("plain text"))).0, 200);
}

/// The mock model copies the continuation of the latest earlier occurrence of its last two tokens. A prompt whose history holds one
/// assistant turn (reasoning, text, two tool calls) therefore makes it write that turn again: the whole real path (GLM template, tokenizer,
/// engine, reasoning split, tool-call splitter, finish reason) runs on text that has the exact shape of a GLM reply.
fn glm_replay_body(extra: Value) -> Value {
    let tools = json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}}
    ]);
    let mut b = json!({"model": "mock", "temperature": 0.0, "max_tokens": 200, "tools": tools, "messages": [
        {"role": "user", "content": "Weather in Paris, and what time is it?"},
        {"role": "assistant", "reasoning_content": "need both", "content": "Checking.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}},
            {"id": "c2", "type": "function", "function": {"name": "now", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "tool", "tool_call_id": "c2", "content": "noon"}]});
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

fn call_pair(c: &Value) -> (String, Value) {
    (
        c["function"]["name"].as_str().unwrap().to_string(),
        serde_json::from_str(c["function"]["arguments"].as_str().unwrap()).unwrap(),
    )
}

#[test]
fn glm_reply_is_split_into_reasoning_content_and_tool_calls() {
    let Some(d) = glm_dir() else { return eprintln!("REXL3_GLM_DIR not set: skipping") };
    let s = start_in(&d, 8192);
    let (st, b) = http(s.port, "POST", "/v1/chat/completions", Some(&glm_replay_body(json!({}))));
    assert_eq!(st, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    let ch = &v["choices"][0];
    assert_eq!(ch["finish_reason"], "tool_calls", "{v}");
    let m = &ch["message"];
    assert_eq!(m["reasoning_content"], "need both", "{v}");
    assert_eq!(m["content"], "Checking.", "{v}");
    let calls = m["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 2, "{v}");
    assert_eq!(call_pair(&calls[0]), ("get_weather".to_string(), json!({"city": "Paris"})));
    assert_eq!(call_pair(&calls[1]), ("now".to_string(), json!({})));
    assert!(calls.iter().all(|c| c["type"] == "function" && c["id"].as_str().unwrap().starts_with("call_")));
    // with no tools offered the same reply is plain text and the finish is a plain stop
    let mut no_tools = glm_replay_body(json!({}));
    no_tools.as_object_mut().unwrap().remove("tools");
    let (_, b2) = http(s.port, "POST", "/v1/chat/completions", Some(&no_tools));
    let v2: Value = serde_json::from_str(&b2).unwrap();
    assert_eq!(v2["choices"][0]["finish_reason"], "stop", "{v2}");
    assert!(v2["choices"][0]["message"].get("tool_calls").is_none());
    let text = v2["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(text.contains("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"), "{v2}");
}

#[test]
fn glm_streamed_reply_carries_the_same_reasoning_text_and_tool_calls() {
    let Some(d) = glm_dir() else { return eprintln!("REXL3_GLM_DIR not set: skipping") };
    let s = start_in(&d, 8192);
    let (st, b) = http(s.port, "POST", "/v1/chat/completions", Some(&glm_replay_body(json!({"stream": true}))));
    assert_eq!(st, 200, "{b}");
    assert!(b.trim_end().ends_with("data: [DONE]"), "{b}");
    let (mut reasoning, mut content) = (String::new(), String::new());
    let mut calls: Vec<(u64, String, Value)> = Vec::new();
    let mut finishes = Vec::new();
    for c in sse(&b) {
        let Some(choice) = c["choices"].get(0) else { continue };
        let d = &choice["delta"];
        reasoning.push_str(d["reasoning_content"].as_str().unwrap_or(""));
        content.push_str(d["content"].as_str().unwrap_or(""));
        for t in d["tool_calls"].as_array().into_iter().flatten() {
            let (name, args) = call_pair(t);
            calls.push((t["index"].as_u64().unwrap(), name, args));
        }
        if !choice["finish_reason"].is_null() {
            finishes.push(choice["finish_reason"].as_str().unwrap().to_string());
        }
    }
    assert_eq!((reasoning.as_str(), content.as_str()), ("need both", "Checking."));
    assert_eq!(calls, vec![(0, "get_weather".to_string(), json!({"city": "Paris"})), (1, "now".to_string(), json!({}))]);
    assert_eq!(finishes, vec!["tool_calls".to_string()]);
}
