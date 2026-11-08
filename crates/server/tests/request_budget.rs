//! Resource/API regression tests run against the actual Axum/Hyper TCP server, never a mocked router.
mod common;
use common::{chat_body, http_timeout, start, Opts};
use serde_json::{json, Value};
use std::time::Duration;

fn post(s: &common::Server, path: &str, body: Value) -> common::Reply {
    http_timeout(s.port, "POST", path, Some(&body), Duration::from_secs(20))
}

#[test]
fn wrong_models_and_wrong_scalar_types_are_not_silently_accepted() {
    let s = start(Opts::default());
    for path in ["/v1/chat/completions", "/v1/chat/completions/render", "/v1/completions"] {
        let mut b = chat_body(json!({"model":"wrong-model", "prompt":[1], "max_tokens":1}));
        let r = post(&s, path, b.clone());
        assert_eq!(r.status, 404, "{path}: {}", r.body);
        assert_eq!(r.json()["error"]["code"], "model_not_found");
        b["model"] = json!(17);
        assert_eq!(post(&s, path, b).status, 400);
    }
    for field in ["stream", "skip_special_tokens", "return_token_ids", "draft", "prefix_cache"] {
        let r = post(&s, "/v1/chat/completions", chat_body(json!({field:"true", "max_tokens":1})));
        assert_eq!(r.status, 400, "{field}: {}", r.body);
        assert_eq!(r.json()["error"]["param"], field);
    }
    for extra in [
        json!({"max_tokens":2.5}),
        json!({"max_completion_tokens":2.5}),
        json!({"top_k":2.5}),
        json!({"stream_options":{"include_usage":"yes"}}),
        json!({"stream_options":true}),
    ] {
        assert_eq!(post(&s, "/v1/chat/completions", chat_body(extra)).status, 400);
    }
}

#[test]
fn output_affecting_unsupported_fields_are_refused_by_name() {
    let s = start(Opts::default());
    for (field, v) in [
        ("response_format", json!({"type":"json_object"})),
        // `top_logprobs: 0` is now a valid off switch for the prompt_logprobs readout; the unsupported
        // shapes are the above-cap and non-numeric ones, refused by name below.
        ("top_logprobs", json!(21)),
        ("top_logprobs", json!("x")),
        ("parallel_tool_calls", json!(false)),
        ("logprobs", json!("yes")),
        ("logit_bias", json!([])),
        ("echo", json!(true)),
        ("suffix", json!("tail")),
        ("functions", json!([])),
        // (a top-level `reasoning_effort` is supported by this family since the reasoning controls: `nemotron_reasoning_api.rs`)
        ("reasoning_effort", json!("maximum")),
        ("verbosity", json!("low")),
        ("unknown_generation_knob", json!(7)),
    ] {
        let r = post(&s, "/v1/chat/completions", chat_body(json!({field:v, "max_tokens":1})));
        assert_eq!(r.status, 400, "{field}: {}", r.body);
        assert_eq!(r.json()["error"]["param"], field);
    }
}

#[test]
fn tool_message_and_schema_caps_apply_before_render_even_with_tools_none() {
    let s = start(Opts::default());
    let tool = json!({"type":"function", "function":{"name":"f", "parameters":{"type":"object"}}});
    for body in [
        chat_body(json!({"tools":vec![tool;65],"tool_choice":"none"})),
        chat_body(json!({"tools":[{"type":"function", "function":{"name":"f", "description":"x".repeat((1<<20)+1)}}]})),
        chat_body(json!({"messages":vec![json!({"role":"user","content":"hi"});10001]})),
    ] {
        let r = post(&s, "/v1/chat/completions/render", body);
        assert_eq!(r.status, 413, "{}", r.body);
        assert!(r.json()["error"]["param"].is_string());
    }
}

#[test]
fn messages_and_parts_must_not_disappear_in_templates() {
    let s = start(Opts::default());
    for messages in [
        json!([{"role":"developer-typo","content":"hi"}]),
        json!([{"role":"user","content":17}]),
        json!([{"role":"user","content":[{"type":"text","text":17}]}]),
        json!([{"role":"user","content":[{"type":"unknown","text":"hi"}]}]),
        json!([17]),
    ] {
        let r = post(&s, "/v1/chat/completions/render", chat_body(json!({"messages":messages})));
        assert_eq!(r.status, 400, "{}", r.body);
    }
}

#[test]
fn legitimate_default_tools_thinking_and_sdk_shape_still_work() {
    let s = start(Opts::default());
    for choice in ["auto", "none"] {
        let r = post(
            &s,
            "/v1/chat/completions",
            chat_body(json!({"max_tokens":1,
            "response_format":{"type":"text"}, "tool_choice":choice,
            "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{}}}}],
            "chat_template_kwargs":{"enable_thinking":false}, "user":"client", "metadata":{"trace":"local"}, "store":false})),
        );
        assert_eq!(r.status, 200, "{}", r.body);
    }
    let r = post(
        &s,
        "/v1/chat/completions/render",
        chat_body(json!({"messages":[{"role":"user","content":[{"type":"text","text":"plain text"}]}]})),
    );
    assert_eq!(r.status, 200, "{}", r.body);
}

use gb_server::budget::{validate_json, Limits, RequestBudget};
use gb_server::chat_format::ChatFormat;
use gb_server::template::TemplateError;
use std::io::{Read, Write};
use std::net::{Shutdown, TcpStream};

#[test]
fn json_gate_counts_values_keys_duplicates_not_escaped_punctuation() {
    let body = br#"{"k":"[,{\"\\\"}","k":[true,null,12.5e-1]}"#;
    assert_eq!(validate_json(body, 8, 128).unwrap(), 8);
    assert_eq!(validate_json(body, 7, 128).unwrap_err().status, 413);
    let scalar = json!("x".repeat(1 << 20)).to_string();
    assert_eq!(validate_json(scalar.as_bytes(), 1, 128).unwrap(), 1);
    for invalid in [
        b"[1,]".as_slice(),
        b"{\"k\":}",
        b"01",
        b"1e+",
        b"true false",
        b"\"bad\\x\"",
        b"[",
        b"{\"k\" 1}",
        b"[1,,2]",
        b"\"bad\n\"",
    ] {
        assert_eq!(validate_json(invalid, 100, 128).unwrap_err().status, 400, "{invalid:?}");
    }
}

#[test]
fn json_gate_handles_wide_objects_deep_arrays_and_generated_escaped_scalars() {
    for depth in [1, 64, 128] {
        let text = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        assert_eq!(validate_json(text.as_bytes(), depth + 1, 128).unwrap(), depth + 1);
        assert_eq!(validate_json(text.as_bytes(), depth + 1, depth - 1).unwrap_err().status, 413);
    }
    let wide = json!((0..300).map(|i| (format!("key-{i}"), json!(false))).collect::<serde_json::Map<_, _>>()).to_string();
    assert_eq!(validate_json(wide.as_bytes(), 601, 128).unwrap(), 601);
    assert_eq!(validate_json(wide.as_bytes(), 600, 128).unwrap_err().status, 413);
    // Quote/escape-aware lexical counting on thousands of generated valid strings.
    let alphabet = ['[', ']', '{', '}', ',', ':', '"', '\\', '\n', '\0', '\u{00e9}'];
    for seed in 0..1000 {
        let string: String = (0..32).map(|i| alphabet[(seed * 17 + i * 7) % alphabet.len()]).collect();
        let text = json!({"key": [string, null, true]}).to_string();
        assert!(serde_json::from_str::<Value>(&text).is_ok());
        assert_eq!(validate_json(text.as_bytes(), 6, 128).unwrap(), 6);
    }
}

#[test]
fn byte_credits_and_preparation_are_fail_fast_checked_and_released() {
    let budget = RequestBudget::new(Limits { total_bytes: 100, preparation_concurrency: 1, ..Limits::default() }).unwrap();
    let mut one = budget.lease();
    one.grow_to(60).unwrap();
    let mut two = budget.lease();
    assert_eq!(two.grow_to(41).unwrap_err().status, 429);
    assert_eq!(budget.used(), 60);
    one.grow_to(usize::MAX).unwrap_err();
    assert_eq!(budget.used(), 60);
    two.grow_to(40).unwrap();
    assert_eq!(budget.used(), 100);
    drop(one);
    assert_eq!(budget.used(), 40);
    drop(two);
    assert_eq!(budget.used(), 0);
    let permit = budget.prepare().unwrap();
    assert_eq!(budget.prepare().unwrap_err().status, 429);
    drop(permit);
    assert!(budget.prepare().is_ok());
    assert!(RequestBudget::new(Limits { preparation_concurrency: usize::MAX, ..Limits::default() }).is_err());
    assert!(RequestBudget::new(Limits { body_bytes: 0, ..Limits::default() }).is_err());
    assert!(
        RequestBudget::new(Limits { body_bytes: 128 << 20, ..Limits::default() }).is_err(),
        "larger raw preparation must require a larger reserve"
    );
    assert!(
        RequestBudget::new(Limits { json_nodes: usize::MAX, ..Limits::default() }).is_err(),
        "JSON expansion arithmetic must be checked"
    );
}

fn socket(port: u16) -> TcpStream {
    let s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    s
}
fn read_response(s: &mut TcpStream) -> String {
    let mut b = Vec::new();
    let _ = s.read_to_end(&mut b);
    String::from_utf8_lossy(&b).into_owned()
}
#[test]
fn tcp_declared_length_is_refused_before_any_body_is_read_and_stalls_are_charged() {
    let s = start(Opts { request_limits: Limits { body_bytes: 256, total_bytes: 256, ..Limits::default() }, ..Opts::default() });
    let mut conn = socket(s.port);
    conn.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: 257\r\n\r\n").unwrap();
    assert!(read_response(&mut conn).starts_with("HTTP/1.1 413"));
    let mut stalled = socket(s.port);
    stalled
        .write_all(
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: 200\r\n\r\n{",
        )
        .unwrap();
    s.wait(Duration::from_secs(5), "declared credit", |s| s.state.request_budget.used() == 200);
    let r = post(&s, "/v1/chat/completions", chat_body(json!({"max_tokens":1})));
    assert_eq!(r.status, 429, "{}", r.body);
    assert_eq!(r.header("retry-after"), Some("2"));
    drop(stalled);
    s.wait(Duration::from_secs(5), "dropped raw credit", |s| s.state.request_budget.used() == 0);
    assert_eq!(post(&s, "/v1/chat/completions", chat_body(json!({"max_tokens":1}))).status, 200);
}

#[test]
fn tcp_chunked_bytes_without_length_are_charged_incrementally_and_capped() {
    let s = start(Opts { request_limits: Limits { body_bytes: 256, total_bytes: 128, ..Limits::default() }, ..Opts::default() });
    let mut a = socket(s.port);
    a.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n40\r\n").unwrap();
    a.write_all(&[b' '; 64]).unwrap();
    a.write_all(b"\r\n").unwrap();
    s.wait(Duration::from_secs(5), "incremental credit", |s| s.state.request_budget.used() == 64);
    a.write_all(b"41\r\n").unwrap();
    let _ = a.write_all(&[b' '; 65]);
    let _ = a.write_all(b"\r\n0\r\n\r\n");
    let reply = read_response(&mut a);
    assert!(reply.starts_with("HTTP/1.1 429"), "{reply}");
    s.wait(Duration::from_secs(5), "overload credit return", |s| s.state.request_budget.used() == 0);
    // A separate server with enough global credit proves the per-body chunked cap, rather than global overload.
    let s = start(Opts { request_limits: Limits { body_bytes: 128, total_bytes: 1024, ..Limits::default() }, ..Opts::default() });
    let mut a = socket(s.port);
    a.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n81\r\n").unwrap();
    let _ = a.write_all(&[b' '; 129]);
    let _ = a.write_all(b"\r\n0\r\n\r\n");
    let reply = read_response(&mut a);
    assert!(reply.starts_with("HTTP/1.1 413"), "{reply}");
}

#[test]
fn tcp_preparation_overload_and_truncated_bodies_return_all_credits() {
    let s = start(Opts {
        request_limits: Limits { preparation_concurrency: 1, preparation_queue: 0, ..Limits::default() },
        ..Opts::default()
    });
    let held = s.state.request_budget.prepare().unwrap();
    assert_eq!(post(&s, "/v1/chat/completions", chat_body(json!({}))).status, 429);
    assert_eq!(s.state.request_budget.used(), 0);
    drop(held);
    let mut c = socket(s.port);
    c.write_all(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: 200\r\n\r\n{",
    )
    .unwrap();
    c.shutdown(Shutdown::Both).unwrap();
    drop(c);
    s.wait(Duration::from_secs(5), "truncated body release", |s| s.state.request_budget.used() == 0);
    assert_eq!(post(&s, "/v1/chat/completions", chat_body(json!({"max_tokens":1}))).status, 200);
}

#[test]
fn tcp_complexity_render_and_tokenizer_caps_and_model_aliases() {
    let s = start(Opts {
        model_aliases: vec!["client-model".into()],
        request_limits: Limits { json_nodes: 64, rendered_bytes: 512, ..Limits::default() },
        ..Opts::default()
    });
    assert_eq!(post(&s, "/v1/chat/completions", chat_body(json!({"model":"client-model","max_tokens":1}))).status, 200);
    let wide = chat_body(json!({"metadata":vec![true;100]}));
    assert_eq!(post(&s, "/v1/chat/completions/render", wide).status, 413);
    let r = post(&s, "/v1/chat/completions/render", chat_body(json!({"messages":[{"role":"user","content":"x".repeat(600)}]})));
    assert_eq!(r.status, 413, "{}", r.body);
    assert_eq!(post(&s, "/v1/completions", json!({"prompt":"x".repeat(513)})).status, 413);
    assert_eq!(s.state.request_budget.used(), 0);
}

#[test]
fn bounded_renderer_stops_during_construction_and_preserves_exact_small_output() {
    let f = gb_server::formats::nemotron_omni::OmniFormat::new("{% for i in range(1000) %}abcdef{% endfor %}").unwrap();
    assert_eq!(f.render_bounded(&json!([]), &serde_json::Map::new(), 100).unwrap_err(), TemplateError::Limit);
    let f = gb_server::formats::nemotron_omni::OmniFormat::new("{{ messages|tojson(indent=1000000000) }}").unwrap();
    assert!(f.render_bounded(&json!(["hi"]), &serde_json::Map::new(), 100).is_err());
    let f = gb_server::formats::nemotron_omni::OmniFormat::new("{{ messages|tojson }}").unwrap();
    assert_eq!(f.render_bounded(&json!(["x".repeat(1000)]), &serde_json::Map::new(), 100).unwrap_err(), TemplateError::Limit);
    let m = json!(["quote: \" and unicode: \u{00e9}"]);
    assert_eq!(f.render(&m, &serde_json::Map::new()).unwrap(), f.render_bounded(&m, &serde_json::Map::new(), 1024).unwrap());
    // Previous refusal must not leak its thread-local cap into subsequent templates.
    assert!(f.render(&json!(["x".repeat(1000)]), &serde_json::Map::new()).is_ok());
}

#[test]
fn default_budgets_allow_the_advertised_262k_text_context() {
    let s = start(Opts { max_model_len: 262144, kv_tokens: 262144, pool: true, ..Opts::default() });
    let r = post(&s, "/v1/completions", json!({"model":"mock","prompt":"x".repeat(262143),"max_tokens":1}));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.json()["usage"]["total_tokens"], 262144);
    // Full-window token-ID prompts are legitimate too: a 250,000-node default would reject the advertised window.
    let r = post(&s, "/v1/completions", json!({"model":"mock", "prompt":vec![120;262143], "max_tokens":1}));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(r.json()["usage"]["total_tokens"], 262144);
}

fn fixture_data_uri() -> String {
    let b =
        std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vision/check_512x512.png")).unwrap();
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::from("data:image/png;base64,");
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    s
}
struct CpuImageEncoder {
    entered: std::sync::Arc<std::sync::atomic::AtomicBool>,
    release: std::sync::Arc<std::sync::atomic::AtomicBool>,
}
impl gb_model::serve::ImageEncoder for CpuImageEncoder {
    fn encode(&mut self, pre: &gb_vision::Preprocessed) -> Result<gb_model::serve::EncodedImage, gb_model::error::ModelError> {
        use std::sync::atomic::Ordering::SeqCst;
        self.entered.store(true, SeqCst);
        while !self.release.load(SeqCst) {
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(gb_model::serve::EncodedImage {
            tokens: pre.n_patches() / 4,
            merged: (pre.grid.1 / 2, pre.grid.2 / 2),
            features: std::sync::Arc::new(()),
        })
    }
}
fn image_server(
    blocked: bool,
) -> (common::Server, std::sync::Arc<std::sync::atomic::AtomicBool>, std::sync::Arc<std::sync::atomic::AtomicBool>) {
    image_server_in(None, blocked)
}
/// [`image_server`] on the model folder `dir` (`None`: the default fixture); the Qwen image case runs on `common::qwen_fixture_dir()`.
fn image_server_in(
    dir: Option<std::path::PathBuf>,
    blocked: bool,
) -> (common::Server, std::sync::Arc<std::sync::atomic::AtomicBool>, std::sync::Arc<std::sync::atomic::AtomicBool>) {
    let entered = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let release = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(!blocked));
    let slot = gb_model::serve::VisionSlot::default();
    slot.set(std::sync::Arc::new(gb_model::serve::VisionService::new(
        Box::new(CpuImageEncoder { entered: entered.clone(), release: release.clone() }),
        4096,
    )))
    .unwrap();
    (start(Opts { vision_slot: Some(slot), dir, ..Opts::default() }), entered, release)
}
#[test]
fn four_data_uri_images_use_real_cpu_decode_and_mock_http_without_gpu() {
    let (s, entered, _) = image_server_in(Some(common::qwen_fixture_dir()), false);
    let part = json!({"type":"image_url","image_url":{"url":fixture_data_uri()}});
    let messages = json!([{"role":"user","content":[{"type":"text","text":"describe"},part,part,part,part]}]);
    let r = post(&s, "/v1/chat/completions", chat_body(json!({"messages":messages,"max_tokens":1})));
    assert_eq!(r.status, 200, "{}", r.body);
    assert!(entered.load(std::sync::atomic::Ordering::SeqCst));
    assert!(r.json()["usage"]["prompt_tokens"].as_u64().unwrap() > 1024);
    let r = post(
        &s,
        "/v1/chat/completions",
        chat_body(
            json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}]}),
        ),
    );
    assert_eq!(r.status, 400);
    assert!(r.body.contains("remote image fetch is disabled") && r.body.contains("messages[0].content[0]"), "{}", r.body);
}
#[test]
fn image_worker_keeps_shared_raw_preparation_accounting_after_tcp_disconnect() {
    let (s, entered, release) = image_server(true);
    let body = chat_body(
        json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":fixture_data_uri()}}]}],"max_tokens":1}),
    );
    let c = common::Conn::post(s.port, "/v1/chat/completions", &body);
    s.wait(Duration::from_secs(5), "CPU image barrier", |_| entered.load(std::sync::atomic::Ordering::SeqCst));
    drop(c);
    std::thread::sleep(Duration::from_millis(100));
    assert!(s.state.request_budget.used() >= body.to_string().len());
    // The raw/preparation lease belongs to the blocking worker regardless of Hyper handler cancellation semantics.
    release.store(true, std::sync::atomic::Ordering::SeqCst);
    s.wait(Duration::from_secs(5), "worker lease return", |s| s.state.request_budget.used() == 0);
}

#[test]
fn endpoint_specific_output_fields_cannot_be_silently_discarded() {
    let s = start(Opts::default());
    for (field, value) in [
        ("messages", json!([])),
        ("tools", json!([])),
        ("tool_choice", json!("none")),
        ("chat_template_kwargs", json!({"enable_thinking":false})),
        ("return_token_ids", json!(true)),
    ] {
        let r = post(&s, "/v1/completions", json!({"prompt":[1],"max_tokens":1,field:value}));
        assert_eq!(r.status, 400, "{field}: {}", r.body);
        assert_eq!(r.json()["error"]["param"], field);
    }
    let r = post(&s, "/v1/chat/completions", chat_body(json!({"prompt":"not part of this chat"})));
    assert_eq!(r.status, 400);
    assert_eq!(r.json()["error"]["param"], "prompt");
    for kw in
        [json!({"enable_thinking":"false"}), json!({"tools":[]}), json!({"messages":[]}), json!({"add_generation_prompt":false})]
    {
        assert_eq!(post(&s, "/v1/chat/completions/render", chat_body(json!({"chat_template_kwargs":kw}))).status, 400);
    }
}
