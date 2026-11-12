//! What a client sees when a request cannot be served: the machine-readable `error.code` OpenAI clients act on. A prompt past the
//! context window must say `context_length_exceeded` (and which field is too long), so an agent compacts its history instead of
//! retrying; every other error keeps the numeric status as its `code` and gains no `param`.

mod common;

use common::{chat_body, completion_body, http_timeout, start, Opts, Server};
use serde_json::{json, Value};
use std::time::Duration;

const WINDOW: usize = 512;

fn server() -> Server {
    start(Opts {
        max_model_len: WINDOW,
        mock: gb_engine::mock::MockOptions { vocab: 4096, ..Default::default() },
        ..Default::default()
    })
}

fn post(s: &Server, path: &str, body: &Value) -> common::Reply {
    http_timeout(s.port, "POST", path, Some(body), Duration::from_secs(30))
}

/// The error object of a refusal, after checking it is a 400 JSON body (never a stream).
fn refusal(r: &common::Reply) -> Value {
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(!r.body.contains("data:"), "an error is one JSON object, not an event stream: {}", r.body);
    r.json()["error"].clone()
}

fn assert_context_error(e: &Value, param: &str) {
    assert_eq!(e["code"], "context_length_exceeded", "{e}");
    assert_eq!(e["type"], "invalid_request_error", "{e}");
    assert_eq!(e["param"], param, "{e}");
    let m = e["message"].as_str().unwrap();
    assert!(m.starts_with(&format!("This model's maximum context length is {WINDOW} tokens")), "{m}");
}

#[test]
fn a_chat_prompt_past_the_window_says_context_length_exceeded_and_names_messages() {
    let s = server();
    let long = json!({"model": "mock", "messages": [{"role": "user", "content": "word ".repeat(2000)}]});
    assert_context_error(&refusal(&post(&s, "/v1/chat/completions", &long)), "messages");
    // the same body asking for a stream is refused the same way, before any event
    let mut streaming = long.clone();
    streaming["stream"] = json!(true);
    streaming["stream_options"] = json!({"include_usage": true});
    let r = post(&s, "/v1/chat/completions", &streaming);
    assert_context_error(&refusal(&r), "messages");
    assert!(r.header("content-type").unwrap_or("").starts_with("application/json"), "{:?}", r.headers);
    // and the engine is untouched
    assert_eq!(post(&s, "/v1/chat/completions", &chat_body(json!({}))).status, 200);
}

#[test]
fn a_completion_prompt_past_the_window_names_the_prompt_field() {
    let s = server();
    for stream in [false, true] {
        let r = post(&s, "/v1/completions", &completion_body(&vec![5u32; WINDOW + 40], 1, stream));
        assert_context_error(&refusal(&r), "prompt");
    }
    let by_text = json!({"model": "mock", "prompt": "word ".repeat(2000), "max_tokens": 4});
    assert_context_error(&refusal(&post(&s, "/v1/completions", &by_text)), "prompt");
}

#[test]
fn the_window_edge_is_exact() {
    let s = server();
    // a prompt of WINDOW - 1 tokens leaves room for exactly one new token; WINDOW tokens leave none
    let ok = post(&s, "/v1/completions", &completion_body(&vec![5u32; WINDOW - 1], 1, false));
    assert_eq!(ok.status, 200, "{}", ok.body);
    for n in [WINDOW, WINDOW + 1] {
        let r = post(&s, "/v1/completions", &completion_body(&vec![5u32; n], 1, false));
        assert_context_error(&refusal(&r), "prompt");
        assert!(r.json()["error"]["message"].as_str().unwrap().contains(&format!("{n} prompt tokens")), "{}", r.body);
    }
}

// ---- the parser and extractor layer (serve hardening, finding 4): a body that is not JSON used to get axum's plain-text 400 ("Failed to parse
// the request body as JSON: ..."), a missing content type a plain 415, an oversized body a plain 413 and an unknown route an empty 404.
// Every refusal of this layer now has the engine's own shape: `{"error": {"message", "type": "invalid_request_error", "code": <status>}}`.

/// The reply is a JSON error object of the engine's shape, with this status.
fn assert_structured(r: &common::Reply, status: u16) -> Value {
    assert_eq!(r.status, status, "{}", r.body);
    assert!(
        r.header("content-type").unwrap_or("").starts_with("application/json"),
        "the error is not JSON: {:?} {}",
        r.headers,
        r.body
    );
    let e = r.json()["error"].clone();
    assert_eq!((e["type"].as_str(), e["code"].clone()), (Some("invalid_request_error"), json!(status)), "{e}");
    assert!(e["message"].as_str().is_some_and(|m| !m.is_empty()), "{e}");
    assert!(e.get("param").is_none(), "no field is known for this one: {e}");
    e
}

const JSON: [(&str, &str); 1] = [("Content-Type", "application/json")];

#[test]
fn malformed_json_gets_the_engines_error_shape_on_every_endpoint() {
    // mutation: take `Json<Value>` back in one handler in http.rs and that endpoint answers a plain-text 400 again
    let s = server();
    let bad: [&[u8]; 4] = [b"{\"model\": \"mock\", \"messages\": [", b"", b"not json at all", b"{\"a\": 1,}"];
    for path in ["/v1/chat/completions", "/v1/completions", "/v1/chat/completions/render"] {
        for body in bad {
            let r = common::raw(s.port, "POST", path, &JSON, body, Duration::from_secs(10));
            let e = assert_structured(&r, 400);
            assert!(e["message"].as_str().unwrap().contains("JSON"), "{path}: {e}");
        }
    }
    // and a well-formed request still works afterwards
    assert_eq!(post(&s, "/v1/chat/completions", &chat_body(json!({}))).status, 200);
}

#[test]
fn a_body_without_a_json_content_type_is_a_structured_415() {
    let s = server();
    for headers in [&[][..], &[("Content-Type", "text/plain")][..]] {
        let r = common::raw(s.port, "POST", "/v1/chat/completions", headers, b"{\"model\": \"mock\"}", Duration::from_secs(10));
        let e = assert_structured(&r, 415);
        assert!(e["message"].as_str().unwrap().contains("application/json"), "{e}");
    }
}

#[test]
fn a_body_past_the_limit_is_a_structured_413() {
    // the limit is 64 MiB (`router`): a body that goes past it is refused as it is read, with the shape of every other refusal
    let s = server();
    let body = vec![b' '; (64usize << 20) + 4096];
    let r = common::raw(s.port, "POST", "/v1/chat/completions", &JSON, &body, Duration::from_secs(30));
    let e = assert_structured(&r, 413);
    assert!(e["message"].as_str().unwrap().contains("64 MiB"), "{e}");
}

#[test]
fn an_unknown_route_and_a_wrong_method_are_structured_errors_too() {
    let s = server();
    let r = common::raw(s.port, "GET", "/v1/nope", &[], b"", Duration::from_secs(10));
    let e = assert_structured(&r, 404);
    assert!(e["message"].as_str().unwrap().contains("/v1/nope"), "{e}");
    let r = common::raw(s.port, "POST", "/v1/models", &JSON, b"{}", Duration::from_secs(10));
    let e = assert_structured(&r, 405);
    assert!(e["message"].as_str().unwrap().contains("POST") && e["message"].as_str().unwrap().contains("/v1/models"), "{e}");
    assert!(r.header("allow").is_some_and(|a| a.contains("GET")), "the Allow header survives: {:?}", r.headers);
    // the health routes are untouched
    assert_eq!(common::raw(s.port, "GET", "/health", &[], b"", Duration::from_secs(10)).status, 200);
}

#[test]
fn other_errors_keep_the_numeric_code_and_report_known_fields() {
    let s = server();
    // Strict count/message policy now names the known invalid field, rather than hiding it.
    for (body, param) in [
        (chat_body(json!({"n": 2})), None),
        (chat_body(json!({"max_tokens": 0})), Some("max_tokens")),
        (json!({"model": "mock"}), Some("messages")),
    ] {
        let r = post(&s, "/v1/chat/completions", &body);
        assert_eq!(r.status, 400, "{}", r.body);
        let e = r.json()["error"].clone();
        assert_eq!(e["code"], 400, "{e}");
        assert_eq!(e["type"], "invalid_request_error", "{e}");
        assert_eq!(e.get("param").and_then(Value::as_str), param, "{e}");
    }
}
