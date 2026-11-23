//! The `parallel_tool_calls` request field, through the real HTTP server. Some client libraries add it to tool requests on their
//! own, so the server takes it instead of refusing it:
//!
//! - absent, null or `true`: the reply is exactly what it was before the field was accepted (every call the model makes);
//! - `false`: the reply ends at its first complete tool call, with `finish_reason` `tool_calls`; the text before the call and the
//!   call itself are the same bytes the uncapped reply carries;
//! - any other type: 400 naming the field; `/v1/completions` (no tools) refuses it by name.
//!
//! The engine emits a fixed token script (two function-block calls, then end of turn) under the Qwen chat format, so the test
//! needs no pack and runs in CI.

mod common;

use common::{http, sse, Opts};
use gb_engine::mock::MockBackend;
use gb_engine::{Backend, Drafter, Engine, EngineConfig, ImageSpan, NoDrafts, PrefillStep, RowOutput, RowRequest, ServeEngine};
use gb_sched::{Admission, CostModel, ModelMemory};
use gb_server::tokenizer::Tok;
use serde_json::{json, Value};
use std::sync::Arc;

const IM_END: u32 = 258; // the mock fixture's end of turn (an eos in its generation_config.json)

fn call_text(a: &str, b: &str) -> String {
    format!("<tool_call>\n<function=add>\n<parameter=a>\n{a}\n</parameter>\n<parameter=b>\n{b}\n</parameter>\n</function>\n</tool_call>")
}

/// What the scripted model writes: a sentence, two calls, then end of turn.
fn reply_text() -> String {
    format!("Adding both.\n\n{}\n{}", call_text("2", "3"), call_text("10.5", "0.25"))
}

/// A backend that emits a fixed token script, one token per prefill/decode row (as `role_stop.rs` does), then the mock's own. The
/// script starts over at each request's first token (the prefill that completes its prompt), so every request gets the same reply.
struct ScriptBackend {
    inner: MockBackend,
    script: Vec<u32>,
    at: usize,
}

impl ScriptBackend {
    fn scripted(&mut self, fallback: RowOutput) -> RowOutput {
        if self.at < self.script.len() {
            let t = self.script[self.at];
            self.at += 1;
            RowOutput { ids: vec![t], values: vec![30.0] }
        } else {
            fallback
        }
    }
}

impl Backend for ScriptBackend {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        let r = self.inner.prefill(slot, start, tokens, candidates);
        self.at = 0;
        self.scripted(r)
    }
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> PrefillStep {
        let mut step = self.inner.prefill_slice(slot, prompt, images, done, candidates);
        if let Some(row) = step.row.take() {
            self.at = 0;
            step.row = Some(self.scripted(row));
        }
        step
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        // a decode step of a few milliseconds, as a real one is: at full CPU speed the ~200-token script fills the request's
        // bounded reply channel (64 messages) faster than a loaded test machine drains it, and the request ends as backpressure
        std::thread::sleep(std::time::Duration::from_millis(2));
        let outs = self.inner.step(reqs, candidates);
        outs.into_iter().map(|rows| rows.into_iter().map(|r| self.scripted(r)).collect()).collect()
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.inner.commit(slot, keep)
    }
    fn commit_path(&mut self, slot: usize, path: &[usize]) {
        self.inner.commit_path(slot, path)
    }
    fn release(&mut self, slot: usize) {
        self.inner.release(slot)
    }
}

/// The scripted reply's token ids, and a server (Qwen format) whose engine emits them for every request it gets.
fn scripted_server() -> (Vec<u32>, common::Server) {
    let dir = common::qwen_fixture_dir();
    let tok = Tok::from_dir(&dir).unwrap();
    let vocab = tok.vocab;
    let mut script = tok.encode(&reply_text()).unwrap();
    script.push(IM_END);
    let ids = script.clone();
    let make = Arc::new(move || {
        let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1024, state_bytes_per_slot: 1 << 20, activation_reserve: 0 };
        let admission = Admission::new(1u64 << 30, 0, mm, 1).unwrap();
        let backend =
            ScriptBackend { inner: MockBackend::new(vocab, 1, gb_engine::mock::Fault::None), script: script.clone(), at: 0 };
        let engine = Engine::new(
            backend,
            admission,
            EngineConfig { max_rows: 64, max_drafts: 7, cost: CostModel::from_table(vec![(1, 1.0), (64, 2.0)]), drafting: false },
            Box::new(|| Box::new(NoDrafts) as Box<dyn Drafter>),
        );
        Box::new(engine) as Box<dyn ServeEngine>
    });
    let s = common::start(Opts { max_model_len: 4096, dir: Some(dir), slots: 1, engine: Some(make), ..Default::default() });
    (ids, s)
}

/// A chat request offering the `add` tool, thinking off; `extra` adds or overrides fields.
fn body(stream: bool, extra: Value) -> Value {
    let mut b = json!({
        "model": "mock",
        "messages": [{"role": "user", "content": "Add 2 and 3, and also add 10.5 and 0.25."}],
        "tools": [{"type": "function", "function": {"name": "add", "description": "Add two numbers.",
            "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}}],
        "chat_template_kwargs": {"enable_thinking": false},
        "max_tokens": 400,
        "temperature": 0.0,
        "stream": stream,
        "stream_options": {"include_usage": true}
    });
    if !stream {
        b.as_object_mut().unwrap().remove("stream_options");
    }
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

/// What a reply carried, the same way for both transports: (content, [(name, arguments as JSON)], finish_reason, completion tokens).
type Seen = (String, Vec<(String, Value)>, String, u64);

fn ask(s: &common::Server, stream: bool, extra: Value) -> Seen {
    let (st, b) = http(s.port, "POST", "/v1/chat/completions", Some(&body(stream, extra)));
    assert_eq!(st, 200, "{b}");
    if !stream {
        let v: Value = serde_json::from_str(&b).unwrap();
        let m = &v["choices"][0]["message"];
        let calls = m["tool_calls"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|c| {
                        let f = &c["function"];
                        (f["name"].as_str().unwrap().to_string(), serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap())
                    })
                    .collect()
            })
            .unwrap_or_default();
        let finish = v["choices"][0]["finish_reason"].as_str().unwrap().to_string();
        return (
            m["content"].as_str().unwrap_or("").to_string(),
            calls,
            finish,
            v["usage"]["completion_tokens"].as_u64().unwrap(),
        );
    }
    let (mut content, mut calls, mut finish, mut used) = (String::new(), Vec::new(), String::new(), 0);
    let mut indexes = Vec::new();
    for c in sse(&b) {
        assert!(c.get("error").is_none(), "the stream ended in an error: {c}");
        if let Some(u) = c["usage"]["completion_tokens"].as_u64() {
            used = u;
        }
        let Some(ch) = c["choices"].as_array().and_then(|a| a.first()) else { continue };
        let d = &ch["delta"];
        content.push_str(d["content"].as_str().unwrap_or(""));
        for tc in d["tool_calls"].as_array().into_iter().flatten() {
            indexes.push(tc["index"].as_u64().unwrap());
            let f = &tc["function"];
            calls
                .push((f["name"].as_str().unwrap().to_string(), serde_json::from_str(f["arguments"].as_str().unwrap()).unwrap()));
        }
        if let Some(f) = ch["finish_reason"].as_str() {
            finish = f.to_string();
        }
    }
    assert_eq!(indexes, (0..calls.len() as u64).collect::<Vec<_>>(), "tool call deltas are indexed from 0");
    (content, calls, finish, used)
}

fn add(a: Value, b: Value) -> (String, Value) {
    ("add".to_string(), json!({"a": a, "b": b}))
}

#[test]
fn absent_null_and_true_return_every_call_as_before() {
    let (script, s) = scripted_server();
    for stream in [false, true] {
        let absent = ask(&s, stream, json!({}));
        assert_eq!(absent.0, "Adding both.", "stream {stream}");
        assert_eq!(absent.1, vec![add(json!(2), json!(3)), add(json!(10.5), json!(0.25))], "stream {stream}");
        assert_eq!(absent.2, "tool_calls", "stream {stream}");
        assert_eq!(absent.3, script.len() as u64, "stream {stream}: the whole script, end of turn included");
        // null and true are the same request as leaving the field out
        assert_eq!(ask(&s, stream, json!({"parallel_tool_calls": null})), absent, "stream {stream}");
        assert_eq!(ask(&s, stream, json!({"parallel_tool_calls": true})), absent, "stream {stream}");
    }
}

#[test]
fn false_ends_the_reply_at_its_first_call() {
    let (script, s) = scripted_server();
    for stream in [false, true] {
        let all = ask(&s, stream, json!({}));
        let one = ask(&s, stream, json!({"parallel_tool_calls": false}));
        assert_eq!(one.0, all.0, "stream {stream}: the text before the call is unchanged");
        assert_eq!(one.1, all.1[..1], "stream {stream}: only the first call, unchanged");
        assert_eq!(one.2, "tool_calls", "stream {stream}");
        assert!(one.3 < script.len() as u64, "stream {stream}: generation stopped after the first call ({} tokens)", one.3);
    }
}

#[test]
fn false_keeps_the_text_a_stop_list_was_holding_back() {
    // a stop matcher holds back the reply's last bytes (one fewer than its longest stop string) in case they begin a match; ending
    // the reply at the first call must still hand them out
    let (_, s) = scripted_server();
    let stop = json!({"stop": ["both.X", "never"]});
    for stream in [false, true] {
        let all = ask(&s, stream, stop.clone());
        assert_eq!(all.0, "Adding both.", "stream {stream}");
        let mut one = stop.clone();
        one["parallel_tool_calls"] = json!(false);
        let one = ask(&s, stream, one);
        assert_eq!(one.0, all.0, "stream {stream}");
        assert_eq!(one.1, all.1[..1], "stream {stream}");
        assert_eq!(one.2, "tool_calls", "stream {stream}");
    }
}

#[test]
fn false_without_a_call_or_without_tools_is_a_normal_reply() {
    let s = common::start(Opts::default());
    let r = common::chat_body(json!({"parallel_tool_calls": false}));
    let (st, b) = http(s.port, "POST", "/v1/chat/completions", Some(&r));
    assert_eq!(st, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    assert!(v["choices"][0]["message"]["tool_calls"].is_null(), "{v}");
    assert_ne!(v["choices"][0]["finish_reason"], "tool_calls", "{v}");
    let r = common::chat_body(json!({"parallel_tool_calls": false, "tool_choice": "none",
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]}));
    assert_eq!(http(s.port, "POST", "/v1/chat/completions", Some(&r)).0, 200);
}

#[test]
fn a_wrong_type_is_refused_naming_the_field() {
    let s = common::start(Opts::default());
    for v in [json!("false"), json!(0), json!(1), json!([]), json!({})] {
        for path in ["/v1/chat/completions", "/v1/chat/completions/render"] {
            let (st, b) = http(s.port, "POST", path, Some(&common::chat_body(json!({"parallel_tool_calls": v}))));
            assert_eq!(st, 400, "{path} {v}: {b}");
            let e: Value = serde_json::from_str(&b).unwrap();
            assert_eq!(e["error"]["param"], "parallel_tool_calls", "{path} {v}: {b}");
        }
    }
}

#[test]
fn plain_completions_refuse_the_field_by_name() {
    let s = common::start(Opts::default());
    for v in [json!(true), json!(false), json!("x")] {
        let mut b = common::completion_body(&[1, 2, 3], 1, false);
        b["parallel_tool_calls"] = v.clone();
        let (st, body) = http(s.port, "POST", "/v1/completions", Some(&b));
        assert_eq!(st, 400, "{v}: {body}");
        let e: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(e["error"]["param"], "parallel_tool_calls", "{v}: {body}");
    }
}
