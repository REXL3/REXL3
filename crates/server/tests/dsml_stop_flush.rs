//! A reply that ends at a closed DSML tool-calls block (the DeepSeek-V4.1 tool format) keeps all of its text when the request
//! carries a `stop` list, through the real HTTP server, streamed and not.
//!
//! The stop matcher holds back the tail of the reply's text in case it begins a stop string. A closed DSML calls block ends the reply
//! there and then, before the engine's own end, so that tail is the end of the reply's text: it must go out, not be dropped (before
//! the fix a reply `Adding both.` + calls came back as `Adding ` under a six-byte stop string). The flush never emits a stop string:
//! a stop string that matches before the block still ends the reply at the match, as it always did.
//!
//! The engine emits a fixed token script (a sentence, a DSML block with two calls, stray text after the block, end of turn) under the
//! DeepSeek-V4.1 chat format with the mock tokenizer, so the test needs no pack and runs in CI.

mod common;

use common::{http, sse, Opts};
use gb_engine::mock::MockBackend;
use gb_engine::{Backend, Drafter, Engine, EngineConfig, ImageSpan, NoDrafts, PrefillStep, RowOutput, RowRequest, ServeEngine};
use gb_sched::{Admission, CostModel, ModelMemory};
use gb_server::formats::deepseek_v41::DeepseekV41Format;
use gb_server::tokenizer::Tok;
use serde_json::{json, Value};
use std::sync::Arc;

const EOS: u32 = 258; // the mock fixture's end of turn (an eos in its generation_config.json)
const MAX_TOKENS: usize = 2000; // well above the script, so no reply here ends on length

fn invoke(a: &str, b: &str) -> String {
    format!(
        "<｜DSML｜ invoke name=\"add\">\n<｜DSML｜ parameter name=\"a\" string=\"false\">{a}</｜DSML｜ parameter>\n\
         <｜DSML｜ parameter name=\"b\" string=\"false\">{b}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
    )
}

/// What the scripted model writes: a sentence, one DSML calls block with two calls, then text the closed block keeps out of the reply.
fn reply_text() -> String {
    format!(
        "Adding both.\n\n<｜DSML｜ calls>\n{}{}</｜DSML｜ calls>\n\nSTRAY TEXT AFTER THE BLOCK",
        invoke("2", "3"),
        invoke("10.5", "0.25")
    )
}

/// A backend that emits a fixed token script, one token per prefill/decode row, then the mock's own. The script starts over at each
/// request's first token (the prefill that completes its prompt), so every request gets the same reply.
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
        // a decode step of a few milliseconds, as a real one is: at full CPU speed the script fills the request's bounded reply
        // channel faster than a loaded test machine drains it, and the request ends as backpressure
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

/// A server with the DeepSeek-V4.1 chat format whose engine emits the scripted reply for every request it gets.
fn scripted_server() -> common::Server {
    let dir = common::fixture_dir();
    let tok = Tok::from_dir(&dir).unwrap();
    let vocab = tok.vocab;
    let mut script = tok.encode(&reply_text()).unwrap();
    // the script is the reply as written (the fullwidth DSML markers survive the mock tokenizer) and fits the request's budget
    assert_eq!(tok.decode(&script, false), reply_text());
    assert!(script.len() + 1 < MAX_TOKENS, "the script is {} tokens", script.len());
    script.push(EOS);
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
    common::start(Opts {
        max_model_len: 4096,
        dir: Some(dir),
        slots: 1,
        engine: Some(make),
        format: Some(Arc::new(DeepseekV41Format)),
        ..Default::default()
    })
}

/// A chat request offering the `add` tool, thinking off, with `stop` when given.
fn body(stream: bool, stop: Option<Value>) -> Value {
    let mut b = json!({
        "model": "mock",
        "messages": [{"role": "user", "content": "Add 2 and 3, and also add 10.5 and 0.25."}],
        "tools": [{"type": "function", "function": {"name": "add", "description": "Add two numbers.",
            "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}}],
        "chat_template_kwargs": {"enable_thinking": false},
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": stream,
    });
    if let Some(stop) = stop {
        b["stop"] = stop;
    }
    b
}

/// What a reply carried, the same way for both transports: (content, [(name, arguments as JSON)], finish_reason).
type Seen = (String, Vec<(String, Value)>, String);

fn ask(s: &common::Server, stream: bool, stop: Option<Value>) -> Seen {
    let (st, b) = http(s.port, "POST", "/v1/chat/completions", Some(&body(stream, stop)));
    assert_eq!(st, 200, "{b}");
    let call = |c: &Value| {
        let f = &c["function"];
        (f["name"].as_str().unwrap().to_string(), serde_json::from_str::<Value>(f["arguments"].as_str().unwrap()).unwrap())
    };
    if !stream {
        let v: Value = serde_json::from_str(&b).unwrap();
        let m = &v["choices"][0]["message"];
        let calls = m["tool_calls"].as_array().map(|a| a.iter().map(call).collect()).unwrap_or_default();
        let finish = v["choices"][0]["finish_reason"].as_str().unwrap().to_string();
        return (m["content"].as_str().unwrap_or("").to_string(), calls, finish);
    }
    let (mut content, mut calls, mut finish) = (String::new(), Vec::new(), String::new());
    for c in sse(&b) {
        assert!(c.get("error").is_none(), "the stream ended in an error: {c}");
        let Some(ch) = c["choices"].as_array().and_then(|a| a.first()) else { continue };
        let d = &ch["delta"];
        content.push_str(d["content"].as_str().unwrap_or(""));
        calls.extend(d["tool_calls"].as_array().into_iter().flatten().map(call));
        if let Some(f) = ch["finish_reason"].as_str() {
            finish = f.to_string();
        }
    }
    (content, calls, finish)
}

fn add(a: Value, b: Value) -> (String, Value) {
    ("add".to_string(), json!({"a": a, "b": b}))
}

/// Every stop list here leaves the reply's text unmatched: the reply is the one served without `stop`. Mismatches are collected so a
/// failure shows every case.
fn keeps_the_text_the_stop_matcher_held_back(stream: bool) {
    let s = scripted_server();
    let want =
        ("Adding both.".to_string(), vec![add(json!(2), json!(3)), add(json!(10.5), json!(0.25))], "tool_calls".to_string());
    // no stop list: the reply the stop list must not change (the stray text after the closed block is not part of it)
    assert_eq!(ask(&s, stream, None), want, "stream={stream}, no stop list");
    let mut wrong = Vec::new();
    for stop in [
        // six bytes: the matcher holds back the last five (`both.`), the reported `Adding ` shape
        json!(["</end>"]),
        json!("</end>"),
        // the held tail begins like the stop string and never completes it
        json!(["both!"]),
        // longer than the whole text: the matcher holds back all of it
        json!(["a stop string longer than the reply's text", "\u{2603}\u{2603}"]),
    ] {
        let got = ask(&s, stream, Some(stop.clone()));
        if got != want {
            wrong.push(format!("stop={stop}: {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "stream={stream}: want {want:?}, got {wrong:#?}");
    assert_eq!(s.in_flight(), 0);
}

#[test]
fn a_reply_that_ends_at_a_dsml_block_keeps_the_text_the_stop_matcher_held_back() {
    keeps_the_text_the_stop_matcher_held_back(false);
}

#[test]
fn a_streamed_reply_that_ends_at_a_dsml_block_keeps_the_text_the_stop_matcher_held_back() {
    keeps_the_text_the_stop_matcher_held_back(true);
}

#[test]
fn a_stop_string_before_the_dsml_block_still_ends_the_reply_at_the_match() {
    let s = scripted_server();
    for stream in [false, true] {
        for (stop, want) in [(json!(["both"]), "Adding "), (json!(["</end>", "ding b"]), "Ad"), (json!(["Adding both."]), "")] {
            let (content, calls, finish) = ask(&s, stream, Some(stop.clone()));
            assert_eq!(content, want, "stream={stream} stop={stop}");
            assert!(calls.is_empty(), "stream={stream} stop={stop}: the reply ended before the block");
            assert_eq!(finish, "stop", "stream={stream} stop={stop}");
        }
    }
    assert_eq!(s.in_flight(), 0);
}
