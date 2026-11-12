//! `usage.prompt_tokens_details.cached_tokens` and `usage.completion_tokens_details.reasoning_tokens` on every reply, streamed or not,
//! chat or plain completion. The cached count is 0 until a prefix cache supplies prompt tokens; the reasoning count is the number of
//! reply tokens through the token that closes the thought.
//!
//! The mock model "continues the latest earlier occurrence of the last bigram", so a user message that holds `<think>\nab cd</think>`
//! makes it repeat that thought after the generation prompt's own `<think>\n`: the reply is `a b _ c d </think> ...`, with a known
//! position of the closing token.

mod common;

use common::{chat_body, completion_body, http, sse, start, Opts, Server};
use serde_json::{json, Value};

fn server() -> Server {
    start(Opts {
        max_model_len: 4096,
        mock: gb_engine::mock::MockOptions { vocab: 4096, ..Default::default() },
        ..Default::default()
    })
}

fn thinking_chat(extra: Value) -> Value {
    // thinking is off unless asked for: the thinking cases say so, and `extra` may still switch it off
    let mut b = chat_body(json!({
        "max_tokens": 12,
        "return_token_ids": true,
        "chat_template_kwargs": {"enable_thinking": true}
    }));
    b["messages"] = json!([{"role": "user", "content": "<think>\nab cd</think>\nxyz"}]);
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

/// The usage block and the reply's token ids of a non-streamed chat reply.
fn plain(s: &Server, body: &Value) -> (Value, Vec<u64>) {
    let (st, b) = http(s.port, "POST", "/v1/chat/completions", Some(body));
    assert_eq!(st, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    let ids = v["choices"][0]["token_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    (v["usage"].clone(), ids)
}

/// The same for a streamed one: the usage chunk and the ids of every delta.
fn streamed(s: &Server, body: &Value) -> (Value, Vec<u64>) {
    let mut body = body.clone();
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    let (st, b) = http(s.port, "POST", "/v1/chat/completions", Some(&body));
    assert_eq!(st, 200, "{b}");
    let chunks = sse(&b);
    let usage = chunks.iter().rev().find(|c| c.get("usage").is_some()).expect("a usage chunk")["usage"].clone();
    let ids = chunks
        .iter()
        .filter_map(|c| c["choices"].get(0).and_then(|ch| ch.get("token_ids")))
        .flat_map(|t| t.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()))
        .collect();
    (usage, ids)
}

fn close_id(s: &Server) -> u64 {
    s.state.tok.token_to_id("</think>").expect("the fixture tokenizer has </think>") as u64
}

#[test]
fn the_reasoning_count_runs_through_the_closing_marker_in_every_reply_shape() {
    let s = server();
    let close = close_id(&s);
    for (name, (usage, ids)) in
        [("plain", plain(&s, &thinking_chat(json!({})))), ("stream", streamed(&s, &thinking_chat(json!({}))))]
    {
        let at =
            ids.iter().position(|&t| t == close).unwrap_or_else(|| panic!("{name}: the reply never closes its thought: {ids:?}"));
        assert!(at >= 2, "{name}: a thought of several tokens, not only the marker: {ids:?}");
        assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], at + 1, "{name}: {usage}");
        assert_eq!(usage["completion_tokens"], 12, "{name}: {usage}");
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 0, "{name}: {usage}");
        // the old fields and the speculation block are where they were
        assert_eq!(usage["total_tokens"], usage["prompt_tokens"].as_u64().unwrap() + 12, "{name}: {usage}");
        assert!(usage["spec"]["rounds"].is_number() && usage["spec"]["drafted"].is_number(), "{name}: {usage}");
    }
}

#[test]
fn a_thought_that_is_cut_off_is_all_reasoning_and_thinking_off_has_none() {
    let s = server();
    // three tokens of reply, all inside the thought
    for (usage, _) in
        [plain(&s, &thinking_chat(json!({"max_tokens": 3}))), streamed(&s, &thinking_chat(json!({"max_tokens": 3})))]
    {
        assert_eq!(usage["completion_tokens"], 3, "{usage}");
        assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 3, "{usage}");
    }
    // thinking switched off: the generation prompt holds the closed block, so the reply is all answer
    let off = thinking_chat(json!({"chat_template_kwargs": {"enable_thinking": false}}));
    for (usage, _) in [plain(&s, &off), streamed(&s, &off)] {
        assert_eq!(usage["completion_tokens"], 12, "{usage}");
        assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 0, "{usage}");
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 0, "{usage}");
    }
}

#[test]
fn plain_completions_carry_both_details_with_no_reasoning() {
    let s = server();
    for stream in [false, true] {
        let mut body = completion_body(&[5, 6, 7, 8], 6, stream);
        if stream {
            body["stream_options"] = json!({"include_usage": true});
        }
        let (st, b) = http(s.port, "POST", "/v1/completions", Some(&body));
        assert_eq!(st, 200, "{b}");
        let usage = if stream {
            sse(&b).iter().rev().find(|c| c.get("usage").is_some()).expect("a usage chunk")["usage"].clone()
        } else {
            serde_json::from_str::<Value>(&b).unwrap()["usage"].clone()
        };
        assert_eq!(usage["completion_tokens"], 6, "stream {stream}: {usage}");
        assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 0, "stream {stream}: {usage}");
        assert_eq!(usage["prompt_tokens_details"]["cached_tokens"], 0, "stream {stream}: {usage}");
    }
}
