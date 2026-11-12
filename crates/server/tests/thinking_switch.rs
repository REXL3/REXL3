//! `chat_template_kwargs.enable_thinking: false` (serve hardening, finding 5). Qwen's chat template closes its reasoning block when the variable is
//! set; GLM-5.3's has no such switch: its generation prompt is always `<|assistant|><think>`, and the one reasoning variable it reads,
//! `reasoning_effort`, sets an effort level ("Reasoning Effort: Low|High|Max" in a system turn), it does not turn thinking off. So on GLM the request is
//! accepted, nothing is invented for the prompt (the exactness rule applies to the prompt the engine is given, and a prefilled empty block would be a
//! prompt the model's own template never produces), and the response says the setting had no effect (`X-Rexl3-Note`). These tests run the GLM template
//! of the pack (committed as a fixture, identical to the pack's `chat_template.jinja`) behind the mock backend.
//! Qwen's case runs the Qwen3.8 template (`tests/common::qwen_fixture_dir`); the Nemotron 3.5 Super VL template (`tests/common`'s default
//! fixture) honours the switch the same way.

mod common;

use common::{chat_body, http_timeout, start, Opts, Server};
use gb_engine::mock::MockOptions;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

/// A model folder that selects the GLM chat format: the mock tokenizer (byte-level: GLM's special tokens are plain text to it, which is all a mock
/// needs), the GLM-5.3 template and a config naming the architecture.
fn glm_dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let fx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let dir = std::env::temp_dir().join(format!("rexl3-mock-glm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(fx.join("mock_model/tokenizer.json"), dir.join("tokenizer.json")).unwrap();
        std::fs::copy(fx.join("glm53_chat_template.jinja"), dir.join("chat_template.jinja")).unwrap();
        std::fs::copy(fx.join("glm53_config_head.json"), dir.join("config.json")).unwrap();
        // an end-of-turn id the mock never produces: a reply runs to its max_tokens
        std::fs::write(dir.join("generation_config.json"), "{\"eos_token_id\": [100000], \"temperature\": 1.0}").unwrap();
        dir
    })
    .clone()
}

fn glm() -> Server {
    start(Opts { dir: Some(glm_dir()), mock: MockOptions { vocab: 4096, ..Default::default() }, ..Default::default() })
}

fn qwen() -> Server {
    start(Opts {
        dir: Some(common::qwen_fixture_dir()),
        mock: MockOptions { vocab: 4096, ..Default::default() },
        ..Default::default()
    })
}

fn nemo() -> Server {
    start(Opts { mock: MockOptions { vocab: 4096, ..Default::default() }, ..Default::default() })
}

fn chat(s: &Server, extra: Value) -> common::Reply {
    http_timeout(s.port, "POST", "/v1/chat/completions", Some(&chat_body(extra)), Duration::from_secs(30))
}

const OFF: fn() -> Value = || json!({"chat_template_kwargs": {"enable_thinking": false}});

#[test]
fn on_glm_enable_thinking_false_is_accepted_changes_nothing_and_the_response_says_so() {
    // mutation: drop `with_note` from the chat handler (no header), or make the note unconditional (a header on every GLM reply)
    let s = glm();
    let render = |extra: Value| {
        let r = http_timeout(s.port, "POST", "/v1/chat/completions/render", Some(&chat_body(extra)), Duration::from_secs(30));
        assert_eq!(r.status, 200, "{}", r.body);
        r.json()["token_ids"].clone()
    };
    // the prompt is the same token for token: there is no switch to throw
    assert_eq!(render(OFF()), render(json!({})));
    assert_eq!(render(OFF()), render(json!({"chat_template_kwargs": {"enable_thinking": true}})));
    // the reply starts inside the reasoning block either way, and with thinking off the response says the setting did nothing
    for stream in [false, true] {
        let r = chat(&s, json!({"chat_template_kwargs": {"enable_thinking": false}, "stream": stream}));
        assert_eq!(r.status, 200, "{}", r.body);
        let note = r.header("x-rexl3-note").unwrap_or_else(|| panic!("no X-Rexl3-Note: {:?}", r.headers));
        assert!(
            note.contains("enable_thinking=false has no effect") && note.contains("always opens a reasoning block"),
            "{note}"
        );
        if !stream {
            assert!(
                r.json()["choices"][0]["message"].get("reasoning_content").is_some(),
                "the reply began inside <think>: {}",
                r.body
            );
        }
    }
    // no note when nothing was asked for: the default, thinking on, and the template's own knob
    for extra in [
        json!({}),
        json!({"chat_template_kwargs": {"enable_thinking": true}}),
        json!({"chat_template_kwargs": {"reasoning_effort": "low"}}),
    ] {
        let r = chat(&s, extra.clone());
        assert_eq!(r.status, 200, "{}", r.body);
        assert!(r.header("x-rexl3-note").is_none(), "{extra}: {:?}", r.headers);
    }
}

#[test]
fn reasoning_effort_is_the_templates_knob_and_it_does_change_the_prompt() {
    // what a client that wants less thinking on GLM can send: a system marker the template renders, not a way to turn thinking off
    let s = glm();
    let render = |extra: Value| {
        http_timeout(s.port, "POST", "/v1/chat/completions/render", Some(&chat_body(extra)), Duration::from_secs(30)).json()
            ["token_ids"]
            .clone()
    };
    assert_ne!(render(json!({"chat_template_kwargs": {"reasoning_effort": "low"}})), render(json!({})));
}

#[test]
fn on_qwen_enable_thinking_false_closes_the_block_and_there_is_no_note() {
    // the other side of the same rule: where the template honours the switch the prompt changes and nothing is said
    let s = qwen();
    let off = chat(&s, OFF());
    assert_eq!(off.status, 200, "{}", off.body);
    assert!(off.header("x-rexl3-note").is_none(), "{:?}", off.headers);
    assert!(off.json()["choices"][0]["message"].get("reasoning_content").is_none(), "{}", off.body);
    let omitted = chat(&s, json!({}));
    assert!(
        omitted.json()["choices"][0]["message"].get("reasoning_content").is_none(),
        "omitted requests answer, they do not think: {}",
        omitted.body
    );
    let on = chat(&s, json!({"chat_template_kwargs": {"enable_thinking": true}}));
    assert!(on.header("x-rexl3-note").is_none());
    assert!(on.json()["choices"][0]["message"].get("reasoning_content").is_some(), "{}", on.body);
}

#[test]
fn enable_thinking_false_closes_the_block_and_there_is_no_note() {
    // the other side of the same rule: where the template honours the switch the prompt changes and nothing is said
    let s = nemo();
    let off = chat(&s, OFF());
    assert_eq!(off.status, 200, "{}", off.body);
    assert!(off.header("x-rexl3-note").is_none(), "{:?}", off.headers);
    assert!(off.json()["choices"][0]["message"].get("reasoning_content").is_none(), "{}", off.body);
    let omitted = chat(&s, json!({}));
    assert!(
        omitted.json()["choices"][0]["message"].get("reasoning_content").is_none(),
        "omitted requests answer, they do not think: {}",
        omitted.body
    );
    let on = chat(&s, json!({"chat_template_kwargs": {"enable_thinking": true}}));
    assert!(on.header("x-rexl3-note").is_none());
    assert!(on.json()["choices"][0]["message"].get("reasoning_content").is_some(), "{}", on.body);
}
