//! DSV4.1's chat template, rendered by this server's own template engine, against DeepSeek's reference encoder.
//!
//! The pack ships no `chat_template.jinja`; `testdata/dsv41/chat_template.json` holds our transcription of the text path of
//! the reference `encode_messages`, stored as an ASCII-escaped JSON string (the copy installed in the pack directory on
//! the Spark is the same text). The expected
//! prompts in `chat_expected.json` were produced by running the reference encoder itself (`drop_thinking=True`,
//! `add_default_bos_token=True`) on each conversation in both modes, so this pins byte-identical prompts, special tokens
//! included, for system / user / assistant / mid-conversation system turns.

use gb_server::chat_format::ChatFormat;
use gb_server::formats::qwen::QwenFormat;
use serde_json::{Map, Value};

/// The template as an ASCII-escaped JSON string: the repo never carries the raw special-token bytes.
const TEMPLATE_JSON: &str = include_str!("../../../testdata/dsv41/chat_template.json");
const EXPECTED: &str = include_str!("../../../testdata/dsv41/chat_expected.json");

fn template() -> String {
    let v: Value = serde_json::from_str(TEMPLATE_JSON).expect("template fixture is JSON");
    v["template"].as_str().expect("template string").to_owned()
}

#[test]
fn dsv41_template_renders_the_reference_encoding() {
    let format = QwenFormat::new(&template()).expect("the DSV4.1 template parses");
    let fixture: Value = serde_json::from_str(EXPECTED).expect("fixture is JSON");
    let cases = fixture["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 8);
    let mut failures = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let mut kwargs = Map::new();
        kwargs.insert("enable_thinking".into(), case["enable_thinking"].clone());
        let got = format.render(&case["messages"], &kwargs).map(|r| r.text);
        let want = case["expected"].as_str().unwrap();
        match got {
            Ok(text) if text == want => {}
            Ok(text) => failures.push(format!(
                "{name}: rendered {} bytes, reference {} bytes; first difference at byte {}",
                text.len(),
                want.len(),
                text.bytes().zip(want.bytes()).position(|(a, b)| a != b).unwrap_or(text.len().min(want.len()))
            )),
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn dsv41_generation_prompt_opens_reasoning_only_in_thinking_mode() {
    let format = QwenFormat::new(&template()).unwrap();
    let messages = serde_json::json!([{"role": "user", "content": "What is 17 * 23?"}]);
    for (thinking, opens) in [(false, false), (true, true)] {
        let mut kwargs = Map::new();
        kwargs.insert("enable_thinking".into(), Value::Bool(thinking));
        let r = format.render(&messages, &kwargs).unwrap();
        assert_eq!(r.starts_in_reasoning, opens, "enable_thinking={thinking}");
    }
}
