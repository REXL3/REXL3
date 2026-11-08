//! Historical Q200 request compatibility: remove obsolete ignored alias without changing prompt bytes.
use gb_server::chat_format::ChatFormat;
use gb_server::formats::qwen::QwenFormat;
use serde_json::json;
#[test]
fn obsolete_thinking_alias_removal_preserves_actual_qwen_template_bytes() {
    let format = QwenFormat::new(include_str!("fixtures/qwen38_chat_template.jinja")).unwrap();
    for content in
        ["Explain a reliable release in English.", "Write a Python function with a test.", "Return a concise factual answer."]
    {
        let messages = json!([{"role":"user","content":content}]);
        let old = json!({"enable_thinking":true,"thinking":true,"reasoning_effort":"low"});
        let new = json!({"enable_thinking":true,"reasoning_effort":"low"});
        let a = format.render_bounded(&messages, old.as_object().unwrap(), 8 << 20).unwrap();
        let b = format.render_bounded(&messages, new.as_object().unwrap(), 8 << 20).unwrap();
        assert_eq!(a.text.as_bytes(), b.text.as_bytes());
        assert_eq!(a.starts_in_reasoning, b.starts_in_reasoning);
        assert!(b.starts_in_reasoning);
    }
}
