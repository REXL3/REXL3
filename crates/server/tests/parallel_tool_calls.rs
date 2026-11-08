//! Parallel tool-call shapes through the real Flash-Next (Qwen-family) tool parser.
//!
//! The baseline gate's stage-3 check "parallel calls in order with typed numbers"
//! (scripts/gates/gb10_baseline.py:1105-1127) sends tools `[get_weather, add,
//! report_codes, describe_image, lookup]` and the user message
//! "Add 2 and 3, and also add 10.5 and 0.25. Call the add tool twice.", then
//! asserts exactly two `tool_calls`, both named `add`, arguments JSON-parseable,
//! in order `[{"a":2,"b":3},{"a":10.5,"b":0.25}]`.
//!
//! Flash-Next serves the `qwen` chat format (crates/model/src/families/
//! qwen_flash_next/mod.rs:64), whose tool syntax is the function block
//! `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`
//! (crates/server/src/tools.rs FUNCTION_BLOCK_TOOL_SYNTAX), and the pack's
//! chat_template.jinja tells the model to emit exactly that. These tests feed
//! plausible model completions through the real `ToolSplitter` (the same code
//! both streamed and non-streamed responses go through, http.rs:929/976-979)
//! and judge each shape against the gate's assertions, classifying pass,
//! parser bug, or model mistake.

use gb_server::tools::{ToolCall, ToolSchemas, ToolSplitter};
use serde_json::{json, Value};

/// The gate's TOOLS (gb10_baseline.py:984-1005), verbatim.
fn gate_schemas() -> ToolSchemas {
    let body = json!({"tools": [
        {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
         "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer", "description": "Forecast days, 1 to 5"},
            "celsius": {"type": "boolean"}},
            "required": ["city"]}}},
        {"type": "function", "function": {"name": "add", "description": "Add two numbers.",
         "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                        "required": ["a", "b"]}}},
        {"type": "function", "function": {"name": "report_codes", "description": "Report the codes found in the record.",
         "parameters": {"type": "object", "properties": {
            "codes": {"type": "array", "items": {"type": "string"}}},
            "required": ["codes"]}}},
        {"type": "function", "function": {"name": "describe_image", "description": "Describe the attached image.",
         "parameters": {"type": "object", "properties": {
            "colour": {"type": "string", "description": "Dominant colour of the image"}},
            "required": ["colour"]}}},
        {"type": "function", "function": {"name": "lookup", "description": "Look up the token for a step.",
         "parameters": {"type": "object", "properties": {"step": {"type": "integer"}},
                        "required": ["step"]}}},
    ]});
    ToolSchemas::from_request(&body).unwrap()
}

/// Run the whole completion through the splitter exactly as the server does:
/// feed decoded text in the same chunk sizes (streaming) and conclude with
/// `finish`. Returns (visible text, calls).
fn run(schemas: &ToolSchemas, text: &str, chunk: usize) -> (String, Vec<ToolCall>) {
    let mut sp = ToolSplitter::new(schemas.clone());
    let mut out = String::new();
    let mut calls = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let end = (i + chunk).min(chars.len());
        let piece: String = chars[i..end].iter().collect();
        let (t, c) = sp.push(&piece);
        out.push_str(&t);
        calls.extend(c);
        i = end;
    }
    let (t, c) = sp.finish();
    out.push_str(&t);
    calls.extend(c);
    (out, calls)
}

fn parsed_args(call: &ToolCall) -> Value {
    serde_json::from_str(&call.arguments).unwrap()
}

/// The gate's assertion, verbatim (gb10_baseline.py:1117-1118).
fn gate_good(calls: &[ToolCall]) -> bool {
    let got: Vec<Value> = calls.iter().map(parsed_args).collect();
    calls.len() == 2
        && calls.iter().all(|c| c.name == "add")
        && got == vec![json!({"a": 2, "b": 3}), json!({"a": 10.5, "b": 0.25})]
}

fn block(a: &str, b: &str) -> String {
    format!(
        "<tool_call>\n<function=add>\n<parameter=a>\n{}\n</parameter>\n<parameter=b>\n{}\n</parameter>\n</function>\n</tool_call>",
        a, b
    )
}

#[test]
fn canonical_back_to_back_blocks_pass_in_order() {
    let s = gate_schemas();
    let text = format!("{}{}", block("2", "3"), block("10.5", "0.25"));
    for chunk in [1usize, 2, 3, 5, 7, 16, 1000, usize::MAX] {
        let (out, calls) = run(&s, &text, chunk);
        assert!(out.is_empty(), "chunk {chunk}: no visible text expected, got {out:?}");
        assert_eq!(calls.len(), 2, "chunk {chunk}");
        assert!(gate_good(&calls), "chunk {chunk}: {calls:?}");
    }
}

#[test]
fn newline_separated_blocks_pass() {
    let s = gate_schemas();
    let text = format!("{}\n\n{}", block("2", "3"), block("10.5", "0.25"));
    let (_, calls) = run(&s, &text, 1000);
    assert!(gate_good(&calls), "{calls:?}");
}

#[test]
fn text_before_between_and_after_calls_keeps_calls() {
    let s = gate_schemas();
    let text =
        format!("I can do that.\n\n{}\n\nLet me compute the second one.\n\n{}\n\nDone.", block("2", "3"), block("10.5", "0.25"));
    let (out, calls) = run(&s, &text, 1000);
    assert_eq!(calls.len(), 2);
    assert!(gate_good(&calls), "{calls:?}");
    assert!(out.contains("I can do that.") && out.contains("Let me compute") && out.contains("Done."));
}

#[test]
fn both_calls_in_one_tool_call_block_keep_only_the_first() {
    // A model mistake: two <function> blocks nested inside ONE <tool_call>.
    // parse_call stops at the first "</function>", so the second call is lost
    // -> the gate sees 1 call and fails. This is the parser behaving per the
    // template's grammar (one function per block), not a parser bug.
    let s = gate_schemas();
    let text = format!(
        "<tool_call>\n<function=add>\n<parameter=a>\n2\n</parameter>\n<parameter=b>\n3\n</parameter>\n</function>\n<function=add>\n<parameter=a>\n10.5\n</parameter>\n<parameter=b>\n0.25\n</parameter>\n</function>\n</tool_call>"
    );
    let (_, calls) = run(&s, &text, 1000);
    assert_eq!(calls.len(), 1, "only the first function survives: {calls:?}");
    assert!(!gate_good(&calls));
}

#[test]
fn unterminated_tail_call_stays_visible_text() {
    // If max_tokens cuts the reply before the closing </tool_call>, the
    // splitter's finish() surfaces the partial block as visible text: the gate
    // sees 1 call -> FAIL (a length cut, not a parser bug).
    let s = gate_schemas();
    let text = format!("{}\n<tool_call>\n<function=add>\n<parameter=a>\n10.5\n</parameter>\n<parameter=b>\n0.2", block("2", "3"));
    let (out, calls) = run(&s, &text, 1000);
    assert_eq!(calls.len(), 1);
    assert!(out.contains("<tool_call>") && out.contains("10.5"));
    assert!(!gate_good(&calls));
}

#[test]
fn wrong_order_fails_the_gate() {
    let s = gate_schemas();
    let text = format!("{}{}", block("10.5", "0.25"), block("2", "3"));
    let (_, calls) = run(&s, &text, 1000);
    assert_eq!(calls.len(), 2);
    assert!(!gate_good(&calls), "order must be additive first: {calls:?}");
}

#[test]
fn string_numbers_do_not_become_typed_values() {
    // model emits "two" / "three": schema typing keeps them as strings -> the
    // args JSON is {"a":"two",...}, which fails the typed equality.
    let s = gate_schemas();
    let text = format!("{}{}", block("two", "three"), block("10.5", "0.25"));
    let (_, calls) = run(&s, &text, 1000);
    assert_eq!(calls.len(), 2);
    assert!(!gate_good(&calls), "{calls:?}");
    assert_eq!(parsed_args(&calls[0]), json!({"a": "two", "b": "three"}));
}

#[test]
fn single_call_only_fails_the_gate() {
    let s = gate_schemas();
    let text = block("2", "3");
    let (_, calls) = run(&s, &text, 1000);
    assert_eq!(calls.len(), 1);
    assert!(!gate_good(&calls));
}

#[test]
fn json_style_calls_are_not_the_qwen_format() {
    // OpenAI-style JSON tool calls are not what the template asks for; the
    // Qwen function-block parser keeps them as visible text -> 0 calls.
    let s = gate_schemas();
    let text = "{[{\"name\": \"add\", \"arguments\": {\"a\": 2, \"b\": 3}}, {\"name\": \"add\", \"arguments\": {\"a\": 10.5, \"b\": 0.25}}]}";
    let (out, calls) = run(&s, text, 1000);
    assert!(calls.is_empty());
    assert!(out.contains("10.5"));
    assert!(!gate_good(&calls));
}

#[test]
fn parameter_keys_in_any_order_still_equal() {
    // JSON objects are unordered: b-before-a is the same args object -> PASS.
    let s = gate_schemas();
    let text = format!(
        "<tool_call>\n<function=add>\n<parameter=b>\n3\n</parameter>\n<parameter=a>\n2\n</parameter>\n</function>\n</tool_call>\
         <tool_call>\n<function=add>\n<parameter=b>\n0.25\n</parameter>\n<parameter=a>\n10.5\n</parameter>\n</function>\n</tool_call>"
    );
    let (_, calls) = run(&s, &text, 1000);
    assert!(gate_good(&calls), "{calls:?}");
}

#[test]
fn bare_function_blocks_without_tool_call_are_not_calls() {
    let s = gate_schemas();
    let text = format!(
        "<function=add>\n<parameter=a>\n2\n</parameter>\n<parameter=b>\n3\n</parameter>\n</function>\n<function=add>\n<parameter=a>\n10.5\n</parameter>\n<parameter=b>\n0.25\n</parameter>\n</function>"
    );
    let (out, calls) = run(&s, &text, 1000);
    assert!(calls.is_empty());
    assert!(out.contains("<function=add>"));
    assert!(!gate_good(&calls));
}

#[test]
fn extra_whitespace_in_values_parses() {
    // The splitter strips one newline at each edge; serde_json ignores the
    // surrounding whitespace, so padded values still become typed numbers.
    let s = gate_schemas();
    let text = format!(
        "<tool_call>\n<function=add>\n<parameter=a>\n 2 \n</parameter>\n<parameter=b>\n3\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=add>\n<parameter=a>\n10.5\n</parameter>\n<parameter=b>\n0.25\n</parameter>\n</function>\n</tool_call>"
    );
    let (_, calls) = run(&s, &text, 1000);
    assert!(gate_good(&calls), "{calls:?}");
}

#[test]
fn streamed_and_non_streamed_agree_on_distinct_parallel_calls() {
    // The same text must yield the same calls whether it lands in one push
    // (non-streamed decode) or any chunking (streamed decode).
    let s = gate_schemas();
    let text = format!("Let me use the tool.\n\n{}\n\n{}\n", block("2", "3"), block("10.5", "0.25"));
    let (out_once, calls_once) = run(&s, &text, usize::MAX);
    for chunk in [1usize, 2, 3, 5, 7, 16, 1000] {
        let (out, calls) = run(&s, &text, chunk);
        assert_eq!(out, out_once, "chunk {chunk}");
        assert_eq!(calls, calls_once, "chunk {chunk}");
        assert!(gate_good(&calls), "chunk {chunk}: {calls:?}");
    }
}
