//! The Rust renderer against renders from HuggingFace's Python Jinja environment (see tools/gen_template_fixtures.py for Qwen and
//! tools/gen_glm_template_fixtures.py for GLM-5.3-Flash).

use gb_server::template::{python_float_repr, python_json, python_json_with, ChatTemplate, JsonOpts, TemplateError};
use serde_json::{json, Value};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn load() -> (ChatTemplate, Vec<Value>) {
    let tpl = std::fs::read_to_string(format!("{DIR}/qwen38_chat_template.jinja")).unwrap();
    let cases: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(format!("{DIR}/chat_template_cases.json")).unwrap()).unwrap();
    (ChatTemplate::new(&tpl).unwrap(), cases)
}

fn load_glm() -> (ChatTemplate, Vec<Value>) {
    let tpl = std::fs::read_to_string(format!("{DIR}/glm53_chat_template.jinja")).unwrap();
    let fx: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{DIR}/glm53_chat_template_cases.json")).unwrap()).unwrap();
    (ChatTemplate::new(&tpl).unwrap(), fx["cases"].as_array().unwrap().clone())
}

#[test]
fn every_fixture_renders_byte_for_byte_like_python() {
    let (t, cases) = load();
    assert!(cases.len() >= 25);
    let mut failures = Vec::new();
    for c in &cases {
        let name = c["name"].as_str().unwrap();
        let mut kwargs = c["kwargs"].as_object().unwrap().clone();
        let agp = kwargs.remove("add_generation_prompt").and_then(|v| v.as_bool()).unwrap_or(true);
        let got = t.render(&c["messages"], agp, &kwargs);
        match (&c["rendered"], &c["error"]) {
            (Value::String(want), Value::Null) => match got {
                Ok(g) if g == *want => {}
                Ok(g) => failures.push(format!("{name}: rendered differently\n--- python\n{want:?}\n--- rust\n{g:?}")),
                Err(e) => failures.push(format!("{name}: {e}")),
            },
            (Value::Null, Value::String(msg)) => match got {
                Err(TemplateError::Raised(m)) if m == *msg => {}
                other => failures.push(format!("{name}: expected the error {msg:?}, got {other:?}")),
            },
            _ => unreachable!(),
        }
    }
    assert!(failures.is_empty(), "{} of {} cases differ:\n{}", failures.len(), cases.len(), failures.join("\n\n"));
}

/// The GLM-5.3-Flash template (`{% break %}`, `tojson(ensure_ascii=False)`, macros, namespaces, tool-result sorting) renders byte for
/// byte like Python's Jinja2 under the HF environment, for text, tools, tool calls and results, reasoning history and media parts.
#[test]
fn every_glm_fixture_renders_byte_for_byte_like_python() {
    let (t, cases) = load_glm();
    assert!(cases.len() >= 100, "the fixture set shrank to {}", cases.len());
    let (mut failures, mut rendered, mut errors) = (Vec::new(), 0, 0);
    for c in &cases {
        let name = c["name"].as_str().unwrap();
        let mut kwargs = c["kwargs"].as_object().unwrap().clone();
        let agp = kwargs.remove("add_generation_prompt").and_then(|v| v.as_bool()).unwrap_or(true);
        let got = t.render(&c["messages"], agp, &kwargs);
        match (&c["rendered"], &c["error"]) {
            (Value::String(want), Value::Null) => {
                rendered += 1;
                match got {
                    Ok(g) if g == *want => {}
                    Ok(g) => failures.push(format!("{name}: rendered differently\n--- python\n{want:?}\n--- rust\n{g:?}")),
                    Err(e) => failures.push(format!("{name}: {e}")),
                }
            }
            // the GLM template never calls raise_exception: every Python error is an undefined or a type error in Jinja, which must also be an
            // error here (the wording differs between the engines)
            (Value::Null, Value::String(msg)) => {
                errors += 1;
                if got.is_ok() {
                    failures.push(format!("{name}: python failed with {msg:?}, rust gave {got:?}"));
                }
            }
            _ => unreachable!(),
        }
    }
    assert!(failures.is_empty(), "{} of {} cases differ:\n{}", failures.len(), cases.len(), failures.join("\n\n"));
    assert!(rendered >= 95 && errors >= 3, "{rendered} rendered, {errors} expected errors");
}

#[test]
fn tojson_matches_python_dumps() {
    assert_eq!(
        python_json(&json!({"b": 1, "a": [1, 2.5, null, true, "x\"y\n\u{e9}\u{2603}"]})),
        "{\"b\": 1, \"a\": [1, 2.5, null, true, \"x\\\"y\\n\u{e9}\u{2603}\"]}"
    );
    assert_eq!(python_json(&json!({})), "{}");
    assert_eq!(python_json(&json!([])), "[]");
}

/// `repr(float)` as Python prints it; every row was taken from CPython 3.12 (`repr(x)`), and serde_json's own text differs for the
/// ones with an exponent.
#[test]
fn python_float_repr_matches_cpython() {
    for (x, want) in [
        (1.0, "1.0"),
        (0.0, "0.0"),
        (-0.0, "-0.0"),
        (2.5, "2.5"),
        (-2.5, "-2.5"),
        (0.1, "0.1"),
        (1e15, "1000000000000000.0"),
        (1e16, "1e+16"),
        (1.5e16, "1.5e+16"),
        (1e17, "1e+17"),
        (1.2345678901234567e22, "1.2345678901234568e+22"),
        (123456789.125, "123456789.125"),
        (1e-4, "0.0001"),
        (1.5e-4, "0.00015"),
        (1e-5, "1e-05"),
        (1.5e-5, "1.5e-05"),
        (1e-7, "1e-07"),
        (5e-324, "5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
        (1e100, "1e+100"),
        (0.30000000000000004, "0.30000000000000004"),
        (100.0, "100.0"),
        (1234567890123456.0, "1234567890123456.0"),
    ] {
        assert_eq!(python_float_repr(x), want, "{x:e}");
    }
    // and through the JSON writer: serde_json would write 1e16 and 1e-7
    assert_eq!(python_json(&json!([1e16, 1e-7, 1.0, 3])), "[1e+16, 1e-07, 1.0, 3]");
}

/// `json.dumps(x, ensure_ascii=True)`, `indent=`, `separators=` and `sort_keys=`, the options HF's `tojson` forwards.
#[test]
fn tojson_options_match_python_dumps() {
    let v = json!({"b": [1, {"z": null, "a": "\u{e9}\u{1f600}\u{7f}\u{1}"}], "a": {}, "c": []});
    let o = |ensure_ascii, indent: Option<&str>, sep: Option<(&str, &str)>, sort_keys| JsonOpts {
        ensure_ascii,
        indent: indent.map(str::to_string),
        separators: sep.map(|(a, b)| (a.to_string(), b.to_string())),
        sort_keys,
    };
    // json.dumps(v, ensure_ascii=True)
    assert_eq!(
        python_json_with(&v, &o(true, None, None, false)),
        "{\"b\": [1, {\"z\": null, \"a\": \"\\u00e9\\ud83d\\ude00\\u007f\\u0001\"}], \"a\": {}, \"c\": []}"
    );
    // json.dumps(v, indent=2, sort_keys=True)
    assert_eq!(
        python_json_with(&v, &o(false, Some("  "), None, true)),
        "{\n  \"a\": {},\n  \"b\": [\n    1,\n    {\n      \"a\": \"\u{e9}\u{1f600}\u{7f}\\u0001\",\n      \"z\": null\n    }\n  ],\n  \"c\": []\n}"
    );
    // json.dumps(v, separators=(",", ":"))
    assert_eq!(
        python_json_with(&v, &o(false, None, Some((",", ":")), false)),
        "{\"b\":[1,{\"z\":null,\"a\":\"\u{e9}\u{1f600}\u{7f}\\u0001\"}],\"a\":{},\"c\":[]}"
    );
    // indent="\t" with the default separators becomes ",\n" between items
    assert_eq!(python_json_with(&json!([1, [2]]), &o(false, Some("\t"), None, false)), "[\n\t1,\n\t[\n\t\t2\n\t]\n]");
}

/// `tojson` as the template calls it: keyword arguments are honoured, and one that is not understood is an error rather than ignored.
#[test]
fn tojson_filter_takes_keyword_arguments() {
    let t = |src: &str| ChatTemplate::new(src).unwrap().render(&json!([]), false, &serde_json::Map::new());
    let src = |args: &str| format!("{{{{ {{'k': '\u{e9}', 'n': [1, 2]}} | tojson{args} }}}}");
    assert_eq!(t(&src("")).unwrap(), "{\"k\": \"\u{e9}\", \"n\": [1, 2]}");
    assert_eq!(t(&src("(ensure_ascii=False)")).unwrap(), "{\"k\": \"\u{e9}\", \"n\": [1, 2]}");
    assert_eq!(t(&src("(ensure_ascii=True)")).unwrap(), "{\"k\": \"\\u00e9\", \"n\": [1, 2]}");
    assert_eq!(t(&src("(sort_keys=True, separators=(',', ':'))")).unwrap(), "{\"k\":\"\u{e9}\",\"n\":[1,2]}");
    assert!(t(&src("(no_such_option=1)")).is_err());
    assert!(t(&src("(separators=(',', ':', 'x'))")).is_err());
    assert!(t(&src("(indent=[1])")).is_err());
}

/// `{% break %}` and `{% continue %}` (HF enables jinja2's `loopcontrols`).
#[test]
fn loop_controls_are_available() {
    let t = ChatTemplate::new(
        "{% for i in range(6) %}{% if i == 1 %}{% continue %}{% endif %}{% if i == 4 %}{% break %}{% endif %}{{ i }}{% endfor %}",
    )
    .unwrap();
    assert_eq!(t.render(&json!([]), false, &serde_json::Map::new()).unwrap(), "023");
}

/// The migration to minijinja 3 moved the whitespace flags onto `SyntaxConfig`
/// (`trim_blocks`/`lstrip_blocks`/`keep_trailing_newline`). These renders pin the
/// resulting behavior so a regression in the config wiring shows up as a diff.
#[test]
fn syntax_config_whitespace_flags_trim_and_strip() {
    let render = |src: &str| ChatTemplate::new(src).unwrap().render(&json!([]), false, &serde_json::Map::new()).unwrap();
    // trim_blocks removes the newline after a block tag that ends its line.
    assert_eq!(render("HEAD\n{% set x = 1 %}\nTAIL {{ x }}\n"), "HEAD\nTAIL 1");
    // lstrip_blocks strips the indentation before a block tag on its own line.
    assert_eq!(render("A\n   {% set y = 2 %}\nB {{ y }}\n"), "A\nB 2");
    // keep_trailing_newline = false strips the final newline.
    assert_eq!(render("Z {{ 1 }}\n"), "Z 1");
}

/// The migration replaced `Value::from_serialize(&ctx)` with minijinja 3's
/// `value::Serde` wrapper for the serde_json context. Renders of nested JSON
/// values (objects, arrays, numbers, booleans) prove the wrapper path works.
#[test]
fn serde_context_wrapper_renders_nested_json() {
    let mut kwargs = serde_json::Map::new();
    kwargs.insert("n".into(), json!(42));
    kwargs.insert("flag".into(), json!(true));
    kwargs.insert("obj".into(), json!({"k": "v", "arr": [1, 2]}));
    let t = ChatTemplate::new("{{ n }}|{{ flag }}|{{ obj.k }}|{{ obj.arr[0] }}|{{ obj.arr | length }}").unwrap();
    assert_eq!(t.render(&json!([]), false, &kwargs).unwrap(), "42|True|v|1|2");
}
