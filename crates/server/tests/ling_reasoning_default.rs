//! The Ling-3.1 thinking default, as the shipped profile sets it (`--default-reasoning-effort on`), against the pack's own chat
//! template (`testdata/ling31-flash-1019/chat_template.jinja`) and the family's declared descriptor.
//!
//! The server's own default is thinking off (`DefaultThinking::Off`): a request that names neither `enable_thinking` nor a
//! `reasoning_effort` renders the closed `<think></think>` block, which this model answers badly. With the profile's default such a
//! request renders exactly what `enable_thinking: true` renders, and a request that names its own choice keeps it.

use gb_server::chat_format::ChatFormat;
use gb_server::formats::bailing_moe_v3::{reasoning_capabilities, BailingFormat};
use gb_server::reasoning_controls::{check_default_renders, default_effort, default_thinks, resolve, DefaultThinking};
use serde_json::{json, Value};
use std::path::Path;

fn manifest(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn format() -> BailingFormat {
    let src = std::fs::read_to_string(manifest("../../testdata/ling31-flash-1019/chat_template.jinja")).unwrap();
    BailingFormat::new(&src).unwrap()
}

/// The prompt a chat request with these extra fields renders under `default`.
fn prompt(default: DefaultThinking, extra: Value) -> String {
    let mut body = json!({"messages": [{"role": "user", "content": "Compute 17 * 23."}]});
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    let mut kwargs = resolve(&body, &reasoning_capabilities(), default).unwrap_or_else(|e| panic!("{extra}: {e:?}")).kwargs;
    kwargs.insert("add_generation_prompt".into(), Value::Bool(true));
    format().render(&body["messages"], &kwargs).unwrap().text
}

fn profile_default() -> DefaultThinking {
    default_effort(&reasoning_capabilities(), "on").expect("the Ling descriptor declares the level on")
}

const OPEN: &str = "<role>ASSISTANT</role>\n<think>";
const CLOSED: &str = "<role>ASSISTANT</role>\n<think></think>";

#[test]
fn the_profile_default_makes_a_request_that_names_nothing_think() {
    let on = profile_default();
    let explicit = prompt(on, json!({"chat_template_kwargs": {"enable_thinking": true}}));
    assert!(explicit.ends_with(OPEN) && explicit.contains("detailed thinking on<|role_end|>"), "{explicit}");
    // byte for byte the prompt of a request that asks for thinking itself
    assert_eq!(prompt(on, json!({})), explicit);
    assert_eq!(prompt(on, json!({"reasoning_effort": "on"})), explicit);
    assert!(default_thinks(&reasoning_capabilities(), on));
    // the engine renders this default once at boot; the pack's template takes it
    assert!(check_default_renders(&format(), &reasoning_capabilities(), on).is_ok());
}

#[test]
fn a_request_that_names_its_own_choice_keeps_it_under_the_profile_default() {
    let on = profile_default();
    for off in [json!({"chat_template_kwargs": {"enable_thinking": false}}), json!({"reasoning_effort": "off"})] {
        let p = prompt(on, off.clone());
        assert!(p.ends_with(CLOSED) && p.contains("detailed thinking off<|role_end|>"), "{off}: {p}");
    }
}

#[test]
fn without_the_profile_default_a_request_that_names_nothing_does_not_think() {
    // the server default the profile replaces: the closed block this model answers with reasoning text in the content
    let p = prompt(DefaultThinking::Off, json!({}));
    assert!(p.ends_with(CLOSED) && p.contains("detailed thinking off<|role_end|>"), "{p}");
    assert!(!default_thinks(&reasoning_capabilities(), DefaultThinking::Off));
}

#[test]
fn a_level_the_descriptor_does_not_declare_is_refused() {
    let e = default_effort(&reasoning_capabilities(), "low").unwrap_err();
    assert!(e.contains("\"low\" is not one of this model's efforts: on, off"), "{e}");
}

#[test]
fn the_shipped_profile_and_recipe_carry_the_default() {
    let read = |rel: &str| -> Value { serde_json::from_str(&std::fs::read_to_string(manifest(rel)).unwrap()).unwrap() };
    let profile = read("../../runtimes/ling-3.1/ling-3.1-flash-1019/profile.json");
    let argv: Vec<&str> =
        profile["profiles"]["ling31-single-262k"]["argv"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let at = argv.iter().position(|a| *a == "--default-reasoning-effort").expect("the profile sets the thinking default");
    assert_eq!(argv[at + 1], "on");
    assert_eq!(argv.iter().filter(|a| **a == "--default-reasoning-effort").count(), 1);
    assert!(!argv.contains(&"--default-thinking"), "one switch sets the default, not two");
    // the level the profile names is one the descriptor declares, and it thinks
    assert!(reasoning_capabilities().level(argv[at + 1]).is_some_and(|l| l.thinks));
    let recipe = read("../../scripts/gates/recipes/ling31-flash-1019-sage-1.51.json");
    assert_eq!(recipe["switch_justifications"]["--default-reasoning-effort"]["value"], "on");
}
