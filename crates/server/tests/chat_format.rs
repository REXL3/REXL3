//! Chat formats: how a family is found (and refused), that Qwen behaves exactly as before the formats existed, that GLM-5.3-Flash renders,
//! streams tool calls and counts image tokens like the references, and that a third family registers without a line of shared code
//! changing. Needs no model folder: every input is a committed fixture (`tests/fixtures`, from tools/gen_glm_template_fixtures.py and
//! tools/gen_template_fixtures.py). The real-tokenizer checks are in `tests/tokenizer.rs`.

use gb_server::api;
use gb_server::chat_format::{render_normalized, ChatFormat, FormatEntry, FormatRegistry, RenderedChat};
use gb_server::formats::glm::vision::GlmImageGeometry;
use gb_server::reasoning::ReasoningSplitter;
use gb_server::template::{ChatTemplate, TemplateError};
use gb_server::tokenizer::Tok;
use gb_server::tools::{ToolCall, ToolSchemas, ToolSplitter, ToolSyntax};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

const FX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn read(name: &str) -> String {
    std::fs::read_to_string(format!("{FX}/{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn json_fixture(name: &str) -> Value {
    serde_json::from_str(&read(name)).unwrap()
}

/// A scratch model folder holding the given files, removed on drop.
struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str, files: &[(&str, &str)]) -> Dir {
        // tests run in parallel threads: every folder is its own
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("rexl3-chat-format-{}-{tag}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        for (name, text) in files {
            std::fs::write(p.join(name), text).unwrap();
        }
        Dir(p)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn qwen_dir() -> Dir {
    Dir::new(
        "qwen",
        &[("config.json", &read("qwen38_config_head.json")), ("chat_template.jinja", &read("qwen38_chat_template.jinja"))],
    )
}

fn glm_dir() -> Dir {
    Dir::new(
        "glm",
        &[
            ("config.json", &read("glm53_config_head.json")),
            ("chat_template.jinja", &read("glm53_chat_template.jinja")),
            ("processor_config.json", &read("glm53_processor_config.json")),
        ],
    )
}

fn open(dir: &Dir) -> Box<dyn ChatFormat> {
    FormatRegistry::builtin().open(dir.path(), None).unwrap()
}

/// The case's template variables, and whether it asks for the generation prompt (the formats always add it).
fn kwargs_of(case: &Value) -> (Map<String, Value>, bool) {
    let mut kwargs = case["kwargs"].as_object().unwrap().clone();
    let agp = kwargs.remove("add_generation_prompt").and_then(|v| v.as_bool()).unwrap_or(true);
    (kwargs, agp)
}

/// A mini tokenizer with `<|image|>` (id 7) and `<|image_pad|>` (id 248056 is Qwen's constant and not looked up), so a `Tok` exists
/// without the 20 MB real file.
fn mini_tok() -> (Dir, Tok) {
    let tj = json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [{"id": 7, "content": "<|image|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
        "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"a": 0, "[UNK]": 1}, "unk_token": "[UNK]"}
    });
    let d = Dir::new("tok", &[("tokenizer.json", &tj.to_string())]);
    let t = Tok::from_dir(d.path()).unwrap();
    (d, t)
}

// ---------------------------------------------------------------------------------------------------------------------
// the registry

#[test]
fn a_model_is_matched_by_architecture_then_model_type() {
    let r = FormatRegistry::builtin();
    assert_eq!(r.ids(), vec!["qwen", "glm5_next", "mimo", "nemotron_omni", "minicpmv", "deepseek_v41", "bailing_moe_v3"]);
    assert_eq!(r.for_config(&json_fixture("qwen38_config_head.json")).unwrap().id, "qwen");
    assert_eq!(r.for_config(&json_fixture("glm53_config_head.json")).unwrap().id, "glm5_next");
    // model_type alone, and as the fallback when the architecture is one nobody registered
    assert_eq!(r.for_config(&json!({"model_type": "glm5_next"})).unwrap().id, "glm5_next");
    assert_eq!(r.for_config(&json!({"architectures": ["SomethingElse"], "model_type": "qwen3_5"})).unwrap().id, "qwen");
    // the architecture is asked first
    assert_eq!(
        r.for_config(&json!({"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "qwen3_5"})).unwrap().id,
        "glm5_next"
    );
    assert_eq!(r.by_id("glm5_next").unwrap().id, "glm5_next");
    // opened from a folder, by config and by id
    assert_eq!(open(&qwen_dir()).id(), "qwen");
    assert_eq!(open(&glm_dir()).id(), "glm5_next");
    assert_eq!(r.open(glm_dir().path(), Some("glm5_next")).unwrap().id(), "glm5_next");
}

#[test]
fn a_model_nothing_matches_is_refused_by_name_and_never_gets_the_qwen_format() {
    let r = FormatRegistry::builtin();
    for cfg in [
        json!({"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4"}),
        json!({"architectures": ["Glm4MoeForCausalLM"]}),
        json!({"model_type": "llama"}),
        json!({}),
    ] {
        let e = r.for_config(&cfg).err().unwrap_or_else(|| panic!("{cfg} must not select a format"));
        assert!(
            e.contains("no chat format is registered")
                && e.contains("qwen")
                && e.contains("glm5_next")
                && e.contains("--chat-format"),
            "{e}"
        );
    }
    let e = r.for_config(&json!({"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4"})).err().unwrap();
    assert!(e.contains("Gemma4ForConditionalGeneration") && e.contains("gemma4"), "the refusal names what it saw: {e}");
    let e = r.by_id("llama").err().unwrap();
    assert!(e.contains("unknown chat format `llama`") && e.contains("qwen"), "{e}");
    // from a folder: an unknown architecture, a missing config.json, an unreadable one, an unknown forced id
    let unknown = Dir::new(
        "unknown",
        &[("config.json", r#"{"architectures": ["Gemma4ForConditionalGeneration"]}"#), ("chat_template.jinja", "x")],
    );
    assert!(r.open(unknown.path(), None).err().unwrap().contains("Gemma4ForConditionalGeneration"));
    let no_config = Dir::new("noconfig", &[("chat_template.jinja", &read("qwen38_chat_template.jinja"))]);
    assert!(r.open(no_config.path(), None).err().unwrap().contains("config.json"));
    let bad = Dir::new("badconfig", &[("config.json", "{nope"), ("chat_template.jinja", "x")]);
    assert!(r.open(bad.path(), None).err().unwrap().contains("config.json"));
    assert!(r.open(qwen_dir().path(), Some("nope")).err().unwrap().contains("unknown chat format"));
    // naming a format directly works for a folder whose config no entry claims
    assert_eq!(r.open(no_config.path(), Some("qwen")).unwrap().id(), "qwen");
}

#[test]
fn an_id_or_a_name_cannot_be_registered_twice() {
    let fake = |id: &'static str, names: &'static [&'static str]| FormatEntry { id, names, open: |_| Err("unused".into()) };
    assert!(FormatRegistry::builtin().with(fake("qwen", &["x"])).err().unwrap().contains("registered twice"));
    assert!(FormatRegistry::builtin().with(fake("other", &["glm5_next"])).err().unwrap().contains("claimed by"));
    assert!(FormatRegistry::builtin().with(fake("other", &["other_arch"])).is_ok());
}

// ---------------------------------------------------------------------------------------------------------------------
// a third family, registered only here

/// A toy family with nothing in common with Qwen or GLM: its own template, its own reasoning close tag, its own tool-call markers and
/// syntax (`[[call]]name|{json}[[/call]]`), its own image part (`pic`) and no vision default for the rest.
struct ToyFormat {
    template: ChatTemplate,
}

fn toy_parse(block: &str, _schemas: &ToolSchemas) -> Option<ToolCall> {
    let (name, args) = block.split_once('|')?;
    serde_json::from_str::<Value>(args).ok().filter(Value::is_object)?;
    Some(ToolCall { name: name.trim().to_string(), arguments: args.trim().to_string() })
}

impl ChatFormat for ToyFormat {
    fn id(&self) -> &'static str {
        "toy"
    }
    fn render(&self, messages: &Value, kwargs: &Map<String, Value>) -> Result<RenderedChat, TemplateError> {
        let text = render_normalized(&self.template, messages, kwargs)?;
        let starts_in_reasoning = self.starts_in_reasoning(&text);
        Ok(RenderedChat { text, starts_in_reasoning })
    }
    fn starts_in_reasoning(&self, prompt: &str) -> bool {
        prompt.ends_with("<reason>")
    }
    fn reasoning_close(&self) -> &'static str {
        "</reason>"
    }
    fn tool_syntax(&self) -> ToolSyntax {
        ToolSyntax { open: "[[call]]", close: "[[/call]]", parse: toy_parse }
    }
    fn is_image_part(&self, part: &Value) -> bool {
        part.get("type").and_then(|t| t.as_str()) == Some("pic")
    }
}

const TOY: FormatEntry = FormatEntry {
    id: "toy",
    names: &["ToyForCausalLM", "toy"],
    open: |dir| {
        let src = std::fs::read_to_string(dir.join("chat_template.jinja")).map_err(|e| format!("chat_template.jinja: {e}"))?;
        Ok(Box::new(ToyFormat { template: ChatTemplate::new(&src).map_err(|e| e.to_string())? }))
    },
};

const TOY_TEMPLATE: &str = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% if m.tool_calls %}{% for c in m.tool_calls %}[[call]]{{ c.function.name }}|{{ c.function.arguments | tojson }}[[/call]]{% endfor %}{% endif %}\n{% endfor %}<assistant><reason>";

#[test]
fn a_new_family_works_through_the_lookup_without_touching_shared_code() {
    let dir =
        Dir::new("toy", &[("config.json", r#"{"architectures": ["ToyForCausalLM"]}"#), ("chat_template.jinja", TOY_TEMPLATE)]);
    // the shipped registry has never heard of it ...
    assert!(FormatRegistry::builtin().open(dir.path(), None).err().unwrap().contains("ToyForCausalLM"));
    // ... one registration makes it a first-class format, found by architecture and by id
    let reg = FormatRegistry::builtin().with(TOY).unwrap();
    assert_eq!(
        reg.ids(),
        vec!["qwen", "glm5_next", "mimo", "nemotron_omni", "minicpmv", "deepseek_v41", "bailing_moe_v3", "toy"]
    );
    let f = reg.open(dir.path(), None).unwrap();
    assert_eq!(f.id(), "toy");
    assert_eq!(reg.open(dir.path(), Some("toy")).unwrap().id(), "toy");
    // the old families still resolve
    assert_eq!(reg.open(glm_dir().path(), None).unwrap().id(), "glm5_next");

    // render: the shared normalisation turns a JSON-string `arguments` into an object before the family's own template sees it
    let messages = json!([{"role": "user", "content": "hi"},
        {"role": "assistant", "content": null, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}]);
    let r = f.render(&messages, &Map::new()).unwrap();
    assert_eq!(r.text, "<user>hi<assistant>[[call]]f|{\"a\": 1}[[/call]]<assistant><reason>");
    assert!(r.starts_in_reasoning);
    assert!(!f.starts_in_reasoning("<assistant>"));

    // the shared tool splitter runs on the family's own markers and parser, however the text is chunked
    let text = "Sure.\n[[call]]f|{\"a\": 1}[[/call]] and [[call]]g|{\"b\": \"\u{e9}\"}[[/call]] done";
    let mut want: Option<(String, Vec<ToolCall>)> = None;
    for cut in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
        let mut sp = f.tool_splitter(ToolSchemas::default());
        let (mut t, mut calls) = sp.push(&text[..cut]);
        let (t2, c2) = sp.push(&text[cut..]);
        let (t3, c3) = sp.finish();
        t.push_str(&t2);
        t.push_str(&t3);
        calls.extend(c2);
        calls.extend(c3);
        match &want {
            None => want = Some((t, calls)),
            Some(w) => assert_eq!((&t, &calls), (&w.0, &w.1), "cut at {cut}"),
        }
    }
    let (t, calls) = want.unwrap();
    assert_eq!(t, "Sure. and done");
    assert_eq!(calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["f", "g"]);
    assert_eq!(calls[1].arguments, "{\"b\": \"\u{e9}\"}");

    // the family's own reasoning close tag
    let mut rs = ReasoningSplitter::with_close(true, f.reasoning_close());
    let a = rs.push("think </rea");
    let b = rs.push("son>answer");
    assert_eq!((a.reasoning + &b.reasoning, a.content + &b.content), ("think ".to_string(), "answer".to_string()));

    // media: a text-only default refuses images; the family's predicate drives extraction and no placeholder is invented
    assert!(f.check_media(&json!([{"role": "user", "content": [{"type": "pic", "image_url": "data:x"}]}]), true).is_err());
    let imgs = f
        .extract_images(&json!([{"role": "user", "content": [{"type": "pic", "image": "data:image/png;base64,AA"}, {"type": "image_url", "image_url": "data:y"}]}]), &Default::default())
        .unwrap();
    assert_eq!(imgs.len(), 1);
    assert_eq!(imgs[0].url, "data:image/png;base64,AA");
    let (_d, tok) = mini_tok();
    assert!(f.expand_images(&tok, &[1, 2], &[]).err().unwrap().message.contains("no image placeholder"));
    assert!(!f.serves_images());
}

// ---------------------------------------------------------------------------------------------------------------------
// Qwen: unchanged

/// The expression `http.rs` used before formats existed, kept here as the oracle for the Qwen rule.
fn old_qwen_starts_in_reasoning(text: &str) -> bool {
    text.trim_end_matches('\n').ends_with("<think>") && !text.ends_with("</think>\n\n")
}

#[test]
fn qwen_renders_and_detects_thinking_exactly_as_before() {
    let f = open(&qwen_dir());
    let cases: Vec<Value> = serde_json::from_str(&read("chat_template_cases.json")).unwrap();
    let (mut rendered, mut raised, mut open_think, mut closed_think) = (0, 0, 0, 0);
    for c in &cases {
        let name = c["name"].as_str().unwrap();
        let (kwargs, agp) = kwargs_of(c);
        if !agp {
            continue; // the format always adds the generation prompt; the raw template with it off is in tests/template.rs
        }
        match (&c["rendered"], &c["error"]) {
            (Value::String(want), Value::Null) => {
                let r = f.render(&c["messages"], &kwargs).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(&r.text, want, "{name}");
                assert_eq!(r.starts_in_reasoning, old_qwen_starts_in_reasoning(want), "{name}");
                rendered += 1;
                if r.starts_in_reasoning {
                    open_think += 1;
                } else {
                    closed_think += 1;
                }
            }
            (Value::Null, Value::String(msg)) => {
                assert_eq!(f.render(&c["messages"], &kwargs), Err(TemplateError::Raised(msg.clone())), "{name}");
                raised += 1;
            }
            _ => unreachable!(),
        }
    }
    assert!(
        rendered >= 23 && raised >= 4 && open_think >= 15 && closed_think >= 2,
        "{rendered} {raised} {open_think} {closed_think}"
    );
    // pinned by hand: a thinking prompt opens the block, a non-thinking one closes it
    let m = json!([{"role": "user", "content": "hi"}]);
    let on = f.render(&m, &Map::new()).unwrap();
    assert_eq!(
        on.text,
        "<|im_start|>system\nReasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    assert!(on.starts_in_reasoning);
    let off = f.render(&m, json!({"enable_thinking": false}).as_object().unwrap()).unwrap();
    assert_eq!(off.text, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    assert!(!off.starts_in_reasoning);
    assert_eq!(f.reasoning_close(), "</think>");
}

#[test]
fn qwen_tool_calls_media_and_images_behave_as_before() {
    let f = open(&qwen_dir());
    // tool calls: the Qwen block syntax
    let schemas = ToolSchemas::from_request(&json!({"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "tags": {"type": "array"}}}}}]}))
    .unwrap();
    let call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n</function>\n</tool_call>";
    let mut sp = f.tool_splitter(schemas.clone());
    let (t, c) = sp.push(&format!("Let me check.\n\n{call}"));
    let (t2, c2) = sp.finish();
    assert_eq!(format!("{t}{t2}"), "Let me check.");
    assert_eq!(c.len() + c2.len(), 1);
    assert_eq!(c[0].name, "get_weather");
    assert_eq!(serde_json::from_str::<Value>(&c[0].arguments).unwrap(), json!({"city": "Paris", "days": 3, "tags": ["a", "b"]}));
    // a GLM-shaped call is not a Qwen call: it stays visible
    let mut sp = f.tool_splitter(schemas);
    let glm_call = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
    let (t, c) = sp.push(glm_call);
    assert_eq!((t.as_str(), c.len()), (glm_call, 0));
    // media policy is `api::refuse_media`, message for message
    let img = json!([{"role": "user", "content": [{"type": "text", "text": "what"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]);
    let keyed = json!([{"role": "user", "content": [{"image": "x"}]}]);
    let video = json!([{"role": "user", "content": [{"type": "video", "video": "x"}]}]);
    let text = json!([{"role": "user", "content": "hi"}, {"role": "user", "content": [{"type": "text", "text": "hi"}]}]);
    for (m, allow) in [
        (&img, true),
        (&img, false),
        (&keyed, true),
        (&keyed, false),
        (&video, true),
        (&video, false),
        (&text, true),
        (&text, false),
    ] {
        assert_eq!(f.check_media(m, allow), api::refuse_media(m, allow), "{m} allow={allow}");
    }
    assert!(f.check_media(&img, true).is_ok() && f.check_media(&img, false).is_err() && f.check_media(&video, true).is_err());
    // images: the template's own test, the `<|image_pad|>` constant, the same expansion and message
    assert!(
        f.is_image_part(&json!({"type": "image"}))
            && f.is_image_part(&json!({"image": "x"}))
            && f.is_image_part(&json!({"image_url": {}}))
    );
    assert!(!f.is_image_part(&json!({"type": "text", "text": "x"})));
    let (_d, tok) = mini_tok();
    assert!(f.image_placeholder(&tok).is_err(), "a tokenizer without the image token must fail closed");
    // The real pack still uses 248056; the CPU fixture's tokenizer declares 261. The format uses the tokenizer, not a hardcoded ID.
    let tok = Tok::from_dir(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mock_model")).unwrap();
    let pad = tok.token_to_id("<|image_pad|>").unwrap();
    assert_eq!(f.image_placeholder(&tok).unwrap(), (pad, "<|image_pad|>"));
    let (ids, spans) = f.expand_images(&tok, &[1, pad, 2], &[3]).unwrap();
    assert_eq!(ids, vec![1, pad, pad, pad, 2]);
    assert_eq!((spans[0].start, spans[0].tokens), (1, 3));
    let e = f.expand_images(&tok, &[1, 2], &[3]).err().unwrap();
    assert!(
        e.message.contains("the prompt has 0 image placeholders but 1 images were sent") && e.message.contains("<|image_pad|>"),
        "{}",
        e.message
    );
    assert!(f.serves_images());
    assert!(f.image_tokens(100, 100).is_none(), "Qwen's count comes with its pixel preprocessing");
}

// ---------------------------------------------------------------------------------------------------------------------
// GLM-5.3-Flash

#[test]
fn glm_renders_every_fixture_through_the_format() {
    let f = open(&glm_dir());
    let fx = json_fixture("glm53_chat_template_cases.json");
    let (mut rendered, mut errors, mut wire, mut skipped) = (0, 0, 0, 0);
    for c in fx["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let (kwargs, agp) = kwargs_of(c);
        if !agp {
            skipped += 1;
            continue;
        }
        if let Some(why) = c.get("hf_refuses") {
            assert_eq!(f.render(&c["messages"], &kwargs), Err(TemplateError::Raised(why.as_str().unwrap().into())), "{name}");
            continue;
        }
        match (&c["rendered"], &c["error"]) {
            (Value::String(want), Value::Null) => {
                let r = f.render(&c["messages"], &kwargs).unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(&r.text, want, "{name}");
                assert!(r.starts_in_reasoning && r.text.ends_with("<think>"), "{name}: a GLM prompt always ends inside <think>");
                rendered += 1;
                // the same conversation as an OpenAI client sends it (string `arguments`, null content)
                if let Some(w) = c.get("wire_messages") {
                    assert_eq!(&f.render(w, &kwargs).unwrap().text, want, "{name} (wire form)");
                    wire += 1;
                }
            }
            (Value::Null, Value::String(_)) => {
                // the raw template fails on all of these; the format's normalisation repairs the one OpenAI clients really send
                // (`arguments` as a JSON string) and nothing else
                let r = f.render(&c["messages"], &kwargs);
                if name.contains("JSON string") {
                    assert!(
                        r.unwrap().text.contains("<tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>"),
                        "{name}"
                    );
                } else {
                    assert!(r.is_err(), "{name}");
                }
                errors += 1;
            }
            _ => unreachable!(),
        }
    }
    assert!(rendered >= 95 && errors >= 3 && wire >= 20 && skipped == 1, "{rendered} {errors} {wire} {skipped}");
    // an OpenAI client's string `arguments` renders; the raw template cannot (the case above with the same text fails there)
    let wire_args = json!([{"role": "user", "content": "q"}, {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}, {"role": "tool", "tool_call_id": "c1", "content": "r"}]);
    let r = f.render(&wire_args, &Map::new()).unwrap();
    assert!(
        r.text.contains(
            "<tool_call>f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call><|observation|><tool_response>r</tool_response>"
        ),
        "{}",
        r.text
    );
    let bad = json!([{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "not json"}}]}]);
    match f.render(&bad, &Map::new()) {
        Err(TemplateError::Raised(m)) => {
            assert!(m.starts_with("messages[0].tool_calls[0].function.arguments is not JSON"), "{m}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn glm_prompt_ends_inside_think_so_the_reply_is_split_at_the_close_tag() {
    let f = open(&glm_dir());
    assert!(f.starts_in_reasoning("[gMASK]<sop><|user|>x<|assistant|><think>"));
    assert!(!f.starts_in_reasoning("[gMASK]<sop><|user|>x<|assistant|><think></think>"));
    assert!(!f.starts_in_reasoning("[gMASK]<sop><|user|>x<|assistant|>"));
    assert!(!f.starts_in_reasoning("...<think>\n"), "GLM writes no newline after <think>");
    assert_eq!(f.reasoning_close(), "</think>");
    // GLM's reply: reasoning, the close tag, the answer, with no newline in between
    let mut rs = ReasoningSplitter::with_close(true, f.reasoning_close());
    let s = rs.push("I should add.</think>The answer is 4.");
    assert_eq!((s.reasoning.as_str(), s.content.as_str()), ("I should add.", "The answer is 4."));
}

#[test]
fn glm_media_policy_matches_what_the_template_renders() {
    let f = open(&glm_dir());
    let user = |parts: Value| json!([{"role": "user", "content": parts}]);
    let ok = |m: &Value, allow: bool| f.check_media(m, allow).is_ok();
    let img = json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}});
    // what the template renders as an image is accepted when the server can take images, refused when it cannot
    for part in [img.clone(), json!({"type": "image", "image": "x"}), json!({"type": "image"})] {
        assert!(ok(&user(json!([part.clone(), {"type": "text", "text": "x"}])), true), "{part}");
        let e = f.check_media(&user(json!([part])), false).err().unwrap();
        assert!(e.message.contains("messages[0].content[0]") && e.message.contains("without image support"), "{}", e.message);
    }
    assert!(ok(&json!([{"role": "user", "content": "plain"}, {"role": "assistant", "content": null}]), false));
    assert!(ok(&user(json!([{"type": "text", "text": "t"}, "bare string", {"type": "mystery"}])), false));
    // what the template drops without a trace is refused, always
    for (part, ty) in [
        (json!({"type": "input_image", "image_url": "x"}), "input_image"),
        (json!({"image": "x"}), "media"),
        (json!({"image_url": {"url": "x"}}), "media"),
        (json!({"type": "input_video"}), "input_video"),
    ] {
        let e = f.check_media(&user(json!([part.clone()])), true).err().unwrap();
        assert!(e.message.contains("dropped silently") && e.message.contains(ty), "{part}: {}", e.message);
    }
    // video and audio are rendered as markers for data the server never has
    for part in [
        json!({"type": "video", "video": "x"}),
        json!({"type": "video_url", "video_url": {"url": "x"}}),
        json!({"type": "audio"}),
        json!({"type": "input_audio", "input_audio": {}}),
    ] {
        assert!(!ok(&user(json!([part])), true));
    }
    // the template reorders tool results, so an image inside one could pair with the wrong placeholder: refused, nested outputs included
    let in_tool = json!([{"role": "tool", "tool_call_id": "c", "content": [{"type": "text", "text": "t"}, img.clone()]}]);
    assert!(f.check_media(&in_tool, true).err().unwrap().message.contains("inside a tool result"));
    let in_output = json!([{"role": "tool", "content": [{"tool_call_id": "c", "output": [img.clone()]}]}]);
    assert!(f.check_media(&in_output, true).err().unwrap().message.contains("messages[0].content[0].output[0]"));
    assert!(!ok(&json!([{"role": "tool", "content": [{"tool_call_id": "c", "output": [{"type": "video"}]}]}]), true));
    assert!(ok(
        &json!([{"role": "tool", "tool_call_id": "c", "content": "text only"}, {"role": "tool", "content": [{"tool_call_id": "c", "output": "t"}]}]),
        false
    ));
    // the images found are exactly the parts the template renders a placeholder for
    let msgs = json!([
        {"role": "system", "content": [{"type": "image", "image": "data:a"}]},
        {"role": "user", "content": [{"type": "input_image", "image_url": "data:ignored"}, {"type": "image_url", "image_url": {"url": "data:b"}}, {"type": "image_url", "image_url": "data:c"}]}
    ]);
    let found = f.extract_images(&msgs, &gb_server::vision::Limits { max_images: 9, ..Default::default() }).unwrap();
    assert_eq!(
        found.iter().map(|r| (r.message, r.part, r.url.as_str())).collect::<Vec<_>>(),
        vec![(0, 0, "data:a"), (1, 1, "data:b"), (1, 2, "data:c")]
    );
    assert!(f.serves_images(), "the GLM tower and pixel preprocessing exist: this format serves images");
}

#[test]
fn glm_image_placeholders_expand_one_run_per_image() {
    let f = open(&glm_dir());
    let (_d, tok) = mini_tok();
    let img = tok.token_to_id("<|image|>").unwrap();
    assert_eq!(f.image_placeholder(&tok).unwrap(), (img, "<|image|>"));
    let (ids, spans) = f.expand_images(&tok, &[1, img, 5, img, 6], &[3, 2]).unwrap();
    assert_eq!(ids, vec![1, img, img, img, 5, img, img, 6]);
    assert_eq!(spans.iter().map(|s| (s.start, s.tokens)).collect::<Vec<_>>(), vec![(1, 3), (5, 2)]);
    // a user who types the placeholder, or an image whose part the template dropped: the counts disagree and the request is refused
    let e = f.expand_images(&tok, &[1, img, img], &[3]).err().unwrap();
    assert!(e.message.contains("2 image placeholders but 1 images") && e.message.contains("<|image|>"), "{}", e.message);
    // a tokenizer without the token
    let no_image = Dir::new(
        "noimg",
        &[(
            "tokenizer.json",
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"a":0,"[UNK]":1},"unk_token":"[UNK]"}}"#,
        )],
    );
    let tok2 = Tok::from_dir(no_image.path()).unwrap();
    assert!(f.expand_images(&tok2, &[1], &[]).err().unwrap().message.contains("has no <|image|> token"));
}

/// The number of `<|image|>` tokens for an image of a given size, against the real `Glm5NextImageProcessor` (its grid, by processing a
/// synthetic image of that size; the largest sizes by its `get_number_of_image_patches`).
#[test]
fn glm_image_token_counts_match_the_hf_image_processor() {
    let f = open(&glm_dir());
    let fx = json_fixture("glm53_image_count_cases.json");
    let g = GlmImageGeometry::default();
    assert_eq!(
        (g.patch_size, g.merge_size, g.temporal_patch_size, g.min_image_tokens, g.max_image_tokens),
        (
            fx["processor"]["patch_size"].as_u64().unwrap() as usize,
            fx["processor"]["merge_size"].as_u64().unwrap() as usize,
            fx["processor"]["temporal_patch_size"].as_u64().unwrap() as usize,
            fx["processor"]["min_image_tokens"].as_u64().unwrap() as usize,
            fx["processor"]["max_image_tokens"].as_u64().unwrap() as usize
        ),
        "the defaults are the published pack's processor_config.json"
    );
    let cases = fx["cases"].as_array().unwrap();
    assert!(cases.len() >= 40);
    let (mut min_hit, mut max_hit) = (false, false);
    for c in cases {
        let (h, w) = (c["height"].as_u64().unwrap() as usize, c["width"].as_u64().unwrap() as usize);
        let tokens = c["tokens"].as_u64().unwrap() as usize;
        assert_eq!(f.image_tokens(h, w).unwrap().unwrap(), tokens, "{h}x{w}");
        let (gh, gw) = g.grid(h, w).unwrap();
        assert_eq!(gh * gw, c["patches"].as_u64().unwrap() as usize, "{h}x{w}");
        if let Some(grid) = c["grid"].as_array() {
            assert_eq!((gh, gw), (grid[1].as_u64().unwrap() as usize, grid[2].as_u64().unwrap() as usize), "{h}x{w}");
        }
        min_hit |= tokens == g.min_image_tokens;
        max_hit |= tokens > g.max_image_tokens - 100; // the canvas is aligned to 28 px, so the cap is approached, not hit
    }
    assert!(min_hit && max_hit, "the sweep must reach both the minimum and the maximum token budget");
    // refusals: nothing to count
    assert!(g.tokens(0, 10).is_err() && g.tokens(10, 0).is_err() && g.tokens(1 << 25, 10).is_err());
    // a processor_config.json that disagrees is read, not ignored
    let custom =
        GlmImageGeometry::from_processor_config(&json!({"image_processor": {"max_image_tokens": 100, "min_image_tokens": 4}}))
            .unwrap();
    assert_eq!(custom.tokens(4000, 4000).unwrap(), 100);
    assert!(GlmImageGeometry::from_processor_config(&json!({"image_processor": {"patch_size": 0}})).is_err());
    assert!(GlmImageGeometry::from_processor_config(&json!({"image_processor": {"min_image_tokens": 9, "max_image_tokens": 3}}))
        .is_err());
    assert_eq!(GlmImageGeometry::from_processor_config(&json_fixture("glm53_processor_config.json")).unwrap(), g);
}

// ---------------------------------------------------------------------------------------------------------------------
// GLM tool calls, as the template writes them

/// Feed `text` to a splitter in the given pieces; the visible text and the calls it found.
fn split(f: &dyn ChatFormat, schemas: &ToolSchemas, pieces: &[&str]) -> (String, Vec<ToolCall>) {
    let mut sp: ToolSplitter = f.tool_splitter(schemas.clone());
    let (mut text, mut calls) = (String::new(), Vec::new());
    for p in pieces {
        let (t, c) = sp.push(p);
        text.push_str(&t);
        calls.extend(c);
    }
    let (t, c) = sp.finish();
    text.push_str(&t);
    calls.extend(c);
    (text, calls)
}

fn schemas_of(tools: Value) -> ToolSchemas {
    ToolSchemas::from_request(&json!({"tools": tools})).unwrap()
}

fn weather_tools() -> ToolSchemas {
    schemas_of(json!([
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}, "hourly": {"type": "boolean"}, "units": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "zip": {"type": "string"}, "ratio": {"type": "number"}, "tags": {"type": "array", "items": {"type": "string"}}}}}},
        {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}, "meta": {"type": "object"}}}}},
        {"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}}
    ]))
}

/// Model replies, written the way the GLM template writes an assistant turn (these exact strings come out of it).
const REPLY: &str = "Checking both.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value><arg_key>hourly</arg_key><arg_value>true</arg_value><arg_key>units</arg_key><arg_value>null</arg_value><arg_key>zip</arg_key><arg_value>02134</arg_value><arg_key>tags</arg_key><arg_value>[\"a\", \"\u{e9}\"]</arg_value></tool_call><tool_call>now</tool_call>";

fn reply_calls() -> Vec<(String, Value)> {
    vec![
        (
            "get_weather".into(),
            json!({"city": "Paris", "days": 3, "hourly": true, "units": null, "zip": "02134", "tags": ["a", "\u{e9}"]}),
        ),
        ("now".into(), json!({})),
    ]
}

fn as_pairs(calls: &[ToolCall]) -> Vec<(String, Value)> {
    calls.iter().map(|c| (c.name.clone(), serde_json::from_str(&c.arguments).unwrap())).collect()
}

#[test]
fn glm_tool_calls_are_parsed_typed_and_ordered() {
    let f = open(&glm_dir());
    let (text, calls) = split(&*f, &weather_tools(), &[REPLY]);
    assert_eq!(text, "Checking both.");
    assert_eq!(as_pairs(&calls), reply_calls());
    // the arguments string is compact JSON in the order the model wrote the keys
    assert_eq!(calls[0].arguments, r#"{"city":"Paris","days":3,"hourly":true,"units":null,"zip":"02134","tags":["a","é"]}"#);
    assert_eq!(calls[1].arguments, "{}");
    // text after the calls is kept, text before is not padded with the whitespace that separated it from the call
    let (text, calls) = split(&*f, &weather_tools(), &["Let me look.\n\n<tool_call>now</tool_call>\n\nDone."]);
    assert_eq!((text.as_str(), calls.len()), ("Let me look.\n\nDone.", 1));
    // GLM-4.5 style layout (newline after the name and around the tags)
    let (text, calls) = split(
        &*f,
        &weather_tools(),
        &["<tool_call>write_file\n<arg_key>path</arg_key>\n<arg_value>notes.txt</arg_value>\n<arg_key>content</arg_key>\n<arg_value>42\n  indented\n</arg_value>\n<arg_key>meta</arg_key>\n<arg_value>{\"mode\": \"append\"}</arg_value>\n</tool_call>"],
    );
    assert_eq!(text, "");
    assert_eq!(
        as_pairs(&calls),
        vec![("write_file".to_string(), json!({"path": "notes.txt", "content": "42\n  indented\n", "meta": {"mode": "append"}}))]
    );
    // a function the request did not declare still comes out (the client decides), typed by the untyped rule
    let (_, calls) = split(&*f, &weather_tools(), &["<tool_call>launch<arg_key>when</arg_key><arg_value>now</arg_value><arg_key>n</arg_key><arg_value>2</arg_value></tool_call>"]);
    assert_eq!(as_pairs(&calls), vec![("launch".to_string(), json!({"when": "now", "n": 2}))]);
}

/// Every way of cutting the stream into two pieces, and the stream one character at a time, gives the same text and the same calls.
#[test]
fn glm_tool_calls_stream_the_same_at_every_character_boundary() {
    let f = open(&glm_dir());
    let schemas = weather_tools();
    let texts = [
        REPLY.to_string(),
        "Let me check.\n\n".to_string() + REPLY + "\n\nAnd that is it \u{4e2d}\u{6587} \u{1f600}.",
        // lookalikes of every marker, a call with the newline layout, a malformed call (kept as text), then a good one
        "a <tool b <tool_c <arg_key> </tool_call> <tool_call>write_file\n<arg_key>path</arg_key>\n<arg_value>p\u{e9}</arg_value>\n</tool_call> x <tool_call>oops<arg_key>k</arg_key><arg_value>v</tool_call> y <tool_call>now</tool_call>".to_string(),
        // truncated inside a call and inside a marker
        "ok <tool_call>get_weather<arg_key>city</arg_key><arg_value>Par".to_string(),
        "ok <tool_cal".to_string(),
        "<tool_call>now</tool_call><tool_call>now</tool_call>".to_string(),
    ];
    for text in &texts {
        let whole = split(&*f, &schemas, &[text]);
        for cut in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
            assert_eq!(split(&*f, &schemas, &[&text[..cut], &text[cut..]]), whole, "cut at {cut} of {text:?}");
        }
        let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let refs: Vec<&str> = chars.iter().map(String::as_str).collect();
        assert_eq!(split(&*f, &schemas, &refs), whole, "one character at a time: {text:?}");
    }
    // pinned results of the awkward ones
    let (text, calls) = split(&*f, &schemas, &[&texts[2]]);
    assert_eq!(text, "a <tool b <tool_c <arg_key> </tool_call> x<tool_call>oops<arg_key>k</arg_key><arg_value>v</tool_call> y");
    assert_eq!(as_pairs(&calls), vec![("write_file".to_string(), json!({"path": "p\u{e9}"})), ("now".to_string(), json!({}))]);
    let (text, calls) = split(&*f, &schemas, &[&texts[3]]);
    assert_eq!(
        (text.as_str(), calls.len()),
        ("ok<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par", 0),
        "an unterminated call is shown as written"
    );
    assert_eq!(split(&*f, &schemas, &[&texts[4]]).0, "ok <tool_cal");
    assert_eq!(split(&*f, &schemas, &[&texts[5]]).1.len(), 2);
}

/// The strings the template writes for tool calls in history are the strings the model learned to write: parsing them with a schema that
/// declares the argument types recovers the arguments exactly, for every call in every fixture conversation (floats with exponents,
/// non-ASCII, control characters, nested values, strings that look like numbers).
#[test]
fn glm_parses_back_what_the_template_writes() {
    let f = open(&glm_dir());
    let fx = json_fixture("glm53_chat_template_cases.json");
    let ty = |v: &Value| match v {
        Value::String(_) => "string",
        Value::Bool(_) => "boolean",
        Value::Null => "null",
        Value::Number(n) if n.is_i64() || n.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    };
    let (mut conversations, mut calls_checked, mut args_checked) = (0, 0, 0);
    for c in fx["cases"].as_array().unwrap() {
        let Some(rendered) = c["rendered"].as_str() else { continue };
        let mut want = Vec::new();
        // one declaration per function name for the whole conversation: each key's type is the type of its values
        let mut declared: std::collections::BTreeMap<String, Map<String, Value>> = Default::default();
        for m in c["messages"].as_array().unwrap() {
            for call in m.get("tool_calls").and_then(|c| c.as_array()).into_iter().flatten() {
                let func = call.get("function").unwrap_or(call);
                let args = func["arguments"].as_object().unwrap();
                let name = func["name"].as_str().unwrap().to_string();
                let props = declared.entry(name.clone()).or_default();
                for (k, v) in args {
                    let t = match (props.get(k).and_then(|p| p["type"].as_str()), ty(v)) {
                        (None, t) => t,
                        (Some(a), b) if a == b => b,
                        (Some("integer"), "number") | (Some("number"), "integer") => "number",
                        (Some(a), b) => panic!("{}: `{k}` is {a} in one call and {b} in another", c["name"]),
                    };
                    props.insert(k.clone(), json!({"type": t}));
                }
                want.push((name, Value::Object(args.clone())));
                args_checked += args.len();
            }
        }
        let tools: Vec<Value> = declared
            .into_iter()
            .map(|(name, props)| json!({"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": props}}}))
            .collect();
        if want.is_empty() {
            continue;
        }
        // the history starts after the system block (which shows the call format with placeholders, and is not a call)
        let history = &rendered[rendered.find("<|user|>").unwrap()..];
        let (_, got) = split(&*f, &schemas_of(Value::Array(tools)), &[history]);
        assert_eq!(as_pairs(&got), want, "{}", c["name"]);
        conversations += 1;
        calls_checked += got.len();
    }
    assert!(conversations >= 14 && calls_checked >= 25 && args_checked >= 60, "{conversations} {calls_checked} {args_checked}");
}

// ---------------------------------------------------------------------------------------------------------------------
// the same registry rules on the Nemotron 3.5 Super VL format

/// The registry tests as nemotron/main has them, on the Nemotron format and its fixture (`testdata/nemotron_omni`; the Nemotron-only tree
/// retargeted them from Qwen and GLM): the tests above are the same rules on the Qwen and GLM fixtures, as main has them. Both run.
/// Changed from nemotron/main's text: the two assertions that pinned the Nemotron-only registry (`ids()`) now pin the Nemotron entry
/// within the full one (the tests above pin the full list).
mod nemotron_tests {
    use super::*;

    const FX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/nemotron_omni");
    fn read(name: &str) -> String {
        std::fs::read_to_string(format!("{FX}/{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
    }
    fn json_fixture(name: &str) -> Value {
        serde_json::from_str(&read(name)).unwrap()
    }
    fn nemo_dir() -> Dir {
        Dir::new("nemo", &[("config.json", &read("config.json")), ("chat_template.jinja", &read("chat_template.jinja"))])
    }
    #[test]
    fn a_model_is_matched_by_architecture_then_model_type() {
        let r = FormatRegistry::builtin();
        assert!(r.ids().contains(&"nemotron_omni"), "{:?}", r.ids());
        assert_eq!(r.for_config(&json_fixture("config.json")).unwrap().id, "nemotron_omni");
        // model_type alone, and as the fallback when the architecture is one nobody registered
        assert_eq!(r.for_config(&json!({"model_type": "nemotron_h_omni"})).unwrap().id, "nemotron_omni");
        assert_eq!(
            r.for_config(&json!({"architectures": ["SomethingElse"], "model_type": "nemotron_h_omni"})).unwrap().id,
            "nemotron_omni"
        );
        // the architecture is asked first
        let toy = FormatRegistry::builtin().with(TOY).unwrap();
        assert_eq!(
            toy.for_config(&json!({"architectures": ["NemotronH_Omni_Reasoning_V3"], "model_type": "toy"})).unwrap().id,
            "nemotron_omni"
        );
        assert_eq!(r.by_id("nemotron_omni").unwrap().id, "nemotron_omni");
        // opened from a folder, by config and by id
        assert_eq!(open(&nemo_dir()).id(), "nemotron_omni");
        assert_eq!(r.open(nemo_dir().path(), Some("nemotron_omni")).unwrap().id(), "nemotron_omni");
    }
    #[test]
    fn a_model_nothing_matches_is_refused_by_name_and_never_gets_a_default_format() {
        let r = FormatRegistry::builtin();
        for cfg in [
            json!({"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4"}),
            json!({"architectures": ["Glm4MoeForCausalLM"]}),
            json!({"model_type": "llama"}),
            json!({}),
        ] {
            let e = r.for_config(&cfg).err().unwrap_or_else(|| panic!("{cfg} must not select a format"));
            assert!(
                e.contains("no chat format is registered") && e.contains("nemotron_omni") && e.contains("--chat-format"),
                "{e}"
            );
        }
        let e =
            r.for_config(&json!({"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4"})).err().unwrap();
        assert!(e.contains("Gemma4ForConditionalGeneration") && e.contains("gemma4"), "the refusal names what it saw: {e}");
        let e = r.by_id("llama").err().unwrap();
        assert!(e.contains("unknown chat format `llama`") && e.contains("nemotron_omni"), "{e}");
        // from a folder: an unknown architecture, a missing config.json, an unreadable one, an unknown forced id
        let unknown = Dir::new(
            "unknown",
            &[("config.json", r#"{"architectures": ["Gemma4ForConditionalGeneration"]}"#), ("chat_template.jinja", "x")],
        );
        assert!(r.open(unknown.path(), None).err().unwrap().contains("Gemma4ForConditionalGeneration"));
        let no_config = Dir::new("noconfig", &[("chat_template.jinja", &read("chat_template.jinja"))]);
        assert!(r.open(no_config.path(), None).err().unwrap().contains("config.json"));
        let bad = Dir::new("badconfig", &[("config.json", "{nope"), ("chat_template.jinja", "x")]);
        assert!(r.open(bad.path(), None).err().unwrap().contains("config.json"));
        assert!(r.open(nemo_dir().path(), Some("nope")).err().unwrap().contains("unknown chat format"));
        // naming a format directly works for a folder whose config no entry claims
        assert_eq!(r.open(no_config.path(), Some("nemotron_omni")).unwrap().id(), "nemotron_omni");
    }
    #[test]
    fn an_id_or_a_name_cannot_be_registered_twice() {
        let fake = |id: &'static str, names: &'static [&'static str]| FormatEntry { id, names, open: |_| Err("unused".into()) };
        assert!(FormatRegistry::builtin().with(fake("nemotron_omni", &["x"])).err().unwrap().contains("registered twice"));
        assert!(FormatRegistry::builtin().with(fake("other", &["nemotron_h_omni"])).err().unwrap().contains("claimed by"));
        assert!(FormatRegistry::builtin().with(fake("other", &["other_arch"])).is_ok());
    }
    #[test]
    fn a_new_family_works_through_the_lookup_without_touching_shared_code() {
        let dir = Dir::new(
            "toy",
            &[("config.json", r#"{"architectures": ["ToyForCausalLM"]}"#), ("chat_template.jinja", TOY_TEMPLATE)],
        );
        // the shipped registry has never heard of it ...
        assert!(FormatRegistry::builtin().open(dir.path(), None).err().unwrap().contains("ToyForCausalLM"));
        // ... one registration makes it a first-class format, found by architecture and by id
        let reg = FormatRegistry::builtin().with(TOY).unwrap();
        let ids = reg.ids();
        assert!(ids.contains(&"nemotron_omni") && ids.last() == Some(&"toy"), "{ids:?}");
        let f = reg.open(dir.path(), None).unwrap();
        assert_eq!(f.id(), "toy");
        assert_eq!(reg.open(dir.path(), Some("toy")).unwrap().id(), "toy");
        // the old families still resolve
        assert_eq!(reg.open(nemo_dir().path(), None).unwrap().id(), "nemotron_omni");

        // render: the shared normalisation turns a JSON-string `arguments` into an object before the family's own template sees it
        let messages = json!([{"role": "user", "content": "hi"},
            {"role": "assistant", "content": null, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}]);
        let r = f.render(&messages, &Map::new()).unwrap();
        assert_eq!(r.text, "<user>hi<assistant>[[call]]f|{\"a\": 1}[[/call]]<assistant><reason>");
        assert!(r.starts_in_reasoning);
        assert!(!f.starts_in_reasoning("<assistant>"));

        // the shared tool splitter runs on the family's own markers and parser, however the text is chunked
        let text = "Sure.\n[[call]]f|{\"a\": 1}[[/call]] and [[call]]g|{\"b\": \"\u{e9}\"}[[/call]] done";
        let mut want: Option<(String, Vec<ToolCall>)> = None;
        for cut in (0..=text.len()).filter(|&i| text.is_char_boundary(i)) {
            let mut sp = f.tool_splitter(ToolSchemas::default());
            let (mut t, mut calls) = sp.push(&text[..cut]);
            let (t2, c2) = sp.push(&text[cut..]);
            let (t3, c3) = sp.finish();
            t.push_str(&t2);
            t.push_str(&t3);
            calls.extend(c2);
            calls.extend(c3);
            match &want {
                None => want = Some((t, calls)),
                Some(w) => assert_eq!((&t, &calls), (&w.0, &w.1), "cut at {cut}"),
            }
        }
        let (t, calls) = want.unwrap();
        assert_eq!(t, "Sure. and done");
        assert_eq!(calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["f", "g"]);
        assert_eq!(calls[1].arguments, "{\"b\": \"\u{e9}\"}");

        // the family's own reasoning close tag
        let mut rs = ReasoningSplitter::with_close(true, f.reasoning_close());
        let a = rs.push("think </rea");
        let b = rs.push("son>answer");
        assert_eq!((a.reasoning + &b.reasoning, a.content + &b.content), ("think ".to_string(), "answer".to_string()));

        // media: a text-only default refuses images; the family's predicate drives extraction and no placeholder is invented
        assert!(f.check_media(&json!([{"role": "user", "content": [{"type": "pic", "image_url": "data:x"}]}]), true).is_err());
        let imgs = f
            .extract_images(&json!([{"role": "user", "content": [{"type": "pic", "image": "data:image/png;base64,AA"}, {"type": "image_url", "image_url": "data:y"}]}]), &Default::default())
            .unwrap();
        assert_eq!(imgs.len(), 1);
        assert_eq!(imgs[0].url, "data:image/png;base64,AA");
        let (_d, tok) = mini_tok();
        assert!(f.expand_images(&tok, &[1, 2], &[]).err().unwrap().message.contains("no image placeholder"));
        assert!(!f.serves_images());
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// deepseek_v41 media: the placeholder ingress (encoding.py's _process_image_blocks, mirrored)

#[test]
fn deepseek_v41_serves_images_with_the_packs_placeholder() {
    let f = open_deepseek_v41();
    assert!(f.serves_images());
    let (_d, tok) = deepseek_mini_tok();
    let (id, literal) = f.image_placeholder(&tok).unwrap();
    assert_eq!(literal, "<｜deepseek_image｜>");
    assert_eq!(
        id,
        tok.token_to_id("<｜deepseek_image｜>").unwrap(),
        "the id is whatever the tokenizer says (129264 on the pack)"
    );
}

#[test]
fn deepseek_v41_renders_an_image_part_as_one_placeholder() {
    let f = open_deepseek_v41();
    let msgs = json!([{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
    ]}]);
    f.check_media(&msgs, true).unwrap();
    let r = f.render(&msgs, &Map::new()).unwrap();
    assert_eq!(r.text.matches("<｜deepseek_image｜>").count(), 1, "one placeholder per image part");
}

#[test]
fn deepseek_v41_refuses_a_placeholder_typed_into_text() {
    let f = open_deepseek_v41();
    let msgs = json!([{"role": "user", "content": [
        {"type": "text", "text": "look at <｜deepseek_image｜> please"}
    ]}]);
    let err = f.render(&msgs, &Map::new()).expect_err("the reference raises on a typed-in placeholder");
    assert!(format!("{err}").contains("image placeholder"), "{err}");
}

/// The format opened over the fixture pack dir (reads its `chat_template.jinja`), as the server does.
fn open_deepseek_v41() -> Box<dyn gb_server::chat_format::ChatFormat> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join("deepseek_v41");
    let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let registry = FormatRegistry::builtin();
    let e = registry.for_config(&cfg).unwrap();
    (e.open)(&dir).unwrap()
}

/// A mini tokenizer whose `<｜deepseek_image｜>` sits at id 7, so `image_placeholder` resolves without the real 20 MB file.
fn deepseek_mini_tok() -> (Dir, Tok) {
    let tj = json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [{"id": 7, "content": "<｜deepseek_image｜>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
        "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"a": 0, "[UNK]": 1}, "unk_token": "[UNK]"}
    });
    let d = Dir::new("tok-ds", &[("tokenizer.json", &tj.to_string())]);
    let t = Tok::from_dir(d.path()).unwrap();
    (d, t)
}
