//! Reasoning capabilities as data (`docs/REASONING_CAPABILITIES.md`), end to end through the HTTP layer on the CPU, for four kinds of
//! template: the shipping Nemotron 3.5 Super VL family, a Qwen-style template whose only switch is `enable_thinking` (the Qwen3.8 template
//! fixture), a GLM-style template that always opens a reasoning block and reads only `reasoning_effort` (a minimal template written here: the
//! GLM-5.3 fixture left the tree with that family), and a template with no reasoning at all; plus the inferred descriptor of a family that
//! declares none, a descriptor the template cannot honour (fail-closed, and the expert override), and an extra vendor alias table.
//!
//! The test families are registered only here (`TemplateFamily`, injected through `Opts::format`); the shipping registry stays Nemotron
//! only. The tokenizer is the shared fixture's (`<think>` / `</think>` single ids).

mod common;

use common::{chat_body, http_timeout, sse, start, Opts, Server};
use gb_engine::{Event, FinishReason, Output, Request, ServeEngine};
use gb_server::chat_format::{render_normalized, ChatFormat, FormatEntry, FormatRegistry, RenderedChat};
use gb_server::reasoning_caps::{
    AliasTable, Control, EffortLevel, Markers, Place, ReasoningCapabilities, Source, TemplateVar, ThinkingSwitch, Verdict,
};
use gb_server::reasoning_controls::DefaultThinking;
use gb_server::template::{ChatTemplate, TemplateError};
use gb_server::tools::{ToolSyntax, FUNCTION_BLOCK_TOOL_SYNTAX};
use serde_json::{json, Map, Value};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------- the test-only families

/// A family that is a template and (optionally) a declared descriptor: everything else is the shared defaults.
struct TemplateFamily {
    template: ChatTemplate,
    caps: Option<ReasoningCapabilities>,
}

impl ChatFormat for TemplateFamily {
    fn id(&self) -> &'static str {
        "test_template_family"
    }
    fn render(&self, messages: &Value, kwargs: &Map<String, Value>) -> Result<RenderedChat, TemplateError> {
        let text = render_normalized(&self.template, messages, kwargs)?;
        let starts_in_reasoning = self.starts_in_reasoning(&text);
        Ok(RenderedChat { text, starts_in_reasoning })
    }
    fn render_bounded(
        &self,
        messages: &Value,
        kwargs: &Map<String, Value>,
        max_bytes: usize,
    ) -> Result<RenderedChat, TemplateError> {
        gb_server::template::with_render_limit(max_bytes, || self.render(messages, kwargs))
    }
    fn starts_in_reasoning(&self, prompt: &str) -> bool {
        prompt.trim_end().ends_with("<think>")
    }
    fn tool_syntax(&self) -> ToolSyntax {
        FUNCTION_BLOCK_TOOL_SYNTAX
    }
    fn reasoning_capabilities(&self) -> Option<ReasoningCapabilities> {
        self.caps.clone()
    }
    fn chat_template(&self) -> Option<&ChatTemplate> {
        Some(&self.template)
    }
}

const QWEN: &str = include_str!("fixtures/qwen38_chat_template.jinja");
const NEMO: &str = include_str!("../../../testdata/nemotron_omni/chat_template.jinja");

/// GLM style: the generation prompt always opens a block; the only reasoning variable is `reasoning_effort` (low, high, else max).
const GLM_STYLE: &str =
    "{%- set effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}\
<|system|>Reasoning Effort: {{ effort | capitalize }}{% for m in messages %}<|{{ m.role }}|>\
{% if m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif %}{{ m.content }}{% endfor %}\
{% if add_generation_prompt %}<|assistant|><think>{% endif %}";

/// No reasoning at all.
const PLAIN: &str =
    "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";

fn think_markers() -> Option<Markers> {
    Some(Markers::new("<think>", "</think>"))
}

/// The Qwen-style descriptor: the switch and `preserve_thinking`, no levels.
fn qwen_style() -> ReasoningCapabilities {
    ReasoningCapabilities {
        reasons: true,
        markers: think_markers(),
        switch: Some(ThinkingSwitch { var: "enable_thinking".into(), closes_block: true }),
        template_thinks_by_default: true,
        variables: vec![TemplateVar::boolean("preserve_thinking").default_value(true.into())],
        levels: Vec::new(),
        budget: true,
        aliases: vec![AliasTable::openai()],
    }
}

/// The GLM-style descriptor: a switch the template cannot honour (accepted, noted), three levels on `reasoning_effort`, a budget.
fn glm_style() -> ReasoningCapabilities {
    ReasoningCapabilities {
        reasons: true,
        markers: think_markers(),
        switch: Some(ThinkingSwitch { var: "enable_thinking".into(), closes_block: false }),
        template_thinks_by_default: true,
        variables: Vec::new(),
        levels: ["low", "high", "max"].iter().map(|l| EffortLevel::new(l, true).sets("reasoning_effort", *l)).collect(),
        budget: true,
        aliases: vec![AliasTable::openai()],
    }
}

fn family(src: &str, caps: Option<ReasoningCapabilities>) -> Arc<dyn ChatFormat> {
    Arc::new(TemplateFamily { template: ChatTemplate::new(src).unwrap(), caps })
}

fn serve(format: Arc<dyn ChatFormat>, default_thinking: DefaultThinking, unverified: bool) -> Server {
    start(Opts { format: Some(format), default_thinking, reasoning_unverified: unverified, ..Default::default() })
}

// ---------------------------------------------------------------- helpers

fn post(s: &Server, path: &str, body: &Value) -> common::Reply {
    http_timeout(s.port, "POST", path, Some(body), Duration::from_secs(30))
}

fn chat(s: &Server, extra: Value) -> common::Reply {
    post(s, "/v1/chat/completions", &chat_body(extra))
}

/// The rendered prompt text of a chat request.
fn prompt(s: &Server, extra: Value) -> String {
    let r = post(s, "/v1/chat/completions/render", &chat_body(extra));
    assert_eq!(r.status, 200, "{}", r.body);
    let ids: Vec<u32> = r.json()["token_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    s.state.tok.decode(&ids, false)
}

fn refused(s: &Server, extra: Value) -> Value {
    let r = chat(s, extra.clone());
    assert_eq!(r.status, 400, "{extra}: {}", r.body);
    r.json()["error"].clone()
}

fn reasoning_of(s: &Server) -> Value {
    let (st, body) = common::http(s.port, "GET", "/v1/models", None);
    assert_eq!(st, 200);
    serde_json::from_str::<Value>(&body).unwrap()["data"][0]["reasoning"].clone()
}

fn names(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(|x| x["name"].as_str().unwrap().to_string()).collect()
}

fn id(s: &Server, t: &str) -> u32 {
    s.state.tok.token_to_id(t).unwrap_or_else(|| panic!("{t} is not one token"))
}

/// The ids a request returned, and the reasoning tokens it counted.
fn reply_ids(s: &Server, extra: Value) -> (Vec<u32>, u64) {
    let mut e = extra;
    e["return_token_ids"] = json!(true);
    let r = chat(s, e.clone());
    assert_eq!(r.status, 200, "{e}: {}", r.body);
    let v = r.json();
    let ids = v["choices"][0]["token_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    (ids, v["usage"]["completion_tokens_details"]["reasoning_tokens"].as_u64().unwrap())
}

/// A budget of `n` forces the closing marker at reply index `n` (the CPU mock never writes it on its own in 12 tokens).
fn assert_budget(s: &Server, extra: Value, n: usize) {
    let close = id(s, "</think>");
    let (ids, reasoning) = reply_ids(s, extra.clone());
    assert!(!ids[..n].contains(&close), "the mock closed early; pick another n");
    assert_eq!(ids[n], close, "{extra}: {ids:?}");
    assert_eq!(reasoning, n as u64 + 1, "{extra}");
}

// ---------------------------------------------------------------- a scripted engine (answers chosen ids)

struct Script {
    reply: Arc<Mutex<Vec<u32>>>,
    live: Vec<(u64, Vec<u32>, usize, Request)>,
    next: u64,
    events: Vec<Event>,
    outs: Vec<Output>,
}

impl ServeEngine for Script {
    fn validate(&self, _req: &Request) -> Result<(), gb_engine::SubmitError> {
        Ok(())
    }
    fn would_admit(&self, _req: &Request) -> Result<(), gb_sched::AdmitError> {
        Ok(())
    }
    fn submit_deferred(&mut self, req: Request) -> Result<u64, gb_engine::SubmitError> {
        self.next += 1;
        self.live.push((self.next, self.reply.lock().unwrap().clone(), 0, req));
        Ok(self.next)
    }
    fn busy(&self) -> bool {
        !self.live.is_empty()
    }
    fn step(&mut self) -> Result<bool, gb_engine::StepError> {
        std::thread::sleep(Duration::from_millis(1));
        let mut still = Vec::new();
        for (id, reply, pos, req) in std::mem::take(&mut self.live) {
            let t = reply[pos];
            let finish = if req.stop.contains(&t) || pos + 1 == reply.len() {
                Some(FinishReason::Stop)
            } else if pos + 1 == req.max_new {
                Some(FinishReason::Length)
            } else {
                None
            };
            self.events.push(Event { id, tokens: vec![t], finish });
            match finish {
                Some(f) => self.outs.push(Output {
                    id,
                    tokens: reply[..=pos].to_vec(),
                    finish: Some(f),
                    stopped: f == FinishReason::Stop,
                    ..Default::default()
                }),
                None => still.push((id, reply, pos + 1, req)),
            }
        }
        self.live = still;
        Ok(!self.live.is_empty())
    }
    fn cancel(&mut self, id: u64) -> bool {
        let n = self.live.len();
        self.live.retain(|l| l.0 != id);
        n != self.live.len()
    }
    fn abort_all(&mut self) -> usize {
        std::mem::take(&mut self.live).len()
    }
    fn fail_active(&mut self, _why: &str) -> usize {
        self.abort_all()
    }
    fn active(&self) -> usize {
        self.live.len()
    }
    fn prefilling(&self) -> usize {
        0
    }
    fn take_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }
    fn take_failed(&mut self) -> Vec<(u64, String)> {
        Vec::new()
    }
    fn take_outputs(&mut self) -> Vec<Output> {
        std::mem::take(&mut self.outs)
    }
    fn spec_stats(&self) -> gb_spec::stats::SpecStats {
        Default::default()
    }
}

/// A server whose engine answers `text` (then `<|im_end|>`): the reply split, streaming and not, as (reasoning, content,
/// reasoning_tokens, the X-Rexl3-Note header) for each request.
struct Scripted {
    s: Server,
    reply: Arc<Mutex<Vec<u32>>>,
}

impl Scripted {
    fn new(format: Arc<dyn ChatFormat>) -> Self {
        let reply: Arc<Mutex<Vec<u32>>> = Default::default();
        let r = reply.clone();
        let s = start(Opts {
            format: Some(format),
            engine: Some(Arc::new(move || {
                Box::new(Script { reply: r.clone(), live: Vec::new(), next: 0, events: Vec::new(), outs: Vec::new() })
                    as Box<dyn ServeEngine>
            })),
            ..Default::default()
        });
        Self { s, reply }
    }
    fn says(&self, text: &str) {
        let mut ids = self.s.state.tok.encode(text).unwrap();
        ids.push(id(&self.s, "<|im_end|>"));
        *self.reply.lock().unwrap() = ids;
    }
    /// (reasoning, content, reasoning_tokens, note) of the plain reply, checked equal to the streamed one.
    fn split(&self, extra: Value) -> (Option<String>, String, u64, Option<String>) {
        let mut body = chat_body(json!({"max_tokens": 200}));
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        let r = post(&self.s, "/v1/chat/completions", &body);
        assert_eq!(r.status, 200, "{}", r.body);
        let note = r.header("x-rexl3-note").map(String::from);
        let v = r.json();
        let m = &v["choices"][0]["message"];
        let reasoning = m.get("reasoning_content").and_then(Value::as_str).map(String::from);
        let content = m["content"].as_str().unwrap_or_default().to_string();
        let tokens = v["usage"]["completion_tokens_details"]["reasoning_tokens"].as_u64().unwrap();
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
        let st = post(&self.s, "/v1/chat/completions", &body);
        assert_eq!(st.header("x-rexl3-note").map(String::from), note, "the note is on the stream too");
        let (mut sr, mut sc, mut su) = (String::new(), String::new(), Value::Null);
        for ev in sse(&st.body) {
            if let Some(u) = ev.get("usage").filter(|u| !u.is_null()) {
                su = u.clone();
            }
            let Some(c) = ev["choices"].get(0) else { continue };
            sr += c["delta"]["reasoning_content"].as_str().unwrap_or_default();
            sc += c["delta"]["content"].as_str().unwrap_or_default();
        }
        assert_eq!((sr.as_str(), sc.as_str()), (reasoning.as_deref().unwrap_or_default(), content.as_str()), "{extra}");
        assert_eq!(su["completion_tokens_details"]["reasoning_tokens"], tokens);
        (reasoning, content, tokens, note)
    }
}

// ---------------------------------------------------------------- the shipping registry

#[test]
fn the_shipping_registry_lists_every_family_and_a_test_family_registers_alone() {
    assert_eq!(
        FormatRegistry::builtin().ids(),
        vec!["qwen", "glm5_next", "mimo", "nemotron_omni", "minicpmv", "deepseek_v41", "bailing_moe_v3"]
    );
    const ENTRY: FormatEntry = FormatEntry {
        id: "test_qwen_style",
        names: &["TestQwenStyle"],
        open: |_dir| Ok(Box::new(TemplateFamily { template: ChatTemplate::new(QWEN).unwrap(), caps: Some(qwen_style()) })),
    };
    let reg = FormatRegistry::builtin().with(ENTRY).unwrap();
    let f = reg.open(std::path::Path::new("/nonexistent"), Some("test_qwen_style")).unwrap();
    assert_eq!(f.reasoning_capabilities(), Some(qwen_style()));
}

// ---------------------------------------------------------------- Nemotron (the shipping descriptor)

#[test]
fn nemotron_is_checked_ok_against_its_template_and_advertised() {
    let s = start(Opts::default());
    let sup = &s.state.reasoning_caps;
    assert_eq!((sup.source, sup.verdict), (Source::Declared, Verdict::Ok), "{:?}", sup.findings);
    assert!(sup.unread.is_empty() && sup.undeclared.is_empty() && sup.findings.is_empty(), "{sup:?}");
    assert_eq!((sup.open_id, sup.close_id), (Some(id(&s, "<think>")), Some(id(&s, "</think>"))));
    let log = sup.log_line();
    for part in [
        "event=reasoning_capabilities source=declared verdict=ok",
        "switch=enable_thinking:closes",
        "controls=enable_thinking,medium_effort,truncate_history_thinking,preserve_thinking,reasoning_effort",
        "levels=none,low,medium,high,xhigh",
        "budget=true",
        "top_level=reasoning_effort,max_reasoning_tokens",
    ] {
        assert!(log.contains(part), "{part}: {log}");
    }
    let r = reasoning_of(&s);
    assert_eq!(r["reasons"], true);
    assert_eq!(r["verdict"], "ok");
    assert_eq!(names(&r["controls"]), ["enable_thinking", "medium_effort", "truncate_history_thinking"]);
    assert_eq!(r["accepted_inert"], json!(["preserve_thinking"]));
    assert_eq!(names(&r["levels"]), ["none", "low", "medium", "high", "xhigh"]);
    assert_eq!(
        r["levels"][1],
        json!({"name": "low", "thinks": true, "sets": {"enable_thinking": true, "medium_effort": true}, "default_budget": 1024})
    );
    assert_eq!(r["levels"][3]["requires"], json!({"medium_effort": false}));
    assert_eq!(r["default"], json!({"template_thinks": true, "server": "off", "thinks": false}));
    assert_eq!(r["budget"], json!({"supported": true, "refusal": null}));
    assert_eq!(
        r["markers"],
        json!({"open": "<think>", "close": "</think>", "open_id": id(&s, "<think>"), "close_id": id(&s, "</think>")})
    );
    assert_eq!(
        r["request_fields"],
        json!([
            "reasoning_effort",
            "max_reasoning_tokens",
            "chat_template_kwargs.enable_thinking",
            "chat_template_kwargs.medium_effort",
            "chat_template_kwargs.truncate_history_thinking",
            "chat_template_kwargs.preserve_thinking",
            "chat_template_kwargs.reasoning_effort"
        ])
    );
    let (_, metrics) = common::http(s.port, "GET", "/metrics", None);
    let line = metrics.lines().find(|l| l.starts_with("rexl3_reasoning_info{")).expect("the info series");
    assert!(
        line.contains("verdict=\"ok\"") && line.contains("levels=\"none,low,medium,high,xhigh\"") && line.ends_with(" 1"),
        "{line}"
    );
    // the template default is advertised as the server applies it
    let t = start(Opts { default_thinking: DefaultThinking::Template, ..Default::default() });
    assert_eq!(reasoning_of(&t)["default"], json!({"template_thinks": true, "server": "template", "thinks": true}));
}

// ---------------------------------------------------------------- Qwen style: the switch only

#[test]
fn a_qwen_style_switch_only_model_offers_the_switch_and_a_budget_and_refuses_levels() {
    let s = serve(family(QWEN, Some(qwen_style())), DefaultThinking::Off, false);
    let sup = &s.state.reasoning_caps;
    assert_eq!(sup.verdict, Verdict::Ok, "{:?}", sup.findings);
    assert_eq!(sup.undeclared, ["reasoning_effort"], "the template reads it, the descriptor does not offer it");
    let r = reasoning_of(&s);
    assert_eq!(r["levels"], json!([]));
    assert_eq!(
        r["request_fields"],
        json!(["max_reasoning_tokens", "chat_template_kwargs.enable_thinking", "chat_template_kwargs.preserve_thinking"])
    );
    // 400s name what this model takes
    let e = refused(&s, json!({"reasoning_effort": "high"}));
    assert_eq!(e["param"], "reasoning_effort");
    assert_eq!(
        e["message"],
        "reasoning_effort is not supported by this server; this model takes: max_reasoning_tokens, \
         chat_template_kwargs.enable_thinking, chat_template_kwargs.preserve_thinking"
    );
    let e = refused(&s, json!({"chat_template_kwargs": {"reasoning_effort": "low"}}));
    assert_eq!(
        e["message"],
        "unsupported chat_template_kwargs.reasoning_effort; this model's template takes: enable_thinking, preserve_thinking"
    );
    // default off closes the block; the template default opens it; an explicit switch wins either way
    assert!(prompt(&s, json!({})).ends_with("<think>\n\n</think>\n\n"), "{}", prompt(&s, json!({})));
    assert!(prompt(&s, json!({"chat_template_kwargs": {"enable_thinking": true}})).ends_with("<think>\n"));
    let t = serve(family(QWEN, Some(qwen_style())), DefaultThinking::Template, false);
    assert!(prompt(&t, json!({})).ends_with("<think>\n"));
    assert_eq!(prompt(&t, json!({"chat_template_kwargs": {"enable_thinking": false}})), prompt(&s, json!({})));
    // the budget: refused with thinking off, forced at its index with thinking on
    let e = refused(&s, json!({"max_reasoning_tokens": 4}));
    assert_eq!(e["message"], "max_reasoning_tokens needs thinking on (enable_thinking true)");
    assert_budget(&s, json!({"chat_template_kwargs": {"enable_thinking": true}, "max_reasoning_tokens": 3}), 3);
    assert_budget(&t, json!({"max_reasoning_tokens": 2}), 2);
}

#[test]
fn a_qwen_style_reply_splits_at_its_marker_streaming_and_not() {
    let sc = Scripted::new(family(QWEN, Some(qwen_style())));
    sc.says("Two and two.</think>\n\nFour.");
    let (reasoning, content, tokens, note) = sc.split(json!({"chat_template_kwargs": {"enable_thinking": true}}));
    assert_eq!((reasoning.as_deref(), content.as_str(), note), (Some("Two and two."), "Four.", None));
    assert_eq!(tokens as usize, sc.s.state.tok.encode("Two and two.</think>").unwrap().len());
    // thinking off (the default): all content, a stray marker included
    let (reasoning, content, tokens, _) = sc.split(json!({}));
    assert_eq!((reasoning, content.as_str(), tokens), (None, "Two and two.</think>\n\nFour.", 0));
}

// ---------------------------------------------------------------- GLM style: always thinks, effort only

#[test]
fn a_glm_style_model_maps_its_levels_onto_its_only_variable_and_cannot_turn_thinking_off() {
    let s = serve(family(GLM_STYLE, Some(glm_style())), DefaultThinking::Off, false);
    let sup = &s.state.reasoning_caps;
    assert_eq!(sup.verdict, Verdict::Ok, "{:?}", sup.findings);
    let r = reasoning_of(&s);
    assert_eq!(names(&r["levels"]), ["low", "high", "max"]);
    assert_eq!(r["controls"], json!([{"name": "enable_thinking", "type": "boolean", "role": "switch", "closes_block": false}]));
    assert_eq!(r["default"], json!({"template_thinks": true, "server": "off", "thinks": true}), "nothing can turn it off");
    assert_eq!(
        r["request_fields"],
        json!([
            "reasoning_effort",
            "max_reasoning_tokens",
            "chat_template_kwargs.enable_thinking",
            "chat_template_kwargs.reasoning_effort"
        ])
    );
    // the levels set the template's own variable, by either route
    assert!(prompt(&s, json!({})).contains("Reasoning Effort: Max"));
    for (e, shown) in [("low", "Low"), ("high", "High"), ("max", "Max")] {
        let p = prompt(&s, json!({"reasoning_effort": e}));
        assert!(p.contains(&format!("Reasoning Effort: {shown}")) && p.ends_with("<think>"), "{p}");
        assert_eq!(p, prompt(&s, json!({"chat_template_kwargs": {"reasoning_effort": e}})));
    }
    let e = refused(&s, json!({"reasoning_effort": "none"}));
    assert_eq!(e["message"], "reasoning_effort must be one of low, high, max");
    let e = refused(&s, json!({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "high"}}));
    assert_eq!(e["message"], "reasoning_effort is \"low\" but chat_template_kwargs.reasoning_effort is \"high\"; give one");
    // the switch is accepted and changes nothing in the prompt
    assert_eq!(prompt(&s, json!({"chat_template_kwargs": {"enable_thinking": false}})), prompt(&s, json!({})));
    // a budget: the template always thinks, so it always applies
    assert_budget(&s, json!({"reasoning_effort": "high", "max_reasoning_tokens": 2}), 2);
    assert_budget(&s, json!({"max_reasoning_tokens": 1}), 1);
}

#[test]
fn a_glm_style_reply_thinks_anyway_and_the_response_says_so_when_off_was_asked() {
    let sc = Scripted::new(family(GLM_STYLE, Some(glm_style())));
    sc.says("Hmm.</think>Four.");
    let (reasoning, content, _, note) = sc.split(json!({"chat_template_kwargs": {"enable_thinking": false}}));
    assert_eq!((reasoning.as_deref(), content.as_str()), (Some("Hmm."), "Four."));
    assert_eq!(
        note.as_deref(),
        Some("enable_thinking=false has no effect for this model: its chat template always opens a reasoning block")
    );
    let (reasoning, _, _, note) = sc.split(json!({"reasoning_effort": "low"}));
    assert_eq!((reasoning.as_deref(), note), (Some("Hmm."), None));
}

// ---------------------------------------------------------------- no reasoning

#[test]
fn a_template_with_no_reasoning_offers_nothing_and_never_splits() {
    let s = serve(family(PLAIN, None), DefaultThinking::Off, false);
    let sup = &s.state.reasoning_caps;
    assert_eq!((sup.source, sup.verdict), (Source::Inferred, Verdict::Inferred));
    let r = reasoning_of(&s);
    assert_eq!((r["reasons"].clone(), r["request_fields"].clone()), (json!(false), json!([])));
    assert_eq!(r["budget"], json!({"supported": false, "refusal": "this model does not reason"}));
    for (body, param) in
        [(json!({"reasoning_effort": "high"}), "reasoning_effort"), (json!({"max_reasoning_tokens": 3}), "max_reasoning_tokens")]
    {
        let e = refused(&s, body);
        assert_eq!(e["param"], param);
        assert!(e["message"].as_str().unwrap().ends_with("; this model has no reasoning controls"), "{e}");
    }
    let e = refused(&s, json!({"chat_template_kwargs": {"enable_thinking": false}}));
    assert_eq!(e["message"], "unsupported chat_template_kwargs.enable_thinking; this model's template takes: none");
    assert!(prompt(&s, json!({})).ends_with("<|assistant|>"));
    let sc = Scripted::new(family(PLAIN, None));
    sc.says("a</think>b");
    assert_eq!(sc.split(json!({})), (None, "a</think>b".to_string(), 0, None));
}

// ---------------------------------------------------------------- inference (no descriptor)

#[test]
fn a_family_with_no_descriptor_gets_a_conservative_one_inferred_from_its_template() {
    let s = serve(family(QWEN, None), DefaultThinking::Off, false);
    let c = &s.state.reasoning_caps.caps;
    assert_eq!(s.state.reasoning_caps.verdict, Verdict::Inferred);
    assert_eq!(c.switch, Some(ThinkingSwitch { var: "enable_thinking".into(), closes_block: true }));
    assert_eq!(c.kwarg_names(), ["enable_thinking", "reasoning_effort", "preserve_thinking"]);
    assert!(c.levels.is_empty(), "no levels without a descriptor");
    assert_eq!((c.budget, c.top_level_names()), (true, vec!["max_reasoning_tokens".to_string()]));
    // the template validates the effort itself; the server neither maps nor exposes it at the top level
    assert_eq!(refused(&s, json!({"reasoning_effort": "xhigh"}))["param"], "reasoning_effort");
    let p = prompt(&s, json!({"chat_template_kwargs": {"reasoning_effort": "xhigh"}}));
    assert!(p.contains("Reasoning effort is set to xhigh") && p.ends_with("<think>\n"), "an effort is the client's choice: {p}");
    let e = refused(&s, json!({"chat_template_kwargs": {"reasoning_effort": "high"}}));
    assert!(e["message"].as_str().unwrap().contains("Unexpected reasoning effort high"), "{e}");
    assert!(prompt(&s, json!({})).ends_with("</think>\n\n"), "default off still applies through the inferred switch");
    assert_budget(&s, json!({"chat_template_kwargs": {"enable_thinking": true}, "max_reasoning_tokens": 2}), 2);
}

#[test]
fn a_default_effort_is_rendered_for_a_request_that_names_nothing_and_never_for_one_that_does() {
    let caps = {
        let probe = serve(family(QWEN, None), DefaultThinking::Off, false);
        probe.state.reasoning_caps.caps.clone()
    };
    let level = gb_server::reasoning_controls::default_effort(&caps, "xhigh").expect("the template takes xhigh");
    let s = serve(family(QWEN, None), level, false);
    // the fixture template's own default is low; the configured default replaces it for a request that says nothing
    let p = prompt(&s, json!({}));
    assert!(p.contains("Reasoning effort is set to xhigh") && p.ends_with("<think>\n"), "{p}");
    // a request that names an effort, or turns thinking off, keeps its own
    assert!(prompt(&s, json!({"chat_template_kwargs": {"reasoning_effort": "low"}})).contains("Reasoning effort is set to low"));
    assert!(prompt(&s, json!({"chat_template_kwargs": {"enable_thinking": false}})).ends_with("</think>\n\n"));
    // the switch on, with no effort named, still gets the default effort
    assert!(prompt(&s, json!({"chat_template_kwargs": {"enable_thinking": true}})).contains("Reasoning effort is set to xhigh"));
    // a value the template's own check refuses is caught by rendering what a request that names nothing would render
    let fmt = family(QWEN, None);
    let check = |level: &str| {
        let d = gb_server::reasoning_controls::default_effort(&caps, level).unwrap();
        gb_server::reasoning_controls::check_default_renders(&*fmt, &caps, d)
    };
    assert!(check("xhigh").is_ok());
    assert!(check("high").unwrap_err().contains("Unexpected reasoning effort high"));
}

// ---------------------------------------------------------------- fail-closed, and the override

fn overclaiming() -> ReasoningCapabilities {
    let mut c = gb_server::formats::nemotron_omni::reasoning_capabilities();
    c.variables.push(TemplateVar::boolean("deep_mode"));
    c.levels.push(EffortLevel::new("deep", true).sets("enable_thinking", true).sets("deep_mode", true));
    c
}

#[test]
fn a_claim_the_template_cannot_honour_is_reported_and_not_offered_unless_overridden() {
    let s = serve(family(NEMO, Some(overclaiming())), DefaultThinking::Off, false);
    let sup = &s.state.reasoning_caps;
    assert_eq!(sup.verdict, Verdict::Mismatch);
    assert_eq!(sup.unread, ["deep_mode"]);
    assert!(sup.findings.iter().any(|f| f.contains("level deep: sets deep_mode")), "{:?}", sup.findings);
    assert!(sup.log_line().contains("verdict=mismatch") && sup.log_line().contains("unread=deep_mode"));
    let r = reasoning_of(&s);
    assert_eq!(names(&r["levels"]), ["none", "low", "medium", "high", "xhigh"]);
    assert!(!names(&r["controls"]).contains(&"deep_mode".to_string()));
    let e = refused(&s, json!({"chat_template_kwargs": {"deep_mode": true}}));
    assert!(e["message"].as_str().unwrap().starts_with("unsupported chat_template_kwargs.deep_mode"), "{e}");
    assert_eq!(
        refused(&s, json!({"reasoning_effort": "deep"}))["message"],
        "reasoning_effort must be one of none, low, medium, high, xhigh"
    );
    // the rest of the descriptor is unaffected: byte-identical prompts to the shipping format
    let n = start(Opts::default());
    for e in ["none", "low", "medium", "high", "xhigh"] {
        assert_eq!(prompt(&s, json!({"reasoning_effort": e})), prompt(&n, json!({"reasoning_effort": e})), "{e}");
    }
    // --reasoning-unverified: offered as declared, and said so
    let o = serve(family(NEMO, Some(overclaiming())), DefaultThinking::Off, true);
    assert_eq!(o.state.reasoning_caps.verdict, Verdict::Overridden);
    assert_eq!(reasoning_of(&o)["verdict"], "overridden");
    assert!(names(&reasoning_of(&o)["levels"]).contains(&"deep".to_string()));
    assert_eq!(chat(&o, json!({"reasoning_effort": "deep"})).status, 200);
}

#[test]
fn a_switch_the_template_never_reads_is_downgraded_to_accepted_with_a_note() {
    // the Qwen-style descriptor on the GLM-style template: enable_thinking cannot close the block there
    let s = serve(family(GLM_STYLE, Some(qwen_style())), DefaultThinking::Off, false);
    let sup = &s.state.reasoning_caps;
    assert_eq!(sup.verdict, Verdict::Mismatch);
    assert_eq!(sup.caps.switch, Some(ThinkingSwitch { var: "enable_thinking".into(), closes_block: false }));
    assert_eq!(sup.unread, ["enable_thinking", "preserve_thinking"]);
    let r = chat(&s, json!({"chat_template_kwargs": {"enable_thinking": false}}));
    assert_eq!(r.status, 200, "{}", r.body);
    assert!(r.header("x-rexl3-note").is_some(), "{:?}", r.headers);
}

#[test]
fn a_budget_needs_a_one_token_closing_marker() {
    let src = "{% for m in messages %}{{ m.content }}{% endfor %}<reason>{# </reason> #}";
    let caps = ReasoningCapabilities {
        markers: Some(Markers::new("<reason>", "</reason>")),
        switch: None,
        variables: Vec::new(),
        levels: Vec::new(),
        budget: true,
        template_thinks_by_default: false,
        aliases: vec![AliasTable::openai()],
        ..ReasoningCapabilities::legacy()
    };
    let s = serve(family(src, Some(caps)), DefaultThinking::Off, false);
    let sup = &s.state.reasoning_caps;
    assert_eq!((sup.close_id, sup.verdict), (None, Verdict::Mismatch));
    assert!(sup.budget_refusal.as_deref().unwrap().contains("(</reason>) to be one token"), "{sup:?}");
    assert_eq!(reasoning_of(&s)["budget"]["supported"], false);
    assert_eq!(refused(&s, json!({"max_reasoning_tokens": 3}))["param"], "max_reasoning_tokens");
}

// ---------------------------------------------------------------- another vendor's request shape: one more table, nothing else

#[test]
fn an_extra_alias_table_adds_request_fields_without_server_changes() {
    let mut caps = qwen_style();
    caps.aliases.push(AliasTable::new(
        "vendor_x",
        &[(Place::TopLevel, "enable_reasoning", Control::Switch), (Place::TopLevel, "thinking_budget", Control::Budget)],
    ));
    let s = serve(family(QWEN, Some(caps)), DefaultThinking::Off, false);
    let r = reasoning_of(&s);
    assert_eq!(
        r["request_fields"].as_array().unwrap()[..3],
        [json!("max_reasoning_tokens"), json!("enable_reasoning"), json!("thinking_budget")]
    );
    assert_eq!(
        prompt(&s, json!({"enable_reasoning": true})),
        prompt(&s, json!({"chat_template_kwargs": {"enable_thinking": true}}))
    );
    assert_budget(&s, json!({"enable_reasoning": true, "thinking_budget": 2}), 2);
    let e = refused(&s, json!({"enable_reasoning": true, "chat_template_kwargs": {"enable_thinking": false}}));
    assert_eq!(e["message"], "enable_reasoning is true but chat_template_kwargs.enable_thinking is false; give one");
    let e = refused(&s, json!({"enable_reasoning": true, "thinking_budget": 2, "max_reasoning_tokens": 3}));
    assert_eq!(e["message"], "max_reasoning_tokens is 3 but thinking_budget is 2; give one");
    assert_eq!(refused(&s, json!({"thinking_budget": 2}))["message"], "thinking_budget needs thinking on (enable_thinking true)");
    // /v1/completions refuses the vendor's fields by name too; an unknown field is still unknown
    let r = post(&s, "/v1/completions", &json!({"prompt": "hi", "max_tokens": 2, "thinking_budget": 4}));
    assert_eq!((r.status, r.json()["error"]["param"].clone()), (400, json!("thinking_budget")), "{}", r.body);
    assert_eq!(refused(&s, json!({"reasoning_budget": 1}))["message"], "unsupported request field: reasoning_budget");
    // and the shipping model does not know them
    let n = start(Opts::default());
    assert_eq!(refused(&n, json!({"thinking_budget": 1}))["message"], "unsupported request field: thinking_budget");
}
