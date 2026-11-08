//! Reasoning controls, the reasoning/content split and tool calls of the Nemotron 3.5 Super VL format, end to end through the HTTP layer on
//! the CPU (`docs/NEMOTRON_REASONING_API.md`). The model folder is the fixture (`tests/common`): the pack's own `chat_template.jinja` and a
//! byte-level tokenizer whose `<think>` / `</think>` / `<tool_call>` / `</tool_call>` are single non-special ids (as in the pack: 12-15).
//!
//! Two engines: the CPU mock (random text, for the render and budget legs) and [`Script`], an engine that answers ids the test chooses, so
//! the parsing of a reply the model WOULD write (reasoning, then an answer or a tool call, in the template's own format) is checked through
//! the real response code, streaming and not.

mod common;

use common::{chat_body, http_timeout, sse, start, Opts, Server};
use gb_engine::{Event, FinishReason, Output, Request, ServeEngine};
use gb_server::formats::nemotron_omni::LOW_EFFORT_REASONING_TOKENS;
use gb_server::reasoning_controls::DefaultThinking;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn post(s: &Server, path: &str, body: &Value) -> common::Reply {
    http_timeout(s.port, "POST", path, Some(body), Duration::from_secs(30))
}

/// The rendered prompt ids of a chat request (`/v1/chat/completions/render`).
fn render(s: &Server, extra: Value) -> Vec<u32> {
    let r = post(s, "/v1/chat/completions/render", &chat_body(extra));
    assert_eq!(r.status, 200, "{}", r.body);
    r.json()["token_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
}

fn text(s: &Server, ids: &[u32]) -> String {
    s.state.tok.decode(ids, false)
}

fn id(s: &Server, t: &str) -> u32 {
    s.state.tok.token_to_id(t).unwrap_or_else(|| panic!("{t} is not one token"))
}

const TAG: &str = "\n\n{reasoning effort: efficient}";

// ---------------------------------------------------------------- the levels render as documented

#[test]
fn the_effort_levels_render_three_distinct_prompts_and_low_equals_medium() {
    let s = start(Opts::default());
    let (open, close) = (id(&s, "<think>"), id(&s, "</think>"));
    let none = render(&s, json!({"reasoning_effort": "none"}));
    let omitted = render(&s, json!({}));
    let low = render(&s, json!({"reasoning_effort": "low"}));
    let medium = render(&s, json!({"reasoning_effort": "medium"}));
    let high = render(&s, json!({"reasoning_effort": "high"}));
    let xhigh = render(&s, json!({"reasoning_effort": "xhigh"}));
    // none: the closed block (the server's default when nothing is said)
    assert_eq!(none[none.len() - 2..], [open, close], "{}", text(&s, &none));
    assert_eq!(none, omitted, "an omitted effort is thinking off under --default-thinking off");
    assert_eq!(none, render(&s, json!({"chat_template_kwargs": {"enable_thinking": false}})));
    // high: an open block and no tag; xhigh is the same prompt; so is an explicit enable_thinking: true
    assert_eq!(text(&s, &high).rsplit("<|im_start|>").next().unwrap(), "assistant\n<think>\n", "{}", text(&s, &high));
    assert!(!text(&s, &high).contains(TAG));
    assert_eq!(high, xhigh);
    assert_eq!(high, render(&s, json!({"chat_template_kwargs": {"enable_thinking": true}})));
    // medium (and low): the open block, and the template's only efficiency lever on the last user message
    assert_eq!(low, medium, "low and medium render the same prompt (low adds a token budget, not a prompt change)");
    let m = text(&s, &medium);
    assert!(m.contains(&format!("What is 2+2?{TAG}<|im_end|>")), "{m}");
    assert_eq!(m.replace(TAG, ""), text(&s, &high), "medium is high plus the tag");
    assert_eq!(medium, render(&s, json!({"chat_template_kwargs": {"medium_effort": true, "enable_thinking": true}})));
    // the kwargs route and the top-level field render the same
    for e in ["none", "low", "medium", "high", "xhigh"] {
        assert_eq!(
            render(&s, json!({"reasoning_effort": e})),
            render(&s, json!({"chat_template_kwargs": {"reasoning_effort": e}})),
            "{e}"
        );
    }
    // three distinct prompts
    assert!(none != medium && medium != high && none != high);
}

/// Every combination the server accepted before the levels existed renders the same prompt as before, except the sanctioned change:
/// `reasoning_effort` low or medium with thinking on now carries the effort tag.
#[test]
fn every_previously_accepted_combination_renders_as_before_but_low_and_medium_gain_the_tag() {
    let s = start(Opts::default());
    let msgs = json!([{"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "earlier thought"},
        {"role": "user", "content": "q2"}]);
    let render_with = |kw: &serde_json::Map<String, Value>| s.state.format.render(&msgs, kw).unwrap().text;
    let mut changed = 0;
    for think in [None, Some(true), Some(false)] {
        for preserve in [None, Some(true), Some(false)] {
            for effort in [None, Some("low"), Some("medium"), Some("high"), Some("xhigh")] {
                let mut kw = serde_json::Map::new();
                if let Some(t) = think {
                    kw.insert("enable_thinking".into(), json!(t));
                }
                if let Some(p) = preserve {
                    kw.insert("preserve_thinking".into(), json!(p));
                }
                if let Some(e) = effort {
                    kw.insert("reasoning_effort".into(), json!(e));
                }
                let body = json!({"messages": msgs, "chat_template_kwargs": kw});
                gb_server::field_policy::validate_chat(&body, &Default::default()).expect("the old policy accepted it");
                let old = render_with(&gb_server::api::chat_template_kwargs(&body).unwrap());
                let new = render_with(&s.state.reasoning(&body).unwrap().kwargs);
                if matches!(effort, Some("low" | "medium")) && think != Some(false) {
                    assert_eq!(new.replace(TAG, ""), old, "{body}");
                    assert_ne!(new, old, "{body}: the tag");
                    changed += 1;
                } else {
                    assert_eq!(new, old, "{body}");
                }
            }
        }
    }
    assert_eq!(changed, 2 * 2 * 3, "low/medium x enable_thinking absent/true x preserve_thinking absent/true/false");
}

#[test]
fn default_thinking_template_follows_the_templates_own_default_which_is_on() {
    let off = start(Opts::default());
    let tmpl = start(Opts { default_thinking: DefaultThinking::Template, ..Default::default() });
    assert_eq!(render(&tmpl, json!({})), render(&off, json!({"reasoning_effort": "high"})));
    // an explicit choice is unaffected by the flag
    for extra in [json!({"chat_template_kwargs": {"enable_thinking": false}}), json!({"reasoning_effort": "medium"})] {
        assert_eq!(render(&tmpl, extra.clone()), render(&off, extra));
    }
}

#[test]
fn history_thinking_is_truncated_by_default_and_kept_on_request() {
    let s = start(Opts::default());
    let msgs = json!([{"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "secret thought"},
        {"role": "user", "content": "q2"}]);
    let dflt = text(&s, &render(&s, json!({"messages": msgs})));
    assert!(!dflt.contains("secret thought"), "{dflt}");
    let kept = text(&s, &render(&s, json!({"messages": msgs, "chat_template_kwargs": {"truncate_history_thinking": false}})));
    assert!(kept.contains("<think>\nsecret thought</think>a1"), "{kept}");
    // preserve_thinking is accepted as before and renders exactly as before (this template does not read it)
    assert_eq!(dflt, text(&s, &render(&s, json!({"messages": msgs, "chat_template_kwargs": {"preserve_thinking": true}}))));
}

#[test]
fn unsupported_controls_are_refused_with_what_is_supported() {
    let s = start(Opts::default());
    let r = post(&s, "/v1/chat/completions", &chat_body(json!({"chat_template_kwargs": {"thinking": true}})));
    assert_eq!(r.status, 400, "{}", r.body);
    let msg = r.json()["error"]["message"].as_str().unwrap().to_string();
    for k in ["enable_thinking", "medium_effort", "truncate_history_thinking", "preserve_thinking", "reasoning_effort"] {
        assert!(msg.contains(k), "{msg}");
    }
    for body in [
        json!({"reasoning_effort": "minimal"}),
        json!({"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": true}}),
        json!({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "high"}}),
        json!({"max_reasoning_tokens": 16}), // thinking is off by default: nothing to cap
        json!({"max_reasoning_tokens": -1, "reasoning_effort": "high"}),
        json!({"chat_template_kwargs": {"force_nonempty_content": true}}),
    ] {
        let r = post(&s, "/v1/chat/completions", &chat_body(body.clone()));
        assert_eq!(r.status, 400, "{body}: {}", r.body);
    }
    for (field, v) in [("reasoning_effort", json!("high")), ("max_reasoning_tokens", json!(3))] {
        let r = post(&s, "/v1/completions", &json!({"prompt": "hi", "max_tokens": 2, field: v}));
        assert_eq!(r.status, 400, "{}", r.body);
        assert_eq!(r.json()["error"]["param"], field);
    }
    // the switches the template reads are accepted
    let r = post(
        &s,
        "/v1/chat/completions",
        &chat_body(json!({"chat_template_kwargs": {"medium_effort": true, "truncate_history_thinking": true}})),
    );
    assert_eq!(r.status, 200, "{}", r.body);
}

// ---------------------------------------------------------------- a scripted engine

/// An engine that answers every request with the ids in `reply` (one token a step), ending on a stop id, at `max_new`, or (as at an
/// end-of-turn) when the script runs out.
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
        if self.live.len() < 8 {
            Ok(())
        } else {
            Err(gb_sched::AdmitError::NoSlot)
        }
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
        // a step takes time, as the mock's `step_ms` does: a zero-time engine outruns the 64-message reply channel on a long script
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

struct Scripted {
    s: Server,
    reply: Arc<Mutex<Vec<u32>>>,
}

impl Scripted {
    fn new() -> Self {
        let reply: Arc<Mutex<Vec<u32>>> = Default::default();
        let r = reply.clone();
        let s = start(Opts {
            engine: Some(Arc::new(move || {
                Box::new(Script { reply: r.clone(), live: Vec::new(), next: 0, events: Vec::new(), outs: Vec::new() })
                    as Box<dyn ServeEngine>
            })),
            ..Default::default()
        });
        Self { s, reply }
    }
    /// Make the engine answer `text` (as the tokenizer encodes it) followed by `<|im_end|>`.
    fn says(&self, text: &str) {
        let mut ids = self.s.state.tok.encode(text).unwrap();
        ids.push(id(&self.s, "<|im_end|>"));
        *self.reply.lock().unwrap() = ids;
    }
    fn chat(&self, extra: Value) -> Value {
        let mut body = chat_body(json!({"max_tokens": 400}));
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        let r = post(&self.s, "/v1/chat/completions", &body);
        assert_eq!(r.status, 200, "{}", r.body);
        r.json()
    }
    /// The same request streamed: (reasoning, content, tool_calls, finish_reason, usage) from the SSE deltas.
    fn stream(&self, extra: Value) -> (String, String, Vec<Value>, String, Value) {
        let mut body = chat_body(json!({"max_tokens": 400, "stream": true, "stream_options": {"include_usage": true}}));
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        let r = post(&self.s, "/v1/chat/completions", &body);
        assert_eq!(r.status, 200, "{}", r.body);
        let (mut reasoning, mut content, mut calls, mut finish, mut usage) =
            (String::new(), String::new(), Vec::new(), String::new(), Value::Null);
        for ev in sse(&r.body) {
            if let Some(u) = ev.get("usage").filter(|u| !u.is_null()) {
                usage = u.clone();
            }
            let Some(c) = ev["choices"].get(0) else { continue };
            let d = &c["delta"];
            if let Some(t) = d["reasoning_content"].as_str() {
                assert_eq!(d["reasoning"], d["reasoning_content"], "both names carry the same text");
                reasoning += t;
            }
            if let Some(t) = d["content"].as_str() {
                content += t;
            }
            if let Some(tc) = d["tool_calls"].as_array() {
                calls.extend(tc.iter().cloned());
            }
            if let Some(f) = c["finish_reason"].as_str() {
                finish = f.to_string();
            }
        }
        (reasoning, content, calls, finish, usage)
    }
}

const ON: fn() -> Value = || json!({"reasoning_effort": "high"});

fn tokens_through(s: &Server, text: &str, marker: &str) -> usize {
    // the reply's tokens up to and including the marker
    let at = text.find(marker).unwrap() + marker.len();
    s.state.tok.encode(&text[..at]).unwrap().len()
}

#[test]
fn reasoning_and_content_are_split_the_same_way_streaming_and_not() {
    let sc = Scripted::new();
    let reply = "Two and two make four.</think>\n\nThe answer is 4.";
    sc.says(reply);
    let r = sc.chat(ON());
    let m = &r["choices"][0]["message"];
    assert_eq!(m["reasoning_content"], "Two and two make four.");
    assert_eq!(m["reasoning"], m["reasoning_content"]);
    assert_eq!(m["content"], "The answer is 4.");
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    let thinking = tokens_through(&sc.s, reply, "</think>");
    assert_eq!(r["usage"]["completion_tokens_details"]["reasoning_tokens"], thinking, "through the closing marker");
    assert_eq!(r["usage"]["completion_tokens"], sc.s.state.tok.encode(reply).unwrap().len() + 1, "and the end-of-turn");
    let (reasoning, content, calls, finish, usage) = sc.stream(ON());
    assert_eq!(
        (reasoning.as_str(), content.as_str(), calls.len(), finish.as_str()),
        ("Two and two make four.", "The answer is 4.", 0, "stop")
    );
    assert_eq!(usage, r["usage"]);
}

#[test]
fn thinking_off_is_all_content_even_a_stray_closing_tag() {
    // the pack parser (`ultra_v3_reasoning_parser.py`): with enable_thinking false everything generated is content
    let sc = Scripted::new();
    sc.says("Four.</think> really");
    for extra in [json!({}), json!({"reasoning_effort": "none"})] {
        let r = sc.chat(extra.clone());
        let m = &r["choices"][0]["message"];
        assert_eq!(m["content"], "Four.</think> really", "{extra}");
        assert!(m.get("reasoning_content").is_none(), "{r}");
        assert_eq!(r["usage"]["completion_tokens_details"]["reasoning_tokens"], 0);
        let (reasoning, content, _, _, _) = sc.stream(extra);
        assert_eq!((reasoning.as_str(), content.as_str()), ("", "Four.</think> really"));
    }
}

#[test]
fn a_budget_that_ends_inside_the_thought_finishes_length_with_empty_content_and_the_reasoning_kept() {
    let sc = Scripted::new();
    sc.says("I am still thinking about this for a long while.</think>\n\nDone.");
    let r = sc.chat(json!({"reasoning_effort": "high", "max_tokens": 8}));
    let c = &r["choices"][0];
    assert_eq!(c["finish_reason"], "length");
    assert_eq!(c["message"]["content"], "");
    assert!(!c["message"]["reasoning_content"].as_str().unwrap().is_empty(), "{r}");
    assert_eq!(r["usage"]["completion_tokens"], 8);
    assert_eq!(r["usage"]["completion_tokens_details"]["reasoning_tokens"], 8, "every token was thinking");
    let (reasoning, content, _, finish, _) = sc.stream(json!({"reasoning_effort": "high", "max_tokens": 8}));
    assert_eq!((content.as_str(), finish.as_str()), ("", "length"));
    assert_eq!(reasoning, c["message"]["reasoning_content"].as_str().unwrap());
}

// ---------------------------------------------------------------- tool calls

fn weather() -> Value {
    json!([{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                       "required": ["city"]}}}])
}

/// What the model writes when it calls `get_weather`, in the format the template's system prompt prescribes.
const CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";

#[test]
fn tools_render_in_the_templates_format_and_tool_choice_none_hides_them() {
    let s = start(Opts::default());
    let with = text(&s, &render(&s, json!({"tools": weather()})));
    for part in [
        "# Tools",
        "<tools>\n<function>\n<name>get_weather</name>",
        "<parameter>\n<name>city</name>\n<type>string</type>",
        "<required>[\"city\"]</required>",
        "<tool_call>\n<function=example_function_name>",
    ] {
        assert!(with.contains(part), "{part} missing from:\n{with}");
    }
    let auto = text(&s, &render(&s, json!({"tools": weather(), "tool_choice": "auto"})));
    assert_eq!(with, auto);
    let none = text(&s, &render(&s, json!({"tools": weather(), "tool_choice": "none"})));
    assert!(!none.contains("<tools>"), "{none}");
    assert_eq!(none, text(&s, &render(&s, json!({}))));
    for forced in [json!("required"), json!({"type": "function", "function": {"name": "get_weather"}})] {
        let r = post(&s, "/v1/chat/completions", &chat_body(json!({"tools": weather(), "tool_choice": forced})));
        assert_eq!(r.status, 400, "{}", r.body);
        assert!(r.body.contains("tool_choice"));
    }
}

#[test]
fn a_tool_call_after_reasoning_becomes_openai_tool_calls_streaming_and_not() {
    let sc = Scripted::new();
    for think in [json!({"reasoning_effort": "high"}), json!({"reasoning_effort": "none"})] {
        let mut extra = json!({"tools": weather()});
        extra.as_object_mut().unwrap().extend(think.as_object().unwrap().clone());
        if think["reasoning_effort"] == "none" {
            sc.says(CALL); // with thinking off the model answers with the call directly
        } else {
            sc.says(&format!("The user wants the weather.</think>\n\n{CALL}"));
        }
        let r = sc.chat(extra.clone());
        let c = &r["choices"][0];
        assert_eq!(c["finish_reason"], "tool_calls", "{r}");
        assert!(c["message"]["content"].is_null(), "{r}");
        let tc = &c["message"]["tool_calls"];
        assert_eq!(tc.as_array().unwrap().len(), 1);
        assert_eq!(tc[0]["type"], "function");
        assert!(tc[0]["id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(tc[0]["function"]["name"], "get_weather");
        let args: Value = serde_json::from_str(tc[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({"city": "Paris", "days": 3}), "typed from the schema");
        let (reasoning, content, calls, finish, _) = sc.stream(extra);
        assert_eq!((content.as_str(), finish.as_str(), calls.len()), ("", "tool_calls", 1));
        assert_eq!(calls[0]["index"], 0);
        assert_eq!(calls[0]["function"]["arguments"], tc[0]["function"]["arguments"]);
        assert_eq!(reasoning, c["message"]["reasoning_content"].as_str().unwrap_or(""));
    }
}

#[test]
fn a_tool_call_inside_the_thought_stays_reasoning() {
    let sc = Scripted::new();
    sc.says(&format!("Maybe {CALL} would do.</think>\n\nIt is sunny."));
    let r = sc.chat(json!({"tools": weather(), "reasoning_effort": "high"}));
    let m = &r["choices"][0]["message"];
    assert!(m.get("tool_calls").is_none(), "{r}");
    assert!(m["reasoning_content"].as_str().unwrap().contains("<function=get_weather>"));
    assert_eq!(m["content"], "It is sunny.");
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
}

#[test]
fn malformed_calls_stay_visible_and_tool_choice_none_parses_nothing() {
    let sc = Scripted::new();
    // a block the template would never produce: shown as written
    sc.says("<tool_call>\nnot a function\n</tool_call>");
    let r = sc.chat(json!({"tools": weather()}));
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    assert_eq!(r["choices"][0]["message"]["content"], "<tool_call>\nnot a function\n</tool_call>");
    // cut by max_tokens inside the call: finish length, the partial call is text
    sc.says(CALL);
    let r = sc.chat(json!({"tools": weather(), "max_tokens": 6}));
    assert_eq!(r["choices"][0]["finish_reason"], "length");
    assert!(r["choices"][0]["message"]["content"].as_str().unwrap().starts_with("<tool_call>"), "{r}");
    assert!(r["choices"][0]["message"].get("tool_calls").is_none());
    // tool_choice none: the tools were not offered, so call-shaped text is content
    let r = sc.chat(json!({"tools": weather(), "tool_choice": "none"}));
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    assert_eq!(r["choices"][0]["message"]["content"], CALL);
}

#[test]
fn a_multi_turn_tool_conversation_renders_and_answers() {
    let sc = Scripted::new();
    let messages = json!([
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": null, "reasoning_content": "Need the tool.",
         "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 3}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"sky\": \"sunny\"}"},
        {"role": "tool", "tool_call_id": "call_2", "content": "second result"}
    ]);
    let ids = render(&sc.s, json!({"messages": messages, "tools": weather(), "reasoning_effort": "high"}));
    let t = text(&sc.s, &ids);
    assert!(
        t.contains("<|im_start|>assistant\n<think>\nNeed the tool.</think>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\n<|im_end|>"),
        "{t}"
    );
    assert!(
        t.contains("<|im_start|>user\n<tool_response>\n{\"sky\": \"sunny\"}\n</tool_response>\n<tool_response>\nsecond result\n</tool_response>\n<|im_end|>"),
        "{t}"
    );
    assert!(t.ends_with("<|im_start|>assistant\n<think>\n"), "{t}");
    sc.says("Sunny.</think>\n\nIt is sunny in Paris.");
    let r = sc.chat(json!({"messages": messages, "tools": weather(), "reasoning_effort": "high"}));
    assert_eq!(r["choices"][0]["message"]["content"], "It is sunny in Paris.");
    // arguments that are not a JSON object are refused before rendering
    let bad = json!([{"role": "user", "content": "x"},
        {"role": "assistant", "content": null, "tool_calls": [{"id": "c", "type": "function", "function": {"name": "get_weather", "arguments": "[1]"}}]}]);
    let r = post(&sc.s, "/v1/chat/completions", &chat_body(json!({"messages": bad, "tools": weather()})));
    assert_eq!(r.status, 400, "{}", r.body);
}

// ---------------------------------------------------------------- the reasoning budget through the server (CPU mock engine)

#[test]
fn a_reasoning_budget_forces_the_closing_marker_at_its_index() {
    let s = start(Opts::default());
    let close = id(&s, "</think>");
    for n in [0usize, 1, 5] {
        let body =
            chat_body(json!({"reasoning_effort": "high", "max_reasoning_tokens": n, "max_tokens": 12, "return_token_ids": true}));
        let r = post(&s, "/v1/chat/completions", &body);
        assert_eq!(r.status, 200, "{}", r.body);
        let v = r.json();
        let ids: Vec<u64> = v["choices"][0]["token_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        assert!(!ids[..n].contains(&(close as u64)), "the mock closed early; pick another n");
        assert_eq!(ids[n], close as u64, "{n}: {ids:?}");
        assert_eq!(v["usage"]["completion_tokens_details"]["reasoning_tokens"], n + 1, "{v}");
        // the same request again, and serially: the same ids
        let again = post(&s, "/v1/chat/completions", &{
            let mut b = body.clone();
            b["draft"] = json!(false);
            b
        });
        assert_eq!(again.json()["choices"][0]["token_ids"], v["choices"][0]["token_ids"]);
    }
    // low: thinking with the tag and the default budget, which a 12-token reply never reaches
    let r = post(
        &s,
        "/v1/chat/completions",
        &chat_body(json!({"reasoning_effort": "low", "max_tokens": 12, "return_token_ids": true})),
    );
    assert_eq!(r.status, 200);
    let v = r.json();
    assert_eq!(
        v["choices"][0]["finish_reason"], "length",
        "a 12-token reply never reaches the {LOW_EFFORT_REASONING_TOKENS}-token budget"
    );
    assert_eq!(v["usage"]["completion_tokens_details"]["reasoning_tokens"], v["usage"]["completion_tokens"]);
}
