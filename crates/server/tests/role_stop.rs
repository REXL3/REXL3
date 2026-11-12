//! A generated chat role token ends a GLM reply. GLM-5.3-Flash (and RED-SNOW, the same family) opens a new turn with a single
//! role token. `generation_config.json`'s eos ends the reply on `<|user|>` and `<|observation|>`, but not on `<|assistant|>`
//! or `<|system|>`; near the context edge a greedy reply can generate `<|assistant|>` after answering and start a second turn
//! (observed on GLM-5.3-Flash two-node at 1,048,576 tokens). The fix adds the GLM role-open tokens to the engine stop set
//! (`ChatFormat::reply_role_stops`, `http::reply_stop_set`), so such a reply finishes with `finish_reason` `stop` and no extra
//! text, and the stop applies the same way to drafted and verified tokens and to streaming and non-streaming.
//!
//! These run in CI without a pack: the GLM chat format is selected by a committed `config.json`, and a byte-level tokenizer
//! carrying the GLM role tokens is assembled from the mock-tokenizer fixture. `tests/tokenizer.rs` pins the real pack's ids.

mod common;

use common::{http, sse, Opts};
use gb_engine::mock::MockBackend;
use gb_engine::{Backend, Drafter, Engine, EngineConfig, ImageSpan, NoDrafts, PrefillStep, RowOutput, RowRequest, ServeEngine};
use gb_sched::{Admission, CostModel, ModelMemory};
use gb_server::chat_format::{ChatFormat, FormatRegistry};
use gb_server::http::reply_stop_set;
use gb_server::tokenizer::Tok;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// Ids the assembled GLM fixture tokenizer gives the role tokens: contiguous after the fixture's 256 byte tokens and its 16
// im_* added tokens (ids 256..=271), so every id stays within the tokenizer's vocabulary (276).
const SYSTEM: u32 = 272;
const USER: u32 = 273;
const ASSISTANT: u32 = 274;
const OBSERVATION: u32 = 275;
const THINK_CLOSE: u32 = 264; // `</think>`, already in the mock fixture (special=false, so it decodes to text and splits reasoning)
const ENDOFTEXT: u32 = 256; // already in the mock fixture

/// A model folder whose `config.json` selects the GLM chat format, with a byte-level tokenizer that carries the GLM role
/// tokens and a `generation_config.json` whose eos is `<|endoftext|>`, `<|user|>`, `<|observation|>` — as the real pack's is,
/// but with the fixture's own ids. Assembled from the committed `glm53_*` fixtures and the mock tokenizer.
fn glm_mock_dir() -> PathBuf {
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let uniq = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("rexl3-glm-rolestop-{}-{uniq}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let mut tok: Value = serde_json::from_str(&std::fs::read_to_string(fx.join("mock_model/tokenizer.json")).unwrap()).unwrap();
    let added = tok["added_tokens"].as_array_mut().unwrap();
    for (id, content) in
        [(SYSTEM, "<|system|>"), (USER, "<|user|>"), (ASSISTANT, "<|assistant|>"), (OBSERVATION, "<|observation|>")]
    {
        added.push(json!({
            "id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
            "normalized": false, "special": true
        }));
    }
    std::fs::write(dir.join("tokenizer.json"), serde_json::to_string(&tok).unwrap()).unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        format!("{{\"eos_token_id\": [{ENDOFTEXT}, {USER}, {OBSERVATION}], \"temperature\": 1.0}}"),
    )
    .unwrap();
    for (from, to) in [
        ("glm53_config_head.json", "config.json"),
        ("glm53_chat_template.jinja", "chat_template.jinja"),
        ("glm53_processor_config.json", "processor_config.json"),
    ] {
        std::fs::copy(fx.join(from), dir.join(to)).unwrap();
    }
    dir
}

/// A backend that emits a fixed token script, so a test can force a reply to generate a chosen token after its answer. It wraps
/// the deterministic mock (keeping its KV/state machine valid for `commit`/`release`) and only replaces the candidate a row
/// returns: the first token comes from the prefill's final row, the rest from each decode step, one scripted token at a time.
/// When the script runs out, the mock's own token is returned. A high logit makes the scripted token the greedy choice.
struct ScriptBackend {
    inner: MockBackend,
    script: Vec<u32>,
    at: usize,
}

impl ScriptBackend {
    fn scripted(&mut self, fallback: RowOutput) -> RowOutput {
        if self.at < self.script.len() {
            let t = self.script[self.at];
            self.at += 1;
            RowOutput { ids: vec![t], values: vec![30.0] }
        } else {
            fallback
        }
    }
}

impl Backend for ScriptBackend {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        let r = self.inner.prefill(slot, start, tokens, candidates);
        self.scripted(r)
    }
    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> PrefillStep {
        let mut step = self.inner.prefill_slice(slot, prompt, images, done, candidates);
        // only the slice that completes the prompt returns the first-token candidates
        if let Some(row) = step.row.take() {
            step.row = Some(self.scripted(row));
        }
        step
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        let outs = self.inner.step(reqs, candidates);
        outs.into_iter().map(|rows| rows.into_iter().map(|r| self.scripted(r)).collect()).collect()
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.inner.commit(slot, keep)
    }
    fn commit_path(&mut self, slot: usize, path: &[usize]) {
        self.inner.commit_path(slot, path)
    }
    fn release(&mut self, slot: usize) {
        self.inner.release(slot)
    }
}

/// A server on the GLM fixture whose engine emits `script` (the generated tokens, in order), everything else the CPU mock.
fn glm_server_emitting(script: Vec<u32>) -> common::Server {
    let dir = glm_mock_dir();
    let vocab = Tok::from_dir(&dir).unwrap().vocab;
    let make = Arc::new(move || {
        let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 1024, state_bytes_per_slot: 1 << 20, activation_reserve: 0 };
        let admission = Admission::new(1u64 << 30, 0, mm, 1).unwrap();
        let backend =
            ScriptBackend { inner: MockBackend::new(vocab, 1, gb_engine::mock::Fault::None), script: script.clone(), at: 0 };
        let engine = Engine::new(
            backend,
            admission,
            EngineConfig { max_rows: 64, max_drafts: 7, cost: CostModel::from_table(vec![(1, 1.0), (64, 2.0)]), drafting: false },
            Box::new(|| Box::new(NoDrafts) as Box<dyn Drafter>),
        );
        Box::new(engine) as Box<dyn ServeEngine>
    });
    common::start(Opts {
        max_model_len: 4096,
        dir: Some(dir),
        slots: 1,
        reasoning_unverified: true,
        engine: Some(make),
        ..Default::default()
    })
}

fn body(stream: bool) -> Value {
    json!({
        "model": "mock",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": stream,
        "stream_options": {"include_usage": true}
    })
}

// Emitted after the GLM generation prompt (which ends `<think>`): close reasoning, answer "4", then the role token that opens a
// second turn, then what the second turn would have copied. The reply must end at the role token.
fn second_turn_script(role: u32) -> Vec<u32> {
    vec![THINK_CLOSE, b'4' as u32, role, THINK_CLOSE, b'4' as u32]
}

#[test]
fn generated_assistant_token_ends_the_reply_non_streaming() {
    let s = glm_server_emitting(second_turn_script(ASSISTANT));
    let (st, b) = http(s.port, "POST", "/v1/chat/completions", Some(&body(false)));
    assert_eq!(st, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
    // the three emitted tokens: </think>, "4", <|assistant|> (the stop). The second turn was never generated.
    assert_eq!(v["usage"]["completion_tokens"], 3, "the reply ran past the role token: {v}");
    let content = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert_eq!(content.matches('4').count(), 1, "the answer was emitted twice (second turn ran): {content:?}");
    // the role token itself is a special token: skip_special_tokens keeps it out of the content and reasoning
    assert!(!content.contains("<|assistant|>"), "the role token leaked into content: {content:?}");
    assert!(
        v["choices"][0]["message"]["reasoning_content"].as_str().unwrap_or("").find("<|assistant|>").is_none(),
        "the role token leaked into reasoning: {v}"
    );
}

#[test]
fn generated_assistant_token_ends_the_reply_streaming() {
    let s = glm_server_emitting(second_turn_script(ASSISTANT));
    let (st, b) = http(s.port, "POST", "/v1/chat/completions", Some(&body(true)));
    assert_eq!(st, 200, "{b}");
    let chunks = sse(&b);
    let finish: Vec<&Value> = chunks
        .iter()
        .filter(|c| c["choices"].as_array().map(|a| !a.is_empty() && !a[0]["finish_reason"].is_null()).unwrap_or(false))
        .collect();
    assert_eq!(finish.len(), 1, "one terminal chunk");
    assert_eq!(finish[0]["choices"][0]["finish_reason"], "stop", "{:?}", finish[0]);
    let usage = chunks.iter().rev().find(|c| !c["usage"].is_null()).expect("a usage chunk");
    assert_eq!(usage["usage"]["completion_tokens"], 3, "the stream ran past the role token: {usage}");
    let content: String = chunks
        .iter()
        .filter_map(|c| c["choices"].as_array().and_then(|a| a.first()).and_then(|c| c["delta"]["content"].as_str()))
        .collect();
    assert_eq!(content.matches('4').count(), 1, "the answer streamed twice (second turn ran): {content:?}");
}

#[test]
fn generated_system_token_also_ends_the_reply() {
    // `<|system|>` opens a turn too and is not an eos; it must stop the reply exactly as `<|assistant|>` does.
    let s = glm_server_emitting(second_turn_script(SYSTEM));
    let (st, b) = http(s.port, "POST", "/v1/chat/completions", Some(&body(false)));
    assert_eq!(st, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "stop", "{v}");
    assert_eq!(v["usage"]["completion_tokens"], 3, "{v}");
}

#[test]
fn the_glm_reply_stop_set_is_eos_plus_assistant_and_system() {
    // the unit the fix turns on: the family's role-open tokens are added to the eos list, once, and nothing else is.
    let dir = glm_mock_dir();
    let tok = Tok::from_dir(&dir).unwrap();
    let format: Box<dyn ChatFormat> = FormatRegistry::builtin().open(&dir, None).unwrap();
    assert_eq!(format.id(), "glm5_next");
    // eos as generation_config.json gives it
    assert_eq!(tok.eos, vec![ENDOFTEXT, USER, OBSERVATION]);
    // the family adds exactly the two role-open tokens that are not already eos, in template order
    assert_eq!(format.reply_role_stops(&tok), vec![ASSISTANT, SYSTEM]);
    // the combined stop set: eos first, then the added role tokens, deduplicated
    assert_eq!(reply_stop_set(&tok, &*format), vec![ENDOFTEXT, USER, OBSERVATION, ASSISTANT, SYSTEM]);
    // tool calls are unaffected: a GLM tool call is `<tool_call>` text and a tool turn ends on `<|observation|>` (already eos);
    // neither the tool-call marker nor `<|observation|>` is added here
    let tool_call = tok.token_to_id("<tool_call>").expect("the fixture has <tool_call>");
    assert!(!format.reply_role_stops(&tok).contains(&tool_call), "a tool-call marker must not be a reply stop");
}

#[test]
fn a_non_glm_family_adds_no_role_stops() {
    // the fix is GLM-family scoped: the default `reply_role_stops` is empty, so a non-GLM family's stop set is exactly its eos.
    let dir = common::fixture_dir(); // the Nemotron 3.5 fixture (im_* chat tokens)
    let tok = Tok::from_dir(&dir).unwrap();
    let format: Box<dyn ChatFormat> = FormatRegistry::builtin().open(&dir, None).unwrap();
    assert!(format.reply_role_stops(&tok).is_empty(), "{} added role stops", format.id());
    assert_eq!(reply_stop_set(&tok, &*format), tok.eos, "the stop set is exactly the eos list");
}
