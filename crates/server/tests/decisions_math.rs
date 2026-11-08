//! Native decisions conformance. CPU/mock only; no optional skip when the route is absent.
mod common;
use common::{http, start, Opts};
use gb_server::chat_format::{ChatFormat, FormatRegistry};
use gb_server::decisions::{build_response, check_labels, logsumexp_full_row, prepare, softmax_labels, ScoredQuestion};
use gb_server::tokenizer::Tok;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

// Separate synthetic BPE fixture: do NOT modify baseline tokenizer/expected generation outputs.
// These three merges exercise literal yes/no as single contextual tokens, not renamed y/n labels.
fn decision_dir() -> PathBuf {
    static D: OnceLock<PathBuf> = OnceLock::new();
    D.get_or_init(|| {
        let src = common::qwen_fixture_dir();
        let dir = std::env::temp_dir().join(format!("rexl3-score-fixture-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for f in ["config.json", "generation_config.json", "chat_template.jinja"] {
            std::fs::copy(src.join(f), dir.join(f)).unwrap();
        }
        let mut t: Value = serde_json::from_slice(&std::fs::read(src.join("tokenizer.json")).unwrap()).unwrap();
        t["model"]["vocab"]["ye"] = json!(272);
        t["model"]["vocab"]["yes"] = json!(273);
        t["model"]["vocab"]["no"] = json!(274);
        t["model"]["merges"] = json!([["y", "e"], ["ye", "s"], ["n", "o"]]);
        std::fs::write(dir.join("tokenizer.json"), serde_json::to_vec(&t).unwrap()).unwrap();
        dir
    })
    .clone()
}
fn env() -> (Tok, Arc<dyn ChatFormat>) {
    let d = decision_dir();
    (Tok::from_dir(&d).unwrap(), FormatRegistry::builtin().open(&d, None).unwrap().into())
}
fn choice(id: &str, n: usize) -> Value {
    json!({"id":id,"type":"choice","question":"Which?","options":(0..n).map(|i|json!({"name":format!("option{i}")})).collect::<Vec<_>>()})
}
fn body() -> Value {
    json!({"input":"GPU memory", "questions":[choice("topic",3),{"id":"quality","type":"score","question":"How clear?","levels":["bad","okay","good"]},{"id":"relevant","type":"yes_no","question":"About GPUs?","yes":"hardware","no":"other"}],"return_prompt_token_ids":true})
}
fn prep(v: Value) -> Vec<gb_server::decisions::PreparedDecision> {
    let (t, f) = env();
    prepare(&t, &*f, v, 65536).unwrap()
}
fn server(o: Opts) -> common::Server {
    start(Opts { dir: Some(decision_dir()), ..o })
}
fn post(s: &common::Server, v: &Value) -> (u16, Value) {
    let (code, b) = http(s.port, "POST", "/v1/decisions", Some(v));
    (code, serde_json::from_str(&b).unwrap())
}
fn assert_shape(v: &Value) {
    assert_eq!(v["object"], "decisions");
    assert_eq!(v["prompt_format_version"], 1);
    assert_eq!(v["usage"]["completion_tokens"], 0);
    assert_eq!(v["usage"]["total_tokens"], v["usage"]["prompt_tokens"]);
    assert!(v["usage"].get("spec").is_none());
    for a in v["answers"].as_object().unwrap().values() {
        let p = a["probabilities"].as_object().unwrap();
        let sum: f64 = p.values().map(|x| x.as_f64().unwrap()).sum();
        assert!((sum - 1.0).abs() < 1e-12);
        let m = a["label_mass"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&m));
    }
}
#[test]
fn http_all_types_real_route_and_no_generation() {
    let s = server(Opts::default());
    let p = prep(body());
    let (code, v) = post(&s, &body());
    assert_eq!(code, 200, "{v}");
    assert_shape(&v);
    assert_eq!(v["model"], "mock");
    assert_eq!(v["answers"]["relevant"]["type"], "yes_no");
    assert!(v["answers"]["relevant"].get("choice").is_none());
    assert!(v["answers"]["relevant"]["label_token_ids"].get("yes").is_some());
    let expected = p.iter().map(|q| q.prompt.len()).sum::<usize>();
    assert_eq!(v["usage"]["prompt_tokens"], expected);
    for q in &p {
        let refrow = gb_engine::mock::mock_logits(&q.prompt, 4096);
        let probs = softmax_labels(&q.label_ids.iter().map(|&id| refrow[id as usize]).collect::<Vec<_>>(), 1.0);
        let a = &v["answers"][&q.id];
        for (name, prob) in q.names.iter().zip(probs) {
            assert!((a["probabilities"][name].as_f64().unwrap() - prob).abs() < 1e-12);
        }
    }
    let m = &s.handle().metrics;
    assert_eq!(m.completion_tokens.load(Relaxed), 0);
    assert_eq!(m.rounds.load(Relaxed), 0);
    assert_eq!(m.drafted.load(Relaxed), 0);
    assert_eq!(m.accepted.load(Relaxed), 0);
    assert_eq!(m.requests.load(Relaxed), 0);
    // the runtime bumps the counter after it hands the row to the HTTP thread, so the response can
    // reach the client first: wait for the third completion instead of reading it once
    s.wait(Duration::from_secs(2), "three decisions completed", |s| s.handle().metrics.decisions_completed.load(Relaxed) == 3);
    assert_eq!(m.decisions_completed.load(Relaxed), 3);
    let (code, again) = post(&s, &body());
    assert_eq!(code, 200);
    assert_eq!(v, again);
    let mut noids = body();
    noids["return_prompt_token_ids"] = json!(false);
    let (code, v) = post(&s, &noids);
    assert_eq!(code, 200);
    assert!(v["answers"]["topic"].get("prompt_token_ids").is_none());
    let (_, metrics) = http(s.port, "GET", "/metrics", None);
    assert!(metrics.contains("rexl3_decisions_total{status=\"completed\"}"));
}
#[test]
fn choice_and_score_all_allowed_sizes() {
    let s = server(Opts { max_model_len: 65536, ..Default::default() });
    for n in 2..=26 {
        let v = json!({"input":"x","questions":[choice("c",n)]});
        let (code, r) = post(&s, &v);
        assert_eq!(code, 200, "n={n} {r}");
        assert_eq!(r["answers"]["c"]["probabilities"].as_object().unwrap().len(), n);
        assert_shape(&r);
    }
    for n in 2..=10 {
        let v = json!({"input":"x","questions":[{"id":"s","type":"score","question":"How?","levels":vec!["level";n]}]});
        let (code, r) = post(&s, &v);
        assert_eq!(code, 200, "{r}");
        let score = r["answers"]["s"]["score"].as_f64().unwrap();
        assert!((0.0..=n as f64 - 1.0).contains(&score));
    }
}
fn invalids() -> Vec<Value> {
    let mut out = vec![];
    for (field, vals) in [
        ("temperature", vec![json!(-1), json!("1"), json!(true), Value::Null]),
        ("prompt_format_version", vec![json!(2), json!(1.0), json!(true), Value::Null]),
        ("return_prompt_token_ids", vec![json!(1), Value::Null]),
        ("model", vec![json!(1), Value::Null]),
        (
            "chat_template_kwargs",
            vec![
                json!({"enable_thinking":true}),
                json!({"enable_thinking":null}),
                json!({"unknown":false}),
                json!([]),
                Value::Null,
            ],
        ),
        ("input", vec![json!(" "), json!(1), json!(false), Value::Null]),
        ("questions", vec![json!([]), json!("x"), Value::Null]),
    ] {
        for val in vals {
            let mut b = body();
            b[field] = val;
            out.push(b);
        }
    }
    let mut b = body();
    b["unknown"] = json!(1);
    out.push(b);
    let mut b = body();
    b["questions"] = json!([choice("same", 2), choice("same", 2)]);
    out.push(b);
    for id in [json!(" "), json!(1), Value::Null] {
        let mut b = body();
        b["questions"][0]["id"] = id;
        out.push(b);
    }
    for n in [1, 27] {
        out.push(json!({"input":"x","questions":[choice("c",n)]}));
    }
    for n in [1, 11] {
        out.push(json!({"input":"x","questions":[{"id":"s","type":"score","question":"q","levels":vec!["l";n]}]}));
    }
    for q in [
        json!({"id":"q","type":"choice","question":"q","options":[{"name":"One"},{"name":"one"}]}),
        json!({"id":"q","type":"choice","question":"q","options":[{"name":"Straße"},{"name":"STRASSE"}]}),
        json!({"id":"q","type":"choice","question":"q","options":[{"name":"a\nb"},{"name":"x"}]}),
        json!({"id":"q","type":"choice","question":"q","options":[{"name":"a","unknown":1},{"name":"x"}]}),
        json!({"id":"q","type":"score","question":"q","levels":["l",{}]}),
        json!({"id":"q","type":"yes_no","question":1}),
        json!({"id":"q","type":"yes_no","question":"q","yes":1}),
        json!({"id":"q","type":"essay","question":"q"}),
    ] {
        out.push(json!({"input":"x","questions":[q]}));
    }
    out
}
#[test]
fn http_strict_invalids_before_admission() {
    let s = server(Opts::default());
    for b in invalids() {
        let (code, v) = post(&s, &b);
        assert_eq!(code, 400, "{b}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }
    assert_eq!(s.handle().metrics.decisions_admitted.load(Relaxed), 0);
}
#[test]
fn each_prompt_is_held_to_the_rendered_cap_not_only_the_aggregate_eight_mib() {
    use gb_server::decisions::{prepare_capped, MAX_PROMPT_BYTES};
    let (t, f) = env();
    let b = json!({"input":"x".repeat(5000),"questions":[choice("c",2)]});
    let whole = prepare(&t, &*f, b.clone(), 65536).unwrap();
    let n = whole[0].prompt_text.len();
    assert!(n > 5000);
    // the cap is what one prompt may be: the same body passes at the prompt's size and is refused one byte under it (the template's own
    // limit error, a 400, as the aggregate's always was)
    assert!(prepare_capped(&t, &*f, b.clone(), 65536, n).is_ok());
    let e = prepare_capped(&t, &*f, b.clone(), 65536, n - 1).unwrap_err();
    assert!(e.message.contains("byte limit"), "{}", e.message);
    // an input over the cap is refused before anything is rendered
    let e = prepare_capped(&t, &*f, b, 65536, 4999).unwrap_err();
    assert_eq!((e.status, e.message.as_str()), (413, "decision input exceeds the rendered prompt byte limit"));
    // and a cap above the aggregate is the aggregate
    let q = json!({"input":"hello","questions":[choice("c",2)]});
    assert_eq!(
        prepare_capped(&t, &*f, q.clone(), 65536, usize::MAX).unwrap()[0].prompt,
        prepare_capped(&t, &*f, q, 65536, MAX_PROMPT_BYTES).unwrap()[0].prompt
    );
}
#[test]
fn prompt_closed_and_context_boundary_prompt_only() {
    let (t, f) = env();
    let p = prep(body());
    for q in &p {
        assert!(q.prompt_text.contains("</think>"));
        assert!(!q.prompt_text.trim_end().ends_with("<think>"));
    }
    let b = json!({"input":"x","questions":[choice("c",2)]});
    let q = prepare(&t, &*f, b.clone(), 4096).unwrap();
    let n = q[0].prompt.len();
    assert!(prepare(&t, &*f, b.clone(), n).is_ok());
    let e = prepare(&t, &*f, b.clone(), n - 1).unwrap_err();
    assert_eq!(e.code, Some("context_length_exceeded"));
    assert_eq!(e.param.as_deref(), Some("input"));
    let s = server(Opts { max_model_len: n, ..Default::default() });
    assert_eq!(post(&s, &b).0, 200);
    let mut over = b;
    over["input"] = json!("xx");
    let (code, v) = post(&s, &over);
    assert_eq!(code, 400);
    assert_eq!(v["error"]["code"], "context_length_exceeded");
}
#[test]
fn contextual_token_boundary_rejects_merge_and_multitoken_and_duplicate() {
    let (t, _) = env();
    let ids = t.encode("y").unwrap();
    assert_eq!(t.encode("e").unwrap().len(), 1);
    assert_eq!(t.encode("ye").unwrap().len(), 1);
    assert!(check_labels(&t, "y", &ids, &["e".into()]).is_err(), "standalone one-token e retokenizes the preceding y");
    assert!(check_labels(&t, "x", &t.encode("x").unwrap(), &["!!".into()]).is_err());
    assert!(check_labels(&t, "x", &t.encode("x").unwrap(), &["A".into(), "A".into()]).is_err());
    let d = common::qwen_fixture_dir();
    let old = Tok::from_dir(&d).unwrap();
    let f = FormatRegistry::builtin().open(&d, None).unwrap();
    let b = json!({"input":"x","questions":[{"id":"y","type":"yes_no","question":"q"}]});
    assert!(prepare(&old, &*f, b, 4096).unwrap_err().message.contains("not one distinct token"));
}
#[test]
fn low_labels_outside_top64_actual_response_and_tie() {
    let mut b = body();
    b["questions"] = json!([choice("topic", 26)]);
    let p = prep(b.clone());
    let mut row = vec![10.0; 4096];
    for &id in &p[0].label_ids {
        row[id as usize] = -80.0;
    }
    let (top, _) = gb_sampling::top_candidates(&row, 64);
    assert!(p[0].label_ids.iter().all(|id| !top.contains(id)));
    let r = build_response(&b, &p, &[ScoredQuestion { logits_row: row.clone(), cached_tokens: 0 }]).unwrap();
    let a = &r["answers"]["topic"];
    assert_eq!(a["choice"], "option0");
    let expected = 26.0 * (-80.0_f64 - logsumexp_full_row(&row)).exp();
    let got = a["label_mass"].as_f64().unwrap();
    assert!((got / expected - 1.0).abs() < 1e-12);
    assert!(got > 0.0);
    assert_shape(&r);
    row.fill(1e30);
    let r = build_response(&b, &p, &[ScoredQuestion { logits_row: row, cached_tokens: 0 }]).unwrap();
    assert!((r["answers"]["topic"]["label_mass"].as_f64().unwrap() - 26.0 / 4096.0).abs() < 1e-15);
}
#[test]
fn http_low_labels_outside_top64_use_full_vocabulary_mass() {
    let s = server(Opts::default());
    let mut matched = false;
    for seed in 0..128 {
        let b = json!({"input":format!("Low label request {seed}"),"questions":[choice("low",2)]});
        let p = prep(b.clone());
        let row = gb_engine::mock::mock_logits(&p[0].prompt, 4096);
        let (top, _) = gb_sampling::top_candidates(&row, 64);
        if p[0].label_ids.iter().any(|id| top.contains(id)) {
            continue;
        }
        let expected: f64 = p[0].label_ids.iter().map(|&id| (row[id as usize] as f64 - logsumexp_full_row(&row)).exp()).sum();
        let (code, r) = post(&s, &b);
        assert_eq!(code, 200, "{r}");
        assert!((r["answers"]["low"]["label_mass"].as_f64().unwrap() / expected - 1.0).abs() < 1e-12);
        assert_eq!(r["usage"]["completion_tokens"], 0);
        matched = true;
        break;
    }
    assert!(matched, "the low-label HTTP fixture must actually rank every label below top64");
}
#[test]
fn identical_questions_distinct_ids_are_independent_valid_answers() {
    let s = server(Opts::default());
    let b = json!({"input":"same passage","questions":[choice("first",2),choice("second",2)]});
    let (code, r) = post(&s, &b);
    assert_eq!(code, 200, "{r}");
    assert_eq!(r["answers"]["first"], r["answers"]["second"]);
    let p = prep(b);
    assert_eq!(r["usage"]["prompt_tokens"], 2 * p[0].prompt.len());
    assert_eq!(s.handle().metrics.completion_tokens.load(Relaxed), 0);
}
#[test]
fn temperature_properties_and_unconditioned_mass() {
    let mut b = body();
    let p = prep(b.clone());
    let rows: Vec<_> = p
        .iter()
        .map(|q| ScoredQuestion { logits_row: gb_engine::mock::mock_logits(&q.prompt, 4096), cached_tokens: 0 })
        .collect();
    let baseline = build_response(&b, &p, &rows).unwrap();
    for t in [f64::MIN_POSITIVE, f64::from_bits(1), 1e-12, 0.1, 1.0, 100.0, f64::MAX] {
        b["temperature"] = json!(t);
        let r = build_response(&b, &p, &rows).unwrap();
        assert_shape(&r);
        for q in &p {
            assert_eq!(r["answers"][&q.id]["label_mass"], baseline["answers"][&q.id]["label_mass"]);
        }
    }
    for n in 2..=26 {
        for seed in 0..32 {
            let row = gb_engine::mock::mock_logits(&[seed, 3], n);
            let p = softmax_labels(&row, 0.7);
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-14);
            let shift: Vec<_> = row.iter().map(|v| v + 1.0).collect();
            let shifted = softmax_labels(&shift, 0.7);
            for (a, b) in p.iter().zip(shifted) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }
}
#[test]
fn bad_rows_never_fabricate_answers() {
    let p = prep(body());
    for mut rows in [
        vec![],
        vec![ScoredQuestion { logits_row: vec![], cached_tokens: 0 }; 3],
        vec![ScoredQuestion { logits_row: vec![0.0; 1], cached_tokens: 0 }; 3],
    ] {
        assert_eq!(build_response(&body(), &p, &rows).unwrap_err().status, 500);
        rows.clear();
    }
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut rows: Vec<_> = p.iter().map(|_| ScoredQuestion { logits_row: vec![0.0; 4096], cached_tokens: 0 }).collect();
        rows[0].logits_row[4000] = bad;
        assert_eq!(build_response(&body(), &p, &rows).unwrap_err().status, 500);
    }
}
#[test]
fn cached_http_equal_to_uncached_and_hits_nonzero() {
    let mut o = Opts::default();
    o.mock.prefix_cache = 8;
    o.mock.slice = 37;
    let mut off = o.clone();
    off.prefix_cache_enabled = false;
    let cached = server(o);
    let cold = server(off);
    let mut b = body();
    b["input"] = json!("Shared contextual prefix with real words. ".repeat(30));
    let (c, first) = post(&cached, &b);
    assert_eq!(c, 200, "{first}");
    let (c, hit) = post(&cached, &b);
    assert_eq!(c, 200, "{hit}");
    let (c, uncached) = post(&cold, &b);
    assert_eq!(c, 200, "{uncached}");
    assert_eq!(first["answers"], hit["answers"]);
    assert_eq!(hit["answers"], uncached["answers"]);
    assert!(hit["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap() > 0);
}
#[test]
fn deadline_504_reclaims_and_no_partial_answers() {
    let mut o = Opts { slots: 1, decision_timeout: Duration::from_millis(20), ..Default::default() };
    o.mock.prefill_ms = 10;
    o.mock.slice = 1;
    let s = server(o);
    let (code, v) = post(&s, &body());
    assert_eq!(code, 504, "{v}");
    assert_eq!(v["error"]["code"], "request_timeout");
    assert!(v.get("answers").is_none());
    s.wait(Duration::from_secs(2), "score reclamation", |s| s.in_flight() == 0);
    assert_eq!(s.handle().metrics.completion_tokens.load(Relaxed), 0);
}
#[test]
fn rendered_output_is_bounded_during_construction() {
    use gb_server::formats::qwen::QwenFormat;
    let (t, _) = env();
    let f = QwenFormat::new("{% for i in range(1000000) %}0123456789{% endfor %}").unwrap();
    let e = prepare(&t, &f, body(), 65536).unwrap_err();
    assert!(e.message.contains("byte limit"));
}
#[test]
fn shared_fifo_queue_full_retry_and_queued_cancel() {
    let mut o = Opts { slots: 1, queue_depth: 1, ..Default::default() };
    o.mock.slice = 1;
    o.mock.prefill_ms = 5;
    let s = server(o);
    let first = common::Conn::post(s.port, "/v1/decisions", &body());
    s.wait(Duration::from_secs(2), "first admitted", |s| s.handle().metrics.decisions_admitted.load(Relaxed) == 1);
    let second = common::Conn::post(s.port, "/v1/decisions", &body());
    s.wait(Duration::from_secs(2), "second queued", |s| s.handle().metrics.queued.load(Relaxed) == 1);
    let refused = common::http_timeout(s.port, "POST", "/v1/decisions", Some(&body()), Duration::from_secs(2));
    assert_eq!(refused.status, 429);
    assert!(refused.header("Retry-After").is_some());
    drop(second);
    s.wait(Duration::from_secs(2), "queued score cancelled", |s| {
        s.handle().metrics.queued.load(Relaxed) == 0 && s.handle().metrics.decisions_cancelled.load(Relaxed) > 0
    });
    drop(first);
    s.wait(Duration::from_secs(2), "all score slots returned", |s| s.in_flight() == 0);
}
#[test]
fn queued_score_timeout_is_503() {
    let mut o = Opts { slots: 1, queue_timeout: Duration::from_millis(20), ..Default::default() };
    o.mock.slice = 1;
    o.mock.prefill_ms = 5;
    let s = server(o);
    let first = common::Conn::post(s.port, "/v1/decisions", &body());
    s.wait(Duration::from_secs(2), "first admitted", |s| s.handle().metrics.decisions_admitted.load(Relaxed) == 1);
    let (code, r) = post(&s, &body());
    assert_eq!(code, 503, "{r}");
    assert!(r.get("answers").is_none());
    assert_eq!(s.handle().metrics.queue_timeouts.load(Relaxed), 1);
    drop(first);
}
#[test]
fn mixed_chat_decodes_while_long_decision_prefills() {
    let mut o = Opts::default();
    o.mock.slice = 16;
    o.mock.prefill_ms = 2;
    let s = server(o);
    let mut b = body();
    b["input"] = json!("long context ".repeat(200));
    let score = common::Conn::post(s.port, "/v1/decisions", &b);
    s.wait(Duration::from_secs(2), "score admitted", |s| s.handle().metrics.decisions_admitted.load(Relaxed) > 0);
    let chat = common::chat_body(json!({"stream":true,"max_tokens":12}));
    let reply = common::http_timeout(s.port, "POST", "/v1/chat/completions", Some(&chat), Duration::from_secs(3));
    assert_eq!(reply.status, 200);
    assert!(reply.body.contains("[DONE]"));
    assert_eq!(s.handle().metrics.decisions_completed.load(Relaxed), 0, "chat finishes while decision still prefilling");
    drop(score);
}
#[test]
fn tcp_disconnect_during_prefill_cancels_before_deadline() {
    let mut o = Opts { slots: 1, decision_timeout: Duration::from_secs(10), ..Default::default() };
    o.mock.prefill_ms = 10;
    o.mock.slice = 1;
    let s = server(o);
    let mut b = body();
    b["input"] = json!("long context ".repeat(100));
    let conn = common::Conn::post(s.port, "/v1/decisions", &b);
    s.wait(Duration::from_secs(2), "score admitted", |s| s.handle().metrics.decisions_admitted.load(Relaxed) > 0);
    conn.sock.shutdown(std::net::Shutdown::Both).unwrap();
    drop(conn);
    s.wait(Duration::from_secs(2), "disconnected score reclaimed before deadline", |s| {
        s.in_flight() == 0 && s.handle().metrics.decisions_cancelled.load(Relaxed) > 0
    });
    assert_eq!(s.handle().metrics.decisions_completed.load(Relaxed), 0);
}
#[test]
fn non_qwen_family_refused_before_render() {
    struct F;
    impl ChatFormat for F {
        fn id(&self) -> &'static str {
            "glm5_next"
        }
        fn render(
            &self,
            _: &Value,
            _: &serde_json::Map<String, Value>,
        ) -> Result<gb_server::chat_format::RenderedChat, gb_server::template::TemplateError> {
            panic!("unsupported family must not render")
        }
        fn starts_in_reasoning(&self, _: &str) -> bool {
            true
        }
        fn tool_syntax(&self) -> gb_server::tools::ToolSyntax {
            gb_server::tools::FUNCTION_BLOCK_TOOL_SYNTAX
        }
    }
    let (t, _) = env();
    let e = prepare(&t, &F, body(), 4096).unwrap_err();
    assert!(e.message.contains("not supported"));
}
