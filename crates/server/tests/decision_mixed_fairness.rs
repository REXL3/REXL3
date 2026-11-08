//! Real HTTP mixed score/text fairness, using the explicit CPU mock backend.
mod common;
use common::*;
use gb_engine::mock::MockOptions;
use serde_json::json;
use std::time::Duration;

#[test]
fn two_long_decision_requests_do_not_hold_unrelated_text_preparation() {
    let s = start(Opts {
        dir: Some(qwen_fixture_dir()),
        slots: 4,
        max_model_len: 8192,
        kv_tokens: 32768,
        pool: true,
        request_timeout: Duration::from_secs(20),
        mock: MockOptions { prefill_ms: 10, slice: 8, ..Default::default() },
        ..Default::default()
    });
    let decision = json!({
        "input": "signal ".repeat(300),
        "questions": [{"id":"kind", "type":"choice", "question":"Which label applies?",
            "options":[{"name":"first", "description":"first label"}, {"name":"second", "description":"second label"}]}]
    });
    let a = Conn::post(s.port, "/v1/decisions", &decision);
    let b = Conn::post(s.port, "/v1/decisions", &decision);
    s.wait(Duration::from_secs(3), "both long native-score mock jobs admitted", |s| s.metric(|m| &m.decisions_admitted) == 2);
    let short =
        http_timeout(s.port, "POST", "/v1/completions", Some(&completion_body(&[1, 2, 3], 2, false)), Duration::from_secs(2));
    assert_eq!(short.status, 200, "{}", short.body);
    assert_eq!(s.metric(|m| &m.decisions_completed), 0, "text should run while long scoring remains in progress");
    for connection in [a, b] {
        let result = connection.finish(Duration::from_secs(20));
        assert_eq!(result.status, 200, "{}", result.body);
        assert_eq!(result.json()["usage"]["completion_tokens"], 0);
    }
    s.wait(Duration::from_secs(1), "all real mock score/text resources reclaimed", |s| s.in_flight() == 0);
}
