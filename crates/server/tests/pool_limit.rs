//! What a client sees when the KV pool the memory plan allocated is smaller than the configured context window (serve hardening, findings 1
//! and 2): a prompt the pool cannot hold is refused at once as `context_length_exceeded` (it used to be admitted, prefilled for minutes and
//! failed `500 prefill of slot 0: OutOfPages`); the generation budget is clipped to the room the pool leaves; requests that fit alone but not
//! together wait for each other; and `/v1/models` and the metrics report the limit a client can use, with the configured window beside it.
//!
//! The in-process server's admission is built with the pool declared (`Opts::pool`), as `MemoryPlan::admission` builds it for the real
//! engines; `process.rs` checks the same through the real binary. The mutation each test was checked against is named in its comment.

mod common;

use common::{completion_body, http, http_timeout, start, Opts, Server};
use gb_engine::mock::MockOptions;
use serde_json::Value;
use std::time::{Duration, Instant};

const SEC: Duration = Duration::from_secs(1);
/// The configured window, and the pool: a prompt of 1,300 tokens is inside the window and past the pool.
const WINDOW: usize = 4096;
const POOL: usize = 1000;

fn server(pool: bool, step_ms: u64) -> Server {
    start(Opts {
        max_model_len: WINDOW,
        kv_tokens: POOL,
        pool,
        slots: 4,
        mock: MockOptions { vocab: 4096, step_ms, ..Default::default() },
        ..Default::default()
    })
}

fn post(s: &Server, body: &Value) -> common::Reply {
    http_timeout(s.port, "POST", "/v1/completions", Some(body), 60 * SEC)
}

fn tokens(n: usize, seed: u32) -> Vec<u32> {
    (0..n).map(|i| 5 + ((i as u32 * 7 + seed) % 400)).collect()
}

#[test]
fn a_prompt_past_the_pool_is_refused_at_once_naming_the_planned_limit() {
    // mutation: drop `.with_kv_tokens(..)` from `MemoryPlan::admission` / the server's limit from `start` and this prompt is admitted (the
    // control below shows what that looked like)
    let s = server(true, 0);
    let t0 = Instant::now();
    for stream in [false, true] {
        let r = post(&s, &completion_body(&tokens(POOL + 300, 1), 1, stream));
        assert_eq!(r.status, 400, "{}", r.body);
        let e = r.json()["error"].clone();
        assert_eq!(
            (e["code"].as_str(), e["type"].as_str(), e["param"].as_str()),
            (Some("context_length_exceeded"), Some("invalid_request_error"), Some("prompt")),
            "{e}"
        );
        let m = e["message"].as_str().unwrap();
        // the planned limit first (the sentence clients parse), then where it comes from and the window as configured
        assert!(m.starts_with(&format!("This model's maximum context length is {POOL} tokens")), "{m}");
        assert!(m.contains(&format!("{} prompt tokens", POOL + 300)), "{m}");
        assert!(m.contains("KV cache") && m.contains(&format!("configured context window is {WINDOW}")), "{m}");
    }
    assert!(t0.elapsed() < 2 * SEC, "a refusal must be immediate, it took {:?}", t0.elapsed());
    // the edge is exact: the pool's size minus one leaves room for one new token, the pool's size leaves none
    assert_eq!(post(&s, &completion_body(&tokens(POOL - 1, 2), 1, false)).status, 200);
    assert_eq!(post(&s, &completion_body(&tokens(POOL, 2), 1, false)).status, 400);
    // the chat endpoint says `messages`
    let chat = serde_json::json!({"model": "mock", "messages": [{"role": "user", "content": "word ".repeat(1500)}]});
    let r = http_timeout(s.port, "POST", "/v1/chat/completions", Some(&chat), 30 * SEC);
    assert_eq!(
        (r.status, r.json()["error"]["code"].clone(), r.json()["error"]["param"].clone()),
        (400, "context_length_exceeded".into(), "messages".into()),
        "{}",
        r.body
    );
    // nothing reached the engine
    assert_eq!(s.metric(|m| &m.requests), 1, "only the one that fit was admitted");

    // the control: without the pool declared the same prompt is accepted (and on a real engine would have run the pool dry)
    let old = server(false, 0);
    assert_eq!(post(&old, &completion_body(&tokens(POOL + 300, 1), 1, false)).status, 200);
}

#[test]
fn the_generation_budget_is_clipped_to_the_room_the_pool_leaves() {
    // a prompt of 900 tokens asks for 500 new ones: the pool holds 1,000, so the reply is 100 tokens and ends as `length`, as it would at the window
    // mutation: compute `room` from the window in `start` and the request is admitted for 1,400 tokens and refused by the engine instead
    let s = server(true, 0);
    let r = post(&s, &completion_body(&tokens(900, 3), 500, false));
    assert_eq!(r.status, 200, "{}", r.body);
    let j = r.json();
    assert_eq!(j["usage"]["completion_tokens"], 100, "{j}");
    assert_eq!(j["choices"][0]["finish_reason"], "length");
}

#[test]
fn the_models_list_and_the_metrics_report_the_limit_and_keep_the_configured_window() {
    // mutation: report `s.max_model_len` again in `models` / `metrics`
    let s = server(true, 0);
    let (status, body) = http(s.port, "GET", "/v1/models", None);
    assert_eq!(status, 200, "{body}");
    let m = serde_json::from_str::<Value>(&body).unwrap()["data"][0].clone();
    assert_eq!((m["max_model_len"].clone(), m["configured_max_model_len"].clone()), (POOL.into(), WINDOW.into()), "{m}");
    let (_, text) = http(s.port, "GET", "/metrics", None);
    let line = |name: &str| {
        text.lines().find(|l| l.starts_with(&format!("{name} "))).unwrap_or_else(|| panic!("no {name} in\n{text}")).to_string()
    };
    assert_eq!(line("rexl3_max_model_len"), format!("rexl3_max_model_len {POOL}"));
    assert_eq!(line("rexl3_configured_max_model_len"), format!("rexl3_configured_max_model_len {WINDOW}"));
    let info = text.lines().find(|l| l.starts_with("rexl3_cache_config_info{")).unwrap();
    assert!(
        info.contains(&format!("kv_cache_size_tokens=\"{POOL}\""))
            && info.contains(&format!("configured_max_model_len=\"{WINDOW}\"")),
        "{info}"
    );

    // a server whose pool is not below the window reports the window, as it always did, and a server that has declared no pool does too
    for (pool, kv) in [(true, 1 << 20), (false, POOL)] {
        let s = start(Opts {
            max_model_len: WINDOW,
            kv_tokens: kv,
            pool,
            mock: MockOptions { vocab: 4096, ..Default::default() },
            ..Default::default()
        });
        let (_, body) = http(s.port, "GET", "/v1/models", None);
        let m = serde_json::from_str::<Value>(&body).unwrap()["data"][0].clone();
        assert_eq!((m["max_model_len"].clone(), m["configured_max_model_len"].clone()), (WINDOW.into(), WINDOW.into()), "{m}");
        let (_, text) = http(s.port, "GET", "/metrics", None);
        assert!(text.contains(&format!("rexl3_max_model_len {WINDOW}\n")), "{text}");
    }
}

#[test]
fn two_requests_that_fit_alone_but_not_together_wait_for_each_other_and_both_finish() {
    // each is 700 tokens (600 prompt + 100 new) of a pool of 1,000: together 1,400. The idle slots' state would let both in at once.
    // mutation: delete the running sum in `Admission::can_admit` (or `with_kv_tokens`) and the second is admitted beside the first
    let s = server(true, 15);
    let (a, b) = (completion_body(&tokens(600, 4), 100, false), completion_body(&tokens(600, 5), 100, false));
    let solo = |body: &Value| {
        let one = server(true, 0);
        post(&one, body).json()
    };
    let (want_a, want_b) = (solo(&a), solo(&b));
    let port = s.port;
    let first = std::thread::spawn(move || http_timeout(port, "POST", "/v1/completions", Some(&a), 120 * SEC));
    s.wait(10 * SEC, "the first request to be running", |s| s.metric(|m| &m.active) + s.metric(|m| &m.prefilling) == 1);
    let second = std::thread::spawn(move || http_timeout(port, "POST", "/v1/completions", Some(&b), 120 * SEC));
    s.wait(10 * SEC, "the second request to be queued behind the first", |s| s.metric(|m| &m.queued) == 1);
    assert_eq!(s.metric(|m| &m.active) + s.metric(|m| &m.prefilling), 1, "it is not in the engine: the pool holds one of them");
    let (first, second) = (first.join().unwrap(), second.join().unwrap());
    assert_eq!((first.status, second.status), (200, 200), "{} / {}", first.body, second.body);
    // each reply is what it is alone: the wait changed nothing but the time
    for (got, want) in [(first.json(), want_a), (second.json(), want_b)] {
        assert_eq!(got["choices"][0]["text"], want["choices"][0]["text"]);
        assert_eq!(got["usage"]["completion_tokens"], 100);
    }
    assert_eq!(s.metric(|m| &m.failed_sequences), 0, "no request was failed for the other");
    assert_eq!(s.metric(|m| &m.rejected), 0);
    s.wait(5 * SEC, "the engine to be empty", |s| s.in_flight() == 0);
}
