//! Request lifecycle through the HTTP server on the CPU mock backend (release plan, Phase 1): the FIFO queue and its limits, streaming
//! headers and keep-alive comments while queued, cancellation when a client goes away, per-request failure, and the engine's fault
//! handling as a client sees it. Every test here fails when the behaviour it guards is removed (the mutation each was checked against is
//! named in its comment).

mod common;

use common::{chat_body, completion_body, http, http_timeout, sse, start, Conn, Opts};
use gb_engine::mock::{Faults, MockOptions};
use serde_json::json;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

const SEC: Duration = Duration::from_secs(1);

/// Mock options that make a request take `step_ms` per decoded round.
fn slow(step_ms: u64) -> MockOptions {
    MockOptions { step_ms, vocab: 4096, ..Default::default() }
}

/// A long streaming chat request that occupies a slot until its connection is dropped.
fn hog(s: &common::Server) -> Conn {
    let admitted = s.metric(|m| &m.requests);
    let mut c = Conn::post(s.port, "/v1/chat/completions", &chat_body(json!({"stream": true, "max_tokens": 3000})));
    assert!(c.read_until(10 * SEC, |t| t.contains("data: ")), "the hog never started: {}", c.text());
    // a stream's first bytes arrive when the request is queued: wait until it really holds a slot
    s.wait(10 * SEC, "the hog to be admitted", |s| s.metric(|m| &m.requests) > admitted);
    c
}

// ------------------------------------------------------------------------------------------------ the queue

#[test]
fn a_full_queue_answers_429_with_retry_after_and_counts_the_request_once() {
    // mutation: drop the `lp.queue.len() >= cfg.queue_depth` refusal in runtime::enqueue and the third waiter is queued (200)
    let s = start(Opts { slots: 1, queue_depth: 2, mock: slow(5), ..Default::default() });
    let _running = hog(&s);
    let waiting: Vec<_> = (0..2)
        .map(|_| {
            let port = s.port;
            std::thread::spawn(move || {
                http_timeout(port, "POST", "/v1/chat/completions", Some(&chat_body(json!({"max_tokens": 4}))), 60 * SEC)
            })
        })
        .collect();
    s.wait(10 * SEC, "two requests to queue", |s| s.metric(|m| &m.queued) == 2);
    let t0 = Instant::now();
    let r = http_timeout(s.port, "POST", "/v1/chat/completions", Some(&chat_body(json!({}))), 10 * SEC);
    assert_eq!(r.status, 429, "{}", r.body);
    assert_eq!(r.header("retry-after"), Some("2"), "{:?}", r.headers);
    let e = r.json();
    assert_eq!(e["error"]["type"], "server_overloaded");
    assert_eq!(e["error"]["code"], 429);
    assert!(e["error"]["message"].as_str().unwrap().contains("queue is full"), "{e}");
    assert!(t0.elapsed() < 2 * SEC, "a refusal must be immediate, it took {:?}", t0.elapsed());
    // once per request, however long it lingered: the old counter counted every 25 ms retry of a waiting request
    assert_eq!((s.metric(|m| &m.rejected), s.metric(|m| &m.queue_full)), (1, 1));
    // the two that were queued still get served when the slot frees (the hog goes away)
    drop(_running);
    for w in waiting {
        assert_eq!(w.join().unwrap().status, 200);
    }
}

#[test]
fn a_request_that_can_never_fit_is_refused_at_once_instead_of_waiting() {
    // mutation: make `Engine::validate` skip `admission.check` (NeverFits) and the request waits in the queue until its timeout
    // one slot and 100 tokens of KV: 500 new tokens need more than the whole capacity, 20 need less
    let s = start(Opts { slots: 1, kv_tokens: 100, queue_timeout: Duration::from_secs(20), mock: slow(1), ..Default::default() });
    let t0 = Instant::now();
    let r = http_timeout(s.port, "POST", "/v1/completions", Some(&completion_body(&[5, 6, 7], 500, false)), 10 * SEC);
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(t0.elapsed() < 2 * SEC, "took {:?}: it must not wait for capacity that can never exist", t0.elapsed());
    let m = r.json()["error"]["message"].as_str().unwrap().to_string();
    assert!(m.contains("KV cache") && m.contains("lower max_tokens"), "{m}");
    assert_eq!(s.metric(|m| &m.rejected), 1);
    assert_eq!(s.metric(|m| &m.queued), 0);
    // a request that fits is served
    assert_eq!(http(s.port, "POST", "/v1/completions", Some(&completion_body(&[5, 6, 7], 20, false))).0, 200);
}

#[test]
fn queued_requests_are_admitted_in_arrival_order() {
    // mutation: `lp.queue.pop_front()` -> `pop_back()` in runtime::admit and the order reverses
    let s = start(Opts { slots: 1, mock: slow(5), ..Default::default() });
    let hog = hog(&s);
    let done: std::sync::Arc<std::sync::Mutex<Vec<usize>>> = Default::default();
    let mut threads = Vec::new();
    for i in 0..4 {
        let (port, done) = (s.port, done.clone());
        threads.push(std::thread::spawn(move || {
            let r =
                http_timeout(port, "POST", "/v1/completions", Some(&completion_body(&[10 + i as u32, 2, 3], 8, false)), 60 * SEC);
            assert_eq!(r.status, 200, "{}", r.body);
            done.lock().unwrap().push(i);
        }));
        // each arrives only after the previous one is queued, so "arrival order" is unambiguous
        s.wait(10 * SEC, "the request to queue", |s| s.metric(|m| &m.queued) == i as u64 + 1);
    }
    drop(hog); // the slot frees: the queue drains one request at a time
    for t in threads {
        t.join().unwrap();
    }
    assert_eq!(*done.lock().unwrap(), vec![0, 1, 2, 3]);
}

#[test]
fn a_request_that_waits_past_the_queue_timeout_gets_503_with_retry_after() {
    // mutation: delete `expire_queue` and the waiter is never answered
    let s = start(Opts { slots: 1, queue_timeout: Duration::from_secs(1), mock: slow(5), ..Default::default() });
    let _running = hog(&s);
    let t0 = Instant::now();
    let r = http_timeout(s.port, "POST", "/v1/chat/completions", Some(&chat_body(json!({"max_tokens": 4}))), 20 * SEC);
    assert_eq!(r.status, 503, "{}", r.body);
    assert!(t0.elapsed() >= Duration::from_millis(900) && t0.elapsed() < 5 * SEC, "{:?}", t0.elapsed());
    assert_eq!(r.header("retry-after"), Some("5"));
    let e = r.json();
    assert_eq!(e["error"]["type"], "server_overloaded");
    assert!(e["error"]["message"].as_str().unwrap().contains("waited 1s"), "{e}");
    assert_eq!((s.metric(|m| &m.queue_timeouts), s.metric(|m| &m.rejected)), (1, 1));
    s.wait(2 * SEC, "the queue gauge to drop", |s| s.metric(|m| &m.queued) == 0);
}

#[test]
fn a_streaming_request_gets_its_headers_and_keepalive_comments_while_it_is_queued() {
    // mutation: send the response only after admission (the old order) and no byte arrives until the hog is gone
    let s = start(Opts { slots: 1, keepalive: Duration::from_millis(100), mock: slow(5), ..Default::default() });
    let _running = hog(&s);
    let mut c = Conn::post(s.port, "/v1/chat/completions", &chat_body(json!({"stream": true, "max_tokens": 4})));
    assert!(
        c.read_until(5 * SEC, |t| t.starts_with("HTTP/1.1 200") && t.to_lowercase().contains("text/event-stream")),
        "no headers: {}",
        c.text()
    );
    assert!(
        c.read_until(5 * SEC, |t| t.matches(": keep-alive").count() >= 2),
        "no keep-alive comments while queued: {}",
        c.text()
    );
    assert_eq!(s.metric(|m| &m.queued), 1, "it is still queued");
    assert!(!c.text().contains("finish_reason\":\"length"), "it cannot have finished");
    // when the slot frees it runs to completion on the same connection
    drop(_running);
    let r = c.finish(30 * SEC);
    let events = sse(&r.body);
    let last = events.iter().rev().find(|e| e["choices"].as_array().is_some_and(|a| !a.is_empty())).unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "length");
}

// ------------------------------------------------------------------------------------------------ cancellation

#[test]
fn a_client_that_disconnects_while_queued_leaves_the_queue_and_is_never_run() {
    // mutation: remove the `CancelGuard` (and the closed-sender checks) and the abandoned request stays queued until a slot frees
    for stream in [true, false] {
        let s = start(Opts { slots: 1, mock: slow(5), ..Default::default() });
        let running = hog(&s);
        let c = Conn::post(s.port, "/v1/chat/completions", &chat_body(json!({"stream": stream, "max_tokens": 4})));
        s.wait(10 * SEC, "the request to queue", |s| s.metric(|m| &m.queued) == 1);
        drop(c);
        s.wait(2 * SEC, &format!("a disconnected queued request (stream={stream}) to leave the queue"), |s| {
            s.metric(|m| &m.queued) == 0
        });
        assert_eq!(s.metric(|m| &m.cancelled), 1);
        drop(running);
        s.wait(10 * SEC, "everything to be given back", |s| s.in_flight() == 0);
        assert_eq!(s.metric(|m| &m.requests), 1, "only the hog was ever admitted (stream={stream})");
    }
}

#[test]
fn a_client_that_disconnects_while_prefilling_gives_its_slot_back_within_a_round() {
    // mutation: remove the cancel path and the prefill (50 slices of 100 ms) runs on for 5 seconds
    let mock = MockOptions { prefill_ms: 100, slice: 4, vocab: 4096, ..Default::default() };
    let s = start(Opts { slots: 1, mock, ..Default::default() });
    let long: Vec<u32> = (0..200).map(|i| 300 + i % 50).collect();
    let c = Conn::post(s.port, "/v1/completions", &completion_body(&long, 4, true));
    s.wait(5 * SEC, "the prompt to be prefilling", |s| s.metric(|m| &m.prefilling) == 1);
    let t0 = Instant::now();
    drop(c);
    s.wait(2 * SEC, "the prefilling sequence to be cancelled", |s| s.in_flight() == 0);
    assert!(t0.elapsed() < Duration::from_millis(900), "slot and KV came back after {:?}, a round is about 100 ms", t0.elapsed());
    assert_eq!(s.metric(|m| &m.cancelled), 1);
    // the slot really is free: the next request is admitted at once and served
    let r = http_timeout(s.port, "POST", "/v1/completions", Some(&completion_body(&[7, 8, 9], 3, false)), 10 * SEC);
    assert_eq!(r.status, 200, "{}", r.body);
}

// ------------------------------------------------------------------------------------------------ failures

#[test]
fn a_poisoned_sequence_fails_alone_over_http_and_the_other_three_complete() {
    // mutation: make the engine fail the whole step on one bad row (engine/src/lib.rs decode_round) and all four get 500
    const POISON: u32 = 11;
    let faults = Faults { poison_first_token: Some(POISON), ..Default::default() };
    let s = start(Opts { mock: MockOptions { step_ms: 5, vocab: 4096, faults, ..Default::default() }, ..Default::default() });
    // the three healthy requests run long enough that the poisoned one joins them mid-flight (all four share decode rounds)
    let mut threads = Vec::new();
    for i in [0u32, 1, 3] {
        let port = s.port;
        threads.push((
            i,
            std::thread::spawn(move || {
                http_timeout(port, "POST", "/v1/completions", Some(&completion_body(&[100 + i, 5, 6, 7], 60, false)), 60 * SEC)
            }),
        ));
    }
    s.wait(10 * SEC, "the three healthy requests to be decoding", |s| s.metric(|m| &m.active) == 3);
    let port = s.port;
    threads.push((
        2,
        std::thread::spawn(move || {
            http_timeout(port, "POST", "/v1/completions", Some(&completion_body(&[POISON, 5, 6, 7], 60, false)), 60 * SEC)
        }),
    ));
    threads.sort_by_key(|t| t.0);
    let replies: Vec<_> = threads.into_iter().map(|(_, t)| t.join().unwrap()).collect();
    for (i, r) in replies.iter().enumerate() {
        if i == 2 {
            assert_eq!(r.status, 500, "{}", r.body);
            let e = r.json();
            assert_eq!(e["error"]["type"], "server_error");
            assert!(e["error"]["message"].as_str().unwrap().contains("NaN"), "{e}");
        } else {
            assert_eq!(r.status, 200, "sequence {i}: {}", r.body);
            assert_eq!(r.json()["usage"]["completion_tokens"], 60);
        }
    }
    // streaming: the poisoned reply ends with an error event, never a finish chunk
    let mut stream_body = completion_body(&[POISON, 5, 6, 7], 30, true);
    stream_body["stream_options"] = json!({"include_usage": true});
    let mut c = Conn::post(s.port, "/v1/completions", &stream_body);
    assert!(c.read_to_end_of_body(10 * SEC), "the stream never ended: {}", c.text());
    let body = common::dechunk(c.text().split_once("\r\n\r\n").unwrap().1);
    let events = sse(&body);
    assert!(events.iter().any(|e| e.get("error").is_some()), "no error event: {body}");
    // an errored stream closes after its error event: `[DONE]` is for a reply that finished
    assert!(!body.contains("[DONE]"), "an errored stream must not end with [DONE]: {body}");
    assert!(events.last().is_some_and(|e| e.get("error").is_some()), "the error event is the last event of the stream: {body}");
    assert!(!body.contains("\"finish_reason\":\"stop\"") && !body.contains("\"finish_reason\":\"length\""), "{body}");
    assert_eq!(s.metric(|m| &m.failed_sequences), 2);
    assert_eq!(http(s.port, "GET", "/health", None).0, 200, "one bad sequence is not an unhealthy server");
}

#[test]
fn consecutive_step_errors_drop_readiness_and_a_good_step_restores_it() {
    // mutation: never leave Degraded (or never enter it) and the /health assertions below fail
    let faults = Faults { retryable_from_step: Some(1), retryable_steps: 3, ..Default::default() };
    let s =
        start(Opts { unready_after: 3, mock: MockOptions { vocab: 4096, faults, ..Default::default() }, ..Default::default() });
    let ask = || http_timeout(s.port, "POST", "/v1/completions", Some(&completion_body(&[5, 6, 7], 10, false)), 10 * SEC);
    for n in 1..=3 {
        let r = ask();
        assert_eq!(r.status, 500, "decode round {n} fails: {}", r.body);
        assert!(r.json()["error"]["message"].as_str().unwrap().contains("injected recoverable"), "{}", r.body);
        let h = http(s.port, "GET", "/health", None);
        if n < 3 {
            assert_eq!(h.0, 200, "after {n} step errors the server is still ready");
        } else {
            assert_eq!(h, (503, "{\"status\":\"degraded\"}".into()), "after 3 in a row it must say so");
            assert_eq!(http(s.port, "GET", "/health/ready", None).0, 503);
        }
    }
    // the fourth round is clean: the request works and readiness comes back (a not-ready server still takes direct requests)
    let r = ask();
    assert_eq!(r.status, 200, "{}", r.body);
    s.wait(5 * SEC, "readiness to return", |s| s.state.handle.metrics.ready());
    assert_eq!(http(s.port, "GET", "/health", None), (200, "{\"status\":\"ok\"}".into()));
    assert_eq!(s.metric(|m| &m.step_errors), 3);
}

#[test]
fn too_many_consecutive_step_errors_declare_the_engine_failed() {
    // mutation: drop the `exit_after` check and on_fatal is never called
    let faults = Faults { retryable_from_step: Some(1), retryable_steps: 100, ..Default::default() };
    let s = start(Opts {
        unready_after: 2,
        exit_after: 3,
        mock: MockOptions { vocab: 4096, faults, ..Default::default() },
        ..Default::default()
    });
    for _ in 0..3 {
        let r = http_timeout(s.port, "POST", "/v1/completions", Some(&completion_body(&[5, 6, 7], 10, false)), 10 * SEC);
        assert!(r.status == 500 || r.status == 503, "{}", r.body);
    }
    s.wait(5 * SEC, "on_fatal", |s| !s.fatal.lock().unwrap().is_empty());
    let info = s.fatal.lock().unwrap()[0].clone();
    assert!(!info.panicked && info.reason.contains("3 consecutive step errors"), "{info:?}");
    assert_eq!(info.exit_code(), 70);
    assert_eq!(http(s.port, "GET", "/health", None), (503, "{\"status\":\"failed\"}".into()));
}

#[test]
fn a_fatal_error_answers_every_request_with_an_error_and_never_a_normal_finish() {
    // mutation: skip `fail_everything` in runtime::die and the replies end some other way (or never)
    let faults = Faults { fatal_at_step: Some(8), ..Default::default() };
    let s = start(Opts { mock: MockOptions { step_ms: 10, vocab: 4096, faults, ..Default::default() }, ..Default::default() });
    let mut streams: Vec<Conn> = (0..2)
        .map(|_| {
            // include_usage on: an errored stream must still END with its error event (no usage chunk, no [DONE] after it)
            Conn::post(
                s.port,
                "/v1/chat/completions",
                &chat_body(json!({"stream": true, "max_tokens": 500, "stream_options": {"include_usage": true}})),
            )
        })
        .collect();
    let plain = {
        let port = s.port;
        std::thread::spawn(move || {
            http_timeout(port, "POST", "/v1/completions", Some(&completion_body(&[5, 6, 7], 500, false)), 30 * SEC)
        })
    };
    let r = plain.join().unwrap();
    assert_eq!(r.status, 503, "{}", r.body);
    assert_eq!(r.json()["error"]["type"], "service_unavailable");
    assert!(r.json()["error"]["message"].as_str().unwrap().contains("injected fatal"), "{}", r.body);
    for c in streams.iter_mut() {
        assert!(c.read_to_end_of_body(10 * SEC), "the stream never ended: {}", c.text());
        let body = common::dechunk(c.text().split_once("\r\n\r\n").unwrap().1);
        assert!(
            sse(&body).iter().any(|e| e["error"]["message"].as_str().is_some_and(|m| m.contains("injected fatal"))),
            "{body}"
        );
        assert!(!body.contains("[DONE]"), "a stream cut by an engine failure must not end with [DONE]: {body}");
        let events = sse(&body);
        assert!(
            events.last().is_some_and(|e| e["error"]["message"].as_str().is_some_and(|m| m.contains("injected fatal"))),
            "the error event is the last event, even with include_usage: {body}"
        );
        assert!(events.iter().all(|e| e.get("usage").is_none_or(|u| u.is_null())), "no usage chunk after an error: {body}");
        assert!(
            !body.contains("\"finish_reason\":\"stop\"") && !body.contains("\"finish_reason\":\"length\""),
            "ended as a normal finish: {body}"
        );
    }
    s.wait(5 * SEC, "on_fatal", |s| !s.fatal.lock().unwrap().is_empty());
    assert_eq!(s.fatal.lock().unwrap().len(), 1);
    assert!(!s.fatal.lock().unwrap()[0].panicked);
    // not ready, and new work is refused rather than queued for an engine that is gone
    assert_eq!(http(s.port, "GET", "/health", None), (503, "{\"status\":\"failed\"}".into()));
    let r = http_timeout(s.port, "POST", "/v1/chat/completions", Some(&chat_body(json!({}))), 10 * SEC);
    assert_eq!(r.status, 503, "{}", r.body);
    let (_, met) = http(s.port, "GET", "/metrics", None);
    assert!(met.contains("rexl3_engine_state{state=\"failed\"} 1") && met.contains("rexl3_ready 0"), "{met}");
}

#[test]
fn a_panic_on_the_engine_thread_is_a_fatal_failure_not_a_dead_thread_behind_a_healthy_server() {
    // mutation: remove the `catch_unwind` in runtime::spawn_with and the replies are never answered
    let faults = Faults { panic_at_step: Some(5), ..Default::default() };
    let s = start(Opts { mock: MockOptions { step_ms: 10, vocab: 4096, faults, ..Default::default() }, ..Default::default() });
    let r = http_timeout(s.port, "POST", "/v1/completions", Some(&completion_body(&[5, 6, 7], 500, false)), 30 * SEC);
    assert_eq!(r.status, 503, "{}", r.body);
    assert!(r.json()["error"]["message"].as_str().unwrap().contains("panicked: injected panic at step 5"), "{}", r.body);
    s.wait(5 * SEC, "on_fatal", |s| !s.fatal.lock().unwrap().is_empty());
    let info = s.fatal.lock().unwrap()[0].clone();
    assert!(info.panicked && info.reason.contains("injected panic at step 5"), "{info:?}");
    assert_eq!(info.exit_code(), 101);
    assert_eq!(http(s.port, "GET", "/health", None).0, 503);
}

// mutation: rename the rexl3_queue_depth series in http::metrics and the first assertion fails
#[test]
fn engine_metrics_gauges_return_to_zero() {
    let s = start(Opts { mock: slow(2), ..Default::default() });
    for _ in 0..3 {
        assert_eq!(http(s.port, "POST", "/v1/chat/completions", Some(&chat_body(json!({})))).0, 200);
    }
    let (_, met) = http(s.port, "GET", "/metrics", None);
    for want in [
        "rexl3_queue_depth 0",
        "rexl3_prefilling_sequences 0",
        "rexl3_active_sequences 0",
        "rexl3_requests_total 3",
        "rexl3_requests_rejected_total 0",
    ] {
        assert!(met.lines().any(|l| l == want), "missing `{want}` in:\n{met}");
    }
    assert_eq!(met.lines().filter(|l| l.starts_with("rexl3_engine_state{")).count(), 5, "one series per state");
    assert_eq!(s.state.handle.metrics.requests.load(Relaxed), 3);
}
