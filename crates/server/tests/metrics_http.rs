//! Real loopback HTTP acceptance of observability, CPU mock only.
mod common;
use common::{completion_body, http_timeout, raw, start, Conn, Opts};
use gb_engine::mock::MockOptions;
use std::time::Duration;

fn scrape(port: u16) -> String {
    let r = http_timeout(port, "GET", "/metrics", None, Duration::from_secs(5));
    assert_eq!(r.status, 200);
    r.body
}
fn value(text: &str, key: &str) -> f64 {
    text.lines()
        .find_map(|l| l.strip_prefix(&format!("{key} ")).map(|s| s.parse().unwrap()))
        .unwrap_or_else(|| panic!("absent series: {key}"))
}

#[test]
fn real_token_batches_latencies_and_capacity() {
    let s = start(Opts {
        pool: true,
        slots: 2,
        kv_tokens: 4096,
        mock: MockOptions { vocab: 4096, prefill_ms: 20, step_ms: 3, ..Default::default() },
        ..Default::default()
    });
    let r = http_timeout(s.port, "POST", "/v1/completions", Some(&completion_body(&[5; 10], 4, true)), Duration::from_secs(5));
    assert_eq!(r.status, 200, "{}", r.body);
    assert!(r.body.contains("[DONE]"));
    // the loop routes [DONE] to the stream before it snapshots the gauges, so a scrape right after the reply can still see the
    // finished request's reservation
    s.wait(Duration::from_secs(5), "the finished request's KV reservation released", |s| {
        s.handle().metrics.observability.capacity.lock().unwrap().is_some_and(|(used, _)| used == 0)
    });
    let text = scrape(s.port);
    for name in [
        "rexl3_request_ttft_seconds",
        "rexl3_engine_submit_ttft_seconds",
        "rexl3_token_batch_itl_seconds",
        "rexl3_request_e2e_seconds",
        "rexl3_request_preprocessing_seconds",
        "rexl3_request_queue_seconds",
        "rexl3_engine_prefill_slice_seconds",
    ] {
        assert!(text.contains(&format!("# TYPE {name} histogram")), "{name}");
        assert_eq!(value(&text, &format!("{name}_count")), if name.contains("itl") { 3.0 } else { 1.0 });
    }
    assert_eq!(value(&text, "rexl3_completion_tokens_total"), 4.0);
    assert_eq!(value(&text, "rexl3_token_batches_total"), 4.0);
    assert_eq!(value(&text, "rexl3_prompt_tokens_total"), 10.0);
    assert_eq!(value(&text, "rexl3_prefill_tokens_processed_total"), 10.0);
    assert_eq!(value(&text, "rexl3_requests_completed_total"), 1.0);
    assert_eq!(value(&text, "rexl3_sse_terminal_total{outcome=\"success\"}"), 1.0);
    assert_eq!(value(&text, "rexl3_kv_pool_tokens"), 4096.0);
    assert_eq!(value(&text, "rexl3_kv_pool_tokens_used"), 0.0);
    assert!(value(&text, "rexl3_request_ttft_seconds_sum") >= 0.020);
}

#[test]
fn every_http_status_and_sanitized_ids_without_content_labels() {
    let s = start(Opts::default());
    let r = raw(s.port, "GET", "/unknown-secret?key=private", &[("x-request-id", "safe-ID_1")], b"", Duration::from_secs(5));
    assert_eq!(r.status, 404);
    assert_eq!(r.header("x-request-id"), Some("safe-ID_1"));
    let r = raw(s.port, "GET", "/metrics", &[("x-request-id", "unsafe content")], b"", Duration::from_secs(5));
    let id = r.header("x-request-id").expect("request ID");
    assert!(id.starts_with("req-") && id.len() == 20, "{id}");
    assert_eq!(http_timeout(s.port, "GET", "/v1/completions", None, Duration::from_secs(5)).status, 405);
    assert_eq!(
        raw(s.port, "POST", "/v1/completions", &[("content-type", "application/json")], b"{", Duration::from_secs(5)).status,
        400
    );
    assert_eq!(raw(s.port, "POST", "/v1/completions", &[], b"{}", Duration::from_secs(5)).status, 415);
    assert_eq!(
        raw(
            s.port,
            "POST",
            "/v1/completions",
            &[("content-type", "application/json")],
            &vec![b' '; 65 << 20],
            Duration::from_secs(10)
        )
        .status,
        413
    );
    let text = scrape(s.port);
    for code in [400, 404, 405, 413, 415] {
        assert_eq!(value(&text, &format!("rexl3_http_responses_total{{code=\"{code}\"}}")), 1.0);
        let route = if code == 404 { "other" } else { "/v1/completions" };
        assert_eq!(value(&text, &format!("rexl3_http_route_responses_total{{route=\"{route}\",code=\"{code}\"}}")), 1.0);
    }
    assert!(value(&text, "rexl3_http_requests_total") >= 7.0);
    for secret in ["safe-ID", "unknown-secret", "private", "unsafe content"] {
        assert!(!text.contains(secret));
    }
}

#[test]
fn sse_engine_failure_is_not_http_failure_and_disconnect_not_server_error() {
    let s = start(Opts {
        mock: MockOptions {
            vocab: 4096,
            faults: gb_engine::mock::Faults { poison_first_token: Some(77), ..Default::default() },
            step_ms: 10,
            ..Default::default()
        },
        ..Default::default()
    });
    let r = http_timeout(s.port, "POST", "/v1/completions", Some(&completion_body(&[77, 4], 4, true)), Duration::from_secs(5));
    assert_eq!(r.status, 200);
    assert!(r.body.contains("error"));
    let mut c = Conn::post(s.port, "/v1/completions", &completion_body(&[5; 100], 1000, true));
    assert!(c.read_until(Duration::from_secs(5), |t| t.contains("data:")));
    drop(c);
    s.wait(Duration::from_secs(5), "cancelled", |s| s.metric(|m| &m.cancelled) > 0);
    std::thread::sleep(Duration::from_millis(30));
    let text = scrape(s.port);
    assert_eq!(value(&text, "rexl3_sse_terminal_total{outcome=\"error\"}"), 1.0);
    assert_eq!(value(&text, "rexl3_sse_terminal_total{outcome=\"cancelled\"}"), 1.0);
    assert_eq!(value(&text, "rexl3_http_responses_total{code=\"500\"}"), 0.0);
}

#[test]
fn actual_prefill_counts_cached_and_cancelled_work_not_admitted_prompt() {
    let s = start(Opts {
        pool: true,
        mock: MockOptions { vocab: 4096, prefix_cache: 6, ..Default::default() },
        ..Default::default()
    });
    let prompt: Vec<u32> = (0..700u32).map(|i| 1 + (i.wrapping_mul(2654435761).wrapping_add(1) >> 7) % 4000).collect();
    let body = completion_body(&prompt, 3, false);
    for _ in 0..2 {
        assert_eq!(http_timeout(s.port, "POST", "/v1/completions", Some(&body), Duration::from_secs(5)).status, 200);
    }
    let text = scrape(s.port);
    assert_eq!(value(&text, "rexl3_prompt_tokens_total"), 1400.0);
    assert_eq!(value(&text, "rexl3_prefill_tokens_processed_total"), 708.0);
    assert_eq!(value(&text, "rexl3_prefill_tokens_cached_total"), 692.0);
    assert_eq!(value(&text, "rexl3_slots_total"), 8.0);
    let s = start(Opts {
        pool: true,
        mock: MockOptions { vocab: 4096, prefill_ms: 15, slice: 10, ..Default::default() },
        ..Default::default()
    });
    let c = Conn::post(s.port, "/v1/completions", &completion_body(&[5; 1000], 3, true));
    s.wait(Duration::from_secs(5), "actual slice progress", |s| {
        s.handle().metrics.observability.prefill.lock().unwrap().is_some_and(|p| p.processed >= 10)
    });
    let busy = scrape(s.port);
    assert!(value(&busy, "rexl3_kv_pool_tokens_used") > 0.0);
    assert!(value(&busy, "rexl3_kv_pool_tokens_used") <= value(&busy, "rexl3_kv_pool_tokens"));
    drop(c);
    s.wait(Duration::from_secs(5), "cancel", |s| s.metric(|m| &m.cancelled) > 0);
    s.wait(Duration::from_secs(5), "the cancelled request's KV reservation released", |s| {
        s.handle().metrics.observability.capacity.lock().unwrap().is_some_and(|(used, _)| used == 0)
    });
    let text = scrape(s.port);
    assert_eq!(value(&text, "rexl3_prompt_tokens_total"), 1000.0);
    let processed = value(&text, "rexl3_prefill_tokens_processed_total");
    assert!((10.0..1000.0).contains(&processed));
    assert_eq!(value(&text, "rexl3_requests_completed_total"), 0.0);
    assert_eq!(value(&text, "rexl3_kv_pool_tokens_used"), 0.0);
}

#[test]
fn undeclared_kv_capacity_is_absent_not_fake_zero() {
    let s = start(Opts { pool: false, ..Default::default() });
    let text = scrape(s.port);
    assert!(!text.lines().any(|l| l.starts_with("rexl3_kv_pool_tokens ") || l.starts_with("rexl3_kv_pool_tokens_used ")));
}

#[test]
fn dashboard_series_contract_checks_real_http() {
    let s = start(Opts { pool: true, ..Default::default() });
    let r = std::process::Command::new("python3")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tools/dashboard_check.py"))
        .arg(format!("http://127.0.0.1:{}", s.port))
        .output()
        .unwrap();
    assert!(r.status.success(), "{}{}", String::from_utf8_lossy(&r.stdout), String::from_utf8_lossy(&r.stderr));
}

/// The graph cache's series appear exactly when the engine reports a graph cache (a backend with graphs), with its counters; a backend
/// without one serves no `rexl3_graph_cache_*` series at all, so its `/metrics` text is what it was before the counters were exported.
#[test]
fn graph_cache_counters_are_exported_only_for_a_backend_with_graphs() {
    for graphs in [false, true] {
        let s = start(Opts { mock: MockOptions { vocab: 4096, step_ms: 1, graphs, ..Default::default() }, ..Default::default() });
        let r =
            http_timeout(s.port, "POST", "/v1/completions", Some(&completion_body(&[5; 10], 4, false)), Duration::from_secs(5));
        assert_eq!(r.status, 200, "{}", r.body);
        if graphs {
            // the graph counters are part of the loop's snapshot, taken after it routes the reply's last event (as the KV gauge above)
            s.wait(Duration::from_secs(5), "the graph counters of the finished request", |s| {
                s.handle().metrics.graphs.lock().unwrap().is_some_and(|g| g.replays == 3 && g.eager_windows == 1)
            });
        }
        let text = scrape(s.port);
        if !graphs {
            assert!(!text.contains("graph_cache"), "a backend without graphs exports no graph series:\n{text}");
            continue;
        }
        // 4 tokens: the first from the prefill, 3 decode steps (each a replay on this mock), one prefill slice (an eager window)
        assert_eq!(value(&text, "rexl3_graph_cache_replays_total"), 3.0);
        assert_eq!(value(&text, "rexl3_graph_cache_eager_windows_total"), 1.0);
        for name in [
            "rexl3_graph_cache_captures_total",
            "rexl3_graph_cache_evictions_total",
            "rexl3_graph_cache_capture_failures_total",
            "rexl3_graph_cache_capture_seconds_total",
            "rexl3_draft_graph_cache_replays_total",
            "rexl3_draft_graph_cache_captures_total",
            "rexl3_draft_graph_cache_eager_total",
        ] {
            assert_eq!(value(&text, name), 0.0, "{name}");
        }
    }
}
