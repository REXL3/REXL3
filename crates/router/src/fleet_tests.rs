//! CPU-only fleet-behavior tests: health recovery, default affinity, in-flight accounting through streams, bounded
//! affinity, drain and runtime membership, retry-before-first-byte, admin auth and the router's own metrics. Mock
//! backends are axum servers on 127.0.0.1:0 that can be told to be unready, unavailable, slow or to break mid-stream.

use crate::admin::build_admin_router;
use crate::app::{App, Config};
use crate::backend::Backend;
use crate::handlers::build_router;
use axum::body::{Body, Bytes};
use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::Router;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpListener;

#[derive(Clone, Copy)]
enum Mode {
    /// 200 `{"backend":"<tag>"}`.
    Json,
    /// 503 `service_unavailable` (the engine is down).
    Unavailable,
    /// 503 `server_overloaded` (the backend's own admission refusal).
    Overloaded,
    /// SSE: `chunks` events, `gap_ms` between them.
    Stream { chunks: usize, gap_ms: u64 },
    /// SSE: one event, then the connection breaks.
    AbortAfterFirst,
    /// SSE: the shape of a reply cut by an engine failure: content events, the in-band error event, and then the stream CLOSES (no
    /// `data: [DONE]`: the server ends an errored stream after its error event).
    ErrorThenClose,
}

/// The bytes of the SSE stream of [`Mode::ErrorThenClose`], whole.
fn error_then_close_body(tag: &str) -> String {
    format!(
        "data: {{\"role\":\"{tag}\"}}\n\ndata: {{\"content\":\"a\"}}\n\ndata: {{\"content\":\"b\"}}\n\n\
         data: {{\"error\":{{\"message\":\"the engine failed: tensor-parallel group lost: the connection to rank 1 failed\",\"type\":\"service_unavailable\",\"code\":503}}}}\n\n"
    )
}

#[derive(Clone)]
struct Mock {
    url: String,
    tag: &'static str,
    /// Chat requests received.
    hits: Arc<AtomicUsize>,
    /// Streams that were cut by the peer before they completed.
    cancelled: Arc<AtomicUsize>,
    /// What `/health` and `/health/ready` answer (200 vs 503).
    ready: Arc<AtomicBool>,
    /// When set, `/health` and `/health/ready` still answer 200 but carry `tensor_parallel.state = "lost"` (a leader whose group is
    /// lost and whose engine has not failed yet).
    group_lost: Arc<AtomicBool>,
}

struct CancelGuard {
    counter: Arc<AtomicUsize>,
    completed: bool,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if !self.completed {
            self.counter.fetch_add(1, Ordering::SeqCst);
        }
    }
}

async fn spawn_mock(tag: &'static str, mode: Mode) -> Mock {
    let hits = Arc::new(AtomicUsize::new(0));
    let cancelled = Arc::new(AtomicUsize::new(0));
    let ready = Arc::new(AtomicBool::new(true));
    let group_lost = Arc::new(AtomicBool::new(false));
    let (h, c, r1, r2) = (hits.clone(), cancelled.clone(), ready.clone(), ready.clone());
    let (l1, l2) = (group_lost.clone(), group_lost.clone());
    let health = move |ready: Arc<AtomicBool>, lost: Arc<AtomicBool>| {
        move || {
            let (ready, lost) = (ready.clone(), lost.clone());
            async move {
                if lost.load(Ordering::SeqCst) {
                    (
                        axum::http::StatusCode::OK,
                        axum::Json(serde_json::json!({"status": "ok", "tensor_parallel": {"ranks": 2, "state": "lost",
                            "lost": "the connection to rank 1 failed: connection closed by the peer", "lost_rank": 1}})),
                    )
                } else if ready.load(Ordering::SeqCst) {
                    (axum::http::StatusCode::OK, axum::Json(serde_json::json!({"status": "ready"})))
                } else {
                    (axum::http::StatusCode::SERVICE_UNAVAILABLE, axum::Json(serde_json::json!({"status": "loading"})))
                }
            }
        }
    };
    let app = Router::new()
        .route("/health", get(health(r1, l1)))
        .route("/health/ready", get(health(r2, l2)))
        .route("/metrics", get(|| async { "rexl3_tokens{kind=\"drafted\"} 5\nrexl3_ready 1\n" }))
        .route(
            "/v1/chat/completions",
            post(move |headers: HeaderMap, body: Bytes| {
                let (hits, cancelled) = (h.clone(), c.clone());
                async move {
                    hits.fetch_add(1, Ordering::SeqCst);
                    let seen_id = headers.get("x-request-id").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
                    let mut b = axum::response::Response::builder().header("x-seen-request-id", seen_id);
                    match mode {
                        Mode::Json => {
                            let echo = format!("{{\"backend\":\"{tag}\",\"bytes\":{}}}", body.len());
                            b = b.header("content-type", "application/json");
                            b.body(Body::from(echo)).unwrap()
                        }
                        Mode::Unavailable => b
                            .status(503)
                            .header("content-type", "application/json")
                            .body(Body::from(
                                "{\"error\":{\"message\":\"the engine failed: boom\",\"type\":\"service_unavailable\",\"code\":503}}",
                            ))
                            .unwrap(),
                        Mode::Overloaded => b
                            .status(503)
                            .header("content-type", "application/json")
                            .body(Body::from(
                                "{\"error\":{\"message\":\"no capacity\",\"type\":\"server_overloaded\",\"code\":503}}",
                            ))
                            .unwrap(),
                        Mode::Stream { chunks, gap_ms } => {
                            let guard = CancelGuard { counter: cancelled, completed: false };
                            let stream = futures_util::stream::unfold((0usize, guard), move |(i, mut guard)| async move {
                                if i >= chunks {
                                    guard.completed = true;
                                    return None;
                                }
                                if i > 0 {
                                    tokio::time::sleep(Duration::from_millis(gap_ms)).await;
                                }
                                let ev = Bytes::from(format!("data: {tag}-{i}\n\n"));
                                Some((Ok::<_, std::io::Error>(ev), (i + 1, guard)))
                            });
                            b.header("content-type", "text/event-stream").body(Body::from_stream(stream)).unwrap()
                        }
                        Mode::ErrorThenClose => {
                            let events = error_then_close_body(tag)
                                .split_inclusive("\n\n")
                                .map(|e| Ok::<_, std::io::Error>(Bytes::from(e.to_string())))
                                .collect::<Vec<_>>();
                            let stream = futures_util::stream::unfold(events.into_iter(), |mut it| async move {
                                tokio::time::sleep(Duration::from_millis(20)).await;
                                it.next().map(|e| (e, it))
                            });
                            b.header("content-type", "text/event-stream").body(Body::from_stream(stream)).unwrap()
                        }
                        Mode::AbortAfterFirst => {
                            let stream = futures_util::stream::unfold(0usize, move |i| async move {
                                match i {
                                    0 => Some((Ok(Bytes::from(format!("data: {tag}-0\n\n"))), 1)),
                                    1 => {
                                        tokio::time::sleep(Duration::from_millis(50)).await;
                                        Some((Err(std::io::Error::other("backend died")), 2))
                                    }
                                    _ => None,
                                }
                            });
                            b.header("content-type", "text/event-stream").body(Body::from_stream(stream)).unwrap()
                        }
                    }
                }
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Mock { url: format!("http://{addr}"), tag, hits, cancelled, ready, group_lost }
}

impl Mock {
    fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }
}

/// A fast-probing config for tests that start the prober.
fn fast_cfg() -> Config {
    Config {
        probe_interval: Duration::from_millis(30),
        probe_timeout: Duration::from_millis(500),
        fail_threshold: 2,
        recover_threshold: 2,
        ..Config::default()
    }
}

async fn spawn_router(backends: Vec<Backend>, cfg: Config) -> (String, App) {
    let app = App::with_config(backends, cfg);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = build_router(app.clone());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{addr}"), app)
}

async fn spawn_admin(app: &App, token: Option<&str>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = build_admin_router(app.clone(), token.map(str::to_string));
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    format!("http://{addr}")
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().timeout(Duration::from_secs(20)).build().unwrap()
}

async fn wait_until(what: &str, cond: impl Fn() -> bool) {
    for _ in 0..500 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for: {what}");
}

/// A chat completion with no affinity header: the key comes from the prompt.
async fn chat(router: &str, system: &str, user: &str) -> reqwest::Response {
    client()
        .post(format!("{router}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "m", "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": 16}))
        .send()
        .await
        .unwrap()
}

async fn chat_keyed(router: &str, key: &str) -> reqwest::Response {
    client()
        .post(format!("{router}/v1/chat/completions"))
        .header("x-affinity-key", key)
        .json(&serde_json::json!({"messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
}

fn served_by(resp: &reqwest::Response) -> String {
    resp.headers().get("x-rexl3-backend").map(|v| v.to_str().unwrap().to_string()).unwrap_or_default()
}

async fn scrape(router: &str) -> String {
    client().get(format!("{router}/metrics?scope=router")).send().await.unwrap().text().await.unwrap()
}

/// The value of the series whose line starts with `prefix` (name and labels, up to the space).
fn metric(text: &str, prefix: &str) -> f64 {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(prefix) {
            if let Some(v) = rest.strip_prefix(' ') {
                return v.trim().parse().unwrap();
            }
        }
    }
    panic!("series {prefix} not found in:\n{text}");
}

fn bid(m: &Mock) -> String {
    m.url.trim_start_matches("http://").to_string()
}

/// 1. A backend that stops answering `/health/ready` is taken out after K failures and comes back after M successes,
/// without restarting the router.
#[tokio::test]
async fn prober_marks_down_and_recovers_a_backend() {
    let m = spawn_mock("a", Mode::Json).await;
    let b = Backend::new(m.url.clone());
    let (router, app) = spawn_router(vec![b.clone()], fast_cfg()).await;
    app.spawn_prober();
    let c = client();

    assert_eq!(chat_keyed(&router, "k").await.status(), 200);
    m.ready.store(false, Ordering::SeqCst);
    wait_until("marked down", || !b.is_healthy()).await;
    let resp = chat_keyed(&router, "k").await;
    assert_eq!(resp.status(), 502, "no routable backend");
    let h = c.get(format!("{router}/health")).send().await.unwrap();
    assert_eq!(h.status(), 503);
    let doc: serde_json::Value = h.json().await.unwrap();
    assert_eq!(doc["any_up"], false);
    assert_eq!(doc["backends"][0]["up"], false);
    assert_eq!(doc["backends"][0]["last_probe"]["status"], 503);

    // the backend returns: up again after M good probes, then it serves
    m.ready.store(true, Ordering::SeqCst);
    wait_until("marked up", || b.is_healthy()).await;
    assert_eq!(chat_keyed(&router, "k").await.status(), 200);
    let text = scrape(&router).await;
    let id = bid(&m);
    assert_eq!(metric(&text, &format!("rexl3_router_health_transitions_total{{backend=\"{id}\",to=\"down\"}}")), 1.0);
    assert_eq!(metric(&text, &format!("rexl3_router_health_transitions_total{{backend=\"{id}\",to=\"up\"}}")), 1.0);
    assert_eq!(metric(&text, &format!("rexl3_router_backend_up{{backend=\"{id}\"}}")), 1.0);
}

/// 1a. A tensor-parallel leader that answers 200 but reports its group lost (found on the real pair, 2026-10-11: for up to 20 s after a
/// follower died with nothing in flight, the leader kept `/health/ready` at 200) is taken out like a dead one, and comes back when the
/// group is whole again (a restarted group answers without `lost`).
#[tokio::test]
async fn a_leader_that_reports_its_group_lost_is_taken_out_though_it_answers_200() {
    let m = spawn_mock("a", Mode::Json).await;
    let b = Backend::new(m.url.clone());
    let (router, app) = spawn_router(vec![b.clone()], fast_cfg()).await;
    app.spawn_prober();
    let c = client();
    assert_eq!(chat_keyed(&router, "k").await.status(), 200);

    m.group_lost.store(true, Ordering::SeqCst);
    wait_until("marked down for a lost group", || !b.is_healthy()).await;
    assert_eq!(chat_keyed(&router, "k").await.status(), 502, "no routable backend");
    let doc: serde_json::Value = c.get(format!("{router}/health")).send().await.unwrap().json().await.unwrap();
    assert_eq!(doc["any_up"], false);
    assert_eq!(doc["backends"][0]["last_probe"]["status"], 200, "the backend itself answered 200");
    let why = doc["backends"][0]["last_probe"]["error"].as_str().unwrap();
    assert!(why.contains("tensor-parallel group lost") && why.contains("rank 1"), "{why}");

    m.group_lost.store(false, Ordering::SeqCst);
    wait_until("marked up again", || b.is_healthy()).await;
    assert_eq!(chat_keyed(&router, "k").await.status(), 200);
}

/// 1b. A single failed probe does not flap a healthy backend; K in a row do (thresholds are on consecutive results).
#[test]
fn probe_thresholds_are_consecutive() {
    use crate::backend::ProbeInfo;
    let b = Backend::new("http://127.0.0.1:1");
    let info = || ProbeInfo::default();
    assert_eq!(b.apply_probe(false, info(), 3, 2), None);
    assert_eq!(b.apply_probe(true, info(), 3, 2), None);
    assert_eq!(b.apply_probe(false, info(), 3, 2), None);
    assert_eq!(b.apply_probe(false, info(), 3, 2), None);
    assert_eq!(b.apply_probe(false, info(), 3, 2), Some(false));
    assert!(!b.is_healthy());
    assert_eq!(b.apply_probe(true, info(), 3, 2), None);
    assert_eq!(b.apply_probe(false, info(), 3, 2), None);
    assert_eq!(b.apply_probe(true, info(), 3, 2), None);
    assert_eq!(b.apply_probe(true, info(), 3, 2), Some(true));
    assert!(b.is_healthy());
}

/// 2. Without `x-affinity-key`, a conversation (same system prompt and first user message) pins to one backend
/// whatever follows, distinct prompts spread over the fleet, and the header still wins.
#[tokio::test]
async fn default_affinity_follows_the_prompt_prefix() {
    let a = spawn_mock("a", Mode::Json).await;
    let b = spawn_mock("b", Mode::Json).await;
    let (router, _app) = spawn_router(vec![Backend::new(a.url.clone()), Backend::new(b.url.clone())], Config::default()).await;
    let c = client();

    // one conversation: later turns and sampling fields change, the prefix does not
    let first = chat(&router, "You are terse.", "What is a prefix cache?").await;
    let home = served_by(&first);
    assert!(!home.is_empty());
    for turn in 0..8 {
        let resp = c
            .post(format!("{router}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "m", "temperature": 0.1 * turn as f64, "stream": false,
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "What is a prefix cache?"},
                    {"role": "assistant", "content": "A cache of KV state."},
                    {"role": "user", "content": format!("follow-up number {turn}")}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(served_by(&resp), home, "turn {turn} must stay on the backend that holds the prefix");
    }

    // 24 distinct conversations spread over both backends
    let mut per_backend = std::collections::HashMap::<String, usize>::new();
    for i in 0..24 {
        let resp = chat(&router, "You are terse.", &format!("question {i} about something else")).await;
        *per_backend.entry(served_by(&resp)).or_default() += 1;
    }
    assert_eq!(per_backend.len(), 2, "both backends take traffic: {per_backend:?}");
    assert!(per_backend.values().all(|&n| n >= 4), "no backend is starved: {per_backend:?}");

    // the header wins over the prompt: two different prompts with one header land together
    let h1 = c
        .post(format!("{router}/v1/chat/completions"))
        .header("x-affinity-key", "tenant-7")
        .json(&serde_json::json!({"messages": [{"role": "user", "content": "alpha"}]}))
        .send()
        .await
        .unwrap();
    let h2 = c
        .post(format!("{router}/v1/chat/completions"))
        .header("x-affinity-key", "tenant-7")
        .json(&serde_json::json!({"messages": [{"role": "user", "content": "completely different"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(served_by(&h1), served_by(&h2));
    assert_eq!(a.hits() + b.hits(), 1 + 8 + 24 + 2);
}

/// 3. In-flight is held until the stream ends: a request mid-stream counts against its backend, so the next request
/// goes elsewhere; the count falls back to zero at the end of the body, and when the client leaves early.
#[tokio::test]
async fn inflight_is_held_through_a_streaming_body() {
    let a = spawn_mock("a", Mode::Stream { chunks: 4, gap_ms: 150 }).await;
    let b = spawn_mock("b", Mode::Stream { chunks: 4, gap_ms: 150 }).await;
    let (ba, bb) = (Backend::new(a.url.clone()), Backend::new(b.url.clone()));
    let (router, _app) = spawn_router(vec![ba.clone(), bb.clone()], Config::default()).await;

    let mut r1 = chat_keyed(&router, "s1").await;
    let first_backend = served_by(&r1);
    assert_eq!(r1.status(), 200);
    let first = r1.chunk().await.unwrap().unwrap();
    assert!(std::str::from_utf8(&first).unwrap().starts_with("data: "));
    // headers and the first event are out, the stream is still running
    let (busy, idle) = if first_backend == ba.id() { (&ba, &bb) } else { (&bb, &ba) };
    assert_eq!(busy.inflight.load(Ordering::SeqCst), 1, "a streaming request stays outstanding");
    assert_eq!(idle.inflight.load(Ordering::SeqCst), 0);

    // least-outstanding sees it: a different key goes to the idle backend
    let mut r2 = chat_keyed(&router, "s2").await;
    assert_eq!(served_by(&r2), idle.id());
    let text = scrape(&router).await;
    assert_eq!(metric(&text, &format!("rexl3_router_backend_inflight{{backend=\"{}\"}}", busy.id())), 1.0);

    // both streams finish byte-for-byte, then everything is released
    let mut body1 = first.to_vec();
    while let Some(c) = r1.chunk().await.unwrap() {
        body1.extend_from_slice(&c);
    }
    let tag = if first_backend == ba.id() { "a" } else { "b" };
    let expect: String = (0..4).map(|i| format!("data: {tag}-{i}\n\n")).collect();
    assert_eq!(String::from_utf8(body1).unwrap(), expect);
    while r2.chunk().await.unwrap().is_some() {}
    wait_until("inflight back to 0", || ba.inflight.load(Ordering::SeqCst) == 0 && bb.inflight.load(Ordering::SeqCst) == 0).await;

    // a client that leaves mid-stream releases the slot and the backend sees the cancel
    let mut r3 = chat_keyed(&router, "s3").await;
    let victim = served_by(&r3);
    r3.chunk().await.unwrap().unwrap();
    drop(r3);
    let (vb, vm) = if victim == ba.id() { (&ba, &a) } else { (&bb, &b) };
    wait_until("released after disconnect", || vb.inflight.load(Ordering::SeqCst) == 0).await;
    wait_until("backend saw the cancel", || vm.cancelled.load(Ordering::SeqCst) >= 1).await;
    let text = scrape(&router).await;
    assert!(metric(&text, "rexl3_router_client_disconnects_total") >= 1.0);
}

/// 4. The affinity map stays within its bound (LRU) while distinct keys keep arriving.
#[tokio::test]
async fn affinity_map_is_bounded() {
    let a = spawn_mock("a", Mode::Json).await;
    let b = spawn_mock("b", Mode::Json).await;
    let cfg = Config { affinity_capacity: 3, ..Config::default() };
    let (router, app) = spawn_router(vec![Backend::new(a.url.clone()), Backend::new(b.url.clone())], cfg).await;
    for i in 0..10 {
        assert_eq!(chat_keyed(&router, &format!("key-{i}")).await.status(), 200);
    }
    assert_eq!(app.affinity_stats().0, 3);
    let text = scrape(&router).await;
    assert_eq!(metric(&text, "rexl3_router_affinity_entries"), 3.0);
    assert_eq!(metric(&text, "rexl3_router_affinity_evictions_total{reason=\"capacity\"}"), 7.0);
    // the most recent keys are still bound: a hit
    assert_eq!(chat_keyed(&router, "key-9").await.status(), 200);
    let text = scrape(&router).await;
    assert_eq!(metric(&text, "rexl3_router_affinity_requests_total{result=\"hit\"}"), 1.0);
}

/// 5. Drain: in-flight finishes, nothing new arrives (not even a key bound there), the aggregate health shows it; enable
/// puts it back; backends can be removed and added at runtime.
#[tokio::test]
async fn drain_keeps_inflight_and_refuses_new() {
    let a = spawn_mock("a", Mode::Stream { chunks: 5, gap_ms: 120 }).await;
    let b = spawn_mock("b", Mode::Stream { chunks: 5, gap_ms: 120 }).await;
    let (router, app) = spawn_router(vec![Backend::new(a.url.clone()), Backend::new(b.url.clone())], Config::default()).await;
    let admin = spawn_admin(&app, None).await;
    let c = client();

    let mut r1 = chat_keyed(&router, "drain-me").await;
    let target = served_by(&r1);
    let (target_mock, other_mock) = if target == bid(&a) { (&a, &b) } else { (&b, &a) };
    r1.chunk().await.unwrap().unwrap();

    let resp = c.post(format!("{admin}/admin/backends/{target}/drain")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let doc: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(doc["admin"], "draining");
    assert_eq!(doc["routable"], false);
    assert_eq!(doc["drained"], false, "still has a request in flight");
    assert_eq!(doc["inflight"], 1);

    // new requests, including the key bound to the drained backend, go to the other one
    for key in ["drain-me", "x1", "x2", "x3"] {
        let mut r = chat_keyed(&router, key).await;
        assert_eq!(served_by(&r), bid(other_mock), "key {key}");
        while r.chunk().await.unwrap().is_some() {}
    }
    assert_eq!(target_mock.hits(), 1);

    // the aggregate health reports it; the router is still up
    let h = c.get(format!("{router}/health")).send().await.unwrap();
    assert_eq!(h.status(), 200);
    let doc: serde_json::Value = h.json().await.unwrap();
    let entry = doc["backends"].as_array().unwrap().iter().find(|e| e["backend"] == target.as_str()).unwrap().clone();
    assert_eq!(entry["admin"], "draining");
    assert_eq!(entry["routable"], false);
    assert_eq!(doc["routable"], 1);

    // the in-flight stream finishes completely
    let mut body = Vec::new();
    while let Some(chunk) = r1.chunk().await.unwrap() {
        body.extend_from_slice(&chunk);
    }
    assert!(String::from_utf8(body).unwrap().ends_with(&format!("data: {}-4\n\n", target_mock.tag)));
    let doc: serde_json::Value = c.get(format!("{admin}/admin/backends/{target}")).send().await.unwrap().json().await.unwrap();
    assert_eq!(doc["drained"], true, "drain completes when in-flight reaches zero");

    // enable: back in rotation
    let resp = c.post(format!("{admin}/admin/backends/{target}/enable")).send().await.unwrap();
    assert_eq!(resp.json::<serde_json::Value>().await.unwrap()["routable"], true);
    let mut landed = false;
    for i in 0..40 {
        let mut r = chat_keyed(&router, &format!("again-{i}")).await;
        landed = served_by(&r) == target;
        while r.chunk().await.unwrap().is_some() {}
        if landed {
            break;
        }
    }
    assert!(landed, "an enabled backend takes traffic again");

    // runtime membership: remove one, the fleet shrinks; add a new one, it probes up and serves
    let resp = c.delete(format!("{admin}/admin/backends/{target}")).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    let doc: serde_json::Value = c.get(format!("{router}/health")).send().await.unwrap().json().await.unwrap();
    assert_eq!(doc["total"], 1);
    let c2 = spawn_mock("c", Mode::Json).await;
    let resp = c.post(format!("{admin}/admin/backends")).json(&serde_json::json!({"url": c2.url})).send().await.unwrap();
    assert_eq!(resp.status(), 201);
    let doc: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(doc["up"], true, "the add probes once, so a ready backend is up at once");
    let dup = c.post(format!("{admin}/admin/backends")).json(&serde_json::json!({"url": c2.url})).send().await.unwrap();
    assert_eq!(dup.status(), 409);
    let mut seen_c = false;
    for i in 0..40 {
        let mut r = chat_keyed(&router, &format!("new-{i}")).await;
        seen_c = served_by(&r) == bid(&c2);
        while r.chunk().await.unwrap().is_some() {}
        if seen_c {
            break;
        }
    }
    assert!(seen_c, "the added backend takes traffic");
}

/// 6. Retry happens before the first response byte only, and only for "the backend cannot serve" failures.
#[tokio::test]
async fn retry_before_first_byte_only() {
    // (a) an engine-unavailable 503 from the preferred backend: re-sent once to the other; the failed one goes down.
    {
        let a = spawn_mock("a", Mode::Unavailable).await;
        let b = spawn_mock("b", Mode::Json).await;
        let (ba, bb) = (Backend::new(a.url.clone()), Backend::new(b.url.clone()));
        bb.inflight.store(3, Ordering::SeqCst); // least-outstanding prefers a
        let (router, _app) = spawn_router(vec![ba.clone(), bb.clone()], Config::default()).await;
        let resp = chat_keyed(&router, "r1").await;
        assert_eq!(resp.status(), 200);
        assert_eq!(served_by(&resp), bid(&b));
        assert_eq!((a.hits(), b.hits()), (1, 1));
        assert!(!ba.is_healthy(), "an unavailable engine is taken out at once");
        let text = scrape(&router).await;
        assert_eq!(metric(&text, "rexl3_router_failovers_total"), 1.0);
        assert_eq!(
            metric(&text, &format!("rexl3_router_backend_errors_total{{backend=\"{}\",kind=\"unavailable\"}}", bid(&a))),
            1.0
        );
        assert_eq!(metric(&text, &format!("rexl3_router_backend_failovers_total{{backend=\"{}\"}}", bid(&a))), 1.0);
    }
    // (b) after the first byte nothing is retried: the client sees a broken stream, the other backend sees nothing
    {
        let a = spawn_mock("a", Mode::AbortAfterFirst).await;
        let b = spawn_mock("b", Mode::Json).await;
        let bb = Backend::new(b.url.clone());
        bb.inflight.store(3, Ordering::SeqCst);
        let (router, _app) = spawn_router(vec![Backend::new(a.url.clone()), bb], Config::default()).await;
        let mut resp = chat_keyed(&router, "r2").await;
        assert_eq!(resp.status(), 200);
        let first = resp.chunk().await.unwrap().unwrap();
        assert_eq!(&first[..], b"data: a-0\n\n");
        let rest = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match resp.chunk().await {
                    Ok(Some(_)) => continue,
                    other => break other,
                }
            }
        })
        .await
        .expect("a dying upstream must end the client's stream, not hang it");
        assert!(rest.is_err(), "the client sees an error, not a clean end: {rest:?}");
        assert_eq!(b.hits(), 0, "no retry once bytes were sent");
        let text = scrape(&router).await;
        assert_eq!(metric(&text, "rexl3_router_failovers_total"), 0.0);
        assert_eq!(metric(&text, &format!("rexl3_router_backend_errors_total{{backend=\"{}\",kind=\"stream\"}}", bid(&a))), 1.0);
    }
    // (c) a refused connection fails over transparently
    {
        let live = spawn_mock("live", Mode::Json).await;
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dead_port = l.local_addr().unwrap().port();
        drop(l);
        let dead = Backend::new(format!("http://127.0.0.1:{dead_port}"));
        let lb = Backend::new(live.url.clone());
        lb.inflight.store(3, Ordering::SeqCst);
        let (router, _app) = spawn_router(vec![dead.clone(), lb], Config::default()).await;
        let resp = chat_keyed(&router, "r3").await;
        assert_eq!(resp.status(), 200);
        assert_eq!(served_by(&resp), bid(&live));
        assert!(!dead.is_healthy());
    }
    // (d) a body above the replay cap is streamed through and cannot be re-sent: 502 with the backend id
    {
        let live = spawn_mock("live", Mode::Json).await;
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dead_port = l.local_addr().unwrap().port();
        drop(l);
        let dead = Backend::new(format!("http://127.0.0.1:{dead_port}"));
        let lb = Backend::new(live.url.clone());
        lb.inflight.store(3, Ordering::SeqCst);
        let cfg = Config { replay_body_cap: 64, ..Config::default() };
        let (router, _app) = spawn_router(vec![dead, lb], cfg).await;
        let big = serde_json::json!({"messages": [{"role": "user", "content": "x".repeat(4096)}]});
        let resp = client().post(format!("{router}/v1/chat/completions")).json(&big).send().await.unwrap();
        assert_eq!(resp.status(), 502);
        let doc: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(doc["backend"], format!("127.0.0.1:{dead_port}"));
        assert_eq!(live.hits(), 0);
        let text = scrape(&router).await;
        assert_eq!(metric(&text, "rexl3_router_body_over_replay_cap_total"), 1.0);
    }
    // (e) an overload 503 is the backend's own decision: passed through, backend stays healthy, no failover
    {
        let a = spawn_mock("a", Mode::Overloaded).await;
        let b = spawn_mock("b", Mode::Json).await;
        let (ba, bb) = (Backend::new(a.url.clone()), Backend::new(b.url.clone()));
        bb.inflight.store(3, Ordering::SeqCst);
        let (router, _app) = spawn_router(vec![ba.clone(), bb], Config::default()).await;
        let resp = chat_keyed(&router, "r5").await;
        assert_eq!(resp.status(), 503);
        let doc: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(doc["error"]["type"], "server_overloaded");
        assert_eq!(b.hits(), 0);
        assert!(ba.is_healthy());
    }
}

/// 7. A bound key is not kept on an overloaded backend (one hot prompt cannot pile the fleet onto one node).
#[tokio::test]
async fn bound_key_yields_to_a_far_less_loaded_backend() {
    let a = spawn_mock("a", Mode::Json).await;
    let b = spawn_mock("b", Mode::Json).await;
    let (ba, bb) = (Backend::new(a.url.clone()), Backend::new(b.url.clone()));
    let cfg = Config { affinity_max_skew: 4, ..Config::default() };
    let (router, _app) = spawn_router(vec![ba.clone(), bb.clone()], cfg).await;
    let home = served_by(&chat_keyed(&router, "hot").await);
    let (home_b, other_b) = if home == ba.id() { (&ba, &bb) } else { (&bb, &ba) };
    // within the skew the key stays
    home_b.inflight.store(4, Ordering::SeqCst);
    assert_eq!(served_by(&chat_keyed(&router, "hot").await), home);
    // beyond it, the least loaded backend takes the request and the binding is kept
    home_b.inflight.store(5, Ordering::SeqCst);
    assert_eq!(served_by(&chat_keyed(&router, "hot").await), other_b.id());
    home_b.inflight.store(0, Ordering::SeqCst);
    assert_eq!(served_by(&chat_keyed(&router, "hot").await), home);
    let text = scrape(&router).await;
    assert_eq!(metric(&text, "rexl3_router_affinity_requests_total{result=\"overflow\"}"), 1.0);
}

/// 8. Admin: token required when set, absent from the public port, input validated.
#[tokio::test]
async fn admin_requires_the_token_and_is_not_on_the_public_port() {
    let a = spawn_mock("a", Mode::Json).await;
    let backend = Backend::new(a.url.clone());
    let (router, app) = spawn_router(vec![backend.clone()], Config::default()).await;
    let admin = spawn_admin(&app, Some("s3cret")).await;
    let c = client();
    let drain = format!("{admin}/admin/backends/{}/drain", backend.id());

    assert_eq!(c.get(format!("{admin}/admin/backends")).send().await.unwrap().status(), 401);
    assert_eq!(c.post(&drain).send().await.unwrap().status(), 401);
    assert_eq!(c.post(&drain).bearer_auth("wrong").send().await.unwrap().status(), 401);
    assert_eq!(c.post(&drain).header("authorization", "s3cret").send().await.unwrap().status(), 401);
    assert!(backend.routable(), "refused requests change nothing");

    assert_eq!(c.get(format!("{admin}/admin/backends")).bearer_auth("s3cret").send().await.unwrap().status(), 200);
    assert_eq!(c.post(&drain).header("x-admin-token", "s3cret").send().await.unwrap().status(), 200);
    assert!(!backend.routable());
    assert_eq!(
        c.post(format!("{admin}/admin/backends/{}/enable", backend.id())).bearer_auth("s3cret").send().await.unwrap().status(),
        200
    );
    assert_eq!(c.post(format!("{admin}/admin/backends/nope:1/drain")).bearer_auth("s3cret").send().await.unwrap().status(), 404);
    let bad = c
        .post(format!("{admin}/admin/backends"))
        .bearer_auth("s3cret")
        .json(&serde_json::json!({"url": "ftp://x:1"}))
        .send()
        .await
        .unwrap();
    assert_eq!(bad.status(), 400);

    // the public port neither serves nor forwards the admin API
    let r = c.post(format!("{router}/admin/backends/{}/drain", backend.id())).bearer_auth("s3cret").send().await.unwrap();
    assert_eq!(r.status(), 404);
    let r = c.get(format!("{router}/admin/backends")).send().await.unwrap();
    assert_eq!(r.status(), 404);
    assert!(backend.routable());
    assert_eq!(a.hits(), 0, "nothing under /admin reached a backend");
}

/// 9. The router's own metrics: requests, per-backend in-flight, errors, failovers, transitions and affinity, next to
/// the merged backend series; request ids are propagated.
#[tokio::test]
async fn metrics_series_and_request_ids() {
    let a = spawn_mock("a", Mode::Json).await;
    let b = spawn_mock("b", Mode::Json).await;
    let (router, _app) = spawn_router(vec![Backend::new(a.url.clone()), Backend::new(b.url.clone())], Config::default()).await;
    let c = client();

    for _ in 0..3 {
        assert_eq!(chat_keyed(&router, "m-key").await.status(), 200);
    }
    // a prompt-keyed request carrying the caller's id
    let resp = c
        .post(format!("{router}/v1/chat/completions"))
        .header("x-request-id", "trace-42")
        .json(&serde_json::json!({"prompt": "no messages here"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.headers().get("x-seen-request-id").unwrap(), "trace-42", "the backend saw the caller's id");
    assert_eq!(resp.headers().get("x-request-id").unwrap(), "trace-42", "and the client gets it back");
    // a request with no key at all and no id: the router makes one and the backend sees it
    let resp = c.post(format!("{router}/v1/chat/completions")).json(&serde_json::json!({})).send().await.unwrap();
    let made = resp.headers().get("x-request-id").unwrap().to_str().unwrap().to_string();
    assert!(made.starts_with("rtr-"), "{made}");
    assert_eq!(resp.headers().get("x-seen-request-id").unwrap().to_str().unwrap(), made);

    wait_until("all released", || _app.backends().iter().all(|b| b.inflight.load(Ordering::SeqCst) == 0)).await;
    let text = scrape(&router).await;
    let (ia, ib) = (bid(&a), bid(&b));
    assert_eq!(metric(&text, "rexl3_router_requests_total"), 5.0);
    assert_eq!(metric(&text, "rexl3_router_affinity_requests_total{result=\"hit\"}"), 2.0);
    assert_eq!(metric(&text, "rexl3_router_affinity_requests_total{result=\"miss\"}"), 2.0);
    assert_eq!(metric(&text, "rexl3_router_affinity_requests_total{result=\"none\"}"), 1.0);
    assert_eq!(metric(&text, "rexl3_router_affinity_key_source_total{source=\"header\"}"), 3.0);
    assert_eq!(metric(&text, "rexl3_router_affinity_key_source_total{source=\"prompt\"}"), 1.0);
    let per = |name: &str| {
        metric(&text, &format!("{name}{{backend=\"{ia}\"}}")) + metric(&text, &format!("{name}{{backend=\"{ib}\"}}"))
    };
    assert_eq!(per("rexl3_router_backend_requests_total"), 5.0);
    assert_eq!(per("rexl3_router_backend_inflight"), 0.0);
    assert_eq!(per("rexl3_router_backend_up"), 2.0);
    let ok2xx = |id: &str| metric(&text, &format!("rexl3_router_backend_responses_total{{backend=\"{id}\",class=\"2xx\"}}"));
    assert_eq!(ok2xx(&ia) + ok2xx(&ib), 5.0);
    assert_eq!(metric(&text, "rexl3_router_failovers_total"), 0.0);
    assert_eq!(metric(&text, "rexl3_router_no_backend_total"), 0.0);
    assert_eq!(metric(&text, &format!("rexl3_router_backend_admin_state{{backend=\"{ia}\",state=\"enabled\"}}")), 1.0);
    assert_eq!(metric(&text, &format!("rexl3_router_backend_errors_total{{backend=\"{ia}\",kind=\"connect\"}}")), 0.0);
    assert!(text.contains("# TYPE rexl3_router_requests_total counter"));

    // the default scrape appends every backend's series under the router prefix with a backend label
    let full = c.get(format!("{router}/metrics")).send().await.unwrap().text().await.unwrap();
    assert!(full.contains(&format!("rexl3_router_rexl3_tokens{{backend=\"{ia}\",kind=\"drafted\"}} 5")), "{full}");
    assert!(full.contains(&format!("rexl3_router_rexl3_ready{{backend=\"{ib}\"}} 1")), "{full}");
}

/// 10. An upstream that goes silent mid-stream is cut after the idle timeout; the client gets an error and the slot is
/// released.
#[tokio::test]
async fn stalled_upstream_is_cut() {
    let a = spawn_mock("a", Mode::Stream { chunks: 3, gap_ms: 5000 }).await;
    let cfg = Config { stream_idle_timeout: Duration::from_millis(300), ..Config::default() };
    let backend = Backend::new(a.url.clone());
    let (router, _app) = spawn_router(vec![backend.clone()], cfg).await;
    let mut resp = chat_keyed(&router, "stall").await;
    assert_eq!(resp.status(), 200);
    assert!(resp.chunk().await.unwrap().is_some());
    let started = std::time::Instant::now();
    let end = tokio::time::timeout(Duration::from_secs(5), resp.chunk()).await.expect("must not hang");
    assert!(end.is_err(), "the broken stream is an error for the client: {end:?}");
    assert!(started.elapsed() < Duration::from_secs(3));
    wait_until("slot released", || backend.inflight.load(Ordering::SeqCst) == 0).await;
}

/// 10b. A stream cut by an engine failure ends with its in-band error event and then CLOSES, with no `[DONE]` (the server's
/// convention for an errored stream, so that a client that takes `[DONE]` for completion never reads a short reply as a whole
/// one). The router must pass those bytes through unchanged, end the client's stream cleanly when the upstream ends, and
/// release the in-flight slot: nothing in it waits for a `[DONE]`. (A silent upstream is the idle timeout's, test 10.)
#[tokio::test]
async fn a_stream_cut_by_an_engine_failure_passes_through_without_done_and_releases_the_slot() {
    let a = spawn_mock("a", Mode::ErrorThenClose).await;
    let backend = Backend::new(a.url.clone());
    // the idle timeout is long: it is the end of the stream that releases the slot and ends the client's body, not the timeout
    let cfg = Config { stream_idle_timeout: Duration::from_secs(60), ..Config::default() };
    let (router, _app) = spawn_router(vec![backend.clone()], cfg).await;
    let resp = chat_keyed(&router, "cut").await;
    assert_eq!(resp.status(), 200);
    let started = std::time::Instant::now();
    let body = tokio::time::timeout(Duration::from_secs(5), resp.text()).await.expect("must not hang").expect("a clean end");
    assert!(started.elapsed() < Duration::from_secs(2), "the stream ended with the upstream's end: {:?}", started.elapsed());
    assert_eq!(body, error_then_close_body("a"), "byte-exact pass-through, error event last");
    assert!(!body.contains("[DONE]"));
    wait_until("slot released", || backend.inflight.load(Ordering::SeqCst) == 0).await;
}

/// 11. A backend that was disabled is not probed, so `enable` probes it once before it can take traffic: one that died
/// meanwhile does not return to rotation, and the prober brings it up when it is ready.
#[tokio::test]
async fn enable_probes_a_disabled_backend_first() {
    let a = spawn_mock("a", Mode::Json).await;
    let b = spawn_mock("b", Mode::Json).await;
    let (ba, bb) = (Backend::new(a.url.clone()), Backend::new(b.url.clone()));
    let (router, app) = spawn_router(vec![ba.clone(), bb.clone()], Config::default()).await;
    let admin = spawn_admin(&app, None).await;
    let c = client();

    let resp = c.post(format!("{admin}/admin/backends/{}/disable", ba.id())).send().await.unwrap();
    assert_eq!(resp.status(), 200);
    a.ready.store(false, Ordering::SeqCst); // it went away while out of rotation; nobody probed it
    let resp = c.post(format!("{admin}/admin/backends/{}/enable", ba.id())).send().await.unwrap();
    let doc: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(doc["admin"], "enabled");
    assert_eq!(doc["up"], false, "the enable probe saw it unready");
    assert_eq!(doc["routable"], false);
    for i in 0..10 {
        assert_eq!(served_by(&chat_keyed(&router, &format!("e-{i}")).await), bid(&b));
    }
    assert_eq!(a.hits(), 0);

    // ready again: the regular prober rounds (M = 2) bring it back
    a.ready.store(true, Ordering::SeqCst);
    app.probe_round(false).await;
    app.probe_round(false).await;
    assert!(ba.routable());
}
