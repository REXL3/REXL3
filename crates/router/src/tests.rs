//! CPU-only integration tests: mock axum backends bound to 127.0.0.1:0.

use crate::app::App;
use crate::backend::Backend;
use crate::handlers::build_router;
use axum::body::Body;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::Router;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

/// A spawn handle for a mock backend: base URL plus per-path hit counts.
#[derive(Clone)]
struct Mock {
    url: String,
    hits: Arc<Mutex<HashMap<String, usize>>>,
}

impl Mock {
    fn count(&self, path: &str) -> usize {
        *self.hits.lock().unwrap().get(path).unwrap_or(&0)
    }
}

/// One mock backend process. The chat handler counts the hit, echoes
/// back which backend served it, and (for `delay_ms > 0`) holds the
/// response open so the request stays in flight.
async fn spawn_backend(tag: &'static str, body: &'static str, delay_ms: u64) -> Mock {
    let hits: Arc<Mutex<HashMap<String, usize>>> = Arc::new(Mutex::new(HashMap::new()));
    let h = hits.clone();
    let app = Router::new()
        .route("/health", get(|| async { axum::Json(serde_json::json!({"status": "ok"})) }))
        .route("/metrics", get(|| async { "rexl3_tokens{kind=\"drafted\"} 5\n" }))
        .route(
            "/v1/chat/completions",
            post(move || {
                let hits = h.clone();
                async move {
                    *hits.lock().unwrap().entry("/v1/chat/completions".to_string()).or_insert(0) += 1;
                    if delay_ms > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    }
                    axum::response::Response::builder().header("content-type", "application/json").body(Body::from(body)).unwrap()
                }
            }),
        )
        .fallback(get(move || async move { format!("fallback:{tag}") }));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Mock { url: format!("http://{addr}"), hits }
}

/// The proxy under test on an ephemeral port.
async fn spawn_router(backends: Vec<Backend>) -> String {
    let app = build_router(App::new(backends));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}")
}

fn client() -> reqwest::Client {
    reqwest::Client::builder().timeout(std::time::Duration::from_secs(30)).build().unwrap()
}

/// Poll `cond` every 10 ms for up to 5 s.
async fn wait_until(what: &str, cond: impl Fn() -> bool) {
    for _ in 0..500 {
        if cond() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("timed out waiting for: {what}");
}

async fn post_chat(router: &str, key: &str) -> reqwest::Response {
    client()
        .post(format!("{router}/v1/chat/completions"))
        .header("x-affinity-key", key)
        .json(&serde_json::json!({"stream": false, "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap()
}

/// 1. least-outstanding-requests distribution across backends: the
/// backend with fewer in-flight requests takes the traffic.
#[tokio::test]
async fn least_outstanding_requests_distribution() {
    let a = spawn_backend("a", "{\"backend\":\"a\"}", 0).await;
    let b = spawn_backend("b", "{\"backend\":\"b\"}", 0).await;
    let backend_a = Backend::new(a.url.clone());
    let backend_b = Backend::new(b.url.clone());
    let router = spawn_router(vec![backend_a.clone(), backend_b.clone()]).await;

    // Phase 1: a carries 5 in-flight requests, b is idle -> 4 fresh
    // requests must all land on b.
    backend_a.inflight.store(5, Ordering::SeqCst);
    for i in 0..4 {
        let resp = post_chat(&router, &format!("loaded-{i}")).await;
        assert_eq!(resp.status(), 200);
    }
    assert_eq!(a.count("/v1/chat/completions"), 0, "loaded backend must be skipped");
    assert_eq!(b.count("/v1/chat/completions"), 4);
    // the reservations were released when the response bodies ended (the bodies here were dropped unread: the router
    // sees the client go away, which also releases)
    wait_until("b's reservations released", || backend_b.inflight.load(Ordering::SeqCst) == 0).await;
    assert_eq!(backend_a.inflight.load(Ordering::SeqCst), 5);

    // Phase 2: the load flips -> traffic goes the other way.
    backend_a.inflight.store(0, Ordering::SeqCst);
    backend_b.inflight.store(7, Ordering::SeqCst);
    for i in 0..3 {
        let resp = post_chat(&router, &format!("flipped-{i}")).await;
        assert_eq!(resp.status(), 200);
    }
    assert_eq!(a.count("/v1/chat/completions"), 3, "idle backend must take over");
    assert_eq!(b.count("/v1/chat/completions"), 4);
    wait_until("a's reservations released", || backend_a.inflight.load(Ordering::SeqCst) == 0).await;
    assert_eq!(backend_b.inflight.load(Ordering::SeqCst), 7);
}

/// 2. prefix-affinity stickiness: the same affinity key maps to the same
/// backend over 10 sequential calls.
#[tokio::test]
async fn prefix_affinity_stickiness() {
    let a = spawn_backend("a", "{\"backend\":\"a\"}", 0).await;
    let b = spawn_backend("b", "{\"backend\":\"b\"}", 0).await;
    let router = spawn_router(vec![Backend::new(a.url.clone()), Backend::new(b.url.clone())]).await;

    for _ in 0..10 {
        let resp = post_chat(&router, "sticky-key").await;
        assert_eq!(resp.status(), 200);
    }
    let (na, nb) = (a.count("/v1/chat/completions"), b.count("/v1/chat/completions"));
    assert_eq!(na + nb, 10);
    assert!(na == 10 || nb == 10, "sticky key must stay on one backend: a={na} b={nb}");
}

/// 3. failover: one backend unreachable → the other serves everything
/// and the aggregated /health still returns 200.
#[tokio::test]
async fn failover_and_aggregate_health() {
    let live = spawn_backend("live", "{\"backend\":\"live\"}", 0).await;
    // A closed port on loopback: connection refused, fails fast.
    let lid = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_port = lid.local_addr().unwrap().port();
    drop(lid);
    let router = spawn_router(vec![Backend::new(format!("http://127.0.0.1:{dead_port}")), Backend::new(live.url.clone())]).await;

    let c = client();
    // Cold affinity key: round 0 may bind to the dead backend (502 with
    // its id in the body, backend marked unhealthy); from then on the
    // key must be re-bound to the live backend and serve everything.
    let mut saw_502 = false;
    let mut ok = 0;
    for _ in 0..6 {
        let resp = post_chat(&router, "failover-key").await;
        if resp.status() == 502 {
            saw_502 = true;
            let body: serde_json::Value = resp.json().await.unwrap();
            assert!(body["backend"].is_string(), "502 body must carry the backend id: {body}");
        } else {
            assert_eq!(resp.status(), 200);
            ok += 1;
        }
    }
    assert!(ok >= 5, "after the failure the live backend serves all: ok={ok}");
    if saw_502 {
        // the one 502 (if any) was the initial dead pick
        assert_eq!(ok, 5);
    } else {
        assert_eq!(ok, 6);
    }
    assert_eq!(live.count("/v1/chat/completions"), ok);

    // Aggregated health is still 200 with one backend up.
    let resp = c.get(format!("{router}/health")).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let doc: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(doc["any_up"], serde_json::json!(true));
}

/// 4. SSE passthrough: 3 streamed chunks arrive byte-exact and in order,
/// unbuffered (chunks observable separately before the stream ends).
#[tokio::test]
async fn sse_passthrough_byte_exact() {
    const CHUNKS: [&str; 3] = ["data: one\n\n", "data: two\n\n", "data: [DONE]\n\n"];
    let expected: String = CHUNKS.concat();

    let app = Router::new().route(
        "/v1/chat/completions",
        post(|| async {
            let stream =
                futures_util::stream::iter(CHUNKS.iter().map(|c| Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(*c))));
            axum::response::Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from_stream(stream))
                .unwrap()
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let backend_addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let router = spawn_router(vec![Backend::new(format!("http://{backend_addr}"))]).await;

    let resp = client()
        .post(format!("{router}/v1/chat/completions"))
        .header("x-affinity-key", "sse")
        .json(&serde_json::json!({"stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(ct.starts_with("text/event-stream"), "content-type: {ct}");

    use futures_util::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut received: Vec<u8> = Vec::new();
    let mut chunks_seen = 0;
    while let Some(item) = stream.next().await {
        let bytes = item.unwrap();
        received.extend_from_slice(&bytes);
        chunks_seen += 1;
    }
    assert_eq!(received, expected.as_bytes(), "byte-exact SSE passthrough");
    assert!(chunks_seen >= 3, "chunks arrived separately, not buffered: {chunks_seen}");
}

/// 5. end-to-end: a forwarded chat completion's response bytes equal the
/// direct-to-backend response bytes.
#[tokio::test]
async fn forwarded_equals_direct() {
    let body = "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}";
    let backend = spawn_backend("x", body, 0).await;
    let router = spawn_router(vec![Backend::new(backend.url.clone())]).await;

    let c = client();
    // Use reqwest for both legs so headers/encoding match exactly.
    let payload = serde_json::json!({"stream": false, "messages": [{"role":"user","content":"hi"}]});
    let direct = c.post(format!("{}/v1/chat/completions", backend.url)).json(&payload).send().await.unwrap();
    let direct_status = direct.status();
    let direct_ct = direct.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let direct_bytes = direct.bytes().await.unwrap();

    let forwarded = c.post(format!("{router}/v1/chat/completions")).json(&payload).send().await.unwrap();
    let forwarded_status = forwarded.status();
    let forwarded_ct = forwarded.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    let forwarded_bytes = forwarded.bytes().await.unwrap();

    assert_eq!(direct_status, forwarded_status);
    assert_eq!(direct_ct, forwarded_ct);
    assert_eq!(direct_bytes, forwarded_bytes, "forwarded response bytes must equal direct bytes");
}
