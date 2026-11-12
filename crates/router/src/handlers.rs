use crate::app::{log_health, unix_ms, Affinity, App};
use crate::backend::{Backend, ErrKind, Reservation};
use crate::keying::{self, Source};
use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get};
use axum::Router;
use futures_util::{Stream, StreamExt};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

/// Header a client may send to pin the affinity key explicitly; otherwise
/// the hash of the request's cacheable prefix (system prompt + first user
/// message) is used.
const AFFINITY_HEADER: &str = "x-affinity-key";
/// Names the backend that served a response.
const BACKEND_HEADER: &str = "x-rexl3-backend";
/// Headers never copied from the backend response to the client.
const HOP_HEADERS: &[&str] =
    &["content-length", "transfer-encoding", "connection", "keep-alive", "te", "trailer", "upgrade", "proxy-authenticate"];
/// Request headers never copied to the backend (`content-length` is added back when the body is streamed unbuffered).
const HOP_REQUEST_HEADERS: &[&str] = &[
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "expect",
    "x-request-id",
];
/// A 503 body is buffered (to classify it) only when it is announced and at most this long.
const MAX_503_BODY: u64 = 64 * 1024;

pub fn build_router(app: App) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/health/ready", get(health))
        .route("/health/live", get(live))
        .route("/metrics", get(metrics))
        // the admin API lives on its own listener (`--admin-bind`); the public port never serves or forwards it
        .route("/admin", any(admin_not_here))
        .route("/admin/{*rest}", any(admin_not_here))
        .fallback(forward)
        .with_state(app)
}

async fn admin_not_here() -> Response {
    let body = serde_json::json!({ "error": "the admin API is not served on this port" });
    (StatusCode::NOT_FOUND, axum::Json(body)).into_response()
}

/// The router process is up (says nothing about backends).
async fn live() -> Response {
    axum::Json(serde_json::json!({ "status": "ok" })).into_response()
}

/// One backend as the router reports it (aggregate `/health`, admin API).
pub fn backend_json(b: &Backend) -> serde_json::Value {
    let (last, fails, oks) = b.probe_snapshot();
    serde_json::json!({
        "backend": b.id(),
        "url": b.url,
        // legacy fields: the HTTP status and body of the last probe
        "status": last.status,
        "body": last.body,
        "up": b.is_healthy(),
        "admin": b.admin().name(),
        "routable": b.routable(),
        "drained": b.drained(),
        "inflight": b.inflight.load(Ordering::Relaxed),
        "consecutive_failures": fails,
        "consecutive_successes": oks,
        "last_probe": {
            "status": last.status,
            "error": last.error,
            "age_ms": last.at.map(|t| t.elapsed().as_millis() as u64),
        },
    })
}

/// Aggregate health: 200 iff at least one backend can take a request (healthy and not drained/disabled). The state is
/// the prober's, not a fresh fan-out per call.
async fn health(axum::extract::State(app): axum::extract::State<App>) -> Response {
    let backends = app.backends();
    let routable = backends.iter().filter(|b| b.routable()).count();
    let parts: Vec<serde_json::Value> = backends.iter().map(backend_json).collect();
    let any_up = routable > 0;
    let doc = serde_json::json!({ "backends": parts, "any_up": any_up, "routable": routable, "total": backends.len() });
    let code = if any_up { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    (code, axum::Json(doc)).into_response()
}

fn esc(v: &str) -> String {
    v.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

fn series(out: &mut String, name: &str, kind: &str, help: &str) {
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n"));
}

/// The router's own series (`rexl3_router_*`).
pub fn render_own_metrics(app: &App) -> String {
    let m = &app.metrics;
    let backends = app.backends();
    let mut o = String::new();
    let c = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);

    series(&mut o, "rexl3_router_requests_total", "counter", "Requests accepted by the proxy path.");
    o.push_str(&format!("rexl3_router_requests_total {}\n", c(&m.requests)));
    series(&mut o, "rexl3_router_no_backend_total", "counter", "Requests refused because no backend was routable.");
    o.push_str(&format!("rexl3_router_no_backend_total {}\n", c(&m.no_backend)));
    series(&mut o, "rexl3_router_failovers_total", "counter", "Requests re-sent to another backend before any response byte.");
    o.push_str(&format!("rexl3_router_failovers_total {}\n", c(&m.failovers)));
    series(&mut o, "rexl3_router_client_disconnects_total", "counter", "Clients that left before the response body ended.");
    o.push_str(&format!("rexl3_router_client_disconnects_total {}\n", c(&m.client_disconnects)));
    series(
        &mut o,
        "rexl3_router_body_over_replay_cap_total",
        "counter",
        "Requests whose body exceeded the replay cap (streamed through, never retried).",
    );
    o.push_str(&format!("rexl3_router_body_over_replay_cap_total {}\n", c(&m.body_overflow)));

    series(&mut o, "rexl3_router_affinity_requests_total", "counter", "Selections by affinity outcome.");
    for (label, v) in [
        ("hit", c(&m.affinity_hit)),
        ("miss", c(&m.affinity_miss)),
        ("overflow", c(&m.affinity_overflow)),
        ("none", c(&m.affinity_none)),
    ] {
        o.push_str(&format!("rexl3_router_affinity_requests_total{{result=\"{label}\"}} {v}\n"));
    }
    series(&mut o, "rexl3_router_affinity_key_source_total", "counter", "Affinity keys by origin.");
    o.push_str(&format!("rexl3_router_affinity_key_source_total{{source=\"header\"}} {}\n", c(&m.key_header)));
    o.push_str(&format!("rexl3_router_affinity_key_source_total{{source=\"prompt\"}} {}\n", c(&m.key_prompt)));
    let (entries, ev_cap, ev_ttl) = app.affinity_stats();
    series(&mut o, "rexl3_router_affinity_entries", "gauge", "Bindings currently held.");
    o.push_str(&format!("rexl3_router_affinity_entries {entries}\n"));
    series(&mut o, "rexl3_router_affinity_evictions_total", "counter", "Bindings dropped, by reason.");
    o.push_str(&format!("rexl3_router_affinity_evictions_total{{reason=\"capacity\"}} {ev_cap}\n"));
    o.push_str(&format!("rexl3_router_affinity_evictions_total{{reason=\"ttl\"}} {ev_ttl}\n"));

    series(&mut o, "rexl3_router_backends", "gauge", "Backends known to the router.");
    o.push_str(&format!("rexl3_router_backends {}\n", backends.len()));
    let per = |o: &mut String, name: &str, kind: &str, help: &str, f: &dyn Fn(&Backend) -> String| {
        series(o, name, kind, help);
        for b in &backends {
            o.push_str(&format!("{name}{{backend=\"{}\"}} {}\n", esc(&b.id()), f(b)));
        }
    };
    per(&mut o, "rexl3_router_backend_up", "gauge", "1 while the backend is healthy.", &|b| u8::from(b.is_healthy()).to_string());
    per(
        &mut o,
        "rexl3_router_backend_routable",
        "gauge",
        "1 while the backend takes new requests (healthy and enabled).",
        &|b| u8::from(b.routable()).to_string(),
    );
    per(&mut o, "rexl3_router_backend_inflight", "gauge", "Requests in flight (held to the end of the response body).", &|b| {
        b.inflight.load(Ordering::Relaxed).to_string()
    });
    per(&mut o, "rexl3_router_backend_requests_total", "counter", "Attempts dispatched to the backend.", &|b| {
        b.stats.requests.load(Ordering::Relaxed).to_string()
    });
    per(&mut o, "rexl3_router_backend_failovers_total", "counter", "Requests that left this backend for another.", &|b| {
        b.stats.failovers_from.load(Ordering::Relaxed).to_string()
    });

    series(&mut o, "rexl3_router_backend_admin_state", "gauge", "1 for the backend's operator state.");
    for b in &backends {
        for st in ["enabled", "draining", "disabled"] {
            o.push_str(&format!(
                "rexl3_router_backend_admin_state{{backend=\"{}\",state=\"{st}\"}} {}\n",
                esc(&b.id()),
                u8::from(b.admin().name() == st)
            ));
        }
    }
    series(&mut o, "rexl3_router_backend_responses_total", "counter", "Backend responses by status class.");
    for b in &backends {
        for (i, class) in ["1xx", "2xx", "3xx", "4xx", "5xx", "other"].iter().enumerate() {
            o.push_str(&format!(
                "rexl3_router_backend_responses_total{{backend=\"{}\",class=\"{class}\"}} {}\n",
                esc(&b.id()),
                b.stats.responses[i].load(Ordering::Relaxed)
            ));
        }
    }
    series(&mut o, "rexl3_router_backend_errors_total", "counter", "Failed attempts by kind.");
    for b in &backends {
        let s = &b.stats;
        for (kind, v) in [
            ("connect", &s.err_connect),
            ("unavailable", &s.err_unavailable),
            ("stream", &s.err_stream),
            ("timeout", &s.err_timeout),
        ] {
            o.push_str(&format!(
                "rexl3_router_backend_errors_total{{backend=\"{}\",kind=\"{kind}\"}} {}\n",
                esc(&b.id()),
                v.load(Ordering::Relaxed)
            ));
        }
    }
    series(&mut o, "rexl3_router_health_transitions_total", "counter", "Health state changes.");
    for b in &backends {
        for (to, v) in [("up", &b.stats.to_up), ("down", &b.stats.to_down)] {
            o.push_str(&format!(
                "rexl3_router_health_transitions_total{{backend=\"{}\",to=\"{to}\"}} {}\n",
                esc(&b.id()),
                v.load(Ordering::Relaxed)
            ));
        }
    }
    series(&mut o, "rexl3_router_probes_total", "counter", "Health probes by result.");
    for b in &backends {
        for (res, v) in [("ok", &b.stats.probes_ok), ("failed", &b.stats.probes_failed)] {
            o.push_str(&format!(
                "rexl3_router_probes_total{{backend=\"{}\",result=\"{res}\"}} {}\n",
                esc(&b.id()),
                v.load(Ordering::Relaxed)
            ));
        }
    }
    o
}

/// `/metrics`: the router's own `rexl3_router_*` series, then every backend's series renamed to
/// `rexl3_router_<name>` and labeled `backend="<id>"` (`?scope=router` for the router's own only). Backends are
/// scraped concurrently under the probe deadline, so a hung backend cannot hang the scrape.
async fn metrics(
    axum::extract::State(app): axum::extract::State<App>,
    axum::extract::Query(q): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    let mut out = render_own_metrics(&app);
    if q.get("scope").map(String::as_str) != Some("router") {
        let backends = app.backends();
        let timeout = app.cfg.probe_timeout;
        let scrapes = backends.iter().map(|b| {
            let client = app.client.clone();
            let url = format!("{}/metrics", b.url);
            async move {
                match client.get(url).timeout(timeout).send().await {
                    Ok(resp) if resp.status().is_success() => resp.text().await.unwrap_or_default(),
                    _ => String::new(),
                }
            }
        });
        let texts = futures_util::future::join_all(scrapes).await;
        for (b, text) in backends.iter().zip(texts) {
            for line in text.lines() {
                if line.starts_with('#') || line.trim().is_empty() {
                    continue;
                }
                let (name_part, rest) = match line.split_once('{') {
                    Some((n, r)) => (n, Some(format!("{{backend=\"{}\",{}", esc(&b.id()), r))),
                    None => (line, None),
                };
                match rest {
                    Some(r) => out.push_str(&format!("rexl3_router_{name_part}{r}\n")),
                    None => match name_part.rsplit_once(' ') {
                        // `name value`: the label set goes between the two
                        Some((n, v)) => out.push_str(&format!("rexl3_router_{n}{{backend=\"{}\"}} {v}\n", esc(&b.id()))),
                        None => out.push_str(&format!("rexl3_router_{name_part}{{backend=\"{}\"}}\n", esc(&b.id()))),
                    },
                }
            }
        }
    }
    ([(HeaderName::from_static("content-type"), HeaderValue::from_static("text/plain; version=0.0.4"))], out).into_response()
}

/// The request body as the proxy sees it: whole when it fits the replay cap, else the cap's worth plus the rest of the
/// stream.
struct Captured {
    head: Bytes,
    rest: Option<axum::body::BodyDataStream>,
}

async fn capture_body(body: Body, cap: usize) -> Result<Captured, axum::Error> {
    let mut stream = body.into_data_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk?);
        if buf.len() > cap {
            return Ok(Captured { head: Bytes::from(buf), rest: Some(stream) });
        }
    }
    Ok(Captured { head: Bytes::from(buf), rest: None })
}

/// How a proxied response ended (the access-log `outcome=`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Ok,
    ClientGone,
    UpstreamError,
    UpstreamStalled,
}

impl Outcome {
    fn name(self) -> &'static str {
        match self {
            Outcome::Ok => "ok",
            Outcome::ClientGone => "client_disconnect",
            Outcome::UpstreamError => "upstream_error",
            Outcome::UpstreamStalled => "upstream_stalled",
        }
    }
}

/// Everything held for the life of a proxied response: the in-flight reservation (so least-outstanding-requests sees a
/// streaming request until its body ends or its client leaves) and the access-log context. Releasing happens on end,
/// error, stall, or `Drop` (the client went away: hyper drops the body, which closes the upstream connection too, and
/// the backend cancels the request).
struct Flight {
    reservation: Reservation,
    backend: Backend,
    app: App,
    request_id: String,
    method: String,
    path: String,
    status: u16,
    attempts: usize,
    started: Instant,
    finished: bool,
}

impl Flight {
    fn finish(&mut self, outcome: Outcome) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.reservation.release();
        match outcome {
            Outcome::Ok => self.backend.passive_success(),
            Outcome::ClientGone => {
                self.app.metrics.client_disconnects.fetch_add(1, Ordering::Relaxed);
            }
            Outcome::UpstreamError | Outcome::UpstreamStalled => {
                self.backend.stats.count_error(ErrKind::Stream);
                if self.backend.passive_failure(false, self.app.cfg.fail_threshold) {
                    log_health(&self.backend, false, "passive: upstream body failed");
                }
            }
        }
        log_request(
            &self.method,
            &self.path,
            &self.backend.id(),
            self.status,
            self.attempts,
            outcome.name(),
            self.started.elapsed(),
            &self.request_id,
        );
    }
}

impl Drop for Flight {
    fn drop(&mut self) {
        self.finish(Outcome::ClientGone);
    }
}

#[allow(clippy::too_many_arguments)]
fn log_request(method: &str, path: &str, backend: &str, status: u16, attempts: usize, outcome: &str, took: Duration, id: &str) {
    let path: String = path.chars().take(96).collect();
    eprintln!(
        "rexl3-router event=proxy ts={} method={method} path={path} backend={backend} status={status} attempts={attempts} outcome={outcome} total_ms={:.1} id={id}",
        unix_ms(),
        took.as_secs_f64() * 1000.0,
    );
}

/// The upstream response body, forwarded chunk by chunk, holding the request's `Flight`.
struct ProxyBody {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    flight: Flight,
    idle: Pin<Box<tokio::time::Sleep>>,
    idle_for: Duration,
    done: bool,
}

impl Stream for ProxyBody {
    type Item = Result<Bytes, std::io::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        if this.done {
            return Poll::Ready(None);
        }
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                this.idle.as_mut().reset(tokio::time::Instant::now() + this.idle_for);
                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => {
                this.done = true;
                this.flight.finish(Outcome::UpstreamError);
                Poll::Ready(Some(Err(std::io::Error::other(e))))
            }
            Poll::Ready(None) => {
                this.done = true;
                this.flight.finish(Outcome::Ok);
                Poll::Ready(None)
            }
            Poll::Pending => {
                if this.idle.as_mut().poll(cx).is_ready() {
                    this.done = true;
                    this.flight.finish(Outcome::UpstreamStalled);
                    return Poll::Ready(Some(Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "upstream stalled"))));
                }
                Poll::Pending
            }
        }
    }
}

/// A failed attempt kept while another backend is tried; it is what the client gets if no other backend takes over.
enum Failure {
    Gateway {
        backend: String,
    },
    /// The backend's own 503, byte for byte.
    Buffered {
        status: StatusCode,
        headers: HeaderMap,
        body: Bytes,
        backend: String,
    },
}

impl Failure {
    fn into_response(self, request_id: &str) -> Response {
        match self {
            Failure::Gateway { backend } => bad_gateway(backend, request_id),
            Failure::Buffered { status, headers, body, backend } => {
                let mut r = Response::builder().status(status);
                for (name, value) in headers.iter() {
                    if HOP_HEADERS.iter().any(|h| name.as_str().eq_ignore_ascii_case(h)) {
                        continue;
                    }
                    r = r.header(name.clone(), value.clone());
                }
                r = r.header(BACKEND_HEADER, backend);
                r.body(Body::from(body)).unwrap_or_else(|_| no_backend(request_id))
            }
        }
    }
}

/// The 503 classes that mean "this backend's engine cannot serve right now" (stopped, failed, loading, shutting down),
/// as `rexl3-serve` words them (`crates/server/src/http.rs`: `service_unavailable`, `server_shutting_down`). Overload
/// (`server_overloaded`) is the backend's own admission decision and is passed through.
fn engine_unavailable(body: &[u8]) -> bool {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else { return false };
    matches!(v["error"]["type"].as_str(), Some("service_unavailable") | Some("server_shutting_down"))
}

fn safe_request_id(v: Option<&HeaderValue>) -> Option<String> {
    let s = v?.to_str().ok()?;
    (!s.is_empty() && s.len() <= 64 && s.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))).then(|| s.to_string())
}

fn new_request_id() -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    format!("rtr-{:x}-{:04x}", unix_ms(), NEXT.fetch_add(1, Ordering::Relaxed) & 0xffff)
}

/// The proxy path: every non-`/health`/`/metrics` request is forwarded
/// to the selected backend and the response is streamed back without
/// buffering.
///
/// The request body is held whole up to `replay_body_cap` (that is what makes retry possible and what the prompt-prefix
/// key is derived from); a larger body is streamed through. A connection failure, or a 503 that says the engine is
/// down, before any byte of the response reaches the client is retried once on another backend; after the first byte
/// nothing is retried (a mid-stream failure is reported to the client as a broken stream, and a transport failure
/// as 502 with the backend id). The backend's in-flight slot is held until the response body ends or the client leaves.
async fn forward(axum::extract::State(app): axum::extract::State<App>, req: axum::extract::Request) -> Response {
    let started = Instant::now();
    let (parts, body) = req.into_parts();
    let request_id = safe_request_id(parts.headers.get("x-request-id")).unwrap_or_else(new_request_id);
    let method_s = parts.method.as_str().to_string();
    let path_s = parts.uri.path().to_string();
    app.metrics.requests.fetch_add(1, Ordering::Relaxed);

    let captured = match capture_body(body, app.cfg.replay_body_cap).await {
        Ok(c) => c,
        Err(_) => {
            let body = serde_json::json!({ "error": "the request body could not be read", "request_id": request_id });
            return (StatusCode::BAD_REQUEST, axum::Json(body)).into_response();
        }
    };
    let replayable = captured.rest.is_none();
    if !replayable {
        app.metrics.body_overflow.fetch_add(1, Ordering::Relaxed);
    }

    // Affinity key: the header wins; else the hash of the cacheable prompt prefix of a request that has a JSON body.
    let (key, source) = match parts.headers.get(AFFINITY_HEADER).filter(|v| !v.as_bytes().is_empty()) {
        Some(v) => (Some(keying::header_key(v.as_bytes())), Some(Source::Header)),
        None if !captured.head.is_empty() => match keying::prefix_key(&captured.head, app.cfg.affinity_prefix_bytes) {
            Some(k) => (Some(k), Some(Source::Prompt)),
            None => (None, None),
        },
        None => (None, None),
    };
    match source {
        Some(Source::Header) => app.metrics.key_header.fetch_add(1, Ordering::Relaxed),
        Some(Source::Prompt) => app.metrics.key_prompt.fetch_add(1, Ordering::Relaxed),
        None => 0,
    };

    let path_and_query = parts.uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/").to_string();
    let method = reqwest::Method::from_bytes(parts.method.as_str().as_bytes()).unwrap_or(reqwest::Method::GET);
    let has_body =
        !captured.head.is_empty() || !replayable || !matches!(parts.method.as_str(), "GET" | "HEAD" | "DELETE" | "OPTIONS");
    let mut streamed: Option<reqwest::Body> = match captured.rest {
        Some(rest) => {
            let head = futures_util::stream::once(futures_util::future::ready(Ok::<_, axum::Error>(captured.head.clone())));
            Some(reqwest::Body::wrap_stream(head.chain(rest)))
        }
        None => None,
    };

    let mut tried: Vec<u64> = Vec::new();
    let mut last_failure: Option<Failure> = None;
    let mut attempt = 0usize;
    loop {
        let Some(picked) = app.select(key, &tried) else {
            app.metrics.no_backend.fetch_add(1, Ordering::Relaxed);
            let resp = match last_failure.take() {
                Some(f) => f.into_response(&request_id),
                None => no_backend(&request_id),
            };
            log_request(&method_s, &path_s, "-", resp.status().as_u16(), attempt, "no_backend", started.elapsed(), &request_id);
            return resp;
        };
        if attempt == 0 {
            let counter = match picked.affinity {
                Affinity::Hit => &app.metrics.affinity_hit,
                Affinity::Miss => &app.metrics.affinity_miss,
                Affinity::Overflow => &app.metrics.affinity_overflow,
                Affinity::None => &app.metrics.affinity_none,
            };
            counter.fetch_add(1, Ordering::Relaxed);
        }
        let backend = picked.backend;
        let mut reservation = picked.reservation;
        backend.stats.requests.fetch_add(1, Ordering::Relaxed);

        let url = format!("{}{}", backend.url, path_and_query);
        let mut rb = app.client.request(method.clone(), url.as_str());
        for (name, value) in parts.headers.iter() {
            let n = name.as_str();
            if HOP_REQUEST_HEADERS.iter().any(|h| n.eq_ignore_ascii_case(h)) {
                continue;
            }
            if replayable && n.eq_ignore_ascii_case("content-length") {
                continue;
            }
            rb = rb.header(name.clone(), value.clone());
        }
        rb = rb.header("x-request-id", request_id.as_str());
        if replayable {
            if has_body {
                rb = rb.body(reqwest::Body::from(captured.head.clone()));
            }
        } else if let Some(b) = streamed.take() {
            rb = rb.body(b);
        }

        let upstream = match rb.send().await {
            Ok(r) => r,
            Err(e) => {
                reservation.release();
                let timed_out = e.is_timeout() && !e.is_connect();
                let id = backend.id();
                if timed_out {
                    backend.stats.count_error(ErrKind::Timeout);
                } else {
                    backend.stats.count_error(ErrKind::Connect);
                    // refused/unreachable: down at once; any other transport failure counts toward the threshold
                    if backend.passive_failure(e.is_connect(), app.cfg.fail_threshold) {
                        log_health(&backend, false, &format!("passive: {}", e.without_url()));
                    }
                }
                tried.push(backend.uid);
                if replayable && !timed_out && attempt < app.cfg.max_failovers {
                    backend.stats.failovers_from.fetch_add(1, Ordering::Relaxed);
                    app.metrics.failovers.fetch_add(1, Ordering::Relaxed);
                    last_failure = Some(Failure::Gateway { backend: id });
                    attempt += 1;
                    continue;
                }
                log_request(&method_s, &path_s, &id, 502, attempt + 1, "backend_unreachable", started.elapsed(), &request_id);
                return bad_gateway(id, &request_id);
            }
        };

        let status = upstream.status().as_u16();
        backend.stats.count_status(status);

        // A 503 that says the engine is down: take the backend out and, before any byte went to the client, try another.
        if status == 503 && upstream.content_length().is_some_and(|n| n <= MAX_503_BODY) {
            let headers = upstream.headers().clone();
            let id = backend.id();
            let bytes = match upstream.bytes().await {
                Ok(b) => b,
                Err(_) => {
                    reservation.release();
                    backend.stats.count_error(ErrKind::Connect);
                    return bad_gateway(id, &request_id);
                }
            };
            reservation.release();
            let buffered =
                Failure::Buffered { status: StatusCode::SERVICE_UNAVAILABLE, headers, body: bytes.clone(), backend: id.clone() };
            if engine_unavailable(&bytes) {
                backend.stats.count_error(ErrKind::Unavailable);
                if backend.passive_failure(true, app.cfg.fail_threshold) {
                    log_health(&backend, false, "passive: 503 engine unavailable");
                }
                tried.push(backend.uid);
                if replayable && attempt < app.cfg.max_failovers {
                    backend.stats.failovers_from.fetch_add(1, Ordering::Relaxed);
                    app.metrics.failovers.fetch_add(1, Ordering::Relaxed);
                    last_failure = Some(buffered);
                    attempt += 1;
                    continue;
                }
            }
            log_request(&method_s, &path_s, &id, 503, attempt + 1, "ok", started.elapsed(), &request_id);
            return buffered.into_response(&request_id);
        }

        // Headers are the point of no return: from here the response is the client's and is never retried.
        let id = backend.id();
        let mut response = Response::builder().status(StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY));
        let mut has_request_id = false;
        for (name, value) in upstream.headers().iter() {
            if HOP_HEADERS.iter().any(|h| name.as_str().eq_ignore_ascii_case(h)) {
                continue;
            }
            has_request_id |= name.as_str().eq_ignore_ascii_case("x-request-id");
            response = response.header(name.clone(), value.clone());
        }
        if !has_request_id {
            response = response.header("x-request-id", request_id.as_str());
        }
        response = response.header(BACKEND_HEADER, id.as_str());
        let flight = Flight {
            reservation,
            backend,
            app: app.clone(),
            request_id: request_id.clone(),
            method: method_s.clone(),
            path: path_s.clone(),
            status,
            attempts: attempt + 1,
            started,
            finished: false,
        };
        let body = ProxyBody {
            inner: Box::pin(upstream.bytes_stream()),
            flight,
            idle: Box::pin(tokio::time::sleep(app.cfg.stream_idle_timeout)),
            idle_for: app.cfg.stream_idle_timeout,
            done: false,
        };
        return response.body(Body::from_stream(body)).unwrap_or_else(|_| bad_gateway(id, &request_id));
    }
}

fn bad_gateway(backend_id: String, request_id: &str) -> Response {
    let body = serde_json::json!({ "error": "backend unreachable", "backend": backend_id, "request_id": request_id });
    (StatusCode::BAD_GATEWAY, axum::Json(body)).into_response()
}

fn no_backend(request_id: &str) -> Response {
    let body = serde_json::json!({ "error": "no healthy backend available", "request_id": request_id });
    (StatusCode::BAD_GATEWAY, axum::Json(body)).into_response()
}
