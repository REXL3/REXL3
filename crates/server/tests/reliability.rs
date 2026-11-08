//! CPU-only lifecycle contracts, including actual Axum/Hyper sockets.
mod common;
use common::*;
use gb_engine::mock::MockOptions;
use serde_json::json;
use std::time::{Duration, Instant};

#[test]
fn liveness_remains_http_responsive_while_readiness_drains() {
    let s = start(Opts::default());
    assert_eq!(http(s.port, "GET", "/health/live", None).0, 200);
    s.handle().drain(Duration::from_secs(1));
    assert_eq!(http(s.port, "GET", "/health/ready", None).0, 503);
    assert_eq!(http(s.port, "GET", "/health/live", None).0, 200);
}

#[test]
fn reading_clients_stream_exactly_the_nonstreamed_tokens() {
    let s = start(Opts { mock: MockOptions { step_ms: 1, ..Default::default() }, ..Default::default() });
    let a = http(s.port, "POST", "/v1/chat/completions", Some(&chat_body(json!({"max_tokens": 256, "return_token_ids": true}))));
    let b = http(
        s.port,
        "POST",
        "/v1/chat/completions",
        Some(&chat_body(json!({"max_tokens": 256, "return_token_ids": true, "stream": true}))),
    );
    assert_eq!((a.0, b.0), (200, 200));
    let a: serde_json::Value = serde_json::from_str(&a.1).unwrap();
    let streamed: Vec<_> =
        sse(&b.1).iter().flat_map(|v| v["choices"][0]["token_ids"].as_array().cloned().unwrap_or_default()).collect();
    assert_eq!(a["choices"][0]["token_ids"].as_array().unwrap(), &streamed);
    assert!(!sse(&b.1).iter().any(|v| v.get("error").is_some()));
}

#[test]
fn tcp_drop_during_prefill_releases_the_sequence() {
    let s = start(Opts { mock: MockOptions { prefill_ms: 10, slice: 32, ..Default::default() }, ..Default::default() });
    let c = Conn::post(s.port, "/v1/completions", &completion_body(&vec![1; 2048], 100, true));
    s.wait(Duration::from_secs(3), "prefill admission", |s| s.metric(|m| &m.requests) == 1);
    drop(c);
    s.wait(Duration::from_secs(3), "TCP-drop cleanup", |s| s.in_flight() == 0 && s.metric(|m| &m.cancelled) == 1);
}

#[test]
fn drain_will_not_start_a_slice_that_cannot_fit() {
    let s = start(Opts { mock: MockOptions { prefill_ms: 200, slice: 32, ..Default::default() }, ..Default::default() });
    let c = Conn::post(s.port, "/v1/completions", &completion_body(&vec![1; 2048], 100, true));
    s.wait(Duration::from_secs(3), "one measured prefill slice", |s| s.metric(|m| &m.prefilling) == 1);
    let t = Instant::now();
    s.handle().drain(Duration::from_millis(50));
    while !s.engine_thread.as_ref().unwrap().is_finished() {
        assert!(t.elapsed() < Duration::from_millis(280), "started another 200ms slice during a 50ms drain");
        std::thread::sleep(Duration::from_millis(2));
    }
    let r = c.finish(Duration::from_secs(2));
    assert!(sse(&r.body).iter().any(|v| v.get("error").is_some()));
    assert!(!sse(&r.body).iter().any(|v| v["choices"][0]["finish_reason"].is_string()));
    assert!(!r.body.contains("[DONE]"), "a stream cut by a drain ends with its error event, not [DONE]: {}", r.body);
}

#[test]
fn deadline_covers_body_wait_and_running_and_queued_work() {
    let s = start(Opts {
        slots: 1,
        request_timeout: Duration::from_millis(900),
        mock: MockOptions { step_ms: 10, ..Default::default() },
        ..Default::default()
    });
    let running = Conn::post(s.port, "/v1/completions", &completion_body(&[1, 2, 3], 3000, true));
    s.wait(Duration::from_secs(2), "first admission", |s| s.metric(|m| &m.requests) == 1);
    let queued = Conn::post(s.port, "/v1/completions", &completion_body(&[4, 5, 6], 3000, true));
    s.wait(Duration::from_secs(2), "second queued", |s| s.metric(|m| &m.queued) == 1);
    for c in [running, queued] {
        let r = c.finish(Duration::from_secs(3));
        let events = sse(&r.body);
        assert!(events.iter().any(|v| v["error"]["type"] == "request_timeout"), "{}", r.body);
        assert!(!events.iter().any(|v| v["choices"][0]["finish_reason"].is_string()));
        assert!(!r.body.contains("[DONE]"), "an expired stream ends with its error event, not [DONE]: {}", r.body);
    }
    s.wait(Duration::from_secs(2), "deadline cleanup", |s| s.in_flight() == 0);
    assert_eq!(s.metric(|m| &m.request_timeouts), 2, "exactly once per expired stream, across engine/decoder/HTTP races");
    use std::io::{Read, Write};
    let mut c = std::net::TcpStream::connect(("127.0.0.1", s.port)).unwrap();
    c.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
    c.write_all(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{").unwrap();
    let mut reply = String::new();
    let _ = c.read_to_string(&mut reply);
    assert!(reply.contains("504 Gateway Timeout") && reply.contains("request_timeout"), "{reply}");
}

#[test]
fn nonstream_deadline_is_a_structured_504() {
    let s = start(Opts {
        request_timeout: Duration::from_millis(700),
        mock: MockOptions { step_ms: 10, ..Default::default() },
        ..Default::default()
    });
    let r =
        http_timeout(s.port, "POST", "/v1/completions", Some(&completion_body(&[1, 2, 3], 3000, false)), Duration::from_secs(3));
    assert_eq!(r.status, 504);
    assert_eq!(r.json()["error"]["type"], "request_timeout");
    s.wait(Duration::from_secs(2), "cancelled 504 cleanup", |s| s.in_flight() == 0);
}

#[test]
fn readiness_uses_a_legitimate_prefill_budget_but_detects_a_decode_wedge() {
    let s = start(Opts {
        heartbeat_stale: Duration::from_millis(30),
        prefill_budget: Duration::from_secs(2),
        mock: MockOptions { prefill_ms: 200, step_ms: 200, ..Default::default() },
        ..Default::default()
    });
    let c = Conn::post(s.port, "/v1/completions", &completion_body(&[1, 2, 3], 100, true));
    s.wait(Duration::from_secs(1), "admitted", |s| s.metric(|m| &m.requests) == 1);
    std::thread::sleep(Duration::from_millis(70));
    assert_eq!(http(s.port, "GET", "/health/ready", None).0, 200, "legitimate prefill is not a wedge");
    std::thread::sleep(Duration::from_millis(210));
    let r = http(s.port, "GET", "/health/ready", None);
    assert_eq!(r.0, 503);
    assert!(r.1.contains("wedged"));
    assert_eq!(http(s.port, "GET", "/health/live", None).0, 200);
    drop(c);
    s.wait(Duration::from_secs(2), "dropped wedge reply cleanup", |s| s.in_flight() == 0);
}

#[test]
fn a_real_tcp_client_that_never_reads_is_bounded_and_cancelled() {
    let s = start(Opts {
        max_model_len: 100_000,
        write_stall: Duration::from_millis(150),
        mock: MockOptions { vocab: 256, step_ms: 1, ..Default::default() },
        ..Default::default()
    });
    // Bound the server's TCP send backlog instead of shrinking an already-connected client's receive window. The latter can leave
    // megabytes of previously accepted data ahead of the error, so an 8 s read can end before reaching the terminal event.
    // Accepted sockets inherit this limit; this is still real Axum/Hyper TCP backpressure, not an unpolled in-process response body.
    let rt = tokio::runtime::Runtime::new().unwrap();
    let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
    socket2::SockRef::from(&listener).set_send_buffer_size(4096).unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = gb_server::http::router(s.state.clone());
    rt.spawn(async move { axum::serve(listener, app).await.unwrap() });
    let mut c = Conn::post(port, "/v1/completions", &completion_body(&[1, 2, 3], 90_000, true));
    s.wait(Duration::from_secs(3), "no-read admission", |s| s.metric(|m| &m.requests) == 1);
    s.wait(Duration::from_secs(40), "no-read backpressure cancels and releases all engine resources", |s| {
        s.in_flight() == 0 && s.metric(|m| &m.cancelled) > 0
    });
    assert!(s.metric(|m| &m.completion_tokens) < 90_000, "not a normal budget finish");
    assert_eq!(s.metric(|m| &m.stream_failures), 1, "terminal failure counted once across every edge");
    let (bytes, _) = wait_for_close(&mut c.sock, Duration::from_secs(8))
        .expect("resumed consumer must receive the end of the response, not stop at a read timeout");
    c.raw.extend(bytes);
    let r = c.finish(Duration::ZERO);
    assert_eq!(r.status, 200, "the request started as a stream");
    let events = sse(&r.body);
    assert!(
        events.last().is_some_and(|v| v.get("error").is_some()),
        "resumed consumer must see terminal error, not silent token loss"
    );
    assert!(!events.iter().any(|v| v["choices"][0]["finish_reason"].is_string()));
    assert!(!r.body.contains("[DONE]"), "a stalled stream must never claim success");
    assert_eq!(s.metric(|m| &m.stream_failures), 1, "reading the terminal event must not count a second failure");
    assert_eq!(http_timeout(port, "GET", "/health", None, Duration::from_secs(3)).status, 200);
}

#[test]
fn preparation_clock_is_not_reset_and_the_blocking_worker_owns_its_lifetime() {
    use gb_server::chat_format::{ChatFormat, RenderedChat};
    use gb_server::template::TemplateError;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    };
    struct BarrierFormat {
        inner: Arc<dyn ChatFormat>,
        entered: Arc<AtomicBool>,
        finished: Arc<AtomicBool>,
        resume: Mutex<std::sync::mpsc::Receiver<()>>,
    }
    impl ChatFormat for BarrierFormat {
        fn id(&self) -> &'static str {
            self.inner.id()
        }
        fn render(
            &self,
            messages: &serde_json::Value,
            kwargs: &serde_json::Map<String, serde_json::Value>,
        ) -> Result<RenderedChat, TemplateError> {
            self.entered.store(true, Ordering::Release);
            self.resume.lock().unwrap().recv().unwrap();
            let result = self.inner.render(messages, kwargs);
            self.finished.store(true, Ordering::Release);
            result
        }
        fn render_bounded(
            &self,
            messages: &serde_json::Value,
            kwargs: &serde_json::Map<String, serde_json::Value>,
            max_bytes: usize,
        ) -> Result<RenderedChat, TemplateError> {
            self.entered.store(true, Ordering::Release);
            self.resume.lock().unwrap().recv().unwrap();
            let result = self.inner.render_bounded(messages, kwargs, max_bytes);
            self.finished.store(true, Ordering::Release);
            result
        }
        fn starts_in_reasoning(&self, prompt: &str) -> bool {
            self.inner.starts_in_reasoning(prompt)
        }
        fn tool_syntax(&self) -> gb_server::tools::ToolSyntax {
            self.inner.tool_syntax()
        }
    }
    let s = start(Opts { request_timeout: Duration::from_millis(200), ..Default::default() });
    let entered = Arc::new(AtomicBool::new(false));
    let finished = Arc::new(AtomicBool::new(false));
    let (release, resume) = std::sync::mpsc::channel();
    let state = Arc::new(gb_server::http::AppState {
        handle: s.state.handle.clone(),
        tok: s.state.tok.clone(),
        format: Arc::new(BarrierFormat {
            inner: s.state.format.clone(),
            entered: entered.clone(),
            finished: finished.clone(),
            resume: Mutex::new(resume),
        }),
        defaults: s.state.defaults.clone(),
        model_name: "mock".into(),
        model_aliases: Vec::new(),
        request_budget: s.state.request_budget.clone(),
        max_model_len: 4096,
        sse_keepalive: Duration::from_secs(15),
        default_max_tokens: 0,
        vision: None,
        default_thinking: Default::default(),
        // the wrapped format's (the barrier format's render blocks, so it is not probed again)
        reasoning_caps: s.state.reasoning_caps.clone(),
    });
    let rt = tokio::runtime::Runtime::new().unwrap();
    let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
    let port = listener.local_addr().unwrap().port();
    rt.spawn(async move {
        axum::serve(listener, gb_server::http::router(state)).await.unwrap();
    });
    let c = Conn::post(port, "/v1/chat/completions", &chat_body(json!({"max_tokens":100})));
    s.wait(Duration::from_secs(1), "barrier-controlled render worker", |_| entered.load(Ordering::Acquire));
    let r = c.finish(Duration::from_secs(2));
    // Release first even when the assertion fails: never strand a blocking fixture worker.
    let retained = !finished.load(Ordering::Acquire);
    release.send(()).unwrap();
    s.wait(Duration::from_secs(1), "preparation worker quiesced", |_| finished.load(Ordering::Acquire));
    assert_eq!(r.status, 504, "{}", r.body);
    assert_eq!(r.json()["error"]["type"], "request_timeout");
    assert!(retained, "dropping the HTTP await did not pretend the worker had stopped");
    assert_eq!(s.metric(|m| &m.requests), 0, "an expired preparation must never submit new engine work");
    assert_eq!(s.metric(|m| &m.request_timeouts), 1);
}

/// Reads `sock` until the server closes it (EOF or a reset); returns the bytes and when it happened, or None after `limit`.
fn wait_for_close(sock: &mut std::net::TcpStream, limit: Duration) -> Option<(Vec<u8>, Duration)> {
    use std::io::Read;
    let t0 = Instant::now();
    let mut got = Vec::new();
    let mut buf = [0u8; 4096];
    sock.set_read_timeout(Some(Duration::from_millis(100))).unwrap();
    while t0.elapsed() < limit {
        match sock.read(&mut buf) {
            Ok(0) => return Some((got, t0.elapsed())),
            Ok(n) => got.extend_from_slice(&buf[..n]),
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
            Err(_) => return Some((got, t0.elapsed())),
        }
    }
    None
}

/// The accept loop of `rexl3-serve` (`gb_server::listen::serve`) with a 1 s header-read timeout: a connection that sends half a
/// request line, or nothing, is closed within the timeout; so is an idle keep-alive connection after its response; a response that
/// takes longer than the timeout (a stream) is not cut; and the server keeps serving everyone else throughout. (Mutation: with
/// `axum::serve`, or without the timer, the half-line connection stays open past the limit.)
#[test]
fn a_half_sent_request_line_is_closed_within_the_header_read_timeout_and_the_server_keeps_serving() {
    use std::io::Write;
    let s = start(Opts { mock: MockOptions { step_ms: 25, ..Default::default() }, ..Default::default() });
    let rt = tokio::runtime::Runtime::new().unwrap();
    let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = gb_server::http::router(s.state.clone());
    rt.spawn(gb_server::listen::serve(listener, app, Duration::from_secs(1), std::future::pending()));
    let limit = Duration::from_millis(3500);

    // half a request line, and a connection that sends nothing at all
    let mut half = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    half.write_all(b"GET /health/li").unwrap();
    let mut silent = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    // meanwhile the server answers others
    assert_eq!(http(port, "GET", "/health/live", None).0, 200);
    let (_, after) = wait_for_close(&mut half, limit).expect("the half request line was not closed within the timeout");
    assert!(after >= Duration::from_millis(500), "closed after {after:?}: before the timeout could have fired");
    let (bytes, _) = wait_for_close(&mut silent, limit).expect("a connection that sent nothing was not closed");
    assert!(bytes.is_empty() || String::from_utf8_lossy(&bytes).starts_with("HTTP/1.1 408"), "{bytes:?}");

    // an idle keep-alive connection is closed once it has waited the timeout for its next request
    let mut ka = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    ka.write_all(b"GET /health/live HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
    let (bytes, _) = wait_for_close(&mut ka, limit).expect("an idle keep-alive connection was not closed");
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.starts_with("HTTP/1.1 200"), "the request on it was answered first: {text}");

    // a stream that lasts longer than the timeout is not bounded by it (40 tokens at 25 ms a step)
    let r = http(port, "POST", "/v1/chat/completions", Some(&chat_body(json!({"max_tokens": 40, "stream": true}))));
    assert_eq!(r.0, 200);
    let events = sse(&r.1);
    assert!(!events.iter().any(|v| v.get("error").is_some()), "{}", r.1);
    assert!(r.1.contains("[DONE]"), "the stream finished normally: {}", r.1);
    assert_eq!(http(port, "GET", "/health/live", None).0, 200);
}

/// The same through the real binary: `--header-read-timeout 1` closes a half-sent request line within the timeout and the server
/// keeps serving; the flag is range-checked like the other lifecycle budgets (0 refuses the boot with exit 2).
#[test]
fn the_binary_closes_a_half_sent_request_line_and_keeps_serving() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};
    let bin = env!("CARGO_BIN_EXE_rexl3-serve");
    let dir = unstoppable_dir();
    let refused = Command::new(bin)
        .arg("--model")
        .arg(&dir)
        .args(["--backend", "mock", "--port", "0", "--header-read-timeout", "0"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert_eq!(refused.code(), Some(2), "a zero header read timeout is a usage error");
    let mut child = Command::new(bin)
        .arg("--model")
        .arg(&dir)
        .args(["--backend", "mock", "--port", "0", "--max-model-len", "4096", "--header-read-timeout", "1"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    let err = child.stderr.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(err).lines().map_while(Result::ok) {
            if let Some(p) =
                line.split("addr=http://127.0.0.1:").nth(1).and_then(|r| r.split_whitespace().next()?.parse::<u16>().ok())
            {
                let _ = tx.send(p);
            }
        }
    });
    let port = rx.recv_timeout(Duration::from_secs(20));
    let result = std::panic::catch_unwind(|| {
        let port = port.expect("the server never started listening");
        let mut half = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
        half.write_all(b"POST /v1/chat/compl").unwrap();
        assert_eq!(http(port, "GET", "/health/live", None).0, 200);
        let (_, after) = wait_for_close(&mut half, Duration::from_millis(3500)).expect("the half request line was not closed");
        assert!(after >= Duration::from_millis(500), "closed after {after:?}");
        assert_eq!(http(port, "GET", "/health/live", None).0, 200, "the server keeps serving");
    });
    let _ = child.kill();
    let _ = child.wait();
    result.unwrap();
}
