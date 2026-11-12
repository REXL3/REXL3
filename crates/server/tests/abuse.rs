//! The HTTP layer under abuse, on the mock backend (CPU only, any machine): oversized and hostile bodies, header and framing abuse, clients that
//! stall, flood or vanish at every stage of a request, and what the engine's books and `/metrics` say afterwards. Every test starts its own
//! in-process server (`common::start`) and ends by proving the server still answers and holds nothing: `in_flight() == 0`, and a request that needs
//! the whole KV pool is admitted (a leaked page or slot would starve it).
//!
//! These are invariants, so they assert statuses, gauges and counters; bounded waits detect stalls, and the async-worker test synchronizes on a
//! blocked tokenizer instead of requiring a minimum preparation time. Sizes that need memory or minutes
//! (60 MiB bodies, 40 s slow-header waits, RSS, thousands of connections) are in `tools/http_abuse.py`, which drives a real process and records
//! the numbers (kept in the maintainers' working repository, summarised in `docs/PRODUCTION_READINESS.md`).
//!
//! The tests run one at a time (`LOCK`): several count the process's file descriptors, and a flood in one would show up in another.
//! Shapes of refusals that axum produces before our handlers run (a body that is not JSON, a missing content type, an unknown route) are asserted
//! by status only; their body is the business of `api_errors.rs`.

mod common;

use common::{chat_body, completion_body, http_timeout, start, Conn, Opts, Reply, Server};
use gb_engine::mock::MockOptions;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

static LOCK: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const SECS: fn(u64) -> Duration = Duration::from_secs;

/// Exact bytes to the server, the whole reply back (until the server closes or `read_timeout` passes). Write errors are ignored: a server that
/// answers and closes before the body is out is behaving.
fn raw(port: u16, bytes: &[u8], read_timeout: Duration) -> Vec<u8> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(read_timeout)).unwrap();
    let _ = s.write_all(bytes);
    let mut out = Vec::new();
    let _ = s.read_to_end(&mut out);
    out
}

fn status_of(reply: &[u8]) -> Option<u16> {
    std::str::from_utf8(reply.get(..12)?).ok()?.split_whitespace().nth(1)?.parse().ok()
}

/// A POST of exact `body` bytes with the given content type.
fn post_raw(port: u16, path: &str, content_type: &str, body: &[u8]) -> Vec<u8> {
    let mut req = format!(
        "POST {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    req.extend_from_slice(body);
    raw(port, &req, SECS(60))
}

fn post_json(port: u16, path: &str, body: &Value) -> Reply {
    http_timeout(port, "POST", path, Some(body), SECS(60))
}

fn chat(port: u16, extra: Value) -> Reply {
    post_json(port, "/v1/chat/completions", &chat_body(extra))
}

fn healthy(port: u16) -> bool {
    status_of(&raw(port, b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n", SECS(5))) == Some(200)
}

/// The server-side ends of the connections to the server on `port` that are still open: sockets in the ESTABLISHED state whose local port is `port` in
/// `/proc/net/tcp` and `tcp6`. Counting the process's descriptors instead also counts every other test's server, which closes its connections on its own
/// schedule (a run of the whole workspace suite timed out waiting for that number to fall).
#[cfg(not(target_os = "macos"))]
fn established_on(port: u16) -> usize {
    let local = format!(":{port:04X}");
    ["/proc/net/tcp", "/proc/net/tcp6"]
        .iter()
        .filter_map(|f| std::fs::read_to_string(f).ok())
        .map(|t| {
            t.lines()
                .skip(1)
                .filter(|l| {
                    let mut c = l.split_whitespace();
                    let (addr, state) = (c.nth(1).unwrap_or(""), c.nth(1).unwrap_or(""));
                    addr.ends_with(&local) && state == "01"
                })
                .count()
        })
        .sum()
}

/// macOS has no `/proc/net/tcp`: the same count from `netstat -an -p tcp`, whose rows are `proto recv-q send-q local foreign state` with the
/// port as the last dot-separated field of an address (`127.0.0.1.8080`, `*.8080`, `::1.8080`). Without this the `== 0` waits below would
/// pass without looking and the `>= 52` wait would never pass.
#[cfg(target_os = "macos")]
fn established_on(port: u16) -> usize {
    let out = std::process::Command::new("/usr/sbin/netstat").args(["-an", "-p", "tcp"]).output().expect("netstat runs");
    let local = format!(".{port}");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter(|l| {
            let c: Vec<&str> = l.split_whitespace().collect();
            c.len() >= 6 && c[0].starts_with("tcp") && c[3].ends_with(&local) && c[5] == "ESTABLISHED"
        })
        .count()
}

fn small_pool() -> Opts {
    // a pool of one full window (2,048 tokens = 2 MiB) beside three slots of state (3 MiB): capacity 5 MiB, one whole-window request costs 3 MiB, so
    // two do not fit, and the whole-pool request below is admitted only if the abuse before it left (almost) nothing behind
    Opts {
        slots: 3,
        max_model_len: 2048,
        kv_tokens: 2048,
        mock: MockOptions { vocab: 4096, ..Default::default() },
        ..Opts::default()
    }
}

/// What every test ends with: the server answers, nothing is held, and a request that needs the whole KV pool (window 2,048 tokens) is
/// admitted and runs to the window: a page or a slot that a vanished client left behind would starve it.
fn assert_clean(s: &Server, what: &str) {
    s.wait(SECS(30), &format!("the engine to give everything back after {what}"), |s| s.in_flight() == 0);
    assert!(healthy(s.port), "{what}: /health");
    // 2,000 prompt tokens and a reply clamped to the room left (48): the cost is the whole window, the work is a prefill, fast whatever the step time
    let r = post_json(s.port, "/v1/completions", &completion_body(&vec![5; 2000], 100_000, false));
    assert_eq!(r.status, 200, "{what}: the whole-pool request: {}", r.body);
    let v = r.json();
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{what}: {v}");
    let (p, c) = (v["usage"]["prompt_tokens"].as_u64().unwrap(), v["usage"]["completion_tokens"].as_u64().unwrap());
    assert_eq!(p + c, 2048, "{what}: it should have run to the window");
    s.wait(SECS(10), "the whole-pool request to be released", |s| s.in_flight() == 0);
}

#[test]
fn a_body_over_the_limit_is_refused_and_a_lying_content_length_leaves_nothing_behind() {
    let _g = serial();
    let s = start(small_pool());
    // 65 MiB: over the 64 MiB limit
    let big = vec![b' '; 65 << 20];
    let r = post_raw(s.port, "/v1/chat/completions", "application/json", &big);
    assert_eq!(status_of(&r), Some(413), "{}", String::from_utf8_lossy(&r[..r.len().min(200)]));
    drop(big);
    // a Content-Length that promises more than arrives, then the client leaves
    for promised in [1000, 100 << 20] {
        let mut c = TcpStream::connect(("127.0.0.1", s.port)).unwrap();
        let _ = c.write_all(
            format!("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {promised}\r\n\r\n{{\"messages\"")
                .as_bytes(),
        );
        std::thread::sleep(Duration::from_millis(100));
        drop(c);
    }
    assert_clean(&s, "oversized and truncated bodies");
}

#[test]
fn deeply_nested_json_is_bounded_before_allocation_and_never_a_stack_overflow() {
    let _g = serial();
    let s = start(small_pool());
    for depth in [1usize, 64, 127, 128, 129, 500, 10_000, 1_000_000] {
        let nest = format!("{}{}", "[".repeat(depth), "]".repeat(depth));
        let body = format!(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"max_tokens\":2,\"metadata\":{{\"nest\":{nest}}}}}"
        );
        let st = status_of(&post_raw(s.port, "/v1/chat/completions", "application/json", body.as_bytes()));
        match depth {
            1..=64 => assert_eq!(st, Some(200), "depth {depth}"),
            200.. => assert_eq!(st, Some(413), "depth {depth}"),
            _ => assert!(matches!(st, Some(200) | Some(400) | Some(413)), "depth {depth}: {st:?}"),
        }
        // the same nesting where a value is parsed, not skipped: in the messages themselves
        let body = format!("{{\"messages\":{nest}}}");
        let st = status_of(&post_raw(s.port, "/v1/chat/completions", "application/json", body.as_bytes()));
        assert!(matches!(st, Some(400) | Some(413)), "messages nested {depth} deep: {st:?}");
        assert!(healthy(s.port), "after depth {depth}");
    }
    assert_clean(&s, "deeply nested JSON");
}

#[test]
fn invalid_utf8_nul_bytes_surrogates_and_a_byte_order_mark_are_answered_not_fatal() {
    let _g = serial();
    let s = start(small_pool());
    let ok = |content: &str| {
        format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"{content}\"}}],\"max_tokens\":2}}").into_bytes()
    };
    let cases: Vec<(&str, Vec<u8>, &[u16])> = vec![
        ("invalid UTF-8 in a string", b"{\"messages\":[{\"role\":\"user\",\"content\":\"\xff\xfe\"}]}".to_vec(), &[400]),
        ("truncated multibyte sequence", b"{\"messages\":[{\"role\":\"user\",\"content\":\"\xe2\x82\"}]}".to_vec(), &[400]),
        ("overlong encoding", b"{\"messages\":[{\"role\":\"user\",\"content\":\"\xc0\xaf\"}]}".to_vec(), &[400]),
        ("a raw NUL inside a string", b"{\"messages\":[{\"role\":\"user\",\"content\":\"a\0b\"}]}".to_vec(), &[400]),
        ("a lone high surrogate escape", ok("\\ud800"), &[400]),
        ("a lone low surrogate escape", ok("\\udc00"), &[400]),
        ("a surrogate pair escape", ok("\\ud83d\\ude00"), &[200]),
        ("an escaped NUL", ok("a\\u0000b"), &[200, 400]),
        ("a byte order mark first", [&b"\xef\xbb\xbf"[..], &ok("hi")].concat(), &[400]),
        ("an empty body", Vec::new(), &[400]),
        ("whitespace only", b"   \n  ".to_vec(), &[400]),
        ("trailing garbage after the object", [&ok("hi")[..], b" junk"].concat(), &[400]),
        ("two objects", [&ok("hi")[..], &ok("hi")].concat(), &[400]),
        (
            "a very long unbroken string key",
            format!("{{\"{}\":1,\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\"max_tokens\":2}}", "k".repeat(1 << 20))
                .into_bytes(),
            &[400], // unknown fields are no longer silently accepted
        ),
    ];
    for (name, body, want) in cases {
        let st = status_of(&post_raw(s.port, "/v1/chat/completions", "application/json", &body)).unwrap_or(0);
        assert!(want.contains(&st), "{name}: answered {st}, wanted one of {want:?}");
        // the rendering endpoint and plain completions read the same bytes
        let st = status_of(&post_raw(s.port, "/v1/chat/completions/render", "application/json", &body)).unwrap_or(0);
        assert!((200..500).contains(&st), "{name} on /render: {st}");
    }
    assert_clean(&s, "hostile bytes");
}

#[test]
fn a_connection_that_stalls_holds_nothing_but_itself() {
    let _g = serial();
    let s = start(small_pool());
    // a body that never completes, a request line that never completes, a connection that never says anything
    let mut stalled: Vec<TcpStream> = Vec::new();
    let mut c = TcpStream::connect(("127.0.0.1", s.port)).unwrap();
    c.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: 1000\r\n\r\n{\"messages\":").unwrap();
    stalled.push(c);
    let mut c = TcpStream::connect(("127.0.0.1", s.port)).unwrap();
    c.write_all(b"POST /v1/chat/comp").unwrap();
    stalled.push(c);
    for _ in 0..50 {
        stalled.push(TcpStream::connect(("127.0.0.1", s.port)).unwrap());
    }
    s.wait(SECS(10), "the server to hold all 52 connections", |s| established_on(s.port) >= 52);
    // everything else is served meanwhile, and the engine has seen none of them
    let t = Instant::now();
    assert!(healthy(s.port));
    let r = chat(s.port, json!({"max_tokens": 4}));
    assert_eq!(r.status, 200, "{}", r.body);
    assert!(t.elapsed() < SECS(20), "two requests took {:?} with 52 stalled connections open", t.elapsed());
    assert_eq!(s.in_flight(), 0);
    assert_eq!(s.metric(|m| &m.requests), 1, "only the one real request reached the engine");
    drop(stalled);
    s.wait(SECS(60), "the stalled connections to be released", |s| established_on(s.port) == 0);
    assert_clean(&s, "stalled connections");
}

#[test]
fn thousands_of_tiny_concurrent_requests_each_get_a_clean_answer_and_the_books_balance() {
    let _g = serial();
    let s = start(Opts {
        slots: 8,
        queue_depth: 512,
        max_model_len: 2048,
        kv_tokens: 2048 * 8,
        mock: MockOptions { vocab: 4096, step_ms: 1, ..Default::default() },
        ..Opts::default()
    });
    let total = 3000usize;
    let next = Arc::new(AtomicUsize::new(0));
    let seen: Arc<Mutex<BTreeMap<u16, usize>>> = Default::default();
    let bad_retry_after = Arc::new(AtomicUsize::new(0));
    let t0 = Instant::now();
    let port = s.port;
    let threads: Vec<_> = (0..256)
        .map(|_| {
            let (next, seen, bad) = (next.clone(), seen.clone(), bad_retry_after.clone());
            std::thread::spawn(move || loop {
                let i = next.fetch_add(1, Relaxed);
                if i >= total {
                    return;
                }
                let body = chat_body(json!({"max_tokens": 3, "messages": [{"role": "user", "content": format!("tiny {i}")}]}));
                let r = http_timeout(port, "POST", "/v1/chat/completions", Some(&body), SECS(120));
                if r.status == 429 && r.header("retry-after").is_none() {
                    bad.fetch_add(1, Relaxed);
                }
                *seen.lock().unwrap().entry(r.status).or_default() += 1;
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    let seen = seen.lock().unwrap().clone();
    println!("{total} requests in {:.1} s from 256 clients: {seen:?}", t0.elapsed().as_secs_f64());
    assert_eq!(seen.values().sum::<usize>(), total);
    // Active preparation admission is deliberately fail-fast: safe 429s replace the old unlimited waiting threads.
    assert!(seen.keys().all(|st| matches!(st, 200 | 429)), "statuses: {seen:?}");
    let completed = seen.get(&200).copied().unwrap_or(0);
    assert!(completed > 0, "at least some legitimate requests must complete");
    assert_eq!(bad_retry_after.load(Relaxed), 0);
    // the counters add up: each request was admitted once and none was refused or failed
    s.wait(SECS(30), "the engine to settle", |s| s.in_flight() == 0);
    assert_eq!(s.metric(|m| &m.requests) as usize, completed);
    assert_eq!(s.metric(|m| &m.queue_full), 0);
    assert_eq!(s.metric(|m| &m.rejected), 0);
    assert_eq!(s.metric(|m| &m.failed_sequences), 0);
    s.wait(SECS(60), "the sockets to close", |s| established_on(s.port) == 0);
    assert_clean(&s, "3000 tiny requests");
}

#[test]
fn streams_dropped_while_queued_prefilling_and_decoding_give_back_their_slots_and_pages() {
    let _g = serial();
    // two slots, so most of the 60 streams queue; a prefill that takes a while (slices of 64 tokens, 15 ms each) so some are dropped inside it
    let s = start(Opts {
        slots: 2,
        queue_depth: 64,
        max_model_len: 2048,
        kv_tokens: 2048,
        mock: MockOptions { vocab: 4096, step_ms: 4, prefill_ms: 15, slice: 64, ..Default::default() },
        ..Opts::default()
    });
    let long_prompt = "word ".repeat(300);
    let mut conns = Vec::new();
    for i in 0..60 {
        let stream = i % 3 != 0; // two of three stream, the rest are plain requests that are abandoned mid-generation
        let content = if i % 2 == 0 { long_prompt.clone() } else { "short".to_string() };
        let body = chat_body(json!({"stream": stream, "max_tokens": 1000, "messages": [{"role": "user", "content": content}]}));
        conns.push(Conn::post(s.port, "/v1/chat/completions", &body));
    }
    // drop them at different moments: some at once, some after a few events, some after a delay
    let mut seed = 0x1234_5678u64;
    for (i, mut c) in conns.into_iter().enumerate() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        match seed % 4 {
            0 => {}
            1 => {
                let _ = c.read_until(Duration::from_millis(50 + seed % 200), |t| t.matches("data:").count() > (i % 4));
            }
            2 => std::thread::sleep(Duration::from_millis(seed % 40)),
            _ => {
                let _ = c.read_until(Duration::from_millis(400), |t| t.matches("data:").count() >= 6);
            }
        }
        drop(c);
    }
    s.wait(SECS(30), "every dropped request to be released", |s| s.in_flight() == 0);
    // a client that leaves before the server has read its request is never a request, and one that leaves after its reply finished is not cancelled: the count
    // only shows that cancelling happened (24 loaded runs: 29 to 60 of 60); the books are what the whole-pool request below proves
    // Preparation refusals never become engine sequences; cancellation is measured only for admitted work.
    assert!(s.metric(|m| &m.requests) > 0);
    assert!(s.metric(|m| &m.cancelled) > 0, "cancelled {} admitted requests", s.metric(|m| &m.cancelled));
    assert_eq!(s.metric(|m| &m.failed_sequences), 0, "a disconnect is not a failure");
    assert_clean(&s, "60 dropped requests");
}

#[test]
fn every_client_dropped_in_the_middle_of_its_prefill_gives_its_slot_back() {
    let _g = serial();
    // two slots, a prefill of about 1.4 s (24 slices of 60 ms: 1,500 tokens of the byte-level fixture tokenizer): each client is dropped once the engine says it is prefilling, so every drop is a cancel in the
    // prefill, whatever the timing. One more drop than there are slots: a slot that a cancel kept would show as the last request never being admitted.
    let s = start(Opts {
        slots: 2,
        max_model_len: 2048,
        kv_tokens: 2048,
        mock: MockOptions { vocab: 4096, prefill_ms: 60, slice: 64, ..Default::default() },
        ..Opts::default()
    });
    let prompt = "word ".repeat(300);
    for round in 0..3 {
        let body =
            chat_body(json!({"stream": true, "max_tokens": 50, "messages": [{"role": "user", "content": prompt.clone()}]}));
        let c = Conn::post(s.port, "/v1/chat/completions", &body);
        s.wait(SECS(5), &format!("round {round}: the request to be admitted and prefilling"), |s| {
            s.metric(|m| &m.prefilling) >= 1
        });
        drop(c);
        s.wait(SECS(10), &format!("round {round}: the cancelled prefill to be given back"), |s| s.in_flight() == 0);
    }
    assert_eq!(s.metric(|m| &m.cancelled), 3);
    assert_eq!(s.metric(|m| &m.failed_sequences), 0);
    assert_clean(&s, "three prefills cancelled by their clients");
}

#[test]
fn a_flood_never_queues_past_the_depth_and_the_refusals_say_when_to_retry() {
    let _g = serial();
    let depth = 4;
    let s = start(Opts {
        slots: 1,
        queue_depth: depth,
        max_model_len: 2048,
        kv_tokens: 2048,
        mock: MockOptions { vocab: 4096, step_ms: 15, ..Default::default() },
        ..Opts::default()
    });
    let port = s.port;
    let statuses: Arc<Mutex<Vec<(u16, bool)>>> = Default::default();
    let threads: Vec<_> = (0..120)
        .map(|_| {
            let st = statuses.clone();
            std::thread::spawn(move || {
                let r = http_timeout(port, "POST", "/v1/chat/completions", Some(&chat_body(json!({"max_tokens": 6}))), SECS(120));
                st.lock().unwrap().push((r.status, r.header("retry-after").is_some()));
            })
        })
        .collect();
    let mut deepest = 0;
    while threads.iter().any(|t| !t.is_finished()) {
        deepest = deepest.max(s.metric(|m| &m.queued));
        std::thread::sleep(Duration::from_millis(1));
    }
    for t in threads {
        t.join().unwrap();
    }
    let st = statuses.lock().unwrap().clone();
    assert_eq!(st.len(), 120);
    assert!(deepest <= depth as u64, "the queue held {deepest} with a depth of {depth}");
    assert!(st.iter().all(|(c, _)| matches!(c, 200 | 429)), "{st:?}");
    assert!(st.iter().filter(|(c, _)| *c == 429).all(|(_, ra)| *ra), "a 429 without Retry-After");
    assert!(
        st.iter().filter(|(c, _)| *c == 429).count() >= 60,
        "a flood of 120 into 1 slot and a queue of {depth} should be refused mostly"
    );
    assert_clean(&s, "a flood");
}

#[test]
fn content_types_methods_and_framing_that_are_wrong_are_refused_with_a_4xx() {
    let _g = serial();
    let s = start(small_pool());
    let body = chat_body(json!({"max_tokens": 2})).to_string().into_bytes();
    for (ct, want_ok) in [
        ("application/json", true),
        ("application/json; charset=utf-8", true),
        ("APPLICATION/JSON", true),
        ("application/vnd.openai+json", true),
        ("text/plain", false),
        ("application/x-www-form-urlencoded", false),
        ("multipart/form-data; boundary=x", false),
        ("", false),
    ] {
        let st = status_of(&post_raw(s.port, "/v1/chat/completions", ct, &body)).unwrap_or(0);
        if want_ok {
            assert_eq!(st, 200, "content type {ct:?}");
        } else {
            assert!((400..500).contains(&st), "content type {ct:?} answered {st}");
        }
    }
    // a multipart body with an image part, as a form-upload client would send it: refused, never parsed as an image
    let multipart = b"--x\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.png\"\r\nContent-Type: image/png\r\n\r\n\x89PNG\r\n\x1a\n\r\n--x--\r\n";
    let st = status_of(&post_raw(s.port, "/v1/chat/completions", "multipart/form-data; boundary=x", multipart)).unwrap_or(0);
    assert!((400..500).contains(&st), "multipart: {st}");
    // methods the routes do not have (no CORS preflight either: OPTIONS is not answered)
    for m in ["PUT", "DELETE", "PATCH", "OPTIONS", "TRACE", "CONNECT"] {
        let r =
            raw(s.port, format!("{m} /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").as_bytes(), SECS(5));
        let st = status_of(&r);
        assert!(st.is_none() || (400..500).contains(&st.unwrap()), "{m}: {st:?}");
    }
    // framing: conflicting lengths, a negative length, both headers, a chunked body, a body on GET
    let head = "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Type: application/json\r\n";
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("negative Content-Length", format!("{head}Content-Length: -1\r\n\r\n").into_bytes()),
        ("two different Content-Length", format!("{head}Content-Length: 2\r\nContent-Length: 3\r\n\r\n{{}}").into_bytes()),
        (
            "Content-Length and Transfer-Encoding",
            format!("{head}Content-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{{}}\r\n0\r\n\r\n").into_bytes(),
        ),
        ("a space in the header name", format!("{head}Bad Header: x\r\nContent-Length: 2\r\n\r\n{{}}").into_bytes()),
        ("an obs-fold header", format!("{head}X-A: a\r\n  folded\r\nContent-Length: 2\r\n\r\n{{}}").into_bytes()),
        ("a bare LF line ending", b"POST /v1/chat/completions HTTP/1.1\nHost: x\nContent-Length: 2\n\n{}".to_vec()),
        ("garbage instead of a request", b"\x16\x03\x01\x02\x00\x01\x00\x01\xfc\x03\x03".to_vec()),
        ("HTTP/0.9", b"GET /health\r\n".to_vec()),
        ("a bad chunk size", format!("{head}Transfer-Encoding: chunked\r\n\r\nzz\r\n").into_bytes()),
    ];
    for (name, req) in cases {
        let st = status_of(&raw(s.port, &req, SECS(1)));
        assert!(st.is_none() || (400..500).contains(&st.unwrap()), "{name}: {st:?}");
        assert!(healthy(s.port), "after {name}");
    }
    assert_clean(&s, "wrong content types and framing");
}

#[test]
fn header_abuse_is_refused_by_the_parser_and_the_server_goes_on() {
    let _g = serial();
    let s = start(small_pool());
    let get = |extra: &[u8]| -> Vec<u8> {
        let mut r = b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n".to_vec();
        r.extend_from_slice(extra);
        r.extend_from_slice(b"\r\n");
        raw(s.port, &r, SECS(10))
    };
    // a megabyte in one header, 10,000 headers, a megabyte of URL
    let one_mib = [b"X-A: ".to_vec(), vec![b'a'; 1 << 20], b"\r\n".to_vec()].concat();
    assert_eq!(status_of(&get(&one_mib)), Some(431));
    let many: Vec<u8> = (0..10_000).flat_map(|i| format!("X-{i}: v\r\n").into_bytes()).collect();
    assert_eq!(status_of(&get(&many)), Some(431));
    let url = [b"GET /".to_vec(), vec![b'a'; 1 << 20], b" HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".to_vec()].concat();
    assert_eq!(status_of(&raw(s.port, &url, SECS(10))), Some(431));
    // a few ordinary extras are fine: a long cookie, a forwarded-for list, an Expect
    assert_eq!(status_of(&get(&[b"Cookie: ".to_vec(), vec![b'c'; 8000], b"\r\n".to_vec()].concat())), Some(200));
    assert_eq!(status_of(&get(b"X-Forwarded-For: 1.2.3.4, 5.6.7.8\r\nAuthorization: Bearer abc\r\n")), Some(200));
    // a health request with a body is not an error
    assert_eq!(
        status_of(&raw(s.port, b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\nContent-Length: 3\r\n\r\nabc", SECS(5))),
        Some(200)
    );
    assert_clean(&s, "header abuse");
}

#[test]
fn http2_is_not_spoken_and_the_attempt_leaves_nothing_open() {
    let _g = serial();
    let s = start(small_pool());
    // prior-knowledge h2c: the connection preface, then a SETTINGS frame
    let r = raw(s.port, b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n\x00\x00\x00\x04\x00\x00\x00\x00\x00", SECS(1));
    assert!(!r.starts_with(&[0, 0]), "an HTTP/2 SETTINGS frame came back: {:?}", &r[..r.len().min(16)]);
    // an Upgrade: h2c request is just an HTTP/1.1 request
    let r = raw(s.port, b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: Upgrade, HTTP2-Settings\r\nUpgrade: h2c\r\nHTTP2-Settings: AAMAAABkAAQAAP__\r\n\r\n", SECS(2));
    assert!(matches!(status_of(&r), Some(200) | Some(400) | None), "{:?}", status_of(&r));
    s.wait(SECS(30), "the connections to close", |s| established_on(s.port) == 0);
    assert_clean(&s, "h2c");
}

#[test]
fn a_huge_max_tokens_is_clamped_to_the_window_and_the_reply_ends_at_length() {
    let _g = serial();
    let s = start(Opts {
        max_model_len: 256,
        kv_tokens: 256,
        slots: 2,
        mock: MockOptions { vocab: 4096, ..Default::default() },
        ..Opts::default()
    });
    for mt in [json!(1_000_000), json!(u64::MAX), json!(1e30), json!(1.0e19), json!(9007199254740993u64)] {
        let r = chat(s.port, json!({"max_tokens": mt.clone(), "temperature": 0.0}));
        assert_eq!(r.status, 200, "max_tokens {mt}: {}", r.body);
        let v = r.json();
        assert_eq!(v["choices"][0]["finish_reason"], "length", "{mt}");
        let (p, c) = (v["usage"]["prompt_tokens"].as_u64().unwrap(), v["usage"]["completion_tokens"].as_u64().unwrap());
        assert_eq!(p + c, 256, "max_tokens {mt} ran to the window and no further");
    }
    // a value that is not a count at all is a 400, not a clamp
    for mt in [json!(0), json!(-5), json!(0.5), json!("many"), json!([5]), json!({"n": 5})] {
        let r = chat(s.port, json!({"max_tokens": mt.clone()}));
        assert_eq!(r.status, 400, "max_tokens {mt}: {}", r.body);
    }
    let r = chat(s.port, json!({"max_completion_tokens": 3, "max_tokens": 200}));
    assert_eq!(r.json()["usage"]["completion_tokens"], 3, "max_completion_tokens wins");
    s.wait(SECS(10), "idle", |s| s.in_flight() == 0);
    assert!(healthy(s.port));
}

#[test]
fn stop_strings_are_capped_in_number_and_size() {
    let _g = serial();
    let s = start(small_pool());
    let strings = |n: usize, len: usize| -> Vec<String> { (0..n).map(|i| format!("{i:0>len$}")).collect() };
    let ok = chat(s.port, json!({"max_tokens": 8, "stop": strings(gb_server::api::MAX_STOP_STRINGS, 8)}));
    assert_eq!(ok.status, 200, "{}", ok.body);
    let ok = chat(s.port, json!({"max_tokens": 8, "stop": "x".repeat(gb_server::api::MAX_STOP_BYTES)}));
    assert_eq!(ok.status, 200, "{}", ok.body);
    for (what, stop) in [
        ("one more than the cap", json!(strings(gb_server::api::MAX_STOP_STRINGS + 1, 8))),
        ("100,000 strings", json!(strings(100_000, 40))),
        ("one string a byte over", json!("x".repeat(gb_server::api::MAX_STOP_BYTES + 1))),
        ("one 4 MiB string", json!("x".repeat(4 << 20))),
    ] {
        let t = Instant::now();
        let r = chat(s.port, json!({"max_tokens": 8, "stop": stop}));
        assert_eq!(r.status, 400, "{what}: {}", r.body);
        assert!(r.body.contains("stop"), "{what}: {}", r.body);
        assert!(t.elapsed() < SECS(5), "{what} took {:?} to refuse", t.elapsed());
    }
    assert_clean(&s, "stop strings");
}

#[test]
fn n_is_one_or_absent_and_every_other_value_is_refused() {
    let _g = serial();
    let s = start(small_pool());
    for n in [json!(1), json!(1.0), json!(null)] {
        assert_eq!(chat(s.port, json!({"max_tokens": 2, "n": n.clone()})).status, 200, "n = {n}");
    }
    for n in [json!(2), json!(100), json!(0), json!(-1), json!(2.5), json!("3"), json!("1"), json!(true), json!([1]), json!({})] {
        let r = chat(s.port, json!({"max_tokens": 2, "n": n.clone()}));
        assert_eq!(r.status, 400, "n = {n} was answered: {}", r.body);
        assert!(r.body.contains("\"n"), "n = {n}: {}", r.body);
    }
    let r = post_json(s.port, "/v1/completions", &json!({"prompt": [1, 2, 3], "max_tokens": 2, "n": 2}));
    assert_eq!(r.status, 400, "{}", r.body);
    assert_clean(&s, "n");
}

#[test]
fn a_completion_prompt_of_token_ids_is_exact_or_refused() {
    let _g = serial();
    let s = start(small_pool());
    let prompt = |p: Value| post_json(s.port, "/v1/completions", &json!({"prompt": p, "max_tokens": 2, "temperature": 0.0}));
    let a = prompt(json!([5, 6, 7]));
    assert_eq!(a.status, 200, "{}", a.body);
    let b = prompt(json!([[5, 6, 7]]));
    assert_eq!(b.status, 200, "{}", b.body);
    assert_eq!(a.json()["choices"][0]["text"], b.json()["choices"][0]["text"], "the nested form is the same prompt");
    let c = a.json()["usage"]["prompt_tokens"].as_u64();
    assert_eq!(c, Some(3));
    for (what, p) in [
        ("an id above u32::MAX (it used to wrap to 5)", json!([4294967296u64 + 5])),
        ("u64::MAX", json!([u64::MAX])),
        ("a negative id", json!([-1])),
        ("a float id", json!([1.5])),
        ("a string among the ids", json!([1, "2", 3])),
        ("a string among the nested ids (it used to be dropped)", json!([[1, "a", 2]])),
        ("null among the nested ids (it used to be dropped)", json!([[1, null, 2]])),
        ("an empty list", json!([])),
        ("an empty nested list", json!([[]])),
        ("two prompts (a batch)", json!([[1, 2], [3, 4]])),
        ("a bool", json!(true)),
        ("an object", json!({"a": 1})),
        ("a number", json!(5)),
    ] {
        let r = prompt(p);
        assert_eq!(r.status, 400, "{what}: {}", r.body);
    }
    // an id past the vocabulary is the engine's refusal
    let r = prompt(json!([100_000]));
    assert_eq!(r.status, 400, "{}", r.body);
    assert_clean(&s, "token-id prompts");
}

#[test]
fn every_refusal_the_api_makes_has_the_error_shape() {
    let _g = serial();
    let s = start(small_pool());
    let chat_to = |b: Value| post_json(s.port, "/v1/chat/completions", &b);
    let m = || json!([{"role": "user", "content": "hi"}]);
    let refusals: Vec<(&str, Reply)> = vec![
        ("messages missing", chat_to(json!({}))),
        ("messages not a list", chat_to(json!({"messages": "hi"}))),
        ("messages empty", chat_to(json!({"messages": []}))),
        // ("no user turn" was refused by the Qwen3.8 template; the Nemotron template renders a system-only conversation)
        ("logprobs", chat_to(json!({"messages": m(), "logprobs": true}))),
        ("logit_bias", chat_to(json!({"messages": m(), "logit_bias": {"1": 5}}))),
        ("presence_penalty", chat_to(json!({"messages": m(), "presence_penalty": 1}))),
        ("repetition_penalty", chat_to(json!({"messages": m(), "repetition_penalty": 1.5}))),
        ("temperature negative", chat_to(json!({"messages": m(), "temperature": -1}))),
        ("temperature a string", chat_to(json!({"messages": m(), "temperature": "hot"}))),
        ("top_p out of range", chat_to(json!({"messages": m(), "top_p": 7}))),
        ("seed negative", chat_to(json!({"messages": m(), "seed": -3}))),
        ("stop of the wrong type", chat_to(json!({"messages": m(), "stop": 5}))),
        ("tool_choice required", chat_to(json!({"messages": m(), "tool_choice": "required"}))),
        ("chat_template_kwargs not an object", chat_to(json!({"messages": m(), "chat_template_kwargs": [1]}))),
        (
            "an image without image support",
            chat_to(
                json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://127.0.0.1:1/a.png"}}]}]}),
            ),
        ),
        ("a video part", chat_to(json!({"messages": [{"role": "user", "content": [{"type": "video", "video": "x"}]}]}))),
        ("a prompt past the window", chat_to(json!({"messages": [{"role": "user", "content": "word ".repeat(5000)}]}))),
        ("a completion prompt of the wrong type", post_json(s.port, "/v1/completions", &json!({"prompt": 5}))),
        ("a completion prompt past the window", post_json(s.port, "/v1/completions", &completion_body(&vec![5; 3000], 4, false))),
    ];
    for (what, r) in refusals {
        assert_eq!(r.status, 400, "{what}: {}", r.body);
        assert!(
            r.header("content-type").unwrap_or("").starts_with("application/json"),
            "{what}: content type {:?}",
            r.header("content-type")
        );
        let v = r.json();
        let e = &v["error"];
        assert!(e["message"].as_str().is_some_and(|m| !m.is_empty()), "{what}: {v}");
        assert_eq!(e["type"], "invalid_request_error", "{what}: {v}");
        assert!(e["code"].is_number() || e["code"].is_string(), "{what}: {v}");
    }
    assert_clean(&s, "refusals");
}

/// One `/metrics` line is `name{labels} value` with a numeric value, preceded by its `# TYPE`; a series appears once.
fn check_exposition(text: &str) -> BTreeMap<String, f64> {
    let mut typed: BTreeMap<String, String> = BTreeMap::new();
    let mut series: BTreeMap<String, f64> = BTreeMap::new();
    for line in text.lines() {
        if let Some(r) = line.strip_prefix("# TYPE ") {
            let (name, kind) = r.split_once(' ').unwrap_or_else(|| panic!("bad TYPE line {line:?}"));
            assert!(matches!(kind, "counter" | "gauge" | "histogram" | "summary" | "untyped"), "{line}");
            assert!(typed.insert(name.to_string(), kind.to_string()).is_none(), "TYPE of {name} twice");
        } else if line.starts_with('#') || line.is_empty() {
            continue;
        } else {
            let (name_labels, value) = line.rsplit_once(' ').unwrap_or_else(|| panic!("bad sample line {line:?}"));
            let v: f64 = value.parse().unwrap_or_else(|_| panic!("value of {line:?} is not a number"));
            let name = name_labels.split('{').next().unwrap();
            assert!(name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':'), "metric name {name:?}");
            let histogram = ["_bucket", "_sum", "_count"].iter().any(|suffix| {
                name.strip_suffix(suffix).is_some_and(|base| typed.get(base).is_some_and(|kind| kind == "histogram"))
            });
            assert!(typed.contains_key(name) || histogram, "{name} has no TYPE line before its first sample");
            if let Some(l) = name_labels.strip_prefix(name).filter(|l| !l.is_empty()) {
                assert!(l.starts_with('{') && l.ends_with('}'), "labels of {line:?}");
            }
            assert!(series.insert(name_labels.to_string(), v).is_none(), "series {name_labels} twice");
        }
    }
    series
}

#[test]
fn metrics_stay_valid_prometheus_text_and_count_what_the_abuse_did() {
    let _g = serial();
    // 5 ms a step: the stream below cannot finish (1,000 tokens, 5 s) before its client leaves, however the runner is loaded
    let s = start(Opts { mock: MockOptions { vocab: 4096, step_ms: 5, ..Default::default() }, ..small_pool() });
    let before = check_exposition(&http_timeout(s.port, "GET", "/metrics", None, SECS(5)).body);
    for k in [
        "rexl3_ready",
        "rexl3_requests_total",
        "rexl3_requests_cancelled_total",
        "rexl3_queue_depth",
        "rexl3_active_sequences",
        "rexl3_failed_sequences_total",
        "rexl3_http_requests_total",
        "rexl3_requests_completed_total",
        "rexl3_token_batches_total",
        "rexl3_slots_total",
        "rexl3_request_ttft_seconds_count",
        "rexl3_token_batch_itl_seconds_count",
        "rexl3_request_e2e_seconds_count",
        "rexl3_http_responses_total{code=\"400\"}",
        "rexl3_sse_terminal_total{outcome=\"error\"}",
    ] {
        assert!(before.contains_key(k), "{k} missing from /metrics");
    }
    // a mix: successes, a client that leaves mid-reply, refusals (which the engine never sees)
    for _ in 0..5 {
        assert_eq!(chat(s.port, json!({"max_tokens": 4})).status, 200);
    }
    let c = Conn::post(s.port, "/v1/chat/completions", &chat_body(json!({"stream": true, "max_tokens": 1000})));
    let mut c = c;
    assert!(c.read_until(SECS(10), |t| t.contains("data:")));
    drop(c);
    for _ in 0..5 {
        assert_eq!(chat(s.port, json!({"n": 2})).status, 400);
    }
    s.wait(SECS(10), "idle", |s| s.in_flight() == 0);
    let after = check_exposition(&http_timeout(s.port, "GET", "/metrics", None, SECS(5)).body);
    assert_eq!(after["rexl3_requests_total"] - before["rexl3_requests_total"], 6.0);
    assert_eq!(after["rexl3_requests_cancelled_total"] - before["rexl3_requests_cancelled_total"], 1.0);
    assert_eq!(after["rexl3_active_sequences"], 0.0);
    assert_eq!(after["rexl3_queue_depth"], 0.0);
    assert_eq!(after["rexl3_prefilling_sequences"], 0.0);
    assert_clean(&s, "metrics");
}

/// Serve `s`'s router from a runtime of only `workers` threads, so a request that holds a worker is felt at once.
fn few_workers(s: &Server, workers: usize) -> (tokio::runtime::Runtime, u16) {
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(workers).enable_all().build().unwrap();
    let listener = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = gb_server::http::router(s.state.clone());
    rt.spawn(async move { axum::serve(listener, app).await.unwrap() });
    (rt, port)
}

#[test]
fn rendering_and_tokenizing_do_not_hold_the_async_workers() {
    let _g = serial();
    let s = start(small_pool());
    // One async worker: if preparation runs on it, /health cannot answer until we release the tokenizer.
    let (_rt, port) = few_workers(&s, 1);
    let gate = s.state.request_budget.tokenizer_gate();
    // Reserve all but less than one input byte's price. This holds no real allocation, and Drop releases it even on assertion failure,
    // before the runtime joins its blocking workers. Wait for an actual encode to block, not an assumed amount of CPU work.
    let held = gate.charge(gate.capacity() / gb_server::budget::TOKENIZER_BYTES_PER_RENDERED_BYTE).unwrap();
    let c = Conn::post(
        port,
        "/v1/chat/completions",
        &chat_body(json!({"messages": [{"role": "user", "content": "x".repeat(3000)}], "max_tokens": 2})),
    );
    s.wait(SECS(5), "rendered prompt waiting for tokenizer credit", |_| gate.counts().0 == 1);
    assert!(healthy(port), "/health must answer while preparation is still blocked");
    assert_eq!(gate.counts(), (1, 0), "the tokenizer must still be waiting, not have timed out");
    assert_eq!(s.metric(|m| &m.requests), 0, "blocked preparation cannot reach the engine");
    drop(held);
    let r = c.finish(SECS(5));
    assert_eq!(r.status, 400, "the text prompt is past the window: {}", r.body);
    assert_eq!(r.json()["error"]["code"], "context_length_exceeded");
    s.wait(SECS(5), "preparation to return its credits", |s| gate.in_use() == 0 && s.state.request_budget.used() == 0);
    assert_clean(&s, "blocked preparation");
}
