//! A long prefill and short requests on one server (docs/LONG_PREFILL_POLICY.md), CPU mock only. The mock's prefill is a slice cost model:
//! 64-row slices of `prefill_ms` each, so a prompt of `P` tokens takes about `ceil(P / 64) x prefill_ms` of engine time, and the engine runs
//! one slice of every admitted prompt in turn (`Engine::prefilling`, round robin). What decides co-tenancy is admission: with the KV pool
//! one context (the default plan) a short request cannot get pages while a maximal one holds them and fails at `--queue-timeout`; with a
//! co-tenancy headroom (`--kv-headroom-tokens`) it is admitted at once and its slices interleave with the long prompt's.
mod common;
use common::{completion_body, http_timeout, start, Opts};
use gb_engine::mock::MockOptions;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

const WINDOW: usize = 4096;
/// The long prompt: 4,000 tokens, 63 slices of 64 rows at 20 ms, about 1.3 s of prefill.
const LONG: usize = 4000;
const SLICE_MS: u64 = 20;
/// The short prompt: 200 tokens, 4 slices.
const SHORT: usize = 200;

fn server(headroom: usize, queue_timeout: Duration) -> common::Server {
    start(Opts {
        pool: true,
        slots: 4,
        max_model_len: WINDOW,
        kv_tokens: WINDOW + headroom,
        queue_timeout,
        mock: MockOptions { vocab: 4096, prefill_ms: SLICE_MS, slice: 64, step_ms: 1, ..Default::default() },
        ..Default::default()
    })
}

/// Start the long request (it holds `LONG + 8` tokens of the pool: all but 88 of the one-context pool) and wait until it is prefilling.
fn start_long(s: &common::Server) -> std::thread::JoinHandle<(u16, Instant)> {
    let port = s.port;
    let long = std::thread::spawn(move || {
        let prompt: Vec<u32> = (0..LONG as u32).map(|i| 5 + i % 3000).collect();
        let r = http_timeout(port, "POST", "/v1/completions", Some(&completion_body(&prompt, 8, false)), Duration::from_secs(60));
        (r.status, Instant::now())
    });
    s.wait(Duration::from_secs(10), "the long prompt to be prefilling", |s| s.metric(|m| &m.prefilling) == 1);
    long
}

fn short(port: u16) -> (u16, String, Instant) {
    let prompt: Vec<u32> = (0..SHORT as u32).map(|i| 7 + i).collect();
    let r = http_timeout(port, "POST", "/v1/completions", Some(&completion_body(&prompt, 4, false)), Duration::from_secs(60));
    (r.status, r.body, Instant::now())
}

/// With a headroom of 1,024 tokens the short request is admitted beside the long one and finishes while the long prompt is still prefilling:
/// its time is a few of its own slices interleaved with as many of the long prompt's (measured about 0.26 s against 1.26 s; the bound below
/// is half the long prefill, for a loaded CI box), and the long prompt completes too.
#[test]
fn with_a_headroom_a_short_request_completes_while_a_long_prefill_runs() {
    let s = server(1024, Duration::from_secs(30));
    let long = start_long(&s);
    let t0 = Instant::now();
    let (status, body, short_done) = short(s.port);
    assert_eq!(status, 200, "{body}");
    let short_latency = short_done - t0;
    // the long prompt has 63 slices; the short one needs 4 of its own and waits at most one of the long prompt's per own slice
    let long_prefill = Duration::from_millis(SLICE_MS * (LONG as u64).div_ceil(64));
    eprintln!("short request: {short_latency:?} beside a long prefill of {long_prefill:?} (63 slices of {SLICE_MS} ms)");
    assert!(short_latency < long_prefill / 2, "short request took {short_latency:?} beside a {long_prefill:?} prefill");
    assert_eq!(s.metric(|m| &m.prefilling), 1, "the long prompt is still prefilling when the short reply is complete");
    let (long_status, long_done) = long.join().unwrap();
    assert_eq!(long_status, 200);
    assert!(long_done > short_done);
    assert_eq!(s.metric(|m| &m.queue_timeouts), 0);
}

/// The same requests on the one-context pool (no headroom, the default plan): the short request cannot get pages while the long one holds
/// them, waits in the queue, and is answered 503 at `--queue-timeout` while the long prompt is still prefilling. This is the behaviour the
/// headroom changes; the default is untouched.
#[test]
fn without_a_headroom_the_short_request_waits_for_the_pool_and_times_out() {
    let s = server(0, Duration::from_millis(300));
    let long = start_long(&s);
    let (status, body, _) = short(s.port);
    assert_eq!(status, 503, "{body}");
    assert!(body.contains("waited"), "{body}");
    assert_eq!(s.metric(|m| &m.prefilling), 1, "the long prompt still holds the pool");
    assert_eq!(s.state.handle.metrics.queue_timeouts.load(Relaxed), 1);
    assert_eq!(long.join().unwrap().0, 200);
}

/// A prompt whose prefill cannot finish inside `--request-timeout`, even at the backend's optimistic floor (here 1 ms a token), is refused
/// at once with a 400 that says so and holds nothing; one that can is served. An engine that states no floor (every mock by default, every
/// family but Nemotron) never refuses on time: the default is unchanged.
#[test]
fn a_prompt_that_cannot_prefill_before_its_deadline_is_refused_at_once() {
    let s = start(Opts {
        request_timeout: Duration::from_secs(2),
        mock: MockOptions { vocab: 4096, prefill_floor_us: 1000, ..Default::default() },
        ..Default::default()
    });
    let long: Vec<u32> = (0..3000u32).map(|i| 5 + i % 3000).collect();
    let t0 = Instant::now();
    let r = http_timeout(s.port, "POST", "/v1/completions", Some(&completion_body(&long, 4, false)), Duration::from_secs(10));
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(t0.elapsed() < Duration::from_secs(1), "refused at once, not at the deadline");
    let e = &r.json()["error"];
    assert_eq!(e["code"], "request_deadline_too_short", "{}", r.body);
    assert!(e["message"].as_str().unwrap().contains("needs at least 3 s of prefill"), "{}", r.body);
    assert_eq!(s.metric(|m| &m.deadline_refusals), 1);
    assert_eq!(s.metric(|m| &m.requests), 0, "it never held a slot");
    let m = http_timeout(s.port, "GET", "/metrics", None, Duration::from_secs(5)).body;
    assert!(m.contains("\nrexl3_request_deadline_refusals_total 1\n"));
    // 1,000 tokens: a 1 s floor inside the 2 s deadline
    let short: Vec<u32> = (0..1000u32).map(|i| 5 + i).collect();
    let r = http_timeout(s.port, "POST", "/v1/completions", Some(&completion_body(&short, 4, false)), Duration::from_secs(10));
    assert_eq!(r.status, 200, "{}", r.body);
    // without a floor the same long prompt is served (the mock is fast): nothing is refused on time
    let s = start(Opts { request_timeout: Duration::from_secs(2), ..Default::default() });
    let r = http_timeout(s.port, "POST", "/v1/completions", Some(&completion_body(&long, 4, false)), Duration::from_secs(10));
    assert_eq!(r.status, 200, "{}", r.body);
    let m = http_timeout(s.port, "GET", "/metrics", None, Duration::from_secs(5)).body;
    assert!(!m.contains("deadline_refusals"), "the series appears only once a refusal happened");
}
