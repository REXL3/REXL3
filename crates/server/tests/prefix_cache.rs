//! The prefix cache through the HTTP path on the CPU mock backend (a mock with paged KV, snapshot slots and a path-dependent state, see
//! `gb_engine::mock`): the request opt-out, `usage.prompt_tokens_details.cached_tokens`, the `/metrics` counters, and that a cached reply is
//! the uncached reply. The engine-level contract has its own tests (`crates/engine/tests/prefix_cache.rs`) and the real model's are in
//! `crates/model/tests/prefix_cache.rs`.

mod common;

use common::{completion_body, http, sse, start, Opts, Server};
use serde_json::{json, Value};

fn server(prefix_cache: usize) -> Server {
    start(Opts {
        max_model_len: 4096,
        mock: gb_engine::mock::MockOptions { vocab: 4096, prefix_cache, ..Default::default() },
        ..Default::default()
    })
}

/// A prompt of `n` token ids with a salt (every id is below the mock's vocabulary).
fn prompt(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| 1 + (i.wrapping_mul(2654435761).wrapping_add(salt) >> 7) % 4000).collect()
}

fn complete(s: &Server, body: Value) -> Value {
    let (st, b) = http(s.port, "POST", "/v1/completions", Some(&body));
    assert_eq!(st, 200, "{b}");
    serde_json::from_str(&b).unwrap()
}

fn cached(v: &Value) -> u64 {
    v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().expect("usage carries prompt_tokens_details.cached_tokens")
}

fn metric(s: &Server, name: &str) -> Option<f64> {
    let (_, m) = http(s.port, "GET", "/metrics", None);
    m.lines().find_map(|l| l.strip_prefix(name).and_then(|r| r.strip_prefix(' ')).and_then(|v| v.trim().parse().ok()))
}

#[test]
fn cache_enabled_state_is_reported_as_soon_as_the_server_is_ready() {
    let enabled = server(6);
    assert!(enabled.handle().metrics.ready());
    assert_eq!(metric(&enabled, "rexl3_prefix_cache_enabled"), Some(1.0));

    let disabled = server(0);
    assert!(disabled.handle().metrics.ready());
    assert_eq!(metric(&disabled, "rexl3_prefix_cache_enabled"), Some(0.0));
}

#[test]
fn a_repeated_prompt_hits_reports_cached_tokens_and_answers_the_same() {
    let s = server(6);
    let body = |extra: Value| {
        let mut b = completion_body(&prompt(700, 1), 24, false);
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        b
    };
    let first = complete(&s, body(json!({})));
    assert_eq!(cached(&first), 0, "a cold server");
    let second = complete(&s, body(json!({})));
    assert_eq!(cached(&second), 692, "the end cut: 700 tokens less the tail margin of 8");
    assert_eq!(second["choices"][0]["text"], first["choices"][0]["text"], "a cached reply is the uncached reply");
    assert_eq!(second["usage"]["completion_tokens"], first["usage"]["completion_tokens"]);
    // the per-request opt-out is the uncached reference path: nothing looked up, same bytes
    let opted_out = complete(&s, body(json!({"prefix_cache": false})));
    assert_eq!(cached(&opted_out), 0);
    assert_eq!(opted_out["choices"][0]["text"], first["choices"][0]["text"]);
    // a longer prompt that starts with it resumes too
    let mut longer = prompt(700, 1);
    longer.extend(prompt(50, 2));
    let third = complete(&s, completion_body(&longer, 24, false));
    assert_eq!(cached(&third), 692);
    // the counters
    assert_eq!(metric(&s, "rexl3_prefix_cache_enabled"), Some(1.0));
    assert_eq!(metric(&s, "rexl3_prefix_cache_lookups_total"), Some(3.0), "the opted-out request did not look");
    assert_eq!(metric(&s, "rexl3_prefix_cache_hits_total"), Some(2.0));
    assert_eq!(metric(&s, "rexl3_prefix_cache_tokens_saved_total"), Some(1384.0));
    assert!(metric(&s, "rexl3_prefix_cache_entries").unwrap() >= 1.0);
    assert!(metric(&s, "rexl3_prefix_cache_bytes").unwrap() > 0.0);
    assert!(metric(&s, "rexl3_prefix_cache_hit_ratio").unwrap() > 0.5);
}

#[test]
fn a_streamed_reply_carries_cached_tokens_in_its_usage_chunk() {
    let s = server(6);
    let mut b = completion_body(&prompt(600, 3), 12, true);
    b["stream_options"] = json!({"include_usage": true});
    let mut texts = Vec::new();
    let mut cached_seen = Vec::new();
    for _ in 0..2 {
        let (st, body) = http(s.port, "POST", "/v1/completions", Some(&b));
        assert_eq!(st, 200);
        let chunks = sse(&body);
        texts.push(chunks.iter().filter_map(|c| c["choices"][0]["text"].as_str()).collect::<String>());
        cached_seen.push(chunks.iter().find(|c| c.get("usage").is_some_and(|u| !u.is_null())).map(cached).unwrap());
    }
    assert_eq!(cached_seen, vec![0, 592]);
    assert_eq!(texts[0], texts[1]);
}

#[test]
fn a_server_without_a_cache_says_so_and_reports_zero() {
    let s = server(0);
    let body = completion_body(&prompt(700, 4), 8, false);
    let a = complete(&s, body.clone());
    let b = complete(&s, body);
    assert_eq!((cached(&a), cached(&b)), (0, 0));
    assert_eq!(metric(&s, "rexl3_prefix_cache_enabled"), Some(0.0));
    assert_eq!(metric(&s, "rexl3_prefix_cache_hits_total"), None, "the counters exist only for a server that has a cache");
    assert_eq!(a["choices"][0]["text"], b["choices"][0]["text"]);
}

#[test]
fn short_prompts_are_never_cached_over_http_either() {
    let s = server(6);
    let body = completion_body(&prompt(100, 5), 8, false);
    for _ in 0..2 {
        assert_eq!(cached(&complete(&s, body.clone())), 0);
    }
    assert_eq!(metric(&s, "rexl3_prefix_cache_lookups_total"), Some(0.0));
    assert_eq!(metric(&s, "rexl3_prefix_cache_entries"), Some(0.0));
}
