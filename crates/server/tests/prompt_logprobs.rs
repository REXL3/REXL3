//! `prompt_logprobs` on /v1/completions, against the real Axum server over the mock backend (whose logits are a pure function
//! of the context, `gb_engine::mock::mock_logits`). The readout follows vLLM's prompt-logprobs convention: entry 0 is null (no
//! prefix to condition on), and entry `i` scores prompt token `i` from the logits after tokens `0..i`.
mod common;
use common::{http_timeout, start, Opts};
use gb_engine::mock::mock_logits;
use serde_json::{json, Value};
use std::time::Duration;

fn post(s: &common::Server, body: Value) -> common::Reply {
    http_timeout(s.port, "POST", "/v1/completions", Some(&body), Duration::from_secs(20))
}

fn readout(s: &common::Server, ids: &[u32], k: usize) -> Value {
    let r = post(s, json!({"model": "mock", "prompt": ids, "max_tokens": 1, "prompt_logprobs": k}));
    assert_eq!(r.status, 200, "{}", r.body);
    r.json()
}

// The test server's mock backend has its own vocabulary (`Opts::default()` sets one; `MockOptions::build` falls back to the
// tokenizer's when it is 0), so the expected rows use it; the prompt ids stay below the tokenizer's vocabulary.

/// The exact log-softmax of `token` under `row`, in f64.
fn log_softmax(row: &[f32], token: u32) -> f64 {
    let peak = row.iter().fold(f64::NEG_INFINITY, |a, &v| a.max(v as f64));
    let z: f64 = row.iter().map(|&v| (v as f64 - peak).exp()).sum();
    row[token as usize] as f64 - peak - z.ln()
}

fn argmax(row: &[f32]) -> u32 {
    (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b]).then(b.cmp(&a))).unwrap() as u32
}

#[test]
fn each_prompt_token_is_scored_from_the_logits_before_it() {
    let o = Opts::default();
    let mock = o.mock.vocab;
    let s = start(o);
    let vocab = if mock == 0 { s.state.tok.vocab } else { mock };
    let ids: Vec<u32> = (0..17u32).map(|i| (i * 97 + 13) % s.state.tok.vocab as u32).collect();
    let k = 4;
    let v = readout(&s, &ids, k);
    let entries = v["prompt_logprobs"].as_array().unwrap();
    let lp = &v["choices"][0]["logprobs"];
    assert_eq!(entries.len(), ids.len(), "one entry per prompt token");
    assert_eq!(lp["token_logprobs"].as_array().unwrap().len(), ids.len());
    assert!(entries[0]["logprob"].is_null() && entries[0]["token_id"].is_null() && lp["token_logprobs"][0].is_null());
    let mut outside_top_k = 0;
    for i in 1..ids.len() {
        let row = mock_logits(&ids[..i], vocab);
        let e = &entries[i];
        assert_eq!((&e["position"], &e["token_id"]), (&json!(i), &json!(ids[i])), "entry {i}");
        let want = log_softmax(&row, ids[i]);
        let got = e["logprob"].as_f64().unwrap();
        assert!((got - want).abs() < 1e-5, "entry {i}: logprob {got}, the row after tokens 0..{i} gives {want}");
        assert_eq!(lp["token_logprobs"][i], e["logprob"]);
        let top: Vec<u32> = e["top_ids"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
        assert_eq!(top.len(), k);
        assert_eq!(top[0], argmax(&row), "entry {i}: the top ids are the row's after tokens 0..{i}");
        if !top.contains(&ids[i]) {
            outside_top_k += 1;
            assert!(
                got < e["top_logprobs"][k - 1].as_f64().unwrap(),
                "entry {i}: a token outside the top k has its own, lower value"
            );
        }
    }
    assert!(outside_top_k > 0, "the prompt exercises tokens outside the top {k}");
}

#[test]
fn appending_a_token_adds_one_entry_and_changes_none_before_it() {
    // the last logits row predicts past the prompt and is not part of the readout
    let o = Opts::default();
    let mock = o.mock.vocab;
    let s = start(o);
    let vocab = if mock == 0 { s.state.tok.vocab } else { mock };
    let ids: Vec<u32> = (0..9u32).map(|i| (i * 53 + 7) % s.state.tok.vocab as u32).collect();
    let short = readout(&s, &ids[..8], 3);
    let long = readout(&s, &ids, 3);
    let (a, b) = (short["prompt_logprobs"].as_array().unwrap(), long["prompt_logprobs"].as_array().unwrap());
    assert_eq!((a.len(), b.len()), (8, 9));
    assert_eq!(&b[..8], &a[..]);
    let want = log_softmax(&mock_logits(&ids[..8], vocab), ids[8]);
    assert!((b[8]["logprob"].as_f64().unwrap() - want).abs() < 1e-5);
}
