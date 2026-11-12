//! Request memory: what the tokenizer stage is priced at, how the rendered-prompt cap follows the window, and that the processing reserve is
//! charged at request time (a prompt waits for its priced bytes, a prompt that could never fit is refused) instead of being only a startup sum.
mod common;
use common::{chat_body, http_timeout, start, Opts};
use gb_server::budget::{default_rendered_bytes, Limits, RequestBudget, TokenizerGate, TOKENIZER_BYTES_PER_RENDERED_BYTE as PER};
use serde_json::json;
use std::sync::mpsc;
use std::time::{Duration, Instant};

const MIB: usize = 1 << 20;

#[test]
fn the_default_rendered_cap_is_sixteen_bytes_a_window_token_with_a_floor_of_one_mib() {
    assert_eq!(default_rendered_bytes(262_144), 4 * MIB);
    assert_eq!(default_rendered_bytes(300_000), 4_800_000);
    assert_eq!(default_rendered_bytes(65_536), MIB);
    assert_eq!(default_rendered_bytes(4_096), MIB, "the floor");
    assert_eq!(default_rendered_bytes(0), MIB);
    assert_eq!(default_rendered_bytes(usize::MAX), usize::MAX, "saturates, never wraps");
    assert_eq!(Limits::default().rendered_bytes, 4 * MIB, "the shipped window is 262,144 tokens");
    assert_eq!(Limits::for_window(1 << 20).rendered_bytes, 16 * MIB);
}

#[test]
fn the_tokenizer_stage_is_priced_at_the_measured_figure_not_128() {
    assert_eq!(PER, 448, "measured 371 to 416 bytes of RSS per rendered byte, plus 8 %");
    // the old arithmetic (128 x the cap x 2 permits) accepted an 8 MiB cap against the 2 GiB reserve: one such prompt is priced at 3.5 GiB now
    let eight = Limits { rendered_bytes: 8 * MIB, processing_reserve_bytes: 2 << 30, ..Limits::default() };
    let e = RequestBudget::new(eight.clone()).err().expect("an 8 MiB cap does not fit a 2 GiB reserve at 448 B a byte");
    assert!(e.contains("448 per rendered byte"), "{e}");
    // the reserve that holds exactly one prompt at the cap (and a decision label) is enough: concurrency is by price, not by permits
    let need = eight.tokenizer_price().unwrap();
    assert_eq!(need, (8 * MIB + 64) * PER);
    assert!(RequestBudget::new(Limits { processing_reserve_bytes: need, ..eight.clone() }).is_ok());
    assert!(RequestBudget::new(Limits { processing_reserve_bytes: need - 1, ..eight }).is_err());
    // permits no longer multiply the rendered term: eight preparation jobs on small bodies need the one prompt's price, not eight of them
    let wide = Limits { body_bytes: MIB, total_bytes: MIB, preparation_concurrency: 8, ..Limits::default() };
    assert!(RequestBudget::new(wide).is_ok());
}

#[test]
fn a_window_past_the_default_reserve_derives_a_reserve_that_holds_one_prompt() {
    let big = Limits::for_window(1 << 20);
    assert!(big.processing_reserve_bytes > 2 << 30, "{}", big.processing_reserve_bytes);
    assert!(big.processing_reserve_bytes >= big.tokenizer_price().unwrap());
    assert!(RequestBudget::new(big).is_ok(), "a 1M-token window must boot with the other defaults");
    assert_eq!(Limits::for_window(262_144).processing_reserve_bytes, 2 << 30, "the shipped window keeps the 2 GiB reserve");
    // 262,144 tokens is 4 MiB; 4.8 MB x 448 is 2.0027 GiB, so a 300,000-token window derives a reserve just above the default
    let w300 = Limits::for_window(300_000);
    assert!(
        w300.processing_reserve_bytes > 2 << 30 && w300.processing_reserve_bytes < (2 << 30) + (8 << 20),
        "{}",
        w300.processing_reserve_bytes
    );
}

#[test]
fn the_gate_refuses_what_could_never_fit_and_gives_the_bytes_back() {
    let gate = TokenizerGate::new(100 * PER, Duration::from_millis(150));
    assert_eq!(gate.capacity(), 100 * PER);
    let e = gate.charge(101).err().expect("101 bytes are priced above the whole reserve");
    assert_eq!(e.status, 413);
    assert_eq!(e.kind, "invalid_request_error");
    assert!(e.message.contains("processing reserve"), "{}", e.message);
    assert!(gate.charge(usize::MAX).is_err(), "the price arithmetic is checked");
    assert_eq!(gate.in_use(), 0);
    let all = gate.charge(100).unwrap();
    assert_eq!(gate.in_use(), 100 * PER);
    let t0 = Instant::now();
    let e = gate.charge(1).err().expect("the reserve is in use");
    assert_eq!(e.status, 429);
    assert_eq!(e.kind, "server_overloaded");
    assert_eq!(e.retry_after, Some(2));
    assert!(t0.elapsed() >= Duration::from_millis(100), "it waited for the reserve before it gave up");
    drop(all);
    assert_eq!(gate.in_use(), 0);
    let (a, b) = (gate.charge(60).unwrap(), gate.charge(40).unwrap());
    assert_eq!(gate.in_use(), 100 * PER);
    drop(a);
    assert_eq!(gate.in_use(), 40 * PER);
    drop(b);
    assert_eq!(gate.in_use(), 0);
    let nothing = gate.charge(0).unwrap();
    assert_eq!(gate.in_use(), 0, "an empty text costs nothing");
    drop(nothing);
}

#[test]
fn a_waiting_prompt_goes_first_come_first_served_and_small_ones_do_not_starve_it() {
    let gate = TokenizerGate::new(100 * PER, Duration::from_secs(20));
    let held = gate.charge(60).unwrap();
    let (tx, rx) = mpsc::channel();
    let (release_big, big_waits) = mpsc::channel::<()>();
    let big = {
        let (gate, tx) = (gate.clone(), tx.clone());
        std::thread::spawn(move || {
            let c = gate.charge(80).unwrap(); // needs the 60 held to go back first
            tx.send("big in").unwrap();
            big_waits.recv().unwrap();
            drop(c);
            tx.send("big out").unwrap();
        })
    };
    while gate.counts().0 < 1 {
        std::thread::sleep(Duration::from_millis(2));
    }
    // 40 are free and 30 would fit, but the 80-byte prompt is ahead of it: it queues behind it instead of jumping
    let small = {
        let (gate, tx) = (gate.clone(), tx.clone());
        std::thread::spawn(move || {
            let c = gate.charge(30).unwrap();
            tx.send("small in").unwrap();
            drop(c);
        })
    };
    while gate.counts().0 < 2 {
        std::thread::sleep(Duration::from_millis(2));
    }
    std::thread::sleep(Duration::from_millis(100));
    assert!(rx.try_recv().is_err(), "neither runs while 60 of 100 are held and the big one is first in line");
    drop(held);
    assert_eq!(rx.recv_timeout(Duration::from_secs(10)).unwrap(), "big in", "the one that arrived first got its bytes first");
    std::thread::sleep(Duration::from_millis(100));
    assert!(rx.try_recv().is_err(), "80 of 100 are held: the 30 bytes wait");
    assert_eq!(gate.in_use(), 80 * PER);
    release_big.send(()).unwrap();
    let rest: Vec<_> = (0..2).map(|_| rx.recv_timeout(Duration::from_secs(10)).unwrap()).collect();
    assert!(rest.contains(&"big out") && rest.contains(&"small in"), "{rest:?}");
    big.join().unwrap();
    small.join().unwrap();
    assert_eq!(gate.in_use(), 0);
    assert_eq!(gate.counts(), (2, 0));
}

fn limits_for_wait(wait: Duration) -> Limits {
    // the default 2 GiB reserve, with the wait a test can afford
    Limits { tokenizer_wait: wait, ..Limits::default() }
}

#[test]
fn a_prompt_the_reserve_cannot_take_now_is_429_over_http_and_runs_when_the_bytes_are_back() {
    let s = start(Opts { request_limits: limits_for_wait(Duration::from_millis(400)), ..Opts::default() });
    let gate = s.state.request_budget.tokenizer_gate().clone();
    let post = |path: &str, body: serde_json::Value| http_timeout(s.port, "POST", path, Some(&body), Duration::from_secs(20));
    // the whole reserve is taken by someone else: every prompt-tokenizing route answers the existing 429 shape
    let hold = gate.charge(gate.capacity() / PER).unwrap();
    for (path, body) in [
        ("/v1/chat/completions", chat_body(json!({"max_tokens": 1}))),
        ("/v1/chat/completions/render", chat_body(json!({"max_tokens": 1}))),
        ("/v1/completions", json!({"model": "mock", "prompt": "hello there", "max_tokens": 1})),
    ] {
        let r = post(path, body);
        assert_eq!(r.status, 429, "{path}: {}", r.body);
        assert_eq!(r.header("retry-after"), Some("2"), "{path}");
        let e = r.json()["error"].clone();
        assert_eq!(e["type"], "server_overloaded", "{path}: {e}");
        assert!(e["message"].as_str().unwrap().contains("processing reserve"), "{path}: {e}");
    }
    drop(hold);
    assert_eq!(gate.in_use(), 0, "a refused request left nothing charged");
    let r = post("/v1/chat/completions", chat_body(json!({"max_tokens": 1})));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(gate.in_use(), 0, "the charge is released as soon as the prompt is tokenized, not when the reply ends");
}

#[test]
fn a_request_that_has_to_wait_for_the_reserve_waits_and_is_served() {
    let s = start(Opts { request_limits: limits_for_wait(Duration::from_secs(30)), ..Opts::default() });
    let gate = s.state.request_budget.tokenizer_gate().clone();
    let hold = gate.charge(gate.capacity() / PER).unwrap();
    let port = s.port;
    let client = std::thread::spawn(move || {
        let t0 = Instant::now();
        let r = http_timeout(
            port,
            "POST",
            "/v1/chat/completions",
            Some(&chat_body(json!({"max_tokens": 1}))),
            Duration::from_secs(25),
        );
        (r, t0.elapsed())
    });
    while gate.counts().0 < 1 {
        std::thread::sleep(Duration::from_millis(5));
    }
    std::thread::sleep(Duration::from_millis(300));
    drop(hold);
    let (r, took) = client.join().unwrap();
    assert_eq!(r.status, 200, "{}", r.body);
    assert!(took >= Duration::from_millis(300), "it waited for the reserve: {took:?}");
    assert_eq!(gate.counts(), (1, 0));
}

#[test]
fn every_encode_of_a_charged_tokenizer_is_priced_and_released() {
    use gb_server::tokenizer::Tok;
    let tok = Tok::from_dir(&common::fixture_dir()).unwrap();
    let gate = TokenizerGate::new(100 * PER, Duration::from_millis(50));
    tok.charge_to(gate.clone());
    // `encode` and `encode_bounded` both take the price of their text first
    for text in ["x".repeat(101), "y".repeat(5000)] {
        let e = tok.encode(&text).expect_err("priced above the whole reserve");
        assert_eq!(e.status, 413, "{}", e.message);
        let e = tok.encode_bounded(&text, 1 << 20).expect_err("priced above the whole reserve");
        assert_eq!(e.status, 413, "{}", e.message);
    }
    assert_eq!(tok.encode("x".repeat(100).as_str()).unwrap().len(), 100);
    assert_eq!(tok.encode_bounded(&"z".repeat(100), 100).unwrap().len(), 100);
    assert_eq!(gate.in_use(), 0, "nothing is held once the ids are returned");
    // a held reserve makes both 429 (not a hang), and an unbounded text refuses at the byte cap first, as before
    let hold = gate.charge(100).unwrap();
    assert_eq!(tok.encode("abc").err().unwrap().status, 429);
    assert_eq!(tok.encode_bounded("abc", 2).err().unwrap().status, 400);
    drop(hold);
    // the first gate set stays (a tokenizer serves one server)
    tok.charge_to(TokenizerGate::new(1, Duration::from_millis(1)));
    assert!(tok.encode("abc").is_ok());
    let (_, refused) = gate.counts();
    assert_eq!(refused, 5);
}
