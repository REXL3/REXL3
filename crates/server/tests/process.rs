//! The real `rexl3-serve` binary on the CPU mock backend, as a supervisor sees it: exit codes, signals, and what is left listening.
//! (`docker stop` is a SIGTERM and a SIGKILL after the grace period; "no zombie" means that after an engine fault the process is gone
//! and its port refuses connections.)
//!
//! The mutation each test was checked against is named in its comment.

mod common;

use common::{chat_body, completion_body, http_timeout, sse, Conn};
use serde_json::json;
use std::io::{BufRead, BufReader};
use std::net::TcpStream;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const SEC: Duration = Duration::from_secs(1);

struct Proc {
    child: Child,
    port: u16,
    log: Arc<Mutex<String>>,
}

impl Proc {
    /// Start `rexl3-serve --backend mock` on a free port with `extra` arguments; returns once it is listening.
    fn start(extra: &[&str]) -> Proc {
        Self::start_in(&common::unstoppable_dir(), extra)
    }

    fn start_in(dir: &std::path::Path, extra: &[&str]) -> Proc {
        Self::start_mode(dir, extra, true)
    }
    fn start_mode(dir: &std::path::Path, extra: &[&str], ready: bool) -> Proc {
        Self::start_env(dir, extra, ready, &[])
    }
    /// As `start_mode`, with the environment variables in `env` set (`Some`) or removed (`None`) for the child.
    fn start_env(dir: &std::path::Path, extra: &[&str], ready: bool, env: &[(&str, Option<&str>)]) -> Proc {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_rexl3-serve"));
        for (k, v) in env {
            match v {
                Some(v) => cmd.env(k, v),
                None => cmd.env_remove(k),
            };
        }
        let mut child = cmd
            .args(["--model"])
            .arg(dir)
            // a KV pool big enough for several long requests at once (the mock charges 64 KiB a token, as the real model does)
            .args([
                "--backend",
                "mock",
                "--port",
                "0",
                "--served-model-name",
                "mock",
                "--max-model-len",
                "4096",
                "--kv-tokens",
                "100000",
            ])
            .args(extra)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start rexl3-serve");
        let log: Arc<Mutex<String>> = Default::default();
        let (l, err) = (log.clone(), child.stderr.take().unwrap());
        std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                let mut l = l.lock().unwrap();
                l.push_str(&line);
                l.push('\n');
            }
        });
        let mut p = Proc { child, port: 0, log };
        let t0 = Instant::now();
        loop {
            if let Some(port) = p
                .log()
                .lines()
                .find_map(|l| l.split("addr=http://127.0.0.1:").nth(1))
                .and_then(|r| r.split_whitespace().next()?.parse().ok())
            {
                p.port = port;
                break;
            }
            assert!(t0.elapsed() < 20 * SEC, "the server never started listening:\n{}", p.log());
            assert!(p.child.try_wait().unwrap().is_none(), "the server exited while starting:\n{}", p.log());
            std::thread::sleep(Duration::from_millis(20));
        }
        // ready: the engine thread has finished building
        let t0 = Instant::now();
        while ready && http_timeout(p.port, "GET", "/health", None, 5 * SEC).status != 200 {
            assert!(t0.elapsed() < 20 * SEC, "never ready:\n{}", p.log());
            std::thread::sleep(Duration::from_millis(20));
        }
        p
    }

    fn log(&self) -> String {
        self.log.lock().unwrap().clone()
    }

    fn sigterm(&self) {
        let ok = Command::new("kill").args(["-TERM", &self.child.id().to_string()]).status().unwrap();
        assert!(ok.success());
    }

    fn wait_exit(&mut self, within: Duration) -> Option<ExitStatus> {
        let t0 = Instant::now();
        loop {
            if let Some(s) = self.child.try_wait().unwrap() {
                return Some(s);
            }
            if t0.elapsed() >= within {
                return None;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn refuses_connections(&self) -> bool {
        TcpStream::connect_timeout(&format!("127.0.0.1:{}", self.port).parse().unwrap(), SEC).is_err()
    }

    fn metrics(&self) -> String {
        http_timeout(self.port, "GET", "/metrics", None, 5 * SEC).body
    }
}

impl Drop for Proc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A streaming chat request that stays busy for a long time.
fn long_stream(port: u16) -> Conn {
    let mut c = Conn::post(port, "/v1/chat/completions", &chat_body(json!({"stream": true, "max_tokens": 3000})));
    assert!(c.read_until(10 * SEC, |t| t.contains("data: ")), "the stream never started: {}", c.text());
    c
}

fn body_of(c: &Conn) -> String {
    common::dechunk(c.text().split_once("\r\n\r\n").map(|x| x.1).unwrap_or(""))
}

fn ended_as_an_error(body: &str, needle: &str) {
    let events = sse(body);
    assert!(
        events.iter().any(|e| e["error"]["message"].as_str().is_some_and(|m| m.contains(needle))),
        "no error event mentioning `{needle}` in:\n{body}"
    );
    assert!(
        !body.contains("\"finish_reason\":\"stop\"") && !body.contains("\"finish_reason\":\"length\""),
        "the reply ended as a normal finish:\n{body}"
    );
    // the stream closes after its error event: `[DONE]` is the end of a reply that finished, and a client that takes it for
    // completion must never read a cut reply as a whole one
    assert!(!body.contains("[DONE]"), "an errored stream ended with [DONE]:\n{body}");
}

// ------------------------------------------------------------------------------------------------ engine faults end the process

#[test]
fn a_panic_on_the_engine_thread_ends_the_process_non_zero_and_closes_the_port() {
    // mutation: delete the catch_unwind in runtime::spawn_with and the process keeps listening with a dead engine
    let mut p = Proc::start(&["--mock", "step_ms=10,panic_step=6,vocab=4096"]);
    let mut c = long_stream(p.port);
    let status = p.wait_exit(10 * SEC).unwrap_or_else(|| panic!("the process did not exit after an engine panic:\n{}", p.log()));
    assert_eq!(status.code(), Some(101), "{}", p.log());
    assert!(p.log().contains("injected panic at step 6") && p.log().contains("event=engine_failed"), "{}", p.log());
    c.read_to_end_of_body(5 * SEC);
    ended_as_an_error(&body_of(&c), "panicked");
    assert!(p.refuses_connections(), "a listener is left behind");
}

#[test]
fn a_fatal_backend_error_ends_the_process_with_70_and_every_stream_gets_an_error() {
    // mutation: ignore a fatal step error in runtime::run (`Err(e) if e.is_fatal()`) and the process stays up
    let mut p = Proc::start(&["--mock", "step_ms=10,fatal_step=6,vocab=4096"]);
    let mut streams = [long_stream(p.port), long_stream(p.port)];
    let status = p.wait_exit(10 * SEC).unwrap_or_else(|| panic!("the process did not exit after a fatal error:\n{}", p.log()));
    assert_eq!(status.code(), Some(70), "{}", p.log());
    for c in streams.iter_mut() {
        c.read_to_end_of_body(5 * SEC);
        ended_as_an_error(&body_of(c), "injected fatal");
    }
    assert!(p.refuses_connections());
}

#[test]
fn a_panic_while_the_engine_is_loading_ends_the_process_instead_of_serving_loading_forever() {
    // mutation: delete the catch_unwind and the engine thread dies silently behind a listener that answers 503 loading for good
    let (status, log) = run_to_exit(&common::fixture_dir(), &["--mock", "panic_boot=1"], 15 * SEC);
    let status = status.unwrap_or_else(|| panic!("still running after the engine failed to build:\n{log}"));
    assert_eq!(status.code(), Some(101), "{log}");
    assert!(log.contains("injected panic while building the engine"), "{log}");
}

/// Start the binary on `dir` with `extra` arguments and wait up to `within` for it to exit; a process still running then is killed
/// and reported as `None`. Returns its status and what it wrote to stderr.
fn run_to_exit(dir: &std::path::Path, extra: &[&str], within: Duration) -> (Option<ExitStatus>, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rexl3-serve"))
        .arg("--model")
        .arg(dir)
        .args(["--backend", "mock", "--port", "0"])
        .args(extra)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let log: Arc<Mutex<String>> = Default::default();
    let (l, err) = (log.clone(), child.stderr.take().unwrap());
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(err).lines().map_while(Result::ok) {
            let mut l = l.lock().unwrap();
            l.push_str(&line);
            l.push('\n');
        }
    });
    let t0 = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break Some(s);
        }
        if t0.elapsed() >= within {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let _ = reader.join();
    let log = log.lock().unwrap().clone();
    (status, log)
}

#[test]
fn consecutive_step_errors_end_the_process_for_the_supervisor_to_restart() {
    // mutation: drop the `exit_after` check
    let mut p = Proc::start(&["--mock", "retry_step=1,retry_count=100,vocab=4096", "--unready-after", "2", "--exit-after", "3"]);
    for _ in 0..3 {
        let r = http_timeout(p.port, "POST", "/v1/completions", Some(&completion_body(&[5, 6, 7], 10, false)), 10 * SEC);
        assert!(r.status == 500 || r.status == 503, "{}", r.body);
    }
    let status = p.wait_exit(10 * SEC).unwrap_or_else(|| panic!("no exit after 3 consecutive step errors:\n{}", p.log()));
    assert_eq!(status.code(), Some(70), "{}", p.log());
}

// ------------------------------------------------------------------------------------------------ SIGTERM

#[test]
fn an_idle_server_exits_zero_on_sigterm_at_once() {
    // (scripts/gpu_gate.sh stops its server with SIGTERM and waits: a stall here would hang the gate)
    let mut p = Proc::start(&[]);
    let t0 = Instant::now();
    p.sigterm();
    let status = p.wait_exit(5 * SEC).unwrap_or_else(|| panic!("an idle server ignored SIGTERM:\n{}", p.log()));
    assert_eq!(status.code(), Some(0), "{}", p.log());
    assert!(t0.elapsed() < 2 * SEC, "took {:?}", t0.elapsed());
    assert!(p.log().contains("event=shutdown_signal") && p.log().contains("event=drained"), "{}", p.log());
}

#[test]
fn sigterm_under_load_drains_inside_the_grace_period_and_marks_what_it_cut_off() {
    // mutation: no drain deadline (or no SIGTERM handler): the process outlives its grace period (docker stop would SIGKILL it)
    let grace = 3;
    let mut p = Proc::start(&[
        "--shutdown-grace",
        &grace.to_string(),
        "--slots",
        "2",
        "--queue-depth",
        "8",
        "--mock",
        "step_ms=20,vocab=4096",
    ]);
    let mut running = [long_stream(p.port), long_stream(p.port)];
    // (a stream's first bytes arrive when its request is queued, so wait until both really hold a slot)
    let t0 = Instant::now();
    while !p.metrics().lines().any(|l| l == "rexl3_active_sequences 2") {
        assert!(t0.elapsed() < 10 * SEC, "the two streams never both ran:\n{}", p.metrics());
        std::thread::sleep(Duration::from_millis(20));
    }
    // a third request has to wait for a slot: it is queued when the signal arrives
    let queued = {
        let port = p.port;
        std::thread::spawn(move || {
            http_timeout(port, "POST", "/v1/chat/completions", Some(&chat_body(json!({"max_tokens": 5}))), 30 * SEC)
        })
    };
    let t0 = Instant::now();
    while !p.metrics().lines().any(|l| l == "rexl3_queue_depth 1") {
        assert!(t0.elapsed() < 10 * SEC, "the third request never queued:\n{}", p.metrics());
        std::thread::sleep(Duration::from_millis(20));
    }
    let signalled = Instant::now();
    p.sigterm();
    // the queued request is answered 503 at once
    let r = queued.join().unwrap();
    assert_eq!(r.status, 503, "{}", r.body);
    assert_eq!(r.json()["error"]["type"], "server_shutting_down", "{}", r.body);
    assert!(signalled.elapsed() < 2 * SEC, "the queued request waited {:?}", signalled.elapsed());
    // while it drains, readiness is down and new work is refused
    let h = http_timeout(p.port, "GET", "/health", None, 5 * SEC);
    assert_eq!((h.status, h.body.as_str()), (503, "{\"status\":\"draining\"}"));
    let late = http_timeout(p.port, "POST", "/v1/chat/completions", Some(&chat_body(json!({}))), 5 * SEC);
    assert_eq!(late.status, 503, "{}", late.body);
    // the running replies cannot finish in the grace period: they end as errors, never as normal stops
    for c in running.iter_mut() {
        c.read_to_end_of_body(10 * SEC);
        ended_as_an_error(&body_of(c), "shut down");
    }
    // and the process is gone inside the grace period, with a clean status
    let status = p.wait_exit(Duration::from_secs(grace) + SEC).unwrap_or_else(|| panic!("still running:\n{}", p.log()));
    let took = signalled.elapsed();
    assert_eq!(status.code(), Some(0), "{}", p.log());
    assert!(took <= Duration::from_secs(grace), "exited {took:?} after SIGTERM, grace is {grace}s");
    assert!(took >= Duration::from_secs(1), "it must have waited for the running replies: {took:?}");
}

#[test]
fn replies_that_finish_inside_the_grace_period_finish_normally() {
    // mutation: make the drain deadline immediate and the reply is cut off
    let mut p = Proc::start(&["--shutdown-grace", "5", "--mock", "step_ms=20,vocab=4096"]);
    let mut c = Conn::post(p.port, "/v1/chat/completions", &chat_body(json!({"stream": true, "max_tokens": 40})));
    assert!(c.read_until(10 * SEC, |t| t.contains("data: ")));
    let signalled = Instant::now();
    p.sigterm();
    c.read_until(10 * SEC, |t| t.contains("[DONE]"));
    let body = body_of(&c);
    let events = sse(&body);
    let finish = events.iter().filter_map(|e| e["choices"][0]["finish_reason"].as_str()).collect::<Vec<_>>();
    assert_eq!(finish, vec!["length"], "{body}");
    assert!(events.iter().all(|e| e.get("error").is_none()), "{body}");
    let status = p.wait_exit(5 * SEC).unwrap_or_else(|| panic!("still running:\n{}", p.log()));
    assert_eq!(status.code(), Some(0), "{}", p.log());
    assert!(signalled.elapsed() < 4 * SEC, "it drained in {:?}; the reply took under 2 s", signalled.elapsed());
}

// ------------------------------------------------------------------------------------------------ startup validation

#[test]
fn a_model_folder_with_no_eos_token_is_a_startup_error() {
    // (release plan R: "missing EOS is an error": a reply could then only end at max_tokens)
    let dir = std::env::temp_dir().join(format!("rexl3-no-eos-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["tokenizer.json", "chat_template.jinja", "config.json"] {
        std::fs::copy(common::fixture_dir().join(f), dir.join(f)).unwrap();
    }
    std::fs::write(dir.join("generation_config.json"), "{\"temperature\": 1.0}").unwrap();
    let (status, log) = run_to_exit(&dir, &[], 10 * SEC);
    let status = status.unwrap_or_else(|| panic!("the server started although no reply could ever stop on its own:\n{log}"));
    assert_eq!(status.code(), Some(1), "{log}");
    assert!(log.contains("eos_token_id"), "{log}");
}

#[test]
fn a_kv_pool_below_the_window_is_the_limit_the_binary_reports_and_enforces() {
    // mutation: drop `.with_kv_tokens(kv_tokens)` from the mock engine in main.rs and the prompt below is admitted, and the models list says 4096
    let p = Proc::start(&["--kv-tokens", "1000", "--mock", "vocab=4096"]);
    let models = http_timeout(p.port, "GET", "/v1/models", None, 5 * SEC).json();
    assert_eq!(
        (models["data"][0]["max_model_len"].clone(), models["data"][0]["configured_max_model_len"].clone()),
        (json!(1000), json!(4096)),
        "{models}"
    );
    assert!(p.metrics().contains("rexl3_max_model_len 1000\n"), "{}", p.metrics());
    let t0 = Instant::now();
    let r = http_timeout(p.port, "POST", "/v1/completions", Some(&completion_body(&vec![5u32; 1300], 1, false)), 10 * SEC);
    assert_eq!(r.status, 400, "{}", r.body);
    assert_eq!(r.json()["error"]["code"], "context_length_exceeded", "{}", r.body);
    assert!(
        r.json()["error"]["message"].as_str().unwrap().starts_with("This model's maximum context length is 1000 tokens"),
        "{}",
        r.body
    );
    assert!(t0.elapsed() < 2 * SEC, "{:?}", t0.elapsed());
    let ok = http_timeout(p.port, "POST", "/v1/completions", Some(&completion_body(&vec![5u32; 900], 50, false)), 30 * SEC);
    assert_eq!(ok.status, 200, "{}", ok.body);
    assert_eq!(ok.json()["usage"]["completion_tokens"], 50);
}

#[test]
fn loading_sigterm_is_cooperative_joined_and_not_a_clean_exit_racing_boot() {
    for ms in ["boot_ms=60000", "boot_ms=80", "boot_ms=0"] {
        let mut p = Proc::start_mode(&common::unstoppable_dir(), &["--mock", ms, "--shutdown-grace", "2"], false);
        p.sigterm();
        assert_eq!(p.wait_exit(5 * SEC).unwrap().code(), Some(0), "{}", p.log());
        assert!(p.log().contains("event=drained"), "must join before claiming stopped: {}", p.log());
        assert!(!p.log().contains("event=engine_failed"), "{}", p.log());
        assert!(p.refuses_connections());
    }
}

#[test]
fn a_wedged_safe_boundary_hits_an_explicit_failure_watchdog() {
    use std::os::unix::process::ExitStatusExt;
    let mut p = Proc::start(&["--mock", "step_ms=60000,vocab=4096", "--shutdown-grace", "1"]);
    let _c = long_stream(p.port);
    std::thread::sleep(Duration::from_millis(200));
    p.sigterm();
    let status = p.wait_exit(7 * SEC).expect("watchdog must bound a genuinely wedged owner");
    assert_eq!(status.signal(), Some(6));
    assert!(p.log().contains("event=shutdown_watchdog"), "{}", p.log());
}

#[test]
fn a_progress_wedge_is_not_reported_as_healthy_forever() {
    use std::os::unix::process::ExitStatusExt;
    let mut p = Proc::start(&["--mock", "step_ms=5000,vocab=4096", "--heartbeat-stale-secs", "1"]);
    let _c = long_stream(p.port);
    let status = p.wait_exit(4 * SEC).expect("progress watchdog must bound a decode wedge");
    assert_eq!(status.signal(), Some(6));
    assert!(p.log().contains("event=progress_watchdog"), "{}", p.log());
}

/// What `/proc/PID/environ` of the running server says `name` is.
fn environ_of(p: &Proc, name: &str) -> Option<String> {
    let raw = std::fs::read(format!("/proc/{}/environ", p.child.id())).expect("the server's environment");
    raw.split(|b| *b == 0)
        .filter_map(|kv| std::str::from_utf8(kv).ok())
        .find_map(|kv| kv.strip_prefix(&format!("{name}=")).map(str::to_string))
}

/// Request memory (`docs/REQUEST_CONTRACT.md`): a server started without a malloc arena cap re-executes itself once with
/// `MALLOC_ARENA_MAX=2` (same pid: the supervisor's `MainPID`, the signals and the exit codes above are unaffected); a caller's own value,
/// or `REXL3_MALLOC_REEXEC=0`, keeps it from doing so. Without the re-execution the first case finds no variable in `/proc/PID/environ`.
#[test]
fn the_server_caps_its_malloc_arenas_unless_the_caller_named_a_value_or_opted_out() {
    let dir = common::unstoppable_dir();
    // nothing set: re-executed with the cap, same process
    let mut p = Proc::start_env(
        &dir,
        &["--mock", "step_ms=1,vocab=4096"],
        true,
        &[("MALLOC_ARENA_MAX", None), ("GLIBC_TUNABLES", None), ("REXL3_MALLOC_REEXEC", None)],
    );
    assert_eq!(environ_of(&p, "MALLOC_ARENA_MAX").as_deref(), Some("2"), "{}", p.log());
    assert_eq!(
        environ_of(&p, "REXL3_MALLOC_REEXEC").as_deref(),
        Some("0"),
        "the new image is marked: it never re-executes twice"
    );
    assert!(p.log().contains("event=reexec reason=malloc_arena_max value=2"), "{}", p.log());
    assert_eq!(p.log().matches("event=reexec reason").count(), 1, "{}", p.log());
    assert!(!p.log().contains("event=reexec_failed"), "{}", p.log());
    // a request is served after the re-exec, and a SIGTERM still ends it with 0 (the pid the supervisor holds is the server)
    let r = http_timeout(p.port, "POST", "/v1/chat/completions", Some(&chat_body(json!({"max_tokens": 2}))), 10 * SEC);
    assert_eq!(r.status, 200, "{}", r.body);
    p.sigterm();
    assert_eq!(p.wait_exit(10 * SEC).unwrap().code(), Some(0), "{}", p.log());
    // the caller's own value is kept and nothing is re-executed
    let mut own = Proc::start_env(
        &dir,
        &["--mock", "step_ms=1,vocab=4096"],
        true,
        &[("MALLOC_ARENA_MAX", Some("5")), ("REXL3_MALLOC_REEXEC", None)],
    );
    assert_eq!(environ_of(&own, "MALLOC_ARENA_MAX").as_deref(), Some("5"));
    assert!(!own.log().contains("event=reexec"), "{}", own.log());
    own.sigterm();
    own.wait_exit(10 * SEC);
    // the opt-out
    let mut off = Proc::start_env(
        &dir,
        &["--mock", "step_ms=1,vocab=4096"],
        true,
        &[("MALLOC_ARENA_MAX", None), ("GLIBC_TUNABLES", None), ("REXL3_MALLOC_REEXEC", Some("0"))],
    );
    assert_eq!(environ_of(&off, "MALLOC_ARENA_MAX"), None);
    assert!(!off.log().contains("event=reexec"), "{}", off.log());
    off.sigterm();
    off.wait_exit(10 * SEC);
}

/// `--version`, `--help`, `--print-fingerprints` and `--print-crc32` print and exit: scripts capture their stderr (`2>&1`), so the
/// arena re-execution must not add its line to it.
#[test]
fn the_modes_that_only_print_are_not_re_executed() {
    for args in [vec!["--version"], vec!["-V"], vec!["--help"], vec!["--print-crc32", "--model", "/nonexistent-model-dir"]] {
        let out = Command::new(env!("CARGO_BIN_EXE_rexl3-serve"))
            .args(&args)
            .env_remove("MALLOC_ARENA_MAX")
            .env_remove("GLIBC_TUNABLES")
            .env_remove("REXL3_MALLOC_REEXEC")
            .stdin(Stdio::null())
            .output()
            .expect("run rexl3-serve");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(!err.contains("event=reexec"), "{args:?}: {err}");
    }
}
