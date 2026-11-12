//! `rexl3-router` — DP-fleet reverse proxy in front of N `rexl3-serve`
//! backends (single-node servers, or the leaders of tensor-parallel groups).
//!
//! The router is a pure dataplane shim: it picks a backend (least
//! outstanding requests, with prefix-affinity stickiness so each node's
//! prefix cache warms), forwards the request, and streams the response
//! back without buffering (the 62 MiB lesson in `crates/server/src/api.rs`).
//! It never touches model bits — a reply forwarded through the router is
//! byte-identical to the reply the backend would have sent directly, so the
//! `drafted == serial` invariant is untouched by construction and is pinned
//! by an end-to-end test on a mock backend.
//!
//! Behavior (`docs/CLUSTER_ROUTER.md` has the operator's view):
//!
//! - Routing: least-outstanding-requests, where a request counts as
//!   outstanding until its response body ends or its client leaves. A request
//!   carries an affinity key: the `x-affinity-key` header when the client
//!   sends one, else a hash of the request's cacheable prefix (system prompt +
//!   first user message of a chat completion, the prompt of a text
//!   completion). A key sticks to one backend (bounded LRU with an idle TTL)
//!   unless that backend is unroutable or overloaded past `--affinity-max-skew`.
//! - Health: a background prober hits every backend's `/health/ready` every
//!   `--probe-interval-ms`; K consecutive failures take it out, M consecutive
//!   successes bring it back, so a restarted group rejoins without restarting
//!   the router. The data path also reports: a refused connection or a 503
//!   "engine unavailable" takes a backend out at once. Aggregate `/health` is
//!   200 iff at least one backend is routable and reports each backend's
//!   probe state, admin state and in-flight count.
//! - Failover: a connect failure or an engine-unavailable 503 before any
//!   response byte is retried once on another routable backend (the request
//!   body is held whole up to `--replay-body-cap-bytes`; a larger body is
//!   streamed and not retried). After the first byte nothing is retried: a
//!   dying stream reaches the client as a broken stream, a transport failure
//!   as 502 with the backend id.
//! - Admin (`--admin-bind`, separate listener, optional `--admin-token`):
//!   list/add/remove/drain/disable/enable backends at runtime. In-flight
//!   requests always finish.
//! - `/metrics`: the router's own `rexl3_router_*` series, then every
//!   backend's series renamed `rexl3_router_<name>` with a `backend` label
//!   (`?scope=router` for the router's own only).

mod admin;
mod affinity;
mod app;
mod backend;
#[cfg(test)]
mod fleet_tests;
mod handlers;
mod keying;
#[cfg(test)]
mod tests;

use app::{App, Config};
use backend::Backend;
use std::time::Duration;

const USAGE: &str = "usage: rexl3-router --endpoint URL [--endpoint URL ...] [--bind IP:PORT (0.0.0.0:8080)]
  [--admin-bind IP:PORT (admin API off when absent)] [--admin-token TOKEN | env REXL3_ROUTER_ADMIN_TOKEN]
  [--probe-interval-ms N (2000)] [--probe-timeout-ms N (2000)] [--probe-fail-threshold K (3)] [--probe-recover-threshold M (2)]
  [--probe-path PATH (/health/ready)]
  [--affinity-capacity N (100000)] [--affinity-ttl-secs N (1800)] [--affinity-prefix-bytes N (32768)] [--affinity-max-skew N (16)]
  [--replay-body-cap-bytes N (2097152)] [--max-failovers N (1)]
  [--connect-timeout-ms N (3000)] [--request-timeout-secs N (3600)] [--stream-idle-timeout-secs N (120)]
  [--shutdown-grace-secs N (30)]";

struct Opts {
    endpoints: Vec<String>,
    bind: String,
    admin_bind: Option<String>,
    admin_token: Option<String>,
    shutdown_grace: Duration,
    cfg: Config,
}

fn parse_args(args: impl IntoIterator<Item = String>, env_token: Option<String>) -> Result<Opts, String> {
    let mut o = Opts {
        endpoints: Vec::new(),
        bind: "0.0.0.0:8080".to_string(),
        admin_bind: None,
        admin_token: env_token.filter(|t| !t.is_empty()),
        shutdown_grace: Duration::from_secs(30),
        cfg: Config::default(),
    };
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        fn num<T: std::str::FromStr>(name: &str, v: String) -> Result<T, String> {
            v.parse().map_err(|_| format!("{name} must be a number, got {v:?}"))
        }
        match arg.as_str() {
            "--endpoint" => o.endpoints.push(val("--endpoint")?),
            "--bind" => o.bind = val("--bind")?,
            "--admin-bind" => o.admin_bind = Some(val("--admin-bind")?),
            "--admin-token" => o.admin_token = Some(val("--admin-token")?).filter(|t| !t.is_empty()),
            "--probe-interval-ms" => {
                o.cfg.probe_interval = Duration::from_millis(num("--probe-interval-ms", val("--probe-interval-ms")?)?)
            }
            "--probe-timeout-ms" => {
                o.cfg.probe_timeout = Duration::from_millis(num("--probe-timeout-ms", val("--probe-timeout-ms")?)?)
            }
            "--probe-fail-threshold" => o.cfg.fail_threshold = num("--probe-fail-threshold", val("--probe-fail-threshold")?)?,
            "--probe-recover-threshold" => {
                o.cfg.recover_threshold = num("--probe-recover-threshold", val("--probe-recover-threshold")?)?
            }
            "--probe-path" => o.cfg.probe_path = val("--probe-path")?,
            "--affinity-capacity" => o.cfg.affinity_capacity = num("--affinity-capacity", val("--affinity-capacity")?)?,
            "--affinity-ttl-secs" => {
                o.cfg.affinity_ttl = Duration::from_secs(num("--affinity-ttl-secs", val("--affinity-ttl-secs")?)?)
            }
            "--affinity-prefix-bytes" => {
                o.cfg.affinity_prefix_bytes = num("--affinity-prefix-bytes", val("--affinity-prefix-bytes")?)?
            }
            "--affinity-max-skew" => o.cfg.affinity_max_skew = num("--affinity-max-skew", val("--affinity-max-skew")?)?,
            "--replay-body-cap-bytes" => o.cfg.replay_body_cap = num("--replay-body-cap-bytes", val("--replay-body-cap-bytes")?)?,
            "--max-failovers" => o.cfg.max_failovers = num("--max-failovers", val("--max-failovers")?)?,
            "--connect-timeout-ms" => {
                o.cfg.connect_timeout = Duration::from_millis(num("--connect-timeout-ms", val("--connect-timeout-ms")?)?)
            }
            "--request-timeout-secs" => {
                o.cfg.request_timeout = Duration::from_secs(num("--request-timeout-secs", val("--request-timeout-secs")?)?)
            }
            "--stream-idle-timeout-secs" => {
                o.cfg.stream_idle_timeout =
                    Duration::from_secs(num("--stream-idle-timeout-secs", val("--stream-idle-timeout-secs")?)?)
            }
            "--shutdown-grace-secs" => {
                o.shutdown_grace = Duration::from_secs(num("--shutdown-grace-secs", val("--shutdown-grace-secs")?)?)
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if o.endpoints.is_empty() {
        return Err("--endpoint required (repeatable)".to_string());
    }
    for e in &o.endpoints {
        if !e.starts_with("http://") {
            return Err(format!("--endpoint {e:?} must be http://host:port (TLS to backends is not compiled in)"));
        }
    }
    if o.cfg.probe_interval.is_zero() || o.cfg.probe_timeout.is_zero() {
        return Err("--probe-interval-ms and --probe-timeout-ms must be positive".to_string());
    }
    if let Some(addr) = &o.admin_bind {
        let sock: std::net::SocketAddr = addr.parse().map_err(|_| format!("--admin-bind {addr:?} must be IP:PORT"))?;
        if !sock.ip().is_loopback() && o.admin_token.is_none() {
            return Err("--admin-bind outside loopback requires --admin-token (or REXL3_ROUTER_ADMIN_TOKEN)".to_string());
        }
    }
    Ok(o)
}

async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
    let mut int = signal(SignalKind::interrupt()).expect("SIGINT handler");
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
    }
}

#[tokio::main]
async fn main() {
    let opts = match parse_args(std::env::args().skip(1), std::env::var("REXL3_ROUTER_ADMIN_TOKEN").ok()) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("rexl3-router: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    let backends = opts.endpoints.iter().map(|u| Backend::new(u.clone())).collect();
    let app = App::with_config(backends, opts.cfg.clone());

    // One probe round before any traffic: a backend that is still loading is not routed to.
    app.probe_round(true).await;
    for b in app.backends() {
        eprintln!("rexl3-router event=backend backend={} state={}", b.id(), if b.is_healthy() { "up" } else { "down" });
    }
    app.spawn_prober();

    if let Some(addr) = &opts.admin_bind {
        let listener = tokio::net::TcpListener::bind(addr).await.expect("bind admin");
        let router = admin::build_admin_router(app.clone(), opts.admin_token.clone());
        eprintln!(
            "rexl3-router admin on {addr} (token {})",
            if opts.admin_token.is_some() { "required" } else { "not set: loopback only" }
        );
        tokio::spawn(async move {
            axum::serve(listener, router).await.expect("serve admin");
        });
    }

    let router = handlers::build_router(app);
    let listener = tokio::net::TcpListener::bind(&opts.bind).await.expect("bind");
    eprintln!("rexl3-router on {} -> {}", opts.bind, opts.endpoints.join(", "));

    // SIGTERM/SIGINT: stop accepting, let in-flight responses finish for up to the grace period, then exit.
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = stop_tx.send(true);
    });
    let mut graceful_rx = stop_rx.clone();
    let serve = axum::serve(listener, router).with_graceful_shutdown(async move {
        let _ = graceful_rx.wait_for(|stopped| *stopped).await;
    });
    let mut grace_rx = stop_rx;
    let grace = opts.shutdown_grace;
    tokio::select! {
        r = serve => r.expect("serve"),
        _ = async move {
            let _ = grace_rx.wait_for(|stopped| *stopped).await;
            eprintln!("rexl3-router: shutdown requested, waiting up to {}s for in-flight responses", grace.as_secs());
            tokio::time::sleep(grace).await;
        } => eprintln!("rexl3-router: grace period over, exiting with responses still in flight"),
    }
}

#[cfg(test)]
mod arg_tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Opts, String> {
        parse_args(args.iter().map(|s| s.to_string()), None)
    }

    #[test]
    fn defaults_and_flags() {
        let o = parse(&["--endpoint", "http://a:1", "--endpoint", "http://b:2", "--probe-interval-ms", "500"]).ok().unwrap();
        assert_eq!(o.endpoints.len(), 2);
        assert_eq!(o.cfg.probe_interval, Duration::from_millis(500));
        assert_eq!(o.cfg.probe_path, "/health/ready");
        assert!(o.admin_bind.is_none());
    }

    #[test]
    fn refuses_bad_input() {
        assert!(parse(&[]).is_err(), "an endpoint is required");
        assert!(parse(&["--endpoint", "a:1"]).is_err(), "endpoints are URLs");
        assert!(parse(&["--endpoint", "https://a:1"]).is_err(), "no TLS client is compiled in");
        assert!(parse(&["--endpoint", "http://a:1", "--nope"]).is_err());
        assert!(parse(&["--endpoint", "http://a:1", "--probe-fail-threshold", "x"]).is_err());
    }

    #[test]
    fn admin_off_loopback_needs_a_token() {
        assert!(parse(&["--endpoint", "http://a:1", "--admin-bind", "0.0.0.0:9090"]).is_err());
        assert!(parse(&["--endpoint", "http://a:1", "--admin-bind", "0.0.0.0:9090", "--admin-token", "s3cret"]).is_ok());
        assert!(parse(&["--endpoint", "http://a:1", "--admin-bind", "127.0.0.1:9090"]).is_ok());
    }
}
