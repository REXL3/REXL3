//! Shared proxy state: the backend list (runtime-mutable), backend selection (least-outstanding-requests with bounded
//! prefix affinity), the health prober and the router's own counters.

use crate::affinity::AffinityMap;
use crate::backend::{Admin, Backend, ProbeInfo, Reservation};
use crate::keying;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Everything tunable; `Default` is the production default of each flag in `main.rs`.
#[derive(Clone, Debug)]
pub struct Config {
    /// Time between health probe rounds.
    pub probe_interval: Duration,
    /// Deadline of one probe (connect + headers + body).
    pub probe_timeout: Duration,
    /// Consecutive failed probes before a healthy backend is marked down.
    pub fail_threshold: u32,
    /// Consecutive successful probes before a down backend is marked up.
    pub recover_threshold: u32,
    /// Path probed on every backend (200 = ready).
    pub probe_path: String,
    /// Most affinity bindings kept (least recently used evicted first).
    pub affinity_capacity: usize,
    /// A binding idle this long is forgotten.
    pub affinity_ttl: Duration,
    /// Most bytes of prompt text hashed into the default affinity key.
    pub affinity_prefix_bytes: usize,
    /// A bound backend is skipped (without re-binding) while it has more than this many more requests in flight than the
    /// least loaded healthy backend: one hot prompt cannot pile a whole fleet's load on one node.
    pub affinity_max_skew: usize,
    /// Request bodies up to this size are held whole, so a request that fails before the first response byte can be sent to
    /// another backend; a larger body is streamed through and such a request is not retried.
    pub replay_body_cap: usize,
    /// Failovers allowed per request (0 disables retry).
    pub max_failovers: usize,
    pub connect_timeout: Duration,
    /// Whole-request deadline, response body included.
    pub request_timeout: Duration,
    /// A response body that delivers nothing for this long is cut (a dead upstream must not hang a client).
    pub stream_idle_timeout: Duration,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            probe_interval: Duration::from_millis(2000),
            probe_timeout: Duration::from_millis(2000),
            fail_threshold: 3,
            recover_threshold: 2,
            probe_path: "/health/ready".to_string(),
            affinity_capacity: 100_000,
            affinity_ttl: Duration::from_secs(1800),
            affinity_prefix_bytes: 32 * 1024,
            affinity_max_skew: 16,
            replay_body_cap: 2 << 20,
            max_failovers: 1,
            connect_timeout: Duration::from_millis(3000),
            request_timeout: Duration::from_secs(3600),
            stream_idle_timeout: Duration::from_secs(120),
        }
    }
}

/// The router's own counters (per-backend ones live in `backend::Stats`).
#[derive(Default)]
pub struct RouterMetrics {
    /// Requests that reached the proxy path.
    pub requests: AtomicU64,
    /// Requests answered because no backend could take them.
    pub no_backend: AtomicU64,
    /// Requests re-sent to another backend before any response byte.
    pub failovers: AtomicU64,
    /// Clients that went away before the response body ended.
    pub client_disconnects: AtomicU64,
    /// Requests whose body was larger than the replay cap (streamed, not retryable).
    pub body_overflow: AtomicU64,
    pub affinity_hit: AtomicU64,
    pub affinity_miss: AtomicU64,
    pub affinity_overflow: AtomicU64,
    pub affinity_none: AtomicU64,
    pub key_header: AtomicU64,
    pub key_prompt: AtomicU64,
}

/// What the affinity map did for a selection.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Affinity {
    /// The request carried no key.
    None,
    /// The key was bound to a usable backend and it took the request.
    Hit,
    /// The key was new, expired, or bound to a backend that cannot take it: bound afresh.
    Miss,
    /// The key was bound to a usable but overloaded backend; a less loaded one took the request (binding kept).
    Overflow,
}

pub struct Picked {
    pub backend: Backend,
    pub reservation: Reservation,
    pub affinity: Affinity,
}

/// Shared proxy state.
#[derive(Clone)]
pub struct App {
    backends: Arc<RwLock<Vec<Backend>>>,
    pub client: reqwest::Client,
    /// Sticky affinity map: affinity key -> backend uid.
    affinity: Arc<Mutex<AffinityMap>>,
    pub cfg: Arc<Config>,
    pub metrics: Arc<RouterMetrics>,
    rr: Arc<AtomicU64>,
}

impl App {
    /// The default configuration (tests; the binary always builds from flags).
    #[cfg(test)]
    pub fn new(backends: Vec<Backend>) -> Self {
        Self::with_config(backends, Config::default())
    }

    pub fn with_config(backends: Vec<Backend>, cfg: Config) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(cfg.connect_timeout)
            .timeout(cfg.request_timeout)
            .tcp_keepalive(Some(Duration::from_secs(30)))
            // A rexl3-serve backend closes idle connections after `--header-read-timeout` (default 30 s). reqwest keeps idle
            // connections for 90 s by default, so a pooled connection could be one the backend had already closed and the
            // first request on it would fail. Drop them from the pool well before the backend does.
            .pool_idle_timeout(Duration::from_secs(20))
            .build()
            .expect("reqwest client");
        Self {
            backends: Arc::new(RwLock::new(backends)),
            client,
            affinity: Arc::new(Mutex::new(AffinityMap::new(cfg.affinity_capacity, cfg.affinity_ttl))),
            cfg: Arc::new(cfg),
            metrics: Arc::new(RouterMetrics::default()),
            rr: Arc::new(AtomicU64::new(0)),
        }
    }

    /// A snapshot of the backend list (handles share state with the live ones).
    pub fn backends(&self) -> Vec<Backend> {
        self.backends.read().unwrap().clone()
    }

    /// The backend with this id (`host:port`) or base URL.
    pub fn find(&self, id: &str) -> Option<Backend> {
        let want = id.trim_end_matches('/');
        self.backends.read().unwrap().iter().find(|b| b.id() == want || b.url == want).cloned()
    }

    /// Add a backend at runtime. It starts down and takes traffic after its first successful probe. `Err` carries the
    /// existing backend when the id is already present.
    pub fn add_backend(&self, url: &str) -> Result<Backend, Backend> {
        let candidate = Backend::new_down(url);
        let mut list = self.backends.write().unwrap();
        if let Some(existing) = list.iter().find(|b| b.url == candidate.url) {
            return Err(existing.clone());
        }
        list.push(candidate.clone());
        Ok(candidate)
    }

    /// Remove a backend: no new request is routed to it, in-flight ones finish (they hold their own handle), and its
    /// affinity bindings are dropped.
    pub fn remove_backend(&self, id: &str) -> Option<Backend> {
        let mut aff = self.affinity.lock().unwrap();
        let mut list = self.backends.write().unwrap();
        let want = id.trim_end_matches('/');
        let pos = list.iter().position(|b| b.id() == want || b.url == want)?;
        let b = list.remove(pos);
        aff.forget_backend(b.uid);
        Some(b)
    }

    /// `(entries, evicted by capacity, evicted by TTL)` of the affinity map.
    pub fn affinity_stats(&self) -> (usize, u64, u64) {
        let a = self.affinity.lock().unwrap();
        (a.len(), a.evicted_capacity, a.evicted_ttl)
    }

    /// Select a backend for a request carrying `key` (if any) and reserve its slot, never one in `exclude`.
    ///
    /// Affinity: a key bound to a routable backend returns that backend (unless it is overloaded past
    /// `affinity_max_skew`). Otherwise the routable backend with the fewest in-flight requests wins; ties go to the
    /// highest rendezvous score of the key (the same key lands on the same backend even after its binding was evicted, and
    /// distinct keys spread over equally loaded backends), or round-robin when there is no key. The key is bound to the
    /// winner. `None` when no backend is routable.
    ///
    /// The chosen backend's in-flight counter is incremented while the affinity lock is still held, so concurrent
    /// selections can never all observe the same stale minimum. The reservation is released when the response body ends
    /// (or the client goes away, or the attempt fails).
    pub fn select(&self, key: Option<u64>, exclude: &[u64]) -> Option<Picked> {
        let now = Instant::now();
        let mut aff = self.affinity.lock().unwrap();
        let list = self.backends.read().unwrap();
        let eligible = |b: &&Backend| b.routable() && !exclude.contains(&b.uid);
        let min_load = list.iter().filter(eligible).map(|b| b.inflight.load(Ordering::Relaxed)).min()?;

        let mut outcome = if key.is_some() { Affinity::Miss } else { Affinity::None };
        if let Some(k) = key {
            if let Some(uid) = aff.lookup(k, now) {
                if let Some(b) = list.iter().find(|b| b.uid == uid).filter(eligible) {
                    if b.inflight.load(Ordering::Relaxed) <= min_load.saturating_add(self.cfg.affinity_max_skew) {
                        aff.bind(k, uid, now);
                        let reservation = Reservation::take(b);
                        return Some(Picked { backend: b.clone(), reservation, affinity: Affinity::Hit });
                    }
                    outcome = Affinity::Overflow;
                }
            }
        }

        let score_key = key.unwrap_or_else(|| keying::mix64(self.rr.fetch_add(1, Ordering::Relaxed)));
        let chosen = list
            .iter()
            .filter(eligible)
            .filter(|b| b.inflight.load(Ordering::Relaxed) <= min_load)
            .max_by_key(|b| keying::rendezvous(score_key, b.hash))?;
        if let (Some(k), Affinity::Miss) = (key, outcome) {
            aff.bind(k, chosen.uid, now);
        }
        let reservation = Reservation::take(chosen);
        Some(Picked { backend: chosen.clone(), reservation, affinity: outcome })
    }

    /// Spawn the background health prober (one round per `probe_interval`, all backends concurrently).
    pub fn spawn_prober(&self) -> tokio::task::JoinHandle<()> {
        let app = self.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(app.cfg.probe_interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            tick.tick().await; // the first tick is immediate; boot already ran a round
            loop {
                tick.tick().await;
                app.probe_round(false).await;
            }
        })
    }

    /// Probe every non-disabled backend once. `initial` sets each backend's state from this one probe (no thresholds):
    /// used at boot, so a backend that is still loading is not routed to.
    pub async fn probe_round(&self, initial: bool) {
        let list = self.backends();
        let probes = list.iter().filter(|b| b.admin() != Admin::Disabled).map(|b| self.probe_one(b, initial));
        futures_util::future::join_all(probes).await;
    }

    /// One probe of `b` against `probe_path`, applied to its health state. Returns whether it answered 200.
    pub async fn probe_one(&self, b: &Backend, initial: bool) -> bool {
        let url = format!("{}{}", b.url, self.cfg.probe_path);
        let at = Instant::now();
        let (ok, info) = match self.client.get(&url).timeout(self.cfg.probe_timeout).send().await {
            Ok(resp) => {
                let status = resp.status().as_u16();
                let body = resp
                    .bytes()
                    .await
                    .ok()
                    .filter(|b| b.len() <= 64 * 1024)
                    .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
                // A tensor-parallel leader answers 200 for up to its 20 s exit backstop after it has lost a follower (nothing is calling
                // the lost runtime, so the engine has not failed yet): its body says so, and a group that is lost never serves again.
                let lost = body.as_ref().and_then(group_loss);
                (
                    status == 200 && lost.is_none(),
                    ProbeInfo { status, error: lost.map(|r| format!("tensor-parallel group lost: {r}")), at: Some(at), body },
                )
            }
            Err(e) => {
                let mut msg = e.to_string();
                msg.truncate(200);
                (false, ProbeInfo { status: 0, error: Some(msg), at: Some(at), body: None })
            }
        };
        let transition = if initial {
            let before = b.is_healthy();
            b.apply_initial_probe(ok, info.clone());
            (before != ok).then_some(ok)
        } else {
            b.apply_probe(ok, info.clone(), self.cfg.fail_threshold, self.cfg.recover_threshold)
        };
        if let Some(up) = transition {
            log_health(b, up, &format!("probe status={} {}", info.status, info.error.as_deref().unwrap_or("")));
        }
        ok
    }
}

/// Why the backend's tensor-parallel group is lost, from the `tensor_parallel` object of its `/health` body (`state` is `lost`).
pub fn group_loss(body: &serde_json::Value) -> Option<String> {
    let tp = body.get("tensor_parallel")?;
    if tp.get("state")?.as_str()? != "lost" {
        return None;
    }
    Some(tp.get("lost").and_then(|v| v.as_str()).unwrap_or("no reason given").chars().take(160).collect())
}

/// One stderr line per health transition (the server's `rexl3 event=...` style).
pub fn log_health(b: &Backend, up: bool, why: &str) {
    eprintln!(
        "rexl3-router event=health ts={} backend={} state={} {}",
        unix_ms(),
        b.id(),
        if up { "up" } else { "down" },
        why.trim()
    );
}

pub fn unix_ms() -> u128 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}
