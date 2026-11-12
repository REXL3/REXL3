//! One upstream `rexl3-serve` instance (a single-node server or the leader of a tensor-parallel group) and its health,
//! admin and traffic state.
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Operator-controlled availability, independent of health: only `Enabled` backends take new requests; in-flight requests
/// always finish.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Admin {
    Enabled,
    /// No new requests; still probed so the operator sees it recover. `Backend::drained` says when it is idle.
    Draining,
    /// Out of rotation and not probed.
    Disabled,
}

impl Admin {
    pub fn name(self) -> &'static str {
        match self {
            Admin::Enabled => "enabled",
            Admin::Draining => "draining",
            Admin::Disabled => "disabled",
        }
    }
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Admin::Draining,
            2 => Admin::Disabled,
            _ => Admin::Enabled,
        }
    }
    fn to_u8(self) -> u8 {
        match self {
            Admin::Enabled => 0,
            Admin::Draining => 1,
            Admin::Disabled => 2,
        }
    }
}

/// How a response (or attempt) ended, for the error counters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ErrKind {
    /// The connection to the backend could not be made (or broke before response headers).
    Connect,
    /// The backend answered 503 because its engine is down, still loading or shutting down.
    Unavailable,
    /// The upstream body broke or stalled after response headers were sent to the client.
    Stream,
    /// The request deadline expired.
    Timeout,
}

/// Monotonic per-backend traffic counters (the router's `/metrics`).
#[derive(Default)]
pub struct Stats {
    /// Attempts dispatched to this backend (a failed-over request counts once per backend it was sent to).
    pub requests: AtomicU64,
    /// Responses by status class: 1xx..5xx at indices 0..5 (index 5 = anything else).
    pub responses: [AtomicU64; 6],
    pub err_connect: AtomicU64,
    pub err_unavailable: AtomicU64,
    pub err_stream: AtomicU64,
    pub err_timeout: AtomicU64,
    /// Requests that left this backend for another one before the first response byte.
    pub failovers_from: AtomicU64,
    pub to_up: AtomicU64,
    pub to_down: AtomicU64,
    pub probes_ok: AtomicU64,
    pub probes_failed: AtomicU64,
}

impl Stats {
    pub fn count_error(&self, kind: ErrKind) {
        let c = match kind {
            ErrKind::Connect => &self.err_connect,
            ErrKind::Unavailable => &self.err_unavailable,
            ErrKind::Stream => &self.err_stream,
            ErrKind::Timeout => &self.err_timeout,
        };
        c.fetch_add(1, Ordering::Relaxed);
    }

    pub fn count_status(&self, status: u16) {
        let i = match status / 100 {
            c @ 1..=5 => c as usize - 1,
            _ => 5,
        };
        self.responses[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// What the last health probe saw.
#[derive(Clone, Debug, Default)]
pub struct ProbeInfo {
    /// HTTP status of the last probe, 0 when the backend was unreachable or never probed.
    pub status: u16,
    pub error: Option<String>,
    pub at: Option<Instant>,
    pub body: Option<serde_json::Value>,
}

#[derive(Default)]
struct HealthCounters {
    consecutive_fail: u32,
    consecutive_ok: u32,
    last: ProbeInfo,
}

static NEXT_UID: AtomicU64 = AtomicU64::new(1);

/// A cloneable handle: all mutable state is shared, so a clone held by an in-flight request keeps working after the
/// backend is removed from the router.
#[derive(Clone)]
pub struct Backend {
    /// Process-unique id; the affinity map binds keys to this, never to a list index (the list changes at runtime).
    pub uid: u64,
    /// Base URL, e.g. `http://127.0.0.1:8001` (no trailing slash).
    pub url: String,
    /// Stable hash of the id, for rendezvous tie-breaking (independent of list order and of restarts).
    pub hash: u64,
    /// Requests dispatched to this backend that have not finished their
    /// response body yet.
    pub inflight: Arc<AtomicUsize>,
    /// 1 = healthy, 0 = down (probe failures or a passive failure on the data path).
    pub healthy: Arc<AtomicUsize>,
    admin: Arc<AtomicU8>,
    pub stats: Arc<Stats>,
    health: Arc<Mutex<HealthCounters>>,
}

impl Backend {
    /// A backend that starts healthy (the router's first probe round corrects that at boot).
    pub fn new(url: impl Into<String>) -> Self {
        let url: String = url.into();
        let url = url.trim_end_matches('/').to_string();
        let hash = crate::keying::fnv64(url.trim_start_matches("http://").trim_start_matches("https://").as_bytes());
        Self {
            uid: NEXT_UID.fetch_add(1, Ordering::Relaxed),
            url,
            hash,
            inflight: Arc::new(0.into()),
            healthy: Arc::new(1.into()),
            admin: Arc::new(AtomicU8::new(Admin::Enabled.to_u8())),
            stats: Arc::new(Stats::default()),
            health: Arc::new(Mutex::new(HealthCounters::default())),
        }
    }

    /// A backend added at runtime: down until its first probe succeeds.
    pub fn new_down(url: impl Into<String>) -> Self {
        let b = Self::new(url);
        b.healthy.store(0, Ordering::Relaxed);
        b
    }

    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed) == 1
    }

    pub fn admin(&self) -> Admin {
        Admin::from_u8(self.admin.load(Ordering::Relaxed))
    }

    pub fn set_admin(&self, a: Admin) {
        self.admin.store(a.to_u8(), Ordering::Relaxed);
    }

    /// Takes new requests: healthy and enabled.
    pub fn routable(&self) -> bool {
        self.is_healthy() && self.admin() == Admin::Enabled
    }

    /// Out of rotation by the operator and idle: safe to restart or remove.
    pub fn drained(&self) -> bool {
        self.admin() != Admin::Enabled && self.inflight.load(Ordering::Relaxed) == 0
    }

    /// Stable human-readable backend id used in error bodies and the
    /// metrics `backend` label.
    pub fn id(&self) -> String {
        self.url.trim_start_matches("http://").trim_start_matches("https://").trim_end_matches('/').to_string()
    }

    fn set_health(&self, up: bool) -> bool {
        let was = self.healthy.swap(up as usize, Ordering::Relaxed) == 1;
        if was != up {
            if up {
                self.stats.to_up.fetch_add(1, Ordering::Relaxed);
            } else {
                self.stats.to_down.fetch_add(1, Ordering::Relaxed);
            }
            return true;
        }
        false
    }

    /// First probe at boot or on add: the state is set outright (no threshold). Returns true when the state changed.
    pub fn apply_initial_probe(&self, ok: bool, info: ProbeInfo) -> bool {
        let mut h = self.health.lock().unwrap();
        h.consecutive_fail = u32::from(!ok);
        h.consecutive_ok = u32::from(ok);
        h.last = info;
        self.count_probe(ok);
        self.set_health(ok)
    }

    /// One background probe result: down after `fail_threshold` consecutive failures, up after `recover_threshold`
    /// consecutive successes. Returns `Some(up)` on a transition.
    pub fn apply_probe(&self, ok: bool, info: ProbeInfo, fail_threshold: u32, recover_threshold: u32) -> Option<bool> {
        let mut h = self.health.lock().unwrap();
        h.last = info;
        self.count_probe(ok);
        if ok {
            h.consecutive_fail = 0;
            h.consecutive_ok = h.consecutive_ok.saturating_add(1);
            if !self.is_healthy() && h.consecutive_ok >= recover_threshold.max(1) {
                self.set_health(true);
                return Some(true);
            }
        } else {
            h.consecutive_ok = 0;
            h.consecutive_fail = h.consecutive_fail.saturating_add(1);
            if self.is_healthy() && h.consecutive_fail >= fail_threshold.max(1) {
                self.set_health(false);
                return Some(false);
            }
        }
        None
    }

    /// A failure seen on the data path. `strong` (connection refused, engine unavailable) takes the backend down at once;
    /// a weaker signal counts toward the same consecutive-failure threshold as probes. Recovery is only ever through the
    /// prober, so a flapping backend does not take traffic on one lucky request.
    pub fn passive_failure(&self, strong: bool, fail_threshold: u32) -> bool {
        let mut h = self.health.lock().unwrap();
        h.consecutive_ok = 0;
        h.consecutive_fail = h.consecutive_fail.saturating_add(1);
        if strong || h.consecutive_fail >= fail_threshold.max(1) {
            return self.set_health(false);
        }
        false
    }

    /// A request completed against this backend: the failure streak ends.
    pub fn passive_success(&self) {
        self.health.lock().unwrap().consecutive_fail = 0;
    }

    pub fn probe_snapshot(&self) -> (ProbeInfo, u32, u32) {
        let h = self.health.lock().unwrap();
        (h.last.clone(), h.consecutive_fail, h.consecutive_ok)
    }

    fn count_probe(&self, ok: bool) {
        let c = if ok { &self.stats.probes_ok } else { &self.stats.probes_failed };
        c.fetch_add(1, Ordering::Relaxed);
    }
}

/// An in-flight reservation: the backend's `inflight` count is held until this is dropped (the response body ended, the
/// client went away, or the attempt failed).
pub struct Reservation {
    inflight: Option<Arc<AtomicUsize>>,
}

impl Reservation {
    pub fn take(backend: &Backend) -> Self {
        backend.inflight.fetch_add(1, Ordering::Relaxed);
        Self { inflight: Some(backend.inflight.clone()) }
    }

    pub fn release(&mut self) {
        if let Some(c) = self.inflight.take() {
            c.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.release();
    }
}
