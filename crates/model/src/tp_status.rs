//! The tensor-parallel group's status, family-neutral: what rank 0's lockstep leader knows of each follower and of the group, as a
//! snapshot a server can read without touching the runtime (`/health` and `/metrics`, `rexl3_tp_*`).
//!
//! The leader of a group registers a snapshot function once ([`register`], from its start); [`snapshot`] calls it. A process that is not
//! the leader of a group (a single process, a follower) has none and [`snapshot`] returns `None`. Every number is read from atomics and
//! one short lock, so the call is cheap enough for a scrape.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// One follower, as rank 0 sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RankStatus {
    pub rank: usize,
    /// The link is open and its reader is running.
    pub connected: bool,
    /// `up`, `lost` (this rank's link ended or it said it is leaving: the group is lost with it) or `closed` (the group was lost
    /// for another reason and this link was closed with it).
    pub state: &'static str,
    /// The highest command this rank has acknowledged.
    pub acked_seq: u64,
    /// The highest command it reports applied (pongs and acks).
    pub applied_seq: u64,
    /// Commands sent and not yet acknowledged by this rank.
    pub lag_commands: u64,
    /// Milliseconds since the last frame (an ack, a pong) from this rank.
    pub last_frame_age_ms: u64,
    /// The last ping's round trip in microseconds, once one has completed.
    pub ping_rtt_us: Option<u64>,
    /// Acks received from it.
    pub acks: u64,
    /// Acks whose result differed from rank 0's (the group is lost at the first one).
    pub diverged: u64,
    /// Frames queued for it and not yet written.
    pub queued_frames: u64,
    pub bytes_sent: u64,
    /// Why it left, when it said or its stream ended.
    pub left: Option<String>,
}

/// The group, as rank 0 sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupStatus {
    /// Ranks in the group, rank 0 included.
    pub ranks: usize,
    /// Commands sent so far (the leader's sequence number).
    pub commands_sent: u64,
    /// The highest command every follower has acknowledged and agreed on.
    pub settled: u64,
    /// Acks received from all followers.
    pub acks: u64,
    pub tolerated: u64,
    /// Why the group was lost, once it is.
    pub lost: Option<String>,
    /// The follower the loss is blamed on, when it is one.
    pub lost_rank: Option<usize>,
    /// Followers, rank 1 first.
    pub per_rank: Vec<RankStatus>,
}

type Provider = Arc<dyn Fn() -> GroupStatus + Send + Sync>;

static PROVIDER: Mutex<Option<Provider>> = Mutex::new(None);
static LOST: AtomicU64 = AtomicU64::new(0);

/// Registers the snapshot function of this process's group (a later leader replaces an earlier one).
pub fn register(f: Provider) {
    *PROVIDER.lock().unwrap_or_else(|p| p.into_inner()) = Some(f);
}

/// The group's status, if this process leads one.
pub fn snapshot() -> Option<GroupStatus> {
    let f = PROVIDER.lock().unwrap_or_else(|p| p.into_inner()).clone();
    f.map(|f| f())
}

/// Counts a group loss (called once per lost group, by its leader).
pub fn note_group_lost() {
    LOST.fetch_add(1, Ordering::Relaxed);
}

/// Groups this process has lost.
pub fn groups_lost() -> u64 {
    LOST.load(Ordering::Relaxed)
}
