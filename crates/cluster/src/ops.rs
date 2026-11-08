//! `up`, `down` and `status` of one group. All state is on the nodes (pid, exit and state files of each rank), so these functions can
//! be run from any host and again after an interrupted run; they act by PID, and only on a pid whose command line holds the binary.
//!
//! The seams: [`Env`] (transport, HTTP, clock, cancellation) is real in the binary and simulated in the tests.

use crate::http;
use crate::schema::{Cluster, Group};
use crate::script::{self, RankState, StatusOpts};
use crate::transport::Transport;
use std::collections::BTreeMap;
use std::io::Write;
use std::time::Duration;

/// Everything outside the cluster file that the operations touch.
pub trait Env {
    fn transport(&self) -> &(dyn Transport + Sync);
    /// `GET /health` or `/health/ready` of a rank 0.
    fn http_get(&self, host: &str, port: u16, path: &str) -> Result<http::Response, String>;
    fn sleep(&self, d: Duration);
    /// Milliseconds on a monotonic clock.
    fn now_ms(&self) -> u64;
    /// The operator interrupted (SIGINT or SIGTERM to the launcher).
    fn cancelled(&self) -> bool;
}

const SSH_TIMEOUT: Duration = Duration::from_secs(60);

/// How long the servers of a group have to appear (their pid files) after the start barrier is released: a moment, they are exec'd.
const START_AFTER_BARRIER_MS: u64 = 60_000;

/// The ranks of a group, as found on the nodes, and the per-node facts that came with them.
#[derive(Debug, Clone, Default)]
pub struct GroupStatus {
    /// Index = rank. A node that could not be reached gives `state = "unreachable"`.
    pub ranks: Vec<RankState>,
    /// `lock`, `gpu_apps`, `serve_procs`, `mem_available_kib` per node name (when asked for).
    pub nodes: BTreeMap<String, BTreeMap<String, String>>,
    pub errors: Vec<String>,
}

impl GroupStatus {
    /// The ranks with anything alive: a server, or a run script still waiting for its node's GPU lock.
    pub fn running(&self) -> Vec<usize> {
        self.ranks.iter().filter(|r| r.active()).map(|r| r.rank).collect()
    }

    pub fn all_running(&self) -> bool {
        !self.ranks.is_empty() && self.ranks.iter().all(|r| r.running())
    }
}

/// Read the state of every rank (one ssh session a node, in parallel).
pub fn group_status(env: &dyn Env, c: &Cluster, g: &Group, o: StatusOpts) -> GroupStatus {
    let order = c.lock_order(g);
    let t = env.transport();
    let results: Vec<(usize, String, Result<crate::transport::Exec, String>)> = std::thread::scope(|s| {
        let hs: Vec<_> = order
            .iter()
            .map(|(rank, node)| {
                let sc = script::status_script(c, g, &[*rank], node, o);
                s.spawn(move || (*rank, node.name.clone(), t.run(node, &sc, SSH_TIMEOUT)))
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("status thread")).collect()
    });
    let mut st = GroupStatus { ranks: vec![RankState::default(); g.ranks.len()], ..Default::default() };
    for (i, r) in st.ranks.iter_mut().enumerate() {
        r.group = g.name.clone();
        r.rank = i;
        r.state = "unknown".into();
    }
    for (rank, node, res) in results {
        match res {
            Ok(e) if e.status == 0 => {
                if let Some(rs) = script::parse_rank_lines(&e.stdout).into_iter().find(|r| r.rank == rank) {
                    st.ranks[rank] = rs;
                }
                st.nodes.insert(node, script::parse_facts(&e.stdout));
            }
            Ok(e) => {
                st.ranks[rank].state = "unreachable".into();
                st.errors.push(format!(
                    "node {node}: status script exited {}: {}",
                    e.status,
                    e.stderr.lines().last().unwrap_or("")
                ));
            }
            Err(e) => {
                st.ranks[rank].state = "unreachable".into();
                st.errors.push(format!("node {node}: {e}"));
            }
        }
    }
    st
}

fn secs(ms: u64) -> String {
    format!("{:.1} s", ms as f64 / 1000.0)
}

/// SIGTERM/SIGKILL of one rank by pid; returns whether a live matching process was signalled.
fn signal(env: &dyn Env, c: &Cluster, g: &Group, rank: usize, sig: &str) -> bool {
    let Some(node) = c.rank_node(g, rank) else { return false };
    match env.transport().run(node, &script::signal_script(c, g, rank, sig), SSH_TIMEOUT) {
        Ok(e) => e.stdout.contains("signalled"),
        Err(_) => false,
    }
}

fn log_tail(env: &dyn Env, c: &Cluster, g: &Group, rank: usize, n: usize) -> String {
    let Some(node) = c.rank_node(g, rank) else { return String::new() };
    match env.transport().run(node, &script::tail_script(c, g, rank, n), SSH_TIMEOUT) {
        Ok(e) => e.stdout,
        Err(e) => format!("(no log: {e})"),
    }
}

/// Stop the ranks of the group that are still running (`only`: just those ranks, so that an `up` that failed stops what IT started and
/// never a group that was already up): SIGTERM, rank 0 first; after `term_wait`, SIGKILL. Returns the ranks that needed the KILL and
/// how long it took.
pub fn stop_all(
    env: &dyn Env,
    c: &Cluster,
    g: &Group,
    term_wait: Duration,
    only: Option<&[usize]>,
    out: &mut dyn Write,
) -> (Vec<usize>, u64) {
    let t0 = env.now_ms();
    let wanted = |r: &usize| only.is_none_or(|o| o.contains(r));
    let st = group_status(env, c, g, StatusOpts::default());
    let mut live: Vec<usize> = st.running().into_iter().filter(wanted).collect();
    for r in &live {
        let _ = writeln!(out, "  SIGTERM rank {r} (node {})", c.rank_node(g, *r).map(|n| n.name.as_str()).unwrap_or("?"));
        signal(env, c, g, *r, "TERM");
        // rank 0 first and a moment for it: its Shutdown is what makes a follower exit 0
        if *r == 0 {
            env.sleep(Duration::from_millis(500));
        }
    }
    let deadline = env.now_ms() + term_wait.as_millis() as u64;
    while !live.is_empty() && env.now_ms() < deadline {
        env.sleep(Duration::from_millis(500));
        live = group_status(env, c, g, StatusOpts::default()).running().into_iter().filter(wanted).collect();
    }
    let mut killed = Vec::new();
    for r in &live {
        let _ = writeln!(out, "  SIGKILL rank {r} (still running after {} s)", term_wait.as_secs());
        signal(env, c, g, *r, "KILL");
        killed.push(*r);
    }
    if !killed.is_empty() {
        env.sleep(Duration::from_secs(2));
    }
    (killed, env.now_ms() - t0)
}

#[derive(Debug, Clone, Default)]
pub struct UpReport {
    /// (rank, node, pid, ms after the start of `up` at which its GPU lock was held)
    pub ranks: Vec<(usize, String, u32, u64)>,
    pub ready_after_ms: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct UpOptions {
    /// Override of `max_runtime_s` for this run.
    pub budget_s: Option<u64>,
    /// How long to wait for rank 0's `/health/ready` after the last rank is started.
    pub ready_timeout: Duration,
}

/// `up --dry-run`: the exact session of each node, in the order the locks are taken. With `with_check` the pre-flight scripts come first.
pub fn up_dry_run(c: &Cluster, g: &Group, budget_s: Option<u64>, with_check: bool) -> String {
    let mut s = String::new();
    let order: Vec<String> = c.lock_order(g).iter().map(|(r, n)| format!("{} (rank {r})", n.name)).collect();
    s.push_str(&format!("# up: cluster {} group {} ({} ranks)\n", c.name, g.name, g.ranks.len()));
    if g.ranks.len() > 1 {
        s.push_str(&format!(
            "# GPU locks are taken in this order; each rank holds its lock and WAITS at a start barrier: {}\n",
            order.join(", ")
        ));
        s.push_str("# once every lock is held the launcher releases the barrier (the go step below, same order) and the servers start together\n");
    } else {
        s.push_str(&format!(
            "# GPU locks are taken in this order, each rank started once its lock is held: {}\n",
            order.join(", ")
        ));
    }
    let leader = c.rank_node(g, 0).map(|n| n.address.as_str()).unwrap_or("?");
    if g.ranks.len() > 1 {
        s.push_str(&format!("# rendezvous: --tp-peer {} on every rank; leader http://{leader}:{}\n", c.tp_peer(g), g.http_port));
    } else {
        s.push_str(&format!("# one rank: a plain single-node server, no group flags; http://{leader}:{}\n", g.http_port));
    }
    if with_check {
        s.push_str(&crate::check::dry_run(c, g));
    }
    for (rank, node) in c.lock_order(g) {
        s.push_str(&crate::transport::describe(
            &c.ssh_options,
            node,
            &format!("start rank {rank}"),
            &script::launch_script(c, g, rank, budget_s),
        ));
    }
    if g.ranks.len() > 1 {
        for (rank, node) in c.lock_order(g) {
            s.push_str(&crate::transport::describe(
                &c.ssh_options,
                node,
                &format!("release the start barrier of rank {rank} (once every rank reports `locked`)"),
                &script::go_script(c, g, rank),
            ));
        }
    }
    s
}

/// Why a start did not complete. `exits` is what each rank had done (its exit status, or `None` while it ran or was never started) when
/// the failure was seen, before the ranks were stopped: the evidence a supervisor classifies.
#[derive(Debug, Clone, Default)]
pub struct UpFailure {
    pub message: String,
    pub exits: Vec<Option<i32>>,
    /// A node's GPU lock was not granted (another session holds it): not a fault of the group.
    pub lock_busy: bool,
    pub interrupted: bool,
}

impl From<String> for UpFailure {
    fn from(message: String) -> Self {
        UpFailure { message, ..Default::default() }
    }
}

impl From<&str> for UpFailure {
    fn from(message: &str) -> Self {
        UpFailure { message: message.to_string(), ..Default::default() }
    }
}

impl std::fmt::Display for UpFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

fn exits_of(st: &GroupStatus) -> Vec<Option<i32>> {
    st.ranks.iter().map(|r| if r.running() { None } else { r.exit }).collect()
}

/// Start rank r of R on each node, in node-list order (the GPU-lock order), and wait for the group to be ready. On ANY failure or an
/// interrupt every rank that was started is stopped by pid before this returns (the exit trap of the maintainers' two-node driver).
///
/// A group of more than one rank starts in two phases: each node's run script takes its GPU lock (in lock order, lowest node first) and
/// holds it at a barrier; when EVERY lock is held the launcher releases the barrier on each node and the servers start together, so that
/// no rank's `--tp-connect-timeout` runs while another rank still queues for its lock. A failure before the barrier starts nothing and
/// releases every lock.
pub fn up(env: &dyn Env, c: &Cluster, g: &Group, o: UpOptions, out: &mut dyn Write) -> Result<UpReport, UpFailure> {
    let t0 = env.now_ms();
    let order = c.lock_order(g);
    let mut report = UpReport::default();
    let mut started: Vec<usize> = Vec::new();
    // Two-phase start (a group of more than one rank): every rank's run script takes its node's GPU lock, in lock order, and WAITS at a
    // barrier holding it; only when every lock is held does the launcher release the barrier, and the servers start together. Rank 0's
    // `--tp-connect-timeout` clock used to run from the moment node 1's lock was granted while the launcher queued on node 4's.
    let two_phase = g.ranks.len() > 1;
    let mut locked_at: BTreeMap<usize, u64> = BTreeMap::new();
    let result = (|| -> Result<(), UpFailure> {
        for (rank, node) in &order {
            let _ = writeln!(
                out,
                "  rank {rank} -> node {} ({}): taking {}{}",
                node.name,
                node.address,
                node.gpu_lock,
                if two_phase { ", then waiting for the other ranks' locks" } else { " then starting the server" }
            );
            let launch = script::launch_script(c, g, *rank, o.budget_s);
            let e = match env.transport().run(node, &launch, SSH_TIMEOUT) {
                Ok(e) => e,
                Err(m) => {
                    // ambiguous (a timeout after the script was sent): it may have started, so the cleanup covers this rank
                    started.push(*rank);
                    return Err(m.into());
                }
            };
            if e.stdout.contains("already_running") {
                // not ours: the cleanup below must not stop a group that was up before this `up`
                return Err(format!(
                    "node {}: rank {rank} is already running ({}): run `rexl3-cluster down` first",
                    node.name,
                    e.stdout.trim()
                )
                .into());
            }
            started.push(*rank);
            if e.status != 0 || !e.stdout.contains("launched") {
                return Err(format!(
                    "node {}: could not start rank {rank} (status {}): {}{}",
                    node.name,
                    e.status,
                    e.stdout.trim(),
                    e.stderr.trim()
                )
                .into());
            }
            // the lock is held once the state says `locked` (a group: the run script then waits at the barrier) or the pid file
            // appears (a single rank starts at once): this is what makes the lock order strict
            let lock_deadline = env.now_ms() + (c.lock_wait_s + 20) * 1000;
            loop {
                if env.cancelled() {
                    return Err(UpFailure { message: "interrupted".into(), interrupted: true, ..Default::default() });
                }
                let st = group_status(env, c, g, StatusOpts::default());
                let r = &st.ranks[*rank];
                if r.state == "barrier_timeout" {
                    return Err(UpFailure {
                        message: format!(
                            "node {}: rank {rank} gave up at the start barrier (the other ranks' locks were not granted in time, or no \
                             launcher answered for {} s)",
                            node.name,
                            script::BARRIER_BEAT_STALE_S
                        ),
                        exits: exits_of(&st),
                        lock_busy: true,
                        ..Default::default()
                    });
                }
                if r.state == "lock_timeout" {
                    return Err(UpFailure {
                        message: format!(
                            "node {}: the GPU lock {} was not free within {} s (another session holds it)",
                            node.name, node.gpu_lock, c.lock_wait_s
                        ),
                        exits: exits_of(&st),
                        lock_busy: true,
                        ..Default::default()
                    });
                }
                if r.state == "env_file_missing" {
                    return Err(UpFailure {
                        message: format!(
                            "node {}: the NCCL env file {} is missing",
                            node.name,
                            c.nccl.env_file.as_deref().unwrap_or("")
                        ),
                        exits: exits_of(&st),
                        ..Default::default()
                    });
                }
                if let Some(code) = r.exit {
                    return Err(UpFailure {
                        message: format!(
                            "rank {rank} on node {} exited {code} right after the start:\n{}",
                            node.name,
                            log_tail(env, c, g, *rank, 15)
                        ),
                        exits: exits_of(&st),
                        ..Default::default()
                    });
                }
                if two_phase && r.state == "locked" && r.exit.is_none() {
                    let at = env.now_ms() - t0;
                    let _ = writeln!(
                        out,
                        "  rank {rank}: GPU lock of node {} held, waiting at the barrier ({})",
                        node.name,
                        secs(at)
                    );
                    locked_at.insert(*rank, at);
                    break;
                }
                if let (Some(pid), true) = (r.pid, r.running()) {
                    let at = env.now_ms() - t0;
                    let _ = writeln!(out, "  rank {rank}: lock held and server started, pid {pid} ({})", secs(at));
                    report.ranks.push((*rank, node.name.clone(), pid, at));
                    break;
                }
                if env.now_ms() > lock_deadline {
                    return Err(UpFailure {
                        message: format!("node {}: rank {rank} did not start within {} s", node.name, c.lock_wait_s + 20),
                        exits: exits_of(&st),
                        ..Default::default()
                    });
                }
                env.sleep(Duration::from_millis(500));
            }
        }
        if two_phase {
            // Phase 2: every lock is held, nothing has started. Re-read the states (a rank that refused meanwhile must stop the start
            // while nothing runs), then release the barrier on every node, lowest node first.
            let st = group_status(env, c, g, StatusOpts::default());
            for r in &st.ranks {
                if r.exit.is_some() || r.state == "env_file_missing" || r.state == "lock_timeout" || r.state == "barrier_timeout"
                {
                    let node = c.rank_node(g, r.rank).map(|n| n.name.as_str()).unwrap_or("?");
                    return Err(UpFailure {
                        message: format!(
                            "rank {} on node {node} stopped before the start barrier (exit {}, state {}):\n{}",
                            r.rank,
                            r.exit.map(|e| e.to_string()).unwrap_or_else(|| "none".into()),
                            r.state,
                            log_tail(env, c, g, r.rank, 15)
                        ),
                        exits: exits_of(&st),
                        lock_busy: r.state == "lock_timeout" || r.state == "barrier_timeout",
                        ..Default::default()
                    });
                }
            }
            if env.cancelled() {
                return Err(UpFailure { message: "interrupted".into(), interrupted: true, ..Default::default() });
            }
            let held: Vec<String> =
                order.iter().map(|(r, n)| format!("{} at {}", n.name, secs(locked_at.get(r).copied().unwrap_or(0)))).collect();
            let _ = writeln!(
                out,
                "  every GPU lock is held ({}): releasing the barrier, the servers start together",
                held.join(", ")
            );
            for (rank, node) in &order {
                let e = env.transport().run(node, &script::go_script(c, g, *rank), SSH_TIMEOUT)?;
                if !e.stdout.contains("go rank") {
                    return Err(format!(
                        "node {}: rank {rank} is not waiting at the start barrier (its run script is gone): {}{}",
                        node.name,
                        e.stdout.trim(),
                        e.stderr.trim()
                    )
                    .into());
                }
            }
            // Phase 3: the servers start (their pid files), within a moment
            let start_deadline = env.now_ms() + START_AFTER_BARRIER_MS;
            loop {
                if env.cancelled() {
                    return Err(UpFailure { message: "interrupted".into(), interrupted: true, ..Default::default() });
                }
                let st = group_status(env, c, g, StatusOpts::default());
                if let Some(r) = st.ranks.iter().find(|r| r.exit.is_some() && !r.running()) {
                    return Err(UpFailure {
                        message: format!(
                            "rank {} on node {} exited {} right after the start barrier:\n{}",
                            r.rank,
                            c.rank_node(g, r.rank).map(|n| n.name.as_str()).unwrap_or("?"),
                            r.exit.unwrap_or(-1),
                            log_tail(env, c, g, r.rank, 15)
                        ),
                        exits: exits_of(&st),
                        ..Default::default()
                    });
                }
                if st.ranks.iter().all(|r| r.pid.is_some() && r.running()) {
                    for (rank, node) in &order {
                        let pid = st.ranks[*rank].pid.unwrap_or(0);
                        let _ = writeln!(out, "  rank {rank}: server started on node {}, pid {pid}", node.name);
                        report.ranks.push((*rank, node.name.clone(), pid, locked_at.get(rank).copied().unwrap_or(0)));
                    }
                    break;
                }
                if env.now_ms() > start_deadline {
                    return Err(UpFailure {
                        message: format!("the servers did not start within {} s of the barrier", START_AFTER_BARRIER_MS / 1000),
                        exits: exits_of(&st),
                        ..Default::default()
                    });
                }
                env.sleep(Duration::from_millis(200));
            }
        }
        // ready: rank 0's /health/ready answers 200, and no rank has exited
        let leader = c.rank_node(g, 0).ok_or("no rank 0")?;
        let _ = writeln!(out, "  waiting for http://{}:{}/health/ready", leader.address, g.http_port);
        let deadline = env.now_ms() + o.ready_timeout.as_millis() as u64;
        let mut last_note = 0;
        let mut unreachable_polls = 0;
        loop {
            if env.cancelled() {
                return Err(UpFailure { message: "interrupted".into(), interrupted: true, ..Default::default() });
            }
            let st = group_status(env, c, g, StatusOpts::default());
            // a lost ssh session is not a lost rank: tolerate a few polls
            if st.ranks.iter().any(|r| r.state == "unreachable") {
                unreachable_polls += 1;
                if unreachable_polls > 5 {
                    return Err(UpFailure {
                        message: format!("a node stayed unreachable for {unreachable_polls} polls: {}", st.errors.join("; ")),
                        exits: exits_of(&st),
                        ..Default::default()
                    });
                }
            } else {
                unreachable_polls = 0;
                for r in &st.ranks {
                    if !r.running() {
                        let tail = log_tail(env, c, g, r.rank, 15);
                        return Err(UpFailure {
                            message: format!(
                                "rank {} on node {} is not running before the group is ready (exit {}, state {}):\n{tail}",
                                r.rank,
                                c.rank_node(g, r.rank).map(|n| n.name.as_str()).unwrap_or("?"),
                                r.exit.map(|e| e.to_string()).unwrap_or_else(|| "none".into()),
                                r.state
                            ),
                            exits: exits_of(&st),
                            ..Default::default()
                        });
                    }
                }
            }
            if let Ok(r) = env.http_get(&leader.address, g.http_port, "/health/ready") {
                if r.status == 200 {
                    report.ready_after_ms = env.now_ms() - t0;
                    return Ok(());
                }
            }
            let now = env.now_ms();
            if now > deadline {
                return Err(UpFailure {
                    message: format!("the group was not ready within {} s", o.ready_timeout.as_secs()),
                    exits: exits_of(&st),
                    ..Default::default()
                });
            }
            if now - t0 > last_note + 30_000 {
                last_note = now - t0;
                let _ = writeln!(out, "  ... {} elapsed, not ready yet", secs(now - t0));
            }
            env.sleep(Duration::from_secs(1));
        }
    })();
    match result {
        Ok(()) => Ok(report),
        Err(mut f) => {
            if f.exits.is_empty() {
                f.exits = exits_of(&group_status(env, c, g, StatusOpts::default()));
            }
            let _ = writeln!(out, "  start failed: {}", f.message);
            let _ = writeln!(out, "  stopping every rank started so far (by pid):");
            let (killed, took) = stop_all(env, c, g, Duration::from_secs(c.tp.shutdown_grace_s + 10), Some(&started), out);
            let _ = writeln!(out, "  stopped in {}{}", secs(took), if killed.is_empty() { "" } else { " (some needed SIGKILL)" });
            Err(f)
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DownReport {
    /// Per rank: the exit status the server left, if it was running at the start.
    pub exits: Vec<(usize, Option<i32>)>,
    pub killed: Vec<usize>,
    pub was_running: Vec<usize>,
    pub total_ms: u64,
    /// Per node: the GPU lock is free and the GPU has no compute app.
    pub verified: Vec<(String, bool, String)>,
}

impl DownReport {
    /// Everything that was running exited 0 by itself after rank 0's SIGTERM, and every node is verified clean.
    pub fn clean(&self) -> bool {
        self.killed.is_empty()
            && self.exits.iter().filter(|(r, _)| self.was_running.contains(r)).all(|(_, e)| *e == Some(0))
            && self.verified.iter().all(|(_, ok, _)| *ok)
    }
}

/// Is this node clean of the group? No rank of it running, and the GPU lock not held by a process of this group. A lock (or a GPU) that
/// another session took the instant ours was released is that session's, and is said so, not counted against `down`.
fn node_clean(st: &GroupStatus, rank: usize, node: &str) -> (bool, String) {
    let Some(f) = st.nodes.get(node) else { return (false, "node not reachable".to_string()) };
    let mut bad = Vec::new();
    let mut notes = Vec::new();
    if st.ranks.get(rank).map(|r| r.active()).unwrap_or(false) {
        bad.push("a rank is still running".to_string());
    }
    match (f.get("lock").map(String::as_str), f.get("lock_ours").map(String::as_str)) {
        (Some("free"), _) => notes.push("GPU lock free".to_string()),
        (Some("held"), Some("1")) => bad.push(format!(
            "the GPU lock is still held by this group ({})",
            f.get("lock_holders").map(String::as_str).unwrap_or("?")
        )),
        (Some("held"), _) => notes.push(format!(
            "the GPU lock is held by another session ({}), not by this group",
            f.get("lock_holders").map(String::as_str).unwrap_or("?").trim_end_matches([';', ' '])
        )),
        _ => bad.push("the GPU lock state is unknown".to_string()),
    }
    let apps = f.get("gpu_apps").map(String::as_str).unwrap_or("?");
    let procs = f.get("serve_procs").map(String::as_str).unwrap_or("?");
    if apps != "0" || procs != "0" {
        notes.push(format!("{apps} compute app(s) and {procs} rexl3-serve process(es) on the node belong to other sessions"));
    }
    if bad.is_empty() {
        (true, format!("no process of this group; {}", notes.join("; ")))
    } else {
        (false, bad.join("; "))
    }
}

/// Graceful stop: SIGTERM rank 0 (its drain sends Shutdown, so followers exit 0 by themselves), wait, SIGTERM and then SIGKILL whatever
/// is left, then verify on every node that no server runs, the GPU lock is free and no compute app is on the GPU.
pub fn down(env: &dyn Env, c: &Cluster, g: &Group, out: &mut dyn Write) -> Result<DownReport, String> {
    let t0 = env.now_ms();
    let st = group_status(env, c, g, StatusOpts::default());
    for e in &st.errors {
        let _ = writeln!(out, "  warning: {e}");
    }
    let mut rep = DownReport { was_running: st.running(), ..Default::default() };
    if rep.was_running.is_empty() {
        let _ = writeln!(out, "  no rank of group {} is running", g.name);
    } else {
        let grace = c.tp.shutdown_grace_s;
        let leader_running = rep.was_running.contains(&0);
        if leader_running {
            let _ = writeln!(
                out,
                "  SIGTERM rank 0 (node {}): it drains and tells the followers to exit",
                c.rank_node(g, 0).map(|n| n.name.as_str()).unwrap_or("?")
            );
            signal(env, c, g, 0, "TERM");
        } else {
            let _ = writeln!(out, "  rank 0 is not running: the followers are lost (they exit by themselves within seconds)");
        }
        // wait for the whole group; log each rank's exit as it is seen
        let mut seen: BTreeMap<usize, u64> = BTreeMap::new();
        let deadline = env.now_ms() + (grace + 20) * 1000;
        let mut followers_termed = false;
        let mut rank0_gone_at: Option<u64> = if leader_running { None } else { Some(env.now_ms()) };
        loop {
            let now = env.now_ms();
            let cur = group_status(env, c, g, StatusOpts::default());
            let mut live = Vec::new();
            for r in &cur.ranks {
                if rep.was_running.contains(&r.rank) {
                    if r.active() {
                        live.push(r.rank);
                    } else if let std::collections::btree_map::Entry::Vacant(e) = seen.entry(r.rank) {
                        e.insert(now - t0);
                        let _ = writeln!(
                            out,
                            "  rank {} exited {} after {}",
                            r.rank,
                            r.exit.map(|e| e.to_string()).unwrap_or_else(|| "(no exit status recorded)".into()),
                            secs(now - t0)
                        );
                    }
                }
            }
            if live.is_empty() {
                break;
            }
            if !live.contains(&0) && rank0_gone_at.is_none() {
                rank0_gone_at = Some(now);
            }
            // a follower that has not followed rank 0 out within a few seconds is stopped too
            if !followers_termed && rank0_gone_at.is_some_and(|t| now - t > 8_000) {
                followers_termed = true;
                for r in &live {
                    let _ = writeln!(out, "  rank {r} did not follow rank 0: SIGTERM");
                    signal(env, c, g, *r, "TERM");
                }
            }
            if now > deadline {
                break;
            }
            env.sleep(Duration::from_millis(500));
        }
        let (killed, _) = stop_all(env, c, g, Duration::from_secs(5), None, out);
        rep.killed = killed;
        let fin = group_status(env, c, g, StatusOpts::default());
        for r in &fin.ranks {
            if rep.was_running.contains(&r.rank) {
                rep.exits.push((r.rank, r.exit));
            }
        }
    }
    // verification, node by node. A wrapper that has just written its exit file may hold the lock for a moment: poll for a few seconds.
    let mut fin = group_status(env, c, g, StatusOpts { lock: true, gpu: true });
    for _ in 0..10 {
        if c.lock_order(g).iter().all(|(rank, node)| node_clean(&fin, *rank, &node.name).0) {
            break;
        }
        env.sleep(Duration::from_millis(500));
        fin = group_status(env, c, g, StatusOpts { lock: true, gpu: true });
    }
    for (rank, node) in c.lock_order(g) {
        let (ok, why) = node_clean(&fin, rank, &node.name);
        let _ = writeln!(out, "  verify node {}: {}{}", node.name, if ok { "ok: " } else { "NOT CLEAN: " }, why);
        rep.verified.push((node.name.clone(), ok, why));
    }
    rep.total_ms = env.now_ms() - t0;
    Ok(rep)
}

/// A printable status of a group; the bool is "the group is up and ready".
pub fn status(env: &dyn Env, c: &Cluster, g: &Group, out: &mut dyn Write) -> bool {
    let st = group_status(env, c, g, StatusOpts { lock: true, gpu: false });
    let _ = writeln!(out, "group {} of cluster {} ({} ranks)", g.name, c.name, g.ranks.len());
    for r in &st.ranks {
        let node = c.rank_node(g, r.rank).map(|n| n.name.as_str()).unwrap_or("?");
        let line = if r.running() {
            format!("up   pid {} for {} s", r.pid.unwrap_or(0), r.etimes.unwrap_or(0))
        } else if let Some(code) = r.exit {
            format!("down exited {code} (state {})", r.state)
        } else {
            format!("down ({})", if r.state.is_empty() { "no state recorded" } else { r.state.as_str() })
        };
        let lock = st.nodes.get(node).and_then(|f| f.get("lock")).map(String::as_str).unwrap_or("?");
        let mem = st
            .nodes
            .get(node)
            .and_then(|f| f.get("mem_available_kib"))
            .and_then(|v| v.parse::<f64>().ok())
            .map(|k| format!("{:.1} GiB free", k / 1024.0 / 1024.0))
            .unwrap_or_default();
        let _ = writeln!(out, "  rank {} node {node:<4} {line}; GPU lock {lock}; {mem}", r.rank);
    }
    for e in &st.errors {
        let _ = writeln!(out, "  warning: {e}");
    }
    let mut ready = false;
    if let Some(leader) = c.rank_node(g, 0) {
        match env.http_get(&leader.address, g.http_port, "/health/ready") {
            Ok(r) => {
                ready = r.status == 200;
                let _ =
                    writeln!(out, "  leader http://{}:{}/health/ready -> {} {}", leader.address, g.http_port, r.status, r.body);
            }
            Err(e) => {
                let _ = writeln!(out, "  leader http://{}:{}/health/ready -> {e}", leader.address, g.http_port);
            }
        }
        if let Ok(r) = env.http_get(&leader.address, g.http_port, "/health") {
            let body: String = r.body.chars().take(400).collect();
            let _ = writeln!(out, "  leader /health -> {} {}", r.status, body);
        }
    }
    let up = st.all_running() && ready;
    let _ = writeln!(
        out,
        "{}",
        if up {
            "group is up and ready"
        } else if st.running().is_empty() {
            "group is down"
        } else {
            "group is NOT fully up"
        }
    );
    up
}

#[cfg(test)]
pub mod fake {
    //! A simulated cluster for the tests: nodes hold rank states, scripts are answered by recognising what they contain.

    use super::*;
    use crate::transport::{Exec, Transport};
    use std::sync::Mutex;

    #[derive(Debug, Clone, Default)]
    pub struct FakeRank {
        pub pid: Option<u32>,
        pub alive: bool,
        pub exit: Option<i32>,
        pub state: String,
        pub etimes: u64,
        /// Behaviour after a launch.
        pub plan: Plan,
        /// Virtual milliseconds since the launch.
        pub since_launch_ms: u64,
        pub launched: bool,
        pub log: String,
        /// Virtual time of the launch of the run script (the lock queue is counted from it).
        pub launch_ms: u64,
        /// The run script has a start barrier (a group of more than one rank).
        pub barrier: bool,
        /// The launcher wrote this rank's go file.
        pub go: bool,
        /// When the node's GPU lock was granted to the run script, and when the server was started.
        pub locked_at_ms: Option<u64>,
        pub started_at_ms: Option<u64>,
    }

    #[derive(Debug, Clone, Default)]
    pub struct Plan {
        /// The lock is not granted (state lock_timeout).
        pub lock_busy: bool,
        /// The lock is granted this many ms after the launch (another session holds it until then: the queue).
        pub lock_after_ms: u64,
        /// The server exits with this status this many ms after it was started (before ready).
        pub exit_at_ms: Option<(u64, i32)>,
        /// The group is ready (rank 0 answers 200) this many ms after the launch of rank 0.
        pub ready_at_ms: u64,
        /// SIGTERM is ignored (needs SIGKILL).
        pub ignores_term: bool,
        /// A follower exits 0 when rank 0 has exited 0 (it saw Shutdown).
        pub follows_leader: bool,
        /// The server exits this many ms after a SIGTERM with this status.
        pub term_exit_ms: u64,
        pub term_exit_code: i32,
    }

    pub struct FakeEnv {
        pub nodes: Mutex<BTreeMap<String, Vec<FakeRank>>>,
        pub now: Mutex<u64>,
        pub sent: Mutex<Vec<(String, String)>>,
        pub lock_held_by_other: Mutex<BTreeMap<String, bool>>,
        pub term_at: Mutex<BTreeMap<(String, usize), u64>>,
        pub ready_ms: Mutex<Option<u64>>,
        pub cancel_at: Mutex<Option<u64>>,
    }

    impl FakeEnv {
        pub fn new(c: &Cluster, g: &Group) -> FakeEnv {
            let mut nodes = BTreeMap::new();
            for (rank, n) in c.lock_order(g) {
                let mut v = vec![FakeRank::default(); g.ranks.len()];
                v[rank].plan = Plan {
                    ready_at_ms: 8_000,
                    term_exit_ms: 1_500,
                    term_exit_code: 0,
                    follows_leader: true,
                    ..Default::default()
                };
                nodes.insert(n.name.clone(), v);
            }
            FakeEnv {
                nodes: Mutex::new(nodes),
                now: Mutex::new(0),
                sent: Mutex::new(vec![]),
                lock_held_by_other: Mutex::new(BTreeMap::new()),
                term_at: Mutex::new(BTreeMap::new()),
                ready_ms: Mutex::new(None),
                cancel_at: Mutex::new(None),
            }
        }

        pub fn plan(&self, node: &str, rank: usize, f: impl FnOnce(&mut Plan)) {
            let mut n = self.nodes.lock().unwrap();
            f(&mut n.get_mut(node).unwrap()[rank].plan);
        }

        pub fn rank(&self, node: &str, rank: usize) -> FakeRank {
            self.nodes.lock().unwrap()[node][rank].clone()
        }

        /// Advance the simulated processes to the current virtual time.
        fn tick(&self) {
            let now = *self.now.lock().unwrap();
            let term_at = self.term_at.lock().unwrap().clone();
            let mut nodes = self.nodes.lock().unwrap();
            for (node, ranks) in nodes.iter_mut() {
                for (i, r) in ranks.iter_mut().enumerate() {
                    if !r.launched || !r.alive {
                        continue;
                    }
                    if r.pid.is_none() {
                        // the run script is alive and has no server: queued for the node's GPU lock, or waiting at the barrier
                        let start = |r: &mut FakeRank| {
                            r.pid = Some(1000 + i as u32 * 100 + (now % 97) as u32);
                            r.state = "running".into();
                            r.started_at_ms = Some(now);
                            r.since_launch_ms = now;
                        };
                        if r.state.is_empty() && now >= r.launch_ms + r.plan.lock_after_ms {
                            r.locked_at_ms = Some(now);
                            if r.barrier {
                                r.state = "locked".into();
                            } else {
                                start(r);
                            }
                        }
                        if r.state == "locked" && r.go {
                            start(r);
                        }
                        continue;
                    }
                    if let Some((at, code)) = r.plan.exit_at_ms {
                        if now >= r.since_launch_ms + at {
                            r.alive = false;
                            r.exit = Some(code);
                            r.state = "exited".into();
                            r.log = format!("error: simulated exit {code}\n");
                            continue;
                        }
                    }
                    if let Some(t) = term_at.get(&(node.clone(), i)) {
                        if !r.plan.ignores_term && now >= t + r.plan.term_exit_ms {
                            r.alive = false;
                            r.exit = Some(r.plan.term_exit_code);
                            r.state = "exited".into();
                        }
                    }
                }
            }
            // the followers see rank 0's Shutdown and exit 0 by themselves
            let leader_clean = nodes.values().any(|v| v[0].launched && !v[0].alive && v[0].exit == Some(0));
            if leader_clean {
                for ranks in nodes.values_mut() {
                    for (i, r) in ranks.iter_mut().enumerate() {
                        if i > 0 && r.launched && r.alive && r.plan.follows_leader {
                            r.alive = false;
                            r.exit = Some(0);
                            r.state = "exited".into();
                        }
                    }
                }
            }
        }
    }

    impl Env for FakeEnv {
        fn transport(&self) -> &(dyn Transport + Sync) {
            self
        }

        fn http_get(&self, host: &str, _port: u16, path: &str) -> Result<http::Response, String> {
            self.tick();
            let now = *self.now.lock().unwrap();
            let nodes = self.nodes.lock().unwrap();
            // the leader is the node whose rank 0 was launched
            for (_, ranks) in nodes.iter() {
                let r = &ranks[0];
                if r.launched && r.alive && r.pid.is_some() {
                    let ready = now >= r.since_launch_ms + r.plan.ready_at_ms;
                    let _ = host;
                    return Ok(if path.ends_with("ready") {
                        http::Response {
                            status: if ready { 200 } else { 503 },
                            body: if ready { "ready".into() } else { "loading".into() },
                        }
                    } else {
                        http::Response { status: 200, body: "{\"state\":\"ready\"}".into() }
                    });
                }
            }
            Err("connection refused".into())
        }

        fn sleep(&self, d: Duration) {
            *self.now.lock().unwrap() += d.as_millis() as u64;
        }

        fn now_ms(&self) -> u64 {
            *self.now.lock().unwrap()
        }

        fn cancelled(&self) -> bool {
            self.cancel_at.lock().unwrap().is_some_and(|t| *self.now.lock().unwrap() >= t)
        }
    }

    impl Transport for FakeEnv {
        fn run(&self, node: &crate::schema::Node, sc: &str, _timeout: Duration) -> Result<Exec, String> {
            self.tick();
            self.sent.lock().unwrap().push((node.name.clone(), sc.to_string()));
            let now = *self.now.lock().unwrap();
            let ok = |s: String| Ok(Exec { status: 0, stdout: s, stderr: String::new() });
            let mut nodes = self.nodes.lock().unwrap();
            let Some(ranks) = nodes.get_mut(&node.name) else { return Err(format!("no such node {}", node.name)) };
            // which rank the script is about: `echo "launched group g rank N"`, `rank$r` with `r=N`, or the file name in a signal
            let rank_of = |sc: &str| -> usize {
                for l in sc.lines() {
                    if let Some(rest) = l.strip_prefix("echo \"launched group ") {
                        if let Some(n) = rest.split("rank ").nth(1) {
                            return n.trim_end_matches('"').parse().unwrap_or(0);
                        }
                    }
                    if let Some(n) = l.strip_prefix("r=") {
                        if let Ok(n) = n.parse() {
                            return n;
                        }
                    }
                    if l.starts_with("tail -n") {
                        if let Some(i) = l.find("/rank") {
                            let n: String = l[i + 5..].chars().take_while(|c| c.is_ascii_digit()).collect();
                            return n.parse().unwrap_or(0);
                        }
                    }
                    if l.starts_with("p=$(cat \"$d/rank") {
                        let n = l.trim_start_matches("p=$(cat \"$d/rank").split('.').next().unwrap_or("0");
                        return n.parse().unwrap_or(0);
                    }
                }
                0
            };
            let rank = rank_of(sc);
            if sc.contains("REXL3_RUN") {
                let r = &mut ranks[rank];
                if r.alive && r.pid.is_some() {
                    return ok(format!("already_running pid={}\n", r.pid.unwrap()));
                }
                let plan = r.plan.clone();
                // a new process: the signal sent to the old one is not its
                self.term_at.lock().unwrap().remove(&(node.name.clone(), rank));
                let busy = plan.lock_busy || *self.lock_held_by_other.lock().unwrap().get(&node.name).unwrap_or(&false);
                // a run script with a barrier (a group of more than one rank) waits for its go file after taking the lock
                let barrier = sc.contains("until [ -e \"$go\" ]");
                *r = FakeRank { plan, launched: true, since_launch_ms: now, launch_ms: now, barrier, ..Default::default() };
                if busy {
                    r.state = "lock_timeout".into();
                    r.exit = Some(75);
                } else {
                    // the run script is alive; the lock is granted (and, without a barrier, the server started) by `tick`
                    r.alive = true;
                }
                return ok(format!("launched group g0 rank {rank}\n"));
            }
            if sc.contains("echo \"go rank") {
                let r = &mut ranks[rank];
                if r.alive && r.pid.is_none() && r.state == "locked" {
                    r.go = true;
                    return ok(format!("go rank {rank}\n"));
                }
                return ok("not_waiting\n".into());
            }
            if sc.contains("kill -TERM \"$p\"") || sc.contains("kill -KILL \"$p\"") {
                let r = &mut ranks[rank];
                if r.alive && r.pid.is_some() {
                    if sc.contains("kill -KILL") {
                        r.alive = false;
                        r.exit = Some(137);
                        r.state = "exited".into();
                    } else {
                        self.term_at.lock().unwrap().insert((node.name.clone(), rank), now);
                    }
                    return ok(format!("signalled pid={} X\n", r.pid.unwrap()));
                }
                if r.alive {
                    // no server yet: the run script (queued for the lock, or waiting at the barrier) is signalled and dies, and with it
                    // the lock it held; it writes no exit status
                    r.alive = false;
                    return ok("signalled pid=7 X (run script, no server yet)\n".into());
                }
                return ok("not_running\n".into());
            }
            if sc.contains("tail -n") {
                return ok(ranks[rank].log.clone());
            }
            // a status script
            let mut out = String::new();
            for (i, r) in ranks.iter().enumerate() {
                if sc.contains(&format!("r={i}\n")) {
                    // `alive` is the server (a pid); `wrapper` is the run script, which has no server while it waits
                    let server = r.alive && r.pid.is_some();
                    out.push_str(&format!(
                        "rank g0 {i} pid={} alive={} match={} ending=0 wrapper={} exit={} state={} etimes={}\n",
                        r.pid.map(|p| p.to_string()).unwrap_or_default(),
                        server as u8,
                        server as u8,
                        r.alive as u8,
                        r.exit.map(|e| e.to_string()).unwrap_or_default(),
                        r.state,
                        (now.saturating_sub(r.since_launch_ms)) / 1000
                    ));
                }
            }
            // the node's GPU lock is held by a server, or by a run script that was granted it (waiting at the barrier)
            let any_alive = ranks.iter().any(|r| r.alive && (r.pid.is_some() || r.state == "locked"));
            if sc.contains("lock=") {
                let other = *self.lock_held_by_other.lock().unwrap().get(&node.name).unwrap_or(&false);
                out.push_str(&format!("lock={}\n", if any_alive || other { "held" } else { "free" }));
                out.push_str(&format!("lock_ours={}\n", any_alive as u8));
                if other {
                    out.push_str("lock_holders=4242: another session's mpirun; \n");
                }
            }
            if sc.contains("gpu_apps=") {
                let servers = ranks.iter().any(|r| r.alive && r.pid.is_some());
                out.push_str(&format!("gpu_apps={}\nserve_procs={}\n", servers as u8, servers as u8));
            }
            out.push_str("mem_available_kib=100000000\n");
            ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::*;
    use super::*;
    use crate::schema::tests_support::TWO_NODE;

    fn setup() -> (Cluster, Group) {
        let c = Cluster::parse(TWO_NODE).unwrap();
        let g = c.groups[0].clone();
        (c, g)
    }

    fn opts() -> UpOptions {
        UpOptions { budget_s: None, ready_timeout: Duration::from_secs(120) }
    }

    #[test]
    fn up_takes_the_locks_in_node_order_waits_for_ready_and_leaves_both_running() {
        let (c, g) = setup();
        let env = FakeEnv::new(&c, &g);
        let mut out = Vec::new();
        let rep = up(&env, &c, &g, opts(), &mut out).expect("up");
        let text = String::from_utf8(out).unwrap();
        assert_eq!(rep.ranks.len(), 2);
        assert_eq!(rep.ranks[0].1, "N1");
        assert_eq!(rep.ranks[1].1, "N4");
        assert!(rep.ready_after_ms >= 8_000, "{}", rep.ready_after_ms);
        assert!(env.rank("N1", 0).alive && env.rank("N4", 1).alive);
        // the launch scripts went to N1 first, then N4
        let sent = env.sent.lock().unwrap();
        let launches: Vec<&str> = sent.iter().filter(|(_, s)| s.contains("REXL3_RUN")).map(|(n, _)| n.as_str()).collect();
        assert_eq!(launches, ["N1", "N4"]);
        assert!(text.contains("taking /tmp/rexl3-gpu.lock"), "{text}");
    }

    #[test]
    fn a_queued_second_lock_starts_no_server_until_every_lock_is_held() {
        // Rank 0's lock is granted at once; node 4's is queued behind another session for 40 s. The servers used to start rank by
        // rank, so rank 0's `--tp-connect-timeout` clock ran for the whole queue; now NO server starts before the last lock.
        let (c, g) = setup();
        let env = FakeEnv::new(&c, &g);
        env.plan("N4", 1, |p| p.lock_after_ms = 40_000);
        let mut out = Vec::new();
        let rep = up(&env, &c, &g, opts(), &mut out).expect("up");
        let text = String::from_utf8(out).unwrap();
        let (r0, r1) = (env.rank("N1", 0), env.rank("N4", 1));
        assert!(r0.locked_at_ms.unwrap() < 2_000, "rank 0's lock was granted at once: {r0:?}");
        assert!(r1.locked_at_ms.unwrap() >= 40_000, "rank 1's lock was queued: {r1:?}");
        let (s0, s1) = (r0.started_at_ms.unwrap(), r1.started_at_ms.unwrap());
        assert!(s0 >= r1.locked_at_ms.unwrap(), "rank 0 started at {s0} ms, before rank 1 held its lock: {r1:?}");
        assert!(s0.abs_diff(s1) <= 1_000, "the servers start together: rank 0 at {s0} ms, rank 1 at {s1} ms");
        // the launcher said what it waited for, and the report keeps the lock times (not the start times)
        assert!(text.contains("waiting at the barrier") && text.contains("every GPU lock is held"), "{text}");
        assert!(rep.ranks[0].3 < 2_000 && rep.ranks[1].3 >= 40_000, "{:?}", rep.ranks);
        // the go (barrier release) reached both nodes after both launches, in lock order
        let sent = env.sent.lock().unwrap();
        let seq: Vec<(&str, &str)> = sent
            .iter()
            .filter_map(|(n, s)| {
                if s.contains("REXL3_RUN") {
                    Some((n.as_str(), "launch"))
                } else if s.contains("echo \"go rank") {
                    Some((n.as_str(), "go"))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(seq, [("N1", "launch"), ("N4", "launch"), ("N1", "go"), ("N4", "go")]);
    }

    #[test]
    fn a_failure_before_the_barrier_starts_nothing_and_releases_every_lock() {
        // node 4's lock is never granted: rank 0 holds node 1's lock at the barrier, and must neither start its server nor keep the lock
        let (c, g) = setup();
        let env = FakeEnv::new(&c, &g);
        env.plan("N4", 1, |p| p.lock_busy = true);
        let mut out = Vec::new();
        let f = up(&env, &c, &g, opts(), &mut out).unwrap_err();
        let text = String::from_utf8(out).unwrap();
        assert!(f.lock_busy && f.message.contains("node N4: the GPU lock"), "{}", f.message);
        let (r0, r1) = (env.rank("N1", 0), env.rank("N4", 1));
        assert!(r0.locked_at_ms.is_some(), "rank 0 did hold node 1's lock at the barrier: {r0:?}");
        assert!(r0.started_at_ms.is_none() && r0.pid.is_none(), "nothing was started: {r0:?}");
        assert!(r1.started_at_ms.is_none() && r1.pid.is_none(), "{r1:?}");
        assert!(!r0.alive && !r1.alive, "every run script is gone, so every lock is released");
        // the lock of node 1 is free, by the probe a `down` runs
        let mut out = Vec::new();
        let rep = down(&env, &c, &g, &mut out).unwrap();
        assert!(rep.clean() && rep.was_running.is_empty(), "{}", String::from_utf8_lossy(&out));
        assert!(text.contains("stopping every rank started so far"), "{text}");
        // and no go was ever sent
        assert!(env.sent.lock().unwrap().iter().all(|(_, s)| !s.contains("echo \"go rank")));
    }

    #[test]
    fn an_interrupt_while_the_second_lock_queues_starts_nothing() {
        let (c, g) = setup();
        let env = FakeEnv::new(&c, &g);
        env.plan("N4", 1, |p| p.lock_after_ms = 60_000);
        *env.cancel_at.lock().unwrap() = Some(10_000);
        let mut out = Vec::new();
        let f = up(&env, &c, &g, opts(), &mut out).unwrap_err();
        assert!(f.interrupted);
        let (r0, r1) = (env.rank("N1", 0), env.rank("N4", 1));
        assert!(r0.started_at_ms.is_none() && r1.started_at_ms.is_none(), "{r0:?} {r1:?}");
        assert!(!r0.alive && !r1.alive, "both run scripts were signalled by pid: the queued one too");
    }

    #[test]
    fn the_run_script_of_a_group_waits_at_a_barrier_and_a_single_rank_does_not() {
        let (c, g) = setup();
        let run = script::run_script(&c, &g, 0, None);
        let (lock, locked, barrier, server) = (
            run.find("flock -w").unwrap(),
            run.find("echo locked").unwrap(),
            run.find("until [ -e \"$go\" ]").expect("a group's run script waits for its go file"),
            // the server's start (the binary's path also appears earlier, in `bin=`, for the watcher's identity check)
            run.find("pid=$!").unwrap(),
        );
        assert!(lock < locked && locked < barrier && barrier < server, "lock, then locked, then the barrier, then the server");
        assert!(run.contains("launcher.beat") && run.contains("barrier_timeout"), "a vanished launcher releases the lock: {run}");
        // a plain single-node server has no peers to wait for
        let one_text = TWO_NODE.replace("\"ranks\": [\"N1\", \"N4\"]", "\"ranks\": [\"N1\"]");
        assert_ne!(one_text, TWO_NODE, "the fixture changed shape");
        let c1 = Cluster::parse(&one_text).expect("a one-rank group");
        let g1 = c1.groups[0].clone();
        assert_eq!(g1.ranks.len(), 1);
        let one_run = script::run_script(&c1, &g1, 0, None);
        assert!(!one_run.contains("until [ -e \"$go\" ]") && one_run.contains("echo locked"), "{one_run}");
        // the launch removes a stale go file (a previous start's), and the go script writes only for a run script that is waiting
        let launch = script::launch_script(&c, &g, 1, None);
        assert!(launch.contains("rank1.go"), "{launch}");
        let go = script::go_script(&c, &g, 1);
        assert!(go.contains("rank1.run") && go.contains("echo \"go rank 1\"") && go.contains("not_waiting"), "{go}");
    }

    #[test]
    fn a_rank_that_dies_before_ready_stops_everything_by_pid() {
        let (c, g) = setup();
        let env = FakeEnv::new(&c, &g);
        env.plan("N4", 1, |p| p.exit_at_ms = Some((3_000, 1)));
        let mut out = Vec::new();
        let f = up(&env, &c, &g, opts(), &mut out).unwrap_err();
        let e = f.message.clone();
        assert_eq!(f.exits, [None, Some(1)], "the exit codes are the evidence a supervisor reads");
        assert!(e.contains("rank 1 on node N4 is not running before the group is ready (exit 1"), "{e}");
        assert!(e.contains("simulated exit 1"), "the log tail is quoted: {e}");
        assert!(!env.rank("N1", 0).alive, "rank 0 was stopped by pid");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("SIGTERM rank 0") || text.contains("stopping every rank started so far"), "{text}");
    }

    #[test]
    fn an_up_on_a_group_that_is_already_up_refuses_and_leaves_it_running() {
        let (c, g) = setup();
        let env = FakeEnv::new(&c, &g);
        up(&env, &c, &g, opts(), &mut Vec::new()).unwrap();
        let mut out = Vec::new();
        let f = up(&env, &c, &g, opts(), &mut out).unwrap_err();
        assert!(f.message.contains("already running") && f.message.contains("down"), "{}", f.message);
        // the cleanup of the failed start stops what that start launched, which is nothing: the healthy group stays up
        assert!(env.rank("N1", 0).alive && env.rank("N4", 1).alive, "{}", String::from_utf8_lossy(&out));
        assert!(!String::from_utf8_lossy(&out).contains("SIGTERM rank"), "{}", String::from_utf8_lossy(&out));
    }

    #[test]
    fn a_busy_gpu_lock_fails_the_start_and_leaves_nothing_running() {
        let (c, g) = setup();
        let env = FakeEnv::new(&c, &g);
        env.plan("N4", 1, |p| p.lock_busy = true);
        let mut out = Vec::new();
        let f = up(&env, &c, &g, opts(), &mut out).unwrap_err();
        assert!(f.lock_busy);
        let e = f.message;
        assert!(e.contains("node N4: the GPU lock /tmp/rexl3-gpu.lock was not free within 300 s"), "{e}");
        assert!(!env.rank("N1", 0).alive);
    }

    #[test]
    fn an_interrupt_during_the_wait_stops_the_started_ranks() {
        let (c, g) = setup();
        let env = FakeEnv::new(&c, &g);
        *env.cancel_at.lock().unwrap() = Some(2_000);
        let mut out = Vec::new();
        let f = up(&env, &c, &g, opts(), &mut out).unwrap_err();
        assert!(f.interrupted);
        assert!(!env.rank("N1", 0).alive && !env.rank("N4", 1).alive);
    }

    #[test]
    fn down_terms_rank_zero_first_and_verifies_a_clean_pair() {
        let (c, g) = setup();
        let env = FakeEnv::new(&c, &g);
        let mut out = Vec::new();
        up(&env, &c, &g, opts(), &mut out).unwrap();
        // the leader drains for 1.5 s; the follower sees Shutdown and exits 0 by itself
        let mut out = Vec::new();
        let rep = down(&env, &c, &g, &mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert_eq!(rep.was_running, [0, 1]);
        assert!(!env.rank("N1", 0).alive && !env.rank("N4", 1).alive);
        assert!(rep.verified.iter().all(|(_, ok, _)| *ok), "{text}");
        assert!(rep.exits.iter().all(|(_, e)| *e == Some(0)), "{:?}", rep.exits);
        assert!(rep.clean(), "{text}");
        assert!(text.contains("rank 0 exited 0") && text.contains("rank 1 exited 0"), "{text}");
        assert!(!text.contains("SIGKILL") && !text.contains("did not follow"), "{text}");
    }

    #[test]
    fn down_kills_a_rank_that_ignores_term_and_says_so() {
        let (c, g) = setup();
        let env = FakeEnv::new(&c, &g);
        let mut out = Vec::new();
        up(&env, &c, &g, opts(), &mut out).unwrap();
        env.plan("N4", 1, |p| {
            p.ignores_term = true;
            p.follows_leader = false;
        });
        let mut out = Vec::new();
        let rep = down(&env, &c, &g, &mut out).unwrap();
        assert_eq!(rep.killed, [1]);
        assert!(!rep.clean());
        assert!(!env.rank("N4", 1).alive);
    }

    #[test]
    fn down_with_nothing_running_is_idempotent_and_verifies_the_locks() {
        let (c, g) = setup();
        let env = FakeEnv::new(&c, &g);
        let mut out = Vec::new();
        let rep = down(&env, &c, &g, &mut out).unwrap();
        assert!(rep.was_running.is_empty() && rep.clean());
        // another session's lock is reported, not hidden
        env.lock_held_by_other.lock().unwrap().insert("N4".into(), true);
        let mut out = Vec::new();
        let rep = down(&env, &c, &g, &mut out).unwrap();
        // the node is clean of THIS group: the other session's lock is named, not counted against `down`
        assert!(rep.clean());
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("held by another session (4242: another session's mpirun)"), "{text}");
    }

    #[test]
    fn status_reports_up_and_down() {
        let (c, g) = setup();
        let env = FakeEnv::new(&c, &g);
        let mut out = Vec::new();
        assert!(!status(&env, &c, &g, &mut out));
        up(&env, &c, &g, opts(), &mut Vec::new()).unwrap();
        let mut out = Vec::new();
        assert!(status(&env, &c, &g, &mut out));
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("group is up and ready") && text.contains("rank 1 node N4"), "{text}");
    }
}
