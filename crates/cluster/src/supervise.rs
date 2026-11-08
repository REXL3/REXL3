//! The group-aware supervisor: restart ALL ranks together, with an exit-code map, a capped backoff and a give-up rule.
//!
//! A tensor-parallel group is a unit (docs/NEMOTRON_TP.md 5.5): any lost rank makes every surviving rank exit non-zero. Per-host
//! systemd restarts each rank on its own and re-forms the group only when every rank's exit is a restartable one; a refusal is where it
//! falls apart (rank 0 exits 101 and restarts, a follower exits 1 and is, by `RestartPreventExitStatus=1`, not restarted: rank 0 then
//! flaps alone). The supervisor reads the exit codes of all ranks of an attempt, classifies the attempt and restarts or stops the whole
//! group. The policy ([`Policy`], [`classify`]) is pure and tested without a process; [`supervise`] drives it over a [`Runner`].
//!
//! The exit-code map (also in docs/CLUSTER_OPERATIONS.md):
//!
//! | code | meaning | action |
//! | --- | --- | --- |
//! | 0 | clean stop (drained on SIGTERM; a follower saw Shutdown) | no restart |
//! | 70 | group lost, or a fatal backend/CUDA error (rank 0's `on_fatal`, a follower's lost leader, the exit backstop) | restart all, backoff |
//! | 101 | a panic on the engine thread, which is also how a refused boot of rank 0 ends | restart once; the second in a row stops |
//! | 134 | SIGABRT: the boot guard, the shutdown watchdog or the progress watchdog | restart all, backoff |
//! | 137, 143, 128+n | killed by a signal | restart all, backoff |
//! | 1 on rank 0 | a refused boot (a flag, the port in use, a plan that does not fit) | before ready: refusal; after ready: restart |
//! | 1 on a follower | before ready: a refused boot (or rank 0 gone during the boot); after ready: rank 0 was lost while idle (a binary from before the follower exited 70 for it) | before ready: refusal; after: restart |
//! | 2, 3 | a usage error; the GLM container's memory guard | refusal |
//! | 75 | the node's GPU lock was not free in `lock_wait_s` (this tool's run script) | restart, backoff |

use crate::ops::{self, Env, UpOptions};
use crate::schema::{Cluster, Group, SupervisorCfg};
use crate::script::StatusOpts;
use std::io::Write;
use std::time::Duration;

/// What one start-and-run of the whole group did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Attempt {
    /// Per rank: the exit status the rank left on its own, or `None` for a rank that was still running (or never started) when the
    /// attempt ended. Ranks the supervisor stopped are `None`: its own signals are not evidence.
    pub exits: Vec<Option<i32>>,
    pub was_ready: bool,
    /// How long the group stayed ready before the first rank ended.
    pub ready_for_s: u64,
    pub start_failed: Option<String>,
    pub lock_busy: bool,
    /// The leader's `/health` said the group is lost (its `tensor_parallel.state`), with its reason, before any rank had ended on its
    /// own: the evidence when no exit status is there.
    pub group_lost: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Class {
    /// Every rank that ended by itself exited 0.
    Clean,
    /// A restart will not help by itself: a refused boot, a panic, a usage error.
    Refused(String),
    /// The group was lost (a rank died, was killed, aborted, or its leader went away).
    Lost(String),
    /// The group could not be started: the lock is busy, a node did not answer, it was not ready in time.
    StartFailed(String),
}

fn lost_code(c: i32) -> bool {
    c == 70 || c == 134 || c >= 128
}

fn rank_name(r: usize) -> String {
    if r == 0 {
        "rank 0 (the leader)".to_string()
    } else {
        format!("rank {r}")
    }
}

/// The exit-code map applied to one attempt.
pub fn classify(a: &Attempt) -> Class {
    let ends: Vec<(usize, i32)> = a.exits.iter().enumerate().filter_map(|(r, e)| e.map(|c| (r, c))).collect();
    if a.lock_busy {
        return Class::StartFailed(a.start_failed.clone().unwrap_or_else(|| "a GPU lock was busy".into()));
    }
    if ends.is_empty() {
        return match (&a.start_failed, &a.group_lost) {
            (Some(m), _) => Class::StartFailed(m.clone()),
            (None, Some(why)) => Class::Lost(format!("the leader reports the group lost: {why}")),
            (None, None) => Class::Lost("a rank vanished with no exit status".into()),
        };
    }
    // a panic on any rank, or a rank 0 that refused
    if let Some((r, _)) = ends.iter().find(|(_, c)| *c == 101) {
        return Class::Refused(format!("{} exited 101 (a panic on the engine thread, or a refused boot)", rank_name(*r)));
    }
    let leader = ends.iter().find(|(r, _)| *r == 0).map(|(_, c)| *c);
    if let Some(c) = leader {
        if matches!(c, 2 | 3 | 78) || (c == 1 && !a.was_ready) {
            return Class::Refused(format!("rank 0 (the leader) exited {c}: a refused boot or a usage error"));
        }
    }
    let leader_lost = leader.is_some_and(lost_code);
    for (r, c) in ends.iter().filter(|(r, _)| *r > 0) {
        if matches!(*c, 2 | 3 | 78) {
            return Class::Refused(format!("{} exited {c}: a usage error or a missing file", rank_name(*r)));
        }
        if *c == 1 && !a.was_ready && !leader_lost {
            return Class::Refused(format!(
                "{} exited 1 before the group was ready: a refused boot (docs/NEMOTRON_TP.md 12.2: the ranks disagree on a plan fact, or it cannot plan its memory)",
                rank_name(*r)
            ));
        }
    }
    if ends.iter().all(|(_, c)| *c == 0) {
        return Class::Clean;
    }
    let what: Vec<String> = ends.iter().map(|(r, c)| format!("{} exited {c}", rank_name(*r))).collect();
    Class::Lost(what.join(", "))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Stop { code: i32, why: String },
    Restart { after_s: u64, why: String },
}

/// Backoff, the refusal rule and the rate limit.
#[derive(Debug, Clone)]
pub struct Policy {
    cfg: SupervisorCfg,
    refusals_in_row: u32,
    failures_in_row: u32,
    recent: Vec<u64>,
}

impl Policy {
    pub fn new(cfg: SupervisorCfg) -> Policy {
        Policy { cfg, refusals_in_row: 0, failures_in_row: 0, recent: Vec::new() }
    }

    fn backoff(&self) -> u64 {
        let exp = self.failures_in_row.saturating_sub(1).min(20);
        self.cfg.backoff_base_s.saturating_mul(1u64 << exp).min(self.cfg.backoff_cap_s)
    }

    fn note(&mut self, now_s: u64) {
        self.failures_in_row += 1;
        self.recent.push(now_s);
        let w = self.cfg.window_s;
        self.recent.retain(|t| now_s.saturating_sub(*t) <= w);
    }

    fn rate_exceeded(&self) -> bool {
        self.recent.len() as u32 > self.cfg.max_restarts
    }

    /// What to do after an attempt that ended at `now_s` (seconds on any monotonic clock).
    pub fn decide(&mut self, now_s: u64, a: &Attempt) -> Verdict {
        if a.was_ready && a.ready_for_s >= self.cfg.stable_after_s {
            // it served for long enough that this is a new incident, not the same failure again
            self.refusals_in_row = 0;
            self.failures_in_row = 0;
            self.recent.clear();
        }
        match classify(a) {
            Class::Clean => Verdict::Stop { code: 0, why: "every rank exited 0: a clean stop, nothing to restart".into() },
            Class::Refused(why) => {
                self.refusals_in_row += 1;
                if self.refusals_in_row >= 2 {
                    return Verdict::Stop {
                        code: crate::exit::GAVE_UP_REFUSED,
                        why: format!("refused twice in a row ({why}): restarting will not help; fix the cause and start again"),
                    };
                }
                self.note(now_s);
                // "restart the group once": the same delay as the first loss, then the second refusal stops
                Verdict::Restart { after_s: self.cfg.backoff_base_s, why: format!("{why}: restarting the group once") }
            }
            Class::Lost(why) => {
                self.refusals_in_row = 0;
                self.note(now_s);
                if self.rate_exceeded() {
                    return Verdict::Stop {
                        code: crate::exit::GAVE_UP_RATE,
                        why: format!(
                            "{} failures within {} s ({why}): giving up (supervisor.max_restarts is {})",
                            self.recent.len(),
                            self.cfg.window_s,
                            self.cfg.max_restarts
                        ),
                    };
                }
                Verdict::Restart { after_s: self.backoff(), why: format!("group lost ({why}): restarting all ranks") }
            }
            Class::StartFailed(why) => {
                self.note(now_s);
                if self.rate_exceeded() {
                    return Verdict::Stop {
                        code: crate::exit::GAVE_UP_RATE,
                        why: format!("{} failed starts within {} s ({why}): giving up", self.recent.len(), self.cfg.window_s),
                    };
                }
                Verdict::Restart { after_s: self.backoff(), why: format!("could not start the group ({why})") }
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct StartFailure {
    pub message: String,
    pub exits: Vec<Option<i32>>,
    pub lock_busy: bool,
    pub interrupted: bool,
}

/// What the supervisor drives: start the group (waiting for ready), read the ranks' exit statuses, stop the group, and the clock.
pub trait Runner {
    fn start(&mut self) -> Result<(), StartFailure>;
    /// Per rank: `Some(status)` once it has ended, `None` while it runs.
    fn poll(&mut self) -> Vec<Option<i32>>;
    /// The leader's own word that the group is lost (its `/health`: `tensor_parallel.state` is `lost`), with the reason. A group that
    /// is lost never serves again, and an idle leader takes 20 s (the runtime's `hang_exit` backstop) to exit after it: waiting for
    /// that exit only delays the restart.
    fn group_lost(&mut self) -> Option<String> {
        None
    }
    /// Stop the group gracefully: rank 0 first, then whatever follows it.
    fn stop(&mut self);
    fn sleep(&mut self, secs: u64);
    fn now_s(&self) -> u64;
    fn stop_requested(&self) -> bool;
}

/// After a rank has ended or the leader has reported the group lost, how long the other ranks get to exit by themselves before the
/// supervisor stops what is left: the evidence is their exit statuses, and a lost group exits in about a second with a request in
/// flight. (Without it the wait for an idle leader is `settle_s`: the leader's own 20 s backstop.)
const LOST_GRACE_S: u64 = 3;

/// Run until the operator stops it or the policy gives up. Returns the process exit status of the supervisor.
pub fn supervise(r: &mut dyn Runner, cfg: &SupervisorCfg, log: &mut dyn FnMut(&str)) -> i32 {
    let mut policy = Policy::new(cfg.clone());
    let mut attempt_no = 0;
    loop {
        if r.stop_requested() {
            log("stop requested: nothing running");
            return crate::exit::OK;
        }
        attempt_no += 1;
        log(&format!("attempt {attempt_no}: starting the group"));
        let attempt = match r.start() {
            Err(f) if f.interrupted || r.stop_requested() => {
                log("stop requested during the start: the ranks started were stopped");
                return crate::exit::OK;
            }
            Err(f) => {
                log(&format!("attempt {attempt_no}: start failed: {}", f.message.lines().next().unwrap_or("")));
                Attempt {
                    exits: f.exits,
                    was_ready: false,
                    ready_for_s: 0,
                    start_failed: Some(f.message),
                    lock_busy: f.lock_busy,
                    group_lost: None,
                }
            }
            Ok(()) => {
                let ready_at = r.now_s();
                log(&format!("attempt {attempt_no}: group ready"));
                // watch until a rank ends, the leader says the group is lost, or the operator stops us
                let mut lost_reason: Option<String> = None;
                loop {
                    if r.stop_requested() {
                        log("stop requested: stopping rank 0 first, the followers follow it");
                        r.stop();
                        return crate::exit::OK;
                    }
                    if r.poll().iter().any(|e| e.is_some()) {
                        break;
                    }
                    if let Some(why) = r.group_lost() {
                        log(&format!("attempt {attempt_no}: the leader reports the group lost: {why}"));
                        lost_reason = Some(why);
                        break;
                    }
                    r.sleep(cfg.poll_s);
                }
                // a rank ended: the others get `settle_s` to end by themselves so that their statuses are the evidence
                let first = r.now_s();
                let mut exits;
                loop {
                    exits = r.poll();
                    let waited = r.now_s().saturating_sub(first);
                    if exits.iter().all(|e| e.is_some()) || waited >= cfg.settle_s {
                        break;
                    }
                    // the leader has said the group is lost: the ranks that are going to exit by themselves have had LOST_GRACE_S
                    if lost_reason.is_some() && waited >= LOST_GRACE_S {
                        break;
                    }
                    if lost_reason.is_none() {
                        lost_reason = r.group_lost();
                        if let Some(why) = &lost_reason {
                            log(&format!("attempt {attempt_no}: the leader reports the group lost: {why}"));
                        }
                    }
                    if r.stop_requested() {
                        r.stop();
                        return crate::exit::OK;
                    }
                    r.sleep(1);
                }
                let shown: Vec<String> = exits
                    .iter()
                    .enumerate()
                    .map(|(i, e)| format!("rank {i}: {}", e.map(|c| c.to_string()).unwrap_or_else(|| "still running".into())))
                    .collect();
                log(&format!(
                    "attempt {attempt_no}: a rank ended after {} s ready: {}",
                    first.saturating_sub(ready_at),
                    shown.join(", ")
                ));
                r.stop();
                Attempt {
                    exits,
                    was_ready: true,
                    ready_for_s: first.saturating_sub(ready_at),
                    start_failed: None,
                    lock_busy: false,
                    group_lost: lost_reason,
                }
            }
        };
        match policy.decide(r.now_s(), &attempt) {
            Verdict::Stop { code, why } => {
                log(&format!("stopping: {why} (exit {code})"));
                return code;
            }
            Verdict::Restart { after_s, why } => {
                log(&format!("{why}; restarting in {after_s} s"));
                let mut left = after_s;
                while left > 0 {
                    if r.stop_requested() {
                        log("stop requested during the backoff");
                        return crate::exit::OK;
                    }
                    r.sleep(1);
                    left -= 1;
                }
            }
        }
    }
}

/// The real runner: `up` and `down` of [`crate::ops`] over an [`Env`].
pub struct ClusterRunner<'a> {
    pub env: &'a dyn Env,
    pub c: &'a Cluster,
    pub g: &'a Group,
    pub up: UpOptions,
    pub out: &'a mut dyn Write,
}

impl Runner for ClusterRunner<'_> {
    fn start(&mut self) -> Result<(), StartFailure> {
        // A group that is already up and ready is adopted (a supervisor that was restarted must not start a second one); ranks that
        // are up without a ready group (a half start, an earlier supervisor's) are stopped first.
        let st = ops::group_status(self.env, self.c, self.g, StatusOpts::default());
        if !st.running().is_empty() {
            let ready = self
                .c
                .rank_node(self.g, 0)
                .and_then(|n| self.env.http_get(&n.address, self.g.http_port, "/health/ready").ok())
                .is_some_and(|r| r.status == 200);
            if st.all_running() && ready {
                let _ = writeln!(self.out, "  the group is already up and ready: adopting it");
                return Ok(());
            }
            let _ = writeln!(self.out, "  ranks {:?} are running without a ready group: stopping them first", st.running());
            let _ = ops::down(self.env, self.c, self.g, self.out);
        }
        match ops::up(self.env, self.c, self.g, self.up, self.out) {
            Ok(rep) => {
                let _ = writeln!(self.out, "  group ready after {:.1} s", rep.ready_after_ms as f64 / 1000.0);
                Ok(())
            }
            Err(f) => {
                Err(StartFailure { message: f.message, exits: f.exits, lock_busy: f.lock_busy, interrupted: f.interrupted })
            }
        }
    }

    fn poll(&mut self) -> Vec<Option<i32>> {
        let st = ops::group_status(self.env, self.c, self.g, StatusOpts::default());
        st.ranks
            .iter()
            .map(|r| {
                if r.running() {
                    None
                } else if r.state == "unreachable" {
                    // an unreachable node says nothing: keep watching
                    None
                } else if r.exit.is_none() && r.state == "running" && r.pid.is_some() {
                    // the server is gone and its run script has not written the status yet (it does so within milliseconds of reaping
                    // it): not ended until the status is there
                    None
                } else if r.exit.is_none() && r.wrapper {
                    // the run script is alive with no server: queued for its GPU lock, or at the start barrier. Not ended.
                    None
                } else {
                    // a pid that is gone with no exit file and no `exited` state (the run script itself was killed): a loss
                    Some(r.exit.unwrap_or(137))
                }
            })
            .collect()
    }

    fn group_lost(&mut self) -> Option<String> {
        let leader = self.c.rank_node(self.g, 0)?;
        let r = self.env.http_get(&leader.address, self.g.http_port, "/health").ok()?;
        let v: serde_json::Value = serde_json::from_str(&r.body).ok()?;
        let tp = v.get("tensor_parallel")?;
        (tp.get("state")?.as_str()? == "lost")
            .then(|| tp.get("lost").and_then(|x| x.as_str()).unwrap_or("no reason given").chars().take(160).collect())
    }

    fn stop(&mut self) {
        match ops::down(self.env, self.c, self.g, self.out) {
            Ok(rep) => {
                let _ = writeln!(
                    self.out,
                    "  stopped in {:.1} s{}",
                    rep.total_ms as f64 / 1000.0,
                    if rep.clean() { "" } else { " (not clean, see above)" }
                );
            }
            Err(e) => {
                let _ = writeln!(self.out, "  stop failed: {e}");
            }
        }
    }

    fn sleep(&mut self, secs: u64) {
        self.env.sleep(Duration::from_secs(secs));
    }

    fn now_s(&self) -> u64 {
        self.env.now_ms() / 1000
    }

    fn stop_requested(&self) -> bool {
        self.env.cancelled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn cfg() -> SupervisorCfg {
        SupervisorCfg {
            backoff_base_s: 10,
            backoff_cap_s: 40,
            stable_after_s: 300,
            max_restarts: 5,
            window_s: 600,
            settle_s: 25,
            poll_s: 2,
        }
    }

    fn att(exits: &[Option<i32>], ready: bool, ready_for: u64) -> Attempt {
        Attempt { exits: exits.to_vec(), was_ready: ready, ready_for_s: ready_for, ..Default::default() }
    }

    // ------------------------------------------------------------------------------------------------ the exit-code map
    #[test]
    fn the_exit_code_map() {
        // clean: the leader drained and the follower saw Shutdown
        assert_eq!(classify(&att(&[Some(0), Some(0)], true, 100)), Class::Clean);
        // lost: the follower exited 70 for a lost leader, and rank 0 for a lost follower
        assert!(matches!(classify(&att(&[Some(70), Some(70)], true, 100)), Class::Lost(_)));
        assert!(matches!(classify(&att(&[None, Some(70)], true, 100)), Class::Lost(_)));
        assert!(matches!(classify(&att(&[Some(134), Some(70)], true, 100)), Class::Lost(_)));
        assert!(matches!(classify(&att(&[Some(137), None], true, 100)), Class::Lost(_)));
        // a refused boot: rank 0's engine thread panic and a follower's exit 1 with it
        assert!(matches!(classify(&att(&[Some(101), Some(1)], false, 0)), Class::Refused(_)));
        // a panic after the group was serving is the same code and the same rule
        assert!(matches!(classify(&att(&[Some(101), Some(70)], true, 50)), Class::Refused(_)));
        // the clash: a follower's 1 is a refusal before ready and a lost idle leader after it
        assert!(matches!(classify(&att(&[None, Some(1)], false, 0)), Class::Refused(_)));
        assert!(
            matches!(classify(&att(&[Some(70), Some(1)], false, 0)), Class::Lost(_)),
            "rank 0 died while the follower was booting"
        );
        assert!(matches!(classify(&att(&[Some(70), Some(1)], true, 9)), Class::Lost(_)));
        assert!(matches!(classify(&att(&[None, Some(1)], true, 9)), Class::Lost(_)));
        // usage errors and the missing env file never restart
        assert!(matches!(classify(&att(&[Some(2), None], false, 0)), Class::Refused(_)));
        assert!(matches!(classify(&att(&[Some(1), None], false, 0)), Class::Refused(_)));
        // a busy lock is the node's other tenant, not the group
        let mut a = att(&[Some(75), None], false, 0);
        a.lock_busy = true;
        a.start_failed = Some("lock".into());
        assert!(matches!(classify(&a), Class::StartFailed(_)));
    }

    // ------------------------------------------------------------------------------------------------ the policy
    #[test]
    fn a_lost_group_restarts_with_a_doubling_backoff_up_to_the_cap() {
        let mut p = Policy::new(SupervisorCfg { max_restarts: 50, ..cfg() });
        let lost = att(&[Some(70), Some(70)], true, 20);
        let mut delays = vec![];
        for i in 0..5 {
            match p.decide(1000 + i * 100, &lost) {
                Verdict::Restart { after_s, .. } => delays.push(after_s),
                v => panic!("{v:?}"),
            }
        }
        assert_eq!(delays, [10, 20, 40, 40, 40]);
    }

    #[test]
    fn a_stable_run_resets_the_backoff() {
        let mut p = Policy::new(cfg());
        let lost = att(&[Some(70), Some(70)], true, 20);
        assert!(matches!(p.decide(100, &lost), Verdict::Restart { after_s: 10, .. }));
        assert!(matches!(p.decide(200, &lost), Verdict::Restart { after_s: 20, .. }));
        // it then served for an hour
        let long = att(&[Some(70), Some(70)], true, 3600);
        assert!(matches!(p.decide(5000, &long), Verdict::Restart { after_s: 10, .. }));
    }

    #[test]
    fn the_second_refusal_in_a_row_stops_but_a_loss_between_resets_it() {
        let mut p = Policy::new(cfg());
        let refused = att(&[Some(101), Some(1)], false, 0);
        assert!(matches!(p.decide(0, &refused), Verdict::Restart { .. }));
        match p.decide(20, &refused) {
            Verdict::Stop { code, why } => {
                assert_eq!(code, crate::exit::GAVE_UP_REFUSED);
                assert!(why.contains("twice in a row"), "{why}");
            }
            v => panic!("{v:?}"),
        }
        let mut p = Policy::new(cfg());
        assert!(matches!(p.decide(0, &refused), Verdict::Restart { .. }));
        assert!(matches!(p.decide(20, &att(&[Some(70), Some(70)], true, 5)), Verdict::Restart { .. }));
        assert!(matches!(p.decide(40, &refused), Verdict::Restart { .. }), "101, 70, 101 is not twice in a row");
    }

    #[test]
    fn too_many_failures_in_the_window_stop_the_supervisor() {
        let mut p = Policy::new(SupervisorCfg { max_restarts: 3, ..cfg() });
        let lost = att(&[Some(70), None], true, 5);
        for i in 0..3 {
            assert!(matches!(p.decide(i * 30, &lost), Verdict::Restart { .. }));
        }
        match p.decide(100, &lost) {
            Verdict::Stop { code, .. } => assert_eq!(code, crate::exit::GAVE_UP_RATE),
            v => panic!("{v:?}"),
        }
        // the same failures spread over more than the window are fine
        let mut p = Policy::new(SupervisorCfg { max_restarts: 3, ..cfg() });
        for i in 0..8 {
            assert!(matches!(p.decide(i * 700, &lost), Verdict::Restart { .. }), "{i}");
        }
    }

    #[test]
    fn a_clean_exit_is_not_restarted() {
        let mut p = Policy::new(cfg());
        assert!(matches!(p.decide(0, &att(&[Some(0), Some(0)], true, 10)), Verdict::Stop { code: 0, .. }));
    }

    // ------------------------------------------------------------------------------------------------ the loop, over a fake runner
    /// A scripted group: each `start` pops a step; a `Run` step says when ranks end after the group was ready.
    enum Step {
        StartFails { exits: Vec<Option<i32>>, lock_busy: bool },
        Run { ends: Vec<(u64, usize, i32)> },
    }

    struct Fake {
        steps: VecDeque<Step>,
        now: u64,
        ready_at: u64,
        ends: Vec<(u64, usize, i32)>,
        running: bool,
        ranks: usize,
        stop_after: Option<u64>,
        stops: Vec<u64>,
        starts: Vec<u64>,
        /// The leader reports the group lost this many seconds after the group was ready.
        lost_after: Option<u64>,
    }

    impl Fake {
        fn new(steps: Vec<Step>) -> Fake {
            Fake {
                steps: steps.into(),
                now: 0,
                ready_at: 0,
                ends: vec![],
                running: false,
                ranks: 2,
                stop_after: None,
                stops: vec![],
                starts: vec![],
                lost_after: None,
            }
        }
    }

    impl Runner for Fake {
        fn start(&mut self) -> Result<(), StartFailure> {
            self.starts.push(self.now);
            match self.steps.pop_front().expect("a step") {
                Step::StartFails { exits, lock_busy } => {
                    self.now += 30;
                    Err(StartFailure { message: "could not start".into(), exits, lock_busy, interrupted: false })
                }
                Step::Run { ends } => {
                    self.now += 10;
                    self.ready_at = self.now;
                    self.ends = ends;
                    self.running = true;
                    Ok(())
                }
            }
        }

        fn poll(&mut self) -> Vec<Option<i32>> {
            let mut v = vec![None; self.ranks];
            for (t, r, c) in &self.ends {
                if self.now >= self.ready_at + t {
                    v[*r] = Some(*c);
                }
            }
            v
        }

        fn group_lost(&mut self) -> Option<String> {
            // reported once, as the real leader's `/health` does until the process exits
            let t = self.lost_after?;
            if self.running && self.now >= self.ready_at + t {
                self.lost_after = None;
                Some("the connection to rank 1 failed: connection closed by the peer".to_string())
            } else {
                None
            }
        }

        fn stop(&mut self) {
            self.stops.push(self.now);
            self.running = false;
        }

        fn sleep(&mut self, secs: u64) {
            self.now += secs;
        }

        fn now_s(&self) -> u64 {
            self.now
        }

        fn stop_requested(&self) -> bool {
            self.stop_after.is_some_and(|t| self.now >= t)
        }
    }

    fn run(f: &mut Fake) -> (i32, Vec<String>) {
        let mut lines = vec![];
        let code = supervise(f, &cfg(), &mut |l| lines.push(l.to_string()));
        (code, lines)
    }

    #[test]
    fn rank_loss_restarts_all_ranks_together_and_a_clean_end_stops() {
        let mut f = Fake::new(vec![
            // rank 1 is killed 100 s into the run; rank 0 follows with 70 a second and a half later
            Step::Run { ends: vec![(100, 1, 137), (102, 0, 70)] },
            // the second run is stopped cleanly by an operator `down`: both 0
            Step::Run { ends: vec![(500, 0, 0), (501, 1, 0)] },
        ]);
        let (code, lines) = run(&mut f);
        assert_eq!(code, 0);
        assert_eq!(f.starts.len(), 2, "restarted once: {lines:#?}");
        assert_eq!(f.stops.len(), 2, "the survivors of each attempt were stopped (rank 0 first by the real runner)");
        assert!(lines.iter().any(|l| l.contains("group lost") && l.contains("restarting all ranks")), "{lines:#?}");
        assert!(lines.last().unwrap().contains("clean stop"), "{lines:#?}");
    }

    #[test]
    fn a_refused_boot_twice_stops_the_supervisor_with_78() {
        let mut f = Fake::new(vec![
            Step::StartFails { exits: vec![Some(101), Some(1)], lock_busy: false },
            Step::StartFails { exits: vec![Some(101), Some(1)], lock_busy: false },
            Step::Run { ends: vec![] },
        ]);
        let (code, lines) = run(&mut f);
        assert_eq!(code, crate::exit::GAVE_UP_REFUSED);
        assert_eq!(f.starts.len(), 2, "one restart, then it stops: {lines:#?}");
    }

    #[test]
    fn a_busy_lock_backs_off_and_tries_again() {
        let mut f = Fake::new(vec![
            Step::StartFails { exits: vec![None, None], lock_busy: true },
            Step::Run { ends: vec![(30, 0, 0), (31, 1, 0)] },
        ]);
        let (code, _) = run(&mut f);
        assert_eq!(code, 0);
        assert_eq!(f.starts.len(), 2);
        assert!(f.starts[1] - f.starts[0] >= 10 + 30, "the backoff was waited: {:?}", f.starts);
    }

    #[test]
    fn sigterm_stops_rank_zero_first_through_one_graceful_stop() {
        let mut f = Fake::new(vec![Step::Run { ends: vec![] }]);
        f.stop_after = Some(400);
        let (code, lines) = run(&mut f);
        assert_eq!(code, 0);
        assert_eq!(f.stops.len(), 1);
        assert!(lines.iter().any(|l| l.contains("stopping rank 0 first")), "{lines:#?}");
    }

    // ------------------------------------------------------------------------------------------------ the real runner, over the simulated cluster
    fn launches(env: &crate::ops::fake::FakeEnv) -> usize {
        env.sent.lock().unwrap().iter().filter(|(_, s)| s.contains("REXL3_RUN")).count()
    }

    fn two_nodes() -> (Cluster, Group) {
        let c = Cluster::parse(crate::schema::tests_support::TWO_NODE).unwrap();
        let g = c.groups[0].clone();
        (c, g)
    }

    #[test]
    fn a_supervisor_adopts_a_group_that_is_already_up_and_ready() {
        let (c, g) = two_nodes();
        let env = crate::ops::fake::FakeEnv::new(&c, &g);
        let o = UpOptions { budget_s: None, ready_timeout: Duration::from_secs(120) };
        ops::up(&env, &c, &g, o, &mut Vec::new()).unwrap();
        assert_eq!(launches(&env), 2);
        let mut out = Vec::new();
        let mut r = ClusterRunner { env: &env, c: &c, g: &g, up: o, out: &mut out };
        r.start().expect("adopted");
        assert_eq!(launches(&env), 2, "no second start");
        assert!(r.poll().iter().all(|e| e.is_none()), "both ranks are running");
        assert!(String::from_utf8(out).unwrap().contains("adopting"));
    }

    #[test]
    fn a_half_started_group_is_stopped_and_started_again() {
        let (c, g) = two_nodes();
        let env = crate::ops::fake::FakeEnv::new(&c, &g);
        // rank 0 alone (an earlier, interrupted start)
        let (rank, n1) = c.lock_order(&g)[0];
        env.transport().run(n1, &crate::script::launch_script(&c, &g, rank, None), Duration::from_secs(5)).unwrap();
        assert!(env.rank("N1", 0).alive && !env.rank("N4", 1).alive);
        let o = UpOptions { budget_s: None, ready_timeout: Duration::from_secs(120) };
        let mut out = Vec::new();
        let mut r = ClusterRunner { env: &env, c: &c, g: &g, up: o, out: &mut out };
        r.start().expect("started");
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("are running without a ready group: stopping them first"), "{text}");
        assert!(env.rank("N1", 0).alive && env.rank("N4", 1).alive, "{text}");
        assert_eq!(launches(&env), 3, "the stray rank 0, then both");
    }

    #[test]
    fn an_idle_leader_that_reports_the_group_lost_is_not_waited_for() {
        // the follower is killed at 100 s; the leader says the group is lost at 101 s but, idle, exits only at 120 s (its backstop)
        let mut f = Fake::new(vec![
            Step::Run { ends: vec![(100, 1, 137), (120, 0, 70)] },
            Step::Run { ends: vec![(500, 0, 0), (501, 1, 0)] },
        ]);
        f.lost_after = Some(101);
        let (code, lines) = run(&mut f);
        assert_eq!(code, 0, "{lines:#?}");
        assert_eq!(f.starts.len(), 2);
        // the second start came before the leader's own exit (ready at 10 s: 120 s after is 130 s) plus the 10 s backoff
        assert!(f.starts[1] < 10 + 120, "restarted without waiting for the leader's backstop: {:?}", f.starts);
        assert!(lines.iter().any(|l| l.contains("the leader reports the group lost")), "{lines:#?}");
        assert!(lines.iter().any(|l| l.contains("rank 0: still running")), "{lines:#?}");
    }

    #[test]
    fn a_group_reported_lost_with_no_rank_ended_is_a_loss_not_a_vanishing() {
        let a = Attempt {
            exits: vec![None, None],
            was_ready: true,
            group_lost: Some("rank 1 stopped answering".into()),
            ..Default::default()
        };
        match classify(&a) {
            Class::Lost(why) => assert!(why.contains("the leader reports the group lost: rank 1 stopped answering"), "{why}"),
            c => panic!("{c:?}"),
        }
        let mut f = Fake::new(vec![Step::Run { ends: vec![] }, Step::Run { ends: vec![(300, 0, 0), (301, 1, 0)] }]);
        f.lost_after = Some(30);
        // the second run is not lost: `lost_after` applies to the first only
        let (code, lines) = run(&mut f);
        assert_eq!(code, 0, "{lines:#?}");
        assert!(lines.iter().any(|l| l.contains("group lost (the leader reports the group lost")), "{lines:#?}");
    }

    #[test]
    fn sigterm_during_the_backoff_does_not_start_again() {
        let mut f = Fake::new(vec![Step::Run { ends: vec![(5, 0, 70), (6, 1, 70)] }, Step::Run { ends: vec![] }]);
        f.stop_after = Some(20);
        let (code, _) = run(&mut f);
        assert_eq!(code, 0);
        assert_eq!(f.starts.len(), 1);
    }
}
