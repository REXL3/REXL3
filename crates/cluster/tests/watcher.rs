//! The run script's watcher, run for real: a rank's generated run script under bash with a stand-in server (a python3 script), no
//! GPU, no ssh, no real lock. What it checks:
//! - a server that exits by itself: the run script ends within a few seconds and leaves no process behind (the old watcher
//!   orphaned a `sleep <budget>` after every clean down);
//! - the run-time budget: SIGTERM at the budget, SIGKILL 20 s later for a server that ignores SIGTERM, the reason in `rank0.stop`;
//! - a lost group: rank 0's run script stops a leader whose `/health` says the group is lost (it used to wait for the budget);
//! - signals reach only the process the script started (`same`: start time and binary);
//! - descriptors the starter left open (a lock) are closed before anything else runs.

use gb_cluster::schema::Cluster;
use gb_cluster::script;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

const SMOKE: &str = include_str!("../../../docs/examples/cluster.smoke-2node.json");

/// The stand-in server. FAKE_MODE: `exit:<s>` (exits 0 after s seconds), `hang` (runs until signalled), `noterm` (ignores SIGTERM),
/// `lost:<s>` (serves /health on --port: ready, then the group lost after s seconds; runs until signalled).
const FAKE: &str = r#"#!/usr/bin/env python3
import http.server, os, signal, sys, time
mode = os.environ.get("FAKE_MODE", "exit:1")
args = sys.argv[1:]
port = int(args[args.index("--port") + 1]) if "--port" in args else 0
t0 = time.time()
if mode == "noterm":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if mode.startswith("exit:"):
    time.sleep(float(mode[5:]))
    sys.exit(0)
if mode.startswith("lost:"):
    after = float(mode[5:])
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            lost = time.time() - t0 >= after
            if lost:
                body = b'{"status":"group_lost","tensor_parallel":{"ranks":2,"state":"lost","lost":"rank 1 gone"}}'
            else:
                body = b'{"status":"ready","tensor_parallel":{"ranks":2,"state":"up"}}'
            self.send_response(503 if lost else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a):
            pass
    http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
while True:
    time.sleep(1)
"#;

fn have(tool: &str) -> bool {
    Command::new("sh").arg("-c").arg(format!("command -v {tool} >/dev/null")).status().is_ok_and(|s| s.success())
}

fn tools_present() -> bool {
    let ok = ["bash", "python3", "flock", "curl"].iter().all(|t| have(t));
    if !ok {
        eprintln!("not run: bash, python3, flock and curl are needed");
    }
    ok
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

struct Rig {
    dir: PathBuf,
    run: PathBuf,
}

impl Rig {
    /// Rank 0 of the smoke file's two-rank group, its server the stand-in, its files under a fresh temp directory, the barrier's go
    /// already given; `budget`: the run-time budget; `lost_stop_s`: the lost-group delay.
    fn new(name: &str, budget: Option<u64>, lost_stop_s: u64) -> Rig {
        let dir = std::env::temp_dir().join(format!("rexl3-watcher-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let fake = dir.join("fake-serve");
        std::fs::write(&fake, FAKE).unwrap();
        std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut c = Cluster::parse(SMOKE).unwrap();
        c.binary = fake.to_string_lossy().into_owned();
        c.state_dir = dir.join("state").to_string_lossy().into_owned();
        c.nccl.env_file = None;
        c.lock_wait_s = 5;
        c.max_runtime_s = budget;
        for n in &mut c.nodes {
            n.gpu_lock = dir.join("gpu.lock").to_string_lossy().into_owned();
        }
        c.groups[0].http_port = free_port();
        let g = c.groups[0].clone();
        assert!(g.ranks.len() == 2, "the smoke group has two ranks");
        let sd = PathBuf::from(c.group_state_dir(&g));
        std::fs::create_dir_all(&sd).unwrap();
        std::fs::write(sd.join("rank0.go"), b"").unwrap();
        std::fs::write(sd.join("launcher.beat"), b"").unwrap();
        let run = sd.join("rank0.run");
        std::fs::write(&run, script::run_script_opts(&c, &g, 0, None, lost_stop_s)).unwrap();
        Rig { dir, run }
    }

    fn state(&self, ext: &str) -> Option<String> {
        std::fs::read_to_string(self.run.with_extension(ext)).ok().map(|s| s.trim().to_string())
    }

    /// Start the run script in its own process group (its pgid is its pid), the stand-in in `mode`.
    fn start(&self, mode: &str) -> Child {
        Command::new("bash").arg(&self.run).env("FAKE_MODE", mode).process_group(0).spawn().unwrap()
    }
}

impl Drop for Rig {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// The pids whose process group is `pgid` (the run script's: everything it started, unless something left the group).
fn group_members(pgid: u32) -> Vec<u32> {
    let mut v = Vec::new();
    for e in std::fs::read_dir("/proc").unwrap().flatten() {
        let Ok(pid) = e.file_name().to_string_lossy().parse::<u32>() else { continue };
        let Ok(stat) = std::fs::read_to_string(e.path().join("stat")) else { continue };
        let Some((_, rest)) = stat.rsplit_once(") ") else { continue };
        let f: Vec<&str> = rest.split_whitespace().collect();
        if f.get(2).and_then(|p| p.parse::<u32>().ok()) == Some(pgid) && f.first() != Some(&"Z") {
            v.push(pid);
        }
    }
    v
}

/// Wait (bounded) for the run script; on the bound, kill its whole group (ours) and fail.
fn finish(mut ch: Child, bound: Duration) -> (i32, Duration) {
    let t0 = Instant::now();
    let pgid = ch.id();
    loop {
        if let Some(st) = ch.try_wait().unwrap() {
            return (st.code().unwrap_or(-1), t0.elapsed());
        }
        if t0.elapsed() > bound {
            let _ = Command::new("kill").arg("-KILL").arg(format!("-{pgid}")).status();
            let _ = ch.wait();
            panic!("the run script did not end within {bound:?}");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Nothing of the run script's group is alive within `bound` after it ended.
fn assert_nothing_left(pgid: u32, bound: Duration) {
    let t0 = Instant::now();
    loop {
        let left = group_members(pgid);
        if left.is_empty() {
            return;
        }
        if t0.elapsed() > bound {
            let what: Vec<String> = left
                .iter()
                .map(|p| {
                    format!(
                        "{p}: {}",
                        std::fs::read_to_string(format!("/proc/{p}/cmdline")).unwrap_or_default().replace('\0', " ")
                    )
                })
                .collect();
            for p in &left {
                let _ = Command::new("kill").arg("-KILL").arg(p.to_string()).status();
            }
            panic!("left behind after the run script ended: {what:?}");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[test]
fn a_server_that_exits_by_itself_ends_the_script_promptly_and_leaves_nothing_behind() {
    if !tools_present() {
        return;
    }
    // a long budget: the old watcher left `sleep 600` running after this
    let r = Rig::new("clean", Some(600), script::LOST_STOP_S);
    let ch = r.start("exit:1");
    let pgid = ch.id();
    let (code, dt) = finish(ch, Duration::from_secs(30));
    assert_eq!(code, 0);
    assert!(dt < Duration::from_secs(10), "the script took {dt:?} after a 1 s server");
    assert_eq!(r.state("exit").as_deref(), Some("0"));
    assert_eq!(r.state("state").as_deref(), Some("exited"));
    assert_eq!(r.state("stop"), None, "nothing was stopped");
    assert_nothing_left(pgid, Duration::from_secs(5));
}

#[test]
fn the_budget_stops_the_server_with_sigterm_and_records_why() {
    if !tools_present() {
        return;
    }
    let r = Rig::new("budget", Some(3), script::LOST_STOP_S);
    let ch = r.start("hang");
    let pgid = ch.id();
    let (code, dt) = finish(ch, Duration::from_secs(40));
    assert_eq!(code, 128 + 15, "SIGTERM ends the stand-in");
    assert!(dt >= Duration::from_secs(3) && dt < Duration::from_secs(15), "{dt:?}");
    assert_eq!(r.state("stop").as_deref(), Some("budget"));
    assert_eq!(r.state("exit").as_deref(), Some("143"));
    assert_nothing_left(pgid, Duration::from_secs(5));
}

#[test]
fn a_server_that_ignores_sigterm_gets_sigkill_twenty_seconds_later() {
    if !tools_present() {
        return;
    }
    let r = Rig::new("noterm", Some(2), script::LOST_STOP_S);
    let ch = r.start("noterm");
    let pgid = ch.id();
    let (code, dt) = finish(ch, Duration::from_secs(60));
    assert_eq!(code, 128 + 9);
    assert!(dt >= Duration::from_secs(20) && dt < Duration::from_secs(35), "{dt:?}");
    assert_eq!(r.state("stop").as_deref(), Some("budget"));
    assert_nothing_left(pgid, Duration::from_secs(5));
}

#[test]
fn rank_0_stops_a_leader_whose_health_says_the_group_is_lost() {
    if !tools_present() {
        return;
    }
    // no budget at all: before, nothing ended this script but the server itself
    let r = Rig::new("lost", None, 3);
    let ch = r.start("lost:2");
    let pgid = ch.id();
    let (code, dt) = finish(ch, Duration::from_secs(40));
    assert_eq!(code, 143);
    assert!(dt >= Duration::from_secs(5) && dt < Duration::from_secs(20), "{dt:?}");
    assert_eq!(r.state("stop").as_deref(), Some("group_lost"));
    assert_nothing_left(pgid, Duration::from_secs(5));
}

#[test]
fn a_healthy_leader_is_left_alone_until_its_budget() {
    if !tools_present() {
        return;
    }
    // /health says ready for 60 s: the watcher must not stop it before the 6 s budget, and then for the budget
    let r = Rig::new("healthy", Some(6), 3);
    let ch = r.start("lost:60");
    let (code, dt) = finish(ch, Duration::from_secs(40));
    assert_eq!(code, 143);
    assert!(dt >= Duration::from_secs(6), "{dt:?}");
    assert_eq!(r.state("stop").as_deref(), Some("budget"));
}

#[test]
fn signals_go_only_to_the_process_with_the_recorded_start_time_and_binary() {
    if !tools_present() {
        return;
    }
    let sh = format!(
        "{}\
         sleep 30 & a=$!; sleep 0.05; sleep 30 & b=$!\n\
         bin=sleep; binr=/nonexistent\n\
         pid=$a; st0=$(stime $a); same && echo a_same\n\
         pid=$b; same || echo b_other_start\n\
         st0=$(stime $b); same && echo b_same\n\
         bin=/no/such/server; same || echo b_other_binary\n\
         bin=sleep; kill $b; wait $b 2>/dev/null; same || echo b_gone\n\
         kill $a; wait $a 2>/dev/null\n",
        script::PROC_IDENTITY_FNS
    );
    let out = Command::new("bash").arg("-c").arg(&sh).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    for want in ["a_same", "b_other_start", "b_same", "b_other_binary", "b_gone"] {
        assert!(text.lines().any(|l| l == want), "{want} missing in {text:?}");
    }
}

#[test]
fn a_lock_the_starter_held_is_not_inherited_by_the_server() {
    if !tools_present() {
        return;
    }
    let r = Rig::new("fd", Some(600), script::LOST_STOP_S);
    let lock = r.dir.join("caller.lock");
    // the starter holds `caller.lock` on fd 8, starts the run script, then closes its own copy: only an inherited copy keeps it held
    let mut ch = Command::new("bash")
        .arg("-c")
        .arg(format!("exec 8>{l}; flock 8; bash {run} & exec 8>&-; wait", l = lock.display(), run = r.run.display()))
        .env("FAKE_MODE", "exit:4")
        .process_group(0)
        .spawn()
        .unwrap();
    std::thread::sleep(Duration::from_millis(1500));
    let free = Command::new("flock").arg("-n").arg(&lock).arg("true").status().unwrap().success();
    let pgid = ch.id();
    let _ = ch.wait();
    assert!(free, "the caller's lock was still held while the server ran: inherited");
    assert_nothing_left(pgid, Duration::from_secs(5));
    let _ = Path::new(&lock);
}
