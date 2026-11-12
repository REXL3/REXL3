//! The remote scripts: what `up`, `down`, `status` and `check` run on a node (through `bash -s`), generated from the cluster file.
//!
//! Every script is plain bash that prints `key=value` lines, so the whole tool is testable by reading its output and `--dry-run` can
//! show exactly what would run. The state of a rank lives on its node in `<state_dir>/<group>/rank<R>.{run,pid,exit,state,log}`:
//! nothing is kept on the launcher, so `down` and `status` work from any host and after a launcher that was interrupted.

use crate::schema::{Cluster, Group, Node};

/// A shell word: plain when it only has characters that need no quoting, single-quoted otherwise.
pub fn sq(s: &str) -> String {
    let plain = !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c));
    if plain {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

/// A path word: a leading `$HOME/` or `~/` is expanded by the remote shell (the nodes' home directories need not be the same path).
/// A variable's value for an `export` line: quoted literally, except the library path of `nccl.lib` (`REXL3_NCCL_LIB`), whose
/// `$HOME/` or `~/` prefix expands on the node like every other path of the cluster file.
pub fn env_value(k: &str, v: &str) -> String {
    if k == crate::schema::NCCL_LIB_ENV {
        qpath(v)
    } else {
        sq(v)
    }
}

pub fn qpath(s: &str) -> String {
    if s == "$HOME" || s == "~" {
        return "\"$HOME\"".to_string();
    }
    match s.strip_prefix("$HOME/").or_else(|| s.strip_prefix("~/")) {
        Some(rest) => format!("\"$HOME\"/{}", sq(rest)),
        None => sq(s),
    }
}

/// How stale the launcher's heartbeat (`launcher.beat`, touched by every status poll) may be before a rank that waits at the start
/// barrier gives up and releases its GPU lock: a launcher that was killed must not leave node 1's lock held for the whole lock wait.
pub const BARRIER_BEAT_STALE_S: u64 = 180;
/// On top of `lock_wait_s`, the most a rank may wait at the barrier in all (each other rank's lock wait is bounded by `lock_wait_s`).
pub const BARRIER_EXTRA_S: u64 = 120;

/// After rank 0's `/health` first says the group is lost, how long its run script lets the server end by itself (the runtime's own
/// backstop exits an idle leader 20 s after the loss, with the evidence in its exit status) before it stops it: SIGTERM, then SIGKILL
/// 20 s later. Without it a leader that does not exit kept its run script (and the GPU lock) until the run-time budget ran out.
pub const LOST_STOP_S: u64 = 25;
/// How often the run script's watcher looks at the server (and, on rank 0 of a group, its `/health`): the watcher ends within this
/// long of the server, so no long `sleep` is ever left behind.
pub const WATCH_POLL_S: u64 = 2;

/// Close every descriptor above 2 that whoever started the script left open (a caller's lock file, say): nothing a rank starts may
/// hold a lock it did not take. Bash reads a script file through fd 255, which stays.
pub const CLOSE_INHERITED_FDS: &str =
    "for f in /proc/$$/fd/*; do n=${f##*/}; [ \"$n\" -gt 2 ] 2>/dev/null && [ \"$n\" -lt 255 ] && eval \"exec $n>&-\"; done 2>/dev/null\n";

/// The shell functions a watcher uses to signal ONLY the server it started: `same` is true while `$pid` has the start time recorded
/// in `$st0` (a recycled pid has another) and is the binary `$bin` (its `/proc/<pid>/exe` is `$binr`, the binary's real path, or its
/// command line holds `$bin`, for an interpreted stand-in). `stime <pid>` prints a process's start time (clock ticks since boot).
pub const PROC_IDENTITY_FNS: &str = "stime() { sed 's/.*) //' \"/proc/$1/stat\" 2>/dev/null | awk '{print $20}'; }\n\
same() { [ -n \"$st0\" ] && [ \"$(stime \"$pid\")\" = \"$st0\" ] && { [ \"$(readlink \"/proc/$pid/exe\" 2>/dev/null)\" = \"$binr\" ] || \
tr '\\0' ' ' < \"/proc/$pid/cmdline\" 2>/dev/null | grep -qF -- \"$bin\"; }; }\n";

/// The prefix that every rank file name has.
fn rank_file(rank: usize, ext: &str) -> String {
    format!("\"$d/rank{rank}.{ext}\"")
}

/// The server's command line for rank `rank`, as shell words.
fn clean_script(c: &Cluster, script: String) -> String {
    if c.runtime_paths.is_empty() {
        return script;
    }
    let body = format!("HOME=$(/usr/bin/getent passwd \"$(/usr/bin/id -u)\" | /usr/bin/cut -d: -f6); export HOME\n{script}");
    format!("exec /usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin LANG=C.UTF-8 LC_ALL=C.UTF-8 /bin/bash --noprofile --norc -c {} \"$0\"\n", sq(&body))
}

pub fn server_cmd(c: &Cluster, g: &Group, rank: usize) -> String {
    let mut cmd = qpath(c.rank_binary(g, rank));
    for a in c.rank_args(g, rank) {
        cmd.push(' ');
        // the pack is a path (may start with $HOME); every other word is quoted as a word
        if a == c.rank_pack(g, rank) {
            cmd.push_str(&qpath(&a));
        } else {
            cmd.push_str(&sq(&a));
        }
    }
    cmd
}

/// What `rexl3-cluster exec` runs on the host of rank `rank`, in the foreground (a per-host systemd unit): the node's GPU lock, the
/// group's environment, then `exec` of the server so that the unit supervises the server itself. The lock is held for as long as the
/// server lives (the descriptor is inherited across the exec).
pub fn exec_script(c: &Cluster, g: &Group, rank: usize) -> String {
    let node = c.rank_node(g, rank).expect("validated rank map");
    let mut s = String::new();
    s.push_str(&format!(
        "# rexl3-cluster exec: cluster {} group {} rank {rank} of {} on node {}\n",
        c.name,
        g.name,
        g.ranks.len(),
        node.name
    ));
    s.push_str(CLOSE_INHERITED_FDS);
    s.push_str(&format!("lock={}\n", qpath(&node.gpu_lock)));
    s.push_str("[ -e \"$lock\" ] || : > \"$lock\"\n");
    s.push_str("exec 9<\"$lock\"\n");
    s.push_str(&format!(
        "flock -w {} 9 || {{ echo \"rexl3-cluster: the GPU lock $lock was not free in {} s\" >&2; exit 75; }}\n",
        c.lock_wait_s, c.lock_wait_s
    ));
    s.push_str("for v in $(compgen -e | grep -E '^(REXL3_|NCCL_)'); do unset \"$v\"; done\n");
    if g.ranks.len() > 1 {
        if let Some(f) = &c.nccl.env_file {
            s.push_str(&format!("envf={}\n", qpath(f)));
            s.push_str("[ -f \"$envf\" ] || { echo \"rexl3-cluster: the NCCL env file $envf is missing\" >&2; exit 78; }\n");
            s.push_str(". \"$envf\"\n");
        }
    }
    for (k, v) in c.rank_env(g, rank) {
        s.push_str(&format!("export {k}={}\n", env_value(&k, &v)));
    }
    s.push_str(&format!("exec {}\n", server_cmd(c, g, rank)));
    clean_script(c, s)
}

/// The run script of rank `rank`: takes the node's GPU lock, then runs the server as a child with the group's environment, records its
/// pid and its exit status. It is written to the node by [`launch_script`] and started detached, so `up` can return.
///
/// A watcher runs beside the server when there is a run-time budget, and always on rank 0 of a group: it polls every
/// [`WATCH_POLL_S`] s and ends with the server (it never leaves a long `sleep` behind), stops the server when the budget runs out or,
/// on rank 0, [`LOST_STOP_S`] s after `/health` first reports the group lost (the reason goes to `rank<R>.stop`), and signals only the
/// process it started ([`PROC_IDENTITY_FNS`]). The script closes every inherited descriptor first, and the watcher the GPU lock's.
pub fn run_script(c: &Cluster, g: &Group, rank: usize, budget_override_s: Option<u64>) -> String {
    run_script_opts(c, g, rank, budget_override_s, LOST_STOP_S)
}

/// [`run_script`] with the lost-group stop delay as a parameter (tests use a short one).
pub fn run_script_opts(c: &Cluster, g: &Group, rank: usize, budget_override_s: Option<u64>, lost_stop_s: u64) -> String {
    let node = c.rank_node(g, rank).expect("validated rank map");
    let mut s = String::new();
    s.push_str("#!/usr/bin/env bash\n");
    s.push_str(&format!(
        "# rexl3-cluster: cluster {} group {} rank {rank} of {} on node {}\n",
        c.name,
        g.name,
        g.ranks.len(),
        node.name
    ));
    s.push_str("# The server's pid is written once the node's GPU lock is held, its exit status when it ends.\n");
    if g.ranks.len() > 1 {
        s.push_str(
            "# Two-phase start: the lock is taken first and the server is NOT started until the launcher has seen every rank's lock\n\
             # held and writes this rank's go file, so no rank's connect clock runs while another rank still queues for its lock.\n",
        );
    }
    s.push_str(CLOSE_INHERITED_FDS);
    s.push_str(&format!("d={}\n", qpath(&c.group_state_dir(g))));
    s.push_str(&format!("bin={}\n", qpath(c.rank_binary(g, rank))));
    s.push_str(&format!("lock={}\n", qpath(&node.gpu_lock)));
    if !c.runtime_paths.is_empty() {
        s.push_str(PROC_IDENTITY_FNS);
        s.push_str(&format!("stime $$ > {}\n", rank_file(rank, "wstart")));
    }
    s.push_str(&format!("echo $$ > {}\n", rank_file(rank, "wpid")));
    s.push_str("[ -e \"$lock\" ] || : > \"$lock\"\n");
    s.push_str("exec 9<\"$lock\"\n");
    s.push_str(&format!("if ! flock -w {} 9; then\n", c.lock_wait_s));
    s.push_str(&format!("  echo lock_timeout > {}; echo 75 > {}; exit 75\n", rank_file(rank, "state"), rank_file(rank, "exit")));
    s.push_str("fi\n");
    if g.ranks.len() > 1 {
        // refused before the barrier: a missing env file must fail the whole start while nothing runs yet
        if let Some(f) = &c.nccl.env_file {
            s.push_str(&format!("envf={}\n", qpath(f)));
            s.push_str(&format!(
                "[ -f \"$envf\" ] || {{ echo env_file_missing > {}; echo 78 > {}; exit 78; }}\n",
                rank_file(rank, "state"),
                rank_file(rank, "exit")
            ));
        }
        s.push_str(&format!("echo locked > {}\n", rank_file(rank, "state")));
        s.push_str(
            "# the barrier: the lock is held, the server waits for the launcher's go (every rank's lock held). A launcher that\n",
        );
        s.push_str("# is gone is noticed: its heartbeat (touched by every status poll) goes stale and the lock is released.\n");
        s.push_str(&format!("go={}\n", rank_file(rank, "go")));
        s.push_str("beat=\"$d/launcher.beat\"; t0=$(date +%s)\n");
        s.push_str("until [ -e \"$go\" ]; do\n");
        s.push_str("  now=$(date +%s); b=$(stat -c %Y \"$beat\" 2>/dev/null || echo \"$t0\")\n");
        s.push_str(&format!(
            "  if [ $((now - b)) -gt {} ] || [ $((now - t0)) -gt {} ]; then\n",
            BARRIER_BEAT_STALE_S,
            c.lock_wait_s + BARRIER_EXTRA_S
        ));
        s.push_str(&format!(
            "    echo barrier_timeout > {}; echo 75 > {}; exit 75\n",
            rank_file(rank, "state"),
            rank_file(rank, "exit")
        ));
        s.push_str("  fi\n");
        s.push_str("  sleep 0.2 9<&-\n");
        s.push_str("done\n");
    } else {
        s.push_str(&format!("echo locked > {}\n", rank_file(rank, "state")));
    }
    s.push_str("# the environment is what the cluster file says: ambient REXL3_* and NCCL_* variables are cleared first\n");
    s.push_str("for v in $(compgen -e | grep -E '^(REXL3_|NCCL_)'); do unset \"$v\"; done\n");
    if g.ranks.len() > 1 && c.nccl.env_file.is_some() {
        s.push_str(". \"$envf\"\n");
    }
    for (k, v) in c.rank_env(g, rank) {
        s.push_str(&format!("export {k}={}\n", env_value(&k, &v)));
    }
    let cmd = server_cmd(c, g, rank);
    s.push_str(&format!("{cmd} >> {} 2>&1 &\n", rank_file(rank, "log")));
    s.push_str("pid=$!\n");
    if !c.runtime_paths.is_empty() {
        s.push_str(&format!("stime \"$pid\" > {}\n", rank_file(rank, "start")));
    }
    s.push_str(&format!("echo \"$pid\" > {}\n", rank_file(rank, "pid")));
    s.push_str(&format!("echo running > {}\n", rank_file(rank, "state")));
    let budget = c.runtime_budget(rank, budget_override_s);
    let watch_lost = rank == 0 && g.ranks.len() > 1;
    let watcher = budget.is_some() || watch_lost;
    if watcher {
        s.push_str("# the watcher: it ends within a poll of the server, signals only the server it started (same start time, same binary),\n");
        if let Some(b) = budget {
            s.push_str(&format!(
                "# stops it after {b} s (rank 0 first, followers 60 s later so that they see Shutdown): SIGTERM, SIGKILL 20 s later\n"
            ));
        }
        if watch_lost {
            s.push_str(&format!(
                "# and stops it {lost_stop_s} s after its /health first says the group is lost (a leader that does not exit by itself)\n"
            ));
            let host = match g.http_host.as_str() {
                "" | "0.0.0.0" | "::" | "[::]" => "127.0.0.1".to_string(),
                h => h.to_string(),
            };
            s.push_str(&format!(
                "health() {{ if command -v curl >/dev/null 2>&1; then curl -s -m 2 {url}; else ( exec 3<>/dev/tcp/{h}/{p} && \
printf 'GET /health HTTP/1.0\\r\\nHost: {h}\\r\\n\\r\\n' >&3 && timeout 2 cat <&3 ) 2>/dev/null; fi; }}\n",
                url = sq(&format!("http://{host}:{}/health", g.http_port)),
                h = host,
                p = g.http_port
            ));
        }
        s.push_str(PROC_IDENTITY_FNS);
        s.push_str("binr=$(readlink -f \"$bin\" 2>/dev/null); st0=$(stime \"$pid\")\n");
        s.push_str("( exec 9<&-\n");
        s.push_str("  t0=$(date +%s); why=; lost=\n");
        s.push_str("  while kill -0 \"$pid\" 2>/dev/null; do\n");
        s.push_str("    now=$(date +%s)\n");
        if let Some(b) = budget {
            s.push_str(&format!("    if [ $((now - t0)) -ge {b} ]; then why=budget; break; fi\n"));
        }
        if watch_lost {
            s.push_str("    case \"$(health)\" in\n");
            s.push_str("      *'\"status\":\"group_lost\"'*|*'\"state\":\"lost\"'*)\n");
            s.push_str(&format!(
                "        lost=${{lost:-$now}}; if [ $((now - lost)) -ge {lost_stop_s} ]; then why=group_lost; break; fi;;\n"
            ));
            s.push_str("      *) lost=;;\n");
            s.push_str("    esac\n");
        }
        s.push_str(&format!("    sleep {WATCH_POLL_S}\n"));
        s.push_str("  done\n");
        s.push_str("  if [ -n \"$why\" ]; then\n");
        s.push_str(&format!("    echo \"$why\" > {}\n", rank_file(rank, "stop")));
        if c.runtime_paths.is_empty() {
            s.push_str("    same && kill -TERM \"$pid\" 2>/dev/null\n");
        } else {
            s.push_str(&format!(
                "/usr/bin/python3 -I -S -c {} {} {rank} {} TERM only-server\n",
                sq(OWNED_SIGNAL_PY),
                qpath(&c.group_state_dir(g)),
                qpath(c.rank_binary(g, rank))
            ));
        }
        s.push_str(&format!(
            "    i=0; while kill -0 \"$pid\" 2>/dev/null && [ $i -lt {} ]; do sleep {WATCH_POLL_S}; i=$((i + 1)); done\n",
            20 / WATCH_POLL_S
        ));
        if c.runtime_paths.is_empty() {
            s.push_str("    same && kill -KILL \"$pid\" 2>/dev/null\n");
        } else {
            s.push_str(&format!(
                "/usr/bin/python3 -I -S -c {} {} {rank} {} KILL only-server\n",
                sq(OWNED_SIGNAL_PY),
                qpath(&c.group_state_dir(g)),
                qpath(c.rank_binary(g, rank))
            ));
        }
        s.push_str("  fi\n");
        s.push_str(") </dev/null >/dev/null 2>&1 &\n");
        s.push_str("watcher=$!\n");
    }
    s.push_str("wait \"$pid\"\n");
    s.push_str("code=$?\n");
    if watcher {
        // the watcher sees the server gone within one poll and ends by itself: reap it (no orphan, nothing signalled late)
        s.push_str("wait \"$watcher\" 2>/dev/null\n");
    }
    s.push_str(&format!("echo \"$code\" > {}\n", rank_file(rank, "exit")));
    s.push_str(&format!("echo exited > {}\n", rank_file(rank, "state")));
    s.push_str("exit \"$code\"\n");
    clean_script(c, s)
}

/// What `up` sends to a node for one rank: writes the run script and starts it detached (`setsid nohup`), refusing when the rank's
/// server is already running.
pub fn launch_script(c: &Cluster, g: &Group, rank: usize, budget_override_s: Option<u64>) -> String {
    let run = run_script(c, g, rank, budget_override_s);
    let mut s = String::new();
    s.push_str(&format!("d={}\n", qpath(&c.group_state_dir(g))));
    s.push_str(&format!("bin={}\n", qpath(c.rank_binary(g, rank))));
    s.push_str("mkdir -p \"$d\" || { echo state_dir_unwritable; exit 74; }\n");
    if g.ranks.len() > 1 {
        // the launcher's heartbeat starts now: a rank waiting at the start barrier gives up when it goes stale
        s.push_str("touch \"$d/launcher.beat\"\n");
    }
    s.push_str(&format!("if [ -s {} ]; then\n", rank_file(rank, "pid")));
    s.push_str(&format!("  p=$(cat {})\n", rank_file(rank, "pid")));
    s.push_str(
        "  if kill -0 \"$p\" 2>/dev/null && tr '\\0' ' ' < \"/proc/$p/cmdline\" 2>/dev/null | grep -qF -- \"$bin\"; then\n",
    );
    s.push_str("    echo \"already_running pid=$p\"; exit 3\n");
    s.push_str("  fi\n");
    s.push_str("fi\n");
    s.push_str(&format!(
        "rm -f {} {} {} {} {} {}\n",
        rank_file(rank, "pid"),
        rank_file(rank, "wpid"),
        rank_file(rank, "exit"),
        rank_file(rank, "state"),
        rank_file(rank, "go"),
        rank_file(rank, "stop")
    ));
    s.push_str(&format!(
        "[ -s {} ] && mv -f {} {}\n",
        rank_file(rank, "log"),
        rank_file(rank, "log"),
        rank_file(rank, "log.prev")
    ));
    s.push_str(&format!("cat > {} <<'REXL3_RUN'\n", rank_file(rank, "run")));
    s.push_str(&run);
    s.push_str("REXL3_RUN\n");
    s.push_str(&format!("setsid nohup bash {} </dev/null >/dev/null 2>&1 &\n", rank_file(rank, "run")));
    s.push_str(&format!("echo \"launched group {} rank {rank}\"\n", g.name));
    clean_script(c, s)
}

/// What `up` sends to a node once EVERY rank's GPU lock is held: the barrier's go for one rank. Written only when that rank's run script
/// is alive and waiting (`go rank R`), so a script that was stopped meanwhile is reported (`not_waiting`) and not "started".
pub fn go_script(c: &Cluster, g: &Group, rank: usize) -> String {
    let mut s = String::new();
    s.push_str(&format!("d={}\n", qpath(&c.group_state_dir(g))));
    s.push_str(&format!("r={rank}\n"));
    s.push_str(&format!("w=$(cat {} 2>/dev/null)\n", rank_file(rank, "wpid")));
    s.push_str(&format!(
        "if [ -n \"$w\" ] && kill -0 \"$w\" 2>/dev/null && tr '\\0' ' ' < \"/proc/$w/cmdline\" 2>/dev/null | grep -qF -- \"rank{rank}.run\"; then\n"
    ));
    s.push_str(&format!("  : > {} && echo \"go rank {rank}\"\n", rank_file(rank, "go")));
    s.push_str("else\n  echo not_waiting\nfi\n");
    clean_script(c, s)
}

/// The GPU-lock probe: `lock=free|held`, and when it is held who holds it: `lock_ours=1` when a holder is this group's server or its
/// run script on this node (their pids are in the rank's pid and wpid files), `lock_holders=` the pid and command of each holder.
/// A lock taken by another session the instant ours is released is not ours: `down` must not call that unclean.
fn lock_probe(c: &Cluster, g: &Group, node: &Node, ranks: &[usize]) -> String {
    let mut s = String::new();
    s.push_str(&format!("lock={}\n", qpath(&node.gpu_lock)));
    s.push_str(&format!("d={}\n", qpath(&c.group_state_dir(g))));
    s.push_str("ours=\n");
    for r in ranks {
        s.push_str(&format!("ours=\"$ours $(cat \"$d/rank{r}.pid\" \"$d/rank{r}.wpid\" 2>/dev/null | tr '\\n' ' ')\"\n"));
    }
    s.push_str("if [ -e \"$lock\" ]; then\n");
    s.push_str("  st=$( ( flock -n 8 && echo free || echo held ) 8<\"$lock\" )\n");
    s.push_str("  echo \"lock=$st\"\n");
    s.push_str("  if [ \"$st\" = held ]; then\n");
    s.push_str("    real=$(readlink -f \"$lock\"); mine=0; holders=\n");
    s.push_str("    for p in /proc/[0-9]*; do\n");
    s.push_str("      ls -l \"$p/fd\" 2>/dev/null | awk -v r=\"$real\" '$NF == r {f=1} END {exit !f}' || continue\n");
    s.push_str("      n=${p#/proc/}\n");
    s.push_str("      holders=\"$holders$n: $(tr '\\0' ' ' < \"$p/cmdline\" 2>/dev/null | cut -c1-100); \"\n");
    s.push_str("      case \" $ours \" in *\" $n \"*) mine=1;; esac\n");
    s.push_str("    done\n");
    s.push_str("    echo \"lock_ours=$mine\"; echo \"lock_holders=$holders\"\n");
    s.push_str("  else echo lock_ours=0; fi\n");
    s.push_str("else echo lock=free; echo lock_ours=0; fi\n");
    s
}

/// Options of [`status_script`].
#[derive(Debug, Clone, Copy, Default)]
pub struct StatusOpts {
    /// Also report whether the node's GPU lock is free (a momentary non-blocking flock).
    pub lock: bool,
    /// Also report the compute apps of the GPU and the running rexl3-serve processes.
    pub gpu: bool,
}

/// The state of the given ranks of group `g` on one node, one line a rank:
/// `rank <group> <rank> pid=<> alive=<0|1> match=<0|1> exit=<> state=<> etimes=<>`.
pub fn status_script(c: &Cluster, g: &Group, ranks: &[usize], node: &Node, o: StatusOpts) -> String {
    let rank = ranks.first().copied().unwrap_or(0);
    let mut s = String::new();
    s.push_str(&format!("d={}\n", qpath(&c.group_state_dir(g))));
    s.push_str(&format!("bin={}\n", qpath(c.rank_binary(g, rank))));
    // the launcher's heartbeat: a rank waiting at the start barrier (`run_script`) gives up when no poll has touched this for a while
    s.push_str("touch \"$d/launcher.beat\" 2>/dev/null\n");
    for r in ranks {
        s.push_str(&format!("r={r}\n"));
        s.push_str("pid=$(cat \"$d/rank$r.pid\" 2>/dev/null); ex=$(cat \"$d/rank$r.exit\" 2>/dev/null); st=$(cat \"$d/rank$r.state\" 2>/dev/null)\n");
        s.push_str("w=$(cat \"$d/rank$r.wpid\" 2>/dev/null); wrapper=0\n");
        s.push_str("if [ -n \"$w\" ] && kill -0 \"$w\" 2>/dev/null && tr '\\0' ' ' < \"/proc/$w/cmdline\" 2>/dev/null | grep -qF -- \"rank$r.run\"; then wrapper=1; fi\n");
        s.push_str("alive=0; match=0; ending=0; et=\n");
        s.push_str("if [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null; then\n");
        s.push_str("  alive=1\n");
        // a process that has exited but whose address space is still being torn down (tens of GiB of device memory take seconds)
        // answers kill -0 and has an EMPTY command line: that is ours and ending, not a recycled pid (which has another command line)
        s.push_str("  cl=$(tr '\\0' ' ' < \"/proc/$pid/cmdline\" 2>/dev/null)\n");
        s.push_str("  case \"$cl\" in *\"$bin\"*) match=1;; \"\") ending=1;; esac\n");
        s.push_str("  et=$(ps -o etimes= -p \"$pid\" 2>/dev/null | tr -d ' ')\n");
        s.push_str("fi\n");
        s.push_str(&format!(
            "echo \"rank {} $r pid=$pid alive=$alive match=$match ending=$ending wrapper=$wrapper exit=$ex state=$st etimes=$et\"\n",
            g.name
        ));
    }
    if o.lock {
        s.push_str(&lock_probe(c, g, node, ranks));
    }
    if o.gpu {
        s.push_str("echo \"gpu_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)\"\n");
        s.push_str("echo \"serve_procs=$(ps -eo comm=,args= | awk '$1 == \"rexl3-serve\" && $0 !~ /--backend mock/ {n++} END {print n+0}')\"\n");
    }
    s.push_str("echo \"mem_available_kib=$(awk '/MemAvailable/{print $2}' /proc/meminfo)\"\n");
    clean_script(c, s)
}

/// Send `sig` to the server of one rank, by the pid in its pid file and only if that pid's command line holds the binary's path. A rank
/// whose server is not up yet (its run script is still waiting for the GPU lock) has its run script signalled instead, so that an
/// interrupted `up` leaves nothing that would start a server minutes later: the script's pid is in the wpid file and only a pid whose
/// command line holds `rank<R>.run` is signalled.
pub const OWNED_SIGNAL_PY: &str = r#"import os, pathlib, signal, sys
base, rank, binary, signame = sys.argv[1:5]
only_server = len(sys.argv) > 5
root = pathlib.Path(base)
for suffix, stamp, wrapper in [('pid','start',False),('wpid','wstart',True)]:
    if wrapper and only_server: continue
    try:
        pid = int((root/('rank'+rank+'.'+suffix)).read_text())
        saved = (root/('rank'+rank+'.'+stamp)).read_text().strip()
        if pid <= 1 or not saved: continue
        fd = os.pidfd_open(pid)
        try:
            proc = pathlib.Path('/proc')/str(pid)
            current = (proc/'stat').read_text().rsplit(')',1)[1].split()[19]
            if current != saved or proc.stat().st_uid != os.getuid(): continue
            if wrapper:
                words = (proc/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
                if str(root/('rank'+rank+'.run')).encode() not in words: continue
            elif os.readlink(proc/'exe') != binary: continue
            signal.pidfd_send_signal(fd, getattr(signal,'SIG'+signame))
            print('signalled pid='+str(pid)+' '+signame)
            break
        finally: os.close(fd)
    except (OSError, ValueError, IndexError): continue
else: print('not_running')
"#;

pub fn signal_script(c: &Cluster, g: &Group, rank: usize, sig: &str) -> String {
    if !c.runtime_paths.is_empty() {
        return clean_script(
            c,
            format!(
                "/usr/bin/python3 -I -S -c {} {} {rank} {} {}\n",
                sq(OWNED_SIGNAL_PY),
                qpath(&c.group_state_dir(g)),
                qpath(c.rank_binary(g, rank)),
                sq(sig)
            ),
        );
    }
    let mut s = String::new();
    s.push_str(&format!("d={}\n", qpath(&c.group_state_dir(g))));
    s.push_str(&format!("bin={}\n", qpath(c.rank_binary(g, rank))));
    s.push_str(&format!("p=$(cat {} 2>/dev/null)\n", rank_file(rank, "pid")));
    s.push_str(&format!("w=$(cat {} 2>/dev/null)\n", rank_file(rank, "wpid")));
    s.push_str("if [ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null && tr '\\0' ' ' < \"/proc/$p/cmdline\" 2>/dev/null | grep -qF -- \"$bin\"; then\n");
    s.push_str(&format!("  kill -{sig} \"$p\" && echo \"signalled pid=$p {sig}\"\n"));
    s.push_str(&format!(
        "elif [ -n \"$w\" ] && kill -0 \"$w\" 2>/dev/null && tr '\\0' ' ' < \"/proc/$w/cmdline\" 2>/dev/null | grep -qF -- \"rank{rank}.run\"; then\n"
    ));
    s.push_str(&format!("  kill -{sig} \"$w\" && echo \"signalled pid=$w {sig} (run script, no server yet)\"\n"));
    s.push_str("else\n  echo not_running\nfi\n");
    clean_script(c, s)
}

/// The last `n` lines of a rank's log.
pub fn tail_script(c: &Cluster, g: &Group, rank: usize, n: usize) -> String {
    clean_script(c, format!("d={}\ntail -n {n} {} 2>/dev/null\n", qpath(&c.group_state_dir(g)), rank_file(rank, "log")))
}

/// The pre-flight facts of one node, as `key=value` lines. `rank` is the rank this node would host: the node that hosts rank 0 also
/// reports whether the rendezvous and HTTP ports are free, a follower pings rank 0's fabric address.
pub fn check_script(c: &Cluster, g: &Group, rank: usize, node: &Node) -> String {
    let mut s = String::new();
    s.push_str("echo \"hostname=$(hostname)\"\n");
    s.push_str("echo \"kernel=$(uname -r)\"\n");
    s.push_str("echo \"driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)\"\n");
    s.push_str("echo \"mem_available_kib=$(awk '/MemAvailable/{print $2}' /proc/meminfo)\"\n");
    s.push_str(&format!("bin={}\n", qpath(c.rank_binary(g, rank))));
    s.push_str("if [ -x \"$bin\" ]; then echo binary=ok; echo \"binary_sha256=$(sha256sum \"$bin\" | cut -d' ' -f1)\"; echo \"binary_bytes=$(stat -L -c %s \"$bin\")\"; else echo binary=missing; fi\n");
    s.push_str(&format!("pack={}\n", qpath(c.rank_pack(g, rank))));
    s.push_str("if [ -d \"$pack\" ]; then\n");
    s.push_str("  echo pack=ok\n");
    s.push_str("  echo \"pack_bytes=$(stat -L -c %s \"$pack\"/*.safetensors 2>/dev/null | awk '{s+=$1} END{print s+0}')\"\n");
    s.push_str("  echo \"pack_identity=$(cat \"$pack/config.json\" \"$pack\"/*.safetensors.index.json 2>/dev/null | sha256sum | cut -d' ' -f1)\"\n");
    s.push_str("else echo pack=missing; fi\n");
    if g.ranks.len() > 1 {
        let present = if let Some(f) = &c.nccl.env_file {
            s.push_str(&format!("envf={}\n", qpath(f)));
            s.push_str("if [ -f \"$envf\" ]; then echo envfile=ok; echo \"envfile_sha256=$(sha256sum \"$envf\" | cut -d' ' -f1)\"; else echo envfile=missing; fi\n");
            s.push_str("if [ -f \"$envf\" ]; then\n");
            true
        } else {
            s.push_str("if true; then\n");
            false
        };
        // the environment the run script would build: ambient cleared, the file sourced, the pins exported
        s.push_str("  bash <<'REXL3_ENV'\n");
        s.push_str("for v in $(compgen -e | grep -E '^(REXL3_|NCCL_)'); do unset \"$v\"; done\n");
        if let (true, Some(f)) = (present, &c.nccl.env_file) {
            s.push_str(&format!(". {} >/dev/null 2>&1\n", qpath(f)));
        }
        for (k, v) in c.rank_env(g, rank) {
            s.push_str(&format!("export {k}={}\n", env_value(&k, &v)));
        }
        s.push_str("env | grep -E '^(NCCL_|REXL3_|LD_LIBRARY_PATH=)' | sort | sed 's/^/resolved./'\n");
        // the library the runtime would load, in its own order (crates/cuda/src/nccl.rs `resolve_library`): $REXL3_NCCL_LIB, then
        // libnccl.so.2 through the loader's search (LD_LIBRARY_PATH, then the ldconfig cache)
        s.push_str("p=\n");
        s.push_str("if [ -n \"$REXL3_NCCL_LIB\" ] && [ -e \"$REXL3_NCCL_LIB\" ]; then p=$REXL3_NCCL_LIB\n");
        s.push_str("else for d in $(echo \"$LD_LIBRARY_PATH\" | tr ':' ' '); do if [ -e \"$d/libnccl.so.2\" ]; then p=$d/libnccl.so.2; break; fi; done\n");
        s.push_str("  if [ -z \"$p\" ]; then p=$(/sbin/ldconfig -p 2>/dev/null | awk '$1 == \"libnccl.so.2\" {print $NF; exit}'); fi; fi\n");
        s.push_str("if [ -n \"$p\" ]; then p=$(readlink -f \"$p\"); echo \"nccl_lib=$p\"; echo \"nccl_lib_sha256=$(sha256sum \"$p\" | cut -d' ' -f1)\"; fi\n");
        s.push_str("REXL3_ENV\n");
        s.push_str("fi\n");
    }
    s.push_str("env | grep -E '^(REXL3_|NCCL_)' | sed 's/^/ambient./'\n");
    s.push_str("echo \"gpu_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)\"\n");
    s.push_str("echo \"serve_procs=$(ps -eo comm=,args= | awk '$1 == \"rexl3-serve\" && $0 !~ /--backend mock/ {n++} END {print n+0}')\"\n");
    s.push_str(&lock_probe(c, g, node, &[rank]));
    s.push_str(&format!("d={}\n", qpath(&c.group_state_dir(g))));
    s.push_str("if mkdir -p \"$d\" 2>/dev/null && [ -w \"$d\" ]; then echo state_dir=ok; else echo state_dir=unwritable; fi\n");
    s.push_str(&format!("bin={}\n", qpath(c.rank_binary(g, rank))));
    s.push_str(&format!("p=$(cat {} 2>/dev/null)\n", rank_file(rank, "pid")));
    s.push_str("if [ -n \"$p\" ] && kill -0 \"$p\" 2>/dev/null && tr '\\0' ' ' < \"/proc/$p/cmdline\" 2>/dev/null | grep -qF -- \"$bin\"; then echo \"rank_running=$p\"; fi\n");
    if rank == 0 {
        for p in [g.tp_port, g.http_port] {
            s.push_str(&format!("if ss -ltnH \"( sport = :{p} )\" 2>/dev/null | grep -q .; then echo port.{p}=busy; else echo port.{p}=free; fi\n"));
        }
    } else if let Some(n0) = c.rank_node(g, 0) {
        let a = &n0.fabric_address;
        s.push_str(&format!(
            "if ping -c1 -W2 {} >/dev/null 2>&1; then echo ping.rank0=ok; else echo ping.rank0=fail; fi\n",
            sq(a)
        ));
    }
    clean_script(c, s)
}

/// `key=value` lines to a map (the first `=` splits; lines without one are ignored; a repeated key keeps the last value).
pub fn parse_facts(out: &str) -> std::collections::BTreeMap<String, String> {
    let mut m = std::collections::BTreeMap::new();
    for l in out.lines() {
        if let Some((k, v)) = l.split_once('=') {
            if !k.is_empty() && !k.contains(' ') {
                m.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
    }
    m
}

/// One rank's line of a status script.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RankState {
    pub group: String,
    pub rank: usize,
    pub pid: Option<u32>,
    pub alive: bool,
    /// The pid's command line holds the binary (so it is ours, not a recycled pid).
    pub matches: bool,
    /// The pid exists with an empty command line: the server has exited and its address space is still being torn down.
    pub ending: bool,
    /// The rank's run script is alive (it may still be waiting for the GPU lock, with no server yet).
    pub wrapper: bool,
    pub exit: Option<i32>,
    pub state: String,
    pub etimes: Option<u64>,
}

impl RankState {
    /// The server of this rank is running, or has exited and is still being torn down (its GPU lock is still held and its exit status
    /// is not yet written): either way it has not ended. Found live (2026-10-11): a leader that `_exit(70)`s with 36 GiB mapped
    /// answers `kill -0` with an empty command line for seconds, and was read as "vanished, killed (137)".
    pub fn running(&self) -> bool {
        self.alive && (self.matches || self.ending)
    }

    /// Something of this rank is alive: its server, or its run script waiting for the GPU lock.
    pub fn active(&self) -> bool {
        self.running() || self.wrapper
    }
}

pub fn parse_rank_lines(out: &str) -> Vec<RankState> {
    let mut v = Vec::new();
    for l in out.lines() {
        let mut it = l.split_whitespace();
        if it.next() != Some("rank") {
            continue;
        }
        let (Some(group), Some(rank)) = (it.next(), it.next().and_then(|r| r.parse::<usize>().ok())) else { continue };
        let mut st = RankState { group: group.to_string(), rank, ..Default::default() };
        for kv in it {
            let Some((k, val)) = kv.split_once('=') else { continue };
            match k {
                "pid" => st.pid = val.parse().ok(),
                "alive" => st.alive = val == "1",
                "match" => st.matches = val == "1",
                "ending" => st.ending = val == "1",
                "wrapper" => st.wrapper = val == "1",
                "exit" => st.exit = val.parse().ok(),
                "state" => st.state = val.to_string(),
                "etimes" => st.etimes = val.parse().ok(),
                _ => {}
            }
        }
        v.push(st);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoting_is_plain_when_it_can_be_and_safe_when_it_must() {
        assert_eq!(sq("--slots"), "--slots");
        assert_eq!(sq("=enP7s7"), "=enP7s7");
        assert_eq!(sq("a b"), "'a b'");
        assert_eq!(sq("it's"), "'it'\\''s'");
        assert_eq!(sq(""), "''");
        assert_eq!(sq("$(rm -rf /)"), "'$(rm -rf /)'");
        assert_eq!(qpath("$HOME/x y/z"), "\"$HOME\"/'x y/z'");
        assert_eq!(qpath("~/p/q"), "\"$HOME\"/p/q");
        assert_eq!(qpath("/abs/p"), "/abs/p");
    }

    #[test]
    fn rank_lines_and_facts_parse() {
        let s = parse_rank_lines(
            "rank g0 1 pid=4242 alive=1 match=1 ending=0 exit= state=running etimes=17\nnoise\n\
             rank g0 0 pid= alive=0 match=0 ending=0 exit=70 state=exited etimes=\n\
             rank g0 2 pid=77 alive=1 match=0 ending=1 exit= state=running etimes=9\n\
             rank g0 3 pid=78 alive=1 match=0 ending=0 exit= state=running etimes=9\n",
        );
        assert_eq!(s.len(), 4);
        assert!(s[0].running() && s[0].pid == Some(4242) && s[0].etimes == Some(17) && s[0].exit.is_none());
        assert!(!s[1].running() && s[1].exit == Some(70) && s[1].state == "exited");
        // exited but still being torn down: not ended; a live pid with another command line (a recycled pid) is not ours
        assert!(s[2].running() && s[2].ending && !s[2].matches);
        assert!(!s[3].running());
        let f = parse_facts("hostname=h1\nmem_available_kib=100\nresolved.NCCL_IB_HCA==a:1,b:1\njunk\n");
        assert_eq!(f["resolved.NCCL_IB_HCA"], "=a:1,b:1");
        assert_eq!(f["hostname"], "h1");
    }
}
