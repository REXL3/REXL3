//! `rexl3-cluster`: validate, check, up, down, status, supervise, logs, exec and env for a tensor-parallel group of any rank count.
//! See docs/CLUSTER_OPERATIONS.md.

use gb_cluster::check;
use gb_cluster::exit;
use gb_cluster::http;
use gb_cluster::ops::{self, Env, UpOptions};
use gb_cluster::schema::{Cluster, Group};
use gb_cluster::script::{self, StatusOpts};
use gb_cluster::signal;
use gb_cluster::supervise::{self, ClusterRunner};
use gb_cluster::transport::{describe, SshTransport, Transport};
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::time::{Duration, Instant};

const USAGE: &str = "\
usage: rexl3-cluster COMMAND --cluster FILE [--group NAME | --all] [options]

commands:
  validate     parse and validate the cluster file (offline: nothing is run, no ssh)
  check        pre-flight over ssh: binary hash, pack identity, NCCL env, ports, free memory against the plan, GPU idle, lock order
  up           check, then start rank r of R on each node (GPU locks in node-list order) and wait for /health/ready;
               stops every rank it started, by pid, if any fails to come up
  down         SIGTERM rank 0 (the followers exit 0 on its Shutdown), then verify: no process, GPU lock free, GPU idle
  status       which ranks are up, the leader's /health and /health/ready
  supervise    run the group under the restart-all policy: any rank exit != 0 restarts every rank (backoff, give-up rules)
  logs         the tail of a rank's log: --rank R [--lines N]
  exec         run rank R on THIS host in the foreground (per-host systemd units): --rank R
  env          shell assignments for the two-node scripts in tools/ (NODE4, LEADER_ADDR, TPPORT, BIN)

options:
  --cluster FILE        the cluster file (or REXL3_CLUSTER_FILE)
  --group NAME          the group (optional when the file has one); --all: every group (check, status, validate)
  --dry-run             print the exact commands per host instead of running them
  --skip-check          up, supervise: do not run the pre-flight first
  --max-runtime SECS    up, supervise: SIGTERM each rank after SECS (rank 0 first), so a session holds the GPU locks at most that long
  --ready-timeout SECS  up: how long to wait for /health/ready after the last rank is started (default: connect + ready timeouts + 120)
  --rank R              logs, exec
  --lines N             logs (default 40)

exit status: 0 done, 1 the operation failed, 2 usage error or invalid cluster file, 78 and 75 the supervisor gave up (refused boot twice
in a row; too many failures in the window).
Never run `up` or `supervise` under `flock` or `nrun --gpu` on the same GPU lock: the launcher takes each node's lock itself.";

struct Opts {
    command: String,
    cluster: Option<String>,
    group: Option<String>,
    all: bool,
    dry_run: bool,
    skip_check: bool,
    max_runtime: Option<u64>,
    ready_timeout: Option<u64>,
    rank: Option<usize>,
    lines: usize,
}

fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        command: String::new(),
        cluster: std::env::var("REXL3_CLUSTER_FILE").ok(),
        group: None,
        all: false,
        dry_run: false,
        skip_check: false,
        max_runtime: None,
        ready_timeout: None,
        rank: None,
        lines: 40,
    };
    let mut i = 0;
    let num = |name: &str, v: Option<&String>| -> Result<u64, String> {
        v.ok_or_else(|| format!("{name} needs a value"))?.parse::<u64>().map_err(|_| format!("{name} must be a number"))
    };
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--cluster" => {
                i += 1;
                o.cluster = Some(args.get(i).ok_or("--cluster needs a file")?.clone());
            }
            "--group" => {
                i += 1;
                o.group = Some(args.get(i).ok_or("--group needs a name")?.clone());
            }
            "--all" => o.all = true,
            "--dry-run" => o.dry_run = true,
            "--skip-check" => o.skip_check = true,
            "--max-runtime" => {
                i += 1;
                o.max_runtime = Some(num("--max-runtime", args.get(i))?);
            }
            "--ready-timeout" => {
                i += 1;
                o.ready_timeout = Some(num("--ready-timeout", args.get(i))?);
            }
            "--rank" => {
                i += 1;
                o.rank = Some(num("--rank", args.get(i))? as usize);
            }
            "--lines" => {
                i += 1;
                o.lines = num("--lines", args.get(i))? as usize;
            }
            "-h" | "--help" | "help" => return Err(String::new()),
            s if s.starts_with('-') => return Err(format!("unknown option {s}")),
            s => {
                if o.command.is_empty() {
                    o.command = s.to_string();
                } else {
                    return Err(format!("unexpected argument {s}"));
                }
            }
        }
        i += 1;
    }
    if o.command.is_empty() {
        return Err("no command".into());
    }
    Ok(o)
}

struct RealEnv {
    t: SshTransport,
    t0: Instant,
}

impl Env for RealEnv {
    fn transport(&self) -> &(dyn Transport + Sync) {
        &self.t
    }

    fn http_get(&self, host: &str, port: u16, path: &str) -> Result<http::Response, String> {
        http::get(host, port, path, Duration::from_secs(3))
    }

    fn sleep(&self, d: Duration) {
        std::thread::sleep(d);
    }

    fn now_ms(&self) -> u64 {
        self.t0.elapsed().as_millis() as u64
    }

    fn cancelled(&self) -> bool {
        signal::requested()
    }
}

fn load(o: &Opts) -> Result<Cluster, i32> {
    let Some(path) = &o.cluster else {
        eprintln!("error: no cluster file: pass --cluster FILE (or set REXL3_CLUSTER_FILE)");
        return Err(exit::USAGE);
    };
    let text = std::fs::read_to_string(path).map_err(|e| {
        eprintln!("error: cannot read {path}: {e}");
        exit::USAGE
    })?;
    Cluster::parse(&text).map_err(|errs| {
        eprintln!("error: {path} is not a valid cluster file ({} problem(s)):", errs.len());
        for e in errs {
            eprintln!("  - {e}");
        }
        exit::USAGE
    })
}

/// The groups a command acts on: `--group`, the only group, every group with `--all` (or by default for read-only commands).
fn pick<'a>(c: &'a Cluster, o: &Opts, many_ok: bool) -> Result<Vec<&'a Group>, i32> {
    if o.all || (o.group.is_none() && many_ok) {
        return Ok(c.groups.iter().collect());
    }
    match c.select_group(o.group.as_deref()) {
        Ok(g) => Ok(vec![g]),
        Err(e) => {
            eprintln!("error: {e}");
            Err(exit::USAGE)
        }
    }
}

fn cmd_validate(c: &Cluster) -> i32 {
    let disabled = c.nodes.iter().filter(|n| n.disabled).count();
    println!(
        "ok: cluster {}: {} node(s) ({} disabled and never used), {} group(s)",
        c.name,
        c.nodes.len(),
        disabled,
        c.groups.len()
    );
    for g in &c.groups {
        println!(
            "  group {}: {} rank(s) on {}; rendezvous {}; leader http://{}:{}",
            g.name,
            g.ranks.len(),
            g.ranks.join(", "),
            c.tp_peer(g),
            c.rank_node(g, 0).map(|n| n.address.as_str()).unwrap_or("?"),
            g.http_port
        );
    }
    exit::OK
}

fn cmd_check(c: &Cluster, groups: &[&Group], dry: bool) -> i32 {
    let env = RealEnv { t: SshTransport::new(c), t0: Instant::now() };
    let mut rc = exit::OK;
    for g in groups {
        println!("check: cluster {} group {}", c.name, g.name);
        if dry {
            print!("{}", check::dry_run(c, g));
            continue;
        }
        let rep = check::run(c, g, env.transport_sync());
        print!("{}", rep.render());
        if rep.failed() {
            rc = exit::FAILED;
        }
    }
    rc
}

impl RealEnv {
    fn transport_sync(&self) -> &(dyn Transport + Sync) {
        &self.t
    }
}

fn up_options(c: &Cluster, o: &Opts) -> UpOptions {
    let default = c.tp.connect_timeout_s + c.tp.ready_timeout_s + 120;
    UpOptions { budget_s: o.max_runtime, ready_timeout: Duration::from_secs(o.ready_timeout.unwrap_or(default)) }
}

fn cmd_up(c: &Cluster, g: &Group, o: &Opts) -> i32 {
    if o.dry_run {
        print!("{}", ops::up_dry_run(c, g, o.max_runtime, !o.skip_check));
        return exit::OK;
    }
    println!("up: cluster {} group {} ({} ranks)", c.name, g.name, g.ranks.len());
    let order: Vec<String> = c.lock_order(g).iter().map(|(r, n)| format!("{} (rank {r})", n.name)).collect();
    if g.ranks.len() > 1 {
        println!(
            "  GPU locks are taken in this order and held at a start barrier; the servers start together once every lock is held: {}",
            order.join(", ")
        );
    } else {
        println!("  GPU locks are taken in this order, each rank started once its lock is held: {}", order.join(", "));
    }
    signal::install();
    let env = RealEnv { t: SshTransport::new(c), t0: Instant::now() };
    if !o.skip_check {
        let rep = check::run(c, g, env.transport_sync());
        print!("{}", rep.render());
        if rep.failed() {
            eprintln!("up: the pre-flight failed; nothing was started");
            return exit::FAILED;
        }
    }
    let mut out = std::io::stdout();
    let t0 = Instant::now();
    match ops::up(&env, c, g, up_options(c, o), &mut out) {
        Ok(rep) => {
            println!(
                "up: group {} is ready: {} ranks, {:.1} s from the first start",
                g.name,
                rep.ranks.len(),
                rep.ready_after_ms as f64 / 1000.0
            );
            for (rank, node, pid, at) in &rep.ranks {
                println!("  rank {rank} on {node}: pid {pid}, lock held and started at {:.1} s", *at as f64 / 1000.0);
            }
            println!("  leader: http://{}:{}", c.rank_node(g, 0).map(|n| n.address.as_str()).unwrap_or("?"), g.http_port);
            println!("  stop with: rexl3-cluster down --cluster {} --group {}", o.cluster.as_deref().unwrap_or("FILE"), g.name);
            let _ = out.flush();
            exit::OK
        }
        Err(f) => {
            eprintln!("up: FAILED after {:.1} s: {}", t0.elapsed().as_secs_f64(), f.message);
            exit::FAILED
        }
    }
}

fn cmd_down(c: &Cluster, g: &Group, dry: bool) -> i32 {
    println!("down: cluster {} group {}", c.name, g.name);
    if dry {
        for (rank, node) in c.lock_order(g) {
            print!(
                "{}",
                describe(
                    &c.ssh_options,
                    node,
                    &format!("status of rank {rank}"),
                    &script::status_script(c, g, &[rank], node, StatusOpts { lock: true, gpu: true })
                )
            );
        }
        if let Some(n0) = c.rank_node(g, 0) {
            print!(
                "{}",
                describe(
                    &c.ssh_options,
                    n0,
                    "SIGTERM to rank 0 (by pid, only when the pid's command line holds the binary)",
                    &script::signal_script(c, g, 0, "TERM")
                )
            );
        }
        println!("# then every rank's status is polled until it exits (followers exit 0 on rank 0's Shutdown); a rank still running after");
        println!("# --shutdown-grace + 20 s is sent SIGTERM and 5 s later SIGKILL, by pid; each node is then verified (no process, lock free, GPU idle)");
        return exit::OK;
    }
    signal::install();
    let env = RealEnv { t: SshTransport::new(c), t0: Instant::now() };
    let mut out = std::io::stdout();
    match ops::down(&env, c, g, &mut out) {
        Ok(rep) => {
            let exits: Vec<String> = rep
                .exits
                .iter()
                .map(|(r, e)| format!("rank {r}: {}", e.map(|c| c.to_string()).unwrap_or_else(|| "no status".into())))
                .collect();
            println!(
                "down: {} in {:.1} s{}",
                if rep.clean() { "clean" } else { "NOT CLEAN" },
                rep.total_ms as f64 / 1000.0,
                if exits.is_empty() { String::new() } else { format!(" ({})", exits.join(", ")) }
            );
            if rep.clean() {
                exit::OK
            } else {
                exit::FAILED
            }
        }
        Err(e) => {
            eprintln!("down: {e}");
            exit::FAILED
        }
    }
}

fn cmd_status(c: &Cluster, groups: &[&Group], dry: bool) -> i32 {
    let env = RealEnv { t: SshTransport::new(c), t0: Instant::now() };
    let mut rc = exit::OK;
    for g in groups {
        if dry {
            for (rank, node) in c.lock_order(g) {
                print!(
                    "{}",
                    describe(
                        &c.ssh_options,
                        node,
                        &format!("status of rank {rank}"),
                        &script::status_script(c, g, &[rank], node, StatusOpts { lock: true, gpu: false })
                    )
                );
            }
            if let Some(n0) = c.rank_node(g, 0) {
                println!("# GET http://{}:{}/health/ready and /health", n0.address, g.http_port);
            }
            continue;
        }
        let mut out = std::io::stdout();
        if !ops::status(&env, c, g, &mut out) {
            rc = exit::FAILED;
        }
    }
    rc
}

fn cmd_logs(c: &Cluster, g: &Group, o: &Opts) -> i32 {
    let Some(rank) = o.rank else {
        eprintln!("error: logs needs --rank R");
        return exit::USAGE;
    };
    let Some(node) = c.rank_node(g, rank) else {
        eprintln!("error: group {} has no rank {rank}", g.name);
        return exit::USAGE;
    };
    let sc = script::tail_script(c, g, rank, o.lines);
    if o.dry_run {
        print!("{}", describe(&c.ssh_options, node, &format!("tail the log of rank {rank}"), &sc));
        return exit::OK;
    }
    match SshTransport::new(c).run(node, &sc, Duration::from_secs(30)) {
        Ok(e) => {
            print!("{}", e.stdout);
            exit::OK
        }
        Err(e) => {
            eprintln!("error: {e}");
            exit::FAILED
        }
    }
}

fn cmd_exec(c: &Cluster, g: &Group, o: &Opts) -> i32 {
    let Some(rank) = o.rank else {
        eprintln!("error: exec needs --rank R");
        return exit::USAGE;
    };
    let Some(node) = c.rank_node(g, rank) else {
        eprintln!("error: group {} has no rank {rank}", g.name);
        return exit::USAGE;
    };
    let sc = script::exec_script(c, g, rank);
    if o.dry_run {
        print!("{sc}");
        return exit::OK;
    }
    // refuse to start rank r on a host that is not its node: a unit file copied to the wrong box must not take another rank's place
    if let Some(want) = &node.expected_hostname {
        let host = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_default();
        if host.trim() != want {
            eprintln!("error: rank {rank} is node {} (host name {want}) but this host is `{}`", node.name, host.trim());
            return exit::USAGE;
        }
    }
    let err = std::process::Command::new("bash").arg("-c").arg(&sc).exec();
    eprintln!("error: cannot exec bash: {err}");
    exit::FAILED
}

fn cmd_env(c: &Cluster, g: &Group) -> i32 {
    // the two-node scripts name the follower's ssh target and the leader's address; defaults only (an exported variable wins)
    let (Some(n0), Some(n1)) = (c.rank_node(g, 0), c.rank_node(g, 1)) else {
        eprintln!("error: group {} has no rank 1: the two-node scripts need a pair", g.name);
        return exit::USAGE;
    };
    println!("NODE4=${{NODE4:-{}}}", script::sq(&format!("{}@{}", n1.user, n1.address)));
    println!("LEADER_ADDR=${{LEADER_ADDR:-{}}}", script::sq(&n0.fabric_address));
    println!("TPPORT=${{TPPORT:-{}}}", g.tp_port);
    println!("BIN=${{BIN:-{}}}", script::qpath(&c.binary));
    println!("export NODE4 LEADER_ADDR TPPORT BIN");
    exit::OK
}

fn cmd_supervise(c: &Cluster, g: &Group, o: &Opts) -> i32 {
    println!("supervise: cluster {} group {} ({} ranks)", c.name, g.name, g.ranks.len());
    let s = &c.supervisor;
    println!(
        "  policy: restart all ranks together; backoff {} s doubling to {} s; refused twice in a row stops (exit {}); more than {} failures in {} s stops (exit {});",
        s.backoff_base_s, s.backoff_cap_s, exit::GAVE_UP_REFUSED, s.max_restarts, s.window_s, exit::GAVE_UP_RATE
    );
    println!(
        "  a run ready for {} s resets the counters; after a rank ends the others get {} s to end by themselves",
        s.stable_after_s, s.settle_s
    );
    if o.dry_run {
        let mut out = std::io::stdout();
        let _ = out.flush();
        for (rank, node) in c.lock_order(g) {
            print!(
                "{}",
                describe(
                    &c.ssh_options,
                    node,
                    &format!("start rank {rank} (each attempt)"),
                    &script::launch_script(c, g, rank, o.max_runtime)
                )
            );
        }
        return exit::OK;
    }
    signal::install();
    let env = RealEnv { t: SshTransport::new(c), t0: Instant::now() };
    if !o.skip_check {
        let rep = check::run(c, g, env.transport_sync());
        print!("{}", rep.render());
        if rep.failed() {
            eprintln!("supervise: the pre-flight failed; nothing was started");
            return exit::FAILED;
        }
    }
    let mut out = std::io::stdout();
    let mut runner = ClusterRunner { env: &env, c, g, up: up_options(c, o), out: &mut out };
    let t0 = Instant::now();
    let code = supervise::supervise(&mut runner, &c.supervisor, &mut |l| {
        println!("[{:>7.1}s] supervise: {l}", t0.elapsed().as_secs_f64());
    });
    println!("supervise: exiting with status {code}");
    code
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let o = match parse_args(&args) {
        Ok(o) => o,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}");
            }
            eprintln!("{USAGE}");
            std::process::exit(if e.is_empty() { exit::OK } else { exit::USAGE });
        }
    };
    let c = match load(&o) {
        Ok(c) => c,
        Err(code) => std::process::exit(code),
    };
    let one = |many_ok: bool| pick(&c, &o, many_ok);
    let code = match o.command.as_str() {
        "validate" => cmd_validate(&c),
        "check" => match one(true) {
            Ok(gs) => cmd_check(&c, &gs, o.dry_run),
            Err(e) => e,
        },
        "status" => match one(true) {
            Ok(gs) => cmd_status(&c, &gs, o.dry_run),
            Err(e) => e,
        },
        "up" => match one(false) {
            Ok(gs) => cmd_up(&c, gs[0], &o),
            Err(e) => e,
        },
        "down" => match one(false) {
            Ok(gs) => cmd_down(&c, gs[0], o.dry_run),
            Err(e) => e,
        },
        "supervise" => match one(false) {
            Ok(gs) => cmd_supervise(&c, gs[0], &o),
            Err(e) => e,
        },
        "logs" => match one(false) {
            Ok(gs) => cmd_logs(&c, gs[0], &o),
            Err(e) => e,
        },
        "exec" => match one(false) {
            Ok(gs) => cmd_exec(&c, gs[0], &o),
            Err(e) => e,
        },
        "env" => match one(false) {
            Ok(gs) => cmd_env(&c, gs[0]),
            Err(e) => e,
        },
        other => {
            eprintln!("error: unknown command `{other}`\n{USAGE}");
            exit::USAGE
        }
    };
    std::process::exit(code);
}
