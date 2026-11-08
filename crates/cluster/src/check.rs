//! `rexl3-cluster check`: the pre-flight. Facts come from one script per node ([`crate::script::check_script`], run in parallel over
//! the transport); [`evaluate`] turns them into findings and is a pure function of the cluster file and the facts, so every rule is
//! tested without a node.
//!
//! What it proves: the file is valid and names no disabled node; every rank's node is reachable (and is the host it should be); the
//! binary, the pack identity, the NCCL environment file and the NCCL library are the same bytes on every node; nothing else holds a GPU
//! or its lock; the rendezvous and HTTP ports are free; each rank has the free memory the plan needs. What it cannot prove: the fabric
//! carries NCCL (that is the boot's rendezvous), and the plan itself (the runtime plans from each rank's `MemAvailable` at boot).

use crate::schema::{Cluster, Group};
use crate::script;
use crate::transport::{describe, Transport};
use std::collections::BTreeMap;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Pass,
    Warn,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub level: Level,
    pub id: &'static str,
    pub msg: String,
}

impl Finding {
    fn new(level: Level, id: &'static str, msg: impl Into<String>) -> Finding {
        Finding { level, id, msg: msg.into() }
    }

    pub fn render(&self) -> String {
        let tag = match self.level {
            Level::Pass => "PASS",
            Level::Warn => "WARN",
            Level::Fail => "FAIL",
        };
        format!("[{tag}] {:<11} {}", self.id, self.msg)
    }
}

/// The facts of one node: its rank, its name and the parsed output of its script (or why there is none).
pub struct NodeFacts {
    pub rank: usize,
    pub node: String,
    pub facts: Result<BTreeMap<String, String>, String>,
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn short(h: &str) -> &str {
    &h[..h.len().min(12)]
}

/// All nodes carry the same value of `key`: a PASS naming it, or a FAIL listing which node has what.
fn same(
    out: &mut Vec<Finding>,
    id: &'static str,
    what: &str,
    key: &str,
    nodes: &[(&NodeFacts, &BTreeMap<String, String>)],
    level_on_diff: Level,
    show: impl Fn(&str) -> String,
) {
    let vals: Vec<(&str, Option<&String>)> = nodes.iter().map(|(n, f)| (n.node.as_str(), f.get(key))).collect();
    if vals.iter().any(|(_, v)| v.is_none()) {
        return; // the missing case was reported by the existence check
    }
    let first = vals[0].1.unwrap();
    if vals.iter().all(|(_, v)| v.unwrap() == first) {
        out.push(Finding::new(Level::Pass, id, format!("{what}: identical on {} ({})", names(nodes), show(first))));
    } else {
        let parts: Vec<String> = vals.iter().map(|(n, v)| format!("{n} has {}", show(v.unwrap()))).collect();
        out.push(Finding::new(level_on_diff, id, format!("{what} differs between the ranks: {}", parts.join(", "))));
    }
}

fn names(nodes: &[(&NodeFacts, &BTreeMap<String, String>)]) -> String {
    nodes.iter().map(|(n, _)| n.node.as_str()).collect::<Vec<_>>().join(", ")
}

/// Findings for one group from the facts of its nodes (in lock order).
pub fn evaluate(c: &Cluster, g: &Group, facts: &[NodeFacts]) -> Vec<Finding> {
    let mut out: Vec<Finding> = Vec::new();
    let ranks = g.ranks.len();

    // the file (it was validated before anything was run)
    let skipped: Vec<&str> = c.nodes.iter().filter(|n| n.disabled).map(|n| n.name.as_str()).collect();
    out.push(Finding::new(
        Level::Pass,
        "file",
        format!(
            "cluster {} group {}: {ranks} ranks on {}; disabled and never touched: {}",
            c.name,
            g.name,
            g.ranks.join(", "),
            if skipped.is_empty() { "none".to_string() } else { skipped.join(", ") }
        ),
    ));
    if ranks > 2 {
        out.push(Finding::new(
            Level::Warn,
            "ranks",
            format!(
                "{ranks} ranks: R > 2 is built for capacity and unmeasured on the fabric (no node count above two exists here), and a binary from before \
                 the N-rank runtime refuses it by name at boot (docs/CLUSTER_OPERATIONS.md)"
            ),
        ));
    }
    if c.nccl.max_nchannels != 4 {
        out.push(Finding::new(
            Level::Warn,
            "nccl_pin",
            format!("NCCL_MAX_NCHANNELS={}: the verified value is 4 (docs/NEMOTRON_TP.md 10.3, 12.4)", c.nccl.max_nchannels),
        ));
    }
    let order: Vec<&str> = c.lock_order(g).iter().map(|(_, n)| n.name.as_str()).collect();
    out.push(Finding::new(Level::Pass, "lock_order", format!("GPU locks are taken in node-list order: {}", order.join(", "))));

    // reachability
    let mut ok: Vec<(&NodeFacts, &BTreeMap<String, String>)> = Vec::new();
    for nf in facts {
        match &nf.facts {
            Ok(f) => ok.push((nf, f)),
            Err(e) => out.push(Finding::new(Level::Fail, "reach", format!("node {} (rank {}): {e}", nf.node, nf.rank))),
        }
    }
    if ok.is_empty() {
        return out;
    }
    if ok.len() == facts.len() {
        out.push(Finding::new(Level::Pass, "reach", format!("{} nodes answered", ok.len())));
    }
    for (nf, f) in &ok {
        let n = c.node(&nf.node).expect("validated");
        if let Some(h) = &n.expected_hostname {
            match f.get("hostname") {
                Some(got) if got == h => {}
                Some(got) => out.push(Finding::new(
                    Level::Fail,
                    "hostname",
                    format!("node {} answers as `{got}`, the file expects `{h}`: wrong box behind {}", n.name, n.address),
                )),
                None => out.push(Finding::new(Level::Fail, "hostname", format!("node {} reported no host name", n.name))),
            }
        }
    }
    same(&mut out, "kernel", "kernel", "kernel", &ok, Level::Warn, |s| s.to_string());
    same(&mut out, "driver", "GPU driver", "driver", &ok, Level::Warn, |s| s.to_string());

    // the binary
    let mut bin_ok = true;
    for (nf, f) in &ok {
        if f.get("binary").map(String::as_str) != Some("ok") {
            bin_ok = false;
            out.push(Finding::new(
                Level::Fail,
                "binary",
                format!("{} is missing or not executable on node {}", c.binary, nf.node),
            ));
        }
    }
    if bin_ok {
        same(&mut out, "binary", "binary hash", "binary_sha256", &ok, Level::Fail, |h| format!("sha256 {}...", short(h)));
    }

    // the pack
    let mut pack_ok = true;
    for (nf, f) in &ok {
        if f.get("pack").map(String::as_str) != Some("ok") {
            pack_ok = false;
            out.push(Finding::new(Level::Fail, "pack", format!("{} is not a directory on node {}", c.pack, nf.node)));
        }
    }
    if pack_ok {
        same(&mut out, "pack", "pack identity (config.json and the index)", "pack_identity", &ok, Level::Fail, |h| {
            format!("sha256 {}...", short(h))
        });
        same(&mut out, "pack_bytes", "pack size", "pack_bytes", &ok, Level::Fail, |b| {
            format!("{:.2} GiB", b.parse::<f64>().unwrap_or(0.0) / GIB)
        });
    }

    // the NCCL environment (one rank: nothing to agree on). What is judged is what the ranks will SEE: the resolved variables (the
    // file sourced, the pins exported, ambient cleared) and the library file the runtime will load. The env file's bytes are only a
    // hint: a file that differs (a comment, a line only an external tool reads) is a note when what resolves is the same.
    if ranks > 1 {
        let mut file_diff: Option<String> = None;
        if c.nccl.env_file.is_some() {
            let mut env_ok = true;
            for (nf, f) in &ok {
                if f.get("envfile").map(String::as_str) != Some("ok") {
                    env_ok = false;
                    out.push(Finding::new(
                        Level::Fail,
                        "nccl_env",
                        format!("NCCL env file {} is missing on node {}", c.nccl.env_file.as_deref().unwrap_or(""), nf.node),
                    ));
                }
            }
            if env_ok {
                let mut tmp = Vec::new();
                same(&mut tmp, "nccl_env", "NCCL env file", "envfile_sha256", &ok, Level::Warn, |h| {
                    format!("sha256 {}...", short(h))
                });
                match tmp.pop() {
                    Some(f) if f.level == Level::Warn => file_diff = Some(f.msg),
                    Some(f) => out.push(f),
                    None => {}
                }
            }
        }
        let resolved: Vec<BTreeMap<&str, &str>> = ok
            .iter()
            .map(|(_, f)| f.iter().filter_map(|(k, v)| k.strip_prefix("resolved.").map(|k| (k, v.as_str()))).collect())
            .collect();
        let mut resolved_same = true;
        let mut loader_diffs: Vec<String> = Vec::new();
        if !resolved.is_empty() && resolved.iter().any(|r| !r.is_empty()) {
            let mut diffs: Vec<String> = Vec::new();
            let mut keys: Vec<&str> = resolved.iter().flat_map(|r| r.keys().copied()).collect();
            keys.sort_unstable();
            keys.dedup();
            for k in keys {
                let vals: Vec<Option<&str>> = resolved.iter().map(|r| r.get(k).copied()).collect();
                if vals.iter().any(|v| *v != vals[0]) {
                    let per: Vec<String> =
                        ok.iter().zip(&vals).map(|((n, _), v)| format!("{}={}", n.node, v.unwrap_or("(unset)"))).collect();
                    // LD_LIBRARY_PATH reaches the runtime only through the library it resolves (judged by nccl_lib below)
                    if k == "LD_LIBRARY_PATH" {
                        loader_diffs.push(format!("{k}: {}", per.join(" ")));
                    } else {
                        diffs.push(format!("{k}: {}", per.join(" ")));
                    }
                }
            }
            if diffs.is_empty() {
                let mc = resolved[0].get("NCCL_MAX_NCHANNELS").copied().unwrap_or("(unset)");
                let lvl = if mc == "4" { Level::Pass } else { Level::Warn };
                out.push(Finding::new(
                    lvl,
                    "nccl_resolved",
                    format!(
                        "resolved NCCL environment identical on {} ({} variables, NCCL_MAX_NCHANNELS={mc})",
                        names(&ok),
                        resolved[0].len()
                    ),
                ));
            } else {
                resolved_same = false;
                out.push(Finding::new(
                    Level::Fail,
                    "nccl_resolved",
                    format!("the NCCL environment the ranks would see differs: {}", diffs.join("; ")),
                ));
            }
        }
        let lib_known = ok.iter().all(|(_, f)| f.contains_key("nccl_lib"));
        let lib_same = lib_known && {
            let first = ok[0].1.get("nccl_lib_sha256");
            ok.iter().all(|(_, f)| f.get("nccl_lib_sha256") == first)
        };
        if lib_known {
            same(&mut out, "nccl_lib", "NCCL library", "nccl_lib_sha256", &ok, Level::Fail, |h| {
                format!("sha256 {}...", short(h))
            });
        } else {
            out.push(Finding::new(
                Level::Warn,
                "nccl_lib",
                "libnccl.so.2 was found neither at $REXL3_NCCL_LIB, nor on LD_LIBRARY_PATH, nor in the ldconfig cache on every node: \
                 set nccl.lib (or REXL3_NCCL_LIB in serve.env), or the runtime refuses to boot a group",
            ));
        }
        // the cosmetic differences: a note when what the ranks see (variables and library bytes) is the same, else they stay
        // warnings beside the failure that explains them
        let cosmetic = resolved_same && lib_same;
        if !loader_diffs.is_empty() {
            let pinned = resolved.iter().all(|r| r.get("REXL3_NCCL_LIB").is_some_and(|v| !v.is_empty()));
            let (lvl, why) = if cosmetic {
                (
                    Level::Pass,
                    if pinned {
                        "cosmetic: REXL3_NCCL_LIB is set on every rank and the library bytes are identical (external NCCL tools such as \
                         nccl-tests still resolve libnccl from LD_LIBRARY_PATH)"
                    } else {
                        "cosmetic: every rank resolves the same libnccl bytes (external NCCL tools such as nccl-tests still resolve \
                         libnccl from LD_LIBRARY_PATH)"
                    },
                )
            } else {
                (Level::Warn, "rexl3-serve loads $REXL3_NCCL_LIB when it is set, else libnccl.so.2 from here")
            };
            out.push(Finding::new(lvl, "nccl_loader", format!("{}: {why}", loader_diffs.join("; "))));
        }
        if let Some(msg) = file_diff {
            if cosmetic {
                out.push(Finding::new(
                    Level::Pass,
                    "nccl_env",
                    format!("{msg} (cosmetic: the resolved NCCL variables and the library are identical on every rank)"),
                ));
            } else {
                out.push(Finding::new(Level::Warn, "nccl_env", format!("{msg} (see nccl_resolved / nccl_lib)")));
            }
        }
    }

    // ambient variables the launch clears
    for (nf, f) in &ok {
        let amb: Vec<String> = f.iter().filter_map(|(k, v)| k.strip_prefix("ambient.").map(|k| format!("{k}={v}"))).collect();
        if !amb.is_empty() {
            out.push(Finding::new(
                Level::Warn,
                "ambient",
                format!(
                    "node {} has {} in its login environment: cleared at launch, the cluster file is the only source",
                    nf.node,
                    amb.join(" ")
                ),
            ));
        }
    }

    // ports (rank 0's node)
    for (nf, f) in &ok {
        if nf.rank != 0 {
            continue;
        }
        for p in [g.tp_port, g.http_port] {
            match f.get(&format!("port.{p}")).map(String::as_str) {
                Some("free") => out.push(Finding::new(Level::Pass, "port", format!("{}:{p} is free", nf.node))),
                Some(_) => out.push(Finding::new(
                    Level::Fail,
                    "port",
                    format!("port {p} is already in use on node {} (rank 0)", nf.node),
                )),
                None => out.push(Finding::new(Level::Fail, "port", format!("port {p} was not probed on node {}", nf.node))),
            }
        }
    }

    // GPU idle and the lock
    for (nf, f) in &ok {
        let apps: i64 = f.get("gpu_apps").and_then(|v| v.parse().ok()).unwrap_or(-1);
        let procs: i64 = f.get("serve_procs").and_then(|v| v.parse().ok()).unwrap_or(-1);
        let lock = f.get("lock").map(String::as_str).unwrap_or("?");
        let mut bad: Vec<String> = Vec::new();
        if apps != 0 {
            bad.push(format!("{apps} compute app(s) on the GPU"));
        }
        if procs != 0 {
            bad.push(format!("{procs} rexl3-serve process(es) running"));
        }
        if lock != "free" {
            let who = f.get("lock_holders").map(|h| h.trim_end_matches([';', ' ']).to_string()).filter(|h| !h.is_empty());
            let ours = f.get("lock_ours").map(String::as_str) == Some("1");
            bad.push(format!(
                "the GPU lock is held by {}{} (the launcher takes the lock itself: it must not be started under `flock` or `nrun --gpu`)",
                who.unwrap_or_else(|| "an unknown process".into()),
                if ours { ", a rank of this group from an earlier run: `rexl3-cluster down` first" } else { "" }
            ));
        }
        if bad.is_empty() {
            out.push(Finding::new(Level::Pass, "gpu_idle", format!("node {}: GPU idle, lock free", nf.node)));
        } else {
            out.push(Finding::new(Level::Fail, "gpu_idle", format!("node {}: {}", nf.node, bad.join("; "))));
        }
        if let Some(p) = f.get("rank_running") {
            out.push(Finding::new(
                Level::Fail,
                "running",
                format!(
                    "rank {} of this group is already running on node {} (pid {p}): `rexl3-cluster down` first",
                    nf.rank, nf.node
                ),
            ));
        }
        match f.get("state_dir").map(String::as_str) {
            Some("ok") => {}
            _ => out.push(Finding::new(
                Level::Fail,
                "state_dir",
                format!("{} is not writable on node {}", c.group_state_dir(g), nf.node),
            )),
        }
    }

    // memory against the plan
    for (nf, f) in &ok {
        let Some(avail_kib) = f.get("mem_available_kib").and_then(|v| v.parse::<f64>().ok()) else { continue };
        let avail = avail_kib * 1024.0 / GIB;
        match c.min_free_for(g, nf.rank) {
            Some(need) if avail >= need => out.push(Finding::new(
                Level::Pass,
                "memory",
                format!("node {} (rank {}): {avail:.1} GiB available, the plan needs {need:.1}", nf.node, nf.rank),
            )),
            Some(need) => out.push(Finding::new(
                Level::Fail,
                "memory",
                format!(
                    "node {} (rank {}): {avail:.1} GiB available, the plan needs {need:.1} (device memory is host memory: stop what else holds it)",
                    nf.node, nf.rank
                ),
            )),
            None => {
                let lower = f.get("pack_bytes").and_then(|b| b.parse::<f64>().ok()).map(|b| b / GIB / ranks as f64);
                match lower {
                    Some(l) if avail < l => out.push(Finding::new(
                        Level::Fail,
                        "memory",
                        format!(
                            "node {} (rank {}): {avail:.1} GiB available, below the {l:.1} GiB lower bound (pack / {ranks} ranks)",
                            nf.node, nf.rank
                        ),
                    )),
                    Some(l) => out.push(Finding::new(
                        Level::Warn,
                        "memory",
                        format!(
                            "node {} (rank {}): {avail:.1} GiB available; no `memory.min_free_gib` declared, only the lower bound {l:.1} GiB \
                             (pack / {ranks}) was checked",
                            nf.node, nf.rank
                        ),
                    )),
                    None => out.push(Finding::new(Level::Warn, "memory", format!("node {}: no pack size to bound the plan", nf.node))),
                }
            }
        }
    }

    // followers reach rank 0
    for (nf, f) in &ok {
        if nf.rank == 0 {
            continue;
        }
        match f.get("ping.rank0").map(String::as_str) {
            Some("ok") => {}
            Some(_) => out.push(Finding::new(
                Level::Warn,
                "peer",
                format!(
                    "node {} cannot ping rank 0 at {} (ICMP may be filtered; the boot's connect to the rendezvous is the real test)",
                    nf.node,
                    c.rank_node(g, 0).map(|n| n.fabric_address.as_str()).unwrap_or("?")
                ),
            )),
            None => {}
        }
    }
    out
}

pub struct CheckReport {
    pub findings: Vec<Finding>,
}

impl CheckReport {
    pub fn failed(&self) -> bool {
        self.findings.iter().any(|f| f.level == Level::Fail)
    }

    pub fn count(&self, l: Level) -> usize {
        self.findings.iter().filter(|f| f.level == l).count()
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        for f in &self.findings {
            s.push_str(&f.render());
            s.push('\n');
        }
        s.push_str(&format!(
            "{}: {} passed, {} warning(s), {} failure(s)\n",
            if self.failed() { "CHECK FAILED" } else { "check ok" },
            self.count(Level::Pass),
            self.count(Level::Warn),
            self.count(Level::Fail)
        ));
        s
    }
}

/// Gather the facts of every rank's node (in parallel) and evaluate them.
pub fn run(c: &Cluster, g: &Group, t: &(dyn Transport + Sync)) -> CheckReport {
    let nodes = c.lock_order(g);
    let results: Vec<NodeFacts> = std::thread::scope(|s| {
        let hs: Vec<_> = nodes
            .iter()
            .map(|(rank, node)| {
                let script = script::check_script(c, g, *rank, node);
                s.spawn(move || {
                    let facts = match t.run(node, &script, Duration::from_secs(180)) {
                        Ok(e) if e.status == 0 => Ok(script::parse_facts(&e.stdout)),
                        Ok(e) => Err(format!(
                            "the check script exited {} (ssh status 255 is a connection failure): {}",
                            e.status,
                            e.stderr.lines().last().unwrap_or("")
                        )),
                        Err(e) => Err(e),
                    };
                    NodeFacts { rank: *rank, node: node.name.clone(), facts }
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("check thread")).collect()
    });
    CheckReport { findings: evaluate(c, g, &results) }
}

/// `--dry-run`: the scripts that `check` would run, per node.
pub fn dry_run(c: &Cluster, g: &Group) -> String {
    let mut s = String::new();
    for (rank, node) in c.lock_order(g) {
        s.push_str(&describe(&c.ssh_options, node, &format!("check, rank {rank}"), &script::check_script(c, g, rank, node)));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::tests_support::TWO_NODE;

    fn facts(rank: usize, node: &str, kv: &[(&str, &str)]) -> NodeFacts {
        NodeFacts { rank, node: node.into(), facts: Ok(kv.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()) }
    }

    fn good(rank: usize, node: &str) -> Vec<(&'static str, String)> {
        let mut v = vec![
            ("hostname", if node == "N1" { "h1".to_string() } else { "h4".to_string() }),
            ("kernel", "6.17".into()),
            ("driver", "580".into()),
            ("mem_available_kib", (100.0 * 1024.0 * 1024.0).to_string()),
            ("binary", "ok".into()),
            ("binary_sha256", "aaaaaaaaaaaaaaaaaaaaaaaa".into()),
            ("pack", "ok".into()),
            ("pack_bytes", (60.0 * GIB).to_string()),
            ("pack_identity", "bbbbbbbbbbbbbbbb".into()),
            ("envfile", "ok".into()),
            ("envfile_sha256", "cccccccccccccccc".into()),
            ("resolved.NCCL_MAX_NCHANNELS", "4".into()),
            ("resolved.NCCL_IB_HCA", "=a:1,b:1".into()),
            ("nccl_lib", "/x/libnccl.so.2.30.7".into()),
            ("nccl_lib_sha256", "dddddddddddddddd".into()),
            ("gpu_apps", "0".into()),
            ("serve_procs", "0".into()),
            ("lock", "free".into()),
            ("state_dir", "ok".into()),
        ];
        if rank == 0 {
            v.push(("port.29600", "free".into()));
            v.push(("port.8941", "free".into()));
        } else {
            v.push(("ping.rank0", "ok".into()));
        }
        let _ = node;
        v
    }

    fn nf(rank: usize, node: &str, patch: &[(&str, &str)]) -> NodeFacts {
        let mut base: BTreeMap<String, String> = good(rank, node).into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        for (k, v) in patch {
            if v.is_empty() {
                base.remove(*k);
            } else {
                base.insert(k.to_string(), v.to_string());
            }
        }
        NodeFacts { rank, node: node.into(), facts: Ok(base) }
    }

    fn cluster() -> Cluster {
        Cluster::parse(&TWO_NODE.replace(r#""address": "192.0.2.4", "#, r#""address": "192.0.2.4", "expected_hostname": "h4", "#))
            .unwrap()
    }

    fn fails(f: &[Finding]) -> Vec<String> {
        f.iter().filter(|f| f.level == Level::Fail).map(|f| format!("{}: {}", f.id, f.msg)).collect()
    }

    #[test]
    fn a_healthy_pair_passes() {
        let c = cluster();
        let g = &c.groups[0];
        let f = evaluate(&c, g, &[nf(0, "N1", &[]), nf(1, "N4", &[])]);
        assert!(fails(&f).is_empty(), "{:#?}", fails(&f));
        assert!(f.iter().any(|x| x.id == "binary" && x.level == Level::Pass));
        assert!(f.iter().any(|x| x.id == "nccl_resolved" && x.level == Level::Pass));
        // the memory is a warning until a plan is declared
        assert!(f.iter().any(|x| x.id == "memory" && x.level == Level::Warn));
        let _ = facts;
    }

    #[test]
    fn a_different_binary_pack_or_env_fails_by_node() {
        let c = cluster();
        let g = &c.groups[0];
        let f = evaluate(
            &c,
            g,
            &[
                nf(0, "N1", &[]),
                nf(
                    1,
                    "N4",
                    &[
                        ("binary_sha256", "eeeeeeeeeeeeeeeeeeeeee"),
                        ("pack_identity", "ffff"),
                        ("resolved.NCCL_MAX_NCHANNELS", "2"),
                    ],
                ),
            ],
        );
        let e = fails(&f);
        assert!(
            e.iter().any(|m| m.starts_with("binary")
                && m.contains("N1 has sha256 aaaaaaaaaaaa")
                && m.contains("N4 has sha256 eeeeeeeeeeee")),
            "{e:?}"
        );
        assert!(e.iter().any(|m| m.starts_with("pack") && m.contains("pack identity")), "{e:?}");
        assert!(e.iter().any(|m| m.starts_with("nccl_resolved") && m.contains("NCCL_MAX_NCHANNELS: N1=4 N4=2")), "{e:?}");
    }

    #[test]
    fn an_env_file_or_loader_path_that_differs_is_a_warning_when_the_nccl_variables_agree() {
        // found live (2026-10-11): node 4's crs812-nccl.env lacked the LD_LIBRARY_PATH line node 1's had; rexl3-serve finds libnccl itself
        let c = cluster();
        let g = &c.groups[0];
        let f = evaluate(
            &c,
            g,
            &[
                nf(0, "N1", &[("resolved.LD_LIBRARY_PATH", "/srv/nccl/lib")]),
                nf(1, "N4", &[("envfile_sha256", "9999999999999999")]),
            ],
        );
        assert!(fails(&f).is_empty(), "{:#?}", fails(&f));
        // the variables and the library bytes agree: both differences are cosmetic notes, not warnings (2026-10-15)
        assert!(f.iter().any(|x| x.id == "nccl_loader"
            && x.level == Level::Pass
            && x.msg.contains("N1=/srv/nccl/lib N4=(unset)")
            && x.msg.contains("cosmetic")));
        assert!(f.iter().any(|x| x.id == "nccl_env" && x.level == Level::Pass && x.msg.contains("cosmetic")), "{f:#?}");
        assert!(!f.iter().any(|x| x.level == Level::Warn && x.id.starts_with("nccl")), "{f:#?}");
        // ...and a different NCCL variable is still a failure, with the file difference left a warning beside it
        let f = evaluate(
            &c,
            g,
            &[nf(0, "N1", &[]), nf(1, "N4", &[("resolved.NCCL_IB_HCA", "=other:1"), ("envfile_sha256", "9999999999999999")])],
        );
        assert!(fails(&f).iter().any(|m| m.contains("NCCL_IB_HCA")));
        assert!(f.iter().any(|x| x.id == "nccl_env" && x.level == Level::Warn));
    }

    #[test]
    fn a_loader_path_that_resolves_another_library_fails_on_the_library_and_keeps_the_loader_difference_a_warning() {
        // node 4 without the LD_LIBRARY_PATH line and without REXL3_NCCL_LIB: the loader finds a system libnccl of other bytes
        let c = cluster();
        let g = &c.groups[0];
        let f = evaluate(
            &c,
            g,
            &[
                nf(0, "N1", &[("resolved.LD_LIBRARY_PATH", "/srv/nccl/lib")]),
                nf(1, "N4", &[("nccl_lib_sha256", "eeeeeeeeeeeeeeee"), ("nccl_lib", "/usr/lib/libnccl.so.2.28.3")]),
            ],
        );
        assert!(fails(&f).iter().any(|m| m.starts_with("nccl_lib")), "{:#?}", fails(&f));
        assert!(f.iter().any(|x| x.id == "nccl_loader" && x.level == Level::Warn));
    }

    #[test]
    fn a_pinned_library_makes_the_loader_difference_a_note_that_says_so() {
        let c = cluster();
        let g = &c.groups[0];
        let pin = ("resolved.REXL3_NCCL_LIB", "/opt/nccl/libnccl.so.2");
        let f = evaluate(&c, g, &[nf(0, "N1", &[pin, ("resolved.LD_LIBRARY_PATH", "/srv/nccl/lib")]), nf(1, "N4", &[pin])]);
        assert!(fails(&f).is_empty(), "{:#?}", fails(&f));
        assert!(f.iter().any(|x| x.id == "nccl_loader" && x.level == Level::Pass && x.msg.contains("REXL3_NCCL_LIB is set")));
    }

    #[test]
    fn busy_gpu_held_lock_ports_and_memory_fail() {
        let mut c = cluster();
        c.min_free_gib = Some(crate::schema::MemReq::Uniform(90.0));
        let g = c.groups[0].clone();
        let low = (40.0 * 1024.0 * 1024.0).to_string();
        let f = evaluate(
            &c,
            &g,
            &[
                nf(0, "N1", &[("port.29600", "busy"), ("lock", "held"), ("serve_procs", "1")]),
                nf(1, "N4", &[("mem_available_kib", low.as_str()), ("gpu_apps", "1")]),
            ],
        );
        let e = fails(&f).join("\n");
        assert!(e.contains("port 29600 is already in use on node N1"), "{e}");
        assert!(e.contains("the GPU lock is held") && e.contains("1 rexl3-serve process"), "{e}");
        assert!(e.contains("node N4 (rank 1): 40.0 GiB available, the plan needs 90.0"), "{e}");
        assert!(e.contains("1 compute app(s)"), "{e}");
    }

    #[test]
    fn the_wrong_host_and_an_unreachable_node_fail() {
        let c = cluster();
        let g = &c.groups[0];
        let f = evaluate(
            &c,
            g,
            &[
                nf(0, "N1", &[("hostname", "someone-else")]),
                NodeFacts { rank: 1, node: "N4".into(), facts: Err("ssh: timed out".into()) },
            ],
        );
        let e = fails(&f).join("\n");
        assert!(e.contains("node N1 answers as `someone-else`, the file expects `h1`"), "{e}");
        assert!(e.contains("node N4 (rank 1): ssh: timed out"), "{e}");
    }

    #[test]
    fn ambient_variables_and_a_missing_plan_are_warnings_and_r_above_two_is_named() {
        let c = cluster();
        let g = &c.groups[0];
        let f = evaluate(&c, g, &[nf(0, "N1", &[("ambient.REXL3_NEMO_GRAPHS", "off")]), nf(1, "N4", &[])]);
        assert!(f.iter().any(|x| x.id == "ambient" && x.level == Level::Warn && x.msg.contains("REXL3_NEMO_GRAPHS=off")));
        let mut g4 = g.clone();
        g4.ranks = vec!["N1".into(), "N4".into(), "N5".into()];
        let f = evaluate(&c, &g4, &[nf(0, "N1", &[]), nf(1, "N4", &[])]);
        assert!(f.iter().any(|x| x.id == "ranks" && x.level == Level::Warn && x.msg.contains("unmeasured")));
    }
}
