//! The cluster file: parse, validate, and derive the command line and environment of rank r of R.
//!
//! JSON, not toml (no new crate dependency; `serde_json::Value` is walked by hand so that every error names its path). The whole file is
//! validated before anything is run: unknown keys, a disabled node in a rank map, a duplicate or missing rank, a missing port, a
//! plan-affecting flag or variable that differs between ranks and an absent `NCCL_MAX_NCHANNELS` are all refused by name, every one of
//! them in one pass so that an operator fixes the file once.

use serde_json::{Map, Value};
use std::collections::BTreeMap;

pub const SCHEMA_VERSION: u64 = 1;

/// Flags that change what a rank plans or computes (docs/NEMOTRON_TP.md 5.6, 12.4). They come from `serve.flags` once; a node or leader
/// override of one of them with another value, or one that `serve.flags` does not pin, is a refusal.
pub const PLAN_FLAGS: &[&str] = &[
    "--model",
    "--draft",
    "--backend",
    "--slots",
    "--max-model-len",
    "--kv-tokens",
    "--kv-headroom-tokens",
    "--prefill-chunk",
    "--window-rows",
    "--spec",
    "--no-spec-tree",
    "--prefix-cache",
    "--prefix-cache-mib",
    "--no-vision",
    "--no-video",
    "--max-image-tokens",
    "--max-video-frames",
    "--max-video-tokens",
    "--kv-dtype",
    "--allow-unvalidated-pack",
    "--verify-pack",
];

/// Flags the launcher derives from the cluster file; naming one in `serve.flags` is a refusal (two sources of one value).
pub const DERIVED_FLAGS: &[&str] = &[
    "--model",
    "--host",
    "--port",
    "--tp-ranks",
    "--tp-rank",
    "--tp-peer",
    "--tp-connect-timeout",
    "--tp-ready-timeout",
    "--tp-peer-timeout",
    "--shutdown-grace",
];

/// Flags that take no value (so the next token is not theirs).
pub const BOOL_FLAGS: &[&str] = &[
    "--no-vision",
    "--no-video",
    "--no-drafts",
    "--no-spec-tree",
    "--allow-unvalidated-pack",
    "--verify-pack",
    "--video-evs-strict",
    "--print-fingerprints",
    "--print-crc32",
];

/// Environment variables every rank must see alike: a node or leader override with another value is a refusal. `NCCL_` is not part of
/// the runtime's boot gate (docs/NEMOTRON_TP.md 12.3 note iii), so this is the only place it is enforced. `REXL3_PREFIX_` (the prefix
/// cache's eviction and checkpoint policy, docs/LONG_PREFILL_POLICY.md): ranks in lockstep must make the same cache decisions.
pub const GROUP_ENV_PREFIXES: &[&str] = &["REXL3_NEMO_", "REXL3_MTP", "REXL3_TP_", "REXL3_KV_", "REXL3_PREFIX_", "NCCL_"];

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub name: String,
    pub number: Option<u64>,
    /// How the launcher reaches the node (ssh, and the HTTP port of a leader).
    pub address: String,
    /// What the other ranks use to reach it (the rendezvous address of rank 0); defaults to `address`.
    pub fabric_address: String,
    pub user: String,
    pub gpu_lock: String,
    pub expected_hostname: Option<String>,
    /// Run through a local `bash` instead of ssh (the launcher runs on this node).
    pub local: bool,
    pub disabled: bool,
    pub reason: Option<String>,
    /// Per-node environment (only variables that are not group-wide).
    pub env: BTreeMap<String, String>,
    /// Per-node flags (only flags that are not plan-affecting).
    pub flags: Vec<String>,
}

/// The variable `nccl.lib` is exported as (`crates/cuda/src/nccl.rs` `LIBRARY_ENV`).
pub const NCCL_LIB_ENV: &str = "REXL3_NCCL_LIB";

#[derive(Debug, Clone, PartialEq)]
pub struct Nccl {
    pub env_file: Option<String>,
    pub max_nchannels: u64,
    pub socket_ifname: Option<String>,
    pub hca: Vec<String>,
    pub extra: BTreeMap<String, String>,
    /// The libnccl file every rank loads (`nccl.lib`, exported as `REXL3_NCCL_LIB`; `$HOME/` expands on the node). Typed so the
    /// pre-flight and the boot gate judge one library, not whatever each node's loader path finds.
    pub lib: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Serve {
    pub flags: Vec<String>,
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TpTimeouts {
    pub connect_timeout_s: u64,
    pub ready_timeout_s: u64,
    pub peer_timeout_s: u64,
    pub shutdown_grace_s: u64,
}

/// The least `MemAvailable` a rank needs before it boots (the plan's number: weights per rank plus headroom, from the boot log of a
/// previous run or the capacity table of docs/CLUSTER_OPERATIONS.md): one number for every rank or one per rank.
#[derive(Debug, Clone, PartialEq)]
pub enum MemReq {
    Uniform(f64),
    PerRank(Vec<f64>),
}

impl MemReq {
    pub fn for_rank(&self, rank: usize) -> f64 {
        match self {
            MemReq::Uniform(v) => *v,
            MemReq::PerRank(v) => v.get(rank).copied().unwrap_or(0.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    pub name: String,
    /// Node name of rank r at index r.
    pub ranks: Vec<String>,
    pub tp_port: u16,
    pub http_port: u16,
    pub http_host: String,
    pub leader_flags: Vec<String>,
    pub min_free_gib: Option<MemReq>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SupervisorCfg {
    pub backoff_base_s: u64,
    pub backoff_cap_s: u64,
    /// A run that stayed ready this long resets the failure counters.
    pub stable_after_s: u64,
    /// More failed attempts than this inside `window_s` and the supervisor stops (the systemd `StartLimitBurst`).
    pub max_restarts: u32,
    pub window_s: u64,
    /// After one rank exits, the others get this long to exit by themselves (the runtime's own backstop is 20 s) so that their exit
    /// codes are the evidence the policy reads; whatever is left is then stopped.
    pub settle_s: u64,
    pub poll_s: u64,
}

impl Default for SupervisorCfg {
    fn default() -> Self {
        SupervisorCfg {
            backoff_base_s: 10,
            backoff_cap_s: 300,
            stable_after_s: 300,
            max_restarts: 5,
            window_s: 600,
            settle_s: 25,
            poll_s: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cluster {
    /// Attested adapter placement only: node -> (binary, pack). Empty preserves legacy behavior.
    pub runtime_paths: BTreeMap<String, (String, String)>,
    pub name: String,
    pub ssh_options: Vec<String>,
    pub binary: String,
    pub binary_max_ranks: Option<u64>,
    pub state_dir: String,
    pub nodes: Vec<Node>,
    pub nccl: Nccl,
    pub pack: String,
    pub serve: Serve,
    pub tp: TpTimeouts,
    pub min_free_gib: Option<MemReq>,
    /// How long a rank waits for its node's GPU lock before it gives up (the launcher's own session budget).
    pub lock_wait_s: u64,
    /// A rank's wall-clock budget: after this long its server is sent SIGTERM (rank 1 gets 60 s more so that it outlives rank 0 and
    /// sees Shutdown, not a signal). `None`: no limit.
    pub max_runtime_s: Option<u64>,
    pub groups: Vec<Group>,
    pub supervisor: SupervisorCfg,
}

/// One token of a flag list.
#[derive(Debug, Clone, PartialEq)]
pub struct Flag {
    pub name: String,
    pub value: Option<String>,
}

/// Split `["--slots", "2", "--no-vision"]` into flags with their values. A value that is not preceded by a flag is an error.
pub fn parse_flags(tokens: &[String]) -> Result<Vec<Flag>, String> {
    let mut out: Vec<Flag> = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let t = &tokens[i];
        if !t.starts_with("--") {
            return Err(format!("`{t}` is not a flag and follows no flag that takes a value"));
        }
        let boolean = BOOL_FLAGS.contains(&t.as_str());
        if !boolean && i + 1 < tokens.len() && !tokens[i + 1].starts_with("--") {
            out.push(Flag { name: t.clone(), value: Some(tokens[i + 1].clone()) });
            i += 2;
        } else {
            out.push(Flag { name: t.clone(), value: None });
            i += 1;
        }
    }
    Ok(out)
}

fn is_group_env(k: &str) -> bool {
    GROUP_ENV_PREFIXES.iter().any(|p| k.starts_with(p))
}

fn valid_env_name(k: &str) -> bool {
    !k.is_empty() && !k.starts_with(|c: char| c.is_ascii_digit()) && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

// ---------------------------------------------------------------------------------------------------------------------------- reading

struct R<'a> {
    errs: &'a mut Vec<String>,
}

impl R<'_> {
    fn err(&mut self, path: &str, msg: impl AsRef<str>) {
        self.errs.push(format!("{path}: {}", msg.as_ref()));
    }

    fn object<'v>(&mut self, v: &'v Value, path: &str) -> Option<&'v Map<String, Value>> {
        match v.as_object() {
            Some(m) => Some(m),
            None => {
                self.err(path, "must be an object");
                None
            }
        }
    }

    /// Unknown keys are refused (a typo in `NCCL_MAX_NCHANNELS` must not pass as "absent"); keys starting with `_` are comments.
    fn keys(&mut self, m: &Map<String, Value>, path: &str, allowed: &[&str]) {
        for k in m.keys() {
            if k.starts_with('_') || allowed.contains(&k.as_str()) {
                continue;
            }
            self.err(path, format!("unknown key `{k}` (allowed: {})", allowed.join(", ")));
        }
    }

    fn string(&mut self, m: &Map<String, Value>, key: &str, path: &str, required: bool) -> Option<String> {
        match m.get(key) {
            None | Some(Value::Null) => {
                if required {
                    self.err(path, format!("missing `{key}`"));
                }
                None
            }
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            Some(_) => {
                self.err(&format!("{path}.{key}"), "must be a non-empty string");
                None
            }
        }
    }

    fn uint(&mut self, m: &Map<String, Value>, key: &str, path: &str, required: bool) -> Option<u64> {
        match m.get(key) {
            None | Some(Value::Null) => {
                if required {
                    self.err(path, format!("missing `{key}`"));
                }
                None
            }
            Some(v) => match v.as_u64() {
                Some(n) => Some(n),
                None => {
                    self.err(&format!("{path}.{key}"), "must be a non-negative integer");
                    None
                }
            },
        }
    }

    fn boolean(&mut self, m: &Map<String, Value>, key: &str, path: &str) -> Option<bool> {
        match m.get(key) {
            None | Some(Value::Null) => None,
            Some(Value::Bool(b)) => Some(*b),
            Some(_) => {
                self.err(&format!("{path}.{key}"), "must be true or false");
                None
            }
        }
    }

    fn strings(&mut self, m: &Map<String, Value>, key: &str, path: &str) -> Vec<String> {
        match m.get(key) {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(a)) => {
                let mut out = Vec::new();
                for (i, v) in a.iter().enumerate() {
                    match v {
                        Value::String(s) => out.push(s.clone()),
                        // numbers are accepted as flag values: ["--slots", 2]
                        Value::Number(n) => out.push(n.to_string()),
                        _ => self.err(&format!("{path}.{key}[{i}]"), "must be a string"),
                    }
                }
                out
            }
            Some(_) => {
                self.err(&format!("{path}.{key}"), "must be an array of strings");
                Vec::new()
            }
        }
    }

    fn string_map(&mut self, m: &Map<String, Value>, key: &str, path: &str) -> BTreeMap<String, String> {
        let mut out = BTreeMap::new();
        match m.get(key) {
            None | Some(Value::Null) => {}
            Some(Value::Object(o)) => {
                for (k, v) in o {
                    if k.starts_with('_') {
                        continue;
                    }
                    if !valid_env_name(k) {
                        self.err(&format!("{path}.{key}"), format!("`{k}` is not a valid variable name"));
                        continue;
                    }
                    match v {
                        Value::String(s) => {
                            out.insert(k.clone(), s.clone());
                        }
                        Value::Number(n) => {
                            out.insert(k.clone(), n.to_string());
                        }
                        Value::Bool(b) => {
                            out.insert(k.clone(), if *b { "1".into() } else { "0".into() });
                        }
                        _ => self.err(&format!("{path}.{key}.{k}"), "must be a string or a number"),
                    }
                }
            }
            Some(_) => self.err(&format!("{path}.{key}"), "must be an object of strings"),
        }
        out
    }

    fn memreq(&mut self, m: &Map<String, Value>, key: &str, path: &str) -> Option<MemReq> {
        match m.get(key) {
            None | Some(Value::Null) => None,
            Some(Value::Number(n)) => match n.as_f64() {
                Some(f) if f > 0.0 => Some(MemReq::Uniform(f)),
                _ => {
                    self.err(&format!("{path}.{key}"), "must be a positive number of GiB");
                    None
                }
            },
            Some(Value::Array(a)) => {
                let mut v = Vec::new();
                for (i, x) in a.iter().enumerate() {
                    match x.as_f64() {
                        Some(f) if f > 0.0 => v.push(f),
                        _ => self.err(&format!("{path}.{key}[{i}]"), "must be a positive number of GiB"),
                    }
                }
                Some(MemReq::PerRank(v))
            }
            Some(_) => {
                self.err(&format!("{path}.{key}"), "must be a number of GiB or an array with one per rank");
                None
            }
        }
    }
}

fn port(r: &mut R<'_>, m: &Map<String, Value>, key: &str, path: &str) -> Option<u16> {
    let n = r.uint(m, key, path, true)?;
    if n == 0 || n > 65535 {
        r.err(&format!("{path}.{key}"), format!("{n} is not a port (1 to 65535)"));
        return None;
    }
    Some(n as u16)
}

impl Cluster {
    /// Parse and validate. Every problem found is returned (not the first only).
    pub fn parse(text: &str) -> Result<Cluster, Vec<String>> {
        let root: Value = serde_json::from_str(text).map_err(|e| vec![format!("cluster file: not valid JSON: {e}")])?;
        let mut errs = Vec::new();
        let c = Cluster::from_value(&root, &mut errs);
        match c {
            Some(c) if errs.is_empty() => Ok(c),
            _ => Err(errs),
        }
    }

    fn from_value(root: &Value, errs: &mut Vec<String>) -> Option<Cluster> {
        let mut r = R { errs };
        let m = r.object(root, "cluster file")?;
        r.keys(
            m,
            "cluster file",
            &[
                "schema_version",
                "runtime_paths",
                "name",
                "user",
                "ssh",
                "binary",
                "binary_max_ranks",
                "state_dir",
                "nodes",
                "nccl",
                "pack",
                "serve",
                "tp",
                "memory",
                "lock_wait_s",
                "max_runtime_s",
                "groups",
                "supervisor",
            ],
        );
        match m.get("schema_version").and_then(|v| v.as_u64()) {
            Some(SCHEMA_VERSION) => {}
            Some(n) => r.err("schema_version", format!("{n} is not supported (this tool reads {SCHEMA_VERSION})")),
            None => r.err("cluster file", "missing `schema_version`"),
        }
        let name = r.string(m, "name", "cluster file", true).unwrap_or_default();
        if !name.is_empty() && !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            r.err("name", format!("`{name}` may hold only letters, digits, `-` and `_` (it names the state directory)"));
        }
        let default_user = r.string(m, "user", "cluster file", false);
        let binary = r.string(m, "binary", "cluster file", true).unwrap_or_default();
        let binary_max_ranks = r.uint(m, "binary_max_ranks", "cluster file", false);
        let state_dir = r.string(m, "state_dir", "cluster file", false).unwrap_or_else(|| format!("/tmp/rexl3-cluster-{name}"));
        let lock_wait_s = r.uint(m, "lock_wait_s", "cluster file", false).unwrap_or(300);
        let max_runtime_s = r.uint(m, "max_runtime_s", "cluster file", false);

        // ssh
        let mut ssh_options: Vec<String> = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        if let Some(v) = m.get("ssh") {
            if let Some(o) = r.object(v, "ssh") {
                r.keys(o, "ssh", &["options"]);
                if o.contains_key("options") {
                    ssh_options = r.strings(o, "options", "ssh");
                }
            }
        }

        // nodes
        let mut nodes: Vec<Node> = Vec::new();
        match m.get("nodes") {
            Some(Value::Array(a)) if !a.is_empty() => {
                for (i, v) in a.iter().enumerate() {
                    let path = format!("nodes[{i}]");
                    let Some(o) = r.object(v, &path) else { continue };
                    r.keys(
                        o,
                        &path,
                        &[
                            "name",
                            "number",
                            "address",
                            "fabric_address",
                            "user",
                            "gpu_lock",
                            "expected_hostname",
                            "local",
                            "disabled",
                            "reason",
                            "env",
                            "flags",
                        ],
                    );
                    let Some(nname) = r.string(o, "name", &path, true) else { continue };
                    let path = format!("nodes[{i}] ({nname})");
                    let address = r.string(o, "address", &path, true).unwrap_or_default();
                    let fabric = r.string(o, "fabric_address", &path, false).unwrap_or_else(|| address.clone());
                    let user = r.string(o, "user", &path, false).or_else(|| default_user.clone());
                    if user.is_none() {
                        r.err(&path, "missing `user` (and no top-level `user`)");
                    }
                    for (k, val) in [("address", &address), ("fabric_address", &fabric)] {
                        // never an ssh option (`-oProxyCommand=...`), never shell syntax
                        if val.starts_with('-') || val.chars().any(|c| c.is_whitespace() || "'\"`$;&|<>\\()".contains(c)) {
                            r.err(&format!("{path}.{k}"), format!("`{val}` is not a host name or an address"));
                        }
                    }
                    if let Some(u) = &user {
                        if u.starts_with('-') || !u.chars().all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c)) {
                            r.err(&format!("{path}.user"), format!("`{u}` is not a user name"));
                        }
                    }
                    nodes.push(Node {
                        name: nname,
                        number: r.uint(o, "number", &path, false),
                        address,
                        fabric_address: fabric,
                        user: user.unwrap_or_default(),
                        gpu_lock: r.string(o, "gpu_lock", &path, false).unwrap_or_else(|| "/tmp/rexl3-gpu.lock".into()),
                        expected_hostname: r.string(o, "expected_hostname", &path, false),
                        local: r.boolean(o, "local", &path).unwrap_or(false),
                        disabled: r.boolean(o, "disabled", &path).unwrap_or(false),
                        reason: r.string(o, "reason", &path, false),
                        env: r.string_map(o, "env", &path),
                        flags: r.strings(o, "flags", &path),
                    });
                }
            }
            Some(_) => r.err("nodes", "must be a non-empty array"),
            None => r.err("cluster file", "missing `nodes`"),
        }

        // nccl
        let mut nccl =
            Nccl { env_file: None, max_nchannels: 0, socket_ifname: None, hca: Vec::new(), extra: BTreeMap::new(), lib: None };
        match m.get("nccl") {
            Some(v) => {
                if let Some(o) = r.object(v, "nccl") {
                    r.keys(o, "nccl", &["env_file", "max_nchannels", "socket_ifname", "hca", "env", "lib"]);
                    nccl.env_file = r.string(o, "env_file", "nccl", false);
                    nccl.lib = r.string(o, "lib", "nccl", false);
                    nccl.socket_ifname = r.string(o, "socket_ifname", "nccl", false);
                    nccl.hca = r.strings(o, "hca", "nccl");
                    nccl.extra = r.string_map(o, "env", "nccl");
                    for k in nccl.extra.keys() {
                        if !k.starts_with("NCCL_") {
                            r.err(
                                "nccl.env",
                                format!("`{k}` is not an NCCL_ variable (shared REXL3_ variables go in serve.env)"),
                            );
                        }
                        if matches!(k.as_str(), "NCCL_MAX_NCHANNELS" | "NCCL_SOCKET_IFNAME" | "NCCL_IB_HCA") {
                            r.err("nccl.env", format!("`{k}` has its own key (max_nchannels, socket_ifname, hca)"));
                        }
                    }
                    match o.get("max_nchannels") {
                        None | Some(Value::Null) => r.err(
                            "nccl",
                            "NCCL_MAX_NCHANNELS is required: set `max_nchannels` (4: NCCL's own tuning is 3 to 10 times slower on this fabric, docs/NEMOTRON_TP.md 10.3)",
                        ),
                        Some(v) => match v.as_u64() {
                            Some(n) if n > 0 => nccl.max_nchannels = n,
                            _ => r.err("nccl.max_nchannels", "must be a positive integer (4)"),
                        },
                    }
                }
            }
            None => r.err(
                "cluster file",
                "missing `nccl`: NCCL_MAX_NCHANNELS=4 is required (set nccl.max_nchannels, docs/NEMOTRON_TP.md 10.3)",
            ),
        }

        // pack
        let pack = match m.get("pack") {
            Some(Value::String(s)) if !s.is_empty() => s.clone(),
            Some(Value::Object(o)) => {
                r.keys(o, "pack", &["path"]);
                r.string(o, "path", "pack", true).unwrap_or_default()
            }
            _ => {
                r.err("cluster file", "missing `pack` (a path, the same on every node)");
                String::new()
            }
        };

        // serve
        let mut serve = Serve::default();
        if let Some(v) = m.get("serve") {
            if let Some(o) = r.object(v, "serve") {
                r.keys(o, "serve", &["flags", "env"]);
                serve.flags = r.strings(o, "flags", "serve");
                serve.env = r.string_map(o, "env", "serve");
            }
        }

        // tp
        let mut tp = TpTimeouts { connect_timeout_s: 300, ready_timeout_s: 300, peer_timeout_s: 15, shutdown_grace_s: 20 };
        if let Some(v) = m.get("tp") {
            if let Some(o) = r.object(v, "tp") {
                r.keys(o, "tp", &["connect_timeout_s", "ready_timeout_s", "peer_timeout_s", "shutdown_grace_s"]);
                tp.connect_timeout_s = r.uint(o, "connect_timeout_s", "tp", false).unwrap_or(tp.connect_timeout_s);
                tp.ready_timeout_s = r.uint(o, "ready_timeout_s", "tp", false).unwrap_or(tp.ready_timeout_s);
                tp.peer_timeout_s = r.uint(o, "peer_timeout_s", "tp", false).unwrap_or(tp.peer_timeout_s);
                tp.shutdown_grace_s = r.uint(o, "shutdown_grace_s", "tp", false).unwrap_or(tp.shutdown_grace_s);
            }
        }

        // memory
        let mut min_free_gib = None;
        if let Some(v) = m.get("memory") {
            if let Some(o) = r.object(v, "memory") {
                r.keys(o, "memory", &["min_free_gib"]);
                min_free_gib = r.memreq(o, "min_free_gib", "memory");
            }
        }

        // supervisor
        let mut sup = SupervisorCfg::default();
        if let Some(v) = m.get("supervisor") {
            if let Some(o) = r.object(v, "supervisor") {
                r.keys(
                    o,
                    "supervisor",
                    &["backoff_base_s", "backoff_cap_s", "stable_after_s", "max_restarts", "window_s", "settle_s", "poll_s"],
                );
                sup.backoff_base_s = r.uint(o, "backoff_base_s", "supervisor", false).unwrap_or(sup.backoff_base_s);
                sup.backoff_cap_s = r.uint(o, "backoff_cap_s", "supervisor", false).unwrap_or(sup.backoff_cap_s);
                sup.stable_after_s = r.uint(o, "stable_after_s", "supervisor", false).unwrap_or(sup.stable_after_s);
                sup.max_restarts = r.uint(o, "max_restarts", "supervisor", false).unwrap_or(sup.max_restarts as u64) as u32;
                sup.window_s = r.uint(o, "window_s", "supervisor", false).unwrap_or(sup.window_s);
                sup.settle_s = r.uint(o, "settle_s", "supervisor", false).unwrap_or(sup.settle_s);
                sup.poll_s = r.uint(o, "poll_s", "supervisor", false).unwrap_or(sup.poll_s).max(1);
                if sup.backoff_cap_s < sup.backoff_base_s {
                    r.err("supervisor", "backoff_cap_s is below backoff_base_s");
                }
            }
        }

        // groups
        let mut groups: Vec<Group> = Vec::new();
        match m.get("groups") {
            Some(Value::Array(a)) if !a.is_empty() => {
                for (i, v) in a.iter().enumerate() {
                    let path = format!("groups[{i}]");
                    let Some(o) = r.object(v, &path) else { continue };
                    r.keys(o, &path, &["name", "ranks", "tp_port", "http_port", "http_host", "leader_flags", "min_free_gib"]);
                    let Some(gname) = r.string(o, "name", &path, true) else { continue };
                    if !gname.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
                        r.err(&path, format!("group name `{gname}` may hold only letters, digits, `-` and `_`"));
                    }
                    let path = format!("groups[{i}] ({gname})");
                    let ranks = read_rank_map(&mut r, o, &path);
                    let tp_port = port(&mut r, o, "tp_port", &path);
                    let http_port = port(&mut r, o, "http_port", &path);
                    groups.push(Group {
                        name: gname,
                        ranks,
                        tp_port: tp_port.unwrap_or(0),
                        http_port: http_port.unwrap_or(0),
                        http_host: r.string(o, "http_host", &path, false).unwrap_or_else(|| "0.0.0.0".into()),
                        leader_flags: r.strings(o, "leader_flags", &path),
                        min_free_gib: r.memreq(o, "min_free_gib", &path),
                    });
                }
            }
            Some(_) => r.err("groups", "must be a non-empty array"),
            None => r.err("cluster file", "missing `groups`"),
        }

        let mut runtime_paths = BTreeMap::new();
        if let Some(v) = m.get("runtime_paths") {
            if let Some(paths) = r.object(v, "runtime_paths") {
                for (node, entry) in paths {
                    let Some(fields) = r.object(entry, "runtime_paths.node") else { continue };
                    r.keys(fields, "runtime_paths.node", &["binary", "pack"]);
                    let bin = r.string(fields, "binary", "runtime_paths.node", true).unwrap_or_default();
                    let pack = r.string(fields, "pack", "runtime_paths.node", true).unwrap_or_default();
                    if !nodes.iter().any(|n| n.name == *node)
                        || [&bin, &pack].iter().any(|p| {
                            !p.starts_with('/')
                                || p.split('/').skip(1).any(|s| s.is_empty() || s == "." || s == "..")
                                || p.chars().any(|c| c.is_control())
                        })
                    {
                        r.err("runtime_paths", "unknown node or noncanonical absolute path");
                    }
                    runtime_paths.insert(node.clone(), (bin, pack));
                }
                if runtime_paths.len() != nodes.len() {
                    r.err("runtime_paths", "every node must have a placement");
                }
            }
        }
        let cluster = Cluster {
            runtime_paths,
            name,
            ssh_options,
            binary,
            binary_max_ranks,
            state_dir,
            nodes,
            nccl,
            pack,
            serve,
            tp,
            min_free_gib,
            lock_wait_s,
            max_runtime_s,
            groups,
            supervisor: sup,
        };
        cluster.validate(&mut r);
        Some(cluster)
    }

    fn validate(&self, r: &mut R<'_>) {
        // node names and numbers
        let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
        for (i, n) in self.nodes.iter().enumerate() {
            if let Some(j) = seen.insert(n.name.as_str(), i) {
                r.err("nodes", format!("node name `{}` appears twice (nodes[{j}] and nodes[{i}])", n.name));
            }
        }
        // lock order: nodes are taken in list order, so the list must be in ascending node number (docs: node 1 then node 4)
        let numbered: Vec<(&Node, u64)> = self.nodes.iter().filter_map(|n| n.number.map(|k| (n, k))).collect();
        for w in numbered.windows(2) {
            if w[1].1 <= w[0].1 {
                r.err(
                    "nodes",
                    format!(
                        "GPU locks are taken in list order and the list must ascend by `number`: {} (number {}) comes after {} (number {})",
                        w[1].0.name, w[1].1, w[0].0.name, w[0].1
                    ),
                );
            }
        }

        // serve flags
        let serve_flags = match parse_flags(&self.serve.flags) {
            Ok(f) => f,
            Err(e) => {
                r.err("serve.flags", e);
                Vec::new()
            }
        };
        let mut pinned: BTreeMap<String, Option<String>> = BTreeMap::new();
        for f in &serve_flags {
            if DERIVED_FLAGS.contains(&f.name.as_str()) {
                r.err(
                    "serve.flags",
                    format!(
                        "`{}` is derived from the cluster file (pack, groups[].tp_port, http_port, tp.*): remove it here",
                        f.name
                    ),
                );
            }
            if let Some(prev) = pinned.insert(f.name.clone(), f.value.clone()) {
                if prev != f.value {
                    r.err("serve.flags", format!("`{}` is given twice with different values", f.name));
                }
            }
        }
        // serve.env: REXL3_ variables only; NCCL_ has its own section
        for k in self.serve.env.keys() {
            if k.starts_with("NCCL_") {
                r.err("serve.env", format!("`{k}` is an NCCL variable: use the `nccl` section"));
            }
        }
        // nccl.lib is the one source of the library: a second one in serve.env or a node's env would be ambiguous
        if self.nccl.lib.is_some() {
            if self.serve.env.contains_key(NCCL_LIB_ENV) {
                r.err("serve.env", format!("`{NCCL_LIB_ENV}` is set by nccl.lib: give it once"));
            }
            for n in self.nodes.iter().filter(|n| n.env.contains_key(NCCL_LIB_ENV)) {
                r.err(
                    &format!("nodes ({}).env", n.name),
                    format!("`{NCCL_LIB_ENV}` is set by nccl.lib for every rank: give it once"),
                );
            }
        }
        let group_env = self.group_env();

        // per-node overrides must not move a plan-affecting flag or a group-wide variable
        for n in &self.nodes {
            let path = format!("nodes ({})", n.name);
            self.check_overrides(r, &path, &n.flags, &n.env, &pinned, &group_env);
        }
        for g in &self.groups {
            let path = format!("groups ({}).leader_flags", g.name);
            self.check_overrides(r, &path, &g.leader_flags, &BTreeMap::new(), &pinned, &group_env);
        }

        // groups
        let mut gnames: BTreeMap<&str, usize> = BTreeMap::new();
        let mut node_group: BTreeMap<&str, &str> = BTreeMap::new();
        for g in &self.groups {
            if gnames.insert(g.name.as_str(), 1).is_some() {
                r.err("groups", format!("group name `{}` appears twice", g.name));
            }
            let path = format!("group {}", g.name);
            if g.ranks.is_empty() {
                r.err(&path, "has no ranks");
            }
            for (rank, nname) in g.ranks.iter().enumerate() {
                match self.node(nname) {
                    None => r.err(&path, format!("rank {rank} maps to `{nname}`, which is not in `nodes`")),
                    Some(n) if n.disabled => r.err(
                        &path,
                        format!(
                            "rank {rank} maps to node {nname}, which is disabled{}: the launcher never uses a disabled node",
                            n.reason.as_ref().map(|s| format!(" ({s})")).unwrap_or_default()
                        ),
                    ),
                    Some(_) => {}
                }
                if let Some(other) = node_group.insert(nname.as_str(), g.name.as_str()) {
                    if other == g.name {
                        r.err(
                            &path,
                            format!("node {nname} hosts more than one rank of the group (one rank per node: one GPU per GB10)"),
                        );
                    } else {
                        r.err(
                            &path,
                            format!("node {nname} is also in group {other}: one big model per node (the unified pool is shared)"),
                        );
                    }
                }
            }
            if g.tp_port != 0 && g.tp_port == g.http_port {
                r.err(&path, format!("tp_port and http_port are both {}", g.tp_port));
            }
            if let Some(MemReq::PerRank(v)) = &g.min_free_gib {
                if v.len() != g.ranks.len() {
                    r.err(&path, format!("min_free_gib has {} entries for {} ranks", v.len(), g.ranks.len()));
                }
            }
            if let Some(max) = self.binary_max_ranks {
                if g.ranks.len() as u64 > max {
                    r.err(
                        &path,
                        format!(
                            "{} ranks, but `binary_max_ranks` says the binary serves at most {max} (the runtime refuses more by name at boot)",
                            g.ranks.len()
                        ),
                    );
                }
            }
        }
    }

    /// The group-wide environment (shared REXL3_ and NCCL_ variables) as every rank gets it.
    pub fn group_env(&self) -> BTreeMap<String, String> {
        let mut e = BTreeMap::new();
        e.insert("NCCL_MAX_NCHANNELS".to_string(), self.nccl.max_nchannels.to_string());
        if let Some(s) = &self.nccl.socket_ifname {
            e.insert("NCCL_SOCKET_IFNAME".into(), s.clone());
        }
        if !self.nccl.hca.is_empty() {
            e.insert("NCCL_IB_HCA".into(), format!("={}", self.nccl.hca.join(",")));
        }
        for (k, v) in &self.nccl.extra {
            e.insert(k.clone(), v.clone());
        }
        for (k, v) in &self.serve.env {
            e.insert(k.clone(), v.clone());
        }
        e
    }

    fn check_overrides(
        &self,
        r: &mut R<'_>,
        path: &str,
        flags: &[String],
        env: &BTreeMap<String, String>,
        pinned: &BTreeMap<String, Option<String>>,
        group_env: &BTreeMap<String, String>,
    ) {
        match parse_flags(flags) {
            Err(e) => r.err(path, e),
            Ok(fs) => {
                for f in fs {
                    if DERIVED_FLAGS.contains(&f.name.as_str()) {
                        r.err(path, format!("`{}` is derived from the cluster file: remove it here", f.name));
                        continue;
                    }
                    if !PLAN_FLAGS.contains(&f.name.as_str()) {
                        continue;
                    }
                    match pinned.get(&f.name) {
                        None => r.err(
                            path,
                            format!(
                                "sets plan-affecting flag `{}` that serve.flags does not pin: it would differ between ranks (set it once in serve.flags)",
                                f.name
                            ),
                        ),
                        Some(v) if *v != f.value => r.err(
                            path,
                            format!(
                                "flag `{}` is {} here but {} in serve.flags: plan-affecting flags must be identical on every rank",
                                f.name,
                                f.value.as_deref().unwrap_or("(present)"),
                                v.as_deref().unwrap_or("(present)")
                            ),
                        ),
                        Some(_) => {}
                    }
                }
            }
        }
        for (k, v) in env {
            if !is_group_env(k) {
                continue;
            }
            match group_env.get(k) {
                None => r.err(
                    path,
                    format!("sets `{k}`, a variable every rank must see alike, that serve.env / nccl does not pin (set it once in serve.env)"),
                ),
                Some(g) if g != v => r.err(
                    path,
                    format!("`{k}` is `{v}` here but `{g}` in serve.env / nccl: group-wide variables must be identical on every rank"),
                ),
                Some(_) => {}
            }
        }
    }

    pub fn node(&self, name: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.name == name)
    }

    pub fn group(&self, name: &str) -> Option<&Group> {
        self.groups.iter().find(|g| g.name == name)
    }

    /// The group an operator means: the named one, or the only one.
    pub fn select_group(&self, name: Option<&str>) -> Result<&Group, String> {
        match (name, self.groups.len()) {
            (Some(n), _) => self.group(n).ok_or_else(|| {
                format!("no group `{n}` (groups: {})", self.groups.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(", "))
            }),
            (None, 1) => Ok(&self.groups[0]),
            (None, _) => Err(format!(
                "the cluster file has {} groups: name one with --group ({})",
                self.groups.len(),
                self.groups.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(", ")
            )),
        }
    }

    /// The nodes of a group with their ranks, in node-list order: the order the GPU locks are taken and the ranks are started.
    pub fn lock_order<'a>(&'a self, g: &'a Group) -> Vec<(usize, &'a Node)> {
        let mut v: Vec<(usize, &Node)> = Vec::new();
        for n in &self.nodes {
            if let Some(rank) = g.ranks.iter().position(|x| *x == n.name) {
                v.push((rank, n));
            }
        }
        v
    }

    pub fn rank_node<'a>(&'a self, g: &Group, rank: usize) -> Option<&'a Node> {
        g.ranks.get(rank).and_then(|n| self.node(n))
    }

    pub fn node_binary(&self, node: &Node) -> &str {
        self.runtime_paths.get(&node.name).map(|p| p.0.as_str()).unwrap_or(&self.binary)
    }

    pub fn rank_binary(&self, g: &Group, rank: usize) -> &str {
        self.rank_node(g, rank).map(|n| self.node_binary(n)).unwrap_or(&self.binary)
    }

    pub fn rank_pack(&self, g: &Group, rank: usize) -> &str {
        self.rank_node(g, rank).and_then(|n| self.runtime_paths.get(&n.name)).map(|p| p.1.as_str()).unwrap_or(&self.pack)
    }

    pub fn group_state_dir(&self, g: &Group) -> String {
        format!("{}/{}", self.state_dir.trim_end_matches('/'), g.name)
    }

    /// The rendezvous address every rank is given: rank 0's fabric address and the group's `tp_port`.
    pub fn tp_peer(&self, g: &Group) -> String {
        let a = self.rank_node(g, 0).map(|n| n.fabric_address.clone()).unwrap_or_default();
        format!("{a}:{}", g.tp_port)
    }

    /// The command line (after the binary) of rank `rank`: the shared flags, the node's own, rank 0's front end, the `--tp-*` flags.
    pub fn rank_args(&self, g: &Group, rank: usize) -> Vec<String> {
        let ranks = g.ranks.len();
        let mut a: Vec<String> = vec!["--model".into(), self.rank_pack(g, rank).to_string()];
        a.extend(self.serve.flags.iter().cloned());
        if let Some(n) = self.rank_node(g, rank) {
            a.extend(n.flags.iter().cloned());
        }
        if rank == 0 {
            a.extend(g.leader_flags.iter().cloned());
            a.extend(["--host".into(), g.http_host.clone(), "--port".into(), g.http_port.to_string()]);
        }
        if ranks > 1 {
            a.extend([
                "--tp-ranks".into(),
                ranks.to_string(),
                "--tp-rank".into(),
                rank.to_string(),
                "--tp-peer".into(),
                self.tp_peer(g),
                "--tp-connect-timeout".into(),
                self.tp.connect_timeout_s.to_string(),
                "--tp-ready-timeout".into(),
                self.tp.ready_timeout_s.to_string(),
                "--tp-peer-timeout".into(),
                self.tp.peer_timeout_s.to_string(),
            ]);
        }
        a.extend(["--shutdown-grace".into(), self.tp.shutdown_grace_s.to_string()]);
        a
    }

    /// The variables exported after the env file is sourced (a group of one rank gets no NCCL pins).
    pub fn rank_env(&self, g: &Group, rank: usize) -> Vec<(String, String)> {
        let mut e: BTreeMap<String, String> = BTreeMap::new();
        if g.ranks.len() > 1 {
            for (k, v) in self.group_env() {
                if k.starts_with("NCCL_") {
                    e.insert(k, v);
                }
            }
            if let Some(l) = &self.nccl.lib {
                e.insert(NCCL_LIB_ENV.to_string(), l.clone());
            }
        }
        for (k, v) in &self.serve.env {
            e.insert(k.clone(), v.clone());
        }
        if let Some(n) = self.rank_node(g, rank) {
            for (k, v) in &n.env {
                e.insert(k.clone(), v.clone());
            }
        }
        e.into_iter().collect()
    }

    /// What the memory check compares `MemAvailable` with (GiB), for rank `rank` of group `g`.
    pub fn min_free_for(&self, g: &Group, rank: usize) -> Option<f64> {
        g.min_free_gib.as_ref().or(self.min_free_gib.as_ref()).map(|m| m.for_rank(rank))
    }

    /// Seconds a rank may run before the launcher's own budget stops it: rank 0 first, followers 60 s later.
    pub fn runtime_budget(&self, rank: usize, override_s: Option<u64>) -> Option<u64> {
        override_s.or(self.max_runtime_s).map(|s| if rank == 0 { s } else { s + 60 })
    }
}

fn read_rank_map(r: &mut R<'_>, o: &Map<String, Value>, path: &str) -> Vec<String> {
    let Some(v) = o.get("ranks") else {
        r.err(path, "missing `ranks` (the rank map: rank -> node)");
        return Vec::new();
    };
    let Some(a) = v.as_array() else {
        r.err(&format!("{path}.ranks"), "must be an array: [\"N1\", \"N4\"] or [{\"rank\": 0, \"node\": \"N1\"}, ...]");
        return Vec::new();
    };
    // entries are node names (rank = index) or {rank, node}
    let mut slots: BTreeMap<u64, String> = BTreeMap::new();
    let mut explicit = false;
    for (i, e) in a.iter().enumerate() {
        match e {
            Value::String(s) => {
                if slots.insert(i as u64, s.clone()).is_some() {
                    r.err(path, format!("rank {i} is assigned twice"));
                }
            }
            Value::Object(m) => {
                explicit = true;
                let p = format!("{path}.ranks[{i}]");
                r.keys(m, &p, &["rank", "node"]);
                let rank = r.uint(m, "rank", &p, true);
                let node = r.string(m, "node", &p, true);
                if let (Some(rank), Some(node)) = (rank, node) {
                    if let std::collections::btree_map::Entry::Vacant(e) = slots.entry(rank) {
                        e.insert(node);
                    } else {
                        r.err(path, format!("duplicate rank {rank}: assigned to {} and to {node}", slots[&rank]));
                    }
                }
            }
            _ => r.err(&format!("{path}.ranks[{i}]"), "must be a node name or {\"rank\": N, \"node\": NAME}"),
        }
    }
    let _ = explicit;
    // the ranks must be exactly 0..R
    let n = slots.len() as u64;
    let mut out = Vec::new();
    for rank in 0..n {
        match slots.get(&rank) {
            Some(s) => out.push(s.clone()),
            None => {
                r.err(
                    path,
                    format!("rank {rank} is missing: the ranks of a group are exactly 0..{} with no gap", n.saturating_sub(1)),
                );
                out.push(String::new());
            }
        }
    }
    for k in slots.keys() {
        if *k >= n {
            r.err(path, format!("rank {k} is out of range for a group of {n} ranks (ranks are 0..{})", n.saturating_sub(1)));
        }
    }
    out
}

#[cfg(test)]
pub mod tests_support {
    pub const TWO_NODE: &str = r#"{
      "schema_version": 1,
      "name": "t2",
      "user": "u",
      "binary": "/opt/rexl3/rexl3-serve",
      "nodes": [
        {"name": "N1", "number": 1, "address": "192.0.2.1", "expected_hostname": "h1"},
        {"name": "N2", "number": 2, "address": "192.0.2.2", "disabled": true, "reason": "taken out of service for maintenance"},
        {"name": "N4", "number": 4, "address": "192.0.2.4", "fabric_address": "198.51.100.4"}
      ],
      "nccl": {"env_file": "$HOME/nccl.env", "max_nchannels": 4, "socket_ifname": "=eth0", "hca": ["mlx5_0:1", "mlx5_1:1"]},
      "pack": "/models/p",
      "serve": {"flags": ["--allow-unvalidated-pack", "--backend", "cuda", "--slots", "2", "--no-vision"], "env": {"REXL3_NEMO_GRAPHS": "on"}},
      "groups": [{"name": "g0", "ranks": ["N1", "N4"], "tp_port": 29600, "http_port": 8941}]
    }"#;
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::tests_support::TWO_NODE;

    #[test]
    fn attested_paths_are_per_node_and_not_serving_overrides() {
        let text = TWO_NODE.replace(
            r#""schema_version": 1,"#,
            r#""schema_version": 1,
            "runtime_paths": {
                "N1":{"binary":"/one/bin/rexl3-serve","pack":"/one/model"},
                "N2":{"binary":"/two/bin/rexl3-serve","pack":"/two/model"},
                "N4":{"binary":"/four/bin/rexl3-serve","pack":"/four/model"}},"#,
        );
        let c = Cluster::parse(&text).unwrap();
        let g = &c.groups[0];
        assert_eq!(c.rank_args(g, 1)[1], "/four/model");
        assert_eq!(c.rank_binary(g, 1), "/four/bin/rexl3-serve");
        assert!(Cluster::parse(&text.replace("/four/model", "/four/../model")).is_err());
        assert!(Cluster::parse(&text.replace(r#""pack":"/four/model""#, r#""pack":"/four/model","flags":[]"#)).is_err());
    }

    fn errs(text: &str) -> Vec<String> {
        Cluster::parse(text).expect_err("must be refused")
    }

    fn has(errs: &[String], needle: &str) -> bool {
        errs.iter().any(|e| e.contains(needle))
    }

    #[test]
    fn nccl_lib_is_exported_to_every_rank_of_a_group_and_given_once() {
        let t = TWO_NODE.replace(r#""max_nchannels": 4, "#, r#""max_nchannels": 4, "lib": "$HOME/nccl/lib/libnccl.so.2", "#);
        let c = Cluster::parse(&t).unwrap();
        let g = c.select_group(None).unwrap();
        for rank in 0..2 {
            assert!(c.rank_env(g, rank).contains(&(NCCL_LIB_ENV.into(), "$HOME/nccl/lib/libnccl.so.2".into())));
        }
        // the export expands $HOME on the node (the other values stay literal)
        assert_eq!(crate::script::env_value(NCCL_LIB_ENV, "$HOME/nccl/lib/libnccl.so.2"), "\"$HOME\"/nccl/lib/libnccl.so.2");
        assert_eq!(crate::script::env_value("REXL3_X", "$HOME/x"), "'$HOME/x'");
        // given twice: refused
        let t2 =
            t.replace(r#""env": {"REXL3_NEMO_GRAPHS": "on"}"#, r#""env": {"REXL3_NEMO_GRAPHS": "on", "REXL3_NCCL_LIB": "/x"}"#);
        assert_ne!(t2, t);
        let e = Cluster::parse(&t2).expect_err("twice");
        assert!(has(&e, "set by nccl.lib"), "{e:?}");
    }

    #[test]
    fn parses_the_two_node_example_and_derives_the_rank_commands() {
        let c = Cluster::parse(TWO_NODE).unwrap();
        let g = c.select_group(None).unwrap();
        assert_eq!(g.ranks, ["N1", "N4"]);
        assert_eq!(c.tp_peer(g), "192.0.2.1:29600");
        let a0 = c.rank_args(g, 0).join(" ");
        assert_eq!(
            a0,
            "--model /models/p --allow-unvalidated-pack --backend cuda --slots 2 --no-vision --host 0.0.0.0 --port 8941 \
             --tp-ranks 2 --tp-rank 0 --tp-peer 192.0.2.1:29600 --tp-connect-timeout 300 --tp-ready-timeout 300 --tp-peer-timeout 15 \
             --shutdown-grace 20"
        );
        let a1 = c.rank_args(g, 1).join(" ");
        assert!(a1.contains("--tp-rank 1 --tp-peer 192.0.2.1:29600"), "{a1}");
        assert!(!a1.contains("--port"), "a follower serves no HTTP: {a1}");
        let e = c.rank_env(g, 1);
        assert!(e.contains(&("NCCL_MAX_NCHANNELS".into(), "4".into())));
        assert!(e.contains(&("NCCL_IB_HCA".into(), "=mlx5_0:1,mlx5_1:1".into())));
        assert!(e.contains(&("REXL3_NEMO_GRAPHS".into(), "on".into())));
        // N1 and N4 take their locks in list order, N2 is skipped
        let order: Vec<_> = c.lock_order(g).into_iter().map(|(r, n)| (r, n.name.clone())).collect();
        assert_eq!(order, [(0, "N1".to_string()), (1, "N4".to_string())]);
    }

    #[test]
    fn a_disabled_node_in_the_rank_map_is_refused_by_name() {
        let t = TWO_NODE.replace(r#""ranks": ["N1", "N4"]"#, r#""ranks": ["N1", "N2"]"#);
        let e = errs(&t);
        assert!(has(&e, "node N2, which is disabled"), "{e:?}");
        assert!(has(&e, "taken out of service for maintenance"), "the reason is quoted: {e:?}");
    }

    #[test]
    fn no_address_is_refused_by_a_built_in_list_only_disabled_nodes_are() {
        // the built-in owner deny list of the gn100 nodes (2026-10-09) was lifted with the four-node cluster (2026-10-16): a node is
        // kept out of a group by `disabled: true` in the file, never by its address
        let t = TWO_NODE.replace(
            r#"{"name": "N4", "number": 4, "address": "192.0.2.4", "fabric_address": "198.51.100.4"}"#,
            r#"{"name": "N3", "number": 3, "address": "192.0.2.3", "expected_hostname": "host-c"}"#,
        );
        let t = t.replace(r#"["N1", "N4"]"#, r#"["N1", "N3"]"#);
        let c = Cluster::parse(&t).unwrap_or_else(|e| panic!("{e:?}"));
        assert_eq!(c.groups[0].ranks.len(), 2);
    }

    #[test]
    fn duplicate_missing_and_out_of_range_ranks_are_refused() {
        let t =
            TWO_NODE.replace(r#""ranks": ["N1", "N4"]"#, r#""ranks": [{"rank": 0, "node": "N1"}, {"rank": 0, "node": "N4"}]"#);
        assert!(has(&errs(&t), "duplicate rank 0"));
        let t =
            TWO_NODE.replace(r#""ranks": ["N1", "N4"]"#, r#""ranks": [{"rank": 0, "node": "N1"}, {"rank": 2, "node": "N4"}]"#);
        let e = errs(&t);
        assert!(has(&e, "rank 1 is missing") && has(&e, "rank 2 is out of range"), "{e:?}");
        // the same node twice: one rank per node
        let t = TWO_NODE.replace(r#""ranks": ["N1", "N4"]"#, r#""ranks": ["N1", "N1"]"#);
        assert!(has(&errs(&t), "hosts more than one rank"));
        let t = TWO_NODE.replace(r#""ranks": ["N1", "N4"]"#, r#""ranks": ["N1", "N9"]"#);
        assert!(has(&errs(&t), "`N9`, which is not in `nodes`"));
    }

    #[test]
    fn missing_ports_and_nchannels_are_refused() {
        let t = TWO_NODE.replace(r#", "tp_port": 29600"#, "");
        assert!(has(&errs(&t), "missing `tp_port`"));
        let t = TWO_NODE.replace(r#", "http_port": 8941"#, "");
        assert!(has(&errs(&t), "missing `http_port`"));
        let t = TWO_NODE.replace(r#""tp_port": 29600, "http_port": 8941"#, r#""tp_port": 8941, "http_port": 8941"#);
        assert!(has(&errs(&t), "tp_port and http_port are both 8941"));
        let t = TWO_NODE.replace(r#""max_nchannels": 4, "#, "");
        assert!(has(&errs(&t), "NCCL_MAX_NCHANNELS is required"));
        let t = TWO_NODE.replace(r#""tp_port": 29600"#, r#""tp_port": 70000"#);
        assert!(has(&errs(&t), "70000 is not a port"));
    }

    #[test]
    fn a_plan_affecting_flag_that_differs_between_ranks_is_refused() {
        // a node override with another value
        let t = TWO_NODE.replace(
            r#""address": "192.0.2.4", "fabric_address": "198.51.100.4""#,
            r#""address": "192.0.2.4", "fabric_address": "198.51.100.4", "flags": ["--slots", "4"]"#,
        );
        let e = errs(&t);
        assert!(has(&e, "flag `--slots` is 4 here but 2 in serve.flags"), "{e:?}");
        // a plan-affecting flag that serve.flags does not pin
        let t = TWO_NODE.replace(
            r#""address": "192.0.2.4", "fabric_address": "198.51.100.4""#,
            r#""address": "192.0.2.4", "fabric_address": "198.51.100.4", "flags": ["--kv-tokens", "100"]"#,
        );
        assert!(has(&errs(&t), "plan-affecting flag `--kv-tokens` that serve.flags does not pin"));
        // the leader may not move one either
        let t = TWO_NODE.replace(r#""http_port": 8941"#, r#""http_port": 8941, "leader_flags": ["--max-model-len", "9"]"#);
        assert!(has(&errs(&t), "leader_flags"));
        // an NCCL / plan variable on one node
        let t = TWO_NODE.replace(
            r#""address": "192.0.2.4", "fabric_address": "198.51.100.4""#,
            r#""address": "192.0.2.4", "fabric_address": "198.51.100.4", "env": {"REXL3_NEMO_GRAPHS": "off", "NCCL_PROTO": "Simple"}"#,
        );
        let e = errs(&t);
        assert!(has(&e, "`REXL3_NEMO_GRAPHS` is `off` here but `on` in serve.env"), "{e:?}");
        assert!(has(&e, "`NCCL_PROTO`"), "{e:?}");
        // the prefix cache's policy on one node only: the ranks would make different cache decisions in lockstep
        let t = TWO_NODE.replace(
            r#""address": "192.0.2.4", "fabric_address": "198.51.100.4""#,
            r#""address": "192.0.2.4", "fabric_address": "198.51.100.4", "env": {"REXL3_PREFIX_EVICTION": "size", "REXL3_PREFIX_CHECKPOINT": "65536"}"#,
        );
        let e = errs(&t);
        assert!(has(&e, "sets `REXL3_PREFIX_EVICTION`, a variable every rank must see alike"), "{e:?}");
        assert!(has(&e, "sets `REXL3_PREFIX_CHECKPOINT`, a variable every rank must see alike"), "{e:?}");
        // the same policy group-wide is fine, and a node may repeat it
        let t = TWO_NODE
            .replace(
                r#""env": {"REXL3_NEMO_GRAPHS": "on"}"#,
                r#""env": {"REXL3_NEMO_GRAPHS": "on", "REXL3_PREFIX_EVICTION": "size"}"#,
            )
            .replace(
                r#""address": "192.0.2.4", "fabric_address": "198.51.100.4""#,
                r#""address": "192.0.2.4", "fabric_address": "198.51.100.4", "env": {"REXL3_PREFIX_EVICTION": "size"}"#,
            );
        assert!(Cluster::parse(&t).is_ok());
        // a harmless per-node variable passes
        let t = TWO_NODE.replace(
            r#""address": "192.0.2.4", "fabric_address": "198.51.100.4""#,
            r#""address": "192.0.2.4", "fabric_address": "198.51.100.4", "env": {"REXL3_LOAD_THREADS": "4"}"#,
        );
        assert!(Cluster::parse(&t).is_ok());
    }

    #[test]
    fn derived_flags_and_unknown_keys_are_refused() {
        let t = TWO_NODE.replace(r#""--slots", "2""#, r#""--slots", "2", "--tp-rank", "1""#);
        assert!(has(&errs(&t), "`--tp-rank` is derived"));
        let t = TWO_NODE.replace(r#""binary": "#, r#""binry": "/x", "binary": "#);
        assert!(has(&errs(&t), "unknown key `binry`"));
        let t = TWO_NODE.replace(r#""max_nchannels": 4"#, r#""max_nchannels": 4, "max_nchanels": 2"#);
        assert!(has(&errs(&t), "unknown key `max_nchanels`"));
        assert!(has(&errs("{ not json"), "not valid JSON"));
    }

    #[test]
    fn an_address_or_user_that_could_be_an_ssh_option_is_refused() {
        let t = TWO_NODE.replace(r#""address": "192.0.2.1""#, r#""address": "-oProxyCommand=x""#);
        assert!(has(&errs(&t), "is not a host name or an address"));
        let t = TWO_NODE.replace(r#""user": "u""#, r#""user": "-oFoo=bar""#);
        assert!(has(&errs(&t), "is not a user name"));
        let t = TWO_NODE.replace(r#""address": "192.0.2.1""#, r#""address": "192.0.2.1; rm -rf /""#);
        assert!(has(&errs(&t), "is not a host name or an address"));
    }

    #[test]
    fn lock_order_must_ascend_by_node_number() {
        let t = TWO_NODE.replace(r#""number": 4"#, r#""number": 0"#);
        assert!(has(&errs(&t), "must ascend by `number`"));
    }

    #[test]
    fn flag_parsing_pairs_values_and_knows_booleans() {
        let f = parse_flags(&["--slots".into(), "2".into(), "--no-vision".into(), "--spec".into(), "auto".into()]).unwrap();
        assert_eq!(f.len(), 3);
        assert_eq!(f[1], Flag { name: "--no-vision".into(), value: None });
        assert!(parse_flags(&["2".into()]).is_err());
    }

    #[test]
    fn ranks_above_the_binary_limit_are_refused_when_declared() {
        let t = TWO_NODE.replace(r#""name": "t2","#, r#""name": "t2", "binary_max_ranks": 1,"#);
        assert!(has(&errs(&t), "binary_max_ranks"));
    }
}
