//! The two-rank group over REXL3's own RoCE all-gather (`gb_cuda::roce`, `docs/ROCE_GATHER.md`), behind [`TpComm`]: the exchanges
//! of the policy's sizes go by RDMA writes signalled from REXL3's kernels, every other one by NCCL (which the group keeps: the
//! fallback, the `aux` communicator, and the reference the boot can compare against).
//!
//! The choice of path is a pure function of the message size and the [`RocePolicy`], and both ranks run the same policy (its
//! [`RocePolicy::fingerprint`] belongs in the boot gate's build string, next to `collective_fingerprint`), so every rank issues the
//! same form of every collective. Exactness is the transport's: a swap is a byte copy and the sum is `add.rn.f32` without
//! flush-to-zero in rank order, bit-identical to NCCL's two-rank ring over the engine's disjoint-support buffers
//! (`crates/cuda/tests/roce_pair.rs`).

use super::tp::{TpComm, TpTopology};
use super::tp_comm::NcclTp;
use gb_cuda::roce::{RoceComm, RoceConfig};
use gb_cuda::{CudaError, DevBuf, Stream};
use std::sync::Arc;

/// `REXL3_TP_TRANSPORT`: `nccl`, `roce` or `auto`. Both ranks must agree (the fingerprint). The default is the family's: NCCL for
/// Nemotron, `auto` for GLM ([`Transport`], [`transport_from`]). `roce` is strict (a RoCE step that fails fails the boot); `auto`
/// brings RoCE up only if every rank can and runs on NCCL otherwise (a named event); `nccl` is the opt-out.
pub const TRANSPORT_ENV: &str = "REXL3_TP_TRANSPORT";
/// `REXL3_TP_ROCE_FAULT=<step>@<rank>` (a gate hook, off unless set; rank-local on purpose, so not in any fingerprint): the named step
/// of the RoCE bring-up (`probe`, `region`, `connect`, `proxy`) fails on that rank only, as a real failure of it would. `probe` acts
/// under `auto` (the strict path has no probe).
pub const FAULT_ENV: &str = "REXL3_TP_ROCE_FAULT";
/// `REXL3_ROCE_MAX_BYTES`: the largest message (bytes a rank) the RoCE path carries; NCCL above. Default: the largest site.
pub const MAX_ENV: &str = "REXL3_ROCE_MAX_BYTES";
/// `REXL3_ROCE_MIN_BYTES`: the smallest; NCCL below (default 1: every size, the transport measured faster at all of them).
pub const MIN_ENV: &str = "REXL3_ROCE_MIN_BYTES";
/// `REXL3_ROCE_SUM_MAX_BYTES`: the largest in-place f32 sum (bytes a rank, `allreduce_sum_f32` / `allgather_sum_f32`) the RoCE path
/// carries; NCCL's all-reduce above it. Default [`SUM_MAX_DEFAULT`]. An in-place sum's source is its destination, so from `bulk_min`
/// the bulk kernel keeps its sequential stage-then-drain form for it (`roce_gather.cu`).
pub const SUM_MAX_ENV: &str = "REXL3_ROCE_SUM_MAX_BYTES";
/// 1 MiB: MEASURED against NCCL's best all-reduce setting (LL / Simple x 1 / 2 / 4 channels, the kit's, untuned), eager no-event
/// means, N3+N2 (N1+N4 alike): RoCE's in-place sum 11.0 / 16.4 / 19.5 / 39.8 / 65.0 us at 16 KiB / 64 KiB / 128 KiB / 512 KiB / 1 MiB
/// against 23.8 / 31.3 / 38.6 / 60.9 / 80.3; at 16 MiB a tie (772 against 785; N1+N4 772 against 770) and at 64 MiB NCCL 2-7 % ahead
/// (3,227 against 3,128; N1+N4 3,358 against 3,143). The largest size measured faster on both pairs (4-8 MiB was not measured).
pub const SUM_MAX_DEFAULT: usize = 1 << 20;
/// `REXL3_ROCE_ZERO_COPY` (lane T): `1` (default) lets a caller that offers it write its message straight into the site's half and
/// read the peer's in place ([`RoceTp::zero_copy_swap`], the GLM P2 part swap); `0` keeps every swap on the standalone kernel's
/// staging and copy-out. The wire is the same either way (the same chunks of the same half), so the ranks need not agree; it is not
/// in the fingerprint.
pub const ZERO_COPY_ENV: &str = "REXL3_ROCE_ZERO_COPY";

/// Which messages take the RoCE path: bytes a rank in `[min_bytes, max_bytes]`, and in-place sums up to `sum_max_bytes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RocePolicy {
    pub min_bytes: usize,
    pub max_bytes: usize,
    pub sum_max_bytes: usize,
}

impl RocePolicy {
    /// From a variable lookup; `max_default` is the largest site the transport was built with.
    pub fn from_vars(get: impl Fn(&str) -> Option<String>, max_default: usize) -> Result<RocePolicy, String> {
        let num = |k: &str, d: usize| -> Result<usize, String> {
            match get(k).filter(|v| !v.trim().is_empty()) {
                None => Ok(d),
                Some(v) => v.trim().parse().map_err(|_| format!("{k} = {v:?} is not a byte count")),
            }
        };
        let p = RocePolicy {
            min_bytes: num(MIN_ENV, 1)?.max(1),
            max_bytes: num(MAX_ENV, max_default)?,
            sum_max_bytes: num(SUM_MAX_ENV, SUM_MAX_DEFAULT)?,
        };
        if p.max_bytes > max_default {
            return Err(format!("{MAX_ENV} = {} is above the largest site ({max_default})", p.max_bytes));
        }
        Ok(p)
    }

    pub fn uses(&self, bytes: usize) -> bool {
        bytes >= self.min_bytes && bytes <= self.max_bytes
    }

    pub fn fingerprint(&self, cfg: &RoceConfig, sites: &[usize]) -> String {
        let s: Vec<String> = sites.iter().map(|c| c.to_string()).collect();
        format!(
            "{},min={},max={},summax={},sites={}",
            cfg.fingerprint(),
            self.min_bytes,
            self.max_bytes,
            self.sum_max_bytes,
            s.join("/")
        )
    }
}

/// The transport's part of the boot gate's collective fingerprint (`tp_comm::collective_fingerprint`), from this process's
/// environment: `None` with NCCL alone (the default: the fingerprint text is unchanged), else the size->path rule both ranks must share
/// (`roce:v1,tc=..,stripe=..,bulk=..,chunk=..,blocks=..` from [`RoceConfig`], then `min=..,max=..` from [`RocePolicy`]'s variables,
/// `max=sites` when unset: the largest site, the same build on both ranks), or `invalid(..)` for a setting that does not parse.
pub fn process_fingerprint() -> Option<String> {
    fingerprint_from_vars(|k| std::env::var(k).ok())
}

/// [`process_fingerprint`] from a variable lookup.
pub fn fingerprint_from_vars(get: impl Fn(&str) -> Option<String>) -> Option<String> {
    let var = |k: &str| get(k).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    match wanted_from(var) {
        Ok(false) => None,
        Err(e) => Some(format!("invalid({e})")),
        Ok(true) => Some(policy_fingerprint_from_vars(&var)),
    }
}

/// The RoCE size->path rule both ranks must share, as [`fingerprint_from_vars`] prints it under `roce`: what a family that runs
/// `auto` puts in its own collective fact (the ranks may end up on RoCE, so a different rule on one of them must be refused).
pub fn policy_fingerprint_from_vars(var: &impl Fn(&str) -> Option<String>) -> String {
    match RoceConfig::from_vars(var) {
        Err(e) => format!("invalid({e})"),
        Ok(cfg) => match RocePolicy::from_vars(var, usize::MAX) {
            Err(e) => format!("invalid({e})"),
            Ok(p) => format!(
                "{},min={},max={},summax={}",
                cfg.fingerprint(),
                p.min_bytes,
                var(MAX_ENV).map_or("sites".to_string(), |_| p.max_bytes.to_string()),
                p.sum_max_bytes
            ),
        },
    }
}

/// [`policy_fingerprint_from_vars`] of this process's environment (empty values count as unset, as in [`process_fingerprint`]).
pub fn policy_fingerprint() -> String {
    let var = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    policy_fingerprint_from_vars(&var)
}

/// Two ranks swap `mine` (the same length on both) over NCCL's `swap_bytes`, synchronously.
fn nccl_swap(n: &NcclTp, st: &Stream, mine: &[u8]) -> Result<Vec<u8>, String> {
    let send = DevBuf::from_slice(mine).map_err(|e| format!("RoCE bootstrap upload: {e}"))?;
    let mut recv = DevBuf::zeroed(mine.len()).map_err(|e| format!("RoCE bootstrap buffer: {e}"))?;
    n.swap_bytes(&send, 0, mine.len(), &mut recv, 0, mine.len(), st)
        .map_err(|e| format!("RoCE bootstrap swap over NCCL: {e}"))?;
    st.sync().map_err(|e| format!("RoCE bootstrap swap over NCCL: {e}"))?;
    recv.to_vec::<u8>().map_err(|e| format!("RoCE bootstrap download: {e}"))
}

// ---------------------------------------------------------------------------------------------------------------------------------
// The bring-up's verdicts, and the probe that `auto` adds in front of them

/// The side channel of the bring-up: this rank's bytes out, the peer's (as many) back.
type SwapFn<'a> = dyn FnMut(&[u8]) -> Result<Vec<u8>, String> + 'a;

/// What a rank can say about a step: one byte (1 = it worked) and up to 128 bytes of why. Two ranks swap these over NCCL after every
/// step, so both end with both answers and the same reason text.
const VERDICT_BYTES: usize = 1 + 128;

fn verdict_encode(r: &Result<(), String>) -> [u8; VERDICT_BYTES] {
    let mut b = [0u8; VERDICT_BYTES];
    match r {
        Ok(()) => b[0] = 1,
        Err(e) => {
            let mut n = e.len().min(VERDICT_BYTES - 1);
            while !e.is_char_boundary(n) {
                n -= 1;
            }
            b[1..1 + n].copy_from_slice(&e.as_bytes()[..n]);
        }
    }
    b
}

fn verdict_decode(b: &[u8]) -> Result<(), String> {
    if b.first() == Some(&1) {
        return Ok(());
    }
    let t = b.get(1..).unwrap_or(&[]);
    let n = t.iter().position(|&x| x == 0).unwrap_or(t.len());
    let why = String::from_utf8_lossy(&t[..n]).into_owned();
    Err(if why.is_empty() { "no reason given".into() } else { why })
}

/// Both ranks' answers to one step as one result, the same text on both: the failing ranks in rank order with their reasons.
fn verdicts_combine(what: &str, rank: usize, mine: Result<(), String>, theirs: Result<(), String>) -> Result<(), String> {
    let (r0, r1) = if rank == 0 { (mine, theirs) } else { (theirs, mine) };
    let failed: Vec<String> =
        [("rank 0", r0), ("rank 1", r1)].into_iter().filter_map(|(who, r)| r.err().map(|e| format!("{who}: {e}"))).collect();
    if failed.is_empty() {
        Ok(())
    } else {
        Err(format!("{what}: {}", failed.join("; ")))
    }
}

/// A step's verdicts swapped: the combined answer (`Ok(Err(..))` = a step that failed on some rank), or `Err` when the swap itself
/// failed (NCCL is gone: nothing to fall back to).
fn verdict_swap(swap: &mut SwapFn<'_>, rank: usize, what: &str, mine: Result<(), String>) -> Result<Result<(), String>, String> {
    let theirs = swap(&verdict_encode(&mine))?;
    Ok(verdicts_combine(what, rank, mine, verdict_decode(&theirs)))
}

/// What a RoCE bring-up came to.
pub enum Connected {
    Roce(RoceTp),
    /// `auto` ran on NCCL after all: the communicator it was given, back, and why (the same text on both ranks).
    Nccl {
        nccl: NcclTp,
        why: String,
    },
}

/// The RDMA devices and the locked-memory limit of this host, against what the plan registers: `Err` names what would make the
/// bring-up fail ("fail" being the state a rank cannot recover from by itself). `avail`: [`gb_cuda::roce::available`]; `memlock`: the
/// soft `RLIMIT_MEMLOCK` in bytes, `None` for unlimited (or for a process allowed to lock without limit); `port_state`: a device's
/// port 1 state text from sysfs (`4: ACTIVE`), `None` for a device this host does not have. Every configured device must be up (the
/// shim opens all of them, and the ranks must agree on the rails).
pub fn probe_with(
    hcas: &str,
    need_bytes: u64,
    avail: Result<(), String>,
    memlock: Option<u64>,
    port_state: impl Fn(&str) -> Option<String>,
) -> Result<(), String> {
    let mut why: Vec<String> = Vec::new();
    if let Err(e) = avail {
        why.push(format!("no RDMA verbs: {e}"));
    }
    for h in hcas.split(',').map(str::trim).filter(|h| !h.is_empty()) {
        match port_state(h) {
            None => why.push(format!("{h}: no such RDMA device")),
            Some(st) if !st.contains("ACTIVE") => why.push(format!("{h}: port 1 is {}", st.trim())),
            Some(_) => {}
        }
    }
    if let Some(limit) = memlock.filter(|&l| l < need_bytes) {
        why.push(format!(
            "locked-memory limit {} KiB is below the {} KiB region (ulimit -l; start from a unit with LimitMEMLOCK=infinity)",
            limit >> 10,
            need_bytes >> 10
        ));
    }
    if why.is_empty() {
        Ok(())
    } else {
        Err(why.join("; "))
    }
}

/// [`probe_with`] on this host and this plan.
pub fn probe(cfg: &RoceConfig, plan: &SitePlan) -> Result<(), String> {
    probe_with(&cfg.hcas, plan.region_lower_bound(), gb_cuda::roce::available(), memlock_limit(), |h| {
        std::fs::read_to_string(format!("/sys/class/infiniband/{h}/ports/1/state")).ok()
    })
}

/// The soft `RLIMIT_MEMLOCK` in bytes; `None` when it is unlimited, when the process holds CAP_IPC_LOCK (it locks past the limit), or
/// when the limit cannot be read (the bring-up's own steps then say).
fn memlock_limit() -> Option<u64> {
    let mut r = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    // SAFETY: r is a valid rlimit to write
    if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut r) } != 0 || r.rlim_cur == libc::RLIM_INFINITY {
        return None;
    }
    // CapEff bit 14 is CAP_IPC_LOCK
    let ipc_lock = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|t| t.lines().find_map(|l| l.strip_prefix("CapEff:").map(|v| v.trim().to_string())))
        .and_then(|v| u64::from_str_radix(&v, 16).ok())
        .is_some_and(|caps| caps & (1 << 14) != 0);
    (!ipc_lock).then_some(r.rlim_cur)
}

/// The steps of a bring-up, as the verdict protocol sees them (the seam of the tests; [`RealSteps`] is `RoceComm`).
trait Steps {
    /// This rank's settings parse (`REXL3_ROCE_*`, the zero-copy and fault variables): an answer the peer must have too, or it would
    /// wait for this rank in a swap that rank never makes.
    fn settings(&mut self) -> Result<(), String>;
    fn probe(&mut self) -> Result<(), String>;
    fn region(&mut self) -> Result<(), String>;
    fn blob(&mut self) -> Vec<u8>;
    fn connect(&mut self, peer: &[u8]) -> Result<(), String>;
    fn start(&mut self) -> Result<(), String>;
}

#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Up,
    /// Soft mode only: a step failed on some rank; the text is the same on both.
    Fallback(String),
}

/// Runs the steps with both ranks' verdicts swapped after each (which is also the barrier between them), so a rank that fails (no RDMA
/// device, a locked-memory limit) fails the bring-up on both ranks at once instead of leaving its peer in a blocking NCCL call. `soft`:
/// a failed step is an [`Outcome::Fallback`] on both ranks (the probe is the first step, then the region, the connection, the proxy);
/// strict: it is an `Err`, and there is no probe step. A swap that itself fails is an `Err` either way.
fn drive(steps: &mut dyn Steps, rank: usize, soft: bool, swap: &mut SwapFn<'_>) -> Result<Outcome, String> {
    let fail = |why: String| if soft { Ok(Outcome::Fallback(why)) } else { Err(why) };
    // a setting that does not parse is the operator's to fix, not a host's inability: an error under `auto` too
    let mine = steps.settings();
    verdict_swap(swap, rank, "the RoCE settings", mine)??;
    if soft {
        let mine = steps.probe();
        if let Err(w) = verdict_swap(swap, rank, "the RoCE probe", mine)? {
            return fail(w);
        }
    }
    let mine = steps.region();
    if let Err(w) = verdict_swap(swap, rank, "the RoCE region", mine)? {
        return fail(w);
    }
    let peer = swap(&steps.blob())?;
    let mine = steps.connect(&peer); // both QPs ready before either proxy posts
    if let Err(w) = verdict_swap(swap, rank, "the RoCE connection", mine)? {
        return fail(w);
    }
    let mine = steps.start();
    if let Err(w) = verdict_swap(swap, rank, "the RoCE proxy", mine)? {
        return fail(w);
    }
    Ok(Outcome::Up)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaultStep {
    Probe,
    Region,
    Connect,
    Proxy,
}

/// [`FAULT_ENV`] as it applies to `rank`: the step to fail here, if any (`probe@1` leaves rank 0 alone). A malformed value is an error.
fn fault_from(get: impl Fn(&str) -> Option<String>, rank: usize) -> Result<Option<FaultStep>, String> {
    let Some(v) = get(FAULT_ENV).map(|v| v.trim().to_string()).filter(|v| !v.is_empty()) else { return Ok(None) };
    let bad = || format!("{FAULT_ENV} = {v:?}: <probe|region|connect|proxy>@<0|1>");
    let (step, r) = v.split_once('@').ok_or_else(bad)?;
    let r: usize = r.parse().map_err(|_| bad())?;
    let step = match step {
        "probe" => FaultStep::Probe,
        "region" => FaultStep::Region,
        "connect" => FaultStep::Connect,
        "proxy" => FaultStep::Proxy,
        _ => return Err(bad()),
    };
    if r > 1 {
        return Err(bad());
    }
    Ok((r == rank).then_some(step))
}

/// This process's RoCE settings, parsed once.
struct Settings {
    cfg: RoceConfig,
    policy: RocePolicy,
    zero_copy: bool,
    fault: Option<FaultStep>,
}

fn settings_of(plan: &SitePlan, rank: usize) -> Result<Settings, String> {
    let cfg = RoceConfig::from_env()?;
    let largest = plan.caps().into_iter().max().ok_or("no RoCE site")?;
    let mut policy = RocePolicy::from_vars(|k| std::env::var(k).ok(), largest)?;
    policy.sum_max_bytes = plan.sum_max_with(|k| std::env::var(k).ok())?;
    Ok(Settings { cfg, policy, zero_copy: zero_copy_setting()?, fault: fault_from(|k| std::env::var(k).ok(), rank)? })
}

/// The steps on a real [`RoceComm`].
struct RealSteps<'a> {
    rank: usize,
    settings: &'a Result<Settings, String>,
    plan: &'a SitePlan,
    caps: &'a [usize],
    comm: Option<RoceComm>,
}

impl RealSteps<'_> {
    /// The parsed settings (the steps after `settings` run only once every rank's parsed).
    fn s(&self) -> Result<&Settings, String> {
        self.settings.as_ref().map_err(|e| e.clone())
    }

    fn injected(&self, step: FaultStep, name: &str) -> Result<(), String> {
        if self.s()?.fault == Some(step) {
            Err(format!("injected fault ({FAULT_ENV}={name}@{})", self.rank))
        } else {
            Ok(())
        }
    }
}

impl Steps for RealSteps<'_> {
    fn settings(&mut self) -> Result<(), String> {
        self.s().map(|_| ())
    }

    fn probe(&mut self) -> Result<(), String> {
        self.injected(FaultStep::Probe, "probe")?;
        probe(&self.s()?.cfg, self.plan)
    }

    fn region(&mut self) -> Result<(), String> {
        self.injected(FaultStep::Region, "region")?;
        self.comm = Some(RoceComm::new(self.rank, &self.s()?.cfg, self.caps)?);
        Ok(())
    }

    fn blob(&mut self) -> Vec<u8> {
        self.comm.as_ref().map(|c| c.blob()).unwrap_or_default()
    }

    fn connect(&mut self, peer: &[u8]) -> Result<(), String> {
        self.injected(FaultStep::Connect, "connect")?;
        self.comm.as_mut().ok_or("no region")?.connect(peer)
    }

    fn start(&mut self) -> Result<(), String> {
        self.injected(FaultStep::Proxy, "proxy")?;
        self.comm.as_ref().ok_or("no region")?.start()
    }
}

/// Whether this process asked for the RoCE transport (`REXL3_TP_TRANSPORT=roce`); an unknown value is an error.
pub fn wanted() -> Result<bool, String> {
    wanted_from(|k| std::env::var(k).ok())
}

/// `roce` asked for outright. `auto` is not "wanted" here: this is the shared fingerprint's question (`tp_comm::collective_fingerprint`,
/// Nemotron's too), whose NCCL-alone text must stay byte-identical; a family that runs `auto` adds its own part
/// ([`policy_fingerprint_from_vars`]).
fn wanted_from(get: impl Fn(&str) -> Option<String>) -> Result<bool, String> {
    match get(TRANSPORT_ENV).as_deref().map(str::trim) {
        None | Some("") | Some("nccl") | Some("auto") => Ok(false),
        Some("roce") => Ok(true),
        Some(o) => Err(format!("{TRANSPORT_ENV} must be auto, nccl or roce, not {o:?}")),
    }
}

/// What a process asks of [`TRANSPORT_ENV`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    /// RoCE if every rank can bring it up (a probe in the boot, then the per-step verdicts), NCCL on every rank otherwise.
    Auto,
    /// RoCE or the boot fails (the strict path).
    Roce,
    /// NCCL alone: the opt-out.
    Nccl,
}

impl Transport {
    pub const fn name(self) -> &'static str {
        match self {
            Transport::Auto => "auto",
            Transport::Roce => "roce",
            Transport::Nccl => "nccl",
        }
    }
}

/// This process's [`Transport`] (`default` when [`TRANSPORT_ENV`] is unset or empty); an unknown value is an error.
pub fn transport(default: Transport) -> Result<Transport, String> {
    transport_from(|k| std::env::var(k).ok(), default)
}

pub fn transport_from(get: impl Fn(&str) -> Option<String>, default: Transport) -> Result<Transport, String> {
    match get(TRANSPORT_ENV).as_deref().map(str::trim) {
        None | Some("") => Ok(default),
        Some("auto") => Ok(Transport::Auto),
        Some("roce") => Ok(Transport::Roce),
        Some("nccl") => Ok(Transport::Nccl),
        Some(o) => Err(format!("{TRANSPORT_ENV} must be auto, nccl or roce, not {o:?}")),
    }
}

/// Per path counters of the RoCE transport: (user, op, size class, went over RoCE) -> count.
type PathCounts = std::collections::BTreeMap<(u8, &'static str, usize, bool), u64>;

/// The group: the RoCE transport for the policy's sizes, NCCL for the rest.
pub struct RoceTp {
    roce: Arc<RoceComm>,
    nccl: NcclTp,
    topo: TpTopology,
    policy: RocePolicy,
    fingerprint: String,
    zero_copy: bool,
    /// Each [`SiteUser`]'s site ids.
    user_sites: [Vec<usize>; SITE_USERS],
    /// Host-side path decisions (lane T): (user, op, size class, RoCE?) -> calls. A call queued into a graph counts once; the device
    /// sequences ([`RoceComm::site_sequences`]) count every RoCE exchange, replays included.
    paths: std::sync::Mutex<PathCounts>,
}

/// [`ZERO_COPY_ENV`] (unset or empty: on).
pub fn zero_copy_setting() -> Result<bool, String> {
    zero_copy_from(std::env::var(ZERO_COPY_ENV).ok().as_deref())
}

fn zero_copy_from(v: Option<&str>) -> Result<bool, String> {
    match v.map(str::trim) {
        None | Some("") | Some("1") | Some("on") => Ok(true),
        Some("0") | Some("off") => Ok(false),
        Some(o) => Err(format!("{ZERO_COPY_ENV} must be 0 or 1, not {o:?}")),
    }
}

/// Who an exchange belongs to (lane T, per-user site lists): each user has its own sites, so two users of one size class never share
/// a site (its per-site order, its zero-copy parity). [`TpComm`]'s untagged calls are the [`SiteUser::Step`]'s (the decode / verify
/// sums and swaps of one stream); a prefill user names itself through [`TpComm::swap_bytes_for`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SiteUser {
    /// The step's exchanges (sums and swaps, in plan order on one stream).
    Step = 0,
    /// The GLM P2 part swap (its own exchange stream).
    PartSwap = 1,
    /// A prefill gather (lane E's KDA-out heads under P2, P5b).
    PrefillGather = 2,
}

/// The number of [`SiteUser`]s.
pub const SITE_USERS: usize = 3;

impl SiteUser {
    pub const ALL: [SiteUser; SITE_USERS] = [SiteUser::Step, SiteUser::PartSwap, SiteUser::PrefillGather];

    pub const fn name(self) -> &'static str {
        match self {
            SiteUser::Step => "step",
            SiteUser::PartSwap => "part_swap",
            SiteUser::PrefillGather => "prefill_gather",
        }
    }
}

/// The sites of each user (bytes a rank, one per size class). The transport's sites are the users' lists concatenated in
/// [`SiteUser::ALL`] order, so both ranks build the same layout from the same plan; [`SitePlan::fact`] names it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SitePlan {
    pub lists: [Vec<usize>; SITE_USERS],
    /// The largest in-place sum on the RoCE path when `REXL3_ROCE_SUM_MAX_BYTES` is unset (lane T: a family whose step sums
    /// beat NCCL past [`SUM_MAX_DEFAULT`] raises it with its step sites); `None`: [`SUM_MAX_DEFAULT`].
    pub sum_max_default: Option<usize>,
}

impl SitePlan {
    /// The step's sites alone.
    pub fn step(caps: &[usize]) -> SitePlan {
        SitePlan::default().with(SiteUser::Step, caps)
    }

    /// This plan with `user`'s sites set to `caps`.
    pub fn with(mut self, user: SiteUser, caps: &[usize]) -> SitePlan {
        self.lists[user as usize] = caps.to_vec();
        self
    }

    /// Every site's cap, in user order.
    pub fn caps(&self) -> Vec<usize> {
        self.lists.iter().flatten().copied().collect()
    }

    /// Each user's site ids in [`caps`](Self::caps).
    pub fn ids(&self) -> [Vec<usize>; SITE_USERS] {
        let mut out: [Vec<usize>; SITE_USERS] = Default::default();
        let mut next = 0;
        for (u, l) in self.lists.iter().enumerate() {
            out[u] = (next..next + l.len()).collect();
            next += l.len();
        }
        out
    }

    /// A lower bound of the pinned region this plan registers: four halves (two parities, two ranks) of every site, each rounded up
    /// to a page, plus the header (`gb_cuda::roce`'s layout adds the ring, the flag slots and a round-up to 64 KiB on top).
    pub fn region_lower_bound(&self) -> u64 {
        self.caps().iter().map(|&c| 4 * (c.max(1).div_ceil(4096) * 4096) as u64).sum::<u64>() + 65536
    }

    /// This plan with the RoCE sum limit's default set (see [`SitePlan::sum_max_default`]).
    pub fn with_sum_max(mut self, bytes: usize) -> SitePlan {
        self.sum_max_default = Some(bytes);
        self
    }

    /// The sum limit a rank of this plan runs: `REXL3_ROCE_SUM_MAX_BYTES` when set, else the plan's default.
    pub fn sum_max_with(&self, get: impl Fn(&str) -> Option<String>) -> Result<usize, String> {
        match get(SUM_MAX_ENV).filter(|v| !v.trim().is_empty()) {
            Some(v) => v.trim().parse().map_err(|_| format!("{SUM_MAX_ENV} = {v:?} is not a byte count")),
            None => Ok(self.sum_max_default.unwrap_or(SUM_MAX_DEFAULT)),
        }
    }

    /// `step:131072/..;part_swap:..[;summax=..]`: the users with sites, in order, and a raised sum limit (a Hello fact: two ranks with
    /// different plans build different regions or route a sum differently).
    pub fn fact(&self) -> String {
        let summax = self.sum_max_default.map(|b| format!(";summax={b}")).unwrap_or_default();
        Self::lists_fact(&self.lists) + &summax
    }

    fn lists_fact(lists: &[Vec<usize>; SITE_USERS]) -> String {
        SiteUser::ALL
            .iter()
            .filter(|u| !lists[**u as usize].is_empty())
            .map(|u| {
                let c: Vec<String> = lists[*u as usize].iter().map(|b| b.to_string()).collect();
                format!("{}:{}", u.name(), c.join("/"))
            })
            .collect::<Vec<_>>()
            .join(";")
    }
}

/// A zero-copy swap in flight (lane T): the site and its two halves `[rank 0's, rank 1's]` for the exchange
/// [`RoceTp::zero_copy_swap`] reserved. This rank writes its message into `halves[rank]` before [`RoceTp::publish_swap`]; after it,
/// `halves[1 - rank]` holds the peer's.
pub struct ZeroCopySwap {
    pub site: usize,
    pub bytes: usize,
    pub halves: [DevBuf; 2],
}

impl RoceTp {
    /// Bring the transport up beside an initialised NCCL group, two ranks, both calling it at the same point. `sites`: bytes a rank of
    /// the largest message of each site (one per size class the engine uses; the policy routes each message to the smallest site
    /// that holds it). `swap` sends this rank's bytes to the peer and returns the peer's (the rendezvous's side channel); `barrier`
    /// returns once both ranks reached it.
    pub fn connect(
        nccl: NcclTp,
        cfg: &RoceConfig,
        policy: RocePolicy,
        sites: &[usize],
        mut swap: impl FnMut(&[u8]) -> Result<Vec<u8>, String>,
        mut barrier: impl FnMut() -> Result<(), String>,
    ) -> Result<RoceTp, String> {
        let topo = nccl.topology();
        if topo.ranks != 2 {
            return Err(format!("the RoCE transport has two ranks, the group {}", topo.ranks));
        }
        let mut c = RoceComm::new(topo.rank, cfg, sites)?;
        let peer = swap(&c.blob())?;
        c.connect(&peer)?;
        barrier()?;
        c.start()?;
        barrier()?;
        let fingerprint = policy.fingerprint(cfg, sites);
        eprintln!(
            "tp: rank {} transport=roce rails={} region_bytes={} proxy_cpu={} isolated_threads={} {fingerprint}",
            topo.rank,
            c.rails(),
            c.region_bytes(),
            cfg.proxy_cpu.map_or("none".to_string(), |p| p.to_string()),
            c.isolated_threads()
        );
        let zero_copy = zero_copy_setting()?;
        let user_sites = SitePlan::step(sites).ids();
        Ok(RoceTp { roce: Arc::new(c), nccl, topo, policy, fingerprint, zero_copy, user_sites, paths: Default::default() })
    }

    /// [`connect`](Self::connect) with NCCL itself as the side channel: the two ranks swap their connection blobs (and pass the
    /// barriers) by `swap_bytes` on the communicator just made, so a family's rendezvous needs no new message. The configuration and the
    /// policy come from this process's environment (`RoceConfig::from_env`, `RocePolicy::from_vars`, the largest site the default
    /// `max_bytes`); both ranks must have the same ones, which the boot gate's collective fingerprint checks before this is called.
    pub fn connect_over_nccl(nccl: NcclTp, sites: &[usize]) -> Result<RoceTp, String> {
        Self::connect_over_nccl_plan(nccl, &SitePlan::step(sites))
    }

    /// [`connect_over_nccl`](Self::connect_over_nccl) with each user's own sites (lane T): the region holds every user's sites, and
    /// a user's exchanges go only to its own (`swap_bytes_for`, `zero_copy_swap`). Strict: a step that fails on either rank is an `Err`
    /// on both.
    pub fn connect_over_nccl_plan(nccl: NcclTp, plan: &SitePlan) -> Result<RoceTp, String> {
        match Self::connect_over_nccl_soft(nccl, plan, true)? {
            Connected::Roce(r) => Ok(r),
            Connected::Nccl { why, .. } => Err(why),
        }
    }

    /// The bring-up of [`connect_over_nccl_plan`](Self::connect_over_nccl_plan), strict or soft. Soft (`strict` false: `auto`): the
    /// first step is a probe of the host (RDMA verbs, the devices' ports, the locked-memory limit: [`probe`]), and a step that fails on
    /// either rank ends the bring-up on both with [`Connected::Nccl`]: the communicator back, and why (the same text on both ranks). A
    /// setting that does not parse is not a host's inability: an `Err` on both ranks (the first verdict is whether every rank's settings
    /// parsed, so no rank waits for a swap the other never makes); a group that is not two ranks is NCCL under `auto`. A swap that
    /// fails (NCCL itself) is an `Err`.
    pub fn connect_over_nccl_soft(nccl: NcclTp, plan: &SitePlan, strict: bool) -> Result<Connected, String> {
        let caps = plan.caps();
        let sites = caps.as_slice();
        let topo = nccl.topology();
        if topo.ranks != 2 {
            let why = format!("the RoCE transport has two ranks, the group {}", topo.ranks);
            return if strict { Err(why) } else { Ok(Connected::Nccl { nccl, why }) };
        }
        let settings = settings_of(plan, topo.rank);
        let st = Stream::new_nonblocking().map_err(|e| format!("a stream for the RoCE bootstrap: {e}"))?;
        let mut steps = RealSteps { rank: topo.rank, settings: &settings, plan, caps: sites, comm: None };
        let outcome = drive(&mut steps, topo.rank, !strict, &mut |b: &[u8]| nccl_swap(&nccl, &st, b))?;
        let c = match (outcome, steps.comm.take()) {
            (Outcome::Fallback(why), c) => {
                // the region and the connection of this rank, whatever state they reached, are released here (`RoceComm`'s drop stops
                // the proxy if it ran and closes the queue pairs); the communicator was never touched but by the verdict swaps
                drop(c);
                return Ok(Connected::Nccl { nccl, why });
            }
            (Outcome::Up, Some(c)) => c,
            (Outcome::Up, None) => return Err("the RoCE bring-up finished without a region".into()),
        };
        // every rank's settings parsed (the first verdict), so these are there
        let Settings { cfg, policy, zero_copy, .. } = settings?;
        let fingerprint = policy.fingerprint(&cfg, sites);
        eprintln!(
            "tp: rank {} transport=roce rails={} region_bytes={} proxy_cpu={} isolated_threads={} {fingerprint}",
            topo.rank,
            c.rails(),
            c.region_bytes(),
            cfg.proxy_cpu.map_or("none".to_string(), |p| p.to_string()),
            c.isolated_threads()
        );
        Ok(Connected::Roce(RoceTp {
            roce: Arc::new(c),
            nccl,
            topo,
            policy,
            fingerprint,
            zero_copy,
            user_sites: plan.ids(),
            paths: Default::default(),
        }))
    }

    /// Count one path decision (the size class: the cap of the user's smallest site that holds `bytes`, or 0 above every site).
    fn count(&self, user: SiteUser, op: &'static str, bytes: usize, roce: bool) {
        let class = self
            .roce
            .site_for_among(&self.user_sites[user as usize], bytes)
            .and_then(|s| self.roce.site(s))
            .map_or(0, |s| s.cap as usize);
        if let Ok(mut m) = self.paths.lock() {
            *m.entry((user as u8, op, class, roce)).or_insert(0) += 1;
        }
    }

    /// The path counters as one line (lane T): `paths=user.op.class:roce|nccl=calls,..` (host decisions; a captured call once) and
    /// `site_seqs=cap:exchanges,..` (the device's count of every RoCE exchange a site ran, replays included).
    pub fn paths_line(&self) -> String {
        let name = |u: u8| SiteUser::ALL.get(u as usize).map_or("?", |u| u.name());
        let size = |b: usize| {
            if b == 0 {
                "above".to_string()
            } else if b % (1 << 20) == 0 {
                format!("{}M", b >> 20)
            } else {
                format!("{}K", b >> 10)
            }
        };
        let paths: Vec<String> = self
            .paths
            .lock()
            .map(|m| {
                m.iter()
                    .map(|((u, op, c, r), n)| format!("{}.{op}.{}:{}={n}", name(*u), size(*c), if *r { "roce" } else { "nccl" }))
                    .collect()
            })
            .unwrap_or_default();
        let seqs = match self.roce.site_sequences() {
            Ok(v) => self.roce.site_caps().iter().zip(v).map(|(c, n)| format!("{}:{n}", size(*c))).collect::<Vec<_>>().join(","),
            Err(e) => format!("unreadable({e})"),
        };
        format!("paths={} site_seqs={seqs} summax={}", paths.join(","), self.policy.sum_max_bytes)
    }

    /// Whether zero-copy swaps are allowed ([`ZERO_COPY_ENV`]).
    pub fn zero_copy(&self) -> bool {
        self.zero_copy
    }

    /// A zero-copy swap of `bytes` a rank (lane T, the GLM P2 part swap), when [`ZERO_COPY_ENV`] allows it, the policy routes `bytes`
    /// to RoCE, the message takes the bulk path (`bulk_min` and up: those sites are the prefill's, queued eagerly) and the site's
    /// parity is known ([`RoceComm::next_halves`]). `None`: use [`TpComm::swap_bytes`]. The caller writes this rank's bytes into
    /// `halves[rank]`, calls [`publish_swap`](Self::publish_swap) on the stream that follows that write, and reads `halves[1 - rank]`
    /// after it (on a stream ordered after the publish), before it next swaps `bytes` twice more. Both ranks make the same calls (the
    /// same `bytes` in the same order), so their sites and parities match; the wire is the standalone swap's.
    pub fn zero_copy_swap(&self, bytes: usize) -> Option<ZeroCopySwap> {
        if !self.zero_copy || bytes == 0 || bytes < self.roce.config().bulk_min {
            return None;
        }
        let site = self.route_for(SiteUser::PartSwap, bytes)?;
        let halves = self.roce.next_halves(site)?;
        Some(ZeroCopySwap { site, bytes, halves })
    }

    /// Publish this rank's half of `z` and wait for the peer's (one block, no staging, no copy-out: `Mode::None`).
    pub fn publish_swap(&self, z: &ZeroCopySwap, st: &Stream) -> Result<(), CudaError> {
        self.count(SiteUser::PartSwap, "zc_swap", z.bytes, true);
        self.roce.exchange(z.site, None, None, z.bytes, z.bytes, 0, gb_cuda::roce::Mode::None, st)
    }

    /// The transport itself: the fused-site API (`dev()`, `site(i)`) for kernels that publish and consume in place.
    pub fn roce(&self) -> &Arc<RoceComm> {
        &self.roce
    }

    pub fn nccl(&self) -> &NcclTp {
        &self.nccl
    }

    pub fn policy(&self) -> RocePolicy {
        self.policy
    }

    /// What both ranks must agree on (for the boot gate).
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// The site of an in-place f32 sum of `bytes` on the RoCE path (`route`, and at most `sum_max_bytes`).
    fn route_sum(&self, bytes: usize) -> Option<usize> {
        if bytes > self.policy.sum_max_bytes {
            return None;
        }
        self.route(bytes)
    }

    /// The step's site of a message of `bytes` on the RoCE path, `None` when NCCL carries it.
    fn route(&self, bytes: usize) -> Option<usize> {
        self.route_for(SiteUser::Step, bytes)
    }

    /// `user`'s site of a message of `bytes` on the RoCE path (the smallest of its own sites that holds it), `None` when NCCL
    /// carries it.
    fn route_for(&self, user: SiteUser, bytes: usize) -> Option<usize> {
        if self.policy.uses(bytes) {
            self.roce.site_for_among(&self.user_sites[user as usize], bytes)
        } else {
            None
        }
    }
}

impl TpComm for RoceTp {
    fn topology(&self) -> TpTopology {
        self.topo
    }

    /// The exact two-rank sum: RoCE (`add.rn` in rank order) within the policy, NCCL's ring outside it. Same bits either way.
    fn allreduce_sum_f32(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
        match self.route_sum(count.saturating_mul(4)) {
            Some(site) if count > 0 => {
                self.count(SiteUser::Step, "sum", count * 4, true);
                self.roce.sum_f32(site, buf, off_bytes, count, st)
            }
            _ => {
                self.count(SiteUser::Step, "sum", count * 4, false);
                self.nccl.allreduce_sum_f32(buf, off_bytes, count, st)
            }
        }
    }

    fn allgather_sum_f32(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
        match self.route_sum(count.saturating_mul(4)) {
            Some(site) if count > 0 => {
                self.count(SiteUser::Step, "sum", count * 4, true);
                self.roce.sum_f32(site, buf, off_bytes, count, st)
            }
            _ => {
                self.count(SiteUser::Step, "sum", count * 4, false);
                self.nccl.allgather_sum_f32(buf, off_bytes, count, st)
            }
        }
    }

    /// Capturable when NCCL (which carries the sizes outside the policy) is: the RoCE path is stream-ordered (one kernel per exchange,
    /// launched with PDL; the host only checks arguments), its sequences live in device memory so a replay runs new exchanges, and it
    /// synchronises nothing on the host (a replayed exchange that finds the abort word raised returns without its result, and the
    /// error surfaces through `async_error`). Proven by `crates/cuda/tests/roce_pair.rs` (d): 300 replays of a 12-exchange graph on
    /// two sites, both parities, fresh data, eager exchanges between, bit-exact against NCCL; and by mb5's checked graph replays.
    /// The contract it relies on is the eager one: every exchange of a site is issued in the same order on both ranks, on one stream
    /// (or streams ordered by events), so two captured branches must not exchange through this group concurrently (the side branches'
    /// exchanges go through [`aux`](TpComm::aux), NCCL's second communicator).
    fn capturable(&self) -> bool {
        self.nccl.capturable()
    }

    fn supports_swap(&self) -> bool {
        true
    }

    /// The decision uses the larger of the two sizes (the same number on both ranks); a pair of sizes on opposite sides of the bulk
    /// threshold goes by NCCL (both ranks see the same pair).
    fn swap_bytes(
        &self,
        send: &DevBuf,
        send_off: usize,
        send_bytes: usize,
        recv: &mut DevBuf,
        recv_off: usize,
        recv_bytes: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        self.swap_bytes_for(SiteUser::Step, send, send_off, send_bytes, recv, recv_off, recv_bytes, st)
    }

    /// [`swap_bytes`](TpComm::swap_bytes) on `user`'s own sites (NCCL outside them).
    fn swap_bytes_for(
        &self,
        user: SiteUser,
        send: &DevBuf,
        send_off: usize,
        send_bytes: usize,
        recv: &mut DevBuf,
        recv_off: usize,
        recv_bytes: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let bulk = self.roce.config().bulk_min;
        let same_path = (send_bytes >= bulk) == (recv_bytes >= bulk);
        let big = send_bytes.max(recv_bytes);
        match self.route_for(user, big) {
            Some(site) if same_path && send_bytes > 0 && recv_bytes > 0 => {
                self.count(user, "swap", big, true);
                self.roce.swap(site, send, send_off, send_bytes, recv, recv_off, recv_bytes, st)
            }
            _ => {
                self.count(user, "swap", big, false);
                self.nccl.swap_bytes(send, send_off, send_bytes, recv, recv_off, recv_bytes, st)
            }
        }
    }

    /// Both: the transport's abort word (spinning kernels return) and `ncclCommAbort`.
    fn abort(&self) {
        self.roce.abort();
        self.nccl.abort();
    }

    fn async_error(&self) -> Option<String> {
        self.roce.async_error().or_else(|| self.nccl.async_error())
    }

    fn aux(&self) -> Option<&dyn TpComm> {
        self.nccl.aux()
    }

    fn roce_tp(&self) -> Option<&RoceTp> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_policy_routes_by_size_and_refuses_a_limit_above_the_sites() {
        let none = |_: &str| None;
        let p = RocePolicy::from_vars(none, 24 << 20).unwrap();
        assert_eq!(p, RocePolicy { min_bytes: 1, max_bytes: 24 << 20, sum_max_bytes: SUM_MAX_DEFAULT });
        assert!(p.uses(4) && p.uses(24 << 20) && !p.uses((24 << 20) + 1));
        let env = |k: &str| match k {
            MIN_ENV => Some("4096".to_string()),
            MAX_ENV => Some("1048576".to_string()),
            _ => None,
        };
        let q = RocePolicy::from_vars(env, 24 << 20).unwrap();
        assert!(!q.uses(4095) && q.uses(4096) && q.uses(1 << 20) && !q.uses((1 << 20) + 1));
        assert!(RocePolicy::from_vars(|k: &str| (k == MAX_ENV).then(|| "99999999999".into()), 24 << 20).is_err());
        let f = q.fingerprint(&RoceConfig::default(), &[65536, 1 << 20]);
        assert!(f.starts_with("roce:v1,") && f.ends_with(",min=4096,max=1048576,summax=1048576,sites=65536/1048576"), "{f}");
        let big = RocePolicy::from_vars(|k: &str| (k == SUM_MAX_ENV).then(|| "67108864".into()), 24 << 20).unwrap();
        assert_eq!(big.sum_max_bytes, 64 << 20);
        assert!(RocePolicy::from_vars(|k: &str| (k == SUM_MAX_ENV).then(|| "lots".into()), 24 << 20).is_err());
    }

    #[test]
    fn zero_copy_is_on_unless_switched_off_and_a_typo_is_refused() {
        assert_eq!(zero_copy_from(None), Ok(true));
        assert_eq!(zero_copy_from(Some(" ")), Ok(true));
        assert_eq!(zero_copy_from(Some("1")), Ok(true));
        assert_eq!(zero_copy_from(Some("0")), Ok(false));
        assert_eq!(zero_copy_from(Some("off")), Ok(false));
        assert!(zero_copy_from(Some("maybe")).unwrap_err().contains(ZERO_COPY_ENV));
    }

    #[test]
    fn the_boot_fingerprint_names_the_size_rule_only_when_roce_is_chosen() {
        let vars = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
        };
        // NCCL alone (unset, empty, nccl): nothing is added, so every existing kit's fingerprint is unchanged
        assert_eq!(fingerprint_from_vars(vars(&[])), None);
        assert_eq!(fingerprint_from_vars(vars(&[(TRANSPORT_ENV, " ")])), None);
        assert_eq!(fingerprint_from_vars(vars(&[(TRANSPORT_ENV, "nccl")])), None);
        let d = fingerprint_from_vars(vars(&[(TRANSPORT_ENV, "roce")])).unwrap();
        assert_eq!(d, format!("{},min=1,max=sites,summax=1048576", RoceConfig::default().fingerprint()));
        let t = fingerprint_from_vars(vars(&[
            (TRANSPORT_ENV, "roce"),
            ("REXL3_ROCE_STRIPE_MIN", "65536"),
            (MIN_ENV, "64"),
            (MAX_ENV, "25165824"),
        ]))
        .unwrap();
        assert!(t.contains(",stripe=65536,") && t.ends_with(",min=64,max=25165824,summax=1048576"), "{t}");
        // the proxy core is per node and stays out of it
        assert_eq!(fingerprint_from_vars(vars(&[(TRANSPORT_ENV, "roce"), ("REXL3_ROCE_CPU", "7")])).unwrap(), d);
        // a typo never matches a valid setting
        assert!(fingerprint_from_vars(vars(&[(TRANSPORT_ENV, "rdma")])).unwrap().starts_with("invalid("));
        assert!(fingerprint_from_vars(vars(&[(TRANSPORT_ENV, "roce"), (MIN_ENV, "x")])).unwrap().starts_with("invalid("));
    }

    fn vars(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
    }

    #[test]
    fn the_transport_setting_has_a_default_per_family_and_a_typo_is_refused() {
        for (v, want) in [("auto", Transport::Auto), (" roce ", Transport::Roce), ("nccl", Transport::Nccl)] {
            let got = transport_from(|_: &str| Some(v.to_string()), Transport::Nccl).unwrap();
            assert_eq!(got, want, "{v:?}");
            assert_eq!(got.name(), v.trim());
        }
        // unset or empty: the family's own default (Nemotron NCCL, GLM auto)
        assert_eq!(transport_from(vars(&[]), Transport::Nccl), Ok(Transport::Nccl));
        assert_eq!(transport_from(vars(&[]), Transport::Auto), Ok(Transport::Auto));
        assert_eq!(transport_from(vars(&[(TRANSPORT_ENV, " ")]), Transport::Auto), Ok(Transport::Auto));
        assert!(transport_from(vars(&[(TRANSPORT_ENV, "rdma")]), Transport::Auto).unwrap_err().contains(TRANSPORT_ENV));
        // the shared fingerprint (Nemotron's too) keeps treating anything but `roce` as NCCL alone, so no existing Hello moves
        assert_eq!(wanted_from(vars(&[(TRANSPORT_ENV, "auto")])), Ok(false));
        assert_eq!(wanted_from(vars(&[(TRANSPORT_ENV, "roce")])), Ok(true));
        assert_eq!(fingerprint_from_vars(vars(&[(TRANSPORT_ENV, "auto")])), None);
        // the RoCE rule `auto` adds to GLM's fact is the one `roce` prints
        let roce = fingerprint_from_vars(vars(&[(TRANSPORT_ENV, "roce"), (MIN_ENV, "64")])).unwrap();
        let policy = policy_fingerprint_from_vars(&vars(&[(TRANSPORT_ENV, "auto"), (MIN_ENV, "64")]));
        assert_eq!(roce, policy);
        assert!(policy_fingerprint_from_vars(&vars(&[(MIN_ENV, "x")])).starts_with("invalid("));
    }

    #[test]
    fn the_region_bound_is_four_halves_of_every_site_and_the_header() {
        let p = SitePlan::step(&[128 << 10, 1 << 20]).with(SiteUser::PartSwap, &[8 << 20]);
        assert_eq!(p.region_lower_bound(), 4 * ((128 << 10) + (1 << 20) + (8 << 20)) as u64 + 65536);
        // a cap that is not a page multiple is rounded up, as the layout does
        assert_eq!(SitePlan::step(&[5000]).region_lower_bound(), 4 * 8192 + 65536);
    }

    #[test]
    fn the_probe_names_what_this_host_cannot_do_and_passes_a_host_that_can() {
        let up = |h: &str| Some(if h.starts_with("roce") { "4: ACTIVE\n".to_string() } else { "1: DOWN\n".to_string() });
        let need = 64 << 20;
        let hcas = "rocep1s0f0,roceP2p1s0f0";
        assert_eq!(probe_with(hcas, need, Ok(()), None, up), Ok(()));
        assert_eq!(probe_with(hcas, need, Ok(()), Some(need), up), Ok(()), "exactly enough locked memory");
        let low = probe_with(hcas, need, Ok(()), Some(8 << 20), up).unwrap_err();
        assert!(low.contains("locked-memory limit 8192 KiB is below the 65536 KiB region"), "{low}");
        let verbs = probe_with(hcas, need, Err("no libibverbs.so.1".into()), None, up).unwrap_err();
        assert!(verbs.contains("no RDMA verbs: no libibverbs.so.1"), "{verbs}");
        let down = probe_with("rocep1s0f1,roceP2p1s0f0", need, Ok(()), None, |h: &str| {
            Some(if h == "rocep1s0f1" { "1: DOWN\n".to_string() } else { "4: ACTIVE\n".to_string() })
        })
        .unwrap_err();
        assert_eq!(down, "rocep1s0f1: port 1 is 1: DOWN");
        let gone = probe_with("mlx5_9", need, Ok(()), None, |_: &str| None).unwrap_err();
        assert_eq!(gone, "mlx5_9: no such RDMA device");
        // every reason, in one text
        let all = probe_with("mlx5_9", need, Err("x".into()), Some(1), |_: &str| None).unwrap_err();
        assert!(all.starts_with("no RDMA verbs: x; mlx5_9: no such RDMA device; locked-memory limit 0 KiB"), "{all}");
    }

    #[test]
    fn verdicts_round_trip_and_both_ranks_read_the_same_text() {
        assert_eq!(verdict_decode(&verdict_encode(&Ok(()))), Ok(()));
        assert_eq!(verdict_decode(&verdict_encode(&Err("no verbs".into()))), Err("no verbs".into()));
        assert_eq!(verdict_decode(&[0]), Err("no reason given".into()));
        // a long reason is cut on a character boundary, never inside one
        let long = "é".repeat(200);
        let Err(cut) = verdict_decode(&verdict_encode(&Err(long))) else { panic!("an error stays one") };
        assert!(cut.len() < VERDICT_BYTES && cut.chars().all(|c| c == 'é'), "{}", cut.len());
        // the same combined text on both ranks, failing ranks in rank order
        let a = verdicts_combine("the RoCE region", 0, Ok(()), Err("ENOMEM".into()));
        let b = verdicts_combine("the RoCE region", 1, Err("ENOMEM".into()), Ok(()));
        assert_eq!(a, b);
        assert_eq!(a, Err("the RoCE region: rank 1: ENOMEM".into()));
        let both = verdicts_combine("the RoCE probe", 1, Err("b".into()), Err("a".into()));
        assert_eq!(both, Err("the RoCE probe: rank 0: a; rank 1: b".into()));
        assert_eq!(verdicts_combine("x", 0, Ok(()), Ok(())), Ok(()));
    }

    #[test]
    fn the_fault_hook_names_a_step_and_a_rank_and_nothing_else() {
        let f = |v: &'static str, rank| fault_from(vars_one(v), rank);
        assert_eq!(f("probe@1", 1), Ok(Some(FaultStep::Probe)));
        assert_eq!(f("probe@1", 0), Ok(None));
        assert_eq!(f("region@0", 0), Ok(Some(FaultStep::Region)));
        assert_eq!(f("connect@1", 1), Ok(Some(FaultStep::Connect)));
        assert_eq!(f(" proxy@1 ", 1), Ok(Some(FaultStep::Proxy)));
        assert_eq!(fault_from(|_: &str| None, 1), Ok(None));
        for bad in ["probe", "probe@", "probe@2", "start@1", "@1", "probe@x"] {
            assert!(f(bad, 1).unwrap_err().contains(FAULT_ENV), "{bad:?}");
        }
    }

    fn vars_one(v: &'static str) -> impl Fn(&str) -> Option<String> {
        move |k: &str| (k == FAULT_ENV).then(|| v.to_string())
    }

    /// A bring-up with scripted failures: the step that fails on this rank, and the calls it saw.
    struct Scripted {
        fail: Option<&'static str>,
        calls: Vec<&'static str>,
    }

    impl Scripted {
        fn step(&mut self, name: &'static str) -> Result<(), String> {
            self.calls.push(name);
            if self.fail == Some(name) {
                Err(format!("{name} refused here"))
            } else {
                Ok(())
            }
        }
    }

    impl Steps for Scripted {
        fn settings(&mut self) -> Result<(), String> {
            self.step("settings")
        }
        fn probe(&mut self) -> Result<(), String> {
            self.step("probe")
        }
        fn region(&mut self) -> Result<(), String> {
            self.step("region")
        }
        fn blob(&mut self) -> Vec<u8> {
            vec![7; 16]
        }
        fn connect(&mut self, peer: &[u8]) -> Result<(), String> {
            assert_eq!(peer, [7; 16], "the peer's blob arrived whole");
            self.step("connect")
        }
        fn start(&mut self) -> Result<(), String> {
            self.step("proxy")
        }
    }

    /// Two ranks driving the steps against each other, their swaps through channels: each rank's outcome and the calls it made.
    /// `peer_gone`: rank 1 never answers (its channel ends), as a process that died would.
    fn run_pair(
        soft: bool,
        fail: [Option<&'static str>; 2],
        peer_gone: bool,
    ) -> [(Result<Outcome, String>, Vec<&'static str>); 2] {
        use std::sync::mpsc::channel;
        let (t01, r01) = channel::<Vec<u8>>();
        let (t10, r10) = channel::<Vec<u8>>();
        let run = |rank: usize, tx: std::sync::mpsc::Sender<Vec<u8>>, rx: std::sync::mpsc::Receiver<Vec<u8>>, hang_up: bool| {
            std::thread::spawn(move || {
                let mut steps = Scripted { fail: fail[rank], calls: Vec::new() };
                if hang_up {
                    return (Err("the peer is gone".to_string()), steps.calls);
                }
                let mut swap = |b: &[u8]| {
                    tx.send(b.to_vec()).map_err(|_| "the peer is gone".to_string())?;
                    rx.recv_timeout(std::time::Duration::from_secs(20)).map_err(|_| "the peer is gone".to_string())
                };
                let out = drive(&mut steps, rank, soft, &mut swap);
                (out, steps.calls)
            })
        };
        let h0 = run(0, t01, r10, false);
        let h1 = run(1, t10, r01, peer_gone);
        [h0.join().unwrap(), h1.join().unwrap()]
    }

    #[test]
    fn a_failed_step_on_one_rank_ends_both_on_nccl_under_auto_and_fails_both_under_roce() {
        let all = ["probe", "region", "connect", "proxy"];
        let with_settings = |steps: &[&'static str]| -> Vec<&'static str> { [&["settings"][..], steps].concat() };
        // nothing fails: both come up; strict skips the probe step
        let [a, b] = run_pair(true, [None, None], false);
        assert_eq!((a.0, a.1.clone()), (Ok(Outcome::Up), with_settings(&all)));
        assert_eq!((b.0, b.1), (Ok(Outcome::Up), with_settings(&all)));
        let [a, b] = run_pair(false, [None, None], false);
        assert_eq!((a.0, a.1), (Ok(Outcome::Up), with_settings(&all[1..])));
        assert_eq!(b.0, Ok(Outcome::Up));
        // each step failing on one rank only (either rank), under `auto`: both say the same, and neither goes past it
        for (step, at) in all.iter().enumerate().map(|(i, s)| (*s, i)) {
            for bad in 0..2 {
                let mut fail = [None, None];
                fail[bad] = Some(step);
                let [a, b] = run_pair(true, fail, false);
                let want = format!("rank {bad}: {step} refused here");
                for (rank, r) in [a, b].into_iter().enumerate() {
                    let Ok(Outcome::Fallback(why)) = &r.0 else { panic!("{step}@{bad} rank {rank}: {:?}", r.0) };
                    assert!(why.ends_with(&want), "{step}@{bad} rank {rank}: {why}");
                    // up to and including the failed step, never beyond: the later steps never ran on either rank
                    assert_eq!(r.1, with_settings(&all[..=at]), "{step}@{bad} rank {rank} calls");
                }
            }
        }
        // the same failures under `roce` are errors on both ranks (the probe is not a strict step)
        for step in ["region", "connect", "proxy"] {
            let [a, b] = run_pair(false, [None, Some(step)], false);
            for r in [a, b] {
                let Err(why) = &r.0 else { panic!("{step}: {:?}", r.0) };
                assert!(why.contains(&format!("rank 1: {step} refused here")), "{why}");
            }
        }
        let [a, b] = run_pair(false, [Some("probe"), Some("probe")], false);
        assert_eq!((a.0, b.0), (Ok(Outcome::Up), Ok(Outcome::Up)), "strict never probes");
        // settings that do not parse on ONE rank: an error on both under either mode (never a fallback, never a rank left waiting),
        // and nothing else ran
        for soft in [true, false] {
            for bad in 0..2 {
                let mut fail = [None, None];
                fail[bad] = Some("settings");
                let [a, b] = run_pair(soft, fail, false);
                for (rank, r) in [a, b].into_iter().enumerate() {
                    let Err(why) = &r.0 else { panic!("settings@{bad} soft={soft} rank {rank}: {:?}", r.0) };
                    assert_eq!(why, &format!("the RoCE settings: rank {bad}: settings refused here"));
                    assert_eq!(r.1, ["settings"], "settings@{bad} soft={soft} rank {rank}");
                }
            }
        }
        // a peer that stops answering is an error, never a fallback (NCCL is what is gone)
        let [a, _] = run_pair(true, [None, None], true);
        assert!(a.0.is_err() && a.0.unwrap_err().contains("gone"));
    }
}
