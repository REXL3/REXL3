//! The collective of the tensor-parallel engine, three ways (`docs/NEMOTRON_TP.md` sections 4 and 10): [`NcclTp`] over the real
//! fabric between nodes, [`LoopbackComm`] for R shards in ONE process on ONE GPU (the development loop and the `cargo test` gate),
//! and the single-rank no-op that lives in `tp.rs`. All implement [`TpComm`]: an in-place f32 sum of `count` elements at a byte offset,
//! queued on the caller's stream, exactly `a + b` with two ranks.
//!
//! # Ring or one-shot
//!
//! The ring allreduce costs 2(R-1) network hops, which is what would make a larger group slower per step. The one-shot form
//! ([`TpComm::allgather_sum_f32`]) sends the rank's range to every peer and sums the R operands in fixed rank order with one kernel:
//! one hop whatever R is, `(R-1) x S` bytes a rank. Because every rank holds exact `-0.0` outside its own elements the sum is a copy of
//! the owner's element in ANY order, so the two forms give the same bits. [`CollectivePolicy`] (env `REXL3_TP_COLLECTIVE=ring|oneshot|
//! auto`, default `ring`) is the one place that decides which form [`TpComm::allreduce_sum_f32`] uses for a message of a given size;
//! the decision is a function of the policy and the element count only, so it is the same on every rank, and
//! [`collective_fingerprint`] puts the policy in the boot gate so that a mismatch is refused (a ring on one rank and a one-shot on
//! another would deadlock).
//!
//! # Contract the callers (TP-M's arms, TP-K's MoE hook) rely on
//!
//! * Every rank issues the same sequence of collectives with the same `count`s (a rank that skips one, or passes another count, is a bug:
//!   the NCCL group would hang or corrupt; the loopback group detects a count mismatch and returns an error).
//! * Disjoint support: on every element at most one rank holds a non-zero value, so the sum is a copy and TP=2 is byte-identical to TP=1.
//! * `off_bytes` is a multiple of 4 (16 for best speed) and `off_bytes + 4 * count <= buf.bytes()`; otherwise the call returns
//!   `cudaErrorInvalidValue`-class [`nccl::ERR_BAD_ARGUMENT`] without launching anything.
//! * The NCCL implementation is stream-ordered and capturable into a CUDA graph (proved across two nodes by
//!   `crates/cuda/tests/nccl_pair.rs`): connect (and run one eager collective) BEFORE capturing, and drop every graph that captured the
//!   communicator before dropping it.
//! * The loopback implementation is NOT capturable: it synchronises its stream and the host (a barrier) inside the call. A loopback run is
//!   graphs-off by construction. It is exact and ordered the same way, which is what the equality gates need.

use super::tp::{TpComm, TpTopology};
use crate::serve::TpOptions;
use gb_cuda::nccl::{self, NcclComm, NcclUniqueId};
use gb_cuda::{CudaError, DevBuf, Stream};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------------------------------------------
// configuration

/// Where this process sits in the TP group and where its peer is, from `REXL3_TP_RANKS` (default 1), `REXL3_TP_RANK` (default 0),
/// `REXL3_TP_PEER` (`HOST:PORT`, required when there are two ranks: rank 0's address, which it listens on and rank 1 connects to) and
/// the three timeouts the server's [`TpOptions`] reads (`REXL3_TP_CONNECT_TIMEOUT_S`, `REXL3_TP_READY_TIMEOUT_S`,
/// `REXL3_TP_PEER_TIMEOUT_S`). It is a view of the SAME parse (`TpOptions::from_vars`, then `validate`): one place that knows the names
/// and the defaults.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TpConfig {
    pub ranks: usize,
    pub rank: usize,
    /// `HOST:PORT` of rank 0's rendezvous / lockstep listener. `None` for a single rank.
    pub peer: Option<String>,
    /// How long rank 0 waits for rank 1 to connect (and rank 1 for rank 0 to listen).
    pub connect_timeout: Duration,
    /// How long a rank waits for the other's shard to load.
    pub ready_timeout: Duration,
    /// No frame from the other rank for this long is a lost peer.
    pub peer_timeout: Duration,
}

impl TpConfig {
    pub const SINGLE: TpConfig = TpConfig {
        ranks: 1,
        rank: 0,
        peer: None,
        connect_timeout: Duration::from_secs(TpOptions::DEFAULT_CONNECT_S),
        ready_timeout: Duration::from_secs(TpOptions::DEFAULT_READY_S),
        peer_timeout: Duration::from_secs(TpOptions::DEFAULT_PEER_S),
    };

    /// Read the variables from the process environment (no variable set: one rank).
    pub fn from_env() -> Result<TpConfig, String> {
        Self::from_vars(|k| std::env::var(k).ok())
    }

    /// [`from_env`](Self::from_env) over any variable lookup (what the unit tests use).
    pub fn from_vars(get: impl Fn(&str) -> Option<String>) -> Result<TpConfig, String> {
        match TpOptions::from_vars(get)? {
            None => Ok(TpConfig::SINGLE),
            Some(o) => {
                o.validate()?;
                Ok(TpConfig::from(&o))
            }
        }
    }

    pub fn topology(&self) -> TpTopology {
        TpTopology { rank: self.rank, ranks: self.ranks }
    }

    pub fn is_single(&self) -> bool {
        self.ranks == 1
    }
}

impl From<&TpOptions> for TpConfig {
    fn from(o: &TpOptions) -> TpConfig {
        TpConfig {
            ranks: o.ranks,
            rank: o.rank,
            peer: (o.ranks > 1).then(|| o.peer.clone()),
            connect_timeout: o.connect_timeout,
            ready_timeout: o.ready_timeout,
            peer_timeout: o.peer_timeout,
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the collective policy: ring or one-shot

/// The variable that picks the algorithm of the group's collectives: `ring` (default), `oneshot` or `auto`. Both ranks need the same
/// value (the boot gate compares [`collective_fingerprint`]).
pub const COLLECTIVE_ENV: &str = "REXL3_TP_COLLECTIVE";
/// The largest message (bytes) the one-shot form carries when the mode is not `ring`; larger messages stay on the ring. Defaults:
/// [`CollectivePolicy::AUTO_MAX_BYTES`] for `auto`, [`CollectivePolicy::ONESHOT_MAX_BYTES`] for `oneshot`.
pub const ONESHOT_MAX_ENV: &str = "REXL3_TP_ONESHOT_MAX_BYTES";

/// Which algorithm carries a collective of a given size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectiveMode {
    /// The ring allreduce for every collective (NCCL's own). The default until the one-shot has passed its A/B on the real model.
    Ring,
    /// The one-shot exchange for every collective up to the limit (sized to cover the largest MoE `y`: 23 MB at 256 rows), the ring above.
    OneShot,
    /// The one-shot exchange up to the (smaller) crossover limit, where `2 x hop x bandwidth` of the ring starts to win, the ring above.
    Auto,
}

/// The decision of [`TpComm::allreduce_sum_f32`] for one message: ring or one-shot, and the staging the one-shot needs. A pure function
/// of the policy and the element count, so every rank decides alike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollectivePolicy {
    pub mode: CollectiveMode,
    /// The largest message (bytes) the one-shot carries; 0 for `Ring`.
    pub max_bytes: usize,
}

impl CollectivePolicy {
    /// `auto`'s crossover: about `2 x hop x bandwidth` = 2 x 15 us x 24 GB/s = 720 KB (the plan's figure,
    /// `docs/CLUSTER_N_NODE_PLAN_2026-10-11.md`; the per-step cost measured at R = 2 is lower and the one-shot did not beat the ring
    /// there, `docs/NEMOTRON_TP.md` section 10.6: a starting value, not a tuned one).
    pub const AUTO_MAX_BYTES: usize = 700 * 1024;
    /// `oneshot`'s limit: MoE `y` at 256 rows is 23,068,672 bytes, the largest collective of a prefill slice.
    pub const ONESHOT_MAX_BYTES: usize = 24 << 20;
    pub const RING: CollectivePolicy = CollectivePolicy { mode: CollectiveMode::Ring, max_bytes: 0 };

    /// Parse `REXL3_TP_COLLECTIVE` and `REXL3_TP_ONESHOT_MAX_BYTES` from a variable lookup. An unknown mode or a malformed or absurd
    /// limit is an error naming the variable (a typo must not select the ring silently on one rank).
    pub fn from_vars(get: impl Fn(&str) -> Option<String>) -> Result<CollectivePolicy, String> {
        let mode = match get(COLLECTIVE_ENV).as_deref().map(str::trim) {
            None | Some("") | Some("ring") => CollectiveMode::Ring,
            Some("oneshot") => CollectiveMode::OneShot,
            Some("auto") => CollectiveMode::Auto,
            Some(other) => return Err(format!("{COLLECTIVE_ENV} must be ring, oneshot or auto, not {other:?}")),
        };
        let default = match mode {
            CollectiveMode::Ring => 0,
            CollectiveMode::OneShot => Self::ONESHOT_MAX_BYTES,
            CollectiveMode::Auto => Self::AUTO_MAX_BYTES,
        };
        let max_bytes = match get(ONESHOT_MAX_ENV).as_deref().map(str::trim) {
            None | Some("") => default,
            Some(v) => {
                let n: usize = v.parse().map_err(|_| format!("{ONESHOT_MAX_ENV} must be a byte count, not {v:?}"))?;
                if !(256..=(1usize << 30)).contains(&n) {
                    return Err(format!("{ONESHOT_MAX_ENV} must be between 256 bytes and 1 GiB, not {n}"));
                }
                if mode == CollectiveMode::Ring {
                    0 // the ring carries everything: a limit is meaningless (and must not make two ranks' fingerprints differ)
                } else {
                    n
                }
            }
        };
        Ok(CollectivePolicy { mode, max_bytes })
    }

    /// The policy of this process's environment, parsed once (so the boot gate's fingerprint and the communicator agree even if the
    /// environment is changed later).
    pub fn process() -> Result<CollectivePolicy, String> {
        static POLICY: std::sync::OnceLock<Result<CollectivePolicy, String>> = std::sync::OnceLock::new();
        POLICY.get_or_init(|| Self::from_vars(|k| std::env::var(k).ok())).clone()
    }

    /// Whether a collective of `bytes` bytes on a group of `ranks` goes by the one-shot exchange. The ring for one rank (nothing to
    /// do), for a message above the limit and for every message in `Ring` mode.
    pub fn use_oneshot(&self, ranks: usize, bytes: usize) -> bool {
        ranks >= 2 && bytes > 0 && bytes <= self.max_bytes
    }

    /// The text the boot gate compares: `ring`, `auto:max=716800`, `oneshot:max=25165824`.
    pub fn fingerprint(&self) -> String {
        match self.mode {
            CollectiveMode::Ring => "ring".into(),
            CollectiveMode::OneShot => format!("oneshot:max={}", self.max_bytes),
            CollectiveMode::Auto => format!("auto:max={}", self.max_bytes),
        }
    }
}

/// The collective policy of this process as text, for the boot gate (`BootGate::build`): both ranks must run the same one, or a ring on
/// one side would wait for a ring the other never issues. An invalid setting reads `invalid(...)`, which never equals a valid one, and
/// `NcclTp::init` refuses it.
pub fn collective_fingerprint() -> String {
    let base = match CollectivePolicy::process() {
        Ok(p) => p.fingerprint(),
        Err(e) => format!("invalid({e})"),
    };
    // with REXL3_TP_TRANSPORT=roce, the RoCE transport's size->path rule too (`tp_roce::process_fingerprint`); NCCL alone: unchanged
    match super::tp_roce::process_fingerprint() {
        None => base,
        Some(r) => format!("{base}+{r}"),
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// NCCL

/// The collective over NCCL between processes (nodes): a thin wrapper that gives [`NcclComm`] the [`TpComm`] face. `allreduce_sum_f32`
/// follows the [`CollectivePolicy`] (ring, one-shot, or by size); `allgather_sum_f32` is the one-shot whenever staging is reserved.
pub struct NcclTp {
    comm: NcclComm,
    topo: TpTopology,
    policy: CollectivePolicy,
    /// lane S: the second communicator of the shared-expert split (`TpComm::aux`), ring-only, no staging
    aux: Option<Box<NcclTp>>,
}

impl NcclTp {
    /// A ring-only wrapper (no staging, nothing read from the environment).
    pub fn new(comm: NcclComm) -> NcclTp {
        let topo = TpTopology { rank: comm.rank(), ranks: comm.ranks() };
        NcclTp { comm, topo, policy: CollectivePolicy::RING, aux: None }
    }

    /// `comm` under `policy`: reserves the one-shot staging (`(R - 1) x policy.max_bytes`, on the current device) when the policy uses
    /// it. An allocation failure is an error here, at boot, never a per-rank fallback.
    pub fn with_policy(comm: NcclComm, policy: CollectivePolicy) -> Result<NcclTp, CudaError> {
        let topo = TpTopology { rank: comm.rank(), ranks: comm.ranks() };
        if policy.max_bytes > 0 && topo.ranks >= 2 {
            comm.reserve_oneshot_staging(policy.max_bytes)?;
            // NCCL connects the point-to-point transport lazily, on the first send/recv, and separately from the ring's: connect it
            // here, eagerly (every rank runs this under the same fingerprint-gated policy), so that the first one-shot of the engine
            // is never inside a graph capture. 16 elements, discarded.
            let st = Stream::new()?;
            let mut scratch = DevBuf::zeroed(16 * 4)?;
            comm.oneshot_sum_f32(&mut scratch, 0, 16, &st)?;
            st.sync()?;
        }
        Ok(NcclTp { comm, topo, policy, aux: None })
    }

    /// Join the group `id` names (rank 0 made it with [`NcclComm::new_unique_id`]). Blocks until every rank has joined. The collective
    /// policy is the process's (`REXL3_TP_COLLECTIVE`, [`CollectivePolicy::process`]); it is logged with the join.
    pub fn init(topo: TpTopology, id: &NcclUniqueId) -> Result<NcclTp, CudaError> {
        let policy = CollectivePolicy::process().map_err(|e| {
            eprintln!("tp: {e}");
            CudaError(nccl::ERR_BAD_ARGUMENT)
        })?;
        let comm = NcclComm::init(topo.rank, topo.ranks, id)?;
        let tp = NcclTp::with_policy(comm, policy)?;
        eprintln!(
            "tp: rank {} of {} collective={} staging_bytes={}",
            topo.rank,
            topo.ranks,
            policy.fingerprint(),
            NcclComm::oneshot_staging_bytes(topo.ranks, policy.max_bytes)
        );
        Ok(tp)
    }

    pub fn policy(&self) -> CollectivePolicy {
        self.policy
    }

    /// The ring allreduce whatever the policy (benchmarks and the A/B).
    pub fn ring_sum_f32(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
        self.comm.allreduce_sum_f32(buf, off_bytes, count, st)
    }

    /// The whole rendezvous of `cfg` (two ranks): rank 0 listens on the port of `REXL3_TP_PEER`, accepts the follower (for at most
    /// `connect_wait`), sends the 128-byte unique id and joins; rank 1 connects (retrying for `connect_wait`), receives the id and
    /// joins. Returns the communicator AND the connected socket, positioned after the id, for the lockstep channel (`TP-R`'s) to
    /// carry its commands on. Each side fails with a message instead of hanging when the other never comes.
    pub fn rendezvous(cfg: &TpConfig, connect_wait: Duration) -> Result<(NcclTp, std::net::TcpStream), String> {
        use std::io::{Read, Write};
        use std::net::{TcpListener, TcpStream};
        if cfg.is_single() {
            return Err("rendezvous needs two ranks".into());
        }
        let peer = cfg.peer.as_deref().ok_or("no REXL3_TP_PEER")?;
        let topo = cfg.topology();
        // the latency of this collective depends on NCCL_* (channel and protocol caps: up to 10x, docs/NEMOTRON_TP.md section 10): record
        // the setting the group comes up under, so that a log line explains a slow step
        eprintln!(
            "tp: rank {} of {} joining via {peer}; libnccl {}; env: {}",
            topo.rank,
            topo.ranks,
            NcclComm::library_path(),
            nccl::nccl_environment().join(" ")
        );
        if topo.rank == 0 {
            let port = peer.rsplit(':').next().unwrap_or("");
            let l = TcpListener::bind(format!("0.0.0.0:{port}")).map_err(|e| format!("bind 0.0.0.0:{port}: {e}"))?;
            let id = NcclComm::new_unique_id().map_err(|e| format!("ncclGetUniqueId: {e}"))?;
            // wait for the follower no longer than `connect_wait` (a leader must not hang in boot when the other node never starts)
            l.set_nonblocking(true).map_err(|e| format!("listener: {e}"))?;
            let t0 = Instant::now();
            let (mut s, _) = loop {
                match l.accept() {
                    Ok(x) => break x,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && t0.elapsed() < connect_wait => {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        return Err(format!("rank 1 did not connect to {peer} within {} s", connect_wait.as_secs()));
                    }
                    Err(e) => return Err(format!("accept: {e}")),
                }
            };
            s.set_nonblocking(false).map_err(|e| format!("socket: {e}"))?;
            s.set_nodelay(true).ok();
            s.write_all(id.as_bytes()).map_err(|e| format!("send the unique id: {e}"))?;
            let comm = NcclTp::init(topo, &id).map_err(|e| format!("ncclCommInitRank: {e}"))?;
            Ok((comm, s))
        } else {
            let t0 = Instant::now();
            let mut s = loop {
                match TcpStream::connect(peer) {
                    Ok(s) => break s,
                    Err(e) if t0.elapsed() < connect_wait => {
                        let _ = e;
                        std::thread::sleep(Duration::from_millis(500));
                    }
                    Err(e) => return Err(format!("could not reach rank 0 at {peer}: {e}")),
                }
            };
            s.set_nodelay(true).ok();
            let mut b = [0u8; NcclUniqueId::BYTES];
            s.read_exact(&mut b).map_err(|e| format!("receive the unique id: {e}"))?;
            let comm = NcclTp::init(topo, &NcclUniqueId(b)).map_err(|e| format!("ncclCommInitRank: {e}"))?;
            Ok((comm, s))
        }
    }

    pub fn comm(&self) -> &NcclComm {
        &self.comm
    }

    /// Lane S: this group with `aux` as its second communicator ([`TpComm::aux`]).
    pub fn with_aux(mut self, aux: NcclTp) -> NcclTp {
        self.aux = Some(Box::new(aux));
        self
    }
}

impl TpComm for NcclTp {
    fn topology(&self) -> TpTopology {
        self.topo
    }

    /// The policy's choice for this message: the one-shot exchange or NCCL's ring allreduce. A function of the policy and `count`
    /// only, so every rank issues the same form.
    fn allreduce_sum_f32(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
        if self.policy.use_oneshot(self.topo.ranks, count.saturating_mul(4)) {
            self.comm.oneshot_sum_f32(buf, off_bytes, count, st)
        } else {
            self.comm.allreduce_sum_f32(buf, off_bytes, count, st)
        }
    }

    /// The one-shot exchange when staging is reserved for this message (whatever the mode), else the ring. The staging is a fixed
    /// fact of the group (the policy's limit), so the choice is the same on every rank.
    fn supports_swap(&self) -> bool {
        self.topo.ranks == 2
    }

    /// NCCL's collectives (the ring and the one-shot exchange) are stream-ordered and capturable (`docs/NEMOTRON_TP.md` 4, 10.2).
    fn capturable(&self) -> bool {
        true
    }

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
        self.comm.sendrecv_bytes(send, send_off, send_bytes, recv, recv_off, recv_bytes, st)
    }

    fn allgather_sum_f32(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
        if count.saturating_mul(4) <= self.comm.oneshot_capacity_bytes() {
            self.comm.oneshot_sum_f32(buf, off_bytes, count, st)
        } else {
            self.comm.allreduce_sum_f32(buf, off_bytes, count, st)
        }
    }

    /// `ncclCommAbort` (see [`NcclComm::abort`]): the kernel a stuck collective left on the stream exits and the stream completes.
    fn abort(&self) {
        self.comm.abort();
        if let Some(a) = &self.aux {
            a.comm.abort();
        }
    }

    fn aux(&self) -> Option<&dyn TpComm> {
        self.aux.as_deref().map(|a| a as &dyn TpComm)
    }

    /// `ncclCommGetAsyncError` ([`NcclComm::async_error`]), named.
    fn async_error(&self) -> Option<String> {
        self.comm.async_error().map(|code| {
            format!(
                "NCCL reports an asynchronous error on rank {}'s communicator: {} (code {code}; the collective fabric failed under a \
                 collective, the command link may still be up)",
                self.topo.rank,
                nccl::error_name(code).unwrap_or("unknown NCCL result")
            )
        })
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the host barrier of the loopback group

/// Why a [`HostBarrier::wait`] did not complete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierError {
    /// A participant gave up (it failed, panicked, or timed out): nobody waits for it any more.
    Broken,
    /// This participant waited the whole timeout for the others.
    TimedOut,
}

struct BarrierState {
    arrived: usize,
    generation: u64,
    broken: bool,
}

/// A reusable barrier for `n` host threads that can be broken: a thread that fails calls [`break_now`](Self::break_now) and every
/// waiter (now and later) returns [`BarrierError::Broken`] instead of waiting for it, and a wait has a timeout. `std::sync::Barrier` has
/// neither, and a loopback test that hangs holds the GPU lock.
pub struct HostBarrier {
    n: usize,
    state: Mutex<BarrierState>,
    cv: Condvar,
}

impl HostBarrier {
    pub fn new(n: usize) -> HostBarrier {
        assert!(n >= 1);
        HostBarrier { n, state: Mutex::new(BarrierState { arrived: 0, generation: 0, broken: false }), cv: Condvar::new() }
    }

    fn lock(&self) -> MutexGuard<'_, BarrierState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Block until all `n` threads have called `wait` for this round. A round that completed is `Ok` for everyone even if the barrier
    /// is broken right after.
    pub fn wait(&self, timeout: Duration) -> Result<(), BarrierError> {
        let deadline = Instant::now() + timeout;
        let mut st = self.lock();
        if st.broken {
            return Err(BarrierError::Broken);
        }
        let generation = st.generation;
        st.arrived += 1;
        if st.arrived == self.n {
            st.arrived = 0;
            st.generation = st.generation.wrapping_add(1);
            self.cv.notify_all();
            return Ok(());
        }
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let (g, res) = self.cv.wait_timeout(st, left).unwrap_or_else(|e| e.into_inner());
            st = g;
            if st.generation != generation {
                return Ok(());
            }
            if st.broken {
                return Err(BarrierError::Broken);
            }
            if res.timed_out() || left.is_zero() {
                st.broken = true;
                self.cv.notify_all();
                return Err(BarrierError::TimedOut);
            }
        }
    }

    /// Release every present and future waiter with [`BarrierError::Broken`].
    pub fn break_now(&self) {
        self.lock().broken = true;
        self.cv.notify_all();
    }

    pub fn is_broken(&self) -> bool {
        self.lock().broken
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the in-process loopback group

/// The shared exchange buffers: one slot per rank, each large enough for the largest collective seen so far.
struct Slots {
    cap: usize,
    bufs: Vec<Option<DevBuf>>,
    /// The element count each rank published in the current round (a mismatch between ranks is a caller bug).
    counts: Vec<usize>,
}

struct LoopShared {
    ranks: usize,
    barrier: HostBarrier,
    slots: Mutex<Slots>,
    timeout: Duration,
    policy: CollectivePolicy,
    /// An injected asynchronous error (`CudaError` code; 0: none), what [`LoopbackComm::inject_async_error`] sets: the test double
    /// of NCCL's `ncclCommGetAsyncError` for the watchdog's poll.
    async_error: std::sync::atomic::AtomicI32,
}

/// R ranks of a TP group inside one process, sharing one GPU, each used from its OWN host thread with its OWN stream and its OWN
/// buffers. Semantics are those of the NCCL collective (an in-place f32 sum of `count` elements at a byte offset), so a model built
/// against [`TpComm`] runs unchanged on one GPU. NOT capturable into a CUDA graph (it synchronises the stream and the host).
///
/// One collective is the one-shot algorithm: (a) wait for the stream, (b) host barrier (every rank idle: the exchange slots may be
/// reallocated), (c) copy the rank's range into its slot and wait for the copy, (d) host barrier (every slot is written), (e) queue the
/// sum of the R slots in FIXED RANK ORDER over the range (`nccl::sum_slots_f32`, the kernel `NcclTp`'s one-shot ends with). With
/// disjoint support (exact `-0.0` outside a rank's own elements) the sum is the owner's element in any order, bit for bit; on dense data
/// every rank computes the same bits because the order is fixed. The next collective's (a) waits for that sum on this rank, and its (b)
/// for the peers', so a slot is never rewritten while a peer still reads it.
///
/// At two ranks under the `ring` policy the reduction is `buf += peer slot` (`a + b`, the same bits as the kernel's `slot0 + slot1`,
/// and the code the two-rank gates have always run); `oneshot` (and `auto` within its limit) runs the kernel at R = 2 too, so the
/// digest golden can be checked under it on one GPU.
pub struct LoopbackGroup;

impl LoopbackGroup {
    /// The `ranks` comms of a group (rank 0 first), under the process's collective policy. Hand each to its own thread. A rank that
    /// waits more than 300 s for a peer fails. Panics if `ranks` is 0 or above [`nccl::MAX_ONESHOT_RANKS`].
    #[allow(clippy::new_ret_no_self)]
    pub fn new(ranks: usize) -> Vec<LoopbackComm> {
        Self::with_timeout(ranks, Duration::from_secs(300))
    }

    /// [`new`](Self::new) with the peer wait limit set (tests use a short one so that a bug fails fast instead of hanging).
    pub fn with_timeout(ranks: usize, timeout: Duration) -> Vec<LoopbackComm> {
        let policy = CollectivePolicy::process().unwrap_or_else(|e| panic!("the loopback group: {e}"));
        Self::with_policy(ranks, timeout, policy)
    }

    /// [`with_timeout`](Self::with_timeout) under an explicit [`CollectivePolicy`] (tests, instead of the environment).
    pub fn with_policy(ranks: usize, timeout: Duration, policy: CollectivePolicy) -> Vec<LoopbackComm> {
        assert!((1..=nccl::MAX_ONESHOT_RANKS).contains(&ranks), "a loopback group has 1 to {} ranks", nccl::MAX_ONESHOT_RANKS);
        let shared = Arc::new(LoopShared {
            ranks,
            barrier: HostBarrier::new(ranks),
            slots: Mutex::new(Slots { cap: 0, bufs: (0..ranks).map(|_| None).collect(), counts: vec![0; ranks] }),
            timeout,
            policy,
            async_error: std::sync::atomic::AtomicI32::new(0),
        });
        (0..ranks).map(|rank| LoopbackComm { rank, shared: shared.clone(), aux: None }).collect()
    }

    /// The two ranks (rank 0, rank 1): the two-rank convenience the pair-only tests and callers use.
    pub fn pair() -> (LoopbackComm, LoopbackComm) {
        Self::pair_with_timeout(Duration::from_secs(300))
    }

    /// [`pair`](Self::pair) with the peer wait limit set.
    pub fn pair_with_timeout(timeout: Duration) -> (LoopbackComm, LoopbackComm) {
        let mut v = Self::with_timeout(2, timeout).into_iter();
        (v.next().expect("rank 0"), v.next().expect("rank 1"))
    }

    /// Lane S: the two ranks, each with a second communicator (a second loopback pair) as [`TpComm::aux`], as a boot with the
    /// shared-expert split makes them (`tp_boot::make_comm`).
    pub fn pair_with_aux() -> (LoopbackComm, LoopbackComm) {
        let (mut a, mut b) = Self::pair();
        let (xa, xb) = Self::pair();
        a.aux = Some(Box::new(xa));
        b.aux = Some(Box::new(xb));
        (a, b)
    }
}

/// One rank of a [`LoopbackGroup`].
pub struct LoopbackComm {
    rank: usize,
    shared: Arc<LoopShared>,
    /// lane S: a second loopback group's comm of the same rank (`TpComm::aux`)
    aux: Option<Box<LoopbackComm>>,
}

fn peer_error(e: BarrierError) -> CudaError {
    eprintln!(
        "tp loopback: the peer rank {}",
        if e == BarrierError::TimedOut { "did not arrive in time" } else { "failed or exited" }
    );
    CudaError(nccl::ERR_PEER)
}

impl LoopbackComm {
    fn slots(&self) -> MutexGuard<'_, Slots> {
        self.shared.slots.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// One collective. `kernel`: reduce with the rank-ordered sum kernel; otherwise (two ranks only) `buf += peer slot`.
    fn run(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream, kernel: bool) -> Result<(), CudaError> {
        let bytes = count * 4;
        let me = self.rank;
        let ranks = self.shared.ranks;
        let sh = &*self.shared;
        // (a) this rank's inputs are complete and its previous sum has run
        st.sync()?;
        // (b) every rank is idle: grow the slots if this collective is larger than any before
        sh.barrier.wait(sh.timeout).map_err(peer_error)?;
        {
            let mut s = self.slots();
            if s.cap < bytes {
                let cap = bytes.max(s.cap.saturating_mul(2)).next_multiple_of(1 << 20);
                // the old slots are freed first (a cudaFree implies a device sync; every rank is idle) so that the peak is not 2x
                for b in s.bufs.iter_mut() {
                    *b = None;
                }
                for b in s.bufs.iter_mut() {
                    *b = Some(DevBuf::alloc(cap)?);
                }
                s.cap = cap;
            }
            s.counts[me] = count;
            let slot = s.bufs[me].as_mut().expect("slots are allocated");
            slot.copy_range_from(0, buf, off_bytes, bytes, st)?;
        }
        // (c) the copy is complete before a peer reads it
        st.sync()?;
        // (d) every slot is written
        sh.barrier.wait(sh.timeout).map_err(peer_error)?;
        {
            let s = self.slots();
            if let Some(r) = (0..ranks).find(|&r| s.counts[r] != count) {
                eprintln!("tp loopback: rank {me} reduces {count} elements, rank {r} {}", s.counts[r]);
                return Err(CudaError(nccl::ERR_BAD_ARGUMENT));
            }
            if kernel {
                // (e) buf = slot 0 + slot 1 + ... + slot R-1, queued on this rank's stream
                let ops: Vec<nccl::SumOperand> =
                    s.bufs.iter().map(|b| nccl::SumOperand::Buf(b.as_ref().expect("slots are allocated"), 0)).collect();
                nccl::sum_slots_f32(buf, off_bytes, &ops, count, st)?;
            } else {
                // (e) two ranks: buf = buf + peer's slot (`a + b`: the same bits on both ranks, since an IEEE add is commutative)
                let src = s.bufs[1 - me].as_ref().expect("slots are allocated");
                nccl::add_f32_into(buf, off_bytes, src, 0, count, st)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn swap(
        &self,
        send: &DevBuf,
        send_off: usize,
        send_bytes: usize,
        recv: &mut DevBuf,
        recv_off: usize,
        recv_bytes: usize,
        st: &Stream,
    ) -> Result<(), CudaError> {
        let (me, sh) = (self.rank, &*self.shared);
        if sh.ranks != 2 || send_off + send_bytes > send.bytes() || recv_off + recv_bytes > recv.bytes() {
            eprintln!("tp loopback: a swap of {send_bytes} / {recv_bytes} bytes does not fit (or the group is not two ranks)");
            return Err(CudaError(nccl::ERR_BAD_ARGUMENT));
        }
        st.sync()?;
        sh.barrier.wait(sh.timeout).map_err(peer_error)?;
        {
            let mut s = self.slots();
            let need = send_bytes.max(recv_bytes);
            if s.cap < need {
                let cap = need.max(s.cap.saturating_mul(2)).next_multiple_of(1 << 20);
                for b in s.bufs.iter_mut() {
                    *b = None;
                }
                for b in s.bufs.iter_mut() {
                    *b = Some(DevBuf::alloc(cap)?);
                }
                s.cap = cap;
            }
            s.counts[me] = send_bytes;
            if send_bytes > 0 {
                let slot = s.bufs[me].as_mut().expect("slots are allocated");
                slot.copy_range_from(0, send, send_off, send_bytes, st)?;
            }
        }
        st.sync()?;
        sh.barrier.wait(sh.timeout).map_err(peer_error)?;
        {
            let s = self.slots();
            if s.counts[1 - me] != recv_bytes {
                eprintln!("tp loopback: rank {me} expects {recv_bytes} bytes, the peer sent {}", s.counts[1 - me]);
                return Err(CudaError(nccl::ERR_BAD_ARGUMENT));
            }
            if recv_bytes > 0 {
                let src = s.bufs[1 - me].as_ref().expect("slots are allocated");
                recv.copy_range_from(recv_off, src, 0, recv_bytes, st)?;
            }
            // the peer's slot must not be rewritten (by its next collective) before this copy has run
            st.sync()?;
        }
        sh.barrier.wait(sh.timeout).map_err(peer_error)?;
        Ok(())
    }

    fn collective(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream, kernel: bool) -> Result<(), CudaError> {
        let end = count.checked_mul(4).and_then(|b| b.checked_add(off_bytes));
        if off_bytes % 4 != 0 || end.is_none_or(|e| e > buf.bytes()) {
            // a caller bug on THIS rank only: break the group so the peers do not wait for a collective that never comes
            eprintln!("tp loopback: {count} f32 at byte offset {off_bytes} does not fit a {}-byte buffer", buf.bytes());
            self.shared.barrier.break_now();
            return Err(CudaError(nccl::ERR_BAD_ARGUMENT));
        }
        if count == 0 || self.shared.ranks == 1 {
            return Ok(());
        }
        let r = self.run(buf, off_bytes, count, st, kernel);
        if r.is_err() {
            self.shared.barrier.break_now();
        }
        r
    }
}

impl TpComm for LoopbackComm {
    fn topology(&self) -> TpTopology {
        TpTopology { rank: self.rank, ranks: self.shared.ranks }
    }

    /// The policy's form for this message (ring means `buf += peer` at two ranks; at more ranks there is no ring here, so the
    /// kernel). Same decision on every rank: a function of the policy and `count`.
    fn allreduce_sum_f32(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
        let ranks = self.shared.ranks;
        let kernel = ranks != 2 || self.shared.policy.use_oneshot(ranks, count.saturating_mul(4));
        self.collective(buf, off_bytes, count, st, kernel)
    }

    /// The one-shot form (the rank-ordered sum kernel over every rank's slot).
    fn allgather_sum_f32(&self, buf: &mut DevBuf, off_bytes: usize, count: usize, st: &Stream) -> Result<(), CudaError> {
        self.collective(buf, off_bytes, count, st, true)
    }

    /// Breaks the group: every peer's collective (it waits on the host barrier) and every later one return `ERR_PEER`.
    fn abort(&self) {
        self.abort_group();
        if let Some(a) = &self.aux {
            a.abort_group();
        }
    }

    /// The injected error ([`LoopbackComm::inject_async_error`]), named as the NCCL group names a real one.
    fn async_error(&self) -> Option<String> {
        let code = self.shared.async_error.load(std::sync::atomic::Ordering::Acquire);
        (code != 0).then(|| {
            format!(
                "NCCL reports an asynchronous error on rank {}'s communicator: {} (code {code}; loopback injection)",
                self.rank,
                nccl::error_name(code).unwrap_or("unknown NCCL result")
            )
        })
    }

    fn supports_swap(&self) -> bool {
        self.shared.ranks == 2
    }

    fn aux(&self) -> Option<&dyn TpComm> {
        self.aux.as_deref().map(|a| a as &dyn TpComm)
    }

    /// Two ranks: publish this rank's bytes in its slot, then copy the peer's slot into `recv` (the same barrier protocol as `run`).
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
        let r = self.swap(send, send_off, send_bytes, recv, recv_off, recv_bytes, st);
        if r.is_err() {
            self.shared.barrier.break_now();
        }
        r
    }
}

impl LoopbackComm {
    fn abort_group(&self) {
        self.shared.barrier.break_now();
    }

    /// Makes every rank's [`TpComm::async_error`] report `code` (a `CudaError` code such as `nccl::NCCL_ERROR_BASE + 6`, ncclRemoteError):
    /// what a fabric failure under a collective looks like to the watchdog, without a fabric. The collectives themselves are untouched
    /// (a real one may hang until the abort; that is what the abort is for).
    pub fn inject_async_error(&self, code: i32) {
        self.shared.async_error.store(code, std::sync::atomic::Ordering::Release);
    }
}

impl Drop for LoopbackComm {
    fn drop(&mut self) {
        // a rank that dies mid-test (panic) must not leave its peers waiting
        if std::thread::panicking() {
            self.shared.barrier.break_now();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gb_cuda::device_available;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn config_defaults_to_one_rank_and_parses_a_pair() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
        };
        assert_eq!(TpConfig::from_vars(env(&[])).unwrap(), TpConfig::SINGLE);
        // a single rank ignores a stray peer
        assert_eq!(TpConfig::from_vars(env(&[("REXL3_TP_PEER", "x:1")])).unwrap(), TpConfig::SINGLE);
        let c =
            TpConfig::from_vars(env(&[("REXL3_TP_RANKS", "2"), ("REXL3_TP_RANK", "1"), ("REXL3_TP_PEER", "192.0.2.1:29500")]))
                .unwrap();
        assert_eq!(c.ranks, 2);
        assert_eq!(c.rank, 1);
        assert_eq!(c.peer.as_deref(), Some("192.0.2.1:29500"));
        // the timeouts default to the server's (`TpOptions`) and are read from its variable names
        assert_eq!(c.connect_timeout.as_secs(), TpOptions::DEFAULT_CONNECT_S);
        assert_eq!(c.peer_timeout.as_secs(), TpOptions::DEFAULT_PEER_S);
        assert_eq!(c.topology(), TpTopology { rank: 1, ranks: 2 });
        assert!(!c.is_single());
        let t = TpConfig::from_vars(env(&[
            ("REXL3_TP_RANKS", "2"),
            ("REXL3_TP_PEER", "h:1"),
            ("REXL3_TP_CONNECT_TIMEOUT_S", "60"),
            ("REXL3_TP_READY_TIMEOUT_S", "3000"),
            ("REXL3_TP_PEER_TIMEOUT_S", "20"),
        ]))
        .unwrap();
        assert_eq!((t.connect_timeout.as_secs(), t.ready_timeout.as_secs(), t.peer_timeout.as_secs()), (60, 3000, 20));
        // and it is the same parse the server uses
        let o =
            TpOptions::from_vars(env(&[("REXL3_TP_RANKS", "2"), ("REXL3_TP_RANK", "1"), ("REXL3_TP_PEER", "192.0.2.1:29500")]))
                .unwrap()
                .unwrap();
        assert_eq!(TpConfig::from(&o), c);
    }

    #[test]
    fn config_refuses_what_cannot_work() {
        let bad = |pairs: &'static [(&'static str, &'static str)]| {
            TpConfig::from_vars(move |k| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())).unwrap_err()
        };
        assert!(bad(&[("REXL3_TP_RANKS", "0")]).contains("--tp-ranks"));
        assert!(bad(&[("REXL3_TP_RANKS", "3"), ("REXL3_TP_RANK", "3"), ("REXL3_TP_PEER", "h:1")]).contains("--tp-rank"));
        assert!(bad(&[("REXL3_TP_RANKS", "two")]).contains("REXL3_TP_RANKS"));
        assert!(bad(&[("REXL3_TP_RANKS", "2"), ("REXL3_TP_RANK", "2"), ("REXL3_TP_PEER", "h:1")]).contains("--tp-rank"));
        assert!(bad(&[("REXL3_TP_RANKS", "2")]).contains("--tp-peer"));
        assert!(bad(&[("REXL3_TP_RANKS", "2"), ("REXL3_TP_PEER", "no-port")]).contains("HOST:PORT"));
        assert!(bad(&[("REXL3_TP_RANKS", "2"), ("REXL3_TP_PEER", "h:99999")]).contains("port"));
        assert!(bad(&[("REXL3_TP_RANKS", "2"), ("REXL3_TP_PEER", "h:1"), ("REXL3_TP_PEER_TIMEOUT_S", "0")]).contains("timeout"));
        assert!(bad(&[("REXL3_TP_RANK", "1")]).contains("--tp-rank"));
    }

    #[test]
    fn the_host_barrier_keeps_two_threads_in_lockstep() {
        // thread A bumps `a` before the barrier and thread B checks it after: B must always see exactly round + 1 bumps, and vice versa
        let bar = Arc::new(HostBarrier::new(2));
        let (a, b) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let rounds = 2000;
        let t = {
            let (bar, a, b) = (bar.clone(), a.clone(), b.clone());
            std::thread::spawn(move || {
                for r in 0..rounds {
                    a.fetch_add(1, Ordering::SeqCst);
                    bar.wait(Duration::from_secs(10)).unwrap();
                    assert_eq!(b.load(Ordering::SeqCst), r + 1, "round {r}: the peer's bump is visible after the barrier");
                    bar.wait(Duration::from_secs(10)).unwrap();
                }
            })
        };
        for r in 0..rounds {
            b.fetch_add(1, Ordering::SeqCst);
            bar.wait(Duration::from_secs(10)).unwrap();
            assert_eq!(a.load(Ordering::SeqCst), r + 1);
            bar.wait(Duration::from_secs(10)).unwrap();
        }
        t.join().unwrap();
        assert!(!bar.is_broken());
    }

    #[test]
    fn a_broken_or_late_peer_releases_the_waiter_instead_of_hanging() {
        let bar = Arc::new(HostBarrier::new(2));
        // nobody else arrives: the wait ends at the timeout and the barrier is broken for good
        let t0 = Instant::now();
        assert_eq!(bar.wait(Duration::from_millis(50)), Err(BarrierError::TimedOut));
        assert!(t0.elapsed() < Duration::from_secs(5));
        assert!(bar.is_broken());
        assert_eq!(bar.wait(Duration::from_secs(5)), Err(BarrierError::Broken));

        // the peer breaks it while this thread waits
        let bar = Arc::new(HostBarrier::new(2));
        let t = {
            let bar = bar.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                bar.break_now();
            })
        };
        assert_eq!(bar.wait(Duration::from_secs(30)), Err(BarrierError::Broken));
        t.join().unwrap();

        // a round that completed stays Ok for both even though the barrier is broken right after
        let bar = Arc::new(HostBarrier::new(2));
        let t = {
            let bar = bar.clone();
            std::thread::spawn(move || {
                let r = bar.wait(Duration::from_secs(10));
                bar.break_now();
                r
            })
        };
        let mine = bar.wait(Duration::from_secs(10));
        assert_eq!(mine, Ok(()));
        assert_eq!(t.join().unwrap(), Ok(()));
    }

    #[test]
    fn a_follower_that_cannot_reach_the_leader_gives_up_with_a_message() {
        // port 1 on loopback: nothing listens there
        let cfg = TpConfig { ranks: 2, rank: 1, peer: Some("127.0.0.1:1".into()), ..TpConfig::SINGLE };
        let t0 = Instant::now();
        let err = NcclTp::rendezvous(&cfg, Duration::from_millis(300)).err().expect("no leader, no group");
        assert!(err.contains("could not reach rank 0"), "{err}");
        assert!(t0.elapsed() < Duration::from_secs(10));
        assert!(NcclTp::rendezvous(&TpConfig::SINGLE, Duration::from_millis(1)).is_err());
    }

    #[test]
    fn the_loopback_group_reports_its_topology() {
        let (a, b) = LoopbackGroup::pair();
        assert_eq!(a.topology(), TpTopology { rank: 0, ranks: 2 });
        assert_eq!(b.topology(), TpTopology { rank: 1, ranks: 2 });
        // and at any size: rank r of R, in rank order
        for r in [1usize, 3, 4, 8, nccl::MAX_ONESHOT_RANKS] {
            let g = LoopbackGroup::with_policy(r, Duration::from_secs(1), CollectivePolicy::RING);
            assert_eq!(g.len(), r);
            for (i, c) in g.iter().enumerate() {
                assert_eq!(c.topology(), TpTopology { rank: i, ranks: r });
            }
        }
    }

    #[test]
    fn the_host_barrier_keeps_r_threads_in_lockstep() {
        // thread t bumps its own counter before the barrier; after it every thread must see all R counters at round + 1, and nobody
        // may have bumped round + 2 yet (the second barrier of the round holds them back)
        for ranks in [3usize, 4, 8] {
            let bar = Arc::new(HostBarrier::new(ranks));
            let counters: Arc<Vec<AtomicUsize>> = Arc::new((0..ranks).map(|_| AtomicUsize::new(0)).collect());
            let rounds = 1000;
            let threads: Vec<_> = (0..ranks)
                .map(|t| {
                    let (bar, counters) = (bar.clone(), counters.clone());
                    std::thread::spawn(move || {
                        for r in 0..rounds {
                            counters[t].fetch_add(1, Ordering::SeqCst);
                            bar.wait(Duration::from_secs(20)).unwrap();
                            for c in counters.iter() {
                                assert_eq!(c.load(Ordering::SeqCst), r + 1, "ranks {ranks} round {r} thread {t}");
                            }
                            bar.wait(Duration::from_secs(20)).unwrap();
                        }
                    })
                })
                .collect();
            for t in threads {
                t.join().unwrap();
            }
            assert!(!bar.is_broken());
        }
    }

    #[test]
    fn a_missing_or_failed_rank_releases_every_waiter_at_any_group_size() {
        for ranks in [3usize, 4, 8] {
            // one rank never arrives: every other waiter times out (and the barrier is broken for good)
            let bar = Arc::new(HostBarrier::new(ranks));
            let waiters: Vec<_> = (0..ranks - 1)
                .map(|_| {
                    let bar = bar.clone();
                    std::thread::spawn(move || bar.wait(Duration::from_millis(100)))
                })
                .collect();
            for w in waiters {
                let r = w.join().unwrap();
                assert!(matches!(r, Err(BarrierError::TimedOut) | Err(BarrierError::Broken)), "ranks {ranks}: {r:?}");
            }
            assert!(bar.is_broken());
            assert_eq!(bar.wait(Duration::from_secs(5)), Err(BarrierError::Broken));

            // one rank breaks the barrier while the others wait: all of them are released at once, long before their own timeout
            let bar = Arc::new(HostBarrier::new(ranks));
            let t0 = Instant::now();
            let waiters: Vec<_> = (0..ranks - 1)
                .map(|_| {
                    let bar = bar.clone();
                    std::thread::spawn(move || bar.wait(Duration::from_secs(60)))
                })
                .collect();
            std::thread::sleep(Duration::from_millis(100));
            bar.break_now();
            for w in waiters {
                assert_eq!(w.join().unwrap(), Err(BarrierError::Broken), "ranks {ranks}");
            }
            assert!(t0.elapsed() < Duration::from_secs(20));
        }
    }

    // ---- the collective policy ----------------------------------------------------------------------------------------------------

    fn policy(pairs: &'static [(&'static str, &'static str)]) -> Result<CollectivePolicy, String> {
        CollectivePolicy::from_vars(move |k| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string()))
    }

    #[test]
    fn the_policy_defaults_to_the_ring_and_parses_the_three_modes() {
        assert_eq!(policy(&[]).unwrap(), CollectivePolicy::RING);
        assert_eq!(policy(&[(COLLECTIVE_ENV, "")]).unwrap(), CollectivePolicy::RING);
        assert_eq!(policy(&[(COLLECTIVE_ENV, "ring")]).unwrap().max_bytes, 0);
        let one = policy(&[(COLLECTIVE_ENV, "oneshot")]).unwrap();
        assert_eq!((one.mode, one.max_bytes), (CollectiveMode::OneShot, CollectivePolicy::ONESHOT_MAX_BYTES));
        let auto = policy(&[(COLLECTIVE_ENV, "auto")]).unwrap();
        assert_eq!((auto.mode, auto.max_bytes), (CollectiveMode::Auto, 700 * 1024));
        let lim = policy(&[(COLLECTIVE_ENV, "auto"), (ONESHOT_MAX_ENV, "262144")]).unwrap();
        assert_eq!(lim.max_bytes, 262_144);
        // a limit with the ring is meaningless and must not make the fingerprints of two ring ranks differ
        assert_eq!(policy(&[(ONESHOT_MAX_ENV, "4096")]).unwrap(), CollectivePolicy::RING);
        assert_eq!(policy(&[(ONESHOT_MAX_ENV, "4096")]).unwrap().fingerprint(), "ring");
        // typos are refused by name, not read as the ring
        assert!(policy(&[(COLLECTIVE_ENV, "one-shot")]).unwrap_err().contains(COLLECTIVE_ENV));
        assert!(policy(&[(COLLECTIVE_ENV, "oneshot"), (ONESHOT_MAX_ENV, "lots")]).unwrap_err().contains(ONESHOT_MAX_ENV));
        assert!(policy(&[(COLLECTIVE_ENV, "oneshot"), (ONESHOT_MAX_ENV, "0")]).unwrap_err().contains(ONESHOT_MAX_ENV));
    }

    #[test]
    fn the_decision_is_a_function_of_the_policy_and_the_size_only() {
        let auto = policy(&[(COLLECTIVE_ENV, "auto")]).unwrap();
        for ranks in [2usize, 3, 8] {
            assert!(auto.use_oneshot(ranks, 4));
            assert!(auto.use_oneshot(ranks, 90 * 1024)); // 1 row of MoE y
            assert!(auto.use_oneshot(ranks, 700 * 1024));
            assert!(!auto.use_oneshot(ranks, 700 * 1024 + 4)); // the logits at wide rows stay on the ring
            assert!(!auto.use_oneshot(ranks, 23_068_672));
            assert!(!CollectivePolicy::RING.use_oneshot(ranks, 4));
        }
        assert!(!auto.use_oneshot(1, 4), "one rank has nothing to exchange");
        assert!(!auto.use_oneshot(2, 0));
        let one = policy(&[(COLLECTIVE_ENV, "oneshot")]).unwrap();
        assert!(one.use_oneshot(2, 23_068_672), "MoE y at 256 rows fits the forced one-shot's staging");
        assert!(!one.use_oneshot(2, 25 << 20));
    }

    #[test]
    fn the_fingerprint_names_the_mode_and_the_limit() {
        assert_eq!(CollectivePolicy::RING.fingerprint(), "ring");
        assert_eq!(policy(&[(COLLECTIVE_ENV, "auto")]).unwrap().fingerprint(), "auto:max=716800");
        assert_eq!(policy(&[(COLLECTIVE_ENV, "oneshot")]).unwrap().fingerprint(), "oneshot:max=25165824");
        // two ranks with different settings have different fingerprints, which the boot gate refuses
        assert_ne!(
            policy(&[(COLLECTIVE_ENV, "auto")]).unwrap().fingerprint(),
            policy(&[(COLLECTIVE_ENV, "auto"), (ONESHOT_MAX_ENV, "262144")]).unwrap().fingerprint()
        );
        assert_ne!(CollectivePolicy::RING.fingerprint(), policy(&[(COLLECTIVE_ENV, "oneshot")]).unwrap().fingerprint());
        // the process's own is a string and never panics
        assert!(!collective_fingerprint().is_empty());
    }

    // ---- exactness of the fixed-order sum, on the host (no device) ----------------------------------------------------------------

    /// Owner of element `i` among `ranks`.
    fn owner_n(i: usize, ranks: usize) -> usize {
        (((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 33) % ranks as u64) as usize
    }

    /// `ranks` disjoint-support operands over `n` elements: operand r holds its own elements (subnormals and signed zeros among them)
    /// and the exact -0.0 filler elsewhere. Also the whole row (each element's owner's value).
    fn disjoint_operands(seed: u64, ranks: usize, n: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
        let whole = vector(seed, n);
        let ops = (0..ranks).map(|r| (0..n).map(|i| if owner_n(i, ranks) == r { whole[i] } else { -0.0 }).collect()).collect();
        (ops, whole)
    }

    fn bits_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// Heap's algorithm: every permutation of `0..n`.
    fn permutations(n: usize) -> Vec<Vec<usize>> {
        fn heap(k: usize, a: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if k == 1 {
                out.push(a.clone());
                return;
            }
            heap(k - 1, a, out);
            for i in 0..k - 1 {
                if k % 2 == 0 {
                    a.swap(i, k - 1);
                } else {
                    a.swap(0, k - 1);
                }
                heap(k - 1, a, out);
            }
        }
        let mut out = Vec::new();
        heap(n, &mut (0..n).collect(), &mut out);
        out
    }

    #[test]
    fn a_disjoint_support_sum_is_the_owners_element_in_every_association_order() {
        // every left-to-right order of R operands, R = 2..=6 (R! up to 720): the sum is the owner's element, bit for bit
        for ranks in 2..=6usize {
            let (ops, whole) = disjoint_operands(0xA11CE + ranks as u64, ranks, 2048);
            let perms = permutations(ranks);
            assert_eq!(perms.len(), (1..=ranks).product::<usize>());
            for p in &perms {
                let sum: Vec<f32> = (0..2048).map(|i| p.iter().skip(1).fold(ops[p[0]][i], |a, &r| a + ops[r][i])).collect();
                assert!(bits_eq(&sum, &whole), "ranks {ranks} order {p:?}");
            }
        }
        // random binary association trees over a random order of R = 8 operands (and R = 3..=7): (a + b) + (c + d) and so on
        fn tree(ops: &[Vec<f32>], items: &[usize], s: &mut u64) -> Vec<f32> {
            if items.len() == 1 {
                return ops[items[0]].clone();
            }
            let k = 1 + (splitmix(s) as usize) % (items.len() - 1);
            let (l, r) = (tree(ops, &items[..k], s), tree(ops, &items[k..], s));
            l.iter().zip(&r).map(|(a, b)| a + b).collect()
        }
        let mut s = 0xBEEF_u64;
        for ranks in [3usize, 4, 5, 6, 7, 8] {
            let (ops, whole) = disjoint_operands(0xB0B + ranks as u64, ranks, 1500);
            for _ in 0..200 {
                let mut order: Vec<usize> = (0..ranks).collect();
                for i in (1..ranks).rev() {
                    order.swap(i, (splitmix(&mut s) as usize) % (i + 1));
                }
                assert!(bits_eq(&tree(&ops, &order, &mut s), &whole), "ranks {ranks} tree over {order:?}");
            }
        }
    }

    #[test]
    fn the_fixed_rank_order_is_what_makes_dense_sums_agree_across_ranks() {
        // with no disjoint support floating-point addition is not associative, so two association orders differ somewhere: the fixed
        // rank order is a CONVENTION that every rank computes the same bits, not a property of the data
        let ranks = 5;
        let ops: Vec<Vec<f32>> = (0..ranks).map(|r| vector(77 + r as u64, 4096)).collect();
        let fwd: Vec<f32> = (0..4096).map(|i| (1..ranks).fold(ops[0][i], |a, r| a + ops[r][i])).collect();
        let rev: Vec<f32> = (0..4096).map(|i| (0..ranks - 1).rev().fold(ops[ranks - 1][i], |a, r| a + ops[r][i])).collect();
        assert!(!bits_eq(&fwd, &rev), "the test data must expose the non-associativity");
    }

    // ---- the GPU gate (skips without a device; in a stub build `device_available()` is false) -------------------------------------

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn vector(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let r = splitmix(&mut s);
                let sign = ((r >> 40) & 1) as u32;
                let mant = (r & 0x7F_FFFF) as u32;
                if (r >> 52) % 23 == 0 {
                    // an exact signed zero (every 23rd): its sum with the -0.0 filler must keep its sign
                    f32::from_bits(sign << 31)
                } else if (r >> 48) % 5 == 0 {
                    // a subnormal (every fifth element): x + -0.0 == x must survive it, so no flush-to-zero anywhere on the path
                    f32::from_bits(sign << 31 | (mant | 1))
                } else {
                    let exp = 127 - 8 + ((r >> 32) % 16) as u32;
                    f32::from_bits(sign << 31 | exp << 23 | mant)
                }
            })
            .collect()
    }

    fn owner(i: usize) -> usize {
        ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 63) as usize
    }

    #[test]
    fn two_loopback_threads_sum_disjoint_buffers_exactly() {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
        const MAX: usize = 1 << 20; // 4 MB of f32
        const ITERS: usize = 1000;
        let sizes = [1usize, 255, 4096, 16384, 65536, MAX];
        let a_full = Arc::new(vector(1, MAX + 4096));
        let b_full = Arc::new(vector(2, MAX + 4096));
        let (c0, c1) = LoopbackGroup::pair_with_timeout(Duration::from_secs(60));
        let worker = |comm: LoopbackComm, a_full: Arc<Vec<f32>>, b_full: Arc<Vec<f32>>| {
            std::thread::spawn(move || -> Result<(), String> {
                let rank = comm.topology().rank;
                let st = Stream::new().map_err(|e| e.to_string())?;
                // 4 MB + room for the offset and a canary tail
                let mut buf = DevBuf::zeroed((MAX + 64 + 16) * 4).map_err(|e| e.to_string())?;
                for it in 0..ITERS {
                    let count = sizes[it % sizes.len()];
                    let off_elems = [0usize, 16, 3][it % 3]; // 3 is a 12-byte offset: not 16-byte aligned, still exact
                    let start = (it * 977) % 4096;
                    let disjoint = it % 4 != 3; // every fourth round is dense: both ranks nonzero everywhere, still exactly a + b
                    let mine = |j: usize| -> f32 {
                        let g = start + j;
                        let v = if rank == 0 { a_full[g] } else { b_full[g] };
                        if disjoint && owner(g) != rank {
                            -0.0
                        } else {
                            v
                        }
                    };
                    let theirs = |j: usize| -> f32 {
                        let g = start + j;
                        let v = if rank == 0 { b_full[g] } else { a_full[g] };
                        if disjoint && owner(g) == rank {
                            -0.0
                        } else {
                            v
                        }
                    };
                    let mut host = vec![f32::from_bits(0x7FC0_1234); off_elems + count + 16];
                    for j in 0..count {
                        host[off_elems + j] = mine(j);
                    }
                    buf.upload(&host).map_err(|e| e.to_string())?;
                    comm.allreduce_sum_f32(&mut buf, off_elems * 4, count, &st)
                        .map_err(|e| format!("rank {rank} it {it}: {e}"))?;
                    st.sync().map_err(|e| e.to_string())?;
                    let mut got = vec![0f32; host.len()];
                    buf.download_range(0, &mut got).map_err(|e| e.to_string())?;
                    for j in 0..count {
                        let want = mine(j) + theirs(j);
                        if got[off_elems + j].to_bits() != want.to_bits() {
                            return Err(format!(
                                "rank {rank} it {it} count {count} off {off_elems}: element {j}: got {:#010x} want {:#010x}",
                                got[off_elems + j].to_bits(),
                                want.to_bits()
                            ));
                        }
                    }
                    let canary = 0x7FC0_1234u32;
                    if got[..off_elems].iter().chain(&got[off_elems + count..]).any(|v| v.to_bits() != canary) {
                        return Err(format!("rank {rank} it {it}: bytes outside the range changed"));
                    }
                }
                Ok(())
            })
        };
        let t0 = worker(c0, a_full.clone(), b_full.clone());
        let t1 = worker(c1, a_full, b_full);
        let (r0, r1) = (t0.join().expect("rank 0 panicked"), t1.join().expect("rank 1 panicked"));
        assert_eq!(r0, Ok(()), "rank 0");
        assert_eq!(r1, Ok(()), "rank 1");
    }

    /// R loopback threads on ONE GPU, `iters` rounds over sizes from 1 element to 4 MB at offsets 0, 16 and 3 elements: disjoint-support
    /// rounds (every rank's elements and the -0.0 filler) must give the owner's element on every rank, dense rounds the left-to-right
    /// sum in rank order, and nothing outside the range may change.
    fn n_rank_loopback_round_trip(ranks: usize, iters: usize, policy: CollectivePolicy, forced_gather: bool) {
        const MAX: usize = 1 << 20; // 4 MB of f32
        let sizes = [1usize, 255, 4096, 16384, 65536, MAX];
        let data: Arc<Vec<Vec<f32>>> = Arc::new((0..ranks).map(|r| vector(100 + r as u64, MAX + 4096)).collect());
        let comms = LoopbackGroup::with_policy(ranks, Duration::from_secs(120), policy);
        let threads: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                let data = data.clone();
                std::thread::spawn(move || -> Result<(), String> {
                    let rank = comm.topology().rank;
                    let st = Stream::new().map_err(|e| e.to_string())?;
                    let mut buf = DevBuf::zeroed((MAX + 64 + 16) * 4).map_err(|e| e.to_string())?;
                    for it in 0..iters {
                        let count = sizes[it % sizes.len()];
                        let off_elems = [0usize, 16, 3][it % 3]; // 3 is a 12-byte offset: not 16-byte aligned, still exact
                        let start = (it * 977) % 4096;
                        let disjoint = it % 4 != 3; // every fourth round is dense: every rank nonzero everywhere
                        let mine = |j: usize| -> f32 {
                            let g = start + j;
                            if disjoint && owner_n(g, ranks) != rank {
                                -0.0
                            } else {
                                data[rank][g]
                            }
                        };
                        let want = |j: usize| -> f32 {
                            let g = start + j;
                            if disjoint {
                                data[owner_n(g, ranks)][g]
                            } else {
                                (1..ranks).fold(data[0][g], |a, r| a + data[r][g])
                            }
                        };
                        let mut host = vec![f32::from_bits(0x7FC0_1234); off_elems + count + 16];
                        for j in 0..count {
                            host[off_elems + j] = mine(j);
                        }
                        buf.upload(&host).map_err(|e| e.to_string())?;
                        let r = if forced_gather {
                            comm.allgather_sum_f32(&mut buf, off_elems * 4, count, &st)
                        } else {
                            comm.allreduce_sum_f32(&mut buf, off_elems * 4, count, &st)
                        };
                        r.map_err(|e| format!("rank {rank} it {it}: {e}"))?;
                        st.sync().map_err(|e| e.to_string())?;
                        let mut got = vec![0f32; host.len()];
                        buf.download_range(0, &mut got).map_err(|e| e.to_string())?;
                        for j in 0..count {
                            let w = want(j);
                            if got[off_elems + j].to_bits() != w.to_bits() {
                                return Err(format!(
                                    "ranks {ranks} rank {rank} it {it} count {count} off {off_elems} disjoint {disjoint}: element {j}: got {:#010x} want {:#010x}",
                                    got[off_elems + j].to_bits(),
                                    w.to_bits()
                                ));
                            }
                        }
                        if got[..off_elems].iter().chain(&got[off_elems + count..]).any(|v| v.to_bits() != 0x7FC0_1234) {
                            return Err(format!("rank {rank} it {it}: bytes outside the range changed"));
                        }
                    }
                    Ok(())
                })
            })
            .collect();
        for (r, t) in threads.into_iter().enumerate() {
            assert_eq!(t.join().unwrap_or_else(|_| panic!("rank {r} panicked")), Ok(()), "ranks {ranks} rank {r}");
        }
    }

    #[test]
    fn r_loopback_threads_sum_disjoint_buffers_exactly_at_3_4_and_8_ranks() {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
        for ranks in [3usize, 4, 8] {
            n_rank_loopback_round_trip(ranks, 1000, CollectivePolicy::RING, false);
        }
    }

    #[test]
    fn the_one_shot_form_is_exact_at_two_ranks_and_under_every_policy() {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
        // ring (`buf += peer`), oneshot (the sum kernel) and auto (kernel up to its limit, `buf += peer` above): the same bits, and the
        // explicit one-shot method of the trait is the kernel whatever the mode is
        let one = CollectivePolicy { mode: CollectiveMode::OneShot, max_bytes: CollectivePolicy::ONESHOT_MAX_BYTES };
        let auto = CollectivePolicy { mode: CollectiveMode::Auto, max_bytes: 16 * 1024 };
        n_rank_loopback_round_trip(2, 300, CollectivePolicy::RING, false);
        n_rank_loopback_round_trip(2, 300, one, false);
        n_rank_loopback_round_trip(2, 300, auto, false);
        n_rank_loopback_round_trip(2, 300, CollectivePolicy::RING, true);
    }

    #[test]
    fn the_sum_kernel_gives_the_owners_element_in_every_operand_order() {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
        let st = Stream::new().unwrap();
        let mut s = 0xD00D_u64;
        for (ranks, count) in [(2usize, 8192usize), (3, 4099), (4, 8192), (4, 4099), (8, 8192), (8, 4099), (16, 1024)] {
            let (ops, whole) = disjoint_operands(0xC0FFEE + ranks as u64, ranks, count);
            let bufs: Vec<DevBuf> = ops.iter().map(|o| DevBuf::from_slice(o).unwrap()).collect();
            let perms: Vec<Vec<usize>> = if ranks <= 4 {
                permutations(ranks)
            } else {
                (0..100)
                    .map(|_| {
                        let mut p: Vec<usize> = (0..ranks).collect();
                        for i in (1..ranks).rev() {
                            p.swap(i, (splitmix(&mut s) as usize) % (i + 1));
                        }
                        p
                    })
                    .collect()
            };
            for p in &perms {
                let mut out = DevBuf::from_slice(&vec![f32::from_bits(0x7FC0_1234); count]).unwrap();
                let order: Vec<nccl::SumOperand> = p.iter().map(|&r| nccl::SumOperand::Buf(&bufs[r], 0)).collect();
                nccl::sum_slots_f32(&mut out, 0, &order, count, &st).unwrap();
                st.sync().unwrap();
                let got: Vec<f32> = out.to_vec().unwrap();
                assert!(bits_eq(&got, &whole), "ranks {ranks} count {count} order {p:?}: {:?}", first_diff(&got, &whole));
            }
        }
    }

    // ---- the two-process gate of `NcclTp` itself (env-gated: one process per node) ---------------------------------------------------

    fn sock_barrier(s: &mut std::net::TcpStream) {
        use std::io::{Read, Write};
        s.write_all(&[0x42]).expect("lockstep write");
        let mut b = [0u8; 1];
        s.read_exact(&mut b).expect("lockstep read");
        assert_eq!(b[0], 0x42);
    }

    /// This rank's input and the expected sum of one case: the disjoint filler is -0.0, so the sum is the owner's element bit for bit.
    fn pair_case(seed: u64, n: usize, rank: usize, disjoint: bool) -> (Vec<f32>, Vec<f32>) {
        let (a, b) = (vector(seed, n), vector(seed ^ 0xA5A5_A5A5, n));
        let (a, b): (Vec<f32>, Vec<f32>) = if disjoint {
            (
                (0..n).map(|i| if owner(i) == 0 { a[i] } else { -0.0 }).collect(),
                (0..n).map(|i| if owner(i) == 1 { b[i] } else { -0.0 }).collect(),
            )
        } else {
            (a, b)
        };
        let sum = (0..n).map(|i| a[i] + b[i]).collect();
        (if rank == 0 { a } else { b }, sum)
    }

    fn first_diff(got: &[f32], want: &[f32]) -> Option<String> {
        got.iter()
            .zip(want)
            .position(|(g, w)| g.to_bits() != w.to_bits())
            .map(|i| format!("element {i}: got {:#010x} want {:#010x}", got[i].to_bits(), want[i].to_bits()))
    }

    /// `NcclTp` across two processes, as TP-R's boot builds it: `TpConfig::from_env()` (`REXL3_TP_RANKS=2`, `REXL3_TP_RANK`,
    /// `REXL3_TP_PEER`), `NcclTp::rendezvous`, then everything through `Arc<dyn TpComm>`: (a) eager sums of disjoint (-0.0 filler,
    /// subnormals, signed zeros) and dense buffers are bit-equal to the host's `a + b`; (b) 50 allreduces captured into one CUDA graph
    /// on a non-blocking stream and replayed 300 times, then fresh seeds; (c) `abort()` frees a rank that is stuck in a collective its
    /// peer never joins (the stream completes instead of hanging), and every later collective fails with `ERR_ABORTED`.
    /// Unset, it prints `not run: ...` and passes.
    #[test]
    fn nccl_tp_pair_rendezvous_capture_replay_and_abort() {
        use gb_cuda::graph::Graph;
        let cfg = match TpConfig::from_env() {
            Ok(c) if !c.is_single() => c,
            Ok(_) => {
                eprintln!("not run: REXL3_TP_RANKS=2 (with REXL3_TP_RANK and REXL3_TP_PEER) is not set");
                return;
            }
            Err(e) => panic!("the tensor-parallel environment is invalid: {e}"),
        };
        if !device_available() {
            eprintln!("not run: no CUDA device");
            return;
        }
        let timeout: u64 = std::env::var("REXL3_TP_TEST_TIMEOUT_S").ok().and_then(|v| v.parse().ok()).unwrap_or(240);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(timeout));
            eprintln!("nccl_tp_pair: watchdog expired after {timeout} s, exiting");
            std::process::exit(3);
        });
        let rank = cfg.rank;
        gb_cuda::rt::init_context().unwrap();
        let t0 = Instant::now();
        let (tp, mut sock) = NcclTp::rendezvous(&cfg, cfg.connect_timeout.min(Duration::from_secs(120))).expect("rendezvous");
        eprintln!("nccl_tp_pair rank {rank}: rendezvous + communicator in {:.2} s", t0.elapsed().as_secs_f64());
        let comm: Arc<dyn TpComm> = Arc::new(tp);
        assert_eq!(comm.topology(), cfg.topology());
        let st = Stream::new_nonblocking().unwrap();
        sock_barrier(&mut sock);

        // (a) eager
        let mut cases = 0;
        for (ci, &count) in [1usize, 1021, 4096, 22528, 112_640, 524_288].iter().enumerate() {
            for &off in &[0usize, 64] {
                for &disjoint in &[true, false] {
                    let (own, want) =
                        pair_case(0x5000 + ci as u64 * 8 + (off as u64 / 64) * 2 + disjoint as u64, count, rank, disjoint);
                    let pad = off / 4;
                    let mut host = vec![f32::from_bits(0x7FC0_1234); pad + count + 8];
                    host[pad..pad + count].copy_from_slice(&own);
                    let mut buf = DevBuf::from_slice(&host).unwrap();
                    gb_cuda::rt::device_sync().unwrap();
                    comm.allreduce_sum_f32(&mut buf, off, count, &st).unwrap();
                    st.sync().unwrap();
                    let got: Vec<f32> = buf.to_vec().unwrap();
                    if let Some(m) = first_diff(&got[pad..pad + count], &want) {
                        panic!("rank {rank}: count {count} off {off} disjoint {disjoint}: {m}");
                    }
                    assert!(
                        got[..pad].iter().chain(&got[pad + count..]).all(|v| v.to_bits() == 0x7FC0_1234),
                        "the allreduce wrote outside its range"
                    );
                    cases += 1;
                }
            }
        }
        eprintln!("nccl_tp_pair rank {rank}: (a) {cases} eager cases bit-equal to the host sum, through Arc<dyn TpComm>");
        sock_barrier(&mut sock);

        // (b) capture and replay
        const N: usize = 4096;
        const INNER: usize = 50;
        let scale = 2.0f32.powi(INNER as i32 - 1);
        let (seed_host, c0) = pair_case(0xBEEF, N, rank, true);
        let mut seed = DevBuf::from_slice(&seed_host).unwrap();
        let mut work = DevBuf::zeroed(N * 4).unwrap();
        gb_cuda::rt::device_sync().unwrap();
        work.copy_from(&seed, N * 4, &st).unwrap();
        comm.allreduce_sum_f32(&mut work, 0, N, &st).unwrap(); // the first collective connects the peers: eagerly, before the capture
        st.sync().unwrap();
        sock_barrier(&mut sock);
        let graph = {
            let cap = Graph::begin(&st).unwrap();
            let ran = (|| -> Result<(), CudaError> {
                work.copy_from(&seed, N * 4, &st)?;
                for _ in 0..INNER {
                    comm.allreduce_sum_f32(&mut work, 0, N, &st)?;
                }
                Ok(())
            })();
            match cap.finish(ran) {
                Ok(g) => g,
                Err(e) => panic!("rank {rank}: capturing {INNER} NcclTp allreduces failed: {e}"),
            }
        };
        sock_barrier(&mut sock);
        let t = Instant::now();
        for _ in 0..300 {
            graph.launch(&st).unwrap();
        }
        st.sync().unwrap();
        let per = t.elapsed().as_secs_f64() * 1e6 / (300 * INNER) as f64;
        let want: Vec<f32> = c0.iter().map(|c| c * scale).collect();
        let got: Vec<f32> = work.to_vec().unwrap();
        if let Some(m) = first_diff(&got, &want) {
            panic!("rank {rank}: after 300 replays: {m}");
        }
        for k in 0..10u64 {
            let (own, c) = pair_case(0xC0DE + k, N, rank, k % 2 == 0);
            seed.upload(&own).unwrap();
            gb_cuda::rt::device_sync().unwrap();
            graph.launch(&st).unwrap();
            st.sync().unwrap();
            let want: Vec<f32> = c.iter().map(|v| v * scale).collect();
            let got: Vec<f32> = work.to_vec().unwrap();
            if let Some(m) = first_diff(&got, &want) {
                panic!("rank {rank}: fresh seed {k}: {m}");
            }
            sock_barrier(&mut sock);
        }
        eprintln!(
            "nccl_tp_pair rank {rank}: (b) {INNER} allreduces captured, 300 replays + 10 fresh seeds equal the host's (a+b)*2^{} ({per:.1} us per allreduce)",
            INNER - 1
        );
        drop(graph); // before the communicator is aborted or dropped
        sock_barrier(&mut sock);

        // (c) abort frees a rank stuck in a collective its peer never joins
        if rank == 0 {
            let c2 = comm.clone();
            let aborter = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(2));
                c2.abort();
            });
            let mut buf = DevBuf::zeroed(1024 * 4).unwrap();
            gb_cuda::rt::device_sync().unwrap();
            let t = Instant::now();
            comm.allreduce_sum_f32(&mut buf, 0, 1024, &st).unwrap(); // rank 1 never issues the matching collective
            let synced = st.sync();
            let waited = t.elapsed();
            aborter.join().unwrap();
            eprintln!(
                "nccl_tp_pair rank 0: (c) the stuck collective was released by the abort after {:.1} s (stream sync: {synced:?})",
                waited.as_secs_f64()
            );
            assert!(waited < Duration::from_secs(60), "the abort did not release the stream");
        } else {
            std::thread::sleep(Duration::from_secs(5));
            comm.abort();
        }
        let mut buf = DevBuf::zeroed(1024 * 4).unwrap();
        assert_eq!(comm.allreduce_sum_f32(&mut buf, 0, 1024, &st), Err(CudaError(nccl::ERR_ABORTED)));
        comm.abort(); // idempotent
        eprintln!("nccl_tp_pair rank {rank}: PASS");
    }

    #[test]
    fn an_abort_releases_the_peer_that_waits_in_a_collective() {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
        let (c0, c1) = LoopbackGroup::pair_with_timeout(Duration::from_secs(60));
        let waiter = std::thread::spawn(move || {
            let st = Stream::new().unwrap();
            let mut buf = DevBuf::zeroed(4096 * 4).unwrap();
            let t0 = Instant::now();
            let r = c0.allreduce_sum_f32(&mut buf, 0, 1024, &st);
            (r, t0.elapsed())
        });
        // rank 1 never joins: it gives up on the group instead (what the lockstep runtime's watchdog does for a lost peer)
        std::thread::sleep(Duration::from_millis(300));
        c1.abort();
        let (r, waited) = waiter.join().unwrap();
        assert_eq!(r, Err(CudaError(nccl::ERR_PEER)));
        assert!(waited < Duration::from_secs(20), "the waiter was released by the abort, not by the 60 s timeout: {waited:?}");
        // and every later collective fails at once
        let st = Stream::new().unwrap();
        let mut buf = DevBuf::zeroed(4096 * 4).unwrap();
        assert_eq!(c1.allreduce_sum_f32(&mut buf, 0, 1024, &st), Err(CudaError(nccl::ERR_PEER)));
    }

    #[test]
    fn a_count_mismatch_between_loopback_ranks_fails_both_instead_of_hanging() {
        if !device_available() {
            eprintln!("no CUDA device: skipping");
            return;
        }
        let (c0, c1) = LoopbackGroup::pair_with_timeout(Duration::from_secs(20));
        let run = |comm: LoopbackComm, count: usize| {
            std::thread::spawn(move || {
                let st = Stream::new().unwrap();
                let mut buf = DevBuf::zeroed(4096 * 4).unwrap();
                comm.allreduce_sum_f32(&mut buf, 0, count, &st)
            })
        };
        let (t0, t1) = (run(c0, 100), run(c1, 200));
        let (r0, r1) = (t0.join().unwrap(), t1.join().unwrap());
        // the rank that reads the other's published count first fails with the argument error; the other is released as a failed peer or
        // fails the same check: neither returns Ok, and neither waited out the timeout
        assert!(r0.is_err() || r1.is_err());
        assert!(r0.is_err() && r1.is_err(), "r0 {r0:?} r1 {r1:?}");
    }
}
