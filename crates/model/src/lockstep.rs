//! The family-neutral lockstep runtime of tensor-parallel serving, for a group of ANY number of ranks `R >= 2`: a leader (rank 0)
//! that forwards every state-changing call of a family's [`LockstepRuntime`] to its `R - 1` followers, and the follower's loop that
//! replays them. No code in this file depends on a model family (the Nemotron family is the first user
//! `families::nemotron_h_omni::tp_runtime`, which is this module instantiated for its [`FamilyCall`] type), and a second family
//! adopts it by implementing [`LockstepRuntime`] and [`FamilyCall`] (`docs/CLUSTER_FAMILY_PORTING.md`). The design and the exactness
//! rule the whole scheme stands on are `docs/NEMOTRON_TP.md` (section 2 and 5; section 14 for the N-rank form).
//!
//! **The invariant is one sentence: every rank sees the same call stream in the same order.** Rank 0 runs the HTTP server, scheduler,
//! engine, sampling and arbiter exactly as the single-node engine does; its runtime is a [`TpLeader`], which for each state-changing call
//! (1) assigns the next sequence number, (2) SENDS the command to every follower, (3) applies it locally, (4) records the local result.
//! Sending first is what lets the followers reach the collectives the local call blocks on. A follower ([`run_follower`]) applies the
//! commands one at a time, in arrival order, on one thread (so its CUDA streams are ordered exactly as the leader's), and answers each
//! with an [`Msg::Ack`]: ok, or the error it hit, plus a digest of the candidates it computed. The leader compares each follower's ack
//! with its own result for the same sequence number (the `Ledger`, one cursor per follower); a disagreement (one rank failed, or the
//! replicated candidates differ) is fatal for the group. Candidates are replicated by the gathered logits, so rank 0's answer is the
//! answer; a follower drops its copy and only reports its digest.
//!
//! **What is core and what is the family's.** The seven calls every family has (`begin_slot`, `end_slot`, `prefill_chunk`,
//! `window_candidates`, `commit`, `flush`, and the read `vocab_size`) are forwarded by this module with message kinds of its own.
//! Everything else a family forwards (a draft head, snapshots, image splices, ...) is a [`FamilyCall`]: an opaque enum the family
//! encodes and applies; the module numbers it, sends it, compares the digest of its result and loses the group when a rank disagrees,
//! exactly as for a core call. The family's boot facts (what its ranks must agree on beyond the shared geometry) are the call type's
//! [`FamilyCall::Facts`] and travel in the Hello at the position the wire has always given them.
//!
//! **What a follower cannot recompute** and so receives: whatever the call carries (for the core calls: the scheduler's row batches
//! ([`RowBatch`], all seven fields), `want` and `count`, the slots and `keep` counts); a family adds its own host-visible inputs to its
//! [`FamilyCall`] bodies. Everything else a call computes must be a pure function of the call stream.
//!
//! **Failure model.** A rank that fails a call another completed, or a candidate digest that differs, is a lost group
//! ([`ModelError::StateLost`], `is_fatal`): every later call on the leader returns it, the in-flight requests fail, nothing waits
//! forever, and EVERY follower's link is closed at once (without a Shutdown) so that a healthy rank blocked in a collective with a dead
//! sibling aborts its communicator at once instead of waiting out its own silence watchdog. A lost peer (EOF, reset, a write that does
//! not drain within the timeout, no frame from THAT rank for the peer timeout, a Bye or an Abort) is the same, and the message names the
//! rank. A rank stuck INSIDE a collective cannot be unstuck from the host without the communicator's abort; the watchdog first runs the
//! optional abort hook and then, after `hang_exit`, aborts the process (the server's own progress watchdog is the backstop).
//!
//! **Cache calls are the one exception** ([`TpLeader::forward_cache_call`]). A prefix snapshot, restore or release is a local copy on
//! every rank with no collective in it, so a rank that answers differently is still in lockstep, and a recoverable cache failure must
//! cost a cold prefill, not the group. Its answers are COLLECTED per rank (never compared with rank 0's: ranks hold different shards),
//! the leader waits for them within [`CacheDeadlines`] before anything depends on the call, and the family decides with a
//! [`SnapshotBook`]: a restore is used only when every rank restored exactly the state it snapshotted (each rank's content digest at
//! restore equals its own at snapshot time); anything else is an error the engine answers with a cold prefill on every rank. A fatal
//! class still loses the group, and so does a follower that never answers a cache call ([`CacheDeadlines::stall`]).
//!
//! The transport is any pair of `Read` and `Write` halves ([`Link`]): TCP between the nodes, a socket pair in the tests.

use crate::error::ModelError;
use crate::forward::Candidates;
use crate::rows::RowBatch;
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// The protocol version byte of every frame. A peer that speaks another version is refused at the first frame. Version 2: the Hello
/// carries the sender's rank, the group size and the NCCL fingerprint (any number of ranks, not two); every command, ack, ping and
/// shutdown frame is byte-identical to version 1 except for this byte. Version 3: the Nemotron facts of the Hello gain the expert
/// ownership table's 64-bit fingerprint (0 = contiguous); no other frame changes but the version byte. Version 4: the Nemotron facts
/// gain the prefix cache's request, snapshot slots, bytes per slot and KV pages (the cache is carved on every rank of a group and the
/// ranks must agree on it before weights load: `docs/NEMOTRON_TP.md` 12.5); no other frame changes but the version byte. (The ack of
/// `EnableSnapshots` also carries the arena's bytes as its digest: the bytes of the frame are the same, its comparison is new.)
/// Version 5 (the rc2 assembly, 2026-10-15; one bump for three lanes): the Nemotron facts gain the prefill settings string (contract, slice
/// cap, absent-expert fast path; lane S) as their last field; the Hello's build string gains `commit=`, `exe_sha256=` and `tp_spin_us=`
/// lines and its NCCL string the NCCL environment hash and the libnccl sha256 (E2: `key=value` lines inside the existing length-prefixed
/// strings, no new field); the plan-flags byte gains bit 7 (int8 KV cache), bits 5 and 6 reserved (K8: `tp_arm::plan_flags`). No other frame
/// changes but the version byte (`testdata/nemotron_omni/tp_wire_v5.txt`; `docs/RC2_ASSEMBLY.md`).
pub const PROTOCOL_VERSION: u8 = 5;
/// The most bytes one frame may carry (a family call's body, such as the image features of a slice, is at most a few MiB).
pub const MAX_FRAME: usize = 64 << 20;
/// The first four bytes of a Hello body.
pub(crate) const MAGIC: [u8; 4] = *b"RXTP";
/// Frame header after the length: version, kind, sequence number.
pub(crate) const HEADER: usize = 1 + 1 + 8;
// ------------------------------------------------------------------------------------------------------------------- wire primitives

/// Why a frame could not be read, written or understood.
#[derive(Debug)]
pub enum WireError {
    Io(io::Error),
    /// The peer closed the connection (clean EOF at a frame boundary).
    Closed,
    /// The peer speaks another protocol version.
    Version {
        got: u8,
        want: u8,
    },
    Truncated(&'static str),
    TooLarge(usize),
    UnknownKind(u8),
    Invalid(String),
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireError::Io(e) => write!(f, "I/O error: {e}"),
            WireError::Closed => write!(f, "connection closed by the peer"),
            WireError::Version { got, want } => {
                write!(f, "the peer speaks protocol version {got}, this build speaks {want} (both ranks must run the same build)")
            }
            WireError::Truncated(what) => write!(f, "truncated frame (reading {what})"),
            WireError::TooLarge(n) => write!(f, "frame of {n} bytes exceeds the {MAX_FRAME}-byte limit"),
            WireError::UnknownKind(k) => write!(f, "unknown message kind {k}"),
            WireError::Invalid(m) => write!(f, "invalid frame: {m}"),
        }
    }
}

impl std::error::Error for WireError {}

impl From<io::Error> for WireError {
    fn from(e: io::Error) -> Self {
        WireError::Io(e)
    }
}

/// Little-endian encoder over a byte vector. These primitives ARE the wire format: a family must write its call bodies and facts with
/// them (and read them back with [`Dec`]) or the bytes drift from protocol 2.
#[derive(Default)]
pub struct Enc(pub Vec<u8>);

impl Enc {
    pub fn new() -> Enc {
        Enc(Vec::new())
    }
    pub fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    pub fn bool(&mut self, v: bool) {
        self.0.push(v as u8);
    }
    pub fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    pub fn bytes(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.0.extend_from_slice(v);
    }
    pub fn str(&mut self, v: &str) {
        self.bytes(v.as_bytes());
    }
    pub fn i32s(&mut self, v: &[i32]) {
        self.u64(v.len() as u64);
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
    pub fn u32s(&mut self, v: &[u32]) {
        self.u64(v.len() as u64);
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
    pub fn opt_i32s(&mut self, v: &Option<Vec<i32>>) {
        match v {
            Some(v) => {
                self.bool(true);
                self.i32s(v);
            }
            None => self.bool(false),
        }
    }
    pub fn opt_u32(&mut self, v: Option<u32>) {
        match v {
            Some(v) => {
                self.bool(true);
                self.u32(v);
            }
            None => self.bool(false),
        }
    }
}

/// Little-endian decoder over a frame body. Every length is checked against what is left before anything is allocated.
pub struct Dec<'a> {
    pub(crate) b: &'a [u8],
    pub(crate) at: usize,
}

impl<'a> Dec<'a> {
    pub fn new(b: &'a [u8]) -> Dec<'a> {
        Dec { b, at: 0 }
    }
    pub fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], WireError> {
        if self.b.len() - self.at < n {
            return Err(WireError::Truncated(what));
        }
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    pub fn u8(&mut self, what: &'static str) -> Result<u8, WireError> {
        Ok(self.take(1, what)?[0])
    }
    pub fn bool(&mut self, what: &'static str) -> Result<bool, WireError> {
        match self.u8(what)? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(WireError::Invalid(format!("{what}: {v} is not a boolean"))),
        }
    }
    pub fn u32(&mut self, what: &'static str) -> Result<u32, WireError> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().expect("4 bytes")))
    }
    pub fn u64(&mut self, what: &'static str) -> Result<u64, WireError> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().expect("8 bytes")))
    }
    pub fn f64(&mut self, what: &'static str) -> Result<f64, WireError> {
        Ok(f64::from_bits(self.u64(what)?))
    }
    pub fn len(&mut self, elem: usize, what: &'static str) -> Result<usize, WireError> {
        let n = self.u64(what)?;
        let n = usize::try_from(n).map_err(|_| WireError::Invalid(format!("{what}: length {n} overflows")))?;
        if n.checked_mul(elem).is_none_or(|bytes| bytes > self.b.len() - self.at) {
            return Err(WireError::Truncated(what));
        }
        Ok(n)
    }
    pub fn bytes(&mut self, what: &'static str) -> Result<Vec<u8>, WireError> {
        let n = self.len(1, what)?;
        Ok(self.take(n, what)?.to_vec())
    }
    pub fn str(&mut self, what: &'static str) -> Result<String, WireError> {
        String::from_utf8(self.bytes(what)?).map_err(|_| WireError::Invalid(format!("{what}: not UTF-8")))
    }
    pub fn i32s(&mut self, what: &'static str) -> Result<Vec<i32>, WireError> {
        let n = self.len(4, what)?;
        let raw = self.take(n * 4, what)?;
        Ok(raw.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().expect("4 bytes"))).collect())
    }
    pub fn u32s(&mut self, what: &'static str) -> Result<Vec<u32>, WireError> {
        let n = self.len(4, what)?;
        let raw = self.take(n * 4, what)?;
        Ok(raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().expect("4 bytes"))).collect())
    }
    pub fn opt_i32s(&mut self, what: &'static str) -> Result<Option<Vec<i32>>, WireError> {
        if self.bool(what)? {
            Ok(Some(self.i32s(what)?))
        } else {
            Ok(None)
        }
    }
    pub fn opt_u32(&mut self, what: &'static str) -> Result<Option<u32>, WireError> {
        if self.bool(what)? {
            Ok(Some(self.u32(what)?))
        } else {
            Ok(None)
        }
    }
    pub fn done(&self, what: &'static str) -> Result<(), WireError> {
        if self.at != self.b.len() {
            return Err(WireError::Invalid(format!("{what}: {} trailing bytes", self.b.len() - self.at)));
        }
        Ok(())
    }
}
// ------------------------------------------------------------------------------------------------------------------- messages

/// A result class: what a rank's call returned, in the terms two ranks can compare. `Ok(aux)` carries a digest of what the call
/// produced (the candidates of a prefill or window, whether a snapshot was taken) so that a divergence of the ranks surfaces at the call
/// that caused it; `Err` is the error's kind and whether it is fatal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Ok(u64),
    Err { kind: u8, fatal: bool },
}

impl Class {
    pub fn is_fatal(&self) -> bool {
        matches!(self, Class::Err { fatal: true, .. })
    }

    fn of_error(e: &ModelError) -> Class {
        Class::Err { kind: error_kind(e), fatal: e.is_fatal() }
    }

    fn of<T>(r: &Result<T, ModelError>, aux: impl FnOnce(&T) -> u64) -> Class {
        match r {
            Ok(v) => Class::Ok(aux(v)),
            Err(e) => Class::of_error(e),
        }
    }

    fn describe(&self) -> String {
        match self {
            Class::Ok(0) => "ok".into(),
            Class::Ok(aux) => format!("ok (digest {aux:016x})"),
            Class::Err { kind, fatal } => {
                format!("{} error{}", kind_name(*kind), if *fatal { " (fatal)" } else { "" })
            }
        }
    }
}

fn error_kind(e: &ModelError) -> u8 {
    match e {
        ModelError::Pack(_) => 1,
        ModelError::Cuda(_) => 2,
        ModelError::Config(_) => 3,
        ModelError::Request(_) => 4,
        ModelError::NotBuilt(_) => 5,
        ModelError::Busy(_) => 6,
        ModelError::StateLost(_) => 7,
    }
}

fn kind_name(k: u8) -> &'static str {
    match k {
        1 => "pack",
        2 => "cuda",
        3 => "configuration",
        4 => "request",
        5 => "not-built",
        6 => "busy",
        7 => "state-lost",
        _ => "unknown",
    }
}
/// What a family contributes to the wire: its commands and its boot facts. Implemented by the family's call enum.
///
/// **Kinds.** Every frame carries a kind byte. This module owns `1..=9` (session frames) and the core commands `16, 17, 18, 20, 21, 22,
/// 29` ([`CORE_KINDS`]); a family's calls take kinds in `16..=127` that are not core kinds, so a family's command is also a *command*
/// (numbered by the leader, acknowledged by the follower). The Nemotron family keeps `19` and `23..=28` (protocol 2 as shipped); a new
/// family should use `32..=127`. A kind that no one claims decodes to [`WireError::UnknownKind`].
pub trait FamilyCall: Clone + std::fmt::Debug + PartialEq + Send + 'static {
    /// What the family's ranks must agree on to run one model in lockstep, beyond the shared geometry of [`BootGate`]: environment
    /// switches that shape the call stream or the arithmetic, a fingerprint of a vocabulary file, whether a head is priced. Carried in
    /// the Hello and compared by [`FamilyCall::check_facts`]. `()` for a family with none.
    type Facts: Clone + std::fmt::Debug + PartialEq + Eq + Send + 'static;

    /// The frame kind of this call (`16..=127`, not a [`CORE_KINDS`] member).
    fn kind(&self) -> u8;
    /// The call's name for the group's error messages (`the ranks diverged at call 12 (mtp_draft)`).
    fn name(&self) -> &'static str;
    /// Appends the body of the frame.
    fn encode(&self, e: &mut Enc);
    /// Reads the body of a frame of `kind`: `Ok(None)` when the kind is not this family's (the frame is then `UnknownKind`). The
    /// module checks that the body was consumed exactly.
    fn decode(kind: u8, d: &mut Dec<'_>) -> Result<Option<Self>, WireError>;
    /// Whether the follower counts this call as a prefill slice in its progress line (`FollowerStats::prefills`).
    fn counts_as_prefill(&self) -> bool {
        false
    }

    /// Appends the family's facts to the Hello, between the geometry and the build.
    fn encode_facts(f: &Self::Facts, e: &mut Enc);
    fn decode_facts(d: &mut Dec<'_>) -> Result<Self::Facts, WireError>;
    /// The facts two ranks disagree on, as `name: rank 0 has X, rank {who} has Y` lines (empty: they agree). The leader's facts first.
    fn check_facts(leader: &Self::Facts, follower: &Self::Facts, who: u32) -> Vec<String>;
}

/// The kinds the module reserves for itself (session frames, then the core commands).
pub const CORE_KINDS: &[u8] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 20, 21, 22, 29];

/// A runtime with no family calls: the type of the session frames that carry none (and of a core-only family's `Call`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NoCall {}

impl FamilyCall for NoCall {
    type Facts = ();
    fn kind(&self) -> u8 {
        match *self {}
    }
    fn name(&self) -> &'static str {
        match *self {}
    }
    fn encode(&self, _e: &mut Enc) {
        match *self {}
    }
    fn decode(_kind: u8, _d: &mut Dec<'_>) -> Result<Option<Self>, WireError> {
        Ok(None)
    }
    fn encode_facts(_f: &(), _e: &mut Enc) {}
    fn decode_facts(_d: &mut Dec<'_>) -> Result<(), WireError> {
        Ok(())
    }
    fn check_facts(_leader: &(), _follower: &(), _who: u32) -> Vec<String> {
        Vec::new()
    }
}

/// What applying a family call answers with: the digest the ack carries (`0`: nothing to compare). A digest that differs between rank 0
/// and a follower is a lost group naming the call, so a family puts in it whatever its call computed that the ranks must agree on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CallReply {
    pub digest: u64,
}

impl CallReply {
    /// The call produced nothing to compare.
    pub const NONE: CallReply = CallReply { digest: 0 };

    pub fn digest(digest: u64) -> CallReply {
        CallReply { digest }
    }
}

/// The runtime a lockstep group replays: the calls every family has, and an opaque [`FamilyCall`] for the family's own.
///
/// The CONTRACT (the determinism argument of `docs/NEMOTRON_TP.md` 5.4, which a family must be able to make for itself): after the same
/// sequence of calls every rank holds the same state, every number a call produces is a pure function of the weights and of the call's
/// arguments, and the candidates of a prefill or window are replicated across the ranks (gathered), so rank 0's answer is the answer.
pub trait LockstepRuntime {
    /// The family's own commands (and boot facts); [`NoCall`] for a family that forwards only the core calls.
    type Call: FamilyCall;

    /// Rows of the vocabulary: compared across ranks at the end of the boot (a rank that loaded another shape is refused).
    fn vocab_size(&self) -> usize;
    /// Whether this rank loaded a draft head (compared at the end of the boot).
    fn has_mtp(&self) -> bool {
        false
    }
    /// Whether this runtime TYPE can snapshot a slot (compared at the end of the boot).
    fn snapshots_supported() -> bool
    where
        Self: Sized,
    {
        false
    }

    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError>;
    fn end_slot(&mut self, slot: usize);
    fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError>;
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError>;
    fn commit(&mut self, slot: usize, keep: usize);
    fn flush(&mut self) -> Result<(), ModelError>;

    /// Applies a family command on a FOLLOWER. `seq` is the leader's sequence number of the command (a family may key its own handles
    /// by it). An error is the call's result class: it must equal the leader's, or the group is lost.
    fn apply_call(&mut self, seq: u64, call: Self::Call) -> Result<CallReply, ModelError>;
}

/// What a rank is, to the other: the facts both ranks must agree on to run one model in lockstep (the boot gate, [`BootGate::check`]).
/// The geometry is every family's; the family's own facts are `family` (the call type's [`FamilyCall::Facts`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BootGate<C: FamilyCall = NoCall> {
    /// `prefix::pack_identity`: the digest of the pack's `config.json` and safetensors index.
    pub pack: String,
    /// Layers loaded (`0`: the whole model).
    pub layers: u32,
    pub slots: u32,
    /// KV pages of the pool (the follower must hold at least the leader's; it then allocates exactly the leader's).
    pub kv_pages: u64,
    pub max_context: u64,
    pub prefill_rows: u32,
    /// The family's facts (on the wire between `prefill_rows` and `build`).
    pub family: C::Facts,
    /// The build ([`build_fingerprint`]): crate version and kernel hash on the first line, then `key=value` lines: the engine commit,
    /// the sha256 of the running executable and the follower spin (`REXL3_TP_SPIN_US`). Compared key by key ([`build_differs`]).
    pub build: String,
    /// The sender's rank (0 is the leader; followers are `1..ranks`).
    pub rank: u32,
    /// The group size the sender was started with (`--tp-ranks`).
    pub ranks: u32,
    /// The collective's settings and library ([`nccl_fingerprint`]): `NCCL_MAX_NCHANNELS` (effective: 4 when unset), `NCCL_PROTO` and the
    /// NCCL library version on the first line, then the canonical hash of the `NCCL_*` environment in effect, the sha256 of the libnccl
    /// file the process loaded and every `NCCL_*` variable as `NAME=value` lines. A different channel count or protocol on one rank is a
    /// different schedule and a 3 to 10 times slower step for the whole group; a different fabric variable or library is a group that
    /// may not form or may run another transport. Refused by name, variable by variable, before any weight is read ([`nccl_differs`]).
    pub nccl: String,
}

impl<C: FamilyCall> BootGate<C> {
    /// A gate for a process that is rank 0 of 1 ([`BootGate::at`] places it): the geometry, the family's facts, this build and the
    /// effective collective settings.
    pub fn new(
        pack: String,
        layers: u32,
        slots: u32,
        kv_pages: u64,
        max_context: u64,
        prefill_rows: u32,
        family: C::Facts,
    ) -> BootGate<C> {
        BootGate {
            pack,
            layers,
            slots,
            kv_pages,
            max_context,
            prefill_rows,
            family,
            build: build_fingerprint(),
            rank: 0,
            ranks: 1,
            nccl: nccl_fingerprint(),
        }
    }

    /// This gate for rank `rank` of a group of `ranks`.
    pub fn at(mut self, rank: usize, ranks: usize) -> BootGate<C> {
        self.rank = rank as u32;
        self.ranks = ranks as u32;
        self
    }

    /// Whether `follower` can run the leader's model in lockstep. Everything that shapes the call stream or the arithmetic must be
    /// equal; the pools the follower holds may be larger (it is then built with exactly the leader's). `Err` is the refusal, naming
    /// both sides.
    pub fn check(leader: &BootGate<C>, follower: &BootGate<C>) -> Result<(), String> {
        let mut why: Vec<String> = Vec::new();
        let who = follower.rank;
        let mut eq = |name: &str, a: String, b: String| {
            if a != b {
                why.push(format!("{name}: rank 0 has {a}, rank {who} has {b}"));
            }
        };
        eq("group size (--tp-ranks)", leader.ranks.to_string(), follower.ranks.to_string());
        eq("pack (config.json and safetensors index digest)", leader.pack.clone(), follower.pack.clone());
        eq("loaded layers (0 is all)", leader.layers.to_string(), follower.layers.to_string());
        eq("max context (--max-model-len)", leader.max_context.to_string(), follower.max_context.to_string());
        eq("prefill rows (--prefill-chunk)", leader.prefill_rows.to_string(), follower.prefill_rows.to_string());
        why.extend(C::check_facts(&leader.family, &follower.family, who));
        why.extend(build_differs(&leader.build, &follower.build, who));
        why.extend(nccl_differs(&leader.nccl, &follower.nccl, who));
        if follower.slots < leader.slots {
            why.push(format!(
                "slots: rank 0 planned {}, rank {who} can hold only {} (pass the same --slots to all ranks, or free memory on rank {who})",
                leader.slots, follower.slots
            ));
        }
        if follower.kv_pages < leader.kv_pages {
            why.push(format!(
                "KV pages: rank 0 planned {}, rank {who} can hold only {} (pass the same --kv-tokens, --kv-headroom-tokens and --slots to all ranks, or free memory on rank {who})",
                leader.kv_pages, follower.kv_pages
            ));
        }
        if why.is_empty() {
            Ok(())
        } else if leader.ranks > 2 {
            Err(format!("rank {who} and rank 0 do not agree, so they would not run one model in lockstep: {}", why.join("; ")))
        } else {
            Err(format!("the two ranks do not agree, so they would not run one model in lockstep: {}", why.join("; ")))
        }
    }
}

/// `name: rank 0 has X, rank {who} has Y` when `a != b`: the line a family's [`FamilyCall::check_facts`] returns for one fact.
pub fn fact_differs(name: &str, a: impl std::fmt::Display, b: impl std::fmt::Display, who: u32) -> Option<String> {
    let (a, b) = (a.to_string(), b.to_string());
    (a != b).then(|| format!("{name}: rank 0 has {a}, rank {who} has {b}"))
}

/// The engine commit the binary was built from, as the server stamps it (`REXL3_COMMIT` of `crates/server/build.rs`): registered once
/// by the binary before any gate is made ([`set_build_commit`]). A process that registered none says `unregistered`, and the gate
/// still compares the executable's own bytes, so two unregistered builds are never taken for one.
static BUILD_COMMIT: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Registers the engine commit of this binary for the boot gate (first call wins).
pub fn set_build_commit(commit: &str) {
    let _ = BUILD_COMMIT.set(commit.trim().to_string());
}

/// The sha256 of a file, hex (`None` when it cannot be read).
pub fn file_sha256(path: &std::path::Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    let mut f = std::fs::File::open(path).ok()?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => h.update(&buf[..n]),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => return None,
        }
    }
    Some(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The sha256 of the running executable (`/proc/self/exe`), computed once: the exact build, whatever the commit says (a `-dirty`
/// tree, an `unknown` commit of an exported source, two builds of one commit with different toolchains are all told apart).
pub fn exe_sha256() -> String {
    static EXE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    EXE.get_or_init(|| {
        let t = Instant::now();
        let h = file_sha256(std::path::Path::new("/proc/self/exe")).unwrap_or_else(|| "unavailable".into());
        eprintln!("rexl3 event=build_identity exe_sha256={} hash_s={:.2}", &h[..h.len().min(16)], t.elapsed().as_secs_f64());
        h
    })
    .clone()
}

/// The build as the gate compares it: crate version and kernel hash (the first line, as protocol 4 has always carried it), then the
/// engine commit, the executable's sha256 and the follower spin. See [`build_fingerprint_of`].
pub fn build_fingerprint() -> String {
    build_fingerprint_of(
        env!("CARGO_PKG_VERSION"),
        gb_cuda::rt::versions().kernel_hash,
        BUILD_COMMIT.get().map(String::as_str),
        &exe_sha256(),
        spin_from_env().as_micros() as u64,
    )
}

/// [`build_fingerprint`] over its inputs (the tests' seam): `version+kernel`, then `commit=`, `exe_sha256=` and `tp_spin_us=` lines.
pub fn build_fingerprint_of(version: &str, kernel: &str, commit: Option<&str>, exe: &str, spin_us: u64) -> String {
    format!(
        "{version}+{kernel}\ncommit={}\nexe_sha256={exe}\ntp_spin_us={spin_us}",
        commit.filter(|c| !c.is_empty()).unwrap_or("unregistered")
    )
}

/// A gate string as its parts: the first line under `""`, every later `key=value` line under its key.
fn gate_lines(s: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for (i, line) in s.split('\n').enumerate() {
        if i == 0 {
            m.insert(String::new(), line.to_string());
        } else if let Some((k, v)) = line.split_once('=') {
            m.insert(k.to_string(), v.to_string());
        }
    }
    m
}

/// The keys two gate strings disagree on, as `(key, leader's value, follower's value)`; a key one side lacks is `(absent: ...)` (a build
/// from before the key existed).
fn gate_diffs(a: &str, b: &str) -> Vec<(String, String, String)> {
    let (ma, mb) = (gate_lines(a), gate_lines(b));
    let mut keys: Vec<&String> = ma.keys().chain(mb.keys()).collect();
    keys.sort();
    keys.dedup();
    let absent = || "(absent: a build from before this field)".to_string();
    keys.into_iter()
        .filter_map(|k| {
            let (x, y) = (ma.get(k), mb.get(k));
            (x != y).then(|| (k.clone(), x.cloned().unwrap_or_else(absent), y.cloned().unwrap_or_else(absent)))
        })
        .collect()
}

/// The refusal lines for two builds ([`build_fingerprint`]), one per differing part, each named.
pub fn build_differs(leader: &str, follower: &str, who: u32) -> Vec<String> {
    gate_diffs(leader, follower)
        .into_iter()
        .map(|(k, a, b)| {
            let name = match k.as_str() {
                "" => "build (crate version and kernel hash)",
                "commit" => "build commit (the engine source both ranks were built from)",
                "exe_sha256" => "build binary sha256 (run the same rexl3-serve bytes on every rank)",
                "tp_spin_us" => "follower spin (REXL3_TP_SPIN_US; latency only, pass the same to every rank)",
                _ => return format!("build field {k}: rank 0 has {a}, rank {who} has {b}"),
            };
            let short = |v: String| if k == "exe_sha256" && v.len() == 64 { v[..16].to_string() } else { v };
            format!("{name}: rank 0 has {}, rank {who} has {}", short(a), short(b))
        })
        .collect()
}

/// The refusal lines for two NCCL fingerprints ([`nccl_fingerprint`]), one per differing part: the settings line, the library file,
/// each `NCCL_*` variable by name. The environment hash is derived from those and is reported only when nothing else differs.
pub fn nccl_differs(leader: &str, follower: &str, who: u32) -> Vec<String> {
    let d = gate_diffs(leader, follower);
    // the hash is derived from the rest: named only when nothing else explains it
    let explained = d.iter().any(|(k, _, _)| k.starts_with("NCCL_") || k == "lib_sha256" || k == "per_node");
    d.into_iter()
        .filter(|(k, _, _)| !(explained && k == "env_sha256"))
        .map(|(k, a, b)| match k.as_str() {
            "" => format!(
                "NCCL settings and library (NCCL_MAX_NCHANNELS, NCCL_PROTO, library version): rank 0 has {a}, rank {who} has {b}"
            ),
            "lib_sha256" => format!(
                "NCCL library file (sha256 of the libnccl the process loaded; REXL3_NCCL_LIB): rank 0 has {a}, rank {who} has {b}"
            ),
            "env_sha256" => {
                format!("NCCL environment (hash of the NCCL_* variables in effect): rank 0 has {a}, rank {who} has {b}")
            }
            "per_node" => format!("NCCL per-node variables (REXL3_TP_NCCL_PER_NODE): rank 0 has {a}, rank {who} has {b}"),
            v if v.starts_with("NCCL_") => {
                let unset = |x: String| if x.starts_with("(absent") { "(unset)".to_string() } else { x };
                format!(
                    "NCCL environment variable {v} (set it the same on every rank): rank 0 has {}, rank {who} has {}",
                    unset(a),
                    unset(b)
                )
            }
            _ => format!("NCCL field {k}: rank 0 has {a}, rank {who} has {b}"),
        })
        .collect()
}

/// The channel cap a group runs under when `NCCL_MAX_NCHANNELS` is not set: NCCL's own tuning is 3 to 10 times slower on the two-node
/// fabric (`docs/NEMOTRON_TP.md` section 10.3), so `tp_boot::make_comm` exports this before it joins the communicator.
pub const DEFAULT_NCCL_CHANNELS: &str = "4";

/// Variables of the `NCCL_*` environment the gate never compares: diagnostics that change no schedule and that usually differ per host
/// (`NCCL_DEBUG_FILE` carries `%h`).
const NCCL_DIAGNOSTIC: &[&str] = &["NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE", "NCCL_DEBUG_TIMESTAMP_FORMAT"];

/// The variable naming `NCCL_*` variables that may differ per node (a comma list, for example an HCA name that differs between two
/// boxes): they are left out of the comparison and the hash; the list itself is compared.
pub const NCCL_PER_NODE_ENV: &str = "REXL3_TP_NCCL_PER_NODE";

/// The collective's settings as the gate compares them: the EFFECTIVE channel cap (the environment's, or [`DEFAULT_NCCL_CHANNELS`] when
/// it is unset, which is what `make_comm` will export), `NCCL_PROTO` and the NCCL library version on the first line, then the
/// environment in effect ([`nccl_env_lines`]). `off` for a group booted without a communicator (`REXL3_TP_COMM=none`). Never fails: a
/// library that cannot be loaded says so in the string (and `make_comm` then refuses by name).
pub fn nccl_fingerprint() -> String {
    let replicated = std::env::var("REXL3_TP_COMM").is_ok_and(|v| v == "none");
    let head = nccl_fingerprint_of(
        replicated,
        std::env::var("NCCL_MAX_NCHANNELS").ok().as_deref(),
        std::env::var("NCCL_PROTO").ok().as_deref(),
        gb_cuda::nccl::NcclComm::version().ok(),
    );
    if replicated {
        return head;
    }
    let vars: Vec<(String, String)> =
        std::env::vars_os().filter_map(|(k, v)| Some((k.into_string().ok()?, v.to_string_lossy().into_owned()))).collect();
    let lib = gb_cuda::nccl::NcclComm::loaded_library_file().and_then(|p| file_sha256(std::path::Path::new(&p)));
    format!("{head}\n{}", nccl_env_lines(&vars, std::env::var(NCCL_PER_NODE_ENV).ok().as_deref(), lib.as_deref()))
}

/// The environment part of [`nccl_fingerprint`] over its inputs (the tests' seam): `env_sha256=` (16 hex of the sha256 of the sorted
/// `NAME=value` lines and the library hash), `lib_sha256=`, `per_node=` and one `NAME=value` line per compared `NCCL_*` variable, sorted.
/// `NCCL_MAX_NCHANNELS` is the effective one (the default when unset), as `make_comm` exports it.
pub fn nccl_env_lines(vars: &[(String, String)], per_node: Option<&str>, lib_sha256: Option<&str>) -> String {
    use sha2::{Digest, Sha256};
    let skip: Vec<&str> = per_node.unwrap_or("").split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    let mut kept: BTreeMap<&str, &str> = vars
        .iter()
        .filter(|(k, _)| k.starts_with("NCCL_") && !NCCL_DIAGNOSTIC.contains(&k.as_str()) && !skip.contains(&k.as_str()))
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    kept.entry("NCCL_MAX_NCHANNELS").or_insert(DEFAULT_NCCL_CHANNELS);
    let lib = lib_sha256.unwrap_or("unavailable");
    let body: Vec<String> = kept.iter().map(|(k, v)| format!("{k}={}", v.replace('\n', " "))).collect();
    let mut h = Sha256::new();
    for l in &body {
        h.update(l.as_bytes());
        h.update(b"\n");
    }
    h.update(format!("lib_sha256={lib}\n").as_bytes());
    let hex: String = h.finalize().iter().take(8).map(|b| format!("{b:02x}")).collect();
    let mut skip_sorted = skip.clone();
    skip_sorted.sort_unstable();
    format!("env_sha256={hex}\nlib_sha256={lib}\nper_node={}\n{}", skip_sorted.join(","), body.join("\n"))
}

/// The first line of a gate string (the part a log line prints) and its environment hash, for the boot log.
pub fn nccl_summary(nccl: &str) -> String {
    let m = gate_lines(nccl);
    match m.get("env_sha256") {
        Some(h) => format!("{} env_sha256={h}", m.get("").map_or("", String::as_str)),
        None => m.get("").cloned().unwrap_or_default(),
    }
}

/// The first line of [`nccl_fingerprint`] over its inputs (the tests' seam).
pub fn nccl_fingerprint_of(replicated: bool, channels: Option<&str>, proto: Option<&str>, version: Option<i32>) -> String {
    if replicated {
        return "off".into();
    }
    format!(
        "channels={},proto={},lib={}",
        channels.map(str::trim).filter(|c| !c.is_empty()).unwrap_or(DEFAULT_NCCL_CHANNELS),
        proto.map(str::trim).filter(|p| !p.is_empty()).unwrap_or("default"),
        version.map_or("unavailable".to_string(), |v| v.to_string())
    )
}

/// What the follower reports once its shard is loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadyInfo {
    pub vocab: u64,
    pub has_mtp: bool,
    pub snapshots: bool,
}
/// A message of the protocol. Commands (everything from `BeginSlot` on) carry the leader's sequence number in the frame header.
#[derive(Clone, Debug, PartialEq)]
pub enum Msg<C: FamilyCall = NoCall> {
    /// Both directions, first frame of a connection.
    Hello(BootGate<C>),
    /// Either direction: the sender refuses the boot or the run, and says why.
    Abort(String),
    /// Leader to follower: bootstrap bytes of the collective (the NCCL unique id).
    CommInit(Vec<u8>),
    /// Follower to leader: the shard is loaded.
    Ready(ReadyInfo),
    Ping(u64),
    /// Reply to a ping: its nonce and the highest command sequence the follower has applied.
    Pong {
        nonce: u64,
        applied: u64,
    },
    /// Follower to leader: the result of command `seq` (the frame header's sequence number).
    Ack(Class, String),
    /// Leader to follower: the leader is going away; stop after the commands already sent.
    Shutdown(String),
    /// Follower to leader: the follower is leaving.
    Bye(String),

    // ---- commands
    BeginSlot(u32),
    EndSlot(u32),
    Prefill {
        batch: RowBatch,
        want: Option<u32>,
    },
    Window {
        batch: RowBatch,
        count: u32,
    },
    Commit {
        slot: u32,
        keep: u32,
    },
    Flush,
    /// Does nothing; its ack says everything before it has been applied.
    Barrier,
    /// A command of the family (its kind byte is the call's own).
    Call(C),
}

const K_HELLO: u8 = 1;
const K_ABORT: u8 = 2;
const K_COMM: u8 = 3;
const K_READY: u8 = 4;
const K_PING: u8 = 5;
const K_PONG: u8 = 6;
const K_ACK: u8 = 7;
const K_SHUTDOWN: u8 = 8;
const K_BYE: u8 = 9;
const K_BEGIN: u8 = 16;
const K_END: u8 = 17;
const K_PREFILL: u8 = 18;
const K_WINDOW: u8 = 20;
const K_COMMIT: u8 = 21;
const K_FLUSH: u8 = 22;
const K_BARRIER: u8 = 29;
pub fn enc_batch(e: &mut Enc, b: &RowBatch) {
    e.u64(b.rows as u64);
    e.i32s(&b.ids);
    e.i32s(&b.positions);
    e.i32s(&b.row_slot);
    e.i32s(&b.seq_start);
    e.i32s(&b.seq_slot);
    e.opt_i32s(&b.rope3);
    e.opt_i32s(&b.parents);
}

pub fn dec_batch(d: &mut Dec<'_>) -> Result<RowBatch, WireError> {
    let rows = d.u64("batch rows")?;
    let b = RowBatch {
        rows: usize::try_from(rows).map_err(|_| WireError::Invalid("batch rows overflow".into()))?,
        ids: d.i32s("batch ids")?,
        positions: d.i32s("batch positions")?,
        row_slot: d.i32s("batch row_slot")?,
        seq_start: d.i32s("batch seq_start")?,
        seq_slot: d.i32s("batch seq_slot")?,
        rope3: d.opt_i32s("batch rope3")?,
        parents: d.opt_i32s("batch parents")?,
    };
    // the shape the model indexes by: one id, position and slot per row, one start per sequence and a closing one
    let ok = b.ids.len() == b.rows
        && b.positions.len() == b.rows
        && b.row_slot.len() == b.rows
        && b.seq_start.len() == b.seq_slot.len() + 1
        && b.rope3.as_ref().is_none_or(|v| v.len() == 3 * b.rows)
        && b.parents.as_ref().is_none_or(|v| v.len() == b.rows);
    if !ok {
        return Err(WireError::Invalid(format!("a row batch of {} rows whose arrays do not match", b.rows)));
    }
    Ok(b)
}
impl<C: FamilyCall> Msg<C> {
    /// A command: a message whose application changes (or may change) the follower's model state, numbered by the leader.
    pub fn is_command(&self) -> bool {
        self.kind() >= K_BEGIN
    }

    pub fn name(&self) -> &'static str {
        match self {
            Msg::Hello(_) => "hello",
            Msg::Abort(_) => "abort",
            Msg::CommInit(_) => "comm_init",
            Msg::Ready(_) => "ready",
            Msg::Ping(_) => "ping",
            Msg::Pong { .. } => "pong",
            Msg::Ack(..) => "ack",
            Msg::Shutdown(_) => "shutdown",
            Msg::Bye(_) => "bye",
            Msg::BeginSlot(_) => "begin_slot",
            Msg::EndSlot(_) => "end_slot",
            Msg::Prefill { .. } => "prefill_chunk",
            Msg::Window { .. } => "window_candidates",
            Msg::Commit { .. } => "commit",
            Msg::Flush => "flush",
            Msg::Barrier => "barrier",
            Msg::Call(c) => c.name(),
        }
    }

    fn kind(&self) -> u8 {
        match self {
            Msg::Hello(_) => K_HELLO,
            Msg::Abort(_) => K_ABORT,
            Msg::CommInit(_) => K_COMM,
            Msg::Ready(_) => K_READY,
            Msg::Ping(_) => K_PING,
            Msg::Pong { .. } => K_PONG,
            Msg::Ack(..) => K_ACK,
            Msg::Shutdown(_) => K_SHUTDOWN,
            Msg::Bye(_) => K_BYE,
            Msg::BeginSlot(_) => K_BEGIN,
            Msg::EndSlot(_) => K_END,
            Msg::Prefill { .. } => K_PREFILL,
            Msg::Window { .. } => K_WINDOW,
            Msg::Commit { .. } => K_COMMIT,
            Msg::Flush => K_FLUSH,
            Msg::Barrier => K_BARRIER,
            Msg::Call(c) => c.kind(),
        }
    }

    /// The whole frame: `u32 length | u8 version | u8 kind | u64 seq | body`, the length covering everything after itself.
    pub fn encode(&self, seq: u64) -> Vec<u8> {
        let mut e = Enc(Vec::with_capacity(64));
        e.0.extend_from_slice(&[0, 0, 0, 0]);
        e.u8(PROTOCOL_VERSION);
        e.u8(self.kind());
        e.u64(seq);
        match self {
            Msg::Hello(g) => {
                e.0.extend_from_slice(&MAGIC);
                e.str(&g.pack);
                e.u32(g.layers);
                e.u32(g.slots);
                e.u64(g.kv_pages);
                e.u64(g.max_context);
                e.u32(g.prefill_rows);
                C::encode_facts(&g.family, &mut e);
                e.str(&g.build);
                e.u32(g.rank);
                e.u32(g.ranks);
                e.str(&g.nccl);
            }
            Msg::Abort(s) | Msg::Shutdown(s) | Msg::Bye(s) => e.str(s),
            Msg::CommInit(b) => e.bytes(b),
            Msg::Ready(r) => {
                e.u64(r.vocab);
                e.bool(r.has_mtp);
                e.bool(r.snapshots);
            }
            Msg::Ping(n) => e.u64(*n),
            Msg::Pong { nonce, applied } => {
                e.u64(*nonce);
                e.u64(*applied);
            }
            Msg::Ack(c, text) => {
                match c {
                    Class::Ok(aux) => {
                        e.u8(0);
                        e.u64(*aux);
                    }
                    Class::Err { kind, fatal } => {
                        e.u8(1);
                        e.u8(*kind);
                        e.bool(*fatal);
                    }
                }
                e.str(text);
            }
            Msg::BeginSlot(s) | Msg::EndSlot(s) => e.u32(*s),
            Msg::Prefill { batch, want } => {
                enc_batch(&mut e, batch);
                e.opt_u32(*want);
            }
            Msg::Window { batch, count } => {
                enc_batch(&mut e, batch);
                e.u32(*count);
            }
            Msg::Commit { slot, keep } => {
                e.u32(*slot);
                e.u32(*keep);
            }
            Msg::Flush | Msg::Barrier => {}
            Msg::Call(c) => c.encode(&mut e),
        }
        let len = (e.0.len() - 4) as u32;
        e.0[..4].copy_from_slice(&len.to_le_bytes());
        e.0
    }

    /// Decodes the payload of a frame (everything after the length): `(seq, message)`.
    pub fn decode(payload: &[u8]) -> Result<(u64, Msg<C>), WireError> {
        let mut d = Dec { b: payload, at: 0 };
        let version = d.u8("version")?;
        if version != PROTOCOL_VERSION {
            return Err(WireError::Version { got: version, want: PROTOCOL_VERSION });
        }
        let kind = d.u8("kind")?;
        let seq = d.u64("sequence number")?;
        let msg = match kind {
            K_HELLO => {
                if d.take(4, "magic")? != MAGIC {
                    return Err(WireError::Invalid("not a rexl3 tensor-parallel hello".into()));
                }
                let pack = d.str("pack")?;
                let layers = d.u32("layers")?;
                let slots = d.u32("slots")?;
                let kv_pages = d.u64("kv pages")?;
                let max_context = d.u64("max context")?;
                let prefill_rows = d.u32("prefill rows")?;
                let family = C::decode_facts(&mut d)?;
                Msg::Hello(BootGate {
                    pack,
                    layers,
                    slots,
                    kv_pages,
                    max_context,
                    prefill_rows,
                    family,
                    build: d.str("build")?,
                    rank: d.u32("rank")?,
                    ranks: d.u32("ranks")?,
                    nccl: d.str("nccl")?,
                })
            }
            K_ABORT => Msg::Abort(d.str("abort reason")?),
            K_SHUTDOWN => Msg::Shutdown(d.str("shutdown reason")?),
            K_BYE => Msg::Bye(d.str("bye reason")?),
            K_COMM => Msg::CommInit(d.bytes("comm bytes")?),
            K_READY => {
                Msg::Ready(ReadyInfo { vocab: d.u64("vocab")?, has_mtp: d.bool("has_mtp")?, snapshots: d.bool("snapshots")? })
            }
            K_PING => Msg::Ping(d.u64("nonce")?),
            K_PONG => Msg::Pong { nonce: d.u64("nonce")?, applied: d.u64("applied")? },
            K_ACK => {
                let class = match d.u8("ack tag")? {
                    0 => Class::Ok(d.u64("ack digest")?),
                    1 => Class::Err { kind: d.u8("ack kind")?, fatal: d.bool("ack fatal")? },
                    t => return Err(WireError::Invalid(format!("ack tag {t}"))),
                };
                Msg::Ack(class, d.str("ack text")?)
            }
            K_BEGIN => Msg::BeginSlot(d.u32("slot")?),
            K_END => Msg::EndSlot(d.u32("slot")?),
            K_PREFILL => Msg::Prefill { batch: dec_batch(&mut d)?, want: d.opt_u32("want")? },
            K_WINDOW => Msg::Window { batch: dec_batch(&mut d)?, count: d.u32("count")? },
            K_COMMIT => Msg::Commit { slot: d.u32("slot")?, keep: d.u32("keep")? },
            K_FLUSH => Msg::Flush,
            K_BARRIER => Msg::Barrier,
            k => match C::decode(k, &mut d)? {
                Some(c) if k >= K_BEGIN && !CORE_KINDS.contains(&k) => Msg::Call(c),
                _ => return Err(WireError::UnknownKind(k)),
            },
        };
        d.done(msg.name())?;
        Ok((seq, msg))
    }
}

/// Reads one frame (blocking). A clean close at a frame boundary is [`WireError::Closed`].
pub fn read_frame<C: FamilyCall>(r: &mut dyn Read) -> Result<(u64, Msg<C>), WireError> {
    let mut len = [0u8; 4];
    let mut got = 0;
    while got < 4 {
        match r.read(&mut len[got..]) {
            Ok(0) if got == 0 => return Err(WireError::Closed),
            Ok(0) => return Err(WireError::Truncated("frame length")),
            Ok(n) => {
                if got == 0 {
                    crate::hosttrace::mark("rx_first_byte", 0);
                }
                got += n;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    let n = u32::from_le_bytes(len) as usize;
    if n > MAX_FRAME {
        return Err(WireError::TooLarge(n));
    }
    if n < HEADER {
        return Err(WireError::Invalid(format!("a frame of {n} bytes has no header")));
    }
    let mut payload = vec![0u8; n];
    r.read_exact(&mut payload).map_err(|e| match e.kind() {
        io::ErrorKind::UnexpectedEof => WireError::Truncated("frame body"),
        _ => WireError::Io(e),
    })?;
    crate::hosttrace::mark("rx_payload", n as u64);
    Msg::decode(&payload)
}

/// Writes one frame (one `write_all`, so a frame is never interleaved with another from the same writer).
pub fn write_frame<C: FamilyCall>(w: &mut dyn Write, seq: u64, m: &Msg<C>) -> Result<(), WireError> {
    let frame = m.encode(seq);
    if frame.len() - 4 > MAX_FRAME {
        return Err(WireError::TooLarge(frame.len() - 4));
    }
    w.write_all(&frame)?;
    w.flush()?;
    Ok(())
}
// ------------------------------------------------------------------------------------------------------------------- transport

/// A connection: a read half, a write half, and a way to wake whoever is blocked on either (shutting the socket down). The halves are
/// separate objects so a reader thread can own one while commands go out the other.
pub struct Link<C: FamilyCall = NoCall> {
    pub(crate) rx: Box<dyn Read + Send>,
    pub(crate) tx: Box<dyn Write + Send>,
    pub(crate) close: Arc<dyn Fn() + Send + Sync>,
    /// Ends this side's stream only (a FIN after what was written): the peer reads everything and then EOF, and what it still sends is
    /// read, not answered with a reset.
    pub(crate) close_write: Arc<dyn Fn() + Send + Sync>,
    pub(crate) set_rx_timeout: Arc<dyn Fn(Option<Duration>) + Send + Sync>,
    /// A non-blocking "is a byte (or the end of the stream) waiting on the read half" check, when the stream has a file descriptor:
    /// what a follower's reader spins on before it parks in `read` (`FollowerConfig::spin`).
    pub(crate) rx_ready: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    /// Who is on the other end, for messages.
    pub peer: String,
    _call: PhantomData<fn() -> C>,
}

impl<C: FamilyCall> std::fmt::Debug for Link<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Link(peer {})", self.peer)
    }
}

/// A stream a [`Link`] can be made of.
pub trait Duplex: Read + Write + Send + Sync + Sized + 'static {
    fn try_clone_stream(&self) -> io::Result<Self>;
    fn shutdown_both(&self);
    fn shutdown_write(&self);
    fn set_read_timeout_of(&self, t: Option<Duration>) -> io::Result<()>;
    fn set_write_timeout_of(&self, t: Option<Duration>) -> io::Result<()>;
    fn describe_peer(&self) -> String;
    /// The stream's file descriptor, for a non-blocking readiness check (`None`: the stream has none).
    fn raw_fd(&self) -> Option<i32> {
        None
    }
}

/// Whether `fd` has a byte, a hangup or an error pending: one `poll` with a zero timeout.
fn poll_readable(fd: i32) -> bool {
    let mut p = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
    // SAFETY: `p` is a valid pollfd for the one entry given, and a zero timeout never blocks.
    let r = unsafe { libc::poll(&mut p, 1, 0) };
    // a failed poll ends the spin: the blocking read reports whatever is wrong
    r != 0
}

impl Duplex for std::net::TcpStream {
    fn try_clone_stream(&self) -> io::Result<Self> {
        self.try_clone()
    }
    fn shutdown_both(&self) {
        let _ = self.shutdown(std::net::Shutdown::Both);
    }
    fn shutdown_write(&self) {
        let _ = self.shutdown(std::net::Shutdown::Write);
    }
    fn set_read_timeout_of(&self, t: Option<Duration>) -> io::Result<()> {
        self.set_read_timeout(t)
    }
    fn set_write_timeout_of(&self, t: Option<Duration>) -> io::Result<()> {
        self.set_write_timeout(t)
    }
    fn describe_peer(&self) -> String {
        self.peer_addr().map(|a| a.to_string()).unwrap_or_else(|_| "peer".into())
    }
    fn raw_fd(&self) -> Option<i32> {
        use std::os::fd::AsRawFd;
        Some(self.as_raw_fd())
    }
}

#[cfg(unix)]
impl Duplex for std::os::unix::net::UnixStream {
    fn try_clone_stream(&self) -> io::Result<Self> {
        self.try_clone()
    }
    fn shutdown_both(&self) {
        let _ = self.shutdown(std::net::Shutdown::Both);
    }
    fn shutdown_write(&self) {
        let _ = self.shutdown(std::net::Shutdown::Write);
    }
    fn set_read_timeout_of(&self, t: Option<Duration>) -> io::Result<()> {
        self.set_read_timeout(t)
    }
    fn set_write_timeout_of(&self, t: Option<Duration>) -> io::Result<()> {
        self.set_write_timeout(t)
    }
    fn describe_peer(&self) -> String {
        "local peer".into()
    }
    fn raw_fd(&self) -> Option<i32> {
        use std::os::fd::AsRawFd;
        Some(self.as_raw_fd())
    }
}

impl<C: FamilyCall> Link<C> {
    /// A link over one duplex stream (TCP, a Unix socket pair). `write_timeout` bounds a write that does not drain: a peer that stopped
    /// reading is a lost peer, never a hang.
    pub fn new<S: Duplex>(s: S, write_timeout: Duration) -> io::Result<Link<C>> {
        s.set_write_timeout_of(Some(write_timeout.max(Duration::from_millis(1))))?;
        let peer = s.describe_peer();
        let rx = s.try_clone_stream()?;
        let tx = s.try_clone_stream()?;
        let fd = s.raw_fd();
        let ctl = Arc::new(s);
        let (c1, c2, c3, c4) = (ctl.clone(), ctl.clone(), ctl.clone(), ctl);
        // the check holds the stream, so the descriptor it polls stays open as long as the check exists
        let rx_ready = fd.map(|fd| {
            let f: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || {
                let _keep = &c4;
                poll_readable(fd)
            });
            f
        });
        Ok(Link {
            rx: Box::new(rx),
            tx: Box::new(tx),
            close: Arc::new(move || c1.shutdown_both()),
            close_write: Arc::new(move || c3.shutdown_write()),
            set_rx_timeout: Arc::new(move |t| {
                let _ = c2.set_read_timeout_of(t);
            }),
            rx_ready,
            peer,
            _call: PhantomData,
        })
    }

    /// A link over TCP with Nagle off (commands are small and latency is the budget).
    pub fn tcp(s: std::net::TcpStream, write_timeout: Duration) -> io::Result<Link<C>> {
        s.set_nodelay(true)?;
        Link::new(s, write_timeout)
    }

    /// A pair of connected in-process links (Unix socket pair), for tests and tools.
    #[cfg(unix)]
    pub fn pair(write_timeout: Duration) -> io::Result<(Link<C>, Link<C>)> {
        let (a, b) = std::os::unix::net::UnixStream::pair()?;
        Ok((Link::new(a, write_timeout)?, Link::new(b, write_timeout)?))
    }

    /// Sends one frame (boot phase: before the link is handed to a leader or follower).
    pub fn send(&mut self, seq: u64, m: &Msg<C>) -> Result<(), WireError> {
        write_frame(&mut *self.tx, seq, m)
    }

    /// Receives one frame, waiting at most `timeout` (`None`: forever). A timeout is `WireError::Io` of kind `WouldBlock`/`TimedOut`.
    pub fn recv(&mut self, timeout: Option<Duration>) -> Result<(u64, Msg<C>), WireError> {
        (self.set_rx_timeout)(timeout);
        let r = read_frame(&mut *self.rx);
        (self.set_rx_timeout)(None);
        r
    }

    /// [`Link::recv`] that calls `poll` about once a second while it waits for a frame to begin (the boot's cancellation point), and
    /// gives up at `timeout` with a `TimedOut` error. Once the first byte of a frame has arrived the rest is read under what is left of the
    /// deadline, so a frame is never torn by a poll.
    pub fn recv_polled(&mut self, timeout: Duration, mut poll: impl FnMut()) -> Result<(u64, Msg<C>), WireError> {
        let deadline = Instant::now() + timeout;
        let mut first = [0u8; 1];
        loop {
            poll();
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                (self.set_rx_timeout)(None);
                return Err(WireError::Io(io::Error::new(io::ErrorKind::TimedOut, "no frame before the deadline")));
            }
            (self.set_rx_timeout)(Some(left.min(Duration::from_secs(1))));
            match self.rx.read(&mut first) {
                Ok(0) => {
                    (self.set_rx_timeout)(None);
                    return Err(WireError::Closed);
                }
                Ok(_) => break,
                Err(e)
                    if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::Interrupted) => {}
                Err(e) => {
                    (self.set_rx_timeout)(None);
                    return Err(e.into());
                }
            }
        }
        let rest = deadline.saturating_duration_since(Instant::now()).max(Duration::from_secs(5));
        (self.set_rx_timeout)(Some(rest));
        let r = read_frame(&mut (&first[..]).chain(&mut *self.rx));
        (self.set_rx_timeout)(None);
        r
    }

    /// Refuses the run: sends an Abort with `why`, ends this side's stream and reads (and drops) what the peer still sends until it
    /// closes or a couple of seconds pass, so the peer reads the reason instead of a reset that races it.
    pub fn refuse(&mut self, why: &str) {
        let _ = self.send(0, &Msg::Abort(why.to_string()));
        (self.close_write)();
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut sink = [0u8; 4096];
        while Instant::now() < deadline {
            (self.set_rx_timeout)(Some(Duration::from_millis(500)));
            match self.rx.read(&mut sink) {
                Ok(0) => break,
                Ok(_) => {}
                Err(e)
                    if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::Interrupted) => {}
                Err(_) => break,
            }
        }
        (self.set_rx_timeout)(None);
    }

    /// Wakes everything blocked on this link and makes its reads and writes fail.
    pub fn close(&self) {
        (self.close)();
    }
}
fn timed_out(e: &WireError) -> bool {
    matches!(e, WireError::Io(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut))
}

// ------------------------------------------------------------------------------------------------------------------- shared pieces

/// The knobs of a run: how long to wait for what.
#[derive(Clone, Debug)]
pub struct TpTimeouts {
    /// How long rank 0 waits for rank 1 to connect, and rank 1 for rank 0 to listen.
    pub connect: Duration,
    /// How long rank 0 waits, after its own shard is loaded, for rank 1's shard (and rank 1 for the leader's first command is unbounded).
    pub ready: Duration,
    /// No frame from the peer for this long is a lost peer.
    pub peer: Duration,
    /// A write that does not drain within this is a lost peer.
    pub write: Duration,
    /// An idle leader pings this often.
    pub ping: Duration,
    /// How long `barrier` and a family's boot-time calls ([`TpLeader::settle`]) wait for every follower's ack.
    pub barrier: Duration,
    /// When the peer is lost while a local call is still in flight (stuck in a collective), abort the process after this much longer;
    /// `None` never aborts (tests).
    pub hang_exit: Option<Duration>,
    /// How often the watchdog polls the communicator's health ([`TpLeader::on_comm_health`], [`FollowerConfig::health`]): an
    /// asynchronous error of the collective fabric (a RoCE link down while the TCP command link still carries pings) loses the group
    /// at the next poll instead of never, or at the server's 60 s progress watchdog.
    pub health_poll: Duration,
}

impl Default for TpTimeouts {
    fn default() -> Self {
        TpTimeouts {
            connect: Duration::from_secs(900),
            ready: Duration::from_secs(1800),
            peer: Duration::from_secs(15),
            write: Duration::from_secs(15),
            ping: Duration::from_secs(1),
            barrier: Duration::from_secs(300),
            hang_exit: Some(Duration::from_secs(20)),
            health_poll: Duration::from_millis(500),
        }
    }
}

/// The error every call of a lost group returns.
pub fn lost_group(why: impl Into<String>) -> ModelError {
    ModelError::StateLost(format!("tensor-parallel group lost: {}", why.into()))
}

/// What a local call's error becomes when it is only the symptom of a group that is already lost.
///
/// A rank whose peer died is inside a collective that can never finish. The group's loss is latched first (a reader saw the link end,
/// or a watchdog saw the silence) and its abort hook aborts the communicator, so the collective comes back with a bare NCCL code
/// (`CUDA error 20103`, an aborted communicator) that says nothing of why. Put the cause where the client and the log read it: that
/// the group was lost, which peer rank when it is known (`blamed`), the latched reason (`failure`) and the call that was cut short
/// (`what`), and keep the raw code for diagnosis.
///
/// Any other error is returned as it was, and so is a collective failure with no loss latched before it: that error is this rank's
/// own and is what loses the group. The result is fatal exactly as the code was (`StateLost`, like every later call of a lost group),
/// so the exit status (70) does not move, and the class compared with the followers' acks is computed from the original error.
pub(crate) fn lost_peer_error(e: ModelError, failure: Option<&str>, blamed: Option<usize>, what: &str) -> ModelError {
    let (ModelError::Cuda(code), Some(why)) = (&e, failure) else { return e };
    if !gb_cuda::nccl::is_collective_failure(code.0) {
        return e;
    }
    let cause = match blamed {
        Some(rank) => format!("peer rank {rank} was lost ({why})"),
        None => why.to_string(),
    };
    lost_group(format!("{cause}; {what} was cut short by its collective: {code}"))
}

/// FNV-1a digest of what a prefill or window produced: the shape and, for every row, the first eight candidates' ids and values
/// (`-0.0` is `0.0`: the sign of a zero is the one thing the exactness argument allows to differ between ranks).
pub fn candidates_digest(c: &Candidates) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |bytes: &[u8]| {
        for b in bytes {
            h = (h ^ *b as u64).wrapping_mul(0x100_0000_01b3);
        }
    };
    eat(&(c.rows as u64).to_le_bytes());
    eat(&(c.count as u64).to_le_bytes());
    let n = c.count.min(8);
    for r in 0..c.rows {
        for j in 0..n {
            let at = r * c.count + j;
            if let (Some(id), Some(v)) = (c.ids.get(at), c.vals.get(at)) {
                eat(&id.to_le_bytes());
                let bits = if *v == 0.0 { 0u32 } else { v.to_bits() };
                eat(&bits.to_le_bytes());
            }
        }
    }
    h | 1
}

/// The communicator's abort: run when the group is lost, on a thread that may be left behind (it can block).
pub type AbortHook = Box<dyn Fn() + Send>;

/// The communicator's health, polled by the watchdog every [`TpTimeouts::health_poll`]: `Some(why)` when the collective has failed
/// asynchronously (NCCL's `ncclCommGetAsyncError`), which loses the group with `why`. Must return at once (it runs on the watchdog
/// thread) and must not take a lock a stuck collective holds.
pub type HealthProbe = Arc<dyn Fn() -> Option<String> + Send + Sync>;

/// What the leader holds of its own result of one command until every follower has met it.
pub(crate) struct LocalResult {
    name: &'static str,
    class: Class,
    tolerant: bool,
    /// How many followers have met it.
    met: usize,
}

/// The pairing of the leader's results with the followers': the leader's local result of command `s` and each follower's ack of it meet
/// here, in either order, and are compared. Every follower's result is compared with the leader's, so all of them equal rank 0's and,
/// by transitivity, each other's. Both streams are produced in sequence order on every rank, so everything up to `settled` (the
/// minimum over the followers' cursors) has met on every rank.
#[derive(Default)]
pub(crate) struct Ledger {
    /// Followers (ranks `1..=n`; rank `r` is index `r - 1` of every per-rank vector).
    pub(crate) n: usize,
    pub(crate) local: BTreeMap<u64, LocalResult>,
    /// Acks that arrived before the leader's own result, by follower.
    pub(crate) acks: Vec<BTreeMap<u64, (Class, String)>>,
    /// Highest command each follower's ack has met.
    pub(crate) cursor: Vec<u64>,
    /// Highest ack number received from each follower.
    pub(crate) ack_seq: Vec<u64>,
    pub(crate) acked_by: Vec<u64>,
    /// Acks whose result differed from the leader's, by follower.
    pub(crate) diverged_by: Vec<u64>,
    pub(crate) settled: u64,
    pub(crate) failed: Option<String>,
    /// The follower the failure is blamed on, when it is one.
    pub(crate) blamed: Option<usize>,
    pub(crate) tolerated: u64,
    pub(crate) acked: u64,
    /// The cache calls in flight ([`TpLeader::forward_cache_call`]): each follower's answer, collected and never compared.
    pub(crate) collect: BTreeMap<u64, Collect>,
}

/// One cache call's answers: registered before the call is sent, so no ack of it can arrive unregistered.
pub(crate) struct Collect {
    name: &'static str,
    /// Follower `r`'s class and text (index `r - 1`), once its ack has arrived.
    answers: Vec<Option<(Class, String)>>,
    /// Milliseconds since `Shared::t0` at which the leader stopped waiting (`None` while it waits): a late ack is consumed silently,
    /// and one that never comes is a stalled follower ([`CacheDeadlines::stall`]).
    abandoned_at: Option<u64>,
}

impl Collect {
    fn complete(&self) -> bool {
        self.answers.iter().all(Option::is_some)
    }
}

/// An ack backlog above this, from one follower, is a protocol violation (acks for calls the leader never made).
pub(crate) const MAX_UNPAIRED_ACKS: usize = 4096;

impl Ledger {
    pub(crate) fn new(followers: usize) -> Ledger {
        Ledger {
            n: followers,
            acks: (0..followers).map(|_| BTreeMap::new()).collect(),
            cursor: vec![0; followers],
            ack_seq: vec![0; followers],
            acked_by: vec![0; followers],
            diverged_by: vec![0; followers],
            ..Ledger::default()
        }
    }

    fn fail(&mut self, why: String) {
        if self.failed.is_none() {
            self.failed = Some(why);
        }
    }

    fn fail_rank(&mut self, rank: usize, why: String) {
        if self.failed.is_none() {
            self.failed = Some(why);
            self.blamed = Some(rank);
        }
    }

    /// Both results of `seq` are known for follower `rank`: compare. `tolerant` calls forgive a non-fatal difference (no call is
    /// tolerant any more: a call may hold collectives, so an asymmetric error is not a local matter).
    #[allow(clippy::too_many_arguments)]
    fn meet(&mut self, rank: usize, seq: u64, name: &'static str, local: Class, tolerant: bool, remote: Class, text: &str) {
        let i = rank - 1;
        self.cursor[i] = self.cursor[i].max(seq);
        self.settled = self.cursor.iter().copied().min().unwrap_or(0);
        if local == remote {
            return;
        }
        self.diverged_by[i] += 1;
        let tail = if text.is_empty() { String::new() } else { format!(": {text}") };
        if remote.is_fatal() || local.is_fatal() {
            self.fail_rank(
                rank,
                format!(
                    "call {seq} ({name}): rank 0 returned {}, rank {rank} returned {}{tail}",
                    local.describe(),
                    remote.describe()
                ),
            );
        } else if tolerant {
            self.tolerated += 1;
        } else {
            self.fail_rank(
                rank,
                format!(
                    "the ranks diverged at call {seq} ({name}): rank 0 returned {}, rank {rank} returned {}{tail}",
                    local.describe(),
                    remote.describe()
                ),
            );
        }
    }

    pub(crate) fn local_done(&mut self, seq: u64, name: &'static str, class: Class, tolerant: bool) {
        let mut met = 0;
        for rank in 1..=self.n {
            if let Some((remote, text)) = self.acks[rank - 1].remove(&seq) {
                self.meet(rank, seq, name, class, tolerant, remote, &text);
                met += 1;
            }
        }
        if met < self.n {
            self.local.insert(seq, LocalResult { name, class, tolerant, met });
        }
    }

    /// Follower `rank` acknowledged command `seq`; `sent` is the highest command the leader has sent.
    pub(crate) fn ack(&mut self, rank: usize, seq: u64, class: Class, text: String, sent: u64) {
        let i = rank - 1;
        self.acked += 1;
        self.acked_by[i] += 1;
        if seq == 0 || seq > sent {
            self.fail_rank(rank, format!("rank {rank} acknowledged call {seq}, which the leader never made"));
            return;
        }
        if seq <= self.cursor[i] {
            self.fail_rank(rank, format!("rank {rank} acknowledged call {seq} twice"));
            return;
        }
        self.ack_seq[i] = self.ack_seq[i].max(seq);
        if let Some(c) = self.collect.get_mut(&seq) {
            // a cache call: the answer is recorded for the caller, never compared with rank 0's (ranks hold different shards, and a
            // recoverable cache failure must fall back cold, not lose the group). Only a fatal class is the group's loss.
            let name = c.name;
            c.answers[i] = Some((class, text.clone()));
            if c.abandoned_at.is_some() && c.complete() {
                self.collect.remove(&seq);
            }
            self.cursor[i] = self.cursor[i].max(seq);
            self.settled = self.cursor.iter().copied().min().unwrap_or(0);
            if class.is_fatal() {
                let tail = if text.is_empty() { String::new() } else { format!(": {text}") };
                self.fail_rank(rank, format!("call {seq} ({name}): rank {rank} returned {}{tail}", class.describe()));
            }
            return;
        }
        if let Some(entry) = self.local.get_mut(&seq) {
            let (name, local, tolerant) = (entry.name, entry.class, entry.tolerant);
            entry.met += 1;
            if entry.met >= self.n {
                self.local.remove(&seq);
            }
            self.meet(rank, seq, name, local, tolerant, class, &text);
        } else if self.acks[i].len() > MAX_UNPAIRED_ACKS {
            self.fail_rank(rank, format!("rank {rank} acknowledged call {seq}, which the leader never made"));
        } else {
            self.acks[i].insert(seq, (class, text));
        }
    }
}

/// What the leader knows of one follower, shared by its reader thread, its writer thread, the watchdog and the status snapshot.
struct PeerShared {
    rank: usize,
    /// Milliseconds since `Shared::t0` of the last frame received from this rank, and of the last frame written to it.
    last_rx: AtomicU64,
    last_tx: AtomicU64,
    /// The highest command sequence number this rank reported applied (pongs).
    applied: AtomicU64,
    /// The last ping's round trip in microseconds (0: none yet) and the ping in flight: its nonce and send time.
    rtt_us: AtomicU64,
    ping_nonce: AtomicU64,
    ping_at_us: AtomicU64,
    /// Frames and bytes handed to this rank's writer and not yet written.
    queued_frames: AtomicU64,
    queued_bytes: AtomicU64,
    /// Bytes written to it (pings excluded).
    bytes: AtomicU64,
    reader_done: AtomicBool,
    /// The link has been closed (by the group's failure or by this rank's end of stream).
    closed: AtomicBool,
    /// Why this rank left, when it said or when its stream ended.
    left: Mutex<Option<String>>,
}

impl PeerShared {
    fn new(rank: usize) -> PeerShared {
        PeerShared {
            rank,
            last_rx: AtomicU64::new(0),
            last_tx: AtomicU64::new(0),
            applied: AtomicU64::new(0),
            rtt_us: AtomicU64::new(0),
            ping_nonce: AtomicU64::new(0),
            ping_at_us: AtomicU64::new(0),
            queued_frames: AtomicU64::new(0),
            queued_bytes: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            reader_done: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            left: Mutex::new(None),
        }
    }
}

/// What the leader's threads and the engine thread share. A follower's loop uses the same type with no peers.
struct Shared {
    t0: Instant,
    ledger: Mutex<Ledger>,
    cv: Condvar,
    /// Milliseconds since `t0` of the last frame received (a follower's view of rank 0; the leader keeps one per rank in `peers`).
    last_rx: AtomicU64,
    /// Milliseconds since `t0` plus one at which the local call in flight began (0: none).
    call_since: AtomicU64,
    /// A follower: microseconds since `t0` plus one of the last COMMAND frame read or the last command acknowledged (0: none yet). A
    /// pong-only link never refreshes it, so an idle group's receive path parks once `FollowerConfig::spin` has passed.
    last_active_us: AtomicU64,
    /// A follower: the highest command sequence number it has applied (the pongs carry it).
    peer_applied: AtomicU64,
    stop: AtomicBool,
    /// The leader: the highest command number handed to the writers.
    sent_seq: AtomicU64,
    lost_at: AtomicU64,
    /// The side effects of a lost group have run (once).
    lost_once: AtomicBool,
    /// Run when the group is declared lost (the communicator's abort, when TP-N provides one).
    abort_hook: Arc<Mutex<Option<AbortHook>>>,
    /// This rank and how long after the group is declared lost the process aborts whatever its threads are doing (`exit_backstop`).
    backstop: Mutex<Option<(usize, Duration)>>,
    /// The communicator's health probe ([`HealthProbe`]), when one was registered.
    health: Mutex<Option<HealthProbe>>,
    /// When the watchdog last polled the probe (ms since `t0`).
    health_at: AtomicU64,
    /// The leader's followers (rank `r` is `peers[r - 1]`).
    peers: Vec<PeerShared>,
    /// Closes every follower's link (shutdown of both halves): wakes whoever is blocked on it.
    closers: Mutex<Vec<Arc<dyn Fn() + Send + Sync>>>,
    /// [`CacheDeadlines`] in milliseconds, read by the engine thread (the ack wait) and the watchdog (the stall deadline).
    cache_ack_ms: AtomicU64,
    cache_stall_ms: AtomicU64,
}

/// The process-level backstop of a lost group: after `after`, whatever the other threads are doing (a model call stuck in a collective, a
/// destructor freeing device memory behind a kernel that waits for a peer that is gone), say so and abort. A rank whose peer is gone has
/// nothing left to serve: the normal path (the engine fails its requests by name and the process ends) takes a couple of seconds, and
/// a supervisor restarts both ranks. Measured on two nodes before this existed (section 12): a SIGKILLed rank 1 left rank 0 silent for the
/// server's whole 60 s progress budget, and a SIGKILLed rank 0 mid-generation left rank 1 alive for ever.
fn exit_backstop(rank: usize, after: Duration, why: String) {
    let _ = std::thread::Builder::new().name("tp-exit-backstop".into()).spawn(move || {
        std::thread::sleep(after);
        eprintln!(
            "rexl3 level=fatal event=tp_group_lost_exit rank={rank} after_s={} reason=\"{why}\" (the process did not end by itself; exiting 70)",
            after.as_secs()
        );
        // `_exit`: no atexit handlers and no destructors (the CUDA and NCCL teardown is what is stuck), no core dump, the status the
        // server uses for a fatal engine error
        // SAFETY: `_exit` takes no pointers and does not return.
        unsafe { libc::_exit(70) }
    });
}

impl Shared {
    fn new() -> Shared {
        Shared::with_followers(0)
    }

    /// The leader's state for `n` followers.
    fn with_followers(n: usize) -> Shared {
        Shared {
            t0: Instant::now(),
            ledger: Mutex::new(Ledger::new(n)),
            cv: Condvar::new(),
            last_rx: AtomicU64::new(0),
            call_since: AtomicU64::new(0),
            last_active_us: AtomicU64::new(0),
            peer_applied: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            sent_seq: AtomicU64::new(0),
            lost_at: AtomicU64::new(0),
            lost_once: AtomicBool::new(false),
            abort_hook: Arc::new(Mutex::new(None)),
            backstop: Mutex::new(None),
            health: Mutex::new(None),
            health_at: AtomicU64::new(0),
            peers: (1..=n).map(PeerShared::new).collect(),
            closers: Mutex::new(Vec::new()),
            cache_ack_ms: AtomicU64::new(CacheDeadlines::default().ack.as_millis() as u64),
            cache_stall_ms: AtomicU64::new(CacheDeadlines::default().stall.as_millis() as u64),
        }
    }

    /// The oldest cache call whose answers the leader stopped waiting for and that a follower has still not answered after `stall_ms`:
    /// `(rank, seq, name, ms since it was abandoned)`. A follower that keeps answering pings but never this call is stuck, not slow.
    fn stalled_cache_call(&self, stall_ms: u64) -> Option<(usize, u64, &'static str, u64)> {
        let now = self.now_ms();
        let l = self.ledger.lock().unwrap_or_else(|p| p.into_inner());
        l.collect.iter().find_map(|(seq, c)| {
            let at = c.abandoned_at?;
            let age = now.saturating_sub(at);
            if age < stall_ms {
                return None;
            }
            let rank = c.answers.iter().position(Option::is_none)? + 1;
            Some((rank, *seq, c.name, age))
        })
    }

    fn now_ms(&self) -> u64 {
        self.t0.elapsed().as_millis() as u64
    }

    /// Polls the health probe when `every` has passed since the last poll: `Some(why)` is an asynchronous failure of the collective.
    fn poll_health(&self, every: Duration) -> Option<String> {
        let now = self.now_ms();
        if now.saturating_sub(self.health_at.load(Ordering::Relaxed)) < every.as_millis() as u64 {
            return None;
        }
        self.health_at.store(now, Ordering::Relaxed);
        let probe = self.health.lock().unwrap_or_else(|p| p.into_inner()).clone()?;
        probe()
    }

    fn now_us(&self) -> u64 {
        self.t0.elapsed().as_micros() as u64
    }

    /// Records command activity (a command frame read, a command acknowledged) for the spin-then-park receive path.
    fn active(&self) {
        self.last_active_us.store(self.now_us() + 1, Ordering::Relaxed);
    }

    /// Whether the group has been busy within the last `window_us`: the receive path polls while it is, and parks when it is not.
    fn busy_within(&self, window_us: u64) -> bool {
        let a = self.last_active_us.load(Ordering::Relaxed);
        a != 0 && self.now_us().saturating_sub(a - 1) < window_us
    }

    fn touch(&self) {
        self.last_rx.store(self.now_ms(), Ordering::Relaxed);
    }

    fn failure(&self) -> Option<String> {
        self.ledger.lock().unwrap_or_else(|p| p.into_inner()).failed.clone()
    }

    /// The latched failure and the rank it is blamed on, when it is one.
    fn failure_and_blame(&self) -> (Option<String>, Option<usize>) {
        let l = self.ledger.lock().unwrap_or_else(|p| p.into_inner());
        (l.failed.clone(), l.blamed)
    }

    /// Declares the group lost (once): wakes the waiters, closes every link and runs the abort hook.
    fn fail(&self, why: impl Into<String>) {
        self.fail_for(None, why);
    }

    /// [`Shared::fail`] blaming follower `rank`.
    fn fail_rank(&self, rank: usize, why: impl Into<String>) {
        self.fail_for(Some(rank), why);
    }

    fn fail_for(&self, rank: Option<usize>, why: impl Into<String>) {
        {
            let mut l = self.ledger.lock().unwrap_or_else(|p| p.into_inner());
            match rank {
                Some(r) => l.fail_rank(r, why.into()),
                None => l.fail(why.into()),
            }
        }
        self.after_failure();
    }

    /// An ack of command `seq` from follower `rank` (its reader thread): paired with the leader's result and compared.
    fn ack_from(&self, rank: usize, seq: u64, class: Class, text: String) {
        {
            let sent = self.sent_seq.load(Ordering::Acquire);
            let mut l = self.ledger.lock().unwrap_or_else(|p| p.into_inner());
            l.ack(rank, seq, class, text, sent);
        }
        self.after_failure();
        self.cv.notify_all();
    }

    /// The leader's own result of command `seq`, paired with whatever acks have arrived.
    fn local_done(&self, seq: u64, name: &'static str, class: Class, tolerant: bool) {
        {
            let mut l = self.ledger.lock().unwrap_or_else(|p| p.into_inner());
            l.local_done(seq, name, class, tolerant);
        }
        self.after_failure();
        self.cv.notify_all();
    }

    /// Runs the side effects of the group's loss, once, if the ledger has failed: the log line, the exit backstop, EVERY follower's link
    /// closed without a Shutdown frame (a healthy follower blocked in a collective with a lost sibling sees the end of the stream
    /// without a goodbye and aborts its communicator, instead of waiting out its own silence watchdog), and the communicator's abort
    /// on a thread of its own (it has been seen to block, section 12.3 B2, and nothing here may wait for it).
    fn after_failure(&self) {
        let Some(why) = self.failure() else { return };
        if self.lost_once.swap(true, Ordering::AcqRel) {
            return;
        }
        crate::tp_status::note_group_lost();
        eprintln!("rexl3 level=error event=tp_group_lost reason=\"{why}\"");
        self.lost_at.store(self.now_ms() + 1, Ordering::Relaxed);
        if let Some((rank, after)) = *self.backstop.lock().unwrap_or_else(|p| p.into_inner()) {
            exit_backstop(rank, after, why);
        }
        self.close_links();
        let hook = self.abort_hook.clone();
        if hook.lock().unwrap_or_else(|p| p.into_inner()).is_some() {
            let _ = std::thread::Builder::new().name("tp-abort-hook".into()).spawn(move || {
                if let Some(h) = hook.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
                    h();
                }
            });
        }
        self.cv.notify_all();
    }

    /// Closes every follower's link (idempotent).
    fn close_links(&self) {
        for p in &self.peers {
            p.closed.store(true, Ordering::Release);
        }
        for c in self.closers.lock().unwrap_or_else(|p| p.into_inner()).iter() {
            c();
        }
    }
}
// ------------------------------------------------------------------------------------------------------------------- the leader

/// Counters of a leader, for the boot log and the tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LeaderStats {
    pub commands: u64,
    /// Bytes written to all the followers (a command goes to each of them).
    pub bytes: u64,
    /// Acks received from all the followers.
    pub acked: u64,
    /// Non-fatal differences forgiven (none: every forwarded call is group-fatal).
    pub tolerated: u64,
}

// ------------------------------------------------------------------------------------------------------------------- cache calls

/// How long the leader waits for the followers' answers to a cache call ([`TpLeader::forward_cache_call`]).
///
/// A cache call (a prefix snapshot, restore or release) is a local copy on every rank with no collective in it, so the followers run it
/// while the leader runs its own: a healthy follower answers within a round trip of the leader's own call. `ack` is the slack over the
/// leader's own duration (a 1M-token snapshot copies about a GiB per rank; the follower is given as long as the leader took, plus
/// this). Past it the call FAILS (the caller falls back to a cold prefill on every rank) and the late answer is consumed silently. A
/// follower that has still not answered `stall` after that is stuck (its reader still answers pings, so the peer timeout never fires):
/// the group is lost by name, so a cold prefill blocked in a collective with it returns instead of waiting for ever.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheDeadlines {
    pub ack: Duration,
    pub stall: Duration,
}

impl Default for CacheDeadlines {
    fn default() -> Self {
        CacheDeadlines { ack: Duration::from_secs(2), stall: Duration::from_secs(30) }
    }
}

/// What one follower answered to a cache call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RankAnswer {
    /// The call succeeded there, with this digest.
    Ok(u64),
    /// A recoverable error: its kind and text. (A fatal one loses the group instead.)
    Err(String),
    /// No answer within the wait (or the leader did not wait).
    Silent,
}

impl std::fmt::Display for RankAnswer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RankAnswer::Ok(d) => write!(f, "ok (digest {d:016x})"),
            RankAnswer::Err(e) => write!(f, "error: {e}"),
            RankAnswer::Silent => write!(f, "no answer"),
        }
    }
}

/// The result of a cache call on every rank: rank 0's own result and each follower's answer (follower `r` at index `r - 1`).
#[derive(Debug)]
pub struct CacheOutcome<T> {
    /// The call's sequence number (the handle every rank knows a snapshot by).
    pub seq: u64,
    pub local: Result<T, ModelError>,
    pub followers: Vec<RankAnswer>,
}

/// A snapshot's digest on a rank that did not take it (a memory guard or a full pool skipped it). Every digest of a snapshot that was
/// taken is odd ([`taken_digest`]), so the two can never be confused.
pub const SNAPSHOT_SKIPPED: u64 = 0;

/// The digest a rank reports for a snapshot it took (or a restore it ran): the content digest of the state, forced odd.
pub fn taken_digest(content: u64) -> u64 {
    content | 1
}

/// The group's book of the prefix snapshots every rank holds, kept by rank 0: per snapshot (by the sequence number of the call that took
/// it, the handle every rank knows), each rank's content digest at the time it was taken. Ranks hold different shards, so a rank's
/// restore is compared with ITS OWN digest at snapshot time, never with another rank's.
///
/// An entry exists only when every rank took the snapshot ([`SnapshotBook::settle_snapshot`]); a restore is verified rank by rank
/// against it ([`SnapshotBook::verify_restore`]). A stale or empty restore that returns Ok is caught here, per request, by the digest
/// of what the rank actually restored.
#[derive(Clone, Debug, Default)]
pub struct SnapshotBook {
    /// `digests[0]` is rank 0's, `digests[r]` follower `r`'s.
    by_call: BTreeMap<u64, Vec<u64>>,
}

impl SnapshotBook {
    /// Records the snapshot taken by call `of`: `rank0` is rank 0's digest ([`SNAPSHOT_SKIPPED`] when it took none), `followers` the
    /// followers' answers. `Ok` only when every rank took it; otherwise nothing is recorded and the reason names the ranks, and the
    /// caller must release the snapshot on every rank (the entry is never used).
    pub fn settle_snapshot(&mut self, of: u64, rank0: u64, followers: &[RankAnswer]) -> Result<(), String> {
        let mut digests = vec![rank0];
        let mut why = Vec::new();
        if rank0 == SNAPSHOT_SKIPPED {
            why.push("rank 0 took no snapshot".to_string());
        }
        for (i, a) in followers.iter().enumerate() {
            match a {
                RankAnswer::Ok(d) if *d != SNAPSHOT_SKIPPED => digests.push(*d),
                RankAnswer::Ok(_) => why.push(format!("rank {} took no snapshot (memory guard or full pool)", i + 1)),
                other => why.push(format!("rank {}: {other}", i + 1)),
            }
        }
        if !why.is_empty() {
            return Err(why.join("; "));
        }
        self.by_call.insert(of, digests);
        Ok(())
    }

    /// Whether every rank holds the snapshot taken by call `of`.
    pub fn holds(&self, of: u64) -> bool {
        self.by_call.contains_key(&of)
    }

    /// Verifies a restore of the snapshot taken by call `of`: every rank must have restored exactly what it snapshotted (its digest of
    /// the restored state equals its own digest at snapshot time). `rank0` is rank 0's result, `followers` the followers' answers.
    pub fn verify_restore(&self, of: u64, rank0: Result<u64, String>, followers: &[RankAnswer]) -> Result<(), String> {
        let want = self.by_call.get(&of).ok_or_else(|| format!("snapshot call {of} is not held by every rank"))?;
        if followers.len() + 1 != want.len() {
            return Err(format!("{} ranks answered a snapshot taken on {}", followers.len() + 1, want.len()));
        }
        let mut why = Vec::new();
        match rank0 {
            Ok(d) if d == want[0] => {}
            Ok(d) => why.push(format!("rank 0 restored digest {d:016x}, its snapshot was {:016x}", want[0])),
            Err(e) => why.push(format!("rank 0: error: {e}")),
        }
        for (i, a) in followers.iter().enumerate() {
            let w = want[i + 1];
            match a {
                RankAnswer::Ok(d) if *d == w => {}
                RankAnswer::Ok(d) => why.push(format!("rank {} restored digest {d:016x}, its snapshot was {w:016x}", i + 1)),
                other => why.push(format!("rank {}: {other}", i + 1)),
            }
        }
        if why.is_empty() {
            Ok(())
        } else {
            Err(why.join("; "))
        }
    }

    /// Forgets the snapshot taken by call `of` (released): `true` when it was held.
    pub fn forget(&mut self, of: u64) -> bool {
        self.by_call.remove(&of).is_some()
    }

    /// Snapshots held by every rank.
    pub fn len(&self) -> usize {
        self.by_call.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_call.is_empty()
    }
}

/// One frame for a follower's writer: encoded once by the engine thread and shared by every follower's queue.
struct Outgoing {
    frame: Arc<Vec<u8>>,
    /// What it is, for the message when the write fails.
    what: &'static str,
    seq: u64,
}

/// A follower's backlog above either bound is a rank that is not draining its link (a stalled process behind a live socket): the group
/// is lost and the message names the rank. Far above anything the lockstep produces (a collective is a barrier, so a healthy follower is
/// at most a few commands behind); the socket's write timeout is the other detector, for a write that does not drain at all.
const MAX_QUEUED_FRAMES: u64 = 1 << 16;
const MAX_QUEUED_BYTES: u64 = 256 << 20;

/// The leader's tx half of a link: one `write_all` per frame is made under this lock, so frames never interleave.
type TxHalf = Arc<Mutex<Box<dyn Write + Send>>>;

/// How the leader's frames reach one follower's socket.
enum Wire {
    /// Through this follower's writer thread (`None` once the leader is shutting down). Used when the group has more than one follower:
    /// a socket that does not drain then delays only its own rank, and never the commands of the others.
    Queued(Option<mpsc::Sender<Outgoing>>),
    /// Written by the calling thread itself under the link's tx lock: no queue, no second thread, no wake-up on the path of every
    /// command. Used when the group has exactly ONE follower (two ranks), where there is no other rank to head-of-line block: it is
    /// what the two-rank leader did before the writer threads, and the lost peer is found the same way (the write timeout).
    Inline {
        tx: TxHalf,
        /// Ends this side's stream after what was written (the shutdown path: the Shutdown frame, then this, in order).
        close_write: Arc<dyn Fn() + Send + Sync>,
    },
}

/// The leader's end of one follower.
struct PeerLink {
    wire: Wire,
}

/// How the watchdog's pings reach a follower: the same two ways, through the same lock or queue as the commands.
enum PingPath {
    Queue(mpsc::Sender<Outgoing>),
    Inline(TxHalf),
}

/// The leader's runtime: `R` (rank 0's model) with every state-changing call forwarded to every follower first. See the module docs.
///
/// The core calls are methods of this type (`begin_slot`, `end_slot`, `prefill_chunk`, `window_candidates`, `commit`, `flush`); a family's
/// own calls go through [`TpLeader::forward_call`] and [`TpLeader::forward_unit`], which a family's implementation of its own runtime
/// trait for `TpLeader<..>` uses.
pub struct TpLeader<R: LockstepRuntime> {
    inner: R,
    shared: Arc<Shared>,
    /// Rank `r`'s link is `peers[r - 1]`.
    peers: Vec<PeerLink>,
    seq: u64,
    timeouts: TpTimeouts,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl<R: LockstepRuntime> TpLeader<R> {
    /// Starts the leader over `link`, the one follower of a two-rank group (the boot handshake is done: [`boot_leader`]).
    pub fn lead(inner: R, link: Link<R::Call>, timeouts: TpTimeouts) -> TpLeader<R> {
        TpLeader::lead_group(inner, vec![link], timeouts)
    }

    /// Starts the leader over the links of ranks `1..=links.len()` (`links[r - 1]` is rank `r`'s; the boot handshake is done:
    /// [`boot_leader_group`]). Spawns one reader thread a follower and the watchdog; with more than one follower also a writer thread
    /// a follower (a command is encoded once and queued to each), with exactly one the engine thread writes its frames itself.
    pub fn lead_group(inner: R, links: Vec<Link<R::Call>>, timeouts: TpTimeouts) -> TpLeader<R> {
        assert!(!links.is_empty(), "a tensor-parallel leader needs at least one follower");
        crate::hosttrace::start_from_env("L");
        let n = links.len();
        let shared = Arc::new(Shared::with_followers(n));
        *shared.backstop.lock().unwrap_or_else(|p| p.into_inner()) = timeouts.hang_exit.map(|d| (0, d));
        let now = shared.now_ms();
        for p in &shared.peers {
            p.last_rx.store(now, Ordering::Relaxed);
            p.last_tx.store(now, Ordering::Relaxed);
        }
        let mut threads = Vec::new();
        let mut peers = Vec::with_capacity(n);
        let mut ping_paths = Vec::with_capacity(n);
        // exactly one follower: its frames are written inline by the engine thread (see `Wire::Inline`)
        let inline = n == 1;
        // every link's closer is in place before any thread runs: the first thing a reader may do is lose the group, and losing it
        // closes every link
        let mut parts = Vec::with_capacity(n);
        let mut closers: Vec<Arc<dyn Fn() + Send + Sync>> = Vec::with_capacity(n);
        for link in links {
            let Link { rx, tx, close, close_write, set_rx_timeout, .. } = link;
            set_rx_timeout(None);
            closers.push(close);
            parts.push((rx, tx, close_write));
        }
        *shared.closers.lock().unwrap_or_else(|p| p.into_inner()) = closers;
        for (i, (rx, tx, close_write)) in parts.into_iter().enumerate() {
            let rank = i + 1;
            let tx: TxHalf = Arc::new(Mutex::new(tx));
            // the reader: acks settle against the leader's own results; pongs and everything else are proof of life
            {
                let shared = shared.clone();
                threads.push(
                    std::thread::Builder::new()
                        .name(format!("tp-leader-rx-{rank}"))
                        .spawn(move || leader_reader::<R::Call>(rank, rx, shared))
                        .expect("spawn the leader's reader"),
                );
            }
            if inline {
                ping_paths.push(PingPath::Inline(tx.clone()));
                peers.push(PeerLink { wire: Wire::Inline { tx, close_write } });
            } else {
                // the writer: this follower's queue, in order, on a thread of its own, so that a socket that does not drain delays
                // only its own rank (and then loses the group at the write timeout) and never the commands of the others
                let (qtx, qrx) = mpsc::channel::<Outgoing>();
                let shared = shared.clone();
                threads.push(
                    std::thread::Builder::new()
                        .name(format!("tp-leader-tx-{rank}"))
                        .spawn(move || leader_writer(rank, qrx, tx, shared, close_write))
                        .expect("spawn the leader's writer"),
                );
                ping_paths.push(PingPath::Queue(qtx.clone()));
                peers.push(PeerLink { wire: Wire::Queued(Some(qtx)) });
            }
        }
        // the watchdog: pings each idle follower, and finds a peer that went silent or a call that can no longer finish
        {
            let (shared, t) = (shared.clone(), timeouts.clone());
            threads.push(
                std::thread::Builder::new()
                    .name("tp-leader-watch".into())
                    .spawn(move || leader_watchdog::<R::Call>(shared, ping_paths, t))
                    .expect("spawn the leader's watchdog"),
            );
        }
        // the group's status, for the server's /health and /metrics (the latest leader of the process is the one reported)
        {
            let shared = shared.clone();
            crate::tp_status::register(Arc::new(move || group_status_of(&shared)));
        }
        TpLeader { inner, shared, peers, seq: 0, timeouts, threads }
    }

    /// Registers what to run when the group is declared lost: the communicator's abort, so a rank stuck in a collective returns.
    pub fn on_group_lost(&self, hook: AbortHook) {
        *self.shared.abort_hook.lock().unwrap_or_else(|p| p.into_inner()) = Some(hook);
    }

    /// Registers the communicator's health probe: the watchdog polls it every [`TpTimeouts::health_poll`] and loses the group with its
    /// text when it reports an asynchronous failure (the collective fabric failed while the command link may still be healthy).
    pub fn on_comm_health(&self, probe: HealthProbe) {
        *self.shared.health.lock().unwrap_or_else(|p| p.into_inner()) = Some(probe);
    }

    /// Rank 0's runtime (the family's reads of its own state go through it).
    pub fn inner(&self) -> &R {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    /// The group's ranks (the leader and its followers).
    pub fn ranks(&self) -> usize {
        self.peers.len() + 1
    }

    pub fn stats(&self) -> LeaderStats {
        let l = self.shared.ledger.lock().unwrap_or_else(|p| p.into_inner());
        LeaderStats {
            commands: self.seq,
            bytes: self.shared.peers.iter().map(|p| p.bytes.load(Ordering::Relaxed)).sum(),
            acked: l.acked,
            tolerated: l.tolerated,
        }
    }

    /// A snapshot of the group: per follower its state, acks, lag and ping; for the group the commands sent and the loss, if any.
    pub fn group_status(&self) -> crate::tp_status::GroupStatus {
        group_status_of(&self.shared)
    }

    /// The same snapshot as a function another thread can call while this leader is in use (what the server reads through
    /// [`crate::tp_status`]).
    pub fn status_source(&self) -> Arc<dyn Fn() -> crate::tp_status::GroupStatus + Send + Sync> {
        let shared = self.shared.clone();
        Arc::new(move || group_status_of(&shared))
    }

    /// The text the group failed with, if it has.
    pub fn failure(&self) -> Option<String> {
        self.shared.failure()
    }

    /// Fails the group from the leader's side (a local fatal error): every follower's link is closed WITHOUT a Shutdown, so a rank
    /// stuck in a collective with this one sees the end of the stream and aborts its communicator (the leader's own is aborted by the
    /// same hook).
    pub fn fail(&mut self, why: impl Into<String>) {
        self.shared.fail(why);
    }

    /// The latched failure as the error every later call returns.
    pub fn guard(&self) -> Result<(), ModelError> {
        match self.shared.failure() {
            Some(why) => Err(ModelError::StateLost(format!("tensor-parallel group lost: {why}"))),
            None => Ok(()),
        }
    }

    /// The sequence number of the command last sent (0: none).
    pub fn last_seq(&self) -> u64 {
        self.seq
    }

    /// Encodes `m` once and gets the frame to every follower, in order: queued to its writer (more than one follower) or written
    /// here, under the link's lock (exactly one). An error names the rank whose link refused it (`0`: the frame itself).
    fn enqueue_all(&self, seq: u64, m: &Msg<R::Call>) -> Result<(), (usize, String)> {
        let frame = Arc::new(m.encode(seq));
        let n = frame.len();
        if n - 4 > MAX_FRAME {
            return Err((0, WireError::TooLarge(n - 4).to_string()));
        }
        let what = m.name();
        for (i, p) in self.peers.iter().enumerate() {
            let rank = i + 1;
            let ps = &self.shared.peers[i];
            match &p.wire {
                Wire::Queued(q) => {
                    let Some(q) = q else { return Err((rank, "the link is closed".into())) };
                    if ps.queued_frames.load(Ordering::Relaxed) > MAX_QUEUED_FRAMES
                        || ps.queued_bytes.load(Ordering::Relaxed) > MAX_QUEUED_BYTES
                    {
                        return Err((
                            rank,
                            format!(
                                "rank {rank} is not draining its link: {} commands ({} bytes) are queued for it",
                                ps.queued_frames.load(Ordering::Relaxed),
                                ps.queued_bytes.load(Ordering::Relaxed)
                            ),
                        ));
                    }
                    ps.queued_frames.fetch_add(1, Ordering::Relaxed);
                    ps.queued_bytes.fetch_add(n as u64, Ordering::Relaxed);
                    if q.send(Outgoing { frame: frame.clone(), what, seq }).is_err() {
                        ps.queued_frames.fetch_sub(1, Ordering::Relaxed);
                        ps.queued_bytes.fetch_sub(n as u64, Ordering::Relaxed);
                        return Err((rank, "its writer has ended".into()));
                    }
                }
                Wire::Inline { tx, .. } => {
                    // one `write_all` of the whole frame under the lock (the watchdog's pings take the same lock); a write that does
                    // not drain fails at the socket's write timeout, as a queued one does in its writer thread
                    let r = {
                        let mut w = tx.lock().unwrap_or_else(|p| p.into_inner());
                        w.write_all(&frame).and_then(|_| w.flush())
                    };
                    match r {
                        Ok(()) => {
                            ps.bytes.fetch_add(n as u64, Ordering::Relaxed);
                            ps.last_tx.store(self.shared.now_ms(), Ordering::Relaxed);
                        }
                        Err(e) => {
                            if !self.shared.stop.load(Ordering::Relaxed) {
                                *ps.left.lock().unwrap_or_else(|p| p.into_inner()) = Some(format!("a write failed: {e}"));
                            }
                            return Err((rank, e.to_string()));
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Numbers and sends one command to every follower. A send that fails loses the group (a follower cannot catch up).
    fn send(&mut self, m: Msg<R::Call>) -> Result<u64, ModelError> {
        self.send_as(m, None)
    }

    /// [`TpLeader::send`]; with `collect` the command is a cache call whose answers are collected under its name, registered before the
    /// frame can reach any follower.
    fn send_as(&mut self, m: Msg<R::Call>, collect: Option<&'static str>) -> Result<u64, ModelError> {
        self.guard()?;
        self.seq += 1;
        let seq = self.seq;
        if let Some(name) = collect {
            let n = self.peers.len();
            let c = Collect { name, answers: vec![None; n], abandoned_at: None };
            self.shared.ledger.lock().unwrap_or_else(|p| p.into_inner()).collect.insert(seq, c);
        }
        self.shared.sent_seq.store(seq, Ordering::Release);
        crate::hosttrace::mark("send_begin", seq);
        crate::hosttrace::mark(m.name(), seq);
        let sent = self.enqueue_all(seq, &m);
        crate::hosttrace::mark("send_end", seq);
        if let Err((rank, e)) = sent {
            let why = if rank == 0 {
                format!("sending {} (call {seq}) failed: {e}", m.name())
            } else {
                format!("sending {} (call {seq}) to rank {rank} failed: {e}", m.name())
            };
            if rank == 0 {
                self.shared.fail(why.clone());
            } else {
                self.shared.fail_rank(rank, why.clone());
            }
            // the first reason wins: a write that failed because another thread had already lost the group reports THAT
            return Err(lost_group(self.shared.failure().unwrap_or(why)));
        }
        Ok(seq)
    }

    /// Runs the local half of a command, with the call marked in flight for the watchdog.
    pub fn run_local<T>(&mut self, f: impl FnOnce(&mut R) -> T) -> T {
        self.shared.call_since.store(self.shared.now_ms() + 1, Ordering::Relaxed);
        crate::hosttrace::mark("local_begin", self.seq);
        let r = f(&mut self.inner);
        crate::hosttrace::mark("local_end", self.seq);
        self.shared.call_since.store(0, Ordering::Relaxed);
        r
    }

    /// Records the local result of command `seq` and returns it; the group's failure, if it has failed meanwhile, replaces an `Ok`.
    fn finish<T>(
        &mut self,
        seq: u64,
        name: &'static str,
        r: Result<T, ModelError>,
        aux: impl FnOnce(&T) -> u64,
        tolerant: bool,
    ) -> Result<T, ModelError> {
        // what was latched BEFORE this result was recorded: recording it can itself fail the group (a result that differs from an
        // ack already in), and that failure is this call's own, not a lost peer
        let (failure, blamed) = if r.is_err() { self.shared.failure_and_blame() } else { (None, None) };
        let class = Class::of(&r, aux);
        self.shared.local_done(seq, name, class, tolerant);
        match r {
            Err(e) => {
                // a collective cut short by a loss that was latched before it (the abort hook ran): say so, by rank, not with the
                // bare NCCL code. The class above was taken from the original error, so the ack comparison does not move.
                let e = lost_peer_error(e, failure.as_deref(), blamed, &format!("{name} (call {seq})"));
                if e.is_fatal() {
                    self.fail(format!("rank 0 failed {name} (call {seq}): {e}"));
                }
                Err(e)
            }
            Ok(v) => {
                self.guard()?;
                Ok(v)
            }
        }
    }

    /// [`TpLeader::finish`] for a call that returns nothing: its class is `Ok`, and the group's failure is latched for the next call.
    fn finish_unit(&mut self, seq: u64, name: &'static str) {
        self.shared.local_done(seq, name, Class::Ok(0), false);
    }

    /// Waits until every follower's ack of `seq` has been compared with the local result, or the group fails, or `timeout`.
    pub fn wait_settled(&self, seq: u64, timeout: Duration) -> Result<(), ModelError> {
        let deadline = Instant::now() + timeout;
        let mut l = self.shared.ledger.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(why) = &l.failed {
                return Err(lost_group(why.clone()));
            }
            if l.settled >= seq {
                return Ok(());
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                let behind: Vec<usize> = (1..=l.n).filter(|r| l.cursor[r - 1] < seq).collect();
                drop(l);
                let list = behind.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", ");
                let why = if behind.len() == 1 {
                    format!("rank {list} did not acknowledge call {seq} within {timeout:?}")
                } else {
                    format!("ranks {list} did not acknowledge call {seq} within {timeout:?}")
                };
                match behind.first() {
                    Some(r) if behind.len() == 1 => self.shared.fail_rank(*r, why.clone()),
                    _ => self.shared.fail(why.clone()),
                }
                return Err(lost_group(why));
            }
            l = self.shared.cv.wait_timeout(l, left.min(Duration::from_millis(100))).unwrap_or_else(|p| p.into_inner()).0;
        }
    }

    /// [`TpLeader::wait_settled`] under the group's barrier timeout (what boot-time calls wait with).
    pub fn settle(&self, seq: u64) -> Result<(), ModelError> {
        self.wait_settled(seq, self.timeouts.barrier)
    }

    /// Sends a barrier and waits until every follower has applied everything before it (and agreed with every result).
    pub fn barrier(&mut self) -> Result<(), ModelError> {
        let seq = self.send(Msg::Barrier)?;
        self.finish_unit(seq, "barrier");
        self.wait_settled(seq, self.timeouts.barrier)
    }

    /// Forwards a family call that returns a value: numbers it, sends it to every follower, applies it locally with `local` (given
    /// this rank's runtime and the call's sequence number), and settles the local result. `digest` is the aux the followers' acks
    /// must carry for it (`|_| 0`: only the result class is compared). Returns the local result; a fatal local error loses the group.
    /// The send comes first, so a collective inside `local` finds its peers already on their way into the same call.
    pub fn forward_call<T>(
        &mut self,
        call: R::Call,
        local: impl FnOnce(&mut R, u64) -> Result<T, ModelError>,
        digest: impl FnOnce(&T) -> u64,
    ) -> Result<T, ModelError> {
        let name = call.name();
        let seq = self.send(Msg::Call(call))?;
        let r = self.run_local(|m| local(m, seq));
        self.finish(seq, name, r, digest, false)
    }

    /// The deadlines of the cache calls ([`CacheDeadlines`]).
    pub fn cache_deadlines(&self) -> CacheDeadlines {
        CacheDeadlines {
            ack: Duration::from_millis(self.shared.cache_ack_ms.load(Ordering::Relaxed)),
            stall: Duration::from_millis(self.shared.cache_stall_ms.load(Ordering::Relaxed)),
        }
    }

    /// Sets the deadlines of the cache calls (the defaults serve; tests shorten them).
    pub fn set_cache_deadlines(&self, d: CacheDeadlines) {
        self.shared.cache_ack_ms.store(d.ack.as_millis() as u64, Ordering::Relaxed);
        self.shared.cache_stall_ms.store(d.stall.as_millis() as u64, Ordering::Relaxed);
    }

    /// Forwards a CACHE call (a prefix snapshot, restore or release): numbered and sent like any call, applied locally with `local`, but
    /// each follower's answer is COLLECTED for the caller instead of compared with rank 0's, and no recoverable difference loses the group
    /// (a cache call holds no collective, so ranks that answer differently are still in lockstep; the caller decides, and a restore that
    /// is not verified on every rank falls back to a cold prefill on every rank).
    ///
    /// With `wait`, the leader waits for every follower's answer for its own call's duration plus [`CacheDeadlines::ack`]; a follower
    /// that has not answered by then is [`RankAnswer::Silent`] (its late answer is consumed silently, and one that never comes within
    /// [`CacheDeadlines::stall`] loses the group). Without `wait` every answer is `Silent` (a release: nothing to decide).
    ///
    /// A fatal error on any rank still loses the group (a sticky device error is not a cache miss); `Err` is then the group's loss.
    pub fn forward_cache_call<T>(
        &mut self,
        call: R::Call,
        local: impl FnOnce(&mut R, u64) -> Result<T, ModelError>,
        wait: bool,
    ) -> Result<CacheOutcome<T>, ModelError> {
        let name = call.name();
        let seq = self.send_as(Msg::Call(call), Some(name))?;
        let t0 = Instant::now();
        let r = self.run_local(|m| local(m, seq));
        let took = t0.elapsed();
        let r = match r {
            Err(e) if e.is_fatal() => {
                self.abandon_collect(seq);
                self.fail(format!("rank 0 failed {name} (call {seq}): {e}"));
                return Err(e);
            }
            r => r,
        };
        let followers = if wait {
            self.wait_collect(seq, took + self.cache_deadlines().ack)?
        } else {
            self.abandon_collect(seq);
            vec![RankAnswer::Silent; self.peers.len()]
        };
        self.guard()?;
        Ok(CacheOutcome { seq, local: r, followers })
    }

    /// Waits up to `timeout` for every follower's answer to cache call `seq` and returns them; the ones missing are `Silent` and the call
    /// is abandoned (see [`CacheDeadlines`]). A group that fails meanwhile is the error.
    fn wait_collect(&self, seq: u64, timeout: Duration) -> Result<Vec<RankAnswer>, ModelError> {
        let deadline = Instant::now() + timeout;
        let mut l = self.shared.ledger.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(why) = &l.failed {
                return Err(lost_group(why.clone()));
            }
            let Some(c) = l.collect.get(&seq) else {
                return Err(lost_group(format!("the answers to cache call {seq} were lost")));
            };
            let left = deadline.saturating_duration_since(Instant::now());
            if c.complete() || left.is_zero() {
                let answers: Vec<RankAnswer> = c
                    .answers
                    .iter()
                    .map(|a| match a {
                        None => RankAnswer::Silent,
                        Some((Class::Ok(d), _)) => RankAnswer::Ok(*d),
                        Some((class, text)) => RankAnswer::Err(if text.is_empty() {
                            class.describe()
                        } else {
                            format!("{}: {text}", class.describe())
                        }),
                    })
                    .collect();
                let (complete, name) = (c.complete(), c.name);
                if complete {
                    l.collect.remove(&seq);
                } else {
                    let now = self.shared.now_ms();
                    if let Some(c) = l.collect.get_mut(&seq) {
                        c.abandoned_at = Some(now);
                    }
                    let silent: Vec<String> = answers
                        .iter()
                        .enumerate()
                        .filter(|(_, a)| **a == RankAnswer::Silent)
                        .map(|(i, _)| (i + 1).to_string())
                        .collect();
                    eprintln!(
                        "rexl3 level=warn event=tp_cache_call_unanswered call={seq} name={} ranks={} waited_ms={} (the call fails; the \
                         group stays up unless the answer is still missing after the stall deadline)",
                        name,
                        silent.join(","),
                        timeout.as_millis()
                    );
                }
                return Ok(answers);
            }
            l = self.shared.cv.wait_timeout(l, left.min(Duration::from_millis(100))).unwrap_or_else(|p| p.into_inner()).0;
        }
    }

    /// Stops waiting for cache call `seq`: its answers are consumed as they come (or dropped at once if all are in).
    fn abandon_collect(&self, seq: u64) {
        let now = self.shared.now_ms();
        let mut l = self.shared.ledger.lock().unwrap_or_else(|p| p.into_inner());
        let done = match l.collect.get_mut(&seq) {
            Some(c) => {
                c.abandoned_at = Some(now);
                c.complete()
            }
            None => false,
        };
        if done {
            l.collect.remove(&seq);
        }
    }

    /// [`TpLeader::forward_call`] for a call that returns nothing: a send that fails is latched and surfaces at the next fallible call.
    pub fn forward_unit(&mut self, call: R::Call, local: impl FnOnce(&mut R)) {
        let name = call.name();
        let Ok(seq) = self.send(Msg::Call(call)) else { return };
        self.run_local(local);
        self.finish_unit(seq, name);
    }

    // ---- the core calls

    pub fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        let seq = self.send(Msg::BeginSlot(slot as u32))?;
        let r = self.run_local(|m| m.begin_slot(slot));
        self.finish(seq, "begin_slot", r, |_| 0, false)
    }

    pub fn end_slot(&mut self, slot: usize) {
        // nothing to return to: a lost group is latched and surfaces at the next fallible call
        let Ok(seq) = self.send(Msg::EndSlot(slot as u32)) else { return };
        self.run_local(|m| m.end_slot(slot));
        self.finish_unit(seq, "end_slot");
    }

    pub fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        let seq = self.send(Msg::Prefill { batch: b.clone(), want: want.map(|w| w as u32) })?;
        let r = self.run_local(|m| m.prefill_chunk(b, want));
        self.finish(seq, "prefill_chunk", r, |c| c.as_ref().map_or(0, candidates_digest), false)
    }

    pub fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        let seq = self.send(Msg::Window { batch: b.clone(), count: count as u32 })?;
        let r = self.run_local(|m| m.window_candidates(b, count));
        self.finish(seq, "window_candidates", r, candidates_digest, false)
    }

    pub fn commit(&mut self, slot: usize, keep: usize) {
        let Ok(seq) = self.send(Msg::Commit { slot: slot as u32, keep: keep as u32 }) else { return };
        self.run_local(|m| m.commit(slot, keep));
        self.finish_unit(seq, "commit");
    }

    pub fn flush(&mut self) -> Result<(), ModelError> {
        let seq = self.send(Msg::Flush)?;
        let r = self.run_local(|m| m.flush());
        self.finish(seq, "flush", r, |_| 0, false)
    }
}

/// The snapshot of a leader's group (what `/health` and `/metrics` report).
fn group_status_of(shared: &Shared) -> crate::tp_status::GroupStatus {
    use crate::tp_status::{GroupStatus, RankStatus};
    let now_ms = shared.now_ms();
    let sent = shared.sent_seq.load(Ordering::Relaxed);
    let l = shared.ledger.lock().unwrap_or_else(|p| p.into_inner());
    let lost = l.failed.clone();
    let per_rank = shared
        .peers
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let acked_seq = l.ack_seq[i];
            let left = p.left.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let connected = !p.closed.load(Ordering::Acquire) && !p.reader_done.load(Ordering::Acquire);
            let rtt = p.rtt_us.load(Ordering::Relaxed);
            RankStatus {
                rank: p.rank,
                connected,
                state: if connected {
                    "up"
                } else if l.blamed == Some(p.rank) || left.is_some() {
                    "lost"
                } else {
                    "closed"
                },
                acked_seq,
                applied_seq: p.applied.load(Ordering::Relaxed).max(acked_seq),
                lag_commands: sent.saturating_sub(acked_seq),
                last_frame_age_ms: now_ms.saturating_sub(p.last_rx.load(Ordering::Relaxed)),
                ping_rtt_us: (rtt != 0).then_some(rtt),
                acks: l.acked_by[i],
                diverged: l.diverged_by[i],
                queued_frames: p.queued_frames.load(Ordering::Relaxed),
                bytes_sent: p.bytes.load(Ordering::Relaxed),
                left,
            }
        })
        .collect();
    GroupStatus {
        ranks: shared.peers.len() + 1,
        commands_sent: sent,
        settled: l.settled,
        acks: l.acked,
        tolerated: l.tolerated,
        lost,
        lost_rank: l.blamed,
        per_rank,
    }
}

impl<R: LockstepRuntime> Drop for TpLeader<R> {
    fn drop(&mut self) {
        let clean = self.shared.failure().is_none();
        let st = self.stats();
        eprintln!(
            "rexl3 event=tp_leader_stats commands={} bytes={} acked={} tolerated={} clean={clean}",
            st.commands, st.bytes, st.acked, st.tolerated
        );
        // `stop` FIRST: the readers must know this is a goodbye before the first follower can answer the Shutdown with its Bye (a Bye
        // read while `stop` is clear is a rank leaving, which loses the group and closes every link, including those of followers whose
        // Shutdown is still queued)
        self.shared.stop.store(true, Ordering::Release);
        if clean {
            // a leader that goes away tells every follower to stop after what it has been sent (through the same ordered queues)
            let _ = self.enqueue_all(0, &Msg::Shutdown("leader shut down".into()));
        }
        // queued links: the writers drain what is queued (the Shutdown), end their streams (a FIN after it) and exit once their queue's
        // senders are gone: ours here, the watchdog's when it sees `stop`. An inline link wrote the Shutdown above, on this thread, so
        // the end of the stream follows it here, in order.
        for p in &mut self.peers {
            match &mut p.wire {
                Wire::Queued(q) => *q = None,
                Wire::Inline { close_write, .. } => {
                    if clean {
                        close_write();
                    }
                }
            }
        }
        if clean {
            // every follower applies what is queued, says Bye and closes; the acks still in flight are read, not answered with a reset
            let deadline = Instant::now() + Duration::from_secs(3);
            while !self.shared.peers.iter().all(|p| p.reader_done.load(Ordering::Acquire)) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        self.shared.close_links();
        self.shared.cv.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}
/// One follower's reader: its acks settle against the leader's own results, its pongs are proof of life and a round trip, and its
/// goodbye, abort, a frame that is not for the leader or the end of its stream loses the group by name.
fn leader_reader<C: FamilyCall>(rank: usize, rx: Box<dyn Read + Send>, shared: Arc<Shared>) {
    crate::hosttrace::set_role("lrx");
    leader_read_loop::<C>(rank, rx, &shared);
    let p = &shared.peers[rank - 1];
    p.reader_done.store(true, Ordering::Release);
}

fn leader_read_loop<C: FamilyCall>(rank: usize, mut rx: Box<dyn Read + Send>, shared: &Arc<Shared>) {
    let peer = &shared.peers[rank - 1];
    let leave = |why: &str| {
        *peer.left.lock().unwrap_or_else(|p| p.into_inner()) = Some(why.to_string());
    };
    loop {
        match read_frame::<C>(&mut *rx) {
            Ok((seq, msg)) => {
                peer.last_rx.store(shared.now_ms(), Ordering::Relaxed);
                match msg {
                    Msg::Ack(class, text) => {
                        crate::hosttrace::mark("ack_rx", seq);
                        shared.ack_from(rank, seq, class, text)
                    }
                    Msg::Pong { nonce, applied } => {
                        peer.applied.fetch_max(applied, Ordering::Relaxed);
                        if nonce != 0 && nonce == peer.ping_nonce.load(Ordering::Relaxed) {
                            let rtt = shared.now_us().saturating_sub(peer.ping_at_us.load(Ordering::Relaxed));
                            peer.rtt_us.store(rtt.max(1), Ordering::Relaxed);
                        }
                    }
                    Msg::Bye(why) => {
                        leave(&format!("said goodbye: {why}"));
                        if !shared.stop.load(Ordering::Relaxed) {
                            shared.fail_rank(rank, format!("rank {rank} left the group: {why}"));
                        }
                        peer.closed.store(true, Ordering::Release);
                        return;
                    }
                    Msg::Abort(why) => {
                        leave(&format!("aborted: {why}"));
                        shared.fail_rank(rank, format!("rank {rank} aborted: {why}"));
                        peer.closed.store(true, Ordering::Release);
                        return;
                    }
                    other => {
                        leave(&format!("sent an unexpected {} message", other.name()));
                        shared.fail_rank(rank, format!("rank {rank} sent an unexpected {} message", other.name()));
                        peer.closed.store(true, Ordering::Release);
                        return;
                    }
                }
            }
            Err(e) => {
                if !shared.stop.load(Ordering::Relaxed) && !peer.closed.load(Ordering::Acquire) {
                    leave(&format!("its stream ended: {e}"));
                    shared.fail_rank(rank, format!("the connection to rank {rank} failed: {e}"));
                }
                peer.closed.store(true, Ordering::Release);
                return;
            }
        }
    }
}

/// One follower's writer: puts its queue on the wire in order. A write that fails (the socket's write timeout is a peer that does not
/// drain) loses the group, naming the rank and the call.
fn leader_writer(
    rank: usize,
    q: mpsc::Receiver<Outgoing>,
    tx: Arc<Mutex<Box<dyn Write + Send>>>,
    shared: Arc<Shared>,
    close_write: Arc<dyn Fn() + Send + Sync>,
) {
    let peer = &shared.peers[rank - 1];
    while let Ok(out) = q.recv() {
        let n = out.frame.len() as u64;
        let r = {
            let mut w = tx.lock().unwrap_or_else(|p| p.into_inner());
            w.write_all(&out.frame).and_then(|_| w.flush())
        };
        peer.queued_frames.fetch_sub(1, Ordering::Relaxed);
        peer.queued_bytes.fetch_sub(n, Ordering::Relaxed);
        match r {
            Ok(()) => {
                if out.what != "ping" {
                    peer.bytes.fetch_add(n, Ordering::Relaxed);
                }
                peer.last_tx.store(shared.now_ms(), Ordering::Relaxed);
            }
            Err(e) => {
                if !shared.stop.load(Ordering::Relaxed) {
                    *peer.left.lock().unwrap_or_else(|p| p.into_inner()) = Some(format!("a write failed: {e}"));
                    shared.fail_rank(rank, format!("sending {} (call {}) to rank {rank} failed: {e}", out.what, out.seq));
                }
                return;
            }
        }
    }
    // the queue is closed and drained: the leader is going away; end this side's stream after what was written
    close_write();
}

fn leader_watchdog<C: FamilyCall>(shared: Arc<Shared>, paths: Vec<PingPath>, t: TpTimeouts) {
    let ping_ms = t.ping.as_millis() as u64;
    let peer_ms = t.peer.as_millis() as u64;
    while !shared.stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(50));
        let now = shared.now_ms();
        for (i, path) in paths.iter().enumerate() {
            let p = &shared.peers[i];
            if p.closed.load(Ordering::Acquire) {
                continue;
            }
            // a silent peer: per rank, so that one chatty follower never masks a dead one
            let idle_rx = now.saturating_sub(p.last_rx.load(Ordering::Relaxed));
            if idle_rx > peer_ms && shared.failure().is_none() {
                shared.fail_rank(
                    p.rank,
                    format!("no frame from rank {} for {:.1}s (peer timeout {:?})", p.rank, idle_rx as f64 / 1e3, t.peer),
                );
            }
            // a ping when this side has been quiet toward it (and nothing is waiting in its queue to be written)
            let idle_tx = now.saturating_sub(p.last_tx.load(Ordering::Relaxed));
            if idle_tx >= ping_ms && shared.failure().is_none() && p.queued_frames.load(Ordering::Relaxed) == 0 {
                match path {
                    PingPath::Queue(q) => {
                        let nonce = p.ping_nonce.load(Ordering::Relaxed) + 1;
                        p.ping_nonce.store(nonce, Ordering::Relaxed);
                        p.ping_at_us.store(shared.now_us(), Ordering::Relaxed);
                        let frame = Arc::new(Msg::<C>::Ping(nonce).encode(0));
                        p.queued_frames.fetch_add(1, Ordering::Relaxed);
                        p.queued_bytes.fetch_add(frame.len() as u64, Ordering::Relaxed);
                        if q.send(Outgoing { frame, what: "ping", seq: 0 }).is_err() {
                            p.queued_frames.fetch_sub(1, Ordering::Relaxed);
                        }
                        // do not ping again before the next interval, whether or not the writer has run yet
                        p.last_tx.store(now, Ordering::Relaxed);
                    }
                    PingPath::Inline(tx) => {
                        // the same lock the engine thread writes its commands under, one `write_all` per frame: a ping never lands
                        // inside a command. A command being written (the lock is taken) is traffic enough: try again next tick.
                        if let Ok(mut w) = tx.try_lock() {
                            let nonce = p.ping_nonce.load(Ordering::Relaxed) + 1;
                            p.ping_nonce.store(nonce, Ordering::Relaxed);
                            p.ping_at_us.store(shared.now_us(), Ordering::Relaxed);
                            let frame = Msg::<C>::Ping(nonce).encode(0);
                            let r = w.write_all(&frame).and_then(|_| w.flush());
                            drop(w);
                            match r {
                                Ok(()) => p.last_tx.store(shared.now_ms(), Ordering::Relaxed),
                                Err(e) => {
                                    if !shared.stop.load(Ordering::Relaxed) {
                                        *p.left.lock().unwrap_or_else(|x| x.into_inner()) = Some(format!("a write failed: {e}"));
                                        shared.fail_rank(p.rank, format!("sending ping (call 0) to rank {} failed: {e}", p.rank));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // a follower that answers pings but never a cache call it was sent: stuck, not slow (`CacheDeadlines`)
        if shared.failure().is_none() {
            if let Some((rank, seq, name, age)) = shared.stalled_cache_call(shared.cache_stall_ms.load(Ordering::Relaxed)) {
                shared.fail_rank(
                    rank,
                    format!(
                        "rank {rank} has not answered cache call {seq} ({name}) {:.1}s after the leader stopped waiting for it (it \
                         answers pings, so it is stuck, not gone)",
                        age as f64 / 1e3
                    ),
                );
            }
        }
        // the collective fabric failed on its own (the pings above travel on the command link, which can stay up)
        if shared.failure().is_none() {
            if let Some(why) = shared.poll_health(t.health_poll) {
                eprintln!("rexl3 level=error event=tp_comm_async_error rank=0 reason=\"{why}\"");
                shared.fail(why);
            }
        }
        // a call that cannot finish because the group is lost
        if let Some(hang) = t.hang_exit {
            let lost_at = shared.lost_at.load(Ordering::Relaxed);
            let since = shared.call_since.load(Ordering::Relaxed);
            if lost_at != 0 && since != 0 && now.saturating_sub(lost_at.max(since)) > hang.as_millis() as u64 {
                eprintln!(
                    "rexl3 level=fatal event=tp_group_lost_hang rank=0 reason=\"{}\" (a model call is stuck in a collective with a lost peer; aborting)",
                    shared.failure().unwrap_or_default()
                );
                std::process::abort();
            }
        }
    }
}
// ------------------------------------------------------------------------------------------------------------------- the follower

/// How a follower run ended (an error is a fatal runtime error or a protocol violation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FollowerExit {
    /// The leader said it is going away; the commands it had sent were applied.
    Shutdown(String),
    /// The connection ended without a Shutdown (the leader crashed, or the peer timeout passed).
    LeaderLost(String),
    /// The caller asked to stop ([`FollowerConfig::stop`]).
    Stopped,
}

/// A progress callback of the follower loop.
pub type ProgressFn = Box<dyn FnMut(&FollowerStats) + Send>;

/// The follower's knobs and the handles its caller keeps.
pub struct FollowerConfig {
    pub timeouts: TpTimeouts,
    /// Set to ask the loop to return after the command in hand.
    pub stop: Arc<AtomicBool>,
    /// Told how many commands were applied, now and then (the stderr health line).
    pub on_progress: Option<ProgressFn>,
    /// Run when the leader is lost while a call is stuck (the communicator's abort).
    pub abort_hook: Option<AbortHook>,
    /// This process's rank in the group (1 for a two-rank group; any rank above 0 for a larger one): it names the rank in the backstop's
    /// and the stall watchdog's messages.
    pub rank: usize,
    /// Spin-then-park receive: while a command arrived or was acknowledged within this long, the follower's reader and its apply loop
    /// poll for the next frame instead of parking in `read` / a channel wait (a quiet group, and a link that carries only pings, parks).
    /// A parked thread is woken by the kernel (an interrupt, a futex) on a core that went idle: tens of microseconds to a millisecond
    /// from a deep idle state, with cold caches, on the path of the first collective of every window (`docs/NEMOTRON_TP.md` 17).
    /// Zero: park at once, as before. Only latency, never what is computed.
    pub spin: Duration,
    /// The communicator's health probe, polled by the follower's watchdog every [`TpTimeouts::health_poll`] ([`HealthProbe`]).
    pub health: Option<HealthProbe>,
}

/// The default of [`FollowerConfig::spin`], microseconds (`REXL3_TP_SPIN_US` overrides it; `0` turns it off). 250 ms: a window follows the
/// previous one by far less while a group is busy, and an idle group parks and burns nothing. Measured -0.73 ms a round (-1.3 percent)
/// at two ranks for two cores of the follower node while busy (`docs/NEMOTRON_TP.md` 17.4); on by default since 2026-10-12 (owner).
pub const DEFAULT_SPIN_US: u64 = 250_000;

fn spin_from_env() -> Duration {
    let us = std::env::var("REXL3_TP_SPIN_US").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(DEFAULT_SPIN_US);
    Duration::from_micros(us)
}

impl FollowerConfig {
    pub fn new(timeouts: TpTimeouts) -> Self {
        FollowerConfig {
            timeouts,
            stop: Arc::new(AtomicBool::new(false)),
            on_progress: None,
            abort_hook: None,
            rank: 1,
            spin: spin_from_env(),
            health: None,
        }
    }

    /// The same config for follower `rank`.
    pub fn with_rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }
}

/// What the follower has done.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FollowerStats {
    pub applied: u64,
    /// Commands whose application returned an error (the leader's did too, or the group is lost).
    pub errors: u64,
    pub windows: u64,
    pub prefills: u64,
}

enum Event<C: FamilyCall> {
    Frame(u64, Msg<C>),
    Closed(String),
}

/// The follower's loop: applies the leader's commands, in order, to `rt` and acknowledges each. Returns on Shutdown, on EOF or when the
/// peer goes silent. A fatal local error is answered (so the leader fails by name) and returned.
pub fn run_follower<R: LockstepRuntime>(
    mut rt: R,
    link: Link<R::Call>,
    mut cfg: FollowerConfig,
) -> Result<FollowerExit, ModelError> {
    let Link { rx, tx, close, set_rx_timeout, rx_ready, .. } = link;
    set_rx_timeout(None);
    crate::hosttrace::start_from_env("F");
    crate::hosttrace::set_role("fapply");
    let spin = cfg.spin;
    let tx: Arc<Mutex<Box<dyn Write + Send>>> = Arc::new(Mutex::new(tx));
    let shared = Arc::new(Shared::new());
    shared.touch();
    // the communicator's abort, shared by the reader (the connection ended), the watchdog (silence) and the loop's exit
    let hook: Arc<Mutex<Option<AbortHook>>> = Arc::new(Mutex::new(cfg.abort_hook.take()));
    *shared.health.lock().unwrap_or_else(|p| p.into_inner()) = cfg.health.take();
    let rank = cfg.rank;
    let (ev_tx, ev_rx) = mpsc::sync_channel::<Event<R::Call>>(1024);
    // the reader answers pings itself, so a long local call never makes the follower look dead
    let reader = {
        let (shared, tx, hook) = (shared.clone(), tx.clone(), hook.clone());
        let hang = cfg.timeouts.hang_exit;
        std::thread::Builder::new()
            // the rank in the name (the OS keeps 15 characters): tests that count one follower's CPU find its reader by name
            .name(format!("tp-follower-rx{}", rank % 10))
            .spawn(move || {
                let mut rx = rx;
                crate::hosttrace::set_role("frx");
                // The leader is gone or says the group is dead. A model call stuck in a collective with it (or with a sibling that
                // died) can never finish and the loop only sees the news between calls, so abort the communicator NOW: the call
                // returns and the loop exits, instead of waiting out the peer timeout (and, before this, for ever). The backstop
                // FIRST: the communicator's abort has been seen to block (a rank 1 left alive for ever, section 12.3 B2), and
                // nothing can complete, so a few seconds are enough.
                // `blamed`: the rank the loss is on when this rank knows it (the end of the leader's own link is rank 0; the
                // leader's words about a sibling blame no one here)
                let lose = |blamed: Option<usize>, why: String| {
                    shared.lost_at.store(shared.now_ms() + 1, Ordering::Relaxed);
                    {
                        let mut l = shared.ledger.lock().unwrap_or_else(|p| p.into_inner());
                        match blamed {
                            Some(r) => l.fail_rank(r, why.clone()),
                            None => l.fail(why.clone()),
                        }
                    }
                    if let Some(after) = hang {
                        exit_backstop(rank, after.min(Duration::from_secs(5)), why);
                    }
                    if let Some(h) = hook.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
                        h();
                    }
                };
                // a Shutdown was read: the end of the stream that follows is the leader leaving, not a loss. An Abort is NOT a
                // goodbye (it says the group is dead), so the end of the stream after it still aborts the communicator
                let mut told = false;
                loop {
                    // spin-then-park: poll for the next frame while the group is busy (a command read or acknowledged within `spin`)
                    // before parking in `read` (a woken thread is a millisecond late on a cold core; a polling one sees the frame in
                    // microseconds)
                    if let (Some(ready), false) = (rx_ready.as_ref(), spin.is_zero()) {
                        let window = spin.as_micros() as u64;
                        while shared.busy_within(window) && !ready() {
                            std::hint::spin_loop();
                        }
                    }
                    match read_frame::<R::Call>(&mut *rx) {
                        Ok((seq, Msg::Ping(nonce))) => {
                            crate::hosttrace::mark("rx_ping", seq);
                            shared.touch();
                            let applied = shared.peer_applied.load(Ordering::Relaxed);
                            let mut w = tx.lock().unwrap_or_else(|p| p.into_inner());
                            let _ = write_frame(&mut **w, 0, &Msg::<R::Call>::Pong { nonce, applied });
                        }
                        Ok((seq, msg)) => {
                            crate::hosttrace::mark("rx_frame", seq);
                            shared.touch();
                            told |= matches!(msg, Msg::Shutdown(_));
                            // an Abort while a call is in flight: that call may be waiting in a collective for ranks that will
                            // never arrive (a sibling is gone), so the communicator is aborted at once, not between calls
                            if let Msg::Abort(why) = &msg {
                                if shared.call_since.load(Ordering::Relaxed) != 0 && shared.lost_at.load(Ordering::Relaxed) == 0 {
                                    lose(None, format!("rank 0 aborted the group: {why}"));
                                }
                            }
                            let cmd = msg.is_command();
                            if cmd {
                                shared.active();
                            }
                            if ev_tx.send(Event::Frame(seq, msg)).is_err() {
                                return;
                            }
                            if cmd {
                                crate::hosttrace::mark("queued", seq);
                            }
                        }
                        Err(e) => {
                            // the end of the leader's stream with no goodbye
                            if !told && !shared.stop.load(Ordering::Relaxed) && shared.lost_at.load(Ordering::Relaxed) == 0 {
                                lose(Some(0), format!("the connection to rank 0 ended: {e}"));
                            }
                            let _ = ev_tx.send(Event::Closed(e.to_string()));
                            return;
                        }
                    }
                }
            })
            .expect("spawn the follower's reader")
    };
    // the watchdog: a leader that went silent, and a call that can no longer finish
    let watch = {
        let (shared, close, t, hook) = (shared.clone(), close.clone(), cfg.timeouts.clone(), hook.clone());
        std::thread::Builder::new()
            .name("tp-follower-watch".into())
            .spawn(move || {
                while !shared.stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(50));
                    let now = shared.now_ms();
                    let idle = now.saturating_sub(shared.last_rx.load(Ordering::Relaxed));
                    if idle > t.peer.as_millis() as u64 && shared.lost_at.load(Ordering::Relaxed) == 0 {
                        shared.lost_at.store(now + 1, Ordering::Relaxed);
                        {
                            let mut l = shared.ledger.lock().unwrap_or_else(|p| p.into_inner());
                            l.fail_rank(0, format!("no frame from rank 0 for {:.1}s (peer timeout {:?})", idle as f64 / 1e3, t.peer));
                        }
                        if let Some(h) = hook.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
                            h();
                        }
                        close();
                    }
                    // the collective fabric failed on its own while the leader's pings still arrive: the group is lost all the same
                    if shared.lost_at.load(Ordering::Relaxed) == 0 {
                        if let Some(why) = shared.poll_health(t.health_poll) {
                            eprintln!("rexl3 level=error event=tp_comm_async_error rank={rank} reason=\"{why}\"");
                            shared.lost_at.store(shared.now_ms() + 1, Ordering::Relaxed);
                            shared.ledger.lock().unwrap_or_else(|p| p.into_inner()).fail(why.clone());
                            if let Some(after) = t.hang_exit {
                                exit_backstop(rank, after.min(Duration::from_secs(5)), why);
                            }
                            if let Some(h) = hook.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
                                h();
                            }
                            close();
                        }
                    }
                    if let Some(hang) = t.hang_exit {
                        let lost_at = shared.lost_at.load(Ordering::Relaxed);
                        let since = shared.call_since.load(Ordering::Relaxed);
                        if lost_at != 0 && since != 0 && now.saturating_sub(lost_at.max(since)) > hang.as_millis() as u64 {
                            eprintln!(
                                "rexl3 level=fatal event=tp_group_lost_hang rank={rank} (a model call is stuck in a collective with a lost peer; aborting)"
                            );
                            std::process::abort();
                        }
                    }
                }
            })
            .expect("spawn the follower's watchdog")
    };

    let mut state = FollowerState { stats: FollowerStats::default(), expect: 1 };
    let mut last_progress = Instant::now();
    let reply = |seq: u64, m: &Msg<R::Call>| -> bool {
        let mut w = tx.lock().unwrap_or_else(|p| p.into_inner());
        write_frame(&mut **w, seq, m).is_ok()
    };
    let result: Result<FollowerExit, ModelError> = loop {
        if cfg.stop.load(Ordering::Relaxed) {
            break Ok(FollowerExit::Stopped);
        }
        // the health line: every 30 s, busy or idle
        if let Some(cb) = cfg.on_progress.as_mut() {
            if last_progress.elapsed() >= Duration::from_secs(30) {
                cb(&state.stats);
                last_progress = Instant::now();
            }
        }
        // spin-then-park: poll the channel while the group is busy before the futex wait below (the reader's hand-off is then seen at once)
        let mut spun = None;
        if !spin.is_zero() {
            let window = spin.as_micros() as u64;
            loop {
                match ev_rx.try_recv() {
                    Ok(ev) => {
                        spun = Some(ev);
                        break;
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        if !shared.busy_within(window) {
                            break;
                        }
                        std::hint::spin_loop();
                    }
                    Err(mpsc::TryRecvError::Disconnected) => break,
                }
            }
        }
        let got = match spun {
            Some(ev) => Ok(ev),
            None => ev_rx.recv_timeout(Duration::from_millis(100)),
        };
        let ev = match got {
            Ok(ev) => ev,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Some(why) = shared.failure() {
                    break Ok(FollowerExit::LeaderLost(why));
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break Ok(FollowerExit::LeaderLost("the reader ended".into())),
        };
        match ev {
            Event::Closed(why) => {
                // a loss latched before the link ended (the watchdog's silence or the fabric's async error closed it) is the reason
                break Ok(FollowerExit::LeaderLost(shared.failure().unwrap_or(why)));
            }
            Event::Frame(_, Msg::Shutdown(why)) => break Ok(FollowerExit::Shutdown(why)),
            // an Abort after the boot says the group is dead: not a goodbye (the exit is not clean)
            Event::Frame(_, Msg::Abort(why)) => break Ok(FollowerExit::LeaderLost(format!("rank 0 aborted the group: {why}"))),
            Event::Frame(seq, msg) if msg.is_command() => {
                if seq != state.expect {
                    let why = format!(
                        "command {} carries sequence number {seq}, expected {}: a command was lost or reordered",
                        msg.name(),
                        state.expect
                    );
                    let _ = reply(seq, &Msg::Abort(why.clone()));
                    break Err(ModelError::StateLost(why));
                }
                state.expect += 1;
                let name = msg.name();
                crate::hosttrace::mark("dequeued", seq);
                crate::hosttrace::mark(name, seq);
                shared.call_since.store(shared.now_ms() + 1, Ordering::Relaxed);
                crate::hosttrace::mark("apply_begin", seq);
                let applied = state.apply(&mut rt, msg, seq);
                crate::hosttrace::mark("apply_end", seq);
                shared.call_since.store(0, Ordering::Relaxed);
                shared.peer_applied.store(seq, Ordering::Relaxed);
                state.stats.applied += 1;
                let (class, text) = match applied {
                    Ok(aux) => (Class::Ok(aux), String::new()),
                    Err(e) => {
                        state.stats.errors += 1;
                        // the class is the call's own; the text says why a collective was cut short when the group is already lost
                        let class = Class::of_error(&e);
                        let (failure, blamed) = shared.failure_and_blame();
                        let e = lost_peer_error(e, failure.as_deref(), blamed, &format!("call {seq}"));
                        (class, format!("{name}: {e}"))
                    }
                };
                if !reply(seq, &Msg::Ack(class, text.clone())) {
                    break Ok(FollowerExit::LeaderLost("the leader's connection failed while acknowledging".into()));
                }
                crate::hosttrace::mark("acked", seq);
                shared.active();
                if class.is_fatal() {
                    let _ = reply(0, &Msg::Bye(text.clone()));
                    break Err(ModelError::StateLost(format!("rank {rank} failed {name} (call {seq}): {text}")));
                }
            }
            Event::Frame(_, other) => {
                let why = format!("unexpected {} message from rank 0", other.name());
                let _ = reply(0, &Msg::Abort(why.clone()));
                break Err(ModelError::StateLost(why));
            }
        }
    };
    // A lost leader (or a fatal local error) ends this process: abort the communicator now so that kernels queued behind the lost peer
    // exit before the model is dropped (its destructor frees device memory, which waits for them), and bound the exit whatever the
    // destructor does (`exit_backstop`).
    let lost = match &result {
        Ok(FollowerExit::LeaderLost(w)) => Some(w.clone()),
        Err(e) => Some(e.to_string()),
        _ => None,
    };
    if let Some(why) = lost {
        // the backstop first (the abort may block), and no wait for a hook another thread is already inside
        if let Some(after) = cfg.timeouts.hang_exit {
            exit_backstop(rank, after, why);
        }
        if let Ok(g) = hook.try_lock() {
            if let Some(h) = g.as_ref() {
                h();
            }
        }
    }
    if let Some(cb) = cfg.on_progress.as_mut() {
        cb(&state.stats);
    }
    let _ = reply(
        0,
        &Msg::Bye(match &result {
            Ok(FollowerExit::Shutdown(w)) => format!("shutdown: {w}"),
            Ok(FollowerExit::LeaderLost(w)) => format!("leader lost: {w}"),
            Ok(FollowerExit::Stopped) => "stopped".into(),
            Err(e) => e.to_string(),
        }),
    );
    shared.stop.store(true, Ordering::Relaxed);
    close();
    let _ = watch.join();
    // a reader blocked in `send` on a full channel ends when the receiver drops
    drop(ev_rx);
    let _ = reader.join();
    result
}

struct FollowerState {
    stats: FollowerStats,
    /// The next command sequence number.
    expect: u64,
}

impl FollowerState {
    /// Applies one command to `rt`. `Ok(aux)` is the digest the ack carries.
    fn apply<R: LockstepRuntime>(&mut self, rt: &mut R, msg: Msg<R::Call>, seq: u64) -> Result<u64, ModelError> {
        match msg {
            Msg::BeginSlot(s) => rt.begin_slot(s as usize).map(|_| 0),
            Msg::EndSlot(s) => {
                rt.end_slot(s as usize);
                Ok(0)
            }
            Msg::Prefill { batch, want } => {
                self.stats.prefills += 1;
                let c = rt.prefill_chunk(&batch, want.map(|w| w as usize))?;
                Ok(c.as_ref().map_or(0, candidates_digest))
            }
            Msg::Window { batch, count } => {
                self.stats.windows += 1;
                let c = rt.window_candidates(&batch, count as usize)?;
                Ok(candidates_digest(&c))
            }
            Msg::Commit { slot, keep } => {
                rt.commit(slot as usize, keep as usize);
                Ok(0)
            }
            Msg::Flush => rt.flush().map(|_| 0),
            Msg::Barrier => Ok(0),
            Msg::Call(call) => {
                if call.counts_as_prefill() {
                    self.stats.prefills += 1;
                }
                rt.apply_call(seq, call).map(|r| r.digest)
            }
            other => Err(ModelError::Request(format!("{} is not a command", other.name()))),
        }
    }
}
// ------------------------------------------------------------------------------------------------------------------- boot

/// Which side of the group a process is: rank 0 (the leader) or any rank above it (a follower).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Leader,
    Follower,
}

/// Two ranks' Hellos, exchanged: both send first, then read, and both apply the same check to the same two facts, so both reach the
/// same verdict (a refusal is also sent as an Abort so the other side prints the reason). Returns the peer's gate. This is the
/// handshake of ONE link; a leader with more than one follower admits them through [`accept_group`] / [`admit_links`], which run the
/// same check per follower and also map each connection to its rank.
pub fn handshake<C: FamilyCall>(
    link: &mut Link<C>,
    role: Role,
    local: &BootGate<C>,
    timeout: Duration,
) -> Result<BootGate<C>, ModelError> {
    link.send(0, &Msg::Hello(local.clone())).map_err(|e| ModelError::Config(format!("tensor-parallel hello: {e}")))?;
    let remote = match link.recv_polled(timeout, gb_engine::boot::checkpoint) {
        Ok((_, Msg::Hello(g))) => g,
        Ok((_, Msg::Abort(why))) => {
            return Err(ModelError::Config(format!("the peer refused the boot: {why}")));
        }
        Ok((_, other)) => {
            return Err(ModelError::Config(format!("tensor-parallel hello: the peer sent a {} message first", other.name())));
        }
        Err(WireError::Version { got, want }) => {
            let why = format!("the peer speaks protocol version {got}, this build speaks {want}");
            link.refuse(&why);
            return Err(ModelError::Config(format!("tensor-parallel hello: {why} (both ranks must run the same build)")));
        }
        Err(e) if timed_out(&e) => {
            return Err(ModelError::Config(format!("tensor-parallel hello: no answer from the peer within {timeout:?}")));
        }
        Err(e) => return Err(ModelError::Config(format!("tensor-parallel hello: {e}"))),
    };
    let (leader, follower) = match role {
        Role::Leader => (local, &remote),
        Role::Follower => (&remote, local),
    };
    if let Err(why) = BootGate::check(leader, follower) {
        link.refuse(&why);
        return Err(ModelError::Config(why));
    }
    Ok(remote)
}

/// Ends the boot of several links at once: each gets an Abort carrying its own reason and an end of stream, then what they still send
/// is read (and dropped) for at most two seconds in all, so each peer reads its reason and not a reset that races it.
fn refuse_links<C: FamilyCall>(mut items: Vec<(Link<C>, String)>) {
    for (l, why) in items.iter_mut() {
        let _ = l.send(0, &Msg::Abort(why.clone()));
        (l.close_write)();
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut open = vec![true; items.len()];
    let mut sink = [0u8; 4096];
    while Instant::now() < deadline && open.iter().any(|o| *o) {
        for (i, (l, _)) in items.iter_mut().enumerate() {
            if !open[i] {
                continue;
            }
            (l.set_rx_timeout)(Some(Duration::from_millis(50)));
            match l.rx.read(&mut sink) {
                Ok(0) => open[i] = false,
                Ok(_) => {}
                Err(e)
                    if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut | io::ErrorKind::Interrupted) => {}
                Err(_) => open[i] = false,
            }
        }
    }
    for (l, _) in items.iter_mut() {
        (l.set_rx_timeout)(None);
    }
}

/// Tells every link the boot is over (an Abort and the end of this side's stream), without waiting for anything: the boot's failure
/// path after the gate, where the followers are busy loading or waiting for the next message.
fn abort_links<C: FamilyCall>(links: &mut [Link<C>], why: &str) {
    for l in links.iter_mut() {
        let _ = l.send(0, &Msg::Abort(why.to_string()));
        (l.close_write)();
    }
}

/// `2` as "rank 2", `[2, 3]` as "ranks 2, 3".
fn rank_list(ranks: &[usize]) -> String {
    let list = ranks.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", ");
    if ranks.len() == 1 {
        format!("rank {list}")
    } else {
        format!("ranks {list}")
    }
}

/// Rank 0's side of admitting a group: the connections made so far, each identified by the rank in its Hello and gated against the
/// leader's own facts. Connections arrive in any order and at different times; the roster is complete when ranks `1..ranks` have each
/// said Hello once and passed the gate.
struct Roster<'a, C: FamilyCall> {
    local: &'a BootGate<C>,
    ranks: usize,
    t: &'a TpTimeouts,
    /// Rank `r` is `slots[r - 1]`.
    slots: Vec<Option<Link<C>>>,
    /// Connections that have been sent the leader's Hello and have not said theirs, with when they connected.
    pending: Vec<(Link<C>, Instant)>,
    /// Connections accepted so far.
    accepted: usize,
}

impl<'a, C: FamilyCall> Roster<'a, C> {
    fn new(local: &'a BootGate<C>, t: &'a TpTimeouts) -> Roster<'a, C> {
        let ranks = (local.ranks as usize).max(2);
        Roster { local, ranks, t, slots: (1..ranks).map(|_| None).collect(), pending: Vec::new(), accepted: 0 }
    }

    fn followers(&self) -> usize {
        self.ranks - 1
    }

    fn complete(&self) -> bool {
        self.slots.iter().all(Option::is_some)
    }

    fn missing(&self) -> Vec<usize> {
        self.slots.iter().enumerate().filter(|(_, s)| s.is_none()).map(|(i, _)| i + 1).collect()
    }

    /// A new connection: it is sent the leader's Hello at once (the follower says its own once it has planned).
    fn begin(&mut self, mut link: Link<C>) -> Result<(), ModelError> {
        link.send(0, &Msg::Hello(self.local.clone())).map_err(|e| ModelError::Config(format!("tensor-parallel hello: {e}")))?;
        self.accepted += 1;
        self.pending.push((link, Instant::now()));
        Ok(())
    }

    /// Ends the boot of everything in the roster except `culprit` (refused with `why`, which every other rank is told too).
    fn fail_boot(&mut self, culprit: Option<Link<C>>, why: &str) {
        let mut items: Vec<(Link<C>, String)> = Vec::new();
        let others = format!("the group's boot was refused: {why}");
        if let Some(l) = culprit {
            items.push((l, why.to_string()));
        }
        for l in self.slots.iter_mut().filter_map(Option::take) {
            items.push((l, others.clone()));
        }
        for (l, _) in self.pending.drain(..) {
            items.push((l, others.clone()));
        }
        refuse_links(items);
    }

    /// Reads whatever Hellos have arrived (waiting at most `slice` for each pending connection), identifies and gates them. An error is
    /// a refusal that has already been sent to every connection.
    fn poll(&mut self, slice: Duration) -> Result<(), ModelError> {
        let mut i = 0;
        while i < self.pending.len() {
            let got = {
                let (link, since) = &mut self.pending[i];
                let left = (*since + self.t.ready).saturating_duration_since(Instant::now());
                if left.is_zero() {
                    Err(WireError::Io(io::Error::new(io::ErrorKind::TimedOut, "no hello")))
                } else {
                    link.recv_polled(slice.min(left), gb_engine::boot::checkpoint)
                }
            };
            match got {
                Ok((_, Msg::Hello(remote))) => {
                    let (link, _) = self.pending.remove(i);
                    self.admit(link, remote)?;
                }
                Ok((_, Msg::Abort(why))) => {
                    let (link, _) = self.pending.remove(i);
                    let peer = link.peer.clone();
                    let msg = if self.ranks > 2 {
                        format!("the peer at {peer} refused the boot: {why}")
                    } else {
                        format!("the peer refused the boot: {why}")
                    };
                    drop(link);
                    self.fail_boot(None, &msg);
                    return Err(ModelError::Config(msg));
                }
                Ok((_, other)) => {
                    let (link, _) = self.pending.remove(i);
                    let msg = format!("tensor-parallel hello: the peer sent a {} message first", other.name());
                    self.fail_boot(Some(link), &msg);
                    return Err(ModelError::Config(msg));
                }
                Err(WireError::Version { got, want }) => {
                    let (link, _) = self.pending.remove(i);
                    let why = format!("the peer at {} speaks protocol version {got}, this build speaks {want}", link.peer);
                    self.fail_boot(Some(link), &why);
                    return Err(ModelError::Config(format!("tensor-parallel hello: {why} (all ranks must run the same build)")));
                }
                Err(e) if timed_out(&e) => {
                    let (link, since) = &self.pending[i];
                    if since.elapsed() >= self.t.ready {
                        let msg =
                            format!("tensor-parallel hello: no answer from the peer at {} within {:?}", link.peer, self.t.ready);
                        let (link, _) = self.pending.remove(i);
                        self.fail_boot(Some(link), &msg);
                        return Err(ModelError::Config(msg));
                    }
                    i += 1;
                }
                Err(e) => {
                    let (link, _) = self.pending.remove(i);
                    let msg = format!("tensor-parallel hello: {e}");
                    drop(link);
                    self.fail_boot(None, &msg);
                    return Err(ModelError::Config(msg));
                }
            }
        }
        Ok(())
    }

    /// Identifies a connection by the rank in its Hello and gates it.
    fn admit(&mut self, link: Link<C>, remote: BootGate<C>) -> Result<(), ModelError> {
        let r = remote.rank as usize;
        let refusal = if remote.ranks != self.local.ranks {
            Some(format!(
                "rank {r} was started with --tp-ranks {} and rank 0 with --tp-ranks {}: every rank must be given the same group size",
                remote.ranks, self.local.ranks
            ))
        } else if r == 0 || r >= self.ranks {
            Some(format!(
                "the peer at {} says it is rank {r}, which a group of {} ranks has no follower for (followers are ranks 1 to {}): \
                 check --tp-rank on that process",
                link.peer,
                self.ranks,
                self.ranks - 1
            ))
        } else if self.slots[r - 1].is_some() {
            Some(format!(
                "rank {r} is claimed by two processes (a second one, at {}, said it is rank {r} too): check --tp-rank on every node",
                link.peer
            ))
        } else {
            BootGate::check(self.local, &remote).err()
        };
        match refusal {
            Some(why) => {
                self.fail_boot(Some(link), &why);
                Err(ModelError::Config(why))
            }
            None => {
                self.slots[r - 1] = Some(link);
                Ok(())
            }
        }
    }

    /// The followers' links, in rank order.
    fn into_links(self) -> Vec<Link<C>> {
        self.slots.into_iter().flatten().collect()
    }
}

/// Rank 0: admits the `ranks - 1` followers of a group that are ALREADY connected (`links`, in any order): each is sent the leader's
/// Hello, identified by its own and gated, and the links come back in rank order (`links[r - 1]` is rank `r`). A duplicate or an
/// out-of-range rank, a group size that differs, a protocol version or any fact the gate compares that differs refuses the whole
/// boot by name, on every link. [`accept_group`] is this over a listening socket.
pub fn admit_links<C: FamilyCall>(links: Vec<Link<C>>, local: &BootGate<C>, t: &TpTimeouts) -> Result<Vec<Link<C>>, ModelError> {
    let mut roster = Roster::new(local, t);
    let need = roster.followers();
    if links.len() != need {
        return Err(ModelError::Config(format!(
            "a group of {} ranks needs {need} follower connections, got {}",
            roster.ranks,
            links.len()
        )));
    }
    for l in links {
        roster.begin(l)?;
    }
    while !roster.complete() {
        roster.poll(Duration::from_millis(20))?;
    }
    Ok(roster.into_links())
}

/// Rank 0: accepts the `ranks - 1` followers of the group under ONE deadline (`t.connect`, from this call), identifying each by the
/// rank in its Hello (they connect in any order, at different times) and gating it. Returns the links in rank order. A rank that never
/// connected is named in the error; a Hello that never comes within `t.ready` of its connection is refused by name.
pub fn accept_group<C: FamilyCall>(
    l: &std::net::TcpListener,
    local: &BootGate<C>,
    t: &TpTimeouts,
) -> Result<Vec<Link<C>>, ModelError> {
    l.set_nonblocking(true).map_err(|e| ModelError::Config(format!("tensor-parallel listener: {e}")))?;
    let deadline = Instant::now() + t.connect;
    let mut next_note = Instant::now() + Duration::from_secs(15);
    let mut roster = Roster::new(local, t);
    let need = roster.followers();
    loop {
        gb_engine::boot::checkpoint();
        let mut idle = true;
        match l.accept() {
            Ok((s, from)) => {
                s.set_nonblocking(false).map_err(|e| ModelError::Config(format!("tensor-parallel connection: {e}")))?;
                eprintln!("rexl3 event=tp_follower_connected from={from}");
                let link = Link::tcp(s, t.write).map_err(|e| ModelError::Config(format!("tensor-parallel connection: {e}")))?;
                // every connection is identified by its Hello, not by its order: a rank started twice is found there (by name), and a
                // stray connection that never says hello is dropped when the roster is complete
                roster.begin(link)?;
                idle = false;
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(ModelError::Config(format!("tensor-parallel accept: {e}"))),
        }
        if !roster.pending.is_empty() {
            let before = roster.pending.len();
            roster.poll(Duration::from_millis(10))?;
            idle &= roster.pending.len() == before;
        }
        if roster.complete() {
            return Ok(roster.into_links());
        }
        if roster.accepted < need && Instant::now() >= deadline {
            let missing = roster.missing();
            let peer = l.local_addr().map(|a| a.to_string()).unwrap_or_default();
            let unsaid = roster.pending.len();
            let msg = if need == 1 {
                format!(
                    "rank 1 did not connect within {:?} (start `rexl3-serve --tp-rank 1 --tp-peer {peer}` on the other node)",
                    t.connect
                )
            } else {
                format!(
                    "{} did not connect within {:?}: {} of {need} followers are up{} (start `rexl3-serve --tp-rank R --tp-peer {peer}` \
                     for each missing rank on its node)",
                    rank_list(&missing),
                    t.connect,
                    need - missing.len(),
                    if unsaid > 0 { format!(", {unsaid} connected but have not said hello yet") } else { String::new() }
                )
            };
            roster.fail_boot(None, &msg);
            return Err(ModelError::Config(msg));
        }
        if Instant::now() >= next_note {
            eprintln!(
                "rexl3 event=tp_waiting_for_follower connected={} of={need} missing=\"{}\"",
                need - roster.missing().len(),
                roster.missing().iter().map(|r| r.to_string()).collect::<Vec<_>>().join(",")
            );
            next_note = Instant::now() + Duration::from_secs(15);
        }
        if idle {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

/// The bootstrap bytes of the collective, moved over the link(s): rank 0 offers them to every follower, each receives, checks and
/// acknowledges. A follower holds one link (to rank 0); the leader holds one per follower, rank `r`'s at index `r - 1`.
pub struct Rendezvous<'a, C: FamilyCall = NoCall> {
    links: Vec<&'a mut Link<C>>,
    role: Role,
    timeout: Duration,
}

impl<'a, C: FamilyCall> Rendezvous<'a, C> {
    /// The rendezvous over one link (a follower's, or a two-rank leader's).
    pub fn pair(link: &'a mut Link<C>, role: Role, timeout: Duration) -> Rendezvous<'a, C> {
        Rendezvous { links: vec![link], role, timeout }
    }

    /// The leader's rendezvous over every follower's link, in rank order.
    pub fn group(links: &'a mut [Link<C>], timeout: Duration) -> Rendezvous<'a, C> {
        Rendezvous { links: links.iter_mut().collect(), role: Role::Leader, timeout }
    }
}

impl<C: FamilyCall> Rendezvous<'_, C> {
    pub fn role(&self) -> Role {
        self.role
    }

    /// [`Rendezvous::exchange_checked`] with nothing to check.
    pub fn exchange(&mut self, bytes: Option<Vec<u8>>) -> Result<Vec<u8>, ModelError> {
        self.exchange_checked(bytes, |_| Ok(()))
    }

    /// Rank 0: sends `bytes` (the NCCL unique id) to EVERY follower and WAITS for all their verdicts, under one deadline, before
    /// returning them. A follower: waits for them, runs `check` on what arrived and answers: an acknowledgement (a Pong), or an Abort
    /// naming why `check` refused it (the boot then fails on every rank by that name).
    ///
    /// The wait matters: `ncclCommInitRank` blocks until every rank has joined and cannot be interrupted from the host, so a rank 0
    /// that went on to join while any follower refuses the bootstrap (a different `REXL3_TP_ATTN`, a build that sized the id
    /// differently) would sit in NCCL for ever.
    pub fn exchange_checked(
        &mut self,
        bytes: Option<Vec<u8>>,
        check: impl FnOnce(&[u8]) -> Result<(), String>,
    ) -> Result<Vec<u8>, ModelError> {
        match (self.role, bytes) {
            (Role::Leader, Some(b)) => {
                let n = self.links.len();
                for (i, l) in self.links.iter_mut().enumerate() {
                    l.send(0, &Msg::CommInit(b.clone()))
                        .map_err(|e| ModelError::Config(format!("rendezvous: sending the bootstrap to rank {}: {e}", i + 1)))?;
                }
                let deadline = Instant::now() + self.timeout;
                let mut acked = vec![false; n];
                while acked.iter().any(|a| !a) {
                    let pending = acked.iter().filter(|a| !**a).count() as u32;
                    let slice = (Duration::from_millis(200) / pending).max(Duration::from_millis(10));
                    for i in 0..n {
                        if acked[i] {
                            continue;
                        }
                        let rank = i + 1;
                        let left = deadline.saturating_duration_since(Instant::now());
                        let verdict = if left.is_zero() {
                            Err(WireError::Io(io::Error::new(io::ErrorKind::TimedOut, "deadline")))
                        } else {
                            self.links[i].recv_polled(slice.min(left), gb_engine::boot::checkpoint)
                        };
                        let fail = match verdict {
                            Ok((_, Msg::Pong { .. })) => {
                                acked[i] = true;
                                None
                            }
                            Ok((_, Msg::Abort(why))) => Some(format!("rank {rank} refused the communicator's bootstrap: {why}")),
                            Ok((_, other)) => {
                                Some(format!("rendezvous: rank {rank} answered the bootstrap with a {} message", other.name()))
                            }
                            Err(e) if timed_out(&e) => {
                                if Instant::now() >= deadline {
                                    let missing: Vec<usize> =
                                        acked.iter().enumerate().filter(|(_, a)| !**a).map(|(i, _)| i + 1).collect();
                                    Some(format!(
                                        "{} did not acknowledge the communicator's bootstrap within {:?}",
                                        rank_list(&missing),
                                        self.timeout
                                    ))
                                } else {
                                    None
                                }
                            }
                            Err(e) => Some(format!("rendezvous: waiting for rank {rank}'s acknowledgement: {e}")),
                        };
                        if let Some(why) = fail {
                            // nobody joins the communicator on a refusal: the others are told, by name, and the boot ends
                            for (j, l) in self.links.iter_mut().enumerate() {
                                if j != i {
                                    let _ = l.send(0, &Msg::Abort(why.clone()));
                                    (l.close_write)();
                                }
                            }
                            return Err(ModelError::Config(why));
                        }
                    }
                }
                Ok(b)
            }
            (Role::Leader, None) => {
                Err(ModelError::Config("rendezvous: rank 0 must offer the communicator's bootstrap bytes".into()))
            }
            (Role::Follower, _) => {
                let link = &mut *self.links[0];
                match link.recv_polled(self.timeout, gb_engine::boot::checkpoint) {
                    Ok((_, Msg::CommInit(b))) => {
                        if let Err(why) = check(&b) {
                            link.refuse(&why);
                            return Err(ModelError::Config(why));
                        }
                        link.send(0, &Msg::Pong { nonce: 0, applied: 0 })
                            .map_err(|e| ModelError::Config(format!("rendezvous: acknowledging the bootstrap: {e}")))?;
                        Ok(b)
                    }
                    Ok((_, Msg::Abort(why))) => Err(ModelError::Config(format!("rank 0 aborted the boot: {why}"))),
                    Ok((_, other)) => Err(ModelError::Config(format!("rendezvous: rank 0 sent a {} message", other.name()))),
                    Err(e) => Err(ModelError::Config(format!("rendezvous: {e}"))),
                }
            }
        }
    }
}

/// Slack on top of `--tp-ready-timeout` before [`BootGuard`] gives up: a rank's own shard load comes first (a cold read of the pack is
/// minutes), and the timeout is the wait for the PEER's shard after it.
pub const BOOT_SLACK: Duration = Duration::from_secs(300);

/// Bounds a boot that cannot be interrupted from the host. After the connection is up the ranks meet in calls that block until the
/// other has arrived and that no thread can cancel: `ncclCommInitRank`, and the draft head's gather of rows at load. A peer that died
/// (refused, ran out of memory, lost its pack) leaves the other in one of them for ever, and no supervisor can tell that from a slow
/// load. When the guard is not dropped within `limit`, the process says so and aborts (as the serving watchdog does for a model call
/// stuck with a lost peer: `exit` would run the CUDA teardown, which waits for the stuck kernel).
pub struct BootGuard {
    state: Arc<(Mutex<bool>, Condvar)>,
}

impl BootGuard {
    /// Arms the guard of `rank`; `phase` names what it bounds in the message.
    pub fn arm(rank: usize, phase: &'static str, limit: Duration) -> BootGuard {
        Self::arm_with(limit, move || {
            eprintln!(
                "rexl3 level=fatal event=tp_boot_timeout rank={rank} phase=\"{phase}\" limit_s={} reason=\"the boot did not finish: the \
                 other rank stopped answering inside a call that cannot be interrupted (its own log says why; --tp-ready-timeout sets \
                 this bound); aborting\"",
                limit.as_secs()
            );
            std::process::abort();
        })
    }

    /// The guard with its own expiry action (tests).
    pub fn arm_with(limit: Duration, on_expiry: impl FnOnce() + Send + 'static) -> BootGuard {
        let state = Arc::new((Mutex::new(false), Condvar::new()));
        let s = state.clone();
        std::thread::Builder::new()
            .name("tp-boot-guard".into())
            .spawn(move || {
                let (m, cv) = &*s;
                let g = m.lock().unwrap_or_else(|p| p.into_inner());
                let (g, r) = cv.wait_timeout_while(g, limit, |done| !*done).unwrap_or_else(|p| p.into_inner());
                let finished = *g;
                drop(g);
                if r.timed_out() && !finished {
                    on_expiry();
                }
            })
            .ok();
        BootGuard { state }
    }
}

impl Drop for BootGuard {
    fn drop(&mut self) {
        let (m, cv) = &*self.state;
        *m.lock().unwrap_or_else(|p| p.into_inner()) = true;
        cv.notify_all();
    }
}

/// Rank 0's boot after the connection is up and gated, for ONE follower (a two-rank group): the handshake, then
/// [`boot_leader_group`]. `local_ready` reads the loaded model's vocabulary and head.
pub fn boot_leader<R, X, L>(
    mut link: Link<R::Call>,
    local: &BootGate<R::Call>,
    t: &TpTimeouts,
    make_comm: impl FnOnce(&mut Rendezvous<'_, R::Call>) -> Result<X, ModelError>,
    load: L,
) -> Result<TpLeader<R>, ModelError>
where
    R: LockstepRuntime,
    L: FnOnce(X) -> Result<R, ModelError>,
{
    handshake(&mut link, Role::Leader, local, t.ready)?;
    boot_leader_group(vec![link], local, t, make_comm, load)
}

/// Rank 0's boot after every follower is connected, identified and gated ([`accept_group`], [`admit_links`]; `links[r - 1]` is rank
/// `r`'s): the communicator (`make_comm`, which may offer bytes through the rendezvous to every follower and waits for ALL their
/// acknowledgements before it returns, so nobody joins a communicator another rank refused), the shard (`load`, given the
/// communicator), every follower's Ready under ONE deadline, and the running leader. A refusal from any rank is a `Config` error naming
/// the rank and the facts that differ.
pub fn boot_leader_group<R, X, L>(
    mut links: Vec<Link<R::Call>>,
    _local: &BootGate<R::Call>,
    t: &TpTimeouts,
    make_comm: impl FnOnce(&mut Rendezvous<'_, R::Call>) -> Result<X, ModelError>,
    load: L,
) -> Result<TpLeader<R>, ModelError>
where
    R: LockstepRuntime,
    L: FnOnce(X) -> Result<R, ModelError>,
{
    let _guard = BootGuard::arm(0, "leader boot", t.ready + BOOT_SLACK);
    let comm = {
        let mut rv = Rendezvous::group(&mut links, t.ready);
        match make_comm(&mut rv) {
            Ok(c) => c,
            Err(e) => {
                abort_links(&mut links, &e.to_string());
                return Err(e);
            }
        }
    };
    let model = match load(comm) {
        Ok(m) => m,
        Err(e) => {
            abort_links(&mut links, &format!("rank 0 failed to load its shard: {e}"));
            return Err(e);
        }
    };
    let mine = ReadyInfo { vocab: model.vocab_size() as u64, has_mtp: model.has_mtp(), snapshots: R::snapshots_supported() };
    // every follower's Ready under ONE deadline, counted from the moment this rank's own shard is loaded (not restarted per follower)
    let deadline = Instant::now() + t.ready;
    let n = links.len();
    let mut ready: Vec<Option<ReadyInfo>> = vec![None; n];
    while ready.iter().any(Option::is_none) {
        let pending = ready.iter().filter(|r| r.is_none()).count() as u32;
        let slice = (Duration::from_millis(200) / pending).max(Duration::from_millis(10));
        for i in 0..n {
            if ready[i].is_some() {
                continue;
            }
            let rank = i + 1;
            let left = deadline.saturating_duration_since(Instant::now());
            let got = if left.is_zero() {
                Err(WireError::Io(io::Error::new(io::ErrorKind::TimedOut, "deadline")))
            } else {
                links[i].recv_polled(slice.min(left), gb_engine::boot::checkpoint)
            };
            let fail = match got {
                Ok((_, Msg::Ready(theirs))) => {
                    if theirs != mine {
                        let why = format!("the loaded shards disagree: rank 0 has {mine:?}, rank {rank} has {theirs:?}");
                        let items = links
                            .drain(..)
                            .enumerate()
                            .map(|(j, l)| (l, if j == i { why.clone() } else { format!("the group's boot was refused: {why}") }))
                            .collect();
                        refuse_links(items);
                        return Err(ModelError::Config(why));
                    }
                    ready[i] = Some(theirs);
                    None
                }
                Ok((_, Msg::Abort(why))) => Some(format!("rank {rank} failed to load its shard: {why}")),
                Ok((_, other)) => Some(format!("waiting for rank {rank}'s shard: it sent a {} message", other.name())),
                Err(e) if timed_out(&e) => {
                    if Instant::now() >= deadline {
                        let missing: Vec<usize> =
                            ready.iter().enumerate().filter(|(_, r)| r.is_none()).map(|(i, _)| i + 1).collect();
                        Some(if missing.len() == 1 {
                            format!("rank {} did not finish loading its shard within {:?}", missing[0], t.ready)
                        } else {
                            format!("{} did not finish loading their shards within {:?}", rank_list(&missing), t.ready)
                        })
                    } else {
                        None
                    }
                }
                Err(e) => Some(format!("waiting for rank {rank}'s shard: {e}")),
            };
            if let Some(why) = fail {
                abort_links(&mut links, &why);
                return Err(ModelError::Config(why));
            }
        }
    }
    Ok(TpLeader::lead_group(model, links, t.clone()))
}

/// Rank `local.rank`'s boot after the connection is up (it is a follower: any rank above 0): the handshake (which yields the options to
/// load with: the leader's), the communicator, the shard, and Ready. The runtime it returns goes to [`run_follower`] with the same
/// link.
pub fn boot_follower<R, X, L>(
    mut link: Link<R::Call>,
    local: &BootGate<R::Call>,
    t: &TpTimeouts,
    make_comm: impl FnOnce(&mut Rendezvous<'_, R::Call>) -> Result<X, ModelError>,
    load: L,
) -> Result<(R, Link<R::Call>), ModelError>
where
    R: LockstepRuntime,
    L: FnOnce(&BootGate<R::Call>, X) -> Result<R, ModelError>,
{
    let _guard = BootGuard::arm(local.rank as usize, "follower boot", t.ready + BOOT_SLACK);
    let leader = handshake(&mut link, Role::Follower, local, t.ready)?;
    let comm = make_comm(&mut Rendezvous::pair(&mut link, Role::Follower, t.ready))?;
    let model = match load(&leader, comm) {
        Ok(m) => m,
        Err(e) => {
            link.refuse(&e.to_string());
            return Err(e);
        }
    };
    let info = ReadyInfo { vocab: model.vocab_size() as u64, has_mtp: model.has_mtp(), snapshots: R::snapshots_supported() };
    link.send(0, &Msg::Ready(info)).map_err(|e| ModelError::Config(format!("sending Ready to rank 0: {e}")))?;
    Ok((model, link))
}

/// Rank 0's listening socket: bound early in the boot so the followers can connect while rank 0 is still planning and loading.
pub fn listen(addr: &str) -> Result<std::net::TcpListener, ModelError> {
    std::net::TcpListener::bind(addr)
        .map_err(|e| ModelError::Config(format!("--tp-peer {addr}: rank 0 cannot listen on it: {e}")))
}

/// Rank 0 of a two-rank group: accepts rank 1's connection, waiting at most `timeout`. ([`accept_group`] is the general form.)
pub fn accept_follower<C: FamilyCall>(
    l: &std::net::TcpListener,
    timeout: Duration,
    write_timeout: Duration,
) -> Result<Link<C>, ModelError> {
    l.set_nonblocking(true).map_err(|e| ModelError::Config(format!("tensor-parallel listener: {e}")))?;
    let deadline = Instant::now() + timeout;
    let mut next_note = Instant::now() + Duration::from_secs(15);
    loop {
        gb_engine::boot::checkpoint();
        match l.accept() {
            Ok((s, from)) => {
                s.set_nonblocking(false).map_err(|e| ModelError::Config(format!("tensor-parallel connection: {e}")))?;
                eprintln!("rexl3 event=tp_follower_connected from={from}");
                return Link::tcp(s, write_timeout).map_err(|e| ModelError::Config(format!("tensor-parallel connection: {e}")));
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(ModelError::Config(format!(
                        "rank 1 did not connect within {timeout:?} (start `rexl3-serve --tp-rank 1 --tp-peer {}` on the other node)",
                        l.local_addr().map(|a| a.to_string()).unwrap_or_default()
                    )));
                }
                if Instant::now() >= next_note {
                    eprintln!("rexl3 event=tp_waiting_for_follower");
                    next_note = Instant::now() + Duration::from_secs(15);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(ModelError::Config(format!("tensor-parallel accept: {e}"))),
        }
    }
}

/// Rank 1: connects to rank 0's rendezvous address, retrying until `timeout` (rank 0 may still be starting). ([`connect_leader_as`]
/// is the general form.)
pub fn connect_leader<C: FamilyCall>(addr: &str, timeout: Duration, write_timeout: Duration) -> Result<Link<C>, ModelError> {
    connect_leader_as(1, addr, timeout, write_timeout)
}

/// Follower `rank` (any rank above 0): connects to rank 0's rendezvous address, retrying until `timeout` (rank 0 may still be
/// starting).
pub fn connect_leader_as<C: FamilyCall>(
    rank: usize,
    addr: &str,
    timeout: Duration,
    write_timeout: Duration,
) -> Result<Link<C>, ModelError> {
    use std::net::ToSocketAddrs;
    let deadline = Instant::now() + timeout;
    let mut next_note = Instant::now() + Duration::from_secs(15);
    let mut last = String::new();
    loop {
        gb_engine::boot::checkpoint();
        let resolved = addr.to_socket_addrs().map_err(|e| ModelError::Config(format!("--tp-peer {addr}: {e}")))?;
        for sa in resolved {
            match std::net::TcpStream::connect_timeout(&sa, Duration::from_secs(5)) {
                Ok(s) => {
                    eprintln!("rexl3 event=tp_connected_to_leader addr={sa}");
                    return Link::tcp(s, write_timeout)
                        .map_err(|e| ModelError::Config(format!("tensor-parallel connection: {e}")));
                }
                Err(e) => last = e.to_string(),
            }
        }
        if Instant::now() >= deadline {
            return Err(ModelError::Config(format!(
                "rank {rank} could not connect to rank 0 at {addr} within {timeout:?}: {last}"
            )));
        }
        if Instant::now() >= next_note {
            eprintln!("rexl3 event=tp_waiting_for_leader addr={addr} last_error=\"{last}\"");
            next_note = Instant::now() + Duration::from_secs(15);
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

// ------------------------------------------------------------------------------------------------------------------- tests: a toy family
//
// The seam proved with a SECOND family: a toy runtime that implements only the core trait and a trivial `FamilyCall`, over socket pairs, at
// R = 2 and R = 4, with nothing from `families::` in scope. Everything the Nemotron tests check of the protocol (same call stream on every
// rank, a digest compare per rank, loss of any one rank, the boot gate) is checked here against a family that shares no code with it.

#[cfg(test)]
mod tests {
    use super::*;
    use gb_engine::RowRequest;

    type Log = Arc<Mutex<Vec<String>>>;

    /// The toy family's commands: one whose result is compared across ranks, one with nothing to compare.
    #[derive(Clone, Debug, PartialEq)]
    enum ToyCall {
        /// Adds `by` to the slot's counter on every rank; the counter afterwards is the digest.
        Poke { slot: u32, by: u64 },
        /// Writes a note into the call log.
        Note(String),
        /// A cache call: answered with the rank's `cache_digest` (or its `cache_error`), after `cache_sleep`.
        Cache { slot: u32 },
    }

    /// The toy family's boot facts: one text.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ToyFacts {
        flavour: String,
    }

    const K_POKE: u8 = 40;
    const K_NOTE: u8 = 41;
    const K_CACHE: u8 = 42;

    impl FamilyCall for ToyCall {
        type Facts = ToyFacts;

        fn kind(&self) -> u8 {
            match self {
                ToyCall::Poke { .. } => K_POKE,
                ToyCall::Note(_) => K_NOTE,
                ToyCall::Cache { .. } => K_CACHE,
            }
        }
        fn name(&self) -> &'static str {
            match self {
                ToyCall::Poke { .. } => "poke",
                ToyCall::Note(_) => "note",
                ToyCall::Cache { .. } => "cache",
            }
        }
        fn encode(&self, e: &mut Enc) {
            match self {
                ToyCall::Poke { slot, by } => {
                    e.u32(*slot);
                    e.u64(*by);
                }
                ToyCall::Note(s) => e.str(s),
                ToyCall::Cache { slot } => e.u32(*slot),
            }
        }
        fn decode(kind: u8, d: &mut Dec<'_>) -> Result<Option<ToyCall>, WireError> {
            Ok(Some(match kind {
                K_POKE => ToyCall::Poke { slot: d.u32("slot")?, by: d.u64("by")? },
                K_NOTE => ToyCall::Note(d.str("note")?),
                K_CACHE => ToyCall::Cache { slot: d.u32("slot")? },
                _ => return Ok(None),
            }))
        }
        fn encode_facts(f: &ToyFacts, e: &mut Enc) {
            e.str(&f.flavour);
        }
        fn decode_facts(d: &mut Dec<'_>) -> Result<ToyFacts, WireError> {
            Ok(ToyFacts { flavour: d.str("flavour")? })
        }
        fn check_facts(l: &ToyFacts, f: &ToyFacts, who: u32) -> Vec<String> {
            fact_differs("flavour", &l.flavour, &f.flavour, who).into_iter().collect()
        }
    }

    /// A deterministic stand-in for a model: candidates are a pure function of the batch, so ranks that see the same calls answer alike.
    struct Toy {
        log: Log,
        vocab: usize,
        /// Perturbs the candidates of every window (a rank that computes something else).
        skew: f32,
        /// Added to every poke on this rank (a rank whose family state diverges).
        poke_bias: u64,
        /// `window_candidates` sleeps this long, or until `abort` is set (a rank stuck in a collective with a dead sibling).
        window_sleep: Duration,
        abort: Option<Arc<AtomicBool>>,
        /// What the aborted `window_candidates` returns: this CUDA error code (a collective that the communicator's abort cut short),
        /// or, with `None`, the toy's own text.
        abort_with: Option<i32>,
        counters: BTreeMap<u32, u64>,
        /// What this rank answers a cache call: its digest, or an error (`Some(true)`: fatal), after sleeping `cache_sleep` (cut short
        /// by `abort`).
        cache_digest: u64,
        cache_error: Option<bool>,
        cache_sleep: Duration,
    }

    impl Toy {
        fn new(log: Log) -> Toy {
            Toy {
                log,
                vocab: 500,
                skew: 0.0,
                poke_bias: 0,
                window_sleep: Duration::ZERO,
                abort: None,
                abort_with: None,
                counters: BTreeMap::new(),
                cache_digest: 5,
                cache_error: None,
                cache_sleep: Duration::ZERO,
            }
        }

        fn cache(&mut self, slot: u32) -> Result<u64, ModelError> {
            let until = Instant::now() + self.cache_sleep;
            while let Some(left) = until.checked_duration_since(Instant::now()) {
                if self.abort.as_ref().is_some_and(|f| f.load(Ordering::SeqCst)) {
                    return Err(ModelError::StateLost("the communicator was aborted".into()));
                }
                std::thread::sleep(left.min(Duration::from_millis(2)));
            }
            self.note("cache", slot.to_string());
            match self.cache_error {
                None => Ok(self.cache_digest),
                Some(false) => Err(ModelError::Request(format!("slot {slot} holds no such snapshot"))),
                Some(true) => Err(ModelError::StateLost(format!("slot {slot}: a sticky device error"))),
            }
        }

        fn note(&mut self, name: &str, detail: String) {
            self.log.lock().unwrap().push(format!("{name} {detail}"));
        }

        fn poke(&mut self, slot: u32, by: u64) -> Result<u64, ModelError> {
            self.note("poke", format!("{slot} {by}"));
            let c = self.counters.entry(slot).or_default();
            *c += by + self.poke_bias;
            Ok(*c)
        }

        fn cands(&self, b: &RowBatch, count: usize) -> Candidates {
            let (mut ids, mut vals) = (Vec::new(), Vec::new());
            for r in 0..b.rows {
                for j in 0..count {
                    ids.push(((b.ids[r] as usize * 17 + j * 5 + b.positions[r] as usize) % self.vocab) as i32);
                    vals.push(8.0 - j as f32 * 0.25 + (b.positions[r] % 3) as f32 + self.skew);
                }
            }
            Candidates { rows: b.rows, count, ids, vals }
        }
    }

    impl LockstepRuntime for Toy {
        type Call = ToyCall;

        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
            self.note("begin_slot", slot.to_string());
            Ok(())
        }
        fn end_slot(&mut self, slot: usize) {
            self.note("end_slot", slot.to_string());
        }
        fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
            self.note("prefill_chunk", format!("slots={:?} rows={} want={want:?}", b.seq_slot, b.rows));
            Ok(want.map(|w| {
                let mut one = b.clone();
                one.ids = vec![b.ids[w]];
                one.positions = vec![b.positions[w]];
                one.rows = 1;
                self.cands(&one, 4)
            }))
        }
        fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
            let until = Instant::now() + self.window_sleep;
            while let Some(left) = until.checked_duration_since(Instant::now()) {
                if self.abort.as_ref().is_some_and(|f| f.load(Ordering::SeqCst)) {
                    return Err(match self.abort_with {
                        Some(code) => ModelError::Cuda(gb_cuda::CudaError(code)),
                        None => ModelError::StateLost("the communicator was aborted".into()),
                    });
                }
                std::thread::sleep(left.min(Duration::from_millis(2)));
            }
            self.note("window_candidates", format!("slots={:?} rows={} count={count}", b.seq_slot, b.rows));
            Ok(self.cands(b, count))
        }
        fn commit(&mut self, slot: usize, keep: usize) {
            self.note("commit", format!("{slot} keep={keep}"));
        }
        fn flush(&mut self) -> Result<(), ModelError> {
            self.note("flush", String::new());
            Ok(())
        }
        fn apply_call(&mut self, _seq: u64, call: ToyCall) -> Result<CallReply, ModelError> {
            match call {
                ToyCall::Poke { slot, by } => self.poke(slot, by).map(CallReply::digest),
                ToyCall::Note(s) => {
                    self.note("note", s);
                    Ok(CallReply::NONE)
                }
                ToyCall::Cache { slot } => self.cache(slot).map(CallReply::digest),
            }
        }
    }

    // what a family's implementation of its own runtime trait on the leader would be: the forwarding primitives
    fn poke(l: &mut TpLeader<Toy>, slot: u32, by: u64) -> Result<u64, ModelError> {
        l.forward_call(ToyCall::Poke { slot, by }, |m, _| m.poke(slot, by), |v: &u64| *v)
    }

    fn cache(l: &mut TpLeader<Toy>, slot: u32, wait: bool) -> Result<CacheOutcome<u64>, ModelError> {
        l.forward_cache_call(ToyCall::Cache { slot }, |m, _| m.cache(slot), wait)
    }

    fn note(l: &mut TpLeader<Toy>, s: &str) {
        let s = s.to_string();
        l.forward_unit(ToyCall::Note(s.clone()), move |m| m.note("note", s));
    }

    fn quick() -> TpTimeouts {
        TpTimeouts {
            connect: Duration::from_secs(5),
            ready: Duration::from_secs(10),
            peer: Duration::from_millis(1500),
            write: Duration::from_secs(2),
            ping: Duration::from_millis(100),
            barrier: Duration::from_secs(5),
            hang_exit: None,
            health_poll: Duration::from_millis(50),
        }
    }

    fn batch(reqs: &[(usize, u64, Vec<u32>)]) -> RowBatch {
        let r: Vec<RowRequest> = reqs
            .iter()
            .map(|(slot, first, toks)| RowRequest { slot: *slot, first_position: *first, tokens: toks.clone() })
            .collect();
        RowBatch::from_requests(&r)
    }

    /// What a scripted engine run asks of the runtime: slots, prefill slices (the last with candidates), family calls, windows with
    /// commits that keep fewer rows than the window had, the end of the slots.
    fn script(l: &mut TpLeader<Toy>) -> Vec<String> {
        let mut out = Vec::new();
        for slot in 0..2 {
            l.begin_slot(slot).unwrap();
        }
        l.flush().unwrap();
        assert!(l.prefill_chunk(&batch(&[(0, 0, vec![11, 12, 13])]), None).unwrap().is_none());
        let c = l.prefill_chunk(&batch(&[(1, 0, vec![21, 22])]), Some(1)).unwrap().unwrap();
        out.push(format!("prefill {:?}", c.ids));
        out.push(format!("poke {}", poke(l, 0, 5).unwrap()));
        note(l, "midway");
        for step in 0..3u64 {
            let w = l.window_candidates(&batch(&[(0, 3 + step, vec![7, 8]), (1, 2 + step, vec![9])]), 4).unwrap();
            out.push(format!("window {:?}", &w.ids[..4]));
            l.commit(0, 2);
            l.commit(1, 1);
            out.push(format!("poke {}", poke(l, 1, step + 1).unwrap()));
        }
        l.end_slot(1);
        l.end_slot(0);
        l.flush().unwrap();
        out
    }

    fn gate(rank: usize, ranks: usize) -> BootGate<ToyCall> {
        BootGate::new("toy-pack-9f3".into(), 0, 4, 100, 4096, 64, ToyFacts { flavour: "vanilla".into() }).at(rank, ranks)
    }

    /// A group of `ranks` over socket pairs: the leader (rank 0) and one `run_follower` thread a follower, all on the toy runtime.
    struct Group {
        leader: Option<TpLeader<Toy>>,
        threads: Vec<Option<std::thread::JoinHandle<Result<FollowerExit, ModelError>>>>,
        /// Closes follower `r`'s end of its link from outside: its process is gone.
        crash: Vec<Arc<dyn Fn() + Send + Sync>>,
        /// Whether follower `r`'s communicator abort hook ran.
        aborted: Vec<Arc<AtomicBool>>,
        logs: Vec<Log>,
        leader_log: Log,
    }

    fn start(ranks: usize, follower_peer: Duration, tune: impl Fn(usize, &mut Toy)) -> Group {
        start_spinning(ranks, follower_peer, Duration::ZERO, tune)
    }

    /// [`start`] with every follower's receive path spinning for `spin` before it parks (`FollowerConfig::spin`).
    fn start_spinning(ranks: usize, follower_peer: Duration, spin: Duration, tune: impl Fn(usize, &mut Toy)) -> Group {
        start_full(ranks, follower_peer, spin, tune, |_| {})
    }

    /// [`start_spinning`] with the leader's own runtime tuned too (`tune` is for the followers).
    fn start_full(
        ranks: usize,
        follower_peer: Duration,
        spin: Duration,
        tune: impl Fn(usize, &mut Toy),
        leader_tune: impl FnOnce(&mut Toy),
    ) -> Group {
        let leader_log = Log::default();
        let (mut links, mut threads) = (Vec::new(), Vec::new());
        let (mut crash, mut aborted, mut logs) = (Vec::new(), Vec::new(), Vec::new());
        for rank in 1..ranks {
            let (a, b) = Link::<ToyCall>::pair(Duration::from_secs(2)).unwrap();
            crash.push(b.close.clone());
            let log = Log::default();
            let mut toy = Toy::new(log.clone());
            let flag = Arc::new(AtomicBool::new(false));
            toy.abort = Some(flag.clone());
            tune(rank, &mut toy);
            // a spinning follower is counted by thread name: its rank is out of the plain groups' range (1..), so no other test of
            // this module (they run in parallel) has a reader of the same name
            let fr = if spin.is_zero() { rank } else { rank + 4 };
            let mut cfg = FollowerConfig::new(TpTimeouts { peer: follower_peer, ..quick() }).with_rank(fr);
            cfg.spin = spin;
            let f = flag.clone();
            cfg.abort_hook = Some(Box::new(move || f.store(true, Ordering::SeqCst)));
            aborted.push(flag);
            logs.push(log);
            threads.push(Some(std::thread::spawn(move || run_follower(toy, b, cfg))));
            links.push(a);
        }
        let mut leader_toy = Toy::new(leader_log.clone());
        leader_tune(&mut leader_toy);
        let leader = TpLeader::lead_group(leader_toy, links, quick());
        Group { leader: Some(leader), threads, crash, aborted, logs, leader_log }
    }

    impl Group {
        fn leader(&mut self) -> &mut TpLeader<Toy> {
            self.leader.as_mut().unwrap()
        }

        fn finished(&mut self, rank: usize, within: Duration) -> Option<Result<FollowerExit, ModelError>> {
            let t0 = Instant::now();
            let h = self.threads[rank - 1].as_ref()?;
            while !h.is_finished() {
                if t0.elapsed() > within {
                    return None;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Some(self.threads[rank - 1].take().unwrap().join().unwrap())
        }

        /// Drops the leader (a clean shutdown when the group is whole) and returns every follower's result, rank 1 first.
        fn shutdown(&mut self) -> Vec<Result<FollowerExit, ModelError>> {
            drop(self.leader.take());
            (1..=self.threads.len()).map(|r| self.finished(r, Duration::from_secs(10)).expect("a follower did not end")).collect()
        }
    }

    impl Drop for Group {
        fn drop(&mut self) {
            drop(self.leader.take());
            for t in self.threads.drain(..).flatten() {
                let _ = t.join();
            }
        }
    }

    #[test]
    fn the_module_imports_nothing_from_a_family() {
        // the neutrality check: the module's own `use` lines name the crate's error, rows and forward types and std, never a family
        let src = include_str!("lockstep.rs");
        for l in src.lines().filter(|l| l.trim_start().starts_with("use ")) {
            assert!(!l.contains("families") && !l.contains("nemotron"), "a family import in the neutral module: {l}");
        }
    }

    #[test]
    fn every_rank_of_a_toy_group_replays_the_same_call_stream_and_each_digest_is_compared() {
        for ranks in [2usize, 4] {
            let mut g = start(ranks, Duration::from_secs(30), |_, _| {});
            let out = script(g.leader());
            g.leader().barrier().unwrap();
            assert_eq!(g.leader().failure(), None, "{ranks} ranks");
            assert_eq!(out.len(), 8);
            let stats = g.leader().stats();
            // every command was acknowledged by every follower (the barrier waited for all of them)
            assert_eq!(stats.acked, stats.commands * (ranks as u64 - 1), "{ranks} ranks: {stats:?}");
            let st = g.leader().group_status();
            assert_eq!(
                (st.ranks, st.commands_sent, st.settled, st.lost.is_none()),
                (ranks, stats.commands, stats.commands, true)
            );
            assert_eq!(st.per_rank.len(), ranks - 1);
            for (i, r) in st.per_rank.iter().enumerate() {
                assert_eq!((r.rank, r.state, r.connected, r.diverged), (i + 1, "up", true, 0), "{r:?}");
                assert_eq!((r.acked_seq, r.lag_commands), (stats.commands, 0));
            }
            let leader_log = g.leader_log.lock().unwrap().clone();
            assert!(leader_log.iter().any(|l| l.starts_with("poke")) && leader_log.iter().any(|l| l.starts_with("note midway")));
            for (i, e) in g.shutdown().into_iter().enumerate() {
                assert_eq!(e.unwrap(), FollowerExit::Shutdown("leader shut down".into()), "rank {} of {ranks}", i + 1);
            }
            for (i, log) in g.logs.iter().enumerate() {
                assert_eq!(*log.lock().unwrap(), leader_log, "rank {} replayed another call stream ({ranks} ranks)", i + 1);
            }
        }
    }

    #[test]
    fn a_spinning_follower_replays_the_same_call_stream_and_still_ends_on_shutdown_and_on_a_crash() {
        // the spin-then-park receive (`FollowerConfig::spin`) changes when a frame is seen, never which frames or in what order: the
        // same script as above gives the same replies and the same call log on every rank, with a budget far shorter than the gaps
        // (it parks between commands) and far longer (it never parks inside the run)
        let _one = SPIN_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        for spin in [Duration::from_micros(50), Duration::from_millis(500)] {
            for ranks in [2usize, 4] {
                let mut plain = start(ranks, Duration::from_secs(30), |_, _| {});
                let want = script(plain.leader());
                plain.leader().barrier().unwrap();
                let want_log = plain.leader_log.lock().unwrap().clone();
                drop(plain);
                let mut g = start_spinning(ranks, Duration::from_secs(30), spin, |_, _| {});
                let out = script(g.leader());
                g.leader().barrier().unwrap();
                assert_eq!(out, want, "{ranks} ranks, spin {spin:?}");
                assert_eq!(g.leader().failure(), None);
                assert_eq!(*g.leader_log.lock().unwrap(), want_log);
                for (i, e) in g.shutdown().into_iter().enumerate() {
                    assert_eq!(e.unwrap(), FollowerExit::Shutdown("leader shut down".into()), "rank {} of {ranks}", i + 1);
                }
                for (i, log) in g.logs.iter().enumerate() {
                    assert_eq!(*log.lock().unwrap(), want_log, "rank {} (spin {spin:?}, {ranks} ranks)", i + 1);
                }
            }
        }
        // a connection that ends while the reader is spinning is seen as the end of the stream (the poll reports the hangup)
        let mut g = start_spinning(2, Duration::from_secs(30), Duration::from_secs(5), |_, _| {});
        g.leader().begin_slot(0).unwrap();
        g.leader().barrier().unwrap();
        (g.crash[0])();
        let exit = g.finished(1, Duration::from_secs(10)).expect("a spinning follower must see the end of its link");
        assert!(matches!(exit, Ok(FollowerExit::LeaderLost(_))), "{exit:?}");
    }

    #[test]
    fn only_command_activity_extends_the_spin_deadline() {
        let mut s = Shared::new();
        assert!(!s.busy_within(u64::MAX), "a new group is idle");
        s.touch();
        assert!(!s.busy_within(u64::MAX), "heartbeat receipt is not command activity");
        s.active();
        assert!(s.busy_within(u64::MAX), "commands activate the spin window");
        assert!(!s.busy_within(0), "a zero spin budget always parks");

        // Age the recorded command deterministically instead of sleeping or
        // requiring a minimum CPU share from a loaded hosted runner.
        s.t0 = Instant::now() - Duration::from_secs(1);
        s.last_active_us.store(1, Ordering::Relaxed);
        let window = Duration::from_millis(80).as_micros() as u64;
        assert!(!s.busy_within(window));
        s.touch();
        assert!(!s.busy_within(window), "pings must not revive an expired window");
        s.active();
        assert!(s.busy_within(u64::MAX));
    }

    /// The tests that start a spinning follower run one at a time: they count CPU by thread name, and a spinner is a CPU hog.
    static SPIN_TESTS: Mutex<()> = Mutex::new(());

    /// CPU ticks (utime + stime) of each thread of this process named `name`, by thread id. A thread that ends between two samples (the
    /// follower of the previous spin test, still winding down) must not make a total go down, so deltas are taken per thread id
    /// ([`ticks_since`]).
    #[cfg(target_os = "linux")]
    fn thread_cpu_ticks(name: &str) -> std::collections::HashMap<String, u64> {
        let mut out = std::collections::HashMap::new();
        for e in std::fs::read_dir("/proc/self/task").unwrap().flatten() {
            let p = e.path();
            if std::fs::read_to_string(p.join("comm")).map(|c| c.trim() == name).unwrap_or(false) {
                let stat = std::fs::read_to_string(p.join("stat")).unwrap_or_default();
                // fields after the parenthesised name: state is field 3, utime 14, stime 15 (1-based)
                let rest = stat.rsplit_once(')').map(|x| x.1).unwrap_or("");
                let f: Vec<&str> = rest.split_whitespace().collect();
                if f.len() > 13 {
                    let ticks = f[11].parse::<u64>().unwrap_or(0) + f[12].parse::<u64>().unwrap_or(0);
                    out.insert(e.file_name().to_string_lossy().into_owned(), ticks);
                }
            }
        }
        out
    }

    /// CPU ticks the threads present in both samples used between them.
    #[cfg(target_os = "linux")]
    fn ticks_since(before: &std::collections::HashMap<String, u64>, now: &std::collections::HashMap<String, u64>) -> u64 {
        now.iter().filter_map(|(tid, n)| before.get(tid).map(|b| n.saturating_sub(*b))).sum()
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "CPU-utilization benchmark: run alone on an idle Linux host with --ignored"]
    fn a_quiet_group_parks_a_spinning_follower_and_a_busy_one_keeps_polling() {
        // the spin follows COMMAND activity only (a link that carries pings alone parks): the reader of an idle group uses no CPU,
        // and while commands flow it polls. Threads named alike belong to every test of the module, so the spinning tests are serialised.
        let _one = SPIN_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        let mut g = start_spinning(2, Duration::from_secs(30), Duration::from_millis(80), |_, _| {});
        g.leader().begin_slot(0).unwrap();
        g.leader().barrier().unwrap();
        // commands every 20 ms for 400 ms: the group is busy the whole time, so the reader polls (about 40 ticks of 10 ms)
        let t0 = thread_cpu_ticks("tp-follower-rx5");
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(400) {
            g.leader().flush().unwrap();
            std::thread::sleep(Duration::from_millis(20));
        }
        let busy = ticks_since(&t0, &thread_cpu_ticks("tp-follower-rx5"));
        // then nothing but the leader's pings (every ping interval of `quick()`): past the 80 ms window the reader is parked
        std::thread::sleep(Duration::from_millis(300));
        let t1 = thread_cpu_ticks("tp-follower-rx5");
        std::thread::sleep(Duration::from_millis(600));
        let idle = ticks_since(&t1, &thread_cpu_ticks("tp-follower-rx5"));
        assert!(busy >= 20, "a busy group's reader should poll: {busy} ticks in 400 ms");
        assert!(idle <= 6, "a quiet group's reader must park: {idle} ticks in 600 ms");
        drop(g);
    }

    #[test]
    fn a_rank_whose_candidates_or_family_digest_differs_fails_the_group_naming_the_rank_and_the_call() {
        for ranks in [2usize, 4] {
            for bad in 1..ranks {
                // candidates that differ: a core call's digest
                let mut g = start(ranks, Duration::from_secs(30), move |r, m| m.skew = if r == bad { 0.25 } else { 0.0 });
                g.leader().begin_slot(0).unwrap();
                let w = g.leader().window_candidates(&batch(&[(0, 0, vec![4, 5])]), 4);
                let r = g.leader().barrier();
                assert!(w.is_err() || r.is_err());
                let why = g.leader().failure().expect("the group must have failed");
                assert!(
                    why.contains("diverged")
                        && why.contains("window_candidates")
                        && why.contains(&format!("rank {bad} returned")),
                    "{why}"
                );
                assert_eq!(g.leader().group_status().lost_rank, Some(bad));
                drop(g);
                // a family call's digest that differs: the counter the call produced
                let mut g = start(ranks, Duration::from_secs(30), move |r, m| m.poke_bias = (r == bad) as u64);
                g.leader().begin_slot(0).unwrap();
                let v = poke(g.leader(), 0, 3);
                let r = g.leader().barrier();
                assert!(v.is_err() || r.is_err());
                let why = g.leader().failure().expect("the group must have failed");
                assert!(
                    why.contains("diverged") && why.contains("(poke)") && why.contains(&format!("rank {bad} returned")),
                    "{why}"
                );
                assert_eq!(g.leader().group_status().lost_rank, Some(bad));
            }
        }
    }

    #[test]
    fn the_loss_of_any_one_rank_fails_the_group_naming_it_and_frees_the_healthy_survivors() {
        for ranks in [2usize, 4] {
            for dead in 1..ranks {
                // the followers' own silence watchdog is far away: only the leader closing their links can free a rank that is stuck
                // inside a call (here a sleep that the abort hook ends)
                let mut g = start(ranks, Duration::from_secs(30), |_, m| m.window_sleep = Duration::from_secs(20));
                g.leader().begin_slot(0).unwrap();
                assert!(g.leader().window_candidates(&batch(&[(0, 0, vec![7, 8])]), 4).is_ok());
                std::thread::sleep(Duration::from_millis(100));
                let t0 = Instant::now();
                (g.crash[dead - 1])();
                let mut why = None;
                while t0.elapsed() < Duration::from_secs(3) {
                    why = g.leader().failure();
                    if why.is_some() {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                let why = why.unwrap_or_else(|| panic!("{ranks} ranks, rank {dead} lost: the group was not declared lost"));
                assert!(why.contains(&format!("rank {dead}")), "{why}");
                assert!(t0.elapsed() < Duration::from_millis(1400), "{ranks} ranks, rank {dead}: {:?}", t0.elapsed());
                assert_eq!(g.leader().group_status().lost_rank, Some(dead));
                for r in 1..ranks {
                    let res = g.finished(r, Duration::from_secs(3));
                    assert!(res.is_some(), "{ranks} ranks, rank {dead} lost: rank {r} is still stuck");
                    assert!(g.aborted[r - 1].load(Ordering::SeqCst), "rank {r} did not abort its communicator");
                    assert!(!matches!(res, Some(Ok(FollowerExit::Shutdown(_)))), "rank {r} was told to shut down cleanly");
                }
                // every later call is a fatal error, the family's included, and the void calls are no-ops
                assert!(matches!(g.leader().flush(), Err(e) if e.is_fatal()));
                assert!(matches!(poke(g.leader(), 0, 1), Err(e) if e.is_fatal()));
                g.leader().commit(0, 1);
                note(g.leader(), "after the loss");
            }
        }
    }

    #[test]
    fn a_collective_cut_short_by_a_lost_peer_says_which_rank_was_lost_and_keeps_the_code() {
        use gb_cuda::nccl::ERR_ABORTED;
        // the leader is inside a call that waits for a collective; rank 1 dies; the loss is latched and the abort hook ends the call
        // with the communicator's code, as the real communicator's abort does
        let flag = Arc::new(AtomicBool::new(false));
        let f = flag.clone();
        let mut g = start_full(
            2,
            Duration::from_secs(30),
            Duration::ZERO,
            |_, _| {},
            move |m| {
                m.window_sleep = Duration::from_secs(20);
                m.abort = Some(f);
                m.abort_with = Some(ERR_ABORTED);
            },
        );
        g.leader().on_group_lost(Box::new(move || flag.store(true, Ordering::SeqCst)));
        g.leader().begin_slot(0).unwrap();
        let crash = g.crash[0].clone();
        let killer = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            crash();
        });
        let t0 = Instant::now();
        let r = g.leader().window_candidates(&batch(&[(0, 0, vec![7, 8])]), 4);
        killer.join().unwrap();
        assert!(t0.elapsed() < Duration::from_secs(5), "the aborted call did not return: {:?}", t0.elapsed());
        let e = r.expect_err("the call that lost its peer must fail");
        assert!(e.is_fatal(), "{e}");
        let text = e.to_string();
        assert!(text.contains("tensor-parallel group lost"), "{text}");
        assert!(text.contains("peer rank 1 was lost"), "{text}");
        assert!(text.contains("window_candidates (call"), "{text}");
        assert!(text.contains("CUDA error 20103 (communicator aborted)"), "{text}");
        assert!(!text.contains("unrecognised"), "{text}");
        // the latched reason, the status and the exit class are the group's own: only the returned text names the collective
        let why = g.leader().failure().expect("the group must be lost");
        assert!(why.contains("the connection to rank 1 failed") && !why.contains("20103"), "{why}");
        assert_eq!(g.leader().group_status().lost_rank, Some(1));
        assert!(matches!(g.leader().flush(), Err(e) if e.is_fatal()));
    }

    #[test]
    fn a_follower_whose_collective_was_cut_short_by_a_lost_group_names_the_loss_and_keeps_the_code() {
        use gb_cuda::nccl::ERR_ABORTED;
        // a hand-driven leader: the follower is inside window call 1 (a collective that waits) when the leader tells it the group is
        // dead; its abort hook ends the call with the communicator's code and the link is still open, so it answers and then ends
        let (mut a, b) = Link::<ToyCall>::pair(Duration::from_secs(2)).unwrap();
        let flag = Arc::new(AtomicBool::new(false));
        let mut toy = Toy::new(Log::default());
        toy.window_sleep = Duration::from_secs(20);
        toy.abort = Some(flag.clone());
        toy.abort_with = Some(ERR_ABORTED);
        let mut cfg = FollowerConfig::new(quick()).with_rank(1);
        cfg.abort_hook = Some(Box::new(move || flag.store(true, Ordering::SeqCst)));
        let h = std::thread::spawn(move || run_follower(toy, b, cfg));
        a.send(1, &Msg::Window { batch: batch(&[(0, 0, vec![7, 8])]), count: 4 }).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        a.send(0, &Msg::Abort("rank 2 is gone".into())).unwrap();
        let mut ack = None;
        while let Ok((seq, m)) = a.recv(Some(Duration::from_secs(5))) {
            if let Msg::Ack(class, text) = m {
                ack = Some((seq, class, text));
                break;
            }
        }
        let (seq, class, text) = ack.expect("the follower must answer the call");
        assert_eq!(seq, 1);
        // the wire class is the call's own (a CUDA error, fatal): the leader compares classes, never text
        assert!(matches!(class, Class::Err { kind: 2, fatal: true }), "{class:?}");
        assert!(text.contains("rank 0 aborted the group: rank 2 is gone"), "{text}");
        assert!(text.contains("CUDA error 20103 (communicator aborted)"), "{text}");
        assert!(!text.contains("unrecognised"), "{text}");
        let res = h.join().unwrap();
        match res {
            Err(e) => assert!(e.to_string().contains("CUDA error 20103 (communicator aborted)"), "{e}"),
            other => panic!("the follower must end with its fatal error, got {other:?}"),
        }
    }

    #[test]
    fn only_a_collective_failure_after_a_latched_loss_is_renamed() {
        use gb_cuda::nccl::{ERR_ABORTED, ERR_BAD_ARGUMENT};
        let cuda = |c: i32| ModelError::Cuda(gb_cuda::CudaError(c));
        let why = "the connection to rank 1 failed: connection closed by the peer";
        // the loss is latched and rank 1 is blamed: a peer rank was lost, the reason and the raw code stay
        let e = lost_peer_error(cuda(ERR_ABORTED), Some(why), Some(1), "window_candidates (call 12)");
        assert!(e.is_fatal());
        let t = e.to_string();
        assert!(t.contains("tensor-parallel group lost: peer rank 1 was lost"), "{t}");
        assert!(t.contains(why) && t.contains("window_candidates (call 12)"), "{t}");
        assert!(t.contains("CUDA error 20103 (communicator aborted)"), "{t}");
        // latched, no rank to blame: the reason carries it, nothing is invented
        let t = lost_peer_error(cuda(ERR_ABORTED), Some("rank 0 aborted the group: x"), None, "call 3").to_string();
        assert!(t.contains("rank 0 aborted the group: x") && !t.contains("peer rank") && t.contains("20103"), "{t}");
        // an NCCL result of the collective itself is named too (ncclRemoteError)
        let t = lost_peer_error(cuda(20_006), Some(why), Some(1), "flush (call 4)").to_string();
        assert!(t.contains("peer rank 1 was lost") && t.contains("CUDA error 20006 (ncclRemoteError)"), "{t}");
        // not latched before the result: the error is this rank's own and is what loses the group, so it is returned as it was
        assert!(matches!(lost_peer_error(cuda(ERR_ABORTED), None, None, "x"), ModelError::Cuda(c) if c.0 == ERR_ABORTED));
        // not a collective's code: a device fault stays a device fault, whatever else is lost
        assert!(matches!(lost_peer_error(cuda(700), Some(why), Some(1), "x"), ModelError::Cuda(c) if c.0 == 700));
        assert!(matches!(lost_peer_error(cuda(2), Some(why), Some(1), "x"), ModelError::Cuda(c) if c.0 == 2));
        assert!(matches!(lost_peer_error(cuda(ERR_BAD_ARGUMENT), Some(why), Some(1), "x"), ModelError::Cuda(_)));
        assert!(matches!(lost_peer_error(ModelError::Request("r".into()), Some(why), Some(1), "x"), ModelError::Request(_)));
    }

    #[test]
    fn a_toy_group_boots_gates_and_serves_at_two_and_four_ranks() {
        for ranks in [2usize, 4] {
            let t = quick();
            let (mut leader_ends, mut followers) = (Vec::new(), Vec::new());
            let logs: Vec<Log> = (0..ranks).map(|_| Log::default()).collect();
            for (rank, log) in logs.iter().enumerate().skip(1) {
                let (a, b) = Link::<ToyCall>::pair(Duration::from_secs(2)).unwrap();
                leader_ends.push(a);
                let (tf, g, log) = (t.clone(), gate(rank, ranks), log.clone());
                followers.push(std::thread::spawn(move || {
                    let (rt, link) = boot_follower(
                        b,
                        &g,
                        &tf,
                        |rv| rv.exchange(None),
                        move |leader: &BootGate<ToyCall>, id: Vec<u8>| {
                            // what the follower loads with is what the leader planned, and the bootstrap arrived
                            assert_eq!((leader.rank, leader.slots, leader.family.flavour.as_str()), (0, 4, "vanilla"));
                            assert_eq!(id, b"toy-bootstrap");
                            Ok(Toy::new(log))
                        },
                    )
                    .unwrap();
                    run_follower(rt, link, FollowerConfig::new(tf).with_rank(rank))
                }));
            }
            let g = gate(0, ranks);
            let links = admit_links(leader_ends, &g, &t).unwrap();
            let log0 = logs[0].clone();
            let mut leader = boot_leader_group(
                links,
                &g,
                &t,
                |rv| rv.exchange(Some(b"toy-bootstrap".to_vec())),
                move |id: Vec<u8>| {
                    assert_eq!(id, b"toy-bootstrap");
                    Ok(Toy::new(log0))
                },
            )
            .unwrap();
            assert_eq!(leader.ranks(), ranks);
            let out = script(&mut leader);
            assert_eq!(out.len(), 8);
            leader.barrier().unwrap();
            assert_eq!(leader.failure(), None);
            drop(leader);
            for f in followers {
                assert!(matches!(f.join().unwrap(), Ok(FollowerExit::Shutdown(_))));
            }
            let want = logs[0].lock().unwrap().clone();
            for l in &logs[1..] {
                assert_eq!(*l.lock().unwrap(), want);
            }
        }
    }

    #[test]
    fn the_gate_refuses_a_different_pack_a_family_fact_and_a_smaller_pool_by_name_on_every_link() {
        for ranks in [2usize, 4] {
            for (what, tweak) in [("pack", 0u8), ("flavour", 1u8), ("slots", 2u8)] {
                let bad = ranks - 1;
                let (mut leader_ends, mut followers) = (Vec::new(), Vec::new());
                for rank in 1..ranks {
                    let (a, mut b) = Link::<ToyCall>::pair(Duration::from_secs(2)).unwrap();
                    leader_ends.push(a);
                    let mut g = gate(rank, ranks);
                    if rank == bad {
                        match tweak {
                            0 => g.pack = "another-pack".into(),
                            1 => g.family.flavour = "chocolate".into(),
                            _ => g.slots = 2,
                        }
                    }
                    followers.push(std::thread::spawn(move || handshake(&mut b, Role::Follower, &g, Duration::from_secs(5))));
                }
                let e = admit_links(leader_ends, &gate(0, ranks), &quick()).unwrap_err().to_string();
                assert!(e.contains(what) && e.contains(&format!("rank {bad}")), "{what}: {e}");
                // every follower is told: the offender by its own gate's check, the others by the Abort that carries the reason
                for f in followers {
                    let _ = f.join().unwrap();
                }
            }
        }
    }

    // ------------------------------------------------------------------------------------------------- the build and NCCL identity

    #[test]
    fn the_gate_names_each_differing_part_of_the_build_and_keeps_the_old_label_for_the_first_line() {
        let a = build_fingerprint_of("1.0.0-dev", "6a2879b137fd8b97", Some("e45ae8bce7cb"), &"a".repeat(64), 250_000);
        assert_eq!(
            a,
            format!("1.0.0-dev+6a2879b137fd8b97\ncommit=e45ae8bce7cb\nexe_sha256={}\ntp_spin_us=250000", "a".repeat(64))
        );
        assert!(build_differs(&a, &a, 1).is_empty());
        // another commit and another binary: both named, the sha shortened to 16 hex
        let b = build_fingerprint_of("1.0.0-dev", "6a2879b137fd8b97", Some("57821a4b41f6-dirty"), &"b".repeat(64), 250_000);
        let d = build_differs(&a, &b, 1);
        assert_eq!(d.len(), 2, "{d:?}");
        assert!(
            d[0].starts_with("build commit") && d[0].contains("rank 0 has e45ae8bce7cb, rank 1 has 57821a4b41f6-dirty"),
            "{d:?}"
        );
        assert!(
            d[1].starts_with("build binary sha256") && d[1].contains(&"b".repeat(16)) && !d[1].contains(&"b".repeat(17)),
            "{d:?}"
        );
        // the same commit, a rebuilt binary (an unregistered commit proves nothing: the bytes still differ)
        let c = build_fingerprint_of("1.0.0-dev", "6a2879b137fd8b97", None, &"c".repeat(64), 250_000);
        let e = build_fingerprint_of("1.0.0-dev", "6a2879b137fd8b97", None, &"e".repeat(64), 250_000);
        assert!(c.contains("commit=unregistered"));
        assert_eq!(build_differs(&c, &e, 1).len(), 1);
        // the spin: latency only, but named
        let s0 = build_fingerprint_of("1.0.0-dev", "k", Some("x"), "h", 0);
        let d = build_differs(&s0, &build_fingerprint_of("1.0.0-dev", "k", Some("x"), "h", 250_000), 1);
        assert_eq!(
            d,
            vec!["follower spin (REXL3_TP_SPIN_US; latency only, pass the same to every rank): rank 0 has 0, rank 1 has 250000"]
        );
        // the first line keeps protocol 4's label; a peer from before these fields says so
        let d = build_differs("1.0.0-dev+aaaa", &s0, 2);
        assert!(d
            .iter()
            .any(|l| l.starts_with("build (crate version and kernel hash): rank 0 has 1.0.0-dev+aaaa, rank 2 has 1.0.0-dev+k")));
        assert!(d.iter().any(|l| l.contains("rank 0 has (absent: a build from before this field)")), "{d:?}");
    }

    fn env(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn the_nccl_environment_is_canonical_and_leaves_out_diagnostics_and_named_per_node_variables() {
        let base = [
            ("NCCL_IB_HCA", "=rocep1s0f0:1,roceP2p1s0f0:1"),
            ("NCCL_IB_TC", "106"),
            ("NCCL_SOCKET_IFNAME", "=enP7s7"),
            ("NCCL_MAX_NCHANNELS", "4"),
            ("LD_LIBRARY_PATH", "/x"),
            ("REXL3_TP_SPIN_US", "0"),
        ];
        let a = nccl_env_lines(&env(&base), None, Some("abc"));
        // order of the process environment does not matter; NCCL_DEBUG*, non-NCCL variables do not count
        let mut shuffled = env(&base);
        shuffled.reverse();
        shuffled.push(("NCCL_DEBUG".into(), "INFO".into()));
        shuffled.push(("NCCL_DEBUG_FILE".into(), "/tmp/nccl.%h.%p".into()));
        assert_eq!(nccl_env_lines(&shuffled, None, Some("abc")), a);
        assert!(
            a.contains("\nNCCL_IB_HCA==rocep1s0f0:1,roceP2p1s0f0:1\n") && !a.contains("LD_LIBRARY_PATH") && !a.contains("REXL3")
        );
        // the effective channel cap when the variable is unset (what make_comm exports)
        let unset: Vec<_> = env(&base).into_iter().filter(|(k, _)| k != "NCCL_MAX_NCHANNELS").collect();
        assert_eq!(nccl_env_lines(&unset, None, Some("abc")), a);
        // the library's bytes are part of the hash
        let other_lib = nccl_env_lines(&env(&base), None, Some("def"));
        let hash = |s: &str| gate_lines(&format!("channels=4\n{s}"))["env_sha256"].clone();
        assert_ne!(hash(&a), hash(&other_lib));
        // a per-node variable is left out, and the list itself is compared
        let mut n4 = env(&base);
        n4[0].1 = "=mlx5_0:1".into();
        let a_pn = nccl_env_lines(&env(&base), Some("NCCL_IB_HCA"), Some("abc"));
        let b_pn = nccl_env_lines(&n4, Some(" NCCL_IB_HCA ,"), Some("abc"));
        assert_eq!(a_pn, b_pn);
        assert!(nccl_differs(&a_pn, &a, 1).iter().any(|l| l.starts_with("NCCL per-node variables")));
    }

    #[test]
    fn the_gate_names_each_differing_nccl_variable_the_library_and_the_settings_line() {
        let head = nccl_fingerprint_of(false, None, None, Some(23007));
        let base = [("NCCL_IB_TC", "106"), ("NCCL_NET", "IB")];
        let a = format!("{head}\n{}", nccl_env_lines(&env(&base), None, Some("abc")));
        assert!(nccl_differs(&a, &a, 1).is_empty());
        // node 4 without NCCL_IB_TC and with another libnccl
        let b = format!("{head}\n{}", nccl_env_lines(&env(&base[1..]), None, Some("def")));
        let d = nccl_differs(&a, &b, 1);
        assert_eq!(d.len(), 2, "{d:?}");
        assert!(d.iter().any(|l| l.starts_with("NCCL library file") && l.contains("rank 0 has abc, rank 1 has def")), "{d:?}");
        assert!(
            d.iter().any(|l| l
                == "NCCL environment variable NCCL_IB_TC (set it the same on every rank): rank 0 has 106, rank 1 has (unset)"),
            "{d:?}"
        );
        // the settings line keeps protocol 4's label (the refusal tests and the verdict match it)
        let c = format!(
            "{}\n{}",
            nccl_fingerprint_of(false, Some("2"), None, Some(23007)),
            nccl_env_lines(&env(&base), None, Some("abc"))
        );
        let d = nccl_differs(&a, &c, 1);
        assert!(d.iter().any(|l| l.starts_with(
            "NCCL settings and library (NCCL_MAX_NCHANNELS, NCCL_PROTO, library version): rank 0 has channels=4,proto=default,lib=23007"
        )));
        // a group without a communicator compares `off`
        assert!(nccl_differs("off", "off", 1).is_empty());
        assert_eq!(nccl_summary(&a), format!("{head} env_sha256={}", gate_lines(&a)["env_sha256"]));
        assert_eq!(nccl_summary("off"), "off");
    }

    #[test]
    fn a_mismatched_build_or_nccl_variable_is_refused_at_the_hello_on_the_offending_link() {
        let (a, mut b) = Link::<ToyCall>::pair(Duration::from_secs(2)).unwrap();
        let mut g = gate(1, 2);
        g.build = build_fingerprint_of("1.0.0-dev", "k", Some("other"), "f00d", 250_000);
        g.nccl = format!("channels=4,proto=default,lib=23007\n{}", nccl_env_lines(&env(&[("NCCL_IB_TC", "105")]), None, None));
        let f = std::thread::spawn(move || handshake(&mut b, Role::Follower, &g, Duration::from_secs(5)));
        let mut lg = gate(0, 2);
        lg.build = build_fingerprint_of("1.0.0-dev", "k", Some("e45ae8bce7cb"), "beef", 250_000);
        lg.nccl = format!("channels=4,proto=default,lib=23007\n{}", nccl_env_lines(&env(&[("NCCL_IB_TC", "106")]), None, None));
        let e = admit_links(vec![a], &lg, &quick()).unwrap_err().to_string();
        for want in [
            "build commit",
            "rank 0 has e45ae8bce7cb, rank 1 has other",
            "build binary sha256",
            "NCCL environment variable NCCL_IB_TC",
        ] {
            assert!(e.contains(want), "{want}: {e}");
        }
        assert!(f.join().unwrap().is_err());
    }

    // ------------------------------------------------------------------------------------------------- the fabric's own failure

    #[test]
    fn an_async_error_of_the_leaders_communicator_loses_the_group_while_the_pings_still_flow() {
        let mut g = start(2, Duration::from_secs(30), |_, _| {});
        let bad = Arc::new(AtomicBool::new(false));
        let b = bad.clone();
        g.leader().on_comm_health(Arc::new(move || {
            b.load(Ordering::SeqCst)
                .then(|| "NCCL reports an asynchronous error on rank 0's communicator: ncclRemoteError".to_string())
        }));
        let aborted = Arc::new(AtomicBool::new(false));
        let ab = aborted.clone();
        g.leader().on_group_lost(Box::new(move || ab.store(true, Ordering::SeqCst)));
        g.leader().begin_slot(0).unwrap();
        // healthy: polled many times (50 ms), nothing happens
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(g.leader().failure(), None);
        bad.store(true, Ordering::SeqCst);
        let t0 = Instant::now();
        while g.leader().failure().is_none() && t0.elapsed() < Duration::from_secs(3) {
            std::thread::sleep(Duration::from_millis(10));
        }
        let why = g.leader().failure().expect("the poll must lose the group");
        assert!(why.contains("ncclRemoteError"), "{why}");
        assert!(t0.elapsed() < Duration::from_secs(1), "{:?}", t0.elapsed());
        // the abort hook ran, every later call fails fatally, and the follower is let go (its link was closed without a goodbye)
        let t1 = Instant::now();
        while !aborted.load(Ordering::SeqCst) && t1.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(aborted.load(Ordering::SeqCst));
        assert!(matches!(g.leader().flush(), Err(e) if e.is_fatal()));
        let r = g.finished(1, Duration::from_secs(5)).expect("the follower did not end");
        assert!(matches!(r, Ok(FollowerExit::LeaderLost(_))), "{r:?}");
    }

    #[test]
    fn an_async_error_of_a_followers_communicator_ends_the_follower_as_a_lost_group_and_the_leader_sees_it() {
        let (a, b) = Link::<ToyCall>::pair(Duration::from_secs(2)).unwrap();
        let bad = Arc::new(AtomicBool::new(false));
        let aborted = Arc::new(AtomicBool::new(false));
        let mut cfg = FollowerConfig::new(TpTimeouts { peer: Duration::from_secs(30), ..quick() }).with_rank(1);
        let bb = bad.clone();
        cfg.health = Some(Arc::new(move || bb.load(Ordering::SeqCst).then(|| "fabric down (ncclSystemError)".to_string())));
        let ab = aborted.clone();
        cfg.abort_hook = Some(Box::new(move || ab.store(true, Ordering::SeqCst)));
        let f = std::thread::spawn(move || run_follower(Toy::new(Log::default()), b, cfg));
        let mut leader = TpLeader::lead_group(Toy::new(Log::default()), vec![a], quick());
        leader.begin_slot(0).unwrap();
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(leader.failure(), None);
        bad.store(true, Ordering::SeqCst);
        let t0 = Instant::now();
        while !f.is_finished() && t0.elapsed() < Duration::from_secs(3) {
            std::thread::sleep(Duration::from_millis(5));
        }
        let r = f.join().unwrap();
        assert!(matches!(&r, Ok(FollowerExit::LeaderLost(w)) if w.contains("ncclSystemError")), "{r:?}");
        assert!(aborted.load(Ordering::SeqCst));
        // the leader sees the follower's link end at once (no 15 s silence)
        let t1 = Instant::now();
        while leader.failure().is_none() && t1.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(leader.failure().is_some_and(|w| w.contains("rank 1")), "{:?}", leader.failure());
    }

    #[test]
    fn a_toy_frame_round_trips_and_a_kind_nobody_claims_is_refused() {
        let msgs: Vec<(u64, Msg<ToyCall>)> = vec![
            (3, Msg::Call(ToyCall::Poke { slot: 2, by: 77 })),
            (4, Msg::Call(ToyCall::Note("hello".into()))),
            (0, Msg::Hello(gate(1, 2))),
            (5, Msg::Flush),
        ];
        for (seq, m) in &msgs {
            let frame = m.encode(*seq);
            let (s, back) = Msg::<ToyCall>::decode(&frame[4..]).unwrap();
            assert_eq!((s, &back), (*seq, m));
            assert_eq!(m.is_command(), !matches!(m, Msg::Hello(_)));
        }
        // the family's kinds sit outside the core set
        assert!(![K_POKE, K_NOTE].iter().any(|k| CORE_KINDS.contains(k)));
        // a kind the family does not claim, and a kind of another family's, are unknown; a trailing byte is refused
        let mut f = Msg::<ToyCall>::Call(ToyCall::Note("x".into())).encode(1);
        f[5] = 99;
        assert!(matches!(Msg::<ToyCall>::decode(&f[4..]), Err(WireError::UnknownKind(99))));
        let mut f = Msg::<ToyCall>::Call(ToyCall::Poke { slot: 1, by: 1 }).encode(1);
        f.push(0);
        assert!(matches!(Msg::<ToyCall>::decode(&f[4..]), Err(WireError::Invalid(_))));
        // the same bytes at another call type that claims nothing are not a command at all
        let f = Msg::<ToyCall>::Call(ToyCall::Note("x".into())).encode(1);
        assert!(matches!(Msg::<NoCall>::decode(&f[4..]), Err(WireError::UnknownKind(K_NOTE))));
    }

    /// A family with only the core calls: `NoCall`, no facts. The seam asks nothing more of it.
    struct CoreOnly {
        calls: Arc<Mutex<u64>>,
    }

    impl LockstepRuntime for CoreOnly {
        type Call = NoCall;
        fn vocab_size(&self) -> usize {
            32
        }
        fn begin_slot(&mut self, _slot: usize) -> Result<(), ModelError> {
            *self.calls.lock().unwrap() += 1;
            Ok(())
        }
        fn end_slot(&mut self, _slot: usize) {}
        fn prefill_chunk(&mut self, _b: &RowBatch, _want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
            Ok(None)
        }
        fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
            Ok(Candidates { rows: b.rows, count, ids: vec![1; b.rows * count], vals: vec![0.5; b.rows * count] })
        }
        fn commit(&mut self, _slot: usize, _keep: usize) {}
        fn flush(&mut self) -> Result<(), ModelError> {
            Ok(())
        }
        fn apply_call(&mut self, _seq: u64, call: NoCall) -> Result<CallReply, ModelError> {
            match call {}
        }
    }

    #[test]
    fn a_family_with_only_the_core_calls_runs_with_no_call_type_and_no_facts() {
        let (a, b) = Link::<NoCall>::pair(Duration::from_secs(2)).unwrap();
        let calls = Arc::new(Mutex::new(0u64));
        let c = calls.clone();
        let t = std::thread::spawn(move || run_follower(CoreOnly { calls: c }, b, FollowerConfig::new(quick())));
        let mut leader = TpLeader::lead(CoreOnly { calls: calls.clone() }, a, quick());
        leader.begin_slot(0).unwrap();
        let w = leader.window_candidates(&batch(&[(0, 0, vec![3])]), 2).unwrap();
        assert_eq!(w.ids, vec![1, 1]);
        leader.barrier().unwrap();
        assert_eq!(leader.failure(), None);
        drop(leader);
        assert_eq!(t.join().unwrap().unwrap(), FollowerExit::Shutdown("leader shut down".into()));
        assert_eq!(*calls.lock().unwrap(), 2, "the leader and the follower each began the slot");
    }

    // ------------------------------------------------------------------------------------------------- cache calls (collected answers)

    fn collect_len(l: &TpLeader<Toy>) -> usize {
        l.shared.ledger.lock().unwrap().collect.len()
    }

    /// Every follower's answer comes back to the caller (rank-own digests, a recoverable error) and no difference loses the group: a
    /// cache call holds no collective, so the caller decides (a restore not verified on every rank falls back cold). Waiting for the
    /// answers is what makes them known before the next call (mutant: no wait -> every answer is Silent).
    #[test]
    fn a_cache_call_collects_each_ranks_answer_and_never_loses_the_group_over_a_difference() {
        let mut g = start(3, Duration::from_secs(30), |rank, t| {
            t.cache_digest = 100 + rank as u64;
            if rank == 2 {
                t.cache_error = Some(false);
            }
        });
        let out = cache(g.leader(), 0, true).unwrap();
        assert_eq!(out.local.as_ref().ok(), Some(&5), "rank 0's own result");
        assert_eq!(out.followers[0], RankAnswer::Ok(101), "rank 1's own digest, not compared with rank 0's");
        assert!(
            matches!(&out.followers[1], RankAnswer::Err(e) if e.contains("request") && e.contains("no such snapshot")),
            "{out:?}"
        );
        assert_eq!(collect_len(g.leader()), 0, "an answered call leaves nothing behind");
        // the group is whole and the ordinary calls are still compared
        assert_eq!(poke(g.leader(), 0, 3).unwrap(), 3);
        g.leader().barrier().unwrap();
        assert_eq!(g.leader().failure(), None);
        assert_eq!(g.leader().group_status().per_rank.iter().map(|r| r.diverged).sum::<u64>(), 0);
        for e in g.shutdown() {
            assert!(matches!(e, Ok(FollowerExit::Shutdown(_))), "{e:?}");
        }
    }

    /// A fatal class is not a cache miss: a sticky device error on a follower still loses the group, naming the rank and the call.
    #[test]
    fn a_fatal_answer_to_a_cache_call_still_loses_the_group() {
        let mut g = start(2, Duration::from_secs(30), |_, t| t.cache_error = Some(true));
        let e = cache(g.leader(), 0, true).unwrap_err().to_string();
        assert!(e.contains("rank 1") && e.contains("cache") && e.contains("fatal"), "{e}");
        assert!(g.leader().failure().is_some());
    }

    /// A follower slower than the wait: its answer is `Silent`, the call fails for the caller, the late answer is consumed silently and
    /// the group stays up (a slow follower costs a cold prefill, never the group).
    #[test]
    fn a_late_cache_answer_is_silent_then_consumed_and_the_group_stays_up() {
        let mut g = start(2, Duration::from_secs(30), |_, t| t.cache_sleep = Duration::from_millis(600));
        g.leader().set_cache_deadlines(CacheDeadlines { ack: Duration::from_millis(100), stall: Duration::from_secs(20) });
        let out = cache(g.leader(), 1, true).unwrap();
        assert_eq!(out.followers, vec![RankAnswer::Silent]);
        assert_eq!(collect_len(g.leader()), 1, "the call is abandoned, still expecting the late answer");
        // the next calls go on; the barrier waits behind the late answer, which is consumed without a comparison
        assert_eq!(poke(g.leader(), 1, 2).unwrap(), 2);
        g.leader().barrier().unwrap();
        assert_eq!(g.leader().failure(), None);
        assert_eq!(collect_len(g.leader()), 0, "the late answer was consumed");
        // a call that is not waited for (a release) is the same: every answer Silent, consumed later, nothing lost
        let out = cache(g.leader(), 2, false).unwrap();
        assert_eq!(out.followers, vec![RankAnswer::Silent]);
        g.leader().barrier().unwrap();
        assert_eq!((g.leader().failure(), collect_len(g.leader())), (None, 0));
        for e in g.shutdown() {
            assert!(matches!(e, Ok(FollowerExit::Shutdown(_))), "{e:?}");
        }
    }

    /// A follower that answers pings but never the cache call (stuck in a copy) is lost at the stall deadline, by name: without it a
    /// cold prefill would wait in a collective with it for ever (mutant: no stall check -> the group is never lost).
    #[test]
    fn a_follower_that_never_answers_a_cache_call_is_lost_at_the_stall_deadline() {
        let mut g = start(2, Duration::from_secs(30), |_, t| t.cache_sleep = Duration::from_secs(120));
        g.leader().set_cache_deadlines(CacheDeadlines { ack: Duration::from_millis(100), stall: Duration::from_millis(400) });
        let out = cache(g.leader(), 0, true).unwrap();
        assert_eq!(out.followers, vec![RankAnswer::Silent], "the call itself fails over to the caller first");
        let t0 = Instant::now();
        while g.leader().failure().is_none() && t0.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(20));
        }
        let why = g.leader().failure().expect("the stalled follower loses the group");
        assert!(why.contains("rank 1") && why.contains("has not answered cache call") && why.contains("(cache)"), "{why}");
        assert!(g.aborted[0].load(Ordering::SeqCst) || g.finished(1, Duration::from_secs(10)).is_some());
    }

    #[test]
    fn the_snapshot_book_keeps_an_entry_only_when_every_rank_took_it_and_verifies_each_rank_against_its_own_digest() {
        let mut b = SnapshotBook::default();
        // every rank took it, with digests of its own shard
        b.settle_snapshot(7, taken_digest(10), &[RankAnswer::Ok(taken_digest(20)), RankAnswer::Ok(taken_digest(30))]).unwrap();
        assert!(b.holds(7) && b.len() == 1);
        // one rank skipped (memory guard), one errored, one was silent, rank 0 skipped: never recorded, each named
        for (r0, f, says) in [
            (taken_digest(1), vec![RankAnswer::Ok(SNAPSHOT_SKIPPED)], "rank 1 took no snapshot"),
            (taken_digest(1), vec![RankAnswer::Err("request error".into())], "rank 1: error"),
            (taken_digest(1), vec![RankAnswer::Silent], "rank 1: no answer"),
            (SNAPSHOT_SKIPPED, vec![RankAnswer::Ok(taken_digest(2))], "rank 0 took no snapshot"),
        ] {
            let e = b.settle_snapshot(8, r0, &f).unwrap_err();
            assert!(e.contains(says), "{e}");
            assert!(!b.holds(8));
        }
        // a restore is verified rank by rank against that rank's OWN digest (ranks hold different shards)
        let ok = [RankAnswer::Ok(taken_digest(20)), RankAnswer::Ok(taken_digest(30))];
        b.verify_restore(7, Ok(taken_digest(10)), &ok).unwrap();
        // a stale or empty restore on rank 2 that returned Ok: caught by its digest
        let e = b.verify_restore(7, Ok(taken_digest(10)), &[ok[0].clone(), RankAnswer::Ok(taken_digest(32))]).unwrap_err();
        assert!(e.contains("rank 2 restored digest") && !e.contains("rank 1"), "{e}");
        // rank 0's own restore is verified too
        assert!(b.verify_restore(7, Ok(taken_digest(12)), &ok).unwrap_err().contains("rank 0 restored digest"));
        assert!(b.verify_restore(7, Err("no such snapshot".into()), &ok).unwrap_err().contains("rank 0: error"));
        // a digest equal to ANOTHER rank's snapshot is still wrong for this rank
        assert!(b.verify_restore(7, Ok(taken_digest(10)), &[ok[1].clone(), ok[1].clone()]).is_err());
        assert!(b
            .verify_restore(7, Ok(taken_digest(10)), &[RankAnswer::Silent, ok[1].clone()])
            .unwrap_err()
            .contains("no answer"));
        assert!(b.verify_restore(7, Ok(taken_digest(10)), &ok[..1]).is_err(), "a rank short");
        assert!(b.verify_restore(9, Ok(taken_digest(10)), &ok).unwrap_err().contains("not held"));
        assert!(b.forget(7) && !b.forget(7) && b.is_empty());
        assert!(b.verify_restore(7, Ok(taken_digest(10)), &ok).is_err(), "a released snapshot is never restored");
        assert_eq!(taken_digest(0) & 1, 1, "a taken digest is never SNAPSHOT_SKIPPED");
    }
}
