//! The lockstep runtime of the tensor-parallel Nemotron engine (TP-R): [`crate::lockstep`] instantiated for this family. The protocol,
//! the leader, the follower loop, the boot and the failure model are the family-neutral module's (read its docs); what is here is what
//! only this family has:
//!
//! * [`NemoCall`]: the commands beyond the core calls (the image-feature prefill, the MTP head's draft and disable, the snapshot arena
//!   and the three snapshot calls), with their wire bodies and kinds exactly as protocol 2 shipped them;
//! * [`NemoFacts`]: the boot facts the ranks must agree on (the head priced, `REXL3_NEMO_GRAPHS`, `REXL3_MTP`, the restricted draft
//!   vocabulary), in the Hello where they have always been;
//! * [`NemoRt`]: the adapter that makes any [`NemotronRuntime`] a [`LockstepRuntime`], and the leader's `NemotronRuntime`
//!   implementation (`TpLeader<R>` is `lockstep::TpLeader<NemoRt<R>>`): the image bytes of a slice travel through a
//!   [`FeatureBridge`], and the snapshot handles' translation between ranks lives here (a handle's `snap` tag is a hash over pages the
//!   rank computed for itself, so two ranks need not agree on it);
//! * the names the rest of the family codes against (`Msg`, `BootGate`, `Link`, `FollowerConfig`, `run_follower`, `boot_leader`, ...),
//!   which are the neutral ones at this family's call type.
//!
//! Design and launch recipe: `docs/NEMOTRON_TP.md` (section 5, and sections 13 and 14 for the N-rank form); the family-neutral seam and
//! how another family adopts it: `docs/CLUSTER_FAMILY_PORTING.md`.

use super::backend::{ModelOptions, NemotronRuntime};
use super::mtp::{DraftAsk, Drafts};
use super::vision::{NemotronImageFeatures, SpliceOp};
use crate::error::ModelError;
use crate::forward::Candidates;
use crate::lockstep;
use crate::rows::RowBatch;
use gb_engine::PrefixHandle;
use gb_sampling::Sampling;
use std::collections::BTreeMap;
use std::io::Read;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

// everything family-neutral, under the names this module has always had
pub use crate::lockstep::*;

// ------------------------------------------------------------------------------------------------------------------- the family's calls

/// One image slice's rows of features: batch rows `dst_row..dst_row + n` are overwritten by `data`, the raw device bytes of the encoder
/// rows `src_row..src_row + n` (bf16 `[n, 4096]`: `n * 8192` bytes). The follower rebuilds a features buffer of exactly these rows, so
/// the splice it applies is `src_row 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpliceWire {
    pub dst_row: u64,
    pub n: u64,
    pub data: Vec<u8>,
}

/// One draft ask: the slot, the drafts wanted and the sequence so far.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AskWire {
    pub slot: u32,
    pub depth: u32,
    pub tokens: Vec<u32>,
}

/// The kinds of this family's commands (protocol 2: the numbers the first version of the runtime gave them; the core calls sit between).
pub const K_PREFILL_MM: u8 = 19;
pub const K_DRAFT: u8 = 23;
pub const K_DISABLE: u8 = 24;
pub const K_ENABLE_SNAP: u8 = 25;
pub const K_SNAPSHOT: u8 = 26;
pub const K_RESTORE: u8 = 27;
pub const K_RELEASE: u8 = 28;

/// The commands of the Nemotron family beyond the core calls. Numbered, sent, applied and acknowledged exactly like a core call.
#[derive(Clone, Debug, PartialEq)]
pub enum NemoCall {
    PrefillMm {
        batch: RowBatch,
        want: Option<u32>,
        splices: Vec<SpliceWire>,
    },
    MtpDraft {
        asks: Vec<AskWire>,
        keyed: Vec<Option<Sampling>>,
    },
    MtpDisable(u32),
    EnableSnapshots(u32),
    SnapshotSlot {
        slot: u32,
        len: u64,
    },
    /// `of` is the sequence number of the `SnapshotSlot` command that made the handle.
    RestoreSnapshot {
        slot: u32,
        len: u64,
        of: u64,
    },
    ReleaseSnapshot {
        of: u64,
    },
}

/// What the Nemotron ranks must agree on, beyond the shared geometry: the facts that shape the call stream or the arithmetic and are
/// read from the environment (`docs/NEMOTRON_TP.md` 5.4.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NemoFacts {
    /// The MTP head is loaded and priced.
    pub mtp: bool,
    /// `REXL3_NEMO_GRAPHS` is not off.
    pub graphs: bool,
    /// `REXL3_MTP` is not off.
    pub mtp_env: bool,
    /// The restricted draft vocabulary: `off`, or the id count and a hash of the ids (the files may sit at different paths).
    pub vocab: String,
    /// The expert ownership table (`REXL3_TP_EXPERT_MAP`, `tp_balance::expert_map_fingerprint`): 0 = contiguous ownership, else the
    /// table's 64-bit content hash. Two ranks holding different tables would each compute bits for pairs the other does not, so the
    /// gate compares the hash itself, not only the flag bit of the rendezvous (found by the balance work, 2026-10-11).
    pub expert_map: u64,
    /// The collective policy (`REXL3_TP_COLLECTIVE`, `tp_comm::collective_fingerprint`): a ring on one rank and a one-shot exchange on
    /// another issue different NCCL calls, which is a deadlock, not an error.
    pub collective: String,
    /// `tp_arm::plan_flags()`: which matrices are sliced or replicated (attention, in_proj, the draft head split, an ownership table).
    /// It also rides in the CommInit byte (belt and braces); in the Hello a refusal happens BEFORE any rank touches NCCL, so a rank
    /// whose sibling refuses the flags never joins the communicator and waits for the boot guard.
    pub plan_flags: u8,
    /// What the prefix cache was asked for (`--prefix-cache`, `--prefix-cache-mib`): `off`, `auto` or `mib=N`. Protocol 4. Two ranks that
    /// were given different flags plan different arenas, and rank 0's `EnableSnapshots(slots)` would be built on rank 1 from a plan
    /// that did not agree to it: refused at the Hello, before any weight is read.
    pub cache_request: String,
    /// Snapshot slots the plan carved (`ServePlan::plan.cache.snapshot_slots`; 0: no cache). Rank 0's count is what the group serves
    /// with (it forwards `EnableSnapshots` of it); a follower must be able to hold at least that many, and is then built with exactly
    /// rank 0's. Protocol 4.
    pub cache_slots: u64,
    /// Bytes of one snapshot slot (`CacheBudget::snapshot_bytes`: every Mamba layer's S and conv, and the head's carry): a function of
    /// the model and `--spec`, so it must be equal. Protocol 4.
    pub cache_slot_bytes: u64,
    /// KV pages the cache may hold on top of admission's (`CacheBudget::page_blocks`); a follower must hold at least rank 0's. Protocol 4.
    pub cache_pages: u64,
    /// The prefill settings (lane S, `contract::prefill_fingerprint`: the contract, the boot's effective prefill rows (`--prefill-chunk` at
    /// most the contract's cap; under P1 the cap is `REXL3_NEMO_PREFILL_ROWS_MAX`), the absent-expert fast path
    /// `REXL3_NEMO_PREFILL_ABSENT_FAST`). The rows and the fast path change no bit, but a rank started with an opt-out is not the
    /// configuration the group was qualified in, and the contract changes the arithmetic: refused by name. Last on the wire (both ranks run
    /// the same build, which the gate's `build` already pins).
    pub prefill: String,
}

impl NemoFacts {
    /// These facts with the prefix cache the plan carved: `request` is [`cache_request_label`] of the option the rank was started with.
    pub fn with_cache(mut self, request: String, budget: &gb_sched::prefix::CacheBudget) -> NemoFacts {
        self.cache_request = request;
        self.cache_slots = budget.snapshot_slots as u64;
        self.cache_slot_bytes = budget.snapshot_bytes;
        self.cache_pages = budget.page_blocks as u64;
        self
    }
}

/// The text the boot gate compares for what a rank was asked to do with the prefix cache: `off` (`--prefix-cache off`,
/// `--prefix-cache-mib 0`, or a family that has none), `auto` (the default), `on` (`--prefix-cache on`) or `mib=N`.
pub fn cache_request_label(asked: crate::serve::PrefixCacheOption) -> String {
    use crate::serve::PrefixCacheOption as P;
    match asked {
        P::Off | P::Mib(0) => "off".into(),
        P::Auto => "auto".into(),
        P::On => "on".into(),
        P::Mib(n) => format!("mib={n}"),
    }
}

fn enc_sampling(e: &mut Enc, s: &Option<Sampling>) {
    match s {
        Some(s) => {
            e.bool(true);
            e.u64(s.seed);
            e.f64(s.temperature);
            e.u64(s.top_k as u64);
            e.f64(s.top_p);
            e.f64(s.min_p);
        }
        None => e.bool(false),
    }
}

fn dec_sampling(d: &mut Dec<'_>) -> Result<Option<Sampling>, WireError> {
    if !d.bool("sampler tag")? {
        return Ok(None);
    }
    Ok(Some(Sampling {
        seed: d.u64("sampler seed")?,
        temperature: d.f64("sampler temperature")?,
        top_k: usize::try_from(d.u64("sampler top_k")?).map_err(|_| WireError::Invalid("top_k overflow".into()))?,
        top_p: d.f64("sampler top_p")?,
        min_p: d.f64("sampler min_p")?,
    }))
}

impl FamilyCall for NemoCall {
    type Facts = NemoFacts;

    fn kind(&self) -> u8 {
        match self {
            NemoCall::PrefillMm { .. } => K_PREFILL_MM,
            NemoCall::MtpDraft { .. } => K_DRAFT,
            NemoCall::MtpDisable(_) => K_DISABLE,
            NemoCall::EnableSnapshots(_) => K_ENABLE_SNAP,
            NemoCall::SnapshotSlot { .. } => K_SNAPSHOT,
            NemoCall::RestoreSnapshot { .. } => K_RESTORE,
            NemoCall::ReleaseSnapshot { .. } => K_RELEASE,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            NemoCall::PrefillMm { .. } => "prefill_chunk_mm",
            NemoCall::MtpDraft { .. } => "mtp_draft",
            NemoCall::MtpDisable(_) => "mtp_disable_slot",
            NemoCall::EnableSnapshots(_) => "enable_snapshots",
            NemoCall::SnapshotSlot { .. } => "snapshot_slot",
            NemoCall::RestoreSnapshot { .. } => "restore_snapshot",
            NemoCall::ReleaseSnapshot { .. } => "release_snapshot",
        }
    }

    fn encode(&self, e: &mut Enc) {
        match self {
            NemoCall::PrefillMm { batch, want, splices } => {
                enc_batch(e, batch);
                e.opt_u32(*want);
                e.u64(splices.len() as u64);
                for s in splices {
                    e.u64(s.dst_row);
                    e.u64(s.n);
                    e.bytes(&s.data);
                }
            }
            NemoCall::MtpDraft { asks, keyed } => {
                e.u64(asks.len() as u64);
                for a in asks {
                    e.u32(a.slot);
                    e.u32(a.depth);
                    e.u32s(&a.tokens);
                }
                e.u64(keyed.len() as u64);
                for k in keyed {
                    enc_sampling(e, k);
                }
            }
            NemoCall::MtpDisable(s) | NemoCall::EnableSnapshots(s) => e.u32(*s),
            NemoCall::SnapshotSlot { slot, len } => {
                e.u32(*slot);
                e.u64(*len);
            }
            NemoCall::RestoreSnapshot { slot, len, of } => {
                e.u32(*slot);
                e.u64(*len);
                e.u64(*of);
            }
            NemoCall::ReleaseSnapshot { of } => e.u64(*of),
        }
    }

    fn decode(kind: u8, d: &mut Dec<'_>) -> Result<Option<NemoCall>, WireError> {
        Ok(Some(match kind {
            K_PREFILL_MM => {
                let batch = dec_batch(d)?;
                let want = d.opt_u32("want")?;
                let n = d.len(16, "splices")?;
                let mut splices = Vec::with_capacity(n);
                for _ in 0..n {
                    splices.push(SpliceWire { dst_row: d.u64("dst_row")?, n: d.u64("n")?, data: d.bytes("splice data")? });
                }
                NemoCall::PrefillMm { batch, want, splices }
            }
            K_DRAFT => {
                let n = d.len(16, "asks")?;
                let mut asks = Vec::with_capacity(n);
                for _ in 0..n {
                    asks.push(AskWire { slot: d.u32("ask slot")?, depth: d.u32("ask depth")?, tokens: d.u32s("ask tokens")? });
                }
                let k = d.len(1, "keyed")?;
                let mut keyed = Vec::with_capacity(k);
                for _ in 0..k {
                    keyed.push(dec_sampling(d)?);
                }
                NemoCall::MtpDraft { asks, keyed }
            }
            K_DISABLE => NemoCall::MtpDisable(d.u32("slot")?),
            K_ENABLE_SNAP => NemoCall::EnableSnapshots(d.u32("snapshot slots")?),
            K_SNAPSHOT => NemoCall::SnapshotSlot { slot: d.u32("slot")?, len: d.u64("len")? },
            K_RESTORE => NemoCall::RestoreSnapshot { slot: d.u32("slot")?, len: d.u64("len")?, of: d.u64("of")? },
            K_RELEASE => NemoCall::ReleaseSnapshot { of: d.u64("of")? },
            _ => return Ok(None),
        }))
    }

    fn counts_as_prefill(&self) -> bool {
        matches!(self, NemoCall::PrefillMm { .. })
    }

    fn encode_facts(f: &NemoFacts, e: &mut Enc) {
        e.bool(f.mtp);
        e.bool(f.graphs);
        e.bool(f.mtp_env);
        e.str(&f.vocab);
        e.u64(f.expert_map);
        e.str(&f.collective);
        e.u8(f.plan_flags);
        e.str(&f.cache_request);
        e.u64(f.cache_slots);
        e.u64(f.cache_slot_bytes);
        e.u64(f.cache_pages);
        e.str(&f.prefill);
    }

    fn decode_facts(d: &mut Dec<'_>) -> Result<NemoFacts, WireError> {
        Ok(NemoFacts {
            mtp: d.bool("mtp")?,
            graphs: d.bool("graphs")?,
            mtp_env: d.bool("mtp env")?,
            vocab: d.str("vocab")?,
            expert_map: d.u64("expert map")?,
            collective: d.str("collective")?,
            plan_flags: d.u8("plan flags")?,
            cache_request: d.str("cache request")?,
            cache_slots: d.u64("cache slots")?,
            cache_slot_bytes: d.u64("cache slot bytes")?,
            cache_pages: d.u64("cache pages")?,
            prefill: d.str("prefill settings")?,
        })
    }

    fn check_facts(leader: &NemoFacts, follower: &NemoFacts, who: u32) -> Vec<String> {
        // the prefix cache's budget (docs/NEMOTRON_TP.md 12.5): rank 0 carves the arena and forwards `EnableSnapshots(slots)`, rank 1
        // builds an arena of exactly that size and a pool of exactly rank 0's pages (it adopts rank 0's options), so what must hold is
        // that rank 1 was asked for the same thing, that a slot is the same bytes, and that its own plan has room for rank 0's slots
        // and pages. Each is named, and none is the generic "KV pages" refusal that the cache's pages would otherwise show up as.
        let mut cache = Vec::new();
        cache.extend(fact_differs(
            "prefix cache request (--prefix-cache / --prefix-cache-mib; pass the same to every rank)",
            &leader.cache_request,
            &follower.cache_request,
            who,
        ));
        cache.extend(fact_differs(
            "prefix cache bytes of one snapshot slot",
            leader.cache_slot_bytes,
            follower.cache_slot_bytes,
            who,
        ));
        if follower.cache_slots < leader.cache_slots {
            cache.push(format!(
                "prefix cache snapshot slots: rank 0 planned {}, rank {who} can hold only {} (pass the same --prefix-cache-mib to all \
                 ranks, or free memory on rank {who})",
                leader.cache_slots, follower.cache_slots
            ));
        }
        if follower.cache_pages < leader.cache_pages {
            cache.push(format!(
                "prefix cache KV pages: rank 0 planned {}, rank {who} can hold only {} (pass the same --prefix-cache-mib to all ranks, \
                 or free memory on rank {who})",
                leader.cache_pages, follower.cache_pages
            ));
        }
        [
            fact_differs("MTP head priced (--spec)", leader.mtp, follower.mtp, who),
            fact_differs("window graphs (REXL3_NEMO_GRAPHS)", leader.graphs, follower.graphs, who),
            fact_differs("MTP head wanted (REXL3_MTP)", leader.mtp_env, follower.mtp_env, who),
            fact_differs("restricted draft vocabulary (REXL3_MTP_VOCAB)", &leader.vocab, &follower.vocab, who),
            fact_differs(
                "expert ownership table (REXL3_TP_EXPERT_MAP; 0 = contiguous)",
                format!("{:016x}", leader.expert_map),
                format!("{:016x}", follower.expert_map),
                who,
            ),
            fact_differs("collective policy (REXL3_TP_COLLECTIVE)", &leader.collective, &follower.collective, who),
            fact_differs(
                "prefill settings (REXL3_NEMO_PREFILL_CONTRACT / --prefill-chunk and REXL3_NEMO_PREFILL_ROWS_MAX / REXL3_NEMO_PREFILL_ABSENT_FAST; under P2 the tp= levers REXL3_NEMO_P2TP_*)",
                &leader.prefill,
                &follower.prefill,
                who,
            ),
            fact_differs(
                "plan flags (REXL3_TP_ATTN / REXL3_TP_INPROJ / REXL3_TP_HEAD / expert map)",
                format!("{:#010b}", leader.plan_flags),
                format!("{:#010b}", follower.plan_flags),
                who,
            ),
        ]
        .into_iter()
        .flatten()
        .chain(cache)
        .collect()
    }
}

// ------------------------------------------------------------------------------------------------------------------- the names

/// A message of the protocol at this family's call type.
pub type Msg = lockstep::Msg<NemoCall>;
/// The boot gate at this family's facts.
pub type BootGate = lockstep::BootGate<NemoCall>;
/// A connection between two ranks of a Nemotron group.
pub type Link = lockstep::Link<NemoCall>;
/// The collective's bootstrap over the links of a Nemotron group.
pub type Rendezvous<'a> = lockstep::Rendezvous<'a, NemoCall>;
/// The leader's runtime: rank 0's model with every state-changing call forwarded. [`NemoRt`] is the adapter that makes `R` lockstep.
pub type TpLeader<R> = lockstep::TpLeader<NemoRt<R>>;

/// Reads one frame (blocking). A clean close at a frame boundary is [`WireError::Closed`].
pub fn read_frame(r: &mut dyn Read) -> Result<(u64, Msg), WireError> {
    lockstep::read_frame(r)
}

impl BootGate {
    /// This process's side of the gate, for the pack at `pack` identity and the options the model will be built with. The rank is set
    /// by [`BootGate::at`] (a single process is rank 0 of 1).
    pub fn local(pack: String, o: &ModelOptions) -> BootGate {
        lockstep::BootGate::new(
            pack,
            o.layers.map_or(0, |n| n as u32),
            o.slots as u32,
            o.kv_pages as u64,
            o.max_context as u64,
            o.prefill_rows as u32,
            NemoFacts {
                mtp: o.mtp,
                graphs: !matches!(std::env::var(super::model::GRAPHS_ENV).as_deref(), Ok("0" | "off" | "false" | "no")),
                mtp_env: super::mtp::head_wanted(None),
                vocab: vocab_fingerprint(),
                expert_map: super::tp_balance::expert_map_fingerprint(),
                collective: super::tp_comm::collective_fingerprint(),
                plan_flags: super::tp_arm::plan_flags(),
                // no prefix cache until `with_cache` says what the plan carved
                cache_request: "off".into(),
                cache_slots: 0,
                cache_slot_bytes: 0,
                cache_pages: 0,
                prefill: super::contract::prefill_fingerprint(o.prefill_rows),
            },
        )
    }

    /// This gate with the prefix cache the plan carved (`request`: [`cache_request_label`] of what this rank was started with).
    pub fn with_cache(mut self, request: String, budget: &gb_sched::prefix::CacheBudget) -> BootGate {
        self.family = self.family.with_cache(request, budget);
        self
    }

    /// The model options this gate describes (what the follower loads with once the gate has passed).
    pub fn model_options(&self) -> ModelOptions {
        ModelOptions {
            slots: self.slots as usize,
            kv_pages: self.kv_pages as usize,
            max_context: self.max_context as usize,
            prefill_rows: self.prefill_rows as usize,
            layers: (self.layers != 0).then_some(self.layers as usize),
            mtp: self.family.mtp,
        }
    }
}

/// The restricted draft vocabulary as the model builds it, as a short fingerprint.
fn vocab_fingerprint() -> String {
    match super::mtp::restricted_vocab_ids_for(&super::mtp::VocabSetting::from_env()) {
        Ok(None) => "off".into(),
        Ok(Some(ids)) => {
            let h = ids.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, id| {
                id.to_le_bytes().iter().fold(h, |h, b| (h ^ *b as u64).wrapping_mul(0x100_0000_01b3))
            });
            format!("n={},h={h:016x}", ids.len())
        }
        Err(e) => format!("unreadable: {e}"),
    }
}

// ------------------------------------------------------------------------------------------------------------------- features bridge

/// How image-feature rows travel between the ranks: the leader reads them off its device, the follower makes a buffer of its own. The
/// model's own types are device buffers, so this is the one place the protocol touches the device.
pub trait FeatureBridge: Send {
    /// The raw bytes of rows `src_row..src_row + n` of the splice's features.
    fn export(&self, op: &SpliceOp) -> Result<Vec<u8>, ModelError>;
    /// A splice of `n` rows at batch row `dst_row` over the received bytes.
    fn import(&self, dst_row: usize, n: usize, data: Vec<u8>) -> Result<SpliceOp, ModelError>;
}

/// The device bridge: `DevBuf` download on the leader, upload on the follower (the features of a prefill slice, bf16 `[n, 4096]`).
pub struct DeviceFeatures;

impl FeatureBridge for DeviceFeatures {
    fn export(&self, op: &SpliceOp) -> Result<Vec<u8>, ModelError> {
        let tokens = op.feats.tokens.max(1);
        let row = op.feats.buf.bytes() / tokens;
        if op.src_row + op.n > op.feats.tokens || row == 0 {
            return Err(ModelError::Request(format!(
                "an image splice of {} rows from row {} of {} cannot be exported",
                op.n, op.src_row, op.feats.tokens
            )));
        }
        let mut out = vec![0u8; op.n * row];
        op.feats.buf.download_range(op.src_row * row, &mut out)?;
        Ok(out)
    }

    fn import(&self, dst_row: usize, n: usize, data: Vec<u8>) -> Result<SpliceOp, ModelError> {
        if n == 0 || data.is_empty() || data.len() % n != 0 {
            return Err(ModelError::Request(format!("an image splice of {n} rows carries {} bytes", data.len())));
        }
        let buf = gb_cuda::DevBuf::from_slice(&data)?;
        // the model's stream is non-blocking: the copy must have landed before the slice that reads it is launched
        gb_cuda::rt::device_sync()?;
        let feats = NemotronImageFeatures { buf, tokens: n, merged: (n, 1), permit: None };
        Ok(SpliceOp { feats: Arc::new(feats), dst_row, src_row: 0, n })
    }
}

/// A bridge that refuses every splice by name (a runtime that cannot splice, the CPU tests).
pub struct NoFeatures;

impl FeatureBridge for NoFeatures {
    fn export(&self, _op: &SpliceOp) -> Result<Vec<u8>, ModelError> {
        Err(ModelError::NotBuilt("this tensor-parallel link carries no image features".into()))
    }
    fn import(&self, _dst_row: usize, _n: usize, _data: Vec<u8>) -> Result<SpliceOp, ModelError> {
        Err(ModelError::NotBuilt("this tensor-parallel link carries no image features".into()))
    }
}

// ------------------------------------------------------------------------------------------------------------------- the adapter

/// A [`NemotronRuntime`] as a [`LockstepRuntime`]: the core calls go to the model, the family's calls ([`NemoCall`]) are applied here.
/// Holds what only this family has of the protocol: the feature bridge and the snapshot handle translation.
pub struct NemoRt<R: NemotronRuntime> {
    inner: R,
    bridge: Box<dyn FeatureBridge>,
    /// A follower's own handle of each snapshot, by the sequence number of the `SnapshotSlot` command that made it.
    follower_snaps: BTreeMap<u64, PrefixHandle>,
    /// The leader's snapshot handles the followers also hold: `handle.snap` to (the sequence number of the call that made it, the
    /// handle's pages). A handle is forwarded only if it is in here exactly as it was made.
    leader_snaps: BTreeMap<u32, (u64, Vec<u32>)>,
}

impl<R: NemotronRuntime> NemoRt<R> {
    pub fn new(inner: R, bridge: Box<dyn FeatureBridge>) -> NemoRt<R> {
        NemoRt { inner, bridge, follower_snaps: BTreeMap::new(), leader_snaps: BTreeMap::new() }
    }

    /// The model.
    pub fn model(&self) -> &R {
        &self.inner
    }

    pub fn model_mut(&mut self) -> &mut R {
        &mut self.inner
    }

    pub fn into_model(self) -> R {
        self.inner
    }
}

impl<R: NemotronRuntime> LockstepRuntime for NemoRt<R> {
    type Call = NemoCall;

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn has_mtp(&self) -> bool {
        self.inner.has_mtp()
    }

    fn snapshots_supported() -> bool {
        R::snapshots_supported()
    }

    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.inner.begin_slot(slot)
    }

    fn end_slot(&mut self, slot: usize) {
        self.inner.end_slot(slot)
    }

    fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        self.inner.prefill_chunk(b, want)
    }

    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        self.inner.window_candidates(b, count)
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        self.inner.commit(slot, keep)
    }

    fn flush(&mut self) -> Result<(), ModelError> {
        self.inner.flush()
    }

    fn apply_call(&mut self, seq: u64, call: NemoCall) -> Result<CallReply, ModelError> {
        match call {
            NemoCall::PrefillMm { batch, want, splices } => {
                let mut ops = Vec::with_capacity(splices.len());
                for s in splices {
                    ops.push(self.bridge.import(s.dst_row as usize, s.n as usize, s.data)?);
                }
                let c = self.inner.prefill_chunk_mm(&batch, want.map(|w| w as usize), &ops)?;
                Ok(CallReply::digest(c.as_ref().map_or(0, candidates_digest)))
            }
            NemoCall::MtpDraft { asks, keyed } => {
                let list: Vec<DraftAsk<'_>> =
                    asks.iter().map(|a| DraftAsk { slot: a.slot as usize, tokens: &a.tokens, depth: a.depth as usize }).collect();
                self.inner.mtp_draft(&list, &keyed).map(|_| CallReply::NONE)
            }
            NemoCall::MtpDisable(s) => {
                self.inner.mtp_disable_slot(s as usize);
                Ok(CallReply::NONE)
            }
            // the arena's bytes are the ack's digest: an arena of another size on this rank is a named boot failure, not an assumption
            NemoCall::EnableSnapshots(n) => {
                self.inner.enable_snapshots(n as usize)?;
                Ok(CallReply::digest(self.inner.snapshot_arena_bytes()))
            }
            NemoCall::SnapshotSlot { slot, len } => match self.inner.snapshot_slot(slot as usize, len as usize)? {
                Some(h) => {
                    self.follower_snaps.insert(seq, h);
                    Ok(CallReply::digest(2))
                }
                None => Ok(CallReply::digest(1)),
            },
            NemoCall::RestoreSnapshot { slot, len, of } => {
                let h = self
                    .follower_snaps
                    .get(&of)
                    .ok_or_else(|| ModelError::Request(format!("restore of snapshot {of}: this rank holds no such snapshot")))?;
                self.inner.restore_snapshot(slot as usize, h, len as usize).map(|_| CallReply::NONE)
            }
            NemoCall::ReleaseSnapshot { of } => {
                // the handle leaves this table only when the model released it, as it leaves the leader's (`TpLeader::release_snapshot`):
                // a release both ranks refuse leaves both tables, and so both models, as they were
                let h =
                    self.follower_snaps.get(&of).cloned().ok_or_else(|| {
                        ModelError::Request(format!("release of snapshot {of}: this rank holds no such snapshot"))
                    })?;
                let r = self.inner.release_snapshot(h);
                if r.is_ok() {
                    self.follower_snaps.remove(&of);
                }
                r.map(|_| CallReply::NONE)
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------- the leader

impl<R: NemotronRuntime> lockstep::TpLeader<NemoRt<R>> {
    /// Starts the leader over `link`, the one follower of a two-rank group (the boot handshake is done: [`boot_leader`]), with the
    /// device bridge for image features.
    pub fn start(inner: R, link: Link, timeouts: TpTimeouts) -> TpLeader<R> {
        TpLeader::start_group(inner, vec![link], timeouts)
    }

    /// Starts the leader over the links of ranks `1..=links.len()` (`links[r - 1]` is rank `r`'s; the boot handshake is done:
    /// [`boot_leader_group`]).
    pub fn start_group(inner: R, links: Vec<Link>, timeouts: TpTimeouts) -> TpLeader<R> {
        lockstep::TpLeader::lead_group(NemoRt::new(inner, Box::new(DeviceFeatures)), links, timeouts)
    }

    /// Replaces the features bridge (the default moves device bytes).
    pub fn with_bridge(mut self, bridge: Box<dyn FeatureBridge>) -> Self {
        self.inner_mut().bridge = bridge;
        self
    }
}

impl<R: NemotronRuntime> NemotronRuntime for lockstep::TpLeader<NemoRt<R>> {
    fn vocab_size(&self) -> usize {
        self.inner().model().vocab_size()
    }

    fn has_mtp(&self) -> bool {
        self.inner().model().has_mtp()
    }

    fn head_form(&self) -> super::spec::HeadForm {
        self.inner().model().head_form()
    }

    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        lockstep::TpLeader::begin_slot(self, slot)
    }

    fn end_slot(&mut self, slot: usize) {
        lockstep::TpLeader::end_slot(self, slot)
    }

    fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        lockstep::TpLeader::prefill_chunk(self, b, want)
    }

    fn prefill_chunk_mm(
        &mut self,
        b: &RowBatch,
        want: Option<usize>,
        splice: &[SpliceOp],
    ) -> Result<Option<Candidates>, ModelError> {
        if splice.is_empty() {
            return lockstep::TpLeader::prefill_chunk(self, b, want);
        }
        self.guard()?;
        let mut splices = Vec::with_capacity(splice.len());
        for op in splice {
            splices.push(SpliceWire { dst_row: op.dst_row as u64, n: op.n as u64, data: self.inner().bridge.export(op)? });
        }
        let call = NemoCall::PrefillMm { batch: b.clone(), want: want.map(|w| w as u32), splices };
        self.forward_call(
            call,
            |m, _| m.inner.prefill_chunk_mm(b, want, splice),
            |c: &Option<Candidates>| c.as_ref().map_or(0, candidates_digest),
        )
    }

    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        lockstep::TpLeader::window_candidates(self, b, count)
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        lockstep::TpLeader::commit(self, slot, keep)
    }

    fn flush(&mut self) -> Result<(), ModelError> {
        lockstep::TpLeader::flush(self)
    }

    fn mtp_draft(&mut self, asks: &[DraftAsk<'_>], keyed: &[Option<Sampling>]) -> Result<Vec<Option<Drafts>>, ModelError> {
        let wire =
            asks.iter().map(|a| AskWire { slot: a.slot as u32, depth: a.depth as u32, tokens: a.tokens.to_vec() }).collect();
        let call = NemoCall::MtpDraft { asks: wire, keyed: keyed.to_vec() };
        // group-fatal like every other forwarded call: the split draft head holds collectives inside this call, so a failure on one
        // rank only leaves its peers inside an allreduce (a stall until the peer timeout), whatever the error's kind
        self.forward_call(call, |m, _| m.inner.mtp_draft(asks, keyed), |_| 0)
    }

    fn mtp_disable_slot(&mut self, slot: usize) {
        self.forward_unit(NemoCall::MtpDisable(slot as u32), |m| m.inner.mtp_disable_slot(slot));
    }

    fn mtp_pending_rows(&self, slot: usize) -> Option<usize> {
        self.inner().model().mtp_pending_rows(slot)
    }
    /// Rank 0's own graph cache: the followers' counters stay in their processes.
    fn graph_counters(&self) -> Option<gb_engine::GraphCounters> {
        self.inner().model().graph_counters()
    }

    fn snapshots_supported() -> bool {
        R::snapshots_supported()
    }

    fn enable_snapshots(&mut self, slots: usize) -> Result<(), ModelError> {
        // the digest is the arena's bytes on each rank (a follower whose arena is another size fails the boot, by name)
        self.forward_call(
            NemoCall::EnableSnapshots(slots as u32),
            |m, _| {
                m.inner.enable_snapshots(slots)?;
                Ok(m.inner.snapshot_arena_bytes())
            },
            |bytes| *bytes,
        )?;
        // a boot-time call: every follower's arena must exist before anything is served
        let seq = self.last_seq();
        self.settle(seq)
    }

    fn snapshot_arena_bytes(&self) -> u64 {
        self.inner().model().snapshot_arena_bytes()
    }

    fn free_snapshots(&self) -> usize {
        self.inner().model().free_snapshots()
    }

    fn snapshot_slot(&mut self, slot: usize, len: usize) -> Result<Option<PrefixHandle>, ModelError> {
        let call = NemoCall::SnapshotSlot { slot: slot as u32, len: len as u64 };
        self.forward_call(
            call,
            |m, seq| {
                let r = m.inner.snapshot_slot(slot, len);
                if let Ok(Some(h)) = &r {
                    m.leader_snaps.insert(h.snap, (seq, h.pages.clone()));
                }
                r
            },
            // "a snapshot was taken" is the digest: one rank with a free arena slot and another without is a divergence
            |o: &Option<PrefixHandle>| o.is_some() as u64 + 1,
        )
    }

    fn restore_snapshot(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), ModelError> {
        match self.inner().leader_snaps.get(&handle.snap) {
            Some((of, pages)) if *pages == handle.pages => {
                let of = *of;
                let call = NemoCall::RestoreSnapshot { slot: slot as u32, len: len as u64, of };
                self.forward_call(call, |m, _| m.inner.restore_snapshot(slot, handle, len), |_| 0)
            }
            _ => {
                // a handle the leader never made, or a stale one: the runtime refuses it by name and the followers never hear of it
                self.guard()?;
                let r = self.run_local(|m| m.inner.restore_snapshot(slot, handle, len));
                if r.is_ok() {
                    self.fail("rank 0 restored a snapshot handle it never forwarded to the followers".to_string());
                    return Err(lost_group("a snapshot restore the followers were not told of"));
                }
                r
            }
        }
    }

    fn release_snapshot(&mut self, handle: PrefixHandle) -> Result<(), ModelError> {
        match self.inner().leader_snaps.get(&handle.snap) {
            Some((of, pages)) if *pages == handle.pages => {
                let of = *of;
                let snap = handle.snap;
                let call = NemoCall::ReleaseSnapshot { of };
                self.forward_call(
                    call,
                    |m, _| {
                        let r = m.inner.release_snapshot(handle);
                        if r.is_ok() {
                            m.leader_snaps.remove(&snap);
                        }
                        r
                    },
                    |_| 0,
                )
            }
            _ => {
                self.guard()?;
                let r = self.run_local(|m| m.inner.release_snapshot(handle));
                if r.is_ok() {
                    self.fail("rank 0 released a snapshot handle it never forwarded to the followers".to_string());
                    return Err(lost_group("a snapshot release the followers were not told of"));
                }
                r
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------- the follower

/// The follower's knobs and the handles its caller keeps (the neutral [`lockstep::FollowerConfig`] and this family's feature bridge).
pub struct FollowerConfig {
    pub timeouts: TpTimeouts,
    pub bridge: Box<dyn FeatureBridge>,
    /// Set to ask the loop to return after the command in hand.
    pub stop: Arc<AtomicBool>,
    /// Told how many commands were applied, now and then (the stderr health line).
    pub on_progress: Option<ProgressFn>,
    /// Run when the leader is lost while a call is stuck (the communicator's abort).
    pub abort_hook: Option<AbortHook>,
    /// This process's rank in the group (1 for a two-rank group; any rank above 0 for a larger one): it names the rank in the backstop's
    /// and the stall watchdog's messages.
    pub rank: usize,
    /// The communicator's health probe (`ncclCommGetAsyncError`), polled by the follower's watchdog.
    pub health: Option<lockstep::HealthProbe>,
}

impl FollowerConfig {
    pub fn new(timeouts: TpTimeouts) -> Self {
        let g = lockstep::FollowerConfig::new(timeouts);
        FollowerConfig {
            timeouts: g.timeouts,
            bridge: Box::new(DeviceFeatures),
            stop: g.stop,
            on_progress: g.on_progress,
            abort_hook: g.abort_hook,
            rank: g.rank,
            health: g.health,
        }
    }

    /// The same config for follower `rank`.
    pub fn with_rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }
}

/// The follower's loop: applies the leader's commands, in order, to `rt` and acknowledges each. Returns on Shutdown, on EOF or when the
/// peer goes silent. A fatal local error is answered (so the leader fails by name) and returned.
pub fn run_follower<R: NemotronRuntime>(rt: R, link: Link, cfg: FollowerConfig) -> Result<FollowerExit, ModelError> {
    let FollowerConfig { timeouts, bridge, stop, on_progress, abort_hook, rank, health } = cfg;
    // `spin` is the generic config's own (`REXL3_TP_SPIN_US`): a latency knob of the receive path, not a fact of the family
    let mut generic = lockstep::FollowerConfig::new(timeouts).with_rank(rank);
    generic.stop = stop;
    generic.on_progress = on_progress;
    generic.abort_hook = abort_hook;
    generic.health = health;
    lockstep::run_follower(NemoRt::new(rt, bridge), link, generic)
}

// ------------------------------------------------------------------------------------------------------------------- boot

/// Rank 0's boot after the connection is up and gated, for ONE follower (a two-rank group): [`lockstep::boot_leader`] for this family.
/// `load` builds the rank's model from the communicator.
pub fn boot_leader<R, C, L>(
    link: Link,
    local: &BootGate,
    t: &TpTimeouts,
    make_comm: impl FnOnce(&mut Rendezvous<'_>) -> Result<C, ModelError>,
    load: L,
) -> Result<TpLeader<R>, ModelError>
where
    R: NemotronRuntime,
    L: FnOnce(C) -> Result<R, ModelError>,
{
    lockstep::boot_leader(link, local, t, make_comm, move |c| load(c).map(|m| NemoRt::new(m, Box::new(DeviceFeatures))))
}

/// Rank 0's boot after every follower is connected, identified and gated: [`lockstep::boot_leader_group`] for this family.
pub fn boot_leader_group<R, C, L>(
    links: Vec<Link>,
    local: &BootGate,
    t: &TpTimeouts,
    make_comm: impl FnOnce(&mut Rendezvous<'_>) -> Result<C, ModelError>,
    load: L,
) -> Result<TpLeader<R>, ModelError>
where
    R: NemotronRuntime,
    L: FnOnce(C) -> Result<R, ModelError>,
{
    lockstep::boot_leader_group(links, local, t, make_comm, move |c| load(c).map(|m| NemoRt::new(m, Box::new(DeviceFeatures))))
}

/// A follower's boot after the connection is up: [`lockstep::boot_follower`] for this family. `load` gets the model options the leader
/// planned (the follower is built with exactly those). The runtime it returns goes to [`run_follower`] with the same link.
pub fn boot_follower<R, C, L>(
    link: Link,
    local: &BootGate,
    t: &TpTimeouts,
    make_comm: impl FnOnce(&mut Rendezvous<'_>) -> Result<C, ModelError>,
    load: L,
) -> Result<(R, Link), ModelError>
where
    R: NemotronRuntime,
    L: FnOnce(&ModelOptions, C) -> Result<R, ModelError>,
{
    lockstep::boot_follower(link, local, t, make_comm, move |gate: &BootGate, c| {
        load(&gate.model_options(), c).map(|m| NemoRt::new(m, Box::new(DeviceFeatures)))
    })
    .map(|(rt, link)| (rt.into_model(), link))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gb_engine::RowRequest;
    use std::io::{self, Write};
    use std::sync::atomic::Ordering;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    // ---------------------------------------------------------------------------------------------------------- a recording runtime

    /// What a mock runtime was asked, as text, in order; the same list from both ranks is the lockstep property.
    type Log = Arc<Mutex<Vec<String>>>;

    /// A deterministic stand-in for a model. Candidates are a pure function of the batch and the call count, so two instances that see
    /// the same calls answer the same.
    struct Mock {
        log: Log,
        rank_tag: u64,
        vocab: usize,
        mtp: bool,
        /// (call name, 1-based occurrence) that fails with `kind` on this instance.
        fail_on: Option<(&'static str, usize, ModelError)>,
        seen: BTreeMap<&'static str, usize>,
        next_snap: u32,
        live: BTreeMap<u32, (usize, Vec<u32>)>,
        /// Perturbs the candidates of every window (a rank that computes something else).
        skew: f32,
        /// Sleeps in `window_candidates` (a slow rank).
        window_sleep: Duration,
        /// Committed rows the head has not absorbed (what `mtp_pending_rows` answers).
        pending: usize,
        /// Set by the follower's abort hook: a `window_sleep` in progress returns an error at once (a collective that the
        /// communicator's abort freed).
        abort_flag: Option<Arc<AtomicBool>>,
        /// Snapshot slots this rank's arena holds once `enable_snapshots` has run (`None`: as many as are asked for, and `free_snapshots`
        /// says 4 like it always did). A full arena answers `snapshot_slot` with `Ok(None)`, as the model does.
        arena_slots: Option<usize>,
        /// Bytes of one slot of this rank's arena (what `snapshot_arena_bytes` multiplies by the slots).
        arena_unit: u64,
        arena_bytes: u64,
        /// How many snapshots this rank holds right now, readable from the test after the follower has taken the mock away.
        live_count: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Mock {
        fn new(log: Log) -> Mock {
            Mock {
                log,
                rank_tag: 0,
                vocab: 1000,
                mtp: true,
                fail_on: None,
                seen: BTreeMap::new(),
                next_snap: 10,
                live: BTreeMap::new(),
                skew: 0.0,
                window_sleep: Duration::ZERO,
                pending: 1,
                abort_flag: None,
                arena_slots: None,
                arena_unit: 0,
                arena_bytes: 0,
                live_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }

        fn note(&mut self, name: &'static str, detail: String) -> Result<(), ModelError> {
            self.log.lock().unwrap().push(format!("{name} {detail}"));
            let n = self.seen.entry(name).or_insert(0);
            *n += 1;
            if let Some((f, at, e)) = &self.fail_on {
                if *f == name && *at == *n {
                    return Err(match e {
                        ModelError::Request(m) => ModelError::Request(m.clone()),
                        ModelError::StateLost(m) => ModelError::StateLost(m.clone()),
                        ModelError::Busy(m) => ModelError::Busy(m.clone()),
                        _ => ModelError::Config("mock".into()),
                    });
                }
            }
            Ok(())
        }

        fn cands(&self, b: &RowBatch, count: usize, skew: f32) -> Candidates {
            let mut ids = Vec::new();
            let mut vals = Vec::new();
            for r in 0..b.rows {
                for j in 0..count {
                    ids.push((b.ids[r] as usize * 31 + j * 7 + b.positions[r] as usize) as i32 % self.vocab as i32);
                    vals.push(10.0 - j as f32 * 0.5 + (b.positions[r] % 5) as f32 + skew);
                }
            }
            Candidates { rows: b.rows, count, ids, vals }
        }
    }

    impl NemotronRuntime for Mock {
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn has_mtp(&self) -> bool {
            self.mtp
        }
        fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
            self.note("begin_slot", format!("{slot}"))
        }
        fn end_slot(&mut self, slot: usize) {
            let _ = self.note("end_slot", format!("{slot}"));
        }
        fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
            self.note("prefill_chunk", format!("slots={:?} pos0={} rows={} want={want:?}", b.seq_slot, b.positions[0], b.rows))?;
            Ok(want.map(|w| {
                let mut one = b.clone();
                one.ids = vec![b.ids[w]];
                one.positions = vec![b.positions[w]];
                one.rows = 1;
                self.cands(&one, 8, 0.0)
            }))
        }
        fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
            if !self.window_sleep.is_zero() {
                let until = Instant::now() + self.window_sleep;
                while let Some(left) = until.checked_duration_since(Instant::now()) {
                    if self.abort_flag.as_ref().is_some_and(|f| f.load(Ordering::SeqCst)) {
                        return Err(ModelError::StateLost("the communicator was aborted".into()));
                    }
                    std::thread::sleep(left.min(Duration::from_millis(2)));
                }
            }
            self.note("window_candidates", format!("slots={:?} rows={} count={count}", b.seq_slot, b.rows))?;
            Ok(self.cands(b, count, self.skew))
        }
        fn commit(&mut self, slot: usize, keep: usize) {
            let _ = self.note("commit", format!("{slot} keep={keep}"));
        }
        fn flush(&mut self) -> Result<(), ModelError> {
            self.note("flush", String::new())
        }
        fn mtp_draft(&mut self, asks: &[DraftAsk<'_>], keyed: &[Option<Sampling>]) -> Result<Vec<Option<Drafts>>, ModelError> {
            let d: Vec<String> = asks.iter().map(|a| format!("{}:{}:{:?}", a.slot, a.depth, a.tokens)).collect();
            self.note("mtp_draft", format!("{} keyed={keyed:?}", d.join(",")))?;
            Ok(asks.iter().map(|a| (a.depth > 0).then(|| Drafts { ids: vec![1; a.depth], vals: vec![0.5; a.depth] })).collect())
        }
        fn mtp_disable_slot(&mut self, slot: usize) {
            let _ = self.note("mtp_disable_slot", format!("{slot}"));
        }
        fn mtp_pending_rows(&self, _slot: usize) -> Option<usize> {
            Some(self.pending)
        }
        fn snapshots_supported() -> bool {
            true
        }
        fn enable_snapshots(&mut self, slots: usize) -> Result<(), ModelError> {
            self.note("enable_snapshots", format!("{slots}"))?;
            if self.arena_slots.is_none() && self.arena_unit > 0 {
                self.arena_slots = Some(slots);
            }
            self.arena_bytes = self.arena_slots.unwrap_or(slots) as u64 * self.arena_unit;
            Ok(())
        }
        fn snapshot_arena_bytes(&self) -> u64 {
            self.arena_bytes
        }
        fn free_snapshots(&self) -> usize {
            match self.arena_slots {
                Some(cap) => cap.saturating_sub(self.live.len()),
                None => 4,
            }
        }
        fn snapshot_slot(&mut self, slot: usize, len: usize) -> Result<Option<PrefixHandle>, ModelError> {
            self.note("snapshot_slot", format!("{slot} len={len}"))?;
            if self.arena_slots.is_some_and(|cap| self.live.len() >= cap) {
                return Ok(None);
            }
            // each rank numbers its handles its own way: the leader must not assume they agree
            self.next_snap += 1 + self.rank_tag as u32;
            let snap = self.next_snap;
            let pages = vec![snap * 2, snap * 2 + 1];
            self.live.insert(snap, (len, pages.clone()));
            self.live_count.store(self.live.len(), Ordering::SeqCst);
            Ok(Some(PrefixHandle { snap, pages }))
        }
        fn restore_snapshot(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), ModelError> {
            match self.live.get(&handle.snap) {
                Some((l, pages)) if *l == len && *pages == handle.pages => {}
                _ => return Err(ModelError::Request("stale or foreign handle".into())),
            }
            self.note("restore_snapshot", format!("{slot} len={len}"))
        }
        fn release_snapshot(&mut self, handle: PrefixHandle) -> Result<(), ModelError> {
            match self.live.get(&handle.snap) {
                Some((_, pages)) if *pages == handle.pages => {}
                _ => return Err(ModelError::Request("stale or foreign handle".into())),
            }
            // a release the model refuses (the injected fault) changes nothing, as the real model's refusals do
            self.note("release_snapshot", String::new())?;
            self.live.remove(&handle.snap);
            self.live_count.store(self.live.len(), Ordering::SeqCst);
            Ok(())
        }
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

    /// A leader over `leader`, a follower thread over `follower`, and the follower's result.
    fn rig(leader: Mock, follower: Mock) -> (TpLeader<Mock>, std::thread::JoinHandle<Result<FollowerExit, ModelError>>) {
        let (a, b) = Link::pair(Duration::from_secs(2)).unwrap();
        let mut cfg = FollowerConfig::new(quick());
        cfg.bridge = Box::new(NoFeatures);
        let t = std::thread::spawn(move || run_follower(follower, b, cfg));
        let l = TpLeader::start(leader, a, quick()).with_bridge(Box::new(NoFeatures));
        (l, t)
    }

    fn batch(reqs: &[(usize, u64, Vec<u32>)]) -> RowBatch {
        let r: Vec<RowRequest> = reqs
            .iter()
            .map(|(slot, first, toks)| RowRequest { slot: *slot, first_position: *first, tokens: toks.clone() })
            .collect();
        RowBatch::from_requests(&r)
    }

    // ---------------------------------------------------------------------------------------------------------- the wire

    fn sample_gate() -> BootGate {
        BootGate {
            pack: "abc123".into(),
            layers: 0,
            slots: 4,
            kv_pages: 900,
            max_context: 65536,
            prefill_rows: 256,
            family: NemoFacts {
                mtp: true,
                graphs: true,
                mtp_env: true,
                vocab: "n=49152,h=00ff".into(),
                expert_map: 0,
                collective: "ring".into(),
                plan_flags: 0,
                cache_request: "auto".into(),
                cache_slots: 12,
                cache_slot_bytes: 170_246_144,
                cache_pages: 1536,
                prefill: "contract=p1,rows=512,absent=1".into(),
            },
            build: "1.0.0-dev+deadbeef".into(),
            // a follower's gate: the tests that use one gate for both sides compare it with itself
            rank: 1,
            ranks: 2,
            nccl: "channels=4,proto=default,lib=23007".into(),
        }
    }

    #[test]
    fn a_rank_with_other_prefill_settings_is_refused_by_name() {
        use crate::lockstep::FamilyCall;
        let leader = sample_gate().family;
        assert!(NemoCall::check_facts(&leader, &leader.clone(), 1).is_empty());
        for other in ["contract=p1,rows=256,absent=1", "contract=p1,rows=512,absent=0", "contract=p2,rows=4096,absent=1"] {
            let follower = NemoFacts { prefill: other.into(), ..leader.clone() };
            let refusals = NemoCall::check_facts(&leader, &follower, 1);
            assert_eq!(refusals.len(), 1, "{other}: {refusals:?}");
            assert!(refusals[0].contains("REXL3_NEMO_PREFILL_ROWS_MAX"), "{}", refusals[0]);
            // the rows field is the boot's effective prefill rows since v0.1.1: the flag that sets them is named too
            assert!(refusals[0].contains("--prefill-chunk"), "{}", refusals[0]);
        }
    }

    #[test]
    fn a_default_rank_and_an_explicit_p1_rank_are_refused_by_name() {
        // since rc2d the default contract is P2: a rank started without REXL3_NEMO_PREFILL_CONTRACT and a rank started with `=p1`
        // must not pair (their prefill rows have other bits); both facts are built the way `contract::prefill_fingerprint` builds them
        use super::super::contract::{lever_label_for, parse, prefill_fingerprint_for, PrefillContract};
        use crate::lockstep::FamilyCall;
        let levers = lever_label_for(|l| l.default_on);
        // a default boot's rows (`--prefill-chunk` 4,096 at the contract's cap): 4,096 under P2, 512 under P1
        let fact = |c: PrefillContract| prefill_fingerprint_for(c, c.max_prefill_rows(512), true, false, &levers);
        let default = fact(parse(None).unwrap());
        let p1 = fact(parse(Some("p1")).unwrap());
        assert_ne!(default, p1);
        assert!(default.starts_with("contract=p2,"), "{default}");
        let leader = NemoFacts { prefill: default, ..sample_gate().family };
        let follower = NemoFacts { prefill: p1, ..leader.clone() };
        assert!(NemoCall::check_facts(&leader, &leader.clone(), 1).is_empty());
        for (a, b) in [(&leader, &follower), (&follower, &leader)] {
            let refusals = NemoCall::check_facts(a, b, 1);
            assert_eq!(refusals.len(), 1, "{refusals:?}");
            assert!(refusals[0].contains("REXL3_NEMO_PREFILL_CONTRACT"), "{}", refusals[0]);
        }
    }

    #[test]
    fn ranks_that_disagree_on_a_p2tp_lever_are_refused_at_the_hello() {
        // lane S: the router split (R), the shared-expert split and its second communicator (S), the head split (M) and the others
        // ride the prefill fact; a rank with any other setting is refused by name before either rank joins NCCL
        use crate::lockstep::FamilyCall;
        let base = "contract=p2,rows=512,absent=1,tp=I1G1F1O1C1Y1M0R1S1";
        let leader = NemoFacts { prefill: base.into(), ..sample_gate().family };
        assert!(NemoCall::check_facts(&leader, &leader.clone(), 1).is_empty());
        for other in [
            "contract=p2,rows=512,absent=1,tp=I1G1F1O1C1Y1M0R1S0",
            "contract=p2,rows=512,absent=1,tp=I1G1F1O1C1Y1M0R0S1",
            "contract=p2,rows=512,absent=1,tp=I1G1F1O1C1Y1M1R1S1",
        ] {
            let follower = NemoFacts { prefill: other.into(), ..leader.clone() };
            let refusals = NemoCall::check_facts(&leader, &follower, 1);
            assert_eq!(refusals.len(), 1, "{other}: {refusals:?}");
            assert!(refusals[0].contains("REXL3_NEMO_P2TP_"), "{}", refusals[0]);
        }
    }

    #[test]
    fn every_message_survives_the_wire() {
        let b = batch(&[(2, 100, vec![5, 6, 7]), (0, 4, vec![9])]);
        let mut with_rope = b.clone();
        with_rope.rope3 = Some((0..12).collect());
        with_rope.parents = Some(vec![-1, 0, 1, -1]);
        let msgs = vec![
            Msg::Hello(sample_gate()),
            Msg::Abort("no".into()),
            Msg::CommInit(vec![1, 2, 3, 255]),
            Msg::Ready(ReadyInfo { vocab: 131072, has_mtp: true, snapshots: false }),
            Msg::Ping(7),
            Msg::Pong { nonce: 7, applied: 99 },
            Msg::Ack(Class::Ok(0xdead_beef), String::new()),
            Msg::Ack(Class::Err { kind: 4, fatal: false }, "window: slot 3 is not live".into()),
            Msg::Shutdown("bye".into()),
            Msg::Bye("later".into()),
            Msg::BeginSlot(3),
            Msg::EndSlot(3),
            Msg::Prefill { batch: b.clone(), want: Some(2) },
            Msg::Prefill { batch: b.clone(), want: None },
            Msg::Call(NemoCall::PrefillMm {
                batch: with_rope,
                want: Some(0),
                splices: vec![
                    SpliceWire { dst_row: 4, n: 2, data: (0..64u8).collect() },
                    SpliceWire { dst_row: 0, n: 1, data: vec![9] },
                ],
            }),
            Msg::Window { batch: b, count: 28 },
            Msg::Commit { slot: 1, keep: 4 },
            Msg::Flush,
            Msg::Call(NemoCall::MtpDraft {
                asks: vec![AskWire { slot: 1, depth: 3, tokens: vec![1, 2, 3] }, AskWire { slot: 2, depth: 0, tokens: vec![] }],
                keyed: vec![Some(Sampling { seed: 42, temperature: 0.7, top_k: 20, top_p: 0.95, min_p: 0.05 }), None],
            }),
            Msg::Call(NemoCall::MtpDisable(2)),
            Msg::Call(NemoCall::EnableSnapshots(6)),
            Msg::Call(NemoCall::SnapshotSlot { slot: 1, len: 4096 }),
            Msg::Call(NemoCall::RestoreSnapshot { slot: 2, len: 4096, of: 17 }),
            Msg::Call(NemoCall::ReleaseSnapshot { of: 17 }),
            Msg::Barrier,
        ];
        for (i, m) in msgs.iter().enumerate() {
            let frame = m.encode(i as u64 + 1);
            let n = u32::from_le_bytes(frame[..4].try_into().unwrap()) as usize;
            assert_eq!(n, frame.len() - 4, "{}", m.name());
            let (seq, back) = Msg::decode(&frame[4..]).unwrap_or_else(|e| panic!("{}: {e}", m.name()));
            assert_eq!((seq, &back), (i as u64 + 1, m), "{}", m.name());
            // and through the reader
            let (seq, back) = read_frame(&mut &frame[..]).unwrap();
            assert_eq!((seq, back), (i as u64 + 1, m.clone()));
        }
    }

    #[test]
    fn a_frame_of_another_version_or_a_damaged_one_is_refused_by_name() {
        let mut frame = Msg::Flush.encode(1);
        frame[4] = PROTOCOL_VERSION + 1;
        match Msg::decode(&frame[4..]) {
            Err(WireError::Version { got, want }) => assert_eq!((got, want), (PROTOCOL_VERSION + 1, PROTOCOL_VERSION)),
            other => panic!("{other:?}"),
        }
        // a body cut short
        let frame = Msg::Prefill { batch: batch(&[(0, 0, vec![1, 2])]), want: None }.encode(1);
        assert!(matches!(Msg::decode(&frame[4..frame.len() - 3]), Err(WireError::Truncated(_) | WireError::Invalid(_))));
        // trailing bytes
        let mut frame = Msg::Flush.encode(1);
        frame.push(0);
        assert!(matches!(Msg::decode(&frame[4..]), Err(WireError::Invalid(_))));
        // an unknown kind
        let mut frame = Msg::Flush.encode(1);
        frame[5] = 200;
        assert!(matches!(Msg::decode(&frame[4..]), Err(WireError::UnknownKind(200))));
        // an oversize length is refused before anything is allocated
        let big = ((MAX_FRAME + 1) as u32).to_le_bytes();
        assert!(matches!(read_frame(&mut &big[..]), Err(WireError::TooLarge(_))));
        // a batch whose arrays do not match its row count
        let mut bad = batch(&[(0, 0, vec![1, 2])]);
        bad.rows = 5;
        let frame = Msg::Prefill { batch: bad, want: None }.encode(1);
        assert!(matches!(Msg::decode(&frame[4..]), Err(WireError::Invalid(_))));
        // a hello without the magic
        let mut frame = Msg::Hello(sample_gate()).encode(0);
        frame[4 + HEADER] = b'X';
        assert!(matches!(Msg::decode(&frame[4..]), Err(WireError::Invalid(_))));
        // clean close at a boundary versus mid-frame
        assert!(matches!(read_frame(&mut &[][..]), Err(WireError::Closed)));
        assert!(matches!(read_frame(&mut &[1u8, 0][..]), Err(WireError::Truncated(_))));
    }

    // ---------------------------------------------------------------------------------------------------------- lockstep

    /// What a scripted engine run asks of the runtime: the calls `NemotronBackend` makes for prefill slices, windows, commits, drafts,
    /// a snapshot and its restore and release, and the end of a slot.
    fn script(rt: &mut impl NemotronRuntime) -> Vec<String> {
        let mut out = Vec::new();
        rt.enable_snapshots(4).unwrap();
        // slot 1: a prompt in two prefill chunks, the last with candidates
        rt.begin_slot(1).unwrap();
        rt.flush().unwrap();
        let c0 = rt.prefill_chunk(&batch(&[(1, 0, vec![11, 12, 13, 14])]), None).unwrap();
        assert!(c0.is_none());
        let c1 = rt.prefill_chunk(&batch(&[(1, 4, vec![15, 16])]), Some(1)).unwrap().unwrap();
        out.push(format!("prefill {:?}", c1.ids));
        // a snapshot at the slice end, then a second slot restored from it
        let h = rt.snapshot_slot(1, 6).unwrap().expect("a free arena slot");
        rt.begin_slot(2).unwrap();
        rt.restore_snapshot(2, &h, 6).unwrap();
        // drafts for both slots (one sampled), a window, the commit
        let t1 = [1u32, 2, 3];
        let t2 = [4u32, 5];
        let keyed = [Some(Sampling { seed: 9, temperature: 0.8, top_k: 20, top_p: 0.9, min_p: 0.0 }), None];
        let asks = [DraftAsk { slot: 1, tokens: &t1, depth: 3 }, DraftAsk { slot: 2, tokens: &t2, depth: 0 }];
        let drafts = rt.mtp_draft(&asks, &keyed).unwrap();
        out.push(format!("drafts {drafts:?}"));
        for step in 0..3u64 {
            rt.flush().unwrap();
            let w =
                rt.window_candidates(&batch(&[(1, 6 + step * 4, vec![20, 21, 22, 23]), (2, 6 + step, vec![30])]), 28).unwrap();
            out.push(format!("window {:?}", &w.ids[..4]));
            rt.commit(1, 3);
            rt.commit(2, 1);
        }
        rt.mtp_disable_slot(2);
        rt.release_snapshot(h).unwrap();
        rt.end_slot(2);
        rt.end_slot(1);
        rt.flush().unwrap();
        out
    }

    #[test]
    fn both_ranks_see_the_same_call_sequence_for_a_scripted_engine_run() {
        let (log_l, log_f) = (Log::default(), Log::default());
        let mut follower = Mock::new(log_f.clone());
        follower.rank_tag = 3; // its snapshot handles differ from the leader's
        let (mut leader, f) = rig(Mock::new(log_l.clone()), follower);
        let out = script(&mut leader);
        leader.barrier().unwrap();
        assert_eq!(leader.failure(), None);
        let stats = leader.stats();
        assert!(stats.commands >= 20 && stats.acked >= stats.commands - 1, "{stats:?}");
        drop(leader); // Shutdown
        assert_eq!(f.join().unwrap().unwrap(), FollowerExit::Shutdown("leader shut down".into()));
        let (l, f) = (log_l.lock().unwrap().clone(), log_f.lock().unwrap().clone());
        assert_eq!(l, f, "the follower replayed a different call stream");
        for name in [
            "begin_slot",
            "end_slot",
            "prefill_chunk",
            "window_candidates",
            "commit",
            "flush",
            "mtp_draft",
            "mtp_disable_slot",
            "enable_snapshots",
            "snapshot_slot",
            "restore_snapshot",
            "release_snapshot",
        ] {
            assert!(l.iter().any(|c| c.starts_with(name)), "the script never made a {name} call");
        }
        assert_eq!(out.len(), 5);
        // the keyed sampler travelled with the ask
        assert!(l.iter().any(|c| c.starts_with("mtp_draft") && c.contains("seed: 9")), "{l:?}");
    }

    /// The real backend over the leader: the calls `NemotronBackend` makes for a cold prompt with cut points, a restore from one of
    /// its snapshots, a round of head drafts, a window, a commit and the release of everything, replayed identically on the follower.
    #[test]
    fn the_real_backend_over_the_leader_replays_identically_on_the_follower() {
        use super::super::backend::NemotronBackend;
        use super::super::spec::{CostTables, SpecConfig};
        use gb_engine::{Backend, ProposeRequest};
        use gb_sched::prefix::CacheBudget;
        use gb_spec::SpecMode;

        let (log_l, log_f) = (Log::default(), Log::default());
        let mut f = Mock::new(log_f.clone());
        f.rank_tag = 2;
        let (leader, t) = rig(Mock::new(log_l.clone()), f);
        let opts = ModelOptions { slots: 4, kv_pages: 64, max_context: 4096, prefill_rows: 8, layers: None, mtp: true };
        let budget = CacheBudget { snapshot_slots: 4, page_blocks: 16, snapshot_bytes: 1, page_bytes: 1 };
        let mut be = NemotronBackend::with_runtime(leader, 8, &opts)
            .unwrap()
            .with_prefix_cache(77, budget)
            .unwrap()
            .with_speculation(SpecConfig::new(SpecMode::Mtp), &CostTables::shipped())
            .unwrap();
        be.m.enable_snapshots(4).unwrap();
        let prompt: Vec<u32> = (100..120).collect();
        // a cold prompt in slices of 8, cut at 8 and 16
        let (mut done, mut captured) = (0, Vec::new());
        while done < prompt.len() {
            let s = be.prefill_slice_cuts(0, &prompt, &[], done, 8, &[8, 16]);
            captured.extend(s.captured);
            done = s.step.done;
        }
        assert_eq!(captured.iter().map(|c| c.len).collect::<Vec<_>>(), vec![8, 16]);
        assert!(be.take_error().is_none());
        // slot 1 starts from the 16-token snapshot and prefills the rest of the same prompt
        be.prefix_restore(1, &captured[1].handle, 16).unwrap();
        let s = be.prefill_slice(1, &prompt, &[], 16, 8);
        assert_eq!(s.done, 20);
        // a round of head drafts for both slots, a window of them, commits
        let samp = gb_sampling::Sampling::default();
        let reqs = [
            ProposeRequest { slot: 0, tokens: &prompt, sampling: &samp, max: 3 },
            ProposeRequest { slot: 1, tokens: &prompt, sampling: &samp, max: 3 },
        ];
        let proposals = be.propose_batch_scored(&reqs);
        assert_eq!(proposals.len(), 2);
        let rows = be.step(
            &[
                RowRequest { slot: 0, first_position: 19, tokens: vec![prompt[19], 5, 6, 7] },
                RowRequest { slot: 1, first_position: 19, tokens: vec![prompt[19]] },
            ],
            8,
        );
        assert_eq!(rows.iter().map(Vec::len).collect::<Vec<_>>(), vec![4, 1]);
        be.commit(0, 2);
        be.commit(1, 1);
        be.release(0);
        be.release(1);
        for c in captured {
            be.prefix_release(c.handle);
        }
        assert!(be.take_error().is_none(), "{:?}", be.take_error());
        be.m.barrier().unwrap();
        assert_eq!(be.m.failure(), None);
        drop(be);
        assert!(matches!(t.join().unwrap(), Ok(FollowerExit::Shutdown(_))));
        let (l, f) = (log_l.lock().unwrap().clone(), log_f.lock().unwrap().clone());
        assert_eq!(l, f);
        for name in [
            "begin_slot",
            "flush",
            "prefill_chunk",
            "snapshot_slot",
            "restore_snapshot",
            "mtp_draft",
            "window_candidates",
            "commit",
            "end_slot",
            "release_snapshot",
        ] {
            assert!(l.iter().any(|c| c.starts_with(name)), "the backend never made a {name} call: {l:?}");
        }
    }

    #[test]
    fn local_reads_are_not_forwarded_and_snapshot_support_is_the_runtimes() {
        let (log_l, log_f) = (Log::default(), Log::default());
        let (leader, f) = rig(Mock::new(log_l.clone()), Mock::new(log_f.clone()));
        assert_eq!(
            (leader.vocab_size(), leader.has_mtp(), leader.mtp_pending_rows(0), leader.free_snapshots()),
            (1000, true, Some(1), 4)
        );
        assert!(<TpLeader<Mock> as NemotronRuntime>::snapshots_supported());
        drop(leader);
        f.join().unwrap().unwrap();
        assert!(log_l.lock().unwrap().is_empty() && log_f.lock().unwrap().is_empty());
    }

    // ---------------------------------------------------------------------------------------------------------- failures

    #[test]
    fn the_same_error_on_both_ranks_is_not_fatal_and_the_run_goes_on() {
        let (log_l, log_f) = (Log::default(), Log::default());
        let mut l = Mock::new(log_l);
        let mut f = Mock::new(log_f);
        // both ranks refuse the second begin_slot the same way (a request the model cannot serve)
        l.fail_on = Some(("begin_slot", 2, ModelError::Request("slot out of range".into())));
        f.fail_on = Some(("begin_slot", 2, ModelError::Request("slot out of range".into())));
        let (mut leader, t) = rig(l, f);
        leader.begin_slot(0).unwrap();
        assert!(matches!(leader.begin_slot(1), Err(ModelError::Request(_))));
        leader.begin_slot(2).unwrap();
        leader.flush().unwrap();
        leader.barrier().unwrap();
        assert_eq!(leader.failure(), None);
        drop(leader);
        assert!(matches!(t.join().unwrap(), Ok(FollowerExit::Shutdown(_))));
    }

    #[test]
    fn a_failure_on_one_rank_only_is_fatal_for_the_group() {
        let (log_l, log_f) = (Log::default(), Log::default());
        let mut f = Mock::new(log_f);
        f.fail_on = Some(("begin_slot", 2, ModelError::Request("only here".into())));
        let (mut leader, t) = rig(Mock::new(log_l), f);
        leader.begin_slot(0).unwrap();
        // the leader's begin succeeds, the follower's does not: the ack arrives asynchronously and the group is lost no later than
        // the next barrier
        let _ = leader.begin_slot(1);
        let r = leader.barrier();
        assert!(matches!(r, Err(ModelError::StateLost(_))), "{r:?}");
        assert!(r.unwrap_err().is_fatal());
        let again = leader.flush();
        assert!(matches!(&again, Err(e) if e.is_fatal()), "{again:?}");
        let why = leader.failure().unwrap();
        assert!(why.contains("begin_slot") && why.contains("call 2"), "{why}");
        drop(leader);
        let _ = t.join();
    }

    #[test]
    fn candidates_that_differ_between_ranks_fail_the_group_at_the_call_that_made_them() {
        let (log_l, log_f) = (Log::default(), Log::default());
        let mut f = Mock::new(log_f);
        f.skew = 0.25;
        let (mut leader, t) = rig(Mock::new(log_l), f);
        leader.begin_slot(0).unwrap();
        leader.prefill_chunk(&batch(&[(0, 0, vec![1, 2, 3])]), Some(2)).unwrap();
        // a window on the skewed rank
        let w = leader.window_candidates(&batch(&[(0, 3, vec![4])]), 4);
        // the verdict may land just after the call returns: wait for it
        let r = leader.barrier();
        assert!(w.is_err() || r.is_err());
        let why = leader.failure().expect("the group must have failed");
        assert!(why.contains("diverged") && why.contains("window_candidates"), "{why}");
        drop(leader);
        let _ = t.join();
    }

    #[test]
    fn a_draft_failure_on_one_rank_only_is_fatal_for_the_group_whatever_its_kind() {
        // the split draft head holds collectives inside the call: an asymmetric error is not a local matter, so it is not forgiven
        // (it used to be: "only the leader consumes the head's results")
        let toks = [1u32, 2];
        let asks = [DraftAsk { slot: 0, tokens: &toks, depth: 2 }];
        for (kind, fatal) in [
            (ModelError::Request("head scratch".into()), false),
            (ModelError::Busy("head busy".into()), false),
            (ModelError::StateLost("device".into()), true),
        ] {
            let (log_l, log_f) = (Log::default(), Log::default());
            let mut f = Mock::new(log_f);
            f.fail_on = Some(("mtp_draft", 1, kind));
            let (mut leader, t) = rig(Mock::new(log_l), f);
            leader.begin_slot(0).unwrap();
            let _ = leader.mtp_draft(&asks, &[None]);
            let r = leader.barrier();
            assert!(matches!(&r, Err(e) if e.is_fatal()), "fatal={fatal}: {r:?}");
            let why = leader.failure().unwrap();
            assert!(why.contains("mtp_draft") && why.contains("rank 1"), "{why}");
            assert_eq!(leader.stats().tolerated, 0);
            drop(leader);
            let _ = t.join();
        }
        // the same error on both ranks is still not a failure of the group
        let (log_l, log_f) = (Log::default(), Log::default());
        let (mut l, mut f) = (Mock::new(log_l), Mock::new(log_f));
        l.fail_on = Some(("mtp_draft", 1, ModelError::Request("head scratch".into())));
        f.fail_on = Some(("mtp_draft", 1, ModelError::Request("head scratch".into())));
        let (mut leader, t) = rig(l, f);
        leader.begin_slot(0).unwrap();
        assert!(matches!(leader.mtp_draft(&asks, &[None]), Err(ModelError::Request(_))));
        leader.barrier().unwrap();
        assert_eq!(leader.failure(), None);
        drop(leader);
        t.join().unwrap().unwrap();
    }

    #[test]
    fn a_follower_that_dies_fails_the_next_call_instead_of_hanging() {
        let (log_l, _) = (Log::default(), Log::default());
        let (a, b) = Link::pair(Duration::from_secs(2)).unwrap();
        let mut leader = TpLeader::start(Mock::new(log_l), a, quick()).with_bridge(Box::new(NoFeatures));
        // the follower end goes away without a word (a crashed process closes its socket)
        b.close();
        drop(b);
        let t0 = Instant::now();
        let mut failed = None;
        while t0.elapsed() < Duration::from_secs(5) {
            match leader.flush() {
                Ok(()) => std::thread::sleep(Duration::from_millis(10)),
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        let e = failed.expect("the leader must notice a dead follower, not hang");
        assert!(e.is_fatal(), "{e}");
        assert!(t0.elapsed() < Duration::from_secs(3), "noticed after {:?}", t0.elapsed());
        // the void calls have nowhere to report: they are no-ops on a lost group, and the failure stays latched
        leader.commit(0, 1);
        leader.end_slot(0);
        leader.mtp_disable_slot(0);
        assert!(leader.begin_slot(0).is_err());
        assert!(leader.window_candidates(&batch(&[(0, 0, vec![1])]), 4).is_err());
    }

    #[test]
    fn a_follower_that_stops_answering_is_a_lost_peer_after_the_peer_timeout() {
        // the follower end stays open but nobody reads or answers: a hung process
        let (a, _b) = Link::pair(Duration::from_secs(2)).unwrap();
        let mut leader = TpLeader::start(Mock::new(Log::default()), a, quick()).with_bridge(Box::new(NoFeatures));
        let t0 = Instant::now();
        loop {
            if let Err(e) = leader.barrier() {
                assert!(e.is_fatal(), "{e}");
                break;
            }
            assert!(t0.elapsed() < Duration::from_secs(10), "no verdict on a silent peer");
        }
        let why = leader.failure().unwrap();
        assert!(why.contains("no frame from rank 1") || why.contains("did not acknowledge"), "{why}");
    }

    #[test]
    fn a_leader_that_dies_ends_the_follower_loop() {
        let (a, b) = Link::pair(Duration::from_secs(2)).unwrap();
        let t = std::thread::spawn(move || run_follower(Mock::new(Log::default()), b, FollowerConfig::new(quick())));
        a.close();
        drop(a);
        match t.join().unwrap() {
            Ok(FollowerExit::LeaderLost(_)) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_command_out_of_sequence_is_refused() {
        let (mut a, b) = Link::pair(Duration::from_secs(2)).unwrap();
        let t = std::thread::spawn(move || run_follower(Mock::new(Log::default()), b, FollowerConfig::new(quick())));
        a.send(1, &Msg::BeginSlot(0)).unwrap();
        a.send(3, &Msg::BeginSlot(1)).unwrap(); // 2 was lost
        let r = t.join().unwrap();
        assert!(matches!(&r, Err(ModelError::StateLost(m)) if m.contains("sequence number 3")), "{r:?}");
    }

    // ---------------------------------------------------------------------------------------------------------- snapshots

    #[test]
    fn a_snapshot_handle_the_follower_made_differently_still_restores_and_a_foreign_one_is_refused_alone() {
        let (log_l, log_f) = (Log::default(), Log::default());
        let mut f = Mock::new(log_f.clone());
        f.rank_tag = 5;
        let (mut leader, t) = rig(Mock::new(log_l.clone()), f);
        leader.enable_snapshots(2).unwrap();
        leader.begin_slot(0).unwrap();
        let h = leader.snapshot_slot(0, 512).unwrap().unwrap();
        leader.begin_slot(1).unwrap();
        leader.restore_snapshot(1, &h, 512).unwrap();
        // a handle the leader never made: the runtime refuses it, the follower is not told, the group is fine
        let foreign = PrefixHandle { snap: 9999, pages: vec![1] };
        assert!(matches!(leader.restore_snapshot(1, &foreign, 512), Err(ModelError::Request(_))));
        // a stale one: same snap, other pages. Neither its restore nor its release is forwarded: the follower holds the live snapshot of
        // that tag and would restore or release IT, so the ranks would part (the leader's own refusal is what keeps them together)
        let stale = PrefixHandle { snap: h.snap, pages: vec![42] };
        assert!(matches!(leader.restore_snapshot(1, &stale, 512), Err(ModelError::Request(_))));
        assert!(matches!(leader.release_snapshot(stale), Err(ModelError::Request(_))));
        leader.release_snapshot(h.clone()).unwrap();
        // released: a second release is refused by name, locally
        assert!(matches!(leader.release_snapshot(h), Err(ModelError::Request(_))));
        leader.barrier().unwrap();
        assert_eq!(leader.failure(), None);
        drop(leader);
        t.join().unwrap().unwrap();
        assert_eq!(log_l.lock().unwrap().clone(), log_f.lock().unwrap().clone());
    }

    // ---- the prefix cache under TP (docs/NEMOTRON_TP.md 12.5)

    /// A pair of mocks with an arena of `leader_slots` and `follower_slots` snapshot slots (the follower's handles numbered differently),
    /// the leader over the first, the follower's loop on a thread, and each rank's count of live snapshots.
    #[allow(clippy::type_complexity)]
    fn cache_rig(
        leader_slots: usize,
        follower_slots: usize,
    ) -> (
        TpLeader<Mock>,
        std::thread::JoinHandle<Result<FollowerExit, ModelError>>,
        Log,
        Log,
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    ) {
        let (log_l, log_f) = (Log::default(), Log::default());
        let (mut l, mut f) = (Mock::new(log_l.clone()), Mock::new(log_f.clone()));
        f.rank_tag = 7;
        l.arena_slots = Some(leader_slots);
        f.arena_slots = Some(follower_slots);
        let (live_l, live_f) = (l.live_count.clone(), f.live_count.clone());
        let (leader, t) = rig(l, f);
        (leader, t, log_l, log_f, live_l, live_f)
    }

    fn live(a: &Arc<std::sync::atomic::AtomicUsize>) -> usize {
        a.load(Ordering::SeqCst)
    }

    #[test]
    fn eviction_in_the_leaders_order_keeps_both_handle_tables_and_both_arenas_in_step() {
        let (mut leader, t, log_l, log_f, live_l, live_f) = cache_rig(2, 2);
        leader.enable_snapshots(2).unwrap();
        leader.begin_slot(0).unwrap();
        // the engine's index holds two entries, oldest first; a third cut does not fit and is skipped, on BOTH ranks
        let a = leader.snapshot_slot(0, 256).unwrap().unwrap();
        let b = leader.snapshot_slot(0, 512).unwrap().unwrap();
        assert!(leader.snapshot_slot(0, 768).unwrap().is_none(), "a full arena skips the cut");
        leader.barrier().unwrap();
        assert_eq!((leader.inner().leader_snaps.len(), live(&live_l), live(&live_f)), (2, 2, 2));
        assert_eq!(leader.free_snapshots(), 0);
        // the oldest is evicted (released); its slot is free again on both ranks and the next cut takes it
        leader.release_snapshot(a.clone()).unwrap();
        let c = leader.snapshot_slot(0, 768).unwrap().expect("the evicted slot is free on both ranks");
        leader.barrier().unwrap();
        assert_eq!((leader.inner().leader_snaps.len(), live(&live_l), live(&live_f)), (2, 2, 2));
        // the evicted handle is refused by name on the leader, and the follower is never told (the group is unaffected)
        leader.begin_slot(1).unwrap();
        let sent = leader.stats().commands;
        assert!(matches!(leader.restore_snapshot(1, &a, 256), Err(ModelError::Request(_))));
        assert!(matches!(leader.release_snapshot(a), Err(ModelError::Request(_))));
        assert_eq!(leader.stats().commands, sent, "a refused handle sends nothing");
        // the survivors restore in any order, and are evicted oldest first
        leader.restore_snapshot(1, &c, 768).unwrap();
        leader.restore_snapshot(1, &b, 512).unwrap();
        leader.release_snapshot(b).unwrap();
        leader.release_snapshot(c).unwrap();
        leader.barrier().unwrap();
        assert_eq!(leader.failure(), None);
        assert_eq!((leader.inner().leader_snaps.len(), live(&live_l), live(&live_f)), (0, 0, 0));
        assert_eq!(leader.free_snapshots(), 2);
        drop(leader);
        t.join().unwrap().unwrap();
        assert_eq!(log_l.lock().unwrap().clone(), log_f.lock().unwrap().clone());
    }

    #[test]
    fn the_followers_table_holds_exactly_the_snapshots_the_leader_has_not_released() {
        // the follower's adapter driven by hand, as its loop does: a table that only grows is a leak per snapshot ever taken
        let mut rt = NemoRt::new(Mock::new(Log::default()), Box::new(NoFeatures));
        let snap =
            |rt: &mut NemoRt<Mock>, seq: u64, len: u64| rt.apply_call(seq, NemoCall::SnapshotSlot { slot: 0, len }).unwrap();
        assert_eq!(snap(&mut rt, 1, 256).digest, 2);
        assert_eq!(snap(&mut rt, 2, 512).digest, 2);
        assert_eq!(rt.follower_snaps.len(), 2);
        rt.apply_call(3, NemoCall::RestoreSnapshot { slot: 1, len: 512, of: 2 }).unwrap();
        assert_eq!(rt.follower_snaps.len(), 2, "a restore does not give a snapshot back");
        rt.apply_call(4, NemoCall::ReleaseSnapshot { of: 1 }).unwrap();
        assert_eq!(rt.follower_snaps.keys().copied().collect::<Vec<_>>(), vec![2]);
        // released: restoring or releasing it again is a refusal by name (the leader never forwards it), and changes nothing
        assert!(matches!(rt.apply_call(5, NemoCall::RestoreSnapshot { slot: 1, len: 256, of: 1 }), Err(ModelError::Request(_))));
        assert!(matches!(rt.apply_call(6, NemoCall::ReleaseSnapshot { of: 1 }), Err(ModelError::Request(_))));
        rt.apply_call(7, NemoCall::ReleaseSnapshot { of: 2 }).unwrap();
        assert!(rt.follower_snaps.is_empty());
    }

    #[test]
    fn a_snapshot_both_ranks_refuse_leaves_no_handle_and_the_group_serves_on() {
        let (log_l, log_f) = (Log::default(), Log::default());
        let (mut l, mut f) = (Mock::new(log_l.clone()), Mock::new(log_f.clone()));
        for m in [&mut l, &mut f] {
            m.arena_slots = Some(2);
            m.fail_on = Some(("snapshot_slot", 1, ModelError::Request("the copy was refused".into())));
        }
        let (live_l, live_f) = (l.live_count.clone(), f.live_count.clone());
        let (mut leader, t) = rig(l, f);
        leader.enable_snapshots(2).unwrap();
        leader.begin_slot(0).unwrap();
        assert!(matches!(leader.snapshot_slot(0, 256), Err(ModelError::Request(_))));
        leader.barrier().unwrap();
        assert_eq!((leader.inner().leader_snaps.len(), live(&live_l), live(&live_f)), (0, 0, 0));
        assert_eq!(leader.free_snapshots(), 2, "a refused snapshot holds no arena slot");
        // the next one is taken and used as if nothing had happened
        let h = leader.snapshot_slot(0, 256).unwrap().unwrap();
        leader.begin_slot(1).unwrap();
        leader.restore_snapshot(1, &h, 256).unwrap();
        leader.release_snapshot(h).unwrap();
        leader.barrier().unwrap();
        assert_eq!(leader.failure(), None);
        assert_eq!((leader.inner().leader_snaps.len(), live(&live_l), live(&live_f)), (0, 0, 0));
        drop(leader);
        t.join().unwrap().unwrap();
        assert_eq!(log_l.lock().unwrap().clone(), log_f.lock().unwrap().clone());
    }

    #[test]
    fn an_arena_that_is_full_on_one_rank_only_is_a_lost_group_naming_the_snapshot() {
        // rank 0 has room for two, rank 1 for one: the second cut succeeds on rank 0 and is skipped on rank 1
        let (mut leader, t, _, _, _, _) = cache_rig(2, 1);
        leader.enable_snapshots(2).unwrap();
        leader.begin_slot(0).unwrap();
        leader.snapshot_slot(0, 256).unwrap().unwrap();
        let second = leader.snapshot_slot(0, 512);
        let verdict = leader.barrier();
        assert!(second.is_err() || verdict.is_err(), "the disagreement must surface");
        let why = leader.failure().expect("the group must have failed");
        assert!(why.contains("diverged") && why.contains("snapshot_slot") && why.contains("rank 1"), "{why}");
        drop(leader);
        let _ = t.join();
    }

    #[test]
    fn a_restore_that_fails_on_one_rank_only_is_a_lost_group_never_silent() {
        let (log_l, log_f) = (Log::default(), Log::default());
        let mut f = Mock::new(log_f);
        f.fail_on = Some(("restore_snapshot", 1, ModelError::Request("page copy refused".into())));
        let (mut leader, t) = rig(Mock::new(log_l), f);
        leader.enable_snapshots(2).unwrap();
        leader.begin_slot(0).unwrap();
        let h = leader.snapshot_slot(0, 256).unwrap().unwrap();
        leader.begin_slot(1).unwrap();
        // rank 0 restored, rank 1 did not: the leader's own call may return before the ack lands, but the group is lost by the barrier
        let r = leader.restore_snapshot(1, &h, 256);
        let verdict = leader.barrier();
        assert!(r.is_err() || verdict.is_err());
        let why = leader.failure().expect("the group must have failed");
        assert!(why.contains("diverged") && why.contains("restore_snapshot"), "{why}");
        // every later call, the cache's included, returns the loss
        assert!(matches!(leader.release_snapshot(h), Err(e) if e.is_fatal()));
        drop(leader);
        let _ = t.join();
    }

    #[test]
    fn a_release_both_models_refuse_leaves_the_handle_in_both_tables() {
        let (log_l, log_f) = (Log::default(), Log::default());
        let (mut l, mut f) = (Mock::new(log_l.clone()), Mock::new(log_f.clone()));
        for m in [&mut l, &mut f] {
            m.fail_on = Some(("release_snapshot", 1, ModelError::Request("the books refused it".into())));
        }
        let (live_l, live_f) = (l.live_count.clone(), f.live_count.clone());
        let (mut leader, t) = rig(l, f);
        leader.enable_snapshots(2).unwrap();
        leader.begin_slot(0).unwrap();
        let h = leader.snapshot_slot(0, 256).unwrap().unwrap();
        assert!(matches!(leader.release_snapshot(h.clone()), Err(ModelError::Request(_))));
        leader.barrier().unwrap();
        // both ranks still hold it, so the engine's retry (and a restore) is understood by both
        assert_eq!((leader.inner().leader_snaps.len(), live(&live_l), live(&live_f)), (1, 1, 1));
        leader.begin_slot(1).unwrap();
        leader.restore_snapshot(1, &h, 256).unwrap();
        leader.release_snapshot(h).unwrap();
        leader.barrier().unwrap();
        assert_eq!(leader.failure(), None);
        assert_eq!((leader.inner().leader_snaps.len(), live(&live_l), live(&live_f)), (0, 0, 0));
        drop(leader);
        t.join().unwrap().unwrap();
        assert_eq!(log_l.lock().unwrap().clone(), log_f.lock().unwrap().clone());
    }

    #[test]
    fn an_arena_of_another_size_on_the_follower_fails_the_boot_by_name() {
        let (log_l, log_f) = (Log::default(), Log::default());
        let (mut l, mut f) = (Mock::new(log_l), Mock::new(log_f));
        l.arena_unit = 170_246_144;
        f.arena_unit = 170_229_760; // a head-less rank's slot: no carry
        let (mut leader, t) = rig(l, f);
        let r = leader.enable_snapshots(3);
        let why = format!("{}", r.expect_err("the arenas differ"));
        assert!(why.contains("enable_snapshots") && why.contains("rank 1"), "{why}");
        drop(leader);
        let _ = t.join();
        // and arenas of one size agree
        let (log_l, log_f) = (Log::default(), Log::default());
        let (mut l, mut f) = (Mock::new(log_l), Mock::new(log_f));
        l.arena_unit = 170_246_144;
        f.arena_unit = 170_246_144;
        let (mut leader, t) = rig(l, f);
        leader.enable_snapshots(3).unwrap();
        assert_eq!(leader.snapshot_arena_bytes(), 3 * 170_246_144);
        leader.barrier().unwrap();
        drop(leader);
        t.join().unwrap().unwrap();
    }

    #[test]
    fn a_follower_that_leaves_between_a_snapshot_and_its_restore_fails_the_request_with_the_group_lost_error() {
        use super::super::backend::NemotronBackend;
        use gb_engine::Backend;
        use gb_sched::prefix::CacheBudget;
        // the follower's loop, with a stop flag: SIGTERM of rank 1 (a SIGKILL ends the stream without a Bye, which
        // `a_follower_that_dies_fails_the_next_call_instead_of_hanging` covers; the leader's verdict is the same)
        let (a, b) = Link::pair(Duration::from_secs(2)).unwrap();
        let mut cfg = FollowerConfig::new(quick());
        cfg.bridge = Box::new(NoFeatures);
        let stop = cfg.stop.clone();
        let t = std::thread::spawn(move || run_follower(Mock::new(Log::default()), b, cfg));
        let leader = TpLeader::start(Mock::new(Log::default()), a, quick()).with_bridge(Box::new(NoFeatures));
        let opts = ModelOptions { slots: 2, kv_pages: 64, max_context: 4096, prefill_rows: 8, layers: None, mtp: false };
        let budget = CacheBudget { snapshot_slots: 2, page_blocks: 8, snapshot_bytes: 1, page_bytes: 1 };
        let mut be = NemotronBackend::with_runtime(leader, 8, &opts).unwrap().with_prefix_cache(5, budget).unwrap();
        be.m.enable_snapshots(2).unwrap();
        let prompt: Vec<u32> = (100..116).collect();
        let s = be.prefill_slice_cuts(0, &prompt, &[], 0, 8, &[8]);
        assert_eq!(s.captured.len(), 1, "the snapshot landed on both ranks");
        let handle = s.captured[0].handle.clone();
        // rank 1 leaves
        stop.store(true, Ordering::SeqCst);
        let t0 = Instant::now();
        while be.m.failure().is_none() {
            let _ = be.m.flush();
            assert!(t0.elapsed() < Duration::from_secs(10), "the leader never noticed that rank 1 left");
            std::thread::sleep(Duration::from_millis(10));
        }
        let e = be.prefix_restore(1, &handle, 8).expect_err("a restore on a lost group fails");
        assert!(e.contains("tensor-parallel group lost"), "{e}");
        let fatal = be.take_error().expect("the loss is the backend's fatal error");
        assert!(fatal.fatal && fatal.message.contains("group lost"), "{fatal:?}");
        let _ = t.join();
    }

    #[test]
    fn the_gate_names_the_prefix_cache_facts_the_ranks_disagree_on() {
        let a = sample_gate();
        assert_eq!((a.family.cache_request.as_str(), a.family.cache_slots), ("auto", 12));
        // the same budget on both: fine; rank 1 holding more slots and pages (it has no tower and half the experts): fine, it adopts
        // rank 0's
        assert!(BootGate::check(&a, &a).is_ok());
        let mut more = a.clone();
        more.family.cache_slots = 20;
        more.family.cache_pages = 4000;
        more.kv_pages = 5000;
        assert!(BootGate::check(&a, &more).is_ok());
        // a different request is refused by name, with both values, and before weights (this is the Hello's check)
        let mut off = a.clone();
        off.family.cache_request = "off".into();
        off.family.cache_slots = 0;
        off.family.cache_pages = 0;
        let e = BootGate::check(&a, &off).unwrap_err();
        assert!(e.contains("prefix cache request") && e.contains("rank 0 has auto, rank 1 has off"), "{e}");
        assert!(e.contains("prefix cache snapshot slots"), "{e}");
        let mut mib = a.clone();
        mib.family.cache_request = "mib=2048".into();
        let e = BootGate::check(&a, &mib).unwrap_err();
        assert!(e.contains("rank 0 has auto, rank 1 has mib=2048"), "{e}");
        // a slot of another size (another model or head) is refused by name
        let mut sz = a.clone();
        sz.family.cache_slot_bytes -= 16_384;
        let e = BootGate::check(&a, &sz).unwrap_err();
        assert!(e.contains("bytes of one snapshot slot") && e.contains("170246144") && e.contains("170229760"), "{e}");
        // a follower that cannot hold rank 0's slots or pages is named for the cache, not only as "KV pages"
        let mut less = a.clone();
        less.family.cache_slots = 4;
        less.family.cache_pages = 100;
        let e = BootGate::check(&a, &less).unwrap_err();
        assert!(e.contains("prefix cache snapshot slots: rank 0 planned 12, rank 1 can hold only 4"), "{e}");
        assert!(e.contains("prefix cache KV pages: rank 0 planned 1536, rank 1 can hold only 100"), "{e}");
        // the request labels
        use crate::serve::PrefixCacheOption as P;
        assert_eq!(
            [P::Off, P::Mib(0), P::Auto, P::On, P::Mib(450)].map(cache_request_label),
            ["off".to_string(), "off".to_string(), "auto".to_string(), "on".to_string(), "mib=450".to_string()]
        );
        // `with_cache` is what the boot uses
        let budget = gb_sched::prefix::CacheBudget {
            snapshot_slots: 6,
            page_blocks: 768,
            snapshot_bytes: 170_246_144,
            page_bytes: 2_359_296,
        };
        let g = BootGate::local(
            "p".into(),
            &ModelOptions { slots: 1, kv_pages: 1, max_context: 1, prefill_rows: 1, layers: None, mtp: true },
        )
        .with_cache("mib=2048".into(), &budget);
        assert_eq!(
            (g.family.cache_request.as_str(), g.family.cache_slots, g.family.cache_slot_bytes, g.family.cache_pages),
            ("mib=2048", 6, 170_246_144, 768)
        );
        // a gate that was never given a cache says so
        let none = BootGate::local(
            "p".into(),
            &ModelOptions { slots: 1, kv_pages: 1, max_context: 1, prefill_rows: 1, layers: None, mtp: true },
        );
        assert_eq!((none.family.cache_request.as_str(), none.family.cache_slots, none.family.cache_pages), ("off", 0, 0));
    }

    // ---------------------------------------------------------------------------------------------------------- the boot gate

    #[test]
    fn the_gate_names_every_fact_the_ranks_disagree_on() {
        let a = sample_gate();
        assert!(BootGate::check(&a, &a).is_ok());
        let mut b = a.clone();
        b.pack = "other".into();
        b.max_context = 4096;
        b.family.mtp = false;
        b.family.graphs = false;
        let e = BootGate::check(&a, &b).unwrap_err();
        for needle in ["pack", "max context", "MTP head priced", "REXL3_NEMO_GRAPHS", "rank 0 has", "rank 1 has"] {
            assert!(e.contains(needle), "{needle} missing from: {e}");
        }
        // two ranks holding different expert ownership tables are refused by name (the hash, not only the plan-flags bit)
        let mut t = a.clone();
        t.family.expert_map = 0x00e3_f472_ccd0_9984;
        let e = BootGate::check(&a, &t).unwrap_err();
        assert!(e.contains("expert ownership table") && e.contains("00e3f472ccd09984"), "{e}");
        assert!(BootGate::check(&t, &t).is_ok());
        // a ring on one rank and a one-shot exchange on the other would deadlock: refused by name at the Hello, before NCCL
        let mut c = a.clone();
        c.family.collective = "oneshot:max=716800".into();
        let e = BootGate::check(&a, &c).unwrap_err();
        assert!(e.contains("collective policy") && e.contains("oneshot"), "{e}");
        // the plan flags (attention / in_proj / head split / ownership table) too: refused before any rank joins the communicator
        let mut f = a.clone();
        f.family.plan_flags = 0b0000_0101;
        let e = BootGate::check(&a, &f).unwrap_err();
        assert!(e.contains("plan flags") && e.contains("00000101"), "{e}");
        // the follower may hold more, never less
        let mut more = a.clone();
        more.slots = 8;
        more.kv_pages = 5000;
        assert!(BootGate::check(&a, &more).is_ok());
        assert_eq!(a.model_options().slots, 4);
        let e = BootGate::check(&more, &a).unwrap_err();
        assert!(e.contains("slots") && e.contains("KV pages"), "{e}");
        // the options round trip
        let o = ModelOptions { slots: 3, kv_pages: 77, max_context: 1000, prefill_rows: 64, layers: Some(5), mtp: false };
        let g = BootGate::local("p".into(), &o);
        assert_eq!(g.model_options(), o);
        // the prefill fact names the rows the model is built for (v0.1.1), not the environment's cap
        let contract = super::super::contract::current().id();
        assert!(g.family.prefill.starts_with(&format!("contract={contract},rows=64,")), "{}", g.family.prefill);
        let wide = BootGate::local("p".into(), &ModelOptions { prefill_rows: 1024, ..o.clone() });
        assert!(wide.family.prefill.starts_with(&format!("contract={contract},rows=1024,")), "{}", wide.family.prefill);
        let whole = ModelOptions { layers: None, ..o };
        assert_eq!(BootGate::local("p".into(), &whole).model_options(), whole);
    }

    #[test]
    fn a_handshake_over_the_pipe_agrees_and_a_mismatch_refuses_both_sides_by_name() {
        let (a, b) = Link::pair(Duration::from_secs(2)).unwrap();
        let g = sample_gate();
        let (g1, g2) = (g.clone(), g.clone());
        let (mut a, mut b) = (a, b);
        let t = std::thread::spawn(move || handshake(&mut b, Role::Follower, &g2, Duration::from_secs(5)));
        assert_eq!(handshake(&mut a, Role::Leader, &g1, Duration::from_secs(5)).unwrap(), g);
        assert_eq!(t.join().unwrap().unwrap(), g);

        // a pack digest that differs
        let (mut a, mut b) = Link::pair(Duration::from_secs(2)).unwrap();
        let mut other = g.clone();
        other.pack = "different-pack".into();
        let t = std::thread::spawn(move || handshake(&mut b, Role::Follower, &other, Duration::from_secs(5)));
        let le = handshake(&mut a, Role::Leader, &g, Duration::from_secs(5)).unwrap_err().to_string();
        let fe = t.join().unwrap().unwrap_err().to_string();
        for e in [le, fe] {
            assert!(e.contains("pack") && e.contains("different-pack") && e.contains("rank 0"), "{e}");
        }
    }

    #[test]
    fn a_peer_of_another_protocol_version_is_refused_at_the_hello() {
        let (mut a, mut b) = Link::pair(Duration::from_secs(2)).unwrap();
        let g = sample_gate();
        // the "peer" is hand-written: a hello framed with another version byte
        let t = std::thread::spawn(move || {
            let mut frame = Msg::Hello(sample_gate()).encode(0);
            frame[4] = PROTOCOL_VERSION + 1;
            b.tx.write_all(&frame).unwrap();
            // it also reads what the leader sent
            let _ = b.recv(Some(Duration::from_secs(5)));
            b
        });
        let e = handshake(&mut a, Role::Leader, &g, Duration::from_secs(5)).unwrap_err().to_string();
        assert!(e.contains("protocol version") && e.contains("same build"), "{e}");
        let _ = t.join();
    }

    #[test]
    fn boot_runs_gate_rendezvous_load_ready_and_then_lockstep() {
        let (a, b) = Link::pair(Duration::from_secs(2)).unwrap();
        let t = quick();
        let g = sample_gate();
        let (log_l, log_f) = (Log::default(), Log::default());
        let (lf, tf, gf) = (log_f.clone(), t.clone(), g.clone());
        let follower = std::thread::spawn(move || {
            let (rt, link) = boot_follower(
                b,
                &gf,
                &tf,
                |rv| rv.exchange(None),
                |opts, id: Vec<u8>| {
                    assert_eq!(id, b"unique-id");
                    assert_eq!(opts.slots, 4);
                    Ok(Mock::new(lf))
                },
            )
            .unwrap();
            let mut cfg = FollowerConfig::new(tf);
            cfg.bridge = Box::new(NoFeatures);
            run_follower(rt, link, cfg)
        });
        let mut leader = boot_leader(
            a,
            &g,
            &t,
            |rv| rv.exchange(Some(b"unique-id".to_vec())),
            |id: Vec<u8>| {
                assert_eq!(id, b"unique-id");
                Ok(Mock::new(log_l.clone()))
            },
        )
        .unwrap();
        leader.begin_slot(0).unwrap();
        leader.flush().unwrap();
        leader.barrier().unwrap();
        drop(leader);
        assert!(matches!(follower.join().unwrap(), Ok(FollowerExit::Shutdown(_))));
        assert_eq!(log_l.lock().unwrap().clone(), log_f.lock().unwrap().clone());
    }

    #[test]
    fn boot_refuses_shards_that_disagree_about_the_vocabulary() {
        let (a, b) = Link::pair(Duration::from_secs(2)).unwrap();
        let t = quick();
        let g = sample_gate();
        let (tf, gf) = (t.clone(), g.clone());
        let follower = std::thread::spawn(move || {
            boot_follower(
                b,
                &gf,
                &tf,
                |_| Ok(()),
                |_, _| {
                    let mut m = Mock::new(Log::default());
                    m.vocab = 999;
                    Ok(m)
                },
            )
            .map(|_| ())
        });
        let e = match boot_leader(a, &g, &t, |_| Ok(()), |_| Ok(Mock::new(Log::default()))) {
            Ok(_) => panic!("must refuse"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("shards disagree"), "{e}");
        let _ = follower.join();
    }

    #[test]
    fn a_follower_that_refuses_the_bootstrap_fails_both_boots_by_name_before_the_leader_joins() {
        let (a, b) = Link::pair(Duration::from_secs(2)).unwrap();
        let t = quick();
        let g = sample_gate();
        let (tf, gf) = (t.clone(), g.clone());
        let follower = std::thread::spawn(move || {
            boot_follower(
                b,
                &gf,
                &tf,
                |rv| {
                    rv.exchange_checked(None, |b| {
                        assert_eq!(b, b"unique-id+0");
                        Err("rank 0 runs with attention replicated and this rank with attention sliced".into())
                    })
                },
                |_, _: Vec<u8>| -> Result<Mock, ModelError> { panic!("a refused rank must not load") },
            )
            .map(|_| ())
        });
        let joined = Arc::new(AtomicBool::new(false));
        let j = joined.clone();
        let t0 = Instant::now();
        let e = match boot_leader(
            a,
            &g,
            &t,
            |rv| {
                rv.exchange(Some(b"unique-id+0".to_vec())).inspect(|_id| {
                    // what the real make_comm does next: join the communicator, which blocks until the peer has joined
                    j.store(true, Ordering::SeqCst);
                })
            },
            |_: Vec<u8>| -> Result<Mock, ModelError> { panic!("the leader must not load after a refusal") },
        ) {
            Ok(_) => panic!("must refuse"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("refused") && e.contains("attention sliced"), "{e}");
        assert!(!joined.load(Ordering::SeqCst), "the leader went on to join the communicator after the refusal");
        assert!(t0.elapsed() < Duration::from_secs(5), "the refusal took {:?}", t0.elapsed());
        let fe = follower.join().unwrap().unwrap_err().to_string();
        assert!(fe.contains("attention sliced"), "{fe}");
    }

    #[test]
    fn a_follower_that_accepts_the_bootstrap_lets_the_leader_go_on() {
        let (a, b) = Link::pair(Duration::from_secs(2)).unwrap();
        let mut a = a;
        let mut rl = Rendezvous::pair(&mut a, Role::Leader, Duration::from_secs(5));
        let follower = std::thread::spawn(move || {
            let mut b = b;
            let mut rf = Rendezvous::pair(&mut b, Role::Follower, Duration::from_secs(5));
            rf.exchange_checked(None, |bytes| if bytes == b"id" { Ok(()) } else { Err("bad id".into()) })
        });
        assert_eq!(rl.exchange(Some(b"id".to_vec())).unwrap(), b"id");
        assert_eq!(follower.join().unwrap().unwrap(), b"id");
    }

    #[test]
    fn the_follower_aborts_the_communicator_when_the_leader_vanishes_but_not_when_it_says_goodbye() {
        for clean in [false, true] {
            let (mut a, b) = Link::pair(Duration::from_secs(2)).unwrap();
            let aborted = Arc::new(AtomicBool::new(false));
            let ab = aborted.clone();
            let t = quick();
            let follower = std::thread::spawn(move || {
                let mut cfg = FollowerConfig::new(t);
                cfg.bridge = Box::new(NoFeatures);
                cfg.abort_hook = Some(Box::new(move || ab.store(true, Ordering::SeqCst)));
                run_follower(Mock::new(Log::default()), b, cfg)
            });
            std::thread::sleep(Duration::from_millis(100));
            if clean {
                a.send(0, &Msg::Shutdown("bye".into())).unwrap();
            }
            drop(a);
            let r = follower.join().unwrap().unwrap();
            if clean {
                assert!(matches!(r, FollowerExit::Shutdown(_)), "{r:?}");
                assert!(!aborted.load(Ordering::SeqCst), "a clean goodbye must not abort the communicator");
            } else {
                assert!(matches!(r, FollowerExit::LeaderLost(_)), "{r:?}");
                assert!(aborted.load(Ordering::SeqCst), "a leader that vanished must abort the communicator");
            }
        }
    }

    #[test]
    fn the_boot_guard_fires_only_when_the_boot_does_not_finish() {
        let fired = Arc::new(AtomicBool::new(false));
        let f = fired.clone();
        let g = BootGuard::arm_with(Duration::from_millis(50), move || f.store(true, Ordering::SeqCst));
        std::thread::sleep(Duration::from_millis(400));
        assert!(fired.load(Ordering::SeqCst), "a boot that never finished must trip the guard");
        drop(g);
        let fired = Arc::new(AtomicBool::new(false));
        let f = fired.clone();
        let g = BootGuard::arm_with(Duration::from_millis(300), move || f.store(true, Ordering::SeqCst));
        std::thread::sleep(Duration::from_millis(30));
        drop(g);
        std::thread::sleep(Duration::from_millis(600));
        assert!(!fired.load(Ordering::SeqCst), "a boot that finished in time must not trip the guard");
    }

    #[test]
    fn boot_over_tcp_connects_and_runs() {
        let l = listen("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let t = quick();
        let g = sample_gate();
        let (ta, ga) = (t.clone(), g.clone());
        let follower = std::thread::spawn(move || {
            let link = connect_leader(&addr, ta.connect, ta.write).unwrap();
            let (rt, link) = boot_follower(link, &ga, &ta, |_| Ok(()), |_, _| Ok(Mock::new(Log::default()))).unwrap();
            let mut cfg = FollowerConfig::new(ta);
            cfg.bridge = Box::new(NoFeatures);
            run_follower(rt, link, cfg)
        });
        let link = accept_follower(&l, t.connect, t.write).unwrap();
        let mut leader = boot_leader(link, &g, &t, |_| Ok(()), |_| Ok(Mock::new(Log::default()))).unwrap();
        for i in 0..50 {
            leader.begin_slot(i % 3).unwrap_or_else(|e| panic!("{e}"));
            leader.window_candidates(&batch(&[(i % 3, i as u64, vec![7, 8])]), 8).unwrap();
            leader.commit(i % 3, 1);
        }
        leader.barrier().unwrap();
        assert_eq!(leader.stats().commands, 151);
        drop(leader);
        assert!(matches!(follower.join().unwrap(), Ok(FollowerExit::Shutdown(_))));
    }

    /// A leader that is dropped with acks still in flight must end the follower with a Shutdown, not with a reset that races it: the
    /// drop ends its write half, reads what the follower still sends, and only then closes.
    #[test]
    fn a_leader_dropped_with_acks_in_flight_ends_the_follower_cleanly_over_tcp() {
        for round in 0..12 {
            let l = listen("127.0.0.1:0").unwrap();
            let addr = l.local_addr().unwrap().to_string();
            let t = quick();
            let ta = t.clone();
            let follower = std::thread::spawn(move || {
                let link = connect_leader(&addr, ta.connect, ta.write).unwrap();
                let mut m = Mock::new(Log::default());
                m.window_sleep = Duration::from_millis(1);
                let mut cfg = FollowerConfig::new(ta);
                cfg.bridge = Box::new(NoFeatures);
                run_follower(m, link, cfg)
            });
            let link = accept_follower(&l, t.connect, t.write).unwrap();
            let leader = TpLeader::start(Mock::new(Log::default()), link, t).with_bridge(Box::new(NoFeatures));
            let mut leader = leader;
            for i in 0..(20 + round) {
                leader.window_candidates(&batch(&[(0, i as u64, vec![7, 8])]), 8).unwrap_or_else(|e| panic!("{e}"));
            }
            // no barrier: the follower is still applying and acking when the leader goes away
            drop(leader);
            match follower.join().unwrap() {
                Ok(FollowerExit::Shutdown(_)) => {}
                other => panic!("round {round}: {other:?}"),
            }
        }
    }

    #[test]
    fn candidate_digests_ignore_the_sign_of_zero_and_see_everything_else() {
        let c = Candidates { rows: 2, count: 2, ids: vec![1, 2, 3, 4], vals: vec![0.0, -1.0, 2.5, 0.0] };
        let mut z = c.clone();
        z.vals[0] = -0.0;
        z.vals[3] = -0.0;
        assert_eq!(candidates_digest(&c), candidates_digest(&z));
        let mut d = c.clone();
        d.vals[2] = 2.5000002;
        assert_ne!(candidates_digest(&c), candidates_digest(&d));
        let mut e = c.clone();
        e.ids[3] = 5;
        assert_ne!(candidates_digest(&c), candidates_digest(&e));
    }

    // ---------------------------------------------------------------------------------------------------------- N followers

    /// A group of `ranks` ranks over socket pairs: the leader (rank 0) and one `run_follower` thread a follower, on the Mock runtime.
    struct Cluster {
        leader: Option<TpLeader<Mock>>,
        threads: Vec<Option<std::thread::JoinHandle<Result<FollowerExit, ModelError>>>>,
        /// Closes follower `r`'s end of its link from outside: its process is gone (a crash).
        crash: Vec<Arc<dyn Fn() + Send + Sync>>,
        /// Whether follower `r`'s communicator abort hook ran.
        aborted: Vec<Arc<AtomicBool>>,
        logs: Vec<Log>,
        leader_log: Log,
    }

    /// What a cluster is made of: the leader's and followers' timeouts, and a hook that tunes each follower's mock.
    struct Plan {
        ranks: usize,
        leader_t: TpTimeouts,
        follower_t: TpTimeouts,
        leader_mock: Option<Mock>,
        tune: Tune,
    }

    type Tune = Box<dyn Fn(usize, &mut Mock)>;

    impl Plan {
        fn new(ranks: usize) -> Plan {
            Plan { ranks, leader_t: quick(), follower_t: quick(), leader_mock: None, tune: Box::new(|_, _| {}) }
        }

        fn tune(mut self, f: impl Fn(usize, &mut Mock) + 'static) -> Plan {
            self.tune = Box::new(f);
            self
        }

        fn start(self) -> Cluster {
            let leader_log = Log::default();
            let mut links = Vec::new();
            let mut threads = Vec::new();
            let (mut crash, mut aborted, mut logs) = (Vec::new(), Vec::new(), Vec::new());
            for rank in 1..self.ranks {
                let (a, b) = Link::pair(Duration::from_secs(2)).unwrap();
                crash.push(b.close.clone());
                let log = Log::default();
                let mut mock = Mock::new(log.clone());
                mock.rank_tag = rank as u64;
                let flag = Arc::new(AtomicBool::new(false));
                mock.abort_flag = Some(flag.clone());
                (self.tune)(rank, &mut mock);
                let mut cfg = FollowerConfig::new(self.follower_t.clone()).with_rank(rank);
                cfg.bridge = Box::new(NoFeatures);
                let f = flag.clone();
                cfg.abort_hook = Some(Box::new(move || f.store(true, Ordering::SeqCst)));
                aborted.push(flag);
                logs.push(log);
                threads.push(Some(std::thread::spawn(move || run_follower(mock, b, cfg))));
                links.push(a);
            }
            let mut leader_mock = self.leader_mock.unwrap_or_else(|| Mock::new(leader_log.clone()));
            leader_mock.log = leader_log.clone();
            let leader = TpLeader::start_group(leader_mock, links, self.leader_t).with_bridge(Box::new(NoFeatures));
            Cluster { leader: Some(leader), threads, crash, aborted, logs, leader_log }
        }
    }

    impl Cluster {
        fn leader(&mut self) -> &mut TpLeader<Mock> {
            self.leader.as_mut().unwrap()
        }

        /// Waits until follower `rank`'s thread has ended (at most `within`) and returns its result.
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

    impl Drop for Cluster {
        fn drop(&mut self) {
            drop(self.leader.take());
            for t in self.threads.drain(..).flatten() {
                let _ = t.join();
            }
        }
    }

    fn gate_of(rank: usize, ranks: usize) -> BootGate {
        BootGate { rank: rank as u32, ranks: ranks as u32, ..sample_gate() }
    }

    fn leader_gate(ranks: usize) -> BootGate {
        gate_of(0, ranks)
    }

    #[test]
    fn every_follower_of_a_group_replays_the_same_call_stream_and_each_digest_is_compared() {
        for ranks in [3usize, 4, 8] {
            let mut c = Plan::new(ranks).start();
            let out = script(c.leader());
            c.leader().barrier().unwrap();
            assert_eq!(c.leader().failure(), None, "{ranks} ranks");
            assert_eq!(out.len(), 5);
            let stats = c.leader().stats();
            // every command was acknowledged by every follower (the barrier waited for all of them)
            assert_eq!(stats.acked, stats.commands * (ranks as u64 - 1), "{ranks} ranks: {stats:?}");
            let st = c.leader().group_status();
            assert_eq!(
                (st.ranks, st.commands_sent, st.settled, st.lost.is_none()),
                (ranks, stats.commands, stats.commands, true)
            );
            assert_eq!(st.per_rank.len(), ranks - 1);
            for (i, r) in st.per_rank.iter().enumerate() {
                assert_eq!(r.rank, i + 1);
                assert_eq!((r.state, r.connected), ("up", true), "{r:?}");
                assert_eq!((r.acked_seq, r.lag_commands, r.acks, r.diverged), (stats.commands, 0, stats.commands, 0), "{r:?}");
                assert!(r.bytes_sent > 0);
            }
            let leader_log = c.leader_log.lock().unwrap().clone();
            let exits = c.shutdown();
            for (i, e) in exits.into_iter().enumerate() {
                assert_eq!(
                    e.unwrap_or_else(|e| FollowerExit::LeaderLost(format!("ERR {e}"))),
                    FollowerExit::Shutdown("leader shut down".into()),
                    "rank {} of {ranks}",
                    i + 1
                );
            }
            assert!(leader_log.len() > 20);
            for (i, log) in c.logs.iter().enumerate() {
                assert_eq!(*log.lock().unwrap(), leader_log, "rank {} replayed another call stream ({ranks} ranks)", i + 1);
            }
        }
    }

    #[test]
    fn a_rank_whose_candidates_differ_fails_the_group_by_name_whichever_rank_it_is() {
        for bad in 1..4usize {
            let mut c = Plan::new(4).tune(move |r, m| m.skew = if r == bad { 0.25 } else { 0.0 }).start();
            c.leader().begin_slot(0).unwrap();
            c.leader().prefill_chunk(&batch(&[(0, 0, vec![1, 2, 3])]), Some(2)).unwrap();
            let w = c.leader().window_candidates(&batch(&[(0, 3, vec![4])]), 4);
            let r = c.leader().barrier();
            assert!(w.is_err() || r.is_err());
            let why = c.leader().failure().expect("the group must have failed");
            assert!(
                why.contains("diverged") && why.contains("window_candidates") && why.contains(&format!("rank {bad} returned")),
                "{why}"
            );
            let st = c.leader().group_status();
            assert_eq!(st.lost_rank, Some(bad));
            assert_eq!(st.per_rank[bad - 1].diverged, 1);
            assert!(st.per_rank.iter().filter(|r| r.rank != bad).all(|r| r.diverged == 0), "{st:?}");
        }
    }

    #[test]
    fn the_loss_of_any_one_rank_fails_the_group_within_the_timeout_and_frees_the_healthy_survivors() {
        for ranks in [3usize, 4, 8] {
            for dead in 1..ranks {
                // the followers' own silence watchdog is far away (30 s): only the leader closing their links can free a rank that is
                // stuck inside a call ("a collective" with the dead sibling, here a sleep that the abort hook ends)
                let mut p = Plan::new(ranks).tune(|_, m| m.window_sleep = Duration::from_secs(20));
                p.follower_t.peer = Duration::from_secs(30);
                let mut c = p.start();
                c.leader().begin_slot(0).unwrap();
                // every follower enters its window and stays there; the leader's own call returns at once
                let w = c.leader().window_candidates(&batch(&[(0, 0, vec![7, 8])]), 4);
                assert!(w.is_ok());
                std::thread::sleep(Duration::from_millis(100));
                let t0 = Instant::now();
                (c.crash[dead - 1])();
                // the group is lost, naming the rank, well inside the leader's own peer timeout (1.5 s)
                let mut why = None;
                while t0.elapsed() < Duration::from_secs(3) {
                    why = c.leader().failure();
                    if why.is_some() {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                let why = why.unwrap_or_else(|| panic!("{ranks} ranks, rank {dead} lost: the group was not declared lost"));
                assert!(why.contains(&format!("rank {dead}")), "{why}");
                assert!(t0.elapsed() < Duration::from_millis(1400), "{ranks} ranks, rank {dead}: {:?}", t0.elapsed());
                let st = c.leader().group_status();
                assert_eq!(st.lost_rank, Some(dead));
                assert_eq!(st.per_rank[dead - 1].state, "lost");
                // every survivor is freed (its communicator abort ran and its loop ended), long before its own 30 s watchdog
                for r in 1..ranks {
                    let res = c.finished(r, Duration::from_secs(3));
                    assert!(res.is_some(), "{ranks} ranks, rank {dead} lost: rank {r} is still stuck after {:?}", t0.elapsed());
                    assert!(c.aborted[r - 1].load(Ordering::SeqCst), "rank {r} did not abort its communicator");
                    if let Some(Ok(FollowerExit::Shutdown(w))) = res {
                        panic!("rank {r} was told to shut down cleanly: {w}");
                    }
                }
                // every later call on the leader is a fatal error, and the void calls are no-ops
                assert!(matches!(c.leader().flush(), Err(e) if e.is_fatal()));
                c.leader().commit(0, 1);
                c.leader().end_slot(0);
                assert!(c.leader().begin_slot(1).is_err());
            }
        }
    }

    #[test]
    fn a_local_fatal_error_closes_every_link_without_a_goodbye_and_frees_the_followers_stuck_in_a_call() {
        let mut leader_mock = Mock::new(Log::default());
        leader_mock.fail_on = Some(("begin_slot", 2, ModelError::StateLost("device lost".into())));
        let mut p = Plan::new(4).tune(|_, m| m.window_sleep = Duration::from_secs(20));
        p.leader_mock = Some(leader_mock);
        p.follower_t.peer = Duration::from_secs(30);
        let mut c = p.start();
        c.leader().begin_slot(0).unwrap();
        c.leader().window_candidates(&batch(&[(0, 0, vec![7, 8])]), 4).unwrap();
        std::thread::sleep(Duration::from_millis(100));
        let t0 = Instant::now();
        assert!(matches!(c.leader().begin_slot(1), Err(e) if e.is_fatal()));
        for r in 1..4 {
            let res = c.finished(r, Duration::from_secs(3)).unwrap_or_else(|| panic!("rank {r} is still stuck"));
            assert!(c.aborted[r - 1].load(Ordering::SeqCst), "rank {r} did not abort its communicator");
            assert!(!matches!(res, Ok(FollowerExit::Shutdown(_))), "rank {r}: {res:?}");
        }
        assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
        assert!(c.leader().failure().unwrap().contains("rank 0 failed begin_slot"));
    }

    #[test]
    fn a_straggler_shows_in_the_lag_while_the_others_run_on_and_ordering_holds() {
        let mut c =
            Plan::new(4).tune(|r, m| m.window_sleep = if r == 2 { Duration::from_millis(15) } else { Duration::ZERO }).start();
        c.leader().begin_slot(0).unwrap();
        let status = c.leader().status_source();
        let sampler_stop = Arc::new(AtomicBool::new(false));
        let max_lag: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(vec![0; 3]));
        let sampler = {
            let (stop, max_lag) = (sampler_stop.clone(), max_lag.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let s = status();
                    let mut m = max_lag.lock().unwrap();
                    for r in &s.per_rank {
                        m[r.rank - 1] = m[r.rank - 1].max(r.lag_commands);
                    }
                    drop(m);
                    std::thread::sleep(Duration::from_millis(2));
                }
            })
        };
        const WINDOWS: u64 = 40;
        for i in 0..WINDOWS {
            c.leader().window_candidates(&batch(&[(0, i, vec![7, 8])]), 4).unwrap();
        }
        // the fast ranks have applied and acknowledged everything while the slow one is still working through its queue
        let t0 = Instant::now();
        let ahead = loop {
            let s = c.leader().group_status();
            if s.per_rank[0].acked_seq > WINDOWS && s.per_rank[2].acked_seq > WINDOWS {
                break s;
            }
            assert!(t0.elapsed() < Duration::from_secs(5), "{s:?}");
            std::thread::sleep(Duration::from_millis(2));
        };
        assert!(
            ahead.per_rank[1].acked_seq < ahead.per_rank[0].acked_seq,
            "the straggler was not behind while the others had finished: {ahead:?}"
        );
        assert!(ahead.per_rank[1].lag_commands > 0 && ahead.per_rank[1].state == "up");
        assert_eq!(ahead.per_rank[0].diverged + ahead.per_rank[1].diverged + ahead.per_rank[2].diverged, 0);
        c.leader().barrier().unwrap();
        sampler_stop.store(true, Ordering::Relaxed);
        sampler.join().unwrap();
        let lag = max_lag.lock().unwrap().clone();
        assert!(lag[1] >= 10, "the straggler's lag was never seen: {lag:?}");
        // ordering: every rank applied the same commands in the same order, the straggler included
        assert_eq!(c.leader().failure(), None);
        let leader_log = c.leader_log.lock().unwrap().clone();
        for e in c.shutdown() {
            assert!(matches!(e, Ok(FollowerExit::Shutdown(_))));
        }
        for log in &c.logs {
            assert_eq!(*log.lock().unwrap(), leader_log);
        }
    }

    #[test]
    fn a_link_that_does_not_drain_delays_only_its_own_rank_and_then_loses_the_group_by_name() {
        // rank 2's end is held open and never read (a stopped process behind a live socket); every window is a 1.2 MB frame, more than
        // a socket buffer holds, so its writer blocks. The others must still receive every command, and the engine thread never waits.
        let (log_l, log_1, log_3) = (Log::default(), Log::default(), Log::default());
        let mut links = Vec::new();
        let mut threads = Vec::new();
        let mut stalled_end = None;
        for rank in 1..=3usize {
            let wt = if rank == 2 { Duration::from_millis(1500) } else { Duration::from_secs(5) };
            let (a, b) = Link::pair(wt).unwrap();
            links.push(a);
            if rank == 2 {
                stalled_end = Some(b);
                continue;
            }
            let log = if rank == 1 { log_1.clone() } else { log_3.clone() };
            let mut cfg = FollowerConfig::new(TpTimeouts { peer: Duration::from_secs(30), ..quick() }).with_rank(rank);
            cfg.bridge = Box::new(NoFeatures);
            threads.push(std::thread::spawn(move || run_follower(Mock::new(log), b, cfg)));
        }
        let mut t = quick();
        t.peer = Duration::from_secs(30);
        let mut leader = TpLeader::start_group(Mock::new(log_l.clone()), links, t).with_bridge(Box::new(NoFeatures));
        leader.begin_slot(0).unwrap();
        let big: Vec<u32> = (0..100_000).map(|i| i % 1000).collect();
        const FRAMES: usize = 6;
        let t0 = Instant::now();
        for i in 0..FRAMES {
            leader.window_candidates(&batch(&[(0, 0, big.clone())]), 1).unwrap_or_else(|e| panic!("frame {i}: {e}"));
        }
        let sending = t0.elapsed();
        assert!(sending < Duration::from_millis(1200), "the engine thread waited for a link that does not drain: {sending:?}");
        // the healthy ranks apply and acknowledge everything while rank 2 sits there
        let t1 = Instant::now();
        loop {
            let s = leader.group_status();
            if s.per_rank[0].acked_seq as usize > FRAMES && s.per_rank[2].acked_seq as usize > FRAMES {
                assert!(s.per_rank[1].acked_seq == 0, "{s:?}");
                break;
            }
            assert!(t1.elapsed() < Duration::from_secs(10), "{s:?}");
            std::thread::sleep(Duration::from_millis(5));
        }
        // and then its write times out and the group is lost naming it
        let t2 = Instant::now();
        let why = loop {
            if let Some(w) = leader.failure() {
                break w;
            }
            assert!(t2.elapsed() < Duration::from_secs(8), "the stalled link never lost the group");
            std::thread::sleep(Duration::from_millis(10));
        };
        assert!(why.contains("rank 2") && why.contains("sending window_candidates"), "{why}");
        assert_eq!(leader.group_status().lost_rank, Some(2));
        drop(leader);
        for t in threads {
            let _ = t.join();
        }
        drop(stalled_end);
        assert_eq!(log_1.lock().unwrap().len(), FRAMES + 1);
        assert_eq!(log_3.lock().unwrap().len(), FRAMES + 1);
    }

    #[test]
    fn one_chatty_rank_does_not_mask_a_silent_one() {
        // rank 1 answers everything; rank 2's end stays open and says nothing at all (pings included)
        let (a1, b1) = Link::pair(Duration::from_secs(2)).unwrap();
        let (a2, _b2) = Link::pair(Duration::from_secs(2)).unwrap();
        let mut cfg = FollowerConfig::new(quick()).with_rank(1);
        cfg.bridge = Box::new(NoFeatures);
        let f = std::thread::spawn(move || run_follower(Mock::new(Log::default()), b1, cfg));
        let mut t = quick();
        t.peer = Duration::from_millis(900);
        let mut leader = TpLeader::start_group(Mock::new(Log::default()), vec![a1, a2], t).with_bridge(Box::new(NoFeatures));
        let t0 = Instant::now();
        let why = loop {
            let _ = leader.flush();
            if let Some(w) = leader.failure() {
                break w;
            }
            assert!(t0.elapsed() < Duration::from_secs(6), "a silent rank was masked by a chatty one");
            std::thread::sleep(Duration::from_millis(20));
        };
        assert!(why.contains("rank 2") && !why.contains("rank 1 "), "{why}");
        assert!(t0.elapsed() < Duration::from_secs(4), "{:?}", t0.elapsed());
        drop(leader);
        let _ = f.join();
    }

    #[test]
    fn the_ledger_keeps_a_cursor_per_rank_and_settles_at_the_slowest() {
        let mut l = Ledger::new(3);
        // the leader finished 1 and 2; rank 1 acked both, rank 2 only the first, rank 3 nothing
        l.local_done(1, "begin_slot", Class::Ok(0), false);
        l.local_done(2, "flush", Class::Ok(0), false);
        l.ack(1, 1, Class::Ok(0), String::new(), 2);
        l.ack(1, 2, Class::Ok(0), String::new(), 2);
        l.ack(2, 1, Class::Ok(0), String::new(), 2);
        assert_eq!((l.cursor.clone(), l.settled), (vec![2, 1, 0], 0));
        l.ack(3, 1, Class::Ok(0), String::new(), 2);
        assert_eq!(l.settled, 1);
        l.ack(3, 2, Class::Ok(0), String::new(), 2);
        l.ack(2, 2, Class::Ok(0), String::new(), 2);
        assert_eq!((l.settled, l.acked, l.failed.clone()), (2, 6, None));
        assert!(l.local.is_empty() && l.acks.iter().all(|a| a.is_empty()), "everything met is forgotten");
        // acks that arrive before the leader's own result wait for it, per rank
        l.ack(1, 3, Class::Ok(7), String::new(), 3);
        l.ack(2, 3, Class::Ok(7), String::new(), 3);
        l.ack(3, 3, Class::Ok(8), "other".into(), 3);
        assert_eq!(l.settled, 2);
        l.local_done(3, "window_candidates", Class::Ok(7), false);
        assert_eq!(l.settled, 3);
        let why = l.failed.clone().unwrap();
        assert!(why.contains("diverged at call 3 (window_candidates)") && why.contains("rank 3 returned"), "{why}");
        assert_eq!((l.blamed, l.diverged_by.clone()), (Some(3), vec![0, 0, 1]));
    }

    #[test]
    fn the_ledger_refuses_an_ack_for_a_call_never_made_a_repeated_one_and_a_backlog_per_rank() {
        let mut l = Ledger::new(2);
        l.ack(2, 5, Class::Ok(0), String::new(), 3);
        assert!(l.failed.clone().unwrap().contains("rank 2 acknowledged call 5, which the leader never made"));
        let mut l = Ledger::new(2);
        l.local_done(1, "flush", Class::Ok(0), false);
        l.ack(1, 1, Class::Ok(0), String::new(), 1);
        l.ack(1, 1, Class::Ok(0), String::new(), 1);
        assert!(l.failed.clone().unwrap().contains("rank 1 acknowledged call 1 twice"), "{:?}", l.failed);
        // the 4096 guard is per rank: 4096 unpaired acks from each of two ranks are fine, one more from either is not
        let mut l = Ledger::new(2);
        for seq in 1..=MAX_UNPAIRED_ACKS as u64 {
            l.ack(1, seq, Class::Ok(0), String::new(), 1 << 20);
            l.ack(2, seq, Class::Ok(0), String::new(), 1 << 20);
        }
        assert_eq!(l.failed, None);
        l.ack(2, MAX_UNPAIRED_ACKS as u64 + 1, Class::Ok(0), String::new(), 1 << 20);
        assert_eq!(l.failed, None);
        l.ack(2, MAX_UNPAIRED_ACKS as u64 + 2, Class::Ok(0), String::new(), 1 << 20);
        assert!(l.failed.clone().unwrap().contains("rank 2 acknowledged call"));
        assert_eq!(l.blamed, Some(2));
    }

    // ---------------------------------------------------------------------------------------------------------- the group's boot

    /// A follower of a TCP boot: connects, boots on the Mock after `load_for` (answering the leader's bootstrap), and replays until
    /// the end.
    fn tcp_follower(
        addr: String,
        rank: usize,
        ranks: usize,
        delay: Duration,
        load_for: Duration,
    ) -> std::thread::JoinHandle<Result<FollowerExit, ModelError>> {
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            let t = TpTimeouts { connect: Duration::from_secs(4), ready: Duration::from_secs(10), ..quick() };
            let link = connect_leader_as(rank, &addr, t.connect, t.write)?;
            let (rt, link) = boot_follower(
                link,
                &gate_of(rank, ranks),
                &t,
                |rv| rv.exchange_checked(None, |_| Ok(())),
                |_, _id: Vec<u8>| {
                    std::thread::sleep(load_for);
                    Ok(Mock::new(Log::default()))
                },
            )?;
            let mut cfg = FollowerConfig::new(t).with_rank(rank);
            cfg.bridge = Box::new(NoFeatures);
            run_follower(rt, link, cfg)
        })
    }

    /// What a follower sees of a boot the leader refuses: its own gate check passes (or fails, by name), and then the next thing the
    /// leader says is the Abort that carries the reason.
    fn follower_view(mut b: Link, g: BootGate) -> Result<(), String> {
        match handshake(&mut b, Role::Follower, &g, Duration::from_secs(5)) {
            Err(e) => Err(e.to_string()),
            Ok(_) => match b.recv(Some(Duration::from_secs(5))) {
                Ok((_, Msg::Abort(why))) => Err(why),
                Ok(_) => Ok(()),
                Err(e) => Err(e.to_string()),
            },
        }
    }

    #[test]
    fn a_group_boots_whatever_order_and_skew_the_followers_arrive_in_within_one_deadline() {
        for ranks in [3usize, 4, 8] {
            let l = listen("127.0.0.1:0").unwrap();
            let addr = l.local_addr().unwrap().to_string();
            let g = leader_gate(ranks);
            // the highest rank connects first, the lowest last, 80 ms apart
            let followers: Vec<_> = (1..ranks)
                .map(|r| {
                    let delay = Duration::from_millis(80 * (ranks - 1 - r) as u64);
                    tcp_follower(addr.clone(), r, ranks, delay, Duration::from_millis(30 * r as u64))
                })
                .collect();
            let t = TpTimeouts { connect: Duration::from_secs(4), ready: Duration::from_secs(10), ..quick() };
            let t0 = Instant::now();
            let links = accept_group(&l, &g, &t).unwrap();
            assert_eq!(links.len(), ranks - 1);
            let mut leader = boot_leader_group(
                links,
                &g,
                &t,
                |rv| rv.exchange(Some(b"nccl-id".to_vec())),
                |id: Vec<u8>| {
                    assert_eq!(id, b"nccl-id");
                    Ok(Mock::new(Log::default()))
                },
            )
            .unwrap();
            assert!(t0.elapsed() < Duration::from_secs(4), "{:?}", t0.elapsed());
            assert_eq!(leader.ranks(), ranks);
            for i in 0..10 {
                leader.begin_slot(i % 3).unwrap();
                leader.window_candidates(&batch(&[(i % 3, i as u64, vec![7, 8])]), 4).unwrap();
            }
            leader.barrier().unwrap();
            let st = leader.group_status();
            assert_eq!(st.per_rank.iter().map(|r| r.rank).collect::<Vec<_>>(), (1..ranks).collect::<Vec<_>>());
            drop(leader);
            for f in followers {
                assert!(matches!(f.join().unwrap(), Ok(FollowerExit::Shutdown(_))));
            }
        }
    }

    #[test]
    fn a_follower_that_never_connects_within_the_shared_deadline_is_named() {
        let l = listen("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let g = leader_gate(4);
        let t = TpTimeouts { connect: Duration::from_millis(700), ready: Duration::from_secs(10), ..quick() };
        // ranks 1 and 3 are there at once; rank 2 never comes
        let f1 = tcp_follower(addr.clone(), 1, 4, Duration::ZERO, Duration::ZERO);
        let f3 = tcp_follower(addr.clone(), 3, 4, Duration::ZERO, Duration::ZERO);
        let t0 = Instant::now();
        let e = match accept_group(&l, &g, &t) {
            Ok(_) => panic!("a group missing rank 2 must not boot"),
            Err(e) => e.to_string(),
        };
        let took = t0.elapsed();
        assert!(e.contains("rank 2") && e.contains("did not connect") && e.contains("2 of 3"), "{e}");
        assert!(took >= Duration::from_millis(650) && took < Duration::from_secs(3), "{took:?}");
        // the ranks that were there are told by name and end their boots
        for f in [f1, f3] {
            let fe = f.join().unwrap().unwrap_err().to_string();
            assert!(fe.contains("rank 2") && fe.contains("did not connect"), "{fe}");
        }
    }

    #[test]
    fn a_duplicate_a_misnumbered_and_a_wrongly_sized_rank_are_refused_by_name_on_every_link() {
        // (the ranks the three followers claim, the group size each claims, what the leader's error must contain)
        let cases: [([usize; 3], [usize; 3], &str); 4] = [
            ([1, 2, 2], [4, 4, 4], "rank 2 is claimed by two processes"),
            ([1, 2, 7], [4, 4, 4], "says it is rank 7"),
            ([1, 0, 3], [4, 4, 4], "says it is rank 0"),
            ([1, 2, 3], [4, 4, 3], "was started with --tp-ranks 3 and rank 0 with --tp-ranks 4"),
        ];
        for (ranks_claimed, sizes, needle) in cases {
            let mut leader_ends = Vec::new();
            let mut followers = Vec::new();
            for (i, &rank) in ranks_claimed.iter().enumerate() {
                let (a, b) = Link::pair(Duration::from_secs(2)).unwrap();
                leader_ends.push(a);
                let g = BootGate { rank: rank as u32, ranks: sizes[i] as u32, ..sample_gate() };
                followers.push(std::thread::spawn(move || follower_view(b, g)));
            }
            let e = admit_links(leader_ends, &leader_gate(4), &quick()).unwrap_err().to_string();
            assert!(e.contains(needle), "{needle}: {e}");
            // every link is told, whichever rank it was: the reason is in each follower's refusal
            for (i, f) in followers.into_iter().enumerate() {
                let fe = f.join().unwrap().unwrap_err();
                assert!(fe.contains(needle) || fe.contains("agree"), "follower {}: {fe}", i + 1);
            }
        }
    }

    #[test]
    fn a_follower_that_speaks_another_protocol_version_refuses_the_whole_group_by_name() {
        let mut leader_ends = Vec::new();
        let mut followers = Vec::new();
        for rank in 1..=3usize {
            let (a, mut b) = Link::pair(Duration::from_secs(2)).unwrap();
            leader_ends.push(a);
            let g = gate_of(rank, 4);
            followers.push(std::thread::spawn(move || {
                if rank == 2 {
                    // a build of another version: its hello is framed with another version byte
                    let mut frame = Msg::Hello(g).encode(0);
                    frame[4] = PROTOCOL_VERSION + 1;
                    b.tx.write_all(&frame).unwrap();
                    return match b.recv(Some(Duration::from_secs(5))) {
                        Ok((_, Msg::Hello(_))) => match b.recv(Some(Duration::from_secs(5))) {
                            Ok((_, Msg::Abort(why))) => Err(why),
                            other => Err(format!("{other:?}")),
                        },
                        other => Err(format!("{other:?}")),
                    };
                }
                follower_view(b, g)
            }));
        }
        let e = admit_links(leader_ends, &leader_gate(4), &quick()).unwrap_err().to_string();
        assert!(e.contains("speaks protocol version") && e.contains("same build"), "{e}");
        for (i, f) in followers.into_iter().enumerate() {
            let fe = f.join().unwrap().unwrap_err();
            assert!(fe.contains("protocol version"), "rank {}: {fe}", i + 1);
        }
    }

    #[test]
    fn a_mismatched_nccl_fingerprint_is_refused_by_name_with_both_values() {
        let mut leader_ends = Vec::new();
        let mut followers = Vec::new();
        for rank in 1..=2usize {
            let (a, b) = Link::pair(Duration::from_secs(2)).unwrap();
            leader_ends.push(a);
            let mut g = gate_of(rank, 3);
            if rank == 2 {
                g.nccl = nccl_fingerprint_of(false, Some("2"), None, Some(23007));
            }
            followers.push(std::thread::spawn(move || follower_view(b, g)));
        }
        let mut lg = leader_gate(3);
        lg.nccl = nccl_fingerprint_of(false, None, None, Some(23007));
        let e = admit_links(leader_ends, &lg, &quick()).unwrap_err().to_string();
        assert!(
            e.contains("NCCL settings and library")
                && e.contains("rank 0 has channels=4,proto=default,lib=23007")
                && e.contains("rank 2 has channels=2,proto=default,lib=23007"),
            "{e}"
        );
        for f in followers {
            let fe = f.join().unwrap().unwrap_err();
            assert!(fe.contains("NCCL") || fe.contains("refused"), "{fe}");
        }
        // the fingerprint: the unset channel cap is the default the boot exports, the protocol and the library are named, and a group
        // booted without a communicator has nothing to compare
        assert_eq!(nccl_fingerprint_of(false, None, None, Some(23007)), "channels=4,proto=default,lib=23007");
        assert_eq!(nccl_fingerprint_of(false, Some("8"), Some("Simple"), Some(23207)), "channels=8,proto=Simple,lib=23207");
        assert_eq!(nccl_fingerprint_of(false, Some(" "), Some(""), None), "channels=4,proto=default,lib=unavailable");
        assert_eq!(nccl_fingerprint_of(true, Some("2"), Some("LL"), Some(1)), "off");
        // a group size that differs is named too
        let mut a = sample_gate();
        a.rank = 0;
        a.ranks = 4;
        let mut b = sample_gate();
        b.ranks = 3;
        let e = BootGate::check(&a, &b).unwrap_err();
        assert!(e.contains("group size (--tp-ranks): rank 0 has 4, rank 1 has 3"), "{e}");
    }

    #[test]
    fn a_follower_that_refuses_the_bootstrap_stops_the_leader_before_it_joins_and_names_the_rank() {
        let ranks = 4usize;
        let mut leader_ends = Vec::new();
        let mut followers = Vec::new();
        for rank in 1..ranks {
            let (a, mut b) = Link::pair(Duration::from_secs(2)).unwrap();
            leader_ends.push(a);
            followers.push(std::thread::spawn(move || {
                let mut rv = Rendezvous::pair(&mut b, Role::Follower, Duration::from_secs(5));
                rv.exchange_checked(
                    None,
                    |_| {
                        if rank == 3 {
                            Err("rank 0 runs with attention replicated".into())
                        } else {
                            Ok(())
                        }
                    },
                )
            }));
        }
        let joined = Arc::new(AtomicBool::new(false));
        let j = joined.clone();
        let t = TpTimeouts { ready: Duration::from_secs(5), ..quick() };
        let e = match boot_leader_group(
            leader_ends,
            &leader_gate(ranks),
            &t,
            |rv| {
                let id = rv.exchange(Some(b"id".to_vec()))?;
                j.store(true, Ordering::SeqCst);
                Ok(id)
            },
            |_: Vec<u8>| -> Result<Mock, ModelError> { panic!("the leader must not load after a refusal") },
        ) {
            Ok(_) => panic!("must refuse"),
            Err(e) => e.to_string(),
        };
        assert!(e.contains("rank 3 refused the communicator's bootstrap") && e.contains("attention replicated"), "{e}");
        assert!(!joined.load(Ordering::SeqCst), "the leader joined the communicator another rank refused");
        for (i, f) in followers.into_iter().enumerate() {
            let r = f.join().unwrap();
            if i + 1 == 3 {
                assert!(r.unwrap_err().to_string().contains("attention replicated"));
            }
        }
    }

    #[test]
    fn every_follower_must_be_ready_inside_one_deadline_not_one_deadline_each() {
        // boots a group of four whose followers take `loads_ms` to load, with `ready_ms` for all of them; the leader's verdict and how
        // long its wait for the Readys took
        let run = |loads_ms: [u64; 3], ready_ms: u64| -> (Result<(), String>, Duration) {
            let ranks = 4usize;
            let l = listen("127.0.0.1:0").unwrap();
            let addr = l.local_addr().unwrap().to_string();
            let g = leader_gate(ranks);
            let followers: Vec<_> = (1..ranks)
                .map(|r| tcp_follower(addr.clone(), r, ranks, Duration::ZERO, Duration::from_millis(loads_ms[r - 1])))
                .collect();
            let t = TpTimeouts { connect: Duration::from_secs(4), ready: Duration::from_millis(ready_ms), ..quick() };
            let links = accept_group(&l, &g, &t).unwrap();
            let t0 = Instant::now();
            let r = boot_leader_group(
                links,
                &g,
                &t,
                |rv| rv.exchange(Some(b"id".to_vec())),
                |_: Vec<u8>| Ok(Mock::new(Log::default())),
            );
            let took = t0.elapsed();
            let verdict = match r {
                Ok(mut leader) => {
                    leader.barrier().unwrap();
                    drop(leader);
                    Ok(())
                }
                Err(e) => Err(e.to_string()),
            };
            for f in followers {
                let _ = f.join();
            }
            (verdict, took)
        };
        // 0.4, 0.8 and 1.4 s of loading against 1.0 s for all of them, counted from the leader's own load: the third is late, named,
        // at about 1.0 s. (A clock restarted for each follower, as before, would have given the third until 1.8 s and let it through.)
        let (r, took) = run([400, 800, 1400], 1000);
        let e = r.unwrap_err();
        assert!(e.contains("rank 3 did not finish loading its shard within 1s"), "{e}");
        assert!(took >= Duration::from_millis(900) && took < Duration::from_millis(1450), "{took:?}");
        // all three inside the deadline: the group boots
        let (r, _) = run([100, 300, 500], 1500);
        r.unwrap();
    }

    #[test]
    fn a_group_of_a_leader_and_two_followers_serves_a_scripted_run_over_tcp_with_the_real_boot_path() {
        let l = listen("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let g = leader_gate(3);
        let followers: Vec<_> =
            (1..3).map(|r| tcp_follower(addr.clone(), r, 3, Duration::from_millis(40 * r as u64), Duration::ZERO)).collect();
        let t = TpTimeouts { connect: Duration::from_secs(4), ready: Duration::from_secs(10), ..quick() };
        let links = accept_group(&l, &g, &t).unwrap();
        let mut leader =
            boot_leader_group(links, &g, &t, |rv| rv.exchange(Some(b"id".to_vec())), |_: Vec<u8>| Ok(Mock::new(Log::default())))
                .unwrap();
        let out = script(&mut leader);
        leader.barrier().unwrap();
        assert_eq!(out.len(), 5);
        assert_eq!(leader.failure(), None);
        drop(leader);
        for f in followers {
            assert!(matches!(f.join().unwrap(), Ok(FollowerExit::Shutdown(_))));
        }
    }

    #[test]
    fn the_hello_carries_the_rank_the_group_size_and_the_nccl_fingerprint() {
        let g = BootGate { rank: 5, ranks: 8, nccl: "channels=4,proto=Simple,lib=23007".into(), ..sample_gate() };
        let frame = Msg::Hello(g.clone()).encode(0);
        let (_, back) = Msg::decode(&frame[4..]).unwrap();
        assert_eq!(back, Msg::Hello(g));
        // `BootGate::at` sets the position; a single process is rank 0 of 1
        let o = ModelOptions { slots: 1, kv_pages: 1, max_context: 1, prefill_rows: 1, layers: None, mtp: false };
        let l = BootGate::local("p".into(), &o);
        assert_eq!((l.rank, l.ranks), (0, 1));
        let l = l.at(6, 8);
        assert_eq!((l.rank, l.ranks), (6, 8));
        assert!(l.nccl.starts_with("channels=") || l.nccl == "off", "{}", l.nccl);
        // the build and NCCL identity ride inside the same two strings (protocol 4's bytes, multi-line content): they round-trip
        assert!(
            l.build.contains("\ncommit=") && l.build.contains("\nexe_sha256=") && l.build.contains("\ntp_spin_us="),
            "{}",
            l.build
        );
        let g = BootGate {
            build: l.build.clone(),
            nccl: format!(
                "channels=4,proto=default,lib=23007\n{}",
                nccl_env_lines(&[("NCCL_IB_TC".to_string(), "106".to_string())], None, Some("abc"))
            ),
            ..sample_gate()
        };
        let frame = Msg::Hello(g.clone()).encode(0);
        assert_eq!(Msg::decode(&frame[4..]).unwrap().1, Msg::Hello(g));
    }

    #[test]
    fn an_abort_while_a_call_is_in_flight_aborts_the_communicator_at_once_and_is_not_a_goodbye() {
        let (mut a, b) = Link::pair(Duration::from_secs(2)).unwrap();
        let flag = Arc::new(AtomicBool::new(false));
        let mut m = Mock::new(Log::default());
        m.window_sleep = Duration::from_secs(20);
        m.abort_flag = Some(flag.clone());
        let mut cfg = FollowerConfig::new(TpTimeouts { peer: Duration::from_secs(30), ..quick() }).with_rank(3);
        cfg.bridge = Box::new(NoFeatures);
        let f = flag.clone();
        cfg.abort_hook = Some(Box::new(move || f.store(true, Ordering::SeqCst)));
        let t = std::thread::spawn(move || run_follower(m, b, cfg));
        // a window the follower enters and stays in, then the leader says the group is dead (the link stays open)
        a.send(1, &Msg::Window { batch: batch(&[(0, 0, vec![7, 8])]), count: 4 }).unwrap();
        std::thread::sleep(Duration::from_millis(150));
        assert!(!flag.load(Ordering::SeqCst));
        let t0 = Instant::now();
        a.send(0, &Msg::Abort("rank 2 left the group: killed".into())).unwrap();
        while !flag.load(Ordering::SeqCst) {
            assert!(t0.elapsed() < Duration::from_secs(2), "the abort did not reach the communicator");
            std::thread::sleep(Duration::from_millis(5));
        }
        match t.join().unwrap() {
            Ok(FollowerExit::LeaderLost(why)) => assert!(why.contains("aborted the group"), "{why}"),
            // the aborted call returned a fatal error first: also a failure, never a clean exit
            Err(ModelError::StateLost(_)) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_idle_leader_pings_every_follower_and_reports_each_round_trip() {
        let mut c = Plan::new(4).start();
        // quick() pings an idle link every 100 ms: each rank's round trip shows up in the status, per rank
        let t0 = Instant::now();
        let st = loop {
            let st = c.leader().group_status();
            if st.per_rank.iter().all(|r| r.ping_rtt_us.is_some()) {
                break st;
            }
            assert!(t0.elapsed() < Duration::from_secs(3), "{st:?}");
            std::thread::sleep(Duration::from_millis(20));
        };
        for r in &st.per_rank {
            let rtt = r.ping_rtt_us.unwrap();
            assert!((1..1_000_000).contains(&rtt), "{r:?}");
            assert!(r.last_frame_age_ms < 1500, "{r:?}");
        }
        assert_eq!(st.commands_sent, 0);
        assert_eq!(c.leader().failure(), None);
    }

    // ---------------------------------------------------------------------------------------------------------- the golden wire

    /// The bytes of the wire as the code writes them: every message kind, and the whole session of a two-rank and of a three-rank
    /// group (boot, a scripted engine run, shutdown) as each link carried it, in both directions. It began as protocol 2 as the code at
    /// d5ec77d (before the family-neutral extraction) wrote it; protocol 3 re-blessed the Hello (the expert ownership fingerprint,
    /// then the collective policy and the plan flags) and protocol 4 the Hello again (the prefix cache's request, slots, bytes per slot
    /// and pages): every other line is byte-identical to protocol 2's but for the version byte.
    /// `testdata/nemotron_omni/tp_wire_v5.txt`. Regenerate ONLY for a deliberate protocol change (and bump `PROTOCOL_VERSION`, and name
    /// the file after it): `REXL3_TP_WIRE_BLESS=/path/to/tp_wire_v5.txt cargo test ... golden_wire`.
    const WIRE_GOLDEN: &str = include_str!("../../../../../testdata/nemotron_omni/tp_wire_v5.txt");

    /// A stream that records every byte read from and written to it (shared by its clones).
    struct Tap {
        inner: std::os::unix::net::UnixStream,
        sent: Arc<Mutex<Vec<u8>>>,
        got: Arc<Mutex<Vec<u8>>>,
    }

    impl Read for Tap {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            let n = self.inner.read(b)?;
            self.got.lock().unwrap().extend_from_slice(&b[..n]);
            Ok(n)
        }
    }

    impl Write for Tap {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            let n = self.inner.write(b)?;
            self.sent.lock().unwrap().extend_from_slice(&b[..n]);
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            self.inner.flush()
        }
    }

    impl Duplex for Tap {
        fn try_clone_stream(&self) -> io::Result<Self> {
            Ok(Tap { inner: self.inner.try_clone()?, sent: self.sent.clone(), got: self.got.clone() })
        }
        fn shutdown_both(&self) {
            let _ = self.inner.shutdown(std::net::Shutdown::Both);
        }
        fn shutdown_write(&self) {
            let _ = self.inner.shutdown(std::net::Shutdown::Write);
        }
        fn set_read_timeout_of(&self, t: Option<Duration>) -> io::Result<()> {
            self.inner.set_read_timeout(t)
        }
        fn set_write_timeout_of(&self, t: Option<Duration>) -> io::Result<()> {
            self.inner.set_write_timeout(t)
        }
        fn describe_peer(&self) -> String {
            "tap".into()
        }
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// Every message kind, framed as a leader (commands numbered 1..) or a follower (acks of the number they answer) would.
    fn golden_frames() -> Vec<(String, Vec<u8>)> {
        let b = batch(&[(2, 100, vec![5, 6, 7]), (0, 4, vec![9])]);
        let mut with_rope = b.clone();
        with_rope.rope3 = Some((0..12).collect());
        with_rope.parents = Some(vec![-1, 0, 1, -1]);
        let list: Vec<(&str, u64, Msg)> = vec![
            ("hello", 0, Msg::Hello(sample_gate())),
            ("abort", 0, Msg::Abort("no".into())),
            ("comm_init", 0, Msg::CommInit(vec![1, 2, 3, 255])),
            ("ready", 0, Msg::Ready(ReadyInfo { vocab: 131072, has_mtp: true, snapshots: false })),
            ("ping", 0, Msg::Ping(7)),
            ("pong", 0, Msg::Pong { nonce: 7, applied: 99 }),
            ("ack_ok", 7, Msg::Ack(Class::Ok(0xdead_beef), String::new())),
            ("ack_err", 8, Msg::Ack(Class::Err { kind: 4, fatal: false }, "window: slot 3 is not live".into())),
            ("ack_fatal", 9, Msg::Ack(Class::Err { kind: 7, fatal: true }, "device lost".into())),
            ("shutdown", 0, Msg::Shutdown("bye".into())),
            ("bye", 0, Msg::Bye("later".into())),
            ("begin_slot", 1, Msg::BeginSlot(3)),
            ("end_slot", 2, Msg::EndSlot(3)),
            ("prefill_some", 3, Msg::Prefill { batch: b.clone(), want: Some(2) }),
            ("prefill_none", 4, Msg::Prefill { batch: b.clone(), want: None }),
            (
                "prefill_mm",
                5,
                Msg::Call(NemoCall::PrefillMm {
                    batch: with_rope,
                    want: Some(0),
                    splices: vec![
                        SpliceWire { dst_row: 4, n: 2, data: (0..64u8).collect() },
                        SpliceWire { dst_row: 0, n: 1, data: vec![9] },
                    ],
                }),
            ),
            ("window", 6, Msg::Window { batch: b, count: 28 }),
            ("commit", 7, Msg::Commit { slot: 1, keep: 4 }),
            ("flush", 8, Msg::Flush),
            (
                "mtp_draft",
                9,
                Msg::Call(NemoCall::MtpDraft {
                    asks: vec![
                        AskWire { slot: 1, depth: 3, tokens: vec![1, 2, 3] },
                        AskWire { slot: 2, depth: 0, tokens: vec![] },
                    ],
                    keyed: vec![Some(Sampling { seed: 42, temperature: 0.7, top_k: 20, top_p: 0.95, min_p: 0.05 }), None],
                }),
            ),
            ("mtp_disable", 10, Msg::Call(NemoCall::MtpDisable(2))),
            ("enable_snapshots", 11, Msg::Call(NemoCall::EnableSnapshots(6))),
            ("snapshot_slot", 12, Msg::Call(NemoCall::SnapshotSlot { slot: 1, len: 4096 })),
            ("restore_snapshot", 13, Msg::Call(NemoCall::RestoreSnapshot { slot: 2, len: 4096, of: 17 })),
            ("release_snapshot", 14, Msg::Call(NemoCall::ReleaseSnapshot { of: 17 })),
            ("barrier", 15, Msg::Barrier),
        ];
        list.into_iter().map(|(name, seq, m)| (format!("frame {name} seq={seq}"), m.encode(seq))).collect()
    }

    /// The whole session of a group of `ranks` over tapped links on the Mock runtime: the boot (Hello both ways, the bootstrap, the
    /// shard's Ready), a scripted engine run (`script`: snapshots, drafts, windows, commits) and the leader's shutdown; per link, the
    /// bytes the leader wrote and the bytes it read.
    fn golden_session(ranks: usize) -> Vec<(String, Vec<u8>)> {
        // no pings (they would interleave by the clock), a peer timeout nobody reaches
        let t = TpTimeouts { ping: Duration::from_secs(3600), peer: Duration::from_secs(60), ..quick() };
        let g = BootGate { rank: 0, ranks: ranks as u32, ..sample_gate() };
        let mut leader_ends = Vec::new();
        let mut taps = Vec::new();
        let mut followers = Vec::new();
        for rank in 1..ranks {
            let (a, b) = std::os::unix::net::UnixStream::pair().unwrap();
            let (sent, got) = (Arc::new(Mutex::new(Vec::new())), Arc::new(Mutex::new(Vec::new())));
            leader_ends.push(Link::new(Tap { inner: a, sent: sent.clone(), got: got.clone() }, Duration::from_secs(2)).unwrap());
            taps.push((rank, sent, got));
            let link = Link::new(b, Duration::from_secs(2)).unwrap();
            let (tf, gf) = (t.clone(), BootGate { rank: rank as u32, ranks: ranks as u32, ..sample_gate() });
            followers.push(std::thread::spawn(move || {
                let (mut rt, link) = {
                    let mut m = Mock::new(Log::default());
                    m.rank_tag = rank as u64;
                    boot_follower(link, &gf, &tf, |rv| rv.exchange(None), move |_, _id: Vec<u8>| Ok(m)).unwrap()
                };
                let _ = &mut rt;
                let mut cfg = FollowerConfig::new(tf).with_rank(rank);
                cfg.bridge = Box::new(NoFeatures);
                run_follower(rt, link, cfg)
            }));
        }
        let links = admit_links(leader_ends, &g, &t).unwrap();
        let mut leader = boot_leader_group(
            links,
            &g,
            &t,
            |rv| rv.exchange(Some(b"unique-id-of-nccl".to_vec())),
            |_: Vec<u8>| Ok(Mock::new(Log::default())),
        )
        .unwrap()
        .with_bridge(Box::new(NoFeatures));
        let out = script(&mut leader);
        assert_eq!(out.len(), 5);
        leader.barrier().unwrap();
        assert_eq!(leader.failure(), None);
        drop(leader);
        for f in followers {
            assert!(matches!(f.join().unwrap(), Ok(FollowerExit::Shutdown(_))));
        }
        let mut lines = Vec::new();
        for (rank, sent, got) in taps {
            lines.push((format!("session ranks={ranks} leader_to_rank{rank}"), sent.lock().unwrap().clone()));
            lines.push((format!("session ranks={ranks} rank{rank}_to_leader"), got.lock().unwrap().clone()));
        }
        lines
    }

    fn golden_text() -> String {
        let mut out = String::from(
            "# rexl3 tensor-parallel wire, protocol 5 (protocol 2 as captured from the code at d5ec77d, before the family-neutral extraction; the Hello re-blessed for 3, 4 and 5).\n\
             # One line per frame (or per direction of a session): `<label> <hex bytes>`. Test: tp_runtime::tests::golden_wire.\n",
        );
        let mut all = golden_frames();
        all.extend(golden_session(2));
        all.extend(golden_session(3));
        for (label, bytes) in all {
            out.push_str(&format!("{label} {}\n", hex(&bytes)));
        }
        out
    }

    #[test]
    fn golden_wire() {
        let now = golden_text();
        if let Ok(path) = std::env::var("REXL3_TP_WIRE_BLESS") {
            std::fs::write(&path, &now).unwrap();
            eprintln!("wrote {path}");
            return;
        }
        let (a, b): (Vec<&str>, Vec<&str>) = (WIRE_GOLDEN.lines().collect(), now.lines().collect());
        assert_eq!(a.len(), b.len(), "the golden has {} lines, this build wrote {}", a.len(), b.len());
        for (want, got) in a.iter().zip(&b) {
            let label = want.rsplit_once(' ').map_or(*want, |(l, _)| l);
            assert_eq!(want, got, "the wire bytes of `{label}` moved (protocol 5 is frozen: bump PROTOCOL_VERSION to change it)");
        }
    }

    // ---------------------------------------------------------------------------------------------------------- the inline link (R = 2)

    #[test]
    fn a_two_rank_leader_writes_inline_in_order_and_its_pings_never_land_inside_a_command() {
        // pings every 5 ms of quiet; each window is a 600 KB frame, written by the engine thread under the link's lock, with quiet gaps
        // between them in which the watchdog pings under the same lock
        let mut p = Plan::new(2);
        p.leader_t.ping = Duration::from_millis(5);
        let mut c = p.start();
        let big: Vec<u32> = (0..50_000).map(|i| i % 1000).collect();
        c.leader().begin_slot(0).unwrap();
        for i in 0..60u64 {
            c.leader().window_candidates(&batch(&[(0, i, big.clone())]), 1).unwrap_or_else(|e| panic!("window {i}: {e}"));
            if i % 7 == 0 {
                c.leader().flush().unwrap();
            }
            std::thread::sleep(Duration::from_millis(15 + (i % 3) * 15));
        }
        c.leader().barrier().unwrap();
        assert_eq!(c.leader().failure(), None);
        let st = c.leader().group_status();
        let commands = c.leader().stats().commands;
        let r = &st.per_rank[0];
        // no queue in an inline link, every command acked, and pings were answered on the same stream
        assert_eq!((r.queued_frames, r.acks, r.diverged, r.state), (0, commands, 0, "up"), "{r:?}");
        assert!(r.ping_rtt_us.is_some(), "no ping round trip: {r:?}");
        let leader_log = c.leader_log.lock().unwrap().clone();
        assert!(c.shutdown().into_iter().all(|e| matches!(e, Ok(FollowerExit::Shutdown(_)))));
        assert_eq!(*c.logs[0].lock().unwrap(), leader_log, "the follower replayed another call stream");
    }

    #[test]
    fn a_two_rank_inline_write_that_does_not_drain_fails_naming_rank_1_and_the_call() {
        // the follower's end is held open and never read: the first 1.2 MB frame does not fit the socket buffer, so the engine
        // thread's own write times out (400 ms) and the call returns the lost group, naming rank 1 and the command
        let (a, b) = Link::pair(Duration::from_millis(400)).unwrap();
        let mut t = quick();
        t.peer = Duration::from_secs(30);
        let mut leader = TpLeader::start_group(Mock::new(Log::default()), vec![a], t).with_bridge(Box::new(NoFeatures));
        leader.begin_slot(0).unwrap();
        let big: Vec<u32> = (0..100_000).map(|i| i % 1000).collect();
        let t0 = Instant::now();
        let mut failed = None;
        for _ in 0..20 {
            if let Err(e) = leader.window_candidates(&batch(&[(0, 0, big.clone())]), 1) {
                failed = Some(e);
                break;
            }
        }
        let e = failed.expect("a window must fail once the link stops draining");
        assert!(e.is_fatal(), "{e}");
        let why = e.to_string();
        assert!(why.contains("sending window_candidates (call") && why.contains("to rank 1 failed"), "{why}");
        assert!(t0.elapsed() >= Duration::from_millis(350) && t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
        // the same text is the latched reason, and the status blames rank 1 and says why it left
        assert!(leader.failure().unwrap().contains("to rank 1 failed"));
        let st = leader.group_status();
        assert_eq!((st.lost_rank, st.per_rank[0].state, st.per_rank[0].queued_frames), (Some(1), "lost", 0));
        assert!(st.per_rank[0].left.as_deref().is_some_and(|l| l.contains("a write failed")), "{st:?}");
        // and everything after it fails at once
        assert!(leader.flush().is_err());
        drop(leader);
        drop(b);
    }

    #[test]
    fn a_two_rank_leader_sends_its_shutdown_and_then_ends_the_stream_in_that_order() {
        // a hand-driven follower end: it acknowledges the commands, notes every frame, answers the Shutdown with a Bye and reads on
        // until the leader's end of stream
        let (a, mut b) = Link::pair(Duration::from_secs(2)).unwrap();
        let t = quick();
        let seen = std::thread::spawn(move || {
            let mut seen: Vec<String> = Vec::new();
            loop {
                match b.recv(Some(Duration::from_secs(5))) {
                    Ok((_, Msg::Ping(nonce))) => {
                        let _ = b.send(0, &Msg::Pong { nonce, applied: 0 });
                    }
                    Ok((seq, Msg::Shutdown(why))) => {
                        seen.push(format!("shutdown {why}"));
                        let _ = b.send(seq, &Msg::Bye("done".into()));
                    }
                    Ok((seq, m)) if m.is_command() => {
                        seen.push(format!("{} {seq}", m.name()));
                        let _ = b.send(seq, &Msg::Ack(Class::Ok(0), String::new()));
                    }
                    Ok((_, other)) => seen.push(format!("unexpected {}", other.name())),
                    Err(WireError::Closed) => {
                        seen.push("eof".into());
                        return seen;
                    }
                    Err(e) => {
                        seen.push(format!("error {e}"));
                        return seen;
                    }
                }
            }
        });
        let mut leader = TpLeader::start(Mock::new(Log::default()), a, t).with_bridge(Box::new(NoFeatures));
        leader.begin_slot(0).unwrap();
        leader.flush().unwrap();
        leader.commit(0, 1);
        leader.end_slot(0);
        drop(leader);
        let seen = seen.join().unwrap();
        assert_eq!(
            seen,
            vec!["begin_slot 1", "flush 2", "commit 3", "end_slot 4", "shutdown leader shut down", "eof"],
            "the stream must carry the commands, the Shutdown, and only then end"
        );
    }
}
