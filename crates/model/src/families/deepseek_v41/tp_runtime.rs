//! The lockstep runtime of the tensor-parallel DeepSeek-V4.1-Flash engine: [`crate::lockstep`] instantiated for this
//! family, the mirror of the VERIFIED nemotron adapter (`families::nemotron_h_omni::tp_runtime`). The protocol, the
//! leader, the follower loop, the boot and the failure model are the family-neutral module's; what is here is what is
//! only this family's:
//!
//! * [`Dsv41Call`]: the commands beyond the core calls — the two score readouts (`prefill_logits_last_row` /
//!   `prefill_logits_rows`), which write KV and run the combine arms like every prefill, so a follower must replay
//!   them and prove its logits byte-equal by digest;
//! * [`Dsv41Facts`]: the boot facts the ranks must agree on — every environment knob that changes this family's
//!   arithmetic (`--kv-dtype`, `REXL3_DSV41_MOE_TOPP`, `REXL3_DSV41_CED_REPLAY`, the engram sidecar variables,
//!   `--spec`) and the prefix cache's request and snapshot pool, compared in the Hello before any weight is read;
//! * the prefix cache's protocol: rank 0 sizes every rank's host snapshot pool (`PrefixEnable`, settled at the boot);
//!   a snapshot, restore and release are CACHE calls ([`lockstep::TpLeader::forward_cache_call`]) keyed by the
//!   sequence number of the call that took the snapshot (no rank-local id crosses the wire); every rank answers a
//!   snapshot with the content digest of the state it captured (or [`lockstep::SNAPSHOT_SKIPPED`] when its own
//!   memory guard or a full pool skipped it) and a restore with the digest of the state it actually restored, and
//!   rank 0 uses a restore only when every rank's digest equals its own at snapshot time ([`lockstep::SnapshotBook`]).
//!   Anything else is a request error, which the engine answers with a cold prefill on every rank;
//! * [`Dsv41Runtime`]: the model surface the backend drives (the family's own `prefill_chunk(slot, start, tokens,
//!   want)` shape, the windows, the commits, the score readouts), implemented by [`Dsv41Model`] and by the leader
//!   [`TpLeader`], so the backend is the same code on a single rank and on rank 0 of a group;
//! * [`gate_of`]/[`BootGate::model_options`]: the boot gate at this family's facts, and the rank-0 geometry the
//!   follower is BUILT with (the follower may hold more memory, never run other numbers);
//! * the names the rest of the family codes against ([`Msg`], [`Link`], [`Rendezvous`], [`run_follower`],
//!   [`boot_leader_group`], ...), which are the neutral ones at this family's call type.
//!
//! What is deliberately NOT here: the DSpark draft round (`check_tp_plain` refuses the drafting for a group boot, and
//! the leader's [`Dsv41Runtime::propose_mtp`] refuses it again at the seam — a draft round would desync the ranks, so
//! it can never run silently), and the vision splice (this branch's tower is not bound; the server's vision gate
//! refuses image serves for it). The core calls (begin/end slot, prefill, window, commit, flush) ride the neutral
//! protocol; a prefill travels as the model's own single-sequence batch ([`prefill_batch`]/[`prefill_of`]), so what
//! the follower reconstructs is exactly what `Dsv41Model::prefill_chunk` would have built itself.

use super::model::{DraftRound, Dsv41Model, ModelOptions};
use crate::error::ModelError;
use crate::forward::Candidates;
use crate::lockstep;
use crate::rows::RowBatch;
use gb_cuda::dsv4_kvq4::KvFmt;
use std::collections::BTreeMap;
use std::sync::Arc;

// everything family-neutral, under the names this module owns
pub use crate::lockstep::{FollowerExit, TpTimeouts};

/// A message of the protocol at this family's call type.
pub type Msg = lockstep::Msg<Dsv41Call>;
/// The boot gate at this family's facts.
pub type BootGate = lockstep::BootGate<Dsv41Call>;
/// A connection between two ranks of a group.
pub type Link = lockstep::Link<Dsv41Call>;
/// The collective's bootstrap over the links of a group.
pub type Rendezvous<'a> = lockstep::Rendezvous<'a, Dsv41Call>;
/// The leader's runtime: rank 0's model with every state-changing call forwarded.
pub type TpLeader<R> = lockstep::TpLeader<Dsv41Rt<R>>;

// ------------------------------------------------------------------------------------------------------------------- the family's calls

/// The kinds of this family's commands (outside [`lockstep::CORE_KINDS`]; the core calls sit between them).
pub const K_SCORE_LAST_ROW: u8 = 30;
pub const K_SCORE_ROWS: u8 = 31;
pub const K_PREFILL_MM: u8 = 33;
pub const K_TREE_PATH_WINDOW: u8 = 34;
pub const K_TREE_COMMIT_WINDOW: u8 = 35;
pub const K_TREE_DROP_ROUND: u8 = 36;
pub const K_PREFILL_EX: u8 = 37;
pub const K_PREFIX_SNAPSHOT: u8 = 38;
pub const K_PREFIX_RESTORE: u8 = 39;
pub const K_PREFIX_RELEASE: u8 = 40;
pub const K_PREFIX_ENABLE: u8 = 41;

/// Wire representation of one [`super::vision_serve::Dsv41Splice`]: `(rows, tokens, merged_h, merged_w, types, src, dst, n)`.
pub type WireSplice = (Vec<f32>, u32, u32, u32, Vec<i8>, u32, u32, u32);

/// The commands of the DSV4.1 family beyond the core calls. Numbered, sent, applied and acknowledged exactly like a
/// core call; the digest of the score calls is a fold over the answer's f32 bits, so a follower whose logits differ
/// by one bit is a lost group naming the call.
#[derive(Clone, Debug, PartialEq)]
pub enum Dsv41Call {
    /// [`Dsv41Runtime::prefill_logits_last_row`]: the last row's full-vocabulary logits of one prefill chunk.
    ScoreLastRow { slot: u32, start: u64, tokens: Vec<u32> },
    /// [`Dsv41Runtime::prefill_logits_rows`]: every row's logits of one prefill chunk (the teacher-forced readout).
    ScoreRows { slot: u32, start: u64, tokens: Vec<u32> },
    /// [`Dsv41Runtime::prefill_chunk_mm`]: a prefill chunk carrying encoded image span splices.
    PrefillMm {
        slot: u32,
        start: u64,
        tokens: Vec<u32>,
        want: u32,
        span_emb: (Vec<f32>, Vec<f32>, Vec<f32>),
        splices: Vec<WireSplice>,
    },
    /// [`super::tree::TreeRoundHost::tree_path_window`]: one root-to-leaf path window of a draft-tree verify round.
    TreePathWindow { tokens: Vec<u32>, first_position: u64, candidates: u32 },
    /// [`super::tree::TreeRoundHost::tree_commit_window`]: commit the accepted path window `idx`'s first `rows` rows.
    TreeCommitWindow { idx: u32, rows: u32, window: Vec<u32> },
    /// [`super::tree::TreeRoundHost::tree_drop_round`]: abandon the uncommitted tree round in flight.
    TreeDropRound,
    /// [`Dsv41Runtime::prefill_chunk_ex`] with `for_snapshot = true`.
    PrefillEx { slot: u32, start: u64, tokens: Vec<u32>, want: u32, for_snapshot: bool },
    /// [`Dsv41Runtime::prefix_snapshot`]: save `slot`'s state at `len` into a host snapshot of this rank's own pool,
    /// known from then on by this call's sequence number. Answered with [`lockstep::taken_digest`] of the captured
    /// state, or [`lockstep::SNAPSHOT_SKIPPED`] when the rank's memory guard or a full pool skipped it.
    PrefixSnapshot { slot: u32, len: u64 },
    /// [`Dsv41Runtime::restore_prefix`]: restore the snapshot taken by call `of` at `len` into `slot`. Answered with
    /// [`lockstep::taken_digest`] of the state the rank actually restored.
    PrefixRestore { slot: u32, len: u64, of: u64 },
    /// [`Dsv41Runtime::release_prefix`]: free the snapshot taken by call `of` (a rank that holds none answers Ok).
    PrefixRelease { of: u64 },
    /// [`Dsv41Runtime::enable_prefix_cache`]: size this rank's snapshot pool to rank 0's `slots` (answered with the
    /// pool's size, compared like any call: a rank with another pool fails the boot by name).
    PrefixEnable { slots: u32 },
}

/// What the DSV4.1 ranks must agree on, beyond the shared geometry: the environment switches that change this
/// family's arithmetic. A rank that prunes its experts at another top-p, replays its prefill under another CED
/// setting, or maps another engram sidecar would run ARITHMETIC its peer does not — not a slower group, a wrong one —
/// so each is refused at the Hello, before any weight is read, naming the knob.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dsv41Facts {
    /// The KV cache format (`--kv-dtype` / `REXL3_DSV41_KV`): the planes' arithmetic and bytes.
    pub kv: KvFmt,
    /// The MoE decode top-p pruning (`REXL3_DSV41_MOE_TOPP`): the effective setting as the kernel reads it — the
    /// raw env text, or `unset` (the kernel's own default then applies). The golden-exact runs set `1.0` on every
    /// rank; the gate makes a half-configured group impossible to boot.
    pub moe_topp: String,
    /// The CED bounded-replay prefill (`REXL3_DSV41_CED_REPLAY`), the effective setting.
    pub ced_replay: bool,
    /// The engram token map override (`REXL3_DSV41_ENGRAM_TOKEN_MAP`): the raw text (`unset` = the pack's own).
    pub engram_token_map: String,
    /// The engram sidecar (`REXL3_DSV41_ENGRAM_SHARDS`): the raw text (`unset` = none).
    pub engram_shards: String,
    /// The speculative mode as a group may run it (`--spec`; a group boots `none` — `check_tp_plain` refuses the
    /// drafting, this names the flag at the rendezvous).
    pub spec: String,
    /// The prefix cache this rank was started with (`--prefix-cache`): every rank of a group must be asked the same.
    pub cache_request: String,
    /// Host snapshot slots this rank's pool would hold ([`super::backend::prefix_snapshot_slots`],
    /// `REXL3_DSV41_PREFIX_SNAPSHOTS`; 0 with the cache off). Rank 0 sizes every rank's pool to its own count
    /// (`PrefixEnable`), so a follower must be able to hold at least that many.
    pub cache_slots: u64,
}

impl Dsv41Facts {
    /// This process's facts, read from the environment and the options the model will be built with. `kv` is the
    /// format the admission resolved (`--kv-dtype` / `REXL3_DSV41_KV`), not re-read here.
    pub fn of(o: &crate::serve::ServeOptions, kv: KvFmt) -> Dsv41Facts {
        let env_text = |key: &str| std::env::var(key).ok().filter(|v| !v.trim().is_empty()).unwrap_or_else(|| "unset".into());
        Dsv41Facts {
            kv,
            moe_topp: gb_cuda::dsv4_moe::runtime_env("REXL3_DSV41_MOE_TOPP").unwrap_or_else(|| "unset".into()),
            ced_replay: super::model::ced_replay_enabled(),
            engram_token_map: env_text(super::engram::MAP_ENV),
            engram_shards: env_text(super::engram::SHARDS_ENV),
            spec: format!("{}", o.spec),
            cache_request: format!("{:?}", o.prefix_cache),
            cache_slots: match o.prefix_cache {
                crate::serve::PrefixCacheOption::Off | crate::serve::PrefixCacheOption::Mib(0) => 0,
                _ => super::backend::prefix_snapshot_slots() as u64,
            },
        }
    }
}

impl lockstep::FamilyCall for Dsv41Call {
    type Facts = Dsv41Facts;

    fn kind(&self) -> u8 {
        match self {
            Dsv41Call::ScoreLastRow { .. } => K_SCORE_LAST_ROW,
            Dsv41Call::ScoreRows { .. } => K_SCORE_ROWS,
            Dsv41Call::PrefillMm { .. } => K_PREFILL_MM,
            Dsv41Call::TreePathWindow { .. } => K_TREE_PATH_WINDOW,
            Dsv41Call::TreeCommitWindow { .. } => K_TREE_COMMIT_WINDOW,
            Dsv41Call::TreeDropRound => K_TREE_DROP_ROUND,
            Dsv41Call::PrefillEx { .. } => K_PREFILL_EX,
            Dsv41Call::PrefixSnapshot { .. } => K_PREFIX_SNAPSHOT,
            Dsv41Call::PrefixRestore { .. } => K_PREFIX_RESTORE,
            Dsv41Call::PrefixRelease { .. } => K_PREFIX_RELEASE,
            Dsv41Call::PrefixEnable { .. } => K_PREFIX_ENABLE,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Dsv41Call::ScoreLastRow { .. } => "score_last_row",
            Dsv41Call::ScoreRows { .. } => "score_rows",
            Dsv41Call::PrefillMm { .. } => "prefill_mm",
            Dsv41Call::TreePathWindow { .. } => "tree_path_window",
            Dsv41Call::TreeCommitWindow { .. } => "tree_commit_window",
            Dsv41Call::TreeDropRound => "tree_drop_round",
            Dsv41Call::PrefillEx { .. } => "prefill_ex",
            Dsv41Call::PrefixSnapshot { .. } => "prefix_snapshot",
            Dsv41Call::PrefixRestore { .. } => "prefix_restore",
            Dsv41Call::PrefixRelease { .. } => "prefix_release",
            Dsv41Call::PrefixEnable { .. } => "prefix_enable",
        }
    }

    fn encode(&self, e: &mut lockstep::Enc) {
        match self {
            Dsv41Call::ScoreLastRow { slot, start, tokens } | Dsv41Call::ScoreRows { slot, start, tokens } => {
                e.u32(*slot);
                e.u64(*start);
                e.u32s(tokens);
            }
            Dsv41Call::PrefillMm { slot, start, tokens, want, span_emb, splices } => {
                e.u32(*slot);
                e.u64(*start);
                e.u32s(tokens);
                e.u32(*want);
                for v in [&span_emb.0, &span_emb.1, &span_emb.2] {
                    let bits: Vec<u32> = v.iter().map(|x| x.to_bits()).collect();
                    e.u32s(&bits);
                }
                e.u32(splices.len() as u32);
                for (rows, tok_cnt, mh, mw, types, src, dst, n) in splices {
                    let row_bits: Vec<u32> = rows.iter().map(|v| v.to_bits()).collect();
                    e.u32s(&row_bits);
                    e.u32(*tok_cnt);
                    e.u32(*mh);
                    e.u32(*mw);
                    let type_u8: Vec<u8> = types.iter().map(|&t| t as u8).collect();
                    e.bytes(&type_u8);
                    e.u32(*src);
                    e.u32(*dst);
                    e.u32(*n);
                }
            }
            Dsv41Call::TreePathWindow { tokens, first_position, candidates } => {
                e.u32s(tokens);
                e.u64(*first_position);
                e.u32(*candidates);
            }
            Dsv41Call::TreeCommitWindow { idx, rows, window } => {
                e.u32(*idx);
                e.u32(*rows);
                e.u32s(window);
            }
            Dsv41Call::TreeDropRound => {}
            Dsv41Call::PrefillEx { slot, start, tokens, want, for_snapshot } => {
                e.u32(*slot);
                e.u64(*start);
                e.u32s(tokens);
                e.u32(*want);
                e.bool(*for_snapshot);
            }
            Dsv41Call::PrefixSnapshot { slot, len } => {
                e.u32(*slot);
                e.u64(*len);
            }
            Dsv41Call::PrefixRestore { slot, len, of } => {
                e.u32(*slot);
                e.u64(*len);
                e.u64(*of);
            }
            Dsv41Call::PrefixRelease { of } => e.u64(*of),
            Dsv41Call::PrefixEnable { slots } => e.u32(*slots),
        }
    }

    fn decode(kind: u8, d: &mut lockstep::Dec<'_>) -> Result<Option<Dsv41Call>, lockstep::WireError> {
        let one = |d: &mut lockstep::Dec<'_>| -> Result<(u32, u64, Vec<u32>), lockstep::WireError> {
            Ok((d.u32("score slot")?, d.u64("score start")?, d.u32s("score tokens")?))
        };
        Ok(Some(match kind {
            K_SCORE_LAST_ROW => {
                let (slot, start, tokens) = one(d)?;
                Dsv41Call::ScoreLastRow { slot, start, tokens }
            }
            K_SCORE_ROWS => {
                let (slot, start, tokens) = one(d)?;
                Dsv41Call::ScoreRows { slot, start, tokens }
            }
            K_PREFILL_MM => {
                let slot = d.u32("mm slot")?;
                let start = d.u64("mm start")?;
                let tokens = d.u32s("mm tokens")?;
                let want = d.u32("mm want")?;
                let s0: Vec<f32> = d.u32s("mm span start")?.into_iter().map(f32::from_bits).collect();
                let s1: Vec<f32> = d.u32s("mm span end")?.into_iter().map(f32::from_bits).collect();
                let s2: Vec<f32> = d.u32s("mm span newline")?.into_iter().map(f32::from_bits).collect();
                let count = d.u32("mm splice count")? as usize;
                let mut splices = Vec::with_capacity(count);
                for _ in 0..count {
                    let row_bits = d.u32s("mm splice rows")?;
                    let rows: Vec<f32> = row_bits.into_iter().map(f32::from_bits).collect();
                    let tok_cnt = d.u32("mm splice tokens")?;
                    let mh = d.u32("mm splice mh")?;
                    let mw = d.u32("mm splice mw")?;
                    let type_u8 = d.bytes("mm splice types")?;
                    let types: Vec<i8> = type_u8.into_iter().map(|b| b as i8).collect();
                    let src = d.u32("mm splice src")?;
                    let dst = d.u32("mm splice dst")?;
                    let n = d.u32("mm splice n")?;
                    splices.push((rows, tok_cnt, mh, mw, types, src, dst, n));
                }
                Dsv41Call::PrefillMm { slot, start, tokens, want, span_emb: (s0, s1, s2), splices }
            }
            K_TREE_PATH_WINDOW => Dsv41Call::TreePathWindow {
                tokens: d.u32s("tree path tokens")?,
                first_position: d.u64("tree path first_position")?,
                candidates: d.u32("tree path candidates")?,
            },
            K_TREE_COMMIT_WINDOW => Dsv41Call::TreeCommitWindow {
                idx: d.u32("tree commit idx")?,
                rows: d.u32("tree commit rows")?,
                window: d.u32s("tree commit window")?,
            },
            K_TREE_DROP_ROUND => Dsv41Call::TreeDropRound,
            K_PREFILL_EX => Dsv41Call::PrefillEx {
                slot: d.u32("ex slot")?,
                start: d.u64("ex start")?,
                tokens: d.u32s("ex tokens")?,
                want: d.u32("ex want")?,
                for_snapshot: d.bool("ex for_snapshot")?,
            },
            K_PREFIX_SNAPSHOT => Dsv41Call::PrefixSnapshot { slot: d.u32("snap slot")?, len: d.u64("snap len")? },
            K_PREFIX_RESTORE => {
                Dsv41Call::PrefixRestore { slot: d.u32("restore slot")?, len: d.u64("restore len")?, of: d.u64("restore of")? }
            }
            K_PREFIX_RELEASE => Dsv41Call::PrefixRelease { of: d.u64("release of")? },
            K_PREFIX_ENABLE => Dsv41Call::PrefixEnable { slots: d.u32("snapshot slots")? },
            _ => return Ok(None),
        }))
    }

    fn encode_facts(f: &Dsv41Facts, e: &mut lockstep::Enc) {
        e.u8(match f.kv {
            KvFmt::F16 => 0,
            KvFmt::Q4 => 1,
        });
        e.str(&f.moe_topp);
        e.bool(f.ced_replay);
        e.str(&f.engram_token_map);
        e.str(&f.engram_shards);
        e.str(&f.spec);
        e.str(&f.cache_request);
        e.u64(f.cache_slots);
    }

    fn decode_facts(d: &mut lockstep::Dec<'_>) -> Result<Dsv41Facts, lockstep::WireError> {
        let kv = match d.u8("kv format")? {
            0 => KvFmt::F16,
            1 => KvFmt::Q4,
            other => return Err(lockstep::WireError::Invalid(format!("the KV format byte {other} is neither fp16 nor q4"))),
        };
        Ok(Dsv41Facts {
            kv,
            moe_topp: d.str("MoE top-p setting")?,
            ced_replay: d.bool("CED replay setting")?,
            engram_token_map: d.str("engram token map setting")?,
            engram_shards: d.str("engram shards setting")?,
            spec: d.str("spec mode")?,
            cache_request: d.str("prefix cache request")?,
            cache_slots: d.u64("prefix cache snapshot slots")?,
        })
    }

    fn check_facts(leader: &Dsv41Facts, follower: &Dsv41Facts, who: u32) -> Vec<String> {
        [
            lockstep::fact_differs(
                "the KV cache format (--kv-dtype / REXL3_DSV41_KV; pass the same to every rank)",
                leader.kv.name(),
                follower.kv.name(),
                who,
            ),
            lockstep::fact_differs(
                "the MoE decode top-p pruning (REXL3_DSV41_MOE_TOPP; the golden-exact runs set 1.0 on every rank)",
                &leader.moe_topp,
                &follower.moe_topp,
                who,
            ),
            lockstep::fact_differs(
                "the CED bounded-replay prefill (REXL3_DSV41_CED_REPLAY)",
                leader.ced_replay,
                follower.ced_replay,
                who,
            ),
            lockstep::fact_differs(
                "the engram token map (REXL3_DSV41_ENGRAM_TOKEN_MAP)",
                &leader.engram_token_map,
                &follower.engram_token_map,
                who,
            ),
            lockstep::fact_differs(
                "the engram sidecar (REXL3_DSV41_ENGRAM_SHARDS)",
                &leader.engram_shards,
                &follower.engram_shards,
                who,
            ),
            lockstep::fact_differs("the speculative mode (--spec; a group runs none)", &leader.spec, &follower.spec, who),
            lockstep::fact_differs(
                "the prefix cache (--prefix-cache; pass the same to every rank)",
                &leader.cache_request,
                &follower.cache_request,
                who,
            ),
            // rank 0 sizes every rank's snapshot pool to its own count (`PrefixEnable`): a follower that cannot hold that
            // many would answer a snapshot with a skip while rank 0 took it, and the cache would stop hitting
            (follower.cache_slots < leader.cache_slots).then(|| {
                format!(
                    "prefix cache snapshot slots: rank 0 plans {}, rank {who} can hold only {} (REXL3_DSV41_PREFIX_SNAPSHOTS; \
                     pass the same to every rank)",
                    leader.cache_slots, follower.cache_slots
                )
            }),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

// ------------------------------------------------------------------------------------------------------------------- the prefill wire shape

/// The batch the family's prefill is on the wire: ONE sequence's chunk, exactly the shape `Dsv41Model::prefill_chunk`
/// builds inside itself, so the follower's reconstruction is the model's own call and no translation can move a bit.
pub fn prefill_batch(slot: usize, start: usize, tokens: &[u32]) -> RowBatch {
    RowBatch {
        rows: tokens.len(),
        ids: tokens.iter().map(|&t| t as i32).collect(),
        positions: (0..tokens.len()).map(|i| (start + i) as i32).collect(),
        row_slot: vec![slot as i32; tokens.len()],
        seq_start: vec![0, tokens.len() as i32],
        seq_slot: vec![slot as i32],
        rope3: None,
        parents: None,
    }
}

/// [`prefill_batch`]'s inverse, the follower's reconstruction of `(slot, start, tokens)`. This family prefills one
/// sequence's chunk a call, so a batch of more sequences is refused by name — never guessed, never split.
pub fn prefill_of(b: &RowBatch) -> Result<(usize, usize, Vec<u32>), ModelError> {
    if b.n_seq() != 1 || b.rows != b.ids.len() {
        return Err(ModelError::Request(format!(
            "tensor parallelism: a prefill chunk on the wire is ONE sequence's, not {} sequences of {} rows",
            b.n_seq(),
            b.rows
        )));
    }
    let slot = b.seq_slot[0] as usize;
    let start = b.positions.first().copied().unwrap_or(0) as usize;
    let tokens: Vec<u32> = b.ids.iter().map(|&t| t as u32).collect();
    Ok((slot, start, tokens))
}

// ------------------------------------------------------------------------------------------------------------------- the runtime surface

/// The model surface the backend drives and a lockstep group replays: the family's own signatures (the prefill of one
/// slot's chunk, the score readouts), implemented by [`Dsv41Model`] and by the leader [`TpLeader`] — the backend is
/// the same code on a single rank and on rank 0 of a group.
pub trait Dsv41Runtime: super::tree::TreeRoundHost {
    fn vocab_size(&self) -> usize;
    fn has_mtp(&self) -> bool;
    /// The most rows a prefill chunk can take (the backend's own chunking).
    fn chunk_rows(&self) -> usize;
    fn max_context(&self) -> usize {
        65_536
    }
    fn image_token_id(&self) -> u32 {
        129264
    }
    fn trees_capable(&self) -> bool {
        false
    }
    fn spec_learn_on(&self) -> bool {
        false
    }
    fn span_embeddings_tuple(&self) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        None
    }
    fn ensure_span_embeddings(&mut self, _emb: (Vec<f32>, Vec<f32>, Vec<f32>)) {}
    fn enable_prefix_cache(&mut self, _snapshot_slots: usize) {}
    /// Slots of the host snapshot pool (`enable_prefix_cache`'s count).
    fn prefix_pool_slots(&self) -> usize {
        0
    }
    fn prefix_free_snapshots(&self) -> usize {
        0
    }
    /// A content digest of `slot`'s restorable state at prefix length `len`, read from the live state (never from a
    /// stored snapshot): equal after a snapshot and after the restore of it exactly when the restore wrote back what
    /// was captured. Only a tensor-parallel group computes it (`Dsv41Model::prefix_state_digest`).
    fn prefix_state_digest(&mut self, _slot: usize, _len: usize) -> Result<u64, ModelError> {
        Err(ModelError::NotBuilt("a prefix state digest is not built for this runtime".into()))
    }
    fn can_snapshot_prefix(&self, _len: usize) -> bool {
        false
    }
    fn alloc_snapshot(&mut self) -> Option<u32> {
        None
    }
    fn release_prefix(&mut self, _snap: u32) {}
    fn prefix_snapshot(&mut self, _slot: usize, _snap: u32, _len: usize) -> Result<(), ModelError> {
        Err(ModelError::NotBuilt("prefix cache is not built for this runtime".into()))
    }
    fn restore_prefix(&mut self, _slot: usize, _snap: u32, _len: usize) -> Result<(), ModelError> {
        Err(ModelError::NotBuilt("prefix cache is not built for this runtime".into()))
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError>;
    fn end_slot(&mut self, slot: usize);
    fn prefill_chunk(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError>;
    fn prefill_chunk_ex(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
        _for_snapshot: bool,
    ) -> Result<Option<Candidates>, ModelError> {
        self.prefill_chunk(slot, start, tokens, want)
    }
    fn prefill_rows(&mut self, slot: usize, start: usize, tokens: &[u32], _head: super::model::Head) -> Result<(), ModelError> {
        self.prefill_chunk(slot, start, tokens, None).map(|_| ())
    }
    fn prefill_chunk_mm(
        &mut self,
        _slot: usize,
        _start: usize,
        _tokens: &[u32],
        _want: Option<usize>,
        _splices: Vec<super::vision_serve::Dsv41Splice>,
    ) -> Result<Option<Candidates>, ModelError> {
        Err(ModelError::NotBuilt("multimodal prefill is not built for this runtime".into()))
    }
    fn window_candidates(&mut self, b: &RowBatch, candidates: usize) -> Result<Candidates, ModelError>;
    fn commit(&mut self, slot: usize, keep: usize);
    fn flush(&mut self) -> Result<(), ModelError>;
    fn prefill_logits_last_row(&mut self, slot: usize, start: usize, tokens: &[u32]) -> Result<Vec<f32>, ModelError>;
    fn prefill_logits_rows(&mut self, slot: usize, start: usize, tokens: &[u32]) -> Result<Vec<Vec<f32>>, ModelError>;
    fn propose_mtp(&mut self, slot: usize, tokens: &[u32], max: usize) -> Result<Option<DraftRound>, ModelError>;
    fn propose_mtp_sampled(
        &mut self,
        slot: usize,
        tokens: &[u32],
        _sampling: &gb_sampling::Sampling,
        max: usize,
    ) -> Result<Option<DraftRound>, ModelError> {
        self.propose_mtp(slot, tokens, max)
    }
}

impl Dsv41Runtime for Dsv41Model {
    fn vocab_size(&self) -> usize {
        self.cfg.vocab
    }
    fn has_mtp(&self) -> bool {
        Dsv41Model::has_mtp(self)
    }
    fn chunk_rows(&self) -> usize {
        Dsv41Model::chunk_rows(self)
    }
    fn max_context(&self) -> usize {
        self.max_context
    }
    fn image_token_id(&self) -> u32 {
        self.cfg.image_token_id
    }
    fn trees_capable(&self) -> bool {
        Dsv41Model::trees_capable(self)
    }
    fn spec_learn_on(&self) -> bool {
        Dsv41Model::spec_learn_on(self)
    }
    fn span_embeddings_tuple(&self) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        self.span_embeddings().map(|e| (e.start.clone(), e.end.clone(), e.newline.clone()))
    }
    fn ensure_span_embeddings(&mut self, emb: (Vec<f32>, Vec<f32>, Vec<f32>)) {
        if self.span_embeddings().is_none() && !emb.0.is_empty() {
            self.set_span_embeddings(super::vision_serve::SpanEmbeddings { start: emb.0, end: emb.1, newline: emb.2 });
        }
    }
    fn enable_prefix_cache(&mut self, snapshot_slots: usize) {
        Dsv41Model::enable_prefix_cache(self, snapshot_slots);
    }
    fn prefix_pool_slots(&self) -> usize {
        Dsv41Model::prefix_pool_slots(self)
    }
    fn prefix_free_snapshots(&self) -> usize {
        Dsv41Model::prefix_free_snapshots(self)
    }
    fn prefix_state_digest(&mut self, slot: usize, len: usize) -> Result<u64, ModelError> {
        Dsv41Model::prefix_state_digest(self, slot, len)
    }
    fn can_snapshot_prefix(&self, len: usize) -> bool {
        Dsv41Model::can_snapshot_prefix(self, len)
    }
    fn alloc_snapshot(&mut self) -> Option<u32> {
        Dsv41Model::alloc_snapshot(self)
    }
    fn release_prefix(&mut self, snap: u32) {
        Dsv41Model::release_prefix(self, snap);
    }
    fn prefix_snapshot(&mut self, slot: usize, snap: u32, len: usize) -> Result<(), ModelError> {
        Dsv41Model::prefix_snapshot(self, slot, snap, len)
    }
    fn restore_prefix(&mut self, slot: usize, snap: u32, len: usize) -> Result<(), ModelError> {
        Dsv41Model::restore_prefix(self, slot, snap, len)
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        Dsv41Model::begin_slot(self, slot)
    }
    fn end_slot(&mut self, slot: usize) {
        Dsv41Model::end_slot(self, slot);
    }
    fn prefill_chunk(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError> {
        Dsv41Model::prefill_chunk(self, slot, start, tokens, want)
    }
    fn prefill_chunk_ex(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
        for_snapshot: bool,
    ) -> Result<Option<Candidates>, ModelError> {
        Dsv41Model::prefill_chunk_ex(self, slot, start, tokens, want, for_snapshot)
    }
    fn prefill_rows(&mut self, slot: usize, start: usize, tokens: &[u32], head: super::model::Head) -> Result<(), ModelError> {
        Dsv41Model::prefill_rows(self, slot, start, tokens, head)
    }
    fn prefill_chunk_mm(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
        splices: Vec<super::vision_serve::Dsv41Splice>,
    ) -> Result<Option<Candidates>, ModelError> {
        Dsv41Model::prefill_chunk_mm(self, slot, start, tokens, want, splices)
    }
    fn window_candidates(&mut self, b: &RowBatch, candidates: usize) -> Result<Candidates, ModelError> {
        Dsv41Model::window_candidates(self, b, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        Dsv41Model::commit(self, slot, keep);
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        Dsv41Model::flush(self)
    }
    fn prefill_logits_last_row(&mut self, slot: usize, start: usize, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        Dsv41Model::prefill_logits_last_row(self, slot, start, tokens)
    }
    fn prefill_logits_rows(&mut self, slot: usize, start: usize, tokens: &[u32]) -> Result<Vec<Vec<f32>>, ModelError> {
        Dsv41Model::prefill_logits_rows(self, slot, start, tokens)
    }
    fn propose_mtp(&mut self, slot: usize, tokens: &[u32], max: usize) -> Result<Option<DraftRound>, ModelError> {
        Dsv41Model::propose_mtp(self, slot, tokens, max)
    }
    fn propose_mtp_sampled(
        &mut self,
        slot: usize,
        tokens: &[u32],
        sampling: &gb_sampling::Sampling,
        max: usize,
    ) -> Result<Option<DraftRound>, ModelError> {
        Dsv41Model::propose_mtp_sampled(self, slot, tokens, sampling, max)
    }
}

fn row_outputs_digest(rows: &[gb_engine::RowOutput]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for r in rows {
        for &id in &r.ids {
            h = (h ^ id as u64).wrapping_mul(0x100_0000_01b3);
        }
        for &v in &r.values {
            h = (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// A fold over f32 bits: the digest a score call answers with. One bit of difference is one bit too many — the
/// group's exactness contract is byte identity, never a tolerance.
fn logits_digest(rows: &[f32]) -> u64 {
    rows.iter().map(|v| v.to_bits() as u64).fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b).wrapping_mul(0x100_0000_01b3))
}

/// All the rows' logits, folded row by row (the last row's digest follows, so a truncated readout differs too).
fn logits_rows_digest(rows: &[Vec<f32>]) -> u64 {
    rows.iter().map(|r| logits_digest(r)).fold(0xcbf2_9ce4_8422_2325u64, |h, d| (h ^ d).wrapping_mul(0x100_0000_01b3))
}

/// Any [`Dsv41Runtime`] as a [`lockstep::LockstepRuntime`]: the core calls go to the model (a prefill through the
/// family's own single-sequence shape), the family's calls are applied here. Holds what only this family has of the
/// protocol: the prefix snapshots' handles, by the sequence number of the call that took each.
pub struct Dsv41Rt<R: Dsv41Runtime> {
    inner: R,
    /// A follower: its own pool's id of each snapshot, by the sequence number of the `PrefixSnapshot` call that took it.
    follower_snaps: BTreeMap<u64, u32>,
    /// Rank 0: the sequence number of the call that took each of its snapshots (by its own pool id), kept from the call
    /// until the release, whatever the call's outcome, so the release reaches every rank.
    leader_snaps: BTreeMap<u32, u64>,
    /// Rank 0: every rank's content digest of each snapshot every rank holds.
    book: lockstep::SnapshotBook,
}

impl<R: Dsv41Runtime> Dsv41Rt<R> {
    pub fn new(inner: R) -> Dsv41Rt<R> {
        Dsv41Rt { inner, follower_snaps: BTreeMap::new(), leader_snaps: BTreeMap::new(), book: lockstep::SnapshotBook::default() }
    }

    /// Snapshots this follower holds for rank 0 (by the call that took each).
    pub fn follower_snapshots(&self) -> usize {
        self.follower_snaps.len()
    }

    /// Snapshots every rank holds, as rank 0's book has them.
    pub fn group_snapshots(&self) -> usize {
        self.book.len()
    }

    /// A follower's half of a `PrefixSnapshot` (call `seq`): the same memory guard rank 0 ran before it forwarded the
    /// call (a full-pool or low-memory rank SKIPS the capture and answers so, never allocates past the guard), then the
    /// capture into a slot of its own pool and the digest of what it captured.
    fn follower_snapshot(&mut self, seq: u64, slot: usize, len: usize) -> Result<u64, ModelError> {
        if !self.inner.can_snapshot_prefix(len) {
            eprintln!(
                "rexl3 level=warn event=tp_prefix_snapshot_skipped call={seq} slot={slot} len={len} free_snapshots={} (the memory \
                 guard or a full pool: rank 0 releases the entry on every rank)",
                self.inner.prefix_free_snapshots()
            );
            return Ok(lockstep::SNAPSHOT_SKIPPED);
        }
        let Some(snap) = self.inner.alloc_snapshot() else { return Ok(lockstep::SNAPSHOT_SKIPPED) };
        match self.inner.prefix_snapshot(slot, snap, len).and_then(|_| self.inner.prefix_state_digest(slot, len)) {
            Ok(d) => {
                self.follower_snaps.insert(seq, snap);
                Ok(lockstep::taken_digest(d))
            }
            Err(e) => {
                self.inner.release_prefix(snap);
                Err(e)
            }
        }
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

impl<R: Dsv41Runtime> lockstep::LockstepRuntime for Dsv41Rt<R> {
    type Call = Dsv41Call;

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn has_mtp(&self) -> bool {
        self.inner.has_mtp()
    }

    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.inner.begin_slot(slot)
    }

    fn end_slot(&mut self, slot: usize) {
        self.inner.end_slot(slot)
    }

    fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        let (slot, start, tokens) = prefill_of(b)?;
        self.inner.prefill_chunk(slot, start, &tokens, want)
    }

    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        self.inner.window_candidates(b, count)
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        self.inner.commit(slot, keep);
    }

    fn flush(&mut self) -> Result<(), ModelError> {
        self.inner.flush()
    }

    fn apply_call(&mut self, seq: u64, call: Dsv41Call) -> Result<lockstep::CallReply, ModelError> {
        match call {
            Dsv41Call::ScoreLastRow { slot, start, tokens } => {
                let v = self.inner.prefill_logits_last_row(slot as usize, start as usize, &tokens)?;
                Ok(lockstep::CallReply::digest(logits_digest(&v)))
            }
            Dsv41Call::ScoreRows { slot, start, tokens } => {
                let v = self.inner.prefill_logits_rows(slot as usize, start as usize, &tokens)?;
                Ok(lockstep::CallReply::digest(logits_rows_digest(&v)))
            }
            Dsv41Call::PrefillMm { slot, start, tokens, want, span_emb, splices } => {
                self.inner.ensure_span_embeddings(span_emb);
                let mut sp_vec = Vec::with_capacity(splices.len());
                for (rows, tok_cnt, mh, mw, types, src, dst, n) in splices {
                    let feats = std::sync::Arc::new(super::vision_serve::Dsv41ImageFeatures {
                        rows,
                        tokens: tok_cnt as usize,
                        merged: (mh as usize, mw as usize),
                        key: 0,
                        permit: None,
                    });
                    sp_vec.push(super::vision_serve::Dsv41Splice {
                        feats,
                        types,
                        src: src as usize,
                        dst: dst as usize,
                        n: n as usize,
                    });
                }
                let want_opt = (want > 0).then_some(want as usize);
                let c = self.inner.prefill_chunk_mm(slot as usize, start as usize, &tokens, want_opt, sp_vec)?;
                let d = match &c {
                    Some(c) => logits_digest(&c.vals),
                    None => 0,
                };
                Ok(lockstep::CallReply::digest(d))
            }
            Dsv41Call::TreePathWindow { tokens, first_position, candidates } => {
                let v = self.inner.tree_path_window(&tokens, first_position, candidates as usize).map_err(ModelError::Request)?;
                Ok(lockstep::CallReply::digest(row_outputs_digest(&v)))
            }
            Dsv41Call::TreeCommitWindow { idx, rows, window } => {
                self.inner.tree_commit_window(idx as usize, rows as usize, &window).map_err(ModelError::Request)?;
                Ok(lockstep::CallReply::NONE)
            }
            Dsv41Call::TreeDropRound => {
                self.inner.tree_drop_round().map_err(ModelError::Request)?;
                Ok(lockstep::CallReply::NONE)
            }
            Dsv41Call::PrefillEx { slot, start, tokens, want, for_snapshot } => {
                let want_opt = (want > 0).then_some(want as usize);
                let c = self.inner.prefill_chunk_ex(slot as usize, start as usize, &tokens, want_opt, for_snapshot)?;
                let d = match &c {
                    Some(c) => logits_digest(&c.vals),
                    None => 0,
                };
                Ok(lockstep::CallReply::digest(d))
            }
            Dsv41Call::PrefixSnapshot { slot, len } => {
                self.follower_snapshot(seq, slot as usize, len as usize).map(lockstep::CallReply::digest)
            }
            Dsv41Call::PrefixRestore { slot, len, of } => {
                let snap = *self.follower_snaps.get(&of).ok_or_else(|| {
                    ModelError::Request(format!("restore of the snapshot taken by call {of}: this rank holds no such snapshot"))
                })?;
                self.inner.restore_prefix(slot as usize, snap, len as usize)?;
                let d = self.inner.prefix_state_digest(slot as usize, len as usize)?;
                Ok(lockstep::CallReply::digest(lockstep::taken_digest(d)))
            }
            Dsv41Call::PrefixRelease { of } => {
                // a snapshot this rank skipped (or never took) is not held: nothing to release, and not an error
                if let Some(snap) = self.follower_snaps.remove(&of) {
                    self.inner.release_prefix(snap);
                }
                Ok(lockstep::CallReply::NONE)
            }
            Dsv41Call::PrefixEnable { slots } => {
                self.follower_snaps.clear();
                self.inner.enable_prefix_cache(slots as usize);
                Ok(lockstep::CallReply::digest(self.inner.prefix_pool_slots() as u64))
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------- the leader

impl<R: Dsv41Runtime> TpLeader<R> {
    /// Starts the leader over `link`, the one follower of a two-rank group (the boot handshake is done:
    /// [`boot_leader_group`]).
    pub fn start(inner: R, link: Link, timeouts: TpTimeouts) -> TpLeader<R> {
        lockstep::TpLeader::lead_group(Dsv41Rt::new(inner), vec![link], timeouts)
    }
}

impl<R: Dsv41Runtime> super::tree::TreeRoundHost for TpLeader<R> {
    fn tree_path_window(
        &mut self,
        tokens: &[u32],
        first_position: u64,
        candidates: usize,
    ) -> Result<Vec<gb_engine::RowOutput>, String> {
        self.guard().map_err(|e| e.to_string())?;
        let call = Dsv41Call::TreePathWindow { tokens: tokens.to_vec(), first_position, candidates: candidates as u32 };
        self.forward_call(
            call,
            |m, _| m.inner.tree_path_window(tokens, first_position, candidates).map_err(ModelError::Request),
            |v| row_outputs_digest(v),
        )
        .map_err(|e| e.to_string())
    }

    fn tree_commit_window(&mut self, idx: usize, rows: usize, window: &[u32]) -> Result<(), String> {
        self.guard().map_err(|e| e.to_string())?;
        let call = Dsv41Call::TreeCommitWindow { idx: idx as u32, rows: rows as u32, window: window.to_vec() };
        self.forward_call(call, |m, _| m.inner.tree_commit_window(idx, rows, window).map_err(ModelError::Request), |_| 0)
            .map_err(|e| e.to_string())
    }

    fn tree_drop_round(&mut self) -> Result<(), String> {
        self.guard().map_err(|e| e.to_string())?;
        self.forward_call(Dsv41Call::TreeDropRound, |m, _| m.inner.tree_drop_round().map_err(ModelError::Request), |_| 0)
            .map_err(|e| e.to_string())
    }
}

impl<R: Dsv41Runtime> Dsv41Runtime for TpLeader<R> {
    fn vocab_size(&self) -> usize {
        self.inner().model().vocab_size()
    }
    fn has_mtp(&self) -> bool {
        self.inner().model().has_mtp()
    }
    fn chunk_rows(&self) -> usize {
        self.inner().model().chunk_rows()
    }
    fn max_context(&self) -> usize {
        self.inner().model().max_context()
    }
    fn image_token_id(&self) -> u32 {
        self.inner().model().image_token_id()
    }
    fn trees_capable(&self) -> bool {
        self.inner().model().trees_capable()
    }
    fn spec_learn_on(&self) -> bool {
        self.inner().model().spec_learn_on()
    }
    /// Sizes EVERY rank's snapshot pool to `snapshot_slots` (rank 0's count) and waits for every follower at the boot:
    /// a follower whose pool differs fails the group by name before anything is served (its answer is its pool's size).
    fn enable_prefix_cache(&mut self, snapshot_slots: usize) {
        let enabled = self.forward_call(
            Dsv41Call::PrefixEnable { slots: snapshot_slots as u32 },
            |m, _| {
                m.leader_snaps.clear();
                m.book = lockstep::SnapshotBook::default();
                m.inner.enable_prefix_cache(snapshot_slots);
                Ok(m.inner.prefix_pool_slots())
            },
            |n| *n as u64,
        );
        let settled = enabled.and_then(|_| self.settle(self.last_seq()));
        match settled {
            Ok(()) => eprintln!("rexl3 event=tp_prefix_pool slots={snapshot_slots} ranks={} (every rank's pool)", self.ranks()),
            Err(e) => {
                // the group is lost (latched): the first request fails by name and the process ends
                eprintln!("rexl3 level=error event=tp_prefix_pool_failed slots={snapshot_slots} error=\"{e}\"");
                self.fail(format!("the snapshot pools could not be sized on every rank: {e}"));
            }
        }
    }
    fn prefix_pool_slots(&self) -> usize {
        self.inner().model().prefix_pool_slots()
    }
    fn prefix_free_snapshots(&self) -> usize {
        self.inner().model().prefix_free_snapshots()
    }
    fn can_snapshot_prefix(&self, len: usize) -> bool {
        self.inner().model().can_snapshot_prefix(len)
    }
    fn alloc_snapshot(&mut self) -> Option<u32> {
        self.inner_mut().model_mut().alloc_snapshot()
    }
    /// Frees rank 0's snapshot `snap` on every rank: the release names the call that took it, and a rank that does not
    /// hold it answers Ok (a skipped capture). Not waited for: nothing depends on it, and a follower that never answers
    /// is a stall ([`lockstep::CacheDeadlines`]). A snapshot rank 0 never forwarded is freed here only.
    fn release_prefix(&mut self, snap: u32) {
        let Some(of) = self.inner_mut().leader_snaps.remove(&snap) else {
            self.inner_mut().model_mut().release_prefix(snap);
            return;
        };
        self.inner_mut().book.forget(of);
        let released = self.forward_cache_call(
            Dsv41Call::PrefixRelease { of },
            |m, _| {
                m.inner.release_prefix(snap);
                Ok(())
            },
            false,
        );
        if let Err(e) = released {
            // a lost group is latched and fails the next call; free rank 0's slot all the same
            self.inner_mut().model_mut().release_prefix(snap);
            eprintln!("rexl3 level=warn event=tp_prefix_release_failed snap={snap} call={of} error=\"{e}\"");
        }
    }
    /// Takes a snapshot of `slot` at `len` on EVERY rank (rank 0's into its pool slot `snap`, each follower's into a slot
    /// of its own) and waits for every rank's answer: `Ok` only when every rank took it, with every rank's content
    /// digest recorded for the restore. Otherwise the snapshot is refused (`Err`): the caller releases it, which frees
    /// it on every rank that took it, and the engine never sees it.
    fn prefix_snapshot(&mut self, slot: usize, snap: u32, len: usize) -> Result<(), ModelError> {
        self.guard()?;
        let out = self.forward_cache_call(
            Dsv41Call::PrefixSnapshot { slot: slot as u32, len: len as u64 },
            |m, _| {
                m.inner.prefix_snapshot(slot, snap, len)?;
                m.inner.prefix_state_digest(slot, len).map(lockstep::taken_digest)
            },
            true,
        )?;
        let seq = out.seq;
        // the handle every rank knows, kept until the release whatever the outcome (a follower may have taken it)
        self.inner_mut().leader_snaps.insert(snap, seq);
        let rank0 = match &out.local {
            Ok(d) => *d,
            Err(e) => {
                eprintln!(
                    "rexl3 level=warn event=tp_prefix_snapshot_failed rank=0 call={seq} slot={slot} len={len} error=\"{e}\""
                );
                lockstep::SNAPSHOT_SKIPPED
            }
        };
        match self.inner_mut().book.settle_snapshot(seq, rank0, &out.followers) {
            Ok(()) => Ok(()),
            Err(why) => {
                eprintln!("rexl3 level=warn event=tp_prefix_snapshot_refused call={seq} slot={slot} len={len} reason=\"{why}\"");
                Err(ModelError::Request(format!("the prefix snapshot of call {seq} is not held by every rank: {why}")))
            }
        }
    }
    /// Restores rank 0's snapshot `snap` into `slot` on EVERY rank and waits for every rank's answer before the suffix
    /// can be dispatched: `Ok` only when every rank restored exactly the state it snapshotted (its digest of the
    /// restored state equals its own digest at snapshot time). Otherwise (an error, no answer within the deadline, a
    /// stale or empty restore that answered Ok) the restore is refused with a request error: the engine drops the
    /// entry and prefills the prompt cold, which begins the slot again on every rank. A snapshot not held by every
    /// rank is refused before anything is touched or forwarded.
    fn restore_prefix(&mut self, slot: usize, snap: u32, len: usize) -> Result<(), ModelError> {
        self.guard()?;
        let of = match self.inner().leader_snaps.get(&snap) {
            Some(&of) if self.inner().book.holds(of) => of,
            Some(&of) => {
                return Err(ModelError::Request(format!(
                    "the prefix snapshot of call {of} is not held by every rank (never restored)"
                )))
            }
            None => {
                return Err(ModelError::Request(format!(
                    "snapshot {snap} is not one rank 0 forwarded to the followers (never restored)"
                )))
            }
        };
        let out = self.forward_cache_call(
            Dsv41Call::PrefixRestore { slot: slot as u32, len: len as u64, of },
            |m, _| {
                m.inner.restore_prefix(slot, snap, len)?;
                m.inner.prefix_state_digest(slot, len).map(lockstep::taken_digest)
            },
            true,
        )?;
        let rank0 = out.local.map_err(|e| e.to_string());
        self.inner().book.verify_restore(of, rank0, &out.followers).map_err(|why| {
            eprintln!(
                "rexl3 level=warn event=tp_prefix_restore_refused call={} of={of} slot={slot} len={len} reason=\"{why}\"",
                out.seq
            );
            ModelError::Request(format!(
                "the prefix restore of call {} (snapshot call {of}) was not verified on every rank: {why}",
                out.seq
            ))
        })
    }

    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        lockstep::TpLeader::begin_slot(self, slot)
    }
    fn end_slot(&mut self, slot: usize) {
        lockstep::TpLeader::end_slot(self, slot);
    }

    fn prefill_chunk(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError> {
        let b = prefill_batch(slot, start, tokens);
        lockstep::TpLeader::prefill_chunk(self, &b, want)
    }

    fn prefill_chunk_ex(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
        for_snapshot: bool,
    ) -> Result<Option<Candidates>, ModelError> {
        if !for_snapshot {
            return Dsv41Runtime::prefill_chunk(self, slot, start, tokens, want);
        }
        self.guard()?;
        let call = Dsv41Call::PrefillEx {
            slot: slot as u32,
            start: start as u64,
            tokens: tokens.to_vec(),
            want: want.unwrap_or(0) as u32,
            for_snapshot,
        };
        self.forward_call(
            call,
            |m, _| m.inner.prefill_chunk_ex(slot, start, tokens, want, for_snapshot),
            |c| match c {
                Some(c) => logits_digest(&c.vals),
                None => 0,
            },
        )
    }

    fn prefill_chunk_mm(
        &mut self,
        slot: usize,
        start: usize,
        tokens: &[u32],
        want: Option<usize>,
        splices: Vec<super::vision_serve::Dsv41Splice>,
    ) -> Result<Option<Candidates>, ModelError> {
        self.guard()?;
        let mut wire_splices = Vec::with_capacity(splices.len());
        for sp in &splices {
            wire_splices.push((
                sp.feats.rows.clone(),
                sp.feats.tokens as u32,
                sp.feats.merged.0 as u32,
                sp.feats.merged.1 as u32,
                sp.types.clone(),
                sp.src as u32,
                sp.dst as u32,
                sp.n as u32,
            ));
        }
        let span_emb = self.inner().model().span_embeddings_tuple().unwrap_or_default();
        let call = Dsv41Call::PrefillMm {
            slot: slot as u32,
            start: start as u64,
            tokens: tokens.to_vec(),
            want: want.unwrap_or(0) as u32,
            span_emb,
            splices: wire_splices,
        };
        self.forward_call(
            call,
            |m, _| m.inner.prefill_chunk_mm(slot, start, tokens, want, splices),
            |c| match c {
                Some(c) => logits_digest(&c.vals),
                None => 0,
            },
        )
    }

    fn window_candidates(&mut self, b: &RowBatch, candidates: usize) -> Result<Candidates, ModelError> {
        lockstep::TpLeader::window_candidates(self, b, candidates)
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        lockstep::TpLeader::commit(self, slot, keep);
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        lockstep::TpLeader::flush(self)
    }

    fn prefill_logits_last_row(&mut self, slot: usize, start: usize, tokens: &[u32]) -> Result<Vec<f32>, ModelError> {
        self.guard()?;
        let call = Dsv41Call::ScoreLastRow { slot: slot as u32, start: start as u64, tokens: tokens.to_vec() };
        self.forward_call(call, |m, _| m.inner.prefill_logits_last_row(slot, start, tokens), |v| logits_digest(v))
    }

    fn prefill_logits_rows(&mut self, slot: usize, start: usize, tokens: &[u32]) -> Result<Vec<Vec<f32>>, ModelError> {
        self.guard()?;
        let call = Dsv41Call::ScoreRows { slot: slot as u32, start: start as u64, tokens: tokens.to_vec() };
        self.forward_call(call, |m, _| m.inner.prefill_logits_rows(slot, start, tokens), |v| logits_rows_digest(v))
    }

    /// The DSpark draft round is refused BY NAME, never forwarded: a group boot refuses the drafting
    /// (`check_tp_plain`), and this is the seam a late `--spec` flip would hit — a draft round on one rank would
    /// leave the other inside a combine, which is a desync, not an error.
    fn propose_mtp(&mut self, slot: usize, tokens: &[u32], max: usize) -> Result<Option<DraftRound>, ModelError> {
        // the plain entry stays the argmax walk: a sampled caller passes its Sampling down `propose_mtp_sampled`
        let greedy = gb_sampling::Sampling { temperature: 0.0, ..gb_sampling::Sampling::default() };
        self.propose_mtp_sampled(slot, tokens, &greedy, max)
    }

    fn propose_mtp_sampled(
        &mut self,
        slot: usize,
        tokens: &[u32],
        sampling: &gb_sampling::Sampling,
        max: usize,
    ) -> Result<Option<DraftRound>, ModelError> {
        if !self.inner().model().has_mtp() {
            return Err(ModelError::NotBuilt(
                "the DSpark drafting of the NextN layers is not built for tensor parallelism (their experts are split across \
                 the ranks); boot the group with --spec none"
                    .into(),
            ));
        }
        lockstep::TpLeader::flush(self)?;
        self.inner_mut().model_mut().propose_mtp_sampled(slot, tokens, sampling, max)
    }
}

// ------------------------------------------------------------------------------------------------------------------- the follower

/// The follower's knobs and the handles its caller keeps (the neutral [`lockstep::FollowerConfig`] at this family's
/// call type).
pub struct FollowerConfig {
    pub timeouts: TpTimeouts,
    /// Set to ask the loop to return after the command in hand.
    pub stop: Arc<std::sync::atomic::AtomicBool>,
    /// Told how many commands were applied, now and then (the stderr health line).
    pub on_progress: Option<lockstep::ProgressFn>,
    /// Run when the leader is lost while a call is stuck (the communicator's abort).
    pub abort_hook: Option<lockstep::AbortHook>,
    /// This process's rank in the group.
    pub rank: usize,
    /// The communicator's health probe (`ncclCommGetAsyncError`), polled by the follower's watchdog.
    pub health: Option<lockstep::HealthProbe>,
}

impl FollowerConfig {
    pub fn new(timeouts: TpTimeouts) -> Self {
        let g = lockstep::FollowerConfig::new(timeouts);
        FollowerConfig {
            timeouts: g.timeouts,
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

/// The follower's loop: applies the leader's commands, in order, to `rt` and acknowledges each. Returns on Shutdown,
/// on EOF or when the peer goes silent. A fatal local error is answered (so the leader fails by name) and returned.
pub fn run_follower<R: Dsv41Runtime>(rt: R, link: Link, cfg: FollowerConfig) -> Result<FollowerExit, ModelError> {
    let FollowerConfig { timeouts, stop, on_progress, abort_hook, rank, health } = cfg;
    let mut generic = lockstep::FollowerConfig::new(timeouts).with_rank(rank);
    generic.stop = stop;
    generic.on_progress = on_progress;
    generic.abort_hook = abort_hook;
    generic.health = health;
    lockstep::run_follower(Dsv41Rt::new(rt), link, generic)
}

// ------------------------------------------------------------------------------------------------------------------- boot

/// Rank 0's boot after every follower is connected, identified and gated: [`lockstep::boot_leader_group`] for this
/// family. `load` builds the rank's model from the communicator (its shard, its combine arms).
pub fn boot_leader_group<C, L>(
    links: Vec<Link>,
    local: &BootGate,
    t: &TpTimeouts,
    make_comm: impl FnOnce(&mut Rendezvous<'_>) -> Result<C, ModelError>,
    load: L,
) -> Result<TpLeader<Dsv41Model>, ModelError>
where
    L: FnOnce(C) -> Result<Dsv41Model, ModelError>,
{
    lockstep::boot_leader_group(links, local, t, make_comm, |c| load(c).map(Dsv41Rt::new))
}

/// A follower's boot after the connection is up: [`lockstep::boot_follower`] for this family. `load` gets the gate
/// (whose [`BootGate::model_options`] is rank 0's planned geometry — the follower is built with exactly those) and the
/// communicator. The model it returns goes to [`run_follower`] with the same link.
pub fn boot_follower<C, L>(
    link: Link,
    local: &BootGate,
    t: &TpTimeouts,
    make_comm: impl FnOnce(&mut Rendezvous<'_>) -> Result<C, ModelError>,
    load: L,
) -> Result<(Dsv41Model, Link), ModelError>
where
    L: FnOnce(&BootGate, C) -> Result<Dsv41Model, ModelError>,
{
    lockstep::boot_follower(link, local, t, make_comm, |gate: &BootGate, c| load(gate, c).map(Dsv41Rt::new))
        .map(|(rt, link)| (rt.into_model(), link))
}

// ------------------------------------------------------------------------------------------------------------------- the gate

/// A short identity of the pack in `dir` for the boot gate: the digest of its `config.json` and its safetensors
/// index — the same header-level shape the nemotron gate uses (`prefix::pack_identity`), domain-tagged for this
/// family.
pub fn pack_identity(dir: &std::path::Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"rexl3-dsv41-pack-v1\0");
    for name in ["config.json", "model.safetensors.index.json"] {
        let path = dir.join(name);
        let bytes = std::fs::read(&path).map_err(|e| format!("pack identity: {}: {e}", path.display()))?;
        h.update(name.as_bytes());
        h.update([0]);
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
    }
    let d = h.finalize();
    Ok(d.iter().take(16).map(|b| format!("{b:02x}")).collect())
}

/// This rank's side of the gate: the geometry the rank planned (its pack's identity, the layers it loads, the slots,
/// KV pages, context and prefill rows of its admission) and the family's facts. `kv_pages` are the model's
/// `kv_blocks`; a follower that planned fewer than rank 0 is refused by the gate, one that planned more is built with
/// exactly rank 0's ([`BootGate::model_options`]).
pub fn gate_of(
    pack: String,
    layers: usize,
    slots: usize,
    kv_pages: usize,
    max_context: usize,
    prefill_rows: usize,
    family: Dsv41Facts,
) -> BootGate {
    lockstep::BootGate::new(pack, layers as u32, slots as u32, kv_pages as u64, max_context as u64, prefill_rows as u32, family)
}

impl BootGate {
    /// The model options this gate describes — rank 0's planned geometry, what the follower loads with once the gate
    /// has passed (the cache format comes from the facts, which the gate has already compared).
    pub fn model_options(&self) -> ModelOptions {
        ModelOptions {
            slots: self.slots as usize,
            kv_blocks: self.kv_pages as usize,
            max_context: self.max_context as usize,
            chunk_rows: self.prefill_rows as usize,
            layers: self.layers as usize,
            kv: self.family.kv,
        }
    }
}
