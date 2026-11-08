//! The tensor-parallel boot of the Nemotron family: rank 0's runtime (a [`TpLeader`] around its shard, which `engine::build_with` then
//! builds the ordinary engine around) and the follower process of every other rank (any rank above 0 of a group of any size). The
//! protocol, the boot handshake and the failure model are `tp_runtime`; this file is the family's glue and the two seams the sibling
//! branches fill in.
//!
//! **The two seams, filled.** [`make_comm`] creates the collective (TP-N's `NcclTp`, bootstrapped over the same connection: rank 0 offers
//! the unique id through the [`Rendezvous`] to every follower, each receives it; [`abort_group_comm`] is wired to the runtime's group-lost hooks) and
//! [`load_rank`] loads this rank's shard with it (TP-M's `NemotronModel::load_tp`). `REXL3_TP_COMM=none` boots a group with NO communicator
//! and every rank loads the WHOLE model: the two processes then run the identical model in lockstep, with no collectives and no speedup
//! (logged as `sharded=false`); the memory plan follows ([`sharded`]).
//!
//! **Boot order** (rank 0, in `engine::build_with`'s `load` closure; the followers in [`follow`]): every rank plans memory from its
//! own free memory; each follower connects (rank 0 has been listening since the start of its boot) and rank 0 accepts them all under
//! one deadline; the Hellos are exchanged, each follower is identified by its rank and gated (`BootGate::check`); the communicator is
//! created (every follower acknowledges the bootstrap before anyone joins); each loads its shard (a follower with RANK 0'S planned
//! options: it may hold more, never less); each follower sends Ready; rank 0 compares the loaded vocabulary and head with each and
//! starts the leader.

use super::backend::ModelOptions;
use super::config::OmniConfig;
use super::engine::{self, PlanInputs};
use super::inventory::Inventory;
use super::model::NemotronModel;
use super::prefix;
use super::tp::{TpComm, TpTopology};
use super::tp_comm::NcclTp;
use super::tp_runtime::{self, BootGate, FollowerConfig, FollowerExit, Rendezvous, TpLeader, TpTimeouts};
use super::weights::{self, LayerPlan};
use crate::error::ModelError;
use crate::family::{BuildCtx, FamilyPlan};
use crate::hostmem::MemInfo;
use crate::serve::TpOptions;
use gb_cuda::nccl::{NcclComm, NcclUniqueId};
use gb_engine::boot;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The timeouts of a group from its options (the rest are the protocol's defaults).
pub fn timeouts(tp: &TpOptions) -> TpTimeouts {
    TpTimeouts {
        connect: tp.connect_timeout,
        ready: tp.ready_timeout,
        peer: tp.peer_timeout,
        write: tp.peer_timeout,
        ..TpTimeouts::default()
    }
}

/// This process's place in the group.
pub fn topology(tp: &TpOptions) -> TpTopology {
    TpTopology { rank: tp.rank, ranks: tp.ranks }
}

/// The environment variable that boots a group WITHOUT a collective (`none`): every rank loads the whole model and runs the identical
/// model in lockstep, no speedup. For bring-up on a fabric that is not ready; the default is NCCL.
pub const COMM_ENV: &str = "REXL3_TP_COMM";

/// Exports `NCCL_MAX_NCHANNELS=4` when it is not set: NCCL's own tuning is 3 to 10 times slower on the two-node fabric
/// (`docs/NEMOTRON_TP.md` section 10.3) and a group whose ranks run different channel counts runs the slower schedule on all of them.
/// Idempotent, and a group booted without a collective has no NCCL to configure. Called before the boot gate is made (the gate
/// compares the effective value: [`tp_runtime::nccl_fingerprint`]) and again before the communicator is joined, on every rank, so a
/// rank started without the variable runs what the others run.
pub fn default_nccl_channels() {
    if sharded() && std::env::var_os("NCCL_MAX_NCHANNELS").is_none() {
        std::env::set_var("NCCL_MAX_NCHANNELS", tp_runtime::DEFAULT_NCCL_CHANNELS);
        eprintln!(
            "rexl3 event=tp_nccl_default NCCL_MAX_NCHANNELS={} (it was not set: NCCL's own tuning is 3 to 10 times slower on this fabric)",
            tp_runtime::DEFAULT_NCCL_CHANNELS
        );
    }
}

/// Whether the group's ranks hold SHARDS (the default) and not whole models (`REXL3_TP_COMM=none`): the memory plan prices what the rank
/// really holds, so it must follow.
pub fn sharded() -> bool {
    !std::env::var(COMM_ENV).is_ok_and(|v| v == "none")
}

/// The group's communicator, kept so the lockstep runtime's watchdog can abort it from its own thread ([`abort_group_comm`]).
static GROUP_COMM: Mutex<Option<Arc<dyn TpComm>>> = Mutex::new(None);

/// How long [`abort_group_comm`] waits for `ncclCommAbort`: it has been seen to block behind pending graph launches
/// (`docs/NEMOTRON_TP.md` 12.3 B2). After this the caller goes on (its exit path, then the process's `exit_backstop` `_exit(70)`), and
/// the abort is left running on its own thread.
pub const ABORT_BOUND: std::time::Duration = std::time::Duration::from_secs(3);

/// Aborts this process's tensor-parallel communicator, if one was created (`ncclCommAbort`: a collective stuck on a lost peer returns).
/// Idempotent and lock-free for the stuck collective, and bounded: waits at most [`ABORT_BOUND`] ([`abort_bounded`]). It is the
/// leader's `on_group_lost` hook and the follower's `abort_hook`.
pub fn abort_group_comm() {
    let c = GROUP_COMM.lock().unwrap_or_else(|p| p.into_inner()).clone();
    if let Some(c) = c {
        abort_bounded(c, ABORT_BOUND);
    }
}

/// Runs `c.abort()` on a thread of its own and waits for it at most `bound`: `true` when it returned in time. A late abort is logged
/// (`event=tp_comm_abort_timeout`) and left behind; nothing here waits for it again, and the process's exit backstop still ends it.
pub fn abort_bounded(c: Arc<dyn TpComm>, bound: std::time::Duration) -> bool {
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let t0 = std::time::Instant::now();
    let spawned = std::thread::Builder::new().name("tp-comm-abort".into()).spawn(move || {
        c.abort();
        let _ = tx.send(());
    });
    if spawned.is_err() {
        eprintln!("rexl3 level=error event=tp_comm_abort_timeout reason=\"could not spawn the abort thread\"");
        return false;
    }
    match rx.recv_timeout(bound) {
        Ok(()) => true,
        Err(_) => {
            eprintln!(
                "rexl3 level=error event=tp_comm_abort_timeout after_s={:.1} (ncclCommAbort did not return; the exit backstop ends the process)",
                t0.elapsed().as_secs_f64()
            );
            false
        }
    }
}

/// The communicator's health as the lockstep watchdog polls it (`TpComm::async_error`: NCCL's `ncclCommGetAsyncError`), `None`
/// before a communicator exists.
pub fn group_comm_health() -> Option<String> {
    let c = GROUP_COMM.lock().unwrap_or_else(|p| p.into_inner()).clone();
    c.and_then(|c| c.async_error())
}

/// What one rank of a group charges the memory plan: the weights it holds (`engine::resident_weight_bytes_tp`) under the serving shard plan,
/// the largest of ALL the group's ranks' figures (`engine::resident_weight_bytes_group` over `0..R`; at two ranks rank 1 holds 128 more
/// `in_proj` columns a layer than rank 0), so that every rank plans the same slots and KV pages from the same free memory and the boot
/// gate compares like with like. The group size is `o.tp`'s ranks (two when the options carry none: the verified group). Whole-model
/// bytes when the group is replicated (`REXL3_TP_COMM=none`).
pub fn rank_weight_bytes(
    plan: &FamilyPlan,
    layers: &LayerPlan,
    cfg: &OmniConfig,
    listed: u64,
    head_listed: u64,
    o: &crate::serve::ServeOptions,
) -> u64 {
    if !sharded() {
        return engine::resident_weight_bytes(listed, head_listed, plan, o);
    }
    let ranks = group_ranks(o);
    let shards: Vec<_> = (0..ranks).map(|rank| super::tp_arm::serving_shard_plan(TpTopology { rank, ranks }, cfg)).collect();
    // lane S: the shared-expert split's second communicator (its NCCL buffers), the same figure on every rank
    let aux = if super::tp_arm::shared_split_wanted(ranks) { super::tp_arm::AUX_COMM_RESERVE_BYTES } else { 0 };
    engine::resident_weight_bytes_group(layers, cfg.experts, &shards, plan, o) + aux
}

/// The ranks of the group these options describe (two when they carry no group: the group this engine was verified and measured at).
pub fn group_ranks(o: &crate::serve::ServeOptions) -> usize {
    o.tp.as_ref().map_or(2, |t| t.ranks.max(2))
}

/// The family plan of a rank of a group of `ranks`: its memory model prices the TP scratch (`NemotronMemory::with_tp_ranks`: `tp_tmp`,
/// `tp_qkv`, `tp_logits`, at this group size's cuts) when the group is sharded, and the boot logs the serving shard plan's
/// per-matrix decision (`tp_arm::shard_plan_line`: which dense matrices are sliced into what widths and which are replicated, with the
/// reason; the same line on every rank).
pub fn plan_for_rank(plan: &mut FamilyPlan, cfg: &OmniConfig, ranks: usize) {
    if sharded() {
        plan.memory = Box::new(super::memory::NemotronMemory::new(cfg, 0).with_tp_ranks(ranks));
        eprintln!("{}", super::tp_arm::shard_plan_line(ranks, cfg));
    }
}

/// Creates this rank's communicator (TP-N's `NcclTp`). NCCL's `ncclUniqueId` is made by rank 0 and carried to rank 1 by
/// `rv.exchange(Some(id))` / `rv.exchange(None)`, with ONE byte appended that both ranks check: whether the attention matrices are
/// sliced (`REXL3_TP_ATTN`, `tp_arm::slice_attention`). A rank that sliced a matrix its peer holds whole would wait for a collective that
/// never comes, so a mismatch is a boot error naming it. Both ranks then join (`NcclTp::init`, which blocks until both have).
/// An EMPTY offer from rank 0 means no communicator (`REXL3_TP_COMM=none`, the replicated lockstep group of the bring-up): both ranks
/// return `Ok(None)` and load the whole model. NCCL that cannot be loaded when a sharded group was planned is an error, not a fallback
/// (the memory plan priced shards). The NCCL environment (`NCCL_*`) is logged with the join.
pub fn make_comm(topo: TpTopology, rv: &mut Rendezvous<'_>) -> Result<Option<Arc<dyn TpComm>>, ModelError> {
    default_nccl_channels();
    // B7: plan_flags() bit 4 says an expert ownership table is configured; the gate must also compare
    // `super::tp_balance::expert_map_fingerprint()` (0 = contiguous, else the table's 64-bit content hash) between the ranks.
    let attn = super::tp_arm::plan_flags();
    // lane S: the shared-expert split's second communicator. NOT a plan flag (bits 5-7 belong to the KV work): both ranks already
    // compared the lever label (`S0/S1`, in the prefill fact) at the Hello; here rank 0 appends a second unique id when it wants one,
    // and rank 1 refuses an offer whose length says otherwise BEFORE either rank joins any communicator
    let aux_wanted = super::tp_arm::shared_split_wanted(topo.ranks);
    let n = NcclUniqueId::BYTES;
    let offer = if topo.rank == 0 {
        if !sharded() {
            eprintln!("rexl3 event=tp_comm rank=0 status=none why=\"REXL3_TP_COMM=none\"");
            Some(Vec::new())
        } else {
            let id = NcclComm::new_unique_id().map_err(|e| {
                ModelError::Config(format!(
                    "ncclGetUniqueId: {e} (is libnccl on LD_LIBRARY_PATH / REXL3_NCCL_LIB? REXL3_TP_COMM=none runs the group replicated)"
                ))
            })?;
            let mut b = id.as_bytes().to_vec();
            b.push(attn);
            if aux_wanted {
                let id2 = NcclComm::new_unique_id()
                    .map_err(|e| ModelError::Config(format!("ncclGetUniqueId (second communicator): {e}")))?;
                b.extend_from_slice(id2.as_bytes());
            }
            Some(b)
        }
    } else {
        None
    };
    // rank 1 checks what arrived BEFORE it acknowledges, and rank 0 does not join the communicator until it has the acknowledgement:
    // `ncclCommInitRank` cannot be interrupted, so a rank 0 that joined while rank 1 refused would never return
    let mismatch = move |b: &[u8]| -> Result<(), String> {
        if b.is_empty() {
            return Ok(());
        }
        if b.len() < n + 1 {
            return Err(format!(
                "the communicator's bootstrap is {} bytes, expected {} (the ranks run different builds)",
                b.len(),
                bootstrap_len(n, aux_wanted)
            ));
        }
        if b[n] != attn {
            let name = |f: u8| {
                format!(
                    "attention {} and in_proj {}",
                    if f & 1 == 1 { "sliced" } else { "replicated" },
                    if f & 2 == 2 { "replicated" } else { "sliced" }
                )
            };
            return Err(format!(
                "rank 0 runs with {} and this rank with {}: set REXL3_TP_ATTN and REXL3_TP_INPROJ the same on both ranks",
                name(b[n]),
                name(attn)
            ));
        }
        check_aux_offer(b.len(), n, aux_wanted)
    };
    let bytes = rv.exchange_checked(offer, mismatch)?;
    if bytes.is_empty() {
        eprintln!(
            "rexl3 event=tp_comm rank={} ranks={} status=none (no collective backend: the ranks run replicated, lockstep only)",
            topo.rank, topo.ranks
        );
        return Ok(None);
    }
    if bytes.len() < n + 1 {
        return Err(ModelError::Config(format!(
            "the communicator's bootstrap is {} bytes, expected {} (the ranks run different builds)",
            bytes.len(),
            bootstrap_len(n, aux_wanted)
        )));
    }
    check_aux_offer(bytes.len(), n, aux_wanted).map_err(ModelError::Config)?;
    if bytes[n] != attn {
        let name = |f: u8| {
            format!(
                "attention {}, in_proj {}, MTP head experts {} and head lm_head {}",
                if f & 1 == 1 { "sliced" } else { "replicated" },
                if f & 2 == 2 { "replicated" } else { "sliced" },
                if f & 4 == 4 { "whole" } else { "split" },
                if f & 8 == 8 { "whole" } else { "split" }
            )
        };
        return Err(ModelError::Config(format!(
            "rank 0 runs with {} and this rank with {}: set REXL3_TP_ATTN, REXL3_TP_INPROJ, and REXL3_TP_HEAD the same on both ranks",
            name(bytes[n]),
            name(attn)
        )));
    }
    let id = NcclUniqueId::from_bytes(&bytes[..n]).ok_or_else(|| ModelError::Config("the NCCL unique id is malformed".into()))?;
    gb_cuda::rt::init_context()?;
    let t0 = std::time::Instant::now();
    let comm = NcclTp::init(topo, &id).map_err(|e| {
        ModelError::Config(format!("rank {}: ncclCommInitRank failed: {e} (library {})", topo.rank, NcclComm::library_path()))
    })?;
    eprintln!(
        "rexl3 event=tp_comm rank={} ranks={} status=nccl attention_sliced={} in_proj_replicated={} head_experts_split={} head_lm_head_split={} library={} version={} init_s={:.2} env=\"{}\"",
        topo.rank,
        topo.ranks,
        attn & 1 == 1,
        attn & 2 == 2,
        attn & 4 == 0,
        attn & 8 == 0,
        NcclComm::library_path(),
        NcclComm::version().unwrap_or(0),
        t0.elapsed().as_secs_f64(),
        gb_cuda::nccl::nccl_environment().join(" ")
    );
    // lane S: the second communicator (both ranks agreed on it at the Hello (lever label S) and by the bootstrap length above, before either joined anything): its NCCL buffers measured
    // here, per rank, against the reserve the memory plan priced (`tp_arm::AUX_COMM_RESERVE_BYTES`)
    let comm = if aux_wanted {
        let id2 = NcclUniqueId::from_bytes(&bytes[n + 1..])
            .ok_or_else(|| ModelError::Config("the second NCCL unique id is malformed".into()))?;
        let host_before = crate::hostmem::MemInfo::read().map(|m| m.available);
        let (dev_before, _) = gb_cuda::rt::mem_info()?;
        let t1 = std::time::Instant::now();
        let c2 = NcclComm::init(topo.rank, topo.ranks, &id2)
            .map_err(|e| ModelError::Config(format!("rank {}: ncclCommInitRank (second communicator) failed: {e}", topo.rank)))?;
        let aux = NcclTp::new(c2);
        // NCCL connects lazily: one 16-byte swap so its point-to-point buffers exist now (and are measured), never inside a slice
        {
            let st = gb_cuda::Stream::new()?;
            let send = gb_cuda::DevBuf::zeroed(16)?;
            let mut recv = gb_cuda::DevBuf::zeroed(16)?;
            aux.swap_bytes(&send, 0, 16, &mut recv, 0, 16, &st)?;
            st.sync()?;
        }
        let (dev_after, _) = gb_cuda::rt::mem_info()?;
        let host_after = crate::hostmem::MemInfo::read().map(|m| m.available);
        let dev_used = dev_before.saturating_sub(dev_after) as u64;
        let host_used = match (host_before, host_after) {
            (Some(a), Some(b)) => a.saturating_sub(b),
            _ => 0,
        };
        let used = dev_used.max(host_used);
        eprintln!(
            "rexl3 event=tp_comm2 rank={} status=nccl purpose=shared_expert_split init_s={:.2} mem_dev_gib={:.3} mem_host_gib={:.3} reserve_gib={:.3}{}",
            topo.rank,
            t1.elapsed().as_secs_f64(),
            dev_used as f64 / (1u64 << 30) as f64,
            host_used as f64 / (1u64 << 30) as f64,
            super::tp_arm::AUX_COMM_RESERVE_BYTES as f64 / (1u64 << 30) as f64,
            if used > super::tp_arm::AUX_COMM_RESERVE_BYTES { " OVER_RESERVE" } else { "" }
        );
        comm.with_aux(aux)
    } else {
        comm
    };
    let comm: Arc<dyn TpComm> = Arc::new(comm);
    *GROUP_COMM.lock().unwrap_or_else(|p| p.into_inner()) = Some(comm.clone());
    Ok(Some(comm))
}

/// The communicator bootstrap's length: the unique id, the flags byte, and a second unique id when the shared-expert split's second
/// communicator is wanted (lane S).
pub fn bootstrap_len(id_bytes: usize, aux_wanted: bool) -> usize {
    id_bytes + 1 + if aux_wanted { id_bytes } else { 0 }
}

/// Rank 1's check of rank 0's bootstrap length against its own shared-expert split setting (the flags byte already matched): a
/// mismatch is refused by name before either rank joins a communicator, never a hang. Pure (for the tests).
pub fn check_aux_offer(len: usize, id_bytes: usize, aux_wanted: bool) -> Result<(), String> {
    if len == bootstrap_len(id_bytes, aux_wanted) {
        return Ok(());
    }
    if len == bootstrap_len(id_bytes, !aux_wanted) {
        let on = |w: bool| if w { "on (a second communicator)" } else { "off" };
        return Err(format!(
            "rank 0 runs the shared-expert split {} and this rank {}: set REXL3_NEMO_P2TP_SHARED (and the contract) the same on both ranks",
            on(!aux_wanted),
            on(aux_wanted)
        ));
    }
    Err(format!(
        "the communicator's bootstrap is {len} bytes, expected {} (the ranks run different builds)",
        bootstrap_len(id_bytes, aux_wanted)
    ))
}

/// Loads this rank's shard of the pack with `comm` (`NemotronModel::load_tp`: the rank's slices of the weights under the serving shard
/// plan, the arms with their collectives). Without a communicator it is the whole model (`NemotronModel::load`).
pub fn load_rank(
    dir: &Path,
    cfg: &OmniConfig,
    layers: &LayerPlan,
    mo: &ModelOptions,
    comm: Option<Arc<dyn TpComm>>,
) -> Result<NemotronModel, ModelError> {
    let sharded = comm.is_some();
    let t0 = std::time::Instant::now();
    let read0 = gb_cuda::loader::totals();
    let m = match comm {
        Some(comm) => NemotronModel::load_tp(dir, cfg, layers, mo.clone(), comm)?,
        None => NemotronModel::load(dir, cfg, layers, mo.clone())?,
    };
    let load_s = t0.elapsed().as_secs_f64();
    eprintln!("rexl3 event=tp_rank_loaded sharded={sharded} load_s={load_s:.1}");
    eprintln!("{}", load_breakdown_line(&gb_cuda::loader::totals().since(&read0), load_s));
    Ok(m)
}

/// The prefix cache's share of this rank's boot gate: what the rank was asked for (`o.prefix_cache`, after the family's own veto) and
/// what its plan carved (`sp.plan.cache`). Both ranks of a group build it from the same function, so the Hello compares like with like.
pub fn gate_with_cache(gate: BootGate, o: &crate::serve::ServeOptions, sp: &engine::ServePlan) -> BootGate {
    gate.with_cache(tp_runtime::cache_request_label(o.prefix_cache), &sp.plan.cache)
}

/// Rank 0: the `load` closure of `engine::build_with_plan` for a group. `listener` has been bound since the start of the boot. Returns
/// the leader runtime the engine is built around (the follower has loaded its shard and said Ready when this returns). `o` and `sp` are
/// the options and the plan the boot gate carries the prefix cache's budget from (`docs/NEMOTRON_TP.md` 12.5).
#[allow(clippy::too_many_arguments)]
pub fn leader_runtime(
    listener: &std::net::TcpListener,
    tp: &TpOptions,
    dir: &Path,
    cfg: &OmniConfig,
    layers: &LayerPlan,
    mo: &ModelOptions,
    o: &crate::serve::ServeOptions,
    sp: &engine::ServePlan,
) -> Result<TpLeader<NemotronModel>, ModelError> {
    let t = timeouts(tp);
    default_nccl_channels();
    let pack = prefix::pack_identity(dir).map_err(ModelError::Config)?;
    let gate = gate_with_cache(BootGate::local(pack, mo).at(tp.rank, tp.ranks), o, sp);
    eprintln!(
        "rexl3 event=tp_leader_waiting peer={} ranks={} slots={} kv_pages={} max_context={} prefill_rows={} mtp={} cache={} cache_slots={} cache_slot_mib={:.1} cache_pages={} free_after_gib={:.2} nccl=\"{}\"",
        tp.peer,
        tp.ranks,
        gate.slots,
        gate.kv_pages,
        gate.max_context,
        gate.prefill_rows,
        gate.family.mtp,
        gate.family.cache_request,
        gate.family.cache_slots,
        gate.family.cache_slot_bytes as f64 / 1048576.0,
        gate.family.cache_pages,
        crate::serve::gib(sp.free_after_weights),
        crate::lockstep::nccl_summary(&gate.nccl)
    );
    // every follower under one deadline, each identified by its rank and gated before anything else happens
    let links = tp_runtime::accept_group(listener, &gate, &t)?;
    let topo = topology(tp);
    let t0 = std::time::Instant::now();
    let leader =
        tp_runtime::boot_leader_group(links, &gate, &t, |rv| make_comm(topo, rv), |comm| load_rank(dir, cfg, layers, mo, comm))?;
    leader.on_group_lost(Box::new(abort_group_comm));
    leader.on_comm_health(Arc::new(group_comm_health));
    boot::mark("tp_group_ready", None);
    eprintln!("rexl3 event=tp_group_ready ranks={} boot_s={:.1}", tp.ranks, t0.elapsed().as_secs_f64());
    Ok(leader)
}

/// What a follower's own plan said, for its boot log line.
struct FollowerPlan {
    weights_bytes: u64,
    free_after_weights: u64,
    kv_blocks: usize,
    spare_bytes: u64,
    cache_bytes: u64,
}

/// A follower (any rank above 0): plans exactly as rank 0 does (no vision tower: this process serves no images), connects, gates,
/// loads its shard with rank 0's options, says Ready and replays rank 0's calls until the group ends.
pub fn follow(plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<(), ModelError> {
    let mut plan = plan;
    let o = ctx.options;
    let tp = o.tp.as_ref().filter(|t| t.is_group() && t.rank >= 1).ok_or_else(|| {
        ModelError::Config(
            "a tensor-parallel follower needs --tp-ranks 2 --tp-rank 1 (or --tp-ranks N --tp-rank R for any rank R above 0)"
                .into(),
        )
    })?;
    default_nccl_channels();
    let dir = plan.dir.clone();
    let t = timeouts(tp);
    // connect first: a rank that cannot plan (or loads a pack the leader does not) says so on the connection, and rank 0 refuses at once
    // instead of waiting out its connect timeout
    let mut link = tp_runtime::connect_leader_as(tp.rank, &tp.peer, t.connect, t.write)?;
    // the P2 rows cap rule rank 0 applies before its bind (`engine::check_prefill_contract`), said on the connection: rank 0 refuses at
    // once
    if let Err(e) = engine::check_prefill_contract(o) {
        link.refuse(&format!("rank {}: {e}", tp.rank));
        return Err(e);
    }
    let planned = (|| -> Result<(OmniConfig, LayerPlan, BootGate, FollowerPlan), ModelError> {
        let root = super::read_config(&dir).map_err(|e| ModelError::Config(e.to_string()))?;
        let cfg = OmniConfig::from_json(&root).map_err(ModelError::Config)?;
        let inv = Inventory::read(&dir).map_err(ModelError::Config)?;
        let layers = LayerPlan::build(&cfg, &inv).map_err(|e| ModelError::Config(e.join("; ")))?;
        let weight_bytes = layers.bytes(&inv);
        plan_for_rank(&mut plan, &cfg, group_ranks(o));
        let head_bytes: u64 = layers
            .tensors(|u| matches!(u, weights::UnitId::Mtp(_)))
            .iter()
            .filter_map(|w| inv.get(&w.name))
            .map(|t| t.nbytes)
            .sum();
        let (cuda_free, _) = gb_cuda::rt::mem_info()?;
        let host = MemInfo::read();
        let inp = PlanInputs {
            cuda_free: cuda_free as u64,
            host: host.as_ref(),
            weight_bytes: rank_weight_bytes(&plan, &layers, &cfg, weight_bytes, head_bytes, o),
            model_max_context: cfg.max_context,
        };
        // the same plan rank 0 makes, from THIS rank's free memory: with a prefix cache asked for, it prices the snapshot arena and the cache's
        // pages here too (`plan_serving_for` carves them for a runtime that can snapshot), so the gate compares what each rank can hold
        let sp = engine::plan_serving_for::<NemotronModel>(&plan, o, &inp)?;
        let pack = prefix::pack_identity(&dir).map_err(ModelError::Config)?;
        let gate = gate_with_cache(BootGate::local(pack, &sp.model).at(tp.rank, tp.ranks), o, &sp);
        let fp = FollowerPlan {
            weights_bytes: inp.weight_bytes,
            free_after_weights: sp.free_after_weights,
            kv_blocks: sp.plan.kv_blocks,
            spare_bytes: sp.plan.spare_bytes,
            cache_bytes: sp.plan.cache.bytes(),
        };
        Ok((cfg, layers, gate, fp))
    })();
    let (cfg, layers, gate, fp) = match planned {
        Ok(p) => p,
        Err(e) => {
            link.refuse(&format!("rank {} cannot plan its memory: {e}", tp.rank));
            return Err(e);
        }
    };
    // free memory after the plan, per rank (rank 0 prints `event=nemotron_plan` and `event=prefix_cache_plan` for its own)
    eprintln!(
        "rexl3 event=tp_follower_plan rank={} peer={} slots={} kv_pages={} max_context={} prefill_rows={} mtp={} weights_gib={:.2} free_after_gib={:.2} kv_blocks={} spare_gib={:.2} cache={} cache_slots={} cache_slot_mib={:.1} cache_pages={} cache_gib={:.2}",
        tp.rank,
        tp.peer,
        gate.slots,
        gate.kv_pages,
        gate.max_context,
        gate.prefill_rows,
        gate.family.mtp,
        crate::serve::gib(fp.weights_bytes),
        crate::serve::gib(fp.free_after_weights),
        fp.kv_blocks,
        crate::serve::gib(fp.spare_bytes),
        gate.family.cache_request,
        gate.family.cache_slots,
        gate.family.cache_slot_bytes as f64 / 1048576.0,
        gate.family.cache_pages,
        crate::serve::gib(fp.cache_bytes)
    );
    boot::mark("tp_follower_planned", None);
    let topo = topology(tp);
    let t0 = std::time::Instant::now();
    let (rt, link) = tp_runtime::boot_follower(
        link,
        &gate,
        &t,
        |rv| make_comm(topo, rv),
        |opts, comm| load_rank(&dir, &cfg, &layers, opts, comm),
    )?;
    eprintln!("rexl3 event=tp_follower_ready boot_s={:.1}", t0.elapsed().as_secs_f64());
    boot::mark("tp_follower_ready", None);
    let mut fc = FollowerConfig::new(t).with_rank(tp.rank);
    fc.stop = crate::serve::follower_stop_flag();
    fc.abort_hook = Some(Box::new(abort_group_comm));
    fc.health = Some(Arc::new(group_comm_health));
    let rank = tp.rank;
    fc.on_progress = Some(Box::new(move |s| {
        eprintln!(
            "rexl3 event=tp_follower_stats applied={} errors={} prefills={} windows={} rank={rank}",
            s.applied, s.errors, s.prefills, s.windows
        );
    }));
    match tp_runtime::run_follower(rt, link, fc)? {
        FollowerExit::Shutdown(why) => {
            eprintln!("rexl3 event=tp_follower_exit reason=\"leader shutdown: {why}\"");
            Ok(())
        }
        FollowerExit::Stopped => {
            eprintln!("rexl3 event=tp_follower_exit reason=\"stopped\"");
            Ok(())
        }
        FollowerExit::LeaderLost(why) => Err(ModelError::StateLost(format!("rank 0 was lost: {why}"))),
    }
}

/// The boot's page-cache breakdown of one rank's load (`event=boot_load_breakdown`): what was read, how much of it came from the SSD
/// (pages not cached when read) and how much from memory, and the wall time of the load. A cold boot (`cache=cold`) reads the rank's
/// whole shard from the SSD; the window 7/8 boots were all warm (`docs/NEMOTRON_TP_OPERATIONS.md`, cold boot).
pub fn load_breakdown_line(t: &gb_cuda::loader::Totals, load_s: f64) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    format!(
        "rexl3 event=boot_load_breakdown cache={} read_gib={:.2} cold_gib={:.2} warm_gib={:.2} direct_gib={:.2} read_worker_s={:.1} load_s={load_s:.1} read_gib_s={:.2}",
        t.cache_state(),
        t.bytes as f64 / GIB,
        t.cold_bytes as f64 / GIB,
        t.bytes.saturating_sub(t.cold_bytes) as f64 / GIB,
        t.direct_bytes as f64 / GIB,
        t.read_s,
        if load_s > 0.0 { t.bytes as f64 / GIB / load_s } else { 0.0 }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::nemotron_h_omni::tp_comm::LoopbackGroup;
    use gb_cuda::{CudaError, DevBuf, Stream};
    use std::time::{Duration, Instant};

    /// A communicator whose abort blocks (what `ncclCommAbort` behind pending graph launches did, `docs/NEMOTRON_TP.md` 12.3 B2).
    struct StuckAbort(Duration);

    impl TpComm for StuckAbort {
        fn topology(&self) -> TpTopology {
            TpTopology { rank: 1, ranks: 2 }
        }
        fn allreduce_sum_f32(&self, _: &mut DevBuf, _: usize, _: usize, _: &Stream) -> Result<(), CudaError> {
            Ok(())
        }
        fn abort(&self) {
            std::thread::sleep(self.0);
        }
    }

    #[test]
    fn the_load_breakdown_names_the_cache_state_and_the_cold_and_warm_bytes() {
        let g = 1u64 << 30;
        let t = gb_cuda::loader::Totals { bytes: 33 * g, cold_bytes: 32 * g, direct_bytes: 31 * g, read_s: 40.0 };
        let l = load_breakdown_line(&t, 11.0);
        assert!(
            l.starts_with(
                "rexl3 event=boot_load_breakdown cache=cold read_gib=33.00 cold_gib=32.00 warm_gib=1.00 direct_gib=31.00"
            ),
            "{l}"
        );
        assert!(l.ends_with("load_s=11.0 read_gib_s=3.00"), "{l}");
        assert!(load_breakdown_line(&gb_cuda::loader::Totals::default(), 0.0).contains("cache=none"));
    }

    #[test]
    fn an_abort_that_blocks_is_left_behind_after_the_bound_and_a_quick_one_is_waited_for() {
        let t0 = Instant::now();
        assert!(!abort_bounded(Arc::new(StuckAbort(Duration::from_secs(30))), Duration::from_millis(200)));
        assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
        assert!(abort_bounded(Arc::new(StuckAbort(Duration::from_millis(10))), Duration::from_secs(5)));
        assert!(ABORT_BOUND <= Duration::from_secs(5), "the follower's exit backstop is 5 s after a loss");
    }

    #[test]
    fn the_loopback_group_reports_an_injected_async_error_on_every_rank_by_name() {
        let (a, b) = LoopbackGroup::pair_with_timeout(Duration::from_secs(1));
        assert_eq!(a.async_error(), None);
        b.inject_async_error(gb_cuda::nccl::NCCL_ERROR_BASE + 6);
        for (rank, c) in [(0, &a), (1, &b)] {
            let e = c.async_error().expect("injected");
            assert!(e.contains("ncclRemoteError") && e.contains(&format!("rank {rank}'s communicator")), "{e}");
        }
        // the abort of a loopback group returns at once (it breaks the host barrier)
        assert!(abort_bounded(Arc::new(a), Duration::from_secs(1)));
    }

    #[test]
    fn ranks_that_disagree_on_the_second_communicator_are_refused_before_nccl() {
        let n = NcclUniqueId::BYTES;
        assert_eq!((bootstrap_len(n, false), bootstrap_len(n, true)), (n + 1, 2 * n + 1));
        // the same setting on both ranks
        assert!(check_aux_offer(n + 1, n, false).is_ok());
        assert!(check_aux_offer(2 * n + 1, n, true).is_ok());
        // rank 0 offers a second id and this rank wants none, and the other way round: refused by name
        for (len, mine) in [(2 * n + 1, false), (n + 1, true)] {
            let e = check_aux_offer(len, n, mine).unwrap_err();
            assert!(e.contains("REXL3_NEMO_P2TP_SHARED") && e.contains("shared-expert split"), "{e}");
        }
        // any other length: a different build
        assert!(check_aux_offer(n + 7, n, false).unwrap_err().contains("different builds"));
    }
}
