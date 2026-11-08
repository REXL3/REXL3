//! The tensor-parallel boot of the DeepSeek-V4.1-Flash family, the mirror of nemotron's `tp_boot` at this family's
//! width: the communicator's construction over the lockstep rendezvous, the group communicator's abort and health
//! hooks, and [`follow`] — the follower process of every rank above 0 (plan from its own memory, connect, gate, load
//! rank 0's planned geometry, say Ready, replay rank 0's calls until the group ends).
//!
//! **The communicator.** Rank 0 offers the NCCL unique id through the [`Rendezvous`] (the same TCP connection that
//! carries the lockstep protocol — the connection rank 1 opened while rank 0 was still scanning, because the family
//! binds the rendezvous listener BEFORE the scan); every follower checks what arrived BEFORE it acknowledges and
//! rank 0 does not join until it has every acknowledgement (`ncclCommInitRank` cannot be interrupted, so a follower
//! that refused must never leave rank 0 inside one). No flags byte rides the offer: this family's shard plan is a
//! function of the config alone, and every environment knob that changes the arithmetic is compared in the Hello's
//! facts ([`Dsv41Facts`]) before anything is offered. NCCL that cannot be loaded is an error, not a fallback.
//!
//! **The failure model.** The group's communicator is kept in [`GROUP_COMM`] so the lockstep watchdog can abort it
//! from its own thread ([`abort_group_comm`], bounded: an `ncclCommAbort` that blocks behind pending launches is
//! left to the exit backstop) and poll its health ([`group_comm_health`]); the follower's loop reports its progress
//! on stderr and exits with the leader's verdict.

use super::config::DeepseekV41Config;
use super::tp::{self, ShardTable, TpComm, TpTopology};
use super::tp_runtime::{self, BootGate, FollowerConfig, Rendezvous, TpTimeouts};
use crate::error::ModelError;
use crate::family::{BuildCtx, FamilyPlan};
use crate::serve::TpOptions;
use gb_cuda::nccl::{NcclComm, NcclUniqueId};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// This process's place in the group (the nemotron `tp_boot::topology` mirror).
pub fn topology(tp: &TpOptions) -> TpTopology {
    TpTopology { rank: tp.rank, ranks: tp.ranks }
}

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

/// The rank's split table for `cfg`, fail-closed: a config whose counts do not divide by the rank count (or a rank past
/// the group's end) is refused here, by name, before anything is planned or read.
pub fn shard_table(tp: &TpOptions, cfg: &DeepseekV41Config) -> Result<ShardTable, ModelError> {
    ShardTable::for_config(topology(tp), cfg).map_err(ModelError::Config)
}

/// [`shard_table`] with NextN experts replicated across ranks when DSpark (`--spec auto` / `--spec mtp`) is active.
pub fn shard_table_for(
    tp: &TpOptions,
    cfg: &DeepseekV41Config,
    o: &crate::serve::ServeOptions,
) -> Result<ShardTable, ModelError> {
    let rep = o.drafting && o.spec.uses_mtp();
    Ok(shard_table(tp, cfg)?.with_replicated_nextn(rep))
}

/// The RoCE transport's step sites for DSV4.1: covers `o_gather` (32 KiB/row) and `y_swap` (60 KiB/row) from 1-row
/// decode through 8-row DSpark verify windows and 128-row prefill chunks (up to 8 MiB).
pub const ROCE_SITES: [usize; 9] = [32 << 10, 64 << 10, 128 << 10, 256 << 10, 512 << 10, 1 << 20, 2 << 20, 4 << 20, 8 << 20];

/// Exports `NCCL_MAX_NCHANNELS=4` when it is not set: NCCL's own tuning is 3 to 10 times slower on the two-node fabric
/// (`docs/NEMOTRON_TP.md` section 10.3) and a group whose ranks run different channel counts runs the slower schedule
/// on all of them. Idempotent. Called before the gate is made (the gate compares the effective value:
/// [`crate::lockstep::nccl_fingerprint`]) and again before the communicator is joined, on every rank, so a rank started
/// without the variable runs what the others run.
pub fn default_nccl_channels(group: bool) {
    if group && std::env::var_os("NCCL_MAX_NCHANNELS").is_none() {
        std::env::set_var("NCCL_MAX_NCHANNELS", crate::lockstep::DEFAULT_NCCL_CHANNELS);
        eprintln!(
            "rexl3 event=tp_nccl_default NCCL_MAX_NCHANNELS={} (it was not set: NCCL's own tuning is 3 to 10 times slower on this fabric)",
            crate::lockstep::DEFAULT_NCCL_CHANNELS
        );
    }
}

/// The group's communicator, kept so the lockstep runtime's watchdog can abort it from its own thread
/// ([`abort_group_comm`]) and poll its health ([`group_comm_health`]).
static GROUP_COMM: Mutex<Option<Arc<dyn TpComm>>> = Mutex::new(None);

/// How long [`abort_group_comm`] waits for `ncclCommAbort`: it has been seen to block behind pending graph launches
/// (`docs/NEMOTRON_TP.md` 12.3 B2). After this the caller goes on (its exit path, then the process's exit backstop),
/// and the abort is left running on its own thread.
pub const ABORT_BOUND: Duration = Duration::from_secs(3);

/// Aborts this process's tensor-parallel communicator, if one was created (`ncclCommAbort`: a collective stuck on a
/// lost peer returns). Idempotent and lock-free for the stuck collective, and bounded: waits at most [`ABORT_BOUND`].
/// It is the leader's `on_group_lost` hook and the follower's `abort_hook`.
pub fn abort_group_comm() {
    let c = GROUP_COMM.lock().unwrap_or_else(|p| p.into_inner()).clone();
    if let Some(c) = c {
        abort_bounded(c, ABORT_BOUND);
    }
}

/// Runs `c.abort()` on a thread of its own and waits for it at most `bound`: `true` when it returned in time. A late
/// abort is logged (`event=tp_comm_abort_timeout`) and left behind; nothing here waits for it again.
pub fn abort_bounded(c: Arc<dyn TpComm>, bound: Duration) -> bool {
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

/// The communicator's health as the lockstep watchdog polls it (`TpComm::async_error`: NCCL's `ncclCommGetAsyncError`),
/// `None` before a communicator exists.
pub fn group_comm_health() -> Option<String> {
    let c = GROUP_COMM.lock().unwrap_or_else(|p| p.into_inner()).clone();
    c.and_then(|c| c.async_error())
}

/// Creates this rank's communicator (the `make_comm` the lockstep boot calls): rank 0 offers the unique id through
/// the [`Rendezvous`], every follower receives it and both join (`NcclTp::init`, which blocks until every rank has).
/// A follower checks the offer's LENGTH before it acknowledges, and rank 0 joins nothing until it has every
/// acknowledgement, so a rank pair that disagrees fails by name instead of hanging inside `ncclCommInitRank`.
pub fn make_comm(topo: TpTopology, rv: &mut Rendezvous<'_>) -> Result<Arc<dyn TpComm>, ModelError> {
    default_nccl_channels(true);
    let n = NcclUniqueId::BYTES;
    let offer = if topo.rank == 0 {
        let id = NcclComm::new_unique_id()
            .map_err(|e| ModelError::Config(format!("ncclGetUniqueId: {e} (is libnccl on LD_LIBRARY_PATH / REXL3_NCCL_LIB?)")))?;
        Some(id.as_bytes().to_vec())
    } else {
        None
    };
    let bytes = rv.exchange_checked(offer, move |b: &[u8]| {
        if b.len() == n {
            return Ok(());
        }
        Err(format!("the communicator's bootstrap is {} bytes, expected {n} (the ranks run different builds)", b.len()))
    })?;
    let id = NcclUniqueId::from_bytes(&bytes).ok_or_else(|| ModelError::Config("the NCCL unique id is malformed".into()))?;
    gb_cuda::rt::init_context()?;
    let t0 = std::time::Instant::now();
    let comm = crate::families::nemotron_h_omni::tp_comm::NcclTp::init(topo, &id).map_err(|e| {
        ModelError::Config(format!("rank {}: ncclCommInitRank failed: {e} (library {})", topo.rank, NcclComm::library_path()))
    })?;
    eprintln!(
        "rexl3 event=dsv41_tp_comm rank={} ranks={} status=nccl library={} version={} init_s={:.2} env=\"{}\"",
        topo.rank,
        topo.ranks,
        NcclComm::library_path(),
        NcclComm::version().unwrap_or(0),
        t0.elapsed().as_secs_f64(),
        gb_cuda::nccl::nccl_environment().join(" ")
    );
    use crate::families::nemotron_h_omni::tp_roce::{self, Connected, RoceTp, SitePlan, Transport};
    let setting = tp_roce::transport(Transport::Auto).map_err(ModelError::Config)?;
    let comm: Arc<dyn TpComm> = match setting {
        Transport::Nccl => Arc::new(comm),
        Transport::Roce | Transport::Auto => {
            let t1 = std::time::Instant::now();
            let plan = SitePlan::step(&ROCE_SITES).with_sum_max(8 << 20);
            let strict = setting == Transport::Roce;
            let up = RoceTp::connect_over_nccl_soft(comm, &plan, strict).map_err(|e| {
                ModelError::Config(format!("rank {}: the RoCE transport (REXL3_TP_TRANSPORT={}): {e}", topo.rank, setting.name()))
            })?;
            match up {
                Connected::Roce(r) => {
                    eprintln!(
                        "rexl3 event=dsv41_tp_comm rank={} transport=roce setting={} init_s={:.2} sites={:?} {}",
                        topo.rank,
                        setting.name(),
                        t1.elapsed().as_secs_f64(),
                        plan.fact(),
                        r.fingerprint()
                    );
                    Arc::new(r)
                }
                Connected::Nccl { nccl, why } => {
                    eprintln!("rexl3 event=dsv41_tp_comm rank={} transport=nccl setting=auto reason=\"{}\"", topo.rank, why);
                    Arc::new(nccl)
                }
            }
        }
    };
    *GROUP_COMM.lock().unwrap_or_else(|p| p.into_inner()) = Some(comm.clone());
    Ok(comm)
}

/// A follower (any rank above 0): plans exactly as rank 0 does from THIS rank's own free memory, connects (rank 0 has
/// been listening since the start of its boot), is gated, loads its shard with RANK 0'S planned geometry, says Ready
/// and replays rank 0's calls until the group ends. Every planning error is said on the connection, so rank 0 refuses
/// at once instead of waiting out its accept deadline.
pub fn follow(_plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<(), ModelError> {
    let o = ctx.options;
    let tp = o.tp.as_ref().filter(|t| t.is_group() && t.rank >= 1).ok_or_else(|| {
        ModelError::Config(
            "a tensor-parallel follower needs --tp-ranks 2 --tp-rank 1 (or --tp-ranks N --tp-rank R for any rank R above 0)"
                .into(),
        )
    })?;
    tp::check_tp_plain(o)?;
    default_nccl_channels(true);
    let t = timeouts(tp);
    // connect first: a rank that cannot plan (or refuses a pack the leader does not) says so on the connection, and
    // rank 0 refuses at once instead of waiting out its accept deadline
    let mut link = crate::lockstep::connect_leader_as(tp.rank, &tp.peer, t.connect, t.write)?;
    let planned = super::engine::plan_follower(o);
    let (fp, gate) = match planned {
        Ok(p) => p,
        Err(e) => {
            link.refuse(&format!("rank {}: {e}", tp.rank));
            return Err(e);
        }
    };
    eprintln!(
        "rexl3 event=tp_follower_plan rank={} peer={} slots={} kv_pages={} max_context={} prefill_rows={} weights_gib={:.2} free_after_gib={:.2} kv_blocks={} spare_gib={:.2} nccl=\"{}\"",
        tp.rank,
        tp.peer,
        gate.slots,
        gate.kv_pages,
        gate.max_context,
        gate.prefill_rows,
        crate::serve::gib(fp.weights_bytes),
        crate::serve::gib(fp.free_after_weights),
        fp.kv_blocks,
        crate::serve::gib(fp.spare_bytes),
        crate::lockstep::nccl_summary(&gate.nccl)
    );
    gb_engine::boot::mark("tp_follower_planned", None);
    let topo = topology(tp);
    let t0 = std::time::Instant::now();
    let (model, link) = tp_runtime::boot_follower(
        link,
        &gate,
        &t,
        |rv| make_comm(topo, rv),
        move |gate: &BootGate, comm| super::engine::load_follower(o, fp, &gate.model_options(), comm),
    )?;
    eprintln!("rexl3 event=tp_follower_ready boot_s={:.1}", t0.elapsed().as_secs_f64());
    gb_engine::boot::mark("tp_follower_ready", None);
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
    match tp_runtime::run_follower(model, link, fc)? {
        tp_runtime::FollowerExit::Shutdown(why) => {
            eprintln!("rexl3 event=tp_follower_exit reason=\"leader shutdown: {why}\"");
            Ok(())
        }
        tp_runtime::FollowerExit::Stopped => {
            eprintln!("rexl3 event=tp_follower_exit reason=\"stopped\"");
            Ok(())
        }
        tp_runtime::FollowerExit::LeaderLost(why) => Err(ModelError::StateLost(format!("rank 0 was lost: {why}"))),
    }
}
