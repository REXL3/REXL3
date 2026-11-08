//! GLM-5.3-Flash DRAFTING WITH THE NEXTN LAYER at R = 2, on one GPU (lane L12, `docs/GLM_TP.md` section 9): an 8-layer prefix of the
//! 2.25bpw pack PLUS its NextN layer, as two ranks of an in-process group (`LoopbackGroup` for the exchange, a `Link` pair for the
//! lockstep protocol: rank 0 is a `GlmTpBackend` over a `TpLeader`, rank 1 runs `run_follower` on its own thread), each rank mapping and
//! holding ONLY its share of the routed experts, the NextN layer's included (subbanks), against one model of every expert with no group
//! driven through `GlmBackend`. Both run the drafting script of `glm_mtp_script` with `--spec mtp` at depth 3: the candidates and every
//! fp16 logit of every answering call, the drafted tokens of every round, the keeps and the speculation counters must be byte-identical
//! (the one model with graphs on and off), with the restricted draft head in every call and with the adaptive one (the serving default).
//! The follower's digest of every prefill, window AND draft call is compared with rank 0's by the protocol (a difference loses the group).
//! The negative control (both ranks own 0..144) must differ.
//!
//!     flock -x /tmp/rexl3-gpu.lock env REXL3_GLM_DIR=$HOME/models/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121 \
//!         cargo test --release -p gb-model --test glm_tp_mtp -- --ignored --nocapture --test-threads=1

mod glm_mtp_script;
mod glm_support;

use gb_cuda::moe::ExpertOwnership;
use gb_engine::Backend;
use gb_model::families::glm::backend::GlmBackend;
use gb_model::families::glm::device::{FfnDevice, MtpMoe};
use gb_model::families::glm::load::{boot_mapped_with, LoadOptions};
use gb_model::families::glm::model::{vocab_ranges, GlmModel, ModelOptions, VocabMode};
use gb_model::families::glm::spec::{SpecConfig, SpecStats};
use gb_model::families::glm::tp::{GlmCall, GlmRank};
use gb_model::families::glm::tp_backend::GlmTpBackend;
use gb_model::families::glm::GlmFamily;
use gb_model::families::nemotron_h_omni::tp_comm::LoopbackGroup;
use gb_model::lockstep::{run_follower, FollowerConfig, FollowerExit, Link, TpLeader, TpTimeouts};
use glm_mtp_script::Run;
use glm_support::*;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

const LAYERS: usize = 8;
const CHUNK: usize = 512;
const E: usize = 288;

fn stats_key(s: &SpecStats) -> String {
    format!(
        "rounds={:?} verified={:?} accepted={:?} committed={:?} reached={:?} hits={:?} by_depth={:?} mtp_calls={} mtp_rows={} windows={} window_rows={}",
        s.rounds, s.verified, s.accepted, s.committed, s.reached, s.hits, s.by_depth, s.mtp_calls, s.mtp_rows, s.windows, s.window_rows
    )
}

fn spec_cfg() -> SpecConfig {
    let mut c = SpecConfig::new(gb_spec::SpecMode::Mtp);
    c.mtp_depth = 3;
    c.restricted_head = true;
    c
}

/// A model of the prefix plus the NextN layer holding `own`'s routed experts (`None`: every expert), with the restricted head built and
/// `mode` set. Also: whether a trunk layer bound as subbanks (a mixed-width pack).
fn load_mixed(dir: &Path, own: Option<&ExpertOwnership>, mode: VocabMode) -> (GlmModel, bool) {
    gb_cuda::rt::init_context().unwrap();
    let pack = match own {
        Some(o) => GlmFamily::map_pack_owned(dir, o).unwrap(),
        None => GlmFamily::map_pack(dir).unwrap(),
    };
    let before = mem_available();
    let b = boot_mapped_with(dir, pack, Some(LAYERS), true, &LoadOptions::from_env(20 * GIB), 0.0).unwrap();
    let nextn = match &b.device.mtp.as_ref().expect("the NextN layer").moe {
        MtpMoe::Single(_) => format!("one bank of {E} experts"),
        MtpMoe::Subbanks(s) => format!("{} subbank(s) of {} experts", s.experts.banks().len(), s.experts.ownership().local()),
    };
    eprintln!(
        "loaded {LAYERS} layers + NextN ({}): regions {:.3} GiB, NextN experts as {nextn}; MemAvailable fell {:.2} GiB",
        own.map_or("every expert".to_string(), |o| format!("experts {}..={}", o.owned()[0], o.owned()[o.local() - 1])),
        gib(b.device.regions.bytes()),
        gib(before.saturating_sub(mem_available()))
    );
    let mixed = b.device.layers.iter().any(|l| matches!(l.ffn, FfnDevice::MoeSubbanks { .. }));
    let cfg = b.pack.config.clone();
    let mut m = GlmModel::new_chunked(
        cfg.clone(),
        b.device,
        &ModelOptions { slots: glm_mtp_script::SLOTS, kv_blocks: 64, max_context: 8192, layers: LAYERS },
        CHUNK,
    )
    .unwrap();
    m.mtp_restrict_vocab(&vocab_ranges(512, cfg.vocab)).unwrap();
    m.set_mtp_vocab(mode);
    (m, mixed)
}

fn load(dir: &Path, own: Option<&ExpertOwnership>, mode: VocabMode) -> GlmModel {
    load_mixed(dir, own, mode).0
}

fn t() -> TpTimeouts {
    let d = Duration::from_secs(300);
    TpTimeouts { connect: d, ready: d, peer: d, write: d, barrier: d, hang_exit: None, ..TpTimeouts::default() }
}

/// The group: rank 0 a `GlmTpBackend` over a leader, rank 1 a follower thread; returns the run, rank 0's counters and its exchanges.
/// `whole`: every rank holds the whole bank of every MoE layer, the NextN layer's included, masked to its share (L2b's path).
fn group_run(dir: &Path, owns: [&ExpertOwnership; 2], mode: VocabMode, whole: bool) -> (Run, String, (u64, u64)) {
    let mut a = load(dir, (!whole).then_some(owns[0]), mode);
    let mut b = load(dir, (!whole).then_some(owns[1]), mode);
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(300)).into_iter();
    a.enable_tp(Arc::new(comms.next().unwrap()), owns[0].clone()).unwrap();
    b.enable_tp(Arc::new(comms.next().unwrap()), owns[1].clone()).unwrap();
    assert!(!a.graphs_on() && !b.graphs_on(), "a rank runs eagerly");
    let (la, lb) = Link::<GlmCall>::pair(Duration::from_secs(300)).unwrap();
    let h = std::thread::spawn(move || run_follower(GlmRank::new(b), lb, FollowerConfig::new(t())).unwrap());
    let leader = TpLeader::lead(GlmRank::new(a), la, t());
    let mut be = GlmTpBackend::with_spec(leader, CHUNK, glm_mtp_script::SLOTS, spec_cfg());
    let r = glm_mtp_script::run(&mut be, |b: &GlmTpBackend<GlmRank>, rows| b.leader.inner().m.logits_host(0, rows).unwrap());
    assert!(be.take_error().is_none());
    be.leader.barrier().expect("the follower agreed with every call");
    let stats = stats_key(&be.spec.as_ref().unwrap().stats);
    let tp = be.leader.inner().m.tp().unwrap();
    let ex = (tp.exchanges, tp.exchanged_bytes);
    drop(be);
    assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)), "the follower ended cleanly");
    (r, stats, ex)
}

#[test]
#[ignore]
fn two_ranks_drafting_with_the_nextn_layer_equal_one_model_byte_for_byte() {
    let Some(dir) = pack_dir() else { return };
    require_available(50 * GIB);
    let guard = Guard::start(20 * GIB);
    let modes = [("restricted", VocabMode::Restricted), ("adaptive", VocabMode::Adaptive(0.9))];

    // one model, every expert, no group: graphs on and off
    let mut reference = Vec::new();
    let (mut m, mixed) = load_mixed(&dir, None, VocabMode::Restricted);
    for (name, mode) in modes {
        let mut runs = Vec::new();
        for graphs in [true, false] {
            m.set_graphs(graphs);
            m.set_mtp_vocab(mode);
            let mut be = GlmBackend::with_spec(m, spec_cfg());
            let r = glm_mtp_script::run(&mut be, |b: &GlmBackend, rows| b.m.logits_host(0, rows).unwrap());
            let s = stats_key(&be.spec.as_ref().unwrap().stats);
            m = be.m;
            runs.push((r, s));
        }
        assert_eq!(runs[0], runs[1], "{name}: one model, graphs and eager differ");
        let (r, s) = runs.pop().unwrap();
        let drafted: usize = r.drafts.iter().flatten().map(|d| d.as_ref().map_or(0, |v| v.len())).sum();
        let accepted: usize = r.keeps.iter().flatten().map(|k| k - 1).sum();
        eprintln!("one model, {name} head: {} lines, {drafted} drafts, {accepted} kept; graphs == eager; {s}", r.lines.len());
        assert!(drafted > 0, "{name}: the head drafted nothing");
        reference.push((r, s));
    }
    drop(m);

    // two ranks, each holding only its share (the NextN layer's experts too)
    let owns = [ExpertOwnership::range(E, 0, 2).unwrap(), ExpertOwnership::range(E, 1, 2).unwrap()];
    for (i, (name, mode)) in modes.into_iter().enumerate() {
        let t0 = std::time::Instant::now();
        let (r, s, ex) = group_run(&dir, [&owns[0], &owns[1]], mode, false);
        eprintln!(
            "two ranks, {name} head, in {:.1?}: {} exchanges of {:.1} MiB in all on rank 0",
            t0.elapsed(),
            ex.0,
            ex.1 as f64 / 1048576.0
        );
        for (x, y) in r.lines.iter().zip(&reference[i].0.lines) {
            assert_eq!(x, y, "{name}: the group's line differs from the one model's");
        }
        assert_eq!(r, reference[i].0, "{name}: drafts, keeps or lines differ");
        assert_eq!(s, reference[i].1, "{name}: the speculation counters differ");
        eprintln!("BYTE-IDENTICAL ({name} head): {} lines (every answering call's candidates and fp16 logits, every round's drafts), keeps, counters", r.lines.len());
    }

    // L2b's path: every rank holds every expert (the NextN layer one bank), masked to its share (a uniform pack only: a mixed-width
    // pack's rank must load its share alone, `enable_tp` refuses whole subbank layers)
    if mixed {
        eprintln!("a mixed-width pack: the whole-bank masked path does not apply (enable_tp refuses it by name)");
    } else {
        let (r, s, ex) = group_run(&dir, [&owns[0], &owns[1]], VocabMode::Restricted, true);
        assert_eq!(r, reference[0].0, "whole banks masked: drafts, keeps or lines differ");
        assert_eq!(s, reference[0].1, "whole banks masked: the speculation counters differ");
        eprintln!(
            "BYTE-IDENTICAL (whole banks masked to the share, restricted head): {} lines, {} exchanges",
            r.lines.len(),
            ex.0
        );
    }

    // the negative control: both ranks own 0..144 (trunk and NextN). Under P2, the serving contract, a rank must own its combine group's
    // experts (`ExpertOwnership::range`), so rank 1 with rank 0's share is refused by name at `enable_tp`, before any call runs (under P1
    // it ran and its answers differed from the one model's)
    let mut n = load(&dir, Some(&owns[0]), VocabMode::Restricted);
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(60)).into_iter();
    let _rank0 = comms.next().unwrap();
    let e = n.enable_tp(Arc::new(comms.next().unwrap()), owns[0].clone()).unwrap_err().to_string();
    assert!(
        e.contains("rank 1 must own the experts of combine group 1"),
        "an ownership that does not partition the experts went unnoticed: {e}"
    );
    drop(n);
    eprintln!("negative control (both ranks own 0..144): refused at enable_tp: {e}");
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}

/// The SERVING rank load with the NextN layer (what `build_leader` and `follow` run with `--spec auto`): `tp_boot::plan_from` (drafting on,
/// so the plan takes the layer) -> `tp_boot::load_rank_with` (the layer's share, the allocation check against `AllocPlan { mtp: true }`,
/// `enable_tp`, then the restricted head built from `REXL3_MTP_VOCAB`'s setting, the default: 512 strips, adaptive at 0.9), for an 8-layer
/// prefix; each rank's regions equal the plan's bytes of the prefix plus the NextN layer; then the drafting script through
/// `GlmTpBackend` and a follower thread against one model with the same head.
#[test]
#[ignore]
fn the_serving_rank_load_with_the_nextn_layer_equals_one_model() {
    use gb_model::families::glm::load::keep_prefix_mtp;
    use gb_model::families::glm::tp_boot::{load_rank_with, plan_from, vocab_setting};
    use gb_model::families::glm::GlmConfig;
    use gb_model::serve::{PrefixCacheOption, ServeOptions};
    let Some(dir) = pack_dir() else { return };
    require_available(50 * GIB);
    let guard = Guard::start(20 * GIB);
    let setting = vocab_setting().unwrap();
    let (strips, share) = setting.expect("REXL3_MTP_VOCAB unset: the default restricted head");
    let mode = if share > 0.0 { VocabMode::Adaptive(share) } else { VocabMode::Restricted };
    assert_eq!(strips, 512, "this test builds the one model's head with the default strips");
    let mut one = load(&dir, None, mode);
    one.set_graphs(true);
    let mut be = GlmBackend::with_spec(one, spec_cfg());
    let reference = glm_mtp_script::run(&mut be, |b: &GlmBackend, rows| b.m.logits_host(0, rows).unwrap());
    let ref_stats = stats_key(&be.spec.as_ref().unwrap().stats);
    drop(be);

    let cfg = GlmConfig::from_dir(&dir).unwrap();
    let mut o = ServeOptions::new(dir.clone());
    (o.slots, o.slots_auto, o.max_model_len, o.kv_tokens, o.prefill_chunk) =
        (glm_mtp_script::SLOTS, false, 8192, Some(16384), CHUNK);
    o.prefix_cache = PrefixCacheOption::Off;
    o.vision = None;
    o.drafting = true;
    o.spec = gb_spec::SpecMode::Auto;
    let plans: Vec<_> = (0..2)
        .map(|r| {
            let ck = gb_exl3::checkpoint::scan(&dir, false).unwrap();
            plan_from(&o, cfg.clone(), ck, 117 * GIB, 0.0, Some(ExpertOwnership::range(E, r, 2).unwrap())).unwrap()
        })
        .collect();
    assert!(plans.iter().all(|p| p.mtp), "drafting on and --spec auto: the plan takes the NextN layer");
    let mo = ModelOptions { slots: glm_mtp_script::SLOTS, kv_blocks: 64, max_context: 8192, layers: LAYERS };
    let mut pcfg = cfg.clone();
    pcfg.layers.truncate(LAYERS);
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(300)).into_iter();
    let mut ranks = Vec::new();
    for p in &plans {
        let c = comms.next().unwrap();
        let want: u64 = p
            .pack
            .plan
            .select_regions(keep_prefix_mtp(LAYERS, Some(cfg.mtp_layer_index(0))))
            .regions
            .iter()
            .map(|r| r.bytes)
            .sum();
        let rank = load_rank_with(
            &dir,
            &pcfg,
            p.pack.clone(),
            p.own.as_ref(),
            &mo,
            p.chunk,
            20 * GIB,
            0.0,
            Some(Arc::new(c)),
            Some(setting),
        )
        .unwrap_or_else(|e| panic!("load_rank_with: {e}"));
        assert!(rank.m.has_mtp() && rank.m.mtp_sub_head().is_some(), "the NextN layer and the restricted head on the rank");
        assert_eq!(rank.m.dev.regions.bytes(), want, "the rank's regions are the plan's bytes of the prefix and the NextN layer");
        eprintln!("serving rank load: regions {:.3} GiB = the plan's (prefix + NextN share)", gib(want));
        ranks.push(rank);
    }
    let b = ranks.pop().unwrap();
    let a = ranks.pop().unwrap();
    let (la, lb) = Link::<GlmCall>::pair(Duration::from_secs(300)).unwrap();
    let h = std::thread::spawn(move || run_follower(b, lb, FollowerConfig::new(t())).unwrap());
    let leader = TpLeader::lead(a, la, t());
    let mut be = GlmTpBackend::with_spec(leader, CHUNK, glm_mtp_script::SLOTS, spec_cfg());
    let r = glm_mtp_script::run(&mut be, |b: &GlmTpBackend<GlmRank>, rows| b.leader.inner().m.logits_host(0, rows).unwrap());
    assert!(be.take_error().is_none());
    be.leader.barrier().expect("the follower agreed with every call");
    let stats = stats_key(&be.spec.as_ref().unwrap().stats);
    drop(be);
    assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)));
    assert_eq!(r, reference, "the serving rank load differs from one model");
    assert_eq!(stats, ref_stats, "the speculation counters differ");
    eprintln!("BYTE-IDENTICAL through tp_boot::plan_from -> load_rank_with: {} lines, keeps, counters", r.lines.len());
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}
