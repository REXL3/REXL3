//! GLM-5.3-Flash IMAGE PROMPTS WITH MTP DRAFTING at R = 2, on one GPU (lane L14, `docs/GLM_TP.md` section 11): an 8-layer prefix of the
//! 2.25bpw pack PLUS its NextN layer, as two ranks of an in-process group (`LoopbackGroup` for the exchange, a `Link` pair for the
//! lockstep protocol: rank 0 is a `GlmTpBackend` with the backend's drafters AND the image path over a `TpLeader`, rank 1 runs
//! `run_follower` on its own thread and holds NO tower), each rank holding only its share of the routed experts (the NextN layer's
//! included), against one model of every expert with no group behind `GlmBackend`. Two synthetic images through the REAL preprocessing
//! and the REAL tower (rank 0's map of the pack; the tower is dropped before the ranks load). Both run `glm_vision_mtp_script` with
//! `--spec mtp` at depth 3: image prompts (one fed at 512 rows, one at 128 rows that ENDS with an image), a text prompt, ten draft
//! rounds. The candidates and every fp16 logit of every answering call, the drafted tokens of every round, the keeps and the
//! speculation counters must be byte-identical (the one model with graphs on and off), with the restricted draft head and the adaptive
//! one. The follower's digest of every prefill, image chunk, window AND draft call is compared with rank 0's by the protocol.
//!
//! Negative controls (must differ from the one model): (a) both ranks own experts 0..144; (b) both ranks' drafters read the image
//! placeholders' own embeddings in place of the image rows (`MtpMutant::PlaceholderImages`: the trunk is untouched, so only the
//! drafter's prefill of an image chunk can make the drafts differ: it proves the NextN head saw the rows `PrefillImage` carried).
//!
//!     flock -x /tmp/rexl3-gpu.lock env REXL3_GLM_DIR=$HOME/models/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121 \
//!         cargo test --release -p gb-model --test glm_tp_vision_mtp -- --ignored --nocapture --test-threads=1

mod glm_support;
mod glm_vision_mtp_script;

use gb_cuda::moe::ExpertOwnership;
use gb_engine::Backend;
use gb_model::families::glm::backend::GlmBackend;
use gb_model::families::glm::load::{boot_mapped_with, LoadOptions};
use gb_model::families::glm::model::{vocab_ranges, GlmModel, ModelOptions, MtpMutant, VocabMode};
use gb_model::families::glm::spec::{SpecConfig, SpecStats};
use gb_model::families::glm::tp::{GlmCall, GlmRank};
use gb_model::families::glm::tp_backend::GlmTpBackend;
use gb_model::families::glm::tp_vision::TpVision;
use gb_model::families::glm::GlmFamily;
use gb_model::families::nemotron_h_omni::tp_comm::LoopbackGroup;
use gb_model::lockstep::{run_follower, FollowerConfig, FollowerExit, Link, TpLeader, TpTimeouts};
use glm_support::*;
use glm_vision_mtp_script::{Prompt, Run};
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

/// A model of the prefix plus the NextN layer holding `own`'s routed experts (`None`: every expert), the restricted head built, `mode`
/// set.
fn load(dir: &Path, own: Option<&ExpertOwnership>, mode: VocabMode) -> GlmModel {
    gb_cuda::rt::init_context().unwrap();
    let pack = match own {
        Some(o) => GlmFamily::map_pack_owned(dir, o).unwrap(),
        None => GlmFamily::map_pack(dir).unwrap(),
    };
    let b = boot_mapped_with(dir, pack, Some(LAYERS), true, &LoadOptions::from_env(20 * GIB), 0.0).unwrap();
    assert!(b.device.mtp.is_some(), "the NextN layer");
    let cfg = b.pack.config.clone();
    let mut m = GlmModel::new_chunked(
        cfg.clone(),
        b.device,
        &ModelOptions { slots: glm_vision_mtp_script::SLOTS, kv_blocks: 64, max_context: 8192, layers: LAYERS },
        CHUNK,
    )
    .unwrap();
    m.mtp_restrict_vocab(&vocab_ranges(512, cfg.vocab)).unwrap();
    m.set_mtp_vocab(mode);
    m
}

fn t() -> TpTimeouts {
    let d = Duration::from_secs(600);
    TpTimeouts { connect: d, ready: d, peer: d, write: d, barrier: d, hang_exit: None, ..TpTimeouts::default() }
}

/// The group: rank 0 a `GlmTpBackend` (drafting and images) over a leader, rank 1 a follower thread with no tower; returns the run, rank
/// 0's counters, its exchanges and the follower's call count.
fn group_run(
    dir: &Path,
    owns: [&ExpertOwnership; 2],
    mode: VocabMode,
    mutant: Option<MtpMutant>,
    image_token: u32,
    p: &[Prompt; 3],
) -> (Run, String, (u64, u64)) {
    let mut a = load(dir, Some(owns[0]), mode);
    let mut b = load(dir, Some(owns[1]), mode);
    a.set_mtp_mutant(mutant);
    b.set_mtp_mutant(mutant);
    let mut comms = LoopbackGroup::with_timeout(2, Duration::from_secs(600)).into_iter();
    a.enable_tp(Arc::new(comms.next().unwrap()), owns[0].clone()).unwrap();
    b.enable_tp(Arc::new(comms.next().unwrap()), owns[1].clone()).unwrap();
    assert!(!a.graphs_on() && !b.graphs_on(), "a rank runs eagerly");
    let (la, lb) = Link::<GlmCall>::pair(Duration::from_secs(600)).unwrap();
    let h = std::thread::spawn(move || run_follower(GlmRank::new(b), lb, FollowerConfig::new(t())).unwrap());
    let leader = TpLeader::lead(GlmRank::new(a), la, t());
    let mut be = GlmTpBackend::with_spec(leader, CHUNK, glm_vision_mtp_script::SLOTS, spec_cfg())
        .with_vision(TpVision::device(Some(image_token)));
    let r = glm_vision_mtp_script::run(&mut be, p, |b: &GlmTpBackend<GlmRank>, rows| {
        b.leader.inner().m.logits_host(0, rows).unwrap()
    });
    assert!(be.take_error().is_none());
    be.leader.barrier().expect("the follower agreed with every call");
    let stats = stats_key(&be.spec.as_ref().unwrap().stats);
    let tp = be.leader.inner().m.tp().unwrap();
    let ex = (tp.exchanges, tp.exchanged_bytes);
    drop(be);
    assert!(matches!(h.join().unwrap(), FollowerExit::Shutdown(_)), "the follower ended cleanly");
    (r, stats, ex)
}

fn drafted(r: &Run) -> usize {
    r.drafts.iter().flatten().map(|d| d.as_ref().map_or(0, |v| v.len())).sum()
}

/// Drafts of the image-prompt slots (0 and 1) in the first round: the head's first draft reads the prompt's carry (slot 1's is an
/// image row's tap) and the drafter's cache over the image rows.
fn first_round_image_drafts(r: &Run) -> usize {
    r.drafts[0][..2].iter().map(|d| d.as_ref().map_or(0, |v| v.len())).sum()
}

#[test]
#[ignore]
fn image_prompts_with_mtp_drafting_on_two_ranks_equal_one_model_byte_for_byte() {
    let Some(dir) = pack_dir() else { return };
    require_available(60 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let owns = [ExpertOwnership::range(E, 0, 2).unwrap(), ExpertOwnership::range(E, 1, 2).unwrap()];

    // rank 0's tower, from rank 0's map of the pack, and the server's preprocessing; dropped before any rank loads
    let (ia, ib, image_token) = {
        let pack0 = GlmFamily::map_pack_owned(&dir, &owns[0]).unwrap();
        glm_vision_mtp_script::encode_two(&dir, &pack0)
    };
    let p = glm_vision_mtp_script::prompts(image_token, &ia, &ib);
    eprintln!(
        "prompts: slot 0 {} tokens (images at {} and {}), slot 1 {} tokens (images at {} and {}, the last ending the prompt), slot 2 {}",
        p[0].tokens.len(),
        p[0].images[0].start,
        p[0].images[1].start,
        p[1].tokens.len(),
        p[1].images[0].start,
        p[1].images[1].start,
        p[2].tokens.len()
    );
    assert_eq!(p[1].images[1].start + p[1].images[1].tokens, p[1].tokens.len(), "slot 1's prompt ends with an image");
    let modes = [("restricted", VocabMode::Restricted), ("adaptive", VocabMode::Adaptive(0.9))];

    // one model, every expert, no group: graphs on and off, both heads
    let mut reference = Vec::new();
    let mut m = load(&dir, None, VocabMode::Restricted);
    for (name, mode) in modes {
        let mut runs = Vec::new();
        for graphs in [true, false] {
            m.set_graphs(graphs);
            m.set_mtp_vocab(mode);
            let mut be = GlmBackend::with_spec(m, spec_cfg()).with_chunk(CHUNK);
            let r = glm_vision_mtp_script::run(&mut be, &p, |b: &GlmBackend, rows| b.m.logits_host(0, rows).unwrap());
            let s = stats_key(&be.spec.as_ref().unwrap().stats);
            m = be.m;
            runs.push((r, s));
        }
        assert_eq!(runs[0], runs[1], "{name}: one model, graphs and eager differ");
        let (r, s) = runs.pop().unwrap();
        let accepted: usize = r.keeps.iter().flatten().map(|k| k - 1).sum();
        eprintln!(
            "one model, {name} head: {} lines, {} drafts ({} in the first round of the image slots), {accepted} kept; graphs == eager; {s}",
            r.lines.len(),
            drafted(&r),
            first_round_image_drafts(&r)
        );
        assert!(first_round_image_drafts(&r) > 0, "{name}: the head drafted nothing right after the image prompts");
        reference.push((r, s));
    }
    // the drafter control on one model: placeholders' embeddings in place of the image rows in the drafter's prefill
    m.set_graphs(false);
    m.set_mtp_vocab(VocabMode::Restricted);
    m.set_mtp_mutant(Some(MtpMutant::PlaceholderImages));
    let mut be = GlmBackend::with_spec(m, spec_cfg()).with_chunk(CHUNK);
    let ph = glm_vision_mtp_script::run(&mut be, &p, |b: &GlmBackend, rows| b.m.logits_host(0, rows).unwrap());
    let ph_differ = ph.lines.iter().zip(&reference[0].0.lines).filter(|(x, y)| x != y).count();
    eprintln!("one model, the drafter without the image rows: {ph_differ} of {} lines differ", ph.lines.len());
    assert!(ph_differ > 0, "the drafter's prefill never read the image rows");
    drop(be);

    // two ranks, each holding only its share (the NextN layer's experts too), the follower with no tower
    for (i, (name, mode)) in modes.into_iter().enumerate() {
        let t0 = std::time::Instant::now();
        let (r, s, ex) = group_run(&dir, [&owns[0], &owns[1]], mode, None, image_token, &p);
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
        eprintln!(
            "BYTE-IDENTICAL ({name} head): {} lines (every answering call's candidates and fp16 logits, every round's drafts), keeps, counters",
            r.lines.len()
        );
    }

    // negative control (a): both ranks own 0..144 (trunk and NextN). Under P2, the serving contract, a rank must own its combine group's
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
    eprintln!("negative control (a) (both ranks own 0..144): refused at enable_tp: {e}");

    // negative control (b): both ranks' drafters ignore the image rows: the group equals the one model's mutant, not the reference
    let (g, _, _) =
        group_run(&dir, [&owns[0], &owns[1]], VocabMode::Restricted, Some(MtpMutant::PlaceholderImages), image_token, &p);
    let differ = g.lines.iter().zip(&reference[0].0.lines).filter(|(x, y)| x != y).count();
    assert!(differ > 0, "the group's drafter never read the image rows");
    assert_eq!(g, ph, "the mutant group differs from the mutant one model");
    eprintln!("negative control (b) (drafters without the image rows, both ranks): {differ} of {} lines differ; == the one model's mutant", g.lines.len());
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}
