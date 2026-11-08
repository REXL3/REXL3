//! The single-node drafting probe (lane L12): one model of an 8-layer prefix of the 2.25bpw pack PLUS its NextN layer, driven through
//! `GlmBackend` with `--spec mtp` at depth 3 (keyed and greedy asks) by the script of `glm_mtp_script`: one digest line per answering call
//! (candidates and every fp16 logit), per round of drafts, and the speculation counters, with graphs on and off and with the restricted
//! and the adaptive draft head. Uses only APIs of the lane's merge base, so the same file runs there: the lines must be identical before
//! and after the lane (single node unchanged by one bit).
//!
//!     flock -x /tmp/rexl3-gpu.lock env REXL3_GLM_DIR=... REXL3_PROBE_OUT=/path/lines.txt \
//!         cargo test --release -p gb-model --test glm_mtp_probe -- --ignored --nocapture --test-threads=1

mod glm_mtp_script;
mod glm_support;

use gb_model::families::glm::backend::GlmBackend;
use gb_model::families::glm::load::{boot_mapped_with, LoadOptions};
use gb_model::families::glm::model::{vocab_ranges, GlmModel, ModelOptions, VocabMode};
use gb_model::families::glm::spec::{SpecConfig, SpecStats};
use gb_model::families::glm::GlmFamily;
use glm_support::*;

const LAYERS: usize = 8;
const CHUNK: usize = 512;

/// The counters that are not wall time.
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

#[test]
#[ignore]
fn single_node_drafting_probe() {
    let Some(dir) = pack_dir() else { return };
    require_available(40 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let pack = GlmFamily::map_pack(&dir).unwrap();
    let b = boot_mapped_with(&dir, pack, Some(LAYERS), true, &LoadOptions::from_env(20 * GIB), 0.0).unwrap();
    let cfg = b.pack.config.clone();
    let mut m = GlmModel::new_chunked(
        cfg.clone(),
        b.device,
        &ModelOptions { slots: glm_mtp_script::SLOTS, kv_blocks: 64, max_context: 8192, layers: LAYERS },
        CHUNK,
    )
    .unwrap();
    assert!(m.has_mtp());
    m.mtp_restrict_vocab(&vocab_ranges(512, cfg.vocab)).unwrap();
    let mut out = Vec::new();
    for (name, mode) in [("restricted", VocabMode::Restricted), ("adaptive", VocabMode::Adaptive(0.9))] {
        for graphs in [true, false] {
            m.set_graphs(graphs);
            m.set_mtp_vocab(mode);
            let mut be = GlmBackend::with_spec(m, spec_cfg());
            let r = glm_mtp_script::run(&mut be, |b: &GlmBackend, rows| b.m.logits_host(0, rows).unwrap());
            let stats = stats_key(&be.spec.as_ref().unwrap().stats);
            m = be.m;
            for l in &r.lines {
                out.push(format!("{name} graphs={graphs} {l}"));
            }
            out.push(format!("{name} graphs={graphs} keeps {:?}", r.keeps));
            out.push(format!("{name} graphs={graphs} stats {stats}"));
        }
    }
    let text = out.join("\n") + "\n";
    eprint!("{text}");
    if let Ok(p) = std::env::var("REXL3_PROBE_OUT") {
        std::fs::write(&p, &text).unwrap();
        eprintln!("written to {p}");
    }
    eprintln!("MemAvailable min {:.1} GiB", gib(guard.finish()));
}
