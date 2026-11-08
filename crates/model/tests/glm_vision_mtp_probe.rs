//! The single-node IMAGE + DRAFTING probe (lane L14): one model of an 8-layer prefix of the 2.25bpw pack PLUS its NextN layer, driven
//! through `GlmBackend` by the script of `glm_vision_mtp_script` (two synthetic images through the real preprocessing and tower, image
//! prompts fed at 512 and 128 rows, one ending with an image; ten draft rounds): one digest line per answering call (candidates and every
//! fp16 logit), per round of drafts, the keeps and the speculation counters; with graphs on and off, the restricted and the adaptive
//! draft head, and once with no drafting at all (`--spec none`, the image path alone). Uses only APIs of the merge base of lanes L12
//! and L13 (`ab6a0771`), so the same file runs there and at L13's head (`c95afea6`): the lines must be identical before and after the
//! merge (single node unchanged by one bit).
//!
//!     flock -x /tmp/rexl3-gpu.lock env REXL3_GLM_DIR=... REXL3_PROBE_OUT=/path/lines.txt \
//!         cargo test --release -p gb-model --test glm_vision_mtp_probe -- --ignored --nocapture --test-threads=1

mod glm_support;
mod glm_vision_mtp_script;

use gb_model::families::glm::backend::GlmBackend;
use gb_model::families::glm::load::{boot_mapped_with, LoadOptions};
use gb_model::families::glm::model::{vocab_ranges, GlmModel, ModelOptions, VocabMode};
use gb_model::families::glm::spec::{SpecConfig, SpecStats};
use gb_model::families::glm::GlmFamily;
use glm_support::*;

const LAYERS: usize = 8;
const CHUNK: usize = 512;

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
fn single_node_image_drafting_probe() {
    let Some(dir) = pack_dir() else { return };
    require_available(40 * GIB);
    let guard = Guard::start(20 * GIB);
    gb_cuda::rt::init_context().unwrap();
    let (ia, ib, image_token) = {
        let pack = GlmFamily::map_pack(&dir).unwrap();
        glm_vision_mtp_script::encode_two(&dir, &pack)
    };
    let p = glm_vision_mtp_script::prompts(image_token, &ia, &ib);
    let pack = GlmFamily::map_pack(&dir).unwrap();
    let b = boot_mapped_with(&dir, pack, Some(LAYERS), true, &LoadOptions::from_env(20 * GIB), 0.0).unwrap();
    let cfg = b.pack.config.clone();
    let mut m = GlmModel::new_chunked(
        cfg.clone(),
        b.device,
        &ModelOptions { slots: glm_vision_mtp_script::SLOTS, kv_blocks: 64, max_context: 8192, layers: LAYERS },
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
            let mut be = GlmBackend::with_spec(m, spec_cfg()).with_chunk(CHUNK);
            let r = glm_vision_mtp_script::run(&mut be, &p, |b: &GlmBackend, rows| b.m.logits_host(0, rows).unwrap());
            let stats = stats_key(&be.spec.as_ref().unwrap().stats);
            m = be.m;
            for l in &r.lines {
                out.push(format!("{name} graphs={graphs} {l}"));
            }
            out.push(format!("{name} graphs={graphs} keeps {:?}", r.keeps));
            out.push(format!("{name} graphs={graphs} stats {stats}"));
        }
    }
    // no drafting: the image path alone (every round's drafts are none)
    for graphs in [true, false] {
        m.set_graphs(graphs);
        let mut be = GlmBackend::new(m).with_chunk(CHUNK);
        let r = glm_vision_mtp_script::run(&mut be, &p, |b: &GlmBackend, rows| b.m.logits_host(0, rows).unwrap());
        m = be.m;
        for l in &r.lines {
            out.push(format!("nospec graphs={graphs} {l}"));
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
