//! The published Nemotron 3.5 Super VL pack, probed by the family. Gated on `REXL3_NEMO_OMNI_DIR` (the pack directory); without it the
//! test reports what it did not run and passes. The pack is ~97 GB; this reads its config and index only, never the weights.

use gb_model::families::nemotron_h_omni::{config::OmniConfig, expert_storage};
use gb_model::family::Registry;

#[test]
fn the_published_pack_probes_to_the_measured_plan_and_names_what_is_unbuilt() {
    let Some(dir) = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(std::path::PathBuf::from) else {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set (the published pack directory)");
        return;
    };
    let (family, plan) = Registry::global().probe(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let cfg = OmniConfig::from_json(&serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap())
        .unwrap();
    let storage = expert_storage(&dir, &cfg).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    assert_eq!(storage.bf16_pairs, vec![(26, 111), (39, 304), (81, 461)], "the measured bf16 expert pairs");
    assert_eq!(family.id(), "nemotron_h_omni");
    assert_eq!(plan.layers.len(), 88);
    assert_eq!(plan.mtp_layers.len(), 2);
    assert_eq!(plan.chat_format, "nemotron_omni");
    // the MTP head (the pack has it and `nemotron_mtp` is built), then the copy drafter: the family offers both since it builds its
    // engine (85b217e3, 2026-10-10), as the family's own unit test pins (`mod.rs`,
    // `a_checkpoint_with_the_expert_tensors_on_every_moe_layer_probes`)
    assert_eq!(plan.drafters, vec!["mtp", "copy"]);
    // and every component the engine needs is built (it was 5 unbuilt ones before the family built its engine)
    let missing = plan.missing_components();
    assert!(missing.is_empty(), "the text components are built: {missing:?}");
}
