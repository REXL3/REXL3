//! The intervention overlay path through the crate's public surface: a direction of another hidden width and a
//! cut-short direction file are load errors, and a start with an intervention directory is refused because no
//! forward applies an overlay in this build. Reader-level cases are unit tests in `src/intervention.rs`.

use gb_model::family::{FamilyPlan, Registry};
use gb_model::intervention::{hidden_width, Intervention};
use gb_model::serve::{self, ServeOptions, INTERVENTION_NOT_APPLIED};
use std::path::{Path, PathBuf};

const MIMO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/mimo");
const DIRECTION: &str = "directions/layer025-post_layer_outputs.pt";
const FILES: [&str; 3] = ["intervention.json", DIRECTION, "directions/pass2-remaining21-layer033-post_layer_outputs.pt"];

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rexl3-iv-public-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The MiMo fixture's plan, probed from a copy that holds only what the registry reads.
fn plan(tag: &str) -> FamilyPlan {
    let dir = scratch(tag);
    for name in ["config.json", "chat_template.jinja"] {
        if let Ok(bytes) = std::fs::read(Path::new(MIMO).join(name)) {
            std::fs::write(dir.join(name), bytes).unwrap();
        }
    }
    let (_, plan) = Registry::global().probe(&dir).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    plan
}

fn sha256_hex(bytes: &[u8]) -> String {
    <sha2::Sha256 as sha2::Digest>::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn a_direction_is_checked_against_the_width_the_pack_config_declares() {
    let overlay = Path::new(MIMO).join("intervention");
    let plan = plan("width");
    let hidden = hidden_width(Path::new(MIMO)).unwrap();
    assert_eq!(hidden, 4096);
    assert_eq!(Intervention::load(&overlay, &plan, hidden).unwrap().entries.len(), 12);
    let e = Intervention::load(&overlay, &plan, 5120).err().unwrap();
    assert!(e.contains("4096 wide") && e.contains("hidden width is 5120"), "{e}");
}

/// The manifest pin is rewritten to match each cut, so the hash check passes and the file readers see the cut.
#[test]
fn a_direction_file_cut_short_is_a_load_error_at_every_cut_of_its_tail() {
    let plan = plan("cut-plan");
    let dir = scratch("cut");
    for f in FILES {
        std::fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        std::fs::copy(Path::new(MIMO).join("intervention").join(f), dir.join(f)).unwrap();
    }
    let whole = std::fs::read(dir.join(DIRECTION)).unwrap();
    let manifest = std::fs::read_to_string(dir.join("intervention.json")).unwrap();
    for cut in [0, 3, whole.len() / 2].into_iter().chain(whole.len() - 400..whole.len()) {
        let bytes = &whole[..cut];
        std::fs::write(dir.join(DIRECTION), bytes).unwrap();
        let mut root: serde_json::Value = serde_json::from_str(&manifest).unwrap();
        root["directions"][0]["sha256"] = serde_json::json!(sha256_hex(bytes));
        std::fs::write(dir.join("intervention.json"), root.to_string()).unwrap();
        let e = Intervention::load(&dir, &plan, 4096).err().unwrap_or_else(|| panic!("the file cut at {cut} loaded"));
        assert!(e.contains(DIRECTION) && !e.contains("hash mismatch"), "cut {cut}: {e}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_start_with_an_intervention_directory_is_refused_before_the_pack_is_read() {
    // neither path exists: the refusal comes before the registry probes the model
    let mut o = ServeOptions::new(PathBuf::from("/nonexistent/model"));
    o.intervention_dir = Some(Path::new(MIMO).join("intervention"));
    let e = serve::start(&o).err().unwrap().to_string();
    assert!(e.contains(INTERVENTION_NOT_APPLIED), "{e}");
    let e = serve::start_follower(&o).unwrap_err().to_string();
    assert!(e.contains(INTERVENTION_NOT_APPLIED), "{e}");
    assert!(INTERVENTION_NOT_APPLIED.starts_with("--intervention is not applied in this build"));
    o.intervention_dir = None;
    assert!(serve::refuse_unapplied_intervention(&o).is_ok());
}
