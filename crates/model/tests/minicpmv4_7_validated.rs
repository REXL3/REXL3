//! The validated list for MiniCPM-V-4.7 (E7 of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`): a pack the list names is served without
//! `--allow-unvalidated-pack` and reported as validated, and a pack it does not name is refused by name (its fingerprint) with the
//! operator's flag as the way through. Adding the finalist's entry (`scripts/minicpmv47_fingerprint.sh`, section 16) is exercised by
//! the same tests: nothing here names a pack, they read the list.
//!
//! * `REXL3_MINICPM_DIR`: the target pack that should be listed (the shipping pack); `REXL3_MINICPM_DRAFT_DIR`: its draft pack (the
//!   gate's name; `REXL3_DSPARK_DIR` is read when it is unset). A pack that is not listed makes the first test fail, naming what to paste.
//! * `REXL3_MINICPM_UNLISTED_DIR` (optional): any other pack of the family (a ladder pack or the 8-layer smoke pack) for the refusal leg;
//!   `REXL3_MINICPM_UNLISTED_DRAFT_DIR` (optional): another draft, to refuse a listed target with the wrong draft.
//! * The device test (`#[ignore]`, GPU, shared lock) boots the listed target with the flag off and reads `report.validated`.
//!
//! ```text
//! REXL3_NO_CUDA=1 REXL3_MINICPM_DIR=PACK REXL3_MINICPM_DRAFT_DIR=DRAFT cargo test --release -p gb-model --test minicpmv4_7_validated
//! flock -s -w 1800 /tmp/rexl3-gpu.lock env REXL3_MINICPM_DIR=PACK cargo test --release -p gb-model --test minicpmv4_7_validated -- --ignored --nocapture
//! ```
use gb_exl3::checkpoint::Checkpoint;
use gb_model::families::minicpmv4_7::boot::{boot_with, BootExtras};
use gb_model::families::minicpmv4_7::config::MiniCpmConfig;
use gb_model::fingerprint::{fingerprint, validated_pack, validation_line, VALIDATED_MINICPMV4_7};
use gb_model::pack::Pack;
use gb_model::serve::{check_packs, ServeOptions};
use std::path::PathBuf;

fn env_dir(names: &[&str]) -> Option<PathBuf> {
    names.iter().find_map(|n| std::env::var(n).ok().map(PathBuf::from).filter(|d| d.join("config.json").exists()))
}

fn fp(dir: &std::path::Path) -> String {
    fingerprint(&Pack::open(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())).ck)
}

#[test]
fn a_listed_pair_passes_the_check_without_the_flag_and_an_unlisted_one_is_refused_by_name() {
    // a synthetic pack needs no files: it is unlisted by construction, and the refusal names it by its fingerprint
    let mut ck = Checkpoint::default();
    ck.config = serde_json::json!({"architectures": ["MiniCPMV4_7ForConditionalGeneration"], "synthetic": "e7"});
    let synthetic = Pack { ck };
    let sfp = fingerprint(&synthetic.ck);
    assert_eq!(validated_pack(&sfp, None), None);
    let e = check_packs(&synthetic, None, false).unwrap_err().to_string();
    assert!(
        e.contains("not one this release was validated on") && e.contains(&sfp) && e.contains("--allow-unvalidated-pack"),
        "{e}"
    );
    check_packs(&synthetic, None, true).expect("the operator's flag lets an unlisted pack through");

    let Some(target) = env_dir(&["REXL3_MINICPM_DIR"]) else {
        return eprintln!("not run (real packs): REXL3_MINICPM_DIR is not set to a pack");
    };
    let draft = env_dir(&["REXL3_MINICPM_DRAFT_DIR", "REXL3_DSPARK_DIR"]);
    let tfp = fp(&target);
    let name = validated_pack(&tfp, None).unwrap_or_else(|| {
        panic!(
            "{} (fingerprint {tfp}) is not in VALIDATED_MINICPMV4_7: run scripts/minicpmv47_fingerprint.sh {} DRAFT and paste both blocks \
             (docs/MINICPMV4_7_INTEGRATION_PLAN.md section 16); the list holds {} entr(ies)",
            target.display(),
            target.display(),
            VALIDATED_MINICPMV4_7.len()
        )
    });
    let tpack = Pack::open(&target).unwrap();
    check_packs(&tpack, None, false).unwrap_or_else(|e| panic!("{name}: listed and refused without the flag: {e}"));
    assert_eq!(validation_line(&tfp, None), format!("validated {name}"));
    eprintln!("target {tfp} is {name}");

    if let Some(draft) = &draft {
        let dfp = fp(draft);
        let dpack = Pack::open(draft).unwrap();
        assert_eq!(validated_pack(&tfp, Some(&dfp)), Some(name), "the draft {dfp} is not the listed draft of {name}");
        check_packs(&tpack, Some(&dpack), false)
            .unwrap_or_else(|e| panic!("{name} + draft: listed and refused without the flag: {e}"));
        assert_eq!(validation_line(&tfp, Some(&dfp)), format!("validated {name}"));
        // the pair is the pair: the target with the draft in the target's place, and the draft alone, are not entries
        assert_eq!(validated_pack(&dfp, Some(&tfp)), None);
        assert_eq!(validated_pack(&dfp, None), None);
        let e = check_packs(&dpack, None, false).unwrap_err().to_string();
        assert!(e.contains(&dfp), "a draft as the target is refused by its fingerprint: {e}");
        eprintln!("draft  {dfp} completes it");
    } else {
        eprintln!("not run (the pair): REXL3_MINICPM_DRAFT_DIR is not set");
    }

    // another pack of the family, and the listed target with another draft: refused by name, and let through by the flag
    if let Some(other) = env_dir(&["REXL3_MINICPM_UNLISTED_DIR"]) {
        let ofp = fp(&other);
        assert_ne!(ofp, tfp, "REXL3_MINICPM_UNLISTED_DIR is the listed pack itself");
        assert_eq!(validated_pack(&ofp, None), None, "{} is listed after all", other.display());
        let opack = Pack::open(&other).unwrap();
        let e = check_packs(&opack, None, false).unwrap_err().to_string();
        assert!(e.contains(&ofp) && e.contains("--allow-unvalidated-pack"), "{e}");
        check_packs(&opack, None, true).unwrap();
        assert_eq!(validation_line(&ofp, None), "validated no (serving needs --allow-unvalidated-pack)");
    }
    if let Some(other) = env_dir(&["REXL3_MINICPM_UNLISTED_DRAFT_DIR"]) {
        let ofp = fp(&other);
        if validated_pack(&tfp, Some(&ofp)).is_none() {
            let opack = Pack::open(&other).unwrap();
            let e = check_packs(&tpack, Some(&opack), false).unwrap_err().to_string();
            assert!(e.contains(&tfp) && e.contains(&ofp), "the refusal names both fingerprints: {e}");
        }
    }
}

/// The boot itself, with the flag off: a listed target gets through the validation and reports its name. The whole boot (25 GiB of
/// weights, the state, the engine's reserve) is the device test of E2; this one reads only the verdict.
#[test]
#[ignore]
fn a_listed_pack_boots_without_the_flag_and_reports_validated() {
    let Some(dir) = env_dir(&["REXL3_MINICPM_DIR"]) else { return eprintln!("not run: REXL3_MINICPM_DIR is not set to a pack") };
    if !gb_cuda::device_available() {
        return eprintln!("skipping: no CUDA device");
    }
    let cfg = MiniCpmConfig::from_dir(&dir).unwrap();
    let mut o = ServeOptions::new(dir.clone());
    o.slots = 1;
    o.slots_auto = false;
    o.allow_unvalidated_pack = false;
    o.prefix_cache = gb_model::serve::PrefixCacheOption::Off;
    if let Some(n) = std::env::var("REXL3_MINICPM_KV_TOKENS").ok().and_then(|v| v.parse::<usize>().ok()) {
        o.kv_tokens = Some(n);
    }
    let want = validated_pack(&fp(&dir), None).expect("the pack is in VALIDATED_MINICPMV4_7 (see the header-only test)");
    let m = boot_with(&o, BootExtras { allow_truncated_layers: cfg.layers < 40 })
        .unwrap_or_else(|e| panic!("a listed pack did not boot without --allow-unvalidated-pack: {e}"));
    assert_eq!(m.report.validated, Some(want));
    eprintln!("BOOT {}", m.report.line());
}
