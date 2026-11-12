//! The validated-pack gate every boot runs (`serve::check_pack_dirs`): only the operator's `--allow-unvalidated-pack` lets a pack outside the
//! validated list through, whatever family its config names (REXL3/REXL3#86). CPU only: toy packs of one tensor, no GPU.

use gb_model::fingerprint::{fingerprint, validated_pack};
use gb_model::pack::Pack;
use gb_model::serve::{check_pack_dirs, ServeOptions, UNVALIDATED_PACK_WARNING};
use std::path::PathBuf;
use std::sync::Mutex;

/// The boot facts are one table per process: the tests that write and read them back take turns.
static FACTS: Mutex<()> = Mutex::new(());

/// A fresh one-shard pack with the given `config.json`: a `model.safetensors` of one bf16 tensor (enough for `Pack::open` and a
/// fingerprint), so it is on no validated list.
fn toy_pack(tag: &str, config: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rexl3-pack-gate-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config).unwrap();
    let header = br#"{"x.weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
    let mut f = (header.len() as u64).to_le_bytes().to_vec();
    f.extend_from_slice(header);
    f.extend_from_slice(&[0, 0x3f, 0, 0x40]);
    std::fs::write(dir.join("model.safetensors"), f).unwrap();
    dir
}

fn fact(name: &str) -> Option<String> {
    gb_engine::boot::facts().into_iter().find(|(k, _)| *k == name).map(|(_, v)| v)
}

/// An unlisted DeepSeek-V4.1 or Qwen3.5-MoE pack, or one whose config only mentions those names in another field, is refused like any other
/// unless `--allow-unvalidated-pack` is given; with the flag it is served and reported as unvalidated (the facts then say the flag was given,
/// which is now only true when it was).
#[test]
fn an_unlisted_pack_is_refused_whatever_family_its_config_names_unless_allowed() {
    let _facts = FACTS.lock().unwrap_or_else(|e| e.into_inner());
    let configs = [
        (
            "unlisted-dsv41",
            r#"{"architectures":["DeepseekV41ForCausalLM"],"model_type":"deepseek_v41","quantization_config":{"quant_method":"exl3"}}"#,
        ),
        (
            "unlisted-qwen35moe",
            r#"{"architectures":["Qwen3_5MoeForConditionalGeneration"],"model_type":"qwen3_5_moe","quantization_config":{"quant_method":"exl3"}}"#,
        ),
        // the names anywhere else in the file (a path, a note) were enough before: they are just text
        (
            "mentions-both",
            r#"{"architectures":["NemotronH_Nano_Omni_Reasoning_V3"],"_name_or_path":"/models/deepseek_v41-and-qwen3_5_moe","quantization_config":{"quant_method":"exl3"}}"#,
        ),
    ];
    for (tag, config) in configs {
        let dir = toy_pack(tag, config);
        let fp = fingerprint(&Pack::open(&dir).unwrap().ck);
        assert_eq!(validated_pack(&fp, None), None, "{tag}: the toy pack is not on the list");
        let mut o = ServeOptions::new(dir.clone());
        let e =
            check_pack_dirs(&o).err().unwrap_or_else(|| panic!("{tag}: an unlisted pack passed without the flag")).to_string();
        assert!(e.contains(&dir.display().to_string()) && e.contains(&fp), "{tag}: {e}");
        assert!(e.contains("pass --allow-unvalidated-pack to serve it anyway"), "{tag}: {e}");
        o.allow_unvalidated_pack = true;
        check_pack_dirs(&o).unwrap_or_else(|e| panic!("{tag}: refused with the flag: {e}"));
        assert_eq!(fact("pack_validated").as_deref(), Some("no (--allow-unvalidated-pack)"), "{tag}");
        assert_eq!(fact("pack_warning").as_deref(), Some(UNVALIDATED_PACK_WARNING), "{tag}");
        assert_eq!(fact("target_fingerprint"), Some(fp), "{tag}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}

/// The gate takes no input from the pack's config text: besides the fingerprints, the operator's flag is all it reads.
#[test]
fn the_gate_reads_no_config_text() {
    let code = include_str!("../src/serve.rs").split("#[cfg(test)]").next().unwrap();
    let body = &code[code.find("pub fn check_pack_dirs(").unwrap()..code.find("pub fn check_packs(").unwrap()];
    assert!(!body.contains("config.json") && !body.contains("read_to_string"), "check_pack_dirs reads the config again");
    assert!(body.contains("check_packs(&target, draft.as_ref(), o.allow_unvalidated_pack)"), "the gate takes the flag alone");
}

/// The shipped DeepSeek-V4.1 recipes (`dsv41-single-262k`, `dsv41-tp2-1m`) boot their pack without `--draft` and without
/// `--allow-unvalidated-pack`, so they pass the gate by this list entry alone: the fingerprint their `model.lock.json` pins resolves, with no
/// draft, to the pack's name (the `pack_validated` fact their receipts record).
#[test]
fn the_shipped_deepseek_v41_pack_is_validated_without_a_draft_or_the_flag() {
    let shipped = "sha256:c34045423c126e40d051943ff46bd57a67c07350d8b1743ed5185415db61351c";
    assert_eq!(validated_pack(shipped, None), Some("vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw"));
}

/// A draft is listed only beside its own target (found by vcruz305 in #289): `DraftCheck::pair` and `DraftCheck::how`, which the GLM pack
/// check and the Flash-Next drafter line ask, name the (target, draft) pair; the same draft beside any other target, or a target off the
/// list, is structural, never another target's pair. The shipped pairs keep their text.
#[test]
fn a_listed_draft_names_its_pair_only_beside_its_own_target() {
    use gb_model::draft::DraftCheck;
    use gb_model::fingerprint::{
        FLASH_NEXT_DFLASH2_Q4_FP, GLM53_DFLASH2_FP, QFN305_FP, VALIDATED_FLASH_NEXT_DRAFT, VALIDATED_GLM53_DRAFT, VALIDATED_TEXT,
    };
    let glm = DraftCheck { fingerprint: GLM53_DFLASH2_FP.into(), geometry: "glm53-flash" };
    let (pair, t, _) = VALIDATED_GLM53_DRAFT[0];
    assert_eq!(glm.pair(t), Some(pair));
    assert_eq!(glm.how(t), "listed (GLM-5.3-Flash-EXL3-2.25bpw-sm121 + GLM-5.3-Flash-DFlash2 (bf16, as published))");
    let (_, redsnow) = *VALIDATED_TEXT.iter().find(|(n, _)| n.starts_with("vcruz305/RED-SNOW-5.3-FLASH")).unwrap();
    assert_eq!(glm.pair(redsnow), None);
    assert_eq!(glm.how(redsnow), "structural (glm53-flash)");
    assert_eq!(glm.how("sha256:unlisted"), "structural (glm53-flash)");
    let fnd = DraftCheck { fingerprint: FLASH_NEXT_DFLASH2_Q4_FP.into(), geometry: "qwen38-flash-next" };
    assert_eq!(fnd.how(QFN305_FP), format!("listed ({})", VALIDATED_FLASH_NEXT_DRAFT[0].0));
    assert_eq!(fnd.how(t), "structural (qwen38-flash-next)");
}

/// `check_packs` returns the target's fingerprint (the Flash-Next drafter line asks for its pair with it).
#[test]
fn the_pack_gate_returns_the_targets_fingerprint() {
    let _turn = FACTS.lock().unwrap_or_else(|e| e.into_inner());
    let dir = toy_pack("returns-fp", "{}");
    let pack = Pack::open(&dir).unwrap();
    assert_eq!(gb_model::serve::check_packs(&pack, None, true).unwrap(), fingerprint(&pack.ck));
    std::fs::remove_dir_all(dir).unwrap();
}
