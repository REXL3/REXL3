use gb_model::fingerprint::{pinned_shards, validated_pack, CF335, CF335_FP, FLASH_NEXT_DFLASH2_Q4_FP};

#[test]
fn the_published_335_pack_passes_the_identity_gate_without_an_override() {
    assert_eq!(validated_pack(CF335_FP, None), Some(CF335));
    assert_eq!(validated_pack(CF335_FP, Some("sha256:unlisted-draft")), None);
    assert_eq!(validated_pack("sha256:not-the-335-pack", None), None);

    let pins = pinned_shards(CF335_FP);
    assert_eq!(pins.len(), 9);
    assert!(pins.iter().all(|(name, sha, _)| {
        name.starts_with("model-")
            && name.ends_with("-of-00009.safetensors")
            && sha.len() == 64
            && sha.bytes().all(|b| b.is_ascii_hexdigit())
    }));
}

#[test]
fn the_335_pack_and_flash_next_dflash2_draft_are_a_named_validated_pair() {
    let name = validated_pack(CF335_FP, Some(FLASH_NEXT_DFLASH2_Q4_FP))
        .expect("CYBER-FROST 3.35 + Flash-Next DFlash2 boots without an override");
    assert!(name.contains("CYBER-FROST"));
    assert!(name.contains("DFlash2"));
}
