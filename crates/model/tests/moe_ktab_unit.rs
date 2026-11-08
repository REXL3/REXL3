//! The fused MoE kernels (CoopMK, FrostMoE) read every `ktab` entry as K2, twice the bits a value. A family that stored plain
//! bits would have its experts decoded at half their width on those paths, as Qwen3.8-Flash-Next and MiniCPM-V-4.7 were when
//! the kernels moved to K2: tool-call prompts came back as garbage and speculative drafts were rejected.

const FAMILIES: &[(&str, &str)] = &[
    ("qwen_flash_next", include_str!("../src/families/qwen_flash_next/device.rs")),
    ("minicpmv4_7", include_str!("../src/families/minicpmv4_7/device.rs")),
    ("bailing_moe_v3", include_str!("../src/families/bailing_moe_v3/device.rs")),
];

#[test]
fn every_family_stores_its_fused_moe_ktab_entries_as_twice_the_bits() {
    for (family, src) in FAMILIES {
        let pushes: Vec<&str> = src.lines().map(str::trim).filter(|l| l.starts_with("kt.push(")).collect();
        assert!(!pushes.is_empty(), "{family}: no ktab entry found");
        for line in &pushes {
            assert!(
                line.contains("ktab_k2(") || line.contains(".twice()"),
                "{family}: a fused-MoE ktab entry is not K2 (twice the bits): {line}"
            );
        }
    }
}
