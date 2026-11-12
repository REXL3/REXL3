//! The real pack's config. Set REXL3_QWEN_DIR (tests skip without it).
use gb_model::config::QwenConfig;

#[test]
fn real_pack_config_parses() {
    let Some(d) = std::env::var("REXL3_QWEN_DIR").ok().map(std::path::PathBuf::from).filter(|p| p.join("config.json").exists())
    else {
        return eprintln!("REXL3_QWEN_DIR not set: skipping");
    };
    let c = QwenConfig::from_dir(&d).unwrap();
    assert_eq!((c.hidden, c.intermediate, c.vocab, c.layers.len()), (5120, 17408, 248320, 64));
    assert_eq!((c.full_layers(), c.rotary_dim, c.rope_theta as u64), (16, 64, 10_000_000));
    assert_eq!(c.eos, vec![248044]);
}
