//! Public config-loader coverage using the shipped fixtures; the optional real Qwen pack uses `REXL3_QWEN_DIR`.
use gb_model::config::QwenConfig;
use gb_model::families::bailing_moe_v3::config::BailingConfig;
use gb_model::families::deepseek_v41::config::DeepseekV41Config;
use serde_json::Value;

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

fn fixture(path: &str) -> Value {
    let text =
        std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn all_shipped_config_fixtures_remain_valid() {
    assert!(QwenConfig::from_json(&fixture("qwen38/config.json")).is_ok());
    assert!(DeepseekV41Config::from_json(&fixture("deepseek_v41/config.json")).is_ok());
    assert!(BailingConfig::from_json(&fixture("ling31/config.json")).is_ok());
}

#[test]
fn public_loaders_fail_closed_on_malformed_configs() {
    let mut qwen = fixture("qwen38/config.json");
    qwen["text_config"]["num_key_value_heads"] = serde_json::json!(0);
    assert!(QwenConfig::from_json(&qwen).unwrap_err().contains("num_key_value_heads"));
    let mut qwen = fixture("qwen38/config.json");
    qwen["text_config"]["linear_num_key_heads"] = serde_json::json!(0);
    assert!(QwenConfig::from_json(&qwen).unwrap_err().contains("linear_num_key_heads"));
    let mut qwen = fixture("qwen38/config.json");
    qwen["text_config"]["eos_token_id"] = serde_json::json!([248044, "bad"]);
    assert!(QwenConfig::from_json(&qwen).unwrap_err().contains("eos_token_id"));

    let mut dsv = fixture("deepseek_v41/config.json");
    dsv["text_config"]["candidate_source_layer_id"] = serde_json::json!(40);
    assert!(DeepseekV41Config::from_json(&dsv).unwrap_err().contains("candidate_source_layer_id"));
    let mut dsv = fixture("deepseek_v41/config.json");
    dsv["text_config"]["engram_layer_ids"] = serde_json::json!([1, 40]);
    assert!(DeepseekV41Config::from_json(&dsv).unwrap_err().contains("engram_layer_ids"));

    let mut ling = fixture("ling31/config.json");
    ling["first_k_dense_replace"] = serde_json::json!("4");
    assert!(BailingConfig::from_json(&ling).unwrap_err().contains("first_k_dense_replace"));
    let mut ling = fixture("ling31/config.json");
    ling["eos_token_id"] = serde_json::json!([156895, null]);
    assert!(BailingConfig::from_json(&ling).unwrap_err().contains("eos_token_id"));
}
