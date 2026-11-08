//! The model crate's family plans and the server's chat-format registry are two lists; this keeps them from drifting apart: every
//! shipped family's plan must name a chat format the server registers, and a new family must come with a fixture here.
use gb_model::family::Registry;
use gb_server::chat_format::FormatRegistry;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join(name)
}

/// The Nemotron fixture directory holds a trimmed index; the family's probe proves EXL3 from the per-expert tensors of every MoE layer, so
/// this builds a directory with the fixture's config and an index that names them (no shards are read by a probe).
fn nemotron_probe_dir() -> PathBuf {
    use gb_model::families::nemotron_h_omni::config::OmniConfig;
    let src = fixture("nemotron_omni");
    let text = std::fs::read_to_string(src.join("config.json")).expect("the nemotron fixture config");
    let cfg = OmniConfig::from_json(&serde_json::from_str(&text).expect("config json"))
        .expect("the fixture config is the published one");
    let dir = std::env::temp_dir().join(format!("rexl3-family-formats-nemotron-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), &text).unwrap();
    let mut map = serde_json::Map::new();
    for (layer, kind) in cfg.pattern.chars().enumerate() {
        if kind != 'E' {
            continue;
        }
        for expert in 0..cfg.experts {
            for proj in ["up_proj", "down_proj"] {
                let key = format!("language_model.backbone.layers.{layer}.mixer.experts.{expert}.{proj}.trellis");
                map.insert(key, serde_json::Value::from("model-00001-of-00001.safetensors"));
            }
        }
    }
    let index = serde_json::json!({ "metadata": {}, "weight_map": map });
    std::fs::write(dir.join("model.safetensors.index.json"), index.to_string()).unwrap();
    dir
}

#[test]
fn every_shipped_family_names_a_registered_chat_format() {
    let registry = Registry::global();
    assert_eq!(registry.families().len(), 8, "a new family needs a config fixture dir listed in this test");
    let formats = FormatRegistry::builtin();
    for dir in ["qwen38", "glm53", "qwen_flash_next", "mimo", "nemotron_omni", "minicpmv4_7", "deepseek_v41", "ling31-flash-1019"]
    {
        let path = if dir == "nemotron_omni" { nemotron_probe_dir() } else { fixture(dir) };
        let (family, plan) = registry.probe(&path).unwrap_or_else(|e| panic!("{dir}: {e}"));
        assert!(
            formats.by_id(plan.chat_format).is_ok(),
            "family {} names the chat format `{}`, which the server does not register",
            family.id(),
            plan.chat_format
        );
        if dir == "qwen_flash_next" {
            assert!(plan.missing_components().is_empty(), "Flash-Next text must build");
            assert_eq!(plan.unbuilt_vision(), Some("vision_flash_next"));
            assert!(family.prefix_cache(), "Flash-Next prefix hooks must be wired");
        }
    }
}
