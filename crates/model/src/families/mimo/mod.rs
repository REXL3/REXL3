//! MiMo-V2.6 (`MiMoV2ForCausalLM`): the probe, the memory model and the capability report (M1).
//!
//! This slice makes the family known to the registry: a MiMo pack is parsed, checked against the shapes the kernels
//! implement, planned (layer recipe, memory model, chat format, drafters) and refused at `build` with the list of
//! components that are not built yet, instead of being called an unknown architecture. Each component lands in its own
//! slice with its row-invariance test, golden hash and oracle comparison, and moves into `BUILT_COMPONENTS` in that
//! commit (M3-M8 of `docs/MIMO_V26_INTEGRATION_PLAN.md`):
//!
//! * `swa_sink_attention`: sliding-window attention with a learned per-head sink logit and a ring KV (nothing exists);
//! * `gqa_asymmetric_v128_q192`: full attention with V head dim 128 under QK head dim 192, partial rotary 64 and the
//!   `attention_value_scale` applied to V before the cache write;
//! * `moe_256_top8`: grouped EXL3 experts at 256 experts / top 8 / intermediate 2,048 (the built `moe` covers other
//!   shapes);
//! * `router_sigmoid_noaux_tc`: the sigmoid router with `e_score_correction_bias` and the +1e-20 renormalisation
//!   (`moe_router_dtype` is `bfloat16` in the pack's config; the reference code is the authority when the kernel lands);
//! * `rope_1m_two_thetas`: RoPE tables for 1,048,576 positions at theta 1e7 (full) and 1e4 (sliding);
//! * `mtp_x3_fp8`: the three MTP layers (FP8-e4m3 with 128x128 block scales, dequantised to bf16 at load; the weights
//!   live in the ORIGINAL XiaomiMiMo release beside the pack, not in the 2.20bpw EXL3 repo);
//! * `dflash_v1`: the DFlash (not DFlash2) drafter, also from the original release;
//! * `vision_mimo`/audio: out of scope for the text milestones, refused by name.
//!
//! The chat template is Qwen2-style BPE (`tokenizer.json`); the `mimo` format lands with the serial forward (M5).

pub mod config;
pub mod inventory;
pub mod memory;

use crate::error::ModelError;
use crate::family::{BuildCtx, FamilyError, FamilyPlan, MemorySource, ModelFamily};
use crate::memory::MemoryOptions;
use config::MiMoConfig;
use gb_engine::ServeEngine;
use gb_sched::{MemoryError, MemoryProfile};
use serde_json::Value;
use std::path::Path;

/// Everything a MiMo engine runs on. The built ones are shared with the other families; the unbuilt are MiMo's own.
pub const REQUIRED: &[&str] = &[
    "exl3_linear",
    "swa_sink_attention",
    "gqa_asymmetric_v128_q192",
    "moe_256_top8",
    "router_sigmoid_noaux_tc",
    "rope_1m_two_thetas",
    "mtp_x3_fp8",
    "dflash_v1",
    "vision_mimo",
    "audio_mimo",
];

pub struct MiMoFamily;

impl ModelFamily for MiMoFamily {
    fn id(&self) -> &'static str {
        "mimo_v2"
    }
    fn architectures(&self) -> &'static [&'static str] {
        &["MiMoV2ForCausalLM"]
    }
    fn model_types(&self) -> &'static [&'static str] {
        &["mimo_v2"]
    }

    fn probe(&self, dir: &Path) -> Result<FamilyPlan, FamilyError> {
        let root = read_config(dir)?;
        crate::family::require_exl3(&root)?;
        let cfg = MiMoConfig::from_json(&root).map_err(FamilyError::Config)?;
        cfg.check_shapes().map_err(FamilyError::Config)?;
        Ok(FamilyPlan {
            family: self.id(),
            dir: dir.to_path_buf(),
            layers: cfg.layer_specs(),
            mtp_layers: cfg.mtp_specs(),
            memory: Box::new(MiMoMemorySource(cfg)),
            vision: None,                  // the 28-block tower is M9; the pack's text loader never runs it either
            chat_format: "mimo",           // the format module lands with the serial forward (M5); the registry accepts the id
            drafters: vec!["mtp", "copy"], // dflash_v1 joins at M8
            required_components: REQUIRED.to_vec(),
        })
    }

    fn build(&self, plan: FamilyPlan, _ctx: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError> {
        // nothing here runs until every component is built; the engine lands with the last of them
        Err(plan.not_built("MiMo-V2.6 engine"))
    }
}

fn read_config(dir: &Path) -> Result<Value, FamilyError> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| FamilyError::Config(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| FamilyError::Config(format!("config.json: {e}")))
}

/// The memory model of a MiMo pack, ready to be evaluated at any serving settings.
pub struct MiMoMemorySource(pub MiMoConfig);

impl MemorySource for MiMoMemorySource {
    fn profile(&self, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
        memory::memory_profile(&self.0, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::config::tests::MIMO_CONFIG;
    use super::*;
    use crate::family::{Ffn, Mixer, Registry, Residual};

    fn dir_with(name: &str, cfg: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("rexl3-mimo-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), cfg).unwrap();
        dir
    }

    #[test]
    fn the_registry_sends_a_mimo_pack_here_and_the_plan_names_what_is_missing() {
        let dir = dir_with("probe", MIMO_CONFIG);
        let (fam, plan) = Registry::global().probe(&dir).unwrap();
        assert_eq!((fam.id(), plan.family), ("mimo_v2", "mimo_v2"));
        assert_eq!((plan.layers.len(), plan.mtp_layers.len()), (48, 3));
        assert_eq!(plan.layers.iter().filter(|l| l.mixer == Mixer::Gqa).count(), 9);
        assert_eq!(plan.layers.iter().filter(|l| l.mixer == Mixer::Gdn).count(), 39);
        assert_eq!(plan.layers.iter().filter(|l| l.ffn == Ffn::Moe).count(), 47);
        assert_eq!(plan.layers.iter().filter(|l| l.ffn == Ffn::Dense).count(), 1);
        assert!(plan.layers.iter().all(|l| l.residual == Residual::Plain));
        assert_eq!(plan.chat_format, "mimo");
        assert_eq!(plan.drafters, vec!["mtp", "copy"]);
        // every MiMo-specific component is named as missing, in order, with the shared built ones absent
        assert_eq!(
            plan.missing_components(),
            vec![
                "swa_sink_attention",
                "gqa_asymmetric_v128_q192",
                "moe_256_top8",
                "router_sigmoid_noaux_tc",
                "rope_1m_two_thetas",
                "mtp_x3_fp8",
                "dflash_v1",
                "vision_mimo",
                "audio_mimo"
            ]
        );
        // the plan's memory model evaluates at serving settings
        let p = plan.memory_profile(&MemoryOptions::serving_defaults()).unwrap();
        assert_eq!(p.family(), "mimo_v2");
        // and build refuses by naming every missing component
        let o = crate::serve::ServeOptions::new(dir.clone());
        let e = match MiMoFamily.build(plan, &BuildCtx { options: &o }) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("built an engine with components missing"),
        };
        assert!(e.contains("MiMo-V2.6 engine") && e.contains("swa_sink_attention") && e.contains("rope_1m_two_thetas"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_non_exl3_pack_is_refused_at_the_probe() {
        let mut v: Value = serde_json::from_str(MIMO_CONFIG).unwrap();
        v["quantization_config"]["quant_method"] = serde_json::json!("gguf");
        let dir = dir_with("gguf", &v.to_string());
        let e = Registry::global().probe(&dir).err().unwrap();
        assert!(format!("{e:?}").contains("EXL3"), "{e:?}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_misshapen_config_is_refused_by_name_through_the_registry() {
        let mut v: Value = serde_json::from_str(MIMO_CONFIG).unwrap();
        v["swa_num_key_value_heads"] = serde_json::json!(4);
        let dir = dir_with("shape", &v.to_string());
        let e = Registry::global().probe(&dir).err().unwrap();
        assert!(format!("{e:?}").contains("swa_num_key_value_heads 4"), "{e:?}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
