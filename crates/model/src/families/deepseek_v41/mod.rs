//! The DeepSeek-V4.1-Flash family (`DeepseekV41ForCausalLM`): config, inventory, weights, plan, load, device, forward, backend and engine.
//!
//! The pieces, in the order a boot uses them:
//!
//! * [`config::DeepseekV41Config`] reads every field of `text_config` and validates the relationships between them (the 43
//!   `compress_ratios` for 40 trunk layers and 3 NextN layers, the two source-layer lists, the engram tables' sizes).
//! * [`inventory::PackInventory::from_rows`] generates every tensor name the config implies, with its dtype and shape, and
//!   refuses a pack that is missing one, has one it does not imply, or has one of the wrong shape — naming the tensors.
//! * [`weights`] is that inventory as a typed tree, and [`bind`] builds it.
//! * [`plan`] classifies every tensor (resident / expert-cache / `mmap` sidecar) and prices admission; [`alloc`] is the arithmetic
//!   each class is charged from.
//! * [`load`] turns the classification into addresses and loads it through the pipelined loader; [`device`] binds the kernel types
//!   over those regions; [`ops`] is the forward as an ordered op list and [`model`] is the executor that walks it.
//! * [`tp`] and [`tp_runtime`] implement the 2-rank lockstep tensor-parallel runtime over RoCEv2 (`RoceTp`) and NCCL (`NcclTp`),
//!   with attention heads split `[32, 32]`, routed experts partitioned `[0..192]` / `[192..384]`, NextN (`DSpark`) replicated on
//!   both ranks for zero-desync speculative drafting, and prefix-cache snapshot/restore replayed in lockstep.
//! * [`backend`] and [`engine`] wire the model into `gb_engine`: [`DeepseekV41Family::build`] maps the pack, runs admission (refusing a
//!   pack that does not fit, by name), loads it, and starts the single-rank or lockstep TP=2 engine.
//!
//! All capabilities are bound and verified (**COMPLETE** serve profile): text generation, vision splicing (`vision::VisionTower`),
//! NextN (`DSpark`) speculative drafting (`model.rs::propose_mtp` behind `SpecMode::Auto|Mtp`, with draft trees under
//! `REXL3_DSV41_SPEC_TREE` — see [`tree`]), prefix caching (`Dsv41SnapPool` behind `prefix_caps`), 1M YaRN long-context (`CED`
//! bounded replay + `P1/P2/P3` candidate kernels), and 100% (`15,360 / 15,360`) resident routed experts on both TP=2 and Single-Rank.

pub mod alloc;
pub mod backend;
pub mod bind;
pub mod config;
pub mod device;
pub mod engine;
pub mod engram;
pub mod inventory;
pub mod load;
pub mod memory;
pub mod model;
pub mod ops;
pub mod plan;
pub mod spec;
pub mod tp;
pub mod tp_arm;
pub mod tp_boot;
pub mod tp_runtime;
pub mod tree;
pub mod vision;
pub mod vision_serve;
pub mod weights;

pub use config::{DeepseekV41Config, MODEL_TYPE, MODEL_TYPE_TEXT};
pub use inventory::{Component, InventoryError, PackInventory, TensorInfo};
pub use memory::DeepseekV41MemorySource;

use crate::error::ModelError;
use crate::family::{BuildCtx, FamilyError, FamilyPlan, ModelFamily, VisionSpec};
use gb_engine::ServeEngine;
use std::path::Path;

/// The components this family requires in [`crate::family::BUILT_COMPONENTS`]. All seven text/MTP components plus
/// `vision_deepseek` and `chat_deepseek_v41` are built and verified.
const REQUIRED: &[&str] = &[
    "exl3_linear",
    // an FP8 block projection with `F8_E8M0` scales, on every dense path of the model
    "mla_fp8",
    // the compressed KV: overlapping and non-overlapping compressors, their gates and their pool
    "dsa_compressor",
    // the sparse-attention indexer: query and key projections, the block scores and the top-k walk
    "dsa_index",
    // the `sqrtsoftplus` / `noaux_tc` router with its correction bias
    "moe_noaux_tc",
    // mHC hyper-connections: four residual streams through a Sinkhorn-normalised matrix
    "hc_streams",
    // the engram token-lookup tables on layers 1 and 14
    "engram",
    // the three NextN layers with the DSpark Markov head and confidence head
    "mtp_dspark",
    // the vision tower is a BUILT component (`BUILT_COMPONENTS::vision_deepseek`), not a required one: a family can
    // serve text before its tower exists (`FamilyPlan::unbuilt_vision` names it, the server's vision gate refuses
    // an image serve without it, and `--no-vision` serves text without it)
];

/// The chat format this family names: `deepseek_v41` (native prompt renderer + DSML tool-call parser, with bundled
/// `testdata/deepseek_v41/chat_template.jinja` fallback).
pub const CHAT_FORMAT: &str = "deepseek_v41";

pub struct DeepseekV41Family;

impl ModelFamily for DeepseekV41Family {
    fn id(&self) -> &'static str {
        "deepseek_v41"
    }

    fn architectures(&self) -> &'static [&'static str] {
        &[config::ARCH]
    }

    fn model_types(&self) -> &'static [&'static str] {
        &[MODEL_TYPE, MODEL_TYPE_TEXT]
    }

    /// The REXL3 pack (`vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw`) carries an FP8
    /// `quantization_config` (`quant_method=fp8`, dynamic activation scheme, `[32, 32]`
    /// block weights, UE8M0 scales, FP4 experts) instead of the EXL3 one, so the shared
    /// EXL3 config check refuses it. The family proves the exact EXL3 + FP8 + Engram
    /// tensor inventory from the safetensors headers ([`inventory::PackInventory::from_rows`]
    /// refuses any non-EXL3 pack by tensor name and shape), so it opts out of the
    /// config-based check — exactly like `NemotronOmniFamily` does.
    fn quantization_from_tensors(&self) -> bool {
        true
    }

    fn probe(&self, dir: &Path) -> Result<FamilyPlan, FamilyError> {
        let cfg = DeepseekV41Config::from_dir(dir).map_err(FamilyError::Config)?;
        let vision = cfg.vision.as_ref().map(|v| VisionSpec {
            tower: "vision_deepseek",
            patch_size: v.patch_size,
            merge_size: v.downsample_ratio,
            temporal_patch_size: 1,
            out_hidden: cfg.hidden,
            image_token_id: cfg.image_token_id,
            max_image_tokens: v.max_image_tokens,
        });
        let drafters = if cfg.nextn > 0 { vec!["mtp", "copy"] } else { vec!["copy"] };
        Ok(FamilyPlan {
            family: self.id(),
            dir: dir.to_path_buf(),
            layers: cfg.layer_specs(),
            mtp_layers: cfg.mtp_specs(),
            memory: Box::new(DeepseekV41MemorySource(cfg)),
            vision,
            chat_format: CHAT_FORMAT,
            drafters,
            required_components: REQUIRED.to_vec(),
        })
    }

    fn prefix_cache(&self) -> bool {
        true
    }

    /// Start the engine: map the pack, plan and admit it, load it through the pipelined loader and wire the backend in
    /// ([`engine::build_engine`]). A pack that does not fit is refused by name before anything is allocated.
    ///
    /// A tensor-parallel group (`--tp-ranks 2 --tp-rank 0`) is validated by [`tp::check_tp_plain`] first, binds its
    /// rendezvous listener BEFORE the scan (`tp_runtime::listen`), runs the Hello gate and collective bootstrap
    /// (`RoceTp` / `NcclTp`), loads the rank's shard (`ShardTable`, with NextN replicated on both ranks when DSpark
    /// is enabled), and builds the lockstep group engine ([`engine::build_engine_group`]). Rank 1 of a group is
    /// routed by the server to [`ModelFamily::run_follower`].
    fn build(&self, plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError> {
        let _ = plan;
        let o = ctx.options;
        let tp = o.tp.as_ref().filter(|t| t.is_group());
        if let Some(t) = tp {
            tp::check_tp_plain(o)?;
            let listener = crate::lockstep::listen(&t.peer)?;
            return Ok(Box::new(engine::build_engine_group(o, listener)?));
        }
        Ok(Box::new(engine::build_engine(ctx.options)?))
    }

    /// The pack's preprocessing (`gb_vision::dsv4`): the pack's `vision_config` numbers with the served token cap.
    /// `None` for a pack with no tower (a text-only pack), whose image requests the vision gate refuses anyway.
    fn image_preprocessor(
        &self,
        dir: &Path,
        o: &crate::serve::ServeOptions,
    ) -> Result<Option<std::sync::Arc<dyn gb_vision::preprocessor::ImagePreprocessor>>, ModelError> {
        image_preprocessor(dir, o)
    }

    /// The arms' combine is a pairwise byte exchange: a two-rank group is what has been verified and measured.
    fn max_tp_ranks(&self) -> usize {
        2
    }

    /// A follower (rank 1 of `--tp-ranks 2`): plans from its own memory, connects, is gated, loads its shard with
    /// rank 0's planned geometry and replays rank 0's calls until the group ends ([`tp_boot::follow`]).
    fn run_follower(&self, plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<(), ModelError> {
        tp_boot::follow(plan, ctx)
    }
}

/// The preprocessing of the DeepSeek-V4.1-Flash pack at `dir`, for these options: the `vision_config`'s
/// `min_pixels` and aspect clamp, with a per-image token cap of the smallest of the operator's
/// `--max-image-tokens`, the pack's `vision_config.max_image_tokens` and the deployment's 1,000.
pub fn image_preprocessor(
    dir: &Path,
    o: &crate::serve::ServeOptions,
) -> Result<Option<std::sync::Arc<dyn gb_vision::preprocessor::ImagePreprocessor>>, ModelError> {
    let cfg = DeepseekV41Config::from_dir(dir).map_err(FamilyError::Config)?;
    let Some(v) = cfg.vision else { return Ok(None) };
    let cap = o.max_image_tokens.min(v.max_image_tokens).min(gb_cuda::dsv4_vision::MAX_IMAGE_TOKENS);
    Ok(Some(std::sync::Arc::new(gb_vision::dsv4::Dsv4Preprocessing::new(v.min_pixels, cap, v.max_wh_ratio))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::Registry;
    use crate::serve::ServeOptions;
    use std::path::PathBuf;

    const CONFIG: &str = include_str!("../../../../../testdata/deepseek_v41/config.json");

    fn pack_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rexl3-dsv41-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), CONFIG).unwrap();
        dir
    }

    #[test]
    fn the_probe_describes_the_pack_in_family_neutral_terms() {
        let dir = pack_dir("probe");
        let plan = DeepseekV41Family.probe(&dir).unwrap();
        assert_eq!((plan.family, plan.chat_format), ("deepseek_v41", CHAT_FORMAT));
        assert_eq!((plan.layers.len(), plan.mtp_layers.len()), (40, 3));
        assert_eq!(plan.drafters, vec!["mtp", "copy"], "the pack's three NextN layers and the copy drafter");
        assert_eq!(plan.required_components, REQUIRED);
        let missing = plan.missing_components();
        assert!(missing.is_empty(), "{missing:?}");
        assert_eq!(plan.unbuilt_vision(), None, "the tower is built: images are served, not refused");
        // the memory model prices a token from the config alone
        let profile = plan.memory_profile(&crate::memory::MemoryOptions::serving_defaults()).unwrap();
        assert_eq!(profile.kv_bytes_per_token(), 71_680);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A pack folder with a config and no shards: the build gets as far as the inventory, and refuses by naming what is missing.
    #[test]
    fn build_refuses_a_pack_with_no_shards_and_names_what_is_missing() {
        let dir = pack_dir("build");
        let plan = DeepseekV41Family.probe(&dir).unwrap();
        let o = ServeOptions::new(dir.clone());
        let e = match DeepseekV41Family.build(plan, &BuildCtx { options: &o }) {
            Ok(_) => panic!("a pack with no shards must not build an engine"),
            Err(e) => e,
        };
        let text = e.to_string();
        assert!(text.contains("missing") && text.contains("tensors"), "{text}");
        assert!(text.contains("aligner.") || text.contains("layers.0."), "{text}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_registry_finds_the_family_by_architecture_and_by_model_type() {
        let dir = pack_dir("registry");
        let (family, _) = Registry::global().probe(&dir).unwrap();
        assert_eq!(family.id(), "deepseek_v41");
        // by model_type, when a config has no `architectures`
        let mut v: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
        v.as_object_mut().unwrap().remove("architectures");
        std::fs::write(dir.join("config.json"), v.to_string()).unwrap();
        let (family, _) = Registry::global().probe(&dir).unwrap();
        assert_eq!(family.id(), "deepseek_v41");
        // and the text model type alone selects it too
        v["model_type"] = serde_json::json!(MODEL_TYPE_TEXT);
        std::fs::write(dir.join("config.json"), v.to_string()).unwrap();
        assert_eq!(Registry::global().probe(&dir).unwrap().0.id(), "deepseek_v41");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_config_that_is_not_this_family_is_not_probed_by_it() {
        // the GLM config has no `compress_ratios`: this family's parser refuses it by name
        let dir = pack_dir("foreign");
        std::fs::write(dir.join("config.json"), include_str!("../../../../../testdata/glm53/config.json")).unwrap();
        let e = match DeepseekV41Family.probe(&dir) {
            Ok(_) => panic!("the GLM config is not a DeepSeek-V4.1 config"),
            Err(e) => e,
        };
        assert!(matches!(e, FamilyError::Config(_)), "{e}");
        assert!(e.to_string().contains("config.json:"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// The released REXL3 pack (`vcruz305/DSV4.1-Flash-1017-SAGE-REXL3-1.55bpw`) names FP8
    /// (`quant_method=fp8`, dynamic activation scheme, `[32, 32]` block weights, UE8M0
    /// scales, FP4 experts) instead of EXL3 in `config.json`; the shared EXL3 config check
    /// refuses it, and the family must accept it out of the box from its tensors
    /// ([`ModelFamily::quantization_from_tensors`]).
    #[test]
    fn the_registry_probes_the_fp8_rexl3_pack_config_out_of_the_box() {
        let dir = pack_dir("rexl3-fp8");
        let mut v: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
        v["quantization_config"] = serde_json::json!({
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        });
        std::fs::write(dir.join("config.json"), v.to_string()).unwrap();
        let (family, _) = Registry::global().probe(&dir).unwrap();
        assert_eq!(family.id(), "deepseek_v41");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
