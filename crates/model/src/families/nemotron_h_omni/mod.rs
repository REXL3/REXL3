//! Nemotron 3.5 Super VL (`NemotronH_Omni_Reasoning_V3`): the text backbone's probe, its layer and MTP specs, and the engine's build.
//!
//! Built: the config parser (`config`), the loader (`inventory`, `weights`), the Mamba2, GQA-without-RoPE and latent-MoE layer kernels
//! and their composition (`model`, `arm_*`), the backend, engine and memory model (`backend`, `engine`, `memory`), and the images: the RADIO
//! tower and its projector as the server's encoder (`vision`), the feature splice in the prefill chunk (`model`, `backend`) and the MTP head's
//! rows for image positions (`mtp`). A server started with `--no-vision` serves text and loads no tower.

pub mod arm_attn;
pub mod arm_mamba;
pub mod arm_moe;
pub mod attn_ref;
pub mod backend;
pub mod bridge;
pub mod config;
pub mod contract;
pub mod device;
pub mod engine;
pub mod inventory;
pub mod memory;
pub mod model;
pub mod moe_ref;
pub mod mtp;
pub mod mtp_ref;
pub mod oracle;
pub mod prefix;
pub mod snapshot;
pub mod spec;
pub mod ssm_ref;
pub mod tapbook;
pub mod tapdump;
pub mod tp;
pub mod tp_arm;
pub mod tp_balance;
pub mod tp_boot;
pub mod tp_comm;
pub mod tp_roce;
pub mod tp_runtime;
pub mod tp_shard;
pub mod vision;
pub mod weights;

use crate::error::ModelError;
use crate::family::{BuildCtx, FamilyError, FamilyPlan, ModelFamily, VisionSpec, BUILT_COMPONENTS};
use crate::hostmem::MemInfo;
use crate::serve::{gib, FeatureBudget, ServeOptions, VisionService};
use config::OmniConfig;
use engine::PlanInputs;
use gb_engine::{boot, ServeEngine};
use inventory::Inventory;
use memory::NemotronMemory;
use model::NemotronModel;
use serde_json::Value;
use std::path::Path;
use weights::LayerPlan;

/// The components the text engine needs. The MTP head is a drafter (offered once `nemotron_mtp` is built) and the vision tower is the
/// plan's `VisionSpec`, so neither stands between a text-only boot and the engine.
pub const REQUIRED: &[&str] = &["nemotron_mamba2", "nemotron_gqa_nope", "nemotron_latent_moe"];

/// The most ranks a Nemotron tensor-parallel group may have: a rank must be able to hold at least one of the 512 routed experts of the
/// published pack (`ShardPlan::experts`; past that ranks would hold none and only add collectives). A statement about what the shard
/// plan, the memory plan and the CPU shard-property tests cover (ranks 2 to 8), NOT about what has run: only two ranks have been
/// measured on a fabric (`docs/NEMOTRON_TP.md` sections 13 and 14).
pub const MAX_TP_RANKS: usize = 512;

pub struct NemotronOmniFamily;

impl ModelFamily for NemotronOmniFamily {
    fn id(&self) -> &'static str {
        "nemotron_h_omni"
    }
    fn architectures(&self) -> &'static [&'static str] {
        &[config::ARCH]
    }
    fn model_types(&self) -> &'static [&'static str] {
        &[config::MODEL_TYPE]
    }
    fn quantization_from_tensors(&self) -> bool {
        true
    }
    /// The decode window is `--window-rows` (64 unless given, at most `model::MAX_WINDOW_ROWS`: many single-row streams in one step, the
    /// aggregate-throughput lever). The shared option itself stops at 128 rows; `REXL3_NEMO_WINDOW_ROWS` asks for more.
    fn window_rows_adjustable(&self) -> bool {
        true
    }
    /// The server asks this BEFORE `build` and forces `--prefix-cache` off for a family that says no. The answer is the model type's own
    /// (`snapshot.rs`: snapshot, restore and release, proven cached == uncached in `tests/nemotron_omni_prefix_gpu.rs`): one place to flip.
    fn prefix_cache(&self) -> bool {
        <NemotronModel as backend::NemotronRuntime>::snapshots_supported()
    }
    fn probe(&self, dir: &Path) -> Result<FamilyPlan, FamilyError> {
        let root = read_config(dir)?;
        let cfg = OmniConfig::from_json(&root).map_err(FamilyError::Config)?;
        expert_storage(dir, &cfg)?;
        let mut drafters = Vec::new();
        if !cfg.mtp_pattern.is_empty() && BUILT_COMPONENTS.contains(&"nemotron_mtp") {
            drafters.push("mtp");
        }
        drafters.push("copy");
        Ok(FamilyPlan {
            family: self.id(),
            dir: dir.to_path_buf(),
            layers: cfg.layer_specs(),
            mtp_layers: cfg.mtp_specs(),
            memory: Box::new(NemotronMemory::new(&cfg, 0)),
            vision: vision_spec(&root),
            chat_format: "nemotron_omni",
            drafters,
            required_components: REQUIRED.to_vec(),
        })
    }
    fn build(&self, plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError> {
        let mut plan = plan;
        if !plan.missing_components().is_empty() {
            return Err(plan.not_built("Nemotron 3.5 Super VL engine"));
        }
        let o = ctx.options;
        // under P2 a prefill rows cap of 64 or less (the MoE layers would run P1's kernels) is refused by name, before the bind (v0.1.1)
        engine::check_prefill_contract(o)?;
        // a tensor-parallel group: rank 0's rendezvous socket is bound first, so rank 1 can connect while this rank plans and loads (a
        // bind that fails fails the boot before a byte of weights is read). Without `--tp-ranks 2` none of this exists.
        let tp = o.tp.as_ref().filter(|t| t.is_group());
        if tp.is_some() {
            // the wide window, the fused and lagged decode and the prefill burst are the single rank's: refused by name, before the bind
            engine::check_tp_plain(o)?;
        }
        let tp_listener = match tp {
            Some(t) => Some(tp_runtime::listen(&t.peer)?),
            None => None,
        };
        let dir = plan.dir.clone();
        let root = read_config(&dir).map_err(|e| ModelError::Config(e.to_string()))?;
        let cfg = OmniConfig::from_json(&root).map_err(ModelError::Config)?;
        let inv = Inventory::read(&dir).map_err(ModelError::Config)?;
        let layers = LayerPlan::build(&cfg, &inv).map_err(|e| ModelError::Config(e.join("; ")))?;
        let weight_bytes = layers.bytes(&inv);
        // the memory model of the pack's forms: the head's weights as the loader holds them (informational) and an EXL3 lm_head's cut
        let head_held = tp_shard::ShardPlan::single().rank_bytes(&layers, cfg.experts, |u| matches!(u, weights::UnitId::Mtp(_)));
        plan.memory = Box::new(
            NemotronMemory::new(&cfg, 0).with_head_weight_bytes(head_held).with_lm_head_exl3(layers.lm_head_exl3().is_some()),
        );
        // a rank of a tensor-parallel group plans what it holds: its shard of the weights and the TP scratch (`tp_boot::rank_weight_bytes`)
        if tp.is_some() {
            tp_boot::plan_for_rank(&mut plan, &cfg, tp_boot::group_ranks(ctx.options));
        }
        // The RADIO tower, when images are served (`engine::vision_wanted`: the plan prices it by the same predicate): its weights and its
        // workspace are allocated BEFORE the free memory is measured, so they are out of the room the KV pool is planned from, and a box that
        // cannot hold them beside the trunk and the floor fails here, by name, before a byte of the tower is allocated (`--no-vision` loads
        // nothing). The features budget is the plan's, so the encoder is built and published once the plan exists and the engine is up.
        let tower = if engine::vision_wanted(&plan, ctx.options) {
            let tokens = vision::image_token_cap(o.max_image_tokens);
            let need = vision::weight_bytes() + vision::scratch_bytes(tokens);
            if let Some(avail) = crate::serve::mem_available() {
                if avail < need.saturating_add(weight_bytes).saturating_add(o.floor_bytes) {
                    return Err(ModelError::Config(format!(
                        "the vision tower needs {:.2} GiB (weights and workspace for images of {tokens} tokens) beside the {:.2} GiB of the trunk, and MemAvailable \
                         is {:.2} GiB with a {:.2} GiB floor; start with --no-vision or a lower --max-image-tokens",
                        gib(need),
                        gib(weight_bytes),
                        gib(avail),
                        gib(o.floor_bytes)
                    )));
                }
            }
            let t0 = std::time::Instant::now();
            let tower = vision::NemotronTower::load(&dir, tokens)?;
            eprintln!(
                "rexl3 event=nemotron_vision_loaded tokens={tokens} weights_gib={:.3} scratch_gib={:.3} load_s={:.2}",
                gib(tower.weight_bytes()),
                gib(tower.scratch_bytes()),
                t0.elapsed().as_secs_f64()
            );
            boot::mark("vision_loaded", Some(gib(tower.weight_bytes() + tower.scratch_bytes())));
            boot::fact("vision_scratch_gib", format!("{:.2}", gib(tower.scratch_bytes())));
            Some(tower)
        } else {
            None
        };
        let (cuda_free, _) = gb_cuda::rt::mem_info()?;
        let host = MemInfo::read();
        // `LayerPlan::bytes` lists the head's tensors whatever the options; they are resident only when the model loads the head
        let head_bytes: u64 = layers
            .tensors(|u| matches!(u, weights::UnitId::Mtp(_)))
            .iter()
            .filter_map(|w| inv.get(&w.name))
            .map(|t| t.nbytes)
            .sum();
        let inp = PlanInputs {
            cuda_free: cuda_free as u64,
            host: host.as_ref(),
            weight_bytes: if tp.is_some() {
                tp_boot::rank_weight_bytes(&plan, &layers, &cfg, weight_bytes, head_bytes, ctx.options)
            } else {
                engine::resident_weight_bytes(weight_bytes, head_bytes, &plan, ctx.options)
            },
            model_max_context: cfg.max_context,
        };
        let slot = o.vision.clone();
        // the decode window this boot runs (`--window-rows`, 64 unless asked): the model's records and buffers are built for it, the plan
        // prices it and the engine's row budget is it (`engine::assemble`). A tensor-parallel group runs the 64-row window.
        let window = engine::window_rows(ctx.options)?;
        let ready = move |sp: &engine::ServePlan| -> Result<(), ModelError> {
            // the tower becomes the server's image encoder only now that the engine exists: a request that carries an image cannot arrive
            // before. The budget is what the plan held back for the features of the images in flight.
            if let (Some(slot), Some(tower)) = (slot, tower) {
                let tokens = tower.max_tokens();
                let enc = vision::NemotronEncoder::new(tower, FeatureBudget::new(sp.plan.vision_reserve_bytes));
                let _ = slot.set(std::sync::Arc::new(VisionService::new(Box::new(enc), tokens)));
            }
            Ok(())
        };
        if let (Some(tp), Some(listener)) = (tp, tp_listener.as_ref()) {
            // rank 0 of a group: the engine is built around the leader runtime, which forwards every state-changing call to rank 1
            // the plan comes to the load closure: the boot gate carries the prefix cache's budget (`tp_boot::gate_with_cache`)
            return engine::build_serve_with_plan(
                &plan,
                ctx,
                &inp,
                |mo, sp| tp_boot::leader_runtime(listener, tp, &dir, &cfg, &layers, mo, o, sp),
                ready,
            );
        }
        let lag = engine::lag_period_for(window)?;
        engine::build_serve_with(
            &plan,
            ctx,
            &inp,
            |mo| NemotronModel::load_lag(&dir, &cfg, &layers, mo.clone(), window, lag),
            ready,
        )
    }

    fn max_tp_ranks(&self) -> usize {
        MAX_TP_RANKS
    }

    fn run_follower(&self, plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<(), ModelError> {
        tp_boot::follow(plan, ctx)
    }

    fn image_preprocessor(
        &self,
        dir: &Path,
        o: &ServeOptions,
    ) -> Result<Option<std::sync::Arc<dyn gb_vision::preprocessor::ImagePreprocessor>>, ModelError> {
        image_preprocessor(dir, o)
    }

    /// The RADIO tower runs a video's tubelets through the video patch projection (`vision::NemotronEncoder::encode_video`); the tower
    /// component is `nemotron_radio_vision` (no component of its own: the weights and the blocks are the image path's).
    fn encodes_video(&self) -> bool {
        true
    }
}

/// The pack's image preprocessing (`preprocessor_config.json`: the HF processor, bit-exact on the CPU) for a server limited to
/// `o.max_image_tokens` tokens an image, at most the tower's ([`vision::image_token_cap`]): the processor shrinks a larger image to the
/// budget, as its `max_num_patches` keyword does, so an image over the limit is scaled and not refused. `None` for a pack with no vision
/// component.
pub fn image_preprocessor(
    dir: &Path,
    o: &ServeOptions,
) -> Result<Option<std::sync::Arc<dyn gb_vision::preprocessor::ImagePreprocessor>>, ModelError> {
    let root = read_config(dir).map_err(|e| ModelError::Config(e.to_string()))?;
    if vision_spec(&root).is_none() {
        return Ok(None);
    }
    let pp = gb_vision::nemotron::NemotronPreprocessing::from_dir(dir)
        .map_err(|e| ModelError::Config(e.to_string()))?
        .with_max_image_tokens(vision::image_token_cap(o.max_image_tokens))
        .map_err(|e| ModelError::Config(e.to_string()))?;
    Ok(Some(std::sync::Arc::new(pp)))
}

/// The vision component the pack declares (`vision_config`, the image placeholder's id). The RADIO tower is built (`vision`): images are served
/// unless the server is started with `--no-vision`, which serves text and loads no tower.
fn vision_spec(root: &Value) -> Option<VisionSpec> {
    root.get("vision_config").filter(|v| v.is_object())?;
    let image = root.get("img_context_token_id").and_then(Value::as_u64)?;
    Some(VisionSpec {
        tower: "nemotron_radio_vision",
        patch_size: 16,
        // pixel shuffle v2 folds 2 x 2 patches into one language-model token
        merge_size: 2,
        temporal_patch_size: 1,
        out_hidden: 4096,
        image_token_id: u32::try_from(image).ok()?,
        // `max_num_patches` 13,312 of the tower, 4 patches per token
        max_image_tokens: 3328,
    })
}

/// What the index says about the routed experts. The published pack keeps a few expert pairs in bf16 (`.weight`) and the rest in EXL3
/// (`.trellis`), so the engine needs the storage of every expert, not just one.
#[derive(Debug, PartialEq)]
pub struct ExpertStorage {
    /// `(layer, expert)` pairs whose up or down projection is bf16. Measured on the published pack: (26, 111), (39, 304), (81, 461).
    pub bf16_pairs: Vec<(usize, usize)>,
}

/// The EXL3 proof for a checkpoint whose config does not name the format. Every routed expert's `up_proj` and `down_proj` must be in the
/// index as either EXL3 `trellis` or bf16 `weight`, and every MoE layer must hold at least one EXL3 expert. A checkpoint that fails is
/// refused with the layer, and the expert where it applies.
pub fn expert_storage(dir: &Path, cfg: &OmniConfig) -> Result<ExpertStorage, FamilyError> {
    let path = dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&path).map_err(|e| FamilyError::Config(format!("{}: {e}", path.display())))?;
    let index: Value =
        serde_json::from_str(&text).map_err(|e| FamilyError::Config(format!("model.safetensors.index.json: {e}")))?;
    let map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| FamilyError::Config("index has no weight_map".into()))?;
    // the names in the pack's own layout (`inventory::Layout`: with or without `language_model.`, never both)
    let layout = inventory::Layout::detect(map.keys().map(String::as_str)).map_err(FamilyError::Config)?;
    let mut bf16_pairs = Vec::new();
    for (layer, block) in cfg.pattern.chars().enumerate() {
        if block != 'E' {
            continue;
        }
        let base = layout.file_name(&format!("{}backbone.layers.{layer}.mixer.experts", inventory::TEXT_PREFIX));
        let mut exl3_in_layer = 0usize;
        for expert in 0..cfg.experts {
            let mut bf16 = false;
            for proj in ["up_proj", "down_proj"] {
                if map.contains_key(&format!("{base}.{expert}.{proj}.trellis")) {
                    exl3_in_layer += 1;
                } else if map.contains_key(&format!("{base}.{expert}.{proj}.weight")) {
                    bf16 = true;
                } else {
                    return Err(FamilyError::Config(format!(
                        "this engine serves EXL3 packs only; layer {layer} expert {expert} has no `{proj}` tensor in either form"
                    )));
                }
            }
            if bf16 {
                bf16_pairs.push((layer, expert));
            }
        }
        if exl3_in_layer == 0 {
            return Err(FamilyError::Config(format!(
                "this engine serves EXL3 packs only; layer {layer} has no EXL3 expert tensor"
            )));
        }
    }
    Ok(ExpertStorage { bf16_pairs })
}

fn read_config(dir: &Path) -> Result<Value, FamilyError> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| FamilyError::Config(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| FamilyError::Config(format!("config.json: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::Registry;

    const PACK: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");

    /// A scratch checkpoint directory: the published config, and an index whose expert tensors are listed for `layers`.
    fn checkpoint(name: &str, index_keys: &[String]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("rexl3-omni-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), PACK).unwrap();
        let weight_map: serde_json::Map<String, Value> =
            index_keys.iter().map(|k| (k.clone(), Value::from("model-00001-of-00001.safetensors"))).collect();
        let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map });
        std::fs::write(dir.join("model.safetensors.index.json"), index.to_string()).unwrap();
        dir
    }

    /// Every routed expert's up and down projections as EXL3 `trellis`, on every MoE layer, in the order (layer, expert, projection).
    fn expert_keys(cfg: &OmniConfig) -> Vec<String> {
        let mut keys = Vec::new();
        for (layer, block) in cfg.pattern.chars().enumerate() {
            if block != 'E' {
                continue;
            }
            for expert in 0..cfg.experts {
                for proj in ["up_proj", "down_proj"] {
                    keys.push(format!("language_model.backbone.layers.{layer}.mixer.experts.{expert}.{proj}.trellis"));
                }
            }
        }
        keys
    }

    #[test]
    fn a_checkpoint_with_the_expert_tensors_on_every_moe_layer_probes() {
        let cfg = OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap();
        let dir = checkpoint("exl3", &expert_keys(&cfg));
        let (family, plan) = Registry::global().probe(&dir).unwrap();
        assert_eq!(family.id(), "nemotron_h_omni");
        assert_eq!(plan.layers.len(), 88);
        // the MTP head is offered (`nemotron_mtp` is built) and the copy drafter is the family's second
        assert_eq!(plan.drafters, vec!["mtp", "copy"]);
        assert_eq!(plan.mtp_layers.len(), 2);
        assert!(plan.missing_components().is_empty(), "the text components are built: {:?}", plan.missing_components());
        assert_eq!(plan.unbuilt_vision(), None, "the RADIO tower is built");
        let v = plan.vision.as_ref().unwrap();
        assert_eq!((v.patch_size, v.merge_size, v.out_hidden, v.image_token_id), (16, 2, 4096, 18));
    }

    #[test]
    fn a_checkpoint_missing_one_moe_layers_expert_tensor_is_refused_by_layer() {
        let cfg = OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap();
        let mut keys = expert_keys(&cfg);
        let dropped = keys.remove(5); // expert 2's down projection on the first MoE layer
        let dir = checkpoint("missing", &keys);
        let err = Registry::global().probe(&dir).err().expect("the probe should refuse");
        let msg = err.to_string();
        assert!(msg.contains("EXL3 packs only"), "{msg}");
        let layer: String = dropped.split('.').nth(3).unwrap().to_string();
        assert!(msg.contains(&format!("layer {layer} ")), "{msg}");
        assert!(msg.contains("expert 2 "), "{msg}");
    }

    #[test]
    fn a_bf16_expert_projection_is_reported_by_expert_and_the_probe_still_passes() {
        let cfg = OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap();
        let mut keys = expert_keys(&cfg);
        let layer = cfg.pattern.chars().position(|c| c == 'E').unwrap();
        let up = format!("language_model.backbone.layers.{layer}.mixer.experts.111.up_proj.trellis");
        let pos = keys.iter().position(|k| *k == up).unwrap();
        keys[pos] = up.replace(".trellis", ".weight");
        let dir = checkpoint("bf16pair", &keys);
        assert_eq!(expert_storage(&dir, &cfg).unwrap().bf16_pairs, vec![(layer, 111)]);
        assert!(Registry::global().probe(&dir).is_ok());
    }

    /// The MixedK pack's layout (no `language_model.` on the text names) probes like ours; an index that mixes the two is refused by name.
    #[test]
    fn a_checkpoint_named_without_the_language_model_prefix_probes_and_a_mixed_one_is_refused() {
        let cfg = OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap();
        let bare: Vec<String> =
            expert_keys(&cfg).iter().map(|k| k.strip_prefix("language_model.").unwrap().to_string()).collect();
        let dir = checkpoint("bare", &bare);
        assert_eq!(expert_storage(&dir, &cfg).unwrap().bf16_pairs, vec![]);
        let (family, plan) = Registry::global().probe(&dir).unwrap();
        assert_eq!((family.id(), plan.layers.len()), ("nemotron_h_omni", 88));
        let mut mixed = bare.clone();
        mixed.push("language_model.lm_head.weight".into());
        let dir = checkpoint("mixed-layout", &mixed);
        let msg = expert_storage(&dir, &cfg).unwrap_err().to_string();
        assert!(msg.contains("`language_model.lm_head.weight`") && msg.contains("one layout"), "{msg}");
    }

    #[test]
    fn a_config_without_quantization_still_needs_the_family_to_prove_it() {
        // Another architecture with no quantization_config is still refused by the shared check, unchanged.
        let dir = std::env::temp_dir().join(format!("rexl3-omni-noquant-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama"}"#).unwrap();
        assert!(Registry::global().probe(&dir).err().expect("the probe should refuse").to_string().contains("EXL3 packs only"));
    }
}
