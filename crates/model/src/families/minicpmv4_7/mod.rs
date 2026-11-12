//! MiniCPM-V-4.7-35B-A3B (`MiniCPMV4_7ForConditionalGeneration`) with its DSpark draft: the probe, the memory model and the
//! capability report (E0 of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`).
//!
//! The family is known to the registry: a MiniCPM-V-4.7 pack is parsed, checked against the shapes the kernels implement, planned
//! (layer recipe, memory model, vision component, chat format, drafters) and, with every component of [`REQUIRED`] built, served as a
//! text engine. Each component landed in its own slice with its row-invariance test, golden hash and oracle comparison, and moved into
//! `BUILT_COMPONENTS` in that commit (E1-E6 of the plan); what is still to come is [`PENDING`] (nothing: the list below is the history
//! of the slices):
//!
//! * `gdn_hv32`, `attention_q16kv2`, `coopmk_h2048_i512`, `exl3_pins_minicpmv`, `topk_pad_mask`: the E1 kernels (the gated delta net at
//!   16 key x 32 value heads, grouped-query attention at 16 query and 2 KV heads of 256 with the fused output gate, the grouped-expert
//!   kernel at hidden 2,048 / intermediate 512, the EXL3 pins of this model's shapes, top-k that masks the 48 pad ids): built;
//! * `attention_q8kv2_draft`, `dspark_drafter`, `dspark_selector_walk`, `dspark_confidence_head`: the draft's non-causal block forward
//!   over runtime taps, the Markov head as the keyed selector walk and the acceptance-confidence output (E4a stand-alone, E4b in the
//!   engine: [`dspark`], [`spec`]): built;
//! * `vision_minicpmv`, `mrope_canvas`, `image_slicing_minicpmv`, `chat_minicpmv`: the 27-layer tower with its intra-tower window
//!   merger, the canvas M-RoPE positions, the multi-slice preprocessor and the image and video layout of the chat format (E5a built
//!   them stand-alone, E5b wired and gated them: [`splice`], the tower in [`engine`], `server::formats::minicpmv`): built;
//! * `prefix_cache_minicpmv`: snapshot and restore of the GDN state and the KV pages of text, image and video prompts are built (E6b,
//!   [`prefix`]; E5b: the image keys, the cut rule outside visual groups), with the draft's pages riding in the same page ids (E4b):
//!   built. Nothing is pending ([`PENDING`] is empty).
//!
//! **E2 (built, gated): the weights loader and the boot with no forward.** [`weights`] binds a pack's tensors to the recipe (every
//! tensor claimed once, the norms' `(1 + w)` widening, the tower recorded and not loaded), [`device`] uploads them through the
//! pipelined loader into typed structs with the expert pointer tables the MoE kernels index, [`state`] allocates the KV pool, the
//! delta-net pools and the window buffers at the sizes `memory` prices, and [`boot`] runs the fail-closed sequence (refusals by name,
//! the validated-pack list, the kernel gates, admission on `MemAvailable`, plan equals allocation) and returns a [`boot::LoadedModel`].
//! `weights_minicpmv4_7` is in `BUILT_COMPONENTS` for it.
//!
//! **E5b (built, gated by the processor's prompt ids and positions and by the oracle): images and video.** [`splice`] is the host
//! side of the splice (spans, pad runs, the chunk grid outside visual groups, the canvas positions of the whole prompt), [`backend`]
//! prefills a prompt with spans chunk by chunk with the feature rows spliced before the first norm and the rope delta set for
//! decode, [`vision`] is the tower as the server's image encoder with a keyed feature cache, and [`engine`] loads it when the
//! server has vision (priced by [`memory`]: weights, workspace, feature budget). `vision_minicpmv`, `mrope_canvas`,
//! `image_slicing_minicpmv` and `chat_minicpmv` are in `BUILT_COMPONENTS` for it.
//!
//! **E3 (built, gated by the oracle): the text forward and the engine.** [`model`] is the serial forward (prefill chunks on the
//! row-invariant `PrefillPath` rule, decode and verify windows of up to 64 rows, GDN records and replay commits, window graphs, the
//! padded head masked everywhere), [`backend`] the engine's `Backend` (slicing, per-request failure isolation, error latching) and
//! [`engine`] the engine around a booted model. `forward_minicpmv4_7` and the E1 kernel names are in `BUILT_COMPONENTS` for it. The
//! text engine serves with `--no-vision`; what the later slices own is [`PENDING`].
//!
//! **E4b (built, gated): the DSpark draft in the engine.** [`spec`] is the draft's interface to the engine: the tap export of five
//! layers' residual rows, the per-slot host book of what the draft has absorbed, the lazy ring of accepted rows absorbed in one batched
//! pass a round, the private overflow page and the draft's own stream; `--draft DIR` is read by [`boot`] (refusals by name, the pair
//! checked, the draft's bytes planned with the pool) and the engine's budget reads the draft's confidences through `Proposal.accept`.
//! The four draft names are in `BUILT_COMPONENTS` for it.
//!
//! **E6b (built, gated): the prefix cache of text prompts.** [`prefix`] is the engine-owned snapshot arena of the 30 GDN layers' state
//! and convolution history (64,389,120 B a snapshot) with the KV pages of a cached prefix shared by reference (a partly filled last
//! page copied on restore); the boot plans and allocates it with the pool (`--prefix-cache-mib`, `auto`), the backend offers the
//! engine's `prefix_*` hooks and `prefill_slice_cuts`, and `ModelFamily::prefix_cache()` answers true.
//!
//! The pack is probed from its `config.json` alone; the draft is a property of the run (`--draft DIR`), parsed by
//! [`config::DSparkConfig`] and priced by [`MiniCpmMemorySource::with_draft`].

pub mod backend;
pub mod boot;
pub mod config;
pub mod device;
pub mod dspark;
pub mod engine;
pub mod inventory;
pub mod memory;
pub mod model;
pub mod prefix;
pub mod spec;
pub mod splice;
pub mod state;
pub mod vision;
pub mod weights;

use crate::error::ModelError;
use crate::family::{BuildCtx, FamilyError, FamilyPlan, MemorySource, ModelFamily};
use crate::memory::MemoryOptions;
use config::{DSparkConfig, MiniCpmConfig};
use gb_engine::ServeEngine;
use gb_sched::{MemoryError, MemoryProfile};
use serde_json::Value;
use std::path::Path;

/// Everything a MiniCPM-V-4.7 TEXT engine runs on: the shared components, the weights loader (E2), the E1 kernels at the shapes of
/// this model and the forward (E3). `serve::start` refuses a family with any of them unbuilt, naming it.
pub const REQUIRED: &[&str] = &[
    "exl3_linear",
    "moe_softmax_topk",
    "plain_residual",
    // E2: the weights loader and the boot (`boot::boot`)
    "weights_minicpmv4_7",
    // E1 kernels, run at model level by the forward
    "gdn_hv32",
    "attention_q16kv2",
    "coopmk_h2048_i512",
    "exl3_pins_minicpmv",
    "topk_pad_mask",
    // E3: the serial forward, prefill, windows, the backend and the engine around `boot::LoadedModel`
    "forward_minicpmv4_7",
];

/// What the later slices own: nothing. Every component of the plan is built (the draft by E4b, the tower and the image layout by E5b,
/// the prefix cache of text, image and video prompts by E6b and E5b); the constant stays because the capability line of `build`'s
/// refusal and the tests read it, and a later component is named here until it is built.
pub const PENDING: &[&str] = &[];

pub struct MiniCpmFamily;

impl ModelFamily for MiniCpmFamily {
    fn id(&self) -> &'static str {
        "minicpmv4_7"
    }
    fn architectures(&self) -> &'static [&'static str] {
        &["MiniCPMV4_7ForConditionalGeneration"]
    }
    fn model_types(&self) -> &'static [&'static str] {
        &["minicpmv4_7"]
    }

    fn probe(&self, dir: &Path) -> Result<FamilyPlan, FamilyError> {
        let root = read_config(dir)?;
        crate::family::require_exl3(&root)?;
        let cfg = MiniCpmConfig::from_json(&root).map_err(FamilyError::Config)?;
        cfg.check_shapes().map_err(FamilyError::Config)?;
        Ok(FamilyPlan {
            family: self.id(),
            dir: dir.to_path_buf(),
            layers: cfg.layer_specs(),
            // the config declares one MTP layer and the checkpoint holds none of its tensors: the family runs none
            mtp_layers: vec![],
            vision: Some(cfg.vision_spec()),
            memory: Box::new(MiniCpmMemorySource { cfg, draft: None }),
            chat_format: "minicpmv", // text rendering is built (`server/src/formats/minicpmv.rs`); the image layout is E5b
            drafters: vec!["dspark", "copy"],
            required_components: REQUIRED.to_vec(),
        })
    }

    /// The image preprocessing of the server (E5b): the checkpoint's slicing, normalisation and `downsample_mode`, with the
    /// environment options of `vision` (resize backend, slice count, 16x or 4x). The encoder re-derives its crop plan from the same
    /// function, so the placeholders the server counts are the rows the tower makes.
    fn image_preprocessor(
        &self,
        dir: &Path,
        _o: &crate::serve::ServeOptions,
    ) -> Result<Option<std::sync::Arc<dyn gb_vision::preprocessor::ImagePreprocessor>>, ModelError> {
        Ok(Some(std::sync::Arc::new(vision::image_preprocessing(dir)?)))
    }

    /// A video is a list of frames through the same tower (E5b): the format takes it when the server enables video.
    fn encodes_video(&self) -> bool {
        true
    }

    /// Snapshot and restore of text prompts are proven cached == uncached on the real pack (`tests/minicpmv4_7_prefix.rs`, section 14
    /// of the plan), and of image and video prompts too (E5b, `crates/server/tests/minicpmv4_7_mm.rs`, section 12): the serve starter
    /// keeps the operator's `--prefix-cache` request, and `build` plans the cache's bytes with the pool.
    fn prefix_cache(&self) -> bool {
        true
    }

    fn build(&self, plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError> {
        // `serve::preflight` calls `build` to learn what is missing and a 28 GB load is not a preflight: nothing is read unless every
        // component the text engine needs is built. A pack's headers are read (never a weight) so the refusal can say whether the pack
        // binds to the recipe, and by name what it lacks when it does not.
        if !plan.missing_components().is_empty() {
            let mut what = format!(
                "MiniCPM-V-4.7 text engine (the weights loader is built, `families::minicpmv4_7::boot`; still pending in later slices: {})",
                PENDING.join(", ")
            );
            if let Some(verdict) = boot::pack_verdict(&plan.dir) {
                what.push_str("; ");
                what.push_str(&verdict);
            }
            return Err(plan.not_built(&what));
        }
        Ok(Box::new(engine::build_engine(ctx.options)?))
    }
}

fn read_config(dir: &Path) -> Result<Value, FamilyError> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| FamilyError::Config(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| FamilyError::Config(format!("config.json: {e}")))
}

/// The memory model of a MiniCPM-V-4.7 pack, ready to be evaluated at any serving settings: the target alone, or with the DSpark
/// draft's cache once a draft is attached.
pub struct MiniCpmMemorySource {
    pub cfg: MiniCpmConfig,
    pub draft: Option<DSparkConfig>,
}

impl MiniCpmMemorySource {
    /// The same model priced with a DSpark draft loaded (what `build` does with `--draft DIR`).
    pub fn with_draft(mut self, draft: DSparkConfig) -> Self {
        self.draft = Some(draft);
        self
    }
}

impl MemorySource for MiniCpmMemorySource {
    fn profile(&self, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
        memory::memory_profile(&self.cfg, self.draft.as_ref(), opts)
    }
}

#[cfg(test)]
mod tests {
    use super::config::tests::{draft, real, DSPARK_CONFIG, MINICPM_CONFIG};
    use super::*;
    use crate::family::{Ffn, Mixer, Registry, Residual};

    fn dir_with(name: &str, cfg: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("rexl3-minicpm-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), cfg).unwrap();
        dir
    }

    fn edited(f: impl FnOnce(&mut Value)) -> String {
        let mut v: Value = serde_json::from_str(MINICPM_CONFIG).unwrap();
        f(&mut v);
        v.to_string()
    }

    #[test]
    fn the_registry_sends_a_minicpm_pack_here_and_the_text_engine_has_every_component() {
        let dir = dir_with("probe", MINICPM_CONFIG);
        let (fam, plan) = Registry::global().probe(&dir).unwrap();
        assert_eq!((fam.id(), plan.family), ("minicpmv4_7", "minicpmv4_7"));
        // 40 layers: 30 GDN and 10 attention, every one MoE on a plain residual; no MTP layer although the config declares one
        assert_eq!((plan.layers.len(), plan.mtp_layers.len()), (40, 0));
        assert_eq!(plan.layers.iter().filter(|l| l.mixer == Mixer::Gqa).count(), 10);
        assert_eq!(plan.layers.iter().filter(|l| l.mixer == Mixer::Gdn).count(), 30);
        assert!(plan.layers.iter().all(|l| l.ffn == Ffn::Moe && l.residual == Residual::Plain));
        assert_eq!((plan.chat_format, plan.drafters.clone()), ("minicpmv", vec!["dspark", "copy"]));
        // the tower: 14-pixel patches, 4 x 4 patches a token in 16x mode, 2,048-wide output, image pad id 248,056
        let v = plan.vision.as_ref().unwrap();
        assert_eq!(
            (v.tower, v.patch_size, v.merge_size, v.temporal_patch_size, v.out_hidden, v.image_token_id),
            ("vision_minicpmv", 14, 4, 1, 2048, 248_056)
        );
        // the tower is built (E5b): images are served when the server loads it
        assert_eq!(plan.unbuilt_vision(), None);
        // snapshot and restore are proven cached == uncached on the real pack (E6b, E5b): the family offers the cache
        assert!(fam.prefix_cache() && !fam.window_rows_adjustable() && fam.max_tp_ranks() == 1);
        // the text engine's components are all built (E3); the tower is refused by name through the plan, the draft through --draft
        assert!(plan.missing_components().is_empty(), "{:?}", plan.missing_components());
        assert!(REQUIRED.iter().all(|c| crate::family::BUILT_COMPONENTS.contains(c)));
        assert!(PENDING.iter().all(|c| !crate::family::BUILT_COMPONENTS.contains(c)), "a pending component is not built");
        // the draft's components moved to the built list with E4b: a drafted engine has everything it names
        for c in ["attention_q8kv2_draft", "dspark_drafter", "dspark_selector_walk", "dspark_confidence_head"] {
            assert!(crate::family::BUILT_COMPONENTS.contains(&c) && !PENDING.contains(&c), "{c}");
        }
        // the plan's memory model evaluates at serving settings: 20,480 B a token, the state of 30 GDN layers
        let p = plan.memory_profile(&MemoryOptions::serving_defaults()).unwrap();
        assert_eq!((p.family(), p.kv_bytes_per_token(), p.state_bytes_per_seq()), ("minicpmv4_7", 20_480, 64_389_120));
        // and build refuses the options it has no component for BEFORE it touches a file or the device
        let mut o = crate::serve::ServeOptions::new(dir.clone());
        o.window_rows = 128;
        let e = match MiniCpmFamily.build(plan, &BuildCtx { options: &o }) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("built an engine with a window the family does not have"),
        };
        assert!(e.contains("--window-rows must be 64"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_server_start_serves_text_and_images_and_refuses_what_is_not_built_by_name() {
        let dir = dir_with("start", MINICPM_CONFIG);
        // images: the tower is a component now, so the start goes on to the pack (which this directory does not hold) instead of
        // refusing by name; the preflight passes with vision on as it does without
        let mut o = crate::serve::ServeOptions::new(dir.clone());
        o.vision = Some(Default::default());
        assert_eq!(crate::serve::preflight_in(Registry::global(), &o).unwrap(), "minicpmv");
        let e = crate::serve::start_in(Registry::global(), &o).err().expect("a directory with no pack started").to_string();
        assert!(
            !e.contains("not built") && !e.contains("--no-vision"),
            "the start must reach the pack, not stop on the tower: {e}"
        );
        // text: the preflight passes (every component is built) and the start goes on to the pack, which this directory does not hold
        let o = crate::serve::ServeOptions::new(dir.clone());
        assert_eq!(crate::serve::preflight_in(Registry::global(), &o).unwrap(), "minicpmv");
        let e = crate::serve::start_in(Registry::global(), &o).err().expect("a directory with no pack started").to_string();
        assert!(!e.contains("not built"), "the start must reach the pack, not stop on a component: {e}");
        // --spec mtp asks for a head this checkpoint does not have: refused by name, listing what is offered
        let mut o = crate::serve::ServeOptions::new(dir.clone());
        o.spec = gb_spec::SpecMode::Mtp;
        let e = crate::serve::start_in(Registry::global(), &o).err().expect("refused").to_string();
        assert!(e.contains("--spec mtp") && e.contains("dspark, copy"), "{e}");
        // a draft is built (E4b): `--draft` is no longer refused for the component, and a directory that is no draft is refused by
        // the draft's own config (or the missing pack) before anything is loaded
        let mut o = crate::serve::ServeOptions::new(dir.clone());
        o.draft_dir = Some(dir.clone());
        let e = crate::serve::start_in(Registry::global(), &o).err().expect("refused").to_string();
        assert!(!e.contains("dspark_drafter") && !e.contains("not built"), "{e}");
        // --window-rows is not this family's option
        let mut o = crate::serve::ServeOptions::new(dir.clone());
        o.window_rows = 128;
        let e = crate::serve::start_in(Registry::global(), &o).err().expect("refused").to_string();
        assert!(e.contains("--window-rows 128") && e.contains("minicpmv4_7"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_family_is_registered_once_and_served_by_architecture_and_model_type() {
        let reg = Registry::global();
        assert!(reg.served().contains(&"MiniCPMV4_7ForConditionalGeneration"));
        assert!(reg.families().iter().any(|f| f.id() == "minicpmv4_7"));
        let by_type = serde_json::json!({"model_type": "minicpmv4_7"});
        assert_eq!(reg.detect(&by_type).map(|f| f.id()), Ok("minicpmv4_7"));
        // the architecture decides, as for every family; the close Qwen names stay with Qwen
        let arch = serde_json::json!({"architectures": ["MiniCPMV4_7ForConditionalGeneration"], "model_type": "qwen3_5"});
        assert_eq!(reg.detect(&arch).map(|f| f.id()), Ok("minicpmv4_7"));
        let qwen = serde_json::json!({"architectures": ["Qwen3_5ForConditionalGeneration"]});
        assert_eq!(reg.detect(&qwen).map(|f| f.id()), Ok("qwen3_5"));
    }

    #[test]
    fn a_sibling_architecture_is_refused_with_a_report_that_names_this_family_as_the_closest() {
        // the previous MiniCPM-V generation has its own architecture string: it is not read by this family, and the report says which
        // served family is nearest and what its config would need
        let mut v: Value = serde_json::from_str(MINICPM_CONFIG).unwrap();
        v["architectures"] = serde_json::json!(["MiniCPMV4_6ForConditionalGeneration"]);
        v["model_type"] = serde_json::json!("minicpmv4_6");
        let dir = dir_with("sibling", &v.to_string());
        let e = Registry::global().probe(&dir).err().unwrap();
        let crate::family::FamilyError::Unsupported(r) = &e else { panic!("{e}") };
        assert_eq!(r.architecture.as_deref(), Some("MiniCPMV4_6ForConditionalGeneration"));
        assert_eq!(r.closest, Some("minicpmv4_7"));
        let text = e.to_string();
        assert!(text.contains("closest served family: minicpmv4_7"), "{text}");
        assert!(text.contains("MiniCPMV4_7ForConditionalGeneration"), "the served list names this family's architecture: {text}");
        assert!(text.contains("--allow-unvalidated-pack does not relax"), "{text}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_non_exl3_pack_is_refused_at_the_probe() {
        // the Hugging Face checkpoint itself (no quantization_config): refused by the format check, naming what was found
        let hf = edited(|v| {
            v.as_object_mut().unwrap().remove("quantization_config");
        });
        let dir = dir_with("bf16", &hf);
        let e = Registry::global().probe(&dir).err().unwrap();
        assert!(e.to_string().contains("EXL3") && e.to_string().contains("no quantization_config"), "{e}");
        let gguf = edited(|v| v["quantization_config"]["quant_method"] = serde_json::json!("gguf"));
        std::fs::write(dir.join("config.json"), gguf).unwrap();
        let e = Registry::global().probe(&dir).err().unwrap();
        assert!(e.to_string().contains("quant_method=gguf"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn corrupted_configs_are_refused_by_name_through_the_registry() {
        type Edit = Box<dyn Fn(&mut Value)>;
        let cases: Vec<(&str, Edit, &str)> = vec![
            (
                "types-short",
                Box::new(|v| {
                    v["text_config"]["layer_types"].as_array_mut().unwrap().pop();
                }),
                "layer_types has 39 entries for 40 layers",
            ),
            ("experts", Box::new(|v| v["text_config"]["num_experts"] = serde_json::json!(128)), "num_experts 128"),
            ("topk", Box::new(|v| v["text_config"]["num_experts_per_tok"] = serde_json::json!(10)), "num_experts_per_tok 10"),
            ("vocab", Box::new(|v| v["text_config"]["vocab_size"] = serde_json::json!(248_320)), "vocab_size 248320"),
            (
                "no-text",
                Box::new(|v| {
                    v.as_object_mut().unwrap().remove("text_config");
                }),
                "text_config is missing",
            ),
            (
                "no-vision",
                Box::new(|v| {
                    v.as_object_mut().unwrap().remove("vision_config");
                }),
                "vision_config is missing",
            ),
            ("downsample", Box::new(|v| v["downsample_mode"] = serde_json::json!("8x")), "downsample_mode \"8x\""),
            ("mrope", Box::new(|v| v["mrope_mode"] = serde_json::json!("flat")), "mrope_mode `flat`"),
            ("insert", Box::new(|v| v["insert_layer_id"] = serde_json::json!(27)), "insert_layer_id 27"),
            ("gate", Box::new(|v| v["text_config"]["attn_output_gate"] = serde_json::json!(false)), "attn_output_gate false"),
            ("tied", Box::new(|v| v["tie_word_embeddings"] = serde_json::json!(true)), "tie_word_embeddings true"),
        ];
        for (tag, edit, needle) in cases {
            let dir = dir_with(tag, &edited(|v| edit(v)));
            let e = Registry::global().probe(&dir).err().unwrap_or_else(|| panic!("{tag}: probed"));
            assert!(e.to_string().contains(needle), "{tag}: {e} does not name {needle:?}");
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn the_memory_source_prices_the_draft_once_one_is_attached() {
        let dir = dir_with("source", MINICPM_CONFIG);
        let plan = MiniCpmFamily.probe(&dir).unwrap();
        let alone = plan.memory_profile(&MemoryOptions::serving_defaults()).unwrap();
        assert_eq!(alone.kv_bytes_per_token(), 20_480);
        let with = MiniCpmMemorySource { cfg: real(), draft: None }.with_draft(draft());
        let p = with.profile(&MemoryOptions::serving_defaults()).unwrap();
        assert_eq!(p.kv_bytes_per_token(), 30_720);
        // a draft of another model is refused when it is attached to the profile
        let mut v: Value = serde_json::from_str(DSPARK_CONFIG).unwrap();
        v["dflash_config"]["target_layer_ids"] = serde_json::json!([3, 11, 19, 27, 41]);
        let other = DSparkConfig::from_json(&v).unwrap();
        let e = MiniCpmMemorySource { cfg: real(), draft: Some(other) }.profile(&MemoryOptions::serving_defaults()).unwrap_err();
        assert!(e.to_string().contains("tap layer 41"), "{e}");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
