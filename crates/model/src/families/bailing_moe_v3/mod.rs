//! The Bailing Moe V3 family (`BailingMoeV3ForCausalLM`, `model_type` `bailing_hybrid` / `bailing_moe_v3`): probe and `build`
//! for Ling-3.1-flash-1019 (and its EXL3-SAGE quantized pack `vcruz305/Ling-3.1-flash-1019-EXL3-SAGE-1.51bpw`). The pack is
//! mapped and loaded, the state and scratch are allocated, and a `BailingBackend` runs the serial forward; with `--spec mtp|auto`
//! the pack's `L72` NextN layer is the drafter.
//!
//! Composition: 72 trunk layers (`L0..3` KDA + dense SwiGLU; `L4..71` KDA except `(lid+1)%8==0` layers which are Gated MLA,
//! all MoE with 512 routed + 1 shared expert), MTP `L72` (Gated MLA + MoE), KDA safe-gate decay, Gated MLA with latent 512 +
//! rope 64 (+ a 576→640 padded `kv_a_proj_with_mqa` group), and per-layer SwiGLU clamps (`10.0` on routed 65..=71, shared
//! 64..=71). Residuals are plain.

pub mod alloc;
pub mod backend;
pub mod bind;
pub mod config;
pub mod device;
pub mod engine;
pub mod inventory;
pub mod load;
pub mod memory;
pub mod model;
pub mod plan;
pub mod weights;

pub use bind::{LayoutError, Problem};
pub use config::BailingConfig;
pub use inventory::{GroupInfo, PackInventory, TensorInfo};
pub use plan::{ItemPart, LoadItem, LoadPlan, PlanTotals, ReadExtent, Region, RegionKind};
pub use weights::{BailingWeights, Xform};

use crate::error::ModelError;
use crate::family::{BuildCtx, FamilyError, FamilyPlan, ModelFamily};
use gb_engine::ServeEngine;
use std::path::Path;

/// The plain bf16/bf16 linear shim: `gb_cuda::bf16_linear::forward` (x fp16, w bf16, y fp16/f32).
pub mod ffi_glue {
    use gb_cuda::bf16_linear;
    use gb_cuda::rt::CudaError;
    use gb_cuda::rt::{DevBuf, Stream};
    /// `y [m, n] = x [m, k] @ w^T [n, k]` (`x` fp16, `w` bf16, `y` fp16/f32).
    pub fn bf16_linear(
        x: &DevBuf,
        w: &DevBuf,
        y: &mut DevBuf,
        m: usize,
        k: usize,
        n: usize,
        y_f32: bool,
        st: &Stream,
    ) -> Result<(), CudaError> {
        bf16_linear::forward(x, w, y, m, k, n, y_f32, st)
    }
}

/// The components the text model needs. `mla_gated` is not in `BUILT_COMPONENTS` until the Gated MLA kernels (the CUDA
/// workstream) land; until then a probe succeeds and a build refuses at the first step with a named error.
const REQUIRED: &[&str] = &["exl3_linear", "kda", "moe", "mla_gated"];

/// The pack's NextN layer is a drafter (`--spec mtp|auto`): the probe offers it, the plan charges its cache and scratch when the
/// mode uses it.
pub const MTP_BUILT: bool = true;

pub struct BailingFamily;

impl ModelFamily for BailingFamily {
    fn id(&self) -> &'static str {
        "bailing_moe_v3"
    }
    fn architectures(&self) -> &'static [&'static str] {
        &["BailingMoeV3ForCausalLM"]
    }
    fn model_types(&self) -> &'static [&'static str] {
        &["bailing_hybrid", "bailing_moe_v3"]
    }

    fn probe(&self, dir: &Path) -> Result<FamilyPlan, FamilyError> {
        let cfg = BailingConfig::from_dir(dir).map_err(FamilyError::Config)?;
        let drafters = if MTP_BUILT && cfg.has_mtp() { vec!["mtp", "copy"] } else { vec!["copy"] };
        Ok(FamilyPlan {
            family: self.id(),
            dir: dir.to_path_buf(),
            layers: cfg.layers.clone(),
            mtp_layers: cfg.mtp.clone(),
            memory: Box::new(memory::BailingMemorySource(cfg)),
            vision: None,
            chat_format: "bailing_moe_v3",
            drafters,
            required_components: REQUIRED.to_vec(),
        })
    }

    fn build(&self, plan: FamilyPlan, ctx: &BuildCtx<'_>) -> Result<Box<dyn ServeEngine>, ModelError> {
        let o = ctx.options;
        if !plan.missing_components().is_empty() {
            return Err(plan.not_built("the Bailing-3.1 engine"));
        }
        engine::check_options(o)?;
        Ok(Box::new(engine::build_engine(o)?))
    }

    /// The Bailing family proves snapshot/restore (the 63 KDA slot states, the 10 MLA page caches and the MTP
    /// drafter's host state), so `--prefix-cache` may be enabled for it.
    fn prefix_cache(&self) -> bool {
        true
    }

    fn max_tp_ranks(&self) -> usize {
        1
    }
}

/// A Bailing pack mapped, from its headers alone.
pub struct BailingPack {
    pub config: BailingConfig,
    pub weights: BailingWeights,
    pub plan: LoadPlan,
}

impl BailingPack {
    pub fn from_inventory(inv: &PackInventory, config: BailingConfig) -> Result<Self, LayoutError> {
        let weights = BailingWeights::map(inv, &config)?;
        let plan = weights.load_plan().map_err(|e| LayoutError { problems: vec![Problem::HeaderIssue(e)] })?;
        Ok(Self { config, weights, plan })
    }

    pub fn map_pack(dir: &Path) -> Result<Self, String> {
        let config = BailingConfig::from_dir(dir)?;
        let ck = gb_exl3::checkpoint::scan(dir, false).map_err(|e| e.to_string())?;
        let inv = PackInventory::from_checkpoint(&ck)?;
        Self::from_inventory(&inv, config).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::collections::BTreeMap;

    const REAL: &str = include_str!("../../../../../testdata/ling31/config.json");

    fn real_config() -> BailingConfig {
        BailingConfig::from_json(&serde_json::from_str(REAL).unwrap()).unwrap()
    }

    /// A synthetic single-shard header map for the whole pack, from the real tensor inventory: every plain tensor and every
    /// EXL3 group part at its real shape. Bits are per-group (the trellis last dim = 16 x bits); the routed experts of a layer
    /// share a per-layer (gate/up, down) bit pair, as the recipe does.
    fn synthetic_headers(cfg: &BailingConfig) -> BTreeMap<String, Value> {
        fn add(h: &mut BTreeMap<String, Value>, name: &str, dtype: &str, shape: &[usize]) {
            let e = serde_json::json!({ "dtype": dtype, "shape": shape, "data_offsets": [0, 1] });
            h.insert(name.to_string(), e);
        }
        fn group(h: &mut BTreeMap<String, Value>, prefix: &str, k: usize, n: usize, bits: f64) {
            let b = (bits * 16.0) as usize;
            add(h, &format!("{prefix}.trellis"), "I16", &[k / 16, n / 16, b]);
            add(h, &format!("{prefix}.suh"), "F16", &[k]);
            add(h, &format!("{prefix}.svh"), "F16", &[n]);
            add(h, &format!("{prefix}.mul1"), "I32", &[]);
        }
        let mut h: BTreeMap<String, Value> = BTreeMap::new();
        add(&mut h, "model.word_embeddings.weight", "BF16", &[cfg.vocab, cfg.hidden]);
        add(&mut h, "model.norm.weight", "BF16", &[cfg.hidden]);
        group(&mut h, "lm_head", cfg.hidden, cfg.vocab, 6.0);

        let hidden = cfg.hidden;
        let kda = &cfg.kda;
        let mla = &cfg.mla;
        let ffn = &cfg.ffn;
        for (i, spec) in cfg.layers.iter().enumerate() {
            let p = format!("model.layers.{i}.");
            add(&mut h, &format!("{p}input_layernorm.weight"), "BF16", &[hidden]);
            add(&mut h, &format!("{p}post_attention_layernorm.weight"), "BF16", &[hidden]);
            match spec.mixer {
                crate::family::Mixer::Kda => {
                    let a = format!("{p}attention.");
                    let channels = kda.channels();
                    group(&mut h, &format!("{a}qkv_proj"), hidden, 3 * channels, 4.0);
                    group(&mut h, &format!("{a}f_proj"), hidden, channels, 4.0);
                    group(&mut h, &format!("{a}g_proj"), hidden, channels, 4.0);
                    group(&mut h, &format!("{a}o_proj"), channels, hidden, 5.0);
                    add(&mut h, &format!("{a}b_proj.weight"), "BF16", &[kda.heads, hidden]);
                    add(&mut h, &format!("{a}A_log"), "F32", &[kda.heads]);
                    add(&mut h, &format!("{a}dt_bias"), "F32", &[channels]);
                    add(&mut h, &format!("{a}q_conv1d.weight"), "BF16", &[channels, 1, kda.conv_kernel]);
                    add(&mut h, &format!("{a}k_conv1d.weight"), "BF16", &[channels, 1, kda.conv_kernel]);
                    add(&mut h, &format!("{a}v_conv1d.weight"), "BF16", &[channels, 1, kda.conv_kernel]);
                    add(&mut h, &format!("{a}o_norm.weight"), "BF16", &[kda.head_dim]);
                }
                crate::family::Mixer::MlaDsa { .. } => {
                    let a = format!("{p}attention.");
                    group(&mut h, &format!("{a}q_a_proj"), hidden, mla.q_lora_rank, 5.0);
                    add(&mut h, &format!("{a}q_a_layernorm.weight"), "BF16", &[mla.q_lora_rank]);
                    group(&mut h, &format!("{a}q_b_proj"), mla.q_lora_rank, mla.heads * mla.qk_head_dim(), 4.0);
                    group(&mut h, &format!("{a}kv_a_proj_with_mqa"), hidden, mla.kv_a_group_n(), 5.0);
                    add(&mut h, &format!("{a}kv_a_layernorm.weight"), "BF16", &[mla.kv_lora_rank]);
                    add(
                        &mut h,
                        &format!("{a}kv_b_proj.weight"),
                        "F16",
                        &[mla.heads * (mla.qk_nope_head_dim + mla.v_head_dim), mla.kv_lora_rank],
                    );
                    add(&mut h, &format!("{a}g_proj.weight"), "BF16", &[mla.heads, hidden]);
                    group(&mut h, &format!("{a}dense"), mla.heads * mla.v_head_dim, hidden, 5.0);
                }
                other => panic!("unexpected mixer {other:?}"),
            }
            match spec.ffn {
                crate::family::Ffn::Dense => {
                    let m = format!("{p}mlp.");
                    group(&mut h, &format!("{m}gate_proj"), hidden, ffn.dense_intermediate, 4.0);
                    group(&mut h, &format!("{m}up_proj"), hidden, ffn.dense_intermediate, 4.0);
                    group(&mut h, &format!("{m}down_proj"), ffn.dense_intermediate, hidden, 4.0);
                }
                crate::family::Ffn::Moe => {
                    let m = format!("{p}mlp.");
                    add(&mut h, &format!("{m}gate.weight"), "BF16", &[ffn.routed_experts, hidden]);
                    add(&mut h, &format!("{m}gate.expert_bias"), "F32", &[ffn.routed_experts]);
                    // mixed-K: gate/up 2.0 in the first four layers, 1.5 after; down 2.0 / 1.5
                    let ku = if i < 4 { 2.0 } else { 1.5 };
                    let kd = if i < 4 { 2.0 } else { 1.5 };
                    for e in 0..ffn.routed_experts {
                        let ep = format!("{m}experts.{e}.");
                        group(&mut h, &format!("{ep}gate_proj"), hidden, ffn.moe_intermediate, ku);
                        group(&mut h, &format!("{ep}up_proj"), hidden, ffn.moe_intermediate, ku);
                        group(&mut h, &format!("{ep}down_proj"), ffn.moe_intermediate, hidden, kd);
                    }
                    let s = format!("{m}shared_experts.");
                    group(&mut h, &format!("{s}gate_proj"), hidden, ffn.moe_intermediate, 4.0);
                    group(&mut h, &format!("{s}up_proj"), hidden, ffn.moe_intermediate, 4.0);
                    group(&mut h, &format!("{s}down_proj"), ffn.moe_intermediate, hidden, 5.0);
                }
                other => panic!("unexpected ffn {other:?}"),
            }
        }
        for (j, _) in cfg.mtp.iter().enumerate() {
            let i = cfg.mtp_layer_index(j);
            let p = format!("model.layers.{i}.");
            add(&mut h, &format!("{p}eh_proj.weight"), "BF16", &[hidden, 2 * hidden]);
            add(&mut h, &format!("{p}enorm.weight"), "BF16", &[hidden]);
            add(&mut h, &format!("{p}hnorm.weight"), "BF16", &[hidden]);
            add(&mut h, &format!("{p}final_layernorm.weight"), "BF16", &[hidden]);
            // the MTP layer body: Gated MLA + MoE (the headers for layer 72 use the same slots as the trunk)
            add(&mut h, &format!("{p}input_layernorm.weight"), "BF16", &[hidden]);
            add(&mut h, &format!("{p}post_attention_layernorm.weight"), "BF16", &[hidden]);
            let a = format!("{p}attention.");
            group(&mut h, &format!("{a}q_a_proj"), hidden, mla.q_lora_rank, 5.0);
            add(&mut h, &format!("{a}q_a_layernorm.weight"), "BF16", &[mla.q_lora_rank]);
            group(&mut h, &format!("{a}q_b_proj"), mla.q_lora_rank, mla.heads * mla.qk_head_dim(), 4.0);
            group(&mut h, &format!("{a}kv_a_proj_with_mqa"), hidden, mla.kv_a_group_n(), 5.0);
            add(&mut h, &format!("{a}kv_a_layernorm.weight"), "BF16", &[mla.kv_lora_rank]);
            add(
                &mut h,
                &format!("{a}kv_b_proj.weight"),
                "F16",
                &[mla.heads * (mla.qk_nope_head_dim + mla.v_head_dim), mla.kv_lora_rank],
            );
            add(&mut h, &format!("{a}g_proj.weight"), "BF16", &[mla.heads, hidden]);
            group(&mut h, &format!("{a}dense"), mla.heads * mla.v_head_dim, hidden, 5.0);
            let m = format!("{p}mlp.");
            add(&mut h, &format!("{m}gate.weight"), "BF16", &[ffn.routed_experts, hidden]);
            add(&mut h, &format!("{m}gate.expert_bias"), "F32", &[ffn.routed_experts]);
            for e in 0..ffn.routed_experts {
                let ep = format!("{m}experts.{e}.");
                group(&mut h, &format!("{ep}gate_proj"), hidden, ffn.moe_intermediate, 1.5);
                group(&mut h, &format!("{ep}up_proj"), hidden, ffn.moe_intermediate, 1.5);
                group(&mut h, &format!("{ep}down_proj"), ffn.moe_intermediate, hidden, 1.5);
            }
            let s = format!("{m}shared_experts.");
            group(&mut h, &format!("{s}gate_proj"), hidden, ffn.moe_intermediate, 4.0);
            group(&mut h, &format!("{s}up_proj"), hidden, ffn.moe_intermediate, 4.0);
            group(&mut h, &format!("{s}down_proj"), ffn.moe_intermediate, hidden, 5.0);
        }
        h
    }

    #[test]
    fn the_synthetic_pack_maps_with_zero_problems() {
        let cfg = real_config();
        let headers = synthetic_headers(&cfg);
        let inv = PackInventory::from_headers(&headers).unwrap();
        assert!(inv.refused.is_empty(), "refused: {:?}", inv.refused);
        let weights = BailingWeights::map(&inv, &cfg).unwrap();
        let plan = weights.load_plan().unwrap();
        plan.validate().unwrap();
        // every tensor of the pack is claimed exactly once
        assert_eq!(inv.plain.len() + inv.groups.len(), 771 + 106_496);
        // the plan's items cover every uploaded part: each group's 3 uploaded parts, and every plain tensor but the embedding
        // (which is a sidecar)
        assert_eq!(plan.items.len(), inv.groups.len() * 3 + (inv.plain.len() - 1));
        // the embedding is in its own sidecar region (never allocated: 0 bytes)
        assert_eq!(plan.regions.iter().filter(|r| r.kind == RegionKind::Sidecar).count(), 1);
        let sidecar = plan.regions.iter().find(|r| r.kind == RegionKind::Sidecar).unwrap();
        assert_eq!(sidecar.bytes, 0);
    }

    #[test]
    fn a_tensor_that_is_not_in_the_pack_is_missing_by_name() {
        let cfg = real_config();
        let mut headers = synthetic_headers(&cfg);
        headers.remove("model.layers.0.attention.b_proj.weight");
        let inv = PackInventory::from_headers(&headers).unwrap();
        let e = BailingWeights::map(&inv, &cfg).unwrap_err();
        assert!(e
            .problems
            .iter()
            .any(|p| matches!(p, Problem::Missing { name, .. } if name == "model.layers.0.attention.b_proj.weight")));
    }

    #[test]
    fn an_unexpected_tensor_is_refused_by_name() {
        let cfg = real_config();
        let mut headers = synthetic_headers(&cfg);
        headers.insert(
            "model.layers.0.attention.surprise.weight".to_string(),
            serde_json::json!({ "dtype": "BF16", "shape": [4], "data_offsets": [0, 1] }),
        );
        let inv = PackInventory::from_headers(&headers).unwrap();
        let e = BailingWeights::map(&inv, &cfg).unwrap_err();
        assert!(e
            .problems
            .iter()
            .any(|p| matches!(p, Problem::Unexpected { name, .. } if name == "model.layers.0.attention.surprise.weight")));
    }

    #[test]
    fn a_wrong_shape_is_a_named_problem() {
        let cfg = real_config();
        let mut headers = synthetic_headers(&cfg);
        headers.insert(
            "model.layers.0.attention.b_proj.weight".to_string(),
            serde_json::json!({ "dtype": "BF16", "shape": [32, 4096], "data_offsets": [0, 1] }),
        );
        let inv = PackInventory::from_headers(&headers).unwrap();
        let e = BailingWeights::map(&inv, &cfg).unwrap_err();
        assert!(e
            .problems
            .iter()
            .any(|p| matches!(p, Problem::Shape { name, .. } if name == "model.layers.0.attention.b_proj.weight")));
    }

    /// The 24-shard stub pack (written by the pack subagent): headers only, no payload. Every real tensor name, shape, dtype and
    /// group parse is read from the shards' headers and mapped; a header-only stub has no data bytes, so the payload-scanning
    /// `scan` cannot price its groups (offsets run past the short files) — the header path is the ground truth here.
    #[test]
    fn the_stub_pack_scans_and_maps_with_zero_problems() {
        let dir = std::env::var("LING31_STUB_PACK")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("ling31_stub_pack"));
        let shards: Vec<_> = std::fs::read_dir(&dir)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if shards.len() != 24 {
            eprintln!("not run: expected 24 stub shards, found {}", shards.len());
            return;
        }
        let cfg = BailingConfig::from_dir(&dir).unwrap();
        let mut headers: BTreeMap<String, Value> = BTreeMap::new();
        for shard in &shards {
            let raw = std::fs::read(shard).unwrap();
            let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
            let hdr: Value = serde_json::from_slice(&raw[8..8 + n]).unwrap();
            let map = hdr.as_object().unwrap();
            for (k, v) in map {
                headers.insert(k.clone(), v.clone());
            }
        }
        let inv = PackInventory::from_headers(&headers).unwrap();
        assert!(inv.refused.is_empty(), "refused groups: {:?}", inv.refused.iter().take(5).collect::<Vec<_>>());
        // the real pack's counts: 106,496 groups (x4 parts) + 771 plain tensors
        assert_eq!(inv.groups.len(), 106_496);
        assert_eq!(inv.plain.len(), 771);
        assert_eq!(inv.tensor_count(), 106_496 * 4 + 771);
        let weights = BailingWeights::map(&inv, &cfg).unwrap();
        let plan = weights.load_plan().unwrap();
        plan.validate().unwrap();
    }
}
