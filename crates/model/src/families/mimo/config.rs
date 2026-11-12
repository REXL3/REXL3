//! MiMo-V2.6 (`MiMoV2ForCausalLM`, model type `mimo_v2`): configuration parsing and the shape checks that refuse a pack
//! the kernels were not built for, before any weight is read.
//!
//! The model, as far as the engine is concerned: 48 decoder layers — 9 full-attention layers (0, 5, 11, 17, 23, 29, 35,
//! 41, 47) and 39 sliding-window layers (window 128) with a learned per-head sink logit — an asymmetric attention head
//! (QK head dim 192, V head dim 128, partial rotary 64), V scaled by `attention_value_scale` before the cache write, a
//! 256-expert top-8 sigmoid-`noaux_tc` MoE on layers 1-47 (no shared expert; layer 0 is dense), and 3 MTP layers whose
//! weights are FP8-e4m3 with 128x128 block scales in the original release (not in the EXL3 pack).
//!
//! Every fact here was read from the pack's `config.json` (the vcruz305 2.20bpw pack and the original XiaomiMiMo
//! release agree on all of them; `moe_router_dtype` is `bfloat16` in both).

use crate::family::{Ffn, LayerSpec, Mixer, Residual};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct MiMoConfig {
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    /// `hybrid_layer_pattern`: 0 = full attention, 1 = sliding window.
    pub hybrid_pattern: Vec<u8>,
    /// `moe_layer_freq`: 0 = dense MLP, 1 = MoE.
    pub moe_freq: Vec<u8>,
    // full attention
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub v_head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f64,
    // sliding-window attention
    pub swa_heads: usize,
    pub swa_kv_heads: usize,
    pub swa_head_dim: usize,
    pub swa_v_head_dim: usize,
    pub swa_rope_theta: f64,
    pub swa_window: usize,
    pub swa_sink: bool,
    // common attention rules
    pub attention_value_scale: f64,
    // FFN
    pub experts: usize,
    pub experts_per_tok: usize,
    pub moe_intermediate: usize,
    pub dense_intermediate: usize,
    pub router_eps: f64,
    // MTP
    pub mtp_layers: usize,
    pub quant_method: String,
}

impl MiMoConfig {
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let root: Value = serde_json::from_str(&text).map_err(|e| format!("config.json: {e}"))?;
        Self::from_json(&root)
    }

    /// Every field is read from the pack; a missing key is an error naming it (no silent default for a shape).
    pub fn from_json(root: &Value) -> Result<Self, String> {
        let u = |k: &str| -> Result<usize, String> {
            root.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json: {k} is missing or not an integer"))
        };
        let f = |k: &str| -> Result<f64, String> {
            root.get(k).and_then(|v| v.as_f64()).ok_or_else(|| format!("config.json: {k} is missing or not a number"))
        };
        let b = |k: &str| -> Result<bool, String> {
            root.get(k).and_then(|v| v.as_bool()).ok_or_else(|| format!("config.json: {k} is missing or not a boolean"))
        };
        let arr_u8 = |k: &str| -> Result<Vec<u8>, String> {
            root.get(k)
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("config.json: {k} is missing"))?
                .iter()
                .map(|x| {
                    x.as_u64().and_then(|v| u8::try_from(v).ok()).ok_or_else(|| format!("config.json: {k} holds a non-0/1 entry"))
                })
                .collect::<Result<_, _>>()
        };
        let head_dim = u("head_dim")?;
        let partial = f("partial_rotary_factor")?;
        let hybrid_pattern = arr_u8("hybrid_layer_pattern")?;
        let moe_freq = arr_u8("moe_layer_freq")?;
        let quant_method = root
            .get("quantization_config")
            .and_then(|q| q.get("quant_method"))
            .and_then(|v| v.as_str())
            .unwrap_or("none")
            .to_string();
        Ok(Self {
            hidden: u("hidden_size")?,
            layers: u("num_hidden_layers")?,
            vocab: u("vocab_size")?,
            rms_norm_eps: f("layernorm_epsilon")?,
            max_position_embeddings: u("max_position_embeddings")?,
            hybrid_pattern,
            moe_freq,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim,
            v_head_dim: u("v_head_dim")?,
            rotary_dim: (head_dim as f64 * partial).round() as usize,
            rope_theta: f("rope_theta")?,
            swa_heads: u("num_attention_heads")?,
            swa_kv_heads: u("swa_num_key_value_heads")?,
            swa_head_dim: u("swa_head_dim")?,
            swa_v_head_dim: u("swa_v_head_dim")?,
            swa_rope_theta: f("swa_rope_theta")?,
            swa_window: u("attention_chunk_size")?,
            swa_sink: b("add_swa_attention_sink_bias")?,
            attention_value_scale: f("attention_value_scale")?,
            experts: u("n_routed_experts")?,
            experts_per_tok: u("num_experts_per_tok")?,
            moe_intermediate: u("moe_intermediate_size")?,
            dense_intermediate: u("intermediate_size")?,
            router_eps: 1e-20,
            mtp_layers: u("num_nextn_predict_layers")?,
            quant_method,
        })
    }

    /// The trunk's layer recipe from the two patterns; both must be `layers` long.
    pub fn layer_specs(&self) -> Vec<LayerSpec> {
        (0..self.layers)
            .map(|i| LayerSpec {
                mixer: if self.hybrid_pattern[i] == 0 { Mixer::Gqa } else { Mixer::Gdn },
                ffn: if self.moe_freq[i] == 0 { Ffn::Dense } else { Ffn::Moe },
                residual: Residual::Plain,
            })
            .collect()
    }

    /// The MTP layers: sliding-style attention with a sink, dense MLP (a plain residual).
    pub fn mtp_specs(&self) -> Vec<LayerSpec> {
        (0..self.mtp_layers).map(|_| LayerSpec { mixer: Mixer::Gdn, ffn: Ffn::Dense, residual: Residual::Plain }).collect()
    }

    /// Layer indices with full attention (hybrid pattern 0).
    pub fn full_layer_ids(&self) -> Vec<usize> {
        (0..self.layers).filter(|&i| self.hybrid_pattern[i] == 0).collect()
    }

    /// Layer indices with sliding-window attention (hybrid pattern 1).
    pub fn swa_layer_ids(&self) -> Vec<usize> {
        (0..self.layers).filter(|&i| self.hybrid_pattern[i] == 1).collect()
    }

    /// Reject a config whose shapes the (future) kernels do not implement, naming every problem.
    pub fn check_shapes(&self) -> Result<(), String> {
        let mut errs = Vec::new();
        if self.hybrid_pattern.len() != self.layers {
            errs.push(format!("hybrid_layer_pattern has {} entries for {} layers", self.hybrid_pattern.len(), self.layers));
        }
        if self.moe_freq.len() != self.layers {
            errs.push(format!("moe_layer_freq has {} entries for {} layers", self.moe_freq.len(), self.layers));
        }
        if !self.hybrid_pattern.iter().all(|&v| v <= 1) {
            errs.push("hybrid_layer_pattern holds a value other than 0/1".into());
        }
        if !self.moe_freq.iter().all(|&v| v <= 1) {
            errs.push("moe_layer_freq holds a value other than 0/1".into());
        }
        if self.heads != 64 || self.swa_heads != 64 {
            errs.push(format!("num_attention_heads {} (swa {}): only 64 is built", self.heads, self.swa_heads));
        }
        if self.head_dim != 192 || self.v_head_dim != 128 || self.swa_head_dim != 192 || self.swa_v_head_dim != 128 {
            errs.push(format!(
                "asymmetric head dims qk {}/{} v {}/{}: only 192/128 is built",
                self.head_dim, self.swa_head_dim, self.v_head_dim, self.swa_v_head_dim
            ));
        }
        if self.kv_heads != 4 {
            errs.push(format!("num_key_value_heads {} on full-attention layers: only 4 is built", self.kv_heads));
        }
        if self.swa_kv_heads != 8 {
            errs.push(format!("swa_num_key_value_heads {}: only 8 is built", self.swa_kv_heads));
        }
        if self.experts != 256 || self.experts_per_tok != 8 {
            errs.push(format!("MoE {} experts top-{}: only 256 top-8 is built", self.experts, self.experts_per_tok));
        }
        if self.vocab != 152_576 {
            errs.push(format!("vocab_size {}: only 152,576 is built", self.vocab));
        }
        if self.max_position_embeddings != 1_048_576 {
            errs.push(format!("max_position_embeddings {}: only 1,048,576 is built", self.max_position_embeddings));
        }
        if self.swa_window != 128 {
            errs.push(format!("attention_chunk_size {} (sliding window): only 128 is built", self.swa_window));
        }
        if !self.swa_sink {
            errs.push("add_swa_attention_sink_bias false: the SWA kernel assumes a learned sink".into());
        }
        if self.quant_method != "exl3" {
            errs.push(format!("quantization_config.quant_method {}: not an EXL3 pack", self.quant_method));
        }
        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs.join("; "))
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// The vcruz305 2.20bpw pack's config, every field the parser reads (facts verified 2026-10-04).
    pub const MIMO_CONFIG: &str = r#"{
        "architectures": ["MiMoV2ForCausalLM"],
        "model_type": "mimo_v2",
        "num_hidden_layers": 48,
        "hidden_size": 4096,
        "vocab_size": 152576,
        "max_position_embeddings": 1048576,
        "layernorm_epsilon": 1e-06,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 192,
        "v_head_dim": 128,
        "partial_rotary_factor": 0.334,
        "rope_theta": 10000000.0,
        "swa_num_key_value_heads": 8,
        "swa_head_dim": 192,
        "swa_v_head_dim": 128,
        "swa_rope_theta": 10000.0,
        "add_swa_attention_sink_bias": true,
        "add_full_attention_sink_bias": false,
        "attention_value_scale": 0.707,
        "attention_chunk_size": 128,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "n_shared_experts": null,
        "intermediate_size": 16384,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": true,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "moe_router_dtype": "bfloat16",
        "num_nextn_predict_layers": 3,
        "hybrid_layer_pattern": [0,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,0],
        "moe_layer_freq": [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        "quantization_config": {"quant_method": "exl3"}
    }"#;

    #[test]
    fn the_config_parses_and_checks() {
        let root: Value = serde_json::from_str(MIMO_CONFIG).unwrap();
        let c = MiMoConfig::from_json(&root).unwrap();
        assert_eq!((c.layers, c.hidden, c.vocab), (48, 4096, 152576));
        assert_eq!((c.rotary_dim, c.head_dim, c.v_head_dim), (64, 192, 128));
        assert_eq!(c.full_layer_ids(), vec![0, 5, 11, 17, 23, 29, 35, 41, 47]);
        assert_eq!(c.swa_layer_ids().len(), 39);
        assert_eq!(c.layer_specs().len(), 48);
        assert_eq!(c.mtp_specs().len(), 3);
        c.check_shapes().unwrap();
    }

    #[test]
    fn wrong_shapes_are_refused_by_name() {
        for (patch, needle) in [
            (serde_json::json!({"num_key_value_heads": 8}), "num_key_value_heads 8"),
            (serde_json::json!({"n_routed_experts": 128}), "only 256 top-8"),
            (serde_json::json!({"quantization_config": {"quant_method": "gguf"}}), "not an EXL3 pack"),
            (serde_json::json!({"vocab_size": 152000}), "vocab_size"),
            (serde_json::json!({"head_dim": 128}), "asymmetric head dims"),
            (serde_json::json!({"swa_num_key_value_heads": 4}), "swa_num_key_value_heads 4"),
        ] {
            let mut v: Value = serde_json::from_str(MIMO_CONFIG).unwrap();
            for (k, val) in patch.as_object().unwrap() {
                v[k] = val.clone();
            }
            let e = MiMoConfig::from_json(&v).and_then(|c| c.check_shapes()).err().unwrap();
            assert!(e.contains(needle), "refusal {e:?} does not name {needle:?}");
        }
    }

    #[test]
    fn a_missing_key_is_an_error_naming_it() {
        let mut v: Value = serde_json::from_str(MIMO_CONFIG).unwrap();
        v.as_object_mut().unwrap().remove("swa_rope_theta");
        let e = MiMoConfig::from_json(&v).err().unwrap();
        assert!(e.contains("swa_rope_theta"), "{e}");
    }
}
