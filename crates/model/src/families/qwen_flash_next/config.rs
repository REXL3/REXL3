//! Qwen3.8-Flash-Next configuration (`Qwen4ExpForConditionalGeneration`, model type `qwen4_exp`): parsing and the shape checks
//! that refuse a pack the kernels were not built for, before any weight is read.
//!
//! The model, as far as the engine is concerned: 48 decoder layers of gated delta net (36) and grouped-query attention with a QSA
//! sparse-attention indexer (12, every fourth), a 512-expert top-10 MoE with a shared expert on every layer, four low-rank
//! hyper-connection residual streams, a hashed n-gram embedding stage (PLE) before one layer, and one MTP layer.

use crate::family::{Ffn, LayerSpec, Mixer, Residual};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct FlashNextConfig {
    pub hidden: usize,
    pub layers: usize,
    pub layer_types: Vec<String>,
    pub vocab: usize,
    pub rms_norm_eps: f64,
    // full attention
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    // gated delta net
    pub lin_key_heads: usize,
    pub lin_value_heads: usize,
    pub lin_key_dim: usize,
    pub lin_value_dim: usize,
    pub lin_conv_kernel: usize,
    // mixture of experts
    pub experts: usize,
    pub experts_per_tok: usize,
    pub moe_intermediate: usize,
    pub shared_intermediate: usize,
    // hyper-connections
    pub hc_streams: usize,
    pub hc_lowrank: usize,
    // QSA indexer
    pub indexer_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_kv_heads: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    // n-gram embedding stage
    pub ple_layer_ids: Vec<usize>,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ple_embed_dim: usize,
    pub ple_conv_kernel: usize,
    pub eos_token_id: u32,
    // drafting
    pub mtp_layers: usize,
    pub quant_method: String,
}

impl FlashNextConfig {
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let root: Value = serde_json::from_str(&text).map_err(|e| format!("config.json: {e}"))?;
        Self::from_json(&root)
    }

    /// Every field is read from the pack; a missing key is an error naming it (no silent default for a shape).
    pub fn from_json(root: &Value) -> Result<Self, String> {
        let t = root.get("text_config").unwrap_or(root);
        let u = |k: &str| -> Result<usize, String> {
            t.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| format!("config.json: text_config.{k} is missing or not an integer"))
        };
        let f = |k: &str| -> Result<f64, String> {
            t.get(k).and_then(|v| v.as_f64()).ok_or_else(|| format!("config.json: text_config.{k} is missing or not a number"))
        };
        let rope = t.get("rope_parameters").ok_or("config.json: text_config.rope_parameters is missing")?;
        let rope_theta =
            rope.get("rope_theta").and_then(|v| v.as_f64()).ok_or("config.json: rope_parameters.rope_theta is missing")?;
        let partial = rope.get("partial_rotary_factor").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let head_dim = u("head_dim")?;
        let layer_types: Vec<String> = t
            .get("layer_types")
            .and_then(|v| v.as_array())
            .ok_or("config.json: text_config.layer_types is missing")?
            .iter()
            .map(|x| x.as_str().map(String::from).ok_or_else(|| "config.json: layer_types holds a non-string".to_string()))
            .collect::<Result<_, _>>()?;
        let ple_layer_ids = t
            .get("ple_layer_ids")
            .and_then(|v| v.as_array())
            .ok_or("config.json: text_config.ple_layer_ids is missing")?
            .iter()
            .map(|x| x.as_u64().map(|v| v as usize).ok_or_else(|| "config.json: ple_layer_ids holds a non-integer".to_string()))
            .collect::<Result<_, _>>()?;
        // eos may be one id or a list; the n-gram hashes use the first
        let eos = t.get("eos_token_id").or_else(|| root.get("eos_token_id"));
        let eos_token_id = eos
            .and_then(|v| v.as_u64().or_else(|| v.as_array().and_then(|a| a.first()).and_then(|x| x.as_u64())))
            .ok_or("config.json: eos_token_id is missing")? as u32;
        let quant_method = root
            .get("quantization_config")
            .and_then(|q| q.get("quant_method"))
            .and_then(|v| v.as_str())
            .unwrap_or("none")
            .to_string();
        Ok(Self {
            hidden: u("hidden_size")?,
            layers: u("num_hidden_layers")?,
            layer_types,
            vocab: u("vocab_size")?,
            rms_norm_eps: f("rms_norm_eps")?,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim,
            rotary_dim: (head_dim as f64 * partial).round() as usize,
            rope_theta,
            max_position_embeddings: u("max_position_embeddings")?,
            lin_key_heads: u("linear_num_key_heads")?,
            lin_value_heads: u("linear_num_value_heads")?,
            lin_key_dim: u("linear_key_head_dim")?,
            lin_value_dim: u("linear_value_head_dim")?,
            lin_conv_kernel: u("linear_conv_kernel_dim")?,
            experts: u("num_experts")?,
            experts_per_tok: u("num_experts_per_tok")?,
            moe_intermediate: u("moe_intermediate_size")?,
            shared_intermediate: u("shared_expert_intermediate_size")?,
            hc_streams: u("hc_count")?,
            hc_lowrank: u("hc_lowrank")?,
            indexer_heads: u("indexer_n_heads")?,
            indexer_head_dim: u("indexer_head_dim")?,
            indexer_kv_heads: u("indexer_kv_heads")?,
            indexer_budget: u("indexer_budget")?,
            indexer_compress_ratio: u("indexer_compress_ratio")?,
            ple_layer_ids,
            ngram_size: u("ngram_size")?,
            heads_per_ngram: u("heads_per_ngram")?,
            ple_embed_dim: u("ple_embed_dim")?,
            ple_conv_kernel: u("ple_conv_kernel_size")?,
            eos_token_id,
            mtp_layers: t.get("mtp_num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
            quant_method,
        })
    }

    pub fn full_layers(&self) -> usize {
        self.layer_types.iter().filter(|t| t.as_str() == "full_attention").count()
    }

    pub fn linear_layers(&self) -> usize {
        self.layer_types.iter().filter(|t| t.as_str() == "linear_attention").count()
    }

    /// Channels of a gated delta net layer's convolution (q, k and v).
    pub fn conv_channels(&self) -> usize {
        2 * self.lin_key_heads * self.lin_key_dim + self.lin_value_heads * self.lin_value_dim
    }

    /// The trunk's layers, in order: every layer is MoE with four hyper-connection streams.
    pub fn layer_specs(&self) -> Vec<LayerSpec> {
        let residual = Residual::Mhc { streams: self.hc_streams };
        self.layer_types
            .iter()
            .map(|t| LayerSpec { mixer: if t == "full_attention" { Mixer::Gqa } else { Mixer::Gdn }, ffn: Ffn::Moe, residual })
            .collect()
    }

    /// The MTP layer(s): full attention (with its own indexer) and the same MoE.
    pub fn mtp_specs(&self) -> Vec<LayerSpec> {
        let residual = Residual::Mhc { streams: self.hc_streams };
        (0..self.mtp_layers).map(|_| LayerSpec { mixer: Mixer::Gqa, ffn: Ffn::Moe, residual }).collect()
    }

    /// The geometry the Flash-Next kernels implement. Anything else is refused by name before a weight is read: a pack with
    /// another shape would load and then compute a different function.
    pub fn check_shapes(&self) -> Result<(), String> {
        let mut bad = Vec::new();
        let mut want = |name: &str, got: usize, exp: usize| {
            if got != exp {
                bad.push(format!("{name} {got} (the kernels implement {exp})"));
            }
        };
        want("hidden_size", self.hidden, 2560);
        want("num_attention_heads", self.heads, 24);
        want("num_key_value_heads", self.kv_heads, 2);
        want("head_dim", self.head_dim, 256);
        want("rotary dims", self.rotary_dim, 64);
        want("linear_num_key_heads", self.lin_key_heads, 16);
        want("linear_num_value_heads", self.lin_value_heads, 48);
        want("linear_key_head_dim", self.lin_key_dim, 128);
        want("linear_value_head_dim", self.lin_value_dim, 128);
        want("linear_conv_kernel_dim", self.lin_conv_kernel, 4);
        want("hc_count", self.hc_streams, 4);
        want("indexer_n_heads", self.indexer_heads, 4);
        want("indexer_head_dim", self.indexer_head_dim, 128);
        want("indexer_kv_heads", self.indexer_kv_heads, 1);
        want("indexer_budget", self.indexer_budget, 2048);
        want("indexer_compress_ratio", self.indexer_compress_ratio, 4);
        want("ngram_size", self.ngram_size, 3);
        want("ple_conv_kernel_size", self.ple_conv_kernel, 4);
        want("ple_embed_dim", self.ple_embed_dim, self.hidden);
        want("ple_layer_ids (count)", self.ple_layer_ids.len(), 1);
        if self.layer_types.len() != self.layers {
            bad.push(format!("layer_types has {} entries for {} layers", self.layer_types.len(), self.layers));
        }
        if let Some(t) = self.layer_types.iter().find(|t| t.as_str() != "full_attention" && t.as_str() != "linear_attention") {
            bad.push(format!("layer type `{t}`"));
        }
        if self.ple_layer_ids.first().is_some_and(|&i| i == 0 || i > self.layers) {
            bad.push(format!("ple_layer_ids {:?} (the stage runs before decoder layer id - 1)", self.ple_layer_ids));
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(format!("Qwen3.8-Flash-Next pack not supported: {}", bad.join("; ")))
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub const FLASH_NEXT_CONFIG: &str = include_str!("../../../../../testdata/qwen_flash_next/config.json");

    fn real() -> FlashNextConfig {
        FlashNextConfig::from_json(&serde_json::from_str(FLASH_NEXT_CONFIG).unwrap()).unwrap()
    }

    #[test]
    fn the_real_config_parses_to_the_shapes_the_kernels_implement() {
        let c = real();
        assert_eq!((c.hidden, c.layers, c.vocab), (2560, 48, 248_320));
        assert_eq!((c.full_layers(), c.linear_layers()), (12, 36));
        assert_eq!((c.heads, c.kv_heads, c.head_dim, c.rotary_dim), (24, 2, 256, 64));
        assert_eq!(c.rope_theta, 1e7);
        assert_eq!((c.experts, c.experts_per_tok, c.moe_intermediate, c.shared_intermediate), (512, 10, 640, 640));
        assert_eq!((c.hc_streams, c.hc_lowrank), (4, 320));
        assert_eq!((c.indexer_heads, c.indexer_head_dim, c.indexer_budget, c.indexer_compress_ratio), (4, 128, 2048, 4));
        assert_eq!((c.ple_layer_ids.clone(), c.ngram_size, c.heads_per_ngram, c.ple_embed_dim), (vec![2], 3, 8, 2560));
        assert_eq!((c.eos_token_id, c.mtp_layers, c.quant_method.as_str()), (248_044, 1, "exl3"));
        assert_eq!(c.conv_channels(), 10_240);
        c.check_shapes().unwrap();
        // every fourth layer is full attention
        let specs = c.layer_specs();
        assert_eq!(specs.len(), 48);
        for (i, s) in specs.iter().enumerate() {
            assert_eq!(s.mixer, if i % 4 == 3 { Mixer::Gqa } else { Mixer::Gdn }, "layer {i}");
            assert_eq!((s.ffn, s.residual), (Ffn::Moe, Residual::Mhc { streams: 4 }));
        }
        assert_eq!(c.mtp_specs().len(), 1);
    }

    #[test]
    fn a_shape_the_kernels_do_not_implement_is_refused_by_name() {
        for (key, val, word) in [
            ("num_key_value_heads", serde_json::json!(4), "num_key_value_heads 4"),
            ("indexer_budget", serde_json::json!(4096), "indexer_budget 4096"),
            ("hc_count", serde_json::json!(2), "hc_count 2"),
            ("ngram_size", serde_json::json!(4), "ngram_size 4"),
        ] {
            let mut v: Value = serde_json::from_str(FLASH_NEXT_CONFIG).unwrap();
            v["text_config"][key] = val;
            let e = FlashNextConfig::from_json(&v).unwrap().check_shapes().unwrap_err();
            assert!(e.contains(word), "{key}: {e}");
        }
    }

    #[test]
    fn a_missing_shape_key_is_an_error_naming_it() {
        let mut v: Value = serde_json::from_str(FLASH_NEXT_CONFIG).unwrap();
        v["text_config"].as_object_mut().unwrap().remove("indexer_head_dim");
        let e = FlashNextConfig::from_json(&v).unwrap_err();
        assert!(e.contains("indexer_head_dim"), "{e}");
    }
}
