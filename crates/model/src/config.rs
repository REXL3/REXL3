//! Model configuration read from a pack's `config.json`, and the memory arithmetic derived from it.

use crate::family::{Ffn, LayerSpec, Mixer, Residual};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated delta net: a recurrent state, no KV cache.
    Linear,
    /// Softmax attention with an output gate: grouped-query KV cache.
    Full,
}

/// The text model of a Qwen3.5-style hybrid (Qwen3.8-27B).
#[derive(Clone, Debug, PartialEq)]
pub struct QwenConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub layers: Vec<LayerKind>,
    pub rms_eps: f64,
    pub max_positions: usize,
    // full attention
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f64,
    // gated delta net
    pub lin_key_heads: usize,
    pub lin_value_heads: usize,
    pub lin_key_dim: usize,
    pub lin_value_dim: usize,
    pub conv_kernel: usize,
    pub eos: Vec<u32>,
}

fn u(v: &Value, k: &str) -> Result<usize, String> {
    v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).ok_or_else(|| format!("config.json: missing or invalid `{k}`"))
}

fn f(v: &Value, k: &str) -> Result<f64, String> {
    v.get(k).and_then(|x| x.as_f64()).ok_or_else(|| format!("config.json: missing or invalid `{k}`"))
}

impl QwenConfig {
    pub fn from_json(root: &Value) -> Result<Self, String> {
        let t = root.get("text_config").unwrap_or(root);
        let types = t.get("layer_types").and_then(|x| x.as_array()).ok_or("config.json: missing `layer_types`")?;
        let layers = types
            .iter()
            .map(|x| match x.as_str() {
                Some("linear_attention") => Ok(LayerKind::Linear),
                Some("full_attention") => Ok(LayerKind::Full),
                other => Err(format!("config.json: unsupported layer type {other:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if layers.len() != u(t, "num_hidden_layers")? {
            return Err("config.json: layer_types does not match num_hidden_layers".into());
        }
        let head_dim = u(t, "head_dim")?;
        let rope = t.get("rope_parameters").ok_or("config.json: missing `rope_parameters`")?;
        let partial = f(rope, "partial_rotary_factor").or_else(|_| f(t, "partial_rotary_factor"))?;
        let eos = match t.get("eos_token_id") {
            Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
            _ => vec![],
        };
        let c = Self {
            hidden: u(t, "hidden_size")?,
            intermediate: u(t, "intermediate_size")?,
            vocab: u(t, "vocab_size")?,
            layers,
            rms_eps: f(t, "rms_norm_eps")?,
            max_positions: u(t, "max_position_embeddings")?,
            heads: u(t, "num_attention_heads")?,
            kv_heads: u(t, "num_key_value_heads")?,
            head_dim,
            rotary_dim: (head_dim as f64 * partial).round() as usize,
            rope_theta: f(rope, "rope_theta")?,
            lin_key_heads: u(t, "linear_num_key_heads")?,
            lin_value_heads: u(t, "linear_num_value_heads")?,
            lin_key_dim: u(t, "linear_key_head_dim")?,
            lin_value_dim: u(t, "linear_value_head_dim")?,
            conv_kernel: u(t, "linear_conv_kernel_dim")?,
            eos,
        };
        c.validate()?;
        Ok(c)
    }

    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("{}: {e}", dir.join("config.json").display()))?;
        Self::from_json(&serde_json::from_str(&text).map_err(|e| format!("config.json: {e}"))?)
    }

    fn validate(&self) -> Result<(), String> {
        if self.lin_value_heads % self.lin_key_heads != 0 {
            return Err("linear value heads must be a multiple of key heads".into());
        }
        if self.heads % self.kv_heads != 0 {
            return Err("attention heads must be a multiple of KV heads".into());
        }
        if self.rotary_dim > self.head_dim || self.rotary_dim % 2 != 0 {
            return Err(format!("rotary dim {} is invalid for head dim {}", self.rotary_dim, self.head_dim));
        }
        Ok(())
    }

    /// The layer composition: gated delta net or grouped-query attention, a dense feed-forward block, a plain residual.
    pub fn layer_specs(&self) -> Vec<LayerSpec> {
        self.layers
            .iter()
            .map(|k| LayerSpec {
                mixer: match k {
                    LayerKind::Linear => Mixer::Gdn,
                    LayerKind::Full => Mixer::Gqa,
                },
                ffn: Ffn::Dense,
                residual: Residual::Plain,
            })
            .collect()
    }

    pub fn full_layers(&self) -> usize {
        self.layers.iter().filter(|k| **k == LayerKind::Full).count()
    }
    pub fn linear_layers(&self) -> usize {
        self.layers.len() - self.full_layers()
    }
    /// Channels through the depthwise conv: q and k (key heads) plus v (value heads).
    pub fn conv_channels(&self) -> usize {
        2 * self.lin_key_heads * self.lin_key_dim + self.lin_value_heads * self.lin_value_dim
    }
    /// KV bytes per token over all full-attention layers (bf16 K and V).
    pub fn kv_bytes_per_token(&self) -> u64 {
        (self.full_layers() * self.kv_heads * self.head_dim * 2 * 2) as u64
    }
    /// Recurrent bytes per sequence: the delta-rule state (fp32) and the conv history (bf16) of every linear layer.
    pub fn state_bytes_per_seq(&self) -> u64 {
        let rec = self.lin_value_heads * self.lin_value_dim * self.lin_key_dim * 4;
        let conv = (self.conv_kernel - 1) * self.conv_channels() * 2;
        (self.linear_layers() * (rec + conv)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample() -> Value {
        let mut types = Vec::new();
        for i in 0..64 {
            types.push(if i % 4 == 3 { "full_attention" } else { "linear_attention" });
        }
        json!({"text_config": {
            "hidden_size": 5120, "intermediate_size": 17408, "vocab_size": 248320, "num_hidden_layers": 64, "layer_types": types,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 262144, "num_attention_heads": 24, "num_key_value_heads": 4,
            "head_dim": 256, "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000},
            "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "eos_token_id": 248044}})
    }

    #[test]
    fn qwen38_27b_numbers() {
        let c = QwenConfig::from_json(&sample()).unwrap();
        assert_eq!((c.full_layers(), c.linear_layers()), (16, 48));
        assert_eq!(c.rotary_dim, 64);
        assert_eq!(c.conv_channels(), 10240);
        // 16 layers x 4 kv heads x 256 x 2 (K, V) x 2 bytes = 64 KiB per token: 16 GiB at 262,144 tokens
        assert_eq!(c.kv_bytes_per_token(), 65536);
        assert_eq!(c.kv_bytes_per_token() * 262144, 16 << 30);
        // 48 layers x (48 x 128 x 128 x 4 B + 3 x 10240 x 2 B)
        assert_eq!(c.state_bytes_per_seq(), 48 * (48 * 128 * 128 * 4 + 3 * 10240 * 2));
    }

    #[test]
    fn the_real_qwen_config_parses_to_the_hybrid_composition() {
        let c =
            QwenConfig::from_json(&serde_json::from_str(include_str!("../../../testdata/qwen38/config.json")).unwrap()).unwrap();
        assert_eq!((c.hidden, c.vocab, c.layers.len(), c.heads, c.kv_heads, c.head_dim), (5120, 248320, 64, 24, 4, 256));
        // the composition in the family seam's terms: GDN on layers 0-2 of every four, GQA on every fourth, dense FFN, plain residual
        let specs = c.layer_specs();
        assert_eq!(specs.len(), 64);
        for (i, s) in specs.iter().enumerate() {
            assert_eq!(s.mixer, if i % 4 == 3 { Mixer::Gqa } else { Mixer::Gdn }, "layer {i}");
            assert_eq!((s.ffn, s.residual), (Ffn::Dense, Residual::Plain), "layer {i}");
        }
        assert_eq!(specs.iter().filter(|s| s.mixer == Mixer::Gqa).count(), c.full_layers());
    }

    #[test]
    fn rejects_inconsistent_configs() {
        let mut v = sample();
        v["text_config"]["num_hidden_layers"] = json!(63);
        assert!(QwenConfig::from_json(&v).is_err());
        let mut v = sample();
        v["text_config"]["linear_num_value_heads"] = json!(47);
        assert!(QwenConfig::from_json(&v).unwrap_err().contains("multiple"));
        let mut v = sample();
        v["text_config"]["layer_types"][0] = json!("sliding_attention");
        assert!(QwenConfig::from_json(&v).unwrap_err().contains("unsupported layer type"));
    }
}
