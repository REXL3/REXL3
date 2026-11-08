//! `BailingMoeV3ForCausalLM` (`bailing_hybrid` / `bailing_moe_v3`) configuration: parse `config.json` into a layer recipe and
//! the numeric parameters the family's weights, memory model and forward need.
//!
//! The layer recipe (from `config.json`):
//! - 72 trunk layers. `lid` 0..3: KDA mixer + dense SwiGLU. `lid` 4..71: KDA mixer when `(lid + 1) % 8 != 0`, Gated MLA when
//!   `(lid + 1) % 8 == 0`, with a 512-expert routed MoE + 1 shared expert. All residuals are plain.
//! - MTP `NextN` layer 72 (when `num_nextn_predict_layers == 1`): Gated MLA + MoE on a plain residual (spec §3).
//! - `expert_swiglu_limit_list` / `share_expert_swiglu_limit_list`: per-layer SwiGLU clamp limits (`10.0` on routed 65..=71,
//!   shared 64..=71).

use crate::family::{Ffn, LayerSpec, Mixer, Residual};
use serde_json::Value;

const DEFAULT_FFN_DENSE: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlaConfig {
    pub heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rope_theta: f64,
    pub rope_interleave: bool,
}

impl MlaConfig {
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
    /// The width of the EXL3 group `kv_a_proj_with_mqa` in the pack: `kv_lora_rank + qk_rope_head_dim` rounded up to a
    /// multiple of 128 (EXL3 Hadamard blocks; 576 -> 640). The first `kv_lora_rank` columns are the latent, the next
    /// `qk_rope_head_dim` the rope, the trailing padding is ignored (spec §5.2).
    pub fn kv_a_group_n(&self) -> usize {
        (self.kv_lora_rank + self.qk_rope_head_dim).div_ceil(128) * 128
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KdaConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    pub linear_silu: bool,
    pub safe_gate: bool,
    pub lower_bound: f64,
}

impl KdaConfig {
    pub fn channels(&self) -> usize {
        self.heads * self.head_dim
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FfnConfig {
    pub dense_intermediate: usize,
    pub moe_intermediate: usize,
    pub routed_experts: usize,
    pub experts_per_token: usize,
    pub shared_experts: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BailingConfig {
    pub hidden: usize,
    pub vocab: usize,
    pub rms_eps: f64,
    pub max_positions: usize,
    /// The 72 trunk layers.
    pub layers: Vec<LayerSpec>,
    /// The MTP layer (index 72) when the pack has one.
    pub mtp: Vec<LayerSpec>,
    pub mla: MlaConfig,
    pub kda: KdaConfig,
    pub ffn: FfnConfig,
    /// `expert_swiglu_limit_list` (`[f32; 72]`).
    pub expert_swiglu_limit_list: Vec<f32>,
    /// `share_expert_swiglu_limit_list` (`[f32; 72]`).
    pub share_expert_swiglu_limit_list: Vec<f32>,
    pub eos: Vec<u32>,
    pub pad_token_id: Option<u32>,
}

impl BailingConfig {
    pub fn has_mtp(&self) -> bool {
        !self.mtp.is_empty()
    }

    pub fn mtp_layer_index(&self, i: usize) -> usize {
        self.layers.len() + i
    }

    pub fn kda_layers(&self) -> Vec<usize> {
        (0..self.layers.len()).filter(|&i| self.layers[i].mixer == Mixer::Kda).collect()
    }

    pub fn mla_layers(&self) -> Vec<usize> {
        (0..self.layers.len()).filter(|&i| self.layers[i].mixer != Mixer::Kda).collect()
    }

    pub fn moe_layers(&self) -> Vec<usize> {
        (0..self.layers.len()).filter(|&i| self.layers[i].ffn == Ffn::Moe).collect()
    }

    /// `swiglu_limit` for a layer's routed experts (0 when the layer has none).
    pub fn routed_swiglu_limit(&self, layer: usize) -> f32 {
        self.expert_swiglu_limit_list.get(layer).copied().unwrap_or(0.0)
    }

    /// `swiglu_limit` for a layer's shared expert (0 when the layer has none).
    pub fn shared_swiglu_limit(&self, layer: usize) -> f32 {
        self.share_expert_swiglu_limit_list.get(layer).copied().unwrap_or(0.0)
    }

    /// KDA state bytes per layer, one buffer: `2 * heads * head_dim * head_dim * 4`.
    pub fn kda_state_bytes_one_buffer(&self) -> u64 {
        let k = &self.kda;
        (2 * k.heads * k.head_dim * k.head_dim * 4) as u64
    }

    /// KDA conv history bytes per layer: `channels * conv_kernel * 2` (u16 history).
    pub fn kda_conv_history_bytes_per_layer(&self) -> u64 {
        (self.kda.channels() * self.kda.conv_kernel * 2) as u64
    }

    /// KDA record (delta-state) bytes per row per layer: the raw bf16 q|k|v rows
    /// `3 * channels * 2` + the fp32 decay `channels * 4` + the fp32 beta `heads * 4`.
    pub fn kda_record_bytes_per_row_per_layer(&self) -> u64 {
        let channels = self.kda.channels();
        (3 * channels * 2 + channels * 4 + self.kda.heads * 4) as u64
    }

    /// MLA KV bytes per token per layer: latent (fp16 `kv_lora_rank`) + rope (fp16 `qk_rope_head_dim`).
    pub fn mla_kv_bytes_per_token_per_layer(&self) -> u64 {
        ((self.mla.kv_lora_rank + self.mla.qk_rope_head_dim) * 2) as u64
    }

    pub fn from_json(root: &Value) -> Result<Self, String> {
        let t = root; // Bailing configs are flat (no text_config)
        let tp = "";
        let at = |k: &str| -> Result<&Value, String> { t.get(k).ok_or_else(|| format!("config.json: missing `{k}`")) };
        let u = |k: &str| -> Result<usize, String> {
            at(k)?.as_u64().map(|x| x as usize).ok_or_else(|| format!("config.json: `{k}` must be a non-negative integer"))
        };
        let f =
            |k: &str| -> Result<f64, String> { at(k)?.as_f64().ok_or_else(|| format!("config.json: `{k}` must be a number")) };
        let b = |k: &str| -> Result<bool, String> {
            at(k)?.as_bool().ok_or_else(|| format!("config.json: `{k}` must be true or false"))
        };
        let s =
            |k: &str| -> Result<&str, String> { at(k)?.as_str().ok_or_else(|| format!("config.json: `{k}` must be a string")) };
        let f32_list = |k: &str| -> Result<Vec<f32>, String> {
            at(k)?
                .as_array()
                .ok_or_else(|| format!("config.json: `{k}` must be a list"))?
                .iter()
                .map(|x| x.as_f64().map(|v| v as f32).ok_or_else(|| format!("config.json: `{k}` must be a list of numbers")))
                .collect()
        };

        let hidden = u("hidden_size")?;
        let vocab = u("vocab_size")?;
        let n = u("num_hidden_layers")?;
        let first_dense = u("first_k_dense_replace").unwrap_or(DEFAULT_FFN_DENSE);
        let layer_group = u("layer_group_size")?;
        if layer_group == 0 {
            return Err("config.json: `layer_group_size` must be positive".into());
        }

        // ---- layer recipe
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let mixer =
                if (i + 1) % layer_group == 0 { Mixer::MlaDsa { indexer: crate::family::IndexerMode::Full } } else { Mixer::Kda };
            let ffn = if i < first_dense { Ffn::Dense } else { Ffn::Moe };
            layers.push(LayerSpec { mixer, ffn, residual: Residual::Plain });
        }
        // every layer past first_k_dense_replace must be MoE; before it, dense
        for (i, l) in layers.iter().enumerate() {
            let want_dense = i < first_dense;
            if (l.ffn == Ffn::Dense) != want_dense {
                return Err(format!(
                    "config.json: layer {i} is {} but `first_k_dense_replace` is {first_dense}",
                    if l.ffn == Ffn::Dense { "dense" } else { "MoE" }
                ));
            }
        }

        // ---- KDA
        let kda = KdaConfig {
            heads: u("num_attention_heads")?,
            head_dim: u("head_dim")?,
            conv_kernel: u("short_conv_kernel_size")?,
            linear_silu: b("linear_silu")?,
            safe_gate: b("kda_safe_gate")?,
            lower_bound: f("kda_lower_bound")?,
        };
        if kda.conv_kernel < 2 {
            return Err("config.json: `short_conv_kernel_size` must be at least 2".into());
        }

        // ---- Gated MLA
        let mla = MlaConfig {
            heads: u("num_attention_heads")?,
            q_lora_rank: u("q_lora_rank")?,
            kv_lora_rank: u("kv_lora_rank")?,
            qk_nope_head_dim: u("qk_nope_head_dim")?,
            qk_rope_head_dim: u("qk_rope_head_dim")?,
            v_head_dim: u("v_head_dim")?,
            rope_theta: f("rope_theta")?,
            rope_interleave: b("rope_interleave")?,
        };
        if u("qk_head_dim")? != mla.qk_head_dim() {
            return Err("config.json: `qk_head_dim` is not qk_nope_head_dim + qk_rope_head_dim".into());
        }
        if u("num_key_value_heads")? != mla.heads {
            return Err("config.json: `num_key_value_heads` must equal `num_attention_heads` (MLA caches one latent)".into());
        }
        if mla.rope_interleave != true {
            return Err("config.json: `rope_interleave` must be true (adjacent-pair rotation)".into());
        }
        if mla.qk_rope_head_dim == 0 || mla.kv_lora_rank == 0 {
            return Err("config.json: `qk_rope_head_dim` and `kv_lora_rank` must be positive".into());
        }

        // ---- FFN / routing
        let ffn = FfnConfig {
            dense_intermediate: u("intermediate_size")?,
            moe_intermediate: u("moe_intermediate_size")?,
            routed_experts: u("num_experts")?,
            experts_per_token: u("num_experts_per_tok")?,
            shared_experts: u("num_shared_experts")?,
            n_group: u("n_group")?,
            topk_group: u("topk_group")?,
            routed_scaling_factor: f("routed_scaling_factor")?,
        };
        if ffn.experts_per_token > ffn.routed_experts {
            return Err("config.json: `num_experts_per_tok` exceeds `num_experts`".into());
        }
        if ffn.n_group == 0 || ffn.topk_group == 0 || ffn.n_group > ffn.routed_experts {
            return Err("config.json: `n_group` / `topk_group` must be positive and nested in the expert count".into());
        }
        if ffn.routed_experts % ffn.n_group != 0 {
            return Err("config.json: `num_experts` must be divisible by `n_group`".into());
        }
        if s("score_function")? != "sigmoid" || s("scoring_func")? != "sigmoid" {
            return Err("config.json: this engine implements the sigmoid router only".into());
        }
        if s("topk_method")? != "noaux_tc" {
            return Err("config.json: this engine implements `noaux_tc` top-k only".into());
        }
        if !b("norm_topk_prob")? {
            return Err("config.json: `norm_topk_prob` must be true".into());
        }

        let rms_eps = f("rms_norm_eps")?;
        if !rms_eps.is_finite() || rms_eps <= 0.0 {
            return Err("config.json: `rms_norm_eps` must be positive".into());
        }

        let expert_swiglu_limit_list = f32_list("expert_swiglu_limit_list")?;
        let share_expert_swiglu_limit_list = f32_list("share_expert_swiglu_limit_list")?;
        if expert_swiglu_limit_list.len() != n || share_expert_swiglu_limit_list.len() != n {
            return Err(format!(
                "config.json: `expert_swiglu_limit_list` / `share_expert_swiglu_limit_list` must have {n} entries (found {} / {})",
                expert_swiglu_limit_list.len(),
                share_expert_swiglu_limit_list.len()
            ));
        }

        // ---- MTP
        let mtp = match u("num_nextn_predict_layers")? {
            0 => vec![],
            1 => vec![LayerSpec {
                mixer: Mixer::MlaDsa { indexer: crate::family::IndexerMode::Full },
                ffn: Ffn::Moe,
                residual: Residual::Plain,
            }],
            k => return Err(format!("config.json: `num_nextn_predict_layers` is {k}; one MTP layer is implemented")),
        };
        if mtp.len() == 1 && b("mtp_use_kda")? {
            return Err("config.json: `mtp_use_kda` is true; this engine implements the Gated MLA MTP layer".into());
        }

        let eos = match t.get("eos_token_id") {
            Some(Value::Number(x)) => x.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
            _ => vec![],
        };
        if let Some(bad) = eos.iter().find(|&&id| id as usize >= vocab) {
            return Err(format!("config.json: eos token {bad} is outside the vocabulary of {vocab}"));
        }
        let pad_token_id = t.get("pad_token_id").and_then(|x| x.as_u64()).map(|x| x as u32);
        if let Some(pad) = pad_token_id {
            if pad as usize >= vocab {
                return Err(format!("config.json: pad token {pad} is outside the vocabulary of {vocab}"));
            }
        }

        Ok(Self {
            hidden,
            vocab,
            rms_eps,
            max_positions: u("max_position_embeddings")?,
            layers,
            mtp,
            mla,
            kda,
            ffn,
            expert_swiglu_limit_list,
            share_expert_swiglu_limit_list,
            eos,
            pad_token_id,
        })
    }

    pub fn from_dir(dir: &std::path::Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let root: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_json(&root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL: &str = include_str!("../../../../../testdata/ling31/config.json");

    #[test]
    fn real_config_parses() {
        let v: Value = serde_json::from_str(REAL).unwrap();
        let c = BailingConfig::from_json(&v).unwrap();
        assert_eq!(c.hidden, 4096);
        assert_eq!(c.vocab, 157184);
        assert_eq!(c.layers.len(), 72);
        assert_eq!(c.mtp.len(), 1);
        assert_eq!(c.mtp_layer_index(0), 72);
        assert_eq!(c.kda_layers().len(), 63);
        assert_eq!(c.mla_layers().len(), 9);
        assert_eq!(c.moe_layers().len(), 68);
        // layer 0..3 dense; 4..71 MoE
        assert!(c.layers[0].ffn == Ffn::Dense && c.layers[3].ffn == Ffn::Dense);
        assert!(c.layers[4].ffn == Ffn::Moe && c.layers[71].ffn == Ffn::Moe);
        // KDA layers
        assert_eq!(c.layers[0].mixer, Mixer::Kda);
        assert_eq!(c.layers[7].mixer, Mixer::MlaDsa { indexer: crate::family::IndexerMode::Full });
        assert_eq!(c.layers[71].mixer, Mixer::MlaDsa { indexer: crate::family::IndexerMode::Full });
        assert_eq!(c.layers[70].mixer, Mixer::Kda);
        // swiglu limits
        assert_eq!(c.routed_swiglu_limit(64), 0.0);
        assert_eq!(c.routed_swiglu_limit(65), 10.0);
        assert_eq!(c.routed_swiglu_limit(71), 10.0);
        assert_eq!(c.shared_swiglu_limit(63), 0.0);
        assert_eq!(c.shared_swiglu_limit(64), 10.0);
        assert_eq!(c.shared_swiglu_limit(71), 10.0);
        // MLA dims
        assert_eq!(c.mla.kv_a_group_n(), 640);
        assert_eq!(c.mla.qk_head_dim(), 192);
        assert_eq!(c.eos, vec![156895]);
        assert_eq!(c.pad_token_id, Some(156892));
        // KDA state bytes: 2*64*128*128*4 = 8,388,608
        assert_eq!(c.kda_state_bytes_one_buffer(), 8_388_608);
        assert_eq!(c.kda_conv_history_bytes_per_layer(), 8192 * 4 * 2);
    }

    #[test]
    fn mtp_off_parses() {
        let mut v: Value = serde_json::from_str(REAL).unwrap();
        v["num_nextn_predict_layers"] = serde_json::json!(0);
        let c = BailingConfig::from_json(&v).unwrap();
        assert!(c.mtp.is_empty());
        assert!(!c.has_mtp());
    }
}
