//! Nemotron 3.5 Super VL (`NemotronH_Omni_Reasoning_V3`): the text backbone's geometry and layer pattern, read from the published
//! `config.json`. Validation is strict: a config that is not the measured geometry is refused by name, and nothing is defaulted.
//!
//! The text backbone is a Nemotron-H hybrid: `M` Mamba2 blocks, `*` attention blocks without RoPE, and `E` latent-MoE blocks, in
//! the order of `llm_config.layers_block_type`. Its MTP head is one block pattern `*E` (attention then latent MoE) that predicts one
//! further token. The vision tower (RADIO) and projector are separate components and are checked by the family's probe, not here.

use crate::family::{Ffn, LayerSpec, Mixer, Residual};
use serde_json::Value;

/// The top-level `architectures[0]` this family accepts.
pub const ARCH: &str = "NemotronH_Omni_Reasoning_V3";
/// The top-level `model_type` this family accepts.
pub const MODEL_TYPE: &str = "nemotron_h_omni";
/// The text backbone's own `llm_config.model_type`.
pub const TEXT_MODEL_TYPE: &str = "nemotron_h";

/// The measured geometry. A different value is a different model, refused by name.
const GEOMETRY: &[(&str, usize)] = &[
    ("hidden_size", 4096),
    ("num_attention_heads", 32),
    ("num_key_value_heads", 2),
    ("head_dim", 128),
    ("mamba_num_heads", 128),
    ("mamba_head_dim", 64),
    ("n_groups", 8),
    ("ssm_state_size", 128),
    ("conv_kernel", 4),
    ("n_routed_experts", 512),
    ("num_experts_per_tok", 22),
    ("moe_intermediate_size", 2688),
    ("moe_latent_size", 1024),
    ("moe_shared_expert_intermediate_size", 5376),
    ("vocab_size", 131072),
    ("max_position_embeddings", 1048576),
];

#[derive(Clone, Debug, PartialEq)]
pub struct OmniConfig {
    /// Trunk block types, one character per layer: `M` Mamba2, `*` attention, `E` latent MoE.
    pub pattern: String,
    /// MTP block types (`*E`) for `num_nextn_predict_layers` = 1, empty for 0.
    pub mtp_pattern: String,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub mamba_heads: usize,
    pub mamba_dim: usize,
    pub groups: usize,
    pub state: usize,
    pub conv: usize,
    pub experts: usize,
    pub topk: usize,
    pub expert_inter: usize,
    pub latent: usize,
    pub shared_inter: usize,
    pub vocab: usize,
    pub max_context: usize,
    pub routed_scaling: f64,
    pub eps: f64,
    /// The Mamba2 dt clamp `(min, max)`. The engine implements `[0, +inf]`, so the published limit must be exactly that.
    pub dt_limit: (f64, f64),
}

/// The Mamba2 dt clamp. The published config encodes infinity as `{"__float__": "Infinity"}`, which plain JSON numbers cannot hold.
/// HF clamps with a 0.001 floor on its chunked path; the fork and this engine use `[0, +inf]`, so any other limit is refused by name.
fn dt_limit(llm: &Value) -> Result<(f64, f64), String> {
    let pair = llm
        .get("time_step_limit")
        .and_then(Value::as_array)
        .filter(|a| a.len() == 2)
        .ok_or("Nemotron 3.5 Super: `time_step_limit` must be a pair")?;
    let number = |v: &Value| -> Option<f64> {
        if let Some(x) = v.as_f64() {
            return Some(x);
        }
        match v.get("__float__").and_then(Value::as_str)? {
            "Infinity" => Some(f64::INFINITY),
            "-Infinity" => Some(f64::NEG_INFINITY),
            _ => None,
        }
    };
    let lo = number(&pair[0]).ok_or("Nemotron 3.5 Super: `time_step_limit` entries must be numbers")?;
    let hi = number(&pair[1]).ok_or("Nemotron 3.5 Super: `time_step_limit` entries must be numbers")?;
    if lo != 0.0 || hi != f64::INFINITY {
        return Err(format!(
            "Nemotron 3.5 Super: `time_step_limit` is [{lo}, {hi}]; the engine implements the measured [0, +inf] clamp"
        ));
    }
    Ok((lo, hi))
}

/// The measured RADIO tower (`vision_config`) and its projector. The tower is not built yet; this checks that the published geometry is
/// the one the engine will implement, and refuses any other by name. Video pruning (`video_pruning_rate`) is not checked here.
fn check_vision(root: &Value) -> Result<(), String> {
    let v = root.get("vision_config").filter(|v| v.is_object()).ok_or("Nemotron 3.5 Super: missing `vision_config`")?;
    for (key, want) in [
        ("hidden_size", 1280usize),
        ("num_hidden_layers", 32),
        ("num_attention_heads", 16),
        ("patch_size", 16),
        ("num_channels", 3),
        ("num_cls_tokens", 3),
        ("num_registers", 7),
        ("min_num_patches", 1024),
        ("max_num_patches", 13312),
    ] {
        let got =
            v.get(key).and_then(Value::as_u64).ok_or_else(|| format!("Nemotron 3.5 Super: vision `{key}` must be an integer"))?;
        if got as usize != want {
            return Err(format!("Nemotron 3.5 Super: vision `{key}` is {got}, the measured tower has {want}"));
        }
    }
    if v.get("mlp_ratio").and_then(Value::as_f64) != Some(4.0) {
        return Err("Nemotron 3.5 Super: vision `mlp_ratio` must be 4.0 (MLP width 5120)".into());
    }
    if v.get("qkv_bias").and_then(Value::as_bool) != Some(true) || v.get("use_swiglu_ffn").and_then(Value::as_bool) != Some(false)
    {
        return Err("Nemotron 3.5 Super: the tower has qkv bias and a plain (non-SwiGLU) GELU MLP".into());
    }
    if root.get("ps_version").and_then(Value::as_str) != Some("v2") {
        return Err("Nemotron 3.5 Super: `ps_version` must be `v2` (the pixel-shuffle layout the projector implements)".into());
    }
    if root.get("downsample_ratio").and_then(Value::as_f64) != Some(0.5) {
        return Err("Nemotron 3.5 Super: `downsample_ratio` must be 0.5".into());
    }
    if root.get("vit_hidden_size").and_then(Value::as_u64) != Some(1280) {
        return Err("Nemotron 3.5 Super: `vit_hidden_size` must be 1280".into());
    }
    if root.get("projector_hidden_size").and_then(Value::as_u64) != Some(20480) {
        return Err("Nemotron 3.5 Super: `projector_hidden_size` must be 20480".into());
    }
    Ok(())
}

fn positive(v: &Value, key: &str) -> Result<usize, String> {
    let n = v
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| format!("Nemotron 3.5 Super: `{key}` must be a positive integer"))?;
    if n == 0 {
        return Err(format!("Nemotron 3.5 Super: `{key}` is zero"));
    }
    Ok(n)
}

fn block_string(list: &[Value], key: &str) -> Result<String, String> {
    list.iter()
        .map(|t| match t.as_str() {
            Some("mamba") => Ok('M'),
            Some("attention") => Ok('*'),
            Some("moe") => Ok('E'),
            Some(other) => Err(format!("Nemotron 3.5 Super: unknown `{key}` block type `{other}`")),
            None => Err(format!("Nemotron 3.5 Super: `{key}` entries must be strings")),
        })
        .collect()
}

/// The backbone's flags that the measured model fixes. A different value means a different model: refused by name.
fn fixed_flags(llm: &Value) -> Result<(), String> {
    for (key, want) in [
        ("attention_bias", false),
        ("mamba_proj_bias", false),
        ("mlp_bias", false),
        ("use_bias", false),
        ("tie_word_embeddings", false),
        ("norm_topk_prob", true),
        ("moe_shared_expert_overlap", false),
        ("use_conv_bias", true),
    ] {
        if llm.get(key).and_then(Value::as_bool) != Some(want) {
            return Err(format!("Nemotron 3.5 Super: `{key}` must be {want}"));
        }
    }
    for (key, want) in [("n_group", 1usize), ("topk_group", 1), ("n_shared_experts", 1)] {
        if llm.get(key).and_then(Value::as_u64) != Some(want as u64) {
            return Err(format!("Nemotron 3.5 Super: `{key}` must be {want}"));
        }
    }
    for (key, want) in [("mamba_hidden_act", "silu"), ("mlp_hidden_act", "relu2"), ("mamba_ssm_cache_dtype", "float32")] {
        if llm.get(key).and_then(Value::as_str) != Some(want) {
            return Err(format!("Nemotron 3.5 Super: `{key}` must be `{want}`"));
        }
    }
    if !llm.get("sliding_window").is_some_and(Value::is_null) {
        return Err("Nemotron 3.5 Super: sliding-window attention is not part of this model".into());
    }
    Ok(())
}

impl OmniConfig {
    pub fn from_json(root: &Value) -> Result<Self, String> {
        if root.get("model_type").and_then(Value::as_str) != Some(MODEL_TYPE) {
            return Err(format!("Nemotron 3.5 Super: model_type must be `{MODEL_TYPE}`"));
        }
        let arch = root.get("architectures").and_then(Value::as_array).and_then(|a| a.first()).and_then(Value::as_str);
        if arch != Some(ARCH) {
            return Err(format!("Nemotron 3.5 Super: architectures[0] must be `{ARCH}`"));
        }
        let llm = root.get("llm_config").filter(|v| v.is_object()).ok_or("Nemotron 3.5 Super: missing `llm_config`")?;
        if llm.get("model_type").and_then(Value::as_str) != Some(TEXT_MODEL_TYPE) {
            return Err(format!("Nemotron 3.5 Super: llm_config.model_type must be `{TEXT_MODEL_TYPE}`"));
        }
        fixed_flags(llm)?;

        let trunk =
            llm.get("layers_block_type").and_then(Value::as_array).ok_or("Nemotron 3.5 Super: missing `layers_block_type`")?;
        let pattern = block_string(trunk, "layers_block_type")?;
        if pattern.is_empty() {
            return Err("Nemotron 3.5 Super: `layers_block_type` is empty".into());
        }
        if let Some(n) = llm.get("num_hidden_layers").and_then(Value::as_u64) {
            if n as usize != pattern.len() {
                return Err(format!(
                    "Nemotron 3.5 Super: num_hidden_layers {n} does not match layers_block_type ({})",
                    pattern.len()
                ));
            }
        }

        let mtp_count = llm
            .get("num_nextn_predict_layers")
            .and_then(Value::as_u64)
            .ok_or("Nemotron 3.5 Super: missing `num_nextn_predict_layers`")?;
        let mtp_pattern = match mtp_count {
            0 => String::new(),
            1 => {
                let list = llm
                    .get("mtp_layers_block_type")
                    .and_then(Value::as_array)
                    .ok_or("Nemotron 3.5 Super: missing `mtp_layers_block_type` for its MTP head")?;
                let p = block_string(list, "mtp_layers_block_type")?;
                if p != "*E" {
                    return Err(format!("Nemotron 3.5 Super: MTP block pattern must be `*E`, not `{p}`"));
                }
                p
            }
            n => return Err(format!("Nemotron 3.5 Super: {n} MTP steps; this model declares one")),
        };

        for (key, want) in GEOMETRY {
            let got = positive(llm, key)?;
            if got != *want {
                return Err(format!("Nemotron 3.5 Super: `{key}` is {got}, the measured model has {want}"));
            }
        }
        let routed_scaling = llm.get("routed_scaling_factor").and_then(Value::as_f64).filter(|x| x.is_finite() && *x > 0.0);
        let routed_scaling = routed_scaling.ok_or("Nemotron 3.5 Super: `routed_scaling_factor` must be finite and positive")?;
        let eps = llm.get("layer_norm_epsilon").and_then(Value::as_f64).filter(|x| x.is_finite() && *x > 0.0);
        let eps = eps.ok_or("Nemotron 3.5 Super: `layer_norm_epsilon` must be finite and positive")?;
        let dt_limit = dt_limit(llm)?;
        check_vision(root)?;

        let heads = positive(llm, "num_attention_heads")?;
        let kv_heads = positive(llm, "num_key_value_heads")?;
        if heads % kv_heads != 0 {
            return Err("Nemotron 3.5 Super: attention heads are not a multiple of the KV heads".into());
        }
        Ok(Self {
            pattern,
            mtp_pattern,
            hidden: positive(llm, "hidden_size")?,
            heads,
            kv_heads,
            head_dim: positive(llm, "head_dim")?,
            mamba_heads: positive(llm, "mamba_num_heads")?,
            mamba_dim: positive(llm, "mamba_head_dim")?,
            groups: positive(llm, "n_groups")?,
            state: positive(llm, "ssm_state_size")?,
            conv: positive(llm, "conv_kernel")?,
            experts: positive(llm, "n_routed_experts")?,
            topk: positive(llm, "num_experts_per_tok")?,
            expert_inter: positive(llm, "moe_intermediate_size")?,
            latent: positive(llm, "moe_latent_size")?,
            shared_inter: positive(llm, "moe_shared_expert_intermediate_size")?,
            vocab: positive(llm, "vocab_size")?,
            max_context: positive(llm, "max_position_embeddings")?,
            routed_scaling,
            eps,
            dt_limit,
        })
    }

    /// The trunk's layers, in order: `M` is a Mamba2 mixer with no FFN, `*` is attention without RoPE with no FFN, and `E` is a
    /// latent MoE behind a `None` mixer. Every block joins the stream plainly (Nemotron-H has no hyper-connections).
    pub fn layer_specs(&self) -> Vec<LayerSpec> {
        self.pattern
            .chars()
            .map(|c| match c {
                'M' => LayerSpec { mixer: Mixer::Mamba2, ffn: Ffn::None, residual: Residual::Plain },
                '*' => LayerSpec { mixer: Mixer::GqaNope, ffn: Ffn::None, residual: Residual::Plain },
                _ => LayerSpec { mixer: Mixer::None, ffn: Ffn::LatentMoeRelu2, residual: Residual::Plain },
            })
            .collect()
    }

    /// The MTP head's layers, in order, as the trunk describes them (`*E`: attention then latent MoE).
    pub fn mtp_specs(&self) -> Vec<LayerSpec> {
        self.mtp_pattern
            .chars()
            .map(|c| match c {
                '*' => LayerSpec { mixer: Mixer::GqaNope, ffn: Ffn::None, residual: Residual::Plain },
                _ => LayerSpec { mixer: Mixer::None, ffn: Ffn::LatentMoeRelu2, residual: Residual::Plain },
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published pack's `config.json`, committed as a fixture (`testdata/nemotron_omni/config.json`).
    const PACK: &str = include_str!("../../../../../testdata/nemotron_omni/config.json");

    fn root() -> Value {
        serde_json::from_str(PACK).unwrap()
    }

    fn set(v: &mut Value, key: &str, val: Value) {
        v["llm_config"][key] = val;
    }

    #[test]
    fn the_published_config_parses_to_the_measured_geometry() {
        let c = OmniConfig::from_json(&root()).unwrap();
        assert_eq!(c.pattern.len(), 88);
        assert_eq!(c.pattern.matches('E').count() + c.pattern.matches('*').count() + c.pattern.matches('M').count(), 88);
        assert_eq!(c.mtp_pattern, "*E");
        assert_eq!((c.hidden, c.experts, c.topk, c.latent), (4096, 512, 22, 1024));
        assert_eq!(c.max_context, 1048576);
        assert_eq!(c.layer_specs().len(), 88);
        assert_eq!(c.mtp_specs().len(), 2);
    }

    #[test]
    fn the_layer_specs_follow_the_pattern_block_for_block() {
        let c = OmniConfig::from_json(&root()).unwrap();
        for (ch, spec) in c.pattern.chars().zip(c.layer_specs()) {
            match ch {
                'M' => assert_eq!((spec.mixer, spec.ffn), (Mixer::Mamba2, Ffn::None)),
                '*' => assert_eq!((spec.mixer, spec.ffn), (Mixer::GqaNope, Ffn::None)),
                _ => assert_eq!((spec.mixer, spec.ffn), (Mixer::None, Ffn::LatentMoeRelu2)),
            }
        }
    }

    #[test]
    fn a_missing_or_mistyped_key_is_refused_by_name() {
        for key in ["hidden_size", "layers_block_type", "routed_scaling_factor", "mamba_num_heads"] {
            let mut v = root();
            v["llm_config"].as_object_mut().unwrap().remove(key);
            let err = OmniConfig::from_json(&v).unwrap_err();
            assert!(err.contains(key), "{key}: {err}");
        }
    }

    #[test]
    fn the_published_dt_clamp_is_zero_to_infinity() {
        let c = OmniConfig::from_json(&root()).unwrap();
        assert_eq!(c.dt_limit, (0.0, f64::INFINITY));
    }

    #[test]
    fn a_different_dt_clamp_is_refused_by_name() {
        let mut v = root();
        v["llm_config"]["time_step_limit"] = serde_json::json!([0.001, {"__float__": "Infinity"}]);
        let err = OmniConfig::from_json(&v).unwrap_err();
        assert!(err.contains("time_step_limit") && err.contains("[0, +inf]"), "{err}");
    }

    #[test]
    fn the_published_vision_geometry_is_accepted() {
        assert!(check_vision(&root()).is_ok());
    }

    #[test]
    fn a_different_vision_tower_or_pixel_shuffle_is_refused_by_name() {
        let mut v = root();
        v["vision_config"]["hidden_size"] = Value::from(1024);
        assert!(check_vision(&v).unwrap_err().contains("hidden_size"));
        let mut v = root();
        v["ps_version"] = Value::from("v1");
        assert!(check_vision(&v).unwrap_err().contains("ps_version"));
    }

    #[test]
    fn a_different_geometry_is_a_different_model_and_is_refused() {
        for (key, bad) in
            [("hidden_size", 2688), ("n_routed_experts", 256), ("moe_latent_size", 2688), ("max_position_embeddings", 262144)]
        {
            let mut v = root();
            set(&mut v, key, Value::from(bad));
            let err = OmniConfig::from_json(&v).unwrap_err();
            assert!(err.contains(key), "{key}: {err}");
        }
    }

    #[test]
    fn an_unknown_block_type_is_refused() {
        let mut v = root();
        v["llm_config"]["layers_block_type"][3] = Value::from("linear_attention");
        assert!(OmniConfig::from_json(&v).unwrap_err().contains("unknown `layers_block_type` block type"));
    }

    #[test]
    fn a_layer_count_that_disagrees_with_the_pattern_is_refused() {
        let mut v = root();
        set(&mut v, "num_hidden_layers", Value::from(87));
        assert!(OmniConfig::from_json(&v).unwrap_err().contains("num_hidden_layers 87"));
    }

    #[test]
    fn a_mtp_pattern_other_than_star_e_is_refused() {
        let mut v = root();
        v["llm_config"]["mtp_layers_block_type"] = serde_json::json!(["moe", "attention"]);
        assert!(OmniConfig::from_json(&v).unwrap_err().contains("MTP block pattern must be `*E`"));
    }

    #[test]
    fn a_second_mtp_step_is_refused() {
        let mut v = root();
        set(&mut v, "num_nextn_predict_layers", Value::from(2));
        assert!(OmniConfig::from_json(&v).unwrap_err().contains("2 MTP steps"));
    }

    #[test]
    fn a_flag_the_measured_model_fixes_is_refused() {
        for (key, bad) in [
            ("norm_topk_prob", Value::Bool(false)),
            ("sliding_window", Value::from(4096)),
            ("mlp_hidden_act", Value::from("gelu")),
        ] {
            let mut v = root();
            set(&mut v, key, bad);
            assert!(OmniConfig::from_json(&v).is_err(), "{key} should be refused");
        }
    }

    #[test]
    fn a_wrong_architecture_or_model_type_is_refused() {
        let mut v = root();
        v["architectures"] = serde_json::json!(["NemotronHForCausalLM"]);
        assert!(OmniConfig::from_json(&v).unwrap_err().contains("architectures[0]"));
        let mut v = root();
        v["model_type"] = Value::from("nemotron_h");
        assert!(OmniConfig::from_json(&v).unwrap_err().contains("model_type"));
    }
}
