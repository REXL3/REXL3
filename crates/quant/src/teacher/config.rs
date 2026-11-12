//! The Qwen3.5 family (dense and MoE; MiniCPM-V 4.7's text backbone): configuration and the quantized targets of a
//! layer, in the converter's order and names.

use serde_json::Value;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Linear,
    Full,
}

pub const ARCHITECTURES: &[&str] = &[
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "MiniCPMV4_7ForConditionalGeneration",
];

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: String,
    pub hidden: usize,
    pub layers: usize,
    pub vocab: usize,
    pub eps: f64,
    pub kinds: Vec<Kind>,
    pub head_dim: usize,
    pub n_q: usize,
    pub n_kv: usize,
    pub rope_theta: f64,
    pub partial_rotary: f64,
    pub lk_heads: usize,
    pub lv_heads: usize,
    pub lk_dim: usize,
    pub lv_dim: usize,
    pub conv_k: usize,
    pub experts: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
    pub inter: usize,
    /// tensor name prefix of the text model (`model.language_model` or `model`)
    pub prefix: String,
}

fn get<'a>(t: &'a Value, k: &str) -> Option<&'a Value> {
    t.get(k).filter(|v| !v.is_null())
}

fn uint(t: &Value, k: &str) -> Result<usize, String> {
    get(t, k).and_then(Value::as_u64).map(|v| v as usize).ok_or_else(|| format!("config: no {k}"))
}

fn uint_or(t: &Value, k: &str, d: usize) -> usize {
    get(t, k).and_then(Value::as_u64).map(|v| v as usize).unwrap_or(d)
}

impl Config {
    pub fn load(dir: &Path) -> Result<Config, String> {
        let s = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.display()))?;
        let v: Value = serde_json::from_str(&s).map_err(|e| format!("config.json: {e}"))?;
        Config::from_json(&v, None)
    }

    /// From config.json; `prefix` overrides the text-model tensor prefix (else `model.language_model` for the
    /// multimodal architectures, `model` otherwise; the caller checks it against the checkpoint).
    pub fn from_json(cfg: &Value, prefix: Option<&str>) -> Result<Config, String> {
        let arch = cfg.get("architectures").and_then(|a| a.get(0)).and_then(Value::as_str).unwrap_or("?").to_string();
        if !ARCHITECTURES.contains(&arch.as_str()) {
            return Err(format!("architecture {arch} is not in the Qwen3.5 family"));
        }
        let t = cfg.get("text_config").unwrap_or(cfg);
        let layers = uint(t, "num_hidden_layers")?;
        let hidden = uint(t, "hidden_size")?;
        let n_q = uint(t, "num_attention_heads")?;
        let kinds = match get(t, "layer_types").and_then(Value::as_array) {
            Some(a) => a
                .iter()
                .map(|x| match x.as_str() {
                    Some("full_attention") => Ok(Kind::Full),
                    Some("linear_attention") => Ok(Kind::Linear),
                    o => Err(format!("config: layer type {o:?}")),
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => {
                let every = uint_or(t, "full_attention_interval", 4);
                (0..layers).map(|i| if (i + 1) % every == 0 { Kind::Full } else { Kind::Linear }).collect()
            }
        };
        if kinds.len() != layers {
            return Err(format!("config: {} layer types for {layers} layers", kinds.len()));
        }
        let rope = get(t, "rope_parameters").cloned().unwrap_or(Value::Null);
        let f = |k: &str, d: f64| rope.get(k).and_then(Value::as_f64).or_else(|| get(t, k).and_then(Value::as_f64)).unwrap_or(d);
        let multimodal = cfg.get("text_config").is_some();
        Ok(Config {
            arch,
            hidden,
            layers,
            vocab: uint(t, "vocab_size")?,
            eps: get(t, "rms_norm_eps").and_then(Value::as_f64).ok_or("config: no rms_norm_eps")?,
            kinds,
            head_dim: uint_or(t, "head_dim", hidden / n_q),
            n_q,
            n_kv: uint(t, "num_key_value_heads")?,
            rope_theta: f("rope_theta", 10000.0),
            partial_rotary: f("partial_rotary_factor", 1.0),
            lk_heads: uint(t, "linear_num_key_heads")?,
            lv_heads: uint(t, "linear_num_value_heads")?,
            lk_dim: uint(t, "linear_key_head_dim")?,
            lv_dim: uint(t, "linear_value_head_dim")?,
            conv_k: uint(t, "linear_conv_kernel_dim")?,
            experts: uint_or(t, "num_experts", 0),
            top_k: uint_or(t, "num_experts_per_tok", 0),
            moe_inter: uint_or(t, "moe_intermediate_size", 0),
            shared_inter: uint_or(t, "shared_expert_intermediate_size", 0),
            inter: uint_or(t, "intermediate_size", 0),
            prefix: prefix
                .map(str::to_string)
                .unwrap_or_else(|| if multimodal { "model.language_model" } else { "model" }.into()),
        })
    }

    pub fn moe(&self) -> bool {
        self.experts > 0
    }
    pub fn key_dim(&self) -> usize {
        self.lk_heads * self.lk_dim
    }
    pub fn value_dim(&self) -> usize {
        self.lv_heads * self.lv_dim
    }
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    pub fn rot_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary) as usize
    }
    pub fn lp(&self, i: usize) -> String {
        format!("{}.layers.{i}", self.prefix)
    }

    /// RoPE inverse frequencies 1 / theta^(2i / rd), in fp64, rounded to fp32.
    pub fn inv_freq(&self) -> Vec<f32> {
        let rd = self.rot_dim();
        (0..rd / 2).map(|i| (1.0 / self.rope_theta.powf((2 * i) as f64 / rd as f64)) as f32).collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    /// checkpoint name without `.weight` (routed experts split: `<p>.mlp.experts.E.{gate,up,down}_proj`)
    pub key: String,
    pub layer: usize,
    pub role: &'static str,
    pub out_features: usize,
    pub in_features: usize,
    /// the input group (the qmap outside all-experts mode)
    pub group: String,
    pub expert: Option<usize>,
}

/// A layer's quantized linears, in the converter's order.
pub fn targets(c: &Config, i: usize) -> Vec<Target> {
    let p = c.lp(i);
    let h = c.hidden;
    let t = |key: String, role, out, inp, group: String, expert| Target {
        key,
        layer: i,
        role,
        out_features: out,
        in_features: inp,
        group,
        expert,
    };
    let mut out = Vec::new();
    match c.kinds[i] {
        Kind::Linear => {
            let a = format!("{p}.linear_attn");
            out.push(t(format!("{a}.in_proj_qkv"), "gdn.qkv", c.conv_dim(), h, format!("{p}.attn_in"), None));
            out.push(t(format!("{a}.in_proj_z"), "gdn.z", c.value_dim(), h, format!("{p}.attn_in"), None));
            out.push(t(format!("{a}.out_proj"), "gdn.o", h, c.value_dim(), format!("{p}.attn_o"), None));
        }
        Kind::Full => {
            let a = format!("{p}.self_attn");
            let kv = c.n_kv * c.head_dim;
            out.push(t(format!("{a}.q_proj"), "attn.q", c.n_q * c.head_dim * 2, h, format!("{p}.attn_in"), None));
            out.push(t(format!("{a}.k_proj"), "attn.k", kv, h, format!("{p}.attn_in"), None));
            out.push(t(format!("{a}.v_proj"), "attn.v", kv, h, format!("{p}.attn_in"), None));
            out.push(t(format!("{a}.o_proj"), "attn.o", h, c.n_q * c.head_dim, format!("{p}.attn_o"), None));
        }
    }
    let m = format!("{p}.mlp");
    if c.moe() {
        let (im, s) = (c.moe_inter, c.shared_inter);
        out.push(t(format!("{m}.shared_expert.gate_proj"), "shared.gate", s, h, format!("{p}.mlp_in"), None));
        out.push(t(format!("{m}.shared_expert.up_proj"), "shared.up", s, h, format!("{p}.mlp_in"), None));
        out.push(t(format!("{m}.shared_expert.down_proj"), "shared.down", h, s, format!("{p}.shared_mid"), None));
        for e in 0..c.experts {
            let x = format!("{m}.experts.{e}");
            out.push(t(format!("{x}.gate_proj"), "expert.gate", im, h, format!("{x}.in"), Some(e)));
            out.push(t(format!("{x}.up_proj"), "expert.up", im, h, format!("{x}.in"), Some(e)));
            out.push(t(format!("{x}.down_proj"), "expert.down", h, im, format!("{x}.mid"), Some(e)));
        }
    } else {
        out.push(t(format!("{m}.gate_proj"), "mlp.gate", c.inter, h, format!("{p}.mlp_in"), None));
        out.push(t(format!("{m}.up_proj"), "mlp.up", c.inter, h, format!("{p}.mlp_in"), None));
        out.push(t(format!("{m}.down_proj"), "mlp.down", h, c.inter, format!("{p}.mlp_mid"), None));
    }
    out
}

/// The qmap (shared input Hessian) of a target: in all-experts mode every routed expert's gate/up and the shared
/// expert's gate/up share `<p>.mlp.experts.input`.
pub fn qmap_of(c: &Config, t: &Target, all_experts: bool) -> String {
    if all_experts && matches!(t.role, "expert.gate" | "expert.up" | "shared.gate" | "shared.up") {
        return format!("{}.mlp.experts.input", c.lp(t.layer));
    }
    t.group.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minicpm() -> Value {
        serde_json::json!({
            "architectures": ["MiniCPMV4_7ForConditionalGeneration"],
            "text_config": {
                "hidden_size": 2048, "num_hidden_layers": 4, "vocab_size": 248144, "rms_norm_eps": 1e-6,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "head_dim": 256, "num_attention_heads": 16, "num_key_value_heads": 2,
                "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25},
                "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
                "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "num_experts": 256, "num_experts_per_tok": 8,
                "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512
            }
        })
    }

    #[test]
    fn minicpm_layer_targets() {
        let c = Config::from_json(&minicpm(), None).unwrap();
        assert_eq!((c.conv_dim(), c.value_dim(), c.rot_dim()), (8192, 4096, 64));
        let t0 = targets(&c, 0);
        assert_eq!(t0.len(), 3 + 3 + 3 * 256);
        assert_eq!(t0[0].key, "model.language_model.layers.0.linear_attn.in_proj_qkv");
        assert_eq!((t0[0].out_features, t0[0].in_features), (8192, 2048));
        assert_eq!(t0[6].key, "model.language_model.layers.0.mlp.experts.0.gate_proj");
        assert_eq!(qmap_of(&c, &t0[6], true), "model.language_model.layers.0.mlp.experts.input");
        assert_eq!(qmap_of(&c, &t0[8], true), "model.language_model.layers.0.mlp.experts.0.mid");
        assert_eq!(qmap_of(&c, &t0[3], true), "model.language_model.layers.0.mlp.experts.input");
        assert_eq!(qmap_of(&c, &t0[5], true), "model.language_model.layers.0.shared_mid");
        let t3 = targets(&c, 3);
        assert_eq!(t3[0].key, "model.language_model.layers.3.self_attn.q_proj");
        assert_eq!(t3[0].out_features, 16 * 256 * 2);
        let f = c.inv_freq();
        assert_eq!(f.len(), 32);
        assert_eq!(f[0], 1.0);
        assert!(Config::from_json(&serde_json::json!({"architectures": ["LlamaForCausalLM"]}), None).is_err());
    }
}
