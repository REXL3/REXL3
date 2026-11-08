//! MiniCPM-V-4.7-35B-A3B (`MiniCPMV4_7ForConditionalGeneration`, model type `minicpmv4_7`) and its DSpark draft
//! (`Qwen3DSparkModel`): configuration parsing and the shape checks that refuse a pack the kernels were not built for,
//! before any weight is read.
//!
//! The text backbone, as far as the engine is concerned (it is `Qwen3_5MoeTextModel`, the Qwen3.6-35B-A3B backbone with a
//! different vocabulary): 40 decoder layers, 30 gated delta net (16 key heads, 32 value heads, 128 wide) and 10 grouped-query
//! attention (16 query heads, 2 KV heads, head dim 256, a per-head output gate fused into `q_proj`, rotary on 64 of the 256
//! dims with interleaved M-RoPE sections 11/11/10), a 256-expert top-8 MoE with one shared expert behind a sigmoid gate on
//! every layer, a plain residual, and an untied LM head. Around it: a 27-layer SigLIP-style tower with a 2x2 window merger
//! inserted after layer 6, and a projector to the text width.
//!
//! Three facts of the real `config.json` the code below encodes on purpose (read 2026-10-12, revision `4c7623a9`):
//! * `mtp_num_hidden_layers` is 1 but the checkpoint holds **no `mtp.*` tensor** (Qwen3.6 ships them, this one does not): the
//!   family runs with zero MTP layers ([`MiniCpmConfig::mtp_layers`]) and the inventory refuses an `mtp.*` tensor by name;
//! * `norm_topk_prob` is absent: the Hugging Face implementation always normalises the top-k routing weights, so the default
//!   here is `true` ([`MiniCpmConfig::norm_topk_prob_declared`] says whether the file stated it);
//! * `vocab_size` is 248,144, not a multiple of 128: the converter zero-pads the LM head to `ceil(vocab / 128) x 128` = 248,192
//!   output rows, so every tensor of the head in a pack is that wide ([`MiniCpmConfig::lm_head_width`]); the embedding and the
//!   sampler's vocabulary stay 248,144.

use crate::family::{Ffn, LayerSpec, Mixer, Residual, VisionSpec};
use serde_json::Value;
use std::path::Path;

/// The converter pads every EXL3 linear's input and output width up to a multiple of this (`Linear.pad_to`).
pub const LM_HEAD_PAD: usize = 128;
/// Side of one slice the image processor works at, in pixels (`scale_resolution`, `docs/MINICPMV4_7_INTEGRATION_PLAN.md`).
pub const SLICE_SCALE: usize = 448;
/// Draft positions the engine's per-position acceptance statistics keep (`gb_spec::stats::MAX_POSITIONS`): a block deeper than this
/// would be counted past them, so the draft's block must fit it (15 drafts leave one slot spare).
pub const MAX_DRAFT_POSITIONS: usize = gb_spec::stats::MAX_POSITIONS;

/// Width of the LM head a pack stores for `vocab` real ids.
pub fn padded_vocab(vocab: usize) -> usize {
    vocab.div_ceil(LM_HEAD_PAD) * LM_HEAD_PAD
}

// ---------------------------------------------------------------------------------------------------------- JSON reading

struct Section<'a> {
    v: &'a Value,
    /// How the section is spelt in a message (`text_config`, `vision_config`, `config.json`).
    name: &'static str,
}

impl Section<'_> {
    fn at(&self, k: &str) -> String {
        format!("config.json: {}.{k}", self.name)
    }
    fn get(&self, k: &str) -> Option<&Value> {
        self.v.get(k).filter(|x| !x.is_null())
    }
    fn usize(&self, k: &str) -> Result<usize, String> {
        self.get(k)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| format!("{} is missing or not an integer", self.at(k)))
    }
    fn usize_or(&self, k: &str, default: usize) -> Result<usize, String> {
        match self.get(k) {
            None => Ok(default),
            Some(_) => self.usize(k),
        }
    }
    fn f64(&self, k: &str) -> Result<f64, String> {
        self.get(k).and_then(|v| v.as_f64()).ok_or_else(|| format!("{} is missing or not a number", self.at(k)))
    }
    fn str(&self, k: &str) -> Result<String, String> {
        self.get(k)
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .ok_or_else(|| format!("{} is missing or not a string", self.at(k)))
    }
    fn str_or(&self, k: &str, default: &str) -> Result<String, String> {
        match self.get(k) {
            None => Ok(default.to_string()),
            Some(_) => self.str(k),
        }
    }
    fn bool(&self, k: &str) -> Result<bool, String> {
        self.get(k).and_then(|v| v.as_bool()).ok_or_else(|| format!("{} is missing or not a boolean", self.at(k)))
    }
    fn bool_or(&self, k: &str, default: bool) -> Result<bool, String> {
        match self.get(k) {
            None => Ok(default),
            Some(_) => self.bool(k),
        }
    }
    fn usizes(&self, k: &str) -> Result<Vec<usize>, String> {
        self.get(k)
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{} is missing or not a list", self.at(k)))?
            .iter()
            .map(|x| x.as_u64().map(|v| v as usize).ok_or_else(|| format!("{} holds a non-integer", self.at(k))))
            .collect()
    }
    fn strs(&self, k: &str) -> Result<Vec<String>, String> {
        self.get(k)
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("{} is missing or not a list", self.at(k)))?
            .iter()
            .map(|x| x.as_str().map(str::to_string).ok_or_else(|| format!("{} holds a non-string", self.at(k))))
            .collect()
    }
    fn id(&self, k: &str) -> Result<u32, String> {
        self.usize(k).and_then(|v| u32::try_from(v).map_err(|_| format!("{} does not fit a token id", self.at(k))))
    }
}

fn read_json(dir: &Path) -> Result<Value, String> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("config.json: {e}"))
}

// -------------------------------------------------------------------------------------------------------------- vision

/// `downsample_mode`: how many patches one language-model token covers. `16x` runs the intra-tower window merger (after
/// `insert_layer_id`) and the projector's 2x2 concatenation: 4 x 4 patches a token, 64 tokens for a 448 x 448 slice. `4x` skips
/// the merger: 2 x 2 patches a token, 256 tokens a slice. The mode is a processor and generate argument, not a tensor property,
/// so both modes are valid for one pack and it belongs in every image cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownsampleMode {
    X16,
    X4,
}

impl DownsampleMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "16x" => Ok(Self::X16),
            "4x" => Ok(Self::X4),
            other => Err(format!("downsample_mode \"{other}\": only \"16x\" and \"4x\" exist")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::X16 => "16x",
            Self::X4 => "4x",
        }
    }
    /// Patches per side folded into one language-model token.
    pub fn merge_side(self) -> usize {
        match self {
            Self::X16 => 4,
            Self::X4 => 2,
        }
    }
    /// Whether the intra-tower window merger runs.
    pub fn has_vit_merger(self) -> bool {
        self == Self::X16
    }
}

/// The vision component of the config: the tower, where the window merger sits and the special token ids the prompt layout
/// uses.
#[derive(Clone, Debug, PartialEq)]
pub struct MiniCpmVision {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub patch_size: usize,
    /// Side of the position table's grid in pixels (`image_size`; the table holds `(image_size / patch_size)^2` rows).
    pub image_size: usize,
    pub layer_norm_eps: f64,
    pub hidden_act: String,
    /// The window merger runs after this encoder layer (16x mode).
    pub insert_layer_id: usize,
    pub downsample_mode: DownsampleMode,
    pub mrope_mode: String,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub image_start_id: u32,
    pub image_end_id: u32,
    pub slice_start_id: u32,
    pub slice_end_id: u32,
    pub newline_id: u32,
}

impl MiniCpmVision {
    /// Rows of the learned position table.
    pub fn position_rows(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }
    /// Language-model tokens one [`SLICE_SCALE`] x [`SLICE_SCALE`] slice takes (64 in 16x mode, 256 in 4x).
    pub fn tokens_per_slice(&self) -> usize {
        let side = SLICE_SCALE / self.patch_size / self.downsample_mode.merge_side();
        side * side
    }
}

// ---------------------------------------------------------------------------------------------------------------- text

#[derive(Clone, Debug, PartialEq)]
pub struct MiniCpmConfig {
    pub hidden: usize,
    pub layers: usize,
    pub layer_types: Vec<String>,
    pub full_attention_interval: usize,
    /// Real ids; the LM head is [`MiniCpmConfig::lm_head_width`] wide.
    pub vocab: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    // full attention
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub attn_output_gate: bool,
    pub partial_rotary_factor: f64,
    pub rotary_dim: usize,
    pub rope_theta: f64,
    pub rope_type: String,
    pub mrope_section: Vec<usize>,
    pub mrope_interleaved: bool,
    // gated delta net
    pub lin_key_heads: usize,
    pub lin_value_heads: usize,
    pub lin_key_dim: usize,
    pub lin_value_dim: usize,
    pub lin_conv_kernel: usize,
    pub mamba_ssm_dtype: String,
    // mixture of experts
    pub experts: usize,
    pub experts_per_tok: usize,
    pub moe_intermediate: usize,
    pub shared_intermediate: usize,
    /// The effective setting: `true` when the file does not say (the reference implementation always renormalises).
    pub norm_topk_prob: bool,
    /// Whether `norm_topk_prob` was in the file.
    pub norm_topk_prob_declared: bool,
    /// What `mtp_num_hidden_layers` says. The checkpoint has no MTP tensor, so the family runs [`MiniCpmConfig::mtp_layers`] = 0.
    pub mtp_declared: usize,
    pub vision: MiniCpmVision,
    pub quant_method: String,
}

impl MiniCpmConfig {
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        Self::from_json(&read_json(dir)?)
    }

    /// Every shape is read from the pack; a missing key is an error naming it. The only defaults are the Hugging Face ones for
    /// switches that do not change a shape the file would otherwise state (`attn_output_gate`, `norm_topk_prob`,
    /// `mrope_interleaved`, `hidden_act`, `attention_bias`, `mamba_ssm_dtype`).
    pub fn from_json(root: &Value) -> Result<Self, String> {
        let top = Section { v: root, name: "config.json" };
        let tv = root
            .get("text_config")
            .filter(|v| v.is_object())
            .ok_or("config.json: text_config is missing (the MiniCPM-V config nests the language model under `text_config`)")?;
        let t = Section { v: tv, name: "text_config" };
        let rv =
            t.get("rope_parameters").filter(|v| v.is_object()).ok_or("config.json: text_config.rope_parameters is missing")?;
        let r = Section { v: rv, name: "text_config.rope_parameters" };

        // partial_rotary_factor sits in both places in the real file; when both are there they must agree
        let partial = match (
            r.get("partial_rotary_factor").and_then(Value::as_f64),
            t.get("partial_rotary_factor").and_then(Value::as_f64),
        ) {
            (Some(a), Some(b)) if a != b => {
                return Err(format!(
                    "config.json: partial_rotary_factor is {a} in text_config.rope_parameters and {b} in text_config"
                ))
            }
            (Some(a), _) | (None, Some(a)) => a,
            (None, None) => return Err("config.json: text_config.partial_rotary_factor is missing".into()),
        };
        let head_dim = t.usize("head_dim")?;
        let layer_types = t.strs("layer_types")?;
        let vision = vision_of(root, &top)?;
        let quant_method = root
            .get("quantization_config")
            .and_then(|q| q.get("quant_method"))
            .and_then(|v| v.as_str())
            .unwrap_or("none")
            .to_string();
        let tie_top = top.bool_or("tie_word_embeddings", false)?;
        let tie_text = t.bool_or("tie_word_embeddings", false)?;
        Ok(Self {
            hidden: t.usize("hidden_size")?,
            layers: t.usize("num_hidden_layers")?,
            layer_types,
            full_attention_interval: t.usize_or("full_attention_interval", 4)?,
            vocab: t.usize("vocab_size")?,
            rms_norm_eps: t.f64("rms_norm_eps")?,
            max_position_embeddings: t.usize("max_position_embeddings")?,
            hidden_act: t.str_or("hidden_act", "silu")?,
            attention_bias: t.bool_or("attention_bias", false)?,
            tie_word_embeddings: tie_top || tie_text,
            heads: t.usize("num_attention_heads")?,
            kv_heads: t.usize("num_key_value_heads")?,
            head_dim,
            attn_output_gate: t.bool_or("attn_output_gate", true)?,
            partial_rotary_factor: partial,
            rotary_dim: (head_dim as f64 * partial).round() as usize,
            rope_theta: r.f64("rope_theta")?,
            rope_type: r.str_or("rope_type", "default")?,
            mrope_section: r.usizes("mrope_section")?,
            mrope_interleaved: r.bool_or("mrope_interleaved", false)?,
            lin_key_heads: t.usize("linear_num_key_heads")?,
            lin_value_heads: t.usize("linear_num_value_heads")?,
            lin_key_dim: t.usize("linear_key_head_dim")?,
            lin_value_dim: t.usize("linear_value_head_dim")?,
            lin_conv_kernel: t.usize("linear_conv_kernel_dim")?,
            mamba_ssm_dtype: t.str_or("mamba_ssm_dtype", "float32")?,
            experts: t.usize("num_experts")?,
            experts_per_tok: t.usize("num_experts_per_tok")?,
            moe_intermediate: t.usize("moe_intermediate_size")?,
            shared_intermediate: t.usize("shared_expert_intermediate_size")?,
            norm_topk_prob: t.bool_or("norm_topk_prob", true)?,
            norm_topk_prob_declared: t.get("norm_topk_prob").is_some(),
            mtp_declared: t.usize_or("mtp_num_hidden_layers", 0)?,
            vision,
            quant_method,
        })
    }

    pub fn full_layers(&self) -> usize {
        self.layer_types.iter().filter(|t| t.as_str() == "full_attention").count()
    }

    pub fn linear_layers(&self) -> usize {
        self.layer_types.iter().filter(|t| t.as_str() == "linear_attention").count()
    }

    /// Indices of the full-attention layers.
    pub fn full_layer_ids(&self) -> Vec<usize> {
        (0..self.layer_types.len()).filter(|&i| self.layer_types[i] == "full_attention").collect()
    }

    pub fn is_qwen3_5_moe_shape(&self) -> bool {
        self.hidden == 3072
    }

    pub fn gdn_shape(&self) -> gb_cuda::gdn::GdnShape {
        if self.lin_value_heads == 64 {
            gb_cuda::gdn::GdnShape::Hv64
        } else {
            gb_cuda::gdn::GdnShape::Hv32
        }
    }

    pub fn attn_shape(&self) -> gb_cuda::attention::AttnShape {
        if self.heads == 32 {
            gb_cuda::attention::AttnShape::Q32Kv2
        } else {
            gb_cuda::attention::AttnShape::Q16Kv2
        }
    }

    pub fn kc(&self) -> usize {
        if self.hidden == 3072 {
            gb_cuda::fmoe::QWEN35_122B_KC
        } else {
            gb_cuda::fmoe::MINICPM_KC
        }
    }

    /// Channels of a gated delta net layer's convolution (q, k and v): 8,192.
    pub fn conv_channels(&self) -> usize {
        2 * self.lin_key_heads * self.lin_key_dim + self.lin_value_heads * self.lin_value_dim
    }

    /// Width of the LM head a pack stores: `vocab` rounded up to [`LM_HEAD_PAD`] (248,192). The embedding is `vocab` rows.
    pub fn lm_head_width(&self) -> usize {
        padded_vocab(self.vocab)
    }

    /// MTP layers the family runs. The config declares [`MiniCpmConfig::mtp_declared`] but the checkpoint carries none.
    pub fn mtp_layers(&self) -> usize {
        0
    }

    /// The trunk's layers, in order: every layer is MoE (routed + shared expert) on a plain residual.
    pub fn layer_specs(&self) -> Vec<LayerSpec> {
        self.layer_types
            .iter()
            .map(|t| LayerSpec {
                mixer: if t == "full_attention" { Mixer::Gqa } else { Mixer::Gdn },
                ffn: Ffn::Moe,
                residual: Residual::Plain,
            })
            .collect()
    }

    /// The vision component the engine core reads: the tower's name, the patch geometry and the language-model token id.
    pub fn vision_spec(&self) -> VisionSpec {
        VisionSpec {
            tower: "vision_minicpmv",
            patch_size: self.vision.patch_size,
            merge_size: self.vision.downsample_mode.merge_side(),
            temporal_patch_size: 1,
            out_hidden: self.hidden,
            image_token_id: self.vision.image_token_id,
            max_image_tokens: 4096,
        }
    }

    /// The geometry the kernels (and, later, the tower) implement. Anything else is refused by name before a weight is read: a
    /// pack with another shape would load and then compute a different function.
    pub fn check_shapes(&self) -> Result<(), String> {
        self.check_shapes_for(false)
    }

    /// [`MiniCpmConfig::check_shapes`], optionally accepting a TRUNCATED layer count: the converter's smoke packs carry the first N
    /// of the 40 layers (their `layer_types` cut to N, every other shape unchanged), which is no model but loads and boots like
    /// one (the boot test of a pack with a few layers, `boot::BootExtras::allow_truncated_layers`). The serving probe never sets
    /// it: a registry probe of an 8-layer pack is refused by name.
    pub fn check_shapes_for(&self, allow_truncated_layers: bool) -> Result<(), String> {
        let mut bad: Vec<String> = Vec::new();
        let mut want = |name: &str, got: usize, exp: usize| {
            if got != exp {
                bad.push(format!("{name} {got} (the kernels implement {exp})"));
            }
        };
        if self.hidden == 3072 {
            if allow_truncated_layers {
                if self.layers == 0 || self.layers > 48 {
                    want("num_hidden_layers", self.layers, 48);
                }
            } else {
                want("num_hidden_layers", self.layers, 48);
            }
            want("num_attention_heads", self.heads, 32);
            want("num_key_value_heads", self.kv_heads, 2);
            want("head_dim", self.head_dim, 256);
            want("rotary dims (head_dim x partial_rotary_factor)", self.rotary_dim, 64);
            want("full_attention_interval", self.full_attention_interval, 4);
            want("linear_num_key_heads", self.lin_key_heads, 16);
            want("linear_num_value_heads", self.lin_value_heads, 64);
            want("linear_key_head_dim", self.lin_key_dim, 128);
            want("linear_value_head_dim", self.lin_value_dim, 128);
            want("linear_conv_kernel_dim", self.lin_conv_kernel, 4);
            want("num_experts", self.experts, 256);
            want("num_experts_per_tok", self.experts_per_tok, 8);
            want("moe_intermediate_size", self.moe_intermediate, 1024);
            want("shared_expert_intermediate_size", self.shared_intermediate, 1024);
            want("vocab_size", self.vocab, 248_320);
            want("max_position_embeddings", self.max_position_embeddings, 262_144);
        } else {
            want("hidden_size", self.hidden, 2048);
            if allow_truncated_layers {
                if self.layers == 0 || self.layers > 40 {
                    want("num_hidden_layers", self.layers, 40);
                }
            } else {
                want("num_hidden_layers", self.layers, 40);
            }
            want("num_attention_heads", self.heads, 16);
            want("num_key_value_heads", self.kv_heads, 2);
            want("head_dim", self.head_dim, 256);
            want("rotary dims (head_dim x partial_rotary_factor)", self.rotary_dim, 64);
            want("full_attention_interval", self.full_attention_interval, 4);
            want("linear_num_key_heads", self.lin_key_heads, 16);
            want("linear_num_value_heads", self.lin_value_heads, 32);
            want("linear_key_head_dim", self.lin_key_dim, 128);
            want("linear_value_head_dim", self.lin_value_dim, 128);
            want("linear_conv_kernel_dim", self.lin_conv_kernel, 4);
            want("num_experts", self.experts, 256);
            want("num_experts_per_tok", self.experts_per_tok, 8);
            want("moe_intermediate_size", self.moe_intermediate, 512);
            want("shared_expert_intermediate_size", self.shared_intermediate, 512);
            want("vocab_size", self.vocab, 248_144);
            want("max_position_embeddings", self.max_position_embeddings, 262_144);
            want("vision_config.hidden_size", self.vision.hidden, 1152);
            want("vision_config.num_hidden_layers", self.vision.layers, 27);
            want("vision_config.num_attention_heads", self.vision.heads, 16);
            want("vision_config.intermediate_size", self.vision.intermediate, 4304);
            want("vision_config.patch_size", self.vision.patch_size, 14);
            want("vision_config.image_size (position table side)", self.vision.image_size, 980);
            want("insert_layer_id", self.vision.insert_layer_id, 6);
        }
        if self.layer_types.len() != self.layers {
            bad.push(format!("layer_types has {} entries for {} layers", self.layer_types.len(), self.layers));
        }
        for (i, t) in self.layer_types.iter().enumerate() {
            match t.as_str() {
                "full_attention" | "linear_attention" => {
                    let full = self.full_attention_interval > 0 && (i + 1) % self.full_attention_interval == 0;
                    if full != (t == "full_attention") {
                        bad.push(format!(
                            "layer_types[{i}] is `{t}`, which disagrees with full_attention_interval {}",
                            self.full_attention_interval
                        ));
                        break;
                    }
                }
                other => {
                    bad.push(format!("layer_types[{i}] is `{other}` (only full_attention and linear_attention are built)"));
                    break;
                }
            }
        }
        if self.mrope_section != [11, 11, 10] || !self.mrope_interleaved {
            bad.push(format!(
                "mrope_section {:?} interleaved {} (the kernels implement [11, 11, 10] interleaved)",
                self.mrope_section, self.mrope_interleaved
            ));
        }
        if self.rope_type != "default" {
            bad.push(format!("rope_type `{}` (only `default` is built)", self.rope_type));
        }
        if self.rope_theta != 1e7 {
            bad.push(format!("rope_theta {} (the kernels' tables are built for 1e7)", self.rope_theta));
        }
        if !self.attn_output_gate {
            bad.push("attn_output_gate false (q_proj is one [query | gate] group; only the gated form is built)".into());
        }
        if self.attention_bias {
            bad.push("attention_bias true (no projection bias is built)".into());
        }
        if self.hidden_act != "silu" {
            bad.push(format!("hidden_act `{}` (only silu is built)", self.hidden_act));
        }
        if self.mamba_ssm_dtype != "float32" {
            bad.push(format!("mamba_ssm_dtype `{}` (the recurrent state is fp32)", self.mamba_ssm_dtype));
        }
        if !self.norm_topk_prob {
            bad.push("norm_topk_prob false (the router renormalises its top-k weights; only that form is built)".into());
        }
        if self.tie_word_embeddings {
            bad.push("tie_word_embeddings true (the pack keeps lm_head as its own EXL3 group)".into());
        }
        if self.vision.mrope_mode != "canvas" {
            bad.push(format!("mrope_mode `{}` (only `canvas` is built)", self.vision.mrope_mode));
        }
        if self.vision.hidden_act != "gelu_pytorch_tanh" {
            bad.push(format!("vision_config.hidden_act `{}` (the tower is gelu_tanh)", self.vision.hidden_act));
        }
        if self.vision.insert_layer_id >= self.vision.layers {
            bad.push(format!(
                "insert_layer_id {} is not below vision_config.num_hidden_layers {}",
                self.vision.insert_layer_id, self.vision.layers
            ));
        }
        if self.quant_method != "exl3" {
            bad.push(format!("quantization_config.quant_method {}: not an EXL3 pack", self.quant_method));
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(format!("MiniCPM-V-4.7 pack not supported: {}", bad.join("; ")))
        }
    }
}

fn vision_of(root: &Value, top: &Section<'_>) -> Result<MiniCpmVision, String> {
    let is_qwen_moe = root
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|s| s.as_str())
        .is_some_and(|s| s.starts_with("Qwen3_5Moe"))
        || root.get("model_type").and_then(|s| s.as_str()).is_some_and(|s| s.starts_with("qwen3_5_moe"));
    if is_qwen_moe {
        let vv = root.get("vision_config").filter(|v| v.is_object());
        let (hidden, layers, heads, intermediate, patch_size, hidden_act) = if let Some(vv) = vv {
            let v = Section { v: vv, name: "vision_config" };
            (
                v.usize_or("hidden_size", 1152)?,
                v.usize_or("depth", 27)?,
                v.usize_or("num_heads", 16)?,
                v.usize_or("intermediate_size", 4304)?,
                v.usize_or("patch_size", 16)?,
                v.str_or("hidden_act", "gelu_pytorch_tanh")?,
            )
        } else {
            (1152, 27, 16, 4304, 16, "gelu_pytorch_tanh".to_string())
        };
        return Ok(MiniCpmVision {
            hidden,
            layers,
            heads,
            intermediate,
            patch_size,
            image_size: 980,
            layer_norm_eps: 1e-6,
            hidden_act,
            insert_layer_id: 6,
            downsample_mode: DownsampleMode::X4,
            mrope_mode: "canvas".to_string(),
            image_token_id: top.usize_or("image_token_id", 248056)? as u32,
            video_token_id: top.usize_or("video_token_id", 248057)? as u32,
            image_start_id: top.usize_or("vision_start_token_id", 248053)? as u32,
            image_end_id: top.usize_or("vision_end_token_id", 248054)? as u32,
            slice_start_id: 0,
            slice_end_id: 0,
            newline_id: 198,
        });
    }
    let vv = root.get("vision_config").filter(|v| v.is_object()).ok_or("config.json: vision_config is missing")?;
    let v = Section { v: vv, name: "vision_config" };
    let mode = DownsampleMode::parse(&top.str("downsample_mode")?).map_err(|e| format!("config.json: {e}"))?;
    Ok(MiniCpmVision {
        hidden: v.usize("hidden_size")?,
        layers: v.usize("num_hidden_layers")?,
        heads: v.usize("num_attention_heads")?,
        intermediate: v.usize("intermediate_size")?,
        patch_size: v.usize("patch_size")?,
        image_size: v.usize("image_size")?,
        layer_norm_eps: v.f64("layer_norm_eps")?,
        hidden_act: v.str("hidden_act")?,
        insert_layer_id: top.usize("insert_layer_id")?,
        downsample_mode: mode,
        mrope_mode: top.str("mrope_mode")?,
        image_token_id: top.id("image_token_id")?,
        video_token_id: top.id("video_token_id")?,
        image_start_id: top.id("image_start_id")?,
        image_end_id: top.id("image_end_id")?,
        slice_start_id: top.id("slice_start_id")?,
        slice_end_id: top.id("slice_end_id")?,
        newline_id: top.id("newline_id")?,
    })
}

// ----------------------------------------------------------------------------------------------------------- the draft

/// The DSpark draft's `config.json` (`architectures[0]` `Qwen3DSparkModel`): a 5-layer Qwen3 body that reads the target's
/// hidden states at five layers, drafts a whole block in one non-causal pass, and adds a low-rank Markov transition bias and a
/// confidence head. It carries **no embedding and no LM head**: both are the target's (so the target head's padding enters the
/// draft's logits).
///
/// What the numbers mean (vLLM `qwen3_dspark.py` / `dspark/speculator.py`, Apache-2.0, read 2026-10-12; the evidence is in
/// `docs/MINICPMV4_7_INTEGRATION_PLAN.md`, "E0 results"):
/// * a **tap** `target_layer_ids[j] = N` is the residual stream **after decoder layer N** (vLLM adds 1 to its capture index and
///   captures `hidden + residual` after layer index `N`): all five are full-attention layers of the target;
/// * `block_size` 15 is the number of **drafted tokens** and of draft query rows: the anchor (the committed, not yet verified
///   token) plus 14 mask rows, every row a next-token prediction (`sample_from_anchor`), so a round verifies the anchor row plus
///   15 drafts: **16 target rows per stream** ([`DSparkConfig::verify_rows`]);
/// * attention is non-causal inside the block, full over the context (no sliding window), rotary over the whole 256-wide head
///   (`rope_dim` 256, plain 1-D positions, no M-RoPE) at theta 1e7.
#[derive(Clone, Debug, PartialEq)]
pub struct DSparkConfig {
    pub architecture: String,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub draft_vocab: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub layer_types: Vec<String>,
    pub sliding_window: Option<usize>,
    pub use_sliding_window: bool,
    pub rope_theta: f64,
    pub rope_type: String,
    pub rope_dim: usize,
    /// Drafted tokens per round (and the draft's query rows).
    pub block_size: usize,
    /// Target layers whose output the draft reads, in `fc`'s concatenation order.
    pub target_layer_ids: Vec<usize>,
    pub num_target_layers: usize,
    pub mask_token_id: u32,
    pub markov_rank: usize,
    pub markov_head_type: String,
    pub enable_confidence_head: bool,
    pub confidence_head_with_markov: bool,
    pub confidence_head_alpha: f64,
    pub attention_mode: String,
    pub projector_type: String,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl DSparkConfig {
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        Self::from_json(&read_json(dir)?)
    }

    pub fn from_json(root: &Value) -> Result<Self, String> {
        let t = Section { v: root, name: "config.json" };
        let arch = root
            .get("architectures")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(|a| a.as_str())
            .ok_or("config.json: architectures[0] is missing")?
            .to_string();
        let dv = root.get("dflash_config").filter(|v| v.is_object()).ok_or("config.json: dflash_config is missing")?;
        let d = Section { v: dv, name: "dflash_config" };
        let rv = root.get("rope_parameters").filter(|v| v.is_object()).ok_or("config.json: rope_parameters is missing")?;
        let r = Section { v: rv, name: "rope_parameters" };
        let head_dim = t.usize("head_dim")?;
        // markov_rank and the confidence switches sit at the top level (what vLLM reads) and again in dflash_config: they must agree
        let agree_usize = |k: &str| -> Result<usize, String> {
            let a = t.usize(k)?;
            match d.get(k).and_then(Value::as_u64) {
                Some(b) if b as usize != a => Err(format!("config.json: {k} is {a} at the top level and {b} in dflash_config")),
                _ => Ok(a),
            }
        };
        let agree_bool = |k: &str| -> Result<bool, String> {
            let a = t.bool_or(k, false)?;
            match d.get(k).and_then(Value::as_bool) {
                Some(b) if b != a => Err(format!("config.json: {k} is {a} at the top level and {b} in dflash_config")),
                _ => Ok(a),
            }
        };
        let vocab = t.usize("vocab_size")?;
        let sliding_window = t
            .get("sliding_window")
            .map(|v| v.as_u64().map(|w| w as usize).ok_or("config.json: sliding_window is not an integer or null"))
            .transpose()?;
        Ok(Self {
            architecture: arch,
            hidden: t.usize("hidden_size")?,
            layers: t.usize("num_hidden_layers")?,
            heads: t.usize("num_attention_heads")?,
            kv_heads: t.usize("num_key_value_heads")?,
            head_dim,
            intermediate: t.usize("intermediate_size")?,
            vocab,
            draft_vocab: t.usize_or("draft_vocab_size", vocab)?,
            rms_norm_eps: t.f64("rms_norm_eps")?,
            max_position_embeddings: t.usize("max_position_embeddings")?,
            layer_types: t.strs("layer_types")?,
            sliding_window,
            use_sliding_window: t.bool_or("use_sliding_window", false)?,
            rope_theta: r.f64("rope_theta")?,
            rope_type: r.str_or("rope_type", "default")?,
            rope_dim: r.usize_or("rope_dim", head_dim)?,
            block_size: t.usize("block_size")?,
            target_layer_ids: d.usizes("target_layer_ids")?,
            num_target_layers: t.usize("num_target_layers")?,
            mask_token_id: d.id("mask_token_id")?,
            markov_rank: agree_usize("markov_rank")?,
            markov_head_type: d.str_or("markov_head_type", "vanilla")?,
            enable_confidence_head: agree_bool("enable_confidence_head")?,
            confidence_head_with_markov: agree_bool("confidence_head_with_markov")?,
            confidence_head_alpha: d.get("confidence_head_alpha").and_then(Value::as_f64).unwrap_or(1.0),
            attention_mode: d.str("attention_mode")?,
            projector_type: d.str("projector_type")?,
            bos_token_id: t.id("bos_token_id")?,
            eos_token_id: t.id("eos_token_id")?,
            pad_token_id: t.id("pad_token_id")?,
        })
    }

    /// Tokens one round drafts: the block size.
    pub fn drafts_per_round(&self) -> usize {
        self.block_size
    }

    /// Rows of the draft's query block: the anchor and `block_size - 1` mask rows.
    pub fn block_rows(&self) -> usize {
        self.block_size
    }

    /// Rows one stream puts in the target's verify window: the anchor row and every draft (`16`).
    pub fn verify_rows(&self) -> usize {
        1 + self.block_size
    }

    /// Width of `fc`'s input: the concatenated taps.
    pub fn fc_in(&self) -> usize {
        self.target_layer_ids.len() * self.hidden
    }

    /// Width of the confidence head's input: the block hidden state, plus the Markov embedding when the head uses it.
    pub fn confidence_in(&self) -> usize {
        self.hidden + if self.confidence_head_with_markov { self.markov_rank } else { 0 }
    }

    /// Context K and V the draft keeps per token over its layers (bf16): 10,240 B.
    pub fn kv_bytes_per_token(&self) -> u64 {
        (self.layers * 2 * self.kv_heads * self.head_dim * 2) as u64
    }

    /// The geometry the draft kernels will implement; anything else is refused by name.
    pub fn check_shapes(&self) -> Result<(), String> {
        let mut bad: Vec<String> = Vec::new();
        let mut want = |name: &str, got: usize, exp: usize| {
            if got != exp {
                bad.push(format!("{name} {got} (the kernels implement {exp})"));
            }
        };
        want("hidden_size", self.hidden, 2048);
        want("num_hidden_layers", self.layers, 5);
        want("num_attention_heads", self.heads, 8);
        want("num_key_value_heads", self.kv_heads, 2);
        want("head_dim", self.head_dim, 256);
        want("intermediate_size", self.intermediate, 6144);
        want("rope_dim (the whole head is rotated)", self.rope_dim, self.head_dim);
        want("markov_rank", self.markov_rank, 256);
        want("block_size", self.block_size, 15);
        if self.architecture != "Qwen3DSparkModel" {
            bad.push(format!("architecture `{}` (only Qwen3DSparkModel is built)", self.architecture));
        }
        if self.layer_types.len() != self.layers || self.layer_types.iter().any(|t| t != "full_attention") {
            bad.push(
                "layer_types: every draft layer must be full_attention (the block is non-causal over the whole context)".into(),
            );
        }
        if self.sliding_window.is_some() || self.use_sliding_window {
            bad.push(format!(
                "sliding_window {:?} / use_sliding_window {} (the draft attends to the whole context)",
                self.sliding_window, self.use_sliding_window
            ));
        }
        if self.rope_type != "default" {
            bad.push(format!("rope_parameters.rope_type `{}` (only `default` is built)", self.rope_type));
        }
        if self.attention_mode != "gqa" {
            bad.push(format!("dflash_config.attention_mode `{}` (only gqa is built)", self.attention_mode));
        }
        if self.projector_type != "dspark" {
            bad.push(format!("dflash_config.projector_type `{}` (only dspark is built)", self.projector_type));
        }
        if self.markov_head_type != "vanilla" {
            bad.push(format!("dflash_config.markov_head_type `{}` (only vanilla is built)", self.markov_head_type));
        }
        if self.target_layer_ids.is_empty() {
            bad.push("dflash_config.target_layer_ids is empty".into());
        }
        if self.block_size > MAX_DRAFT_POSITIONS {
            bad.push(format!(
                "block_size {} drafts {} positions a round; the engine's per-position statistics keep {MAX_DRAFT_POSITIONS}",
                self.block_size, self.block_size
            ));
        }
        if self.draft_vocab != self.vocab {
            bad.push(format!(
                "draft_vocab_size {} differs from vocab_size {} (a reduced draft vocabulary needs a draft-to-target id map this family does not build)",
                self.draft_vocab, self.vocab
            ));
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(format!("DSpark draft not supported: {}", bad.join("; ")))
        }
    }

    /// Whether this draft belongs to `target`: the same width and vocabulary, taps that exist (and are full-attention layers: a
    /// tap on a recurrent layer is read the same way, but every shipped draft taps attention layers, so anything else is
    /// refused rather than guessed), and a mask token inside the vocabulary.
    pub fn check_pairing(&self, target: &MiniCpmConfig) -> Result<(), String> {
        let mut bad: Vec<String> = Vec::new();
        if self.hidden != target.hidden {
            bad.push(format!("hidden_size {} against the target's {}", self.hidden, target.hidden));
        }
        if self.vocab != target.vocab {
            bad.push(format!("vocab_size {} against the target's {}", self.vocab, target.vocab));
        }
        if self.num_target_layers != target.layers {
            bad.push(format!("num_target_layers {} against the target's {} layers", self.num_target_layers, target.layers));
        }
        let full = target.full_layer_ids();
        for &id in &self.target_layer_ids {
            if id >= target.layers {
                bad.push(format!("tap layer {id} is not a layer of the target ({} layers)", target.layers));
            } else if !full.contains(&id) {
                bad.push(format!("tap layer {id} is not a full-attention layer of the target"));
            }
        }
        if self.mask_token_id as usize >= target.vocab {
            bad.push(format!("mask_token_id {} is outside the target's {} ids", self.mask_token_id, target.vocab));
        }
        if bad.is_empty() {
            Ok(())
        } else {
            Err(format!("the DSpark draft does not belong to this target: {}", bad.join("; ")))
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The pack's `config.json`: the Hugging Face file of `openbmb/MiniCPM-V-4.7-35B-A3B` at `4c7623a9`, plus the
    /// `quantization_config` block the converter writes.
    pub const MINICPM_CONFIG: &str = include_str!("../../../../../testdata/minicpmv4_7/config.json");
    /// `openbmb/MiniCPM-V-4.7-35B-A3B-DSpark` at `5facb75c`, verbatim.
    pub const DSPARK_CONFIG: &str = include_str!("../../../../../testdata/minicpmv4_7/draft_config.json");

    pub fn real() -> MiniCpmConfig {
        MiniCpmConfig::from_json(&serde_json::from_str(MINICPM_CONFIG).unwrap()).unwrap()
    }

    pub fn draft() -> DSparkConfig {
        DSparkConfig::from_json(&serde_json::from_str(DSPARK_CONFIG).unwrap()).unwrap()
    }

    fn edited(f: impl FnOnce(&mut Value)) -> Value {
        let mut v: Value = serde_json::from_str(MINICPM_CONFIG).unwrap();
        f(&mut v);
        v
    }

    #[test]
    fn the_real_config_parses_to_the_shapes_the_kernels_implement() {
        let c = real();
        assert_eq!((c.hidden, c.layers, c.vocab, c.lm_head_width()), (2048, 40, 248_144, 248_192));
        assert_eq!((c.full_layers(), c.linear_layers()), (10, 30));
        assert_eq!(c.full_layer_ids(), vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
        assert_eq!((c.heads, c.kv_heads, c.head_dim, c.rotary_dim), (16, 2, 256, 64));
        assert!(c.attn_output_gate && !c.attention_bias && !c.tie_word_embeddings);
        assert_eq!((c.rope_theta, c.rope_type.as_str(), c.partial_rotary_factor), (1e7, "default", 0.25));
        assert_eq!((c.mrope_section.clone(), c.mrope_interleaved), (vec![11, 11, 10], true));
        assert_eq!(
            (c.lin_key_heads, c.lin_value_heads, c.lin_key_dim, c.lin_value_dim, c.lin_conv_kernel),
            (16, 32, 128, 128, 4)
        );
        assert_eq!(c.conv_channels(), 8192);
        assert_eq!((c.experts, c.experts_per_tok, c.moe_intermediate, c.shared_intermediate), (256, 8, 512, 512));
        assert_eq!(c.full_attention_interval, 4);
        assert_eq!((c.max_position_embeddings, c.hidden_act.as_str(), c.mamba_ssm_dtype.as_str()), (262_144, "silu", "float32"));
        assert_eq!(c.quant_method, "exl3");
        c.check_shapes().unwrap();
    }

    #[test]
    fn the_three_deliberate_readings_of_the_real_file() {
        let c = real();
        // the file says 1 MTP layer, the checkpoint has no MTP tensor: the family runs none
        assert_eq!((c.mtp_declared, c.mtp_layers()), (1, 0));
        // norm_topk_prob is absent from the file: the effective setting is the reference implementation's (renormalise)
        assert!(!c.norm_topk_prob_declared && c.norm_topk_prob);
        // a stated `false` is honoured, and the kernels refuse it by name
        let v = edited(|v| v["text_config"]["norm_topk_prob"] = serde_json::json!(false));
        let c = MiniCpmConfig::from_json(&v).unwrap();
        assert!(c.norm_topk_prob_declared && !c.norm_topk_prob);
        assert!(c.check_shapes().unwrap_err().contains("norm_topk_prob false"));
        // the head is padded, the embedding and the vocabulary are not
        assert_eq!(padded_vocab(248_144), 248_192);
        assert_eq!(padded_vocab(248_320), 248_320);
        assert_eq!(padded_vocab(128), 128);
        assert_eq!(padded_vocab(129), 256);
    }

    #[test]
    fn the_layer_recipe_is_thirty_gdn_and_ten_attention_layers_all_moe_on_a_plain_residual() {
        let specs = real().layer_specs();
        assert_eq!(specs.len(), 40);
        for (i, s) in specs.iter().enumerate() {
            assert_eq!(s.mixer, if i % 4 == 3 { Mixer::Gqa } else { Mixer::Gdn }, "layer {i}");
            assert_eq!((s.ffn, s.residual), (Ffn::Moe, Residual::Plain), "layer {i}");
        }
    }

    #[test]
    fn the_vision_component_reads_the_tower_the_merger_and_the_special_ids() {
        let c = real();
        let v = &c.vision;
        assert_eq!((v.hidden, v.layers, v.heads, v.intermediate, v.patch_size), (1152, 27, 16, 4304, 14));
        assert_eq!((v.image_size, v.position_rows(), v.insert_layer_id), (980, 4900, 6));
        assert_eq!((v.downsample_mode, v.mrope_mode.as_str()), (DownsampleMode::X16, "canvas"));
        assert_eq!((v.image_token_id, v.video_token_id), (248_056, 248_057));
        assert_eq!(
            (v.image_start_id, v.image_end_id, v.slice_start_id, v.slice_end_id, v.newline_id),
            (248_078, 248_079, 248_088, 248_089, 198)
        );
        // 448 / 14 = 32 patches a side; 16x folds 4 x 4 of them into a token, 4x folds 2 x 2
        assert_eq!(v.tokens_per_slice(), 64);
        let c4 = MiniCpmConfig::from_json(&edited(|v| v["downsample_mode"] = serde_json::json!("4x"))).unwrap();
        assert_eq!(c4.vision.tokens_per_slice(), 256);
        assert!(!c4.vision.downsample_mode.has_vit_merger() && v.downsample_mode.has_vit_merger());
        let s = c.vision_spec();
        assert_eq!(
            (s.tower, s.patch_size, s.merge_size, s.temporal_patch_size, s.out_hidden, s.image_token_id),
            ("vision_minicpmv", 14, 4, 1, 2048, 248_056)
        );
    }

    #[test]
    fn a_shape_the_kernels_do_not_implement_is_refused_by_name() {
        let text = |k: &'static str, val: serde_json::Value| move |v: &mut Value| v["text_config"][k] = val.clone();
        let vision = |k: &'static str, val: serde_json::Value| move |v: &mut Value| v["vision_config"][k] = val.clone();
        let top = |k: &'static str, val: serde_json::Value| move |v: &mut Value| v[k] = val.clone();
        type Edit = Box<dyn Fn(&mut Value)>;
        let cases: Vec<(Edit, &str)> = vec![
            (Box::new(text("num_key_value_heads", 4.into())), "num_key_value_heads 4"),
            (Box::new(text("num_experts", 128.into())), "num_experts 128"),
            (Box::new(text("num_experts_per_tok", 4.into())), "num_experts_per_tok 4"),
            (Box::new(text("moe_intermediate_size", 640.into())), "moe_intermediate_size 640"),
            (Box::new(text("linear_num_value_heads", 48.into())), "linear_num_value_heads 48"),
            (Box::new(text("head_dim", 128.into())), "head_dim 128"),
            (Box::new(text("vocab_size", 248_320.into())), "vocab_size 248320"),
            (Box::new(text("attn_output_gate", false.into())), "attn_output_gate false"),
            (Box::new(text("hidden_act", "gelu".into())), "hidden_act `gelu`"),
            (Box::new(text("mamba_ssm_dtype", "bfloat16".into())), "mamba_ssm_dtype `bfloat16`"),
            (Box::new(text("tie_word_embeddings", true.into())), "tie_word_embeddings true"),
            (Box::new(text("full_attention_interval", 3.into())), "full_attention_interval 3"),
            (Box::new(vision("num_hidden_layers", 24.into())), "vision_config.num_hidden_layers 24"),
            (Box::new(vision("image_size", 448.into())), "vision_config.image_size"),
            (Box::new(top("insert_layer_id", 9.into())), "insert_layer_id 9"),
            (Box::new(top("mrope_mode", "flat".into())), "mrope_mode `flat`"),
            (Box::new(top("quantization_config", serde_json::json!({"quant_method": "awq"}))), "quant_method awq"),
        ];
        for (edit, needle) in cases {
            let v = edited(|v| edit(v));
            let e = MiniCpmConfig::from_json(&v).and_then(|c| c.check_shapes()).unwrap_err();
            assert!(e.contains(needle), "refusal {e:?} does not name {needle:?}");
        }
    }

    #[test]
    fn a_layer_types_list_that_disagrees_with_the_layer_count_or_the_interval_is_refused() {
        let short = edited(|v| {
            v["text_config"]["layer_types"].as_array_mut().unwrap().pop();
        });
        let e = MiniCpmConfig::from_json(&short).unwrap().check_shapes().unwrap_err();
        assert!(e.contains("layer_types has 39 entries for 40 layers"), "{e}");
        // an attention layer moved off the every-fourth grid
        let moved = edited(|v| {
            let lt = v["text_config"]["layer_types"].as_array_mut().unwrap();
            lt.swap(3, 2);
        });
        let e = MiniCpmConfig::from_json(&moved).unwrap().check_shapes().unwrap_err();
        assert!(e.contains("layer_types[2]") && e.contains("full_attention_interval 4"), "{e}");
        // a layer type nothing builds
        let other = edited(|v| v["text_config"]["layer_types"][5] = serde_json::json!("sliding_attention"));
        let e = MiniCpmConfig::from_json(&other).unwrap().check_shapes().unwrap_err();
        assert!(e.contains("layer_types[5] is `sliding_attention`"), "{e}");
    }

    #[test]
    fn rope_and_mrope_settings_are_checked() {
        let sections = edited(|v| v["text_config"]["rope_parameters"]["mrope_section"] = serde_json::json!([16, 8, 8]));
        assert!(MiniCpmConfig::from_json(&sections).unwrap().check_shapes().unwrap_err().contains("mrope_section [16, 8, 8]"));
        let flat = edited(|v| v["text_config"]["rope_parameters"]["mrope_interleaved"] = serde_json::json!(false));
        assert!(MiniCpmConfig::from_json(&flat).unwrap().check_shapes().unwrap_err().contains("interleaved false"));
        let theta = edited(|v| v["text_config"]["rope_parameters"]["rope_theta"] = serde_json::json!(1e6));
        assert!(MiniCpmConfig::from_json(&theta).unwrap().check_shapes().unwrap_err().contains("rope_theta"));
        let rotary = edited(|v| {
            v["text_config"]["partial_rotary_factor"] = serde_json::json!(0.5);
            v["text_config"]["rope_parameters"]["partial_rotary_factor"] = serde_json::json!(0.5);
        });
        assert!(MiniCpmConfig::from_json(&rotary).unwrap().check_shapes().unwrap_err().contains("rotary dims"));
        // the two places the factor is written must agree
        let split = edited(|v| v["text_config"]["partial_rotary_factor"] = serde_json::json!(0.5));
        let e = MiniCpmConfig::from_json(&split).unwrap_err();
        assert!(e.contains("partial_rotary_factor"), "{e}");
    }

    #[test]
    fn an_unknown_downsample_mode_a_missing_text_config_and_missing_keys_are_refused_by_name() {
        let e = MiniCpmConfig::from_json(&edited(|v| v["downsample_mode"] = serde_json::json!("8x"))).unwrap_err();
        assert!(e.contains("downsample_mode \"8x\"") && e.contains("\"16x\" and \"4x\""), "{e}");
        let e = MiniCpmConfig::from_json(&edited(|v| {
            v.as_object_mut().unwrap().remove("text_config");
        }))
        .unwrap_err();
        assert!(e.contains("text_config is missing"), "{e}");
        let e = MiniCpmConfig::from_json(&edited(|v| {
            v.as_object_mut().unwrap().remove("vision_config");
        }))
        .unwrap_err();
        assert!(e.contains("vision_config is missing"), "{e}");
        for (section, key) in [
            ("text_config", "linear_num_key_heads"),
            ("text_config", "shared_expert_intermediate_size"),
            ("text_config", "layer_types"),
        ] {
            let e = MiniCpmConfig::from_json(&edited(|v| {
                v[section].as_object_mut().unwrap().remove(key);
            }))
            .unwrap_err();
            assert!(e.contains(key), "{key}: {e}");
        }
        let e = MiniCpmConfig::from_json(&edited(|v| {
            v.as_object_mut().unwrap().remove("insert_layer_id");
        }))
        .unwrap_err();
        assert!(e.contains("insert_layer_id"), "{e}");
    }

    // ---- the draft

    #[test]
    fn the_real_draft_config_parses_and_pairs_with_the_target() {
        let d = draft();
        assert_eq!(
            (d.architecture.as_str(), d.hidden, d.layers, d.heads, d.kv_heads, d.head_dim, d.intermediate),
            ("Qwen3DSparkModel", 2048, 5, 8, 2, 256, 6144)
        );
        assert_eq!((d.vocab, d.draft_vocab, d.max_position_embeddings), (248_144, 248_144, 262_144));
        assert_eq!((d.block_size, d.num_target_layers, d.target_layer_ids.clone()), (15, 40, vec![3, 11, 19, 27, 35]));
        assert_eq!((d.mask_token_id, d.markov_rank, d.markov_head_type.as_str()), (248_076, 256, "vanilla"));
        assert!(d.enable_confidence_head && d.confidence_head_with_markov);
        assert_eq!((d.attention_mode.as_str(), d.projector_type.as_str()), ("gqa", "dspark"));
        // every draft layer is full attention, there is no sliding window, the whole 256-wide head is rotated at theta 1e7
        assert!(d.layer_types.iter().all(|t| t == "full_attention") && d.layer_types.len() == 5);
        assert_eq!((d.sliding_window, d.use_sliding_window), (None, false));
        assert_eq!((d.rope_theta, d.rope_type.as_str(), d.rope_dim), (1e7, "default", 256));
        assert_eq!((d.bos_token_id, d.eos_token_id, d.pad_token_id), (248_045, 248_046, 248_044));
        d.check_shapes().unwrap();
        d.check_pairing(&real()).unwrap();
    }

    #[test]
    fn the_depth_the_draft_implies_is_fifteen_drafts_and_a_sixteen_row_verify_window() {
        let d = draft();
        // 15 draft tokens a round, 15 query rows in the draft's block (the anchor and 14 masks), the anchor's row plus 15 drafts
        // in the target's window
        assert_eq!((d.drafts_per_round(), d.block_rows(), d.verify_rows()), (15, 15, 16));
        // a 64-row verify window holds four such streams, and the per-position statistics hold the 15 draft positions with one spare
        assert_eq!(64 / d.verify_rows(), 4);
        assert!(d.drafts_per_round() < MAX_DRAFT_POSITIONS && MAX_DRAFT_POSITIONS == 16);
        // fc reads five taps of 2048, the confidence head reads the block hidden state and the Markov embedding
        assert_eq!((d.fc_in(), d.confidence_in()), (10_240, 2_304));
        // 5 layers x 2 (K, V) x 2 kv heads x 256 x 2 B
        assert_eq!(d.kv_bytes_per_token(), 10_240);
    }

    #[test]
    fn a_draft_the_kernels_do_not_implement_or_that_does_not_fit_the_target_is_refused_by_name() {
        let edited_draft = |f: &dyn Fn(&mut Value)| {
            let mut v: Value = serde_json::from_str(DSPARK_CONFIG).unwrap();
            f(&mut v);
            DSparkConfig::from_json(&v)
        };
        let e = edited_draft(&|v| v["block_size"] = serde_json::json!(16)).unwrap().check_shapes().unwrap_err();
        assert!(e.contains("block_size 16 (the kernels implement 15)"), "{e}");
        let e = edited_draft(&|v| v["block_size"] = serde_json::json!(17)).unwrap().check_shapes().unwrap_err();
        assert!(e.contains("block_size 17 drafts 17 positions") && e.contains("keep 16"), "{e}");
        let e = edited_draft(&|v| v["sliding_window"] = serde_json::json!(1024)).unwrap().check_shapes().unwrap_err();
        assert!(e.contains("sliding_window Some(1024)"), "{e}");
        let e =
            edited_draft(&|v| v["layer_types"][2] = serde_json::json!("sliding_attention")).unwrap().check_shapes().unwrap_err();
        assert!(e.contains("every draft layer must be full_attention"), "{e}");
        let e = edited_draft(&|v| v["rope_parameters"]["rope_dim"] = serde_json::json!(64)).unwrap().check_shapes().unwrap_err();
        assert!(e.contains("rope_dim"), "{e}");
        let e = edited_draft(&|v| v["draft_vocab_size"] = serde_json::json!(32_000)).unwrap().check_shapes().unwrap_err();
        assert!(e.contains("draft_vocab_size 32000"), "{e}");
        let e =
            edited_draft(&|v| v["architectures"] = serde_json::json!(["Qwen3DFlashModel"])).unwrap().check_shapes().unwrap_err();
        assert!(e.contains("Qwen3DFlashModel"), "{e}");
        let e = edited_draft(&|v| v["markov_rank"] = serde_json::json!(128)).unwrap_err();
        assert!(e.contains("markov_rank is 128 at the top level and 256 in dflash_config"), "{e}");
        let e = edited_draft(&|v| {
            v.as_object_mut().unwrap().remove("dflash_config");
        })
        .unwrap_err();
        assert!(e.contains("dflash_config is missing"), "{e}");
        // pairing: a tap off the target, a tap on a recurrent layer, a mask id outside the vocabulary, another width
        let t = real();
        let e = edited_draft(&|v| v["dflash_config"]["target_layer_ids"] = serde_json::json!([3, 11, 19, 27, 40]))
            .unwrap()
            .check_pairing(&t)
            .unwrap_err();
        assert!(e.contains("tap layer 40 is not a layer of the target"), "{e}");
        let e = edited_draft(&|v| v["dflash_config"]["target_layer_ids"] = serde_json::json!([2, 11, 19, 27, 35]))
            .unwrap()
            .check_pairing(&t)
            .unwrap_err();
        assert!(e.contains("tap layer 2 is not a full-attention layer"), "{e}");
        let e = edited_draft(&|v| v["dflash_config"]["mask_token_id"] = serde_json::json!(248_144))
            .unwrap()
            .check_pairing(&t)
            .unwrap_err();
        assert!(e.contains("mask_token_id 248144"), "{e}");
        let e = edited_draft(&|v| v["hidden_size"] = serde_json::json!(4096)).unwrap().check_pairing(&t).unwrap_err();
        assert!(e.contains("hidden_size 4096 against the target's 2048"), "{e}");
    }

    #[test]
    fn the_draft_may_carry_the_converters_quantization_block() {
        let mut v: Value = serde_json::from_str(DSPARK_CONFIG).unwrap();
        v["quantization_config"] = serde_json::json!({"quant_method": "exl3", "bits": 6.0, "codebook": "mul1"});
        let d = DSparkConfig::from_json(&v).unwrap();
        d.check_shapes().unwrap();
    }
}
