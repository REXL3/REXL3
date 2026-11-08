//! DeepSeek-V4.1-Flash (`DeepseekV41ForCausalLM`, model types `deepseek_v41` / `deepseek_v41_text`): the pack's `config.json`,
//! read into typed fields with strict validation.
//!
//! The text model is 40 trunk layers, every one of them multi-head latent attention with one shared KV head of width
//! `head_dim` (512), followed by a DeepSeek sparse-attention (DSA) indexer on eight layers and a KV compressor on four. The
//! feed-forward block is a mixture of 384 routed experts (top 6, `sqrtsoftplus` scoring, `noaux_tc` selection) plus one shared
//! expert, and every residual join goes through mHC hyper-connections with `hc_mult` streams. Two layers (1 and 14) carry an
//! "engram" token-lookup table. Three NextN layers follow the trunk, each with a DSpark Markov head and a confidence head.
//!
//! Three config conventions differ from GLM-5.3's and are worth naming here, because the rest of the family reads them:
//!
//! * `compress_ratios` has one entry per **trunk layer plus NextN layer** (43 for 40 + 3), not per trunk layer. Layer 0 says
//!   how many tokens one compressed entry spans; 0 means the layer caches no compressed entries.
//! * `kv_source_layer_ids` names the four layers that own a compressor (`attn.compressor.*`) and an indexer key projection
//!   (`attn.indexer.wk`); `index_source_layer_ids` names the eight that own indexer query projections. They are different
//!   sets, and the inventory is built from both.
//! * `engram_num_embeddings` is a *list* parallel to `engram_layer_ids`, not a scalar: the two tables have different sizes.
//!
//! Every field of `text_config` is read; [`KNOWN_TEXT_KEYS`] lists them and a test proves the list covers the pack's config
//! exactly, so a new key in a future pack is a test failure rather than a silently ignored value.

use crate::family::{Ffn, IndexerMode, LayerSpec, Mixer, Residual};
use serde_json::Value;
use std::path::Path;

/// The `architectures[0]` this family accepts.
pub const ARCH: &str = "DeepseekV41ForCausalLM";
/// The `model_type` values this family accepts when a config has no `architectures`.
pub const MODEL_TYPE: &str = "deepseek_v41";
pub const MODEL_TYPE_TEXT: &str = "deepseek_v41_text";

/// Every key this parser reads from `text_config`. A test checks it against the pack's config, so an unread key cannot hide.
pub const KNOWN_TEXT_KEYS: &[&str] = &[
    "model_type",
    "vocab_size",
    "hidden_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "qk_rope_head_dim",
    "q_lora_rank",
    "o_lora_rank",
    "o_groups",
    "hidden_act",
    "swiglu_limit",
    "rms_norm_eps",
    "attention_bias",
    "attention_dropout",
    "initializer_range",
    "use_cache",
    "tie_word_embeddings",
    "max_position_embeddings",
    "rope_theta",
    "rope_scaling",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "scoring_func",
    "topk_method",
    "norm_topk_prob",
    "routed_scaling_factor",
    "sliding_window",
    "compress_ratios",
    "compress_rope_theta",
    "kv_source_layer_ids",
    "index_source_layer_ids",
    "index_n_heads",
    "index_head_dim",
    "index_topk",
    "candidate_source_layer_id",
    "candidate_topk_blocks",
    "candidate_block_size",
    "hc_mult",
    "hc_sinkhorn_iters",
    "hc_eps",
    "engram_layer_ids",
    "engram_num_embeddings",
    "engram_max_ngram_size",
    "engram_vocab_size",
    "engram_n_heads",
    "engram_head_dim",
    "engram_pad_token_id",
    "engram_compressed_vocab_size",
    "num_nextn_predict_layers",
    "dspark_block_size",
    "dspark_noise_token_id",
    "dspark_target_layer_ids",
    "dspark_markov_rank",
    "dspark_n_routed_experts",
    "dspark_num_experts_per_tok",
];

// -------------------------------------------------------------------------------------------------------------------- pieces

/// Multi-head latent attention: 64 heads of 512 over one shared latent of 512, with low-rank query and output projections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlaConfig {
    pub heads: usize,
    pub kv_heads: usize,
    /// Width of one head. The cached latent is `kv_heads * head_dim`.
    pub head_dim: usize,
    /// The rotary part of the query/key, carved out of `head_dim`.
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    /// The output projection runs as this many independent low-rank groups.
    pub o_groups: usize,
}

impl MlaConfig {
    /// Width of one query (all heads).
    pub fn q_width(&self) -> usize {
        self.heads * self.head_dim
    }
    /// Width of the cached latent (one KV head).
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }
    /// `wo_a`'s input width: one `head_dim` slice per output group.
    pub fn o_inner(&self) -> usize {
        self.o_groups * self.head_dim
    }
    /// `wo_a`'s output width.
    pub fn o_rank(&self) -> usize {
        self.o_groups * self.o_lora_rank
    }
}

/// Rotary embedding: the trunk's base and the compressor's, plus the YaRN scaling both share.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeConfig {
    pub theta: f64,
    pub compress_theta: f64,
    pub yarn: YarnConfig,
}

#[derive(Clone, Debug, PartialEq)]
pub struct YarnConfig {
    pub factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub original_max_positions: usize,
}

/// The routed mixture and its shared expert.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeConfig {
    pub intermediate: usize,
    pub routed_experts: usize,
    pub shared_experts: usize,
    pub experts_per_token: usize,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub swiglu_limit: f64,
}

/// The compressed KV a layer with a non-zero `compress_ratio` caches.
#[derive(Clone, Debug, PartialEq)]
pub struct CompressConfig {
    /// One entry per trunk layer **and** per NextN layer (`num_hidden_layers + num_nextn_predict_layers` entries).
    pub ratios: Vec<u32>,
    /// The layers that own `attn.compressor.{wkv,wgate?,norm}`.
    pub source_layer_ids: Vec<usize>,
    pub rope_theta: f64,
}

impl CompressConfig {
    pub fn ratio(&self, layer: usize) -> u32 {
        self.ratios.get(layer).copied().unwrap_or(0)
    }
    pub fn is_source(&self, layer: usize) -> bool {
        self.source_layer_ids.contains(&layer)
    }
    /// A compressor gate (`wgate`) exists only where an entry spans more than one token: the pack's four source layers
    /// carry three gates between them, and the one with ratio 1 carries none.
    pub fn has_gate(&self, layer: usize) -> bool {
        self.is_source(layer) && self.ratio(layer) > 1
    }
}

/// The DeepSeek sparse-attention indexer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexerConfig {
    /// The layers that own `attn.indexer.{wq_b,weights_proj}`.
    pub source_layer_ids: Vec<usize>,
    pub n_heads: usize,
    pub head_dim: usize,
    pub topk: usize,
    /// The layer whose block scores seed the candidate walk.
    pub candidate_source_layer_id: usize,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
}

impl IndexerConfig {
    pub fn is_source(&self, layer: usize) -> bool {
        self.source_layer_ids.contains(&layer)
    }
    /// Only the layers that own `indexer.wk` score their own selection; the rest reuse the nearest preceding one's key.
    pub fn mode(&self, layer: usize, compress: &CompressConfig) -> IndexerMode {
        if self.is_source(layer) && compress.is_source(layer) {
            IndexerMode::Full
        } else {
            IndexerMode::Shared
        }
    }
}

/// mHC hyper-connections: `streams` residual streams mixed by a Sinkhorn-normalised matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct HcConfig {
    pub mult: usize,
    pub sinkhorn_iters: usize,
    pub eps: f64,
}

/// The engram token-lookup tables on layers 1 and 14.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngramConfig {
    pub layer_ids: Vec<usize>,
    /// Parallel to `layer_ids`: one table size per layer.
    pub num_embeddings: Vec<usize>,
    pub max_ngram_size: usize,
    pub vocab_size: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub pad_token_id: u32,
    pub compressed_vocab_size: usize,
}

impl EngramConfig {
    /// The table width of the engram on `layer`, if it has one.
    pub fn table(&self, layer: usize) -> Option<usize> {
        self.layer_ids.iter().position(|&l| l == layer).map(|i| self.num_embeddings[i])
    }
}

/// The NextN layers' DSpark Markov head.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DsparkConfig {
    pub block_size: usize,
    pub noise_token_id: u32,
    pub target_layer_ids: Vec<usize>,
    pub markov_rank: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
}

/// The vision tower (`vision_config`), read but not built.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub model_type: String,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub patch_size: usize,
    pub rope_theta: f64,
    pub downsample_ratio: usize,
    pub max_image_tokens: usize,
    pub min_pixels: usize,
    /// `null` in the pack: no aspect-ratio cap is stated.
    pub max_wh_ratio: Option<f64>,
}

impl VisionConfig {
    /// Values in one patch: 3 channels x patch x patch.
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }
}

/// The text model, its NextN layers and its vision tower.
#[derive(Clone, Debug, PartialEq)]
pub struct DeepseekV41Config {
    pub model_type: String,
    pub vocab: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub hidden_act: String,
    pub swiglu_limit: f64,
    pub rms_eps: f64,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub initializer_range: f64,
    pub use_cache: bool,
    pub tie_word_embeddings: bool,
    pub max_positions: usize,
    pub rope: RopeConfig,
    pub moe: MoeConfig,
    pub sliding_window: usize,
    pub compress: CompressConfig,
    pub indexer: IndexerConfig,
    pub hc: HcConfig,
    pub engram: EngramConfig,
    /// `num_nextn_predict_layers`: the NextN layers after the trunk.
    pub nextn: usize,
    pub dspark: DsparkConfig,
    pub vision: Option<VisionConfig>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: Option<u32>,
    pub image_token_id: u32,
}

// ------------------------------------------------------------------------------------------------------------------ parsing

fn field<'a>(v: &'a Value, path: &str, k: &str) -> Result<&'a Value, String> {
    v.get(k).ok_or_else(|| format!("config.json: missing `{path}{k}`"))
}

fn usize_at(v: &Value, path: &str, k: &str) -> Result<usize, String> {
    field(v, path, k)?
        .as_u64()
        .map(|x| x as usize)
        .ok_or_else(|| format!("config.json: `{path}{k}` must be a non-negative integer"))
}

fn u32_at(v: &Value, path: &str, k: &str) -> Result<u32, String> {
    field(v, path, k)?
        .as_u64()
        .and_then(|x| u32::try_from(x).ok())
        .ok_or_else(|| format!("config.json: `{path}{k}` must be a 32-bit unsigned integer"))
}

fn f64_at(v: &Value, path: &str, k: &str) -> Result<f64, String> {
    field(v, path, k)?.as_f64().ok_or_else(|| format!("config.json: `{path}{k}` must be a number"))
}

fn bool_at(v: &Value, path: &str, k: &str) -> Result<bool, String> {
    field(v, path, k)?.as_bool().ok_or_else(|| format!("config.json: `{path}{k}` must be true or false"))
}

fn str_at<'a>(v: &'a Value, path: &str, k: &str) -> Result<&'a str, String> {
    field(v, path, k)?.as_str().ok_or_else(|| format!("config.json: `{path}{k}` must be a string"))
}

fn positive(v: &Value, path: &str, k: &str) -> Result<usize, String> {
    match usize_at(v, path, k)? {
        0 => Err(format!("config.json: `{path}{k}` must be positive, found 0")),
        n => Ok(n),
    }
}

fn indices(v: &Value, path: &str, k: &str) -> Result<Vec<usize>, String> {
    field(v, path, k)?
        .as_array()
        .ok_or_else(|| format!("config.json: `{path}{k}` must be a list"))?
        .iter()
        .map(|x| x.as_u64().map(|x| x as usize).ok_or_else(|| format!("config.json: `{path}{k}` must be a list of indices")))
        .collect()
}

fn ratios(v: &Value, path: &str, k: &str) -> Result<Vec<u32>, String> {
    field(v, path, k)?
        .as_array()
        .ok_or_else(|| format!("config.json: `{path}{k}` must be a list"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .and_then(|x| u32::try_from(x).ok())
                .ok_or_else(|| format!("config.json: `{path}{k}` must be a list of small integers"))
        })
        .collect()
}

fn expect_str(v: &Value, path: &str, k: &str, want: &str) -> Result<(), String> {
    let got = str_at(v, path, k)?;
    if got != want {
        return Err(format!("config.json: `{path}{k}` is `{got}`, this engine implements `{want}`"));
    }
    Ok(())
}

fn token_id(v: &Value, path: &str, k: &str, vocab: usize) -> Result<u32, String> {
    let id = usize_at(v, path, k)?;
    if id >= vocab {
        return Err(format!("config.json: `{path}{k}` is {id}, outside the vocabulary of {vocab}"));
    }
    Ok(id as u32)
}

impl DeepseekV41Config {
    /// Parse and validate a whole `config.json`.
    pub fn from_json(root: &Value) -> Result<Self, String> {
        let t = root.get("text_config").ok_or("config.json: missing `text_config`")?;
        let tp = "text_config.";

        let vocab = positive(t, tp, "vocab_size")?;
        let hidden = positive(t, tp, "hidden_size")?;
        let layers = positive(t, tp, "num_hidden_layers")?;
        let nextn = usize_at(t, tp, "num_nextn_predict_layers")?;

        let mla = MlaConfig {
            heads: positive(t, tp, "num_attention_heads")?,
            kv_heads: positive(t, tp, "num_key_value_heads")?,
            head_dim: positive(t, tp, "head_dim")?,
            qk_rope_head_dim: usize_at(t, tp, "qk_rope_head_dim")?,
            q_lora_rank: positive(t, tp, "q_lora_rank")?,
            o_lora_rank: positive(t, tp, "o_lora_rank")?,
            o_groups: positive(t, tp, "o_groups")?,
        };
        if mla.qk_rope_head_dim >= mla.head_dim {
            return Err(format!(
                "config.json: `{tp}qk_rope_head_dim` ({}) must leave a non-rotary part of `head_dim` ({})",
                mla.qk_rope_head_dim, mla.head_dim
            ));
        }
        if mla.heads % mla.o_groups != 0 {
            return Err(format!(
                "config.json: `{tp}num_attention_heads` ({}) is not a multiple of `o_groups` ({})",
                mla.heads, mla.o_groups
            ));
        }

        let rope_scaling = field(t, tp, "rope_scaling")?;
        let rope = RopeConfig {
            theta: f64_at(t, tp, "rope_theta")?,
            compress_theta: f64_at(t, tp, "compress_rope_theta")?,
            yarn: YarnConfig {
                factor: f64_at(rope_scaling, "rope_scaling.", "factor")?,
                beta_fast: f64_at(rope_scaling, "rope_scaling.", "beta_fast")?,
                beta_slow: f64_at(rope_scaling, "rope_scaling.", "beta_slow")?,
                original_max_positions: usize_at(rope_scaling, "rope_scaling.", "original_max_position_embeddings")?,
            },
        };
        expect_str(rope_scaling, "rope_scaling.", "rope_type", "yarn")?;

        let moe = MoeConfig {
            intermediate: positive(t, tp, "moe_intermediate_size")?,
            routed_experts: positive(t, tp, "n_routed_experts")?,
            shared_experts: usize_at(t, tp, "n_shared_experts")?,
            experts_per_token: positive(t, tp, "num_experts_per_tok")?,
            scoring_func: str_at(t, tp, "scoring_func")?.to_string(),
            topk_method: str_at(t, tp, "topk_method")?.to_string(),
            norm_topk_prob: bool_at(t, tp, "norm_topk_prob")?,
            routed_scaling_factor: f64_at(t, tp, "routed_scaling_factor")?,
            swiglu_limit: f64_at(t, tp, "swiglu_limit")?,
        };
        expect_str(t, tp, "scoring_func", "sqrtsoftplus")?;
        expect_str(t, tp, "topk_method", "noaux_tc")?;
        if moe.experts_per_token > moe.routed_experts {
            return Err(format!(
                "config.json: `{tp}num_experts_per_tok` ({}) exceeds `n_routed_experts` ({})",
                moe.experts_per_token, moe.routed_experts
            ));
        }

        let compress = CompressConfig {
            ratios: ratios(t, tp, "compress_ratios")?,
            source_layer_ids: indices(t, tp, "kv_source_layer_ids")?,
            rope_theta: f64_at(t, tp, "compress_rope_theta")?,
        };
        let mtp_total = layers + nextn;
        if compress.ratios.len() != mtp_total {
            return Err(format!(
                "config.json: `{tp}compress_ratios` has {} entries but `num_hidden_layers` + `num_nextn_predict_layers` is {mtp_total}",
                compress.ratios.len()
            ));
        }
        for &l in &compress.source_layer_ids {
            if l >= layers {
                return Err(format!("config.json: `{tp}kv_source_layer_ids` names layer {l}, past the {layers} trunk layers"));
            }
        }

        let indexer = IndexerConfig {
            source_layer_ids: indices(t, tp, "index_source_layer_ids")?,
            n_heads: positive(t, tp, "index_n_heads")?,
            head_dim: positive(t, tp, "index_head_dim")?,
            topk: positive(t, tp, "index_topk")?,
            candidate_source_layer_id: usize_at(t, tp, "candidate_source_layer_id")?,
            candidate_topk_blocks: usize_at(t, tp, "candidate_topk_blocks")?,
            candidate_block_size: positive(t, tp, "candidate_block_size")?,
        };
        for &l in &indexer.source_layer_ids {
            if l >= layers {
                return Err(format!("config.json: `{tp}index_source_layer_ids` names layer {l}, past the {layers} trunk layers"));
            }
        }

        let hc = HcConfig {
            mult: positive(t, tp, "hc_mult")?,
            sinkhorn_iters: positive(t, tp, "hc_sinkhorn_iters")?,
            eps: f64_at(t, tp, "hc_eps")?,
        };

        let engram = EngramConfig {
            layer_ids: indices(t, tp, "engram_layer_ids")?,
            num_embeddings: indices(t, tp, "engram_num_embeddings")?,
            max_ngram_size: positive(t, tp, "engram_max_ngram_size")?,
            vocab_size: positive(t, tp, "engram_vocab_size")?,
            n_heads: positive(t, tp, "engram_n_heads")?,
            head_dim: positive(t, tp, "engram_head_dim")?,
            pad_token_id: token_id(t, tp, "engram_pad_token_id", vocab)?,
            compressed_vocab_size: positive(t, tp, "engram_compressed_vocab_size")?,
        };
        if engram.layer_ids.len() != engram.num_embeddings.len() {
            return Err(format!(
                "config.json: `{tp}engram_layer_ids` has {} entries but `engram_num_embeddings` has {}",
                engram.layer_ids.len(),
                engram.num_embeddings.len()
            ));
        }

        let dspark = DsparkConfig {
            block_size: positive(t, tp, "dspark_block_size")?,
            noise_token_id: token_id(t, tp, "dspark_noise_token_id", vocab)?,
            target_layer_ids: indices(t, tp, "dspark_target_layer_ids")?,
            markov_rank: positive(t, tp, "dspark_markov_rank")?,
            n_routed_experts: positive(t, tp, "dspark_n_routed_experts")?,
            num_experts_per_tok: positive(t, tp, "dspark_num_experts_per_tok")?,
        };
        if dspark.target_layer_ids.len() != nextn {
            return Err(format!(
                "config.json: `{tp}dspark_target_layer_ids` has {} entries but `num_nextn_predict_layers` is {nextn}",
                dspark.target_layer_ids.len()
            ));
        }
        for &l in &dspark.target_layer_ids {
            if l >= layers + nextn {
                return Err(format!("config.json: `{tp}dspark_target_layer_ids` names layer {l}, past layer {}", layers + nextn));
            }
        }

        let vision = match root.get("vision_config") {
            None | Some(Value::Null) => None,
            Some(v) => {
                let vp = "vision_config.";
                Some(VisionConfig {
                    model_type: str_at(v, vp, "model_type")?.to_string(),
                    layers: positive(v, vp, "num_hidden_layers")?,
                    hidden: positive(v, vp, "hidden_size")?,
                    heads: positive(v, vp, "num_attention_heads")?,
                    intermediate: positive(v, vp, "intermediate_size")?,
                    patch_size: positive(v, vp, "patch_size")?,
                    rope_theta: f64_at(v, vp, "rope_theta")?,
                    downsample_ratio: positive(v, vp, "downsample_ratio")?,
                    max_image_tokens: positive(v, vp, "max_image_tokens")?,
                    min_pixels: positive(v, vp, "min_pixels")?,
                    max_wh_ratio: field(v, vp, "max_wh_ratio")?.as_f64(),
                })
            }
        };

        Ok(Self {
            model_type: str_at(t, tp, "model_type")?.to_string(),
            vocab,
            hidden,
            layers,
            heads: mla.heads,
            kv_heads: mla.kv_heads,
            head_dim: mla.head_dim,
            qk_rope_head_dim: mla.qk_rope_head_dim,
            q_lora_rank: mla.q_lora_rank,
            o_lora_rank: mla.o_lora_rank,
            o_groups: mla.o_groups,
            hidden_act: str_at(t, tp, "hidden_act")?.to_string(),
            swiglu_limit: moe.swiglu_limit,
            rms_eps: f64_at(t, tp, "rms_norm_eps")?,
            attention_bias: bool_at(t, tp, "attention_bias")?,
            attention_dropout: f64_at(t, tp, "attention_dropout")?,
            initializer_range: f64_at(t, tp, "initializer_range")?,
            use_cache: bool_at(t, tp, "use_cache")?,
            tie_word_embeddings: bool_at(t, tp, "tie_word_embeddings")?,
            max_positions: positive(t, tp, "max_position_embeddings")?,
            rope,
            moe,
            sliding_window: usize_at(t, tp, "sliding_window")?,
            compress,
            indexer,
            hc,
            engram,
            nextn,
            dspark,
            vision,
            bos_token_id: u32_at(root, "", "bos_token_id")?,
            eos_token_id: u32_at(root, "", "eos_token_id")?,
            pad_token_id: Some(u32_at(root, "", "pad_token_id")?),
            image_token_id: u32_at(root, "", "image_token_id")?,
        })
    }

    /// Parse `dir/config.json`.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let json: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_json(&json)
    }

    pub fn mla(&self) -> MlaConfig {
        MlaConfig {
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            qk_rope_head_dim: self.qk_rope_head_dim,
            q_lora_rank: self.q_lora_rank,
            o_lora_rank: self.o_lora_rank,
            o_groups: self.o_groups,
        }
    }

    /// The trunk's layers, in order: every one an MLA layer, its indexer mode read from the two source lists, its FFN the
    /// routed mixture, its residual mHC.
    pub fn layer_specs(&self) -> Vec<LayerSpec> {
        let spec = LayerSpec {
            mixer: Mixer::MlaDsa { indexer: IndexerMode::Shared },
            ffn: Ffn::Moe,
            residual: Residual::Mhc { streams: self.hc.mult },
        };
        (0..self.layers)
            .map(|l| LayerSpec { mixer: Mixer::MlaDsa { indexer: self.indexer.mode(l, &self.compress) }, ..spec })
            .collect()
    }

    /// The NextN layers after the trunk: the same MLA + MoE + mHC block, on the plain residual.
    pub fn mtp_specs(&self) -> Vec<LayerSpec> {
        let spec = LayerSpec {
            mixer: Mixer::MlaDsa { indexer: IndexerMode::Full },
            ffn: Ffn::Moe,
            residual: Residual::Mhc { streams: self.hc.mult },
        };
        vec![spec; self.nextn]
    }

    /// Whether this config carries the fields this family's inventory is built from (a cheap guard for the registry).
    pub fn has_engram(&self) -> bool {
        !self.engram.layer_ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pack's `config.json`, committed as the family's fixture.
    const CONFIG: &str = include_str!("../../../../../testdata/deepseek_v41/config.json");

    fn json() -> Value {
        serde_json::from_str(CONFIG).unwrap()
    }

    fn real() -> DeepseekV41Config {
        DeepseekV41Config::from_json(&json()).unwrap()
    }

    #[test]
    fn the_committed_fixture_is_the_packs_own_config() {
        // the working copy of the pack's config.json, when this machine has it next to the repository
        if let Ok(text) = std::fs::read_to_string("/tmp/ds41_config.json") {
            let pack: Value = serde_json::from_str(&text).unwrap();
            assert_eq!(pack, json(), "testdata/deepseek_v41/config.json has drifted from the pack's config.json");
        }
    }

    #[test]
    fn the_pack_config_parses_into_the_values_it_states() {
        let c = real();
        assert_eq!((c.model_type.as_str(), c.vocab, c.hidden), ("deepseek_v41_text", 129_280, 5_120));
        assert_eq!((c.layers, c.nextn), (40, 3));
        assert_eq!((c.heads, c.kv_heads, c.head_dim, c.qk_rope_head_dim), (64, 1, 512, 64));
        assert_eq!((c.q_lora_rank, c.o_lora_rank, c.o_groups), (1_280, 1_024, 8));
        assert_eq!((c.hidden_act.as_str(), c.swiglu_limit), ("silu", 10.0));
        assert_eq!(c.rms_eps, 1e-20);
        assert!(!c.attention_bias && !c.tie_word_embeddings);
        assert_eq!(c.max_positions, 1_048_576);
        assert_eq!((c.bos_token_id, c.eos_token_id, c.pad_token_id), (0, 1, Some(2)));
        assert_eq!(c.image_token_id, 129_264);
        // the MLA projections the tensor shapes follow from
        assert_eq!((c.mla().q_width(), c.mla().kv_width()), (32_768, 512));
        assert_eq!((c.mla().o_inner(), c.mla().o_rank()), (4_096, 8_192));
        // RoPE: a 16x YaRN over 65,536 positions, and a separate base for the compressor
        assert_eq!((c.rope.theta, c.rope.compress_theta), (10_000.0, 160_000.0));
        assert_eq!(
            (c.rope.yarn.factor, c.rope.yarn.beta_fast, c.rope.yarn.beta_slow, c.rope.yarn.original_max_positions),
            (16.0, 32.0, 1.0, 65_536)
        );
    }

    #[test]
    fn the_mixture_the_indexer_and_the_compressors_are_read_whole() {
        let c = real();
        assert_eq!((c.moe.routed_experts, c.moe.shared_experts, c.moe.experts_per_token), (384, 1, 6));
        assert_eq!((c.moe.intermediate, c.moe.routed_scaling_factor), (2_304, 1.5));
        assert_eq!((c.moe.scoring_func.as_str(), c.moe.topk_method.as_str()), ("sqrtsoftplus", "noaux_tc"));
        assert!(c.moe.norm_topk_prob);
        assert_eq!(c.sliding_window, 128);

        // 43 ratios for 40 trunk layers and 3 NextN layers: 2 on 2..=19, 1 on 20..=39, 0 elsewhere
        assert_eq!(c.compress.ratios.len(), 43);
        assert_eq!((c.compress.ratio(0), c.compress.ratio(1)), (0, 0));
        assert_eq!(c.compress.ratio(2), 2);
        assert_eq!(c.compress.ratio(19), 2);
        assert_eq!(c.compress.ratio(20), 1);
        assert_eq!(c.compress.ratio(39), 1);
        assert_eq!((c.compress.ratio(40), c.compress.ratio(42)), (0, 0));
        assert_eq!(c.compress.source_layer_ids, vec![2, 8, 14, 20]);
        // the gate exists exactly where an entry spans more than one token: three of the four source layers
        let gated: Vec<usize> = (0..c.layers).filter(|&l| c.compress.has_gate(l)).collect();
        assert_eq!(gated, vec![2, 8, 14]);

        assert_eq!(c.indexer.source_layer_ids, vec![2, 8, 14, 20, 24, 28, 32, 36]);
        assert_eq!((c.indexer.n_heads, c.indexer.head_dim, c.indexer.topk), (32, 128, 512));
        assert_eq!(
            (c.indexer.candidate_source_layer_id, c.indexer.candidate_topk_blocks, c.indexer.candidate_block_size),
            (20, 2_048, 8)
        );
        // the eight selectors: four own their key (`wk`), four only their queries
        let full: Vec<usize> = (0..c.layers).filter(|&l| c.indexer.mode(l, &c.compress) == IndexerMode::Full).collect();
        assert_eq!(full, vec![2, 8, 14, 20]);
    }

    #[test]
    fn the_engram_tables_the_hyper_connections_and_the_dspark_head_are_read_whole() {
        let c = real();
        assert_eq!(c.engram.layer_ids, vec![1, 14]);
        assert_eq!(c.engram.num_embeddings, vec![384_006_168, 384_016_682]);
        assert_eq!(c.engram.table(1), Some(384_006_168));
        assert_eq!(c.engram.table(14), Some(384_016_682));
        assert_eq!(c.engram.table(2), None);
        assert_eq!((c.engram.max_ngram_size, c.engram.vocab_size, c.engram.n_heads, c.engram.head_dim), (4, 16_000_000, 8, 256));
        assert_eq!((c.engram.pad_token_id, c.engram.compressed_vocab_size), (2, 99_092));
        assert!(c.has_engram());

        assert_eq!((c.hc.mult, c.hc.sinkhorn_iters), (4, 20));
        assert_eq!(c.hc.eps, 1e-6);

        assert_eq!((c.dspark.block_size, c.dspark.markov_rank), (5, 256));
        assert_eq!(c.dspark.noise_token_id, 128_799);
        assert_eq!(c.dspark.target_layer_ids, vec![37, 38, 39]);
        assert_eq!((c.dspark.n_routed_experts, c.dspark.num_experts_per_tok), (128, 3));
    }

    #[test]
    fn the_vision_tower_is_read_but_not_required() {
        let v = real().vision.expect("the pack ships a vision tower");
        assert_eq!((v.layers, v.hidden, v.heads, v.intermediate), (32, 1_024, 16, 2_816));
        assert_eq!((v.patch_size, v.downsample_ratio), (14, 3));
        assert_eq!((v.max_image_tokens, v.min_pixels), (1_024, 295_936));
        assert_eq!(v.max_wh_ratio, None, "the pack states no aspect-ratio cap");
        assert_eq!(v.patch_dim(), 588);
        // a text-only config parses to no tower
        let mut j = json();
        j.as_object_mut().unwrap().remove("vision_config");
        assert!(DeepseekV41Config::from_json(&j).unwrap().vision.is_none());
    }

    #[test]
    fn every_key_of_the_pack_config_is_read() {
        let t = json()["text_config"].as_object().unwrap().clone();
        let mut unknown: Vec<&str> = t.keys().map(String::as_str).filter(|k| !KNOWN_TEXT_KEYS.contains(k)).collect();
        unknown.sort_unstable();
        assert!(unknown.is_empty(), "`text_config` keys this parser does not read: {unknown:?}");
        let mut missing: Vec<&str> = KNOWN_TEXT_KEYS.iter().copied().filter(|k| !t.contains_key(*k)).collect();
        missing.sort_unstable();
        assert!(missing.is_empty(), "`KNOWN_TEXT_KEYS` names keys the pack does not have: {missing:?}");
    }

    #[test]
    fn the_layer_recipe_is_forty_mla_moe_mhc_layers_and_three_nextn_layers() {
        let c = real();
        let layers = c.layer_specs();
        assert_eq!(layers.len(), 40);
        assert!(layers.iter().all(|l| l.ffn == Ffn::Moe && l.residual == Residual::Mhc { streams: 4 }));
        assert_eq!(c.mtp_specs().len(), 3);
        assert_eq!(c.mtp_specs()[0].residual, Residual::Mhc { streams: 4 });
    }

    #[test]
    fn a_config_that_contradicts_itself_is_refused_by_name() {
        let bad = |edit: &dyn Fn(&mut Value), needle: &str| {
            let mut j = json();
            edit(&mut j["text_config"]);
            let e = DeepseekV41Config::from_json(&j).unwrap_err();
            assert!(e.contains(needle), "expected `{needle}`, got `{e}`");
        };
        // 43 ratios for 40 + 3 layers: dropping one is a config that cannot describe a pack
        bad(
            &|t| {
                t["compress_ratios"].as_array_mut().unwrap().pop();
            },
            "compress_ratios",
        );
        bad(&|t| t["kv_source_layer_ids"] = serde_json::json!([2, 8, 14, 40]), "kv_source_layer_ids");
        bad(&|t| t["index_source_layer_ids"] = serde_json::json!([2, 40]), "index_source_layer_ids");
        bad(&|t| t["engram_num_embeddings"] = serde_json::json!([1]), "engram_num_embeddings");
        bad(&|t| t["dspark_target_layer_ids"] = serde_json::json!([37]), "dspark_target_layer_ids");
        bad(&|t| t["head_dim"] = serde_json::json!(0), "head_dim");
        bad(&|t| t["qk_rope_head_dim"] = serde_json::json!(512), "qk_rope_head_dim");
        bad(&|t| t["o_groups"] = serde_json::json!(7), "o_groups");
        bad(&|t| t["num_experts_per_tok"] = serde_json::json!(385), "num_experts_per_tok");
        bad(&|t| t["scoring_func"] = serde_json::json!("softmax"), "scoring_func");
        bad(&|t| t["dspark_noise_token_id"] = serde_json::json!(999_999), "dspark_noise_token_id");
        // and a text_config with no `text_config` at all
        let mut j = json();
        j.as_object_mut().unwrap().remove("text_config");
        assert!(DeepseekV41Config::from_json(&j).unwrap_err().contains("missing `text_config`"));
    }
}
