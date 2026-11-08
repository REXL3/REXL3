//! Mapping a pack's inventory onto [`DeepseekV41Weights`]: [`bind`].
//!
//! One pure function, `(inventory, config) -> Result<DeepseekV41Weights, BindError>`, that walks the config's recipe and claims,
//! for every slot, the one tensor that may fill it. Every dtype and every dimension is derived from `config.json` — the hidden
//! size, the ranks, the head counts, the expert counts, the engram table sizes — never from a literal, so a pack whose shapes
//! are not the ones its own config implies cannot bind.
//!
//! The contract is **first mismatch, named, no partial result**:
//!
//! * a tensor the config implies that the pack lacks is [`BindError::Missing`], carrying the path and the role it plays;
//! * a tensor whose header dtype or shape is not the implied one is [`BindError::Dtype`] or [`BindError::Shape`], carrying the
//!   path, the expectation and what was found;
//! * an EXL3 trellis whose `[k / 16, n / 16, 16 x bits]` is not the implied one is [`BindError::Trellis`] — its own variant
//!   because the width is per expert and a wrong width is a different failure from a wrong matrix;
//! * a pack that is short a whole layer is [`BindError::LayerCount`] (or [`BindError::LayerHole`] for a gap), and one that is
//!   short experts is [`BindError::ExpertCount`] — both found before any tensor is claimed, so deleting a layer reports the layer
//!   and not the first of its 4,638 tensors;
//! * a config that cannot describe the layout it claims (an FP8 tile that is not a multiple of 32, an odd int8 reduction width)
//!   is [`BindError::Config`];
//! * and a tensor in the pack that no slot claimed is [`BindError::Unexpected`] — the check that makes the result exact.
//!
//! The walk is fallible at every step and returns on the first failure, so a successful bind is a proof that the pack holds
//! exactly the 188,245 tensors the config describes, each with the right dtype and shape, and nothing else.

use super::config::DeepseekV41Config;
use super::inventory::{HeadFormat, MtpExpertFormat, PackInventory};
use super::weights::*;
use std::collections::{BTreeMap, BTreeSet};

/// The FP8 block scale is 32x32: a projection's two sides must both be multiples of this.
const FP8_TILE: usize = 32;
/// An int8 expert packs two values per byte.
const INT4_PACK: usize = 2;
/// The largest trellis width in bits (`trellis.shape[2] / 16`).
const MAX_BITS: u8 = 8;

/// Why a pack does not bind onto this family's weights. Every variant names the tensor (or the layer) it is about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindError {
    /// A tensor the config implies is not in the pack. `role` says what the forward would have done with it.
    Missing { path: String, role: &'static str },
    /// A tensor is present with the wrong dtype.
    Dtype { path: String, want: Dtype, got: String },
    /// A tensor is present with the wrong shape.
    Shape { path: String, want: Vec<usize>, got: Vec<usize> },
    /// An EXL3 trellis is not `[k / 16, n / 16, 16 x bits]` for the implied `(k, n)` and a width in `1..=8`.
    Trellis { path: String, want: (usize, usize), got: Vec<usize> },
    /// The pack has a different number of trunk layers than the config states.
    LayerCount { want: usize, got: usize },
    /// The pack's trunk layers are not `0..layers`: `index` is the first one missing.
    LayerHole { index: usize, layers: usize },
    /// The pack has a different number of NextN layers than the config states.
    MtpCount { want: usize, got: usize },
    /// A trunk layer has a different number of routed experts than the config states.
    ExpertCount { layer: usize, want: usize, got: usize },
    /// A NextN layer has a different number of routed experts than the config states.
    MtpExpertCount { layer: usize, want: usize, got: usize },
    /// A tensor is in the pack and no slot of this family takes it.
    Unexpected { path: String },
    /// The config itself does not describe a layout this family can build.
    Config(String),
}

impl std::fmt::Display for BindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BindError::Missing { path, role } => write!(f, "deepseek_v41: `{path}` is not in the pack ({role})"),
            BindError::Dtype { path, want, got } => {
                write!(f, "deepseek_v41: `{path}`: expected {want}, found {got}")
            }
            BindError::Shape { path, want, got } => {
                write!(f, "deepseek_v41: `{path}`: expected shape {want:?}, found {got:?}")
            }
            BindError::Trellis { path, want, got } => write!(
                f,
                "deepseek_v41: `{path}`: expected an EXL3 trellis [{}, {}, 16..=128], found {got:?}",
                want.0 / 16,
                want.1 / 16
            ),
            BindError::LayerCount { want, got } => {
                write!(f, "deepseek_v41: the pack has {got} trunk layers, the config states {want}")
            }
            BindError::LayerHole { index, layers } => {
                write!(f, "deepseek_v41: the pack is missing trunk layer {index} of 0..{layers}")
            }
            BindError::MtpCount { want, got } => {
                write!(f, "deepseek_v41: the pack has {got} NextN layers, the config states {want}")
            }
            BindError::ExpertCount { layer, want, got } => {
                write!(f, "deepseek_v41: trunk layer {layer} has {got} routed experts, the config states {want}")
            }
            BindError::MtpExpertCount { layer, want, got } => {
                write!(f, "deepseek_v41: NextN layer {layer} has {got} routed experts, the config states {want}")
            }
            BindError::Unexpected { path } => {
                write!(f, "deepseek_v41: `{path}` is in the pack and no slot of this family takes it")
            }
            BindError::Config(m) => write!(f, "deepseek_v41: {m}"),
        }
    }
}

impl std::error::Error for BindError {}

// -------------------------------------------------------------------------------------------------------------- the pre-flight

/// `"<index>.<rest>"` for a name whose first segment is a layer index.
fn split_index(rest: &str) -> Option<(usize, &str)> {
    let dot = rest.find('.')?;
    Some((rest[..dot].parse().ok()?, &rest[dot + 1..]))
}

/// The trunk layer indices the pack has *any* tensor for.
fn trunk_layers(inv: &PackInventory) -> BTreeSet<usize> {
    inv.tensors.keys().filter_map(|n| n.strip_prefix("layers.").and_then(split_index).map(|(l, _)| l)).collect()
}

/// The NextN layer indices the pack has any tensor for.
fn mtp_layers(inv: &PackInventory) -> BTreeSet<usize> {
    inv.tensors.keys().filter_map(|n| n.strip_prefix("mtp.").and_then(split_index).map(|(l, _)| l)).collect()
}

/// The routed expert ids each layer of `root` (`"layers"` or `"mtp"`) has any tensor for.
fn experts_by_layer(inv: &PackInventory, root: &str) -> BTreeMap<usize, BTreeSet<usize>> {
    let mut out: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for name in inv.tensors.keys() {
        let Some(rest) = name.strip_prefix(root).and_then(|r| r.strip_prefix('.')) else { continue };
        let Some((layer, rest)) = split_index(rest) else { continue };
        let Some(rest) = rest.strip_prefix("ffn.experts.") else { continue };
        let Some((expert, _)) = split_index(rest) else { continue };
        out.entry(layer).or_default().insert(expert);
    }
    out
}

// ------------------------------------------------------------------------------------------------------------------ the binder

struct Binder<'a> {
    inv: &'a PackInventory,
    claimed: BTreeSet<&'a str>,
}

impl<'a> Binder<'a> {
    /// Claim one tensor, checking the dtype and shape the config implies.
    fn tensor(&mut self, path: &str, dtype: Dtype, shape: &[usize], role: &'static str) -> Result<Tensor, BindError> {
        let Some((key, t)) = self.inv.tensors.get_key_value(path) else {
            return Err(BindError::Missing { path: path.to_string(), role });
        };
        self.claimed.insert(key.as_str());
        if t.dtype != dtype.name() {
            return Err(BindError::Dtype { path: path.to_string(), want: dtype, got: t.dtype.clone() });
        }
        if t.shape != shape {
            return Err(BindError::Shape { path: path.to_string(), want: shape.to_vec(), got: t.shape.clone() });
        }
        Ok(Tensor::new(path, dtype, shape))
    }

    /// [`Self::tensor`] for a tensor the packs ship in more than one dtype (the engram's n-gram
    /// projections: bf16 in the base layout, f32 in the quantized one). The descriptor keeps the
    /// listing's dtype, so the plan charges the bytes the pack really holds.
    fn tensor_dtype_any(
        &mut self,
        path: &str,
        dtypes: &[Dtype],
        shape: &[usize],
        role: &'static str,
    ) -> Result<Tensor, BindError> {
        let Some((key, t)) = self.inv.tensors.get_key_value(path) else {
            return Err(BindError::Missing { path: path.to_string(), role });
        };
        self.claimed.insert(key.as_str());
        let Some(dt) = dtypes.iter().find(|d| d.name() == t.dtype) else {
            return Err(BindError::Dtype { path: path.to_string(), want: dtypes[0], got: t.dtype.clone() });
        };
        if t.shape != shape {
            return Err(BindError::Shape { path: path.to_string(), want: shape.to_vec(), got: t.shape.clone() });
        }
        Ok(Tensor::new(path, *dt, shape))
    }

    /// An FP8 projection `[n, k]` with its `[n / 32, k / 32]` block scales (or an EXL3 group `k x n` when quantized in place).
    fn fp8(&mut self, prefix: &str, n: usize, k: usize, role: &'static str) -> Result<Fp8Linear, BindError> {
        if n % FP8_TILE != 0 || k % FP8_TILE != 0 {
            return Err(BindError::Config(format!(
                "`{prefix}` is {n}x{k}, which is not a whole number of {FP8_TILE}x{FP8_TILE} scale tiles"
            )));
        }
        if self.inv.tensors.contains_key(&format!("{prefix}.trellis")) {
            let g = self.exl3(prefix, k, n)?;
            return Ok(Fp8Linear {
                weight: Tensor::new(format!("{prefix}.weight"), Dtype::Fp8, [n, k]),
                scale: Tensor::new(format!("{prefix}.scale"), Dtype::E8m0, [n / FP8_TILE, k / FP8_TILE]),
                exl3: Some(g),
            });
        }
        Ok(Fp8Linear {
            weight: self.tensor(&format!("{prefix}.weight"), Dtype::Fp8, &[n, k], role)?,
            scale: self.tensor(
                &format!("{prefix}.scale"),
                Dtype::E8m0,
                &[n / FP8_TILE, k / FP8_TILE],
                "the 32x32 block scales of an FP8 projection",
            )?,
            exl3: None,
        })
    }

    /// An EXL3 group `k x n`: the trellis (whose width is the expert's own), the `mul1` marker and the two fp16 scales.
    fn exl3(&mut self, prefix: &str, k: usize, n: usize) -> Result<Exl3Linear, BindError> {
        let path = format!("{prefix}.trellis");
        let Some((key, t)) = self.inv.tensors.get_key_value(&path) else {
            return Err(BindError::Missing { path, role: "an EXL3 routed expert projection" });
        };
        self.claimed.insert(key.as_str());
        if t.dtype != Dtype::I16.name() {
            return Err(BindError::Dtype { path, want: Dtype::I16, got: t.dtype.clone() });
        }
        let ok = t.shape.len() == 3
            && t.shape[0] == k / 16
            && t.shape[1] == n / 16
            && t.shape[2] % 16 == 0
            && (16..=16 * MAX_BITS as usize).contains(&t.shape[2]);
        if !ok {
            return Err(BindError::Trellis { path, want: (k, n), got: t.shape.clone() });
        }
        let bits = (t.shape[2] / 16) as u8;
        let trellis = Tensor::new(&path, Dtype::I16, t.shape.clone());
        Ok(Exl3Linear {
            prefix: prefix.to_string(),
            k,
            n,
            bits,
            trellis,
            mul1: self.tensor(&format!("{prefix}.mul1"), Dtype::I32, &[], "the EXL3 codebook marker")?,
            suh: self.tensor(&format!("{prefix}.suh"), Dtype::F16, &[k], "an EXL3 input scale")?,
            svh: self.tensor(&format!("{prefix}.svh"), Dtype::F16, &[n], "an EXL3 output scale")?,
        })
    }

    /// An int8 4-bit group `k x n`: the packed weight and one scale per 32 inputs.
    fn int4(&mut self, prefix: &str, k: usize, n: usize) -> Result<Int4Linear, BindError> {
        if k % (INT4_PACK * FP8_TILE) != 0 {
            return Err(BindError::Config(format!(
                "`{prefix}` reduces over {k}, which is not a whole number of {FP8_TILE}-input int8 scale groups"
            )));
        }
        Ok(Int4Linear {
            prefix: prefix.to_string(),
            k,
            n,
            weight: self.tensor(
                &format!("{prefix}.weight"),
                Dtype::I8,
                &[n, k / INT4_PACK],
                "an int8 4-bit expert projection",
            )?,
            scale: self.tensor(
                &format!("{prefix}.scale"),
                Dtype::E8m0,
                &[n, k / FP8_TILE],
                "the per-32-input scales of an int8 expert",
            )?,
        })
    }

    fn mla(&mut self, p: &str, cfg: &DeepseekV41Config) -> Result<Mla, BindError> {
        let h = cfg.hidden;
        let q_rank = cfg.q_lora_rank;
        let q_width = cfg.heads * cfg.head_dim;
        let kv = cfg.kv_heads * cfg.head_dim;
        let o_inner = cfg.o_groups * cfg.head_dim;
        let o_rank = cfg.o_groups * cfg.o_lora_rank;
        Ok(Mla {
            wq_a: self.fp8(&format!("{p}wq_a"), q_rank, h, "the query's low-rank projection")?,
            wq_b: self.fp8(&format!("{p}wq_b"), q_width, q_rank, "the query's up-projection")?,
            wkv: self.fp8(&format!("{p}wkv"), kv, h, "the shared KV latent")?,
            wo_a: self.fp8(&format!("{p}wo_a"), o_rank, o_inner, "the output's grouped low-rank projection")?,
            wo_b: self.fp8(&format!("{p}wo_b"), h, o_rank, "the output's up-projection")?,
            q_norm: self.tensor(&format!("{p}q_norm.weight"), Dtype::Bf16, &[q_rank], "the query norm")?,
            kv_norm: self.tensor(&format!("{p}kv_norm.weight"), Dtype::Bf16, &[kv], "the latent norm")?,
            attn_sink: self.tensor(&format!("{p}attn_sink"), Dtype::F32, &[cfg.heads], "the per-head sink logits")?,
        })
    }

    fn router(&mut self, p: &str, experts: usize, hidden: usize) -> Result<Router, BindError> {
        Ok(Router {
            weight: self.tensor(&format!("{p}gate.weight"), Dtype::Bf16, &[experts, hidden], "the router logits")?,
            bias: self.tensor(&format!("{p}gate.bias"), Dtype::F32, &[experts], "the router's correction bias")?,
            bias_vl: self.tensor(&format!("{p}gate.bias_vl"), Dtype::F32, &[experts], "the router's vision-path bias")?,
        })
    }

    fn shared(&mut self, p: &str, hidden: usize, inter: usize) -> Result<SharedExperts, BindError> {
        Ok(SharedExperts {
            w1: self.fp8(&format!("{p}shared_experts.w1"), inter, hidden, "the shared expert's gate")?,
            w2: self.fp8(&format!("{p}shared_experts.w2"), hidden, inter, "the shared expert's down projection")?,
            w3: self.fp8(&format!("{p}shared_experts.w3"), inter, hidden, "the shared expert's up projection")?,
        })
    }

    fn hc(&mut self, p: &str, which: &str, cfg: &DeepseekV41Config) -> Result<Hc, BindError> {
        let mult = cfg.hc.mult;
        Ok(Hc {
            fn_: self.tensor(
                &format!("{p}hc_{which}_fn"),
                Dtype::F32,
                &[HC_COEFFS_PER_STREAM * mult, mult * cfg.hidden],
                "an mHC mix matrix",
            )?,
            base: self.tensor(&format!("{p}hc_{which}_base"), Dtype::F32, &[HC_COEFFS_PER_STREAM * mult], "an mHC mix bias")?,
            scale: self.tensor(&format!("{p}hc_{which}_scale"), Dtype::F32, &[HC_SCALE], "the mHC scale")?,
        })
    }

    fn trunk_layer(&mut self, cfg: &DeepseekV41Config, index: usize) -> Result<TrunkLayer, BindError> {
        let p = format!("layers.{index}.");
        let h = cfg.hidden;
        let inter = cfg.moe.intermediate;
        let attn_norm = self.tensor(&format!("{p}attn_norm.weight"), Dtype::Bf16, &[h], "the attention norm")?;
        let ffn_norm = self.tensor(&format!("{p}ffn_norm.weight"), Dtype::Bf16, &[h], "the FFN norm")?;
        let hc_attn = self.hc(&p, "attn", cfg)?;
        let hc_ffn = self.hc(&p, "ffn", cfg)?;
        let mla = self.mla(&format!("{p}attn."), cfg)?;
        let router = self.router(&format!("{p}ffn."), cfg.moe.routed_experts, h)?;
        let shared = self.shared(&format!("{p}ffn."), h, inter)?;
        let mut experts = Vec::with_capacity(cfg.moe.routed_experts);
        for e in 0..cfg.moe.routed_experts {
            let x = format!("{p}ffn.experts.{e}.");
            experts.push(Expert {
                w1: self.exl3(&format!("{x}w1"), h, inter)?,
                w2: self.exl3(&format!("{x}w2"), inter, h)?,
                w3: self.exl3(&format!("{x}w3"), h, inter)?,
            });
        }
        let compressor = if cfg.compress.is_source(index) {
            let c = format!("{p}attn.compressor.");
            let d = cfg.head_dim;
            Some(Compressor {
                wkv: self.tensor(&format!("{c}wkv.weight"), Dtype::Bf16, &[d, h], "the compressed-KV projection")?,
                norm: self.tensor(&format!("{c}norm.weight"), Dtype::Bf16, &[d], "the compressor norm")?,
                gate: if cfg.compress.has_gate(index) {
                    Some(self.tensor(&format!("{c}wgate.weight"), Dtype::Bf16, &[d, h], "the compressor gate")?)
                } else {
                    None
                },
            })
        } else {
            None
        };
        let indexer = if cfg.indexer.is_source(index) {
            let i = format!("{p}attn.indexer.");
            let (n_heads, d) = (cfg.indexer.n_heads, cfg.indexer.head_dim);
            Some(Indexer {
                wq_b: self.fp8(&format!("{i}wq_b"), n_heads * d, cfg.q_lora_rank, "the indexer's query projection")?,
                weights_proj: self.tensor(
                    &format!("{i}weights_proj.weight"),
                    Dtype::Bf16,
                    &[n_heads, h],
                    "the indexer's per-head score weights",
                )?,
                wk: if cfg.compress.is_source(index) {
                    Some(self.tensor(
                        &format!("{i}wk.weight"),
                        Dtype::Bf16,
                        &[d, cfg.head_dim],
                        "the indexer's key projection",
                    )?)
                } else {
                    None
                },
                k_norm: if cfg.compress.is_source(index) {
                    Some(self.tensor(&format!("{i}k_norm.weight"), Dtype::Bf16, &[d], "the indexer's key norm")?)
                } else {
                    None
                },
            })
        } else {
            None
        };
        let engram = match cfg.engram.table(index) {
            None => None,
            Some(rows) => {
                let e = format!("{p}engram.");
                let d = cfg.engram.head_dim;
                Some(Engram {
                    table: MmapTable {
                        layer: index,
                        rows,
                        embed: self.tensor(
                            &format!("{e}embed.weight"),
                            Dtype::Fp8,
                            &[rows, d],
                            "an engram table, addressed through an mmap of its shard",
                        )?,
                        scale: self.tensor(
                            &format!("{e}embed.scale"),
                            Dtype::E8m0,
                            &[rows, d / FP8_TILE],
                            "an engram table's block scales",
                        )?,
                    },
                    q_weight: self.tensor_dtype_any(
                        &format!("{e}q_weight"),
                        &[Dtype::Bf16, Dtype::F32],
                        &[cfg.engram.max_ngram_size, h],
                        "the engram's query projection",
                    )?,
                    k_weight: self.tensor_dtype_any(
                        &format!("{e}k_weight"),
                        &[Dtype::Bf16, Dtype::F32],
                        &[cfg.engram.max_ngram_size, h],
                        "the engram's key projection",
                    )?,
                    wkv: if self.inv.tensors.contains_key(&format!("{e}wkv.trellis")) {
                        WkvProj::Group(self.exl3(
                            &format!("{e}wkv"),
                            ENGRAM_WKV_COL_FACTOR * cfg.engram.n_heads * d,
                            ENGRAM_WKV_ROW_FACTOR * h,
                        )?)
                    } else {
                        WkvProj::Plain(self.fp8(
                            &format!("{e}wkv"),
                            ENGRAM_WKV_ROW_FACTOR * h,
                            ENGRAM_WKV_COL_FACTOR * cfg.engram.n_heads * d,
                            "the engram's fused key/value projection",
                        )?)
                    },
                })
            }
        };
        Ok(TrunkLayer {
            index,
            attn_norm,
            ffn_norm,
            mla,
            router,
            shared,
            experts: ExpertBank { layer: index, experts },
            hc_attn,
            hc_ffn,
            compressor,
            indexer,
            engram,
        })
    }

    fn mtp_layer(&mut self, cfg: &DeepseekV41Config, index: usize) -> Result<MtpLayer, BindError> {
        let p = format!("mtp.{index}.");
        let h = cfg.hidden;
        let inter = cfg.moe.intermediate;
        let n = cfg.dspark.n_routed_experts;
        let attn_norm = self.tensor(&format!("{p}attn_norm.weight"), Dtype::Bf16, &[h], "the attention norm")?;
        let ffn_norm = self.tensor(&format!("{p}ffn_norm.weight"), Dtype::Bf16, &[h], "the FFN norm")?;
        let hc_attn = self.hc(&p, "attn", cfg)?;
        let hc_ffn = self.hc(&p, "ffn", cfg)?;
        let mla = self.mla(&format!("{p}attn."), cfg)?;
        let router = self.router(&format!("{p}ffn."), n, h)?;
        let shared = self.shared(&format!("{p}ffn."), h, inter)?;
        let mut experts = Vec::with_capacity(n);
        // the format is the pack's, not the config's: the first upload stores int8 4-bit, the re-upload EXL3 K2
        let fmt = self.inv.mtp_experts();
        for e in 0..n {
            let x = format!("{p}ffn.experts.{e}.");
            experts.push(match fmt {
                MtpExpertFormat::Int4 => MtpExpert::Int4(Int4Expert {
                    w1: self.int4(&format!("{x}w1"), h, inter)?,
                    w2: self.int4(&format!("{x}w2"), inter, h)?,
                    w3: self.int4(&format!("{x}w3"), h, inter)?,
                }),
                MtpExpertFormat::Exl3 => MtpExpert::Exl3(Expert {
                    w1: self.exl3(&format!("{x}w1"), h, inter)?,
                    w2: self.exl3(&format!("{x}w2"), inter, h)?,
                    w3: self.exl3(&format!("{x}w3"), h, inter)?,
                }),
            });
        }
        let first = index == 0;
        let last = index + 1 == cfg.nextn;
        Ok(MtpLayer {
            index,
            attn_norm,
            ffn_norm,
            mla,
            router,
            shared,
            experts,
            hc_attn,
            hc_ffn,
            main_proj: if first {
                Some(self.fp8(&format!("{p}main_proj"), h, MAIN_PROJ_TAPS * h, "the three trunk taps this layer fuses")?)
            } else {
                None
            },
            main_norm: if first {
                Some(self.tensor(&format!("{p}main_norm.weight"), Dtype::Bf16, &[h], "the fused-tap norm")?)
            } else {
                None
            },
            markov_embed: if last {
                Some(self.tensor(
                    &format!("{p}markov_head.embed.weight"),
                    Dtype::Bf16,
                    &[cfg.vocab, cfg.dspark.markov_rank],
                    "the DSpark Markov embedding",
                )?)
            } else {
                None
            },
            markov_head: if last {
                Some(self.tensor(
                    &format!("{p}markov_head.head.weight"),
                    Dtype::Bf16,
                    &[cfg.vocab, cfg.dspark.markov_rank],
                    "the DSpark Markov head",
                )?)
            } else {
                None
            },
            confidence: if last {
                Some(self.tensor(
                    &format!("{p}confidence_head.proj.weight"),
                    Dtype::Bf16,
                    &[1, h + cfg.dspark.markov_rank],
                    "the DSpark confidence head",
                )?)
            } else {
                None
            },
            norm: if last {
                Some(self.tensor(&format!("{p}norm.weight"), Dtype::Bf16, &[h], "the NextN block's final norm")?)
            } else {
                None
            },
        })
    }

    fn vision_linear(&mut self, base: &str, w: &[usize], n: usize, role: &'static str) -> Result<VisionLinear, BindError> {
        Ok(VisionLinear {
            weight: self.tensor(&format!("{base}.weight"), Dtype::Bf16, w, role)?,
            bias: self.tensor(&format!("{base}.bias"), Dtype::Bf16, &[n], role)?,
        })
    }

    fn vision(&mut self, cfg: &DeepseekV41Config, v: &super::config::VisionConfig) -> Result<Vision, BindError> {
        let (h, i) = (v.hidden, v.intermediate);
        let patch_embed = self.vision_linear("vision.patch_embed.proj", &[h, v.patch_dim()], h, "a vision tower projection")?;
        let mut blocks = Vec::with_capacity(v.layers);
        for b in 0..v.layers {
            let p = format!("vision.blocks.{b}.");
            blocks.push(VisionBlock {
                norm1: self.tensor(&format!("{p}norm1.weight"), Dtype::Bf16, &[h], "a vision block norm")?,
                norm2: self.tensor(&format!("{p}norm2.weight"), Dtype::Bf16, &[h], "a vision block norm")?,
                qkv: self.vision_linear(&format!("{p}attn.wqkv"), &[3 * h, h], 3 * h, "a vision block projection")?,
                wo: self.vision_linear(&format!("{p}attn.wo"), &[h, h], h, "a vision block projection")?,
                mlp_w1: self.tensor(&format!("{p}mlp.w1.weight"), Dtype::Bf16, &[2 * i, h], "a vision block MLP gate")?,
                mlp_w2: self.tensor(&format!("{p}mlp.w2.weight"), Dtype::Bf16, &[h, i], "a vision block MLP down projection")?,
            });
        }
        Ok(Vision {
            patch_embed,
            blocks,
            norm: self.tensor("vision.norm.weight", Dtype::Bf16, &[h], "the tower's final norm")?,
            image_start: self.tensor("image_start", Dtype::Bf16, &[cfg.hidden], "the row spliced before an image")?,
            image_end: self.tensor("image_end", Dtype::Bf16, &[cfg.hidden], "the row spliced after an image")?,
            image_newline: self.tensor("image_newline", Dtype::Bf16, &[cfg.hidden], "the row spliced between image lines")?,
        })
    }
}

impl DeepseekV41Weights {
    /// Bind a pack's inventory onto the family's weights. See the module documentation: the result is exact (every tensor of the
    /// pack is claimed once, and nothing else) or the first [`BindError`].
    pub fn bind(inv: &PackInventory, cfg: &DeepseekV41Config) -> Result<DeepseekV41Weights, BindError> {
        bind(inv, cfg)
    }
}

/// Map a pack's inventory onto [`DeepseekV41Weights`], or name the first thing that does not fit.
pub fn bind(inv: &PackInventory, cfg: &DeepseekV41Config) -> Result<DeepseekV41Weights, BindError> {
    // ---- pre-flight: layer and expert counts, so a pack that is short a layer reports the layer
    let trunk = trunk_layers(inv);
    if trunk.len() != cfg.layers {
        return Err(BindError::LayerCount { want: cfg.layers, got: trunk.len() });
    }
    if let Some(index) = (0..cfg.layers).find(|i| !trunk.contains(i)) {
        return Err(BindError::LayerHole { index, layers: cfg.layers });
    }
    let mtp = mtp_layers(inv);
    if mtp.len() != cfg.nextn {
        return Err(BindError::MtpCount { want: cfg.nextn, got: mtp.len() });
    }
    if let Some(_hole) = (0..cfg.nextn).find(|i| !mtp.contains(i)) {
        return Err(BindError::MtpCount { want: cfg.nextn, got: mtp.len() });
    }
    let trunk_experts = experts_by_layer(inv, "layers");
    for layer in 0..cfg.layers {
        let got = trunk_experts.get(&layer).map_or(0, BTreeSet::len);
        if got != cfg.moe.routed_experts {
            return Err(BindError::ExpertCount { layer, want: cfg.moe.routed_experts, got });
        }
    }
    let mtp_experts = experts_by_layer(inv, "mtp");
    for layer in 0..cfg.nextn {
        let got = mtp_experts.get(&layer).map_or(0, BTreeSet::len);
        if got != cfg.dspark.n_routed_experts {
            return Err(BindError::MtpExpertCount { layer, want: cfg.dspark.n_routed_experts, got });
        }
    }

    // ---- the walk
    let mut b = Binder { inv, claimed: BTreeSet::new() };
    let h = cfg.hidden;
    let embed = b.tensor("embed.weight", Dtype::Bf16, &[cfg.vocab, h], "the token embedding")?;
    let head = match inv.head_format() {
        HeadFormat::Bf16 => LmHead::Bf16(b.tensor("head.weight", Dtype::Bf16, &[cfg.vocab, h], "the LM head")?),
        HeadFormat::Exl3 => LmHead::Exl3(b.exl3("head", h, cfg.vocab)?),
    };
    let norm = b.tensor("norm.weight", Dtype::Bf16, &[h], "the final norm")?;

    let mut layers = Vec::with_capacity(cfg.layers);
    for index in 0..cfg.layers {
        layers.push(b.trunk_layer(cfg, index)?);
    }
    let mut mtp_layers = Vec::with_capacity(cfg.nextn);
    for index in 0..cfg.nextn {
        mtp_layers.push(b.mtp_layer(cfg, index)?);
    }

    let (vision, aligner) = match &cfg.vision {
        None => (None, None),
        Some(v) => {
            let tower = b.vision(cfg, v)?;
            let m2 = v.downsample_ratio * v.downsample_ratio;
            let aligner = Aligner {
                w1: VisionLinear {
                    weight: b.tensor("aligner.w1.weight", Dtype::Bf16, &[h, v.hidden * m2], "the aligner's up-projection")?,
                    bias: b.tensor("aligner.w1.bias", Dtype::Bf16, &[h], "the aligner's up-projection bias")?,
                },
                w2: VisionLinear {
                    weight: b.tensor("aligner.w2.weight", Dtype::Bf16, &[h, h], "the aligner's output projection")?,
                    bias: b.tensor("aligner.w2.bias", Dtype::Bf16, &[h], "the aligner's output projection bias")?,
                },
            };
            (Some(tower), Some(aligner))
        }
    };

    // ---- fail closed: whatever no slot claimed is an error
    if let Some(path) = inv.tensors.keys().find(|n| !b.claimed.contains(n.as_str())) {
        return Err(BindError::Unexpected { path: path.clone() });
    }

    Ok(DeepseekV41Weights { embed, head, norm, layers, mtp: mtp_layers, vision, aligner })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::deepseek_v41::config::DeepseekV41Config;
    use crate::families::deepseek_v41::inventory::TensorInfo;
    use serde_json::Value;

    /// The pack's `config.json`, committed as the family's fixture.
    const CONFIG: &str = include_str!("../../../../../testdata/deepseek_v41/config.json");
    /// The complete tensor listing of the real 17-shard pack (188,245 tensors), when this machine has it.
    const PACK: &str = "/tmp/dsv41_inventory.json";

    fn cfg() -> DeepseekV41Config {
        DeepseekV41Config::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    /// The real pack's inventory, or `None` on a machine that does not have the listing.
    fn pack() -> Option<(DeepseekV41Config, PackInventory)> {
        let text = std::fs::read_to_string(PACK).ok()?;
        let listing: Value = serde_json::from_str(&text).unwrap();
        let cfg = cfg();
        let inv = PackInventory::from_json(&cfg, &listing).expect("the real pack classifies against its config");
        Some((cfg, inv))
    }

    /// `Some((cfg, inventory))`, or skip: the 18 MB listing is not in the repository.
    macro_rules! pack_or_skip {
        () => {
            match pack() {
                Some(x) => x,
                None => {
                    eprintln!("not run: {PACK} is not on this machine");
                    return;
                }
            }
        };
    }

    /// Drop every tensor under `prefix`, as a pack that never had them.
    fn drop_prefix(inv: &mut PackInventory, prefix: &str) -> usize {
        let before = inv.tensors.len();
        inv.tensors.retain(|k, _| !k.starts_with(prefix));
        before - inv.tensors.len()
    }

    #[test]
    fn the_real_pack_binds_exactly() {
        let (cfg, inv) = pack_or_skip!();
        let w = bind(&inv, &cfg).unwrap_or_else(|e| panic!("the real pack does not bind: {e}"));

        // every tensor of the listing is described once, and the tree describes nothing the pack lacks
        assert_eq!(w.len(), 188_245);
        assert_eq!(w.len(), inv.len());
        assert_eq!(w.bytes(), 332_513_253_720, "309.7 GiB of tensors over 17 shards");

        // the trunk, the NextN layers, and the two engram tables
        assert_eq!((w.layers.len(), w.mtp.len()), (40, 3));
        assert!(w.layers.iter().enumerate().all(|(i, l)| l.index == i));
        assert_eq!(w.experts(0).unwrap().len(), 384);
        assert_eq!(w.mtp[0].experts.len(), 128);
        let tables: Vec<(usize, usize)> = w.engram_tables().map(|t| (t.layer, t.rows)).collect();
        assert_eq!(tables, vec![(1, 384_006_168), (14, 384_016_682)]);
        let t1 = w.engram_tables().next().unwrap();
        assert_eq!((t1.embed.shape.clone(), t1.scale.shape.clone()), (vec![384_006_168, 256], vec![384_006_168, 8]));

        // the three branches that exist only on some layers
        let compressors = w.layers.iter().filter(|l| l.compressor.is_some()).count();
        let gated = w.layers.iter().filter(|l| l.compressor.as_ref().is_some_and(|c| c.gate.is_some())).count();
        assert_eq!((compressors, gated), (4, 3), "four compressors, three of them gated");
        assert_eq!(w.layers.iter().filter(|l| l.indexer.is_some()).count(), 8);
        assert_eq!(w.layers.iter().filter(|l| l.engram.is_some()).count(), 2);

        // the bit width is the expert's, and the pack uses 1..=6 of the 8 the format has
        let widths = w.experts(0).unwrap().widths();
        assert_eq!(widths, vec![(1, 725), (2, 254), (3, 115), (4, 52), (5, 6)], "layer 0's 1,152 projections");
        let all: usize = w.layers.iter().flat_map(|l| l.experts.widths()).map(|(_, n)| n).sum();
        assert_eq!(all, 40 * 384 * 3);

        // the dtypes the tree describes
        assert_eq!(
            w.dtypes(),
            vec![
                (Dtype::Bf16, 520),
                (Dtype::F16, 92_160),
                (Dtype::F32, 387),
                (Dtype::I16, 46_080),
                (Dtype::I32, 46_080),
                (Dtype::I8, 1_152),
                (Dtype::Fp8, 357),
                (Dtype::E8m0, 1_509),
            ]
        );
    }

    #[test]
    fn a_missing_tensor_is_named_with_its_role() {
        let (cfg, mut inv) = pack_or_skip!();
        let path = "layers.5.attn.wo_b.scale";
        assert_eq!(drop_prefix(&mut inv, path), 1);
        let e = bind(&inv, &cfg).unwrap_err();
        assert_eq!(e, BindError::Missing { path: path.into(), role: "the 32x32 block scales of an FP8 projection" });
        assert!(e.to_string().contains(path), "{e}");
    }

    #[test]
    fn a_wrong_dtype_is_named_with_both() {
        let (cfg, mut inv) = pack_or_skip!();
        let path = "layers.0.attn.wq_a.weight";
        inv.tensors.get_mut(path).unwrap().dtype = "BF16".into();
        let e = bind(&inv, &cfg).unwrap_err();
        assert_eq!(e, BindError::Dtype { path: path.into(), want: Dtype::Fp8, got: "BF16".into() });
        assert!(e.to_string().contains("expected F8_E4M3, found BF16"), "{e}");
    }

    #[test]
    fn a_wrong_shape_is_named_with_both() {
        let (cfg, mut inv) = pack_or_skip!();
        let path = "layers.3.attn.wq_b.weight";
        inv.tensors.get_mut(path).unwrap().shape = vec![32_768, 1_281];
        let e = bind(&inv, &cfg).unwrap_err();
        assert_eq!(e, BindError::Shape { path: path.into(), want: vec![32_768, 1_280], got: vec![32_768, 1_281] });
    }

    #[test]
    fn a_wrong_expert_trellis_is_named_with_its_k_and_n() {
        let (cfg, mut inv) = pack_or_skip!();
        let path = "layers.0.ffn.experts.7.w2.trellis";
        // a trellis whose N is one block short: the shape is well formed and still wrong
        inv.tensors.get_mut(path).unwrap().shape = vec![144, 319, 32];
        let e = bind(&inv, &cfg).unwrap_err();
        assert_eq!(e, BindError::Trellis { path: path.into(), want: (2_304, 5_120), got: vec![144, 319, 32] });
        assert!(e.to_string().contains("[144, 320, 16..=128]"), "{e}");

        // a width that is not a whole number of 16-bit steps is the same variant
        let (_, mut inv) = pack().unwrap();
        inv.tensors.get_mut(path).unwrap().shape = vec![144, 320, 40];
        assert!(matches!(bind(&inv, &cfg), Err(BindError::Trellis { .. })));
        // and so is a width past the 8 bits the format has
        let (_, mut inv) = pack().unwrap();
        inv.tensors.get_mut(path).unwrap().shape = vec![144, 320, 160];
        assert!(matches!(bind(&inv, &cfg), Err(BindError::Trellis { .. })));
    }

    #[test]
    fn a_wrong_expert_count_is_named_by_layer() {
        let (cfg, mut inv) = pack_or_skip!();
        assert_eq!(drop_prefix(&mut inv, "layers.0.ffn.experts.383."), 12, "one expert: three projections, four tensors");
        let e = bind(&inv, &cfg).unwrap_err();
        assert_eq!(e, BindError::ExpertCount { layer: 0, want: 384, got: 383 });
        assert!(e.to_string().contains("trunk layer 0 has 383 routed experts"), "{e}");

        // a NextN layer's bank is checked against its own count
        let (_, mut inv) = pack().unwrap();
        assert_eq!(drop_prefix(&mut inv, "mtp.2.ffn.experts.127."), 6);
        let e = bind(&inv, &cfg).unwrap_err();
        assert_eq!(e, BindError::MtpExpertCount { layer: 2, want: 128, got: 127 });
    }

    #[test]
    fn a_wrong_layer_count_is_named_before_any_tensor_is_claimed() {
        let (cfg, mut inv) = pack_or_skip!();
        let gone = drop_prefix(&mut inv, "layers.39.");
        assert!(gone > 4_000, "a whole layer: {gone} tensors");
        let e = bind(&inv, &cfg).unwrap_err();
        assert_eq!(e, BindError::LayerCount { want: 40, got: 39 });
        assert!(e.to_string().contains("39 trunk layers"), "{e}");

        // a gap with the right count is a hole, not a count mismatch
        let (_, mut inv) = pack().unwrap();
        let moved: Vec<(String, TensorInfo)> = inv
            .tensors
            .iter()
            .filter(|(k, _)| k.starts_with("layers.39."))
            .map(|(k, v)| (k.replacen("layers.39.", "layers.40.", 1), v.clone()))
            .collect();
        for (k, _) in &moved {
            inv.tensors.remove(&k.replacen("layers.40.", "layers.39.", 1));
        }
        for (k, v) in moved {
            inv.tensors.insert(k, v);
        }
        assert_eq!(bind(&inv, &cfg).unwrap_err(), BindError::LayerHole { index: 39, layers: 40 });

        // and a pack with no NextN layers at all
        let (_, mut inv) = pack().unwrap();
        drop_prefix(&mut inv, "mtp.");
        assert_eq!(bind(&inv, &cfg).unwrap_err(), BindError::MtpCount { want: 3, got: 0 });
    }

    #[test]
    fn a_missing_engram_is_named() {
        let (cfg, mut inv) = pack_or_skip!();
        assert_eq!(drop_prefix(&mut inv, "layers.1.engram."), 6, "the table, its scales, two n-gram taps and wkv");
        let e = bind(&inv, &cfg).unwrap_err();
        assert_eq!(
            e,
            BindError::Missing {
                path: "layers.1.engram.embed.weight".into(),
                role: "an engram table, addressed through an mmap of its shard",
            }
        );
    }

    #[test]
    fn a_tensor_no_slot_takes_is_named() {
        let (cfg, mut inv) = pack_or_skip!();
        inv.tensors.insert(
            "layers.0.attn.q_proj.weight".into(),
            TensorInfo {
                dtype: "BF16".into(),
                shape: vec![5_120, 5_120],
                shard: 1,
                bytes: 52_428_800,
                component: crate::families::deepseek_v41::inventory::Component::MlaQ,
            },
        );
        assert_eq!(bind(&inv, &cfg).unwrap_err(), BindError::Unexpected { path: "layers.0.attn.q_proj.weight".into() });
    }

    #[test]
    fn the_first_mismatch_is_the_one_reported() {
        // two corruptions: the one the walk reaches first is the error, and the result is never partial
        let (cfg, mut inv) = pack_or_skip!();
        inv.tensors.get_mut("layers.0.attn.wq_a.weight").unwrap().dtype = "F16".into();
        inv.tensors.get_mut("layers.9.attn.wq_b.weight").unwrap().dtype = "F16".into();
        assert_eq!(
            bind(&inv, &cfg).unwrap_err(),
            BindError::Dtype { path: "layers.0.attn.wq_a.weight".into(), want: Dtype::Fp8, got: "F16".into() }
        );
    }

    #[test]
    fn a_config_that_cannot_tile_an_fp8_projection_is_refused() {
        // `wo_a` is `[o_groups x o_lora_rank, o_groups x head_dim]`: an `o_lora_rank` that leaves a partial 32-wide tile block
        // has no scale tensor, so the config, not the pack, is what is wrong
        let (_, inv) = pack_or_skip!();
        let mut j: Value = serde_json::from_str(CONFIG).unwrap();
        j["text_config"]["o_lora_rank"] = serde_json::json!(1_023);
        let cfg = DeepseekV41Config::from_json(&j).unwrap();
        match bind(&inv, &cfg) {
            Err(BindError::Config(m)) => assert!(m.contains("scale tiles"), "{m}"),
            other => panic!("expected a config error, got {other:?}"),
        }
    }
}
