//! What a DeepSeek-V4.1 pack holds, as plain data: every tensor's name, dtype, shape, byte count and shard, and which
//! component of the model it belongs to.
//!
//! The inventory is built **from the config**: every name the model must have is generated from `text_config` (layer counts,
//! the two source-layer lists, the expert count, the engram table sizes), given the dtype and shape the layout implies, and
//! then matched against the pack. Three things can go wrong and each is an error that names the tensors:
//!
//! * a tensor the config implies is **missing** from the pack,
//! * a tensor the pack has is **not** implied by the config (a stray, renamed or newly added tensor),
//! * a tensor whose **dtype or shape** is not the one the config implies.
//!
//! Nothing here reads a byte of weights, opens a shard or needs a GPU: it is a pure function of the config and the list of
//! names the pack's headers declare, so a test can hand it a synthetic pack and edit it.
//!
//! Two shape rules are worth stating because they are not obvious from a tensor name:
//!
//! * an FP8 projection is stored as `weight: [n, k]` with block scales `[n / 32, k / 32]` (`F8_E8M0`), for every projection
//!   in the pack — query, KV, output, shared expert, compressor, indexer and the MTP main projection alike;
//! * an EXL3 routed expert is `trellis: [k / 16, n / 16, bits * 16]` with `suh: [k]`, `svh: [n]` and a scalar `mul1`. The bit
//!   width is **per expert**, not per layer: the pack's 46,080 routed experts carry trellises from 1 to 6 bits, so the last
//!   dimension is checked for shape and range rather than pinned to one value.

use super::config::DeepseekV41Config;
use serde_json::Value;
use std::collections::BTreeMap;

/// The engram's fused key/value projection. Its shape is the one tensor extent in the pack that `config.json` does not state:
/// both engram layers carry `[25600, 6144]`, which is `[5 * hidden, 3 * engram_n_heads * engram_head_dim]`. The factor five
/// is not in any config key, so it lives here until the forward says what it is.
const ENGRAM_WKV: (usize, usize) = (25_600, 6_144);
/// Coefficients the mHC hyper-connections keep per residual stream (`hc_*_base` is `[6 * hc_mult]`).
const HC_COEFFS_PER_STREAM: usize = 6;
/// Coefficients of the mHC scale vector.
const HC_SCALE: usize = 3;
/// Hidden rows the first NextN layer's `main_proj` fuses: three trunk-layer taps, not `hc_mult`'s four.
const MAIN_PROJ_TAPS: usize = 3;

const BF16: &str = "BF16";
const F32: &str = "F32";
const F16: &str = "F16";
const I16: &str = "I16";
const I32: &str = "I32";
const I8: &str = "I8";
const FP8: &str = "F8_E4M3";
const E8M0: &str = "F8_E8M0";

/// Which format the NextN (MTP) routed experts are stored in.
///
/// Two uploads of this pack exist and they differ in exactly this: the first stores each NextN expert as an int8 4-bit
/// `weight` `[n, k / 2]` with one `F8_E8M0` `scale` a 32 inputs, the re-upload as an EXL3 `K2` group
/// (`trellis` `[k / 16, n / 16, 32]` + `suh`/`svh`/`mul1`) — the same format the trunk's 46,080 routed experts use. Nothing in
/// `config.json` states it, so it is read off the pack's own headers ([`MtpExpertFormat::detect`]) and every reader of the
/// NextN experts branches on it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MtpExpertFormat {
    /// int8 4-bit: `.weight` `[n, k / 2]` and `.scale` `[n / 32, k / 32]`.
    #[default]
    Int4,
    /// EXL3 `K2` (2.00 bpw): `.trellis` `[k / 16, n / 16, 32]`, `.suh` `[k]`, `.svh` `[n]`, `.mul1` `[]`.
    Exl3,
}

impl MtpExpertFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            MtpExpertFormat::Int4 => "int4",
            MtpExpertFormat::Exl3 => "exl3",
        }
    }

    /// The format a listing declares, from the first `mtp.<k>.ffn.experts.<e>.w1` projection it carries: `.trellis` is an EXL3
    /// group, `.weight` an int8 one. `None` when the pack has no NextN expert at all (a config with no drafter), which is not a
    /// format decision — [`MtpExpertFormat::default`] stands.
    pub fn detect<'a>(names: impl IntoIterator<Item = &'a str>) -> Option<Self> {
        for name in names {
            let Some(rest) = name.strip_prefix("mtp.") else { continue };
            if !rest.contains(".ffn.experts.") {
                continue;
            }
            if name.ends_with(".trellis") {
                return Some(MtpExpertFormat::Exl3);
            }
            if name.ends_with(".weight") {
                return Some(MtpExpertFormat::Int4);
            }
        }
        None
    }
}

impl std::fmt::Display for MtpExpertFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which format the LM head (`head.*`) is stored in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeadFormat {
    /// Unquantised bf16 `head.weight` `[vocab, hidden]`.
    #[default]
    Bf16,
    /// EXL3 group `head.{trellis,mul1,suh,svh}` (`k = hidden, n = vocab`).
    Exl3,
}

impl HeadFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            HeadFormat::Bf16 => "bf16",
            HeadFormat::Exl3 => "exl3",
        }
    }

    pub fn detect<'a>(names: impl IntoIterator<Item = &'a str>) -> Option<Self> {
        for name in names {
            if name == "head.trellis" {
                return Some(HeadFormat::Exl3);
            }
            if name == "head.weight" {
                return Some(HeadFormat::Bf16);
            }
        }
        None
    }
}

impl std::fmt::Display for HeadFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The components a tensor can belong to, in the order the pack lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Component {
    Embed,
    Head,
    Norm,
    MlaQ,
    MlaKv,
    MlaO,
    DsaIndex,
    DsaCompressor,
    MoeRouter,
    MoeShared,
    MoeExperts,
    HcAttn,
    HcFfn,
    Engram,
    Mtp,
    Vision,
}

impl Component {
    pub const ALL: [Component; 16] = [
        Component::Embed,
        Component::Head,
        Component::Norm,
        Component::MlaQ,
        Component::MlaKv,
        Component::MlaO,
        Component::DsaIndex,
        Component::DsaCompressor,
        Component::MoeRouter,
        Component::MoeShared,
        Component::MoeExperts,
        Component::HcAttn,
        Component::HcFfn,
        Component::Engram,
        Component::Mtp,
        Component::Vision,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Component::Embed => "embed",
            Component::Head => "head",
            Component::Norm => "norm",
            Component::MlaQ => "mla_q",
            Component::MlaKv => "mla_kv",
            Component::MlaO => "mla_o",
            Component::DsaIndex => "dsa_index",
            Component::DsaCompressor => "dsa_compressor",
            Component::MoeRouter => "moe_router",
            Component::MoeShared => "moe_shared",
            Component::MoeExperts => "moe_experts",
            Component::HcAttn => "hc_attn",
            Component::HcFfn => "hc_ffn",
            Component::Engram => "engram",
            Component::Mtp => "mtp",
            Component::Vision => "vision",
        }
    }
}

impl std::fmt::Display for Component {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One tensor of a pack: what its header says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub shard: u32,
    pub bytes: u64,
    pub component: Component,
}

/// Every tensor of a pack, by name, with the component it belongs to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PackInventory {
    pub tensors: BTreeMap<String, TensorInfo>,
}

/// Why a pack does not match the layout its config implies. Every variant names the tensors concerned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InventoryError {
    /// The config itself does not describe a buildable layout.
    Config(String),
    /// Tensors the config implies that the pack does not have.
    Missing(Vec<String>),
    /// Tensors the pack has that the config does not imply.
    Unexpected(Vec<String>),
    /// Tensors whose dtype or shape is not the one the config implies.
    Mismatch(Vec<String>),
    /// The same name twice in one listing.
    Duplicate(String),
}

impl std::fmt::Display for InventoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = |what: &str, names: &[String]| {
            let head: Vec<&str> = names.iter().take(4).map(String::as_str).collect();
            let more = names.len().saturating_sub(head.len());
            if more > 0 {
                write!(f, "{what} {} tensors: {} (+{more} more)", names.len(), head.join(", "))
            } else {
                write!(f, "{what} {} tensors: {}", names.len(), head.join(", "))
            }
        };
        match self {
            InventoryError::Config(m) => write!(f, "deepseek_v41: {m}"),
            InventoryError::Missing(n) => list("the pack is missing", n),
            InventoryError::Unexpected(n) => list("the pack has unexpected", n),
            InventoryError::Mismatch(n) => list("the pack disagrees with the config on", n),
            InventoryError::Duplicate(n) => write!(f, "deepseek_v41: `{n}` is listed twice"),
        }
    }
}

impl std::error::Error for InventoryError {}

/// One row of a pack listing, as a scanner would report it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub shard: u32,
    pub bytes: u64,
}

impl Row {
    pub fn new(name: impl Into<String>, dtype: impl Into<String>, shape: Vec<usize>, shard: u32) -> Self {
        let dtype = dtype.into();
        let bytes = shape.iter().product::<usize>() as u64 * elem_bytes(&dtype).unwrap_or(1);
        Self { name: name.into(), dtype, shape, shard, bytes }
    }
}

fn elem_bytes(dtype: &str) -> Option<u64> {
    match dtype {
        "F64" | "I64" | "U64" => Some(8),
        "F32" | "I32" | "U32" => Some(4),
        "BF16" | "F16" | "I16" | "U16" => Some(2),
        "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "I8" | "U8" | "BOOL" => Some(1),
        _ => None,
    }
}

/// The engram's fused key/value projection's pack form, from the listing: plain FP8
/// (`engram.wkv.{weight,scale}`, the base layout) or an EXL3 group
/// (`engram.wkv.{trellis,suh,svh,mul1}`, the quantized packs' form, decoded at boot).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EngramWkvFormat {
    #[default]
    Fp8,
    Exl3,
}

impl EngramWkvFormat {
    /// `(format, both)` from the listing's names: `both` flags a pack that carries the tensor AND
    /// the group, which `from_rows` refuses by name.
    pub fn detect<'a>(names: impl IntoIterator<Item = &'a str>, layer: usize) -> (Self, bool) {
        let (mut plain, mut group) = (false, false);
        for n in names {
            if n == format!("layers.{layer}.engram.wkv.weight") {
                plain = true;
            }
            if n == format!("layers.{layer}.engram.wkv.trellis") {
                group = true;
            }
        }
        let f = if group { Self::Exl3 } else { Self::Fp8 };
        (f, plain && group)
    }
}

/// The shape a tensor must have. Most are fixed by the config; an EXL3 trellis is `[k/16, n/16, bits*16]` with a per-expert
/// bit width.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Shape {
    Exact(Vec<usize>),
    /// `[k / 16, n / 16, bits * 16]`, 1 <= bits <= 8.
    Trellis {
        k: usize,
        n: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Expect {
    dtype: &'static str,
    shape: Shape,
    component: Component,
}

impl Shape {
    fn describe(&self) -> String {
        match self {
            Shape::Exact(dims) => format!("{dims:?}"),
            Shape::Trellis { k, n } => format!("[{}, {}, 16..=128]", k / 16, n / 16),
        }
    }
    fn matches(&self, got: &[usize]) -> bool {
        match self {
            Shape::Exact(dims) => got == dims.as_slice(),
            Shape::Trellis { k, n } => {
                got.len() == 3 && got[0] == k / 16 && got[1] == n / 16 && got[2] % 16 == 0 && (16..=128).contains(&got[2])
            }
        }
    }
}

/// Builds the expected tensor map from a config: name -> dtype, shape, component.
pub struct Expected {
    tensors: BTreeMap<String, Expect>,
}

impl Expected {
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
    /// The component a name belongs to, when the config implies it.
    pub fn component(&self, name: &str) -> Option<Component> {
        self.tensors.get(name).map(|e| e.component)
    }

    /// The expected map as inventory rows, every tensor on `shard`: a listing that matches the config by construction,
    /// for a caller that needs a pack to walk without holding one (the TP seam's CPU tests bind a tree from it) or a
    /// tool writing one. The trellis last dimension is 48 (3 bits): any width the range accepts, priced consistently.
    pub fn rows(&self, shard: u32) -> Vec<Row> {
        self.tensors
            .iter()
            .map(|(name, x)| match &x.shape {
                Shape::Exact(dims) => Row::new(name.clone(), x.dtype, dims.clone(), shard),
                Shape::Trellis { k, n } => Row::new(name.clone(), x.dtype, vec![k / 16, n / 16, 48], shard),
            })
            .collect()
    }
}

impl PackInventory {
    /// The names, dtypes and shapes a pack of `cfg` must have, with its NextN routed experts in `mtp`'s format.
    pub fn expected(cfg: &DeepseekV41Config, mtp: MtpExpertFormat) -> Result<Expected, InventoryError> {
        Self::expected_with(cfg, mtp, HeadFormat::Bf16, EngramWkvFormat::default(), false)
    }

    /// [`Self::expected`] with the LM head's format stated explicitly.
    pub fn expected_with(
        cfg: &DeepseekV41Config,
        mtp: MtpExpertFormat,
        head: HeadFormat,
        wkv: EngramWkvFormat,
        qk_f32: bool,
    ) -> Result<Expected, InventoryError> {
        let mut t: BTreeMap<String, Expect> = BTreeMap::new();
        // every name is generated once; a duplicated generation is a bug in this function, not in the pack
        fn add(
            t: &mut BTreeMap<String, Expect>,
            name: String,
            dtype: &'static str,
            shape: Shape,
            component: Component,
        ) -> Result<(), InventoryError> {
            if t.insert(name.clone(), Expect { dtype, shape, component }).is_some() {
                return Err(InventoryError::Duplicate(name));
            }
            Ok(())
        }

        let hidden = cfg.hidden;
        let q_rank = cfg.q_lora_rank;
        let kv = cfg.kv_heads * cfg.head_dim;
        let o_inner = cfg.o_groups * cfg.head_dim;
        let o_rank = cfg.o_groups * cfg.o_lora_rank;
        let inter = cfg.moe.intermediate;

        // ---- the embedding, the head and the final norm
        add(&mut t, "embed.weight".into(), BF16, Shape::Exact(vec![cfg.vocab, hidden]), Component::Embed)?;
        match head {
            HeadFormat::Bf16 => {
                add(&mut t, "head.weight".into(), BF16, Shape::Exact(vec![cfg.vocab, hidden]), Component::Head)?;
            }
            HeadFormat::Exl3 => {
                add(&mut t, "head.trellis".into(), I16, Shape::Trellis { k: hidden, n: cfg.vocab }, Component::Head)?;
                add(&mut t, "head.mul1".into(), I32, Shape::Exact(vec![]), Component::Head)?;
                add(&mut t, "head.suh".into(), F16, Shape::Exact(vec![hidden]), Component::Head)?;
                add(&mut t, "head.svh".into(), F16, Shape::Exact(vec![cfg.vocab]), Component::Head)?;
            }
        }
        add(&mut t, "norm.weight".into(), BF16, Shape::Exact(vec![hidden]), Component::Norm)?;

        // ---- an FP8 projection: weight [n, k] and its block scales
        let fp8 = |t: &mut BTreeMap<String, Expect>, prefix: &str, n: usize, k: usize, c: Component| {
            t.insert(format!("{prefix}.weight"), Expect { dtype: FP8, shape: Shape::Exact(vec![n, k]), component: c });
            t.insert(format!("{prefix}.scale"), Expect { dtype: E8M0, shape: Shape::Exact(vec![n / 32, k / 32]), component: c });
        };

        for l in 0..cfg.layers {
            let p = format!("layers.{l}");
            // ---- MLA: low-rank query, one shared KV head, grouped low-rank output
            fp8(&mut t, &format!("{p}.attn.wq_a"), q_rank, hidden, Component::MlaQ);
            fp8(&mut t, &format!("{p}.attn.wq_b"), cfg.heads * cfg.head_dim, q_rank, Component::MlaQ);
            fp8(&mut t, &format!("{p}.attn.wkv"), kv, hidden, Component::MlaKv);
            fp8(&mut t, &format!("{p}.attn.wo_a"), o_rank, o_inner, Component::MlaO);
            fp8(&mut t, &format!("{p}.attn.wo_b"), hidden, o_rank, Component::MlaO);
            add(&mut t, format!("{p}.attn.q_norm.weight"), BF16, Shape::Exact(vec![q_rank]), Component::MlaQ)?;
            add(&mut t, format!("{p}.attn.kv_norm.weight"), BF16, Shape::Exact(vec![kv]), Component::MlaKv)?;
            add(&mut t, format!("{p}.attn.attn_sink"), F32, Shape::Exact(vec![cfg.heads]), Component::MlaO)?;

            // ---- the two norms a layer owns
            add(&mut t, format!("{p}.attn_norm.weight"), BF16, Shape::Exact(vec![hidden]), Component::Norm)?;
            add(&mut t, format!("{p}.ffn_norm.weight"), BF16, Shape::Exact(vec![hidden]), Component::Norm)?;

            // ---- the router and its shared expert
            add(
                &mut t,
                format!("{p}.ffn.gate.weight"),
                BF16,
                Shape::Exact(vec![cfg.moe.routed_experts, hidden]),
                Component::MoeRouter,
            )?;
            add(&mut t, format!("{p}.ffn.gate.bias"), F32, Shape::Exact(vec![cfg.moe.routed_experts]), Component::MoeRouter)?;
            add(&mut t, format!("{p}.ffn.gate.bias_vl"), F32, Shape::Exact(vec![cfg.moe.routed_experts]), Component::MoeRouter)?;
            fp8(&mut t, &format!("{p}.ffn.shared_experts.w1"), inter, hidden, Component::MoeShared);
            fp8(&mut t, &format!("{p}.ffn.shared_experts.w2"), hidden, inter, Component::MoeShared);
            fp8(&mut t, &format!("{p}.ffn.shared_experts.w3"), inter, hidden, Component::MoeShared);

            // ---- the routed experts: EXL3 trellis, per-expert bit width
            for e in 0..cfg.moe.routed_experts {
                for (proj, n, k) in [("w1", inter, hidden), ("w2", hidden, inter), ("w3", inter, hidden)] {
                    let x = format!("{p}.ffn.experts.{e}.{proj}");
                    add(&mut t, format!("{x}.trellis"), I16, Shape::Trellis { k, n }, Component::MoeExperts)?;
                    add(&mut t, format!("{x}.mul1"), I32, Shape::Exact(vec![]), Component::MoeExperts)?;
                    add(&mut t, format!("{x}.suh"), F16, Shape::Exact(vec![k]), Component::MoeExperts)?;
                    add(&mut t, format!("{x}.svh"), F16, Shape::Exact(vec![n]), Component::MoeExperts)?;
                }
            }

            // ---- mHC hyper-connections, one set for the mixer and one for the FFN
            for (which, c) in [("attn", Component::HcAttn), ("ffn", Component::HcFfn)] {
                add(&mut t, format!("{p}.hc_{which}_base"), F32, Shape::Exact(vec![HC_COEFFS_PER_STREAM * cfg.hc.mult]), c)?;
                add(
                    &mut t,
                    format!("{p}.hc_{which}_fn"),
                    F32,
                    Shape::Exact(vec![HC_COEFFS_PER_STREAM * cfg.hc.mult, cfg.hc.mult * hidden]),
                    c,
                )?;
                add(&mut t, format!("{p}.hc_{which}_scale"), F32, Shape::Exact(vec![HC_SCALE]), c)?;
            }

            // ---- the compressor: only on the layers that own one, and its gate only where an entry spans several tokens
            // the compressor is bf16, not the FP8 block format the dense projections use: it has no scales in the pack
            if cfg.compress.is_source(l) {
                add(
                    &mut t,
                    format!("{p}.attn.compressor.wkv.weight"),
                    BF16,
                    Shape::Exact(vec![cfg.head_dim, hidden]),
                    Component::DsaCompressor,
                )?;
                add(
                    &mut t,
                    format!("{p}.attn.compressor.norm.weight"),
                    BF16,
                    Shape::Exact(vec![cfg.head_dim]),
                    Component::DsaCompressor,
                )?;
                if cfg.compress.has_gate(l) {
                    add(
                        &mut t,
                        format!("{p}.attn.compressor.wgate.weight"),
                        BF16,
                        Shape::Exact(vec![cfg.head_dim, hidden]),
                        Component::DsaCompressor,
                    )?;
                }
            }

            // ---- the indexer: queries on every selector, keys only on the layers that own a pool
            if cfg.indexer.is_source(l) {
                fp8(
                    &mut t,
                    &format!("{p}.attn.indexer.wq_b"),
                    cfg.indexer.n_heads * cfg.indexer.head_dim,
                    q_rank,
                    Component::DsaIndex,
                );
                add(
                    &mut t,
                    format!("{p}.attn.indexer.weights_proj.weight"),
                    BF16,
                    Shape::Exact(vec![cfg.indexer.n_heads, hidden]),
                    Component::DsaIndex,
                )?;
                if cfg.compress.is_source(l) {
                    add(
                        &mut t,
                        format!("{p}.attn.indexer.wk.weight"),
                        BF16,
                        Shape::Exact(vec![cfg.indexer.head_dim, cfg.head_dim]),
                        Component::DsaIndex,
                    )?;
                    add(
                        &mut t,
                        format!("{p}.attn.indexer.k_norm.weight"),
                        BF16,
                        Shape::Exact(vec![cfg.indexer.head_dim]),
                        Component::DsaIndex,
                    )?;
                }
            }

            // ---- the engram table, on the layers that carry one
            if let Some(table) = cfg.engram.table(l) {
                add(
                    &mut t,
                    format!("{p}.engram.embed.weight"),
                    FP8,
                    Shape::Exact(vec![table, cfg.engram.head_dim]),
                    Component::Engram,
                )?;
                add(
                    &mut t,
                    format!("{p}.engram.embed.scale"),
                    E8M0,
                    Shape::Exact(vec![table, cfg.engram.head_dim / 32]),
                    Component::Engram,
                )?;
                add(
                    &mut t,
                    format!("{p}.engram.q_weight"),
                    if qk_f32 { F32 } else { BF16 },
                    Shape::Exact(vec![cfg.engram.max_ngram_size, hidden]),
                    Component::Engram,
                )?;
                add(
                    &mut t,
                    format!("{p}.engram.k_weight"),
                    if qk_f32 { F32 } else { BF16 },
                    Shape::Exact(vec![cfg.engram.max_ngram_size, hidden]),
                    Component::Engram,
                )?;
                if wkv == EngramWkvFormat::Exl3 {
                    // k is the input width (6144 = 3 x 8 x 256), n the output (25600 = 5 x hidden): the
                    // group the engine decodes at boot (`Fp8Matrix::from_group`), [k/16, n/16, 16 x bits]
                    add(
                        &mut t,
                        format!("{p}.engram.wkv.trellis"),
                        I16,
                        Shape::Trellis { k: ENGRAM_WKV.1, n: ENGRAM_WKV.0 },
                        Component::Engram,
                    )?;
                    add(&mut t, format!("{p}.engram.wkv.suh"), F16, Shape::Exact(vec![ENGRAM_WKV.1]), Component::Engram)?;
                    add(&mut t, format!("{p}.engram.wkv.svh"), F16, Shape::Exact(vec![ENGRAM_WKV.0]), Component::Engram)?;
                    add(&mut t, format!("{p}.engram.wkv.mul1"), I32, Shape::Exact(vec![]), Component::Engram)?;
                } else {
                    add(
                        &mut t,
                        format!("{p}.engram.wkv.weight"),
                        FP8,
                        Shape::Exact(vec![ENGRAM_WKV.0, ENGRAM_WKV.1]),
                        Component::Engram,
                    )?;
                    add(
                        &mut t,
                        format!("{p}.engram.wkv.scale"),
                        E8M0,
                        Shape::Exact(vec![ENGRAM_WKV.0 / 32, ENGRAM_WKV.1 / 32]),
                        Component::Engram,
                    )?;
                }
            }
        }

        // ---- the NextN layers: the same MLA + mHC block, an int8 4-bit expert per projection, and the DSpark heads
        for k in 0..cfg.nextn {
            let p = format!("mtp.{k}");
            fp8(&mut t, &format!("{p}.attn.wq_a"), q_rank, hidden, Component::Mtp);
            fp8(&mut t, &format!("{p}.attn.wq_b"), cfg.heads * cfg.head_dim, q_rank, Component::Mtp);
            fp8(&mut t, &format!("{p}.attn.wkv"), kv, hidden, Component::Mtp);
            fp8(&mut t, &format!("{p}.attn.wo_a"), o_rank, o_inner, Component::Mtp);
            fp8(&mut t, &format!("{p}.attn.wo_b"), hidden, o_rank, Component::Mtp);
            add(&mut t, format!("{p}.attn.q_norm.weight"), BF16, Shape::Exact(vec![q_rank]), Component::Mtp)?;
            add(&mut t, format!("{p}.attn.kv_norm.weight"), BF16, Shape::Exact(vec![kv]), Component::Mtp)?;
            add(&mut t, format!("{p}.attn.attn_sink"), F32, Shape::Exact(vec![cfg.heads]), Component::Mtp)?;
            add(&mut t, format!("{p}.attn_norm.weight"), BF16, Shape::Exact(vec![hidden]), Component::Mtp)?;
            add(&mut t, format!("{p}.ffn_norm.weight"), BF16, Shape::Exact(vec![hidden]), Component::Mtp)?;

            let experts = cfg.dspark.n_routed_experts;
            add(&mut t, format!("{p}.ffn.gate.weight"), BF16, Shape::Exact(vec![experts, hidden]), Component::Mtp)?;
            add(&mut t, format!("{p}.ffn.gate.bias"), F32, Shape::Exact(vec![experts]), Component::Mtp)?;
            add(&mut t, format!("{p}.ffn.gate.bias_vl"), F32, Shape::Exact(vec![experts]), Component::Mtp)?;
            fp8(&mut t, &format!("{p}.ffn.shared_experts.w1"), inter, hidden, Component::Mtp);
            fp8(&mut t, &format!("{p}.ffn.shared_experts.w2"), hidden, inter, Component::Mtp);
            fp8(&mut t, &format!("{p}.ffn.shared_experts.w3"), inter, hidden, Component::Mtp);
            // an int8 expert: the weight is packed two values to a byte, the scale is one per 32 inputs; an EXL3 expert is the
            // trunk's group format at a fixed 2 bits (the re-upload)
            for e in 0..experts {
                for (proj, n, k) in [("w1", inter, hidden), ("w2", hidden, inter), ("w3", inter, hidden)] {
                    let x = format!("{p}.ffn.experts.{e}.{proj}");
                    match mtp {
                        MtpExpertFormat::Int4 => {
                            t.insert(
                                format!("{x}.weight"),
                                Expect { dtype: I8, shape: Shape::Exact(vec![n, k / 2]), component: Component::Mtp },
                            );
                            t.insert(
                                format!("{x}.scale"),
                                Expect { dtype: E8M0, shape: Shape::Exact(vec![n, k / 32]), component: Component::Mtp },
                            );
                        }
                        MtpExpertFormat::Exl3 => {
                            add(&mut t, format!("{x}.trellis"), I16, Shape::Trellis { k, n }, Component::Mtp)?;
                            add(&mut t, format!("{x}.mul1"), I32, Shape::Exact(vec![]), Component::Mtp)?;
                            add(&mut t, format!("{x}.suh"), F16, Shape::Exact(vec![k]), Component::Mtp)?;
                            add(&mut t, format!("{x}.svh"), F16, Shape::Exact(vec![n]), Component::Mtp)?;
                        }
                    }
                }
            }
            for (which, c) in [("attn", Component::Mtp), ("ffn", Component::Mtp)] {
                add(&mut t, format!("{p}.hc_{which}_base"), F32, Shape::Exact(vec![HC_COEFFS_PER_STREAM * cfg.hc.mult]), c)?;
                add(
                    &mut t,
                    format!("{p}.hc_{which}_fn"),
                    F32,
                    Shape::Exact(vec![HC_COEFFS_PER_STREAM * cfg.hc.mult, cfg.hc.mult * hidden]),
                    c,
                )?;
                add(&mut t, format!("{p}.hc_{which}_scale"), F32, Shape::Exact(vec![HC_SCALE]), c)?;
            }
            // the tapped hidden rows the first NextN layer fuses, and the last one's Markov and confidence heads
            if k == 0 {
                fp8(&mut t, &format!("{p}.main_proj"), hidden, MAIN_PROJ_TAPS * hidden, Component::Mtp);
                add(&mut t, format!("{p}.main_norm.weight"), BF16, Shape::Exact(vec![hidden]), Component::Mtp)?;
            }
            if k + 1 == cfg.nextn {
                add(
                    &mut t,
                    format!("{p}.markov_head.embed.weight"),
                    BF16,
                    Shape::Exact(vec![cfg.vocab, cfg.dspark.markov_rank]),
                    Component::Mtp,
                )?;
                add(
                    &mut t,
                    format!("{p}.markov_head.head.weight"),
                    BF16,
                    Shape::Exact(vec![cfg.vocab, cfg.dspark.markov_rank]),
                    Component::Mtp,
                )?;
                add(
                    &mut t,
                    format!("{p}.confidence_head.proj.weight"),
                    BF16,
                    Shape::Exact(vec![1, hidden + cfg.dspark.markov_rank]),
                    Component::Mtp,
                )?;
                add(&mut t, format!("{p}.norm.weight"), BF16, Shape::Exact(vec![hidden]), Component::Mtp)?;
            }
        }

        // ---- the vision tower and the projector that splices its features into the text stream
        if let Some(v) = &cfg.vision {
            let (h, i, m2) = (v.hidden, v.intermediate, v.downsample_ratio * v.downsample_ratio);
            add(&mut t, "vision.patch_embed.proj.weight".into(), BF16, Shape::Exact(vec![h, v.patch_dim()]), Component::Vision)?;
            add(&mut t, "vision.patch_embed.proj.bias".into(), BF16, Shape::Exact(vec![h]), Component::Vision)?;
            for b in 0..v.layers {
                let p = format!("vision.blocks.{b}");
                add(&mut t, format!("{p}.attn.wqkv.weight"), BF16, Shape::Exact(vec![3 * h, h]), Component::Vision)?;
                add(&mut t, format!("{p}.attn.wqkv.bias"), BF16, Shape::Exact(vec![3 * h]), Component::Vision)?;
                add(&mut t, format!("{p}.attn.wo.weight"), BF16, Shape::Exact(vec![h, h]), Component::Vision)?;
                add(&mut t, format!("{p}.attn.wo.bias"), BF16, Shape::Exact(vec![h]), Component::Vision)?;
                add(&mut t, format!("{p}.mlp.w1.weight"), BF16, Shape::Exact(vec![2 * i, h]), Component::Vision)?;
                add(&mut t, format!("{p}.mlp.w2.weight"), BF16, Shape::Exact(vec![h, i]), Component::Vision)?;
                add(&mut t, format!("{p}.norm1.weight"), BF16, Shape::Exact(vec![h]), Component::Vision)?;
                add(&mut t, format!("{p}.norm2.weight"), BF16, Shape::Exact(vec![h]), Component::Vision)?;
            }
            add(&mut t, "vision.norm.weight".into(), BF16, Shape::Exact(vec![h]), Component::Vision)?;
            add(&mut t, "aligner.w1.weight".into(), BF16, Shape::Exact(vec![hidden, h * m2]), Component::Vision)?;
            add(&mut t, "aligner.w1.bias".into(), BF16, Shape::Exact(vec![hidden]), Component::Vision)?;
            add(&mut t, "aligner.w2.weight".into(), BF16, Shape::Exact(vec![hidden, hidden]), Component::Vision)?;
            add(&mut t, "aligner.w2.bias".into(), BF16, Shape::Exact(vec![hidden]), Component::Vision)?;
            // three learned rows the prompt splices around an image's features
            for t_ in ["image_start", "image_end", "image_newline"] {
                add(&mut t, t_.to_string(), BF16, Shape::Exact(vec![hidden]), Component::Vision)?;
            }
        }

        Ok(Expected { tensors: t })
    }

    /// Check a pack listing against the config and classify every tensor. The NextN experts' format is read from the listing
    /// itself ([`MtpExpertFormat::detect`]), so one reader serves both uploads.
    pub fn from_rows(cfg: &DeepseekV41Config, rows: impl IntoIterator<Item = Row>) -> Result<Self, InventoryError> {
        let rows: Vec<Row> = rows.into_iter().collect();
        let mtp = MtpExpertFormat::detect(rows.iter().map(|r| r.name.as_str())).unwrap_or_default();
        let head = HeadFormat::detect(rows.iter().map(|r| r.name.as_str())).unwrap_or_default();
        // the engram wkv's form is a whole-pack fact (every engram layer the same), and the
        // n-gram projections' dtype likewise: bf16 in the base layout, f32 in the quantized packs
        let (wkv, qk_f32) = if cfg.engram.layer_ids.is_empty() {
            (EngramWkvFormat::default(), false)
        } else {
            let names = rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>();
            let mut wkv = None;
            let mut qk_f32 = false;
            let mut mixed = None;
            for &layer in &cfg.engram.layer_ids {
                let (f, both) = EngramWkvFormat::detect(names.iter().copied(), layer);
                if both {
                    mixed = Some(layer);
                }
                wkv = Some(match wkv {
                    None => f,
                    Some(prev) if prev == f => f,
                    Some(prev) => {
                        return Err(InventoryError::Config(format!(
                            "layers carry the engram wkv in different forms ({prev:?} vs {f:?})"
                        )))
                    }
                });
            }
            if let Some(layer) = mixed {
                return Err(InventoryError::Config(format!(
                    "layers.{layer}.engram.wkv is present as BOTH a plain FP8 tensor and an EXL3 group"
                )));
            }
            if rows.iter().any(|r| r.name.ends_with(".engram.q_weight") && r.dtype == F32) {
                qk_f32 = true;
            }
            (wkv.unwrap_or_default(), qk_f32)
        };
        let mut expected = Self::expected_with(cfg, mtp, head, wkv, qk_f32)?;
        // Upgrade any FP8 trunk projection (`attn.wq_b`, `attn.wo_b`, `ffn.shared_experts.{w1,w2,w3}`)
        // that the pack stores as an EXL3 group (`{prefix}.{trellis,mul1,suh,svh}`).
        let exl3_prefixes: Vec<String> =
            rows.iter().filter_map(|r| r.name.strip_suffix(".trellis").map(str::to_string)).collect();
        for prefix in exl3_prefixes {
            let w_key = format!("{prefix}.weight");
            let s_key = format!("{prefix}.scale");
            if let Some(exp_w) = expected.tensors.get(&w_key).cloned() {
                if let Shape::Exact(dims) = exp_w.shape {
                    if dims.len() == 2 && expected.tensors.contains_key(&s_key) {
                        let (n, k) = (dims[0], dims[1]);
                        let c = exp_w.component;
                        expected.tensors.remove(&w_key);
                        expected.tensors.remove(&s_key);
                        expected.tensors.insert(
                            format!("{prefix}.trellis"),
                            Expect { dtype: I16, shape: Shape::Trellis { k, n }, component: c },
                        );
                        expected
                            .tensors
                            .insert(format!("{prefix}.mul1"), Expect { dtype: I32, shape: Shape::Exact(vec![]), component: c });
                        expected
                            .tensors
                            .insert(format!("{prefix}.suh"), Expect { dtype: F16, shape: Shape::Exact(vec![k]), component: c });
                        expected
                            .tensors
                            .insert(format!("{prefix}.svh"), Expect { dtype: F16, shape: Shape::Exact(vec![n]), component: c });
                    }
                }
            }
        }
        let mut tensors: BTreeMap<String, TensorInfo> = BTreeMap::new();
        let mut unexpected = Vec::new();
        let mut mismatch = Vec::new();
        let mut duplicate = None;
        for row in rows {
            let Some(e) = expected.tensors.get(&row.name) else {
                unexpected.push(row.name);
                continue;
            };
            if row.dtype != e.dtype || !e.shape.matches(&row.shape) {
                mismatch.push(format!(
                    "`{}`: expected {} {}, found {} {}",
                    row.name,
                    e.dtype,
                    e.shape.describe(),
                    row.dtype,
                    format_args!("{:?}", row.shape)
                ));
                continue;
            }
            let info =
                TensorInfo { dtype: row.dtype, shape: row.shape, shard: row.shard, bytes: row.bytes, component: e.component };
            if tensors.insert(row.name.clone(), info).is_some() {
                duplicate.get_or_insert(row.name);
            }
        }
        if let Some(name) = duplicate {
            return Err(InventoryError::Duplicate(name));
        }
        let missing: Vec<String> = expected.tensors.keys().filter(|n| !tensors.contains_key(*n)).cloned().collect();
        // every error at once, so one run says everything that is wrong with a pack
        let mut problems = Vec::new();
        if !missing.is_empty() {
            problems.push(InventoryError::Missing(missing));
        }
        if !unexpected.is_empty() {
            problems.push(InventoryError::Unexpected(unexpected));
        }
        if !mismatch.is_empty() {
            problems.push(InventoryError::Mismatch(mismatch));
        }
        match problems.len() {
            0 => Ok(Self { tensors }),
            1 => Err(problems.pop().unwrap()),
            _ => {
                // the first problem is the one reported; the rest are folded into its text
                let mut parts: Vec<String> = problems.iter().map(|p| p.to_string()).collect();
                parts.sort();
                Err(InventoryError::Config(parts.join("; ")))
            }
        }
    }

    /// Check a pack listing in the shape `tools/dsv41_inventory.json` writes: an object from name to
    /// `{ "shard": N, "dtype": "F32", "shape": [..], "bytes": N }`.
    pub fn from_json(cfg: &DeepseekV41Config, listing: &Value) -> Result<Self, InventoryError> {
        let obj = listing.as_object().ok_or_else(|| InventoryError::Config("the listing must be an object".into()))?;
        let mut rows = Vec::with_capacity(obj.len());
        for (name, v) in obj {
            let dtype = v
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| InventoryError::Config(format!("`{name}`: no dtype")))?
                .to_string();
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| InventoryError::Config(format!("`{name}`: no shape")))?
                .iter()
                .map(|d| d.as_u64().unwrap_or(0) as usize)
                .collect();
            let shard = v.get("shard").and_then(Value::as_u64).unwrap_or(1) as u32;
            let bytes = v.get("bytes").and_then(Value::as_u64).unwrap_or(0);
            rows.push(Row { name: name.clone(), dtype, shape, shard, bytes });
        }
        Self::from_rows(cfg, rows)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Tensors of one component.
    pub fn of(&self, component: Component) -> impl Iterator<Item = (&String, &TensorInfo)> {
        self.tensors.iter().filter(move |(_, t)| t.component == component)
    }

    /// Which format this pack stores its NextN routed experts in, read from the names it holds.
    pub fn mtp_experts(&self) -> MtpExpertFormat {
        MtpExpertFormat::detect(self.tensors.keys().map(String::as_str)).unwrap_or_default()
    }

    /// Which format this pack stores its LM head in, read from the names it holds.
    pub fn head_format(&self) -> HeadFormat {
        HeadFormat::detect(self.tensors.keys().map(String::as_str)).unwrap_or_default()
    }

    /// How many tensors each component has, in component order.
    pub fn counts(&self) -> Vec<(Component, usize)> {
        Component::ALL.iter().map(|&c| (c, self.of(c).count())).collect()
    }

    /// Bytes each component occupies.
    pub fn bytes(&self) -> u64 {
        self.tensors.values().map(|t| t.bytes).sum()
    }

    /// The shards the pack is split over.
    pub fn shards(&self) -> usize {
        let mut s: Vec<u32> = self.tensors.values().map(|t| t.shard).collect();
        s.sort_unstable();
        s.dedup();
        s.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::deepseek_v41::config::DeepseekV41Config;

    const CONFIG: &str = include_str!("../../../../../testdata/deepseek_v41/config.json");
    /// The complete tensor listing of the real 17-shard pack (188,245 tensors), when this machine has it.
    const PACK: &str = "/tmp/dsv41_inventory.json";

    fn real() -> DeepseekV41Config {
        DeepseekV41Config::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
    }

    /// A tiny pack: two trunk layers, two experts, no NextN layer, no engram, no vision. Every shape is the real layout's,
    /// shrunk — so a test exercises the same code path the 188,245-tensor pack does without holding 310 GiB.
    fn toy() -> DeepseekV41Config {
        let mut v: Value = serde_json::from_str(CONFIG).unwrap();
        let t = v["text_config"].as_object_mut().unwrap();
        t.insert("num_hidden_layers".into(), 2.into());
        t.insert("num_nextn_predict_layers".into(), 0.into());
        t.insert("n_routed_experts".into(), 2.into());
        t.insert("num_experts_per_tok".into(), 2.into());
        t.insert("compress_ratios".into(), serde_json::json!([2, 2]));
        t.insert("kv_source_layer_ids".into(), serde_json::json!([0]));
        t.insert("index_source_layer_ids".into(), serde_json::json!([0, 1]));
        t.insert("engram_layer_ids".into(), serde_json::json!([]));
        t.insert("engram_num_embeddings".into(), serde_json::json!([]));
        t.insert("dspark_target_layer_ids".into(), serde_json::json!([]));
        t.insert("q_lora_rank".into(), 64.into());
        t.insert("o_lora_rank".into(), 32.into());
        t.insert("hidden_size".into(), 512.into());
        t.insert("moe_intermediate_size".into(), 128.into());
        t.insert("vocab_size".into(), 131_072.into());
        t.insert("head_dim".into(), 64.into());
        t.insert("qk_rope_head_dim".into(), 8.into());
        t.insert("index_head_dim".into(), 16.into());
        t.insert("index_n_heads".into(), 4.into());
        t.insert("dspark_n_routed_experts".into(), 2.into());
        t.insert("dspark_markov_rank".into(), 16.into());
        v.as_object_mut().unwrap().remove("vision_config");
        DeepseekV41Config::from_json(&v).unwrap()
    }

    /// Every tensor the toy config implies, as a listing, with its NextN experts in `mtp`'s format.
    fn toy_rows(cfg: &DeepseekV41Config) -> Vec<Row> {
        Expected::build_rows(cfg, MtpExpertFormat::Int4)
    }

    /// Turn the expected map into rows, so a test starts from a pack that matches exactly.
    impl Expected {
        fn build_rows(cfg: &DeepseekV41Config, mtp: MtpExpertFormat) -> Vec<Row> {
            PackInventory::expected(cfg, mtp).unwrap().rows(1)
        }
    }

    #[test]
    fn a_pack_that_matches_its_config_maps_completely() {
        let cfg = toy();
        let rows = toy_rows(&cfg);
        let inv = PackInventory::from_rows(&cfg, rows.clone()).unwrap();
        assert_eq!(inv.len(), rows.len());
        // 2 layers x (2 experts x 3 projections x 4 tensors + 5 MLA projections x 2 + ...) : every component accounted for
        let counts: BTreeMap<&str, usize> = inv.counts().into_iter().map(|(c, n)| (c.as_str(), n)).collect();
        assert_eq!(counts["moe_experts"], 2 * 2 * 3 * 4);
        assert_eq!(counts["mla_q"], 2 * 5, "wq_a and wq_b, each weight and scale, plus q_norm");
        assert_eq!(counts["mla_kv"], 2 * 3, "wkv, its scale and kv_norm");
        assert_eq!(counts["mla_o"], 2 * 5, "wo_a and wo_b, each weight and scale, plus the sink");
        assert_eq!(counts["dsa_compressor"], 3, "one compressor: wkv, its gate and the norm");
        assert_eq!(counts["dsa_index"], 2 + 2 + 2 + 2, "two selectors, one of them owning a key and its norm");
        assert_eq!(counts["hc_attn"] + counts["hc_ffn"], 2 * 6);
        assert_eq!(counts["engram"], 0);
        assert_eq!(counts["vision"], 0);
        assert_eq!(counts["embed"] + counts["head"] + counts["norm"], 1 + 1 + 1 + 2 * 2);
        assert_eq!(inv.shards(), 1);
        assert!(inv.bytes() > 0);
    }

    #[test]
    fn a_missing_tensor_is_named() {
        let cfg = toy();
        let mut rows = toy_rows(&cfg);
        let gone = "layers.1.ffn.experts.1.w2.trellis".to_string();
        rows.retain(|r| r.name != gone);
        let e = PackInventory::from_rows(&cfg, rows).unwrap_err();
        assert!(matches!(e, InventoryError::Missing(ref n) if n == &vec![gone.clone()]), "{e}");
        assert!(e.to_string().contains(&gone), "{e}");
    }

    #[test]
    fn an_unexpected_tensor_is_named() {
        let cfg = toy();
        let mut rows = toy_rows(&cfg);
        rows.push(Row::new("layers.0.attn.q_proj.weight", "BF16", vec![512, 512], 1));
        let e = PackInventory::from_rows(&cfg, rows).unwrap_err();
        assert!(matches!(e, InventoryError::Unexpected(ref n) if n == &vec!["layers.0.attn.q_proj.weight".to_string()]), "{e}");
        // a tensor of a layer past the trunk is just as unexpected as a misspelt one
        let mut rows = toy_rows(&cfg);
        rows.push(Row::new("layers.2.attn_norm.weight", "BF16", vec![512], 1));
        let e = PackInventory::from_rows(&cfg, rows).unwrap_err();
        assert!(e.to_string().contains("layers.2.attn_norm.weight"), "{e}");
    }

    #[test]
    fn a_wrong_shape_or_dtype_is_named_with_both() {
        let cfg = toy();
        // the compressed KV is 64 wide in this toy config, not 128
        let mut rows = toy_rows(&cfg);
        rows.iter_mut().find(|r| r.name == "layers.0.attn.compressor.wkv.weight").unwrap().shape = vec![128, 512];
        let e = PackInventory::from_rows(&cfg, rows).unwrap_err();
        let text = e.to_string();
        assert!(text.contains("layers.0.attn.compressor.wkv.weight"), "{text}");
        assert!(text.contains("expected BF16 [64, 512]"), "{text}");

        let mut rows = toy_rows(&cfg);
        rows.iter_mut().find(|r| r.name == "layers.0.attn.wkv.weight").unwrap().dtype = "BF16".into();
        let e = PackInventory::from_rows(&cfg, rows).unwrap_err();
        assert!(e.to_string().contains("expected F8_E4M3"), "{e}");

        // an expert trellis whose last dimension is not a whole number of 16-bit steps
        let mut rows = toy_rows(&cfg);
        rows.iter_mut().find(|r| r.name == "layers.0.ffn.experts.0.w1.trellis").unwrap().shape = vec![32, 8, 40];
        assert!(PackInventory::from_rows(&cfg, rows).is_err());
    }

    #[test]
    fn the_expert_trellis_accepts_every_bit_width_the_pack_uses() {
        let cfg = toy();
        for last in [16usize, 32, 48, 64, 80, 96] {
            let mut rows = toy_rows(&cfg);
            for r in rows.iter_mut().filter(|r| r.name.ends_with(".w1.trellis")) {
                r.shape = vec![512 / 16, 128 / 16, last];
            }
            assert!(PackInventory::from_rows(&cfg, rows).is_ok(), "trellis last dim {last}");
        }
        // 0 bits or more than 8 are not widths the format has
        for last in [0usize, 144, 160] {
            let mut rows = toy_rows(&cfg);
            for r in rows.iter_mut().filter(|r| r.name.ends_with(".w1.trellis")) {
                r.shape = vec![512 / 16, 128 / 16, last];
            }
            assert!(PackInventory::from_rows(&cfg, rows).is_err(), "trellis last dim {last}");
        }
    }

    /// The toy config with one NextN layer, so the drafter's experts are in play: the toy has none.
    fn toy_drafter() -> DeepseekV41Config {
        let mut v: Value = serde_json::from_str(CONFIG).unwrap();
        let t = v["text_config"].as_object_mut().unwrap();
        t.insert("num_hidden_layers".into(), 2.into());
        t.insert("num_nextn_predict_layers".into(), 1.into());
        t.insert("n_routed_experts".into(), 2.into());
        t.insert("num_experts_per_tok".into(), 2.into());
        t.insert("compress_ratios".into(), serde_json::json!([2, 2, 2]));
        t.insert("kv_source_layer_ids".into(), serde_json::json!([0]));
        t.insert("index_source_layer_ids".into(), serde_json::json!([0, 1]));
        t.insert("engram_layer_ids".into(), serde_json::json!([]));
        t.insert("engram_num_embeddings".into(), serde_json::json!([]));
        t.insert("dspark_target_layer_ids".into(), serde_json::json!([1]));
        t.insert("q_lora_rank".into(), 64.into());
        t.insert("o_lora_rank".into(), 32.into());
        t.insert("hidden_size".into(), 512.into());
        t.insert("moe_intermediate_size".into(), 128.into());
        t.insert("vocab_size".into(), 131_072.into());
        t.insert("head_dim".into(), 64.into());
        t.insert("qk_rope_head_dim".into(), 8.into());
        t.insert("index_head_dim".into(), 16.into());
        t.insert("index_n_heads".into(), 4.into());
        t.insert("dspark_n_routed_experts".into(), 2.into());
        t.insert("dspark_markov_rank".into(), 16.into());
        v.as_object_mut().unwrap().remove("vision_config");
        DeepseekV41Config::from_json(&v).unwrap()
    }

    /// **Both uploads inventory.** The NextN experts are int8 4-bit in the first pack and EXL3 K2 in the re-upload; nothing in
    /// the config says which, so the reader detects it from the names and accepts exactly the format it found.
    #[test]
    fn both_nextn_expert_formats_are_detected_and_accepted() {
        let cfg = toy_drafter();
        for fmt in [MtpExpertFormat::Int4, MtpExpertFormat::Exl3] {
            let rows = Expected::build_rows(&cfg, fmt);
            let inv = PackInventory::from_rows(&cfg, rows.clone()).unwrap_or_else(|e| panic!("{fmt}: {e}"));
            assert_eq!(inv.mtp_experts(), fmt, "the format is read off the headers");
            assert_eq!(inv.len(), rows.len(), "{fmt}: every tensor is classified");
            let mtp = inv.of(Component::Mtp).count();
            // 2 experts x 3 projections x (2 int8 parts | 4 EXL3 parts) + the block's own 37 tensors
            assert_eq!(mtp, 2 * 3 * if fmt == MtpExpertFormat::Exl3 { 4 } else { 2 } + 37, "{fmt}: {mtp} NextN tensors");
        }
        // the two formats are the same matrix in different bytes: the EXL3 expert is smaller and its parts are the trunk's
        let int4 = PackInventory::from_rows(&cfg, Expected::build_rows(&cfg, MtpExpertFormat::Int4)).unwrap();
        let exl3 = PackInventory::from_rows(&cfg, Expected::build_rows(&cfg, MtpExpertFormat::Exl3)).unwrap();
        let bytes = |i: &PackInventory| -> u64 {
            i.tensors.iter().filter(|(n, _)| n.contains(".ffn.experts.") && n.starts_with("mtp.")).map(|(_, t)| t.bytes).sum()
        };
        assert!(bytes(&exl3) < bytes(&int4), "EXL3 {} B, int8 {} B", bytes(&exl3), bytes(&int4));
        // a listing that carries both shapes for one expert is not a pack of either format
        let mut mixed = Expected::build_rows(&cfg, MtpExpertFormat::Int4);
        mixed.extend(Expected::build_rows(&cfg, MtpExpertFormat::Exl3));
        assert!(PackInventory::from_rows(&cfg, mixed).is_err(), "a mixed-format pack must not inventory");
    }

    /// **The re-upload's own headers.** The twelve parts of one NextN expert, exactly as
    /// `vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw` (revision `d4959c8`) declares them: `K2` trellises and the fp16 scales.
    /// 8,891,916 B an expert, 384 experts, 3,414,495,744 B — 3.180 GiB, against the first upload's 7,219,445,760 B.
    #[test]
    fn the_reuploads_nextn_experts_are_the_ones_the_reader_expects() {
        const HEADERS: &str = include_str!("../../../../../testdata/deepseek_v41/mtp_experts_exl3.json");
        let real = DeepseekV41Config::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap();
        let want = PackInventory::expected(&real, MtpExpertFormat::Exl3).unwrap();
        let declared: BTreeMap<String, Value> = serde_json::from_str(HEADERS).unwrap();
        assert_eq!(declared.len(), 12, "three projections x four parts");
        let mut one = 0u64;
        for (name, v) in &declared {
            let dtype = v["dtype"].as_str().unwrap();
            let shape: Vec<usize> = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
            let row = Row::new(name.clone(), dtype, shape, 1);
            let e = want.tensors.get(name).unwrap_or_else(|| panic!("`{name}` is not implied by the config"));
            assert_eq!(row.dtype, e.dtype, "`{name}`");
            assert!(e.shape.matches(&row.shape), "`{name}`: expected {}, found {:?}", e.shape.describe(), row.shape);
            one += row.bytes;
            assert_eq!(e.component, Component::Mtp);
        }
        assert_eq!(one, 8_891_916, "one NextN expert");
        assert_eq!(one * 3 * 128, 3_414_495_744, "three layers x 128 experts, 3.180 GiB");
        // and the first upload's format is still what the old pack's headers say
        let old = PackInventory::expected(&real, MtpExpertFormat::Int4).unwrap();
        assert_eq!(
            old.tensors["mtp.0.ffn.experts.0.w1.weight"].shape,
            Shape::Exact(vec![2304, 2560]),
            "the old pack's int8 expert"
        );
    }

    #[test]
    fn the_gate_follows_the_compress_ratio() {
        let mut v: Value = serde_json::from_str(CONFIG).unwrap();
        let t = v["text_config"].as_object_mut().unwrap();
        t.insert("num_hidden_layers".into(), 2.into());
        t.insert("num_nextn_predict_layers".into(), 0.into());
        t.insert("n_routed_experts".into(), 2.into());
        t.insert("num_experts_per_tok".into(), 2.into());
        // layer 0 compresses two tokens into one (a gate), layer 1 one token (no gate)
        t.insert("compress_ratios".into(), serde_json::json!([2, 1]));
        t.insert("kv_source_layer_ids".into(), serde_json::json!([0, 1]));
        t.insert("index_source_layer_ids".into(), serde_json::json!([0, 1]));
        t.insert("engram_layer_ids".into(), serde_json::json!([]));
        t.insert("engram_num_embeddings".into(), serde_json::json!([]));
        t.insert("dspark_target_layer_ids".into(), serde_json::json!([]));
        v.as_object_mut().unwrap().remove("vision_config");
        let cfg = DeepseekV41Config::from_json(&v).unwrap();
        let rows = toy_rows(&cfg);
        let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"layers.0.attn.compressor.wgate.weight"));
        assert!(!names.contains(&"layers.1.attn.compressor.wgate.weight"));
        // and the layer with ratio 1 still has its key projection: the two lists are independent
        assert!(names.contains(&"layers.1.attn.indexer.wk.weight"));
        assert!(PackInventory::from_rows(&cfg, rows).is_ok());
    }

    #[test]
    fn the_real_pack_inventory_maps_completely() {
        let Ok(text) = std::fs::read_to_string(PACK) else {
            eprintln!("not run: {PACK} is not on this machine");
            return;
        };
        let listing: Value = serde_json::from_str(&text).unwrap();
        let cfg = real();
        let inv = match PackInventory::from_json(&cfg, &listing) {
            Ok(inv) => inv,
            Err(e) => panic!("the real pack does not match the config: {e}"),
        };
        assert_eq!(inv.len(), 188_245, "the pack's tensor count");
        assert_eq!(listing.as_object().unwrap().len(), inv.len(), "every tensor of the listing is classified");
        assert_eq!(inv.shards(), 17);
        let counts: BTreeMap<&str, usize> = inv.counts().into_iter().map(|(c, n)| (c.as_str(), n)).collect();
        // 46,080 routed experts x 3 projections x 4 tensors
        assert_eq!(counts["moe_experts"], 40 * 384 * 3 * 4);
        assert_eq!(counts["mtp"], 3 * 798 + 7, "three NextN layers, plus the first one's projection and the last one's heads");
        assert_eq!(counts["engram"], 12);
        assert_eq!(
            (counts["embed"], counts["head"], counts["norm"]),
            (1, 1, 81),
            "one embedding, one head, 80 layer norms and the final one"
        );
        assert_eq!(counts["vision"], 32 * 8 + 2 + 1 + 4 + 3);
        // the six components that only exist on some layers
        assert_eq!(counts["dsa_compressor"], 4 * 3 - 1, "four compressors, three of them gated");
        assert_eq!(
            counts["dsa_index"],
            8 * 3 + 4 * 2,
            "eight selectors of queries and scores, four of them owning a key and its norm"
        );
        assert_eq!(counts["hc_attn"], 40 * 3, "the trunk's hyper-connections (the NextN layers' are `mtp`)");
        assert_eq!(counts["hc_ffn"], 40 * 3);
        assert_eq!(counts["moe_router"], 40 * 3);
        // 309.7 GiB of tensors over 17 shards
        assert_eq!(inv.bytes(), 332_513_253_720, "309.7 GiB of tensors over 17 shards");
    }
}
