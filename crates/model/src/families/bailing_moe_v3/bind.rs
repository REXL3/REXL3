//! Mapping a pack's inventory onto the typed description: [`BailingWeights::map`].
//!
//! One pure function walks the config's layer recipe and claims, for every slot, the tensor it expects: the name from the layer's
//! position, the dtype and shape from the config's fields, never from literals. Claiming marks the tensor used; after the walk, a
//! tensor nobody claimed is an error. A claim that finds nothing, finds the wrong dtype or shape, or finds a group the kernels
//! cannot take, is an error too. All problems are collected and reported together, each named; an `Ok` tree is therefore complete
//! and exact: every tensor of the pack is in it once, and it holds nothing the pack lacks.
//!
//! The same walk places every leaf: it opens the regions (see `plan::RegionKind`) and gives each tensor its offset, 256-byte
//! aligned, in the order of the walk; the routed experts of a layer are laid out as a bank (`ExpertBank`).

use super::config::{BailingConfig, KdaConfig};
use super::inventory::{GroupInfo, PackInventory, TensorInfo};
use super::plan::{Region, RegionKind};
use super::weights::*;
use crate::family::{Ffn, LayerSpec, Mixer};
use gb_exl3::budget::Category;
use std::collections::BTreeSet;

const P: &str = "model.layers.";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Problem {
    Missing { name: String, group: bool },
    Unexpected { name: String, group: bool },
    Dtype { name: String, want: &'static str, got: String },
    Shape { name: String, want: Vec<usize>, got: Vec<usize> },
    GroupShape { prefix: String, want: (usize, usize), got: (usize, usize) },
    Unsupported { prefix: String, why: String },
    NonUniform { layer: usize, what: String },
    Refused { prefix: String, why: String },
    HeaderIssue(String),
}

impl std::fmt::Display for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = |g: &bool| if *g { "EXL3 group" } else { "tensor" };
        match self {
            Problem::Missing { name, group } => write!(f, "{} `{name}` is not in the pack", kind(group)),
            Problem::Unexpected { name, group } => {
                write!(f, "{} `{name}` is in the pack and no slot of this family takes it", kind(group))
            }
            Problem::Dtype { name, want, got } => write!(f, "tensor `{name}`: expected {want}, found {got}"),
            Problem::Shape { name, want, got } => write!(f, "tensor `{name}`: expected shape {want:?}, found {got:?}"),
            Problem::GroupShape { prefix, want, got } => {
                write!(f, "EXL3 group `{prefix}`: expected K={} N={}, found K={} N={}", want.0, want.1, got.0, got.1)
            }
            Problem::Unsupported { prefix, why } => write!(f, "EXL3 group `{prefix}` is not loadable: {why}"),
            Problem::NonUniform { layer, what } => write!(f, "layer {layer}'s routed experts are not uniform: {what}"),
            Problem::Refused { prefix, why } => write!(f, "EXL3 group `{prefix}` was refused by the pack scan: {why}"),
            Problem::HeaderIssue(m) => write!(f, "pack header issue: {m}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayoutError {
    pub problems: Vec<Problem>,
}

impl std::fmt::Display for LayoutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const SHOWN: usize = 12;
        write!(f, "the pack does not match the Bailing-3.1 weight layout ({} problem(s)):", self.problems.len())?;
        for p in self.problems.iter().take(SHOWN) {
            write!(f, "\n  - {p}")?;
        }
        if self.problems.len() > SHOWN {
            write!(f, "\n  - and {} more", self.problems.len() - SHOWN)?;
        }
        Ok(())
    }
}

impl std::error::Error for LayoutError {}

fn is_benign_issue(issue: &str) -> bool {
    issue.contains(" is also stored in ") && issue.ends_with(" as the index says")
}

struct Binder<'a> {
    inv: &'a PackInventory,
    cfg: &'a BailingConfig,
    problems: Vec<Problem>,
    used_plain: BTreeSet<&'a str>,
    used_groups: BTreeSet<&'a str>,
    regions: Vec<Region>,
}

fn src(t: &TensorInfo) -> Src {
    Src { file: t.file, offset: t.offset }
}

impl<'a> Binder<'a> {
    fn region(&mut self, kind: RegionKind) -> RegionId {
        let id = self.regions.len() as RegionId;
        self.regions.push(Region { id, kind, bytes: 0 });
        id
    }

    fn place(&mut self, region: RegionId, bytes: u64) -> Dst {
        let r = &mut self.regions[region as usize];
        let offset = align_up(r.bytes);
        r.bytes = offset + bytes;
        Dst { region, offset }
    }

    fn plain_x(
        &mut self,
        name: &str,
        dtype: PlainDtype,
        shape: &[usize],
        xform: Xform,
        region: RegionId,
        category: Category,
    ) -> PlainTensor {
        let bytes = shape.iter().product::<usize>() as u64 * dtype.size();
        let mut out = PlainTensor {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            bytes,
            xform,
            dst_bytes: xform.dst_bytes(bytes),
            src: Src { file: 0, offset: 0 },
            dst: Dst::NONE,
            category,
        };
        let Some((key, t)) = self.inv.plain.get_key_value(name) else {
            self.problems.push(Problem::Missing { name: name.to_string(), group: false });
            return out;
        };
        self.used_plain.insert(key.as_str());
        if t.dtype != dtype.name() {
            self.problems.push(Problem::Dtype { name: name.to_string(), want: dtype.name(), got: t.dtype.clone() });
        } else if t.shape != shape {
            self.problems.push(Problem::Shape { name: name.to_string(), want: shape.to_vec(), got: t.shape.clone() });
        } else {
            out.src = src(t);
            out.dst = self.place(region, out.dst_bytes);
        }
        out
    }

    fn plain(&mut self, name: &str, dtype: PlainDtype, shape: &[usize], region: RegionId, category: Category) -> PlainTensor {
        self.plain_x(name, dtype, shape, Xform::None, region, category)
    }

    /// Claim a plain tensor without placing it (no region, no destination): for sources that feed a fused slot (`CatQkvConv`).
    fn claim_plain(&mut self, name: &str, dtype: PlainDtype, shape: &[usize], category: Category) -> PlainTensor {
        let bytes = shape.iter().product::<usize>() as u64 * dtype.size();
        let mut out = PlainTensor {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            bytes,
            xform: Xform::None,
            dst_bytes: bytes,
            src: Src { file: 0, offset: 0 },
            dst: Dst::NONE,
            category,
        };
        let Some((key, t)) = self.inv.plain.get_key_value(name) else {
            self.problems.push(Problem::Missing { name: name.to_string(), group: false });
            return out;
        };
        self.used_plain.insert(key.as_str());
        if t.dtype != dtype.name() {
            self.problems.push(Problem::Dtype { name: name.to_string(), want: dtype.name(), got: t.dtype.clone() });
        } else if t.shape != shape {
            self.problems.push(Problem::Shape { name: name.to_string(), want: shape.to_vec(), got: t.shape.clone() });
        } else {
            out.src = src(t);
        }
        out
    }

    /// The fused KDA conv1d: claims `q/k/v_conv1d.weight` and stacks them into one `[3 * channels, 1, conv]` slot. The three
    /// source tensors are checked (dtype, shape) and recorded; the single fused slot is placed.
    fn cat_qkv_conv(&mut self, base: &str, k: &KdaConfig, region: RegionId, category: Category) -> CatQkvConv {
        let channels = k.channels();
        let conv = k.conv_kernel;
        let elem = (channels * conv * 2) as u64;
        let mut parts: [Option<PlainTensor>; 3] = [None, None, None];
        for (i, which) in ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"].iter().enumerate() {
            let name = format!("{base}{which}");
            parts[i] = Some(self.claim_plain(&name, PlainDtype::Bf16, &[channels, 1, conv], category));
        }
        let mut fused = PlainTensor {
            name: format!("{base}q_conv1d.weight"),
            dtype: PlainDtype::Bf16,
            shape: vec![3 * channels, 1, conv],
            bytes: 3 * elem,
            xform: Xform::CatQkvConv,
            dst_bytes: 3 * elem,
            src: Src { file: 0, offset: 0 },
            dst: Dst::NONE,
            category,
        };
        let [q, k, v] = parts.map(|p| p.expect("three parts"));
        // the fused slot is one contiguous allocation; its source is the first part (the plan reads the three separately)
        fused.src = q.src;
        fused.dst = self.place(region, 3 * elem);
        CatQkvConv { parts: [q, k, v], fused }
    }

    fn group(&mut self, prefix: &str, k: usize, n: usize, region: Option<RegionId>, category: Category) -> Exl3Linear {
        let none = Exl3Part { bytes: 0, src: Src { file: 0, offset: 0 }, dst: Dst::NONE };
        let mut out = Exl3Linear {
            prefix: prefix.to_string(),
            k,
            n,
            bits: gb_exl3::Bits::from_twice(2).expect("2 is a width"),
            codebook: gb_exl3::Codebook::Mul1,
            trellis: none,
            suh: none,
            svh: none,
            marker: None,
            category,
        };
        let Some((key, g)) = self.inv.groups.get_key_value(prefix) else {
            self.problems.push(Problem::Missing { name: prefix.to_string(), group: true });
            return out;
        };
        self.used_groups.insert(key.as_str());
        if (g.k, g.n) != (k, n) {
            self.problems.push(Problem::GroupShape { prefix: prefix.to_string(), want: (k, n), got: (g.k, g.n) });
            return out;
        }
        if let Some(why) = unsupported(g) {
            self.problems.push(Problem::Unsupported { prefix: prefix.to_string(), why });
            return out;
        }
        let part = |t: &TensorInfo| Exl3Part { bytes: t.bytes, src: src(t), dst: Dst::NONE };
        out.bits = g.bits;
        out.codebook = g.codebook;
        out.trellis = part(&g.trellis);
        out.suh = part(&g.in_scale);
        out.svh = part(&g.out_scale);
        out.marker = g.marker.as_ref().map(|m| (src(m), m.bytes));
        if let Some(r) = region {
            out.trellis.dst = self.place(r, out.trellis.bytes);
            out.suh.dst = self.place(r, out.suh.bytes);
            out.svh.dst = self.place(r, out.svh.bytes);
        }
        out
    }

    fn kda(&mut self, l: &str, region: RegionId, cat: Category) -> KdaWeights {
        let (h, k) = (self.cfg.hidden, &self.cfg.kda);
        let (heads, d) = (k.heads, k.head_dim);
        let channels = heads * d;
        let a = format!("{l}attention.");
        use PlainDtype::*;
        KdaWeights {
            qkv_proj: self.group(&format!("{a}qkv_proj"), h, 3 * channels, Some(region), cat),
            f_proj: self.group(&format!("{a}f_proj"), h, channels, Some(region), cat),
            g_proj: self.group(&format!("{a}g_proj"), h, channels, Some(region), cat),
            o_proj: self.group(&format!("{a}o_proj"), channels, h, Some(region), cat),
            b_proj: self.plain(&format!("{a}b_proj.weight"), Bf16, &[heads, h], region, cat),
            a_log: self.plain(&format!("{a}A_log"), F32, &[heads], region, cat),
            dt_bias: self.plain(&format!("{a}dt_bias"), F32, &[channels], region, cat),
            conv1d_weight: self.cat_qkv_conv(&a, k, region, cat),
            o_norm: self.plain(&format!("{a}o_norm.weight"), Bf16, &[d], region, cat),
        }
    }

    fn mla(&mut self, l: &str, region: RegionId, cat: Category) -> MlaWeights {
        let (h, m) = (self.cfg.hidden, &self.cfg.mla);
        let (heads, q_rank, kv_rank) = (m.heads, m.q_lora_rank, m.kv_lora_rank);
        let (qk, nope, v) = (m.qk_head_dim(), m.qk_nope_head_dim, m.v_head_dim);
        let a = format!("{l}attention.");
        use PlainDtype::*;
        MlaWeights {
            q_a_proj: self.group(&format!("{a}q_a_proj"), h, q_rank, Some(region), cat),
            q_a_layernorm: self.plain_x(&format!("{a}q_a_layernorm.weight"), Bf16, &[q_rank], Xform::Bf16ToF32, region, cat),
            q_b_proj: self.group(&format!("{a}q_b_proj"), q_rank, heads * qk, Some(region), cat),
            kv_a_proj_with_mqa: self.group(&format!("{a}kv_a_proj_with_mqa"), h, m.kv_a_group_n(), Some(region), cat),
            // kv_a_layernorm.weight stays BF16 on the GPU: kv_prep_kernel<T> reads norm_w as T (bf16), not f32.
            kv_a_layernorm: self.plain(&format!("{a}kv_a_layernorm.weight"), Bf16, &[kv_rank], region, cat),
            kv_b_proj: self.plain_x(
                &format!("{a}kv_b_proj.weight"),
                F16,
                &[heads * (nope + v), kv_rank],
                Xform::PackKvB,
                region,
                cat,
            ),
            g_proj: self.plain(&format!("{a}g_proj.weight"), Bf16, &[heads, h], region, cat),
            dense: self.group(&format!("{a}dense"), heads * v, h, Some(region), cat),
        }
    }

    fn gated_mlp(&mut self, prefix: &str, inter: usize, region: RegionId, cat: Category) -> GatedMlp {
        let h = self.cfg.hidden;
        GatedMlp {
            gate_proj: self.group(&format!("{prefix}gate_proj"), h, inter, Some(region), cat),
            up_proj: self.group(&format!("{prefix}up_proj"), h, inter, Some(region), cat),
            down_proj: self.group(&format!("{prefix}down_proj"), inter, h, Some(region), cat),
        }
    }

    fn expert_bank(&mut self, l: &str, layer: usize, cat: Category) -> ExpertBank {
        use gb_cuda::moe::{subbank_partition, SubbankKey};
        let (h, mi, n) = (self.cfg.hidden, self.cfg.ffn.moe_intermediate, self.cfg.ffn.routed_experts);
        let region = self.region(RegionKind::Experts(layer));
        let before = self.problems.len();
        let mut experts = Vec::with_capacity(n);
        for e in 0..n {
            let p = format!("{l}mlp.experts.{e}.");
            experts.push(ExpertWeights {
                gate_proj: self.group(&format!("{p}gate_proj"), h, mi, None, cat),
                up_proj: self.group(&format!("{p}up_proj"), h, mi, None, cat),
                down_proj: self.group(&format!("{p}down_proj"), mi, h, None, cat),
            });
        }
        let unplaced = ExpertBank {
            layer,
            region,
            hidden: h,
            inter: mi,
            experts: vec![],
            layout: ExpertLayout { gate: ProjLayout::default(), up: ProjLayout::default(), down: ProjLayout::default() },
            subbanks: vec![],
        };
        if self.problems.len() != before {
            return ExpertBank { experts, ..unplaced };
        }
        // the key of every expert: its three projections share a codebook; fractional widths are ordinary here and keep their
        // halves (`Bits::twice()`: 2 = 1.0, 3 = 1.5, 4 = 2.0, ...) so 2.0 and 2.5 are different subbanks
        let mut keys = Vec::with_capacity(n);
        for (e, x) in experts.iter().enumerate() {
            if x.up_proj.codebook != x.gate_proj.codebook || x.down_proj.codebook != x.gate_proj.codebook {
                self.problems
                    .push(Problem::NonUniform { layer, what: format!("expert {e}'s projections use different codebooks") });
            }
            let b = |g: &Exl3Linear| g.bits.twice();
            keys.push(SubbankKey {
                bits_gate: b(&x.gate_proj),
                bits_up: b(&x.up_proj),
                bits_down: b(&x.down_proj),
                codebook: x.gate_proj.codebook,
            });
        }
        if self.problems.len() != before {
            return ExpertBank { experts, ..unplaced };
        }
        let own = gb_cuda::moe::ExpertOwnership::all(n);
        let parts = subbank_partition(&keys, &own).expect("ownership covers every expert");
        // within a subbank every member has the first member's byte sizes
        for (_, o) in &parts {
            let f = &experts[o.owned()[0] as usize];
            for &g in o.owned() {
                let x = &experts[g as usize];
                for (proj, a, want) in [
                    ("gate_proj", &x.gate_proj, &f.gate_proj),
                    ("up_proj", &x.up_proj, &f.up_proj),
                    ("down_proj", &x.down_proj, &f.down_proj),
                ] {
                    if (a.trellis.bytes, a.suh.bytes, a.svh.bytes) != (want.trellis.bytes, want.suh.bytes, want.svh.bytes) {
                        self.problems.push(Problem::NonUniform {
                            layer,
                            what: format!(
                                "expert {g} {proj} has a different byte size from expert {}'s of the same width",
                                o.owned()[0]
                            ),
                        });
                    }
                }
            }
        }
        if self.problems.len() != before {
            return ExpertBank { experts, ..unplaced };
        }
        let mut cursor = 0u64;
        let mut subbanks = Vec::with_capacity(parts.len());
        for (key, o) in parts {
            let nb = o.local() as u64;
            let f = &experts[o.owned()[0] as usize];
            let (fg, fu, fd) = (f.gate_proj.clone(), f.up_proj.clone(), f.down_proj.clone());
            let mut block = |bytes: u64| {
                let off = align_up(cursor);
                cursor = off + bytes;
                off
            };
            let (tg, tu, td) = (block(nb * fg.trellis.bytes), block(nb * fu.trellis.bytes), block(nb * fd.trellis.bytes));
            let (suh_g, svh_g) = (block(nb * fg.suh.bytes), block(nb * fg.svh.bytes));
            let (suh_u, svh_u) = (block(nb * fu.suh.bytes), block(nb * fu.svh.bytes));
            let (suh_d, svh_d) = (block(nb * fd.suh.bytes), block(nb * fd.svh.bytes));
            let proj = |trellis, suh, svh, g: &Exl3Linear| ProjLayout {
                trellis,
                trellis_stride: g.trellis.bytes,
                suh,
                suh_stride: g.suh.bytes,
                svh,
                svh_stride: g.svh.bytes,
            };
            let layout = ExpertLayout {
                gate: proj(tg, suh_g, svh_g, &fg),
                up: proj(tu, suh_u, svh_u, &fu),
                down: proj(td, suh_d, svh_d, &fd),
            };
            for (local, &g) in o.owned().iter().enumerate() {
                let x = &mut experts[g as usize];
                for (g, lay) in [(&mut x.gate_proj, &layout.gate), (&mut x.up_proj, &layout.up), (&mut x.down_proj, &layout.down)]
                {
                    g.trellis.dst = Dst { region, offset: lay.trellis_at(local) };
                    g.suh.dst = Dst { region, offset: lay.suh_at(local) };
                    g.svh.dst = Dst { region, offset: lay.svh_at(local) };
                }
            }
            subbanks.push(ExpertSubbank { key, ids: o.owned().to_vec(), layout });
        }
        self.regions[region as usize].bytes = align_up(cursor);
        let Some(first) = subbanks.first() else {
            return ExpertBank { experts, ..unplaced };
        };
        ExpertBank { experts, layout: first.layout, subbanks, ..unplaced }
    }

    fn layer(
        &mut self,
        index: usize,
        spec: LayerSpec,
        region: RegionId,
        trunk: Category,
        routed: Category,
    ) -> Option<BailingLayerWeights> {
        let (h, ffn_cfg) = (self.cfg.hidden, &self.cfg.ffn);
        let (inter, shared_n, dense_inter) = (ffn_cfg.moe_intermediate, ffn_cfg.shared_experts, ffn_cfg.dense_intermediate);
        let l = format!("{P}{index}.");
        let input_layernorm =
            self.plain_x(&format!("{l}input_layernorm.weight"), PlainDtype::Bf16, &[h], Xform::Bf16ToF32, region, trunk);
        let post_attention_layernorm =
            self.plain_x(&format!("{l}post_attention_layernorm.weight"), PlainDtype::Bf16, &[h], Xform::Bf16ToF32, region, trunk);
        let mixer = match spec.mixer {
            Mixer::Kda => MixerWeights::Kda(Box::new(self.kda(&l, region, trunk))),
            // the gated-MLA layers of this family reuse the MlaDsa label in the neutral recipe; the binder claims the gated tensors
            Mixer::MlaDsa { .. } => MixerWeights::Mla(Box::new(self.mla(&l, region, trunk))),
            other => {
                self.problems.push(Problem::HeaderIssue(format!("layer {index}: the Bailing family has no mixer {:?}", other)));
                return None;
            }
        };
        let ffn = match spec.ffn {
            Ffn::Dense => FfnWeights::Dense(Box::new(self.gated_mlp(&format!("{l}mlp."), dense_inter, region, trunk))),
            Ffn::Moe => {
                let router =
                    self.plain(&format!("{l}mlp.gate.weight"), PlainDtype::Bf16, &[ffn_cfg.routed_experts, h], region, trunk);
                let expert_bias =
                    self.plain(&format!("{l}mlp.gate.expert_bias"), PlainDtype::F32, &[ffn_cfg.routed_experts], region, trunk);
                let experts = self.expert_bank(&l, index, routed);
                let shared = (shared_n > 0).then(|| self.gated_mlp(&format!("{l}mlp.shared_experts."), inter, region, trunk));
                FfnWeights::Moe(Box::new(MoeWeights { router, expert_bias, experts, shared }))
            }
            other => {
                self.problems.push(Problem::HeaderIssue(format!("layer {index}: the Bailing family has no FFN {:?}", other)));
                return None;
            }
        };
        Some(BailingLayerWeights { index, input_layernorm, post_attention_layernorm, mixer, ffn })
    }
}

/// Why a group is in a form the loader cannot take, if it is. Fractional widths are ordinary (the pack is mixed-K, including
/// halves); a group with a bias, or with non-fp16 scales, is refused.
fn unsupported(g: &GroupInfo) -> Option<String> {
    if g.in_scales != "suh" || g.out_scales != "svh" {
        return Some(format!("scales are `{}`/`{}`, only fp16 `suh`/`svh` are uploaded as stored", g.in_scales, g.out_scales));
    }
    if g.bias.is_some() {
        return Some("it has a bias".into());
    }
    None
}

impl BailingWeights {
    /// Map a pack's inventory onto the family's weights. See the module documentation: the result is exact (every tensor of the
    /// pack is in the tree once, and nothing else is) or an error naming every problem. The embedding is placed in its own
    /// `Sidecar` region (an `mmap` of shard 1 on the target), so it claims no device memory.
    pub fn map(inv: &PackInventory, cfg: &BailingConfig) -> Result<BailingWeights, LayoutError> {
        let mut b =
            Binder { inv, cfg, problems: vec![], used_plain: BTreeSet::new(), used_groups: BTreeSet::new(), regions: vec![] };
        for (prefix, why) in &inv.refused {
            b.problems.push(Problem::Refused { prefix: prefix.clone(), why: why.clone() });
        }
        for issue in inv.issues.iter().filter(|i| !is_benign_issue(i)) {
            b.problems.push(Problem::HeaderIssue(issue.clone()));
        }

        let sidecar = b.region(RegionKind::Sidecar);
        let mut embed_tokens = b.plain_x(
            "model.word_embeddings.weight",
            PlainDtype::Bf16,
            &[cfg.vocab, cfg.hidden],
            Xform::None,
            sidecar,
            Category::Embedding,
        );
        // the embedding is addressed through the mmap sidecar, never placed in (or loaded into) a device region
        embed_tokens.dst = Dst::NONE;
        let global = b.region(RegionKind::Global);
        let final_norm =
            b.plain_x("model.norm.weight", PlainDtype::Bf16, &[cfg.hidden], Xform::Bf16ToF32, global, Category::Other);
        let lm_head = b.group("lm_head", cfg.hidden, cfg.vocab, Some(global), Category::Head);

        let mut layers = Vec::with_capacity(cfg.layers.len());
        for (i, spec) in cfg.layers.iter().enumerate() {
            let region = b.region(RegionKind::Trunk(i));
            layers.extend(b.layer(i, *spec, region, Category::Decoder, Category::RoutedExpert));
        }
        let mut mtp = Vec::with_capacity(cfg.mtp.len());
        for (j, spec) in cfg.mtp.iter().enumerate() {
            let index = cfg.mtp_layer_index(j);
            let region = b.region(RegionKind::Trunk(index));
            let l = format!("{P}{index}.");
            let h = cfg.hidden;
            let eh_proj = b.plain(&format!("{l}eh_proj.weight"), PlainDtype::Bf16, &[h, 2 * h], region, Category::Mtp);
            let enorm = b.plain_x(&format!("{l}enorm.weight"), PlainDtype::Bf16, &[h], Xform::Bf16ToF32, region, Category::Mtp);
            let hnorm = b.plain_x(&format!("{l}hnorm.weight"), PlainDtype::Bf16, &[h], Xform::Bf16ToF32, region, Category::Mtp);
            let final_layernorm =
                b.plain_x(&format!("{l}final_layernorm.weight"), PlainDtype::Bf16, &[h], Xform::Bf16ToF32, region, Category::Mtp);
            if let Some(layer) = b.layer(index, *spec, region, Category::Mtp, Category::Mtp) {
                mtp.push(BailingMtpWeights { eh_proj, enorm, hnorm, final_layernorm, layer });
            }
        }

        // fail closed: whatever no slot claimed is an error
        for name in inv.plain.keys().filter(|n| !b.used_plain.contains(n.as_str())) {
            b.problems.push(Problem::Unexpected { name: name.clone(), group: false });
        }
        for name in inv.groups.keys().filter(|n| !b.used_groups.contains(n.as_str())) {
            b.problems.push(Problem::Unexpected { name: name.clone(), group: true });
        }
        if !b.problems.is_empty() {
            return Err(LayoutError { problems: b.problems });
        }
        let mut regions = b.regions;
        for r in &mut regions {
            if r.kind == RegionKind::Sidecar {
                r.bytes = 0; // the embedding is a sidecar: the region is never allocated
            } else {
                r.bytes = align_up(r.bytes);
            }
        }
        Ok(BailingWeights { files: inv.files.clone(), regions, embed_tokens, final_norm, lm_head, layers, mtp })
    }
}
