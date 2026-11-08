//! CPU-only MiniCPM-V-4.7 weight binding and upload recipe (E2 of `docs/MINICPMV4_7_INTEGRATION_PLAN.md`).
//!
//! [`WeightMap::bind`] claims every tensor of a pack inventory once (the recipe of [`super::inventory`], which names every missing,
//! misshapen, misplaced or unknown tensor first), says for each what the device holds (a 16-bit copy, an fp32 widening with or without
//! the RMSNorm `+1`, one part of an EXL3 group) and where it is counted: one region per layer for the trunk, one per layer for the 768
//! routed-expert groups, one for the head, the embedding and the final norm. [`WeightMap::from_dir`] / [`WeightMap::from_checkpoint`]
//! bind a scanned pack and record every tensor's file extent. Nothing here reads a weight or touches a device: the pipelined loader
//! (`super::device`) executes the plan, and the memory plan counts its bytes (`LoadPlan::upload_bytes`) before a byte moves.
//!
//! The regions are the plan's accounting unit, not placements: the loader's arena carves each item at its own 256-aligned size in
//! source order (the MoE kernels address experts through pointer tables, so no layout inside a layer is needed), so the arena's
//! carved bytes equal the sum of the regions to the byte, and the boot checks that they do.
//!
//! What differs from the other families' maps, and why:
//!
//! * the norms of the trunk (`input_layernorm`, `post_attention_layernorm`, the final `norm`, and the attention's `q_norm` / `k_norm`)
//!   are stored as `w` of `(1 + w)` and widen to fp32 with the one added ([`Transform::Bf16AddOneToF32`]); the gated delta net's own
//!   `norm` is a gated RMSNorm whose weight is used as is ([`Transform::Bf16ToF32`]), as are `A_log` and `dt_bias`;
//! * `in_proj_a`, `in_proj_b`, the router and the shared expert's gate are plain fp16 matrices the small-linear kernel reads;
//! * the tower and the projector (459 tensors, 1.106 GB, bf16) are claimed by the recipe so a pack that lacks one is refused, and
//!   [`Kind::Recorded`] here: present and counted, not uploaded (the vision slice, E5, owns them);
//! * the head is one EXL3 group whose output width is the padded 248,192; [`WeightMap::lm_head_valid`] is the 248,144 real ids.

use super::config::MiniCpmConfig;
use super::inventory::{self, Tensors, MERGER_PREFIX, TEXT_PREFIX, VISION_PREFIX};
use gb_exl3::checkpoint::Checkpoint;
use gb_exl3::{Bits, Codebook};
use std::collections::BTreeMap;
use std::path::Path;

/// Alignment of every upload item in a region (and of every carve of the loader's arena).
pub const ALIGN: u64 = gb_cuda::arena::ALIGN as u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transform {
    /// The bytes as stored.
    None,
    /// bf16 widened to fp32.
    Bf16ToF32,
    /// bf16 widened to fp32 with 1 added in fp32 (an RMSNorm `(1 + w)` gamma).
    Bf16AddOneToF32,
    /// bf16 converted to fp16.
    Bf16ToF16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A plain tensor the device holds.
    Plain,
    /// `trellis`, `suh` or `svh` of an EXL3 group.
    Exl3Part,
    /// The `mul1` codebook marker: checked by the scan, not uploaded.
    Marker,
    /// A tensor the pack carries and the recipe claims that this slice does not upload (the tower and the projector).
    Recorded,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Source {
    pub file: String,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Destination {
    pub region: u32,
    pub offset: u64,
    pub bytes: u64,
}

/// One EXL3 linear: its four parts are named `<prefix>.{trellis,suh,svh,mul1}` in `slots`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exl3Group {
    pub prefix: String,
    pub k: usize,
    pub n: usize,
    pub bits: Bits,
    pub codebook: Codebook,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub kind: Kind,
    pub transform: Transform,
    pub source: Option<Source>,
    pub destination: Option<Destination>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub name: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadPlan {
    /// The copies the device executes, ordered by source file and offset when sources are known.
    pub items: Vec<Slot>,
    /// The codebook markers: validated by the scan, never uploaded.
    pub markers: Vec<Slot>,
    /// The tower and the projector: present, counted, not uploaded.
    pub recorded: Vec<Slot>,
    /// Global, per-layer trunk and per-layer routed-expert regions; offsets and sizes are [`ALIGN`]-byte aligned.
    pub regions: Vec<Region>,
}

impl LoadPlan {
    /// Bytes of every region: what the loader's arena carves for the weights (each item rounded up to [`ALIGN`], so the sum is
    /// the arena's `used_bytes` exactly).
    pub fn upload_bytes(&self) -> u64 {
        self.regions.iter().map(|r| r.bytes).sum()
    }

    /// Bytes the recorded tensors hold in the pack (what the vision slice will have to load).
    pub fn recorded_bytes(&self) -> u64 {
        self.recorded.iter().map(slot_bytes).sum()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightMap {
    pub slots: BTreeMap<String, Slot>,
    pub groups: BTreeMap<String, Exl3Group>,
    pub regions: Vec<Region>,
    /// Real vocabulary ids (the head's columns past it are padding the sampler must never see).
    pub vocab: usize,
    /// Width of the head's output: the padded vocabulary.
    pub lm_head_width: usize,
}

/// (dtype the device expects, what the slot is, its conversion) of a tensor name.
fn specification(name: &str, dtype: &str) -> (&'static str, Kind, Transform) {
    if name.starts_with(VISION_PREFIX)
        || name.starts_with(MERGER_PREFIX)
        || name.starts_with("model.visual.")
        || name.starts_with("mtp.")
    {
        return ("BF16", Kind::Recorded, Transform::None);
    }
    match name.rsplit('.').next() {
        Some("trellis") => return ("I16", Kind::Exl3Part, Transform::None),
        Some("suh" | "svh") => return ("F16", Kind::Exl3Part, Transform::None),
        Some("mul1") => return ("I32", Kind::Marker, Transform::None),
        _ => {}
    }
    if name == format!("{TEXT_PREFIX}.embed_tokens.weight") {
        return ("BF16", Kind::Plain, Transform::None);
    }
    if name.contains(".linear_attn.") {
        // the gated RMSNorm of the delta net (no +1), and its decay parameters
        if name.ends_with("A_log") || name.ends_with("dt_bias") || name.ends_with(".norm.weight") {
            return ("BF16", Kind::Plain, Transform::Bf16ToF32);
        }
        if name.ends_with(".conv1d.weight") {
            return ("BF16", Kind::Plain, Transform::None);
        }
    }
    // every other RMSNorm of the trunk is `(1 + w)`
    if name.ends_with(".input_layernorm.weight")
        || name.ends_with(".post_attention_layernorm.weight")
        || name == format!("{TEXT_PREFIX}.norm.weight")
        || (name.contains(".self_attn.") && (name.ends_with(".q_norm.weight") || name.ends_with(".k_norm.weight")))
    {
        return ("BF16", Kind::Plain, Transform::Bf16AddOneToF32);
    }
    // in_proj_a / in_proj_b, the router and the shared expert's gate
    if dtype == "BF16" {
        ("F16", Kind::Plain, Transform::Bf16ToF16)
    } else {
        ("F16", Kind::Plain, Transform::None)
    }
}

/// The region a tensor lands in.
fn region_name(name: &str) -> String {
    if let Some(rest) = name.strip_prefix(&format!("{TEXT_PREFIX}.layers.")) {
        if let Some((index, part)) = rest.split_once('.') {
            let kind = if part.starts_with("mlp.experts.") { "experts" } else { "trunk" };
            return format!("layer_{index}_{kind}");
        }
    }
    "global".into()
}

fn slot_bytes(slot: &Slot) -> u64 {
    let size = gb_exl3::checkpoint::dtype_size(&slot.dtype).unwrap_or(0);
    slot.shape.iter().fold(size, |n, &d| n.saturating_mul(d as u64))
}

fn round_up(n: u64) -> u64 {
    n.div_ceil(ALIGN) * ALIGN
}

impl WeightMap {
    /// Bind the text recipe of `cfg` (and record the tower) against an inventory, each tensor exactly once. Every refusal names the
    /// tensor; the list is [`inventory::check`]'s, so nothing is bound from a pack that disagrees with its own config.
    pub fn bind(cfg: &MiniCpmConfig, tensors: &Tensors) -> Result<Self, Vec<String>> {
        let (errors, claimed) = inventory::check_with_claims(cfg, tensors);
        if !errors.is_empty() {
            return Err(errors);
        }
        let mut errors: Vec<String> = Vec::new();
        let mut slots = BTreeMap::new();
        let mut groups = BTreeMap::new();
        let mut regions: Vec<Region> = Vec::new();
        let mut region_ids: BTreeMap<String, u32> = BTreeMap::new();
        for name in &claimed {
            let Some((dtype, shape)) = tensors.get(name) else {
                errors.push(format!("{name}: the recipe claims it and the inventory lacks it"));
                continue;
            };
            let (_, kind, transform) = specification(name, dtype);
            if kind == Kind::Exl3Part && name.ends_with(".trellis") && shape.len() == 3 {
                let prefix = name.trim_end_matches(".trellis");
                match Bits::from_trellis_last(shape[2]) {
                    Some(bits) if bits.is_integer() => {
                        groups.insert(
                            prefix.to_string(),
                            Exl3Group {
                                prefix: prefix.to_string(),
                                k: shape[0] * 16,
                                n: shape[1] * 16,
                                bits,
                                codebook: Codebook::Mul1,
                            },
                        );
                    }
                    _ => errors.push(format!("{name}: unsupported EXL3 bit width (last dimension {})", shape[2])),
                }
            }
            let destination = if matches!(kind, Kind::Plain | Kind::Exl3Part) {
                let region = region_name(name);
                let id = *region_ids.entry(region.clone()).or_insert_with(|| {
                    regions.push(Region { name: region, bytes: 0 });
                    (regions.len() - 1) as u32
                });
                let size = gb_exl3::checkpoint::dtype_size(dtype).unwrap_or(0);
                let stored = shape.iter().fold(size, |n, &d| n.saturating_mul(d as u64));
                let bytes = if matches!(transform, Transform::None | Transform::Bf16ToF16) { stored } else { stored * 2 };
                let r = &mut regions[id as usize];
                let offset = round_up(r.bytes);
                r.bytes = offset + bytes;
                Some(Destination { region: id, offset, bytes })
            } else {
                None
            };
            slots.insert(
                name.clone(),
                Slot {
                    name: name.clone(),
                    dtype: dtype.clone(),
                    shape: shape.clone(),
                    kind,
                    transform,
                    source: None,
                    destination,
                },
            );
        }
        for r in &mut regions {
            r.bytes = round_up(r.bytes);
        }
        if !errors.is_empty() {
            return Err(errors);
        }
        let lm = groups.get("lm_head").ok_or_else(|| vec!["lm_head: no EXL3 group bound".to_string()])?;
        let (vocab, lm_head_width) = (cfg.vocab, lm.n);
        if lm_head_width != cfg.lm_head_width() {
            return Err(vec![format!(
                "lm_head: {lm_head_width} output rows, the recipe needs {} (ceil({vocab} / 128) x 128)",
                cfg.lm_head_width()
            )]);
        }
        Ok(Self { slots, groups, regions, vocab, lm_head_width })
    }

    /// Bind a scanned checkpoint: the scan's own refusals (a group it refused, a header issue, a codebook that is not `mul1`)
    /// come first and with the recipe's, every one by name, then each tensor's file extent is recorded.
    pub fn from_checkpoint(ck: &Checkpoint, cfg: &MiniCpmConfig) -> Result<Self, Vec<String>> {
        let mut errors: Vec<String> = ck
            .bad
            .iter()
            .map(|(name, why)| format!("{name}: refused EXL3 group: {why}"))
            .chain(
                ck.issues
                    .iter()
                    .filter(|issue| !issue.contains(" as the index says"))
                    .map(|issue| format!("pack header: {issue}")),
            )
            .collect();
        for (prefix, g) in &ck.groups {
            if g.codebook != Codebook::Mul1 {
                errors
                    .push(format!("{prefix}: codebook {} (the engine runs `mul1`: convert with `-cb mul1`)", g.codebook.name()));
            }
            if g.in_scales != "suh" || g.out_scales != "svh" {
                errors.push(format!(
                    "{prefix}: {}/{} scales (the engine reads fp16 `suh` and `svh`, not packed signs)",
                    g.in_scales, g.out_scales
                ));
            }
            if g.bias && !prefix.starts_with("model.visual.") && !prefix.starts_with("mtp.") {
                errors.push(format!("{prefix}: carries a bias (no linear of this model has one)"));
            }
        }
        let mut tensors = Tensors::new();
        let mut sources: BTreeMap<String, Source> = BTreeMap::new();
        for (name, e) in &ck.plain {
            let offset = ck.file_offset(e).map_err(|err| vec![format!("{name}: {err}")])?;
            tensors.insert(name.clone(), (e.dtype.clone(), e.shape.clone()));
            sources.insert(name.clone(), Source { file: e.file.clone(), offset, bytes: e.nbytes() });
        }
        for (prefix, g) in &ck.groups {
            let mut parts = vec!["trellis", g.in_scales, g.out_scales, g.codebook.name()];
            if g.bias && !prefix.starts_with("model.visual.") && !prefix.starts_with("mtp.") {
                parts.push("bias");
            }
            for suffix in parts {
                let name = format!("{prefix}.{suffix}");
                if let Some(e) = ck.part(&name) {
                    let offset = ck.file_offset(e).map_err(|err| vec![format!("{name}: {err}")])?;
                    tensors.insert(name.clone(), (e.dtype.clone(), e.shape.clone()));
                    sources.insert(name, Source { file: e.file.clone(), offset, bytes: e.nbytes() });
                }
            }
        }
        match Self::bind(cfg, &tensors) {
            Ok(mut map) if errors.is_empty() => {
                for (name, slot) in &mut map.slots {
                    slot.source = sources.remove(name);
                }
                Ok(map)
            }
            Ok(_) => Err(errors),
            Err(more) => {
                errors.extend(more);
                Err(errors)
            }
        }
    }

    /// Scan the pack in `dir` (headers and codebook markers, no weight array) and bind it.
    pub fn from_dir(dir: &Path, cfg: &MiniCpmConfig) -> Result<Self, Vec<String>> {
        let ck = gb_exl3::checkpoint::scan(dir, true).map_err(|e| vec![format!("{}: {e}", dir.display())])?;
        Self::from_checkpoint(&ck, cfg)
    }

    /// The valid ids of the head's output: everything at or past it is padding.
    pub fn lm_head_valid(&self) -> usize {
        self.vocab
    }

    pub fn load_plan(&self) -> LoadPlan {
        let (mut items, mut markers, mut recorded) = (Vec::new(), Vec::new(), Vec::new());
        for slot in self.slots.values() {
            match slot.kind {
                Kind::Plain | Kind::Exl3Part => items.push(slot.clone()),
                Kind::Marker => markers.push(slot.clone()),
                Kind::Recorded => recorded.push(slot.clone()),
            }
        }
        items.sort_by(|a, b| {
            a.source
                .as_ref()
                .map(|p| (&p.file, p.offset))
                .cmp(&b.source.as_ref().map(|p| (&p.file, p.offset)))
                .then(a.name.cmp(&b.name))
        });
        LoadPlan { items, markers, recorded, regions: self.regions.clone() }
    }

    /// Bytes the weights take on the device (the sum of the regions).
    pub fn upload_bytes(&self) -> u64 {
        self.regions.iter().map(|r| r.bytes).sum()
    }

    /// Bytes of the payload alone (no alignment): what `DevBuf::bytes` of every weight buffer sums to.
    pub fn payload_bytes(&self) -> u64 {
        self.slots.values().filter_map(|s| s.destination.as_ref()).map(|d| d.bytes).sum()
    }

    /// EXL3 groups by bit width (the width, then how many groups run at it).
    pub fn bit_histogram(&self) -> BTreeMap<String, usize> {
        let mut h = BTreeMap::new();
        for g in self.groups.values() {
            *h.entry(format!("{}", g.bits.as_f64())).or_insert(0) += 1;
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::real;
    use super::*;

    const TABLE: &str = include_str!("../../../../../testdata/minicpmv4_7/tensor_inventory.tsv");

    fn table() -> Tensors {
        inventory::expand_table(TABLE).unwrap()
    }

    fn has(errs: &[String], needle: &str) -> bool {
        errs.iter().any(|e| e.contains(needle))
    }

    /// The config of the first `n` layers, as the converter's smoke pack carries it.
    fn truncated(n: usize) -> MiniCpmConfig {
        let mut c = real();
        c.layers = n;
        c.layer_types.truncate(n);
        c
    }

    fn only_layers(t: &Tensors, n: usize) -> Tensors {
        t.iter()
            .filter(|(name, _)| {
                name.strip_prefix("model.language_model.layers.")
                    .and_then(|r| r.split('.').next())
                    .and_then(|l| l.parse::<usize>().ok())
                    .is_none_or(|l| l < n)
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    #[test]
    fn alignment_is_the_loaders_arena_alignment() {
        assert_eq!(ALIGN, 256);
    }

    #[test]
    fn every_tensor_of_the_committed_inventory_binds_once() {
        let t = table();
        let map = WeightMap::bind(&real(), &t).unwrap();
        // 40 x 3,106 + head 4 + embedding + final norm + 459 tower tensors: every one is a slot
        assert_eq!(map.slots.len(), 124_705);
        assert_eq!(map.slots.len(), t.len());
        // 23,220 GDN-layer groups (3 + 768 + 3 each x 30) + 7,750 attention-layer groups (4 + 768 + 3 each x 10) + the head
        assert_eq!(map.groups.len(), 30 * 774 + 10 * 775 + 1);
        assert_eq!(map.groups.len(), 30_971);
        let plan = map.load_plan();
        // plain on the device: 30 x 10 (norms 2, a, b, conv, A_log, dt_bias, norm, router, shared gate) + 10 x 6 + embedding + final norm
        let plain = plan.items.iter().filter(|s| s.kind == Kind::Plain).count();
        assert_eq!(plain, 30 * 10 + 10 * 6 + 2);
        let parts = plan.items.iter().filter(|s| s.kind == Kind::Exl3Part).count();
        assert_eq!(parts, 30_971 * 3);
        assert_eq!(plan.markers.len(), 30_971);
        assert_eq!(plan.items.len() + plan.markers.len() + plan.recorded.len(), map.slots.len());
        // the tower and the projector: counted, never uploaded
        assert_eq!((plan.recorded.len(), plan.recorded_bytes()), (inventory::VISION_TENSORS, inventory::VISION_BYTES));
        assert!(plan.recorded.iter().all(|s| s.destination.is_none() && s.kind == Kind::Recorded));
        assert!(plan.markers.iter().all(|s| s.destination.is_none() && s.name.ends_with(".mul1")));
        // regions: the global one, and a trunk and an experts region per layer
        assert_eq!(plan.regions.len(), 1 + 40 * 2);
        assert!(plan.regions.iter().any(|r| r.name == "global") && plan.regions.iter().any(|r| r.name == "layer_39_experts"));
        // every upload item sits aligned inside its region
        let mut ends: BTreeMap<u32, u64> = BTreeMap::new();
        let mut spans: BTreeMap<u32, Vec<(u64, u64)>> = BTreeMap::new();
        for s in &plan.items {
            let d = s.destination.as_ref().unwrap();
            assert_eq!(d.offset % ALIGN, 0, "{}", s.name);
            assert!(d.offset + d.bytes <= plan.regions[d.region as usize].bytes, "{}", s.name);
            spans.entry(d.region).or_default().push((d.offset, d.offset + d.bytes));
            let e = ends.entry(d.region).or_default();
            *e = (*e).max(d.offset + d.bytes);
        }
        for (r, mut v) in spans {
            v.sort_unstable();
            assert!(v.windows(2).all(|w| w[0].1 <= w[1].0), "region {r}: items overlap");
            assert_eq!(plan.regions[r as usize].bytes, ends[&r].div_ceil(ALIGN) * ALIGN);
        }
    }

    #[test]
    fn the_head_is_padded_and_its_valid_width_is_the_vocabulary() {
        let map = WeightMap::bind(&real(), &table()).unwrap();
        let lm = &map.groups["lm_head"];
        assert_eq!((lm.k, lm.n, lm.bits.twice()), (2048, 248_192, 12));
        assert_eq!((map.lm_head_valid(), map.lm_head_width), (248_144, 248_192));
        // the embedding is the real vocabulary, bf16 as stored
        let e = &map.slots["model.language_model.embed_tokens.weight"];
        assert_eq!((e.shape.clone(), e.transform, e.dtype.as_str()), (vec![248_144, 2048], Transform::None, "BF16"));
    }

    #[test]
    fn each_norm_and_parameter_has_the_conversion_the_reference_gives_it() {
        let map = WeightMap::bind(&real(), &table()).unwrap();
        let t = |n: &str| map.slots[n].transform;
        let p = "model.language_model.layers";
        // `(1 + w)`: the trunk's norms and the attention's q/k norms
        assert_eq!(t(&format!("{p}.0.input_layernorm.weight")), Transform::Bf16AddOneToF32);
        assert_eq!(t(&format!("{p}.39.post_attention_layernorm.weight")), Transform::Bf16AddOneToF32);
        assert_eq!(t("model.language_model.norm.weight"), Transform::Bf16AddOneToF32);
        assert_eq!(t(&format!("{p}.3.self_attn.q_norm.weight")), Transform::Bf16AddOneToF32);
        assert_eq!(t(&format!("{p}.7.self_attn.k_norm.weight")), Transform::Bf16AddOneToF32);
        // the gated RMSNorm of the delta net has no +1; its decay parameters are widened as they are
        assert_eq!(t(&format!("{p}.0.linear_attn.norm.weight")), Transform::Bf16ToF32);
        assert_eq!(t(&format!("{p}.0.linear_attn.A_log")), Transform::Bf16ToF32);
        assert_eq!(t(&format!("{p}.2.linear_attn.dt_bias")), Transform::Bf16ToF32);
        // copied as stored: the convolution, the fp16 small matrices, the embedding
        for n in [
            format!("{p}.0.linear_attn.conv1d.weight"),
            format!("{p}.0.linear_attn.in_proj_a.weight"),
            format!("{p}.0.linear_attn.in_proj_b.weight"),
            format!("{p}.5.mlp.gate.weight"),
            format!("{p}.5.mlp.shared_expert_gate.weight"),
            "model.language_model.embed_tokens.weight".to_string(),
        ] {
            assert_eq!(t(&n), Transform::None, "{n}");
        }
        // a widened tensor takes twice its stored bytes on the device
        let n = &map.slots[&format!("{p}.0.input_layernorm.weight")];
        assert_eq!(n.destination.as_ref().unwrap().bytes, 2048 * 4);
        let c = &map.slots[&format!("{p}.0.linear_attn.conv1d.weight")];
        assert_eq!(c.destination.as_ref().unwrap().bytes, 8192 * 4 * 2);
        // and the device dtype the stager asks the pack for is what the table states
        for s in map.slots.values().filter(|s| s.kind == Kind::Plain) {
            let (dtype, _, _) = specification(&s.name, &s.dtype);
            assert_eq!(s.dtype, dtype, "{}", s.name);
        }
    }

    #[test]
    fn region_sizes_are_hand_summed_from_the_shapes() {
        let map = WeightMap::bind(&real(), &table()).unwrap();
        let size = |n: &str| map.regions.iter().find(|r| r.name == n).unwrap().bytes;
        let a = |n: u64| n.div_ceil(256) * 256;
        // the table's nominal widths: experts 4 bit (trellis 128 x 32 x 64 words), attention / GDN / shared / head 6 bit
        let expert = |k: u64, n: u64| a(k * n * 4 / 8) + a(k * 2) + a(n * 2);
        let six = |k: u64, n: u64| a(k * n * 6 / 8) + a(k * 2) + a(n * 2);
        // a GDN layer's experts: 256 x (gate, up, down)
        assert_eq!(size("layer_0_experts"), 256 * (2 * expert(2048, 512) + expert(512, 2048)));
        // its trunk: two norms (f32), qkv + z + out, a and b (fp16 [32, 2048]), conv (bf16 [8192, 1, 4], copied as stored), A_log, dt_bias, norm,
        // the router (fp16 [256, 2048]), the shared expert's three groups and its gate (fp16 [1, 2048])
        let trunk = 2 * a(2048 * 4)
            + six(2048, 8192)
            + six(2048, 4096)
            + six(4096, 2048)
            + 2 * a(32 * 2048 * 2)
            + a(8192 * 4 * 2)
            + 2 * a(32 * 4)
            + a(128 * 4)
            + a(256 * 2048 * 2)
            + 2 * six(2048, 512)
            + six(512, 2048)
            + a(2048 * 2);
        assert_eq!(size("layer_0_trunk"), trunk);
        // an attention layer's trunk: q (2048 -> 8192, query and gate), k, v, o, q_norm and k_norm widened
        let attn = 2 * a(2048 * 4)
            + six(2048, 8192)
            + 2 * six(2048, 512)
            + six(4096, 2048)
            + 2 * a(256 * 4)
            + a(256 * 2048 * 2)
            + 2 * six(2048, 512)
            + six(512, 2048)
            + a(2048 * 2);
        assert_eq!(size("layer_3_trunk"), attn);
        // the global region: the embedding (bf16, as stored), the final norm (f32) and the padded head
        assert_eq!(
            size("global"),
            a(248_144 * 2048 * 2) + a(2048 * 4) + a(2048 * 248_192 * 6 / 8) + a(2048 * 2) + a(248_192 * 2)
        );
        // the whole upload, and the payload without alignment is no larger
        assert!(map.payload_bytes() <= map.upload_bytes());
        assert_eq!(map.upload_bytes(), map.regions.iter().map(|r| r.bytes).sum::<u64>());
        assert_eq!(map.load_plan().upload_bytes(), map.upload_bytes());
    }

    #[test]
    fn a_mixed_width_expert_changes_its_region_and_its_group() {
        let mut t = table();
        // expert 7 of layer 5 at 5 bits (trellis last dimension 80)
        for p in ["gate_proj", "up_proj"] {
            t.get_mut(&format!("model.language_model.layers.5.mlp.experts.7.{p}.trellis")).unwrap().1 = vec![128, 32, 80];
        }
        let base = WeightMap::bind(&real(), &table()).unwrap();
        let mixed = WeightMap::bind(&real(), &t).unwrap();
        let g = &mixed.groups["model.language_model.layers.5.mlp.experts.7.up_proj"];
        assert_eq!((g.k, g.n, g.bits.twice()), (2048, 512, 10));
        let r = |m: &WeightMap| m.regions.iter().find(|r| r.name == "layer_5_experts").unwrap().bytes;
        assert_eq!(r(&mixed) - r(&base), 2 * (2048 * 512 / 8));
        let h = mixed.bit_histogram();
        assert_eq!(h["5"], 2 + base.bit_histogram().get("5").copied().unwrap_or(0));
    }

    #[test]
    fn a_pack_that_disagrees_with_the_recipe_is_refused_by_name() {
        let cfg = real();
        // a missing tensor
        let mut t = table();
        t.remove("model.language_model.layers.7.self_attn.v_proj.trellis");
        let e = WeightMap::bind(&cfg, &t).unwrap_err();
        assert!(has(&e, "layers.7.self_attn.v_proj: missing EXL3 linear"), "{e:?}");
        // a missing plain tensor
        let mut t = table();
        t.remove("model.language_model.layers.12.mlp.shared_expert_gate.weight");
        let e = WeightMap::bind(&cfg, &t).unwrap_err();
        assert!(has(&e, "layers.12.mlp.shared_expert_gate.weight: missing"), "{e:?}");
        // the wrong shape: an expert's down projection, the router, and the convolution
        let mut t = table();
        t.get_mut("model.language_model.layers.20.mlp.experts.123.down_proj.trellis").unwrap().1 = vec![64, 128, 64];
        t.get_mut("model.language_model.layers.1.mlp.gate.weight").unwrap().1 = vec![255, 2048];
        t.get_mut("model.language_model.layers.2.linear_attn.conv1d.weight").unwrap().1 = vec![10240, 1, 4];
        let e = WeightMap::bind(&cfg, &t).unwrap_err();
        assert!(has(&e, "layers.20.mlp.experts.123.down_proj.trellis: shape [64, 128, 64]"), "{e:?}");
        assert!(has(&e, "layers.1.mlp.gate.weight: shape [255, 2048]"), "{e:?}");
        assert!(has(&e, "layers.2.linear_attn.conv1d.weight: shape [10240, 1, 4]"), "{e:?}");
        // the wrong bits: a 9-bit trellis, and a 2.5-bit one (the kernels run integer widths 1 to 8)
        let mut t = table();
        t.get_mut("model.language_model.layers.4.mlp.experts.0.gate_proj.trellis").unwrap().1 = vec![128, 32, 144];
        t.get_mut("model.language_model.layers.4.mlp.experts.1.gate_proj.trellis").unwrap().1 = vec![128, 32, 40];
        let e = WeightMap::bind(&cfg, &t).unwrap_err();
        assert!(has(&e, "layers.4.mlp.experts.0.gate_proj.trellis: shape [128, 32, 144]"), "{e:?}");
        assert!(has(&e, "layers.4.mlp.experts.1.gate_proj.trellis: shape [128, 32, 40]"), "{e:?}");
        // the `.mcg` codebook marker of another family of packs in place of `.mul1`
        let mut t = table();
        let m = t.remove("model.language_model.layers.3.self_attn.k_proj.mul1").unwrap();
        t.insert("model.language_model.layers.3.self_attn.k_proj.mcg".into(), m);
        let e = WeightMap::bind(&cfg, &t).unwrap_err();
        assert!(has(&e, "layers.3.self_attn.k_proj.mul1: missing (the codebook marker)"), "{e:?}");
        assert!(has(&e, "layers.3.self_attn.k_proj.mcg: not part of the MiniCPM-V-4.7 recipe"), "{e:?}");
        // the head at the unpadded vocabulary (what a converter that does not pad writes), and a padded embedding
        let mut t = table();
        t.get_mut("lm_head.trellis").unwrap().1 = vec![128, 15_509, 96];
        t.get_mut("lm_head.svh").unwrap().1 = vec![248_144];
        let e = WeightMap::bind(&cfg, &t).unwrap_err();
        assert!(has(&e, "lm_head.trellis: shape [128, 15509, 96]") && has(&e, "zero-padded"), "{e:?}");
        let mut t = table();
        t.get_mut("model.language_model.embed_tokens.weight").unwrap().1 = vec![248_192, 2048];
        let e = WeightMap::bind(&cfg, &t).unwrap_err();
        assert!(has(&e, "embed_tokens.weight: shape [248192, 2048]"), "{e:?}");
        // an MTP head, which this checkpoint does not ship and the family does not run
        let mut t = table();
        t.insert("mtp.fc.weight".into(), ("BF16".into(), vec![2048, 4096]));
        let e = WeightMap::bind(&cfg, &t).unwrap_err();
        assert!(has(&e, "mtp.fc.weight: present, but this checkpoint ships no MTP head"), "{e:?}");
        // a norm in the wrong dtype, a tower tensor missing, and a tensor outside the recipe
        let mut t = table();
        t.get_mut("model.language_model.layers.0.input_layernorm.weight").unwrap().0 = "F16".into();
        t.remove("model.vision_tower.post_layernorm.weight");
        t.insert("model.language_model.rotary_emb.inv_freq".into(), ("F32".into(), vec![32]));
        let e = WeightMap::bind(&cfg, &t).unwrap_err();
        assert!(has(&e, "layers.0.input_layernorm.weight: dtype F16, the recipe needs BF16"), "{e:?}");
        assert!(has(&e, "vision_tower.post_layernorm.weight: missing"), "{e:?}");
        assert!(has(&e, "rotary_emb.inv_freq: not part of the MiniCPM-V-4.7 recipe"), "{e:?}");
    }

    /// A directory holding one safetensors shard of the given tensors (name, dtype, shape, bytes), as the converter writes them.
    fn write_shard(tag: &str, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("rexl3-minicpm-weights-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut head = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            head.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let text = serde_json::Value::Object(head).to_string().into_bytes();
        let mut file = (text.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&text);
        file.extend_from_slice(&data);
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        dir
    }

    /// The scan-level refusals, on files: a group written with the `mcg` codebook (the converter default of other packs), and a group
    /// the scan itself refuses (a trellis that is not int16). Both come with the recipe's list, by group name.
    #[test]
    fn a_scanned_pack_with_an_mcg_or_a_refused_group_is_named() {
        let trellis = vec![0u8; 8 * 8 * 64 * 2];
        let (mcg, mul1) = (0xcbac_1fedu32.to_le_bytes().to_vec(), gb_exl3::MARKER_MUL1.to_le_bytes().to_vec());
        let scales = || vec![0u8; 256];
        let dir = write_shard(
            "mcg",
            &[
                // a well-formed `mcg` group: the scan accepts it as one, and the bind refuses the codebook by group name
                ("lm_head.trellis", "I16", vec![8, 8, 64], trellis.clone()),
                ("lm_head.suh", "F16", vec![128], scales()),
                ("lm_head.svh", "F16", vec![128], scales()),
                ("lm_head.mcg", "I32", vec![], mcg),
                // a trellis that is not int16: the scan refuses the group
                ("model.language_model.layers.0.mlp.experts.0.up_proj.trellis", "F16", vec![8, 8, 64], trellis.clone()),
                ("model.language_model.layers.0.mlp.experts.0.up_proj.suh", "F16", vec![128], scales()),
                ("model.language_model.layers.0.mlp.experts.0.up_proj.svh", "F16", vec![128], scales()),
                ("model.language_model.layers.0.mlp.experts.0.up_proj.mul1", "I32", vec![], mul1.clone()),
                // a `mul1` marker holding another value (the scan reads every marker)
                ("model.language_model.layers.1.mlp.experts.0.gate_proj.trellis", "I16", vec![8, 8, 64], trellis),
                ("model.language_model.layers.1.mlp.experts.0.gate_proj.suh", "F16", vec![128], scales()),
                ("model.language_model.layers.1.mlp.experts.0.gate_proj.svh", "F16", vec![128], scales()),
                ("model.language_model.layers.1.mlp.experts.0.gate_proj.mul1", "I32", vec![], vec![0u8; 4]),
            ],
        );
        let ck = gb_exl3::checkpoint::scan(&dir, true).unwrap();
        let e = WeightMap::from_checkpoint(&ck, &real()).unwrap_err();
        let head = &e[..e.len().min(5)];
        assert!(has(&e, "lm_head: codebook mcg (the engine runs `mul1`: convert with `-cb mul1`)"), "{head:?}");
        assert!(has(&e, "layers.0.mlp.experts.0.up_proj: refused EXL3 group: trellis must be int16"), "{head:?}");
        assert!(has(&e, "layers.1.mlp.experts.0.gate_proj: refused EXL3 group: .mul1 marker holds 0x0"), "{head:?}");
        // and the recipe's own list rides along (the pack is nowhere near whole)
        assert!(has(&e, "lm_head.mul1: missing (the codebook marker)"), "{head:?}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn the_smoke_pack_of_eight_layers_binds_with_its_own_layer_list() {
        // layers 0..8 of the table: GDN at 0, 1, 2, 4, 5, 6 and attention at 3 and 7
        let t = only_layers(&table(), 8);
        assert_eq!(t.len(), 8 * inventory::TENSORS_PER_LAYER + 6 + 459);
        assert_eq!(t.len(), 25_313);
        let cfg = truncated(8);
        let map = WeightMap::bind(&cfg, &t).unwrap();
        assert_eq!(map.slots.len(), 25_313);
        assert_eq!(map.groups.len(), 6 * 774 + 2 * 775 + 1);
        // the 40-layer config refuses the 8-layer pack, naming what is missing
        let e = WeightMap::bind(&real(), &t).unwrap_err();
        assert!(has(&e, "layers.8.input_layernorm.weight: missing"), "{:?}", &e[..e.len().min(3)]);
        // and the 8-layer config refuses a pack that carries layer 8
        let e = WeightMap::bind(&cfg, &table()).unwrap_err();
        assert!(
            has(&e, "layers.8.input_layernorm.weight: present, but the config has fewer layers"),
            "{:?}",
            &e[..e.len().min(3)]
        );
    }
}
