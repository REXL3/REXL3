//! The vision tower's half of the weight map, on CPU: the bf16 tower of the 2.25 bpw pack (`testdata/glm53`) and the EXL3 tower of the
//! sc_4.10bpw pack (exllamav3 1.6.2; `testdata/glm53_sc410_vision/vision_inventory.tsv`, the tower's 1,007 tensors as the pack's headers name
//! them). The sc_4.10bpw inventory here is the 2.25 bpw table without its `model.visual.*` rows plus those 1,007 (the text side of the two
//! packs has the same names, dtypes and shapes; only trellis widths differ), and its config is the 2.25 bpw one with
//! `quantization_config.vision_bits` 6 (the two configs differ only in `quantization_config`).
//!
//! First the guards of the two hazards of an fp16 / EXL3 tower: (a) the device's in-place bf16 -> fp16 conversion must never touch an fp16
//! tensor, and (b) an EXL3 group bias the kernels read as `const half*` must be fp16 or be refused by name.

use super::bind::{Problem, VISION_PREFIX};
use super::config::{GlmConfig, VisionTowerFormat};
use super::inventory::PackInventory;
use super::plan::RegionKind;
use super::vision;
use super::weights::*;
use std::sync::OnceLock;

const TABLE: &str = include_str!("../../../../../testdata/glm53/tensor_inventory.tsv");
const SC410_VISION: &str = include_str!("../../../../../testdata/glm53_sc410_vision/vision_inventory.tsv");
const CONFIG: &str = include_str!("../../../../../testdata/glm53/config.json");

fn config() -> GlmConfig {
    GlmConfig::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
}

fn sc410_config_json() -> serde_json::Value {
    let mut v: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    v["quantization_config"]["vision_bits"] = serde_json::json!(6);
    v
}

fn sc410_config() -> GlmConfig {
    GlmConfig::from_json(&sc410_config_json()).unwrap()
}

fn inventory() -> PackInventory {
    static INV: OnceLock<PackInventory> = OnceLock::new();
    INV.get_or_init(|| PackInventory::from_table(TABLE).unwrap()).clone()
}

/// The 2.25 bpw table's text rows and the sc_4.10bpw tower's rows.
fn sc410_inventory() -> PackInventory {
    static INV: OnceLock<PackInventory> = OnceLock::new();
    INV.get_or_init(|| {
        let text: String = TABLE.lines().filter(|l| !l.starts_with(VISION_PREFIX)).map(|l| format!("{l}\n")).collect();
        PackInventory::from_table(&format!("{text}{SC410_VISION}")).unwrap()
    })
    .clone()
}

fn all() -> gb_cuda::moe::ExpertOwnership {
    gb_cuda::moe::ExpertOwnership::all(288)
}

fn sc410() -> &'static GlmWeights {
    static W: OnceLock<GlmWeights> = OnceLock::new();
    W.get_or_init(|| GlmWeights::map(&sc410_inventory(), &sc410_config()).unwrap())
}

fn refused_sc410(inv: &PackInventory) -> Vec<Problem> {
    GlmWeights::map(inv, &sc410_config()).expect_err("this mutation must be refused").problems
}

fn visual_tensors(inv: &PackInventory) -> usize {
    let plain = inv.plain.keys().filter(|n| n.starts_with(VISION_PREFIX)).count();
    let parts: usize = inv
        .groups
        .iter()
        .filter(|(n, _)| n.starts_with(VISION_PREFIX))
        .map(|(_, g)| 3 + g.bias.is_some() as usize + g.marker.is_some() as usize)
        .sum();
    plain + parts
}

// ------------------------------------------------------------------------------------------------- hazard (a): no bf16 decode of fp16

#[test]
fn hazard_a_only_bf16_tensors_are_converted_on_the_device() {
    // the bf16 tower: every tensor the device converts in place is a bf16 tensor of the plan, and they are the Linear / Conv / LayerNorm ones
    let w = GlmWeights::map(&inventory(), &config()).unwrap();
    let v = w.vision.as_ref().unwrap();
    assert_eq!(v.format, VisionTowerFormat::Bf16);
    let conv = v.bf16_conversions();
    assert!(conv.iter().all(|t| t.dtype == PlainDtype::Bf16 && t.xform == Xform::None), "only bf16 tensors are converted");
    // 2 biases (patch, downsample) + 24 x 10 block tensors + 4 merger weights + the merger LayerNorm's 2
    assert_eq!(conv.len(), 2 + 24 * 10 + 4 + 2);
    assert!(conv
        .iter()
        .all(|t| !t.name.ends_with("norm1.weight") && !t.name.contains("q_norm") && !t.name.contains("post_layernorm")));

    // the EXL3 tower: nothing is converted, every plain tensor is fp16 as stored or a bf16 RMSNorm gamma, and the patch and downsample
    // weights reach the device through the fp16 host fold and permutation, never through a bf16 path
    let v = sc410().vision.as_ref().unwrap();
    assert_eq!(v.format, VisionTowerFormat::Exl3 { bits: 6 });
    assert!(v.bf16_conversions().is_empty(), "an fp16 tensor reinterpreted as bf16 decodes garbage");
    let mut plain = vec![];
    v.visit(&mut |l| {
        if let Leaf::Plain(t) = l {
            plain.push(t.clone())
        }
    });
    for t in &plain {
        let gamma = t.name.ends_with("norm1.weight")
            || t.name.ends_with("norm2.weight")
            || t.name.ends_with("q_norm.weight")
            || t.name.ends_with("k_norm.weight")
            || t.name.ends_with("post_layernorm.weight");
        match (gamma, t.dtype) {
            (true, PlainDtype::Bf16) | (false, PlainDtype::F16) => {}
            other => panic!("{}: {other:?}", t.name),
        }
        let want = if t.name.ends_with("patch_embed.proj.weight") {
            Xform::FoldPatchF16
        } else if t.name.ends_with("downsample.weight") {
            Xform::PermuteDownsampleF16
        } else {
            Xform::None
        };
        assert_eq!(t.xform, want, "{}", t.name);
    }
}

#[test]
fn hazard_a_an_fp16_tower_tensor_in_a_bf16_tower_is_refused_by_name() {
    // a bf16-tower config over a pack whose merger norm is fp16: the dtype check refuses it before any device conversion could run
    let mut inv = inventory();
    let name = "model.visual.merger.post_projection_norm.weight";
    inv.plain.get_mut(name).unwrap().dtype = "F16".into();
    let p = GlmWeights::map(&inv, &config()).unwrap_err().problems;
    assert_eq!(p, vec![Problem::Dtype { name: name.into(), want: "BF16", got: "F16".into() }]);
    // and the sc_4.10bpw pack under a config without `vision_bits` (the bf16 tower) does not map: its tower is refused by name
    let mut v = sc410_config_json();
    v["quantization_config"].as_object_mut().unwrap().remove("vision_bits");
    let p = GlmWeights::map(&sc410_inventory(), &GlmConfig::from_json(&v).unwrap()).unwrap_err().problems;
    assert!(p.contains(&Problem::Missing { name: "model.visual.blocks.0.attn.proj.weight".into(), group: false }), "{p:?}");
    assert!(p.contains(&Problem::Unexpected { name: "model.visual.blocks.0.attn.q_proj".into(), group: true }), "{p:?}");
    assert!(p.contains(&Problem::Dtype { name: "model.visual.patch_embed.proj.weight".into(), want: "BF16", got: "F16".into() }));
}

// ------------------------------------------------------------------------------------------------- hazard (b): fp16 group biases only

#[test]
fn hazard_b_a_vision_group_bias_that_is_not_fp16_is_refused_by_name() {
    for (dtype, size) in [("BF16", 2u64), ("F32", 4)] {
        let mut inv = sc410_inventory();
        let prefix = "model.visual.blocks.7.mlp.gate_proj";
        let b = inv.groups.get_mut(prefix).unwrap().bias.as_mut().unwrap();
        b.dtype = dtype.into();
        b.bytes = 4096 * size;
        assert_eq!(
            refused_sc410(&inv),
            vec![Problem::Dtype { name: format!("{prefix}.bias"), want: "F16", got: dtype.into() }],
            "{dtype}"
        );
    }
    // a vision group without the bias the config's `attention_bias` implies is missing it, by name
    let mut inv = sc410_inventory();
    inv.groups.get_mut("model.visual.blocks.3.attn.k_proj").unwrap().bias = None;
    assert_eq!(
        refused_sc410(&inv),
        vec![Problem::Missing { name: "model.visual.blocks.3.attn.k_proj.bias".into(), group: false }]
    );
    // the merger's groups have no bias; one with a bias is refused (only the block linears take one)
    let mut inv = sc410_inventory();
    let some_bias = inv.groups["model.visual.blocks.0.attn.proj"].bias.clone();
    inv.groups.get_mut("model.visual.merger.proj").unwrap().bias = some_bias;
    assert_eq!(
        refused_sc410(&inv),
        vec![Problem::Unsupported { prefix: "model.visual.merger.proj".into(), why: "it has a bias".into() }]
    );
}

#[test]
fn hazard_b_text_groups_still_refuse_a_bias() {
    let mut inv = sc410_inventory();
    let prefix = "model.language_model.layers.0.self_attn.o_proj";
    let some_bias = inv.groups["model.visual.blocks.0.attn.proj"].bias.clone();
    inv.groups.get_mut(prefix).unwrap().bias = some_bias;
    assert_eq!(refused_sc410(&inv), vec![Problem::Unsupported { prefix: prefix.into(), why: "it has a bias".into() }]);
}

// ------------------------------------------------------------------------------------------------- the EXL3 tower

#[test]
fn the_vision_bits_of_the_config_choose_the_tower_format() {
    assert_eq!(config().vision.unwrap().tower, VisionTowerFormat::Bf16);
    assert_eq!(sc410_config().vision.unwrap().tower, VisionTowerFormat::Exl3 { bits: 6 });
    for bad in [serde_json::json!(4.5), serde_json::json!(0), serde_json::json!(9), serde_json::json!("6")] {
        let mut v = sc410_config_json();
        v["quantization_config"]["vision_bits"] = bad.clone();
        let e = GlmConfig::from_json(&v).unwrap_err();
        assert!(e.contains("vision_bits"), "{bad}: {e}");
    }
}

#[test]
fn the_sc410_tower_maps_every_tensor_once() {
    let inv = sc410_inventory();
    let w = sc410();
    let v = w.vision.as_ref().unwrap();
    let vc = sc410_config().vision.unwrap();
    assert_eq!(visual_tensors(&inv), 1_007);
    assert_eq!(vc.tensor_count(), 1_007);
    // the tree's vision leaves: per block 4 gammas + 7 groups + 7 biases, then patch 2, post norm 1, downsample 2, merger 4 groups + 2 norms
    let (mut groups, mut plain) = (0, 0);
    v.visit(&mut |l| match l {
        Leaf::Plain(_) => plain += 1,
        Leaf::Exl3(_) => groups += 1,
    });
    assert_eq!((groups, plain), (24 * 7 + 4, 24 * (4 + 7) + 2 + 1 + 2 + 2));
    // the fused bf16 qkv is claimed (dtype and shape checked) and not loaded: 48 tensors in `skipped`, nothing else
    assert_eq!(w.skipped.len(), 48);
    assert!(w.skipped.iter().all(|s| !s.group && s.name.contains(".attn.qkv.") && s.why.contains("exllamav3 1.6.2")));
    // every group is 6-bit with a mul1 codebook; block groups carry an fp16 bias, the merger's none
    for b in &v.blocks {
        let VisionQkv::Split(qkv) = &b.qkv else { panic!("the EXL3 tower splits q, k and v") };
        for l in qkv.iter().chain([&b.proj, &b.gate_proj, &b.up_proj, &b.down_proj]) {
            let VisionLinear::Exl3 { group, bias } = l else { panic!("{}: not EXL3", l.name()) };
            assert_eq!((group.bits.twice(), group.codebook), (12, gb_exl3::Codebook::Mul1), "{}", group.prefix);
            let bias = bias.as_ref().unwrap();
            assert_eq!((bias.dtype, bias.shape.as_slice()), (PlainDtype::F16, [group.n].as_slice()));
        }
    }
    for l in [&v.merger.proj, &v.merger.gate_proj, &v.merger.up_proj, &v.merger.down_proj] {
        assert!(matches!(l, VisionLinear::Exl3 { bias: None, .. }), "{}", l.name());
    }
}

#[test]
fn the_qkv_biases_are_one_span_in_the_region() {
    let v = sc410().vision.as_ref().unwrap();
    for b in &v.blocks {
        let VisionQkv::Split(qkv) = &b.qkv else { unreachable!() };
        let bs: Vec<&PlainTensor> = qkv.iter().map(|l| l.bias().unwrap()).collect();
        for w in bs.windows(2) {
            assert_eq!(w[1].dst.region, w[0].dst.region);
            assert_eq!(w[1].dst.offset, w[0].dst.offset + w[0].dst_bytes, "{} follows {}", w[1].name, w[0].name);
        }
    }
}

#[test]
fn the_exl3_tower_region_is_the_memory_models_arithmetic() {
    let w = sc410();
    let vc = sc410_config().vision.unwrap();
    let plan = w.load_plan().unwrap();
    let region: u64 = plan.regions.iter().filter(|r| r.kind == RegionKind::Vision).map(|r| r.bytes).sum();
    assert_eq!(region, vision::weight_region_bytes(&vc));
    // the uploaded bytes are the config's weight bytes (no fused qkv, no markers), about 0.42 GiB against the bf16 tower's 1.05 GiB
    let uploaded: u64 = plan.items.iter().filter(|i| i.category == gb_exl3::budget::Category::Vision).map(|i| i.bytes).sum();
    assert_eq!(uploaded, vc.weight_bytes());
    assert!(uploaded < 450 << 20 && uploaded > 400 << 20, "{uploaded}");
    let bf16 = config().vision.unwrap();
    assert!(vision::weight_region_bytes(&bf16) > 2 * region);
    // the 172 groups' codebook markers are checks of the tower's plan (the bf16 tower has none)
    let markers = plan.markers.iter().filter(|m| m.name.starts_with(VISION_PREFIX)).count();
    assert_eq!(markers, 24 * 7 + 4);
}

#[test]
fn the_exl3_tower_scratch_adds_its_staging() {
    let (bf16, exl3) = (VisionTowerFormat::Bf16, VisionTowerFormat::Exl3 { bits: 6 });
    let a = vision::scratch_items(32_000, bf16);
    let b = vision::scratch_items(32_000, exl3);
    assert_eq!(&b[..a.len()], &a[..], "the bf16 tower's buffers, then the staging");
    let extra: Vec<(&str, u64)> = b[a.len()..].to_vec();
    let rows = vision::EXL3_ROWS as u64;
    assert_eq!(
        extra,
        vec![
            ("exl3_w", 4096 * 10240 * 2),
            ("exl3_x", rows * 10240 * 2),
            ("exl3_xh", rows * 10240 * 2),
            ("exl3_c", rows * 10240 * 4),
            ("exl3_y", rows * 10240 * 2),
        ]
    );
    // 180 MiB whatever the image's size, against the 640 MiB of weights the EXL3 tower saves
    assert_eq!(extra.iter().map(|i| i.1).sum::<u64>(), 180 << 20);
    // a small cap stages fewer rows
    let small = vision::scratch_items(256, exl3);
    assert_eq!(small.iter().find(|i| i.0 == "exl3_c").unwrap().1, 256 * 10240 * 4);
    // the config's encoder scratch is the tower's
    let vc = sc410_config().vision.unwrap();
    assert_eq!(vc.encoder_scratch_bytes(8_000), vision::scratch_bytes(32_000, exl3));
}

#[test]
fn a_vision_group_of_another_width_is_refused() {
    // vision_bits 6, one group 5-bit: named, not silently priced at 6 bits
    let mut v = sc410_config_json();
    v["quantization_config"]["vision_bits"] = serde_json::json!(5);
    let p = GlmWeights::map(&sc410_inventory(), &GlmConfig::from_json(&v).unwrap()).unwrap_err().problems;
    assert_eq!(p.len(), 24 * 7 + 4);
    assert!(p.iter().all(
        |p| matches!(p, Problem::Unsupported { why, .. } if why.contains("6-bit tiles") && why.contains("vision_bits` is 5"))
    ));
}

#[test]
fn the_fused_qkv_is_optional_and_checked_when_present() {
    // a pack without the redundant fused copy maps the same tower
    let mut inv = sc410_inventory();
    inv.plain.retain(|n, _| !n.contains(".attn.qkv."));
    let w = GlmWeights::map(&inv, &sc410_config()).unwrap();
    assert!(w.skipped.is_empty());
    assert_eq!(w.vision, sc410().vision);
    // present with the wrong shape: claimed, and refused by name
    let mut inv = sc410_inventory();
    inv.plain.get_mut("model.visual.blocks.2.attn.qkv.weight").unwrap().shape = vec![3072, 1023];
    assert_eq!(
        refused_sc410(&inv),
        vec![Problem::Shape {
            name: "model.visual.blocks.2.attn.qkv.weight".into(),
            want: vec![3072, 1024],
            got: vec![3072, 1023]
        }]
    );
}

// ------------------------------------------------------------------------------------------------- --no-vision

#[test]
fn no_vision_claims_and_skips_the_tower_of_either_format() {
    for (inv, cfg, tensors) in [(inventory(), config(), 347), (sc410_inventory(), sc410_config(), 1_007)] {
        assert_eq!(visual_tensors(&inv), tensors);
        let w = GlmWeights::map_with(&inv, &cfg, &all(), false).unwrap();
        assert!(w.vision.is_none());
        // one record per plain tensor and per group, all of the tower, nothing else
        let n_plain = inv.plain.keys().filter(|n| n.starts_with(VISION_PREFIX)).count();
        let n_groups = inv.groups.keys().filter(|n| n.starts_with(VISION_PREFIX)).count();
        assert_eq!(w.skipped.len(), n_plain + n_groups);
        assert!(w.skipped.iter().all(|s| s.name.starts_with(VISION_PREFIX) && s.why.contains("--no-vision")));
        // no vision region, no vision item, no vision marker: nothing of the tower is read or charged
        let plan = w.load_plan().unwrap();
        assert!(plan.regions.iter().all(|r| r.kind != RegionKind::Vision));
        assert!(plan.items.iter().all(|i| !i.name.starts_with(VISION_PREFIX)));
        assert!(plan.markers.iter().all(|m| !m.name.starts_with(VISION_PREFIX)));
        // the text model is the one the full map has
        let full = GlmWeights::map(&inv, &cfg).unwrap();
        assert_eq!(
            (w.layers.clone(), w.mtp.clone(), w.lm_head.clone()),
            (full.layers.clone(), full.mtp.clone(), full.lm_head.clone())
        );
    }
}

#[test]
fn no_vision_still_refuses_an_unknown_tensor_outside_the_tower() {
    let mut inv = sc410_inventory();
    let extra = inv.plain["model.language_model.norm.weight"].clone();
    inv.plain.insert("model.language_model.stray.weight".into(), extra.clone());
    // and a stray tensor under the tower's prefix is claimed with the tower
    inv.plain.insert("model.visual.stray.weight".into(), extra);
    let p = GlmWeights::map_with(&inv, &sc410_config(), &all(), false).unwrap_err().problems;
    assert_eq!(p, vec![Problem::Unexpected { name: "model.language_model.stray.weight".into(), group: false }]);
    // with the tower bound, the tower's stray is named too
    let p = GlmWeights::map(&inv, &sc410_config()).unwrap_err().problems;
    assert!(p.contains(&Problem::Unexpected { name: "model.visual.stray.weight".into(), group: false }), "{p:?}");
}

#[test]
fn no_vision_maps_a_tower_the_binder_would_refuse() {
    // a vision group the scan refused and a tower tensor of the wrong dtype do not stop a text-only map
    let mut inv = sc410_inventory();
    inv.refused.insert("model.visual.blocks.1.attn.v_proj".into(), "2.5-bit tiles need the mul1 codebook".into());
    inv.plain.get_mut("model.visual.downsample.weight").unwrap().dtype = "BF16".into();
    assert!(GlmWeights::map(&inv, &sc410_config()).is_err());
    let w = GlmWeights::map_with(&inv, &sc410_config(), &all(), false).unwrap();
    assert!(w.vision.is_none());
    assert!(w.skipped.iter().any(|s| s.name == "model.visual.blocks.1.attn.v_proj" && s.group));
    // a refused text group still stops it
    inv.refused.insert("lm_head".into(), "x".into());
    assert!(GlmWeights::map_with(&inv, &sc410_config(), &all(), false).is_err());
}

#[test]
fn a_rank_maps_the_same_text_with_and_without_the_tower() {
    // a TP follower maps with its expert share and no tower: the text regions and their items are the vision-mapped rank's, byte for byte
    let own = gb_cuda::moe::ExpertOwnership::range(288, 1, 2).unwrap();
    let a = GlmWeights::map_with(&sc410_inventory(), &sc410_config(), &own, true).unwrap().load_plan().unwrap();
    let b = GlmWeights::map_with(&sc410_inventory(), &sc410_config(), &own, false).unwrap().load_plan().unwrap();
    let text = |p: &super::plan::LoadPlan| {
        let keep = super::load::keep_prefix_mtp(45, Some(45));
        let s = p.select_regions(keep);
        (s.regions.iter().map(|r| (r.kind, r.bytes)).collect::<Vec<_>>(), s.items.len(), s.markers.len())
    };
    assert_eq!(text(&a), text(&b));
}

// ------------------------------------------------------------------------------------------------- the host conversions

fn f16_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|&x| gb_exl3::f16::f64_to_f16(x).to_le_bytes()).collect()
}

fn f16_values(b: &[u8]) -> Vec<f64> {
    b.chunks_exact(2).map(|c| gb_exl3::f16::f16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect()
}

#[test]
fn the_fp16_patch_fold_sums_the_two_slots_exactly_once_rounded() {
    use super::load::convert;
    let (o, k, kp) = (3usize, vision::PATCH_K, vision::PATCH_KP);
    // [o, 3, 2, 196]: slot 0 and slot 1 with values whose fp16 sum needs a rounding
    let mut src = vec![0f64; o * 2 * k];
    for r in 0..o {
        for c in 0..3 {
            for t in 0..2 {
                for i in 0..196 {
                    let x = (r * 1000 + c * 300 + i) as f64 / 7.0 * if t == 0 { 1.0 } else { -0.0009765625 };
                    src[r * 2 * k + c * 392 + t * 196 + i] = gb_exl3::f16::f16_to_f64(gb_exl3::f16::f64_to_f16(x));
                }
            }
        }
    }
    let out = f16_values(&convert(Xform::FoldPatchF16, &f16_bytes(&src)).unwrap());
    assert_eq!(out.len(), src.len(), "the slot keeps its size");
    for r in 0..o {
        for kk in 0..kp {
            let want = if kk < k {
                let (c, i) = (kk / 196, kk % 196);
                let s = src[r * 2 * k + c * 392 + i] + src[r * 2 * k + c * 392 + 196 + i];
                gb_exl3::f16::f16_to_f64(gb_exl3::f16::f64_to_f16(s))
            } else {
                0.0
            };
            assert_eq!(out[r * kp + kk].to_bits(), want.to_bits(), "row {r} col {kk}");
        }
    }
    assert!(out[o * kp..].iter().all(|&x| x == 0.0), "the tail of the slot is zero");
    // not a whole number of rows, or a stored inf: refused
    assert!(convert(Xform::FoldPatchF16, &f16_bytes(&src[1..])).is_err());
    let mut bad = src.clone();
    bad[5] = f64::INFINITY;
    assert!(convert(Xform::FoldPatchF16, &f16_bytes(&bad)).unwrap_err().contains("not finite"));
}

#[test]
fn the_fp16_downsample_permutation_is_the_kernels_merge_order() {
    use super::load::convert;
    let (o, c) = (2usize, vision::HIDDEN);
    let src: Vec<f64> = (0..o * c * 4).map(|i| (i % 2000) as f64).collect();
    let out = f16_values(&convert(Xform::PermuteDownsampleF16, &f16_bytes(&src)).unwrap());
    for r in 0..o {
        for ch in 0..c {
            for j in 0..4 {
                // the conv's [O, C, kh, kw] element (r, ch, j) is the GEMM's column j * C + ch: the merged patches' order
                assert_eq!(out[r * 4 * c + j * c + ch], src[r * 4 * c + ch * 4 + j]);
            }
        }
    }
    assert!(convert(Xform::PermuteDownsampleF16, &f16_bytes(&src[..c])).is_err());
}
