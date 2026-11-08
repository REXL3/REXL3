//! The real GLM-5.3-Flash packs, mapped on the CPU: no weight is read and no CUDA context is created, only the shard headers.
//! Set REXL3_GLM_DIR to the 2.25bpw pack (31 shards) and REXL3_GLM_SC410_DIR to the `sc_4.10bpw` pack (turboderp c6b8ac3f, 23 shards,
//! exllamav3 1.6.2); each pack's tests skip without its variable:
//!
//! `REXL3_GLM_DIR=~/models/r0b0tlab/GLM-5.3-Flash-EXL3-2.25bpw-sm121 cargo test --release -p gb-model --test glm_pack -- --nocapture`
//! `REXL3_GLM_SC410_DIR=~/projects/glm53-flash-exl3-sc410-c6b8ac3f cargo test --release -p gb-model --test glm_pack -- --nocapture`

use gb_cuda::moe::ExpertOwnership;
use gb_exl3::budget::{categorize, Bytes, Category};
use gb_model::families::glm::plan::bytes_of;
use gb_model::families::glm::tp_boot::{rank_weight_bytes, rank_weight_bytes_with};
use gb_model::families::glm::weights::{FfnWeights, GlmWeights, MixerWeights};
use gb_model::families::glm::{GlmConfig, GlmFamily, GlmPack, PackInventory, RegionKind};
use gb_model::pack::Pack;
use std::collections::BTreeMap;
use std::path::PathBuf;

const TABLE: &str = include_str!("../../../testdata/glm53/tensor_inventory.tsv");
const SC410_TABLE: &str = include_str!("../../../testdata/glm53-sc410/tensor_inventory.tsv");

fn pack_dir() -> Option<PathBuf> {
    std::env::var("REXL3_GLM_DIR").ok().map(PathBuf::from).filter(|p| p.join("config.json").exists())
}

fn sc410_dir() -> Option<PathBuf> {
    let d = std::env::var("REXL3_GLM_SC410_DIR").ok().map(PathBuf::from).filter(|p| p.join("config.json").exists());
    if d.is_none() {
        eprintln!("REXL3_GLM_SC410_DIR not set: not run");
    }
    d
}

/// Every tensor name of an inventory with its dtype and shape (a group's parts included).
fn tensors(inv: &PackInventory) -> BTreeMap<String, (String, Vec<usize>)> {
    let mut m = BTreeMap::new();
    for (n, t) in &inv.plain {
        m.insert(n.clone(), (t.dtype.clone(), t.shape.clone()));
    }
    for (p, g) in &inv.groups {
        for (suffix, t) in [
            ("trellis", Some(&g.trellis)),
            (g.in_scales, Some(&g.in_scale)),
            (g.out_scales, Some(&g.out_scale)),
            ("bias", g.bias.as_ref()),
            (g.codebook.name(), g.marker.as_ref()),
        ] {
            if let Some(t) = t {
                m.insert(format!("{p}.{suffix}"), (t.dtype.clone(), t.shape.clone()));
            }
        }
    }
    m
}

#[test]
fn the_committed_inventory_is_the_real_pack() {
    let Some(dir) = pack_dir() else { return eprintln!("REXL3_GLM_DIR not set: skipping") };
    let pack = Pack::open(&dir).unwrap();
    assert_eq!((pack.ck.groups.len(), pack.ck.plain.len(), pack.ck.bad.len()), (37_420, 1_214, 0));
    let real = PackInventory::from_checkpoint(&pack.ck).unwrap();
    let table = PackInventory::from_table(TABLE).unwrap();
    let (a, b) = (tensors(&real), tensors(&table));
    let diff: Vec<_> =
        a.keys().filter(|k| a.get(*k) != b.get(*k)).chain(b.keys().filter(|k| !a.contains_key(*k))).take(5).collect();
    assert!(diff.is_empty() && a.len() == b.len(), "the committed table drifted from the pack at {diff:?}");
    for (p, g) in &real.groups {
        let t = &table.groups[p];
        assert_eq!(
            (g.k, g.n, g.bits, g.codebook, g.in_scales, g.out_scales),
            (t.k, t.n, t.bits, t.codebook, t.in_scales, t.out_scales),
            "{p}"
        );
    }
    // the six tensors stored in two shards are resolved by the index, and nothing else is noteworthy
    assert_eq!(real.issues.len(), 6, "{:?}", real.issues);
    assert!(real.issues.iter().all(|i| i.contains("as the index says")));
}

#[test]
fn the_real_pack_maps_and_plans() {
    let Some(dir) = pack_dir() else { return eprintln!("REXL3_GLM_DIR not set: skipping") };
    let t0 = std::time::Instant::now();
    let pack = Pack::open(&dir).unwrap();
    let scan = t0.elapsed();
    let cfg = GlmConfig::from_dir(&dir).unwrap();
    let t1 = std::time::Instant::now();
    let inv = PackInventory::from_checkpoint(&pack.ck).unwrap();
    let mapped = GlmPack::from_inventory(&inv, cfg.clone()).unwrap();
    let map = t1.elapsed();
    let (w, plan) = (&mapped.weights, &mapped.plan);
    assert_eq!(w.counts(), (37_420, 1_214), "every group and every plain tensor lands in one slot");
    eprintln!("scan {scan:?} (31 shard headers, markers read), inventory + map + plan {map:?}");

    // ---- bytes: the plan against `exl3-inspect`'s classification, which reads names; the plan's reads slots
    let mut on_disk = Bytes::default();
    for (name, g) in &pack.ck.groups {
        on_disk.add(categorize(name, Some(45)), g.bytes);
    }
    for (name, e) in &pack.ck.plain {
        on_disk.add(categorize(name, Some(45)), e.nbytes());
    }
    let t = plan.totals();
    let mut got = t.uploaded;
    for c in [
        Category::RoutedExpert,
        Category::Decoder,
        Category::Head,
        Category::Other,
        Category::Mtp,
        Category::Embedding,
        Category::Vision,
    ] {
        got.add(c, bytes_of(&t.skipped, c));
    }
    assert_eq!(got, on_disk, "uploaded + markers == bytes on disk, per category");
    let gb = |b: u64| format!("{:.3}", b as f64 / 1e9);
    eprintln!(
        "uploaded GB: routed {} decoder {} head {} mtp {} embedding {} vision {} other {} (always-read {}); markers not uploaded: {} B",
        gb(t.uploaded.routed),
        gb(t.uploaded.decoder),
        gb(t.uploaded.head),
        gb(t.uploaded.mtp),
        gb(t.uploaded.embedding),
        gb(t.uploaded.vision),
        gb(t.uploaded.other),
        gb(t.uploaded.always_read()),
        t.skipped_total()
    );
    assert_eq!(
        (gb(on_disk.routed), gb(on_disk.mtp), gb(on_disk.embedding), gb(on_disk.vision)),
        ("87.419".into(), "3.756".into(), "1.269".into(), "1.127".into())
    );
    assert_eq!(gb(on_disk.always_read()), "4.827");
    // against the memory model slice 1 shipped: it counts every group's bytes (markers included) and every plain tensor
    let slice1 = gb_model::families::glm::pack_weight_bytes(&pack.ck, true);
    assert_eq!(
        slice1,
        on_disk.routed + on_disk.decoder + on_disk.head + on_disk.other + on_disk.mtp + on_disk.embedding + on_disk.vision
    );
    // what the items occupy on the device: the same bytes, except the plain tensors the loader converts (norm gammas bf16 -> fp32 grow,
    // each mHC `fn` fp32 -> fp16 halves: 90 matrices of 1.5 MiB)
    let on_device: u64 = plan.items.iter().map(|i| i.dst_bytes).sum();
    let converted: u64 = plan.items.iter().filter(|i| i.xform != gb_model::families::glm::Xform::None).count() as u64;
    eprintln!(
        "slice-1 weights {slice1} B; uploaded {} B (-{} B of markers); items on the device {on_device} B ({converted} converted); \
         device regions {} B (+{} B alignment padding)",
        t.uploaded_total(),
        t.skipped_total(),
        t.region_bytes,
        t.region_bytes - on_device
    );
    assert_eq!(t.uploaded_total() + t.skipped_total(), slice1);
    assert!(t.region_bytes - on_device < 4 << 20, "alignment padding is a few MB at most");
    // the conversions the plan owes the executor, by kind: gammas (45 layers x 2 norms, 11 MLA layers x 2, the final norm, the MTP
    // layer's 3 + its 4), the 90 mHC `fn`, the 43 routers' biases (layers 3-44 and the MTP layer), the 12 `kv_b_proj` (11 + the MTP layer)
    let count = |x: gb_model::families::glm::Xform| plan.items.iter().filter(|i| i.xform == x).count();
    use gb_model::families::glm::Xform;
    assert_eq!(
        (count(Xform::Bf16ToF32), count(Xform::F32ToF16Exact), count(Xform::CentredBias), count(Xform::PackKvB)),
        (90 + 22 + 1 + 3 + 4, 90, 43, 12)
    );
    assert_eq!(converted, 120 + 90 + 43 + 12);

    // ---- bits, read from the trellis shapes
    let three = [3usize, 4, 5, 14, 24, 33, 39, 40, 41, 42, 43, 44];
    let mut layers_by_bits: BTreeMap<u8, Vec<usize>> = BTreeMap::new();
    for b in w.banks() {
        assert_eq!(b.bits_gate_up, b.bits_down, "layer {}: gate, up and down share a width", b.layer);
        assert_eq!(b.experts.len(), 288);
        // the bank region is exactly the bank: 288 x (three trellises of (4096 x 2048 x bits / 8) bytes + six scale vectors)
        let region = plan.region(b.region).unwrap();
        assert_eq!(region.bytes, 288 * (3 * (1u64 << 20) * (b.bits_gate_up.twice() as u64 / 2) + 36_864), "layer {}", b.layer);
        layers_by_bits.entry(b.bits_gate_up.twice() / 2).or_default().push(b.layer);
    }
    assert_eq!(layers_by_bits[&3], three);
    assert_eq!(layers_by_bits[&4], vec![45]);
    assert_eq!(layers_by_bits[&2], (3..45).filter(|l| !three.contains(l)).collect::<Vec<_>>());
    // exl3-inspect: decoder groups are 3-bit x34, 4-bit x222, 5-bit x2; the MTP head is 4-bit x873; lm_head is 6-bit
    let mut hist: BTreeMap<(Category, u8), usize> = BTreeMap::new();
    w.visit(&mut |l| {
        if let gb_model::families::glm::weights::Leaf::Exl3(g) = l {
            assert!(g.bits.is_integer() && g.codebook == gb_exl3::Codebook::Mul1);
            *hist.entry((g.category, g.bits.twice() / 2)).or_default() += 1;
        }
    });
    assert_eq!(hist[&(Category::Decoder, 3)], 34);
    assert_eq!(hist[&(Category::Decoder, 4)], 222);
    assert_eq!(hist[&(Category::Decoder, 5)], 2);
    assert_eq!(hist[&(Category::Mtp, 4)], 873);
    assert_eq!(hist[&(Category::Head, 6)], 1);
    assert_eq!(hist[&(Category::RoutedExpert, 2)], 25_920);
    assert_eq!(hist[&(Category::RoutedExpert, 3)], 10_368);
    eprintln!("routed layers by width: {layers_by_bits:?}");

    // ---- the typed slots hold what the pack holds
    let FfnWeights::Moe(m3) = &w.layers[3].ffn else { panic!("layer 3 is MoE") };
    let MixerWeights::Mla(mla3) = &w.layers[3].mixer else { panic!("layer 3 is MLA") };
    assert_eq!((mla3.kv_b_proj.shape.clone(), m3.router.shape.clone()), (vec![32768, 512], vec![288, 4096]));
    assert_eq!(w.mtp.len(), 1);
    assert!(w.mtp[0].layer.mhc.is_none() && w.layers.iter().all(|l| l.mhc.is_some()));

    // ---- the tower is a component of its own: 347 bf16 tensors, and exactly the bytes slice 1's formula says
    let v = cfg.vision.as_ref().unwrap();
    let tower = plan.select(|c| c == Category::Vision);
    assert_eq!(tower.items.len(), 347);
    assert_eq!(tower.totals().uploaded.vision, v.weight_bytes());
    assert_eq!(tower.regions.len(), 1);
    assert!(matches!(tower.regions[0].kind, RegionKind::Vision));
    let text_only = plan.select(|c| c != Category::Vision);
    assert_eq!(text_only.items.len() + tower.items.len(), plan.items.len());

    // ---- layout facts of the shards the pipeline relies on
    let (mut multi_file, mut contiguous, mut groups) = (0usize, 0usize, 0usize);
    for g in inv.groups.values() {
        groups += 1;
        let (s, v, m, t) = (&g.in_scale, &g.out_scale, g.marker.as_ref().unwrap(), &g.trellis);
        if [v.file, m.file, t.file].iter().any(|f| *f != s.file) {
            multi_file += 1;
        }
        if v.offset == s.offset + s.bytes && m.offset == v.offset + v.bytes && t.offset == m.offset + m.bytes && s.file == t.file
        {
            contiguous += 1;
        }
    }
    eprintln!("groups: {groups}; parts in more than one shard: {multi_file}; suh|svh|mul1|trellis contiguous: {contiguous}");
    assert_eq!(multi_file, 0, "every group lives in one shard");
    let extents = plan.extents(16, 64 << 20);
    let long = extents.iter().filter(|e| e.items.len() > 3).count();
    eprintln!(
        "read extents (gap <= 16 B, <= 64 MiB): {} for {} items, {} of them spanning more than one group; plan regions: {}",
        extents.len(),
        plan.items.len(),
        long,
        plan.regions.len()
    );
    assert!(extents.len() < 2000, "the pack's layout lets a pipeline issue about 1,500 reads, not 113 thousand");
}

#[test]
fn the_family_maps_the_real_pack_and_accepts_images() {
    let Some(dir) = pack_dir() else { return eprintln!("REXL3_GLM_DIR not set: not run") };
    let p = GlmFamily::map_pack(&dir).unwrap();
    // `map_pack` does not read the markers; the scan that does yields the same plan (the markers are checks the plan carries)
    let verified = GlmPack::from_checkpoint(&Pack::open(&dir).unwrap().ck, GlmConfig::from_dir(&dir).unwrap()).unwrap();
    assert_eq!(p.plan, verified.plan);
    assert_eq!(p.plan.markers.len(), 37_420);
    assert_eq!(p.weights.counts(), (37_420, 1_214));
    assert_eq!(p.plan.items.len(), 37_420 * 3 + 1_214);
    let mut o = gb_model::serve::ServeOptions::new(dir.clone());
    o.vision = Some(Default::default());
    // the text model and the tower are built: the preflight passes with and without images (it reads no tensor)
    assert_eq!(gb_model::serve::preflight(&o).unwrap(), "glm5_next");
    o.vision = None;
    assert_eq!(gb_model::serve::preflight(&o).unwrap(), "glm5_next");
    // the family's image preprocessing is the pack's processor config at the Fork profile, limited to the server's tokens
    let mut with = gb_model::serve::ServeOptions::new(dir.clone());
    with.max_image_tokens = 6000;
    let pp = gb_model::serve::image_preprocessor(&with).unwrap().unwrap();
    assert!(
        pp.id().starts_with("glm5-next-vision-v") && pp.id().contains(":fork:") && pp.id().contains("tok16-6000"),
        "{}",
        pp.id()
    );
    assert!(pp.plan(4000, 6000).unwrap().tokens <= 6000);
}

#[test]
fn a_pack_with_a_tensor_removed_is_refused_by_name() {
    let Some(dir) = pack_dir() else { return eprintln!("REXL3_GLM_DIR not set: skipping") };
    let pack = Pack::open(&dir).unwrap();
    let mut inv = PackInventory::from_checkpoint(&pack.ck).unwrap();
    let name = "model.language_model.layers.44.mlp.gate.e_score_correction_bias";
    inv.plain.remove(name).unwrap();
    let e = GlmWeights::map(&inv, &GlmConfig::from_dir(&dir).unwrap()).unwrap_err();
    assert_eq!(e.problems.len(), 1, "{e}");
    assert!(e.to_string().contains(name), "{e}");
}

// ---------------------------------------------------------------------------------- the sc_4.10bpw pack (c6b8ac3f, exllamav3 1.6.2)

const TOWER: &str = "model.visual.";

/// The text side of an inventory and a config: the tower's tensors and `vision_config` removed (lane G-V owns the EXL3 tower).
fn text_side(inv: &PackInventory, cfg: &GlmConfig) -> (PackInventory, GlmConfig) {
    let mut inv = inv.clone();
    inv.groups.retain(|p, _| !p.starts_with(TOWER));
    inv.plain.retain(|n, _| !n.starts_with(TOWER));
    let mut cfg = cfg.clone();
    cfg.vision = None;
    (inv, cfg)
}

/// The committed `testdata/glm53-sc410` inventory is the real pack's, tensor by tensor; the scan refuses no group (the tower's 172
/// biased and unbiased K6 groups included, so the fingerprint covers them) and notices nothing: no tensor of this pack is stored in two
/// shards (the 2.25bpw pack has six).
#[test]
fn the_committed_sc410_inventory_is_the_real_pack() {
    let Some(dir) = sc410_dir() else { return };
    let pack = Pack::open(&dir).unwrap();
    assert_eq!((pack.ck.groups.len(), pack.ck.plain.len(), pack.ck.bad.len()), (37_592, 1_018, 0), "{:?}", pack.ck.bad);
    assert_eq!(pack.ck.groups.values().filter(|g| g.bias).count(), 168);
    let real = PackInventory::from_checkpoint(&pack.ck).unwrap();
    assert_eq!(real.files.len(), 23);
    assert_eq!(real.tensor_count(), 151_554);
    let table = PackInventory::from_table(SC410_TABLE).unwrap();
    let (a, b) = (tensors(&real), tensors(&table));
    let diff: Vec<_> =
        a.keys().filter(|k| a.get(*k) != b.get(*k)).chain(b.keys().filter(|k| !a.contains_key(*k))).take(5).collect();
    assert!(diff.is_empty() && a.len() == b.len(), "the committed table drifted from the pack at {diff:?}");
    for (p, g) in &real.groups {
        let t = &table.groups[p];
        assert_eq!(
            (g.k, g.n, g.bits, g.codebook, g.in_scales, g.out_scales, g.bias.is_some()),
            (t.k, t.n, t.bits, t.codebook, t.in_scales, t.out_scales, t.bias.is_some()),
            "{p}"
        );
    }
    // no tensor in two shards, no other header issue
    assert!(real.issues.is_empty() && pack.ck.issues.is_empty(), "{:?} {:?}", real.issues, pack.ck.issues);
    let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let q = &raw["quantization_config"];
    assert_eq!((q["version"].as_str(), q["bits"].as_f64(), q["vision_bits"].as_u64()), (Some("1.6.2"), Some(4.11), Some(6)));
}

/// RED-SNOW-style: the text side of the real pack maps with no refusal, one node and both ranks of a TP = 2 group (a rank reads only its
/// experts); the whole pack maps or refuses only the tower (true before and after lane G-V).
#[test]
fn the_text_side_of_the_real_sc410_pack_maps_on_one_node_and_on_each_rank() {
    let Some(dir) = sc410_dir() else { return };
    let pack = Pack::open(&dir).unwrap();
    let inv = PackInventory::from_checkpoint(&pack.ck).unwrap();
    let cfg = GlmConfig::from_dir(&dir).unwrap();
    match GlmPack::from_inventory(&inv, cfg.clone()) {
        Ok(p) => eprintln!("the whole pack maps (the tower included): {:?}", p.weights.counts()),
        Err(gb_model::families::glm::GlmPackError::Layout(e)) => {
            let outside: Vec<String> =
                e.problems.iter().map(|p| p.to_string()).filter(|s| !s.contains(&format!("`{TOWER}"))).take(5).collect();
            assert!(outside.is_empty(), "{} problem(s), outside the tower: {outside:?}", e.problems.len());
            eprintln!("the whole pack: {} problem(s), every one the tower's (lane G-V)", e.problems.len());
        }
        Err(e) => panic!("{e}"),
    }
    let (tinv, tcfg) = text_side(&inv, &cfg);
    let one = GlmPack::from_inventory(&tinv, tcfg.clone()).unwrap_or_else(|e| panic!("the text side must map: {e}"));
    assert_eq!(one.weights.counts(), (37_420, 867));
    one.plan.validate().unwrap();
    // bytes: the plan against `exl3-inspect`'s name classification
    let mut on_disk = Bytes::default();
    for (name, g) in pack.ck.groups.iter().filter(|(n, _)| !n.starts_with(TOWER)) {
        on_disk.add(categorize(name, Some(45)), g.bytes);
    }
    for (name, e) in pack.ck.plain.iter().filter(|(n, _)| !n.starts_with(TOWER)) {
        on_disk.add(categorize(name, Some(45)), e.nbytes());
    }
    let t = one.plan.totals();
    let mut got = t.uploaded;
    for c in [Category::RoutedExpert, Category::Decoder, Category::Head, Category::Other, Category::Mtp, Category::Embedding] {
        got.add(c, bytes_of(&t.skipped, c));
    }
    assert_eq!(got, on_disk, "uploaded + markers == bytes on disk, per category");
    for b in one.weights.banks() {
        let want = if b.layer == 3 { 5.0 } else { 4.0 };
        assert_eq!((b.bits_gate_up.as_f64(), b.bits_down.as_f64(), b.experts.len()), (want, want, 288), "layer {}", b.layer);
    }
    let (mut multi_file, mut contiguous, mut groups) = (0usize, 0usize, 0usize);
    for g in tinv.groups.values() {
        groups += 1;
        let (s, v, m, t) = (&g.in_scale, &g.out_scale, g.marker.as_ref().unwrap(), &g.trellis);
        multi_file += [v.file, m.file, t.file].iter().any(|f| *f != s.file) as usize;
        contiguous +=
            (v.offset == s.offset + s.bytes && m.offset == v.offset + v.bytes && t.offset == m.offset + m.bytes) as usize;
    }
    assert_eq!((groups, multi_file), (37_420, 0), "every group lives in one shard");
    let extents = one.plan.extents(16, 64 << 20);
    eprintln!(
        "text side: {} plan items, {} read extents (gap <= 16 B, <= 64 MiB), {} regions; suh|svh|mul1|trellis contiguous: {contiguous}",
        one.plan.items.len(),
        extents.len(),
        one.plan.regions.len()
    );
    // 2,645 on 2026-10-17 (the 2.25bpw pack: under 2,000): still about a thousandth of the 113 thousand items
    assert!(extents.len() < 3000, "{} read extents", extents.len());
    // each rank of a TP = 2 group: only its experts, the trunk replicated, the two shares the single node's experts
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let mut shares = 0u64;
    for r in 0..2 {
        let own = ExpertOwnership::range(288, r, 2).unwrap();
        let p = GlmPack::from_inventory_owned(&tinv, tcfg.clone(), &own).unwrap();
        let foreign = p.plan.items.iter().filter(|i| i.name.contains(".mlp.experts.")).any(|i| {
            let e: usize = i.name.split(".mlp.experts.").nth(1).unwrap().split('.').next().unwrap().parse().unwrap();
            !own.owns(e)
        });
        assert!(!foreign, "rank {r} reads an expert it does not own");
        let (bytes, with_mtp) = (rank_weight_bytes(&p), rank_weight_bytes_with(&p, true));
        shares +=
            p.plan.regions.iter().filter(|x| matches!(x.kind, RegionKind::Experts(l) if l < 45)).map(|x| x.bytes).sum::<u64>();
        eprintln!("rank {r}: {:.2} GiB of text weights, {:.2} GiB with its NextN share", gib(bytes), gib(with_mtp));
        assert!(
            (80.0..81.5).contains(&gib(bytes)) && (82.0..83.5).contains(&gib(with_mtp)),
            "rank {r}: {:.2} / {:.2}",
            gib(bytes),
            gib(with_mtp)
        );
    }
    let whole: u64 =
        one.plan.regions.iter().filter(|x| matches!(x.kind, RegionKind::Experts(l) if l < 45)).map(|x| x.bytes).sum();
    assert!((shares as i64 - whole as i64).unsigned_abs() < 64 << 20, "ranks {shares} vs one node {whole}");
}
