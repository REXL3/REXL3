//! The RED-SNOW 4.91bpw GLM-5.3-Flash pack mapped on the CPU: only the 36 shard headers are read, no CUDA context is created. It differs
//! from the 2.25bpw pack in three ways the family must take or refuse by name: routed experts of 4, 5 and 6 bits MIXED inside a layer (one
//! with gate != up), non-expert linears of 5 to 8 bits, and the DSA indexer's `wq_b` stored as plain fp16 instead of an EXL3 group. Set
//! `REXL3_REDSNOW_DIR` to the pack; every test prints "not run" and passes without it.
//!
//! `REXL3_REDSNOW_DIR=~/models/vcruz305/RED-SNOW-5.3-FLASH-EXL3-SAGE-4.91bpw cargo test --release -p gb-model --test glm_redsnow -- --nocapture`

use gb_cuda::moe::ExpertOwnership;
use gb_model::families::glm::tp_boot::{plan_from, rank_weight_bytes};
use gb_model::families::glm::weights::{FfnWeights, IndexerWqB, MixerWeights};
use gb_model::families::glm::{GlmConfig, GlmPack, PackInventory, RegionKind};
use gb_model::pack::Pack;
use gb_model::serve::{PrefixCacheOption, ServeOptions};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const GIB: f64 = (1u64 << 30) as f64;

fn pack_dir() -> Option<PathBuf> {
    let d = std::env::var("REXL3_REDSNOW_DIR").ok().map(PathBuf::from).filter(|p| p.join("config.json").exists());
    if d.is_none() {
        eprintln!("REXL3_REDSNOW_DIR not set: not run");
    }
    d
}

/// The whole pack maps with NO refusal: every tensor of the 36 shards is claimed by a slot (the mapper fails closed on an unclaimed one),
/// the 12 indexer `wq_b` are plain fp16, every EXL3 group is a width and codebook the kernels take (whole bits 1..=8, mul1), every expert
/// width class is listed, and the layers that bind as subbanks on one node are the mixed ones.
#[test]
fn the_whole_pack_maps_with_no_refusal() {
    let Some(dir) = pack_dir() else { return };
    let pack = Pack::open(&dir).unwrap();
    assert_eq!(pack.ck.bad.len(), 0, "the scan refused groups: {:?}", pack.ck.bad);
    let inv = PackInventory::from_checkpoint(&pack.ck).unwrap();
    let cfg = GlmConfig::from_dir(&dir).unwrap();
    let p = GlmPack::from_inventory(&inv, cfg.clone()).unwrap_or_else(|e| panic!("RED-SNOW must map: {e}"));
    eprintln!(
        "mapped: {} EXL3 groups, {} plain tensors, {} plan items, {} regions",
        inv.groups.len(),
        inv.plain.len(),
        p.plan.items.len(),
        p.plan.regions.len()
    );
    let w = &p.weights;
    // the indexer's wq_b: plain fp16 in every MLA layer of the trunk and in the NextN layer
    let mut plain_wq_b = 0;
    for l in w.layers.iter().chain(w.mtp.iter().map(|m| &m.layer)) {
        if let MixerWeights::Mla(m) = &l.mixer {
            let ix = m.indexer.as_ref().expect("every MLA layer of this pack has its own indexer");
            match &ix.wq_b {
                IndexerWqB::PlainF16(t) => {
                    assert_eq!(t.shape, vec![4096, 1536], "layer {}", l.index);
                    plain_wq_b += 1;
                }
                IndexerWqB::Exl3(_) => panic!("layer {}: wq_b is an EXL3 group in this pack?", l.index),
            }
        }
    }
    assert_eq!(plain_wq_b, 12);
    // every EXL3 group's width and codebook (non-expert linears and experts)
    let mut widths: BTreeMap<String, BTreeSet<u8>> = BTreeMap::new();
    for (prefix, g) in &inv.groups {
        let class = if prefix.contains(".mlp.experts.") {
            "routed experts"
        } else if prefix.contains("shared_experts") {
            "shared experts"
        } else if prefix.contains(".mlp.") {
            "dense MLP"
        } else if prefix.starts_with("lm_head") {
            "lm_head"
        } else if prefix.contains("eh_proj") {
            "NextN eh_proj"
        } else {
            "attention"
        };
        assert!(g.bits.twice() % 2 == 0 && (2..=16).contains(&g.bits.twice()), "{prefix}: width {}", g.bits.twice() as f64 / 2.0);
        assert_eq!(g.codebook, gb_exl3::Codebook::Mul1, "{prefix}");
        widths.entry(class.into()).or_default().insert(g.bits.twice() / 2);
    }
    for (c, b) in &widths {
        eprintln!("  {c}: bits {b:?}, codebook mul1");
    }
    // the experts: per layer, the subbank keys; the NextN layer and layer 3 are uniform, every other layer mixes
    let mut mixed = Vec::new();
    let mut gate_ne_up = Vec::new();
    for b in w.banks() {
        assert!(b.own.is_all());
        if b.subbanks.len() > 1 {
            mixed.push(b.layer);
        }
        for sb in &b.subbanks {
            if sb.key.bits_gate != sb.key.bits_up {
                gate_ne_up.push((b.layer, sb.key, sb.ids.clone()));
            }
        }
    }
    eprintln!("  mixed-width expert layers (subbanks on one node): {} of 43; gate != up: {gate_ne_up:?}", mixed.len());
    assert_eq!(mixed.len(), 41);
    assert!(!mixed.contains(&3) && !mixed.contains(&45));
    assert_eq!(gate_ne_up.len(), 1);
    assert_eq!(gate_ne_up[0].0, 13);
    // the dense layers and every MoE layer's shared expert are present
    for l in &w.layers {
        match &l.ffn {
            FfnWeights::Dense(_) => assert!(l.index < 3),
            FfnWeights::Moe(m) => assert!(m.shared.is_some(), "layer {} has a shared expert", l.index),
        }
    }
    assert!(w.vision.is_some(), "the vision tower maps (it is not loaded with --no-vision)");
    // the extra config keys nothing consumes (metadata of how the pack was made: no tensor of the pack carries them)
    let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    fn keys_with(v: &serde_json::Value, needle: &str, path: &str, out: &mut Vec<String>) {
        if let Some(o) = v.as_object() {
            for (k, x) in o {
                let p = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
                if k.contains(needle) {
                    out.push(p.clone());
                }
                keys_with(x, needle, &p, out);
            }
        }
    }
    let mut extra = Vec::new();
    keys_with(&raw, "dwm", "", &mut extra);
    eprintln!("  config keys no slot consumes (metadata of the pack's making, no tensor carries them): {extra:?}");
}

/// A rank of a two-rank group maps only its share (`ExpertOwnership::range(288, r, 2)`): the bytes of the regions it allocates are the
/// replicated trunk and its experts (no NextN layer, no vision), about 97.5 GiB; the two ranks' expert regions add up to the whole
/// pack's; and the rank planner's numbers at the kit's flags.
#[test]
fn a_rank_maps_and_plans_only_its_share() {
    let Some(dir) = pack_dir() else { return };
    let pack = Pack::open(&dir).unwrap();
    let inv = PackInventory::from_checkpoint(&pack.ck).unwrap();
    let cfg = GlmConfig::from_dir(&dir).unwrap();
    let whole = GlmPack::from_inventory(&inv, cfg.clone()).unwrap();
    let kinds = |p: &GlmPack, with_mtp: bool| -> (u64, u64, u64) {
        let (mut g, mut t, mut e) = (0, 0, 0);
        for r in &p.plan.regions {
            match r.kind {
                RegionKind::Global => g += r.bytes,
                RegionKind::Trunk(l) if l < 45 || with_mtp => t += r.bytes,
                RegionKind::Experts(l) if l < 45 || with_mtp => e += r.bytes,
                _ => {}
            }
        }
        (g, t, e)
    };
    let w = kinds(&whole, false);
    let w_mtp = kinds(&whole, true);
    eprintln!(
        "one node (every expert, text, no NextN layer): global {:.2} + trunk {:.2} + experts {:.2} = {:.2} GiB; the NextN layer adds {:.2} GiB",
        w.0 as f64 / GIB,
        w.1 as f64 / GIB,
        w.2 as f64 / GIB,
        (w.0 + w.1 + w.2) as f64 / GIB,
        ((w_mtp.1 + w_mtp.2) - (w.1 + w.2)) as f64 / GIB
    );
    assert_eq!(rank_weight_bytes(&whole), w.0 + w.1 + w.2, "the NextN layer is not counted");
    let mut experts = 0;
    for r in 0..2 {
        let own = ExpertOwnership::range(288, r, 2).unwrap();
        let p = GlmPack::from_inventory_owned(&inv, cfg.clone(), &own).unwrap();
        let k = kinds(&p, false);
        let bytes = rank_weight_bytes(&p);
        assert_eq!(bytes, k.0 + k.1 + k.2);
        assert_eq!((k.0, k.1), (w.0, w.1), "the trunk is replicated");
        experts += k.2;
        // the rank's plan reads only its experts
        let foreign = p.plan.items.iter().filter(|i| i.name.contains(".mlp.experts.")).any(|i| {
            let e: usize = i.name.split(".mlp.experts.").nth(1).unwrap().split('.').next().unwrap().parse().unwrap();
            !own.owns(e)
        });
        assert!(!foreign, "rank {r} reads an expert it does not own");
        eprintln!(
            "rank {r} (experts {}..={}): {:.2} GiB = replicated {:.2} + experts {:.2} (subbanks per layer: {:?})",
            own.owned()[0],
            own.owned()[own.local() - 1],
            bytes as f64 / GIB,
            (k.0 + k.1) as f64 / GIB,
            k.2 as f64 / GIB,
            p.weights.banks().filter(|b| b.layer < 45).map(|b| b.subbanks.len()).collect::<Vec<_>>()
        );
        // the header arithmetic's 97.5 GiB a rank counts the NextN layer's expert share and the vision tower, which a --spec none
        // --no-vision rank never allocates; the device figure here is the regions' (norms widened, mHC `fn` narrowed)
        assert!((90.0..97.6).contains(&(bytes as f64 / GIB)), "rank {r}: {:.2} GiB", bytes as f64 / GIB);
    }
    // the ranks' expert regions are the whole pack's experts, split (plus each subbank's alignment)
    let slack = experts as i64 - w.2 as i64;
    eprintln!(
        "expert regions: ranks {:.3} GiB vs one node {:.3} GiB (alignment {slack} B)",
        experts as f64 / GIB,
        w.2 as f64 / GIB
    );
    assert!(slack.unsigned_abs() < 64 << 20);

    // the least free memory (before the load) at which the rank planner grants the kit's flags: the rank's weights, its runtime (KV pool,
    // KDA state, scratch) and the serving floor
    let kit = |free: u64| {
        let mut o = ServeOptions::new(dir.clone());
        (o.slots, o.slots_auto, o.max_model_len, o.kv_tokens, o.prefill_chunk) = (2, false, 8192, Some(16384), 512);
        o.prefix_cache = PrefixCacheOption::Off;
        o.vision = None;
        let ck = gb_exl3::checkpoint::scan(&dir, false).unwrap();
        let floor = o.floor_bytes;
        plan_from(&o, cfg.clone(), ck, free, 0.0, Some(ExpertOwnership::range(288, 0, 2).unwrap()))
            .ok()
            .filter(|p| p.mo.kv_blocks * 256 >= 16384)
            .map(|p| (p.weight_bytes, floor))
    };
    let (mut lo, mut hi) = (90u64 << 30, 130u64 << 30);
    assert!(kit(hi).is_some() && kit(lo).is_none());
    while hi - lo > 64 << 20 {
        let mid = (lo + hi) / 2;
        if kit(mid).is_some() {
            hi = mid
        } else {
            lo = mid
        }
    }
    let (wb, floor) = kit(hi).unwrap();
    eprintln!(
        "the kit's flags (--slots 2 --max-model-len 8192 --kv-tokens 16384 --prefill-chunk 512) plan from {:.2} GiB free: weights {:.2} + \
         runtime {:.2} + floor {:.2} GiB; on a node with 117 GiB available, about {:.1} GiB stay available while the group runs",
        hi as f64 / GIB,
        wb as f64 / GIB,
        (hi - wb - floor) as f64 / GIB,
        floor as f64 / GIB,
        117.0 - (hi - floor) as f64 / GIB
    );

    // the rank planner at the kit's flags (slots, context, pool) on 117 GiB free before the load
    for (slots, len, kv) in [(2usize, 4096usize, 8192usize), (2, 8192, 16384), (4, 8192, 16384), (4, 8192, 32768)] {
        let mut o = ServeOptions::new(dir.clone());
        o.slots = slots;
        o.slots_auto = false;
        o.max_model_len = len;
        o.kv_tokens = Some(kv);
        o.prefill_chunk = 512;
        o.prefix_cache = PrefixCacheOption::Off;
        o.vision = None;
        let ck = gb_exl3::checkpoint::scan(&dir, false).unwrap();
        let own = ExpertOwnership::range(288, 0, 2).unwrap();
        match plan_from(&o, cfg.clone(), ck, 117 << 30, 0.0, Some(own)) {
            Ok(p) => eprintln!(
                "planner at 117 GiB, --slots {slots} --max-model-len {len} --kv-tokens {kv}: weights {:.2} GiB, free after {:.2} GiB, \
                 kv pages {} ({} tokens), chunk {}",
                p.weight_bytes as f64 / GIB,
                p.free_after_weights as f64 / GIB,
                p.mo.kv_blocks,
                p.mo.kv_blocks * 256,
                p.chunk
            ),
            Err(e) => eprintln!("planner at 117 GiB, --slots {slots} --max-model-len {len} --kv-tokens {kv}: REFUSED: {e}"),
        }
    }
}

/// The embedding read in place (`REXL3_GLM_EMBED_IN_PLACE`, `GlmPack::from_inventory_embed`): a rank's plan of the RED-SNOW 4.91 bpw pack
/// (and the 2.25 bpw pack's, `REXL3_GLM_DIR`) holds exactly the embedding table's bytes less, in a `Mapped` region of one item that no
/// load plan keeps; the plan without it is the default plan item for item. CPU only, on the pack headers.
#[test]
fn the_embedding_read_in_place_leaves_the_rank_plan_by_exactly_its_bytes() {
    use gb_model::families::glm::plan::RegionKind;
    use gb_model::families::glm::tp_boot::rank_weight_bytes_with;
    let dirs: Vec<PathBuf> = ["REXL3_REDSNOW_DIR", "REXL3_GLM_DIR"]
        .iter()
        .filter_map(|k| std::env::var(k).ok().map(PathBuf::from))
        .filter(|p| p.join("config.json").exists())
        .collect();
    if dirs.is_empty() {
        eprintln!("REXL3_REDSNOW_DIR / REXL3_GLM_DIR not set: not run");
        return;
    }
    for dir in dirs {
        // the headers only (no marker reads): a header-only copy of a pack is enough
        let ck = gb_exl3::checkpoint::scan(&dir, false).unwrap();
        let inv = PackInventory::from_checkpoint(&ck).unwrap();
        let cfg = GlmConfig::from_dir(&dir).unwrap().with_kda_rank(0, 2).unwrap();
        let own = ExpertOwnership::range(288, 0, 2).unwrap();
        let copy = GlmPack::from_inventory_embed(&inv, cfg.clone(), &own, false, false).unwrap();
        let mapped = GlmPack::from_inventory_embed(&inv, cfg.clone(), &own, false, true).unwrap();
        // the default plan is unchanged by the option's existence
        let default = GlmPack::from_inventory_with(&inv, cfg.clone(), &own, false).unwrap();
        assert_eq!(default.plan.items, copy.plan.items, "{}: the default plan", dir.display());
        assert_eq!(default.plan.regions, copy.plan.regions);
        let m: Vec<_> = mapped.plan.regions.iter().filter(|r| r.kind == RegionKind::Mapped).collect();
        assert_eq!(m.len(), 1, "one mapped region");
        let items: Vec<_> = mapped.plan.items.iter().filter(|i| i.region == m[0].id).collect();
        assert_eq!(items.len(), 1);
        assert!(items[0].name.ends_with("embed_tokens.weight"), "{}", items[0].name);
        let table = items[0].bytes;
        assert_eq!(table, 154_880 * 4096 * 2);
        for mtp in [false, true] {
            let (a, b) = (rank_weight_bytes_with(&copy, mtp), rank_weight_bytes_with(&mapped, mtp));
            // the Global region shrinks by the table and its alignment padding, nothing else moves
            assert!(a - b >= table && a - b < table + 4096, "{}: mtp {mtp}: {a} - {b} vs the table's {table}", dir.display());
            eprintln!(
                "{}: mtp {mtp}: rank 0 weights {:.3} -> {:.3} GiB (-{:.3}); the table {} bytes at file offset {} (mod 8 = {})",
                dir.file_name().unwrap().to_string_lossy(),
                a as f64 / GIB,
                b as f64 / GIB,
                (a - b) as f64 / GIB,
                table,
                items[0].offset,
                items[0].offset % 8
            );
        }
    }
}

/// One layer of a real pack booted twice, the embedding copied and read in place (`GlmPack::from_inventory_embed`): the executed plan
/// holds no mapped region, the mapped boot holds exactly the table less device memory, the two tables are the same bytes, and the trunk's
/// gather (`mhc::embed_expand`) gives the same rows from both, bit for bit. GPU (under the GPU lock); needs `REXL3_GLM_DIR` or
/// `REXL3_REDSNOW_DIR` (a real pack: the markers are read).
#[test]
#[ignore]
fn a_one_layer_boot_with_the_embedding_in_place_holds_the_copys_bytes() {
    use gb_model::families::glm::load::{boot_mapped, LoadOptions};
    use gb_model::families::glm::plan::RegionKind;
    let Some(dir) = ["REXL3_GLM_DIR", "REXL3_REDSNOW_DIR"].iter().find_map(|k| std::env::var(k).ok().map(PathBuf::from)) else {
        eprintln!("REXL3_GLM_DIR / REXL3_REDSNOW_DIR not set: not run");
        return;
    };
    if !gb_cuda::device_available() || !gb_cuda::rt::pageable_memory_access() {
        eprintln!("no device that reads pageable memory: not run");
        return;
    }
    let ck = gb_exl3::checkpoint::scan(&dir, true).unwrap();
    let inv = PackInventory::from_checkpoint(&ck).unwrap();
    let cfg = GlmConfig::from_dir(&dir).unwrap();
    let all = ExpertOwnership::all(cfg.ffn.routed_experts);
    let copy = GlmPack::from_inventory_embed(&inv, cfg.clone(), &all, false, false).unwrap();
    let mapped = GlmPack::from_inventory_embed(&inv, cfg.clone(), &all, false, true).unwrap();
    let o = LoadOptions::from_env(4 << 30);
    let a = boot_mapped(&dir, copy, Some(1), &o, 0.0).unwrap();
    let b = boot_mapped(&dir, mapped, Some(1), &o, 0.0).unwrap();
    let table = (cfg.vocab * cfg.hidden * 2) as u64;
    assert!(!b.plan.regions.iter().any(|r| r.kind == RegionKind::Mapped), "the executed plan allocates no mapped region");
    assert_eq!(b.device.regions.mapped_bytes(), table);
    let (da, db) = (a.device.regions.bytes(), b.device.regions.bytes());
    assert!(da - db >= table && da - db < table + 4096, "device bytes {da} -> {db}, the table {table}");
    let (ea, eb) = (a.device.embed.to_vec::<u16>().unwrap(), b.device.embed.to_vec::<u16>().unwrap());
    assert!(ea == eb, "the table read in place is not the copy's bytes");
    let st = gb_cuda::Stream::new().unwrap();
    let ids: Vec<i32> = vec![0, 1, 2, 4095, 65_537, 154_855, cfg.vocab as i32 - 1, -1, cfg.vocab as i32];
    let rows = ids.len();
    let di = gb_cuda::DevBuf::from_slice(&ids).unwrap();
    let gather = |t: &gb_cuda::DevBuf| -> Vec<u32> {
        let x = gb_cuda::DevBuf::zeroed(rows * gb_cuda::mhc::ROW * 4).unwrap();
        unsafe { gb_cuda::mhc::embed_expand(t.ptr(), di.ptr(), x.ptr() as *mut f32, rows, cfg.vocab, &st).unwrap() };
        st.sync().unwrap();
        x.to_vec::<f32>().unwrap().iter().map(|v| v.to_bits()).collect()
    };
    assert!(gather(&a.device.embed) == gather(&b.device.embed), "the trunk's gather differs");
    eprintln!(
        "one layer, embedding in place: device {:.3} -> {:.3} GiB (-{:.3}), mapped {:.3} GiB, table pointer mod 8 = {}, {rows} rows gathered bit for bit",
        da as f64 / GIB,
        db as f64 / GIB,
        (da - db) as f64 / GIB,
        table as f64 / GIB,
        b.device.embed.ptr() as usize % 8
    );
}
