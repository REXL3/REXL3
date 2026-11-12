//! CPU-only Flash-Next weight and n-gram contracts.
use gb_model::families::qwen_flash_next::{
    config::FlashNextConfig,
    inventory,
    ngram::{row_ids, TokContext},
    weights::{Transform, WeightMap},
};

const CONFIG: &str = include_str!("../../../testdata/qwen_flash_next/config.json");
const TABLE: &str = include_str!("../../../testdata/qwen_flash_next/tensor_inventory.tsv");

fn fixture() -> (FlashNextConfig, inventory::Tensors) {
    let cfg = FlashNextConfig::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap();
    (cfg, inventory::expand_table(TABLE).unwrap())
}

#[test]
fn every_text_tensor_binds_once_with_named_transforms_and_sidecars() {
    let (cfg, t) = fixture();
    let map = WeightMap::bind(&cfg, &t).unwrap();
    assert_eq!(map.slots.len(), t.keys().filter(|n| !n.starts_with("model.visual.")).count());
    let plan = map.load_plan();
    assert_eq!(plan.items.len() + plan.sidecars.len(), map.slots.len());
    assert!(plan.items.iter().all(|s| s
        .destination
        .as_ref()
        .is_some_and(|d| d.offset % 256 == 0 && d.offset + d.bytes <= plan.regions[d.region as usize].bytes)));
    assert!(plan.sidecars.iter().all(|s| s.destination.is_none()));
    let q = &map.groups["model.language_model.layers.3.self_attn.q_proj"];
    assert_eq!((q.k, q.n, q.bits.twice(), q.codebook), (2560, 12288, 10, gb_exl3::Codebook::Mul1));
    assert!(plan.regions.iter().any(|r| r.name == "layer_0_experts"));
    assert!(plan.regions.iter().any(|r| r.name == "mtp_0_experts"));
    assert_eq!(plan.sidecars.iter().filter(|s| s.name.ends_with(".trellis")).count(), 128);
    assert_eq!(map.slots["model.language_model.layers.3.self_attn.q_norm.weight"].transform, Transform::Bf16AddOneToF32);
    assert_eq!(map.slots["model.language_model.layers.0.linear_attn.A_log"].transform, Transform::Bf16ToF32);
    assert_eq!(map.slots["model.language_model.layers.1.ple.norm_key.weight"].transform, Transform::Bf16AddOneToF32);
    assert_eq!(map.slots["mtp.pre_fc_norm_embedding.weight"].transform, Transform::Bf16AddOneToF32);
}

#[test]
fn missing_wrong_shape_wrong_dtype_and_extra_tensors_are_named() {
    let (cfg, mut t) = fixture();
    let key = "mtp.layers.0.self_attn.indexer.k_layernorm.weight";
    t.remove(key);
    assert!(WeightMap::bind(&cfg, &t).unwrap_err().iter().any(|e| e.contains(key)));
    let (cfg, mut t) = fixture();
    let key = "model.language_model.layers.5.mlp.experts.511.down_proj.trellis";
    t.get_mut(key).unwrap().1[0] += 1;
    assert!(WeightMap::bind(&cfg, &t).unwrap_err().iter().any(|e| e.contains(key)));
    let (cfg, mut t) = fixture();
    let key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.trellis";
    t.get_mut(key).unwrap().0 = "F16".into();
    assert!(WeightMap::bind(&cfg, &t).unwrap_err().iter().any(|e| e.contains(key)));
    let (cfg, mut t) = fixture();
    let key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.trellis";
    t.remove(key);
    assert!(WeightMap::bind(&cfg, &t).unwrap_err().iter().any(|e| e.contains("shard_127.trellis")));
    let (cfg, mut t) = fixture();
    let key = "mtp.hyper_connection_mixer.hc_norm.weight";
    t.remove(key);
    assert!(WeightMap::bind(&cfg, &t).unwrap_err().iter().any(|e| e.contains(key)));
    let (cfg, mut t) = fixture();
    t.insert("mtp.layers.0.mlp.unused.weight".into(), ("F16".into(), vec![1]));
    assert!(WeightMap::bind(&cfg, &t).unwrap_err().iter().any(|e| e.contains("mtp.layers.0.mlp.unused.weight")));
}

/// Header-only real pack binding. Intentionally ignored: do not run while the production server holds memory.
#[test]
#[ignore]
fn real_pack_load_plan_from_headers() {
    let Ok(dir) = std::env::var("REXL3_FLASH_NEXT_DIR") else {
        eprintln!("not run: REXL3_FLASH_NEXT_DIR unset");
        return;
    };
    let dir = std::path::Path::new(&dir);
    let cfg = FlashNextConfig::from_dir(dir).unwrap();
    let map = WeightMap::from_dir(dir, &cfg).unwrap();
    let plan = map.load_plan();
    assert!(plan.items.iter().all(|s| s.source.is_some()));
    assert!(plan.sidecars.iter().all(|s| s.source.is_some()));
    assert_eq!(
        map.slots.len(),
        304_240 - inventory::expand_table(TABLE).unwrap().keys().filter(|n| n.starts_with("model.visual.")).count()
    );
}

#[test]
fn indexed_main_pack_and_unindexed_ple_shard_sidecar_are_scanned_separately() {
    use gb_model::pack::Pack;
    use std::{collections::BTreeMap, fs};
    let dir = std::env::temp_dir().join(format!("flash-next-ngram-index-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("config.json"), CONFIG).unwrap();

    // One canonical indexed main shard. The n-gram sidecar is deliberately absent from the index.
    let main_name = "model.language_model.embed_tokens.weight";
    let mut main_entries = serde_json::Map::new();
    main_entries.insert(main_name.into(), serde_json::json!({"dtype":"BF16", "shape":[1], "data_offsets":[0,2]}));
    let main_json = serde_json::to_vec(&serde_json::Value::Object(main_entries)).unwrap();
    let mut main = (main_json.len() as u64).to_le_bytes().to_vec();
    main.extend(main_json);
    main.extend([0u8; 2]);
    fs::write(dir.join("model-00001-of-00001.safetensors"), main).unwrap();
    let mut index_map = serde_json::Map::new();
    index_map.insert(main_name.into(), serde_json::json!("model-00001-of-00001.safetensors"));
    fs::write(dir.join("model.safetensors.index.json"), serde_json::json!({"weight_map": index_map}).to_string()).unwrap();

    let cfg = FlashNextConfig::from_dir(&dir).unwrap();
    let prefix = format!("model.language_model.layers.{}.ple.ple_embedding.ngram_embedding", cfg.ple_layer_ids[0] - 1);
    let mut header = BTreeMap::new();
    let mut offset = 0usize;
    for shard in 0..128 {
        let name = format!("{prefix}.shard_{shard}.trellis");
        let end = offset + 4 * inventory::ngram_row_words(1);
        header.insert(
            name,
            serde_json::json!({"dtype":"I16", "shape":[2, inventory::ngram_row_words(1)], "data_offsets":[offset,end]}),
        );
        offset = end;
    }
    let nheads = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
    for (name, dtype, shape, bytes) in [
        (format!("{prefix}.head_offsets"), "I64", vec![nheads], nheads * 8),
        (format!("{prefix}.head_vocab_sizes"), "I64", vec![nheads], nheads * 8),
        (format!("{prefix}.layer_multipliers"), "I64", vec![cfg.ngram_size], cfg.ngram_size * 8),
        (format!("{prefix}.head_bias"), "F16", vec![nheads, cfg.ple_embed_dim / nheads], cfg.ple_embed_dim * 2),
    ] {
        let end = offset + bytes;
        header.insert(name, serde_json::json!({"dtype":dtype, "shape":shape, "data_offsets":[offset,end]}));
        offset = end;
    }
    let sidecar_head = serde_json::Value::Object(header.into_iter().collect()).to_string().into_bytes();
    let mut sidecar = (sidecar_head.len() as u64).to_le_bytes().to_vec();
    sidecar.extend(sidecar_head);
    sidecar.resize(sidecar.len() + offset, 0);
    fs::write(dir.join(inventory::NGRAM_FILE), sidecar).unwrap();

    let pack = Pack::open(&dir).unwrap();
    assert!(pack.ck.bad.is_empty(), "indexed main pack refused: {:?}", pack.ck.bad);
    assert_eq!(pack.ck.plain.len(), 1, "the unindexed sidecar must not enter the main pack scan");

    // The full model is intentionally not copied into this fixture. The loader should report its absent
    // trunk weights, but not treat the 128 valid sidecar shard headers as absent or malformed.
    let errors = WeightMap::from_dir(&dir, &cfg).unwrap_err();
    assert!(!errors.iter().any(|e| e.contains("ngram_embedding.safetensors") || e.contains("shard_")), "{errors:?}");
    let _ = fs::remove_dir_all(&dir);
}

/// Uses the captured ExLlamaV3 PLE token history; the capture has no saved row IDs, so the
/// first-position IDs are pinned from a separate signed-wrap/Euclidean oracle calculation.
#[test]
fn oracle_capture_hash_and_packed_row_gather() {
    let (Ok(dir), Ok(reference)) = (std::env::var("REXL3_FLASH_NEXT_NGRAM_DIR"), std::env::var("REXL3_FLASH_NEXT_ORACLE_DIR"))
    else {
        eprintln!("not run: REXL3_FLASH_NEXT_NGRAM_DIR and REXL3_FLASH_NEXT_ORACLE_DIR unset");
        return;
    };
    let cfg = FlashNextConfig::from_dir(std::path::Path::new(&dir)).unwrap();
    let table = gb_model::families::qwen_flash_next::ngram::NgramTable::open(std::path::Path::new(&dir), &cfg).unwrap();
    let raw = std::fs::read(std::path::Path::new(&reference).join("ple_history.npy")).unwrap();
    assert_eq!(&raw[..6], b"\x93NUMPY");
    let hdr = if raw[6] == 1 {
        10 + u16::from_le_bytes(raw[8..10].try_into().unwrap()) as usize
    } else {
        12 + u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize
    };
    let ids: Vec<u32> = raw[hdr..].chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as u32).collect();
    assert_eq!(ids.len(), 1026);
    assert_eq!(&ids[..2], &[cfg.eos_token_id, cfg.eos_token_id]);
    let mut rows = [0i64; 16];
    table.row_ids(ids[2], ids[1], ids[0], &mut rows).unwrap();
    assert_eq!(
        rows,
        [
            1246164, 35492977, 46431022, 66016679, 86771609, 114503770, 126864847, 144770411, 164997853, 180369151, 204412240,
            229362832, 249738277, 271322272, 298785664, 318115908
        ]
    );
    let mut packed = vec![0u16; table.heads * table.words];
    table.gather(&ids[..3], 2, &mut packed).unwrap();
    assert!(packed.iter().any(|&w| w != 0));
    // Full captured history, not just the EOS-boundary first token. The digest is from an
    // independent Python implementation of the reference signed-wrap hash over the capture.
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    for at in 2..ids.len() {
        table.row_ids(ids[at], ids[at - 1], ids[at - 2], &mut rows).unwrap();
        for row in rows {
            digest.update(row.to_le_bytes());
        }
    }
    assert_eq!(format!("{:x}", digest.finalize()), "85b9b81e647b0f06623d05dba4dccc4e7c243fc6d0cf59bedda985e64176cbf6");
}

#[test]
fn hash_wraps_and_eos_blocks_three_gram_crossing() {
    let (cfg, _) = fixture();
    let mult = [i64::MAX, -2, 19];
    let offsets = [0, 10];
    let sizes = [101, 97];
    let mut a = [0; 2];
    let mut b = [0; 2];
    row_ids([42, cfg.eos_token_id, 1234], cfg.eos_token_id, &mult, &offsets, &sizes, 1, &mut a).unwrap();
    row_ids([42, cfg.eos_token_id, 9876], cfg.eos_token_id, &mult, &offsets, &sizes, 1, &mut b).unwrap();
    assert_eq!(a, b);
    let m = 42i64.wrapping_mul(mult[0]) ^ (cfg.eos_token_id as i64).wrapping_mul(mult[1]);
    assert_eq!(a[0], m.rem_euclid(101));
    assert_eq!(a[1], (m ^ (cfg.eos_token_id as i64).wrapping_mul(mult[2])).rem_euclid(97) + 10);
}

#[test]
fn token_ring_keeps_slots_isolated_across_wrap_and_draft_replacement() {
    let mut ring = TokContext::new(2);
    assert_eq!(ring.history(1, 0, &[5, 6], 9), [9, 9, 5, 6]);
    ring.store(1, 0, &[5, 6, 7]);
    assert_eq!(ring.history(1, 3, &[8], 9), [6, 7, 8]);
    let ids: Vec<i32> = (0..1000).collect();
    ring.store(0, 0, &ids);
    assert_eq!(ring.history(0, 1000, &[1], 9), [998, 999, 1]);
    ring.store(0, 999, &[77]);
    assert_eq!(ring.history(0, 1000, &[78], 9), [998, 77, 78]);
    assert_eq!(ring.history(1, 3, &[8], 9), [6, 7, 8]);
}
