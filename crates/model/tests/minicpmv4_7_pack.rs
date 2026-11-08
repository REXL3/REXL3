//! A real MiniCPM-V-4.7-35B-A3B pack and its DSpark draft, read on the CPU: only the shard headers (no weight, no CUDA context).
//! Tests skip without their directory.
//!
//! * `REXL3_MINICPMV47_DIR`: the EXL3 pack (the full 40-layer pack, or a truncated smoke pack whose config says fewer layers: the
//!   first N layers are then compared with the committed table);
//! * `REXL3_MINICPMV47_DRAFT_DIR`: the DSpark draft (the Hugging Face checkpoint or its EXL3 conversion).
//!
//! `REXL3_MINICPMV47_DIR=~/models/minicpmv47-exl3-4.00bpw cargo test --release -p gb-model --test minicpmv4_7_pack -- --nocapture`
//!
//! The first run against a pack the converter wrote is the smoke that settles whether the loader accepts what the fork writes
//! (`.mul1` markers, the padded LM head, the per-expert layout): `tools/gen_minicpmv4_7_inventory.py diff` does the same comparison
//! outside Cargo and reports the bit allocation.
use gb_model::families::minicpmv4_7::config::{DSparkConfig, MiniCpmConfig};
use gb_model::families::minicpmv4_7::inventory::{check, check_draft, expand_table, Tensors, TENSORS_PER_LAYER};
use gb_model::pack::Pack;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

const TABLE: &str = include_str!("../../../testdata/minicpmv4_7/tensor_inventory.tsv");

fn dir_of(var: &str) -> Option<PathBuf> {
    std::env::var(var).ok().map(PathBuf::from).filter(|p| p.join("config.json").exists())
}

/// The tensors one safetensors file declares: its 8-byte header length, then the JSON header.
fn header(path: &Path) -> Tensors {
    let mut f = std::fs::File::open(path).unwrap();
    let mut len = [0u8; 8];
    f.read_exact(&mut len).unwrap();
    let mut buf = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut buf).unwrap();
    let head: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&buf).unwrap();
    head.into_iter()
        .filter(|(k, _)| k != "__metadata__")
        .map(|(k, v)| {
            let shape = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
            (k, (v["dtype"].as_str().unwrap().to_string(), shape))
        })
        .collect()
}

fn all_headers(dir: &Path) -> Tensors {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "{}: no safetensors files", dir.display());
    let mut real = Tensors::new();
    for f in &files {
        for (k, v) in header(f) {
            assert!(real.insert(k.clone(), v).is_none(), "{k} is in two shards");
        }
    }
    real
}

/// The layer index of a text-layer tensor name.
fn text_layer(name: &str) -> Option<usize> {
    name.strip_prefix("model.language_model.layers.")?.split('.').next()?.parse().ok()
}

#[test]
fn the_minicpm_pack_matches_the_committed_inventory_and_its_config() {
    // `REXL3_MINICPM_DIR` is the name the boot test and the gate use for the same pack
    let Some(dir) = dir_of("REXL3_MINICPMV47_DIR").or_else(|| dir_of("REXL3_MINICPM_DIR")) else {
        return eprintln!("not run: REXL3_MINICPMV47_DIR is not set");
    };
    // the shards scan as EXL3 with nothing refused (markers, scales and trellises of every group agree)
    let pack = Pack::open(&dir).unwrap();
    assert!(pack.ck.bad.is_empty());
    let real = all_headers(&dir);
    if let Ok(index) = std::fs::read(dir.join("model.safetensors.index.json")) {
        let index: serde_json::Value = serde_json::from_slice(&index).unwrap();
        assert_eq!(real.len(), index["weight_map"].as_object().unwrap().len(), "the index and the shard headers disagree");
    }
    let cfg = MiniCpmConfig::from_dir(&dir).unwrap();
    let table = expand_table(TABLE).unwrap();
    // a truncated smoke pack carries the first N layers; everything else (head, embedding, tower) is the same
    let layers = cfg.layers;
    let expected: BTreeMap<_, _> = table.iter().filter(|(n, _)| text_layer(n).is_none_or(|l| l < layers)).collect();
    let mut depth_only = 0usize;
    let mut structural: Vec<String> = Vec::new();
    for (name, (dtype, shape)) in &real {
        match expected.get(name) {
            None => structural.push(format!("{name}: not in the expected layout")),
            Some((edt, eshape)) => {
                let trellis_depth =
                    name.ends_with(".trellis") && shape.len() == 3 && eshape.len() == 3 && shape[..2] == eshape[..2];
                if dtype != edt {
                    structural.push(format!("{name}: dtype {dtype}, expected {edt}"));
                } else if shape != eshape && trellis_depth {
                    depth_only += 1;
                } else if shape != eshape {
                    structural.push(format!("{name}: shape {shape:?}, expected {eshape:?}"));
                }
            }
        }
    }
    for name in expected.keys().filter(|n| !real.contains_key(**n)).take(5) {
        structural.push(format!("{name}: missing from the pack"));
    }
    assert!(structural.is_empty(), "{} differences, e.g. {:?}", structural.len(), &structural[..structural.len().min(8)]);
    assert_eq!(real.len(), layers * TENSORS_PER_LAYER + 6 + 459, "{layers} layers");
    if layers == 40 {
        cfg.check_shapes().unwrap();
        assert_eq!(check(&cfg, &real), Vec::<String>::new());
    }
    eprintln!("{} tensors, {layers} layers, {depth_only} trellis tensors at a depth other than the nominal one", real.len());
}

#[test]
fn the_dspark_draft_matches_its_config_and_the_target() {
    let Some(dir) = dir_of("REXL3_MINICPMV47_DRAFT_DIR") else {
        return eprintln!("not run: REXL3_MINICPMV47_DRAFT_DIR is not set");
    };
    let draft = DSparkConfig::from_dir(&dir).unwrap();
    draft.check_shapes().unwrap();
    let real = all_headers(&dir);
    let errors = check_draft(&draft, &real);
    assert!(errors.is_empty(), "{:#?}", &errors[..errors.len().min(10)]);
    // paired with the target's committed config
    let target: serde_json::Value = serde_json::from_str(include_str!("../../../testdata/minicpmv4_7/config.json")).unwrap();
    let target = MiniCpmConfig::from_json(&target).unwrap();
    draft.check_pairing(&target).unwrap();
    eprintln!("{} tensors, block {}, taps {:?}", real.len(), draft.block_size, draft.target_layer_ids);
}
