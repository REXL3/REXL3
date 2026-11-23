//! The real Qwen3.8-Flash-Next pack, read on the CPU: only the shard headers (no weight, no CUDA context). Set
//! REXL3_FLASH_NEXT_DIR to the pack (tests skip without it):
//!
//! `REXL3_FLASH_NEXT_DIR=~/models/flashnext-exl3-3.05bpw cargo test --release -p gb-model --test flash_next_pack -- --nocapture`
use gb_model::families::qwen_flash_next::config::FlashNextConfig;
use gb_model::families::qwen_flash_next::inventory::{check, expand_table, ngram_table, Tensors, MTP_MIXER_FILE, NGRAM_FILE};
use gb_model::pack::Pack;
use std::io::Read;
use std::path::{Path, PathBuf};

const TABLE: &str = include_str!("../../../testdata/qwen_flash_next/tensor_inventory.tsv");

fn pack_dir() -> Option<PathBuf> {
    std::env::var("REXL3_FLASH_NEXT_DIR").ok().map(PathBuf::from).filter(|p| p.join("config.json").exists())
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

#[test]
fn the_flash_next_pack_matches_the_committed_inventory_and_its_config() {
    let Some(dir) = pack_dir() else { return eprintln!("not run: REXL3_FLASH_NEXT_DIR is not set") };
    // the indexed shards scan as EXL3 with nothing refused
    let pack = Pack::open(&dir).unwrap();
    assert!(pack.ck.bad.is_empty());
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).unwrap()).unwrap();
    let indexed = index["weight_map"].as_object().unwrap().len();
    // every shard's header, the two sidecars included
    let mut real = Tensors::new();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    for f in &files {
        real.extend(header(f));
    }
    let sidecars: usize = [NGRAM_FILE, MTP_MIXER_FILE].iter().map(|f| header(&dir.join(f)).len()).sum();
    assert_eq!(real.len(), indexed + sidecars, "the index and the shard headers disagree");
    let table = expand_table(TABLE).unwrap();
    let diff: Vec<_> = real
        .keys()
        .filter(|k| real.get(*k) != table.get(*k))
        .chain(table.keys().filter(|k| !real.contains_key(*k)))
        .take(5)
        .collect();
    assert!(diff.is_empty(), "the committed table drifted from the pack at {diff:?}");
    let cfg = FlashNextConfig::from_dir(&dir).unwrap();
    cfg.check_shapes().unwrap();
    assert_eq!(check(&cfg, &real), Vec::<String>::new());
    eprintln!(
        "{} tensors ({} indexed, {} in the sidecars), n-gram table {:?}",
        real.len(),
        indexed,
        sidecars,
        ngram_table(&cfg, &real)
    );
}

/// The boot scans the pack once and builds the weight map and the pack from that scan (`WeightMap::from_checkpoint`,
/// `Pack::from_checkpoint`): both must equal what scanning again gives (`WeightMap::from_dir`, `Pack::open`).
#[test]
fn one_scan_gives_the_weight_map_and_the_pack_a_fresh_scan_gives() {
    use gb_model::families::qwen_flash_next::weights::WeightMap;
    let Some(dir) = pack_dir() else { return eprintln!("not run: REXL3_FLASH_NEXT_DIR is not set") };
    let cfg = FlashNextConfig::from_dir(&dir).unwrap();
    let ck = gb_exl3::checkpoint::scan(&dir, true).unwrap();
    assert_eq!(WeightMap::from_checkpoint(&dir, &ck, &cfg).unwrap(), WeightMap::from_dir(&dir, &cfg).unwrap());
    let once = Pack::from_checkpoint(ck).unwrap();
    let again = Pack::open(&dir).unwrap();
    assert_eq!((&once.ck.groups, &once.ck.plain, &once.ck.markers), (&again.ck.groups, &again.ck.plain, &again.ck.markers));
}

/// A scan with a refused EXL3 group is refused by `Pack::from_checkpoint` with the message `Pack::open` gives for it.
#[test]
fn a_pack_from_a_scan_is_refused_as_pack_open_refuses_it() {
    let dir = std::env::temp_dir().join(format!("rexl3-fn-scan-once-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // one group whose trellis is the wrong dtype: the scan refuses the group (`ck.bad`)
    let head = serde_json::json!({
        "m.trellis": {"dtype": "F16", "shape": [1, 1, 32], "data_offsets": [0, 64]},
        "m.suh": {"dtype": "F16", "shape": [16], "data_offsets": [64, 96]},
        "m.svh": {"dtype": "F16", "shape": [16], "data_offsets": [96, 128]},
    })
    .to_string();
    let mut bytes = (head.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(head.as_bytes());
    bytes.resize(bytes.len() + 128, 0);
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    let ck = gb_exl3::checkpoint::scan(&dir, true).unwrap();
    assert!(!ck.bad.is_empty());
    let from_scan = Pack::from_checkpoint(ck).err().expect("a refused group is refused").to_string();
    let opened = Pack::open(&dir).err().expect("a refused group is refused").to_string();
    assert_eq!(from_scan, opened);
    let _ = std::fs::remove_dir_all(&dir);
}
