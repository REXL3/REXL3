//! The published Nemotron 3.5 Super VL pack, through its index.
//!
//! - `the_full_pack_inventory_matches_the_index_and_the_plan_builds`: headers only, no tensor data. Needs `REXL3_NEMO_OMNI_DIR`.
//! - `gpu_load_of_trunk_layers_0_26_7_and_the_lm_head_matches_the_file_bytes`: uploads one Mamba2 layer (0), one latent-MoE layer
//!   with a measured bf16 expert pair (26), one attention layer (7) and the lm_head, then compares every uploaded tensor's bytes with
//!   the file's bytes by SHA-256. The file bytes are read once more here, for the digest; the load itself reads each tensor once.
//!   Needs `REXL3_NEMO_OMNI_DIR` and a CUDA device; without either it reports what it did not run and passes.
#![allow(clippy::needless_borrows_for_generic_args)]

use gb_cuda::loader::Options;
use gb_cuda::DevBuf;
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::inventory::Inventory;
use gb_model::families::nemotron_h_omni::weights::{upload, LayerPlan, UnitId};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// Bytes per device read and per file read when digesting (64 MiB keeps the host buffer small next to the 1 GiB lm_head).
const CHUNK: usize = 64 << 20;

fn pack_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("REXL3_NEMO_OMNI_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("not run: REXL3_NEMO_OMNI_DIR is not set (the published pack directory)");
    }
    dir
}

fn config(dir: &Path) -> OmniConfig {
    let text = std::fs::read_to_string(dir.join("config.json")).unwrap_or_else(|e| panic!("config.json: {e}"));
    OmniConfig::from_json(&serde_json::from_str(&text).unwrap()).unwrap_or_else(|e| panic!("{e}"))
}

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// What the index of a pack says, as MEASURED from that pack (headers only).
struct PackFacts {
    which: &'static str,
    tensors: usize,
    shards: usize,
    indexed_bytes: u64,
    index_total_size: u64,
    stray_shards: usize,
    strays: usize,
    stray_bytes: u64,
}

/// The ORIGINAL pack (index sha256 4bd072f9..., 41 shards; the first upload), as this test pinned it when written: every index
/// that is not one of [`PACK_FACTS_BY_INDEX`] is held to these.
const ORIGINAL_PACK: PackFacts = PackFacts {
    which: "the original pack's index (4bd072f9) or an index this test does not know",
    tensors: 166_757,
    shards: 41,
    indexed_bytes: 69_978_889_808,
    index_total_size: 96_803_702_944,
    stray_shards: 5,
    strays: 56_016,
    stray_bytes: 20_348_565_968,
};

/// Packs told apart by the FNV-1a 64 of their `model.safetensors.index.json` (as `crates/cuda/tests/nemotron_moe.rs` selects its
/// fixture). The published Hub pack (sha256 d2c82b14...: the repaired tensors re-sharded into 33 shards), MEASURED from its headers
/// by lane G3 on 2026-10-15: the same 166,757 indexed tensors and 69,978,889,808 indexed bytes as the original, 5,544 duplicate header
/// entries (2,038,841,256 bytes) in 9 shards, every one of them a duplicate of an indexed name.
const PACK_FACTS_BY_INDEX: [(u64, PackFacts); 1] = [(
    0xc7f15360ccd846e4,
    PackFacts {
        which: "the published Hub pack's index (d2c82b14)",
        tensors: 166_757,
        shards: 33,
        indexed_bytes: 69_978_889_808,
        index_total_size: 71_002_606_716,
        stray_shards: 9,
        strays: 5_544,
        stray_bytes: 2_038_841_256,
    },
)];

fn pack_facts(dir: &Path) -> &'static PackFacts {
    let index = dir.join("model.safetensors.index.json");
    let bytes = std::fs::read(&index).unwrap_or_else(|e| panic!("{}: {e}", index.display()));
    let h = gb_exl3::fnv1a(bytes.iter().copied());
    let facts = PACK_FACTS_BY_INDEX.iter().find(|(f, _)| *f == h).map_or(&ORIGINAL_PACK, |(_, p)| p);
    eprintln!("pack index fnv1a {h:016x}: {}", facts.which);
    facts
}

#[test]
fn the_full_pack_inventory_matches_the_index_and_the_plan_builds() {
    let Some(dir) = pack_dir() else { return };
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let want = pack_facts(&dir);
    assert_eq!(inv.tensors.len(), want.tensors, "tensors the index names");
    assert_eq!(inv.shards.len(), want.shards, "shards the index names");
    assert_eq!(inv.bf16_expert_pairs(), vec![(26, 111), (39, 304), (81, 461)], "the measured bf16 expert pairs");
    assert_eq!(inv.indexed_bytes(), want.indexed_bytes, "bytes of the index-mapped tensors");
    assert_eq!(inv.index_total_size, Some(want.index_total_size), "the index's own total_size (it also counts the duplicates)");
    // the duplicates: header entries of some shards that the index places in another shard. Never loaded.
    assert!(inv.strays.iter().all(|s| s.indexed_in.is_some()), "every stray is a duplicate of an indexed name");
    let mut stray_shards: Vec<&str> = inv.strays.iter().map(|s| s.shard.as_str()).collect();
    stray_shards.sort();
    stray_shards.dedup();
    assert_eq!(stray_shards.len(), want.stray_shards, "{stray_shards:?}");
    assert_eq!(inv.strays.len(), want.strays, "duplicate header entries");
    assert_eq!(inv.stray_bytes(), want.stray_bytes, "bytes of the duplicate header entries");

    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let planned: usize = plan.units.iter().map(|u| u.tensors.len()).sum();
    assert_eq!(planned, 166_361, "the text backbone's tensors");
    assert_eq!(plan.outside_plan, 396, "vision and projector tensors, outside the plan");
    assert_eq!(planned + plan.outside_plan, 166_757);
    let outside_bytes: u64 = inv.tensors.values().filter(|t| !t.name.starts_with("language_model.")).map(|t| t.nbytes).sum();
    assert_eq!(
        plan.bytes(&inv) + outside_bytes,
        inv.indexed_bytes(),
        "the plan and the outside tensors cover every indexed byte once"
    );
    eprintln!(
        "pack: {} indexed tensors in {} shards; plan {} tensors ({} bytes) in {} units; {} outside; {} strays ({} bytes) in {} shards",
        inv.tensors.len(),
        inv.shards.len(),
        planned,
        plan.bytes(&inv),
        plan.units.len(),
        plan.outside_plan,
        inv.strays.len(),
        inv.stray_bytes(),
        stray_shards.len()
    );
}

/// The kernel tests (`crates/cuda/tests/nemotron_moe.rs`) read the pack through `gb_exl3::checkpoint::scan_indexed`; the engine reads it
/// through `Inventory`. They must place every tensor in the same shard at the same byte, for any pack (the repaired pack stores 1,736
/// tensors in three shards, which plain `scan` refuses). Headers only. Needs `REXL3_NEMO_OMNI_DIR`.
#[test]
fn the_checkpoint_scan_the_kernel_tests_use_places_every_tensor_where_the_inventory_does() {
    let Some(dir) = pack_dir() else { return };
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let ck = gb_exl3::checkpoint::scan_indexed(&dir, false).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    assert!(ck.bad.is_empty(), "groups the scan refuses: {:?}", ck.bad.iter().take(5).collect::<Vec<_>>());
    let (mut unresolved, mut moved, mut grouped) = (Vec::new(), Vec::new(), 0usize);
    for (name, t) in &inv.tensors {
        let entry = match ck.plain.get(name) {
            Some(e) => e,
            None => match ck.part(name) {
                Some(e) => {
                    grouped += 1;
                    e
                }
                None => {
                    unresolved.push(name.clone());
                    continue;
                }
            },
        };
        let at = ck.file_offset(entry).unwrap_or_else(|e| panic!("{name}: {e}"));
        if entry.file != t.shard || at != t.offset || entry.nbytes() != t.nbytes {
            moved.push(format!(
                "{name}: scan {} @ {at} (+{}), inventory {} @ {} (+{})",
                entry.file,
                entry.nbytes(),
                t.shard,
                t.offset,
                t.nbytes
            ));
        }
    }
    assert!(
        unresolved.is_empty(),
        "{} indexed tensors the scan does not place, e.g. {:?}",
        unresolved.len(),
        &unresolved[..unresolved.len().min(5)]
    );
    assert!(
        moved.is_empty(),
        "{} tensors placed differently from the inventory, e.g. {:?}",
        moved.len(),
        &moved[..moved.len().min(5)]
    );
    eprintln!(
        "scan_indexed and Inventory agree on all {} indexed tensors ({} in EXL3 groups, {} plain); the scan recorded {} stray or duplicate notes, the inventory {} strays",
        inv.tensors.len(),
        grouped,
        inv.tensors.len() - grouped,
        ck.issues.len(),
        inv.strays.len()
    );
}

/// SHA-256 of `len` device bytes from the start of `buf`, read back in chunks.
fn digest_device(buf: &DevBuf, len: usize) -> [u8; 32] {
    let mut h = Sha256::new();
    let mut chunk = vec![0u8; CHUNK.min(len).max(1)];
    let mut at = 0;
    while at < len {
        let n = CHUNK.min(len - at);
        buf.download_range(at, &mut chunk[..n]).unwrap_or_else(|e| panic!("download at {at}: {e}"));
        h.update(&chunk[..n]);
        at += n;
    }
    h.finalize().into()
}

/// SHA-256 of `len` bytes of `path` from byte `off`.
fn digest_file(path: &Path, off: u64, len: u64) -> [u8; 32] {
    let f = File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut h = Sha256::new();
    let mut chunk = vec![0u8; CHUNK.min(len as usize).max(1)];
    let mut at = 0u64;
    while at < len {
        let n = (CHUNK as u64).min(len - at) as usize;
        f.read_exact_at(&mut chunk[..n], off + at).unwrap_or_else(|e| panic!("{} at {}: {e}", path.display(), off + at));
        h.update(&chunk[..n]);
        at += n as u64;
    }
    h.finalize().into()
}

#[test]
fn gpu_load_of_trunk_layers_0_26_7_and_the_lm_head_matches_the_file_bytes() {
    let Some(dir) = pack_dir() else { return };
    if !gb_cuda::device_available() {
        eprintln!("not run: no CUDA device");
        return;
    }
    gb_cuda::rt::init_context().unwrap_or_else(|e| panic!("CUDA: {e}"));
    let cfg = config(&dir);
    let inv = Inventory::read(&dir).unwrap_or_else(|e| panic!("{e}"));
    let plan = LayerPlan::build(&cfg, &inv).unwrap_or_else(|e| panic!("{}", e.join("\n")));
    let picked = |u: UnitId| matches!(u, UnitId::Trunk(0) | UnitId::Trunk(26) | UnitId::Trunk(7) | UnitId::LmHead);
    let mut w = upload(&dir, &inv, &plan, picked, &Options::from_env()).unwrap_or_else(|e| panic!("{e}"));

    let expected: u64 = plan.tensors(&picked).iter().map(|t| inv.get(&t.name).unwrap().nbytes).sum();
    assert_eq!(w.bytes, expected, "the upload's header sum");
    assert_eq!(w.report.bytes, expected, "the loader copied the header sum, no more and no less");
    eprintln!(
        "loader: {} tensors, {} bytes in {:.2} s ({} jobs, {} spans, {} direct, {} workers)",
        w.len(),
        w.report.bytes,
        w.report.seconds,
        w.report.jobs,
        w.report.spans,
        w.report.direct_spans,
        w.report.workers
    );

    for id in [UnitId::Trunk(0), UnitId::Trunk(26), UnitId::Trunk(7), UnitId::LmHead] {
        let unit = plan.unit(id).unwrap();
        let mut all = Sha256::new();
        let mut bytes = 0u64;
        for want in &unit.tensors {
            let t = w.get(&want.name).unwrap_or_else(|| panic!("`{}` was not uploaded", want.name));
            let on_device = digest_device(&t.buf, t.nbytes as usize);
            let in_file = digest_file(&dir.join(&t.shard), t.offset, t.nbytes);
            assert_eq!(on_device, in_file, "`{}` on the device differs from its file bytes", want.name);
            if want.name.ends_with(".mul1") {
                let mut v = [0u8; 4];
                t.buf.download_range(0, &mut v).unwrap();
                assert_eq!(u32::from_le_bytes(v), gb_exl3::MARKER_MUL1, "`{}` is not the mul1 codebook marker", want.name);
            }
            all.update(on_device);
            bytes += t.nbytes;
        }
        eprintln!("unit {}: {} tensors, {} bytes, digest {}", unit.label, unit.tensors.len(), bytes, hex(&all.finalize()[..16]));
    }

    // typed lookups: an EXL3 linear's parts, a bf16 expert projection, and the absence of the EXL3 form of a bf16 expert
    let parts = w.linear("language_model.backbone.layers.0.mixer.in_proj").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((parts.linear.k, parts.linear.n), (4096, 18560));
    w.typed("language_model.backbone.layers.26.mixer.experts.111.up_proj.weight", "BF16", &[2688, 1024])
        .unwrap_or_else(|e| panic!("{e}"));
    assert!(w.get("language_model.backbone.layers.26.mixer.experts.111.up_proj.trellis").is_none());
    let layer = w.take_linear("language_model.backbone.layers.7.mixer.q_proj").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!((layer.k, layer.n, layer.bits), (4096, 4096, 4));
    eprintln!("typed lookups and Exl3Weights::from_parts ok (q_proj of layer 7 moved out as a device layer)");
}
