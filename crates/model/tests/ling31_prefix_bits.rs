//! The Ling-3.1 prefix cache at the bit level, on the whole model (the real pack, MTP on) behind its backend: a prompt resumed from a
//! snapshot at a cut leaves exactly the bits of the cold prefill of the whole prompt (every KDA layer's state and conv history, every MLA
//! layer's rows, the last row's logits and candidates), because the cuts lie on the KDA chunk grid (`engine::CUT_GRID`: the prefill's
//! recurrence steps 16 rows at a time from each call's first row). `#[ignore]`, env-gated (`REXL3_LING_DIR`; synthetic token ids), one
//! model per process, with a 1 GiB `MemAvailable` guard that aborts the process.
//!
//! * the grid itself, on the model: a prefill split at a multiple of 16 (mid-chunk) leaves the cold bits; one split off the grid is
//!   reported (it is what an off-grid snapshot would resume from);
//! * a mid-prompt cut: the donor's own prefill, and a slot restored from its snapshot and finished, both leave the cold bits; the backend
//!   takes no snapshot off the grid;
//! * the full hit (a prompt that ends exactly at a snapshot, `done >= n`): its first candidates are the cold prompt's.

mod ling31_support;

use gb_engine::{Backend, CacheBudget, RowOutput};
use gb_model::families::bailing_moe_v3::backend::{prefix_namespace, snapshot_fixed_bytes, BailingBackend};
use gb_model::families::bailing_moe_v3::config::BailingConfig;
use gb_model::families::bailing_moe_v3::spill::SpillStore;
use ling31_support::*;

/// The packs MTP mode for this test: the L72 drafters MLA cache ordinal assumes the full 72-layer trunk, so a
/// truncated pack (ling_pack8) runs the same RAM/disk prefix-cache identity contract without MTP.
fn mtp_for_pack() -> bool {
    let Some(d) = pack_dir() else { return false };
    BailingConfig::from_dir(&d).map(|c| c.layers.len() >= 72 && c.has_mtp()).unwrap_or(false)
}

const N: usize = 2600;
/// A cut on the grid inside the third 512-row chunk.
const CUT: usize = 1296;
/// The same place off the grid.
const OFF: usize = 1300;

fn same(a: &RowOutput, b: &RowOutput) -> bool {
    a.bits_eq(b)
}

#[test]
#[ignore]
fn a_prefix_resumed_at_a_cut_leaves_the_cold_bits() {
    // one slot and one snapshot slot (the profile's slot; what fits beside the weights on a GB10): every run below begins the slot
    // again or restores into it, and each result is compared by its digests
    let Some(model) = load(1, 32 * 256, mtp_for_pack()) else { return };
    let cfg = model.cfg.clone();
    let budget = CacheBudget {
        snapshot_slots: 1,
        page_blocks: N.div_ceil(256),
        snapshot_bytes: snapshot_fixed_bytes(&cfg, true),
        page_bytes: gb_cuda::mla_gated::MlaGatedCache::block_bytes(gb_cuda::exl3_linear::Dtype::BF16) as u64 * 10,
    };
    let mut be = BailingBackend::new(model, true).with_prefix_cache(prefix_namespace("test", true), budget).unwrap();
    let toks = ids(7, N);

    // the cold references: the whole prompt, and the prompt up to the cut, each in one slice
    let cold_prefix = be.prefill_slice(0, &toks[..CUT], &[], 0, 8).row.unwrap();
    let cold = be.prefill_slice(0, &toks, &[], 0, 8);
    assert_eq!(cold.done, N);
    let cold_row = cold.row.unwrap();
    let cold_d = slot_digest(&be.model, 0, N, 0);
    let cold_logits = logits_digest(&be.model);

    // the grid, on the model: a prefill split at a multiple of 16 is the cold one; one split off it is reported
    for (split, must) in [(CUT, true), (OFF, false)] {
        let m = &mut be.model;
        m.begin_slot(0).unwrap();
        m.prefill_chunk(0, 0, &toks[..split], None).unwrap();
        m.prefill_chunk(0, split, &toks[split..], Some(8)).unwrap();
        let equal = slot_digest(m, 0, N, 0) == cold_d && logits_digest(m) == cold_logits;
        eprintln!("a prefill split at {split} {} the cold bits", if equal { "leaves" } else { "does NOT leave" });
        if must {
            assert!(equal, "a prefill split at {split} (on the grid) left other bits than the cold one");
        }
    }

    // a mid-prompt cut: the donor's prefill takes the snapshot on the way and still leaves the cold bits
    let donor = be.prefill_slice_cuts(0, &toks, &[], 0, 8, &[CUT]);
    assert_eq!(donor.step.done, N);
    assert!(same(donor.step.row.as_ref().unwrap(), &cold_row), "the donor's candidates");
    assert_eq!(donor.captured.len(), 1, "one snapshot at the cut");
    assert_eq!(donor.captured[0].len, CUT);
    assert_eq!(slot_digest(&be.model, 0, N, 0), cold_d, "the donor's state");
    let handle = donor.captured[0].handle.clone();

    // the slot restored from it (dirty: another prompt was there) and finished in one slice leaves the cold bits
    let _ = be.prefill_slice(0, &ids(9, 700), &[], 0, 8);
    be.prefix_restore(0, &handle, CUT).unwrap();
    let resumed = be.prefill_slice_cuts(0, &toks, &[], CUT, 8, &[]);
    assert_eq!(resumed.step.done, N);
    assert!(same(resumed.step.row.as_ref().unwrap(), &cold_row), "the resumed slot's candidates");
    assert_eq!(slot_digest(&be.model, 0, N, 0), cold_d, "the resumed slot's state");
    assert_eq!(logits_digest(&be.model), cold_logits, "the resumed slot's last-row logits");

    // the full hit: a prompt that ends exactly at the snapshot gets the cold prompt's first candidates (the tap
    // ring this needs exists only when the pack carries the L72 MTP head; on a truncated pack, the restored-slot
    // assertions above already proved the cached == cold contract)
    if mtp_for_pack() {
        be.prefix_restore(0, &handle, CUT).unwrap();
        let full = be.prefill_slice_cuts(0, &toks[..CUT], &[], CUT, 8, &[]);
        assert_eq!(full.step.done, CUT);
        let full_row = full.step.row.unwrap();
        eprintln!("full hit {:?} / cold {:?}", &full_row.ids[..4], &cold_prefix.ids[..4]);
        assert!(same(&full_row, &cold_prefix), "the full hit's candidates are not the cold prompt's");
        be.prefix_release(handle);
    } else {
        eprintln!("truncated pack: skipping the full-hit candidates assert (no L72 tap ring)");
        be.prefix_release(handle);
    }

    // with the snapshot slot free again, the backend takes no snapshot off the grid, and does take one on it
    let off = be.prefill_slice_cuts(0, &toks, &[], 0, 8, &[OFF]);
    assert!(off.captured.is_empty(), "a snapshot off the grid: {:?}", off.captured.iter().map(|c| c.len).collect::<Vec<_>>());
    assert!(same(off.step.row.as_ref().unwrap(), &cold_row));
    let on = be.prefill_slice_cuts(0, &toks, &[], 0, 8, &[CUT]);
    assert_eq!(on.captured.iter().map(|c| c.len).collect::<Vec<_>>(), vec![CUT]);
    for c in on.captured {
        be.prefix_release(c.handle);
    }
    assert!(be.take_error().is_none());
    eprintln!("MemAvailable low {:.2} GiB", gib(guard().low()));
}

/// The same contract through the NVMe spill store: a snapshot written to disk and restored streams back through the staging
/// buffer, every block checksum verified, and the resumed slot leaves exactly the cold bits (MLA pages, KDA states, taps,
/// last-row logits and candidates). This is the RAM-test's identical scenario with the store attached and the RAM path
/// bypassed; together they prove disk-restored state is bit-identical to RAM-restored state and to uncached whole-prompt
/// prefill.
#[test]
#[ignore]
fn a_prefix_resumed_from_disk_leaves_the_cold_bits() {
    let Some(model) = load(1, 32 * 256, mtp_for_pack()) else { return };
    let cfg = model.cfg.clone();
    let dir = std::env::temp_dir().join(format!("ling-spill-bits-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let store = SpillStore::open_at(dir.clone(), 8 << 30, 64 << 20).expect("a writable spill dir");
    let mut model = model;
    model.set_spill_store(Some(store));
    let budget = CacheBudget {
        snapshot_slots: 1,
        page_blocks: N.div_ceil(256),
        snapshot_bytes: snapshot_fixed_bytes(&cfg, true),
        page_bytes: gb_cuda::mla_gated::MlaGatedCache::block_bytes(gb_cuda::exl3_linear::Dtype::BF16) as u64 * 10,
    };
    let mut be = BailingBackend::new(model, true).with_prefix_cache(prefix_namespace("test", true), budget).unwrap();
    let toks = ids(7, N);

    // the cold references, as in the RAM test
    let cold_prefix = be.prefill_slice(0, &toks[..CUT], &[], 0, 8).row.unwrap();
    let cold = be.prefill_slice(0, &toks, &[], 0, 8);
    assert_eq!(cold.done, N);
    let cold_row = cold.row.unwrap();
    let cold_d = slot_digest(&be.model, 0, N, 0);
    let cold_logits = logits_digest(&be.model);

    // a mid-prompt cut, captured through the spill store
    let donor = be.prefill_slice_cuts(0, &toks, &[], 0, 8, &[CUT]);
    assert_eq!(donor.step.done, N);
    assert!(same(donor.step.row.as_ref().unwrap(), &cold_row), "the donor's candidates");
    assert_eq!(donor.captured.len(), 1, "one snapshot at the cut");
    assert_eq!(donor.captured[0].len, CUT);
    assert_eq!(slot_digest(&be.model, 0, N, 0), cold_d, "the donor's state");
    let handle = donor.captured[0].handle.clone();
    let files: Vec<_> =
        std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    assert_eq!(files.len(), 1, "the snapshot is on disk: {files:?}");

    // restore from disk into a dirty slot and finish: the cold bits
    let _ = be.prefill_slice(0, &ids(9, 700), &[], 0, 8);
    be.prefix_restore(0, &handle, CUT).unwrap();
    let resumed = be.prefill_slice_cuts(0, &toks, &[], CUT, 8, &[]);
    assert_eq!(resumed.step.done, N);
    assert!(same(resumed.step.row.as_ref().unwrap(), &cold_row), "the resumed slot's candidates");
    assert_eq!(slot_digest(&be.model, 0, N, 0), cold_d, "the resumed slot's state");
    assert_eq!(logits_digest(&be.model), cold_logits, "the resumed slot's last-row logits");

    // the full hit (see the RAM test: needs the L72 tap ring, so on a truncated pack the restored-slot assertions above carry the proof)
    if mtp_for_pack() {
        be.prefix_restore(0, &handle, CUT).unwrap();
        let full = be.prefill_slice_cuts(0, &toks[..CUT], &[], CUT, 8, &[]);
        assert_eq!(full.step.done, CUT);
        let full_row = full.step.row.unwrap();
        assert!(same(&full_row, &cold_prefix), "the full hit's candidates are not the cold prompt's");
        be.prefix_release(handle);
    } else {
        be.prefix_release(handle);
    }
    assert!(std::fs::read_dir(&dir).unwrap().next().is_none(), "release deleted the spill file");
    assert!(be.take_error().is_none());
    let _ = std::fs::remove_dir_all(&dir);
    eprintln!("MemAvailable low {:.2} GiB", gib(guard().low()));
}
