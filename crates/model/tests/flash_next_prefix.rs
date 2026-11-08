//! Real-pack prefix equivalence gate. WRITE ONLY on a shared Spark with the production
//! server running; run explicitly on an idle GPU after the Flash-Next forward port lands:
//! REXL3_FLASH_NEXT_DIR=~/models/flashnext-exl3-3.05bpw REXL3_FLASH_NEXT_REF=~/rexl3-oracle/ref305 \
//! cargo test --release -p gb-model --test flash_next_prefix -- --ignored --nocapture
use gb_engine::Backend;
use gb_model::families::qwen_flash_next::backend::FlashNextBackend;
use gb_model::families::qwen_flash_next::model::{FlashNextModel, ModelOptions};
use gb_model::families::qwen_flash_next::prefix::{self, SliceRequest, SnapshotArena};
use gb_sched::prefix::CacheBudget;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

fn fixture() -> Option<(PathBuf, Vec<u32>)> {
    let dir = std::env::var_os("REXL3_FLASH_NEXT_DIR").map(PathBuf::from);
    let refdir = std::env::var_os("REXL3_FLASH_NEXT_REF").map(PathBuf::from);
    let missing = match (&dir, &refdir) {
        (Some(dir), Some(refdir)) if dir.join("config.json").is_file() && refdir.join("ids_0.u32").is_file() => None,
        (Some(dir), Some(refdir)) => Some(format!(
            "needs a Flash-Next pack with config.json ({}) and reference ids_0.u32 ({})",
            dir.display(),
            refdir.join("ids_0.u32").display()
        )),
        _ => Some("needs REXL3_FLASH_NEXT_DIR and REXL3_FLASH_NEXT_REF".to_owned()),
    };
    if let Some(reason) = missing {
        assert!(!require_gpu(), "REXL3_REQUIRE_GPU=1 but {reason}");
        eprintln!("not run: {reason}");
        return None;
    }
    let dir = dir.unwrap();
    let refdir = refdir.unwrap();
    let raw = std::fs::read(refdir.join("ids_0.u32")).unwrap();
    let ids: Vec<u32> = raw.chunks_exact(4).map(|v| u32::from_le_bytes(v.try_into().unwrap())).collect();
    assert!(ids.len() > 2300);
    Some((dir, ids))
}
fn require_gpu() -> bool {
    std::env::var("REXL3_REQUIRE_GPU").is_ok_and(|v| v == "1")
}
fn model(dir: &Path) -> FlashNextModel {
    FlashNextModel::load(dir, ModelOptions { slots: 2, kv_pages: 64, max_context: 4096, prefill_rows: 1024, warm: false })
        .unwrap()
}
fn arena(m: &FlashNextModel) -> SnapshotArena {
    let snap = prefix::snapshot_bytes(&m.cfg, m.mtp_kv.is_some());
    let page = prefix::page_bytes(&m.cfg, m.mtp_kv.is_some());
    SnapshotArena::new(m, CacheBudget { snapshot_slots: 2, page_blocks: 16, snapshot_bytes: snap, page_bytes: page }).unwrap()
}
fn hash_range(h: &mut Sha256, buf: &gb_cuda::DevBuf, at: usize, n: usize) {
    let mut data = vec![0u8; n];
    buf.download_range(at, &mut data).unwrap();
    h.update(data);
}
/// Hash only rows a prefill actually wrote; unused bytes in a physical page need not match.
fn digest(m: &mut FlashNextModel, slot: usize, len: usize) -> Vec<u8> {
    m.flush().unwrap();
    m.stream.sync().unwrap();
    let mut h = Sha256::new();
    let opts = m.opts;
    for (ss, cc) in m.pools.s.iter().zip(&m.pools.conv) {
        hash_range(&mut h, ss, slot * ss.bytes() / opts.slots, ss.bytes() / opts.slots);
        hash_range(&mut h, cc, slot * cc.bytes() / opts.slots, cc.bytes() / opts.slots);
    }
    hash_range(&mut h, &m.ple_ring, slot * m.ple_ring.bytes() / opts.slots, m.ple_ring.bytes() / opts.slots);
    for ring in &m.qsa_ring {
        hash_range(&mut h, ring, slot * ring.bytes() / opts.slots, ring.bytes() / opts.slots);
    }
    let previous = m.ple_tok.history(slot, len, &[], m.cfg.eos_token_id);
    for t in previous {
        h.update(t.to_le_bytes());
    }
    h.update([m.mtp_done.get(&slot).is_some_and(|&x| x == len) as u8]);
    let blocks = m.kv_tab.blocks_of(slot);
    for cache in m.kv.iter().chain(m.mtp_kv.iter()) {
        let n = len
            + usize::from(m.mtp_kv.as_ref().is_some_and(|mtp| std::ptr::eq(cache, mtp) && m.mtp_done.get(&slot) == Some(&len)));
        for p in 0..n {
            let at = (blocks[p / 256] as usize * 256 + p % 256) * cache.kv_heads.token_elems() * 2;
            let bytes = cache.kv_heads.token_elems() * 2;
            hash_range(&mut h, &cache.k, at, bytes);
            hash_range(&mut h, &cache.v, at, bytes);
        }
    }
    let mtp_done = m.mtp_done.get(&slot).copied() == Some(len);
    for (pool_index, pool) in m.qsa_pool.iter().enumerate() {
        hash_qsa_entries(
            &mut h,
            len,
            qsa_includes_mtp_one_ahead(mtp_done, pool_index, m.qsa_pool.len()),
            m.kv_tab.blocks_of(slot),
            qsa_group(),
            |page, group, h| {
                hash_range(h, pool, qsa_pool_entry_offset(page, group), 256);
            },
        );
    }
    h.finalize().to_vec()
}
fn qsa_includes_mtp_one_ahead(mtp_done: bool, pool_index: usize, pool_count: usize) -> bool {
    mtp_done && pool_index + 1 == pool_count
}
fn hash_qsa_entries(
    h: &mut Sha256,
    len: usize,
    include_mtp_one_ahead: bool,
    blocks: &[u32],
    group_size: usize,
    mut hash_entry: impl FnMut(usize, usize, &mut Sha256),
) {
    let groups = len / group_size + usize::from(include_mtp_one_ahead);
    for group in 0..groups {
        let page = blocks[group / 64] as usize;
        hash_entry(page, group, h);
    }
}
fn qsa_group() -> usize {
    gb_cuda::qsa::CR
}

fn qsa_pool_entry_offset(page: usize, group: usize) -> usize {
    (page * 64 + group % 64) * 256
}

fn qsa_pools_digest(pools: &[Vec<u8>], len: usize, mtp_done: bool, blocks: &[u32]) -> Vec<u8> {
    let mut h = Sha256::new();
    for (pool_index, pool) in pools.iter().enumerate() {
        hash_qsa_entries(
            &mut h,
            len,
            qsa_includes_mtp_one_ahead(mtp_done, pool_index, pools.len()),
            blocks,
            qsa_group(),
            |page, group, h| {
                let at = qsa_pool_entry_offset(page, group);
                h.update(&pool[at..at + 256]);
            },
        );
    }
    h.finalize().to_vec()
}

#[test]
fn digest_tracks_only_the_actual_mtp_one_ahead_qsa_pool_entry() {
    let group_size = qsa_group();
    let len = 64 * group_size;
    let blocks = [7u32, 11u32];
    let mut pool = vec![0u8; (11 * 64 + 64) * 256];
    for (index, byte) in pool.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(37).wrapping_add(11);
    }
    let pools = vec![pool.clone(), pool];

    // The actual last pool includes one MTP-written group just beyond len / CR.
    let baseline = qsa_pools_digest(&pools, len, true, &blocks);
    let boundary_group = len / group_size;
    let boundary_page = blocks[boundary_group / 64] as usize;
    let boundary = qsa_pool_entry_offset(boundary_page, boundary_group);
    assert_eq!(group_size, 4);
    assert_eq!(boundary_group, 64);
    assert_eq!(boundary_page, 11);
    assert_eq!(boundary, (11 * 64) * 256);

    let mut changed_boundary = pools.clone();
    changed_boundary[1][boundary] ^= 0xff;
    assert_eq!(changed_boundary[1][boundary], pools[1][boundary] ^ 0xff);
    assert_eq!(changed_boundary[1].iter().zip(&pools[1]).filter(|(a, b)| a != b).count(), 1);
    assert_eq!(changed_boundary[0], pools[0]);
    assert_ne!(qsa_pools_digest(&changed_boundary, len, true, &blocks), baseline);

    // A later group is outside the digest's span even though it is in the pool.
    let later_group = boundary_group + 1;
    let later_page = blocks[later_group / 64] as usize;
    let later = qsa_pool_entry_offset(later_page, later_group);
    let mut changed_non_boundary = pools.clone();
    changed_non_boundary[1][later] ^= 0xff;
    assert_eq!(changed_non_boundary[1].iter().zip(&pools[1]).filter(|(a, b)| a != b).count(), 1);
    assert_eq!(changed_non_boundary[0], pools[0]);
    assert_eq!(qsa_pools_digest(&changed_non_boundary, len, true, &blocks), baseline);

    // The actual last-pool predicate excludes this same boundary entry in pool 0.
    let mut changed_excluded = pools.clone();
    changed_excluded[0][boundary] ^= 0xff;
    assert_eq!(changed_excluded[0].iter().zip(&pools[0]).filter(|(a, b)| a != b).count(), 1);
    assert_eq!(changed_excluded[1], pools[1]);
    assert!(!qsa_includes_mtp_one_ahead(true, 0, pools.len()));
    assert_eq!(qsa_pools_digest(&changed_excluded, len, true, &blocks), baseline);
}

/// Donor's cut may be on the chunk grid, off the page grid, or inside the
/// natural 1024-row slice. The suffix grid differs from the donor's.
#[test]
#[ignore]
fn cached_logits_and_full_state_equal_cold_at_grid_off_grid_and_mid_slice_cuts() {
    let Some((dir, all)) = fixture() else {
        return;
    };
    gb_cuda::rt::init_context().unwrap();
    let mut m = model(&dir);
    let mut a = arena(&m);
    let prompt = &all[..2300];
    for cut in [1024usize, 777, 513] {
        let mut done = 0;
        let handle = loop {
            let s = prefix::slice_with_cuts(
                &mut m,
                &mut a,
                SliceRequest { slot: 0, prompt, done, candidates: 32, cuts: &[cut], chunk: 1024 },
            )
            .unwrap();
            done = s.end;
            if let Some(c) = s.captured.into_iter().next() {
                break c.handle;
            }
            assert!(done < prompt.len());
        };
        let at_cut = digest(&mut m, 0, cut);
        prefix::restore(&mut m, &a, 1, &handle, cut).unwrap();
        // Host n-gram history is restored from the prompt at the beginning of the suffix slice.
        let from = cut.saturating_sub(128);
        let hist: Vec<i32> = prompt[from..cut].iter().map(|&x| x as i32).collect();
        m.ple_tok.store(1, from, &hist);
        assert_eq!(digest(&mut m, 1, cut), at_cut, "state at cut {cut}");
        let warm = loop {
            let s = prefix::slice_with_cuts(
                &mut m,
                &mut a,
                SliceRequest { slot: 1, prompt, done, candidates: 32, cuts: &[], chunk: 257 },
            )
            .unwrap();
            done = s.end;
            if let Some(c) = s.row {
                break c;
            }
        };
        m.end_slot(0);
        let mut cold_done = 0;
        let cold = loop {
            let s = prefix::slice_with_cuts(
                &mut m,
                &mut a,
                SliceRequest { slot: 0, prompt, done: cold_done, candidates: 32, cuts: &[], chunk: 1024 },
            )
            .unwrap();
            cold_done = s.end;
            if let Some(c) = s.row {
                break c;
            }
        };
        assert_eq!(warm.ids, cold.ids, "candidate ids after cut {cut}");
        assert!(warm.vals.iter().zip(&cold.vals).all(|(x, y)| x.to_bits() == y.to_bits()), "logit bits after cut {cut}");
        assert_eq!(digest(&mut m, 1, prompt.len()), digest(&mut m, 0, prompt.len()), "state after cut {cut}");
        m.end_slot(0);
        m.end_slot(1);
        prefix::release(&mut m, &mut a, handle).unwrap();
        assert_eq!(a.free_count(), 2);
    }
}

/// Exercise the production Backend hooks: capture a real prefix, restore it, then use the
/// ordinary `prefill_slice` entry point (no cuts and no manual host PLE history restoration).
/// Compare a cold continuation with a restored continuation at both page-boundary shapes.
#[test]
#[ignore]
fn backend_prefix_restore_then_ordinary_suffix_matches_cold_without_manual_ple_seed() {
    let Some((dir, all)) = fixture() else {
        return;
    };
    gb_cuda::rt::init_context().unwrap();
    let m = model(&dir);
    let budget = CacheBudget {
        snapshot_slots: 2,
        page_blocks: 16,
        snapshot_bytes: prefix::snapshot_bytes(&m.cfg, m.mtp_kv.is_some()),
        page_bytes: prefix::page_bytes(&m.cfg, m.mtp_kv.is_some()),
    };
    let mut b =
        FlashNextBackend::new(m, 257, 32).with_prefix_cache(budget, "flash-next-real-pack-prefix-regression".into()).unwrap();

    // 512 is page-aligned; 513 is non-aligned. The final case leaves exactly one
    // suffix row so backend route and PLE history rebuilding cover the short tail.
    for (cut, total) in [(512usize, 2300usize), (513, 2300), (513, 514)] {
        let prompt = &all[..total];
        let mut done = 0;
        let handle = loop {
            let step = b.prefill_slice_cuts(0, prompt, &[], done, 32, &[cut]);
            assert!(step.step.row.is_none(), "cut prefix must not produce final logits");
            done = step.step.done;
            if let Some(captured) = step.captured.into_iter().next() {
                assert_eq!(captured.len, cut);
                break captured.handle;
            }
            assert!(done < prompt.len(), "expected a snapshot at cut {cut}");
        };

        // Stop the donor before a cold write: the captured prefix's partial page is
        // immutable while a handle owns it. The cold arm uses the other backend slot.
        b.release(0);
        let mut cold_done = 0;
        let cold_row = loop {
            let step = b.prefill_slice(1, prompt, &[], cold_done, 32);
            cold_done = step.done;
            if let Some(row) = step.row {
                break row;
            }
        };
        assert!(b.take_error().is_none(), "cold backend prefill failed at cut {cut}");
        let cold_state = digest(&mut b.m, 1, total);

        b.prefix_restore(0, &handle, cut).unwrap();
        let mut warm_done = cut;
        let warm_row = loop {
            // Intentionally invoke Backend::prefill_slice rather than the lower-level
            // slice function; that is the production post-hit suffix route.
            let step = b.prefill_slice(0, prompt, &[], warm_done, 32);
            warm_done = step.done;
            if let Some(row) = step.row {
                break row;
            }
        };
        assert!(b.take_error().is_none(), "restored backend prefill failed at cut {cut}");
        assert_eq!(warm_row.ids, cold_row.ids, "candidate ids after cut {cut}, total {total}");
        assert!(
            warm_row.values.iter().zip(&cold_row.values).all(|(x, y)| x.to_bits() == y.to_bits()),
            "candidate logit bits after cut {cut}, total {total}"
        );
        assert_eq!(digest(&mut b.m, 0, total), cold_state, "PLE/QSA/KV/MTP state after cut {cut}, total {total}");
        b.release(0);
        b.release(1);
        b.prefix_release(handle);
        assert_eq!(b.prefix_free_snapshots(), 2);
    }
}

#[test]
#[ignore]
fn a_short_prompt_never_creates_a_snapshot() {
    let Some((dir, all)) = fixture() else {
        return;
    };
    gb_cuda::rt::init_context().unwrap();
    let mut m = model(&dir);
    let mut a = arena(&m);
    let s = prefix::slice_with_cuts(
        &mut m,
        &mut a,
        SliceRequest { slot: 0, prompt: &all[..100], done: 0, candidates: 16, cuts: &[64], chunk: 128 },
    )
    .unwrap();
    assert_eq!(s.end, 100);
    assert!(s.captured.is_empty());
    assert_eq!(a.free_count(), 2);
    m.end_slot(0);
}

/// The engine's row limit (`set_prefill_row_limit(Some(128))`, what `Engine::step` asks for near a drain or request deadline) changes
/// no bit, on the uncached route and on the prefix-cache route (`prefill_slice` goes through `cached_slice` when the backend has an
/// arena, the production default): a 2300-token prompt sliced on the 128 grid throughout, or with the limit switched on and off
/// mid-prompt, leaves the last row, the full slot state (GDN, conv, PLE, QSA, KV, MTP) and two decode windows bit-identical to the
/// unlimited 1024-row slicing.
#[test]
#[ignore]
fn the_engines_row_limit_leaves_the_same_bits() {
    let Some((dir, all)) = fixture() else {
        return;
    };
    gb_cuda::rt::init_context().unwrap();
    let total = 2300;
    let prompt = &all[..total];
    type Limit<'a> = &'a dyn Fn(usize) -> Option<usize>;
    // the prompt through `prefill_slice` with the limit `limit_at(done)` set around each slice: the slice ends, the state digest and
    // the last row followed by two decode windows
    let run = |b: &mut FlashNextBackend, slot: usize, limit_at: Limit| -> (Vec<usize>, Vec<u8>, Vec<gb_engine::RowOutput>) {
        let (mut ends, mut done) = (Vec::new(), 0);
        let last = loop {
            assert!(b.set_prefill_row_limit(limit_at(done)));
            let step = b.prefill_slice(slot, prompt, &[], done, 32);
            b.set_prefill_row_limit(None);
            done = step.done;
            ends.push(done);
            if let Some(row) = step.row {
                break row;
            }
        };
        assert!(b.take_error().is_none(), "prefill failed");
        let state = digest(&mut b.m, slot, total);
        let mut rows = vec![last];
        for k in 0..2 {
            let t = rows.last().unwrap().ids[0];
            let out = b.step(&[gb_engine::RowRequest { slot, first_position: (total + k) as u64, tokens: vec![t] }], 32);
            b.commit(slot, 1);
            rows.push(out[0][0].clone());
        }
        assert!(b.take_error().is_none(), "decode failed");
        b.release(slot);
        (ends, state, rows)
    };
    let cases: [(&str, Limit); 3] = [
        ("on throughout", &|_| Some(128)),
        ("on mid-prompt", &|d| (500..1500).contains(&d).then_some(128)),
        ("37 rows", &|_| Some(37)),
    ];
    let check = |b: &mut FlashNextBackend, route: &str| {
        let (ends0, state0, rows0) = run(b, 0, &|_| None);
        for (what, limit_at) in cases {
            let (ends, state, rows) = run(b, 1, limit_at);
            assert!(ends.len() > ends0.len(), "{route}, {what}: the limit shrank the slices ({ends:?} against {ends0:?})");
            assert_eq!(state, state0, "{route}, {what}: the slot state after the prompt differs");
            for (i, (x, y)) in rows.iter().zip(&rows0).enumerate() {
                assert!(x.bits_eq(y), "{route}, {what}: row {i} (0 = last prompt row, then the decode windows) differs");
            }
            eprintln!(
                "{route}, row limit {what}: {} slices; state, last row and two decode windows are the 1024-row slicing's bits",
                ends.len()
            );
        }
    };
    let mut b = FlashNextBackend::new(model(&dir), 1024, 32);
    check(&mut b, "uncached");
    let m = b.m;
    let budget = CacheBudget {
        snapshot_slots: 2,
        page_blocks: 16,
        snapshot_bytes: prefix::snapshot_bytes(&m.cfg, m.mtp_kv.is_some()),
        page_bytes: prefix::page_bytes(&m.cfg, m.mtp_kv.is_some()),
    };
    let mut b = FlashNextBackend::new(m, 1024, 32).with_prefix_cache(budget, "flash-next-row-limit".into()).unwrap();
    check(&mut b, "prefix-cache route");
}
