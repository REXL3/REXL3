//! The GLM prefix cache on the whole model with the NextN layer (90.5 GiB resident, anonymized host-b or anonymized host-d): a snapshot of a slot after N tokens, a restore of it
//! into another slot (dirty: another prompt was there), and the rest of the prompt prefilled in another chunk grid must leave EXACTLY what a cold prefill
//! of the whole prompt leaves: the KDA state and conv history of 34 layers, the latent rows and pool rows of 11 trunk layers, the DSA ring tail of each,
//! the NextN layer's cache, pool rows, ring tail and carry, the last row's logits, the drafter's drafts, and the logits of the next decode steps.
//!
//! `#[ignore]`, env-gated (`REXL3_GLM_DIR`; no oracle dumps needed: the token ids are synthetic, row invariance does not depend on them), under the GPU
//! lock, one model per process (`--test-threads=1`), `REXL3_GLM_CHUNK_ROWS` defaults to 1,024 here so that the large-row kernels and slabbed attention run
//! (the cache's claim is that a snapshot is valid at ANY cut of ANY chunk grid, which is the row-invariance claim of `glm_prefill.rs`).
//!
//! * `resume_equals_cold_at_every_cut`: 23 cuts (every residue mod 4, the page boundary and its neighbours, 127/128/129, the 2,048 sparse boundary and
//!   its neighbours), the donor cut into chunks that end at the cut, the restored slot finishing in a different grid, the digest compared at the cut
//!   and at the end.
//! * `interior_cuts_equal_cold_prefill`: the snapshots taken INSIDE a large chunk (no extra forward) are those of a donor that stopped at the cut.
//! * `the_donor_goes_on_and_a_sharer_comes_later`: copy-on-write both ways.
//! * `allocations_equal_the_plan_with_the_cache`, `a_restore_of_the_wrong_snapshot_is_refused_and_leaks_nothing`.

mod glm_support;

use gb_model::families::glm::alloc::{AllocParams, AllocPlan};
use gb_model::families::glm::model::{DraftAsk, GlmModel, Head, Mode, PrefixDigest};
use gb_model::families::glm::prefix;
use gb_model::rows::RowBatch;
use glm_support::*;

const SLOTS: usize = 4;
const KV_BLOCKS: usize = 160; // 40,960 tokens
const MAX_CONTEXT: usize = 8192;
const N: usize = 2400;
const SNAPSHOTS: usize = 6;

fn model() -> Option<std::sync::MutexGuard<'static, GlmModel>> {
    if std::env::var("REXL3_GLM_CHUNK_ROWS").is_err() {
        std::env::set_var("REXL3_GLM_CHUNK_ROWS", "1024");
    }
    let mut m = shared_model_mtp(SLOTS, KV_BLOCKS, MAX_CONTEXT)?;
    // these tests cut at ANY position and resume in ANY chunk grid: that is the P1 contract (the process default is P2, whose cuts and
    // chunks sit on the segment's 16-row grid; its restores are `glm_p2.rs`'s and `glm_prefix_engine.rs`'s)
    m.set_prefill_contract(gb_model::families::glm::contract::PrefillContract::P1).unwrap();
    let _ = m.enable_prefix_cache(SNAPSHOTS); // a second call finds it enabled
    Some(m)
}

/// Deterministic token ids that are not degenerate.
fn ids(seed: u64, n: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1 + (x % 150_000) as u32
        })
        .collect()
}

/// Prefill `toks[from..to]` into `slot` in chunks of the sizes in `grid` (cycled, clamped to the model's chunk). With `want`, the last chunk also asks
/// for candidates and the raw logits of its last row are returned.
fn prefill(
    m: &mut GlmModel,
    slot: usize,
    toks: &[u32],
    from: usize,
    to: usize,
    grid: &[usize],
    want: Option<usize>,
) -> Option<(gb_model::forward::Candidates, Vec<u16>)> {
    let (mut at, mut i) = (from, 0usize);
    let mut out = None;
    while at < to {
        let size = grid[i % grid.len()].min(m.chunk_rows());
        i += 1;
        let end = (at + size).min(to);
        let w = (end == to).then_some(want).flatten();
        let c = m.prefill_chunk(slot, at, &toks[at..end], w).unwrap();
        if let Some(c) = c {
            out = Some((c, m.logits_host(0, 1).unwrap()));
        }
        at = end;
    }
    out
}

/// What a finished prompt must show: the state (digest at the end), the last row's logits, what the drafter proposes, and the next decode steps.
#[derive(Debug, PartialEq)]
struct Final {
    digest: PrefixDigest,
    top: (Vec<i32>, Vec<u32>),
    logits: Vec<u16>,
    drafts: (Vec<u32>, Vec<u32>),
    decoded: Vec<Vec<u16>>,
}

fn finish(m: &mut GlmModel, slot: usize, toks: &[u32], cand: gb_model::forward::Candidates, logits: Vec<u16>) -> Final {
    let digest = m.prefix_digest(slot, toks.len()).unwrap();
    let first = cand.ids[0] as u32;
    let mut seq = toks.to_vec();
    seq.push(first);
    // the NextN drafter over the prompt it has built: absorbs the pending carry and drafts three
    let d = m.mtp_draft(&[DraftAsk { slot, tokens: &seq, depth: 3 }]).unwrap();
    let drafts = d[0].as_ref().map(|d| (d.ids.clone(), d.vals.iter().map(|v| v.to_bits()).collect())).unwrap_or_default();
    let mut decoded = Vec::new();
    let mut tok = first;
    for j in 0..6 {
        let b = RowBatch::from_requests(&[gb_engine::RowRequest {
            slot,
            tokens: vec![tok],
            first_position: (toks.len() + j) as u64,
        }]);
        m.forward(&b, Mode::Window, Head::AllRows).unwrap();
        decoded.push(m.logits_host(0, 1).unwrap());
        let c = m.candidates_of(1, 1).unwrap();
        m.commit(slot, 1);
        tok = c.ids[0] as u32;
    }
    Final { digest, top: (cand.ids.clone(), cand.vals.iter().map(|v| v.to_bits()).collect()), logits, drafts, decoded }
}

fn describe(a: &Final, b: &Final) -> String {
    let mut d = Vec::new();
    let st = a.digest.describe_diff(&b.digest);
    if !st.is_empty() {
        d.push(format!("state: {st}"));
    }
    if a.logits != b.logits {
        d.push(format!(
            "last-row logits differ in {} of {} elements",
            a.logits.iter().zip(&b.logits).filter(|(x, y)| x != y).count(),
            a.logits.len()
        ));
    }
    if a.top != b.top {
        d.push("top candidates".into());
    }
    if a.drafts != b.drafts {
        d.push(format!("drafts {:?} vs {:?}", a.drafts.0, b.drafts.0));
    }
    if let Some(j) = a.decoded.iter().zip(&b.decoded).position(|(x, y)| x != y) {
        d.push(format!("the logits of decode step {j}"));
    }
    d.join("; ")
}

const COLD_GRID: &[usize] = &[512];

/// The cold reference: the whole prompt in `slot`, finished.
fn cold(m: &mut GlmModel, slot: usize, toks: &[u32]) -> Final {
    m.begin_slot(slot).unwrap();
    let (c, l) = prefill(m, slot, toks, 0, toks.len(), COLD_GRID, Some(8)).unwrap();
    let f = finish(m, slot, toks, c, l);
    m.end_slot(slot);
    f
}

/// The cuts: every residue mod 4 at several places, the page boundary and its neighbours, the decode kernels' 128, the sparse boundary at 2,048.
const CUTS: [usize; 23] =
    [100, 101, 102, 103, 127, 128, 129, 255, 256, 257, 258, 259, 511, 1000, 1023, 1024, 1025, 2047, 2048, 2049, 2050, 2051, 2100];

/// The cuts the sweep runs: all of [`CUTS`], or the comma list of `REXL3_GLM_PREFIX_CUTS` (the mutation tool runs a few of them per mutant).
fn cuts() -> Vec<usize> {
    match std::env::var("REXL3_GLM_PREFIX_CUTS") {
        Ok(v) => v.split(',').map(|x| x.trim().parse().expect("REXL3_GLM_PREFIX_CUTS is a comma list of token counts")).collect(),
        Err(_) => CUTS.to_vec(),
    }
}

/// Grids the donor and the restored slot run in, by cut index: small chunks (the decode kernels), odd ones, large ones.
fn grid(k: usize) -> &'static [usize] {
    const G: [&[usize]; 5] = [&[1024], &[129], &[300, 100, 700], &[128], &[512, 1, 255]];
    G[k % G.len()]
}

#[test]
#[ignore]
fn resume_equals_cold_at_every_cut() {
    let Some(mut m) = model() else { return };
    let m = &mut *m;
    let toks = ids(1, N);
    let stranger = ids(99, 300);
    let want = cold(m, 0, &toks);
    let free_before = m.free_pages();
    let mut compared = 0;
    let cuts = cuts();
    for (k, &c) in cuts.iter().enumerate() {
        // the donor: the prompt up to the cut, in chunks that end at it; the snapshot is taken when it is quiet, its pages kept by reference; the donor goes
        m.begin_slot(1).unwrap();
        assert!(prefill(m, 1, &toks, 0, c, grid(k), None).is_none());
        let at_cut = m.prefix_digest(1, c).unwrap();
        let snap = m.alloc_snapshot().expect("a free snapshot slot");
        m.prefix_snapshot(1, snap, c).unwrap();
        let pages = m.retain_prefix_pages(1, c);
        m.end_slot(1);
        // a stranger's prompt fills the slot that will be restored into: its KDA state (and parity), its ring rows, its pages' pools, its drafter's store
        m.begin_slot(2).unwrap();
        prefill(m, 2, &stranger, 0, stranger.len(), &[256], Some(4)).unwrap();
        m.restore_prefix(2, snap, &pages, c).unwrap();
        let restored = m.prefix_digest(2, c).unwrap();
        assert!(
            restored == at_cut,
            "cut {c} (residue {}): the restored slot is not the state the donor had at the cut: {}",
            c % 4,
            restored.describe_diff(&at_cut)
        );
        // the rest of the prompt, in another grid than the donor's
        let (cand, logits) = prefill(m, 2, &toks, c, N, grid(k + 2), Some(8)).unwrap();
        let got = finish(m, 2, &toks, cand, logits);
        assert!(
            got == want,
            "cut {c} (residue {}): the resumed prompt differs from the cold one: {}",
            c % 4,
            describe(&got, &want)
        );
        m.end_slot(2);
        m.release_prefix(snap, &pages);
        assert_eq!(m.free_pages(), free_before, "cut {c}: every page is back after the entry and both slots are gone");
        assert_eq!(m.prefix_free_snapshots(), SNAPSHOTS);
        compared += 1;
        eprintln!("cut {c:5} (mod 4 = {}) donor grid {:?}, resumed grid {:?}: identical", c % 4, grid(k), grid(k + 2));
    }
    assert_eq!(compared, cuts.len());
}

/// Snapshots taken INSIDE a large prefill chunk (`prefill_chunk_cuts`: no extra forward) are the snapshots of a donor that stopped at the cut: the state a
/// restore leaves equals the state after a cold prefill of exactly the cut's tokens, and the prompt resumed from it equals the cold prompt. The chunks are
/// placed at the start of the prompt and after an earlier chunk, with the cuts at every residue mod 4, one row after the chunk's start, one row before
/// its end, on a slab boundary (a slab is 128 rows) and across the 2,048 sparse boundary.
#[test]
#[ignore]
fn interior_cuts_equal_cold_prefill() {
    let Some(mut m) = model() else { return };
    let m = &mut *m;
    let toks = ids(5, N);
    let stranger = ids(98, 300);
    let want = cold(m, 0, &toks);
    let free_before = m.free_pages();
    type Case = (usize, usize, &'static [usize]);
    const CASES: [Case; 6] = [
        (0, 700, &[101]),
        (0, 1000, &[129, 130, 131, 257, 700, 999]),
        (300, 600, &[301, 556, 899]),
        (1800, 600, &[2047, 2048, 2049, 2051, 2100]),
        // the chunks a decode-kernel chunk would have been: a first chunk of 100 rows, and the tail of a prompt (60 rows, the cut 16 tokens before its end)
        (0, 100, &[44]),
        (512, 60, &[556]),
    ];
    // the cases to run: all, or the comma list of indices in REXL3_GLM_PREFIX_CASES (the mutation tool runs two of them per mutant)
    let only: Option<Vec<usize>> = std::env::var("REXL3_GLM_PREFIX_CASES").ok().map(|v| {
        v.split(',').map(|x| x.trim().parse().expect("REXL3_GLM_PREFIX_CASES is a comma list of case indices")).collect()
    });
    let mut compared = 0;
    for (k, &(start, len, cuts)) in CASES.iter().enumerate() {
        if only.as_ref().is_some_and(|o| !o.contains(&k)) {
            continue;
        }
        // the donor: [0, start) in its own chunks, then ONE chunk of `len` rows with the snapshots taken inside it
        m.begin_slot(1).unwrap();
        if start > 0 {
            prefill(m, 1, &toks, 0, start, &[512], None);
        }
        let mut cut_snaps = Vec::new();
        for &c in cuts {
            assert!(c > start && c < start + len);
            cut_snaps.push((c, m.alloc_snapshot().expect("a free snapshot slot")));
        }
        m.prefill_chunk_cuts(1, start, &toks[start..start + len], None, Vec::new(), &cut_snaps).unwrap();
        let held: Vec<Vec<u32>> = cuts.iter().map(|&c| m.retain_prefix_pages(1, c)).collect();
        m.end_slot(1);
        for (j, &(c, snap)) in cut_snaps.iter().enumerate() {
            // the reference: a cold donor that stops exactly at the cut
            m.begin_slot(3).unwrap();
            prefill(m, 3, &toks, 0, c, grid(k + j), None);
            let at_cut = m.prefix_digest(3, c).unwrap();
            m.end_slot(3);
            // a dirty slot takes the restore
            m.begin_slot(2).unwrap();
            prefill(m, 2, &stranger, 0, stranger.len(), &[256], Some(4)).unwrap();
            m.restore_prefix(2, snap, &held[j], c).unwrap();
            let restored = m.prefix_digest(2, c).unwrap();
            assert!(
                restored == at_cut,
                "chunk {start}+{len}, cut {c} (residue {}): the snapshot taken inside the chunk is not the state after a cold prefill of {c} tokens: {}",
                c % 4,
                restored.describe_diff(&at_cut)
            );
            let (cand, logits) = prefill(m, 2, &toks, c, N, grid(k + j + 2), Some(8)).unwrap();
            let got = finish(m, 2, &toks, cand, logits);
            assert!(
                got == want,
                "chunk {start}+{len}, cut {c} (residue {}): the resumed prompt differs from the cold one: {}",
                c % 4,
                describe(&got, &want)
            );
            m.end_slot(2);
            compared += 1;
            eprintln!("chunk {start}+{len}: cut {c:5} (mod 4 = {}) taken inside the chunk: identical", c % 4);
        }
        for (j, &(_, snap)) in cut_snaps.iter().enumerate() {
            m.release_prefix(snap, &held[j]);
        }
        assert_eq!(m.free_pages(), free_before, "chunk {start}+{len}: every page is back");
        assert_eq!(m.prefix_free_snapshots(), SNAPSHOTS);
    }
    if only.is_none() {
        assert_eq!(compared, 17);
    } else {
        assert!(compared > 0);
    }
}

#[test]
#[ignore]
fn snapshots_inside_a_chunk_are_refused_where_the_state_cannot_be_taken() {
    let Some(mut m) = model() else { return };
    let m = &mut *m;
    let toks = ids(6, 600);
    let snap = m.alloc_snapshot().unwrap();
    m.begin_slot(1).unwrap();
    // a chunk of one row has no inside, whatever the cut
    assert!(m.prefill_chunk_cuts(1, 0, &toks[..1], None, Vec::new(), &[(1, snap)]).is_err());
    // a cut at the chunk's start or end is not inside it; descending or repeated cuts and a snapshot slot out of range are refused
    assert!(m.prefill_chunk_cuts(1, 0, &toks[..400], None, Vec::new(), &[(400, snap)]).is_err());
    assert!(m.prefill_chunk_cuts(1, 0, &toks[..400], None, Vec::new(), &[(0, snap)]).is_err());
    assert!(m.prefill_chunk_cuts(1, 0, &toks[..400], None, Vec::new(), &[(200, snap), (100, snap)]).is_err());
    assert!(m.prefill_chunk_cuts(1, 0, &toks[..400], None, Vec::new(), &[(100, snap), (200, snap)]).is_err());
    assert!(m.prefill_chunk_cuts(1, 0, &toks[..400], None, Vec::new(), &[(100, SNAPSHOTS as u32)]).is_err());
    // nothing was touched by a refusal: the slot still prefills from its start
    assert!(prefill(m, 1, &toks, 0, 600, &[512], Some(4)).is_some());
    m.end_slot(1);
    m.release_prefix(snap, &[]);
    assert_eq!(m.prefix_free_snapshots(), SNAPSHOTS);
}

#[test]
#[ignore]
fn the_donor_goes_on_and_a_sharer_comes_later() {
    let Some(mut m) = model() else { return };
    let m = &mut *m;
    let toks = ids(2, N);
    let want = cold(m, 0, &toks);
    let c = 1001; // a partial last page (233 of 256 rows), residue 1: the first pool group is incomplete
                  // the donor takes the snapshot, then a first sharer restores and prefills over the partial page while the donor is still alive and only then goes on
    m.begin_slot(1).unwrap();
    prefill(m, 1, &toks, 0, c, &[512, 489], None);
    let snap = m.alloc_snapshot().unwrap();
    m.prefix_snapshot(1, snap, c).unwrap();
    let pages = m.retain_prefix_pages(1, c);
    let shared: Vec<u32> = m.slot_pages(1).iter().map(|p| p.0).collect();
    m.restore_prefix(2, snap, &pages, c).unwrap();
    let mine = m.slot_pages(2);
    // the full pages are the donor's and the entry's, the partial last page is the sharer's own
    assert_eq!(mine[..3].iter().map(|p| p.0).collect::<Vec<_>>(), shared[..3], "the three full pages are shared");
    assert!(mine[..3].iter().all(|p| p.1 == 3), "each is held by the donor, the entry and the sharer: {mine:?}");
    assert_ne!(mine[3].0, shared[3], "the partial last page is copied");
    assert_eq!(mine[3].1, 1);
    let (cand, logits) = prefill(m, 2, &toks, c, N, &[700], Some(8)).unwrap();
    let first = finish(m, 2, &toks, cand, logits);
    assert!(first == want, "the first sharer: {}", describe(&first, &want));
    m.end_slot(2);
    // the donor, whose partial page the sharer's prefill must not have touched, goes on and finishes
    let (cand, logits) = prefill(m, 1, &toks, c, N, &[512], Some(8)).unwrap();
    let donor = finish(m, 1, &toks, cand, logits);
    assert!(donor == want, "the donor after a sharer wrote over the same positions: {}", describe(&donor, &want));
    m.end_slot(1);
    // a second sharer restores AFTER the donor wrote above the cut in the page the entry still holds: the copy carries rows the entry never reads
    m.restore_prefix(3, snap, &pages, c).unwrap();
    let (cand, logits) = prefill(m, 3, &toks, c, N, &[300, 129], Some(8)).unwrap();
    let second = finish(m, 3, &toks, cand, logits);
    assert!(second == want, "the second sharer, after the donor went on: {}", describe(&second, &want));
    m.end_slot(3);
    m.release_prefix(snap, &pages);
    assert_eq!(m.prefix_free_snapshots(), SNAPSHOTS);
}

#[test]
#[ignore]
fn a_restore_of_the_wrong_snapshot_is_refused_and_leaks_nothing() {
    let Some(mut m) = model() else { return };
    let m = &mut *m;
    let toks = ids(3, 700);
    m.begin_slot(1).unwrap();
    prefill(m, 1, &toks, 0, 700, &[512], None);
    let free0 = m.free_pages();
    let snap = m.alloc_snapshot().unwrap();
    m.prefix_snapshot(1, snap, 700).unwrap();
    let pages = m.retain_prefix_pages(1, 700);
    m.end_slot(1);
    // a length the snapshot is not for, a snapshot slot that holds nothing, and a page list of the wrong size
    assert!(m.restore_prefix(2, snap, &pages, 699).is_err());
    let empty = m.alloc_snapshot().unwrap();
    assert!(m.restore_prefix(2, empty, &pages, 700).is_err());
    assert!(m.restore_prefix(2, snap, &pages[..2], 700).is_err());
    m.release_prefix(empty, &[]);
    m.release_prefix(snap, &pages);
    assert_eq!(m.free_pages(), free0 + 3, "the three pages the entry held are back, nothing else moved");
    assert_eq!(m.prefix_free_snapshots(), SNAPSHOTS);
}

#[test]
#[ignore]
fn allocations_equal_the_plan_with_the_cache() {
    let Some(m) = model() else { return };
    let p = AllocParams {
        slots: SLOTS,
        kv_blocks: KV_BLOCKS,
        max_context: MAX_CONTEXT,
        max_rows: gb_model::families::glm::alloc::MAX_ROWS,
        window_rows: gb_model::families::glm::alloc::WINDOW_ROWS,
        mtp: m.has_mtp(),
    };
    let plan = AllocPlan::new_chunked(&m.cfg, &p, m.chunk_rows())
        .unwrap()
        .with_prefix_snapshots(&m.cfg, SNAPSHOTS, m.has_mtp())
        .unwrap();
    m.check_allocations(&plan).unwrap_or_else(|e| panic!("the memory model and the allocations disagree: {e}"));
    let per = prefix::snapshot_bytes(&m.cfg, true).unwrap();
    assert_eq!(plan.get("prefix_snapshots"), SNAPSHOTS as u64 * per);
    eprintln!(
        "snapshot slot {per} B ({:.1} MiB), {SNAPSHOTS} slots {:.1} MiB",
        per as f64 / 1048576.0,
        (SNAPSHOTS as u64 * per) as f64 / 1048576.0
    );
}
