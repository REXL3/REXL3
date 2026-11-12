//! The Nemotron prefix cache (milestone M5, `docs/NEMOTRON_LONG_CONTEXT.md`) over the fake runtime of `nemotron_omni_backend_support`
//! (`SnapFake`: refcounted pages, a head stripe in them, the Mamba state `(s, conv)`, the head's carry and `TapBook`, snapshots in an arena)
//! and the family's real memory model. No GPU, no pack.
//!
//! What these tests pin, and what each fault of the fake (`SnapFault`) needs to be caught:
//!
//! * **cached == uncached == serial** through the engine, tokens, finish reason AND the head's `drafted`/`accepted`/`rounds` (the head's
//!   cache is a function of the committed rows alone and row-invariant, so a hit changes none of them), at cuts of every residue mod 64 and
//!   mod 256, aligned and not;
//! * the **books**: with nothing running, the pages and the arena slots in use are exactly what the index holds;
//! * a restored slot **equals a cold donor that stopped at the cut** (state, conv, the pages' tokens, the head's entries, the carry, the book)
//!   and the prompt resumed from it equals the cold prompt, after a stranger dirtied the slot;
//! * the plan: the cache is carved only for a runtime type that can snapshot, the pool holds its pages and admission is not told, an
//!   explicit size is taken first and refused by name below two units, `Auto` never shrinks the KV plan, and the numbers of the design
//!   document's tables.

mod nemotron_omni_backend_support;

use gb_engine::{
    Backend, CacheBudget, CutPolicy, Drafter, Engine, EngineConfig, ImageSpan, NoDrafts, Output, PrefixConfig, PrefixHandle,
    Request, RowRequest,
};
use gb_model::families::nemotron_h_omni::backend::{ModelOptions, NemotronBackend, NemotronRuntime};
use gb_model::families::nemotron_h_omni::config::OmniConfig;
use gb_model::families::nemotron_h_omni::engine::{
    assemble_with_cost, build, plan_serving, plan_serving_for, PlanInputs, ServePlan,
};
use gb_model::families::nemotron_h_omni::memory::NemotronMemory;
use gb_model::families::nemotron_h_omni::model::{chunk_bucket, NemotronModel};
use gb_model::families::nemotron_h_omni::prefix;
use gb_model::families::nemotron_h_omni::spec::{CostTables, SpecConfig};
use gb_model::family::{BuildCtx, FamilyPlan, ModelFamily, VisionSpec};
use gb_model::hostmem::MemInfo;
use gb_model::serve::{PrefixCacheOption, ServeOptions, SpecMode};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};
use nemotron_omni_backend_support::{fake_features, Call, FakeRuntime, HeadScript, Site, SlotImage, SnapFake, SnapFault, VOCAB};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

const GIB: u64 = 1 << 30;
const PACK: &str = include_str!("../../../testdata/nemotron_omni/config.json");
/// Slots of the engines below, and the pool's pages (every slot a few pages, and room for the entries' own).
const SLOTS: usize = 4;
const PAGES: usize = 768;
const TAIL: usize = 8;
const NS: u64 = 0x006E_656D_6F70_6678;
/// The snapshot slot and the page the budgets are made of (the real figures, with the head).
const SNAP_BYTES: u64 = 170_246_144;
const PAGE_BYTES: u64 = 2_359_296;

type PBackend = NemotronBackend<SnapFake>;

fn budget(units: usize) -> CacheBudget {
    CacheBudget { snapshot_slots: units, page_blocks: units * 16, snapshot_bytes: SNAP_BYTES, page_bytes: PAGE_BYTES }
}

/// Every cut a test can ask for is allowed: the policy's floor is one slice and its gap nothing.
fn policy() -> PrefixConfig {
    PrefixConfig {
        policy: CutPolicy { min_cut: 64, tail: TAIL, ghost_min_gain: 256, min_gap: 0 },
        ghost_blocks: 1 << 12,
        ..Default::default()
    }
}

fn toks(seed: u64, n: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            4 + (x % (VOCAB as u64 - 8)) as u32
        })
        .collect()
}

/// A head that is right most of the time and not always, so that the speculation counters depend on its cache.
fn lossy_head() -> HeadScript {
    HeadScript::Truth { accept: vec![0.85, 0.7, 0.55, 0.4] }
}

fn runtime(slots: usize, units: usize, fault: SnapFault) -> SnapFake {
    let mut rt = SnapFake::new(slots, units.max(1), PAGES, fault);
    rt.head = lossy_head();
    rt
}

fn model_options(slots: usize) -> ModelOptions {
    ModelOptions { slots, kv_pages: PAGES, max_context: 16_384, prefill_rows: 64, layers: None, mtp: true }
}

/// The backend with a cache of `units` snapshot slots and the head drafting (`--spec mtp`).
fn snap_backend(slots: usize, units: usize, fault: SnapFault) -> PBackend {
    NemotronBackend::with_runtime(runtime(slots, units, fault), 64, &model_options(slots))
        .unwrap()
        .with_speculation(SpecConfig::new(SpecMode::Mtp), &CostTables::shipped())
        .unwrap()
        .with_prefix_cache(NS, budget(units))
        .unwrap()
}

fn make_engine(slots: usize, units: usize, fault: SnapFault, cache: bool, drafting: bool) -> Engine<PBackend> {
    let backend = if cache {
        snap_backend(slots, units, fault)
    } else {
        NemotronBackend::with_runtime(runtime(slots, units, fault), 64, &model_options(slots))
            .unwrap()
            .with_speculation(SpecConfig::new(SpecMode::Mtp), &CostTables::shipped())
            .unwrap()
    };
    let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 9216, state_bytes_per_slot: 170_754_048, activation_reserve: 0 };
    let admission = Admission::new(96 << 30, 0, mm, slots).unwrap();
    // flat: the arbiter (here a fixed depth) decided how many rows ride, as the production engine's budget is under `--spec mtp`
    let cost = CostModel::from_table(vec![(1, 1.0), (64, 1.0)]);
    let make: Box<dyn Fn() -> Box<dyn Drafter>> = Box::new(|| Box::new(NoDrafts));
    let e = Engine::new(backend, admission, EngineConfig { max_rows: 64, max_drafts: 7, cost, drafting }, make);
    if cache {
        e.with_prefix_cache(policy())
    } else {
        e
    }
}

fn cached(units: usize, fault: SnapFault) -> Engine<PBackend> {
    make_engine(SLOTS, units, fault, true, true)
}

fn request(prompt: Vec<u32>, max_new: usize, seed: u64, temperature: f64) -> Request {
    Request {
        prompt,
        sampling: Sampling { seed, temperature, top_k: 20, top_p: 0.95, min_p: 0.0 },
        max_new,
        stop: vec![],
        draft: true,
        images: vec![],
        cache: true,
    }
}

fn greedy(prompt: Vec<u32>, max_new: usize) -> Request {
    request(prompt, max_new, 7, 0.0)
}

fn one(e: &mut Engine<PBackend>, r: Request) -> Output {
    let id = e.submit(r).unwrap();
    e.run().unwrap();
    assert!(e.take_failed().is_empty(), "no request may fail");
    let mut outs = e.take_outputs();
    outs.retain(|o| o.id == id);
    assert_eq!(outs.len(), 1);
    outs.remove(0)
}

/// Tokens, finish reason AND the head's counters: the drafter's state is a function of the committed rows alone, so a hit changes none.
fn same(a: &Output, b: &Output) -> Result<(), String> {
    if a.tokens != b.tokens {
        let at = a.tokens.iter().zip(&b.tokens).position(|(x, y)| x != y).unwrap_or(a.tokens.len().min(b.tokens.len()));
        return Err(format!("tokens differ at {at} ({} vs {} tokens)", a.tokens.len(), b.tokens.len()));
    }
    if (a.finish, a.stopped) != (b.finish, b.stopped) {
        return Err(format!("finish {:?} vs {:?}", a.finish, b.finish));
    }
    if (a.drafted, a.accepted, a.rounds) != (b.drafted, b.accepted, b.rounds) {
        return Err(format!(
            "speculation counters differ: drafted/accepted/rounds {}/{}/{} vs {}/{}/{}",
            a.drafted, a.accepted, a.rounds, b.drafted, b.accepted, b.rounds
        ));
    }
    Ok(())
}

/// The uncached, drafted reply: a server without the cache.
fn reference(r: &Request) -> Output {
    let mut e = make_engine(SLOTS, 1, SnapFault::None, false, true);
    one(&mut e, Request { cache: false, ..r.clone() })
}

/// The serial reply (`--spec none`): no drafts at all.
fn serial(r: &Request) -> Output {
    let mut e = make_engine(SLOTS, 1, SnapFault::None, false, false);
    one(&mut e, Request { cache: false, draft: false, ..r.clone() })
}

/// Pages and arena slots are all accounted for: with nothing running, what the runtime holds is exactly what the index holds.
fn no_leaks(e: &Engine<PBackend>) -> Result<(), String> {
    let st = e.prefix_stats();
    if e.backend.m.pages_in_use() != st.pages_held as usize {
        return Err(format!("{} pages in use, the index holds {}", e.backend.m.pages_in_use(), st.pages_held));
    }
    if e.backend.m.snapshots_in_use() != st.entries as usize {
        return Err(format!("{} snapshot slots in use, the index holds {} entries", e.backend.m.snapshots_in_use(), st.entries));
    }
    if !e.backend.m.violations.is_empty() {
        return Err(format!("the backend broke the runtime's contract: {:?}", e.backend.m.violations));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------------------------------------------
// the backend: the contract of its hooks
// ------------------------------------------------------------------------------------------------------------------------------------

#[test]
fn a_runtime_that_cannot_snapshot_has_no_cache_and_the_backend_refuses_one_by_name() {
    // the plain fake keeps the defaults: not supported, every snapshot call is NotBuilt
    assert!(!FakeRuntime::snapshots_supported());
    assert!(SnapFake::snapshots_supported());
    let mut rt = FakeRuntime::new(2);
    assert!(rt.enable_snapshots(4).unwrap_err().to_string().contains("nemotron prefix snapshots"));
    assert!(rt.snapshot_slot(0, 10).unwrap_err().to_string().contains("not built") || rt.snapshot_slot(0, 10).is_err());
    assert_eq!((rt.free_snapshots(), rt.snapshot_arena_bytes()), (0, 0));
    // a backend over it takes no cache: refused by name, nothing planned
    let opts = ModelOptions { slots: 2, kv_pages: 64, max_context: 16_384, prefill_rows: 64, layers: None, mtp: false };
    let e =
        NemotronBackend::with_runtime(FakeRuntime::new(2), 64, &opts).unwrap().with_prefix_cache(NS, budget(4)).err().unwrap();
    assert!(e.to_string().contains("cannot snapshot"), "{e}");
    let b = NemotronBackend::with_runtime(FakeRuntime::new(2), 64, &opts).unwrap();
    assert!(b.prefix_caps().is_none() && b.prefix_path(1000).is_none());
    // a budget that holds no entry is refused too
    let e = NemotronBackend::with_runtime(runtime(2, 2, SnapFault::None), 64, &model_options(2))
        .unwrap()
        .with_prefix_cache(NS, CacheBudget::NONE)
        .err()
        .unwrap();
    assert!(e.to_string().contains("at least one snapshot slot"), "{e}");
}

#[test]
fn a_backend_without_a_cache_ignores_the_cuts_it_is_handed_and_runs_the_plain_slice() {
    // the engine hands cuts only to a backend that has a cache; one that was handed them anyway takes no snapshot and fails nothing
    let opts = ModelOptions { slots: 2, kv_pages: 64, max_context: 16_384, prefill_rows: 64, layers: None, mtp: false };
    let mut b = NemotronBackend::with_runtime(FakeRuntime::new(2), 64, &opts).unwrap();
    let prompt = toks(8, 200);
    let c = b.prefill_slice_cuts(0, &prompt, &[], 0, 4, &[30, 100]);
    assert_eq!((c.step.done, c.captured.len()), (64, 0), "no cache: the slice is the grid's, a cut shortens nothing");
    assert!(b.take_error().is_none());
    // and a release with no cache is an error by name, not a call into the runtime
    b.prefix_release(PrefixHandle { snap: 0, pages: vec![] });
    assert!(b.take_error().unwrap().message.contains("no prefix cache"));
    assert!(b.m.violations.is_empty());
}

#[test]
fn the_caps_the_path_and_the_free_count_follow_the_cache() {
    let mut b = snap_backend(2, 3, SnapFault::None);
    let caps = b.prefix_caps().expect("a cache");
    assert_eq!((caps.namespace, caps.budget), (NS, budget(3)));
    // the path follows the position rule: a prompt whose rows are all below 8,192 is the tree's arithmetic alone, a longer one has rows
    // the flash kernel computed (tripwire: the id is the prompt's total, not the length of a slice or of a cut)
    for total in [1usize, 2, 63, 64, 100, 128, 129, 8191, 8192] {
        assert_eq!(b.prefix_path(total), Some(prefix::PATH_ID_TREE), "{total}");
    }
    for total in [8193usize, 8256, 10_000, 1_000_000] {
        assert_eq!(b.prefix_path(total), Some(prefix::PATH_ID_FLASH), "{total}");
    }
    assert_ne!(prefix::PATH_ID_TREE, prefix::PATH_ID_FLASH);
    assert_eq!(b.prefix_free_snapshots(), 3);
    // a snapshot takes a slot and a release gives it back, once
    let prompt = toks(5, 200);
    let c = b.prefill_slice_cuts(0, &prompt, &[], 0, 4, &[64]);
    assert_eq!((c.step.done, c.captured.len()), (64, 1));
    assert_eq!(b.prefix_free_snapshots(), 2);
    let cap = c.captured.into_iter().next().unwrap();
    assert_eq!(cap.len, 64);
    assert_eq!(cap.handle.pages.len(), 1);
    b.prefix_release(cap.handle.clone());
    assert_eq!(b.prefix_free_snapshots(), 3);
    assert!(b.take_error().is_none());
    // a second release of the same handle is the runtime's error by name, recorded (retryable), and changes nothing
    b.prefix_release(cap.handle);
    let e = b.take_error().expect("a double release is an error");
    assert!(!e.fatal && e.message.contains("not live"), "{e:?}");
    assert_eq!(b.prefix_free_snapshots(), 3);
}

#[test]
fn a_cut_inside_a_slice_ends_it_there_and_the_grid_after_it_is_the_one_the_prompt_would_have_had() {
    let mut b = snap_backend(2, 4, SnapFault::None);
    let prompt = toks(1, 300);
    let (mut done, mut ends, mut caps) = (0usize, Vec::new(), Vec::new());
    while done < prompt.len() {
        let c = b.prefill_slice_cuts(0, &prompt, &[], done, 4, &[30, 100, 200]);
        caps.extend(c.captured.iter().map(|x| x.len));
        done = c.step.done;
        ends.push(done);
        assert_eq!(c.step.row.is_some(), done == prompt.len());
    }
    assert_eq!(ends, vec![30, 64, 100, 128, 192, 200, 256, 300]);
    assert_eq!(caps, vec![30, 100, 200]);
    let slices: Vec<(usize, usize)> =
        b.m.calls
            .iter()
            .filter_map(|c| match c {
                Call::Prefill { first, rows, .. } => Some((*first, *rows)),
                _ => None,
            })
            .collect();
    assert_eq!(slices, vec![(0, 30), (30, 34), (64, 36), (100, 28), (128, 64), (192, 8), (200, 56), (256, 44)]);
    assert_eq!(b.m.snapshots_in_use(), 3);
    assert!(b.m.violations.is_empty(), "{:?}", b.m.violations);
    // a cut on the grid costs no extra slice: the slice ends where it would have anyway
    let mut g = snap_backend(2, 4, SnapFault::None);
    let c = g.prefill_slice_cuts(0, &prompt, &[], 0, 4, &[64, 128]);
    assert_eq!((c.step.done, c.captured.len()), (64, 1));
    // a cut at the prompt's end is no cut (a strict prefix leaves a row to compute)
    let mut g = snap_backend(2, 4, SnapFault::None);
    let p = &prompt[..100];
    let c = g.prefill_slice_cuts(0, p, &[], 0, 4, &[100]);
    assert_eq!((c.step.done, c.captured.len()), (64, 0));
    let c = g.prefill_slice_cuts(0, p, &[], 64, 4, &[100]);
    assert_eq!((c.step.done, c.captured.len(), c.step.row.is_some()), (100, 0, true));
    assert_eq!(g.m.snapshots_in_use(), 0);
}

#[test]
fn a_prompt_with_cuts_leaves_exactly_the_state_of_one_without() {
    let prompt = toks(2, 333);
    let mut plain = snap_backend(2, 4, SnapFault::None);
    let want_row = plain.prefill(0, 0, &prompt, 8);
    let want = plain.m.slot_image(0).unwrap();
    for cuts in [vec![1usize], vec![63, 64, 65], vec![100, 200, 300], vec![255, 256, 257], vec![332]] {
        let mut b = snap_backend(2, 4, SnapFault::None);
        let (mut done, mut row) = (0usize, None);
        while done < prompt.len() {
            let c = b.prefill_slice_cuts(0, &prompt, &[], done, 8, &cuts);
            done = c.step.done;
            row = c.step.row;
        }
        assert!(row.as_ref().unwrap().bits_eq(&want_row), "last row, cuts {cuts:?}");
        assert_eq!(b.m.slot_image(0).unwrap(), want, "state, cuts {cuts:?}");
        assert_eq!(b.m.snapshots_in_use(), cuts.len());
        assert!(b.m.violations.is_empty());
    }
}

/// The GPU test's shape (`glm_prefix.rs::resume_equals_cold_at_every_cut`) on the fake: the donor stops at the cut, takes the snapshot and is
/// released; a stranger's prompt dirties the slot; the restore is compared AT THE CUT with a cold donor that stopped exactly there and the
/// prompt resumed from it AT THE END (last row, state, six decoded tokens) with the cold prompt; then the books.
fn resume_equals_cold(fault: SnapFault, cut: usize, rest: usize) -> Result<(), String> {
    let prompt = toks(cut as u64 * 31 + 7, cut + rest);
    let mut b = snap_backend(3, 4, fault);
    let mut done = 0;
    let handle = loop {
        let c = b.prefill_slice_cuts(0, &prompt, &[], done, 4, &[cut]);
        done = c.step.done;
        if let Some(cap) = c.captured.into_iter().next() {
            break cap.handle;
        }
        if done >= prompt.len() {
            return Err(format!("cut {cut}: no snapshot was taken"));
        }
    };
    if done != cut {
        return Err(format!("cut {cut}: the slice ended at {done}"));
    }
    b.release(0);
    // a stranger takes the slot, its pages, its store and its book, and leaves them as they are
    let other = toks(999 + cut as u64, 150 + cut % 97);
    b.prefill(0, 0, &other, 4);
    b.release(0);
    b.prefix_restore(0, &handle, cut).map_err(|e| format!("cut {cut}: the restore failed: {e}"))?;
    let restored = b.m.slot_image(0).ok_or("the restored slot is not begun")?;
    let mut cold = snap_backend(3, 4, SnapFault::None);
    cold.prefill(0, 0, &prompt[..cut], 4);
    let want = cold.m.slot_image(0).unwrap();
    if restored != want {
        return Err(format!(
            "cut {cut} (mod 64 {}, mod 256 {}): the restored slot is not the cold donor's: {}",
            cut % 64,
            cut % 256,
            describe(&restored, &want)
        ));
    }
    // resume to the end in the engine's slices, then decode
    let (mut done, mut last) = (cut, None);
    while done < prompt.len() {
        let s = b.prefill_slice(0, &prompt, &[], done, 4);
        done = s.done;
        last = s.row;
    }
    let mut full = snap_backend(3, 4, SnapFault::None);
    let cold_row = full.prefill(0, 0, &prompt, 4);
    let row = last.ok_or("no last row")?;
    if !row.bits_eq(&cold_row) {
        return Err(format!("cut {cut}: the last row differs from the cold prompt's"));
    }
    if b.m.slot_image(0).unwrap() != full.m.slot_image(0).unwrap() {
        return Err(format!(
            "cut {cut}: the state at the end differs: {}",
            describe(&b.m.slot_image(0).unwrap(), &full.m.slot_image(0).unwrap())
        ));
    }
    let (mut ta, mut tb) = (row.ids[0], cold_row.ids[0]);
    let mut pos = prompt.len() as u64;
    for _ in 0..6 {
        let ra = b.step(&[RowRequest { slot: 0, tokens: vec![ta], first_position: pos }], 4);
        b.commit(0, 1);
        let rb = full.step(&[RowRequest { slot: 0, tokens: vec![tb], first_position: pos }], 4);
        full.commit(0, 1);
        if !ra[0][0].bits_eq(&rb[0][0]) {
            return Err(format!("cut {cut}: a decoded row differs at position {pos}"));
        }
        ta = ra[0][0].ids[0];
        tb = rb[0][0].ids[0];
        pos += 1;
    }
    // the books
    b.release(0);
    b.prefix_release(handle);
    if b.m.pages_in_use() != 0 || b.m.snapshots_in_use() != 0 {
        return Err(format!("cut {cut}: {} pages and {} snapshot slots still held", b.m.pages_in_use(), b.m.snapshots_in_use()));
    }
    if !b.m.violations.is_empty() {
        return Err(format!("cut {cut}: {:?}", b.m.violations));
    }
    Ok(())
}

fn describe(a: &SlotImage, b: &SlotImage) -> String {
    let mut d = Vec::new();
    if a.tokens != b.tokens {
        d.push("the pages' tokens".to_string());
    }
    if a.s != b.s {
        d.push("the recurrent state".into());
    }
    if a.conv != b.conv {
        d.push("the conv history".into());
    }
    if a.head != b.head {
        let at = a.head.iter().zip(&b.head).position(|(x, y)| x != y);
        d.push(format!("the head's entries (first at {at:?})"));
    }
    if a.carry != b.carry {
        d.push("the head's carry".into());
    }
    if a.book != b.book {
        d.push(format!("the head's book {:?} vs {:?}", a.book, b.book));
    }
    d.join(", ")
}

/// Cuts at every residue mod 64 that matters and at both sides of every page edge, the aligned ones (`% 256 == 0`) included.
const CUTS: [usize; 27] = [
    1, 2, 3, 63, 64, 65, 100, 127, 128, 129, 191, 192, 254, 255, 256, 257, 258, 300, 320, 511, 512, 513, 700, 1000, 1023, 1024,
    1025,
];

#[test]
fn a_restored_slot_equals_a_cold_donor_at_every_cut_and_the_resumed_prompt_equals_the_cold_one() {
    for &cut in &CUTS {
        for rest in [1usize, 30, 130] {
            resume_equals_cold(SnapFault::None, cut, rest).unwrap_or_else(|e| panic!("{e}"));
        }
    }
}

#[test]
fn the_first_slice_after_a_restore_continues_the_head_and_the_restored_slot_drafts() {
    // the book `{ ok, first: len, n: 1 }` is what lets the resumed prompt's first slice continue the store: a gap would switch the head off
    let mut b = snap_backend(2, 2, SnapFault::None);
    let prompt = toks(3, 400);
    let mut done = 0;
    let h = loop {
        let c = b.prefill_slice_cuts(0, &prompt, &[], done, 4, &[200]);
        done = c.step.done;
        if let Some(cap) = c.captured.into_iter().next() {
            break cap;
        }
    };
    assert_eq!((done, h.len), (200, 200));
    b.release(0);
    b.prefix_restore(1, &h.handle, 200).unwrap();
    assert_eq!(b.m.mtp_pending_rows(1), Some(1), "the carry is the one pending row");
    let s = b.prefill_slice(1, &prompt, &[], 200, 4);
    assert_eq!(s.done, 256);
    assert!(b.m.head_lost.is_empty(), "{:?}", b.m.head_lost);
    assert!(!b.m.head_off(1));
}

#[test]
fn a_restore_that_fails_gives_the_slot_up_once_and_a_cold_prefill_goes_on() {
    let mut b = snap_backend(2, 2, SnapFault::None);
    let prompt = toks(4, 300);
    let c = b.prefill_slice_cuts(0, &prompt, &[], 0, 4, &[30]);
    let h = c.captured.into_iter().next().unwrap().handle;
    b.release(0);
    // the wrong length is refused by name and changes nothing
    let e = b.prefix_restore(0, &h, 31).unwrap_err();
    assert!(e.contains("refused"), "{e}");
    assert!(b.m.begun().is_empty(), "the slot was given up");
    assert_eq!(b.m.ends(0), 2, "ended once by the release and once by the failed restore: never a third");
    // an injected failure of the restore itself
    b.m.fail = Some((Site::Restore, gb_model::error::ModelError::Request("injected".into())));
    let e = b.prefix_restore(0, &h, 30).unwrap_err();
    assert!(e.contains("injected"), "{e}");
    assert!(b.m.begun().is_empty());
    // the slot is free to begin again and the snapshot is intact
    b.prefix_restore(0, &h, 30).unwrap();
    assert_eq!(b.m.committed(0).unwrap(), &prompt[..30]);
    b.release(0);
    b.prefix_release(h);
    assert_eq!((b.m.pages_in_use(), b.m.snapshots_in_use()), (0, 0));
    assert!(b.m.violations.is_empty(), "{:?}", b.m.violations);
}

#[test]
fn a_snapshot_that_fails_fails_its_slice_and_leaves_nothing_allocated() {
    let mut b = snap_backend(2, 2, SnapFault::None);
    let prompt = toks(6, 200);
    b.m.fail = Some((Site::Snapshot, gb_model::error::ModelError::Request("injected".into())));
    let c = b.prefill_slice_cuts(0, &prompt, &[], 0, 4, &[64]);
    // the slice failed as a whole: an error row, the slot given up, nothing captured, nothing held
    assert!(c.captured.is_empty() && c.step.row.unwrap().values[0].is_nan());
    assert!(b.take_error().is_some());
    assert_eq!((b.m.pages_in_use(), b.m.snapshots_in_use()), (0, 0));
    assert_eq!(b.prefix_free_snapshots(), 2);
}

// ------------------------------------------------------------------------------------------------------------------------------------
// the engine: cached == uncached == serial
// ------------------------------------------------------------------------------------------------------------------------------------

/// The end cut of a prompt of `total` tokens (what the policy plans for an otherwise cold prompt), or 0.
fn end_cut(total: usize) -> usize {
    let c = total.saturating_sub(TAIL);
    if c >= 64 {
        c
    } else {
        0
    }
}

fn sweep(fault: SnapFault, cuts: &[usize], exts: &[usize]) -> Result<usize, String> {
    let mut cases = 0;
    for &c in cuts {
        for &ext in exts {
            let a_prompt = toks(c as u64 + 11, c + TAIL);
            let mut b_prompt = a_prompt[..c].to_vec();
            b_prompt.extend(toks(1000 + ext as u64, ext));
            let (a, b) = (greedy(a_prompt, 24), request(b_prompt, 40, 5, 0.8));
            let mut e = cached(4, fault);
            let oa = one(&mut e, a.clone());
            same(&oa, &reference(&a)).map_err(|m| format!("A at cut {c}: {m}"))?;
            assert_eq!(oa.cached_tokens, 0, "a cold server has nothing to hit");
            // an unrelated prompt takes the slot (its pages, its store, its book) between A and B: whatever a restore does not bring back is the
            // stranger's, not A's
            one(&mut e, greedy(toks(7000 + c as u64, 150), 4));
            let ob = one(&mut e, b.clone());
            same(&ob, &reference(&b)).map_err(|m| format!("B at cut {c}+{ext} (mod 64 {}, mod 256 {}): {m}", c % 64, c % 256))?;
            if ob.cached_tokens != c {
                return Err(format!("B at cut {c}+{ext} hit {} tokens, expected {c}", ob.cached_tokens));
            }
            // serial: the same tokens with no drafts at all
            if ob.tokens != serial(&b).tokens {
                return Err(format!("B at cut {c}+{ext}: the cached reply is not the serial reply"));
            }
            no_leaks(&e).map_err(|m| format!("at cut {c}+{ext}: {m}"))?;
            cases += 2;
        }
    }
    Ok(cases)
}

/// Cuts (the end cut of a prompt of `cut + 8` tokens) on both sides of the slice grid and of the page edges.
const SWEEP_CUTS: [usize; 17] = [64, 65, 100, 127, 128, 129, 255, 256, 257, 300, 511, 512, 513, 768, 1000, 1024, 2049];

#[test]
fn cached_replies_and_the_heads_counters_equal_uncached_and_serial_at_every_cut_and_residue() {
    let n = sweep(SnapFault::None, &SWEEP_CUTS, &[1, 9, 40, 300]).unwrap_or_else(|m| panic!("{m}"));
    assert_eq!(n, SWEEP_CUTS.len() * 4 * 2);
}

#[test]
fn the_counters_are_not_vacuous_a_head_without_its_entries_would_show() {
    // the control of the sweep: the same scenario with the head's stripe not copied differs in the counters, so the counters it compares move
    let e = sweep(SnapFault::HeadStripeNotCopied, &SWEEP_CUTS, &[40]).unwrap_err();
    assert!(e.contains("speculation counters differ"), "{e}");
}

#[test]
fn a_hit_is_counted_and_a_prompt_that_extends_the_cached_one_reuses_exactly_its_cut() {
    let mut e = cached(4, SnapFault::None);
    let a = toks(21, 700);
    let oa = one(&mut e, greedy(a.clone(), 20));
    assert_eq!(oa.cached_tokens, 0);
    let st = e.prefix_stats();
    assert_eq!((st.lookups, st.hits, st.inserts, st.entries), (1, 0, 1, 1));
    // the same prompt again (a retry): the end cut, 692 tokens
    let again = one(&mut e, greedy(a.clone(), 20));
    assert_eq!(again.cached_tokens, 700 - TAIL);
    same(&again, &oa).unwrap();
    // a longer one
    let mut longer = a.clone();
    longer.extend(toks(22, 90));
    let ob = one(&mut e, greedy(longer.clone(), 30));
    assert_eq!(ob.cached_tokens, 700 - TAIL);
    same(&ob, &reference(&greedy(longer, 30))).unwrap();
    let st = e.prefix_stats();
    assert_eq!((st.hits, st.tokens_saved), (2, 2 * 692));
    no_leaks(&e).unwrap();
}

#[test]
fn a_request_that_opts_out_is_untouched() {
    let mut e = cached(4, SnapFault::None);
    let r = Request { cache: false, ..greedy(toks(81, 600), 12) };
    let o = one(&mut e, r.clone());
    let s = e.prefix_stats();
    assert_eq!((o.cached_tokens, s.lookups, s.entries), (0, 0, 0));
    same(&o, &reference(&r)).unwrap();
    one(&mut e, greedy(toks(81, 600), 12));
    assert_eq!(e.prefix_stats().entries, 1);
    let o = one(&mut e, r);
    assert_eq!(o.cached_tokens, 0, "an opted-out request does not hit what is stored either");
    no_leaks(&e).unwrap();
}

#[test]
fn a_differing_prefix_hits_nothing_and_changes_nothing() {
    let mut e = cached(4, SnapFault::None);
    one(&mut e, greedy(toks(91, 700), 12));
    // the same length, one token different early on: no entry is a prefix of it
    let mut other = toks(91, 700);
    other[5] = other[5] % (VOCAB as u32 - 8) + 4;
    other[5] = if other[5] == toks(91, 700)[5] { other[5] + 1 } else { other[5] };
    let r = greedy(other, 30);
    let o = one(&mut e, r.clone());
    assert_eq!(o.cached_tokens, 0);
    same(&o, &reference(&r)).unwrap();
    no_leaks(&e).unwrap();
}

#[test]
fn concurrent_sharers_each_equal_their_uncached_reply_and_the_books_balance() {
    let sys = toks(41, 903);
    let mut e = cached(6, SnapFault::None);
    // two prompts that share the system prompt leave its block-aligned prefix as an entry on the second sighting
    for q in 0..2u64 {
        let mut p = sys.clone();
        p.extend(toks(50 + q, 20));
        one(&mut e, greedy(p, 12));
    }
    let (mut ids, mut reqs) = (Vec::new(), Vec::new());
    for q in 0..3u64 {
        let mut p = sys.clone();
        p.extend(toks(60 + q, 17 + q as usize));
        let r = request(p, 50, 100 + q, 0.7);
        ids.push(e.submit(r.clone()).unwrap());
        reqs.push(r);
    }
    e.run().unwrap();
    assert!(e.take_failed().is_empty());
    let outs = e.take_outputs();
    for (id, r) in ids.iter().zip(&reqs) {
        let o = outs.iter().find(|o| o.id == *id).unwrap();
        assert!(o.cached_tokens >= 768, "a sharer of the system prompt resumes from the shared blocks ({})", o.cached_tokens);
        same(o, &reference(r)).unwrap_or_else(|m| panic!("sharer of {}: {m}", r.prompt.len()));
    }
    no_leaks(&e).unwrap();
}

#[test]
fn a_cache_of_two_entries_evicts_the_least_recently_used_and_every_reply_stays_exact() {
    // many prompts into 2 snapshot slots: each a pair, the second extends the first; the books balance after every one
    let mut e = cached(2, SnapFault::None);
    let mut hits = 0u64;
    for i in 0..9u64 {
        let base = toks(300 + i, 260 + 37 * i as usize);
        let first = greedy(base.clone(), 10);
        let o = one(&mut e, first.clone());
        same(&o, &reference(&first)).unwrap_or_else(|m| panic!("prompt {i}: {m}"));
        let mut ext = base.clone();
        ext.extend(toks(400 + i, 50));
        let second = request(ext, 16, i, 0.6);
        let o = one(&mut e, second.clone());
        same(&o, &reference(&second)).unwrap_or_else(|m| panic!("extension {i}: {m}"));
        hits += u64::from(o.cached_tokens > 0);
        no_leaks(&e).unwrap_or_else(|m| panic!("after pair {i}: {m}"));
    }
    let st = e.prefix_stats();
    assert!(st.entries <= 2 && st.evictions > 0, "{st:?}");
    assert_eq!(hits, 9, "each extension finds its base even with two slots");
    assert_eq!(st.restore_failures, 0);
    // every entry is released at the end: nothing is held
    drop(e);
}

#[test]
fn a_slot_that_ended_mid_prefill_gives_back_its_snapshots() {
    let mut e = cached(4, SnapFault::None);
    let p = toks(55, 900);
    let id = e.submit_deferred(greedy(p, 10)).unwrap();
    // a few slices in: the end cut is not taken yet, the shared cut does not exist, so cancel
    for _ in 0..3 {
        e.step().unwrap();
    }
    assert!(e.cancel(id));
    e.run().unwrap();
    let st = e.prefix_stats();
    assert_eq!(st.entries, 0);
    no_leaks(&e).unwrap();
    // and a prompt cancelled AFTER its first cut (a shared prefix, found on the second sighting) discards the snapshot it took: an earlier
    // prompt that shares P's first two blocks and then goes its own way leaves no entry that is a prefix of P, but its blocks are remembered
    let p = toks(56, 900);
    let mut q = p[..520].to_vec();
    q.extend(toks(57, 180));
    one(&mut e, greedy(q, 4));
    let before = e.prefix_stats();
    let id = e.submit_deferred(greedy(p, 10)).unwrap();
    // 9 slices reach the shared cut at 512; the tenth is past it
    for _ in 0..10 {
        e.step().unwrap();
    }
    assert!(e.backend.m.snapshots_in_use() as u64 > before.entries, "the shared cut was taken and is pending on the prompt");
    assert!(e.cancel(id));
    e.run().unwrap();
    let st = e.prefix_stats();
    assert_eq!(st.captures_discarded, before.captures_discarded + 1);
    assert_eq!(st.entries, before.entries, "a prompt that never completed leaves nothing visible");
    no_leaks(&e).unwrap();
}

// ------------------------------------------------------------------------------------------------------------------------------------
// images: the key is the image's content
// ------------------------------------------------------------------------------------------------------------------------------------

const IMAGE_TOKEN: u32 = 18;

/// A prompt of `before` text tokens, an image of `n` placeholders, then `after` text tokens, and its span (the image's features carry `id`).
fn image_prompt(
    seed: u64,
    before: usize,
    n: usize,
    after: usize,
    feats: &Arc<gb_model::families::nemotron_h_omni::vision::NemotronImageFeatures>,
    key: u64,
) -> (Vec<u32>, ImageSpan) {
    let mut p = toks(seed, before);
    p.extend(std::iter::repeat_n(IMAGE_TOKEN, n));
    p.extend(toks(seed + 1, after));
    let features: Arc<dyn std::any::Any + Send + Sync> = feats.clone();
    (p, ImageSpan { start: before, tokens: n, merged: (n / 4, 4), features, key })
}

fn image_engine(units: usize, cache: bool) -> Engine<PBackend> {
    let mut rt = runtime(SLOTS, units, SnapFault::None).0.with_images();
    rt.head = lossy_head();
    let backend = NemotronBackend::with_runtime(SnapFake(rt), 64, &model_options(SLOTS))
        .unwrap()
        .with_image_token(IMAGE_TOKEN)
        .with_speculation(SpecConfig::new(SpecMode::Mtp), &CostTables::shipped())
        .unwrap();
    let backend = if cache { backend.with_prefix_cache(NS, budget(units)).unwrap() } else { backend };
    let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 9216, state_bytes_per_slot: 170_754_048, activation_reserve: 0 };
    let admission = Admission::new(96 << 30, 0, mm, SLOTS).unwrap();
    let make: Box<dyn Fn() -> Box<dyn Drafter>> = Box::new(|| Box::new(NoDrafts));
    let cost = CostModel::from_table(vec![(1, 1.0), (64, 1.0)]);
    let e = Engine::new(backend, admission, EngineConfig { max_rows: 64, max_drafts: 7, cost, drafting: true }, make);
    if cache {
        e.with_prefix_cache(policy())
    } else {
        e
    }
}

fn with_image(prompt: Vec<u32>, span: ImageSpan, max_new: usize) -> Request {
    Request { images: vec![span], ..greedy(prompt, max_new) }
}

#[test]
fn an_image_prefix_is_keyed_by_its_content_a_cut_never_falls_inside_it_and_the_reply_is_the_uncached_one() {
    let (img_a, img_b) = (fake_features(128, (32, 4)), fake_features(128, (32, 4)));
    // the same text around the same image: a hit that reaches across the image
    let (pa, sa) = image_prompt(11, 300, 128, 200, &img_a, 0xA11CE);
    let mut e = image_engine(4, true);
    let oa = one(&mut e, with_image(pa.clone(), sa.clone(), 20));
    let mut reference_e = image_engine(4, false);
    let want = one(&mut reference_e, Request { cache: false, ..with_image(pa.clone(), sa.clone(), 20) });
    same(&oa, &want).unwrap();
    assert_eq!(oa.cached_tokens, 0);
    // a follow-up: the same prompt and more text after it
    let mut longer = pa.clone();
    longer.extend(toks(12, 80));
    let ob = one(&mut e, with_image(longer.clone(), sa.clone(), 20));
    assert_eq!(
        ob.cached_tokens,
        pa.len() - TAIL,
        "the end cut is past the image (a cut inside would have moved to its first placeholder)"
    );
    let want = one(&mut reference_e, Request { cache: false, ..with_image(longer, sa.clone(), 20) });
    same(&ob, &want).unwrap();
    // ANOTHER image under the same text and the same placeholder tokens: nothing across it is reused (the key is the content, the tokens alone name
    // nothing), and the reply is the uncached reply of that image
    let (pb, sb) = image_prompt(11, 300, 128, 200, &img_b, 0xB0B);
    assert_eq!(pa, pb, "the placeholders and the text are the same tokens");
    let oc = one(&mut e, with_image(pb.clone(), sb.clone(), 20));
    assert_eq!(oc.cached_tokens, 0, "no entry is a prefix of this prompt's key stream");
    let want = one(&mut reference_e, Request { cache: false, ..with_image(pb.clone(), sb.clone(), 20) });
    same(&oc, &want).unwrap();
    // the second sighting of the text before the image left a block-aligned entry (256 tokens, outside the image): a third image reuses exactly that
    let img_c = fake_features(128, (32, 4));
    let (pc, sc) = image_prompt(11, 300, 128, 200, &img_c, 0xC0C);
    let od = one(&mut e, with_image(pc.clone(), sc.clone(), 20));
    assert_eq!(od.cached_tokens, 256, "the text's first block, nothing of the image");
    let want = one(&mut reference_e, Request { cache: false, ..with_image(pc, sc, 20) });
    same(&od, &want).unwrap();
    assert_eq!(e.prefix_stats().restore_failures, 0);
    // a prompt that ENDS in an image: its end cut would fall inside it, so it moves to the image's first placeholder and a follow-up that
    // continues past the image resumes from the state before the image
    let (pe, se) = image_prompt(31, 400, 128, 0, &img_a, 0xA11CE);
    assert_eq!(pe.len(), 528);
    let oe = one(&mut e, with_image(pe.clone(), se.clone(), 8));
    let want = one(&mut reference_e, Request { cache: false, ..with_image(pe.clone(), se.clone(), 8) });
    same(&oe, &want).unwrap();
    let mut pf = pe.clone();
    pf.extend(toks(32, 60));
    let of = one(&mut e, with_image(pf.clone(), se.clone(), 8));
    assert_eq!(of.cached_tokens, 400, "the cut moved to the image's first placeholder");
    let want = one(&mut reference_e, Request { cache: false, ..with_image(pf, se, 8) });
    same(&of, &want).unwrap();
    // an image with no content key (0) bypasses the cache
    let (pz, sz) = image_prompt(11, 300, 128, 200, &img_a, 0);
    let lookups = e.prefix_stats().lookups;
    let oz = one(&mut e, with_image(pz, sz, 10));
    assert_eq!((oz.cached_tokens, e.prefix_stats().lookups), (0, lookups));
    no_leaks(&e).unwrap();
}

// ------------------------------------------------------------------------------------------------------------------------------------
// the faults: each is caught by a named test
// ------------------------------------------------------------------------------------------------------------------------------------

fn expect_caught(fault: SnapFault, cuts: &[usize], needle: &str) -> String {
    sweep(SnapFault::None, cuts, &[40]).expect("the honest fake passes the scenario");
    let e = sweep(fault, cuts, &[40]).expect_err("the fault must be caught");
    assert!(e.contains(needle), "{fault:?}: {e}");
    e
}

#[test]
fn mutant_recurrent_state_not_restored_is_caught() {
    let e = expect_caught(SnapFault::StateNotRestored, &[100, 300, 513], "tokens differ");
    println!("state not restored: {e}");
    assert!(resume_equals_cold(SnapFault::StateNotRestored, 100, 30).unwrap_err().contains("recurrent state"));
}

#[test]
fn mutant_conv_history_not_restored_is_caught() {
    // the conv history steers three rows after the cut: a wrong one shows in the restored slot at once and in some reply
    assert!(resume_equals_cold(SnapFault::ConvNotRestored, 100, 30).unwrap_err().contains("conv history"));
    let mut caught = false;
    for c in [100usize, 128, 300, 513, 700, 1000] {
        if sweep(SnapFault::ConvNotRestored, &[c], &[40]).is_err() {
            caught = true;
        }
    }
    assert!(caught, "a stale conv history must change some reply");
}

#[test]
fn mutant_head_carry_not_restored_is_caught_by_the_counters_alone() {
    // the carry steers the head's first entry after the cut and nothing else: every reply is right, the counters are not
    assert!(resume_equals_cold(SnapFault::CarryNotRestored, 100, 30).unwrap_err().contains("carry"));
    let (mut tokens_differ, mut counters_differ) = (false, false);
    for c in [300usize, 700, 1000] {
        if let Err(m) = sweep(SnapFault::CarryNotRestored, &[c], &[60]) {
            if m.contains("tokens differ") {
                tokens_differ = true;
            }
            if m.contains("speculation counters differ") {
                counters_differ = true;
            }
        }
    }
    assert!(!tokens_differ, "the head's carry can never change a reply");
    assert!(counters_differ, "a stale carry must show in the speculation counters");
}

#[test]
fn mutant_head_bookkeeping_not_restored_is_caught_by_the_counters_alone() {
    // a reset book makes the first resumed chunk a gap: the slot's head is lost, it drafts nothing, every reply is still right
    let r = resume_equals_cold(SnapFault::BookNotRestored, 100, 30).unwrap_err();
    assert!(r.contains("book") || r.contains("head"), "{r}");
    let e = expect_caught(SnapFault::BookNotRestored, &[300, 513], "speculation counters differ");
    assert!(!e.contains("tokens differ"));
}

#[test]
fn mutant_a_partial_last_page_is_shared_is_caught() {
    // a donor still decoding while a sharer prefills over the shared last page: the sharer clobbers the donor's tokens
    let scenario = |fault: SnapFault| -> Result<(), String> {
        let a_prompt = toks(11, 700);
        let mut b_prompt = a_prompt[..end_cut(700)].to_vec();
        b_prompt.extend(toks(12, 120));
        let (a, b) = (greedy(a_prompt, 90), request(b_prompt, 80, 9, 0.7));
        let mut e = cached(4, fault);
        let ida = e.submit(a.clone()).unwrap();
        for _ in 0..14 {
            e.step().map_err(|x| x.to_string())?;
        }
        let idb = e.submit(b.clone()).unwrap();
        e.run().map_err(|x| x.to_string())?;
        let outs = e.take_outputs();
        let get = |id| outs.iter().find(|o| o.id == id).unwrap().clone();
        same(&get(ida), &reference(&a)).map_err(|m| format!("the donor: {m}"))?;
        same(&get(idb), &reference(&b)).map_err(|m| format!("the sharer: {m}"))?;
        no_leaks(&e)
    };
    scenario(SnapFault::None).unwrap();
    let e = scenario(SnapFault::LastPageShared).unwrap_err();
    println!("last page shared: {e}");
    assert!(e.contains("the donor") || e.contains("the sharer"), "{e}");
}

#[test]
fn mutant_a_full_last_page_is_shared_when_the_cut_is_aligned_is_caught() {
    // the Nemotron-specific one: at a cut that is a multiple of 256 the last page is FULL, and the families without a head one position behind
    // the trunk copy only a partial page. Here the resumed prompt's first slice writes the head's entry of position `len - 1` into it, which
    // the donor (still decoding, drafting from its own entries) reads: replies are right, the donor's speculation counters are not
    let scenario = |fault: SnapFault, cut: usize| -> Result<(), String> {
        let a_prompt = toks(21, cut + TAIL);
        let mut b_prompt = a_prompt[..cut].to_vec();
        b_prompt.extend(toks(22, 150));
        let (a, b) = (greedy(a_prompt, 120), greedy(b_prompt, 60));
        let mut e = cached(4, fault);
        let ida = e.submit(a.clone()).unwrap();
        let mut guard = 0;
        // run until the donor has its first token (its prefill is done and its entry is in the index)
        while e.prefix_stats().entries == 0 && guard < 200 {
            e.step().map_err(|x| x.to_string())?;
            guard += 1;
        }
        for _ in 0..3 {
            e.step().map_err(|x| x.to_string())?;
        }
        let idb = e.submit(b.clone()).unwrap();
        e.run().map_err(|x| x.to_string())?;
        let outs = e.take_outputs();
        let get = |id| outs.iter().find(|o| o.id == id).unwrap().clone();
        let donor = get(ida);
        let sharer = get(idb);
        if sharer.cached_tokens != cut {
            return Err(format!("the sharer hit {} tokens, expected {cut}", sharer.cached_tokens));
        }
        same(&donor, &reference(&a)).map_err(|m| format!("the donor: {m}"))?;
        same(&sharer, &reference(&b)).map_err(|m| format!("the sharer: {m}"))?;
        no_leaks(&e)
    };
    // honest at an aligned cut and at an unaligned one
    scenario(SnapFault::None, 512).unwrap();
    scenario(SnapFault::None, 300).unwrap();
    // the fault: caught at an aligned cut, by the donor's counters and not by any reply
    let e = scenario(SnapFault::FullLastPageShared, 512).unwrap_err();
    println!("full last page shared: {e}");
    assert!(e.contains("the donor") && e.contains("speculation counters differ"), "{e}");
    // and it is exactly the aligned cut: at 300 the page is partial, the fault's rule copies it, nothing differs
    scenario(SnapFault::FullLastPageShared, 300).unwrap();
}

#[test]
fn mutant_head_stripe_not_copied_with_the_page_is_caught_by_the_counters_alone() {
    let r = resume_equals_cold(SnapFault::HeadStripeNotCopied, 300, 30).unwrap_err();
    assert!(r.contains("head's entries"), "{r}");
    let e = expect_caught(SnapFault::HeadStripeNotCopied, &[300, 700], "speculation counters differ");
    assert!(!e.contains("tokens differ"), "the head's entries can never change a reply: {e}");
}

#[test]
fn mutant_the_wrong_snapshot_is_caught() {
    // two live entries: the restore of one reads the other's state
    let scenario = |fault: SnapFault| -> Result<(), String> {
        let mut e = cached(4, fault);
        let a = toks(31, 600);
        let c = toks(32, 640);
        one(&mut e, greedy(a.clone(), 8));
        one(&mut e, greedy(c, 8));
        let mut b_prompt = a.clone();
        b_prompt.extend(toks(33, 50));
        let b = request(b_prompt, 30, 3, 0.8);
        let ob = one(&mut e, b.clone());
        same(&ob, &reference(&b)).map_err(|m| format!("B: {m}"))
    };
    scenario(SnapFault::None).unwrap();
    // the arena slot the stranger's entry holds is the one the fault restores from; whichever order they sit in, one of the two scenarios hits it
    let e = scenario(SnapFault::WrongSnapshot).unwrap_err();
    assert!(e.contains("B:"), "{e}");
}

#[test]
fn mutant_a_release_that_keeps_the_slot_unbalances_the_books() {
    // an entry evicted and released must give its arena slot back: with the fault the arena fills and the books are off
    let books = |fault: SnapFault| -> Result<(), String> {
        let mut e = cached(2, fault);
        for i in 0..5u64 {
            one(&mut e, greedy(toks(500 + i, 300 + 10 * i as usize), 4));
            no_leaks(&e).map_err(|m| format!("after {i}: {m}"))?;
        }
        Ok(())
    };
    books(SnapFault::None).unwrap();
    let e = books(SnapFault::ReleaseLeaksSlot).unwrap_err();
    assert!(e.contains("snapshot slots in use"), "{e}");
}

#[test]
fn mutant_a_release_that_keeps_the_pages_unbalances_the_books() {
    let books = |fault: SnapFault| -> Result<(), String> {
        let mut e = cached(2, fault);
        for i in 0..5u64 {
            one(&mut e, greedy(toks(600 + i, 300 + 10 * i as usize), 4));
            no_leaks(&e).map_err(|m| format!("after {i}: {m}"))?;
        }
        Ok(())
    };
    books(SnapFault::None).unwrap();
    let e = books(SnapFault::ReleaseKeepsPages).unwrap_err();
    assert!(e.contains("pages in use"), "{e}");
}

// ------------------------------------------------------------------------------------------------------------------------------------
// the plan
// ------------------------------------------------------------------------------------------------------------------------------------

/// A scratch pack directory: the published config and an index (what `prefix::pack_identity` reads).
fn pack_dir() -> &'static PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join(format!("rexl3-nemotron-prefix-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), PACK).unwrap();
        std::fs::write(dir.join("model.safetensors.index.json"), r#"{"metadata":{},"weight_map":{}}"#).unwrap();
        dir
    })
}

fn cfg() -> OmniConfig {
    OmniConfig::from_json(&serde_json::from_str(PACK).unwrap()).unwrap()
}

fn family(head: bool, tower: bool) -> FamilyPlan {
    let c = cfg();
    FamilyPlan {
        family: "nemotron_h_omni",
        dir: pack_dir().clone(),
        layers: c.layer_specs(),
        mtp_layers: c.mtp_specs(),
        memory: Box::new(NemotronMemory::new(&c, 0)),
        vision: tower.then_some(VisionSpec {
            tower: "nemotron_radio_vision",
            patch_size: 16,
            merge_size: 2,
            temporal_patch_size: 1,
            out_hidden: 4096,
            image_token_id: 18,
            max_image_tokens: 3328,
        }),
        chat_format: "nemotron_omni",
        drafters: if head { vec!["mtp", "copy"] } else { vec!["copy"] },
        required_components: vec![],
    }
}

fn options(slots: usize, max_model_len: usize) -> ServeOptions {
    let mut o = ServeOptions::new(pack_dir().clone());
    o.prefill_chunk = 64;
    o.slots = slots;
    o.max_model_len = max_model_len;
    o.floor_bytes = 8 * GIB;
    o
}

/// MemAvailable of an idle 121 GiB box (115 GiB, measured: `docs/GLM_LOAD.md`) with `loaded` bytes already taken (the tower).
fn host(avail: u64) -> MemInfo {
    MemInfo { total: 121 * GIB, free: avail, available: avail, cached: 0 }
}

fn weights(head: bool) -> u64 {
    // the pack's 63.6 GiB resident with the head; the head's 5,884,651,520 B are not loaded without it
    (63.6 * GIB as f64) as u64 - if head { 0 } else { 5_884_651_520 }
}

fn plan_on(fam: &FamilyPlan, o: &ServeOptions, avail: u64, w: u64, snapshots: bool) -> Result<ServePlan, String> {
    let h = host(avail);
    let inp = PlanInputs { cuda_free: avail, host: Some(&h), weight_bytes: w, model_max_context: cfg().max_context };
    let r = if snapshots { plan_serving_for::<SnapFake>(fam, o, &inp) } else { plan_serving_for::<FakeRuntime>(fam, o, &inp) };
    r.map_err(|e| e.to_string())
}

#[test]
fn the_cache_is_carved_only_for_a_runtime_type_that_can_snapshot() {
    let o = options(2, 262_144);
    let fam = family(true, false);
    let plain = plan_on(&fam, &o, 115 * GIB, weights(true), false).unwrap();
    let snap = plan_on(&fam, &o, 115 * GIB, weights(true), true).unwrap();
    // the plain runtime (and `plan_serving`) is planned exactly as before: no cache, the pool is the plan's blocks
    assert!(!plain.plan.cache.enabled() && plain.pack_identity.is_none());
    assert_eq!(plain.model.kv_pages, plain.plan.kv_blocks);
    let reference = plan_serving(
        &fam,
        &o,
        &PlanInputs {
            cuda_free: 115 * GIB,
            host: Some(&host(115 * GIB)),
            weight_bytes: weights(true),
            model_max_context: 1 << 20,
        },
    )
    .unwrap();
    assert_eq!((reference.plan.clone(), reference.model.clone()), (plain.plan.clone(), plain.model.clone()));
    // the snapshotting type carves it: the default `Auto`, at the default context
    let c = snap.plan.cache;
    assert!(c.enabled());
    assert_eq!((c.snapshot_slots, c.page_blocks), (18, 2304), "an eighth of the 262,144-token context a unit: 128 pages");
    assert_eq!((c.snapshot_bytes, c.page_bytes), (SNAP_BYTES, PAGE_BYTES));
    // KV is the same plan: `Auto` only spends what the pool left over
    assert_eq!(
        (snap.plan.kv_blocks, snap.plan.kv_tokens, snap.plan.admit_bytes),
        (plain.plan.kv_blocks, plain.plan.kv_tokens, plain.plan.admit_bytes)
    );
    // the pool the model is built with holds the cache's pages on top, and admission was never told about them
    assert_eq!(snap.model.kv_pages, snap.plan.kv_blocks + 2304);
    assert_eq!(snap.model.slots, 2);
    assert_eq!(snap.plan.planned_bytes(2), plain.plan.planned_bytes(2) + c.bytes());
    assert!(snap.pack_identity.as_deref().is_some_and(|s| s.len() == 32));
    // the cache only ever spends what the pool left: at most half of the spare, at most the cap
    assert!(c.bytes() <= snap.plan.spare_bytes / 2 && c.bytes() <= 8 * GIB);
    // `--prefix-cache off` and `--prefix-cache-mib 0` plan none, whatever the type
    for opt in [PrefixCacheOption::Off, PrefixCacheOption::Mib(0)] {
        let mut o = o.clone();
        o.prefix_cache = opt;
        assert!(!plan_on(&fam, &o, 115 * GIB, weights(true), true).unwrap().plan.cache.enabled(), "{opt:?}");
    }
}

#[test]
fn the_cache_budget_on_the_121_gib_box_for_the_shipped_contexts() {
    let fam = family(true, false);
    // (max_model_len, entries, pages, one unit in MiB, the cap in GiB): the table of docs/NEMOTRON_LONG_CONTEXT.md section 1.5
    let table = [
        (32_768usize, 41usize, 656usize, 198.4, 8.0),
        (131_072, 26, 1664, 306.4, 8.0),
        (262_144, 18, 2304, 450.4, 8.0),
        (524_288, 11, 2816, 738.4, 8.0),
        (1_048_576, 8, 4096, 1314.4, 10.27),
    ];
    for (ctx, entries, pages, unit, cap) in table {
        let o = options(2, ctx);
        let sp = plan_on(&fam, &o, 115 * GIB, weights(true), true).unwrap();
        let c = sp.plan.cache;
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        eprintln!(
            "ctx {ctx:>9}: kv pool {:.2} GiB, spare {:.1} GiB, cache {} entries x {} pages = {:.2} GiB (unit {:.1} MiB)",
            (sp.plan.kv_blocks as u64 * sp.plan.block_bytes) as f64 / GIB as f64,
            sp.plan.spare_bytes as f64 / GIB as f64,
            c.snapshot_slots,
            c.page_blocks,
            c.bytes() as f64 / GIB as f64,
            mib(c.bytes()) / c.snapshot_slots as f64
        );
        assert_eq!((c.snapshot_slots, c.page_blocks), (entries, pages), "ctx {ctx}");
        let u = mib(prefix::unit_bytes(SNAP_BYTES, PAGE_BYTES, prefix::entry_pages(ctx)));
        assert!((u - unit).abs() < 0.06, "ctx {ctx}: unit {u:.1} MiB");
        assert!(c.bytes() as f64 / GIB as f64 <= cap + 1e-9, "ctx {ctx}");
        // one entry of the longest admissible prompt fits the page budget whatever the context
        assert!(
            c.page_blocks >= ctx.div_ceil(256),
            "ctx {ctx}: {} pages for {} pages of context",
            c.page_blocks,
            ctx.div_ceil(256)
        );
    }
    // the shared planner's split of the same 8 GiB could not store a prompt of 100,000 tokens
    let shared = CacheBudget::fit(8 * GIB, SNAP_BYTES, PAGE_BYTES);
    assert_eq!((shared.snapshot_slots, shared.page_blocks), (45, 360));
}

#[test]
fn an_explicit_size_is_carved_first_split_into_units_and_refused_by_name_below_two() {
    let fam = family(true, false);
    let mut o = options(2, 1 << 20);
    let auto = plan_on(&fam, &o, 115 * GIB, weights(true), true).unwrap();
    // 16 GiB at a 1M context: units of 1,314.4 MiB, 12 of them
    o.prefix_cache = PrefixCacheOption::Mib(16 * 1024);
    let fixed = plan_on(&fam, &o, 115 * GIB, weights(true), true).unwrap();
    let c = fixed.plan.cache;
    assert_eq!((c.snapshot_slots, c.page_blocks), (12, 6144));
    assert!(c.bytes() <= 16 * GIB);
    // taken before the pool is sized: the pool is planned from what remains (here the token cap still binds, so the pool is the same)
    assert_eq!(fixed.plan.kv_tokens, auto.plan.kv_tokens);
    assert_eq!(fixed.model.kv_pages, fixed.plan.kv_blocks + 6144);
    assert_eq!(
        fixed.plan.planned_bytes(2),
        fixed.plan.engine_reserve_bytes
            + fixed.plan.state_bytes_per_slot * 2
            + fixed.plan.kv_blocks as u64 * fixed.plan.block_bytes
            + c.bytes()
    );
    // below two units: refused by name, with the size that would do, before anything is planned from it
    o.prefix_cache = PrefixCacheOption::Mib(2000);
    let e = plan_on(&fam, &o, 115 * GIB, weights(true), true).unwrap_err();
    assert!(
        e.contains("--prefix-cache-mib 2000") && e.contains("two entries") && e.contains("512 pages") && e.contains("2629"),
        "{e}"
    );
    // taken FIRST: on a box where the pool is what the memory limits, the explicit size comes out of the pool and the plan still fits
    // (20 GiB after the weights: the default context's pool would take 2.3 GiB, a 6.5 GiB cache leaves less than that)
    let tight = options(2, 262_144);
    let plain = plan_on(&fam, &tight, 84 * GIB, weights(true), true).unwrap();
    let mut big = tight.clone();
    big.prefix_cache = PrefixCacheOption::Mib(6700);
    let carved = plan_on(&fam, &big, 84 * GIB, weights(true), true).unwrap();
    assert!(carved.plan.cache.enabled());
    assert!(carved.plan.kv_blocks < plain.plan.kv_blocks, "{} pages against {}", carved.plan.kv_blocks, plain.plan.kv_blocks);
    let free_after = 84 * GIB - weights(true);
    assert!(carved.plan.planned_bytes(2) + 8 * GIB <= free_after, "the plan fits what the box has beside the floor");
    // an explicit size the box cannot hold beside the slot state fails closed like any plan
    o.prefix_cache = PrefixCacheOption::Mib(90 * 1024);
    assert!(plan_on(&fam, &o, 115 * GIB, weights(true), true).is_err());
}

#[test]
fn auto_never_makes_the_kv_plan_smaller_and_a_box_with_no_spare_plans_no_cache() {
    let fam = family(true, false);
    // a tight box (20 GiB after the weights): the KV pool of the default context fits and a few GiB are spare
    let o = options(2, 262_144);
    let tight = plan_on(&fam, &o, 84 * GIB, weights(true), true).unwrap();
    let tight_plain = plan_on(&fam, &o, 84 * GIB, weights(true), false).unwrap();
    assert_eq!(tight.plan.kv_blocks, tight_plain.plan.kv_blocks, "the cache never takes the pool's memory under Auto");
    assert_eq!(tight.plan.kv_tokens, tight_plain.plan.kv_tokens);
    let c = tight.plan.cache;
    assert!(c.enabled() && c.bytes() <= tight.plan.spare_bytes / 2, "{c:?} of {} spare", tight.plan.spare_bytes);
    assert_eq!(tight.plan.planned_bytes(2), tight_plain.plan.planned_bytes(2) + c.bytes());
    // a box whose pool takes everything: no spare, no cache, and the plan is the plain one
    let none = plan_on(&fam, &o, 78 * GIB, weights(true), true).unwrap();
    assert!(!none.plan.cache.enabled());
    assert_eq!(none.plan.kv_blocks, plan_on(&fam, &o, 78 * GIB, weights(true), false).unwrap().plan.kv_blocks);
    assert!(none.pack_identity.is_none());
    assert_eq!(none.model.kv_pages, none.plan.kv_blocks);
}

/// A rank of a two-rank tensor-parallel group plans its own memory (`tp_boot::plan_for_rank`: the TP scratch; the shard's weights) and prices
/// the snapshot arena exactly as a single node does: a snapshot slot is the replicated recurrent state, so it is the same bytes, and the
/// cache a rank carves from the same request is the same split. `docs/NEMOTRON_TP.md` 12.5.
#[test]
fn a_rank_of_a_tensor_parallel_group_prices_the_arena_like_a_single_node_and_the_ranks_agree_on_the_budget() {
    let c = cfg();
    let tp_family = |tower: bool| {
        let mut f = family(true, tower);
        f.memory = Box::new(NemotronMemory::new(&c, 0).with_tp_ranks(2));
        f
    };
    // what the ranks were measured to plan with (docs/NEMOTRON_TP.md 12.2): 36 GiB of shard weights each; node 1 holds the vision tower
    // (1.6 GiB of weights and 0.7 GiB of scratch, loaded before the plan measures) and has less memory free than node 4
    let shard = 36 * GIB;
    let (avail0, avail1) = (115 * GIB - 2 * GIB - GIB / 3, 115 * GIB + 2 * GIB);
    for (slots, ctx) in [(2usize, 8192usize), (20, 40_000), (20, 262_144), (4, 131_072)] {
        let o = options(slots, ctx);
        let single = plan_on(&family(true, false), &o, 115 * GIB, weights(true), true).unwrap();
        let r0 = plan_on(&tp_family(true), &o, avail0, shard, true).unwrap();
        let r1 = plan_on(&tp_family(false), &o, avail1, shard, true).unwrap();
        for (who, sp) in [("rank 0", &r0), ("rank 1", &r1)] {
            let (g, s) = (sp.plan.cache, single.plan.cache);
            // the snapshot slot is the same bytes as on one node (the state is replicated, not sharded) and is priced by every rank
            assert_eq!(g.snapshot_bytes, s.snapshot_bytes, "{who} at {slots} slots, {ctx} tokens");
            assert_eq!(g.snapshot_bytes, SNAP_BYTES);
            assert!(g.enabled(), "{who} at {slots} slots, {ctx} tokens: no cache");
            // the cache is part of what the plan holds: the pool holds its pages, and the plan fits beside the floor
            assert_eq!(sp.model.kv_pages, sp.plan.kv_blocks + g.page_blocks);
            assert!(sp.plan.planned_bytes(slots) + 8 * GIB <= sp.free_after_weights + 1, "{who}: the plan fits");
        }
        // the group's budget: rank 1 can hold at least rank 0's (it adopts rank 0's), and where the cap binds they are equal
        let (c0, c1) = (r0.plan.cache, r1.plan.cache);
        assert!(c1.snapshot_slots >= c0.snapshot_slots && c1.page_blocks >= c0.page_blocks, "{c0:?} against {c1:?}");
        assert_eq!(c0, c1, "at {slots} slots and {ctx} tokens the 8 GiB cap binds on both ranks: one budget");
    }
    // a Fixed request is carved first and is the same split on every rank whatever their memory
    let mut o = options(20, 40_000);
    o.prefix_cache = PrefixCacheOption::Mib(4096);
    let (f0, f1) = (
        plan_on(&tp_family(true), &o, avail0, shard, true).unwrap().plan.cache,
        plan_on(&tp_family(false), &o, avail1, shard, true).unwrap().plan.cache,
    );
    assert_eq!(f0, f1);
    assert!(f0.enabled() && f0.bytes() <= 4096 << 20);
    // a rank that is short of memory plans fewer entries than its peer: the gate's "rank 1 can hold only" case
    let poor = plan_on(&tp_family(false), &options(20, 262_144), 68 * GIB, shard, true).unwrap().plan.cache;
    let rich = plan_on(&tp_family(true), &options(20, 262_144), avail0, shard, true).unwrap().plan.cache;
    assert!(poor.snapshot_slots < rich.snapshot_slots, "{poor:?} against {rich:?}");
}

/// The plan at 1M tokens before and after the attention scratch was sized by what the two arithmetics need (`Scratch::new_serving`): the
/// tree's partials for the 16 chunks a row below 8,192 can have and the flash path's for one split a row per 8,192 keys, a window's 64
/// rows wide (a prefill slice of up to 256 rows runs in passes). BEFORE is the previous model's term, `rows x ceil(context / 512)
/// chunks x 32 heads x 520 B` at the step's rows, with its reserve rule; the table prints with `--nocapture`.
#[test]
fn the_plan_at_a_million_tokens_before_and_after_the_attention_scratch_follows_the_rule() {
    let mem = NemotronMemory::new(&cfg(), 0);
    println!("chunk slots head | free after weights | KV | state | partials before -> after | scratch before -> after | reserve before -> after | spare before -> after (GiB)");
    for chunk in [64usize, 256] {
        for head in [false, true] {
            for slots in [1usize, 2] {
                let fam = family(head, false);
                let mut o = options(slots, 1 << 20);
                o.prefill_chunk = chunk;
                let sp = plan_on(&fam, &o, 115 * GIB, weights(head), false).unwrap();
                let p = &sp.plan;
                let rows = chunk as u64;
                let old_partials = rows * 2048 * 32 * 520;
                let new_partials = mem.scratch_with(rows, 1 << 20, head).attn_partials;
                assert_eq!(new_partials, 153_354_760, "a window's rows whatever the slice");
                let records = sp.profile.window_record_bytes();
                let scratch_after = sp.profile.scratch_bytes();
                let scratch_before = scratch_after - new_partials + old_partials;
                let reserve_after = p.engine_reserve_bytes;
                let reserve_before = (scratch_before + records).max(5 * GIB);
                assert_eq!(reserve_after, (scratch_after + records).max(5 * GIB));
                let kv = p.kv_blocks as u64 * p.block_bytes;
                let state = slots as u64 * p.state_bytes_per_slot;
                let spare = |reserve: u64| sp.free_after_weights as i64 - (kv + state + reserve + 8 * GIB) as i64;
                let g = |b: i64| b as f64 / GIB as f64;
                println!(
                    "{chunk:>5} {slots:>5} {:>4} | {:>6.2} | {:>5.2} | {:>5.2} | {:>5.2} -> {:>5.2} | {:>5.2} -> {:>5.2} | {:>5.2} -> {:>5.2} | {:>5.2} -> {:>5.2}",
                    if head { "on" } else { "off" },
                    g(sp.free_after_weights as i64),
                    g(kv as i64),
                    g(state as i64),
                    g(old_partials as i64),
                    g(new_partials as i64),
                    g(scratch_before as i64),
                    g(scratch_after as i64),
                    g(reserve_before as i64),
                    g(reserve_after as i64),
                    g(spare(reserve_before)),
                    g(spare(reserve_after)),
                );
                // after: the blanket governs at every slice width; before, 256 rows at a million tokens raised the reserve
                assert_eq!(reserve_after, 5 * GIB);
                assert_eq!(reserve_before > 5 * GIB, chunk == 256, "{chunk} rows: before the reserve was {reserve_before}");
            }
        }
    }
}

#[test]
fn the_plan_at_a_million_tokens_for_one_and_two_slots_with_and_without_the_head_and_the_tower() {
    // docs/NEMOTRON_LONG_CONTEXT.md section 2.1 (text-only rows plan no cache here: the plain runtime)
    let per_image = 3328u64 * 4096 * 2;
    let tower_bytes = 1_727_774_720u64 + 709_118_980;
    for tower in [false, true] {
        for head in [false, true] {
            for slots in [1usize, 2] {
                let fam = family(head, tower);
                let mut o = options(slots, 1 << 20);
                if tower {
                    o.vision = Some(Arc::new(OnceLock::new()));
                }
                let avail = 115 * GIB - if tower { tower_bytes } else { 0 };
                let sp = plan_on(&fam, &o, avail, weights(head), false).unwrap();
                let p = &sp.plan;
                // one context of pool whatever the slots: 4,096 pages and a page a slot
                assert_eq!((p.kv_tokens, p.kv_blocks), (1 << 20, 4096 + slots), "{slots} slots, head {head}, tower {tower}");
                assert_eq!(p.block_bytes, if head { 2_359_296 } else { 2_097_152 });
                assert_eq!((p.kv_blocks as u64 * p.block_bytes) >> 30, if head { 9 } else { 8 });
                assert_eq!(p.state_bytes_per_slot, if head { 170_754_048 } else { 170_229_760 });
                assert_eq!(p.engine_reserve_bytes, 5 * GIB, "scratch and records are 2.4 GiB: the blanket governs");
                assert_eq!(p.vision_reserve_bytes, if tower { slots as u64 * 4 * per_image } else { 0 });
                // the attention partials: 64 rows x (16 tree chunks + 128 flash splits) x 32 heads x 520 B and the row plan (the tree's
                // alone, sized for the context, was 64 x 2,048 chunks = 2,181,038,080 B)
                let s = NemotronMemory::new(&cfg(), 0).scratch_with(64, 1 << 20, head);
                assert_eq!(s.attn_partials, 153_354_760);
                assert!(s.total() + sp.profile.window_record_bytes() < p.engine_reserve_bytes);
                // the free memory after the weights, as the table prints it
                let free_gib = sp.free_after_weights as f64 / GIB as f64;
                // With the head the plan prices the restricted draft vocabulary (the default, 49,152 ids x 8,196 B = 384 MiB);
                // `REXL3_MTP_VOCAB=off` prices the full head and gives the numbers before the default moved (51.40, 48.40).
                let restricted = !matches!(std::env::var("REXL3_MTP_VOCAB").as_deref(), Ok("off"));
                let head_cost = if head && restricted { 0.375 } else { 0.0 };
                let want = match (head, tower) {
                    (false, false) => 56.88,
                    (true, false) => 51.40,
                    (false, true) => 53.88,
                    (true, true) => 48.40,
                } - head_cost;
                assert!((free_gib - want).abs() < 0.01, "free {free_gib}");
                // and the chunk buckets of the context: 12, 1 to 2,048
                let buckets: std::collections::BTreeSet<usize> = (1..=2048).map(|r| chunk_bucket(r, 2048)).collect();
                assert_eq!(buckets.len(), 12);
                assert_eq!((buckets.first().copied(), buckets.last().copied()), (Some(1), Some(2048)));
            }
        }
    }
}

#[test]
fn the_bucket_flips_are_where_the_chunk_count_crosses_a_power_of_two() {
    // positions 2^(9+k) - 1 (the last of bucket 2^k) and 2^(9+k) (the first of bucket 2^(k+1)), for the caps of 1,048,576 and 1,000,000 tokens
    let real = |pos: usize| (pos + 1).div_ceil(512);
    for cap in [2048usize, 1_000_000usize.div_ceil(512)] {
        for k in 0..11u32 {
            let last = (1usize << (9 + k)) - 1;
            let first = 1usize << (9 + k);
            let (a, b) = (chunk_bucket(real(last), cap), chunk_bucket(real(first), cap));
            assert_eq!(a, (1usize << k).min(cap), "k {k}: the last position of bucket 2^{k}");
            assert_eq!(b, (1usize << (k + 1)).min(cap).max(real(first)), "k {k}: the first position past it");
            assert!(a >= real(last) && b >= real(first), "a bucket never serves a row it cannot cover");
        }
        // the cap is the top bucket and the last position of the context is inside it
        let last = (cap * 512).min(1 << 20) - 1;
        assert_eq!(chunk_bucket(real(last), cap), cap);
    }
    // 2,047 / 2,048: 4 chunks then 5, so bucket 4 then 8
    assert_eq!((real(2047), real(2048)), (4, 5));
    assert_eq!((chunk_bucket(4, 2048), chunk_bucket(5, 2048)), (4, 8));
    // 2^19 - 1 / 2^19: 1,024 chunks then 1,025, so bucket 1,024 then 2,048
    assert_eq!((real((1 << 19) - 1), real(1 << 19)), (1024, 1025));
    assert_eq!((chunk_bucket(1024, 2048), chunk_bucket(1025, 2048)), (1024, 2048));
}

fn load_for(o: &ServeOptions, sp: &ServePlan) -> SnapFake {
    // what the model's loader would do with the plan's options: a pool of the plan's pages and the arena the plan counted
    let _ = o;
    SnapFake::new(sp.model.slots, sp.plan.cache.snapshot_slots.max(1), sp.model.kv_pages, SnapFault::None)
}

#[test]
fn build_wires_the_arena_the_budget_and_the_engine_and_the_cached_engine_serves() {
    let fam = family(true, false);
    let mut o = options(2, 262_144);
    o.spec = SpecMode::Mtp;
    let h = host(115 * GIB);
    let inp = PlanInputs { cuda_free: 115 * GIB, host: Some(&h), weight_bytes: weights(true), model_max_context: 1 << 20 };
    let mut seen = Vec::new();
    let mut engine = build(&fam, &BuildCtx { options: &o }, &inp, |opts| {
        seen.push(opts.clone());
        let mut rt = SnapFake::new(opts.slots, 18, opts.kv_pages, SnapFault::None);
        rt.head = lossy_head();
        Ok(rt)
    })
    .unwrap();
    // the loader was given the pool with the cache's pages in it
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].kv_pages, 1026 + 2304);
    // the engine runs the cache: a prompt long enough for the shipped cut floor (128), then its extension
    let a = toks(77, 900);
    let ida = engine.submit(greedy(a.clone(), 12)).unwrap();
    engine.run().unwrap();
    let mut longer = a.clone();
    longer.extend(toks(78, 100));
    let idb = engine.submit(greedy(longer, 12)).unwrap();
    engine.run().unwrap();
    let outs = engine.take_outputs();
    let (oa, ob) = (outs.iter().find(|o| o.id == ida).unwrap(), outs.iter().find(|o| o.id == idb).unwrap());
    assert_eq!(oa.cached_tokens, 0);
    assert_eq!(ob.cached_tokens, 900 - 64, "the shipped interactive tail of 64 tokens");
    let st = engine.prefix_stats();
    assert!(st.enabled && st.hits == 1 && st.snapshot_slots == 18 && st.page_blocks == 2304, "{st:?}");
    assert!(engine.backend.m.violations.is_empty());
    // the namespace is the pack's identity and the head's presence, the budget the plan's
    let caps = engine.backend.prefix_caps().unwrap();
    assert_eq!(caps.namespace, prefix::namespace_of(&prefix::pack_identity(pack_dir()).unwrap(), true));
    assert_eq!((caps.budget.snapshot_slots, caps.budget.page_blocks), (18, 2304));
    // with the cache off the same boot is the plain one
    let mut off = o.clone();
    off.prefix_cache = PrefixCacheOption::Off;
    let e =
        build(&fam, &BuildCtx { options: &off }, &inp, |opts| Ok(SnapFake::new(opts.slots, 1, opts.kv_pages, SnapFault::None)))
            .unwrap();
    assert!(!e.prefix_stats().enabled);
}

#[test]
fn assemble_refuses_what_the_plan_and_the_runtime_disagree_about() {
    let fam = family(true, false);
    let o = options(2, 262_144);
    let sp = plan_on(&fam, &o, 115 * GIB, weights(true), true).unwrap();
    let flat = [(1, 70.0), (64, 190.0)];
    // an arena of another size than the plan counted
    let wrong = SnapFake::new(2, sp.plan.cache.snapshot_slots + 1, sp.model.kv_pages, SnapFault::None);
    let e = assemble_with_cost(wrong, &o, &sp, &flat).err().unwrap();
    assert!(e.to_string().contains("snapshot slots") || e.to_string().contains("carved"), "{e}");
    // an arena of the right slot count whose slots are another size than the plan counted (a runtime without the head's carry row: the plan
    // priced the head)
    let mut headless = SnapFake::new(2, sp.plan.cache.snapshot_slots, sp.model.kv_pages, SnapFault::None);
    headless.0.mtp = false;
    let e = assemble_with_cost(headless, &o, &sp, &flat).err().unwrap();
    assert!(e.to_string().contains("disagree") && e.to_string().contains("snapshot arena"), "{e}");
    // the device refusing the arena although the plan fits is the boot's named refusal (what was planned, free memory...), never a bare CUDA code
    let mut oom = load_for(&o, &sp);
    oom.fail = Some((Site::Arena, gb_model::error::ModelError::Cuda(gb_cuda::CudaError(2))));
    let e = assemble_with_cost(oom, &o, &sp, &flat).err().unwrap();
    let text = e.to_string();
    assert!(
        matches!(e, gb_model::error::ModelError::Config(_))
            && text.contains("the prefix cache's snapshot arena")
            && text.contains("GiB planned"),
        "{text}"
    );
    // a plan with a cache for a runtime that cannot snapshot
    let e = assemble_with_cost(FakeRuntime::new(2), &o, &sp, &flat).err().unwrap();
    assert!(e.to_string().contains("cannot snapshot"), "{e}");
    // the plain plan assembles on the plain runtime, as before
    let plain = plan_serving(
        &fam,
        &o,
        &PlanInputs {
            cuda_free: 115 * GIB,
            host: Some(&host(115 * GIB)),
            weight_bytes: weights(true),
            model_max_context: 1 << 20,
        },
    )
    .unwrap();
    assert!(assemble_with_cost(FakeRuntime::new(2), &o, &plain, &flat).is_ok());
    // and the right arena assembles
    assert!(assemble_with_cost(load_for(&o, &sp), &o, &sp, &flat).is_ok());
}

#[test]
fn the_family_says_it_can_cache_exactly_when_its_model_can_snapshot() {
    // `ModelFamily::prefix_cache` is the server's gate (it forces the cache off before `build` for a family that says no). The model type's own
    // answer is `snapshots_supported`; the two must move together, so the GPU worker who flips one is told about the other by this test.
    use gb_model::families::nemotron_h_omni::NemotronOmniFamily;
    assert_eq!(
        NemotronOmniFamily.prefix_cache(),
        <NemotronModel as NemotronRuntime>::snapshots_supported(),
        "flip `snapshots_supported` for NemotronModel and `NemotronOmniFamily::prefix_cache` together (docs/NEMOTRON_LONG_CONTEXT.md, 3.4)"
    );
    // both: the GPU hooks are built and gated (`nemotron_omni_prefix_gpu.rs`)
    assert!(NemotronOmniFamily.prefix_cache());
}

/// `--kv-headroom-tokens` (docs/LONG_PREFILL_POLICY.md): the pool is one context PLUS the headroom, exactly, with the `Auto` cache unchanged; one
/// request is still limited to the context, so the headroom only seats other requests beside a maximal one; a headroom the box cannot hold, or one
/// that would cost the `Auto` cache its one maximal entry, refuses the boot by name. Without the flag the plan is the one-context plan.
#[test]
fn a_kv_headroom_is_planned_exactly_on_top_of_one_context_or_refused_by_name() {
    let fam = family(true, false);
    let ctx = 1usize << 20;
    let base = plan_on(&fam, &options(2, ctx), 115 * GIB, weights(true), true).unwrap();
    assert_eq!(base.plan.kv_tokens, ctx as u64, "no headroom: the one-context pool");
    let mut o = options(2, ctx);
    o.kv_headroom_tokens = 131_072;
    let sp = plan_on(&fam, &o, 115 * GIB, weights(true), true).unwrap();
    assert_eq!(sp.max_context, ctx);
    assert_eq!(sp.plan.kv_tokens, (ctx + 131_072) as u64);
    // 512 more pages of 2.25 MiB (9,216 B a token with the head): 1.125 GiB, and the cache keeps its 8 units of 512 pages
    assert_eq!(sp.plan.kv_blocks, base.plan.kv_blocks + 512);
    assert_eq!(
        sp.plan.kv_blocks as u64 * sp.plan.block_bytes - base.plan.kv_blocks as u64 * base.plan.block_bytes,
        131_072 * 9_216
    );
    assert_eq!(sp.plan.cache, base.plan.cache);
    assert_eq!((sp.plan.cache.snapshot_slots, sp.plan.cache.page_blocks), (8, 4096));
    assert_eq!(sp.model.kv_pages, sp.plan.kv_blocks + 4096);
    // admission from the plan: a maximal request and a 131,072-token one side by side; one more token does not fit beside it; nothing above the
    // context is ever admitted
    let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 9_216, state_bytes_per_slot: 1, activation_reserve: 0 };
    let mut a = sp.plan.admission(mm.clone(), 2, sp.max_context).unwrap();
    assert_eq!(a.request_limit(), Some(ctx as u64));
    assert!(matches!(a.check(ctx as u64 + 1), Err(gb_sched::AdmitError::ContextTooLong { .. })));
    a.admit(ctx as u64).unwrap();
    assert!(matches!(a.can_admit(131_073), Err(gb_sched::AdmitError::NoMemory { .. })));
    a.admit(131_072).unwrap();
    // the one-context plan: the second request waits for the whole first one
    let mut b = base.plan.admission(mm, 2, base.max_context).unwrap();
    b.admit(ctx as u64).unwrap();
    assert!(matches!(b.can_admit(1), Err(gb_sched::AdmitError::NoMemory { .. })));
    // more than the box holds: refused, never planned smaller
    o.kv_headroom_tokens = 64 << 20;
    let e = plan_on(&fam, &o, 115 * GIB, weights(true), true).unwrap_err();
    assert!(e.contains("--kv-headroom-tokens 67108864") && e.contains("the box holds"), "{e}");
    // a headroom that fits but halves the spare below one maximal cache entry: refused, naming the cache
    o.kv_headroom_tokens = 1_200_000;
    let e = plan_on(&fam, &o, 115 * GIB, weights(true), true).unwrap_err();
    assert!(e.contains("prefix cache") && e.contains("4096 pages"), "{e}");
    // ... unless the operator sized the cache explicitly
    o.prefix_cache = PrefixCacheOption::Mib(11 * 1024);
    let sp = plan_on(&fam, &o, 115 * GIB, weights(true), true).unwrap();
    assert_eq!(sp.plan.kv_tokens, (ctx + 1_200_128) as u64, "whole pages");
}

/// Progressive cuts on the Nemotron backend (docs/LONG_PREFILL_POLICY.md): a 1,600-token prompt prefilled in 64-row slices with a checkpoint
/// every 256 tokens and cancelled at 1,152 keeps its checkpoint at 1,024 (an aligned cut: the page holding position 1,023 is the one a
/// restore copies, the head's entry there is rewritten by the resumer), and the retry and a second question that shares only 1,100 tokens
/// both resume from it with the uncached replies and the head's counters, the books balanced. `Err` names what differed.
fn checkpoint_after_cancel(fault: SnapFault) -> Result<(), String> {
    let backend = snap_backend(SLOTS, 6, fault);
    let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 9216, state_bytes_per_slot: 170_754_048, activation_reserve: 0 };
    let cost = CostModel::from_table(vec![(1, 1.0), (64, 1.0)]);
    let make: Box<dyn Fn() -> Box<dyn Drafter>> = Box::new(|| Box::new(NoDrafts));
    let mut e = Engine::new(
        backend,
        Admission::new(96 << 30, 0, mm, SLOTS).unwrap(),
        EngineConfig { max_rows: 64, max_drafts: 7, cost, drafting: true },
        make,
    )
    .with_prefix_cache(PrefixConfig { checkpoint_every: 256, ..policy() });
    let doc = toks(77, 1600);
    let r = greedy(doc.clone(), 12);
    let id = e.submit_deferred(r.clone()).unwrap();
    for _ in 0..18 {
        e.step().unwrap();
    }
    assert_eq!(e.prefilling(), 1);
    assert!(e.cancel(id));
    e.run().unwrap();
    let _ = e.take_outputs();
    let st = e.prefix_stats();
    assert_eq!((st.checkpoints, st.checkpoints_superseded, st.kept_on_cancel, st.entries), (4, 3, 1, 1), "{st:?}");
    no_leaks(&e)?;
    let out = one(&mut e, r.clone());
    assert_eq!(out.cached_tokens, 1024);
    same(&out, &reference(&r)).map_err(|m| format!("the retry: {m}"))?;
    let q2 = request([&doc[..1100], &toks(78, 200)[..]].concat(), 20, 3, 0.7);
    let out = one(&mut e, q2.clone());
    assert_eq!(out.cached_tokens, 1024);
    same(&out, &reference(&q2)).map_err(|m| format!("the second question: {m}"))?;
    no_leaks(&e)
}

#[test]
fn a_cancelled_prefill_keeps_its_last_checkpoint_and_resumes_exactly_with_the_head() {
    checkpoint_after_cancel(SnapFault::None).unwrap();
    // the restore faults a resumed checkpoint would expose: each one is caught by this scenario (the shared-last-page mutants need a donor that
    // goes on writing into the page, which a cancelled donor never does: `mutant_a_full_last_page_is_shared_when_the_cut_is_aligned_is_caught`)
    for f in
        [SnapFault::StateNotRestored, SnapFault::ConvNotRestored, SnapFault::CarryNotRestored, SnapFault::HeadStripeNotCopied]
    {
        assert!(checkpoint_after_cancel(f).is_err(), "{f:?} was not caught");
    }
}
