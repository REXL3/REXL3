//! The Nemotron backend against a fake runtime (no GPU, no pack): the bookkeeping the engine relies on, and the exactness harness
//! (`gb_engine::harness`) run through it. The fake's logic makes row invariance and rollback hold by construction (see
//! `nemotron_omni_backend_support`), so what these tests exercise is the backend's slicing, slot table, window table and error
//! latch. The same harness calls run on the real model in task I3 through a `Shared` wrapper, which is also tested here.

mod nemotron_omni_backend_support;

use gb_engine::harness::{
    chunked_prefill_equals_whole, drafted_equals_serial, rollback_equals_replay, row_invariance, serial_batched_equals_solo,
};
use gb_engine::{
    Backend, Drafter, Engine, EngineConfig, ImageSpan, Proposal, ProposeRequest, Request, RowOutput, RowRequest, StepError,
};
use gb_model::error::ModelError;
use gb_model::families::nemotron_h_omni::backend::{
    next_chunk_end, prefill_rows_max, window_is_text, NemotronBackend, NemotronRuntime, MAX_CANDIDATES, WINDOW_ROWS,
};
use gb_model::families::nemotron_h_omni::spec::{CostTables, SpecConfig, MTP_TAP_ROWS};
use gb_sampling::Sampling;
use gb_sched::{Admission, CostModel, ModelMemory};
use gb_spec::{CopyDrafter, SpecMode};
use nemotron_omni_backend_support::{Call, FakeRuntime, HeadScript, Site, SLOTS, TAP_ROWS, VOCAB};
use std::cell::RefCell;
use std::rc::Rc;

type Backend64 = NemotronBackend<FakeRuntime>;

fn backend_with(rt: FakeRuntime, chunk: usize) -> Backend64 {
    let opts = rt.opts.clone();
    NemotronBackend::with_runtime(rt, chunk, &opts).unwrap()
}

/// The serve default chunk (4096): the model's 64 rows govern.
fn backend() -> Backend64 {
    backend_with(FakeRuntime::new(SLOTS), 4096)
}

fn tokens(n: usize, seed: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 37 + seed * 11 + 3) % (VOCAB - 8)) as u32 + 4).collect()
}

fn prefills(b: &Backend64) -> Vec<(usize, usize, Option<usize>)> {
    b.m.calls
        .iter()
        .filter_map(|c| match c {
            Call::Prefill { first, rows, want, .. } => Some((*first, *rows, *want)),
            _ => None,
        })
        .collect()
}

fn window_req(slot: usize, first_position: u64, n: usize) -> RowRequest {
    RowRequest { slot, first_position, tokens: (0..n).map(|i| (i % 50) as u32 + 4).collect() }
}

fn assert_clean(b: &Backend64) {
    assert!(b.m.violations.is_empty(), "the backend broke the runtime's contract: {:?}", b.m.violations);
}

// ---- slicing

#[test]
fn next_chunk_end_is_the_next_multiple_of_the_chunk() {
    assert_eq!(next_chunk_end(0, 130, 64), 64);
    assert_eq!(next_chunk_end(64, 130, 64), 128);
    assert_eq!(next_chunk_end(67, 130, 64), 128, "a resume off the grid ends on the grid");
    assert_eq!(next_chunk_end(128, 130, 64), 130);
    assert_eq!(next_chunk_end(128, 129, 64), 129);
    assert_eq!(next_chunk_end(0, 5, 64), 5);
    assert_eq!(next_chunk_end(usize::MAX - 3, usize::MAX, 64), usize::MAX, "no overflow");
}

#[test]
fn prefill_slices_clamp_to_the_model_rows_and_only_the_last_asks_for_candidates() {
    let mut b = backend();
    assert_eq!(b.prefill_rows(), 64, "chunk 4096 against a model of 64 rows");
    let prompt = tokens(130, 1);
    let mut done = 0;
    let mut progress = Vec::new();
    while done < prompt.len() {
        let s = b.prefill_slice(1, &prompt, &[], done, 5);
        progress.push((done, s.done, s.row.as_ref().map(|r| r.ids.len())));
        done = s.done;
    }
    assert_eq!(progress, vec![(0, 64, None), (64, 128, None), (128, 130, Some(5))]);
    // the runtime saw the same grid; `want` is the row within the slice, and only the final slice has one
    assert_eq!(prefills(&b), vec![(0, 64, None), (64, 64, None), (128, 2, Some(1))]);
    assert_eq!(b.m.committed(1), Some(&prompt[..]), "every token reached the slot exactly once, in order");
    assert_clean(&b);
    assert!(b.take_error().is_none());
}

#[test]
fn the_chunk_is_the_smallest_of_the_option_the_model_and_the_family_cap() {
    assert_eq!(backend_with(FakeRuntime::new(SLOTS), 4096).prefill_rows(), 64);
    assert_eq!(backend_with(FakeRuntime::new(SLOTS), 48).prefill_rows(), 48);
    assert_eq!(backend_with(FakeRuntime::new(SLOTS).with_prefill_rows(16), 4096).prefill_rows(), 16);
    // a model built for more rows (after the large-row MoE prefill lands) is still held to the family cap until the cap moves
    assert_eq!(backend_with(FakeRuntime::new(SLOTS).with_prefill_rows(4096), 4096).prefill_rows(), prefill_rows_max());
    let rt = FakeRuntime::new(SLOTS);
    let opts = rt.opts.clone();
    assert!(NemotronBackend::with_runtime(rt, 0, &opts).is_err());
    let mut bad = opts.clone();
    bad.slots = 0;
    assert!(NemotronBackend::with_runtime(FakeRuntime::new(SLOTS), 64, &bad).is_err());
    bad = opts;
    bad.prefill_rows = 0;
    assert!(NemotronBackend::with_runtime(FakeRuntime::new(SLOTS), 64, &bad).is_err());
}

#[test]
fn pieces_of_a_prompt_land_on_one_absolute_grid_wherever_they_start() {
    let prompt = tokens(130, 2);
    let mut b = backend_with(FakeRuntime::new(SLOTS), 32);
    // 67 tokens, then the rest: the second piece starts off the grid and its first slice ends on it
    let first = b.prefill_piece(0, 0, &prompt[..67], 130, 6);
    let second = b.prefill_piece(0, 67, &prompt[67..], 130, 6);
    assert_eq!(
        prefills(&b),
        vec![(0, 32, None), (32, 32, None), (64, 3, Some(2)), (67, 29, None), (96, 32, None), (128, 2, Some(1)),]
    );
    assert_eq!((first.ids.len(), second.ids.len()), (6, 6));
    assert_eq!(b.m.committed(0), Some(&prompt[..]));
    assert_clean(&b);
}

#[test]
fn a_piece_equals_the_whole_prompt_whatever_its_cuts() {
    let prompt = tokens(200, 3);
    let whole = {
        let mut b = backend();
        let row = b.prefill_piece(0, 0, &prompt, 200, 12);
        assert_eq!(b.m.committed(0), Some(&prompt[..]));
        row
    };
    for cuts in [vec![1], vec![63], vec![64], vec![65], vec![17, 90], vec![64, 128, 192], vec![199], vec![1, 2, 3, 130]] {
        let mut b = backend();
        let mut at = 0;
        let mut last = None;
        for &c in cuts.iter().chain(std::iter::once(&prompt.len())) {
            last = Some(b.prefill_piece(0, at as u64, &prompt[at..c], prompt.len(), 12));
            at = c;
        }
        assert!(last.unwrap().bits_eq(&whole), "cuts {cuts:?}: the last row's candidates changed");
        assert_eq!(b.m.committed(0), Some(&prompt[..]), "cuts {cuts:?}");
        assert_clean(&b);
    }
}

#[test]
fn a_deadline_limit_puts_slices_on_a_finer_grid_and_lifting_it_restores_the_grid() {
    let mut b = backend();
    let prompt = tokens(130, 4);
    // the engine asks for 128 rows: slices are at most 64 already, so nothing changes
    assert!(b.set_prefill_row_limit(Some(128)));
    assert_eq!(b.prefill_slice(0, &prompt, &[], 0, 3).done, 64);
    assert!(b.set_prefill_row_limit(Some(16)));
    assert_eq!(b.prefill_slice(0, &prompt, &[], 64, 3).done, 80);
    assert!(b.set_prefill_row_limit(None));
    assert_eq!(b.prefill_slice(0, &prompt, &[], 80, 3).done, 128);
    assert_eq!(b.prefill_slice(0, &prompt, &[], 128, 3).done, 130);
    assert_eq!(b.m.committed(0), Some(&prompt[..]));
    assert_clean(&b);
}

#[test]
fn a_prefill_that_overflows_or_is_empty_is_a_request_error_and_never_reaches_the_runtime() {
    let mut b = backend();
    assert!(b.prefill(1, u64::MAX, &[3], 1).values[0].is_nan());
    let e = b.take_error().unwrap();
    assert!(!e.fatal);
    assert!(b.prefill_piece(1, 0, &[3, 4], 1, 1).values[0].is_nan(), "a piece past the prompt total");
    assert!(b.take_error().is_some());
    assert!(b.prefill_piece(1, 0, &[], 0, 1).values[0].is_nan(), "an empty piece");
    assert!(b.take_error().is_some());
    let s = b.prefill_slice(1, &[3, 4], &[], 2, 1);
    assert!(s.row.unwrap().values[0].is_nan(), "a completed slice");
    assert!(b.take_error().is_some());
    assert!(
        b.m.calls.iter().all(|c| !matches!(c, Call::Prefill { .. } | Call::Begin(_))),
        "no work was started: {:?}",
        b.m.calls
    );
}

#[test]
fn a_prefill_that_resumes_a_slot_nobody_began_is_refused() {
    let mut b = backend();
    let prompt = tokens(100, 5);
    let s = b.prefill_slice(2, &prompt, &[], 64, 4);
    assert!(s.row.unwrap().values[0].is_nan());
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("no prefill in progress"), "{e:?}");
    assert!(prefills(&b).is_empty());
}

// ---- windows and commit

#[test]
fn commit_keeps_one_to_sixty_four_rows_of_a_window_that_exists_and_nothing_else() {
    let mut b = backend();
    let prompt = tokens(10, 6);
    b.prefill(0, 0, &prompt, 4);
    // a 64-row window may keep all 64
    let out = b.step(&[window_req(0, 10, 64)], 4);
    assert_eq!(out[0].len(), 64);
    assert!(out[0].iter().all(|r| r.values.iter().all(|v| v.is_finite())));
    assert_eq!(b.pending_rows(0), Some(64));
    b.commit(0, 64);
    assert!(b.take_error().is_none());
    assert_eq!(b.m.committed(0).unwrap().len(), 74);
    assert_eq!(b.pending_rows(0), None, "a commit consumes its window");

    // an 8-row window: 0, more than its rows, and more than 64 are refused and change nothing
    b.step(&[window_req(0, 74, 8)], 4);
    let commits_before = b.m.count(|c| matches!(c, Call::Commit { .. }));
    for bad in [0, 9, 64, 65, 1000] {
        b.commit(0, bad);
        let e = b.take_error().unwrap_or_else(|| panic!("keep {bad} was accepted"));
        assert!(!e.fatal, "{e:?}");
        assert_eq!(b.pending_rows(0), Some(8), "a refused commit leaves the window as it was");
    }
    assert_eq!(b.m.count(|c| matches!(c, Call::Commit { .. })), commits_before, "no refused commit reached the runtime");
    b.commit(0, 8);
    assert!(b.take_error().is_none());
    assert_eq!(b.m.committed(0).unwrap().len(), 82);
    // the window is gone: a second commit would advance the state twice
    b.commit(0, 1);
    let e = b.take_error().unwrap();
    assert!(e.message.contains("no window to commit"), "{e:?}");
    assert_eq!(b.m.committed(0).unwrap().len(), 82);
    assert_clean(&b);
}

#[test]
fn a_window_never_exceeds_sixty_four_rows_and_the_refusal_names_the_limit() {
    let mut b = backend();
    for slot in 0..3 {
        b.prefill(slot, 0, &tokens(8, slot), 4);
    }
    let windows_before = b.m.count(|c| matches!(c, Call::Window { .. }));
    // 40 + 25 rows of two sequences: 65
    let reqs = [window_req(0, 8, 40), window_req(1, 8, 25)];
    let out = b.step(&reqs, 4);
    assert_eq!((out[0].len(), out[1].len()), (40, 25));
    assert!(out.iter().flatten().all(|r| r.values.iter().all(|v| v.is_nan())), "NaN rows, as every refused step answers");
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("1..=64") && e.message.contains("65"), "{e:?}");
    assert_eq!(b.m.count(|c| matches!(c, Call::Window { .. })), windows_before, "the runtime never saw the wide window");
    assert_eq!((b.pending_rows(0), b.pending_rows(1)), (None, None));
    // exactly 64 across several sequences is fine
    let ok = [window_req(0, 8, 32), window_req(1, 8, 16), window_req(2, 8, 16)];
    let out = b.step(&ok, 4);
    assert!(b.take_error().is_none());
    assert_eq!(out.iter().map(Vec::len).sum::<usize>(), 64);
    assert_eq!(WINDOW_ROWS, 64);
    assert_clean(&b);
}

#[test]
fn a_window_on_a_slot_that_is_not_live_or_that_repeats_is_refused_before_the_runtime() {
    let mut b = backend();
    b.prefill(0, 0, &tokens(8, 7), 4);
    let before = b.m.calls.len();
    let out = b.step(&[window_req(3, 0, 2)], 4);
    assert!(out[0][0].values[0].is_nan());
    assert!(b.take_error().unwrap().message.contains("no prefilled sequence"));
    let out = b.step(&[window_req(0, 8, 2), window_req(0, 8, 2)], 4);
    assert!(out[0][0].values[0].is_nan());
    assert!(b.take_error().unwrap().message.contains("twice"));
    let out = b.step(&[window_req(SLOTS + 1, 8, 1)], 4);
    assert!(out[0][0].values[0].is_nan());
    assert!(b.take_error().unwrap().message.contains("out of range"));
    assert_eq!(b.step(&[], 4).len(), 0, "an empty step is an empty answer, not an error");
    assert!(b.take_error().is_none());
    assert_eq!(b.m.calls.len(), before, "none of them touched the runtime");
}

#[test]
fn the_candidate_count_is_bounded_before_the_runtime_sees_it() {
    let mut b = backend();
    b.prefill(0, 0, &tokens(8, 8), 4);
    for bad in [0, MAX_CANDIDATES + 1] {
        let out = b.step(&[window_req(0, 8, 1)], bad);
        assert_eq!(out[0].len(), 1);
        assert!(b.take_error().unwrap().message.contains("1..=64"));
        let row = b.prefill(1, 0, &tokens(3, 1), bad);
        assert_eq!(row.ids.len(), bad);
        assert!(b.take_error().is_some());
    }
    assert!(b.m.violations.is_empty());
}

// ---- slots

#[test]
fn release_is_idempotent_and_ends_the_slot_exactly_once() {
    let mut b = backend();
    b.prefill(2, 0, &tokens(20, 9), 4);
    assert_eq!(b.live_slots(), vec![2]);
    b.step(&[window_req(2, 20, 3)], 4);
    b.release(2);
    b.release(2);
    b.release(2);
    assert_eq!(b.m.ends(2), 1);
    assert_eq!((b.live_slots(), b.pending_rows(2)), (vec![], None));
    assert_eq!(b.m.begun(), Vec::<usize>::new());
    // a slot nobody began is not the runtime's business
    b.release(5);
    assert_eq!(b.m.ends(5), 0);
    // and a released slot refuses a window and a commit
    b.step(&[window_req(2, 20, 1)], 4);
    assert!(b.take_error().is_some());
    b.commit(2, 1);
    assert!(b.take_error().is_some());
    // the slot is reusable
    b.prefill(2, 0, &tokens(5, 10), 4);
    assert_eq!(b.live_slots(), vec![2]);
    b.release(2);
    assert_eq!(b.m.ends(2), 2, "one end per sequence");
    assert_clean(&b);
}

#[test]
fn a_failed_begin_slot_leaves_no_slot_registered_and_ends_it_once() {
    let mut b = backend();
    b.m.fail = Some((Site::Begin, ModelError::Request("pool is full".into())));
    let row = b.prefill(1, 0, &tokens(30, 11), 4);
    assert!(row.values.iter().all(|v| v.is_nan()));
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("pool is full") && e.message.contains("slot 1"), "{e:?}");
    assert_eq!(b.live_slots(), Vec::<usize>::new(), "no half-registered slot");
    assert_eq!(b.m.ends(1), 1, "the runtime's half-begun slot is cleaned up once");
    assert!(prefills(&b).is_empty(), "nothing ran on the unusable slot");
    // the engine releases a failed sequence's slot: no second end
    b.release(1);
    assert_eq!(b.m.ends(1), 1);
    // and nothing can run on it until a prefill from 0 begins it again
    b.step(&[window_req(1, 0, 1)], 4);
    assert!(b.take_error().unwrap().message.contains("no prefilled sequence"));
    let row = b.prefill(1, 0, &tokens(30, 11), 4);
    assert!(row.values.iter().all(|v| v.is_finite()));
    assert_eq!(b.live_slots(), vec![1]);
    assert_clean(&b);
}

#[test]
fn a_fatal_begin_leaves_the_dead_runtime_alone() {
    let mut b = backend();
    b.m.fail = Some((Site::Begin, ModelError::StateLost("pool lost".into())));
    assert!(b.prefill(1, 0, &tokens(30, 11), 4).values[0].is_nan());
    assert!(b.take_error().unwrap().fatal);
    assert_eq!(b.m.ends(1), 0, "nothing is asked of a runtime that said its state is lost");
    assert_eq!(b.live_slots(), Vec::<usize>::new());
}

#[test]
fn a_slice_that_fails_midway_gives_the_slot_up_once() {
    let mut b = backend();
    let prompt = tokens(200, 12);
    assert_eq!(b.prefill_slice(0, &prompt, &[], 0, 4).done, 64);
    b.m.fail = Some((Site::Prefill, ModelError::Request("kernel refused".into())));
    let s = b.prefill_slice(0, &prompt, &[], 64, 4);
    assert_eq!(s.done, prompt.len(), "the engine is told the prompt is over");
    assert!(s.row.unwrap().values[0].is_nan());
    assert!(b.take_error().unwrap().message.contains("kernel refused"));
    assert_eq!(b.live_slots(), Vec::<usize>::new());
    b.release(0);
    assert_eq!(b.m.ends(0), 1);
    // the next slice of the dead prompt is refused, not run on a slot with half its state
    let s = b.prefill_slice(0, &prompt, &[], 128, 4);
    assert!(s.row.unwrap().values[0].is_nan());
    assert!(b.take_error().is_some());
    assert_eq!(b.m.ends(0), 1, "a slot already given up is not ended again by the refusal that followed");
    assert_clean(&b);
}

#[test]
fn a_reused_slot_starts_over_at_position_zero_without_a_release() {
    let mut b = backend();
    b.prefill(0, 0, &tokens(10, 13), 4);
    b.step(&[window_req(0, 10, 5)], 4);
    assert_eq!(b.pending_rows(0), Some(5));
    // the harness re-prefills slot 0 on one shared backend: the new sequence replaces the old, window and all
    b.prefill(0, 0, &tokens(7, 14), 4);
    assert_eq!(b.pending_rows(0), None);
    assert_eq!(b.m.committed(0), Some(&tokens(7, 14)[..]));
    assert_eq!(b.m.ends(0), 1, "the live slot is ended once before it begins again (a runtime refuses to begin a begun slot)");
    b.commit(0, 1);
    assert!(b.take_error().is_some(), "the old window is not committable into the new sequence");
    assert_clean(&b);
}

// ---- errors

#[test]
fn runtime_errors_are_latched_once_and_fatal_ones_stay_fatal() {
    let mut b = backend();
    b.prefill(0, 0, &tokens(8, 15), 4);
    let req = [window_req(0, 8, 2)];
    b.m.fail = Some((Site::Window, ModelError::Request("bad input".into())));
    let out = b.step(&req, 4);
    assert!(out[0].iter().all(|r| r.values.iter().all(|v| v.is_nan())));
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("window step") && e.message.contains("bad input"), "{e:?}");
    assert!(b.take_error().is_none(), "a retryable error is reported once");
    assert_eq!(b.pending_rows(0), None);
    // the backend is still usable
    assert!(b.step(&req, 4)[0][0].values[0].is_finite());
    assert!(b.take_error().is_none());

    // StateLost is fatal whatever call it came from; so is a sticky CUDA error
    let fatal = ModelError::Cuda(gb_cuda::CudaError(700));
    assert!(fatal.is_fatal(), "the test needs a code the CUDA layer calls fatal");
    for (site, err) in [(Site::Window, ModelError::StateLost("lost state".into())), (Site::Window, fatal)] {
        let mut b = backend();
        b.prefill(0, 0, &tokens(8, 16), 4);
        b.m.fail = Some((site, err));
        b.step(&req, 4);
        let e = b.take_error().unwrap();
        assert!(e.fatal, "{e:?}");
        // it stays: every later call answers NaN, reports the same error, and leaves the runtime alone
        let calls = b.m.calls.len();
        assert!(b.step(&req, 4)[0][0].values[0].is_nan());
        assert!(b.prefill(1, 0, &tokens(4, 1), 4).values[0].is_nan());
        assert!(b.prefill_slice(1, &tokens(4, 1), &[], 0, 4).row.unwrap().values[0].is_nan());
        b.commit(0, 1);
        b.release(0);
        assert!(b.take_error().unwrap().fatal);
        assert!(b.take_error().unwrap().fatal, "a fatal error is not consumed by being read");
        assert_eq!(b.m.calls.len(), calls, "a dead backend does not call the runtime");
    }
}

#[test]
fn a_flush_error_surfaces_on_the_call_that_flushed_and_quiesce_reports_it() {
    let mut b = backend();
    b.m.fail = Some((Site::Flush, ModelError::StateLost("stream error".into())));
    let row = b.prefill(0, 0, &tokens(4, 17), 4);
    assert!(row.values[0].is_nan());
    assert!(b.take_error().unwrap().fatal);

    let mut b = backend();
    assert!(b.quiesce().is_ok());
    assert_eq!(b.m.calls, vec![Call::Flush], "quiesce flushes");
    b.m.fail = Some((Site::Flush, ModelError::Request("device said no".into())));
    assert!(b.quiesce().unwrap_err().contains("device said no"));
    assert!(b.quiesce().is_ok());
}

#[test]
fn a_runtime_that_answers_short_is_a_named_error_not_a_panic() {
    // fewer candidates than the sampler needs
    let mut rt = FakeRuntime::new(SLOTS);
    rt.prefill_count = 4;
    let mut b = backend_with(rt, 64);
    let row = b.prefill(0, 0, &tokens(10, 18), 12);
    assert!(row.values.iter().all(|v| v.is_nan()));
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("4 candidates per row") && e.message.contains("needs 12"), "{e:?}");
    assert_eq!(b.m.ends(0), 1);
    // enough of them: the first 12 of 64
    let mut b = backend();
    let row = b.prefill(0, 0, &tokens(10, 18), 12);
    assert_eq!(row.ids.len(), 12);
    assert!(row.values.windows(2).all(|w| w[0] >= w[1]), "still sorted: the first n of a sorted list");
}

#[test]
fn images_are_refused_by_name_and_the_slot_is_given_up() {
    let mut b = backend();
    let features: std::sync::Arc<dyn std::any::Any + Send + Sync> = std::sync::Arc::new(());
    let span = ImageSpan { start: 1, tokens: 2, merged: (1, 2), features, key: 7 };
    let prompt = tokens(10, 19);
    let s = b.prefill_slice(0, &prompt, std::slice::from_ref(&span), 0, 4);
    assert!(s.row.unwrap().values[0].is_nan());
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("nemotron_radio_vision"), "{e:?}");
    let row = b.prefill_mm(1, 0, &prompt, std::slice::from_ref(&span), 4);
    assert!(row.values[0].is_nan());
    assert!(b.take_error().unwrap().message.contains("nemotron_radio_vision"));
    // nobody began those slots, so nobody ends them
    assert_eq!((b.m.ends(0), b.m.ends(1)), (0, 0));
    // a live slot that is then refused an image is given up once
    b.prefill(2, 0, &prompt, 4);
    b.prefill_slice(2, &prompt, std::slice::from_ref(&span), 4, 4);
    assert!(b.take_error().is_some());
    assert_eq!(b.m.ends(2), 1);
    b.release(2);
    assert_eq!(b.m.ends(2), 1);
    // no images: `prefill_mm` is a plain prefill
    assert!(b.prefill_mm(1, 0, &prompt, &[], 4).values[0].is_finite());
    assert_clean(&b);
}

// ---- the harness

fn sampling(seed: u64) -> Sampling {
    Sampling { seed, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 }
}

fn make() -> Backend64 {
    backend()
}

fn make_engine(drafting: bool) -> Engine<Backend64> {
    let mm = ModelMemory { weights_bytes: 0, kv_bytes_per_token: 8192, state_bytes_per_slot: 170_229_760, activation_reserve: 0 };
    let admission = Admission::new(64 << 30, 0, mm, SLOTS).unwrap();
    // a synthetic curve with a knee at 16 rows, as the engine's own exactness tests use: drafting is cheap enough to be asked for
    let cost = CostModel::from_table(vec![(1, 70.0), (16, 74.0), (32, 110.0), (64, 190.0)]);
    let make: Box<dyn Fn() -> Box<dyn Drafter>> = Box::new(|| Box::new(CopyDrafter::new(8, 7)));
    Engine::new(backend(), admission, EngineConfig { max_rows: WINDOW_ROWS, max_drafts: 7, cost, drafting }, make)
}

/// Prompts of lengths on both sides of the slice grid, repetitive enough for the copy drafter to hit.
fn requests() -> Vec<Request> {
    [5usize, 40, 63, 64, 65, 128, 129, 200]
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let unit: Vec<u32> = (0..12).map(|j| ((i * 13 + j * 5) % (VOCAB - 8)) as u32 + 4).collect();
            let mut prompt: Vec<u32> = unit.iter().cycle().take(n).copied().collect();
            if let Some(first) = prompt.first_mut() {
                *first = 3 + i as u32;
            }
            Request {
                sampling: Sampling { seed: 900 + i as u64, temperature: [0.0, 0.7][i % 2], top_k: 20, top_p: 0.95, min_p: 0.0 },
                prompt,
                max_new: 48 + (i % 3) * 9,
                stop: vec![],
                draft: true,
                images: vec![],
                cache: true,
            }
        })
        .collect()
}

#[test]
fn rows_are_invariant_through_the_backend() {
    let cases: Vec<(Vec<u32>, Vec<u32>)> =
        (0..6).map(|i| (tokens(4 + i * 11, i), (0..(1 + i * 3)).map(|j| (j * 3 + 1) as u32).collect())).collect();
    // 1 + 4 + 7 + 10 + 13 + 16 = 51 rows: the harness puts every case in one window, which the backend holds to 64
    row_invariance(&make, &cases, 28).unwrap();
}

#[test]
fn batched_serial_equals_solo_through_the_engine() {
    let r = serial_batched_equals_solo(&make_engine, &requests()).unwrap();
    assert_eq!(r.sequences, 8);
}

#[test]
fn drafted_equals_serial_with_the_copy_drafter() {
    let reqs: Vec<Request> = requests()
        .into_iter()
        .map(|mut r| {
            r.sampling.temperature = 0.0;
            r.max_new = 160;
            r
        })
        .collect();
    let r = drafted_equals_serial(&make_engine, &reqs).unwrap();
    assert!(r.accepted > 0, "no copy draft was accepted, so verify windows were never committed past one row");
    assert!(r.drafted > r.accepted, "no draft was rejected, so a partial commit never ran");
    eprintln!("copy drafter: {} drafted, {} accepted, {:.2} tokens/round", r.drafted, r.accepted, r.tokens_per_round());
}

#[test]
fn chunked_prefill_equals_whole_through_the_backend() {
    let prompt = tokens(200, 20);
    let s = sampling(5);
    for cuts in [vec![1], vec![63, 64], vec![64], vec![65], vec![100], vec![16, 64, 130], vec![199]] {
        chunked_prefill_equals_whole(&make, &prompt, &cuts, &s, 24).unwrap_or_else(|e| panic!("{cuts:?}: {e}"));
    }
}

#[test]
fn rollback_equals_replay_for_every_keep_through_the_backend() {
    let ctx = tokens(70, 21);
    for rows in [2usize, 8, 33, 64] {
        let window: Vec<u32> = (0..rows).map(|j| ((j * 5 + 9) % 200) as u32 + 4).collect();
        let keeps: Vec<usize> = if rows <= 8 { (1..=rows).collect() } else { vec![1, 2, rows / 2, rows - 1, rows] };
        for keep in keeps {
            let next = vec![17u32, 18, 19];
            rollback_equals_replay(&make, &ctx, &window, keep, &next, 28)
                .unwrap_or_else(|e| panic!("window {rows} keep {keep}: {e}"));
        }
    }
}

// ---- the engine's error paths

#[test]
fn a_recoverable_error_fails_the_round_and_the_engine_serves_on() {
    let mut e = make_engine(false);
    e.submit(requests().remove(1)).unwrap();
    e.backend.m.fail = Some((Site::Window, ModelError::Request("transient".into())));
    let err = e.run().unwrap_err();
    assert!(matches!(&err, StepError::Backend(m) if m.contains("transient")), "{err:?}");
    assert_eq!(e.fail_active("round failed"), 1);
    // the slot came back clean: the same request now matches a fresh engine's output
    let again = {
        e.submit(requests().remove(1)).unwrap();
        e.run().unwrap();
        e.take_outputs().remove(0).tokens
    };
    let want = {
        let mut f = make_engine(false);
        f.submit(requests().remove(1)).unwrap();
        f.run().unwrap();
        f.take_outputs().remove(0).tokens
    };
    assert_eq!(again, want);
    assert!(e.backend.m.violations.is_empty());
}

#[test]
fn a_fatal_error_stops_the_engine() {
    let mut e = make_engine(false);
    e.submit(requests().remove(2)).unwrap();
    e.backend.m.fail = Some((Site::Window, ModelError::StateLost("accepted rows were not replayed".into())));
    let err = e.run().unwrap_err();
    assert!(err.is_fatal(), "{err:?}");
}

// ---- reuse of one backend across harness runs (the I3 shape)

#[derive(Clone)]
struct Shared(Rc<RefCell<Backend64>>);

impl Backend for Shared {
    fn vocab_size(&self) -> usize {
        self.0.borrow().vocab_size()
    }
    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.0.borrow_mut().prefill(slot, start, tokens, candidates)
    }
    fn prefill_piece(&mut self, slot: usize, start: u64, tokens: &[u32], total: usize, candidates: usize) -> RowOutput {
        self.0.borrow_mut().prefill_piece(slot, start, tokens, total, candidates)
    }
    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        self.0.borrow_mut().step(reqs, candidates)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.0.borrow_mut().commit(slot, keep)
    }
    fn release(&mut self, slot: usize) {
        self.0.borrow_mut().release(slot)
    }
}

#[test]
fn one_backend_serves_every_harness_check_in_turn() {
    let shared = Shared(Rc::new(RefCell::new(backend())));
    let make_shared = || shared.clone();
    let cases: Vec<(Vec<u32>, Vec<u32>)> =
        (0..4).map(|i| (tokens(6 + i * 7, i), (0..(2 + i * 5)).map(|j| (j * 3 + 1) as u32).collect())).collect();
    row_invariance(&make_shared, &cases, 28).unwrap();
    let prompt = tokens(150, 22);
    chunked_prefill_equals_whole(&make_shared, &prompt, &[10, 64, 100], &sampling(9), 16).unwrap();
    let window: Vec<u32> = (0..20).map(|j| (j * 7 % 200) as u32 + 4).collect();
    for keep in [1, 7, 20] {
        rollback_equals_replay(&make_shared, &tokens(40, 23), &window, keep, &[5, 6], 28).unwrap();
    }
    let b = shared.0.borrow();
    assert!(b.m.violations.is_empty(), "{:?}", b.m.violations);
}

// ---- a live slot that is begun again

#[test]
fn a_prefill_at_zero_on_a_live_slot_ends_it_first_and_begins_it_again() {
    let mut b = backend();
    // by `prefill` (position 0) and by `prefill_slice` (done 0), three times over, with no release between: the runtime refuses to
    // begin a slot that is already begun (the fake records a violation and an error), so the backend must end it first
    for round in 0..3 {
        let p = tokens(9 + round, 30 + round);
        let row = b.prefill(0, 0, &p, 4);
        assert!(row.values.iter().all(|v| v.is_finite()), "round {round}: {row:?}");
        assert!(b.take_error().is_none(), "round {round}");
        assert_eq!(b.m.committed(0), Some(&p[..]), "the new sequence replaced the old");
        let p2 = tokens(70, 40 + round);
        let step = b.prefill_slice(0, &p2, &[], 0, 4);
        assert!(step.row.is_none() || step.row.unwrap().values.iter().all(|v| v.is_finite()));
        assert!(b.take_error().is_none(), "round {round}: slice");
        assert_eq!(b.live_slots(), vec![0]);
    }
    // one begin per prefill, and an end before every begin but the first
    assert_eq!(b.m.count(|c| *c == Call::Begin(0)), 6);
    assert_eq!(b.m.ends(0), 5);
    b.release(0);
    assert_eq!(b.m.ends(0), 6, "and the release ends the last one, once");
    assert_clean(&b);
}

#[test]
fn a_begin_that_fails_after_the_old_sequence_was_ended_is_cleaned_up_once() {
    let mut b = backend();
    b.prefill(3, 0, &tokens(10, 50), 4);
    assert_eq!(b.m.ends(3), 0);
    b.m.fail = Some((Site::Begin, ModelError::Request("pool is full".into())));
    let row = b.prefill(3, 0, &tokens(12, 51), 4);
    assert!(row.values.iter().all(|v| v.is_nan()));
    assert!(b.take_error().unwrap().message.contains("pool is full"));
    // the old sequence was ended before the begin, and the half-begun slot after it: two ends, each of a slot that existed
    assert_eq!(b.m.ends(3), 2);
    assert_eq!(b.live_slots(), Vec::<usize>::new());
    b.release(3);
    assert_eq!(b.m.ends(3), 2);
    assert_clean(&b);
}

// ---- the speculation runtime on the backend

fn spec_backend(script: HeadScript, mode: SpecMode) -> Backend64 {
    let rt = FakeRuntime::new(SLOTS).with_head(script);
    backend_with(rt, 4096).with_speculation(SpecConfig::new(mode), &CostTables::shipped()).unwrap()
}

fn greedy() -> Sampling {
    Sampling { seed: 0, temperature: 0.0, top_k: 20, top_p: 1.0, min_p: 0.0 }
}

fn sampled(seed: u64) -> Sampling {
    Sampling { seed, temperature: 0.8, top_k: 20, top_p: 0.95, min_p: 0.0 }
}

/// Prefill `prompt` into `slot` and return the sequence a round starts from: the prompt and the token sampled after it.
fn start(b: &mut Backend64, slot: usize, prompt: &[u32]) -> Vec<u32> {
    let row = b.prefill(slot, 0, prompt, 20);
    let mut seq = prompt.to_vec();
    seq.push(gb_sampling::choose(&row.values, &row.ids, prompt.len() as u64, &greedy()));
    seq
}

fn ask<'a>(slot: usize, tokens: &'a [u32], sampling: &'a Sampling, max: usize) -> ProposeRequest<'a> {
    ProposeRequest { slot, tokens, sampling, max }
}

#[test]
fn a_backend_without_speculation_has_no_say_and_the_engines_drafter_fills_in() {
    let mut b = backend();
    let seq = start(&mut b, 0, &tokens(12, 60));
    let g = greedy();
    assert_eq!(b.propose_batch_scored(&[ask(0, &seq, &g, 3)]), vec![None]);
    assert_eq!(b.propose_batch(&[ask(0, &seq, &g, 3)]), vec![None]);
    assert!(b.propose_batch_scored(&[]).is_empty());
    assert_clean(&b);
}

#[test]
fn every_stream_that_wants_head_drafts_is_asked_in_one_call_with_its_own_sampler_when_it_is_sampled() {
    let mut b = spec_backend(HeadScript::perfect(), SpecMode::Mtp);
    let (p0, p1, p2) = (tokens(14, 61), tokens(15, 62), tokens(16, 63));
    let (s0, s1, s2) = (start(&mut b, 0, &p0), start(&mut b, 1, &p1), start(&mut b, 2, &p2));
    let (g, a, c) = (greedy(), sampled(1234), sampled(5678));
    let out = b.propose_batch_scored(&[ask(0, &s0, &g, 3), ask(1, &s1, &a, 3), ask(2, &s2, &c, 2)]);
    // ONE draft call served the three streams, at the fixed depth each, capped by what the engine can take
    let calls = b.m.mtp_calls();
    assert_eq!(calls.len(), 1, "{calls:?}");
    let (slots, lens, depths, keyed) = &calls[0];
    assert_eq!((slots.as_slice(), depths.as_slice()), (&[0, 1, 2][..], &[3, 3, 2][..]));
    assert_eq!(lens.as_slice(), &[s0.len(), s1.len(), s2.len()]);
    // a sampled request carries its own sampler (its seed), a greedy one has no noise to key
    assert_eq!(keyed.as_slice(), &[None, Some(1234), Some(5678)]);
    // the drafts and the head's confidence in each, aligned
    let lens: Vec<(usize, usize)> = out.iter().map(|p| p.as_ref().map(|p| (p.tokens.len(), p.conf.len())).unwrap()).collect();
    assert_eq!(lens, vec![(3, 3), (3, 3), (2, 2)]);
    assert!(out.iter().all(|p| p.as_ref().unwrap().tokens.iter().all(|&t| (t as usize) < VOCAB)));
    // the scored answer and the plain one agree on the tokens
    let mut b2 = spec_backend(HeadScript::perfect(), SpecMode::Mtp);
    let t0 = start(&mut b2, 0, &p0);
    assert_eq!(t0, s0);
    let plain = b2.propose_batch(&[ask(0, &t0, &g, 3)]);
    assert_eq!(plain, vec![Some(out[0].as_ref().unwrap().tokens.clone())]);
    assert_clean(&b);
    assert_clean(&b2);
}

#[test]
fn the_argmax_setting_hands_no_sampler_to_anyone() {
    let rt = FakeRuntime::new(SLOTS).with_head(HeadScript::perfect());
    let mut cfg = SpecConfig::new(SpecMode::Mtp);
    cfg.noise = gb_spec::DraftPick::Argmax;
    let mut b = backend_with(rt, 4096).with_speculation(cfg, &CostTables::shipped()).unwrap();
    let s0 = start(&mut b, 0, &tokens(10, 64));
    let a = sampled(99);
    b.propose_batch_scored(&[ask(0, &s0, &a, 2)]);
    assert_eq!(b.m.mtp_calls()[0].3, vec![None]);
    assert_clean(&b);
}

#[test]
fn a_head_error_costs_the_round_its_head_drafts_and_nothing_else() {
    let mut b = spec_backend(HeadScript::Errors { from_call: 1, count: 1, fatal: false }, SpecMode::Mtp);
    let prompt = tokens(12, 65);
    let seq = start(&mut b, 0, &prompt);
    let g = greedy();
    let out = b.propose_batch_scored(&[ask(0, &seq, &g, 3)]);
    // the engine's own drafter is NOT consulted (`Some`): the round goes on with no drafts, and no request fails
    assert_eq!(out, vec![Some(Proposal::default())]);
    assert!(b.take_error().is_none(), "a recoverable head error is not the request's error");
    assert!(b.m.calls.contains(&Call::MtpDisable(0)), "the runtime was told to stop drafting for the slot");
    assert!(b.spec.as_ref().unwrap().mtp_off(0));
    assert_eq!(b.spec.as_ref().unwrap().stats.mtp_errors, 1);
    // the slot is not asked again, whatever the runtime says about it
    let out = b.propose_batch_scored(&[ask(0, &seq, &g, 3)]);
    assert_eq!(out, vec![Some(Proposal::default())]);
    assert_eq!(b.m.mtp_draft_calls, 1);
    // and the trunk was never touched: a window runs, commits and gives the serial rows
    let rows = b.step(&[RowRequest { slot: 0, first_position: prompt.len() as u64, tokens: vec![*seq.last().unwrap()] }], 8);
    assert!(rows[0][0].values.iter().all(|v| v.is_finite()));
    b.commit(0, 1);
    assert!(b.take_error().is_none());
    assert_clean(&b);
}

#[test]
fn a_head_error_switches_off_every_slot_of_the_call_and_a_new_sequence_in_the_slot_drafts_again() {
    let mut b = spec_backend(HeadScript::Errors { from_call: 1, count: 1, fatal: false }, SpecMode::Mtp);
    let (s0, s1) = (start(&mut b, 0, &tokens(10, 66)), start(&mut b, 1, &tokens(11, 67)));
    let g = greedy();
    let out = b.propose_batch_scored(&[ask(0, &s0, &g, 2), ask(1, &s1, &g, 2)]);
    assert_eq!(out, vec![Some(Proposal::default()), Some(Proposal::default())]);
    assert!(b.m.calls.contains(&Call::MtpDisable(0)) && b.m.calls.contains(&Call::MtpDisable(1)));
    assert!(b.spec.as_ref().unwrap().mtp_off(0) && b.spec.as_ref().unwrap().mtp_off(1));
    // slot 0 starts a new sequence: its drafting is back (the scripted failure was one call)
    let s0 = start(&mut b, 0, &tokens(13, 68));
    assert!(!b.spec.as_ref().unwrap().mtp_off(0));
    let out = b.propose_batch_scored(&[ask(0, &s0, &g, 2)]);
    assert_eq!(out[0].as_ref().unwrap().tokens.len(), 2);
    assert_clean(&b);
}

#[test]
fn a_fatal_head_error_is_latched_like_every_other_fatal_error() {
    let mut b = spec_backend(HeadScript::Errors { from_call: 1, count: usize::MAX, fatal: true }, SpecMode::Mtp);
    let seq = start(&mut b, 0, &tokens(10, 69));
    let g = greedy();
    let out = b.propose_batch_scored(&[ask(0, &seq, &g, 3)]);
    assert_eq!(out, vec![None], "the engine is told the backend has no answer");
    let e = b.take_error().unwrap();
    assert!(e.fatal && e.message.contains("drafting"), "{e:?}");
    // latched: it stays, and nothing is asked of the runtime again
    assert!(b.take_error().unwrap().fatal);
    let calls = b.m.mtp_draft_calls;
    assert_eq!(b.propose_batch_scored(&[ask(0, &seq, &g, 3)]), vec![None]);
    assert_eq!(b.m.mtp_draft_calls, calls);
    assert_clean(&b);
}

#[test]
fn a_request_for_a_slot_the_model_does_not_have_is_refused_by_name_not_indexed() {
    let mut b = spec_backend(HeadScript::perfect(), SpecMode::Mtp);
    let g = greedy();
    let seq = vec![5u32; 12];
    let out = b.propose_batch_scored(&[ask(SLOTS + 3, &seq, &g, 3)]);
    assert_eq!(out, vec![None]);
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("slot"), "{e:?}");
    assert_eq!(b.m.mtp_draft_calls, 0);
}

#[test]
fn the_head_is_not_asked_for_a_stream_that_cannot_draft_and_the_copy_mode_never_asks_it() {
    // `--spec copy` with a runtime that has a head: the head is never called, whatever the streams hold
    let mut b = spec_backend(HeadScript::perfect(), SpecMode::Copy);
    let seq = start(&mut b, 0, &tokens(30, 70));
    let g = greedy();
    assert_eq!(b.propose_batch_scored(&[ask(0, &seq, &g, 3)]), vec![Some(Proposal::default())]);
    assert_eq!(b.m.mtp_draft_calls, 0);
    // a stream that may take no draft this round (the last token of its budget) is not asked either
    let mut b = spec_backend(HeadScript::perfect(), SpecMode::Mtp);
    let seq = start(&mut b, 0, &tokens(30, 71));
    assert_eq!(b.propose_batch_scored(&[ask(0, &seq, &g, 0)]), vec![Some(Proposal::default())]);
    assert_eq!(b.m.mtp_draft_calls, 0);
    assert_clean(&b);
}

/// Commit `rounds` one-row windows on `slot`, so that the head's store holds `1 + rounds` pending rows (the prompt's carry and one a
/// round). The tokens are distinct (no 8-gram repeats: the copy drafter has nothing to say) and the model's own choice is not needed.
fn pile_up(b: &mut Backend64, slot: usize, seq: &mut Vec<u32>, rounds: usize) {
    for _ in 0..rounds {
        let pos = (seq.len() - 1) as u64;
        b.step(&[RowRequest { slot, first_position: pos, tokens: vec![*seq.last().unwrap()] }], 20);
        b.commit(slot, 1);
        seq.push(((seq.len() * 13 + 1000) % 480) as u32 + 4);
    }
}

#[test]
fn a_slot_near_its_stores_capacity_absorbs_even_when_the_arbiter_drafts_nothing() {
    // a head so dear that the arbiter never picks it: the only head work is the forced absorb that makes room in the store
    let mut costs = CostTables::shipped();
    costs.mtp.lm_head = 1.0e6;
    costs.mtp.step = 1.0e6;
    let rt = FakeRuntime::new(SLOTS).with_head(HeadScript::perfect());
    let mut b = backend_with(rt, 4096).with_speculation(SpecConfig::new(SpecMode::Auto), &costs).unwrap();
    // no probe of the head either (the arbiter's one-deep try after 16 rounds without it): the test is about the forced absorb
    b.spec.as_mut().unwrap().arbiter.cfg.probe_every = u32::MAX;
    let mut seq = start(&mut b, 0, &tokens(10, 72));
    let g = greedy();
    // the store holds 24 rows: another window of up to 8 fits (24 + 8 = 32), so nothing is absorbed and nothing is drafted
    pile_up(&mut b, 0, &mut seq, 23);
    assert_eq!(b.m.mtp_pending_rows(0), Some(24));
    assert_eq!(MTP_TAP_ROWS, TAP_ROWS, "the fake's store has the capacity the backend assumes");
    let out = b.propose_batch_scored(&[ask(0, &seq, &g, 7)]);
    assert!(out[0].as_ref().unwrap().tokens.is_empty());
    assert_eq!(b.m.mtp_draft_calls, 0, "room for a whole window: no head work");
    // one more row and a window of eight would overflow: the slot absorbs (depth 0) and drafts nothing
    pile_up(&mut b, 0, &mut seq, 1);
    assert_eq!(b.m.mtp_pending_rows(0), Some(25));
    let out = b.propose_batch_scored(&[ask(0, &seq, &g, 7)]);
    assert!(out[0].as_ref().unwrap().tokens.is_empty(), "an absorb-only ask drafts nothing");
    let calls = b.m.mtp_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!((calls[0].0.as_slice(), calls[0].2.as_slice()), (&[0][..], &[0][..]), "one slot, depth 0");
    assert_eq!(b.m.mtp_pending_rows(0), Some(0), "the store was emptied");
    assert_eq!(b.spec.as_ref().unwrap().stats.absorb_only_calls, 1);
    assert!(b.m.head_lost.is_empty(), "{:?}", b.m.head_lost);
    // and the slot is still a drafting slot: after another round it has a row to absorb again
    pile_up(&mut b, 0, &mut seq, 1);
    assert_eq!(b.m.mtp_pending_rows(0), Some(1));
    assert_clean(&b);
}

#[test]
fn a_slot_that_never_absorbs_loses_its_head_to_the_full_store_which_is_what_the_absorb_prevents() {
    // the fake mirrors the real store: more rows than it holds switch the slot's drafting off (never an error)
    let mut b = spec_backend(HeadScript::perfect(), SpecMode::Auto);
    let mut seq = start(&mut b, 0, &tokens(10, 73));
    pile_up(&mut b, 0, &mut seq, TAP_ROWS + 2);
    assert_eq!(b.m.mtp_pending_rows(0), None);
    assert!(b.m.head_off(0));
    assert_eq!(b.m.head_lost.len(), 1);
    // a slot with no head state is not asked, and its round goes on without head drafts
    let g = greedy();
    let out = b.propose_batch_scored(&[ask(0, &seq, &g, 3)]);
    assert!(out[0].as_ref().unwrap().tokens.is_empty());
    assert_eq!(b.m.mtp_draft_calls, 0);
}

#[test]
fn the_image_hook_is_the_interval_test_and_is_always_true_until_images_are_built() {
    let spans = vec![10..20, 40..45];
    assert!(window_is_text(&[], 0, 100));
    assert!(window_is_text(&spans, 0, 10));
    assert!(!window_is_text(&spans, 0, 11));
    assert!(!window_is_text(&spans, 12, 2));
    assert!(!window_is_text(&spans, 19, 5));
    assert!(window_is_text(&spans, 20, 20));
    assert!(!window_is_text(&spans, 20, 21));
    assert!(window_is_text(&spans, 45, 1000));
    assert!(!window_is_text(&spans, 15, 0));
    assert!(!window_is_text(&spans, 5, usize::MAX), "a huge row count does not overflow");
    // the backend's hook: no slot has a span (a prompt with images is refused by name), so every window is text
    let b = spec_backend(HeadScript::perfect(), SpecMode::Mtp);
    assert!(b.spec_allowed_for_window(0, 0, 4096) && b.spec_allowed_for_window(SLOTS, 0, 1));
}

#[test]
fn a_draft_outside_the_vocabulary_is_dropped_before_the_trunk_sees_it() {
    let mut b = spec_backend(HeadScript::BadIds, SpecMode::Mtp);
    let seq = start(&mut b, 0, &tokens(12, 80));
    let g = greedy();
    let out = b.propose_batch_scored(&[ask(0, &seq, &g, 3)]);
    assert_eq!(out, vec![Some(Proposal::default())], "the whole offer is dropped, not trimmed to its valid prefix");
    assert_eq!(b.spec.as_ref().unwrap().stats.mtp_bad_drafts, 1);
    assert!(b.take_error().is_none());
    assert_clean(&b);
}

#[test]
fn the_absorb_rule_covers_the_commit_of_the_round_for_every_store_level_and_every_draft_count() {
    // a head so dear that the arbiter never picks it: the only head work is the absorb the rule forces. For every number of pending rows
    // and every number of drafts the engine may ask for, the round's window commits all its rows (the base row and `max` drafts) after the
    // propose of that round: the slot must be alive afterwards, and an absorb is made exactly when the commit would not fit
    let mut costs = CostTables::shipped();
    costs.mtp.lm_head = 1.0e6;
    costs.mtp.step = 1.0e6;
    for n in 1..=TAP_ROWS {
        for max in [1usize, 2, 3, 4, 7] {
            let rt = FakeRuntime::new(SLOTS).with_head(HeadScript::perfect());
            let mut b = backend_with(rt, 4096).with_speculation(SpecConfig::new(SpecMode::Auto), &costs).unwrap();
            b.spec.as_mut().unwrap().arbiter.cfg.probe_every = u32::MAX;
            let mut seq = start(&mut b, 0, &tokens(10, 90));
            pile_up(&mut b, 0, &mut seq, n - 1);
            assert_eq!(b.m.mtp_pending_rows(0), Some(n), "n {n}");
            let g = greedy();
            let out = b.propose_batch_scored(&[ask(0, &seq, &g, max)]);
            assert!(out[0].as_ref().unwrap().tokens.is_empty(), "the arbiter drafts nothing here");
            let must = n + 1 + max > TAP_ROWS;
            assert_eq!(
                b.m.mtp_draft_calls,
                usize::from(must),
                "n {n} max {max}: an absorb exactly when the round's commit would not fit"
            );
            // the round's window and its full commit
            let pos = (seq.len() - 1) as u64;
            let window: Vec<u32> = std::iter::once(*seq.last().unwrap()).chain((0..max).map(|j| 300 + j as u32)).collect();
            b.step(&[RowRequest { slot: 0, first_position: pos, tokens: window }], 8);
            b.commit(0, 1 + max);
            assert!(b.take_error().is_none());
            assert!(b.m.head_lost.is_empty(), "n {n} max {max}: {:?}", b.m.head_lost);
            let after = b.m.mtp_pending_rows(0).expect("the slot still drafts");
            assert_eq!(after, if must { 1 + max } else { n + 1 + max }, "n {n} max {max}");
            assert_clean(&b);
        }
    }
}

#[test]
fn one_call_serves_a_stream_with_a_nearly_full_store_and_a_fresh_one() {
    let rt = FakeRuntime::new(SLOTS).with_head(HeadScript::perfect());
    let mut b = backend_with(rt, 4096).with_speculation(SpecConfig::new(SpecMode::Mtp), &CostTables::shipped()).unwrap();
    // `--spec mtp` drafts every ready stream at the fixed depth, and a draft call absorbs: slot 0 has piled up rows without one
    let mut s0 = start(&mut b, 0, &tokens(10, 91));
    let s1 = start(&mut b, 1, &tokens(12, 92));
    pile_up(&mut b, 0, &mut s0, 26);
    assert_eq!(b.m.mtp_pending_rows(0), Some(27));
    let g = greedy();
    let out = b.propose_batch_scored(&[ask(0, &s0, &g, 3), ask(1, &s1, &g, 3)]);
    let calls = b.m.mtp_calls();
    assert_eq!(calls.len(), 1, "one call serves both");
    assert_eq!(calls[0].0, vec![0, 1]);
    assert_eq!(out[0].as_ref().unwrap().tokens.len(), 3);
    assert_eq!(out[1].as_ref().unwrap().tokens.len(), 3);
    assert_eq!((b.m.mtp_pending_rows(0), b.m.mtp_pending_rows(1)), (Some(0), Some(0)));
    assert!(b.m.head_lost.is_empty());
}

#[test]
fn an_absorb_only_ask_rides_in_the_same_call_as_the_drafts_of_the_other_stream() {
    // slot 0 is kept off the head by its cost (a dear head for it is not expressible per stream), so make the arbiter's choice by the
    // acceptance estimates instead: slot 0's estimates say the head is useless, slot 1's say it is perfect
    let rt = FakeRuntime::new(SLOTS).with_head(HeadScript::perfect());
    // a verify table whose extra rows are dear (the measured one has flattened to 1 to 3 ms a row past 8 rows, where a poor stream's
    // first draft is worth riding in the window: the arbiter's own call, which this test is not about)
    let dear = CostTables::shipped().with_verify(&[(1, 38.0), (2, 50.0), (4, 80.0), (8, 140.0), (16, 280.0), (64, 1100.0)]);
    let mut b = backend_with(rt, 4096).with_speculation(SpecConfig::new(SpecMode::Auto), &dear).unwrap();
    // no probes (set before the sequences begin: a stream's state takes the gap at its start)
    b.spec.as_mut().unwrap().arbiter.cfg.probe_every = u32::MAX;
    let mut s0 = start(&mut b, 0, &tokens(10, 94));
    let s1 = start(&mut b, 1, &tokens(12, 95));
    {
        let arb = &mut b.spec.as_mut().unwrap().arbiter;
        for _ in 0..40 {
            arb.observe(0, gb_spec::arbiter::Choice::Mtp(3), 3, 0);
            arb.observe(1, gb_spec::arbiter::Choice::Mtp(3), 3, 3);
        }
    }
    pile_up(&mut b, 0, &mut s0, 26);
    let g = greedy();
    let out = b.propose_batch_scored(&[ask(0, &s0, &g, 7), ask(1, &s1, &g, 7)]);
    let calls = b.m.mtp_calls();
    assert_eq!(calls.len(), 1);
    let (slots, _, depths, _) = &calls[0];
    assert_eq!(slots, &vec![0, 1], "one call");
    assert_eq!(depths[0], 0, "the stream the arbiter keeps off the head is only absorbed");
    assert!(depths[1] >= 1, "the other drafts");
    assert!(out[0].as_ref().unwrap().tokens.is_empty() && !out[1].as_ref().unwrap().tokens.is_empty());
    assert_eq!(b.m.mtp_pending_rows(0), Some(0));
    assert!(b.m.head_lost.is_empty());
}

#[test]
fn a_commit_of_more_than_the_stores_rows_switches_the_slot_off_which_no_round_of_the_engine_makes() {
    // the limit is the runtime's own: a window of 33 rows kept whole overflows a store that had room for 32, whatever it held (the
    // harness commits windows of up to 64 rows; the engine commits at most 1 + 7)
    let mut b = spec_backend(HeadScript::perfect(), SpecMode::Mtp);
    let seq = start(&mut b, 0, &tokens(10, 93));
    let pos = (seq.len() - 1) as u64;
    let window: Vec<u32> = (0..=TAP_ROWS as u32).map(|j| 200 + j).collect();
    assert_eq!(window.len(), TAP_ROWS + 1);
    b.step(&[RowRequest { slot: 0, first_position: pos, tokens: window }], 8);
    b.commit(0, TAP_ROWS + 1);
    assert!(b.take_error().is_none(), "never an error: the slot just drafts nothing");
    assert_eq!(b.m.mtp_pending_rows(0), None);
    assert_eq!(b.m.head_lost.len(), 1);
    let g = greedy();
    let mut longer = seq.clone();
    longer.extend(200..200 + TAP_ROWS as u32 + 1);
    assert_eq!(b.propose_batch_scored(&[ask(0, &longer, &g, 3)]), vec![Some(Proposal::default())]);
    assert_eq!(b.m.mtp_draft_calls, 0);
}
