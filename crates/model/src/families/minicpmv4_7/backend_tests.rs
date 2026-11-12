use super::super::config::tests::real;
use super::*;
use crate::error::ModelError;
use crate::families::minicpmv4_7::splice::{FrameLayout, HostFeatures, IMAGE_PAD, VIDEO_PAD};
use crate::forward::{Candidates, PrefillPath};
use crate::mrope_canvas::{canvas_positions, tests as rope_fx, CanvasInput, Downsample};
use crate::rows::RowBatch;
use gb_engine::prefix::Captured;
use gb_engine::{Backend, RowRequest};
use std::sync::Arc;

/// One multimodal chunk as the fake runtime saw it.
#[derive(Clone, Debug)]
struct MmCall {
    first: usize,
    rows: usize,
    rope3: Option<Vec<i32>>,
    /// The slot's rope delta at the time of the call.
    delta: i32,
    /// `(address of the features, dst row, src row, n)`.
    splice: Vec<(usize, usize, usize, usize)>,
    want: Option<usize>,
    path: PrefillPath,
}

/// A runtime that records what it was asked and answers each row with candidates that depend only on the row's token and position
/// (so a row's output is the same alone and in a batch, as the real forward's is).
#[derive(Default)]
struct Fake {
    calls: Vec<(usize, usize, PrefillPath, Option<usize>)>,
    windows: Vec<Vec<(usize, usize, usize)>>,
    fail: Option<ModelError>,
    committed: Vec<(usize, usize)>,
    begun: Vec<usize>,
    ended: Vec<usize>,
    flushes: usize,
    /// A draft is attached: what it answers per slot (`None`: cannot draft), and the asks it saw (slot, anchor, max).
    draft: Option<Vec<Option<SpecProposal>>>,
    asks: Vec<Asked>,
    propose_fails: Option<ModelError>,
    mm: Vec<MmCall>,
    rows_limit: Option<usize>,
    delta: [i32; SLOTS],
    delta_log: Vec<(usize, i32)>,
    /// The prefix hooks: slices with cuts as `(slot, done, cuts)`, restores as `(slot, len)`, and the handles released.
    slices: Vec<(usize, usize, Vec<usize>)>,
    restores: Vec<(usize, usize)>,
    released: Vec<u32>,
    /// The prefill MoE arm the fake runs: E3's CoopMK instead of the grouped default.
    coopmk: bool,
    /// The identity of the draft the fake claims to run (E4b): the namespace of the cache is keyed by it.
    identity: Option<String>,
}

/// One draft round's asks: (slot, anchor, max) of each stream.
type Asked = Vec<(usize, u32, usize)>;

const VOCAB: usize = 42;
const SLOTS: usize = 4;
const CONTEXT: usize = 100;

impl MiniCpmRuntime for Fake {
    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.begun.push(slot);
        Ok(())
    }
    fn end_slot(&mut self, slot: usize) {
        self.ended.push(slot);
    }
    fn reserve(&mut self, slot: usize, first: usize, n: usize) -> Result<(), ModelError> {
        if slot >= SLOTS {
            return Err(ModelError::Request(format!("slot {slot} of {SLOTS}")));
        }
        if first + n > CONTEXT {
            return Err(ModelError::Request(format!("position {} is past the context of {CONTEXT}", first + n - 1)));
        }
        Ok(())
    }
    fn prefill_chunk(&mut self, b: &RowBatch, path: PrefillPath, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        self.calls.push((b.positions[0] as usize, b.rows, path, want));
        if let Some(e) = self.fail.take() {
            return Err(e);
        }
        Ok(want.map(|count| Candidates { rows: 1, count, ids: vec![7; count], vals: vec![1.0; count] }))
    }
    fn prefill_chunk_mm(
        &mut self,
        b: &RowBatch,
        path: PrefillPath,
        want: Option<usize>,
        splice: &[ChunkSplice<'_>],
    ) -> Result<Option<Candidates>, ModelError> {
        let slot = b.seq_slot[0] as usize;
        self.mm.push(MmCall {
            first: b.positions[0] as usize,
            rows: b.rows,
            rope3: b.rope3.clone(),
            delta: self.delta[slot],
            splice: splice
                .iter()
                .map(|s| {
                    (s.features as *const (dyn std::any::Any + Send + Sync) as *const () as usize, s.dst_row, s.src_row, s.n)
                })
                .collect(),
            want,
            path,
        });
        if let Some(e) = self.fail.take() {
            return Err(e);
        }
        Ok(want.map(|count| Candidates { rows: 1, count, ids: vec![7; count], vals: vec![1.0; count] }))
    }
    fn prefill_rows(&self) -> usize {
        self.rows_limit.unwrap_or(usize::MAX)
    }
    fn set_rope_delta(&mut self, slot: usize, delta: i32) -> Result<(), ModelError> {
        self.delta[slot] = delta;
        self.delta_log.push((slot, delta));
        Ok(())
    }
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        if let Some(e) = self.fail.take() {
            return Err(e);
        }
        self.windows.push((0..b.rows).map(|r| (b.row_slot[r] as usize, b.positions[r] as usize, b.ids[r] as usize)).collect());
        let mut ids = Vec::new();
        let mut vals = Vec::new();
        for r in 0..b.rows {
            for c in 0..count {
                ids.push(((b.ids[r] as usize + c) % VOCAB) as i32);
                vals.push(b.positions[r] as f32 + b.row_slot[r] as f32 * 0.25 - c as f32);
            }
        }
        Ok(Candidates { rows: b.rows, count, ids, vals })
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.committed.push((slot, keep));
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        self.flushes += 1;
        Ok(())
    }
    fn has_draft(&self) -> bool {
        self.draft.is_some()
    }
    fn draft_identity(&self) -> Option<String> {
        self.identity.clone()
    }
    fn propose_round(&mut self, asks: &[SpecAsk<'_>]) -> Result<Vec<Option<SpecProposal>>, ModelError> {
        self.asks.push(asks.iter().map(|a| (a.slot, *a.tokens.last().unwrap(), a.max)).collect());
        if let Some(e) = self.propose_fails.take() {
            return Err(e);
        }
        let answers = self.draft.clone().unwrap_or_default();
        Ok(asks
            .iter()
            .enumerate()
            .map(|(i, a)| {
                answers.get(i).cloned().flatten().map(|mut p| {
                    p.tokens.truncate(a.max);
                    p.accept.truncate(a.max);
                    p
                })
            })
            .collect())
    }
    fn prefill_moe_grouped(&self) -> bool {
        !self.coopmk
    }
    fn cached_slice(&mut self, arena: &mut SnapshotArena, r: SliceRequest<'_>) -> Result<SliceResult, ModelError> {
        self.slices.push((r.slot, r.done, r.cuts.to_vec()));
        if r.done == 0 {
            self.begun.push(r.slot);
        }
        if let Some(e) = self.fail.take() {
            return Err(e);
        }
        let cuts = if prefix::path(r.prompt.len(), !self.coopmk).is_some() { r.cuts } else { &[] };
        let end = prefix::slice_end(r.done, r.prompt.len(), r.chunk, cuts);
        let captured = match arena.book_with_path(end, 0, prefix::arm_path(!self.coopmk)) {
            Some(id) if end < r.prompt.len() && cuts.contains(&end) => {
                vec![Captured { len: end, handle: PrefixHandle { snap: id, pages: (0..end.div_ceil(256) as u32).collect() } }]
            }
            Some(id) => {
                arena.unbook(id).unwrap();
                Vec::new()
            }
            None => Vec::new(),
        };
        let row = (end == r.prompt.len()).then(|| Candidates {
            rows: 1,
            count: r.candidates,
            ids: vec![7; r.candidates],
            vals: vec![1.0; r.candidates],
        });
        Ok(SliceResult { end, row, captured })
    }
    fn restore_snapshot(
        &mut self,
        arena: &SnapshotArena,
        slot: usize,
        handle: &PrefixHandle,
        len: usize,
    ) -> Result<(), ModelError> {
        self.restores.push((slot, len));
        if let Some(e) = self.fail.take() {
            return Err(e);
        }
        match arena.meta(handle.snap) {
            Some(m) if m.len == len => {
                // the real restore puts the snapshot's rope delta in the slot over whatever it had
                self.delta[slot] = m.rope_delta;
                Ok(())
            }
            _ => Err(ModelError::Request("prefix handle or token length is invalid".into())),
        }
    }
    fn release_snapshot(&mut self, arena: &mut SnapshotArena, handle: PrefixHandle) -> Result<(), ModelError> {
        self.released.push(handle.snap);
        arena.unbook(handle.snap)
    }
    fn capture_snapshot(&mut self, arena: &mut SnapshotArena, slot: usize, len: usize) -> Result<Option<Captured>, ModelError> {
        if let Some(e) = self.fail.take() {
            return Err(e);
        }
        Ok(arena
            .book_with_path(len, self.delta[slot], prefix::arm_path(!self.coopmk))
            .map(|id| Captured { len, handle: PrefixHandle { snap: id, pages: (0..len.div_ceil(256) as u32).collect() } }))
    }
}

const FINGERPRINT: &str = "sha256:9bc579ceb1ae00bd016922d354e340bb395e9da52438cc226d1c898d5e63d03f";

/// A backend with a bookkeeping-only arena of `snapshots` snapshot slots.
fn cached(chunk: usize, snapshots: usize) -> MiniCpmBackend<Fake> {
    let c = real();
    let unit = prefix::snapshot_bytes(&c) + 8 * prefix::page_bytes(&c);
    let budget = prefix::budget_for_bytes(&c, snapshots as u64 * unit);
    assert_eq!(budget.snapshot_slots, snapshots);
    let arena = SnapshotArena::bookkeeping_only(&c, budget).unwrap();
    MiniCpmBackend::with_runtime(Fake::default(), chunk).with_prefix_arena(arena, FINGERPRINT.into()).unwrap()
}

fn req(slot: usize, first: u64, tokens: Vec<u32>) -> RowRequest {
    RowRequest { slot, first_position: first, tokens }
}

#[test]
fn prompt_total_fixes_path_for_one_row_tail_and_resume_grid() {
    let mut b = MiniCpmBackend::with_runtime(Fake::default(), 64);
    let prompt = vec![3; 130];
    for done in [0, 64, 128] {
        let step = b.prefill_slice(1, &prompt, &[], done, 1);
        assert_eq!(step.done, (done + 64).min(130));
    }
    assert_eq!(
        b.m.calls,
        vec![(0, 64, PrefillPath::Tiled, None), (64, 64, PrefillPath::Tiled, None), (128, 2, PrefillPath::Tiled, Some(1))]
    );
    assert_eq!(b.m.begun, vec![1], "only a slice that starts the prompt begins the slot");
    let mut short = MiniCpmBackend::with_runtime(Fake::default(), 64);
    short.prefill_slice(1, &vec![3; 128], &[], 0, 1);
    assert_eq!(short.m.calls[0].2, PrefillPath::Gemv);
    assert_eq!(next_chunk_end(67, 130, 64), 128, "a resumed slice ends at the next absolute boundary");
    assert_eq!(next_chunk_end(128, 129, 64), 129);
}

#[test]
fn a_piece_keeps_the_total_path_even_if_the_first_piece_is_short() {
    let mut b = MiniCpmBackend::with_runtime(Fake::default(), 64);
    b.prefill_piece(1, 0, &vec![3; 65], 129, 1);
    assert_eq!(b.m.calls, vec![(0, 64, PrefillPath::Tiled, None), (64, 1, PrefillPath::Tiled, Some(1))]);
}

#[test]
fn a_bad_request_is_dropped_and_answered_alone_and_the_others_run_as_they_would_alone() {
    let mut b = MiniCpmBackend::with_runtime(Fake::default(), 64);
    let good_a = req(0, 10, vec![5, 6, 7]);
    let good_b = req(3, 20, vec![8]);
    let alone_a = b.step(std::slice::from_ref(&good_a), 4);
    let alone_b = b.step(std::slice::from_ref(&good_b), 4);
    let is_nan = |rows: &Vec<RowOutput>| rows.iter().all(|r| r.values.iter().all(|v| v.is_nan()));
    // a slot that does not exist, rows past the context, an id outside the vocabulary, a request with no rows
    let bad = [req(9, 0, vec![1, 2]), req(1, 99, vec![1, 2]), req(1, 0, vec![1, VOCAB as u32]), req(2, 0, vec![])];
    for at in 0..=2 {
        for bad_one in &bad {
            let mut reqs = vec![good_a.clone(), good_b.clone()];
            reqs.insert(at, bad_one.clone());
            let out = b.step(&reqs, 4);
            assert_eq!(out.len(), 3);
            for (r, rows) in reqs.iter().zip(&out) {
                assert_eq!(rows.len(), r.tokens.len(), "one output row per input row, whatever happened to the request");
            }
            let (at_a, at_b) = if at == 0 {
                (1, 2)
            } else if at == 1 {
                (0, 2)
            } else {
                (0, 1)
            };
            assert_eq!(out[at_a].len(), alone_a[0].len());
            for (got, want) in out[at_a].iter().zip(&alone_a[0]) {
                assert!(got.bits_eq(want), "{bad_one:?} at {at}: a request beside a bad one changed");
            }
            for (got, want) in out[at_b].iter().zip(&alone_b[0]) {
                assert!(got.bits_eq(want), "{bad_one:?} at {at}: a request beside a bad one changed");
            }
            assert!(is_nan(&out[at]), "{bad_one:?} is answered with NaN rows");
        }
    }
    // a slot named twice: the later request of the pair is the bad one
    let out = b.step(&[good_a.clone(), req(0, 50, vec![1]), good_b.clone()], 4);
    assert!(
        out[0].iter().zip(&alone_a[0]).all(|(g, w)| g.bits_eq(w)) && out[2].iter().zip(&alone_b[0]).all(|(g, w)| g.bits_eq(w))
    );
    assert!(is_nan(&out[1]));
    assert!(b.take_error().is_none(), "a rejected request records no error: an error fails every sequence the call served");
}

#[test]
fn a_step_wider_than_the_window_keeps_the_requests_that_fit_in_order() {
    let mut b = MiniCpmBackend::with_runtime(Fake::default(), 64);
    let reqs = [req(0, 0, vec![1; 40]), req(1, 0, vec![2; 30]), req(2, 0, vec![3; 24]), req(3, 0, vec![4; 1])];
    let out = b.step(&reqs, 2);
    let nan = |rows: &Vec<RowOutput>| rows.iter().all(|r| r.values.iter().all(|v| v.is_nan()));
    // 40 + 24 = 64 rows run; the 30-row request does not fit behind the 40 and the later 1-row one does not fit behind 64
    assert!(!nan(&out[0]) && nan(&out[1]) && !nan(&out[2]) && nan(&out[3]));
    assert_eq!(b.m.windows.last().unwrap().len(), 64);
}

#[test]
fn errors_are_reported_once_and_fatal_errors_latch() {
    let mut b = MiniCpmBackend::with_runtime(Fake::default(), 64);
    b.m.fail = Some(ModelError::Request("bad input".into()));
    let out = b.step(&[req(0, 0, vec![3, 4])], 1);
    assert!(out[0].iter().all(|r| r.values[0].is_nan()), "every row of a failed launch is NaN");
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("bad input"));
    assert!(b.take_error().is_none());
    b.m.fail = Some(ModelError::StateLost("lost state".into()));
    b.step(&[req(0, 0, vec![3])], 1);
    let e = b.take_error().unwrap();
    assert!(e.fatal && e.message.contains("lost state"));
    // fatal: later calls answer NaN, commit and release do nothing, and the error stays
    let out = b.step(&[req(0, 0, vec![3])], 1);
    assert!(out[0][0].values[0].is_nan());
    b.commit(0, 1);
    b.release(0);
    assert!(b.m.committed.is_empty() && b.m.ended.is_empty());
    assert!(b.take_error().unwrap().fatal);
}

#[test]
fn a_failed_prefill_ends_its_slot_and_blames_only_itself() {
    let mut b = MiniCpmBackend::with_runtime(Fake::default(), 64);
    b.m.fail = Some(ModelError::Request("no pages".into()));
    let row = b.prefill(2, 0, &[3, 4, 5], 3);
    assert!(row.values.iter().all(|v| v.is_nan()));
    assert_eq!(b.m.ended, vec![2]);
    assert!(!b.take_error().unwrap().fatal);
    // the next prompt runs on a backend that is still alive
    let row = b.prefill(1, 0, &[3, 4, 5], 3);
    assert!(row.values.iter().all(|v| v.is_finite()));
}

#[test]
fn overflowing_prefill_start_is_refused_and_features_of_another_kind_are_refused_by_name() {
    let mut b = MiniCpmBackend::with_runtime(Fake::default(), 64);
    let row = b.prefill(1, u64::MAX, &[3], 1);
    assert!(row.values[0].is_nan());
    assert!(!b.take_error().unwrap().fatal);
    assert!(b.m.calls.is_empty());
    let span = gb_engine::ImageSpan { start: 0, tokens: 1, merged: (1, 1), features: Arc::new(()), key: 1 };
    let step = b.prefill_slice(0, &[1, 2, 3], &[span], 0, 1);
    assert!(step.row.unwrap().values[0].is_nan());
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("not MiniCPM-V-4.7") && e.message.contains("vision_minicpmv"), "{}", e.message);
    assert!(b.m.mm.is_empty() && b.m.calls.is_empty(), "nothing was launched for a span that cannot be spliced");
    assert_eq!(b.m.ended, vec![1, 0], "the failed prompt's slot is ended (after the overflowing start's)");
}

#[test]
fn commit_and_release_pass_through_and_the_model_has_no_native_drafter() {
    let mut b = MiniCpmBackend::with_runtime(Fake::default(), 64);
    b.commit(2, 5);
    b.release(2);
    assert_eq!(b.m.committed, vec![(2, 5)]);
    assert_eq!(b.m.ended, vec![2]);
    let sampling = gb_sampling::Sampling::default();
    let r = [gb_engine::ProposeRequest { slot: 1, tokens: &[3], sampling: &sampling, max: 4 }];
    assert_eq!(b.propose_batch(&r), vec![None], "the family has no MTP head and no draft until E4");
    assert_eq!(b.vocab_size(), VOCAB);
}

fn ask<'a>(slot: usize, tokens: &'a [u32], sampling: &'a gb_sampling::Sampling, max: usize) -> ProposeRequest<'a> {
    ProposeRequest { slot, tokens, sampling, max }
}

#[test]
fn without_a_draft_every_stream_gets_none_and_the_runtime_is_not_asked() {
    let mut b = MiniCpmBackend::with_runtime(Fake::default(), 64);
    let s = gb_sampling::Sampling::default();
    let out = b.propose_batch_scored(&[ask(0, &[1, 2, 3], &s, 7), ask(1, &[4, 5], &s, 7)]);
    assert!(out.iter().all(Option::is_none) && out.len() == 2);
    assert!(b.m.asks.is_empty(), "no draft: nothing reaches the runtime");
    assert!(b.take_error().is_none());
}

#[test]
fn the_drafts_confidences_become_the_proposals_accept_and_the_ask_reaches_the_runtime_whole() {
    let f = Fake {
        draft: Some(vec![
            Some(SpecProposal { tokens: vec![9, 8, 7, 6], accept: vec![0.9, 0.8, 0.7, 0.6] }),
            None, // a stream the draft cannot serve now
        ]),
        ..Fake::default()
    };
    let mut b = MiniCpmBackend::with_runtime(f, 64);
    let s = gb_sampling::Sampling { seed: 5, temperature: 0.7, top_k: 20, top_p: 0.95, min_p: 0.0 };
    let out = b.propose_batch_scored(&[ask(2, &[1, 2, 3], &s, 3), ask(1, &[4, 5], &s, 3)]);
    assert_eq!(b.m.asks, vec![vec![(2, 3, 3), (1, 5, 3)]]);
    let p = out[0].as_ref().unwrap();
    // cut to the ask, the confidences are the proposal's `accept` and nothing else of it is set
    assert_eq!((p.tokens.clone(), p.accept.clone()), (vec![9, 8, 7], vec![0.9, 0.8, 0.7]));
    assert!(p.conf.is_empty() && p.margin.is_empty() && p.lattice.is_none());
    assert!(out[1].is_none(), "the engine leaves that stream at plain decoding");
    assert!(b.take_error().is_none(), "a stream the draft declines is not an error");
}

#[test]
fn a_draft_error_costs_the_round_its_drafts_and_a_fatal_one_stops_the_backend() {
    let f = Fake {
        draft: Some(vec![Some(SpecProposal { tokens: vec![1], accept: vec![0.5] })]),
        propose_fails: Some(ModelError::Request("anything".into())),
        ..Fake::default()
    };
    let mut b = MiniCpmBackend::with_runtime(f, 64);
    let s = gb_sampling::Sampling::default();
    let out = b.propose_batch_scored(&[ask(0, &[1, 2], &s, 1)]);
    assert!(out[0].is_none());
    // a recoverable error is the engine's to consume right after the call (it only costs the round its drafts) ...
    let e = b.take_error().expect("recorded");
    assert!(!e.fatal);
    // ... and the next round drafts again
    let out = b.propose_batch_scored(&[ask(0, &[1, 2], &s, 1)]);
    assert!(out[0].is_some());
    // a fatal error is sticky
    b.m.propose_fails = Some(ModelError::StateLost("gone".into()));
    let out = b.propose_batch_scored(&[ask(0, &[1, 2], &s, 1)]);
    assert!(out[0].is_none());
    assert!(b.take_error().unwrap().fatal);
    let out = b.propose_batch_scored(&[ask(0, &[1, 2], &s, 1)]);
    assert!(out[0].is_none(), "a dead backend drafts nothing");
    assert_eq!(b.m.asks.len(), 3, "and is not asked");
}

// ---------------------------------------------------------------------------------------------------- images and video (E5b)

/// A request built from one of the E5a fixtures (the prompt of the Hugging Face processor's layout and `get_rope_index`'s positions):
/// the ids, the spans (one per frame, from the canvas rule's own frame ranges, trailing newline included, never overlapping), the
/// positions the reference gives every token and its rope delta.
struct FixtureRequest {
    name: String,
    ids: Vec<u32>,
    spans: Vec<gb_engine::ImageSpan>,
    want: Vec<[i32; 3]>,
    delta: i32,
}

fn fixture_requests() -> Vec<FixtureRequest> {
    let d = rope_fx::doc();
    let mut out = Vec::new();
    for case in d["cases"].as_array().unwrap() {
        let r = rope_fx::render(case, d);
        let mode = Downsample::parse(case["mode"].as_str().unwrap()).unwrap();
        let reference = canvas_positions(&CanvasInput {
            ids: &r.ids,
            mm_token_type_ids: &r.mm,
            image_sizes: &r.img,
            video_sizes: &r.vid,
            downsample: mode,
            markers: rope_fx::markers_of(d),
        })
        .unwrap();
        if reference.frames.is_empty() {
            continue;
        }
        // the frames' crops in appearance order: every modality keeps its own list
        let (mut img_at, mut vid_at) = (0usize, 0usize);
        let mut spans = Vec::new();
        let frames = &reference.frames;
        for (k, f) in frames.iter().enumerate() {
            let crops = 1 + f.slices.len();
            let video = f.modality == crate::mrope_canvas::Modality::Video;
            let sizes = if video { &r.vid } else { &r.img };
            let at = if video { &mut vid_at } else { &mut img_at };
            let Some(mine) = sizes.get(*at..*at + crops).map(<[_]>::to_vec) else { break };
            *at += crops;
            let pad = if video { VIDEO_PAD } else { IMAGE_PAD };
            // a span ends where the next frame begins (frames of a clip touch)
            let end = frames.get(k + 1).map_or(f.span.end, |n| f.span.end.min(n.span.start));
            let rows = r.ids[f.span.start..end].iter().filter(|&&t| t == pad).count();
            let layout = FrameLayout { rows, crops: mine, downsample: mode, slice_grid: f.grid };
            spans.push(gb_engine::ImageSpan {
                start: f.span.start,
                tokens: end - f.span.start,
                merged: (1, end - f.span.start),
                features: Arc::new(HostFeatures(layout)),
                key: 1 + k as u64,
            });
        }
        if spans.len() != frames.len() || img_at != r.img.len() || vid_at != r.vid.len() {
            continue; // a hand-made layout whose crops do not pair with its frames (the canvas rule's own tests cover it)
        }
        out.push(FixtureRequest {
            name: case["name"].as_str().unwrap().to_string(),
            ids: r.ids.clone(),
            spans,
            want: rope_fx::decode_positions(&rope_fx::b64(case["pos"].as_str().unwrap()), r.ids.len()),
            delta: case["delta"].as_i64().unwrap() as i32,
        });
    }
    out
}

/// Drives a whole prompt through `prefill_slice` and returns the backend (with the chunks its runtime saw) and the last step.
fn run_prompt(chunk: usize, req: &FixtureRequest, slot: usize) -> (MiniCpmBackend<Fake>, gb_engine::PrefillStep) {
    let mut b = MiniCpmBackend::with_runtime(Fake { rows_limit: Some(chunk), ..Fake::default() }, chunk);
    let mut done = 0;
    loop {
        let step = b.prefill_slice(slot, &req.ids, &req.spans, done, 4);
        assert!(step.done > done, "{}: no progress at {done}", req.name);
        done = step.done;
        if step.row.is_some() || done >= req.ids.len() {
            return (b, step);
        }
    }
}

fn pad_of(req: &FixtureRequest, k: usize) -> u32 {
    let s = &req.spans[k];
    if req.ids[s.start..s.start + s.tokens].contains(&VIDEO_PAD) {
        VIDEO_PAD
    } else {
        IMAGE_PAD
    }
}

#[test]
fn every_token_is_at_the_hugging_face_canvas_position_through_the_whole_request_path() {
    let reqs = fixture_requests();
    assert!(reqs.len() >= 100, "{} fixture prompts with a frame", reqs.len());
    let mut tokens = 0usize;
    for req in &reqs {
        for chunk in [13usize, 100, 512, 4096] {
            let (b, step) = run_prompt(chunk, req, 1);
            assert!(step.row.is_some(), "{}: the prompt did not complete", req.name);
            // what the runtime was told, row by row: explicit triples, else index + the slot's delta on all three streams
            let mut got: Vec<[i32; 3]> = Vec::new();
            for c in &b.m.mm {
                assert_eq!(c.first, got.len(), "{}: chunks are consecutive", req.name);
                match &c.rope3 {
                    Some(r3) => {
                        assert_eq!(r3.len(), 3 * c.rows);
                        got.extend(r3.chunks(3).map(|t| [t[0], t[1], t[2]]));
                    }
                    None => got.extend((0..c.rows).map(|i| {
                        let p = (c.first + i) as i32 + c.delta;
                        [p, p, p]
                    })),
                }
            }
            assert_eq!(got.len(), req.ids.len(), "{}", req.name);
            if let Some(i) = (0..got.len()).find(|&i| got[i] != req.want[i]) {
                panic!("{} chunk {chunk}: token {i} is at {:?}, the reference has {:?}", req.name, got[i], req.want[i]);
            }
            // decode goes on at index + delta, the reference's rope_deltas
            assert_eq!(b.m.delta[1], req.delta, "{} chunk {chunk}: the slot's delta after the prompt", req.name);
            // only the last chunk asks for candidates
            assert!(b.m.mm.iter().enumerate().all(|(i, c)| c.want.is_some() == (i + 1 == b.m.mm.len())), "{}", req.name);
            assert!(b.m.mm.iter().all(|c| c.path == PrefillPath::for_prompt(req.ids.len())));
            tokens += got.len();
        }
    }
    assert!(tokens > 100_000, "{tokens} tokens compared");
    eprintln!("{} fixture prompts with a frame, {tokens} token positions compared (4 chunk sizes each)", reqs.len());
}

/// What a prefix cut may do (E6's question): the prompt cut at the end of a span and prefilled alone has, for every token before the
/// cut, the position the whole prompt gives it. Cuts outside every visual group must (the canvas rule's claim); cuts between the
/// frames of a clip are inside a group by that rule, and are measured here.
#[test]
fn a_prefix_cut_at_a_span_boundary_leaves_the_positions_before_it_as_the_whole_prompt_has_them() {
    let (mut outside, mut inside_group, mut inside_equal) = (0usize, 0usize, 0usize);
    for req in fixture_requests() {
        let full = crate::families::minicpmv4_7::splice::Geometry::new(&req.ids, &req.spans).unwrap();
        for k in 0..req.spans.len() {
            let c = req.spans[k].start + req.spans[k].tokens;
            if c >= req.ids.len() {
                continue;
            }
            let prefix_spans: Vec<_> = req.spans[..=k].to_vec();
            let Ok(part) = crate::families::minicpmv4_7::splice::Geometry::new(&req.ids[..c], &prefix_spans) else { continue };
            let same = part.canvas.positions[..c] == full.canvas.positions[..c];
            if full.canvas.is_cut_outside_frames(c) {
                outside += 1;
                assert!(same, "{}: a cut at {c}, outside every group, moved the positions before it", req.name);
            } else {
                inside_group += 1;
                inside_equal += same as usize;
            }
        }
    }
    eprintln!("cuts at span ends: {outside} outside every group (positions equal), {inside_group} between frames of a clip ({inside_equal} equal)");
    assert!(outside > 50 && inside_group > 0);
}

#[test]
fn spliced_rows_land_on_pad_rows_only_in_feature_order_and_every_feature_row_is_used_once() {
    for req in fixture_requests().iter() {
        for chunk in [7usize, 64, 1000] {
            let (b, _) = run_prompt(chunk, req, 0);
            let addr = |i: usize| Arc::as_ptr(&req.spans[i].features) as *const () as usize;
            let mut used = vec![0usize; req.spans.len()];
            let mut covered = vec![false; req.ids.len()];
            for c in &b.m.mm {
                for &(a, dst, src, n) in &c.splice {
                    let k = (0..req.spans.len()).find(|&i| addr(i) == a).expect("the features of a span");
                    let pad = pad_of(req, k);
                    assert!(
                        req.ids[c.first + dst..c.first + dst + n].iter().all(|&t| t == pad),
                        "{}: a splice over a row that is not a pad of its span",
                        req.name
                    );
                    let s = &req.spans[k];
                    assert!(c.first + dst >= s.start && c.first + dst + n <= s.start + s.tokens);
                    assert_eq!(src, used[k], "{}: rows of span {k} out of order", req.name);
                    used[k] += n;
                    for (j, slot) in covered[c.first + dst..c.first + dst + n].iter_mut().enumerate() {
                        assert!(!*slot, "{}: row {} spliced twice", req.name, c.first + dst + j);
                        *slot = true;
                    }
                }
            }
            for (k, s) in req.spans.iter().enumerate() {
                let rows = crate::families::minicpmv4_7::splice::layout_of(&*s.features).unwrap().rows;
                assert_eq!(used[k], rows, "{}: span {k} used {} of {rows} feature rows", req.name, used[k]);
            }
            // every pad token of a span is covered, and no other token is
            for (p, &t) in req.ids.iter().enumerate() {
                let in_span = req.spans.iter().any(|s| s.start <= p && p < s.start + s.tokens);
                assert_eq!(covered[p], in_span && (t == IMAGE_PAD || t == VIDEO_PAD), "{} token {p}", req.name);
            }
        }
    }
}

#[test]
fn a_chunk_never_ends_inside_a_group_that_fits_it_and_text_chunks_carry_no_triples() {
    for req in fixture_requests().iter() {
        // the visual groups of the prompt, from the reference rule over the same spans
        let geo = crate::families::minicpmv4_7::splice::Geometry::new(&req.ids, &req.spans).unwrap();
        let groups = geo.canvas.groups.clone();
        for chunk in [5usize, 33, 64, 200, 700, 4096] {
            let (b, _) = run_prompt(chunk, req, 0);
            for c in &b.m.mm {
                let end = c.first + c.rows;
                for g in &groups {
                    if g.start < end && end < g.end {
                        assert!(
                            g.len() > chunk,
                            "{} chunk {chunk}: a chunk ends at {end} inside a group of {} that fits",
                            req.name,
                            g.len()
                        );
                    }
                }
                // a chunk with no frame token is plain text: no triples, the slot's delta carries its position
                let touches = (c.first..end).any(|p| groups.iter().any(|g| g.start <= p && p < g.end));
                assert_eq!(
                    c.rope3.is_some(),
                    touches,
                    "{} chunk at {}: triples exactly when a frame token is touched",
                    req.name,
                    c.first
                );
            }
        }
    }
}

#[test]
fn a_bad_image_prompt_fails_that_prefill_alone_and_the_next_prompt_runs() {
    let reqs = fixture_requests();
    let good = &reqs[0];
    let mut b = MiniCpmBackend::with_runtime(Fake::default(), 64);
    // a span whose features hold one row too few
    let mut bad = good.spans.clone();
    let l = crate::families::minicpmv4_7::splice::layout_of(&*bad[0].features).unwrap();
    bad[0].features = Arc::new(HostFeatures(FrameLayout { rows: l.rows - 1, ..l }));
    let step = b.prefill_slice(2, &good.ids, &bad, 0, 4);
    assert!(step.row.unwrap().values.iter().all(|v| v.is_nan()));
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("pad tokens"), "{}", e.message);
    assert_eq!(b.m.ended, vec![2], "the failed prompt's slot is ended");
    assert!(b.m.mm.is_empty(), "the failure was found before the first launch");
    // the backend is alive: the same prompt with its real spans prefills, and a text prompt does too
    let mut done = 0;
    let row = loop {
        let s = b.prefill_slice(1, &good.ids, &good.spans, done, 4);
        done = s.done;
        if let Some(r) = s.row {
            break r;
        }
    };
    assert!(row.values.iter().all(|v| v.is_finite()));
    assert!(b.take_error().is_none());
    let t = b.prefill_slice(0, &[3; 20], &[], 0, 4);
    assert!(t.row.unwrap().values.iter().all(|v| v.is_finite()));
    assert_eq!(b.m.delta[0], 0, "a text prompt starts with no rope delta");
}

// ------------------------------------------------------------------------------------------------------------ the prefix cache

#[test]
fn the_prefix_hooks_exist_only_with_an_arena_and_a_real_fingerprint() {
    let plain = MiniCpmBackend::with_runtime(Fake::default(), 64);
    assert!(plain.prefix_caps().is_none() && plain.prefix_path(5000).is_none());
    assert_eq!(plain.prefix_free_snapshots(), 0);
    let b = cached(64, 3);
    let caps = b.prefix_caps().unwrap();
    assert_eq!((caps.budget.snapshot_slots, caps.budget.snapshot_bytes), (3, 64_389_120));
    assert_eq!(b.prefix_free_snapshots(), 3);
    // the arithmetic of the prompt's total: the decode-GEMV arm is never cached, the pinned arm has one path id
    assert_eq!((b.prefix_path(128), b.prefix_path(129), b.prefix_path(200_000)), (None, Some(1), Some(1)));
    // a cache keyed by anything but a pack fingerprint is refused
    let c = real();
    let arena = SnapshotArena::bookkeeping_only(&c, prefix::budget_for_bytes(&c, 3 << 30)).unwrap();
    let e = MiniCpmBackend::with_runtime(Fake::default(), 64).with_prefix_arena(arena, "pack-a".into()).err().unwrap();
    assert!(e.to_string().contains("fingerprint"), "{e}");
    // another pack is another namespace: the engine's keys are bound to it
    let other = {
        let arena = SnapshotArena::bookkeeping_only(&c, prefix::budget_for_bytes(&c, 3 << 30)).unwrap();
        let fp = FINGERPRINT.replace("d03f", "d03e");
        MiniCpmBackend::with_runtime(Fake::default(), 64).with_prefix_arena(arena, fp).unwrap()
    };
    assert_ne!(b.prefix_caps().unwrap().namespace, other.prefix_caps().unwrap().namespace);
}

#[test]
fn the_drafts_identity_is_part_of_the_caches_namespace() {
    let c = real();
    let make = |identity: Option<&str>| {
        let arena = SnapshotArena::bookkeeping_only(&c, prefix::budget_for_bytes(&c, 3 << 30)).unwrap();
        let f = Fake { identity: identity.map(String::from), ..Fake::default() };
        MiniCpmBackend::with_runtime(f, 64).with_prefix_arena(arena, FINGERPRINT.into()).unwrap()
    };
    let ns = |b: &MiniCpmBackend<Fake>| b.prefix_caps().unwrap().namespace;
    let none = ns(&make(None));
    let a = ns(&make(Some("dspark|pack=sha256:aaaa|taps=[3, 11, 19, 27, 35]|kv=bf16|page=2621440")));
    let b = ns(&make(Some("dspark|pack=sha256:bbbb|taps=[3, 11, 19, 27, 35]|kv=bf16|page=2621440")));
    let taps = ns(&make(Some("dspark|pack=sha256:aaaa|taps=[2, 11, 19, 27, 35]|kv=bf16|page=2621440")));
    assert_eq!(none, ns(&make(None)), "deterministic");
    for (what, other) in [("no draft against a draft", a), ("another draft pack", b), ("other tap layers", taps)] {
        assert_ne!(none, other, "{what}");
    }
    assert_ne!(a, b, "two draft packs are two namespaces");
    assert_ne!(a, taps, "the tap layers are part of the draft's identity");
}

#[test]
fn a_cut_ends_its_slice_and_hands_the_snapshot_to_the_engine() {
    let mut b = cached(64, 2);
    let prompt = vec![3u32; 300];
    // cuts at 100 (inside the second chunk) and 200: the slices end there, each takes one snapshot
    let s1 = b.prefill_slice_cuts(1, &prompt, &[], 0, 4, &[100, 200]);
    assert_eq!(s1.step.done, 64);
    let s2 = b.prefill_slice_cuts(1, &prompt, &[], 64, 4, &[100, 200]);
    assert_eq!((s2.step.done, s2.captured.len()), (100, 1));
    assert_eq!(s2.captured[0].len, 100);
    assert_eq!(s2.captured[0].handle.pages.len(), 1, "a cut inside the first page holds one page, partly filled");
    let s3 = b.prefill_slice_cuts(1, &prompt, &[], 100, 4, &[100, 200]);
    assert_eq!(s3.step.done, 128, "the resumed slice ends at the next absolute chunk boundary");
    assert_eq!(b.prefix_free_snapshots(), 1);
    assert_eq!(b.m.begun, vec![1], "only the slice that starts the prompt begins the slot");
    // a final slice carries the candidates; the engine owns the handles and gives them back once each
    let mut done = 128;
    let mut later = Vec::new();
    let last = loop {
        let s = b.prefill_slice_cuts(1, &prompt, &[], done, 4, &[100, 200]);
        done = s.step.done;
        later.extend(s.captured);
        if let Some(row) = s.step.row {
            break row;
        }
    };
    assert!(last.values.iter().all(|v| v.is_finite()));
    assert_eq!(later.iter().map(|c| c.len).collect::<Vec<_>>(), vec![200], "the second cut was taken once, at its place");
    assert_eq!(b.prefix_free_snapshots(), 0);
    b.prefix_release(s2.captured[0].handle.clone());
    b.prefix_release(later.remove(0).handle);
    assert_eq!(b.prefix_free_snapshots(), 2);
    // a prompt on the GEMV arm takes no cut at all, and a plain slice is the same routine with none
    let short = vec![3u32; 100];
    let g = b.prefill_slice_cuts(2, &short, &[], 0, 4, &[50]);
    assert!(g.captured.is_empty() && g.step.done == 64);
    let before = b.m.slices.len();
    b.prefill_slice(3, &prompt, &[], 0, 4);
    assert_eq!(b.m.slices.len(), before + 1, "with a cache every slice runs through the one routine");
    assert!(b.take_error().is_none());
}

#[test]
fn a_failed_restore_ends_only_its_slot_and_a_fatal_one_latches() {
    let mut b = cached(64, 2);
    let prompt = vec![3u32; 300];
    let mut done = 0;
    let cap = loop {
        let s = b.prefill_slice_cuts(0, &prompt, &[], done, 4, &[100]);
        done = s.step.done;
        if let Some(c) = s.captured.into_iter().next() {
            break c;
        }
    };
    // a length the handle was not taken at: refused, the slot is ended, the others are not touched, no error is recorded (the
    // engine drops the entry and prefills cold)
    assert!(b.prefix_restore(2, &cap.handle, 99).is_err());
    assert_eq!(b.m.ended, vec![2]);
    assert!(b.take_error().is_none());
    // a handle that was already released
    b.prefix_release(cap.handle.clone());
    assert!(b.prefix_restore(3, &cap.handle, 100).is_err());
    assert_eq!(b.m.ended, vec![2, 3]);
    // a second release of it is recorded, and frees nothing
    b.prefix_release(cap.handle.clone());
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("already released"), "{}", e.message);
    assert_eq!(b.prefix_free_snapshots(), 2);
    // a state-losing restore is the engine's fatal error
    let mut b2 = cached(64, 2);
    let mut at = 0;
    let c2 = loop {
        let s = b2.prefill_slice_cuts(0, &prompt, &[], at, 4, &[100]);
        at = s.step.done;
        if let Some(c) = s.captured.into_iter().next() {
            break c;
        }
    };
    b2.m.fail = Some(ModelError::StateLost("device lost".into()));
    assert!(b2.prefix_restore(1, &c2.handle, 100).is_err());
    assert!(b2.take_error().unwrap().fatal);
}

#[test]
fn a_restored_slot_only_continues_on_the_arithmetic_its_snapshot_was_made_under() {
    let mut b = cached(64, 2);
    let long = vec![3u32; 300];
    let mut done = 0;
    let cap = loop {
        let s = b.prefill_slice_cuts(0, &long, &[], done, 4, &[100]);
        done = s.step.done;
        if let Some(c) = s.captured.into_iter().next() {
            break c;
        }
    };
    // the prompt that shares the 100 tokens but is 120 long takes the decode GEMV: the pinned-GEMM snapshot must not continue it
    let short: Vec<u32> = long[..120].to_vec();
    b.prefix_restore(1, &cap.handle, 100).unwrap();
    let step = b.prefill_slice(1, &short, &[], 100, 4);
    assert!(step.row.unwrap().values.iter().all(|v| v.is_nan()), "refused: NaN row, the sequence fails alone");
    let e = b.take_error().unwrap();
    assert!(!e.fatal && e.message.contains("other arithmetic") && e.message.contains("120 tokens"), "{}", e.message);
    assert_eq!(b.m.ended.last(), Some(&1), "the slot is ended");
    // the same snapshot continues a prompt that takes the same arithmetic, through either slice entry point
    let longer: Vec<u32> = long[..200].to_vec();
    b.prefix_restore(1, &cap.handle, 100).unwrap();
    let step = b.prefill_slice_cuts(1, &longer, &[], 100, 4, &[]);
    assert_eq!(step.step.done, 128);
    assert!(b.take_error().is_none());
    // a restore resumed anywhere but at its cut is refused, and a prompt that starts over forgets the restore
    b.prefix_restore(2, &cap.handle, 100).unwrap();
    let step = b.prefill_slice(2, &longer, &[], 64, 4);
    assert!(step.row.unwrap().values.iter().all(|v| v.is_nan()));
    assert!(b.take_error().unwrap().message.contains("resumes it at 64"));
    b.prefix_restore(2, &cap.handle, 100).unwrap();
    let step = b.prefill_slice(2, &short, &[], 0, 4);
    assert_eq!(step.done, 64, "a prompt from 0 begins the slot over and is on its own arithmetic");
    assert!(b.take_error().is_none());
    // release forgets a pending restore
    b.prefix_restore(3, &cap.handle, 100).unwrap();
    b.release(3);
    let step = b.prefill_slice(3, &short, &[], 100, 4);
    assert!(step.row.is_some() && b.take_error().is_none(), "no restore is pending, so the slice is judged as any other");
}

#[test]
fn features_of_another_kind_are_refused_by_name_on_the_cached_route_too() {
    let mut b = cached(64, 2);
    let span = gb_engine::ImageSpan { start: 0, tokens: 1, merged: (1, 1), features: std::sync::Arc::new(()), key: 1 };
    let step = b.prefill_slice_cuts(0, &[1; 200], std::slice::from_ref(&span), 0, 1, &[100]);
    assert!(step.step.row.unwrap().values[0].is_nan() && step.captured.is_empty());
    assert!(b.take_error().unwrap().message.contains("vision_minicpmv"));
    let step = b.prefill_slice(0, &[1; 200], &[span], 0, 1);
    assert!(step.row.unwrap().values[0].is_nan());
    assert!(b.m.slices.is_empty(), "nothing reached the model");
}

#[test]
fn a_snapshot_of_one_prefill_moe_arm_is_not_found_and_not_continued_under_the_other() {
    let mut b = cached(64, 2);
    let long = vec![3u32; 300];
    let mut done = 0;
    let cap = loop {
        let s = b.prefill_slice_cuts(0, &long, &[], done, 4, &[100]);
        done = s.step.done;
        if let Some(c) = s.captured.into_iter().next() {
            break c;
        }
    };
    let arena_meta = |b: &MiniCpmBackend<Fake>| b.prefix_arena().unwrap().meta(cap.handle.snap).unwrap();
    assert_eq!(arena_meta(&b).path, prefix::PATH_ID, "the default arm's snapshot carries the grouped path");
    let (path_before, namespace_before) = (b.prefix_path(300), b.prefix_caps().unwrap().namespace);
    // the same pack under E3's CoopMK prefill MoE: another path id (the index never looks the entry up) and another namespace
    b.m.coopmk = true;
    assert_eq!((path_before, b.prefix_path(300)), (Some(prefix::PATH_ID), Some(prefix::PATH_ID_COOPMK)));
    assert_ne!(namespace_before, b.prefix_caps().unwrap().namespace);
    // and if a caller restores it anyway, the first slice refuses to continue it, by name
    b.prefix_restore(1, &cap.handle, 100).unwrap();
    let step = b.prefill_slice(1, &long, &[], 100, 4);
    assert!(step.row.unwrap().values.iter().all(|v| v.is_nan()));
    let e = b.take_error().unwrap();
    assert!(e.message.contains("other arithmetic") && e.message.contains("grouped"), "{}", e.message);
    // back on its own arm it continues
    b.m.coopmk = false;
    b.prefix_restore(1, &cap.handle, 100).unwrap();
    let step = b.prefill_slice(1, &long, &[], 100, 4);
    assert_eq!(step.done, 128);
    assert!(b.take_error().is_none());
}

// ------------------------------------------------------------------------------------- images under the prefix cache (E5b)

/// One token as the runtime was told it: its position triple and, for a pad row, `(address of the span's features, row)`.
type Told = ([i32; 3], Option<(usize, usize)>);

/// What the runtime was told for each token its chunks covered: the position triple (explicit, or index + the slot's delta) and, for a
/// pad row, the feature row it took.
fn effective(calls: &[MmCall], len: usize) -> Vec<Told> {
    let mut out = vec![([i32::MIN; 3], None); len];
    for c in calls {
        for i in 0..c.rows {
            out[c.first + i].0 = match &c.rope3 {
                Some(r3) => [r3[3 * i], r3[3 * i + 1], r3[3 * i + 2]],
                None => {
                    let p = (c.first + i) as i32 + c.delta;
                    [p, p, p]
                }
            };
        }
        for &(feat, dst, src, n) in &c.splice {
            for j in 0..n {
                out[c.first + dst + j].1 = Some((feat, src + j));
            }
        }
    }
    out
}

/// The fixture prompts a prefix cache can serve (over 160 tokens: the pinned-GEMM arm) whose first visual group has text before and
/// after it, with that group.
fn cacheable() -> Vec<(FixtureRequest, std::ops::Range<usize>)> {
    fixture_requests()
        .into_iter()
        .filter_map(|r| {
            if r.ids.len() <= 160 {
                return None;
            }
            let g = crate::families::minicpmv4_7::splice::Geometry::new(&r.ids, &r.spans).ok()?.canvas.groups.first()?.clone();
            (g.start > 2 && g.end + 2 < r.ids.len()).then_some((r, g))
        })
        .collect()
}

/// Slices from `done` to the end of the prompt (or to the first snapshot with `stop_at_capture`): the slice ends and what was taken.
fn drive_cuts(
    b: &mut MiniCpmBackend<Fake>,
    slot: usize,
    req: &FixtureRequest,
    mut done: usize,
    cuts: &[usize],
    stop_at_capture: bool,
) -> (Vec<usize>, Vec<Captured>) {
    let (mut ends, mut caps) = (Vec::new(), Vec::new());
    loop {
        let r = b.prefill_slice_cuts(slot, &req.ids, &req.spans, done, 4, cuts);
        assert!(r.step.done > done, "{}: no progress at {done}", req.name);
        done = r.step.done;
        ends.push(done);
        let took = !r.captured.is_empty();
        caps.extend(r.captured);
        if r.step.row.is_some() || done >= req.ids.len() || (stop_at_capture && took) {
            return (ends, caps);
        }
    }
}

fn pad_rows(calls: &[MmCall], len: usize) -> Vec<Option<usize>> {
    effective(calls, len).iter().map(|x| x.1.map(|f| f.1)).collect()
}

#[test]
fn a_cut_inside_a_frame_is_never_taken_and_a_cut_outside_ends_its_slice_and_snapshots() {
    let reqs = cacheable();
    assert!(reqs.len() > 20, "{} cacheable fixture prompts", reqs.len());
    for (req, g) in &reqs {
        // a cut before the group, one strictly inside it, one at its end and one after it: the engine would have moved the inside one
        // to the group's start already, the backend refuses it anyway
        let inside = g.start + 1;
        let after = (g.end + 1).min(req.ids.len() - 1);
        let planned = [g.start, inside, g.end, after];
        let mut b = cached(64, 8);
        let (ends, caps) = drive_cuts(&mut b, 0, req, 0, &planned, false);
        let lens: Vec<usize> = caps.iter().map(|c| c.len).collect();
        assert_eq!(lens, vec![g.start, g.end, after], "{}: the snapshots taken (cuts {planned:?}, group {g:?})", req.name);
        assert!(ends.contains(&g.start) && ends.contains(&g.end), "{}: a taken cut ends its slice: {ends:?}", req.name);
        assert!(!ends.contains(&inside), "{}: a slice ended inside the frame group", req.name);
        // the prompt came out as without a cache: every token at the reference's position, every pad on its feature row
        let got = effective(&b.m.mm, req.ids.len());
        for (i, (pos, _)) in got.iter().enumerate() {
            assert_eq!(*pos, req.want[i], "{}: token {i} with cuts", req.name);
        }
        let (plain, _) = run_prompt(64, req, 0);
        assert_eq!(
            pad_rows(&b.m.mm, req.ids.len()),
            pad_rows(&plain.m.mm, req.ids.len()),
            "{}: feature rows with cuts",
            req.name
        );
        assert_eq!(b.m.delta[0], req.delta, "{}: the slot's delta after the prompt", req.name);
    }
}

#[test]
fn a_prompt_continued_after_a_restore_is_prefilled_as_the_uncached_prompt_from_the_cut_on() {
    let reqs = cacheable();
    for (req, g) in &reqs {
        // the cold run stops right after the snapshot at the end of the frame group (the cut a following turn would hit)
        let mut b = cached(64, 8);
        let (_, caps) = drive_cuts(&mut b, 0, req, 0, &[g.end], true);
        assert_eq!(caps.len(), 1, "{}", req.name);
        let cap = &caps[0];
        assert_eq!(cap.len, g.end);
        let cold_calls = b.m.mm.len();
        // another slot restores it and continues: the slice that follows starts at the cut
        b.prefix_restore(1, &cap.handle, cap.len).unwrap();
        // the snapshot names the offset the text after the cut has: the reference's position of the token at the cut, less its index
        assert_eq!(
            b.m.delta[1],
            req.want[cap.len][0] - cap.len as i32,
            "{}: the restored slot's delta at the cut {}",
            req.name,
            cap.len
        );
        let (_, more) = drive_cuts(&mut b, 1, req, cap.len, &[], false);
        assert!(more.is_empty());
        assert_eq!(b.m.mm[cold_calls].first, cap.len, "{}: the first slice after the restore", req.name);
        // together the two runs told the runtime exactly what the uncached prompt is told: positions, pad rows and the delta of decode
        let got = effective(&b.m.mm, req.ids.len());
        for (i, (pos, _)) in got.iter().enumerate() {
            assert_eq!(*pos, req.want[i], "{}: token {i} after a restore at {}", req.name, cap.len);
        }
        let (plain, _) = run_prompt(64, req, 0);
        assert_eq!(pad_rows(&b.m.mm, req.ids.len()), pad_rows(&plain.m.mm, req.ids.len()), "{}", req.name);
        assert_eq!(b.m.delta[1], req.delta, "{}: decode after a restore runs at index + the canvas's delta", req.name);
        assert!(b.take_error().is_none());
    }
}

#[test]
fn the_namespace_carries_the_image_preprocessing_and_never_matches_a_text_engine() {
    let text = cached(64, 3);
    let a = cached(64, 3).with_image_preprocessing("minicpmv4.7-vision-v1:torch:s448p14m9:slice:16x".into());
    let a2 = cached(64, 3).with_image_preprocessing("minicpmv4.7-vision-v1:torch:s448p14m9:slice:16x".into());
    let slices = cached(64, 3).with_image_preprocessing("minicpmv4.7-vision-v1:torch:s448p14m36:slice:16x".into());
    let mode = cached(64, 3).with_image_preprocessing("minicpmv4.7-vision-v1:torch:s448p14m9:slice:4x".into());
    let ns = |b: &MiniCpmBackend<Fake>| b.prefix_caps().unwrap().namespace;
    assert_eq!(ns(&a), ns(&a2), "the same preprocessing is one namespace");
    for (name, other) in [("a text engine", &text), ("another slice limit", &slices), ("another downsample mode", &mode)] {
        assert_ne!(ns(&a), ns(other), "an engine with images shares its entries with {name}");
    }
}

// ------------------------------------------------------------------------------------------- the engine's row limit (drain, timeouts)

/// Drives `prompt` through `prefill_slice` with the engine's row limit `limit_at(done)` set before each slice, as `Engine::step` does
/// when a deadline comes near and goes: the slice ends and the last step.
fn drive_limited(
    b: &mut MiniCpmBackend<Fake>,
    slot: usize,
    prompt: &[u32],
    spans: &[gb_engine::ImageSpan],
    limit_at: impl Fn(usize) -> Option<usize>,
) -> (Vec<usize>, gb_engine::PrefillStep) {
    let mut ends = Vec::new();
    let mut done = 0;
    loop {
        assert!(b.set_prefill_row_limit(limit_at(done)), "MiniCPM-V-4.7 supports the row limit");
        let step = b.prefill_slice(slot, prompt, spans, done, 4);
        b.set_prefill_row_limit(None);
        assert!(step.done > done, "no progress at {done}");
        done = step.done;
        ends.push(done);
        if step.row.is_some() || done >= prompt.len() {
            return (ends, step);
        }
    }
}

#[test]
fn the_row_limit_puts_text_slices_on_its_absolute_grid_and_keeps_the_prompt_path() {
    // chunk 4096, a 300-token prompt: one slice without the limit, three on the 128 grid with it, the same path and the same last row
    let prompt = vec![3u32; 300];
    let mut plain = MiniCpmBackend::with_runtime(Fake::default(), 4096);
    let (ends0, last0) = drive_limited(&mut plain, 1, &prompt, &[], |_| None);
    assert_eq!(ends0, vec![300]);
    let mut lim = MiniCpmBackend::with_runtime(Fake::default(), 4096);
    let (ends, last) = drive_limited(&mut lim, 1, &prompt, &[], |_| Some(128));
    assert_eq!(ends, vec![128, 256, 300]);
    assert_eq!(
        lim.m.calls,
        vec![(0, 128, PrefillPath::Tiled, None), (128, 128, PrefillPath::Tiled, None), (256, 44, PrefillPath::Tiled, Some(4))],
        "the path is the prompt's (keyed on its total), only the last slice asks for candidates"
    );
    assert!(last.row.as_ref().unwrap().bits_eq(last0.row.as_ref().unwrap()));
    assert_eq!(lim.m.begun, vec![1]);
    // the limit comes on mid-prompt and goes again: the slices stay on the absolute grids (256, then 128, then 256 again)
    let prompt = vec![5u32; 700];
    let mut b = MiniCpmBackend::with_runtime(Fake::default(), 256);
    let (ends, _) = drive_limited(&mut b, 0, &prompt, &[], |done| (256..512).contains(&done).then_some(128));
    assert_eq!(ends, vec![256, 384, 512, 700]);
    // a smaller limit than the engine's, and a zero one, never panic: the grid is at least one row
    let mut b = MiniCpmBackend::with_runtime(Fake::default(), 64);
    let (ends, _) = drive_limited(&mut b, 0, &[5u32; 3], &[], |_| Some(0));
    assert_eq!(ends, vec![1, 2, 3]);
    // a limit above the chunk changes nothing
    let mut b = MiniCpmBackend::with_runtime(Fake::default(), 64);
    let (ends, _) = drive_limited(&mut b, 0, &vec![5u32; 130], &[], |_| Some(128));
    assert_eq!(ends, vec![64, 128, 130]);
    assert!(b.take_error().is_none());
}

#[test]
fn the_row_limit_reaches_the_prefix_cache_slices() {
    // with an arena every slice goes through the cached routine: the limit must reach it, and a cut still ends its slice
    let prompt = vec![3u32; 300];
    let mut b = cached(4096, 2);
    let (ends, last) = drive_limited(&mut b, 1, &prompt, &[], |_| Some(128));
    assert_eq!(ends, vec![128, 256, 300]);
    assert!(last.row.is_some());
    let mut b = cached(4096, 2);
    assert!(b.set_prefill_row_limit(Some(128)));
    let s1 = b.prefill_slice_cuts(1, &prompt, &[], 0, 4, &[100, 200]);
    assert_eq!((s1.step.done, s1.captured.len()), (100, 1), "a cut inside the limited slice ends it");
    let s2 = b.prefill_slice_cuts(1, &prompt, &[], 100, 4, &[100, 200]);
    assert_eq!(s2.step.done, 128, "the resumed slice ends on the limit's absolute grid");
    b.set_prefill_row_limit(None);
    let s3 = b.prefill_slice_cuts(1, &prompt, &[], 128, 4, &[100, 200]);
    assert_eq!((s3.step.done, s3.captured.len()), (200, 1));
    for c in s1.captured.into_iter().chain(s3.captured) {
        b.prefix_release(c.handle);
    }
    assert!(b.take_error().is_none());
}

#[test]
fn under_the_row_limit_an_image_prompt_is_cut_as_on_a_128_grid_and_every_token_keeps_its_position() {
    // the limited run must feed the runtime exactly what a backend built with 128-row chunks feeds it (positions, triples, splices,
    // the slot's delta), and every token must end at the position the unlimited run gives it
    let reqs = fixture_requests();
    let mut compared = 0usize;
    for req in &reqs {
        let mk = |chunk| MiniCpmBackend::with_runtime(Fake { rows_limit: Some(4096), ..Fake::default() }, chunk);
        let mut whole = mk(4096);
        let (_, w) = drive_limited(&mut whole, 1, &req.ids, &req.spans, |_| None);
        let mut grid = mk(128);
        let (ends_grid, g) = drive_limited(&mut grid, 1, &req.ids, &req.spans, |_| None);
        let mut lim = mk(4096);
        let (ends, l) = drive_limited(&mut lim, 1, &req.ids, &req.spans, |_| Some(128));
        assert_eq!(ends, ends_grid, "{}: the limited slices are the 128-row backend's", req.name);
        let key = |c: &MmCall| (c.first, c.rows, c.rope3.clone(), c.delta, c.splice.clone(), c.want, c.path);
        assert_eq!(
            lim.m.mm.iter().map(key).collect::<Vec<_>>(),
            grid.m.mm.iter().map(key).collect::<Vec<_>>(),
            "{}: the runtime was told the same",
            req.name
        );
        let n = req.ids.len();
        assert_eq!(effective(&lim.m.mm, n), effective(&whole.m.mm, n), "{}: a token moved", req.name);
        assert_eq!(lim.m.delta[1], whole.m.delta[1], "{}: the slot's delta after the prompt", req.name);
        assert!(l.row.unwrap().bits_eq(&w.row.unwrap()) && g.row.is_some());
        compared += n;
    }
    eprintln!(
        "{} fixture prompts, {compared} token positions: the row limit feeds the runtime what a 128-row grid does",
        reqs.len()
    );
}
