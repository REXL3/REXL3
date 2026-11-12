//! The engine's MiniCPM-V-4.7 backend (text only). The model owns all device state; this layer chooses the prompt-wide arithmetic,
//! slices a prefill on an absolute grid, isolates a bad request from the others of its step, and translates errors.
//!
//! What is wired and what is not (E3, E5b, E6b): the serial forward, prefill slices and verify windows of up to 16 rows a stream (64 a
//! step), and image and video prompts (E5b: the spans' feature rows are spliced over the pad tokens of each prefill chunk, positions
//! are the canvas of the whole prompt, `splice.rs`). `has_mtp` is false and no native drafter exists (`propose*` keep the trait's
//! `None`: the copy drafter of the engine core is the only speculation until E4). The prefix cache (E6b, `prefix.rs`) is offered
//! when the engine hands the backend its snapshot arena ([`MiniCpmBackend::with_prefix_arena`]): `prefix_caps`, `prefix_path`,
//! `prefix_free_snapshots`, `prefix_restore`, `prefix_release` and `prefill_slice_cuts`, with the same slicing routine for cached and
//! uncached prefill, and the guard that a restored slot only continues into a prompt of the arithmetic its snapshot was made under.
//! Image and video prompts are cached too (E5b: the engine's index keys carry each image's content key, the namespace the
//! preprocessing's identity, and a cut is only taken outside every visual group)
//!
//! **Failure isolation.** A step is validated per request on the host before anything is launched: a slot that does not exist, rows
//! past the context, an id outside the vocabulary, a slot named twice, a window wider than 64 rows or a pool without a page for it. A
//! request that fails is dropped from the batch and answered with NaN rows (the engine fails that sequence alone through its
//! sampler); the others run as if it had never been asked, which is bit-for-bit what they would have got alone (the forward is row
//! invariant), and no backend error is recorded, because a recorded error fails every sequence the call served. An error of the
//! launch itself (a device error) is not one request's: it is recorded and every row of the call is NaN.

use super::model::{MiniCpmModel, WINDOW_ROWS};
use super::prefix::{self, SliceRequest, SliceResult, SnapshotArena};
use super::spec::{SpecAsk, SpecProposal};
use super::splice::Geometry;
use super::vision::MiniCpmFeatures;
use crate::error::ModelError;
use crate::forward::{Candidates, PrefillPath, SpliceOp};
use crate::rows::RowBatch;
use crate::serve::feature_ref;
use gb_engine::prefix::{PrefixCaps, PrefixHandle};
use gb_engine::{Backend, BackendError, CutStep, ImageSpan, PrefillStep, Proposal, ProposeRequest, RowOutput, RowRequest};
use std::collections::{BTreeMap, BTreeSet};

/// One run of a chunk's rows that take feature rows instead of token embeddings: rows `src_row..src_row + n` of `features` (the span's
/// encoder output, whatever type the runtime's encoder made) replace chunk rows `dst_row..dst_row + n`.
pub struct ChunkSplice<'a> {
    pub features: &'a (dyn std::any::Any + Send + Sync),
    pub dst_row: usize,
    pub src_row: usize,
    pub n: usize,
}

/// What the backend needs of a model, so that its host scheduling contract (slicing, path choice, isolation, error latching) can be
/// run against a fake without a GPU or a pack.
pub trait MiniCpmRuntime {
    fn vocab_size(&self) -> usize;
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError>;
    fn end_slot(&mut self, slot: usize);
    /// Whether `n` rows of `slot` from position `first` can run (the slot exists, the context and the pool have room); takes the pages.
    fn reserve(&mut self, slot: usize, first: usize, n: usize) -> Result<(), ModelError>;
    fn prefill_chunk(&mut self, b: &RowBatch, path: PrefillPath, want: Option<usize>) -> Result<Option<Candidates>, ModelError>;
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError>;
    fn commit(&mut self, slot: usize, keep: usize);
    fn flush(&mut self) -> Result<(), ModelError>;
    fn quiesce(&mut self) -> Result<(), ModelError> {
        Ok(())
    }
    /// Whether a DSpark draft is attached (E4b): without one `propose_batch_scored` answers `None` for every stream, as in E3.
    fn has_draft(&self) -> bool {
        false
    }
    /// The attached draft's identity (the prefix cache's namespace and a snapshot's draft context are keyed by it); `None` without one.
    fn draft_identity(&self) -> Option<String> {
        None
    }
    /// Drafts for the streams of a round (E4b); `None` for a stream that cannot draft now. Only a fatal error is an `Err`.
    fn propose_round(&mut self, asks: &[SpecAsk<'_>]) -> Result<Vec<Option<SpecProposal>>, ModelError> {
        Ok(vec![None; asks.len()])
    }
    /// [`MiniCpmRuntime::prefill_chunk`] with feature rows spliced over the embedded chunk (E5b); `b.rope3` may carry the chunk's
    /// explicit triples. The default refuses a splice by naming the tower.
    fn prefill_chunk_mm(
        &mut self,
        b: &RowBatch,
        path: PrefillPath,
        want: Option<usize>,
        splice: &[ChunkSplice<'_>],
    ) -> Result<Option<Candidates>, ModelError> {
        if !splice.is_empty() || b.rope3.is_some() {
            return Err(ModelError::NotBuilt("this MiniCPM-V-4.7 runtime has no vision_minicpmv splice".into()));
        }
        self.prefill_chunk(b, path, want)
    }
    /// The rope offset of every later text token of `slot` (`rope position - cache index`, all three streams).
    fn set_rope_delta(&mut self, _slot: usize, delta: i32) -> Result<(), ModelError> {
        if delta == 0 {
            Ok(())
        } else {
            Err(ModelError::NotBuilt("this MiniCPM-V-4.7 runtime has no rope delta".into()))
        }
    }
    /// The most rows one prefill chunk may take.
    fn prefill_rows(&self) -> usize {
        usize::MAX
    }
    /// Whether prefill chunks run the grouped prefill MoE (the default arm) or E3's CoopMK: the cache's path ids and namespace carry it.
    fn prefill_moe_grouped(&self) -> bool {
        true
    }
    /// One prefill slice that also snapshots at `req.cuts` (`prefix::slice_with_cuts`); the host tests' fake answers it.
    fn cached_slice(&mut self, _arena: &mut SnapshotArena, _req: SliceRequest<'_>) -> Result<SliceResult, ModelError> {
        Err(ModelError::Config("prefix snapshots need the device model".into()))
    }
    /// Start `slot` over from a snapshot (`prefix::restore`); on error the slot is ended.
    fn restore_snapshot(
        &mut self,
        _arena: &SnapshotArena,
        _slot: usize,
        _handle: &PrefixHandle,
        _len: usize,
    ) -> Result<(), ModelError> {
        Err(ModelError::Config("prefix snapshots need the device model".into()))
    }
    /// Give back a snapshot slot and its page references (`prefix::release`), exactly once per handle.
    fn release_snapshot(&mut self, _arena: &mut SnapshotArena, _handle: PrefixHandle) -> Result<(), ModelError> {
        Err(ModelError::Config("prefix snapshots need the device model".into()))
    }
    /// Snapshot `slot`'s state after `len` prefill rows (`prefix::capture`): what a multimodal slice does at a cut, where the chunk is
    /// not the text routine's. `Ok(None)`: no snapshot slot was free.
    fn capture_snapshot(
        &mut self,
        _arena: &mut SnapshotArena,
        _slot: usize,
        _len: usize,
    ) -> Result<Option<gb_engine::prefix::Captured>, ModelError> {
        Err(ModelError::Config("prefix snapshots need the device model".into()))
    }
}

impl MiniCpmRuntime for MiniCpmModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.begin_slot(slot)
    }
    fn end_slot(&mut self, slot: usize) {
        self.end_slot(slot)
    }
    fn reserve(&mut self, slot: usize, first: usize, n: usize) -> Result<(), ModelError> {
        self.reserve(slot, first, n)
    }
    fn prefill_chunk(&mut self, b: &RowBatch, path: PrefillPath, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        self.prefill_chunk(b, path, want)
    }
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        self.window_candidates(b, count)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.commit(slot, keep)
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        self.flush()
    }
    fn quiesce(&mut self) -> Result<(), ModelError> {
        self.stream.sync()?;
        Ok(())
    }
    fn has_draft(&self) -> bool {
        self.has_draft()
    }
    fn draft_identity(&self) -> Option<String> {
        MiniCpmModel::draft_identity(self).map(String::from)
    }
    fn propose_round(&mut self, asks: &[SpecAsk<'_>]) -> Result<Vec<Option<SpecProposal>>, ModelError> {
        self.propose_round(asks)
    }
    fn prefill_chunk_mm(
        &mut self,
        b: &RowBatch,
        path: PrefillPath,
        want: Option<usize>,
        splice: &[ChunkSplice<'_>],
    ) -> Result<Option<Candidates>, ModelError> {
        let mut ops = Vec::with_capacity(splice.len());
        for s in splice {
            let f = feature_ref::<MiniCpmFeatures>(s.features)
                .ok_or_else(|| ModelError::Request("an image span's features are not MiniCPM-V-4.7 projector rows".into()))?;
            ops.push(SpliceOp { feats: &f.buf, dst_row: s.dst_row, src_row: s.src_row, n: s.n });
        }
        self.prefill_chunk_spliced(b, path, want, &ops)
    }
    fn set_rope_delta(&mut self, slot: usize, delta: i32) -> Result<(), ModelError> {
        MiniCpmModel::set_rope_delta(self, slot, delta)
    }
    fn prefill_rows(&self) -> usize {
        self.opts.prefill_rows
    }
    fn prefill_moe_grouped(&self) -> bool {
        self.prefill_moe_grouped()
    }
    fn cached_slice(&mut self, arena: &mut SnapshotArena, req: SliceRequest<'_>) -> Result<SliceResult, ModelError> {
        prefix::slice_with_cuts(self, arena, req)
    }
    fn restore_snapshot(
        &mut self,
        arena: &SnapshotArena,
        slot: usize,
        handle: &PrefixHandle,
        len: usize,
    ) -> Result<(), ModelError> {
        prefix::restore(self, arena, slot, handle, len)
    }
    fn release_snapshot(&mut self, arena: &mut SnapshotArena, handle: PrefixHandle) -> Result<(), ModelError> {
        prefix::release(self, arena, handle)
    }
    fn capture_snapshot(
        &mut self,
        arena: &mut SnapshotArena,
        slot: usize,
        len: usize,
    ) -> Result<Option<gb_engine::prefix::Captured>, ModelError> {
        prefix::capture(self, arena, slot, len)
    }
}

/// A slot that was restored from a snapshot and has not yet continued: the cut and the arithmetic path its snapshot was made under.
#[derive(Clone, Copy, Debug)]
struct Restored {
    len: usize,
    path: u32,
}

/// The prefix cache of a backend: the snapshot arena, the pack's fingerprint (the namespace of the engine's keys) and the slots a
/// restore left waiting for their suffix.
struct PrefixState {
    arena: SnapshotArena,
    fingerprint: String,
    restored: BTreeMap<usize, Restored>,
    /// The identity of the image and video preprocessing in the namespace (E5b; `prefix::NO_IMAGES` for a text-only engine).
    images: String,
}

pub struct MiniCpmBackend<M: MiniCpmRuntime = MiniCpmModel> {
    pub m: M,
    /// Rows of a prefill chunk (the model's `prefill_rows` or less).
    pub chunk: usize,
    /// The engine's row limit for a prefill slice while a deadline is near (`Backend::set_prefill_row_limit`): the slices go on the
    /// absolute grid of `min(chunk, limit)` rows, which the forward's row invariance makes the same bits (`effective_chunk`).
    deadline_rows: Option<usize>,
    error: Option<BackendError>,
    fatal: Option<BackendError>,
    prefix: Option<PrefixState>,
}

fn nan_row(n: usize) -> RowOutput {
    RowOutput { ids: (0..n as u32).collect(), values: vec![f32::NAN; n] }
}

fn row_of(c: &Candidates, r: usize, n: usize) -> RowOutput {
    RowOutput {
        ids: c.ids[r * c.count..r * c.count + n].iter().map(|&x| x as u32).collect(),
        values: c.vals[r * c.count..r * c.count + n].to_vec(),
    }
}

/// A resumed slice ends at the next absolute chunk boundary, not `done + chunk`.
fn next_chunk_end(done: usize, total: usize, chunk: usize) -> usize {
    done.saturating_add(chunk - done % chunk).min(total)
}

impl MiniCpmBackend<MiniCpmModel> {
    pub fn new(m: MiniCpmModel, chunk: usize) -> Self {
        Self::with_runtime(m, chunk)
    }
}

impl MiniCpmBackend<MiniCpmModel> {
    /// Oracle seam (E5b, Leg C): prefill `prompt` with its spans exactly as `prefill_slice` does, chunk by chunk, and hand the raw head
    /// logits of rows `[from, to)` to `sink(first_row, rows, logits)` as their chunk completes (`logits` is `rows x vocab` of the
    /// head's dtype: fp16 bits unless `REXL3_MINICPM_LOGITS=bf16`). The slot is begun and ended here.
    pub fn prefill_logits(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        from: usize,
        to: usize,
        mut sink: impl FnMut(usize, usize, &[u16]),
    ) -> Result<(), ModelError> {
        let geo = Geometry::new(prompt, images)?;
        let total = prompt.len();
        if from >= to || to > total {
            return Err(ModelError::Request(format!("logit rows {from}..{to} of a prompt of {total}")));
        }
        self.m.begin_slot(slot)?;
        let max_rows = self.m.prefill_rows().max(self.chunk);
        let vocab = self.m.vocab_size();
        let mut buf = vec![0u16; WINDOW_ROWS * vocab];
        let mut done = 0;
        let run = (|| -> Result<(), ModelError> {
            while done < total {
                let end = geo.chunk_end(done, total, self.chunk, max_rows);
                self.m.flush()?;
                self.mm_chunk(slot, prompt, images, &geo, done, end, None)?;
                // the rows of this chunk that were asked for, in slices of a window
                let (lo, hi) = (from.max(done), to.min(end));
                let mut at = lo;
                while at < hi {
                    let k = WINDOW_ROWS.min(hi - at);
                    self.m.last_rows_logits(at - done, k, &mut buf)?;
                    sink(at, k, &buf[..k * vocab]);
                    at += k;
                }
                done = end;
            }
            Ok(())
        })();
        self.m.end_slot(slot);
        run
    }
}

impl<M: MiniCpmRuntime> MiniCpmBackend<M> {
    pub fn with_runtime(m: M, chunk: usize) -> Self {
        assert!(chunk > 0, "prefill chunk is nonzero");
        Self { m, chunk, deadline_rows: None, error: None, fatal: None, prefix: None }
    }

    /// The rows of the grid the engine-driven slices are cut on: the chunk, or the engine's row limit when it is smaller. Only the
    /// slices of `prefill_slice` / `prefill_slice_cuts` take it (the whole-prompt `prefill_piece` and the oracle seam keep the chunk).
    fn effective_chunk(&self) -> usize {
        self.chunk.min(self.deadline_rows.unwrap_or(usize::MAX)).max(1)
    }

    /// Serve with a prefix cache: `arena` is the snapshot arena the boot allocated for the plan's budget (its KV pages are the
    /// pool's `cache.page_blocks` above the admitted share) and `fingerprint` the pack's (`sha256:` and 64 hex digits), which every
    /// key of the engine's index is bound to.
    pub fn with_prefix_arena(mut self, arena: SnapshotArena, fingerprint: String) -> Result<Self, ModelError> {
        prefix::check_fingerprint(&fingerprint)?;
        self.prefix = Some(PrefixState { arena, fingerprint, restored: BTreeMap::new(), images: prefix::NO_IMAGES.to_string() });
        Ok(self)
    }

    /// Serve images with this preprocessing identity (E5b: `MiniCpmPreprocessing::id()` of the images and of the video frames): it
    /// joins the cache's namespace, so that a snapshot made under one preprocessing is never found under another, and an engine
    /// that has images never shares an entry with one that has none. A no-op without a cache arena.
    pub fn with_image_preprocessing(mut self, id: String) -> Self {
        if let Some(p) = self.prefix.as_mut() {
            p.images = id;
        }
        self
    }

    /// The snapshot arena's bookkeeping (tests and the engine's boot report).
    pub fn prefix_arena(&self) -> Option<&SnapshotArena> {
        self.prefix.as_ref().map(|p| &p.arena)
    }

    /// The model and the snapshot arena together, for the primitives of `prefix` that take both (tests of capture and restore on a
    /// state the slicing routine does not produce). `None` without a cache.
    pub fn with_model_and_arena<R>(&mut self, f: impl FnOnce(&mut M, &mut SnapshotArena) -> R) -> Option<R> {
        let p = self.prefix.as_mut()?;
        Some(f(&mut self.m, &mut p.arena))
    }

    fn record(&mut self, context: &str, e: &ModelError) {
        let message = format!("{context}: {e}");
        if e.is_fatal() {
            self.fatal.get_or_insert_with(|| BackendError::fatal(message));
        } else {
            self.error.get_or_insert_with(|| BackendError::retryable(message));
        }
    }

    fn dead(&self) -> bool {
        self.fatal.is_some()
    }

    fn run_chunk(
        &mut self,
        slot: usize,
        first: usize,
        tokens: &[u32],
        total: usize,
        want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError> {
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: first as u64, tokens: tokens.to_vec() }]);
        self.m.prefill_chunk(&b, PrefillPath::for_prompt(total), want)
    }

    /// The arithmetic guard of a restored slot: its first slice after the restore must be at the cut and in a prompt that takes the
    /// arithmetic path the snapshot was made under (the restore hook has no prompt, so this is where the prompt's total is known).
    /// A prompt that starts over (`done == 0`) forgets the restore.
    fn check_restored(&mut self, slot: usize, total: usize, done: usize) -> Result<(), ModelError> {
        let Some(p) = self.prefix.as_mut() else { return Ok(()) };
        let Some(r) = p.restored.get(&slot).copied() else { return Ok(()) };
        p.restored.remove(&slot);
        if done == 0 {
            return Ok(());
        }
        if done != r.len {
            return Err(ModelError::Request(format!(
                "slot {slot} was restored from a snapshot of {} tokens and a slice resumes it at {done}",
                r.len
            )));
        }
        if prefix::path(total, self.m.prefill_moe_grouped()) != Some(r.path) {
            return Err(ModelError::Request(format!(
                "slot {slot}: a snapshot of {} tokens was made on prefill path {} (the pinned GEMM, the {} prefill MoE) and cannot \
                 continue a prompt of {total} tokens, which takes the other arithmetic",
                r.len,
                r.path,
                prefix::arm_name(r.path)
            )));
        }
        Ok(())
    }

    fn cached_slice_inner(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> Result<CutStep, ModelError> {
        self.check_restored(slot, prompt.len(), done)?;
        if !images.is_empty() {
            return self.cached_slice_mm(slot, prompt, images, done, candidates, cuts);
        }
        let chunk = self.effective_chunk();
        let p = self.prefix.as_mut().ok_or_else(|| ModelError::Config("prefix arena unavailable".into()))?;
        let result = self.m.cached_slice(&mut p.arena, SliceRequest { slot, prompt, done, candidates, cuts, chunk })?;
        Ok(CutStep {
            step: PrefillStep { done: result.end, row: result.row.map(|c| row_of(&c, 0, candidates)) },
            captured: result.captured,
        })
    }

    /// One slice of a prompt with spans under the prefix cache (E5b): [`MiniCpmBackend::prefill_slice_mm`] with the planned cuts. The
    /// chunk ends at the first cut inside it, the state is captured there, and a cut is only taken OUTSIDE every visual group (the
    /// canvas rule: a prefix that ends inside a group would hold positions computed for a truncated canvas), whatever the engine
    /// planned: a prompt whose cuts are all inside frames is prefilled with none, never wrongly cached.
    fn cached_slice_mm(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> Result<CutStep, ModelError> {
        if done >= prompt.len() {
            return Err(ModelError::Request("empty or completed prefill slice".into()));
        }
        let geo = Geometry::new(prompt, images)?;
        let total = prompt.len();
        let grouped = self.m.prefill_moe_grouped();
        let cuts: Vec<usize> = if prefix::path(total, grouped).is_some() {
            cuts.iter().copied().filter(|&c| c > 0 && c < total && geo.canvas.is_cut_outside_frames(c)).collect()
        } else {
            Vec::new()
        };
        if cuts.windows(2).any(|w| w[0] >= w[1]) {
            return Err(ModelError::Request("invalid MiniCPM-V-4.7 prefill slice or unordered cuts".into()));
        }
        let max_rows = self.m.prefill_rows().max(self.chunk);
        let mut end = geo.chunk_end(done, total, self.effective_chunk(), max_rows);
        if let Some(&c) = cuts.iter().find(|&&c| c > done && c < end) {
            end = c;
        }
        if done == 0 {
            self.m.begin_slot(slot)?;
        }
        self.m.flush()?;
        let row = self.mm_chunk(slot, prompt, images, &geo, done, end, (end == total).then_some(candidates))?;
        let mut captured = Vec::new();
        if end < total && cuts.contains(&end) {
            // the snapshot's host record names the offset the text after the cut will have (the next slice sets its own)
            self.m.set_rope_delta(slot, geo.text_offset(end))?;
            let p = self.prefix.as_mut().ok_or_else(|| ModelError::Config("prefix arena unavailable".into()))?;
            captured.extend(self.m.capture_snapshot(&mut p.arena, slot, end)?);
        }
        Ok(CutStep { step: PrefillStep { done: end, row: row.map(|c| row_of(&c, 0, candidates)) }, captured })
    }

    fn prefill_slice_inner(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> Result<PrefillStep, ModelError> {
        if done >= prompt.len() {
            return Err(ModelError::Request("empty or completed prefill slice".into()));
        }
        if !images.is_empty() {
            return self.prefill_slice_mm(slot, prompt, images, done, candidates);
        }
        if done == 0 {
            self.m.begin_slot(slot)?;
        }
        self.m.flush()?;
        let end = next_chunk_end(done, prompt.len(), self.effective_chunk());
        let c = self.run_chunk(slot, done, &prompt[done..end], prompt.len(), (end == prompt.len()).then_some(candidates))?;
        Ok(PrefillStep { done: end, row: c.map(|c| row_of(&c, 0, candidates)) })
    }

    /// One slice of a prompt with image or video spans (E5b): the chunk ends where [`Geometry::chunk_end`] says, its rows carry the
    /// canvas triples of the whole prompt (explicit when it touches a frame, a plain offset otherwise), the spans' feature rows are
    /// spliced over the pad tokens, and the slot's rope delta is set when the prompt completes. Everything that can be refused is
    /// refused before the first launch of the slice.
    fn prefill_slice_mm(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> Result<PrefillStep, ModelError> {
        let geo = Geometry::new(prompt, images)?;
        let total = prompt.len();
        let max_rows = self.m.prefill_rows().max(self.chunk);
        let end = geo.chunk_end(done, total, self.effective_chunk(), max_rows);
        if done == 0 {
            self.m.begin_slot(slot)?;
        }
        self.m.flush()?;
        let c = self.mm_chunk(slot, prompt, images, &geo, done, end, (end == total).then_some(candidates))?;
        Ok(PrefillStep { done: end, row: c.map(|c| row_of(&c, 0, candidates)) })
    }

    /// Rows `[done, end)` of a prompt with spans as one prefill chunk: positions, splice, and the slot's delta once the prompt is in.
    #[allow(clippy::too_many_arguments)]
    fn mm_chunk(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        geo: &Geometry,
        done: usize,
        end: usize,
        want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError> {
        let total = prompt.len();
        if end <= done || end > total || end - done > self.m.prefill_rows() {
            return Err(ModelError::Request(format!(
                "a prefill chunk of rows {done}..{end} of {total} (the runtime takes {} a chunk)",
                self.m.prefill_rows()
            )));
        }
        let mut b =
            RowBatch::from_requests(&[RowRequest { slot, first_position: done as u64, tokens: prompt[done..end].to_vec() }]);
        if geo.is_explicit(done, end) {
            b.rope3 = Some(geo.rope3(done, end));
        } else {
            self.m.set_rope_delta(slot, geo.text_offset(done))?;
        }
        let ops = geo.splice(prompt, done, end);
        let splice: Vec<ChunkSplice<'_>> = ops
            .iter()
            .map(|o| ChunkSplice { features: &*images[o.span].features, dst_row: o.dst_row, src_row: o.src_row, n: o.n })
            .collect();
        let c = self.m.prefill_chunk_mm(&b, PrefillPath::for_prompt(total), want, &splice)?;
        if end == total {
            self.m.set_rope_delta(slot, geo.rope_delta())?;
        }
        Ok(c)
    }

    /// The host-side check of one request of a step, before anything is launched: `rows_so_far` is what the earlier accepted requests
    /// of the step already take of the window, `slots` the slots they use.
    fn check_request(&mut self, r: &RowRequest, rows_so_far: usize, slots: &BTreeSet<usize>) -> Result<(), ModelError> {
        if r.tokens.is_empty() {
            return Err(ModelError::Request(format!("slot {}: a request with no rows", r.slot)));
        }
        if rows_so_far + r.tokens.len() > WINDOW_ROWS {
            return Err(ModelError::Request(format!(
                "slot {}: {} rows do not fit the {WINDOW_ROWS}-row window ({rows_so_far} already taken)",
                r.slot,
                r.tokens.len()
            )));
        }
        if slots.contains(&r.slot) {
            return Err(ModelError::Request(format!("slot {} appears twice in one step", r.slot)));
        }
        let vocab = self.m.vocab_size();
        if let Some(&t) = r.tokens.iter().find(|&&t| t as usize >= vocab) {
            return Err(ModelError::Request(format!("slot {}: token id {t} is outside the vocabulary of {vocab}", r.slot)));
        }
        let first = usize::try_from(r.first_position)
            .map_err(|_| ModelError::Request(format!("slot {}: position {} overflows", r.slot, r.first_position)))?;
        self.m.reserve(r.slot, first, r.tokens.len())
    }
}

impl<M: MiniCpmRuntime> Backend for MiniCpmBackend<M> {
    /// Row-invariant slicing (the forward's bits do not depend on the chunk grid, `a_tiled_prefill_leaves_the_same_bits_in_any_chunk_grid`):
    /// the engine's slices go on the absolute grid of the limit. A visual group that fits the model's chunk is still carried whole, so
    /// a slice over an image can be longer than the limit (the geometry rule; the bits do not depend on it either way).
    fn set_prefill_row_limit(&mut self, rows: Option<usize>) -> bool {
        self.deadline_rows = rows.map(|r| r.max(1));
        true
    }
    fn prefix_caps(&self) -> Option<PrefixCaps> {
        self.prefix.as_ref().map(|p| {
            // the draft's identity is part of the namespace: a snapshot made with one draft is invisible to an engine with another
            let draft = self.m.draft_identity();
            prefix::caps_with_draft(
                &p.arena,
                &p.fingerprint,
                &gb_cuda::exl3_linear::prefill_pin_name(),
                self.m.prefill_moe_grouped(),
                &p.images,
                draft.as_deref().unwrap_or(prefix::NO_DRAFT),
            )
        })
    }

    fn prefix_path(&self, total: usize) -> Option<u32> {
        self.prefix.as_ref().and_then(|_| prefix::path(total, self.m.prefill_moe_grouped()))
    }

    fn prefix_free_snapshots(&self) -> usize {
        self.prefix.as_ref().map_or(0, |p| p.arena.free_count())
    }

    /// Restore `slot` from `handle`. Only that slot is touched: on an error it is ended (`restore` has begun it already, and the
    /// recurrent state of every other slot is never written), and the engine drops the entry and prefills cold.
    fn prefix_restore(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), String> {
        if self.dead() {
            return Err("the backend has failed".into());
        }
        let result = match self.prefix.as_mut() {
            Some(p) => {
                p.restored.remove(&slot);
                let r = self.m.restore_snapshot(&p.arena, slot, handle, len);
                if r.is_ok() {
                    let path = p.arena.meta(handle.snap).map_or(prefix::PATH_ID, |m| m.path);
                    p.restored.insert(slot, Restored { len, path });
                }
                r
            }
            None => Err(ModelError::Config("prefix arena unavailable".into())),
        };
        if let Err(e) = &result {
            if e.is_fatal() {
                self.record("prefix restore", e);
            }
            if !self.dead() {
                self.m.end_slot(slot);
            }
        }
        result.map_err(|e| e.to_string())
    }

    fn prefix_release(&mut self, handle: PrefixHandle) {
        let result = match self.prefix.as_mut() {
            Some(p) => self.m.release_snapshot(&mut p.arena, handle),
            None => Err(ModelError::Config("prefix arena unavailable".into())),
        };
        if let Err(e) = result {
            self.record("prefix release", &e);
        }
    }

    fn prefill_slice_cuts(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
    ) -> CutStep {
        if self.dead() {
            return CutStep { step: PrefillStep { done: prompt.len(), row: Some(nan_row(candidates)) }, captured: Vec::new() };
        }
        match self.cached_slice_inner(slot, prompt, images, done, candidates, cuts) {
            Ok(cut) => cut,
            Err(e) => {
                self.record(&format!("prefill slice with cuts of slot {slot}"), &e);
                if !self.dead() {
                    self.m.end_slot(slot);
                }
                CutStep { step: PrefillStep { done: prompt.len(), row: Some(nan_row(candidates)) }, captured: Vec::new() }
            }
        }
    }

    fn vocab_size(&self) -> usize {
        self.m.vocab_size()
    }

    fn take_error(&mut self) -> Option<BackendError> {
        self.fatal.clone().or_else(|| self.error.take())
    }

    fn quiesce(&mut self) -> Result<(), String> {
        self.m.quiesce().map_err(|e| e.to_string())
    }

    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.prefill_piece(slot, start, tokens, (start as usize).saturating_add(tokens.len()), candidates)
    }

    fn prefill_piece(&mut self, slot: usize, start: u64, tokens: &[u32], total: usize, candidates: usize) -> RowOutput {
        if self.dead() {
            return nan_row(candidates);
        }
        let run = (|| -> Result<RowOutput, ModelError> {
            let first = usize::try_from(start).map_err(|_| ModelError::Request("prefill position overflows usize".into()))?;
            if tokens.is_empty() || first.checked_add(tokens.len()).is_none_or(|n| n > total) {
                return Err(ModelError::Request("empty prefill piece or piece past prompt total".into()));
            }
            if first == 0 {
                self.m.begin_slot(slot)?;
                if let Some(p) = self.prefix.as_mut() {
                    p.restored.remove(&slot);
                }
            }
            self.m.flush()?;
            let mut at = first;
            let mut last = None;
            while at < first + tokens.len() {
                let end = next_chunk_end(at, first + tokens.len(), self.chunk);
                let c = self.run_chunk(
                    slot,
                    at,
                    &tokens[at - first..end - first],
                    total,
                    (end == first + tokens.len()).then_some(candidates),
                )?;
                last = c;
                at = end;
            }
            let c = last.ok_or_else(|| ModelError::Request("prefill produced no candidates".into()))?;
            Ok(row_of(&c, 0, candidates))
        })();
        match run {
            Ok(row) => row,
            Err(e) => {
                self.record(&format!("prefill of slot {slot}"), &e);
                if !self.dead() {
                    self.m.end_slot(slot);
                }
                nan_row(candidates)
            }
        }
    }

    fn prefill_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> PrefillStep {
        if self.dead() {
            return PrefillStep { done: prompt.len(), row: Some(nan_row(candidates)) };
        }
        // with a prefix cache every slice goes through the one routine (with no cuts it is the uncached path)
        let result = if self.prefix.is_some() {
            self.cached_slice_inner(slot, prompt, images, done, candidates, &[]).map(|cut| cut.step)
        } else {
            self.prefill_slice_inner(slot, prompt, images, done, candidates)
        };
        match result {
            Ok(step) => step,
            Err(e) => {
                self.record(&format!("prefill slice of slot {slot}"), &e);
                if !self.dead() {
                    self.m.end_slot(slot);
                }
                PrefillStep { done: prompt.len(), row: Some(nan_row(candidates)) }
            }
        }
    }

    /// A whole multimodal prompt in one call (the engine itself drives `prefill_slice`): slices until the prompt is in.
    fn prefill_mm(&mut self, slot: usize, start: u64, tokens: &[u32], images: &[ImageSpan], candidates: usize) -> RowOutput {
        if start != 0 {
            self.record(
                "multimodal prefill",
                &ModelError::Request("a multimodal prefill starts at the beginning of the prompt".into()),
            );
            if !self.dead() {
                self.m.end_slot(slot);
            }
            return nan_row(candidates);
        }
        let mut done = 0;
        loop {
            let step = self.prefill_slice(slot, tokens, images, done, candidates);
            done = step.done;
            if let Some(row) = step.row {
                return row;
            }
            if done >= tokens.len() {
                return nan_row(candidates);
            }
        }
    }

    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        let nans = |reqs: &[RowRequest]| -> Vec<Vec<RowOutput>> {
            reqs.iter().map(|r| r.tokens.iter().map(|_| nan_row(candidates)).collect()).collect()
        };
        if self.dead() {
            return nans(reqs);
        }
        if let Err(e) = self.m.flush() {
            self.record("window step", &e);
            return nans(reqs);
        }
        // per-request validation: the requests that fail are dropped from the batch and answered with NaN rows alone
        let (mut run, mut rows, mut slots) = (Vec::with_capacity(reqs.len()), 0usize, BTreeSet::new());
        for (i, r) in reqs.iter().enumerate() {
            match self.check_request(r, rows, &slots) {
                Ok(()) => {
                    rows += r.tokens.len();
                    slots.insert(r.slot);
                    run.push(i);
                }
                Err(e) => eprintln!("rexl3 level=warn event=minicpm_request_rejected slot={} reason=\"{e}\"", r.slot),
            }
        }
        let mut out = nans(reqs);
        if run.is_empty() {
            return out;
        }
        let batch = RowBatch::from_requests(&run.iter().map(|&i| reqs[i].clone()).collect::<Vec<_>>());
        match self.m.window_candidates(&batch, candidates) {
            Ok(c) => {
                for (k, &i) in run.iter().enumerate() {
                    out[i] = batch.seq_rows(k).map(|r| row_of(&c, r, candidates)).collect();
                }
                out
            }
            Err(e) => {
                self.record("window step", &e);
                nans(reqs)
            }
        }
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        if self.dead() {
            return;
        }
        self.m.commit(slot, keep);
    }

    /// The DSpark draft's proposals with its calibrated confidences (`Proposal::accept`). A stream the draft cannot serve now (its
    /// context is not the target's, a failure of its own) gets `None` and the engine leaves it at plain decoding; a failure of the
    /// draft is never recorded as a backend error (a recorded error fails every sequence of the call it surfaces in), only a fatal
    /// device error is.
    fn propose_batch_scored(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Proposal>> {
        if self.dead() || !self.m.has_draft() || reqs.is_empty() {
            return vec![None; reqs.len()];
        }
        let asks: Vec<SpecAsk<'_>> =
            reqs.iter().map(|r| SpecAsk { slot: r.slot, tokens: r.tokens, sampling: r.sampling, max: r.max }).collect();
        match self.m.propose_round(&asks) {
            Ok(v) => {
                v.into_iter().map(|o| o.map(|p| Proposal { tokens: p.tokens, accept: p.accept, ..Default::default() })).collect()
            }
            Err(e) => {
                self.record(&format!("drafting for {} sequences", reqs.len()), &e);
                vec![None; reqs.len()]
            }
        }
    }

    fn release(&mut self, slot: usize) {
        if let Some(p) = self.prefix.as_mut() {
            p.restored.remove(&slot);
        }
        if !self.dead() {
            self.m.end_slot(slot);
        }
    }
}

#[cfg(test)]
#[path = "backend_tests.rs"]
mod tests;
