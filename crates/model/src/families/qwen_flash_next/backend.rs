//! The engine's text-only Flash-Next backend. The model owns all device state; this layer
//! chooses prompt-wide arithmetic, slices on an absolute grid and translates errors.
use super::dflash::ctx_from;
use super::graph_cache::GraphStats;
use super::model::{DflashAsk, FlashNextModel};
use super::prefix::{self, SliceRequest, SliceResult, SnapshotArena};
use super::spec::{DraftSpec, Pick};
use crate::error::ModelError;
use crate::forward::{Candidates, PrefillPath};
use crate::rows::RowBatch;
use gb_engine::prefix::{PrefixCaps, PrefixHandle};
use gb_engine::score::RowRecords;
use gb_engine::{Backend, BackendError, CutStep, ImageSpan, PrefillStep, Proposal, ProposeRequest, RowOutput, RowRequest};
use std::collections::BTreeMap;

/// Narrow adapter so the host scheduling contract can run without a GPU or a model pack.
pub trait FlashNextRuntime {
    fn vocab_size(&self) -> usize;
    fn has_mtp(&self) -> bool;
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError>;
    fn end_slot(&mut self, slot: usize);
    fn prefill_chunk(&mut self, b: &RowBatch, path: PrefillPath, want: Option<usize>) -> Result<Option<Candidates>, ModelError>;
    fn mtp_after_prefill(
        &mut self,
        slot: usize,
        first: usize,
        chunk: &[u32],
        next: Option<u32>,
        path: PrefillPath,
    ) -> Result<(), ModelError>;
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError>;
    /// Like [`FlashNextRuntime::window_candidates`], but with the device-side greedy gather: the same
    /// graph runs, and the returned candidates (when `Some`) carry one candidate per row — the greedy
    /// argmax. `None` means the runtime does not support the gather: the caller falls back to the full
    /// `window_candidates` readback. A runtime that supports the gather always returns `Some` — there
    /// is no switch.
    fn window_candidates_dv(&mut self, _b: &RowBatch, _count: usize) -> Result<Option<Candidates>, ModelError> {
        Ok(None)
    }
    fn commit(&mut self, slot: usize, keep: usize);
    fn flush(&mut self) -> Result<(), ModelError>;
    fn mtp_propose(&mut self, slot: usize, tokens: &[u32], k: usize) -> Result<Vec<u32>, ModelError>;
    fn cached_slice(&mut self, _arena: &mut SnapshotArena, _req: SliceRequest<'_>) -> Result<SliceResult, ModelError> {
        Err(ModelError::Config("prefix snapshots need the device model".into()))
    }
    fn restore_snapshot(
        &mut self,
        _arena: &SnapshotArena,
        _slot: usize,
        _handle: &PrefixHandle,
        _len: usize,
    ) -> Result<(), ModelError> {
        Err(ModelError::Config("prefix snapshots need the device model".into()))
    }
    fn release_snapshot(&mut self, _arena: &mut SnapshotArena, _handle: PrefixHandle) -> Result<(), ModelError> {
        Err(ModelError::Config("prefix snapshots need the device model".into()))
    }
    /// Whether the runtime drafts with a DFlash2 draft (`--draft`) instead of the MTP head.
    fn has_dflash(&self) -> bool {
        false
    }
    /// The tokens of the chunk after the one about to be prefilled (two context tokens, then its positions): a runtime with an
    /// n-gram table reads their pages ahead while the chunk runs. The default does nothing.
    fn ple_readahead(&mut self, _history: &[u32]) {}
    /// The first prompt position of `slot`'s prompt the draft must absorb (`dflash::ctx_from`), before each prefill slice.
    fn set_dflash_from(&mut self, _slot: usize, _from: usize) {}
    /// Whether `slot` may draft for a sequence of `tokens` tokens (the last one not yet fed).
    fn dflash_ready(&self, _slot: usize, _tokens: usize) -> bool {
        false
    }
    /// One DFlash2 round for these sequences (each [`FlashNextRuntime::dflash_ready`]): their drafts, in ask order.
    fn dflash_draft(&mut self, _asks: &[DflashAsk<'_>]) -> Result<Vec<Vec<u32>>, ModelError> {
        Err(ModelError::Config("this runtime has no DFlash2 draft".into()))
    }
    /// Whether this runtime reads out the log-probability records of a prefill chunk ([`FlashNextRuntime::chunk_row_records`]),
    /// which is what the `prompt_logprobs` readout needs. The single-node model does; a runtime that does not say so (a
    /// tensor-parallel group) has the readout refused by name.
    fn row_records_supported(&self) -> bool {
        false
    }
    /// [`FlashNextModel::chunk_row_records`]: the records of the rows of the prefill chunk that just ran.
    fn chunk_row_records(&mut self, _chosen: &[i32], _out: &mut RowRecords) -> Result<(), ModelError> {
        Err(ModelError::Request("this Flash-Next runtime does not read out prompt log-probabilities".into()))
    }
    /// The full-vocabulary logits of the last row of the prefill chunk that just ran.
    fn chunk_last_row_logits(&mut self) -> Result<Vec<f32>, ModelError> {
        Err(ModelError::Request("this Flash-Next runtime does not read out prompt log-probabilities".into()))
    }
    /// The counters of the runtime's window and chain graph caches; `None` for a runtime that keeps none.
    fn graph_stats(&self) -> Option<GraphStats> {
        None
    }
}

impl FlashNextRuntime for FlashNextModel {
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
    fn has_mtp(&self) -> bool {
        self.has_mtp()
    }
    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        self.begin_slot(slot)
    }
    fn end_slot(&mut self, slot: usize) {
        self.end_slot(slot)
    }
    fn ple_readahead(&mut self, history: &[u32]) {
        self.ple_readahead(history)
    }
    fn prefill_chunk(&mut self, b: &RowBatch, path: PrefillPath, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        self.prefill_chunk(b, path, want)
    }
    fn mtp_after_prefill(
        &mut self,
        slot: usize,
        first: usize,
        chunk: &[u32],
        next: Option<u32>,
        path: PrefillPath,
    ) -> Result<(), ModelError> {
        self.mtp_after_prefill(slot, first, chunk, next, path)
    }
    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        self.window_candidates(b, count)
    }
    fn window_candidates_dv(&mut self, b: &RowBatch, count: usize) -> Result<Option<Candidates>, ModelError> {
        self.window_candidates_dv(b, count)
    }
    fn commit(&mut self, slot: usize, keep: usize) {
        self.commit(slot, keep)
    }
    fn flush(&mut self) -> Result<(), ModelError> {
        self.flush()
    }
    fn mtp_propose(&mut self, slot: usize, tokens: &[u32], k: usize) -> Result<Vec<u32>, ModelError> {
        self.mtp_propose(slot, tokens, k)
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
    fn has_dflash(&self) -> bool {
        self.has_dflash()
    }
    fn set_dflash_from(&mut self, slot: usize, from: usize) {
        self.set_dflash_from(slot, from)
    }
    fn dflash_ready(&self, slot: usize, tokens: usize) -> bool {
        self.dflash_ready(slot, tokens)
    }
    fn dflash_draft(&mut self, asks: &[DflashAsk<'_>]) -> Result<Vec<Vec<u32>>, ModelError> {
        self.dflash_draft(asks)
    }
    fn row_records_supported(&self) -> bool {
        true
    }
    fn chunk_row_records(&mut self, chosen: &[i32], out: &mut RowRecords) -> Result<(), ModelError> {
        self.chunk_row_records(chosen, out)
    }
    fn chunk_last_row_logits(&mut self) -> Result<Vec<f32>, ModelError> {
        self.chunk_last_row_logits()
    }
    fn graph_stats(&self) -> Option<GraphStats> {
        Some(self.graph_stats())
    }
}

pub struct FlashNextBackend<M: FlashNextRuntime = FlashNextModel> {
    pub m: M,
    pub chunk: usize,
    pub mtp_k: usize,
    /// The engine's row limit for a prefill slice while a deadline is near (`Backend::set_prefill_row_limit`): the slices go on the
    /// absolute grid of `min(chunk, limit)` rows, which the chunk-invariant forward makes the same bits (`effective_chunk`).
    deadline_rows: Option<usize>,
    adaptive: bool,
    last_acc: BTreeMap<usize, usize>,
    error: Option<BackendError>,
    fatal: Option<BackendError>,
    prefix: Option<(SnapshotArena, String)>,
    /// The DFlash2 draft's arbiter (`spec.rs`), made at the first draft round of a runtime that drafts with one.
    spec: Option<DraftSpec>,
    /// The `prompt_logprobs` readouts in flight, by slot (`Backend::score_begin`): the records of the rows scored so far.
    records: BTreeMap<usize, RowRecords>,
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

impl FlashNextBackend<FlashNextModel> {
    pub fn new(m: FlashNextModel, chunk: usize, mtp_k: usize) -> Self {
        let adaptive = std::env::var("REXL3_MTP_ADAPT").map_or(true, |v| v != "0");
        Self::with_runtime(m, chunk, mtp_k, adaptive)
    }

    pub fn with_prefix_cache(mut self, budget: gb_sched::prefix::CacheBudget, fingerprint: String) -> Result<Self, ModelError> {
        self.prefix = Some((SnapshotArena::new(&self.m, budget)?, fingerprint));
        Ok(self)
    }
}

impl<M: FlashNextRuntime> FlashNextBackend<M> {
    pub fn with_runtime(m: M, chunk: usize, mtp_k: usize, adaptive: bool) -> Self {
        assert!(chunk > 0, "prefill chunk is nonzero");
        Self {
            m,
            chunk,
            mtp_k,
            deadline_rows: None,
            adaptive,
            last_acc: BTreeMap::new(),
            error: None,
            fatal: None,
            prefix: None,
            spec: None,
            records: BTreeMap::new(),
        }
    }

    /// The rows of the grid the engine-driven slices are cut on: the chunk, or the engine's row limit when it is smaller. Only the
    /// slices of `prefill_slice` / `prefill_slice_cuts` take it (the whole-prompt `prefill_piece` keeps the chunk).
    fn effective_chunk(&self) -> usize {
        self.chunk.min(self.deadline_rows.unwrap_or(usize::MAX)).max(1)
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
        next: Option<u32>,
        total: usize,
        want: Option<usize>,
    ) -> Result<Option<Candidates>, ModelError> {
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: first as u64, tokens: tokens.to_vec() }]);
        let path = PrefillPath::for_prompt(total);
        let c = self.m.prefill_chunk(&b, path, want)?;
        if self.mtp_k > 0 {
            self.m.mtp_after_prefill(slot, first, tokens, next, path)?;
        }
        Ok(c)
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
        if !images.is_empty() {
            return Err(ModelError::Request("Flash-Next vision is not built".into()));
        }
        let chunk = self.effective_chunk();
        if done == 0 {
            self.spec_begin(slot);
        }
        let (arena, _) = self.prefix.as_mut().ok_or_else(|| ModelError::Config("prefix arena unavailable".into()))?;
        let result = self.m.cached_slice(arena, SliceRequest { slot, prompt, done, candidates, cuts, chunk })?;
        Ok(CutStep {
            step: PrefillStep { done: result.end, row: result.row.map(|c| row_of(&c, 0, candidates)) },
            captured: result.captured,
        })
    }

    fn prefill_slice_inner(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
    ) -> Result<PrefillStep, ModelError> {
        if !images.is_empty() {
            return Err(ModelError::Request("Flash-Next vision is not built".into()));
        }
        if done >= prompt.len() {
            return Err(ModelError::Request("empty or completed prefill slice".into()));
        }
        if done == 0 {
            self.m.begin_slot(slot)?;
            self.spec_begin(slot);
        }
        self.m.flush()?;
        self.m.set_dflash_from(slot, ctx_from(prompt.len(), &[], done));
        let end = next_chunk_end(done, prompt.len(), self.effective_chunk());
        if end < prompt.len() && end >= 2 {
            let after = next_chunk_end(end, prompt.len(), self.effective_chunk());
            self.m.ple_readahead(&prompt[end - 2..after]);
        }
        let c = self.run_chunk(
            slot,
            done,
            &prompt[done..end],
            prompt.get(end).copied(),
            prompt.len(),
            (end == prompt.len()).then_some(candidates),
        )?;
        Ok(PrefillStep { done: end, row: c.map(|c| row_of(&c, 0, candidates)) })
    }
}

impl<M: FlashNextRuntime> FlashNextBackend<M> {
    /// One slice of a `prompt_logprobs` readout: the chunk a plain prefill would run next (the same absolute grid, the same
    /// forward), then the records of its rows, read out while the collapsed trunk rows are still on the device. No snapshot
    /// is taken or restored (the readout is submitted without the prefix cache) and the MTP head does not run: nothing is
    /// generated from this slot, and the head would overwrite the rows.
    fn score_slice_inner(&mut self, slot: usize, prompt: &[u32], done: usize) -> Result<gb_engine::score::ScoreStep, ModelError> {
        let n = prompt.len();
        if done >= n {
            return Err(ModelError::Request("empty or completed score slice".into()));
        }
        if done == 0 {
            self.m.begin_slot(slot)?;
            self.spec_begin(slot);
        }
        self.m.flush()?;
        self.m.set_dflash_from(slot, ctx_from(n, &[], done));
        let end = next_chunk_end(done, n, self.effective_chunk());
        let b = RowBatch::from_requests(&[RowRequest { slot, first_position: done as u64, tokens: prompt[done..end].to_vec() }]);
        self.m.prefill_chunk(&b, PrefillPath::for_prompt(n), None)?;
        // row `i` predicts token `i + 1`; the prompt's last row has none
        let chosen: Vec<i32> = (done..end).map(|i| prompt.get(i + 1).map_or(-1, |&t| t as i32)).collect();
        let rec = self.records.get_mut(&slot).ok_or_else(|| ModelError::Request("no readout was begun in this slot".into()))?;
        self.m.chunk_row_records(&chosen, rec)?;
        let row = if end == n { Some(self.m.chunk_last_row_logits()?) } else { None };
        Ok(gb_engine::score::ScoreStep { done: end, row, captured: Vec::new() })
    }

    /// A new sequence in `slot`: the draft arbiter's estimates for it start again.
    fn spec_begin(&mut self, slot: usize) {
        if let Some(s) = self.spec.as_mut() {
            s.begin(slot);
        }
    }

    /// The window just verified: each sequence's rows and the window's time (the arbiter's verify cost tracks it).
    fn spec_window(&mut self, reqs: &[RowRequest], ms: f64) {
        if let Some(s) = self.spec.as_mut() {
            let rows: usize = reqs.iter().map(|r| r.tokens.len()).sum();
            for r in reqs {
                s.on_step(r.slot, r.tokens.len());
            }
            s.cost.observe_window(rows, ms);
        }
    }

    /// The DFlash2 round: the arbiter (`spec.rs`, `--spec auto`) picks each sequence's draft depth, the copy drafter's proposal or no drafts
    /// from its measured acceptance and the round's cost; the sequences picked for the draft are drafted in one batched call (the block runs
    /// its 8 rows whatever the depth). A sampled request's walk draws with its sampler. Drafts only decide how many rows the verify window
    /// carries, never which token is committed.
    fn propose_dflash(&mut self, reqs: &[ProposeRequest<'_>]) -> Vec<Option<Proposal>> {
        let mut out: Vec<Option<Proposal>> = vec![None; reqs.len()];
        let ready: Vec<bool> = reqs.iter().map(|r| r.max > 0 && self.m.dflash_ready(r.slot, r.tokens.len())).collect();
        let FlashNextBackend { spec, .. } = self;
        let spec = spec.get_or_insert_with(|| DraftSpec::new(super::engine::STEP_MS));
        let picks = spec.choose(&reqs.iter().zip(&ready).map(|(r, &ok)| (r.slot, r.tokens, r.max, ok)).collect::<Vec<_>>());
        let asked: Vec<usize> = (0..reqs.len()).filter(|&i| matches!(picks[i], Pick::Draft(_))).collect();
        let mut drafted: Vec<Vec<u32>> = vec![Vec::new(); reqs.len()];
        if !asked.is_empty() {
            let asks: Vec<DflashAsk<'_>> = asked
                .iter()
                .map(|&i| {
                    let r = &reqs[i];
                    let Pick::Draft(depth) = picks[i] else { unreachable!() };
                    DflashAsk {
                        slot: r.slot,
                        tokens: r.tokens,
                        depth: depth.min(r.max).min(crate::draft::DRAFTS),
                        sampling: (r.sampling.temperature > 0.0).then(|| r.sampling.clone()),
                    }
                })
                .collect();
            let t = std::time::Instant::now();
            match self.m.dflash_draft(&asks) {
                Ok(drafts) => {
                    if let Some(s) = self.spec.as_mut() {
                        s.cost.observe_draft(asks.len(), t.elapsed().as_secs_f64() * 1e3);
                    }
                    for (&i, mut tokens) in asked.iter().zip(drafts) {
                        tokens.truncate(reqs[i].max);
                        drafted[i] = tokens;
                    }
                }
                Err(e) => self.record("DFlash2 draft round", &e),
            }
        }
        let spec = self.spec.as_mut().expect("made above");
        for (i, r) in reqs.iter().enumerate() {
            // the arbiter's pick, final: a serial round is no drafts at all (not the engine's copy drafter, which it already weighed)
            let tokens = match &picks[i] {
                Pick::Draft(_) => std::mem::take(&mut drafted[i]),
                Pick::Copy(c) => c.clone(),
                Pick::Serial => Vec::new(),
            };
            spec.offered(r.slot, &picks[i], tokens.len());
            out[i] = Some(Proposal { tokens, ..Default::default() });
        }
        out
    }
}

impl<M: FlashNextRuntime> Backend for FlashNextBackend<M> {
    /// Row-invariant slicing: the GDN prefill is chunk-invariant (`gb_cuda::gdn::prefill`), the pinned prefill GEMM row invariant and the
    /// prefill path keyed on the prompt's total, so the engine's slices go on the absolute grid of the limit with the same bits (the
    /// off-grid cuts of the prefix cache rely on the same property).
    /// The teacher-forced readout only: last-row scoring (`score_supported`) is not served by this family.
    fn score_rows_supported(&self) -> bool {
        self.m.row_records_supported()
    }
    fn score_begin(&mut self, slot: usize, all_rows: bool) {
        self.records.remove(&slot);
        if all_rows && self.m.row_records_supported() {
            self.records.insert(slot, RowRecords::new(gb_engine::score::RECORD_TOP.min(self.m.vocab_size())));
        }
    }
    fn take_score_records(&mut self, slot: usize) -> Option<RowRecords> {
        self.records.remove(&slot)
    }
    fn score_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        done: usize,
        _cuts: &[usize],
    ) -> Result<gb_engine::score::ScoreStep, BackendError> {
        if let Some(e) = self.fatal.clone() {
            return Err(e);
        }
        if !self.records.contains_key(&slot) {
            return Err(gb_engine::score::unsupported_backend());
        }
        self.score_slice_inner(slot, prompt, done).map_err(|e| {
            self.records.remove(&slot);
            if e.is_fatal() {
                self.record(&format!("prompt readout of slot {slot}"), &e);
            } else {
                self.m.end_slot(slot);
            }
            // a recoverable error fails this request alone, through the returned value (not `take_error`)
            self.fatal.clone().unwrap_or_else(|| BackendError::retryable(format!("prompt readout of slot {slot}: {e}")))
        })
    }
    fn set_prefill_row_limit(&mut self, rows: Option<usize>) -> bool {
        self.deadline_rows = rows.map(|r| r.max(1));
        true
    }
    fn prefix_caps(&self) -> Option<PrefixCaps> {
        self.prefix
            .as_ref()
            .map(|(arena, fingerprint)| prefix::caps(arena, fingerprint, &gb_cuda::exl3_linear::prefill_pin_name()))
    }
    fn prefix_path(&self, total: usize) -> Option<u32> {
        self.prefix.as_ref().and_then(|_| prefix::path(total))
    }
    fn prefix_free_snapshots(&self) -> usize {
        self.prefix.as_ref().map_or(0, |(arena, _)| arena.free_count())
    }
    fn prefix_restore(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), String> {
        self.spec_begin(slot);
        let result = match self.prefix.as_ref() {
            Some((arena, _)) => self.m.restore_snapshot(arena, slot, handle, len),
            None => Err(ModelError::Config("prefix arena unavailable".into())),
        };
        if result.is_err() {
            self.m.end_slot(slot);
        }
        result.map_err(|e| e.to_string())
    }
    fn prefix_release(&mut self, handle: PrefixHandle) {
        if let Some((arena, _)) = self.prefix.as_mut() {
            if let Err(e) = self.m.release_snapshot(arena, handle) {
                self.record("prefix release", &e);
            }
        } else {
            self.record("prefix release", &ModelError::Config("prefix arena unavailable".into()));
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
                self.spec_begin(slot);
            }
            self.m.flush()?;
            self.m.set_dflash_from(slot, ctx_from(total, &[], first));
            let mut at = first;
            let mut last = None;
            while at < first + tokens.len() {
                let end = next_chunk_end(at, first + tokens.len(), self.chunk);
                let c = self.run_chunk(
                    slot,
                    at,
                    &tokens[at - first..end - first],
                    tokens.get(end - first).copied(),
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

    fn prefill_mm(&mut self, slot: usize, _: u64, _: &[u32], _: &[ImageSpan], candidates: usize) -> RowOutput {
        self.record("multimodal prefill", &ModelError::Request("Flash-Next vision is not built".into()));
        if !self.dead() {
            self.m.end_slot(slot);
        }
        nan_row(candidates)
    }

    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        let nans = || reqs.iter().map(|r| r.tokens.iter().map(|_| nan_row(candidates)).collect()).collect();
        if self.dead() {
            return nans();
        }
        let t = std::time::Instant::now();
        let run = (|| -> Result<_, ModelError> {
            let b = RowBatch::from_requests(reqs);
            let c = self.m.window_candidates(&b, candidates)?;
            Ok((0..reqs.len()).map(|i| b.seq_rows(i).map(|r| row_of(&c, r, candidates)).collect()).collect())
        })();
        match run {
            Ok(rows) => {
                self.spec_window(reqs, t.elapsed().as_secs_f64() * 1e3);
                rows
            }
            Err(e) => {
                self.record("window step", &e);
                nans()
            }
        }
    }

    /// [`Backend::step`] with the device-side greedy gather: when every request is greedy, the window
    /// runs as usual but the candidate table is packed to its argmax column on the device
    /// (`FlashNextRuntime::window_candidates_dv`), so the round reads back `rows * 4` bytes instead of
    /// `rows * count * 8`. The gathered rows are the same candidates the full download would give a
    /// greedy sampler (column 0 of the sorted table), so the engine's accept/counters are untouched.
    /// When the runtime does not gather and for any batch with a sampled request, this is exactly
    /// [`Backend::step`].
    fn step_verified(&mut self, reqs: &[RowRequest], candidates: usize, greedy: &[bool]) -> Vec<Vec<RowOutput>> {
        let nans = || reqs.iter().map(|r| r.tokens.iter().map(|_| nan_row(candidates)).collect()).collect();
        if self.dead() {
            return nans();
        }
        let all_greedy = greedy.len() == reqs.len() && greedy.iter().all(|&g| g);
        if !all_greedy {
            return self.step(reqs, candidates);
        }
        let t = std::time::Instant::now();
        let run = (|| -> Result<_, ModelError> {
            let b = RowBatch::from_requests(reqs);
            if let Some(c) = self.m.window_candidates_dv(&b, candidates)? {
                debug_assert_eq!(c.count, 1, "the gather returns one candidate per row");
                return Ok((0..reqs.len())
                    .map(|i| b.seq_rows(i).map(|r| RowOutput { ids: vec![c.ids[r] as u32], values: vec![c.vals[r]] }).collect())
                    .collect());
            }
            // fallback: the full readback, exactly like `step`
            let c = self.m.window_candidates(&b, candidates)?;
            Ok((0..reqs.len()).map(|i| b.seq_rows(i).map(|r| row_of(&c, r, candidates)).collect()).collect())
        })();
        match run {
            Ok(rows) => {
                self.spec_window(reqs, t.elapsed().as_secs_f64() * 1e3);
                rows
            }
            Err(e) => {
                self.record("window step", &e);
                nans()
            }
        }
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        if self.dead() {
            return;
        }
        self.last_acc.insert(slot, keep.saturating_sub(1));
        if let Some(s) = self.spec.as_mut() {
            s.on_commit(slot, keep);
        }
        self.m.commit(slot, keep);
    }
    fn release(&mut self, slot: usize) {
        self.records.remove(&slot); // a readout that was cancelled or failed leaves its records behind
        self.last_acc.remove(&slot);
        self.spec_begin(slot);
        if !self.dead() {
            self.m.end_slot(slot);
        }
    }

    fn propose_batch(&mut self, reqs: &[ProposeRequest<'_>]) -> Vec<Option<Vec<u32>>> {
        self.propose_batch_scored(reqs).into_iter().map(|p| p.map(|p| p.tokens)).collect()
    }

    fn graph_counters(&self) -> Option<gb_engine::GraphCounters> {
        self.m.graph_stats().map(|s| s.counters())
    }

    fn propose_batch_scored(&mut self, reqs: &[ProposeRequest<'_>]) -> Vec<Option<Proposal>> {
        if !self.dead() && self.m.has_dflash() {
            return self.propose_dflash(reqs);
        }
        if self.dead() || self.mtp_k == 0 || reqs.len() != 1 {
            return vec![None; reqs.len()];
        }
        let r = &reqs[0];
        let k = if self.adaptive {
            self.last_acc.get(&r.slot).copied().unwrap_or(2).saturating_add(2).min(self.mtp_k)
        } else {
            self.mtp_k
        };
        let k = k.min(r.max);
        if k == 0 {
            return vec![None];
        }
        let result = self.m.mtp_propose(r.slot, r.tokens, k);
        match result {
            Ok(mut tokens) if !tokens.is_empty() => {
                tokens.truncate(k);
                vec![Some(Proposal { tokens, ..Default::default() })]
            }
            Ok(_) => vec![None],
            Err(e) => {
                self.record(&format!("draft of slot {}", r.slot), &e);
                vec![None]
            }
        }
    }
}

#[cfg(test)]
#[path = "backend_tests.rs"]
mod tests;
