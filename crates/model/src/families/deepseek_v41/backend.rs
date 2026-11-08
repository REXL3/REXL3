//! `gb_engine::Backend` over DeepSeek-V4.1-Flash.
//!
//! The trait's methods cannot fail, so a GPU or request error is logged, answered with NaN candidate rows (the engine's sampler refuses NaN)
//! and recorded for `Backend::take_error`, which the engine reads after every call: a recoverable error fails the sequences the call served,
//! a fatal one (a sticky CUDA error, lost state) stops the engine and the process. Once a fatal error has been seen it is latched and no
//! further call touches the device.
//!
//! A prompt is fed in chunks of [`Dsv41Backend::chunk`] rows (up to the model's chunk capacity), one chunk per `prefill_slice` so the engine
//! can run a decode round for the other sequences in between. Every chunk commits as it goes, and the model's arithmetic is row-invariant, so
//! the bits of a prompt row do not depend on how the prompt was cut.
use super::model::{Dsv41Model, Head, DEFAULT_PREFIX_SNAPSHOTS};
use super::tp_runtime::Dsv41Runtime;
use crate::error::ModelError;
use crate::forward::Candidates;
use crate::rows::RowBatch;
use gb_engine::{
    Backend, BackendError, CacheBudget, Captured, CutStep, ImageSpan, PrefillStep, PrefixCaps, PrefixHandle, Proposal,
    ProposeRequest, RowOutput, RowRequest, TreeRequest,
};
use gb_spec::SpecMode;

/// The prefix cache path id for DeepSeek-V4.1-Flash.
pub const PATH_ID: u32 = 1;

/// Estimated bytes per host-RAM snapshot slot (~8 MiB at 4K tokens).
pub const ESTIMATED_SNAPSHOT_BYTES: u64 = 8 << 20;

/// Compute the prefix cache namespace hash for this pack, KV format, and MTP configuration.
pub fn prefix_namespace(target_fingerprint: &str, kv_fmt: &str, mtp: bool) -> u64 {
    let text =
        format!("rexl3-dsv41-prefix-cache-v1|{target_fingerprint}|kv={kv_fmt}|mtp={}", if mtp { "dspark" } else { "none" });
    text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

/// Slots of the host-RAM snapshot pool. REXL3 field patch P2: `REXL3_DSV41_PREFIX_SNAPSHOTS` sizes it (default
/// [`DEFAULT_PREFIX_SNAPSHOTS`]). Every rank of a group reads its own, says it in the Hello (`Dsv41Facts::cache_slots`),
/// and is then sized to rank 0's count.
pub fn prefix_snapshot_slots() -> usize {
    gb_cuda::dsv4_moe::runtime_env("REXL3_DSV41_PREFIX_SNAPSHOTS")
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(DEFAULT_PREFIX_SNAPSHOTS)
}

/// The backend over the model surface a single rank or a group leader shares ([`Dsv41Runtime`]): the plain
/// [`Dsv41Model`] on one node, the lockstep leader on rank 0 of a tensor-parallel group (every state-changing call
/// below is forwarded to the followers and applied on both ranks — the backend is the same code either way).
pub struct Dsv41Backend<R: Dsv41Runtime = Dsv41Model> {
    pub m: R,
    /// The first recoverable error since `take_error` last ran.
    error: Option<BackendError>,
    /// The first fatal error: it never clears.
    fatal: Option<BackendError>,
    /// Rows of one prefill slice.
    chunk: usize,
    /// The full-vocabulary logits of every row of the prompt being scored, when the score request asked for all rows.
    score_rows: Vec<Vec<f32>>,
    /// Whether the score request in flight wants [`Self::score_rows`] (see `Backend::score_all_rows`).
    all_rows: bool,
    spec: SpecMode,
    drafting: bool,
    /// The tree round `step_tree` planned and verified, until its `commit_path` (or the next round, or the
    /// slot's release) closes it (T2.5a lane 2).
    tree: Option<super::tree::TreeWindows>,
    prefix: Option<PrefixCaps>,
}

fn nan_row(n: usize) -> RowOutput {
    RowOutput { ids: (0..n as u32).collect(), values: vec![f32::NAN; n] }
}

fn row_of(c: &Candidates, r: usize, n: usize) -> RowOutput {
    let ids = c.ids[r * c.count..r * c.count + n].iter().map(|&x| x as u32).collect();
    let values = c.vals[r * c.count..r * c.count + n].to_vec();
    RowOutput { ids, values }
}

impl<R: Dsv41Runtime> Dsv41Backend<R> {
    pub fn new(m: R) -> Self {
        let chunk = m.chunk_rows();
        Self {
            m,
            error: None,
            fatal: None,
            chunk,
            score_rows: Vec::new(),
            all_rows: false,
            spec: SpecMode::Auto,
            drafting: false,
            tree: None,
            prefix: None,
        }
    }

    pub fn with_spec(mut self, spec: SpecMode, drafting: bool) -> Self {
        self.spec = spec;
        self.drafting = drafting;
        self
    }

    pub fn with_prefix_cache(mut self, namespace: u64, budget: CacheBudget) -> Self {
        if budget.enabled() {
            self.m.enable_prefix_cache(budget.snapshot_slots);
            self.prefix = Some(PrefixCaps { namespace, budget });
        } else {
            self.m.enable_prefix_cache(0);
            self.prefix = None;
        }
        self
    }

    pub fn with_default_prefix_cache(self, namespace: u64) -> Self {
        let snaps = prefix_snapshot_slots();
        eprintln!("rexl3 event=dsv41_prefix_snapshots slots={snaps} (field patch P2)");
        let budget = CacheBudget {
            snapshot_slots: snaps,
            page_blocks: self.m.max_context().div_ceil(crate::kvtables::BLOCK_TOKENS) * snaps,
            snapshot_bytes: ESTIMATED_SNAPSHOT_BYTES,
            page_bytes: 0,
        };
        self.with_prefix_cache(namespace, budget)
    }

    pub fn with_chunk(mut self, rows: usize) -> Self {
        self.chunk = rows.clamp(1, self.m.chunk_rows());
        self
    }

    pub fn chunk(&self) -> usize {
        self.chunk
    }

    fn dead(&self) -> bool {
        self.fatal.is_some()
    }

    fn record(&mut self, what: &str, e: &ModelError) {
        let fatal = e.is_fatal();
        let message = format!("{what}: {e}");
        eprintln!("rexl3 level={} event=backend_error {message}", if fatal { "fatal" } else { "error" });
        if fatal {
            self.fatal.get_or_insert(BackendError::fatal(message));
        } else if self.error.is_none() {
            self.error = Some(BackendError::retryable(message));
        }
    }

    fn drop_slot(&mut self, slot: usize) {
        self.m.end_slot(slot);
    }

    /// The image spans of the chunk `at..end`, as the splices the model writes after its embedding: each span's
    /// features are downcast (they were made by this family's encoder), checked against the span the prompt
    /// reserves for them, and clipped to the chunk. Empty when the prompt has no images, or none in this chunk.
    fn image_splices(
        &self,
        prompt: &[u32],
        images: &[ImageSpan],
        at: usize,
        end: usize,
    ) -> Result<Vec<super::vision_serve::Dsv41Splice>, ModelError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let placeholder = self.m.image_token_id();
        let mut whole = Vec::with_capacity(images.len());
        let mut reached = 0usize;
        for im in images {
            let f = std::sync::Arc::clone(&im.features)
                .downcast::<super::vision_serve::Dsv41ImageFeatures>()
                .map_err(|_| ModelError::Request("image features were not produced by this backend's encoder".into()))?;
            if f.tokens != im.tokens || f.merged != im.merged {
                return Err(ModelError::Request(format!(
                    "image has {} encoder rows ({}x{}) for {} placeholders ({}x{})",
                    f.tokens, f.merged.0, f.merged.1, im.tokens, im.merged.0, im.merged.1
                )));
            }
            if im.start < reached || im.start + im.tokens > prompt.len() {
                return Err(ModelError::Request(format!(
                    "image span at {} of {} tokens does not fit the prompt of {} tokens",
                    im.start,
                    im.tokens,
                    prompt.len()
                )));
            }
            if prompt[im.start..im.start + im.tokens].iter().any(|&x| x != placeholder) {
                return Err(ModelError::Request(format!(
                    "the {} tokens at {} are not all the image placeholder {placeholder}",
                    im.tokens, im.start
                )));
            }
            reached = im.start + im.tokens;
            whole.push(super::vision_serve::Dsv41Splice::whole(f, im.start)?);
        }
        Ok(whole.into_iter().filter_map(|sp| sp.clip(at, end)).collect())
    }

    fn compute_slice_end(&self, done: usize, n: usize, cuts: &[usize]) -> usize {
        let mut slice_end = (done + self.chunk).min(n);
        if slice_end < n {
            if let Some(&last_in_slice) = cuts.iter().filter(|&&c| c > done && c <= slice_end && c < n).max() {
                return last_in_slice;
            }
            let next_target = cuts.iter().copied().find(|&c| c > slice_end && c < n).unwrap_or(n);
            // Only back up toward a cut that actually has a snapshot (next_target < n). At the prompt's
            // end there is no cut to snapshot; reshaping the tail by the chunk grid produced chunk-size-
            // dependent final slices (incl. a lone 1-row slice at chunk 128) -> tail bits differed. The
            // batched prefill is row-invariant across every alignment, so a plain remainder final slice
            // is exact (and the CED tail splice re-assembles short remainders identically across grids).
            if next_target < n && next_target - slice_end < 128 && next_target > 128 {
                let target = (next_target - 128) & !1;
                if target > done && target < slice_end {
                    slice_end = target;
                }
            }
        }
        slice_end
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_slice_at(
        &mut self,
        slot: usize,
        prompt: &[u32],
        images: &[ImageSpan],
        done: usize,
        candidates: usize,
        cuts: &[usize],
        capture: bool,
    ) -> CutStep {
        let n = prompt.len();
        if self.dead() {
            return CutStep { step: PrefillStep { done: n, row: Some(nan_row(candidates)) }, captured: Vec::new() };
        }
        let slice_end = self.compute_slice_end(done, n, cuts);
        let mut taken: Vec<u32> = Vec::new();
        let mut captured: Vec<Captured> = Vec::new();
        let has_prefix = self.prefix.is_some();
        let mut run = |taken: &mut Vec<u32>, captured: &mut Vec<Captured>| -> Result<PrefillStep, ModelError> {
            if done == 0 {
                self.m.begin_slot(slot)?;
            }
            let cap_limit = if !images.is_empty() {
                0
            } else if !capture {
                cuts.len()
            } else if has_prefix {
                self.m.prefix_free_snapshots()
            } else {
                0
            };
            let mut eligible_cuts: Vec<usize> = cuts
                .iter()
                .copied()
                .filter(|&c| c > done && c <= slice_end && c < n && (!capture || self.m.can_snapshot_prefix(c)))
                .collect();
            let active_cuts: Vec<usize> = if eligible_cuts.len() > cap_limit {
                eligible_cuts.split_off(eligible_cuts.len() - cap_limit)
            } else {
                eligible_cuts
            };
            let mut waypoints = active_cuts.clone();
            if waypoints.last().copied() != Some(slice_end) {
                waypoints.push(slice_end);
            }
            let mut cur = done;
            let mut last_cand = None;
            for &seg_end in &waypoints {
                let is_cut = active_cuts.contains(&seg_end);
                let want = (seg_end == n).then_some(candidates);
                let need_window = is_cut;
                let splices = self.image_splices(prompt, images, cur, seg_end)?;
                last_cand = if splices.is_empty() {
                    self.m.prefill_chunk_ex(slot, cur, &prompt[cur..seg_end], want, need_window)?
                } else {
                    self.m.prefill_chunk_mm(slot, cur, &prompt[cur..seg_end], want, splices)?
                };
                if is_cut && capture && self.m.can_snapshot_prefix(seg_end) {
                    if let Some(snap) = self.m.alloc_snapshot() {
                        taken.push(snap);
                        match self.m.prefix_snapshot(slot, snap, seg_end) {
                            Ok(()) => {
                                taken.clear();
                                captured.push(Captured { len: seg_end, handle: PrefixHandle { snap, pages: Vec::new() } });
                            }
                            Err(_) => {
                                taken.clear();
                                self.m.release_prefix(snap);
                            }
                        }
                    }
                }
                cur = seg_end;
            }
            Ok(PrefillStep { done: slice_end, row: last_cand.map(|c| row_of(&c, 0, candidates)) })
        };
        match run(&mut taken, &mut captured) {
            Ok(step) => CutStep { step, captured },
            Err(e) => {
                for snap in taken {
                    self.m.release_prefix(snap);
                }
                for c in captured {
                    self.m.release_prefix(c.handle.snap);
                }
                self.record(&format!("prefill of slot {slot}"), &e);
                self.drop_slot(slot);
                CutStep { step: PrefillStep { done: n, row: Some(nan_row(candidates)) }, captured: Vec::new() }
            }
        }
    }
}

impl<R: Dsv41Runtime> Backend for Dsv41Backend<R> {
    fn quiesce(&mut self) -> Result<(), String> {
        gb_cuda::rt::device_sync().map_err(|e| e.to_string())
    }

    fn vocab_size(&self) -> usize {
        self.m.vocab_size()
    }

    fn score_supported(&self) -> bool {
        true
    }

    fn score_all_rows(&mut self, on: bool) {
        self.all_rows = on;
        self.score_rows = Vec::new();
    }

    fn take_score_rows(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.score_rows)
    }

    /// One score slice. `all_rows` keeps every row's full-vocabulary logits (the oracle's teacher-forced readout);
    /// otherwise only the prompt's last row is read out, which is what decisions scoring asks for.
    fn score_slice(
        &mut self,
        slot: usize,
        prompt: &[u32],
        done: usize,
        cuts: &[usize],
    ) -> Result<gb_engine::score::ScoreStep, BackendError> {
        if let Some(e) = self.fatal.clone() {
            return Err(e);
        }
        let n = prompt.len();
        let cuts = if self.prefix.is_some() { cuts } else { &[] };
        let slice_end = self.compute_slice_end(done, n, cuts);
        let all_rows = self.all_rows;
        let has_prefix = self.prefix.is_some();
        let mut taken: Vec<u32> = Vec::new();
        let mut captured: Vec<Captured> = Vec::new();
        let mut run = |taken: &mut Vec<u32>, captured: &mut Vec<Captured>| -> Result<Vec<Vec<f32>>, ModelError> {
            if done == 0 {
                self.m.begin_slot(slot)?;
            }
            let free = if has_prefix { self.m.prefix_free_snapshots() } else { 0 };
            let active_cuts: Vec<usize> = cuts
                .iter()
                .copied()
                .filter(|&c| c > done && c <= slice_end && c < n && self.m.can_snapshot_prefix(c))
                .take(free)
                .collect();
            let mut waypoints = active_cuts.clone();
            if waypoints.last().copied() != Some(slice_end) {
                waypoints.push(slice_end);
            }
            let mut cur = done;
            let mut rows_out = Vec::new();
            for &seg_end in &waypoints {
                let is_cut = active_cuts.contains(&seg_end);
                if all_rows {
                    let r = self.m.prefill_logits_rows(slot, cur, &prompt[cur..seg_end])?;
                    rows_out.extend(r);
                } else if seg_end == n {
                    let r = self.m.prefill_logits_last_row(slot, cur, &prompt[cur..seg_end])?;
                    rows_out.push(r);
                } else {
                    let need_window = is_cut;
                    let head = if need_window { Head::WindowOnly } else { Head::None };
                    self.m.prefill_rows(slot, cur, &prompt[cur..seg_end], head)?;
                }
                if is_cut && self.m.can_snapshot_prefix(seg_end) {
                    if let Some(snap) = self.m.alloc_snapshot() {
                        taken.push(snap);
                        match self.m.prefix_snapshot(slot, snap, seg_end) {
                            Ok(()) => {
                                taken.clear();
                                captured.push(Captured { len: seg_end, handle: PrefixHandle { snap, pages: Vec::new() } });
                            }
                            Err(_) => {
                                taken.clear();
                                self.m.release_prefix(snap);
                            }
                        }
                    }
                }
                cur = seg_end;
            }
            Ok(rows_out)
        };
        match run(&mut taken, &mut captured) {
            Ok(rows) => {
                let last = rows.last().cloned().unwrap_or_default();
                if all_rows {
                    self.score_rows.extend(rows);
                }
                Ok(gb_engine::score::ScoreStep { done: slice_end, row: (slice_end == n).then_some(last), captured })
            }
            Err(e) => {
                for snap in taken {
                    self.m.release_prefix(snap);
                }
                for c in captured {
                    self.m.release_prefix(c.handle.snap);
                }
                self.record(&format!("score of slot {slot}"), &e);
                self.drop_slot(slot);
                Err(self.fatal.clone().unwrap_or_else(|| BackendError::retryable(e.to_string())))
            }
        }
    }

    fn take_error(&mut self) -> Option<BackendError> {
        self.fatal.clone().or_else(|| self.error.take())
    }

    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.prefill_piece(slot, start, tokens, start as usize + tokens.len(), candidates)
    }

    fn prefill_piece(&mut self, slot: usize, start: u64, tokens: &[u32], total: usize, candidates: usize) -> RowOutput {
        let _ = total; // the arithmetic does not depend on the prompt's length
        if self.dead() {
            return nan_row(candidates);
        }
        let chunk = self.chunk;
        let mut run = || -> Result<RowOutput, ModelError> {
            if start == 0 {
                self.m.begin_slot(slot)?;
            }
            let (mut pos, n) = (start as usize, tokens.len());
            let mut at = 0;
            let mut last = None;
            while at < n {
                let end = (at + chunk).min(n);
                let want = (end == n).then_some(candidates);
                last = self.m.prefill_chunk(slot, pos, &tokens[at..end], want)?;
                pos += end - at;
                at = end;
            }
            let c = last.ok_or_else(|| ModelError::Request("empty prefill".into()))?;
            Ok(row_of(&c, 0, candidates))
        };
        match run() {
            Ok(r) => r,
            Err(e) => {
                self.record(&format!("prefill of slot {slot}"), &e);
                self.drop_slot(slot);
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
        let step = self.prefill_slice_at(slot, prompt, images, done, candidates, &[], false).step;
        step
    }

    fn prefix_caps(&self) -> Option<PrefixCaps> {
        self.prefix
    }

    fn prefix_path(&self, _total: usize) -> Option<u32> {
        self.prefix.as_ref().map(|_| PATH_ID)
    }

    fn prefix_free_snapshots(&self) -> usize {
        if self.prefix.is_some() {
            self.m.prefix_free_snapshots()
        } else {
            0
        }
    }

    fn prefix_restore(&mut self, slot: usize, handle: &PrefixHandle, len: usize) -> Result<(), String> {
        if self.dead() {
            return Err("the backend has failed".into());
        }
        if self.prefix.is_none() {
            return Err("this backend has no prefix cache".into());
        }
        match self.m.restore_prefix(slot, handle.snap, len) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.record(&format!("prefix restore into slot {slot}"), &e);
                self.drop_slot(slot);
                Err(e.to_string())
            }
        }
    }

    fn prefix_release(&mut self, handle: PrefixHandle) {
        if self.prefix.is_some() {
            self.m.release_prefix(handle.snap);
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
        let cuts = if self.prefix.is_some() && images.is_empty() { cuts } else { &[] };
        self.prefill_slice_at(slot, prompt, images, done, candidates, cuts, true)
    }

    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        let nans = || reqs.iter().map(|q| q.tokens.iter().map(|_| nan_row(candidates)).collect()).collect();
        if self.dead() {
            return nans();
        }
        let b = RowBatch::from_requests(reqs);
        match self.m.window_candidates(&b, candidates) {
            Ok(c) => (0..reqs.len()).map(|i| b.seq_rows(i).map(|r| row_of(&c, r, candidates)).collect()).collect(),
            Err(e) => {
                self.record("window step", &e);
                nans()
            }
        }
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        self.m.commit(slot, keep);
    }

    /// The trunk verifies DRAFT TREES on this model (T2.5a): every layer's window ring is fp16, the surface
    /// the round's commit surgery runs on. The knob that routes rounds through the trees is the family
    /// engine's (`Engine::with_trees`, `REXL3_DSV41_SPEC_TREE`); without it the engine keeps chain rounds and
    /// this is never called.
    fn supports_trees(&self) -> bool {
        self.m.trees_capable()
    }

    /// One tree window, as one serial-exact CHAIN window per root-to-leaf path (`super::tree`): the family's
    /// window is chain-causal, so the paths run one after another over the per-round base state (the model's
    /// host isolates each one: the engram ring and the compressor carry restored before it runs), each node's
    /// output read from its designated path's window. The CHAIN requests of the same call run LAST as one
    /// plain window — exactly `step` — so the model's last window serves their own commits.
    fn step_tree(&mut self, reqs: &[TreeRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        let nans = || reqs.iter().map(|q| q.tokens.iter().map(|_| nan_row(candidates)).collect()).collect();
        if self.dead() {
            return nans();
        }
        // a new round supersedes an uncommitted one: put the model's base state back first
        if self.tree.take().is_some() {
            if let Err(e) = super::tree::TreeRoundHost::tree_drop_round(&mut self.m) {
                self.record("tree round drop", &ModelError::Request(e));
            }
        }
        let mut out: Vec<Option<Vec<RowOutput>>> = vec![None; reqs.len()];
        let trees: Vec<(usize, &TreeRequest)> = reqs.iter().enumerate().filter(|(_, r)| !r.is_chain()).collect();
        if trees.len() > 1 || trees.first().is_some_and(|(_, r)| r.slot != 0) {
            // only slot 0 drafts a lattice (the DSpark round is single-sequence); anything else is ours to refuse
            let what = if trees.len() > 1 {
                format!("a tree step of {} tree windows (this trunk verifies one)", trees.len())
            } else {
                format!("a tree window for slot {} (this trunk verifies trees on slot 0)", trees[0].1.slot)
            };
            self.record("tree window step", &ModelError::Request(what));
        } else if let Some(&(i, req)) = trees.first() {
            let planned =
                super::tree::TreeWindows::plan(&req.tokens, &req.parents, req.first_position, super::model::WINDOW_ROWS)
                    .and_then(|plan| super::tree::verify_tree(&mut self.m, &plan, candidates).map(|rows| (plan, rows)));
            match planned {
                Ok((plan, rows)) => {
                    self.tree = Some(plan);
                    out[i] = Some(rows);
                }
                Err(e) => self.record("tree window step", &ModelError::Request(e)),
            }
        }
        // the chain requests of the same call ride one plain window, LAST (see the method doc)
        let chains: Vec<RowRequest> = reqs
            .iter()
            .filter(|r| r.is_chain())
            .map(|r| RowRequest { slot: r.slot, tokens: r.tokens.clone(), first_position: r.first_position })
            .collect();
        if !chains.is_empty() {
            let chain_out = self.step(&chains, candidates);
            let mut k = 0;
            for (i, r) in reqs.iter().enumerate() {
                if r.is_chain() {
                    out[i] = Some(chain_out[k].clone());
                    k += 1;
                }
            }
        }
        out.into_iter().zip(reqs).map(|(o, r)| o.unwrap_or_else(|| vec![nan_row(candidates); r.tokens.len().max(1)])).collect()
    }

    /// A chain window's commit is the trait's own (a prefix); a TREE window's is the surgery the model runs
    /// (`tree_commit_window`: the window rings, the engram rings, the compressor carry and the DSpark tap
    /// reconciled to the serial replay of the accepted path). A failed tree commit leaves state out of step
    /// with the committed tokens: fatal, like the Qwen trunk's.
    fn commit_path(&mut self, slot: usize, path: &[usize]) {
        if slot == 0 {
            if let Some(plan) = self.tree.take() {
                match super::tree::commit_round(&mut self.m, &plan, path) {
                    Ok(()) => {}
                    Err(e) => {
                        // the state may be half-reconciled: this sequence ends here, and the error is fatal
                        self.record(&format!("commit of a tree path in slot {slot}"), &ModelError::Request(e));
                        self.fatal.get_or_insert(BackendError::fatal("a tree path commit left the model state out of step"));
                    }
                }
                return;
            }
        }
        if !path.iter().enumerate().all(|(i, &r)| i == r) {
            self.record(
                &format!("commit of a path in slot {slot}"),
                &ModelError::Request(format!("not a chain prefix: {path:?}")),
            );
            self.fatal.get_or_insert(BackendError::fatal("a chain commit path that is not a prefix"));
            return;
        }
        self.m.commit(slot, path.len());
    }

    fn propose_batch_scored(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Proposal>> {
        if self.dead() || !self.drafting || !matches!(self.spec, SpecMode::Auto | SpecMode::Mtp) || !self.m.has_mtp() {
            return vec![None; reqs.len()];
        }
        let mut out = Vec::with_capacity(reqs.len());
        for r in reqs {
            let max_d = if reqs.len() <= 1 {
                r.max
            } else {
                ((super::model::WINDOW_ROWS / reqs.len()).saturating_sub(1)).min(super::model::MTP_MAX_DRAFTS).min(r.max)
            };
            match self.m.propose_mtp_sampled(r.slot, r.tokens, r.sampling, max_d) {
                Ok(Some((tokens, accept, lattice))) => {
                    out.push(Some(Proposal { tokens, accept, lattice, ..Default::default() }));
                }
                Ok(None) => out.push(None),
                Err(e) => {
                    self.record("mtp draft", &e);
                    out.push(None);
                }
            }
        }
        out
    }

    fn propose_batch(&mut self, reqs: &[ProposeRequest]) -> Vec<Option<Vec<u32>>> {
        self.propose_batch_scored(reqs).into_iter().map(|p| p.map(|p| p.tokens)).collect()
    }

    fn release(&mut self, slot: usize) {
        if slot == 0 {
            self.tree = None;
        }
        self.drop_slot(slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_namespace_separates_packs_formats_and_mtp() {
        let a = prefix_namespace("sha256:aaaa", "q4", true);
        assert_eq!(a, prefix_namespace("sha256:aaaa", "q4", true));
        assert_ne!(a, prefix_namespace("sha256:bbbb", "q4", true));
        assert_ne!(a, prefix_namespace("sha256:aaaa", "f16", true));
        assert_ne!(a, prefix_namespace("sha256:aaaa", "q4", false));
    }
}
