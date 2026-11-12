//! The `gb_engine::Backend` of the Bailing family: prefill chunks, decode/verify windows, the lazy KDA commit, the
//! native `L72` MTP drafter and the prefix cache (snapshot at a cut, restore from a hit).

use super::model::BailingModel;
use crate::forward::Candidates;
use crate::rows::RowBatch;
use gb_engine::{
    Backend, BackendError, CacheBudget, Captured, CutStep, ImageSpan, PrefillStep, PrefixCaps, PrefixHandle, RowOutput,
    RowRequest,
};
use gb_sampling::Sampling;

fn nan_row(n: usize) -> RowOutput {
    RowOutput { ids: (0..n as u32).collect(), values: vec![f32::NAN; n] }
}

fn row_of(c: &Candidates, r: usize, n: usize) -> RowOutput {
    let ids = c.ids[r * c.count..r * c.count + n].iter().map(|&x| x as u32).collect();
    let values = c.vals[r * c.count..r * c.count + n].to_vec();
    RowOutput { ids, values }
}

/// The prefill path id of this family: one arithmetic for every prompt length (a prefill chunk is the trunk
/// kernels' bits at any row count), so a snapshot is interchangeable across totals.
pub const PATH_ID: u32 = 1;

/// Tokens the engine rounds a planned prefix cut down to a multiple of. A KDA prefix snapshot resumes by running
/// `kda::window_chunked` over the tail, whose recurrence advances in 16-row chunks: a cut on that grid means the resumed
/// chunk never re-derives a partial chunk of an already-snapshotted prefix. (KV pages of this family are whole-page
/// snapshots already; this is the KDA chunk grid, not the page size.)
pub const CUT_ALIGN: usize = super::alloc::PREFILL_ALIGN;

/// Bytes of one host-RAM snapshot slot apart from its MLA page copies: every KDA layer's committed state as the pool holds it (both
/// state buffers, the conv history and the parity word: `kda::Dims::slot_bytes`, 8.14 MiB a layer on Ling-3.1) and, when the MTP head
/// drafts, its tap ring (`CAP_TAPS` bf16 rows, 64 MiB): 576.9 MiB for Ling-3.1 with MTP. A snapshot also copies the
/// MLA pages of its prefix to host RAM (one KV page's bytes per 256 tokens, `MlaCacheSnap`); those copies are the cache's pages in the
/// plan (`engine::build_engine`), so this is the plan's snapshot slot.
pub fn snapshot_fixed_bytes(cfg: &super::config::BailingConfig, with_mtp: bool) -> u64 {
    let dims = gb_cuda::kda::Dims { heads: cfg.kda.heads, hidden: cfg.hidden };
    let kda = cfg.kda_layers().len() as u64 * dims.slot_bytes() as u64;
    let mtp = if with_mtp { (super::model::CAP_TAPS * cfg.hidden * 2) as u64 } else { 0 };
    kda + mtp
}

/// Whether a snapshot at `cut` of a prefill segment that starts at `start` lies on the segment's KDA chunk grid (a whole number of
/// `engine::CUT_GRID`-row chunks after `start`): the prefill's chunks start there too, so the snapshot holds a cold prefill's bits.
pub fn on_cut_grid(cut: usize, start: usize) -> bool {
    cut >= start && (cut - start) % super::engine::CUT_GRID == 0
}

/// The prefix cache namespace of this pack pair: everything a snapshot's bytes depend on besides the tokens
/// (the pack's fingerprint, whether the MTP drafter is on, the page size, the cache formats, and the
/// Gated-MLA arithmetic contract — P2's tensor-core bits differ from P1's, so a snapshot written under one
/// contract must never be restored under the other).
pub fn prefix_namespace(target_fingerprint: &str, mtp: bool) -> u64 {
    let c = super::contract::current();
    let text = format!(
        "rexl3-bailing-prefix-cache-v1|{target_fingerprint}|mtp={}|page={}|kv=bf16|state=f32+bf16|mla={}{}",
        if mtp { "l72" } else { "none" },
        crate::kvtables::BLOCK_TOKENS,
        c.id(),
        c.arith_tag(),
    );
    text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

pub struct BailingBackend {
    pub model: BailingModel,
    error: Option<BackendError>,
    fatal: Option<BackendError>,
    /// The MTP native drafter is on when the model carries the L72 layer (engine config turns drafting on).
    with_mtp: bool,
    /// The prefix cache this backend reserved (its namespace and budget), when one is enabled.
    prefix: Option<PrefixCaps>,
}

impl BailingBackend {
    pub fn new(model: BailingModel, with_mtp: bool) -> Self {
        Self { model, error: None, fatal: None, with_mtp, prefix: None }
    }

    /// Enable the prefix cache: allocates `budget.snapshot_slots` snapshot slots. `namespace` binds every key to what
    /// the snapshots depend on (see `serve::prefix_namespace`). The KV pool must already hold `budget.page_blocks`
    /// pages more than admission hands out (the engine's plan does that).
    pub fn with_prefix_cache(mut self, namespace: u64, budget: CacheBudget) -> Result<Self, crate::error::ModelError> {
        if budget.enabled() {
            self.model.enable_prefix_cache(budget.snapshot_slots);
            self.prefix = Some(PrefixCaps { namespace, budget });
        }
        Ok(self)
    }

    fn record(&mut self, what: &str, e: &crate::error::ModelError, fatal_anyway: bool) {
        let fatal = fatal_anyway || e.is_fatal();
        let message = format!("{what}: {e}");
        eprintln!("rexl3 level={} event=backend_error {message}", if fatal { "fatal" } else { "error" });
        if fatal {
            self.fatal.get_or_insert(BackendError::fatal(message));
        } else if self.error.is_none() {
            self.error = Some(BackendError::retryable(message));
        }
    }

    fn dead(&self) -> bool {
        self.fatal.is_some()
    }

    fn drop_slot(&mut self, slot: usize) {
        self.model.end_slot(slot);
    }

    /// Prefill `prompt` from `done` through `n`, taking a snapshot of the slot at each of `cuts` (ascending, absolute)
    /// it passes when `capture` is set. The whole remaining prompt is prefilled in one call (chunked inside the model,
    /// `PREFILL_ROWS` rows at a time); the cuts only add the snapshot points.
    fn prefill_at(
        &mut self,
        slot: usize,
        prompt: &[u32],
        done: usize,
        candidates: usize,
        cuts: &[usize],
        capture: bool,
    ) -> CutStep {
        let n = prompt.len();
        if self.dead() {
            return CutStep { step: PrefillStep { done: n, row: Some(nan_row(candidates)) }, captured: Vec::new() };
        }
        let mut taken: Vec<u32> = Vec::new();
        let mut captured: Vec<Captured> = Vec::new();
        let result = (|| -> Result<PrefillStep, crate::error::ModelError> {
            if done == 0 {
                self.model.begin_slot(slot)?;
            }
            if done >= n {
                // a full hit (restored exactly to the prompt's end): no tokens to prefill, and the restored state already
                // holds the last row, so it must not run again (a window of it would step every KDA state and conv history
                // past it once more and rewrite its MLA rows from those). Its candidates are the head over its normed trunk
                // row, which the restored MTP tap ring holds: the bytes and the kernel of the cold prefill's last row.
                let c = self.model.tap_candidates(slot, n - 1, candidates)?;
                return Ok(PrefillStep { done: n, row: Some(row_of(&c, 0, candidates)) });
            }
            let has_prefix = self.prefix.is_some();
            let cap_limit = if !capture {
                cuts.len()
            } else if has_prefix {
                // at most one snapshot cut per prompt: the furthest eligible cut. More cuts would split the prompt into
                // extra sub-calls and download redundant KDA states (each snapshot is ~540 MiB) for no cache benefit
                self.model.prefix_free_snapshots().min(1)
            } else {
                0
            };
            // only cuts on the segment's KDA chunk grid (`engine::CUT_GRID` rows from its first row, `done`): a snapshot off the
            // grid would resume with other bits than a cold prefill (the engine's cuts are on it already, `engine::prefix_config`)
            let mut eligible: Vec<usize> = Vec::new();
            // cuts the host-memory guard declined (`BailingModel::can_snapshot_prefix`): counted, never silent
            let mut declined = 0u64;
            let mut first_declined: Option<super::model::SnapshotHeadroom> = None;
            for &c in cuts {
                if !(c > done && c <= n && on_cut_grid(c, done)) {
                    continue;
                }
                if capture && !self.model.can_snapshot_prefix(c) {
                    if let Some(h) = self.model.snapshot_headroom(c).filter(|h| h.avail < h.needed) {
                        declined += 1;
                        first_declined.get_or_insert(h);
                    }
                    continue;
                }
                eligible.push(c);
            }
            if let Some(h) = first_declined {
                super::counters::note_skipped_low_memory(declined, h.avail, h.needed);
            }
            let active: Vec<usize> =
                if eligible.len() > cap_limit { eligible.split_off(eligible.len() - cap_limit) } else { eligible };
            let mut waypoints = active.clone();
            if waypoints.last().copied() != Some(n) {
                waypoints.push(n);
            }
            let mut cur = done;
            let mut last = None;
            for &seg_end in &waypoints {
                let is_cut = active.contains(&seg_end);
                let want = (seg_end == n).then_some(candidates);
                last = self.model.prefill_chunk(slot, cur, &prompt[cur..seg_end], want)?;
                if is_cut && capture && !self.model.can_snapshot_prefix(seg_end) {
                    // the guard passed when the cut was planned and fails now (memory moved while the segment prefilled)
                    if let Some(h) = self.model.snapshot_headroom(seg_end).filter(|h| h.avail < h.needed) {
                        super::counters::note_skipped_low_memory(1, h.avail, h.needed);
                    }
                }
                if is_cut && capture && self.model.can_snapshot_prefix(seg_end) {
                    if let Some(snap) = self.model.alloc_snapshot() {
                        taken.push(snap);
                        // pre-absorb L72 up to the cut so the snapshot's MTP state already has the absorbed pages
                        // (`absorbed = seg_end - 1`, `head_dirty = false`): a warm hit then only re-absorbs the tail,
                        // instead of re-absorbing the whole prompt inside the first `mtp_propose`
                        match self
                            .model
                            .mtp_absorb_up_to(slot, &prompt[..seg_end])
                            .and_then(|_| self.model.prefix_snapshot(slot, snap, seg_end))
                        {
                            Ok(()) => {
                                taken.clear();
                                captured.push(Captured { len: seg_end, handle: PrefixHandle { snap, pages: Vec::new() } });
                            }
                            Err(e) => {
                                // the cut is lost and the prompt goes on cold: counted and logged, not swallowed
                                super::counters::note_capture_failed(&e.to_string());
                                taken.clear();
                                self.model.release_prefix(snap);
                            }
                        }
                    }
                }
                cur = seg_end;
            }
            Ok(PrefillStep { done: n, row: last.map(|c| row_of(&c, 0, candidates)) })
        })();
        match result {
            Ok(step) => CutStep { step, captured },
            Err(e) => {
                for snap in taken {
                    self.model.release_prefix(snap);
                }
                for c in &captured {
                    self.model.release_prefix(c.handle.snap);
                }
                self.record(&format!("prefill of slot {slot}"), &e, false);
                self.drop_slot(slot);
                CutStep { step: PrefillStep { done: n, row: Some(nan_row(candidates)) }, captured: Vec::new() }
            }
        }
    }
}

impl Backend for BailingBackend {
    fn vocab_size(&self) -> usize {
        self.model.cfg.vocab
    }

    fn quiesce(&mut self) -> Result<(), String> {
        gb_cuda::rt::device_sync().map_err(|e| e.to_string())
    }

    fn take_error(&mut self) -> Option<BackendError> {
        self.fatal.clone().or_else(|| self.error.take())
    }

    fn prefill(&mut self, slot: usize, start: u64, tokens: &[u32], candidates: usize) -> RowOutput {
        self.prefill_piece(slot, start, tokens, start as usize + tokens.len(), candidates)
    }

    fn prefill_piece(&mut self, slot: usize, start: u64, tokens: &[u32], _total: usize, candidates: usize) -> RowOutput {
        if self.dead() {
            return nan_row(candidates);
        }
        let mut run = || -> Result<RowOutput, crate::error::ModelError> {
            if start == 0 {
                self.model.begin_slot(slot as usize)?;
            }
            let c = self
                .model
                .prefill_chunk(slot as usize, start as usize, tokens, Some(candidates))?
                .ok_or_else(|| crate::error::ModelError::Request("empty prefill".into()))?;
            let r = &c;
            Ok(RowOutput {
                ids: r.ids[..candidates.min(r.count)].iter().map(|&x| x as u32).collect(),
                values: r.vals[..candidates.min(r.count)].to_vec(),
            })
        };
        match run() {
            Ok(r) => r,
            Err(e) => {
                self.record(&format!("prefill of slot {slot}"), &e, false);
                self.drop_slot(slot as usize);
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
        debug_assert!(images.is_empty(), "bailing_moe_v3 has no vision");
        self.prefill_at(slot, prompt, done, candidates, &[], false).step
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
        self.prefill_at(slot, prompt, done, candidates, cuts, true)
    }

    fn step(&mut self, reqs: &[RowRequest], candidates: usize) -> Vec<Vec<RowOutput>> {
        let nans = || reqs.iter().map(|q| q.tokens.iter().map(|_| nan_row(candidates)).collect()).collect();
        if self.dead() {
            return nans();
        }
        let b = RowBatch::from_requests(reqs);
        match self.model.window_candidates(&b, candidates) {
            Ok(c) => {
                let out: Vec<Vec<RowOutput>> = (0..reqs.len())
                    .map(|i| {
                        b.seq_rows(i)
                            .map(|r| RowOutput {
                                ids: c.ids[r * c.count..r * c.count + candidates.min(c.count)]
                                    .iter()
                                    .map(|&x| x as u32)
                                    .collect(),
                                values: c.vals[r * c.count..r * c.count + candidates.min(c.count)].to_vec(),
                            })
                            .collect()
                    })
                    .collect();
                out
            }
            Err(e) => {
                self.record("window step", &e, false);
                nans()
            }
        }
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        self.model.commit(slot, keep);
    }

    fn release(&mut self, slot: usize) {
        self.drop_slot(slot);
    }

    fn propose(&mut self, slot: usize, tokens: &[u32], _sampling: &Sampling, max: usize) -> Option<Vec<u32>> {
        if !self.with_mtp || self.dead() {
            return None;
        }
        match self.model.mtp_propose(slot, tokens, max.min(8)) {
            Ok(Some(d)) => Some(d),
            Ok(None) => None,
            Err(e) => {
                self.record(&format!("MTP draft of slot {slot}"), &e, false);
                None
            }
        }
    }

    fn prefix_caps(&self) -> Option<PrefixCaps> {
        self.prefix
    }

    /// The prefix cache's cut grid (the engine rounds every planned cut down to a multiple of this): the KDA
    /// `window_chunked` 16-row chunk grid, so a restored prefix resumes on a chunk boundary.
    fn prefix_cut_align(&self) -> usize {
        CUT_ALIGN
    }

    fn prefix_path(&self, _total: usize) -> Option<u32> {
        self.prefix.as_ref().map(|_| PATH_ID)
    }

    fn prefix_free_snapshots(&self) -> usize {
        if self.prefix.is_some() {
            self.model.prefix_free_snapshots()
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
        match self.model.restore_prefix(slot, handle.snap, &handle.pages, len) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.record(&format!("prefix restore into slot {slot}"), &e, false);
                self.drop_slot(slot);
                Err(e.to_string())
            }
        }
    }

    fn prefix_release(&mut self, handle: PrefixHandle) {
        if self.prefix.is_some() {
            self.model.release_prefix(handle.snap);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine reads the family's cut grid through the `Backend` trait (`prefix_cut_align`), and the default
    /// `PrefixConfig` leaves `cut_align` off so `Engine::with_prefix_cache` fills it in from here.
    #[test]
    fn the_prefix_cut_align_is_the_kda_chunk_grid() {
        assert_eq!(CUT_ALIGN, 16, "kda::window_chunked's 16-row chunk grid");
        assert_eq!(CUT_ALIGN, super::super::alloc::PREFILL_ALIGN);
        // the Backend impl reports it (the method exists on the trait and is overridden here)
        fn grid_of<B: Backend>(b: &B) -> usize {
            b.prefix_cut_align()
        }
        let _ = grid_of::<BailingBackend>;
        // the engine's default leaves the grid unset, so the backend's grid applies
        assert!(gb_engine::PrefixConfig::default().cut_align <= 1);
    }

    #[test]
    fn a_cut_is_on_the_grid_of_its_segment() {
        // a cold prefill's segment starts at 0: its cuts are the multiples of 16
        assert!(on_cut_grid(1296, 0) && on_cut_grid(16, 0) && on_cut_grid(0, 0));
        assert!(!on_cut_grid(1300, 0) && !on_cut_grid(1, 0));
        // a resumed one starts at its hit (itself a cut, so on the grid): 16-row chunks from there
        assert!(on_cut_grid(2048 + 512, 2048) && !on_cut_grid(2048 + 8, 2048));
        // a segment that started off the grid keeps its own grid
        assert!(on_cut_grid(100 + 32, 100) && !on_cut_grid(128, 100));
        assert!(!on_cut_grid(10, 20));
    }
}
