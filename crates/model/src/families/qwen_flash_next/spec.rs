//! The speculation arbiter of a Flash-Next server that drafts with a DFlash2 draft (`--draft`): per round and per stream, the draft's
//! depth, the copy drafter's proposal, or no drafts, chosen by `gb_spec::arbiter::Arbiter` from each stream's measured acceptance by depth
//! and what the round would cost (`--spec auto`). The draft call costs about the same whatever its depth (the block runs its 8 rows), and the
//! verify window grows by about 4.5 ms a row, so text that is hard to draft gets fewer verify rows, and a stream the draft does not help is
//! not drafted. Drafts only decide how many rows a window carries, never which token is committed.
//!
//! The costs are this family's measured step table (`engine::STEP_MS`) and a draft call MEASURED on a GB10 (a steady round under nsys:
//! 4.4 ms for one sequence; `tests/flash_next_dflash2.rs`), each rescaled by what the server observes, as GLM's are.

use gb_sched::CostModel;
use gb_spec::arbiter::{Arbiter, ArbiterConfig, ArbiterCost, Ask, Choice};
use gb_spec::CopyDrafter;
use std::collections::BTreeMap;

/// The draft call for one sequence (ms), MEASURED: a steady DFlash2 round's draft at depth 7 (5 layers over the 8-row block, the target's
/// `lm_head` over the asked rows, the selector's walk) on a GB10.
pub const DRAFT_MS: f64 = 4.4;
/// What each further sequence of a batched draft call adds (ms) [estimate: the block's 8 rows a sequence through the same weights].
pub const DRAFT_PER_STREAM_MS: f64 = 0.6;
/// The deepest draft (the block's rows past the anchor).
pub const MAX_DEPTH: usize = crate::draft::DRAFTS;

const SCALE_DECAY: f64 = 0.9;
const SCALE_RANGE: (f64, f64) = (0.7, 1.6);

/// What a Flash-Next round costs: the verify window by its rows (the family's step table) and the draft call, each times a scale that
/// tracks what was observed.
pub struct FnCost {
    verify: CostModel,
    pub verify_scale: f64,
    pub draft_scale: f64,
}

impl FnCost {
    pub fn new(step_ms: &[(usize, f64)]) -> Self {
        Self { verify: CostModel::from_table(step_ms.to_vec()), verify_scale: 1.0, draft_scale: 1.0 }
    }
    fn blend(scale: f64, ratio: f64) -> f64 {
        if !ratio.is_finite() || ratio <= 0.0 {
            return scale;
        }
        (scale * SCALE_DECAY + ratio * (1.0 - SCALE_DECAY)).clamp(SCALE_RANGE.0, SCALE_RANGE.1)
    }
    fn draft_unscaled(streams: usize) -> f64 {
        DRAFT_MS + DRAFT_PER_STREAM_MS * streams.saturating_sub(1) as f64
    }
    /// A window of `rows` rows took `ms`.
    pub fn observe_window(&mut self, rows: usize, ms: f64) {
        self.verify_scale = Self::blend(self.verify_scale, ms / self.verify.ms(rows));
    }
    /// A draft call for `streams` sequences took `ms`.
    pub fn observe_draft(&mut self, streams: usize, ms: f64) {
        self.draft_scale = Self::blend(self.draft_scale, ms / Self::draft_unscaled(streams));
    }
}

impl ArbiterCost for FnCost {
    fn verify_ms(&self, rows: usize) -> f64 {
        self.verify.ms(rows) * self.verify_scale
    }
    /// The draft runs its whole block whatever the depth, and absorbs the kept rows of the last window on the way (a few rows: free).
    fn mtp_ms(&self, streams: usize, _absorb_rows: usize, depth: usize) -> f64 {
        if streams == 0 || depth == 0 {
            return 0.0;
        }
        Self::draft_unscaled(streams) * self.draft_scale
    }
}

/// The arbiter's settings for the DFlash2 draft. The acceptance prior is MEASURED (the conditional acceptance by depth of `gates.py bench`
/// on a GB10: about 0.8 at the first draft, flatter behind it than a head's chain, 0.65-0.9 by class).
pub fn config() -> ArbiterConfig {
    ArbiterConfig {
        allow_mtp: true,
        allow_copy: true,
        max_mtp_depth: MAX_DEPTH,
        warmup_depth: 3,
        warmup_rounds: 2,
        max_rows: super::memory::WINDOW_ROWS,
        mtp_prior: vec![0.82, 0.78, 0.76, 0.75, 0.75, 0.75, 0.75],
        mtp_chain: 0.95,
        ..ArbiterConfig::default()
    }
}

/// One stream's round in flight: what was offered, and how many of its drafts the engine verified.
#[derive(Clone, Copy, Debug)]
struct Round {
    choice: Choice,
    drafts: usize,
    verified: usize,
}

/// The arbiter, its costs, and each slot's copy drafter and round in flight.
pub struct DraftSpec {
    pub arbiter: Arbiter,
    pub cost: FnCost,
    copy: BTreeMap<usize, CopyDrafter>,
    rounds: BTreeMap<usize, Round>,
}

/// What the arbiter decided for one sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Pick {
    /// Ask the draft for this many drafts.
    Draft(usize),
    /// Offer the copy drafter's proposal.
    Copy(Vec<u32>),
    /// No drafts this round.
    Serial,
}

impl DraftSpec {
    pub fn new(step_ms: &[(usize, f64)]) -> Self {
        Self { arbiter: Arbiter::new(config()), cost: FnCost::new(step_ms), copy: BTreeMap::new(), rounds: BTreeMap::new() }
    }

    /// A new sequence in `slot`: its estimates start from the priors, its copy drafter from nothing.
    pub fn begin(&mut self, slot: usize) {
        self.arbiter.begin(slot);
        self.copy.remove(&slot);
        self.rounds.remove(&slot);
    }

    /// This round's pick for each sequence `(slot, tokens, max, the draft's context is ready)`.
    pub fn choose(&mut self, reqs: &[(usize, &[u32], usize, bool)]) -> Vec<Pick> {
        let copies: Vec<Vec<u32>> = reqs
            .iter()
            .map(|&(slot, tokens, max, _)| self.copy.entry(slot).or_insert_with(|| CopyDrafter::new(8, 7)).propose(tokens, max))
            .collect();
        let asks: Vec<Ask> = reqs
            .iter()
            .zip(&copies)
            .map(|(&(slot, _, max, ready), copy)| Ask {
                stream: slot,
                max: max.min(MAX_DEPTH),
                copy_len: copy.len().min(max),
                mtp_ok: ready,
                absorb_rows: 0,
                must_absorb: false,
            })
            .collect();
        let choices = self.arbiter.choose(&self.cost, &asks);
        choices
            .into_iter()
            .zip(copies)
            .map(|(c, copy)| match c {
                Choice::Mtp(d) if d > 0 => Pick::Draft(d),
                Choice::Copy(n) if n > 0 => Pick::Copy(copy[..n.min(copy.len())].to_vec()),
                _ => Pick::Serial,
            })
            .collect()
    }

    /// What `slot` was offered this round (after the draft ran: `drafts` is what it returned).
    pub fn offered(&mut self, slot: usize, pick: &Pick, drafts: usize) {
        let choice = match pick {
            Pick::Draft(_) if drafts > 0 => Choice::Mtp(drafts),
            Pick::Copy(c) if !c.is_empty() => Choice::Copy(c.len()),
            _ => Choice::None,
        };
        self.rounds.insert(slot, Round { choice, drafts, verified: 0 });
    }

    /// The engine put `rows` rows (the base row and the drafts it kept) in `slot`'s window.
    pub fn on_step(&mut self, slot: usize, rows: usize) {
        if let Some(r) = self.rounds.get_mut(&slot) {
            r.verified = rows.saturating_sub(1).min(r.drafts);
        }
    }

    /// `keep` rows of `slot`'s window were accepted: `keep - 1` drafts.
    pub fn on_commit(&mut self, slot: usize, keep: usize) {
        let Some(r) = self.rounds.remove(&slot) else { return };
        let choice = if r.verified == 0 { Choice::None } else { r.choice };
        self.arbiter.observe(slot, choice, r.verified, keep.saturating_sub(1).min(r.verified));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP: &[(usize, f64)] = &[(1, 29.0), (2, 34.1), (4, 43.6), (8, 59.2), (16, 85.0), (64, 220.0)];

    #[test]
    fn a_stream_whose_drafts_land_is_drafted_deep_and_one_whose_drafts_miss_is_not() {
        let toks: Vec<u32> = (0..50).collect();
        // a stream that accepts every draft: the arbiter goes deep
        let mut s = DraftSpec::new(STEP);
        s.begin(0);
        let mut last = 0;
        for _ in 0..30 {
            let p = s.choose(&[(0, &toks, 12, true)]).remove(0);
            let Pick::Draft(d) = p else { panic!("expected a draft, got {p:?}") };
            s.offered(0, &Pick::Draft(d), d);
            s.on_step(0, d + 1);
            s.on_commit(0, d + 1);
            last = d;
        }
        assert_eq!(last, MAX_DEPTH, "every draft accepted: the whole block");
        // a stream whose first draft always misses: the arbiter stops drafting it (probes aside)
        let mut s = DraftSpec::new(STEP);
        s.begin(1);
        let mut drafted = Vec::new();
        for _ in 0..40 {
            let p = s.choose(&[(1, &toks, 12, true)]).remove(0);
            let d = if let Pick::Draft(d) = p { d } else { 0 };
            s.offered(1, &p, d);
            s.on_step(1, d + 1);
            s.on_commit(1, 1);
            drafted.push(d);
        }
        assert!(drafted[30..].iter().filter(|&&d| d > 0).count() <= 2, "a useless draft is not kept on: {drafted:?}");
    }

    #[test]
    fn a_stream_without_a_ready_context_is_never_asked_for_drafts() {
        let mut s = DraftSpec::new(STEP);
        let toks: Vec<u32> = (0..10).collect();
        for p in s.choose(&[(0, &toks, 7, false), (1, &toks, 7, false)]) {
            assert!(!matches!(p, Pick::Draft(_)), "{p:?}");
        }
    }

    #[test]
    fn the_costs_follow_what_is_observed_within_bounds() {
        let mut c = FnCost::new(STEP);
        assert!((c.verify_ms(8) - 59.2).abs() < 1e-9 && (c.mtp_ms(1, 0, 7) - DRAFT_MS).abs() < 1e-9);
        assert_eq!(c.mtp_ms(2, 0, 1), c.mtp_ms(2, 0, 7), "the block costs the same whatever the depth");
        assert_eq!(c.mtp_ms(1, 0, 0), 0.0);
        for _ in 0..100 {
            c.observe_window(8, 2.0 * 59.2);
            c.observe_draft(1, 0.1);
        }
        assert!((c.verify_scale - SCALE_RANGE.1).abs() < 1e-9 && (c.draft_scale - SCALE_RANGE.0).abs() < 1e-9);
    }
}
